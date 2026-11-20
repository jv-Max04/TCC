"""Vetor de características e a cascata de classificação do diagnóstico.

O diagnóstico recebe a matriz crua, antes de qualquer correção, e a classifica em
`clean`, `step_change`, `gradual_drift`, `erratic` ou `unassessable` por uma cascata
de regras em que a primeira que casa decide. Uma regra anterior pode esconder o que
as seguintes fariam, então `classify` devolve também qual regra casou, e os testes
afirmam a regra além da classe.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from refino.diagnosis.batches import propose_batches
from refino.diagnosis.changepoint import detect_changepoints
from refino.diagnosis.policy import DiagnosisPolicy
from refino.diagnosis.trajectory import (
    jump_ratio,
    monotonicity_net_over_gross,
    rss_minimizing_split,
    sustained_step_ratio,
    total_movement,
)
from refino.domain.diagnosis import DefectProfile
from refino.domain.labels import DefectType, PreTransform, RsdVariant, validate_label
from refino.domain.matrix import RefinoMatrix
from refino.metrics.rsd import rsd
from refino.metrics.structure import StructuralMetrics, structural_metrics

__all__ = ["Classification", "classify", "diagnose", "extract_features"]

# Pré-transformação aplicada antes do PCA, para matriz de intensidade crua. Os limiares
# das regras 2 e 3 foram medidos sobre a trajetória que ela produz, e trocá-la
# invalidaria a calibração. Por isso não é parâmetro do chamador.
PRETRANSFORM: PreTransform = "log10p1"

# Variante de RSD do vetor: `outlier_removed` é a que a `RSD()` do `tools.R` implementa.
RSD_VARIANT: RsdVariant = "outlier_removed"


@dataclass(frozen=True)
class Classification:
    """Classe, regra que a produziu e confiança.

    `rule` permite ao teste distinguir a regra certa casando de outra regra que
    produzisse a mesma classe.
    """

    defect_type: DefectType
    rule: int
    confidence: float


def _margem(valor: float, limiar: float) -> float:
    """Distância relativa ao limiar, saturada em 1. Zero quando o valor é o limiar."""
    if not np.isfinite(valor) or limiar == 0.0:
        return 0.0
    return float(min(1.0, abs(valor - limiar) / abs(limiar)))


def _rsd_por_feature(matrix: RefinoMatrix) -> np.ndarray:
    """RSD dos QCs por feature, em fração."""
    qcs = np.asarray(matrix.values)[matrix.qc_mask]
    return rsd(qcs, axis=0, variant=RSD_VARIANT)


def _estatisticas_de_rsd(vetor: np.ndarray) -> dict[str, float]:
    """Estatísticas do RSD dos QCs, calculadas sobre as features de RSD definido.

    RSD pode ser indefinido, por exemplo quando todos os QCs da feature são zero.
    Essas features ficam fora, e `n_features_rsd_definido` vai junto para dizer sobre
    quantas o resumo foi calculado.

    Quando nenhuma feature tem RSD definido, as estatísticas saem `nan` e a contagem
    sai zero: não havia o que agregar.
    """
    definidas = vetor[np.isfinite(vetor)]
    n = int(definidas.size)
    if n == 0:
        vazio = float("nan")
        return {
            "rsd_mediana_fracao": vazio,
            "rsd_media_fracao": vazio,
            "rsd_p25_fracao": vazio,
            "rsd_p75_fracao": vazio,
            "rsd_p90_fracao": vazio,
            "frac_sob_20": vazio,
            "frac_sob_30": vazio,
            "assimetria_fisher": vazio,
            "n_features_rsd_definido": 0.0,
        }

    media = float(definidas.mean())
    desvio = float(definidas.std(ddof=0))
    # Assimetria de Fisher, sem correção de viés amostral. O estimador vai no nome da
    # chave porque as convenções diferem por um fator que depende de `n`.
    assimetria = 0.0 if desvio == 0.0 else float(((definidas - media) ** 3).mean() / desvio**3)
    return {
        "rsd_mediana_fracao": float(np.median(definidas)),
        "rsd_media_fracao": media,
        "rsd_p25_fracao": float(np.percentile(definidas, 25)),
        "rsd_p75_fracao": float(np.percentile(definidas, 75)),
        "rsd_p90_fracao": float(np.percentile(definidas, 90)),
        "frac_sob_20": float((definidas < 0.20).mean()),
        "frac_sob_30": float((definidas < 0.30).mean()),
        "assimetria_fisher": assimetria,
        "n_features_rsd_definido": float(n),
    }


def _esparsidade(matrix: RefinoMatrix) -> dict[str, float]:
    """Zeros por célula e por feature.

    Não há chave de fração de ausentes: o equipamento deixa a célula vazia quando não
    detecta, então o zero é o ausente e a chave duplicaria `fracao_zeros`.
    """
    valores = np.asarray(matrix.values)
    zeros_por_feature = (valores == 0.0).sum(axis=0)
    return {
        "fracao_zeros": float((valores == 0.0).mean()),
        "frac_features_sem_zero": float((zeros_por_feature == 0).mean()),
        "frac_features_mais_80_zeros": float((zeros_por_feature > 0.80 * matrix.n_samples).mean()),
    }


def _intensidade(matrix: RefinoMatrix) -> dict[str, float]:
    """Nível e espalhamento das intensidades, em `log10` e sobre células positivas.

    Zeros ficam fora: são ausência de detecção, e incluí-los faria a mediana depender
    da fração de não detectados em vez do nível de sinal.
    """
    valores = np.asarray(matrix.values)
    positivas = valores[valores > 0.0]
    if positivas.size == 0:
        vazio = float("nan")
        return {
            "log_intensidade_mediana": vazio,
            "log_intensidade_iqr": vazio,
            "razao_p99_p50": vazio,
        }
    log = np.log10(positivas)
    p50 = float(np.percentile(positivas, 50))
    return {
        "log_intensidade_mediana": float(np.median(log)),
        "log_intensidade_iqr": float(np.percentile(log, 75) - np.percentile(log, 25)),
        "razao_p99_p50": float(np.percentile(positivas, 99) / p50) if p50 > 0 else float("nan"),
    }


def _trajetoria(
    metricas: StructuralMetrics, matrix: RefinoMatrix
) -> tuple[dict[str, float], list[int]]:
    """As características de trajetória, mais os cortes detectados.

    Devolve os cortes junto para que o chamador use a mesma detecção que produziu as
    características, sem refazê-la.
    """
    trajetoria = np.asarray(metricas.qc_pc1_trajectory, dtype=np.float64)
    ordens_qc = matrix.injection_orders[matrix.qc_mask]
    cortes = detect_changepoints(trajetoria, ordens_qc)

    saltos = np.abs(np.diff(trajetoria)) if trajetoria.size >= 2 else np.array([])
    corte_indice = rss_minimizing_split(trajetoria)
    sustentacao = (
        sustained_step_ratio(trajetoria, split=corte_indice)
        if corte_indice is not None
        else float("nan")
    )
    # `rho` de Spearman entre posição do QC e PC1. Fica no vetor como insumo da política
    # aprendida e não entra em regra: posto sobre poucos pontos quase empatados é instável.
    rho = (
        float(abs(spearmanr(np.arange(trajetoria.size), trajetoria).statistic))
        if trajetoria.size >= 2
        else float("nan")
    )

    caracteristicas = {
        "salto_mediano": float(np.median(saltos)) if saltos.size else float("nan"),
        "salto_maximo": float(saltos.max()) if saltos.size else float("nan"),
        "razao_maior_salto_sobre_mediana_dos_demais": jump_ratio(trajetoria),
        "monotonicidade_liquido_bruto": monotonicity_net_over_gross(trajetoria),
        "movimento_total": total_movement(trajetoria),
        "rho_posicao_qc_pc1": rho,
        "amplitude_total": (
            float(trajetoria.max() - trajetoria.min()) if trajetoria.size else float("nan")
        ),
        "salto_sobre_amplitude": sustentacao,
    }
    return caracteristicas, cortes


def extract_features(matrix: RefinoMatrix) -> dict[str, float]:
    """Vetor de características da matriz, entrada da classificação e da política aprendida.

    Toda chave de RSD carrega a unidade no nome (`_fracao`): `rsd()` devolve fração, e
    confundi-la com percentual muda a classe. A função descreve e não decide: nenhum
    limiar entra aqui, e quem compara contra limiar é `classify`.
    """
    metricas = structural_metrics(matrix, pretransform=PRETRANSFORM)
    trajetoria, _ = _trajetoria(metricas, matrix)
    explicada = metricas.explained_variance_ratio

    return {
        "n_samples": float(matrix.n_samples),
        "n_features": float(matrix.n_features),
        "n_qcs": float(int(matrix.qc_mask.sum())),
        "razao_p_n": float(matrix.n_features / matrix.n_samples),
        **_esparsidade(matrix),
        **_estatisticas_de_rsd(_rsd_por_feature(matrix)),
        **trajetoria,
        "pc1_injection_correlation": metricas.pc1_injection_correlation_abs,
        "qc_dispersion_ratio": metricas.qc_dispersion_ratio,
        "variancia_explicada_pc1": explicada[0],
        "variancia_explicada_pc2": explicada[1],
        **_intensidade(matrix),
    }


def classify(matrix: RefinoMatrix, *, policy: DiagnosisPolicy) -> Classification:
    """A cascata das regras 0 a 4: a primeira que casa decide.

    Devolve a regra junto com a classe, para que o teste possa afirmar qual casou.

    A ordem importa:

    - Regra 0, `unassessable`: vem primeiro porque `nan < limiar` é `False`, e a matriz
      sem métrica definida escorregaria até `erratic` por acidente aritmético.
    - Regra 1, `clean`: RSD mediano de QC e correlação de PC1 com a ordem de injeção
      abaixo dos limiares.
    - Regra 2, `step_change`: vem antes de `gradual_drift` porque degrau também é
      monótono, e a estatística da regra 3 não distingue os dois.
    - Regra 3, `gradual_drift`.
    - Regra 4, `erratic`: o caso restante.

    A cascata é heurística deste trabalho. A separação entre deriva contínua e degrau
    discreto tem contraparte publicada (Čuklina et al. 2021; Westgard et al. 1977);
    os cortes, a ordem das regras e a classe de abstenção são daqui. Broadhurst et al.
    (2018) nomeia as formas de deriva sem dar procedimento de decisão.

    Não há mapa de `defect_type` para corretor. O tipo, a confiança e o número de
    lotes sugerido entram como características da política aprendida
    (`decision/features.py`), e o número de lotes é pré-preenchido na janela.
    """
    metricas = structural_metrics(matrix, pretransform=PRETRANSFORM)

    if not metricas.is_assessable(max_spread=policy.max_spread):
        return Classification(defect_type="unassessable", rule=0, confidence=0.0)

    estatisticas = _estatisticas_de_rsd(_rsd_por_feature(matrix))
    rsd_mediana = estatisticas["rsd_mediana_fracao"]
    correlacao = metricas.pc1_injection_correlation_abs

    # Regra 1. A confiança é a menor das duas margens: quem decide é a condição mais
    # apertada.
    if (
        np.isfinite(rsd_mediana)
        and rsd_mediana < policy.rsd_mediana_max_de(matrix.mode)
        and correlacao < policy.pc1_injection_correlation_max
    ):
        return Classification(
            defect_type="clean",
            rule=1,
            confidence=min(
                _margem(rsd_mediana, policy.rsd_mediana_max_de(matrix.mode)),
                _margem(correlacao, policy.pc1_injection_correlation_max),
            ),
        )

    trajetoria, cortes = _trajetoria(metricas, matrix)
    sustentacao = trajetoria["salto_sobre_amplitude"]

    if cortes and np.isfinite(sustentacao) and sustentacao > policy.step_sustain_min:
        return Classification(
            defect_type="step_change",
            rule=2,
            confidence=_margem(sustentacao, policy.step_sustain_min),
        )

    # Piso de movimento: trajetória sem movimento não é errática, é ausência de deriva.
    # Sem esta guarda a monotonicidade indefinida cairia no caso restante.
    if trajetoria["movimento_total"] < policy.trajectory_movement_min:
        return Classification(defect_type="unassessable", rule=3, confidence=0.0)

    monotonicidade = trajetoria["monotonicidade_liquido_bruto"]
    if np.isfinite(monotonicidade) and monotonicidade >= policy.qc_monotonicity_min:
        return Classification(
            defect_type="gradual_drift",
            rule=3,
            confidence=_margem(monotonicidade, policy.qc_monotonicity_min),
        )

    # Regra 4. A confiança mede quão longe dos limiares as regras 2 e 3 falharam: perto
    # do limiar a classificação é frágil.
    return Classification(
        defect_type="erratic",
        rule=4,
        confidence=min(
            _margem(sustentacao, policy.step_sustain_min),
            _margem(monotonicidade, policy.qc_monotonicity_min),
        ),
    )


def diagnose(matrix: RefinoMatrix, *, policy: DiagnosisPolicy) -> DefectProfile:
    """Diagnóstico completo: classe, lotes propostos e o vetor de características.

    `policy` é obrigatória e não tem padrão: os limiares são de quem tem os dados para
    calibrá-los. O `defect_type` é validado aqui em tempo de execução, além da
    validação de `DefectProfile`, porque `Literal` só é conferido na análise estática
    e este valor é gravado no banco e lido de volta.
    """
    classificacao = classify(matrix, policy=policy)
    validate_label("DefectType", classificacao.defect_type)

    metricas = structural_metrics(matrix, pretransform=PRETRANSFORM)
    _, cortes = _trajetoria(metricas, matrix)
    caracteristicas = extract_features(matrix)
    valores = np.asarray(matrix.values)

    return DefectProfile(
        defect_type=classificacao.defect_type,
        confidence=classificacao.confidence,
        suggested_batches=tuple(int(rotulo) for rotulo in propose_batches(matrix, cortes)),
        changepoints=tuple(int(c) for c in cortes),
        qc_rsd_median=caracteristicas["rsd_mediana_fracao"],
        qc_rsd_p75=caracteristicas["rsd_p75_fracao"],
        zero_fraction=caracteristicas["fracao_zeros"],
        # Célula não finita, que é diferente de zero: o não detectado é o zero e está
        # em `zero_fraction`; este campo conta o que a ingestão não pôde ler.
        missing_fraction=float((~np.isfinite(valores)).mean()),
        pc1_injection_correlation=metricas.pc1_injection_correlation_abs,
        qc_dispersion_ratio=metricas.qc_dispersion_ratio,
        n_features=matrix.n_features,
        n_samples=matrix.n_samples,
        n_qcs=int(matrix.qc_mask.sum()),
        raw_features=caracteristicas,
    )

"""Estabilidade de cada feature ao longo das células da grade.

Para cada feature, mede em que fração das especificações ela sai significativa
(`q < 0,05`), com que consistência de sinal e em que posto. O resultado é reportado
como distribuição, junto com a decomposição por eixo: a grade contém especificações
que destroem qualquer sinal, então um limiar único sobre a fração não faria sentido.

As agregações declaram em `ORDER_CLASS` se dependem da ordem das células. Fração,
contagem e mediana não dependem; média depende, porque a soma de ponto flutuante
muda com a ordem dos termos. `tests/unit/test_multiverse_stability.py` compara duas
ordens.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from refino.domain.matrix import RefinoMatrix
from refino.multiverse.executor import CellResult
from refino.statistics.univariate import univariate_test

__all__ = [
    "AggregationOrderClass",
    "FeatureStability",
    "MetricWinner",
    "StabilityThresholds",
    "Verdict",
    "axis_decomposition",
    "best_by_metric",
    "classifica_veredito",
    "feature_stability",
    "metrics_disagree",
]

Verdict = Literal["robust", "fragile", "absent"]

# Classe de invariância de cada agregação à ordem das células. Fração e contagem são
# invariantes; mediana e IQR também, porque ordenam internamente; média não é, porque
# a soma acumula.
AggregationOrderClass = Literal["order_invariant", "order_dependent"]

ORDER_CLASS: dict[str, AggregationOrderClass] = {
    "frac_significant": "order_invariant",
    "sign_consistency": "order_invariant",
    "median_qvalue": "order_invariant",
    "median_rank": "order_invariant",
    "rank_iqr": "order_invariant",
    # `axis_decomposition` usa média. Declarada como dependente para que o teste de
    # ordem a inclua.
    "axis_decomposition_mean": "order_dependent",
}


@dataclass(frozen=True)
class StabilityThresholds:
    """Os limiares do veredito, configuráveis.

    O veredito é leitura do perfil e não critério de aceitação: quem lê escolhe onde
    cortar.
    """

    robust_frac: float = 0.80
    robust_sign: float = 0.95
    absent_frac: float = 0.05


@dataclass(frozen=True)
class FeatureStability:
    """Estabilidade de uma feature, com o número de células em que ela foi testada.

    `n_cells` acompanha os números porque a mesma fração sobre 4 células e sobre 9.600
    é evidência de peso diferente.
    """

    feature: str
    frac_significant: float
    median_qvalue: float
    median_rank: float
    rank_iqr: float
    sign_consistency: float
    verdict: Verdict
    n_cells: int


def feature_stability(
    cells: tuple[CellResult, ...],
    groups: tuple[str, str],
    *,
    thresholds: StabilityThresholds | None = None,
    alpha: float = 0.05,
) -> tuple[FeatureStability, ...]:
    """Estabilidade por feature sobre as células, na ordem das features da primeira célula.

    A ordem da saída é a das features, e não a de chegada das células, para que duas
    execuções com paralelismo diferente devolvam tuplas iguais.

    Feature ausente de uma célula não entra no denominador dela. As especificações
    filtram conjuntos diferentes de features, e contar a filtrada como não
    significativa confundiria 'testada e não passou' com 'não testada'. Feature
    testada em nenhuma célula sai `absent`.
    """
    limiares = thresholds if thresholds is not None else StabilityThresholds()
    if not cells:
        return ()

    # `q`, sinal e posto por feature, célula a célula. Dicionário por nome e não matriz,
    # porque o conjunto de features varia entre células.
    qs: dict[str, list[float]] = {}
    sinais: dict[str, list[float]] = {}
    postos: dict[str, list[float]] = {}

    for celula in cells:
        # Sem `fold_change_matrix`: o sinal do fold change sai da matriz processada.
        # `sign_consistency` é persistida, e trocar a matriz mudaria os valores já
        # gravados. `tests/unit/test_multiverse_stability.py` fixa esses valores.
        resultado = univariate_test(celula.matrix, groups)
        q = np.asarray(resultado.qvalue_fdr, dtype=np.float64)
        # Posto dentro da célula: 1 é o mais significativo. O `q` bruto não é comparável
        # entre células, porque depende do número de features testadas.
        ordem = np.argsort(np.where(np.isfinite(q), q, np.inf), kind="stable")
        posto = np.empty(q.size, dtype=np.float64)
        posto[ordem] = np.arange(1, q.size + 1, dtype=np.float64)

        for indice, nome in enumerate(resultado.feature):
            if not np.isfinite(q[indice]):
                continue
            qs.setdefault(nome, []).append(float(q[indice]))
            postos.setdefault(nome, []).append(float(posto[indice]))
            fold = float(resultado.log2fc[indice])
            sinais.setdefault(nome, []).append(np.sign(fold) if np.isfinite(fold) else 0.0)

    saida: list[FeatureStability] = []
    for nome in cells[0].matrix.feature_names:
        valores = qs.get(nome, [])
        if not valores:
            saida.append(_ausente(nome))
            continue
        arranjo = np.asarray(valores)
        frac = float((arranjo < alpha).mean())
        sinal_arranjo = np.asarray(sinais[nome])
        nao_nulos = sinal_arranjo[sinal_arranjo != 0.0]
        consistencia = (
            float(max((nao_nulos > 0).mean(), (nao_nulos < 0).mean())) if nao_nulos.size else 0.0
        )
        posto_arranjo = np.asarray(postos[nome])
        saida.append(
            FeatureStability(
                feature=nome,
                frac_significant=frac,
                median_qvalue=float(np.median(arranjo)),
                median_rank=float(np.median(posto_arranjo)),
                rank_iqr=float(np.percentile(posto_arranjo, 75) - np.percentile(posto_arranjo, 25)),
                sign_consistency=consistencia,
                verdict=classifica_veredito(frac, consistencia, limiares),
                n_cells=arranjo.size,
            )
        )
    return tuple(saida)


def _ausente(nome: str) -> FeatureStability:
    """Feature testada em nenhuma célula: veredito `absent` e `n_cells` zero.

    Os agregados de `q`, posto e sinal saem `nan`, que diz 'não medido'; zero diria
    'medido e deu zero'.
    """
    return FeatureStability(
        feature=nome,
        frac_significant=0.0,
        median_qvalue=float("nan"),
        median_rank=float("nan"),
        rank_iqr=float("nan"),
        sign_consistency=float("nan"),
        verdict="absent",
        n_cells=0,
    )


def classifica_veredito(frac: float, sign: float, limiares: StabilityThresholds) -> Verdict:
    """O veredito a partir da fração de células significativas e da consistência de sinal.

    `absent` abaixo de `absent_frac`; `robust` com fração e sinal acima dos limiares;
    `fragile` no resto. O resto inclui fração alta com sinal inconsistente: o efeito
    aparece e troca de direção.

    A API recalcula o veredito a partir dos dois números persistidos, em vez de gravar
    o rótulo. Assim, mudar os limiares reclassifica também as execuções antigas.
    """
    if frac < limiares.absent_frac:
        return "absent"
    if frac >= limiares.robust_frac and sign >= limiares.robust_sign:
        return "robust"
    return "fragile"


def axis_decomposition(
    cells: tuple[CellResult, ...], groups: tuple[str, str], *, alpha: float = 0.05
) -> dict[str, dict[str, float]]:
    """Fração média de features significativas por opção de cada eixo.

    O perfil de `frac_significant` diz quanto a grade dissolve o sinal; a decomposição
    diz quais escolhas o preservam e quais o destroem.

    A média é sobre células e é a única agregação daqui que depende da ordem; está em
    `ORDER_CLASS` como tal.
    """
    eixos = (
        "correction",
        "rsd_cutoff",
        "missing_threshold",
        "imputation",
        "sample_normalization",
        "transformation",
        "scaling",
    )
    acumulado: dict[str, dict[str, list[float]]] = {eixo: {} for eixo in eixos}

    for celula in cells:
        # Features sem `q` definido ficam fora do denominador. O valor é persistido, e
        # incluir essas features mudaria números já gravados; a contagem delas está em
        # `resultado.counts.qvalue_undefined`.
        resultado = univariate_test(celula.matrix, groups)
        q = np.asarray(resultado.qvalue_fdr, dtype=np.float64)
        definidos = np.isfinite(q)
        fracao = float((q[definidos] < alpha).mean()) if definidos.any() else 0.0
        for eixo in eixos:
            chave = str(getattr(celula.spec, eixo))
            acumulado[eixo].setdefault(chave, []).append(fracao)

    return {
        eixo: {chave: float(np.mean(vals)) for chave, vals in sorted(opcoes.items())}
        for eixo, opcoes in acumulado.items()
    }


def _qc_tighter_frac(matrix: RefinoMatrix) -> float:
    """Fração de features em que o IQR dos QCs é menor que o das amostras.

    Mede repetibilidade em qualquer ponto da grade. Razões do tipo `sd/média` perdem o
    sentido quando a média muda de sinal, o que acontece sob `log10` seguido de `auto`.
    A comparação de IQR dentro da mesma feature não muda com o escalonamento, que
    aplica o mesmo fator a QC e a amostra.

    Feature sem IQR finito, ou com IQR de amostra nulo, não entra no denominador.
    """
    valores = np.asarray(matrix.values, dtype=np.float64)
    qc = valores[matrix.qc_mask]
    amostra = valores[matrix.sample_mask]
    if qc.shape[0] < 2 or amostra.shape[0] < 2 or valores.shape[1] == 0:
        return float("nan")
    iqr_qc = np.subtract(*np.nanpercentile(qc, [75, 25], axis=0))
    iqr_am = np.subtract(*np.nanpercentile(amostra, [75, 25], axis=0))
    avaliaveis = np.isfinite(iqr_qc) & np.isfinite(iqr_am) & (iqr_am > 0)
    if not avaliaveis.any():
        return float("nan")
    return float((iqr_qc[avaliaveis] < iqr_am[avaliaveis]).mean())


@dataclass(frozen=True)
class MetricWinner:
    """A especificação vencedora de uma métrica, com a evidência de que houve disputa.

    Só o `fingerprint` do vencedor não distingue 'esta especificação venceu' de 'todas
    empataram e esta é a primeira'. Por isso o tipo carrega quantas células foram
    pontuadas, quantos valores distintos houve e quais empataram no ótimo. Métrica que
    não distingue célula nenhuma (`discriminates` falso) não conta como voto.
    """

    metric: str
    fingerprint: str
    value: float
    n_cells_scored: int
    n_distinct_values: int
    tied_at_best: tuple[str, ...]

    @property
    def n_tied_at_best(self) -> int:
        return len(self.tied_at_best)

    @property
    def discriminates(self) -> bool:
        """A métrica separou ao menos duas células? Se não, o vencedor é ordem, não mérito."""
        return self.n_distinct_values >= 2

    @property
    def unique_winner(self) -> bool:
        """Houve um vencedor único, ou o ótimo é empate entre várias especificações?"""
        return self.n_tied_at_best == 1


def metrics_disagree(winners: dict[str, MetricWinner]) -> bool:
    """As métricas discordam? Sim quando a interseção dos conjuntos ótimos é vazia.

    Comparar os `fingerprint` não basta: ele é um representante do conjunto ótimo,
    escolhido por ordem quando há empate. Duas métricas com representantes diferentes
    podem ter conjuntos ótimos que se cruzam, e então há uma especificação que serve
    às duas.

    Métrica que não discrimina não vota: o conjunto ótimo dela é a grade inteira, e
    qualquer interseção ficaria não vazia.
    """
    votantes = [w.tied_at_best for w in winners.values() if w.discriminates]
    if len(votantes) < 2:
        return False
    comum: set[str] = set(votantes[0])
    for conjunto in votantes[1:]:
        comum &= set(conjunto)
    return not comum


def best_by_metric(
    cells: tuple[CellResult, ...], groups: tuple[str, str], *, alpha: float = 0.05
) -> dict[str, MetricWinner]:
    """A melhor especificação por métrica, como `nome da métrica -> MetricWinner`.

    O nome de cada métrica carrega o ponto de medição. Quatro são medidas no fim do
    pipeline e uma logo depois da correção:

    - `n_significant_fdr_post_pipeline`: mais features com `q < alpha`.
    - `qc_dispersion_ratio_post_pipeline`: menor dispersão de QC sobre a das amostras.
    - `pc1_injection_correlation_post_pipeline`: menor correlação de PC1 com a ordem
      de injeção.
    - `qc_tighter_than_sample_frac_post_pipeline`: mais features com QC mais apertado
      que as amostras.
    - `qc_rsd_median_post_correction`: menor RSD mediano de QC.

    As métricas são de famílias diferentes de propósito (significância, estrutura,
    deriva, repetibilidade), para que a discordância seja entre réguas e não entre
    variantes da mesma.

    O RSD só é medido depois da correção, em escala de intensidade: ele é `sd/média`,
    e depois de `log10` com `auto` a média por feature pode ser negativa, o que
    inverte o sentido de minimizar. No fim do pipeline a repetibilidade é medida por
    `_qc_tighter_frac`. A matriz pós-correção depende só do prefixo
    `(correction, n_batches)`, então a métrica desse ponto ordena corretores, e não
    especificações; o `n_tied_at_best` dela mostra isso.

    A chave do vencedor é o `fingerprint`, comparável entre execuções e serializável.
    """
    from refino.metrics.rsd import rsd
    from refino.metrics.structure import structural_metrics

    if not cells:
        return {}

    # `nome -> [(pontuação, valor bruto, fingerprint)]`, com todas as células: contar
    # valores distintos e empatados no ótimo exige a população inteira.
    pontuacoes: dict[str, list[tuple[float, float, str]]] = {}

    def _registra(nome: str, valor: float, chave: str, *, maior_e_melhor: bool) -> None:
        if not np.isfinite(valor):
            return
        pontuacoes.setdefault(nome, []).append((-valor if maior_e_melhor else valor, valor, chave))

    for celula in cells:
        chave = celula.spec.fingerprint()
        # Ponto 1: fim do pipeline. `pretransform="none"`, porque a matriz já passou pela
        # transformação da própria especificação.
        #
        # Features sem `q` definido ficam fora da contagem, como em `axis_decomposition`.
        resultado = univariate_test(celula.matrix, groups)
        q = np.asarray(resultado.qvalue_fdr, dtype=np.float64)
        _registra(
            "n_significant_fdr_post_pipeline",
            float((q[np.isfinite(q)] < alpha).sum()),
            chave,
            maior_e_melhor=True,
        )
        _registra(
            "qc_tighter_than_sample_frac_post_pipeline",
            _qc_tighter_frac(celula.matrix),
            chave,
            maior_e_melhor=True,
        )
        try:
            pos_pipeline = structural_metrics(celula.matrix, pretransform="none", n_components=2)
        except ValueError:
            pos_pipeline = None
        if pos_pipeline is not None:
            _registra(
                "qc_dispersion_ratio_post_pipeline",
                pos_pipeline.qc_dispersion_ratio,
                chave,
                maior_e_melhor=False,
            )
            _registra(
                "pc1_injection_correlation_post_pipeline",
                pos_pipeline.pc1_injection_correlation_abs,
                chave,
                maior_e_melhor=False,
            )

        # Ponto 2: pós-correção, em escala de intensidade, onde a média é positiva e
        # `sd/média` mede repetibilidade.
        corrigida = celula.corrected
        qc = np.asarray(corrigida.values, dtype=np.float64)[corrigida.qc_mask]
        if qc.shape[0] >= 2:
            _registra(
                "qc_rsd_median_post_correction",
                float(np.nanmedian(rsd(qc, axis=0, variant="outlier_removed"))),
                chave,
                maior_e_melhor=False,
            )

    vencedores: dict[str, MetricWinner] = {}
    for nome, marcas in pontuacoes.items():
        melhor = min(marcas, key=lambda m: m[0])
        vencedores[nome] = MetricWinner(
            metric=nome,
            fingerprint=melhor[2],
            value=melhor[1],
            n_cells_scored=len(marcas),
            n_distinct_values=len({m[0] for m in marcas}),
            tied_at_best=tuple(sorted(m[2] for m in marcas if m[0] == melhor[0])),
        )
    return vencedores

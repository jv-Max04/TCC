"""Os dois gatilhos de `unassessable`, que os quatro modos reais NÃO alcançam.

**Existe porque o diagnóstico exige os seis desfechos de `diagnose`
com teste, e os quatro arquivos do laboratório cobrem quatro.** `clean`, `step_change`,
`gradual_drift` e `erratic` saem do dado real; `unassessable` não sai de nenhum deles,
porque as métricas estruturais são bem-comportadas nos quatro. Um ramo que só o dado real
alcançaria ficaria construído e nunca exercitado.

**E é o ramo que mais importa exercitar**, porque é o único que existe para impedir um
desfecho por **acidente aritmético**: sem a regra 0, `nan < 0,30` é `False` por IEEE 754,
a matriz escorrega pelas regras 1 a 3 e cai em `erratic` — e `erratic` sugere tentar
método robusto, enquanto não avaliável sugere olhar o dado antes de corrigi-lo. Duas
ações opostas a partir do mesmo silêncio.

Os dois gatilhos são **quantidades diferentes** e por isso testes diferentes:

| Gatilho | Regra | Condição |
|---|---|---|
| métrica estrutural indefinida ou instável | 0 | `is_assessable(max_spread)` é `False` |
| trajetória sem movimento | 3 | `Σ|Δv|` abaixo do piso da política |
"""

import numpy as np
import pytest

from refino.diagnosis.policy import DiagnosisPolicy
from refino.diagnosis.profile import classify
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.metrics.structure import structural_metrics

SEMENTE = 7
N_QC = 3


def _matriz(valores: np.ndarray) -> RefinoMatrix:
    """Matriz sintética com os primeiros `N_QC` como QC."""
    n = valores.shape[0]
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i < N_QC else "sample",
            group="QC" if i < N_QC else "G",
        )
        for i in range(n)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(valores.shape[1])),
        samples=amostras,
        mode="sintetico",
    )


def _amostras_identicas() -> RefinoMatrix:
    """Amostras biológicas IDÊNTICAS entre si: a dispersão delas é zero.

    É a configuração degenerada — a razão de dispersão salta
    entre `nan` e 5,3e+29 em torno do denominador nulo. Aqui ela vai a ~9,7e+16
    com espalhamento ~2,7e+15, o que é exactamente "métrica que não sustenta
    classificação".
    """
    gerador = np.random.default_rng(SEMENTE)
    valores = gerador.normal(1000.0, 100.0, size=(10, 30))
    valores[N_QC:] = valores[N_QC]  # todas as amostras iguais
    valores[:N_QC] = gerador.normal(1000.0, 100.0, size=(N_QC, 30))
    return _matriz(valores)


def _qcs_identicos() -> RefinoMatrix:
    """QCs IDÊNTICOS: a trajetória de PC1 é constante e o movimento é zero."""
    gerador = np.random.default_rng(SEMENTE)
    valores = gerador.normal(1000.0, 100.0, size=(10, 30))
    valores[:N_QC] = valores[0]
    return _matriz(valores)


def _politica(**trocas: float) -> DiagnosisPolicy:
    base = {
        "max_spread": 0.10,
        "rsd_mediana_max": 0.20,
        "pc1_injection_correlation_max": 0.30,
        "step_sustain_min": 2.0,
        "qc_monotonicity_min": 0.487,
        "trajectory_movement_min": 1e-9,
    }
    return DiagnosisPolicy(**{**base, **trocas})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Gatilho 1: a regra 0
# ---------------------------------------------------------------------------


def test_metrica_instavel_sai_unassessable_pela_regra_0() -> None:
    """Amostras idênticas tornam a razão de dispersão instável, e a regra 0 pega."""
    matriz = _amostras_identicas()
    metricas = structural_metrics(matriz, pretransform="log10p1")
    assert not metricas.is_assessable(max_spread=0.10), (
        "o fixture não é degenerado; sem isso o teste abaixo passaria por outro caminho"
    )

    resultado = classify(matriz, policy=_politica())
    assert resultado.defect_type == "unassessable"
    assert resultado.rule == 0
    assert resultado.confidence == 0.0, "sem limiar consultado não há margem a reportar"


def test_a_regra_0_decide_antes_de_qualquer_limiar() -> None:
    """**A ordem é o conteúdo, e este teste é o que a afirma.**

    Com uma política cujos outros limiares classificariam a matriz de outro jeito, o
    desfecho continua `unassessable` — o que prova que a regra 0 intercepta em vez de
    coincidir. Sem esta asserção, "saiu `unassessable`" seria compatível com a regra 0
    nunca ter sido consultada.
    """
    matriz = _amostras_identicas()

    # Limiares que fariam a regra 1 casar com folga se ela fosse consultada.
    frouxa = _politica(rsd_mediana_max=1.0, pc1_injection_correlation_max=1.0)
    resultado = classify(matriz, policy=frouxa)
    assert resultado.rule == 0, (
        f"a regra {resultado.rule} casou antes da 0; a cascata está fora de ordem e a "
        "classificação por acidente aritmético que a regra 0 impede voltou"
    )
    assert resultado.defect_type == "unassessable"


def test_max_spread_frouxo_deixa_a_cascata_seguir() -> None:
    """**O par causal da regra 0**: com o limiar frouxo ela NÃO casa.

    É o que distingue guarda ativa de guarda que sempre dispara. Sem este teste, uma
    regra 0 que devolvesse `False` incondicionalmente passaria pelos dois testes acima e
    tornaria toda matriz não avaliável.
    """
    matriz = _amostras_identicas()
    resultado = classify(matriz, policy=_politica(max_spread=1e30))
    assert resultado.rule != 0, "com `max_spread` enorme a regra 0 não pode casar"
    assert resultado.defect_type != "unassessable" or resultado.rule == 3


# ---------------------------------------------------------------------------
# Gatilho 2: o piso de movimento, na regra 3
# ---------------------------------------------------------------------------


def test_trajetoria_sem_movimento_sai_unassessable_e_nao_erratic() -> None:
    """**A distinção que o piso existe para fazer**, e ela leva a ações opostas.

    QCs idênticos dão trajetória constante. Sem o piso, a monotonicidade é `nan`, a
    regra 3 não casa, e a matriz cai no caso restante — `erratic`, que sugere método
    robusto. Mas ausência de movimento não é erraticidade: é ausência de deriva, e a
    ação certa é olhar o dado.

    `max_spread` entra enorme para que a regra 0 não intercepte, e `rsd_mediana_max`
    minúsculo para que a regra 1 não case: o alvo aqui é a regra 3, e chegar até ela é
    parte do que se afirma.
    """
    matriz = _qcs_identicos()
    metricas = structural_metrics(matriz, pretransform="log10p1")
    trajetoria = np.asarray(metricas.qc_pc1_trajectory)
    movimento = float(np.abs(np.diff(trajetoria)).sum())
    assert movimento < 1e-9, f"o fixture não tem trajetória constante: movimento={movimento:.2e}"

    resultado = classify(
        matriz,
        policy=_politica(max_spread=1e30, rsd_mediana_max=1e-9, trajectory_movement_min=1e-9),
    )
    assert resultado.defect_type == "unassessable", (
        f"saiu {resultado.defect_type!r} pela regra {resultado.rule}. Se saiu `erratic`, o "
        "piso de movimento não está sendo consultado e o acidente aritmético voltou"
    )
    assert resultado.rule == 3
    assert resultado.confidence == 0.0


def test_piso_de_movimento_abaixo_do_real_deixa_a_cascata_seguir() -> None:
    """**O par causal do piso**: com o piso menor que o movimento, a regra 3 decide.

    Sem esta direção, um piso que disparasse sempre passaria pelo teste acima e faria
    toda matriz sair `unassessable`.
    """
    gerador = np.random.default_rng(SEMENTE)
    valores = gerador.normal(1000.0, 100.0, size=(10, 30))
    matriz = _matriz(valores)  # QCs distintos: há movimento

    metricas = structural_metrics(matriz, pretransform="log10p1")
    movimento = float(np.abs(np.diff(np.asarray(metricas.qc_pc1_trajectory))).sum())
    assert movimento > 1e-6, "o fixture tem de ter movimento para esta direção valer"

    resultado = classify(
        matriz,
        policy=_politica(max_spread=1e30, rsd_mediana_max=1e-9, trajectory_movement_min=1e-12),
    )
    assert resultado.defect_type != "unassessable", (
        "com piso abaixo do movimento real a regra 3 tem de consultar a monotonicidade"
    )
    assert resultado.rule in (3, 4)


@pytest.mark.parametrize(
    "desfecho", ["clean", "step_change", "gradual_drift", "erratic", "unassessable"]
)
def test_os_cinco_valores_de_defect_type_sao_alcancaveis(desfecho: str) -> None:
    """Exaustividade: nenhum valor do vocabulário fica sem caminho que o produza.

    Um valor de `DefectType` que nenhuma entrada alcança é vocabulário morto — e a
    camada de decisão aprende sobre o vetor de características assumindo que as classes existem.
    Os quatro primeiros saem dos arquivos reais, em `tests/integration/`; o quinto sai
    deste módulo, e esta parametrização é o que amarra os dois lugares num só inventário.
    """
    from typing import get_args

    from refino.domain.labels import DefectType

    assert desfecho in get_args(DefectType)

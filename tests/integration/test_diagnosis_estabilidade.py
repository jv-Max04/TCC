"""A classificação não muda sob ruído de 1%, e a MEDIÇÃO veio antes da asserção.

**O critério da especificação não era mensurável como escrito.** "Perturbar a matriz com 1%
de ruído gaussiano não deve alterar a classificação" não diz de que sigma = 1% é fração
- do valor, da mediana global, do desvio da feature, e são três testes diferentes -, não
diz quantas réplicas, e não declara semente, que a regra 3 do README exige.

**As três definições foram medidas antes de qualquer número ser fixado aqui**, com os
desfechos escritos primeiro: estável nas três, estável em algumas, ou instável em todas
- e o terceiro seria ACHADO, não falha da unidade, porque significaria que a regra que
decide aquele modo não sustenta perturbação. Fixar a asserção antes de medir faria o
teste defender o valor errado.

Medido, com N = 10 réplicas por par (modo, definição):

| Definição | sigma | Perturbação relativa medida | Desfecho |
|---|---|---|---|
| A | `x * (1 + e)`, `e ~ N(0, 0.01)` | **0,0080** | estável nos quatro |
| B | `x + e`, `e ~ N(0, 0.01 * mediana)` | 0,0003 | estável nos quatro |
| C | `x + e_j`, `e_j ~ N(0, 0.01 * sd_j)` | 0,0042 | estável nos quatro |

**Desfecho A do pré-registro**, e o teste fixa a definição de **maior** perturbação
medida - a multiplicativa -, porque testar a mais fraca seria escolher o ensaio mais
fácil depois de ver o resultado.

**Aplica-se só a células positivas, e a escolha é declarada.** Zero neste domínio
significa **não detectado**, não "medida igual a zero". Perturbar
zero criaria detecção falsa e mediria outra pergunta - robustez a mudança de esparsidade,
não a ruído de medição.
"""

import numpy as np
import pytest

from refino.diagnosis.policy import DiagnosisPolicy
from refino.diagnosis.profile import classify
from refino.domain.matrix import RefinoMatrix
from refino.io.readers import load_matrix
from tests.conftest import measurements_path, sample_list_path

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MODOS = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")
SEMENTE = 42
SIGMA_RELATIVO = 0.01

# N = 5 aqui contra N = 10 na medição, e a redução é decisão de CUSTO com a medição
# atrás: as doze combinações deram 10/10 iguais, então cinco réplicas exercitam o
# caminho sem repetir uma varredura já feita. O número está aqui e não escondido.
N_REPLICAS = 5

POLITICA = DiagnosisPolicy(
    max_spread=0.10,
    rsd_mediana_max=0.20,
    pc1_injection_correlation_max=0.30,
    step_sustain_min=2.0,
    qc_monotonicity_min=0.487,
    trajectory_movement_min=1e-9,
)


def _perturba(matriz: RefinoMatrix, *, sigma: float, semente: list[int]) -> RefinoMatrix:
    """Ruído multiplicativo nas células POSITIVAS, com semente derivada.

    Fluxo derivado de `(semente, réplica)` e nunca semente global: o número de sorteios
    gastos por uma réplica deslocaria as seguintes, e o determinismo passaria a depender
    da ordem de execução.
    """
    gerador = np.random.default_rng(semente)
    valores = np.asarray(matriz.values, dtype=np.float64).copy()
    positivas = valores > 0.0
    valores[positivas] *= 1.0 + gerador.normal(0.0, sigma, size=int(positivas.sum()))
    return matriz.with_values(np.clip(valores, 0.0, None), f"estabilidade:sigma={sigma}")


@pytest.fixture(scope="module")
def matrizes() -> dict[str, RefinoMatrix]:
    return {
        modo: load_matrix(measurements_path(modo), sample_list_path(modo), modo) for modo in MODOS
    }


def test_o_instrumento_responde_a_perturbacao_grande(matrizes: dict) -> None:
    """**A âncora exigida, e ela vem ANTES de o resultado ser lido.**

    "Estável sob 1%" é indistinguível de "o arnês não perturba nada" sem esta
    verificação. Uma perturbação grande - sigma de 3,0, trezentas vezes a do critério -
    **tem** de mudar alguma classe; se não mudar, o teste de estabilidade passa por
    vacuidade e não mede nada.

    Medido: sigma de 3,0 vira o `neg_hilic` de `step_change` para `gradual_drift` e o
    `neg_c18` de `clean` para `erratic`. Os dois erráticos continuam erráticos, o que é
    esperado - eles já são o caso restante, e não há classe "mais errática" para onde ir.
    """
    viradas = []
    for modo in MODOS:
        base = classify(matrizes[modo], policy=POLITICA)
        grande = _perturba(matrizes[modo], sigma=3.0, semente=[SEMENTE, 999])

        assert not np.array_equal(np.asarray(grande.values), np.asarray(matrizes[modo].values)), (
            f"{modo}: a perturbação não alterou a matriz; o arnês não está perturbando nada"
        )

        if classify(grande, policy=POLITICA).defect_type != base.defect_type:
            viradas.append(modo)

    assert viradas, (
        "nenhuma classe mudou sob perturbação de sigma=3,0. O instrumento não responde, "
        "então 'estável sob 1%' não seria evidência de estabilidade - seria evidência de "
        "que a classificação ignora a matriz."
    )


@pytest.mark.parametrize("modo", MODOS)
def test_a_classificacao_e_estavel_sob_um_por_cento(modo: str, matrizes: dict) -> None:
    """O critério obrigatório da especificação, com a definição de sigma e a semente declaradas.

    Afirma a classe **e a regra**: se a classe se mantivesse por regra diferente, a
    cascata teria mudado sob ruído e o desfecho estaria estável por coincidência.

    E **reporta a distância ao limiar** em caso de reprovação, porque passar perto é
    indistinguível de passar com folga no agregado - a mensagem tem de dizer qual dos
    dois aconteceu.
    """
    base = classify(matrizes[modo], policy=POLITICA)

    for replica in range(N_REPLICAS):
        perturbada = _perturba(matrizes[modo], sigma=SIGMA_RELATIVO, semente=[SEMENTE, replica])
        resultado = classify(perturbada, policy=POLITICA)
        assert (resultado.defect_type, resultado.rule) == (base.defect_type, base.rule), (
            f"{modo}, réplica {replica}: {base.defect_type}/regra {base.rule} "
            f"(confiança {base.confidence:.3f}) virou {resultado.defect_type}/regra "
            f"{resultado.rule} sob sigma={SIGMA_RELATIVO}. Se a confiança da base é baixa, "
            "isto é achado sobre a regra que decide este modo - ela não sustenta "
            "perturbação - e não defeito de código."
        )


def test_a_perturbacao_preserva_o_padrao_de_zeros(matrizes: dict) -> None:
    """Célula não detectada continua não detectada, e é o que a escolha declara.

    Sem esta asserção, uma implementação que perturbasse zeros passaria pelo teste de
    estabilidade e estaria medindo outra pergunta: robustez a mudança de esparsidade em
    vez de robustez a ruído de medição. As duas são legítimas e não são a mesma.
    """
    matriz = matrizes["pos_hilic"]
    perturbada = _perturba(matriz, sigma=SIGMA_RELATIVO, semente=[SEMENTE, 0])

    zeros_antes = np.asarray(matriz.values) == 0.0
    zeros_depois = np.asarray(perturbada.values) == 0.0
    assert np.array_equal(zeros_antes, zeros_depois), (
        "a perturbação mudou o padrão de zeros; célula não detectada não pode virar "
        "detectada, nem o contrário"
    )


def test_a_perturbacao_e_determinista(matrizes: dict) -> None:
    """Regra 3 do README - a mesma semente dá a mesma réplica."""
    primeira = _perturba(matrizes["neg_c18"], sigma=SIGMA_RELATIVO, semente=[SEMENTE, 3])
    segunda = _perturba(matrizes["neg_c18"], sigma=SIGMA_RELATIVO, semente=[SEMENTE, 3])
    assert np.array_equal(np.asarray(primeira.values), np.asarray(segunda.values))

    outra = _perturba(matrizes["neg_c18"], sigma=SIGMA_RELATIVO, semente=[SEMENTE, 4])
    assert not np.array_equal(np.asarray(primeira.values), np.asarray(outra.values)), (
        "réplicas diferentes deram a mesma matriz; o fluxo de semente não está derivado"
    )

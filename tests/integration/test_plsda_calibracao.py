"""`[CA-6.3a]` e `[CA-6.3b]` — os dois limiares, cada um derivado da sua própria nula.

**Os limiares não são escritos, são derivados**. O critério original exigia `95 de 100` das
duas condições juntas, e isso era insatisfazível: sob a nula o p-valor de permutação é ~U(0,1),
logo `P(p > 0,05) = 0,95` **exatamente**, e exigir ≥ 95 de uma quantidade cuja média é 95 dá
`P = 0,616` — reprovação intermitente numa implementação perfeita.

| Critério | Nula | Quantil inferior a 99% | Medido em 01/08 |
|---|---|---|---|
| `[CA-6.3a]` — `p > 0,05` | `Binomial(100; 0,95)`, conhecida | **>= 89** | 95 |
| `[CA-6.3b]` — `q2 <= 0` | `Binomial(100; 0,86)`, **medida** | **>= 77** | 86 |

A nula do `p` é conhecida por construção; a do `q2` **não era**, e foi medida antes de o limiar
existir. `q2` sob a nula tem mediana -0,2910 e cauda positiva — sai positivo em 14 de 100 sorteios,
com 29 amostras e 7 dobras.

**A cobertura de 99% é declarada e justificada:** este teste roda no noturno, e reprovação falsa
ali custa uma investigação inteira e consome a credibilidade das reprovações seguintes.
99,9% custaria 2 e 3 pontos de limiar, e cada ponto é sensibilidade perdida para detectar
descalibração real.
"""

import numpy as np
import pytest
from scipy.stats import binom

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.spec import PipelineSpec
from refino.io.readers import load_matrix
from refino.preprocessing.pipeline import process_matrix
from refino.statistics.plsda import plsda
from tests.conftest import measurements_path, sample_list_path

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MODO = "pos_hilic"
PAR = ("ATLL", "NoHTLV")
N_EXECUCOES = 100
N_PERMUTACOES = 200
SEMENTE_BASE = 20260801

# As duas nulas. A primeira e conhecida por construcao; a segunda foi MEDIDA e entra aqui
# como observacao sobre esta configuracao — 29 amostras, 7 dobras, 2 componentes.
NULA_P = 0.95
NULA_Q2 = 0.86
COBERTURA = 0.99


def _limiar(nula: float) -> int:
    """Maior `k` tal que `P(X >= k) >= COBERTURA`, com `X ~ Binomial(N_EXECUCOES, nula)`.

    Derivado e não escrito: mudar `COBERTURA` ou a nula muda o limiar sem que ninguém digite um
    número. É o que separa "o critério afirma o que é verdade" de "o critério soa rigoroso".
    """
    k = N_EXECUCOES
    while k > 0 and binom.sf(k - 1, N_EXECUCOES, nula) < COBERTURA:
        k -= 1
    return k


@pytest.fixture(scope="module")
def varredura() -> tuple[np.ndarray, np.ndarray]:
    """As duas quantidades sobre `N_EXECUCOES` sorteios de rótulos aleatórios.

    Uma fixture só para os dois critérios: a varredura custa ~70 s e medi-la duas vezes seria
    pagar duas vezes pelo mesmo número.
    """
    crua = load_matrix(measurements_path(MODO), sample_list_path(MODO), MODO)
    processada, _ = process_matrix(
        crua,
        PipelineSpec(
            correction="none",
            n_batches=1,
            rsd_cutoff=0.30,
            ranking_filter="none",
            ranking_keep_fraction=None,
            ranking_filter_2="none",
            ranking_keep_fraction_2=None,
            rsd_variant="outlier_removed",
            missing_threshold=0.8,
            imputation="one_fifth_min",
            sample_normalization="sum",
            normalization_reference=None,
            normalization_factors=None,
            transformation="log10",
            scaling="auto",
            seed=42,
        ),
    )
    rotulos = np.array([a.group for a in processada.samples])
    mascara = np.isin(rotulos, PAR)

    q2s, ps = [], []
    for execucao in range(N_EXECUCOES):
        gerador = np.random.default_rng([SEMENTE_BASE, execucao])
        embaralhados = rotulos.copy()
        # Reatribui os rotulos do par entre as MESMAS amostras: preserva as contagens por grupo.
        # Sem preservar, o desbalanceamento variaria entre execucoes e a nula mediria duas coisas.
        embaralhados[mascara] = gerador.permutation(rotulos[mascara])
        amostras = tuple(
            SampleMetadata(
                name=a.name,
                injection_order=a.injection_order,
                sample_type=a.sample_type,
                group=str(g),
                batch=a.batch,
            )
            for a, g in zip(processada.samples, embaralhados, strict=True)
        )
        sorteada = RefinoMatrix(
            values=np.asarray(processada.values),
            mode=processada.mode,
            feature_names=processada.feature_names,
            samples=amostras,
        )
        resultado = plsda(sorteada, PAR, n_permutations=N_PERMUTACOES, seed=execucao)
        q2s.append(resultado.q2)
        ps.append(resultado.permutation_pvalue)
    return np.array(q2s), np.array(ps)


def test_ca_6_3a_o_p_valor_de_permutacao_e_calibrado(
    varredura: tuple[np.ndarray, np.ndarray],
) -> None:
    """`[CA-6.3a]`: `p > 0,05` em ao menos `_limiar(0,95)` de 100."""
    _, ps = varredura
    acertos = int((ps > 0.05).sum())
    piso = _limiar(NULA_P)

    assert acertos >= piso, (
        f"{acertos} de {N_EXECUCOES} execuções com `p > 0,05`, abaixo do piso derivado {piso} "
        f"(quantil inferior de Binomial({N_EXECUCOES}; {NULA_P}) a {COBERTURA:.0%}). Sob a nula "
        "o p-valor de permutação é ~U(0,1) por construção, então esta fração é 0,95 EXATA: "
        "desvio aqui é descalibração do teste de permutação, não limiar apertado. NÃO suba o "
        "piso — a nula não é negociável"
    )


def test_ca_6_3b_o_q2_e_negativo_sob_a_nula(varredura: tuple[np.ndarray, np.ndarray]) -> None:
    """`[CA-6.3b]`: `q2 <= 0` em ao menos `_limiar(0,86)` de 100."""
    q2s, _ = varredura
    acertos = int((q2s <= 0.0).sum())
    piso = _limiar(NULA_Q2)

    assert acertos >= piso, (
        f"{acertos} de {N_EXECUCOES} execuções com `q2 <= 0`, abaixo do piso derivado {piso}. A "
        f"nula de {NULA_Q2} foi MEDIDA nesta configuração (DV19) e não é propriedade do método: "
        "com outra contagem de componentes, de dobras ou de amostras ela muda, e o piso muda com "
        "ela. Desvio aqui é achado sobre a configuração"
    )


def test_as_duas_fracoes_nao_sao_a_mesma_quantidade(
    varredura: tuple[np.ndarray, np.ndarray],
) -> None:
    """**O par que justifica os dois critérios em vez de um.**

    Se as duas frações coincidissem, o limiar único do critério original seria defensável e a
    divisão seria cerimônia. Elas não coincidem: a do `q2` é materialmente menor, e é por isso que
    um piso comum tornava um dos dois lados insatisfazível.
    """
    q2s, ps = varredura
    frac_p = float((ps > 0.05).mean())
    frac_q2 = float((q2s <= 0.0).mean())

    assert frac_p - frac_q2 > 0.03, (
        f"as duas frações convergiram — p: {frac_p:.3f}, q2: {frac_q2:.3f}. Se isso for estável, "
        "a nula do `q2` mudou e os dois pisos precisam ser remedidos; a divisão em `[CA-6.3a]` e "
        "`[CA-6.3b]` foi feita porque elas diferiam em 9 pontos"
    )


def test_a_cauda_positiva_do_q2_e_o_mecanismo(varredura: tuple[np.ndarray, np.ndarray]) -> None:
    """Por que a nula do `q2` não é 0,95: ele sai positivo por acaso.

    Sem esta asserção, "a fração é 0,86" seria número sem mecanismo. Com ela, o mecanismo está
    afirmado: a distribuição tem massa acima de zero, e a fração é o complemento dela.
    """
    q2s, _ = varredura
    assert q2s.max() > 0.0, (
        "nenhuma das execuções deu `q2` positivo sob a nula; então a fração de 0,86 não vem de "
        "cauda positiva e a explicação de DV19 perdeu o mecanismo"
    )
    assert float(np.median(q2s)) < 0.0, (
        "a mediana do `q2` sob a nula não é negativa; o modelo está aprendendo de rótulo "
        "aleatório, e isso é defeito e não calibração"
    )

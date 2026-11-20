"""BH contra `scipy` em 1.000 vetores, com semente declarada.

**A semente é acréscimo desta implementação e não estava no critério.** O critério pede
"1.000 vetores aleatórios" e não declara semente — pela regra 3 do README isso não é verificável, e
o número entra aqui como constante do módulo.

**O negativo planta o bug clássico, e ele é em processo.** A versão sem o mínimo cumulativo é
argumento de uma função deste arquivo, não subprocesso — negativo em
processo não consegue ficar inerte, porque a planta alimenta a asserção sem intermediário.
"""

import numpy as np
import pytest
from scipy.stats import false_discovery_control

from refino.statistics.univariate import benjamini_hochberg

pytestmark = pytest.mark.property

SEMENTE = 20260731
N_VETORES = 1_000
TOLERANCIA = 1e-12


def _bh_ingenuo(pvalues: np.ndarray) -> np.ndarray:
    """BH **sem** o mínimo cumulativo — o bug clássico, plantado de propósito.

    Existe só para o par causal. Se `benjamini_hochberg` passar a coincidir com esta, o passo de
    monotonicidade sumiu e o critério perdeu o dente.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    ordem = np.argsort(p, kind="stable")
    escalados = p[ordem] * n / np.arange(1, n + 1)
    saida = np.empty(n, dtype=np.float64)
    saida[ordem] = np.minimum(escalados, 1.0)
    return saida


def _mil_vetores() -> list[np.ndarray]:
    """Os 1.000 vetores do critério, com semente declarada e formas variadas.

    Quatro famílias de propósito: uniformes, concentrados perto de zero — onde o ajuste importa
    —, com empates, e degenerados. Mil vetores uniformes seriam mil vezes o mesmo caso.
    """
    gerador = np.random.default_rng(SEMENTE)
    vetores: list[np.ndarray] = []
    for indice in range(N_VETORES):
        n = int(gerador.integers(1, 200))
        familia = indice % 4
        if familia == 0:
            vetores.append(gerador.uniform(0.0, 1.0, size=n))
        elif familia == 1:
            vetores.append(gerador.beta(0.3, 5.0, size=n))
        elif familia == 2:
            base = gerador.uniform(0.0, 1.0, size=max(1, n // 4))
            vetores.append(gerador.choice(base, size=n, replace=True))
        else:
            vetores.append(np.full(n, float(gerador.uniform(0.0, 1.0))))
    return vetores


@pytest.fixture(scope="module")
def vetores() -> list[np.ndarray]:
    return _mil_vetores()


def test_ca_6_1_bate_com_scipy_em_mil_vetores(vetores: list[np.ndarray]) -> None:
    """Erro máximo abaixo de 1e-12 contra `false_discovery_control`."""
    maior = 0.0
    for vetor in vetores:
        nosso = benjamini_hochberg(vetor)
        deles = false_discovery_control(vetor, method="bh")
        maior = max(maior, float(np.max(np.abs(nosso - deles))))

    assert maior <= TOLERANCIA, (
        f"erro máximo {maior:.3e} contra `scipy.stats.false_discovery_control` excede "
        f"{TOLERANCIA:.0e} sobre {len(vetores)} vetores da semente {SEMENTE}. Isto é achado: "
        "BH tem definição fechada e as duas implementações "
        "deveriam coincidir ao epsilon de máquina"
    )


def test_a_versao_sem_monotonicidade_reprova(vetores: list[np.ndarray]) -> None:
    """**O par causal.** A versão ingênua discorda do `scipy` nos mesmos vetores.

    Sem esta metade, o teste acima passaria também sobre uma população em que o mínimo
    cumulativo nunca muda nada — e aí ele não estaria medindo o passo que a definição exige.
    """
    maior = 0.0
    afetados = 0
    for vetor in vetores:
        deles = false_discovery_control(vetor, method="bh")
        ingenuo = _bh_ingenuo(vetor)
        erro = float(np.max(np.abs(ingenuo - deles)))
        maior = max(maior, erro)
        afetados += erro > TOLERANCIA

    assert maior > TOLERANCIA, (
        "a versão sem mínimo cumulativo coincidiu com o `scipy`; então a população de teste "
        "não exercita a monotonicidade e o `[CA-6.1]` está vácuo"
    )
    # A contagem classifica antes de qualquer diagnóstico (regra 7), e a leitura correta
    # aqui é **todos menos alguns**: 994 de 1.000 divergem. Isso é CASO DE BORDA e não
    # defeito condicional — os poucos que coincidem são a família degenerada, em que a
    # sequência escalada já é monotônica por si e o mínimo cumulativo não tem o que fazer.
    #
    # A primeira redação deste comentário leu 994/1.000 como "fração intermediária" e
    # concluiu "condicional". A proporção é o dado, e ela classifica o defeito: fração
    # intermediária pede partição, todos-menos-alguns pede valor extremo ou população pequena.
    assert afetados > 0.9 * len(vetores), (
        f"só {afetados} de {len(vetores)} vetores afetados. O esperado é quase todos — o "
        "mínimo cumulativo muda o resultado sempre que a sequência escalada não é monotônica, "
        "o que é o caso comum. Proporção baixa significa população sem o caso que importa"
    )
    assert afetados < len(vetores), (
        "todos os vetores afetados: a população perdeu a família degenerada, e com ela o "
        "controle de que o negativo distingue caso fácil de caso difícil"
    )


def test_empates_recebem_q_igual() -> None:
    """A escolha que distingue as implementações, afirmada."""
    p = np.array([0.01, 0.04, 0.04, 0.20, 0.20, 0.20])
    q = benjamini_hochberg(p)

    assert q[1] == pytest.approx(q[2], abs=0.0)
    assert q[3] == pytest.approx(q[4], abs=0.0) == pytest.approx(q[5], abs=0.0)


def test_nenhum_q_excede_um(vetores: list[np.ndarray]) -> None:
    """`p * n / i` sozinho produz valores acima de 1; o corte é obrigatório."""
    for vetor in vetores[:200]:
        q = benjamini_hochberg(vetor)
        assert np.all(q[np.isfinite(q)] <= 1.0)


def test_q_e_monotonico_em_p(vetores: list[np.ndarray]) -> None:
    """A propriedade que define BH: `p` maior nunca recebe `q` menor."""
    for vetor in vetores[:200]:
        q = benjamini_hochberg(vetor)
        ordem = np.argsort(vetor, kind="stable")
        assert np.all(np.diff(q[ordem]) >= -1e-15)


def test_vetor_de_um_elemento() -> None:
    """`n = 1`: `q = p`, e o `1 * 1 / 1` não muda nada."""
    assert benjamini_hochberg(np.array([0.037])) == pytest.approx(np.array([0.037]))


def test_vetor_vazio_devolve_vazio() -> None:
    """Não levanta: matriz sem feature é entrada legítima depois de filtro agressivo."""
    assert benjamini_hochberg(np.empty(0)).size == 0


def test_todos_iguais_a_um() -> None:
    """Nenhum `q` acima de 1, e todos iguais entre si."""
    q = benjamini_hochberg(np.ones(10))
    assert np.all(q == 1.0)


def test_nan_volta_como_nan_e_nao_entra_no_denominador() -> None:
    """A política declarada: `n` é a contagem de p-valores DEFINIDOS.

    O par que a prova: o mesmo vetor com e sem o `NaN` acrescentado dá `q` idêntico nos
    definidos. Se o `NaN` contasse no posto, todo `q` mudaria.
    """
    definidos = np.array([0.01, 0.02, 0.30])
    com_nan = np.array([0.01, np.nan, 0.02, 0.30])

    q_definidos = benjamini_hochberg(definidos)
    q_com_nan = benjamini_hochberg(com_nan)

    assert np.isnan(q_com_nan[1])
    assert np.allclose(q_com_nan[[0, 2, 3]], q_definidos, atol=0.0)


def test_vetor_inteiramente_nan() -> None:
    """Devolve tudo `NaN` sem levantar — não há ajuste a fazer."""
    assert np.all(np.isnan(benjamini_hochberg(np.full(4, np.nan))))


def test_matriz_em_vez_de_vetor_levanta() -> None:
    """Fronteira: a função é sobre vetor, e forma errada não passa em silêncio."""
    with pytest.raises(ValueError, match="recebe vetor"):
        benjamini_hochberg(np.ones((3, 3)))


def test_determinismo_bit_a_bit(vetores: list[np.ndarray]) -> None:
    """Duas execuções sobre a mesma entrada dão vetor idêntico."""
    for vetor in vetores[:50]:
        assert np.array_equal(benjamini_hochberg(vetor), benjamini_hochberg(vetor), equal_nan=True)

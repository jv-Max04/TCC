"""A orientação canônica, e o par causal que a torna verificável.

**O par causal é a metade que importa, e é o enunciado do critério.** Um teste que só afirmasse
"`X` e `-X` dão o mesmo sinal" passaria também se a convenção não existisse e o `numpy` por
acidente devolvesse sinais coincidentes naquele fixture. O critério exige as duas direções: com
a convenção, os sinais coincidem; **sem** ela, divergem. A segunda é o que prova que o fixture
exercita a convenção em vez de documentar uma coincidência.
"""

import numpy as np
import pytest

from refino.statistics.orientation import orient_canonically

pytestmark = pytest.mark.property


def _svd_bruto(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """SVD sem orientação — o que a biblioteca devolve, que é o estado de partida."""
    centrada = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centrada, full_matrices=False)
    return u[:, :k] * s[:k], vt[:k].T


def test_a_matriz_negada_produz_o_mesmo_sinal() -> None:
    """A direção positiva do critério: `X` e `-X` dão loadings de sinal igual."""
    gerador = np.random.default_rng(7)
    x = gerador.normal(0.0, 1.0, size=(20, 6))

    _, carga = orient_canonically(*_svd_bruto(x, 3))
    _, carga_negada = orient_canonically(*_svd_bruto(-x, 3))

    assert np.allclose(np.sign(carga), np.sign(carga_negada)), (
        "loadings de `X` e `-X` divergiram em sinal com a convenção aplicada; a Fase 8 "
        "compara direções entre células e conta com esta propriedade"
    )


def test_sem_a_convencao_os_sinais_divergem() -> None:
    """**O par causal.** Sem orientar, `X` e `-X` dão loadings de sinal oposto.

    Se este teste passasse a falhar, o teste acima teria se tornado vácuo: significaria que os
    sinais já coincidem sem a convenção, e ele deixaria de distinguir convenção ativa de
    coincidência da biblioteca.
    """
    gerador = np.random.default_rng(7)
    x = gerador.normal(0.0, 1.0, size=(20, 6))

    _, carga = _svd_bruto(x, 3)
    _, carga_negada = _svd_bruto(-x, 3)

    assert not np.allclose(np.sign(carga), np.sign(carga_negada)), (
        "sem a convenção os sinais coincidiram; então o teste positivo não está medindo a "
        "convenção, e o `[CA-6.7]` ficou vácuo"
    )


def test_scores_e_loadings_viram_juntos() -> None:
    """A reconstrução `scores @ loadings.T` não muda com a orientação.

    Virar só um dos dois mudaria a matriz reconstruída, e nenhum teste que olhasse um lado
    isoladamente veria. É o defeito silencioso que esta asserção fecha.
    """
    gerador = np.random.default_rng(11)
    x = gerador.normal(0.0, 1.0, size=(15, 5))
    escores, cargas = _svd_bruto(x, 4)

    orientados, cargas_orientadas = orient_canonically(escores, cargas)

    assert np.allclose(escores @ cargas.T, orientados @ cargas_orientadas.T, atol=1e-12)


def test_o_maior_modulo_fica_positivo() -> None:
    """A convenção afirmada diretamente, componente a componente."""
    cargas = np.array([[0.2, -0.9], [-0.8, 0.1], [0.3, 0.4]])
    escores = np.ones((4, 2))

    _, orientadas = orient_canonically(escores, cargas)

    for k in range(orientadas.shape[1]):
        maior = int(np.argmax(np.abs(orientadas[:, k])))
        assert orientadas[maior, k] > 0.0, f"componente {k}: o maior módulo ficou negativo"


def test_empate_de_modulo_resolve_pelo_menor_indice() -> None:
    """A regra de desempate declarada, e ela precisa de teste porque é escolha.

    Empate exato existe em fixture pequeno, e sem regra declarada a convenção dependeria do
    padrão do `argmax` — invariante por acidente em vez de contrato.
    """
    # Módulos iguais nas posições 0 e 1, com sinais opostos: a posição 0 decide.
    cargas = np.array([[-0.5], [0.5], [0.1]])
    _, orientadas = orient_canonically(np.ones((3, 1)), cargas)

    assert orientadas[0, 0] > 0.0, "o empate deveria ser resolvido pelo menor índice"
    assert orientadas[1, 0] < 0.0


def test_componente_degenerado_nao_levanta() -> None:
    """Loading todo nulo, e loading não finito: sinal inalterado, sem exceção.

    Matriz com coluna constante depois de filtro agressivo é entrada legítima, e a orientação
    não é o lugar de decidir sobre componente degenerado.
    """
    nulo = np.zeros((3, 1))
    escores, cargas = orient_canonically(np.ones((2, 1)), nulo)
    assert np.array_equal(cargas, nulo)
    assert np.array_equal(escores, np.ones((2, 1)))

    nao_finito = np.array([[np.nan], [np.inf]])
    _, cargas_nf = orient_canonically(np.ones((2, 1)), nao_finito)
    assert np.isnan(cargas_nf[0, 0])


def test_a_orientacao_e_idempotente() -> None:
    """Aplicar duas vezes é o mesmo que aplicar uma.

    Convenção não idempotente dependeria de quantas vezes foi aplicada, e o caminho que a
    aplica duas vezes existe assim que `plsda` reusar o helper depois de `pca`.
    """
    gerador = np.random.default_rng(13)
    escores, cargas = _svd_bruto(gerador.normal(0.0, 1.0, size=(12, 4)), 3)

    uma = orient_canonically(escores, cargas)
    duas = orient_canonically(*uma)

    assert np.allclose(uma[0], duas[0], atol=0.0)
    assert np.allclose(uma[1], duas[1], atol=0.0)


def test_contagem_de_componentes_incompativel_levanta() -> None:
    """A orientação é por componente e exige o mesmo número dos dois lados."""
    with pytest.raises(ValueError, match="orientação é por componente"):
        orient_canonically(np.ones((5, 2)), np.ones((7, 3)))

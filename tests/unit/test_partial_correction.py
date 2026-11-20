"""Execução parcial de correção, e a reverificação.

**O que a modificação declara preservar, e como isso é reafirmado.** O invariante
era "conjunto de features da saída idêntico ao da **entrada**" e passou a ser
"idêntico ao conjunto **PEDIDO**", com execução completa como o caso em que pedido
= entrada. Isso obriga duas coisas, e nenhuma delas é ler o diff: o comportamento
preservado é reafirmado por teste, e o comportamento novo recebe as duas direções.

O preservado é reafirmado pelos 695 testes unitários que já existiam e continuam
verdes — `select_features` devolve a **própria** matriz quando o pedido é o
conjunto completo, então execução completa não passa por caminho novo. Este módulo
cobre o novo.

**O teste que importa é o par causal de exatidão.** Execução parcial tem de ser
**bit-idêntica** às colunas correspondentes da execução completa. Se não for, a
amostragem de features do critério mede outra quantidade, e o critério passaria
a comparar uma coisa afirmando outra.

O que torna isso verdadeiro está enumerado e foi verificado por leitura do código,
não suposto: o passo 1 é por feature; o passo 3 usa o pool **completo**, e é por
isso que a matriz não é fatiada; o passo 10 é por feature — `for index in
range(shape[1])`, blocos de QC e de amostra separados; o passo 11 é por célula. E o
fluxo de semente carrega o índice **original** da feature, então a mesma feature
recebe o mesmo sorteio nas duas execuções. Trocar o índice original pela posição no
subconjunto é o defeito que este par causal existe para pegar.
"""

import numpy as np
import pytest

from refino.correction.base import resolve_target_features, select_features
from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.correction.serrf import SerrfCorrector
from refino.domain.correction import CorrectionResult
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata

N_QC = 7


def _matrix(n_samples: int = 24, n_features: int = 12, seed: int = 5) -> RefinoMatrix:
    rng = np.random.default_rng(seed)
    qc_at = tuple(round(i) for i in np.linspace(0, n_samples - 1, N_QC))
    drift = 1.0 + 0.004 * np.arange(n_samples)
    base = rng.uniform(1e3, 1e5, size=n_features)
    values = (
        base[None, :] * drift[:, None] * (1.0 + 0.06 * rng.standard_normal((n_samples, n_features)))
    )
    samples = tuple(
        SampleMetadata(
            name=f"s{i:03d}",
            injection_order=i + 1,
            sample_type="qc" if i in qc_at else "sample",
            group="qc" if i in qc_at else "case",
            batch="A",
        )
        for i in range(n_samples)
    )
    return RefinoMatrix(
        values=values,
        feature_names=tuple(f"f{j}" for j in range(n_features)),
        samples=samples,
        mode="synthetic",
    )


CORRETORES = [
    pytest.param(NoneCorrector(), id="none"),
    pytest.param(LoessCorrector(), id="loess"),
    pytest.param(SerrfCorrector(), id="serrf"),
]


# ---------------------------------------------------------------------------
# O par causal de exatidão
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corretor", CORRETORES)
def test_parcial_e_bit_identica_as_colunas_da_completa(corretor: object) -> None:
    """**O par causal, e ele vale para os três corretores.**

    `array_equal` e não `allclose`: se a amostragem mudasse o número "pouco", o
    O critério compararia uma quantidade afirmando outra, e "pouco" é exatamente o
    que não se negocia aqui.

    O pedido é **alternado** e não um prefixo de propósito: prefixo passaria mesmo se
    a implementação usasse a posição no subconjunto como índice de semente, porque
    para o prefixo as duas coincidem nas primeiras posições. Alternar as separa.
    """
    matrix = _matrix()
    pedido = frozenset(matrix.feature_names[::2])

    completa = corretor.fit_transform(matrix, seed=42)  # type: ignore[attr-defined]
    parcial = corretor.fit_transform(matrix, seed=42, target_features=pedido)  # type: ignore[attr-defined]

    posicao = {name: i for i, name in enumerate(completa.matrix.feature_names)}
    esperado = completa.matrix.values[:, [posicao[n] for n in parcial.matrix.feature_names]]

    assert np.array_equal(parcial.matrix.values, esperado), (
        "execução parcial divergiu das colunas correspondentes da completa: o índice "
        "do fluxo de semente ou o pool de preditores está sendo derivado do "
        "subconjunto em vez do original"
    )


def test_o_pedido_alternado_e_o_que_pega_indice_do_subconjunto() -> None:
    """A pré-condição do par causal, isolada: as duas indexações de fato divergem.

    Sem isto, "alternado" seria escolha estética. Com o pedido `f0, f2, f4, …`, a
    posição no subconjunto é `0, 1, 2, …` e o índice original é `0, 2, 4, …`: só
    coincidem em zero, então qualquer uso da posição errada aparece.
    """
    matrix = _matrix()
    requested = resolve_target_features(matrix, frozenset(matrix.feature_names[::2]))
    original = [matrix.feature_names.index(name) for name in requested]
    assert original == list(range(0, matrix.n_features, 2))
    assert original != list(range(len(requested)))


# ---------------------------------------------------------------------------
# O invariante novo, nas duas direções
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("corretor", CORRETORES)
def test_o_conjunto_da_saida_e_o_pedido(corretor: object) -> None:
    """O invariante, afirmado sobre os três corretores."""
    matrix = _matrix()
    pedido = frozenset({"f1", "f3", "f7"})
    resultado = corretor.fit_transform(matrix, seed=42, target_features=pedido)  # type: ignore[attr-defined]

    assert set(resultado.matrix.feature_names) == pedido
    assert set(resultado.requested or ()) == pedido
    assert resultado.matrix.n_features == 3
    assert resultado.matrix.n_samples == matrix.n_samples, "execução parcial mexeu em amostras"


@pytest.mark.parametrize("corretor", CORRETORES)
def test_execucao_completa_declara_o_pedido_igual_a_entrada(corretor: object) -> None:
    """`requested = None` normaliza para a entrada — um significado, não dois."""
    resultado = corretor.fit_transform(_matrix(), seed=42)  # type: ignore[attr-defined]
    assert resultado.requested == resultado.matrix.feature_names


def test_o_invariante_reprova_com_pedido_que_nao_bate() -> None:
    """**A direção de reprovação, plantada como argumento do construtor.**

    Em processo pelo critério: a planta alimenta a asserção direto, sem
    intermediário que possa não recebê-la.
    """
    matrix = _matrix(n_features=4)
    with pytest.raises(ValueError, match="idêntico ao PEDIDO"):
        CorrectionResult(matrix=matrix, requested=("f0", "f1"))

    with pytest.raises(ValueError, match="idêntico ao PEDIDO"):
        CorrectionResult(matrix=matrix, requested=(*matrix.feature_names, "inexistente"))


def test_o_invariante_aceita_o_par_correspondente() -> None:
    """O controle: sem a planta, o mesmo caminho não acusa."""
    matrix = _matrix(n_features=4)
    assert CorrectionResult(matrix=matrix, requested=matrix.feature_names).requested == (
        matrix.feature_names
    )


def test_causa_nao_pode_nomear_feature_fora_do_pedido() -> None:
    """ "Não pedida" não é causa de não correção — é ausência de pedido.

    Pô-la no mapa de causas seria um erro, pois `unassessable` existe porque
    "não avaliável" não é tipo de defeito. O construtor já recusava causa apontando
    para fora da matriz, e com o pedido igual à saída isso passa a cobrir o pedido.
    """
    matrix = _matrix(n_features=4)
    recorte = select_features(matrix, ("f0", "f1"))
    with pytest.raises(ValueError, match="não existe na matriz"):
        CorrectionResult(
            matrix=recorte,
            uncorrected={"fit_failed": frozenset({"f2"})},
            requested=("f0", "f1"),
        )


# ---------------------------------------------------------------------------
# O resolvedor do pedido
# ---------------------------------------------------------------------------


def test_a_ordem_do_pedido_e_a_da_matriz_e_nao_a_do_conjunto() -> None:
    """**Determinismo pela porta mais silenciosa possível (regra 3 do README).**

    `frozenset` não tem ordem. Derivar a ordem das colunas de saída de uma iteração
    de conjunto tornaria a saída dependente do hash do interpretador — e o teste de
    determinismo passaria dentro de um processo e reprovaria entre processos.
    """
    matrix = _matrix(n_features=6)
    for pedido in ({"f4", "f0", "f2"}, {"f2", "f4", "f0"}, {"f0", "f4", "f2"}):
        assert resolve_target_features(matrix, frozenset(pedido)) == ("f0", "f2", "f4")


def test_pedido_desconhecido_ou_vazio_levanta() -> None:
    """Pedido para fora do espaço não é auditável; pedido vazio não é execução."""
    matrix = _matrix(n_features=4)
    with pytest.raises(ValueError, match="não existe na matriz"):
        resolve_target_features(matrix, frozenset({"f0", "nao_existe"}))
    with pytest.raises(ValueError, match="vazio"):
        resolve_target_features(matrix, frozenset())


def test_select_features_devolve_a_propria_matriz_no_pedido_completo() -> None:
    """É o que mantém execução completa fora do caminho novo, e a reverificação depende disso.

    Identidade e não igualdade: se fosse cópia, execução completa passaria a alocar
    e a comparar por outro caminho, e os 695 testes que reafirmam o preservado
    deixariam de reafirmá-lo.
    """
    matrix = _matrix()
    assert select_features(matrix, tuple(matrix.feature_names)) is matrix
    assert select_features(matrix, ("f0", "f1")) is not matrix

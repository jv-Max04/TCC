"""As duas propriedades obrigatórias.

**A primeira é o ARGUMENTO da ordem canônica, e não só um teste**. A ordem é
`normalização por amostra → transformação → escalonamento` e não dizia por quê. A versão
anterior deste teste pedia "associatividade em relação à ordem fixa", que não é afirmação
verificável: associatividade é sobre **agrupamento**, e com a ordem fixa não há agrupamento a
variar. A propriedade real é que as três **não comutam** — e é ela que justifica haver uma
ordem em vez de descrevê-la.

A segunda é a soma: normalização por soma faz toda linha somar 1, no caso não degenerado.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.preprocessing.normalization import (
    SOMA_CONSTANTE,
    normalize_samples,
    scale_features,
    transform,
)

pytestmark = pytest.mark.property


def _matriz(valores: np.ndarray) -> RefinoMatrix:
    n = valores.shape[0]
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i < 2 else "sample",
            group="QC" if i < 2 else "G",
        )
        for i in range(n)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(valores.shape[1])),
        samples=amostras,
        mode="sintetico",
    )


# ---------------------------------------------------------------------------
# Propriedade 1: as três NÃO comutam
# ---------------------------------------------------------------------------


def test_normalizacao_por_amostra_e_escalonamento_nao_comutam() -> None:
    """**O argumento da ordem canônica**, no par em que ele é mais claro.

    Normalizar por soma opera nas **linhas**; escalonar opera nas **colunas**. Fazer a linha
    somar 1 e depois centrar a coluna não é o mesmo que centrar a coluna e depois fazer a
    linha somar 1 — no segundo caso a soma da linha já não é 1, porque a centragem a
    destruiu.

    Se este teste passasse a reprovar, a ordem canônica teria perdido o motivo de existir, e
    isso seria achado e não conveniência.
    """
    gerador = np.random.default_rng(7)
    matriz = _matriz(gerador.normal(100.0, 20.0, size=(6, 4)))

    primeiro = scale_features(normalize_samples(matriz, "sum"), "auto")
    segundo = normalize_samples(scale_features(matriz, "auto"), "sum")

    assert not np.allclose(np.asarray(primeiro.values), np.asarray(segundo.values)), (
        "normalizar-depois-escalonar deu o mesmo que escalonar-depois-normalizar; a ordem "
        "canônica perdeu o argumento"
    )


def test_transformacao_e_escalonamento_nao_comutam() -> None:
    """O segundo par: o log é não linear, então ele não passa pela centragem.

    `log(x - média)` não é `log(x) - média(log x)`, e é essa não linearidade que faz a ordem
    importar. Com uma transformação **linear** as duas ordens coincidiriam — é o que torna
    este par informativo e não trivial.
    """
    gerador = np.random.default_rng(11)
    matriz = _matriz(np.abs(gerador.normal(100.0, 20.0, size=(6, 4))) + 1.0)

    primeiro = scale_features(transform(matriz, "log10"), "auto")
    segundo = transform(scale_features(matriz, "auto"), "log10")

    assert not np.allclose(np.asarray(primeiro.values), np.asarray(segundo.values))


def test_normalizacao_por_amostra_e_transformacao_nao_comutam() -> None:
    """O terceiro par, que fecha os três: o log não comuta com divisão por linha.

    `log(x / s)` é `log(x) - log(s)`, e dividir **depois** do log seria `log(x) / s`. As duas
    diferem sempre que `s != 1`.
    """
    gerador = np.random.default_rng(13)
    matriz = _matriz(np.abs(gerador.normal(100.0, 20.0, size=(6, 4))) + 1.0)

    primeiro = transform(normalize_samples(matriz, "sum"), "log10")
    segundo = normalize_samples(transform(matriz, "log10"), "sum")

    assert not np.allclose(np.asarray(primeiro.values), np.asarray(segundo.values))


def test_a_ordem_canonica_e_a_que_o_pipeline_aplica() -> None:
    """A ordem afirmada como sequência, e não como conjunto.

    Os três testes acima provam que a ordem **importa**; este fixa **qual** é. Sem ele, o
    argumento existiria e a escolha ficaria implícita.
    """
    gerador = np.random.default_rng(17)
    matriz = _matriz(np.abs(gerador.normal(100.0, 20.0, size=(6, 4))) + 1.0)

    canonica = scale_features(transform(normalize_samples(matriz, "sum"), "log10"), "auto")
    trocada = transform(scale_features(normalize_samples(matriz, "sum"), "auto"), "log10")

    assert not np.allclose(np.asarray(canonica.values), np.asarray(trocada.values))
    # E o `provenance` registra a sequência aplicada, que é o que o `--log` reproduz.
    aplicadas = [
        nota
        for nota in canonica.provenance
        if "normalize" in nota or "transform" in nota or "scale" in nota
    ]
    assert [nota.split("(")[0] for nota in aplicadas] == [
        "normalize_samples",
        "transform",
        "scale_features",
    ]


# ---------------------------------------------------------------------------
# Propriedade 2: a soma
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    valores=st.lists(
        st.lists(
            st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=8,
        ),
        min_size=3,
        max_size=8,
    )
)
def test_soma_faz_toda_linha_somar_a_constante(valores: list[list[float]]) -> None:
    """A propriedade obrigatória, sobre matrizes geradas.

    `min_value=0.1` mantém as linhas fora do caso degenerado — soma zero tem política própria
    e está testada em `test_normalization.py`. Aqui a propriedade é sobre o caso em que ela
    vale, e misturar os dois faria o teste afirmar as duas coisas ao mesmo tempo e nenhuma
    com clareza.
    """
    largura = min(len(linha) for linha in valores)
    matriz = np.array([linha[:largura] for linha in valores], dtype=np.float64)

    saida = np.asarray(normalize_samples(_matriz(matriz), "sum").values)
    assert np.allclose(saida.sum(axis=1), SOMA_CONSTANTE, rtol=1e-9)

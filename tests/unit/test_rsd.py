"""`rsd` e `qc_rsd` — as duas variantes, o denominador, e a política de indefinido.

O portão contra o R está em `tests/golden/test_qc_rsd_none_column.py`. Aqui ficam
as propriedades que não dependem dos arquivos do laboratório: o que a função faz
com vetor curto, média zero, `nan`, eixo, e escala.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from refino.domain.labels import InvalidLabelError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.metrics.rsd import (
    VARIANT_IMPLEMENTATIONS,
    boxplot_outlier_mask,
    qc_rsd,
    rsd,
)


def _matrix(values: np.ndarray, tipos: tuple[str, ...]) -> RefinoMatrix:
    samples = tuple(
        SampleMetadata(
            name=f"amostra{i + 1}" if tipo != "qc" else f"QC{i + 1}",
            injection_order=i + 1,
            sample_type=tipo,  # type: ignore[arg-type]
            group="QC" if tipo == "qc" else "NoHTLV",
        )
        for i, tipo in enumerate(tipos)
    )
    return RefinoMatrix(
        values=values,
        feature_names=tuple(f"f{j + 1}" for j in range(values.shape[1])),
        samples=samples,
        mode="teste",
    )


# ---------------------------------------------------------------------------
# A fórmula
# ---------------------------------------------------------------------------


def test_rsd_e_fracao_e_nao_percentual() -> None:
    """0,25 e não 25 — declara-se fração, e o `QC - RSD.csv` traz fração."""
    v = np.array([8.0, 12.0])  # média 10, sd(ddof=1) = 2.828...
    esperado = float(np.std(v, ddof=1) / np.mean(v))
    assert esperado < 1.0
    np.testing.assert_allclose(rsd(v), esperado)


def test_denominador_e_n_menos_um() -> None:
    v = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(rsd(v), np.std(v, ddof=1) / np.mean(v))
    assert not np.isclose(rsd(v), np.std(v, ddof=0) / np.mean(v))


def test_variante_outlier_removed_descarta_antes_de_medir() -> None:
    """E o efeito é para BAIXO — foi o sinal que denunciou o achado 4.8 errado."""
    v = np.array([10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 1e4])
    assert rsd(v, variant="outlier_removed") < rsd(v, variant="plain")


def test_sem_outlier_as_duas_variantes_coincidem() -> None:
    v = np.array([10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 10.3])
    np.testing.assert_allclose(rsd(v, variant="plain"), rsd(v, variant="outlier_removed"))


# ---------------------------------------------------------------------------
# Indefinido — a política aplicada à métrica
# ---------------------------------------------------------------------------


def test_media_zero_produz_nan_e_nao_inf() -> None:
    """É o caso das features cujos QCs são todos zero: 234 no `pos_hilic`."""
    assert np.isnan(rsd(np.zeros(7)))


def test_menos_de_dois_valores_produz_nan() -> None:
    assert np.isnan(rsd(np.array([42.0])))
    assert np.isnan(rsd(np.array([], dtype=np.float64)))


def test_um_unico_valor_presente_entre_ausentes_produz_nan() -> None:
    assert np.isnan(rsd(np.array([np.nan, 5.0, np.nan])))


def test_ausentes_nao_entram_no_calculo() -> None:
    com = np.array([1.0, np.nan, 2.0, 3.0, np.nan])
    sem = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(rsd(com), rsd(sem))


def test_um_ponto_presente_produz_nan_na_variante_com_remocao() -> None:
    """A remoção nunca reduz abaixo de dois — quem reduz é a ausência.

    A guarda contra "a remoção deixou menos de dois" é inalcançável por
    construção; a prova de que procurei o contraexemplo está em
    `test_remocao_nunca_deixa_menos_de_dois_valores`.
    """
    assert not np.isnan(rsd(np.array([np.nan, 1.0, 1.0]), variant="outlier_removed"))
    assert np.isnan(rsd(np.array([1.0, np.nan]), variant="outlier_removed"))


def test_soma_zero_com_sinais_opostos_produz_nan() -> None:
    """Média zero sem que nenhum valor seja zero. Intensidade negativa não ocorre
    em LC-MS, mas a função não deve devolver `inf` para ela."""
    assert np.isnan(rsd(np.array([-5.0, 5.0])))


# ---------------------------------------------------------------------------
# Eixo e forma
# ---------------------------------------------------------------------------


def test_axis_zero_e_rsd_por_feature() -> None:
    """Orientação canônica: amostras em linhas, features em colunas."""
    valores = np.array([[10.0, 100.0], [12.0, 100.0], [8.0, 100.0]])
    obtido = rsd(valores, axis=0)
    assert obtido.shape == (2,)
    np.testing.assert_allclose(obtido[0], rsd(valores[:, 0]))
    assert obtido[1] == 0.0, "coluna constante tem RSD zero, não indefinido"


def test_axis_um_opera_por_linha() -> None:
    valores = np.array([[10.0, 12.0, 8.0], [100.0, 100.0, 100.0]])
    obtido = rsd(valores, axis=1)
    assert obtido.shape == (2,)
    np.testing.assert_allclose(obtido[0], rsd(valores[0]))


def test_vetor_devolve_escalar_em_array() -> None:
    obtido = rsd(np.array([1.0, 2.0, 3.0]))
    assert obtido.shape == ()


def test_entrada_nao_e_modificada() -> None:
    v = np.array([1.0, 2.0, 3.0, 1e9])
    copia = v.copy()
    rsd(v, variant="outlier_removed")
    np.testing.assert_array_equal(v, copia)


# ---------------------------------------------------------------------------
# Propriedade de desenho: invariância a escala
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [2.0, 0.5, 1e6, 1e-6])
@pytest.mark.parametrize("variant", ["plain", "outlier_removed"])
def test_invariancia_exata_a_escala_positiva(c: float, variant: str) -> None:
    """`sd(cx)/mean(cx) = sd(x)/mean(x)` exatamente — decorre da definição.

    Vale nas duas variantes porque a máscara também é invariante a escala
    positiva: dobradiças e limites escalam juntos.
    """
    v = np.array([10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 1e4])
    np.testing.assert_allclose(
        rsd(v * c, variant=variant),  # type: ignore[arg-type]
        rsd(v, variant=variant),  # type: ignore[arg-type]
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# A guarda e o registro
# ---------------------------------------------------------------------------


def test_variante_invalida_levanta_em_vez_de_cair_no_outro_ramo() -> None:
    """O modo de falha insidioso: valor inválido devolvendo a outra fórmula."""
    with pytest.raises(InvalidLabelError):
        rsd(np.array([1.0, 2.0]), variant="outlier removed")  # type: ignore[arg-type]
    with pytest.raises(InvalidLabelError):
        rsd(np.array([1.0, 2.0]), variant="")  # type: ignore[arg-type]


def test_registro_de_variantes_cobre_o_literal() -> None:
    """A paridade completa está em `test_labels.py`; aqui fica a ponta local."""
    assert set(VARIANT_IMPLEMENTATIONS) == {"plain", "outlier_removed"}


# ---------------------------------------------------------------------------
# `qc_rsd`
# ---------------------------------------------------------------------------


def test_qc_rsd_usa_so_as_linhas_de_qc() -> None:
    valores = np.array([[10.0], [12.0], [8.0], [1e6], [2e6]])
    matrix = _matrix(valores, ("qc", "qc", "qc", "sample", "sample"))
    np.testing.assert_allclose(qc_rsd(matrix), rsd(valores[:3, 0]))


def test_qc_rsd_ignora_validate() -> None:
    """`validate` não é QC nem amostra — é o terceiro tipo, e já se mostrou que
    tratar `!= "qc"` como "amostra" corrompe a estatística."""
    valores = np.array([[10.0], [12.0], [8.0], [1e6]])
    com_validate = _matrix(valores, ("qc", "qc", "qc", "validate"))
    so_qc = _matrix(valores[:3], ("qc", "qc", "qc"))
    np.testing.assert_allclose(qc_rsd(com_validate), qc_rsd(so_qc))


def test_qc_rsd_sem_qc_devolve_nan_por_feature() -> None:
    """Matriz sem QC não tem régua: devolve indefinido, não levanta nem devolve
    número sobre as amostras."""
    valores = np.array([[10.0, 1.0], [12.0, 2.0]])
    matrix = _matrix(valores, ("sample", "sample"))
    obtido = qc_rsd(matrix)
    assert obtido.shape == (2,)
    assert np.all(np.isnan(obtido))


def test_qc_rsd_propaga_a_variante() -> None:
    valores = np.array([[10.0], [10.5], [9.5], [10.2], [9.8], [10.1], [1e4]])
    matrix = _matrix(valores, ("qc",) * 7)
    assert qc_rsd(matrix, variant="outlier_removed") < qc_rsd(matrix, variant="plain")


# ---------------------------------------------------------------------------
# Por que a guarda de "remoção deixou menos de dois" é inalcançável
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    valores=arrays(
        dtype=np.float64,
        shape=array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=40),
        elements=st.floats(min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False),
    )
)
def test_remocao_nunca_deixa_menos_de_dois_valores(valores: np.ndarray) -> None:
    """A guarda em `_rsd_1d` é defensiva e a construção a impede — e é isso que
    justifica o `pragma` nela, em vez de o `pragma` ser desistência.

    O argumento: as duas dobradiças são estatísticas de ordem em posições
    interiores. Quando `spread` é zero, todos os valores entre as duas posições
    são iguais ao valor da dobradiça, e são pelo menos três para `n` ímpar e
    quatro para `n` par — nenhum deles fica estritamente fora. Quando `spread` é
    positivo, os limites ficam além das dobradiças, e o intervalo entre elas
    sobrevive inteiro.

    Este teste não prova o argumento; ele o exercita em milhares de vetores, que é
    o que separa "acredito que seja inalcançável" de "procurei o contraexemplo".
    """
    mantidos = valores[~boxplot_outlier_mask(valores)]
    assert mantidos.size >= 2, (
        f"a remoção deixou {mantidos.size} valor(es) de {valores.size}: a guarda "
        "de `_rsd_1d` é alcançável e o `pragma` nela está errado"
    )

"""`boxplot_outlier_mask` contra o oráculo do R.

Dois testes carregam o critério, e eles são complementares:

- `test_paridade_com_o_oraculo_do_r` afirma que a implementação bate com o R nos
  36 vetores do fixture, nos dois `coef`, cobrindo `n` par e ímpar.
- `test_implementacao_por_quartil_type7_reprova_o_fixture` afirma que a
  implementação **errada** — a que a v1.1 da especificação pedia — reprova. Sem
  ele, "o fixture cobre n par" seria afirmação sobre o gerador que ninguém
  verifica, e o critério seria indistinguível de decoração.
"""

import json
from typing import Any

import numpy as np
import pytest

from refino.metrics.rsd import boxplot_outlier_mask, tukey_fivenum
from tests.conftest import REPO

FIXTURE = REPO / "data" / "fixtures" / "boxplot_stats_r.json"


def _carregar() -> list[dict[str, Any]]:
    if not FIXTURE.is_file():  # pragma: no cover - versionado; guarda para checkout parcial
        pytest.skip(
            f"{FIXTURE.relative_to(REPO)} ausente; a fixture é versionada, "
            "restaure-a do repositório"
        )
    dados: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    vetores: list[dict[str, Any]] = dados["vectors"]
    return vetores


VETORES = _carregar() if FIXTURE.is_file() else []
IDS = [v["id"] for v in VETORES]


def _valores(registro: dict[str, Any]) -> np.ndarray:
    return np.array(
        [np.nan if x is None else float(x) for x in registro["values"]], dtype=np.float64
    )


def _indices_1based(mask: np.ndarray) -> list[int]:
    return [int(i) + 1 for i in np.flatnonzero(mask)]


def _mascara_por_quartil_type7(v: np.ndarray, coef: float) -> np.ndarray:
    """A implementação que a v1.1 da especificação pedia — `numpy.percentile`
    com interpolação linear, que é o `type = 7` do R."""
    finite = v[np.isfinite(v)]
    if finite.size == 0:  # pragma: no cover - nenhum vetor do fixture é vazio
        return np.zeros(v.shape, dtype=bool)
    q1, q3 = np.percentile(finite, [25, 75], method="linear")
    iqr = q3 - q1
    resultado: np.ndarray = (v < q1 - coef * iqr) | (v > q3 + coef * iqr)
    return resultado


# ---------------------------------------------------------------------------
# Paridade com o R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("registro", VETORES, ids=IDS)
def test_paridade_com_o_oraculo_do_r(registro: dict[str, Any]) -> None:
    valores = _valores(registro)
    obtido = _indices_1based(boxplot_outlier_mask(valores, coef=float(registro["coef"])))
    assert obtido == registro["out_indices_1based"], (
        f"{registro['id']}: n={registro['n']}, coef={registro['coef']}. "
        f"R marcou {registro['out_indices_1based']}, obtive {obtido}"
    )


@pytest.mark.parametrize("registro", VETORES, ids=IDS)
def test_as_cinco_dobradicas_batem_com_fivenum_do_r(registro: dict[str, Any]) -> None:
    """Paridade mais estrita que a da máscara: os cinco números, não o desfecho.

    A máscara pode coincidir por acidente quando nenhum ponto está perto do
    limite; as dobradiças não.
    """
    esperado = np.array(
        [np.nan if x is None else float(x) for x in registro["fivenum"]], dtype=np.float64
    )
    obtido = tukey_fivenum(_valores(registro))
    np.testing.assert_allclose(obtido, esperado, rtol=0, atol=0, err_msg=registro["id"])


def test_implementacao_por_quartil_type7_reprova_o_fixture() -> None:
    """**O teste que prova que a paridade consegue reprovar.**

    Se este teste falhar, o fixture perdeu os vetores discriminantes e a paridade
    acima ficou decorativa — qualquer das duas famílias de quartil passaria.
    """
    divergencias = [
        registro["id"]
        for registro in VETORES
        if _indices_1based(_mascara_por_quartil_type7(_valores(registro), float(registro["coef"])))
        != registro["out_indices_1based"]
    ]
    assert divergencias, (
        "a implementação por quartil type 7 passou em TODOS os vetores do "
        "fixture. O critério não distingue as duas famílias — reveja a regra de "
        "seleção do gerador."
    )


def test_a_divergencia_do_type7_acontece_so_em_n_par() -> None:
    """A identidade algébrica, verificada dos dois lados.

    Para `n` ímpar as duas fórmulas são a mesma expressão: divergência ali seria
    defeito de uma das duas implementações, não propriedade dos dados.
    """
    for registro in VETORES:
        valores = _valores(registro)
        coef = float(registro["coef"])
        hinge = _indices_1based(boxplot_outlier_mask(valores, coef))
        type7 = _indices_1based(_mascara_por_quartil_type7(valores, coef))
        if registro["n_finite"] % 2 == 1:
            assert hinge == type7, (
                f"{registro['id']}: n_finite ímpar ({registro['n_finite']}) e as "
                "duas famílias discordaram — impossível por identidade algébrica"
            )


# ---------------------------------------------------------------------------
# Comportamento próprio da função
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("coef", [1.5, 3.0])
def test_com_no_maximo_dois_pontos_nada_e_marcado(n: int, coef: float) -> None:
    """Decorre da definição: as dobradiças são os extremos."""
    v = np.array([1.0, 1e9][:n], dtype=np.float64)
    assert not boxplot_outlier_mask(v, coef=coef).any()


def test_vetor_constante_nao_tem_outlier() -> None:
    v = np.full(9, 42.0)
    assert not boxplot_outlier_mask(v).any()


def test_ausente_nunca_e_marcado_e_infinito_e() -> None:
    """Transcrição verificada em R 4.4.2, e o resultado surpreende.

    `boxplot.stats(c(1,2,3,4,5,NA,Inf,-Inf,1e9))$out` devolve `Inf, -Inf` — e
    NÃO devolve 1e9, porque `±Inf` entra no `fivenum` e desloca a dobradiça
    superior para 5e8, cuja cerca fica em 1,25e9. Descartar infinito antes de
    calcular daria dobradiças diferentes E marcaria 1e9.
    """
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.inf, -np.inf, 1e9])
    mask = boxplot_outlier_mask(v)
    assert not mask[5], "ausente marcado"
    assert mask[6] and mask[7], "±Inf deixou de ser marcado"
    assert not mask[8], "1e9 marcado: sinal de que ±Inf foi descartado do cálculo das dobradiças"
    # O R imprime `5.0e+08` arredondado; o valor é a média dos dois vizinhos da
    # posição semi-inteira, `0.5*(5 + 1e9)`.
    np.testing.assert_allclose(
        tukey_fivenum(v),
        np.array([-np.inf, 0.5 * (1.0 + 2.0), 3.5, 0.5 * (5.0 + 1e9), np.inf]),
        rtol=0,
        atol=0,
    )


def test_dobradicas_ignoram_ausentes_mas_nao_infinitos() -> None:
    """`fivenum(x, na.rm=TRUE)`: remove `NA`, mantém `±Inf`."""
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    com_nan = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, np.nan, 6.0, 7.0])
    np.testing.assert_allclose(tukey_fivenum(base), tukey_fivenum(com_nan), rtol=0, atol=0)

    com_inf = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.inf])
    assert not np.allclose(tukey_fivenum(base)[3], tukey_fivenum(com_inf)[3]), (
        "infinito não alterou a dobradiça superior — foi descartado indevidamente"
    )


def test_vetor_sem_valor_presente_nao_marca_e_nao_levanta() -> None:
    v = np.array([np.nan, np.nan])
    assert not boxplot_outlier_mask(v).any()
    assert np.all(np.isnan(tukey_fivenum(v)))


def test_spread_indefinido_marca_infinito_e_nao_o_ausente() -> None:
    """O ramo `out <- !is.finite(x)` do R, restrito ao que ele devolve.

    `boxplot.stats(c(NA, Inf))$out` devolve `Inf`: o `iqr` é `NA`, o ramo marca
    todo não finito, e `out & nna` remove o ausente antes de devolver.
    """
    v = np.array([np.nan, np.inf])
    mask = boxplot_outlier_mask(v)
    assert mask[1], "Inf deixou de ser marcado no ramo de spread indefinido"
    assert not mask[0], "ausente marcado: falta o filtro `out & nna` do R"


def test_coef_zero_devolve_mascara_vazia() -> None:
    """No R é `do.out <- FALSE`, não o resultado da comparação.

    A diferença é grande: com `spread` zero a comparação marcaria todo valor fora
    das dobradiças, que num vetor comum é quase metade dele.
    """
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1e9])
    assert not boxplot_outlier_mask(v, coef=0.0).any()


def test_coef_negativo_levanta() -> None:
    """`stop("'coef' must not be negative")` no R."""
    with pytest.raises(ValueError, match="negativo"):
        boxplot_outlier_mask(np.arange(9, dtype=np.float64), coef=-1.0)


def test_duplicata_do_valor_do_outlier_e_marcada_nas_duas_posicoes() -> None:
    """O R remove por pertinência de valor; a máscara por limiar coincide.

    Se as duas formulações divergissem aqui, a réplica estaria errada no caso que
    o `remove_outlier` do `tools.R` de fato produz.
    """
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1e6, 1e6])
    mask = boxplot_outlier_mask(v)
    assert mask[7] and mask[8]
    assert mask.sum() == 2


def test_permutar_a_entrada_permuta_a_mascara() -> None:
    """A marcação não depende da ordem de apresentação."""
    rng = np.random.default_rng(20260726)
    v = np.concatenate([rng.normal(10, 1, 30), np.array([1e4])])
    ordem = rng.permutation(v.size)
    esperado = boxplot_outlier_mask(v)[ordem]
    np.testing.assert_array_equal(boxplot_outlier_mask(v[ordem]), esperado)


def test_invariancia_a_transformacao_afim_positiva() -> None:
    """Dobradiças e limites escalam juntos, então a máscara não muda."""
    rng = np.random.default_rng(20260726)
    v = np.concatenate([rng.normal(100, 5, 40), np.array([1e5])])
    for a, b in [(2.0, 0.0), (0.5, 3.0), (1e3, -7.0)]:
        np.testing.assert_array_equal(boxplot_outlier_mask(a * v + b), boxplot_outlier_mask(v))


def test_coef_maior_marca_subconjunto_do_que_coef_menor_marca() -> None:
    """Monotonicidade em `coef` — decorre de os limites se afastarem."""
    rng = np.random.default_rng(1)
    v = np.concatenate([rng.normal(0, 1, 60), np.array([8.0, -9.0, 20.0])])
    m15 = boxplot_outlier_mask(v, coef=1.5)
    m3 = boxplot_outlier_mask(v, coef=3.0)
    assert np.all(m3 <= m15), "coef=3 marcou ponto que coef=1.5 não marcou"


def test_mascara_tem_o_shape_da_entrada() -> None:
    v = np.arange(11, dtype=np.float64)
    mask = boxplot_outlier_mask(v)
    assert mask.shape == v.shape
    assert mask.dtype == bool


def test_matriz_levanta_em_vez_de_achatar() -> None:
    """Achatar silenciosamente daria número plausível sobre o vetor errado."""
    with pytest.raises(ValueError, match="vetor"):
        boxplot_outlier_mask(np.ones((3, 4)))


def test_entrada_nao_e_modificada() -> None:
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1e9])
    copia = v.copy()
    boxplot_outlier_mask(v)
    tukey_fivenum(v)
    np.testing.assert_array_equal(v, copia)


def test_o_fixture_foi_carregado() -> None:
    """Guarda contra a paridade passar vacuamente com zero casos coletados."""
    assert len(VETORES) >= 20, f"apenas {len(VETORES)} vetores coletados do fixture"

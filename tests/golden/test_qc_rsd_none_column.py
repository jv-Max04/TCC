"""A coluna `none` do `QC - RSD.csv`, o único oráculo in-sample.

Este é **o portão**: a única comparação de RSD contra artefato externo
que existe. Ele valida o instrumento, e é o que autoriza as doze medianas
in-sample a valerem como regressão.

Por que só a coluna `none`: as colunas `SERRF` e `batch_loess` vêm da
subamostragem repetida 5/2 do R (achado 4.3) — medem fora da amostra, com o
corretor reajustado a cada sorteio. Nenhuma implementação in-sample as reproduz,
e medi-lo: zero features dentro de 1e-6. Quem as cobra é o teste de contenção do R.
"""

import numpy as np
import pytest

from refino.metrics.rsd import qc_rsd
from tests.conftest import AnchorOutcome, matrix_from_toserrf, read_qc_rsd_column

pytestmark = pytest.mark.golden

# Contagens de RSD indefinido na coluna `none`, medidas nos quatro arquivos do
# laboratório. São OBSERVAÇÃO sobre estes arquivos, não invariante:
# um dataset em que nenhuma feature tenha todos os QCs zero daria zero aqui.
INDEFINIDOS_ESPERADOS = {
    "pos_hilic": 234,
    "neg_hilic": 212,
    "pos_c18": 60,
    "neg_c18": 33,
}


def test_reproduz_a_coluna_none(mode: str) -> None:
    """1e-6 em ≥99% das definidas **e** igualdade exata do conjunto de indefinidas.

    A segunda metade é a que tem dente. Sob "1e-6 em 99% das features" sem
    política de indefinido declarada, a mesma implementação correta passa ou
    reprova conforme se conte par de indefinidos como acordo: no `neg_hilic` a
    concordância cai de 100% para 89,60%, abaixo do portão.
    """
    matrix = matrix_from_toserrf(mode)
    features, esperado = read_qc_rsd_column(mode, "none")
    assert matrix.feature_names == features, "ordem de features divergente do QC - RSD.csv"

    obtido = qc_rsd(matrix, variant="outlier_removed")

    np.testing.assert_array_equal(
        np.isnan(obtido),
        np.isnan(esperado),
        err_msg=(
            f"{mode}: conjunto de features com RSD indefinido difere do R. "
            "Igualdade exata é exigida — 'quase o mesmo conjunto' significa que "
            "uma feature foi medida onde o R não mediu, ou o contrário."
        ),
    )

    definidos = ~np.isnan(obtido)
    erro = np.abs(obtido[definidos] - esperado[definidos])
    fracao = float(np.mean(erro < 1e-6))
    assert fracao >= 0.99, (
        f"{mode}: {100 * fracao:.2f}% das features definidas dentro de 1e-6 "
        f"(máximo erro {erro.max():.3e})"
    )


def test_contagem_de_indefinidos_bate_com_o_medido(mode: str) -> None:
    """Contagem esperada sobre os quatro arquivos do laboratório, não invariante."""
    matrix = matrix_from_toserrf(mode)
    obtido = qc_rsd(matrix, variant="outlier_removed")
    assert int(np.isnan(obtido).sum()) == INDEFINIDOS_ESPERADOS[mode], (
        f"{mode}: contagem de RSD indefinido mudou. Se um `measurements` foi "
        "regerado, é achado a relatar, não número a ajustar."
    )


def test_denominador_n_menos_um_e_o_que_reproduz(mode: str) -> None:
    """`ddof=0` reproduz ZERO das features — o denominador não é detalhe.

    A especificação detalhava o boxplot em sete passos e omitia isto. É o teste
    que transforma um dia de depuração em cinco palavras de contrato.
    """
    matrix = matrix_from_toserrf(mode)
    _, esperado = read_qc_rsd_column(mode, "none")
    qc = matrix.values[matrix.qc_mask]

    from refino.metrics.rsd import boxplot_outlier_mask

    populacional = []
    for coluna in qc.T:
        presentes = coluna[~np.isnan(coluna)]
        mantidos = presentes[~boxplot_outlier_mask(presentes)]
        media = float(np.mean(mantidos)) if mantidos.size >= 2 else 0.0
        populacional.append(float(np.std(mantidos, ddof=0) / media) if media != 0.0 else np.nan)
    ddof0 = np.array(populacional)

    definidos = ~np.isnan(ddof0) & ~np.isnan(esperado)
    dentro = float(np.mean(np.abs(ddof0[definidos] - esperado[definidos]) < 1e-6))
    assert dentro == 0.0, (
        f"{mode}: com ddof=0, {100 * dentro:.2f}% das features caíram dentro de "
        "1e-6. Se isso passar a acontecer, a hipótese de que o R usa n-1 precisa "
        "ser reexaminada."
    )


def test_a_variante_plain_nao_reproduz_a_coluna(mode: str) -> None:
    """A fórmula importa, e o efeito é grande: 167,77% contra 143,81%.

    Sem este teste, o portão passaria com a variante errada em algum modo em
    que as duas quase coincidem.
    """
    matrix = matrix_from_toserrf(mode)
    _, esperado = read_qc_rsd_column(mode, "none")
    plain = qc_rsd(matrix, variant="plain")

    definidos = ~np.isnan(plain) & ~np.isnan(esperado)
    dentro = float(np.mean(np.abs(plain[definidos] - esperado[definidos]) < 1e-6))
    assert dentro < 0.99, (
        f"{mode}: a variante `plain` reproduziu {100 * dentro:.2f}% da coluna. "
        "A coluna do R é `outlier_removed` (achado 4.8) — se `plain` também a "
        "reproduz, uma das duas variantes não está fazendo o que declara."
    )


def test_a_ancora_esta_verde(rsd_instrument_anchor: AnchorOutcome) -> None:
    """A fixture de sessão que os doze testes de mediana consomem.

    Este teste é o que a torna portão: ela mesma só expõe o resultado, porque
    fixture que afirma faz o dependente dar `error` em vez de `skip`.
    """
    assert rsd_instrument_anchor.ok, rsd_instrument_anchor.detalhe
    assert set(rsd_instrument_anchor.por_modo) == set(INDEFINIDOS_ESPERADOS), (
        "a âncora precisa cobrir os quatro modos para autorizar a regressão"
    )

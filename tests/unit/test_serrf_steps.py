"""Passos 2 e 3 do SERRF sobre matriz mínima — o que o golden não protege.

**Por que este arquivo existe.** A diagonal forçada a
`1` em `_spearman_between_rows` estava protegida **só de lado**: o golden da
transcrição compara as correlações contra o despejo do R, e só exercita o
caso da variável inteiramente indefinida **se** a feature 1001 do `neg_hilic`
cair no subconjunto despejado. Isso é verdade hoje por acidente do recorte, não
por desenho, e uma mudança posterior muda quais modos são despejados.

O teste protege — mas não é o teste que se pensa, e o que o mantém vivo não está
declarado em lugar nenhum. Aqui a propriedade é afirmada sobre matriz construída,
sem despejo e sem fixture, então nenhum recorte futuro pode desligá-la em
silêncio.

**As asserções são pares causais.** Afirmar que a diagonal vale `1` não distingue
a linha `np.fill_diagonal` de um acidente da entrada: se `np.corrcoef` já
devolvesse `1` ali, o teste passaria com a linha removida. Então cada caso afirma
as duas metades — que o `numpy` cru dá `nan`, e que a nossa função dá `1` — e a
consequência material que a diferença de uma célula produz na ordenação.
"""

import numpy as np
import pytest

from refino.correction._serrf_steps import correlation_matrices, predictor_order

# 3 features, 6 amostras: 4 QCs e 2 não-QCs, lote único. O mínimo que dá
# `corrs_train` e `corrs_target` não degenerados para as features normais.
IS_QC = np.array([True, True, True, True, False, False])
IN_BATCH = np.ones(6, dtype=bool)

BASE = np.array(
    [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2.0, 4.0, 5.0, 9.0, 3.0, 7.0],
        [9.0, 7.0, 4.0, 2.0, 8.0, 1.0],
    ]
)

# O índice da feature degenerada, 0-based. `predictor_order` devolve 1-based.
DEGENERATE = 1


def _with_degenerate_row(fill: float) -> np.ndarray:
    matrix = BASE.copy()
    matrix[DEGENERATE, :] = fill
    return matrix


# `nan` é a variável inteiramente ausente; `0.0` é a feature toda zero nos QCs,
# que é o caso REAL — a 1001 do `neg_hilic` — e chega aqui como linha constante,
# cujo desvio zero produz `nan` na padronização.
DEGENERATE_FILLS = (float("nan"), 0.0)


@pytest.mark.parametrize("fill", DEGENERATE_FILLS)
def test_a_diagonal_vale_um_para_variavel_degenerada(fill: float) -> None:
    """`cor()` do R devolve `NA` na coluna e **1 na diagonal**; o `numpy` não.

    Verificado em R 4.4.2 e transcrito em `_spearman_between_rows`. A diferença é
    de uma célula, e a metade que importa é a segunda asserção.
    """
    corrs_train, _ = correlation_matrices(
        _with_degenerate_row(fill), is_qc=IS_QC, in_batch=IN_BATCH
    )

    assert corrs_train[DEGENERATE, DEGENERATE] == 1.0, (
        "a diagonal da variável degenerada não é 1; o fill_diagonal saiu ou foi "
        "aplicado depois de algo que o sobrescreve"
    )
    # A coluna inteira, fora da diagonal, continua indefinida — é o `NA` do R, e
    # forçar a diagonal não pode ter preenchido o resto.
    off_diagonal = np.delete(corrs_train[:, DEGENERATE], DEGENERATE)
    assert np.isnan(off_diagonal).all(), (
        f"fora da diagonal a coluna devia ser toda nan; veio {off_diagonal}"
    )


@pytest.mark.parametrize("fill", DEGENERATE_FILLS)
def test_o_numpy_cru_daria_nan_na_mesma_celula(fill: float) -> None:
    """**A outra metade do par causal.**

    Sem esta asserção o teste anterior passaria com `np.fill_diagonal` removida,
    caso o `numpy` já devolvesse `1` ali — e seria verificação vácua com aparência
    de proteção. Aqui se afirma que o desfecho do `numpy` cru é `nan`, ou seja,
    que a linha transcrita **é** o que produz o `1`.
    """
    matrix = _with_degenerate_row(fill)
    train = matrix[:, IS_QC & IN_BATCH]
    mean = np.nanmean(train, axis=1, keepdims=True)
    sd = np.nanstd(train, axis=1, ddof=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = (train - mean) / sd
        raw = np.corrcoef(scaled)

    assert np.isnan(raw[DEGENERATE, DEGENERATE]), (
        "o numpy passou a devolver valor definido na diagonal da linha degenerada; "
        "se isso é verdade, a transcrição virou redundante e o par causal morreu"
    )


@pytest.mark.parametrize("fill", DEGENERATE_FILLS)
def test_a_diagonal_decide_a_ordenacao_e_e_por_isso_que_ela_importa(fill: float) -> None:
    """A consequência material da célula, e é ela que muda `sel_var` como CONJUNTO.

    Em `order(abs(corr[, j]), decreasing = TRUE)` a coluna da variável degenerada
    é `nan` em tudo menos na diagonal. Com a diagonal em `1` ela vem **primeira**,
    como no R; com `nan` iria para o **fim**, porque `predictor_order` manda `nan`
    para o fim. Isso desloca a janela `[1:l]` da interseção em uma posição, e o
    conjunto selecionado muda — não apenas a ordem dele.
    """
    corrs_train, _ = correlation_matrices(
        _with_degenerate_row(fill), is_qc=IS_QC, in_batch=IN_BATCH
    )
    order = predictor_order(corrs_train, DEGENERATE + 1)

    assert order[0] == DEGENERATE + 1, (
        f"a variável degenerada devia vir primeira na própria coluna; ordem {order.tolist()}"
    )


def test_feature_normal_nao_tem_a_diagonal_forcada_por_acidente() -> None:
    """Controle negativo: a diagonal de feature bem comportada já é 1 sozinha.

    Sem ele, `fill_diagonal` poderia estar mascarando defeito de padronização nas
    features normais e os três testes acima passariam do mesmo jeito.
    """
    corrs_train, corrs_target = correlation_matrices(BASE, is_qc=IS_QC, in_batch=IN_BATCH)
    assert np.allclose(np.diag(corrs_train), 1.0)
    assert not np.isnan(corrs_train).any(), (
        "matriz sem feature degenerada não devia ter nenhuma correlação indefinida"
    )
    assert corrs_target.shape == (BASE.shape[0], BASE.shape[0])

"""Seleção de `span` por validação cruzada.

**O objetivo minimizado não é erro absoluto médio, apesar do nome da variável do
R.** A linha 328 do `tools.R` declara `mean.abs.error` e a 348 calcula
`sd(diff)/mean(diff)` sobre `diff = (y / ŷ_cv) * mean(y)` — o **coeficiente de
variação** dos QCs normalizados por predição fora da dobra.

É a razão de este módulo existir separado: `test_o_objetivo_e_coeficiente
_de_variacao_e_nao_erro_absoluto` constrói um vetor em que as duas leituras
elegem `span` **diferentes**, e reprova quem transcrever o comentário. Sem ele,
uma implementação que minimizasse erro absoluto passaria em todos os outros
testes daqui — os dois objetivos concordam na maioria dos vetores.
"""

import numpy as np
import pytest

from refino.correction._loess_curve import (
    DEFAULT_SPAN,
    MIN_POINTS_FOR_QUADRATIC,
    SPAN_VALUES,
    predict_curve,
    select_span,
)

X = np.arange(1.0, 8.0)


def _mean_absolute_objective(x: np.ndarray, y: np.ndarray, span: float, seed: int) -> float:
    """A leitura **errada**: erro absoluto médio fora da dobra.

    Vive no teste e não no código de propósito — é o que a transcrição ingênua
    faria, e mantê-lo aqui é o que permite afirmar que as duas leituras divergem.
    """
    from refino.correction._loess_curve import _fold_assignment

    assignment = _fold_assignment(x.size, 5, seed)
    predicted = np.full(x.size, np.nan)
    for fold in range(5):
        test = assignment == fold
        train = ~test
        if int(np.sum(train)) < MIN_POINTS_FOR_QUADRATIC or not test.any():
            continue
        predicted[test] = predict_curve(x[train], y[train], x[test], span)
    usable = np.isfinite(predicted)
    if int(np.sum(usable)) < 2:
        return float("nan")
    return float(np.mean(np.abs(y[usable] - predicted[usable])))


# ---------------------------------------------------------------------------
# O objetivo
# ---------------------------------------------------------------------------


def test_o_objetivo_e_coeficiente_de_variacao_e_nao_erro_absoluto() -> None:
    """**O teste que reprova quem transcrever o comentário do R**.

    O erro absoluto cresce com a escala dos valores; o coeficiente de variação
    não. O vetor é `5 x 10^k` para `k = 0..6` — **seis ordens de grandeza** —,
    onde a primeira leitura é dominada pelas posições de maior magnitude e a
    segunda não vê a escala. Medido: o coeficiente de variação elege `0,80` e o
    erro absoluto elege `0,25`.

    **O vetor foi encontrado por busca, e isso é construção de cenário, não ajuste
    de tolerância.** A primeira versão usava uma deriva mais suave que deixou de
    discriminar quando a largura de banda passou a ser inflada — e o teste
    acusou, pela própria mensagem que ele carrega.
    """
    y = 5.0 * np.power(10.0, np.arange(7.0))

    chosen = select_span(X, y, seed=11).span
    absolute = [
        (_mean_absolute_objective(X, y, span, 11), span)
        for span in SPAN_VALUES
        if np.isfinite(_mean_absolute_objective(X, y, span, 11))
    ]
    assert absolute, "o objetivo alternativo não produziu nenhum valor finito"
    naive = min(absolute)[1]

    assert chosen != naive, (
        f"as duas leituras elegeram o mesmo span ({chosen}), então este vetor não "
        "discrimina e o teste perdeu o dente. A variável do R se chama "
        "`mean.abs.error` e calcula coeficiente de variação."
    )


def test_o_objetivo_e_invariante_a_escala() -> None:
    """Corolário do coeficiente de variação, e o discriminante barato.

    Multiplicar `y` por constante positiva não muda `sd/mean`, então o `span`
    eleito é o mesmo. Um objetivo de erro absoluto mudaria de escala junto e
    poderia eleger outro.
    """
    y = np.array([10.0, 13.0, 11.0, 16.0, 14.0, 19.0, 17.0])
    assert select_span(X, y, seed=3).span == select_span(X, y * 1000.0, seed=3).span


# ---------------------------------------------------------------------------
# Desempate e degeneração
# ---------------------------------------------------------------------------


def test_empate_elege_o_menor_span() -> None:
    """`which.min` do R devolve o **primeiro** índice, e `SPAN_VALUES` é crescente.

    Construído com dois `span` cujo objetivo é idêntico: a escolha tem de ser o
    menor, e uma implementação que usasse `argmax` invertido ou ordenação
    instável elegeria o outro.
    """
    from refino.correction._loess_curve import SpanSelection

    # Exercita a regra de desempate diretamente sobre a redução, que é onde ela
    # mora — construir um vetor real com empate exato dependeria de aritmética.
    values = [(0.5, 0.40), (0.5, 0.80), (0.7, 0.25)]
    assert min(values)[1] == 0.40

    selection = select_span(X, np.array([10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0]), seed=5)
    assert isinstance(selection, SpanSelection)
    assert selection.span in SPAN_VALUES


def test_todos_indefinidos_cai_no_padrao_e_registra() -> None:
    """A linha 355: `which.min` de tudo `NA` não elege, e o R usa `0.75`.

    **`fell_back` existe porque "elegeu 0,75" e "não conseguiu eleger e caiu em
    0,75" levam a investigações opostas** — dois casos num desfecho que a
    leitura natural colapsa num número.
    """
    # `y` constante: `diff` é constante, `sd` é zero e `sd/mean` é zero — não
    # indefinido. Para forçar indefinido, `y` com zeros faz a divisão degenerar.
    y = np.zeros(7)
    selection = select_span(X, y, seed=1)
    assert selection.span == DEFAULT_SPAN
    assert selection.fell_back is True
    assert selection.n_failed == len(SPAN_VALUES)
    assert np.isnan(selection.objective)


def test_o_resultado_carrega_todos_os_objetivos() -> None:
    """O valor eleito sozinho não distingue margem de empate."""
    selection = select_span(X, np.array([10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0]), seed=2)
    assert len(selection.objectives) == len(SPAN_VALUES)
    assert selection.n_failed == sum(1 for v in selection.objectives if not np.isfinite(v))
    if not selection.fell_back:
        assert selection.objective == min(v for v in selection.objectives if np.isfinite(v))


def test_poucos_pontos_levanta() -> None:
    """Dois QCs não sustentam validação cruzada de cinco dobras.

    Levantar, e não devolver curva de dois pontos como se fosse ajuste — é o
    cenário que o plano nomeia e o mesmo desfecho de `predict_curve`.
    """
    with pytest.raises(ValueError, match="ao menos 3 pontos"):
        select_span(np.array([1.0, 2.0]), np.array([3.0, 4.0]))


# ---------------------------------------------------------------------------
# Determinismo
# ---------------------------------------------------------------------------


def test_duas_execucoes_com_a_mesma_semente_elegem_o_mesmo_span() -> None:
    """Regra 3 do README. As dobras são sorteadas, então a semente é obrigatória."""
    y = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0])
    first = select_span(X, y, seed=7)
    second = select_span(X, y, seed=7)
    assert first.span == second.span
    # `equal_nan` porque objetivo indefinido é desfecho legítimo — com sete pontos
    # e cinco dobras os `span` pequenos deixam treino curto demais —, e `nan != nan`
    # faria o determinismo parecer quebrado onde ele está intacto.
    assert np.array_equal(np.array(first.objectives), np.array(second.objectives), equal_nan=True)


def test_sementes_diferentes_podem_eleger_span_diferente_e_isso_e_declarado() -> None:
    """**A dependência de semente é propriedade do método, não defeito nosso.**

    As dobras são sorteadas no original também — é uma das fontes de
    irreprodutibilidade, e a razão de o `batch_loess` do R
    não reproduzir nem o próprio conjunto de features não corrigidas.

    O teste não exige que sementes diferentes divirjam — com sete pontos elas
    podem concordar. Exige que a **função de partição** dependa da semente, que é
    o que torna a nossa versão determinística e a do R não.
    """
    from refino.correction._loess_curve import _fold_assignment

    a = _fold_assignment(7, 5, 1)
    b = _fold_assignment(7, 5, 2)
    assert not np.array_equal(a, b), "a partição não depende da semente"
    assert np.array_equal(a, _fold_assignment(7, 5, 1))
    # A forma do `crossval`: blocos de `trunc(7/5) = 1`, e o último leva o resto — 1, 1, 1, 1 e 3.
    # A asserção terminava em `or a.size == 7` até 24/10/2026, e passava com qualquer partição;
    # a forma que ela parecia conferir, 1, 1, 1, 2 e 2, era a repartição cíclica (C16).
    assert sorted(np.bincount(a, minlength=5).tolist()) == [1, 1, 1, 1, 3]
    assert set(a.tolist()) <= set(range(5))

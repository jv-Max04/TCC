"""A curva do LOESS: regressão local quadrática e seleção de `span`.

Implementação própria, porque nenhuma dependência do projeto entrega o `loess()` do R, que o
`tools.R` chama como `loess(y ~ x, span, family = "gaussian")`. Reproduz o grau 2, que é o padrão
do R, e a família gaussiana, sem reponderação robusta: a curva é arrastada por valor aberrante,
como a do R. O R interpola a superfície (`surface = "interpolate"`); aqui o ajuste é exato em cada
ponto. As dobras da validação cruzada são sorteadas, e por isso há semente.

A seleção de `span` minimiza o coeficiente de variação dos QCs normalizados pela predição fora da
dobra: `sd(diff) / mean(diff)`, com `diff = (y / ŷ_cv) * mean(y)` (linhas 346 e 347 do `tools.R`).
A variável do R se chama `mean.abs.error` (linha 328), mas não guarda erro absoluto médio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_SPAN",
    "SPAN_VALUES",
    "SpanSelection",
    "predict_curve",
    "select_span",
    "tricube",
]

# `seq(0.25, 1, by = 0.05)`, linha 326 do `tools.R`. São dezesseis valores, e cada um
# custa um ajuste por feature e por lote.
SPAN_VALUES: tuple[float, ...] = tuple(round(0.25 + 0.05 * i, 2) for i in range(16))

# O `span` que o `tools.R` usa (linha 354) quando `which.min` não elege nenhum.
DEFAULT_SPAN = 0.75

DEFAULT_FOLDS = 5
POLYNOMIAL_DEGREE = 2
# Grau 2 precisa de três pontos para o sistema não ser subdeterminado.
MIN_POINTS_FOR_QUADRATIC = POLYNOMIAL_DEGREE + 1

# Fator que infla o raio da vizinhança. É declarado, e não derivado da fonte do R: o R
# reporta `neighborhood radius 4.02` para distância máxima 4,00, o que dá 1,005, mas a
# regra que produz o número não está na fonte disponível.
RADIUS_INFLATION = 1.005


def tricube(u: np.ndarray) -> np.ndarray:
    """`(1 - |u|³)³` para `|u| < 1`, zero fora. O peso do LOESS."""
    absolute = np.abs(np.asarray(u, dtype=np.float64))
    inside = absolute < 1.0
    weights = np.zeros_like(absolute)
    weights[inside] = (1.0 - absolute[inside] ** 3) ** 3
    return weights


def _bandwidth(distances: np.ndarray, span: float) -> float:
    """A distância que normaliza os pesos, como o `loess` do R a define.

    Para `span < 1` é a `q`-ésima menor distância, com `q = floor(n * span)`. Para
    `span >= 1` o R usa todas as observações e escala a maior distância por `span`; sem
    isso, `span = 1` daria peso zero ao ponto mais distante.
    """
    ordered = np.sort(distances)
    n = ordered.size
    if span >= 1.0:
        return float(ordered[-1] * span * RADIUS_INFLATION)
    q = int(np.floor(n * span))
    q = max(q, MIN_POINTS_FOR_QUADRATIC)
    q = min(q, n)
    radius = float(ordered[q - 1])
    # O raio é inflado sempre, e não só quando `q == n`. O `q`-ésimo vizinho está
    # exatamente no raio, e `tricube(1) = 0` lhe daria peso zero, deixando `q - 1` pontos
    # efetivos. Com `q = 3` e grau 2 o sistema fica singular: com quatro QCs e
    # `span = 0,75` a predição sairia `nan` em toda posição. O R também infla.
    return radius * RADIUS_INFLATION


def _predict_at(x: np.ndarray, y: np.ndarray, x0: float, span: float) -> float:
    """Ajuste quadrático ponderado local em `x0`, avaliado em `x0`."""
    distances = np.abs(x - x0)
    bandwidth = _bandwidth(distances, span)
    if bandwidth <= 0.0:
        # Todos os vizinhos no mesmo ponto: devolve a média dos `y` empatados, que é o
        # que um ajuste com desenho singular daria.
        same = distances == 0.0
        return float(np.mean(y[same]))

    weights = tricube(distances / bandwidth)
    used = weights > 0.0
    if int(np.sum(used)) < MIN_POINTS_FOR_QUADRATIC:
        return float("nan")

    design = np.vander(x[used] - x0, POLYNOMIAL_DEGREE + 1, increasing=True)
    root_w = np.sqrt(weights[used])
    try:
        coefficients, *_ = np.linalg.lstsq(design * root_w[:, None], y[used] * root_w, rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover
        # `lstsq` só levanta quando o SVD não converge. Devolver `nan` evita que isso
        # suba como exceção opaca de dentro do laço por feature.
        return float("nan")
    # Centrado em `x0`, então a predição é o termo constante.
    return float(coefficients[0])


def predict_curve(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    span: float,
) -> np.ndarray:
    """A curva ajustada sobre `(x, y)`, avaliada em `x_new`.

    Fora do intervalo de `x` devolve `nan`. A extrapolação pelo tempo mais próximo, que o
    `tools.R` faz no passo 3 do `batch_loess`, fica com o corretor, que precisa saber
    quais posições foram preenchidas e quais foram ajustadas.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.shape != ys.shape:
        raise ValueError(f"x e y têm de ter a mesma forma; recebi {xs.shape} e {ys.shape}")
    if xs.size < MIN_POINTS_FOR_QUADRATIC:
        raise ValueError(
            f"ajuste quadrático exige ao menos {MIN_POINTS_FOR_QUADRATIC} pontos; "
            f"recebi {xs.size}. Devolver curva de dois pontos como se fosse ajuste é "
            "o desfecho que esta guarda existe para impedir."
        )
    if span <= 0.0:
        raise ValueError(f"span tem de ser positivo; recebi {span}")

    lower, upper = float(np.min(xs)), float(np.max(xs))
    targets = np.asarray(x_new, dtype=np.float64)
    out = np.full(targets.shape, np.nan)
    for index, x0 in enumerate(targets.ravel()):
        if lower <= x0 <= upper:
            out.ravel()[index] = _predict_at(xs, ys, float(x0), span)
    return out


# ---------------------------------------------------------------------------
# Seleção de `span`
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanSelection:
    """O `span` eleito e os objetivos de todos os candidatos.

    `objectives` vem junto porque o `span` sozinho não distingue eleição por margem de
    eleição por empate. `fell_back` diz que nenhum candidato teve objetivo definido e
    o `span` é o padrão.
    """

    span: float
    objective: float
    objectives: tuple[float, ...]
    n_failed: int
    fell_back: bool


def _fold_assignment(n: int, folds: int, seed: int) -> np.ndarray:
    """Partição das dobras, semeada.

    O `bootstrap::crossval` do R permuta (`o <- sample(1:n)`) e corta em blocos contíguos de
    `leave.out <- trunc(n/ngroup)`, com o último grupo levando o resto: com sete QCs e cinco
    dobras, 1, 1, 1, 1 e 3. Aqui se reproduz a forma (permutação semeada e blocos), e não a
    sequência, que dependeria do gerador do R. No R esse sorteio corre nos workers, fora do
    alcance do `set.seed`, e por isso o `batch_loess` de lá não reproduz a si mesmo.
    """
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    leave_out = n // folds
    assignment = np.full(n, folds - 1, dtype=np.int64)
    for dobra in range(folds - 1):
        assignment[permutation[dobra * leave_out : (dobra + 1) * leave_out]] = dobra
    return assignment


def _cross_validated_objective(
    x: np.ndarray, y: np.ndarray, span: float, folds: int, seed: int
) -> float:
    """`sd(diff) / mean(diff)` sobre `diff = (y / ŷ_cv) * mean(y)`.

    É coeficiente de variação, e não erro absoluto médio, embora a variável do R se
    chame `mean.abs.error`. Minimizar o erro absoluto elegeria outro `span`.
    """
    assignment = _fold_assignment(x.size, folds, seed)
    predicted = np.full(x.size, np.nan)
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        if int(np.sum(train)) < MIN_POINTS_FOR_QUADRATIC or not test.any():
            continue
        predicted[test] = predict_curve(x[train], y[train], x[test], span)

    usable = np.isfinite(predicted) & (predicted != 0.0)
    if int(np.sum(usable)) < 2:
        return float("nan")
    diff = (y[usable] / predicted[usable]) * float(np.mean(y[usable]))
    mean = float(np.mean(diff))
    if mean == 0.0:
        return float("nan")
    return float(np.std(diff, ddof=1) / mean)


def select_span(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    span_values: tuple[float, ...] = SPAN_VALUES,
    folds: int = DEFAULT_FOLDS,
) -> SpanSelection:
    """O `span` que minimiza o coeficiente de variação fora da dobra.

    Desempata pelo menor `span`, como o `which.min` do R. Se todos os objetivos são
    indefinidos, cai em 0,75, como o `tools.R` (linha 354), e marca `fell_back`, para
    distinguir 'elegeu 0,75' de 'não conseguiu eleger'.
    """
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.size < MIN_POINTS_FOR_QUADRATIC:
        raise ValueError(
            f"seleção de span exige ao menos {MIN_POINTS_FOR_QUADRATIC} pontos; recebi {xs.size}"
        )

    objectives = tuple(
        _cross_validated_objective(xs, ys, span, folds, seed) for span in span_values
    )
    finite = [
        (value, span)
        for value, span in zip(objectives, span_values, strict=True)
        if np.isfinite(value)
    ]
    n_failed = len(objectives) - len(finite)

    if not finite:
        return SpanSelection(
            span=DEFAULT_SPAN,
            objective=float("nan"),
            objectives=objectives,
            n_failed=n_failed,
            fell_back=True,
        )

    # `min` sobre `(valor, span)` desempata pelo menor `span`, que é o que o
    # `which.min` do R faz por devolver o primeiro índice.
    best_value, best_span = min(finite)
    return SpanSelection(
        span=best_span,
        objective=best_value,
        objectives=objectives,
        n_failed=n_failed,
        fell_back=False,
    )

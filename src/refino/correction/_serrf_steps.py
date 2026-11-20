"""Os passos do SERRF como funções puras, transcritos de `serrfR`, do `tools.R`.

Nada aqui corrige uma matriz: quem encadeia os passos é `serrf.py`. A separação permite alimentar
cada passo com a saída do R para o passo anterior e comparar por igualdade exata. Os passos 2 e 3
são determinísticos dada a entrada. O passo 1 sorteia os valores, mas os parâmetros da
distribuição são determinísticos, e `step1_params` os expõe.

No passo 1, o `min` da linha que substitui zeros é avaliado, no R, sobre o vetor que ainda os
contém. Logo `min = 0` sempre que há ao menos um zero, e daí `mean = 1` e `sd = 0.01` exatos,
qualquer que seja a magnitude da feature. Não é o mínimo positivo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RANGER_MAX_FEATURES",
    "RANGER_MIN_SAMPLES_SPLIT",
    "RANGER_N_ESTIMATORS",
    "Step1Params",
    "correlation_matrices",
    "normalize_qc",
    "normalize_samples",
    "predictor_order",
    "repair_assembled_row",
    "replace_non_finite",
    "rescale_by_median",
    "scale_by_columns_of",
    "select_predictors",
    "step1_params",
    "subtractive_outlier_repair",
    "usable_predictor_columns",
]

# Os literais do passo 1, das linhas 1042 e 1043 do `tools.R`.
ZERO_MEAN_OFFSET = 1.0
NA_MEAN_MIN_FACTOR = 0.5
SD_SLOPE = 0.1
SD_INTERCEPT = 0.1


@dataclass(frozen=True)
class Step1Params:
    """Parâmetros da chamada de `rnorm` para uma `(feature, lote)`, por ramo.

    `mean` e `sd` são o que o R pede à distribuição, e não os valores sorteados.
    """

    n: int
    n_zeros: int
    n_na: int
    min_non_na: float
    mean: float
    sd: float


def _min_non_na(values: np.ndarray) -> float:
    """`min(v[!is.na(v)])` do R, com o `Inf` que ele produz sobre vetor vazio.

    O R devolve `Inf` com aviso, e a média da `rnorm` herda o infinito. Devolver `nan`
    seria outro comportamento.
    """
    present = values[~np.isnan(values)]
    if present.size == 0:
        return float("inf")
    return float(np.min(present))


def step1_params(values: np.ndarray, *, phase: str) -> Step1Params:
    """Parâmetros do ramo `phase`, `"zeros"` ou `"na"`, para o vetor de um lote.

    Os dois ramos usam mínimos diferentes por causa da ordem de avaliação do R. O ramo dos
    zeros roda primeiro e vê o vetor original; o dos ausentes roda depois e vê o vetor com
    os zeros já substituídos, onde o mínimo não é mais zero. Com `phase="na"`, quem chama
    passa o vetor já substituído.
    """
    if phase not in ("zeros", "na"):
        raise ValueError(f"phase é 'zeros' ou 'na'; recebi {phase!r}")

    v = np.asarray(values, dtype=np.float64)
    minimum = _min_non_na(v)
    if phase == "zeros":
        mean = minimum + ZERO_MEAN_OFFSET
    else:
        mean = NA_MEAN_MIN_FACTOR * minimum + ZERO_MEAN_OFFSET
    return Step1Params(
        n=int(v.size),
        n_zeros=int(np.sum(v == 0)),
        n_na=int(np.sum(np.isnan(v))),
        min_non_na=minimum,
        mean=float(mean),
        sd=float(SD_SLOPE * (minimum + SD_INTERCEPT)),
    )


def _scale_rows(matrix: np.ndarray) -> np.ndarray:
    """`t(apply(m, 1, scale))` do R: z-score por linha, `ddof = 1`."""
    m = np.asarray(matrix, dtype=np.float64)
    mean = np.nanmean(m, axis=1, keepdims=True)
    sd = np.nanstd(m, axis=1, ddof=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray((m - mean) / sd, dtype=np.float64)


def _scale_columns(matrix: np.ndarray) -> np.ndarray:
    """`scale(m)` do R: z-score por coluna, `ddof = 1`.

    A assimetria com `_scale_rows` é a do passo 2 do `tools.R`: a linha 1053 escala por
    linha e a 1057 por coluna. Sob Spearman a primeira não tem efeito, porque transformação
    afim positiva por feature não muda posto, e a segunda tem.
    """
    m = np.asarray(matrix, dtype=np.float64)
    mean = np.nanmean(m, axis=0, keepdims=True)
    sd = np.nanstd(m, axis=0, ddof=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray((m - mean) / sd, dtype=np.float64)


def _spearman_between_rows(matrix: np.ndarray) -> np.ndarray:
    """`cor(t(m), method = "spearman")`: correlação entre as linhas de `m`.

    Spearman é Pearson sobre postos, e os postos são por linha (por feature) ao longo das
    colunas. `average` nos empates é o `ties.method` que o `cor` do R usa.
    """
    m = np.asarray(matrix, dtype=np.float64)
    ranks = np.empty_like(m)
    for i in range(m.shape[0]):
        ranks[i] = _average_ranks(m[i])
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.asarray(np.corrcoef(ranks), dtype=np.float64)

    # A diagonal é 1 mesmo para variável inteiramente `NaN`, como no R: `cor` sobre uma
    # linha constante devolve `NA` na coluna toda e 1 na diagonal, e `np.corrcoef`
    # devolve `NaN` também ali. A célula importa: em `order(abs(corr[, j]))` o R põe a
    # diagonal em primeiro e o Python a poria em último, o que desloca a janela `[1:l]`
    # da interseção e muda `sel_var` como conjunto. O caso existe nos dados reais: há
    # feature toda zero nos QCs.
    np.fill_diagonal(result, 1.0)
    return result


def _average_ranks(v: np.ndarray) -> np.ndarray:
    """Postos com média nos empates, propagando `nan`."""
    finite = ~np.isnan(v)
    out = np.full(v.shape, np.nan)
    if not finite.any():
        return out
    values = v[finite]
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # Média nos empates, grupo a grupo.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(unique.size, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    out[finite] = ranks
    return out


def correlation_matrices(
    before_step1: np.ndarray,
    *,
    is_qc: np.ndarray,
    in_batch: np.ndarray,
    is_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """As duas matrizes do passo 2 para um lote, `(corrs_train, corrs_target)`.

    `before_step1` é `(features, amostras)`, a orientação do arquivo e a que o R usa aqui; a
    transposição da ingestão não acontece neste módulo.

    A matriz é a de antes do passo 1. As linhas 1053 e 1057 do `tools.R` escalam `train` e
    `target`, os argumentos de `serrfR`, e o passo 1 altera `all`, que é uma cópia (`cbind`,
    linha 1037).

    `is_target` são as colunas do alvo. Ausente, o alvo é `~is_qc`, a forma do R dentro de
    `serrfR`, onde só há QCs e amostras. Com amostras `validate` na matriz, quem chama passa a
    máscara das amostras de estudo: o R chama o `serrfR` das amostras com `target = e_sample`
    (linha 1235) e corrige `validate` numa chamada à parte (linha 1289).
    """
    matrix = np.asarray(before_step1, dtype=np.float64)
    alvo = ~is_qc if is_target is None else np.asarray(is_target, dtype=bool)
    train = matrix[:, is_qc & in_batch]
    target = matrix[:, alvo & in_batch]
    return (
        _spearman_between_rows(_scale_rows(train)),
        _spearman_between_rows(_scale_columns(target)),
    )


def predictor_order(correlations: np.ndarray, j: int) -> np.ndarray:
    """`order(abs(corr[, j]), decreasing = TRUE)` do R, em índices 1-based.

    O `order` do R deixa empates na ordem original (índice crescente) e põe `NA` no fim.
    `argsort(kind="stable")` sobre `-|x|`, com `nan` mapeado para `+inf`, reproduz os dois
    comportamentos.

    O desempate decide a seleção: no `neg_hilic`, cerca de um quarto dos valores de `|corr|`
    está empatado, e com duas colunas no alvo a correlação é ±1 ou indefinida.
    """
    column = np.abs(np.asarray(correlations, dtype=np.float64)[:, j - 1])
    key = -column
    key = np.where(np.isnan(key), np.inf, key)
    return np.asarray(np.argsort(key, kind="stable") + 1, dtype=np.int64)


def select_predictors(
    corrs_train: np.ndarray,
    corrs_target: np.ndarray,
    j: int,
    num: int,
    *,
    max_iterations: int | None = None,
) -> list[int]:
    """`sel_var` do passo 3, em índices 1-based, na ordem que o R produz.

    Transcrição do laço das linhas 1091 a 1097. O `intersect` do R preserva a ordem do
    primeiro argumento e remove duplicatas, e é essa ordem que o passo 4 consome.

    O teto de iterações não existe no R. Lá o `while(length(sel_var) < num)` cresce `l`
    indefinidamente e, com menos features que `num`, não termina. Aqui a função devolve o
    que houver, e quem chama registra o número efetivo.
    """
    n_features = int(np.asarray(corrs_train).shape[0])
    ceiling = max_iterations if max_iterations is not None else n_features

    train_order = predictor_order(corrs_train, j)
    target_order = predictor_order(corrs_target, j)

    selected: list[int] = []
    length = num
    while len(selected) < num and length <= ceiling:
        head_target = set(target_order[:length].tolist())
        selected = [int(v) for v in train_order[:length] if int(v) in head_target and int(v) != j]
        length += 1
    return selected


# ---------------------------------------------------------------------------
# Passos 4 a 10
# ---------------------------------------------------------------------------
#
# As funções recebem `local` e `global_` separados. No `tools.R`, `e[j, …]` é o lote
# corrente e `all[j, …]` é a matriz inteira depois do passo 1. Com um lote os dois
# coincidem numericamente, e trocar um pelo outro só aparece com dois lotes
# (`test_serrf_scope.py`).

# Os hiperparâmetros do `ranger` que a floresta reproduz. São explícitos para não
# dependerem dos padrões do `scikit-learn`, que mudam entre versões.
RANGER_N_ESTIMATORS = 500
RANGER_MAX_FEATURES = "sqrt"
# `min.node.size` do `ranger` corresponde a `min_samples_split`, e não a
# `min_samples_leaf`. A documentação do `ranger` diz "minimal node size to split at":
# restringe o nó candidato a dividir, e as folhas podem ficar menores. Com sete QCs,
# exigir 5 em cada folha impede toda divisão; a floresta prediz a média de `y`, que é
# centrado no passo 4, e a predição vira zero.
RANGER_MIN_SAMPLES_SPLIT = 5

# Passo 8: `N(0, 0.01 * sd(finitos))`, linha 1156.
STEP8_SD_FACTOR = 0.01
# Passo 10: `N(min(presentes), 0.1 * sd(presentes))`, linhas 1213 a 1229.
STEP10_SD_FACTOR = 0.1


def scale_by_columns_of(block: np.ndarray) -> np.ndarray:
    """Padroniza cada feature pelas estatísticas do próprio bloco.

    `block` é `(features, amostras)`. O passo 4 chama duas vezes, com o bloco de treino e
    com o de teste: o R padroniza `X_teste` com a média e o desvio das amostras, e não com
    os do treino. Um `StandardScaler` ajustado no treino e aplicado ao teste mudaria toda
    predição.
    """
    m = np.asarray(block, dtype=np.float64)
    mean = np.nanmean(m, axis=1, keepdims=True)
    sd = np.nanstd(m, axis=1, ddof=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray((m - mean) / sd, dtype=np.float64)


def usable_predictor_columns(x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Máscara das features preditoras sem `NaN` no treino nem no teste.

    Linhas 1111 a 1124: o descarte é pela união dos defeitos. Preditora utilizável no
    treino e degenerada no teste não serve, porque a predição que importa é a do teste.
    """
    train = np.asarray(x_train, dtype=np.float64)
    test = np.asarray(x_test, dtype=np.float64)
    return ~(np.isnan(train).any(axis=1) | np.isnan(test).any(axis=1))


def normalize_qc(
    local_qc: np.ndarray, predicted_qc: np.ndarray, global_qc: np.ndarray
) -> np.ndarray:
    """Passo 5, linha 1144. Média, e não mediana, nos dois lados.

        norm[qc] = e[j, qc] / ((pred_qc + mean(e[j, qc])) / mean(all[j, qc]))

    `global_qc` são os QCs de todos os lotes, depois do passo 1. A assimetria com o passo 6
    (média aqui, mediana lá) é a do R, e fica porque os testes comparam este caminho com ele.
    """
    local = np.asarray(local_qc, dtype=np.float64)
    denominator = (np.asarray(predicted_qc, dtype=np.float64) + np.mean(local)) / np.mean(
        np.asarray(global_qc, dtype=np.float64)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(local / denominator, dtype=np.float64)


def normalize_samples(
    local_sample: np.ndarray, predicted_sample: np.ndarray, global_sample: np.ndarray
) -> np.ndarray:
    """Passo 6, linhas 1147 e 1148. Média no numerador, mediana no divisor global.

        norm[sample] = e[j, sample] / ((pred_sample + mean(e[j, sample])) / median(all[j, sample]))

    Os negativos voltam ao valor original (linha 1148). O sorteio de negativos é do passo
    10, noutro escopo.
    """
    local = np.asarray(local_sample, dtype=np.float64)
    denominator = (np.asarray(predicted_sample, dtype=np.float64) + np.mean(local)) / np.median(
        np.asarray(global_sample, dtype=np.float64)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.asarray(local / denominator, dtype=np.float64)
    negative = result < 0
    result[negative] = local[negative]
    return result


def rescale_by_median(normalized: np.ndarray, global_reference: np.ndarray) -> np.ndarray:
    """Passo 7, linhas 1154 e 1155, aplicado a QCs e amostras separadamente.

        norm = norm / (median(norm) / median(all[j, ...]))

    A referência é global e de depois do passo 1: a mediana do vetor com os zeros já
    substituídos por `≈1`, e não a da matriz de entrada.
    """
    values = np.asarray(normalized, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        factor = np.median(values) / np.median(np.asarray(global_reference, dtype=np.float64))
        return np.asarray(values / factor, dtype=np.float64)


def replace_non_finite(vector: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Passo 8, linha 1156. `N(0, 0.01 * sd(finitos))` nas posições não finitas.

    O vetor é o do lote corrente inteiro, QCs e amostras juntos, e `!is.finite` cobre
    `±Inf`, `NaN` e ausente. As posições finitas não mudam.
    """
    values = np.array(np.asarray(vector, dtype=np.float64), copy=True)
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return values
    scale = STEP8_SD_FACTOR * float(np.std(values[finite], ddof=1))
    values[~finite] = rng.normal(0.0, scale, size=int((~finite).sum()))
    return values


def subtractive_outlier_repair(
    normalized_sample: np.ndarray,
    raw_sample: np.ndarray,
    predicted_sample: np.ndarray,
    global_sample: np.ndarray,
    outlier_sample_mask: np.ndarray,
) -> np.ndarray:
    """Passo 9, forma subtrativa, linhas 1162 e 1163.

        e[j, sample] - ((pred_sample + mean(e[j, sample])) - median(all[j, sample]))

    Só nas posições de amostra marcadas. A máscara vem de `boxplot_outlier_mask(coef=3)`
    sobre o vetor do lote inteiro, QCs e amostras juntos, e só depois é restrita às
    amostras. É o que a linha 1161 faz: `boxplot.stats(norm, coef = 3)$out` vê o vetor
    completo e a substituição indexa `norm[!train.index=='qc']`. Calcular a máscara só
    sobre as amostras daria dobradiças diferentes e outro conjunto marcado.

    `raw_sample` é `e[j, sample]`, de `e_current_batch` (linha 1083): o lote depois do passo
    1 e sem normalizar. Dele sai o valor das posições marcadas, e a ele revertem os
    negativos (linha 1163), marcados ou não. O que não é marcado nem negativo continua com
    o valor de `normalized_sample`.
    """
    normalized = np.asarray(normalized_sample, dtype=np.float64)
    raw = np.asarray(raw_sample, dtype=np.float64)
    mask = np.asarray(outlier_sample_mask, dtype=bool)
    shift = (np.asarray(predicted_sample, dtype=np.float64) + np.mean(raw)) - np.median(
        np.asarray(global_sample, dtype=np.float64)
    )
    result = np.array(normalized, copy=True)
    with np.errstate(invalid="ignore"):
        repaired = raw - shift
    result[mask] = repaired[mask]
    negative = result < 0
    result[negative] = raw[negative]
    return result


def repair_assembled_row(row: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Passo 10, linhas 1213 a 1229. Um sorteio uniforme por linha, e não por posição.

    Opera sobre a linha montada, atravessando lotes, e é chamado com o bloco de QC e o de
    amostra separadamente. Na ordem do R:

    - ausentes recebem `N(min(presentes), 0.1 * sd(presentes))`;
    - negativos recebem `runif(1) * min(positivos)`, com um único `runif(1)` reutilizado
      em todas as posições negativas da linha.

    Não é o passo 8, que opera dentro de um lote. O RSD da coluna `SERRF` do R é medido
    depois deste reparo (linha 1271).
    """
    values = np.array(np.asarray(row, dtype=np.float64), copy=True)

    missing = np.isnan(values)
    present = values[~missing]
    if missing.any() and present.size > 0:
        centre = float(np.min(present))
        scale = STEP10_SD_FACTOR * float(np.std(present, ddof=1)) if present.size > 1 else 0.0
        values[missing] = rng.normal(centre, scale, size=int(missing.sum()))

    negative = values < 0
    if negative.any():
        positives = values[values > 0]
        if positives.size > 0:
            # Um sorteio, reutilizado em todas as posições negativas, como no R. Um por
            # posição daria dispersão que o original não tem.
            multiplier = float(rng.uniform())
            values[negative] = multiplier * float(np.min(positives))
    return values

"""RSD e a réplica de `boxplot.stats` do R.

`boxplot.stats` não usa `quantile`: usa `fivenum`, que devolve as dobradiças de
Tukey. As duas famílias coincidem para `n` ímpar (a posição da dobradiça é
`(n+3)/4` e a do quartil `type = 7` é `1 + (n-1)/4`, a mesma expressão) e
divergem em um quarto de posição de rank para `n` par.

Com os 7 QCs deste estudo a escolha não muda nada. O passo 9 do SERRF opera
sobre as 68 injeções do lote, número par, e ali a escolha errada muda o conjunto
de amostras que recebe a fórmula subtrativa em 3% a 6% das features.
"""

import math
from collections.abc import Callable

import numpy as np

from refino.domain.labels import RsdVariant, validate_label
from refino.domain.matrix import RefinoMatrix

__all__ = [
    "VARIANT_IMPLEMENTATIONS",
    "boxplot_outlier_mask",
    "qc_rsd",
    "rsd",
    "tukey_fivenum",
]

# Mínimo de valores para que `sd` com denominador `n - 1` exista.
MIN_VALUES_FOR_SD = 2


def tukey_fivenum(v: np.ndarray) -> np.ndarray:
    """Réplica de `fivenum()` do R: mínimo, dobradiça inferior, mediana,
    dobradiça superior, máximo.

    Descarta apenas ausentes, como `fivenum(x, na.rm = TRUE)`: `±Inf` permanece
    e entra na contagem e na ordenação. É o que o R faz (o comentário
    `# including +/- Inf` está na fonte de `boxplot.stats`, ao lado de
    `n <- sum(nna)`): `fivenum(c(1,2,3,4,5,NA,Inf,-Inf,1e9), na.rm=TRUE)` devolve
    `-Inf, 1.5, 3.5, 5e8, Inf`. Nenhum vetor dos arquivos de referência tem
    infinito, então descartar `±Inf` passaria nos testes golden e ainda assim
    daria dobradiças diferentes das do R.

    Vetor sem nenhum valor não ausente devolve cinco `nan`.

    A transcrição é literal, incluindo os índices semi-inteiros:

        n4 = floor((n + 3) / 2) / 2
        d  = (1, n4, (n + 1) / 2, n + 1 - n4, n)
        H  = 0.5 * (x[floor(d)] + x[ceil(d)])
    """
    values = np.asarray(v, dtype=np.float64)
    presentes = np.sort(values[~np.isnan(values)])
    n = presentes.size
    if n == 0:
        return np.full(5, np.nan)

    n4 = math.floor((n + 3) / 2) / 2
    d = np.array([1.0, n4, (n + 1) / 2, n + 1 - n4, float(n)])
    lo = np.floor(d).astype(np.intp) - 1
    hi = np.ceil(d).astype(np.intp) - 1
    result: np.ndarray = 0.5 * (presentes[lo] + presentes[hi])
    return result


def boxplot_outlier_mask(v: np.ndarray, coef: float = 1.5) -> np.ndarray:
    """Réplica de `boxplot.stats(v, coef)$out` do R, como máscara booleana.

    Marca todo valor estritamente fora de `H1 - coef*spread` e
    `H3 + coef*spread`, onde `spread = H3 - H1` é a distância entre as
    dobradiças, e não o IQR de quartis.

    Devolve máscara porque índice é o que o passo 9 do SERRF consome. O R devolve
    valores e depois faz `which(v %in% out)`, o que é equivalente, inclusive
    quanto a duplicatas.

    `coef` é 1,5 no `remove_outlier()`/`RSD()` do `tools.R` e 3 no passo 9 do
    SERRF. Os casos de borda seguem a fonte de `boxplot.stats`, e nenhum deles é
    alcançado pelos arquivos de referência:

    - `coef < 0` levanta, como o `stop("'coef' must not be negative")` do R.
    - `coef == 0` devolve máscara vazia. No R é `do.out <- FALSE`, e não a
      comparação com `spread` zero, que marcaria tudo fora das dobradiças.
    - `spread` indefinido (um só valor presente, ou as duas dobradiças no mesmo
      infinito) cai no ramo `out <- !is.finite(x)` do R: marca `±Inf` e nada
      mais. `nan` fica de fora porque o R filtra com `out & nna`.

    Ausente nunca é marcado: a comparação com `nan` é falsa nas duas direções.

    Com `n <= 2` nenhum ponto é marcado para `coef > 0`: as dobradiças são os
    próprios extremos, e os limites caem fora da faixa dos dados. Isso importa
    porque uma partição em dois lotes pode deixar um lote com dois QCs.
    """
    values = np.asarray(v, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"boxplot_outlier_mask opera sobre vetor; recebi shape {values.shape}. "
            "Para matriz, aplique por eixo em `rsd`."
        )
    if coef < 0:
        raise ValueError(f"coef não pode ser negativo; recebi {coef}")
    if coef == 0:
        return np.zeros(values.shape, dtype=bool)

    hinges = tukey_fivenum(values)
    # `inf - inf` é `nan`: é o caso em que o R cai no outro ramo.
    with np.errstate(invalid="ignore"):
        spread = hinges[3] - hinges[1]
    if not np.isfinite(spread):
        # O ramo `!is.finite(x)` do R, restrito ao que ele de fato devolve.
        return np.isinf(values)

    lower = hinges[1] - coef * spread
    upper = hinges[3] + coef * spread
    mask: np.ndarray = (values < lower) | (values > upper)
    return mask


# ---------------------------------------------------------------------------
# RSD
# ---------------------------------------------------------------------------


def _keep_all(v: np.ndarray) -> np.ndarray:
    """Variante `plain`: usa todos os valores presentes."""
    return np.asarray(v, dtype=np.float64)


def _keep_non_outliers(v: np.ndarray) -> np.ndarray:
    """Variante `outlier_removed`: o que a `RSD()` do `tools.R` mede.

    `remove_outlier()` descarta o que `boxplot.stats()` marca com `coef = 1,5` e
    só então calcula `sd/mean`. O efeito é grande e para baixo: no `pos_hilic`
    sem correção, a mediana vai de 167,77% para 143,81%.
    """
    values = np.asarray(v, dtype=np.float64)
    mantidos: np.ndarray = values[~boxplot_outlier_mask(values)]
    return mantidos


# O teste de paridade compara as chaves deste registro com `get_args(RsdVariant)`,
# nas duas direções: chave sem valor no `Literal`, ou o inverso, reprova.
VARIANT_IMPLEMENTATIONS: dict[RsdVariant, Callable[[np.ndarray], np.ndarray]] = {
    "plain": _keep_all,
    "outlier_removed": _keep_non_outliers,
}


def _rsd_1d(v: np.ndarray, variant: RsdVariant) -> float:
    """RSD de um vetor, ou `nan` quando indefinido.

    Indefinido tem três causas, e nenhuma é erro:

    - menos de dois valores presentes: `sd` com denominador `n - 1` não existe;
    - média zero: a razão não existe. Acontece com QCs todos zero e, com mais
      frequência sob `outlier_removed`, quando a remoção apaga os QCs não nulos e
      deixa zeros, que é feature detectada em um único QC;
    - a remoção de outliers deixou menos de dois valores.

    Devolve `nan`, e não exceção: RSD indefinido é informação sobre a feature.
    Quem decide o que fazer com ela é `filter_by_qc_rsd`, que a reprova.
    """
    values = np.asarray(v, dtype=np.float64)
    presentes = values[~np.isnan(values)]
    if presentes.size < MIN_VALUES_FOR_SD:
        return float("nan")

    mantidos = VARIANT_IMPLEMENTATIONS[variant](presentes)
    if mantidos.size < MIN_VALUES_FOR_SD:  # pragma: no cover
        # Inalcançável: as dobradiças são estatísticas de ordem em posições
        # interiores, então o intervalo entre elas sobrevive à remoção. A guarda
        # fica para que o `sd` de um ponto não vire `nan` silencioso mais acima.
        # `test_remocao_nunca_deixa_menos_de_dois_valores` procura o contraexemplo.
        return float("nan")

    # `errstate` porque `inf - inf` é `nan` aqui, e o desfecho é o correto: quem
    # tem medição infinita não tem RSD. As matrizes golden do `batch_loess` têm
    # células infinitas. Sob `plain` a feature fica indefinida; sob
    # `outlier_removed` o infinito já saiu pelo ramo `!is.finite` e ela pode ter
    # RSD definido.
    with np.errstate(invalid="ignore"):
        mean = float(np.mean(mantidos))
        if mean == 0.0:
            return float("nan")
        # `ddof=1` porque é o `sd` do R. Com `ddof=0` nenhuma feature do
        # `QC - RSD.csv` é reproduzida.
        return float(np.std(mantidos, ddof=1) / mean)


def rsd(
    values: np.ndarray,
    axis: int = 0,
    variant: RsdVariant = "plain",
) -> np.ndarray:
    """RSD ao longo de `axis`. Devolve fração (0,25 = 25%), e não percentual.

    Na orientação interna, com amostras em linhas e features em colunas, `axis=0`
    é o RSD por feature.

    `variant` é validado em runtime: valor inválido levanta em vez de cair num
    ramo que devolveria a outra fórmula.
    """
    validate_label("RsdVariant", variant)
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        return np.array(_rsd_1d(matrix, variant))
    result: np.ndarray = np.apply_along_axis(lambda v: _rsd_1d(v, variant), axis, matrix)
    return result


def qc_rsd(matrix: RefinoMatrix, variant: RsdVariant = "plain") -> np.ndarray:
    """RSD por feature sobre as linhas de `qc_mask`. Estimador in-sample.

    Do `QC - RSD.csv`, só a coluna `none` é reproduzível assim. As colunas `SERRF`
    e `batch_loess` vêm da subamostragem repetida 5/2 do R, que mede fora da
    amostra, e quem as reproduz é o estimador de compatibilidade.

    Não compare o resultado desta função entre métodos para julgar qual corrigiu
    melhor. RSD in-sample premia o método que gasta graus de liberdade nos
    pontos que a métrica mede: o LOESS ajusta a curva através dos QCs e mede nos
    QCs, e a mediana in-sample dele é 1,9e-15 no `pos_c18`.
    """
    qc_values = matrix.values[matrix.qc_mask]
    if qc_values.shape[0] == 0:
        return np.full(matrix.n_features, np.nan)
    return rsd(qc_values, axis=0, variant=variant)

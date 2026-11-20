"""Concordância entre dois vetores de RSD e o piso aritmético dela.

`1 - spearman(a, a)`, o vetor contra si mesmo, não é exatamente zero para `n`
grande: a diferença é da ordem de 1e-16. Por isso a divergência é comparada com
esse piso medido, e não com zero nem com um limiar escolhido, que mediria a
aritmética do `scipy`.

As fórmulas ficam em `src/`, e não nos testes que as usam como critério, para que
ninguém as ajuste como se fossem parte do teste.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "Divergence",
    "self_divergence_floor",
    "spearman_divergence",
    "tolerance_from_dispersions",
]

# Spearman precisa de três pontos: abaixo disso a correlação é ±1 ou indefinida
# e não carrega informação.
MIN_POINTS_FOR_SPEARMAN = 3


@dataclass(frozen=True)
class Divergence:
    """`d = 1 - rho` com o que o número precisa para ser lido.

    `n_used` vai junto porque `rho` sobre outro subconjunto é outro `rho`, e
    `floor` porque `d` se compara com o piso, e não com zero.
    """

    d: float
    n_used: int
    n_excluded: int
    floor: float

    @property
    def is_at_floor(self) -> bool:
        """`True` quando a divergência não excede o piso aritmético.

        Substitui a asserção `d == 0`: os dois vetores concordam tanto quanto a
        aritmética permite.
        """
        return self.d <= self.floor


def _defined_in_both(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def self_divergence_floor(values: np.ndarray) -> float:
    """`1 - spearman(v, v)` sobre as posições definidas: o piso da comparação.

    É medido, e não suposto zero: para `n` grande o `spearmanr` do `scipy` não
    devolve 1 exato, e exigir `d == 0` de dois vetores bit-idênticos reprovaria.
    """
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size < MIN_POINTS_FOR_SPEARMAN:
        return float("nan")
    return 1.0 - float(stats.spearmanr(finite, finite).statistic)


def spearman_divergence(a: np.ndarray, b: np.ndarray) -> Divergence:
    """`d = 1 - rho_Spearman` entre dois vetores de RSD, com contagem e piso.

    Exclui par a par as features indefinidas em qualquer um dos dois. Excluir pela
    união de várias comparações descartaria feature utilizável neste par, e não
    excluir propagaria `nan` para o `rho` inteiro.
    """
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError(
            f"os dois vetores têm de ter a mesma forma; recebi {first.shape} e {second.shape}"
        )

    both = _defined_in_both(first, second)
    used = int(both.sum())
    excluded = int(first.size - used)
    if used < MIN_POINTS_FOR_SPEARMAN:
        return Divergence(d=float("nan"), n_used=used, n_excluded=excluded, floor=float("nan"))

    d = 1.0 - float(stats.spearmanr(first[both], second[both]).statistic)
    # O piso sai do mesmo subconjunto: medido sobre o vetor inteiro, seria de
    # outra amostra.
    floor = 1.0 - float(stats.spearmanr(second[both], second[both]).statistic)
    return Divergence(d=d, n_used=used, n_excluded=excluded, floor=floor)


def tolerance_from_dispersions(k: float, sd_python: float, sd_r: float) -> float:
    """`k * sqrt(sd_python² + sd_r²)`: a tolerância derivada das duas dispersões.

    A soma é quadrática porque as duas fontes são independentes (a semente do
    lado do Python, a ausência de `set.seed` no bloco SERRF do R), e para fontes
    independentes somam-se variâncias. A soma aritmética inflaria a tolerância em
    41,4% quando os dois desvios são parecidos.

    `k` é argumento, e não constante: ele é pré-registrado em
    `data/fixtures/r_serrf_divergence.json`, e uma constante local deixaria de
    acompanhar o arquivo.
    """
    if k < 1.0:
        raise ValueError(
            f"k >= 1 é exigido; recebi {k}. Abaixo de 1 o Python seria cobrado "
            "mais estrito que o próprio oráculo contra si mesmo."
        )
    if sd_python < 0.0 or sd_r < 0.0:
        raise ValueError(f"desvios são não negativos; recebi {sd_python} e {sd_r}")
    return float(k * np.sqrt(sd_python**2 + sd_r**2))

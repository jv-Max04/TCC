"""Quanto uma correção recupera a verdade conhecida de uma matriz sintética.

O erro de recuperação e o RSD dos QCs medem coisas diferentes: uma correção pode
aproximar a matriz da verdade e, ao mesmo tempo, aumentar a dispersão dos QCs. Por
isso os dois são reportados lado a lado, e nenhum substitui o outro.
"""

import numpy as np

from refino.domain.matrix import RefinoMatrix
from refino.domain.synthetic import RecoveryError

__all__ = ["recovery_error"]


def recovery_error(corrected: RefinoMatrix, ground_truth: np.ndarray) -> RecoveryError:
    """Erro relativo mediano e correlação com a verdade, mais o número de features comparadas.

    Compara só células em que a verdade é positiva e os dois lados são finitos: zero
    é ausência de detecção, e erro relativo com denominador zero não tem valor.

    `fraction_worsened` sai `nan` aqui. Ela exige a matriz não corrigida, e quem a
    calcula é `worsened_fraction`.
    """
    corrigida = np.asarray(corrected.values, dtype=np.float64)
    verdade = np.asarray(ground_truth, dtype=np.float64)
    if corrigida.shape != verdade.shape:
        raise ValueError(
            f"a matriz corrigida tem forma {corrigida.shape} e a verdade {verdade.shape}; "
            "comparar formas diferentes mediria alinhamento e não recuperação"
        )

    utilizavel = (verdade > 0) & np.isfinite(corrigida) & np.isfinite(verdade)
    if not utilizavel.any():
        return RecoveryError(
            median_relative_error=float("nan"),
            correlation_with_truth=float("nan"),
            fraction_worsened=float("nan"),
            n_features_compared=0,
        )

    relativo = np.full(verdade.shape, np.nan)
    relativo[utilizavel] = np.abs(corrigida[utilizavel] - verdade[utilizavel]) / verdade[utilizavel]

    colunas = np.flatnonzero(utilizavel.any(axis=0))
    return RecoveryError(
        median_relative_error=float(np.nanmedian(relativo)),
        correlation_with_truth=_correlacao(corrigida, verdade, utilizavel),
        fraction_worsened=float("nan"),
        n_features_compared=int(colunas.size),
    )


def _correlacao(corrigida: np.ndarray, verdade: np.ndarray, utilizavel: np.ndarray) -> float:
    """Pearson em `log10`, sobre as células utilizáveis.

    Em `log10` porque a intensidade de LC-MS varia por ordens de grandeza, e no espaço
    linear a correlação seria dominada pelas features mais abundantes.
    """
    a = np.log10(corrigida[utilizavel])
    b = np.log10(verdade[utilizavel])
    finitos = np.isfinite(a) & np.isfinite(b)
    if finitos.sum() < 2 or a[finitos].std() == 0 or b[finitos].std() == 0:
        return float("nan")
    return float(np.corrcoef(a[finitos], b[finitos])[0, 1])


def worsened_fraction(
    corrected: RefinoMatrix, uncorrected: RefinoMatrix, ground_truth: np.ndarray
) -> float:
    """Fração de features cujo erro absoluto mediano aumentou com a correção.

    Separada de `recovery_error` porque exige também a matriz não corrigida. Feature
    com menos de duas células comparáveis fica fora do denominador.
    """
    verdade = np.asarray(ground_truth, dtype=np.float64)
    antes = np.asarray(uncorrected.values, dtype=np.float64)
    depois = np.asarray(corrected.values, dtype=np.float64)
    utilizavel = (verdade > 0) & np.isfinite(antes) & np.isfinite(depois)

    piorou = 0
    total = 0
    for coluna in range(verdade.shape[1]):
        mascara = utilizavel[:, coluna]
        if mascara.sum() < 2:
            continue
        total += 1
        erro_antes = np.median(np.abs(antes[mascara, coluna] - verdade[mascara, coluna]))
        erro_depois = np.median(np.abs(depois[mascara, coluna] - verdade[mascara, coluna]))
        piorou += erro_depois > erro_antes
    return piorou / total if total else float("nan")

"""Contratos do gerador sintético: `SyntheticSpec`, `SyntheticDataset` e `RecoveryError`.

Três escolhas estão no nome dos campos:
- `drift_to_noise_ratio`, e não uma magnitude solta: a amplitude da deriva só significa algo em
  relação à variação de fundo, e a unidade vai no nome.
- `differential_fraction`, e não uma contagem: uma contagem fixa sobre um total que varia muda
  de significado a cada matriz.
- `imitated_mode`: as propriedades reais variam demais entre os modos para haver um alvo único.
  O RSD mediano de QC vai de 15,50% no `neg_c18` a 143,81% no `pos_hilic`, e a razão de
  dispersão de 0,2609 a 1,1728. Uma matriz 'realista' precisa dizer qual modo imita.
"""

from dataclasses import dataclass

import numpy as np

from refino.domain.labels import ImitatedMode, InjectableDefect, validate_label
from refino.domain.matrix import RefinoMatrix

__all__ = ["RecoveryError", "SyntheticDataset", "SyntheticSpec"]

# `InjectableDefect` e `ImitatedMode` vivem em `domain/labels.py`, com o resto do vocabulário, e
# não aqui: `DefectType` já está lá, e redefinir os valores neste módulo criaria dois conjuntos
# para a mesma pergunta. Um teste de paridade afirma a relação de subconjunto.


@dataclass(frozen=True)
class SyntheticSpec:
    """A especificação de uma matriz sintética.

    `imitated_mode` e `drift_to_noise_ratio` não têm valor padrão: não existe matriz realista
    sem dizer qual modo ela imita, e um padrão para a razão seria um número inventado valendo em
    silêncio.
    """

    imitated_mode: ImitatedMode
    drift_to_noise_ratio: float
    defect_type: InjectableDefect = "gradual_drift"
    n_samples: int = 68
    n_features: int = 2000
    n_groups: int = 4
    qc_every: int = 11
    n_batches: int = 1
    changepoint_position: float | None = None
    zero_fraction: float = 0.15
    mnar_fraction: float = 0.8
    differential_fraction: float = 0.025
    effect_size: float = 1.0
    n_feature_blocks: int = 20
    seed: int = 42

    def __post_init__(self) -> None:
        """Valida rótulos e intervalos em runtime, porque `Literal` só é conferido pelo mypy."""
        validate_label("InjectableDefect", self.defect_type)
        validate_label("ImitatedMode", self.imitated_mode)

        if self.drift_to_noise_ratio < 0.0:
            raise ValueError(
                f"drift_to_noise_ratio não pode ser negativo; recebi {self.drift_to_noise_ratio}. "
                "É uma razão entre a amplitude da deriva e o desvio residual por feature"
            )
        if self.defect_type == "clean" and self.drift_to_noise_ratio != 0.0:
            raise ValueError(
                f"defect_type='clean' com drift_to_noise_ratio={self.drift_to_noise_ratio}: a "
                "combinação é contraditória, e aceitá-la produziria matriz com deriva rotulada "
                "como limpa — que é o pior insumo possível para conferir se o diagnóstico recupera "
                "o tipo de defeito"
            )
        for nome, valor in (
            ("zero_fraction", self.zero_fraction),
            ("mnar_fraction", self.mnar_fraction),
            ("differential_fraction", self.differential_fraction),
        ):
            if not 0.0 <= valor <= 1.0:
                raise ValueError(f"{nome} tem de estar em [0, 1]; recebi {valor}")
        if self.changepoint_position is not None and not 0.0 < self.changepoint_position < 1.0:
            raise ValueError(
                f"changepoint_position é fração da corrida e tem de estar em (0, 1); "
                f"recebi {self.changepoint_position}"
            )
        if self.defect_type == "step_change" and self.changepoint_position is None:
            raise ValueError(
                "defect_type='step_change' exige changepoint_position: sem ela o salto cairia num "
                "lugar arbitrário e não haveria posição a conferir na trajetória de QC"
            )
        for nome, valor in (
            ("n_samples", self.n_samples),
            ("n_features", self.n_features),
            ("n_groups", self.n_groups),
            ("qc_every", self.qc_every),
            ("n_feature_blocks", self.n_feature_blocks),
        ):
            if valor < 1:
                raise ValueError(f"{nome} tem de ser positivo; recebi {valor}")
        if self.n_feature_blocks > self.n_features:
            raise ValueError(
                f"n_feature_blocks={self.n_feature_blocks} excede n_features={self.n_features}"
            )


@dataclass(frozen=True)
class SyntheticDataset:
    """A matriz degradada, a verdade e a razão deriva/ruído efetivamente medida.

    `spec.drift_to_noise_ratio` é a razão pedida; `effective_drift_to_noise` é a medida no
    resultado. Se as duas divergem, o defeito é do gerador, e sem o campo não haveria como saber,
    porque a matriz sai plausível de qualquer jeito.
    """

    matrix: RefinoMatrix
    ground_truth: np.ndarray
    true_differential: tuple[str, ...]
    applied_drift: np.ndarray
    effective_drift_to_noise: float
    spec: SyntheticSpec


@dataclass(frozen=True)
class RecoveryError:
    """As três medidas do erro de recuperação, como campos e não como `dict`.

    Chave de dicionário não tem contrato: o consumidor leria `resultado["erro"]` sem nada
    verificar que a chave existe ou que mede o que ele pensa.
    """

    median_relative_error: float
    correlation_with_truth: float
    fraction_worsened: float
    n_features_compared: int

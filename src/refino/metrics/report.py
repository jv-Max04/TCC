"""`build_quality_report`: agrega as métricas de qualidade num `QualityReport`.

Recebe os vetores de estimador fora da amostra como argumentos, nunca um
corretor. É o que mantém `metrics/` sem dependência de `correction/`: o estimador
fora da amostra precisa do corretor, e o corretor precisa de métrica para ser
avaliado, então quem tem o corretor calcula o vetor e o passa para cá.

Chamado sem os dois vetores, o relatório declara que não os calculou.
"""

from typing import TYPE_CHECKING

import numpy as np

from refino.domain.labels import MeasurementPoint, PreTransform, RsdEstimator, RsdVariant
from refino.domain.matrix import RefinoMatrix
from refino.domain.spec import REQUIRED_METRIC_FIELDS, QualityReport

if TYPE_CHECKING:  # pragma: no cover - só para o mypy; `metrics` não depende de `statistics`
    from refino.statistics.plsda import PLSDAResult
    from refino.statistics.plsda_multiclasse import MulticlassPLSDAResult
from refino.metrics.rsd import qc_rsd
from refino.metrics.structure import (
    STABILITY_REPLICATES,
    STABILITY_SEED,
    structural_metrics,
)

__all__ = ["build_quality_report"]

# Os dois pontos de corte de RSD, 20% e 30%, como fração.
DEFAULT_RSD_CUTOFFS = (0.20, 0.30)


def _median_or_nan(values: np.ndarray) -> float:
    """Mediana das definidas, ou `nan` quando não há nenhuma.

    `nan` quer dizer 'calculada e indefinida', que é o caso da matriz cujas
    features têm todas RSD indefinido. `None` diria que ninguém tentou.
    """
    finitas = values[np.isfinite(values)]
    return float(np.median(finitas)) if finitas.size else float("nan")


def _mean_or_nan(values: np.ndarray) -> float:
    finitas = values[np.isfinite(values)]
    return float(np.mean(finitas)) if finitas.size else float("nan")


def _fraction_under(values: np.ndarray, cutoff: float) -> float:
    """Fração das features com RSD definido abaixo de `cutoff`.

    O denominador são as definidas: feature com RSD indefinido não está abaixo
    nem acima do corte. O tamanho desse denominador vai no relatório, em
    `n_features_rsd_defined`, ao lado da contagem das indefinidas.
    """
    finitas = values[np.isfinite(values)]
    if finitas.size == 0:
        return float("nan")
    return float(np.mean(finitas < cutoff))


def _defined_count(values: np.ndarray) -> int:
    """Quantas features têm RSD definido: o denominador das agregações de RSD.

    Usa a mesma expressão de `_median_or_nan`, `_mean_or_nan` e `_fraction_under`.
    Não se deriva de outras contagens do relatório, que medem outros pontos do
    pipeline.
    """
    return int(values[np.isfinite(values)].size)


def build_quality_report(
    matrix: RefinoMatrix,
    *,
    measurement_point: MeasurementPoint,
    structural_pretransform: PreTransform,
    variant: RsdVariant = "plain",
    rsd_estimator: RsdEstimator = "in_sample",
    n_components: int = 2,
    stability_replicates: int = STABILITY_REPLICATES,
    stability_seed: int = STABILITY_SEED,
    rsd_cutoffs: tuple[float, float] = DEFAULT_RSD_CUTOFFS,
    qc_rsd_cv: np.ndarray | None = None,
    qc_rsd_r_cv: np.ndarray | None = None,
    n_significant_fdr: int | None = None,
    plsda: "PLSDAResult | None" = None,
    plsda_groups: tuple[str, str] | None = None,
    plsda_cv_folds: int | None = None,
    plsda_multiclass: "MulticlassPLSDAResult | None" = None,
) -> QualityReport:
    """Agrega as métricas de qualidade num `QualityReport`.

    `measurement_point` e `structural_pretransform` são os dois obrigatórios,
    mesmo sendo acoplados por `REQUIRED_PRETRANSFORM`. Derivar um do outro
    esconderia o erro de quem acha que mede pós-correção e passou a matriz
    pós-pipeline: o valor derivado concordaria com o engano.

    Os dois vetores fora da amostra são opcionais. Fornecê-los põe os nomes
    correspondentes em `computed`; omiti-los deixa os campos `None` e os nomes
    fora, que é como se registra 'não calculado'.

    `plsda` recebe o `PLSDAResult` inteiro, e não `q2` e `permutation_pvalue`
    soltos: o dataclass é congelado e tem todos os campos obrigatórios, então não
    há como entregar scores sem Q² e sem p de permutação. `plsda_groups` e
    `plsda_cv_folds` entram à parte porque não estão no `PLSDAResult`: são
    argumentos de quem chamou. Omitir os três deixa os campos do PLS-DA em `None`
    e fora de `computed`.

    `plsda_multiclass` recebe o `MulticlassPLSDAResult` inteiro pelo mesmo
    motivo. Ele não pede grupos nem dobras à parte: as classes modeladas e o
    regime de validação já estão no resultado.
    """
    under_20, under_30 = rsd_cutoffs
    rsd_values = qc_rsd(matrix, variant=variant)
    estrutural = structural_metrics(
        matrix,
        pretransform=structural_pretransform,
        n_components=n_components,
        stability_replicates=stability_replicates,
        stability_seed=stability_seed,
    )

    computed = set(REQUIRED_METRIC_FIELDS)
    if qc_rsd_cv is not None:
        computed.add("qc_rsd_cv_median")
    if qc_rsd_r_cv is not None:
        computed.add("qc_rsd_r_cv_median")
    # `is not None`, e não a verdade do valor: zero features significativas é
    # resultado calculado, e é o que estes dados dão.
    if n_significant_fdr is not None:
        computed.add("n_significant_fdr")
    if plsda is not None:
        # Os dois juntos: Q² sem o p de permutação seria o achado sem o teste
        # que diz se ele é ruído.
        computed.add("plsda_q2")
        computed.add("plsda_permutation_p")
    if plsda_multiclass is not None:
        # Os dois juntos, como acima. `FAMILIAS_DO_PLSDA`, que a guarda de
        # `QualityReport` percorre, reprova meio par em qualquer das famílias.
        computed.add("plsda_multiclass_q2")
        computed.add("plsda_multiclass_permutation_p")

    return QualityReport(
        measurement_point=measurement_point,
        rsd_variant=variant,
        rsd_estimator=rsd_estimator,
        structural_pretransform=structural_pretransform,
        computed=frozenset(computed),
        qc_rsd_median=_median_or_nan(rsd_values),
        qc_rsd_mean=_mean_or_nan(rsd_values),
        frac_features_under_20=_fraction_under(rsd_values, under_20),
        frac_features_under_30=_fraction_under(rsd_values, under_30),
        n_features_rsd_undefined=int((~np.isfinite(rsd_values)).sum()),
        n_features_rsd_defined=_defined_count(rsd_values),
        qc_dispersion_ratio=estrutural.qc_dispersion_ratio,
        pc1_injection_correlation_abs=estrutural.pc1_injection_correlation_abs,
        structural_features_used=estrutural.n_features_used,
        structural_features_dropped_zero_variance=(estrutural.n_features_dropped_zero_variance),
        structural_features_dropped_non_finite=estrutural.n_features_dropped_non_finite,
        structural_column_digest=estrutural.retained_column_digest,
        qc_dispersion_ratio_spread=estrutural.qc_dispersion_ratio_spread,
        pc1_injection_correlation_spread=estrutural.pc1_injection_correlation_spread,
        stability_replicates=estrutural.stability_replicates,
        stability_seed=estrutural.stability_seed,
        qc_rsd_cv_median=None if qc_rsd_cv is None else _median_or_nan(qc_rsd_cv),
        qc_rsd_r_cv_median=None if qc_rsd_r_cv is None else _median_or_nan(qc_rsd_r_cv),
        n_significant_fdr=n_significant_fdr,
        # Os campos do PLS-DA saem todos da mesma condição, para que não haja
        # forma de preencher só parte deles.
        plsda_q2=None if plsda is None else plsda.q2,
        plsda_permutation_p=None if plsda is None else plsda.permutation_pvalue,
        plsda_groups=None if plsda is None else plsda_groups,
        plsda_r2y=None if plsda is None else plsda.r2y,
        plsda_n_components=None if plsda is None else plsda.n_components,
        plsda_cv_folds=None if plsda is None else plsda_cv_folds,
        plsda_n_permutations=None if plsda is None else plsda.n_permutations,
        plsda_seed=None if plsda is None else plsda.seed,
        # O mesmo vale para os campos do multiclasse.
        plsda_multiclass_q2=None if plsda_multiclass is None else plsda_multiclass.q2,
        plsda_multiclass_permutation_p=(
            None if plsda_multiclass is None else plsda_multiclass.permutation_pvalue
        ),
        plsda_multiclass_groups=None if plsda_multiclass is None else plsda_multiclass.groups,
        plsda_multiclass_n_components=(
            None if plsda_multiclass is None else plsda_multiclass.n_components
        ),
        plsda_multiclass_cv_folds=None if plsda_multiclass is None else plsda_multiclass.cv_folds,
        plsda_multiclass_n_permutations=(
            None if plsda_multiclass is None else plsda_multiclass.n_permutations
        ),
        plsda_multiclass_seed=None if plsda_multiclass is None else plsda_multiclass.seed,
    )

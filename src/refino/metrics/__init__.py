"""Métricas de qualidade, medidas sem corrigir.

Nenhuma função aqui reduz qualidade a um número: `build_quality_report` devolve o
conjunto completo, com os rótulos de como mediu, porque RSD e agrupamento de QC no
PCA discordam e elegem métodos diferentes.

O que o consumidor precisa saber antes de usar:

- RSD in-sample não compara métodos de correção: o LOESS o zera por construção.
  Comparar métodos exige estimador fora da amostra.
- `pretransform` é obrigatório nas métricas estruturais. Não há padrão correto,
  porque ele depende de onde a medição acontece.
- Toda agregação de RSD usa como denominador as features com RSD definido, e a
  contagem das indefinidas vai no relatório.

`metrics/` não conhece `correction/`: os estimadores fora da amostra chegam como
vetores, nunca como corretor.
"""

from refino.metrics.agreement import (
    Divergence,
    self_divergence_floor,
    spearman_divergence,
    tolerance_from_dispersions,
)
from refino.metrics.report import build_quality_report
from refino.metrics.rsd import (
    VARIANT_IMPLEMENTATIONS,
    boxplot_outlier_mask,
    qc_rsd,
    rsd,
    tukey_fivenum,
)
from refino.metrics.structure import (
    PRETRANSFORM_IMPLEMENTATIONS,
    StructuralMetrics,
    pc1_injection_correlation,
    qc_dispersion_ratio,
    qc_pc1_trajectory,
    retained_column_digest,
    structural_metrics,
)

__all__ = [
    "PRETRANSFORM_IMPLEMENTATIONS",
    "VARIANT_IMPLEMENTATIONS",
    "Divergence",
    "StructuralMetrics",
    "boxplot_outlier_mask",
    "build_quality_report",
    "pc1_injection_correlation",
    "qc_dispersion_ratio",
    "qc_pc1_trajectory",
    "qc_rsd",
    "retained_column_digest",
    "rsd",
    "self_divergence_floor",
    "spearman_divergence",
    "structural_metrics",
    "tolerance_from_dispersions",
    "tukey_fivenum",
]

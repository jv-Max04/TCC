"""`DefectProfile`, o resultado do diagnóstico de uma matriz.

Fica em `domain/` porque é entidade do modelo; `diagnosis/` guarda as funções que a
produzem.
"""

from dataclasses import dataclass, field

from refino.domain.labels import DefectType, validate_label

__all__ = ["DefectProfile"]


@dataclass(frozen=True)
class DefectProfile:
    """Diagnóstico de uma matriz.

    `defect_type` usa `DefectType`, de `domain/labels.py`, em vez de repetir o `Literal`
    aqui. A lista tem cinco valores (os quatro tipos de defeito mais `unassessable`) e
    uma segunda cópia sairia de sincronia.

    `confidence` é derivada, não calibrada: não é probabilidade. É a distância relativa
    do valor decisor ao limiar que classificou, saturada em 1, e vale 0,0 em
    `unassessable`, onde nenhum limiar é consultado.
    """

    defect_type: DefectType
    confidence: float
    suggested_batches: tuple[int, ...]
    changepoints: tuple[int, ...]
    qc_rsd_median: float
    qc_rsd_p75: float
    zero_fraction: float
    missing_fraction: float
    pc1_injection_correlation: float
    qc_dispersion_ratio: float
    n_features: int
    n_samples: int
    n_qcs: int
    raw_features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Valida o rótulo em runtime.

        O mypy só confere `Literal` nos pontos de chamada internos, e este objeto também
        é reconstruído a partir do banco. Sem a guarda, um valor inválido cairia num
        ramo qualquer e se comportaria como outra classe.
        """
        validate_label("DefectType", self.defect_type)

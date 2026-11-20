"""Metadados de uma injeção: `SampleMetadata`."""

from dataclasses import dataclass
from typing import Literal

SampleType = Literal["qc", "sample", "validate"]

# Os três literais em forma verificável em runtime. O mypy só prova os pontos de
# chamada internos; dado lido de disco não passa por ele. Um `sampleType` inválido
# daria uma amostra fora de `qc_mask`, `sample_mask` e `validate_mask`, que some
# em silêncio da estatística de QC e da de amostras.
SAMPLE_TYPES: frozenset[str] = frozenset({"qc", "sample", "validate"})

# Rótulo que a linha `LABEL` usa para controles de qualidade.
QC_LABEL = "QC"


@dataclass(frozen=True)
class SampleMetadata:
    """Metadados de uma injeção. Imutável.

    `injection_order` é 1-based e único dentro da matriz. É a posição temporal da
    amostra na corrida e só existe no `Sample_List`. Sem ela não há correção de
    deriva possível.

    `sample_type` admite `"validate"`, que o código R suporta e os dados do
    laboratório não usam.
    """

    name: str
    injection_order: int
    sample_type: SampleType
    group: str
    batch: str = "A"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("nome de amostra vazio")
        if self.injection_order < 1:
            raise ValueError(
                f"injection_order é 1-based; recebi {self.injection_order} para {self.name!r}"
            )
        if not self.batch:
            raise ValueError(f"batch vazio para {self.name!r}")
        if self.sample_type not in SAMPLE_TYPES:
            raise ValueError(
                f"sample_type {self.sample_type!r} inválido para {self.name!r}; "
                f"esperava um de {sorted(SAMPLE_TYPES)}. Valor desconhecido produziria "
                "amostra invisível às três máscaras, desaparecendo em silêncio da "
                "estatística de QC e da de amostras."
            )

    @property
    def is_qc(self) -> bool:
        """Controle de qualidade, a régua do sistema."""
        return self.sample_type == "qc"

    @property
    def is_study_sample(self) -> bool:
        """Amostra biológica do estudo.

        `validate` não é amostra de estudo nem QC. É um terceiro tipo, usado para
        medir a correção fora da amostra de treino, e tratá-lo como `sample`
        contaminaria a estatística a jusante.
        """
        return self.sample_type == "sample"

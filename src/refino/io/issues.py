"""`ValidationIssue` e o catálogo de códigos de validação.

Módulo próprio, e não parte de `validators.py`, porque `sniffer.py` também emite
issue e o validador recebe o dialeto que o sniffer produz: importar o validador
criaria ciclo.

Os códigos se dividem por escopo, cada um com a sua tupla:

- documento: o que um par de arquivos sozinho permite concluir, em
  `FILE_SCOPE_CODES`.
- métrica: exige uma métrica calculada, em `METRIC_SCOPE_CODES`.
- declaração: compara um parâmetro do usuário com o arquivo, em
  `DECLARATION_SCOPE_CODES`.
- estudo: exige ver os quatro modos juntos, em `STUDY_SCOPE_CODES`, emitidos por
  `validate_study`.
"""

from dataclasses import dataclass
from typing import Literal

Severity = Literal["error", "warning", "info"]

# As três severidades, para conferir em tempo de execução.
SEVERITIES: frozenset[str] = frozenset({"error", "warning", "info"})


@dataclass(frozen=True)
class ValidationIssue:
    """Um problema encontrado na validação.

    `severity` decide se a ingestão aborta: qualquer `error` a interrompe, com a
    lista completa de problemas e não só o primeiro. Por isso coletar e abortar
    são passos distintos: `validate_structure` só coleta, e `load_matrix`
    decide.

    `location` é uma referência legível ao lugar do defeito: nome de amostra,
    nome de feature ou índice. É `None` quando o problema é da matriz como um
    todo, como `HIGH_ZERO_FRACTION`.
    """

    severity: Severity
    code: str
    message: str
    location: str | None = None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("ValidationIssue exige um código")
        if not self.message:
            raise ValueError(f"ValidationIssue {self.code} exige mensagem")
        if self.severity not in SEVERITIES:
            raise ValueError(
                f"severity {self.severity!r} inválida em {self.code}; esperava uma de "
                f"{sorted(SEVERITIES)}. Valor desconhecido não seria `error` e a "
                "ingestão prosseguiria sobre dado que deveria abortá-la."
            )

    @property
    def is_error(self) -> bool:
        """Aborta a ingestão."""
        return self.severity == "error"

    def __str__(self) -> str:
        where = f" [{self.location}]" if self.location else ""
        return f"{self.severity.upper()} {self.code}{where}: {self.message}"


# Códigos de escopo documento.
#
# "Documento" e não "arquivo" porque `DUPLICATE_INJECTION_TIME` vale para os dois
# formatos que carregam ordem de injeção: o par measurements/Sample_List, onde
# `validate_structure` o decide, e o `toserrf`, onde `read_toserrf` o decide a
# partir da linha `time`.
#
# IMPLAUSIBLE_INTENSITY_RANGE é conhecimento de domínio e por isso pertence a
# `validate_structure`, não ao sniffer: a camada léxica não sabe o que é uma
# intensidade de LC-MS.
FILE_SCOPE_CODES: tuple[str, ...] = (
    "MISSING_NAME_ROW",
    "MISSING_LABEL_ROW",
    "RAGGED_ROWS",
    "DUPLICATE_SAMPLE_NAME",
    "DUPLICATE_FEATURE_NAME",
    "SAMPLE_LIST_MISMATCH",
    "NO_QC_SAMPLES",
    "INSUFFICIENT_QCS",
    "NON_NUMERIC_CELL",
    "NEGATIVE_INTENSITY",
    "ALL_ZERO_FEATURE",
    "HIGH_ZERO_FRACTION",
    "THOUSANDS_SEPARATOR_CORRUPTION",
    "INJECTION_ORDER_GAP",
    "GROUP_LABEL_VARIANT",
    "LABEL_NAME_MISMATCH",
    "IMPLAUSIBLE_INTENSITY_RANGE",
    "DUPLICATE_INJECTION_TIME",
)

# Códigos de escopo métrica.
#
# `UNDEFINED_QC_RSD` depende de `qc_rsd`, que a validação estrutural não calcula.
# Quem o emite é `filter_by_qc_rsd`, que precisa distinguir "RSD acima do corte"
# de "RSD indefinido" ao contar descartes.
METRIC_SCOPE_CODES: tuple[str, ...] = ("UNDEFINED_QC_RSD",)

# Códigos de escopo declaração: comparam um parâmetro do usuário com o arquivo.
#
# Os de `FILE_SCOPE_CODES` perguntam se o arquivo é coerente consigo mesmo; este
# pergunta se o que o usuário declarou casa com o que o arquivo mostra. Fica fora
# daquela tupla porque os códigos dela são injetáveis pelas factories de teste, e
# este exige um parâmetro externo.
#
# A severidade é `warning` e nunca `error`: o padrão dos nomes de amostra é
# observação sobre os arquivos deste laboratório, não invariante do domínio.
DECLARATION_SCOPE_CODES: tuple[str, ...] = ("MODE_CONTENT_MISMATCH",)

# Códigos de escopo estudo: exigem ver as quatro matrizes.
STUDY_SCOPE_CODES: tuple[str, ...] = (
    "GROUP_LABEL_INCONSISTENT_ACROSS_MODES",
    "PATIENT_GROUP_CONFLICT",
)

# Severidade fixa por código. `LABEL_NAME_MISMATCH` está ausente de propósito:
# a severidade dele depende de parâmetro (`on_label_mismatch`).
FIXED_SEVERITY: dict[str, Severity] = {
    "MISSING_NAME_ROW": "error",
    "MISSING_LABEL_ROW": "error",
    "RAGGED_ROWS": "error",
    "DUPLICATE_SAMPLE_NAME": "error",
    "DUPLICATE_FEATURE_NAME": "error",
    "SAMPLE_LIST_MISMATCH": "error",
    "NO_QC_SAMPLES": "error",
    "INSUFFICIENT_QCS": "warning",
    "NON_NUMERIC_CELL": "error",
    "NEGATIVE_INTENSITY": "warning",
    "ALL_ZERO_FEATURE": "info",
    "HIGH_ZERO_FRACTION": "warning",
    "THOUSANDS_SEPARATOR_CORRUPTION": "error",
    "INJECTION_ORDER_GAP": "warning",
    "GROUP_LABEL_VARIANT": "warning",
    "IMPLAUSIBLE_INTENSITY_RANGE": "warning",
    "DUPLICATE_INJECTION_TIME": "error",
    "GROUP_LABEL_INCONSISTENT_ACROSS_MODES": "warning",
    "PATIENT_GROUP_CONFLICT": "error",
    "UNDEFINED_QC_RSD": "info",
}

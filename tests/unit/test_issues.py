"""`ValidationIssue` e o catálogo de códigos.

Os caminhos descobertos aqui eram guardas de
construção e formatação de mensagem, e caminho de erro é exatamente onde o teste
importa. Nenhum `pragma` seria justificável.
"""

from __future__ import annotations

import pytest

from refino.io.issues import (
    FILE_SCOPE_CODES,
    FIXED_SEVERITY,
    STUDY_SCOPE_CODES,
    ValidationIssue,
)


def test_issue_requires_a_code() -> None:
    with pytest.raises(ValueError, match="exige um código"):
        ValidationIssue(severity="error", code="", message="algo")


def test_issue_requires_a_message() -> None:
    """Código sem mensagem é inútil para quem recebe a lista de problemas."""
    with pytest.raises(ValueError, match="exige mensagem"):
        ValidationIssue(severity="error", code="RAGGED_ROWS", message="")


def test_is_error_is_true_only_for_error_severity() -> None:
    """É a propriedade que `load_matrix` consulta para decidir abortar."""
    assert ValidationIssue("error", "X", "m").is_error
    assert not ValidationIssue("warning", "X", "m").is_error
    assert not ValidationIssue("info", "X", "m").is_error


def test_str_includes_severity_code_and_message() -> None:
    text = str(ValidationIssue("warning", "INSUFFICIENT_QCS", "só 2 QCs"))
    assert "WARNING" in text
    assert "INSUFFICIENT_QCS" in text
    assert "só 2 QCs" in text


def test_str_includes_location_when_present() -> None:
    text = str(ValidationIssue("error", "DUPLICATE_SAMPLE_NAME", "repetido", location="TINY_QC1"))
    assert "[TINY_QC1]" in text


def test_str_omits_location_brackets_when_absent() -> None:
    """`HIGH_ZERO_FRACTION` é da matriz como um todo — não tem lugar."""
    text = str(ValidationIssue("warning", "HIGH_ZERO_FRACTION", "40% de zeros"))
    assert "[" not in text


def test_issue_is_immutable() -> None:
    import dataclasses

    issue = ValidationIssue("error", "X", "m")
    with pytest.raises(dataclasses.FrozenInstanceError):
        issue.severity = "info"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# O catálogo
# ---------------------------------------------------------------------------


def test_there_are_eighteen_document_scope_codes() -> None:
    """18: os 16, mais IMPLAUSIBLE_INTENSITY_RANGE e
    DUPLICATE_INJECTION_TIME."""
    assert len(FILE_SCOPE_CODES) == 18
    assert len(set(FILE_SCOPE_CODES)) == 18


def test_duplicate_time_is_error_and_gap_is_warning() -> None:
    """A consequência é diferente, então a severidade também.

    Lacuna ainda permite ordenar; duplicata torna a ordem ambígua e nenhum
    método batch-aware pode modelar deriva sobre ela.
    """
    assert FIXED_SEVERITY["DUPLICATE_INJECTION_TIME"] == "error"
    assert FIXED_SEVERITY["INJECTION_ORDER_GAP"] == "warning"


def test_toserrf_group_unrecoverable_is_gone() -> None:
    """Não era defeito do arquivo, era argumento insuficiente do chamador.

    Virou `MissingGroupsError`, levantado por `to_matrix`. Este teste impede que
    volte como código de validação numa refatoração distraída.
    """
    assert "TOSERRF_GROUP_UNRECOVERABLE" not in FILE_SCOPE_CODES
    assert "TOSERRF_GROUP_UNRECOVERABLE" not in STUDY_SCOPE_CODES


def test_there_are_two_study_scope_codes() -> None:
    assert set(STUDY_SCOPE_CODES) == {
        "GROUP_LABEL_INCONSISTENT_ACROSS_MODES",
        "PATIENT_GROUP_CONFLICT",
    }


def test_the_two_scopes_do_not_overlap() -> None:
    assert not set(FILE_SCOPE_CODES) & set(STUDY_SCOPE_CODES)


def test_label_name_mismatch_has_no_fixed_severity() -> None:
    """É o único cujo `severity` depende de parâmetro (`on_label_mismatch`).

    Fixá-lo no catálogo esconderia essa dependência — e ela é o mecanismo pelo
    qual `prefer_name` é o padrão sem perder a opção estrita.
    """
    assert "LABEL_NAME_MISMATCH" not in FIXED_SEVERITY


def test_every_other_code_has_a_fixed_severity() -> None:
    """Nenhum código fica sem severidade declarada por esquecimento."""
    declared = set(FILE_SCOPE_CODES) | set(STUDY_SCOPE_CODES)
    missing = declared - set(FIXED_SEVERITY) - {"LABEL_NAME_MISMATCH"}
    assert not missing, f"códigos sem severidade fixa: {sorted(missing)}"


def test_patient_group_conflict_is_the_only_new_error_in_study_scope() -> None:
    """Um paciente com duas classes clínicas invalida a análise a jusante."""
    assert FIXED_SEVERITY["PATIENT_GROUP_CONFLICT"] == "error"
    assert FIXED_SEVERITY["GROUP_LABEL_INCONSISTENT_ACROSS_MODES"] == "warning"


@pytest.mark.parametrize(
    ("code", "severity"),
    [
        ("MISSING_NAME_ROW", "error"),
        ("INSUFFICIENT_QCS", "warning"),
        ("ALL_ZERO_FEATURE", "info"),
        ("GROUP_LABEL_VARIANT", "warning"),
        ("THOUSANDS_SEPARATOR_CORRUPTION", "error"),
    ],
)
def test_severities_match_the_specification_table(code: str, severity: str) -> None:
    """Amostra da tabela de severidades, incluindo os três níveis."""
    assert FIXED_SEVERITY[code] == severity


def test_fixed_severity_declares_no_unknown_code() -> None:
    """O catálogo não tem entrada órfã — sinal de código renomeado pela metade."""
    from refino.io.issues import METRIC_SCOPE_CODES

    known = set(FILE_SCOPE_CODES) | set(STUDY_SCOPE_CODES) | set(METRIC_SCOPE_CODES)
    assert not set(FIXED_SEVERITY) - known


def test_undefined_qc_rsd_is_metric_scope_not_file_scope() -> None:
    """Depende de `qc_rsd`, que é das métricas.

    Pôr este código em FILE_SCOPE_CODES faria o teste de exaustividade da
    partição exigir que `validate_structure` o emitisse — e a ingestão não sabe
    calcular RSD. O dono é `filter_by_qc_rsd`, no pós-correção.
    """
    from refino.io.issues import METRIC_SCOPE_CODES

    assert METRIC_SCOPE_CODES == ("UNDEFINED_QC_RSD",)
    assert "UNDEFINED_QC_RSD" not in FILE_SCOPE_CODES
    assert "UNDEFINED_QC_RSD" not in STUDY_SCOPE_CODES
    assert FIXED_SEVERITY["UNDEFINED_QC_RSD"] == "info"

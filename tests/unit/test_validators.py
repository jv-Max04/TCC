"""Validação de escopo arquivo.

**A partição dos 17 códigos.** O plano tratava os validadores e os leitores como se a
validação fosse independente da leitura, e não é: quatro códigos dependem da
estrutura de linhas ou do texto de cabeçalho, e nenhum é derivável de um
`MeasurementsDocument` já parseado. Este arquivo cobre os 12 que
`validate_structure` decide sozinha, e afirma que a partição é exaustiva.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from refino.io.documents import MeasurementsDocument
from refino.io.issues import FILE_SCOPE_CODES
from refino.io.sniffer import Dialect
from refino.io.validators import (
    CANONICAL_GROUPS,
    canonical_group,
    group_from_name,
    validate_structure,
)
from tests.factories import MatrixSpec, base_spec, inject, inject_both_label_codes

# Os 12 que `validate_structure` decide a partir do documento parseado.
OWNED_BY_VALIDATE_STRUCTURE = frozenset(
    {
        "DUPLICATE_SAMPLE_NAME",
        "DUPLICATE_FEATURE_NAME",
        "SAMPLE_LIST_MISMATCH",
        "NO_QC_SAMPLES",
        "INSUFFICIENT_QCS",
        "NEGATIVE_INTENSITY",
        "ALL_ZERO_FEATURE",
        "HIGH_ZERO_FRACTION",
        "INJECTION_ORDER_GAP",
        "GROUP_LABEL_VARIANT",
        "LABEL_NAME_MISMATCH",
        "IMPLAUSIBLE_INTENSITY_RANGE",
        "DUPLICATE_INJECTION_TIME",
    }
)

# Dependem da estrutura de linhas ou do texto de cabeçalho: emitidos pelo leitor
# e incorporados via `doc.issues`. Cobertos quando os leitores existirem.
OWNED_BY_READER = frozenset(
    {"MISSING_NAME_ROW", "MISSING_LABEL_ROW", "RAGGED_ROWS", "NON_NUMERIC_CELL"}
)

# Emitido pelo sniffer no passo 3 e incorporado via `doc.dialect.issues`.
OWNED_BY_SNIFFER = frozenset({"THOUSANDS_SEPARATOR_CORRUPTION"})

_DIALECT = Dialect(
    delimiter=",", decimal=".", thousands=None, encoding="utf-8", line_terminator="\r\n"
)


def _document(spec: MatrixSpec) -> MeasurementsDocument:
    """Converte um `MatrixSpec` da factory num documento já parseado.

    Atalho deliberado: não passa pelo leitor, porque o leitor é outra etapa. Os códigos
    que exigiriam o leitor estão em OWNED_BY_READER e não são testados aqui.
    """
    return MeasurementsDocument(
        values=np.array(spec.values, dtype=np.float64),
        feature_names=spec.feature_names,
        sample_names=tuple(s.name for s in spec.samples),
        group_labels=tuple(s.label for s in spec.samples),
        raw_cells=tuple(tuple(str(v) for v in row) for row in spec.values),
        dialect=_DIALECT,
    )


def _orders(spec: MatrixSpec) -> Mapping[str, int]:
    return {
        s.name: spec.injection_order_override.get(s.name, s.injection_order)
        for s in spec.samples
        if s.name not in spec.drop_from_sample_list
    }


def _codes(spec: MatrixSpec) -> list[str]:
    return [i.code for i in validate_structure(_document(spec), _orders(spec))]


# ---------------------------------------------------------------------------
# A partição é exaustiva
# ---------------------------------------------------------------------------


def test_the_three_owners_partition_all_document_scope_codes() -> None:
    """Nenhum código fica sem dono, e nenhum tem dois.

    A partição cobre o caminho de ingestão measurements/Sample_List.
    `DUPLICATE_INJECTION_TIME` também é emitido por `read_toserrf` para o seu
    formato — caminho diferente, emissor próprio, testado à parte.

    Se alguém acrescentar um código à tabela sem decidir quem o emite, este teste
    reprova — que é melhor do que descobrir na integração que ninguém o emite.
    """
    partition = OWNED_BY_VALIDATE_STRUCTURE | OWNED_BY_READER | OWNED_BY_SNIFFER
    assert partition == set(FILE_SCOPE_CODES), (
        f"sem dono: {sorted(set(FILE_SCOPE_CODES) - partition)}; "
        f"dono sem código: {sorted(partition - set(FILE_SCOPE_CODES))}"
    )
    assert len(OWNED_BY_VALIDATE_STRUCTURE) + len(OWNED_BY_READER) + len(OWNED_BY_SNIFFER) == 18


# ---------------------------------------------------------------------------
# canonical_group
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label", ["HAM/TSP", "HAM-TSP", "HAMTSP", "ham tsp", "Ham_Tsp"])
def test_canonical_group_collapses_hamtsp_notations(label: str) -> None:
    assert canonical_group(label) == "hamtsp"


def test_canonical_group_does_not_fix_typos() -> None:
    """A distinção que separa variante de notação de dado errado."""
    assert canonical_group("Assyntomatic") != canonical_group("Asymptomatic")


@pytest.mark.parametrize("label", ["", "///", "   "])
def test_canonical_group_handles_degenerate_input(label: str) -> None:
    assert canonical_group(label) == ""


def test_group_from_name_extracts_the_suffix() -> None:
    assert group_from_name("HTLV_C18_PLASMA_MSe_NEG_0_ID99_Asymptomatic") == "Asymptomatic"


def test_group_from_name_is_none_for_qc_names() -> None:
    """Ausência de informação não é divergência — é por isso que QCs não disparam."""
    assert group_from_name("HTLV_C18_PLASMA_MSe_NEG_QC1") is None


# ---------------------------------------------------------------------------
# Os 12 códigos, um caso cada
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", sorted(OWNED_BY_VALIDATE_STRUCTURE))
def test_each_owned_code_fires_with_the_expected_severity(code: str) -> None:
    """Um caso por código, afirmando código E severidade."""
    from refino.io.issues import FIXED_SEVERITY

    issues = validate_structure(_document(inject(code)), _orders(inject(code)))
    matching = [i for i in issues if i.code == code]
    assert matching, f"{code} não disparou; códigos vistos: {sorted({i.code for i in issues})}"

    if code in FIXED_SEVERITY:
        assert matching[0].severity == FIXED_SEVERITY[code]
    else:
        # LABEL_NAME_MISMATCH: severidade depende de on_label_mismatch, e
        # validate_structure emite o caso base como warning.
        assert code == "LABEL_NAME_MISMATCH"
        assert matching[0].severity == "warning"


def test_clean_base_has_no_error_severity_issues() -> None:
    """A base da factory é limpa: nenhum `error`.

    Não "nenhum issue": a base pode legitimamente carregar `info`. O que ela não
    pode ter é problema que aborte a ingestão.
    """
    issues = validate_structure(_document(base_spec()), _orders(base_spec()))
    assert not [i for i in issues if i.is_error], [str(i) for i in issues if i.is_error]


def test_validate_structure_never_raises_on_ruined_input() -> None:
    """Coletar e abortar são passos distintos."""
    doc = MeasurementsDocument(
        values=np.zeros((0, 0), dtype=np.float64),
        feature_names=(),
        sample_names=(),
        group_labels=(),
        raw_cells=(),
        dialect=_DIALECT,
    )
    issues = validate_structure(doc)
    assert any(i.code == "NO_QC_SAMPLES" for i in issues)


def test_the_complete_list_is_returned_not_the_first_problem() -> None:
    """A lista completa de problemas, nunca apenas o primeiro"."""
    spec = inject("DUPLICATE_SAMPLE_NAME")
    values = [list(r) for r in spec.values]
    values[0][0] = -5.0  # NEGATIVE_INTENSITY, um segundo defeito independente
    spec = type(spec)(**{**spec.__dict__, "values": tuple(tuple(r) for r in values)})

    codes = set(_codes(spec))
    assert {"DUPLICATE_SAMPLE_NAME", "NEGATIVE_INTENSITY"} <= codes


def test_sample_list_mismatch_fires_in_both_directions() -> None:
    spec = base_spec()
    orders = dict(_orders(spec))
    orders["FANTASMA"] = 99  # na lista, fora da matriz
    del orders[spec.samples[0].name]  # na matriz, fora da lista

    issues = validate_structure(_document(spec), orders)
    messages = [i.message for i in issues if i.code == "SAMPLE_LIST_MISMATCH"]
    assert len(messages) == 2
    assert any("mas não na lista" in m for m in messages)
    assert any("mas não na matriz" in m for m in messages)


def test_injection_order_gap_is_not_reported_for_already_missing_samples() -> None:
    """Uma ausência já reportada não deve virar também aviso de lacuna.

    Sem esse cuidado, todo SAMPLE_LIST_MISMATCH arrastaria um INJECTION_ORDER_GAP
    espúrio, e a lista de problemas ficaria com ruído correlacionado.
    """
    spec = inject("SAMPLE_LIST_MISMATCH")
    codes = _codes(spec)
    assert "SAMPLE_LIST_MISMATCH" in codes
    assert "INJECTION_ORDER_GAP" not in codes


# ---------------------------------------------------------------------------
# Os dois códigos de rótulo
# ---------------------------------------------------------------------------


def test_a_sample_can_receive_both_label_codes() -> None:
    """A não-exclusividade que corrigiu o invariante.

    `LABEL` grafa `HAM-TSP` (variante de `HAMTSP`) e o nome diz `ATLL`: a amostra
    é variante **e** divergente do nome. Dois defeitos distintos, e nenhum teste
    deve afirmar exclusividade.
    """
    spec = inject_both_label_codes()
    issues = validate_structure(_document(spec), _orders(spec))

    by_sample: dict[str, set[str]] = {}
    for issue in issues:
        if issue.location and issue.code in {"GROUP_LABEL_VARIANT", "LABEL_NAME_MISMATCH"}:
            by_sample.setdefault(issue.location, set()).add(issue.code)

    both = [s for s, codes in by_sample.items() if len(codes) == 2]
    assert both, f"esperava amostra com os dois códigos; vi {by_sample}"


def test_group_label_variant_compares_against_the_canonical_set_not_the_name() -> None:
    """O código não consulta o nome da amostra.

    Rótulo em caixa baixa dispara mesmo quando o nome concorda com ele, porque a
    comparação é contra `CANONICAL_GROUPS`.
    """
    spec = base_spec()
    samples = tuple(
        type(s)(name="TINY_9_ID9_hamtsp", label="hamtsp", injection_order=s.injection_order)
        if s.label == "HAMTSP"
        else s
        for s in spec.samples
    )
    spec = type(spec)(**{**spec.__dict__, "samples": samples})

    codes = _codes(spec)
    assert "GROUP_LABEL_VARIANT" in codes
    assert "LABEL_NAME_MISMATCH" not in codes, "nome e LABEL concordam"


def test_qc_samples_receive_neither_label_code() -> None:
    """Os QCs não expõem grupo no nome e têm LABEL próprio."""
    spec = base_spec()
    issues = validate_structure(_document(spec), _orders(spec))
    qc_names = {s.name for s in spec.samples if s.label == "QC"}
    assert not [
        i
        for i in issues
        if i.location in qc_names and i.code in {"GROUP_LABEL_VARIANT", "LABEL_NAME_MISMATCH"}
    ]


def test_canonical_groups_is_a_parameter() -> None:
    """Outro estudo redeclara os grupos sem editar o validador."""
    spec = base_spec()
    issues = validate_structure(
        _document(spec), _orders(spec), canonical_groups=frozenset({"OUTRO"})
    )
    # Nenhum rótulo colapsa em "OUTRO", então nenhuma variante é reconhecida.
    assert not [i for i in issues if i.code == "GROUP_LABEL_VARIANT"]


def test_canonical_groups_default_matches_the_specification() -> None:
    assert frozenset({"NoHTLV", "Asymptomatic", "HAMTSP", "ATLL"}) == CANONICAL_GROUPS


# ---------------------------------------------------------------------------
# IMPLAUSIBLE_INTENSITY_RANGE
# ---------------------------------------------------------------------------


def test_implausible_range_reports_both_readings_and_the_ratio() -> None:
    """Reporta as duas leituras e a razão, e NÃO decide."""
    spec = inject("IMPLAUSIBLE_INTENSITY_RANGE")
    issue = next(
        i
        for i in validate_structure(_document(spec), _orders(spec))
        if i.code == "IMPLAUSIBLE_INTENSITY_RANGE"
    )
    assert "Leitura alternativa" in issue.message
    assert "razão" in issue.message
    assert "é do analista" in issue.message
    assert issue.severity == "warning"


def test_implausible_range_names_the_detected_dialect() -> None:
    """O diagnóstico só é avaliável com o dialeto que produziu a leitura."""
    spec = inject("IMPLAUSIBLE_INTENSITY_RANGE")
    issue = next(
        i
        for i in validate_structure(_document(spec), _orders(spec))
        if i.code == "IMPLAUSIBLE_INTENSITY_RANGE"
    )
    assert "decimal" in issue.message
    assert "structural" in issue.message


def test_plausible_range_is_silent() -> None:
    spec = base_spec()
    assert "IMPLAUSIBLE_INTENSITY_RANGE" not in _codes(spec)


def test_all_zero_matrix_does_not_trip_the_range_check() -> None:
    """Sem valor não-nulo não há mediana a julgar — e não se inventa uma."""
    spec = base_spec()
    zeros = tuple(tuple(0.0 for _ in row) for row in spec.values)
    spec = type(spec)(**{**spec.__dict__, "values": zeros})
    assert "IMPLAUSIBLE_INTENSITY_RANGE" not in _codes(spec)

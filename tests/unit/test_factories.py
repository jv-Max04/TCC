"""Verifica a fixture factory e os nove fixtures.

**Verificação textual, deliberadamente.** `validate_structure` não existe ainda,
então nada aqui depende dele. Se dependesse, o critério viraria teste
de si mesmo: eu escreveria a factory para produzir o que o validador encontra e
então afirmaria que o validador encontra o que a factory produz. Estes testes
inspecionam o CSV emitido, não o resultado da validação.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pytest

from tests.factories import (
    CANONICAL_GROUPS,
    FILE_SCOPE_CODES,
    QC_LABEL,
    base_spec,
    inject,
    inject_both_label_codes,
    render_measurements,
    render_pair,
    render_sample_list,
    spec_without_group_in_name,
)
from tests.fixture_files import (
    DRIFT_PER_INJECTION,
    FIXTURE_BUILDERS,
    STEP_AFTER_INJECTION,
    STEP_FACTOR,
    ZERO_POSITIONS,
)

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO / "data" / "fixtures"

STUDY_SCOPE_CODES = ("GROUP_LABEL_INCONSISTENT_ACROSS_MODES", "PATIENT_GROUP_CONFLICT")


def _parse(text: str, delimiter: str = ",") -> list[list[str]]:
    return list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))


def _canon(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


# ---------------------------------------------------------------------------
# A base é válida
# ---------------------------------------------------------------------------


def test_base_spec_has_five_qcs_so_it_does_not_trip_insufficient_qcs() -> None:
    """A base da factory é limpa — 5 QCs, acima do corte de INSUFFICIENT_QCS.

    Se caísse para 2 como `tiny_clean`, todo caso injetado carregaria esse aviso
    de fundo e "injeta aquele defeito e nenhum outro" seria inafirmável.
    """
    spec = base_spec()
    assert sum(1 for s in spec.samples if s.label == QC_LABEL) == 5


def test_base_spec_labels_are_all_canonical() -> None:
    """Nenhum rótulo da base é variante nem desconhecido."""
    spec = base_spec()
    for sample in spec.samples:
        assert sample.label == QC_LABEL or sample.label in CANONICAL_GROUPS


def test_base_spec_label_agrees_with_the_group_in_the_name() -> None:
    """Base sem LABEL_NAME_MISMATCH: nome e LABEL concordam."""
    for sample in base_spec().samples:
        match = re.search(r"_ID\d+_(.+)$", sample.name)
        if match:
            assert _canon(match.group(1)) == _canon(sample.label)


def test_base_spec_injection_orders_are_contiguous() -> None:
    """Base sem INJECTION_ORDER_GAP."""
    spec = base_spec()
    orders = sorted(s.injection_order for s in spec.samples)
    assert orders == list(range(1, len(spec.samples) + 1))


def test_base_spec_has_no_zeros_and_no_negatives() -> None:
    """Base sem ALL_ZERO_FEATURE, HIGH_ZERO_FRACTION nem NEGATIVE_INTENSITY."""
    flat = [v for row in base_spec().values for v in row]
    assert all(v > 0 for v in flat)


def test_base_spec_renders_a_rectangular_matrix() -> None:
    """Base sem RAGGED_ROWS: toda linha tem a mesma contagem de campos."""
    rows = _parse(render_measurements(base_spec()))
    assert len({len(r) for r in rows}) == 1


def test_base_spec_header_cells_are_the_literals_the_spec_requires() -> None:
    """Base sem MISSING_NAME_ROW nem MISSING_LABEL_ROW."""
    rows = _parse(render_measurements(base_spec()))
    assert rows[0][0] == "NAME"
    assert rows[1][0] == "LABEL"


# ---------------------------------------------------------------------------
# Injeção — um caso por código de escopo arquivo
# ---------------------------------------------------------------------------


def test_there_are_exactly_eighteen_document_scope_codes() -> None:
    """18: mais IMPLAUSIBLE_INTENSITY_RANGE e DUPLICATE_INJECTION_TIME."""
    assert len(FILE_SCOPE_CODES) == 18
    assert len(set(FILE_SCOPE_CODES)) == 18
    assert "IMPLAUSIBLE_INTENSITY_RANGE" in FILE_SCOPE_CODES
    assert "DUPLICATE_INJECTION_TIME" in FILE_SCOPE_CODES


@pytest.mark.parametrize("code", FILE_SCOPE_CODES)
def test_inject_produces_a_renderable_pair(code: str) -> None:
    """Cada código injetável produz o par measurements + Sample_List."""
    spec = inject(code)
    measurements, sample_list = render_pair(spec)
    assert measurements and sample_list
    assert sample_list.startswith("ID,inj_order_name")


def test_inject_missing_name_row() -> None:
    assert _parse(render_measurements(inject("MISSING_NAME_ROW")))[0][0] != "NAME"


def test_inject_missing_label_row() -> None:
    assert _parse(render_measurements(inject("MISSING_LABEL_ROW")))[1][0] != "LABEL"


def test_inject_ragged_rows() -> None:
    rows = _parse(render_measurements(inject("RAGGED_ROWS")))
    assert len({len(r) for r in rows}) > 1


def test_inject_duplicate_sample_name() -> None:
    names = _parse(render_measurements(inject("DUPLICATE_SAMPLE_NAME")))[0][1:]
    assert len(names) != len(set(names))


def test_inject_duplicate_feature_name() -> None:
    rows = _parse(render_measurements(inject("DUPLICATE_FEATURE_NAME")))
    features = [r[0] for r in rows[2:]]
    assert len(features) != len(set(features))


def test_inject_sample_list_mismatch() -> None:
    spec = inject("SAMPLE_LIST_MISMATCH")
    measurements, sample_list = render_pair(spec)
    in_matrix = set(_parse(measurements)[0][1:])
    in_list = {r[1] for r in _parse(sample_list)[1:] if r}
    assert in_matrix - in_list, "alguma amostra da matriz deveria faltar na lista"


def test_inject_no_qc_samples() -> None:
    labels = _parse(render_measurements(inject("NO_QC_SAMPLES")))[1][1:]
    assert QC_LABEL not in labels


def test_inject_insufficient_qcs() -> None:
    labels = _parse(render_measurements(inject("INSUFFICIENT_QCS")))[1][1:]
    assert 0 < labels.count(QC_LABEL) < 5


def test_inject_non_numeric_cell() -> None:
    rows = _parse(render_measurements(inject("NON_NUMERIC_CELL")))
    with pytest.raises(ValueError):
        float(rows[2][1])


def test_inject_negative_intensity() -> None:
    rows = _parse(render_measurements(inject("NEGATIVE_INTENSITY")))
    assert float(rows[2][1]) < 0


def test_inject_all_zero_feature() -> None:
    rows = _parse(render_measurements(inject("ALL_ZERO_FEATURE")))
    assert all(float(v) == 0 for v in rows[2][1:])


def test_inject_high_zero_fraction() -> None:
    rows = _parse(render_measurements(inject("HIGH_ZERO_FRACTION")))
    cells = [float(v) for r in rows[2:] for v in r[1:]]
    assert sum(1 for v in cells if v == 0) / len(cells) > 0.30


def test_inject_thousands_separator_corruption() -> None:
    """A célula casa o padrão do caractere que NÃO é o decimal.

    A base é `,`/`.`, então o separador detectável é a VÍRGULA — e o
    serializador a cita automaticamente por conter o delimitador. Pôr `1.234`
    aqui seria indetectável por construção: com ponto decimal aquilo é ambíguo.
    """
    spec = inject("THOUSANDS_SEPARATOR_CORRUPTION")
    assert spec.decimal == "."
    rows = _parse(render_measurements(spec))
    assert re.fullmatch(r"\d{1,3}(,\d{3})+", rows[2][1]), (
        f"esperava milhar por vírgula, veio {rows[2][1]!r}"
    )


def test_inject_injection_order_gap() -> None:
    sample_list = render_sample_list(inject("INJECTION_ORDER_GAP"))
    orders = sorted(int(r[0]) for r in _parse(sample_list)[1:] if r)
    assert orders != list(range(1, len(orders) + 1))


def test_inject_group_label_variant_collapses_to_a_canonical_member() -> None:
    """O LABEL colapsa num membro do conjunto canônico, com texto diferente."""
    labels = _parse(render_measurements(inject("GROUP_LABEL_VARIANT")))[1][1:]
    variants = [
        lbl
        for lbl in labels
        if lbl != QC_LABEL
        and lbl not in CANONICAL_GROUPS
        and _canon(lbl) in {_canon(g) for g in CANONICAL_GROUPS}
    ]
    assert variants, "esperava ao menos um rótulo variante"
    assert "HAM-TSP" in variants


def test_inject_label_name_mismatch_does_not_collapse() -> None:
    """O LABEL NÃO colapsa no grupo embutido no nome."""
    rows = _parse(render_measurements(inject("LABEL_NAME_MISMATCH")))
    names, labels = rows[0][1:], rows[1][1:]
    mismatches = [
        (name, label)
        for name, label in zip(names, labels, strict=True)
        if (m := re.search(r"_ID\d+_(.+)$", name)) and _canon(m.group(1)) != _canon(label)
    ]
    assert len(mismatches) == 1
    assert mismatches[0][1] == "Assyntomatic"


# ---------------------------------------------------------------------------
# Casos que os dados reais não têm
# ---------------------------------------------------------------------------


def test_a_sample_can_trip_both_label_codes() -> None:
    """A não-exclusividade dos dois códigos de rótulo.

    `LABEL` grafa `HAM-TSP` (variante de `HAMTSP`) e o nome diz `ATLL`. A
    amostra é variante **e** divergente do nome — dois defeitos distintos na
    mesma coluna. Impossível nos dados reais, possível por construção.
    """
    rows = _parse(render_measurements(inject_both_label_codes()))
    names, labels = rows[0][1:], rows[1][1:]
    both = []
    canon_set = {_canon(g) for g in CANONICAL_GROUPS}
    for name, label in zip(names, labels, strict=True):
        if label == QC_LABEL:
            continue
        is_variant = label not in CANONICAL_GROUPS and _canon(label) in canon_set
        match = re.search(r"_ID\d+_(.+)$", name)
        is_mismatch = bool(match) and _canon(match.group(1)) != _canon(label)
        if is_variant and is_mismatch:
            both.append((name, label))
    assert len(both) == 1, f"esperava uma amostra com os dois defeitos, achei {both}"


def test_spec_without_group_in_name_exposes_no_group() -> None:
    """Amostras sem `_ID<n>_`: LABEL_NAME_MISMATCH não tem com o que comparar."""
    names = _parse(render_measurements(spec_without_group_in_name()))[0][1:]
    assert not any(re.search(r"_ID\d+_", n) for n in names)


@pytest.mark.parametrize("code", STUDY_SCOPE_CODES)
def test_study_scope_codes_are_not_injectable(code: str) -> None:
    """Códigos de escopo estudo falham alto em vez de devolver CSV inútil."""
    with pytest.raises(ValueError, match=r"escopo estudo|não injetável"):
        inject(code)


def test_unknown_code_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"não injetável"):
        inject("CODIGO_QUE_NAO_EXISTE")


def test_factory_is_deterministic() -> None:
    """Duas chamadas com os mesmos argumentos produzem bytes idênticos."""
    for code in FILE_SCOPE_CODES:
        assert render_pair(inject(code)) == render_pair(inject(code))


# ---------------------------------------------------------------------------
# Os nove fixtures em disco
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(FIXTURE_BUILDERS))
def test_fixture_file_exists_with_its_sample_list(name: str) -> None:
    """Cada fixture está em disco, com o Sample_List par."""
    assert (FIXTURES / f"{name}.csv").is_file()
    assert (FIXTURES / f"{name}_sample_list.csv").is_file()


def test_section_8_3_names_exactly_nine_fixtures() -> None:
    assert len(FIXTURE_BUILDERS) == 9


def test_tiny_clean_has_six_samples_two_qcs_five_features() -> None:
    """6 amostras (2 QCs), 5 features."""
    rows = _parse((FIXTURES / "tiny_clean.csv").read_text(encoding="utf-8"))
    assert len(rows[0]) - 1 == 6
    assert rows[1][1:].count(QC_LABEL) == 2
    assert len(rows) - 2 == 5


def test_tiny_clean_would_trip_insufficient_qcs_and_that_is_expected() -> None:
    """Documenta a tensão em vez de deixá-la implícita.

    Nenhuma matriz pequena o bastante para ser legível à mão evita
    INSUFFICIENT_QCS. É `warning`, não aborta, e os testes de matriz limpa
    afirmam ausência de `error` — nunca ausência total de issues.
    """
    rows = _parse((FIXTURES / "tiny_clean.csv").read_text(encoding="utf-8"))
    assert rows[1][1:].count(QC_LABEL) < 5


def test_tiny_drift_grows_monotonically_with_injection_order() -> None:
    """Deriva de 10% por injeção, medida contra tiny_clean."""
    clean = _parse((FIXTURES / "tiny_clean.csv").read_text(encoding="utf-8"))
    drift = _parse((FIXTURES / "tiny_drift.csv").read_text(encoding="utf-8"))
    order = {
        r[1]: int(r[0])
        for r in _parse((FIXTURES / "tiny_clean_sample_list.csv").read_text(encoding="utf-8"))[1:]
        if r
    }
    names = clean[0][1:]
    ratios = {}
    for col, name in enumerate(names, start=1):
        ratio = float(drift[2][col]) / float(clean[2][col])
        ratios[order[name]] = ratio
    ordered = [ratios[k] for k in sorted(ratios)]
    assert ordered == sorted(ordered), "a razão deveria crescer com a ordem de injeção"
    expected = (1 + DRIFT_PER_INJECTION) ** (max(ratios) - 1)
    assert ordered[-1] == pytest.approx(expected, rel=1e-9)


def test_tiny_step_is_flat_then_jumps_after_the_third_injection() -> None:
    """Salto de 50% após a terceira injeção, e nada antes dela."""
    clean = _parse((FIXTURES / "tiny_clean.csv").read_text(encoding="utf-8"))
    step = _parse((FIXTURES / "tiny_step.csv").read_text(encoding="utf-8"))
    order = {
        r[1]: int(r[0])
        for r in _parse((FIXTURES / "tiny_clean_sample_list.csv").read_text(encoding="utf-8"))[1:]
        if r
    }
    for col, name in enumerate(clean[0][1:], start=1):
        ratio = float(step[2][col]) / float(clean[2][col])
        expected = STEP_FACTOR if order[name] > STEP_AFTER_INJECTION else 1.0
        assert ratio == pytest.approx(expected, rel=1e-9), f"{name}: ordem {order[name]}"


def test_tiny_zeros_has_zeros_exactly_where_declared() -> None:
    """Zeros nas posições conhecidas, e valor não-nulo em todas as outras."""
    rows = _parse((FIXTURES / "tiny_zeros.csv").read_text(encoding="utf-8"))
    for f, row in enumerate(rows[2:]):
        for s, cell in enumerate(row[1:]):
            is_zero = float(cell) == 0.0
            assert is_zero == ((f, s) in ZERO_POSITIONS), f"posição ({f}, {s})"


def test_tiny_tab_uses_tab_delimiter() -> None:
    text = (FIXTURES / "tiny_tab.csv").read_text(encoding="utf-8")
    rows = _parse(text, delimiter="\t")
    assert rows[0][0] == "NAME"
    assert len(rows[0]) == 7


def test_tiny_quoted_has_a_comma_inside_a_quoted_field() -> None:
    """A contagem ingênua de vírgulas difere da contagem de campos."""
    text = (FIXTURES / "tiny_quoted.csv").read_text(encoding="utf-8")
    rows = _parse(text)
    feature_row = next(r for r in rows[2:] if "," in r[0])
    assert "," in feature_row[0], "algum nome de feature deve ter vírgula interna"

    line = next(ln for ln in text.splitlines() if ln.startswith('"1.00_100'))
    parsed = _parse(line)[0]
    assert line.count(",") > len(parsed) - 1, "a contagem ingênua deveria superestimar"


def test_corrupt_thousands_is_ptbr_with_dot_grouping() -> None:
    """O defeito é DETECTÁVEL: `;`/`,` com ponto agrupando milhares.

    E precisa de decimais com vírgula em abundância: sem eles o passo 2 não tem
    evidência para eleger a vírgula como decimal, e `1.234` volta a ser ambíguo.
    A detecção de decimal e a de milhar são mutuamente dependentes.
    """
    rows = _parse((FIXTURES / "corrupt_thousands.csv").read_text(encoding="utf-8"), delimiter=";")
    cells = [v for r in rows[2:] for v in r[1:]]

    assert any(re.fullmatch(r"\d{1,3}(\.\d{3})+", c) for c in cells), "milhar por ponto"
    assert sum(1 for c in cells if re.fullmatch(r"\d+,\d+", c)) >= 4, (
        "sem decimais com vírgula o passo 2 não resolve, e o defeito fica indetectável"
    )


def test_corrupt_ragged_has_an_uneven_row() -> None:
    rows = _parse((FIXTURES / "corrupt_ragged.csv").read_text(encoding="utf-8"))
    assert len({len(r) for r in rows}) > 1


def test_corrupt_semicolon_is_a_valid_ptbr_file_not_a_corrupt_one() -> None:
    """pt-BR legítimo, com decimal de três casas.

    A segunda metade do critério depende deste arquivo ter um valor que casa
    `\\d{1,3}(,\\d{3})+` sem ser corrupção. Sem isso, o critério não tem como
    provar que a detecção respeita o dialeto.
    """
    rows = _parse((FIXTURES / "corrupt_semicolon.csv").read_text(encoding="utf-8"), delimiter=";")
    assert rows[0][0] == "NAME"
    assert len({len(r) for r in rows}) == 1, "o arquivo é estruturalmente válido"

    cells = [v for r in rows[2:] for v in r[1:]]
    three_decimals = [c for c in cells if re.fullmatch(r"\d{1,3},\d{3}", c)]
    assert three_decimals, "precisa de valor com exatamente três casas decimais"


def test_all_fixtures_use_crlf_like_the_real_files() -> None:
    for name in FIXTURE_BUILDERS:
        raw = (FIXTURES / f"{name}.csv").read_bytes()
        assert raw.count(b"\r\n") > 0
        assert raw.count(b"\n") == raw.count(b"\r\n"), f"{name}: LF solto"


def test_regenerating_fixtures_is_idempotent(tmp_path: Path) -> None:
    """Duas gerações produzem bytes idênticos — o gerador é determinístico."""
    from tests.fixture_files import write_all

    first = {p.name: p.read_bytes() for p in write_all(tmp_path / "a")}
    second = {p.name: p.read_bytes() for p in write_all(tmp_path / "b")}
    assert first == second


def test_fixtures_on_disk_match_what_the_generator_produces(tmp_path: Path) -> None:
    """Os arquivos versionados não derivaram do gerador.

    Sem isto, alguém edita um fixture à mão, o gerador passa a discordar, e a
    próxima regeneração desfaz a edição em silêncio.
    """
    from tests.fixture_files import write_all

    for path in write_all(tmp_path):
        committed = FIXTURES / path.name
        assert committed.read_bytes() == path.read_bytes(), (
            f"{path.name} em disco difere do gerador; rode `python -m tests.fixture_files`"
        )


def test_inject_implausible_intensity_range() -> None:
    """Mediana muito abaixo do piso de 1e2 para LC-MS."""
    import statistics

    spec = inject("IMPLAUSIBLE_INTENSITY_RANGE")
    flat = [v for row in spec.values for v in row]
    assert statistics.median(flat) < 1e2

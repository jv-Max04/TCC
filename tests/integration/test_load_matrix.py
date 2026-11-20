"""`load_matrix` sobre os quatro modos reais.

As contagens de rótulo aqui são **contagens esperadas sobre os arquivos do
laboratório**, não invariantes de desenho. Estão nomeadas como tal, e o
teste cita os arquivos.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from refino.io.readers import IngestionError, load_matrix, load_matrix_issues, read_sample_list
from tests.conftest import (
    EXPECTED_FEATURES,
    EXPECTED_QC_ORDERS,
    EXPECTED_SAMPLES,
    MODES,
    measurements_path,
    sample_list_path,
)

# Contagens MEDIDAS nos quatro `measurements` do laboratório.
EXPECTED_VARIANT = {"pos_hilic": 0, "neg_hilic": 14, "pos_c18": 14, "neg_c18": 14}
EXPECTED_MISMATCH = {"pos_hilic": 0, "neg_hilic": 0, "pos_c18": 0, "neg_c18": 3}
MISMATCHED_SAMPLES_NEG_C18 = ("ID33", "ID56", "ID58")


def _load(mode: str, **kwargs: object) -> object:
    return load_matrix(measurements_path(mode), sample_list_path(mode), mode, **kwargs)  # type: ignore[arg-type]


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
def test_ca_1_1_four_modes_load_under_defaults(mode: str) -> None:
    """Os quatro carregam sob os parâmetros padrão, com as dimensões da especificação."""
    matrix = _load(mode)
    assert matrix.values.shape == (EXPECTED_SAMPLES, EXPECTED_FEATURES[mode])  # type: ignore[attr-defined]
    assert len(matrix.feature_names) == EXPECTED_FEATURES[mode]  # type: ignore[attr-defined]


@pytest.mark.integration
def test_ca_1_1_neg_c18_loads_with_three_resolved_mismatches() -> None:
    """O `neg_c18` emite exatamente três avisos e resolve para Asymptomatic."""
    _, _, issues = load_matrix_issues(measurements_path("neg_c18"), sample_list_path("neg_c18"))
    mismatches = [i for i in issues if i.code == "LABEL_NAME_MISMATCH"]
    assert len(mismatches) == 3
    for marker in MISMATCHED_SAMPLES_NEG_C18:
        assert any(marker in (i.location or "") for i in mismatches), marker

    matrix = _load("neg_c18")
    assert "Assyntomatic" not in matrix.groups  # type: ignore[attr-defined]
    assert "Asymptomatic" in matrix.groups  # type: ignore[attr-defined]


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
def test_ca_1_2_seven_qcs_at_the_documented_orders(mode: str) -> None:
    """7 QCs nas injeções 1, 12, 23, 34, 45, 56 e 68."""
    matrix = _load(mode)
    qc_orders = sorted(matrix.injection_orders[matrix.qc_mask].tolist())  # type: ignore[attr-defined]
    assert qc_orders == list(EXPECTED_QC_ORDERS)


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
def test_ca_1_3_samples_are_sorted_by_injection_order_without_gaps(mode: str) -> None:
    """Ordenado por `injection_order` crescente, sem lacunas de 1 a 68."""
    matrix = _load(mode)
    assert matrix.injection_orders.tolist() == list(range(1, EXPECTED_SAMPLES + 1))  # type: ignore[attr-defined]


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
def test_transposition_puts_samples_in_rows(mode: str) -> None:
    """Orientação canônica: o arquivo tem features em linhas, a matriz não."""
    matrix = _load(mode)
    assert matrix.values.shape[0] == EXPECTED_SAMPLES  # type: ignore[attr-defined]
    assert matrix.values.shape[1] == EXPECTED_FEATURES[mode]  # type: ignore[attr-defined]


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
def test_transposition_preserves_each_value(mode: str) -> None:
    """A transposição não reordena valores dentro da coluna.

    Verifica um valor por canto: se a transposição trocasse eixos sem reindexar
    as amostras, as dimensões continuariam certas e os números estariam errados.
    """
    doc, orders, _ = load_matrix_issues(measurements_path(mode), sample_list_path(mode))
    matrix = _load(mode)

    for feature_index in (0, doc.n_features - 1):
        for name in (doc.sample_names[0], doc.sample_names[-1]):
            column = doc.sample_names.index(name)
            row = orders[name] - 1
            assert matrix.values[row, feature_index] == doc.values[feature_index, column]  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Contagens de rótulo — OBSERVAÇÃO sobre os arquivos do laboratório
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
def test_measured_label_code_counts_on_the_lab_files(mode: str) -> None:
    """Contagens medidas em `data/reference/input/measurements_{modo}.csv`.

    NÃO é invariante de desenho: é predição verificável sobre dado que não
    controlo. Se o laboratório regerar um arquivo e a contagem mudar, este teste
    reprova — que é o sinal desejado.
    """
    _, _, issues = load_matrix_issues(measurements_path(mode), sample_list_path(mode))
    counts = Counter(i.code for i in issues)
    assert counts["GROUP_LABEL_VARIANT"] == EXPECTED_VARIANT[mode]
    assert counts["LABEL_NAME_MISMATCH"] == EXPECTED_MISMATCH[mode]


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
def test_no_error_severity_issue_in_any_real_mode(mode: str) -> None:
    """Nenhum dos quatro arquivos tem problema que aborte a ingestão."""
    _, _, issues = load_matrix_issues(measurements_path(mode), sample_list_path(mode))
    assert not [i for i in issues if i.is_error], [str(i) for i in issues if i.is_error]


@pytest.mark.integration
@pytest.mark.parametrize("code", ["ALL_ZERO_FEATURE", "NEGATIVE_INTENSITY", "HIGH_ZERO_FRACTION"])
def test_codes_that_are_silent_on_every_real_mode(code: str) -> None:
    """Observação: nenhuma feature totalmente zero, nenhum negativo, e o pior
    caso de zeros é 25,4% no `neg_hilic` — abaixo do corte de 30%."""
    for mode in MODES:
        _, _, issues = load_matrix_issues(measurements_path(mode), sample_list_path(mode))
        assert not [i for i in issues if i.code == code], f"{mode}: {code} disparou"


@pytest.mark.integration
def test_canonicalization_converges_the_four_modes_to_the_same_groups() -> None:
    """Depois de `prefer_name`, os quatro modos têm o MESMO conjunto de grupos.

    É a consequência visível das duas políticas juntas: a canonicalização colapsa
    `HAM/TSP` e `HAM-TSP` em `HAMTSP`, e `prefer_name` resolve `Assyntomatic`.
    Antes, os quatro discordavam.
    """
    groups = {mode: set(_load(mode).groups) for mode in MODES}  # type: ignore[attr-defined]
    assert len(set(map(frozenset, groups.values()))) == 1, groups
    assert next(iter(groups.values())) == {"QC", "NoHTLV", "Asymptomatic", "HAMTSP", "ATLL"}


# ---------------------------------------------------------------------------
# Políticas de rótulo
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_error_policy_rejects_neg_c18_and_accepts_the_other_three() -> None:
    """`on_label_mismatch="error"` recusa o `neg_c18`, e a recusa é correta."""
    with pytest.raises(IngestionError) as caught:
        _load("neg_c18", on_label_mismatch="error")
    assert "LABEL_NAME_MISMATCH" in str(caught.value)

    for mode in ("pos_hilic", "neg_hilic", "pos_c18"):
        assert _load(mode, on_label_mismatch="error") is not None


@pytest.mark.integration
def test_prefer_label_keeps_the_typo_and_records_the_choice() -> None:
    """`prefer_label` mantém `Assyntomatic` — e grava a decisão."""
    matrix = _load("neg_c18", on_label_mismatch="prefer_label")
    assert "Assyntomatic" in matrix.groups  # type: ignore[attr-defined]
    assert any("prefer_label" in note for note in matrix.provenance)  # type: ignore[attr-defined]


@pytest.mark.integration
def test_warn_and_prefer_label_produce_the_same_data_but_different_provenance() -> None:
    """Na proveniência, "alguém decidiu confiar no LABEL" ≠ "ninguém decidiu nada"."""
    warned = _load("neg_c18", on_label_mismatch="warn")
    preferred = _load("neg_c18", on_label_mismatch="prefer_label")

    assert warned.groups == preferred.groups  # type: ignore[attr-defined]
    assert warned.provenance != preferred.provenance  # type: ignore[attr-defined]


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
def test_provenance_records_dialect_and_label_policy(mode: str) -> None:
    """A matriz sabe dizer de onde veio e sob que política."""
    provenance = " | ".join(_load(mode).provenance)  # type: ignore[attr-defined]
    assert "ingestão de" in provenance
    assert "dialeto" in provenance
    assert "prefer_name" in provenance


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
def test_loaded_values_are_not_writeable(mode: str) -> None:
    """A imutabilidade da especificação sobrevive à ingestão."""
    with pytest.raises(ValueError, match=r"read-only|assignment destination"):
        _load(mode).values[0, 0] = 1.0  # type: ignore[attr-defined]


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
def test_loading_twice_is_deterministic(mode: str) -> None:
    """Ordenação estável: duas ingestões dão a mesma ordem de linhas."""
    first, second = _load(mode), _load(mode)
    assert first.sample_names == second.sample_names  # type: ignore[attr-defined]
    assert np.array_equal(first.values, second.values)  # type: ignore[attr-defined]


@pytest.mark.integration
@pytest.mark.parametrize("mode", MODES)
def test_read_sample_list_covers_one_to_sixty_eight(mode: str) -> None:
    orders = read_sample_list(sample_list_path(mode))
    assert sorted(orders.values()) == list(range(1, EXPECTED_SAMPLES + 1))

"""Entidades de domínio.

Cobre `SampleMetadata` e `RefinoMatrix`. O teste central é o da imutabilidade
do array: `frozen=True` sozinho protegeria só o atributo, e "imutabilidade da
matriz" seria decorativa.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata


def _samples(n: int = 4, qc_positions: tuple[int, ...] = (0, 3)) -> tuple[SampleMetadata, ...]:
    out = []
    for i in range(n):
        is_qc = i in qc_positions
        out.append(
            SampleMetadata(
                name=f"S{i + 1}",
                injection_order=i + 1,
                sample_type="qc" if is_qc else "sample",
                group="QC" if is_qc else "NoHTLV",
            )
        )
    return tuple(out)


def _matrix(n_samples: int = 4, n_features: int = 3, **kwargs: object) -> RefinoMatrix:
    values = np.arange(n_samples * n_features, dtype=np.float64).reshape(n_samples, n_features)
    return RefinoMatrix(
        values=values,
        feature_names=tuple(f"F{i + 1}" for i in range(n_features)),
        samples=_samples(n_samples),
        mode="tiny",
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# SampleMetadata
# ---------------------------------------------------------------------------


def test_sample_metadata_is_immutable() -> None:
    sample = SampleMetadata("S1", 1, "qc", "QC")
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.name = "outro"  # type: ignore[misc]


def test_sample_metadata_batch_defaults_to_a() -> None:
    """`batch` tem padrão `"A"` — o que é um erro
    do pipeline atual que o diagnóstico vai corrigir."""
    assert SampleMetadata("S1", 1, "sample", "NoHTLV").batch == "A"


@pytest.mark.parametrize("sample_type", ["qc", "sample", "validate"])
def test_sample_metadata_accepts_the_three_types(sample_type: str) -> None:
    """`validate` é aceito mesmo sem uso no dataset (achado 4.3)."""
    sample = SampleMetadata("S1", 1, sample_type, "QC")  # type: ignore[arg-type]
    assert sample.sample_type == sample_type


def test_injection_order_is_one_based() -> None:
    with pytest.raises(ValueError, match="1-based"):
        SampleMetadata("S1", 0, "qc", "QC")


def test_empty_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="nome de amostra vazio"):
        SampleMetadata("", 1, "qc", "QC")


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch vazio"):
        SampleMetadata("S1", 1, "qc", "QC", batch="")


def test_validate_is_neither_qc_nor_study_sample() -> None:
    """O terceiro tipo não é nenhum dos dois — a distinção que protege as métricas."""
    sample = SampleMetadata("V1", 1, "validate", "NoHTLV")
    assert not sample.is_qc
    assert not sample.is_study_sample


# ---------------------------------------------------------------------------
# RefinoMatrix — construção
# ---------------------------------------------------------------------------


def test_matrix_exposes_canonical_orientation() -> None:
    """Amostras em linhas, features em colunas."""
    matrix = _matrix(n_samples=4, n_features=3)
    assert matrix.values.shape == (4, 3)
    assert matrix.n_samples == 4
    assert matrix.n_features == 3


def test_matrix_is_immutable_at_the_attribute() -> None:
    matrix = _matrix()
    with pytest.raises(dataclasses.FrozenInstanceError):
        matrix.mode = "outro"  # type: ignore[misc]


def test_matrix_values_are_not_writeable() -> None:
    """A imutabilidade alcança o ARRAY, não só o atributo.

    Sem isto, `frozen=True` daria uma falsa sensação de segurança: o nome
    estaria protegido e o conteúdo, não.
    """
    matrix = _matrix()
    with pytest.raises(ValueError, match=r"read-only|assignment destination"):
        matrix.values[0, 0] = 999.0


def test_matrix_rejects_sample_count_mismatch() -> None:
    values = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="metadados de amostra"):
        RefinoMatrix(values, ("F1", "F2", "F3"), _samples(4), "tiny")


def test_matrix_rejects_feature_count_mismatch() -> None:
    values = np.zeros((4, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="nomes de feature"):
        RefinoMatrix(values, ("F1", "F2"), _samples(4), "tiny")


def test_matrix_rejects_non_float64() -> None:
    values = np.zeros((4, 3), dtype=np.int32)
    with pytest.raises(ValueError, match="float64"):
        RefinoMatrix(values, ("F1", "F2", "F3"), _samples(4), "tiny")


def test_matrix_rejects_one_dimensional_values() -> None:
    with pytest.raises(ValueError, match="2-D"):
        RefinoMatrix(np.zeros(4, dtype=np.float64), ("F1",), _samples(4), "tiny")


def test_matrix_rejects_samples_out_of_injection_order() -> None:
    """A ordenação é invariante de construção, não convenção.

    As métricas medem correlação do PC1 com a ordem de injeção; uma matriz
    desordenada produziria correlação errada e nada ali detectaria.
    """
    shuffled = tuple(reversed(_samples(4)))
    with pytest.raises(ValueError, match="injection_order crescente"):
        RefinoMatrix(np.zeros((4, 3), dtype=np.float64), ("F1", "F2", "F3"), shuffled, "tiny")


def test_provenance_starts_empty() -> None:
    assert _matrix().provenance == ()


# ---------------------------------------------------------------------------
# RefinoMatrix — máscaras e derivados
# ---------------------------------------------------------------------------


def test_qc_mask_marks_exactly_the_qcs() -> None:
    matrix = _matrix(n_samples=4)
    assert matrix.qc_mask.tolist() == [True, False, False, True]
    assert matrix.qc_mask.dtype == bool


def test_sample_mask_is_not_the_complement_of_qc_mask() -> None:
    """Amostras `validate` são `False` nas DUAS máscaras.

    Um complemento silencioso as trataria como amostras de estudo. Este é o
    teste que impede alguém de "simplificar" sample_mask para ~qc_mask.
    """
    samples = (
        SampleMetadata("QC1", 1, "qc", "QC"),
        SampleMetadata("S1", 2, "sample", "NoHTLV"),
        SampleMetadata("V1", 3, "validate", "NoHTLV"),
    )
    matrix = RefinoMatrix(np.zeros((3, 2), dtype=np.float64), ("F1", "F2"), samples, "tiny")

    assert matrix.qc_mask.tolist() == [True, False, False]
    assert matrix.sample_mask.tolist() == [False, True, False]
    assert matrix.validate_mask.tolist() == [False, False, True]
    assert not (matrix.qc_mask | matrix.sample_mask).all(), "validate não está em nenhuma"


def test_injection_orders_follow_row_order() -> None:
    matrix = _matrix(n_samples=4)
    assert matrix.injection_orders.tolist() == [1, 2, 3, 4]


def test_derived_metadata_tuples_follow_row_order() -> None:
    matrix = _matrix(n_samples=4)
    assert matrix.sample_names == ("S1", "S2", "S3", "S4")
    assert matrix.groups == ("QC", "NoHTLV", "NoHTLV", "QC")
    assert matrix.batches == ("A", "A", "A", "A")


# ---------------------------------------------------------------------------
# RefinoMatrix — transformação
# ---------------------------------------------------------------------------


def test_with_values_returns_a_new_matrix_and_leaves_the_original_intact() -> None:
    original = _matrix()
    updated = original.with_values(np.ones_like(original.values), "teste")

    assert updated is not original
    assert updated.values[0, 0] == 1.0
    assert original.values[0, 0] == 0.0, "a original não deve mudar"


def test_with_values_extends_provenance() -> None:
    matrix = (
        _matrix().with_values(np.ones((4, 3)), "primeira").with_values(np.zeros((4, 3)), "segunda")
    )
    assert matrix.provenance == ("primeira", "segunda")


def test_with_values_rejects_a_different_shape() -> None:
    with pytest.raises(ValueError, match="incompatível"):
        _matrix().with_values(np.zeros((2, 2)), "teste")


def test_with_values_requires_naming_the_operation() -> None:
    """Provenance sem nome não serve para nada — exige-se rastreabilidade."""
    with pytest.raises(ValueError, match="nomear a operação"):
        _matrix().with_values(np.ones((4, 3)), "")


def test_with_values_coerces_to_float64() -> None:
    matrix = _matrix().with_values(np.ones((4, 3), dtype=np.int32), "coerção")
    assert matrix.values.dtype == np.float64


def test_with_values_output_is_also_not_writeable() -> None:
    """A imutabilidade se propaga: a matriz derivada também é congelada."""
    derived = _matrix().with_values(np.ones((4, 3)), "teste")
    with pytest.raises(ValueError, match=r"read-only|assignment destination"):
        derived.values[0, 0] = 5.0


def test_with_provenance_records_a_decision_without_touching_values() -> None:
    """Registrar intenção quando nenhum número mudou."""
    original = _matrix()
    annotated = original.with_provenance("política de rótulo: prefer_name")

    assert annotated.provenance == ("política de rótulo: prefer_name",)
    assert np.array_equal(annotated.values, original.values)
    assert original.provenance == ()


def test_with_provenance_rejects_an_empty_note() -> None:
    with pytest.raises(ValueError, match="nota não vazia"):
        _matrix().with_provenance("")

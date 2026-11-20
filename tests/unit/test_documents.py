"""DTOs de formato.

Cobre `MeasurementsDocument`. `ToSerrfDocument` e
`MissingGroupsError` ficam descobertos aqui — 85% é
piso, e forçar cobertura em código sem consumidor produziria teste que exercita
linha sem afirmar comportamento.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from refino.io.documents import MeasurementsDocument
from refino.io.sniffer import Dialect

_DIALECT = Dialect(
    delimiter=",", decimal=".", thousands=None, encoding="utf-8", line_terminator="\r\n"
)


def _doc(n_features: int = 3, n_samples: int = 4) -> MeasurementsDocument:
    return MeasurementsDocument(
        values=np.ones((n_features, n_samples), dtype=np.float64),
        feature_names=tuple(f"F{i}" for i in range(n_features)),
        sample_names=tuple(f"S{i}" for i in range(n_samples)),
        group_labels=("QC",) * n_samples,
        raw_cells=(("1",) * n_samples,) * n_features,
        dialect=_DIALECT,
    )


def test_dimensions_follow_the_file_orientation() -> None:
    """Features em LINHAS — a orientação do arquivo, não a canônica.

    A transposição é de `load_matrix`. Um documento que já viesse transposto
    esconderia a conversão dentro do leitor.
    """
    doc = _doc(n_features=3, n_samples=4)
    assert doc.values.shape == (3, 4)
    assert doc.n_features == 3
    assert doc.n_samples == 4


def test_document_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _doc().dialect = _DIALECT  # type: ignore[misc]


def test_issues_defaults_to_empty() -> None:
    """Documento sem problema descoberto na leitura carrega tupla vazia."""
    assert _doc().issues == ()


def test_the_document_carries_what_reading_discovered() -> None:
    """`issues` viaja no valor de retorno, não em canal lateral.

    Sem isso o chamador teria de reler o arquivo — 6 MB no `pos_hilic` — ou
    confiar em estado global para saber o que a leitura omitiu.
    """
    from refino.io.issues import ValidationIssue

    issue = ValidationIssue("error", "RAGGED_ROWS", "linha 4 tem 5 campos, esperava 7")
    doc = dataclasses.replace(_doc(), issues=(issue,))
    assert doc.issues == (issue,)

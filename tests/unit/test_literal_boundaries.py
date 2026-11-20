"""`Literal` na fronteira de confiança.

`Literal` é ferramenta de análise estática. O mypy prova os pontos de chamada
internos, mas dado lido de disco nunca passou pelo mypy: todo `Literal` que
atravessa fronteira de confiança precisa de validador em runtime **na fronteira**.

Este arquivo cobre os três `Literal` do projeto, mais a invariante de
exaustividade que protege contra um defeito diferente.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SAMPLE_TYPES, SampleMetadata
from refino.io.issues import SEVERITIES, ValidationIssue
from refino.io.readers import LABEL_POLICIES, load_matrix, read_toserrf, to_matrix
from tests.conftest import measurements_path, sample_list_path

# ---------------------------------------------------------------------------
# SampleType — o achado P1 da revisão
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sample_type", sorted(SAMPLE_TYPES))
def test_the_three_valid_sample_types_are_accepted(sample_type: str) -> None:
    assert SampleMetadata("S1", 1, sample_type, "QC").sample_type == sample_type  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", ["lixo", "QC", "Sample", "", "validate "])
def test_invalid_sample_type_is_rejected_at_construction(invalid: str) -> None:
    """A guarda que o `Literal` não dá em runtime."""
    with pytest.raises(ValueError, match="sample_type"):
        SampleMetadata("S1", 1, invalid, "QC")  # type: ignore[arg-type]


def test_to_matrix_rejects_an_invalid_sample_type_from_a_file(tmp_path: Path) -> None:
    """O caminho real do defeito: `sampleType` inválido lido de um `toserrf`.

    Antes da guarda, isto produzia uma amostra invisível às TRÊS máscaras. A
    consequência nas métricas é pior do que a perda de uma coluna:
    `leave_one_qc_out_rsd` faria sete iterações declaradas sobre seis QCs reais,
    e a validação cruzada que existe para tornar a métrica honesta reportaria
    número otimista — o defeito comprometeria exatamente o mecanismo do achado 4.3.
    """
    path = tmp_path / "t.csv"
    path.write_text(
        "NA,batch,A\r\nNA,sampleType,lixo\r\nNA,time,1\r\nNo,label,S1\r\n1,F1,100\r\n",
        encoding="utf-8",
        newline="",
    )
    doc = read_toserrf(path)
    assert doc.sample_type == ("lixo",), "o documento de FORMATO preserva o que o arquivo diz"

    with pytest.raises(ValueError, match="sample_type"):
        to_matrix(doc, {"S1": "NoHTLV"})


# ---------------------------------------------------------------------------
# Exaustividade das máscaras — defeito DIFERENTE
# ---------------------------------------------------------------------------


def test_masks_cover_every_sample_exactly_once() -> None:
    """Toda amostra pertence a exatamente uma máscara."""
    samples = (
        SampleMetadata("QC1", 1, "qc", "QC"),
        SampleMetadata("S1", 2, "sample", "NoHTLV"),
        SampleMetadata("V1", 3, "validate", "NoHTLV"),
    )
    matrix = RefinoMatrix(np.zeros((3, 2), dtype=np.float64), ("F1", "F2"), samples, "tiny")

    coverage = matrix.qc_mask.astype(int) + matrix.sample_mask.astype(int)
    coverage += matrix.validate_mask.astype(int)
    assert coverage.tolist() == [1, 1, 1]


def test_a_fourth_sample_type_without_a_mask_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protege contra `SampleType` ganhar valor novo sem a máscara correspondente.

    Defeito DIFERENTE do que a guarda de `SampleMetadata` previne: aqui o valor é
    declarado válido, mas nenhuma máscara o reconhece. Simulado acrescentando um
    quarto tipo ao conjunto sem tocar as máscaras — que é exatamente o que uma
    refatoração distraída faria.
    """
    import refino.domain.metadata as metadata

    monkeypatch.setattr(metadata, "SAMPLE_TYPES", frozenset({*SAMPLE_TYPES, "blank"}))
    orphan = SampleMetadata("B1", 1, "blank", "NoHTLV")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="não cobrem exatamente uma vez"):
        RefinoMatrix(np.zeros((1, 2), dtype=np.float64), ("F1", "F2"), (orphan,), "tiny")


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("severity", sorted(SEVERITIES))
def test_the_three_valid_severities_are_accepted(severity: str) -> None:
    assert ValidationIssue(severity, "X", "m").severity == severity  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", ["ERROR", "fatal", "warn", ""])
def test_invalid_severity_is_rejected(invalid: str) -> None:
    """Severidade desconhecida não seria `error`, e a ingestão prosseguiria."""
    with pytest.raises(ValueError, match="severity"):
        ValidationIssue(invalid, "X", "m")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LabelPolicy — o mais insidioso dos três
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", sorted(LABEL_POLICIES))
def test_the_four_valid_policies_are_accepted(policy: str) -> None:
    matrix = load_matrix(
        measurements_path("pos_hilic"),
        sample_list_path("pos_hilic"),
        "pos_hilic",
        on_label_mismatch=policy,  # type: ignore[arg-type]
    )
    assert any(policy in note for note in matrix.provenance)


@pytest.mark.parametrize("invalid", ["prefer", "trust_name", "PREFER_NAME", ""])
def test_invalid_label_policy_is_rejected_before_reading_anything(invalid: str) -> None:
    """O caso mais insidioso: um valor inválido caía no `else` de `_resolve_group`
    e se comportava como `prefer_label` — razoável, e portanto invisível.

    O defeito nunca apareceria como erro, só como política silenciosamente
    diferente da pedida, com o `provenance` registrando o nome errado.

    `trust_name` está entre os inválidos de propósito: era o nome antigo, e
    um chamador desatualizado obteria `prefer_label` em silêncio.
    """
    with pytest.raises(ValueError, match="on_label_mismatch"):
        load_matrix(
            measurements_path("pos_hilic"),
            sample_list_path("pos_hilic"),
            "pos_hilic",
            on_label_mismatch=invalid,  # type: ignore[arg-type]
        )


def test_the_policy_is_validated_before_the_file_is_read(tmp_path: Path) -> None:
    """A validação vem antes da leitura: falha rápido, sem custo de I/O."""
    with pytest.raises(ValueError, match="on_label_mismatch"):
        load_matrix(
            tmp_path / "nao_existe.csv", tmp_path / "nem_esta.csv", "x", on_label_mismatch="?"
        )  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# read_toserrf — mensagem à altura de read_sample_list
# ---------------------------------------------------------------------------


def test_non_integer_time_names_the_file_and_the_cell(tmp_path: Path) -> None:
    """Mesma cortesia de `read_sample_list`, que já nomeava arquivo e linha."""
    path = tmp_path / "ruim.csv"
    path.write_text(
        "NA,batch,A\r\nNA,sampleType,qc\r\nNA,time,x\r\nNo,label,QC1\r\n1,F1,100\r\n",
        encoding="utf-8",
        newline="",
    )
    with pytest.raises(ValueError) as caught:
        read_toserrf(path)

    message = str(caught.value)
    assert "ruim.csv" in message, "a mensagem deve nomear o arquivo"
    assert "'x'" in message, "e a célula ofensora"
    assert "adivinhá-la não é opção" in message

"""Equivalência semântica entre dois `ToSerrfDocument`.

Compara documentos e **não** matrizes: `RefinoMatrix` exige `group`, que o
formato não carrega, então uma comparação via domínio precisaria inventar ou
reconstruir grupo para os dois lados — introduzindo no critério de equivalência
de formato uma informação que não está em nenhum dos dois arquivos.

E **não afirma nada sobre a qualidade do arquivo**: o critério é fidelidade
de ida e volta. `time` repetido deve ser reproduzido repetido.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from refino.io.documents import MissingGroupsError, ToSerrfDocument
from refino.io.readers import load_matrix, read_toserrf, to_matrix
from refino.io.writers import write_toserrf
from tests.conftest import MODES, measurements_path, sample_list_path, toserrf_path

SIGNIFICANT_DIGITS = 10


def _same_number(a: float, b: float) -> bool:
    """Igualdade em 10 dígitos significativos; zeros comparam exatamente."""
    if a == 0 or b == 0:
        return a == b
    return bool(np.isclose(a, b, rtol=10.0**-SIGNIFICANT_DIGITS, atol=0.0))


def assert_equivalent(produced: ToSerrfDocument, reference: ToSerrfDocument) -> None:
    """Equivalência campo a campo.

    Fora da comparação, por serem artefatos de serialização e não conteúdo:
    terminador de linha, aspas, `NA` versus célula vazia na primeira coluna, e a
    caixa do expoente em notação científica. Nada disso chega aqui — os dois
    lados já passaram pelo parser, que normaliza os quatro.
    """
    assert produced.sample_names == reference.sample_names, "nomes de amostra"
    assert produced.batch == reference.batch, "batch"
    assert produced.sample_type == reference.sample_type, "sampleType"
    assert produced.time == reference.time, "time"
    assert produced.feature_names == reference.feature_names, "ordem das features"
    assert produced.values.shape == reference.values.shape, "dimensões"

    mismatches = [
        (f, s, float(produced.values[f, s]), float(reference.values[f, s]))
        for f in range(produced.n_features)
        for s in range(produced.n_samples)
        if not _same_number(float(produced.values[f, s]), float(reference.values[f, s]))
    ]
    assert not mismatches, f"{len(mismatches)} célula(s) divergente(s): {mismatches[:5]}"


@pytest.mark.golden
@pytest.mark.parametrize("mode", MODES)
def test_ca_1_4_write_toserrf_is_equivalent_to_the_reference(mode: str, tmp_path: Path) -> None:
    """Nos quatro modos, comparando documentos."""
    matrix = load_matrix(measurements_path(mode), sample_list_path(mode), mode)
    produced_path = tmp_path / f"{mode}_toserrf.csv"
    write_toserrf(matrix, produced_path)

    assert_equivalent(read_toserrf(produced_path), read_toserrf(toserrf_path(mode)))


@pytest.mark.golden
@pytest.mark.parametrize("mode", MODES)
def test_reference_toserrf_uses_na_in_the_first_column(mode: str) -> None:
    """`NA` literal sem aspas, não célula vazia.

    Documenta o artefato que o critério exclui. Se ele desaparecer de um
    arquivo regerado, o critério continua passando — que é o ponto de excluí-lo.
    """
    first_line = toserrf_path(mode).read_bytes().split(b"\r\n")[0]
    assert first_line.startswith(b"NA,")


@pytest.mark.golden
@pytest.mark.parametrize("mode", MODES)
def test_uppercase_exponent_in_the_reference_does_not_break_equivalence(mode: str) -> None:
    """As 38 células com `E` maiúsculo são normalizadas pelo parser."""
    doc = read_toserrf(toserrf_path(mode))
    assert np.isfinite(doc.values).all(), "o parser aceitou toda a notação do arquivo"


@pytest.mark.golden
@pytest.mark.parametrize("mode", MODES)
def test_zeros_survive_the_write(mode: str, tmp_path: Path) -> None:
    """Achado 4.6 — os zeros da entrada estão nas mesmas posições na saída."""
    matrix = load_matrix(measurements_path(mode), sample_list_path(mode), mode)
    produced = tmp_path / "out.csv"
    write_toserrf(matrix, produced)

    written = read_toserrf(produced)
    assert np.array_equal(matrix.values.T == 0, written.values == 0)


# ---------------------------------------------------------------------------
# O documento não tem grupo, e to_matrix exige o mapa
# ---------------------------------------------------------------------------


def test_toserrf_document_has_no_group_attribute() -> None:
    """Verificado por introspecção, para que ninguém o acrescente.

    Se `group` aparecer aqui, o critério volta a poder ser escrito via domínio
    e a informação inventada volta ao critério de equivalência.
    """
    assert not hasattr(ToSerrfDocument, "group")
    assert "group" not in ToSerrfDocument.__dataclass_fields__


@pytest.mark.golden
@pytest.mark.parametrize("mode", MODES)
def test_to_matrix_reproduces_the_origin_matrix(mode: str, tmp_path: Path) -> None:
    """Ida e volta completa, com promoção explícita a domínio."""
    origin = load_matrix(measurements_path(mode), sample_list_path(mode), mode)
    produced = tmp_path / "out.csv"
    write_toserrf(origin, produced)

    groups = dict(zip(origin.sample_names, origin.groups, strict=True))
    restored = to_matrix(read_toserrf(produced), groups, mode)

    assert restored.sample_names == origin.sample_names
    assert restored.groups == origin.groups
    assert restored.injection_orders.tolist() == origin.injection_orders.tolist()
    assert restored.batches == origin.batches
    assert np.allclose(restored.values, origin.values, rtol=1e-9, atol=0.0)


@pytest.mark.golden
def test_to_matrix_names_every_missing_group_not_just_the_first() -> None:
    """Quem precisa completar o mapa precisa da lista inteira."""
    doc = read_toserrf(toserrf_path("neg_c18"))
    non_qc = [n for n, k in zip(doc.sample_names, doc.sample_type, strict=True) if k != "qc"]

    with pytest.raises(MissingGroupsError) as caught:
        to_matrix(doc, {name: "NoHTLV" for name in non_qc[3:]})

    assert len(caught.value.missing) == 3
    for name in non_qc[:3]:
        assert name in str(caught.value)


@pytest.mark.golden
def test_to_matrix_assigns_qc_without_consulting_the_map() -> None:
    """QCs não precisam estar em `groups` — o `sampleType` já os identifica."""
    doc = read_toserrf(toserrf_path("neg_c18"))
    non_qc = [n for n, k in zip(doc.sample_names, doc.sample_type, strict=True) if k != "qc"]

    matrix = to_matrix(doc, {name: "NoHTLV" for name in non_qc})
    assert set(matrix.groups) == {"QC", "NoHTLV"}
    assert matrix.qc_mask.sum() == 7


@pytest.mark.golden
@pytest.mark.parametrize("mode", MODES)
def test_read_toserrf_reports_no_duplicate_time_on_the_reference(mode: str) -> None:
    """Observação, não invariante.

    Os quatro arquivos de referência têm `time` de 1 a 68 sem repetição. Isso é
    propriedade DESTES arquivos: o critério não afirma monotonicidade, e um
    `toserrf` com `time` repetido deve ter o repetido reproduzido pela ida e
    volta — só recebe `DUPLICATE_INJECTION_TIME` na validação.
    """
    doc = read_toserrf(toserrf_path(mode))
    assert not [i for i in doc.issues if i.code == "DUPLICATE_INJECTION_TIME"]
    assert sorted(doc.time) == list(range(1, 69))


def test_duplicate_time_is_reported_but_faithfully_round_tripped(tmp_path: Path) -> None:
    """A ida e volta preserva o `time` que existir, e a validação julga.

    É o teste que separa fidelidade de qualidade. Um arquivo com `time` repetido
    passa a equivalência campo a campo E recebe o código de erro.
    """
    path = tmp_path / "dup.csv"
    path.write_text(
        "NA,batch,A,A\r\n"
        "NA,sampleType,qc,sample\r\n"
        "NA,time,5,5\r\n"
        "No,label,QC1,S1\r\n"
        "1,F1,100,200\r\n",
        encoding="utf-8",
        newline="",
    )

    doc = read_toserrf(path)
    assert doc.time == (5, 5), "a leitura preserva o repetido"
    assert [i.code for i in doc.issues] == ["DUPLICATE_INJECTION_TIME"]
    assert doc.issues[0].is_error

    assert_equivalent(doc, read_toserrf(path))

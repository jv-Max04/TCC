"""Leitores. Os quatro códigos que só o leitor pode emitir.

Fecha a partição já declarada: `MISSING_NAME_ROW`, `MISSING_LABEL_ROW`,
`RAGGED_ROWS` e `NON_NUMERIC_CELL` dependem da estrutura de linhas e não
sobrevivem ao parse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refino.io.issues import FILE_SCOPE_CODES
from refino.io.readers import IngestionError, read_measurements, read_sample_list
from refino.io.validators import validate_structure
from tests.factories import inject, render_pair

READER_CODES = ("MISSING_NAME_ROW", "MISSING_LABEL_ROW", "RAGGED_ROWS", "NON_NUMERIC_CELL")


def _write(tmp_path: Path, code: str) -> tuple[Path, Path]:
    measurements, sample_list = render_pair(inject(code))
    m = tmp_path / "m.csv"
    s = tmp_path / "s.csv"
    m.write_text(measurements, encoding="utf-8", newline="")
    s.write_text(sample_list, encoding="utf-8", newline="")
    return m, s


@pytest.mark.parametrize("code", READER_CODES)
def test_reader_emitted_code_reaches_validate_structure(tmp_path: Path, code: str) -> None:
    """O código é emitido pelo leitor e INCORPORADO por `validate_structure`.

    Fecha a partição: o chamador recebe uma lista só, e não precisa saber
    qual camada descobriu o quê.
    """
    m, s = _write(tmp_path, code)
    doc = read_measurements(m)
    orders = read_sample_list(s, doc.dialect)

    assert code in {i.code for i in doc.issues}, "o leitor deveria emitir"
    assert code in {i.code for i in validate_structure(doc, orders)}, "e o validador incorporar"


def test_the_partition_of_seventeen_codes_is_now_fully_covered() -> None:
    """Todos os 17 têm um teste em algum lugar da suíte.

    Os validadores cobrem 12, o sniffer 1, e este arquivo os 4 restantes. O teste de
    exaustividade provava que a partição era completa; este prova que ela
    está exercitada.
    """
    from tests.unit.test_validators import (
        OWNED_BY_READER,
        OWNED_BY_SNIFFER,
        OWNED_BY_VALIDATE_STRUCTURE,
    )

    assert set(READER_CODES) == OWNED_BY_READER
    covered = OWNED_BY_VALIDATE_STRUCTURE | OWNED_BY_READER | OWNED_BY_SNIFFER
    assert covered == set(FILE_SCOPE_CODES)


def test_thousands_corruption_is_not_double_reported(tmp_path: Path) -> None:
    """Célula corrompida não vira também `NON_NUMERIC_CELL`.

    Sem esse cuidado o mesmo defeito apareceria duas vezes com dois nomes, e a
    contagem por código deixaria de ser interpretável.
    """
    m, s = _write(tmp_path, "THOUSANDS_SEPARATOR_CORRUPTION")
    doc = read_measurements(m)
    codes = {i.code for i in validate_structure(doc, read_sample_list(s, doc.dialect))}

    assert "THOUSANDS_SEPARATOR_CORRUPTION" in codes
    assert "NON_NUMERIC_CELL" not in codes


def test_ragged_row_is_skipped_not_padded(tmp_path: Path) -> None:
    """Linha desigual é reportada e descartada, nunca completada com zeros.

    Preencher inventaria dado — e a matriz resultante teria dimensão plausível
    com conteúdo falso.
    """
    m, _ = _write(tmp_path, "RAGGED_ROWS")
    doc = read_measurements(m)
    assert doc.n_features < len(inject("RAGGED_ROWS").feature_names)


def test_almost_empty_file_raises_with_the_reason(tmp_path: Path) -> None:
    path = tmp_path / "curto.csv"
    path.write_text("NAME,A\r\n", encoding="utf-8", newline="")
    with pytest.raises(IngestionError, match="esperava NAME, LABEL"):
        read_measurements(path)


def test_sample_list_with_non_integer_id_raises(tmp_path: Path) -> None:
    """A ordem de injeção é a única informação temporal — não se adivinha."""
    path = tmp_path / "s.csv"
    path.write_text("ID,inj_order_name\r\nx,AMOSTRA\r\n", encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="não é inteiro"):
        read_sample_list(path)


def test_raw_cells_preserve_the_original_text(tmp_path: Path) -> None:
    """O texto sobrevive à conversão para float64."""
    from tests.conftest import measurements_path

    doc = read_measurements(measurements_path("neg_c18"))
    assert doc.raw_cells[0][0] == "300392.570088058"
    assert doc.values[0, 0] == float(doc.raw_cells[0][0])


def test_raw_cells_exclude_name_and_label_rows(tmp_path: Path) -> None:
    """O recorte que impede `NON_NUMERIC_CELL` de disparar em rótulo clínico."""
    from tests.conftest import measurements_path

    doc = read_measurements(measurements_path("neg_c18"))
    flat = {cell for row in doc.raw_cells for cell in row}
    assert "NAME" not in flat
    assert "LABEL" not in flat
    assert "QC" not in flat


def test_ingestion_error_lists_every_error_not_just_the_first(tmp_path: Path) -> None:
    """A exceção carrega a lista completa."""
    from dataclasses import replace

    from refino.io.readers import load_matrix
    from tests.factories import base_spec, render_measurements, render_sample_list

    spec = inject("DUPLICATE_SAMPLE_NAME")
    spec = replace(spec, name_cell="NOME")  # segundo error, independente

    m = tmp_path / "m.csv"
    s = tmp_path / "s.csv"
    m.write_text(render_measurements(spec), encoding="utf-8", newline="")
    s.write_text(render_sample_list(base_spec()), encoding="utf-8", newline="")

    with pytest.raises(IngestionError) as caught:
        load_matrix(m, s, "tiny")

    codes = {i.code for i in caught.value.issues}
    assert {"MISSING_NAME_ROW", "DUPLICATE_SAMPLE_NAME"} <= codes


# ---------------------------------------------------------------------------
# Caminhos de borda — testados em vez de pragma
# ---------------------------------------------------------------------------


def test_empty_cell_is_read_as_zero(tmp_path: Path) -> None:
    """Célula vazia é ausência, e os arquivos reais escrevem zero para ausência.

    Ler como NaN divergiria do que o Progenesis QI produz: o
    equipamento deixa a célula vazia mas os arquivos convertidos trazem `0`.
    Tratar os dois de formas diferentes criaria duas classes de ausência que o
    resto do pipeline teria de distinguir sem base para isso.
    """
    path = tmp_path / "m.csv"
    path.write_text("NAME,A,B\r\nLABEL,QC,QC\r\nF1,,200\r\n", encoding="utf-8", newline="")
    doc = read_measurements(path)
    assert doc.values[0, 0] == 0.0
    assert not [i for i in doc.issues if i.code == "NON_NUMERIC_CELL"]


def test_ptbr_decimal_is_converted_on_read(tmp_path: Path) -> None:
    """O leitor respeita o decimal do dialeto, não assume ponto."""
    path = tmp_path / "m.csv"
    path.write_text("NAME;A;B\r\nLABEL;QC;QC\r\nF1;1020,5;12,34\r\n", encoding="utf-8", newline="")
    doc = read_measurements(path)
    assert doc.dialect.decimal == ","
    assert doc.values[0, 0] == pytest.approx(1020.5)
    assert doc.values[0, 1] == pytest.approx(12.34)


def test_sample_list_skips_short_rows(tmp_path: Path) -> None:
    """Linha truncada na lista é ignorada, não interpretada pela metade.

    Adivinhar qual campo falta inventaria ordem de injeção, e a ordem é a única
    informação temporal do estudo.
    """
    path = tmp_path / "s.csv"
    path.write_text(
        "ID,inj_order_name\r\n1,PRIMEIRA\r\n2\r\n3,TERCEIRA\r\n", encoding="utf-8", newline=""
    )
    orders = read_sample_list(path)
    assert orders == {"PRIMEIRA": 1, "TERCEIRA": 3}


def test_toserrf_without_the_four_metadata_rows_raises(tmp_path: Path) -> None:
    """Sem as quatro linhas de metadado não há formato `toserrf` a ler."""
    from refino.io.readers import read_toserrf

    path = tmp_path / "curto.csv"
    path.write_text("NA,batch,A\r\nNA,sampleType,qc\r\n", encoding="utf-8", newline="")
    with pytest.raises(IngestionError, match="esperava 4"):
        read_toserrf(path)

"""Detecção de dialeto.

A ordem dos testes neste arquivo é deliberada. O primeiro bloco é o falso
positivo pt-BR: o caso que uma implementação ingênua do passo 3 erra, e o que
prova que a condicionalidade ao decimal faz trabalho real. Escrito antes
do resto porque é o único que distingue a regra correta da regra plausível.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from refino.io.sniffer import Dialect, sniff_dialect
from tests.conftest import measurements_path

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO / "data" / "fixtures"

THOUSANDS_CODE = "THOUSANDS_SEPARATOR_CORRUPTION"


def _codes(dialect: Dialect) -> list[str]:
    return [issue.code for issue in dialect.issues]


# ---------------------------------------------------------------------------
# A metade que importa vem primeiro
# ---------------------------------------------------------------------------


def test_valid_ptbr_file_with_three_decimals_is_not_corruption() -> None:
    """O caso que a regra ingênua rejeita por engano.

    `corrupt_semicolon.csv` é pt-BR VÁLIDO: delimitador `;`, decimal `,`, e um
    valor `1,234` que significa um vírgula duzentos e trinta e quatro. Esse
    texto casa `\\d{1,3}(,\\d{3})+`, então aplicar os dois padrões de milhar
    incondicionalmente o rejeitaria — pondo o critério em conflito direto com
    o critério, cujo fixture de dialeto é exatamente este arquivo.

    A detecção tem de procurar apenas o caractere que **não** é o decimal.
    """
    dialect = sniff_dialect(FIXTURES / "corrupt_semicolon.csv")

    assert dialect.delimiter == ";"
    assert dialect.decimal == ","
    assert THOUSANDS_CODE not in _codes(dialect), (
        "arquivo pt-BR legítimo foi acusado de corrupção de milhar; "
        "o passo 3 não está condicionado ao decimal detectado"
    )


def test_the_symmetric_case_also_passes() -> None:
    """Simétrico do anterior: `,`/`.` com três casas depois do ponto.

    Sem esta metade, alguém poderia condicionar o passo 3 só num sentido e
    passar o teste acima por acidente.
    """
    dialect = sniff_dialect(FIXTURES / "tiny_clean.csv")
    assert dialect.delimiter == ","
    assert dialect.decimal == "."
    assert THOUSANDS_CODE not in _codes(dialect)


def test_real_thousands_corruption_is_rejected() -> None:
    """O defeito que o laboratório relata dispara `error`."""
    dialect = sniff_dialect(FIXTURES / "corrupt_thousands.csv")

    assert THOUSANDS_CODE in _codes(dialect)
    issue = next(i for i in dialect.issues if i.code == THOUSANDS_CODE)
    assert issue.severity == "error"


def test_corruption_is_reported_not_repaired() -> None:
    """Rejeitado, não corrigido.

    A correção é ambígua e um palpite errado corrompe o dado em silêncio, que é
    exatamente o problema original. O sniffer reporta o candidato detectado e
    não expõe valor consertado.
    """
    dialect = sniff_dialect(FIXTURES / "corrupt_thousands.csv")
    assert dialect.thousands is not None, "o candidato detectado é reportado"
    assert dialect.thousands != dialect.decimal


def test_the_undetectable_case_is_not_flagged() -> None:
    """Separador ambíguo NÃO é reportado, e isso é correto.

    `1.234` num arquivo cujo decimal é `.` pode ser um vírgula duzentos e trinta
    e quatro ou mil duzentos e trinta e quatro. Nada no arquivo distingue, então
    a especificação manda não adivinhar. Este teste existe para que ninguém "melhore" o
    detector até ele acusar o caso ambíguo — a primeira versão do fixture
    `corrupt_thousands.csv` era exatamente isto, e testava nada.
    """
    from tests.factories import base_spec, render_measurements

    path = FIXTURES / "_ambiguo.csv"
    spec = base_spec()
    spec = type(spec)(**{**spec.__dict__, "raw_overrides": {(0, 0): "1.234"}})
    path.write_text(render_measurements(spec), encoding="utf-8", newline="")
    try:
        dialect = sniff_dialect(path)
        assert dialect.decimal == "."
        assert THOUSANDS_CODE not in _codes(dialect), (
            "o detector acusou um valor ambíguo; com ponto decimal, `1.234` é "
            "indistinguível de um decimal legítimo"
        )
    finally:
        path.unlink(missing_ok=True)


def test_thousands_is_none_when_no_corruption_found() -> None:
    """Sem corrupção, `thousands` é `None` — não um caractere especulativo."""
    assert sniff_dialect(FIXTURES / "tiny_clean.csv").thousands is None


# ---------------------------------------------------------------------------
# As quatro combinações
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "delimiter", "decimal"),
    [
        ("tiny_clean.csv", ",", "."),
        ("corrupt_semicolon.csv", ";", ","),
        ("tiny_tab.csv", "\t", "."),
        ("tiny_quoted.csv", ",", "."),
    ],
)
def test_ca_1_7_identifies_four_dialect_combinations(
    fixture: str, delimiter: str, decimal: str
) -> None:
    """Vírgula/ponto, ponto-e-vírgula/vírgula, tab/ponto, e aspas."""
    dialect = sniff_dialect(FIXTURES / fixture)
    assert dialect.delimiter == delimiter, f"{fixture}: delimitador"
    assert dialect.decimal == decimal, f"{fixture}: decimal"


def test_quoted_delimiter_does_not_fool_the_count() -> None:
    """A cláusula "fora de aspas" do passo 1 tem de valer.

    `tiny_quoted.csv` tem vírgula literal dentro de campo entre aspas, então há
    mais vírgulas na linha do que campos. Uma contagem ingênua superestima e
    poderia eleger o delimitador errado num arquivo com aspas mais densas.
    """
    text = (FIXTURES / "tiny_quoted.csv").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith('"1.00_100'))
    assert line.count(",") > line.count('","') + 1, "o fixture precisa ter vírgula citada"

    assert sniff_dialect(FIXTURES / "tiny_quoted.csv").delimiter == ","


# ---------------------------------------------------------------------------
# Os arquivos reais
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["neg_c18", "neg_hilic", "pos_c18", "pos_hilic"])
def test_real_measurements_are_comma_dot_utf8_crlf(mode: str) -> None:
    """Os quatro reais: `,`/`.`, utf-8, CRLF, e zero corrupção de milhar.

    Medido antes de escrever o sniffer: os valores têm 15 dígitos significativos
    e nunca terminam em grupos de exatamente três dígitos, então nem a regra
    ingênua daria falso positivo aqui. A condicionalidade protege contra uma
    exportação arredondada, não contra estes arquivos.
    """
    dialect = sniff_dialect(measurements_path(mode))

    assert dialect.delimiter == ","
    assert dialect.decimal == "."
    assert dialect.encoding == "utf-8"
    assert dialect.line_terminator == "\r\n"
    assert THOUSANDS_CODE not in _codes(dialect)


# ---------------------------------------------------------------------------
# Bordas
# ---------------------------------------------------------------------------


def test_empty_file_does_not_raise(tmp_path: Path) -> None:
    """Arquivo vazio devolve dialeto padrão em vez de explodir.

    Quem reporta arquivo inutilizável é `validate_structure`; o sniffer não é o
    lugar de abortar.
    """
    path = tmp_path / "vazio.csv"
    path.write_text("", encoding="utf-8")
    dialect = sniff_dialect(path)
    assert dialect.delimiter == ","


def test_single_line_file_does_not_raise(tmp_path: Path) -> None:
    """Uma linha só: não há consistência entre linhas para medir."""
    path = tmp_path / "uma.csv"
    path.write_text("NAME,A,B\r\n", encoding="utf-8")
    assert sniff_dialect(path).delimiter == ","


def test_sample_bytes_limits_the_read(tmp_path: Path) -> None:
    """`sample_bytes` é respeitado — não se lê 6 MB para detectar dialeto."""
    path = tmp_path / "grande.csv"
    path.write_text("NAME,A,B\r\n" + ("x,1,2\r\n" * 20000), encoding="utf-8")

    dialect = sniff_dialect(path, sample_bytes=64)
    assert dialect.delimiter == ","
    assert dialect.bytes_read <= 64


def test_latin1_fallback_is_recorded(tmp_path: Path) -> None:
    """Passo 4 — cai para `latin-1` e registra qual encoding usou.

    Nenhum arquivo real exercita isto: os oito são ASCII puro. O fixture é
    fabricado de propósito, e é a única forma de cobrir a queda.
    """
    path = tmp_path / "latin.csv"
    path.write_bytes("NAME,Ação,B\r\nLABEL,QC,QC\r\n".encode("latin-1"))

    dialect = sniff_dialect(path)
    assert dialect.encoding == "latin-1"
    assert dialect.delimiter == ","


def test_lf_only_terminator_is_detected(tmp_path: Path) -> None:
    """Terminador LF puro é reportado como tal, não normalizado em silêncio."""
    path = tmp_path / "lf.csv"
    path.write_text("NAME,A,B\nLABEL,QC,QC\n", encoding="utf-8", newline="")
    assert sniff_dialect(path).line_terminator == "\n"


def test_dialect_is_immutable() -> None:
    """`Dialect` é frozen — dialeto detectado não muda depois."""
    import dataclasses

    dialect = sniff_dialect(FIXTURES / "tiny_clean.csv")
    with pytest.raises(dataclasses.FrozenInstanceError):
        dialect.delimiter = ";"  # type: ignore[misc]


def test_missing_file_raises_rather_than_guessing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sniff_dialect(tmp_path / "nao_existe.csv")


def test_oversized_field_does_not_crash_delimiter_detection(tmp_path: Path) -> None:
    """Campo acima do limite do `csv` descarta o candidato, não propaga o erro.

    Caminho de erro real e alcançável — `csv.Error: field larger than field
    limit (131072)` —, coberto por teste em vez de `pragma`, porque
    caminho de erro é justamente onde o teste importa. Sem o tratamento, um
    arquivo com campo gigante derrubaria a detecção de dialeto antes de qualquer
    validação poder reportá-lo.
    """
    path = tmp_path / "gigante.csv"
    oversized = '"' + "x" * 200_000 + '"'
    path.write_text(f"NAME,A,B\r\nLABEL,QC,QC\r\nF1,{oversized},3\r\n", encoding="utf-8")

    dialect = sniff_dialect(path, sample_bytes=400_000)
    assert dialect.delimiter in (",", ";", "\t", "|")


# ---------------------------------------------------------------------------
# As duas evidências léxicas fortes, e o limite estreito da ambiguidade
# ---------------------------------------------------------------------------


def _semicolon_file(tmp_path: Path, cells: list[str], name: str = "ev.csv") -> Path:
    """Arquivo `;`-delimitado com as células dadas na primeira linha de feature."""
    path = tmp_path / name
    header = "NAME;A;B;C"
    labels = "LABEL;QC;QC;NoHTLV"
    row = "F1;" + ";".join(cells)
    path.write_text(f"{header}\r\n{labels}\r\n{row}\r\n", encoding="utf-8", newline="")
    return path


def test_evidence_form_1_group_not_three_digits_resolves_the_file(tmp_path: Path) -> None:
    """Forma 1: `12,34` não pode ser milhar, logo a vírgula é decimal.

    Milhar agrupa em exatamente três dígitos. Um grupo de dois é prova de que o
    caractere é decimal, e UMA célula em qualquer posição resolve o arquivo.
    """
    dialect = sniff_dialect(_semicolon_file(tmp_path, ["1000", "12,34", "2000"]))
    assert dialect.decimal == ","
    assert dialect.decimal_evidence == "lexical"


def test_evidence_form_1_works_from_the_last_cell(tmp_path: Path) -> None:
    """A resolução é por ARQUIVO: a evidência vale de qualquer posição."""
    dialect = sniff_dialect(_semicolon_file(tmp_path, ["1,234", "2,345", "9,5"]))
    assert dialect.decimal == ","
    assert dialect.decimal_evidence == "lexical"


def test_evidence_form_2_two_separators_in_one_cell(tmp_path: Path) -> None:
    """Forma 2: `1.234.567` só se explica como milhar, logo decimal é `,`."""
    dialect = sniff_dialect(_semicolon_file(tmp_path, ["1.234.567", "1000", "2000"]))
    assert dialect.decimal == ","
    assert dialect.decimal_evidence == "lexical"


def test_only_three_digit_fractions_is_the_irreducible_case(tmp_path: Path) -> None:
    """O limite estreito: SÓ aqui a ambiguidade é irredutível.

    Todas as células com um separador e exatamente três dígitos depois. Nenhuma
    das duas formas dispara, o arquivo inteiro foi varrido, e o decimal cai no
    padrão — marcado `default` para que quem leia o dialeto saiba que a base é
    fraca. Declarar ambiguidade **antes** de varrer é desistir cedo.
    """
    dialect = sniff_dialect(_semicolon_file(tmp_path, ["1,234", "2,345", "3,456"]))
    assert dialect.decimal_evidence == "default"


def test_integers_only_is_also_irreducible(tmp_path: Path) -> None:
    """Sem nenhum separador não há o que resolver — e o padrão é declarado."""
    dialect = sniff_dialect(_semicolon_file(tmp_path, ["1000", "2000", "3000"]))
    assert dialect.decimal_evidence == "default"


def test_comma_delimited_file_is_resolved_structurally(tmp_path: Path) -> None:
    """Num arquivo `,`-delimitado a vírgula não pode ser decimal.

    Não há evidência léxica a buscar: o delimitador elimina o candidato. É o
    desfecho mais forte dos três, e o dos quatro arquivos reais.
    """
    dialect = sniff_dialect(FIXTURES / "tiny_clean.csv")
    assert dialect.decimal == "."
    assert dialect.decimal_evidence == "structural"


def test_ptbr_fixture_is_resolved_lexically_not_by_default(tmp_path: Path) -> None:
    """`corrupt_semicolon.csv` tem a evidência que um pt-BR real tem.

    Se caísse em `default`, o decimal viraria `.` e `1,234` seria acusado de
    corrupção — reintroduzindo o falso positivo que a condicionalidade ao decimal existe para
    impedir. A primeira versão do fixture tinha só frações de 3 dígitos e caía nisso.
    """
    dialect = sniff_dialect(FIXTURES / "corrupt_semicolon.csv")
    assert dialect.decimal == ","
    assert dialect.decimal_evidence == "lexical"


@pytest.mark.parametrize("mode", ["neg_c18", "neg_hilic", "pos_c18", "pos_hilic"])
def test_real_files_resolve_structurally(mode: str) -> None:
    """Os quatro reais são `,`-delimitados: nenhuma ambiguidade a resolver."""
    assert sniff_dialect(measurements_path(mode)).decimal_evidence == "structural"

"""Detecção de dialeto de CSV: delimitador, decimal, separador de milhar e encoding.

A ordem dos passos importa. A detecção de corrupção por separador de milhar
depende do decimal já determinado: sem ele, um arquivo pt-BR legítimo seria
rejeitado porque `1,234` casa o padrão de milhar da vírgula.
"""

import codecs
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from refino.io.issues import ValidationIssue

# Candidatos a delimitador, na ordem em que são testados.
DELIMITER_CANDIDATES = (",", ";", "\t", "|")

# Linhas amostradas para medir consistência de contagem de campos.
SAMPLE_LINES = 50

# Encodings tentados, nesta ordem.
ENCODINGS = ("utf-8", "latin-1")

# Um grupo de milhar é exatamente três dígitos, repetido, precedido de 1 a 3.
# O caractere é interpolado na detecção, um de cada vez.
_THOUSANDS_TEMPLATE = r"^\d{{1,3}}(?:{sep}\d{{3}})+$"


@dataclass(frozen=True)
class Dialect:
    """Dialeto detectado de um arquivo CSV.

    `thousands` é o caractere candidato a separador de milhar quando corrupção
    foi detectada, e `None` caso contrário.

    `decimal_evidence` diz como o decimal foi resolvido: `structural` (o
    delimitador elimina um candidato), `lexical` (uma célula inequívoca resolveu
    o arquivo) ou `default` (as células amostradas não trazem evidência forte).
    Só no `default` a detecção de milhar pode calar por falta de base, e por isso
    o campo é exposto.

    `issues` carrega o que a detecção descobriu, hoje apenas
    `THOUSANDS_SEPARATOR_CORRUPTION`.
    """

    delimiter: str
    decimal: str
    thousands: str | None
    encoding: str
    line_terminator: str
    bytes_read: int = 0
    decimal_evidence: str = "structural"
    issues: tuple[ValidationIssue, ...] = field(default=())


def _read_sample(path: Path, sample_bytes: int) -> tuple[str, str, int]:
    """Lê o início do arquivo. Devolve (texto, encoding, bytes lidos).

    Tenta `utf-8` e cai para `latin-1`, que mapeia todo byte e por isso nunca
    falha.
    """
    with path.open("rb") as handle:
        raw = handle.read(sample_bytes)
        # Até três bytes além da fatia: o resto de um caractere UTF-8 que o corte tenha partido.
        seguinte = handle.read(3)

    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding), encoding, len(raw)
        except UnicodeDecodeError as erro:
            if encoding == "utf-8" and _o_corte_partiu_um_caractere(raw, seguinte):
                return raw[: erro.start].decode("utf-8"), "utf-8", len(raw)
            continue
    # Inalcançável: latin-1 decodifica qualquer byte. Mantido para que a
    # função tenha retorno total.
    return raw.decode("latin-1", errors="replace"), "latin-1", len(raw)  # pragma: no cover


def _o_corte_partiu_um_caractere(raw: bytes, seguinte: bytes) -> bool:
    """Diz se a fatia só é inválida porque o corte partiu um caractere UTF-8 ao meio.

    A leitura para em `sample_bytes`, e um caractere UTF-8 de dois a quatro bytes que atravesse
    esse byte deixa a fatia inválida num arquivo válido. Sem esta pergunta o arquivo inteiro
    cairia para `latin-1` e todo nome acentuado sairia trocado.

    A resposta vem dos bytes que completam o caractere, e não só da forma da cauda: `0xE9` é o
    `é` do `latin-1` e também o primeiro byte de um caractere UTF-8 de três. O decodificador
    incremental guarda a cauda incompleta sem acusar, e os bytes seguintes dizem se ela se
    completa; no fim do arquivo, `final=True` acusa o caractere que não termina.
    """
    decodificador = codecs.getincrementaldecoder("utf-8")()
    try:
        decodificador.decode(raw)
        decodificador.decode(seguinte, final=len(seguinte) < 3)
    except UnicodeDecodeError:
        return False
    return True


def _detect_line_terminator(text: str) -> str:
    """CRLF quando predomina; LF caso contrário.

    O texto não é normalizado. O terminador é registrado para quem escreve
    saber o que foi lido: os arquivos do laboratório são CRLF.
    """
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf >= lf and crlf > 0 else "\n"


def _field_counts(text: str, delimiter: str) -> list[int]:
    """Contagem de campos por linha, respeitando aspas.

    Usa o `csv` da biblioteca padrão em vez de `str.count`: contar fora de
    aspas impede que uma vírgula dentro de campo citado eleja o delimitador
    errado.
    """
    reader = csv.reader(text.splitlines()[:SAMPLE_LINES], delimiter=delimiter)
    try:
        return [len(row) for row in reader if row]
    except csv.Error:
        return []


def _score(counts: list[int]) -> tuple[int, int]:
    """Ordena candidatos por consistência, depois por número de campos.

    O delimitador correto é o que produz a mesma contagem de campos em todas
    as linhas. Frequência elegeria a vírgula num arquivo delimitado por `;` e
    cheio de decimais com vírgula.
    """
    if not counts:
        return (0, 0)
    modal = max(set(counts), key=counts.count)
    if modal < 2:
        return (0, 0)
    return (counts.count(modal), modal)


def _detect_delimiter(text: str) -> str:
    """Escolhe o candidato com contagem de campos mais consistente."""
    best = ","
    best_score = (0, 0)
    for candidate in DELIMITER_CANDIDATES:
        score = _score(_field_counts(text, candidate))
        if score > best_score:
            best, best_score = candidate, score
    return best


def _data_cells(text: str, delimiter: str) -> list[str]:
    """Células que deveriam ser numéricas: linha 3 em diante, coluna 2 em diante.

    As duas primeiras linhas são `NAME` e `LABEL`, e a primeira coluna é nome
    de feature: todas texto por contrato. Incluí-las faria a detecção de decimal
    medir pontuação de rótulo.
    """
    try:
        rows = list(csv.reader(text.splitlines()[:SAMPLE_LINES], delimiter=delimiter))
    except csv.Error:
        # Mesma guarda de `_field_counts`: um campo acima do limite do `csv`
        # não pode derrubar `sniff_dialect`. Sem células, o decimal cai no
        # padrão e a detecção de milhar não acusa nada; quem reporta arquivo
        # inutilizável é `validate_structure`.
        return []
    return [cell.strip() for row in rows[2:] for cell in row[1:] if cell.strip()]


def _strong_evidence_for_decimal(cells: list[str], candidate: str, other: str) -> str | None:
    """Procura evidência léxica forte de que `candidate` é o decimal.

    Basta uma célula, em qualquer posição da amostra, para resolver o arquivo
    inteiro. Há duas formas, ambas inequívocas:

    1. Grupo de tamanho diferente de 3 depois de `candidate`. `12.34` não pode
       ser separador de milhar, porque milhar agrupa em exatamente três dígitos.
       Logo o ponto é decimal.
    2. Duas ou mais ocorrências de `other` na mesma célula. `1.234.567` só se
       explica como agrupamento de milhar, logo o ponto é milhar e a vírgula é
       o decimal.

    Devolve o decimal quando encontra evidência, `None` quando não encontra.
    """
    esc_cand = re.escape(candidate)
    esc_other = re.escape(other)

    # Forma 1: candidate seguido de grupo cujo tamanho não é 3.
    non_thousands = re.compile(rf"^-?\d+{esc_cand}\d+$")
    for cell in cells:
        if other in cell:
            continue  # célula mista: forma 2 resolve, não esta
        if non_thousands.match(cell):
            fraction = cell.split(candidate)[-1]
            if len(fraction) != 3:
                return candidate

    # Forma 2: duas ou mais ocorrências de `other`, que é agrupamento de milhar.
    for cell in cells:
        if cell.count(other) >= 2 and re.fullmatch(rf"^-?[\d{esc_other}]+$", cell):
            return candidate

    return None


def _detect_decimal(cells: list[str], delimiter: str) -> tuple[str, str]:
    """Resolve o decimal do arquivo. Devolve (decimal, como foi resolvido).

    Três desfechos, em ordem de força:

    - `structural`: o delimitador elimina um dos candidatos. Num arquivo
      delimitado por vírgula ela não pode ser decimal.
    - `lexical`: uma célula inequívoca em qualquer posição resolve o arquivo
      inteiro, por `_strong_evidence_for_decimal`.
    - `default`: as células amostradas não trazem evidência forte, e o padrão
      é `.`.
    """
    candidates = [c for c in (".", ",") if c != delimiter]
    if len(candidates) == 1:
        return candidates[0], "structural"

    for candidate in candidates:
        other = next(c for c in candidates if c != candidate)
        if _strong_evidence_for_decimal(cells, candidate, other) is not None:
            return candidate, "lexical"

    return ".", "default"


def _detect_thousands_corruption(
    cells: list[str], decimal: str
) -> tuple[str | None, tuple[ValidationIssue, ...]]:
    """Procura separador de milhar nas células, dado o decimal já detectado.

    Só o caractere que não é o decimal do arquivo é procurado. Aplicar os dois
    padrões rejeitaria `1,234` num arquivo pt-BR válido, onde aquilo é um
    vírgula duzentos e trinta e quatro.

    Não tenta consertar: a correção é ambígua e um palpite errado corrompe o
    dado em silêncio.
    """
    candidate = "," if decimal == "." else "."
    pattern = re.compile(_THOUSANDS_TEMPLATE.format(sep=re.escape(candidate)))

    offenders = [cell for cell in cells if pattern.match(cell)]
    if not offenders:
        return None, ()

    shown = ", ".join(repr(o) for o in offenders[:3])
    issue = ValidationIssue(
        severity="error",
        code="THOUSANDS_SEPARATOR_CORRUPTION",
        message=(
            f"{len(offenders)} célula(s) com {candidate!r} em posição de separador de "
            f"milhar, num arquivo cujo decimal é {decimal!r}: {shown}. "
            "A ingestão foi recusada em vez de adivinhar o valor — a correção é "
            "ambígua e um palpite errado corrompe o dado em silêncio."
        ),
        location=None,
    )
    return candidate, (issue,)


def sniff_dialect(path: Path, sample_bytes: int = 65536) -> Dialect:
    """Detecta o dialeto de um CSV a partir de uma amostra do início do arquivo.

    Levanta `FileNotFoundError` se o arquivo não existe. Arquivo vazio ou de
    uma linha devolve dialeto padrão: quem reporta arquivo inutilizável é
    `validate_structure`.
    """
    if not path.exists():
        raise FileNotFoundError(f"arquivo não encontrado: {path}")

    text, encoding, bytes_read = _read_sample(path, sample_bytes)
    terminator = _detect_line_terminator(text)
    delimiter = _detect_delimiter(text)
    cells = _data_cells(text, delimiter)
    decimal, evidence = _detect_decimal(cells, delimiter)
    thousands, issues = _detect_thousands_corruption(cells, decimal)

    return Dialect(
        delimiter=delimiter,
        decimal=decimal,
        thousands=thousands,
        encoding=encoding,
        line_terminator=terminator,
        bytes_read=bytes_read,
        decimal_evidence=evidence,
        issues=issues,
    )

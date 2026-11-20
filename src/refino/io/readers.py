"""Leitura dos formatos de entrada: `measurements`, `Sample_List` e `toserrf`.

Quatro códigos só o leitor pode emitir, porque dependem da estrutura de linhas
ou do texto de cabeçalho e não sobrevivem ao parse: `MISSING_NAME_ROW`,
`MISSING_LABEL_ROW`, `RAGGED_ROWS` e `NON_NUMERIC_CELL`. Chegam em
`doc.issues`, e `validate_structure` os incorpora.
"""

import csv
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from refino.domain.labels import Orientation, validate_label
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata, SampleType
from refino.io.documents import (
    MeasurementsDocument,
    MissingGroupsError,
    ToSerrfDocument,
)
from refino.io.issues import ValidationIssue
from refino.io.sniffer import Dialect, sniff_dialect
from refino.io.validators import (
    CANONICAL_GROUPS,
    QC_LABEL,
    canonical_group,
    group_from_name,
    validate_structure,
)

NAME_CELL = "NAME"
LABEL_CELL = "LABEL"

LabelPolicy = Literal["error", "warn", "prefer_name", "prefer_label"]

# As quatro políticas, para conferir em tempo de execução. Um valor inválido
# cairia no ramo final de `_resolve_group` e se comportaria como `prefer_label`
# sem erro, com o nome errado gravado em `provenance`.
LABEL_POLICIES: frozenset[str] = frozenset({"error", "warn", "prefer_name", "prefer_label"})

# Padrão de milhar, para distinguir célula corrompida de célula só não-numérica.
_THOUSANDS = re.compile(r"^-?\d{1,3}(?:[.,]\d{3})+$")


def _rows_unchanged(rows: list[list[str]]) -> list[list[str]]:
    """A orientação do laboratório: features em linhas, nada a fazer."""
    return rows


def _rows_transposed(rows: list[list[str]]) -> list[list[str]]:
    """Transpõe a grade crua, antes de qualquer interpretação.

    O restante do leitor lê `NAME` e `LABEL` das duas primeiras linhas; transpor
    depois de interpretar exigiria dois caminhos de parse.

    Grade irregular levanta aqui. Transposição de linhas de comprimento diferente
    não é definida, e completar com vazio apagaria `RAGGED_ROWS`: o defeito
    reapareceria como célula não numérica em posição arbitrária.
    """
    widths = {len(row) for row in rows}
    if len(widths) > 1:
        raise IngestionError(
            [
                ValidationIssue(
                    severity="error",
                    code="RAGGED_ROWS",
                    message=(
                        f"linhas com {sorted(widths)} campos; transpor grade irregular "
                        "não é definido, e completar com vazio apagaria este defeito."
                    ),
                )
            ],
            "orientation=samples-in-rows",
        )
    return [list(column) for column in zip(*rows, strict=True)]


# Uma função por valor de `Orientation`. Um teste compara as chaves com os
# valores do `Literal`, nas duas direções.
ORIENTATION_HANDLERS: dict[str, Callable[[list[list[str]]], list[list[str]]]] = {
    "features-in-rows": _rows_unchanged,
    "samples-in-rows": _rows_transposed,
}


class IngestionError(ValueError):
    """A ingestão foi abortada por um ou mais `error`.

    Carrega a lista completa de problemas, e não só o primeiro.
    """

    def __init__(self, issues: Sequence[ValidationIssue], source: str) -> None:
        self.issues = tuple(issues)
        errors = [i for i in self.issues if i.is_error]
        detail = "\n".join(f"  - {i}" for i in errors)
        super().__init__(
            f"{len(errors)} erro(s) impedem a ingestão de {source} "
            f"({len(self.issues)} problema(s) no total):\n{detail}"
        )


def _parse_number(text: str, decimal: str) -> float | None:
    """Converte respeitando o decimal do dialeto. `None` se não é número."""
    candidate = text.strip()
    if not candidate:
        return 0.0  # célula vazia é ausência, e os arquivos reais escrevem zero
    if decimal != ".":
        candidate = candidate.replace(decimal, ".")
    try:
        return float(candidate)
    except ValueError:
        return None


def read_measurements(
    path: Path,
    dialect: Dialect | None = None,
    *,
    orientation: Orientation = "features-in-rows",
) -> MeasurementsDocument:
    """Lê `measurements_{modo}.csv` e devolve o documento com features em linhas.

    `orientation` diz em que orientação o arquivo está: `samples-in-rows`
    transpõe a grade crua para a que o resto do leitor espera. Essa transposição
    é distinta da que leva à orientação interna, feita por `load_matrix`; trocar
    uma pela outra daria matriz com eixo trocado cuja soma total fecha.

    `orientation` vem da CLI, então é conferido em tempo de execução e não só
    pela anotação.
    """
    validate_label("Orientation", orientation)
    dialect = dialect or sniff_dialect(path)
    issues: list[ValidationIssue] = []

    # O "CSV UTF-8" do Excel começa por um BOM. Sem o `-sig` ele fica colado ao `NAME` da primeira
    # célula, `strip()` não o remove, e o arquivo seria recusado com `MISSING_NAME_ROW`. Sem BOM,
    # `utf-8-sig` lê exatamente como `utf-8`.
    codificacao = "utf-8-sig" if dialect.encoding == "utf-8" else dialect.encoding
    with path.open(newline="", encoding=codificacao) as handle:
        raw = list(csv.reader(handle, delimiter=dialect.delimiter))
    # As linhas vazias do fim saem; sem isso uma linha em branco depois da última feature viraria
    # `RAGGED_ROWS`. As do meio continuam a acusar, porque ali uma linha vazia é dado faltante e
    # não formatação.
    while raw and not any(celula.strip() for celula in raw[-1]):
        raw.pop()
    rows = ORIENTATION_HANDLERS[orientation](raw)

    if len(rows) < 2:
        raise IngestionError(
            [
                ValidationIssue(
                    severity="error",
                    code="MISSING_NAME_ROW",
                    message=f"arquivo tem {len(rows)} linha(s); esperava NAME, LABEL e features",
                )
            ],
            str(path),
        )

    # A primeira célula pode não existir: uma linha vazia no topo chega como `[]`, e `rows[0][0]`
    # levantaria `IndexError` sobre um arquivo que é do usuário.
    if not any(celula.strip() for celula in rows[0]):
        # Sem a linha NAME não há colunas, e cada linha seguinte viraria `RAGGED_ROWS`. A recusa
        # é uma só, e diz qual é.
        raise IngestionError(
            [
                ValidationIssue(
                    severity="error",
                    code="MISSING_NAME_ROW",
                    message=(
                        "a linha 1 está vazia; o arquivo de medições começa pela linha "
                        f"{NAME_CELL!r}. Apague as linhas em branco do topo"
                    ),
                )
            ],
            str(path),
        )
    primeira_do_nome = rows[0][0]
    primeira_do_rotulo = rows[1][0] if rows[1] else ""
    if primeira_do_nome.strip() != NAME_CELL:
        issues.append(
            ValidationIssue(
                severity="error",
                code="MISSING_NAME_ROW",
                message=(
                    f"primeira célula da linha 1 é {primeira_do_nome!r}, esperava {NAME_CELL!r}"
                ),
            )
        )
    if primeira_do_rotulo.strip() != LABEL_CELL:
        issues.append(
            ValidationIssue(
                severity="error",
                code="MISSING_LABEL_ROW",
                message=(
                    f"primeira célula da linha 2 é {primeira_do_rotulo!r}, esperava {LABEL_CELL!r}"
                ),
            )
        )

    expected_fields = len(rows[0])
    sample_names = tuple(cell.strip() for cell in rows[0][1:])
    group_labels = tuple(cell.strip() for cell in rows[1][1:])

    feature_names: list[str] = []
    raw_rows: list[tuple[str, ...]] = []
    numeric_rows: list[list[float]] = []

    for line_number, row in enumerate(rows[2:], start=3):
        if len(row) != expected_fields:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="RAGGED_ROWS",
                    message=(
                        f"linha {line_number} tem {len(row)} campos, "
                        f"o cabeçalho tem {expected_fields}"
                    ),
                    location=row[0] if row else f"linha {line_number}",
                )
            )
            continue

        feature_names.append(row[0].strip())
        cells = tuple(cell.strip() for cell in row[1:])
        raw_rows.append(cells)

        parsed: list[float] = []
        for column, cell in enumerate(cells):
            value = _parse_number(cell, dialect.decimal)
            if value is None:
                # Corrupção de milhar já foi reportada pelo sniffer com código
                # próprio; reportá-la também como NON_NUMERIC_CELL contaria o
                # mesmo defeito duas vezes.
                if not _THOUSANDS.match(cell):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            code="NON_NUMERIC_CELL",
                            message=f"célula {cell!r} não é conversível para float",
                            location=(
                                f"{row[0].strip()} / "
                                f"{sample_names[column] if column < len(sample_names) else column}"
                            ),
                        )
                    )
                parsed.append(float("nan"))
            else:
                parsed.append(value)
        numeric_rows.append(parsed)

    values = (
        np.array(numeric_rows, dtype=np.float64)
        if numeric_rows
        else np.zeros((0, len(sample_names)), dtype=np.float64)
    )

    return MeasurementsDocument(
        values=values,
        feature_names=tuple(feature_names),
        sample_names=sample_names,
        group_labels=group_labels,
        raw_cells=tuple(raw_rows),
        dialect=dialect,
        issues=tuple(issues),
    )


def read_sample_list(path: Path, dialect: Dialect | None = None) -> dict[str, int]:
    """Lê `Sample_List_{modo}.csv`. Devolve `{nome_amostra: ordem_de_injecao}`.

    É o único lugar onde a ordem temporal existe. Sem ela não há correção de
    deriva possível.
    """
    dialect = dialect or sniff_dialect(path)
    orders: dict[str, int] = {}

    with path.open(newline="", encoding=dialect.encoding) as handle:
        rows = [row for row in csv.reader(handle, delimiter=dialect.delimiter) if row]

    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) < 2:
            continue
        try:
            orders[row[1].strip()] = int(row[0].strip())
        except ValueError as exc:
            raise ValueError(
                f"{path.name} linha {line_number}: ID {row[0]!r} não é inteiro. "
                "A ordem de injeção é a única informação temporal do estudo — "
                "adivinhá-la não é opção."
            ) from exc
    return orders


def _resolve_group(
    name: str,
    label: str,
    policy: LabelPolicy,
    canonical_groups: frozenset[str],
) -> str:
    """Grupo final da amostra, conforme a política de rótulo.

    Variantes de grafia são canonicalizadas sempre (`HAM/TSP` vira `HAMTSP`),
    porque isso é normalização determinística. A política decide apenas o caso
    em que nome e `LABEL` discordam.
    """
    if label == QC_LABEL:
        return QC_LABEL

    canonical_by_key = {canonical_group(g): g for g in canonical_groups}
    canonical = canonical_by_key.get(canonical_group(label))
    normalized = canonical if canonical is not None else label

    embedded = group_from_name(name)
    if embedded is None or canonical_group(embedded) == canonical_group(label):
        return normalized

    if policy == "prefer_name":
        from_name = canonical_by_key.get(canonical_group(embedded))
        return from_name if from_name is not None else embedded
    return normalized


def load_matrix(
    measurements: Path,
    sample_list: Path,
    mode: str,
    *,
    on_label_mismatch: LabelPolicy = "prefer_name",
    canonical_groups: frozenset[str] = CANONICAL_GROUPS,
    orientation: Orientation = "features-in-rows",
) -> RefinoMatrix:
    """Caminho principal: lê, valida, transpõe, junta ordem de injeção e ordena.

    Levanta `IngestionError` com a lista completa de problemas quando há `error`.

    `on_label_mismatch` tem padrão `prefer_name`: nos dados do laboratório o nome
    da amostra é corroborado por quatro arquivos independentes por paciente,
    enquanto o `LABEL` discordante existe num único lugar.

    Avisos e informações ficam de fora do retorno; quem precisa deles chama
    `load_matrix_with_issues`, que é este mesmo caminho.
    """
    matrix, _ = load_matrix_with_issues(
        measurements,
        sample_list,
        mode,
        on_label_mismatch=on_label_mismatch,
        canonical_groups=canonical_groups,
        orientation=orientation,
    )
    return matrix


def load_matrix_with_issues(
    measurements: Path,
    sample_list: Path,
    mode: str,
    *,
    on_label_mismatch: LabelPolicy = "prefer_name",
    canonical_groups: frozenset[str] = CANONICAL_GROUPS,
    orientation: Orientation = "features-in-rows",
    rotulo: str | None = None,
) -> tuple[RefinoMatrix, tuple[ValidationIssue, ...]]:
    """`load_matrix`, devolvendo também os problemas que não recusam a matriz.

    Quando a matriz é aceita, os avisos que `validate_structure` calculou (rótulo de grupo grafado
    de outro jeito, nome que discorda da linha `LABEL`, QCs insuficientes) saem no segundo valor,
    para que a tela de validação os mostre.

    `rotulo` é o nome que a procedência e a recusa mostram, e nunca vira caminho. Ausente, é o
    nome dos dois arquivos. A API o passa porque grava o upload em nomes fixos, e a procedência
    tem de dizer o nome do arquivo do usuário.
    """
    if on_label_mismatch not in LABEL_POLICIES:
        raise ValueError(
            f"on_label_mismatch {on_label_mismatch!r} inválido; esperava um de "
            f"{sorted(LABEL_POLICIES)}. Um valor desconhecido se comportaria como "
            "`prefer_label` sem avisar, e o provenance registraria o nome errado."
        )

    doc = read_measurements(measurements, orientation=orientation)
    orders = read_sample_list(sample_list, doc.dialect)

    issues = validate_structure(doc, orders, canonical_groups=canonical_groups)

    # A severidade de LABEL_NAME_MISMATCH é do chamador, não do validador: é
    # como `prefer_name` pode ser o padrão sem perder a opção estrita. Promover a
    # `error` aqui é a única mutação de severidade que existe.
    if on_label_mismatch == "error":
        issues = [
            ValidationIssue("error", i.code, i.message, i.location)
            if i.code == "LABEL_NAME_MISMATCH"
            else i
            for i in issues
        ]

    fonte = rotulo if rotulo is not None else f"{measurements.name} + {sample_list.name}"
    if any(i.is_error for i in issues):
        raise IngestionError(issues, fonte)

    # Ordenação com desempate explícito por nome: sem isso, duas execuções
    # sobre o mesmo arquivo com ordens repetidas poderiam produzir colunas em
    # ordens diferentes.
    order_of = [(orders[name], name, index) for index, name in enumerate(doc.sample_names)]
    order_of.sort(key=lambda item: (item[0], item[1]))

    samples: list[SampleMetadata] = []
    columns: list[int] = []
    for injection_order, name, index in order_of:
        label = doc.group_labels[index]
        sample_type: SampleType = "qc" if label == QC_LABEL else "sample"
        samples.append(
            SampleMetadata(
                name=name,
                injection_order=injection_order,
                sample_type=sample_type,
                group=_resolve_group(name, label, on_label_mismatch, canonical_groups),
            )
        )
        columns.append(index)

    # Transposição para a orientação interna: amostras em linhas.
    values = np.ascontiguousarray(doc.values[:, columns].T, dtype=np.float64)

    resolved = [i for i in issues if i.code == "LABEL_NAME_MISMATCH"]
    provenance = [
        f"ingestão de {fonte}",
        f"dialeto: delimitador {doc.dialect.delimiter!r}, decimal "
        f"{doc.dialect.decimal!r}, resolvido por {doc.dialect.decimal_evidence}",
        f"orientação do arquivo: {orientation}",
        f"política de rótulo: {on_label_mismatch}"
        + (f" — {len(resolved)} divergência(s) resolvida(s)" if resolved else ""),
    ]

    matriz = RefinoMatrix(
        values=values,
        feature_names=doc.feature_names,
        samples=tuple(samples),
        mode=mode,
        provenance=tuple(provenance),
    )
    return matriz, tuple(i for i in issues if not i.is_error)


def load_matrix_issues(
    measurements: Path,
    sample_list: Path,
    *,
    canonical_groups: frozenset[str] = CANONICAL_GROUPS,
    orientation: Orientation = "features-in-rows",
) -> tuple[MeasurementsDocument, Mapping[str, int], list[ValidationIssue]]:
    """Executa a ingestão até a validação e devolve o que encontrou, sem abortar.

    Serve para diagnóstico: permite ver a lista completa de problemas de um
    arquivo que `load_matrix` recusaria.
    """
    doc = read_measurements(measurements, orientation=orientation)
    orders = read_sample_list(sample_list, doc.dialect)
    return doc, orders, validate_structure(doc, orders, canonical_groups=canonical_groups)


# ---------------------------------------------------------------------------
# Formato toserrf
# ---------------------------------------------------------------------------

TOSERRF_METADATA_ROWS = 4


def read_toserrf(path: Path, dialect: Dialect | None = None) -> ToSerrfDocument:
    """Lê o formato intermediário do SERRF e devolve o documento.

    Não devolve `RefinoMatrix` nem tenta descobrir grupo: o formato não carrega
    rótulo clínico, e a linha `label` contém nomes de amostra. Quem promove a
    matriz é `to_matrix`.

    Emite `DUPLICATE_INJECTION_TIME` quando a linha `time` repete uma posição.
    """
    dialect = dialect or sniff_dialect(path)

    with path.open(newline="", encoding=dialect.encoding) as handle:
        rows = [row for row in csv.reader(handle, delimiter=dialect.delimiter) if row]

    if len(rows) < TOSERRF_METADATA_ROWS:
        raise IngestionError(
            [
                ValidationIssue(
                    severity="error",
                    code="RAGGED_ROWS",
                    message=(
                        f"toserrf tem {len(rows)} linha(s); esperava {TOSERRF_METADATA_ROWS} "
                        "de metadado (batch, sampleType, time, label) mais as features"
                    ),
                )
            ],
            str(path),
        )

    issues: list[ValidationIssue] = []

    # A primeira coluna traz `NA` literal nas três primeiras linhas e `"No"` na
    # quarta. Nenhum dos dois é dado: são artefatos de `write.csv` do R.
    batch = tuple(cell.strip() for cell in rows[0][2:])
    sample_type = tuple(cell.strip() for cell in rows[1][2:])
    time_cells = tuple(cell.strip() for cell in rows[2][2:])
    sample_names = tuple(cell.strip() for cell in rows[3][2:])

    try:
        times = tuple(int(cell) for cell in time_cells)
    except ValueError as exc:
        # Como em `read_sample_list`, a mensagem nomeia arquivo e célula. A ordem
        # de injeção é a única informação temporal do estudo, e não se adivinha.
        offenders = [cell for cell in time_cells if not cell.lstrip("-").isdigit()]
        raise ValueError(
            f"{path.name}: a linha `time` tem célula(s) não inteira(s): "
            f"{', '.join(repr(c) for c in offenders[:5])}. A ordem de injeção é a única "
            "informação temporal do estudo — adivinhá-la não é opção."
        ) from exc

    by_time: dict[int, list[str]] = {}
    for order, name in zip(times, sample_names, strict=True):
        by_time.setdefault(order, []).append(name)
    issues.extend(
        ValidationIssue(
            severity="error",
            code="DUPLICATE_INJECTION_TIME",
            message=(
                f"ordem de injeção {order} atribuída a {len(names)} amostras "
                f"({', '.join(names)}); a ordem temporal fica ambígua"
            ),
            location=names[0],
        )
        for order, names in sorted(by_time.items())
        if len(names) > 1
    )

    feature_names: list[str] = []
    numeric_rows: list[list[float]] = []
    for row in rows[TOSERRF_METADATA_ROWS:]:
        feature_names.append(row[1].strip())
        parsed = [_parse_number(cell, dialect.decimal) for cell in row[2:]]
        numeric_rows.append([float("nan") if v is None else v for v in parsed])

    values = (
        np.array(numeric_rows, dtype=np.float64)
        if numeric_rows
        else np.zeros((0, len(sample_names)), dtype=np.float64)
    )

    return ToSerrfDocument(
        batch=batch,
        sample_type=sample_type,
        time=times,
        sample_names=sample_names,
        feature_names=tuple(feature_names),
        values=values,
        issues=tuple(issues),
    )


def to_matrix(
    doc: ToSerrfDocument,
    groups: Mapping[str, str],
    mode: str = "",
) -> RefinoMatrix:
    """Promove um `ToSerrfDocument` a `RefinoMatrix`, com os grupos fornecidos.

    `groups` é obrigatório porque é a informação que o formato não tem. Amostras
    `qc` recebem `"QC"` sem consultar o mapa.

    Levanta `MissingGroupsError` nomeando todas as amostras ausentes do mapa, e
    não só a primeira.
    """
    missing = tuple(
        name
        for name, kind in zip(doc.sample_names, doc.sample_type, strict=True)
        if kind != "qc" and name not in groups
    )
    if missing:
        raise MissingGroupsError(missing)

    order = sorted(range(len(doc.sample_names)), key=lambda i: (doc.time[i], doc.sample_names[i]))

    samples = tuple(
        SampleMetadata(
            name=doc.sample_names[i],
            injection_order=doc.time[i],
            sample_type=doc.sample_type[i],  # type: ignore[arg-type]
            group=QC_LABEL if doc.sample_type[i] == "qc" else groups[doc.sample_names[i]],
            batch=doc.batch[i],
        )
        for i in order
    )
    values = np.ascontiguousarray(doc.values[:, order].T, dtype=np.float64)

    return RefinoMatrix(
        values=values,
        feature_names=doc.feature_names,
        samples=samples,
        mode=mode,
        provenance=("promoção de ToSerrfDocument com grupos fornecidos",),
    )

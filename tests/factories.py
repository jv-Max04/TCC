"""Fixture factory.

Gera pares `measurements` + `Sample_List` com um defeito injetado, para que o
teste cubra os códigos de escopo documento sem um arquivo mantido à mão por
código. A contagem é `len(FILE_SCOPE_CODES)`, que `tests/unit/test_factories.py` confere.

Vive em `tests/` e não em `src/refino/`: fora de `--cov=src/refino`,
e não é código de produto — `src/refino/synthetic/` existe para gerar matrizes
com deriva conhecida, que é propósito diferente.

**Duas bases, deliberadamente.** A especificação define `tiny_clean.csv` com 6 amostras
e 2 QCs, e esse arquivo é gerado aqui como está escrito. Mas 2 QCs disparam
`INSUFFICIENT_QCS` (< 5), então `tiny_clean` **não** é base válida para a
factory: todo caso injetado carregaria esse aviso de fundo, e "injeta aquele
defeito e nenhum outro" seria impossível de afirmar. A base da factory tem 10
amostras e 5 QCs, e é limpa de verdade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

# Grupos canônicos da especificação.
CANONICAL_GROUPS = ("NoHTLV", "Asymptomatic", "HAMTSP", "ATLL")
QC_LABEL = "QC"

CRLF = "\r\n"


@dataclass(frozen=True)
class SampleSpec:
    """Uma coluna da matriz: nome, grupo declarado no LABEL, ordem de injeção."""

    name: str
    label: str
    injection_order: int


@dataclass(frozen=True)
class MatrixSpec:
    """Especificação completa de um par measurements/Sample_List.

    Tudo o que a factory injeta é uma mutação desta estrutura, e a serialização
    é um passo separado — o que permite injetar defeitos estruturais (linha
    ragged, cabeçalho errado) que nenhuma estrutura de dados válida expressa.
    """

    samples: tuple[SampleSpec, ...]
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]  # (n_features, n_samples)
    delimiter: str = ","
    decimal: str = "."
    name_cell: str = "NAME"
    label_cell: str = "LABEL"
    ragged_row: int | None = None  # índice da linha de feature a truncar
    raw_overrides: dict[tuple[int, int], str] = field(default_factory=dict)
    drop_from_sample_list: tuple[str, ...] = ()
    injection_order_override: dict[str, int] = field(default_factory=dict)


def _feature_name(index: int) -> str:
    """Nome no formato `RT_mz`, ex. `1.00_100.0000m/z`."""
    return f"{index + 1}.00_{(index + 1) * 100}.0000m/z"


def base_spec(n_features: int = 5, n_samples: int = 10, n_qcs: int = 5) -> MatrixSpec:
    """Base VÁLIDA da factory: zero issues, 5 QCs por padrão.

    Os QCs vêm primeiro nas colunas e as amostras são agrupadas por classe,
    espelhando os arquivos reais — onde a ordem das colunas NÃO é a ordem de
    injeção. Os QCs ficam distribuídos na ordem de injeção, como no estudo real.
    """
    if n_qcs > n_samples:
        raise ValueError(f"n_qcs={n_qcs} > n_samples={n_samples}")

    n_patients = n_samples - n_qcs
    # QCs em posições de injeção espalhadas: 1, depois a cada passo regular.
    step = max(1, n_samples // n_qcs)
    qc_orders = [1 + i * step for i in range(n_qcs)]
    qc_orders = [o for o in qc_orders if o <= n_samples][:n_qcs]
    while len(qc_orders) < n_qcs:
        candidate = max(qc_orders) + 1
        if candidate > n_samples:
            break
        qc_orders.append(candidate)
    patient_orders = [o for o in range(1, n_samples + 1) if o not in qc_orders]

    samples: list[SampleSpec] = []
    for i, order in enumerate(qc_orders, start=1):
        samples.append(SampleSpec(f"TINY_QC{i}", QC_LABEL, order))
    for i, order in enumerate(patient_orders[:n_patients]):
        group = CANONICAL_GROUPS[i % len(CANONICAL_GROUPS)]
        samples.append(SampleSpec(f"TINY_{i + 1}_ID{i + 1}_{group}", group, order))

    # Valor previsível: cada célula é (feature+1)*1000 + posição_da_coluna*10.
    values = tuple(
        tuple(float((f + 1) * 1000 + (s + 1) * 10) for s in range(len(samples)))
        for f in range(n_features)
    )
    return MatrixSpec(
        samples=tuple(samples),
        feature_names=tuple(_feature_name(f) for f in range(n_features)),
        values=values,
    )


def tiny_clean_spec() -> MatrixSpec:
    """A base da especificação, literal: 6 amostras (2 QCs), 5 features.

    NÃO é base válida da factory — 2 QCs disparam INSUFFICIENT_QCS, que é
    `warning` e esperado. Existe porque a especificação a nomeia.
    """
    return base_spec(n_features=5, n_samples=6, n_qcs=2)


# ---------------------------------------------------------------------------
# Serialização
# ---------------------------------------------------------------------------


def _format_number(value: float, decimal: str) -> str:
    text = repr(value) if value != int(value) else str(int(value))
    return text.replace(".", decimal) if decimal != "." else text


def _quote(field_text: str, delimiter: str) -> str:
    """Cita só quando necessário — como o `csv` da biblioteca padrão."""
    if any(ch in field_text for ch in (delimiter, '"', "\r", "\n")):
        return '"' + field_text.replace('"', '""') + '"'
    return field_text


def render_measurements(spec: MatrixSpec) -> str:
    """Serializa o `measurements`, com terminador CRLF como os arquivos reais."""
    d = spec.delimiter
    rows: list[list[str]] = [
        [spec.name_cell, *[s.name for s in spec.samples]],
        [spec.label_cell, *[s.label for s in spec.samples]],
    ]
    for f, feature in enumerate(spec.feature_names):
        cells = [feature]
        for s in range(len(spec.samples)):
            override = spec.raw_overrides.get((f, s))
            cells.append(
                override
                if override is not None
                else _format_number(spec.values[f][s], spec.decimal)
            )
        if spec.ragged_row == f:
            cells = cells[:-1]  # tira um campo: contagem diverge do cabeçalho
        rows.append(cells)
    return CRLF.join(d.join(_quote(c, d) for c in row) for row in rows) + CRLF


def render_sample_list(spec: MatrixSpec) -> str:
    """Serializa o `Sample_List`, ordenado por ordem de injeção."""
    entries = [
        (spec.injection_order_override.get(s.name, s.injection_order), s.name)
        for s in spec.samples
        if s.name not in spec.drop_from_sample_list
    ]
    rows = ["ID,inj_order_name"]
    rows.extend(f"{order},{name}" for order, name in sorted(entries))
    return CRLF.join(rows) + CRLF


def render_pair(spec: MatrixSpec) -> tuple[str, str]:
    """O par completo. Nenhum caso de teste usa um lado sem o outro."""
    return render_measurements(spec), render_sample_list(spec)


# ---------------------------------------------------------------------------
# Injeção de defeitos — um por código de escopo arquivo
# ---------------------------------------------------------------------------

FILE_SCOPE_CODES = (
    "MISSING_NAME_ROW",
    "MISSING_LABEL_ROW",
    "RAGGED_ROWS",
    "DUPLICATE_SAMPLE_NAME",
    "DUPLICATE_FEATURE_NAME",
    "SAMPLE_LIST_MISMATCH",
    "NO_QC_SAMPLES",
    "INSUFFICIENT_QCS",
    "NON_NUMERIC_CELL",
    "NEGATIVE_INTENSITY",
    "ALL_ZERO_FEATURE",
    "HIGH_ZERO_FRACTION",
    "THOUSANDS_SEPARATOR_CORRUPTION",
    "INJECTION_ORDER_GAP",
    "GROUP_LABEL_VARIANT",
    "LABEL_NAME_MISMATCH",
    "IMPLAUSIBLE_INTENSITY_RANGE",
    "DUPLICATE_INJECTION_TIME",
)


def inject(code: str, spec: MatrixSpec | None = None) -> MatrixSpec:
    """Devolve a base com exatamente o defeito de `code` injetado.

    Levanta ValueError para código desconhecido — inclusive para os de escopo
    estudo, que não são injetáveis numa matriz sozinha.
    """
    spec = spec or base_spec()

    if code == "MISSING_NAME_ROW":
        return replace(spec, name_cell="NOME")
    if code == "MISSING_LABEL_ROW":
        return replace(spec, label_cell="GRUPO")
    if code == "RAGGED_ROWS":
        return replace(spec, ragged_row=1)
    if code == "DUPLICATE_SAMPLE_NAME":
        samples = list(spec.samples)
        samples[-1] = replace(samples[-1], name=samples[0].name)
        return replace(spec, samples=tuple(samples))
    if code == "DUPLICATE_FEATURE_NAME":
        names = list(spec.feature_names)
        names[-1] = names[0]
        return replace(spec, feature_names=tuple(names))
    if code == "SAMPLE_LIST_MISMATCH":
        return replace(spec, drop_from_sample_list=(spec.samples[-1].name,))
    if code == "NO_QC_SAMPLES":
        samples = tuple(
            replace(s, label=CANONICAL_GROUPS[0], name=s.name.replace("QC", "P"))
            if s.label == QC_LABEL
            else s
            for s in spec.samples
        )
        return replace(spec, samples=samples)
    if code == "INSUFFICIENT_QCS":
        return base_spec(n_features=len(spec.feature_names), n_samples=10, n_qcs=2)
    if code == "NON_NUMERIC_CELL":
        return replace(spec, raw_overrides={**spec.raw_overrides, (0, 0): "n/d"})
    if code == "NEGATIVE_INTENSITY":
        values = [list(row) for row in spec.values]
        values[0][0] = -1.0
        return replace(spec, values=tuple(tuple(r) for r in values))
    if code == "ALL_ZERO_FEATURE":
        values = [list(row) for row in spec.values]
        values[0] = [0.0] * len(spec.samples)
        return replace(spec, values=tuple(tuple(r) for r in values))
    if code == "HIGH_ZERO_FRACTION":
        # Zera as duas primeiras features de cinco: 40% > corte de 30%.
        values = [list(row) for row in spec.values]
        for f in range(max(1, (len(values) * 2) // 5)):
            values[f] = [0.0] * len(spec.samples)
        return replace(spec, values=tuple(tuple(r) for r in values))
    if code == "THOUSANDS_SEPARATOR_CORRUPTION":
        # A base é ","/"." — então o separador de milhar DETECTÁVEL é a vírgula,
        # que o serializador cita automaticamente por conter o delimitador.
        # Pôr "1.234" aqui seria indetectável por construção: com ponto decimal
        # aquilo é ambíguo, e não se adivinha.
        return replace(spec, raw_overrides={**spec.raw_overrides, (0, 0): "1,234"})
    if code == "INJECTION_ORDER_GAP":
        last = spec.samples[-1]
        return replace(spec, injection_order_override={last.name: len(spec.samples) + 5})
    if code == "GROUP_LABEL_VARIANT":
        # LABEL grafa HAM-TSP; o canônico é HAMTSP. Colapsa, texto difere.
        samples = tuple(
            replace(s, label="HAM-TSP") if s.label == "HAMTSP" else s for s in spec.samples
        )
        return replace(spec, samples=samples)
    if code == "DUPLICATE_INJECTION_TIME":
        # Duas amostras na mesma posição temporal: a ordem fica ambígua e nenhum
        # método batch-aware pode modelar deriva sobre isso.
        first, second = spec.samples[0], spec.samples[-1]
        return replace(spec, injection_order_override={second.name: first.injection_order})
    if code == "IMPLAUSIBLE_INTENSITY_RANGE":
        # Mediana ~1e-2, muito abaixo do piso de 1e2 para LC-MS. A causa típica
        # no mundo real é dialeto mal resolvido; aqui o valor é implausível de
        # forma direta, para que o teste isole o código.
        values = tuple(tuple(v / 1e6 for v in row) for row in spec.values)
        return replace(spec, values=values)
    if code == "LABEL_NAME_MISMATCH":
        # LABEL diz Assyntomatic; o nome diz Asymptomatic. NÃO colapsa.
        samples = list(spec.samples)
        for i, s in enumerate(samples):
            if s.label == "Asymptomatic":
                samples[i] = replace(s, label="Assyntomatic")
                break
        return replace(spec, samples=tuple(samples))

    raise ValueError(
        f"código não injetável num measurements: {code!r}. "
        f"Códigos de escopo estudo (GROUP_LABEL_INCONSISTENT_ACROSS_MODES, "
        f"PATIENT_GROUP_CONFLICT) exigem várias matrizes — ver U9. "
        f"Injetáveis: {FILE_SCOPE_CODES}"
    )


def inject_both_label_codes(spec: MatrixSpec | None = None) -> MatrixSpec:
    """Amostra que dispara os DOIS códigos de rótulo ao mesmo tempo.

    Ausente dos dados reais, possível por construção. `LABEL` grafa `HAM-TSP`
    (variante de `HAMTSP`) enquanto o nome diz `ATLL`: colapsa num membro do
    conjunto canônico com texto diferente (`GROUP_LABEL_VARIANT`) **e**
    discorda do nome após canonicalização (`LABEL_NAME_MISMATCH`).
    """
    spec = spec or base_spec()
    samples = list(spec.samples)
    for i, s in enumerate(samples):
        if s.label == "ATLL":
            samples[i] = replace(s, label="HAM-TSP")
            break
    else:  # pragma: no cover - a base sempre tem ATLL
        raise AssertionError("base sem amostra ATLL")
    return replace(spec, samples=tuple(samples))


def spec_without_group_in_name(spec: MatrixSpec | None = None) -> MatrixSpec:
    """Amostras cujos nomes não expõem grupo — sem o sufixo `_ID<n>_`."""
    spec = spec or base_spec()
    samples = tuple(replace(s, name=re.sub(r"_ID\d+_.*$", "_ANON", s.name)) for s in spec.samples)
    return replace(spec, samples=samples)

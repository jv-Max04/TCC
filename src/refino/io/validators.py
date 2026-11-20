"""Validação estrutural das matrizes de entrada.

`validate_structure` cobre o que uma matriz sozinha permite concluir, e
`validate_study` o que exige ver os quatro modos juntos. Nenhuma das duas aborta
por defeito dos dados: devolvem a lista completa de problemas, e quem decide
abortar é `load_matrix`.

Quatro códigos dependem do texto bruto do cabeçalho ou da contagem de campos
por linha (`MISSING_NAME_ROW`, `MISSING_LABEL_ROW`, `RAGGED_ROWS` e
`NON_NUMERIC_CELL`) e não são deriváveis de um `MeasurementsDocument` já
parseado. Quem lê os emite em `doc.issues`, e `validate_structure` os incorpora
para que o chamador receba uma lista só.
"""

import csv
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from refino.domain.matrix import RefinoMatrix
from refino.io.documents import MeasurementsDocument
from refino.io.issues import ValidationIssue

# Conjunto canônico de rótulos clínicos do estudo. É parâmetro das validações,
# com este padrão: outro estudo tem outros grupos, e `GROUP_LABEL_VARIANT`
# precisa de um alvo declarado, em vez de derivar a norma dos próprios dados
# que está validando.
CANONICAL_GROUPS: frozenset[str] = frozenset({"NoHTLV", "Asymptomatic", "HAMTSP", "ATLL"})
QC_LABEL = "QC"

MIN_QCS = 5
MAX_ZERO_FRACTION = 0.30

# Faixa de intensidade esperada para LC-MS.
MIN_PLAUSIBLE_INTENSITY = 1e2
MAX_PLAUSIBLE_INTENSITY = 1e9

# Grupo embutido no nome da amostra: sufixo após `_ID<n>_`.
_GROUP_IN_NAME = re.compile(r"_ID\d+_(.+)$")


def canonical_group(label: str) -> str:
    """Forma canônica de um rótulo: sem não-alfanuméricos, em caixa baixa.

    `"HAM/TSP"`, `"HAM-TSP"` e `"HAMTSP"` colapsam em `"hamtsp"`.

    Não corrige erro de digitação: `"Assyntomatic"` e `"Asymptomatic"`
    permanecem distintos. É essa distinção que separa `GROUP_LABEL_VARIANT`
    (notação, normalizável) de `LABEL_NAME_MISMATCH` (dado errado em um dos dois
    lugares).
    """
    return re.sub(r"[^a-z0-9]", "", label.lower())


def group_from_name(sample_name: str) -> str | None:
    """Grupo embutido no nome da amostra, ou `None` se o nome não o expõe.

    `None` não é divergência: ausência de informação não é conflito. Os QCs
    caem aqui, porque o nome deles não tem `_ID<n>_`, e por isso nenhum dos dois
    códigos de rótulo dispara neles.
    """
    match = _GROUP_IN_NAME.search(sample_name)
    return match.group(1) if match else None


def _duplicates(names: tuple[str, ...]) -> list[str]:
    return sorted(name for name, count in Counter(names).items() if count > 1)


def _check_duplicate_names(doc: MeasurementsDocument) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for kind, names, code in (
        ("amostra", doc.sample_names, "DUPLICATE_SAMPLE_NAME"),
        ("feature", doc.feature_names, "DUPLICATE_FEATURE_NAME"),
    ):
        issues.extend(
            ValidationIssue(
                severity="error",
                code=code,
                message=f"nome de {kind} repetido: {name!r}",
                location=name,
            )
            for name in _duplicates(names)
        )
    return issues


def _check_sample_list(
    doc: MeasurementsDocument, injection_orders: Mapping[str, int] | None
) -> list[ValidationIssue]:
    """`SAMPLE_LIST_MISMATCH` nos dois sentidos, e a lacuna de ordem."""
    if injection_orders is None:
        return []

    issues: list[ValidationIssue] = []
    in_matrix = set(doc.sample_names)
    in_list = set(injection_orders)

    issues.extend(
        ValidationIssue(
            severity="error",
            code="SAMPLE_LIST_MISMATCH",
            message=f"amostra {name!r} está na matriz mas não na lista de injeção",
            location=name,
        )
        for name in sorted(in_matrix - in_list)
    )
    issues.extend(
        ValidationIssue(
            severity="error",
            code="SAMPLE_LIST_MISMATCH",
            message=f"amostra {name!r} está na lista de injeção mas não na matriz",
            location=name,
        )
        for name in sorted(in_list - in_matrix)
    )

    # Duplicata e lacuna têm severidades diferentes. Uma lacuna ainda permite
    # ordenar, porque as posições são distintas; duas amostras na mesma posição
    # tornam a ordem ambígua e inviabilizam qualquer método que dependa dela.
    by_order: dict[int, list[str]] = {}
    for name in sorted(in_matrix & in_list):
        by_order.setdefault(injection_orders[name], []).append(name)
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
        for order, names in sorted(by_order.items())
        if len(names) > 1
    )

    # A lacuna é medida sobre as amostras presentes nos dois lados: uma ausência
    # já reportada acima não deve virar também um aviso de lacuna.
    shared = sorted(injection_orders[n] for n in in_matrix & in_list)
    if shared and shared != list(range(min(shared), min(shared) + len(shared))):
        issues.append(
            ValidationIssue(
                severity="warning",
                code="INJECTION_ORDER_GAP",
                message=(
                    f"ordens de injeção não formam sequência contígua: {len(shared)} "
                    f"amostras cobrindo de {min(shared)} a {max(shared)}"
                ),
            )
        )
    return issues


def _check_qcs(doc: MeasurementsDocument) -> list[ValidationIssue]:
    n_qcs = sum(1 for label in doc.group_labels if label == QC_LABEL)
    if n_qcs == 0:
        return [
            ValidationIssue(
                severity="error",
                code="NO_QC_SAMPLES",
                message=(
                    f"nenhuma amostra com grupo {QC_LABEL!r}. Os QCs são a régua do "
                    "sistema — sem eles não há como medir erro técnico nem corrigir deriva"
                ),
            )
        ]
    if n_qcs < MIN_QCS:
        return [
            ValidationIssue(
                severity="warning",
                code="INSUFFICIENT_QCS",
                message=(
                    f"{n_qcs} QCs, menos que os {MIN_QCS} mínimos — correção de deriva "
                    "fica frágil e toda estatística baseada em QC herda a fragilidade"
                ),
            )
        ]
    return []


def _check_values(doc: MeasurementsDocument) -> list[ValidationIssue]:
    """Códigos derivados dos valores numéricos, não do texto."""
    issues: list[ValidationIssue] = []
    values = doc.values

    negatives = int(np.count_nonzero(values < 0))
    if negatives:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="NEGATIVE_INTENSITY",
                message=(
                    f"{negatives} valor(es) negativo(s) — intensidade negativa é "
                    "fisicamente impossível e indica erro de processamento a montante"
                ),
            )
        )

    issues.extend(
        ValidationIssue(
            severity="info",
            code="ALL_ZERO_FEATURE",
            message="feature zerada em todas as amostras",
            location=doc.feature_names[int(index)],
        )
        for index in np.flatnonzero(~values.any(axis=1))
    )

    if values.size:
        zero_fraction = float(np.count_nonzero(values == 0) / values.size)
        if zero_fraction > MAX_ZERO_FRACTION:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="HIGH_ZERO_FRACTION",
                    message=(
                        f"{zero_fraction:.1%} das células são zero, acima do corte de "
                        f"{MAX_ZERO_FRACTION:.0%}"
                    ),
                )
            )
    return issues


def _check_intensity_range(doc: MeasurementsDocument) -> list[ValidationIssue]:
    """Emite `IMPLAUSIBLE_INTENSITY_RANGE` quando a mediana foge da faixa de LC-MS.

    Fica aqui e não no sniffer porque a camada léxica não sabe o que é uma
    intensidade de LC-MS.

    Reporta as duas leituras possíveis e a razão entre elas, e não decide. A
    causa típica é dialeto mal resolvido: um decimal lido como milhar desloca a
    magnitude por mil. Decidir sozinho seria adivinhar a magnitude de um dado
    científico por heurística.
    """
    nonzero = doc.values[doc.values != 0]
    if nonzero.size == 0:
        return []

    # np.median direto sobre o array, sem converter as células para lista.
    median = float(np.median(nonzero))
    if MIN_PLAUSIBLE_INTENSITY <= median <= MAX_PLAUSIBLE_INTENSITY:
        return []

    factor = 1000.0 if median < MIN_PLAUSIBLE_INTENSITY else 0.001
    alternative = median * factor
    verdict = (
        "ESTÁ"
        if MIN_PLAUSIBLE_INTENSITY <= alternative <= MAX_PLAUSIBLE_INTENSITY
        else ("também não está")
    )

    return [
        ValidationIssue(
            severity="warning",
            code="IMPLAUSIBLE_INTENSITY_RANGE",
            message=(
                f"mediana das intensidades não-nulas é {median:.4g}, fora da faixa "
                f"esperada para LC-MS ({MIN_PLAUSIBLE_INTENSITY:.0e} a "
                f"{MAX_PLAUSIBLE_INTENSITY:.0e}). Leitura alternativa, se o dialeto "
                f"trocou decimal por separador de milhar: {alternative:.4g} (razão "
                f"{factor:g}), que {verdict} na faixa. Dialeto detectado: delimitador "
                f"{doc.dialect.delimiter!r}, decimal {doc.dialect.decimal!r}, resolvido "
                f"por {doc.dialect.decimal_evidence}. A escolha entre as duas leituras "
                "é do analista — não há decisão automática aqui."
            ),
        )
    ]


def _check_group_labels(
    doc: MeasurementsDocument, canonical_groups: frozenset[str]
) -> list[ValidationIssue]:
    """Os dois códigos de rótulo, que são testes independentes.

    `GROUP_LABEL_VARIANT` compara o `LABEL` contra o conjunto canônico declarado.
    `LABEL_NAME_MISMATCH` compara o `LABEL` contra o grupo embutido no nome.
    Uma amostra pode receber os dois.

    `LABEL_NAME_MISMATCH` sai com severidade `warning`; `load_matrix` a promove a
    `error` quando `on_label_mismatch="error"`.
    """
    canonical_by_key = {canonical_group(g): g for g in canonical_groups}
    issues: list[ValidationIssue] = []

    # A guarda vem antes do `zip(strict=True)` por causa da mensagem. Um CSV com linha mais curta
    # que o cabeçalho, comum em arquivo que passou por planilha, faria o `strict=True` levantar um
    # erro que não diz qual arquivo está desalinhado nem em quanto.
    if len(doc.sample_names) != len(doc.group_labels):
        raise ValueError(
            f"o arquivo de medições tem {len(doc.sample_names)} amostras e "
            f"{len(doc.group_labels)} rótulos de grupo. Os dois têm de vir na mesma quantidade — "
            "normalmente isto é uma linha mais curta que o cabeçalho, ou uma coluna a mais."
        )

    for name, label in zip(doc.sample_names, doc.group_labels, strict=True):
        if label == QC_LABEL:
            continue

        key = canonical_group(label)
        canonical = canonical_by_key.get(key)
        if canonical is not None and label != canonical:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="GROUP_LABEL_VARIANT",
                    message=(
                        f"rótulo {label!r} colapsa no grupo canônico {canonical!r} mas "
                        f"grafa diferente; normalizado para {canonical!r}"
                    ),
                    location=name,
                )
            )

        embedded = group_from_name(name)
        if embedded is not None and canonical_group(embedded) != key:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="LABEL_NAME_MISMATCH",
                    message=(
                        f"a linha LABEL diz {label!r} e o nome da amostra diz "
                        f"{embedded!r}; não colapsam sob canonicalização, então um dos "
                        "dois está errado"
                    ),
                    location=name,
                )
            )
    return issues


def validate_structure(
    doc: MeasurementsDocument,
    injection_orders: Mapping[str, int] | None = None,
    *,
    canonical_groups: frozenset[str] = CANONICAL_GROUPS,
) -> list[ValidationIssue]:
    """Aplica as validações que uma matriz sozinha permite.

    Devolve a lista completa de problemas, e não só o primeiro; quem decide
    abortar é `load_matrix`. Só levanta para conjunto canônico não injetivo ou
    para documento com amostras e rótulos em quantidades diferentes.

    Incorpora `doc.issues` e `doc.dialect.issues`: os códigos que dependem da
    estrutura de linhas e do dialeto são emitidos por quem leu o arquivo, e o
    chamador recebe uma lista só.
    """
    verify_canonical_map(canonical_groups)
    return [
        *doc.issues,
        *doc.dialect.issues,
        *_check_duplicate_names(doc),
        *_check_sample_list(doc, injection_orders),
        *_check_qcs(doc),
        *_check_values(doc),
        *_check_intensity_range(doc),
        *_check_group_labels(doc, canonical_groups),
    ]


# ---------------------------------------------------------------------------
# Conferência do conjunto canônico
# ---------------------------------------------------------------------------


class NonInjectiveCanonicalMapError(ValueError):
    """Dois membros do conjunto canônico colapsam no mesmo destino.

    Isso cegaria `PATIENT_GROUP_CONFLICT` sem que nada acusasse: se `ATLL` e
    `HAMTSP` canonicalizassem igual, um paciente classificado como um num modo e
    como outro em outro modo deixaria de ser conflito antes de `validate_study`
    olhar, e zero conflitos deixaria de significar ausência de conflito.
    """

    def __init__(self, collisions: Mapping[str, tuple[str, ...]]) -> None:
        self.collisions = dict(collisions)
        detail = "; ".join(f"{key!r} <- {list(names)}" for key, names in collisions.items())
        super().__init__(
            "o mapa canônico não é injetivo: "
            f"{detail}. Dois grupos clínicos distintos que colapsam no mesmo destino "
            "tornam PATIENT_GROUP_CONFLICT cego — o conflito desaparece antes da "
            "validação vê-lo, e o zero deixa de significar ausência de conflito."
        )


def verify_canonical_map(canonical_groups: frozenset[str]) -> None:
    """Afirma que `canonical_group` é injetiva sobre o conjunto declarado.

    Chamada na entrada de `validate_structure` e `validate_study`. Levanta
    `NonInjectiveCanonicalMapError` se dois grupos colapsam no mesmo destino.
    """
    by_key: dict[str, list[str]] = {}
    for group in sorted(canonical_groups):
        by_key.setdefault(canonical_group(group), []).append(group)

    collisions = {key: tuple(names) for key, names in by_key.items() if len(names) > 1}
    if collisions:
        raise NonInjectiveCanonicalMapError(collisions)


# ---------------------------------------------------------------------------
# Escopo estudo
# ---------------------------------------------------------------------------


def _patient_id(sample_name: str) -> str | None:
    """Identidade do paciente: o `ID<n>` do nome. `None` para QC.

    Um QC não é paciente: é uma mistura de alíquotas de todas as amostras, e o
    mesmo nome-base aparece nos quatro modos sem que isso signifique identidade
    de sujeito.
    """
    match = re.search(r"_ID(\d+)_", sample_name)
    return f"ID{match.group(1)}" if match else None


def validate_study(
    matrices: Mapping[str, "RefinoMatrix"],
    *,
    canonical_groups: frozenset[str] = CANONICAL_GROUPS,
) -> list[ValidationIssue]:
    """Validações que exigem ver o estudo inteiro, as quatro matrizes juntas.

    - `GROUP_LABEL_INCONSISTENT_ACROSS_MODES` (warning): duas matrizes grafam o
      mesmo grupo canônico de formas diferentes.
    - `PATIENT_GROUP_CONFLICT` (error): o mesmo paciente tem grupos canônicos
      diferentes em modos diferentes.

    Nenhum dos dois é alcançável por `validate_structure`, que recebe uma matriz
    só. A canonicalização normaliza grafias do mesmo grupo e não funde grupos
    distintos, e `prefer_name` resolve `LABEL` contra nome dentro de um arquivo,
    nunca entre arquivos: um paciente `Asymptomatic` no `pos_hilic` e `ATLL` no
    `neg_c18` passaria pela ingestão sem nenhum issue.

    Não levanta por defeito dos dados; devolve a lista, como
    `validate_structure`.
    """
    verify_canonical_map(canonical_groups)
    issues: list[ValidationIssue] = []

    spellings: dict[str, set[str]] = {}
    patient_groups: dict[str, dict[str, str]] = {}

    for mode, matrix in sorted(matrices.items()):
        for sample in matrix.samples:
            if sample.group == QC_LABEL:
                continue
            spellings.setdefault(canonical_group(sample.group), set()).add(sample.group)

            patient = _patient_id(sample.name)
            if patient is not None:
                patient_groups.setdefault(patient, {})[mode] = canonical_group(sample.group)

    issues.extend(
        ValidationIssue(
            severity="warning",
            code="GROUP_LABEL_INCONSISTENT_ACROSS_MODES",
            message=(
                f"o grupo canônico {key!r} é grafado de {len(variants)} formas no estudo: "
                f"{sorted(variants)}"
            ),
            location=key,
        )
        for key, variants in sorted(spellings.items())
        if len(variants) > 1
    )

    issues.extend(
        ValidationIssue(
            severity="error",
            code="PATIENT_GROUP_CONFLICT",
            message=(
                f"paciente {patient} tem grupos diferentes entre modos: "
                + ", ".join(f"{mode}={group}" for mode, group in sorted(by_mode.items()))
                + ". Um paciente com duas classes clínicas invalida qualquer análise a jusante"
            ),
            location=patient,
        )
        for patient, by_mode in sorted(patient_groups.items())
        if len(set(by_mode.values())) > 1
    )
    return issues


# ---------------------------------------------------------------------------
# Conformidade ao formato do MetaboAnalyst
# ---------------------------------------------------------------------------

# Os elementos obrigatórios do formato, transcritos em
# `docs/formato-metaboanalyst.md` com a procedência. A paridade entre este
# registro e aquele documento é verificada em
# `tests/test_metaboanalyst_conformance.py`, por teste e não por leitura em tempo
# de execução, porque `io/` não deve depender do layout do repositório.
#
# `mechanism=False` significa declaradamente não verificado. Só o `[F-6]` está
# assim: o export só produz o caso não emparelhado, que é conforme por
# construção.


@dataclass(frozen=True)
class FormatRule:
    """Um elemento obrigatório do formato, com a procedência da citação."""

    element: str
    status: str
    mechanism: bool
    summary: str


FORMAT_RULES: tuple[FormatRule, ...] = (
    FormatRule(
        element="F-1",
        status="derivado",
        mechanism=True,
        summary="arquivo `.csv` é separado por vírgula",
    ),
    FormatRule(
        element="F-2",
        status="verbatim",
        mechanism=True,
        summary="apenas valores numéricos e positivos; vazio ou NA para ausente",
    ),
    FormatRule(
        element="F-3",
        status="verbatim",
        mechanism=True,
        summary="nenhum espaço dentro de número",
    ),
    FormatRule(
        element="F-4",
        status="verbatim",
        mechanism=True,
        summary="nomes únicos, em letras inglesas, sublinhado e número",
    ),
    FormatRule(
        element="F-5",
        status="verbatim",
        mechanism=True,
        summary="rótulo de classe na segunda coluna, com amostras em linhas",
    ),
    FormatRule(
        element="F-6",
        status="verbatim",
        mechanism=False,
        summary="rótulo de classe numérico ou de caractere no caso não emparelhado",
    ),
)

_MA_CONFORMING_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_MA_MISSING = frozenset({"", "NA"})


def check_metaboanalyst_conformance(path: Path) -> list[str]:
    """Confere um arquivo exportado contra os elementos obrigatórios do formato.

    Devolve a lista de violações, cada uma nomeando o elemento que falhou. Lista
    vazia é conformidade.

    O `[F-1]` é `derivado`: confere-se que este arquivo, de extensão `.csv`, é
    separado por vírgula, sem reprovar tabulação em geral, que a especificação
    aceita. O `[F-6]` não é conferido (`mechanism=False` no registro).
    """
    violations: list[str] = []
    text = path.read_text(encoding="utf-8")
    rows = list(csv.reader(text.splitlines()))

    if not rows:
        return ["[F-1] o arquivo está vazio: não há cabeçalho nem amostra"]

    header = rows[0]
    # `[F-1]`: menos de três colunas no cabeçalho significa que o separador não
    # é vírgula. Não se afirma qual separador o arquivo usa.
    if len(header) < 3:
        violations.append(
            f"[F-1] o cabeçalho tem {len(header)} coluna(s); um `.csv` deste export "
            "tem `Sample`, `Label` e ao menos uma feature separadas por vírgula"
        )
        return violations

    # `[F-5]`: rótulo de classe na segunda coluna.
    if header[1] != "Label":
        violations.append(
            f"[F-5] a segunda coluna do cabeçalho é {header[1]!r} e não 'Label'; com "
            "amostras em linhas o rótulo de classe fica na segunda coluna"
        )

    # `[F-4]`: nomes conformes e únicos, em features e amostras.
    feature_names = header[2:]
    offenders = [name for name in feature_names if not _MA_CONFORMING_NAME.match(name)]
    if offenders:
        violations.append(
            f"[F-4] {len(offenders)} nome(s) de feature fora de [A-Za-z0-9_], "
            f"o primeiro é {offenders[0]!r}"
        )
    duplicated = [name for name, count in Counter(feature_names).items() if count > 1]
    if duplicated:
        violations.append(
            f"[F-4] nome(s) de feature repetido(s): {duplicated[:3]}; o formato exige únicos"
        )

    sample_names = [row[0] for row in rows[1:] if row]
    bad_samples = [name for name in sample_names if not _MA_CONFORMING_NAME.match(name)]
    if bad_samples:
        violations.append(
            f"[F-4] {len(bad_samples)} nome(s) de amostra fora de [A-Za-z0-9_], "
            f"o primeiro é {bad_samples[0]!r}"
        )
    repeated_samples = [name for name, count in Counter(sample_names).items() if count > 1]
    if repeated_samples:
        violations.append(f"[F-4] nome(s) de amostra repetido(s): {repeated_samples[:3]}")

    # `[F-2]` e `[F-3]`: as células de intensidade.
    negatives = 0
    spaced = 0
    non_numeric: list[str] = []
    for row in rows[1:]:
        for cell in row[2:]:
            if cell in _MA_MISSING:
                continue
            if " " in cell:
                spaced += 1
                continue
            try:
                value = float(cell)
            except ValueError:
                non_numeric.append(cell)
                continue
            if value <= 0:
                negatives += 1

    if spaced:
        violations.append(
            f"[F-3] {spaced} célula(s) com espaço dentro do número. A própria "
            "especificação declara que o modo de falha é silencioso: `1 600` é lido como `1`"
        )
    if non_numeric:
        violations.append(
            f"[F-2] {len(non_numeric)} célula(s) não numérica(s) e não ausente(s), "
            f"a primeira é {non_numeric[0]!r}; ausente se escreve vazio ou `NA`"
        )
    if negatives:
        violations.append(
            f"[F-2] {negatives} célula(s) com valor não positivo; o formato pede "
            '"only numeric and positive values"'
        )

    return violations


# ---------------------------------------------------------------------------
# O modo declarado contra o conteúdo
# ---------------------------------------------------------------------------

# Os dois eixos do rótulo de modo, e os tokens que os nomes de amostra usam nos quatro
# arquivos deste estudo. É observação, e por isso o desfecho é aviso.
_TOKENS_DO_MODO: dict[str, tuple[str, ...]] = {
    "hilic": ("HILIC",),
    "c18": ("C18",),
    "pos": ("_POS_", "_POS"),
    "neg": ("_NEG_", "_NEG"),
}


def deduce_mode_from_sample_names(sample_names: Sequence[str]) -> str | None:
    """O modo que os nomes de amostra mostram, ou `None`. É sugestão, nunca invariante.

    Usa os mesmos tokens de `check_mode_matches_content`, que são observação sobre os arquivos
    deste laboratório. O valor serve de preenchimento inicial de um campo que o usuário vê e pode
    trocar; quem tem outra convenção de nomes recebe `None`.

    Exige maioria estrita em cada eixo, contando só os nomes que trazem um lado e não o outro: os
    QCs e os brancos podem não trazer token nenhum, e um nome com os dois não diz nada.
    """
    if not sample_names:
        return None
    partes: list[str] = []
    for eixo in (("pos", "neg"), ("hilic", "c18")):
        contagem = {
            lado: sum(
                1
                for nome in sample_names
                if any(t in nome.upper() for t in _TOKENS_DO_MODO[lado])
                and not any(t in nome.upper() for t in _TOKENS_DO_MODO[outro])
            )
            for lado, outro in (eixo, eixo[::-1])
        }
        vencedor = [lado for lado, n in contagem.items() if n > len(sample_names) / 2]
        if len(vencedor) != 1:
            return None
        partes.append(vencedor[0])
    return "_".join(partes)


def check_mode_matches_content(mode: str, sample_names: Sequence[str]) -> list[ValidationIssue]:
    """Compara o modo declarado com o que os nomes de amostra mostram. Emite `warning`.

    Na interface o modo é escolhido num seletor ao lado do envio de arquivo, e escolher errado
    produz um resultado inteiro com o rótulo trocado sem que nada mais acuse.

    É aviso e não recusa porque o padrão `HILIC`/`C18` e `POS`/`NEG` é observação sobre os
    arquivos deste laboratório; recusar quebraria um estudo cujos nomes não sigam a convenção.

    A ausência de token não acusa. Só acusa quando o nome traz o token do eixo oposto: nome sem
    `HILIC` nem `C18` é convenção diferente, e não evidência de modo trocado.
    """
    if not sample_names:
        return []
    partes = mode.lower().split("_")
    if len(partes) != 2:
        return []
    achados: list[ValidationIssue] = []
    for declarado, oposto in (
        (partes[0], "neg" if partes[0] == "pos" else "pos"),
        (partes[1], "c18" if partes[1] == "hilic" else "hilic"),
    ):
        se_declarado = _TOKENS_DO_MODO.get(declarado, ())
        se_oposto = _TOKENS_DO_MODO.get(oposto, ())
        if not se_declarado or not se_oposto:
            continue
        com_oposto = sum(
            1
            for nome in sample_names
            if any(t in nome.upper() for t in se_oposto)
            and not any(t in nome.upper() for t in se_declarado)
        )
        if com_oposto > len(sample_names) / 2:
            achados.append(
                ValidationIssue(
                    code="MODE_CONTENT_MISMATCH",
                    severity="warning",
                    message=(
                        f"o modo declarado é `{mode}`, e {com_oposto} de {len(sample_names)} nomes "
                        f"de amostra trazem `{oposto.upper()}` em vez de `{declarado.upper()}`. "
                        "Se o modo estiver errado, o resultado sai com o rótulo trocado e nada "
                        "mais acusa. Isto é AVISO e não recusa: o padrão de nomes é convenção "
                        "deste laboratório, não invariante do formato"
                    ),
                )
            )
    return achados

"""Propriedades da ingestão, via `hypothesis`.

**A invariante é POSICIONAL, não a soma.** A especificação pedia "preserva a soma
total dos valores", e essa propriedade é fraca por dois motivos independentes.

Ela passa mesmo se o leitor embaralhar qual valor foi para qual célula — e o risco
real da ingestão é justamente a transposição de orientação, que um bug de eixo
trocado preserva a soma perfeitamente. E somar 6035 por 68 floats em ordens diferentes
exige tolerância, o que enfraquece ainda mais o que já era fraco.

A invariante substituta: o valor na posição lógica (feature i, amostra j) é igual
à célula de origem, com a ordenação por injeção aplicada. A soma sobrevive como
*smoke check*, não como a propriedade.

**Escopo da estratégia, e isto é escolha e não verdade sobre a ingestão.** As
matrizes geradas são numericamente válidas: sem célula não-numérica, sem valor
ausente. O caminho com `NaN` — `read_measurements` converte célula não-conversível
em `NaN` e emite `NON_NUMERIC_CELL` — é coberto pelos testes desse código, não
pela propriedade. Afirmar preservação posicional sobre `NaN` exigiria comparar
`NaN` com `NaN`, que é falso por definição de IEEE 754, e o teste falharia por
motivo correto e confuso.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from refino.io.readers import load_matrix, read_toserrf, to_matrix
from refino.io.writers import write_toserrf
from tests.factories import (
    CANONICAL_GROUPS,
    QC_LABEL,
    MatrixSpec,
    SampleSpec,
    render_measurements,
    render_sample_list,
)

# Intensidades plausíveis para LC-MS, para não disparar
# IMPLAUSIBLE_INTENSITY_RANGE e poluir a propriedade com issues irrelevantes.
INTENSITIES = st.floats(
    min_value=1e2, max_value=1e8, allow_nan=False, allow_infinity=False, width=64
)

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,  # escreve arquivos em disco; deadline por exemplo é ruidoso
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def valid_matrices(draw: st.DrawFn) -> MatrixSpec:
    """Matriz numericamente VÁLIDA, com ordem de injeção embaralhada.

    As colunas saem em ordem arbitrária de propósito: é o embaralhamento que dá
    dente à invariante posicional. Uma estratégia que gerasse colunas já em ordem
    de injeção não distinguiria ingestão correta de ingestão que ignora o join.
    """
    n_features = draw(st.integers(min_value=1, max_value=6))
    n_qcs = draw(st.integers(min_value=5, max_value=6))
    n_patients = draw(st.integers(min_value=2, max_value=6))
    n_samples = n_qcs + n_patients

    orders = draw(st.permutations(list(range(1, n_samples + 1))))

    samples: list[SampleSpec] = []
    for i in range(n_qcs):
        samples.append(SampleSpec(f"RUN_QC{i + 1}", QC_LABEL, orders[i]))
    for i in range(n_patients):
        group = CANONICAL_GROUPS[i % len(CANONICAL_GROUPS)]
        samples.append(SampleSpec(f"RUN_{i + 1}_ID{i + 1}_{group}", group, orders[n_qcs + i]))

    values = tuple(tuple(draw(INTENSITIES) for _ in range(n_samples)) for _ in range(n_features))
    return MatrixSpec(
        samples=tuple(samples),
        feature_names=tuple(f"{i + 1}.00_{(i + 1) * 100}.0000m/z" for i in range(n_features)),
        values=values,
    )


def _write(spec: MatrixSpec, tmp_path: Path) -> tuple[Path, Path]:
    m, s = tmp_path / "m.csv", tmp_path / "s.csv"
    m.write_text(render_measurements(spec), encoding="utf-8", newline="")
    s.write_text(render_sample_list(spec), encoding="utf-8", newline="")
    return m, s


# ---------------------------------------------------------------------------
# A propriedade: preservação POSICIONAL
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(spec=valid_matrices())
def test_ingestion_preserves_every_value_positionally(spec: MatrixSpec, tmp_path: Path) -> None:
    """A invariante que um bug de transposição NÃO sobrevive.

    Para toda feature i e amostra j: o valor em `matrix.values[linha_de_j, i]` é a
    célula `spec.values[i][coluna_de_j]`. Um eixo trocado preserva a soma e
    quebra isto.
    """
    m, s = _write(spec, tmp_path)
    matrix = load_matrix(m, s, "prop")

    row_of = {name: index for index, name in enumerate(matrix.sample_names)}
    for feature_index in range(len(spec.feature_names)):
        for column, sample in enumerate(spec.samples):
            expected = spec.values[feature_index][column]
            actual = matrix.values[row_of[sample.name], feature_index]
            assert actual == expected, (
                f"feature {feature_index}, amostra {sample.name}: "
                f"esperava {expected!r}, veio {actual!r}"
            )


@PROPERTY_SETTINGS
@given(spec=valid_matrices())
def test_ingestion_sorts_rows_by_injection_order(spec: MatrixSpec, tmp_path: Path) -> None:
    """As colunas chegam embaralhadas e saem em ordem de injeção."""
    m, s = _write(spec, tmp_path)
    matrix = load_matrix(m, s, "prop")
    assert matrix.injection_orders.tolist() == sorted(s_.injection_order for s_ in spec.samples)


@PROPERTY_SETTINGS
@given(spec=valid_matrices())
def test_ingestion_preserves_the_sets_of_names(spec: MatrixSpec, tmp_path: Path) -> None:
    """Nenhum nome se perde nem se inventa."""
    m, s = _write(spec, tmp_path)
    matrix = load_matrix(m, s, "prop")
    assert set(matrix.sample_names) == {x.name for x in spec.samples}
    assert matrix.feature_names == spec.feature_names


@PROPERTY_SETTINGS
@given(spec=valid_matrices())
def test_canonical_orientation_holds_for_any_generated_matrix(
    spec: MatrixSpec, tmp_path: Path
) -> None:
    """Amostras em linhas, features em colunas — independentemente da entrada."""
    m, s = _write(spec, tmp_path)
    matrix = load_matrix(m, s, "prop")
    assert matrix.values.shape == (len(spec.samples), len(spec.feature_names))


@PROPERTY_SETTINGS
@given(spec=valid_matrices())
def test_qc_mask_counts_the_qc_labels_of_the_input(spec: MatrixSpec, tmp_path: Path) -> None:
    m, s = _write(spec, tmp_path)
    matrix = load_matrix(m, s, "prop")
    assert int(matrix.qc_mask.sum()) == sum(1 for x in spec.samples if x.label == QC_LABEL)


@PROPERTY_SETTINGS
@given(spec=valid_matrices())
def test_toserrf_round_trip_preserves_values(spec: MatrixSpec, tmp_path: Path) -> None:
    """Ida e volta pelo formato `toserrf`, a 1e-9 relativo."""
    m, s = _write(spec, tmp_path)
    origin = load_matrix(m, s, "prop")

    out = tmp_path / "out.csv"
    write_toserrf(origin, out)
    groups = dict(zip(origin.sample_names, origin.groups, strict=True))
    restored = to_matrix(read_toserrf(out), groups, "prop")

    assert restored.sample_names == origin.sample_names
    assert np.allclose(restored.values, origin.values, rtol=1e-9, atol=0.0)


def test_o_canto_do_toserrf_leva_na_e_e_o_delimitador_do_formato(tmp_path: Path) -> None:
    """As três primeiras linhas abrem com `NA`, e ele é ESTRUTURA e não resíduo de planilha.

    ## Por que este teste existe

    `NA` no canto **parece** artefato de serialização — junto com as aspas em
    campos numéricos, o expoente `E` maiúsculo e o terminador de linha —, e a decisão foi não
    reproduzir nenhum dos quatro. **Sem esta asserção, a leitura natural de quem encontrar três
    `NA` literais no escritor é que alguém introduziu bug-compatibilidade, e a correção óbvia é
    removê-los.**

    O que ela impede é a remoção. Medido em contêiner: `serrf.read.data` (`tools.R:250-253`) deriva
    o layout inteiro por `min(which(!is.na(...)))`, e com o canto vazio reprova o nosso arquivo com
    `Error in 1:(ncol(p) - 1) : argument of length 0`. São **3 células em 153.160**, e são o único
    delimitador do formato.

    ## O que ela NÃO afirma

    Não afirma que o R lê o arquivo — isso é `pytest -m docker`, e o portão de commit não sobe
    contêiner. Afirma a **condição** que a medição em contêiner mostrou ser necessária, que é o que
    esta camada pode afirmar sem o oráculo de pé.

    E não afirma os outros três artefatos: eles continuam a **não** ser reproduzidos, e
    continuam excluídos da comparação. É a distância entre COMPARAÇÃO e INTERCÂMBIO
    que o item 60 declara poder existir sem contradição.
    """
    # A menor matriz que a ingestão aceita: cinco QCs é o piso de `valid_matrices`, e a forma dos
    # nomes é a que `render_sample_list` sabe serializar. Construída à mão e não sorteada, porque a
    # propriedade afirmada é do CABEÇALHO e não depende de nenhum valor.
    amostras = [SampleSpec(f"RUN_QC{i + 1}", QC_LABEL, i + 1) for i in range(5)]
    amostras += [
        SampleSpec(f"RUN_{i + 1}_ID{i + 1}_{CANONICAL_GROUPS[i]}", CANONICAL_GROUPS[i], 6 + i)
        for i in range(2)
    ]
    spec = MatrixSpec(
        samples=tuple(amostras),
        feature_names=("1.00_100.0000m/z", "2.00_200.0000m/z"),
        values=((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), (8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)),
    )
    m, s = _write(spec, tmp_path)
    origin = load_matrix(m, s, "prop")

    saida = tmp_path / "canto.csv"
    write_toserrf(origin, saida)
    linhas = saida.read_bytes().split(b"\r\n")

    assert len(linhas) > 4, "âncora: o arquivo é curto demais e as asserções abaixo seriam vácuas"
    for i, nome in enumerate((b"batch", b"sampleType", b"time")):
        assert linhas[i].startswith(b"NA," + nome), (
            f"a linha {nome.decode()!r} não abre com `NA` — o leitor do R deriva o layout do "
            f"primeiro não-NA da coluna, e sem isto ele reprova o arquivo inteiro. "
            f"Veio: {linhas[i][:40]!r}"
        )
    # **SEM aspas**, embora o golden do R cite todos os campos (`"No","label"`): as aspas são um
    # dos três artefatos de serialização que não se reproduzem. A distinção é exactamente a
    # que este bloco existe para manter: o `NA` entra porque carrega estrutura, as aspas não entram
    # porque não carregam.
    assert linhas[3].startswith(b"No,label"), (
        "a quarta linha mudou de forma, e ela é o resto do delimitador que o R usa"
    )


# ---------------------------------------------------------------------------
# Smoke check — a soma, rebaixada ao que ela é
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(spec=valid_matrices())
def test_total_sum_is_preserved_as_a_smoke_check(spec: MatrixSpec, tmp_path: Path) -> None:
    """A soma total, com tolerância — e apenas como smoke check.

    Deliberadamente NÃO é a propriedade principal: passa mesmo com os valores
    embaralhados entre células, e é insensível ao único bug que a ingestão
    realmente arrisca. A tolerância existe porque a ordem de acumulação muda com
    a transposição, e soma de floats não é associativa.
    """
    m, s = _write(spec, tmp_path)
    matrix = load_matrix(m, s, "prop")

    expected = sum(sum(row) for row in spec.values)
    assert float(matrix.values.sum()) == np.float64(expected).astype(float) or np.isclose(
        matrix.values.sum(), expected, rtol=1e-12, atol=0.0
    )

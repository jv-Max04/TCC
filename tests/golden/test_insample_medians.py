"""Duas camadas com naturezas diferentes, e a âncora que as autoriza.

A separação é o conteúdo do critério, não organização de arquivo:

- **Instrumentação:** as vinte e quatro medianas contra a tabela do terceiro eixo
  do achado 4.4, tolerância de 0,5 pp. Em três delas a mediana é da ordem de
  1e-13 pp e a asserção **não pode reprovar** — inclusive contra uma
  implementação que devolvesse zero constante. Ficam porque são espelho de
  calibração, e o nome de cada teste diz o que ela é.
- **Portão:** a fração de features com RSD numericamente indistinguível de zero
  no `batch_loess`, que é a quantidade que caracteriza aquelas células. É onde a
  implementação que devolve zero constante reprova.
- **Regressão:** a implementação de `refino` contra o fixture gerado por
  transcrição independente, célula por célula.

Tudo aqui depende da âncora da coluna `none`: sem instrumento validado contra
artefato externo, número medido por ele não informa nada. Âncora reprovada
produz `skip`, nunca `fail` — e há teste afirmando que o encanamento
funciona, porque "os doze dependem da âncora" sem verificação é afirmação sobre
encanamento que ninguém checou.
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from refino.metrics.rsd import rsd
from tests.conftest import AnchorOutcome

pytestmark = pytest.mark.golden

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURE = REPO / "data" / "fixtures" / "insample_rsd_medians.json"

# Tabela do terceiro eixo do achado 4.4, em pontos percentuais: (plain, outlier_removed).
# `None` é a célula que a especificação publica como "~0" — asserção de proximidade de zero
# em vez de valor, porque publicar 1,9e-13 como expectativa seria congelar ruído de
# cancelamento.
PUBLISHED_MEDIANS_PCT: dict[tuple[str, str], tuple[float | None, float | None]] = {
    ("pos_hilic", "none"): (167.77, 143.81),
    ("pos_hilic", "batch_loess"): (113.16, 3.81),
    ("pos_hilic", "SERRF"): (47.66, 40.63),
    ("pos_c18", "none"): (24.69, 24.15),
    ("pos_c18", "batch_loess"): (None, None),
    ("pos_c18", "SERRF"): (8.38, 7.50),
    ("neg_hilic", "none"): (69.42, 67.18),
    ("neg_hilic", "batch_loess"): (30.74, 4.67),
    ("neg_hilic", "SERRF"): (45.80, 21.66),
    ("neg_c18", "none"): (18.29, 15.50),
    ("neg_c18", "batch_loess"): (4.75, None),
    ("neg_c18", "SERRF"): (6.74, 3.64),
}

MEDIAN_TOLERANCE_PCT = 0.5

# O PORTÃO. Fração de features com RSD in-sample `< 1e-6` (sobre o RSD como
# fração) no `batch_loess`, variante `outlier_removed`, denominador = features com
# RSD **definido**.
#
# **Observação sobre estes quatro arquivos, não invariante**: decorre de
# quanto o LOESS interpolou em cada corrida, e da configuração de lote.
#
# O predicado é `< 1e-6` e não `== 0.0`: 14 a 27 pp das features vivem no
# aglomerado de epsilon de máquina, onde cair em `0.0` ou em `3e-16` é artefato de
# ordem de acumulação. `test_o_limiar_do_portao_esta_no_vao` é o que prova que o
# limiar não é carga estrutural.
GATE_FRAC_NEAR_ZERO_PCT: dict[str, float] = {
    "pos_hilic": 44.2,
    "pos_c18": 64.4,
    "neg_hilic": 38.3,
    "neg_c18": 51.5,
}

GATE_TOLERANCE_PCT = 0.1

# Tolerância da sonda do vão, **derivada de medição e pré-registrada**.
#
# Medido sobre estes quatro arquivos: em seis das oito combinações de modo e
# variante a diferença entre a fração em 1e-6 e a em 1e-12 é **exatamente zero**.
# As duas exceções são o `pos_hilic`, com quatro features em `plain` (0,072 pp) e
# três em `outlier_removed` (0,054 pp) na faixa entre os dois limiares. O máximo
# medido é 0,072 pp e a tolerância é 0,15 pp — cerca de 2x, e o registro do máximo
# é o que permite notar se um dia crescer.
#
# **O vão não é vazio, e afirmar que era seria mais forte do que o dado permite.**
# O que ele é: mover o limiar seis ordens de grandeza mexe no número congelado em
# até 0,072 pp, contra 14 a 27 pp de mudar o predicado para `== 0.0`. É essa razão
# de trezentos para um que decide, não uma vacuidade.
GAP_TOLERANCE_PCT = 0.15

CELLS = list(PUBLISHED_MEDIANS_PCT)
VARIANTS = ("plain", "outlier_removed")


# ---------------------------------------------------------------------------
# Fixtures e a dependência da âncora
# ---------------------------------------------------------------------------


def require_anchor(anchor: AnchorOutcome) -> None:
    """`skip` quando o instrumento não está validado. **Nunca `fail`.**

    Doze reprovações causadas por um instrumento quebrado escondem qual é o
    defeito; doze `skip` com o motivo da âncora o nomeiam.
    """
    if not anchor.ok:
        pytest.skip(
            "âncora do [CA-2.1] reprovou — instrumento não validado contra o "
            f"QC - RSD.csv, e número medido por ele não informa nada: {anchor.detalhe}"
        )


@pytest.fixture(scope="session")
def frozen() -> dict[str, Any]:
    if not FIXTURE.exists():
        pytest.skip(f"{FIXTURE.name} ausente; gere por scripts/gen_insample_medians.py")
    return dict(json.loads(FIXTURE.read_text(encoding="utf-8")))


@pytest.fixture(scope="session")
def by_cell(frozen: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(c["mode"], c["method"]): c for c in frozen["cells"]}


def _measured_rsd(mode: str, method: str, variant: str) -> np.ndarray:
    """RSD por feature pela implementação de `refino`, sobre a golden normalizada.

    Passa pelos resolvedores de `tests/conftest.py` e pelo `sampleType` do
    `toserrf`, não por casamento de `"QC"` no nome — o mesmo caminho do gerador.
    """
    import csv

    from tests.conftest import normalized_path, toserrf_path

    with toserrf_path(mode).open(newline="", encoding="utf-8-sig") as handle:
        head = list(csv.reader(handle))
    types = next(r for r in head if len(r) > 1 and r[1] == "sampleType")
    labels = next(r for r in head if len(r) > 1 and r[1] == "label")
    qc_names = [
        labels[i] for i in range(2, len(types)) if types[i].strip().strip('"').lower() == "qc"
    ]

    with normalized_path(mode, method).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    columns = [header.index(name) for name in qc_names]
    data = [
        [np.nan if row[c].strip() in ("", "NA", "NaN") else float(row[c]) for c in columns]
        for row in rows[1:]
        if row and row[0]
    ]
    # `(features, qcs)` no arquivo; `rsd` com `axis=1` dá um valor por feature.
    return rsd(np.asarray(data, dtype=np.float64), axis=1, variant=variant)


# ---------------------------------------------------------------------------
# Camada 1 — INSTRUMENTAÇÃO: as medianas contra a tabela publicada
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "method"), CELLS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_instrumentacao_mediana_reproduz_tabela_4_4(
    mode: str,
    method: str,
    variant: str,
    by_cell: dict[tuple[str, str], dict[str, Any]],
    rsd_instrument_anchor: AnchorOutcome,
) -> None:
    """Espelho de calibração, **não portão** — o nome diz o que a asserção é.

    Nas três células que a especificação publica como "~0" nenhuma tolerância de 0,5 pp
    pode reprovar. Quem tem dente ali é `test_portao_*`.
    """
    require_anchor(rsd_instrument_anchor)
    published = PUBLISHED_MEDIANS_PCT[(mode, method)][VARIANTS.index(variant)]
    measured = by_cell[(mode, method)][variant]["median_rsd_pct"]

    if published is None:
        assert abs(measured) < MEDIAN_TOLERANCE_PCT, (
            f"{mode}/{method}/{variant}: a §4.4 publica ~0 e medi {measured:.6g} pp"
        )
        return
    assert abs(measured - published) <= MEDIAN_TOLERANCE_PCT, (
        f"{mode}/{method}/{variant}: §4.4 publica {published} pp e medi {measured:.4f} pp"
    )


def test_instrumentacao_tres_dos_vinte_e_quatro_numeros_sao_indistinguiveis_de_zero(
    by_cell: dict[tuple[str, str], dict[str, Any]],
    rsd_instrument_anchor: AnchorOutcome,
) -> None:
    """Quantas asserções de mediana são vacuamente verdadeiras.

    Contá-las é o que impede a próxima sessão de olhar vinte e quatro asserções
    verdes e concluir que vinte e quatro coisas foram verificadas.
    """
    require_anchor(rsd_instrument_anchor)
    vacuas = [
        (mode, method, variant)
        for (mode, method) in CELLS
        for variant in VARIANTS
        if abs(by_cell[(mode, method)][variant]["median_rsd_pct"]) < MEDIAN_TOLERANCE_PCT
    ]
    assert len(vacuas) == 3, f"esperava três asserções vacuamente verdadeiras, achei {vacuas}"
    assert all(method == "batch_loess" for _, method, _ in vacuas)


# ---------------------------------------------------------------------------
# Camada 2 — PORTÃO: a fração de RSD indistinguível de zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", list(GATE_FRAC_NEAR_ZERO_PCT))
def test_portao_fracao_de_rsd_indistinguivel_de_zero(
    mode: str,
    by_cell: dict[tuple[str, str], dict[str, Any]],
    rsd_instrument_anchor: AnchorOutcome,
) -> None:
    """O portão. Observação sobre estes quatro arquivos."""
    require_anchor(rsd_instrument_anchor)
    esperado = GATE_FRAC_NEAR_ZERO_PCT[mode]
    obtido = by_cell[(mode, "batch_loess")]["outlier_removed"]["frac_near_zero_pct"]
    assert abs(obtido - esperado) <= GATE_TOLERANCE_PCT, (
        f"{mode}: esperava {esperado}% das features definidas com RSD < 1e-6 no "
        f"batch_loess/outlier_removed e medi {obtido:.2f}%"
    )


@pytest.mark.parametrize("mode", list(GATE_FRAC_NEAR_ZERO_PCT))
@pytest.mark.parametrize("variant", VARIANTS)
def test_o_limiar_do_portao_esta_no_vao_e_nao_numa_encosta(
    mode: str,
    variant: str,
    by_cell: dict[tuple[str, str], dict[str, Any]],
    rsd_instrument_anchor: AnchorOutcome,
) -> None:
    """A fração em 1e-12 iguala a fração em 1e-6, e é isso que autoriza o predicado.

    Se divergir, o número congelado passou a depender do limiar escolhido e o
    portão deixou de medir "quantas o LOESS interpolou" para medir "onde eu pus a
    régua" — que é exatamente o motivo de o predicado não ser `== 0.0`.
    """
    require_anchor(rsd_instrument_anchor)
    cell = by_cell[(mode, "batch_loess")][variant]
    assert cell["frac_near_zero_pct"] == pytest.approx(
        cell["frac_below_gap_probe_pct"], abs=GAP_TOLERANCE_PCT
    ), (
        f"{mode}/{variant}: fração em 1e-6 é {cell['frac_near_zero_pct']:.2f}% e em 1e-12 é "
        f"{cell['frac_below_gap_probe_pct']:.2f}% — o limiar ficou numa encosta"
    )


@pytest.mark.parametrize("mode", list(GATE_FRAC_NEAR_ZERO_PCT))
def test_o_predicado_zero_exato_reprova_o_portao(
    mode: str,
    rsd_instrument_anchor: AnchorOutcome,
) -> None:
    """Contar `== 0.0` **não** reproduz o número congelado.

    É o teste que impede a volta do defeito. As duas medições que
    se contradisseram na auditoria eram `== 0.0` e `< 1e-6` sobre o mesmo dado, e
    a diferença é de 14 a 27 pontos percentuais — quem "corrigir" o predicado de
    volta para zero exato reprova aqui, com a distância no texto da falha.

    A tolerância é a do portão, não uma folga inventada: o ponto é que a distância
    entre os dois predicados é três ordens de grandeza maior que ela.
    """
    require_anchor(rsd_instrument_anchor)
    medido = _measured_rsd(mode, "batch_loess", "outlier_removed")
    definidas = medido[np.isfinite(medido)]
    exatamente_zero_pct = 100.0 * float(np.mean(definidas == 0.0))
    congelado = GATE_FRAC_NEAR_ZERO_PCT[mode]
    assert abs(exatamente_zero_pct - congelado) > 10 * GATE_TOLERANCE_PCT, (
        f"{mode}: contar zero exato dá {exatamente_zero_pct:.1f}% e o portão espera "
        f"{congelado}%. Se a distância sumiu, os dois predicados voltaram a ser "
        "indistinguíveis e o portão não protege mais a escolha do predicado"
    )


def test_zero_constante_passa_na_mediana_e_reprova_no_portao(
    by_cell: dict[tuple[str, str], dict[str, Any]],
    rsd_instrument_anchor: AnchorOutcome,
) -> None:
    """A prova de que o portão discrimina onde a mediana não discrimina.

    Sem este teste, "a fração é o portão" é afirmação de desenho que nada mede.
    Uma implementação que devolvesse zero constante é o pior caso concreto: ela
    não mede nada e passa em três das vinte e quatro asserções de mediana.
    """
    require_anchor(rsd_instrument_anchor)
    n = 1000
    constante = np.zeros(n)

    # Passa na mediana das três células que a especificação publica como ~0.
    zeradas = [
        (mode, method, variant)
        for (mode, method) in CELLS
        for variant in VARIANTS
        if PUBLISHED_MEDIANS_PCT[(mode, method)][VARIANTS.index(variant)] is None
    ]
    assert zeradas, "a tabela precisa ter célula ~0 para este teste ter objeto"
    for _, _, _ in zeradas:
        assert abs(float(np.median(constante))) < MEDIAN_TOLERANCE_PCT

    # E reprova no portão, nos quatro modos.
    fracao_constante_pct = 100.0 * float(np.mean(constante < 1e-6))
    for mode, esperado in GATE_FRAC_NEAR_ZERO_PCT.items():
        assert abs(fracao_constante_pct - esperado) > GATE_TOLERANCE_PCT, (
            f"{mode}: zero constante daria {fracao_constante_pct}% e o portão espera "
            f"{esperado}% — o portão não discrimina e o critério é decorativo"
        )
    # E a mediana real do modo em que ela NÃO é ~0 também o reprovaria, para que
    # o teste não dependa só das células degeneradas.
    assert by_cell[("pos_hilic", "batch_loess")]["outlier_removed"]["median_rsd_pct"] > 1.0


# ---------------------------------------------------------------------------
# Camada 3 — REGRESSÃO: `refino` contra a transcrição independente
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "method"), CELLS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_regressao_implementacao_concorda_com_a_transcricao_independente(
    mode: str,
    method: str,
    variant: str,
    by_cell: dict[tuple[str, str], dict[str, Any]],
    rsd_instrument_anchor: AnchorOutcome,
) -> None:
    """Duas transcrições da mesma fórmula, e é a independência que dá conteúdo.

    As contagens são comparadas por **igualdade exata**: são inteiros sobre o
    mesmo predicado, e tolerância aqui esconderia divergência de uma feature.
    """
    require_anchor(rsd_instrument_anchor)
    esperado = by_cell[(mode, method)][variant]
    medido = _measured_rsd(mode, method, variant)
    definidas = np.isfinite(medido)

    assert medido.size == esperado["n_features"]
    assert int(definidas.sum()) == esperado["n_defined"]
    assert int((~definidas).sum()) == esperado["n_undefined"]
    assert int((medido[definidas] < 1e-6).sum()) == esperado["n_near_zero"]
    assert int((medido[definidas] < 1e-12).sum()) == esperado["n_below_gap_probe"]
    assert float(np.median(medido[definidas])) * 100.0 == pytest.approx(
        esperado["median_rsd_pct"], rel=1e-9, abs=1e-15
    )


def test_indefinidas_da_coluna_none_batem_com_a_contagem_de_d25(
    by_cell: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """234 / 212 / 60 / 33 — contagem observada sobre estes quatro arquivos.

    Vale como verificação cruzada do gerador: ele chega aos mesmos indefinidos por
    um caminho que não passa por `refino` nem pelo `QC - RSD.csv`.

    **A contagem é da variante `outlier_removed`, e o mecanismo não é o que se
    supunha**. Ver o teste seguinte.
    """
    esperado = {"pos_hilic": 234, "neg_hilic": 212, "pos_c18": 60, "neg_c18": 33}
    for mode, count in esperado.items():
        assert by_cell[(mode, "none")]["outlier_removed"]["n_undefined"] == count


def test_indefinido_e_dependente_de_variante_e_qcs_todos_zero_e_a_causa_minoritaria(
    by_cell: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Onde o RSD indefinido nasce, medido — e não é onde se supunha.

    Na coluna `none` não há infinito nem ausente, então sob `plain` a única via
    para indefinido é média zero, isto é, **QCs todos zero**: 102 / 136 / 35 / 25
    features. As 234 / 212 / 60 / 33 da coluna do R são da variante
    `outlier_removed`, e a diferença — 132 features no `pos_hilic`, a maioria — são
    features em que a **remoção de outliers** apagou os QCs não nulos e deixou um
    vetor de zeros.

    Importa: uma política de indefinido escrita acreditando que a
    causa é "QCs todos zero" estaria errada sobre 56% dos casos do `pos_hilic`, e
    trataria como feature não detectada o que é feature detectada em um único QC.
    """
    todos_zero = {"pos_hilic": 102, "neg_hilic": 136, "pos_c18": 35, "neg_c18": 25}
    do_r = {"pos_hilic": 234, "neg_hilic": 212, "pos_c18": 60, "neg_c18": 33}
    for mode, count in todos_zero.items():
        assert by_cell[(mode, "none")]["plain"]["n_undefined"] == count
        assert count < do_r[mode], (
            f"{mode}: a remoção de outliers precisa criar indefinido, senão AR12 é falso"
        )


# ---------------------------------------------------------------------------
# O fixture e sua procedência
# ---------------------------------------------------------------------------


def test_procedencia_presente_e_completa(frozen: dict[str, Any]) -> None:
    """Sem procedência, "mudou o numpy ou mudou nosso código?" é indecidível."""
    prov = frozen["provenance"]
    for campo in (
        "python_version",
        "numpy_version",
        "command",
        "generated_at",
        "units",
        "near_zero_threshold_rsd_fraction",
        "gap_probe_threshold_rsd_fraction",
        "outlier_coef",
        "ddof",
        "qc_selection_rule",
        "sources",
    ):
        assert prov.get(campo) not in (None, "", {}), f"procedência sem {campo}"
    assert prov["near_zero_threshold_rsd_fraction"] == 1e-6
    assert prov["gap_probe_threshold_rsd_fraction"] == 1e-12
    assert prov["ddof"] == 1
    assert prov["outlier_coef"] == 1.5
    assert prov["seed"] is None


def test_sha256_das_origens_casa_com_o_manifesto(frozen: dict[str, Any]) -> None:
    """Número congelado sobre dado que mudou é pior que número ausente."""
    from tests.conftest import read_manifest

    known = read_manifest()
    if not known:
        pytest.skip("manifesto de integridade indisponível")
    for relative, digest in frozen["provenance"]["sources"].items():
        assert relative in known, f"{relative} fora do manifesto"
        assert known[relative] == digest, (
            f"{relative} divergiu do manifesto; regenere por scripts/gen_insample_medians.py"
        )


def test_fixture_cobre_as_doze_celulas(frozen: dict[str, Any]) -> None:
    presentes = {(c["mode"], c["method"]) for c in frozen["cells"]}
    assert presentes == set(CELLS)
    for cell in frozen["cells"]:
        assert cell["n_qc"] == len(cell["qc_sample_names"])
        for variant in VARIANTS:
            assert (
                cell[variant]["n_defined"] + cell[variant]["n_undefined"]
                == (cell[variant]["n_features"])
            )


def test_percentuais_do_fixture_concordam_com_as_contagens(frozen: dict[str, Any]) -> None:
    """A redundância entre contagem e percentual é deliberada; a deriva não é."""
    for cell in frozen["cells"]:
        for variant in VARIANTS:
            v = cell[variant]
            assert v["frac_near_zero_pct"] == pytest.approx(
                100.0 * v["n_near_zero"] / v["n_defined"], rel=1e-12
            )
            assert v["frac_below_gap_probe_pct"] == pytest.approx(
                100.0 * v["n_below_gap_probe"] / v["n_defined"], rel=1e-12
            )


def test_o_gerador_nao_importa_refino() -> None:
    """Verificado estaticamente, e não por confiança.

    Um gerador que importasse `refino` reproduziria o defeito que a camada de
    regressão deveria acusar: regenerar apagaria a evidência.
    """
    import ast

    fonte = (REPO / "scripts" / "gen_insample_medians.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(fonte)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("refino"), f"importa {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("refino"), f"importa de {node.module}"


def test_a_regeneracao_reproduz_o_fixture_versionado() -> None:
    """O fixture não foi editado à mão, e é isso que o congelamento significa.

    Sem este teste, "congelado com procedência" descreve um arquivo que qualquer
    um pode ter ajustado para fazer um teste passar — que é exatamente a operação
    que a regra 2 do README proíbe. `generated_at` é ignorado na comparação,
    porque muda a cada execução por definição.
    """
    import subprocess

    resultado = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "gen_insample_medians.py"), "--check"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO,
    )
    if "dados de referência ausentes" in resultado.stderr:
        pytest.skip("dados de referência indisponíveis para regenerar")
    assert resultado.returncode == 0, (
        f"a regeneração divergiu do fixture versionado:\n{resultado.stdout}\n{resultado.stderr}"
    )


# ---------------------------------------------------------------------------
# A dependência da âncora, verificada
# ---------------------------------------------------------------------------


def test_ancora_reprovada_produz_skip_e_nao_fail() -> None:
    """O encanamento da âncora, medido em vez de suposto.

    Sem este teste, "os doze dependem da âncora" é afirmação sobre encanamento
    que ninguém checou — e o modo de falha é o silencioso: uma âncora que sempre
    diz `ok` deixaria a camada de regressão valendo como oráculo sem ser.
    """
    reprovada = AnchorOutcome(
        ok=False,
        detalhe="forçada a reprovar por teste",
        por_modo={"pos_hilic": "conjuntos de indefinidos DIFERENTES"},
    )
    # `pytest.skip.Exception` e não `Exception`: `Skipped` deriva de
    # `BaseException`, então `pytest.raises(Exception)` **não** a captura — o
    # `skip` escaparia, este teste seria reportado como pulado, e um teste pulado
    # que parece verde é a verificação decorativa em pessoa. Aconteceu
    # aqui na primeira escrita, e é por isso que o comentário fica.
    with pytest.raises(pytest.skip.Exception, match=r"âncora do \[CA-2\.1\] reprovou"):
        require_anchor(reprovada)


def test_ancora_aprovada_nao_produz_skip() -> None:
    """O outro lado: uma guarda que sempre pula é indistinguível de teste ausente."""
    require_anchor(AnchorOutcome(ok=True, detalhe="ok", por_modo={}))

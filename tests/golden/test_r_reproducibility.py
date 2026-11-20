"""A tabela do que o oráculo reproduz, e o que não.

**Este módulo não roda o R.** Ele lê `data/fixtures/r_serrf_divergence.json`, que
a camada `docker` gerou e congelou, e afirma a tabela sobre ele. É o padrão do
O critério e a razão de ser da especificação: portão que só roda à noite é portão que
ninguém vê reprovar no momento em que quebra.

**Metade das asserções afirma irreprodutibilidade, e isso é deliberado.** Se uma
segunda execução do R passasse a reproduzir o SERRF, ou a devolver o mesmo
conjunto de features não corrigidas no `batch_loess`, a premissa
estaria errada e a tolerância derivada do critério teria denominador zero.
Reprovar ali é o desfecho certo — é o mesmo argumento do teste que afirma que a
tradução de caminho **acontece** em vez de supor que sim.

A segunda linha da tabela — a coluna `none` do `QC - RSD.csv` — não é reafirmada
aqui. Ela é a âncora do critério, vive em `test_qc_rsd_none_column.py`, e
duplicá-la criaria dois números que passam a divergir.
"""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.golden

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURE = REPO / "data" / "fixtures" / "r_serrf_divergence.json"
GENERATOR = REPO / "scripts" / "gen_r_divergence.py"

MODES = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")


@pytest.fixture(scope="module")
def frozen() -> dict[str, Any]:
    if not FIXTURE.is_file():
        pytest.skip(
            "r_serrf_divergence.json ausente; a fixture é versionada, restaure-a do repositório"
        )
    return json.loads(FIXTURE.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def by_mode(frozen: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["mode"]: entry for entry in frozen["modes"]}


# ---------------------------------------------------------------------------
# O pré-registro não pode ter derivado do código que o declarou
# ---------------------------------------------------------------------------


def test_o_preregistro_do_fixture_bate_com_o_do_gerador(frozen: dict[str, Any]) -> None:
    """`k`, `N` e `M` no JSON são os mesmos do `gen_r_divergence.py`.

    Sem esta asserção, editar a constante no script e não regenerar — ou o
    inverso — deixaria dois pré-registros divergentes, e o critério leria o que
    estivesse mais à mão. É a família: conjunto declarado num
    lugar, derivado de outro, divergência reprovando nas duas direções.
    """
    source = GENERATOR.read_text(encoding="utf-8")
    declared: dict[str, float] = {}
    for name, key in (
        ("K_CA_3_1", "k_ca_3_1"),
        ("K_CA_3_2", "k_ca_3_2"),
        ("N_PYTHON_SEEDS", "n_python_seeds"),
        ("M_R_RUNS", "m_r_runs"),
    ):
        line = next(ln for ln in source.splitlines() if ln.startswith(f"{name} = "))
        declared[key] = float(line.split("=", 1)[1].strip())

    registered = frozen["preregistration"]
    for key, value in declared.items():
        assert float(registered[key]) == value, (
            f"{key}: fixture diz {registered[key]}, gerador diz {value}. "
            "Um dos dois foi editado sem o outro."
        )


def test_o_preregistro_carrega_as_clausulas_que_o_tornam_verificavel(
    frozen: dict[str, Any],
) -> None:
    """Justificativa, caveat e cláusula de não negociação estão no arquivo.

    O valor de `k` sem o motivo é número declarado por gosto — exatamente o que
    foi aposentado ao matar o `rho > 0,95`. E a cláusula de não negociação só serve
    se estiver onde quem for negociar vai ler.
    """
    registered = frozen["preregistration"]
    for key in (
        "k_justification",
        "k_caveat_not_a_lever",
        "no_negotiation_clause",
        "tolerance_formula",
        "quantity_note",
        "pair_reduction_note",
        "undefined_rsd_policy",
    ):
        assert registered.get(key), f"pré-registro sem `{key}`"

    assert "sqrt" in registered["tolerance_formula"], (
        "a fórmula tem de ser a soma QUADRÁTICA — a aritmética inflava 41,4%"
    )
    assert "NAO autoriza aumentar k" in registered["no_negotiation_clause"]


def test_k_e_maior_ou_igual_a_um(frozen: dict[str, Any]) -> None:
    """`k >= 1` é o que a especificação exige: o Python não pode ser cobrado mais estrito
    que o próprio oráculo contra si mesmo."""
    registered = frozen["preregistration"]
    assert registered["k_ca_3_1"] >= 1
    assert registered["k_ca_3_2"] >= 1


# ---------------------------------------------------------------------------
# A tabela do critério
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_15_none_e_oraculo_exato(by_mode: dict[str, dict[str, Any]], mode: str) -> None:
    """Linha 1: `normalized by - none.csv` — sim, exatamente.

    Duas afirmações, porque são dois modos de falha: as `M` execuções idênticas
    **entre si**, e idênticas **ao golden**. Só a primeira passaria num runner
    que executasse consistentemente a coisa errada.
    """
    none = by_mode[mode]["none"]
    assert none["identical_across_runs"], (
        f"{mode}: as execuções do caminho `none` divergiram entre si. "
        "`none` não tem estocasticidade nenhuma: divergência aqui é defeito do runner."
    )
    assert none["matches_golden"], (
        f"{mode}: o `none` do runner não reproduz o golden do laboratório; "
        f"max |dif| = {none['max_abs_diff_vs_golden']} sobre {none['n_cells']} células."
    )
    assert none["max_abs_diff_vs_golden"] == 0.0


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_15_serrf_nao_e_reproduzivel_e_a_divergencia_e_medida(
    by_mode: dict[str, dict[str, Any]], mode: str
) -> None:
    """Linha 4: a coluna `SERRF` — **não**.

    `d > 0` em **todos** os pares. Divergência zero significaria que o bloco
    semeia, que a premissa do achado 4.3 está errada, e que o denominador do
    O critério é zero — o que tornaria a tolerância derivada indefinida em vez de
    estrita. Reprovar aqui é o desfecho certo.
    """
    serrf = by_mode[mode]["serrf"]
    pairs = serrf["pairs"]
    expected = by_mode[mode]["n_runs"] * (by_mode[mode]["n_runs"] - 1) // 2
    assert len(pairs) == expected, f"{mode}: {len(pairs)} pares, esperados {expected}"

    zeros = [p for p in pairs if p["d"] <= 0.0]
    assert not zeros, (
        f"{mode}: {len(zeros)} pares com d <= 0. O bloco SERRF do tools.R não chama "
        "set.seed (achado 4.3) e não pode reproduzir a si mesmo. Se isto passou a "
        "acontecer, o oráculo mudou e o [CA-3.1] perdeu o denominador."
    )
    assert serrf["d_r1_r2_for_ca_3_1"] > 0.0
    assert serrf["sd_r_pct_for_ca_3_2"] > 0.0


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_16_o_que_o_serrf_reproduz_e_forma_e_posicao_de_zero(
    by_mode: dict[str, dict[str, Any]], mode: str
) -> None:
    """O que **é** reprodutível no caminho SERRF, e o critério nomeia.

    Sem esta metade, "não é reprodutível" seria compatível com o runner devolver
    matriz de tamanho errado — e as duas conclusões pedem investigações opostas.
    """
    serrf = by_mode[mode]["serrf"]
    assert serrf["shapes_identical"], f"{mode}: as execuções têm formas diferentes"
    assert serrf["zero_positions_identical"], (
        f"{mode}: zeros em posições diferentes entre execuções. A §4.6 diz que zeros "
        "sobrevivem à correção; se deixaram de sobreviver, o oráculo mudou."
    )


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_15_conjunto_nao_corrigido_do_batch_loess_e_estocastico(
    by_mode: dict[str, dict[str, Any]], mode: str
) -> None:
    """Linha 3: o `batch_loess` — e o conjunto de não corrigidas **também** varia.

    É medido. A v1.4 cobrava "o conjunto reproduzido exatamente"; duas
    execuções sobre o `neg_c18` deram 32 e 26 contra 20 no golden. O que fica
    afirmado é o que é verdadeiro: o conjunto varia, e a magnitude da variação é
    registrada em vez de colapsada num booleano.
    """
    uncorrected = by_mode[mode]["batch_loess"]["uncorrected"]
    assert not uncorrected["identical_across_runs"], (
        f"{mode}: o conjunto de não corrigidas ficou idêntico entre execuções. "
        "A medição deu o contrário, e o mecanismo é o sorteio de dobras do crossval "
        "dentro do parSapply sem clusterSetRNGStream. Se isto mudou, o oráculo mudou."
    )
    assert uncorrected["n_sometimes_uncorrected"] > 0
    assert uncorrected["n_intersection"] <= min(uncorrected["n_by_run"])
    assert uncorrected["n_union"] >= max(uncorrected["n_by_run"])


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_15_golden_do_batch_loess_nao_e_alvo_reproduzivel(
    by_mode: dict[str, dict[str, Any]], mode: str
) -> None:
    """O corolário que muda o alvo dos golden tests do LOESS.

    O conjunto do laboratório não é o núcleo estável nem subconjunto de toda
    execução. Quem cobrar 1.561 / 165 / 20 / 10 tem de cobrar contra o **artefato
    congelado**, nunca contra outra execução do R.
    """
    golden = by_mode[mode]["batch_loess"]["uncorrected"]["golden"]
    assert golden["n"] > 0, f"{mode}: o golden não tem features não corrigidas?"
    assert not golden["equals_intersection"] or not golden["is_subset_of_every_run"], (
        f"{mode}: o golden coincidiu com o núcleo estável E é subconjunto de toda "
        "execução. Se isso vale nos quatro modos, a medição estava errada e o conjunto é "
        "reproduzível — investigar antes de afrouxar."
    )


# ---------------------------------------------------------------------------
# Integridade da medição
# ---------------------------------------------------------------------------


def test_ca_3_16_execucoes_compartilham_entrada_e_configuracao(frozen: dict[str, Any]) -> None:
    """`rho` entre execuções com particionamento diferente não mede o que se pensa.

    `cores_detected` e `detectcores_ratio` decidem como o `parSapply` reparte as
    features, e a repartição decide quais sorteios caem em qual worker. Execuções
    com contagem de worker diferente não são amostras da mesma distribuição —
    conclusão correta sobre a quantidade errada.
    """
    for mode, runs in frozen["provenance"]["runs"].items():
        for key in ("input_sha256", "image_digest", "r_version", "cores_detected"):
            values = {json.dumps(run.get(key), sort_keys=True) for run in runs}
            assert len(values) == 1, f"{mode}: execuções divergem em `{key}`: {sorted(values)}"


def test_ca_3_16_numero_de_execucoes_respeita_o_preregistro(frozen: dict[str, Any]) -> None:
    """`M >= 5`: com duas execuções não há desvio a estimar."""
    minimum = frozen["preregistration"]["m_r_runs"]
    assert minimum >= 5, "o pré-registro fixou M abaixo de 5"
    for entry in frozen["modes"]:
        assert entry["n_runs"] >= minimum, (
            f"{entry['mode']}: {entry['n_runs']} execuções, pré-registro exige {minimum}"
        )


def test_ca_3_16_os_quatro_modos_estao_medidos(frozen: dict[str, Any]) -> None:
    """Os critérios são "nos quatro modos", e três não são quatro."""
    measured = {entry["mode"] for entry in frozen["modes"]}
    assert measured == set(MODES), f"faltam modos: {sorted(set(MODES) - measured)}"


def test_ca_3_16_rho_de_cada_par_declara_sobre_quantas_features_foi_medido(
    frozen: dict[str, Any],
) -> None:
    """`rho` sobre subconjunto diferente não é o mesmo `rho`.

    A contagem viaja com o número, e um par medido sobre punhado de features
    seria `rho` sem conteúdo com aparência de evidência.
    """
    for entry in frozen["modes"]:
        for pair in entry["serrf"]["pairs"]:
            used = pair["n_features_used"]
            assert used >= 3, f"{entry['mode']}: par medido sobre {used} features"
            assert used <= entry["n_features"]
            assert used + pair["n_features_excluded_undefined"] == entry["n_features"]


def test_a_entrada_medida_bate_com_o_manifesto(frozen: dict[str, Any]) -> None:
    """Divergência de `sha256` reprova pedindo regeneração, nunca compara em silêncio.

    É o padrão do critério: número congelado sobre dado que mudou é pior que
    número ausente, porque tem aparência de evidência.
    """
    manifest = REPO / "scripts" / "reference_data.sha256"
    if not manifest.is_file():
        pytest.skip("manifesto de integridade indisponível")

    known: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            digest, _, relative = stripped.partition("  ")
            if digest and relative:
                known[relative] = digest

    drifted = [
        (relative, digest)
        for relative, digest in frozen["provenance"]["input_sources"].items()
        if relative in known and known[relative] != digest
    ]
    assert not drifted, (
        f"as entradas mudaram desde a geração: {drifted}. "
        "Regenere pelo script — não compare número congelado contra dado diferente."
    )


def test_a_regua_de_rsd_esta_nomeada_nos_dois_lados(frozen: dict[str, Any]) -> None:
    """Variante e estimador viajam com o número, ou ele não é comparável.

    O mesmo arquivo golden dá 3,78% ou 104,73% para o `batch_loess` conforme o
    estimador, e o ranking entre métodos **inverte**.
    """
    provenance = frozen["provenance"]
    assert provenance["rsd_variant"] == "outlier_removed"
    assert provenance["rsd_estimator"] == "insample"
    assert "refino.metrics.rsd" in provenance["rsd_implementation"]
    assert "[CA-2.1]" in provenance["rsd_implementation"], (
        "a régua tem de declarar contra o que ela própria foi ancorada"
    )


def test_o_contrato_de_regeneracao_esta_no_arquivo(frozen: dict[str, Any]) -> None:
    """Quem regenera daqui a uma fase lê o arquivo, não o teste."""
    contract = frozen["regeneration_contract"]
    assert contract["exact"], "contrato sem bloco de igualdade exata"
    assert contract["not_reproducible_by_design"], (
        "contrato sem o bloco que declara o que NÃO pode bater"
    )
    assert "set.seed" in contract["note"]


def test_o_gerador_e_o_fixture_concordam_sobre_o_python(frozen: dict[str, Any]) -> None:
    """Regeneração sob outro minor de Python é mudança de instrumento, não de dado."""
    recorded = frozen["provenance"]["python_version"]
    assert recorded.startswith("3.12"), (
        f"fixture gerado sob Python {recorded}; o projeto pina 3.12 exato. "
        f"Este intérprete é {sys.version.split()[0]}."
    )


def test_a_incerteza_do_proprio_orcamento_esta_relatada(frozen: dict[str, Any]) -> None:
    """As duas estimativas de `sd_r` lado a lado — transparência, não ajuste.

    `sd_r` com `M = 5` tem 4 graus de liberdade, e duas gerações independentes do
    `neg_hilic` deram 0,1072 pp e 0,2221 pp: **fator 2**. O numerador é estável
    (0,006535 contra 0,006680, 2,2%); o denominador não é.

    Sem este bloco o leitor toma o orçamento de 0,0196 como exato. Com ele, vê que
    passar num orçamento derivado de um `sd` com 2x de incerteza é evidência mais
    fraca do que o número aparenta. O valor pré-registrado **não muda** e exceder
    continua não autorizando mexer em `M` — o que se corrige é a leitura,
    não o portão.
    """
    log = frozen.get("observed_regenerations")
    assert log, (
        "nenhuma geração independente registrada. O orçamento derivado seria lido "
        "como exato, e a fragilidade de sd_r com M = 5 ficaria invisível."
    )
    by_mode = {entry["mode"]: entry for entry in log}
    measured = {entry["mode"]: entry for entry in frozen["modes"]}
    for mode, entry in by_mode.items():
        assert mode in measured, f"log cita modo `{mode}` que não está medido"
        for key in ("source", "m_r_runs", "d_mean", "sd_r_pct", "note"):
            assert entry.get(key) is not None, f"entrada de {mode} sem `{key}`"
        assert entry["m_r_runs"] == frozen["preregistration"]["m_r_runs"], (
            f"{mode}: regeneração com M diferente do pré-registro não é comparável"
        )
        # A razão entre as duas estimativas do denominador é a quantidade que o
        # leitor precisa ver; reportá-la aqui evita que alguém a recalcule errado.
        first = measured[mode]["serrf"]["sd_r_pct_for_ca_3_2"]
        ratio = max(first, entry["sd_r_pct"]) / min(first, entry["sd_r_pct"])
        assert ratio > 1.0, (
            f"{mode}: as duas estimativas de sd_r são idênticas. Com M = 5 e 4 graus "
            "de liberdade isso não deveria acontecer — verificar se a regeneração "
            "leu o próprio fixture em vez de medir."
        )

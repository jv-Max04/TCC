"""Ensaio de transcrição dos passos 1 a 3 do SERRF.

**A única verificação não estatística da fase.** Os passos 2 e 3 são
determinísticos dada a entrada, então a comparação é por **igualdade exata**: sem
tolerância a calibrar, sem denominador estimado, sem viés que possa passar. Não
sofre nenhuma das três ressalvas do critério — propensão a empate,
incerteza de `sd_R` em fator 2, viés sistemático residual.

**O que este módulo fixou na primeira execução, e é a razão de a unidade existir.**
O passo 2 opera sobre a matriz **antes** do passo 1: as linhas 1053 e 1057
escalam `train` e `target`, os argumentos, enquanto o passo 1 muta `all`, que é
uma cópia. Alimentado com a matriz pós-passo-1 — a forma natural — `corrs_train`
diverge em **1,857** sobre uma quantidade que vive em `[-1, 1]`. Não é
imprecisão: é matriz errada, e o efeito apareceria só como divergência
estatística no critério, indistinguível de ruído do `ranger`.

**Regra de comparação da ordem.** Onde os valores são exatos, ordem exata;
onde não são, cobra-se o **conjunto**. As correlações batem a ~5e-16, então onde o
R tem empate exato o Python tem diferença de um bit e `order` desempata por
índice — ordem diferente sem que nenhum esteja errado.

**E a ordem não afeta o resultado neste caminho**, o que está medido e não
suposto: ela só decide quais colunas o passo 4 descarta por `NA`, e nenhuma
feature tem desvio nulo ou ausente nos QCs depois do passo 1. Ver
`test_o_passo_4_nao_descarta_coluna_nenhuma`.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from refino.correction._serrf_steps import (
    correlation_matrices,
    predictor_order,
    select_predictors,
    step1_params,
)

pytestmark = pytest.mark.golden

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURE_JSON = REPO / "data" / "fixtures" / "serrf_intermediates_r.json"
FIXTURE_NPZ = REPO / "data" / "fixtures" / "serrf_intermediates_r.npz"

# A linha "reprodução aritmética pura" da especificação. Não igualdade bit a bit: Spearman
# é Pearson sobre postos, e a ordem de acumulação difere entre R e `numpy` — uma
# asserção exata aqui mediria a soma, não a transcrição.
CORRELATION_TOLERANCE = 1e-12

# Congeladas, e são **observação medida**, não limiar escolhido — a
# distinção importa porque a especificação proíbe declarar tolerância por gosto
# sobre quantidade estocástica, e isto não é nem tolerância nem estocástico: é
# uma propriedade determinística do dado, congelada para detectar MUDANÇA.
#
# Com 7 QCs, o Spearman assume poucos valores discretos e quase tudo empata —
# medido, 98,95% no `train`. Com 61 amostras o `target` empata muito menos. Os
# tetos ficam logo acima do medido: se a fração subir, a cláusula por conjunto
# passou a cobrir mais do que quando foi escrita.
MAX_TIED_FRACTION_TRAIN = 0.995
MAX_TIED_FRACTION_TARGET = 0.60
TIE_EPSILON = 1e-12


@pytest.fixture(scope="module")
def frozen() -> dict[str, Any]:
    if not FIXTURE_JSON.is_file():
        pytest.skip(
            "serrf_intermediates_r.json ausente; a fixture é versionada, restaure-a do repositório"
        )
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def sidecar() -> Any:
    if not FIXTURE_NPZ.is_file():
        pytest.skip(
            "serrf_intermediates_r.npz ausente; a fixture é versionada, restaure-a do repositório"
        )
    return np.load(FIXTURE_NPZ)


@pytest.fixture(scope="module")
def masks(frozen: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    is_qc = np.array([t == "qc" for t in frozen["sample_type_by_column"]])
    in_batch = np.array([b == "A" for b in frozen["batch_by_column"]])
    return is_qc, in_batch


# ---------------------------------------------------------------------------
# O par JSON + sidecar
# ---------------------------------------------------------------------------


def test_o_digest_do_sidecar_bate(frozen: dict[str, Any]) -> None:
    """Sidecar sem digest muda sem aparecer no diff."""
    recorded = frozen["provenance"]["sidecar_sha256"]
    assert recorded, "o JSON não registra o `sha256` do sidecar"
    actual = hashlib.sha256(FIXTURE_NPZ.read_bytes()).hexdigest()
    assert actual == recorded, (
        f"o `.npz` mudou sem o JSON acompanhar: {actual[:12]} contra {recorded[:12]}. "
        "Regenere o par pelo script; não edite o binário à mão."
    )


def test_o_fixture_nao_carrega_nome_de_amostra(frozen: dict[str, Any]) -> None:
    """Pseudônimo de paciente mais diagnóstico não sai de `data/reference/`."""
    import re

    pattern = re.compile(r"_ID\d+_")
    offending = [n for n in frozen["feature_names"] if pattern.search(n)]
    assert not offending, f"nome de feature parece nome de amostra: {offending[:3]}"


def test_o_fixture_declara_o_escopo_que_nao_cobre(frozen: dict[str, Any]) -> None:
    """A chamada de validação cruzada foi filtrada, e isso é escopo.

    Sem a declaração dentro do artefato, alguém lê "os passos 1 a 3 estão
    verificados" como válido em todos os contextos — e a etapa seguinte usa os mesmos passos
    num regime de duas colunas que esta unidade não tocou.
    """
    scope = frozen["provenance"]["scope"]
    text = scope if isinstance(scope, str) else " ".join(scope)
    assert "filtradas por ncol" in text
    assert "validacao cruzada" in text or "validação cruzada" in text


# ---------------------------------------------------------------------------
# Passo 1 — parâmetros, por igualdade exata
# ---------------------------------------------------------------------------


def test_ca_u4_passo1_parametros_batem_exatamente(
    frozen: dict[str, Any], sidecar: Any, masks: tuple[np.ndarray, np.ndarray]
) -> None:
    """O discriminante de F1, por igualdade e não por amostragem.

    Sob a leitura errada da v1.2 — "em torno do mínimo positivo" — o parâmetro
    estaria entre 1e2 e 1e9. Aqui ele é 1, exato, porque `min` é avaliado sobre o
    vetor que ainda contém os zeros.
    """
    _, in_batch = masks
    before = sidecar["step1_before"]
    j_values = sidecar["step1_params_j"]
    is_na_phase = sidecar["step1_params_is_na_phase"]
    r_min = sidecar["step1_params_min_non_na"]
    r_zeros = sidecar["step1_params_n_zeros"]

    checked = 0
    for k in range(len(j_values)):
        if is_na_phase[k]:
            continue  # o ramo dos NA vê o vetor já modificado; não é reconstituível daqui
        j = int(j_values[k])
        params = step1_params(before[j - 1, in_batch], phase="zeros")
        assert params.min_non_na == r_min[k], (
            f"feature {j}: min {params.min_non_na} contra {r_min[k]} do R"
        )
        assert params.n_zeros == int(r_zeros[k])
        checked += 1

    assert checked == frozen["step1"]["n_feature_batch_pairs"], (
        f"conferi {checked} pares e o fixture declara {frozen['step1']['n_feature_batch_pairs']}"
    )


def test_ca_u4_passo1_media_e_um_e_desvio_e_centesimo_onde_ha_zero(
    frozen: dict[str, Any], sidecar: Any, masks: tuple[np.ndarray, np.ndarray]
) -> None:
    """**Independência de magnitude, e ela é exata.**

    Para toda `(feature, lote)` com ao menos um zero, `min = 0` e portanto
    `mean = 1` e `sd = 0.01` — quaisquer que sejam as magnitudes da feature. É a
    asserção que reprova a especificação da v1.2 por ordens de grandeza.
    """
    _, in_batch = masks
    before = sidecar["step1_before"]
    j_values = sidecar["step1_params_j"]
    is_na_phase = sidecar["step1_params_is_na_phase"]
    r_zeros = sidecar["step1_params_n_zeros"]

    with_zero = 0
    magnitudes: list[float] = []
    for k in range(len(j_values)):
        if is_na_phase[k] or int(r_zeros[k]) == 0:
            continue
        j = int(j_values[k])
        row = before[j - 1, in_batch]
        params = step1_params(row, phase="zeros")
        assert params.mean == 1.0, f"feature {j}: mean {params.mean}, esperado 1.0 exato"
        assert params.sd == pytest.approx(0.01, abs=0.0, rel=1e-15)
        positive = row[np.isfinite(row) & (row > 0)]
        if positive.size:
            magnitudes.append(float(np.min(positive)))
        with_zero += 1

    assert with_zero == frozen["step1"]["n_with_zero"]
    assert with_zero == frozen["step1"]["n_min_is_zero"], (
        "alguma feature com zero teve `min` diferente de zero — o mecanismo de F1 mudou"
    )
    # A independência de magnitude não é retórica: o mínimo positivo varia por
    # ordens de grandeza e o parâmetro não varia nada.
    assert max(magnitudes) / min(magnitudes) > 1e3, (
        "as magnitudes do subconjunto não variam o bastante para a asserção ter dente"
    )


# ---------------------------------------------------------------------------
# Passo 2 — igualdade exata, e a matriz certa
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def python_correlations(
    sidecar: Any, masks: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    is_qc, in_batch = masks
    return correlation_matrices(sidecar["step1_before"], is_qc=is_qc, in_batch=in_batch)


def test_ca_u4_passo2_reproduz_as_correlacoes_do_r(
    sidecar: Any, python_correlations: tuple[np.ndarray, np.ndarray]
) -> None:
    """As duas matrizes, na tolerância de reprodução aritmética pura da especificação."""
    train, target = python_correlations
    subset = [int(v) for v in sidecar["corr_subset_1based"]]
    for label, mine, theirs in (
        ("train", train, sidecar["corrs_train_A"]),
        ("target", target, sidecar["corrs_target_A"]),
    ):
        worst = 0.0
        for position, j in enumerate(subset):
            a = mine[:, j - 1]
            b = theirs[:, position]
            both = np.isfinite(a) & np.isfinite(b)
            assert np.array_equal(np.isfinite(a), np.isfinite(b)), (
                f"corrs_{label}, feature {j}: `NA` em posições diferentes"
            )
            if both.any():
                worst = max(worst, float(np.max(np.abs(a[both] - b[both]))))
        assert worst < CORRELATION_TOLERANCE, (
            f"corrs_{label}: max |dif| = {worst:.3e}, acima de {CORRELATION_TOLERANCE:.0e}"
        )


def test_ca_u4_passo2_a_matriz_pos_passo1_reprova(
    sidecar: Any, masks: tuple[np.ndarray, np.ndarray]
) -> None:
    """**O teste que prova que o ensaio consegue reprovar** — a correção fica fixada.

    Alimentar o passo 2 com a saída do passo 1 é a **forma natural** de escrever,
    e produz matriz errada. Sem esta asserção, a correção poderia ser
    desfeita numa refatoração e nada acusaria até o critério divergir, onde é
    indistinguível de ruído do `ranger`.
    """
    is_qc, in_batch = masks
    train, _ = correlation_matrices(sidecar["step1_after"], is_qc=is_qc, in_batch=in_batch)
    subset = [int(v) for v in sidecar["corr_subset_1based"]]
    theirs = sidecar["corrs_train_A"]

    worst = 0.0
    for position, j in enumerate(subset):
        a = train[:, j - 1]
        b = theirs[:, position]
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            worst = max(worst, float(np.max(np.abs(a[both] - b[both]))))
    assert worst > 1.0, (
        f"alimentar o passo 2 com a matriz pós-passo-1 deu max |dif| = {worst:.3e}, "
        "quando deveria divergir grosseiramente. Ou o oráculo mudou, ou `all` deixou "
        "de ser cópia de `cbind(train, target)` — e a correção precisa ser reavaliada."
    )


# ---------------------------------------------------------------------------
# Passo 3 — conjunto exato, ordem onde os valores permitem
# ---------------------------------------------------------------------------


def test_ca_u4_passo3_sel_var_bate_como_conjunto(
    sidecar: Any, python_correlations: tuple[np.ndarray, np.ndarray], frozen: dict[str, Any]
) -> None:
    """Cláusula 1: o **conjunto** é o que a linha 1094 constrói e o passo 4 usa."""
    train, target = python_correlations
    subset = [int(v) for v in sidecar["corr_subset_1based"]]
    j_values = [int(v) for v in sidecar["sel_var_j"]]
    flat = sidecar["sel_var_flat"]
    offsets = sidecar["sel_var_offsets"]
    by_feature = {
        j: set(int(v) for v in flat[offsets[i] : offsets[i + 1]]) for i, j in enumerate(j_values)
    }

    num = 10
    compared = 0
    for j in subset:
        if j not in by_feature:
            continue
        mine = set(select_predictors(train, target, j, num))
        assert mine == by_feature[j], (
            f"feature {j}: sel_var difere como conjunto.\n"
            f"  só no Python: {sorted(mine - by_feature[j])}\n"
            f"  só no R:      {sorted(by_feature[j] - mine)}"
        )
        compared += 1
    assert compared == len(subset), f"comparei {compared} de {len(subset)} features"


def test_ca_u4_passo3_a_fracao_de_empate_esta_congelada(
    sidecar: Any, python_correlations: tuple[np.ndarray, np.ndarray]
) -> None:
    """Cláusula 3: se o empate crescer, a cláusula protege mais do que deveria.

    É a guarda contra o afrouxamento silencioso chegando pelos dados em
    vez de pelo código — a métrica pararia de medir sem ninguém ter editado teste.
    """
    subset = [int(v) for v in sidecar["corr_subset_1based"]]
    for label, matrix, ceiling in (
        ("train", python_correlations[0], MAX_TIED_FRACTION_TRAIN),
        ("target", python_correlations[1], MAX_TIED_FRACTION_TARGET),
    ):
        fractions = []
        for j in subset:
            column = np.abs(matrix[:, j - 1])
            finite = column[np.isfinite(column)]
            if finite.size < 2:
                continue
            distinct = np.unique(np.round(finite / max(TIE_EPSILON, 1e-300)) * TIE_EPSILON)
            fractions.append(1.0 - distinct.size / finite.size)
        worst = max(fractions)
        assert worst <= ceiling, (
            f"corrs_{label}: {worst:.2%} das posições empatadas, acima do teto "
            f"congelado de {ceiling:.0%}. A cláusula por conjunto passou a cobrir "
            "mais do que quando foi escrita — reavaliá-la antes de subir o teto."
        )


def test_o_passo_4_nao_descarta_coluna_nenhuma(
    sidecar: Any, masks: tuple[np.ndarray, np.ndarray]
) -> None:
    """**A medição que torna a cláusula de ordem barata**.

    A ordem de `sel_var` só afeta o resultado através do descarte de colunas com
    `NA` no passo 4, linhas 1110 a 1114. `NA` ali vem de `scale` sobre feature
    constante nos QCs. Depois do passo 1 isso não acontece: ele substitui zeros e
    ausentes por sorteios distintos.

    **Feature sem zeros e constante por natureza continuaria sendo descartada** —
    é propriedade do dado, não do desenho —, e é por isso que este teste
    mede em vez de afirmar invariante. Se algum dia contar mais que zero, a ordem
    passa a importar e a cláusula precisa ser reavaliada.
    """
    is_qc, _ = masks
    qc_values = sidecar["step1_after"][:, is_qc]
    flat = sidecar["sel_var_flat"]
    used = np.unique(flat)

    rows = qc_values[used - 1]
    with np.errstate(invalid="ignore"):
        constant = int(np.sum(np.nanstd(rows, axis=1, ddof=1) == 0))
    missing = int(np.sum(np.isnan(rows).any(axis=1)))

    assert constant == 0 and missing == 0, (
        f"o passo 4 descartaria {constant + missing} colunas — {constant} constantes "
        f"e {missing} com ausente. A ordem de sel_var passou a afetar o resultado, "
        "e a cláusula por conjunto deixou de ser suficiente."
    )


def test_a_ordem_bate_onde_os_valores_sao_distintos(
    sidecar: Any, python_correlations: tuple[np.ndarray, np.ndarray]
) -> None:
    """Cláusula 2: ordem exata só onde `|corr|` é distinto além do épsilon.

    O `corrs_target` — 61 pontos, poucos empates — fecha a ordem inteira. O
    `corrs_train` — 7 pontos, empates densos — não fecha, e não deve: ali o R e o
    Python diferem em um bit e `order` desempata por índice.
    """
    subset = [int(v) for v in sidecar["corr_subset_1based"]]
    target = python_correlations[1]
    theirs = sidecar["corrs_target_A"]

    exact = 0
    for position, j in enumerate(subset):
        mine = predictor_order(target, j)
        column = np.abs(theirs[:, position])
        key = np.where(np.isnan(-column), np.inf, -column)
        r_order = np.argsort(key, kind="stable") + 1
        if np.array_equal(mine, r_order):
            exact += 1
    assert exact == len(subset), (
        f"a ordem do `corrs_target` bateu em {exact} de {len(subset)}. Ali os valores "
        "são distintos o bastante para a ordem ser determinada, então divergência é "
        "defeito de transcrição e não empate."
    )


# ---------------------------------------------------------------------------
# O `--check` do gerador, e a exclusão por declaração
# ---------------------------------------------------------------------------


def _comparable_arrays() -> Any:
    """Importa a função do gerador sem torná-lo pacote."""
    import importlib.util

    caminho = REPO / "scripts" / "build_serrf_intermediates.py"
    spec = importlib.util.spec_from_file_location("_build_serrf", caminho)
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo.comparable_arrays


def test_o_check_exclui_exatamente_os_arrays_declarados(frozen: dict[str, Any]) -> None:
    """A exclusão vem da **declaração** no fixture, não de uma lista no script.

    É o que torna o `--check` escrevível: sem a partição, quem o escrevesse
    compararia tudo, veria `step1_after` reprovar a cada execução e concluiria que o
    GERADOR está quebrado — quando o irreprodutível é o oráculo (portão que
    sempre reprova).
    """
    comparable_arrays = _comparable_arrays()
    declaration = frozen["provenance"]["reproducibility"]
    todos = [*declaration["reproducible_arrays"], *declaration["non_reproducible_arrays"]]

    comparados = comparable_arrays(declaration, todos)
    assert set(comparados) == set(declaration["reproducible_arrays"])
    assert not set(comparados) & set(declaration["non_reproducible_arrays"])


def test_o_check_levanta_se_a_declaracao_nomear_array_inexistente(
    frozen: dict[str, Any],
) -> None:
    """Declaração órfã não pode passar em silêncio.

    Isenção que sobreviveu ao que isentava deixa de excluir o que existe e passa a
    não excluir nada — e o desfecho seria um `--check` que compara o array
    irreprodutível sem que ninguém tenha mudado a declaração de propósito.
    """
    comparable_arrays = _comparable_arrays()
    declaration = {"non_reproducible_arrays": ["array_que_nao_existe"]}
    with pytest.raises(SystemExit, match="nomeia"):
        comparable_arrays(declaration, ["step1_before"])


def test_a_declaracao_versionada_cobre_os_dois_arrays_sorteados(frozen: dict[str, Any]) -> None:
    """Os dois nomes, explicitamente, porque são o achado e não detalhe.

    `step1_after` carrega os sorteios de `rnorm`; `min_non_na` os carrega **só na
    fase NA**, porque na fase de zeros o `min` é avaliado sobre o vetor que ainda os
    contém e vale 0 — o achado F1 aparecendo como propriedade do fixture.
    """
    declarados = set(frozen["provenance"]["reproducibility"]["non_reproducible_arrays"])
    assert declarados == {"step1_after", "step1_params_min_non_na"}

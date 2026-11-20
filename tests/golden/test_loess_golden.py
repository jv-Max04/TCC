"""Golden do `batch_loess` por feature.

**A quantidade é a razão POR FEATURE, e não a razão de medianas**. Cinco
sessões de investigação compararam mediana da nossa divergência sobre mediana do
piso do oráculo, por modo, e obtiveram 8,4x / 1,0x / 2,6x / 2,6x — um padrão que
parecia exigir um segundo mecanismo. Refeito como mediana das razões por feature:
1,30 / 1,03 / 1,55 / 1,10, amplitude de 1,51x. **Não havia resíduo.**

O mecanismo do artefato: numerador e denominador dependem ambos do RSD da feature
— `rho` de 0,300 a 0,632 entre o piso e o RSD, nos quatro modos —, então agregar
antes de dividir faz a diferença entre as distribuições de RSD dos modos aparecer
como diferença de razão.

**Mediana e p90, os dois congelados.** Mediana sozinha é frouxa: metade das
features poderia degradar e passar. O p90 fecha essa porta e não custa nada a
mais.

**O que a comparação com o `batch_loess` pode e não pode cobrar.** O golden não é
reproduzível nem por si mesmo — o `crossval` sorteia dobras sem semente alcançável
—, então exatidão é impossível e o piso do próprio oráculo é o denominador honesto.
Razão em torno de **1** significa "divergimos do golden tanto quanto duas execuções
do R divergem entre si", que é o melhor definível.

E `p25 < 1` nos quatro modos: para um quarto das features estamos **mais perto** do
golden do que duas execuções do R estão uma da outra. Nenhuma razão de medianas
mostraria isso.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from refino.correction.loess import LoessCorrector
from refino.domain.matrix import RefinoMatrix
from tests.conftest import matrix_from_normalized

pytestmark = [pytest.mark.golden, pytest.mark.slow]

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURE_JSON = REPO / "data" / "fixtures" / "loess_golden_r.json"
FIXTURE_NPZ = REPO / "data" / "fixtures" / "loess_golden_r.npz"

MODES = ("pos_hilic", "neg_hilic", "pos_c18", "neg_c18")

# As quatro contagens da especificação, observação sobre estes quatro arquivos.
EXPECTED_UNCORRECTED = {"pos_hilic": 1561, "neg_hilic": 165, "pos_c18": 10, "neg_c18": 20}

# Folga sobre os valores congelados, **derivada e não declarada**.
#
# `SLACK` é margem sobre o PORTÃO, e a distinção com `RADIUS_INFLATION` é a que
# decide o que ele precisa ter: aquele entra no número corrigido, e errar nele
# corrompe a saída; este só decide quando o teste reprova. Logo não precisa de
# derivação teórica — precisa de justificativa sobre a variabilidade que absorve.
#
# A variabilidade que ele vê é **uma**: o piso vem de cinco execuções de um oráculo
# sem `set.seed` alcançável, e regenerar o fixture com outras cinco o move. Medida
# por jackknife sobre as execuções: `cv` de 0,205 no pior par (modo, estatística),
# que é o p90 do `pos_hilic`. A `k = 3` do pré-registro dá `1 + 3 x 0,205 = 1,62`.
#
# A variabilidade por **amostra** foi medida e está **fora** da margem de
# propósito: os 400 índices estão congelados no sidecar e o teste os lê de lá, então
# o portão compara a mesma amostra com a mesma amostra. Margem comprada para uma
# variação que o portão nunca vê só serve para deixar passar regressão.
#
# **O 1,5 anterior era apertado demais**, e o valor é o mesmo da direção contrária:
# 61% de variação possível do oráculo contra 50% concedidos significa que regenerar
# o piso poderia reprovar o portão sem nenhuma regressão no corretor.
SLACK = 1.62


@pytest.fixture(scope="module")
def frozen() -> dict[str, Any]:
    if not FIXTURE_JSON.is_file():
        pytest.skip(
            "loess_golden_r.json ausente; a fixture é versionada, restaure-a do repositório"
        )
    return json.loads(FIXTURE_JSON.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def sidecar() -> Any:
    if not FIXTURE_NPZ.is_file():
        pytest.skip("loess_golden_r.npz ausente; a fixture é versionada, restaure-a do repositório")
    return np.load(FIXTURE_NPZ)


def test_o_digest_do_sidecar_bate(frozen: dict[str, Any]) -> None:
    """Sidecar sem digest muda sem aparecer no diff."""
    recorded = frozen["provenance"]["sidecar_sha256"]
    actual = hashlib.sha256(FIXTURE_NPZ.read_bytes()).hexdigest()
    assert actual == recorded, f"o `.npz` mudou sem o JSON acompanhar: {actual[:12]}"


def test_o_slack_do_teste_e_o_derivado_no_fixture(frozen: dict[str, Any]) -> None:
    """Paridade entre a constante e a derivação, nas duas direções.

    Sem isto o `SLACK` volta a ser número escolhido à mão: alguém afrouxa a
    constante aqui e a derivação no fixture continua dizendo outra coisa, ou
    regenera o fixture com `k` diferente e o teste segue no valor antigo. Nenhuma
    das duas aparece no diff como afrouxamento.
    """
    derived = frozen["slack"]
    assert derived["value"] == SLACK, (
        f"o teste usa SLACK = {SLACK} e o fixture derivou {derived['value']} "
        f"(exato {derived['derived_exact']:.4f}, pior caso {derived['worst_case']}). "
        "Um dos dois foi mudado sem o outro."
    )
    assert derived["k"] == 3.0, (
        f"a derivação usou k = {derived['k']}; o pré-registro fixou 3, e exceder "
        "k * sd não autoriza aumentar k."
    )


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_6_o_conjunto_nao_corrigido_do_golden_bate_a_contagem(sidecar: Any, mode: str) -> None:
    """As 1.561 / 165 / 20 / 10 da especificação, contra o artefato do laboratório.

    **Contra o golden, não contra outra execução do R**: o conjunto do R é
    ele próprio estocástico — 32 e 26 contra 20 no `neg_c18` —, então cobrar
    igualdade contra o oráculo seria cobrar contra alvo móvel. O artefato
    congelado é fixo.
    """
    untouched = sidecar[f"golden_uncorrected_{mode}"]
    assert int(untouched.sum()) == EXPECTED_UNCORRECTED[mode], (
        f"{mode}: {int(untouched.sum())} features bit-idênticas à matriz `none`, "
        f"esperadas {EXPECTED_UNCORRECTED[mode]}. O golden mudou."
    )


@pytest.mark.parametrize("mode", MODES)
def test_o_piso_por_feature_esta_no_fixture_e_e_positivo(sidecar: Any, mode: str) -> None:
    """O denominador correto existe, por feature, e não é degenerado.

    Sem esta asserção o teste seguinte poderia dividir por `nan` em toda feature e
    passar por vacuidade — nenhuma razão calculada, nenhuma comparação feita.
    """
    floor = sidecar[f"floor_{mode}"]
    usable = np.isfinite(floor) & (floor > 0)
    assert int(usable.sum()) > 100, (
        f"{mode}: só {int(usable.sum())} features com piso utilizável; o denominador "
        "não sustenta a comparação"
    )


def _ratios(mode: str, sidecar: Any, threshold: float) -> np.ndarray:
    """Razões por feature sobre a amostra congelada no fixture."""
    none_matrix = matrix_from_normalized(mode, "none")
    golden = matrix_from_normalized(mode, "batch_loess")
    floor = sidecar[f"floor_{mode}"]
    untouched = sidecar[f"golden_uncorrected_{mode}"]
    indices = sidecar[f"sample_{mode}"]

    subset = RefinoMatrix(
        values=np.array(none_matrix.values[:, indices], copy=True),
        feature_names=tuple(none_matrix.feature_names[j] for j in indices),
        samples=none_matrix.samples,
        mode=mode,
    )
    result = LoessCorrector().fit_transform(subset, seed=42)
    marked = {name for features in result.uncorrected.values() for name in features}

    out: list[float] = []
    for position, j in enumerate(indices):
        name = none_matrix.feature_names[j]
        if untouched[j] or name in marked:
            continue
        mine = result.matrix.values[:, position]
        theirs = golden.values[:, j]
        usable = np.isfinite(mine) & np.isfinite(theirs) & (theirs != 0)
        if int(usable.sum()) <= 5 or not np.isfinite(floor[j]) or floor[j] <= threshold:
            continue
        divergence = float(
            np.median(np.abs(mine[usable] - theirs[usable]) / np.abs(theirs[usable]))
        )
        out.append(divergence / float(floor[j]))
    return np.array(out)


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_6_a_razao_por_feature_respeita_mediana_e_p90_congelados(
    frozen: dict[str, Any], sidecar: Any, mode: str
) -> None:
    """**Os dois percentis, e a mediana sozinha não bastaria.**

    Com só a mediana congelada, metade das features poderia degradar
    arbitrariamente e o teste passaria. O p90 fecha a cauda, e o par é o que torna
    a asserção sensível a degradação parcial.
    """
    expected = frozen["modes"][mode]
    ratios = _ratios(mode, sidecar, frozen["provenance"]["floor_near_zero_threshold"])

    assert ratios.size >= expected["n_ratios"] // 2, (
        f"{mode}: {ratios.size} razões contra {expected['n_ratios']} congeladas — "
        "a população encolheu, e a comparação deixou de ser a mesma"
    )

    median = float(np.median(ratios))
    p90 = float(np.percentile(ratios, 90))
    assert median <= expected["median"] * SLACK, (
        f"{mode}: mediana da razão {median:.3f}, congelada {expected['median']:.3f}. "
        "Divergimos do golden mais que o oráculo diverge de si mesmo, e o piso é "
        "por feature — isto não é artefato de agregação (AR40)."
    )
    assert p90 <= expected["p90"] * SLACK, (
        f"{mode}: p90 da razão {p90:.3f}, congelado {expected['p90']:.3f}. A cauda "
        "degradou mesmo com a mediana intacta — é o que este percentil existe para pegar."
    )


@pytest.mark.parametrize("mode", MODES)
def test_um_quarto_das_features_bate_melhor_que_o_oraculo_contra_si_mesmo(
    sidecar: Any, frozen: dict[str, Any], mode: str
) -> None:
    """`p25 < 1`, e é a confirmação pela positiva que razão de medianas esconde.

    Para um quarto das features a nossa divergência contra o golden é **menor** que
    a divergência entre duas execuções do R. Não é anomalia: é o que se espera
    quando as duas quantidades são governadas pelo mesmo ruído e têm magnitude
    comparável — e é impossível de ver agregando antes de dividir.
    """
    ratios = _ratios(mode, sidecar, frozen["provenance"]["floor_near_zero_threshold"])
    p25 = float(np.percentile(ratios, 25))
    assert p25 < 1.0, (
        f"{mode}: p25 da razão é {p25:.3f}. Se subiu acima de 1, deixamos de estar "
        "dentro do ruído do oráculo em qualquer parte da população."
    )


def test_sobre_a_mesma_populacao_as_duas_razoes_coincidem(
    frozen: dict[str, Any], sidecar: Any
) -> None:
    """**O teste que corrigiu a explicação do artefato, e a correção é a parte útil.**

    A armadilha foi registrada dizendo que o artefato vinha da **ordem das
    operações** — razão de agregados contra agregado de razões. Este teste foi
    escrito para prová-lo e mostrou o contrário: sobre a **mesma população**, as
    duas formas quase coincidem — 1,277 contra 1,298, dentro de 5%.

    Logo o artefato de cinco sessões não veio da ordem das operações. Veio de
    numerador e denominador terem sido calculados sobre **populações diferentes**
    e com ordens de agregação diferentes: o piso de 0,0048 era
    `mediana_pares(mediana_features(...))`, agregando features **dentro de cada
    par** e sobre as que aquele par corrigiu; o numerador era sobre as features que
    sobreviviam às exclusões da comparação. Duas populações, dois níveis de
    agregação, e a divisão entre elas não significava nada.

    **Este teste é o que impede a explicação errada de voltar.** Ele afirma a
    coincidência, que é o fato; se algum dia as duas formas divergirem sobre a
    mesma população, então a ordem das operações **passou** a importar e a explicação
    original volta a valer.
    """
    mode = "pos_hilic"
    threshold = frozen["provenance"]["floor_near_zero_threshold"]

    none_matrix = matrix_from_normalized(mode, "none")
    golden = matrix_from_normalized(mode, "batch_loess")
    floor = sidecar[f"floor_{mode}"]
    untouched = sidecar[f"golden_uncorrected_{mode}"]
    indices = sidecar[f"sample_{mode}"]

    subset = RefinoMatrix(
        values=np.array(none_matrix.values[:, indices], copy=True),
        feature_names=tuple(none_matrix.feature_names[j] for j in indices),
        samples=none_matrix.samples,
        mode=mode,
    )
    result = LoessCorrector().fit_transform(subset, seed=42)
    marked = {name for features in result.uncorrected.values() for name in features}

    numerators: list[float] = []
    denominators: list[float] = []
    for position, j in enumerate(indices):
        if untouched[j] or none_matrix.feature_names[j] in marked:
            continue
        mine = result.matrix.values[:, position]
        theirs = golden.values[:, j]
        usable = np.isfinite(mine) & np.isfinite(theirs) & (theirs != 0)
        if int(usable.sum()) <= 5 or not np.isfinite(floor[j]) or floor[j] <= threshold:
            continue
        numerators.append(
            float(np.median(np.abs(mine[usable] - theirs[usable]) / np.abs(theirs[usable])))
        )
        denominators.append(float(floor[j]))

    numerator = np.array(numerators)
    denominator = np.array(denominators)
    median_of_ratios = float(np.median(numerator / denominator))
    ratio_of_medians = float(np.median(numerator)) / float(np.median(denominator))

    assert np.isclose(ratio_of_medians, median_of_ratios, rtol=0.10), (
        f"as duas formas divergiram sobre a MESMA população: razão de medianas "
        f"{ratio_of_medians:.3f} contra mediana das razões {median_of_ratios:.3f}. "
        "A ordem das operações passou a importar, e AR40 na formulação original "
        "volta a valer — reavaliar antes de tratar como equivalentes."
    )

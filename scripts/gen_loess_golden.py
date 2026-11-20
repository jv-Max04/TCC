#!/usr/bin/env python3
"""Congela o piso POR FEATURE do `batch_loess` e a razão contra ele.

Escreve **dois** artefatos que viajam juntos, no mesmo arranjo dos demais fixtures do R:

- `data/fixtures/loess_golden_r.npz` — o dado do lado do R: o piso por feature, o
  conjunto não corrigido do golden, e a amostra de features congelada;
- `data/fixtures/loess_golden_r.json` — o legível: procedência, definição
  **completa** do piso, as estatísticas da razão e a derivação do `SLACK`, mais o
  `sha256` do sidecar.

**Por que este script existe.** O
fixture foi gerado por script ad hoc, e `tests/golden/test_loess_golden.py` fazia
`pytest.skip` citando um caminho que não existia. `skip` que nomeia mecanismo
ausente é `skip` que **nunca vira execução**: dezoito testes que parecem existir e
nunca rodam, com a política de `skip` aceitando porque o motivo casa com um regime
declarado.

**E escrever o gerador foi a verificação.** Reproduzir o fixture achou três
lacunas que a procedência ad hoc tinha:

1. `"seed": 42` era **um campo para duas coisas**. A amostra foi
   sorteada com semente **1**; 42 é a semente do corretor. Lido como semente de
   amostragem, o valor registrado era simplesmente errado, e ninguém poderia
   reproduzir a amostra a partir dele.
2. A `floor_definition` registrada — "mediana, sobre os 10 pares, da mediana por
   célula de |a-b|/|b|" — **não determina o número**. Faltavam três condições:
   pares em que **qualquer** das duas execuções deixou a feature sem correção são
   descartados; célula só conta se houver mais de `MIN_CELLS` delas; e feature só
   recebe piso com pares válidos em pelo menos **metade** dos pares disponíveis.
   Sem elas o `pos_hilic` dá 346 features sem piso em vez de 2.516.
3. **Nenhum registro de quais execuções do R produziram o piso.** As cinco viviam
   em `/tmp`, e "regenere" não era verificável. Agora cada matriz usada entra com
   `sha256`, e a configuração é conferida contra a que o pré-registro da divergência
   mediu — mesma imagem, mesma entrada, mesma contagem de núcleos.

**O que precisa das cinco execuções e o que não precisa.** O piso precisa: ele é
propriedade do oráculo. As estatísticas da razão precisam do nosso corretor, e por
isso este gerador **importa `refino`** — ao contrário de `gen_insample_medians.py`,
onde importar reproduziria o defeito que o fixture existe para acusar. Aqui não há
esse risco: o numerador é a nossa divergência, e é justamente o que se mede aqui.

Uso:

    scripts/gen_loess_golden.py --runs-dir /tmp/refino-divergence-full
    scripts/gen_loess_golden.py --runs-dir DIR --check      # não escreve
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

FIXTURE_JSON = REPO / "data" / "fixtures" / "loess_golden_r.json"
FIXTURE_NPZ = REPO / "data" / "fixtures" / "loess_golden_r.npz"
U3_FIXTURE = REPO / "data" / "fixtures" / "r_serrf_divergence.json"

MODES = ("pos_hilic", "neg_hilic", "pos_c18", "neg_c18")

# ---------------------------------------------------------------------------
# As constantes da medição, cada uma com o motivo de ser o que é
# ---------------------------------------------------------------------------

# Semente da AMOSTRA de features, separada da do corretor de propósito: um campo
# `seed` para as duas foi o defeito de procedência que este script achou.
SAMPLE_SEED = 1
SAMPLE_SIZE = 400
CORRECTOR_SEED = 42

# Célula só conta com MAIS de `MIN_CELLS` utilizáveis. Mediana sobre cinco células
# ou menos, com 68 colunas disponíveis, é resumo de sobra de dado — e o mesmo corte
# vale no numerador, senão as duas pontas da razão descrevem populações diferentes,
# que descreveria populações diferentes.
MIN_CELLS = 5

# Feature só recebe piso se pelo menos METADE dos pares disponíveis for válida.
# Fração, não contagem: com 10 pares dá os 5 do fixture original, e sob jackknife
# (6 pares) dá 3 — a mesma regra, e não um segundo número escolhido à mão.
MIN_PAIR_FRACTION = 0.5

# Abaixo disto o piso é zero para efeito de divisão, e a feature sai da razão.
FLOOR_NEAR_ZERO = 1e-12

# `k` da margem, o mesmo do pré-registro da divergência. Ver a derivação do `SLACK`.
K_SLACK = 3.0

# Sementes de amostragem alternativas, para medir quanto o número congelado depende
# de QUAL amostra de 400 features caiu. Fixas para que `--check` seja determinístico.
SLACK_SAMPLE_SEEDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


def _load_divergence_module() -> Any:
    """Reusa os leitores de `gen_r_divergence.py` em vez de transcrevê-los.

    Transcrever `read_matrix` faria o piso e o `d(R_i, R_j)` serem
    lidos por dois códigos diferentes, e divergência entre eles apareceria como
    divergência do oráculo — régua diferente nas duas pontas.
    """
    spec = importlib.util.spec_from_file_location(
        "_gen_r_divergence", REPO / "scripts" / "gen_r_divergence.py"
    )
    if spec is None or spec.loader is None:  # pragma: no cover - caminho de import quebrado
        raise SystemExit("não foi possível carregar scripts/gen_r_divergence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# O piso
# ---------------------------------------------------------------------------


def uncorrected_mask(none_values: np.ndarray, method_values: np.ndarray) -> np.ndarray:
    """Features que o método devolveu **bit-idênticas** à matriz `none`.

    Igualdade exata e não tolerância: o `batch_loess` devolve a
    linha original sem marcar quando a curva ajustada teve valor negativo, e
    tolerância aqui colheria feature corrigida cuja correção foi pequena — outra
    coisa.
    """
    return np.array(
        [
            np.array_equal(none_values[i], method_values[i], equal_nan=True)
            for i in range(none_values.shape[0])
        ]
    )


def pair_floor(a: np.ndarray, b: np.ndarray, untouched: np.ndarray) -> np.ndarray:
    """Divergência por feature entre duas execuções do R, `nan` onde não se mede.

    `nan`, e **não zero**, quando qualquer das duas deixou a feature sem correção:
    ali `|a-b|` é zero por construção, e um zero entrando na mediana dos pares
    empurraria o piso para baixo dizendo "as duas execuções concordam" quando o
    fato é "nenhuma das duas corrigiu". Isso
    atinge 25,9% das features do `pos_hilic`.
    """
    usable = np.isfinite(a) & np.isfinite(b) & (b != 0)
    relative = np.where(usable, np.abs(a - b) / np.where(usable, np.abs(b), 1.0), np.nan)
    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(np.where(usable, relative, np.nan), axis=1)
    enough = usable.sum(axis=1) > MIN_CELLS
    return np.where(enough & ~untouched, medians, np.nan)


def feature_floor(matrices: list[np.ndarray], none_values: np.ndarray) -> np.ndarray:
    """Piso por feature: mediana, sobre os pares válidos, do `pair_floor`.

    Mediana e não média porque a distribuição por par é assimétrica à direita — e
    porque confundir os níveis de agregação desta quantidade leva a conclusões erradas.
    """
    untouched = [uncorrected_mask(none_values, m) for m in matrices]
    per_pair = np.array(
        [
            pair_floor(matrices[i], matrices[j], untouched[i] | untouched[j])
            for i, j in itertools.combinations(range(len(matrices)), 2)
        ]
    )
    valid = np.isfinite(per_pair).sum(axis=0)
    minimum = math.ceil(MIN_PAIR_FRACTION * per_pair.shape[0])
    with np.errstate(invalid="ignore"):
        floor = np.nanmedian(per_pair, axis=0)
    return np.where(valid >= minimum, floor, np.nan)


def sample_indices(n_features: int, seed: int) -> np.ndarray:
    """Amostra sem reposição, **não ordenada** — a ordem é a que o gerador sorteou."""
    return np.random.default_rng(seed).choice(n_features, size=SAMPLE_SIZE, replace=False)


# ---------------------------------------------------------------------------
# A razão
# ---------------------------------------------------------------------------


def ratios_for(
    mode: str,
    floor: np.ndarray,
    untouched: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Razões por feature, e a contagem excluída por piso degenerado.

    A contagem sai junto porque exclusão silenciosa é o que torna um portão frouxo
    sem parecer frouxo: 39 features fora no `pos_hilic` é informação, e
    `n_ratios` sozinho não a carrega.
    """
    from refino.correction.loess import LoessCorrector
    from refino.domain.matrix import RefinoMatrix
    from tests.conftest import matrix_from_normalized

    none_matrix = matrix_from_normalized(mode, "none")
    golden = matrix_from_normalized(mode, "batch_loess")

    subset = RefinoMatrix(
        values=np.array(none_matrix.values[:, indices], copy=True),
        feature_names=tuple(none_matrix.feature_names[j] for j in indices),
        samples=none_matrix.samples,
        mode=mode,
    )
    result = LoessCorrector().fit_transform(subset, seed=CORRECTOR_SEED)
    marked = {name for features in result.uncorrected.values() for name in features}

    out: list[float] = []
    excluded = 0
    for position, j in enumerate(indices):
        if untouched[j] or none_matrix.feature_names[j] in marked:
            continue
        mine = result.matrix.values[:, position]
        theirs = golden.values[:, j]
        usable = np.isfinite(mine) & np.isfinite(theirs) & (theirs != 0)
        if int(usable.sum()) <= MIN_CELLS or not np.isfinite(floor[j]):
            continue
        if floor[j] <= FLOOR_NEAR_ZERO:
            excluded += 1
            continue
        divergence = float(
            np.median(np.abs(mine[usable] - theirs[usable]) / np.abs(theirs[usable]))
        )
        out.append(divergence / float(floor[j]))
    return np.array(out), excluded


def statistics_of(ratios: np.ndarray, excluded: int, n_untouched: int) -> dict[str, Any]:
    return {
        "n_ratios": int(ratios.size),
        "n_excluded_floor_near_zero": excluded,
        "median": float(np.median(ratios)),
        "p90": float(np.percentile(ratios, 90)),
        "p25": float(np.percentile(ratios, 25)),
        "p75": float(np.percentile(ratios, 75)),
        "n_golden_uncorrected": n_untouched,
    }


# ---------------------------------------------------------------------------
# A derivação do `SLACK`
# ---------------------------------------------------------------------------

_SLACK_RATIONALE = (
    "O SLACK e margem sobre o portao, nao constante do resultado. A distincao "
    "importa: RADIUS_INFLATION entra no numero corrigido e errar nele corrompe a "
    "saida; o SLACK so decide quando o teste reprova. Por isso ele nao precisa de "
    "derivacao teorica — precisa de justificativa sobre a variabilidade que "
    "absorve, e e isso que esta medido aqui. Mesmo raciocinio do k = 3 do "
    "pre-registro, num portao mais simples."
)

_SLACK_SOURCES = (
    "Duas fontes candidatas, e so UMA entra na margem. (1) O PISO: vem de M "
    "execucoes de um oraculo sem set.seed alcancavel, e regenerar com outras M "
    "execucoes o move — esta ENTRA, porque o portao le o piso do sidecar e uma "
    "regeneracao legitima o troca. (2) A AMOSTRA de 400 features: esta NAO entra, "
    "porque os indices sorteados estao CONGELADOS no proprio sidecar e o teste os "
    "le de la. O portao compara a mesma amostra com a mesma amostra, e margem "
    "comprada para uma variacao que o portao nunca ve e margem que so serve para "
    "deixar passar regressao — conclusao correta sobre a quantidade errada. "
    "Ela esta MEDIDA e registrada porque diz outra coisa, que importa: "
    "quanto o numero congelado depende de qual amostra caiu."
)

_SLACK_NO_QUADRATIC_SUM = (
    "Nao ha soma em quadratura aqui, e a ausencia e o ponto. A correcao que o k do "
    "[CA-3.1] recebeu — somar sd_py e sd_R em quadratura e nao aritmeticamente — "
    "valia porque as DUAS dispersoes alcancavam a mesma quantidade comparada. Aqui "
    "a segunda fonte nao alcanca o portao, e somar duas fontes das quais uma nao "
    "atua e o mesmo erro com o sinal trocado: inflaria a margem em vez de reduzi-la."
)

_SAMPLE_CAVEAT = (
    "O que a variabilidade por amostra diz, e nao e sobre a margem: o p90 do "
    "pos_hilic tem cv de 0,32 entre dez amostras de 400 features. Logo o p90 "
    "congelado descreve ESTA amostra, e nao a populacao de features do modo. "
    "Cobra-lo como portao e legitimo — a amostra e a mesma nas duas pontas —, mas "
    "citar '7,76' como propriedade do pos_hilic seria a escala de medicao de AR16. "
    "A mediana e bem mais estavel: cv de 0,13 no pior modo."
)


def jackknife_over_runs(
    matrices: list[np.ndarray],
    none_values: np.ndarray,
    mode: str,
    untouched: np.ndarray,
    indices: np.ndarray,
    full: dict[str, Any],
) -> dict[str, Any]:
    """Variabilidade do número congelado quando UMA execução do R sai do piso.

    Jackknife e não bootstrap porque a unidade de reamostragem são as cinco
    execuções, que é o que uma regeneração de fato troca. O numerador não é
    recalculado: só o piso muda, então o corretor roda uma vez.
    """
    estimates: dict[str, list[float]] = {"median": [], "p90": []}
    for dropped in range(len(matrices)):
        kept = [m for i, m in enumerate(matrices) if i != dropped]
        floor = feature_floor(kept, none_values)
        ratios, _ = ratios_for(mode, floor, untouched, indices)
        estimates["median"].append(float(np.median(ratios)))
        estimates["p90"].append(float(np.percentile(ratios, 90)))

    out: dict[str, Any] = {"n_leave_one_out": len(matrices)}
    for statistic, values in estimates.items():
        array = np.asarray(values)
        n = array.size
        # sd do jackknife: sqrt((n-1)/n * soma dos quadrados dos desvios).
        sd = float(np.sqrt((n - 1) / n * np.sum((array - array.mean()) ** 2)))
        out[statistic] = {
            "leave_one_out": [float(v) for v in array],
            "sd_jackknife": sd,
            "cv": sd / float(full[statistic]),
        }
    return out


def sample_variability(
    mode: str,
    floor: np.ndarray,
    untouched: np.ndarray,
    n_features: int,
    full: dict[str, Any],
) -> dict[str, Any]:
    """Variabilidade quando OUTRA amostra de 400 features cai.

    Ela existe mesmo com a semente congelada: a semente fixa **qual** amostra, não
    torna a amostra representativa. Esta medida é o que responde se 400 bastam.
    """
    estimates: dict[str, list[float]] = {"median": [], "p90": []}
    for seed in SLACK_SAMPLE_SEEDS:
        ratios, _ = ratios_for(mode, floor, untouched, sample_indices(n_features, seed))
        estimates["median"].append(float(np.median(ratios)))
        estimates["p90"].append(float(np.percentile(ratios, 90)))

    out: dict[str, Any] = {"n_seeds": len(SLACK_SAMPLE_SEEDS), "seeds": list(SLACK_SAMPLE_SEEDS)}
    for statistic, values in estimates.items():
        array = np.asarray(values)
        sd = float(np.std(array, ddof=1))
        out[statistic] = {
            "by_seed": [float(v) for v in array],
            "sd_ddof1": sd,
            "cv": sd / float(full[statistic]),
        }
    return out


def derive_slack(per_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`SLACK = 1 + k * cv_oraculo`, pelo pior par (modo, estatística).

    Pelo **pior** e não pela média: um portão calibrado na média reprova sozinho no
    modo mais variável, e portão que reprova sem regressão é portão que se aprende
    a ignorar. E só `cv_oraculo` entra — ver `_SLACK_SOURCES`.
    """
    worst = {"value": 0.0, "mode": "", "statistic": ""}
    detail: dict[str, Any] = {}
    for mode, measured in per_mode.items():
        detail[mode] = {}
        for statistic in ("median", "p90"):
            cv_oracle = measured["jackknife"][statistic]["cv"]
            detail[mode][statistic] = {
                "cv_oracle_in_margin": cv_oracle,
                "cv_sample_excluded": measured["sample"][statistic]["cv"],
            }
            if cv_oracle > worst["value"]:
                worst = {"value": cv_oracle, "mode": mode, "statistic": statistic}

    exact = 1.0 + K_SLACK * float(worst["value"])
    return {
        "k": K_SLACK,
        "rationale": _SLACK_RATIONALE,
        "sources": _SLACK_SOURCES,
        "no_quadratic_sum": _SLACK_NO_QUADRATIC_SUM,
        "sample_caveat": _SAMPLE_CAVEAT,
        "formula": "SLACK = 1 + k * cv_oracle, pelo pior (modo, estatistica)",
        "worst_case": worst,
        "per_mode": detail,
        "derived_exact": exact,
        # Arredondado para cima em duas casas: portão cujo limite tem dezessete
        # dígitos significativos sugere precisão que a medição não tem.
        "value": math.ceil(exact * 100) / 100,
        "supersedes": (
            "SLACK = 1,5, DECLARADO e nao derivado, era APERTADO DEMAIS. O oraculo "
            "sozinho move o p90 do pos_hilic com cv de 0,205; a k = 3 isso e 61%, "
            "acima dos 50% que 1,5 concedia. Regenerar o piso com outras cinco "
            "execucoes do R poderia reprovar o portao SEM nenhuma regressao no "
            "corretor — e um portao que reprova sem regressao ensina a ignora-lo."
        ),
    }


# ---------------------------------------------------------------------------
# Procedência
# ---------------------------------------------------------------------------


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_same_runs_as_u3(mode: str, runs: list[dict[str, Any]]) -> str:
    """As cinco execuções são as que o pré-registro da divergência mediu, ou o piso é outro.

    Não é zelo: `cores_detected` particiona o `parSapply` e decide quais sorteios
    caem em qual worker. Um piso medido sob outra partição não é amostra da mesma
    distribuição que o `d(R_i, R_j)`, e comparar os dois seria
    comparar duas configurações.
    """
    if not U3_FIXTURE.is_file():
        return "pre-registro da U3 ausente; vinculo NAO verificado"
    frozen = json.loads(U3_FIXTURE.read_text(encoding="utf-8"))
    expected = frozen.get("provenance", {}).get("runs", {}).get(mode)
    if not expected:
        return f"pre-registro da U3 nao registra execucoes para {mode}; vinculo NAO verificado"
    keys = ("input_sha256", "image_digest", "r_version", "cores_detected", "detectcores_ratio")
    for key in keys:
        here = {json.dumps(r.get(key), sort_keys=True) for r in runs}
        there = {json.dumps(r.get(key), sort_keys=True) for r in expected}
        if here != there:
            raise SystemExit(
                f"{mode}: `{key}` das execucoes deste piso {sorted(here)} difere do que a U3 "
                f"pre-registrou {sorted(there)}. O piso e de outra configuracao, e "
                "compara-lo com o d(R_i, R_j) do [CA-3.16] misturaria duas."
            )
    return "conferido contra r_serrf_divergence.json: mesma entrada, imagem, R e nucleos"


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------


def build(runs_root: Path, with_slack: bool) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    divergence = _load_divergence_module()
    root = divergence.reference_root()

    from tests.conftest import matrix_from_normalized

    arrays: dict[str, np.ndarray] = {}
    stats: dict[str, Any] = {}
    provenance_runs: dict[str, Any] = {}
    slack_input: dict[str, dict[str, Any]] = {}

    for mode in MODES:
        mode_dir = runs_root / mode
        if not mode_dir.is_dir():
            raise SystemExit(
                f"ausente: {mode_dir}. As execucoes vem de scripts/gen_r_divergence.sh, "
                "que preserva e imprime o diretorio de trabalho ao final."
            )
        run_dirs = sorted(
            (d for d in mode_dir.iterdir() if d.is_dir() and d.name.startswith("run")),
            key=lambda d: int(d.name.removeprefix("run")),
        )
        if len(run_dirs) < 2:
            raise SystemExit(f"{mode}: {len(run_dirs)} execucao(oes); o piso precisa de pares")

        runs = divergence.run_provenance(run_dirs)
        divergence.assert_same_configuration(mode, runs)
        paths = [divergence.normalized_path(d, mode, "batch_loess") for d in run_dirs]
        for entry, path in zip(runs, paths, strict=True):
            entry["batch_loess_sha256"] = digest_of(path)
        link = assert_same_runs_as_u3(mode, runs)
        provenance_runs[mode] = {"link_to_u3": link, "runs": runs}

        matrices = [divergence.read_matrix(p)[2] for p in paths]
        if len({m.shape for m in matrices}) != 1:
            raise SystemExit(f"{mode}: as execucoes tem formas diferentes")
        _, _, none_values = divergence.read_matrix(
            divergence.normalized_path(run_dirs[0], mode, "none")
        )

        none_matrix = matrix_from_normalized(mode, "none")
        golden = matrix_from_normalized(mode, "batch_loess")
        n_features = none_matrix.n_features
        untouched = np.array(
            [
                np.array_equal(none_matrix.values[:, j], golden.values[:, j], equal_nan=True)
                for j in range(n_features)
            ]
        )

        floor = feature_floor(matrices, none_values)
        indices = sample_indices(n_features, SAMPLE_SEED)
        arrays[f"floor_{mode}"] = floor
        arrays[f"golden_uncorrected_{mode}"] = untouched
        arrays[f"sample_{mode}"] = indices.astype(np.int64)

        ratios, excluded = ratios_for(mode, floor, untouched, indices)
        stats[mode] = statistics_of(ratios, excluded, int(untouched.sum()))
        print(
            f"{mode:>10} n={ratios.size:>4} mediana={stats[mode]['median']:.4f} "
            f"p90={stats[mode]['p90']:.4f} excl={excluded}",
            file=sys.stderr,
        )

        if with_slack:
            slack_input[mode] = {
                "jackknife": jackknife_over_runs(
                    matrices, none_values, mode, untouched, indices, stats[mode]
                ),
                "sample": sample_variability(mode, floor, untouched, n_features, stats[mode]),
            }
            print(f"{mode:>10} slack medido", file=sys.stderr)

    payload: dict[str, Any] = {
        "provenance": {
            "command": "scripts/gen_loess_golden.py --runs-dir DIR",
            "generated_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "sidecar": FIXTURE_NPZ.name,
            "sidecar_sha256": None,
            "reference_root": str(root.relative_to(REPO)),
            "floor_definition": (
                "Piso POR FEATURE. Para cada par das M execucoes do R: mediana, sobre as "
                "celulas com |b| finito e nao nulo e ambos os lados finitos, de |a-b|/|b|; "
                "o par e DESCARTADO se qualquer das duas execucoes devolveu a feature "
                "bit-identica a matriz `none`, porque ali |a-b| e zero por construcao "
                "e o zero diria 'concordam' onde o fato e 'nenhuma corrigiu'; e a celula so "
                f"conta se houver mais de {MIN_CELLS} utilizaveis. O piso da feature e a "
                "mediana sobre os pares validos, e fica indefinido se os pares validos forem "
                f"menos de {MIN_PAIR_FRACTION:.0%} dos pares disponiveis."
            ),
            "floor_near_zero_threshold": FLOOR_NEAR_ZERO,
            "min_cells": MIN_CELLS,
            "min_pair_fraction": MIN_PAIR_FRACTION,
            "sample_size": SAMPLE_SIZE,
            "sample_seed": SAMPLE_SEED,
            "corrector_seed": CORRECTOR_SEED,
            "seed_note": (
                "DOIS campos porque sao duas coisas. A versao ad hoc deste fixture "
                "registrava um unico `seed: 42`, que e a semente do CORRETOR; a amostra "
                "foi sorteada com 1. Lido como semente de amostragem o valor era errado, e "
                "ninguem reproduziria a amostra a partir dele."
            ),
            # Vazia e EXPLICITA, e a diferenca com a ausencia e o ponto: ausencia
            # nao distingue "tudo reproduz" de "ninguem verificou". Aqui
            # `--check` compara os doze arrays e passa, o que e a evidencia.
            "reproducibility": {
                "reproducible_arrays": sorted(arrays),
                "non_reproducible_arrays": [],
                "measured": (
                    "--check reproduz os doze arrays bit a bit a partir das mesmas cinco "
                    "execucoes. O piso vem de matrizes JA gravadas em disco, e nao de nova "
                    "chamada ao oraculo: irreprodutibilidade do R entra pelo VALOR do piso, "
                    "e nao pela leitura dele."
                ),
                "caveat": (
                    "Reproduzivel dadas AS MESMAS cinco execucoes, que e o que o campo diz. "
                    "Com outras cinco o piso muda, e e essa variacao que o SLACK absorve — "
                    "cv de 0,205 por jackknife."
                ),
            },
            "runs": provenance_runs,
            "note": (
                "A razao POR FEATURE e a quantidade; razao de medianas e artefato de "
                "populacao e nivel de agregacao (AR40, na forma corrigida)."
            ),
        },
        "modes": stats,
    }
    if with_slack:
        payload["slack"] = derive_slack(slack_input)
        payload["slack"]["measurements"] = slack_input
    return payload, arrays


def serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


# **A granularidade em que `python_version` é comparado.** O argumento inteiro está em
# `scripts/gen_insample_medians.py`. Este é o TERCEIRO membro da classe, e foi achado por
# varredura e **não por reprovação**: o `--check` daqui exige `--runs-dir` com os despejos do R,
# então ninguém o corre por rotina e a divergência esperaria a próxima regeneração — onde a
# leitura natural seria *"a regeneração divergiu"*, sobre um artefato do oráculo.
#
# `tests/test_procedencia_comparavel.py` exercita esta função e as das outras três, e é ele que
# impede as quatro cópias de divergirem.
_PATCH_DO_PYTHON = re.compile(r'("python_version":\s*"\d+\.\d+)\.\d+"')


def comparavel(raw: str) -> list[str]:
    """Linhas que devem casar entre fixture e regeneração.

    `generated_at` muda por definição, e o `sidecar_sha256` depende do `.npz` que
    a regeneração escreveria — `--check` não escreve, então compara o digest do
    sidecar versionado à parte.

    `python_version` **fica**, truncado em `major.minor`: é a granularidade que os quatro pinos
    do projeto declaram, e comparar o patch tornava este arquivo mais estrito que eles.
    """
    skip = ('"generated_at"', '"sidecar_sha256"')
    return [
        _PATCH_DO_PYTHON.sub(r'\1"', line)
        for line in raw.splitlines()
        if not any(k in line for k in skip)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path, help="raiz com {modo}/run{i}/")
    parser.add_argument("--json", type=Path, default=FIXTURE_JSON)
    parser.add_argument("--npz", type=Path, default=FIXTURE_NPZ)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compara com o par versionado sem escrever; `generated_at` e o digest saem fora",
    )
    parser.add_argument(
        "--no-slack",
        action="store_true",
        help="pula a derivacao do SLACK, que custa 5 pisos + 10 amostras por modo",
    )
    args = parser.parse_args()

    payload, arrays = build(args.runs_dir, with_slack=not args.no_slack)

    if args.check:
        if not args.json.is_file() or not args.npz.is_file():
            print("par versionado ausente; rode sem --check", file=sys.stderr)
            return 1
        frozen = json.loads(args.json.read_text(encoding="utf-8"))
        recorded = frozen["provenance"]["sidecar_sha256"]
        actual = digest_of(args.npz)
        if actual != recorded:
            print(f"o .npz mudou sem o JSON acompanhar: {actual[:12]}", file=sys.stderr)
            return 1
        with np.load(args.npz) as sidecar:
            for name, value in arrays.items():
                if name not in sidecar.files:
                    print(f"o sidecar nao traz `{name}`", file=sys.stderr)
                    return 1
                if not np.array_equal(sidecar[name], value, equal_nan=True):
                    print(f"`{name}` divergiu da regeneracao", file=sys.stderr)
                    return 1
        if comparavel(serialize(frozen)) != comparavel(serialize(payload)):
            print("o JSON DIVERGE da regeneracao; rode sem --check", file=sys.stderr)
            return 1
        print("par identico a regeneracao (ignorando generated_at)")
        return 0

    np.savez_compressed(args.npz, **arrays)
    payload["provenance"]["sidecar_sha256"] = digest_of(args.npz)
    args.json.write_text(serialize(payload), encoding="utf-8")
    size = args.npz.stat().st_size / 1024
    print(f"escrito: {args.json.name} + {args.npz.name} ({size:.0f} KB)")
    if "slack" in payload:
        exact = payload["slack"]["derived_exact"]
        print(f"SLACK derivado: {payload['slack']['value']} (exato {exact:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

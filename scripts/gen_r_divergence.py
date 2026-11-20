#!/usr/bin/env python3
"""Mede a divergência do `tools.R` contra si mesmo e congela.

Lê `M` execuções do runner por modo, já produzidas, e escreve
`data/fixtures/r_serrf_divergence.json`.

**O que este arquivo é.** Um pré-registro. Os quatro parâmetros do bloco
`preregistration` — `k`, `N`, `M` e a cláusula de não negociação — são fixados
**antes** de existir qualquer corretor Python, e o `sha256` do arquivo entra no
tronco antes de haver número nosso para condicioná-los. É o que separa "derivar
da variância" de afrouxar tolerância com nome técnico.

**Por que este gerador IMPORTA `refino`, ao contrário de
`scripts/gen_insample_medians.py`.** Lá a proibição é o ponto: aquele fixture é o
oráculo da nossa própria fórmula de RSD, e importar apagaria a regressão que ele
existe para detectar. Aqui o papel é outro. A comparação de divergência é

    d(Python, R_i)  <=  k * d(R_i, R_j)

e as duas distâncias só são comparáveis se a **mesma régua** medir os dois lados.
Duas transcrições da fórmula no numerador e no denominador fariam a desigualdade
comparar quantidades diferentes, com a aparência de rigor extra. A
régua é `refino.metrics.rsd`, e ela não é auto-referencial aqui: ela
já está ancorada, a 1e-6, contra a coluna `none` do `QC - RSD.csv` do próprio R.

**Regeneração não reproduz o bloco `measured`, e isso é desenho**. O bloco
SERRF do R não chama `set.seed`, então outra geração mede outras
execuções. O que a regeneração confere exatamente é `preregistration`,
`provenance` e os fatos estruturais; para o medido ela afirma invariantes de
sinal e forma, e reporta.

Uso:

    .venv/bin/python scripts/gen_r_divergence.py --runs-dir DIR
    .venv/bin/python scripts/gen_r_divergence.py --runs-dir DIR --out FILE
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from refino.metrics.rsd import rsd  # noqa: E402

FIXTURE = REPO / "data" / "fixtures" / "r_serrf_divergence.json"
MANIFEST = REPO / "scripts" / "reference_data.sha256"

MODES = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")

# A variante e o estimador usados nos dois lados da comparação. In-sample
# porque é o estimador desta fase; `outlier_removed` porque é o que a `RSD()` do
# `tools.R` mede. Trocar qualquer um dos dois muda o número e
# inverte ranking — por isso viajam junto com a medição.
RSD_VARIANT = "outlier_removed"
RSD_ESTIMATOR = "insample"

# ---------------------------------------------------------------------------
# O pré-registro
# ---------------------------------------------------------------------------
#
# Estes quatro valores são a razão de o arquivo existir. Mudá-los depois de o
# `sha256` estar no tronco aparece no diff, que é o mecanismo inteiro.

K_CA_3_1 = 3.0
K_CA_3_2 = 3.0
N_PYTHON_SEEDS = 20
M_R_RUNS = 5

_K_JUSTIFICATION = (
    "k = 3, mesmo valor nos dois criterios. Dois fatores empurram acima do k = 2 "
    "usual. (a) Multiplicidade: o [CA-3.2] cobra quatro modos, oito com o "
    "batch_loess; a k = 2 a chance de ao menos um falso alarme e 1 - 0.954^8 ~= "
    "31%, e portao que reprova em um terco das execucoes sem defeito e o "
    "intermitente que este projeto ja gastou duas sessoes aprendendo a nao "
    "construir. A k = 3 cada comparacao fica em ~0.27% e a familia em ~2%. "
    "(b) O desvio e estimado, nao conhecido, e sd_r e a estimativa mais fragil "
    "porque M e pequeno. Nenhum motivo foi encontrado para k diferente entre os "
    "dois criterios, entao o valor e unico."
)

# O caveat que o registro carrega e que NAO e alavanca.
_K_CAVEAT = (
    "A taxa de familia de ~2% supoe sigma CONHECIDO. Com sigma estimado a "
    "referencia e t: t(19) a 0.27% bilateral e ~3.58, e sd_r com M = 5 tem 4 "
    "graus de liberdade, onde t(4) e ~5.6. Se sd_r dominar a soma quadratica, "
    "k = 3 corresponde a uma taxa de familia MAIOR que ~2%. Este caveat e "
    "registrado como medida, nao como permissao: exceder a tolerancia nao "
    "autoriza subir k, pela mesma clausula que vale para o vies."
)

_NO_NEGOTIATION = (
    "k vezes dispersao limita RUIDO, nao VIES. O achado F8 identificou diferenca "
    "sistematica entre ranger e scikit-learn e ela foi reduzida espelhando os "
    "hiperparametros, mas espelhar mtry e min.node.size nao iguala escolha de "
    "divisao, desempate nem bootstrap: pode sobrar vies residual, e nenhum k "
    "resolve vies. Portanto: exceder k * sd NAO autoriza aumentar k. O desfecho e "
    "investigar. Se a causa for diferenca sistematica identificada, ela vira "
    "divergencia com medicao propria e o criterio passa a cobrar a diferenca "
    "DESCONTADO o vies medido, com o vies versionado como o k. Alargar tolerancia "
    "para acomodar vies e a regra 2 do README violada com aparencia de rigor."
)

_TOLERANCE_FORMULA = (
    "tolerancia(modo) = k * sqrt(sd_python(modo)^2 + sd_r(modo)^2). Soma "
    "QUADRATICA, nao aritmetica: as duas fontes sao independentes — semente do "
    "Python de um lado, ausencia de set.seed no bloco SERRF do R do outro — "
    "entao variancias somam e desvios nao. A soma aritmetica infla 41.4% quando "
    "os dois desvios sao parecidos, e a folga vira o espaco onde um defeito de "
    "porte se esconde. Corrige a formula da secao 7 Fase 3 [CA-3.2]."
)

_QUANTITY_NOTE = (
    "sd_python e sd_r sao DESVIOS PADRAO amostrais (ddof=1), e o nome mudou "
    "junto com a quantidade de proposito. A secao 7 dizia 'a AMPLITUDE da mediana "
    "do SERRF em Python sobre 20 sementes': amplitude e max - min, e para n = 20 "
    "normal a razao amplitude/sd e ~3.79 — manter o rotulo 'dispersao' e trocar a "
    "quantidade inflaria a tolerancia por um fator 3.8. E o [CA-3.16] definia "
    "dispersao_R como 'a diferenca entre as medianas das duas execucoes', que "
    "para duas amostras da mesma distribuicao vale E|m1-m2| = 2*sigma/sqrt(pi) "
    "~= 1.13*sigma — perto de sigma o bastante para o troco passar despercebido."
)

_PAIR_REDUCTION_NOTE = (
    "Com M execucoes ha M*(M-1)/2 pares, entao d(R_i, R_j) deixa de ser um "
    "numero. A regra de reducao e declarada aqui e nao escolhida depois: o "
    "denominador do [CA-3.1] e a MEDIA dos d pareados, e o numerador e a media "
    "dos d(Python, R_i) — mesma especie de quantidade nos dois lados. Mediana, "
    "minimo, maximo e erro-padrao acompanham porque colapsar dez valores em um "
    "apaga a informacao que separa acoes opostas."
)

_UNDEFINED_POLICY = (
    "RSD indefinido e excluido PAR A PAR: o rho de um par e calculado sobre "
    "as features definidas nas DUAS execucoes daquele par, e n_features_used vem "
    "junto de cada par. Excluir pela uniao de todas as execucoes descartaria "
    "feature utilizavel num par por causa de outro; nao excluir propagaria nan "
    "para o rho inteiro. A contagem viaja com o numero porque rho sobre "
    "subconjunto diferente nao e o mesmo rho."
)


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


_REFERENCE_CANDIDATES = (REPO / "data" / "reference", REPO / "serrf" / "data")


def reference_root() -> Path:
    for candidate in _REFERENCE_CANDIDATES:
        if (candidate / "input").is_dir():
            return candidate
    raise SystemExit(
        "dados de referência ausentes: nenhum de "
        f"{[str(c) for c in _REFERENCE_CANDIDATES]} tem `input/`. "
        "Rode scripts/fetch_reference_data.sh."
    )


def qc_sample_names(root: Path, mode: str) -> tuple[str, ...]:
    """Nomes das amostras com `sampleType == "qc"`, lidos do `{modo}_toserrf.csv`.

    Vem do `sampleType`, não de casar `"QC"` no nome — nestes quatro arquivos os
    dois caminhos coincidem e a coincidência é observação sobre eles.
    """
    rows = _rows(root / "output" / f"{mode}_toserrf.csv")
    types = next(r for r in rows if len(r) > 1 and r[1] == "sampleType")
    labels = next(r for r in rows if len(r) > 1 and r[1] == "label")
    return tuple(
        labels[i] for i in range(2, len(types)) if types[i].strip().strip('"').lower() == "qc"
    )


def normalized_path(run_dir: Path, mode: str, method: str) -> Path:
    return (
        run_dir
        / f"{mode}_toserrf - normalization result"
        / "normalized datasets"
        / f"normalized by - {method}.csv"
    )


def read_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Devolve (nomes de feature, cabeçalho de amostra, matriz `(features, amostras)`)."""
    rows = _rows(path)
    if not rows:
        raise SystemExit(f"{path} vazio")
    header = rows[0]
    body = [r for r in rows[1:] if r and r[0]]
    features = [r[0] for r in body]
    values = np.array(
        [
            [np.nan if c.strip() in ("", "NA", "NaN") else float(c) for c in r[1 : len(header)]]
            for r in body
        ],
        dtype=np.float64,
    )
    return features, header[1:], values


def qc_columns(sample_header: list[str], qc_names: tuple[str, ...], where: Path) -> list[int]:
    missing = [n for n in qc_names if n not in sample_header]
    if missing:
        raise SystemExit(f"{where} não traz as colunas de QC {missing}")
    return [sample_header.index(n) for n in qc_names]


def rsd_vector(values: np.ndarray, columns: list[int]) -> np.ndarray:
    """RSD por feature sobre as colunas de QC, in-sample, variante `outlier_removed`.

    `axis=1` porque a matriz do R é `(features, amostras)` — a orientação do
    arquivo, não a canônica interna. A transposição da ingestão não
    acontece aqui: este gerador lê saída do R, não alimenta o núcleo.
    """
    result: np.ndarray = rsd(values[:, columns], axis=1, variant=RSD_VARIANT)
    return result


# ---------------------------------------------------------------------------
# Medição
# ---------------------------------------------------------------------------


def pairwise_divergence(vectors: list[np.ndarray]) -> list[dict[str, Any]]:
    """`d = 1 - rho_Spearman` para cada par, sobre as features definidas nos dois."""
    pairs: list[dict[str, Any]] = []
    for i, j in itertools.combinations(range(len(vectors)), 2):
        a, b = vectors[i], vectors[j]
        both = np.isfinite(a) & np.isfinite(b)
        n_used = int(both.sum())
        if n_used < 3:
            raise SystemExit(
                f"par ({i + 1},{j + 1}) tem {n_used} features com RSD definido nas duas "
                "execuções; Spearman não significa nada aí."
            )
        rho = float(stats.spearmanr(a[both], b[both]).statistic)
        pairs.append(
            {
                "run_i": i + 1,
                "run_j": j + 1,
                "spearman_rho": rho,
                "d": 1.0 - rho,
                "n_features_used": n_used,
                "n_features_excluded_undefined": int(a.size - n_used),
            }
        )
    return pairs


def summarize(values: list[float]) -> dict[str, float]:
    """Resumo de uma amostra. Nunca só a média."""
    array = np.asarray(values, dtype=np.float64)
    n = array.size
    sd = float(np.std(array, ddof=1)) if n > 1 else float("nan")
    return {
        "n": int(n),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "sd_ddof1": sd,
        "stderr_of_mean": sd / float(np.sqrt(n)) if n > 1 else float("nan"),
    }


def zero_positions_identical(matrices: list[np.ndarray]) -> bool:
    first = matrices[0] == 0
    return all(np.array_equal(first, m == 0) for m in matrices[1:])


def shapes_identical(matrices: list[np.ndarray]) -> bool:
    return all(m.shape == matrices[0].shape for m in matrices[1:])


def uncorrected_features(none_values: np.ndarray, method_values: np.ndarray) -> set[int]:
    """Índices das features que o método devolveu **bit-idênticas** à matriz `none`.

    Bit-idêntico e não "próximo": o `batch_loess` devolve a linha
    original sem marcar quando a curva ajustada teve valor negativo. Igualdade
    exata é o discriminante; tolerância aqui colheria feature corrigida cuja
    correção foi pequena, que é outra coisa.
    """
    same = [
        i
        for i in range(none_values.shape[0])
        if np.array_equal(none_values[i], method_values[i], equal_nan=True)
    ]
    return set(same)


def golden_uncorrected(root: Path, mode: str) -> set[int]:
    """As features que o `batch_loess` do laboratório devolveu sem correção.

    A expectativa publicada — 1.561 / 165 / 20 / 10 — é **observação sobre
    estes quatro arquivos**, e é contra ela que se compara.
    """
    base = root / "output" / f"{mode}_toserrf - normalization result" / "normalized datasets"
    _, _, none_values = read_matrix(base / "normalized by - none.csv")
    _, _, loess_values = read_matrix(base / "normalized by - batch_loess.csv")
    return uncorrected_features(none_values, loess_values)


def measure_uncorrected_sets(per_run: list[set[int]], golden: set[int]) -> dict[str, Any]:
    """Estabilidade do conjunto de não corrigidas — e ela **não** é um booleano.

    Medido no `neg_c18`: 32 e 26 features em duas execuções contra 20 no
    golden, com interseção de 20 e apenas 15 em comum com o golden. Um campo
    `identical: false` registraria o desfecho e apagaria a magnitude, que é o que
    separa "flutua na margem" de "flutua no corpo".

    A interseção ter o mesmo TAMANHO do golden e não ser ele é exatamente por que
    o conjunto é comparado por identidade e não por contagem.
    """
    intersection: set[int] = set.intersection(*per_run) if per_run else set()
    union: set[int] = set.union(*per_run) if per_run else set()
    return {
        "n_by_run": [len(s) for s in per_run],
        "identical_across_runs": all(s == per_run[0] for s in per_run[1:]),
        "n_intersection": len(intersection),
        "n_union": len(union),
        "n_always_uncorrected": len(intersection),
        "n_sometimes_uncorrected": len(union - intersection),
        "golden": {
            "n": len(golden),
            "n_shared_with_intersection": len(golden & intersection),
            "n_shared_with_union": len(golden & union),
            "is_subset_of_every_run": all(golden <= s for s in per_run),
            "equals_intersection": golden == intersection,
        },
        "note": (
            "O conjunto NAO e estavel entre execucoes, contra o que a secao 7 linha "
            "1967 e o plano da U3 afirmavam. Mecanismo: a "
            "selecao de span por bootstrap::crossval tambem roda dentro do "
            "parSapply da linha 908, cujo cluster nao recebe clusterSetRNGStream, "
            "entao o sorteio muda a curva ajustada e mudar a curva muda SE ela "
            "fica negativa. Ver DV9."
        ),
    }


def measure_mode(mode: str, run_dirs: list[Path], root: Path) -> dict[str, Any]:
    qc_names = qc_sample_names(root, mode)

    serrf_rsd: list[np.ndarray] = []
    serrf_matrices: list[np.ndarray] = []
    serrf_medians: list[float] = []
    none_matrices: list[np.ndarray] = []
    loess_uncorrected: list[set[int]] = []
    loess_matrices: list[np.ndarray] = []
    feature_names: list[str] = []

    for run_dir in run_dirs:
        none_path = normalized_path(run_dir, mode, "none")
        serrf_path = normalized_path(run_dir, mode, "SERRF")
        loess_path = normalized_path(run_dir, mode, "batch_loess")
        for path in (none_path, serrf_path, loess_path):
            if not path.is_file():
                raise SystemExit(f"ausente: {path}")

        features, header, none_values = read_matrix(none_path)
        if not feature_names:
            feature_names = features
        columns = qc_columns(header, qc_names, none_path)

        _, _, serrf_values = read_matrix(serrf_path)
        _, _, loess_values = read_matrix(loess_path)

        none_matrices.append(none_values)
        serrf_matrices.append(serrf_values)
        loess_matrices.append(loess_values)

        vector = rsd_vector(serrf_values, columns)
        serrf_rsd.append(vector)
        defined = vector[np.isfinite(vector)]
        serrf_medians.append(float(np.median(defined)))
        loess_uncorrected.append(uncorrected_features(none_values, loess_values))

    pairs = pairwise_divergence(serrf_rsd)
    d_values = [p["d"] for p in pairs]

    # `none` é oráculo exato, e são DUAS afirmações: as M execuções idênticas
    # entre si, e idênticas ao golden do laboratório. Só a primeira passaria num
    # runner que executasse consistentemente a coisa errada — é a linha 1 da
    # tabela de invariantes, e ela é a âncora que dá sentido às outras três.
    # Célula a célula e não byte a byte: aspas, terminador e
    # caixa do expoente são artefatos de serialização, excluídos
    # deliberadamente da comparação.
    none_identical = all(
        np.array_equal(none_matrices[0], m, equal_nan=True) for m in none_matrices[1:]
    )
    golden_none_path = (
        root
        / "output"
        / f"{mode}_toserrf - normalization result"
        / "normalized datasets"
        / "normalized by - none.csv"
    )
    _, _, golden_none = read_matrix(golden_none_path)
    none_matches_golden = golden_none.shape == none_matrices[0].shape and np.array_equal(
        golden_none, none_matrices[0], equal_nan=True
    )
    with np.errstate(invalid="ignore"):
        none_max_abs_diff = (
            float(np.nanmax(np.abs(golden_none - none_matrices[0])))
            if golden_none.shape == none_matrices[0].shape
            else float("nan")
        )

    loess_differs_elsewhere = any(
        not np.array_equal(loess_matrices[0], m, equal_nan=True) for m in loess_matrices[1:]
    )
    loess_sets = measure_uncorrected_sets(loess_uncorrected, golden_uncorrected(root, mode))

    return {
        "mode": mode,
        "n_runs": len(run_dirs),
        "n_features": len(feature_names),
        "n_qc": len(qc_names),
        "serrf": {
            "pairs": pairs,
            "d_summary": summarize(d_values),
            # O denominador da divergência, nomeado para que ninguém tenha de
            # reconstruir a regra de redução a partir dos dez pares.
            "d_r1_r2_for_ca_3_1": float(np.mean(d_values)),
            "median_rsd_by_run_pct": [m * 100.0 for m in serrf_medians],
            "median_rsd_summary_pct": summarize([m * 100.0 for m in serrf_medians]),
            # A `sd_R` do estagio 2, em pontos percentuais.
            "sd_r_pct_for_ca_3_2": float(np.std([m * 100.0 for m in serrf_medians], ddof=1)),
            "shapes_identical": shapes_identical(serrf_matrices),
            "zero_positions_identical": zero_positions_identical(serrf_matrices),
        },
        "batch_loess": {
            "uncorrected": loess_sets,
            "values_differ_across_runs_elsewhere": loess_differs_elsewhere,
        },
        "none": {
            "identical_across_runs": none_identical,
            "matches_golden": none_matches_golden,
            "max_abs_diff_vs_golden": none_max_abs_diff,
            "n_cells": int(none_matrices[0].size),
        },
    }


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_digests() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, _, relative = stripped.partition("  ")
        if digest and relative:
            entries[relative] = digest
    return entries


def run_provenance(run_dirs: list[Path]) -> list[dict[str, Any]]:
    """Procedência de cada execução, lida do `response.json` que o runner escreveu."""
    out: list[dict[str, Any]] = []
    for index, run_dir in enumerate(run_dirs, start=1):
        response = run_dir / "response.json"
        if not response.is_file():
            raise SystemExit(f"ausente: {response} — a execução {index} não registrou procedência")
        body = json.loads(response.read_text(encoding="utf-8"))
        provenance = body["provenance"]
        out.append(
            {
                "run": index,
                "r_version": provenance["r_version"],
                "image": provenance["image"],
                "image_digest": provenance["image_digest"],
                "input_sha256": provenance.get("input_sha256"),
                # `cores_detected` e `detectcores_ratio` entram porque são o que
                # PARTICIONA o `parSapply`, e a partição decide quais sorteios
                # caem em qual worker. Duas execuções com contagem de worker
                # diferente não são duas amostras da mesma distribuição, e um
                # `rho` medido entre elas misturaria diferença de configuração
                # com a irreprodutibilidade que o critério quer medir.
                "cores_detected": provenance.get("cores_detected"),
                "detectcores_ratio": provenance.get("detectcores_ratio"),
                "package_versions": provenance.get("package_versions"),
                "cran_snapshot": provenance.get("cran_snapshot"),
                "elapsed_run_seconds": body.get("elapsed_run_seconds"),
            }
        )
    return out


# O que tem de ser igual nas M execuções para que elas sejam amostras da mesma
# distribuição. `input_sha256` é o "sobre o mesmo `_toserrf.csv`";
# os outros quatro são a configuração que decide como o sorteio é particionado.
INVARIANT_ACROSS_RUNS = (
    "input_sha256",
    "image_digest",
    "r_version",
    "cores_detected",
    "detectcores_ratio",
)


def assert_same_configuration(mode: str, runs: list[dict[str, Any]]) -> None:
    """As M execuções compartilham entrada e configuração, ou a medição não vale.

    Sem esta guarda, uma máquina com carga diferente entre execuções mudaria
    `cores_detected`, e o `d(R_i, R_j)` resultante mediria a mudança de partição
    em vez da ausência de `set.seed` — conclusão correta sobre a quantidade
    errada.
    """
    for key in INVARIANT_ACROSS_RUNS:
        values = {json.dumps(run.get(key), sort_keys=True) for run in runs}
        if len(values) != 1:
            raise SystemExit(
                f"{mode}: as {len(runs)} execuções divergem em `{key}`: {sorted(values)}. "
                "Elas têm de ser sobre a mesma entrada e a mesma configuração; "
                "senão rho mede diferença de configuração, não irreprodutibilidade."
            )


def build(runs_root: Path, modes: tuple[str, ...]) -> dict[str, Any]:
    root = reference_root()
    known = manifest_digests()

    sources: dict[str, str] = {}
    per_mode: list[dict[str, Any]] = []
    provenance_runs: dict[str, list[dict[str, Any]]] = {}

    for mode in modes:
        mode_dir = runs_root / mode
        run_dirs = sorted(
            (d for d in mode_dir.iterdir() if d.is_dir() and d.name.startswith("run")),
            key=lambda d: int(d.name.removeprefix("run")),
        )
        if len(run_dirs) < M_R_RUNS:
            raise SystemExit(
                f"{mode}: {len(run_dirs)} execuções, e o pré-registro fixa M = {M_R_RUNS}. "
                "Abaixo disso a estimativa do denominador fica frágil demais."
            )
        relative = f"output/{mode}_toserrf.csv"
        sources[relative] = digest_of(root / relative)
        runs = run_provenance(run_dirs)
        assert_same_configuration(mode, runs)
        per_mode.append(measure_mode(mode, run_dirs, root))
        provenance_runs[mode] = runs

    drifted = {r: d for r, d in sources.items() if r in known and known[r] != d}
    if drifted:
        raise SystemExit(
            f"os arquivos de entrada divergem do manifesto: {sorted(drifted)}. "
            "Regenere pelo script; não vou congelar número sobre dado que mudou."
        )

    generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S%z")
    return {
        "preregistration": {
            "k_ca_3_1": K_CA_3_1,
            "k_ca_3_2": K_CA_3_2,
            "n_python_seeds": N_PYTHON_SEEDS,
            "m_r_runs": M_R_RUNS,
            "tolerance_formula": _TOLERANCE_FORMULA,
            "k_justification": _K_JUSTIFICATION,
            "k_caveat_not_a_lever": _K_CAVEAT,
            "no_negotiation_clause": _NO_NEGOTIATION,
            "quantity_note": _QUANTITY_NOTE,
            "pair_reduction_note": _PAIR_REDUCTION_NOTE,
            "undefined_rsd_policy": _UNDEFINED_POLICY,
            "n_justification": (
                f"N = {N_PYTHON_SEEDS} sementes do Python da ~16% de incerteza no desvio. "
                f"M = {M_R_RUNS} execucoes do R dao {M_R_RUNS * (M_R_RUNS - 1) // 2} pares; "
                "abaixo de 5 a estimativa do denominador fica fragil demais."
            ),
        },
        "provenance": {
            "command": "scripts/gen_r_divergence.sh",
            "generated_at": generated_at,
            "rsd_estimator": RSD_ESTIMATOR,
            "rsd_variant": RSD_VARIANT,
            "rsd_implementation": (
                "refino.metrics.rsd — a MESMA regua dos dois lados do [CA-3.1]. Duas "
                "transcricoes fariam numerador e denominador medirem quantidades "
                "diferentes. Nao e auto-referencial: o [CA-2.1] ancora esta "
                "regua a 1e-6 contra a coluna none do QC - RSD.csv do proprio R."
            ),
            "units": (
                "median_rsd_*_pct e sd_r_pct_* estao em pontos percentuais. d e "
                "1 - rho_Spearman, adimensional em [0, 2]."
            ),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "timing_caveat": (
                "elapsed_run_seconds NAO e medida confiavel nesta geracao. A maquina "
                "tinha geradores de carga orfaos de sessoes anteriores (AR34, load "
                "average 96 em 12 nucleos), removidos DURANTE a 5a execucao do "
                "pos_hilic. Os VALORES — rho, medianas, conjuntos de features — sao "
                "validos, porque carga nao os altera; as DURACOES sao de duas "
                "condicoes diferentes e nao devem ser agregadas. A 5a execucao do "
                "pos_hilic atravessou a transicao: SERRF 206 s contra 340-363 s das "
                "anteriores, batch_loess 232 s inalterado na mesma execucao. Pergunta "
                "em aberto em AR34: ou 232 s e a base limpa e as anteriores estavam "
                "infladas, ou o batch_loess nao e limitado por CPU."
            ),
            "runs": provenance_runs,
            "input_sources": dict(sorted(sources.items())),
        },
        "regeneration_contract": {
            "exact": ["preregistration", "provenance.rsd_*", "provenance.input_sources"],
            "not_reproducible_by_design": ["modes[].serrf.pairs", "modes[].serrf.*_summary*"],
            "note": (
                "O bloco SERRF do tools.R nao chama set.seed (achado 4.3), entao outra "
                "geracao mede outras execucoes: o bloco medido NAO pode bater, e a "
                "irreprodutibilidade dele E a medicao. A secao 7 pedia igualdade em tudo "
                "menos generated_at, o que e insatisfazivel por construcao. O que "
                "protege os numeros medidos e o pre-registro com sha256 no tronco, nao a "
                "reprodutibilidade."
            ),
            "appended_not_measured": [
                "observed_regenerations — log append-only de geracoes independentes, anotado "
                "a mao. NAO e medicao desta geracao e NAO entra na igualdade exata da "
                "regeneracao."
            ],
        },
        "modes": per_mode,
    }


def carry_over_regeneration_log(payload: dict[str, Any], existing: Path) -> dict[str, Any]:
    """Preserva `observed_regenerations` de um fixture anterior.

    O bloco é um **registro append-only de gerações independentes**, e não uma
    medição desta geração: cada entrada é o que uma regeneração de fato mediu,
    anotada à mão a partir da execução. Ele existe porque `sd_r` com `M = 5` tem
    4 graus de liberdade e varia por fator 2 entre amostras da mesma distribuição
    — medido: 0,1072 pp e 0,2221 pp no `neg_hilic`.

    **Relatar as duas estimativas lado a lado é transparência, não ajuste.** O
    valor pré-registrado não muda e exceder o orçamento continua não autorizando
    mexer em `M`; o que muda é o leitor ver a incerteza do próprio orçamento em
    vez de tomar 0,0196 como exato. Perder o registro a cada regeneração
    devolveria a aparência de precisão que ele existe para remover.
    """
    if not existing.is_file():
        return payload
    try:
        previous = json.loads(existing.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover
        # Fixture ilegível não deve derrubar uma geração de duas horas; o log é
        # acessório e a ausência dele é visível no arquivo novo.
        return payload
    log = previous.get("observed_regenerations")
    if log:
        payload["observed_regenerations"] = log
    return payload


def serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path, help="raiz com {modo}/run{i}/")
    parser.add_argument("--out", type=Path, default=FIXTURE)
    parser.add_argument(
        "--modes", default=",".join(MODES), help="lista de modos, para uso em smoke test"
    )
    args = parser.parse_args()

    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"modo desconhecido: {unknown}. Os quatro são {list(MODES)}.")

    payload = build(args.runs_dir, modes)
    payload = carry_over_regeneration_log(payload, args.out)
    args.out.write_text(serialize(payload), encoding="utf-8")
    print(f"escrito: {args.out}  ({len(payload['modes'])} modos)")
    for entry in payload["modes"]:
        serrf = entry["serrf"]
        print(
            f"  {entry['mode']:>10}  d(R,R) medio {serrf['d_r1_r2_for_ca_3_1']:.6f}"
            f"  sd_R {serrf['sd_r_pct_for_ca_3_2']:.4f} pp"
            f"  mediana {serrf['median_rsd_summary_pct']['mean']:.2f} pp"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

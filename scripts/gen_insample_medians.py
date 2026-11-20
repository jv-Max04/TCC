#!/usr/bin/env python3
"""Gera `data/fixtures/insample_rsd_medians.json` — a expectativa in-sample.

**Este script não importa `refino`, e a proibição é o ponto.** Ele é a segunda
transcrição da mesma fórmula: lê as doze matrizes golden com `csv` e `numpy` e
reimplementa `fivenum`, a máscara de outlier e `sd/mean`. Um gerador que
importasse `refino` produziria exatamente o defeito que o teste deveria acusar —
regenerar apagaria a regressão que o arquivo existe para detectar. Há teste
estático sobre os imports deste arquivo.

O que ele mede, e por quê cada coisa:

- **Mediana de RSD**, nas duas variantes. É **instrumentação**: espelho de
  calibração contra a tabela do terceiro eixo. Em três dos vinte e
  quatro números a mediana é da ordem de 1e-13 pontos percentuais, e nenhuma
  tolerância de 0,5 pp pode reprovar ali — inclusive contra uma implementação que
  devolvesse zero constante.
- **Fração de features com RSD numericamente indistinguível de zero** — o
  **portão discriminante**, que é a quantidade que caracteriza aquelas células:
  mede quantas features o LOESS interpolou.

**Por que o portão é `< 1e-6` e não `== 0.0`**. Medido sobre estes quatro
arquivos, a faixa `(0, 1e-18]` é **vazia** e a fração abaixo de 1e-12 iguala a
fração abaixo de 1e-6 na primeira decimal em todas as oito combinações de modo e
variante: existe um vão de seis ordens de grandeza entre o aglomerado de epsilon
de máquina e a população real. De 14 a 27 pontos percentuais das features vivem
nesse aglomerado, onde o `sd` de sete valores quase idênticos sofre cancelamento
catastrófico — se uma delas cai em `0.0` ou em `3e-16` é artefato da ordem de
acumulação, não do dado. Um portão em zero exato apostaria vinte pontos
percentuais da população numa mudança de versão de `numpy` ou de BLAS.
`frac_below_gap_probe_pct` existe para que o teste prove que o limiar está no vão
e não numa encosta.

**Toda quantidade deste arquivo tem sufixo de unidade**, e isso é reparo de
defeito real: uma fonte trazia `1,9e-15` ao lado de uma tabela em pontos
percentuais, onde a mesma medição vale `1,9e-13`. O limiar de proximidade de zero
é declarado sobre o RSD **como fração** (0,25 = 25%), que é a escala em que
`refino.metrics.rsd` devolve.

Uso:

    .venv/bin/python scripts/gen_insample_medians.py            # escreve
    .venv/bin/python scripts/gen_insample_medians.py --check    # não escreve
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "data" / "fixtures" / "insample_rsd_medians.json"
MANIFEST = REPO / "scripts" / "reference_data.sha256"

MODES = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")
METHODS = ("none", "batch_loess", "SERRF")
VARIANTS = ("plain", "outlier_removed")

# Sobre o RSD como **fração** (0,25 = 25%), não como pontos percentuais.
NEAR_ZERO_RSD = 1e-6
# Sonda do vão: se a fração aqui iguala a de `NEAR_ZERO_RSD`, o limiar não é
# carga estrutural do portão. Se divergir, o limiar está numa encosta e o número
# congelado passa a depender dele — que é o que o portão quer impedir.
GAP_PROBE_RSD = 1e-12

# Mínimo de valores para que `sd` com denominador `n - 1` exista.
MIN_VALUES_FOR_SD = 2
OUTLIER_COEF = 1.5

# As duas raízes candidatas dos dados de referência, na mesma ordem do `tests/conftest.py`. A
# duplicação é deliberada: este script não pode importar de `tests/`, e replicar
# duas linhas é melhor que criar dependência entre um gerador e a suíte.
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


# ---------------------------------------------------------------------------
# Transcrição independente: fivenum, máscara de outlier, RSD
# ---------------------------------------------------------------------------


def fivenum(values: np.ndarray) -> np.ndarray:
    """As cinco dobradiças de Tukey, como o `fivenum()` do R.

    Transcrição própria, deliberadamente não importada de `refino.metrics.rsd`:
    duas transcrições da mesma fórmula é o que dá conteúdo à comparação.
    """
    present = np.sort(values[~np.isnan(values)])
    n = present.size
    if n == 0:
        return np.full(5, np.nan)
    n4 = math.floor((n + 3) / 2) / 2
    d = np.array([1.0, n4, (n + 1) / 2, n + 1 - n4, float(n)])
    lo = np.floor(d).astype(np.intp) - 1
    hi = np.ceil(d).astype(np.intp) - 1
    return np.asarray(0.5 * (present[lo] + present[hi]), dtype=np.float64)


def non_outliers(values: np.ndarray, coef: float = OUTLIER_COEF) -> np.ndarray:
    """O que sobra depois do `remove_outlier()` do `tools.R`: comparação estrita."""
    hinges = fivenum(values)
    with np.errstate(invalid="ignore"):
        spread = hinges[3] - hinges[1]
    if not np.isfinite(spread):
        return values[np.isfinite(values)]
    lower = hinges[1] - coef * spread
    upper = hinges[3] + coef * spread
    return values[~((values < lower) | (values > upper))]


def rsd_of(values: np.ndarray, variant: str) -> float:
    """`sd/mean` com `ddof=1`, ou `nan` nos três caminhos de indefinido.

    Descarta **apenas ausentes**, e `±Inf` sobrevive — como em `refino.metrics.rsd`
    e como no R. Isso não é hipotético nestes arquivos: as doze matrizes golden do
    `batch_loess` têm células infinitas, e só elas. Sob `plain` o `sd` de um
    vetor com infinito é `nan` por `inf - inf`, e a feature fica indefinida; sob
    `outlier_removed` o infinito é removido antes, pelo ramo `!is.finite` do
    `boxplot.stats`, e a feature pode voltar a ter RSD definido.
    """
    present = values[~np.isnan(values)]
    if present.size < MIN_VALUES_FOR_SD:
        return float("nan")
    kept = present if variant == "plain" else non_outliers(present)
    if kept.size < MIN_VALUES_FOR_SD:
        return float("nan")
    # `inf - inf` é `nan` legitimamente, e é o desfecho correto: quem tem medição
    # infinita não tem RSD. O aviso do numpy descreveria um caso já decidido.
    with np.errstate(invalid="ignore"):
        mean = float(np.mean(kept))
        if mean == 0.0:
            return float("nan")
        return float(np.std(kept, ddof=1) / mean)


# ---------------------------------------------------------------------------
# Leitura dos arquivos reais
# ---------------------------------------------------------------------------


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.reader(handle))


def qc_sample_names(root: Path, mode: str) -> tuple[str, ...]:
    """Nomes das amostras com `sampleType == "qc"`, lidos do `{modo}_toserrf.csv`.

    Vem do `sampleType` e **não** de casar `"QC"` no nome da amostra. Nos quatro
    arquivos deste estudo os dois caminhos coincidem, e essa coincidência é
    observação sobre eles: um estudo cujo nome de amostra biológica
    contivesse "QC" faria o atalho colher amostra que não é QC, em silêncio.
    """
    rows = _rows(root / "output" / f"{mode}_toserrf.csv")
    types = next(r for r in rows if len(r) > 1 and r[1] == "sampleType")
    labels = next(r for r in rows if len(r) > 1 and r[1] == "label")
    return tuple(
        labels[i] for i in range(2, len(types)) if types[i].strip().strip('"').lower() == "qc"
    )


def qc_matrix(root: Path, mode: str, method: str, qc_names: tuple[str, ...]) -> np.ndarray:
    """Matriz `(features, qcs)` de uma golden normalizada, colunas na ordem de `qc_names`."""
    path = (
        root
        / "output"
        / f"{mode}_toserrf - normalization result"
        / "normalized datasets"
        / f"normalized by - {method}.csv"
    )
    rows = _rows(path)
    header = rows[0]
    missing = [name for name in qc_names if name not in header]
    if missing:
        raise SystemExit(f"{path.name} de {mode} não traz as colunas de QC {missing}")
    columns = [header.index(name) for name in qc_names]
    data = [
        [np.nan if row[c].strip() in ("", "NA", "NaN") else float(row[c]) for c in columns]
        for row in rows[1:]
        if row and row[0]
    ]
    return np.asarray(data, dtype=np.float64)


# ---------------------------------------------------------------------------
# Medição
# ---------------------------------------------------------------------------


def measure(values: np.ndarray, variant: str) -> dict[str, Any]:
    rsd = np.array([rsd_of(values[i, :], variant) for i in range(values.shape[0])])
    defined = np.isfinite(rsd)
    n_defined = int(defined.sum())
    if n_defined == 0:
        raise SystemExit(f"nenhuma feature com RSD definido na variante {variant}")
    kept = rsd[defined]
    n_near_zero = int((kept < NEAR_ZERO_RSD).sum())
    n_gap_probe = int((kept < GAP_PROBE_RSD).sum())
    return {
        "n_features": int(rsd.size),
        "n_defined": n_defined,
        "n_undefined": int(rsd.size - n_defined),
        "median_rsd_pct": float(np.median(kept)) * 100.0,
        "n_near_zero": n_near_zero,
        # Redundante com `n_near_zero / n_defined` de propósito: o portão é
        # comparado com o número publicado em pontos percentuais, a regressão é
        # comparada com a contagem, e um teste afirma que as duas concordam. A
        # redundância sem esse teste seria deriva esperando acontecer.
        "frac_near_zero_pct": 100.0 * n_near_zero / n_defined,
        "n_below_gap_probe": n_gap_probe,
        "frac_below_gap_probe_pct": 100.0 * n_gap_probe / n_defined,
    }


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


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    root = reference_root()
    known = manifest_digests()
    sources: dict[str, str] = {}
    cells: list[dict[str, Any]] = []

    for mode in MODES:
        qc_names = qc_sample_names(root, mode)
        toserrf = f"output/{mode}_toserrf.csv"
        sources[toserrf] = digest_of(root / toserrf)
        for method in METHODS:
            relative = (
                f"output/{mode}_toserrf - normalization result/"
                f"normalized datasets/normalized by - {method}.csv"
            )
            sources[relative] = digest_of(root / relative)
            values = qc_matrix(root, mode, method, qc_names)
            cells.append(
                {
                    "mode": mode,
                    "method": method,
                    "n_qc": len(qc_names),
                    "qc_sample_names": list(qc_names),
                    **{variant: measure(values, variant) for variant in VARIANTS},
                }
            )

    drifted = {
        relative: digest
        for relative, digest in sources.items()
        if relative in known and known[relative] != digest
    }
    if drifted:
        raise SystemExit(
            "os arquivos de origem divergem do manifesto: "
            f"{sorted(drifted)}. Não vou congelar número sobre dado que mudou."
        )

    generated_at = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S%z")
    return {
        "provenance": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "command": "scripts/gen_insample_medians.py",
            "generated_at": generated_at,
            "seed": None,
            "note": (
                "sd/mean e fivenum sao deterministicos: nenhuma semente e necessaria. "
                "Este gerador NAO importa refino, por KTD-7: e a segunda transcricao da "
                "formula, e importar apagaria a regressao que o arquivo existe para detectar."
            ),
            "units": (
                "Toda quantidade tem sufixo de unidade. median_rsd_pct e frac_*_pct estao em "
                "pontos percentuais. Os dois limiares estao sobre o RSD como FRACAO "
                "(0,25 = 25%), que e a escala em que refino.metrics.rsd devolve."
            ),
            "near_zero_threshold_rsd_fraction": NEAR_ZERO_RSD,
            "gap_probe_threshold_rsd_fraction": GAP_PROBE_RSD,
            "outlier_coef": OUTLIER_COEF,
            "ddof": 1,
            "qc_selection_rule": (
                "Colunas com sampleType == 'qc' no {modo}_toserrf.csv, nao casamento de "
                "'QC' no nome da amostra. Nestes quatro arquivos os dois caminhos "
                "coincidem, e a coincidencia e observacao sobre eles."
            ),
            "sources": dict(sorted(sources.items())),
        },
        "cells": cells,
    }


def serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


# **A granularidade em que `python_version` é comparado.**
#
# `sys.version.split()[0]` grava o interpretador inteiro, `3.12.14`, e é isso que fica no
# arquivo: procedência quer o número exato de quem produziu. O que a COMPARAÇÃO olha é só
# `major.minor`, e o motivo é que essa é a granularidade que o projeto **declara** pinar, nos
# lugares em que o projeto o declara:
#
#   pyproject.toml       requires-python = ">=3.12,<3.13"
#   scripts/env_probe.py major, minor = sys.version_info[:2]
#   [tool.mypy]          python_version = "3.12"
#
# Comparar o patch tornava este arquivo **mais estrito que os pinos declarados**, e
# reprovava por uma quantidade que ele não mede: entre 3.12.13 e 3.12.14
# as doze células e todas as quantidades são idênticas, e a única linha divergente era esta.
#
# **Isto não afrouxa o que o teste existe para afirmar.** O referente declarado é *"o fixture
# não foi editado à mão"*, e todo número continua comparado literalmente — a normalização
# alcança UMA linha, e só a parte dela que o projeto não pina. Rodar em 3.11 continua a
# reprovar, porque o `major.minor` muda.
#
# São QUATRO os geradores com esta mesma normalização, e a duplicação é deliberada: eles são
# autónomos por desenho, e este aqui é uma segunda transcrição da fórmula por desenho.
# `tests/test_procedencia_comparavel.py` exercita as quatro e é o que impede que divirjam.
_PATCH_DO_PYTHON = re.compile(r'("python_version":\s*"\d+\.\d+)\.\d+"')


def comparavel(raw: str) -> list[str]:
    """As linhas que o `--check` compara, com as duas normalizações declaradas.

    Função de módulo, e não corpo de `main`, para que o teste negativo invoque **esta** e não
    uma cópia: negativo que reimplementa a regra fica verde enquanto a regra apodrece.
    É a mesma forma de `comparable_arrays` em `build_serrf_intermediates.py`.

    `generated_at` **sai inteiro** porque muda a cada execução por definição.
    `python_version` **fica, truncado em `major.minor`** — ver o comentário acima.
    """
    return [
        _PATCH_DO_PYTHON.sub(r'\1"', linha)
        for linha in raw.splitlines()
        if '"generated_at"' not in linha
    ]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "compara com o fixture versionado sem escrever; `generated_at` sai da comparação "
            "e `python_version` é comparado em major.minor"
        ),
    )
    args = parser.parse_args()
    payload = build()
    text = serialize(payload)

    if not args.check:
        FIXTURE.write_text(text, encoding="utf-8")
        print(f"escrito: {FIXTURE.relative_to(REPO)}  ({len(payload['cells'])} células)")
        return 0

    if not FIXTURE.exists():
        print(f"ausente: {FIXTURE.relative_to(REPO)}", file=sys.stderr)
        return 1

    if comparavel(FIXTURE.read_text(encoding="utf-8")) == comparavel(text):
        print("fixture idêntico à regeneração (ignorando generated_at e o patch do CPython)")
        return 0
    print("fixture DIVERGE da regeneração; rode sem --check para atualizar", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

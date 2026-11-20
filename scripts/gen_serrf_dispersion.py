#!/usr/bin/env python3
"""Gera `data/fixtures/serrf_dispersion_python.json` — a dispersão do SERRF entre execuções.

**Uma passada produz os dois critérios**, porque os dois precisam do mesmo objeto
caro: a matriz corrigida pelo nosso SERRF, por semente, sobre a matriz **inteira**.

- A divergência precisa de `d(Python, R) = 1 - rho_Spearman` entre o vetor de RSD por
  feature do nosso lado e o do R. O denominador — `d(R1, R2)`, a média dos dez
  pares — já está pré-registrado em `r_serrf_divergence.json` e **não é
  redeclarado aqui**.
- A dispersão precisa de `sd_python`, o desvio padrão amostral das 20 medianas.

**A quantidade é nomeada nos dois lados, e o nome sozinho não basta:**
vetor de RSD **por feature**, estimador **in-sample**, variante **`outlier_removed`**,
calculado com `refino.metrics.rsd` sobre a matriz corrigida — a nossa de um lado,
a golden `normalized by - SERRF.csv` do outro. **Não** é a coluna `SERRF` do
`QC - RSD.csv`, que vem do `r_cv` e que não é a quantidade comparada aqui.

**A mesma régua dos dois lados, e isso não é auto-referencial.** `refino.metrics.rsd` está
ancorado a 1e-6 contra a coluna `none` do próprio R. Duas transcrições
fariam numerador e denominador medirem quantidades diferentes.

**Política de indefinido, declarada.** O `rho` de cada par é sobre as features
com RSD definido nos **dois** vetores, e `n_features_used` viaja junto: `rho` sobre
subconjunto diferente não é o mesmo `rho`.

**Custo.** 67,3 / 72,4 / 86,0 / 216,5 s por ajuste completo, porque aqui a
correlação do passo 3 escala com o pool inteiro de features. Projeção: **147,4 min**
para os 80 ajustes.

Uso:

    .venv/bin/python scripts/gen_serrf_dispersion.py
"""

import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from refino.correction.serrf import SerrfCorrector
from refino.io.readers import load_matrix
from refino.metrics.rsd import rsd

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "data" / "fixtures" / "serrf_dispersion_python.json"
PREREG = REPO / "data" / "fixtures" / "r_serrf_divergence.json"

MODES = ("neg_hilic", "neg_c18", "pos_c18", "pos_hilic")
VARIANT = "outlier_removed"

# **As 20 sementes são `1..20`, sem o espaçamento de 50.000 do `r_cv`.** Lá ele existia
# porque o `LoessCorrector` deriva fluxo por `seed + index` e faixas próximas colidem. Aqui só o
# `SerrfCorrector` roda, e os cinco sítios dele derivam de `SeedSequence` sobre
# tupla — `default_rng([seed, index, lote, passo])` —, que não colide
# aditivamente. Copiar o espaçamento seria carregar uma precaução cujo mecanismo
# não existe neste caminho, e precaução sem mecanismo é ruído que alguém depois
# tenta explicar.
SEEDS = tuple(range(1, 21))

_REFERENCE_CANDIDATES = (REPO / "data" / "reference", REPO / "serrf" / "data")


def reference_root() -> Path:
    for candidate in _REFERENCE_CANDIDATES:
        if (candidate / "input").is_dir():
            return candidate
    raise SystemExit("dados de referência ausentes; rode scripts/fetch_reference_data.sh")


def golden_serrf_rsd(root: Path, mode: str) -> tuple[tuple[str, ...], np.ndarray]:
    """Vetor de RSD por feature da matriz golden do R, mesmo estimador e variante."""
    import csv

    caminho = (
        root
        / "output"
        / f"{mode}_toserrf - normalization result"
        / "normalized datasets"
        / "normalized by - SERRF.csv"
    )
    with caminho.open(newline="", encoding="utf-8") as handle:
        linhas = list(csv.reader(handle))
    cabecalho = linhas[0]
    qc = [i for i, nome in enumerate(cabecalho) if i > 0 and "QC" in nome.upper()]
    features = tuple(linha[0] for linha in linhas[1:])
    valores = np.array(
        [
            [float(linha[i]) if linha[i] not in ("", "NA") else np.nan for i in qc]
            for linha in linhas[1:]
        ],
        dtype=np.float64,
    )
    return features, rsd(valores.T, axis=0, variant=VARIANT)


def main() -> int:
    root = reference_root()
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    denominadores = {
        m["mode"]: {
            "d_r1_r2_for_ca_3_1": m["serrf"]["d_r1_r2_for_ca_3_1"],
            "sd_r_pct_for_ca_3_2": m["serrf"]["sd_r_pct_for_ca_3_2"],
            "median_rsd_R_mean_pct": m["serrf"]["median_rsd_summary_pct"]["mean"],
        }
        for m in prereg["modes"]
    }

    por_modo: dict[str, Any] = {}
    for mode in MODES:
        matrix = load_matrix(
            root / "input" / f"measurements_{mode}.csv",
            root / "input" / f"Sample_List_{mode}.csv",
            mode,
        )
        nomes_golden, rsd_golden = golden_serrf_rsd(root, mode)
        if tuple(matrix.feature_names) != nomes_golden:
            raise SystemExit(f"{mode}: ordem de features diverge da matriz golden")

        por_semente: list[dict[str, Any]] = []
        inicio = time.perf_counter()
        for seed in SEEDS:
            resultado = SerrfCorrector(n_jobs=-1).fit_transform(matrix, seed=seed)
            qcs = np.asarray(resultado.matrix.values)[resultado.matrix.qc_mask]
            nosso = rsd(qcs, axis=0, variant=VARIANT)
            par = np.isfinite(nosso) & np.isfinite(rsd_golden)
            rho = float(spearmanr(nosso[par], rsd_golden[par]).statistic)
            por_semente.append(
                {
                    "corrector_seed": seed,
                    "median_rsd_pct": float(np.median(nosso[np.isfinite(nosso)]) * 100.0),
                    "n_defined": int(np.isfinite(nosso).sum()),
                    "spearman_vs_golden": rho,
                    "d_python_vs_r": 1.0 - rho,
                    "n_features_used": int(par.sum()),
                }
            )
            print(f"  {mode} seed={seed:>2}  d={1.0 - rho:.6f}", flush=True)

        medianas = np.array([linha["median_rsd_pct"] for linha in por_semente])
        distancias = np.array([linha["d_python_vs_r"] for linha in por_semente])
        por_modo[mode] = {
            "n_features": matrix.n_features,
            "por_semente": por_semente,
            "median_rsd_summary_pct": {
                "n": int(medianas.size),
                "mean": float(medianas.mean()),
                "median": float(np.median(medianas)),
                "min": float(medianas.min()),
                "max": float(medianas.max()),
                "sd_ddof1": float(medianas.std(ddof=1)),
            },
            "sd_python_pct_for_ca_3_2": float(medianas.std(ddof=1)),
            "d_summary": {
                "n": int(distancias.size),
                "mean": float(distancias.mean()),
                "median": float(np.median(distancias)),
                "min": float(distancias.min()),
                "max": float(distancias.max()),
                "sd_ddof1": float(distancias.std(ddof=1)),
            },
            "d_python_vs_r_for_ca_3_1": float(distancias.mean()),
            "denominadores_preregistrados": denominadores[mode],
            "segundos_de_parede": time.perf_counter() - inicio,
        }

    payload = {
        "unidade": "U12",
        "criterios": ["[CA-3.1]", "[CA-3.2]"],
        "quantidade": (
            "vetor de RSD POR FEATURE, estimador in-sample, variante outlier_removed, "
            "calculado com refino.metrics.rsd sobre a matriz corrigida — a nossa de um lado e a "
            "golden 'normalized by - SERRF.csv' do outro. NAO e a coluna SERRF do QC - RSD.csv"
        ),
        "corrector_seeds": list(SEEDS),
        "corrector_seeds_nota": (
            "1..20, e NAO as do [CA-3.12]. La o espacamento de 50.000 existia pelo seed+index do "
            "LoessCorrector; aqui so o SerrfCorrector roda e os cinco sitios dele "
            "derivam de SeedSequence sobre tupla, que nao colide aditivamente"
        ),
        "rsd_variant": VARIANT,
        "undefined_policy": (
            "rho por par sobre as features definidas nos DOIS vetores; n_features_used viaja "
            "junto, "
            "porque rho sobre subconjunto diferente nao e o mesmo rho"
        ),
        "provenance": {
            "command": "scripts/gen_serrf_dispersion.py",
            "generated_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S+0000"),
            "python_version": (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
            "numpy_version": np.__version__,
            "denominadores_lidos_de": (
                "data/fixtures/r_serrf_divergence.json (pre-registrado na U3)"
            ),
        },
        "por_modo": por_modo,
    }
    FIXTURE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nescrito: {FIXTURE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

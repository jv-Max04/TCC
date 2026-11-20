#!/usr/bin/env python3
"""Monta o fixture dos intermediários do SERRF.

Lê o despejo que `scripts/gen_serrf_intermediates.sh` produziu e escreve **dois**
artefatos que viajam juntos:

- `data/fixtures/serrf_intermediates_r.json` — o legível: procedência,
  subconjunto declarado, parâmetros do passo 1 e `sel_var`, mais o `sha256` do
  sidecar;
- `data/fixtures/serrf_intermediates_r.npz` — o dado: as matrizes, em `float64`.

**Por que `.npz` e não CSV ou parquet.** A unidade compara por **igualdade
exata**, e round-trip textual não preserva `float64` sem depender de `repr`.
`.npz` guarda os bits. O JSON continua sendo o artefato que se lê e revisa; o
`.npz` é o que se compara.

**Por que não cortar o lado grande.** A matriz pós-passo-1 é 2.039 x 68 e o custo
está nas 61 colunas de amostra, não nas 7 de QC. Mas é justamente o lado caro que
carrega o comportamento não trivial: o escalonamento por **linha** nos QCs é
aritmeticamente vácuo sob Spearman, enquanto o por **coluna** nas amostras não é.
Cortar por tamanho cortaria o que a unidade existe para verificar.

**O digest é obrigatório e tem teste.** Sidecar sem digest é fixture que muda sem
aparecer no diff, e o que se audita é o diff. O JSON carrega o
`sha256` do `.npz`, e `tests/golden/test_serrf_transcription.py` reprova se o par
divergir.

**Nenhum nome de amostra entra em nenhum dos dois**. Índice de injeção e
nome de feature — que é `RT_mz`, não paciente — e nada mais.

Uso:

    .venv/bin/python scripts/build_serrf_intermediates.py --dump-dir DIR --mode neg_hilic
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIXTURE_JSON = REPO / "data" / "fixtures" / "serrf_intermediates_r.json"
FIXTURE_NPZ = REPO / "data" / "fixtures" / "serrf_intermediates_r.npz"

# Padrão de nome de amostra do estudo — `..._ID99_Asymptomatic`. A varredura de
# privacidade procura por ele em `data/fixtures/`; aqui a defesa é anterior,
# e consiste em nunca escrever cabeçalho de coluna de amostra.
SAMPLE_NAME_PATTERN = re.compile(r"_ID\d+_")


def read_matrix_tsv(path: Path) -> tuple[list[str], np.ndarray]:
    """`write.table(..., col.names = NA)` do R: primeira coluna é o nome da linha."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    body = [r for r in rows[1:] if r and r[0]]
    names = [r[0] for r in body]
    values = np.array(
        [[np.nan if c.strip() in ("", "NA", "NaN") else float(c) for c in r[1:]] for r in body],
        dtype=np.float64,
    )
    return names, values


def read_vector_tsv(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").strip().split("\t")


def read_step1_params(path: Path) -> list[dict[str, Any]]:
    """`j, batch, fase, n_zeros, n_na, n, min_non_na` — uma linha por chamada."""
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        j, batch, phase, n_zeros, n_na, n, minval = line.split("\t")
        out.append(
            {
                "j": int(j),
                "batch": batch,
                "phase": phase,
                "n_zeros": int(n_zeros),
                "n_na": int(n_na),
                "n": int(n),
                "min_non_na": float("nan") if minval.strip() == "NA" else float(minval),
            }
        )
    return out


def read_sel_var(dump: Path) -> dict[tuple[int, str], list[int]]:
    """Um arquivo por worker; a união tem de cobrir cada feature exatamente uma vez."""
    found: dict[tuple[int, str], list[int]] = {}
    files = sorted(dump.glob("sel_var_pid*.tsv"))
    if not files:
        raise SystemExit("nenhum arquivo sel_var_pid*.tsv — o despejo do passo 3 não rodou")
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            j, batch, joined = line.split("\t")
            key = (int(j), batch)
            if key in found:
                raise SystemExit(
                    f"feature {j} no lote {batch} aparece duas vezes no despejo. "
                    "Ou o filtro de escopo falhou, ou dois workers escreveram no "
                    "mesmo arquivo — os dois apagam a correspondência 1 para 1."
                )
            found[key] = [int(v) for v in joined.split(",") if v]
    return found


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(dump: Path, mode: str, response: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    suffix = _tag(dump)
    features, before = read_matrix_tsv(dump / f"step1_before_{suffix}.tsv")
    _, after = read_matrix_tsv(dump / f"step1_after_{suffix}.tsv")
    batch = read_vector_tsv(dump / f"step1_batch_{suffix}.tsv")
    sample_type = read_vector_tsv(dump / f"step1_sampletype_{suffix}.tsv")

    if before.shape != after.shape:
        raise SystemExit(f"before {before.shape} e after {after.shape} discordam")
    if len(batch) != before.shape[1]:
        raise SystemExit(f"batch tem {len(batch)} entradas para {before.shape[1]} colunas")

    for name in features:
        if SAMPLE_NAME_PATTERN.search(name):
            raise SystemExit(
                f"nome de feature parece nome de amostra: {name!r}. Pseudônimo de "
                "paciente mais diagnóstico não pode sair de data/reference/."
            )

    params = read_step1_params(dump / "step1_params.tsv")
    sel_var = read_sel_var(dump)

    corr_files = sorted(dump.glob(f"corrs_train_*_{suffix}.tsv"))
    if not corr_files:
        raise SystemExit("nenhum despejo de correlação; o subconjunto ficou vazio?")
    arrays: dict[str, np.ndarray] = {
        "step1_before": before,
        "step1_after": after,
    }
    batches: list[str] = []
    for path in corr_files:
        label = path.stem.removeprefix("corrs_train_").removesuffix(f"_{suffix}")
        batches.append(label)
        _, train = read_matrix_tsv(path)
        _, target = read_matrix_tsv(dump / f"corrs_target_{label}_{suffix}.tsv")
        arrays[f"corrs_train_{label}"] = train
        arrays[f"corrs_target_{label}"] = target

    zeros_phase = [p for p in params if p["phase"] == "zeros"]
    with_zero = [p for p in zeros_phase if p["n_zeros"] > 0]
    min_is_zero = [p for p in with_zero if p["min_non_na"] == 0.0]

    # As colunas do subconjunto vêm do cabeçalho do despejo — que o R escreveu com
    # os NOMES das features, não com os índices. Converter aqui, uma vez, evita que
    # cada teste reconstrua a correspondência e erre de forma diferente.
    subset = _subset_columns(dump)
    out_of_range = [j for j in subset if not 1 <= j <= len(features)]
    if out_of_range:
        raise SystemExit(f"índice do subconjunto fora de {len(features)} features: {out_of_range}")
    n_dumped = arrays[f"corrs_train_{batches[0]}"].shape[1]
    if n_dumped != len(subset):
        raise SystemExit(
            f"o despejo tem {n_dumped} colunas e o subconjunto declara {len(subset)}. "
            "O filtro de `refino_corr_subset` descartou índices em silêncio."
        )
    arrays["corr_subset_1based"] = np.array(subset, dtype=np.int64)

    # Os blocos volumosos vao para o npz, e nao para o JSON: 4.078 parametros e
    # 2.039 `sel_var` levaram o JSON a 1,2 MB na primeira montagem, o que
    # contradiz o desenho declarado — o JSON e o artefato que se LE. No npz eles
    # sao arrays; no JSON ficam os agregados e o digest.
    order = sorted(range(len(params)), key=lambda i: (params[i]["j"], params[i]["phase"]))
    arrays["step1_params_j"] = np.array([params[i]["j"] for i in order], dtype=np.int64)
    arrays["step1_params_is_na_phase"] = np.array(
        [params[i]["phase"] == "na" for i in order], dtype=bool
    )
    for field in ("n_zeros", "n_na", "n"):
        arrays[f"step1_params_{field}"] = np.array(
            [params[i][field] for i in order], dtype=np.int64
        )
    arrays["step1_params_min_non_na"] = np.array(
        [params[i]["min_non_na"] for i in order], dtype=np.float64
    )

    # `sel_var` e ragged — numero de preditores varia por feature. Achatado com
    # deslocamentos, que e a forma que preserva a ORDEM, e a ordem e o que o
    # passo 3 decide.
    keys = sorted(sel_var)
    flat: list[int] = []
    offsets = [0]
    for key in keys:
        flat.extend(sel_var[key])
        offsets.append(len(flat))
    arrays["sel_var_j"] = np.array([j for j, _ in keys], dtype=np.int64)
    arrays["sel_var_flat"] = np.array(flat, dtype=np.int64)
    arrays["sel_var_offsets"] = np.array(offsets, dtype=np.int64)

    payload: dict[str, Any] = {
        "provenance": {
            "mode": mode,
            "command": "scripts/gen_serrf_intermediates.sh + build_serrf_intermediates.py",
            "generated_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "r_version": response["provenance"]["r_version"],
            "image": response["provenance"]["image"],
            "image_digest": response["provenance"]["image_digest"],
            "input_sha256": response["provenance"].get("input_sha256"),
            "cores_detected": response["provenance"].get("cores_detected"),
            "detectcores_ratio": response["provenance"].get("detectcores_ratio"),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "sidecar": FIXTURE_NPZ.name,
            "sidecar_sha256": None,  # preenchido depois de escrever o npz
            "privacy": (
                "Nenhum nome de amostra em nenhum dos dois artefatos. Colunas sao "
                "identificadas por INDICE de injecao; linhas por nome de feature, que e "
                "RT_mz e nao paciente.",
            ),
            "scope": (
                "Chamada de CORRECAO apenas. serrfR roda seis vezes e as cinco da "
                "validacao cruzada foram filtradas por ncol.",
            ),
            # Declarado e MEDIDO, nao suposto: dois despejos do mesmo `_toserrf.csv`
            # foram comparados array por array. Sem esta declaracao, um `--check`
            # escrito depois compararia tudo e reprovaria sempre,
            # e quem visse `step1_after` mudar concluiria que o fixture nao reproduz,
            # quando o fato e que o ORACULO nao reproduz.
            "reproducibility": {
                # PARTICAO, e ela e exaustiva por teste: reproduziveis mais nao
                # reproduziveis tem de dar exatamente os arrays do .npz, sem
                # sobreposicao. Declarar so um lado deixaria array novo em limbo —
                # nem declarado irreprodutivel, nem afirmado reprodutivel —, e e o
                # limbo que faz a proxima pessoa repetir a investigacao.
                "reproducible_arrays": [
                    "corr_subset_1based",
                    "corrs_target_A",
                    "corrs_train_A",
                    "sel_var_flat",
                    "sel_var_j",
                    "sel_var_offsets",
                    "step1_before",
                    "step1_params_is_na_phase",
                    "step1_params_j",
                    "step1_params_n",
                    "step1_params_n_na",
                    "step1_params_n_zeros",
                ],
                "non_reproducible_arrays": ["step1_after", "step1_params_min_non_na"],
                "measured": (
                    "Duas execucoes do R com a MESMA instrumentacao, entrada, imagem, "
                    "versao e contagem de nucleos: step1_after difere em 35.166 celulas, "
                    "TODAS em posicao de zero do step1_before; min_non_na difere em 1.072 "
                    "de 4.078 entradas, TODAS da fase NA e NENHUMA da fase zeros. Os "
                    "outros doze arrays batem bit a bit."
                ),
                "measured_caveat": (
                    "A primeira versao desta medicao comparou um despejo produzido ANTES "
                    "da correcao do format() sobre vetor de caractere — o `qc` acolchoado "
                    "que fazia o consumidor ver zero QCs. Igualdade de input_sha256 foi "
                    "lida como 'mesma configuracao', e input_sha256 identifica a ENTRADA, "
                    "nao a instrumentacao. Refeita com terceiro despejo limpo, "
                    "a ESTRUTURA se confirmou identica — os mesmos dois arrays, a mesma "
                    "assimetria entre as fases, e min_non_na com a mesma contagem de "
                    "1.072 — e so a contagem de celulas de step1_after mudou, de 35.158 "
                    "para 35.166, que e quantos sorteios coincidem por acaso."
                ),
                "why": (
                    "O passo 1 substitui zero por rnorm(n, mean = 1, sd = 0.01) e o bloco "
                    "SERRF do tools.R nao chama set.seed. A assimetria entre as duas fases "
                    "e o proprio achado F1: na fase de zeros o `min` e avaliado sobre o "
                    "vetor que AINDA os contem, entao vale 0 e e deterministico; na fase "
                    "NA ele e avaliado depois da substituicao, sobre valores sorteados."
                ),
                "check_contract": (
                    "E isto e o que torna um --check escrevivel: ele exclui os dois "
                    "arrays POR DECLARACAO e nao por descoberta. Sem a particao, quem "
                    "escrevesse o --check compararia tudo, veria reprovacao sempre, e "
                    "concluiria que o GERADOR esta quebrado."
                ),
                "consequence": (
                    "Nenhuma assercao da U4 depende dos dois. As correlacoes e o sel_var "
                    "saem do step1_before; o step1_after entra apenas em duas medicoes de "
                    "ordem de magnitude — que a matriz pos-passo-1 divergiria grosseiramente "
                    "e que o passo 4 nao descarta coluna. Se um teste futuro "
                    "cobrar igualdade neles, o defeito e do teste."
                ),
            },
        },
        "shape": {
            "n_features": int(before.shape[0]),
            "n_samples": int(before.shape[1]),
            "n_qc": sum(1 for t in sample_type if t == "qc"),
            "batches": sorted(set(batch)),
        },
        "step1": {
            "n_feature_batch_pairs": len(zeros_phase),
            "n_with_zero": len(with_zero),
            "n_min_is_zero": len(min_is_zero),
            "params_in_sidecar": [
                "step1_params_j",
                "step1_params_is_na_phase",
                "step1_params_n_zeros",
                "step1_params_n_na",
                "step1_params_n",
                "step1_params_min_non_na",
            ],
            "note": (
                "O discriminante de F1. Para (feature, lote) com ao menos um zero, o `min` "
                "da linha que substitui zeros e avaliado sobre o vetor que AINDA os contem, "
                "entao min = 0 e os parametros sao (mean=1, sd=0.01) EXATOS, qualquer que "
                "seja a magnitude. Sob a especificacao da v1.2 seriam o minimo positivo, "
                "entre 1e2 e 1e9."
            ),
        },
        "step3": {
            "n_features_with_sel_var": len(sel_var),
            "n_predictors_total": sum(len(v) for v in sel_var.values()),
            "n_predictors_min": min(len(v) for v in sel_var.values()),
            "n_predictors_max": max(len(v) for v in sel_var.values()),
            "sel_var_in_sidecar": ["sel_var_j", "sel_var_flat", "sel_var_offsets"],
            "batches_seen": sorted({b for _, b in sel_var}),
            "note": (
                "sel_var por (feature, lote), indices 1-based do R. A conversao para "
                "0-based e do lado Python, feita uma vez e explicitamente."
            ),
        },
        "corr_subset_1based": subset,
        "feature_names": features,
        "batch_by_column": batch,
        "sample_type_by_column": sample_type,
    }
    return payload, arrays


def _tag(dump: Path) -> str:
    """O sufixo `n<ncol>` que o despejo usa, lido do arquivo em vez de suposto."""
    candidates = sorted(dump.glob("step1_after_n*.tsv"))
    if len(candidates) != 1:
        raise SystemExit(
            f"esperava exatamente um step1_after_n*.tsv e achei {len(candidates)}: "
            f"{[c.name for c in candidates]}. Mais de um significa que o filtro de "
            "escopo deixou passar outra chamada de serrfR."
        )
    return candidates[0].stem.removeprefix("step1_after_")


def _subset_columns(dump: Path) -> list[int]:
    """Os índices 1-based do subconjunto, lidos do `subset.txt` que o despejo grava.

    Não do cabeçalho do TSV: `cor()` no R devolve matriz **sem dimnames**, então as
    colunas saem como `V1, V2, V3` e a correspondência com os índices declarados se
    perderia. Reconstruí-la a partir da variável de ambiente faria dois lugares ter
    de concordar, e é a forma de deriva que se quer evitar.
    """
    path = dump / "subset.txt"
    if not path.is_file():
        raise SystemExit(
            f"ausente: {path}. O despejo tem de gravar o proprio subconjunto — sem "
            "ele nao ha como saber a que feature cada coluna corresponde."
        )
    return [int(v) for v in path.read_text(encoding="utf-8").strip().split(",") if v.strip()]


def comparable_arrays(declaration: dict[str, Any], names: Iterable[str]) -> list[str]:
    """Quais arrays um `--check` compara: todos, menos os declarados irreprodutíveis.

    **Exclusão por DECLARAÇÃO e não por descoberta**, e é essa a diferença que faz
    este alvo ser escrevível. Sem a partição no fixture, quem escrevesse o `--check`
    compararia tudo, veria reprovação em `step1_after` a cada execução, e concluiria
    que o GERADOR está quebrado — quando o irreprodutível é o oráculo, que substitui
    zeros por `rnorm` sem `set.seed`. Portão que sempre reprova não serve.

    Levanta se a declaração nomear array que não existe: declaração órfã é isenção
    que sobreviveu ao que isentava, e aqui ela silenciaria uma comparação de verdade.
    """
    todos = list(names)
    excluidos = list(declaration.get("non_reproducible_arrays", []))
    orfaos = sorted(set(excluidos) - set(todos))
    if orfaos:
        raise SystemExit(
            f"a declaracao de reprodutibilidade nomeia {orfaos}, que nao esta(o) no "
            "despejo. Isencao orfa deixa de excluir o que existe e passa a nao "
            "excluir nada — conferir antes de rodar --check."
        )
    return sorted(set(todos) - set(excluidos))


def sem_ruido(raw: dict[str, Any]) -> str:
    """O JSON que o `--check` compara, sem o que muda por definicao.

    **Esta no modulo e nao dentro de `main`.** Enquanto era local, nenhum teste podia
    invoca-la, e a unica forma de a exercitar era correr o `--check` inteiro, que exige
    `--dump-dir` com os despejos do R.

    `generated_at` e `sidecar_sha256` saem; `min_non_na` sai de `step1.params` porque a fase NA
    dele e sorteada.

    **`python_version` FICA, truncado em `major.minor`.** O argumento inteiro esta em
    `scripts/gen_insample_medians.py`: os quatro pinos que o projeto declara sao todos no minor,
    e comparar o patch tornava este arquivo mais estrito que eles. Este e o QUARTO membro da
    classe, achado por varredura e nao por reprovacao — o `--check` daqui precisa dos despejos,
    entao a divergencia esperaria a proxima regeneracao para aparecer.
    """
    copia = json.loads(json.dumps(raw))
    copia["provenance"].pop("generated_at", None)
    copia["provenance"].pop("sidecar_sha256", None)
    versao = copia["provenance"].get("python_version")
    if isinstance(versao, str):
        copia["provenance"]["python_version"] = ".".join(versao.split(".")[:2])
    # `min_non_na` viaja tambem no JSON, dentro de step1.params, e a fase NA dele
    # e sorteada — mesma exclusao, pelo mesmo motivo, no outro artefato.
    for entry in copia.get("step1", {}).get("params", []):
        entry.pop("min_non_na", None)
    return json.dumps(copia, indent=2, ensure_ascii=False, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--json", type=Path, default=FIXTURE_JSON)
    parser.add_argument("--npz", type=Path, default=FIXTURE_NPZ)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "compara com o par versionado sem escrever. Os arrays declarados "
            "irreproduziveis em provenance.reproducibility ficam FORA da comparacao"
        ),
    )
    args = parser.parse_args()

    response_path = args.dump_dir.parent / "response.json"
    if not response_path.is_file():
        raise SystemExit(f"ausente: {response_path} — sem procedência não se congela nada")
    response = json.loads(response_path.read_text(encoding="utf-8"))

    payload, arrays = build(args.dump_dir, args.mode, response)

    if args.check:
        return _check(args.json, args.npz, payload, arrays)

    np.savez_compressed(args.npz, **arrays)
    payload["provenance"]["sidecar_sha256"] = digest_of(args.npz)

    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"escrito: {args.json.name} ({args.json.stat().st_size / 1024:.0f} KB)")
    print(f"escrito: {args.npz.name} ({args.npz.stat().st_size / 1024:.0f} KB)")
    print(f"  arrays: {', '.join(sorted(arrays))}")
    step1 = payload["step1"]
    print(
        f"  passo 1: {step1['n_with_zero']} de {step1['n_feature_batch_pairs']} pares com "
        f"zero, min == 0 em {step1['n_min_is_zero']}"
    )
    return 0


def _check(
    json_path: Path, npz_path: Path, payload: dict[str, Any], arrays: dict[str, np.ndarray]
) -> int:
    """Compara sem escrever, e RELATA o que ficou fora — silêncio aqui seria pior.

    Um `--check` que excluísse arrays sem dizer quais produziria a impressão de
    comparação completa, que é um modo de falha por omissão: saída
    indistinguível de sucesso.
    """
    if not json_path.is_file() or not npz_path.is_file():
        print("par versionado ausente; rode sem --check", file=sys.stderr)
        return 1
    frozen = json.loads(json_path.read_text(encoding="utf-8"))
    declaration = frozen["provenance"].get("reproducibility", {})

    actual = digest_of(npz_path)
    if actual != frozen["provenance"]["sidecar_sha256"]:
        print(f"o .npz mudou sem o JSON acompanhar: {actual[:12]}", file=sys.stderr)
        return 1

    comparaveis = comparable_arrays(declaration, arrays)
    excluidos = sorted(set(arrays) - set(comparaveis))
    with np.load(npz_path) as sidecar:
        if set(sidecar.files) != set(arrays):
            print(
                f"o sidecar traz {sorted(set(sidecar.files) - set(arrays))} a mais e "
                f"{sorted(set(arrays) - set(sidecar.files))} a menos",
                file=sys.stderr,
            )
            return 1
        divergentes = [
            name
            for name in comparaveis
            if not np.array_equal(sidecar[name], arrays[name], equal_nan=True)
        ]
    if divergentes:
        print(f"divergiram da regeneracao: {divergentes}", file=sys.stderr)
        return 1

    if sem_ruido(frozen) != sem_ruido(payload):
        print("o JSON DIVERGE da regeneracao; rode sem --check", file=sys.stderr)
        return 1

    print(f"par identico a regeneracao em {len(comparaveis)} array(s)")
    print(f"  FORA da comparacao, por declaracao: {excluidos or 'nenhum'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Gera `src/refino/decision/politica_aprendida.json` a partir do corpus rotulado.

**Este script ESCREVE; quem confere é `tests/test_politica_aprendida.py`.** A separação é a de
sempre neste repositório: um gerador que também verificasse produziria o desfecho de sucesso por
construção, e o portão não teria como reprovar.

Uso:

    .venv/bin/python scripts/gera_politica_aprendida.py
    .venv/bin/python scripts/gera_politica_aprendida.py --conferir   # não escreve; só compara

A semente é **42** e não é escolha livre: é a que os scripts de medição usam, e mudá-la
produziria um modelo diferente daquele sobre o qual as medições reportaram.

**O corpus padrão é o de 7 QCs, rotulado com a floresta do SERRF que conta como o `ranger`**
(`data/fixtures/corpus_fase9_7qcs_ranger.jsonl`): 40 perfis diagnosticados com a política do
aplicativo — a mesma que `POLITICA_DE_DIAGNOSTICO_DO_CORPUS` declara e que a árvore recebe em
produção —, com os rótulos que essa floresta produz. Os corpora anteriores continuam no
repositório, e cada um regera o seu modelo:

- `--corpus data/fixtures/corpus_fase9_7qcs.jsonl` regera o Palpite da floresta `scikit_learn`, que
  está guardado em `data/fixtures/politica_aprendida_scikit_learn.json` para quem quiser voltar;
- `--corpus data/fixtures/corpus_fase9.jsonl` regera o de oito QCs;
- `--corpus data/fixtures/corpus_pela_verdade_88.jsonl --destino
  src/refino/decision/politica_pela_verdade.json` regera o Palpite pela verdade.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# A saída deste script tem prosa acentuada, e a console deste clone abre em `cp932`. Sem isto ele
# morre num `UnicodeEncodeError` com saída 1 — reprovação INVENTADA, que custa mais que a escondida
# porque manda alguém procurar um defeito que não existe.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

CORPUS = REPO / "data" / "fixtures" / "corpus_fase9_7qcs_ranger.jsonl"
DESTINO = REPO / "src" / "refino" / "decision" / "politica_aprendida.json"
SEMENTE = 42


def carrega(caminho: Path) -> tuple[list[Any], list[str], list[float]]:
    """Perfis, rótulos e margens do corpus.

    Reconstrói o `DefectProfile` do registro **inteiro** — as treze quantidades —, e não de um
    subconjunto: o corpus as grava justamente porque treinar a partir de três exigiria regerar as
    40 matrizes.
    """
    from refino.domain.diagnosis import DefectProfile

    perfis: list[Any] = []
    rotulos: list[str] = []
    margens: list[float] = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        registro = json.loads(linha)
        p = registro["profile"]
        perfis.append(
            DefectProfile(
                defect_type=p["defect_type"],
                confidence=p["confidence"],
                suggested_batches=tuple(p["suggested_batches"]),
                changepoints=tuple(p["changepoints"]),
                qc_rsd_median=p["qc_rsd_median"],
                qc_rsd_p75=p["qc_rsd_p75"],
                zero_fraction=p["zero_fraction"],
                missing_fraction=p["missing_fraction"],
                pc1_injection_correlation=p["pc1_injection_correlation"],
                qc_dispersion_ratio=p["qc_dispersion_ratio"],
                n_features=p["n_features"],
                n_samples=p["n_samples"],
                n_qcs=p["n_qcs"],
            )
        )
        rotulos.append(registro["rotulo"])
        margens.append(float(registro["margem"]))
    return perfis, rotulos, margens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--destino", type=Path, default=DESTINO)
    ap.add_argument("--semente", type=int, default=SEMENTE)
    ap.add_argument(
        "--conferir",
        action="store_true",
        help="não escreve: compara o artefato em disco com o que seria gerado agora",
    )
    args = ap.parse_args()

    from refino.decision.artefato import exporta
    from refino.decision.model import LearnedPolicy

    if not args.corpus.exists():
        print(f"corpus ausente: {args.corpus}")
        return 2

    perfis, rotulos, margens = carrega(args.corpus)
    if not perfis:
        print(f"corpus vazio: {args.corpus}")
        return 2

    politica = LearnedPolicy(seed=args.semente).fit(perfis, rotulos, margens)  # type: ignore[arg-type]
    dados = exporta(politica, corpus=args.corpus, n_corpus=len(perfis))

    if args.conferir:
        if not args.destino.exists():
            print(f"FALHA: artefato ausente em {args.destino}")
            return 1
        # A procedência carrega a data de geração, que muda a cada execução e não é comportamento.
        # Comparar as duas com ela dentro faria este modo reprovar todos os dias.
        atual = json.loads(args.destino.read_text(encoding="utf-8"))
        for lado in (atual, dados):
            lado.get("procedencia", {}).pop("gerado_em", None)
        if atual == dados:
            print(f"OK: {args.destino.name} bate com o corpus de {len(perfis)} perfis")
            return 0
        print(f"FALHA: {args.destino.name} diverge do que o corpus produz agora. Regere.")
        return 1

    args.destino.parent.mkdir(parents=True, exist_ok=True)
    args.destino.write_text(
        json.dumps(dados, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    nos_corretor = len(dados["corretor"]["nos"])
    nos_margem = len(dados["margem"]["nos"]) if dados["margem"] else 0
    print(
        f"escrito: {args.destino}\n"
        f"  corpus     : {args.corpus.name}, {len(perfis)} perfis\n"
        f"  semente    : {args.semente}\n"
        f"  corretor   : {nos_corretor} nós\n"
        f"  margem     : {nos_margem} nós"
        + ("" if nos_margem else "  (cabeça ausente: nenhuma margem finita no corpus)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

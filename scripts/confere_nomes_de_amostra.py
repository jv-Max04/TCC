"""Confere que nenhum nome de amostra de paciente entrou nos arquivos preparados para o commit.

Os dados do laboratório ficam fora do repositório, e o nome de uma amostra de paciente é
dado do laboratório. Os nomes vêm do cabeçalho de `serrf/data/input/measurements_*.csv` e
da coluna `inj_order_name` de `Sample_List_*.csv`. Os QCs ficam de fora: são a mistura de
controle, não um paciente.

Nunca imprime um nome: só o arquivo e o comprimento do nome que casou.

Uso (da raiz do repositório):

    python -I -B scripts/confere_nomes_de_amostra.py        # o índice do git
    python -I -B scripts/confere_nomes_de_amostra.py A B    # estes arquivos

Saída 0: nenhum nome. Saída 1: algum nome entrou. Saída 2: sem os arquivos do laboratório,
a conferência não aconteceu.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENTRADA = REPO / "serrf" / "data" / "input"


def nomes_de_paciente(entrada: Path) -> set[str]:
    """Os nomes de amostra menos os que a linha `LABEL` de algum `measurements` marca `QC`."""
    nomes: set[str] = set()
    qcs: set[str] = set()
    for medicoes in sorted(entrada.glob("measurements_*.csv")):
        with medicoes.open(encoding="utf-8-sig", newline="") as arquivo:
            leitor = csv.reader(arquivo)
            cabecalho = [c.strip() for c in next(leitor, [])]
            rotulos = [c.strip() for c in next(leitor, [])]
        if not rotulos or rotulos[0] != "LABEL":
            raise ValueError(f"{medicoes.name}: a segunda linha não é a `LABEL`")
        for nome, rotulo in zip(cabecalho[1:], rotulos[1:], strict=True):
            if nome:
                (qcs if rotulo == "QC" else nomes).add(nome)
    for lista in sorted(entrada.glob("Sample_List_*.csv")):
        with lista.open(encoding="utf-8-sig", newline="") as arquivo:
            for linha in csv.DictReader(arquivo):
                nome = (linha.get("inj_order_name") or "").strip()
                if nome and nome not in qcs:
                    nomes.add(nome)
    return nomes


def arquivos_no_indice() -> list[Path]:
    saida = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    return [REPO / nome for nome in saida.split("\0") if nome]


def main() -> int:
    for fluxo in (sys.stdout, sys.stderr):
        fluxo.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arquivos", nargs="*", type=Path)
    ap.add_argument("--entrada", type=Path, default=ENTRADA)
    args = ap.parse_args()

    try:
        nomes = nomes_de_paciente(args.entrada)
    except ValueError as erro:
        print(f"{erro}: a conferência NÃO aconteceu")
        return 2
    if not nomes:
        print(f"sem nomes de amostra em {args.entrada}: a conferência NÃO aconteceu")
        return 2
    alvos = args.arquivos or arquivos_no_indice()
    casos = 0
    for alvo in alvos:
        if not alvo.is_file():
            continue
        texto = alvo.read_bytes().decode("utf-8", errors="replace")
        for nome in nomes:
            if nome in texto:
                casos += 1
                print(f"NOME DE AMOSTRA em {alvo} (um nome de {len(nome)} caracteres)")
    print(f"{len(nomes)} nomes de paciente, {len(alvos)} arquivos conferidos, {casos} casos")
    return 1 if casos else 0


if __name__ == "__main__":
    raise SystemExit(main())

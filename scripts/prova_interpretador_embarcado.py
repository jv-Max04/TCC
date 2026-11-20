#!/usr/bin/env python3
"""O interpretador embarcado ARRANCA e importa do que já viaja em `_internal`.

## O que esta sonda decide

É ela que fecha uma incógnita antes de qualquer empacotamento: se um
interpretador comum, posto dentro de `_internal`, consegue importar `numpy`, `sklearn`, `joblib` e
`refino` **dos `.pyc` soltos que o `noarchive` produziu** — ou se são precisas segundas cópias das
dependências, que custariam centenas de megabytes.

## As três asserções, e a segunda é a que separa as máquinas

1. **O interpretador arranca.** Um `pythonw.exe` copiado sozinho morre em `init_fs_encoding` antes
   de ver argumento nenhum, e essa falha é diferente de *"o numpy não importou"*.
2. **`sys.prefix` e TODA entrada de `sys.path` resolvem dentro de `_internal`.** Sem esta, a sonda
   passa na máquina que constrói e falha na do usuário: o pino do uv está instalado nesta máquina,
   e sem o `._pth` o interpretador cai no registro e importa de lá. **A asserção de `import`
   sozinha não distingue os dois casos.**
3. **Um módulo que NÃO viaja falha ao importar.** É a âncora de não-vacuidade: sem ela, um
   `sys.path` que alcançasse a `.venv` deste clone satisfaria tudo o que está acima.

## Por que ela corre o interpretador de FORA em vez de importar aqui

O que se mede é o processo que o `loky` vai lançar, com o ambiente que ele vai ter — não o
interpretador que corre esta sonda. `PYTHONHOME` e `PYTHONPATH` são limpos de propósito: um deles
herdado mediria a máquina de quem constrói.

## Uso

    .venv/Scripts/python.exe scripts/prova_interpretador_embarcado.py [--pasta dist/refino]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

REPO = Path(__file__).resolve().parent.parent

# Os cinco que o worker do `loky` precisa de alcançar. `popen_loky_win32` está aqui porque é o
# módulo que o filho executa: sem ele, o worker morre antes de tocar no corretor.
ESSENCIAIS = (
    "numpy",
    "sklearn",
    "joblib",
    "refino.correction.serrf",
    "joblib.externals.loky.backend.popen_loky_win32",
)

# **A âncora de não-vacuidade.** `pytest` é dependência de desenvolvimento e está excluída do
# congelado (`EXCLUIDOS` no `.spec`) — se ele importar, o `sys.path` alcançou a `.venv` deste
# clone e tudo o que a sonda afirma passa a ser sobre a máquina errada.
NAO_VIAJA = "pytest"

# O programa que corre DENTRO do interpretador embarcado. Ele não levanta: devolve um JSON, para
# que a sonda distinga «não arrancou» de «arrancou e não importou».
PROGRAMA = """
import json, sys
saida = {"prefix": sys.prefix, "path": list(sys.path), "version": sys.version.split()[0]}
saida["importados"] = {}
for nome in %(essenciais)r:
    try:
        __import__(nome)
        mod = sys.modules[nome]
        saida["importados"][nome] = getattr(mod, "__file__", "sem __file__") or "namespace"
    except BaseException as erro:
        saida["importados"][nome] = "ERRO: %%s: %%s" %% (type(erro).__name__, erro)
try:
    __import__(%(nao_viaja)r)
    saida["nao_viaja"] = "IMPORTOU — o sys.path alcancou algo fora de _internal"
except BaseException as erro:
    saida["nao_viaja"] = "recusou: %%s" %% type(erro).__name__
print(json.dumps(saida))
"""


def _tamanho(pasta: Path) -> tuple[int, int]:
    arquivos = [c for c in pasta.rglob("*") if c.is_file()]
    return len(arquivos), sum(c.stat().st_size for c in arquivos)


def main() -> int:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--pasta", default="dist/refino")
    argumentos = analisador.parse_args()

    pasta = REPO / argumentos.pasta
    interno = pasta / "_internal"
    interpretador = interno / "pythonw.exe"

    print(f"pasta   : {pasta}")
    if not interpretador.is_file():
        print(f"FALHA: o interpretador embarcado não está em {interpretador}", file=sys.stderr)
        print("  Congele primeiro: powershell -File scripts\\congela_windows.ps1", file=sys.stderr)
        return 1
    pth = interno / "python312._pth"
    print(f"._pth   : {'presente' if pth.is_file() else 'AUSENTE'}")

    # **O ambiente vai LIMPO.** Herdar `PYTHONHOME` ou `PYTHONPATH` faria a sonda medir esta
    # máquina em vez da pasta entregue, que é precisamente o desfecho falso que ela existe para não
    # produzir.
    ambiente = {k: v for k, v in os.environ.items() if k not in ("PYTHONHOME", "PYTHONPATH")}

    programa = PROGRAMA % {"essenciais": list(ESSENCIAIS), "nao_viaja": NAO_VIAJA}
    try:
        corrida = subprocess.run(
            [str(interpretador), "-c", programa],
            capture_output=True,
            text=True,
            timeout=300,
            env=ambiente,
            cwd=str(interno),
        )
    except OSError as erro:
        print(f"FALHA: o interpretador não pôde ser lançado: {erro}", file=sys.stderr)
        return 1

    if corrida.returncode != 0 or not corrida.stdout.strip():
        print(
            f"FALHA: o interpretador NÃO ARRANCOU — código {corrida.returncode}, "
            f"stdout vazio={not corrida.stdout.strip()}",
            file=sys.stderr,
        )
        print(f"  stderr: {corrida.stderr[:600]}", file=sys.stderr)
        print(
            "  É falha diferente de «o numpy não importou»: sem `._pth` ele procura a stdlib pelo "
            "marco `Lib\\os.py`, que não existe em `_internal`.",
            file=sys.stderr,
        )
        return 1

    try:
        lido = json.loads(corrida.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        print(f"FALHA: saída não é JSON: {corrida.stdout[:400]!r}", file=sys.stderr)
        return 1

    print(f"versão  : {lido['version']}")
    print(f"prefix  : {lido['prefix']}")

    problemas: list[str] = []

    # (1) os cinco essenciais
    for nome, origem in lido["importados"].items():
        marca = "ok " if not str(origem).startswith("ERRO") else "NAO"
        print(f"  {marca} {nome:<48} {str(origem)[:90]}")
        if str(origem).startswith("ERRO"):
            problemas.append(f"{nome} não importou")

    # (2) TODA entrada de `sys.path` dentro de `_internal` — a asserção que separa as duas máquinas
    fora = [
        c for c in lido["path"] if c and not Path(c).resolve().is_relative_to(interno.resolve())
    ]
    if fora:
        problemas.append(f"{len(fora)} entrada(s) de sys.path fora de _internal")
        for c in fora:
            print(f"  NAO sys.path fora de _internal: {c}")
    if not Path(lido["prefix"]).resolve().is_relative_to(interno.resolve()):
        problemas.append("sys.prefix fora de _internal")

    # (3) a âncora de não-vacuidade
    print(f"  âncora  {NAO_VIAJA} → {lido['nao_viaja']}")
    if not str(lido["nao_viaja"]).startswith("recusou"):
        problemas.append(f"{NAO_VIAJA} importou — o sys.path alcança fora da pasta")

    n, bytes_ = _tamanho(pasta)
    print(f"\npasta   : {n:,} arquivos · {bytes_ / 1024 / 1024:.1f} MB".replace(",", "."))

    if problemas:
        print(f"\nFALHA: {len(problemas)} — {'; '.join(problemas)}", file=sys.stderr)
        return 1
    print("\nos cinco importam, o sys.path fica dentro de `_internal`, e o que não viaja não entra")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Monta a pasta das bibliotecas de dados das vias, com os hashes do manifesto.

É o que a máquina que congela corre para ter a pasta que viaja **só na pasta do laboratório**.
Usa a mesma fronteira que a janela (`refino.io.dados_vias`), com os mesmos controles de
transporte — e uma regra a mais, que é a diferença entre os dois regimes: **aqui o hash TEM de ser
o do manifesto.** O servidor troca os arquivos sem versão; se trocou, a montagem para e diz qual, em
vez de levar ao laboratório uma biblioteca que ninguém validou.

    .venv/Scripts/python.exe scripts/busca_dados_vias.py --destino <pasta>

Sai 0 com a pasta conferida, 1 com o motivo. Uma pasta que já confere não é baixada de novo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from refino.io import dados_vias


def _arquivos_a_mais(destino: Path, esperados: set[str]) -> list[str]:
    """Os arquivos da pasta que o manifesto não lista, em QUALQUER nível, pelo caminho relativo.

    Recursivo porque o congelamento copia esta pasta com `Copy-Item -Recurse`
    (`scripts/congela_windows.ps1`): um arquivo numa subpasta viaja do mesmo jeito que um do
    primeiro nível. Olhar só o primeiro nível deixava passar `sub/x` com saída 0. A comparação é
    pelo caminho relativo, e não pelo nome: `sub/alfa.qs` não é a `alfa.qs` que o manifesto lista,
    e o nome igual não pode escondê-la.
    """
    if not destino.is_dir():
        return []
    relativos = (p.relative_to(destino).as_posix() for p in destino.rglob("*") if p.is_file())
    return sorted(r for r in relativos if r not in esperados)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--destino", required=True)
    ap.add_argument(
        "--so-confere",
        action="store_true",
        help=(
            "só confere, nunca baixa; reprova também arquivo que o manifesto não lista, "
            "inclusive dentro de subpasta"
        ),
    )
    a = ap.parse_args()
    destino = Path(a.destino)
    bibliotecas = dados_vias.le_manifesto()
    antes = dados_vias.confere_pasta(destino, bibliotecas)
    if a.so_confere:
        # O modo do congelamento: a pasta que viaja é a que está aqui, e um arquivo a mais
        # nela — em qualquer nível, porque a cópia é recursiva — viajaria sem ninguém o ter
        # conferido.
        sobram = _arquivos_a_mais(destino, {b.arquivo for b in bibliotecas})
        print(
            f"conferência: {antes.estado}; ausentes {list(antes.ausentes) or 'nenhuma'}; "
            f"divergentes {list(antes.divergentes) or 'nenhuma'}; sobram {sobram or 'nenhum'}"
        )
        return 0 if antes.estado == "ok" and not sobram else 1
    if antes.estado == "ok":
        print(f"a pasta já confere com o manifesto: {len(bibliotecas)} bibliotecas em {destino}")
        return 0
    resultado = dados_vias.baixa_bibliotecas(destino, bibliotecas, confirmado=True)
    if resultado.desfecho != "baixado":
        print(f"o download não terminou ({resultado.desfecho}): {resultado.motivo}")
        return 1
    trocadas = [b.arquivo for b in resultado.baixadas if not b.validada]
    if trocadas:
        # O download é tudo-ou-nada e já renomeou os arquivos; aqui eles saem, para que a pasta não
        # fique com uma biblioteca que ninguém validou à espera de ser empacotada.
        for b in resultado.baixadas:
            (destino / b.arquivo).unlink(missing_ok=True)
        (destino / dados_vias.REGISTRO_DE_DOWNLOAD).unlink(missing_ok=True)
        print(
            f"o servidor trocou {len(trocadas)} biblioteca(s) desde o manifesto: "
            f"{', '.join(trocadas)}"
        )
        print(
            "a pasta do laboratório só leva cópias com o hash do manifesto; "
            "as baixadas foram apagadas."
        )
        return 1
    # Na pasta do laboratório quem vale é o manifesto, e não o registro de download — que é o da
    # release pública. Ele sai para que as duas pastas não pareçam a mesma coisa.
    (destino / dados_vias.REGISTRO_DE_DOWNLOAD).unlink(missing_ok=True)
    depois = dados_vias.confere_pasta(destino, bibliotecas)
    print(f"conferência: {depois.estado}; {len(bibliotecas)} bibliotecas em {destino}")
    return 0 if depois.estado == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

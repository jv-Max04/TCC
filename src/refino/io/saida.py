"""A saída de texto em UTF-8, para os dois pontos de entrada do programa.

Fica em `io/` porque a linha de comando precisa da mesma guarda e não pode importar o pacote da
janela: a CLI é instalada também sem o grupo `desktop`. Num Windows cuja página de código não
codifica os acentos, a saída em tubo levanta `UnicodeEncodeError` na primeira mensagem acentuada,
e o comando termina com erro mesmo depois de ter gravado o arquivo.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["garante_utf8"]


def garante_utf8(fluxos: Sequence[str] | None = None) -> tuple[str, ...]:
    """Reconfigura as saídas de texto para UTF-8 e devolve os nomes dos fluxos reconfigurados.

    Tolera fluxo ausente: um executável `--windowed` do PyInstaller roda sem console, `sys.stdout`
    é `None`, e levantar ali impediria o arranque.
    """
    nomes = ("stdout", "stderr") if fluxos is None else tuple(fluxos)
    feitos: list[str] = []
    for nome in nomes:
        fluxo = getattr(sys, nome, None)
        reconfigura = getattr(fluxo, "reconfigure", None)
        if reconfigura is None:
            continue
        try:
            reconfigura(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            # Fluxo redirecionado para algo que não aceita reconfiguração. Não impede o arranque.
            continue
        feitos.append(nome)
    return tuple(feitos)

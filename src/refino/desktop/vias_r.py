"""Onde estão o R embarcado, o roteiro das vias e as bibliotecas de dados.

Os caminhos vêm de `sys.executable` e de mais lugar nenhum, pelo mesmo motivo de
`paralelismo.resolve_o_interpretador`: o `PATH` não é consultado e nada é resolvido contra o
diretório de trabalho, então um `Rscript.exe` plantado onde o programa foi aberto não é executado.
O lançador (`io/enriquecimento_r.py`) recebe os caminhos já resolvidos e diz 'motor ausente', com o
caminho procurado, quando o R não está lá.

As duas pastas de bibliotecas têm regras distintas:

- `vias-dados/`, ao lado do executável, é a da pasta do laboratório: viaja fixada, e só passa cópia
  com o hash do manifesto;
- `%LOCALAPPDATA%\\refino\\vias-bibliotecas-baixadas\\` é a da versão pública, que sai sem as
  bibliotecas: o download, confirmado pelo usuário, vai para lá, e o resultado sai marcado como
  calculado com biblioteca baixada e não validada quando o hash difere do manifesto.

O nome da segunda é outro de propósito, para que as duas não se confundam no código nem na tela.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from refino.desktop.banco import PASTA

__all__ = ["PASTA_DAS_BIBLIOTECAS", "PASTA_DO_R", "ROTEIRO", "MotorDoR", "resolve_motor"]

PASTA_DO_R: Final = "R"
PASTA_DAS_BIBLIOTECAS: Final = "vias-dados"
ROTEIRO: Final = "vias_entrypoint.R"
# `PASTA` e não a string: a pasta-raiz por usuário é uma só, a mesma do banco e do cache de
# figuras.
_BAIXADAS: Final = (PASTA, "vias-bibliotecas-baixadas")


@dataclass(frozen=True)
class MotorDoR:
    r_home: Path
    roteiro: Path
    bibliotecas_fixadas: Path
    bibliotecas_baixadas: Path


def resolve_motor(
    executavel: str | None = None, ambiente: Mapping[str, str] | None = None
) -> MotorDoR:
    """Os caminhos do motor, sempre; quem confere se existem é o lançador ou o painel."""
    raiz = Path(executavel if executavel is not None else sys.executable).resolve().parent
    env = os.environ if ambiente is None else ambiente
    base_local = Path(env.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    r_home = raiz / PASTA_DO_R
    return MotorDoR(
        r_home=r_home,
        roteiro=r_home / ROTEIRO,
        bibliotecas_fixadas=raiz / PASTA_DAS_BIBLIOTECAS,
        bibliotecas_baixadas=base_local.joinpath(*_BAIXADAS),
    )

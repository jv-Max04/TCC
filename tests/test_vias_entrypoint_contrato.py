"""O contrato de `r_service/vias_entrypoint.R`, conferido sem R.

O roteiro só corre contra um R de verdade na prova do executável congelado. O que cabe aqui é a
leitura do arquivo, e ela pega três descuidos de edição que só apareceriam tarde:

- **o vocabulário de recusa é fechado e usado inteiro.** Todo `recusar("x")` usa um nome declarado
  em `DESFECHOS`, e todo nome declarado é usado. O lado Python valida esse vocabulário como
  fronteira de confiança; um desfecho novo no R que o Python não conhece cairia num
  `else`;
- **o roteiro não fala com a rede.** É o roteiro do modo local: nenhuma chamada de download, de
  `httr`, de `curl` ou de conexão;
- **os contornos estão instalados.**

As plantas são em processo: a função recebe o texto, e o texto plantado é o argumento.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROTEIRO = REPO / "r_service" / "vias_entrypoint.R"

# Os nomes de rede que o roteiro do modo local não pode chamar. `download.file` é o que
# `SetCurrentMsetLib` chamaria; os outros são os clientes que o pacote usa no modo externo.
REDE = (
    r"\bdownload\.file\s*\(",
    r"\bhttr::",
    r"\bcurl::",
    r"\burl\s*\(",
    r"\bsocketConnection\s*\(",
)
CONTORNOS = (
    ".get.my.lib",
    ".do.api.call",
    "ExportResultMatArrow",
    "ExportOraMembershipJson",
    "PlotORAMembership",
)


def _codigo(texto: str) -> str:
    """O roteiro sem comentários: a prosa cita `download.file` para explicar por que não o chama."""
    return "\n".join(linha.split("#", 1)[0] for linha in texto.splitlines())


def declarados(texto: str) -> set[str]:
    bloco = re.search(r"DESFECHOS\s*<-\s*c\(([^)]*)\)", texto, re.S)
    assert bloco is not None, "o roteiro não declara DESFECHOS"
    return set(re.findall(r'"([a-z_]+)"', bloco.group(1)))


def usados(texto: str) -> set[str]:
    return set(re.findall(r'recusar\(\s*"([a-z_]+)"', _codigo(texto)))


def chamadas_de_rede(texto: str) -> list[str]:
    codigo = _codigo(texto)
    return [padrao for padrao in REDE if re.search(padrao, codigo)]


def test_o_vocabulario_de_recusa_e_fechado_e_usado_inteiro() -> None:
    texto = ROTEIRO.read_text(encoding="utf-8")
    d, u = declarados(texto), usados(texto)
    assert len(d) >= 5, f"âncora: {sorted(d)}"
    assert u == d, (
        f"declarados e não usados: {sorted(d - u)}; usados e não declarados: {sorted(u - d)}"
    )
    # par causal: um desfecho novo usado e não declarado reprova
    plantado = texto + '\nrecusar("desfecho_inventado", "x")\n'
    assert usados(plantado) - declarados(plantado) == {"desfecho_inventado"}


def test_o_roteiro_do_modo_local_nao_chama_a_rede() -> None:
    texto = ROTEIRO.read_text(encoding="utf-8")
    assert "download.file" in texto, "âncora: a prosa explica o download que o roteiro evita"
    assert chamadas_de_rede(texto) == []
    # par causal: a mesma chamada FORA de comentário reprova
    assert chamadas_de_rede(texto + '\ndownload.file("https://x", "y")\n') == [REDE[0]]


def test_os_contornos_estao_instalados_no_namespace() -> None:
    codigo = _codigo(ROTEIRO.read_text(encoding="utf-8"))
    assert "assignInNamespace" in codigo
    for nome in CONTORNOS:
        assert f'"{nome}"' in codigo, f"o roteiro não substitui {nome}"


def test_o_vocabulario_do_roteiro_e_o_que_o_lancador_valida() -> None:
    """O `desfecho` que o R devolve é validado contra `ViasRecusa`; os dois são um só."""
    from typing import get_args

    from refino.domain.labels import ViasRecusa

    texto = ROTEIRO.read_text(encoding="utf-8")
    assert declarados(texto) == set(get_args(ViasRecusa))
    # par causal: um nome a mais no roteiro desfaz a igualdade
    plantado = texto.replace('"motor_falhou")', '"motor_falhou", "novo_desfecho")', 1)
    assert declarados(plantado) - set(get_args(ViasRecusa)) == {"novo_desfecho"}

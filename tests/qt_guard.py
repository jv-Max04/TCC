"""A guarda do grupo opcional `desktop` do aplicativo nativo.

**Por que uma guarda própria e não `pytest.importorskip`.** O `importorskip` produz uma mensagem que
o `pytest` escolhe, e `tests/skip_policy.py` casa o motivo do `skip` contra um regime declarado: um
motivo que não controlamos não é declarável, e `skip` cuja mensagem nenhum padrão cobre reprova a
sessão pela direção de runtime daquele portão.

**Por que a função é PURA e o ambiente entra por argumento.** A planta é
parâmetro, então o negativo percorre este mesmo caminho e não pode ficar inerte. A alternativa —
uma função que sonda o ambiente por dentro — só é exercitável na máquina que **não** tem o grupo,
que é exactamente a máquina onde ninguém corre o teste.

## A interação com o portão de pinos, e ela é deliberada

`tests/test_dependency_pins.py` trata grupo **parcialmente** instalado como defeito. Esta guarda
pula com **qualquer** membro ausente, inclusive nesse caso. As duas coisas juntas dão o desfecho
certo: o portão de pinos **reprova** e este módulo **pula**, então a reprovação fica visível e o
`skip` não a esconde.

> Sem a atomicidade do portão de pinos, grupo meio instalado produziria `skipped` — o desfecho
> que a política ensina a ler como legítimo — e absorveria uma instalação quebrada.

## O motivo nomeia QUEM FALTA, e não uma causa suposta

A primeira versão do regime `sem_endereco_externo` dizia *"máquina sem endereço IPv4
não-loopback"* e isso era causa **afirmada**: numa máquina com endereço mas sem rota, o `skip`
disparava com motivo falso. Aqui a mensagem enumera os membros ausentes medidos, não uma hipótese
sobre por que faltam.
"""

from __future__ import annotations

import importlib.metadata as metadata
import os
from typing import Final

__all__ = [
    "MEMBROS_EXIGIDOS",
    "PLATAFORMA_SEM_TELA",
    "garante_plataforma_offscreen",
    "membros_ausentes",
    "sonda_o_ambiente",
]

# **`offscreen` e nao `minimal`:** o `minimal` nao entrega eventos de rato, e `qtbot.mouseClick` é
# metade do que os testes de janela exercitam. Medido numa venv descartável: sob `offscreen`, um
# clique real muda o texto de um rótulo (`1 passed in 0.02s`).
PLATAFORMA_SEM_TELA: Final = "offscreen"

# **Os dois que os testes de janela precisam para RODAR**, e não os três do grupo. `pyinstaller` é
# do congelamento, que não corre no WSL — exigi-lo aqui faria a janela pular por falta de uma
# ferramenta que ela não usa. A composição dos três é conferida pelo portão de pinos.
MEMBROS_EXIGIDOS: Final[tuple[str, ...]] = ("PySide6", "pytest-qt")


def garante_plataforma_offscreen() -> str:
    """Fixa `QT_QPA_PLATFORM` antes de existir `QApplication`. Devolve a plataforma efetiva.

    **`setdefault` e não atribuição**, para que quem tenha servidor gráfico possa exportar outra
    coisa e **ver** a janela ao depurar. Sobrescrever tiraria essa possibilidade sem ganho.

    **Por que aqui e não no `tests/conftest.py`.** O conftest é insumo de 2903 testes, e a evidência
    verde de todos eles é sobre a versão de agora — mexer nele para uma linha que só o Qt lê
    invalidaria essa evidência por um ganho de conveniência. E esquecer esta chamada num
    módulo Qt futuro **não é silencioso**: o Qt aborta com «could not connect to display», que é
    ruído impossível de ler como sucesso.

    Devolve o valor em vez de `None` para o chamador poder afirmar sobre o que ficou valendo — a
    função que não devolve nada não é verificável sem ler o ambiente por fora.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", PLATAFORMA_SEM_TELA)
    return os.environ["QT_QPA_PLATFORM"]


def sonda_o_ambiente() -> dict[str, str | None]:
    """A versão instalada de cada membro, ou `None`. **O único ponto que toca o ambiente.**"""
    saida: dict[str, str | None] = {}
    for nome in MEMBROS_EXIGIDOS:
        try:
            saida[nome] = metadata.version(nome)
        except metadata.PackageNotFoundError:
            saida[nome] = None
    return saida


def membros_ausentes(instalados: dict[str, str | None]) -> tuple[str, ...]:
    """Quem falta, em ordem. Tupla vazia quando há o que medir.

    ## Devolve QUEM FALTA e não a frase do `skip`, e a divisão é mecânica

    **`tests/skip_policy.py` extrai a mensagem do `skip` por AST, colhendo constantes de string.**
    Uma chamada `pytest.skip(motivo)` com o motivo numa **variável** não entrega literal nenhum ao
    extractor, e o regime declarado ficaria sem emissão — reprovando por *"padrão sem emissão é
    autorização larga que ninguém exerce"*. Medido: foi exactamente o que aconteceu na primeira
    versão desta guarda.

    Então a repartição é: **esta função decide**, o sítio do `skip` **frasea**, com o literal
    visível ao extractor. Uma fonte para a frase — o sítio —, e uma para a decisão — esta.

    E o resultado nomeia **quem falta e nada mais**: enumerar o medido em vez de afirmar uma causa é
    o que separa esta guarda da primeira versão do regime `sem_endereco_externo`, que dizia
    *"máquina sem endereço IPv4"* e disparava com motivo falso onde havia endereço e não havia rota.
    """
    return tuple(sorted(nome for nome, versao in instalados.items() if versao is None))

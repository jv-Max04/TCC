"""O decorrido desde o disparo de um trabalho, sem dependência de interface.

O decorrido congelado de uma corrida não pode sobreviver ao disparo seguinte: o contador ficaria
parado, plausível, e pertenceria a outra execução. Toda escrita da chave de disparo passa pelo funil
`carimba_disparo`, e `tests/test_funil_do_decorrido.py` percorre por AST os arquivos que importam
este módulo e acusa quem escreva a chave por fora.

O depósito é um `Protocol` de duas operações, e `DepositoEmMemoria` o implementa para quem não tem
sessão, como a janela.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

__all__ = [
    "Deposito",
    "DepositoEmMemoria",
    "agora",
    "carimba_disparo",
    "chave_do_congelado",
    "decorrido_da_leitura",
    "instante_utc",
]


class Deposito(Protocol):
    """Onde o carimbo do disparo e o decorrido congelado vivem.

    Duas operações, e não um `MutableMapping`: estas funções só gravam e leem por chave. Um mapa
    exporia `__delitem__`, `__iter__` e `__len__`, e cada um é uma forma de apagar o congelado por
    fora do funil.
    """

    def guarda(self, chave: str, valor: float | None) -> None: ...

    def le(self, chave: str) -> float | None: ...


class DepositoEmMemoria:
    """Um depósito por dicionário, para quem não tem sessão: a janela e os testes diretos."""

    __slots__ = ("_valores",)

    def __init__(self) -> None:
        self._valores: dict[str, float | None] = {}

    def guarda(self, chave: str, valor: float | None) -> None:
        self._valores[chave] = valor

    def le(self, chave: str) -> float | None:
        return self._valores.get(chave)


def agora() -> float:
    """O relógio monotônico, num ponto só, para o teste o substituir.

    Uma interação no teste leva cerca de 0,01 s, e duas leituras seguidas dariam o mesmo texto em
    segundos: sem substituir o relógio, a asserção de que a segunda é maior passaria ou reprovaria
    conforme a máquina.
    """
    return time.monotonic()


def instante_utc() -> datetime:
    """O instante de parede, em UTC, para carimbar o que se exporta.

    `agora` é monotônico e só serve para diferença. Mora aqui pelo mesmo motivo de `agora`: um teste
    recusa leitura de relógio em `desktop/janela/`, e o painel que carimba recebe esta função por
    injeção.
    """
    return datetime.now(UTC)


def chave_do_congelado(chave_do_inicio: str) -> str:
    """O nome da chave que guarda o decorrido congelado, derivado do carimbo do disparo.

    É função porque quem lê e quem descarta precisam do mesmo nome: com dois literais, uma mudança
    no sufixo faria o descarte limpar uma chave que ninguém lê, e o congelado sobreviveria.
    """
    return f"{chave_do_inicio}__ate_a_leitura_do_fim"


def carimba_disparo(deposito: Deposito, chave_do_inicio: str, instante: float | None) -> None:
    """Carimba, ou limpa, o instante do disparo, e descarta o decorrido congelado.

    `decorrido_da_leitura` guarda o decorrido na primeira leitura terminal e lê-o dali em diante.
    Sem o descarte, um disparo novo deixaria na chave congelada o valor da corrida anterior, e a
    tela mostraria um contador parado e plausível que pertence a outra execução. Um contador
    parado convence mais que um que cresce, e por isso este defeito engana mais.

    As duas escritas ficam numa função para que nenhum lugar que carimba o disparo esqueça a
    segunda. O funil sozinho não basta, porque uma escrita direta na chave reintroduziria o defeito;
    `test_todo_carimbo_de_disparo_passa_pelo_funil_de_cada_arquivo` a acusa em cada arquivo que
    importa este módulo. `instante=None` limpa o carimbo e também o congelado, que de outro modo
    ficaria órfão.
    """
    deposito.guarda(chave_do_inicio, instante)
    deposito.guarda(chave_do_congelado(chave_do_inicio), None)


def decorrido_da_leitura(
    deposito: Deposito,
    *,
    chave_do_inicio: str,
    terminou: bool,
    relogio: Callable[[], float] = agora,
) -> tuple[float, bool] | None:
    """Os segundos desde o disparo, e se o valor está congelado.

    Devolve `None` quando não há instante de disparo conhecido, e o chamador não mostra a linha:
    contar a partir do disparo de outro trabalho daria um número plausível sobre a coisa errada.

    Depois do fim o valor congela, para não continuar a crescer ao lado de '100% concluído', onde
    seria lido como a duração do trabalho. Mesmo congelado ele não é a duração: é o intervalo até a
    leitura que viu o fim, um teto, porque o fim só é observado quando alguém consulta e a resposta
    de progresso não traz marca de tempo.

    `relogio` entra por argumento para que cada chamador, e o teste, escolha o seu relógio.
    """
    inicio = deposito.le(chave_do_inicio)
    if inicio is None:
        return None
    chave_congelada = chave_do_congelado(chave_do_inicio)
    if not terminou:
        return relogio() - float(inicio), False
    # Guardado na primeira leitura terminal e lido dali em diante, senão cada leitura daria um
    # número maior.
    guardado = deposito.le(chave_congelada)
    if guardado is None:
        guardado = relogio() - float(inicio)
        deposito.guarda(chave_congelada, guardado)
    return float(guardado), True

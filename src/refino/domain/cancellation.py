"""Parada cooperativa: o vocabulário com que o núcleo obedece a um pedido de cancelamento.

Quem pede o cancelamento é a fronteira de rede, e quem obedece é o núcleo (o executor da grade,
os corretores, o laço por feature). O núcleo não conhece HTTP nem banco, então o que atravessa é
só uma pergunta, 'devo parar?', na forma de `Callable[[], bool]`. Não é um `threading.Event`
porque `Callable` aceita `event.is_set`, uma função que consulta um registro sob trava ou um
`lambda` de teste. O núcleo lê a bandeira e nunca a escreve.

A parada sai por exceção, e não por retorno parcial, para que 'o usuário parou' não se confunda
com 'falhou' nem com 'terminou'. A latência é a do intervalo entre dois pontos de checagem. Com
`n_jobs > 1` o `joblib` executa em processos, que não veem a bandeira do pai: a checagem corre no
despacho e impede o envio de trabalho novo, e o que acontece com o que já está em voo depende de
como o despacho para (ver `CancellableDispatch`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

__all__ = [
    "CancelCheck",
    "CancellableDispatch",
    "ExecutionCancelledError",
    "raise_if_cancelled",
]


type CancelCheck = Callable[[], bool]
"""A pergunta 'devo parar?', que o núcleo faz e a fronteira responde.

Nas assinaturas o padrão é `None`, e não uma função que devolve sempre `False`: sem ninguém a
perguntar, a chamada não paga por um mecanismo que não foi pedido."""


class ExecutionCancelledError(Exception):
    """A execução parou porque alguém pediu, e não porque algo falhou.

    O sufixo `Error` é só a convenção de nomes do projeto (`N818` do `ruff`). O significado está
    no estado para onde ela é traduzida: `cancelado`, nunca `falhou`.

    Herda de `Exception`, e não de `BaseException`, para não escapar dos `except Exception` que
    fazem limpeza no caminho. Por isso quem a distingue tem de apanhá-la antes desses blocos,
    como `api/progresso.py` faz.

    `where` diz qual ponto de checagem disparou. Entra na mensagem e fica no atributo, para que
    os testes não dependam do texto.
    """

    def __init__(self, where: str) -> None:
        super().__init__(f"execução cancelada a pedido, em {where}")
        self.where = where


def raise_if_cancelled(check: CancelCheck | None, where: str) -> None:
    """Levanta `ExecutionCancelledError` se a resposta for sim. Não faz mais nada.

    É uma função, e não um `if` repetido, para que `where` seja obrigatório em cada ponto de
    checagem e a guarda de `None` fique num lugar só. `None` é o caminho normal, sem
    cancelamento.
    """
    if check is not None and check():
        raise ExecutionCancelledError(where)


class CancellableDispatch:
    """Despacho de um `Parallel` que para de produzir quando pedem, e levanta depois.

    O `Parallel` consome o iterável à medida que despacha, então a checagem no gerador corre no
    pai, entre um lote e o seguinte. No `joblib` 1.5.3, levantar ali leva o `Parallel` a `_abort`,
    e o `loky` encerra com `abort_everything` o executor reutilizável, que é um por processo: os
    futuros pendentes de qualquer outro `Parallel` do processo falham com `RuntimeError`, e um
    trabalho que ninguém parou morre. Parando de produzir, o `Parallel` vê o fim do iterável e
    espera os lotes já entregues, sem tocar no executor.

    Uso, nesta ordem:

        despacho = CancellableDispatch(cancelled, "loess: despacho de feature")
        saida = list(Parallel(...)(delayed(f)(i) for i in despacho.over(colunas)))
        despacho.raise_if_stopped()  # antes de qualquer uso de `saida`
        return saida

    `raise_if_stopped` vem antes de a saída ser usada, porque depois de uma parada ela só tem o
    que foi despachado. A decisão é pelo que aconteceu (o gerador parou), e não por uma segunda
    pergunta à bandeira, que pode não responder igual duas vezes.

    O preço é a latência: a parada espera os lotes em voo e, com outro trabalho no mesmo
    executor, também as tarefas dele que estão à frente na fila. O SERRF (`correction/serrf.py`)
    não usa esta classe e ainda levanta no despacho.
    """

    def __init__(self, check: CancelCheck | None, where: str) -> None:
        self._check = check
        self.where = where
        self.stopped = False

    def over[T](self, items: Iterable[T]) -> Iterator[T]:
        """Os itens de `items`, até a bandeira dizer sim. Depois disso, nenhum.

        É gerador, e não lista, porque o `Parallel` o consome à medida que despacha e a bandeira
        é lida entre um lote e o seguinte. Uma lista pronta leria a bandeira toda antes do
        primeiro lote.
        """
        for item in items:
            if self._check is not None and self._check():
                self.stopped = True
                return
            yield item

    def raise_if_stopped(self) -> None:
        """Levanta `ExecutionCancelledError` com o sítio, se o despacho parou antes do fim."""
        if self.stopped:
            raise ExecutionCancelledError(self.where)

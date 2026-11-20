"""Como o `loky` diz que o pool de workers morreu, e a vez no executor que o processo partilha.

`MORTE_DO_POOL` é a família de exceções com que o `loky` avisa que o pool morreu. O LOESS e a
régua fora da amostra a apanham para refazer em série o que ficou por fazer. Fica em `domain/`
porque `correction/` e `metrics/` precisam dela e uma não importa a outra. `correction/serrf.py`
mantém uma cópia própria, porque testes de procedência comparam os bytes daquele arquivo. A
importação é defensiva: numa árvore de `joblib` com outra forma o tuplo fica vazio, `except ()`
nunca casa e a exceção sobe, em vez de se recuar para série sobre uma falha que não é a morte do
pool.

`vez_no_pool` existe porque o executor reutilizável do `loky` é um por processo, e mudar o número
de workers dele enquanto outro trabalho ainda lhe submete lotes trava o processo inteiro. No
`joblib` 1.5.3, `get_reusable_executor` muda o executor com a trava global do `loky` na mão, e o
callback do outro trabalho espera a mesma trava para submeter o lote seguinte. A vez põe a espera
antes de o `Parallel` pedir o executor, onde ela não prende ninguém.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

from refino.domain.cancellation import CancelCheck, raise_if_cancelled

__all__ = ["MORTE_DO_POOL", "vez_no_pool"]

try:  # pragma: no cover - o ramo de exceção só corre numa árvore de joblib com outra forma
    from joblib.externals.loky.process_executor import (
        BrokenProcessPool as _BrokenProcessPool,
    )
    from joblib.externals.loky.process_executor import (
        TerminatedWorkerError as _TerminatedWorkerError,
    )

    MORTE_DO_POOL: tuple[type[BaseException], ...] = (
        _TerminatedWorkerError,
        _BrokenProcessPool,
    )
except Exception:  # pragma: no cover - idem
    MORTE_DO_POOL = ()

# Intervalo entre duas leituras da bandeira de parada durante a espera. Fica bem abaixo dos 2 s
# que a parada tem como critério, e não custa nada: a thread que espera não faz trabalho nenhum
# entre as leituras.
_PASSO_DA_ESPERA_S = 0.1


def _n_efetivo(n_jobs: int) -> int:
    """O número de workers que o `LokyBackend` do `joblib` 1.5.3 pede ao executor por este `n_jobs`.

    Repete a conta de `LokyBackend.effective_n_jobs` com `nesting_level=0`, a do `Parallel` de
    topo, em vez de chamá-la: numa thread que não é a principal, `LokyBackend()` sem
    `nesting_level` devolve 1, e `joblib.effective_n_jobs` calcula pelo backend ativo, que pode não
    ser o `loky`. `-1` e `cpu_count()` dão o mesmo número, e por isso a janela, que pede `-1`, e a
    API, que aceita o número, entram juntas.
    """
    from joblib import cpu_count

    if n_jobs < 0:
        return max(int(cpu_count()) + 1 + n_jobs, 1)
    return n_jobs


class _VezDoPool:
    """Quem usa o executor do `loky`, com quantos workers, e quem espera. Um por processo.

    Regras da vez:
    - um pedido cujo `n` difere do que está em uso espera até sair o último dos que estão dentro,
      e pedidos com o mesmo `n` entram juntos;
    - a fila é por ordem de chegada. Um pedido com o `n` em uso não passa à frente de um que
      espera com outro, senão uma sucessão de trabalhos iguais deixaria o diferente à espera para
      sempre;
    - o pedido parado enquanto espera sai da fila antes de levantar, para não prender o seguinte;
    - a vez é devolvida também quando o corpo levanta. Quem apanha a morte do pool fora do `with`
      segue vivo e refaz em série, e uma vez retida prenderia todo outro `n_jobs` do processo.
    """

    def __init__(self) -> None:
        self._condicao = threading.Condition()
        self._n_em_uso: int | None = None
        self._dentro = 0
        self._fila: deque[object] = deque()

    def _pode_entrar(self, bilhete: object, n: int) -> bool:
        if self._fila[0] is not bilhete:
            return False
        return self._dentro == 0 or self._n_em_uso == n

    @contextmanager
    def vez(self, n: int, cancelled: CancelCheck | None, where: str) -> Iterator[None]:
        bilhete = object()
        with self._condicao:
            self._fila.append(bilhete)
        try:
            while True:
                with self._condicao:
                    if self._pode_entrar(bilhete, n):
                        self._fila.popleft()
                        self._n_em_uso = n
                        self._dentro += 1
                        # O seguinte da fila pode ter o mesmo `n` e entrar também.
                        self._condicao.notify_all()
                        break
                    self._condicao.wait(_PASSO_DA_ESPERA_S)
                # A bandeira é lida fora da trava: um `CancelCheck` que consulte um registro sob a
                # trava dele não pode esperar esta.
                raise_if_cancelled(cancelled, where)
        except BaseException:
            with self._condicao:
                if bilhete in self._fila:
                    self._fila.remove(bilhete)
                self._condicao.notify_all()
            raise
        try:
            yield
        finally:
            with self._condicao:
                self._dentro -= 1
                if self._dentro == 0:
                    self._n_em_uso = None
                self._condicao.notify_all()


_VEZ = _VezDoPool()


@contextmanager
def vez_no_pool(n_jobs: int, cancelled: CancelCheck | None, where: str) -> Iterator[None]:
    """Espera, se for preciso, até o executor do `loky` poder servir `n_jobs` sem mudar por baixo de
    outro trabalho, e o segura até a saída do bloco.

    O bloco é o `Parallel` inteiro, do pedido do executor à última saída: depois disso o trabalho
    já não submete nada e o executor pode mudar. Um trabalho de um núcleo não espera, porque o
    `joblib` o corre em série sem tocar no executor. Parado enquanto espera, levanta
    `ExecutionCancelledError(where)`, com um sítio próprio para quem lê a parada saber que o
    trabalho nunca chegou ao despacho.

    A vez só vê quem a pede: um `Parallel` fora dela, como o do `SerrfCorrector` usado
    diretamente, continua sujeito à trava. O produto constrói o SERRF por
    `correction/serrf_na_vez.py`, que põe o `fit_transform` dentro da vez. Ela não é reentrante:
    uma thread que já está dentro e pede outra vez com outro `n_jobs` espera por si mesma.
    """
    n = _n_efetivo(n_jobs)
    if n == 1:
        yield
        return
    with _VEZ.vez(n, cancelled, where):
        yield

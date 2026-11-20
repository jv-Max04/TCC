"""Memória de correções: o mesmo `fit_transform` não corre duas vezes.

As rotas que corrigem, o `/execucoes` e as prévias do passo a passo fazem a mesma chamada a
`corretor.fit_transform`, e esta memória deixa a segunda reusar a primeira. Devolver o mesmo
objeto a várias rotas é seguro porque `RefinoMatrix` marca o array como não escrevível; quem
afrouxar essa imutabilidade derruba este módulo junto. Fica em `api/`, e não em `correction/`,
para o núcleo não ganhar estado entre chamadas.

Não fecha o caso simultâneo: dois disparos ao mesmo tempo pagam duas correções, porque esperar
pela correção de outro trabalho deixaria a thread surda ao cancelamento. Não alcança a medição do
corretor, cujos reajustes são sobre matrizes diferentes, uma por QC retirado.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from refino.domain.correction import CorrectionResult
    from refino.domain.matrix import RefinoMatrix

__all__ = [
    "LIMITE_DE_CORRECOES_RETIDAS",
    "CorretorIdentificavel",
    "MemoriaDeCorrecoes",
    "chave_da_correcao",
    "digest_da_matriz",
]

# Quantas correções ficam: os três corretores dos dois últimos modos abertos. Uma matriz
# `pos_hilic` corrigida ocupa cerca de 3,13 MiB, então seis somam uns 20 MB. Constante própria,
# mesmo que coincidisse com outra, para as duas não mudarem juntas quando só uma devia mudar.
LIMITE_DE_CORRECOES_RETIDAS: Final[int] = 6


class CorretorIdentificavel(Protocol):
    """Um corretor que diz o que, no construtor dele, muda o valor que ele produz.

    `Protocol`, e não a classe `Corrector`: a fronteira só precisa da forma das implementações.
    """

    name: str

    def identidade(self) -> str: ...


def digest_da_matriz(matriz: RefinoMatrix) -> str:
    """Resume tudo o que, na matriz, alcança o corretor.

    Entram os valores, os nomes das features e, por amostra, nome, grupo, tipo, ordem de injeção e
    lote. `provenance` fica de fora: traz nomes de arquivo, e duas cópias do mesmo CSV com nomes
    diferentes são a mesma matriz. `mode` também, porque não muda o valor da correção.
    """
    resumo = hashlib.sha256()
    resumo.update("\n".join(matriz.feature_names).encode("utf-8"))
    # O separador `|` não aparece em nenhum dos cinco campos; sem ele, `("AB", "C")` e
    # `("A", "BC")` colidiriam.
    resumo.update(
        "\n".join(
            f"{a.name}|{a.group}|{a.sample_type}|{a.injection_order}|{a.batch}"
            for a in matriz.samples
        ).encode("utf-8")
    )
    resumo.update(np.ascontiguousarray(matriz.values).tobytes())
    return resumo.hexdigest()


def chave_da_correcao(
    corretor: CorretorIdentificavel,
    matriz: RefinoMatrix,
    *,
    batches: np.ndarray | None,
    seed: int,
    target_features: frozenset[str] | None,
) -> str:
    """A identidade de uma correção: tudo o que decide o valor que `fit_transform` devolve.

    O que muda o valor da correção tem de entrar na chave, senão dois pedidos diferentes recebem o
    mesmo resultado, com o rótulo certo e o conteúdo errado. A chave é montada dos argumentos que
    de fato viajam para `fit_transform`, e a metade do construtor é conferida por teste.

    Os `batches` entram como vetor, e não como `n_batches`: `_lotes` devolve `None` quando há um
    lote só, e `None` segue outro caminho de código, diferente do vetor de zeros. O marcador de
    `None` é uma palavra, para não colidir com um vetor vazio.
    """
    resumo = hashlib.sha256()
    resumo.update(corretor.identidade().encode("utf-8"))
    resumo.update(digest_da_matriz(matriz).encode("utf-8"))
    if batches is None:
        resumo.update(b"batches=None")
    else:
        resumo.update(b"batches=")
        resumo.update(np.ascontiguousarray(batches).tobytes())
    resumo.update(f"seed={seed}".encode())
    # O alvo entra ordenado: iterar um `frozenset` dependeria do hash do interpretador.
    if target_features is None:
        resumo.update(b"target=None")
    else:
        resumo.update(("target=" + "\n".join(sorted(target_features))).encode("utf-8"))
    return resumo.hexdigest()


class MemoriaDeCorrecoes:
    """Um `OrderedDict` com teto, `Lock` e despejo LRU, instanciável de propósito.

    Instanciável para cada teste de unidade ter a sua. Os testes de rota partilham
    `rotas.CORRECOES`, e cada um afirma que a primeira chamada não acertou, para resíduo de outro
    teste não passar por reuso.

    LRU, com o acerto contando como uso: o que se quer reter é a correção que será pedida outra
    vez. A poda acontece antes de inserir, para não esquecer o que acabou de entrar, e só sucesso
    entra. `guarda` devolve o que despejou, para um teto que nunca dispara ser visível.
    """

    def __init__(self, limite: int = LIMITE_DE_CORRECOES_RETIDAS) -> None:
        if limite < 1:
            raise ValueError(
                f"limite de {limite} correções retidas: um teto de zero é uma memória que nunca "
                f"acerta, e isso é custo puro — se a intenção é desligar o reuso, não a instancie"
            )
        self._limite = limite
        self._trava = threading.Lock()
        self._itens: OrderedDict[str, CorrectionResult] = OrderedDict()
        self.acertos = 0
        self.erros = 0

    def busca(self, chave: str) -> CorrectionResult | None:
        """O resultado guardado sob a chave, ou `None`, contando acertos e erros.

        Os contadores deixam o teste afirmar que a memória acerta; uma que nunca acertasse deixaria
        os testes de fidelidade verdes sem poupar nada.
        """
        with self._trava:
            resultado = self._itens.get(chave)
            if resultado is None:
                self.erros += 1
                return None
            # O acerto vai para o fim, como uma inserção: é o que distingue LRU de FIFO.
            self._itens.move_to_end(chave)
            self.acertos += 1
            return resultado

    def guarda(self, chave: str, resultado: CorrectionResult) -> tuple[str, ...]:
        """Guarda e devolve as chaves despejadas para caber no teto."""
        with self._trava:
            despejadas: list[str] = []
            self._itens.pop(chave, None)
            while len(self._itens) >= self._limite:
                despejadas.append(self._itens.popitem(last=False)[0])
            self._itens[chave] = resultado
            return tuple(despejadas)

    def __len__(self) -> int:
        with self._trava:
            return len(self._itens)

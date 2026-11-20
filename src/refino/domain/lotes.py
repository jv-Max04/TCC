"""Lotes declarados: a partição da corrida em blocos pela ordem de injeção.

Quando o usuário declara `n` lotes num arquivo que não os traz, as amostras são partidas, em
ordem de injeção, em `n` blocos contíguos. Há duas formas de cortar:
- blocos iguais: `n` partes de tamanho quase igual, quando só o número é declarado;
- cortes declarados: fronteiras nas ordens de injeção dadas, cada uma a primeira de um lote. É
  como chega a proposta do diagnóstico.

Blocos iguais não sabem onde está o degrau de nível. No `neg_hilic` ele fica entre as injeções 45
e 56, e dois blocos iguais cortam entre a 34 e a 35, deixando o degrau dentro do segundo lote.

O SERRF e o `batch_loess` corrigem cada lote com os QCs daquele lote, e por isso a contagem de QCs
por lote importa: sete QCs em três lotes dão três, dois e dois. A API e a janela usam estas
funções. O multiverso tem uma partição própria, só com blocos iguais, e
`tests/test_domain_lotes.py` afirma que as duas coincidem.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

__all__ = ["confere_cortes", "lotes_declarados", "qcs_por_lote_declarado"]


def confere_cortes(cortes: Sequence[int], n_lotes: int) -> tuple[int, ...]:
    """Os cortes como tupla de inteiros, ou `ValueError` dizendo o que está errado neles.

    Só a forma é conferida aqui: um corte por fronteira (`n_lotes - 1`), inteiros, em ordem
    estritamente crescente. Se cada corte cai dentro da corrida depende das ordens de injeção, e
    `lotes_declarados` confere isso. Não se supõe que a corrida comece em 1: o leitor aceita
    qualquer inteiro na linha `time`.
    """
    valores = tuple(cortes)
    if any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) for c in valores):
        raise ValueError(f"os cortes de lote são ordens de injeção inteiras; recebi {valores!r}")
    valores = tuple(int(c) for c in valores)
    if len(valores) != n_lotes - 1:
        raise ValueError(
            f"{n_lotes} lote(s) pedem {n_lotes - 1} corte(s), e recebi {len(valores)}: "
            f"{list(valores)}. Cada corte é a primeira injeção de um lote novo."
        )
    if any(b <= a for a, b in pairwise(valores)):
        raise ValueError(
            "os cortes de lote têm de estar em ordem estritamente crescente; recebi "
            f"{list(valores)}"
        )
    return valores


def lotes_declarados(
    ordens_de_injecao: NDArray[np.int64],
    n_lotes: int,
    cortes: Sequence[int] | None = None,
) -> NDArray[np.int64] | None:
    """O lote de cada amostra, na ordem das entradas, ou `None` quando é lote único.

    `None`, e não um vetor de zeros, porque o corretor distingue 'sem declaração de lote' de 'um
    lote declarado' e segue caminhos de código diferentes.

    Com `cortes`, cada corte é a ordem de injeção que inicia um lote, a mesma convenção de
    `diagnosis/batches.py::propose_batches`, de onde os cortes vêm. Corte na primeira injeção ou
    além da última levanta: produziria um lote sem amostra, que o corretor saltaria em silêncio
    por não ter QC.
    """
    if n_lotes <= 1:
        if cortes:
            raise ValueError(f"lote único não tem corte; recebi {list(cortes)}")
        return None
    ordens = np.asarray(ordens_de_injecao)
    if cortes is not None:
        fronteiras = confere_cortes(cortes, n_lotes)
        menor, maior = int(ordens.min()), int(ordens.max())
        fora = [c for c in fronteiras if not menor < c <= maior]
        if fora:
            raise ValueError(
                f"corte(s) {fora} fora do intervalo de ordens de injeção ({menor} a {maior}): "
                "o lote que eles abririam, ou o que deixariam antes do primeiro, ficaria sem "
                "amostra nenhuma"
            )
        return np.searchsorted(np.asarray(fronteiras), ordens, side="right").astype(np.int64)
    lotes = np.empty(len(ordens), dtype=np.int64)
    for indice, bloco in enumerate(np.array_split(np.argsort(ordens), n_lotes)):
        lotes[bloco] = indice
    return lotes


def qcs_por_lote_declarado(
    ordens_de_injecao: NDArray[np.int64],
    qc: NDArray[np.bool_],
    n_lotes: int,
    cortes: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Quantos QCs caem em cada um dos `n_lotes` declarados, na ordem dos lotes.

    Lote sem QC aparece como zero em vez de sumir, porque é o caso que precisa ser visto. Com um
    lote só, ou nenhum declarado, devolve o total.
    """
    mascara = np.asarray(qc, dtype=bool)
    rotulos = lotes_declarados(ordens_de_injecao, n_lotes, cortes)
    if rotulos is None:
        return (int(np.count_nonzero(mascara)),)
    return tuple(int(np.count_nonzero(mascara[rotulos == i])) for i in range(n_lotes))

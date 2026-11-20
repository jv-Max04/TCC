"""Orientação canônica do sinal de um componente.

SVD e PLS definem cada componente a menos de sinal: `(u, v)` e `(-u, -v)` reconstroem a mesma
matriz, e qual dos dois a biblioteca devolve depende de detalhe interno. `sklearn` não garante
orientação. Numa execução isso é inofensivo. O multiverso, porém, compara loadings entre
especificações e conta em quantas uma feature sobe, e sob `scaling="vast"` a matriz sai
apontando para o lado oposto de forma sistemática: a média por feature é negativa depois de
`sum` e `log10`, e o fator `média/desvio` herda o sinal. Sem convenção, a mesma conclusão
descrita com dois sinais apareceria como feature instável.

A convenção: o elemento de maior módulo do loading é positivo, e empate de módulo se resolve
pelo menor índice. O sinal pela soma dos loadings não serve, porque a soma pode ser quase zero
num loading balanceado e a decisão ficaria com o ruído do último bit.

`pca` e `plsda` chamam esta mesma função, para a convenção ser uma só.
"""

import numpy as np

__all__ = ["orient_canonically"]


def orient_canonically(scores: np.ndarray, loadings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fixa o sinal de cada componente pelo elemento de maior módulo do loading.

    Recebe `scores` de forma `(n_samples, n_components)` e `loadings` de forma
    `(n_features, n_components)`, e devolve os dois com o mesmo sinal aplicado por componente.
    Os dois viram juntos porque `scores @ loadings.T` reconstrói a matriz, e virar só um lado
    mudaria a reconstrução.

    Componente cujo loading é todo nulo ou não finito fica como está: não há elemento de maior
    módulo para decidir. A função é idempotente.
    """
    scores = np.asarray(scores, dtype=np.float64)
    loadings = np.asarray(loadings, dtype=np.float64)
    if scores.shape[1] != loadings.shape[1]:
        raise ValueError(
            f"scores tem {scores.shape[1]} componente(s) e loadings tem {loadings.shape[1]}; "
            "a orientação é por componente e exige o mesmo número dos dois lados"
        )

    sinais = np.array([_sinal_do_componente(loadings[:, k]) for k in range(loadings.shape[1])])
    return scores * sinais, loadings * sinais


def _sinal_do_componente(loading: np.ndarray) -> float:
    """`-1.0` quando o elemento de maior módulo é negativo, `1.0` caso contrário.

    `np.argmax` devolve o primeiro máximo, e é isso que resolve o empate pelo menor índice.
    """
    finitos = np.isfinite(loading)
    if not finitos.any():
        return 1.0
    modulos = np.where(finitos, np.abs(loading), -np.inf)
    maximo = int(np.argmax(modulos))
    if modulos[maximo] == 0.0:
        return 1.0
    return -1.0 if loading[maximo] < 0.0 else 1.0

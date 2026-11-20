"""Cache de `_get_param_names` do scikit-learn durante o ajuste das florestas.

O SERRF instancia estimadores do scikit-learn milhões de vezes para ajustar sobre poucos pontos, e
boa parte do tempo vai em `get_params`, que relê a assinatura de `__init__` por
`inspect.signature` a cada chamada. `_get_param_names` é `classmethod` e função pura da classe:
guardar o resultado por classe não muda árvore nenhuma. `check_random_state` também custa, mas
constrói um `RandomState`, e reusar a instância mudaria os sorteios de cada árvore; por isso não é
tocada.

A floresta ajustada tem de ser a mesma, bit a bit, com e sem o cache.
`tests/unit/test_forest_speedup.py` compara as duas predições com a mesma semente e a mesma
entrada.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sklearn.base import BaseEstimator

_ATTRIBUTE = "_get_param_names"


@contextmanager
def cached_param_names(*, enabled: bool = True) -> Iterator[bool]:
    """Instala o cache por classe pela duração do bloco e cede `True` se instalou.

    O valor cedido deixa o teste afirmar que o caminho com cache está ativo. O cache
    guarda tuplas e devolve uma lista nova a cada chamada, para que quem a recebe não
    altere o que está guardado.

    No `finally` volta o objeto `classmethod` original, e não a versão ligada que o
    acesso pelo atributo devolveria: reatribuir a ligada transformaria o `classmethod`
    num atributo comum e quebraria a herança para as subclasses.
    """
    if not enabled:
        yield False
        return

    original = BaseEstimator.__dict__[_ATTRIBUTE]
    underlying = original.__func__
    cache: dict[type, tuple[str, ...]] = {}

    def cached(cls: type) -> list[str]:
        names = cache.get(cls)
        if names is None:
            names = tuple(underlying(cls))
            cache[cls] = names
        return list(names)

    setattr(BaseEstimator, _ATTRIBUTE, classmethod(cached))
    try:
        yield True
    finally:
        setattr(BaseEstimator, _ATTRIBUTE, original)


def is_installed() -> bool:
    """Diz se o cache está instalado. Usado pelo teste que confere que ele está ativo."""
    name: str = BaseEstimator.__dict__[_ATTRIBUTE].__func__.__name__
    return name == "cached"

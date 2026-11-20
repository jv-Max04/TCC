"""Floresta do SERRF que conta o tamanho do nó como o `ranger` conta.

O `min_samples_split` do `scikit-learn` conta as linhas distintas do bootstrap; o `min.node.size`
do `ranger` conta os sorteios, com multiplicidade. Com sete QCs, sortear 7 com reposição dá cerca
de 4,5 linhas distintas, e o corte de 5 trava a raiz em cerca de 41% das árvores do
`RandomForestRegressor`, o que não acontece no `ranger`. Aqui o bootstrap é feito antes e as linhas
entram repetidas na árvore, então o corte de 5 é comparado contra 7.

É a floresta padrão do SERRF (`FLORESTA_PADRAO` em `serrf.py`) e expõe só o que ele usa: `fit`,
`predict` e `estimators_` com `get_depth()`. Consome o gerador na mesma ordem de
`scripts/floresta_ranger_emulada.py`, e `tests/unit/test_serrf_floresta_ranger.py` afirma a
igualdade bit a bit entre os dois.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor

__all__ = ["FlorestaRanger"]


class FlorestaRanger:
    """`n_estimators` árvores, cada uma ajustada sobre um bootstrap com as linhas repetidas."""

    def __init__(
        self,
        *,
        n_estimators: int,
        max_features: str | int | float,
        min_samples_split: int,
        random_state: int,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.estimators_: list[DecisionTreeRegressor] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> FlorestaRanger:
        rng = np.random.default_rng(self.random_state)
        n = x.shape[0]
        self.estimators_ = []
        for _ in range(self.n_estimators):
            # As linhas sorteadas vão repetidas para a árvore: é o que faz a raiz contar 7, como
            # no `ranger`. A ordem de consumo do gerador (os índices, depois a semente da árvore)
            # define as árvores; trocá-la mudaria todas.
            indices = rng.integers(0, n, size=n)
            arvore = DecisionTreeRegressor(
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            arvore.fit(x[indices], y[indices])
            self.estimators_.append(arvore)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("FlorestaRanger.predict antes de fit")
        return np.asarray(np.mean([a.predict(x) for a in self.estimators_], axis=0))

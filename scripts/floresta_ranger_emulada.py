"""Floresta com a semântica de contagem do `ranger`, para MEDIÇÃO — não é produção.

**O defeito que ela existe para isolar.** O
`min_samples_split` do `scikit-learn` conta linhas **distintas** do bootstrap; o
`min.node.size` do `ranger` conta amostras sorteadas **com multiplicidade**. Com sete
QCs, sortear 7 com reposição dá ~4,5 distintas, então o corte de 5 trava a raiz em 41%
das nossas árvores e **nunca** na do `ranger`.

Baixar o corte para 2 foi medido e **refutado**: ganha 42% no RSD dos QCs e não recupera
melhor a verdade, piorando 4,4 vezes mais features. Esta classe faz a coisa
diferente — **o sorteio à mão, com as linhas repetidas** —, de forma que o corte de 5
passe a ser comparado contra 7 e não contra 4. É o conserto FIEL, e o que ele mede é se
o vão vem do defeito ou da regularização.

Interface mínima do `RandomForestRegressor` que `serrf.py` usa: `fit`, `predict` e
`estimators_` com `get_depth()`.
"""

from __future__ import annotations

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class FlorestaRangerEmulada:
    """`n_estimators` árvores, cada uma sobre um bootstrap com as linhas REPETIDAS."""

    def __init__(
        self,
        *,
        n_estimators: int = 500,
        max_features: str | int | float = "sqrt",
        min_samples_split: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.estimators_: list[DecisionTreeRegressor] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> FlorestaRangerEmulada:
        rng = np.random.default_rng(self.random_state)
        n = x.shape[0]
        self.estimators_ = []
        for _ in range(self.n_estimators):
            # O sorteio é NOSSO, e as linhas vão repetidas: é isso que faz
            # `n_node_samples` valer 7 na raiz, como o `ranger` vê.
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
        return np.mean([a.predict(x) for a in self.estimators_], axis=0)

"""O SERRF que o produto constrói: o de `serrf.py`, com a chamada inteira dentro da vez no pool.

O `Parallel` do SERRF pede o executor reutilizável do `loky` com o `n_jobs` dele. Se outro despacho
do mesmo processo ainda submete lotes com outro `n_jobs`, o processo trava. A vez
(`domain/pool.py`) faz o ajuste esperar até poder pedir o executor. Ela cobre o `fit_transform`
inteiro, e não só o `Parallel` das florestas: com o mesmo `n_jobs` efetivo isso não serializa nada;
com outro, quem espera aguarda também o passo 1, as correlações e os passos 10 e 11. A vez não é
reentrante, e nenhum chamador está nela ao chamar: o executor do multiverso corrige antes de
despachar as folhas, e a régua em vários núcleos corre o corretor nos workers com `n_jobs=1`, que
não a pede.

Todo construtor do produto monta esta classe, e não `SerrfCorrector`. A régua fora da amostra e o
jackknife, em `metrics/`, chamam `fit_transform` no corretor que recebem e não podem importar
`correction/`, então a vez tem de vir na classe. `tests/unit/test_serrf_na_vez.py` reprova
`SerrfCorrector` usado como valor em `src/` fora deste arquivo e de `serrf.py`.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from refino.correction.serrf import SerrfCorrector
from refino.domain.cancellation import CancelCheck
from refino.domain.correction import CorrectionResult
from refino.domain.matrix import RefinoMatrix
from refino.domain.pool import vez_no_pool

__all__ = ["SITIO_DA_ESPERA_DO_SERRF", "SerrfNaVez"]

# Sítio informado quando a parada chega durante a espera pela vez: o SERRF não chegou ao passo 1.
SITIO_DA_ESPERA_DO_SERRF: Final = "serrf: espera da vez no pool"


class SerrfNaVez(SerrfCorrector):
    """O `SerrfCorrector`, com o `fit_transform` dentro da vez no pool pelo `n_jobs` dele.

    Mesmo nome, mesma identidade e mesma saída: a chave de reuso de correções
    (`api/reuso.py`) não o distingue do `SerrfCorrector`.
    """

    def fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        cancelled: CancelCheck | None = None,
    ) -> CorrectionResult:
        """O ajuste de `SerrfCorrector.fit_transform`, depois de a vez deixar.

        Com `n_jobs` efetivo 1 a vez não espera, porque o `joblib` corre o ajuste em série sem
        tocar no executor. Parado enquanto espera, levanta `ExecutionCancelledError` com
        `SITIO_DA_ESPERA_DO_SERRF`.
        """
        with vez_no_pool(self.n_jobs, cancelled, SITIO_DA_ESPERA_DO_SERRF):
            return super().fit_transform(
                matrix,
                batches=batches,
                seed=seed,
                target_features=target_features,
                cancelled=cancelled,
            )

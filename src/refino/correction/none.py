"""Corretor `none`: nenhuma correção, registrada como tal.

`none` é opção legítima de correção. No modo `neg_c18` do laboratório, corrigir tem tanta chance
de piorar quanto de melhorar, e a comparação entre métodos precisa dele como ramo igual aos
outros. Devolve uma matriz nova com os mesmos valores, `provenance` estendido e mapa de causas
vazio; a matriz de entrada não é tocada.
"""

import numpy as np

from refino.correction.base import resolve_target_features, select_features
from refino.domain.cancellation import CancelCheck, raise_if_cancelled
from refino.domain.correction import CorrectionResult
from refino.domain.matrix import RefinoMatrix

__all__ = ["NoneCorrector"]


class NoneCorrector:
    """O caminho `none`: nenhuma correção, e o registro disso.

    Satisfaz `Corrector` estruturalmente. `batches` e `seed` estão na assinatura e não
    são usados: o protocolo é o mesmo para os três corretores, para que nenhum
    consumidor precise ramificar por método.
    """

    name = "none"

    def identidade(self) -> str:
        """Identidade de construtor: `none()`, porque não há eixos.

        Não é a string vazia, que colidiria com a de qualquer outro corretor sem eixos.
        """
        return f"{self.name}()"

    def fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        cancelled: CancelCheck | None = None,
    ) -> CorrectionResult:
        """Devolve matriz nova, com valores idênticos e mapa de causas vazio.

        `with_values` recebe uma cópia do array: sem ela a saída compartilharia memória
        com a entrada, e quem escrevesse numa alcançaria a outra.

        Com `target_features` a saída traz só as features pedidas. `cancelled` é lido uma
        vez, na entrada, porque não há laço em que parar.
        """
        del batches, seed  # do protocolo; `none` não tem lote nem sorteio
        raise_if_cancelled(cancelled, "none: entrada")
        requested = resolve_target_features(matrix, target_features)
        subset = select_features(matrix, requested)
        return CorrectionResult(
            matrix=subset.with_values(np.array(subset.values, copy=True), "correction:none"),
            uncorrected={},
            requested=requested,
        )

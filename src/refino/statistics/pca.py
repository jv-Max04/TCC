"""PCA sobre matriz já processada.

Há duas PCAs no projeto. A de `metrics/structure.py` é instrumento de métrica estrutural:
aplica a pré-transformação declarada e autoescalona internamente, porque mede agrupamento de QC
sobre matriz crua. Esta recebe a matriz com normalização, transformação e escalonamento já
aplicados, e repeti-los aqui processaria o dado duas vezes.

Por isso a variância explicada é `s_k^2 / sum(s^2)` sobre a SVD da matriz centrada, que vale
para qualquer entrada. O atalho da outra PCA, `total = n_samples * n_features_used`, só vale
sob autoescalonamento com `ddof=0`; com `scaling` igual a `none`, `pareto`, `range` ou `vast`
ele daria número errado sem levantar. As duas contas concordam sobre entrada autoescalonada e
divergem fora dela, e `tests/unit/test_pca.py` confere as duas metades.
"""

import numpy as np

from refino.domain.matrix import RefinoMatrix
from refino.domain.statistics import PCAResult
from refino.metrics.structure import column_usability, retained_column_digest
from refino.statistics.orientation import orient_canonically

__all__ = ["pca"]


def pca(matrix: RefinoMatrix, n_components: int = 5) -> PCAResult:
    """Componentes principais, com orientação canônica e as contagens de coluna.

    A matriz não é pré-transformada nem autoescalonada. É só centrada por coluna, o mínimo para
    os componentes serem de variância e não de magnitude.

    Colunas de variância zero e colunas não finitas são descartadas, contadas e resumidas num
    digest, como em `structural_metrics`: contagem igual não é conjunto igual, e o multiverso
    compara resultados entre especificações. Coluna de variância zero não contribui direção
    nenhuma.
    """
    valores = np.asarray(matrix.values, dtype=np.float64)
    # O mesmo predicado da PCA estrutural: coluna constante é a de `ptp == 0`. Testar por
    # `sd == 0` falha porque o desvio de uma coluna constante não fecha em zero exato.
    utilizavel, descartadas_constantes, descartadas_nao_finitas = column_usability(valores)

    n_utilizaveis = int(utilizavel.sum())
    if n_utilizaveis == 0:
        raise ValueError(
            f"nenhuma das {valores.shape[1]} features é utilizável: "
            f"{descartadas_nao_finitas} com valor não finito e "
            f"{descartadas_constantes} com variância zero. PCA sobre conjunto vazio "
            "produziria componentes de nada, com aparência de medida"
        )

    retidas = valores[:, utilizavel]
    centradas = retidas - retidas.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centradas, full_matrices=False)

    if s.size < n_components:
        raise ValueError(
            f"pedi {n_components} componentes e a matriz só admite {s.size}: "
            f"{matrix.n_samples} amostras e {n_utilizaveis} features úteis"
        )

    escores, cargas = orient_canonically(
        u[:, :n_components] * s[:n_components], vt[:n_components].T
    )
    return PCAResult(
        scores=escores,
        loadings=cargas,
        explained_variance_ratio=_explained_variance_ratio(s, n_components),
        n_components=n_components,
        n_features_used=n_utilizaveis,
        n_features_dropped_zero_variance=descartadas_constantes,
        n_features_dropped_non_finite=descartadas_nao_finitas,
        column_digest=retained_column_digest(matrix.feature_names, utilizavel),
    )


def _explained_variance_ratio(
    valores_singulares: np.ndarray, n_components: int
) -> tuple[float, ...]:
    """`s_k^2 / sum(s^2)`, a conta geral, válida sob qualquer escalonamento de entrada.

    O denominador soma todos os valores singulares, e não só os `n_components` pedidos: a
    fração é da variância total da matriz, que é o que o `sklearn` devolve. Somar só os retidos
    faria os primeiros componentes explicarem 100% por construção.
    """
    total = float((valores_singulares**2).sum())
    if total == 0.0:
        return tuple(0.0 for _ in range(n_components))
    return tuple(float(valores_singulares[k] ** 2 / total) for k in range(n_components))

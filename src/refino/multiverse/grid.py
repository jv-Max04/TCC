"""A grade de especificações de pipeline que o multiverso executa.

A grade é o produto cartesiano de oito eixos: correção, número de lotes, corte de
RSD, limiar de ausência, imputação, normalização por amostra, transformação e
escalonamento. Com os padrões são 9.600 especificações; `size()` calcula o número e
`composition()` diz de onde ele vem.

Opções implementadas que ficam fora da grade padrão, e por quê:

- imputação `missforest`: custo superlinear no número de features.
- imputação `none`: `PipelineSpec` a recusa combinada com normalização, porque a
  normalização receberia `NaN`.
- imputação `exclude`: muda a forma da matriz, e a grade compara relatórios sobre o
  mesmo conjunto de features.
- imputação `knn_features`: um sexto valor levaria a grade de 9.600 a 11.520.
- transformação `log2`: é `log10` vezes uma constante, e as estatísticas que o
  multiverso consome (teste de hipótese, ranking, RSD) não mudam com fator global.
- escalonamentos `range` e `vast`: invariantes a fator constante, como `auto`; com
  `log2` no eixo ao lado, repetiriam células.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from refino.domain.labels import (
    Correction,
    Imputation,
    RankingFilter,
    RsdVariant,
    SampleNormalization,
    Scaling,
    Transformation,
)
from refino.domain.spec import PipelineSpec

__all__ = ["GridDefinition"]


@dataclass(frozen=True)
class GridDefinition:
    """Os oito eixos da grade; `enumerate()` produz o produto cartesiano deles.

    A ordem dos campos é a da árvore de prefixos: o executor agrupa por
    `(correction, n_batches)`, os dois primeiros, porque a correção é a etapa cara e
    é feita uma vez por grupo.
    """

    corrections: tuple[Correction, ...] = ("none", "serrf", "loess")
    batch_options: tuple[int, ...] = (1,)
    rsd_cutoffs: tuple[float | None, ...] = (None, 0.20, 0.25, 0.30)
    missing_thresholds: tuple[float, ...] = (0.5, 0.8)
    imputations: tuple[Imputation, ...] = (
        "one_fifth_min",
        "col_min",
        "knn",
        "median",
        "mean",
    )
    sample_norms: tuple[SampleNormalization, ...] = (
        "none",
        "sum",
        "pqn_group",
        "median",
        "quantile",
    )
    transformations: tuple[Transformation, ...] = ("none", "log10", "sqrt", "cbrt")
    scalings: tuple[Scaling, ...] = ("none", "mean_center", "auto", "pareto")
    seed: int = 42
    rsd_variant: RsdVariant = "outlier_removed"
    # `ranking_filter` é escalar e não eixo: varrê-lo multiplicaria a grade por oito.
    # Fica em `none` por padrão e fora de `_axis_names`, como `seed`.
    ranking_filter: RankingFilter = "none"
    # Os três campos do segundo passe do filtro também são escalares. `None` na fração
    # mantém a regra por tamanho que a etapa já aplica.
    ranking_keep_fraction: float | None = None
    ranking_filter_2: RankingFilter = "none"
    ranking_keep_fraction_2: float | None = None
    # A amostra de referência da normalização é escalar por outro motivo: o nome de uma
    # amostra pertence ao arquivo, e não há conjunto de opções a varrer. `None` significa
    # que nenhum método de referência corre.
    normalization_reference: str | None = None
    # Os fatores de normalização (por exemplo, o peso seco de cada amostra) também
    # pertencem ao estudo, e são escalares pelo mesmo motivo.
    normalization_factors: tuple[tuple[str, float], ...] | None = None

    # Nomes dos eixos, na ordem da árvore. `size()` e `composition()` derivam desta
    # lista, para que as duas contas não possam divergir.
    _axis_names: tuple[str, ...] = field(
        default=(
            "corrections",
            "batch_options",
            "rsd_cutoffs",
            "missing_thresholds",
            "imputations",
            "sample_norms",
            "transformations",
            "scalings",
        ),
        repr=False,
    )

    def __post_init__(self) -> None:
        """Recusa eixo vazio e a combinação que `PipelineSpec` rejeitaria.

        Um eixo vazio zeraria o produto, e a execução de uma grade vazia termina sem
        resultado e sem erro. `imputation="none"` com normalização não nula é recusada
        aqui porque, de outro modo, `enumerate()` levantaria no meio, depois de já ter
        produzido especificações.
        """
        for nome in self._axis_names:
            if not getattr(self, nome):
                raise ValueError(
                    f"eixo {nome!r} vazio: o produto zeraria a grade, e executor sobre grade "
                    "vazia termina rápido e sem resultado — indistinguível de sucesso"
                )
        if "none" in self.imputations and any(n != "none" for n in self.sample_norms):
            raise ValueError(
                "a grade tem imputation='none' e normalização não nula: o PipelineSpec recusa a "
                "combinação na construção, e a enumeração levantaria no meio. Para varrer "
                "`none`, a grade precisa de sample_norms=('none',)"
            )

    def size(self) -> int:
        """O número de especificações: o produto dos tamanhos dos eixos."""
        total = 1
        for nome in self._axis_names:
            total *= len(getattr(self, nome))
        return total

    def composition(self) -> tuple[tuple[str, int], ...]:
        """`(nome do eixo, número de opções)` por eixo, na ordem da árvore.

        Dois eixos com contagens trocadas dão o mesmo `size()` e grades diferentes; a
        composição distingue.
        """
        return tuple((nome, len(getattr(self, nome))) for nome in self._axis_names)

    def prefix_keys(self) -> tuple[tuple[Correction, int], ...]:
        """Os pares `(correction, n_batches)` distintos, raiz da árvore de prefixos.

        O executor chama o corretor uma vez por par, e não uma vez por especificação.
        """
        return tuple((c, b) for c in self.corrections for b in self.batch_options)

    def enumerate(self) -> Iterator[PipelineSpec]:
        """As especificações, agrupadas por prefixo `(correction, n_batches)`.

        O executor consome o iterador nesta ordem e corrige uma vez por grupo.
        """
        for correcao, lotes in self.prefix_keys():
            for corte in self.rsd_cutoffs:
                for ausente in self.missing_thresholds:
                    for imputacao in self.imputations:
                        for norma in self.sample_norms:
                            for transformacao in self.transformations:
                                for escalonamento in self.scalings:
                                    yield PipelineSpec(
                                        correction=correcao,
                                        n_batches=lotes,
                                        rsd_cutoff=corte,
                                        ranking_filter=self.ranking_filter,
                                        ranking_keep_fraction=self.ranking_keep_fraction,
                                        ranking_filter_2=self.ranking_filter_2,
                                        ranking_keep_fraction_2=self.ranking_keep_fraction_2,
                                        rsd_variant=self.rsd_variant,
                                        missing_threshold=ausente,
                                        imputation=imputacao,
                                        sample_normalization=norma,
                                        normalization_reference=self.normalization_reference,
                                        normalization_factors=self.normalization_factors,
                                        transformation=transformacao,
                                        scaling=escalonamento,
                                        seed=self.seed,
                                    )

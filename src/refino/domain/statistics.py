"""Tipos de resultado das análises estatísticas que o domínio define.

Ficam aqui, e não em `statistics/`, os tipos que atravessam camadas: `statistics/` os produz, o
multiverso os compara e o relatório os serializa. `PLSDAResult` vive em `statistics/plsda.py`.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "AnovaCounts",
    "AnovaResult",
    "DunnComparison",
    "FoldChangeInterval",
    "FoldChangeIntervalUndefined",
    "NormalityDecision",
    "PCAResult",
    "PostHocComparison",
    "PostHocDecision",
    "UnivariateCounts",
    "UnivariateResult",
]

NormalityRoute = Literal["forced", "shapiro", "shapiro_unavailable"]
"""Como `parametric` foi decidido.

São três rotas porque 'Shapiro correu e rejeitou a normalidade' e 'não havia resíduos
suficientes para Shapiro correr' levam os dois a Mann-Whitney, mas a figura tem de declarar qual
foi: 'medi' e 'não medi' são fatos diferentes. `forced` é o teste escolhido pelo pedido.

É produzido por `statistics/univariate.py` e lido por quem desenha, sem passar por arquivo nem
por entrada de usuário. Por isso não tem validador em runtime.
"""


@dataclass(frozen=True)
class NormalityDecision:
    """A decisão de normalidade, com a evidência. É uma para a matriz inteira.

    O Shapiro-Wilk corre uma vez sobre os resíduos de todas as features concatenados, e o
    veredito vale para a tabela toda. Decidir por feature seria outro método, com o `q` de BH
    ajustando p-valores de testes diferentes.

    `shapiro_pvalue` é `None` quando o Shapiro não correu: `parametric` foi forçado, ou não havia
    resíduos suficientes. `n_residuals` é a população em que ele correu. É `None` sob `forced`,
    onde ninguém contou resíduos; `0` significa 'contei e não havia nenhum', que é o que
    `shapiro_unavailable` reporta.
    """

    route: NormalityRoute
    shapiro_pvalue: float | None
    n_residuals: int | None


@dataclass(frozen=True)
class UnivariateCounts:
    """Quantas features ficaram indefinidas em cada coluna.

    A figura desenha `-log10(p)` contra `log2fc`, e nos dois eixos há features sem valor.
    Filtrá-las por `isfinite` na hora de desenhar faria a legenda descrever uma população menor
    que a testada. Com estas contagens a figura declara o que ficou de fora.

    `pvalue_zero` é contado à parte porque não é indefinido. Uma feature constante dentro de cada
    grupo e diferente entre eles dá `statistic = ±inf` e `pvalue = 0.0` em Welch. Esse zero entra
    no ajuste de Benjamini-Hochberg e conta no `n` dele, mas vira `+inf` em `-log10`.

    `qvalue_undefined` é medido, e não copiado de `pvalue_undefined`. Os dois coincidem porque
    `benjamini_hochberg` devolve `nan` onde o p não é finito.
    """

    n_features: int
    log2fc_undefined: int
    pvalue_undefined: int
    pvalue_zero: int
    qvalue_undefined: int


@dataclass(frozen=True)
class PCAResult:
    """Componentes principais de uma matriz já pré-processada.

    `explained_variance_ratio` é uma tupla, com um valor por componente, para não fixar o número
    de componentes no nome dos campos.

    As contagens de coluna e o `column_digest` existem porque contagem igual não é conjunto
    igual: duas células do multiverso podem descartar o mesmo número de colunas e colunas
    diferentes, e a comparação trataria métricas de espaços diferentes como comensuráveis.

    `explained_variance_ratio` é calculada dos valores singulares. `metrics/structure.py` usa
    `n_samples * n_features_used` como variância total, o que só vale porque lá a matriz é
    autoescalonada com `ddof=0`. Aqui o `scaling` pode ser `none`, `pareto`, `range` ou `vast`, e
    o atalho daria um número errado sem levantar.
    """

    scores: np.ndarray
    loadings: np.ndarray
    explained_variance_ratio: tuple[float, ...]
    n_components: int
    n_features_used: int
    n_features_dropped_zero_variance: int
    n_features_dropped_non_finite: int
    column_digest: str


@dataclass(frozen=True)
class UnivariateResult:
    """A tabela do teste univariado, por feature.

    As seis colunas (`feature`, `statistic`, `pvalue`, `qvalue_fdr`, `fold_change`, `log2fc`) são
    campos, e não um `DataFrame`: o núcleo não importa `pandas`, que fica nas bordas de I/O.

    Os rótulos do teste viajam com os números. O `statistic` de Welch e o de Mann-Whitney são
    grandezas diferentes com o mesmo nome, e `parametric` diz qual é.
    """

    feature: tuple[str, ...]
    statistic: np.ndarray
    pvalue: np.ndarray
    qvalue_fdr: np.ndarray
    fold_change: np.ndarray
    log2fc: np.ndarray

    # Como foi medido. Sem estes dois, `statistic` e `pvalue` não são comparáveis com nada.
    parametric: bool
    groups: tuple[str, str]

    # As duas procedências, porque os dois eixos podem vir de matrizes diferentes: o `p` sai da
    # matriz processada, e o fold change pode sair da matriz anterior à transformação, que é onde
    # o MetaboAnalyst o calcula (`fold_change_matrix` em `univariate_test`).
    #
    # São as tuplas de `provenance` das matrizes, e não um rótulo novo. Quem desenha responde 'os
    # dois eixos vêm da mesma matriz?' por comparação. Tupla vazia é legítima: matriz construída
    # à mão em teste não tem procedência.
    pvalue_provenance: tuple[str, ...]
    fold_change_provenance: tuple[str, ...]

    # As contagens de indefinidos e a decisão de normalidade. São obrigatórios, e não
    # `X | None`: `univariate_test` é o único construtor deste tipo e preenche sempre os dois.
    counts: UnivariateCounts
    normality: NormalityDecision

    @property
    def fold_change_from_other_matrix(self) -> bool:
        """Se os dois eixos vieram de matrizes diferentes, o que a figura tem de declarar.

        É derivada, e não gravada, para não haver uma segunda fonte que discorde das duas tuplas
        de procedência.
        """
        return self.pvalue_provenance != self.fold_change_provenance


FoldChangeIntervalUndefined = Literal[
    "non_finite_value",
    "order_statistic_not_positive",
    "median_over_imputed_values",
]
"""Por que o intervalo do fold change de uma feature não existe. Um motivo por feature.

- `non_finite_value`: há célula não finita num dos dois grupos. Acontece com
  `imputation="none"`, quando o corretor deixa não finitos.
- `order_statistic_not_positive`: um valor de ordem do grupo é ≤ 0, e os limites seriam `0` e
  `inf`.
- `median_over_imputed_values`: metade ou mais de um grupo foi imputada, e a largura do
  intervalo ficaria perto de zero.

A ordem da lista é a ordem da verificação, e cada feature recebe o primeiro motivo que a
alcança. Os três formam uma partição, o que permite contá-los por motivo.

Vai na resposta da rota. Quem o valida em runtime é o `pydantic`, ao construir o esquema da
resposta com este mesmo `Literal`.
"""


@dataclass(frozen=True)
class FoldChangeInterval:
    """Intervalo de confiança da razão de medianas, por feature, pelo método de Bonett-Price.

    Fica ao lado de `UnivariateResult`, e não dentro, porque o multiverso consome o resultado de
    `univariate_test` (`sign_consistency`) e este intervalo não entra lá. Os dois descrevem as
    mesmas features, na mesma ordem, quando calculados sobre a mesma matriz, e `provenance`
    permite conferir isso contra `UnivariateResult.fold_change_provenance`.

    As duas escalas são o mesmo intervalo: `lower`/`upper` na razão, e `log2_lower`/`log2_upper`
    no eixo em que o `log2fc` é desenhado. Onde `undefined` tem motivo, as quatro são `nan`,
    nunca `inf`. `undefined` é `None` exatamente onde os limites são finitos.
    """

    feature: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    log2_lower: np.ndarray
    log2_upper: np.ndarray
    undefined: tuple[FoldChangeIntervalUndefined | None, ...]
    confidence: float
    groups: tuple[str, str]
    provenance: tuple[str, ...]


# ---------------------------------------------------------------------------
# ANOVA de N grupos
# ---------------------------------------------------------------------------
#
# É um tipo próprio, e não uma extensão de `UnivariateResult`, porque aquele carrega
# `groups: tuple[str, str]`, `fold_change` e `log2fc`, pareados por construção. Com N grupos não
# existe 'a razão entre grupos', e alargar a tupla quebraria quem lê o volcano, que desempacota
# dois.


AnovaMethod = Literal["anova", "kruskal_wallis"]
"""Qual teste de N grupos produziu `AnovaResult.statistic` e `AnovaResult.pvalue`.

`statistic` carrega o `F` da ANOVA de uma via ou o `H` de Kruskal-Wallis. As duas grandezas têm
distribuições nulas diferentes, `F(k-1, N-k)` e `χ²(k-1)`, e por isso o campo tem nome neutro,
como em `UnivariateResult`. Quem escreve legenda, coluna ou eixo lê `method` antes de escrever
'F'.

É produzido por `statistics/anova.py` e lido por quem desenha, sem passar por arquivo nem por
entrada de usuário. Por isso não tem validador em runtime.
"""


PostHocMethod = Literal["tukey", "fisher", "dunn"]
"""Qual post-hoc correu. Superconjunto próprio de `PostHoc`, que é o que o usuário escolhe.

`PostHoc`, em `domain/labels.py`, é campo de `AnalysisSpec`: os dois do caminho paramétrico,
`tukey` e `fisher`. `dunn` é o post-hoc do caminho não paramétrico e não é escolha: quem corre
Kruskal-Wallis recebe Dunn. Pôr `dunn` em `PostHoc` ofereceria na tela uma opção que combina com
o teste errado.

Um teste afirma a relação de subconjunto, como em `InjectableDefect` contra `DefectType`.
"""


PostHocRoute = Literal["ran", "no_survivors", "nothing_testable"]
"""Se o post-hoc correu e, quando não correu, por quê.

São três estados porque 'nenhuma feature sobreviveu ao FDR' e 'nenhuma feature era testável'
produzem a mesma figura vazia e significam coisas diferentes: a segunda indica que a matriz
chegou degenerada.

- `ran`: houve sobrevivente, e as comparações estão em `AnovaResult.comparisons`.
- `no_survivors`: houve feature com `p` definido, e nenhuma passou o corte.
- `nothing_testable`: nenhuma feature teve `p` definido.

É produzido por `statistics/anova.py` e lido por quem desenha. Não tem validador em runtime.
"""


@dataclass(frozen=True)
class PostHocDecision:
    """Qual post-hoc foi pedido, e se ele correu.

    `method` é preenchido mesmo quando `route` não é `ran`, para a figura poder escrever
    'post-hoc de Tukey: nenhuma feature sobreviveu ao FDR'.

    Quantas sobreviveram e qual foi o corte ficam em `AnovaCounts` e em `AnovaResult.fdr_alpha`,
    e não são repetidos aqui.
    """

    route: PostHocRoute
    method: PostHocMethod
    """Qual post-hoc correu. É `PostHocMethod`, e não `PostHoc`: inclui o `dunn`, que ninguém
    escolhe porque vem com o teste não paramétrico.
    """


@dataclass(frozen=True)
class PostHocComparison:
    """Uma comparação par a par, numa feature que sobreviveu ao FDR.

    `pvalue` é cru, não ajustado. A correção de múltiplos testes corre uma vez, sobre o vetor de
    `p` da ANOVA de todas as features. O post-hoc tem a própria proteção: Tukey controla a taxa
    por família pela distribuição do intervalo estudentizado, e Fisher é o recuo que não a
    controla. Por isso este tipo não tem `qvalue`.

    Também não tem `statistic`: o `t` do Fisher e o intervalo estudentizado do Tukey são
    grandezas diferentes, e num campo só duas células da mesma coluna não seriam comparáveis.
    Ficam o efeito (`difference`) e o teste (`pvalue`); o método está em `PostHocDecision.method`.

    `difference` é a média de `group_a` menos a média de `group_b`, sobre as observações
    utilizáveis daquela feature. O sinal diz qual grupo está acima.
    """

    feature: str
    group_a: str
    group_b: str
    difference: float
    pvalue: float


@dataclass(frozen=True)
class DunnComparison:
    """Uma comparação par a par de Dunn, depois de Kruskal-Wallis.

    É um tipo próprio, e não `PostHocComparison` com um campo a mais, porque
    `mean_rank_difference` e `difference` são grandezas diferentes: uma é a diferença das médias
    dos postos, sobre a ordenação conjunta de todas as observações da feature, e a outra é a
    diferença das médias dos valores. Como `AnovaResult.comparisons` é a união dos dois tipos, o
    mypy recusa ler `.difference` sem estreitar primeiro.

    Aqui há `qvalue_fdr`, e em `PostHocComparison` não. O Dunn é um `z` normal por par, sem
    proteção embutida, e o ajuste usual é Benjamini-Hochberg sobre as `k(k-1)/2` comparações
    daquela feature. São duas famílias de testes: entre features, em `AnovaResult.qvalue_fdr`, e
    dentro de uma feature, neste campo. `pvalue` continua cru.

    `mean_rank_difference` é a média dos postos de `group_a` menos a de `group_b`, sobre as
    observações utilizáveis daquela feature. O sinal diz qual grupo está acima. É o numerador do
    `z` de Dunn.
    """

    feature: str
    group_a: str
    group_b: str
    mean_rank_difference: float
    pvalue: float
    qvalue_fdr: float


@dataclass(frozen=True)
class AnovaCounts:
    """Quantas features ficaram indefinidas, e por quê.

    `statistic_undefined` conta `nan`, e não 'não finito'. `F = +inf` com `p = 0` é o desfecho
    exato de uma feature constante dentro de cada grupo e diferente entre eles. É uma medida, e
    conta em `pvalue_zero`, como em `UnivariateCounts`.

    `constant` e `insufficient_groups` são uma partição de `statistic_undefined`, e um teste
    afirma a soma:
    - `constant`: todos os grupos tinham observações suficientes, e a feature não varia (`0/0`).
    - `insufficient_groups`: algum grupo não chegou ao mínimo de observações utilizáveis nesta
      feature. Não é caso de borda: os zeros de não detecção (15,8% das células no `pos_hilic`)
      o produzem.

    `qvalue_undefined` é medido, e não copiado de `pvalue_undefined`. Os dois coincidem porque
    `benjamini_hochberg` devolve `nan` onde o `p` não é finito.
    """

    n_features: int
    statistic_undefined: int
    pvalue_undefined: int
    pvalue_zero: int
    qvalue_undefined: int
    constant: int
    insufficient_groups: int
    survivors: int


@dataclass(frozen=True)
class AnovaResult:
    """ANOVA de uma via sobre N grupos, com o post-hoc das features que sobreviveram ao FDR.

    Não é um `DataFrame` pelo mesmo motivo de `UnivariateResult`: o núcleo não importa `pandas`.

    Há dois denominadores. `n_per_group` é quantas amostras cada grupo tem na matriz, e é o que a
    figura declara. `n_usable_per_group`, com forma `(n_features, n_groups)`, é o que cada feature
    de fato usou. Uma feature em que algum grupo não chega ao mínimo fica com `nan` e é contada,
    em vez de ter a estatística calculada sobre menos grupos.

    `qvalue_fdr` é Benjamini-Hochberg sobre `pvalue`, o vetor inteiro de todas as features. O
    `pvalue` de cada comparação em `comparisons` é cru.
    """

    feature: tuple[str, ...]
    statistic: np.ndarray
    pvalue: np.ndarray
    qvalue_fdr: np.ndarray

    # ---- a população: do cabeçalho e por feature ----------------------------------
    groups: tuple[str, ...]
    n_per_group: tuple[int, ...]
    n_usable_per_group: np.ndarray

    # ---- como foi medido ----------------------------------------------------------
    method: AnovaMethod
    """Qual teste produziu `statistic` e `pvalue`.

    `anova` põe o `F` de uma via em `statistic`; `kruskal_wallis` põe o `H`. Ver `AnovaMethod`.
    """

    fdr_alpha: float

    min_per_group: int
    """O mínimo de observações utilizáveis por grupo que esta execução exigiu.

    É campo, e não constante deste módulo, porque `statistics/univariate.py` já declara
    `MIN_AMOSTRAS_POR_GRUPO = 2` (com uma observação não há dispersão dentro do grupo). Quem
    calcula passa o valor de lá, e `n_groups_used` é derivável sem o domínio importar aquele
    módulo.
    """

    provenance: tuple[str, ...]
    """A procedência da matriz sobre a qual isto correu.

    É a tupla de `provenance` da matriz, como em `UnivariateResult`. Tupla vazia é legítima:
    matriz construída à mão em teste não tem procedência. É uma só, e não duas como no volcano,
    porque aqui não há razão entre grupos calculada sobre outra matriz.
    """

    counts: AnovaCounts
    post_hoc: PostHocDecision

    comparisons: tuple[PostHocComparison | DunnComparison, ...]
    """As comparações par a par das features que sobreviveram ao FDR.

    O caminho paramétrico devolve `PostHocComparison`, com `difference` na unidade do dado e
    `pvalue` cru. O não paramétrico devolve `DunnComparison`, com `mean_rank_difference` e um
    `qvalue_fdr` da família de comparações daquela feature. `post_hoc.method` diz qual dos dois
    veio, e a união obriga quem lê a estreitar o tipo antes de usar o efeito.
    """

    @property
    def n_groups_used(self) -> np.ndarray:
        """Quantos grupos tinham observações suficientes, por feature.

        É derivada, e não gravada, para não haver uma segunda fonte que discorde de
        `n_usable_per_group`.
        """
        # `asarray` explícito: nos stubs do numpy, `count_nonzero` com `axis` devolve `Any`, e
        # `warn_return_any` recusa devolver `Any` de uma função anotada.
        return np.asarray(np.count_nonzero(self.n_usable_per_group >= self.min_per_group, axis=1))

    @property
    def survived(self) -> np.ndarray:
        """Máscara por feature: `q` definido e abaixo do corte.

        Derivada de `qvalue_fdr` e `fdr_alpha`. `AnovaCounts.survivors` é a contagem desta
        máscara, e um teste afirma que as duas concordam.
        """
        return np.isfinite(self.qvalue_fdr) & (self.qvalue_fdr < self.fdr_alpha)

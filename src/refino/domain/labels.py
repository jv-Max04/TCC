"""Vocabulário do domínio: os `Literal` que nomeiam conjuntos fechados de opções.

O tipo de vocabulário mora aqui e a implementação mora no módulo dela. `QualityReport` e as
outras entidades de `domain` precisam dos rótulos, e `domain` não importa as camadas de fora.
Definir o rótulo nos dois lugares criaria duas listas de valores válidos.

Como o nome e a implementação ficam separados, alguém pode acrescentar um método e esquecer de
estender o `Literal`. `tests/unit/test_labels.py` compara `get_args` de cada tipo com as chaves
do registro de implementações correspondente, nas duas direções.
"""

from typing import Literal, get_args

# --------------------------------------------------------------------------
# Vocabulário de `PipelineSpec`: nomeia coisas implementadas fora daqui
# --------------------------------------------------------------------------

Correction = Literal["none", "serrf", "loess"]
"""Método de correção de sinal. Implementado em `correction/`."""

RsdVariant = Literal["plain", "outlier_removed"]
"""Fórmula do RSD. `outlier_removed` aplica `boxplot_outlier_mask` antes de
`sd/mean`, como a `RSD()` do `tools.R`. Implementado em `metrics/rsd.py`."""

Imputation = Literal[
    "one_fifth_min",
    "knn",
    "knn_features",
    "median",
    "mean",
    "col_min",
    "exclude",
    "missforest",
    "none",
]
"""Estratégia de imputação de ausentes. Implementado em `preprocessing/missing.py`.

`col_min` e `one_fifth_min` medem a mesma coluna e diferem por um fator 5: o `ReplaceMin` do
MetaboAnalyst é o `one_fifth_min`, e o `min` deles é o `col_min`. Ver
`docs/metodos-metaboanalyst.md`.

`knn` e `knn_features` diferem em quem é o vizinho. `knn` procura amostras parecidas, o
`KNN (sample-wise)` do MetaboAnalyst; `knn_features` procura features parecidas, o
`KNN (feature-wise)`. O nome `knn` não pode mudar: está gravado nas execuções já salvas em
`refino.db` e nos arquivos de `--log`."""

SampleNormalization = Literal[
    "none",
    "sum",
    "median",
    "pqn_group",
    "sample_reference",
    "feature_reference",
    "supplied_factors",
    "quantile",
]
"""Normalização por amostra. Implementado em `preprocessing/normalization.py`.

Quatro valores correspondem a métodos do MetaboAnalyst:
- `pqn_group`: o `GroupPQN`, com o grupo fixo nos QCs. O rótulo antigo era `pqn`, e
  `PipelineSpec` ainda o aceita na fronteira.
- `sample_reference`: o `SamplePQN`, a mesma conta com a amostra que o usuário escolhe em
  `normalization_reference`.
- `feature_reference`: o `CompNorm`, normalização por padrão interno. Cada amostra é dividida
  pelo valor que ela tem na feature nomeada, tipicamente um composto adicionado em quantidade
  conhecida.
- `supplied_factors`: o `SpecNorm`, normalização por uma quantidade medida fora do equipamento
  (peso seco, volume de urina, número de células). Os fatores viajam como mapa resolvido e não
  como caminho de arquivo, porque a impressão digital carregaria o caminho e não o conteúdo."""

Transformation = Literal["none", "log10", "log2", "sqrt", "cbrt"]
"""Transformação de escala. Implementado em `preprocessing/normalization.py`.

Os cinco têm os nomes do MetaboAnalyst. O `sqrt` usa a forma estabilizada do `SrNorm`."""

Scaling = Literal["none", "mean_center", "auto", "pareto", "range", "vast"]
"""Escalonamento por feature. Implementado em `preprocessing/normalization.py`.

`mean_center` centra sem dividir, como o MetaboAnalyst oferece; `none` não centra. `vast` não
tem contraparte no MetaboAnalyst."""

RankingFilter = Literal["none", "iqr", "sd", "mad", "rsd", "nrsd", "mean", "median"]
"""Redução de features por posto. Implementado em `preprocessing/filtering.py`.

Cada valor é a estatística por coluna que ordena as features; o corte mantém a fração
superior. `none` salta a etapa e é o padrão.

`rsd` é `sd/mean` e `nrsd` é `mad/median`, medidos sobre todas as amostras. Não confundir com
o `rsd_cutoff` de `filter_by_qc_rsd`, que é outra etapa e mede só os QCs."""

# --------------------------------------------------------------------------
# Vocabulário de `AnalysisSpec`: os parâmetros de análise, e não de pipeline
# --------------------------------------------------------------------------
#
# Ficam num bloco próprio porque a distinção é de contrato. Os de cima descrevem como a matriz é
# processada e vivem em `PipelineSpec`, cuja impressão digital é a chave de cache do multiverso e
# o insumo do log reexecutável. Estes descrevem o que se calcula da matriz já processada: dentro
# de `PipelineSpec`, uma mudança de post-hoc invalidaria uma matriz processada que ela não toca.

PostHoc = Literal["tukey", "fisher"]
"""Post-hoc da ANOVA de N grupos. Implementado em `statistics/anova.py`.

São os dois que o MetaboAnalyst tem, `TukeyHSD` e `FisherLSD`.

`tukey` controla a taxa de erro por família sobre todos os pares, pela distribuição do
intervalo estudentizado. `fisher` é o t par a par com o quadrado médio residual agrupado da
ANOVA inteira: mais potente, e anticonservador acima de três ou quatro grupos. Recalcular a
variância por par dá outro número."""

ClusterMetric = Literal["euclidean", "correlation", "spearman"]
"""Distância do agrupamento hierárquico. Implementado em `statistics/clustering.py`.

`euclidean` é o padrão do MetaboAnalyst, e é obrigatório quando a ligação é `ward`, por
restrição matemática do método."""

ClusterLinkage = Literal["ward", "average", "complete", "single"]
"""Critério de ligação do agrupamento hierárquico. Implementado em `statistics/clustering.py`.

`ward` é o padrão do MetaboAnalyst."""

CorrelationMethod = Literal["pearson", "spearman", "kendall"]
"""Coeficiente do mapa de correlação e da busca por padrão. Implementado em
`statistics/correlacao.py`.

`pearson` é o padrão do MetaboAnalyst. Os três não são intercambiáveis: sobre dado monotônico
não linear, `pearson` e `spearman` divergem."""

ClassifierKind = Literal["random_forest", "svm"]
"""Classificador supervisionado. Implementado em `statistics/classificadores.py`.

Os dois que o MetaboAnalyst oferece no grupo 'Classification & Feature Selection'."""

UnivariateTest = Literal["automatic", "welch", "mann_whitney"]
"""Qual teste o contraste entre dois grupos corre.

`automatic` deixa um Shapiro-Wilk sobre os resíduos de todas as features juntos escolher entre
os dois para a matriz inteira. `welch` e `mann_whitney` forçam um deles, o `parametric=True` e
o `parametric=False` de `univariate_test`. O despacho é
`domain/analysis.PARAMETRIC_BY_UNIVARIATE_TEST`, e `tests/unit/test_labels.py` confere a
paridade das chaves com este `Literal`.

O valor é `automatic` e não `auto` porque `auto` já é um valor de `Scaling`, o
autoescalonamento, e as duas palavras apareceriam na mesma aba com sentidos diferentes."""

DefectType = Literal["gradual_drift", "step_change", "erratic", "clean", "unassessable"]
"""Classificação do defeito de uma matriz. Implementado em `diagnosis/profile.py`.

`unassessable` não é um quinto tipo de defeito: é a ausência de medida. Uma matriz cuja
métrica estrutural é indefinida ou instável não foi avaliada. Sem este valor, `nan < limiar`
é `False`, a matriz passa pelas regras com limiar e cai em `erratic`, que é o caso restante:
uma classificação por acidente aritmético. Por isso a regra que devolve `unassessable` é
testada antes das outras."""

# --------------------------------------------------------------------------
# Vocabulário de medição, fora de `PipelineSpec`
# --------------------------------------------------------------------------

PreTransform = Literal["log10p1", "none"]
"""Pré-transformação aplicada antes das métricas estruturais. Depende de onde se mede:

- `log10p1`, pós-correção: a matriz é de intensidade bruta e precisa de compressão de escala.
- `none`, pós-pipeline: a transformação da própria especificação já fez esse papel. Aplicar
  `log10(x+1)` sobre matriz já transformada é transformação dupla; sobre matriz normalizada
  por soma, com valores muito abaixo de 1, quase não transforma.

Nenhum dos dois erros levanta. Por isso o argumento é obrigatório e sem padrão nas
assinaturas que o recebem."""

MeasurementPoint = Literal["post_correction", "post_pipeline"]
"""Onde o relatório foi medido. Acoplado a `PreTransform` por guarda no
`QualityReport`."""

RsdEstimator = Literal["in_sample", "leave_one_qc_out", "r_cv"]
"""Como o RSD foi estimado. Sem este rótulo, dois números de RSD não são comparáveis:
`in_sample` mede nos mesmos QCs em que o corretor foi ajustado e sai otimista.
`leave_one_qc_out` mede fora da amostra e `r_cv` reproduz a validação cruzada do `tools.R`; os
dois exigem um corretor."""

# --------------------------------------------------------------------------
# Vocabulário de ingestão
# --------------------------------------------------------------------------

Orientation = Literal["features-in-rows", "samples-in-rows"]
"""Orientação do arquivo de entrada, não a interna.

O padrão da CLI é `features-in-rows`, igual aos arquivos do laboratório, que são exportação
do Progenesis QI. A orientação interna é a outra, amostras em linhas, e a transposição
acontece uma vez, na ingestão.

É um vocabulário e não um booleano `transpose` porque o booleano obriga quem chama a saber
qual sentido é `True`. Implementado em `io/readers.py`."""

# --------------------------------------------------------------------------
# Vocabulário de correção
# --------------------------------------------------------------------------

NullDenominatorPolicy = Literal["r_faithful", "uncorrected_feature"]
"""Os dois modos do `batch_loess` diante de denominador nulo.

- `r_faithful`: divide e produz `Inf`/`NaN` onde o R produz. É o padrão, porque as saídas de
  referência do laboratório têm os não finitos e 'consertá-los' quebraria a paridade.
- `uncorrected_feature`: devolve a feature sem correção, marcada.

Implementado em `correction/loess.py`."""

UncorrectedCause = Literal[
    "no_usable_predictors",
    "empty_predictor_intersection",
    "negative_fitted_curve",
    "fit_failed",
    "null_denominator",
]
"""Por que uma feature voltou sem correção.

Cada valor diz algo diferente sobre o dado, e por isso não é um booleano:

- `no_usable_predictors`: SERRF, zero colunas após descartar `NaN` (`tools.R:1130`). A feature
  não tem parceiras correlacionadas. É propriedade da feature, não do corretor.
- `empty_predictor_intersection`: SERRF, a seleção de preditores não tem interseção. Matriz
  pequena, ou feature isolada no espaço de correlação.
- `negative_fitted_curve`: LOESS, mais de um quinto do lote com curva ajustada negativa
  (`tools.R:933`).
- `fit_failed`: LOESS, o ajuste ou a seleção de `span` levantou (`tools.R:953`).
- `null_denominator`: LOESS, curva zero, só no modo `uncorrected_feature`.

Volta desserializado do `metrics_json`, e por isso tem guarda em runtime além do `Literal`."""


class InvalidLabelError(ValueError):
    """Rótulo de vocabulário fora do conjunto declarado.

    Herda de `ValueError` porque é a mesma classe de defeito que `SampleMetadata` levanta para
    `sample_type` inválido, e quem ingere já trata `ValueError` na fronteira.
    """


# Os quatro defeitos que o gerador sintético injeta. Subconjunto próprio de `DefectType`:
# `unassessable` é desfecho da classificação, a ausência de medida, e não há como injetá-lo.
# `tests/unit/test_labels.py` afirma a relação de subconjunto.
InjectableDefect = Literal["gradual_drift", "step_change", "erratic", "clean"]

# Os quatro modos do estudo, como rótulo fechado. O gerador sintético precisa dele porque uma
# matriz 'realista' tem de dizer qual modo imita.
ImitatedMode = Literal["pos_hilic", "pos_c18", "neg_hilic", "neg_c18"]

# Os três da predição de vias (`pathways/`). Chegam pela janela, e por isso têm guarda em
# runtime.
#
# `IonizationMode` poderia ser derivado do modo do estudo (`pos_*` é positivo), mas a derivação
# só vale para os nomes de modo deste laboratório. Outro arquivo pode nomear o modo de outra
# forma.
IonizationMode = Literal["positive", "negative"]

# O que fazer com a feature cuja massa já vem deconvoluída (sufixo `n`): assumir o íon primário
# do modo, ou deixá-la fora. Sem a guarda, um terceiro valor cairia no `else` e se comportaria
# como `exclude`, descartando features com o relatório dizendo que as incluiu.
NeutralMassPolicy = Literal["primary_ion", "exclude"]

# Quantas formas iônicas procurar por composto. `default` é a lista `pos_default` do mummichog,
# com 8 adutos. `broad` é a `generic_positive`, com 25, e aproxima o espaço de busca do
# MetaboAnalystR: com ela a correlação de Spearman das vias contra o MetaboAnalystR é 0,91, e
# com `default`, 0,36.
#
# No modo negativo os dois valores selecionam a mesma lista, porque o mummichog só publica uma
# para essa polaridade. `RelatorioDeVias.adutos_procurados` reporta a lista usada.
AdductSearch = Literal["default", "broad"]

# As recusas do roteiro R das vias (`r_service/vias_entrypoint.R`). São texto lido da saída de
# um processo: o lançador `io/enriquecimento_r.py` valida o `desfecho` que o R devolve contra
# este conjunto, e um valor de fora vira 'resposta malformada' em vez de cair num `else`. O
# vetor `DESFECHOS` do roteiro tem de ser o mesmo conjunto, e
# `tests/test_vias_entrypoint_contrato.py` confere isso.
ViasRecusa = Literal[
    "pedido_invalido",
    "biblioteca_ausente",
    "fundo_invalido",
    "fundo_recusado",
    "lista_recusada",
    "lista_igual_ao_fundo",
    "sem_casamento_na_biblioteca",
    "motor_falhou",
]

# O registro é derivado por `get_args`, e não digitado à mão, para não divergir dos tipos.
VOCABULARY: dict[str, tuple[str, ...]] = {
    "Correction": get_args(Correction),
    "RsdVariant": get_args(RsdVariant),
    "Imputation": get_args(Imputation),
    "SampleNormalization": get_args(SampleNormalization),
    "Transformation": get_args(Transformation),
    "Scaling": get_args(Scaling),
    "RankingFilter": get_args(RankingFilter),
    "DefectType": get_args(DefectType),
    "InjectableDefect": get_args(InjectableDefect),
    "ImitatedMode": get_args(ImitatedMode),
    "PreTransform": get_args(PreTransform),
    "MeasurementPoint": get_args(MeasurementPoint),
    "RsdEstimator": get_args(RsdEstimator),
    "Orientation": get_args(Orientation),
    "NullDenominatorPolicy": get_args(NullDenominatorPolicy),
    "UncorrectedCause": get_args(UncorrectedCause),
    # Os cinco de `AnalysisSpec`: parâmetros de análise, não de pipeline.
    "PostHoc": get_args(PostHoc),
    "ClusterMetric": get_args(ClusterMetric),
    "ClusterLinkage": get_args(ClusterLinkage),
    "CorrelationMethod": get_args(CorrelationMethod),
    "ClassifierKind": get_args(ClassifierKind),
    # O teste do contraste, escolhido na tela.
    "UnivariateTest": get_args(UnivariateTest),
    # Os três da predição de vias (`pathways/analise.py`).
    "IonizationMode": get_args(IonizationMode),
    "NeutralMassPolicy": get_args(NeutralMassPolicy),
    "AdductSearch": get_args(AdductSearch),
    # As recusas do roteiro R das vias.
    "ViasRecusa": get_args(ViasRecusa),
}


def validate_label(kind: str, value: object) -> None:
    """Guarda em runtime para um rótulo de vocabulário.

    Levanta `InvalidLabelError` quando `value` não pertence ao conjunto de `kind`, e `KeyError`
    quando `kind` não é um tipo do vocabulário. O segundo é defeito de programação, não de dado.

    O mypy só prova os pontos de chamada internos; dado lido de arquivo, da CLI ou do
    `metrics_json` não passa por ele. Sem a guarda, um valor inválido cairia num `else` e se
    comportaria como outra política, com o `provenance` registrando o nome errado.
    """
    allowed = VOCABULARY[kind]
    if value not in allowed:
        raise InvalidLabelError(
            f"{kind} {value!r} inválido; esperava um de {list(allowed)}. "
            "Rótulo desconhecido não levanta por si: cai no ramo de outra "
            "política e produz número medido de forma diferente da pedida."
        )

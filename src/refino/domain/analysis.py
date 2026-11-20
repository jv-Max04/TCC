"""`AnalysisSpec`: o que se calcula da matriz processada, e com que parâmetros.

Fica ao lado de `PipelineSpec`, e não dentro dele. `PipelineSpec` descreve como a matriz é
processada, e a impressão digital dele é a chave de cache do multiverso e a chave de reexecução
do log da CLI. Um parâmetro de análise lá dentro invalidaria uma matriz processada que ele não
toca (trocar `tukey` por `fisher` não move uma célula), mudaria o formato do log e seria
oferecido como eixo metodológico sobrescrevível. O preço da separação: o registro de execuções
grava `spec_json` a partir de `PipelineSpec`, então duas execuções que difiram só num parâmetro
daqui ficam indistinguíveis no banco.

Não há `seed` aqui: as análises estocásticas derivam o fluxo delas de `(seed, identificador da
unidade)`, com o `seed` de `PipelineSpec`. Nos padrões, as famílias que custam segundos vêm
ligadas e as que custam minutos vêm desligadas.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from refino.domain.labels import (
    ClassifierKind,
    ClusterLinkage,
    ClusterMetric,
    CorrelationMethod,
    PostHoc,
    UnivariateTest,
    validate_label,
)

__all__ = ["PARAMETRIC_BY_UNIVARIATE_TEST", "AnalysisSpec"]

PARAMETRIC_BY_UNIVARIATE_TEST: dict[UnivariateTest, bool | None] = {
    "automatic": None,
    "welch": True,
    "mann_whitney": False,
}
"""O `parametric` que cada valor de `UnivariateTest` passa a `univariate_test`.

`automatic` é `None`: o Shapiro-Wilk sobre os resíduos de todas as features juntos escolhe entre
os dois testes para a matriz inteira. As outras duas forçam o teste.

É uma tabela, e não um `if`, porque `tests/unit/test_labels.py` a confere contra
`UnivariateTest` nas duas direções. Um valor novo no `Literal` sem entrada aqui levantaria
`KeyError` na rota depois de a correção ter corrido.
"""


@dataclass(frozen=True)
class AnalysisSpec:
    """Quais famílias de análise correm sobre a matriz processada, e com que parâmetros.

    Congelado, como `PipelineSpec`: identifica um trabalho, e o que entra numa chave não pode
    mudar depois.
    """

    # ---- Univariadas de N grupos -------------------------------------------------
    anova: bool = True
    """Corre a ANOVA de uma via sobre todos os grupos da matriz.

    Ligada por padrão: o cálculo é vetorizado sobre as features, e o post-hoc só corre nas que
    sobrevivem ao FDR.

    O conjunto de grupos não é campo deste objeto. Ele é derivado da matriz, que já entra na
    chave de invalidação pelo digest e pelos metadados de amostra."""

    anova_parametric: bool | None = None
    """`True` força ANOVA de uma via, `False` força Kruskal-Wallis, `None` decide por normalidade.

    São três estados porque 'o teste de normalidade decidiu' e 'o usuário forçou' são
    procedências diferentes, e o relatório declara qual foi."""

    post_hoc: PostHoc = "tukey"
    """Qual post-hoc corre nas features que sobrevivem ao FDR.

    O padrão é `tukey`, e não o `fisher` do MetaboAnalyst, de propósito: o estudo do laboratório
    tem quatro grupos, e o `fisher` é anticonservador acima de três."""

    # ---- PLS-DA multiclasse ------------------------------------------------------
    plsda_multiclass: bool = False
    """Corre o PLS-DA sobre todos os grupos de uma vez.

    Desligado por padrão: as permutações custam minutos."""

    plsda_multiclass_components: int = 2
    plsda_multiclass_folds: int = 7

    plsda_multiclass_permutations: int = 1000
    """Quantos sorteios de rótulo o p de permutação usa.

    O padrão é mil. O menor p possível com N sorteios é `1/(N+1)`, então com 100 sorteios um
    `p < 0,01` não é obtenível. Para comparação, o caminho binário usa 200, o `ropls` 20 e o
    MetaboAnalyst 100."""

    # ---- Agrupamento -------------------------------------------------------------
    clustering: bool = False
    """Corre o agrupamento hierárquico das amostras.

    Desligado por padrão: a ligação é quadrática em tempo e memória sobre as features."""

    cluster_metric: ClusterMetric = "euclidean"
    cluster_linkage: ClusterLinkage = "ward"

    kmeans: bool = False
    kmeans_k: int = 3

    som: bool = False
    som_rows: int = 4
    som_cols: int = 4

    # ---- Correlação --------------------------------------------------------------
    correlation: bool = True
    """Corre o mapa de correlação entre features. Ligado: é vetorizado."""

    correlation_method: CorrelationMethod = "pearson"

    pattern_reference: str | None = None
    """A feature ou o perfil contra o qual a busca por padrão correlaciona.

    `None` salta a busca por padrão sem desligar o mapa de correlação. As duas partilham o
    coeficiente, não o alvo."""

    # ---- PERMANOVA ---------------------------------------------------------------
    permanova: bool = True
    """Testa separação entre grupos nos componentes da PCA, por permutação.

    Ligado por padrão: as permutações correm sobre os scores (dezenas de linhas por meia dúzia de
    colunas), não sobre a matriz de features.

    O teste de dispersão corre sempre junto e não tem bandeira própria. O p da PERMANOVA é
    confundido por heterogeneidade de dispersão: sozinho, não separa 'os centroides diferem' de
    'a dispersão difere'."""

    permanova_permutations: int = 999

    # ---- Normalidade por variável e por grupo ------------------------------------
    normality: bool = True
    """Corre o Shapiro-Wilk por variável, dentro de cada grupo de estudo, sobre a matriz processada.

    Ligada por padrão: é vetorizada sobre as features de cada grupo
    (`statistics/normalidade.py`) e custa milissegundos. Desligada, a rota não a calcula e a
    resposta traz `normalidade` a `None`.

    É diagnóstico, não decisão: não escolhe o teste do contraste. Quem escolhe é
    `univariate_test`, logo abaixo, e o modo `automatic` não lê esta tabela.
    """

    # ---- O teste do contraste entre os dois grupos -------------------------------
    univariate_test: UnivariateTest = "automatic"
    """Qual teste o contraste entre os dois grupos declarados corre: 'Automático', 'Welch' ou
    'Mann-Whitney'.

    `automatic` é o padrão: um Shapiro-Wilk sobre os resíduos de todas as features juntos faz uma
    escolha para a matriz inteira. `welch` e `mann_whitney` forçam o teste, o `parametric=True` e
    o `parametric=False` de `univariate_test`.

    A escolha é uma para o contraste, e não por variável: por variável, a coluna `statistic`
    misturaria duas estatísticas e o `q` de BH ajustaria p-valores de testes diferentes.

    Vive aqui, e não em `PipelineSpec`, porque trocar o teste não move uma célula da matriz
    processada.
    """

    # ---- Rede de correlação parcial esparsa --------------------------------------
    dspc: bool = False
    dspc_alpha: float = 0.05

    # ---- Classificadores ---------------------------------------------------------
    classifiers: tuple[ClassifierKind, ...] = ()
    """Quais classificadores correm. Vazio é o padrão: cada um custa minutos.

    É uma tupla, e não uma bandeira por classificador, porque o pesquisador pode querer comparar
    os dois e porque assim o contrato não cresce a cada classificador novo."""

    classifier_folds: int = 5

    classifier_permutations: int = 200
    """Quantos sorteios de rótulo o `p` de permutação do classificador usa.

    O padrão é 200, e não os mil do PLS-DA multiclasse, por causa do custo por sorteio: lá cada
    permutação é um ajuste de `PLSRegression`; aqui são `classifier_folds` ajustes de uma floresta
    de `classifier_trees` árvores, ou de um SVM linear. O preço é a resolução: com o padrão,
    `p = 1/201 = 0,00498` é o mínimo possível, e não uma medida. `ClassifierResult.n_permutations`
    viaja com o número para que o piso não seja lido como resultado.

    O campo entra na chave de invalidação do trabalho de `/execucoes`, por `analise_para_json`:
    duas execuções que difiram só neste número são trabalhos distintos, e a segunda re-executa
    correção, pipeline e relatório.
    """

    classifier_trees: int = 300
    """Quantas árvores a floresta do `random_forest` tem. Não afeta o SVM.

    O valor é declarado, e não herdado do padrão do `scikit-learn`, que já mudou entre versões: um
    padrão implícito faria o resultado depender da versão instalada.

    São 300, e não as 500 do SERRF, por custo: o `p` de permutação corre
    `(classifier_permutations + 1) x classifier_folds` ajustes desta floresta. Como
    `classifier_permutations`, entra na chave de invalidação, e mudá-lo re-executa o trabalho
    inteiro.
    """

    def __post_init__(self) -> None:
        """Valida em runtime os rótulos e as faixas numéricas.

        Estes valores chegam de formulário HTTP e da janela, onde o mypy não alcança. Sem a
        guarda, um `post_hoc` inválido cairia num `else` e se comportaria como o outro, com a
        figura declarando o nome errado.
        """
        validate_label("PostHoc", self.post_hoc)
        validate_label("ClusterMetric", self.cluster_metric)
        validate_label("ClusterLinkage", self.cluster_linkage)
        validate_label("CorrelationMethod", self.correlation_method)
        # O teste do contraste chega da caixa da tela e de `analise_json`. Um valor de fora
        # cairia no `KeyError` de `PARAMETRIC_BY_UNIVARIATE_TEST` depois da correção.
        validate_label("UnivariateTest", self.univariate_test)
        for classificador in self.classifiers:
            validate_label("ClassifierKind", classificador)

        # `ward` exige distância euclidiana: o critério de Ward minimiza a variância
        # intra-grupo, que só é definida sobre ela. O `scipy` recusa a combinação, e recusar
        # aqui dá a mensagem no vocabulário do domínio, antes do cálculo.
        if self.cluster_linkage == "ward" and self.cluster_metric != "euclidean":
            raise ValueError(
                f"ligação `ward` exige distância `euclidean` e recebi {self.cluster_metric!r}. "
                "Não é escolha nossa: o critério de Ward minimiza variância intra-grupo, que só "
                "está definida sobre distância euclidiana. Escolha outra ligação, ou a distância "
                "euclidiana."
            )

        # Dois pedidos do mesmo classificador com a mesma semente dão o mesmo resultado, e a
        # figura teria duas entradas indistinguíveis.
        if len(set(self.classifiers)) != len(self.classifiers):
            raise ValueError(
                f"`classifiers` repete um classificador: {list(self.classifiers)}. Cada um corre "
                "uma vez; dois pedidos iguais produziriam duas entradas indistinguíveis na figura."
            )

        for nome, valor, minimo in (
            ("plsda_multiclass_components", self.plsda_multiclass_components, 1),
            ("plsda_multiclass_folds", self.plsda_multiclass_folds, 2),
            ("plsda_multiclass_permutations", self.plsda_multiclass_permutations, 1),
            ("kmeans_k", self.kmeans_k, 1),
            ("som_rows", self.som_rows, 1),
            ("som_cols", self.som_cols, 1),
            ("permanova_permutations", self.permanova_permutations, 1),
            ("classifier_folds", self.classifier_folds, 2),
            # Os dois do classificador chegam de `analise_json` e das caixas da janela. Sem a
            # guarda, `classifier_trees=0` só falharia dentro do `scikit-learn`, no meio do
            # ajuste, e `classifier_permutations=0` daria `p = 1/1 = 1` sem distribuição nula
            # nenhuma.
            ("classifier_permutations", self.classifier_permutations, 1),
            ("classifier_trees", self.classifier_trees, 1),
        ):
            if valor < minimo:
                raise ValueError(
                    f"`{nome}` é {valor} e o mínimo é {minimo}. Abaixo disso a quantidade não "
                    "está definida — validação cruzada precisa de ao menos duas dobras, e um "
                    "número de sorteios não positivo não produz p nenhum."
                )

        # A faixa do alfa é aberta nos dois extremos: `0` não deixa sobreviver aresta nenhuma e
        # `1` deixa sobreviver todas, e os dois dão uma figura bem formada e sem informação.
        if not 0.0 < self.dspc_alpha < 1.0:
            raise ValueError(
                f"`dspc_alpha` é {self.dspc_alpha} e tem de estar entre 0 e 1, exclusive. "
                "Em 0 nenhuma aresta sobrevive e em 1 todas sobrevivem: os dois dão figura bem "
                "formada e sem informação, que se lê como resultado."
            )

        if self.pattern_reference is not None and not self.pattern_reference.strip():
            raise ValueError(
                "`pattern_reference` é um nome de feature e recebi vazio ou só espaços. Para não "
                "correr a busca por padrão, deixe-o como `None`."
            )

    def pediu_alguma_coisa(self) -> bool:
        """Se alguma família foi pedida.

        Existe para o ponto de chamada não ter de enumerar as bandeiras.
        """
        return bool(
            self.anova
            or self.plsda_multiclass
            or self.clustering
            or self.kmeans
            or self.som
            or self.correlation
            or self.permanova
            or self.normality
            or self.dspc
            or self.classifiers
        )


PADRAO_DA_ANALISE: AnalysisSpec = AnalysisSpec()
"""O contrato com os padrões: as famílias baratas ligadas, as caras desligadas.

É uma constante porque o objeto é congelado e pode ser partilhado, e uma instância só permite
comparar por identidade.
"""


def analise_para_json(analise: AnalysisSpec) -> dict[str, Any]:
    """O contrato inteiro, com as chaves na ordem dos campos.

    É derivado de `fields`, e não escrito à mão, para que um campo novo no dataclass entre
    sozinho. Uma lista manual o deixaria fora da chave de invalidação.
    """
    return {campo.name: getattr(analise, campo.name) for campo in fields(analise)}

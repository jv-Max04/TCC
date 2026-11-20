"""Modelos de entrada e saída da API.

A API recebe corpo de requisição, e nada nele passou pelo mypy. Todo `Literal` que entra por aqui
é validado na fronteira, e não no ponto de uso: um valor inválido que caísse num `else` razoável
produziria um número medido de outra forma, com o `provenance` registrando o nome errado.

As respostas marcam o que é sugestão, como a fronteira de lote proposta. Devolvida sem a marca,
ela seria lida como fato, e a confirmação humana perderia o objeto.
"""

from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, Field, model_validator

from refino.cli import (
    AUTO_APRENDIDA,
    AUTO_CAVEATS,
    AUTO_MEDIDA,
    METHOD_CAVEATS,
    RESSALVA_PELA_VERDADE,
    ROTULO_PELA_VERDADE,
    ROTULO_PELO_RSD,
)
from refino.domain.labels import Correction, PreTransform, RsdEstimator, RsdVariant
from refino.domain.metadata import SampleType
from refino.domain.spec import SUPPORTED_OVERRIDES
from refino.domain.statistics import FoldChangeIntervalUndefined
from refino.multiverse.stability import Verdict
from refino.statistics.normalidade import NormalityUnassessable

__all__ = [
    "COEFICIENTE_DE_TUKEY",
    "DEFINICAO_DE_ABUNDANCIA",
    "DEFINICAO_DE_FALTANTE",
    "INDISPONIVEIS",
    "RESSALVAS",
    "RESSALVA_DO_ESTIMADOR",
    "TETO_DE_CAIXAS",
    "CaixaDaDistribuicao",
    "CaixaPorAmostra",
    "ClasseDaComposicao",
    "ComposicaoPorClasse",
    "CurvaDoLoessResposta",
    "DefinicaoDeAbundancia",
    "DefinicaoDeFaltante",
    "DiagnosticoResposta",
    "EstabilidadeDeFeature",
    "ExecucaoResposta",
    "GrupoComContagem",
    "IntervaloDaBarra",
    "ItemComRessalva",
    "ItemIndisponivel",
    "LimiaresDoVeredito",
    "MedicaoDoCorretorResposta",
    "MultiversoAceito",
    "ParDePcas",
    "ProblemaResposta",
    "ProgressoResposta",
    "RecomendacaoResposta",
    "ResultadoDoMultiverso",
    "ResumoDaPca",
    "ResumoDasDistribuicoes",
    "ResumoDoModeloUmContraResto",
    "ResumoDoPlsda",
    "ResumoDoPlsdaMulticlasse",
    "RsdAntesEDepois",
    "RsdPorFeature",
    "SegundaOpiniao",
    "SugestaoDeLotes",
    "TrabalhoAceito",
    "TrajetoriaDosQcs",
    "ValidacaoResposta",
]


class ItemIndisponivel(BaseModel):
    """Algo que o fluxo não oferece, com o motivo; a ausência silenciosa é o que isto evita."""

    item: str
    motivo: str
    criterio: str


class ItemComRessalva(BaseModel):
    """Algo que o fluxo oferece, com o recorte medido que quem escolhe precisa ver.

    Os textos vêm de `METHOD_CAVEATS`, a mesma fonte que a CLI imprime.
    """

    item: str
    resumo: str
    detalhe: str
    evidencia: str
    portao_inerte: str


# Itens que o fluxo não oferece. Hoje vazia: o `serrf` e a delegação ao `--auto` passaram a
# `RESSALVAS`, oferecidos com ressalva. A tabela fica como lugar de um bloqueio futuro. Os testes
# percorrem a união com `RESSALVAS`, porque `all(...)` sobre tupla vazia passa sem medir nada.
INDISPONIVEIS: tuple[ItemIndisponivel, ...] = ()


# Derivada de `METHOD_CAVEATS` e `AUTO_CAVEATS`, para que ressalva nova na CLI apareça na API e na
# tela sem cópia. O prefixo `--auto ` no `item` separa os rótulos de política dos nomes de método.
RESSALVAS: tuple[ItemComRessalva, ...] = tuple(
    ItemComRessalva(
        item=nome,
        resumo=ressalva.resumo,
        detalhe=ressalva.detalhe,
        evidencia=ressalva.evidencia,
        portao_inerte=ressalva.portao_inerte,
    )
    for nome, ressalva in [
        *sorted(METHOD_CAVEATS.items()),
        *((f"--auto {rotulo}", r) for rotulo, r in sorted(AUTO_CAVEATS.items())),
    ]
)


class SegundaOpiniao(BaseModel):
    """A segunda opinião do Palpite, pela verdade: informa ao lado da primeira e não decide.

    Não tem confiança porque a margem dela mede distância de erro, que é outra quantidade.
    """

    rotulo: str = Field(default=ROTULO_PELA_VERDADE, min_length=1)
    correction_proposta: str
    # O caminho de decisão da árvore, na prosa de `descreve_caminho`, a mesma do `rationale`.
    caminho: str = Field(min_length=1)
    ressalva: str = Field(default=RESSALVA_PELA_VERDADE.resumo, min_length=1)


class RecomendacaoResposta(BaseModel):
    """O ponto de decisão automática, com justificativa obrigatória e eixos sobrescrevíveis."""

    # A ressalva da política viaja na mesma resposta que o palpite: é o número dela que impede ler
    # o palpite como conselho. O texto vem de `AUTO_CAVEATS`, o mesmo que a CLI imprime.
    ressalva_da_politica: str = Field(default=AUTO_CAVEATS[AUTO_APRENDIDA].resumo, min_length=1)
    # O rótulo diz por que critério opina a primeira opinião, que a execução delegada segue. A
    # segunda está desligada e vem `null`; o campo fica para religá-la sem mudar o contrato.
    rotulo_da_politica: str = Field(default=ROTULO_PELO_RSD, min_length=1)
    segunda_opiniao: SegundaOpiniao | None = None
    correction_proposta: str
    confianca: float
    abstem: bool
    # `avaliavel` separa 'margem medida e minúscula' de 'margem que não pôde ser medida': mesma
    # ação, contagens diferentes.
    avaliavel: bool
    rationale: str = Field(min_length=1)
    origem: str
    disponivel_para_execucao: bool
    eixos_sobrescreviveis: list[str] = Field(default_factory=lambda: sorted(SUPPORTED_OVERRIDES))
    indisponiveis: list[ItemIndisponivel] = Field(default_factory=lambda: list(INDISPONIVEIS))
    # A ressalva do método proposto chega aqui porque este é o ponto de decisão da API.
    ressalvas: list[ItemComRessalva] = Field(default_factory=lambda: list(RESSALVAS))


# A ressalva do estimador, num lugar só, viaja com a resposta. Medido dentro da amostra, o LOESS
# melhora por construção: ajusta a curva através dos próprios QCs e a métrica mede nos QCs. Sem o
# rótulo ao lado dos dois números, a melhora seria lida como ordenação de métodos.
#
# A frase sai da API porque quem escolheu o estimador foi a rota. O botão e a aba que ela cita são
# conferidos contra o widget em `tests/test_desktop_figuras.py`, e `tests/test_desktop_qualidade.py`
# conta o negrito e as crases do texto.
RESSALVA_DO_ESTIMADOR: Final[str] = (
    "ATENÇÃO AO ESTIMADOR: estes dois números são RSD **in-sample**, e o LOESS o reduz **por "
    "construção** — ele ajusta a curva através dos próprios QCs e a métrica mede nos QCs. A "
    "melhora abaixo NÃO ordena métodos de correção. A alternativa é o estimador "
    "fora da amostra `leave_one_qc_out` (`leave_one_qc_out_rsd`), que esta rota não serve: o botão "
    "«Medir qual corretor vence», na aba «Correção», mede-o para cada corretor."
)


class RsdAntesEDepois(BaseModel):
    """O RSD dos QCs antes e depois da correção, com o estimador nomeado.

    As duas medianas usam a mesma população: as features com RSD finito nos dois lados.
    """

    correcao: str
    rsd_variant: RsdVariant
    rsd_estimator: RsdEstimator
    estimador_alternativo: RsdEstimator
    mediana_antes: float
    mediana_depois: float
    n_features_comparadas: int
    n_features: int


# O teto de caixas é o do MetaboAnalyst, que desenha 50 features sorteadas, para as duas telas
# serem comparáveis lado a lado.
TETO_DE_CAIXAS: Final[int] = 50

# Constante porque a legenda da figura o cita. É o `whis=1.5` de `matplotlib.cbook.boxplot_stats`,
# a definição original de Tukey.
COEFICIENTE_DE_TUKEY: Final[float] = 1.5


class CaixaDaDistribuicao(BaseModel):
    """A caixa de Tukey de uma feature, com o nome dela e os bigodes de `boxplot_stats`.

    `n_discrepantes` conta os pontos fora dos bigodes, e é `None` quando a coluna não tem valor.
    """

    feature: str
    # Anuláveis: a coluna inteiramente `nan` dá cinco `nan`, que chegam ao cliente como `null`.
    bigode_inferior: float | None
    q1: float | None
    mediana: float | None
    q3: float | None
    bigode_superior: float | None
    n_discrepantes: int | None


class CaixaPorAmostra(BaseModel):
    """A caixa de Tukey de uma amostra, calculada sobre as features."""

    amostra: str
    # Anuláveis pela mesma razão de `CaixaDaDistribuicao`.
    bigode_inferior: float | None
    q1: float | None
    mediana: float | None
    q3: float | None
    bigode_superior: float | None
    n_discrepantes: int | None


class ResumoDasDistribuicoes(BaseModel):
    """As distribuições antes e depois do pré-processamento, prontas para desenhar.

    Tudo é calculado sobre as features comuns aos dois lados; as caixas são um sorteio de até
    `teto_de_caixas` por eixo, e `amostrada` diz se houve sorteio.
    """

    matriz_antes: str
    matriz_depois: str
    caixas_antes: list[CaixaDaDistribuicao]
    caixas_depois: list[CaixaDaDistribuicao]
    caixas_por_amostra_antes: list[CaixaPorAmostra]
    caixas_por_amostra_depois: list[CaixaPorAmostra]
    # Anuláveis: a média de uma feature sem detecção é `nan`. A posição sobrevive, porque as listas
    # indexam as mesmas features.
    medias_por_feature_antes: list[float | None]
    medias_por_feature_depois: list[float | None]
    medias_por_amostra_antes: list[float | None]
    medias_por_amostra_depois: list[float | None]
    n_caixas: int
    n_caixas_por_amostra: int
    teto_de_caixas: int
    n_features_comparadas: int
    n_features_antes: int
    n_features_depois: int
    n_amostras: int
    n_amostras_comparadas: int
    amostrada: bool
    amostrada_por_amostra: bool
    seed: int


class ResumoDoVolcano(BaseModel):
    """Os vetores por feature do volcano, alinhados por posição, e como o contraste foi medido.

    `grupo_a` é o numerador do fold change. O `p` sai da matriz processada e o fold change da
    pré-transformação; `matrizes_diferem` diz se as duas diferem.
    """

    features: list[str]
    log2fc: list[float | None]
    pvalue: list[float | None]
    qvalue_fdr: list[float | None]
    statistic: list[float | None]
    """A estatística do teste: o `t` de Welch, com sinal, quando `parametrico`; senão o `U` de
    Mann-Whitney, uma contagem sem sinal em `[0, n_a * n_b]`.

    Viaja para a predição de vias, que a usa só para eleger a feature representante de cada
    composto empírico. Não substitui o `log2fc`, que sai de outra matriz.
    """

    # ------------------------------------------------------------------
    # O intervalo do fold change
    # ------------------------------------------------------------------

    log2fc_ci_lower: list[float | None]
    log2fc_ci_upper: list[float | None]
    """Os limites do intervalo de Bonett-Price da razão de medianas, na escala do `log2fc`.

    O nível é marginal: vale para cada feature sozinha e é otimista nas escolhidas por `p` ou por
    fold change. Para `R` escolhidas entre `m`, o nível que controla a cobertura falsa é
    `1 - R*alfa/m` (Benjamini e Yekutieli, 2005). O intervalo sai da matriz pré-transformação e o
    `p` da processada, então ele pode excluir o zero com o `q` acima do corte. `None` onde
    `fold_change_ci_undefined` diz por quê, nunca um limite infinito.
    """
    fold_change_ci_lower: list[float | None]
    fold_change_ci_upper: list[float | None]
    """Os mesmos limites na escala da razão, `2 ** log2`, para quem lê não ter de convertê-los."""
    fold_change_ci_undefined: list[FoldChangeIntervalUndefined | None]
    """Por que o intervalo de cada feature não existe, e `None` onde existe.

    O vocabulário é fechado: um motivo novo no núcleo reprova ao montar a resposta.
    """
    nivel_do_intervalo: float
    """O nível dos intervalos, o mesmo que a legenda escreve."""

    grupo_a: str
    """O grupo do NUMERADOR do fold change."""
    grupo_b: str
    """O grupo do DENOMINADOR do fold change."""
    n_grupo_a: int
    n_grupo_b: int
    """As duas populações, que a figura mostra para separar 'não há efeito' de 'não há poder'."""

    parametrico: bool
    """`True` para Welch, `False` para Mann-Whitney. O nome do teste fica com o lado do desenho."""
    rota_da_normalidade: str
    shapiro_pvalue: float | None
    n_residuos: int | None
    """As três partes de `NormalityDecision`. `None` nos dois últimos quer dizer que Shapiro não
    correu, o que é diferente de zero resíduos."""

    alfa_fdr: float
    correcao: str
    """A correção de múltiplos testes, por extenso, como a legenda a imprime."""

    matriz_do_pvalue: str
    matriz_do_fold_change: str
    matrizes_diferem: bool

    n_features_testadas: int
    """A população em que o teste correu. Redundante com `len(features)` de propósito: um vetor
    truncado no caminho não pode passar por população menor."""


class ResumoDaAnova(BaseModel):
    """Os vetores por feature da ANOVA de N grupos, alinhados por posição, e como ela foi medida.

    `statistic` é `F` sob `anova` e `H` sob `kruskal_wallis`: leia `metodo` antes de rotulá-lo.
    """

    features: list[str]
    statistic: list[float | None]
    pvalue: list[float | None]
    qvalue_fdr: list[float | None]

    grupos: list[str]
    n_por_grupo: list[int]
    """O `n` de cada grupo, pareado com `grupos`; o menor deles decide o poder do teste."""

    metodo: str
    """`anova` ou `kruskal_wallis`, o `AnovaMethod` que produziu `statistic` e `pvalue`."""

    post_hoc_metodo: str
    post_hoc_rota: str
    """O `PostHocMethod` pedido e a `PostHocRoute` que aconteceu. São três rotas porque 'nenhuma
    sobreviveu' e 'nenhuma era testável' dão a mesma figura e são fatos opostos."""

    alfa_fdr: float
    correcao: str
    min_por_grupo: int

    matriz: str
    """A matriz de onde o eixo saiu. Uma só: sem razão entre grupos, não há segunda matriz."""

    n_features_testadas: int
    """A população em que o teste correu. Redundante com `len(features)` de propósito, e
    `plots.anova.pontos_da_anova` recusa quando os dois divergem."""


class ResumoDaCorrelacao(BaseModel):
    """Um recorte sorteado da matriz de correlação entre features, com semente declarada.

    A matriz inteira, quadrática no número de features, não viaja; `n_features_totais` é o total.
    """

    features: list[str]
    """Os nomes retidos, na ordem das linhas e das colunas de `valores`."""

    valores: list[list[float | None]]
    """`(k, k)`, simétrica, diagonal 1. `None` onde havia `nan`, que o JSON não representa."""

    metodo: str
    """`pearson`, `spearman` ou `kendall`; quem desenha recusa valor fora do vocabulário."""

    matriz: str
    n_amostras: int
    """Quantas amostras de ESTUDO entraram. QC e `validate` ficam de fora."""

    n_features_totais: int
    """O denominador do rodapé: quantas features a matriz processada tem."""

    n_indefinidas: int
    """Quantas features saíram por não terem coeficiente definido: constantes ou não finitas."""

    amostragem: str
    """A frase que a figura escreve sobre o recorte, montada por quem o aplicou.

    Só a rota sabe se houve sorteio, com que semente e contra que teto; `plots/` não importa
    `refino.api`.
    """


class ResumoDaBuscaPorPadrao(BaseModel):
    """As features ordenadas pela aderência a um perfil de referência por grupo.

    A busca corre sobre todas as features, e `ordem_dos_grupos` diz a que grupo pertence cada
    número do perfil.
    """

    features: list[str]
    """Em ordem decrescente do coeficiente com sinal; a ordem é o resultado."""

    coeficiente: list[float | None]
    """Um por feature, na mesma ordem, com `None` onde havia `nan`."""

    referencia: str
    """O que foi pedido, verbatim: um nome de feature ou os números do perfil."""

    tipo_da_referencia: str
    """`feature` ou `profile`; a figura escreve qual foi."""

    perfil: list[float]
    """O vetor de referência por grupo, na ordem de `ordem_dos_grupos`.

    Contra uma feature, é a média dela em cada grupo.
    """

    ordem_dos_grupos: list[str]
    metodo: str
    matriz: str
    n_amostras: int
    n_features_totais: int
    n_indefinidas: int


class ResumoDoKmeans(BaseModel):
    """A partição das amostras em `k` agrupamentos, ao lado do grupo clínico de cada uma.

    `semente` é a derivada que de fato correu. A inércia mede ajuste e cai com `k`.
    """

    amostras: list[str]
    """Os nomes das amostras, na ordem das linhas da matriz, a mesma de `rotulos` e `grupos`."""

    rotulos: list[int]
    """O agrupamento de cada amostra, em `[0, k)`."""

    grupos: list[str]
    """O grupo clínico por amostra, `[]` quando o estudo não declara grupo.

    `QC` entra como os outros: a família mede sobre a matriz inteira, como o dendrograma ao lado.
    """

    k: int
    inercia: float
    semente: int
    n_iteracoes: int
    """Quantas iterações o melhor arranque levou; diagnóstico, não resultado."""

    matriz: str
    n_amostras: int
    n_features_usadas: int
    n_features_totais: int
    amostragem: str
    """A frase que a figura escreve sobre o recorte; ver `ResumoDaCorrelacao.amostragem`."""


class ResumoDoSom(BaseModel):
    """Em que unidade da grade do mapa auto-organizado cada amostra caiu.

    `unidades` traz o índice plano; `linhas` e `colunas` dão a forma da grade, que o vetor sozinho
    não determina.
    """

    amostras: list[str]
    unidades: list[int]
    """A unidade de cada amostra, em `[0, linhas·colunas)`, na ordem das linhas da matriz."""

    grupos: list[str]
    """O grupo clínico por amostra; ver `ResumoDoKmeans.grupos`."""

    linhas: int
    colunas: int
    epocas: int
    """Quantas passagens sobre as amostras o mapa fez.

    A figura a declara: um mapa com poucas épocas ainda não ordenou, e parece um mapa ordenado
    sobre dado sem estrutura.
    """

    semente: int
    erro_de_quantizacao: float
    """A distância média de cada amostra à unidade vencedora, na escala do z-score.

    Mede ajuste e não qualidade: cai com o número de unidades por construção.
    """

    matriz: str
    n_amostras: int
    n_features_usadas: int
    n_features_totais: int
    amostragem: str


class ArestaDaRede(BaseModel):
    """Uma aresta da rede de correlação parcial, com a força e a confiança, ambas obrigatórias."""

    feature_a: str
    feature_b: str

    correlacao_parcial: float
    """`-T_ij / √(Θ̂_ii Θ̂_jj)`, com o numerador desenviesado.

    Positiva é associação direta condicionada a todas as outras features: duas features movidas
    por uma terceira não têm aresta, e por isso a rede é mais esparsa que o mapa de correlação.
    """

    estatistica: float
    """O `z` assintótico de Jankova e van de Geer; a legenda diz qual teste o produziu."""

    pvalue: float
    qvalue_fdr: float
    """Benjamini-Hochberg sobre os pares testados. Não é o corte: ver `ResumoDaRede.alfa`."""


class ResumoDaRede(BaseModel):
    """A rede de correlação parcial esparsa desenviesada: as arestas que passaram o corte.

    `arestas` vazia é resultado, nenhum par passou; o objeto ausente quer dizer que não correu.
    """

    features: list[str]
    """As features que entraram na rede, os vértices; a figura desenha também as isoladas."""

    arestas: list[ArestaDaRede]
    n_arestas: int
    n_pares_testados: int
    """`k(k-1)/2`, o denominador que torna legível a contagem de arestas."""

    alfa: float
    """O corte, aplicado ao `p` cru e não ao `q`.

    Quem publicar `n_arestas` publica uma contagem a `p` cru sobre `n_pares_testados` testes. O `q`
    de cada aresta viaja ao lado para quem quiser cortar mais duro.
    """

    penalidade: float
    """O `λ` que a validação cruzada escolheu para o ajuste esparso.

    É ele que governa a esparsidade; sem ele, duas redes não se comparam.
    """

    n_amostras_usadas: int
    """As amostras de estudo. Os QCs ficam de fora: são réplicas da mesma mistura, e a covariância
    deles, quase singular, produziria arestas enormes onde não há nada."""

    n_features_usadas: int
    n_features_totais: int
    n_features_descartadas_constantes: int
    n_features_descartadas_nao_finitas: int

    metodo: str
    """O estimador e a correção, por extenso, como a execução os fez."""

    amostragem: str
    """Se houve sorteio de features, e sobre quantas. A rede é o ajuste mais caro da aba e o recorte
    é real; sem a frase, a contagem de arestas pareceria ser sobre a matriz inteira."""


class ResumoDoClassificador(BaseModel):
    """O desempenho fora da amostra de um classificador, sempre com o `p` de permutação.

    A figura destaca a acurácia balanceada: sob desbalanço, a simples premia quem não classifica.
    """

    kind: str
    """`random_forest` ou `svm`, o rótulo do domínio que a rota despacha."""

    nome: str
    """`Random Forest` ou `SVM`, o nome que a figura imprime, vindo de quem calcula."""

    grupos: list[str]
    """As classes, em ordem alfabética. Alinhada com `contagens`, `recall` e `confusao`."""

    contagens: list[int]
    """Quantas amostras de estudo em cada classe. O desbalanço é dado, e explica a distância
    entre as duas acurácias."""

    recall: list[float]
    """O recall de cada classe, fora da amostra; a média deles é `balanced_accuracy`.

    O agregado não separa 'acerta tudo mal' de 'acerta a classe grande e erra as pequenas'.
    """

    balanced_accuracy: float
    accuracy: float
    macro_f1: float
    permutation_pvalue: float
    """Obrigatório: não há desempenho sem o `p` de permutação ao lado."""

    chance_level: float
    """`1/K`, calculado no núcleo: o valor da acurácia balanceada quando a predição independe do
    rótulo."""

    confusao: list[list[int]]
    """`confusao[i][j]` = amostras da classe `i` preditas como `j`. Contagens, nunca frações."""

    importancias: list[float]
    """Uma importância por feature, alinhada com `features`. A da floresta (impureza) e a do SVM
    (peso linear) não se comparam; `importance_kind` diz qual é."""

    importance_kind: str
    features: list[str]
    cv_folds: int
    n_permutations: int
    semente: int
    matriz: str
    n_amostras: int
    n_features_usadas: int
    n_features_totais: int
    amostragem: str
    """A frase que a figura escreve sobre o recorte; ver `ResumoDaCorrelacao.amostragem`."""


class ResumoDoPlsda(BaseModel):
    """Os escores, cargas e VIP do PLS-DA, com `q2` e `p` de permutação obrigatórios.

    `grupo_positivo` é a classe `y = 1`. `vip` traz `None` onde o VIP não é calculável.
    """

    q2: float
    permutation_pvalue: float
    """Sem valor padrão, como `q2`: não existe corpo com escores e sem a validação."""

    r2y: float
    """O ajuste dentro da amostra, ao lado do `Q²`: a distância entre os dois indica sobreajuste."""

    n_permutations: int
    cv_folds: int
    n_components: int
    seed: int
    """O regime em que o par foi medido. Com `n` permutações, o menor `p` possível é `1/(n + 1)`;
    sem `cv_folds`, dois `Q²` de regimes diferentes pareceriam a mesma quantidade."""

    scores_componente_1: list[float | None]
    scores_componente_2: list[float | None]
    scores_cv_componente_1: list[float | None]
    scores_cv_componente_2: list[float | None]
    """Os escores de validação cruzada, que o gráfico de escores desenha.

    Cada amostra é projetada pelo modelo da dobra que não a viu, girado por Procrustes para o
    referencial da calibração. Os de calibração separam os grupos mesmo sobre rótulos sem
    informação quando há muito mais features que amostras; por isso estes são obrigatórios.
    """

    grupos_por_amostra: list[str]
    """O grupo de cada amostra do contraste, na ordem das linhas da matriz processada. A rota
    confere o comprimento contra os escores, porque os vetores por amostra só valem alinhados."""

    grupo_positivo: str
    grupo_negativo: str
    n_positivo: int
    n_negativo: int
    """As duas populações do contraste, para separar 'não há separação' de 'não há amostra'."""

    features: list[str]
    loadings_componente_1: list[float | None]
    loadings_componente_2: list[float | None]
    vip: list[float | None]
    n_features: int
    """A população em que o modelo correu. Redundante com `len(features)` de propósito: um vetor
    truncado no caminho não pode passar por população menor."""


class ResumoDoModeloUmContraResto(BaseModel):
    """Um modelo um-contra-resto do PLS-DA multiclasse, com a validação obrigatória ao lado.

    `vip` traz `None` onde a importância não é calculável, e `n_vip_indefinido` as conta.
    """

    q2: float
    permutation_pvalue: float
    """Sem valor padrão, como `q2`: não existe corpo com escores e sem a validação."""

    classe: str
    """A classe positiva deste modelo; as outras são o resto. Sem ela, o `Q²` não diz que pergunta
    respondeu."""

    r2y: float
    n_positivo: int
    n_negativo: int
    """As duas populações do contraste; o resto é a soma das outras classes."""

    scores_componente_1: list[float | None]
    scores_componente_2: list[float | None]
    loadings_componente_1: list[float | None]
    loadings_componente_2: list[float | None]
    vip: list[float | None]
    n_vip_indefinido: int


class ResumoDoPlsdaMulticlasse(BaseModel):
    """Os K modelos um-contra-resto e o regime em que foram medidos, sem agregado.

    `q2` e `permutation_pvalue` repetem os dos modelos, e a construção recusa se divergirem.
    """

    q2: list[float]
    permutation_pvalue: list[float]
    """Obrigatórios, alinhados com `classes`. Não há agregado: uma média de K `Q²` apaga a amplitude
    entre modelos, e combinar K p-valores exige uma regra que muda a pergunta respondida."""

    classes: list[str]
    modelos: list[ResumoDoModeloUmContraResto]
    n_modelos: int
    """A população de modelos, redundante com `len(modelos)` de propósito."""

    n_permutations: int
    cv_folds: int
    n_components: int
    seed: int
    """O regime dos K pares. Com `n` permutações, o menor `p` possível é `1/(n + 1)`."""

    grupos_por_amostra: list[str]
    features: list[str]
    n_features: int
    """Os eixos partilhados pelos K modelos, que tornam os K conjuntos de escores comparáveis."""

    @model_validator(mode="after")
    def _o_par_alinha_com_os_modelos(self) -> "ResumoDoPlsdaMulticlasse":
        """Recusa as listas e a contagem que não descrevem os mesmos K modelos.

        Sem isto, `q2[2]` poderia responder por outra classe que não `classes[2]`.
        """
        tamanhos = {
            "q2": len(self.q2),
            "permutation_pvalue": len(self.permutation_pvalue),
            "classes": len(self.classes),
            "modelos": len(self.modelos),
            "n_modelos": self.n_modelos,
        }
        if len(set(tamanhos.values())) != 1:
            raise ValueError(
                f"o resumo multiclasse desalinhou: {tamanhos}. As quatro listas e a contagem "
                "descrevem os mesmos K modelos, e um desalinhamento troca a classe de cada Q²"
            )
        for indice, modelo in enumerate(self.modelos):
            if (modelo.q2, modelo.permutation_pvalue, modelo.classe) != (
                self.q2[indice],
                self.permutation_pvalue[indice],
                self.classes[indice],
            ):
                raise ValueError(
                    f"o resumo e o modelo {indice} discordam: resumo diz "
                    f"({self.q2[indice]!r}, {self.permutation_pvalue[indice]!r}, "
                    f"{self.classes[indice]!r}) e o modelo diz ({modelo.q2!r}, "
                    f"{modelo.permutation_pvalue!r}, {modelo.classe!r}). As listas do topo repetem "
                    "os modelos para que a validação do PLS-DA alcance quem lê só o resumo, e a "
                    "repetição "
                    "só é legítima enquanto for verificada"
                )
        return self


class ParDePcas(BaseModel):
    """As duas PCAs, antes e depois da correção, do painel que `tools.R:1527-1600` desenha.

    Cada lado traz o digest das colunas retidas, porque contagem igual não é conjunto igual. O
    sinal de cada componente é arbitrário e não se compara entre os painéis.
    """

    matriz_antes: str
    matriz_depois: str
    """O estágio do pipeline de cada painel, 'ingerida' e 'corrigida', como a figura o imprime."""

    pretransformacao: PreTransform
    n_componentes: int
    """Os rótulos do par, que tornam os números comparáveis.

    A pré-transformação é `none` porque `tools.R:1577` faz `prcomp(t(...), scale.=TRUE)` sobre a
    matriz crua; a do diagnóstico é `log10p1`, e as duas razões de dispersão não são o mesmo
    número. `n_componentes` entra porque a razão é a distância média ao centroide sobre todos eles.
    """

    corretor: Correction
    """O corretor que produziu a matriz do lado depois, que o painel direito escreve no eixo x.

    Reproduz `tools.R:1597`, que põe ali o nome do corretor. É do par: a matriz ingerida não passou
    por corretor, e `none` afirmaria que passou. É o corretor que correu, não o vencedor das barras.
    """

    scores_pc1_antes: list[float | None]
    scores_pc2_antes: list[float | None]
    tipos_de_amostra_antes: list[SampleType]
    variancia_explicada_antes: list[float]
    qc_dispersion_ratio_antes: float | None
    n_features_usadas_antes: int
    n_features_totais_antes: int
    digest_das_colunas_retidas_antes: str

    scores_pc1_depois: list[float | None]
    scores_pc2_depois: list[float | None]
    tipos_de_amostra_depois: list[SampleType]
    """Um vetor de tipos por lado: que nenhuma etapa remove amostras é observação de hoje, não
    garantia. `SampleType` e não `str`, para que um tipo inválido não esconda o QC na figura."""

    variancia_explicada_depois: list[float]
    qc_dispersion_ratio_depois: float | None
    n_features_usadas_depois: int
    n_features_totais_depois: int
    digest_das_colunas_retidas_depois: str


class RsdPorFeature(BaseModel):
    """Os RSD por feature dos QCs e das amostras, só os finitos, cada lado com a sua contagem.

    As duas populações diferem, e os vetores não se alinham por índice. O estimador é dentro da
    amostra, e a figura não ordena corretores.
    """

    matriz: str
    """O estágio do pipeline: 'corrigida' na execução, para que o resto do pipeline não aperte os
    QCs em nome do corretor; 'ingerida' quando a carga vem do diagnóstico, antes de corrigir."""

    corretor: Correction | None
    """O corretor que produziu a matriz medida, ou `None` para a matriz ingerida.

    `None` e não `none`: `none` é corretor legítimo e afirmaria que a matriz crua foi corrigida.
    """

    rsd_variant: RsdVariant
    rsd_estimator: RsdEstimator
    """Os dois rótulos sem os quais o número não se compara: o mesmo arquivo dá RSD muito
    diferente conforme o estimador, e o ranking entre métodos inverte."""

    rsd_dos_qcs: list[float]
    rsd_das_amostras: list[float]
    """Os dois vetores, só com os valores finitos e sem alinhamento entre si; o comprimento de
    cada um é a contagem do seu lado."""

    n_features: int
    """O total de features da matriz, o denominador de 'N de M features'."""

    n_features_finitas_dos_qcs: int
    n_features_finitas_das_amostras: int
    """As features com RSD definido em cada lado; as duas diferem, e nenhuma serve de proxy da
    outra. A contagem não separa as duas causas de indefinido: QCs todos zero, ou QCs não nulos
    apagados pela remoção de outliers."""

    n_injecoes_dos_qcs: int
    n_injecoes_das_amostras: int
    """Quantas injeções entraram em cada lado, para separar 'não há QC' de 'nenhuma feature
    definida'. As amostras são `sample_mask`, que não é o complemento de `qc_mask`."""


# As duas definições que a composição por classe usa, como vocabulário fechado. A figura traduz o
# token para a frase da imagem, em `plots/composicao.py`; um membro novo reprova no mapa da figura
# em vez de cair num `else` razoável.
DefinicaoDeFaltante = Literal["zero_ou_nao_finita"]
DefinicaoDeAbundancia = Literal["mediana_do_log10_das_detectadas"]

DEFINICAO_DE_FALTANTE: Final[DefinicaoDeFaltante] = "zero_ou_nao_finita"
DEFINICAO_DE_ABUNDANCIA: Final[DefinicaoDeAbundancia] = "mediana_do_log10_das_detectadas"


class ClasseDaComposicao(BaseModel):
    """Uma classe clínica e o que a matriz ingerida tem dela: ausências e abundância típica.

    As três contagens de ausência viajam para que se veja qual delas o número é; coincidem nos
    arquivos de referência, mas não depois de um corretor.
    """

    grupo: str
    """O rótulo clínico, como `RefinoMatrix.groups` o devolve. Não é vocabulário fechado: cada
    estudo traz os seus."""

    n_injecoes: int
    """Quantas injeções desta classe, o `n` que a figura escreve sob cada barra."""

    n_celulas: int
    """`n_injecoes * n_features`, o denominador do percentual de faltantes desta classe, enviado
    junto do numerador para a divisão não depender de outro objeto."""

    n_zeros: int
    n_nao_finitas: int
    n_faltantes: int
    """As três candidatas a faltante. `n_faltantes` é a união, a mesma de
    `preprocessing.missing.is_missing`, e é a que a figura desenha."""

    n_detectadas: int
    """Células finitas e estritamente positivas, a população em que a abundância é medida.

    Não é `n_celulas - n_faltantes`: a diferença é a célula finita negativa.
    """

    log10_mediana_das_detectadas: float | None
    """A abundância típica, `median(log10(células detectadas))`, ou `None` sem nenhuma detectada.

    A mediana do log e não a média: a média é governada pela cauda e deixa as classes com barras
    quase iguais. É a redução de `diagnosis/profile.py::_intensidade`, aplicada por classe.
    """


class ComposicaoPorClasse(BaseModel):
    """A composição por classe clínica da matriz ingerida: quanto falta e quão abundante é.

    É a ingerida porque o corretor cria zeros e não finitos; a população é `sample_mask`, sem QC.
    """

    matriz: str
    """O estágio do pipeline, 'ingerida'."""

    populacao: SampleType
    """Qual máscara de `SampleType` define a população, `sample`.

    O tipo recusa valor fora do vocabulário, não o tipo errado: quem garante que a agregação usou
    `sample_mask` é o teste da agregação.
    """

    n_injecoes_fora_da_populacao: int
    """Quantas injeções ficaram de fora, QCs e validação somados, para a figura explicar por que
    o total do arquivo não é a soma das classes."""

    n_features: int
    """O total de features da matriz, que multiplica `n_injecoes` em `n_celulas`."""

    definicao_de_faltante: DefinicaoDeFaltante
    definicao_de_abundancia: DefinicaoDeAbundancia
    """Os dois tokens que a figura traduz para a definição escrita na imagem. Obrigatórios: sem a
    definição, 'faltante' poderia ser qualquer uma das três contagens."""

    classes: list[ClasseDaComposicao]
    """Uma entrada por classe, em ordem alfabética do rótulo, para a ordem das barras não depender
    da leitura do arquivo."""


class NormalidadeDoGrupo(BaseModel):
    """O Shapiro-Wilk de cada variável num grupo de estudo, e o resumo pronto do grupo.

    Os vetores seguem `ResumoDaNormalidade.features`, com `None` onde o número não existe; variável
    com valor imputado no grupo fica sem `qvalue`, fora da família do BH.
    """

    grupo: str
    n_amostras: int
    w: list[float | None]
    pvalue: list[float | None]
    qvalue: list[float | None]
    n_valores: list[int]
    nao_avaliavel: list[NormalityUnassessable | None]
    valor_imputado: list[bool]
    n_variaveis: int
    n_menos_de_tres: int
    n_constantes: int
    n_com_imputado: int
    n_com_imputado_rejeitadas: int
    n_testadas: int
    n_rejeitadas: int
    fracao_rejeitada: float | None
    """`n_rejeitadas / n_testadas`; `None` sem variável testada, e não `0`."""
    n_rejeitadas_fdr: int
    fracao_minima_nao_normal: float | None
    """`1 - π̂₀` de Storey, cortado em zero: o piso da fração que não segue a normal.

    Não leia π̂₀ como fração normal: com pouco poder ele sai alto mesmo sem variável normal. `None`
    sem variável testada.
    """


class ResumoDaNormalidade(BaseModel):
    """A normalidade por variável e por grupo de estudo, sobre a matriz processada.

    `grupos` vazio quer dizer que não há amostra de estudo; o QC não entra como grupo.
    """

    features: list[str]
    grupos: list[NormalidadeDoGrupo]
    teste: str
    """O nome do teste, por extenso, para o texto e o arquivo não o escreverem à mão."""
    matriz: str
    """O estágio do pipeline sobre o qual o teste correu, como `matriz_do_pvalue` do volcano."""
    alfa: float
    """O nível de cada teste por variável, e a fração que se espera abaixo dele sob a normal."""
    nivel_do_fdr: float
    lambda_de_storey: float


class EtapaDaCorrecaoResposta(BaseModel):
    """A prévia da etapa de correção do passo a passo: os resumos das figuras da correção.

    Sem estado no servidor; o cache de correção por conteúdo liga esta etapa ao `/execucoes`.
    """

    # O corretor efetivo, depois da delegação, que o `/execucoes` repete para acertar o cache.
    correcao: str
    chave_da_etapa: str
    rsd_antes_e_depois: RsdAntesEDepois
    pcas_da_correcao: ParDePcas
    rsd_por_feature: RsdPorFeature
    composicao_por_classe: ComposicaoPorClasse


class EtapaDoPreprocessamentoResposta(EtapaDaCorrecaoResposta):
    """A prévia da etapa de pré-processamento: os resumos da correção e os das distribuições.

    Cumulativa, como `ExecucaoResposta`: refazer esta etapa também refresca as figuras da correção.
    """

    n_features_saida: int
    distribuicoes: ResumoDasDistribuicoes


class ExecucaoResposta(BaseModel):
    """O que a execução deixou no banco, o estado do par e os resumos para as figuras.

    `estado_concordancia` é lido do banco, não do que a rota compôs. `qualidade` traz o
    `QualityReport` inteiro, porque RSD e agrupamento de QC no PCA elegem métodos diferentes.
    """

    matrix_id: int
    run_id: int
    decision_id: int
    decision_proposta_id: int | None
    estado_concordancia: str
    campos_sobrescritos: list[str]
    apresentacao: str
    # Fica por compatibilidade: é o que `metrics_json` grava e o que clientes antigos leem.
    qc_rsd_median: float
    n_features_saida: int
    # `dict` e não modelo: o relatório é um `dataclass` do núcleo, e a tela lê `computed` antes
    # do valor, para separar 'não calculei' de 'calculei e é indefinido'.
    qualidade: dict[str, Any]
    rsd_antes_e_depois: RsdAntesEDepois
    chave_de_invalidacao: str
    """A identidade desta execução, para a figura saber que o número que desenha é o corrente.

    É `_chave_do_trabalho`, que cobre a spec inteira, o estudo, o par de grupos, os eixos
    sobrescritos e os valores: uma chave de modo, método e bytes serviria o resultado de outra
    configuração com o rótulo certo. Sempre preenchida, porque duas ausências comparariam iguais.
    """
    volcano: ResumoDoVolcano | None = None
    """Os vetores por feature do volcano, ou `None` quando não há par de grupos declarado.

    `None` é 'não calculado', não 'calculado e vazio'. Sai da mesma chamada a `univariate_test` que
    conta `n_significant_fdr` no `QualityReport`, para a figura e a aba de qualidade concordarem.
    """
    anova: ResumoDaAnova | None = None
    """Os vetores por feature da ANOVA de N grupos, ou `None` quando ela não correu.

    `None` cobre duas ausências: a família não foi pedida, ou a matriz não sustenta o teste. O
    caso normal, sem sobrevivente ao FDR, vem preenchido. Não depende do par de grupos.
    """
    plsda: ResumoDoPlsda | None = None
    """O PLS-DA com a validação ao lado, ou `None` sem par de grupos declarado.

    O objeto ausente não devolve escore nenhum; o que não pode existir é o objeto parcial, e
    `ResumoDoPlsda` não o admite.
    """
    correlacao: ResumoDaCorrelacao | None = None
    """A matriz de correlação entre features, ou `None` quando ela não correu.

    `None` cobre duas ausências: a família não foi pedida, ou a matriz não sustenta o cálculo
    (menos de três amostras de estudo, ou nenhuma feature utilizável).
    """
    busca_por_padrao: ResumoDaBuscaPorPadrao | None = None
    """As features ordenadas pela aderência ao perfil, ou `None` sem referência declarada.

    'Correu e nenhuma feature tem coeficiente definido' é o objeto preenchido com a lista vazia.
    """
    kmeans: ResumoDoKmeans | None = None
    """A partição das amostras por K-means, ou `None` quando ela não correu.

    `None` cobre a família não pedida (desligada por padrão, porque custa) e a matriz que não
    sustenta o cálculo. Um `kmeans_k` maior que o número de amostras é 422, não `None`: 'não pedi'
    e 'pedi e não cabe' têm consertos opostos.
    """
    som: ResumoDoSom | None = None
    """O mapa auto-organizável das amostras, ou `None` quando ele não correu.

    `None` cobre as mesmas ausências de `kmeans`. Uma grade maior que a coorte é legítima: as
    unidades vazias são informação.
    """
    rede: ResumoDaRede | None = None
    """A rede de correlação parcial esparsa desenviesada, ou `None` quando ela não correu.

    `None` cobre a família não pedida (desligada por padrão, é o ajuste mais caro da aba) e a matriz
    que não sustenta o cálculo. 'Correu e não achou aresta' é `arestas` vazia.
    """
    random_forest: ResumoDoClassificador | None = None
    """A floresta aleatória sobre os grupos clínicos, ou `None` quando ela não correu.

    `None` cobre o classificador não pedido (cada um custa minutos) e a matriz que não sustenta o
    cálculo. `classifier_folds` acima da menor classe é 422, não `None`.
    """
    svm: ResumoDoClassificador | None = None
    """A máquina de vetores de suporte sobre os grupos clínicos, no mesmo esquema da floresta."""
    normalidade: ResumoDaNormalidade | None = None
    """A normalidade por variável e por grupo de estudo, ou `None` com a família desligada."""
    distribuicoes: ResumoDasDistribuicoes
    """O resumo antes e depois que a figura de normalização consome, sempre presente.

    O caso sem feature comum é calculado e vazio, com `n_features_comparadas` a dizê-lo.
    """
    pcas_da_correcao: ParDePcas
    """As duas PCAs do painel que replica o PNG do R, sempre presentes.

    O lado antes nasce nesta resposta, e não do diagnóstico guardado: os dois painéis ficam sob a
    mesma chave de invalidação, e trocar de arquivo não mistura o antes de um com o depois de outro.
    """
    rsd_por_feature: RsdPorFeature
    """Os RSD por feature da matriz corrigida, sempre presentes.

    Corrigida e não processada: o corte por RSD de QC já teria removido as features que a
    distribuição existe para mostrar.
    """
    composicao_por_classe: ComposicaoPorClasse
    """A composição por classe clínica da matriz ingerida, sempre presente.

    É a única carga de figura sobre a ingerida, porque o corretor cria células faltantes.
    """


class MultiversoAceito(BaseModel):
    """A resposta do disparo do multiverso, com `202`: identificador e total de células.

    O total vem já no disparo, para não haver leitura com `feito=0` e denominador desconhecido.
    """

    identificador: str
    total: int
    persistido: bool = False
    """A execução foi gravada em disco e sobrevive ao fechamento do programa.

    `False` por padrão: omitido, quer dizer que ninguém gravou, e a interface não promete
    persistência que não houve.
    """

    motivo_nao_gravado: str | None = None
    """Por que a execução não foi gravada, ou `None` quando foi.

    A gravação é melhor esforço e não impede a execução, mas a falha é dita. O texto nomeia a
    configuração errada, nunca o valor: a URL de Postgres traz a senha, e este campo chega à tela.
    """


class DelegacaoResposta(BaseModel):
    """O que a política automática escolheu e por quê, com a ressalva que a acompanha.

    Vem no `202`, porque a escolha já está tomada antes de a correção, que é longa, começar.
    """

    rotulo: str
    politica: str
    correction: str
    confidence: float
    abstains: bool
    assessable: bool
    rationale: str
    evidence: dict[str, float]
    caveat: ItemComRessalva


class IntervaloDaBarra(BaseModel):
    """O intervalo jackknife-t de uma barra do painel da correção, ou o motivo de não haver um.

    Mede quanto a barra depende de quais QCs foram injetados, com nível nominal. Os limites, em
    fração, vêm todos ou nenhum, e `motivo` é `None` exatamente quando existem.
    """

    metodo: str
    nivel: float
    n_qc: int
    inferior: float | None = None
    superior: float | None = None
    erro_padrao: float | None = None
    motivo: str | None = None


class MedicaoDoCorretorResposta(BaseModel):
    """A medição dos corretores pelo RSD dos QCs fora da amostra: a recomendação que não adivinha.

    `n_features_medidas` é a população medida; `palpite_acertou` é `None` sem palpite.
    """

    vencedor: str
    scores: dict[str, float]
    margem: float | None
    avaliavel: bool
    abstem: bool
    estimador: str
    variante: str
    n_features_medidas: int | None
    # `palpite` é o Palpite medido: o vencedor da amostra de 200 features da mesma matriz e lotes,
    # se foi medido antes nesta sessão do servidor. Sem ele, os dois campos vêm `None`.
    palpite: str | None = None
    palpite_acertou: bool | None = None
    ressalva_da_politica: str = Field(default=AUTO_CAVEATS[AUTO_MEDIDA].resumo, min_length=1)
    # A partição em que a medição correu: quem lê o vencedor precisa saber sobre quantos lotes, e
    # com quantos QCs cada, ele venceu.
    n_lotes: int = 1
    qcs_por_lote: list[int] = Field(default_factory=list)
    # O intervalo por barra, só quando pedido. `None` quando não pedido, para não confundir com
    # 'pedido e nenhum corretor o teve'. Não muda a eleição: `vencedor`, `scores`, `margem` e
    # `abstem` são os mesmos do pedido sem intervalo.
    intervalos: dict[str, IntervaloDaBarra] | None = None
    # Os corretores não medidos, com o motivo: o lote menor tem menos QCs do que o ajuste fora da
    # amostra precisa. `scores` traz `null` para eles; `null` sem entrada aqui é RSD indefinido em
    # toda feature. Os dois ficam fora de `vencedor` e `margem`, e o `none` nunca é recusado.
    recusas: dict[str, str] = Field(default_factory=dict)


class CurvaDoLoessResposta(BaseModel):
    """A curva do LOESS de uma feature, as curvas refeitas sem cada QC e o envelope delas.

    Os vetores por amostra seguem as linhas da matriz, em ordem de injeção, com `null` na posição
    do valor não finito; cada ausência traz o seu motivo.
    """

    feature: str
    n_features: int
    semente: int
    ordens_de_injecao: list[int]
    lotes: list[str]
    qc: list[bool]
    valores: list[float | None]
    curva: list[float | None]
    curva_preenchida: list[bool]
    curva_sorteada: list[bool]
    qc_no_ajuste: list[bool]
    motivo_sem_curva: str | None = None
    spans_por_lote: list[float] = Field(default_factory=list)
    predicao_sem_o_qc: list[float | None]
    motivo_sem_predicao: list[str | None]
    envelope_inferior: list[float | None]
    envelope_superior: list[float | None]
    envelope_nao_ajustado: list[bool]
    n_curvas_do_envelope: int
    motivo_sem_envelope: str | None = None


class PalpiteResposta(BaseModel):
    """O Palpite medido: o vencedor do RSD dos QCs fora da amostra numa amostra de 200 features.

    `limiar_de_empate` é o corte do ruído do sorteio, abaixo do qual o Palpite se abstém; com
    `abstem`, a proposta continua a ser o vencedor da amostra.
    """

    rotulo_da_politica: str = Field(default=ROTULO_PELO_RSD, min_length=1)
    segunda_opiniao: SegundaOpiniao | None = None
    correction_proposta: str
    escores: dict[str, float]
    margem: float | None
    segundo: str | None
    confianca: float
    abstem: bool
    avaliavel: bool
    rationale: str = Field(min_length=1)
    n_features_medidas: int
    limiar_de_empate: float
    estimador: str = "leave_one_qc_out"
    variante: str
    n_lotes: int = 1
    qcs_por_lote: list[int] = Field(default_factory=list)
    origem: str = "modelo"
    disponivel_para_execucao: bool
    ressalva_da_politica: str = Field(default=AUTO_CAVEATS[AUTO_MEDIDA].resumo, min_length=1)
    eixos_sobrescreviveis: list[str] = Field(default_factory=lambda: sorted(SUPPORTED_OVERRIDES))
    # Os corretores que o Palpite não mediu, com o motivo, como em `MedicaoDoCorretorResposta`; o
    # `rationale` também os cita, porque vai ao registro da decisão.
    recusas: dict[str, str] = Field(default_factory=dict)


class TrabalhoAceito(BaseModel):
    """A resposta do disparo de uma operação longa, com `202`.

    `reclamado` diz se o identificador é de um trabalho que já corria para a mesma entrada; `total`
    conta etapas, não células.
    """

    identificador: str
    total: int
    reclamado: bool
    # Ausente quando não houve delegação: `exclude_none` no serializador da rota o tira do corpo.
    delegacao: DelegacaoResposta | None = None


class FeatureRemovidaResposta(BaseModel):
    """Uma feature que saiu no corte de `POST /matrizes/exportar`, com a causa.

    `rsd` é `None` quando a causa é não haver medida; zero a faria parecer a mais estável.
    """

    nome: str
    causa: str
    rsd: float | None = None


class RelatorioDeExportacao(BaseModel):
    """O que o corte da exportação manteve, removeu e por qual causa.

    `removidas` traz os nomes, para o corte ser contestável; as contagens por causa saem dela.
    """

    mantidas: int
    removidas: tuple[FeatureRemovidaResposta, ...] = ()
    zeros_como_vazias: int = 0
    nomes_sanitizados: int = 0


class ProgressoResposta(BaseModel):
    """Uma leitura do progresso de uma execução longa.

    Pergunte `terminou`, e não `estado == "concluido"`: uma execução que falhou ou foi cancelada
    também terminou.
    """

    identificador: str
    estado: str
    feito: int
    total: int
    fracao: float
    terminou: bool
    celula_que_falhou: str | None = None
    erro: str | None = None
    cancelamento_pedido: bool = False
    """Alguém pediu para parar, o que não é o mesmo que ter parado.

    Permite à tela dizer 'a parar' entre o clique e a parada. `False` por padrão, para clientes
    antigos não quebrarem.
    """
    etapa_em_curso: str | None = None
    """O nome da etapa que corre agora, como a rota a declara.

    `feito` e `total` contam etapas terminadas, não dizem qual corre. O nome vem sem acento, no
    vocabulário da barra. `None` quando a rota não nomeia etapas ou ainda não anunciou nenhuma.
    """


class EstabilidadeDeFeature(BaseModel):
    """A estabilidade de uma feature ao longo da grade, com o número de células que a mediu.

    `None` onde o núcleo diz 'não medido'. O `verdict` só tem sentido ao lado dos limiares que o
    produziram, servidos em `LimiaresDoVeredito`.
    """

    feature: str
    frac_significant: float
    median_qvalue: float | None
    median_rank: float | None
    rank_iqr: float | None
    sign_consistency: float | None
    n_cells: int
    verdict: Verdict


class MultiversoNoHistorico(BaseModel):
    """Uma execução do multiverso como está gravada em disco, a linha da tabela do histórico.

    Os estados não são os de `ProgressoResposta`: em disco não há `pendente`, e há `interrompido`.
    """

    identificador: str
    estado: str
    estudo: str
    mode: str
    total: int
    n_feitas: int
    iniciado_em: datetime
    terminado_em: datetime | None = None
    n_features: int = 0
    grupos: tuple[str, str] | None = None
    celula_que_falhou: str | None = None
    erro: str | None = None


class HistoricoDeMultiversos(BaseModel):
    """A lista de execuções gravadas e o teto que a cortou.

    Sem `limite`, uma lista com exatamente o teto não diria se há mais execuções.
    """

    execucoes: tuple[MultiversoNoHistorico, ...] = ()
    limite: int = 0


class LimiaresDoVeredito(BaseModel):
    """Os limiares que produziram os vereditos desta execução.

    Não há limiar para `fragile`: é o que sobra quando nem `robust` nem `absent` se aplicam.
    """

    robust_frac: float
    robust_sign: float
    absent_frac: float


class ConclusaoDoHistorico(BaseModel):
    """Uma execução gravada e a tabela de estabilidade dela, vazia se nada foi concluído."""

    execucao: MultiversoNoHistorico
    estabilidade: tuple[EstabilidadeDeFeature, ...] = ()
    # O corte viaja com os vereditos. Tem padrão porque uma execução cancelada antes de concluir
    # devolve tabela vazia, e não há corte a declarar.
    limiares: LimiaresDoVeredito | None = None


class ResultadoDoMultiverso(BaseModel):
    """A conclusão de uma execução do multiverso: a estabilidade de cada feature num contraste.

    `grupos` diz o contraste medido. Não há teto: `n_features` é igual a `len(estabilidade)`.
    """

    identificador: str
    grupos: tuple[str, str]
    n_celulas: int
    n_features: int
    estabilidade: tuple[EstabilidadeDeFeature, ...]
    # Os limiares viajam uma vez por execução e não por feature: o corte é um só, e repeti-lo por
    # linha convidaria a lê-lo como propriedade da feature.
    limiares: LimiaresDoVeredito


class ProblemaResposta(BaseModel):
    """Um problema de validação, na forma que a ingestão emite."""

    codigo: str
    severidade: str
    mensagem: str


class ValidacaoResposta(BaseModel):
    """O resultado da validação, com a lista completa de problemas e não só o primeiro."""

    aceita: bool
    n_samples: int | None = None
    n_features: int | None = None
    problemas: list[ProblemaResposta] = Field(default_factory=list)
    # Os nomes de amostra e de feature, para a tela oferecer a referência das normalizações em vez
    # de pedir que seja digitada. Vazios quando a validação recusa.
    sample_names: list[str] = Field(default_factory=list)
    feature_names: list[str] = Field(default_factory=list)
    # A ordem de injeção e a marca de QC de cada amostra, alinhadas a `sample_names`, para a tela
    # contar os QCs de cada lote pela mesma partição da correção (`domain/lotes.py`).
    ordens_de_injecao: list[int] = Field(default_factory=list)
    qc: list[bool] = Field(default_factory=list)


class SugestaoDeLotes(BaseModel):
    """Fronteiras de lote propostas, marcadas como sugestão por um literal constante.

    A palavra, presente em toda serialização, é mais difícil de ignorar que um booleano.
    """

    natureza: Literal["sugestao"] = "sugestao"
    changepoints: list[int]
    rotulos_por_amostra: list[int]
    n_lotes: int
    # Quantos QCs caem em cada lote, na ordem dos lotes; a tela não distingue QC em
    # `rotulos_por_amostra`. Um lote com um QC não tem dispersão a medir, e a sugestão que o
    # produz parece fundamentada sem ser.
    qcs_por_lote: list[int]
    # Os cortes que a detecção achou e quantos QCs cada lote teria com eles. Iguais a
    # `changepoints` quando a sugestão os adota; diferentes quando eles deixariam um lote com QC
    # de menos para medir fora da amostra, e aí a sugestão é um lote único e estes dois campos
    # dizem o que foi detectado e por que não entrou.
    cortes_detectados: list[int]
    qcs_por_lote_detectados: list[int]


class GrupoComContagem(BaseModel):
    """Um grupo a contrastar e quantas amostras de estudo ele tem, sem os QCs.

    A contagem diz se o contraste tem poder; nome e `n` vêm juntos para não se desalinharem.
    """

    nome: str
    n: int


class ResumoDaPermanova(BaseModel):
    """O par PERMANOVA e teste de dispersão sobre os componentes da PCA, nunca pela metade.

    Corre só sobre as amostras de estudo, então `n_por_grupo` soma menos que a nuvem desenhada.
    """

    grupos: list[str]
    n_por_grupo: list[int]
    n_componentes: int
    permutacoes: int
    semente: int

    estatistica: float
    pvalue: float
    estatistica_da_dispersao: float
    pvalue_da_dispersao: float

    distancia_media_ao_centroide: list[float]
    """A dispersão média de cada grupo, na ordem de `grupos`; sem ela, o `p` da dispersão não diz
    qual grupo é o largo."""

    n_descartadas_nao_finitas: int
    metodo: str
    """A distância e o esquema de permutação, por extenso, como a execução os fez."""


class TrajetoriaDosQcs(BaseModel):
    """A deriva do QC ao longo da corrida, com a ordem de injeção e o lote de cada QC.

    A ordem vem do `Sample_List`, a única fonte da ordem temporal. O sinal do PC1 é arbitrário.
    """

    scores: tuple[float, ...] = ()
    ordens_de_injecao: tuple[int, ...] = ()
    lotes: tuple[str, ...] = ()
    n_features_usadas: int = 0
    n_features_totais: int = 0


class ResumoDaPca(BaseModel):
    """Escores, cargas, variância explicada e o tipo de cada amostra, da PCA do diagnóstico.

    É a mesma `structural_pca` de que sai a trajetória do QC. O sinal dos componentes é arbitrário.
    """

    scores_pc1: list[float | None]
    scores_pc2: list[float | None]
    tipos_de_amostra: list[str]
    """Uma entrada por amostra, em ordem de injeção, com os valores de `SampleType`; é o que separa
    o QC da nuvem na figura."""

    grupos_por_amostra: list[str | None]
    """O grupo de cada amostra de estudo, e `null` nas outras, na ordem de `tipos_de_amostra`.

    Sai de `sample_mask`: o QC é a régua, não um grupo, e não pode ganhar cor e região próprias.
    """

    features: list[str]
    loadings_pc1: list[float | None]
    loadings_pc2: list[float | None]
    """As cargas das features retidas pela PCA, com os nomes em `features`, e não as da matriz
    inteira, que rotulariam cada ponto com o nome de outra feature."""

    variancia_explicada: list[float]
    """A fração da variância total por componente. Soma 1 sobre todos os componentes, não sobre
    os desenhados."""

    n_features_usadas: int
    n_features_totais: int
    n_componentes: int

    permanova: ResumoDaPermanova | None = None
    """O par PERMANOVA e dispersão sobre estes escores, ou `None` quando a população não sustenta
    o teste. 'Correu e não separou' é `pvalue` alto, não `None`."""


class DiagnosticoResposta(BaseModel):
    """O `DefectProfile`, com a sugestão de lotes num objeto próprio que carrega a marca."""

    defect_type: str
    confidence: float
    qc_rsd_median: float
    qc_rsd_p75: float
    zero_fraction: float
    missing_fraction: float
    pc1_injection_correlation: float
    qc_dispersion_ratio: float
    n_features: int
    n_samples: int
    lotes: SugestaoDeLotes
    # A contagem de features sem RSD mensurável e a variante que a produziu, sempre juntas: a
    # remoção de outliers cria indefinição, e o número sem a variante não se compara. A leitura
    # corre antes da correção, e é ali que se decide se vale corrigir.
    n_features_rsd_undefined: int
    rsd_variant: str
    # Obrigatória: preenchida só por `POST /matrizes/diagnosticar`, e um padrão serviria uma
    # trajetória vazia sem acusar o esquecimento.
    trajetoria_dos_qcs: TrajetoriaDosQcs
    # Obrigatória pela mesma razão, e sai do mesmo objeto de PCA que a trajetória.
    pca: ResumoDaPca
    # O RSD por feature da matriz ingerida, obrigatório pela mesma razão: é a figura de antes da
    # correção, com a variante de `n_features_rsd_undefined`, para o 'N de M' bater com ele.
    rsd_por_feature: RsdPorFeature
    grupos_disponiveis: tuple[GrupoComContagem, ...] = ()
    """Os grupos a contrastar, com nome e contagem, ordenados por nome.

    O multiverso e as análises de contraste precisam de um par de grupos existente, e é aqui que a
    interface descobre quais há. Vazio por padrão, para clientes antigos não quebrarem.
    """

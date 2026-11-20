"""O registro dos construtores de figura e o vocabulário fechado de estado do catálogo.

A janela mostra as figuras da seção Statistical Analysis [one factor] do MetaboAnalyst para
as quais este projeto produz o número, e nenhuma outra. Para conferir isso é preciso nomear
os dois lados: o catálogo `[G-1]` de `docs/metodos-metaboanalyst.md` e este registro.

A correspondência não é com o catálogo inteiro. Entradas excluídas do escopo ou adiadas não
têm construtor, e há construtores que não são figura do MetaboAnalyst e respondem a um item
do relatório. Por isso cada entrada do catálogo tem um estado de vocabulário fechado e cada
construtor declara contra qual conjunto responde. Entrada sem estado reprova, e assim não
existe lista de exceções para estender.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Sequence

__all__ = [
    "ARTIGOS_DO_CONSTRUTOR",
    "CONSTRUTORES",
    "PREFIXO_DO_CONSTRUTOR",
    "ConstrutorDeFigura",
    "EstadoDoCatalogo",
    "Vinculo",
    "familias_do_seletor",
    "rotulo_do_construtor",
]

# Toda função de módulo em `refino.plots.*` cujo nome começa assim é um construtor de figura
# e tem de estar em `CONSTRUTORES`. Um construtor novo que ninguém registrou reprova no teste
# de bijeção.
PREFIXO_DO_CONSTRUTOR: Final = "figura_"


class EstadoDoCatalogo(StrEnum):
    """Os três estados que uma entrada do `[G-1]` pode ter.

    O vocabulário é fechado para que um estado desconhecido reprove. Tratado como 'adiada', um
    erro de digitação viraria isenção silenciosa.
    """

    CONSTRUIDA = "construída"
    EXCLUIDA_KD3 = "excluída-KD3"
    ADIADA = "adiada"


class Vinculo(StrEnum):
    """Contra qual conjunto o construtor é conferido: o catálogo ou os itens do relatório."""

    CATALOGO = "catálogo"
    """Figura do MetaboAnalyst: responde a uma entrada `construída` do `[G-1]`."""

    RELATORIO = "relatório"
    """Não é figura do MetaboAnalyst: responde a um item de `ITENS_OBRIGATORIOS` do relatório."""


@dataclass(frozen=True)
class ConstrutorDeFigura:
    """Um construtor, o módulo onde vive e a que ele responde."""

    modulo: str
    """Nome do módulo, sem o prefixo `refino.plots.`. Por exemplo, `volcano`."""

    funcao: str
    """Nome da função, que começa por `PREFIXO_DO_CONSTRUTOR`."""

    vinculo: Vinculo

    chave: str
    """A entrada do `[G-1]` quando o vínculo é `CATALOGO`; `item-N` quando é `RELATORIO`.

    É também a família da figura no seletor. Ver `familias_do_seletor`.
    """


# Os artigos que separam o prefixo `figura_` do nome da figura. É constante, e não literal
# dentro da função, porque o teste do rótulo único também a lê. A lista cobre os artigos
# possíveis, e não só os que as funções atuais usam. A ordem não importa: `dos_` não é casado
# por `do_`, porque o sublinhado faz parte do padrão.
ARTIGOS_DO_CONSTRUTOR: Final = ("do_", "da_", "de_", "dos_", "das_")


def rotulo_do_construtor(funcao: str) -> str:
    """O nome legível de uma figura: `figura_dos_scores_do_plsda` vira `"scores do plsda"`.

    É o rótulo que o seletor da janela mostra. Deriva do nome da função, e não de uma tabela,
    para a figura nova não ficar sem rótulo. E deriva da função, e não do par módulo e chave,
    porque esse par se repete: as quatro figuras de PCA teriam o mesmo rótulo.
    """
    resto = funcao.removeprefix(PREFIXO_DO_CONSTRUTOR)
    for artigo in ARTIGOS_DO_CONSTRUTOR:
        if resto.startswith(artigo):
            resto = resto.removeprefix(artigo)
            break
    return resto.replace("_", " ")


# A trajetória do QC responde ao relatório, e não ao catálogo: ela é a metade de trajetória do
# item 3 ('Perfil de diagnóstico, com trajetória dos QCs e pontos de mudança'). Os pontos de
# mudança não são desenhados.
#
# As quatro entradas seguintes são as de normalização do `[G-1]`. Uma entrada só vira
# `construída` no documento quando o construtor existe aqui; qualquer das duas metades
# sozinha reprova no teste de bijeção.
CONSTRUTORES: Final[tuple[ConstrutorDeFigura, ...]] = (
    ConstrutorDeFigura(
        modulo="deriva",
        funcao="figura_da_deriva_do_qc",
        vinculo=Vinculo.RELATORIO,
        chave="item-3",
    ),
    ConstrutorDeFigura(
        modulo="distribuicao",
        funcao="figura_das_caixas_por_feature",
        vinculo=Vinculo.CATALOGO,
        chave="Caixas (Feature View)",
    ),
    ConstrutorDeFigura(
        modulo="distribuicao",
        funcao="figura_das_caixas_por_amostra",
        vinculo=Vinculo.CATALOGO,
        chave="Caixas (Sample View)",
    ),
    ConstrutorDeFigura(
        modulo="distribuicao",
        funcao="figura_da_densidade_por_feature",
        vinculo=Vinculo.CATALOGO,
        chave="Densidade (Feature View)",
    ),
    ConstrutorDeFigura(
        modulo="distribuicao",
        funcao="figura_da_densidade_por_amostra",
        vinculo=Vinculo.CATALOGO,
        chave="Densidade (Sample View)",
    ),
    # A figura sai vazia neste estudo, porque nenhuma feature sobrevive ao FDR, e a entrada é
    # `construída` do mesmo jeito: o estado diz que há construtor, e não que há dado.
    ConstrutorDeFigura(
        modulo="volcano",
        funcao="figura_do_volcano",
        vinculo=Vinculo.CATALOGO,
        chave="Volcano plot",
    ),
    # `Fold Change`, `T-tests` e `One-way ANOVA` são três entradas distintas do `[G-1]`, uma
    # função para cada. As duas primeiras leem o mesmo resultado univariado que o volcano; a
    # ANOVA tem cálculo próprio.
    #
    # Os nomes das funções têm de ser distintos entre si em todo o registro: o rótulo do
    # seletor deriva deles, e uma colisão daria duas linhas iguais.
    ConstrutorDeFigura(
        modulo="univariada",
        funcao="figura_do_fold_change",
        vinculo=Vinculo.CATALOGO,
        chave="Fold Change",
    ),
    ConstrutorDeFigura(
        modulo="univariada",
        funcao="figura_dos_testes_t",
        vinculo=Vinculo.CATALOGO,
        chave="T-tests",
    ),
    ConstrutorDeFigura(
        modulo="anova",
        funcao="figura_da_anova",
        vinculo=Vinculo.CATALOGO,
        chave="One-way ANOVA",
    ),
    # As quatro compartilham a chave `PCA`. O `[G-1]` tem uma entrada só, que o MetaboAnalyst
    # abre em seis abas, e a bijeção compara conjuntos de chaves. Das seis abas, `Overview` e
    # `Synchronized 3D Plots` não estão construídas.
    #
    # Nenhuma delas responde ao item 6 do relatório, que pede a PCA antes e depois: elas
    # descrevem uma matriz só. Quem o satisfaz é o painel da correção, no fim desta tupla.
    ConstrutorDeFigura(
        modulo="pca",
        funcao="figura_do_scree",
        vinculo=Vinculo.CATALOGO,
        chave="PCA",
    ),
    ConstrutorDeFigura(
        modulo="pca",
        funcao="figura_dos_scores",
        vinculo=Vinculo.CATALOGO,
        chave="PCA",
    ),
    ConstrutorDeFigura(
        modulo="pca",
        funcao="figura_das_cargas",
        vinculo=Vinculo.CATALOGO,
        chave="PCA",
    ),
    ConstrutorDeFigura(
        modulo="pca",
        funcao="figura_do_biplot",
        vinculo=Vinculo.CATALOGO,
        chave="PCA",
    ),
    # As três compartilham a chave `PLS-DA`, pela mesma razão das quatro do PCA. Ficam de fora
    # o `Overview`, que é par a par de vários componentes, e a aba de permutação, que desenha
    # o histograma dos sorteios: `PLSDAResult` guarda o p, e não os Q² sorteados.
    ConstrutorDeFigura(
        modulo="plsda",
        funcao="figura_dos_scores_do_plsda",
        vinculo=Vinculo.CATALOGO,
        chave="PLS-DA",
    ),
    ConstrutorDeFigura(
        modulo="plsda",
        funcao="figura_das_cargas_do_plsda",
        vinculo=Vinculo.CATALOGO,
        chave="PLS-DA",
    ),
    ConstrutorDeFigura(
        modulo="plsda",
        funcao="figura_do_vip",
        vinculo=Vinculo.CATALOGO,
        chave="PLS-DA",
    ),
    # As três do PLS-DA multiclasse compartilham a chave `PLS-DA` com as do binário. Os nomes
    # das funções são distintos dos do caminho binário para os rótulos do seletor não
    # colidirem.
    ConstrutorDeFigura(
        modulo="plsda_multiclasse",
        funcao="figura_dos_escores_um_contra_resto",
        vinculo=Vinculo.CATALOGO,
        chave="PLS-DA",
    ),
    ConstrutorDeFigura(
        modulo="plsda_multiclasse",
        funcao="figura_das_cargas_um_contra_resto",
        vinculo=Vinculo.CATALOGO,
        chave="PLS-DA",
    ),
    ConstrutorDeFigura(
        modulo="plsda_multiclasse",
        funcao="figura_da_importancia_um_contra_resto",
        vinculo=Vinculo.CATALOGO,
        chave="PLS-DA",
    ),
    # `Dendrogram` e `Heatmaps` são duas entradas do `[G-1]`, uma função para cada. A chave
    # é a entrada tal como está escrita no documento, no plural, porque a bijeção compara
    # texto.
    ConstrutorDeFigura(
        modulo="agrupamento",
        funcao="figura_do_dendrograma",
        vinculo=Vinculo.CATALOGO,
        chave="Dendrogram",
    ),
    ConstrutorDeFigura(
        modulo="agrupamento",
        funcao="figura_do_mapa_de_calor",
        vinculo=Vinculo.CATALOGO,
        chave="Heatmaps",
    ),
    # `K-means` e `Self Organizing Map (SOM)` são as outras duas entradas de
    # `Cluster Analysis`, uma função para cada, com a chave escrita como no documento.
    ConstrutorDeFigura(
        modulo="agrupamento",
        funcao="figura_do_kmeans",
        vinculo=Vinculo.CATALOGO,
        chave="K-means",
    ),
    ConstrutorDeFigura(
        modulo="agrupamento",
        funcao="figura_do_mapa_auto_organizavel",
        vinculo=Vinculo.CATALOGO,
        chave="Self Organizing Map (SOM)",
    ),
    # `Correlation Heatmaps` e `Pattern Search` são duas entradas de `Univariate Analysis`,
    # uma função para cada: a primeira desenha a matriz de correlação, a segunda ordena as
    # features pela aderência a um perfil de referência.
    ConstrutorDeFigura(
        modulo="correlacao",
        funcao="figura_do_mapa_de_correlacao",
        vinculo=Vinculo.CATALOGO,
        chave="Correlation Heatmaps",
    ),
    ConstrutorDeFigura(
        modulo="correlacao",
        funcao="figura_da_busca_por_padrao",
        vinculo=Vinculo.CATALOGO,
        chave="Pattern Search",
    ),
    # `Random Forest` e `SVM` são as duas entradas de `Classification & Feature Selection`,
    # uma função para cada.
    ConstrutorDeFigura(
        modulo="classificacao",
        funcao="figura_do_random_forest",
        vinculo=Vinculo.CATALOGO,
        chave="Random Forest",
    ),
    ConstrutorDeFigura(
        modulo="classificacao",
        funcao="figura_do_svm",
        vinculo=Vinculo.CATALOGO,
        chave="SVM",
    ),
    # `Correlation Networks (DSPC)` é a última entrada de `Univariate Analysis`. A chave não
    # é a mesma de `Correlation Heatmaps`: o mapa desenha a correlação simples par a par, e a
    # rede desenha a correlação parcial, condicionada a todas as outras features.
    ConstrutorDeFigura(
        modulo="rede",
        funcao="figura_da_rede_de_correlacao_parcial",
        vinculo=Vinculo.CATALOGO,
        chave="Correlation Networks (DSPC)",
    ),
    # O painel da correção é um construtor só, com a chave do item 6 ('Gráfico de PCA antes e
    # depois, com QCs destacados'). Ele contém as barras de RSD de QC e as duas PCAs, como o
    # PNG que o pipeline R entrega. O item 4 fica sem construtor próprio: a metade de cima da
    # mesma imagem não é uma segunda figura. O painel não é figura do MetaboAnalyst.
    ConstrutorDeFigura(
        modulo="correcao",
        funcao="figura_do_painel_da_correcao",
        vinculo=Vinculo.RELATORIO,
        chave="item-6",
    ),
    # O violino de RSD por feature responde ao item 10 do relatório ('Dispersão do RSD por
    # feature, das amostras contra a dos QCs, com o n de cada lado').
    ConstrutorDeFigura(
        modulo="dispersao",
        funcao="figura_do_violino_do_rsd_por_feature",
        vinculo=Vinculo.RELATORIO,
        chave="item-10",
    ),
    # A composição por classe responde ao item 11 do relatório ('Composição por classe, com o
    # percentual de células faltantes e a abundância típica'). É medida sobre a matriz
    # ingerida.
    ConstrutorDeFigura(
        modulo="composicao",
        funcao="figura_da_composicao_por_classe",
        vinculo=Vinculo.RELATORIO,
        chave="item-11",
    ),
    # A curva do LOESS com o envelope sem um QC responde ao item 12 do relatório.
    ConstrutorDeFigura(
        modulo="curva_do_loess",
        funcao="figura_da_curva_do_loess",
        vinculo=Vinculo.RELATORIO,
        chave="item-12",
    ),
)


def familias_do_seletor(
    construtores: Sequence[ConstrutorDeFigura] = CONSTRUTORES,
) -> tuple[tuple[str, tuple[ConstrutorDeFigura, ...]], ...]:
    """Pares de família e construtores dela, na ordem do registro. A família é a `chave`.

    O agrupamento deriva do campo que o registro já tem: os quatro construtores de chave `PCA`
    são a família PCA. Uma lista de famílias escrita à parte poderia deixar uma figura nova
    no grupo errado ou em nenhum.

    A ordem é a de primeiro aparecimento no registro, entre as famílias e dentro delas. Para
    a PCA isso dá scree, scores, cargas e biplot, que é a ordem em que se olha uma PCA.
    """
    por_familia: dict[str, list[ConstrutorDeFigura]] = {}
    for construtor in construtores:
        por_familia.setdefault(construtor.chave, []).append(construtor)
    return tuple((chave, tuple(membros)) for chave, membros in por_familia.items())

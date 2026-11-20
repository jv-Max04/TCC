"""Composição por classe: o percentual de células faltantes e a abundância típica.

Dois painéis empilhados sobre o mesmo eixo de classes, uma barra por classe clínica: em cima o
percentual de células faltantes, embaixo a abundância típica. A pergunta é se as classes estão
igualmente bem medidas. Uma classe com muito mais ausência ou muito menos sinal que as outras
é um problema de aquisição, anterior a qualquer estatística.

As duas definições vão nomeadas na imagem. 'Faltante' tem três leituras (zero, não finita e a
união das duas), que coincidem nas matrizes ingeridas do laboratório por acaso e se separam
depois de um corretor. Por isso as três contagens atravessam o fio, a figura desenha a união e
o rodapé escreve as outras duas. 'Abundância' é a mediana do log10 das células detectadas,
como em `diagnosis/profile.py`, calculada por classe; a média é governada pela cauda e deixa o
painel plano.

A matriz é a ingerida. A figura mede o estudo, não o corretor: um corretor cria células
faltantes, e sobre a matriz corrigida a figura diria que o equipamento não detectou o que ele
detectou.

O eixo da abundância é linear e a quantidade desenhada é `log10(intensidade)`; escala
logarítmica é proibida nos montadores porque a ampliação da janela trabalha em unidades de
eixo. A intensidade bruta vai anotada sobre cada barra.

São quatro desfechos:

- classes com célula detectada: as duas filas de barras;
- uma classe sem nenhuma detectada: a barra de faltantes dela e nenhuma de abundância;
- nenhuma classe com detectada: só a fila de faltantes, e o painel de baixo declara a causa;
- zero classes: nenhuma barra, e a causa em texto.

Zero classes não levanta `MatrizVaziaError`: 'esta matriz não tem injeção de estudo' é fato
sobre o desenho do estudo e é rotulado como achado. As cores vêm do ciclo do tema, ou dos
estilos de grupo quando são passados, e as duas barras de uma classe têm a mesma cor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from matplotlib.lines import Line2D

from refino.plots.base import (
    GID_REQUEBRAVEL,
    NAO_DECLARADA,
    cores_do_ciclo,
    denominador_ou_recusa,
    inteiro_do_fio,
    nova_figura,
    numero_do_fio,
    objetos_do_fio,
    texto_do_fio,
)
from refino.plots.distribuicao import (
    rotulos_cabem_na_altura,
)
from refino.plots.recorte import (
    TAMANHO_DA_FONTE,
    Recorte,
    aplica,
    colunas_do_eixo,
    decimal,
    em_percentagem,
    legenda_fora,
    milhar,
)
from refino.plots.series import ordem_canonica

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Mapping

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from refino.plots.series import EstilosDosGrupos

__all__ = [
    "ALTURA_DA_COMPOSICAO",
    "DEFINICOES_DA_ABUNDANCIA",
    "DEFINICOES_DO_FALTANTE",
    "LARGURA_DA_BARRA_DA_CLASSE",
    "PAPEL_DA_ABUNDANCIA",
    "PAPEL_DOS_FALTANTES",
    "ROTULO_DO_EIXO_DA_ABUNDANCIA",
    "ROTULO_DO_EIXO_DOS_FALTANTES",
    "SEM_CLASSE_DECLARADA",
    "SEM_MARCAS_DE_CLASSE",
    "TITULO_DA_ABUNDANCIA",
    "TITULO_DA_COMPOSICAO",
    "TITULO_DOS_FALTANTES",
    "ClasseNaFigura",
    "ComposicaoDasClasses",
    "MarcasDasClasses",
    "classes_do_fio",
    "composicao_do_fio",
    "definicao_lida",
    "figura_da_composicao_por_classe",
    "frase_sem_abundancia",
    "grupos_da_composicao",
    "marca_da_classe",
    "marcas_das_classes",
    "recorte_da_composicao_por_classe",
    "rotulo_da_classe",
    "rotulos_das_classes_cabem",
]

TITULO_DA_COMPOSICAO: Final = "Composição por classe — o que falta, e em que nível de sinal"

TITULO_DOS_FALTANTES: Final = "Células faltantes por classe"
TITULO_DA_ABUNDANCIA: Final = "Abundância típica por classe"

ROTULO_DO_EIXO_DOS_FALTANTES: Final = "células faltantes (%)"

# A quantidade desenhada é o log, num eixo linear, e o rótulo diz isso. É curto porque um
# rótulo longo de eixo y atravessa o `suptitle`; a população ('células detectadas') está no
# título do painel e no rodapé.
ROTULO_DO_EIXO_DA_ABUNDANCIA: Final = "log10 da intensidade"

# Papel de cada artista de dado. Uma classe tem uma barra em cada painel, e os testes indexam
# as séries por rótulo: sem o papel as duas colapsariam numa.
PAPEL_DOS_FALTANTES: Final = "faltantes"
PAPEL_DA_ABUNDANCIA: Final = "abundância"

# Espessura da barra, em pontos de linha. As barras são `ax.vlines` e não `ax.bar`, que cria
# `Rectangle` em `ax.patches`, fora do que os testes e o cursor da janela leem.
LARGURA_DA_BARRA_DA_CLASSE: Final = 18.0

# Polegadas. São dois eixos empilhados, com o valor anotado sobre cada barra e um rodapé longo:
# com `ALTURA_PADRAO` cada painel ficaria com menos de uma polegada e as anotações tocariam o
# topo.
ALTURA_DA_COMPOSICAO: Final = 6.0


class MarcasDasClasses(StrEnum):
    """Como as classes são nomeadas no eixo x. Vocabulário fechado de três desfechos."""

    HORIZONTAIS = "horizontais"
    """Os nomes cabem lado a lado, com o `n` na linha de baixo."""

    RODADAS = "rodadas"
    """Os nomes não cabem na largura e são escritos a 90 graus, com o `n` como sufixo."""

    AUSENTES = "ausentes"
    """Nem rodados cabem na altura: não há marca, e a legenda nomeia as classes."""


SEM_MARCAS_DE_CLASSE: Final = (
    "os nomes das classes não cabem no eixo, nem rodados: quem as nomeia é a legenda, na ordem das "
    "barras"
)

SEM_CLASSE_DECLARADA: Final = (
    "nenhuma injeção de estudo nesta matriz: não há classe clínica a compor. As injeções que ela "
    "tem são de QC ou de validação, e nenhuma das duas é classe — o QC é uma mistura de TODAS as "
    "amostras, e a composição dele seria uma média ponderada das que não estão aqui"
)

# Recuo de cada rótulo que pode faltar no corpo. Nenhum é um valor plausível: a ausência se
# declara com o nome do que falta, nunca com o valor mais provável.
POPULACAO_NAO_DECLARADA: Final = "população ausente do corpo"
DEFINICAO_AUSENTE: Final = "definição ausente do corpo"

# As duas definições, traduzidas do token que o corpo traz. A chave é a mesma cadeia dos
# `Literal` de `api/esquemas.py`, e um teste compara as duas listas por `get_args`. O texto traz
# só a definição, porque vai impresso no rodapé de toda figura.
DEFINICOES_DO_FALTANTE: Final[Mapping[str, str]] = {
    "zero_ou_nao_finita": (
        "célula igual a zero OU não finita, que é `preprocessing.missing.is_missing` — a mesma "
        "definição com que o pipeline recusa `imputation=none`. O zero domina e é MNAR: o "
        "equipamento deixa a célula vazia quando não detecta"
    )
}

DEFINICOES_DA_ABUNDANCIA: Final[Mapping[str, str]] = {
    "mediana_do_log10_das_detectadas": (
        "mediana de log10(x) sobre as células DETECTADAS — finitas e estritamente positivas, e "
        "NÃO a média de todas, que é governada pela cauda e achata o painel"
    )
}

# O que o rodapé escreve quando o token não está no mapa. Nomeia o token em vez de escolher uma
# definição plausível, que poria a imagem a declarar uma definição que a rota não usou.
DEFINICAO_DESCONHECIDA: Final = "definição NÃO RECONHECIDA por esta figura"

# Quanto o painel dos faltantes sobe acima da barra mais alta, em fração do alcance, para o
# valor anotado caber. O painel da abundância tira a folga de `_faixa_da_abundancia`. É maior
# que a folga do painel da correção porque cada eixo aqui é bem mais baixo e a anotação tem
# altura fixa em pontos: com folga menor os valores ficam em cima da moldura.
FOLGA_DA_COMPOSICAO: Final = 0.28


def rotulo_da_classe(grupo: str, papel: str) -> str:
    """`("ATLL", "faltantes")` vira `"ATLL — faltantes"`. Único por artista de dado.

    Cada classe tem duas barras, uma em cada painel, e os testes indexam as séries por rótulo:
    com o nome do grupo sozinho as duas colapsariam numa.
    """
    return f"{grupo} — {papel}"


def definicao_lida(token: str, mapa: Mapping[str, str]) -> str:
    """A frase de uma definição, ou o token nomeado quando ele não é conhecido.

    O token vem do fio. Se a figura não sabe traduzi-lo, ela o diz; escolher a única frase do
    mapa declararia uma definição que a rota não usou.
    """
    frase = mapa.get(token)
    return frase if frase is not None else f"{DEFINICAO_DESCONHECIDA}: «{token}»"


@dataclass(frozen=True)
class ClasseNaFigura:
    """Uma classe clínica, com as contagens prontas e a abundância que pode não existir.

    As contagens vêm da rota. A única conta feita aqui é `n_faltantes / n_celulas`, para o eixo
    em percentagem.
    """

    grupo: str
    n_injecoes: int
    n_celulas: int
    n_zeros: int
    n_nao_finitas: int
    n_faltantes: int
    n_detectadas: int
    log10_mediana: float
    """A abundância típica, ou `nan` quando não há célula detectada.

    No fio a ausência é `None`, e `numero_do_fio` a transforma em `nan`. Ela não pode virar
    `0.0`: num eixo de log10 a barra no piso se leria como intensidade 1, o oposto de 'não se
    detectou nada'.
    """

    @property
    def tem_abundancia(self) -> bool:
        """Se há célula detectada e, portanto, barra no painel da abundância."""
        return self.n_detectadas > 0 and math.isfinite(self.log10_mediana)

    @property
    def fracao_faltante(self) -> float:
        """`n_faltantes / n_celulas`, com o denominador da própria classe.

        `n_celulas` vem por classe em vez de ser derivado de `n_injecoes * n_features` do
        conjunto, que não é um total desta classe.
        """
        return self.n_faltantes / self.n_celulas if self.n_celulas else float("nan")

    @property
    def intensidade_tipica(self) -> float:
        """`10 ** log10_mediana`: a mediana em unidades brutas, para a anotação sobre a barra.

        `log10` é monótona, então a mediana do log é o log da mediana. O valor bruto é anotado e
        não desenhado porque o eixo é linear sobre o log.
        """
        return float(10.0**self.log10_mediana)


@dataclass(frozen=True)
class ComposicaoDasClasses:
    """As classes, mais os rótulos sem os quais nenhum dos dois números é interpretável.

    `matriz`, `populacao`, `n_features`, `n_injecoes_fora` e as duas definições são do conjunto
    e não de cada classe: todas as classes saem da mesma matriz, sob a mesma máscara e as mesmas
    definições.
    """

    classes: tuple[ClasseNaFigura, ...]
    matriz: str
    populacao: str
    n_features: int
    n_injecoes_fora: int
    definicao_de_faltante: str
    definicao_de_abundancia: str

    @property
    def desenhavel(self) -> bool:
        return bool(self.classes)

    @property
    def n_injecoes(self) -> int:
        """As injeções que entraram, somadas das classes."""
        return sum(classe.n_injecoes for classe in self.classes)

    @property
    def sem_abundancia(self) -> tuple[ClasseNaFigura, ...]:
        """As classes sem barra no painel da abundância."""
        return tuple(classe for classe in self.classes if not classe.tem_abundancia)


def classes_do_fio(valores: Any) -> tuple[ClasseNaFigura, ...]:
    """Lê a lista de classes do corpo de `/execucoes`, com guarda em cada campo.

    Nada do que vem do fio é tipado: um campo pode chegar `null`, a lista pode chegar `None` e
    um elemento pode não ser objeto. `objetos_do_fio` trata os dois últimos casos.

    `log10_mediana_das_detectadas` é o único campo que pode legitimamente chegar `null`;
    `numero_do_fio` o traduz em `nan`, que `tem_abundancia` lê como 'sem barra'. O nome do grupo
    passa por `texto_do_fio`, para um campo ausente não virar o rótulo `"None"`.
    """
    return tuple(
        ClasseNaFigura(
            grupo=texto_do_fio(bruto.get("grupo"), NAO_DECLARADA),
            n_injecoes=inteiro_do_fio(bruto.get("n_injecoes")),
            n_celulas=inteiro_do_fio(bruto.get("n_celulas")),
            n_zeros=inteiro_do_fio(bruto.get("n_zeros")),
            n_nao_finitas=inteiro_do_fio(bruto.get("n_nao_finitas")),
            n_faltantes=inteiro_do_fio(bruto.get("n_faltantes")),
            n_detectadas=inteiro_do_fio(bruto.get("n_detectadas")),
            log10_mediana=numero_do_fio(bruto.get("log10_mediana_das_detectadas")),
        )
        for bruto in objetos_do_fio(valores)
    )


def composicao_do_fio(
    *,
    classes: Any,
    matriz: Any,
    populacao: Any,
    n_features: Any,
    n_injecoes_fora: Any,
    definicao_de_faltante: Any,
    definicao_de_abundancia: Any,
) -> ComposicaoDasClasses:
    """Monta a composição a partir do fio, com os rótulos passados pela guarda de fronteira.

    Os dois tokens de definição passam por `texto_do_fio` e não por uma guarda de vocabulário:
    o vocabulário é validado na fronteira, por `ComposicaoPorClasse`, e uma segunda lista aqui
    poderia divergir. Token desconhecido é nomeado no rodapé, por `definicao_lida`.
    """
    return ComposicaoDasClasses(
        classes=classes_do_fio(classes),
        matriz=texto_do_fio(matriz, NAO_DECLARADA),
        populacao=texto_do_fio(populacao, POPULACAO_NAO_DECLARADA),
        n_features=inteiro_do_fio(n_features),
        n_injecoes_fora=inteiro_do_fio(n_injecoes_fora),
        definicao_de_faltante=texto_do_fio(definicao_de_faltante, DEFINICAO_AUSENTE),
        definicao_de_abundancia=texto_do_fio(definicao_de_abundancia, DEFINICAO_AUSENTE),
    )


def grupos_da_composicao(classes: Any) -> tuple[str, ...]:
    """Os grupos clínicos que a composição traz, na ordem canônica: a coorte do estudo.

    A composição é a raiz usada porque vem em toda resposta `200` de `/execucoes` e nas prévias
    das etapas, lista todos os grupos de estudo da matriz e já chega ordenada. As outras raízes
    trazem um recorte (o par do PLS-DA e do volcano), incluem o QC (K-means e SOM) ou só vêm
    quando pedidas (ANOVA).

    Nome que não chegou é descartado. `classes_do_fio` o troca por um recuo que serve de rótulo
    de barra, não de nome de grupo.
    """
    return ordem_canonica(
        classe.grupo for classe in classes_do_fio(classes) if classe.grupo != NAO_DECLARADA
    )


def frase_sem_abundancia(composicao: ComposicaoDasClasses) -> str:
    """As classes sem barra no painel da abundância, nomeadas.

    O fato é 'nenhuma célula desta classe foi detectada', e ele é desenhado como ausência com o
    nome ao lado; uma barra no piso do eixo diria uma intensidade plausível e falsa. A frase é
    curta porque vai dentro do painel e uma frase longa encostaria na anotação da barra vizinha.
    """
    nomes = ", ".join(classe.grupo for classe in composicao.sem_abundancia)
    return f"sem abundância típica: {nomes} — nenhuma célula detectada, logo não há mediana"


def recorte_da_composicao_por_classe(*, composicao: ComposicaoDasClasses) -> Recorte:
    """O rodapé: a população, a procedência e as duas definições por extenso.

    `Recorte.texto()` escreve 'N de M features', então o par do topo é `n_features` dos dois
    lados: a figura agrega sobre todas as features da matriz. As injeções, que são o que ela
    reparte, vão na `amostragem`, com a unidade escrita.

    As três contagens de ausência vão no `metodo`: a figura desenha a união, e quem lê vê
    quanto dela é zero e quanto é não finito.
    """
    return Recorte(
        desenhadas=composicao.n_features,
        disponiveis=denominador_ou_recusa(composicao.n_features, de="a composição por classe"),
        amostragem=(
            f"nenhuma: entram TODAS as features e TODAS as injeções da população "
            f"«{composicao.populacao}» — {milhar(composicao.n_injecoes)} injeções em "
            f"{milhar(len(composicao.classes))} classe(s), com "
            f"{milhar(composicao.n_injecoes_fora)} fora dela (QC e validação). A percentagem de "
            "cada classe é sobre as células DELA"
        ),
        matriz=f"uma — «{composicao.matriz}»",
        metodo=_metodo_da_composicao(composicao),
    )


def _metodo_da_composicao(composicao: ComposicaoDasClasses) -> str:
    """O campo `metodo`: as duas definições por extenso e as três contagens de ausência.

    'Faltante' e 'abundância' têm mais de uma leitura cada, e nos arquivos do laboratório as
    três leituras de 'faltante' coincidem por acaso. Sem o rótulo, a imagem passaria a
    significar outra coisa no primeiro dado com uma célula não finita.
    """
    zeros = sum(classe.n_zeros for classe in composicao.classes)
    nao_finitas = sum(classe.n_nao_finitas for classe in composicao.classes)
    faltantes = sum(classe.n_faltantes for classe in composicao.classes)
    return (
        f"FALTANTE = {definicao_lida(composicao.definicao_de_faltante, DEFINICOES_DO_FALTANTE)}; "
        f"na população desenhada são {milhar(faltantes)} células, {milhar(zeros)} por valor zero e "
        f"{milhar(nao_finitas)} por não finito. ABUNDÂNCIA = "
        f"{definicao_lida(composicao.definicao_de_abundancia, DEFINICOES_DA_ABUNDANCIA)}; o eixo "
        "traz o log10 e a anotação a intensidade bruta. Sem teste estatístico: as classes são "
        "descritas, não comparadas entre si"
    )


def figura_da_composicao_por_classe(
    *,
    composicao: ComposicaoDasClasses,
    recorte: Recorte,
    estilos: EstilosDosGrupos | None = None,
) -> Figure:
    """Os dois painéis da composição por classe, empilhados sobre o mesmo eixo de classes.

    Desenha a população inteira que recebe; não amostra.

    São dois eixos porque as unidades diferem: um é percentagem e o outro é log10 de
    intensidade. São empilhados, e não lado a lado, porque um painel de meia largura não
    comporta os nomes das classes com o `n`, o rótulo do eixo y e a legenda. Empilhado, cada
    painel tem a largura inteira.

    Os painéis partilham o eixo das classes, e a marca aparece uma vez, sob o painel de baixo.
    A legenda só existe no desfecho `AUSENTES`, em que nenhuma marca nomeia as classes.

    Zero classes não levanta: a figura sai com um eixo só e a causa escrita.
    """
    figura = nova_figura(altura=ALTURA_DA_COMPOSICAO)
    if not composicao.desenhavel:
        # Um eixo e não dois: com dois a mesma frase sairia duas vezes.
        _declara(figura.add_subplot(111), SEM_CLASSE_DECLARADA)
        figura.suptitle(TITULO_DA_COMPOSICAO)
        aplica(figura, recorte)
        return figura

    dos_faltantes = figura.add_subplot(211)
    # Os dois painéis descrevem a mesma partição, então partilham o eixo x.
    da_abundancia = figura.add_subplot(212, sharex=dos_faltantes)

    # Com estilo, a cor de cada barra é a do grupo, a mesma das figuras de pontos. O símbolo
    # não entra, porque barra não tem marcador. Sem estilo, o ciclo na ordem das classes.
    if estilos is None:
        cores = cores_do_ciclo(len(composicao.classes))
    else:
        do_grupo = estilos.para(classe.grupo for classe in composicao.classes)
        cores = tuple(do_grupo[classe.grupo].hex for classe in composicao.classes)
    marcas = marcas_das_classes(figura, dos_faltantes, composicao)
    _desenha_os_faltantes(dos_faltantes, composicao, cores, marcas)
    rotulou_embaixo = _desenha_a_abundancia(da_abundancia, composicao, cores, marcas)
    # Com `sharex` as marcas aparecem nos dois painéis; escondê-las em cima deixa uma só. Quando
    # o painel de baixo não rotula nada (nenhuma classe com detectada), as de cima ficam, senão
    # os nomes das classes sumiriam dos dois.
    if rotulou_embaixo:
        dos_faltantes.tick_params(axis="x", labelbottom=False)
        dos_faltantes.set_xlabel("")

    # A legenda só existe quando as marcas do eixo não nomeiam as classes; com as duas, cada
    # nome sairia duas vezes. Os artistas de prova não entram no eixo.
    if marcas is MarcasDasClasses.AUSENTES:
        legenda_fora(
            dos_faltantes,
            [
                Line2D([], [], color=cor, label=f"{classe.grupo} (n={milhar(classe.n_injecoes)})")
                for classe, cor in zip(composicao.classes, cores, strict=True)
            ],
        )
    figura.suptitle(TITULO_DA_COMPOSICAO)
    aplica(figura, recorte)
    return figura


def _declara(eixo: Axes, frase: str) -> None:
    """Um eixo sem barra nenhuma, com a causa escrita."""
    eixo.set_axis_off()
    texto = eixo.text(
        0.02,
        0.5,
        frase,
        transform=eixo.transAxes,
        ha="left",
        va="center",
        fontsize=TAMANHO_DA_FONTE,
    )
    texto.set_gid(GID_REQUEBRAVEL)
    # Fora do `tight_layout`: com o eixo desligado e só este texto, ainda numa linha, o primeiro
    # `tight_layout` de `recorte.aplica` emite o aviso de layout não aplicado. Só serve a um eixo
    # cuja única coisa é o texto.
    texto.set_in_layout(False)


def rotulos_das_classes_cabem(eixo: Axes, composicao: ComposicaoDasClasses) -> bool:
    """Se os nomes das classes cabem lado a lado neste eixo; senão têm de ser rodados.

    Não há limiar fixo de caracteres: a resposta depende da largura do eixo, do número de
    classes e do comprimento dos nomes. A conta usa `recorte.colunas_do_eixo`, que mede o eixo
    e não a figura, porque o rótulo do eixo y, as marcas e as margens tiram largura dele. Há
    uma folga de um caractere entre vizinhos.

    O que se mede é a marca que `marca_da_classe` escreve, linha a linha, e não o nome do
    grupo: a marca horizontal tem duas linhas e a do `n` pode ser a mais larga.
    """
    if not composicao.desenhavel:  # pragma: no cover - quem chama já ramificou por `desenhavel`
        return False
    maior = max(
        len(linha)
        for classe in composicao.classes
        for linha in marca_da_classe(classe, rodado=False).split("\n")
    )
    return colunas_do_eixo(eixo, TAMANHO_DA_FONTE) >= (maior + 1) * len(composicao.classes)


def marca_da_classe(classe: ClasseNaFigura, *, rodado: bool) -> str:
    """O texto de uma marca do eixo x: o nome da classe e o `n` de injeções dela.

    O `n` vai na marca porque é a população da barra acima dela, e a percentagem esconde que
    as classes têm denominadores diferentes. Rodado, ele vira sufixo na mesma linha, porque
    duas linhas rodadas ficam lado a lado e o `n` cairia sobre a classe vizinha.
    """
    separador = " " if rodado else "\n"
    return f"{classe.grupo}{separador}n={milhar(classe.n_injecoes)}"


def marcas_das_classes(
    figura: Figure, eixo: Axes, composicao: ComposicaoDasClasses
) -> MarcasDasClasses:
    """Qual dos três desfechos de marcação este eixo suporta.

    - `HORIZONTAIS`: os nomes cabem lado a lado.
    - `RODADAS`: não cabem na largura; horizontais, se sobreporiam.
    - `AUSENTES`: nem rodados cabem na altura; atravessariam o rodapé.

    A largura é decidida por `rotulos_das_classes_cabem` e a altura por
    `distribuicao.rotulos_cabem_na_altura`. É um vocabulário fechado, e não dois booleanos,
    porque o par 'não rodado e sem marca' não significa nada.
    """
    if rotulos_das_classes_cabem(eixo, composicao):
        return MarcasDasClasses.HORIZONTAIS
    rodadas = [marca_da_classe(classe, rodado=True) for classe in composicao.classes]
    if rotulos_cabem_na_altura(figura, rodadas):
        return MarcasDasClasses.RODADAS
    return MarcasDasClasses.AUSENTES


def _rotula_as_classes(
    eixo: Axes, composicao: ComposicaoDasClasses, marcas: MarcasDasClasses
) -> None:
    """Escreve as marcas do eixo x no desfecho que `marcas_das_classes` decidiu.

    Sem marcas, quem nomeia as classes é a legenda, e o `xlabel` diz onde procurar.
    """
    eixo.set_xticks(list(range(len(composicao.classes))))
    eixo.set_xlim(-0.6, len(composicao.classes) - 0.4)
    if marcas is MarcasDasClasses.AUSENTES:
        eixo.set_xticklabels([])
        eixo.set_xlabel(SEM_MARCAS_DE_CLASSE, fontsize=TAMANHO_DA_FONTE)
        return
    rodado = marcas is MarcasDasClasses.RODADAS
    eixo.set_xticklabels(
        [marca_da_classe(classe, rodado=rodado) for classe in composicao.classes],
        fontsize=TAMANHO_DA_FONTE,
    )
    if rodado:
        eixo.tick_params(axis="x", labelrotation=90, labelsize=TAMANHO_DA_FONTE)


def _desenha_os_faltantes(
    eixo: Axes,
    composicao: ComposicaoDasClasses,
    cores: tuple[Any, ...],
    marcas: MarcasDasClasses,
) -> None:
    """O painel de cima: uma barra por classe, em percentagem de células faltantes.

    Cada barra é uma chamada de `ax.vlines`. Uma chamada só daria uma `LineCollection` com um
    rótulo, e as classes colapsariam numa série.

    Não há guarda contra lista vazia: quem chama já ramificou por `desenhavel`.

    `fracao_faltante` devolve `nan` quando `n_celulas` é zero. Essa classe não ganha barra nem
    anotação e não entra no `max` dos limites: um `nan` ali tiraria as outras barras do eixo, e
    um segmento com `nan` não é legível pelos testes de figura. A ausência não é declarada em
    texto; a classe continua nomeada na marca do eixo, sem barra por cima.
    """
    alturas = [classe.fracao_faltante * 100.0 for classe in composicao.classes]
    for posicao, (classe, cor, altura) in enumerate(
        zip(composicao.classes, cores, alturas, strict=True)
    ):
        if not math.isfinite(altura):
            continue
        eixo.vlines(
            posicao,
            0.0,
            altura,
            colors=cor,
            linewidth=LARGURA_DA_BARRA_DA_CLASSE,
            label=rotulo_da_classe(classe.grupo, PAPEL_DOS_FALTANTES),
        )
        eixo.annotate(
            em_percentagem(classe.fracao_faltante),
            xy=(posicao, altura),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=TAMANHO_DA_FONTE,
        )

    _rotula_as_classes(eixo, composicao, marcas)
    # O eixo parte de zero: zero por cento de faltantes é a origem natural da quantidade, e
    # cortar a base exageraria diferenças de décimos de ponto.
    finitas = [altura for altura in alturas if math.isfinite(altura)]
    maior = max(finitas) if finitas else 0.0
    eixo.set_ylim(0.0, maior * (1.0 + FOLGA_DA_COMPOSICAO) if maior > 0.0 else 1.0)
    eixo.set_ylabel(ROTULO_DO_EIXO_DOS_FALTANTES)
    faltantes = sum(classe.n_faltantes for classe in composicao.classes)
    celulas = sum(classe.n_celulas for classe in composicao.classes)
    eixo.set_title(
        f"{TITULO_DOS_FALTANTES}\n{milhar(faltantes)} de {milhar(celulas)} células",
        fontsize=TAMANHO_DA_FONTE,
    )


def _desenha_a_abundancia(
    eixo: Axes,
    composicao: ComposicaoDasClasses,
    cores: tuple[Any, ...],
    marcas: MarcasDasClasses,
) -> bool:
    """O painel de baixo: a mediana do log10 das detectadas, uma barra por classe.

    A barra parte do piso do eixo e não de zero, e a base vai declarada no título. As medianas
    de log10 ficam próximas entre si, e barras desde zero teriam todas a mesma aparência.

    A classe sem detectada não vira barra; a ausência é declarada por `frase_sem_abundancia`.
    Quando nenhuma classe tem detectada, o eixo é desligado e só a frase aparece.

    Devolve se rotulou o eixo das classes. O painel de cima só esconde as próprias marcas
    quando este as desenhou; no desfecho sem barra nenhuma, este eixo não rotula nada.
    """
    com_barra = [classe for classe in composicao.classes if classe.tem_abundancia]
    if not com_barra:
        _declara(eixo, frase_sem_abundancia(composicao))
        eixo.set_title(TITULO_DA_ABUNDANCIA, fontsize=TAMANHO_DA_FONTE)
        return False

    piso, teto = _faixa_da_abundancia([classe.log10_mediana for classe in com_barra])
    for posicao, (classe, cor) in enumerate(zip(composicao.classes, cores, strict=True)):
        if not classe.tem_abundancia:
            continue
        eixo.vlines(
            posicao,
            piso,
            classe.log10_mediana,
            colors=cor,
            linewidth=LARGURA_DA_BARRA_DA_CLASSE,
            label=rotulo_da_classe(classe.grupo, PAPEL_DA_ABUNDANCIA),
        )
        # A intensidade em unidades brutas vai sobre a barra, porque é o número que se lê; o
        # eixo continua a ser o log.
        eixo.annotate(
            milhar(round(classe.intensidade_tipica)),
            xy=(posicao, classe.log10_mediana),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=TAMANHO_DA_FONTE,
        )

    if composicao.sem_abundancia:
        ausente = eixo.text(
            0.02,
            0.98,
            frase_sem_abundancia(composicao),
            transform=eixo.transAxes,
            ha="left",
            va="top",
            fontsize=TAMANHO_DA_FONTE,
        )
        ausente.set_gid(GID_REQUEBRAVEL)

    _rotula_as_classes(eixo, composicao, marcas)
    eixo.set_ylim(piso, teto)
    eixo.set_ylabel(ROTULO_DO_EIXO_DA_ABUNDANCIA)
    detectadas = sum(classe.n_detectadas for classe in com_barra)
    eixo.set_title(
        f"{TITULO_DA_ABUNDANCIA}\nmediana sobre {milhar(detectadas)} detectadas\n"
        f"base do eixo {decimal(piso, casas=3)}, NÃO zero",
        fontsize=TAMANHO_DA_FONTE,
    )
    return True


def _faixa_da_abundancia(valores: list[float]) -> tuple[float, float]:
    """O piso e o teto do eixo da abundância, a partir das medianas que vão ser desenhadas.

    A margem de cada lado é do tamanho do alcance, para a diferença entre classes ocupar cerca
    de um terço do eixo.

    Com alcance zero, que é o caso de uma classe só, a margem passa a ser 10% do próprio valor,
    ou 1 quando a mediana é exatamente zero. Sem esse ramo o piso e o teto coincidiriam.
    """
    menor, maior = min(valores), max(valores)
    alcance = maior - menor
    margem = alcance if alcance > 0.0 else (abs(menor) * 0.1 or 1.0)
    return menor - margem, maior + margem

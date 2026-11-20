"""O painel da correção: as barras de RSD de QC por corretor e a PCA antes e depois.

É uma figura só, com a disposição do `Bar Plot and PCA plot.png` que o pipeline R entrega
(`tools.R:1527-1600`): as barras em cima, na largura toda, e 'Before' e 'After' lado a lado
embaixo. A aparência segue o R: barras em ordem decrescente de mediana (`:1528`); cinza por
padrão, vermelho na penúltima, âmbar na melhor e preto no `none`, aplicado por último
(`:1529-1532`); valor escrito só sobre a barra do `none` e a da melhor, como
`signif(v, 4) * 100` (`:1554-1555`); amostra em marcador preto aberto e QC em vermelho cheio
(`:1573-1574`); 'raw data' sob o painel da esquerda e o nome do corretor sob o da direita
(`:1585`, `:1597`). Há duas diferenças deliberadas: os eixos de PCA trazem a variância
explicada, que o R esconde, e o valor sobre a barra mais alta cabe dentro do eixo, enquanto
no PNG do R ele é recortado. As cores são literais neste módulo porque codificam um fato (o
corretor, o tipo de amostra), em vez de só separar séries como nas outras figuras.

O estimador de cada barra é escolhido pelo servidor e a figura o imprime. Sem a medição dos
corretores não há fila de barras, e o lugar dela leva um texto que diz como obtê-la: o RSD
in-sample não serve de substituto, porque é medido nos QCs em que a correção ajustou a curva
e favorece quem ajusta. A população de cada barra não é servida, e o rodapé declara isso.
Os eixos dos dois painéis de PCA não são comparáveis: são duas decomposições de matrizes
diferentes, com sinal arbitrário em cada componente. O que se lê é o agrupamento dos QCs,
escrito no título de cada painel, e o rodapé diz se os dois lados retiveram o mesmo conjunto
de colunas. Cada painel leva a região de predição de Hotelling T² a 95% das amostras de
estudo, calculada sobre os próprios scores e declarada na legenda; ela descreve a dispersão
das amostras e não é teste de diferença entre grupos.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from matplotlib import transforms
from matplotlib.lines import Line2D

from refino.decision.policy import ABSTENTION_MARGIN
from refino.plots.base import (
    CORRETOR_NAO_DECLARADO,
    GID_REQUEBRAVEL,
    NAO_DECLARADA,
    REGUA_AUSENTE,
    MatrizVaziaError,
    denominador_ou_recusa,
    inteiro_do_fio,
    nova_figura,
    numero_do_fio,
    texto_do_fio,
    vetor_do_fio_ou_nucleo,
)
from refino.plots.pca import (
    ALFA_DA_ELIPSE,
    MINIMO_PARA_ELIPSE,
    Elipse,
    ElipseAusente,
    Escores,
    TipoDeAmostra,
    elipse_das_amostras,
    escores_da_pca,
)
from refino.plots.recorte import (
    TAMANHO_DA_FONTE,
    Recorte,
    aplica,
    decimal,
    legenda_fora,
    milhar,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "ALTURA_DO_PAINEL",
    "AMBAR_DO_MELHOR",
    "CINZA_POR_OMISSAO",
    "CORES_DA_PCA",
    "COR_DA_REGIAO",
    "COR_DO_INTERVALO",
    "DICA_DE_MEDIR",
    "INTERVALO_FALTANTE",
    "INTERVALO_MALFORMADO",
    "LARGURA_DA_BARRA",
    "NOME_DO_NONE",
    "PRETO_DO_NONE",
    "ROTULO_ANTES",
    "ROTULO_DA_REGIAO",
    "ROTULO_DEPOIS",
    "ROTULO_DO_EIXO_DAS_BARRAS",
    "ROTULO_DO_EIXO_DO_ANTES",
    "ROTULO_DO_INTERVALO",
    "ROTULO_SEM_QC",
    "TITULO_DAS_BARRAS",
    "TITULO_DO_PAINEL",
    "VERMELHO_DO_SEGUNDO",
    "Barra",
    "FilaAusente",
    "FilaDeBarras",
    "IntervaloAusente",
    "IntervaloDeBarra",
    "LadoDaCorrecao",
    "ParDaCorrecao",
    "cores_da_fila",
    "figura_do_painel_da_correcao",
    "fila_do_fio",
    "frase_do_lado",
    "intervalo_do_fio",
    "lado_da_correcao",
    "par_da_correcao",
    "percentagem_do_r",
    "recorte_do_painel_da_correcao",
    "rotulo_da_regiao",
    "rotulo_da_serie",
]

TITULO_DO_PAINEL: Final = "Painel da correção — RSD de QC por corretor, e a PCA antes e depois"

# `main="QC RSD"` e `ylab="RSD (%)"` em `tools.R:1553`. O eixo vai em percentagem, como o do
# R; `metrics/rsd.py` devolve fração, e a conversão é feita no desenho.
TITULO_DAS_BARRAS: Final = "RSD de QC por corretor"
ROTULO_DO_EIXO_DAS_BARRAS: Final = "RSD de QC (%)"

ROTULO_ANTES: Final = "Before"
ROTULO_DEPOIS: Final = "After"

# `xlab='raw data'` à esquerda (`tools.R:1585`). À direita o R escreve o nome do corretor
# (`:1597`), que aqui vem de `ParDaCorrecao.corretor`.
ROTULO_DO_EIXO_DO_ANTES: Final = "raw data"


# O nome do corretor que o `tools.R` pinta de preto, e que é a linha de base de toda a fila.
NOME_DO_NONE: Final = "none"

# As quatro cores da fila (`tools.R:1529-1532`). Aqui a cor codifica um fato. São constantes
# porque o desenho e os testes leem o mesmo valor.
CINZA_POR_OMISSAO: Final = "grey"
VERMELHO_DO_SEGUNDO: Final = "red"
AMBAR_DO_MELHOR: Final = "#ffbf00"
PRETO_DO_NONE: Final = "black"

# QC em vermelho cheio (`pch=16`) e amostra em preto aberto (`pch=1`), como em
# `tools.R:1573-1574`. A terceira cor é para `validate`, para a série não sair da cor de uma
# das outras duas quando houver amostra desse tipo.
CORES_DA_PCA: Final[Mapping[TipoDeAmostra, str]] = {
    TipoDeAmostra.QC: "#d62728",
    TipoDeAmostra.AMOSTRA: "black",
    TipoDeAmostra.VALIDACAO: "#1f77b4",
}

# Os tipos desenhados com marcador cheio. O R usa `pch=16` para `qc` e para os tipos de
# validação, e `pch=1`, aberto, para `sample`.
CHEIOS: Final = (TipoDeAmostra.QC, TipoDeAmostra.VALIDACAO)

# A ordem de desenho. O QC vem por último para ficar por cima: a pergunta da figura é se o QC
# agrupou, e ele não pode ficar escondido sob as amostras.
ORDEM_DAS_SERIES: Final = (TipoDeAmostra.VALIDACAO, TipoDeAmostra.AMOSTRA, TipoDeAmostra.QC)

_ROTULO_DO_TIPO: Final[Mapping[TipoDeAmostra, str]] = {
    TipoDeAmostra.QC: "QC",
    TipoDeAmostra.AMOSTRA: "amostra de estudo",
    TipoDeAmostra.VALIDACAO: "amostra de validação",
}

# A forma curta, para a segunda linha do título de um painel, que tem um terço da largura da
# figura. O texto longo fica na legenda.
_ROTULO_CURTO: Final[Mapping[TipoDeAmostra, str]] = {
    TipoDeAmostra.QC: "QC",
    TipoDeAmostra.AMOSTRA: "amostras",
    TipoDeAmostra.VALIDACAO: "validação",
}

# O rótulo da região de predição de cada painel, que vai na legenda. O nível sai de
# `ALFA_DA_ELIPSE`, a mesma constante que `pca.elipse_das_amostras` usa no cálculo, para o
# número da legenda e o quantil do desenho não discordarem.
#
# É curto de propósito: a forma por extenso alargava a legenda, estreitava as duas PCAs e
# fazia a legenda sair da figura no menor tamanho de painel da janela.
ROTULO_DA_REGIAO: Final = f"Região de predição {(1.0 - ALFA_DA_ELIPSE) * 100:.0f}% (T²)"

# Cinza, porque a região não é uma das séries. Sem cor explícita o `matplotlib` daria a
# primeira do ciclo, o azul das amostras de validação.
COR_DA_REGIAO: Final = "#666666"

# Tracejado, como a elipse de `pca.figura_dos_scores`: uma linha contínua seria lida como
# dado. É constante porque a linha e a entrada de legenda usam o mesmo valor.
TRACO_DA_REGIAO: Final = "--"

# Largura da barra, em pontos de linha. A barra é um segmento de `ax.vlines`, e a espessura o
# faz parecer a barra do `barplot` do R. Não se usa `ax.bar`: ele produz `Rectangle` em
# `ax.patches`, que o arnês de teste e a identificação por clique não leem. `ax.vlines` dá
# uma `LineCollection`, com `get_segments`.
LARGURA_DA_BARRA: Final = 26.0

# Folga acima da barra mais alta, em fração do alcance, para o valor escrito sobre ela ficar
# dentro do eixo. No PNG do R o valor do `none` cai acima do topo do eixo e é recortado.
FOLGA_DO_ROTULO: Final = 0.12

# Polegadas. É a única figura do pacote que não usa `base.ALTURA_PADRAO`, porque tem três
# eixos em duas linhas. A largura é a padrão.
ALTURA_DO_PAINEL: Final = 7.0

# A repartição vertical entre a fila de barras e a linha dos dois painéis de PCA. As barras
# precisam de menos altura que as duas nuvens.
#
# Não há `hspace` no `gridspec`: com ele o `tight_layout` considera os eixos incompatíveis e
# não é aplicado. A folga entre as duas linhas é dada pelo próprio `tight_layout`.
ALTURAS_DAS_LINHAS: Final = (1.0, 1.6)

# A linha de cima encolhe quando não há fila, porque leva só o texto de `DICA_DE_MEDIR`. 0,7
# é o menor valor que deixa o texto de cinco linhas afastado do título da figura e dos
# títulos das PCAs no tamanho de construção.
ALTURAS_SEM_FILA: Final = (0.7, 1.6)

ROTULO_SEM_QC: Final = "sem QC nesta matriz: não há régua técnica a desenhar neste painel"

# O texto diz onde clicar e que a medição leva minutos. O tempo em si não é escrito: depende
# da matriz e do número de núcleos, e a janela mostra a estimativa ao lado do botão. O botão
# e a aba citados são conferidos contra este texto por um teste da janela.
#
# É curto porque a linha que o leva é a encolhida de `ALTURAS_SEM_FILA`. O motivo completo
# vai no rodapé.
DICA_DE_MEDIR: Final = (
    "sem fila de barras: a medição dos corretores não correu para esta matriz. Para a ter, clique "
    "em «Medir qual corretor vence» na aba «Correção» — leva minutos, e a "
    "estimativa para esta matriz aparece logo abaixo do botão. O RSD in-sample não a "
    "substitui: medido nos QCs em que a correção ajustou a curva, ele favorece quem a ajusta"
)

SEM_RANKING: Final = (
    "sem fila de barras: a medição correu e o RSD ficou indefinido em todos os corretores, então "
    "não houve comparação a fazer. É outro fato que «a medição não correu»"
)

# A causa do corretor recusado, na forma curta que cabe no título das barras. O motivo
# completo vai no rodapé.
CAUSA_DA_RECUSA: Final = "QCs insuficientes no lote"


# O que o rodapé escreve no lugar da população de cada barra. Ver o cabeçalho deste módulo.
POPULACAO_NAO_SERVIDA: Final = (
    "população de cada barra: NÃO SERVIDA — o oráculo reduz o vetor por feature a uma mediana e "
    "descarta a contagem de finitos; o que o corpo traz é o teto pedido ao amostrador, não a "
    "população"
)


# ---------------------------------------------------------------------------
# O intervalo por barra, opcional
# ---------------------------------------------------------------------------
#
# Só existe quando a medição o trouxe, e ela só o traz quando a pessoa o pediu, porque custa
# `(n+1)/2` vezes a medição. Sem ele a figura não ganha nenhum artista nem texto de rodapé.
#
# O rodapé diz o que ele mede. O jackknife-t sobre os QCs mostra quanto a barra depende de
# quais QCs foram injetados: cada replicado refaz a régua sem um QC, com reajuste. Não mostra
# a dependência das features, que são fixas no estudo, e o nível é nominal.
#
# A margem entre os dois melhores vai ao lado, comparada com `ABSTENTION_MARGIN`, que é o
# limiar de abstenção da medição. O limiar do palpite é sobre uma amostra de features, outra
# população, e não aparece aqui.

ROTULO_DO_INTERVALO: Final = "intervalo jackknife-t"
"""O começo do rótulo da série de cada intervalo; o resto é o nome do corretor."""

# Preto, fino e ao lado da barra. No centro o traço precisaria de contorno para ser lido
# sobre a barra preta do `none`, e o contorno cobriria o topo da barra. Ao lado ele fica
# sobre o fundo branco, seja qual for a cor da barra.
COR_DO_INTERVALO: Final = "black"
LARGURA_DO_INTERVALO: Final = 1.4
# As tampas são o marcador `_` nas duas pontas, com tamanho em pontos, que não muda com o
# tamanho da figura.
TAMANHO_DA_TAMPA: Final = 7.0
# O deslocamento do traço para a direita do centro da barra, em pontos: meia barra
# (`LARGURA_DA_BARRA`) mais uma folga, para não tocar a barra em tamanho nenhum.
DESLOCAMENTO_DO_INTERVALO: Final = LARGURA_DA_BARRA / 2.0 + 4.0

INTERVALO_MALFORMADO: Final = (
    "o corpo trouxe um intervalo sem os dois limites finitos, com o inferior acima do superior, "
    "ou com o nível fora de (0, 1), e sem motivo"
)
INTERVALO_FALTANTE: Final = "a medição pediu o intervalo e o corpo não o trouxe para esta barra"


@dataclass(frozen=True)
class IntervaloDeBarra:
    """O intervalo de uma barra, em fração, como `Barra.rsd`."""

    inferior: float
    superior: float
    nivel: float
    n_qc: int
    metodo: str


@dataclass(frozen=True)
class IntervaloAusente:
    """A barra não leva intervalo, e o motivo.

    Um traço de largura zero no lugar dele seria lido como certeza.
    """

    motivo: str


def intervalo_do_fio(
    *,
    inferior: Any,
    superior: Any,
    nivel: Any,
    n_qc: Any,
    metodo: Any,
    motivo: Any,
) -> IntervaloDeBarra | IntervaloAusente:
    """O intervalo de uma barra vindo de `MedicaoDoCorretorResposta.intervalos`, com as guardas.

    O motivo, quando vem, prevalece: é o servidor dizendo que não houve intervalo. Sem motivo,
    os dois limites têm de ser finitos, o inferior não pode passar do superior e o nível tem
    de estar em (0, 1). O que não confere vira `IntervaloAusente`.
    """
    if isinstance(motivo, str) and motivo.strip():
        return IntervaloAusente(motivo.strip())
    baixo, alto, nv = numero_do_fio(inferior), numero_do_fio(superior), numero_do_fio(nivel)
    if not (math.isfinite(baixo) and math.isfinite(alto) and baixo <= alto and 0.0 < nv < 1.0):
        return IntervaloAusente(INTERVALO_MALFORMADO)
    return IntervaloDeBarra(
        inferior=baixo,
        superior=alto,
        nivel=nv,
        n_qc=inteiro_do_fio(n_qc),
        metodo=texto_do_fio(metodo, REGUA_AUSENTE),
    )


# A forma curta de `POPULACAO_NAO_SERVIDA`, usada quando o rodapé também leva a frase do
# intervalo ou de uma recusa.
POPULACAO_NAO_SERVIDA_CURTA: Final = "população de cada barra: NÃO SERVIDA"


# ---------------------------------------------------------------------------
# A fila de barras
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Barra:
    """Um corretor e o RSD de QC dele, em fração.

    A conversão para percentagem é feita uma vez só, no desenho, que é onde o R também
    converte.
    """

    corretor: str
    rsd: float


@dataclass(frozen=True)
class FilaDeBarras:
    """As barras já ordenadas, com os rótulos que as tornam legíveis.

    `barras` vem em ordem decrescente de `rsd`, a pior primeiro, como em `tools.R:1528`. O
    empate é resolvido pelo nome, para a figura não depender da ordem do dicionário recebido.
    """

    barras: tuple[Barra, ...]
    nao_medidos: tuple[str, ...]
    """Os corretores que vieram com RSD indefinido. Não viram barra, e são citados no título:
    uma barra de altura zero pareceria uma correção perfeita."""

    vencedor: str
    abstem: bool
    estimador: str
    variante: str

    teto_do_amostrador: int | None
    """O `n_features_medidas` do corpo: o teto pedido ao amostrador, e não a população em que a
    mediana foi calculada. Não é desenhado."""

    intervalos: Mapping[str, IntervaloDeBarra | IntervaloAusente] | None = None
    """O intervalo de cada barra, pelo nome do corretor. `None` quando a medição não o pediu;
    quando pediu, toda barra tem uma entrada, com o intervalo ou com o motivo da ausência."""

    margem: float | None = None
    """A margem relativa entre os dois melhores. Só é escrita junto com o intervalo. `None`
    quando não há segundo colocado."""

    recusados: Mapping[str, str] = field(default_factory=dict)
    """Os corretores que a medição não mediu, com o motivo de cada um.

    Não viram barra e não entram em `nao_medidos`: o indefinido é o que a medição devolveu, e
    o recusado é o que ela não fez. O título cita cada caso com uma frase própria, e o rodapé
    traz o motivo completo."""

    @property
    def melhor(self) -> Barra | None:
        """A barra do `vencedor`, ou `None` quando ele não tem barra.

        Isso acontece quando o RSD do vencedor veio indefinido: não há barra a destacar nem
        valor a escrever.
        """
        for barra in self.barras:
            if barra.corretor == self.vencedor:
                return barra
        return None


@dataclass(frozen=True)
class FilaAusente:
    """A fila não é desenhada, e o motivo.

    É um tipo próprio, e não `None`, porque 'a medição não correu' e 'correu e não foi
    avaliável' são fatos diferentes. O motivo é o texto que a figura escreve no lugar das
    barras.
    """

    motivo: str


def fila_do_fio(
    *,
    scores: Any,
    vencedor: Any,
    abstem: Any,
    estimador: Any,
    variante: Any,
    n_features_medidas: Any,
    intervalos: Any = None,
    margem: Any = None,
    recusas: Any = None,
) -> FilaDeBarras | FilaAusente:
    """A fila vinda do corpo de `POST /matrizes/medir-corretor`, com as guardas de fronteira.

    `nan` é serializado como `null`, então uma barra pode chegar `None`. Esse corretor vai
    para `nao_medidos`, e não vira barra de altura zero.

    Devolve `FilaAusente` em vez de levantar, porque as duas causas são fatos legítimos: a
    medição não correu, ou correu e nenhum corretor deu número.

    `recusas` traz os corretores que a medição não mediu, com o motivo. Só entra o que for
    texto não vazio, e o motivo prevalece sobre um número que venha junto.
    """
    valores = scores if isinstance(scores, dict) else {}
    if not valores:
        return FilaAusente(DICA_DE_MEDIR)
    recusados = _recusas_do_fio(recusas)

    medidas: list[Barra] = []
    ausentes: list[str] = []
    for nome, bruto in valores.items():
        if str(nome) in recusados:
            continue
        rsd = numero_do_fio(bruto)
        if math.isfinite(rsd):
            medidas.append(Barra(corretor=str(nome), rsd=rsd))
        else:
            ausentes.append(str(nome))
    if not medidas:
        if not recusados:
            return FilaAusente(SEM_RANKING)
        # Com um corretor recusado, 'indefinido em todos' seria falso: o texto diz os dois
        # fatos. A janela não chega a este ramo, porque só passa o corpo de uma medição
        # avaliável; ele serve a quem chama a função diretamente.
        indefinidos = ", ".join(sorted(ausentes)) or "nenhum"
        return FilaAusente(
            "sem fila de barras: o RSD ficou indefinido em "
            f"{indefinidos}, e "
            + "; ".join(f"{nome} não foi medido: {motivo}" for nome, motivo in recusados.items())
        )

    # Decrescente por RSD, com o empate resolvido pelo nome. O `sort(decreasing = TRUE)` de
    # `tools.R:1528` mantém a ordem de entrada nos empates; aqui a entrada é um dicionário
    # vindo de JSON, e o desempate tem de ser explícito.
    ordenadas = sorted(medidas, key=lambda b: (-b.rsd, b.corretor))
    # Uma entrada por barra. A que falta no corpo recebe `INTERVALO_FALTANTE`, para não passar
    # por barra sem intervalo pedido.
    por_barra: dict[str, IntervaloDeBarra | IntervaloAusente] | None = None
    if isinstance(intervalos, Mapping):
        por_barra = {
            barra.corretor: (
                intervalos[barra.corretor]
                if isinstance(intervalos.get(barra.corretor), IntervaloDeBarra | IntervaloAusente)
                else IntervaloAusente(INTERVALO_FALTANTE)
            )
            for barra in ordenadas
        }
    valor_da_margem = numero_do_fio(margem)
    return FilaDeBarras(
        barras=tuple(ordenadas),
        nao_medidos=tuple(sorted(ausentes)),
        vencedor=texto_do_fio(vencedor, ""),
        abstem=bool(abstem),
        # O recuo é `REGUA_AUSENTE`, e não `NAO_DECLARADA`, que fala da matriz.
        estimador=texto_do_fio(estimador, REGUA_AUSENTE),
        variante=texto_do_fio(variante, REGUA_AUSENTE),
        teto_do_amostrador=(
            None if n_features_medidas is None else inteiro_do_fio(n_features_medidas) or None
        ),
        intervalos=por_barra,
        margem=valor_da_margem if math.isfinite(valor_da_margem) else None,
        recusados=recusados,
    )


def _recusas_do_fio(recusas: Any) -> dict[str, str]:
    """O `recusas` do corpo como `{corretor: motivo}`, só com motivo de texto não vazio."""
    if not isinstance(recusas, Mapping):
        return {}
    return {
        str(nome): motivo.strip()
        for nome, motivo in recusas.items()
        if isinstance(motivo, str) and motivo.strip()
    }


def cores_da_fila(fila: FilaDeBarras) -> tuple[str, ...]:
    """Uma cor por barra, na ordem da fila, com a ordem de aplicação de `tools.R:1529-1532`.

    Cinza por padrão, vermelho na penúltima, âmbar na melhor e preto no `none`. O preto é
    aplicado por último e prevalece sobre o âmbar quando o `none` é o vencedor: a linha de
    base tem de continuar reconhecível.

    Com `abstem`, o âmbar não é aplicado. Destacar a barra mais baixa afirmaria uma eleição
    que a política recusou.
    """
    escolhidas = [CINZA_POR_OMISSAO] * len(fila.barras)
    if len(escolhidas) >= 2:
        escolhidas[-2] = VERMELHO_DO_SEGUNDO
    if not fila.abstem:
        for posicao, barra in enumerate(fila.barras):
            if barra.corretor == fila.vencedor:
                escolhidas[posicao] = AMBAR_DO_MELHOR
    for posicao, barra in enumerate(fila.barras):
        if barra.corretor == NOME_DO_NONE:
            escolhidas[posicao] = PRETO_DO_NONE
    return tuple(escolhidas)


def percentagem_do_r(rsd: float) -> str:
    """O valor escrito sobre a barra, pela conta de `tools.R:1554`: `0.69299` vira `"69,3%"`.

    A conta é `signif(v, 4) * 100`: quatro algarismos significativos da fração, e só depois a
    percentagem. Na ordem inversa sairia `69,30%`. Assim o número é a mesma cadeia que aparece
    no PNG do R. A vírgula decimal é a de `recorte.decimal`.
    """
    if not math.isfinite(rsd):  # pragma: no cover - `fila_do_fio` já separou os não medidos
        return "indefinido"
    return f"{float(f'{rsd:.4g}') * 100:g}".replace(".", ",") + "%"


# ---------------------------------------------------------------------------
# Os dois lados da PCA
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LadoDaCorrecao:
    """Um painel de PCA: os pontos, a variância, o agrupamento e o par contagem e digest.

    Com `n_features_usadas` e `digest` a figura pode dizer se uma queda de dispersão veio do
    agrupamento ou de o lado 'depois' ter retido outro conjunto de colunas.
    """

    rotulo: str
    matriz: str
    escores: Escores
    variancia: np.ndarray
    dispersao: float
    """`qc_dispersion_ratio`, ou `nan`. Zero afirmaria agrupamento perfeito; a razão é
    indefinida quando as amostras estão todas no mesmo ponto."""

    n_features_usadas: int
    n_features_totais: int
    digest: str


def lado_da_correcao(
    *,
    rotulo: str,
    matriz: Any,
    pc1: Any,
    pc2: Any,
    tipos: Any,
    variancia_explicada: Any,
    qc_dispersion_ratio: Any,
    n_features_usadas: Any,
    n_features_totais: Any,
    digest_das_colunas_retidas: Any,
) -> LadoDaCorrecao:
    """Um lado vindo do corpo, com as guardas de fronteira.

    O tipo de amostra é validado por `pca.escores_da_pca`: um valor fora dos três tipos é
    recusado, em vez de a amostra sumir da figura.
    """
    return LadoDaCorrecao(
        rotulo=rotulo,
        matriz=texto_do_fio(matriz, NAO_DECLARADA),
        escores=escores_da_pca(pc1=pc1, pc2=pc2, tipos=tipos),
        variancia=vetor_do_fio_ou_nucleo(variancia_explicada),
        dispersao=numero_do_fio(qc_dispersion_ratio),
        n_features_usadas=inteiro_do_fio(n_features_usadas),
        n_features_totais=inteiro_do_fio(n_features_totais),
        digest=texto_do_fio(digest_das_colunas_retidas, NAO_DECLARADA),
    )


@dataclass(frozen=True)
class ParDaCorrecao:
    """Os dois painéis, mais os rótulos comuns a eles.

    `pretransformacao` e `n_componentes` são do par, e não de cada lado, porque as duas PCAs
    correm com o mesmo regime.
    """

    antes: LadoDaCorrecao
    depois: LadoDaCorrecao
    pretransformacao: str
    n_componentes: int

    corretor: str
    """O corretor que produziu a matriz do lado 'depois'. É o `xlab` do painel direito do R.

    É do par, e não de um lado: um campo por lado pediria um valor para o 'antes', e `none`
    faria a figura afirmar que a matriz ingerida passou por uma correção.

    É o corretor que correu, e não o que venceu a fila. O R escreve ali o vencedor porque o
    painel 'After' dele desenha a matriz do vencedor; aqui o painel desenha a matriz que a
    execução produziu.
    """

    @property
    def espacos_divergem(self) -> bool:
        """Se os dois painéis retiveram conjuntos de colunas diferentes.

        Compara o digest, e não a contagem: uma feature que entra enquanto outra sai mantém o
        número e muda o espaço.
        """
        return self.antes.digest != self.depois.digest


def par_da_correcao(
    *,
    antes: LadoDaCorrecao,
    depois: LadoDaCorrecao,
    pretransformacao: Any,
    n_componentes: Any,
    corretor: Any,
) -> ParDaCorrecao:
    """O par, com os três rótulos passados pela guarda de fronteira.

    `corretor` não é conferido contra o vocabulário aqui: essa validação é do esquema da API.
    O que a função garante é o recuo, `CORRETOR_NAO_DECLARADO`, que não é nome de corretor
    nenhum.
    """
    return ParDaCorrecao(
        antes=antes,
        depois=depois,
        pretransformacao=texto_do_fio(pretransformacao, NAO_DECLARADA),
        n_componentes=inteiro_do_fio(n_componentes),
        corretor=texto_do_fio(corretor, CORRETOR_NAO_DECLARADO),
    )


# ---------------------------------------------------------------------------
# Os rótulos e as frases
# ---------------------------------------------------------------------------


def rotulo_da_serie(tipo: TipoDeAmostra, lado: str) -> str:
    """O rótulo de uma série de pontos: `(QC, "Before")` vira `"QC — Before"`.

    É único entre os dois painéis porque o arnês de teste indexa as séries por rótulo.
    """
    return f"{_ROTULO_DO_TIPO[tipo]} — {lado}"


def rotulo_da_regiao(lado: str) -> str:
    """O rótulo da linha da região: `"Região de predição 95% (T²) — Before"`.

    É único por painel, pela mesma razão de `rotulo_da_serie`. A legenda leva uma entrada só,
    com `ROTULO_DA_REGIAO`.
    """
    return f"{ROTULO_DA_REGIAO} — {lado}"


def frase_do_lado(lado: LadoDaCorrecao) -> str:
    """A segunda linha do título de um painel: o agrupamento dos QCs e a contagem por série.

    O `qc_dispersion_ratio` fica no título do painel, junto da nuvem a que pertence, porque é
    o número que responde se o QC agrupou.
    """
    contagens = []
    for tipo in (TipoDeAmostra.QC, TipoDeAmostra.AMOSTRA, TipoDeAmostra.VALIDACAO):
        quantas = int(lado.escores.mascara(tipo).sum())
        if quantas:
            contagens.append(f"{milhar(quantas)} {_ROTULO_CURTO[tipo]}")
    dispersao = (
        "agrupamento de QC indefinido"
        if not math.isfinite(lado.dispersao)
        else f"agrupamento de QC {decimal(lado.dispersao)}"
    )
    return f"{dispersao} · {' · '.join(contagens) if contagens else 'nada desenhável'}"


def _frase_das_barras(fila: FilaDeBarras) -> str:
    """A segunda linha do título das barras.

    É curta porque `recorte.aplica` dá a cada título um terço da largura da figura: ela tem
    três colunas de eixos, e o título das barras entra na mesma conta. O estimador e a variante
    ficam no rodapé, que quebra em linhas.
    """
    plural = "corretor" if len(fila.barras) == 1 else "corretores"
    corpo = f"{milhar(len(fila.barras))} {plural}"
    if fila.nao_medidos:
        corpo += f" · sem barra, RSD indefinido: {', '.join(fila.nao_medidos)}"
    if fila.recusados:
        corpo += f" · sem barra, {CAUSA_DA_RECUSA}: {', '.join(sorted(fila.recusados))}"
    if fila.abstem:
        corpo += " · a política ABSTEVE-SE: nenhum eleito"
    return corpo


# ---------------------------------------------------------------------------
# O rodapé
# ---------------------------------------------------------------------------


def recorte_do_painel_da_correcao(
    *, par: ParDaCorrecao, fila: FilaDeBarras | FilaAusente
) -> Recorte:
    """O rodapé: a população de cada painel, a não-comparabilidade e a régua das barras.

    O par de contagens do topo é o do painel 'Before', e o texto diz isso, porque o formato do
    recorte só tem lugar para um par e a figura tem duas populações.

    Sem execução, `denominador_ou_recusa` levanta `MatrizVaziaError`.
    """
    return Recorte(
        desenhadas=par.antes.n_features_usadas,
        disponiveis=denominador_ou_recusa(par.antes.n_features_totais, de="a PCA do «Before»"),
        amostragem=(
            f"nenhuma; o par acima é do «{par.antes.rotulo}», e o «{par.depois.rotulo}» reteve "
            f"{milhar(par.depois.n_features_usadas)} de "
            f"{milhar(par.depois.n_features_totais)} — espaços {_frase_dos_espacos(par)}"
        ),
        matriz=(f"duas — «{par.antes.matriz}» à esquerda e «{par.depois.matriz}» à direita"),
        metodo=_metodo_do_painel(par, fila),
    )


def _frase_dos_espacos(par: ParDaCorrecao) -> str:
    """A frase que diz se os dois painéis retiveram o mesmo conjunto de colunas.

    Cita o digest dos dois lados mesmo quando coincidem, para a afirmação poder ser conferida.
    """
    marca = f"{par.antes.digest[:8]}/{par.depois.digest[:8]}"
    if not par.espacos_divergem:
        return f"COINCIDENTES ({marca})"
    return (
        f"DIVERGENTES ({marca}): parte de uma queda de dispersão pode vir do descarte e não do "
        "agrupamento"
    )


def _metodo_do_painel(par: ParDaCorrecao, fila: FilaDeBarras | FilaAusente) -> str:
    """O campo `metodo`: a PCA de cada painel, a não-comparabilidade e a régua das barras.

    A frase da não-comparabilidade é obrigatória: são duas decomposições de matrizes
    diferentes, com sinal arbitrário, e sem ela um ponto que mudou de lado seria lido como
    fato do dado.
    """
    pca = (
        f"duas PCAs independentes, «{par.pretransformacao}» + autoescalonamento + SVD, "
        f"{milhar(par.n_componentes)} componentes, sem coluna constante nem não finita. Os eixos "
        "dos dois painéis NÃO são comparáveis: leia agrupamento, nunca o lado nem a escala"
    )
    if isinstance(fila, FilaAusente):
        return f"{pca}. Barras: {fila.motivo}"
    barras = (
        f"{pca}. Barras: mediana do RSD de QC, estimador «{fila.estimador}», variante "
        f"«{fila.variante}», do pior para o melhor; "
    )
    # O motivo de cada corretor recusado, uma frase por corretor, na mesma forma do 'Sem
    # traço em' do intervalo.
    recusas = "".join(
        f". Sem barra em {nome}: {motivo}" for nome, motivo in sorted(fila.recusados.items())
    )
    if fila.intervalos is None and not recusas:
        return barras + POPULACAO_NAO_SERVIDA
    # Com recusa ou intervalo, a população passa à forma curta: no menor tamanho de painel da
    # janela o rodapé mais longo já ocupa todas as linhas que cabem.
    if fila.intervalos is None:
        return f"{barras}{POPULACAO_NAO_SERVIDA_CURTA}{recusas}"
    return f"{barras}{POPULACAO_NAO_SERVIDA_CURTA}{recusas}. {_frase_do_intervalo(fila)}"


def _frase_do_intervalo(fila: FilaDeBarras) -> str:
    """O que o traço mede, as barras sem ele com o motivo, e a margem com o limiar da medição.

    A frase é curta porque, no menor tamanho de painel da janela, o rodapé sem o intervalo já
    ocupa as linhas que cabem em `RESERVA_MAXIMA`. Com o intervalo, a população de cada barra
    passa à forma curta, e esta frase usa o espaço liberado. A palavra 'nominal' diz que a
    cobertura não foi medida.

    Um motivo comum a todas as barras aparece uma vez, sem a lista de corretores. Com barras
    com e sem traço na mesma figura, o rodapé pode passar da faixa naquele tamanho. O caso
    não é raro: num lote de exatamente quatro QCs, o LOESS exige cinco por lote e o `none` e o
    SERRF quatro (`metrics.crossval.jackknife_qc_minimum`).

    A margem é relativa, `(segundo - melhor) / |melhor|`, e a frase diz isso.
    """
    assert fila.intervalos is not None  # pragma: no cover - só se chama com o intervalo pedido
    desenhados = [i for i in fila.intervalos.values() if isinstance(i, IntervaloDeBarra)]
    ausentes: dict[str, list[str]] = {}
    for corretor, intervalo in fila.intervalos.items():
        if isinstance(intervalo, IntervaloAusente):
            ausentes.setdefault(intervalo.motivo, []).append(corretor)
    frases: list[str] = []
    if desenhados:
        primeiro = desenhados[0]
        frases.append(
            f"Traço: {primeiro.metodo} sobre {milhar(primeiro.n_qc)} QCs, "
            f"{primeiro.nivel * 100:.0f}% nominal — sensível aos QCs injetados, não às features"
        )
    if not desenhados and len(ausentes) == 1:
        frases.append(f"Sem traço: {next(iter(ausentes))}")
    else:
        frases.extend(
            f"Sem traço em {', '.join(corretores)}: {motivo}"
            for motivo, corretores in ausentes.items()
        )
    # A frase diz 'relativa' porque, ao lado de barras em % de RSD, um percentual sem a
    # palavra seria lido como a diferença entre as barras em pontos percentuais.
    margem = (
        "indefinida"
        if fila.margem is None
        else f"{decimal(fila.margem * 100.0, casas=3)}% do melhor"
    )
    frases.append(
        f"Margem relativa dos dois melhores: {margem}; a medição abstém-se abaixo de "
        f"{decimal(ABSTENTION_MARGIN * 100.0)}%"
    )
    return ". ".join(frases)


# ---------------------------------------------------------------------------
# A figura
# ---------------------------------------------------------------------------


def figura_do_painel_da_correcao(
    *,
    par: ParDaCorrecao,
    fila: FilaDeBarras | FilaAusente,
    recorte: Recorte,
) -> Figure:
    """A fila de barras em cima e as duas PCAs embaixo, na disposição de `tools.R:1548`.

    São três eixos numa figura: a primeira linha do `gridspec` é um eixo que ocupa as duas
    colunas, como o `layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE))` do R.

    Sem a medição dos corretores a figura sai do mesmo jeito, e o lugar da fila leva o motivo
    em texto, marcado como requebrável. Cada PCA leva a região de predição, declarada na
    legenda.

    A altura não é a padrão do pacote: com três eixos em duas linhas e o rodapé mais carregado,
    a altura padrão faz títulos e marcas se sobreporem. Ela vale para a construção e para a
    exportação; na janela, a figura é redimensionada ao painel e o rodapé é refeito.
    """
    figura = nova_figura(altura=ALTURA_DO_PAINEL)
    # `height_ratios` em vez de metade para cada linha, como faz o R: as duas nuvens precisam
    # de mais altura que as barras. Sem `hspace`; ver `ALTURAS_DAS_LINHAS`.
    alturas = ALTURAS_SEM_FILA if isinstance(fila, FilaAusente) else ALTURAS_DAS_LINHAS
    grelha = figura.add_gridspec(2, 2, height_ratios=alturas)
    das_barras = figura.add_subplot(grelha[0, :])
    do_antes = figura.add_subplot(grelha[1, 0])
    do_depois = figura.add_subplot(grelha[1, 1])

    _desenha_a_fila(das_barras, fila)
    # Uma região por lado, calculada uma vez e usada no desenho e na legenda. Cada lado usa os
    # próprios scores, porque o sinal dos componentes não é o mesmo nas duas decomposições.
    regioes = (elipse_das_amostras(par.antes.escores), elipse_das_amostras(par.depois.escores))
    # 'raw data' à esquerda e o nome do corretor à direita, como em `tools.R:1585` e `:1597`.
    # O estágio do pipeline fica no título de cada painel e no rodapé.
    for eixo, lado, rotulo_x, regiao in (
        (do_antes, par.antes, ROTULO_DO_EIXO_DO_ANTES, regioes[0]),
        (do_depois, par.depois, par.corretor, regioes[1]),
    ):
        _desenha_o_lado(eixo, lado, rotulo_x, regiao)

    # Uma legenda para a figura toda, no painel da direita, porque a codificação é a mesma nos
    # dois painéis. As entradas são artistas que não entram no eixo.
    com_regiao = any(isinstance(regiao, Elipse) for regiao in regioes)
    legenda_fora(do_depois, _provas_da_legenda(par, com_regiao=com_regiao))
    figura.suptitle(TITULO_DO_PAINEL)
    aplica(figura, recorte)
    return figura


def _provas_da_legenda(par: ParDaCorrecao, *, com_regiao: bool) -> list[Artist]:
    """As entradas de legenda: uma por tipo de amostra presente em algum painel, e a da região.

    Tipo sem amostra não ganha entrada, para a legenda não apontar uma série que não existe.
    A região só entra quando algum lado a desenhou.
    """
    provas: list[Artist] = []
    for tipo in ORDEM_DAS_SERIES:
        existe = any(int(lado.escores.mascara(tipo).sum()) for lado in (par.antes, par.depois))
        if not existe:
            continue
        cor = CORES_DA_PCA[tipo]
        provas.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=cor,
                markerfacecolor=cor if tipo in CHEIOS else "none",
                label=_ROTULO_DO_TIPO[tipo],
            )
        )
    if com_regiao:
        provas.append(
            Line2D([], [], linestyle=TRACO_DA_REGIAO, color=COR_DA_REGIAO, label=ROTULO_DA_REGIAO)
        )
    return provas


def _desenha_a_fila(eixo: Axes, fila: FilaDeBarras | FilaAusente) -> None:
    """Desenha a fila de barras, ou o texto que diz por que ela não existe.

    Há uma chamada de `ax.vlines` por barra. Uma chamada só daria uma `LineCollection` com um
    rótulo, e o arnês de teste, que indexa por rótulo, veria as barras como uma série.
    """
    if isinstance(fila, FilaAusente):
        eixo.set_axis_off()
        texto = eixo.text(
            0.02,
            0.5,
            fila.motivo,
            transform=eixo.transAxes,
            ha="left",
            va="center",
            fontsize=TAMANHO_DA_FONTE,
        )
        texto.set_gid(GID_REQUEBRAVEL)
        # Fora do `tight_layout`: o texto nasce numa linha só, mais larga que a figura, e o
        # primeiro `tight_layout` de `recorte.aplica`, anterior à quebra, avisaria e não seria
        # aplicado. A geometria final é a mesma, porque o que vale é o `tight_layout` posterior
        # à quebra. A altura é garantida por `ALTURAS_SEM_FILA`.
        texto.set_in_layout(False)
        return

    cores = cores_da_fila(fila)
    alturas = [barra.rsd * 100.0 for barra in fila.barras]
    for posicao, (barra, cor) in enumerate(zip(fila.barras, cores, strict=True)):
        eixo.vlines(
            posicao,
            0.0,
            barra.rsd * 100.0,
            colors=cor,
            linewidth=LARGURA_DA_BARRA,
            label=barra.corretor,
        )

    topos = _desenha_os_intervalos(eixo, fila)

    # O valor só sobre a barra do `none` e a do vencedor, como em `tools.R:1554-1555`. Quando
    # o `none` vence sai um rótulo só. Com o intervalo, o valor fica acima dele.
    melhor = fila.melhor
    com_valor = {NOME_DO_NONE} | ({melhor.corretor} if melhor is not None else set())
    for posicao, barra in enumerate(fila.barras):
        if barra.corretor in com_valor:
            eixo.annotate(
                percentagem_do_r(barra.rsd),
                xy=(posicao, topos[posicao]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=TAMANHO_DA_FONTE,
            )

    eixo.set_xticks(list(range(len(fila.barras))))
    eixo.set_xticklabels([barra.corretor for barra in fila.barras])
    eixo.set_xlim(-0.6, len(fila.barras) - 0.4)
    # A folga em cima é para o valor escrito caber dentro do eixo. Ver `FOLGA_DO_ROTULO`.
    maior = max([*alturas, *topos]) if alturas else 0.0
    eixo.set_ylim(0.0, maior * (1.0 + FOLGA_DO_ROTULO) if maior > 0.0 else 1.0)
    eixo.set_ylabel(ROTULO_DO_EIXO_DAS_BARRAS)
    eixo.set_title(f"{TITULO_DAS_BARRAS}\n{_frase_das_barras(fila)}", fontsize=TAMANHO_DA_FONTE)


def _desenha_os_intervalos(eixo: Axes, fila: FilaDeBarras) -> list[float]:
    """Desenha o intervalo das barras que o têm e devolve o topo de cada posição.

    Há uma linha por intervalo, com o corretor no rótulo, porque o arnês de teste indexa por
    rótulo. O traço e as duas tampas são o mesmo artista: a linha de `inferior` a `superior`,
    com o marcador `_` nas pontas. Os dados ficam na posição da barra, em unidades do eixo, e
    só o desenho é deslocado para a direita, em pontos (`DESLOCAMENTO_DO_INTERVALO`).

    O topo devolvido, em percentagem, é o da barra ou o do intervalo, o que for maior. Serve
    para posicionar o valor escrito e para o limite do eixo.
    """
    topos = [barra.rsd * 100.0 for barra in fila.barras]
    if fila.intervalos is None:
        return topos
    # O que `transforms.offset_copy` faz por dentro: pontos, 72 por polegada, na escala da figura.
    ao_lado = eixo.transData + transforms.ScaledTranslation(
        DESLOCAMENTO_DO_INTERVALO / 72.0, 0.0, eixo.figure.dpi_scale_trans
    )
    for posicao, barra in enumerate(fila.barras):
        intervalo = fila.intervalos.get(barra.corretor)
        if not isinstance(intervalo, IntervaloDeBarra):
            continue
        baixo, alto = intervalo.inferior * 100.0, intervalo.superior * 100.0
        eixo.plot(
            [posicao, posicao],
            [baixo, alto],
            color=COR_DO_INTERVALO,
            linewidth=LARGURA_DO_INTERVALO,
            marker="_",
            markersize=TAMANHO_DA_TAMPA,
            markeredgewidth=LARGURA_DO_INTERVALO,
            transform=ao_lado,
            label=f"{ROTULO_DO_INTERVALO} {barra.corretor}",
        )
        topos[posicao] = max(topos[posicao], alto)
    return topos


def _desenha_o_lado(
    eixo: Axes, lado: LadoDaCorrecao, rotulo_x: str, regiao: Elipse | ElipseAusente
) -> None:
    """Um painel de PCA: a região, as séries por tipo, os rótulos dos eixos e o título.

    Série sem pontos não é desenhada. A ausência do QC, porém, é declarada em texto: ele é a
    régua técnica da figura, e um painel sem QC e sem aviso pareceria um painel em que o QC
    agrupou.

    A região é desenhada primeiro, para ficar por baixo das séries e não esconder um QC.
    """
    _desenha_a_regiao(eixo, lado, regiao)
    desenhadas = 0
    for tipo in ORDEM_DAS_SERIES:
        mascara = lado.escores.mascara(tipo)
        quantas = int(mascara.sum())
        if quantas == 0:
            continue
        desenhadas += quantas
        cor = CORES_DA_PCA[tipo]
        eixo.plot(
            lado.escores.x[mascara],
            lado.escores.y[mascara],
            marker="o",
            linestyle="none",
            color=cor,
            markerfacecolor=cor if tipo in CHEIOS else "none",
            label=rotulo_da_serie(tipo, lado.rotulo),
        )

    if desenhadas == 0:
        raise MatrizVaziaError(
            f"o painel «{lado.rotulo}» tem {lado.escores.n} amostra(s) e nenhuma com PC1 e PC2 "
            "finitos: não há ponto a desenhar. Um eixo montado e vazio ao lado de um cheio lê-se "
            "como figura quebrada, e é o outro lado que ficaria a parecer errado"
        )

    if not int(lado.escores.mascara(TipoDeAmostra.QC).sum()):
        sem_qc = eixo.text(
            0.02,
            0.98,
            ROTULO_SEM_QC,
            transform=eixo.transAxes,
            ha="left",
            va="top",
            fontsize=TAMANHO_DA_FONTE,
        )
        sem_qc.set_gid(GID_REQUEBRAVEL)

    eixo.set_xlabel(f"{rotulo_x}\n{_rotulo_do_componente(lado, 0)}")
    eixo.set_ylabel(_rotulo_do_componente(lado, 1))
    eixo.set_title(
        f"{lado.rotulo} — {lado.matriz}\n{frase_do_lado(lado)}", fontsize=TAMANHO_DA_FONTE
    )


def _desenha_a_regiao(eixo: Axes, lado: LadoDaCorrecao, regiao: Elipse | ElipseAusente) -> None:
    """Desenha a região de predição do lado, ou um texto curto que diz por que ela falta.

    A região é uma `Line2D`, e não um `patch` nem um `fill_between`, para ficar visível ao
    arnês de teste e à identificação por clique.

    O texto da ausência fica dentro do eixo, no canto de baixo: no rodapé ele não cabe, e o
    canto de cima é o de `ROTULO_SEM_QC`.
    """
    if isinstance(regiao, ElipseAusente):
        texto = eixo.text(
            0.02,
            0.02,
            _frase_sem_regiao(lado, regiao),
            transform=eixo.transAxes,
            ha="left",
            va="bottom",
            fontsize=TAMANHO_DA_FONTE,
        )
        texto.set_gid(GID_REQUEBRAVEL)
        return
    x, y = regiao.contorno()
    eixo.plot(
        x,
        y,
        linestyle=TRACO_DA_REGIAO,
        color=COR_DA_REGIAO,
        label=rotulo_da_regiao(lado.rotulo),
    )


def _frase_sem_regiao(lado: LadoDaCorrecao, regiao: ElipseAusente) -> str:
    """O motivo resumido da região ausente, como `"sem região: menos de 3 amostras de estudo (2)"`.

    O motivo completo de `ElipseAusente` é longo demais para um eixo de um terço da figura.
    A contagem é a da máscara das amostras de estudo, a mesma de `pca.elipse_das_amostras`, e
    o limiar é `MINIMO_PARA_ELIPSE`. Fora desse caso, usa o texto do motivo até os dois
    pontos, sem adivinhar a causa.
    """
    estudo = int(lado.escores.mascara(TipoDeAmostra.AMOSTRA).sum())
    if estudo < MINIMO_PARA_ELIPSE:
        return f"sem região: menos de {MINIMO_PARA_ELIPSE} amostras de estudo ({milhar(estudo)})"
    return f"sem região: {regiao.motivo.split(':', 1)[0]}"


def _rotulo_do_componente(lado: LadoDaCorrecao, indice: int) -> str:
    """O rótulo de eixo de um componente, com a variância explicada: `"PC1 (23,4%)"`.

    O R esconde a numeração dos eixos. Aqui a variância vai no rótulo porque, sem ela, não se
    vê que um eixo explica muito mais que o outro, e a distância horizontal não se compara
    com a vertical dentro do painel.
    """
    fracao = float(lado.variancia[indice]) if lado.variancia.size > indice else float("nan")
    if not math.isfinite(fracao):
        return f"PC{indice + 1} (variância não declarada)"
    return f"PC{indice + 1} ({fracao * 100:.1f}%)".replace(".", ",")

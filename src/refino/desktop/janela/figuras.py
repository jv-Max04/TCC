"""O widget que embute uma figura de `refino.plots` na janela.

`PainelDeFigura` é widget filho, e não painel composto: não tem chave em `PAINEIS` nem aba própria.
Vive dentro de `PainelDeGraficos`, que responde pela aba 'Gráficos', e nas vistas das abas de
etapa. Desenha a figura escolhida no combo, anota o ponto clicado, aproxima com Ctrl+roda e grava
o arquivo; o botão de gravar fica em `graficos.py`, ligado pelo sinal `exportavel`. Não há
`NavigationToolbar2QT`: o usuário a rejeitou, porque o botão de gravar dela duplica o da aba e muda
o aspecto da tela.

A figura sai quando a execução muda. A chave vem de `interface.cliente.chave_de_invalidacao`, que
devolve
a chave do servidor num `200` e um marcador irrepetível (`divergente:<motivo>:<uuid4>`) no resto,
então duas ausências nunca comparam iguais e a figura da execução anterior não sobrevive a um erro
nem a um `202`. No lugar dela entra texto, nunca espaço em branco, e os estados de `EstadoDaFigura`
formam um vocabulário fechado porque cada um pede uma ação diferente de quem lê.
"""

from __future__ import annotations

import warnings
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, cast

from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.backend_bases import scroll_handler
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PIL.Image import DecompressionBombError
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette, QShowEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.janela.aparencia import FiltroDaRoda, sem_roda_sem_foco
from refino.desktop.janela.componentes import ComboComNomes, desliga_com_motivo, religa
from refino.desktop.janela.nomes import NOMES_DAS_FIGURAS
from refino.desktop.janela.tema import ICONES, poe_icone
from refino.interface.cliente import PREFIXO_DE_DIVERGENCIA, Resposta, chave_de_invalidacao
from refino.plots.base import (
    GID_DAS_FEATURES,
    GID_DO_RODAPE,
    MatrizVaziaError,
    RecusaDoConstrutorError,
)
from refino.plots.catalogo import CONSTRUTORES, ConstrutorDeFigura, rotulo_do_construtor
from refino.plots.recorte import decimal, folga_do_rodape, milhar, reajusta, retira

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from typing import Protocol

    from matplotlib.artist import Artist
    from matplotlib.backend_bases import MouseEvent, NavigationToolbar2, PickEvent
    from matplotlib.text import Annotation
    from PySide6.QtWidgets import QLayout

    class EscolhedorDeArquivo(Protocol):
        """Para onde a figura vai. `None` quando o profissional cancelou o diálogo.

        Escolhe um caminho, e não recebe bytes: com bytes a figura teria de ser renderizada antes
        de se saber se o profissional vai cancelar, e o `OSError` da escrita cairia longe de quem
        sabe nomear o caminho. Com o caminho, o `savefig` escreve direto no destino e a recusa é
        apanhada onde a frase se escreve.
        """

        def __call__(self, nome_sugerido: str) -> Path | None: ...


__all__ = [
    "CHAVE_DE_REPOUSO",
    "CORES_DA_FIGURA",
    "DPIS_DA_EXPORTACAO",
    "DPI_DA_EXPORTACAO",
    "DPI_DE_RECUO",
    "FORMATOS_DA_EXPORTACAO",
    "MM_POR_POLEGADA",
    "MODIFICADOR_DO_ZOOM",
    "MOTIVO_SEM_VISTA",
    "POR_QUE_TODAS_SAEM",
    "SUFIXO_DA_EXPORTACAO",
    "SUFIXO_DO_RODAPE",
    "TECLA_DO_ZOOM",
    "TEXTO_DO_DESFECHO",
    "TEXTO_DO_ESTADO",
    "DesfechoDaExportacao",
    "EstadoDaFigura",
    "Fonte",
    "FonteAusenteError",
    "HistoricoMudo",
    "PainelDeFigura",
    "aplica_o_tema",
    "arquivo_por_dialogo",
    "caminho_do_rodape",
    "cores_do_tema",
    "dpi_da_tela",
    "entradas_do_seletor",
    "estado_da_resposta",
    "exporta_habilitado",
    "frase_do_minimo",
    "menor_altura_que_cabe",
    "mm_inteiros",
    "motivo_da_chave",
    "texto_do_desfecho",
    "texto_do_estado",
    "vista_habilitada",
]

# A chave de repouso: 'nasci e nada chegou'. É diferente de 'chegou sem chave', que tem o marcador
# de `interface.cliente`, e não colide com nenhum dos dois por construção: uma chave real é o
# `hexdigest()` de um `sha256` (64 caracteres de `0-9a-f`), e um marcador começa por `divergente:`.
CHAVE_DE_REPOUSO: Final = "sem-execucao-ainda"

# O que o `matplotlib` usa quando não há tela nenhuma de onde ler. Não é escolha de estilo: é o
# valor que `rcParams["figure.dpi"]` já teria, e serve para o recuo não mudar o desenho.
DPI_DE_RECUO: Final = 100.0

# As cores com que a figura é pintada, `(fundo da figura, fundo dos eixos, tinta)`, na ordem de
# `cores_do_tema`. Fixas, e não as do tema, por escolha do usuário: sobre o fundo escuro, as linhas
# de caixa, bigode e tampa que o `matplotlib` desenha a preto ficam a 1,52:1. A tinta vai junto,
# senão a rotulagem viria do tema escuro e sairia clara sobre branco. Os valores são os do
# `matplotlib` de origem, para o que os construtores desenham sem passar por aqui (a mediana, a
# moldura de uma caixa) não dar dois pretos na mesma imagem.
CORES_DA_FIGURA: Final[tuple[str, str, str]] = ("#ffffff", "#ffffff", "#000000")

# O DPI padrão do arquivo gravado. É convenção, e não medição: 300 ppp é o que as revistas da área
# pedem para figura de linha, e o padrão de quem monta pôster.
DPI_DA_EXPORTACAO: Final = 300.0

# Os DPI que a exportação aceita, vocabulário fechado conferido em runtime. 600 é o mínimo de
# figura com texto em parte das revistas; 1.200, o de figura de linha em outras.
DPIS_DA_EXPORTACAO: Final[tuple[int, ...]] = (300, 600, 1200)

# Os formatos aceitos, pelo sufixo do nome escolhido. O TIFF sai achatado sobre branco, em RGB e
# com LZW, porque o `savefig` grava RGBA e há revista que recusa canal alfa; o PDF leva as fontes
# TrueType embutidas e nenhuma data, para duas gravações da mesma figura serem iguais.
FORMATOS_DA_EXPORTACAO: Final[dict[str, str]] = {
    ".png": "png",
    ".tif": "tiff",
    ".tiff": "tiff",
    ".pdf": "pdf",
}

MM_POR_POLEGADA: Final = 25.4

# O sufixo do nome que o diálogo sugere; os aceitos são os de `FORMATOS_DA_EXPORTACAO`. A
# comparação é em minúsculas contra `Path.suffix`, e a guarda corre antes da gravação: uma exceção
# dentro de um slot do Qt vai para o `stderr` e o painel para de atualizar em silêncio.
SUFIXO_DA_EXPORTACAO: Final = ".png"

# O fim do nome do `.txt` que leva o rodapé quando ele não cabe na figura: `figura.png` dá
# `figura-rodape.txt`. O sufixo diz o que o arquivo é, e o nome diz de qual figura.
SUFIXO_DO_RODAPE: Final = "-rodape.txt"

# Altura mínima do canvas, em pixels. Abaixo disto os rótulos dos eixos colidem com o rodapé do
# recorte, e o `QScrollArea` que hospeda o painel dá-lhe a altura do `sizeHint` se nada a pedir.
ALTURA_MINIMA_DO_CANVAS: Final = 260

# A largura de leitura da frase do estado vazio. Sem teto ela corre a largura da aba, e uma linha
# desse tamanho lê-se mal; 560 px dão perto de 80 caracteres a 10 pt.
LARGURA_DO_TEXTO_DO_VAZIO: Final = 560

# O modificador que arma o zoom. O token é `'ctrl'`, e não `'control'`:
# `matplotlib/backends/backend_qt.py:434` troca `'control'` por `'ctrl'` quando a tecla é
# modificador. Lido de `evento.modifiers`, nunca de `evento.key`: num `scroll_event` vindo do Qt o
# `key` é `None` mesmo com o Ctrl premido (`wheelEvent` passa `modifiers=` e não `key=`), e um
# manipulador escrito contra `key` nunca dispararia.
MODIFICADOR_DO_ZOOM: Final = "ctrl"

# O nome que `scroll_handler` exige em `evento.key` para aproximar os dois eixos. É `"control"`
# por extenso porque o `matplotlib` chama a tecla de `'control'` quando ela é a tecla e de `'ctrl'`
# quando é modificador (`backend_qt.py:432-434`), e `scroll_handler` compara a tecla:
# `if event.key in {"control", "x", "y"}`.
TECLA_DO_ZOOM: Final = "control"

# Quem constrói a figura de uma entrada do registo, e os nomes de feature que ela põe em cena.
# Injetada, e não importada, porque o construtor precisa dos números da execução corrente e a
# janela não recalcula estatística: quem os tem é o painel que hospeda.
type Fonte = Callable[[ConstrutorDeFigura], tuple[Figure, tuple[str, ...]]]


class FonteAusenteError(ValueError):
    """Recusa tipada: há construtor registado e ninguém para o chamar.

    Levanta na construção em vez de mostrar 'sem construtor' porque as duas coisas leem-se igual na
    tela e são opostas: 'nenhuma figura está registada' é fato do projeto, e 'há figuras
    registadas e este painel não sabe construí-las' é defeito de ligação.
    """


class EstadoDaFigura(StrEnum):
    """As situações em que este painel pode estar. Vocabulário fechado, como `EstadoDoCatalogo`.

    Estado novo entra aqui e ganha texto em `TEXTO_DO_ESTADO`, senão `texto_do_estado` levanta:
    forma nova reprova em vez de cair num `else` razoável.
    """

    DESENHADA = "desenhada"
    """Há execução corrente e há figura registada: o canvas mostra."""

    SEM_EXECUCAO = "sem-execucao"
    """O painel nasceu e nenhuma resposta chegou ainda."""

    SEM_CONSTRUTOR = "sem-construtor"
    """Há execução corrente e nenhuma figura registada em `plots.CONSTRUTORES`."""

    SEM_DADO = "sem-dado"
    """O construtor recusou por população vazia: `MatrizVaziaError`."""

    DADO_RECUSADO = "dado-recusado"
    """O construtor recusou por outro motivo: uma `RecusaDoConstrutorError` que não é a vazia.

    Separado de `SEM_DADO` porque os consertos são opostos: população que não veio é execução que
    não correu, e população que não descreve os mesmos objetos é corpo montado errado (um vetor de
    tipos com 67 entradas para 68 scores).
    """

    LIGACAO_QUEBRADA = "ligacao-quebrada"
    """Há construtor registado e quem hospeda o painel não sabe chamá-lo (`FonteAusenteError`).

    Separado de `SEM_CONSTRUTOR` pelo argumento de `FonteAusenteError`: 'nenhuma figura está
    registada' é fato do projeto, e 'este painel não sabe construí-las' é defeito de ligação. É
    estado, e não exceção, porque a exceção levantada dentro do slot do Qt vai para o `stderr` e o
    painel para de atualizar em silêncio.
    """

    SEM_CAMPO = "sem-campo"
    """A resposta veio `200` e nenhuma rota pôs chave naquele corpo."""

    CAMPO_ILEGIVEL = "campo-ilegivel"
    """A chave veio e não é uma chave."""

    EM_CURSO = "em-curso"
    """O `202`: o trabalho ainda corre, e vai ter chave."""

    CANCELADA = "cancelada"
    """O `410`: o profissional cancelou a execução.

    Separado de `FALHOU` porque não houve defeito: sem ele o painel escreveria 'a última chamada
    não devolveu resultado' sobre um cancelamento pedido, declarando defeito onde houve escolha.
    """

    FALHOU = "falhou"
    """Um código que não é `200` nem `202`: não há resultado a desenhar."""

    ETAPA_DESATUALIZADA = "etapa-desatualizada"
    """Figura de uma etapa do passo a passo cujos parâmetros já não são os da tela.

    Separado de `SEM_EXECUCAO`: a ação é refazer a etapa, e a frase daquele manda esperar o
    pipeline.
    """

    SEM_DIAGNOSTICO = "sem-diagnostico"
    """Figura do diagnóstico sem diagnóstico desta matriz.

    As figuras que saem de `/matrizes/diagnosticar` não dependem da execução: valem enquanto o
    diagnóstico for da matriz corrente. Separado de `SEM_EXECUCAO` porque a ação é rodar o
    diagnóstico, e não esperar o pipeline. O violino de RSD por feature também usa este estado até
    chegar o primeiro resultado de corretor da matriz: antes dele, desenha a matriz ingerida que o
    diagnóstico traz.
    """


# Por que todas as figuras saem juntas. A chave de invalidação é escalar e cobre a execução
# inteira, então mudar um parâmetro de uma análise retira também a PCA e o volcano. Sem esta frase
# na tela, o esvaziamento simultâneo parece o aplicativo a quebrar.
POR_QUE_TODAS_SAEM: Final = (
    "Saem TODAS as figuras da execução juntas, e não só esta: a identidade é da execução inteira, "
    "então mudar um parâmetro de uma análise retira também as figuras que nada têm com ela. As do "
    "diagnóstico ficam, porque são da matriz e não da execução"
)

TEXTO_DO_ESTADO: Final[dict[EstadoDaFigura, str]] = {
    EstadoDaFigura.DESENHADA: "",
    EstadoDaFigura.SEM_EXECUCAO: (
        "Nenhuma execução ainda — a figura aparece quando o pipeline devolver resultado"
    ),
    EstadoDaFigura.ETAPA_DESATUALIZADA: (
        "A figura é de uma etapa do passo a passo feita com parâmetros que já não são os da tela "
        "— refaça essa etapa"
    ),
    EstadoDaFigura.SEM_DIAGNOSTICO: (
        "Nenhum diagnóstico desta matriz ainda — a figura aparece quando o diagnóstico da aba "
        "«Preparar o estudo» voltar, sem precisar executar o pipeline"
    ),
    EstadoDaFigura.SEM_CONSTRUTOR: (
        "Há execução corrente e nenhuma figura registada para desenhar. As figuras entram uma a "
        "uma nas unidades seguintes, e cada uma se declara em `refino.plots.CONSTRUTORES`"
    ),
    EstadoDaFigura.SEM_DADO: (
        "A figura foi retirada: o construtor recebeu população vazia e recusou desenhar. Figura em "
        "branco seria indistinguível de figura quebrada"
    ),
    EstadoDaFigura.DADO_RECUSADO: (
        "A figura foi retirada: o construtor recusou o que veio na resposta. Os vetores que ela "
        "traz não descrevem os mesmos objetos — desenhar assim daria uma figura bem formada com o "
        "ponto de uma amostra sobre o nome de outra"
    ),
    EstadoDaFigura.LIGACAO_QUEBRADA: (
        "A figura está registada e este painel não sabe construí-la. Isto é defeito de ligação do "
        "programa, e não fato da sua execução — o resultado está lá, e quem falhou fomos nós"
    ),
    EstadoDaFigura.SEM_CAMPO: (
        "A figura foi retirada: esta resposta não traz a identidade da execução, e desenhar sobre "
        "ela mostraria número de outra corrida"
    ),
    EstadoDaFigura.CAMPO_ILEGIVEL: (
        "A figura foi retirada: a identidade da execução veio ilegível, e não há como afirmar que "
        "os números desenhados são os desta corrida"
    ),
    EstadoDaFigura.EM_CURSO: (
        "A figura foi retirada: a execução ainda corre. Ela volta quando o resultado chegar. "
        f"{POR_QUE_TODAS_SAEM}"
    ),
    EstadoDaFigura.CANCELADA: (
        "A figura foi retirada: VOCÊ cancelou esta execução, e ela parou antes de produzir "
        "resultado. Não há defeito nenhum no seu arquivo nem na ferramenta — dispare outra vez se "
        "ainda a quiser, e será uma execução nova, do início"
    ),
    EstadoDaFigura.FALHOU: (
        "A figura foi retirada: a última chamada não devolveu resultado. O que ficaria na tela "
        "seria a execução anterior"
    ),
}
"""A frase de cada estado. A de `DESENHADA` é vazia de propósito: nesse estado quem fala é a
figura, e um rótulo por cima dela seria a segunda fonte da mesma verdade."""

# O marcador de `interface.cliente` para um código que não é `200`, sem o número. O `202` e o `410`
# têm
# estado próprio: 'vai ter chave' e 'não vai' são fatos opostos.
_PREFIXO_DO_CODIGO: Final = "codigo-"
_MOTIVO_DO_EM_CURSO: Final = f"{_PREFIXO_DO_CODIGO}202"
# O `410` é cancelamento, e não falha: sem esta linha ele cairia no recuo e seria descrito como a
# chamada que não devolveu resultado, atribuindo à ferramenta uma decisão do profissional.
_MOTIVO_DO_CANCELAMENTO: Final = f"{_PREFIXO_DO_CODIGO}410"

_ESTADO_DO_MOTIVO: Final[dict[str, EstadoDaFigura]] = {
    "sem-campo": EstadoDaFigura.SEM_CAMPO,
    "campo-ilegivel": EstadoDaFigura.CAMPO_ILEGIVEL,
    _MOTIVO_DO_EM_CURSO: EstadoDaFigura.EM_CURSO,
    _MOTIVO_DO_CANCELAMENTO: EstadoDaFigura.CANCELADA,
}


def _par(limites: tuple[float, ...]) -> tuple[float, float]:
    """Um limite de eixo como par de dois, e não como tupla de tamanho aberto.

    `get_xlim` devolve dois valores e o tipo declarado não o diz; `set_xlim` exige o par. Converter
    aqui, uma vez, é o que impede um `# type: ignore` em cada uma das quatro chamadas.
    """
    return float(limites[0]), float(limites[1])


class HistoricoMudo:
    """O `toolbar` que `scroll_handler` exige, reduzido ao que ele chama.

    `backend_bases.scroll_handler` recusa-se a trabalhar sem barra de ferramentas, e o próprio
    `matplotlib` comenta que a restrição é histórica, não técnica. Do toolbar ele só chama
    `push_current`, que empilha a vista no histórico dos botões 'anterior' e 'seguinte' da barra.
    Esta janela não tem barra, e o caminho de volta é `_vista_original`, guardado no desenho;
    guardar as vistas intermédias seria um histórico que nenhum controle lê.
    """

    def push_current(self) -> None:
        """Não guarda nada: o caminho de volta deste painel é `_vista_original`."""


def vista_habilitada(estado: EstadoDaFigura) -> bool:
    """Se o controle de voltar à vista deve estar habilitado neste estado.

    Sem figura na tela, `_vista_original` está vazio e o botão não faria nada; um controle vivo que
    não responde parece painel quebrado. O corpo é igual ao de `exporta_habilitado`, e a função é
    separada de propósito: 'posso gravar?' e 'posso voltar?' coincidem hoje, mas são perguntas
    diferentes. Só `DESENHADA` habilita, pela mesma razão do `is` de lá.
    """
    return estado is EstadoDaFigura.DESENHADA


# Por que 'Voltar à vista original' está desligado, dito ao lado dele. O botão só se desliga sem
# figura na tela, e o bloco logo abaixo já diz o que falta para ela aparecer; a frase aponta para
# ele em vez de repetir a razão do estado.
MOTIVO_SEM_VISTA: Final = (
    "Sem figura na tela, não há vista a que voltar — o quadro abaixo diz o que falta para ela "
    "aparecer."
)


def motivo_da_chave(chave: str) -> str:
    """O motivo dentro de um marcador de divergência, ou `""` quando a chave é real.

    `divergente:sem-campo:9f3...` dá `"sem-campo"`. O nonce fica de fora: ele existe para duas
    ausências não compararem iguais, não para ser lido. `""` não é ambíguo, porque um motivo é
    sempre não vazio.
    """
    if not chave.startswith(PREFIXO_DE_DIVERGENCIA):
        return ""
    return chave[len(PREFIXO_DE_DIVERGENCIA) :].split(":", 1)[0]


def estado_da_resposta(chave: str, *, ha_construtor: bool) -> EstadoDaFigura:
    """Em que estado o painel fica com esta chave. Total: toda chave cai em exatamente um.

    A obsolescência decide primeiro, porque uma figura de outra execução tem de sair mesmo que haja
    construtor para a redesenhar; a disponibilidade só se pergunta com a execução corrente.
    """
    if chave == CHAVE_DE_REPOUSO:
        return EstadoDaFigura.SEM_EXECUCAO
    motivo = motivo_da_chave(chave)
    if motivo:
        # Todo código sem estado próprio (`codigo-500`, `codigo-404`) diz a mesma coisa ao
        # profissional: não há resultado.
        return _ESTADO_DO_MOTIVO.get(motivo, EstadoDaFigura.FALHOU)
    return EstadoDaFigura.DESENHADA if ha_construtor else EstadoDaFigura.SEM_CONSTRUTOR


def texto_do_estado(estado: EstadoDaFigura) -> str:
    """A frase daquele estado. Levanta `KeyError` para estado sem texto, e isso é o portão.

    Um `.get` com recuo faria um estado novo nascer mudo, e mudo na tela lê-se como 'não há nada a
    dizer'.
    """
    return TEXTO_DO_ESTADO[estado]


class DesfechoDaExportacao(StrEnum):
    """O que aconteceu quando o profissional pediu o arquivo. Vocabulário fechado.

    Vários desfechos, e não um `bool`, porque cada um leva a um gesto diferente: cancelar não é
    erro, a pasta que recusa pede outra pasta, o sufixo errado pede outro nome, e gravar sem o
    rodapé avisa que o texto dele está noutro arquivo. Desfecho novo ganha texto em
    `TEXTO_DO_DESFECHO`, senão `texto_do_desfecho` levanta.
    """

    GRAVADA = "gravada"
    """O arquivo foi escrito no destino escolhido."""

    GRAVADA_SEM_RODAPE = "gravada-sem-rodape"
    """No tamanho pedido o rodapé encostaria no gráfico. A figura foi escrita sem ele, e o texto
    dele foi para um `.txt` ao lado, com o nome de `caminho_do_rodape`, para a legenda."""

    CANCELADA = "cancelada"
    """O diálogo fechou sem escolha. Não é erro, e nada foi escrito."""

    SEM_FIGURA = "sem-figura"
    """Não há figura na tela para gravar: o painel está em algum estado de retirada.

    Separado de `CANCELADA`, ainda que nos dois nada seja escrito: aqui o diálogo nem abre, e o
    profissional precisa de saber que falta uma execução.
    """

    SUFIXO_RECUSADO = "sufixo-recusado"
    """O nome escolhido não termina num dos sufixos de `FORMATOS_DA_EXPORTACAO`."""

    DPI_RECUSADO = "dpi-recusado"
    """O DPI pedido não é um dos de `DPIS_DA_EXPORTACAO`. Nada foi escrito."""

    GRANDE_DEMAIS = "grande-demais"
    """A imagem no tamanho e no DPI pedidos não coube na memória, ou passou do limite de pixels do
    Pillow. A figura não foi escrita, e a frase diz quantos pixels ela teria.

    Separado de `ESCRITA_FALHOU` porque o gesto é outro: diminuir o tamanho ou a resolução, e não
    escolher outra pasta."""

    ESCRITA_FALHOU = "escrita-falhou"
    """O destino existe como escolha e o sistema recusou a escrita: um `OSError`."""


TEXTO_DO_DESFECHO: Final[dict[DesfechoDaExportacao, str]] = {
    DesfechoDaExportacao.GRAVADA: "Figura gravada em {detalhe}, a {dpi:.0f} pontos por polegada",
    DesfechoDaExportacao.GRAVADA_SEM_RODAPE: (
        "Figura gravada em {detalhe}, a {dpi:.0f} pontos por polegada, sem o rodapé: no tamanho "
        "pedido ele encostaria no gráfico. O texto do rodapé, para a legenda, está em {rodape}"
    ),
    DesfechoDaExportacao.DPI_RECUSADO: (
        "Só sei gravar a 300, 600 ou 1.200 pontos por polegada, e o pedido foi {detalhe} — nada "
        "foi escrito"
    ),
    DesfechoDaExportacao.GRANDE_DEMAIS: (
        "Em {detalhe}, a figura é grande demais para gravar, e não foi escrita — diminua o "
        "tamanho ou a resolução"
    ),
    DesfechoDaExportacao.CANCELADA: "Gravação cancelada — nenhum arquivo foi escrito",
    DesfechoDaExportacao.SEM_FIGURA: (
        "Não há figura na tela para gravar. O estado do painel é: {detalhe}"
    ),
    DesfechoDaExportacao.SUFIXO_RECUSADO: (
        "Só sei gravar {sufixo}, e o nome escolhido termina em {detalhe} — nada foi escrito"
    ),
    DesfechoDaExportacao.ESCRITA_FALHOU: "Não foi possível gravar em {detalhe}",
}
"""A frase de cada desfecho. Todas nomeiam o que não aconteceu, porque o silêncio depois de um
clique lê-se como sucesso, e a maior parte dos desfechos não escreve arquivo nenhum."""


def texto_do_desfecho(
    desfecho: DesfechoDaExportacao,
    detalhe: str,
    *,
    dpi: float = DPI_DA_EXPORTACAO,
    rodape: str = "",
) -> str:
    """A frase daquele desfecho, com o caminho ou a causa dentro dela.

    Levanta `KeyError` para desfecho sem texto, pela razão de `texto_do_estado`. `dpi` é o da
    gravação, para a frase dizer o que foi escrito; `rodape` é o caminho do `.txt`, que a frase da
    gravação sem rodapé cita.
    """
    return TEXTO_DO_DESFECHO[desfecho].format(
        detalhe=detalhe, dpi=dpi, rodape=rodape, sufixo=", ".join(FORMATOS_DA_EXPORTACAO)
    )


def caminho_do_rodape(destino: Path) -> Path:
    """O `.txt` do rodapé, ao lado da figura e com o nome dela.

    Sem o sufixo da figura (não `figura.png.txt`): o PNG e o PDF da mesma figura levam o mesmo
    rodapé, e um arquivo serve aos dois.
    """
    return destino.with_name(f"{destino.stem}{SUFIXO_DO_RODAPE}")


def menor_altura_que_cabe(cabe: Callable[[int], bool | None]) -> int | None:
    """O menor número inteiro de mm, entre 20 e 600, em que `cabe` diz sim.

    `None` quando nem 600 cabem, ou quando `cabe` não sabe dizer (`None`: a figura não tem rodapé).

    Inteiro porque os campos da aba são inteiros e o número da tela é o que a pessoa digita. Buscar
    em mm fracionários e arredondar erra o inteiro: `round` pode cair abaixo do limiar (a tela
    sugeriria uma altura em que o rodapé não cabe) e `ceil` pode subir um mm (diria que o rodapé
    vai para o `.txt` numa altura em que ele cabe). A busca binária sobre inteiros mantém
    `cabe(alto)` verdadeiro e `cabe(baixo)` falso, para quando os dois são vizinhos, e supõe que
    caber é monótono na altura.
    """
    if cabe(600) is not True:
        return None
    if cabe(20):
        return 20
    baixo, alto = 20, 600
    while alto - baixo > 1:
        meio = (baixo + alto) // 2
        if cabe(meio):
            alto = meio
        else:
            baixo = meio
    return alto


def frase_do_minimo(minima_mm: float | None, altura_mm: float) -> str:
    """A frase que a aba 'Gráficos' mostra ao lado do tamanho.

    Diz o que a altura escolhida faz com o rodapé, e não só o mínimo: abaixo dele a figura sai sem
    o rodapé, e quem escolheu tem de saber antes de salvar que o texto vai para um `.txt` e com que
    nome. Vazia quando a figura não tem rodapé. Mora aqui, ao lado de `texto_do_desfecho`, porque a
    frase é desta exportação.
    """
    if minima_mm is None:
        return ""
    destino = (
        f"o texto dele vai para um arquivo ao lado dela, com o nome da figura terminado em "
        f"«{SUFIXO_DO_RODAPE}», para a legenda."
    )
    inicio = f"Nesta largura, a altura mínima para o rodapé caber na figura é {_mm(minima_mm)} mm."
    if altura_mm < minima_mm:
        return f"{inicio} Na altura escolhida, a figura sai sem o rodapé, e {destino}"
    return f"{inicio} Abaixo dela, a figura sai sem o rodapé, e {destino}"


def _em_pixels(largura_mm: float, altura_mm: float, dpi: float) -> str:
    """O tamanho pedido e os pixels que ele dá, para a frase da imagem grande demais.

    Os pixels são truncados como o Agg os trunca: é o número que a pessoa compara com o que o
    programa de destino aceita.
    """
    largura_px = int(largura_mm / MM_POR_POLEGADA * dpi)
    altura_px = int(altura_mm / MM_POR_POLEGADA * dpi)
    return (
        f"{_mm(largura_mm)} x {_mm(altura_mm)} mm a {milhar(int(dpi))} pontos por polegada, "
        f"{milhar(largura_px)} x {milhar(altura_px)} pixels"
    )


def _mm_da(figura: Figure) -> tuple[float, float]:
    """O tamanho atual da figura, em mm: o recuo quando não há tamanho de construção guardado."""
    largura, altura = (float(v) * MM_POR_POLEGADA for v in figura.get_size_inches())
    return largura, altura


def _mm(valor: float) -> str:
    """Milímetros na tela: inteiro quando é inteiro, e vírgula decimal quando não."""
    return f"{valor:.0f}" if float(valor).is_integer() else f"{valor:.1f}".replace(".", ",")


def mm_inteiros(tamanho_mm: tuple[float, float]) -> tuple[int, int]:
    """O tamanho em mm arredondado para os campos inteiros da aba 'Gráficos'.

    A conta mora aqui porque o portão de AST de `graficos.py` proíbe toda aritmética nele, `round`
    incluído (`test_u7_o_painel_nao_faz_aritmetica_nenhuma`).
    """
    largura, altura = tamanho_mm
    return round(largura), round(altura)


def _grava(figura: Figure, destino: Path, formato: str, dpi: float) -> None:
    """Escreve a figura no formato pedido, com os mesmos pinos de `savefig` nos três formatos.

    Fundo e borda vêm da própria figura, `transparent=False`, e `savefig.bbox` fica em `None` por
    `rc_context`. Passar `bbox_inches=None` não protege: `print_figure` lê o `None` como 'não
    informado' e vai ao `rcParams`, e com `"tight"` o arquivo sairia recortado à volta dos
    artistas. Declarar os três tira o resultado das mãos de `rcParams` globais que uma versão nova
    do `matplotlib` pode mudar.

    O TIFF passa pelo PNG em memória e pelo Pillow: o `savefig` em TIFF grava RGBA, e há revista
    que recusa canal alfa, então ele é achatado sobre branco, em RGB, com LZW e a tag de DPI. O PDF
    leva `pdf.fonttype` 42 (TrueType embutida, nenhuma Tipo 3, que revista e editor tratam mal) e
    nenhuma data de criação, para a mesma figura dar o mesmo arquivo.
    """
    comum: dict[str, Any] = {
        "dpi": dpi,
        "facecolor": figura.get_facecolor(),
        "edgecolor": figura.get_edgecolor(),
        "transparent": False,
    }
    if formato == "png":
        with rc_context({"savefig.bbox": None}):
            figura.savefig(destino, format="png", **comum)
        return
    if formato == "pdf":
        with rc_context({"savefig.bbox": None, "pdf.fonttype": 42}):
            figura.savefig(destino, format="pdf", metadata={"CreationDate": None}, **comum)
        return
    from io import BytesIO

    from PIL import Image

    memoria = BytesIO()
    with rc_context({"savefig.bbox": None}):
        figura.savefig(memoria, format="png", **comum)
    memoria.seek(0)
    with Image.open(memoria) as aberta:
        # `convert` garante o canal alfa que a máscara lê, venha o PNG com ele ou sem ele.
        rgba = aberta.convert("RGBA")
        branco = Image.new("RGB", rgba.size, (255, 255, 255))
        branco.paste(rgba, mask=rgba.getchannel("A"))
    branco.save(destino, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))


def exporta_habilitado(estado: EstadoDaFigura) -> bool:
    """Se o controle de gravar deve estar habilitado neste estado. Pura, e total sobre o enum.

    Função do enum, e não um `isEnabled()` lido do widget, para os testes percorrerem todos os
    estados sem um condutor escrito à mão para cada um. Só `DESENHADA` habilita, e a forma do `is`
    é o conteúdo: um `not in {...}` faria um estado novo nascer habilitado, e clicar gravaria a
    `Figure()` vazia que `_retira` põe no lugar.
    """
    return estado is EstadoDaFigura.DESENHADA


def arquivo_por_dialogo(pai: QWidget | None = None) -> EscolhedorDeArquivo:
    """O escolhedor de produção: `QFileDialog.getSaveFileName`, e nada mais.

    Fábrica, no molde de `pasta_por_dialogo` (`janela/artefatos.py`), e injetada porque um
    `QFileDialog` real abre janela modal, e um arnês que não o substitua pendura.

    String vazia é o cancelamento, e a tradução para `None` é explícita: `Path("")` é o diretório
    corrente, e a gravação escreveria num caminho que ninguém escolheu. `OSError` não é tratado
    aqui, senão o `None` cobriria 'cancelou' e 'a escrita falhou', que pedem ações opostas; quem
    trata é `PainelDeFigura.exporta`, que sabe nomear o caminho na mensagem.
    """
    from pathlib import Path as _Path

    from PySide6.QtWidgets import QFileDialog

    def escolhe(nome_sugerido: str) -> _Path | None:
        # Os três formatos no filtro; quem decide o formato é o sufixo do nome, que a exportação
        # confere antes de escrever.
        escolhido, _filtro = QFileDialog.getSaveFileName(
            pai, "Salvar figura", nome_sugerido, "PNG (*.png);;TIFF (*.tif *.tiff);;PDF (*.pdf)"
        )
        return _Path(escolhido) if escolhido else None

    return escolhe


def entradas_do_seletor(
    construtores: Sequence[ConstrutorDeFigura] = CONSTRUTORES,
) -> tuple[tuple[str, ConstrutorDeFigura], ...]:
    """`(rótulo, construtor)` para cada figura registada, na ordem do registo.

    Derivada do registo, nunca redigitada: uma lista escrita aqui omitiria a figura nova em
    silêncio, e o portão de bijeção corre sobre o registo, não sobre este combo.

    O rótulo vem do nome da função (`c.funcao`), que é único no registo. O par `modulo · chave` não
    serve: várias entradas dariam 'pca · PCA', e o combo mostraria linhas iguais para figuras
    diferentes.
    """
    return tuple((_nome_legivel(c.funcao), c) for c in construtores)


def _nome_legivel(funcao: str) -> str:
    """`figura_dos_scores_do_plsda` vira `scores do plsda`. Delega ao catálogo, que é o dono.

    A regra mora em `plots.catalogo.rotulo_do_construtor`, que não conhece Qt: o portão da
    unicidade dos rótulos corre lá, sem widget, sobre o registo inteiro. Uma segunda cópia aqui
    concordaria só por coincidência de texto, e um artigo acrescentado de um lado não chegaria ao
    outro.
    """
    return rotulo_do_construtor(funcao)


def cores_do_tema(paleta: QPalette) -> tuple[str, str, str]:
    """`(fundo da figura, fundo dos eixos, tinta)` lidos da paleta vigente.

    `Window` é o fundo do painel, `Base` o de uma área de conteúdo e `WindowText` a cor com que o
    sistema escreve sobre o primeiro. Nenhuma das três é literal: a função é total sobre a paleta
    que recebe.

    Não tem chamador de produção, porque a figura usa a trinca fixa `CORES_DA_FIGURA`. Fica como o
    caminho de volta a seguir o tema: a trinca que devolve encaixa em `aplica_o_tema` sem mais nada
    mudar.
    """
    return (
        paleta.color(QPalette.ColorRole.Window).name(),
        paleta.color(QPalette.ColorRole.Base).name(),
        paleta.color(QPalette.ColorRole.WindowText).name(),
    )


def aplica_o_tema(figura: Figure, cores: tuple[str, str, str]) -> None:
    """Pinta a figura inteira com a trinca `(fundo, área dos eixos, tinta)`. Idempotente.

    Percorre tudo o que o `matplotlib` pinta de preto ou de branco por omissão: o fundo da figura,
    o fundo e as espinhas de cada eixo, as marcas, os rótulos, o título, a legenda e os textos de
    figura, onde vive o rodapé do recorte. Recebe a trinca, e não a `QPalette`, porque quem escolhe
    as cores é quem chama; a forma é a que `cores_do_tema` devolve.

    Não toca a cor das séries: o ciclo de cores do `matplotlib` é legível nos dois fundos, e
    repintá-lo apagaria a distinção entre séries.
    """
    fundo, area, tinta = cores
    figura.set_facecolor(fundo)
    figura.set_edgecolor(fundo)
    for texto in figura.texts:
        texto.set_color(tinta)
    for eixo in figura.axes:
        eixo.set_facecolor(area)
        # Os textos dentro do eixo são outra coleção, que `figura.texts` não alcança: as frases
        # de ausência declarada dos construtores e os números das setas do biplot. Repintá-los aqui
        # faz de `CORES_DA_FIGURA` a única fonte da tinta.
        for texto in eixo.texts:
            texto.set_color(tinta)
        for espinha in eixo.spines.values():
            espinha.set_color(tinta)
        eixo.tick_params(colors=tinta, which="both")
        eixo.xaxis.label.set_color(tinta)
        eixo.yaxis.label.set_color(tinta)
        eixo.title.set_color(tinta)
        legenda = eixo.get_legend()
        if legenda is None:
            continue
        moldura = legenda.get_frame()
        moldura.set_facecolor(area)
        moldura.set_edgecolor(tinta)
        for texto in legenda.get_texts():
            texto.set_color(tinta)


def dpi_da_tela(widget: QWidget) -> float:
    """Os pontos por polegada lógicos da tela em que este widget está.

    `logicalDotsPerInch`, e não `devicePixelRatio`: o canvas já acompanha o `devicePixelRatio`
    para a nitidez, mas não o fator de escala do sistema, e `figure.dpi` fica nos 100 de
    `rcParams`. A 150%, os 10 pt de um rótulo da figura sairiam pequenos ao lado de widgets que
    cresceram 50%.

    Lê a métrica do próprio widget, e não `widget.screen()`: no PySide6 6.11 o objeto que
    `screen()` devolve nasce filho do widget, e destruir widgets de figura invalidava o objeto da
    tela, que é global, corrompendo o heap na consulta seguinte. `logicalDpiX` e `logicalDpiY` são
    lidos no C++ sem objeto de tela, valem para a tela em que a janela está e existem mesmo para
    widget ainda não mostrado; a média dos dois é o que `logicalDotsPerInch` devolve.
    """
    return (widget.logicalDpiX() + widget.logicalDpiY()) / 2.0


class PainelDeFigura(QWidget):
    """A tela onde uma figura de `refino.plots` aparece, com seletor, recusa e interação.

    Widget filho, e não painel composto (ver o cabeçalho do módulo). O oráculo dos testes é o
    widget, e não o valor de retorno: `estado_visivel` lê o texto e confere que o rótulo está no
    layout e visível.
    """

    exportavel = Signal(bool)
    """Emitido a cada mudança de estado com `exporta_habilitado(estado)`.

    O controle de gravar vive em `PainelDeGraficos`; sem o sinal ele teria de perguntar o estado.
    Sai de `desenha` e `_retira`, por onde todo estado passa, e carrega o `bool` e não o estado,
    para cada ouvinte não reaplicar o predicado.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        construtores: Sequence[ConstrutorDeFigura] = CONSTRUTORES,
        fonte: Fonte | None = None,
        escala: Callable[[], float] | None = None,
        escolhedor: EscolhedorDeArquivo | None = None,
        validade_propria: Callable[[ConstrutorDeFigura], EstadoDaFigura | None] | None = None,
        frase_propria: Callable[[ConstrutorDeFigura, EstadoDaFigura], str | None] | None = None,
        adiada: bool = False,
    ) -> None:
        """`adiada` faz a figura só ser construída com o painel à vista, como na vista de uma etapa.

        A vista da etapa é um segundo painel sobre o mesmo dono de estado, e a mesma `Figure` não
        pode ter dois canvas, então cada painel constrói a sua na thread da interface. Construir
        nas abas que ninguém vê custaria até 1,4 s nas distribuições com 6.035 caixas. Adiado, o
        pedido fica pendente e corre em `showEvent`. Desligado por padrão: a aba 'Gráficos'
        constrói na hora.
        """
        super().__init__(parent)
        self._adiada = adiada
        # Se uma troca de figura foi pedida com o painel escondido e ainda não correu. O adiado
        # nasce pendente: a primeira vez que aparece ele avalia a figura escolhida, em vez de
        # confiar no repouso da construção.
        self._pendente = adiada
        # O tamanho, em mm, com que o construtor desenhou a figura visível: o padrão do arquivo.
        # `None` até a primeira figura com eixos.
        self.tamanho_de_construcao_mm: tuple[float, float] | None = None
        entradas = entradas_do_seletor(construtores)
        # A validade própria: quem hospeda diz, por construtor, `None` quando a figura segue a
        # chave da execução, e senão o estado dela. A do diagnóstico vale pela matriz, a de uma
        # etapa pela fotografia da etapa. Sem o parâmetro, toda figura segue a execução.
        self._validade_propria = validade_propria
        # E a frase própria: o estado é um vocabulário fechado com uma frase cada, e quem hospeda
        # pode dar, por construtor, uma frase mais precisa (a de etapa desatualizada nomeia qual
        # etapa refazer). `None` fica com a frase do estado.
        self._frase_propria = frase_propria
        if entradas and fonte is None:
            raise FonteAusenteError(
                f"há {len(entradas)} figura(s) registada(s) e nenhuma fonte para as construir. "
                "Um painel assim mostraria «nenhuma figura registada» sobre um registo cheio, que "
                "é a frase errada para o fato oposto"
            )
        self._entradas = entradas
        self._fonte = fonte
        # A escala entra por injeção para o teste ver uma tela a 150% sem ter uma; o padrão lê a
        # tela no instante do desenho.
        self._escala: Callable[[], float] = escala if escala is not None else self._dpi_do_widget
        # O diálogo de gravação também entra por injeção: um `QFileDialog` modal não fecha em
        # teste, e o arnês que não o substitua pendura a suíte. A fábrica é chamada aqui; o
        # diálogo só abre no clique.
        self._escolhedor: EscolhedorDeArquivo = (
            escolhedor if escolhedor is not None else arquivo_por_dialogo(self)
        )
        # O estado corrente. `exporta` precisa de saber se há figura na tela, e ler o texto do
        # rótulo para isso seria comparar prosa. Só `_anuncia` o escreve.
        self._estado = EstadoDaFigura.SEM_EXECUCAO
        self._chave = CHAVE_DE_REPOUSO
        self._nomes: tuple[str, ...] = ()
        # Os artistas a que `self._nomes` se refere. Vazio quer dizer 'nenhum nesta figura', e aí
        # toda identificação cai no rótulo do artista (ver `_nome_do_ponto`).
        self._nomeados: tuple[Artist, ...] = ()
        self._vista_original: tuple[tuple[Axes, tuple[float, float], tuple[float, float]], ...] = ()
        self._anotacao: Annotation | None = None
        # O filtro é atributo porque o painel é o dono do tempo de vida dele; ver
        # `aparencia.sem_roda_sem_foco`, cuja docstring mede o que um filtro sem pai custa.
        self._filtro_da_roda = FiltroDaRoda(self)

        disposicao = QVBoxLayout(self)

        controles = QHBoxLayout()
        # Nome legível por cima do rótulo derivado: `currentText()` continua a ser o derivado, que
        # é o que os atalhos, o nome do arquivo e os testes leem.
        self.seletor = sem_roda_sem_foco(
            ComboComNomes(NOMES_DAS_FIGURAS, self), self._filtro_da_roda
        )
        for rotulo, _construtor in entradas:
            self.seletor.addItem(rotulo)
        # Escondido, e não desabilitado, quando o registo está vazio: com zero figuras não há entre
        # o que escolher, e o portão da roda (`test_desktop_roda.py`) exige que nenhum `QComboBox`
        # da janela esteja desabilitado, porque desabilitado engole a roda sem passar pelo filtro.
        self.seletor.setVisible(bool(entradas))
        # O seletor leva a largura da linha; o botão fica do tamanho do que diz.
        controles.addWidget(self.seletor, 1)

        # Volta aos limites guardados no desenho. Com o zoom de `_amplia` há de onde voltar, e sem
        # barra de navegação o caminho de volta não vem de graça.
        self.botao_da_vista = QPushButton("Voltar à vista original", self)
        self.botao_da_vista.clicked.connect(self._volta_a_vista_original)
        self.botao_da_vista.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        controles.addWidget(self.botao_da_vista)
        disposicao.addLayout(controles)

        # O rótulo de repouso nasce de uma função, e não de uma constante, como em
        # `interface/fluxo.py::estado_de_ausencia`: um rótulo semeado com o texto errado afirmaria
        # que
        # outra execução o preenche. O estado vazio é um bloco centrado, com ícone e largura de
        # leitura, que ocupa o lugar do canvas e some com ele; o rótulo vive no layout do bloco,
        # que é o que `_desenhado` pergunta.
        self.estado_vazio = QFrame(self)
        self.estado_vazio.setObjectName("estado_vazio")
        self.estado_vazio.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dentro_do_vazio = QVBoxLayout(self.estado_vazio)
        dentro_do_vazio.setSpacing(12)
        dentro_do_vazio.addStretch(1)
        icone_do_vazio = QLabel(self.estado_vazio)
        # A cor de destaque pelo token do tema, que a troca de esquema repinta.
        poe_icone(icone_do_vazio, ICONES["Gráficos"], "destaque_texto", 40)
        dentro_do_vazio.addWidget(icone_do_vazio, 0, Qt.AlignmentFlag.AlignHCenter)
        self.repouso = QLabel(
            texto_do_estado(estado_da_resposta(self._chave, ha_construtor=bool(entradas))),
            self.estado_vazio,
        )
        self.repouso.setObjectName("texto_do_vazio")
        self.repouso.setWordWrap(True)
        self.repouso.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.repouso.setMaximumWidth(LARGURA_DO_TEXTO_DO_VAZIO)
        # Centrado por esticadores, e não por alinhamento no `addWidget`: com alinhamento o layout
        # dá ao rótulo a altura de uma linha e a frase sai cortada.
        linha_do_texto = QHBoxLayout()
        linha_do_texto.addStretch(1)
        linha_do_texto.addWidget(self.repouso, 10)
        linha_do_texto.addStretch(1)
        dentro_do_vazio.addLayout(linha_do_texto)
        dentro_do_vazio.addStretch(1)
        # Fator 1, como o canvas: os dois nunca estão à vista ao mesmo tempo, e o que está ocupa a
        # altura que a aba dá à figura.
        disposicao.addWidget(self.estado_vazio, 1)

        # A identificação não é ligada aqui: `_retira`, logo abaixo, troca esta figura pela de
        # repouso, e o registo de eventos é da figura. Quem liga é `_poe_no_canvas`.
        self.canvas = FigureCanvasQTAgg(Figure())
        self.canvas.setParent(self)
        self.canvas.setMinimumHeight(ALTURA_MINIMA_DO_CANVAS)
        # A folga é do canvas (`addWidget(canvas, 1)`), sem `addStretch` ao fim: um esticador
        # roubaria à figura a altura que ela existe para ocupar.
        disposicao.addWidget(self.canvas, 1)

        self._retira(EstadoDaFigura.SEM_EXECUCAO)
        # Ligado depois de o combo ser preenchido: antes, cada `addItem` dispararia a troca sobre
        # um painel meio construído.
        self.seletor.currentIndexChanged.connect(self._troca_a_figura)

    # ------------------------------------------------------------------
    # A recusa por obsolescência
    # ------------------------------------------------------------------

    def seleciona(self, funcao: str) -> bool:
        """Põe no combo a figura cujo construtor se chama `funcao`. Devolve se achou.

        É a porta dos atalhos da aba 'Corrigir'. Não desenha por conta própria: a troca do combo
        passa por `_troca_a_figura`, que consulta a chave antes, então o atalho obedece à mesma
        regra que o clique.
        """
        for indice, (_rotulo, construtor) in enumerate(self._entradas):
            if construtor.funcao == funcao:
                self.seletor.setCurrentIndex(indice)
                return True
        return False

    def funcao_escolhida(self) -> str | None:
        """O nome do construtor que o combo mostra, ou `None` com o registo vazio.

        É o par de `seleciona`: a vista de uma etapa passa-o à aba 'Gráficos' para ela mostrar a
        mesma figura. Lido daqui, e não do texto do combo, que é rótulo derivado.
        """
        indice = self.seletor.currentIndex()
        if not 0 <= indice < len(self._entradas):
            return None
        return self._entradas[indice][1].funcao

    def reavalia(self) -> str:
        """Refaz a figura escolhida quando chega um diagnóstico. Devolve o texto.

        A figura do diagnóstico vale sem execução, então a chegada dele pode mudar a tela sem que o
        combo mude. A decisão continua a ser de `_troca_a_figura`.
        """
        return self._troca_a_figura()

    def _estado_proprio(self) -> EstadoDaFigura | None:
        """`None` quando a figura escolhida segue a execução; senão, o estado próprio dela."""
        indice = self.seletor.currentIndex()
        if self._validade_propria is None or not 0 <= indice < len(self._entradas):
            return None
        return self._validade_propria(self._entradas[indice][1])

    def define_execucao(self, resposta: Resposta) -> str:
        """A identidade da execução corrente; retira a figura se ela mudar. Devolve o texto.

        Segue a forma de `PainelDeArtefatos.define_entrada` (retorno antecipado na igualdade,
        depois limpar), mas `chave_de_invalidacao` devolve um marcador irrepetível quando não há
        chave: duas ausências nunca comparam iguais, e o retorno antecipado não absorve um erro nem
        um `202`. Por isso é chamado para toda resposta de `/execucoes`, e não só no `200`.
        """
        return self._passa_a_chave(chave_de_invalidacao(resposta))

    def esquece_a_execucao(self) -> str:
        """A execução corrente deixou de valer sem resposta que a substitua.

        A janela pede isto na troca de matriz, quando a que chega não tem execução guardada e o
        painel mostrava a de outra. Volta ao repouso ('Nenhuma execução ainda') pelo caminho de
        `define_execucao`, com a chave de repouso, e devolve o texto escrito.
        """
        return self._passa_a_chave(CHAVE_DE_REPOUSO)

    def _passa_a_chave(self, chave: str) -> str:
        """O corpo de `define_execucao`, partilhado com `esquece_a_execucao`."""
        if chave == self._chave:
            return self.repouso.text()
        self._chave = chave
        estado = estado_da_resposta(chave, ha_construtor=bool(self._entradas))
        # A figura com validade própria (diagnóstico, etapa) não sai com a execução, e só é
        # reconstruída se o estado próprio mudou: a chave de todo não-200 é irrepetível, e cada
        # releitura do progresso reconstruiria a figura na thread da interface, apagando o zoom e
        # a anotação. Quem muda os dados dessas figuras chama `reavalia`.
        proprio = self._estado_proprio()
        if proprio is not None:
            if proprio is self._estado:
                return self.repouso.text()
            return self._troca_a_figura()
        if estado is EstadoDaFigura.DESENHADA:
            return self._troca_a_figura()
        self._retira(estado)
        return texto_do_estado(estado)

    # ------------------------------------------------------------------
    # O desenho
    # ------------------------------------------------------------------

    def desenha(self, figura: Figure, *, nomes: Sequence[str] = ()) -> None:
        """Põe esta figura no canvas, com a trinca fixa de cores e a escala da tela do painel.

        `nomes` é o vetor de nomes de feature na ordem em que o construtor pôs os pontos, para a
        identificação dizer qual feature é. Vazio é legítimo: um scree ou um histograma identifica
        pelo rótulo da série.

        O vetor indexa os pontos dos artistas nomeados, que o construtor marca com
        `GID_DAS_FEATURES`, e não a figura inteira: aplicado a outro artista, como a cruz da origem
        numa figura de cargas, ele daria o nome de uma feature errada. Vários artistas podem ser
        nomeados, porque uma figura de antes e depois põe duas séries paralelas sobre as mesmas
        features.

        A vista original é guardada aqui: o limite que o construtor derivou do dado é o que o
        profissional viu antes de aproximar.
        """
        aplica_o_tema(figura, CORES_DA_FIGURA)
        figura.set_dpi(self._escala())
        artistas = self._artistas(figura)
        for artista in artistas:
            artista.set_picker(True)

        self._anotacao = None
        self._nomes = tuple(nomes)
        # Resolvido uma vez, por identidade: a pergunta é se este artista é um daqueles a que o
        # vetor de nomes se refere, e ela é sobre a figura desenhada.
        self._nomeados = tuple(a for a in artistas if a.get_gid() == GID_DAS_FEATURES)
        self._poe_no_canvas(figura)
        self._vista_original = tuple(
            (eixo, _par(eixo.get_xlim()), _par(eixo.get_ylim())) for eixo in figura.axes
        )
        self.repouso.setText(texto_do_estado(EstadoDaFigura.DESENHADA))
        self.estado_vazio.setVisible(False)
        self.canvas.setVisible(True)
        self._anuncia(EstadoDaFigura.DESENHADA)
        self._redesenha()

    def estado_visivel(self) -> dict[str, str]:
        """O que o painel mostra agora; é o oráculo dos testes.

        Cada entrada é vazia quando o widget não está desenhado (fora do layout ou escondido): um
        rótulo que existe e não está no layout é um painel em branco.
        """
        return {
            "repouso": self.repouso.text() if self._desenhado(self.repouso) else "",
            "seletor": self.seletor.currentText() if self._desenhado(self.seletor) else "",
            "anotacao": self._texto_da_anotacao(),
            "figura": (
                f"{len(self.canvas.figure.axes)} eixo(s)" if self._desenhado(self.canvas) else ""
            ),
        }

    # ------------------------------------------------------------------
    # A exportação
    # ------------------------------------------------------------------

    def _anuncia(self, estado: EstadoDaFigura) -> None:
        """Guarda o estado corrente e diz a quem ouve se a gravação está disponível.

        `desenha` e `_retira` chamam isto, e nenhum outro sítio escreve `self._estado`: uma linha
        repetida em cada ramo acaba por faltar num deles, e esse fica com o botão vivo sobre um
        canvas sem figura.
        """
        self._estado = estado
        self.exportavel.emit(exporta_habilitado(estado))
        # O botão de voltar segue o mesmo sítio único, e só se desliga com a frase ao lado
        # (`desliga_com_motivo`): um botão apagado sem explicação não diz o que fazer.
        if vista_habilitada(estado):
            religa(self.botao_da_vista)
        else:
            desliga_com_motivo(self.botao_da_vista, MOTIVO_SEM_VISTA)

    def nome_sugerido(self) -> str:
        """O nome que o diálogo propõe: o rótulo da figura e a identidade da execução.

        Com o rótulo só, gravar as figuras de duas execuções faria a segunda sobrescrever a
        primeira sem aviso. Vão doze caracteres da chave (um `sha256` em hexadecimal): a chave
        inteira deixaria o nome ilegível e perto do teto de caminho do Windows. Espaço vira hífen
        porque nome de arquivo com espaço atravessa mal a linha de comando.
        """
        rotulo = self.seletor.currentText().replace(" ", "-")
        return f"{rotulo}-{self._chave[:12]}{SUFIXO_DA_EXPORTACAO}"

    def _reajusta_e_mede(self) -> float | None:
        """Refaz o rodapé no tamanho atual da figura e mede a folga dele.

        Numa altura pequena demais o `tight_layout` avisa que os eixos não cabem, e é isso mesmo
        que se mede; o aviso fica calado aqui, e só aqui, para não encher a saída de erro.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Tight layout not applied")
            self._reajusta_o_recorte(None)
            return folga_do_rodape(self.canvas.figure)

    def altura_minima_mm(self, largura_mm: float) -> float | None:
        """A menor altura, em mm inteiros, em que o rodapé não encosta no gráfico.

        Medida, e não fixa: a figura visível é posta no tamanho, o rodapé refeito por `reajusta`, e
        `folga_do_rodape` diz se colide; `menor_altura_que_cabe` acha o menor inteiro entre 20 e
        600 mm. `None` quando a figura não tem rodapé ou quando nem 600 mm bastam. O canvas não
        muda: o tamanho e o rodapé da tela voltam no `finally`.

        O número é a fronteira entre as duas saídas da exportação, com e sem rodapé, e tem de
        estar certo dos dois lados: nele o rodapé cabe, um mm abaixo não.
        """
        figura = self.canvas.figure
        if not figura.axes:
            return None
        tela_largura, tela_altura = (float(v) for v in figura.get_size_inches())

        def cabe(altura_mm: int) -> bool | None:
            figura.set_size_inches(
                largura_mm / MM_POR_POLEGADA, altura_mm / MM_POR_POLEGADA, forward=False
            )
            folga = self._reajusta_e_mede()
            return None if folga is None else folga > 0.0

        try:
            minima = menor_altura_que_cabe(cabe)
            return None if minima is None else float(minima)
        finally:
            figura.set_size_inches(tela_largura, tela_altura, forward=False)
            self._reajusta_o_recorte(None)

    def exporta(
        self,
        *,
        dpi: float = DPI_DA_EXPORTACAO,
        tamanho_mm: tuple[float, float] | None = None,
    ) -> tuple[DesfechoDaExportacao, str]:
        """Grava a figura visível no formato, no DPI e no tamanho pedidos, e devolve o desfecho.

        O tamanho padrão é o da construção da figura, e não o da janela, que muda com a tela. O
        rodapé é refeito no tamanho pedido por `reajusta`, com a colisão medida antes de gravar. A
        figura na tela não muda: o tamanho volta no `finally`, e o `matplotlib` repõe o `dpi`
        sozinho dentro de `print_figure`. Os `rcParams` de `savefig` ficam pinados em `_grava`.

        Nada sobe para o slot do Qt, onde uma exceção pararia o painel em silêncio. O formato e o
        DPI são conferidos antes de escrever; `OSError` (outra pasta) é apanhado, e também
        `MemoryError` e o `DecompressionBombError` do Pillow (tamanho grande demais: um A3 a 1.200
        ppp passa do limite de pixels). `OSError`, e não `Exception`, para não esconder defeito de
        programação.

        Onde o rodapé não cabe, a figura sai sem ele (`recorte.retira`) e o texto vai para
        `caminho_do_rodape(destino)`, para a legenda. O `.txt` é escrito antes da figura: se ele
        falhar, nada foi escrito; o contrário perderia a declaração do recorte sem aviso.
        """
        if not exporta_habilitado(self._estado):
            return DesfechoDaExportacao.SEM_FIGURA, texto_do_desfecho(
                DesfechoDaExportacao.SEM_FIGURA, texto_do_estado(self._estado)
            )
        # O DPI é vocabulário fechado, e a guarda vem antes do diálogo: um valor fora dos três
        # chegaria ao `savefig` e gravaria outra coisa que a pedida.
        if dpi not in DPIS_DA_EXPORTACAO:
            return DesfechoDaExportacao.DPI_RECUSADO, texto_do_desfecho(
                DesfechoDaExportacao.DPI_RECUSADO, f"{dpi:g}"
            )
        destino = self._escolhedor(self.nome_sugerido())
        if destino is None:
            return DesfechoDaExportacao.CANCELADA, texto_do_desfecho(
                DesfechoDaExportacao.CANCELADA, ""
            )
        formato = FORMATOS_DA_EXPORTACAO.get(destino.suffix.casefold())
        if formato is None:
            # 'nada' por extenso: `Path("figura")` tem `suffix == ""`, e a frase ficaria sem
            # palavra.
            sufixo = destino.suffix or "nada"
            return DesfechoDaExportacao.SUFIXO_RECUSADO, texto_do_desfecho(
                DesfechoDaExportacao.SUFIXO_RECUSADO, sufixo
            )
        figura = self.canvas.figure
        largura_mm, altura_mm = tamanho_mm or self.tamanho_de_construcao_mm or _mm_da(figura)
        tela_largura, tela_altura = (float(v) for v in figura.get_size_inches())
        rodape_em: Path | None = None
        escrevendo = destino
        try:
            # A figura da tela, com o zoom que a pessoa vê, posta no tamanho do arquivo sem tocar o
            # widget (`forward=False`), com o rodapé refeito e a colisão medida antes de escrever.
            # O `finally` devolve o tamanho da tela.
            figura.set_size_inches(
                largura_mm / MM_POR_POLEGADA, altura_mm / MM_POR_POLEGADA, forward=False
            )
            folga = self._reajusta_e_mede()
            if folga is not None and folga <= 0.0:
                # O rodapé não cabe: sai da figura, e o texto vai antes para o `.txt`.
                # `_retira_o_rodape` só devolve `None` sem rodapé marcado, e a medição acima só
                # acha rodapé pela marca.
                texto = self._retira_o_rodape()
                if texto is not None:
                    rodape_em = escrevendo = caminho_do_rodape(destino)
                    rodape_em.write_text(f"{texto}\n", encoding="utf-8")
                    escrevendo = destino
            _grava(figura, destino, formato, float(dpi))
        except OSError as recusa:
            return DesfechoDaExportacao.ESCRITA_FALHOU, texto_do_desfecho(
                DesfechoDaExportacao.ESCRITA_FALHOU, f"{escrevendo}: {recusa}"
            )
        except (MemoryError, DecompressionBombError):
            return DesfechoDaExportacao.GRANDE_DEMAIS, texto_do_desfecho(
                DesfechoDaExportacao.GRANDE_DEMAIS, _em_pixels(largura_mm, altura_mm, float(dpi))
            )
        finally:
            # O rodapé da tela volta também quando foi retirado: `retira` deixa o texto vazio, e
            # `reajusta` refaz todo rodapé cujo texto difere do que ele quebraria.
            figura.set_size_inches(tela_largura, tela_altura, forward=False)
            self._reajusta_o_recorte(None)
        if rodape_em is not None:
            return DesfechoDaExportacao.GRAVADA_SEM_RODAPE, texto_do_desfecho(
                DesfechoDaExportacao.GRAVADA_SEM_RODAPE,
                str(destino),
                dpi=float(dpi),
                rodape=str(rodape_em),
            )
        return DesfechoDaExportacao.GRAVADA, texto_do_desfecho(
            DesfechoDaExportacao.GRAVADA, str(destino), dpi=float(dpi)
        )

    def _retira_o_rodape(self) -> str | None:
        """Tira o rodapé da figura para a gravar sem ele, e devolve o texto dele.

        Filtra pelo `gid` como `_reajusta_o_recorte`, porque `figura.texts` traz também o título de
        figura. O aviso do `tight_layout` é calado como em `_reajusta_e_mede`: é nos tamanhos
        pequenos que esta retirada corre.
        """
        figura = self.canvas.figure
        textos: list[str] = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Tight layout not applied")
            for texto in figura.texts:
                if texto.get_gid() == GID_DO_RODAPE:
                    retirado = retira(figura, texto)
                    if retirado is not None:
                        textos.append(retirado)
        return "\n".join(textos) if textos else None

    # ------------------------------------------------------------------
    # O que acontece por dentro
    # ------------------------------------------------------------------

    def _troca_a_figura(self) -> str:
        """Constrói a figura escolhida no combo e desenha-a. Devolve o texto de repouso.

        Duas portas chegam a `desenha()`, esta (o combo) e `define_execucao`, e as duas consultam a
        chave. O portão de obsolescência vem primeiro, na ordem de `estado_da_resposta`: sem ele,
        depois de um `202` ou de um erro, escolher outra figura desenharia de novo com os números
        da execução anterior, ou cairia em `SEM_DADO` sobre uma execução que falhou.

        Num painel adiado e escondido nada corre: o pedido fica pendente e `showEvent` o retoma.
        """
        if self._adiada and not self.isVisible():
            self._pendente = True
            return self.repouso.text()
        self._pendente = False
        # As figuras do diagnóstico e as de etapa têm obsolescência própria: a da matriz e a da
        # fotografia da etapa. Para as outras vale a chave da execução.
        estado_da_chave = self._estado_proprio() or estado_da_resposta(
            self._chave, ha_construtor=bool(self._entradas)
        )
        if estado_da_chave is not EstadoDaFigura.DESENHADA:
            self._retira(estado_da_chave)
            return self.repouso.text()
        indice = self.seletor.currentIndex()
        if self._fonte is None or not 0 <= indice < len(self._entradas):
            self._retira(EstadoDaFigura.SEM_CONSTRUTOR)
            return texto_do_estado(EstadoDaFigura.SEM_CONSTRUTOR)
        _rotulo, construtor = self._entradas[indice]
        try:
            figura, nomes = self._fonte(construtor)
        except FonteAusenteError:
            # Apanhada porque isto é um slot do Qt: levantada, a exceção iria para o `stderr` e o
            # painel pararia de atualizar em silêncio. Fica num `except` separado do de baixo
            # porque é defeito de ligação nosso, e não fato do dado.
            self._retira(EstadoDaFigura.LIGACAO_QUEBRADA)
            return texto_do_estado(EstadoDaFigura.LIGACAO_QUEBRADA)
        except RecusaDoConstrutorError as recusa:
            # Apanhada pela mesma razão de slot do Qt. A base, e não só `MatrizVaziaError`, porque
            # cada figura tem recusas próprias (um `sampleType` inválido levanta
            # `PcaDesalinhadaError`), e sem isto a tela ficaria muda com a figura anterior. A base,
            # e não `ValueError`, porque um `except` largo absorveria defeito de programação. O
            # subtipo escolhe o texto: população vazia e dado que não bate têm consertos opostos.
            estado = (
                EstadoDaFigura.SEM_DADO
                if isinstance(recusa, MatrizVaziaError)
                else EstadoDaFigura.DADO_RECUSADO
            )
            self._retira(estado)
            return texto_do_estado(estado)
        self.desenha(figura, nomes=nomes)
        return texto_do_estado(EstadoDaFigura.DESENHADA)

    def _retira(self, estado: EstadoDaFigura) -> None:
        """Tira a figura da tela e põe no lugar dela o texto que diz o porquê.

        A figura é descartada, e não só escondida: um canvas escondido com a figura velha voltaria
        a mostrá-la se alguém o tornasse visível sem redesenhar.
        """
        self._anotacao = None
        self._nomes = ()
        # Os artistas nomeados descrevem a figura que sai, como `self._nomes`.
        self._nomeados = ()
        self._vista_original = ()
        # Só troca quando há figura a descartar: `_poe_no_canvas` não desliga a figura antiga do
        # canvas, e trocar uma `Figure()` vazia por outra deixaria uma órfã inútil. `figure.axes` é
        # o discriminante, porque uma figura vazia não tem eixos; 'o canvas está visível' não
        # serve, porque num widget nunca mostrado é falso.
        if self.canvas.figure.axes:
            self._poe_no_canvas(Figure())
        self.canvas.setVisible(False)
        self.repouso.setText(self._texto_de(estado))
        self.estado_vazio.setVisible(True)
        self._anuncia(estado)

    def showEvent(self, evento: QShowEvent) -> None:  # noqa: N802 - nome do Qt
        """O painel adiado aparece: a troca pedida enquanto ele estava escondido corre agora.

        Só com algo pendente: aparecer sem mudança não reconstrói, e o zoom e a anotação ficam.
        """
        super().showEvent(evento)
        if self._pendente:
            self._troca_a_figura()

    def _texto_de(self, estado: EstadoDaFigura) -> str:
        """A frase do estado, ou a que quem hospeda define para a figura escolhida."""
        indice = self.seletor.currentIndex()
        if self._frase_propria is not None and 0 <= indice < len(self._entradas):
            frase = self._frase_propria(self._entradas[indice][1], estado)
            if frase is not None:
                return frase
        return texto_do_estado(estado)

    def _poe_no_canvas(self, figura: Figure) -> None:
        """Troca a figura do canvas e volta a ligar a identificação. Toda troca passa por aqui.

        O registo de eventos é da figura, não do canvas: `canvas.callbacks` devolve
        `figura._canvas_callbacks`. Trocar a figura apaga toda ligação feita por `mpl_connect`, sem
        erro e com o canvas a desenhar na mesma; por isso a religação mora num sítio só.

        O dimensionamento também. `canvas.figure = figura` é atribuição simples, e o único
        sincronizador do `matplotlib` é o `resizeEvent` do canvas, que trocar a figura não dispara:
        sem isto a figura nova ficaria no tamanho de construção, cortada ou com área morta. A conta
        copia a do `resizeEvent` (pixels físicos sobre os pontos por polegada da figura), com
        `forward=False` porque quem manda no tamanho é o widget.

        A guarda de `figura.axes` deixa de fora a `Figure()` vazia que `_retira` põe no lugar. O
        rodapé é reajustado logo depois de dimensionar, porque a faixa dele é fração da altura e o
        texto mede-se em pontos.
        """
        self.canvas.figure = figura
        figura.set_canvas(self.canvas)
        self._liga_o_pick()
        if figura.axes:
            # O tamanho de construção, guardado antes de o widget o trocar, é o padrão do arquivo
            # exportado.
            largura, altura = (float(v) for v in figura.get_size_inches())
            self.tamanho_de_construcao_mm = (
                largura * MM_POR_POLEGADA,
                altura * MM_POR_POLEGADA,
            )
            proporcao = self.canvas.device_pixel_ratio
            figura.set_size_inches(
                self.canvas.width() * proporcao / figura.dpi,
                self.canvas.height() * proporcao / figura.dpi,
                forward=False,
            )
            self._reajusta_o_recorte(None)

    def _liga_o_pick(self) -> None:
        """Liga `_identifica` ao registo de eventos da figura que está no canvas agora.

        Liga também o reajuste do recorte e o zoom, pela mesma razão: o registo vive na figura, e
        trocar a figura apaga as ligações juntas.
        """
        self.canvas.mpl_connect("pick_event", self._identifica)
        self.canvas.mpl_connect("resize_event", self._reajusta_o_recorte)
        self.canvas.mpl_connect("scroll_event", self._amplia)

    def _reajusta_o_recorte(self, _evento: object) -> None:
        """Refaz a faixa do rodapé quando o canvas muda de tamanho.

        `plots.recorte.aplica` calcula a faixa a partir da altura da figura na construção. O canvas
        redimensiona a figura para caber no widget, e a fração fixada deixa de bastar: o texto
        mede-se em pontos, que são absolutos, e a faixa é fração da altura. Sem o reajuste, numa
        janela menor que a de construção, o rótulo do eixo sai impresso por cima do rodapé.

        Ligado ao `resize_event`, e não a `_redesenha`, que só corre em gestos que não mudam o
        tamanho. Não redesenha: quem emite `resize_event` é o `matplotlib`, que desenha logo a
        seguir.
        """
        figura = self.canvas.figure
        # Só o rodapé, pelo `gid`: o `matplotlib` põe o `_suptitle` dentro de `figura.texts`, e sem
        # o filtro o reajuste trocaria o título das figuras que têm título de figura pelo texto do
        # rodapé. `GID_DO_RODAPE` é declarado por `plots.recorte.aplica`; comparar com o
        # `_suptitle` dependeria de atributo privado. `aplica_o_tema` não filtra, porque lá pintar
        # o título com a mesma tinta é o que se quer.
        for texto in figura.texts:
            if texto.get_gid() == GID_DO_RODAPE:
                reajusta(figura, texto)

    def _redesenha(self) -> None:
        """Redesenha o canvas com `draw()`, e não `draw_idle()`, para evitar uma queda.

        `draw_idle` agenda `QTimer.singleShot(0, ...)` sobre um método do canvas. Se o widget for
        destruído antes de o temporizador correr, como quando a janela fecha com um desenho
        pendente, ele acorda sobre um objeto C++ já apagado: `libshiboken` reclama e o processo
        pode cair com `access violation`. O `_draw_idle` do `matplotlib` engole a exceção e só
        imprime, o que faz o primeiro sintoma parecer ruído.

        A coalescência que se perde não faz falta: as chamadas vêm de gestos humanos (execução
        recebida, identificação, volta à vista). O `resizeEvent` do `matplotlib` continua a usar
        `draw_idle`.
        """
        self.canvas.draw()

    def _identifica(self, evento: PickEvent) -> None:
        """Anota o ponto sob o cursor com o nome da feature e o valor.

        Abre com a guarda do `widgetlock`, como todo manipulador deste painel. `Figure.pick` já não
        despacha com o cadeado tomado (`matplotlib/figure.py:2666`), então um teste da guarda tem
        de chamar este manipulador direto; ela cobre qualquer `PickEvent` que não passe por ali.

        A anotação anterior é apagada antes da nova. O `matplotlib` dispara um evento por artista
        na ordem de desenho, então o último a chegar é o de cima, e é o dele que fica.
        """
        if not self.canvas.widgetlock.available(self):
            return
        # `is None` e não `or ()`: `evento.ind` é um `ndarray`, e `np.array([0]) or ()` avalia
        # `bool(0)`, que é falso. Com `or`, o primeiro ponto de cada série nunca seria identificado.
        crus = getattr(evento, "ind", None)
        if crus is None:
            return
        indices = [int(i) for i in crus]
        if not indices:
            return
        posicao = indices[0]
        ponto = self._ponto_do_artista(evento.artist, posicao)
        if ponto is None:
            return
        x, y = ponto
        nome = self._nome_do_ponto(evento.artist, posicao)
        eixo = evento.artist.axes
        if not isinstance(eixo, Axes):  # pragma: no cover - um pickable está sempre num Axes
            return
        self._limpa_a_anotacao()
        # A tinta vem da trinca fixa do fundo; lida do tema escuro, sairia clara sobre branco.
        _fundo, _area, tinta = CORES_DA_FIGURA
        self._anotacao = eixo.annotate(
            # Vírgula decimal por `recorte.decimal`, como nas marcas dos eixos e no rodapé do
            # recorte; com ponto, `y=1.523` ficaria ao lado de marcas `1,5` na mesma imagem.
            # `decimal` usa o mesmo `:.4g`: só o separador muda.
            f"{nome}\nx={decimal(x)}  y={decimal(y)}",
            xy=(x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color=tinta,
        )
        self._redesenha()

    def _nome_do_ponto(self, artista: Artist, posicao: int) -> str:
        """O nome a anotar: o da feature quando o artista é nomeado, o rótulo dele quando não é.

        A comparação é por identidade, e não pelo `gid` outra vez: o que importa é se este artista
        estava na figura quando o vetor de nomes foi recebido. Um artista acrescentado depois do
        `desenha`, como a própria anotação do clique, fica de fora.

        O recuo é o rótulo do artista, nunca um nome do vetor. Um rótulo genérico ('origem de PC2',
        'amostra de estudo') o profissional reconhece como pouca informação; o nome de outra
        feature ele leria como informação.
        """
        if posicao < len(self._nomes) and any(artista is x for x in self._nomeados):
            return self._nomes[posicao]
        return str(artista.get_label())

    def _amplia(self, evento: MouseEvent) -> None:
        """Ctrl+roda aproxima e afasta o eixo sob o cursor, em torno do cursor.

        Sem Ctrl não faz nada: as abas rolam com a roda, e um zoom armado na roda nua roubaria esse
        gesto. A guarda não devolve o evento ao Qt (`FigureCanvasQT.wheelEvent` aceita o
        `QWheelEvent` sempre); ela só garante que o painel não mexe em eixo nenhum sem Ctrl.

        A conta é a de `backend_bases.scroll_handler`: escala em torno do cursor, `0,85 ** step`
        por entalhe, em coordenadas de `ax.transScale`. O gesto fica igual ao de qualquer figura do
        `matplotlib`, eixo logarítmico sai correto sem código nosso, e o passo não é constante
        nossa. Nenhum construtor registado usa eixo logarítmico (um teste confere); o `log10` do
        volcano e da PCA é transformação do dado.

        Três condições de `scroll_handler` que um canvas embutido não cumpre sozinho:
        - ele só é subscrito em `FigureManagerBase.__init__`, e canvas embutido não tem manager;
          por isso `_liga_o_pick` liga este método ao `scroll_event`;
        - ele abre com `if toolbar is None: return`, daí `HistoricoMudo`;
        - ele compara `evento.key`, que o `matplotlib` enche a partir de `key_press_event`, e isso
          exige foco de teclado no canvas, que aqui é `NoFocus` (quem põe `StrongFocus` é
          `FigureManagerQT`). Por isso o Ctrl é lido de `evento.modifiers`, que vem do teclado vivo
          (`QApplication.keyboardModifiers()`), e traduzido para `key` logo abaixo.

        Recusa sem levantar com o `widgetlock` tomado e sem Ctrl. Cursor fora de eixo, eixo não
        retilíneo e passo zero, o próprio `scroll_handler` recusa; repeti-los aqui seria um segundo
        oráculo.

        A anotação do clique fica: ela está ancorada em coordenadas de dado, e o zoom a leva junto.
        Apagá-la a cada entalhe mataria o gesto de identificar uma feature num volcano denso e
        depois aproximar para ver a vizinhança. Voltar à vista, ao contrário, desfaz a exploração e
        apaga a anotação.

        Não há teto de aproximação, como no `matplotlib`: aproximar sem fim degenera o eixo (o `x`
        do VIP chega lá em cerca de 230 entalhes), e o caminho de volta está a um clique. Um teto
        nosso seria uma constante sem origem.
        """
        if not self.canvas.widgetlock.available(self):
            return
        if MODIFICADOR_DO_ZOOM not in evento.modifiers:
            return
        # `scroll_handler` compara `evento.key`, que num canvas embutido é sempre `None` (ver a
        # docstring). Escrever aqui o nome da tecla faz a conta dele correr com o modificador que
        # o Qt entrega. `x` e `y` são preservados: com elas `scroll_handler` aproxima um só eixo,
        # e sobrescrevê-las mudaria o gesto no dia em que chegarem.
        anterior = evento.key
        if anterior is None:
            evento.key = TECLA_DO_ZOOM
        try:
            # O `cast` declara que o contrato é menor que o tipo: `scroll_handler` anota
            # `NavigationToolbar2 | None` e só chama `push_current()` no objeto. Uma
            # `NavigationToolbar2` de verdade traria a barra que o usuário rejeitou.
            scroll_handler(evento, toolbar=cast("NavigationToolbar2", HistoricoMudo()))
        finally:
            # O evento é partilhado com os ouvintes seguintes do mesmo despacho; deixar `key`
            # escrito faria um ouvinte a jusante ler uma tecla que ninguém premiu.
            evento.key = anterior

    def _volta_a_vista_original(self) -> None:
        """Devolve os limites que o construtor derivou do dado e apaga a anotação.

        Usa os limites guardados, e não `autoscale`: a anotação do clique é um artista com posição,
        e um `autoscale` a incluiria, fazendo a vista original depender de onde se clicou.
        """
        if not self.canvas.widgetlock.available(self):
            return
        for eixo, xlim, ylim in self._vista_original:
            eixo.set_xlim(xlim)
            eixo.set_ylim(ylim)
        self._limpa_a_anotacao()
        self._redesenha()

    def _limpa_a_anotacao(self) -> None:
        """Tira da tela a anotação da identificação anterior, se houver."""
        if self._anotacao is None:
            return
        self._anotacao.remove()
        self._anotacao = None

    def _texto_da_anotacao(self) -> str:
        """O texto da anotação corrente, ou vazio. Vazio nos dois casos, de propósito.

        'Não há anotação' e 'a anotação está em branco' dão a mesma resposta a quem pergunta se o
        nome da feature está na tela: não.
        """
        if self._anotacao is None:
            return ""
        return str(self._anotacao.get_text())

    def _dpi_do_widget(self) -> float:
        """O padrão de `escala`: os pontos por polegada da tela em que este painel está.

        Um método, e não `dpi_da_tela` direto, porque o padrão de um parâmetro é avaliado na
        construção e o painel pode mudar de tela depois: a leitura tem de acontecer no desenho.
        """
        return dpi_da_tela(self)

    @staticmethod
    def _artistas(figura: Figure) -> list[Artist]:
        """Todo artista de dado da figura: as linhas e as coleções de cada eixo.

        É a mesma população que `tests/arnes_de_figura.series` lê, e a coincidência é requisito:
        um artista que o arnês afirma e que não fosse pickable seria um ponto visível que não
        responde ao cursor.
        """
        return [artista for eixo in figura.axes for artista in [*eixo.lines, *eixo.collections]]

    @staticmethod
    def _ponto_do_artista(artista: Artist, posicao: int) -> tuple[float, float] | None:
        """As coordenadas do ponto `posicao` daquele artista, ou `None` se ele não as tem.

        Linha e coleção guardam o dado de formas diferentes, e o `pick_event` não distingue as
        duas: o índice que ele traz é o do artista, seja ele qual for.
        """
        dados_x = getattr(artista, "get_xdata", None)
        if callable(dados_x):
            xs = list(dados_x())
            ys = list(artista.get_ydata())  # type: ignore[attr-defined]
            if posicao < len(xs) and posicao < len(ys):
                return float(xs[posicao]), float(ys[posicao])
            return None
        pontos = getattr(artista, "get_offsets", None)
        if callable(pontos):
            todos = pontos()
            if posicao < len(todos):
                return float(todos[posicao][0]), float(todos[posicao][1])
        return None

    def _desenhado(self, widget: QWidget) -> bool:
        """Se este widget está no layout e visível para o painel.

        As duas metades contam: fora do layout ele não é desenhado ainda que visível, e escondido
        não é desenhado ainda que no layout. Sob `offscreen` e sem `show()`, `isVisible` devolve
        `False` para tudo; `isVisibleTo` responde à pergunta certa.

        A busca desce pelos layouts aninhados, porque `QLayout.indexOf` só vê os itens diretos, e o
        seletor vive numa linha horizontal dentro da pilha.
        """
        pai = widget.parentWidget()
        raiz = pai.layout() if pai is not None else None
        if raiz is None or not self._dentro(raiz, widget):
            return False
        return widget.isVisibleTo(self)

    @classmethod
    def _dentro(cls, disposicao: QLayout, widget: QWidget) -> bool:
        """Se `widget` está neste layout ou em algum layout aninhado nele."""
        if disposicao.indexOf(widget) >= 0:
            return True
        for posicao in range(disposicao.count()):
            item = disposicao.itemAt(posicao)
            aninhado = item.layout() if item is not None else None
            if aninhado is not None and cls._dentro(aninhado, widget):
                return True
        return False

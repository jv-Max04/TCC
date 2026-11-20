"""A aba 'Preparar o estudo': os arquivos e as três leituras na mesma página.

Este módulo compõe `PainelDeArquivos` e `PainelDeLeitura` e não é um terceiro painel: os dois
continuam com as chaves que tinham, e os testes que casam cada seção do recorte com as chaves da
janela seguem válidos. Fica em `janela/`, e não em `principal.py`, para entrar no teste que confere
o dono da folga de cada layout, que exclui `principal.py`.

A página tem duas seções numeradas e a legenda dos diagnósticos num bloco fechado. As três rotas de
leitura são independentes na API, e cada uma reingere a matriz; na janela, quem impõe a ordem é a
cadeia de `JanelaPrincipal` (`ORDEM_DAS_LEITURAS`): pedir o Diagnóstico roda antes a Validação que
faltar, e pedir o Palpite roda antes as duas.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.janela.arquivos import PainelDeArquivos
from refino.desktop.janela.componentes import Recolhivel, Secao
from refino.desktop.janela.leitura import PainelDeLeitura
from refino.desktop.janela.tema import SegueAAltura
from refino.interface.fluxo import (
    INTERPRETACAO_DO_SINAL,
    INTRODUCAO_DA_INTERPRETACAO_DO_SINAL,
    TITULO_DA_INTERPRETACAO_DO_SINAL,
)

__all__ = [
    "SUBTITULO_DAS_LEITURAS",
    "TITULO_DA_LEGENDA",
    "AbaDePreparo",
]

# O subtítulo da seção 2 diz a ordem e a dependência. Nada obriga a pessoa a pedir leitura nenhuma:
# a frase diz o que acontece quando ela pede uma.
SUBTITULO_DAS_LEITURAS = (
    "Validação → Diagnóstico → Palpite, nesta ordem: o Diagnóstico precisa da Validação, e o "
    "Palpite das duas. Ao pedir um deles, o app roda antes os que faltarem."
)

# O título nomeia a pergunta, e não o conteúdo: quem chega aqui acabou de ler 'errático' no
# diagnóstico e quer saber o que aquilo quer dizer. É o título do laboratório, e mora em
# `interface/fluxo.py` com o texto dele.
TITULO_DA_LEGENDA = TITULO_DA_INTERPRETACAO_DO_SINAL


class AbaDePreparo(QWidget):
    """Os dois painéis do preparo, numa página que rola.

    Os painéis entram por parâmetro: quem os constrói é `JanelaPrincipal`, que os liga às rotas e os
    põe em `_preenchidos`. Um composer que os construísse seria uma segunda fonte sobre quais
    painéis existem, e o painel do recorte deriva da primeira.
    """

    def __init__(
        self,
        arquivos: PainelDeArquivos,
        leitura: PainelDeLeitura,
        titulos: dict[str, str],
        parent: QWidget | None = None,
    ) -> None:
        """`titulos` é `{chave: título}` de `PAINEIS`.

        A moldura de cada painel mostra o título que `PAINEIS` declara, e não um escrito aqui: o
        teste do LEIA-ME confere o de `PAINEIS`, e uma segunda frase na tela caducaria em silêncio.
        """
        super().__init__(parent)
        # Duas seções numeradas e um bloco de referência: '1' é o formulário dos arquivos, com o
        # estado do par no cabeçalho; '2' são os três cartões de resultado; e a legenda dos
        # diagnósticos é um bloco fechado, que abre num clique. Os títulos das seções são os de
        # `PAINEIS`, que o LEIA-ME cita.
        #
        # A página rola, e a área rolável nasce dentro da página, antes de ela entrar na aba (ver
        # `tema.pagina_rolavel`).
        externo = QVBoxLayout(self)
        externo.setContentsMargins(0, 0, 0, 0)
        rolagem = QScrollArea(self)
        rolagem.setWidgetResizable(True)
        rolagem.setFrameShape(QFrame.Shape.NoFrame)
        rolagem.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        conteudo = QWidget(rolagem)
        rolagem.setWidget(conteudo)
        externo.addWidget(rolagem, 1)
        raiz = QVBoxLayout(conteudo)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.setSpacing(0)
        folha = QFrame(conteudo)
        folha.setObjectName("folha_de_etapa")
        folha.setProperty("folhaDeEtapa", True)
        folha.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        secoes = QVBoxLayout(folha)
        secoes.setContentsMargins(20, 20, 20, 20)
        secoes.setSpacing(24)
        raiz.addWidget(folha)

        self.secao_dos_arquivos = Secao(
            titulos["arquivos"],
            folha,
            numero=1,
            subtitulo=(
                "As medições e a lista de amostras. Os fatores de normalização, se for usar, "
                "ficam na aba «Normalização»."
            ),
            referencia=True,
        )
        self.secao_dos_arquivos.acoes.addWidget(arquivos.selo)
        self.secao_dos_arquivos.corpo.addWidget(arquivos)
        secoes.addWidget(self.secao_dos_arquivos)

        # O subtítulo diz a ordem e a dependência das leituras; a ordem é `ORDEM_DAS_LEITURAS`, na
        # janela.
        self.secao_da_leitura = Secao(
            titulos["leitura"],
            folha,
            numero=2,
            subtitulo=SUBTITULO_DAS_LEITURAS,
            referencia=True,
        )
        self.secao_da_leitura.corpo.addWidget(leitura)
        self.secao_da_leitura.corpo.setSpacing(20)
        # A folga pertence ao fim da página, não aos cartões. Ao recolher um bloco,
        # deixar estes dois expandirem distribuía temporariamente a altura antiga
        # entre os cartões antes de a rolagem recalcular o conteúdo: a tela saltava.
        leitura.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.secao_da_leitura.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )
        secoes.addWidget(self.secao_da_leitura)
        # Texto que quebra linha dentro de cartões: a altura depende da largura, e a moldura de fora
        # não o sabe. Ver `SegueAAltura`, que tem a medição.
        self._alturas = (SegueAAltura(arquivos), SegueAAltura(leitura))

        self.legenda = self._constroi_legenda()
        self.legenda.setProperty("referencia", True)
        if (layout_da_legenda := self.legenda.layout()) is not None:
            layout_da_legenda.setContentsMargins(0, 0, 0, 0)
            layout_da_legenda.setSpacing(12)
        self.legenda.alterna.setMinimumHeight(20)
        self.secao_da_leitura.corpo.addWidget(self.legenda)
        # A folga vai para o fim da página: sobra por baixo quando cabe, e rola quando não cabe.
        raiz.addStretch(1)

    def _constroi_legenda(self) -> Recolhivel:
        """O texto do laboratório: a introdução e uma classe por bloco recolhível.

        A legenda é estática: mostra os cinco desfechos possíveis, sempre. Uma legenda que só
        explicasse a classe recebida esconderia contra o quê aquilo foi decidido, e `unassessable`
        só aparece quando não houve medição, que é quando alguém precisa saber que ele existe.

        Um bloco fechado por classe, porque o texto do laboratório tem três ou quatro perguntas por
        classe, com um exemplo numérico; quem leu 'errático' no diagnóstico abre só o dele. A ordem
        é a de `INTERPRETACAO_DO_SINAL`, a mesma de `ROTULOS_DE_DEFEITO`: os quatro desfechos de
        medição e, por último, a ausência de medição. É texto simples, e não markdown, para um
        asterisco num exemplo não virar ênfase; a pergunta de cada campo sai em negrito pelo tema
        (`pergunta_da_legenda`).
        """
        # Um bloco fechado que abre num clique: é referência, e não fluxo.
        bloco = Recolhivel(TITULO_DA_LEGENDA, self)
        moldura = bloco.corpo
        moldura.setObjectName("grade_da_legenda")
        moldura.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        coluna = QVBoxLayout(moldura)
        coluna.setContentsMargins(16, 16, 16, 16)
        coluna.setSpacing(10)

        self.introducao_da_legenda: list[QLabel] = []
        for paragrafo in INTRODUCAO_DA_INTERPRETACAO_DO_SINAL:
            rotulo = _texto(paragrafo, "introducao_da_legenda", moldura)
            self.introducao_da_legenda.append(rotulo)
            coluna.addWidget(rotulo)

        self.nomes_da_legenda: dict[str, QToolButton] = {}
        self.blocos_da_legenda: dict[str, Recolhivel] = {}
        self.campos_da_legenda: dict[str, list[tuple[QLabel, QLabel]]] = {}
        for defeito, classe in INTERPRETACAO_DO_SINAL.items():
            bloco_da_classe = Recolhivel(f"{classe.nome} — {classe.resumo}", moldura)
            bloco_da_classe.setObjectName("classe_da_legenda")
            corpo = QVBoxLayout(bloco_da_classe.corpo)
            corpo.setContentsMargins(26, 4, 0, 8)
            corpo.setSpacing(4)
            campos: list[tuple[QLabel, QLabel]] = []
            for indice, (pergunta, resposta) in enumerate(classe.campos):
                if indice:
                    corpo.addSpacing(6)
                rotulo_da_pergunta = _texto(pergunta, "pergunta_da_legenda", bloco_da_classe.corpo)
                rotulo_da_resposta = _texto(resposta, "descricao_da_legenda", bloco_da_classe.corpo)
                corpo.addWidget(rotulo_da_pergunta)
                corpo.addWidget(rotulo_da_resposta)
                campos.append((rotulo_da_pergunta, rotulo_da_resposta))
            self.nomes_da_legenda[defeito] = bloco_da_classe.alterna
            self.blocos_da_legenda[defeito] = bloco_da_classe
            self.campos_da_legenda[defeito] = campos
            coluna.addWidget(bloco_da_classe)

        return bloco


def _texto(texto: str, nome: str, pai: QWidget) -> QLabel:
    """Um parágrafo da legenda: texto simples, que quebra linha e cresce para baixo."""
    rotulo = QLabel(texto, pai)
    rotulo.setObjectName(nome)
    rotulo.setTextFormat(Qt.TextFormat.PlainText)
    rotulo.setWordWrap(True)
    rotulo.setAlignment(Qt.AlignmentFlag.AlignTop)
    return rotulo

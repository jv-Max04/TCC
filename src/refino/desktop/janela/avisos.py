"""Apresentação dos desfechos: texto integral, hierarquia visual e fechamento explícito."""

from __future__ import annotations

from typing import Final, Literal

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.janela.componentes import RotuloQueSome, repole
from refino.desktop.janela.tema import poe_icone

TipoDeAviso = Literal["info", "aviso", "erro", "sucesso"]
_TIPOS: Final = frozenset({"info", "aviso", "erro", "sucesso"})
_TRACO_DO_AVISO: Final = '<circle cx="12" cy="12" r="9"/><path d="M12 7v6M12 17v.01"/>'
# A cor do ícone de cada tipo, pelo token do tema: cores fixas ficavam, no tema claro, a menos de
# 2:1 do fundo do aviso.
_COR_DO_ICONE: Final[dict[str, str]] = {
    "info": "nota_info_icone",
    "aviso": "nota_aviso",
    "erro": "nota_erro",
    "sucesso": "nota_sucesso",
}
TEMPO_DO_AVISO_MS: Final = 7000


class _RotuloDeDesfecho(RotuloQueSome):
    """Conserva a API do rótulo e avisa a moldura quando qualquer chamador escreve nele."""

    texto_definido = Signal(str)

    def setText(self, texto: str) -> None:  # noqa: N802 - API do Qt
        super().setText(texto)
        self.texto_definido.emit(texto)


class AvisoDeDesfecho(QFrame):
    """Uma notificação sobre o conteúdo; fechar não apaga o último desfecho."""

    def __init__(self, repouso: str, pai: QWidget | None = None) -> None:
        super().__init__(pai)
        self.setObjectName("aviso_do_estudo")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._repouso = repouso
        self._proxima_apresentacao: tuple[str, TipoDeAviso, str | None] | None = None
        # Um desfecho que se registra sem aparecer: uma resposta `2xx` não pede leitura, porque o
        # resultado está no painel dela. Ela troca o texto, que continua a ser o último desfecho
        # para quem o consulta, e recolhe o aviso aberto sem abrir outro.
        self._aparece_da_vez = True
        self._origem: str | None = None
        self._reposicionando = False
        self.setProperty("tipo", "info")
        self._temporizador = QTimer(self)
        self._temporizador.setSingleShot(True)
        self._temporizador.timeout.connect(self.oculta)

        linha = QHBoxLayout(self)
        linha.setContentsMargins(14, 12, 10, 12)
        linha.setSpacing(10)
        self.icone = QLabel(self)
        self.icone.setObjectName("icone_do_aviso")
        self.icone.setFixedSize(20, 20)
        self.icone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        poe_icone(self.icone, _TRACO_DO_AVISO, _COR_DO_ICONE["info"], 18)
        linha.addWidget(self.icone, 0, Qt.AlignmentFlag.AlignTop)

        textos = QVBoxLayout()
        textos.setContentsMargins(0, 0, 0, 0)
        textos.setSpacing(4)
        self.titulo = QLabel("Mensagem do estudo", self)
        self.titulo.setObjectName("titulo_do_aviso")
        self.titulo.setWordWrap(True)
        textos.addWidget(self.titulo)
        self.corpo = QScrollArea(self)
        self.corpo.setObjectName("corpo_do_aviso")
        self.corpo.setFrameShape(QFrame.Shape.NoFrame)
        self.corpo.setWidgetResizable(True)
        self.corpo.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.corpo.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.corpo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.corpo.setMinimumWidth(0)
        self.rotulo = _RotuloDeDesfecho(repouso, vazios=frozenset({repouso}))
        self.rotulo.setObjectName("rotulo_de_desfecho")
        self.rotulo.setWordWrap(True)
        self.rotulo.setTextFormat(Qt.TextFormat.MarkdownText)
        self.rotulo.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.rotulo.setOpenExternalLinks(False)
        self.rotulo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.corpo.setWidget(self.rotulo)
        self.rotulo.hide()
        self.corpo.viewport().installEventFilter(self)
        textos.addWidget(self.corpo, 1)
        linha.addLayout(textos, 1)

        self.fechar = QToolButton(self)
        self.fechar.setObjectName("fechar_aviso")
        self.fechar.setAccessibleName("Fechar aviso")
        self.fechar.setToolTip("Fechar aviso")
        poe_icone(self.fechar, '<path d="m6 6 12 12M18 6 6 18"/>', "nota_fechar", 14)
        self.fechar.setIconSize(QSize(14, 14))
        self.fechar.setFixedSize(26, 26)
        self.fechar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fechar.clicked.connect(self.oculta)
        linha.addWidget(self.fechar, 0, Qt.AlignmentFlag.AlignTop)
        self.rotulo.texto_definido.connect(self._texto_mudou)
        self.hide()
        if pai is not None:
            pai.installEventFilter(self)

    def mostra(
        self,
        texto: str,
        *,
        titulo: str = "Mensagem do estudo",
        tipo: TipoDeAviso = "info",
        origem: str | None = None,
        aparece: bool = True,
    ) -> None:
        if tipo not in _TIPOS:
            raise ValueError(f"tipo de aviso desconhecido: {tipo!r}")
        self._proxima_apresentacao = (titulo, tipo, origem)
        self._aparece_da_vez = aparece
        self.rotulo.setText(texto)

    def oculta(self) -> None:
        """O último texto continua disponível a quem consulta o desfecho."""
        self._temporizador.stop()
        self.hide()

    def oculta_da_origem(self, origem: str) -> None:
        """Uma dependência corrigida encerra só o aviso que pertencia a ela."""
        if self._origem == origem:
            self.oculta()

    def _texto_mudou(self, texto: str) -> None:
        self._temporizador.stop()
        titulo, tipo, self._origem = self._proxima_apresentacao or (
            "Mensagem do estudo",
            "info",
            None,
        )
        self._proxima_apresentacao = None
        aparece, self._aparece_da_vez = self._aparece_da_vez, True
        self.titulo.setText(titulo)
        self.setAccessibleName(titulo)
        self.setAccessibleDescription(texto)
        self.setProperty("tipo", tipo)
        poe_icone(self.icone, _TRACO_DO_AVISO, _COR_DO_ICONE[tipo], 18)
        repole(self)
        preenchido = bool(texto.strip()) and texto != self._repouso
        if preenchido and aparece:
            self.reposiciona()
            self.show()
            self.raise_()
            # Todo aviso que não é erro some depois de um prazo, para as frases informativas não
            # ficarem por cima das abas. O erro fica até ser fechado: é o que a pessoa tem de ler.
            if tipo != "erro":
                self._temporizador.start(TEMPO_DO_AVISO_MS)
        else:
            self.hide()

    def reposiciona(self) -> None:
        """Ancora no canto superior direito da área central."""
        pai = self.parentWidget()
        if pai is None or self._reposicionando:
            return
        self._reposicionando = True
        try:
            margem = 16
            largura = max(120, min(400, pai.width() - margem * 2))
            self.setFixedWidth(largura)
            disposicao = self.layout()
            if disposicao is not None:
                disposicao.activate()
            # O corpo curto usa a altura do texto; o longo permanece completo numa área rolável.
            largura_do_texto = max(1, self.corpo.maximumViewportSize().width())
            self.rotulo.setMinimumHeight(0)
            self.rotulo.setMaximumHeight(16777215)
            altura_do_texto = max(
                self.rotulo.fontMetrics().height(),
                self.rotulo.heightForWidth(largura_do_texto),
            )
            if altura_do_texto > 128:
                largura_do_texto = max(
                    1, largura_do_texto - self.corpo.verticalScrollBar().sizeHint().width()
                )
                altura_do_texto = self.rotulo.heightForWidth(largura_do_texto)
            self.rotulo.setFixedHeight(altura_do_texto)
            self.corpo.setFixedHeight(min(128, altura_do_texto))
            self.adjustSize()
            self.move(max(margem, pai.width() - self.width() - margem), margem)
            self.raise_()
        finally:
            self._reposicionando = False

    def eventFilter(self, watched: QObject, evento: QEvent) -> bool:  # noqa: N802 - API do Qt
        if (
            evento.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest)
            and (watched is self.parentWidget() or watched is self.corpo.viewport())
            and not self.isHidden()
        ):
            self.reposiciona()
        return False

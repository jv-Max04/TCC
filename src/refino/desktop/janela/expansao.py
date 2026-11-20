"""Revela conteúdo pela altura, sem remontar nem repintar a janela inteira.

O widget mantém seu pai e seus controles. Durante os 180 ms da transição, seu
conteúdo fica na geometria natural e a moldura o recorta; os textos não são
espremidos a cada quadro. Ao terminar, as restrições originais são restauradas.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QObject,
    QRect,
    Qt,
    QVariantAnimation,
)
from PySide6.QtWidgets import QApplication, QLabel, QWidget

__all__ = ["ExpansaoSuave"]

DURACAO_MS = 180


class ExpansaoSuave(QObject):
    """Anima um widget existente; ``conclui(False)`` limpa-o imediatamente.

    ``QApplication.setProperty('refino_animacoes', False)`` desabilita movimento.
    Páginas ocultas também recebem o estado final diretamente. Não há referência
    Python ao alvo: ele é o pai Qt deste objeto, inclusive durante a destruição.
    """

    def __init__(self, alvo: QWidget) -> None:
        super().__init__(alvo)
        self._aberto = not alvo.isHidden()
        self._restaurando = False
        self._preparada = False
        self._minimo = alvo.minimumHeight()
        self._maximo = alvo.maximumHeight()
        self._layout_habilitado = True
        self._alinhamento = alvo.alignment() if isinstance(alvo, QLabel) else None
        self._animacao = QVariantAnimation(self)
        self._animacao.setDuration(DURACAO_MS)
        self._animacao.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._animacao.valueChanged.connect(self._aplica_altura)
        self._animacao.finished.connect(self.conclui)
        alvo.installEventFilter(self)

    @property
    def em_andamento(self) -> bool:
        return self._animacao.state() == QAbstractAnimation.State.Running

    def _alvo(self) -> QWidget:
        alvo = self.parent()
        assert isinstance(alvo, QWidget)
        return alvo

    def alterna(self, aberto: bool) -> None:
        self.define(aberto)

    def define(self, aberto: bool, *, animar: bool = True) -> None:
        alvo = self._alvo()
        pai = alvo.parentWidget()
        app = QApplication.instance()
        pode_animar = (
            animar
            and (pai.isVisible() if pai is not None else alvo.isVisible())
            and app is not None
            and app.property("refino_animacoes") is not False
        )
        self._aberto = aberto
        if not pode_animar:
            self.conclui(aberto)
            return
        if not self.em_andamento and aberto == (not alvo.isHidden()):
            return

        estava_oculto = alvo.isHidden()
        inicial = alvo.height() if not estava_oculto else 0
        if self.em_andamento:
            self._animacao.stop()
        else:
            self._preparada = True
            self._minimo, self._maximo = alvo.minimumHeight(), alvo.maximumHeight()
            layout = alvo.layout()
            self._layout_habilitado = layout.isEnabled() if layout is not None else True
            if layout is not None:
                layout.setEnabled(False)
            if isinstance(alvo, QLabel):
                self._alinhamento = alvo.alignment()
                # A linha de cima fica imóvel enquanto o recorte desce.
                alvo.setAlignment(
                    (alvo.alignment() & ~Qt.AlignmentFlag.AlignVertical_Mask)
                    | Qt.AlignmentFlag.AlignTop
                )

        alvo.setFixedHeight(inicial)
        alvo.show()
        if estava_oculto and pai is not None and (layout_do_pai := pai.layout()) is not None:
            # Resolve a largura do item antes oculto, sem processar eventos de pintura.
            layout_do_pai.activate()
        natural = self._altura_natural()
        self._arruma_conteudo(natural)
        destino = natural if aberto else 0
        if inicial == destino:
            self.conclui(aberto)
            return
        self._animacao.setStartValue(inicial)
        self._animacao.setEndValue(destino)
        self._animacao.start()

    def _altura_natural(self) -> int:
        alvo = self._alvo()
        layout = alvo.layout()
        if layout is not None:
            altura = layout.totalHeightForWidth(alvo.width())
            if altura < 0:
                altura = layout.totalSizeHint().height()
        else:
            altura = alvo.heightForWidth(alvo.width())
            if altura < 0:
                altura = alvo.sizeHint().height()
        return max(self._minimo, min(self._maximo, altura))

    def _arruma_conteudo(self, altura: int) -> None:
        alvo = self._alvo()
        layout = alvo.layout()
        if layout is not None:
            margens = alvo.contentsMargins()
            layout.setGeometry(
                QRect(
                    margens.left(),
                    margens.top(),
                    max(0, alvo.width() - margens.left() - margens.right()),
                    max(0, altura - margens.top() - margens.bottom()),
                )
            )

    def _aplica_altura(self, altura: int) -> None:
        self._alvo().setFixedHeight(altura)

    def conclui(self, aberto: bool | None = None) -> None:
        """Conclui a transição, ou aplica um estado novo sem movimento."""
        if self._restaurando:
            return
        self._restaurando = True
        try:
            if aberto is not None:
                self._aberto = aberto
            self._animacao.stop()
            alvo = self._alvo()
            alvo.setVisible(self._aberto)
            if self._preparada:
                alvo.setMinimumHeight(self._minimo)
                alvo.setMaximumHeight(self._maximo)
                layout = alvo.layout()
                if layout is not None:
                    layout.setEnabled(self._layout_habilitado)
                    layout.invalidate()
                    if self._layout_habilitado:
                        layout.activate()
                if isinstance(alvo, QLabel) and self._alinhamento is not None:
                    alvo.setAlignment(self._alinhamento)
                self._preparada = False
            alvo.updateGeometry()
        finally:
            self._restaurando = False

    def eventFilter(self, watched: QObject, evento: QEvent) -> bool:  # noqa: N802
        if self._restaurando or not self.em_andamento:
            return False
        if evento.type() == QEvent.Type.Hide:
            self.conclui()
        elif evento.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest):
            natural = self._altura_natural()
            self._arruma_conteudo(natural)
            if self._aberto and self._animacao.endValue() != natural:
                self._animacao.setEndValue(natural)
        return False

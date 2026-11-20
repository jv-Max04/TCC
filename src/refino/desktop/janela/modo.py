"""Seletor compacto dos modos sugeridos, sobre o combo editável original."""

from __future__ import annotations

from typing import override

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QStackedLayout,
    QWidget,
)

__all__ = ["ModoSegmentado"]

# O rótulo de cada segmento é o valor que ele envia: um texto diferente do valor faria a pessoa ler
# um modo e o arquivo de saída levar outro nome.
_MODOS = (
    ("pos_hilic", "pos_hilic"),
    ("neg_hilic", "neg_hilic"),
    ("pos_c18", "pos_c18"),
    ("neg_c18", "neg_c18"),
)
_DICA = (
    "Escolha o modo do estudo. Para digitar outro modo, pressione F2 "
    "ou dê dois cliques no modo selecionado."
)


class ModoSegmentado(QWidget):
    """Apresenta os quatro atalhos sem fechar o conjunto de modos aceitos.

    ``combo`` continua a guardar e emitir o valor usado pelo aplicativo, e cada segmento mostra
    exatamente o valor que envia.
    O mesmo combo fica acessível para rótulos personalizados, por F2 ou duplo clique.
    """

    def __init__(self, combo: QComboBox, pai: QWidget | None = None) -> None:
        super().__init__(pai)
        self.setObjectName("modoSegmentado")
        self.setMaximumHeight(36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(_DICA)
        self.combo = combo
        self._editando = False
        self._valores = dict(_MODOS)
        self.botoes: dict[str, QPushButton] = {}

        self._paginas = QStackedLayout(self)
        self._paginas.setContentsMargins(0, 0, 0, 0)
        self._paginas.setSpacing(0)
        self._segmentos = QFrame(self)
        self._segmentos.setObjectName("faixaModos")
        faixa = QHBoxLayout(self._segmentos)
        faixa.setContentsMargins(4, 4, 4, 4)
        faixa.setSpacing(4)
        self._grupo = QButtonGroup(self)
        self._grupo.setExclusive(True)
        for indice, (rotulo, valor) in enumerate(_MODOS):
            botao = QPushButton(rotulo, self._segmentos)
            botao.setObjectName("segmentoModo")
            botao.setProperty("valorModo", valor)
            botao.setProperty(
                "posicaoSegmento",
                "inicio" if indice == 0 else "fim" if indice == len(_MODOS) - 1 else "meio",
            )
            botao.setCheckable(True)
            botao.setFixedWidth((89, 89, 79, 79)[indice])
            botao.setCursor(Qt.CursorShape.PointingHandCursor)
            botao.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            botao.setToolTip(_DICA)
            botao.setAccessibleName(f"Modo {rotulo}")
            botao.installEventFilter(self)
            botao.clicked.connect(lambda _marcado=False, modo=valor: self._escolhe(modo))
            self._grupo.addButton(botao)
            self.botoes[rotulo] = botao
            faixa.addWidget(botao, 1)
        self._paginas.addWidget(self._segmentos)

        combo.setEditable(True)
        combo.setMaximumHeight(36)
        combo.setToolTip(_DICA)
        combo.setAccessibleName("Modo do estudo, inclusive personalizado")
        self._paginas.addWidget(combo)
        combo.currentTextChanged.connect(self._sincroniza)
        editor = combo.lineEdit()
        if editor is not None:
            editor.editingFinished.connect(self._conclui_edicao)

        self._atalho_editar = QShortcut(QKeySequence(Qt.Key.Key_F2), self)
        self._atalho_editar.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._atalho_editar.activated.connect(self.abre_editor)
        self._sincroniza(combo.currentText())

    def _escolhe(self, valor: str) -> None:
        self._editando = False
        self.combo.setCurrentText(valor)
        # Clicar no valor já corrente não emite currentTextChanged.
        self._sincroniza(self.combo.currentText())

    def _sincroniza(self, valor: str) -> None:
        self._grupo.setExclusive(False)
        for rotulo, botao in self.botoes.items():
            botao.setChecked(self._valores[rotulo] == valor)
        self._grupo.setExclusive(True)
        # Vazio é 'ainda não escolhido': os segmentos ficam à vista, nenhum marcado, para a escolha
        # ser um clique e não um texto digitado.
        conhecido = valor in self._valores.values() or not valor
        self._paginas.setCurrentWidget(
            self.combo if self._editando or not conhecido else self._segmentos
        )

    def abre_editor(self) -> None:
        """Expõe o combo original para editar qualquer rótulo de modo."""
        self._editando = True
        self._paginas.setCurrentWidget(self.combo)
        self.combo.setFocus(Qt.FocusReason.ShortcutFocusReason)
        editor = self.combo.lineEdit()
        if editor is not None:
            editor.selectAll()

    def _conclui_edicao(self) -> None:
        self._editando = False
        self._sincroniza(self.combo.currentText())

    @override
    def eventFilter(self, observado: QObject, evento: QEvent) -> bool:
        if (
            evento.type() == QEvent.Type.MouseButtonDblClick
            and isinstance(observado, QPushButton)
            and observado.isChecked()
        ):
            self.abre_editor()
            return True
        return super().eventFilter(observado, evento)

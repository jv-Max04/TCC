"""O painel que declara o recorte, e a lista de referências dos métodos.

A política vive em `refino.desktop.recorte`, e este arquivo só desenha: nenhuma frase de motivo,
nenhuma seção e nenhuma procedência nasce aqui. Um teste de AST confere que este módulo não declara
literal de motivo próprio.

O painel não sabe o que a janela tem. Ele recebe o conjunto dos painéis compostos e a tabela de
painéis declarados, e chama `itens_para_a_janela`, que é onde os três estados se derivam.

A lista de referências dos métodos fica num bloco que abre num clique, abaixo dos limites. O texto
vem pronto de `refino.interface.referencias`, como os motivos vêm de `refino.desktop.recorte`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.janela.componentes import Divisor, Recolhivel, Secao, Selo
from refino.desktop.janela.expansao import ExpansaoSuave
from refino.desktop.janela.tema import poe_icone
from refino.desktop.recorte import CATEGORIAS_DO_RECORTE, itens_para_a_janela
from refino.interface.referencias import texto_das_referencias

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Mapping

__all__ = [
    "SEM_ITENS",
    "TITULO_DAS_REFERENCIAS",
    "ItemDoRecorte",
    "PainelDeRecorte",
    "texto_do_item",
]

# O título do bloco que abre a lista de referências dos métodos.
TITULO_DAS_REFERENCIAS: Final = "Referências dos métodos"

# O texto antes de `define`. Ele não diz 'não há recorte': um painel ainda vazio e um corte sem
# limitação são fatos opostos, e sem este texto os dois ficariam iguais na tela.
SEM_ITENS: Final = "O recorte ainda não foi calculado."


def texto_do_item(cabeca: str, motivo: str) -> str:
    """A frase de um item como o rótulo a mostra.

    O painel e os testes leem a formatação daqui, para que ela exista num lugar só. O separador é
    `\\n\\n` porque o rótulo é Markdown, e o Markdown colapsa quebra única.
    """
    return f"{cabeca}\n\n{motivo}"


# O traço das setas de abrir e fechar um item, no estilo dos ícones da janela.
_SETA_FECHADA: Final = '<path d="M9 6l6 6-6 6"/>'
_SETA_ABERTA: Final = '<path d="M6 9l6 6 6-6"/>'


class ItemDoRecorte(QWidget):
    """Um item: o título numa linha clicável, e o motivo inteiro por baixo, fechado.

    O título é a cabeça sem o prefixo, que vira o grupo; o corpo é o motivo, em Markdown e
    selecionável.
    """

    def __init__(self, prefixo: str, titulo: str, motivo: str, pai: QWidget) -> None:
        super().__init__(pai)
        # A folga pertence ao fim da página, sem esticar itens durante o recolhimento.
        politica = self.sizePolicy()
        politica.setVerticalPolicy(QSizePolicy.Policy.Maximum)
        self.setSizePolicy(politica)
        self.prefixo = prefixo
        coluna = QVBoxLayout(self)
        coluna.setContentsMargins(0, 10, 0, 10)
        coluna.setSpacing(4)
        self.cabeca = QPushButton(titulo, self)
        self.cabeca.setObjectName("item_do_recorte")
        self.cabeca.setCheckable(True)
        self.cabeca.setCursor(Qt.CursorShape.PointingHandCursor)
        # A cor de destaque vem pelo token do tema, que a troca de esquema repinta.
        poe_icone(self.cabeca, _SETA_FECHADA, "destaque_texto", 14, tracos_ligado=_SETA_ABERTA)
        coluna.addWidget(self.cabeca)
        self.corpo = QLabel(motivo, self)
        self.corpo.setObjectName("corpo_do_recorte")
        # O texto de `desktop/recorte.py` traz negrito e crases, que em `PlainText` saem literais.
        self.corpo.setTextFormat(Qt.TextFormat.MarkdownText)
        self.corpo.setWordWrap(True)
        self.corpo.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.corpo.setVisible(False)
        coluna.addWidget(self.corpo)
        self._expansao = ExpansaoSuave(self.corpo)
        self.cabeca.toggled.connect(self._expansao.alterna)


class PainelDeRecorte(QWidget):
    """O painel 7: o que este corte não faz, e o que ele faz com ressalva.

    Cada categoria é uma `Secao` com a contagem num selo, e cada item é uma linha com o título; o
    motivo inteiro abre por baixo, num clique.

    A composição é reconstruída a cada `define`, porque o conjunto de painéis compostos pode mudar
    entre unidades e um painel que só acrescentasse mostraria o estado antigo junto do novo.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        disposicao = QVBoxLayout(self)
        disposicao.setContentsMargins(0, 0, 0, 0)

        # A lista é longa, então ela rola. Sem o `QScrollArea` os últimos itens ficariam fora da
        # janela, e o item que fica fora é indistinguível do item que não foi declarado.
        self._area = QScrollArea(self)
        self._area.setWidgetResizable(True)
        self._area.setFrameShape(QFrame.Shape.NoFrame)
        self._dentro = QWidget(self._area)
        self._pilha = QVBoxLayout(self._dentro)
        self._dentro.setProperty("folhaDeEtapa", True)
        self._dentro.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._pilha.setContentsMargins(20, 20, 20, 20)
        self._pilha.setSpacing(24)
        self._area.setWidget(self._dentro)
        disposicao.addWidget(self._area)

        self._aviso = QLabel(SEM_ITENS, self._dentro)
        self._aviso.setWordWrap(True)
        self._pilha.addWidget(self._aviso)
        # As referências dos métodos ficam abaixo dos limites, num bloco que abre num clique: são
        # consulta e não fluxo. `define` insere os grupos antes deste bloco, para que os limites
        # venham primeiro em qualquer recomposição.
        self.referencias = Recolhivel(TITULO_DAS_REFERENCIAS, self._dentro)
        dentro_das_referencias = QVBoxLayout(self.referencias.corpo)
        dentro_das_referencias.setContentsMargins(0, 0, 0, 0)
        self.lista_das_referencias = QLabel(self.referencias.corpo)
        self.lista_das_referencias.setTextFormat(Qt.TextFormat.MarkdownText)
        self.lista_das_referencias.setWordWrap(True)
        self.lista_das_referencias.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.lista_das_referencias.setText(texto_das_referencias())
        dentro_das_referencias.addWidget(self.lista_das_referencias)
        politica = self.referencias.sizePolicy()
        politica.setVerticalPolicy(QSizePolicy.Policy.Maximum)
        self.referencias.setSizePolicy(politica)
        self._pilha.addWidget(self.referencias)
        # O esticador é o último item da pilha, e `define` insere antes dele; anexar poria a folga
        # entre o aviso e a lista.
        self._pilha.addStretch(1)
        self._grupos: list[Secao] = []
        self._itens: list[ItemDoRecorte] = []

    def define(
        self,
        *,
        compostos: frozenset[str],
        operaveis: frozenset[str],
        declarados: Mapping[str, str],
        calculam_aqui: frozenset[str] = frozenset(),
    ) -> None:
        """Recalcula a lista a partir do que a janela de facto tem.

        Levanta `PainelDesconhecidoError` quando uma seção aponta para painel que a janela não
        declara. Deixar propagar é deliberado: absorver o erro produziria um painel de recorte
        incompleto sem aviso, que é o defeito que este painel não pode ter.

        Levanta `KeyError` em prefixo sem categoria, pela mesma razão: um item fora de todos os
        grupos não seria desenhado, e item que não aparece é indistinguível de item que não existe.
        """
        for grupo in self._grupos:
            self._pilha.removeWidget(grupo)
            grupo.setParent(None)
        self._grupos.clear()
        self._itens.clear()

        itens = itens_para_a_janela(
            compostos=compostos,
            operaveis=operaveis,
            declarados=declarados,
            calculam_aqui=calculam_aqui,
        )
        self._aviso.setVisible(not itens)
        por_prefixo: dict[str, list[tuple[str, str]]] = {
            categoria.prefixo: [] for categoria in CATEGORIAS_DO_RECORTE
        }
        for cabeca, motivo in itens:
            prefixo, _, titulo = cabeca.partition(" — ")
            por_prefixo[prefixo].append((titulo, motivo))

        for categoria in CATEGORIAS_DO_RECORTE:
            entradas = por_prefixo[categoria.prefixo]
            if not entradas:
                continue
            grupo = Secao(categoria.titulo, self._dentro, subtitulo=categoria.subtitulo)
            politica = grupo.sizePolicy()
            politica.setVerticalPolicy(QSizePolicy.Policy.Maximum)
            grupo.setSizePolicy(politica)
            grupo.setProperty("painel_plano", True)
            grupo.acoes.addWidget(Selo(str(len(entradas)), categoria.tipo_do_selo, grupo))
            grupo.corpo.setSpacing(4)
            for posicao, (titulo, motivo) in enumerate(entradas):
                if posicao:
                    grupo.corpo.addWidget(Divisor(grupo))
                item = ItemDoRecorte(categoria.prefixo, titulo, motivo, grupo)
                grupo.corpo.addWidget(item)
                self._itens.append(item)
            self._pilha.insertWidget(self._pilha.indexOf(self.referencias), grupo)
            self._grupos.append(grupo)

    def textos(self) -> tuple[str, ...]:
        """O que está desenhado no painel, para o teste: um texto por item.

        Lê os widgets e não uma cópia da lista, porque o que se quer afirmar é o que a tela mostra.
        O título e o motivo saem do botão e do rótulo; o prefixo sai do grupo em que o item foi
        posto. Item que exista e não esteja dentro de um grupo da pilha não conta.
        """
        return tuple(
            texto_do_item(f"{item.prefixo} — {item.cabeca.text()}", item.corpo.text())
            for item in self._itens
            if any(grupo.isAncestorOf(item) for grupo in self._grupos)
            and all(self._pilha.indexOf(grupo) >= 0 for grupo in self._grupos)
        )

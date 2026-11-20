"""As convenções de aparência e de interação da janela, ditas uma vez.

As funções daqui são regras da casa que valem para uma classe de widgets, e não só para as
instâncias de hoje: consertar as de hoje não impede a próxima de nascer errada. Por isso cada uma
tem teste por classe em `tests/test_desktop_aparencia.py`, que percorre os painéis e reprova
quando acha um widget sem a política. Os defeitos que as motivaram só aparecem com a janela
montada e em uso.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStyle,
    QWidget,
)

__all__ = [
    "CaixaDeAutomatico",
    "FiltroDaRoda",
    "icone_de_aviso",
    "linha_com_icone",
    "poe_no_campo",
    "sem_roda_sem_foco",
    "valor_do_campo",
    "veste_de_botao",
]

# O aviso em maiúsculas que o ícone substitui. Fica nomeado aqui, e não solto em `f`-strings,
# porque o teste confere que ele não volta ao texto da tela: ícone e palavra a dizerem a mesma
# coisa é o dobro do peso visual.
GRITO_ANTIGO = "ATENÇÃO — "


def icone_de_aviso(pai: QWidget) -> QLabel:
    """Um ícone de aviso com dica, para carregar a ressalva sem a gritar. Nasce escondido.

    Sem consumidor de produção: os ícones das ressalvas no ponto de escolha saíram a pedido do
    usuário. A função e a irmã `linha_com_icone` ficam para a ressalva de eixo
    (`mostra_ressalva_do_eixo`, no quadro do pré-processamento), que ainda escreve 'ATENÇÃO' em
    maiúsculas num rótulo sempre aberto; até lá, só os testes as exercitam.

    Um `QLabel` com pixmap, e não um `QPushButton`: um botão convida ao clique e não faria nada, e
    a dica é o gesto que revela a informação. O ícone vem do estilo, como em `veste_de_botao`, para
    não haver arquivo a viajar no executável.
    """
    icone = QLabel(pai)
    estilo = icone.style()
    if estilo is not None:  # pragma: no branch - um QWidget construído sempre tem estilo
        tamanho = estilo.pixelMetric(QStyle.PixelMetric.PM_SmallIconSize)
        icone.setPixmap(
            estilo.standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning).pixmap(tamanho, tamanho)
        )
    icone.setVisible(False)
    return icone


def veste_de_botao(botao: QPushButton) -> QPushButton:
    """Dá moldura e ícone de informação a um botão que revela prosa. Devolve o próprio botão.

    Um botão revelador com `setFlat(True)` perde a moldura e lê-se como texto, e
    estes têm de parecer botões. O `setCheckable(True)` fica, porque alternar mostrar e esconder é a
    semântica certa para um revelador; o que sai é a supressão da moldura. Marcado, o botão é
    pintado pela regra `QPushButton:checked` da folha do tema.

    O ícone vem do estilo (`SP_MessageBoxInformation`), e não de um arquivo: um `.png` teria de
    viajar no executável e entrar nas listas do congelamento, e um ícone é o gênero de arquivo que
    passa vazio sem nada acusar.
    """
    botao.setFlat(False)
    estilo = botao.style()
    if estilo is not None:  # pragma: no branch - um QWidget construído sempre tem estilo
        botao.setIcon(estilo.standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
    return botao


class FiltroDaRoda(QObject):
    """Come o evento da roda quando o widget não tem foco. Instalado por `sem_roda_sem_foco`.

    Por padrão do Qt, passar a roda sobre um combo ou um spin muda o valor. Num formulário longo que
    rola, quem desce a página muda em silêncio o campo sob o cursor, e o campo alterado vai para o
    payload. Por isso os campos só mudam com clique.

    A forma escolhida é a mais estreita: com o campo focado (depois de um clique ou de um `Tab`) a
    roda continua a funcionar, e quem rola a página não mexe em nada. O que a torna segura é o
    `StrongFocus` que `sem_roda_sem_foco` fixa junto; sem ele o widget ganharia foco pela própria
    roda, e o primeiro evento já mudaria o valor.
    """

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - API do Qt
        """`True` engole o evento. Só engole roda, e só em widget sem foco."""
        if event.type() is not QEvent.Type.Wheel:
            return False
        # `isinstance`, e não `cast`: `eventFilter` recebe `QObject`, e um objeto que não seja
        # `QWidget` não tem foco a consultar. Engolir é o lado seguro, porque roda que ninguém
        # pediu não muda valor nenhum.
        tem_foco = isinstance(watched, QWidget) and watched.hasFocus()
        return not tem_foco


def sem_roda_sem_foco[R: QWidget](widget: R, filtro: FiltroDaRoda) -> R:
    """Fixa `StrongFocus` e instala o filtro. Devolve o próprio widget.

    O filtro é passado, e não criado aqui, por tempo de vida: um `QObject` sem pai criado dentro
    desta função seria coletado no retorno, e o filtro deixaria de agir sem nada acusar. Cada
    painel guarda o seu como atributo.

    `StrongFocus` é metade do conserto. O padrão de um `QComboBox` é `WheelFocus`, que ganha foco
    pela roda: o primeiro giro focaria o widget e o segundo mudaria o valor.
    """
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    widget.installEventFilter(filtro)
    return widget


def valor_do_campo(widget: QWidget) -> Any:
    """O que um campo rolável carrega: o texto do combo, ou o número do spin.

    Uma função, e não ramos espalhados, porque vários sítios precisam da mesma resposta (a caixa
    'Automático', o botão 'Seguir sugestões do app', os testes da roda e do repouso), e cópias de
    `isinstance` divergiriam no dia em que um tipo novo de campo entrasse.
    """
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QAbstractSpinBox):
        return widget.value()  # type: ignore[attr-defined]
    raise TypeError(f"campo de tipo inesperado: {type(widget).__name__}")


def poe_no_campo(widget: QWidget, valor: Any) -> None:
    """O par de `valor_do_campo`. Levanta no tipo inesperado, em vez de ignorar em silêncio.

    Ignorar seria a caixa 'Automático' a marcar sem repor nada: um desfazer que não desfaz, e
    indistinguível de sucesso para quem clica.
    """
    if isinstance(widget, QComboBox):
        widget.setCurrentText(str(valor))
        return
    if isinstance(widget, QAbstractSpinBox):
        widget.setValue(valor)  # type: ignore[attr-defined]
        return
    raise TypeError(f"campo de tipo inesperado: {type(widget).__name__}")


class CaixaDeAutomatico(QCheckBox):
    """Trava um grupo de campos nos valores de repouso, e repõe-nos ao voltar a marcar.

    Serve aos limiares da conclusão na aba 'Teste de robustez (beta)'. Marcada, os campos ficam
    travados no valor com que nasceram; desmarcada, ficam livres, e marcar de novo devolve o
    estado inicial.

    Repõe o valor com que o campo nasceu, que é o padrão declarado, e não uma proposta da máquina.
    Os valores são lidos dos campos na construção, e não redigitados, para não haver segunda fonte
    do mesmo número e o desfazer devolver exatamente o que a tela mostrava ao abrir.
    """

    def __init__(self, texto: str, pai: QWidget, *, campos: tuple[QWidget, ...]) -> None:
        super().__init__(texto, pai)
        if not campos:
            raise ValueError(
                "uma caixa «Automático» sem campos é um controle que não faz nada — e um controle "
                "que não faz nada é indistinguível de um que está partido"
            )
        self._campos = campos
        self._repouso = tuple(valor_do_campo(c) for c in campos)
        self.setChecked(True)
        self.toggled.connect(self._aplica)
        self._aplica(self.isChecked())

    def _aplica(self, ticado: bool) -> None:
        """Marcada trava e repõe; desmarcada liberta.

        Repor antes de travar não muda nada aqui (um `QWidget` desabilitado aceita `setValue`); a
        ordem escrita é a que se lê melhor.
        """
        for campo, repouso in zip(self._campos, self._repouso, strict=True):
            if ticado:
                poe_no_campo(campo, repouso)
            campo.setEnabled(not ticado)

    def valores_de_repouso(self) -> tuple[Any, ...]:
        """O que ela repõe, na ordem dos campos. Acessor para os testes afirmarem o desfazer sem
        reescrever a travessia dos widgets em cada asserção."""
        return self._repouso


def linha_com_icone(icone: QLabel, texto: QLabel) -> QWidget:
    """O ícone à esquerda e a prosa à direita, numa linha que o `QFormLayout` aceita.

    `AlignTop` no ícone, e não no meio, porque a prosa quebra em várias linhas e um ícone centrado
    contra quatro linhas de texto parece pertencer à terceira. Vive aqui, e não num painel, porque
    é encanamento sem domínio. Está sem consumidor de produção, pela razão dita em `icone_de_aviso`.
    """
    linha = QWidget()
    disposicao = QHBoxLayout(linha)
    disposicao.setContentsMargins(0, 0, 0, 0)
    disposicao.addWidget(icone, 0, Qt.AlignmentFlag.AlignTop)
    disposicao.addWidget(texto, 1)
    return linha

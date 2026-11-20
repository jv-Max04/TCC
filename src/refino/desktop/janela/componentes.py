"""Os componentes da interface: o vocabulário visual que os painéis usam.

A interface não se explica em prosa: estados viram selos, ajudas viram ícones com dica, e
resultados longos ficam recolhidos. Os componentes:
- `Secao`: cabeçalho numerado, ações à direita, divisória e corpo;
- `linha_de_formulario`: rótulo à esquerda, campo à direita, ⓘ opcional;
- `Selo`: o estado numa palavra, com cor por significado;
- `Ajuda`: um ⓘ cuja dica carrega o texto inteiro;
- `CartaoDeLeitura`: selo e manchete, com o texto inteiro em 'Detalhes';
- `Recolhivel`: um bloco de referência fechado, que abre num clique;
- `FaixaDeAviso`: o aviso que precisa ficar, fixo na página, amarelo ou vermelho;
- `desliga_com_motivo`: o botão desligado com a frase do motivo ao lado.

Nenhum componente cria lógica. Os controles (botões, combos, rótulos que a lógica escreve) são
criados e ligados pelo painel dono deles, com o mesmo nome e o mesmo sinal; os componentes só os
recebem e decidem onde aparecem. Assim a API, os contratos e os testes dos painéis continuam a
falar com os controles, e não com a geometria. O texto que sai da tela não sai do programa: vai
para uma dica, para 'Detalhes' ou para um bloco fechado, no mesmo objeto e com o mesmo texto.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal

from PySide6.QtCore import QLocale, QSize, Qt, Signal
from PySide6.QtGui import QPaintEvent, QResizeEvent
from PySide6.QtWidgets import (
    QBoxLayout,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStylePainter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.janela.expansao import ExpansaoSuave
from refino.desktop.janela.tema import ICONES_DAS_FICHAS, poe_icone

__all__ = [
    "ALTURA_MINIMA_DO_SELO",
    "ICONE_DE_AJUDA",
    "NOMES_DOS_NIVEIS",
    "PREFIXOS_DOS_NIVEIS",
    "TIPOS_DE_SELO",
    "Ajuda",
    "CartaoDeLeitura",
    "ComboComNomes",
    "Divisor",
    "FaixaDeAviso",
    "GradeResponsiva",
    "NivelDeAviso",
    "Recolhivel",
    "RotuloQueSome",
    "Secao",
    "Selo",
    "com_maiuscula",
    "desliga_com_motivo",
    "em_grade",
    "linha_de_formulario",
    "religa",
    "repole",
]

# Os significados que um selo pode ter, lidos pela folha como `[tipo="..."]`. Domínio fechado e
# conferido em runtime em `Selo.define`: um tipo digitado errado cairia no estilo neutro sem nada
# acusar, e um 'Recusada' pintado de cinzento diz menos do que em vermelho.
TIPOS_DE_SELO: Final = frozenset({"neutro", "sucesso", "aviso", "erro", "info"})

# A altura mínima de um selo, em pixels: o dobro do raio da folha. Ver `Selo.__init__`.
ALTURA_MINIMA_DO_SELO: Final = 20

ICONE_DE_AJUDA: Final = '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>'

# Os dois níveis do aviso fixo, e o que cada um diz sem a cor. Pedido do usuário: avisos simples em
# amarelo, críticos em vermelho. Vermelho quando um pedido que a pessoa fez falhou ou foi recusado;
# amarelo quando nada falhou mas algo pede atenção, inclusive uma recusa sabida antes do clique,
# porque nada foi pedido ainda. A cor não basta (WCAG 1.4.1): o texto abre com o prefixo do nível,
# e o nome acessível diz o nível. Domínio fechado e conferido em runtime em `FaixaDeAviso.mostra`.
NivelDeAviso = Literal["aviso", "critico"]
PREFIXOS_DOS_NIVEIS: Final[dict[str, str]] = {"aviso": "Atenção:", "critico": "Não foi possível:"}
NOMES_DOS_NIVEIS: Final[dict[str, str]] = {"aviso": "Aviso", "critico": "Aviso crítico"}

# O nome do rótulo que diz por que um botão está desligado. Ver `desliga_com_motivo`.
_NOME_DO_MOTIVO: Final = "motivo_do_botao"


def repole(widget: QWidget) -> None:
    """Reavalia a folha sobre um widget cuja propriedade dinâmica mudou.

    O Qt só relê seletores por propriedade (`[tipo="erro"]`) quando o widget é repolido; sem isto
    um selo que muda de 'Pendente' para 'Recusada' muda o texto e fica com a cor antiga.
    """
    estilo = widget.style()
    if estilo is not None:  # pragma: no branch - um QWidget construído sempre tem estilo
        estilo.unpolish(widget)
        estilo.polish(widget)


class Divisor(QFrame):
    """Uma linha horizontal de 1 px, na cor da borda. O que separa cabeçalho de corpo."""

    def __init__(self, pai: QWidget | None = None) -> None:
        super().__init__(pai)
        self.setObjectName("divisor")
        self.setFixedHeight(1)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


class Selo(QLabel):
    """O estado numa palavra, com a cor do que ele significa."""

    def __init__(self, texto: str, tipo: str = "neutro", pai: QWidget | None = None) -> None:
        super().__init__(pai)
        self.setObjectName("selo_de_estado")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        # A altura mínima mantém a pílula redonda: o Qt desiste do raio quando ele passa da metade
        # da altura e desenha o retângulo reto, e os selos cheios, sem borda, ficariam abaixo dos 20
        # px que o raio de 10 pede.
        self.setMinimumHeight(ALTURA_MINIMA_DO_SELO)
        self.define(texto, tipo)

    def define(self, texto: str, tipo: str) -> None:
        """Troca texto e significado. Levanta em tipo fora de `TIPOS_DE_SELO`."""
        if tipo not in TIPOS_DE_SELO:
            raise ValueError(
                f"tipo de selo desconhecido: {tipo!r}; os válidos são {sorted(TIPOS_DE_SELO)}"
            )
        self.setText(texto)
        self.setProperty("tipo", tipo)
        repole(self)

    @property
    def tipo(self) -> str:
        """O significado em vigor; os testes leem isto em vez de adivinhar pela cor."""
        return str(self.property("tipo"))


class RotuloQueSome(QLabel):
    """Um rótulo que só ocupa espaço quando diz alguma coisa.

    Rótulos que a lógica escreve depois (a medição, os QCs por lote) nascem vazios e, sem isto,
    deixam vãos de uma linha no meio dos formulários. Este some enquanto o texto for um dos
    `vazios` e volta quando a lógica escreve. O objeto, o nome e o texto são os de sempre (quem o lê
    com `text()` lê o mesmo); muda só se ele ocupa a linha.

    `acompanha` registra widgets que somem com ele, como o rótulo da coluna esquerda de uma linha de
    `QFormLayout`, que sem isto ficaria sozinho a segurar a linha.
    """

    def __init__(
        self, texto: str = "", pai: QWidget | None = None, *, vazios: frozenset[str] = frozenset()
    ) -> None:
        super().__init__(pai)
        self._vazios = vazios | {""}
        self._companheiros: list[QWidget] = []
        self.setText(texto)

    def acompanha(self, widget: QWidget | None) -> None:
        """`widget` passa a aparecer e sumir junto com este rótulo."""
        if widget is not None:
            self._companheiros.append(widget)
            widget.setHidden(self.isHidden())

    def setText(self, texto: str) -> None:  # noqa: N802 - sobrescreve o método do Qt
        """Escreve, e some se o texto não diz nada."""
        super().setText(texto)
        vazio = texto.strip() in self._vazios
        self.setHidden(vazio)
        for companheiro in self._companheiros:
            companheiro.setHidden(vazio)


class FaixaDeAviso(QFrame):
    """O aviso que fica na página até deixar de valer: amarelo (`aviso`) ou vermelho (`critico`).

    Fixo, e não flutuante. O aviso flutuante (`AvisoDeDesfecho`) serve ao desfecho passageiro e
    some sozinho; este fica no lugar dele na página, porque diz o que falta para a pessoa chegar a
    um resultado, e isso não perde a validade com o tempo. Escondida, não ocupa altura, como
    `RotuloQueSome`.

    O botão de ação, quando há, é a saída que a faixa oferece ('Corrigir de novo'): ele emite
    `acionada`, e quem decide o que fazer é o dono da faixa. As cores vêm só da folha, pelos tokens
    `nota_aviso_*` e `nota_erro_*`, escolhidos pela propriedade `nivel`.
    """

    acionada = Signal()

    def __init__(self, pai: QWidget | None = None) -> None:
        super().__init__(pai)
        self.setObjectName("faixa_de_aviso")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setProperty("nivel", "aviso")
        linha = QHBoxLayout(self)
        linha.setContentsMargins(14, 10, 12, 10)
        linha.setSpacing(12)
        textos = QVBoxLayout()
        textos.setSpacing(2)
        self.titulo = QLabel(self)
        self.titulo.setObjectName("titulo_da_faixa_de_aviso")
        self.titulo.setWordWrap(True)
        self.titulo.hide()
        textos.addWidget(self.titulo)
        self.frase = QLabel(self)
        self.frase.setObjectName("frase_da_faixa_de_aviso")
        self.frase.setWordWrap(True)
        self.frase.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        textos.addWidget(self.frase)
        linha.addLayout(textos, 1)
        self.botao = QPushButton(self)
        self.botao.setObjectName("acao_da_faixa_de_aviso")
        self.botao.hide()
        self.botao.clicked.connect(self.acionada)
        linha.addWidget(self.botao, 0, Qt.AlignmentFlag.AlignVCenter)
        self.hide()

    def mostra(
        self,
        frase: str,
        *,
        nivel: NivelDeAviso,
        titulo: str | None = None,
        acao: str | None = None,
    ) -> None:
        """Mostra a faixa no nível dado; frase em branco a esconde.

        Levanta em nível fora de `PREFIXOS_DOS_NIVEIS`, antes de mudar qualquer coisa. O prefixo do
        nível abre a primeira linha: o título, se houver, senão a frase. Sem `acao`, o botão some,
        porque a ação de um aviso anterior não pode ficar no seguinte.
        """
        if nivel not in PREFIXOS_DOS_NIVEIS:
            raise ValueError(
                f"nível de aviso desconhecido: {nivel!r}; os válidos são "
                f"{sorted(PREFIXOS_DOS_NIVEIS)}"
            )
        if not frase.strip():
            self.limpa()
            return
        prefixo = PREFIXOS_DOS_NIVEIS[nivel]
        self.titulo.setText(f"{prefixo} {titulo}" if titulo else "")
        self.titulo.setHidden(not titulo)
        self.frase.setText(frase if titulo else f"{prefixo} {frase}")
        self.botao.setText(acao or "")
        self.botao.setHidden(not acao)
        self.setProperty("nivel", nivel)
        repole(self)
        nome = NOMES_DOS_NIVEIS[nivel]
        self.setAccessibleName(f"{nome}: {titulo}" if titulo else nome)
        self.setAccessibleDescription(self.texto)
        self.show()

    def limpa(self) -> None:
        """Esconde a faixa e apaga o texto: o aviso deixou de valer."""
        self.titulo.clear()
        self.frase.clear()
        self.botao.hide()
        self.setAccessibleDescription("")
        self.hide()

    @property
    def texto(self) -> str:
        """O que a faixa diz, na ordem de leitura: é o que os testes e o leitor de tela leem."""
        linhas = (self.titulo.text(), self.frase.text())
        return "\n".join(linha for linha in linhas if linha)


def _caixa_que_contem(widget: QWidget) -> tuple[QBoxLayout, int]:
    """A caixa de disposição em que `widget` está, e a posição dele nela.

    Levanta `TypeError` fora de uma caixa: sem pai, sem disposição, ou numa grade ou num
    formulário, onde não há 'a seguir' sem mexer nas linhas dos outros.
    """
    pai = widget.parentWidget()
    pendentes: list[QLayout] = []
    raiz = pai.layout() if pai is not None else None
    if raiz is not None:
        pendentes.append(raiz)
    while pendentes:
        disposicao = pendentes.pop()
        indice = disposicao.indexOf(widget)
        if indice >= 0:
            if isinstance(disposicao, QBoxLayout):
                return disposicao, indice
            break
        for posicao in range(disposicao.count()):
            item = disposicao.itemAt(posicao)
            interna = item.layout() if item is not None else None
            if interna is not None:
                pendentes.append(interna)
    raise TypeError(
        f"o botão {widget.objectName() or widget!r} não está numa caixa de disposição "
        "(QHBoxLayout ou QVBoxLayout); o motivo não teria lugar ao lado dele"
    )


def _rotulo_do_motivo(botao: QPushButton) -> RotuloQueSome | None:
    pai = botao.parentWidget()
    if pai is None:
        return None
    for rotulo in pai.findChildren(
        RotuloQueSome, _NOME_DO_MOTIVO, Qt.FindChildOption.FindDirectChildrenOnly
    ):
        if rotulo.buddy() is botao:
            return rotulo
    return None


def desliga_com_motivo(botao: QPushButton, frase: str) -> RotuloQueSome:
    """Desliga `botao` e escreve `frase` num rótulo à vista logo depois dele. Devolve o rótulo.

    Pedido do usuário: nenhum botão desligado sem explicação. A frase diz por que o botão está
    desligado e o que fazer para ligá-lo; frase em branco levanta, porque seria justamente o botão
    sem explicação. O rótulo nasce no primeiro pedido, na caixa do botão e logo depois dele, e é
    reaproveitado nos seguintes. Ele é o `buddy` do botão: é por essa ligação, que o Qt também dá ao
    leitor de tela, que quem percorre a janela acha a frase de cada botão desligado. `religa` liga
    o botão e esvazia a frase.
    """
    if not frase.strip():
        raise ValueError("um botão desligado precisa de um motivo, e a frase veio em branco")
    rotulo = _rotulo_do_motivo(botao)
    if rotulo is None:
        caixa, indice = _caixa_que_contem(botao)
        rotulo = RotuloQueSome(pai=botao.parentWidget())
        rotulo.setObjectName(_NOME_DO_MOTIVO)
        rotulo.setWordWrap(True)
        rotulo.setBuddy(botao)
        caixa.insertWidget(indice + 1, rotulo)
    botao.setEnabled(False)
    rotulo.setText(frase)
    return rotulo


def religa(botao: QPushButton) -> None:
    """Liga `botao` e esvazia a frase do motivo, se havia: o motivo velho não fica ao lado."""
    botao.setEnabled(True)
    rotulo = _rotulo_do_motivo(botao)
    if rotulo is not None:
        rotulo.setText("")


class _DelegadoDeNomes(QStyledItemDelegate):
    """Desenha cada item da lista aberta pelo nome legível, e não pelo valor."""

    def __init__(self, nomes: Mapping[str, str], pai: QWidget) -> None:
        super().__init__(pai)
        self._nomes = nomes

    def displayText(self, valor: Any, local: QLocale | QLocale.Language) -> str:  # noqa: N802
        """O nome legível do valor; o valor cru se a tabela não o tiver."""
        texto = super().displayText(valor, local)
        return self._nomes.get(texto, texto)


class ComboComNomes(QComboBox):
    """Um seletor que mostra o nome legível e guarda o valor técnico.

    O texto de cada item continua a ser o valor: `currentText()` devolve `serrf`, o pedido leva
    `serrf`, e toda leitura existente continua a valer. Muda só o desenho: o campo fechado e a lista
    aberta mostram `nomes[valor]`. Valor fora da tabela aparece como veio, para nunca esconder uma
    opção que o domínio ganhou depois da tabela.
    """

    def __init__(self, nomes: Mapping[str, str], pai: QWidget | None = None) -> None:
        super().__init__(pai)
        self.nomes = nomes
        self.setItemDelegate(_DelegadoDeNomes(nomes, self))

    def nome_atual(self) -> str:
        """O que está desenhado no campo fechado."""
        return self.nomes.get(self.currentText(), self.currentText())

    def paintEvent(self, evento: QPaintEvent) -> None:  # noqa: N802 - API do Qt
        """O campo fechado com o nome legível no lugar do valor."""
        if self.isEditable():
            super().paintEvent(evento)
            return
        pintor = QStylePainter(self)
        opcao = QStyleOptionComboBox()
        self.initStyleOption(opcao)
        opcao.currentText = self.nome_atual()
        pintor.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opcao)
        pintor.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opcao)

    def sizeHint(self) -> QSize:  # noqa: N802 - API do Qt
        """Largo o bastante para o nome mais longo, e não só para o valor mais longo."""
        base = super().sizeHint()
        metrica = self.fontMetrics()
        maior = max(
            (
                metrica.horizontalAdvance(self.nomes.get(self.itemText(i), self.itemText(i)))
                for i in range(self.count())
            ),
            default=0,
        )
        # 48 px para a seta e o respiro dos dois lados, que a folha dá ao campo.
        return QSize(max(base.width(), maior + 48), base.height())


class Ajuda(QLabel):
    """Um ⓘ que carrega a explicação inteira na dica. O texto não some: muda de lugar."""

    def __init__(self, texto: str, pai: QWidget | None = None) -> None:
        super().__init__(pai)
        self.setObjectName("ajuda")
        # A cor de destaque pelo token do tema, que a troca de esquema repinta.
        poe_icone(self, ICONE_DE_AJUDA, "destaque_texto", 16)
        self.setToolTip(texto)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


class Secao(QFrame):
    """Um bloco da página: cabeçalho (número, título, subtítulo, ações), divisória, corpo.

    `acoes` é a ponta direita do cabeçalho, onde vai o selo de estado da seção ou um botão.
    `corpo` é onde o dono põe o conteúdo.
    `painel_plano` usa o separador inferior da referência desktop, sem uma caixa interna.
    """

    def __init__(
        self,
        titulo: str,
        pai: QWidget | None = None,
        *,
        numero: int | None = None,
        subtitulo: str | None = None,
        referencia: bool = False,
        painel_plano: bool = False,
    ) -> None:
        super().__init__(pai)
        self.setObjectName("secao")
        self.setProperty("referencia", referencia)
        self.setProperty("painelPlano", painel_plano)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        externo = QVBoxLayout(self)
        externo.setContentsMargins(0, 0, 0, 0)
        externo.setSpacing(0)

        cabeca = QHBoxLayout()
        margens_do_cabecalho = (
            (0, 20, 0, 0) if painel_plano else (24, 24, 24, 24) if referencia else (20, 18, 20, 18)
        )
        cabeca.setContentsMargins(*margens_do_cabecalho)
        cabeca.setSpacing(16 if referencia or painel_plano else 12)
        if numero is not None:
            self.numero = QLabel(str(numero), self)
            self.numero.setObjectName("numero_da_secao")
            self.numero.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lado = 28 if referencia or painel_plano else 26
            self.numero.setFixedSize(lado, lado)
            cabeca.addWidget(self.numero, 0, Qt.AlignmentFlag.AlignVCenter)
        textos = QVBoxLayout()
        textos.setSpacing(0 if referencia or painel_plano else 4)
        self.titulo = QLabel(titulo, self)
        self.titulo.setObjectName("titulo_da_secao")
        self.titulo.setWordWrap(True)
        if referencia or painel_plano:
            self.titulo.setMinimumHeight(24)
        textos.addWidget(self.titulo)
        if subtitulo is not None:
            self.subtitulo = QLabel(subtitulo, self)
            self.subtitulo.setObjectName("subtitulo_da_secao")
            self.subtitulo.setWordWrap(True)
            if referencia or painel_plano:
                self.subtitulo.setMinimumHeight(16)
            textos.addWidget(self.subtitulo)
        cabeca.addLayout(textos, 1)
        # O bloco mantém a altura do texto, mesmo quando uma ação pede uma linha alta.
        cabeca.setAlignment(textos, Qt.AlignmentFlag.AlignVCenter)
        self.acoes = QHBoxLayout()
        self.acoes.setSpacing(8)
        cabeca.addLayout(self.acoes)
        cabeca.setAlignment(self.acoes, Qt.AlignmentFlag.AlignVCenter)
        externo.addLayout(cabeca)
        if not referencia and not painel_plano:
            externo.addWidget(Divisor(self))

        self.corpo = QVBoxLayout()
        margens_do_corpo = (
            (0, 16, 0, 20) if painel_plano else (24, 0, 24, 24) if referencia else (20, 18, 20, 20)
        )
        self.corpo.setContentsMargins(*margens_do_corpo)
        self.corpo.setSpacing(12)
        externo.addLayout(self.corpo, 1)
        if painel_plano:
            externo.addWidget(Divisor(self))


def linha_de_formulario(
    grade: QGridLayout,
    linha: int,
    rotulo: str,
    campo: QWidget | QHBoxLayout,
    *,
    pai: QWidget,
    ajuda: str | None = None,
    nota: QLabel | None = None,
) -> QLabel:
    """Uma linha de formulário numa grade de três colunas: rótulo, campo, ⓘ.

    `nota`, quando vem, fica por baixo do campo, em tipo pequeno: é o lugar do texto curto que
    precisa de estar à vista (o nome que a coisa tem no MetaboAnalyst, por exemplo), e não numa
    dica. Devolve o rótulo, para o dono o guardar se precisar.
    """
    nome = QLabel(rotulo, pai)
    nome.setObjectName("rotulo_do_campo")
    nome.setWordWrap(True)
    grade.addWidget(nome, linha * 2, 0, Qt.AlignmentFlag.AlignVCenter)
    if isinstance(campo, QHBoxLayout):
        grade.addLayout(campo, linha * 2, 1)
    else:
        grade.addWidget(campo, linha * 2, 1)
    if ajuda is not None:
        grade.addWidget(Ajuda(ajuda, pai), linha * 2, 2, Qt.AlignmentFlag.AlignVCenter)
    if nota is not None:
        nota.setObjectName("nota_do_campo")
        nota.setWordWrap(True)
        grade.addWidget(nota, linha * 2 + 1, 1)
    return nome


def com_maiuscula(texto: str) -> str:
    """O texto com a primeira letra em maiúscula, mesmo que ele comece por marcação.

    A prosa da API e de `interface/fluxo.py` começa em minúscula porque é escrita para entrar no
    meio de
    outra frase ('Valor: não calculada'). Quando a janela a põe no início de uma frase, a primeira
    letra sobe. Salta a marcação (asteriscos, crases, aspas), porque pôr em maiúscula um asterisco
    não muda nada e a letra ficaria em minúscula. Só a primeira letra muda.
    """
    for posicao, caractere in enumerate(texto):
        if caractere.isalpha():
            return texto[:posicao] + caractere.upper() + texto[posicao + 1 :]
    return texto


def em_grade(grade: QGridLayout, widgets: list[QWidget], colunas: int) -> None:
    """Põe os widgets na grade, linha a linha, `colunas` por linha.

    A conta da posição vive aqui, e não em quem chama, porque o painel de qualidade tem um teste que
    proíbe qualquer aritmética no módulo dele (`test_u7_o_painel_nao_faz_aritmetica_nenhuma`), para
    nenhum número de métrica nascer na janela. Índice de grade não é número de métrica, e o lugar
    dele é o módulo de layout.
    """
    for posicao, widget in enumerate(widgets):
        grade.addWidget(widget, posicao // colunas, posicao % colunas)


class GradeResponsiva(QWidget):
    """Cartões em até `colunas` colunas, e em menos quando não cabem, como numa página web.

    Com colunas fixas, um título longo de cartão (que não quebra linha por ser título de moldura)
    faz a página pedir mais largura do que tem a 1280x760, e aparece rolagem horizontal. Esta grade
    mede a largura que recebe e escolhe o número de colunas que cabe.

    O mínimo que ela declara é o de uma coluna, e é isso que a deixa encolher: com o mínimo de duas,
    o pai nunca lhe daria largura menor, e ela nunca passaria a uma. O layout de dentro não impõe o
    seu mínimo ao widget (`SetNoConstraint`), pela mesma razão.
    """

    def __init__(
        self,
        widgets: list[QWidget],
        colunas: int,
        pai: QWidget | None = None,
        espaco: int = 12,
        *,
        alinhar_ao_topo: bool = False,
    ) -> None:
        super().__init__(pai)
        self._widgets = list(widgets)
        self._maximo = max(1, colunas)
        self._alinhar_ao_topo = alinhar_ao_topo
        self._grade = QGridLayout(self)
        self._grade.setContentsMargins(0, 0, 0, 0)
        self._grade.setSpacing(espaco)
        self._grade.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.colunas = 0
        for widget in self._widgets:
            widget.setParent(self)
        self._arruma(self._maximo)

    def _arruma(self, colunas: int) -> None:
        """Redistribui os cartões em `colunas` colunas, se o número mudou."""
        if colunas == self.colunas:
            return
        for widget in self._widgets:
            self._grade.removeWidget(widget)
        for coluna in range(self._maximo):
            self._grade.setColumnStretch(coluna, 1 if coluna < colunas else 0)
        em_grade(self._grade, self._widgets, colunas)
        if self._alinhar_ao_topo:
            for widget in self._widgets:
                self._grade.setAlignment(widget, Qt.AlignmentFlag.AlignTop)
        self.colunas = colunas
        self.updateGeometry()

    def colunas_que_cabem(self, largura: int) -> int:
        """Quantas colunas cabem em `largura` px, do máximo para baixo; ao menos uma."""
        maior = max(
            (max(w.minimumWidth(), w.minimumSizeHint().width()) for w in self._widgets), default=0
        )
        espaco = self._grade.horizontalSpacing()
        for colunas in range(self._maximo, 1, -1):
            if colunas * maior + (colunas - 1) * espaco <= largura:
                return colunas
        return 1

    def resizeEvent(self, evento: QResizeEvent) -> None:  # noqa: N802 - API do Qt
        """A cada nova largura, o número de colunas que cabe nela."""
        super().resizeEvent(evento)
        self._arruma(self.colunas_que_cabem(evento.size().width()))
        if self._alinhar_ao_topo and evento.size().width() != evento.oldSize().width():
            self.updateGeometry()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - API do Qt
        """A largura de uma coluna, e a altura do arranjo em vigor."""
        maior = max(
            (max(w.minimumWidth(), w.minimumSizeHint().width()) for w in self._widgets), default=0
        )
        return QSize(maior, self._grade.minimumSize().height())

    def sizeHint(self) -> QSize:  # noqa: N802 - API do Qt
        """O que o arranjo em vigor pede."""
        preferida = self._grade.sizeHint()
        if self._alinhar_ao_topo:
            altura = self._grade.totalHeightForWidth(self.width())
            if altura >= 0:
                return QSize(preferida.width(), altura)
        return preferida


class CartaoDeLeitura(QFrame):
    """Um resultado: ícone e título, selo de estado, manchete, o botão, e o texto inteiro recolhido.

    `rodape` recebe o botão do dono; `detalhes` recebe o rótulo que a lógica escreve, o mesmo objeto
    com o mesmo texto, agora atrás de 'Detalhes' em vez de corrido na tela.
    """

    def __init__(
        self,
        chave_do_icone: str,
        titulo: str,
        frase: str,
        pai: QWidget,
        *,
        detalhes_no_cartao: bool = True,
    ) -> None:
        super().__init__(pai)
        self.setObjectName("cartao")
        self.setProperty("referencia", True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(210)
        coluna = QVBoxLayout(self)
        coluna.setContentsMargins(20, 20, 20, 20)
        coluna.setSpacing(0)

        cabeca = QHBoxLayout()
        cabeca.setSpacing(10)
        figura = QLabel(self)
        figura.setObjectName("icone_do_cartao")
        figura.setAlignment(Qt.AlignmentFlag.AlignCenter)
        figura.setFixedSize(36, 36)
        poe_icone(figura, ICONES_DAS_FICHAS[chave_do_icone], "destaque_texto", 16)
        nome = QLabel(titulo, self)
        nome.setObjectName("titulo_do_cartao")
        nome.setWordWrap(True)
        nome.setMinimumHeight(24)
        cabeca.addWidget(figura)
        cabeca.addStretch(1)
        self.ponto = QLabel(self)
        self.ponto.setObjectName("ponto_do_estado")
        self.ponto.setProperty("tipo", "neutro")
        self.ponto.setFixedSize(6, 6)
        cabeca.addWidget(self.ponto)
        self.selo = Selo("Não lido", "neutro", self)
        cabeca.addWidget(self.selo)
        coluna.addLayout(cabeca)
        coluna.addSpacing(16)
        coluna.addWidget(nome)
        coluna.addSpacing(4)

        self.frase = QLabel(frase, self)
        self.frase.setObjectName("nota_do_campo")
        self.frase.setWordWrap(True)
        self.frase.setMinimumHeight(40)
        self.frase.setAlignment(Qt.AlignmentFlag.AlignTop)
        coluna.addWidget(self.frase)

        self.manchete = QLabel("—", self)
        self.manchete.setObjectName("manchete")
        self.manchete.setWordWrap(True)
        self.manchete.setProperty("vazia", True)
        self.manchete.hide()
        coluna.addWidget(self.manchete)
        coluna.addStretch(1)
        coluna.addSpacing(20)

        self.rodape = QHBoxLayout()
        self.rodape.setSpacing(8)
        self.alterna = QToolButton(self)
        self.alterna.setObjectName("alterna")
        self.alterna.setText("Detalhes")
        self.alterna.setCheckable(True)
        self.alterna.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.alterna.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        poe_icone(self.alterna, '<path d="m9 5 7 7-7 7"/>', "texto_apagado", 14)
        self.alterna.setIconSize(QSize(14, 14))
        self.alterna.setCursor(Qt.CursorShape.PointingHandCursor)
        self.alterna.setMinimumHeight(28)
        self.alterna.toggled.connect(self._alterna)
        self.rodape.addStretch(1)
        self.rodape.addWidget(self.alterna)
        coluna.addLayout(self.rodape)

        self.detalhes = QFrame(self if detalhes_no_cartao else pai)
        self.detalhes.setObjectName("detalhes")
        self.detalhes.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._pilha_dos_detalhes = QVBoxLayout(self.detalhes)
        self._pilha_dos_detalhes.setContentsMargins(12, 12, 12, 12)
        self.detalhes.setVisible(False)
        self.expansao = ExpansaoSuave(self.detalhes)
        if detalhes_no_cartao:
            coluna.addWidget(self.detalhes)

    def poe_botao(self, botao: QPushButton) -> None:
        """O botão do dono vai para o início do rodapé, antes de 'Detalhes'."""
        botao.setObjectName("executa_leitura")
        poe_icone(botao, '<path d="m8 4 11 8-11 8z"/>', "sobre_invertido", 14)
        botao.setIconSize(QSize(14, 14))
        self.rodape.insertWidget(0, botao)

    def poe_detalhe(self, widget: QWidget) -> None:
        """O texto inteiro do dono vai para dentro de 'Detalhes'."""
        self._pilha_dos_detalhes.addWidget(widget)

    def mostra_resumo(self, tipo: str, estado: str, manchete: str) -> None:
        """Atualiza selo e manchete. Quem os calcula é o dono, a partir do corpo da resposta."""
        self.selo.define(estado, tipo)
        self.ponto.setProperty("tipo", tipo)
        repole(self.ponto)
        self.manchete.setText(manchete)
        self.manchete.setProperty("vazia", False)
        self.manchete.show()
        repole(self.manchete)

    def limpa(self) -> None:
        """Volta ao estado anterior à leitura quando a entrada deixa de estar anexada."""
        self.selo.define("Não lido", "neutro")
        self.ponto.setProperty("tipo", "neutro")
        repole(self.ponto)
        self.manchete.setText("—")
        self.manchete.setProperty("vazia", True)
        self.manchete.hide()
        self.alterna.setChecked(False)
        self.expansao.conclui(False)
        repole(self.manchete)

    def _alterna(self, aberto: bool) -> None:
        self.expansao.alterna(aberto)
        traco = '<path d="m5 9 7 7 7-7"/>' if aberto else '<path d="m9 5 7 7-7 7"/>'
        poe_icone(self.alterna, traco, "texto_apagado", 14)


class Recolhivel(QFrame):
    """Um bloco de referência que nasce fechado e abre num clique, como a legenda.

    Fechado por padrão porque é referência, e não fluxo: quem precisa de saber o que é 'errático'
    abre, e quem já sabe não paga a altura dele em toda visita à página.
    """

    def __init__(self, titulo: str, pai: QWidget | None = None) -> None:
        super().__init__(pai)
        self.setObjectName("recolhivel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        coluna = QVBoxLayout(self)
        coluna.setContentsMargins(18, 10, 18, 10)
        coluna.setSpacing(6)
        self.alterna = QToolButton(self)
        self.alterna.setObjectName("alterna")
        self.alterna.setText(titulo)
        self.alterna.setCheckable(True)
        poe_icone(self.alterna, '<path d="m9 5 7 7-7 7"/>', "texto_apagado", 16)
        self.alterna.setIconSize(QSize(16, 16))
        self.alterna.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.alterna.setCursor(Qt.CursorShape.PointingHandCursor)
        self.alterna.setMinimumHeight(30)
        self.alterna.toggled.connect(self._alterna)
        coluna.addWidget(self.alterna)
        self.corpo = QWidget(self)
        self.corpo.setVisible(False)
        self.expansao = ExpansaoSuave(self.corpo)
        coluna.addWidget(self.corpo)

    def _alterna(self, aberto: bool) -> None:
        self.expansao.alterna(aberto)
        traco = '<path d="m5 9 7 7 7-7"/>' if aberto else '<path d="m9 5 7 7-7 7"/>'
        poe_icone(self.alterna, traco, "texto_apagado", 16)

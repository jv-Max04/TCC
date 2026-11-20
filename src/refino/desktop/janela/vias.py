"""O painel das vias metabólicas: predição a partir dos picos, sem identificação.

O painel não calcula estatística. Ele consome o mesmo corpo de `/execucoes` que o volcano usa
(nomes de feature, p-valores e a estatística do teste) e chama `refino.pathways`, em processo
local; recalcular a univariada aqui daria um segundo número para a mesma pergunta. Ele mostra as
vias, ordenadas pelo p ajustado por permutação, e o recorte: quantas features entraram, a causa
de cada saída e o corte de significância sob o qual a tabela vale, que a biblioteca escolhe
quando o sinal é fraco.

A aba está em beta e tem um segundo modo, escolhido no topo, que enriquece a lista identificada
no Progenesis com o MetaboAnalystR num R portátil (`janela.enriquecimento`). Os dois modos
dividem a aba e não o estado: cada um tem a sua piscina, a sua geração e o seu `encerra`, e o
volcano chega aos dois por `recebe_volcano`.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Final

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.janela.aparencia import FiltroDaRoda, sem_roda_sem_foco
from refino.desktop.janela.componentes import Secao, desliga_com_motivo, religa
from refino.desktop.janela.enriquecimento import PainelDeEnriquecimento
from refino.desktop.recorte import FUNCIONALIDADE_EM_BETA
from refino.pathways.analise import EntradaRecusadaError, prediz_vias

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Callable

    from refino.pathways.analise import RelatorioDeVias

__all__ = [
    "A_CALCULAR",
    "COLUNAS",
    "MOTIVO_A_CALCULAR",
    "MOTIVO_SEM_EXECUCAO",
    "MOTIVO_SEM_PAR_DE_GRUPOS",
    "SEM_EXECUCAO",
    "SEM_PAR_DE_GRUPOS",
    "PainelDeVias",
    "numero",
    "rotulos_da_busca",
    "texto_do_recorte",
]


# O texto enquanto o worker corre. O botão desligado não basta: sem uma frase, o recorte da
# corrida anterior ficaria na tela durante o cálculo.
A_CALCULAR: Final = "A predizer vias... isto leva alguns segundos."


class _SinaisDeVias(QObject):
    """O portador dos sinais, porque `QRunnable` não é `QObject` e não pode declará-los.

    É a mesma forma que `principal.py` usa para as chamadas de rota. Não se importa a de lá porque
    `principal.py` importa este módulo, e o caminho inverso seria circular.
    """

    # O segundo argumento é a geração do trabalho, que viaja com ele: ver `_recebe`.
    concluido = Signal(object, int)
    falhou = Signal(object, int)


def despacha_vias(
    chamada: Callable[[], RelatorioDeVias], sinais: _SinaisDeVias, geracao: int
) -> None:
    """A lógica de `_TrabalhoDeVias.run`, numa função chamável da thread principal.

    O `coverage` não enxerga as threads nativas do `QThreadPool`, e código dentro de `run` aparece
    com 0% mesmo quando corre. Com a lógica aqui, os dois desfechos são medidos na thread principal
    e o caminho real continua exercitável com `qtbot.waitSignal`, como em `principal.despacha`.

    Captura `BaseException` e não `Exception` porque uma exceção que escapa de um `QRunnable` some:
    o `QThreadPool` não a propaga.
    """
    try:
        relatorio = chamada()
    except BaseException as erro:
        sinais.falhou.emit(erro, geracao)
        return
    sinais.concluido.emit(relatorio, geracao)


class _TrabalhoDeVias(QRunnable):
    """A predição fora do laço de eventos.

    Recebe a chamada já fechada em vez dos argumentos dela, para que o teste injete uma chamada
    lenta ou uma que levanta sem esta classe saber o que é uma predição de via.
    """

    def __init__(
        self, chamada: Callable[[], RelatorioDeVias], sinais: _SinaisDeVias, geracao: int
    ) -> None:
        super().__init__()
        self._chamada = chamada
        self._sinais = sinais
        self._geracao = geracao

    def run(self) -> None:
        # Uma linha, sem `pragma`: chamar `run()` na thread principal instrumenta o despacho.
        despacha_vias(self._chamada, self._sinais, self._geracao)


def numero(valor: float, casas: str = ".4g") -> str:
    """Um número com a vírgula decimal da prosa portuguesa que o resto do painel usa.

    É a mesma escolha que `plots/base.py` faz para os eixos, e como lá ela não alcança nome de
    feature: `21,37_508,2876m/z` não é o nome de nada. Por isso a função recebe um número e não um
    texto.
    """
    return format(valor, casas).replace(".", ",")


COLUNAS: Final[tuple[tuple[str, str], ...]] = (
    ("nome", "Via metabólica"),
    ("sobreposicao", "Compostos significativos"),
    ("detectados", "Detectados na via"),
    ("pvalue", "p (permutação)"),
    ("pvalue_fisher", "p (Fisher)"),
)

# O texto antes de haver execução concluída. Ele não diz 'nenhuma via', que afirmaria um resultado
# que ninguém mediu. Diz 'concluída' porque o painel também recebe o progresso de uma execução que
# ainda corre.
SEM_EXECUCAO: Final = (
    "Ainda não há execução concluída. Corrija uma matriz com um par de grupos declarado e "
    "volte aqui: a predição de vias usa os mesmos p-valores que o volcano usa."
)

# A execução existe e não traz volcano: a rota só corre o teste univariado com `group_a` e
# `group_b`, e devolve `volcano = None` sem eles. Quem acabou de ver a execução concluir não deve
# ler que ela não existe.
SEM_PAR_DE_GRUPOS: Final = (
    "A execução concluída não declarou um par de grupos, e sem ele não há p-valor por "
    "feature para a predição usar. Preencha «Grupo A do contraste» e «Grupo B do "
    "contraste» e execute de novo."
)

# As frases ao lado de 'Predizer vias' desligado, uma por motivo, para que o botão nunca fique
# desligado sem explicação. A ação de quem não rodou nada e a de quem rodou sem o par diferem. São
# curtas porque ficam na linha do botão; a explicação longa fica no recorte.
MOTIVO_SEM_EXECUCAO: Final = (
    "Falta uma execução concluída: rode a «Análise estatística» com «Grupo A do contraste» e "
    "«Grupo B do contraste» preenchidos."
)
MOTIVO_SEM_PAR_DE_GRUPOS: Final = (
    "A execução não declarou um par de grupos: preencha «Grupo A do contraste» e «Grupo B do "
    "contraste» e rode a «Análise estatística» de novo."
)
MOTIVO_A_CALCULAR: Final = "A predição está em curso; o botão volta quando ela terminar."

# A polaridade sai do nome do modo, e o combo deixa mudá-la. A derivação vale para os nomes de
# modo deste laboratório e não é invariante: um nome fora do padrão fica no positivo, e uma
# polaridade errada desloca toda massa neutra em dois prótons sem nada acusar.
_POLARIDADE_PADRAO: Final = "positive"

_MODOS = (
    ("positive", "Positivo — [M+H]+ e adutos"),
    ("negative", "Negativo — [M-H]- e adutos"),
)

_POLITICAS = (
    ("primary_ion", "Incluir, assumindo o íon primário"),
    ("exclude", "Deixar de fora"),
)

# As duas buscas de adutos. A ampla vem primeiro porque é o padrão, e num `QComboBox` o primeiro
# item é o selecionado: ela concorda melhor com o MetaboAnalystR (Spearman 0,91 contra 0,36 no
# `pos_hilic`). O rótulo não traz o número de formas porque ele não vale sempre: no modo negativo o
# mummichog publica uma lista só, e as duas opções procuram as mesmas formas. Quem quer o número
# lê-o no recorte, que reporta `adutos_procurados` da lista que valeu.
#
# O rótulo da ampla depende da polaridade porque a concordância só foi medida no modo positivo. O
# valor de cada item é o mesmo nas duas polaridades.
#
# Duas constantes e não um dicionário por polaridade porque o teste de contrato da janela lê todo
# `x["chave"]` dos módulos da janela como leitura de corpo de resposta.
_BUSCAS_POSITIVO: Final[tuple[tuple[str, str], ...]] = (
    ("broad", "Ampla — concorda com o MetaboAnalyst"),
    ("default", "Estreita — o padrão do algoritmo original"),
)
_BUSCAS_NEGATIVO: Final[tuple[tuple[str, str], ...]] = (
    ("broad", "Ampla — no negativo, a mesma lista da estreita; concordância não medida"),
    ("default", "Estreita — o padrão do algoritmo original"),
)


def rotulos_da_busca(polaridade: object) -> tuple[tuple[str, str], ...]:
    """Os itens do combo da busca para esta polaridade.

    Polaridade desconhecida cai nos rótulos do modo negativo, que são os que não afirmam
    concordância medida.
    """
    return _BUSCAS_POSITIVO if polaridade == "positive" else _BUSCAS_NEGATIVO


def texto_do_recorte(relatorio: RelatorioDeVias) -> str:
    """A frase do recorte como o painel a mostra; o painel e o teste leem a formatação daqui."""
    r = relatorio.recorte
    saidas = [
        (r.massas_neutras_excluidas, "massas neutras deixadas de fora por escolha sua"),
        (r.nao_interpretadas, "nomes que não têm a forma `tempo_massa`"),
        (r.sem_estatistica, "sem p-valor utilizável"),
        (r.fora_da_faixa_de_massa, "fora da faixa de 50 a 2.000 Da"),
    ]
    detalhe = "; ".join(f"{quantas} {porque}" for quantas, porque in saidas if quantas)
    linhas = [
        f"**{r.analisadas} de {r.recebidas} features entraram na análise.**",
        f"Fora: {detalhe}." if detalhe else "Nenhuma feature ficou de fora.",
        f"Destas, **{r.features_significativas}** features entraram na lista "
        f"significativa, sob o corte "
        f"**p < {numero(r.corte_de_significancia)}**.",
        f"Modelo metabólico: {relatorio.versao_do_modelo}; "
        f"{relatorio.adutos_procurados} formas iônicas procuradas por composto; "
        f"{relatorio.permutacoes} permutações; semente {relatorio.seed}.",
    ]
    return "\n\n".join(linhas)


# Uma frase por seção, do lado de quem usa.
SUBTITULO_DOS_PICOS: Final = "A polaridade, as massas deconvoluídas e as formas iônicas a procurar."
SUBTITULO_DAS_VIAS: Final = "As vias que a predição aponta, com o recorte de onde elas saem."

# A altura mínima da tabela das vias, em pixels: umas seis linhas.
ALTURA_MINIMA_DA_TABELA: Final = 220


class PainelDeVias(QWidget):
    """As vias preditas, os parâmetros que as produziram, e o que ficou de fora.

    A tabela é reconstruída a cada cálculo e nunca acrescentada: um painel que acrescentasse
    mostraria o resultado de uma polaridade junto do de outra, e as duas não se comparam linha a
    linha.
    """

    # Cada predição que termina, com o corpo da execução de que ela é e se foi pintada. A que
    # termina com outra matriz na tela não pinta, mas não se perde: a janela guarda-a com a matriz
    # dona do corpo e repõe-na quando a pessoa volta a ela.
    predicao_concluida = Signal(object, object, bool)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        calculadora: Callable[..., RelatorioDeVias] = prediz_vias,
        enriquecimento: dict[str, Any] | None = None,
    ) -> None:
        """`calculadora` é injetável para o teste rodar sem o custo do cálculo real.

        O padrão é a função de produção, e não um duplo: um padrão falso passaria nos testes e não
        calcularia nada na máquina do usuário.
        """
        super().__init__(parent)
        self._calcula = calculadora
        self._volcano: Any = None
        # Por que não há volcano: nenhuma execução concluída, ou uma sem o par de grupos. É a
        # frase ao lado do botão desligado, escolhida em `recebe_volcano`.
        self._motivo_sem_volcano: str = MOTIVO_SEM_EXECUCAO
        self._relatorio: RelatorioDeVias | None = None
        # O token de geração: uma execução nova que chega com uma predição em voo avança
        # `_geracao`, e o resultado que volta com a geração antiga não pinta. A geração viaja com o
        # trabalho em vez de ficar num atributo do painel: com um segundo clique antes de a predição
        # velha acabar, o atributo voltaria a igualar `_geracao` e a velha pintaria sob o recorte
        # novo.
        self._geracao = 0
        # O corpo da execução na tela, e o de cada predição em voo, pela geração dela. O corpo é o
        # objeto que a janela passou, e não uma cópia: é por ele que ela acha a matriz dona da
        # predição que volta.
        self._corpo: Any = None
        self._corpo_do_voo: dict[int, Any] = {}

        # Piscina própria e não a da janela, pelo mesmo motivo dos sinais: `principal.py` importa
        # este módulo, e o caminho inverso é circular.
        self._piscina = QThreadPool()
        self._piscina.setMaxThreadCount(1)
        self._sinais = _SinaisDeVias(self)
        self._sinais.concluido.connect(self._recebe)
        self._sinais.falhou.connect(self._recebe_falha)

        raiz = QVBoxLayout(self)
        self.setProperty("folhaDeEtapa", True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        raiz.setContentsMargins(20, 20, 20, 20)
        raiz.setSpacing(16)

        # A primeira linha do painel, antes dos parâmetros: quem abre a aba lê que ela está em beta
        # antes de predizer. O texto vem de `desktop/recorte.py` para que a mesma frase, repetida
        # nos dois painéis em beta e no recorte, seja escrita num lugar só.
        self.rotulo_de_beta = QLabel(FUNCIONALIDADE_EM_BETA, self)
        self.rotulo_de_beta.setWordWrap(True)
        # A faixa de aviso, a mesma do painel do multiverso.
        self.rotulo_de_beta.setObjectName("faixa_beta")
        raiz.addWidget(self.rotulo_de_beta)

        # O filtro é atributo do painel porque o painel é o dono: criado solto dentro da chamada,
        # seria coletado ao retornar e deixaria de agir sem nada acusar.
        self._filtro_da_roda = FiltroDaRoda(self)

        # A linha 1 é o seletor do modo. Os dois modos respondem a perguntas diferentes: o primeiro
        # prediz vias a partir dos picos, sem identificação; o segundo enriquece a lista que o
        # laboratório identificou no Progenesis, com o MetaboAnalystR. A pilha preserva o estado de
        # cada um ao trocar.
        self.modo = sem_roda_sem_foco(QComboBox(self), self._filtro_da_roda)
        self.modo.addItem("Predição a partir dos picos (sem identificação)", "picos")
        self.modo.addItem(
            "Enriquecimento da lista identificada no Progenesis (MetaboAnalystR)", "metaboanalystr"
        )
        raiz.addWidget(self.modo)
        self.pilha = QStackedWidget(self)
        raiz.addWidget(self.pilha, 1)
        pagina = QWidget(self.pilha)
        corpo = QVBoxLayout(pagina)
        corpo.setContentsMargins(0, 0, 0, 0)
        # O cabeçalho seguinte fica entre as duas divisórias sem um vão extra acima.
        corpo.setSpacing(0)
        self.pilha.addWidget(pagina)
        self.enriquecimento = PainelDeEnriquecimento(self.pilha, **(enriquecimento or {}))
        self.pilha.addWidget(self.enriquecimento)
        self.modo.currentIndexChanged.connect(self.pilha.setCurrentIndex)

        # ------------------------------------------------------------------
        # Os parâmetros
        # ------------------------------------------------------------------
        # Duas seções numeradas: o que se escolhe, com o botão que calcula no fim, e o que volta
        # (o recorte, a tabela e o rodapé).
        self.secao_dos_picos = Secao(
            "Como interpretar os picos", pagina, numero=1, subtitulo=SUBTITULO_DOS_PICOS
        )
        self.secao_dos_picos.setProperty("painel_plano", True)
        moldura = QWidget(self.secao_dos_picos)
        self.secao_dos_picos.corpo.addWidget(moldura)
        formulario = QFormLayout(moldura)
        formulario.setContentsMargins(0, 0, 0, 0)
        formulario.setHorizontalSpacing(16)
        formulario.setVerticalSpacing(10)

        self.polaridade = sem_roda_sem_foco(QComboBox(moldura), self._filtro_da_roda)
        for valor, rotulo in _MODOS:
            self.polaridade.addItem(rotulo, valor)
        formulario.addRow("Polaridade da ionização", self.polaridade)

        self.politica = sem_roda_sem_foco(QComboBox(moldura), self._filtro_da_roda)
        for valor, rotulo in _POLITICAS:
            self.politica.addItem(rotulo, valor)
        # O rótulo em `MarkdownText`: passado como texto ao `addRow`, as crases de `…n` apareceriam
        # literais na tela.
        rotulo_da_politica = QLabel("Massas já deconvoluídas (`…n`)", moldura)
        rotulo_da_politica.setTextFormat(Qt.TextFormat.MarkdownText)
        formulario.addRow(rotulo_da_politica, self.politica)

        self.busca = sem_roda_sem_foco(QComboBox(moldura), self._filtro_da_roda)
        for valor, rotulo in rotulos_da_busca(self.polaridade.currentData()):
            self.busca.addItem(rotulo, valor)
        formulario.addRow("Formas iônicas procuradas", self.busca)
        # O rótulo acompanha a polaridade, e a escolha do usuário não muda, só o texto.
        # `currentIndexChanged` dispara também na troca programática, então `sugere_polaridade`
        # passa por aqui sem caminho próprio.
        self.polaridade.currentIndexChanged.connect(self._rotula_busca)

        # A explicação do combo fica ao lado dele e não num tooltip. O sufixo `n` separa massa
        # neutra de `m/z`, e no `pos_hilic` essa escolha decide o destino de 23% das features.
        nota = QLabel(
            "O equipamento publica duas formas de nome. As terminadas em `m/z` são o íon "
            "medido; as terminadas em `n` são a **massa neutra**, que o Progenesis já "
            "resolveu. Incluí-las assume que foram vistas como `[M+H]+` ou `[M-H]-` — a "
            "massa verdadeira fica entre os candidatos testados, junto de alguns falsos.",
            moldura,
        )
        nota.setTextFormat(Qt.TextFormat.MarkdownText)
        nota.setWordWrap(True)
        nota.setObjectName("nota_do_campo")
        formulario.addRow(nota)

        corpo.addWidget(self.secao_dos_picos)

        linha = QHBoxLayout()
        # `veste_de_botao` não é usado aqui: ele serve a botão que revela prosa, com ícone de
        # informação e `setCheckable(True)`. Este calcula, e o estado marcado deixaria o botão aceso
        # depois do clique.
        self.botao = QPushButton("Predizer vias", self)
        self.botao.clicked.connect(self._calcula_e_mostra)
        linha.addWidget(self.botao)
        linha.addStretch(1)
        self.secao_dos_picos.corpo.addLayout(linha)
        # Desligado ao nascer, e com a frase ao lado: ela só tem lugar depois de o botão estar na
        # caixa da seção.
        self._libera_o_botao()

        self.secao_das_vias = Secao(
            "As vias preditas", pagina, numero=2, subtitulo=SUBTITULO_DAS_VIAS
        )
        self.secao_das_vias.setProperty("painel_plano", True)
        corpo.addWidget(self.secao_das_vias, 1)
        resultado = self.secao_das_vias.corpo

        # ------------------------------------------------------------------
        # O recorte: a segunda coisa que o painel diz
        # ------------------------------------------------------------------
        self.recorte = QLabel(SEM_EXECUCAO, self)
        self.recorte.setTextFormat(Qt.TextFormat.MarkdownText)
        self.recorte.setWordWrap(True)
        self.recorte.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        resultado.addWidget(self.recorte)

        self.tabela = QTableWidget(0, len(COLUNAS), self)
        self.tabela.setHorizontalHeaderLabels([titulo for _chave, titulo in COLUNAS])
        cabecalho = self.tabela.horizontalHeader()
        if cabecalho is not None:  # pragma: no cover - o Qt devolve-o num widget construído
            cabecalho.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            # A coluna do nome estica e as quatro numéricas não: são os nomes de via, como
            # 'Vitamin B3 (nicotinate and nicotinamide) metabolism', que precisam do espaço.
            cabecalho.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tabela.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        # O espaço que sobra é da tabela, que é o conteúdo; os parâmetros têm altura fixa. A seção
        # tem fator 1 na página e a tabela fator 1 na seção, e o mínimo impede a página que rola de
        # a espremer.
        self.tabela.setMinimumHeight(ALTURA_MINIMA_DA_TABELA)
        resultado.addWidget(self.tabela, 1)

        self.rodape = QLabel("", self)
        self.rodape.setTextFormat(Qt.TextFormat.MarkdownText)
        self.rodape.setWordWrap(True)
        resultado.addWidget(self.rodape)

    # ----------------------------------------------------------------------
    # Entrada de dados
    # ----------------------------------------------------------------------

    def recebe_volcano(self, corpo: Any, *, por_troca: bool = False) -> None:
        """Guarda os vetores do último `/execucoes`. Não calcula.

        A predição leva segundos e o corpo chega quando a execução termina; calcular aqui faria
        toda correção pagar uma análise que o usuário pode não querer, antes de ele escolher a
        polaridade.

        `por_troca` diz que o corpo vem de a janela trocar de matriz, e não de uma execução nova: o
        enriquecimento em voo da matriz que sai não é cancelado (ver
        `PainelDeEnriquecimento.recebe_volcano`). A predição em voo não se cancela em caso nenhum.
        """
        self._corpo = corpo
        # Corpo que não é dicionário vira dicionário vazio em vez de rebentar no `.get`, como no
        # resto da janela.
        corpo = corpo if isinstance(corpo, dict) else {}
        self._geracao += 1
        self.enriquecimento.recebe_volcano(self._corpo, por_troca=por_troca)

        # A tabela anterior é invalidada aqui, e não só quando a próxima predição termina: sem isso,
        # a tabela da execução anterior ficaria na tela sob o recorte da nova.
        self._relatorio = None
        self.tabela.setRowCount(0)
        self.rodape.setText("")

        self._volcano = corpo.get("volcano")
        self._motivo_sem_volcano = (
            MOTIVO_SEM_PAR_DE_GRUPOS if "volcano" in corpo else MOTIVO_SEM_EXECUCAO
        )
        self._libera_o_botao()
        # Três estados: o campo `volcano` existe em todo corpo de execução concluída (`None` quando
        # não houve par de grupos), e o progresso, o erro e o corpo que não é dicionário não o têm.
        # A presença da chave separa 'a execução não tem par' de 'ainda não há execução concluída'.
        if self._volcano is not None:
            texto = "Execução nova recebida. Clique em «Predizer vias» para analisá-la."
        elif "volcano" in corpo:
            texto = SEM_PAR_DE_GRUPOS
        else:
            texto = SEM_EXECUCAO
        self.recorte.setText(texto)

    def _libera_o_botao(self) -> None:
        """Liga 'Predizer vias' se há volcano; sem ele, desliga com a frase da ausência ao lado."""
        if self._volcano is not None:
            religa(self.botao)
        else:
            desliga_com_motivo(self.botao, self._motivo_sem_volcano)

    def _rotula_busca(self, _indice: int = -1) -> None:
        """Reescreve os rótulos do combo da busca para a polaridade corrente. Não calcula."""
        # Por valor e não por posição, para que a mesma escolha continue marcada. As duas listas têm
        # os mesmos valores, então `findData` acha sempre.
        for valor, rotulo in rotulos_da_busca(self.polaridade.currentData()):
            self.busca.setItemText(self.busca.findData(valor), rotulo)

    def sugere_polaridade(self, modo: str | None) -> None:
        """Pré-seleciona a polaridade a partir do modo do estudo: `pos_*` é positivo.

        É sugestão e não decisão: o combo continua editável. Um modo que não comece por `pos` nem
        por `neg` deixa o combo como está, em vez de escolher uma polaridade que pode estar errada
        e cujo único sintoma seria 'menos vias'.
        """
        if not modo:
            return
        if modo.lower().startswith("pos"):
            alvo = "positive"
        elif modo.lower().startswith("neg"):
            alvo = "negative"
        else:
            return
        indice = self.polaridade.findData(alvo)
        if indice >= 0:
            self.polaridade.setCurrentIndex(indice)

    # ----------------------------------------------------------------------
    # O cálculo
    # ----------------------------------------------------------------------

    def _calcula_e_mostra(self) -> None:
        if self._volcano is None:  # pragma: no cover - o botão está desabilitado
            return
        # O local chama-se `volcano` por contrato: o teste de contrato da janela extrai a base de
        # cada leitura (o nome do local) e confere as chaves contra o esquema `ResumoDoVolcano`.
        # Ler de `self._volcano` direto deixaria as três chaves sem esquema contra o que conferir.
        volcano = self._volcano
        nomes = tuple(volcano["features"])
        pvalores = _vetor(volcano["pvalue"])
        estatisticas = _vetor(volcano["statistic"])

        chamada = functools.partial(
            self._calcula,
            nomes,
            pvalores,
            estatisticas,
            modo_de_ionizacao=self.polaridade.currentData(),
            politica_de_massa_neutra=self.politica.currentData(),
            busca_de_adutos=self.busca.currentData(),
        )

        # O cálculo sai do laço de eventos: no primeiro clique da sessão ele leva alguns segundos,
        # porque o modelo metabólico ainda não foi importado, e uma janela parada é indistinguível
        # de uma que travou. Desligar o botão sem sair do laço não bastaria: com a thread da GUI
        # bloqueada, o botão não repinta e os cliques enfileirados chegam depois.
        desliga_com_motivo(self.botao, MOTIVO_A_CALCULAR)
        self.recorte.setText(A_CALCULAR)
        self._corpo_do_voo[self._geracao] = self._corpo
        self._piscina.start(_TrabalhoDeVias(chamada, self._sinais, self._geracao))

    def _recebe(self, relatorio: object, geracao: int) -> None:
        """O caminho de sucesso, já de volta à thread da GUI.

        A predição sai pelo sinal com o corpo de que é, pintada ou não: a janela guarda-a com a
        matriz dona desse corpo, e é isso que a faz voltar com a matriz.
        """
        corpo = self._corpo_do_voo.pop(geracao, None)
        pintada = geracao == self._geracao
        # Na mesma geração há volcano, de onde o cálculo saiu, e o botão religa; noutra, vale o que
        # a execução nova trouxe, e sem volcano o botão desliga com a frase.
        self._libera_o_botao()
        if pintada:
            self._mostra(relatorio)  # type: ignore[arg-type]
        # De uma execução que já não é a da tela: não pinta, mas a predição viaja.
        if corpo is not None:
            self.predicao_concluida.emit(corpo, relatorio, pintada)

    def _recebe_falha(self, erro: object, geracao: int) -> None:
        """Toda falha limpa a tela; nenhuma deixa o resultado anterior à vista.

        Trocar de polaridade, clicar, não receber erro nenhum e ler o resultado da polaridade
        antiga é o pior desfecho que esta aba pode ter. Num executável sem console, uma exceção que
        escapasse do slot só apareceria em stderr, onde ninguém a vê.
        """
        self._corpo_do_voo.pop(geracao, None)
        self._libera_o_botao()
        if geracao != self._geracao:
            return
        self._relatorio = None
        self.tabela.setRowCount(0)
        self.rodape.setText("")
        if isinstance(erro, EntradaRecusadaError):
            # Só as recusas do núcleo têm a mensagem que a tela promete: elas nomeiam as causas com
            # a contagem de cada uma. Captura-se a base, e não uma filha, porque a entrada pequena
            # demais também é recusa com motivo.
            self.recorte.setText(f"**Não foi possível predizer vias.** {erro}")
            return
        # Qualquer outra é falha inesperada, e a tela diz isso: a mensagem de um `ValueError` do
        # `scipy` sob o cabeçalho das recusas leria-se como diagnóstico da entrada.
        self.recorte.setText(
            "**A predição de vias falhou de forma inesperada.** Isto não é um "
            f"diagnóstico da sua entrada — é um defeito a reportar: `{type(erro).__name__}: {erro}`"
        )

    def _mostra(self, relatorio: RelatorioDeVias) -> None:
        self._relatorio = relatorio
        self.recorte.setText(texto_do_recorte(relatorio))

        self.tabela.setRowCount(len(relatorio.vias))
        for linha, via in enumerate(relatorio.vias):
            celulas = (
                via.nome,
                str(via.sobreposicao),
                str(via.detectados_na_via),
                numero(via.pvalue),
                numero(via.pvalue_fisher),
            )
            for coluna, texto in enumerate(celulas):
                item = QTableWidgetItem(texto)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tabela.setItem(linha, coluna, item)

        quantas = len(relatorio.significativas)
        # O rodapé diz o que a tabela não afirma: os compostos são candidatos por massa, e nenhum
        # deles foi identificado.
        self.rodape.setText(
            f"**{quantas} via(s) abaixo de p = 0,05.** Os compostos por trás de cada via são "
            "**candidatos compatíveis em massa**, não identificações: este programa não "
            "identifica metabólitos a partir de `tempo_massa`, e a predição de via existe "
            "justamente para dispensar essa etapa."
        )

    def _em_voo(self) -> bool:
        """Se há uma predição em voo para a execução da tela, a da geração corrente."""
        return self._geracao in self._corpo_do_voo

    def corpo_na_tela(self) -> Any:
        """O corpo da execução que o painel mostra: o objeto que a janela passou."""
        return self._corpo

    def repoe_a_predicao(self, relatorio: RelatorioDeVias) -> None:
        """Pinta uma predição guardada do corpo da tela, na volta à matriz dela.

        Só a janela a chama, e só com o corpo de que a predição é. Com uma predição corrente em
        voo, não pinta: a dela chega a seguir, e o botão desligado diz que ela corre.
        """
        if self._em_voo():
            return
        self._mostra(relatorio)
        self.botao.setEnabled(self._volcano is not None)

    def encerra(self, teto_ms: int = 10_000) -> bool:
        """No fechamento da janela: cancela o R em voo e só então drena as duas piscinas.

        O destrutor de `QThreadPool` espera sem teto, e um R em voo prenderia o processo depois de
        a janela sumir; o cancelamento encerra o Job do R, e aí a espera tem fim.
        """
        enriquecimento_ok = self.enriquecimento.encerra(teto_ms)
        self._piscina.clear()
        # As duas esperas correm sempre: um `and` curto-circuitado deixaria de drenar esta piscina
        # quando a do enriquecimento não drena.
        vias_ok = bool(self._piscina.waitForDone(teto_ms))
        return enriquecimento_ok and vias_ok

    def linhas(self) -> tuple[tuple[str, ...], ...]:
        """O que está visível na tabela, para o teste.

        Lê os widgets e não o relatório guardado: uma linha construída fora do layout não é uma
        linha que o usuário vê.
        """
        return tuple(
            tuple(self._celula(linha, coluna) for coluna in range(self.tabela.columnCount()))
            for linha in range(self.tabela.rowCount())
        )

    def _celula(self, linha: int, coluna: int) -> str:
        """O texto de uma célula, ou `""` quando ela não existe.

        A célula ausente vira texto vazio e não é omitida porque o teste compara posições: uma
        linha que encolhesse deslocaria as seguintes, e a coluna errada seria comparada sem aviso.
        """
        item = self.tabela.item(linha, coluna)
        return "" if item is None else item.text()


def _vetor(valores: list[float | None]) -> Any:
    """`None` do JSON vira `nan`, que o núcleo sabe recusar por feature.

    O esquema manda `float | None` porque as listas indexam as mesmas features e a posição tem de
    sobreviver; `None` não atravessa `np.isfinite`. Converter aqui mantém o alinhamento e deixa a
    contagem de descartados com o núcleo.
    """
    import numpy as np

    return np.array([np.nan if v is None else float(v) for v in valores], dtype=np.float64)

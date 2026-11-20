"""Painel 3: os trabalhos em curso.

Responde a uma pergunta, 'já ficou pronto?', para todos os trabalhos longos num lugar só: o
Palpite, a medição dos corretores, as etapas do passo a passo, os três arquivos e a execução.
Cada trabalho é uma linha com o nome legível, o estado num selo, a barra de etapas, a etapa em
curso e o botão de parar daquela linha; o identificador fica na dica do nome. A faixa se relê
sozinha enquanto houver trabalho sem desfecho. O seletor com 'Parar o trabalho selecionado'
ficou redundante com o botão de cada linha; retirá-lo é escolha de produto, ainda não feita.

Os trabalhos não sobrevivem ao fechamento: a API corre numa thread do mesmo processo
(`arranque.arranca`), e o registro de progresso vive em memória. O painel diz isso em vez de
prometer recuperação. A lista vem da janela, que registra `(chave, identificador)` de todo `202`
que passa, e não de uma rota de índice, que exporia os trabalhos retidos numa API sem
autenticação. Como nos outros painéis, nenhuma chamada de rota nasce aqui: ele compõe texto e
emite sinal, e quem chama a rota é a janela.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from refino.api.progresso import TERMINAIS
from refino.desktop.janela.aparencia import FiltroDaRoda, sem_roda_sem_foco
from refino.desktop.janela.componentes import (
    Ajuda,
    Selo,
    com_maiuscula,
    desliga_com_motivo,
    religa,
)
from refino.desktop.janela.tema import marca_papel

__all__ = [
    "AVISO_DO_FECHO",
    "CHAVE_DOS_TRABALHOS",
    "CHAVE_DO_CANCELAMENTO_DE_TRABALHO",
    "DICA_DA_BARRA",
    "INTERVALO_DA_RELEITURA_MS",
    "LARGURA_DO_ESTADO",
    "MOTIVO_JA_TERMINOU",
    "NAO_LIDO",
    "NOMES_DOS_TRABALHOS",
    "PARANDO",
    "ROTULOS_DOS_ESTADOS",
    "SEM_TRABALHOS",
    "SEM_TRABALHO_A_PARAR",
    "TONS_DOS_ESTADOS",
    "LinhaDoTrabalho",
    "PainelDeTrabalhos",
    "linha_do_trabalho",
    "nome_do_trabalho",
]

CHAVE_DOS_TRABALHOS: Final = "trabalhos"
"""A chave deste painel. Constante porque aparece no painel, em `PAINEIS` e na composição."""

CHAVE_DO_CANCELAMENTO_DE_TRABALHO: Final = "cancelamento_de_trabalho"
"""A chave da resposta do pedido de parada.

Tem base própria, e não `trabalhos@…`, porque `_encaminha` roteia pela base: com a base
`trabalhos`, a resposta de cancelamento seria lida como leitura de progresso."""

SEM_TRABALHO_A_PARAR: Final = (
    "Não há trabalho selecionado para parar. Dispare uma operação longa numa das outras abas — ela "
    "aparece nesta lista — e escolha-a no seletor acima."
)
"""Frase e não silêncio: botão que recusa sem dizer é indistinguível de botão morto."""

MOTIVO_JA_TERMINOU: Final = "Terminou: não há o que parar."
"""A frase ao lado do 'Parar' de um trabalho que já chegou ao fim. Curta porque a linha empilhada
vive na barra lateral, ao lado do nome; o estado do fim já está no selo da mesma linha."""

SEM_TRABALHOS: Final = (
    "Nenhum trabalho disparado nesta janela. As operações longas — o Palpite, a medição dos "
    "corretores, as etapas do passo a passo, a execução do pipeline e os três arquivos — continuam "
    "a correr depois do clique; elas aparecem aqui quando você as disparar."
)
"""O repouso, e ele diz o que faz um trabalho aparecer.

Frase e não lista vazia, porque lista vazia é indistinguível de painel quebrado."""

AVISO_DO_FECHO: Final = (
    "FECHAR O APLICATIVO ENCERRA OS TRABALHOS. Eles correm dentro deste processo, junto com a "
    "janela: fechá-la interrompe o que ainda estiver a correr, e reabrir o aplicativo não os "
    "retoma — o identificador de cada um, na dica do nome dele, só vale enquanto esta janela "
    "estiver aberta."
)
"""O aviso de que fechar o aplicativo encerra os trabalhos.

A API corre dentro deste processo, então o identificador de um trabalho não sobrevive ao
fechamento. Uma instrução como 'guarde este identificador' afirmaria uma recuperação que não
existe."""

INTERVALO_DA_RELEITURA_MS: Final = 10_000
"""De quanto em quanto a faixa se relê sozinha, enquanto houver trabalho sem desfecho.

Dez segundos, e não o segundo da sondagem da execução: aquela escreve o progresso de um trabalho,
e esta resume todos e só precisa acertar o desfecho. É um GET por trabalho a cada dez segundos, e
nenhum quando todos terminaram. Por isso a faixa pode levar até dez segundos para dizer
'concluído' depois do painel do pipeline; fechar essa folga exigiria o painel do pipeline avisar
a faixa, e essa ligação vive em `principal.py`."""

NOMES_DOS_TRABALHOS: Final[dict[str, str]] = {
    "recomendacao": "Palpite",
    "medicao": "Medição dos corretores",
    "etapa_correcao": "Etapa 1 — correção",
    "etapa_preprocessamento": "Etapa 2 — normalização",
    "corrigir": "Matriz corrigida",
    "processar": "Matriz normalizada",
    "exportar": "Arquivo para o MetaboAnalyst",
    "execucao": "Etapa 3 — análise estatística",
}
"""O nome de cada trabalho, na língua de quem o disparou.

As chaves são as que `principal.py` registra, e um teste confere a bijeção. Os nomes são escritos
aqui e não derivados dos rótulos dos botões, porque um trabalho é um nome e um botão é uma ação:
'Gerar e baixar a matriz corrigida' dispara a 'Matriz corrigida'. A chave `recomendacao` diz
'Palpite' porque é a chave do ponto de decisão, e o que corre por ela é a medição do Palpite.

Chave fora da tabela não levanta: aparece marcada, ver `nome_do_trabalho`."""

ROTULOS_DOS_ESTADOS: Final[dict[str, str]] = {
    "pendente": "Pendente",
    "executando": "Executando",
    "concluido": "Concluído",
    "falhou": "Falhou",
    "cancelado": "Parado a seu pedido",
}
"""O estado que o registro produz, legível; um teste confere a bijeção com `api.progresso.ESTADOS`.

As palavras são as do registro, e não sinônimos: o painel do pipeline escreve 'Executando: 0/7
etapas' na mesma tela, e dois nomes para o mesmo estado lado a lado confundiriam. A exceção é
`cancelado`, que diz 'parado a seu pedido': a parada é obediência e não pode ler-se como falha."""

TONS_DOS_ESTADOS: Final[dict[str, str]] = {
    "pendente": "neutro",
    "executando": "info",
    "concluido": "sucesso",
    "falhou": "erro",
    "cancelado": "aviso",
}
"""A cor do selo de cada estado. Mesmo domínio que `ROTULOS_DOS_ESTADOS`, e o mesmo teste."""

NAO_LIDO: Final = "Não lido"
"""O estado de um trabalho que a faixa ainda não leu, com a palavra dos cartões de resultado da
leitura (`componentes.py`).

Não é 'Pendente': aquele é o que o registro diz; este é a faixa a dizer que não perguntou."""

PARANDO: Final = "A parar…"
"""O intervalo entre o pedido de parada e a parada. Vem da resposta do pedido e, depois dela, de
toda leitura que traga `cancelamento_pedido` com o trabalho ainda a correr."""

_PRAZO_DA_PARADA: Final = (
    "o pedido foi aceito. O trabalho para no próximo ponto de checagem; correção pesada pode levar "
    "mais que alguns segundos"
)
# O selo diz o estado numa palavra, e a explicação vai para a nota: a coluna do estado tem largura
# fixa e precisa caber em todas as linhas, e frases como 'Identificador desconhecido' não cabem.
_DESCONHECIDO: Final = "Desconhecido"
_PORQUE_DESCONHECIDO: Final = "o identificador foi podado, ou a API reiniciou"
_JA_TERMINARA: Final = "Já tinha terminado"
_NAO_PAROU: Final = "Não parou"

LARGURA_DO_ESTADO: Final = 150
"""A coluna do selo, em px. A pílula mais larga do vocabulário fixo, 'Parado a seu pedido', mede
143 px no topo da janela (fonte de 12 px); um estado fora do vocabulário cai em 'Desconhecido'."""

LARGURA_DO_PARAR: Final = 200
"""A coluna do 'Parar' na linha larga, em px, com a frase do botão desligado dentro dela.

Fixa pelo mesmo motivo da coluna do estado: a frase só aparece na linha de um trabalho terminado,
e se a largura dela decidisse a coluna, a barra mudaria de lugar só nessa linha. O botão ocupa
58 px, e `MOTIVO_JA_TERMINOU` recebe o resto, em duas linhas."""

DICA_DA_BARRA: Final = (
    "A barra conta etapas terminadas: ela só anda quando uma etapa termina, e a etapa em curso não "
    "reporta fração por dentro."
)
"""A barra parada não é trabalho parado, e a dica diz por quê: a correção é quase toda a espera e
é uma etapa só, então '0/4 etapas' pode durar muitos minutos."""


@dataclass(frozen=True)
class _Leitura:
    """O que a última resposta disse de um trabalho, nas peças que a linha desenha.

    Uma estrutura e não a frase pronta, porque a linha tem quatro widgets: guardar só a frase
    obrigaria a linha a reparti-la de volta. `maximo == 0` é a barra indeterminada, a convenção do
    `QProgressBar`.
    """

    estado: str
    tom: str
    nota: str = ""
    maximo: int = 1
    valor: int = 0
    de_etapas: bool = False

    @property
    def contagem(self) -> str:
        """'v de m etapas' quando a barra conta etapas, e vazio quando não conta nada."""
        return f"{self.valor} de {self.maximo} etapas" if self.de_etapas else ""

    @property
    def frase(self) -> str:
        """A linha inteira em texto, na ordem em que a linha a desenha: estado, contagem, nota."""
        return " — ".join(parte for parte in (self.estado, self.contagem, self.nota) if parte)

    def desfecho(self, estado: str, tom: str, nota: str = "", *, cheia: bool = False) -> _Leitura:
        """Um fim, ou o pedido que já não precisa de animação: a barra para onde estava.

        Cheia só quando `cheia` (o concluído), porque um trabalho parado ou falhado na etapa 2 de 4
        não chegou ao fim, e uma barra cheia ao lado de 'Falhou' contradiria o selo. A
        indeterminada vira vazia: animação depois do fim é sinal de vida falso.
        """
        if self.maximo == 0:
            return _Leitura(estado, tom, nota, maximo=1, valor=1 if cheia else 0)
        valor = self.maximo if cheia else self.valor
        return _Leitura(estado, tom, nota, self.maximo, valor, self.de_etapas)


_REPOUSO_DA_LINHA: Final = _Leitura(NAO_LIDO, "neutro")
"""A linha antes da primeira leitura: selo neutro e barra vazia, e não indeterminada.

O `202` do disparo diz que o trabalho foi aceito; o que ele está a fazer, a faixa ainda não
perguntou, e animação aqui seria sinal de vida falso."""


def nome_do_trabalho(chave: str) -> str:
    """O nome legível da chave, ou a chave marcada; nunca silêncio nem exceção.

    Levantar aqui mataria o slot que registra o trabalho, e com ele a espera do artefato que vem a
    seguir em `_encaminha`. A marca é o que torna o defeito visível na tela em vez de no `stderr`.
    """
    nome = NOMES_DOS_TRABALHOS.get(chave)
    return nome if nome is not None else f"«{chave}» — sem nome legível, defeito desta tela"


def linha_do_trabalho(chave: str, estado: str) -> str:
    """Uma linha da lista em texto: o que é, e como está.

    É a frase que `PainelDeTrabalhos.mostra` devolve; o identificador vive na dica do nome.
    """
    return f"{nome_do_trabalho(chave)} — {estado}"


def _leitura_em_curso(corpo: dict[str, Any], anterior: _Leitura) -> _Leitura:
    """Um `202` de `GET /trabalhos/{id}` nas peças da linha.

    A barra vem de `feito`/`total`, e sem `total` fica indeterminada: há trabalho e não se sabe
    quanto falta. Um estado terminal num `202`, que a rota hoje não serve, lê-se como fim pelo
    conjunto `TERMINAIS`. Estado fora do vocabulário aparece marcado, com o valor cru, em vez de
    cair num rótulo razoável que o leria como outro.
    """
    bruto = corpo.get("estado")
    estado = "" if bruto is None else str(bruto)
    if estado in TERMINAIS:
        return anterior.desfecho(
            ROTULOS_DOS_ESTADOS[estado], TONS_DOS_ESTADOS[estado], cheia=estado == "concluido"
        )
    etapa = corpo.get("etapa_em_curso")
    nota = f"etapa em curso: {etapa}" if isinstance(etapa, str) and etapa else ""
    if not estado:
        rotulo, tom = "Em curso", "info"
    elif estado in ROTULOS_DOS_ESTADOS:
        rotulo, tom = ROTULOS_DOS_ESTADOS[estado], TONS_DOS_ESTADOS[estado]
    else:
        rotulo, tom = _DESCONHECIDO, "aviso"
        nota = " — ".join(p for p in (f"estado «{estado}», que esta faixa não conhece", nota) if p)
    if corpo.get("cancelamento_pedido") is True:
        rotulo, tom = PARANDO, "aviso"
    feito, total = corpo.get("feito"), corpo.get("total")
    if isinstance(total, int) and total > 0 and isinstance(feito, int):
        valor = min(max(feito, 0), total)
        return _Leitura(rotulo, tom, nota, maximo=total, valor=valor, de_etapas=True)
    return _Leitura(rotulo, tom, nota, maximo=0, valor=0)


class LinhaDoTrabalho(QWidget):
    """Uma linha da faixa: o nome, o estado, a barra, a etapa e o botão de parar.

    O botão emite a chave desta linha, guardada no construtor. Um `lambda` sobre a variável do laço
    que cria as linhas faria todos os botões emitirem a última chave.
    """

    parar = Signal(str)

    def __init__(
        self, chave: str, parent: QWidget | None = None, *, compacta: bool = False
    ) -> None:
        super().__init__(parent)
        self.chave = chave
        self.compacta = compacta
        disposicao = QHBoxLayout()
        disposicao.setContentsMargins(0, 0, 0, 0)
        disposicao.setSpacing(10)

        # As colunas alinham entre linhas porque a caixa do estado tem largura fixa e a nota está
        # sempre presente; a política de tamanho do nome não muda a coluna.
        self.nome = QLabel(nome_do_trabalho(chave), self)
        self.nome.setObjectName("nome_do_trabalho")
        self.nome.setTextFormat(Qt.TextFormat.PlainText)
        self.nome.setWordWrap(True)
        self.nome.setMinimumWidth(130)
        fonte = self.nome.font()
        fonte.setBold(True)
        self.nome.setFont(fonte)
        disposicao.addWidget(self.nome, 2)

        # O selo guarda o tamanho natural da pílula, dentro de uma caixa da largura da coluna.
        caixa_do_estado = QWidget(self)
        caixa_do_estado.setFixedWidth(LARGURA_DO_ESTADO)
        linha_do_estado = QHBoxLayout(caixa_do_estado)
        linha_do_estado.setContentsMargins(0, 0, 0, 0)
        self.estado = Selo(NAO_LIDO, "neutro", caixa_do_estado)
        linha_do_estado.addWidget(self.estado)
        linha_do_estado.addStretch(1)
        disposicao.addWidget(caixa_do_estado)

        self.barra = QProgressBar(self)
        self.barra.setObjectName("barra_do_trabalho")
        self.barra.setFixedWidth(150)
        self.barra.setToolTip(DICA_DA_BARRA)
        disposicao.addWidget(self.barra)

        self.nota = QLabel("", self)
        self.nota.setObjectName("nota_do_trabalho")
        self.nota.setTextFormat(Qt.TextFormat.PlainText)
        self.nota.setWordWrap(True)
        # A nota longa, como o `detail` de uma falha, não alarga o mínimo da faixa.
        self.nota.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        disposicao.addWidget(self.nota, 3)

        # O 'Parar' numa caixa própria, e é nela que a frase do botão desligado entra ao lado dele
        # (`desliga_com_motivo`). Na linha larga a caixa tem a largura fixa de `LARGURA_DO_PARAR`.
        self.caixa_do_parar = QWidget(self)
        linha_do_parar = QHBoxLayout(self.caixa_do_parar)
        linha_do_parar.setContentsMargins(0, 0, 0, 0)
        linha_do_parar.setSpacing(6)
        self.botao_de_parar = QPushButton("Parar", self.caixa_do_parar)
        # O mesmo papel de 'Parar o trabalho selecionado', marcado aqui porque as linhas nascem
        # depois de `veste_a_janela` ter marcado os botões que existiam.
        marca_papel(self.botao_de_parar, "perigo")
        self.botao_de_parar.setToolTip("Pede a parada deste trabalho")
        self.botao_de_parar.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.botao_de_parar.clicked.connect(self._pede_parada)
        linha_do_parar.addWidget(self.botao_de_parar, 0, Qt.AlignmentFlag.AlignLeft)
        self.caixa_do_parar.setFixedWidth(LARGURA_DO_PARAR)
        disposicao.addWidget(self.caixa_do_parar)
        if compacta:
            self._empilha(disposicao, caixa_do_estado)
        else:
            self.setLayout(disposicao)
        self.atualiza("", _REPOUSO_DA_LINHA, terminou=False)

    def _empilha(self, em_linha: QHBoxLayout, caixa_do_estado: QWidget) -> None:
        """A mesma linha em três andares, para a barra lateral.

        Os widgets são os mesmos e só a disposição muda. Em uma fileira, o nome (130 px), a caixa
        do estado (150) e a barra (150) não cabem nos cerca de 196 px úteis da barra lateral.
        Empilhado, o nome divide a primeira fileira com o 'Parar', o estado e a barra dividem a
        segunda, e a nota fica embaixo.

        A nota vazia some aqui, e na linha larga fica (`atualiza`): mantê-la serve para alinhar a
        coluna da barra entre linhas, e empilhado não há coluna a alinhar.
        """
        while em_linha.count():
            em_linha.takeAt(0)
        coluna = QVBoxLayout(self)
        coluna.setContentsMargins(0, 0, 0, 0)
        coluna.setSpacing(4)
        self.nome.setMinimumWidth(0)
        primeira = QHBoxLayout()
        primeira.setSpacing(6)
        primeira.addWidget(self.nome, 1)
        # A caixa do 'Parar' perde a largura fixa da linha larga, como a do estado, e não cresce
        # além do que o botão e a frase pedem: a folga da fileira é do nome.
        self.caixa_do_parar.setMinimumWidth(0)
        self.caixa_do_parar.setMaximumWidth(16777215)
        self.caixa_do_parar.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        primeira.addWidget(self.caixa_do_parar, 0, Qt.AlignmentFlag.AlignTop)
        coluna.addLayout(primeira)
        segunda = QHBoxLayout()
        segunda.setSpacing(6)
        # A caixa do estado perde a largura fixa, que existia para a coluna alinhar entre linhas,
        # e fica do tamanho do selo. O selo não muda de pai.
        caixa_do_estado.setMinimumWidth(0)
        caixa_do_estado.setMaximumWidth(16777215)
        caixa_do_estado.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        segunda.addWidget(caixa_do_estado, 0)
        self.barra.setMinimumWidth(40)
        self.barra.setMaximumWidth(16777215)
        self.barra.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        segunda.addWidget(self.barra, 1)
        coluna.addLayout(segunda)
        coluna.addWidget(self.nota)

    def _pede_parada(self) -> None:
        self.parar.emit(self.chave)

    def atualiza(self, identificador: str, leitura: _Leitura, *, terminou: bool) -> str:
        """Desenha a leitura e devolve a linha em texto.

        O botão desliga-se no fim, e só no fim: depois do desfecho, carregar nele produziria só um
        `409`; enquanto 'A parar…', pedir outra vez é idempotente na rota e fica possível.
        Desligado, leva a frase ao lado (`desliga_com_motivo`).
        """
        self.nome.setToolTip(f"Identificador do trabalho: {identificador}" if identificador else "")
        self.estado.define(leitura.estado, leitura.tom)
        if leitura.maximo == 0:
            self.barra.setRange(0, 0)
        else:
            self.barra.setRange(0, leitura.maximo)
            self.barra.setValue(leitura.valor)
        self.barra.setFormat("%v de %m etapas" if leitura.de_etapas else "")
        self.barra.setTextVisible(leitura.de_etapas)
        # A nota fica na linha mesmo vazia: escondê-la devolveria o espaço dela ao nome, e a coluna
        # da barra mudaria de lugar só nessa linha. Na linha empilhada não há coluna, e ela some.
        self.nota.setText(com_maiuscula(leitura.nota))
        if self.compacta:
            self.nota.setVisible(bool(self.nota.text()))
        if terminou:
            desliga_com_motivo(self.botao_de_parar, MOTIVO_JA_TERMINOU)
        else:
            religa(self.botao_de_parar)
        return linha_do_trabalho(self.chave, leitura.frase)


class PainelDeTrabalhos(QWidget):
    """Os trabalhos que esta janela disparou, com um botão que pergunta por todos.

    `atualizar` não leva argumento: o que se pede é 'atualiza todos', e quem sabe quais são é a
    janela. Um sinal com a chave obrigaria o painel a decidir a ordem das consultas, que é decisão
    de quem as despacha. O botão e o relógio da releitura automática emitem o mesmo sinal, porque
    pedem a mesma coisa.
    """

    atualizar = Signal()
    cancelar = Signal(str)
    """Pede a parada de um trabalho. Leva a chave e nunca o identificador.

    Quem mapeia chave para identificador é a janela (`principal.trabalhos_conhecidos`). Emitir o
    identificador daqui obrigaria o painel a guardar uma segunda cópia do mapa, e duas cópias
    divergem no dia em que uma execução nova reusar a chave. Emitem-no o botão do seletor e o
    'Parar' de cada linha (`LinhaDoTrabalho.parar`)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Uma barra de status: à esquerda, o estado numa palavra (`Selo`) e, só quando há trabalho,
        # a lista dele; à direita, os controles. O aviso do fecho mostra à vista a primeira frase,
        # que é o fato, e inteiro no ⓘ, com a causa e a consequência.
        disposicao = QVBoxLayout(self)
        disposicao.setContentsMargins(0, 0, 0, 0)
        disposicao.setSpacing(4)
        comando = QHBoxLayout()
        comando.setSpacing(8)

        self.selo = Selo("Nenhum trabalho", "neutro", self)
        self.selo.setToolTip(SEM_TRABALHOS)
        comando.addWidget(self.selo)

        self.rotulo_da_lista = QLabel(SEM_TRABALHOS, self)
        self.rotulo_da_lista.setWordWrap(True)
        # Markdown porque `AutoText` escolhe entre texto simples e HTML e não reconhece markdown, e
        # a nota da releitura usa itálico. O rótulo fica com o que é frase e não trabalho: a nota da
        # releitura e a recusa do botão do seletor.
        self.rotulo_da_lista.setTextFormat(Qt.TextFormat.MarkdownText)
        # A frase de repouso vive na dica do selo; a lista só aparece quando há o que listar.
        self.rotulo_da_lista.setVisible(False)
        comando.addWidget(self.rotulo_da_lista, 1)
        comando.addStretch(0)

        # Um filtro da roda por painel, com pai, e os campos roláveis passam por
        # `sem_roda_sem_foco`. Sem pai ele seria coletado e deixaria de agir sem nada acusar.
        self._filtro_da_roda = FiltroDaRoda(self)

        # O seletor e o botão de parar dele continuam, com o nome legível e a chave como dado do
        # item. Ficaram redundantes com o 'Parar' de cada linha; retirá-los é escolha de produto,
        # ainda não feita.
        self.seletor_de_trabalho = sem_roda_sem_foco(QComboBox(self), self._filtro_da_roda)
        # O combo acompanha o conteúdo, com mínimo de 150 px e teto de 220: sem teto ele comeria
        # metade da linha, e o nome mais longo pede cerca de 162 px.
        self.seletor_de_trabalho.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.seletor_de_trabalho.setMinimumWidth(150)
        self.seletor_de_trabalho.setMaximumWidth(220)
        comando.addWidget(self.seletor_de_trabalho)

        self.botao_de_atualizacao = QPushButton("Atualizar", self)
        self.botao_de_atualizacao.clicked.connect(self.atualizar.emit)
        self.botao_de_atualizacao.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        comando.addWidget(self.botao_de_atualizacao)

        self.botao_de_cancelamento = QPushButton("Parar o trabalho selecionado", self)
        self.botao_de_cancelamento.clicked.connect(self.pede_cancelamento)
        self.botao_de_cancelamento.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        comando.addWidget(self.botao_de_cancelamento)
        disposicao.addLayout(comando)

        # As linhas, uma por trabalho, na ordem das chaves, debaixo dos controles. A janela vestida
        # põe a nota da releitura entre os controles e elas (`tema`); o conteúdo é o mesmo nas duas
        # composições.
        self.lista = QWidget(self)
        self.lista.setObjectName("linhas_dos_trabalhos")
        self._pilha_das_linhas = QVBoxLayout(self.lista)
        self._pilha_das_linhas.setContentsMargins(0, 2, 0, 0)
        self._pilha_das_linhas.setSpacing(6)
        self.lista.setVisible(False)
        disposicao.addWidget(self.lista)
        self.linhas: dict[str, LinhaDoTrabalho] = {}
        # A forma das linhas que `_acerta_as_linhas` cria: larga por padrão; a barra lateral pede a
        # empilhada com `compacta_as_linhas`.
        self.linhas_compactas = False

        # O aviso inteiro continua neste rótulo, que os testes conferem, e vai para o ⓘ. À vista
        # fica a primeira frase, derivada da mesma constante: o fato que não pode faltar a quem
        # fecha a janela, sem as maiúsculas.
        self.rotulo_do_aviso = QLabel(AVISO_DO_FECHO, self)
        self.rotulo_do_aviso.setWordWrap(True)
        self.rotulo_do_aviso.setVisible(False)
        primeira_frase = AVISO_DO_FECHO.split(". ", 1)[0].capitalize() + "."
        # O aviso curto nasce aqui e a janela o coloca na linha do título da faixa, como
        # `PainelDeLeitura.sobrescrita`: o painel constrói, a janela desenha. Na linha dos controles
        # ele alargaria o mínimo da janela; na linha do título, que é curta, não soma nada.
        self.aviso_do_fecho = QWidget(self)
        linha_do_aviso = QHBoxLayout(self.aviso_do_fecho)
        linha_do_aviso.setContentsMargins(0, 0, 0, 0)
        linha_do_aviso.setSpacing(6)
        self.aviso_curto = QLabel(primeira_frase, self.aviso_do_fecho)
        self.aviso_curto.setObjectName("aviso_do_fecho")
        linha_do_aviso.addWidget(self.aviso_curto)
        linha_do_aviso.addWidget(Ajuda(AVISO_DO_FECHO, self.aviso_do_fecho))

        # A folga vai para o fim: este painel é de rótulos, e sem dono ela reparte-se entre eles.
        disposicao.addStretch(1)

        # `chave -> leitura`, escrito por `mostra_estado` e lido por `mostra`. É estado do painel e
        # não da janela: a janela sabe quais trabalhos existem; o estado de cada um é o que a última
        # leitura devolveu.
        self._leituras: dict[str, _Leitura] = {}
        # As chaves cujo último estado lido é um fim. Separado do rótulo porque decidir 'ainda
        # corre?' por substring de prosa quebraria na primeira mudança de texto.
        self._terminados: set[str] = set()
        # `chave -> identificador` da última lista mostrada. Redisparar a execução troca o
        # identificador na mesma chave, e é este mapa que deixa `mostra` esquecer o estado lido do
        # trabalho anterior.
        self._identificadores: dict[str, str] = {}

        # A releitura automática: um relógio de disparo único, rearmado por `mostra` enquanto houver
        # chave sem desfecho, que pede a mesma atualização do botão. O painel continua sem chamar
        # rota: emite `atualizar`, e quem chama é a janela.
        self.intervalo_da_releitura_ms = INTERVALO_DA_RELEITURA_MS
        self._releitura = QTimer(self)
        self._releitura.setSingleShot(True)
        self._releitura.timeout.connect(self._rele_sozinha)

    def mostra(self, trabalhos: dict[str, str]) -> str:
        """Escreve a lista e devolve o texto escrito.

        Vazio escreve a frase de repouso e não uma lista de zero linhas, porque 'não disparei nada'
        e 'o painel não está a mostrar' se leem de forma diferente. Com trabalhos, o texto devolvido
        é o das linhas, uma por trabalho e na ordem das chaves, tal como `LinhaDoTrabalho.atualiza`
        as desenhou.
        """
        # O seletor acompanha a lista preservando a seleção, pela chave, que é o dado do item.
        # Repovoá-lo do zero faria a escolha saltar para o primeiro item enquanto a pessoa lê, e o
        # botão de parar passaria a apontar para outro trabalho sem nada mudar na tela.
        escolhida = self.seletor_de_trabalho.currentData()
        self.seletor_de_trabalho.clear()
        for chave in sorted(trabalhos):
            self.seletor_de_trabalho.addItem(nome_do_trabalho(chave), chave)
        if escolhida in trabalhos:
            self.seletor_de_trabalho.setCurrentIndex(self.seletor_de_trabalho.findData(escolhida))

        # Um identificador novo na mesma chave é outro trabalho, e o que se leu do anterior não é
        # dele. Sem isso o 'concluído' da corrida passada ficaria ao lado do identificador novo, e a
        # releitura nem armaria.
        for chave, identificador in trabalhos.items():
            if self._identificadores.get(chave, identificador) != identificador:
                self._leituras.pop(chave, None)
                self._terminados.discard(chave)
        self._identificadores = dict(trabalhos)
        self._acerta_as_linhas(trabalhos)

        em_curso = sorted(chave for chave in trabalhos if chave not in self._terminados)
        if not em_curso:
            self._releitura.stop()
        elif not self._releitura.isActive():
            # `isActive` em vez de rearmar sempre: cada resposta chama `mostra`, e rearmar a cada
            # uma empurraria a próxima leitura para a frente enquanto chegassem respostas.
            self._releitura.start(self.intervalo_da_releitura_ms)

        # O selo da barra: o estado numa palavra. A lista só aparece quando há o que listar.
        if not trabalhos:
            self.selo.define("Nenhum trabalho", "neutro")
        elif em_curso:
            self.selo.define(f"{len(em_curso)} em execução", "info")
        else:
            self.selo.define("Concluídos", "sucesso")
        self.lista.setVisible(bool(trabalhos))
        if not trabalhos:
            self.rotulo_da_lista.setText(SEM_TRABALHOS)
            self.rotulo_da_lista.setVisible(False)
            return SEM_TRABALHOS
        texto = "\n\n".join(
            self.linhas[chave].atualiza(
                trabalhos[chave],
                self._leituras.get(chave, _REPOUSO_DA_LINHA),
                terminou=chave in self._terminados,
            )
            for chave in sorted(trabalhos)
        )
        # A nota só é escrita enquanto o relógio está armado, para a faixa não prometer releitura
        # depois de ter parado. Com tudo terminado o rótulo some em vez de ocupar a linha vazio.
        nota = (
            f"*A faixa é relida sozinha a cada {self.intervalo_da_releitura_ms / 1000:.0f} s "
            "enquanto houver trabalho em curso — «Atualizar» lê agora*"
            if em_curso
            else ""
        )
        self.rotulo_da_lista.setText(nota)
        self.rotulo_da_lista.setVisible(bool(nota))
        return f"{texto}\n\n{nota}" if nota else texto

    def compacta_as_linhas(self) -> None:
        """As linhas criadas daqui em diante são as empilhadas, que cabem na barra lateral.

        Levanta se já houver linha, em vez de deixar as antigas largas ao lado das novas: a mistura
        não acusaria nada, e a linha larga transborda a barra. Quem chama é `tema.veste_a_janela`,
        antes de qualquer trabalho existir.
        """
        if self.linhas:
            raise RuntimeError(
                f"compacta_as_linhas com {len(self.linhas)} linha(s) já desenhada(s): elas "
                "ficariam largas ao lado das novas. A forma escolhe-se antes do primeiro trabalho"
            )
        self.linhas_compactas = True

    def _acerta_as_linhas(self, trabalhos: dict[str, str]) -> None:
        """Uma `LinhaDoTrabalho` por chave, na ordem das chaves: cria as novas, tira as que saíram.

        As linhas que ficam são as mesmas, e não recriadas a cada resposta: `mostra` corre a cada
        leitura, e recriar o botão de parar debaixo do ponteiro faria o clique cair num widget que
        já não existe.
        """
        for chave in [c for c in self.linhas if c not in trabalhos]:
            linha = self.linhas.pop(chave)
            self._pilha_das_linhas.removeWidget(linha)
            linha.setParent(None)
            linha.deleteLater()
        for posicao, chave in enumerate(sorted(trabalhos)):
            if chave not in self.linhas:
                linha = LinhaDoTrabalho(chave, self.lista, compacta=self.linhas_compactas)
                linha.parar.connect(self.cancelar)
                self.linhas[chave] = linha
                self._pilha_das_linhas.insertWidget(posicao, linha)

    def _rele_sozinha(self) -> None:
        """O disparo do relógio: pede a leitura só se a janela está à vista.

        Janela escondida não tem leitor: uma janela fechada, ou ainda não mostrada, não pede nada,
        e o relógio não se rearma, porque quem o arma é `mostra`. Minimizar não esconde: um widget
        minimizado continua `isVisible()`. O caso que importa é o fechamento: sem a guarda, o
        relógio de uma janela fechada e ainda não destruída continuaria a pedir leituras à piscina
        durante o encerramento.

        A pergunta é sobre a janela e não sobre o painel: a faixa vive na barra lateral e nasce
        recolhida, e uma guarda sobre o painel pararia a releitura com a janela aberta, congelando
        o selo do cabeçalho. Com o painel solto, como nos testes, `window()` é o próprio painel.
        """
        if not self.window().isVisible():
            return
        self.atualizar.emit()

    def pede_cancelamento(self) -> None:
        """O clique de parar. Sem trabalho selecionado, diz por quê em vez de voltar em silêncio.

        O rótulo da recusa é mostrado porque fica escondido quando a lista está vazia, que é o
        único caso sem seleção. A chave vem do dado do item, porque o texto é o nome legível.
        """
        chave = self.seletor_de_trabalho.currentData()
        if not isinstance(chave, str) or not chave:
            self.rotulo_da_lista.setText(SEM_TRABALHO_A_PARAR)
            self.rotulo_da_lista.setVisible(True)
            return
        self.cancelar.emit(chave)

    def mostra_cancelamento(self, chave: str, codigo: int, corpo: Any) -> str:
        """A resposta do pedido de parada. Anota o estado e devolve o rótulo.

        `409` não é falha a mostrar como erro: significa que o trabalho já terminou, e a ação é
        reclamar o resultado. Tratá-lo pelo ramo genérico escreveria 'falhou' sobre um trabalho que
        concluiu bem.
        """
        if codigo == 202:
            # A frase não promete prazo: ela ramifica só pelo código HTTP e não sabe qual rota está
            # a correr, e quanto a parada demora depende da rota e do paralelismo (com `n_jobs=-1`,
            # a checagem só deixa de enviar trabalho novo aos workers, e o que já corre termina). O
            # que ela pode afirmar é que o pedido foi aceito; o estado passa a `cancelado` quando a
            # parada pega. A barra fica como estava: o trabalho corre até o ponto de checagem.
            anterior = self._leituras.get(chave, _REPOUSO_DA_LINHA)
            leitura = _Leitura(
                PARANDO,
                "aviso",
                _PRAZO_DA_PARADA,
                anterior.maximo,
                anterior.valor,
                anterior.de_etapas,
            )
        elif codigo == 409:
            leitura = self._leituras.get(chave, _REPOUSO_DA_LINHA).desfecho(
                _JA_TERMINARA, "neutro", "não havia o que parar"
            )
        elif codigo == 404:
            leitura = _Leitura(_DESCONHECIDO, "aviso", _PORQUE_DESCONHECIDO)
        else:
            detalhe = corpo.get("detail") if isinstance(corpo, dict) else None
            anterior = self._leituras.get(chave, _REPOUSO_DA_LINHA)
            leitura = _Leitura(
                _NAO_PAROU,
                "erro",
                str(detalhe) if detalhe else f"código {codigo}",
                anterior.maximo,
                anterior.valor,
                anterior.de_etapas,
            )
        self._leituras[chave] = leitura
        # `409` e `404` são fim; o `202` do pedido de parada não é: o trabalho só para no próximo
        # ponto de checagem, e é a releitura que o vê passar a `410`.
        if codigo in (404, 409):
            self._terminados.add(chave)
        else:
            self._terminados.discard(chave)
        return leitura.frase

    def mostra_estado(self, chave: str, codigo: int, corpo: Any) -> str:
        """Anota o estado de um trabalho, do que a leitura devolveu. Devolve o rótulo do estado.

        O estado vem do código HTTP e não de `corpo["terminou"]`: a rota serve `ProgressoResposta`
        só enquanto o trabalho corre, então `terminou=True` nunca chega ao cliente.

        - `202`: o `estado` do corpo, com a etapa em curso e a barra (`feito/total`; sem `total`, a
          barra fica indeterminada);
        - `200`: concluído;
        - `404`: identificador desconhecido, que não é falha do trabalho;
        - `410`: parado a pedido;
        - outro: falhou, com o `detail` quando há.

        O rótulo devolvido é a frase da linha: estado, contagem e nota.
        """
        anterior = self._leituras.get(chave, _REPOUSO_DA_LINHA)
        if codigo == 200:
            leitura = anterior.desfecho(
                ROTULOS_DOS_ESTADOS["concluido"], TONS_DOS_ESTADOS["concluido"], cheia=True
            )
        elif codigo == 404:
            leitura = _Leitura(_DESCONHECIDO, "aviso", _PORQUE_DESCONHECIDO)
        elif codigo == 410:
            # `410` é o cancelamento a pedido, nunca 'falhou': o ramo genérico escreveria 'falhou'
            # sobre a mensagem que diz que o trabalho foi cancelado a seu pedido.
            leitura = anterior.desfecho(
                ROTULOS_DOS_ESTADOS["cancelado"], TONS_DOS_ESTADOS["cancelado"]
            )
        elif codigo == 202 and isinstance(corpo, dict):
            leitura = _leitura_em_curso(corpo, anterior)
        else:
            detalhe = corpo.get("detail") if isinstance(corpo, dict) else None
            leitura = anterior.desfecho(
                ROTULOS_DOS_ESTADOS["falhou"],
                TONS_DOS_ESTADOS["falhou"],
                str(detalhe) if detalhe else f"código {codigo}",
            )
        self._leituras[chave] = leitura
        # O fim vem do código, e do `estado` só pelo conjunto `TERMINAIS`, nunca por um literal
        # repetido aqui, onde um estado novo seria lido como 'ainda corre' por omissão.
        terminou = codigo != 202 or (
            isinstance(corpo, dict) and str(corpo.get("estado", "")) in TERMINAIS
        )
        if terminou:
            self._terminados.add(chave)
        else:
            self._terminados.discard(chave)
        return leitura.frase

    def estados_conhecidos(self) -> dict[str, str]:
        """O último estado lido de cada trabalho, em cópia, como `trabalhos_conhecidos`."""
        return {chave: leitura.frase for chave, leitura in self._leituras.items()}

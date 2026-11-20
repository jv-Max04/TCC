"""Painel 1: os seus arquivos, escolhidos uma vez.

Os bytes de cada arquivo são lidos uma vez, na escolha, e as chamadas seguintes reusam os mesmos
bytes. O seletor entra por injeção: `QFileDialog` abre uma janela modal que um teste não pode
fechar, e com o seletor injetado o duplo do teste conta as invocações.

O modo do estudo é rótulo aberto. `fluxo.MODOS_SUGERIDOS` é sugestão e não domínio, e o combo é
editável: o núcleo aceita qualquer `mode: str`, e um combo fechado deixaria de fora um estudo de
lipidômica ou de outra coluna. O limiar de RSD é procurado pelo arquivo e não pelo rótulo, então
escolher `pos_hilic` para um estudo novo dá o limiar de reserva, e não o calibrado daquele modo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.janela.aparencia import FiltroDaRoda, sem_roda_sem_foco
from refino.desktop.janela.componentes import GradeResponsiva, Recolhivel, Selo
from refino.desktop.janela.modo import ModoSegmentado
from refino.desktop.janela.nomes import NOMES_DAS_OPCOES
from refino.desktop.janela.tema import marca_preenchido, pasta_de_recursos, poe_icone
from refino.interface.fluxo import (
    MODOS_SUGERIDOS,
    ORIGEM_ESCOLHIDA_A_MAO,
    ORIGEM_NO_NOME_DO_ARQUIVO,
    ORIGEM_NOS_NOMES_DAS_AMOSTRAS,
    equivalencia_do_eixo,
    modo_do_nome_do_arquivo,
    prosa_do_modo,
)
from refino.io.validators import deduce_mode_from_sample_names

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from pathlib import Path

__all__ = [
    "CAMPOS_DOS_ARQUIVOS",
    "FORMATO_DOS_FATORES",
    "MODO_AUSENTE",
    "NOME_DO_EXEMPLO",
    "ROTULO_DAS_MEDIDAS",
    "ROTULO_DA_LISTA",
    "ROTULO_DOS_FATORES",
    "ROTULO_DO_BOTAO",
    "ROTULO_DO_EXEMPLO",
    "SEM_ARQUIVO",
    "SEM_LISTA",
    "SEM_MEDIDAS",
    "SEM_PAR",
    "TEXTO_DOS_FATORES",
    "TIPO_DO_ARQUIVO",
    "Escolhedor",
    "ModoAusenteError",
    "PainelDeArquivos",
    "ParDeArquivos",
    "Salvador",
    "escolhedor_por_dialogo",
    "salvador_por_dialogo",
]

TIPO_DO_ARQUIVO: Final = "text/csv"

# A recusa quando não há modo diz onde escolhê-lo: o campo fica em 'Opções avançadas', recolhido,
# e uma recusa que apontasse para um campo que ninguém vê não ajudaria.
MODO_AUSENTE: Final = (
    "Falta o modo do estudo: ele não foi deduzido do nome do arquivo nem dos nomes das amostras. "
    "Escolha-o em «Opções avançadas», em «Preparar o estudo»."
)


class ModoAusenteError(ValueError):
    """Não há modo a enviar. `ValueError` porque o funil da janela já trata essa família."""


# Texto inicial distinto de qualquer nome de arquivo: sem ele, 'o painel não tem par' e 'o painel
# tem um par sem nome' seriam o mesmo estado.
SEM_PAR: Final = "Nenhum par escolhido"

# Cada metade tem texto próprio: com dois botões há um estado intermédio, 'escolhi as medidas e
# falta a lista', e um rótulo que dissesse `SEM_PAR` nele mandaria a pessoa reescolher o arquivo
# que já escolheu.
SEM_MEDIDAS: Final = "measurements: nenhum escolhido"
SEM_LISTA: Final = "Sample_List: nenhum escolhido"

# Os rótulos dos botões são constantes porque o mesmo texto titula a janela do `QFileDialog`: quem
# clica em 'Escolher measurements' lê o mesmo na barra de título e sabe qual dos dois arquivos
# está a escolher.
ROTULO_DAS_MEDIDAS: Final = "Escolher measurements"
ROTULO_DA_LISTA: Final = "Escolher Sample_List"
# O nome que o laboratório pediu; ele nomeia o botão e titula o diálogo, pela razão acima.
ROTULO_DOS_FATORES: Final = "Importar fatores de normalização (opcional)"

# O texto do laboratório. Ele diz quando usar; `FORMATO_DOS_FATORES` diz como o arquivo tem de ser.
TEXTO_DOS_FATORES: Final = (
    "Utilize esta opção quando for necessário normalizar as intensidades dos metabólitos por uma "
    "característica específica de cada amostra, como massa do tecido, volume, quantidade de "
    "proteínas ou número de células."
)
# O valor da opção pelo nome que o seletor mostra, e não `supplied_factors`, que a tela não mostra.
# Derivado de `NOMES_DAS_OPCOES`, o mapa que o seletor desenha, para os dois não divergirem.
_FATORES_NO_SELETOR: Final = NOMES_DAS_OPCOES["supplied_factors"]
FORMATO_DOS_FATORES: Final = (
    "Duas colunas: o nome de cada amostra, como está no arquivo de medições, e o fator dela. Toda "
    "amostra precisa de uma linha, os QCs incluídos. Só age com «Normalização por amostra» em "
    f"«{_FATORES_NO_SELETOR}»."
)

# O arquivo de exemplo viaja com o aplicativo, em `recursos/`, que o `.spec` leva inteira.
ROTULO_DO_EXEMPLO: Final = "Salvar um arquivo de exemplo…"
NOME_DO_EXEMPLO: Final = "exemplo-fatores-de-normalizacao.csv"

# Os três campos do formulário: o rótulo de cada arquivo e a ajuda dele, ditos uma vez. As medidas
# são as intensidades; a lista traz só a ordem de injeção e o nome de cada amostra
# (`read_sample_list` lê duas colunas), e a ordem temporal só existe nela. O tipo e o grupo de cada
# amostra vêm da linha `LABEL` das medições. A ajuda dos fatores é montada no construtor.
#
# O nome do terceiro campo, à parte porque o bloco dos fatores o lê fora do laço dos cartões.
ROTULO_DO_CAMPO_DOS_FATORES: Final = "Fatores de normalização"
CAMPOS_DOS_ARQUIVOS: Final[dict[str, tuple[str, str]]] = {
    "medidas": ("Medições", "A tabela de intensidades de cada feature em cada amostra."),
    "lista": (
        "Lista de amostras",
        "A ordem de injeção e o nome de cada amostra. Quem é QC, e o grupo de cada amostra, vêm do "
        "arquivo de medições.",
    ),
    "fatores": (ROTULO_DO_CAMPO_DOS_FATORES, ""),
}

# O que o campo mostra sem arquivo, distinto de qualquer nome de arquivo pela razão de `SEM_PAR`.
SEM_ARQUIVO: Final = "Nenhum arquivo selecionado"

# O texto dos três botões. Curto, porque o título do cartão já diz qual arquivo; os rótulos longos
# (`ROTULO_DAS_MEDIDAS` e irmãos) continuam a titular o diálogo de escolha.
ROTULO_DO_BOTAO: Final = "Selecionar arquivo"


@dataclass(frozen=True)
class ParDeArquivos:
    """Os bytes escolhidos, com o nome que o servidor vai ver.

    O modo não fica aqui: os bytes são escolhidos uma vez e congelam, e o modo é escolha viva. Com
    o modo guardado no par, corrigir o combo depois de escolher os arquivos enviaria o modo antigo
    com o novo na tela, e `mode` decide `check_mode_matches_content`, os limiares por modo e o nome
    do artefato.

    `frozen=True` com os bytes dentro, e não o caminho: guardar `Path` faria cada chamada de rota
    ler o disco de novo, e um arquivo removido, substituído ou numa rede que caiu faria a segunda
    chamada enviar conteúdo diferente da primeira sem nada acusar.
    """

    nome_das_medidas: str
    bytes_das_medidas: bytes
    nome_da_lista: str
    bytes_da_lista: bytes


class Escolhedor(Protocol):
    """O diálogo de arquivo; `Protocol` e não classe, pelo mesmo motivo de `Transporte`.

    Devolve um caminho, ou `None` quando o usuário cancelou. `None` é desfecho de primeira classe:
    cancelar não é erro, e um painel que tratasse os dois como o mesmo estado exibiria falha onde
    não houve.

    Um arquivo por chamada, e o `titulo` diz qual. Com um seletor de vários arquivos, a ordem dos
    caminhos é a do sistema de arquivos e não a dos cliques, e qual deles virava `measurements`
    dependeria da sorte. O `titulo` entra por parâmetro, e não fixado pela fábrica, para que haja um
    só escolhedor a injetar e o duplo do teste possa afirmar a sequência de títulos pedidos.
    """

    def __call__(self, titulo: str) -> Path | None: ...


def escolhedor_por_dialogo(pai: QWidget | None = None) -> Escolhedor:
    """O escolhedor de produção: `QFileDialog.getOpenFileName`, um arquivo por chamada.

    O cancelamento chega como string vazia e vira `None`, o contrato do `Escolhedor`. É
    `getOpenFileName` no singular, que devolve string; a versão no plural devolve lista e pediria
    outra guarda. Cada chamada sabe qual dos dois arquivos está a pedir, porque o botão que a
    disparou o diz no `titulo`.
    """
    # `Path` é importado aqui porque no topo ele está sob `TYPE_CHECKING` e não existe em runtime.
    from pathlib import Path as _Path

    from PySide6.QtWidgets import QFileDialog

    def escolhe(titulo: str) -> Path | None:
        caminho, _filtro = QFileDialog.getOpenFileName(pai, titulo, "", "CSV (*.csv)")
        if not caminho:
            return None
        return _Path(caminho)

    return escolhe


def _veste_o_botao_de_arquivo(botao: QPushButton) -> None:
    """O botão de escolher arquivo, com o ícone de envio: um molde para os três."""
    botao.setObjectName("seleciona_arquivo")
    botao.setCursor(Qt.CursorShape.PointingHandCursor)
    # A cor vem do token do tema, para acompanhar a troca de esquema.
    poe_icone(botao, '<path d="M12 16V3m-4 4 4-4 4 4M4 15v5h16v-5"/>', "acao_suave", 14)
    botao.setIconSize(QSize(14, 14))


class Salvador(Protocol):
    """O diálogo de salvar: recebe o nome sugerido e devolve o destino, ou `None` se cancelou.

    É a assinatura de `figuras.EscolhedorDeArquivo`, e a janela injeta o mesmo objeto nos dois: um
    duplo de teste serve aos dois botões que salvam. Declarado aqui, e não importado de
    `figuras.py`, para o painel dos arquivos não carregar o `matplotlib` só por um tipo.
    """

    def __call__(self, nome_sugerido: str) -> Path | None: ...


def salvador_por_dialogo(
    pai: QWidget | None = None,
    *,
    titulo: str = ROTULO_DO_EXEMPLO.rstrip("…"),
    filtro: str = "CSV (*.csv)",
) -> Salvador:
    """O salvador de produção: `QFileDialog.getSaveFileName`, com filtro de CSV por padrão.

    String vazia é o cancelamento e vira `None`: `Path("")` é o diretório corrente, e deixá-la
    passar gravaria num caminho que ninguém escolheu. O título e o filtro são parâmetros porque a
    tabela de normalidade (`qualidade.py`) e o resumo metodológico (`execucao.py`) usam o mesmo
    corpo; o título é o do botão que abriu o diálogo.
    """
    from pathlib import Path as _Path

    from PySide6.QtWidgets import QFileDialog

    def salva(nome_sugerido: str) -> Path | None:
        caminho, _filtro = QFileDialog.getSaveFileName(pai, titulo, nome_sugerido, filtro)
        return _Path(caminho) if caminho else None

    return salva


class PainelDeArquivos(QWidget):
    """O par e o modo, escolhidos uma vez e reusados.

    `par_escolhido` é o sinal que os painéis seguintes esperam. Ele carrega o `ParDeArquivos`,
    e não um aviso de que algo mudou: quem o recebe não tem de ir buscar o estado a este
    painel.

    `metade_escolhida` diz que uma das duas metades foi guardada, com a chave dela, mesmo sem
    completar o par: o estado vazio das abas de etapa diz qual arquivo falta, e com uma metade só
    `par_escolhido` não emite.
    """

    par_escolhido = Signal(object)
    arquivo_removido = Signal(str)
    metade_escolhida = Signal(str)

    def __init__(
        self,
        escolhedor: Escolhedor,
        parent: QWidget | None = None,
        *,
        salvador: Salvador | None = None,
    ) -> None:
        super().__init__(parent)
        self._escolhedor = escolhedor
        # O diálogo de salvar entra por injeção, como `escolhedor`, e o de produção nasce no clique:
        # um `QFileDialog` real é modal, e um teste que não o substitua fica pendurado.
        self._salvador = salvador
        # O terceiro arquivo vive fora do par: `None` até alguém o escolher, e nunca obrigatório.
        self._fatores: tuple[str, bytes] | None = None
        # As duas metades vivem separadas, e o par é derivado delas. Guardar só o `ParDeArquivos`
        # obrigaria a metade escolhida a caber num objeto que exige as quatro colunas, e a saída
        # seria preencher a que falta com string vazia: um par que existe e mente.
        self._medidas: tuple[str, bytes] | None = None
        self._lista: tuple[str, bytes] | None = None
        self._par: ParDeArquivos | None = None

        # Os cartões dos arquivos usam a `GradeResponsiva`; o modo fica num bloco recolhido abaixo
        # deles. O combo controla o valor, e os segmentos são atalhos visuais.
        disposicao = QVBoxLayout(self)
        disposicao.setContentsMargins(0, 0, 0, 0)
        disposicao.setSpacing(24)

        # Um filtro da roda por painel, com pai, e os campos roláveis passam por
        # `sem_roda_sem_foco`. Sem pai ele seria coletado e deixaria de agir sem nada acusar.
        self._filtro_da_roda = FiltroDaRoda(self)

        self.modo = sem_roda_sem_foco(QComboBox(self), self._filtro_da_roda)
        self.modo.addItems(list(MODOS_SUGERIDOS))
        # Editável porque a lista é sugestão: ver o docstring do módulo.
        self.modo.setEditable(True)
        self.modo.setMinimumWidth(220)
        # Nasce vazio, e não no primeiro da lista, que seria um padrão silencioso: um estudo
        # `neg_c18` sem modo escolhido sairia rotulado `pos_hilic`. O valor vem da dedução, com a
        # origem à vista, ou não vem, e a chamada é recusada.
        self.modo.setCurrentIndex(-1)
        self.modo.setEditText("")
        self._origem_do_modo: str | None = None
        self._deduzindo = False
        # O rótulo acompanha o combo, senão ele afirma um modo e o payload envia outro.
        self.modo.currentTextChanged.connect(self._modo_mudou)
        linha_do_modo = QHBoxLayout()
        linha_do_modo.setSpacing(16)
        titulo_do_modo = QLabel("Modo do estudo", self)
        titulo_do_modo.setObjectName("rotulo_do_modo")
        titulo_do_modo.setFixedWidth(176)
        # O modo rotula as saídas e confere os nomes das amostras; o limiar vem do conteúdo do
        # arquivo, e não do modo.
        dica_do_modo = (
            "O nome da corrida — a ionização e a coluna, como pos_hilic. Ele é deduzido do nome do "
            "arquivo ou dos nomes das amostras; aqui você o troca. Ele nomeia os arquivos de saída "
            "e é conferido contra os nomes das amostras. Pressione F2 ou dê dois cliques no modo "
            "selecionado para digitar outro."
        )
        titulo_do_modo.setToolTip(dica_do_modo)
        linha_do_modo.addWidget(titulo_do_modo)
        self.modo_segmentado = ModoSegmentado(self.modo, self)
        self.modo_segmentado.setFixedWidth(332)
        self.segmentos_do_modo = self.modo_segmentado.botoes
        linha_do_modo.addWidget(self.modo_segmentado)
        linha_do_modo.addStretch(1)
        # O seletor sai da vista, não da ferramenta: o modo é deduzido, e o controle fica num bloco
        # fechado, abaixo dos cartões, para quem precisa de o trocar.
        self.opcoes_avancadas = Recolhivel("Opções avançadas", self)
        self.opcoes_avancadas.setProperty("referencia", True)
        corpo_das_opcoes = QVBoxLayout(self.opcoes_avancadas.corpo)
        corpo_das_opcoes.setContentsMargins(0, 8, 0, 0)
        corpo_das_opcoes.addLayout(linha_do_modo)

        # O campo de cada arquivo: o nome do que foi escolhido, ou `SEM_ARQUIVO`. Fica verde quando
        # preenchido (`marca_preenchido`), e é ele que o arnês lê para saber o estado de cada um.
        self.estados: dict[str, QLabel] = {}
        for chave in CAMPOS_DOS_ARQUIVOS:
            campo = QLabel(SEM_ARQUIVO, self)
            campo.setObjectName("campo_de_arquivo")
            campo.setWordWrap(True)
            campo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            campo.setVisible(False)
            self.estados[chave] = campo

        # Dois botões, um por arquivo, para que a ordem não dependa do sistema de arquivos (ver
        # `Escolhedor`). Cada botão tem o seu slot nomeado, e não um `lambda` com bandeira:
        # `clicked` entrega um `bool` ao slot, e um slot com parâmetro posicional o receberia como
        # se fosse o título.
        self.botao_medidas = QPushButton(ROTULO_DO_BOTAO, self)
        self.botao_medidas.setToolTip(ROTULO_DAS_MEDIDAS)
        self.botao_medidas.clicked.connect(self.escolhe_medidas)

        self.botao_lista = QPushButton(ROTULO_DO_BOTAO, self)
        self.botao_lista.setToolTip(ROTULO_DA_LISTA)
        self.botao_lista.clicked.connect(self.escolhe_lista)

        # O terceiro arquivo não faz parte do par: os dois de cima são obrigatórios e viajam juntos
        # como `ParDeArquivos`; este é opcional e só o método `supplied_factors` o consome. Por isso
        # não dispara o sinal, e a ausência dele nunca bloqueia uma chamada; escolher aquele método
        # sem ele é recusado pela rota. O botão mora na página 'Normalização' (ver
        # `importacao_dos_fatores`), mas o estado continua neste painel.
        self.botao_fatores = QPushButton(ROTULO_DOS_FATORES, self)
        self.botao_fatores.clicked.connect(self.escolhe_fatores)
        # A explicação dos fatores vai no ⓘ da linha e na dica do botão, para quem já usa o
        # MetaboAnalyst entender para que serve. É montada dos campos `grupo` e `rotulo`, e não da
        # prosa de `prosa_da_equivalencia`, porque a dica aceita HTML e não markdown.
        _fatores_no_metabo = equivalencia_do_eixo("sample_normalization", "supplied_factors")
        ajuda_dos_fatores = (
            "Opcional. Duas colunas: nome da amostra e o fator pelo qual dividir a linha dela — "
            "peso seco, volume de urina, número de células, medidos FORA do equipamento.\n\n"
            f"Só age se «Normalização por amostra» estiver em «{_FATORES_NO_SELETOR}». E precisa "
            "de fator para TODAS as amostras, QCs incluídos: normalizar as amostras sem normalizar "
            "os QCs faz o antes-e-depois deixar de ser sobre a mesma escala."
        )
        # Guarda e não `assert`, porque `assert` some sob `-O` e isto desenha a tela. Se a
        # contraparte sair da constante, a linha perde a nota e continua a funcionar.
        nota_dos_fatores: QLabel | None = None
        if _fatores_no_metabo is not None:  # pragma: no branch - a constante declara a contraparte
            ajuda_dos_fatores += (
                f"\n\nNo MetaboAnalyst é o {_fatores_no_metabo.grupo} → "
                f"{_fatores_no_metabo.rotulo}."
            )
            # O nome do MetaboAnalyst fica à vista, e não só na dica: dica exige passar o rato, e
            # quem já usa a plataforma reconhece o termo como ela o escreve.
            self.rotulo_do_equivalente = QLabel(self)
            self.rotulo_do_equivalente.setTextFormat(Qt.TextFormat.MarkdownText)
            self.rotulo_do_equivalente.setText(f"No MetaboAnalyst: `{_fatores_no_metabo.rotulo}`")
            nota_dos_fatores = self.rotulo_do_equivalente
        self.botao_fatores.setToolTip(ajuda_dos_fatores)

        botoes = {"medidas": self.botao_medidas, "lista": self.botao_lista}
        self.botoes_de_remocao: dict[str, QPushButton] = {}
        # Cada arquivo obrigatório ocupa o seu cartão; os fatores ficam na página 'Normalização'.
        # A composição muda sem tocar a leitura dos bytes nem o contrato do par.
        cartoes: list[QWidget] = []
        for chave in ("medidas", "lista"):
            rotulo, ajuda = CAMPOS_DOS_ARQUIVOS[chave]
            cartao = QFrame(self)
            cartao.setObjectName("cartao_de_arquivo")
            cartao.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            cartao.setMinimumWidth(220)
            cartao.setMinimumHeight(150)
            coluna = QVBoxLayout(cartao)
            coluna.setContentsMargins(20, 20, 20, 20)
            coluna.setSpacing(0)

            topo = QHBoxLayout()
            figura = QLabel(cartao)
            figura.setObjectName("icone_do_arquivo")
            figura.setAlignment(Qt.AlignmentFlag.AlignCenter)
            figura.setFixedSize(20, 20)
            poe_icone(
                figura,
                '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>'
                '<path d="M14 3v6h6M8 13h8v6H8zM8 16h8M12 13v6"/>',
                "icone_do_arquivo",
                20,
            )
            topo.addWidget(figura)
            topo.addStretch(1)
            obrigatoriedade = QLabel("Obrigatório", cartao)
            obrigatoriedade.setObjectName("obrigatoriedade_do_arquivo")
            obrigatoriedade.setProperty("obrigatorio", True)
            topo.addWidget(obrigatoriedade, 0, Qt.AlignmentFlag.AlignTop)
            coluna.addLayout(topo)
            coluna.addSpacing(12)

            titulo = QLabel(rotulo, cartao)
            titulo.setObjectName("titulo_do_arquivo")
            titulo.setWordWrap(True)
            titulo.setMinimumHeight(24)
            coluna.addWidget(titulo)
            coluna.addSpacing(4)
            descricao = QLabel(
                "Matriz de intensidades (CSV/TSV)"
                if chave == "medidas"
                # `read_sample_list` lê duas colunas, a ordem e o nome: o grupo vem da linha
                # `LABEL` das medições, e os lotes declaram-se em 'Correção'.
                else "Ordem de injeção e nome de cada amostra",
                cartao,
            )
            descricao.setObjectName("descricao_do_arquivo")
            descricao.setWordWrap(True)
            descricao.setMinimumHeight(16)
            descricao.setToolTip(ajuda)
            coluna.addWidget(descricao)
            coluna.addStretch(1)
            coluna.addWidget(self.estados[chave])
            coluna.addSpacing(16)

            rodape = QHBoxLayout()
            rodape.setContentsMargins(0, 0, 0, 0)
            _veste_o_botao_de_arquivo(botoes[chave])
            botoes[chave].setToolTip(ajuda)
            rodape.addWidget(botoes[chave], 0, Qt.AlignmentFlag.AlignLeft)
            rodape.addStretch(1)
            remover = self._botao_de_remocao(chave, rotulo, cartao)
            rodape.addWidget(remover, 0, Qt.AlignmentFlag.AlignRight)
            coluna.addLayout(rodape)
            cartoes.append(cartao)
        self.grade_dos_arquivos = GradeResponsiva(cartoes, 2, self, espaco=12)
        disposicao.addWidget(self.grade_dos_arquivos)

        # O modo à vista, com a origem, que torna a dedução auditável: quem lê 'pos_c18 (do nome do
        # arquivo)' sabe o que conferir.
        self.rotulo_do_modo = QLabel(self)
        self.rotulo_do_modo.setObjectName("modo_deduzido")
        self.rotulo_do_modo.setWordWrap(True)
        disposicao.addWidget(self.rotulo_do_modo)
        disposicao.addWidget(self.opcoes_avancadas)
        self._mostra_o_modo()

        self.importacao_dos_fatores = self._constroi_importacao_dos_fatores(nota_dos_fatores)

        # O estado do par numa palavra, para o cabeçalho da seção, onde a página (`AbaDePreparo`) o
        # põe. Sem pai: ele é do layout em que a página o puser.
        self.selo = Selo("0 / 2 obrigatórios", "neutro")
        self.selo.setProperty("contagem_arquivos", True)

        # O rótulo do par é por onde a falha de leitura chega à tela ('não foi possível ler X'),
        # como nota por baixo do formulário.
        self.rotulo = QLabel(SEM_PAR, self)
        self.rotulo.setObjectName("nota_do_campo")
        self.rotulo.setWordWrap(True)
        # Só aparece com erro de leitura, a única informação que só ele carrega: o estado do par já
        # está no selo e nos campos.
        self.rotulo.setVisible(False)
        disposicao.addWidget(self.rotulo)

        # A folga vai para o fim; sem esta linha o `QVBoxLayout` a reparte entre os widgets, e os
        # rótulos flutuam no meio da tela.
        disposicao.addStretch(1)

    def _botao_de_remocao(self, chave: str, rotulo: str, pai: QWidget) -> QPushButton:
        """O botão 'Remover' de um arquivo, escondido até haver o que remover."""
        remover = QPushButton("Remover", pai)
        remover.setObjectName("remove_arquivo")
        poe_icone(
            remover,
            '<path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/>',
            "acao_suave",
            14,
        )
        remover.setIconSize(QSize(14, 14))
        remover.setCursor(Qt.CursorShape.PointingHandCursor)
        remover.setToolTip(f"Remover {rotulo.lower()} deste estudo")
        remover.setAccessibleName(f"Remover {rotulo.lower()}")
        remover.clicked.connect(lambda _marcado=False, arquivo=chave: self.remover_arquivo(arquivo))
        remover.hide()
        self.botoes_de_remocao[chave] = remover
        return remover

    def _constroi_importacao_dos_fatores(self, nota_do_metabo: QLabel | None) -> QFrame:
        """O bloco dos fatores que a janela põe na página 'Normalização'.

        Construído aqui e mostrado lá, porque o estado é deste painel: `_fatores`,
        `arquivos_da_execucao` e o 'Remover' continuam os mesmos, e só a moldura muda de página.
        Nasce sem pai: quem o põe num layout é a janela, e um filho deste painel fora de qualquer
        layout seria desenhado por cima dos cartões.

        A ordem é a do pedido do laboratório: o botão, o texto dele, o termo do MetaboAnalyst, o
        formato do arquivo, e o exemplo que se salva da própria janela.
        """
        bloco = QFrame()
        bloco.setObjectName("importacao_dos_fatores")
        bloco.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        coluna = QVBoxLayout(bloco)
        coluna.setContentsMargins(0, 12, 0, 0)
        coluna.setSpacing(8)

        linha = QHBoxLayout()
        linha.setSpacing(12)
        _veste_o_botao_de_arquivo(self.botao_fatores)
        linha.addWidget(self.botao_fatores, 0, Qt.AlignmentFlag.AlignLeft)
        self.botao_do_exemplo = QPushButton(ROTULO_DO_EXEMPLO, bloco)
        self.botao_do_exemplo.setObjectName("salva_exemplo")
        self.botao_do_exemplo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.botao_do_exemplo.setToolTip(
            "Grava uma cópia do modelo onde você escolher. Troque os nomes pelos das suas "
            "amostras e os valores pelos fatores medidos."
        )
        self.botao_do_exemplo.clicked.connect(self.salva_exemplo_dos_fatores)
        linha.addWidget(self.botao_do_exemplo, 0, Qt.AlignmentFlag.AlignLeft)
        linha.addStretch(1)
        linha.addWidget(
            self._botao_de_remocao("fatores", ROTULO_DO_CAMPO_DOS_FATORES, bloco),
            0,
            Qt.AlignmentFlag.AlignRight,
        )
        coluna.addLayout(linha)

        self.texto_dos_fatores = QLabel(TEXTO_DOS_FATORES, bloco)
        self.texto_dos_fatores.setObjectName("texto_dos_fatores")
        self.texto_dos_fatores.setWordWrap(True)
        coluna.addWidget(self.texto_dos_fatores)
        if nota_do_metabo is not None:  # pragma: no branch - ver a guarda no construtor
            nota_do_metabo.setObjectName("descricao_do_arquivo")
            nota_do_metabo.setWordWrap(True)
            coluna.addWidget(nota_do_metabo)
        formato = QLabel(FORMATO_DOS_FATORES, bloco)
        formato.setObjectName("descricao_do_arquivo")
        formato.setWordWrap(True)
        coluna.addWidget(formato)
        coluna.addWidget(self.estados["fatores"])

        # A falha de leitura e a confirmação do exemplo têm linha própria aqui, e não no rótulo do
        # par, que mora em 'Preparar o estudo': a frase sairia na aba que ninguém está a ver.
        self.aviso_dos_fatores = QLabel(bloco)
        self.aviso_dos_fatores.setObjectName("nota_do_campo")
        self.aviso_dos_fatores.setWordWrap(True)
        self.aviso_dos_fatores.setVisible(False)
        coluna.addWidget(self.aviso_dos_fatores)
        # A altura deste bloco é protegida em `AbaDeEtapa`, que é quem prende a folha da página de
        # etapa ao `sizeHint`; uma guarda aqui não basta, porque o lugar dele no formulário
        # continuaria espremido.
        return bloco

    def salva_exemplo_dos_fatores(self) -> Path | None:
        """Grava o arquivo de exemplo onde a pessoa escolher, byte a byte igual ao que viaja.

        Devolve o destino, ou `None` se ela cancelou ou a escrita falhou; a frase diz qual dos dois,
        porque as ações são opostas: quem cancelou não precisa de nada, e quem não conseguiu gravar
        precisa de escolher outro lugar.
        """
        salvador = self._salvador or salvador_por_dialogo(self.window())
        destino = salvador(NOME_DO_EXEMPLO)
        if destino is None:
            return None
        try:
            destino.write_bytes((pasta_de_recursos() / NOME_DO_EXEMPLO).read_bytes())
        except OSError as erro:
            self.aviso_dos_fatores.setText(
                f"Não foi possível salvar o exemplo em {destino}: {type(erro).__name__}"
            )
            self.aviso_dos_fatores.setVisible(True)
            return None
        self.aviso_dos_fatores.setText(f"Exemplo salvo em {destino}")
        self.aviso_dos_fatores.setVisible(True)
        return destino

    # ------------------------------------------------------------------
    # A escolha
    # ------------------------------------------------------------------

    @property
    def par(self) -> ParDeArquivos | None:
        """O par corrente, ou `None` enquanto faltar uma metade.

        Somente leitura: trocá-lo é `escolhe_medidas` ou `escolhe_lista`.
        """
        return self._par

    @property
    def metades_que_faltam(self) -> tuple[str, ...]:
        """As chaves de `CAMPOS_DOS_ARQUIVOS` das metades do par ainda não escolhidas, na ordem
        dos cartões. Vazio exatamente quando há par: `_recompoe` deriva um do outro."""
        return tuple(
            chave
            for chave, metade in (("medidas", self._medidas), ("lista", self._lista))
            if metade is None
        )

    def escolhe_medidas(self) -> ParDeArquivos | None:
        """Pede só o `measurements`. Devolve o par se ele ficou completo, senão o que havia."""
        return self._guarda(ROTULO_DAS_MEDIDAS, medidas=True)

    def escolhe_lista(self) -> ParDeArquivos | None:
        """Pede só o `Sample_List`. Devolve o par se ele ficou completo, senão o que havia."""
        return self._guarda(ROTULO_DA_LISTA, medidas=False)

    def remover_arquivo(self, chave: str) -> None:
        """Desanexa somente a escolha indicada, mantendo o arquivo original no disco.

        O par deixa de existir assim que uma metade é retirada. O sinal próprio
        informa a janela também quando o par fica incompleto, sem alterar o
        contrato de ``par_escolhido`` de emitir apenas pares completos.
        """
        if chave not in CAMPOS_DOS_ARQUIVOS:
            raise ValueError(f"tipo de arquivo desconhecido: {chave!r}")
        if chave == "fatores":
            if self._fatores is None:
                return
            self._fatores = None
            self.estados[chave].setText(SEM_ARQUIVO)
            self.estados[chave].hide()
            marca_preenchido(self.estados[chave], False)
            self.botoes_de_remocao[chave].hide()
            self.rotulo.hide()
        else:
            if chave == "medidas":
                if self._medidas is None:
                    return
                self._medidas = None
            else:
                if self._lista is None:
                    return
                self._lista = None
            self._recompoe()
            self._deduz_o_modo()
            self._atualiza_rotulo()
        self.arquivo_removido.emit(chave)

    def _guarda(self, titulo: str, *, medidas: bool) -> ParDeArquivos | None:
        """Invoca o seletor uma vez para uma metade e guarda os bytes dela.

        Devolve sempre o par, para o chamador poder afirmar o que ficou guardado sem ler o
        atributo. Um corpo para os dois botões, com a metade por bandeira nomeada: duas cópias
        divergiriam no dia em que a guarda de `OSError` mudasse num lado.

        O cancelamento e o `OSError` devolvem o mesmo valor de propósito: para a decisão, 'não
        troque esta metade', os dois são iguais. O que os separa é o rótulo, onde a distinção tem
        leitor.
        """
        caminho = self._escolhedor(titulo)
        if caminho is None:
            # Cancelar não apaga o que havia: quem cancelou não pediu para descartar.
            return self._par

        try:
            lidos = caminho.read_bytes()
        except OSError as erro:
            # O arquivo pode ter sido removido entre a escolha e a leitura, estar sem permissão, ou
            # vir de uma rede que caiu; deixar a exceção escapar do slot seria silêncio. A metade
            # anterior fica, e o nome do arquivo entra na mensagem porque há dois botões.
            self.rotulo.setText(f"Não foi possível ler {caminho.name}: {type(erro).__name__}")
            self.rotulo.setVisible(True)
            return self._par

        metade = (caminho.name, lidos)
        if medidas:
            self._medidas = metade
        else:
            self._lista = metade

        self._recompoe()
        self._deduz_o_modo()
        self._atualiza_rotulo()
        self.metade_escolhida.emit("medidas" if medidas else "lista")
        if self._par is not None:
            # Só emite com o par completo: o sinal carrega um `ParDeArquivos`, e não há par de uma
            # metade. Emitir antes mandaria a janela compor um payload a que falta um arquivo.
            self.par_escolhido.emit(self._par)
        return self._par

    def _recompoe(self) -> None:
        """Deriva o par das duas metades. `None` enquanto faltar uma.

        Recompõe uma vez, em vez de construir um par novo a cada leitura, porque `par` é comparado
        por identidade: um `ParDeArquivos` construído dentro do `property` seria um objeto novo por
        chamada.
        """
        if self._medidas is None or self._lista is None:
            self._par = None
            return
        self._par = ParDeArquivos(
            nome_das_medidas=self._medidas[0],
            bytes_das_medidas=self._medidas[1],
            nome_da_lista=self._lista[0],
            bytes_da_lista=self._lista[1],
        )

    # ------------------------------------------------------------------
    # O que vai para a rota
    # ------------------------------------------------------------------

    def arquivos_para_envio(self) -> dict[str, Any]:
        """O `files` do multipart, na forma que as rotas aceitam.

        Três elementos por entrada (nome, bytes, tipo), com as chaves `measurements` e
        `sample_list`; outra forma seria recusada pela rota com um `422` a culpar o arquivo.

        Levanta quando não há par: devolver `{}` faria a chamada seguir com multipart vazio e a
        rota responder `422`, mandando a pessoa conferir um arquivo que ela nunca escolheu.
        """
        if self._par is None:
            raise ValueError(
                "nenhum par de arquivos escolhido: escolha measurements e Sample_List antes "
                f"de chamar uma rota — {self._o_que_falta()}"
            )
        arquivos: dict[str, Any] = {
            "measurements": (
                self._par.nome_das_medidas,
                self._par.bytes_das_medidas,
                TIPO_DO_ARQUIVO,
            ),
            "sample_list": (
                self._par.nome_da_lista,
                self._par.bytes_da_lista,
                TIPO_DO_ARQUIVO,
            ),
        }
        return arquivos

    def arquivos_da_execucao(self) -> dict[str, Any]:
        """Os dois do par mais o CSV de fatores, quando escolhido.

        Método separado, e não um `if` dentro de `arquivos_para_envio`, porque aquele serve a todas
        as rotas, e a maioria não declara `normalization_factors`: o FastAPI descarta campo não
        declarado em silêncio, e a carga afirmaria uma influência que aquelas rotas não têm. Quem
        chama escolhe; hoje, `/execucoes` e `/matrizes/processar`.
        """
        arquivos = self.arquivos_para_envio()
        if self._fatores is not None:
            arquivos["normalization_factors"] = (
                self._fatores[0],
                self._fatores[1],
                TIPO_DO_ARQUIVO,
            )
        return arquivos

    def escolhe_fatores(self) -> None:
        """Escolhe o CSV de fatores, o terceiro arquivo, que é opcional.

        Não passa por `_guarda` nem emite sinal: aquele corpo mantém o par coerente, e este arquivo
        não é metade de par nenhum; reusá-lo faria a escolha dos fatores disparar o fluxo de leitura
        das rotas. A guarda de `OSError` é a mesma de `_guarda`.
        """
        caminho = self._escolhedor(ROTULO_DOS_FATORES)
        if caminho is None:
            return  # cancelar não apaga o que havia

        try:
            lidos = caminho.read_bytes()
        except OSError as erro:
            # Na linha dos fatores, que mora na página onde o botão foi clicado.
            self.aviso_dos_fatores.setText(
                f"Não foi possível ler {caminho.name}: {type(erro).__name__}"
            )
            self.aviso_dos_fatores.setVisible(True)
            return

        self.aviso_dos_fatores.setVisible(False)
        self._fatores = (caminho.name, lidos)
        self.estados["fatores"].setText(caminho.name)
        self.estados["fatores"].setVisible(True)
        marca_preenchido(self.estados["fatores"], True)
        self.botoes_de_remocao["fatores"].show()

    def campos_para_envio(self) -> dict[str, str]:
        """O `data` do multipart, com o modo lido do widget a cada chamada.

        O modo não vem do par: o combo é a escolha viva da pessoa, e corrigi-lo depois de escolher
        os arquivos tem de mudar o que se envia. Continua a exigir um par: o modo sozinho não é
        chamada.
        """
        if self._par is None:
            raise ValueError(
                f"nenhum par de arquivos escolhido: não há o que enviar — {self._o_que_falta()}"
            )
        modo = self.modo.currentText().strip()
        if not modo:
            # Recusa e abre o campo: ver `MODO_AUSENTE`.
            self.mostra_opcoes_avancadas()
            raise ModoAusenteError(MODO_AUSENTE)
        return {"mode": modo}

    def mostra_opcoes_avancadas(self) -> None:
        """Abre 'Opções avançadas' e põe o foco no seletor do modo, o destino da recusa."""
        self.opcoes_avancadas.alterna.setChecked(True)
        self.modo_segmentado.setFocus(Qt.FocusReason.OtherFocusReason)

    @property
    def origem_do_modo(self) -> str | None:
        """De onde veio o modo corrente: do nome do arquivo, dos nomes das amostras, ou à mão."""
        return self._origem_do_modo

    def _modo_mudou(self, _valor: str) -> None:
        """O combo mudou: se não foi a dedução, foi a pessoa, e a escolha dela passa a valer."""
        if not self._deduzindo:
            self._origem_do_modo = ORIGEM_ESCOLHIDA_A_MAO if self.modo.currentText() else None
        self._mostra_o_modo()
        self._atualiza_rotulo()

    def _deduz_o_modo(self) -> None:
        """O modo dos arquivos escolhidos, se a pessoa não escolheu um à mão.

        A ordem é o nome do arquivo de medições, o da lista, e só então os nomes das amostras no
        cabeçalho das medições. Quando o nome do arquivo e os nomes das amostras discordam, vale o
        nome, e o aviso `MODE_CONTENT_MISMATCH` da validação diz a discordância.
        """
        if self._origem_do_modo == ORIGEM_ESCOLHIDA_A_MAO:
            return
        modo, origem = "", None
        for metade in (self._medidas, self._lista):
            if metade is not None and (do_nome := modo_do_nome_do_arquivo(metade[0])):
                modo, origem = do_nome, ORIGEM_NO_NOME_DO_ARQUIVO
                break
        else:
            if self._medidas is not None:
                dos_nomes = deduce_mode_from_sample_names(_nomes_do_cabecalho(self._medidas[1]))
                if dos_nomes is not None:
                    modo, origem = dos_nomes, ORIGEM_NOS_NOMES_DAS_AMOSTRAS
        self._deduzindo = True
        try:
            self.modo.setEditText(modo)
        finally:
            self._deduzindo = False
        self._origem_do_modo = origem
        self._mostra_o_modo()

    def _mostra_o_modo(self) -> None:
        """A linha do modo: o valor e a origem, ou o que falta, só depois de haver arquivo."""
        if self._medidas is None and self._lista is None and not self.modo.currentText():
            self.rotulo_do_modo.setText(
                "Modo: deduzido do nome do arquivo ou dos nomes das amostras, quando você os "
                "escolher."
            )
            return
        self.rotulo_do_modo.setText(prosa_do_modo(self.modo.currentText(), self._origem_do_modo))

    def _o_que_falta(self) -> str:
        """Qual das duas metades falta; com dois botões, 'falta o par' não diz em qual clicar."""
        # Sem ramo para 'não falta nada': os dois chamadores estão dentro de `if self._par is None`,
        # e `_par is None` vale exatamente quando falta uma metade. Se um chamador novo alcançar o
        # caso, o `' e '.join` de lista vazia produz 'falta ', visível em vez de plausível.
        faltam = [
            nome
            for metade, nome in ((self._medidas, "measurements"), (self._lista, "Sample_List"))
            if metade is None
        ]
        return f"falta {' e '.join(faltam)}"

    def _atualiza_rotulo(self) -> None:
        """O rótulo, com o modo vivo; ligado a `currentTextChanged`, senão ele mentiria.

        Três estados: `SEM_PAR` para quem não escolheu nada; com uma metade, o nome do arquivo que
        já está e o do que falta; com o par, os dois nomes e o modo corrente.
        """
        # Os estados dos dois cartões seguem as metades, com o nome do arquivo de cada uma.
        for chave, metade in (("medidas", self._medidas), ("lista", self._lista)):
            self.estados[chave].setText(metade[0] if metade is not None else SEM_ARQUIVO)
            self.estados[chave].setVisible(metade is not None)
            marca_preenchido(self.estados[chave], metade is not None)
            self.botoes_de_remocao[chave].setVisible(metade is not None)
        self.rotulo.setVisible(False)
        escolhidas = sum(m is not None for m in (self._medidas, self._lista))
        if escolhidas == 2:
            self.selo.define(f"{escolhidas} / 2 obrigatórios", "sucesso")
        else:
            self.selo.define(f"{escolhidas} / 2 obrigatórios", "aviso" if escolhidas else "neutro")
        if self._medidas is None and self._lista is None:
            self.rotulo.setText(SEM_PAR)
            return
        medidas = self._medidas[0] if self._medidas is not None else SEM_MEDIDAS
        lista = self._lista[0] if self._lista is not None else SEM_LISTA
        if self._par is None:
            self.rotulo.setText(f"{medidas}\n{lista}")
            return
        self.rotulo.setText(f"{medidas} + {lista} — modo {self.modo.currentText()}")


def _nomes_do_cabecalho(bruto: bytes) -> list[str]:
    """Os nomes de amostra da primeira linha das medições, só para deduzir o modo.

    Não é a leitura do arquivo, que é de `io/readers.py` e acontece na rota: aqui basta o
    cabeçalho, e um arquivo que não se deixe ler devolve lista vazia, que a dedução lê como 'sem
    modo'. O delimitador é o que mais aparece na linha, entre vírgula, ponto e vírgula e tabulação;
    a primeira célula é o rótulo da coluna dos nomes de feature, e sai.
    """
    texto = bruto[:65536].decode("utf-8-sig", errors="replace")
    primeira = next((linha for linha in texto.splitlines() if linha.strip()), "")
    if not primeira:
        return []
    delimitador = max((",", ";", "\t"), key=primeira.count)
    celulas = [c.strip().strip('"') for c in primeira.split(delimitador)]
    return [c for c in celulas[1:] if c]

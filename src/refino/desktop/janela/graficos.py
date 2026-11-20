"""A aba 'Gráficos': uma figura do registo de cada vez, com a aba inteira para ela.

Este painel é dono do estado das figuras. Os corpos que as alimentam vivem aqui, e
`JanelaPrincipal._encaminha` os entrega aos `recebe_*` antes de chamar `define_execucao`. A ordem
importa: `define_execucao` constrói a figura no instante em que reconhece a execução nova, e guardar
depois desenharia o corpo da execução anterior sob o rótulo da corrente.

`_montadores()` existe para o teste conferir a bijeção com `plots.CONSTRUTORES`, que reprova
quando uma figura entra no registo sem montador aqui. O botão de gravar fica neste painel e o
`savefig` corre em `PainelDeFigura`, dono da `Figure`. Toda conta vive em `plots/`; este módulo não
faz aritmética.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Final

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from refino.api.esquemas import COEFICIENTE_DE_TUKEY
from refino.desktop.etapas import ENTRADAS_DA_ETAPA, NOMES_DAS_ETAPAS, etapa_a_refazer
from refino.desktop.janela.aparencia import FiltroDaRoda, sem_roda_sem_foco
from refino.desktop.janela.componentes import Recolhivel, Secao, desliga_com_motivo, religa
from refino.desktop.janela.figuras import (
    DPIS_DA_EXPORTACAO,
    EstadoDaFigura,
    FonteAusenteError,
    PainelDeFigura,
    exporta_habilitado,
    frase_do_minimo,
    mm_inteiros,
)
from refino.desktop.janela.tema import poe_icone, veste_os_seletores
from refino.plots.agrupamento import (
    arvore_do_fio,
    figura_do_dendrograma,
    figura_do_kmeans,
    figura_do_mapa_auto_organizavel,
    figura_do_mapa_de_calor,
    kmeans_do_fio,
    mapa_do_fio,
    recorte_do_dendrograma,
    recorte_do_kmeans,
    recorte_do_mapa_auto_organizavel,
    recorte_do_mapa_de_calor,
    som_do_fio,
)
from refino.plots.anova import (
    contexto_do_fio,
    figura_da_anova,
    pontos_da_anova,
    recorte_da_anova,
)
from refino.plots.base import NAO_DECLARADA, MatrizVaziaError, vetor_do_fio
from refino.plots.catalogo import CONSTRUTORES
from refino.plots.classificacao import (
    classificacao_do_fio,
    figura_do_random_forest,
    figura_do_svm,
    recorte_do_classificador,
)
from refino.plots.composicao import (
    composicao_do_fio,
    figura_da_composicao_por_classe,
    grupos_da_composicao,
    recorte_da_composicao_por_classe,
)
from refino.plots.correcao import (
    ROTULO_ANTES,
    ROTULO_DEPOIS,
    IntervaloAusente,
    IntervaloDeBarra,
    figura_do_painel_da_correcao,
    fila_do_fio,
    intervalo_do_fio,
    lado_da_correcao,
    par_da_correcao,
    recorte_do_painel_da_correcao,
)
from refino.plots.correlacao import (
    FEATURES_DA_BUSCA,
    busca_do_fio,
    figura_da_busca_por_padrao,
    figura_do_mapa_de_correlacao,
    mapa_de_correlacao_do_fio,
    recorte_da_busca_por_padrao,
    recorte_do_mapa_de_correlacao,
)
from refino.plots.curva_do_loess import (
    curva_do_fio,
    figura_da_curva_do_loess,
    recorte_da_curva_do_loess,
)
from refino.plots.deriva import figura_da_deriva_do_qc, recorte_da_deriva
from refino.plots.dispersao import (
    ROTULO_DAS_AMOSTRAS,
    ROTULO_DOS_QCS,
    figura_do_violino_do_rsd_por_feature,
    lado_do_rsd,
    par_do_rsd,
    recorte_do_violino_do_rsd_por_feature,
)
from refino.plots.distribuicao import (
    caixas_do_fio,
    caixas_por_amostra_do_fio,
    figura_da_densidade_por_amostra,
    figura_da_densidade_por_feature,
    figura_das_caixas_por_amostra,
    figura_das_caixas_por_feature,
    recorte_da_densidade_por_amostra,
    recorte_da_densidade_por_feature,
    recorte_das_caixas,
    recorte_das_caixas_por_amostra,
)
from refino.plots.pca import (
    SETAS_DO_BIPLOT,
    cargas_da_pca,
    escores_da_pca,
    figura_das_cargas,
    figura_do_biplot,
    figura_do_scree,
    figura_dos_scores,
    permanova_do_fio,
    recorte_das_cargas,
    recorte_do_biplot,
    recorte_do_scree,
    recorte_dos_scores,
    regiao_dos_scores,
)
from refino.plots.plsda import (
    FEATURES_DO_VIP,
    cargas_do_plsda,
    escores_do_plsda,
    figura_das_cargas_do_plsda,
    figura_do_vip,
    figura_dos_scores_do_plsda,
    recorte_das_cargas_do_plsda,
    recorte_do_vip,
    recorte_dos_scores_do_plsda,
    selecao_do_vip,
    validacao_do_fio,
    vip_do_plsda,
)
from refino.plots.plsda_multiclasse import (
    FEATURES_DA_IMPORTANCIA,
    figura_da_importancia_um_contra_resto,
    figura_das_cargas_um_contra_resto,
    figura_dos_escores_um_contra_resto,
    recorte_da_importancia_um_contra_resto,
    recorte_das_cargas_um_contra_resto,
    recorte_dos_escores_um_contra_resto,
)
from refino.plots.recorte import milhar
from refino.plots.rede import (
    figura_da_rede_de_correlacao_parcial,
    recorte_da_rede,
    rede_do_fio,
)
from refino.plots.series import (
    CORES_DOS_GRUPOS,
    PARES_DE_ESTILO,
    SIMBOLOS_DOS_GRUPOS,
    EstiloDoGrupo,
    EstilosDosGrupos,
    estilos_dos_grupos,
    ordem_canonica,
)
from refino.plots.univariada import (
    figura_do_fold_change,
    figura_dos_testes_t,
    pontos_do_fold_change,
    pontos_dos_testes_t,
    recorte_do_fold_change,
    recorte_dos_testes_t,
)
from refino.plots.volcano import (
    contraste_do_fio,
    figura_do_volcano,
    intervalos_escolhidos,
    pontos_do_volcano,
    recorte_do_volcano,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Callable, Mapping

    from matplotlib.figure import Figure

    from refino.desktop.etapas import Entrada
    from refino.desktop.janela.figuras import EscolhedorDeArquivo, Fonte
    from refino.interface.cliente import Resposta
    from refino.plots.catalogo import ConstrutorDeFigura

__all__ = [
    "AJUDA_DOS_ESTILOS",
    "MOTIVO_DA_EXPORTACAO_DESLIGADA",
    "MOTIVO_DA_IDA_AOS_GRAFICOS_DESLIGADA",
    "ROTULO_DA_EXPORTACAO",
    "ROTULO_DA_IDA_AOS_GRAFICOS",
    "SEM_EXPORTACAO",
    "SEM_GRUPOS_NA_TABELA",
    "TITULO_DOS_ESTILOS_DOS_GRUPOS",
    "TITULO_DOS_GRAFICOS_DA_ETAPA",
    "VISTA_DA_FIGURA",
    "PainelDeGraficos",
    "TabelaDosGrupos",
    "VistaDaEtapa",
    "frase_da_etapa_desatualizada",
    "frase_da_etapa_sem_resultado",
    "frase_de_grupos_demais",
    "frase_do_par_repetido",
]

# O texto do botão é constante para os testes falarem dele sem redigitar a frase. O formato do
# arquivo é o do nome escolhido no diálogo.
ROTULO_DA_EXPORTACAO: Final = "Salvar figura…"

# Por que 'Salvar figura…' está desligado, dito ao lado dele. Ele só se desliga sem figura na tela.
MOTIVO_DA_EXPORTACAO_DESLIGADA: Final = (
    "Sem figura na tela para salvar — o quadro acima diz o que falta para ela aparecer."
)

# O caminho da vista de uma etapa até a aba 'Gráficos'. A vista não grava: o tamanho e a resolução
# da gravação vivem só nesta aba.
ROTULO_DA_IDA_AOS_GRAFICOS: Final = "Exportar esta figura em Gráficos"
MOTIVO_DA_IDA_AOS_GRAFICOS_DESLIGADA: Final = (
    "Sem figura aqui para exportar — o quadro acima diz o que falta para ela aparecer."
)

# O título da vista dentro da aba da etapa, que é o bloco 3 da página.
TITULO_DOS_GRAFICOS_DA_ETAPA: Final = "Gráficos desta etapa"

# A linha do desfecho antes do primeiro clique. Vazia, leria-se como 'gravou e não tem nada a
# dizer', que é o fato oposto.
SEM_EXPORTACAO: Final = "Nenhuma gravação ainda"


# As cinco figuras que saem de `/matrizes/diagnosticar`. Valem sem execução enquanto o diagnóstico
# guardado for o da matriz corrente; as outras seguem a chave da execução.
FIGURAS_DO_DIAGNOSTICO: Final = frozenset(
    f.__name__
    for f in (
        figura_da_deriva_do_qc,
        figura_do_scree,
        figura_dos_scores,
        figura_das_cargas,
        figura_do_biplot,
    )
)


# A curva do LOESS de uma feature não segue a execução nem o diagnóstico: chega por rota própria,
# pedida no quadro da correção para uma feature, e vale enquanto for da matriz corrente, como a
# medição dos corretores. A janela usa o nome do construtor para ir à figura quando a curva volta.
FIGURA_DA_CURVA_DO_LOESS: Final = figura_da_curva_do_loess.__name__

# O que a figura diz antes de alguém pedir a curva, com o lugar onde se pede.
FRASE_SEM_CURVA_DO_LOESS: Final = (
    "Nenhuma curva do LOESS pedida para esta matriz — escolha uma feature no quadro «Correção do "
    "sinal» e clique em «Ver a curva do LOESS»"
)


def _intervalos_da_medicao(
    intervalos: Any,
) -> dict[str, IntervaloDeBarra | IntervaloAusente] | None:
    """Lê, chave a chave, o campo `intervalos` de `MedicaoDoCorretorResposta`.

    As leituras ficam em `desktop/janela/` e não em `plots/` porque o teste de contrato confere aqui
    cada chave da raiz `intervalo` contra `IntervaloDaBarra`. Quem valida os valores é
    `intervalo_do_fio`, na fronteira do construtor.

    Devolve `None` quando o corpo não traz o campo, ou o traz nulo (medição sem intervalo). Um
    dicionário vazio quer dizer que o intervalo foi pedido e nenhuma barra o teve; a fila declara a
    falta em cada uma.
    """
    if not isinstance(intervalos, dict):
        return None
    lidos: dict[str, IntervaloDeBarra | IntervaloAusente] = {}
    for corretor, intervalo in intervalos.items():
        if not isinstance(intervalo, dict):
            continue
        lidos[str(corretor)] = intervalo_do_fio(
            inferior=intervalo.get("inferior"),
            superior=intervalo.get("superior"),
            nivel=intervalo.get("nivel"),
            n_qc=intervalo.get("n_qc"),
            metodo=intervalo.get("metodo"),
            motivo=intervalo.get("motivo"),
        )
    return lidos


def _barras_da_medicao(corpo: Any) -> Any:
    """O corpo da medição como fila de barras, ou `None` quando ele não é uma.

    `recebe_medicao` e `define_matriz` (ao repor a medição guardada) fazem a mesma pergunta, e uma
    função só evita que as respostas divirjam. Quem decide é o servidor, pelo campo `avaliavel`.
    """
    corpo = corpo if isinstance(corpo, dict) else {}
    scores = corpo.get("scores")
    avaliavel = corpo.get("avaliavel")
    return corpo if avaliavel and isinstance(scores, dict) and scores else None


def _curva_da_resposta(corpo: Any) -> Any:
    """O corpo quando traz a curva, ou `None`; o par de `_barras_da_medicao`."""
    return corpo if isinstance(corpo, dict) and "curva" in corpo else None


def frase_da_etapa_desatualizada(etapa: int, refazer: int) -> str:
    """A frase de figura de etapa desatualizada, com a etapa dela e a que se refaz.

    A linha das etapas, que diz qual refazer, fica noutra aba; por isso a frase diz as duas.
    """
    return (
        f"A figura é da etapa {etapa}, {NOMES_DAS_ETAPAS[etapa]}, e os parâmetros da tela já não "
        f"são os dela — refaça a etapa {refazer}, {NOMES_DAS_ETAPAS[refazer]}"
    )


def frase_da_etapa_sem_resultado(etapa: int) -> str:
    """A frase de figura de uma etapa que ainda não escreveu os dados dela.

    Cada etapa é dona dos dados das próprias figuras, então a figura espera a etapa. A frase de
    `SEM_EXECUCAO` mandaria esperar o pipeline, quando o gesto que a faz aparecer é o botão da
    própria aba.
    """
    return (
        f"Nenhum resultado da etapa {etapa}, {NOMES_DAS_ETAPAS[etapa]}, ainda — a figura aparece "
        "quando ela terminar"
    )


# As figuras das etapas 1 e 2 do passo a passo: as da correção e as das distribuições. As outras
# dependem das análises, que só a etapa 3 (a execução) calcula.
ETAPA_DA_FIGURA: Final[dict[str, int]] = {
    **{
        f.__name__: 1
        for f in (
            figura_do_painel_da_correcao,
            figura_do_violino_do_rsd_por_feature,
            figura_da_composicao_por_classe,
        )
    },
    **{
        f.__name__: 2
        for f in (
            figura_das_caixas_por_feature,
            figura_das_caixas_por_amostra,
            figura_da_densidade_por_feature,
            figura_da_densidade_por_amostra,
        )
    },
}

# A etapa em cuja aba cada figura também aparece, derivada do registo. A etapa 1 tem as dela e a
# curva do LOESS, pedida no quadro da correção; a 2, as quatro distribuições; a 3, as das análises.
# As cinco do diagnóstico saem de 'Preparar o estudo' e ficam só na aba 'Gráficos'.
VISTA_DA_FIGURA: Final[dict[str, int]] = {
    construtor.funcao: ETAPA_DA_FIGURA.get(
        construtor.funcao, 1 if construtor.funcao == FIGURA_DA_CURVA_DO_LOESS else 3
    )
    for construtor in CONSTRUTORES
    if construtor.funcao not in FIGURAS_DO_DIAGNOSTICO
}

# Quem escreveu por último os dados das figuras de uma etapa: a prévia da etapa, ou uma execução
# que corresponde a ela. Sem dono, a figura espera a etapa (`frase_da_etapa_sem_resultado`).
_DONO_ETAPA: Final = "etapa"
_DONO_EXECUCAO: Final = "execucao"


# ---------------------------------------------------------------------------
# A cor e o símbolo de cada grupo
# ---------------------------------------------------------------------------
#
# O laboratório pediu para poder mudar as cores e os símbolos dos grupos experimentais. A escolha
# vive nesta aba, vale para a sessão e entra como o parâmetro `estilos` em toda figura que pinta por
# grupo. O vocabulário e as guardas são de `plots/series.py`; esta tabela só os mostra e os lê.

TITULO_DOS_ESTILOS_DOS_GRUPOS: Final = "Cores e símbolos dos grupos"

# A frase do vazio diz de onde os grupos vêm. Sem ela, a tabela vazia leria-se como 'este estudo
# não tem grupos'.
SEM_GRUPOS_NA_TABELA: Final = (
    "Ainda não há grupos. Eles aparecem aqui quando a correção ou a execução terminam, com os "
    "grupos de estudo da composição por classe; o QC não é grupo e fica de fora."
)

AJUDA_DOS_ESTILOS: Final = (
    "A cor e o símbolo valem para todas as figuras que desenham grupos, na tela e no arquivo "
    "salvo, até fechar o aplicativo. Dois grupos não podem ter a mesma cor e o mesmo símbolo."
)


def frase_do_par_repetido(outro: str, estilo: EstiloDoGrupo) -> str:
    """A recusa do par repetido, com o nome do grupo que já o usa."""
    return (
        f"«{outro}» já usa {estilo.descricao}: dois grupos com a mesma cor e o mesmo símbolo não "
        "se distinguem na figura. Mude a cor ou o símbolo."
    )


def frase_de_grupos_demais(quantos: int) -> str:
    """O terceiro desfecho: há grupos, e não há pares distintos para todos."""
    return (
        f"O estudo tem {milhar(quantos)} grupos e há {milhar(len(PARES_DE_ESTILO))} pares "
        "distintos de cor e símbolo: a escolha fica desligada, e as figuras usam as cores "
        "automáticas."
    )


def _amostra_da_cor(hexadecimal: str) -> QIcon:
    """O quadradinho da cor ao lado do nome: 'azul' e 'azul-celeste' só se separam vendo-as."""
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(hexadecimal))
    return QIcon(pixmap)


class TabelaDosGrupos(QWidget):
    """Uma linha por grupo da coorte, com o seletor da cor e o do símbolo.

    Os seletores mostram o que as figuras recebem: os dois leem `estilos()`, a resolução da coorte
    com as escolhas por cima. Antes de a pessoa escolher, cada linha mostra o par da posição
    canônica do grupo, e não um 'automático' que esconderia a cor.

    O par repetido é recusado aqui, onde a pessoa o pede: o seletor volta ao que estava e a frase
    diz qual grupo já o usa. Sem isso, a recusa de `plots/` levantaria dentro do slot do Qt, que a
    engole.

    `define_coorte` não redesenha: logo depois a janela chama `figuras.define_execucao`, e
    redesenhar aqui desenharia a figura nova sob a chave da execução anterior. Só a escolha aceita
    emite `mudou`.
    """

    mudou = Signal()
    """Emitido quando a pessoa muda a cor ou o símbolo de um grupo e a escolha é aceita."""

    def __init__(self, parent: QWidget | None = None, *, filtro_da_roda: FiltroDaRoda) -> None:
        super().__init__(parent)
        self._filtro_da_roda = filtro_da_roda
        self._coorte: tuple[str, ...] = ()
        # As escolhas da sessão, por nome de grupo. Ficam quando a coorte muda: o mesmo estudo
        # noutro modo traz os mesmos grupos, e a escolha volta a valer.
        self._escolhidos: dict[str, EstiloDoGrupo] = {}
        self._estilos: EstilosDosGrupos | None = None
        self.seletores: dict[str, tuple[QComboBox, QComboBox]] = {}
        disposicao = QVBoxLayout(self)
        disposicao.setContentsMargins(0, 0, 0, 0)
        disposicao.setSpacing(8)
        self.frase = QLabel(SEM_GRUPOS_NA_TABELA, self)
        self.frase.setObjectName("nota_do_campo")
        self.frase.setWordWrap(True)
        disposicao.addWidget(self.frase)
        self._grade = QGridLayout()
        self._grade.setHorizontalSpacing(12)
        self._grade.setVerticalSpacing(6)
        disposicao.addLayout(self._grade)

    def define_coorte(self, grupos: tuple[str, ...]) -> None:
        """Recebe os grupos da composição guardada e refaz as linhas só se a coorte mudou."""
        coorte = ordem_canonica(grupos)
        if coorte == self._coorte:
            return
        self._coorte = coorte
        self._remonta()

    def estilos(self) -> EstilosDosGrupos | None:
        """O que os montadores passam aos construtores. Com `None`, valem as cores automáticas."""
        return self._estilos

    def _remonta(self) -> None:
        """As linhas da coorte nova, com o padrão ou a escolha de cada grupo já posto."""
        while (item := self._grade.takeAt(0)) is not None:
            if (antigo := item.widget()) is not None:
                # Sai da árvore já, e não só no `deleteLater`: um seletor velho ainda filho deste
                # widget seria achado por `findChildren` até o laço de eventos correr.
                antigo.setParent(None)
                antigo.deleteLater()
        self.seletores = {}
        self._estilos = estilos_dos_grupos(self._coorte, self._escolhidos)
        if self._estilos is None:
            self.frase.setText(
                frase_de_grupos_demais(len(self._coorte)) if self._coorte else SEM_GRUPOS_NA_TABELA
            )
            return
        for coluna, titulo in enumerate(("Grupo", "Cor", "Símbolo")):
            cabecalho = QLabel(titulo, self)
            cabecalho.setObjectName("rotulo_do_campo")
            self._grade.addWidget(cabecalho, 0, coluna)
        for linha, grupo in enumerate(self._coorte, start=1):
            cor = sem_roda_sem_foco(QComboBox(self), self._filtro_da_roda)
            for nome_da_cor, hexadecimal in CORES_DOS_GRUPOS.items():
                cor.addItem(_amostra_da_cor(hexadecimal), nome_da_cor, nome_da_cor)
            simbolo = sem_roda_sem_foco(QComboBox(self), self._filtro_da_roda)
            for nome_do_simbolo in SIMBOLOS_DOS_GRUPOS:
                simbolo.addItem(nome_do_simbolo, nome_do_simbolo)
            self._grade.addWidget(QLabel(grupo, self), linha, 0)
            self._grade.addWidget(cor, linha, 1)
            self._grade.addWidget(simbolo, linha, 2)
            self.seletores[grupo] = (cor, simbolo)
        self._mostra_os_estilos()
        # Ligados depois de os seletores mostrarem o padrão: ligados antes, cada `addItem` e cada
        # `setCurrentIndex` contariam como escolha da pessoa.
        for grupo, par in self.seletores.items():
            for seletor in par:
                seletor.currentIndexChanged.connect(partial(self._escolheu, grupo))
        veste_os_seletores(self)
        self.frase.setText(AJUDA_DOS_ESTILOS)

    def _mostra_os_estilos(self) -> None:
        """Põe cada seletor no estilo resolvido, sem contar como escolha."""
        if self._estilos is None:  # pragma: no cover - `_remonta` só cria linhas com estilos
            return
        for grupo, (cor, simbolo) in self.seletores.items():
            estilo = self._estilos.por_grupo[grupo]
            for seletor, valor in ((cor, estilo.cor), (simbolo, estilo.simbolo)):
                bloqueado = seletor.blockSignals(True)
                seletor.setCurrentIndex(seletor.findData(valor))
                seletor.blockSignals(bloqueado)

    def _escolheu(self, grupo: str, _indice: int) -> None:
        """O slot dos seletores: aceita a escolha, ou volta atrás e diz quem já tem o par."""
        if self._estilos is None:  # pragma: no cover - só há seletor com estilos
            return
        cor, simbolo = self.seletores[grupo]
        candidato = EstiloDoGrupo(cor=cor.currentData(), simbolo=simbolo.currentData())
        outro = next(
            (
                nome
                for nome, estilo in self._estilos.por_grupo.items()
                if nome != grupo and estilo.par == candidato.par
            ),
            None,
        )
        if outro is not None:
            self._mostra_os_estilos()
            self.frase.setText(frase_do_par_repetido(outro, candidato))
            return
        # A escolha antiga com o mesmo par sai. Ela só pode ser de um grupo ausente, porque a de um
        # presente foi recusada acima; sem isto, a coorte que o trouxesse de volta teria dois grupos
        # com o mesmo par, e `EstilosDosGrupos` recusaria a tabela inteira.
        self._escolhidos = {
            nome: estilo
            for nome, estilo in self._escolhidos.items()
            if nome == grupo or estilo.par != candidato.par
        }
        self._escolhidos[grupo] = candidato
        self._estilos = estilos_dos_grupos(self._coorte, self._escolhidos)
        self._mostra_os_estilos()
        self.frase.setText(AJUDA_DOS_ESTILOS)
        self.mudou.emit()


def _veste_o_painel_de_figura(figuras: PainelDeFigura) -> None:
    """Põe o rótulo 'Figura' antes do combo, o nome do combo e o ícone do botão de voltar.

    Um sítio só para a aba 'Gráficos' e as vistas das etapas: as quatro são o mesmo controle, e
    vestidas à mão em cada lugar divergiriam na primeira mudança.
    """
    if (disposicao_da_figura := figuras.layout()) is not None:
        disposicao_da_figura.setContentsMargins(0, 0, 0, 0)
        disposicao_da_figura.setSpacing(12)
        primeiro = disposicao_da_figura.itemAt(0)
        if primeiro is not None and isinstance(
            controles_da_figura := primeiro.layout(), QHBoxLayout
        ):
            rotulo_da_figura = QLabel("Figura", figuras)
            rotulo_da_figura.setObjectName("rotulo_do_campo")
            controles_da_figura.insertWidget(0, rotulo_da_figura)
    figuras.seletor.setObjectName("seletor_da_figura")
    figuras.seletor.setMaximumWidth(520)
    # A largura mínima do combo é a de dez caracteres, e não a do nome mais longo, para a aba caber
    # numa janela de 1024 px com a frase do motivo ao lado de 'Voltar à vista original'. A largura
    # preferida continua a do nome mais longo (`ComboComNomes.sizeHint`).
    figuras.seletor.setSizeAdjustPolicy(
        QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    figuras.seletor.setMinimumContentsLength(10)
    figuras.botao_da_vista.setCursor(Qt.CursorShape.PointingHandCursor)
    # A cor do texto dos botões pelo token do tema, que a troca de esquema repinta.
    poe_icone(
        figuras.botao_da_vista,
        '<path d="M3 10a9 9 0 1 1 2.6 8.4M3 4v6h6"/>',
        "texto",
        14,
    )
    figuras.botao_da_vista.setIconSize(QSize(14, 14))


class VistaDaEtapa(QWidget):
    """Os gráficos de uma etapa na aba dela, com o mesmo estado da aba 'Gráficos'.

    É uma vista, e não uma cópia: o `PainelDeFigura` daqui recebe do `PainelDeGraficos` a mesma
    fonte, a mesma validade própria e a mesma frase própria, então a figura e a frase do vazio são
    as de 'Gráficos' por construção. Mudam o combo, que lista só as figuras da etapa
    (`VISTA_DA_FIGURA`), e a construção, adiada enquanto a aba não está à vista.

    Nasce uma vez e não é destruída: criar e destruir widgets de figura corrompia o heap no Windows.
    O `PainelDeGraficos` a cria e a janela a põe no lugar dos gráficos da aba da etapa.

    Não grava. O botão emite `exporta_em_graficos` com o nome do construtor, e a janela leva a
    pessoa à aba 'Gráficos', onde estão o tamanho e a resolução.
    """

    exporta_em_graficos = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        etapa: int,
        fonte: Fonte,
        validade_propria: Callable[[ConstrutorDeFigura], EstadoDaFigura | None],
        frase_propria: Callable[[ConstrutorDeFigura, EstadoDaFigura], str | None],
    ) -> None:
        super().__init__(parent)
        self.etapa = etapa
        disposicao = QVBoxLayout(self)
        # Sem margem: o respiro entre os blocos é o da folha da página da etapa, 24 px, como em
        # 'Preparar o estudo'.
        disposicao.setContentsMargins(0, 0, 0, 0)
        disposicao.setSpacing(0)
        # O bloco 3 da página da etapa, inteiro, com o número e a moldura de referência dos blocos 1
        # e 2 (`execucao.AbaDeEtapa`), para a página não lhe pôr um segundo título.
        self.secao = Secao(TITULO_DOS_GRAFICOS_DA_ETAPA, self, numero=3, referencia=True)
        disposicao.addWidget(self.secao)
        self.figuras = PainelDeFigura(
            self.secao,
            construtores=tuple(c for c in CONSTRUTORES if VISTA_DA_FIGURA.get(c.funcao) == etapa),
            fonte=fonte,
            validade_propria=validade_propria,
            frase_propria=frase_propria,
            adiada=True,
        )
        self.figuras.setObjectName("vista_dos_graficos_da_etapa")
        _veste_o_painel_de_figura(self.figuras)
        self.secao.corpo.addWidget(self.figuras, 1)
        linha = QHBoxLayout()
        linha.setSpacing(12)
        self.botao_de_exportar = QPushButton(ROTULO_DA_IDA_AOS_GRAFICOS, self.secao)
        self.botao_de_exportar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.botao_de_exportar.clicked.connect(self._pede_os_graficos)
        linha.addWidget(self.botao_de_exportar)
        linha.addStretch(1)
        self.secao.corpo.addLayout(linha)
        # O estado inicial pelo predicado, e só depois a ligação, como em 'Salvar figura…': a
        # emissão que o painel faz no próprio construtor não tem ouvinte.
        self._segue_o_exportavel(exporta_habilitado(EstadoDaFigura.SEM_EXECUCAO))
        self.figuras.exportavel.connect(self._segue_o_exportavel)

    def _segue_o_exportavel(self, habilitado: bool) -> None:
        """Ligado com figura na tela; sem ela, desligado com a frase do que falta ao lado."""
        if habilitado:
            religa(self.botao_de_exportar)
        else:
            desliga_com_motivo(self.botao_de_exportar, MOTIVO_DA_IDA_AOS_GRAFICOS_DESLIGADA)

    def _pede_os_graficos(self) -> None:
        """O clique: a figura desta vista, pelo nome do construtor, a quem troca de aba."""
        funcao = self.figuras.funcao_escolhida()
        if funcao is not None:
            self.exporta_em_graficos.emit(funcao)


class PainelDeGraficos(QWidget):
    """A aba inteira para uma figura de cada vez, e os corpos que as alimentam.

    A janela alimenta este painel: `JanelaPrincipal._encaminha` roteia o corpo de
    `/matrizes/diagnosticar` e o de `/execucoes` para os `recebe_*`, e só depois chama
    `define_execucao`, que leva a chave a `figuras` e às vistas das três abas de etapa
    (`VistaDaEtapa`). É o mesmo estado em quatro lugares.

    O corpo da medição chega por uma terceira rota e é o único que não segue a chave de invalidação
    da execução; ver `recebe_medicao` e `define_matriz`.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        escolhedor: EscolhedorDeArquivo | None = None,
    ) -> None:
        """`escolhedor` é o diálogo de gravação, e ele atravessa até `PainelDeFigura`.

        Quem grava é o widget que tem a figura; este painel só o repassa, para a janela poder
        injetar um duplo. Um `QFileDialog` real abre modal, e um teste que não o substitua pendura
        a suíte.
        """
        super().__init__(parent)

        # O corpo do último `POST /matrizes/diagnosticar`, ou `None`. `None` e não `{}`: 'nenhum
        # diagnóstico correu' e 'correu e a trajetória veio vazia' mostram o mesmo texto na tela,
        # mas são fatos diferentes.
        self._trajetoria: Any = None
        self._distribuicoes: Any = None
        self._volcano: Any = None
        # A ANOVA tem raiz própria porque a população é outra: o volcano descreve o par de grupos
        # declarado, a ANOVA todos os grupos de estudo da matriz. As duas faltam por motivos que não
        # se sobrepõem, e um campo só faria a ausência de uma parecer implicar a da outra.
        self._anova: Any = None
        # Estas duas raízes não têm caminho de fio: nenhuma rota devolve o que os construtores de
        # `plsda_multiclasse` e `agrupamento` consomem. Ficam em `None`, e o montador leva à recusa
        # por população vazia, que é a verdade; sem montador, a pessoa receberia
        # `FonteAusenteError`, que o painel descreve como defeito de ligação do programa.
        self._plsda_multiclasse: Any = None
        self._agrupamento: Any = None
        # `POST /execucoes` devolve `correlacao` e `busca_por_padrao` em raízes separadas porque
        # `pattern_reference` a `None` salta a busca sem desligar o mapa: as duas partilham o
        # coeficiente, não o alvo.
        self._correlacao: Any = None
        self._busca_por_padrao: Any = None
        # `kmeans` e `som` têm raízes separadas porque as bandeiras de `AnalysisSpec` são separadas,
        # as duas desligadas por padrão, e pedir uma sem a outra é o caso comum.
        self._kmeans: Any = None
        self._som: Any = None
        # `random_forest` e `svm` têm raízes separadas porque `AnalysisSpec.classifiers` é uma
        # tupla: pedir um sem o outro é o caso comum, porque cada um custa minutos.
        self._random_forest: Any = None
        self._svm: Any = None
        # A rede tem raiz própria porque `AnalysisSpec.dspc` é bandeira separada da correlação: a
        # rede estima a precisão e o mapa a covariância, e pedir um sem o outro é o caso comum.
        self._rede: Any = None
        self._pca: Any = None
        self._plsda: Any = None
        # O par de PCAs de `/execucoes`, que o painel da correção consome.
        self._pcas_da_correcao: Any = None
        # O par de vetores de RSD por feature, que o violino consome.
        self._rsd_por_feature: Any = None
        # O mesmo par sobre a matriz ingerida, vindo do diagnóstico: é o violino antes da primeira
        # correção. Vale, como as figuras do diagnóstico, até chegar um resultado de corretor para a
        # matriz corrente; `_violino_corrigido` diz se chegou, e só a troca de matriz o desfaz.
        self._rsd_da_ingerida: Any = None
        self._violino_corrigido = False
        # A composição por classe, medida sobre a matriz ingerida: ela descreve o estudo, e não a
        # correção.
        self._composicao: Any = None
        # A fila de barras de `POST /matrizes/medir-corretor`, ou `None` quando não há o que
        # desenhar. Os outros corpos guardam o que vier e deixam o construtor recusar; este decide
        # aqui porque o servidor o diz em `avaliavel`, e repetir a decisão no desenho seria uma
        # segunda fonte.
        self._medicao: Any = None
        # A última resposta aceita da medição, e a matriz dela. `_medicao`, acima, é a que se vê: a
        # guardada quando a matriz dela é a corrente, e `None` quando não é. `define_matriz` a
        # esconde em vez de a apagar, e a volta àquela matriz a repõe, porque medir custa minutos.
        # O corpo vai inteiro, com a recusa, porque a janela reescreve o rótulo da medição a partir
        # dele (`resposta_da_medicao_na_tela`).
        self._resposta_da_medicao: Any = None
        self._medicao_de: str | None = None
        # `None` é corpo legítimo (o `500` servido como texto chega assim), então 'há resposta' fica
        # à parte.
        self._ha_resposta_da_medicao = False
        # A curva do LOESS de uma feature e a matriz a que ela pertence, no molde da medição: rota
        # própria, sem chave de invalidação no corpo, escondida na troca de matriz e reposta na
        # volta.
        self._curva_do_loess: Any = None
        self._identidade_da_curva: str | None = None
        self._resposta_da_curva: Any = None
        self._curva_de: str | None = None
        self._ha_resposta_da_curva = False
        # O corretor que ia correr no disparo, carimbado com a matriz: a rota devolve a curva do
        # LOESS qualquer que seja o corretor, e a figura só a diz 'a que corrigiu o sinal' quando
        # ele é o LOESS. `None` é 'nenhuma curva pedida'.
        self._corretor_da_curva: str | None = None
        # O corretor da curva guardada, que é o que a figura diz: com a curva reposta na volta a uma
        # matriz, o carimbo acima pode ser o de um pedido feito noutra.
        self._corretor_da_curva_guardada: str | None = None
        # A identidade da matriz corrente (`interface/fluxo.chave_da_entrada` com `metodo=None`). É
        # o que
        # `define_matriz` compara, porque o corpo da medição não traz `chave_de_invalidacao`.
        self._identidade_da_matriz: str | None = None
        # O carimbo da medição em voo, posto no disparo. Sem ele, a resposta de A chegada com a tela
        # já em B seria guardada como de B. Ver `carimba_a_medicao` e `recebe_medicao`.
        self._identidade_da_medicao: str | None = None
        # O carimbo do diagnóstico, no molde do da medição: `carimba_o_diagnostico` guarda a matriz
        # do disparo, e `recebe_diagnostico` o passa ao corpo guardado. As cinco figuras do
        # diagnóstico valem sem execução enquanto ele for o da matriz corrente.
        self._identidade_do_diagnostico: str | None = None
        self._diagnostico_de: str | None = None
        self._ha_diagnostico = False
        # As figuras das etapas 1 e 2 são as mesmas da execução, e os dados delas vivem nos mesmos
        # campos: quem os escreveu por último (a prévia de uma etapa ou a execução) decide a regra
        # de validade. `_fonte_da_etapa` guarda isso, e `_etapa_em_dia` o frescor que a janela
        # calcula pela fotografia da etapa.
        self._fonte_da_etapa: dict[int, str] = {}
        self._etapa_em_dia: dict[int, bool] = {}
        # E a entrada com que cada etapa foi feita: a frase de figura desatualizada segue-a.
        self._entradas_das_etapas: dict[int, Entrada] = {}
        # Uma execução reescreveu dados de que a etapa continua dona: `define_execucao` refaz as
        # figuras das etapas, que a chave nova não refaria (`_da_execucao`).
        self._refazer_as_figuras_das_etapas = False

        disposicao = QVBoxLayout(self)
        self.setProperty("folhaDeEtapa", True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        disposicao.setContentsMargins(20, 20, 20, 20)
        disposicao.setSpacing(12)

        # A fonte entra na construção: `PainelDeFigura` levanta `FonteAusenteError` quando há
        # construtor registado e ninguém para o chamar. `construtores=` não é passado, porque o
        # padrão já é `plots.CONSTRUTORES`.
        self.figuras = PainelDeFigura(
            self,
            fonte=self._constroi_figura,
            escolhedor=escolhedor,
            validade_propria=self._validade_propria,
            frase_propria=self._frase_propria,
        )
        self.figuras.setObjectName("area_dos_graficos")
        _veste_o_painel_de_figura(self.figuras)

        # As vistas das três abas de etapa, com o mesmo dono de estado. Nascem aqui, uma vez, e a
        # janela as põe no lugar dos gráficos de cada aba; até lá ficam escondidas, para não se
        # desenharem soltas sobre esta aba. Todo ponto que refaz, troca a execução ou escolhe figura
        # passa pelos métodos deste painel que alcançam as quatro (`_paineis_de_figura`).
        self.vistas_das_etapas: dict[int, VistaDaEtapa] = {}
        for etapa in sorted(set(VISTA_DA_FIGURA.values())):
            vista = VistaDaEtapa(
                self,
                etapa=etapa,
                fonte=self._constroi_figura,
                validade_propria=self._validade_propria,
                frase_propria=self._frase_propria,
            )
            vista.hide()
            self.vistas_das_etapas[etapa] = vista

        # `addWidget(..., 1)` e nenhum `addStretch` no fim: a folga é da figura, que esta aba existe
        # para deixar crescer. Um esticador no fim lhe roubaria a altura.
        disposicao.addWidget(self.figuras, 1)

        # O controle de gravar e a linha do desfecho ficam abaixo do canvas; acima, separariam o
        # combo da figura que ele escolhe. O tamanho vai numa linha e a resolução ao lado do botão,
        # porque os três campos numa linha só alargariam a aba além do que cabe numa janela de
        # 1024 px; a frase do mínimo quebra linha pelo mesmo motivo. Os campos seguem o tamanho com
        # que a figura foi construída até a pessoa os mudar, e passam pelo filtro da roda como todo
        # campo rolável desta janela.
        self._filtro_da_roda = FiltroDaRoda(self)
        tamanho = QHBoxLayout()
        tamanho.setSpacing(8)
        self.dpi_da_exportacao = sem_roda_sem_foco(QComboBox(self), self._filtro_da_roda)
        for valor in DPIS_DA_EXPORTACAO:
            self.dpi_da_exportacao.addItem(f"{valor:,} ppp".replace(",", "."), valor)
        self.largura_da_exportacao = sem_roda_sem_foco(QSpinBox(self), self._filtro_da_roda)
        self.altura_da_exportacao = sem_roda_sem_foco(QSpinBox(self), self._filtro_da_roda)
        for campo in (self.largura_da_exportacao, self.altura_da_exportacao):
            campo.setRange(20, 600)
            campo.setSuffix(" mm")
            campo.editingFinished.connect(self._tamanho_editado)
        self._tamanho_escolhido = False
        self.minimo_da_exportacao = QLabel("", self)
        self.minimo_da_exportacao.setObjectName("nota_do_campo")
        self.minimo_da_exportacao.setWordWrap(True)
        for rotulo, controle in (
            ("Largura", self.largura_da_exportacao),
            ("Altura", self.altura_da_exportacao),
        ):
            tamanho.addWidget(QLabel(rotulo, self))
            tamanho.addWidget(controle)
        tamanho.addWidget(self.minimo_da_exportacao, 1)
        disposicao.addLayout(tamanho)

        controles = QHBoxLayout()
        controles.setSpacing(12)
        resolucao = QHBoxLayout()
        resolucao.setSpacing(8)
        resolucao.addWidget(QLabel("Resolução", self))
        resolucao.addWidget(self.dpi_da_exportacao)
        controles.addLayout(resolucao)
        self.botao_de_exportar = QPushButton(ROTULO_DA_EXPORTACAO, self)
        self.botao_de_exportar.setCursor(Qt.CursorShape.PointingHandCursor)
        poe_icone(
            self.botao_de_exportar,
            '<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
            "texto",
            14,
        )
        self.botao_de_exportar.setIconSize(QSize(14, 14))
        self.botao_de_exportar.clicked.connect(self._exporta)
        controles.addWidget(self.botao_de_exportar)
        self.exportacao = QLabel(SEM_EXPORTACAO, self)
        self.exportacao.setObjectName("nota_do_campo")
        self.exportacao.setWordWrap(True)
        controles.addWidget(self.exportacao, 1)
        disposicao.addLayout(controles)
        # Depois de o botão estar na caixa dele, porque a frase do motivo entra logo a seguir a ele
        # (`desliga_com_motivo`).
        self._segue_o_exportavel(exporta_habilitado(EstadoDaFigura.SEM_EXECUCAO))

        # A cor e o símbolo de cada grupo, num bloco que nasce fechado: aberta, a tabela come a
        # altura da figura. A escolha aceita refaz a figura da tela; `reavalia` passa pela mesma
        # porta do combo, então a figura de outra execução não volta por aqui.
        self.recolhivel_dos_grupos = Recolhivel(TITULO_DOS_ESTILOS_DOS_GRUPOS, self)
        corpo_dos_grupos = QVBoxLayout(self.recolhivel_dos_grupos.corpo)
        corpo_dos_grupos.setContentsMargins(0, 4, 0, 0)
        self.tabela_dos_grupos = TabelaDosGrupos(
            self.recolhivel_dos_grupos.corpo, filtro_da_roda=self._filtro_da_roda
        )
        corpo_dos_grupos.addWidget(self.tabela_dos_grupos)
        disposicao.addWidget(self.recolhivel_dos_grupos)
        # A escolha de cor e símbolo vale para as figuras das vistas das etapas também.
        self.tabela_dos_grupos.mudou.connect(self.reavalia)

        # Ligado depois da construção do `PainelDeFigura`: a emissão do `__init__` dele não tem
        # ouvinte, e por isso o estado inicial do botão está posto acima.
        self.figuras.exportavel.connect(self._segue_o_exportavel)
        self.figuras.exportavel.connect(self._segue_o_tamanho_da_figura)

    def _segue_o_exportavel(self, habilitado: bool) -> None:
        """Liga 'Salvar figura…' com figura na tela; sem ela, desliga com a frase do que falta.

        `desliga_com_motivo`, e não `setEnabled`, para o botão desligado ter a explicação ao lado.
        """
        if habilitado:
            religa(self.botao_de_exportar)
        else:
            desliga_com_motivo(self.botao_de_exportar, MOTIVO_DA_EXPORTACAO_DESLIGADA)

    # ------------------------------------------------------------------
    # As quatro vistas: a aba 'Gráficos' e a de cada etapa
    # ------------------------------------------------------------------

    def _paineis_de_figura(self) -> tuple[PainelDeFigura, ...]:
        """A aba 'Gráficos' e as vistas das etapas: tudo o que mostra uma figura deste estado."""
        return (self.figuras, *(vista.figuras for vista in self.vistas_das_etapas.values()))

    def reavalia(self) -> None:
        """Refaz a figura escolhida nas quatro vistas. As escondidas adiam a construção."""
        for painel in self._paineis_de_figura():
            painel.reavalia()

    def define_execucao(self, resposta: Resposta) -> None:
        """Passa a identidade da execução corrente às quatro vistas.

        Os `recebe_*` correm antes desta chamada. As figuras das etapas cujos dados a execução
        reescreveu sem os tirar da etapa são refeitas aqui, só nos painéis que mostram uma delas: a
        validade delas é o frescor da etapa, que a chave nova não muda, e sem isto a tela ficaria
        com o desenho dos dados de antes.
        """
        for painel in self._paineis_de_figura():
            painel.define_execucao(resposta)
        if self._refazer_as_figuras_das_etapas:
            self._refazer_as_figuras_das_etapas = False
            for painel in self._paineis_de_figura():
                if painel.funcao_escolhida() in ETAPA_DA_FIGURA:
                    painel.reavalia()

    def esquece_a_execucao(self) -> None:
        """Esquece, nas quatro vistas, a execução que deixou de valer sem resposta nova."""
        for painel in self._paineis_de_figura():
            painel.esquece_a_execucao()

    def seleciona(self, funcao: str) -> bool:
        """Põe a figura `funcao` em cada vista que a lista. Devolve se a aba 'Gráficos' a tem.

        A vista de uma etapa que não a lista fica como está: o combo dela não tem a figura de outra
        etapa, e por isso o retorno é o de 'Gráficos', que tem todas.
        """
        for vista in self.vistas_das_etapas.values():
            vista.figuras.seleciona(funcao)
        return self.figuras.seleciona(funcao)

    def _exporta(self) -> None:
        """O slot do botão: manda gravar e escreve na tela o que aconteceu.

        A frase vai para a tela sempre, também no sucesso: a maior parte dos desfechos não escreve
        arquivo, e silêncio depois de um clique lê-se como 'gravou'. A gravação sem rodapé diz onde
        ficou o texto dele, que é o que quem monta a legenda vai procurar.

        Nada levanta daqui, porque uma exceção dentro de um slot do Qt é engolida e o painel para de
        atualizar em silêncio. `PainelDeFigura.exporta` apanha o `OSError`, a falta de memória e o
        limite de pixels do Pillow, e sabe nomear o caminho e o tamanho.
        """
        _desfecho, frase = self.figuras.exporta(
            dpi=self.dpi_da_exportacao.currentData(),
            tamanho_mm=(
                float(self.largura_da_exportacao.value()),
                float(self.altura_da_exportacao.value()),
            ),
        )
        self.exportacao.setText(frase)

    def _segue_o_tamanho_da_figura(self, desenhada: bool) -> None:
        """Os campos do tamanho seguem a figura nova, enquanto a pessoa não os mudou."""
        construcao = self.figuras.tamanho_de_construcao_mm
        if not desenhada or self._tamanho_escolhido or construcao is None:
            return
        largura, altura = mm_inteiros(construcao)
        self.largura_da_exportacao.setValue(largura)
        self.altura_da_exportacao.setValue(altura)
        self.minimo_da_exportacao.setText("")

    def _tamanho_editado(self) -> None:
        """A pessoa mudou o tamanho: ele passa a valer, e o mínimo daquela largura aparece."""
        self._tamanho_escolhido = True
        self.atualiza_o_minimo()

    def atualiza_o_minimo(self) -> str:
        """Mede e mostra a altura mínima na largura escolhida, antes de a pessoa salvar.

        Abaixo do mínimo a figura sai sem rodapé e o texto dele vai para um `.txt` ao lado; a frase
        diz isso, e o nome do arquivo, antes do clique.
        """
        minima = self.figuras.altura_minima_mm(float(self.largura_da_exportacao.value()))
        texto = frase_do_minimo(minima, float(self.altura_da_exportacao.value()))
        self.minimo_da_exportacao.setText(texto)
        return texto

    # ------------------------------------------------------------------
    # Os corpos guardados, e nenhum deles desenha
    # ------------------------------------------------------------------

    def recebe_diagnostico(self, corpo: Any) -> None:
        """Guarda a trajetória do último `POST /matrizes/diagnosticar`, sem desenhar.

        Guardar e desenhar são separados: quem decide quando a figura entra na tela é
        `PainelDeFigura`. Quando o diagnóstico chega, a janela chama `figuras.reavalia`. O corpo
        guardado leva o carimbo da matriz do disparo.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        self._trajetoria = corpo.get("trajetoria_dos_qcs")
        # O violino da matriz ingerida, que vale até chegar o primeiro corretor.
        self._rsd_da_ingerida = corpo.get("rsd_por_feature")
        self._diagnostico_de = self._identidade_do_diagnostico
        self._ha_diagnostico = True

    def recebe_distribuicoes(self, corpo: Any, *, corresponde: bool) -> None:
        """Guarda o resumo das distribuições do último `/execucoes`, sem desenhar.

        `corresponde` diz se a entrada da execução corresponde à etapa 2 guardada, e é a janela que
        o diz, lendo o estado das etapas. Sem correspondência o resumo não é escrito: a figura é da
        etapa 2, e só ela, ou uma execução que a reproduz, a escreve (`_da_execucao`).

        Corre antes de `figuras.define_execucao`, que constrói a figura no instante em que reconhece
        a execução nova; guardar depois desenharia o resumo da execução anterior sob o rótulo da
        corrente.

        Corre para toda resposta, e é isso que apaga o resumo velho num `202` ou num erro: sem
        `distribuicoes` o campo fica `None`, e o construtor recusa por população vazia. Dados vindos
        de uma prévia da etapa ficam na tela até a execução voltar com sucesso.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        if self._da_execucao(2, "distribuicoes", corpo, corresponde=corresponde):
            self._distribuicoes = corpo.get("distribuicoes")

    def recebe_volcano(self, corpo: Any) -> None:
        """Guarda os vetores do volcano do último `/execucoes`, sem desenhar.

        Método próprio porque as ausências diferem: `distribuicoes` vem sempre (toda execução tem
        uma matriz antes e uma depois), e `volcano` só quando o par de grupos foi declarado. Um
        método para as duas faria crer que a falta de uma implica a da outra. Corre antes de
        `figuras.define_execucao`.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        self._volcano = corpo.get("volcano")

    def recebe_anova(self, corpo: Any) -> None:
        """Guarda os vetores da ANOVA de N grupos do último `/execucoes`, sem desenhar.

        Método próprio porque a população é outra: o volcano é o par de grupos declarado, a ANOVA
        todos os grupos de estudo da matriz. Elas faltam por motivos que não se sobrepõem: o volcano
        sem par, a ANOVA quando a família não foi pedida ou a matriz não a sustenta.

        Corre para toda resposta, e um corpo sem `anova` deixa o campo `None`, que
        `recorte_da_anova` recusa por população vazia. Corre antes de `figuras.define_execucao`.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        self._anova = corpo.get("anova")

    def recebe_correlacao(self, corpo: Any) -> None:
        """Guarda as duas raízes da correlação do último `/execucoes`, sem desenhar.

        Um método para as duas porque vêm da mesma família pedida e da mesma resposta:
        `correlation` ligado produz o mapa, e a busca acompanha quando há referência. O alvo, que as
        separa, fica nos dois atributos.

        Corre para toda resposta, e um corpo sem os campos os deixa `None`, que os construtores
        recusam por população vazia. Corre antes de `figuras.define_execucao`.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        self._correlacao = corpo.get("correlacao")
        self._busca_por_padrao = corpo.get("busca_por_padrao")

    def recebe_agrupamento_particional(self, corpo: Any) -> None:
        """Guarda as duas raízes particionais (k-means e SOM) do último `/execucoes`, sem desenhar.

        Um método para as duas porque a resposta traz sempre as duas chaves; a bandeira, que as
        separa, fica nos dois atributos. Corre para toda resposta, e um corpo sem os campos os deixa
        `None`, que os construtores recusam por população vazia. Corre antes de
        `figuras.define_execucao`.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        self._kmeans = corpo.get("kmeans")
        self._som = corpo.get("som")

    def recebe_classificadores(self, corpo: Any) -> None:
        """Guarda as duas raízes de classificação do último `/execucoes`, sem desenhar.

        Um método para as duas porque a resposta traz sempre as duas chaves; qual classificador
        entrou na tupla fica nos dois atributos. Corre para toda resposta, e um corpo sem os campos
        os deixa `None`, que os construtores recusam por população vazia.

        Corre antes de `figuras.define_execucao`. Aqui a ordem pesa mais: um `p` de permutação sob o
        rótulo de outra execução é exatamente o número que a validação existe para proteger.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        self._random_forest = corpo.get("random_forest")
        self._svm = corpo.get("svm")

    def recebe_rede(self, corpo: Any) -> None:
        """Guarda a rede de correlação parcial do último `/execucoes`, sem desenhar.

        Método próprio porque `AnalysisSpec.correlation` e `AnalysisSpec.dspc` são bandeiras
        separadas, e a rede é o ajuste mais caro desta aba: pedir o mapa sem ela é o caso comum.
        Corre para toda resposta, e um corpo sem `rede` deixa o campo `None`, que o construtor
        recusa por população vazia. Corre antes de `figuras.define_execucao`.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        self._rede = corpo.get("rede")

    def recebe_pca(self, corpo: Any) -> None:
        """Guarda a PCA do último `POST /matrizes/diagnosticar`, sem desenhar.

        Método próprio, como `recebe_volcano`: aqui as duas raízes saem do mesmo `StructuralPCA` e
        vêm sempre juntas, mas a coincidência não deve virar acoplamento. Corre para toda resposta,
        e é isso que apaga a PCA velha num erro de leitura.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        self._pca = corpo.get("pca")
        # O mesmo corpo do diagnóstico, então o mesmo carimbo.
        self._diagnostico_de = self._identidade_do_diagnostico
        self._ha_diagnostico = True

    def recebe_plsda(self, corpo: Any) -> None:
        """Guarda os vetores do PLS-DA do último `/execucoes`, sem desenhar.

        A raiz vem `None` quando não há par de grupos declarado, porque o PLS-DA é pareado; isso é
        'não calculado', e os construtores recusam por população vazia. Corre antes de
        `figuras.define_execucao`.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        self._plsda = corpo.get("plsda")

    def recebe_pca_do_par(self, corpo: Any, *, corresponde: bool) -> None:
        """Guarda o par de PCAs do último `/execucoes`, sem desenhar.

        `corresponde` segue a regra de `recebe_distribuicoes`: o par é figura da etapa 1.

        A raiz vem de `/execucoes`, e não do diagnóstico que `recebe_pca` guarda: a sequência
        'arquivo A, diagnosticar, executar, arquivo B, diagnosticar' desenharia o antes de B ao
        lado do depois de A, porque o diagnóstico não segue a chave da execução. Vindo os dois lados
        da mesma resposta, os dois painéis ficam sob a mesma chave. Corre antes de
        `figuras.define_execucao`.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        if self._da_execucao(1, "pcas_da_correcao", corpo, corresponde=corresponde):
            self._pcas_da_correcao = corpo.get("pcas_da_correcao")

    def recebe_rsd_por_feature(self, corpo: Any, *, corresponde: bool) -> None:
        """Guarda os dois vetores de RSD por feature do último `/execucoes`, sem desenhar.

        `corresponde` segue a regra de `recebe_distribuicoes`: o violino é figura da etapa 1. Sem
        correspondência o violino continua na regra do diagnóstico, porque a execução não corrigiu
        nada que a aba mostre.

        A raiz é medida sobre a matriz corrigida, e não sobre a processada: entre as duas está o
        corte por RSD de QC, que remove exatamente as features cuja dispersão o violino mostra.

        O primeiro resultado de corretor tira o violino da regra do diagnóstico; o `202` e o erro
        não, porque sem corretor a matriz ingerida é o que há a desenhar. Depois do primeiro, o
        violino segue a chave da execução, e um `202` seguinte o retira em vez de voltar à
        ingerida, para a figura não ir e vir. Corre antes de `figuras.define_execucao`.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        if self._da_execucao(1, "rsd_por_feature", corpo, corresponde=corresponde):
            self._rsd_por_feature = corpo.get("rsd_por_feature")
            self._violino_corrigido = self._violino_corrigido or isinstance(
                self._rsd_por_feature, dict
            )

    def recebe_composicao(self, corpo: Any, *, corresponde: bool) -> None:
        """Guarda a composição por classe do último `/execucoes`, sem desenhar.

        `corresponde` segue a regra de `recebe_distribuicoes`: a composição é figura da etapa 1. A
        coorte da tabela dos estilos vem do corpo mesmo sem correspondência, porque ela é do estudo
        e as figuras da análise pintam por ela; chega antes de `define_execucao` desenhar, e a
        figura nasce já com os estilos.

        A raiz é medida sobre a matriz ingerida, ao contrário das outras, que medem a correção: o
        corretor cria ausência, e a composição descreve o estudo. Corre antes de
        `figuras.define_execucao`.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        if self._da_execucao(1, "composicao_por_classe", corpo, corresponde=corresponde):
            self._composicao = corpo.get("composicao_por_classe")
            self._atualiza_a_coorte(self._composicao)
        elif "composicao_por_classe" in corpo:
            self._atualiza_a_coorte(corpo.get("composicao_por_classe"))

    def _atualiza_a_coorte(self, composicao: Any) -> None:
        """Põe na tabela dos estilos a coorte da composição que chegou.

        A composição é a fonte da coorte porque as outras raízes trazem um recorte dela, ou o QC
        junto, ou só vêm quando pedidas; quem lê as classes é
        `plots.composicao.grupos_da_composicao`.
        """
        composicao = composicao if isinstance(composicao, dict) else {}
        self.tabela_dos_grupos.define_coorte(grupos_da_composicao(composicao.get("classes")))

    # ------------------------------------------------------------------
    # A medição dos corretores, a única carga que não segue a chave da execução
    # ------------------------------------------------------------------

    def recebe_medicao(self, corpo: Any) -> bool:
        """Guarda o corpo de `POST /matrizes/medir-corretor`, sem desenhar.

        Devolve se a resposta foi aceita (guardada, ou um corpo sem barras que apagou a anterior) e
        `False` quando o carimbo era de outra matriz e ela foi descartada; sem a distinção, a janela
        escreveria sob B a prosa da medição de A. A aceita fica guardada inteira, com a matriz dela,
        e `define_matriz` a esconde e repõe pela identidade, como o diagnóstico. A descartada não
        toca a guardada.

        Não chama `figuras.define_execucao`: o corpo não traz `chave_de_invalidacao`, e o painel
        reconheceria uma execução nova e retiraria todas as figuras por uma resposta que não é de
        `/execucoes`.

        Quem decide se há barras é o servidor, por `avaliavel` (`Label.assessable`), o mesmo campo
        que `interface/fluxo.prosa_da_medicao` lê; um predicado próprio deixaria a aba da correção
        dizer
        'não foi avaliável' enquanto esta desenha barras. O corpo é guardado como chegou: um `score`
        pode vir `null` com `avaliavel=True`, e trocá-lo por zero poria a barra do corretor não
        medido colada ao eixo, o retrato de uma correção perfeita. A população por barra não vem no
        corpo, e `n_features_medidas` é o teto pedido ao amostrador, não ela.

        O carimbo é posto no disparo (`carimba_a_medicao`) e conferido aqui, porque ler a identidade
        na chegada tomaria a matriz nova como a medida. A resposta de outra matriz é descartada, e
        não guardada com aviso: barras com uma nota ao lado seriam lidas como as desta matriz. Sem
        carimbo a guarda não corre, para as barras não sumirem em silêncio se o disparo deixar de
        carimbar; quem cobre isso é um teste sobre o disparo.
        """
        if self.medicao_de_outra_matriz():
            return False
        self._resposta_da_medicao = corpo
        self._medicao_de = self._identidade_da_matriz
        self._ha_resposta_da_medicao = True
        self._medicao = _barras_da_medicao(corpo)
        return True

    def medicao_de_outra_matriz(self) -> bool:
        """Diz se a medição carimbada no disparo é de outra matriz que a da tela.

        É a guarda de `recebe_medicao`, sem receber nada. A janela pergunta antes de mostrar o
        desfecho no aviso, e `recebe_medicao` faz a mesma pergunta para descartar; uma conta só
        impede o aviso de abrir sobre uma resposta que o painel descartou. Sem carimbo, `False`.
        """
        carimbada = self._identidade_da_medicao
        return carimbada is not None and carimbada != self._identidade_da_matriz

    def curva_de_outra_matriz(self) -> bool:
        """Diz se a curva do LOESS em voo foi carimbada com outra matriz que a da tela.

        É a guarda de `recebe_curva_do_loess`, sem receber nada, pelo motivo de
        `medicao_de_outra_matriz`.
        """
        return self._identidade_da_curva != self._identidade_da_matriz

    def resposta_da_medicao_na_tela(self) -> tuple[bool, Any]:
        """`(True, corpo)` da última medição aceita, se ela for da matriz corrente.

        A janela o lê para reescrever o rótulo da medição quando a matriz muda: o rótulo e as
        barras são as duas metades da mesma resposta e seguem a mesma identidade. Devolve
        `(False, None)` quando não há; o par existe porque um corpo `None` aceito é possível.
        """
        if self._ha_resposta_da_medicao and self._medicao_de == self._identidade_da_matriz:
            return True, self._resposta_da_medicao
        return False, None

    def resposta_da_curva_na_tela(self) -> tuple[bool, Any]:
        """`(True, corpo)` da última curva do LOESS aceita, se for da matriz corrente."""
        if self._ha_resposta_da_curva and self._curva_de == self._identidade_da_matriz:
            return True, self._resposta_da_curva
        return False, None

    def carimba_a_medicao(self) -> None:
        """Carimba a medição que está a ser disparada com a matriz corrente. Chamado no disparo.

        Corre depois de `define_matriz` e antes do `POST`, a ordem que `_dispara_medicao` garante:
        `_pedido_de_arquivos` define a matriz, e só então o carimbo copia a identidade vigente.
        Invertida, carimbaria a matriz anterior, e `recebe_medicao` descartaria a resposta certa.

        É método próprio, e não argumento de `define_matriz`, porque aquela corre em todo disparo
        de arquivo (executar, gerar artefato, medir) e esta só na medição. Juntas, carimbariam a
        cada execução, e a medição de A sobreviveria a uma execução de A seguida de troca para B.
        """
        self._identidade_da_medicao = self._identidade_da_matriz

    def carimba_a_curva(self, corretor: str | None) -> None:
        """Carimba a curva do LOESS em voo com a matriz corrente e o `corretor`, no disparo.

        As duas coisas no mesmo instante, como a medição: o corretor lido na chegada seria o do
        combo de depois, e a curva pedida com o LOESS passaria a dizer-se de outro corretor.
        """
        self._identidade_da_curva = self._identidade_da_matriz
        self._corretor_da_curva = corretor

    def recebe_curva_do_loess(self, corpo: Any) -> bool:
        """Guarda a curva do LOESS de `POST /matrizes/curva-do-loess` e refaz a figura.

        Corre para toda resposta da rota, e é o que faz um erro (uma feature que a matriz não tem)
        apagar a curva anterior, em vez de a deixar desenhada sob o pedido seguinte.

        A resposta de outra matriz é descartada: o carimbo do disparo tem de ser a matriz corrente,
        senão a curva de A seria desenhada sob os arquivos de B. A descartada não toca a guardada.

        Devolve se a resposta foi aceita (guardada, ou um erro que apagou a anterior) e `False`
        quando foi descartada, para a janela não ir à figura de uma curva descartada.
        """
        if self.curva_de_outra_matriz():
            return False
        # Guardada com a matriz e o corretor do disparo, para `define_matriz` a esconder e repor
        # pela identidade, como a medição.
        self._resposta_da_curva = corpo
        self._curva_de = self._identidade_da_matriz
        self._ha_resposta_da_curva = True
        self._corretor_da_curva_guardada = self._corretor_da_curva
        self._curva_do_loess = _curva_da_resposta(corpo)
        self.reavalia()
        return True

    def carimba_o_diagnostico(self) -> None:
        """Carimba o diagnóstico em voo com a matriz corrente, no disparo, como a medição."""
        self._identidade_do_diagnostico = self._identidade_da_matriz

    def recebe_etapa(self, etapa: int, corpo: Any, *, entrada: Entrada) -> None:
        """Guarda os resumos de uma prévia do passo a passo e refaz a figura escolhida.

        Cada etapa é dona dos dados das próprias figuras, e só ela os escreve. O corpo da etapa 2
        traz também os resumos da correção, mas só a entrada `corrigida` é a Correção guardada:
        sobre a `original`, a correção do corpo é outra, e escrevê-la poria na aba da Correção o
        par de PCAs de outra correção. `entrada` é a da etapa feita, que a janela lê do estado das
        etapas; para a etapa 1 só existe a `original`.

        A coorte da tabela dos estilos vem do corpo nas duas: a composição é do estudo, medida
        sobre a matriz ingerida, e é a mesma qualquer que seja a entrada.
        """
        if entrada not in ENTRADAS_DA_ETAPA.get(etapa, ()):
            raise ValueError(
                f"«{entrada}» não é entrada da etapa {etapa!r}; as dela são "
                f"{ENTRADAS_DA_ETAPA.get(etapa, ())}"
            )
        corpo = corpo if isinstance(corpo, dict) else {}
        if etapa == 1 or entrada == "corrigida":
            self._pcas_da_correcao = corpo.get("pcas_da_correcao")
            self._rsd_por_feature = corpo.get("rsd_por_feature")
            self._violino_corrigido = self._violino_corrigido or isinstance(
                self._rsd_por_feature, dict
            )
            self._composicao = corpo.get("composicao_por_classe")
            self._fonte_da_etapa[1] = _DONO_ETAPA
        self._atualiza_a_coorte(corpo.get("composicao_por_classe"))
        if etapa == 2:
            self._distribuicoes = corpo.get("distribuicoes")
            self._fonte_da_etapa[2] = _DONO_ETAPA
        self.reavalia()

    def define_frescor_das_etapas(
        self, em_dia: dict[int, bool], *, entradas: Mapping[int, Entrada]
    ) -> None:
        """Recebe o frescor que a janela mede pela fotografia de cada etapa. Refaz só se mudou.

        Recebe também a entrada com que cada etapa foi feita, que a frase de figura desatualizada
        segue para dizer qual etapa refazer (`etapa_a_refazer`): uma Normalização sobre a original
        não depende da Correção, e pedir para refazer a Correção mandaria a pessoa à etapa errada.
        Etapa não feita fica de fora.
        """
        if em_dia == self._etapa_em_dia and entradas == self._entradas_das_etapas:
            return
        self._etapa_em_dia = dict(em_dia)
        self._entradas_das_etapas = dict(entradas)
        self.reavalia()

    def _da_execucao(self, etapa: int, campo: str, corpo: Any, *, corresponde: bool) -> bool:
        """Diz se o corpo da execução deve sobrescrever os dados das figuras da `etapa`.

        Cada etapa é dona dos dados das próprias figuras, e a execução só os escreve quando
        corresponde à etapa guardada; quem diz é a janela, lendo o estado das etapas. Uma Análise
        sobre a matriz original não tem correção a mostrar, e o par de PCAs dela por cima do da
        Correção feita trocaria a figura da aba da Correção sem nada o dizer.

        - Não corresponde: nada se escreve. Dados que uma execução correspondente tinha escrito
          passam à etapa, que os concluiu com ela; senão a figura seguiria a chave desta outra
          execução e desenharia sob ela o que não é dela.
        - Corresponde: um corpo sem o campo (o `202`, um erro) não apaga os dados de uma prévia, e
          a figura da etapa continua na tela até a execução voltar com sucesso. O `200` traz o
          campo, obrigatório em `ExecucaoResposta`, e escreve.
        - Dono: se a etapa já era, continua, e a figura segue o frescor dela; a Análise que a
          reproduz traz os mesmos dados. Passá-la à execução faria a figura da Correção sumir da
          aba dela enquanto uma Análise sobre a corrigida corre, e mudar o corretor não a
          desatualizaria. Como a validade própria não muda com a chave, a figura é refeita em
          `define_execucao`, depois de todos os `recebe_*`. Sem dono, a execução passa a ser, e a
          figura segue a chave dela.
        """
        corpo = corpo if isinstance(corpo, dict) else {}
        dono = self._fonte_da_etapa.get(etapa)
        if not corresponde:
            if dono == _DONO_EXECUCAO:
                self._fonte_da_etapa[etapa] = _DONO_ETAPA
            return False
        if dono == _DONO_ETAPA:
            if campo not in corpo:
                return False
            self._refazer_as_figuras_das_etapas = True
            return True
        self._fonte_da_etapa[etapa] = _DONO_EXECUCAO
        return True

    def _validade_propria(self, construtor: ConstrutorDeFigura) -> EstadoDaFigura | None:
        """O estado das figuras que não seguem a chave da execução.

        O violino de RSD por feature é figura do diagnóstico até chegar o primeiro resultado de
        corretor da matriz corrente; depois segue a regra das etapas e da execução.

        As figuras das etapas 1 e 2 têm dono. Escritas pela prévia, valem pelo frescor da etapa;
        escritas por uma execução que corresponde à etapa, seguem a chave dela; sem dono, esperam a
        etapa (`SEM_EXECUCAO`, com a frase de `frase_da_etapa_sem_resultado`), e não a chave de uma
        execução que não as escreveu.
        """
        if construtor.funcao == FIGURA_DA_CURVA_DO_LOESS:
            # A curva vale enquanto for a da matriz corrente, e não segue a execução: sem curva
            # pedida, o estado é o de nenhuma resposta, com a frase que diz onde pedir.
            if isinstance(self._curva_do_loess, dict):
                return EstadoDaFigura.DESENHADA
            return EstadoDaFigura.SEM_EXECUCAO
        violino_da_ingerida = (
            construtor.funcao == figura_do_violino_do_rsd_por_feature.__name__
            and not self._violino_corrigido
        )
        if construtor.funcao in FIGURAS_DO_DIAGNOSTICO or violino_da_ingerida:
            if self._diagnostico_vale():
                return EstadoDaFigura.DESENHADA
            return EstadoDaFigura.SEM_DIAGNOSTICO
        etapa = ETAPA_DA_FIGURA.get(construtor.funcao)
        if etapa is None:
            return None
        dono = self._fonte_da_etapa.get(etapa)
        if dono == _DONO_EXECUCAO:
            return None
        if dono != _DONO_ETAPA:
            return EstadoDaFigura.SEM_EXECUCAO
        if self._etapa_em_dia.get(etapa, False):
            return EstadoDaFigura.DESENHADA
        return EstadoDaFigura.ETAPA_DESATUALIZADA

    def _frase_propria(self, construtor: ConstrutorDeFigura, estado: EstadoDaFigura) -> str | None:
        """A frase de figura de etapa desatualizada, que nomeia a etapa a refazer.

        A etapa a refazer é a mais a montante fora de dia na cadeia da entrada usada, e não a da
        figura nem a ordem 1-2-3. Uma Normalização feita sobre a corrigida depende da Correção:
        mudar o corretor desatualiza as duas, e as distribuições pedem a Correção. Uma feita sobre
        a original não depende, e as distribuições pedem a própria Normalização. As entradas vêm da
        janela, com o frescor (`define_frescor_das_etapas`).

        A figura de etapa sem dono espera a etapa, com a frase dela, e não a do pipeline.
        """
        if construtor.funcao == FIGURA_DA_CURVA_DO_LOESS and estado is EstadoDaFigura.SEM_EXECUCAO:
            return FRASE_SEM_CURVA_DO_LOESS
        etapa = ETAPA_DA_FIGURA.get(construtor.funcao)
        if etapa is None:
            return None
        if estado is EstadoDaFigura.SEM_EXECUCAO and self._fonte_da_etapa.get(etapa) is None:
            return frase_da_etapa_sem_resultado(etapa)
        if estado is not EstadoDaFigura.ETAPA_DESATUALIZADA:
            return None
        refazer = etapa_a_refazer(etapa, self._etapa_em_dia, self._entradas_das_etapas)
        return frase_da_etapa_desatualizada(etapa, refazer)

    def _diagnostico_vale(self) -> bool:
        """Diz se o diagnóstico guardado é o da matriz corrente."""
        # A igualdade, e não 'os dois preenchidos': em produção o disparo sempre registra a matriz
        # antes de o diagnóstico chegar, e um painel montado sem matriz nenhuma (o dos testes)
        # compara `None` com `None`, porque o diagnóstico é o da única matriz que ele conhece.
        return self._ha_diagnostico and self._diagnostico_de == self._identidade_da_matriz

    def define_matriz(self, identidade: str | None) -> bool:
        """Define a matriz com que a janela está a trabalhar. Devolve se ela mudou.

        A medição e a curva do LOESS de outra matriz ficam escondidas, e a volta à matriz delas as
        repõe, como o diagnóstico. A janela chama isto quando a pessoa escolhe outros arquivos ou
        muda o modo, que muda a cada tecla; apagar custaria a medição, que leva minutos, a quem
        corrigisse o modo e voltasse ao mesmo texto. Com o retorno, a janela reescreve junto o
        rótulo da medição e o da curva.

        A medição é a única carga sem chave de invalidação: chega de rota própria e o corpo não traz
        chave. Sem esta comparação, 'arquivo A, medir, arquivo B, executar' desenharia as barras de
        A sob a execução de B. As outras cargas já têm quem as invalide e ficam. A identidade é
        `interface/fluxo.chave_da_entrada` com `metodo=None`: a medição corre os três corretores, e
        pôr o
        método aqui a esconderia no gesto que ela serve (medir, escolher o vencedor, executar).

        O diagnóstico é reavaliado, e não descartado: a regra vive em `_diagnostico_vale`, que só
        `_troca_a_figura` lê, e sem a reavaliação a deriva e as PCAs da matriz anterior ficariam na
        tela, exportáveis, com outra matriz.
        """
        if identidade == self._identidade_da_matriz:
            return False
        self._identidade_da_matriz = identidade
        # Esconder e repor, e não apagar: a medição e a curva guardadas ficam, cada uma com a sua
        # matriz, e o que se vê é a da matriz que chega.
        tinha_curva = self._curva_do_loess is not None
        self._medicao = _barras_da_medicao(self.resposta_da_medicao_na_tela()[1])
        self._curva_do_loess = _curva_da_resposta(self.resposta_da_curva_na_tela()[1])
        # A matriz nova ainda não foi corrigida, então o violino volta a ser o da ingerida dela
        # quando o diagnóstico chegar. Reavalia também quando ele seguia a execução, senão o violino
        # corrigido da matriz anterior ficaria na tela até a resposta seguinte.
        estava_corrigido, self._violino_corrigido = self._violino_corrigido, False
        curva_mudou = tinha_curva != (self._curva_do_loess is not None)
        if self._ha_diagnostico or estava_corrigido or curva_mudou:
            self.reavalia()
        return True

    # ------------------------------------------------------------------
    # A fonte do `PainelDeFigura`
    # ------------------------------------------------------------------

    def _montadores(self) -> dict[str, Callable[[], tuple[Figure, tuple[str, ...]]]]:
        """Que construtor do registo este painel sabe alimentar, e com que método.

        A chave é o `__name__` da função de `plots/`, e não uma string escrita à mão, para o nome
        viver num lugar só. É método porque os valores são métodos ligados a `self`, e é privado
        porque o único leitor fora de `_constroi_figura` é o teste da bijeção com o registo.
        """
        return {
            figura_da_deriva_do_qc.__name__: self._trajetoria_do_qc,
            figura_das_caixas_por_feature.__name__: self._caixas_da_distribuicao,
            figura_das_caixas_por_amostra.__name__: self._caixas_por_amostra,
            figura_da_densidade_por_feature.__name__: self._densidade_por_feature,
            figura_da_densidade_por_amostra.__name__: self._densidade_por_amostra,
            figura_do_volcano.__name__: self._volcano_do_par,
            figura_do_fold_change.__name__: self._fold_change_do_par,
            figura_dos_testes_t.__name__: self._testes_t_do_par,
            figura_da_anova.__name__: self._anova_dos_grupos,
            figura_do_scree.__name__: self._scree,
            figura_dos_scores.__name__: self._scores_da_pca,
            figura_das_cargas.__name__: self._cargas_da_pca,
            figura_do_biplot.__name__: self._biplot,
            figura_dos_scores_do_plsda.__name__: self._scores_do_plsda,
            figura_das_cargas_do_plsda.__name__: self._cargas_do_plsda,
            figura_do_vip.__name__: self._vip,
            figura_do_painel_da_correcao.__name__: self._painel_da_correcao,
            figura_do_violino_do_rsd_por_feature.__name__: self._violino_do_rsd,
            figura_da_composicao_por_classe.__name__: self._composicao_por_classe,
            figura_dos_escores_um_contra_resto.__name__: self._escores_multiclasse,
            figura_das_cargas_um_contra_resto.__name__: self._cargas_multiclasse,
            figura_da_importancia_um_contra_resto.__name__: self._importancia_multiclasse,
            figura_do_dendrograma.__name__: self._dendrograma,
            figura_do_mapa_de_calor.__name__: self._mapa_de_calor,
            figura_do_mapa_de_correlacao.__name__: self._mapa_de_correlacao,
            figura_da_busca_por_padrao.__name__: self._busca_de_padrao,
            figura_do_kmeans.__name__: self._kmeans_das_amostras,
            figura_do_mapa_auto_organizavel.__name__: self._som_das_amostras,
            figura_do_random_forest.__name__: self._floresta_das_amostras,
            figura_do_svm.__name__: self._svm_das_amostras,
            figura_da_rede_de_correlacao_parcial.__name__: self._rede_das_features,
            FIGURA_DA_CURVA_DO_LOESS: self._curva_do_loess_da_feature,
        }

    def _constroi_figura(self, construtor: ConstrutorDeFigura) -> tuple[Figure, tuple[str, ...]]:
        """A `Fonte` de `PainelDeFigura`: o corpo guardado vira `Figure`.

        A janela não recalcula estatística: os vetores vêm do fio, e este método escolhe o
        construtor e lhe passa o que chegou. `MatrizVaziaError` sobe de propósito: é a exceção que
        `_troca_a_figura` apanha e mostra como população vazia, que é a verdade sem diagnóstico.

        `FonteAusenteError` é inalcançável em produção pela bijeção de `_montadores`, e o ramo fica:
        um recuo para a figura de repouso faria um construtor sem montador não desenhar nada, em
        silêncio.
        """
        montadores = self._montadores()
        montador = montadores.get(construtor.funcao)
        if montador is None:
            raise FonteAusenteError(
                f"{construtor.funcao!r} está no registo e este painel não sabe construí-lo. "
                f"Ele alimenta {sorted(montadores)}, e o seletor é derivado de "
                "`plots.CONSTRUTORES` — chegar aqui significa que o registo ganhou uma figura e "
                "este painel não ganhou o montador dela"
            )
        return montador()

    def _curva_do_loess_da_feature(self) -> tuple[Figure, tuple[str, ...]]:
        """A curva do LOESS da feature pedida, com o envelope sem um QC.

        O segundo elemento é `()`: a figura é de uma feature, e os pontos são injeções; a
        identificação vai pelo rótulo da série. A base chama-se `curva_do_loess` porque
        `RAIZES_ANINHADAS` casa cada base de leitura com o esquema da rota pelo nome
        (`CurvaDoLoessResposta`).
        """
        curva_do_loess = self._curva_do_loess if isinstance(self._curva_do_loess, dict) else {}
        curva = curva_do_fio(
            feature=curva_do_loess.get("feature"),
            ordens_de_injecao=curva_do_loess.get("ordens_de_injecao"),
            lotes=curva_do_loess.get("lotes"),
            qc=curva_do_loess.get("qc"),
            valores=curva_do_loess.get("valores"),
            curva=curva_do_loess.get("curva"),
            curva_preenchida=curva_do_loess.get("curva_preenchida"),
            curva_sorteada=curva_do_loess.get("curva_sorteada"),
            qc_no_ajuste=curva_do_loess.get("qc_no_ajuste"),
            motivo_sem_curva=curva_do_loess.get("motivo_sem_curva"),
            predicao_sem_o_qc=curva_do_loess.get("predicao_sem_o_qc"),
            motivo_sem_predicao=curva_do_loess.get("motivo_sem_predicao"),
            envelope_inferior=curva_do_loess.get("envelope_inferior"),
            envelope_superior=curva_do_loess.get("envelope_superior"),
            envelope_nao_ajustado=curva_do_loess.get("envelope_nao_ajustado"),
            n_curvas_do_envelope=curva_do_loess.get("n_curvas_do_envelope"),
            motivo_sem_envelope=curva_do_loess.get("motivo_sem_envelope"),
            semente=curva_do_loess.get("semente"),
            spans_por_lote=curva_do_loess.get("spans_por_lote"),
            corretor=self._corretor_da_curva_guardada,
        )
        recorte = recorte_da_curva_do_loess(
            curva=curva, n_features_totais=curva_do_loess.get("n_features")
        )
        return figura_da_curva_do_loess(curva=curva, recorte=recorte), ()

    def _trajetoria_do_qc(self) -> tuple[Figure, tuple[str, ...]]:
        """A trajetória dos QCs: o corpo guardado de `/matrizes/diagnosticar` vira `Figure`.

        O segundo elemento é `()` porque o vetor de nomes serve para dizer qual feature é um ponto,
        e esta figura põe um ponto por QC. `PainelDeFigura.desenha` identifica então pelo rótulo da
        série, que aqui é o lote.
        """
        trajetoria = self._trajetoria if isinstance(self._trajetoria, dict) else {}
        figura = figura_da_deriva_do_qc(
            scores=trajetoria.get("scores") or (),
            ordens_de_injecao=trajetoria.get("ordens_de_injecao") or (),
            lotes=trajetoria.get("lotes") or (),
            recorte=recorte_da_deriva(
                n_features_usadas=trajetoria.get("n_features_usadas"),
                n_features_totais=trajetoria.get("n_features_totais"),
            ),
        )
        return figura, ()

    # ------------------------------------------------------------------
    # As quatro distribuições
    # ------------------------------------------------------------------
    #
    # Os quatro montadores leem a mesma raiz e nenhum recalcula nada: escolhem que vetores do
    # resumo alimentam que construtor e passam o `Recorte` que só o servidor sabe montar (as duas
    # pontas da legenda, a semente e o teto).
    #
    # `COEFICIENTE_DE_TUKEY` vem de `esquemas.py` e é injetado no recorte. `plots/` não pode
    # importar `refino.api`, e escrever `1,5` do lado do desenho criaria uma segunda fonte do
    # número com que o servidor calcula o bigode. Este painel pode importar os dois.

    def _resumo(self) -> dict[str, Any]:
        """O resumo guardado, ou um dicionário vazio, que cai na recusa por população vazia."""
        return self._distribuicoes if isinstance(self._distribuicoes, dict) else {}

    def _caixas_da_distribuicao(self) -> tuple[Figure, tuple[str, ...]]:
        """As caixas de Tukey por feature, antes e depois.

        O segundo elemento é `()` porque `PainelDeFigura` indexa o vetor de nomes pela posição que
        o `pick_event` traz, e num `bxp` essa posição é a do ponto dentro do artista, não a da
        feature: o nome sairia errado com aparência de certo. Quem identifica é o rótulo de cada
        artista, que `plots/distribuicao.rotulo_do_artista` escreve com o nome da feature.
        """
        distribuicoes = self._resumo()
        figura = figura_das_caixas_por_feature(
            antes=caixas_do_fio(distribuicoes.get("caixas_antes")),
            depois=caixas_do_fio(distribuicoes.get("caixas_depois")),
            recorte=recorte_das_caixas(
                n_caixas=distribuicoes.get("n_caixas"),
                n_comparadas=distribuicoes.get("n_features_comparadas"),
                n_antes=distribuicoes.get("n_features_antes"),
                amostrada=distribuicoes.get("amostrada"),
                seed=distribuicoes.get("seed"),
                teto_de_caixas=distribuicoes.get("teto_de_caixas"),
                coeficiente_de_tukey=COEFICIENTE_DE_TUKEY,
                matriz_antes=distribuicoes.get("matriz_antes"),
                matriz_depois=distribuicoes.get("matriz_depois"),
            ),
            matriz_antes=distribuicoes.get("matriz_antes"),
            matriz_depois=distribuicoes.get("matriz_depois"),
        )
        return figura, ()

    def _caixas_por_amostra(self) -> tuple[Figure, tuple[str, ...]]:
        """As caixas de Tukey por amostra, antes e depois, como a `Sample View` do MetaboAnalyst.

        A carga é `caixas_por_amostra_*`, e não `medias_por_amostra_*`: os dois campos têm a mesma
        cardinalidade, e trocá-los não levanta; a figura sairia bem formada, com uma caixa por
        amostra feita de um número só. O segundo elemento é `()` pela razão das caixas por feature.
        """
        distribuicoes = self._resumo()
        figura = figura_das_caixas_por_amostra(
            antes=caixas_por_amostra_do_fio(distribuicoes.get("caixas_por_amostra_antes")),
            depois=caixas_por_amostra_do_fio(distribuicoes.get("caixas_por_amostra_depois")),
            recorte=recorte_das_caixas_por_amostra(
                n_caixas=distribuicoes.get("n_caixas_por_amostra"),
                n_amostras_comparadas=distribuicoes.get("n_amostras_comparadas"),
                n_comparadas=distribuicoes.get("n_features_comparadas"),
                n_antes=distribuicoes.get("n_features_antes"),
                amostrada=distribuicoes.get("amostrada_por_amostra"),
                seed=distribuicoes.get("seed"),
                teto_de_caixas=distribuicoes.get("teto_de_caixas"),
                coeficiente_de_tukey=COEFICIENTE_DE_TUKEY,
                matriz_antes=distribuicoes.get("matriz_antes"),
                matriz_depois=distribuicoes.get("matriz_depois"),
            ),
            matriz_antes=distribuicoes.get("matriz_antes"),
            matriz_depois=distribuicoes.get("matriz_depois"),
        )
        return figura, ()

    def _densidade_por_feature(self) -> tuple[Figure, tuple[str, ...]]:
        """A densidade das médias por feature: uma média por coluna, nunca por célula."""
        distribuicoes = self._resumo()
        figura = figura_da_densidade_por_feature(
            antes=vetor_do_fio(distribuicoes.get("medias_por_feature_antes")),
            depois=vetor_do_fio(distribuicoes.get("medias_por_feature_depois")),
            recorte=recorte_da_densidade_por_feature(
                n_comparadas=distribuicoes.get("n_features_comparadas"),
                n_antes=distribuicoes.get("n_features_antes"),
                matriz_antes=distribuicoes.get("matriz_antes"),
                matriz_depois=distribuicoes.get("matriz_depois"),
            ),
            matriz_antes=distribuicoes.get("matriz_antes"),
            matriz_depois=distribuicoes.get("matriz_depois"),
        )
        return figura, ()

    def _densidade_por_amostra(self) -> tuple[Figure, tuple[str, ...]]:
        """A densidade das médias por amostra: uma média por linha, nunca por célula."""
        distribuicoes = self._resumo()
        figura = figura_da_densidade_por_amostra(
            antes=vetor_do_fio(distribuicoes.get("medias_por_amostra_antes")),
            depois=vetor_do_fio(distribuicoes.get("medias_por_amostra_depois")),
            recorte=recorte_da_densidade_por_amostra(
                n_comparadas=distribuicoes.get("n_features_comparadas"),
                n_antes=distribuicoes.get("n_features_antes"),
                n_amostras=distribuicoes.get("n_amostras"),
                matriz_antes=distribuicoes.get("matriz_antes"),
                matriz_depois=distribuicoes.get("matriz_depois"),
            ),
            matriz_antes=distribuicoes.get("matriz_antes"),
            matriz_depois=distribuicoes.get("matriz_depois"),
        )
        return figura, ()

    # ------------------------------------------------------------------
    # O volcano, a primeira figura deste painel que devolve nomes
    # ------------------------------------------------------------------

    def _volcano_do_par(self) -> tuple[Figure, tuple[str, ...]]:
        """O volcano do par de grupos declarado. Aqui o segundo elemento não é `()`.

        Esta figura põe um ponto por feature num artista só, então o vetor de nomes serve: sem ele,
        `PainelDeFigura._identifica` cairia no rótulo do artista e devolveria o mesmo texto para
        todos os pontos. Os nomes vêm da mesma seleção que desenhou os pontos (`pontos_do_volcano`
        devolve coordenadas e nomes na mesma ordem); uma segunda filtragem aqui faria a
        identificação devolver o nome do vizinho, com a figura bem formada.

        O `alfa` é lido uma vez e atravessa por `contraste.alfa`, porque o que classifica os pontos
        e o que a legenda escreve têm de ser o mesmo número. As leituras de chave ficam em
        `desktop/janela/`, onde o teste de contrato as confere contra o esquema; um `n_grupo_A` mal
        escrito em `plots/` devolveria `0` para sempre sem nada levantar.
        """
        volcano = self._volcano if isinstance(self._volcano, dict) else {}
        contraste = contraste_do_fio(
            grupo_a=volcano.get("grupo_a"),
            grupo_b=volcano.get("grupo_b"),
            n_grupo_a=volcano.get("n_grupo_a"),
            n_grupo_b=volcano.get("n_grupo_b"),
            parametrico=volcano.get("parametrico"),
            alfa_fdr=volcano.get("alfa_fdr"),
            correcao=volcano.get("correcao"),
            rota_da_normalidade=volcano.get("rota_da_normalidade"),
            shapiro_pvalue=volcano.get("shapiro_pvalue"),
            n_residuos=volcano.get("n_residuos"),
            matriz_do_pvalue=volcano.get("matriz_do_pvalue"),
            matriz_do_fold_change=volcano.get("matriz_do_fold_change"),
            matrizes_diferem=volcano.get("matrizes_diferem"),
        )
        pontos = pontos_do_volcano(
            features=volcano.get("features"),
            log2fc=volcano.get("log2fc"),
            pvalue=volcano.get("pvalue"),
            qvalue_fdr=volcano.get("qvalue_fdr"),
            n_features_testadas=volcano.get("n_features_testadas"),
            alfa=contraste.alfa,
        )
        intervalos = self._intervalos_do_univariado(pontos.n_testadas)
        figura = figura_do_volcano(
            pontos=pontos,
            contraste=contraste,
            recorte=recorte_do_volcano(pontos, contraste, intervalos),
            intervalos=intervalos,
        )
        return figura, pontos.nomes

    # ------------------------------------------------------------------
    # As três univariadas e a ANOVA de N grupos
    # ------------------------------------------------------------------
    #
    # `_contraste_do_univariado` duplica a montagem do contraste de `_volcano_do_par`; a docstring
    # dela diz por que as duas ainda não foram fundidas.

    def _fold_change_do_par(self) -> tuple[Figure, tuple[str, ...]]:
        """O fold change por feature. Devolve nomes, como o volcano.

        Um ponto por feature num artista só, então o vetor de nomes serve para a identificação dizer
        qual feature é. Os nomes vêm da mesma seleção que desenhou os pontos, e o `alfa` é lido uma
        vez e atravessa por `contraste.alfa`, pelos motivos de `_volcano_do_par`.
        """
        # A variável chama-se `volcano` porque é o nome da raiz, e o teste de contrato lê o nome de
        # quem recebe o `.get`.
        volcano = self._univariado()
        contraste = self._contraste_do_univariado()
        pontos = pontos_do_fold_change(
            features=volcano.get("features"),
            log2fc=volcano.get("log2fc"),
            qvalue_fdr=volcano.get("qvalue_fdr"),
            n_features_testadas=volcano.get("n_features_testadas"),
            alfa=contraste.alfa,
        )
        intervalos = self._intervalos_do_univariado(pontos.n_testadas)
        figura = figura_do_fold_change(
            pontos=pontos,
            contraste=contraste,
            recorte=recorte_do_fold_change(pontos, contraste, intervalos),
            intervalos=intervalos,
        )
        return figura, pontos.nomes

    def _testes_t_do_par(self) -> tuple[Figure, tuple[str, ...]]:
        """O `-log10(p bruto)` por feature, com a mesma raiz e o mesmo contraste do volcano.

        O eixo é o `p` bruto, e não o `q`, como no volcano: as duas figuras desenham o mesmo vetor,
        e quantidades diferentes as fariam discordar sobre a altura da mesma feature. O argumento
        está em `plots/univariada.py`.
        """
        volcano = self._univariado()
        contraste = self._contraste_do_univariado()
        pontos = pontos_dos_testes_t(
            features=volcano.get("features"),
            pvalue=volcano.get("pvalue"),
            qvalue_fdr=volcano.get("qvalue_fdr"),
            n_features_testadas=volcano.get("n_features_testadas"),
            alfa=contraste.alfa,
        )
        figura = figura_dos_testes_t(
            pontos=pontos,
            contraste=contraste,
            recorte=recorte_dos_testes_t(pontos, contraste),
        )
        return figura, pontos.nomes

    def _univariado(self) -> dict[str, Any]:
        """A raiz `volcano` do último `/execucoes`, ou um dicionário vazio.

        Uma leitura só para as três figuras univariadas, que consomem a mesma raiz.
        """
        return self._volcano if isinstance(self._volcano, dict) else {}

    def _intervalos_do_univariado(self, n_testadas: int) -> Any:
        """O intervalo do fold change nas features escolhidas.

        Uma leitura para o volcano e para o fold change, que desenham o mesmo `log2fc` e põem a
        barra nas mesmas escolhidas; duas cópias poderiam pôr barras em features diferentes. As
        chaves são lidas aqui, onde o teste de contrato as confere contra `ResumoDoVolcano`; mal
        escritas em `plots/`, devolveriam `None` para sempre. `n_testadas` vem da figura que chama,
        porque é a população que ela já declara: a seleção estica os vetores a esse número e o usa
        como o `m` do nível.
        """
        volcano = self._univariado()
        return intervalos_escolhidos(
            pvalue=volcano.get("pvalue"),
            log2fc_ci_lower=volcano.get("log2fc_ci_lower"),
            log2fc_ci_upper=volcano.get("log2fc_ci_upper"),
            fold_change_ci_undefined=volcano.get("fold_change_ci_undefined"),
            nivel_do_intervalo=volcano.get("nivel_do_intervalo"),
            n_testadas=n_testadas,
        )

    def _contraste_do_univariado(self) -> Any:
        """O `Contraste` das figuras univariadas, com as leituras de chave feitas aqui.

        O teste de contrato confere cada chave lida em `desktop/janela/` contra o esquema da raiz;
        um `n_grupo_A` mal escrito em `plots/` devolveria `0` para sempre sem nada levantar.

        `_volcano_do_par` monta o mesmo objeto a partir das mesmas chaves, e fundir as duas é uma
        linha (o volcano passaria a chamar isto). Falta ver a fusão reprovar nos dois sentidos
        contra o teste do volcano sobre o fio real (`tests/unit/test_plots_volcano.py`).
        """
        volcano = self._univariado()
        return contraste_do_fio(
            grupo_a=volcano.get("grupo_a"),
            grupo_b=volcano.get("grupo_b"),
            n_grupo_a=volcano.get("n_grupo_a"),
            n_grupo_b=volcano.get("n_grupo_b"),
            parametrico=volcano.get("parametrico"),
            alfa_fdr=volcano.get("alfa_fdr"),
            correcao=volcano.get("correcao"),
            rota_da_normalidade=volcano.get("rota_da_normalidade"),
            shapiro_pvalue=volcano.get("shapiro_pvalue"),
            n_residuos=volcano.get("n_residuos"),
            matriz_do_pvalue=volcano.get("matriz_do_pvalue"),
            matriz_do_fold_change=volcano.get("matriz_do_fold_change"),
            matrizes_diferem=volcano.get("matrizes_diferem"),
        )

    def _anova_dos_grupos(self) -> tuple[Figure, tuple[str, ...]]:
        """A ANOVA sobre todos os grupos de estudo, com raiz e população próprias.

        Não lê `volcano`: o volcano descreve o par de grupos declarado, e esta todos os grupos que a
        matriz tem. Ler os dois da mesma raiz poria o `n` de dois grupos no rodapé de uma figura
        sobre quatro.

        Devolve nomes, como as univariadas: um ponto por feature, num artista de dado só, marcado
        com `GID_DAS_FEATURES`. A linha do corte é o segundo artista e não é de dado; sem a marca,
        um clique nela devolveria o nome da primeira feature. Toda conta da ANOVA vive em
        `plots/anova.py`.
        """
        # O nome da variável é a base que o teste de contrato lê: ele tira a base do nome de quem
        # recebe o `.get` e confere cada chave contra o esquema daquela raiz. Chamar-lhe `corpo`
        # atribuiria as leituras à raiz do corpo de resposta.
        anova = self._anova if isinstance(self._anova, dict) else {}
        contexto = contexto_do_fio(
            grupos=anova.get("grupos"),
            n_por_grupo=anova.get("n_por_grupo"),
            metodo=anova.get("metodo"),
            post_hoc_metodo=anova.get("post_hoc_metodo"),
            post_hoc_rota=anova.get("post_hoc_rota"),
            alfa_fdr=anova.get("alfa_fdr"),
            correcao=anova.get("correcao"),
            matriz=anova.get("matriz"),
            min_por_grupo=anova.get("min_por_grupo"),
        )
        pontos = pontos_da_anova(
            features=anova.get("features"),
            pvalue=anova.get("pvalue"),
            qvalue_fdr=anova.get("qvalue_fdr"),
            n_features_testadas=anova.get("n_features_testadas"),
            alfa=contexto.alfa,
        )
        figura = figura_da_anova(
            pontos=pontos,
            contexto=contexto,
            recorte=recorte_da_anova(pontos, contexto),
        )
        return figura, pontos.nomes

    # ------------------------------------------------------------------
    # Os cinco sem caminho de fio: registados, com recusa por população vazia
    # ------------------------------------------------------------------
    #
    # Existem para o combo não oferecer entrada que quebra. Sem montador, a entrada levantaria
    # `FonteAusenteError` ao ser escolhida, que o painel descreve como defeito de ligação do
    # programa; o que se passa é que nenhuma rota devolve ainda o que eles consomem, e 'população
    # vazia' é a frase verdadeira. Quando a rota existir, muda a raiz e não o montador.

    def _escores_multiclasse(self) -> tuple[Figure, tuple[str, ...]]:
        """Os scores do um-contra-resto. Recusa por coleção vazia enquanto não houver rota."""
        plsda_multiclasse = (
            self._plsda_multiclasse if isinstance(self._plsda_multiclasse, dict) else {}
        )
        modelos = tuple(plsda_multiclasse.get("escores") or ())
        recorte = recorte_dos_escores_um_contra_resto(
            modelos=modelos, n_features_totais=plsda_multiclasse.get("n_features_totais")
        )
        figura = figura_dos_escores_um_contra_resto(
            modelos=modelos, recorte=recorte, estilos=self.tabela_dos_grupos.estilos()
        )
        return figura, ()

    def _cargas_multiclasse(self) -> tuple[Figure, tuple[str, ...]]:
        """As cargas do um-contra-resto. Mesma raiz e mesma recusa dos scores."""
        plsda_multiclasse = (
            self._plsda_multiclasse if isinstance(self._plsda_multiclasse, dict) else {}
        )
        modelos = tuple(plsda_multiclasse.get("cargas") or ())
        recorte = recorte_das_cargas_um_contra_resto(
            modelos=modelos, n_features_totais=plsda_multiclasse.get("n_features_totais")
        )
        return figura_das_cargas_um_contra_resto(modelos=modelos, recorte=recorte), ()

    def _importancia_multiclasse(self) -> tuple[Figure, tuple[str, ...]]:
        """A importância do um-contra-resto. Mesma raiz e mesma recusa."""
        plsda_multiclasse = (
            self._plsda_multiclasse if isinstance(self._plsda_multiclasse, dict) else {}
        )
        modelos = tuple(plsda_multiclasse.get("importancia") or ())
        recorte = recorte_da_importancia_um_contra_resto(
            modelos=modelos, n_desenhadas_por_modelo=FEATURES_DA_IMPORTANCIA
        )
        return (
            figura_da_importancia_um_contra_resto(
                modelos=modelos, n_features=FEATURES_DA_IMPORTANCIA, recorte=recorte
            ),
            (),
        )

    def _dendrograma(self) -> tuple[Figure, tuple[str, ...]]:
        """O dendrograma das amostras. Recusa por ligação inválida enquanto não houver rota."""
        agrupamento = self._agrupamento if isinstance(self._agrupamento, dict) else {}
        arvore = arvore_do_fio(
            ligacao=agrupamento.get("ligacao"),
            rotulos=agrupamento.get("rotulos"),
            metrica=agrupamento.get("metrica"),
            ligacao_metodo=agrupamento.get("ligacao_metodo"),
        )
        recorte = recorte_do_dendrograma(
            arvore=arvore,
            n_features_usadas=agrupamento.get("n_features_usadas", 0),
            n_features_totais=agrupamento.get("n_features_totais"),
            matriz=agrupamento.get("matriz", ""),
            amostragem=agrupamento.get("amostragem", ""),
        )
        return figura_do_dendrograma(
            arvore=arvore,
            grupos=agrupamento.get("grupos", ()),
            recorte=recorte,
            estilos=self.tabela_dos_grupos.estilos(),
        ), ()

    def _mapa_de_correlacao(self) -> tuple[Figure, tuple[str, ...]]:
        """O mapa de correlação entre features.

        O segundo elemento é `()`: o vetor de nomes pressupõe um ponto por feature, e esta figura
        tem células. `PainelDeFigura.desenha` identifica então pelo rótulo da série. O recorte, a
        semente e a contagem de indefinidas são do servidor, que sabe se houve sorteio e contra que
        teto.
        """
        correlacao = self._correlacao if isinstance(self._correlacao, dict) else {}
        mapa = mapa_de_correlacao_do_fio(
            valores=correlacao.get("valores"),
            features=correlacao.get("features"),
            metodo=correlacao.get("metodo"),
        )
        recorte = recorte_do_mapa_de_correlacao(
            mapa=mapa,
            n_features_totais=correlacao.get("n_features_totais"),
            n_amostras=correlacao.get("n_amostras"),
            n_indefinidas=correlacao.get("n_indefinidas"),
            matriz=correlacao.get("matriz", ""),
            amostragem=correlacao.get("amostragem", ""),
        )
        return figura_do_mapa_de_correlacao(mapa=mapa, recorte=recorte), ()

    def _busca_de_padrao(self) -> tuple[Figure, tuple[str, ...]]:
        """A busca por padrão contra o perfil de referência.

        `FEATURES_DA_BUSCA` é injetado daqui, e não lido pelo construtor, porque quantas features
        desenhar é decisão do ponto de chamada, como `FEATURES_DA_IMPORTANCIA` no multiclasse. O
        construtor recebe a lista inteira, que é o que a rota devolve.
        """
        # A base chama-se `busca_por_padrao` porque `RAIZES_ANINHADAS` casa cada base de leitura com
        # um esquema do OpenAPI pelo nome; uma local de outro nome reprovaria por classe não
        # declarada.
        busca_por_padrao = (
            self._busca_por_padrao if isinstance(self._busca_por_padrao, dict) else {}
        )
        busca = busca_do_fio(
            features=busca_por_padrao.get("features"),
            coeficiente=busca_por_padrao.get("coeficiente"),
            referencia=busca_por_padrao.get("referencia", ""),
            tipo_da_referencia=busca_por_padrao.get("tipo_da_referencia", ""),
            perfil=busca_por_padrao.get("perfil", ()),
            ordem_dos_grupos=busca_por_padrao.get("ordem_dos_grupos", ()),
            metodo=busca_por_padrao.get("metodo"),
        )
        recorte = recorte_da_busca_por_padrao(
            busca=busca,
            n_desenhadas=min(FEATURES_DA_BUSCA, len(busca.features)),
            n_features_totais=busca_por_padrao.get("n_features_totais"),
            n_amostras=busca_por_padrao.get("n_amostras"),
            n_indefinidas=busca_por_padrao.get("n_indefinidas"),
            matriz=busca_por_padrao.get("matriz", ""),
        )
        return (
            figura_da_busca_por_padrao(busca=busca, n_features=FEATURES_DA_BUSCA, recorte=recorte),
            (),
        )

    def _kmeans_das_amostras(self) -> tuple[Figure, tuple[str, ...]]:
        """A partição particional das amostras.

        O segundo elemento é `()`: esta figura tem um segmento por `(grupo, agrupamento)`, e não um
        ponto por feature, então identifica pelo rótulo da série. A partição, a inércia e a semente
        derivada são do servidor, que correu o método.
        """
        kmeans = self._kmeans if isinstance(self._kmeans, dict) else {}
        particao = kmeans_do_fio(
            rotulos=kmeans.get("rotulos"),
            grupos=kmeans.get("grupos", ()),
            k=kmeans.get("k"),
            inercia=kmeans.get("inercia"),
        )
        recorte = recorte_do_kmeans(
            particao=particao,
            n_features_usadas=kmeans.get("n_features_usadas", 0),
            n_features_totais=kmeans.get("n_features_totais"),
            matriz=kmeans.get("matriz", ""),
            amostragem=kmeans.get("amostragem", ""),
            semente=kmeans.get("semente"),
        )
        return figura_do_kmeans(
            particao=particao, recorte=recorte, estilos=self.tabela_dos_grupos.estilos()
        ), ()

    def _som_das_amostras(self) -> tuple[Figure, tuple[str, ...]]:
        """O mapa auto-organizável das amostras.

        `linhas` e `colunas` vêm do corpo e nunca são inferidas do maior índice: uma grade cuja
        última coluna ficou vazia sairia com uma coluna a menos, bem formada, e a contagem de
        unidades vazias do rodapé não bateria com a imagem.
        """
        som = self._som if isinstance(self._som, dict) else {}
        grade = som_do_fio(
            unidades=som.get("unidades"),
            grupos=som.get("grupos", ()),
            linhas=som.get("linhas"),
            colunas=som.get("colunas"),
            epocas=som.get("epocas"),
            erro_de_quantizacao=som.get("erro_de_quantizacao"),
        )
        recorte = recorte_do_mapa_auto_organizavel(
            grade=grade,
            n_features_usadas=som.get("n_features_usadas", 0),
            n_features_totais=som.get("n_features_totais"),
            matriz=som.get("matriz", ""),
            amostragem=som.get("amostragem", ""),
            semente=som.get("semente"),
        )
        return figura_do_mapa_auto_organizavel(
            grade=grade, recorte=recorte, estilos=self.tabela_dos_grupos.estilos()
        ), ()

    def _classificacao_de(self, raiz: Any) -> tuple[Any, Any]:
        """A carga e o recorte de uma raiz de classificação, num leitor só para os dois.

        Os dois classificadores devolvem a mesma estrutura, e duas cópias do leitor divergiriam no
        dia em que uma ganhasse uma chave nova. O construtor não é partilhado, e a guarda de nome
        dele recusa a carga trocada.

        As chaves são lidas uma a uma, e não por `classificacao_do_fio(**raiz)`, para o teste de
        contrato as conferir aqui contra o esquema. O local chama-se `classificador` porque o teste
        classifica pelo nome: ele casa com `ResumoDoClassificador`, o esquema que as duas raízes
        partilham. A acurácia, o `p` e a semente derivada vêm do servidor, que correu a validação
        cruzada.
        """
        classificador = raiz if isinstance(raiz, dict) else {}
        classificacao = classificacao_do_fio(
            nome=classificador.get("nome"),
            grupos=classificador.get("grupos", ()),
            contagens=classificador.get("contagens", ()),
            recall=classificador.get("recall"),
            balanceada=classificador.get("balanced_accuracy"),
            simples=classificador.get("accuracy"),
            macro_f1=classificador.get("macro_f1"),
            p=classificador.get("permutation_pvalue"),
            acaso=classificador.get("chance_level"),
            cv_folds=classificador.get("cv_folds"),
            n_permutacoes=classificador.get("n_permutations"),
            semente=classificador.get("semente"),
        )
        recorte = recorte_do_classificador(
            classificacao=classificacao,
            n_features_usadas=classificador.get("n_features_usadas", 0),
            n_features_totais=classificador.get("n_features_totais"),
            matriz=classificador.get("matriz", ""),
            amostragem=classificador.get("amostragem", ""),
        )
        return classificacao, recorte

    def _floresta_das_amostras(self) -> tuple[Figure, tuple[str, ...]]:
        """O recall por classe da floresta aleatória, com o `p` de permutação.

        O segundo elemento é `()`: esta figura tem um segmento por classe, e não um ponto por
        feature, então identifica pelo rótulo da série.
        """
        classificacao, recorte = self._classificacao_de(self._random_forest)
        figura = figura_do_random_forest(
            classificacao=classificacao, recorte=recorte, estilos=self.tabela_dos_grupos.estilos()
        )
        return figura, ()

    def _svm_das_amostras(self) -> tuple[Figure, tuple[str, ...]]:
        """O recall por classe da máquina de vetores de suporte.

        A guarda de nome do construtor torna visível a troca de raiz: sem ela, um
        `self._random_forest` aqui daria uma figura bem formada com o título errado e os números de
        outro modelo.
        """
        classificacao, recorte = self._classificacao_de(self._svm)
        figura = figura_do_svm(
            classificacao=classificacao, recorte=recorte, estilos=self.tabela_dos_grupos.estilos()
        )
        return figura, ()

    def _rede_das_features(self) -> tuple[Figure, tuple[str, ...]]:
        """A rede de correlação parcial esparsa desenviesada.

        O segundo elemento não é `()`: a figura põe um ponto por feature, na ordem de
        `rede.features`, e o construtor marca a dispersão com `GID_DAS_FEATURES`, que
        `PainelDeFigura.desenha` usa para dizer qual feature o clique atingiu. Sem o vetor, não
        haveria como identificar um vértice, porque a figura não escreve rótulo de feature.

        As chaves são lidas uma a uma, para o teste de contrato as conferir aqui. O `n_arestas` é o
        do servidor, contado por caminho próprio no núcleo, e o construtor o compara com a lista.

        A matriz vai `NAO_DECLARADA` porque `ResumoDaRede` é a única raiz desta aba sem o campo
        `matriz`, e ler uma chave fora do esquema reprovaria no teste de contrato. Escrever 'a
        matriz corrigida' seria inventar procedência; o conserto é acrescentar o campo ao esquema.
        """
        rede = self._rede if isinstance(self._rede, dict) else {}
        carga = rede_do_fio(
            features=rede.get("features", ()),
            arestas=rede.get("arestas"),
            n_arestas=rede.get("n_arestas"),
            n_pares_testados=rede.get("n_pares_testados"),
            alfa=rede.get("alfa"),
            penalidade=rede.get("penalidade"),
            n_amostras_usadas=rede.get("n_amostras_usadas"),
            metodo=rede.get("metodo"),
        )
        recorte = recorte_da_rede(
            rede=carga,
            n_features_totais=rede.get("n_features_totais"),
            matriz=NAO_DECLARADA,
            amostragem=rede.get("amostragem", ""),
        )
        figura = figura_da_rede_de_correlacao_parcial(rede=carga, recorte=recorte)
        return figura, carga.features

    def _mapa_de_calor(self) -> tuple[Figure, tuple[str, ...]]:
        """O mapa de calor. Recusa por matriz vazia enquanto não houver rota."""
        agrupamento = self._agrupamento if isinstance(self._agrupamento, dict) else {}
        mapa = mapa_do_fio(
            valores=agrupamento.get("valores"),
            amostras=agrupamento.get("amostras"),
            features=agrupamento.get("features"),
            grupos=agrupamento.get("grupos", ()),
        )
        recorte = recorte_do_mapa_de_calor(
            mapa=mapa,
            metrica=agrupamento.get("metrica", ""),
            ligacao_metodo=agrupamento.get("ligacao_metodo", ""),
            n_features_totais=agrupamento.get("n_features_totais"),
            matriz=agrupamento.get("matriz", ""),
            amostragem=agrupamento.get("amostragem", ""),
        )
        return figura_do_mapa_de_calor(
            mapa=mapa, recorte=recorte, estilos=self.tabela_dos_grupos.estilos()
        ), ()

    # ------------------------------------------------------------------
    # As quatro de PCA
    # ------------------------------------------------------------------
    #
    # As quatro leem a mesma raiz e nenhuma recalcula estatística: o SVD correu uma vez, no
    # servidor, dentro de `POST /matrizes/diagnosticar`, e dele saem também
    # `pc1_injection_correlation` e a trajetória do QC.
    #
    # As quatro devolvem `()` no vetor de nomes. `desenha` resolve `self._nomeados` por
    # `get_gid() == GID_DAS_FEATURES` e `_nome_do_ponto` compara a identidade do artista, então um
    # artista não marcado cai no rótulo dele ('origem de PC2') e nunca num nome de feature.
    #
    # Preencher o vetor nas cargas exigiria alinhá-lo à população desenhada: a nuvem desenha
    # `cargas.x[desenhaveis]`, então o vetor certo seria o dos nomes das desenháveis,
    # `tuple(n for n, d in zip(cargas.features, cargas.desenhaveis) if d)`, e nunca
    # `pca["features"]`, que é a população inteira e devolveria o nome do vizinho. Isso muda
    # comportamento e fica para decisão. No `scree` (um ponto por componente) e no biplot (duas
    # populações no mesmo eixo), o `()` é a resposta certa, e `figura_do_biplot` imprime o número
    # da seta.

    def _resumo_da_pca(self) -> dict[str, Any]:
        """A PCA guardada, ou um dicionário vazio, que cai na recusa por população vazia."""
        return self._pca if isinstance(self._pca, dict) else {}

    def _escores_da_pca(self, *, com_grupos: bool) -> Any:
        """Os scores e o tipo de cada amostra, do fio. A fronteira valida os três tipos.

        `escores_da_pca` recusa um `sampleType` que não seja `qc`, `sample` ou `validate`, em vez
        de o pôr numa série por omissão; é por isso que a leitura das chaves vive aqui e não em
        `plots/`.

        Com `com_grupos`, lê também o grupo de cada amostra de estudo, para quem pinta por grupo
        (os scores). A fronteira o confere contra o tipo: um QC com grupo é recusa, e um nome em
        branco dá a figura pintada por tipo, com o título a dizer por quê; a chave ausente chega
        como `None` e dá a mesma figura. O biplot passa `False`: ele pinta por tipo, e um grupo
        recusado o tiraria da tela por causa de um vetor que ele não usa. O argumento não tem
        padrão, para cada montador decidir.
        """
        pca = self._resumo_da_pca()
        return escores_da_pca(
            pc1=pca.get("scores_pc1"),
            pc2=pca.get("scores_pc2"),
            tipos=pca.get("tipos_de_amostra"),
            grupos=pca.get("grupos_por_amostra") if com_grupos else None,
        )

    def _cargas_do_fio(self) -> Any:
        """As cargas por feature, do fio, alinhadas aos nomes das features retidas pelo passo 3."""
        pca = self._resumo_da_pca()
        return cargas_da_pca(
            pc1=pca.get("loadings_pc1"),
            pc2=pca.get("loadings_pc2"),
            features=pca.get("features"),
        )

    def _scree(self) -> tuple[Figure, tuple[str, ...]]:
        """A variância explicada por componente. Um ponto por componente, então `()`."""
        pca = self._resumo_da_pca()
        figura = figura_do_scree(
            variancia_explicada=vetor_do_fio(pca.get("variancia_explicada")),
            recorte=recorte_do_scree(
                n_features_usadas=pca.get("n_features_usadas"),
                n_features_totais=pca.get("n_features_totais"),
            ),
        )
        return figura, ()

    def _scores_da_pca(self) -> tuple[Figure, tuple[str, ...]]:
        """PC1 contra PC2, com `qc`, `sample` e `validate` em séries distintas.

        A região sai de `regiao_dos_scores`, e não de uma seleção feita aqui, porque a população é
        o que ela afirma. Com o grupo de cada amostra no fio, a nuvem é pintada por grupo, com uma
        região de predição por grupo recortada ao quadro dos pontos e a cor e o símbolo da tabela
        dos grupos; o QC continua série própria, por cima. Sem a chave, a figura pinta por tipo.
        """
        pca = self._resumo_da_pca()
        escores = self._escores_da_pca(com_grupos=True)
        regiao = regiao_dos_scores(escores)
        figura = figura_dos_scores(
            escores=escores,
            variancia_explicada=vetor_do_fio(pca.get("variancia_explicada")),
            regiao=regiao,
            recorte=recorte_dos_scores(
                n_features_usadas=pca.get("n_features_usadas"),
                n_features_totais=pca.get("n_features_totais"),
                regiao=regiao,
                # A PERMANOVA sai da mesma raiz que os scores, e por isso do mesmo SVD: corre em
                # `/matrizes/diagnosticar` sobre `StructuralPCA.scores`. Um `p` vindo de
                # `/execucoes` ficaria sob outra chave, e 'diagnosticar A, executar, diagnosticar
                # B' poria o número de A no rodapé da nuvem de B. O dicionário atravessa sem ser
                # indexado aqui; quem o valida é `plots.pca.permanova_do_fio`, e ler as chaves
                # aqui abriria uma segunda base de leitura a classificar em `RAIZES_ANINHADAS`.
                permanova=permanova_do_fio(pca.get("permanova")),
            ),
            estilos=self.tabela_dos_grupos.estilos(),
        )
        return figura, ()

    def _cargas_da_pca(self) -> tuple[Figure, tuple[str, ...]]:
        """Um ponto por feature retida: a carga em PC1 contra a carga em PC2."""
        pca = self._resumo_da_pca()
        cargas = self._cargas_do_fio()
        figura = figura_das_cargas(
            cargas=cargas,
            variancia_explicada=vetor_do_fio(pca.get("variancia_explicada")),
            recorte=recorte_das_cargas(
                cargas=cargas, n_features_totais=pca.get("n_features_totais")
            ),
        )
        return figura, ()

    def _biplot(self) -> tuple[Figure, tuple[str, ...]]:
        """As amostras e as features de maior carga, no mesmo par de eixos."""
        pca = self._resumo_da_pca()
        escores = self._escores_da_pca(com_grupos=False)
        cargas = self._cargas_do_fio()
        figura = figura_do_biplot(
            escores=escores,
            cargas=cargas,
            variancia_explicada=vetor_do_fio(pca.get("variancia_explicada")),
            n_setas=SETAS_DO_BIPLOT,
            recorte=recorte_do_biplot(
                escores=escores,
                cargas=cargas,
                n_setas=SETAS_DO_BIPLOT,
                n_features_totais=pca.get("n_features_totais"),
            ),
        )
        return figura, ()

    # ------------------------------------------------------------------
    # As três de PLS-DA, sempre com a validação
    # ------------------------------------------------------------------
    #
    # A `Validacao` é lida uma vez e passada às três, porque o `q2` e o `p` de permutação que as
    # figuras imprimem têm de ser os mesmos. `escores_do_plsda`, `cargas_do_plsda` e `vip_do_plsda`
    # exigem uma `Validacao` sem valor padrão, então nenhum caminho deste painel constrói um score
    # de PLS-DA sem o par ao lado; o Q² viaja no título da figura de dispersão.
    #
    # A variável chama-se `vetores_do_plsda`, e não `plsda`, porque o nome é contrato: em
    # `qualidade.py`, `self.plsda` é o dicionário de rótulos da validação, e o teste de AST
    # identifica o dono pelo nome. `RAIZES_ANINHADAS` declara a base como `vetores_do_plsda`.

    def _resumo_do_plsda(self) -> dict[str, Any]:
        """Os vetores guardados, ou um dicionário vazio, que cai na recusa por população vazia."""
        return self._plsda if isinstance(self._plsda, dict) else {}

    def _validacao(self) -> Any:
        """O par obrigatório do PLS-DA (`q2` e `p` de permutação), com o regime em que foi medido.

        As leituras ficam aqui, e não em `plots/`, para o teste de contrato as conferir contra o
        esquema; um `permutation_pvalor` mal escrito em `plots/` devolveria `nan` para sempre, com a
        figura a imprimir 'não declarado' e nada a levantar.
        """
        vetores_do_plsda = self._resumo_do_plsda()
        return validacao_do_fio(
            q2=vetores_do_plsda.get("q2"),
            permutation_pvalue=vetores_do_plsda.get("permutation_pvalue"),
            r2y=vetores_do_plsda.get("r2y"),
            n_permutacoes=vetores_do_plsda.get("n_permutations"),
            cv_folds=vetores_do_plsda.get("cv_folds"),
            n_componentes=vetores_do_plsda.get("n_components"),
        )

    def _scores_do_plsda(self) -> tuple[Figure, tuple[str, ...]]:
        """Componente 1 contra componente 2, uma série e uma região por grupo, e o Q² no título.

        O grupo positivo vem do fio e não é adivinhado. `statistics/plsda.py` monta
        `y = (rótulos == segundo)`; trocar a ordem não muda número nenhum, porque a orientação do
        sinal é canônica, e decide só qual série a legenda chama `y = 1`.

        Os escores lidos são os de validação cruzada (`scores_cv_componente_*`). Os de calibração
        tendem a separar os grupos, mesmo sobre rótulos sem informação quando há muito mais features
        que amostras, e uma região por grupo por cima deles reforçaria uma separação que o Q²
        desmente. Sem os de validação cruzada a figura é recusada, e a recusa distingue o PLS-DA que
        não correu (sem score nenhum) do corpo que traz só os de calibração, porque os consertos são
        opostos.
        """
        vetores_do_plsda = self._resumo_do_plsda()
        if (
            vetor_do_fio(vetores_do_plsda.get("scores_componente_1")).size
            and not vetor_do_fio(vetores_do_plsda.get("scores_cv_componente_1")).size
        ):
            raise MatrizVaziaError(
                "o corpo do PLS-DA traz os escores de calibração e não os de validação cruzada "
                "(`scores_cv_componente_*`), que são os que esta figura desenha. O PLS-DA correu; "
                "os de calibração não entram no lugar: com muito mais features que amostras, eles "
                "separam grupos mesmo sem informação"
            )
        escores = escores_do_plsda(
            comp1=vetores_do_plsda.get("scores_cv_componente_1"),
            comp2=vetores_do_plsda.get("scores_cv_componente_2"),
            grupos=vetores_do_plsda.get("grupos_por_amostra"),
            grupo_positivo=vetores_do_plsda.get("grupo_positivo"),
            grupo_negativo=vetores_do_plsda.get("grupo_negativo"),
            validacao=self._validacao(),
        )
        figura = figura_dos_scores_do_plsda(
            escores=escores,
            recorte=recorte_dos_scores_do_plsda(
                escores=escores, n_features_totais=vetores_do_plsda.get("n_features")
            ),
            # O estilo da coorte, e não dos dois grupos do contraste: dá ao mesmo grupo a mesma
            # cor em qualquer par que a pessoa declare.
            estilos=self.tabela_dos_grupos.estilos(),
        )
        return figura, ()

    def _cargas_do_plsda(self) -> tuple[Figure, tuple[str, ...]]:
        """Um ponto por feature: a carga nos dois componentes.

        `()` no vetor de nomes: são três artistas, e `_identifica` indexa por posição sem saber de
        qual deles veio o evento.
        """
        vetores_do_plsda = self._resumo_do_plsda()
        cargas = cargas_do_plsda(
            comp1=vetores_do_plsda.get("loadings_componente_1"),
            comp2=vetores_do_plsda.get("loadings_componente_2"),
            features=vetores_do_plsda.get("features"),
            validacao=self._validacao(),
        )
        figura = figura_das_cargas_do_plsda(
            cargas=cargas,
            recorte=recorte_das_cargas_do_plsda(
                cargas=cargas, n_features_totais=vetores_do_plsda.get("n_features")
            ),
        )
        return figura, ()

    def _vip(self) -> tuple[Figure, tuple[str, ...]]:
        """As features de maior VIP. O rótulo de cada marca do eixo já diz qual feature é.

        `()` no vetor de nomes: a figura escreve o nome de cada feature desenhada no eixo y, o que
        responde melhor que uma anotação, e o vetor não pode ser passado porque a linha de
        referência também é clicável. A seleção é recalculada pelo recorte, o que é seguro porque
        `selecao_do_vip` é pura e determinística, como em `recorte_do_biplot`.
        """
        vetores_do_plsda = self._resumo_do_plsda()
        importancia = vip_do_plsda(
            vip=vetores_do_plsda.get("vip"),
            features=vetores_do_plsda.get("features"),
            validacao=self._validacao(),
        )
        figura = figura_do_vip(
            importancia=importancia,
            n_features=FEATURES_DO_VIP,
            recorte=recorte_do_vip(
                importancia=importancia,
                n_desenhadas=len(selecao_do_vip(importancia, FEATURES_DO_VIP)),
            ),
        )
        return figura, ()

    # ------------------------------------------------------------------
    # O painel da correção, a única figura que lê duas cargas, de duas rotas
    # ------------------------------------------------------------------
    #
    # As leituras ficam aqui, e não em `plots/`, para o teste de contrato as conferir contra o
    # esquema; um `scores_pc1_antes` mal escrito em `plots/` devolveria vetor vazio para sempre, e o
    # painel diria 'população vazia' sobre um corpo completo. As bases chamam-se `par` e `medicao`
    # porque `RAIZES_ANINHADAS` as declara contra `ParDePcas` e `MedicaoDoCorretorResposta`.

    def _resumo_do_par(self) -> dict[str, Any]:
        """O par guardado, ou um dicionário vazio, que cai na recusa por população vazia."""
        return self._pcas_da_correcao if isinstance(self._pcas_da_correcao, dict) else {}

    def _painel_da_correcao(self) -> tuple[Figure, tuple[str, ...]]:
        """As barras de RSD de QC e as duas PCAs, numa figura: o painel que o laboratório lê.

        A execução é a condição de existir figura; a medição decide só se há fila de barras. Sem
        execução não há par de PCAs, e o construtor recusa por população vazia. Sem medição, a
        figura desenha as duas PCAs e põe o texto declarado no lugar das barras.

        `fila_do_fio` recebe o corpo como chegou: quem decidiu se ele é fila foi o servidor, por
        `avaliavel`, a mesma pergunta de `interface/fluxo.prosa_da_medicao`. O segundo elemento é
        `()`:
        esta figura põe um ponto por injeção e uma barra por corretor, e nenhum é uma feature; a
        identificação cai no rótulo da série.
        """
        par = self._resumo_do_par()
        medicao = self._medicao if isinstance(self._medicao, dict) else {}
        pareado = par_da_correcao(
            antes=lado_da_correcao(
                rotulo=ROTULO_ANTES,
                matriz=par.get("matriz_antes"),
                pc1=par.get("scores_pc1_antes"),
                pc2=par.get("scores_pc2_antes"),
                tipos=par.get("tipos_de_amostra_antes"),
                variancia_explicada=par.get("variancia_explicada_antes"),
                qc_dispersion_ratio=par.get("qc_dispersion_ratio_antes"),
                n_features_usadas=par.get("n_features_usadas_antes"),
                n_features_totais=par.get("n_features_totais_antes"),
                digest_das_colunas_retidas=par.get("digest_das_colunas_retidas_antes"),
            ),
            depois=lado_da_correcao(
                rotulo=ROTULO_DEPOIS,
                matriz=par.get("matriz_depois"),
                pc1=par.get("scores_pc1_depois"),
                pc2=par.get("scores_pc2_depois"),
                tipos=par.get("tipos_de_amostra_depois"),
                variancia_explicada=par.get("variancia_explicada_depois"),
                qc_dispersion_ratio=par.get("qc_dispersion_ratio_depois"),
                n_features_usadas=par.get("n_features_usadas_depois"),
                n_features_totais=par.get("n_features_totais_depois"),
                digest_das_colunas_retidas=par.get("digest_das_colunas_retidas_depois"),
            ),
            pretransformacao=par.get("pretransformacao"),
            n_componentes=par.get("n_componentes"),
            # O nome do corretor, que é o `xlab` do painel direito (`tools.R:1597`). Vem da raiz, e
            # não de `self._medicao`, cujo `vencedor` é outra quantidade: o que venceu a
            # comparação, e não o que produziu esta matriz.
            corretor=par.get("corretor"),
        )
        fila = fila_do_fio(
            scores=medicao.get("scores"),
            vencedor=medicao.get("vencedor"),
            abstem=medicao.get("abstem"),
            estimador=medicao.get("estimador"),
            variante=medicao.get("variante"),
            n_features_medidas=medicao.get("n_features_medidas"),
            # O intervalo por barra, quando a medição o trouxe. `None` quando a caixa não foi
            # marcada, e então a figura não muda.
            intervalos=_intervalos_da_medicao(medicao.get("intervalos")),
            margem=medicao.get("margem"),
            # O corretor cujo lote menor não lhe deixa QCs de treino: sem barra, com o motivo no
            # título e no rodapé, em vez de cair entre os de RSD indefinido.
            recusas=medicao.get("recusas"),
        )
        figura = figura_do_painel_da_correcao(
            par=pareado,
            fila=fila,
            recorte=recorte_do_painel_da_correcao(par=pareado, fila=fila),
        )
        return figura, ()

    # ------------------------------------------------------------------
    # O violino de RSD por feature
    # ------------------------------------------------------------------
    #
    # As leituras ficam aqui, e não em `plots/`, para o teste de contrato as conferir contra o
    # esquema; um `n_features_finitas_dos_qcs` mal escrito em `plots/` devolveria `0` para sempre,
    # e a figura declararia zero features definidas com a imagem bem formada. A base chama-se
    # `rsd_por_feature` porque `RAIZES_ANINHADAS` a declara contra `RsdPorFeature`.

    def _resumo_do_rsd(self) -> dict[str, Any]:
        """Os dois vetores guardados, ou um dicionário vazio, que cai na recusa por população vazia.

        São os da matriz ingerida, do diagnóstico, até chegar o primeiro resultado de corretor. Quem
        decide é `_violino_corrigido`, a mesma bandeira que escolhe a regra de validade em
        `_validade_propria`; duas perguntas separadas desenhariam um conjunto de dados sob a regra
        do outro."""
        guardado = self._rsd_por_feature if self._violino_corrigido else self._rsd_da_ingerida
        return guardado if isinstance(guardado, dict) else {}

    def _violino_do_rsd(self) -> tuple[Figure, tuple[str, ...]]:
        """As duas distribuições de RSD por feature: os QCs contra as amostras.

        O `n` escrito na imagem é o de features (`n_features_finitas_*`), e não `n_injecoes_*`. A
        contagem de injeções viaja no mesmo objeto para separar 'não há QC nenhum' de 'há QCs e
        nenhuma feature com RSD definido', e é o número errado ao alcance da mão; os dois vão ao
        construtor com o nome de cada um, o que impede a troca de passar em silêncio.

        O segundo elemento é `()`: a figura desenha um contorno por lado, e não um ponto por
        feature; a identificação cai no rótulo da série, que aqui é o lado e o papel do artista.
        """
        rsd_por_feature = self._resumo_do_rsd()
        par = par_do_rsd(
            qcs=lado_do_rsd(
                rotulo=ROTULO_DOS_QCS,
                valores=rsd_por_feature.get("rsd_dos_qcs"),
                n_features_finitas=rsd_por_feature.get("n_features_finitas_dos_qcs"),
                n_injecoes=rsd_por_feature.get("n_injecoes_dos_qcs"),
            ),
            amostras=lado_do_rsd(
                rotulo=ROTULO_DAS_AMOSTRAS,
                valores=rsd_por_feature.get("rsd_das_amostras"),
                n_features_finitas=rsd_por_feature.get("n_features_finitas_das_amostras"),
                n_injecoes=rsd_por_feature.get("n_injecoes_das_amostras"),
            ),
            matriz=rsd_por_feature.get("matriz"),
            corretor=rsd_por_feature.get("corretor"),
            variante=rsd_por_feature.get("rsd_variant"),
            estimador=rsd_por_feature.get("rsd_estimator"),
            n_features=rsd_por_feature.get("n_features"),
        )
        figura = figura_do_violino_do_rsd_por_feature(
            par=par, recorte=recorte_do_violino_do_rsd_por_feature(par=par)
        )
        return figura, ()

    # ------------------------------------------------------------------
    # A composição por classe
    # ------------------------------------------------------------------
    #
    # As leituras da raiz ficam aqui, para o teste de contrato as conferir. As leituras por classe
    # (`grupo`, `n_zeros`, `n_detectadas`) ficam em `classes_do_fio`, em `plots/`, como as das
    # caixas: o teste parte da base, e o elemento de uma lista não é base. A base chama-se
    # `composicao` porque `RAIZES_ANINHADAS` a declara contra `ComposicaoPorClasse`.

    def _resumo_da_composicao(self) -> dict[str, Any]:
        """A composição guardada, ou um dicionário vazio, que cai no texto declarado."""
        return self._composicao if isinstance(self._composicao, dict) else {}

    def _composicao_por_classe(self) -> tuple[Figure, tuple[str, ...]]:
        """Os dois painéis da composição por classe: o que falta, e em que nível de sinal.

        O denominador da percentagem é `n_celulas`, o número de células da classe, e não
        `n_injecoes`, que viaja no mesmo objeto porque vai na marca do eixo; os dois vão ao
        construtor com o nome de cada um, o que impede a troca de passar em silêncio.

        O segundo elemento é `()`: a figura desenha uma barra por grupo clínico, e não um ponto por
        feature; a identificação cai no rótulo da série, que aqui é o grupo e o papel do artista.
        """
        composicao = self._resumo_da_composicao()
        pronta = composicao_do_fio(
            classes=composicao.get("classes"),
            matriz=composicao.get("matriz"),
            populacao=composicao.get("populacao"),
            n_features=composicao.get("n_features"),
            n_injecoes_fora=composicao.get("n_injecoes_fora_da_populacao"),
            definicao_de_faltante=composicao.get("definicao_de_faltante"),
            definicao_de_abundancia=composicao.get("definicao_de_abundancia"),
        )
        figura = figura_da_composicao_por_classe(
            composicao=pronta,
            recorte=recorte_da_composicao_por_classe(composicao=pronta),
            estilos=self.tabela_dos_grupos.estilos(),
        )
        return figura, ()

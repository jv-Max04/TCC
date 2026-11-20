"""O painel de qualidade: as métricas, o PLS-DA validado, o antes-e-depois do RSD dos QCs, a
normalidade por grupo e o resumo metodológico da Análise.

Nenhuma métrica única: o conjunto exibido é `fluxo.METRICAS_DA_TELA_DE_QUALIDADE`, derivado de
`QualityReport` e não redigitado aqui, então métrica nova no relatório entra sozinha. RSD e
agrupamento de QC no PCA discordam e elegem métodos diferentes, e eleger um valor aqui reduziria
qualidade a um número na camada em que o usuário olha. As métricas aparecem por tema, na partição
de `fluxo.agrupa_metricas_da_qualidade`, cada uma num cartão com o nome em português, a linha do
valor e a explicação em 'Como ler'; o nome técnico fecha a explicação, para quem o procura num
CSV.

PLS-DA nunca sem validação: o score e o par de validação são escritos pelo mesmo laço, sobre
`nomes_do_plsda()`, e não há caminho de código que produza um sem o outro. Um portão por AST
confere isso, porque um teste de comportamento só cobre os caminhos que alguém pensou em
exercitar. As frases, as traduções e as contas vivem em `interface/fluxo.py`; este módulo só lê o
relatório e desenha, sem aritmética nenhuma, o que outro portão por AST confere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from refino.api.esquemas import RESSALVA_DO_ESTIMADOR
from refino.desktop.janela.arquivos import Salvador, salvador_por_dialogo
from refino.desktop.janela.componentes import GradeResponsiva, desliga_com_motivo, religa
from refino.desktop.janela.execucao import PainelDoResumo
from refino.desktop.janela.expansao import ExpansaoSuave
from refino.interface.fluxo import (
    LIMITE_DA_NORMALIDADE,
    NOME_DA_TABELA_DA_NORMALIDADE,
    NORMALIDADE_EM_REPOUSO,
    NORMALIDADE_NAO_PEDIDA,
    PARES_DE_VALIDACAO_OBRIGATORIA,
    ROTULO_DA_TABELA_DA_NORMALIDADE,
    SOB_PAR_DE_GRUPOS,
    SOB_PAR_DE_GRUPOS_NA_JANELA,
    TITULO_DA_NORMALIDADE,
    agrupa_metricas_da_qualidade,
    aviso_do_plsda,
    csv_da_normalidade,
    estado_de_ausencia,
    legenda_da_metrica,
    prosa_do_estimador_de_rsd,
    prosa_do_numero,
    prosa_dos_grupos_da_normalidade,
    prosa_dos_rotulos_do_rsd,
    rotulo_da_metrica,
    valor_da_metrica,
)

__all__ = [
    "ESTILO_DA_AUSENCIA",
    "ESTILO_DA_LEGENDA",
    "ESTILO_DO_VALOR",
    "MOTIVO_SEM_TABELA_DA_NORMALIDADE",
    "NOTA_DA_REPETICAO_DO_PLSDA",
    "PREFIXO_DO_VALOR",
    "SEM_COMPARACAO",
    "TITULO_DA_SECAO_DO_PLSDA",
    "TITULO_DO_RESUMO_DA_ANALISE",
    "PainelDeQualidade",
    "aviso_do_plsda",
    "estilo_do_valor",
    "legenda_da_metrica",
    "nomes_do_plsda",
    "rotulo_da_metrica",
    "texto_da_comparacao",
    "texto_da_metrica",
    "valor_da_metrica",
]

SEM_COMPARACAO: Final = "Nenhuma comparação disponível"

# O título da moldura do resumo metodológico da Análise, que fica nesta aba.
TITULO_DO_RESUMO_DA_ANALISE: Final = "Resumo metodológico da análise"

# A frase ao lado de 'Salvar a tabela por variável (CSV)…' desligado. Uma frase para os dois
# estados sem tabela (nada correu, ou correu sem a família), porque a ação é a mesma nos dois;
# o que os separa é o resumo logo acima, que já diz qual é.
MOTIVO_SEM_TABELA_DA_NORMALIDADE: Final = (
    "A tabela sai da «Análise estatística» com «Normalidade por grupo» marcada em «Análises a "
    "correr»: rode-a assim para salvá-la."
)

# O prefixo da linha do valor. O nome técnico fica no fim da explicação, e o prefixo mantém a
# forma `algo: valor`, que os testes leem com `texto.split(": ", 1)[1]`: sem ele, esse split
# devolveria '' e acusaria um painel em branco que não existe. A forma funciona também quando
# a frase de ausência tem `:` várias vezes, porque o split é na primeira ocorrência.
PREFIXO_DO_VALOR: Final = "Valor: "

# A hierarquia é do widget, e não do texto: o valor tem corpo maior e peso, e a explicação é
# menor, para o número e o parágrafo não pesarem igual. São literais, sem conta nem formato,
# e por isso ficam fora do portão de aritmética. 13 pt, e não maior: o valor de uma métrica
# calculada é o número e a população dele numa linha só (`fluxo.valor_da_metrica`), e num
# corpo maior a frase da população tomaria o cartão. O número continua o maior texto do
# cartão, acima dos 9,5 pt da ausência.
ESTILO_DO_VALOR: Final = "font-family: 'IBM Plex Mono'; font-size: 13pt; font-weight: 500;"
ESTILO_DA_LEGENDA: Final = "font-size: 9pt;"
# A ausência não tem o corpo do número: sem o par de grupos, vários rótulos de valor repetem a
# frase de ausência inteira, e em 13 pt com peso ela tomaria o lugar do número. A frase continua
# inteira e no mesmo rótulo (é o que `texto_visivel` lê); ela só pesa como a nota que é.
ESTILO_DA_AUSENCIA: Final = "font-size: 9.5pt; font-weight: 400;"

# Até duas métricas por linha dentro de um tema. O título do cartão é o da moldura, que não
# quebra linha, e o mais longo mede uns 485 px em negrito a 10 pt: três colunas punham rolagem
# horizontal a 1440 px, e duas cabem a 1280 px.
COLUNAS_DOS_CARTOES: Final = 2

# O botão que abre a explicação de uma métrica. A explicação é a de sempre (`legenda_da_metrica`).
ROTULO_DO_COMO_LER: Final = "Como ler"

# O título da seção dedicada do PLS-DA, com a regra que ela mostra: PLS-DA nunca sem validação.
TITULO_DA_SECAO_DO_PLSDA: Final = "PLS-DA — nunca sem validação"
NOTA_DA_REPETICAO_DO_PLSDA: Final = (
    "O Q² e o p de permutação repetem os da lista acima: aqui ficam ao lado do score do PLS-DA, "
    "que nunca aparece sem eles."
)


def nomes_do_plsda() -> tuple[str, ...]:
    """O score e os exigidos de cada par, sem repetição e em ordem estável.

    O score entra na lista, e não só os exigidos: o par declarado põe o score dentro dos
    exigidos, mas um par com score de nome próprio faria o painel deixar de mostrar o score sem
    nada acusar.

    Uma função e não uma constante de módulo, porque ela é o referente do portão de AST: o
    portão afirma que toda escrita nos rótulos do PLS-DA é dirigida por esta lista.
    """
    vistos: list[str] = []
    for score, exigidos in PARES_DE_VALIDACAO_OBRIGATORIA:
        for nome in (score, *exigidos):
            if nome not in vistos:
                vistos.append(nome)
    return tuple(vistos)


def texto_da_metrica(
    nome: str, relatorio: dict[str, Any], caminhos: dict[str, str] | None = None
) -> str:
    """O valor de uma métrica, ou qual estado de ausência ela tem.

    A ordem das perguntas é o conteúdo: `computed` primeiro, e só depois o caminho da métrica
    (`interface/fluxo.py::estado_de_ausencia`). Invertê-la faria uma métrica calculada sem caminho
    declarado sair como 'sem caminho'. Quem formata o valor calculado é
    `fluxo.valor_da_metrica`, com a unidade e a população; o portão de AST deste módulo proíbe a
    conta aqui.

    `caminhos` entra por injeção com o padrão certo, para um teste ver os desfechos sem depender
    do valor que `CAMINHO_DA_METRICA` tem hoje; nenhum chamador de produção o passa. Nome
    desconhecido cai em 'não calculada', o mais conservador, em vez de levantar `KeyError`
    dentro de um slot de Qt.
    """
    calculadas = set(relatorio.get("computed") or ())
    if nome in calculadas:
        return valor_da_metrica(nome, relatorio)
    # Para o par de grupos, a janela usa a frase própria dela, que diz onde o par se escolhe nela.
    ausencia = estado_de_ausencia(nome, caminhos)
    return SOB_PAR_DE_GRUPOS_NA_JANELA if ausencia == SOB_PAR_DE_GRUPOS else ausencia


def estilo_do_valor(nome: str, relatorio: dict[str, Any]) -> str:
    """O corpo do rótulo do valor: o do número se a métrica foi calculada, o da nota se não.

    A pergunta é a de `texto_da_metrica`, e na mesma forma: `computed` decide, e nada mais.
    Perguntar ao texto, se ele começa por 'não', mudaria o corpo de uma métrica calculada cujo
    valor fosse uma frase, sem ninguém decidir.
    """
    calculadas = set(relatorio.get("computed") or ())
    return ESTILO_DO_VALOR if nome in calculadas else ESTILO_DA_AUSENCIA


def texto_da_comparacao(comparacao: Any) -> str:
    """O par antes-e-depois do RSD dos QCs, com os seis campos do esquema.

    A ressalva do estimador viaja com o número: `rsd_estimator` in-sample favorece quem ajusta
    a curva nos mesmos QCs em que depois mede, e um par de medianas sem o nome do estimador
    convida à conclusão errada (o tamanho do favorecimento está no cabeçalho de
    `plots/correcao.py`). `RESSALVA_DO_ESTIMADOR` é a constante da API, e não uma frase escrita
    aqui.

    `rsd_variant` é obrigatório: RSD carrega dois rótulos, variante e estimador, ou não é
    comparável, e duas execuções do mesmo modo sob variantes diferentes dão medianas diferentes.
    Os rótulos saem em português por `prosa_dos_rotulos_do_rsd`, a mesma função das outras
    frases que os citam.

    `None` numa mediana passa por `prosa_do_numero` e não vira 'não há comparação': `nan` é
    alcançável quando nenhuma feature tem RSD definido nos dois lados, e colapsá-lo apagaria a
    diferença entre 'não pedi' e 'pedi e não deu'. `SEM_COMPARACAO` fica para o campo ausente
    ou que não é dicionário.

    A população é a mesma nos dois números, de propósito, e o texto o declara: corrigir muda
    quais features têm RSD definido, e um denominador por lado compararia conjuntos diferentes.

    O rótulo que mostra este texto é `MarkdownText`, porque `RESSALVA_DO_ESTIMADOR` traz negrito
    e crases; por isso a ressalva vem depois de uma linha em branco, já que em markdown uma
    quebra só colapsa num espaço. As interpolações não levam crases: aspas de código são
    vocabulário de quem programa, e quem lê esta tela opera o espectrômetro.
    """
    if not isinstance(comparacao, dict):
        return SEM_COMPARACAO
    if "mediana_antes" not in comparacao or "mediana_depois" not in comparacao:
        return SEM_COMPARACAO
    rotulos = prosa_dos_rotulos_do_rsd(
        comparacao.get("rsd_variant"), comparacao.get("rsd_estimator")
    )
    alternativa = prosa_do_estimador_de_rsd(comparacao.get("estimador_alternativo"))
    return (
        f"RSD mediana dos QCs: {prosa_do_numero(comparacao.get('mediana_antes'))} antes, "
        f"{prosa_do_numero(comparacao.get('mediana_depois'))} depois — sobre as "
        f"{comparacao.get('n_features_comparadas')} de {comparacao.get('n_features')} features "
        f"com RSD definido nos DOIS lados. Correção {comparacao.get('correcao')}, RSD "
        f"{rotulos} (alternativa: {alternativa}).\n\n"
        f"{RESSALVA_DO_ESTIMADOR}"
    )


class _MolduraDaQualidade(QGroupBox):
    """Preserva o título consultável e acompanha a altura do conteúdo na largura real.

    O título nativo do QGroupBox participa da largura mínima mesmo quando a folha o oculta.
    Aqui a medida vem do conteúdo real: um nome longo ocupa duas linhas em vez de alargar a aba.
    """

    def __init__(self, titulo: str, pai: QWidget) -> None:
        super().__init__(titulo, pai)
        # A folga fica no fim da página, sem esticar o cartão ao recolher sua explicação.
        politica = self.sizePolicy()
        politica.setVerticalPolicy(QSizePolicy.Policy.Maximum)
        self.setSizePolicy(politica)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - API do Qt
        disposicao = self.layout()
        return disposicao.minimumSize() if disposicao is not None else super().minimumSizeHint()

    def sizeHint(self) -> QSize:  # noqa: N802 - API do Qt
        disposicao = self.layout()
        if disposicao is None:
            return super().sizeHint()
        preferida = disposicao.sizeHint()
        # Maximum precisa da altura na largura real, incluindo a prosa de 'Como ler' aberta.
        altura = disposicao.totalHeightForWidth(self.width())
        return QSize(preferida.width(), altura) if altura >= 0 else preferida

    def resizeEvent(self, evento: QResizeEvent) -> None:  # noqa: N802 - API do Qt
        super().resizeEvent(evento)
        if evento.size().width() != evento.oldSize().width():
            self.updateGeometry()


class PainelDeQualidade(QWidget):
    """O conjunto completo das métricas, o PLS-DA validado, e o antes-e-depois.

    Um rótulo por métrica, com a chave da métrica, e a enumeração vem de
    `fluxo.agrupa_metricas_da_qualidade`, que reparte `METRICAS_DA_TELA_DE_QUALIDADE` por tema e
    garante que nenhuma se perde. Uma lista escrita à mão aqui omitiria a métrica nova em
    silêncio.

    A janela alimenta este painel: `JanelaPrincipal._recebe` roteia o corpo de `/execucoes` para
    `mostra_qualidade`, `mostra_plsda` e `mostra_comparacao`.
    """

    def __init__(self, parent: QWidget | None = None, *, salvador: Salvador | None = None) -> None:
        super().__init__(parent)
        # O diálogo de salvar da tabela de normalidade entra por injeção, pela razão de
        # `PainelDeArquivos`: um `QFileDialog` real é modal, e um arnês que não o remende pendura.
        # A janela passa o mesmo `arquivo` que dá à figura e ao exemplo dos fatores.
        self._salvador = salvador
        # A raiz `normalidade` da última execução, ou `None`. É dela que o CSV sai, no clique.
        self._normalidade: dict[str, Any] | None = None

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        self.rolagem = QScrollArea(self)
        self.rolagem.setWidgetResizable(True)
        self.rolagem.setFrameShape(QFrame.Shape.NoFrame)
        # O fator 1 declara a rolagem dona da folga desta raiz, onde ela é o único widget.
        raiz.addWidget(self.rolagem, 1)

        interno = QWidget(self.rolagem)
        interno.setProperty("folhaDeEtapa", True)
        interno.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        disposicao = QVBoxLayout(interno)
        disposicao.setContentsMargins(20, 20, 20, 20)
        disposicao.setSpacing(24)

        # Uma moldura por métrica, com o valor antes da explicação: quem abre a aba veio buscar o
        # número, e a explicação é o que se lê quando o número surpreende. A moldura separa as
        # métricas umas das outras, e o título dela é o nome em português (`rotulo_da_metrica`).
        # A linha do valor mantém a forma `algo: valor`, com `PREFIXO_DO_VALOR`, porque é ela que
        # `texto_visivel` e os testes leem.
        #
        # As métricas se leem por tema, numa moldura de tema por fora (`QGroupBox` dentro de
        # `QGroupBox`). A partição vem de `fluxo.agrupa_metricas_da_qualidade` e não é escrita
        # aqui, e nenhuma métrica é escondida, por construção: a função põe cada métrica de
        # `METRICAS_DA_TELA_DE_QUALIDADE` em exatamente um grupo, e a que não tiver tema sai num
        # grupo marcado.
        self.metricas: dict[str, QLabel] = {}
        self.legendas: dict[str, QLabel] = {}
        self.temas: dict[str, QGroupBox] = {}
        self.notas_dos_temas: dict[str, QLabel] = {}
        for grupo in agrupa_metricas_da_qualidade():
            moldura_do_tema = _MolduraDaQualidade(grupo.titulo, interno)
            moldura_do_tema.setObjectName("tema_da_qualidade")
            politica = moldura_do_tema.sizePolicy()
            politica.setVerticalPolicy(QSizePolicy.Policy.Maximum)
            moldura_do_tema.setSizePolicy(politica)
            dentro_do_tema = QVBoxLayout(moldura_do_tema)
            dentro_do_tema.setContentsMargins(0, 26, 0, 24)
            dentro_do_tema.setSpacing(16)
            # `MarkdownText`, porque a prosa de `interface/fluxo.py` vem em markdown e `AutoText`
            # não o vê.
            nota = QLabel(grupo.nota, moldura_do_tema)
            nota.setWordWrap(True)
            nota.setTextFormat(Qt.TextFormat.MarkdownText)
            nota.setObjectName("nota_do_campo")
            dentro_do_tema.addWidget(nota)
            self.temas[grupo.chave] = moldura_do_tema
            self.notas_dos_temas[grupo.chave] = nota
            # As métricas de um tema são cartões numa grade: o cartão traz o nome e o valor, e o
            # parágrafo abre em 'Como ler'. O rótulo da explicação continua no layout do cartão
            # (é o que `texto_visivel` confere), recolhido e não removido.
            cartoes: list[QWidget] = []
            for nome in grupo.metricas:
                moldura = _MolduraDaQualidade(rotulo_da_metrica(nome), moldura_do_tema)
                moldura.setObjectName("metrica_da_qualidade")
                moldura.setMinimumHeight(142)
                dentro = QVBoxLayout(moldura)
                dentro.setContentsMargins(16, 16, 16, 16)
                dentro.setSpacing(12)
                titulo = QLabel(rotulo_da_metrica(nome), moldura)
                titulo.setObjectName("titulo_da_metrica")
                titulo.setWordWrap(True)
                dentro.addWidget(titulo)
                # Semeado com o estado certo, e não com `NAO_CALCULADA` fixo: duas das onze só
                # podem estar em `SEM_CAMINHO`, e um rótulo que nasce dizendo 'não calculada' diria
                # que outra execução pode preenchê-la, e nenhuma pode.
                rotulo = QLabel(f"{PREFIXO_DO_VALOR}{texto_da_metrica(nome, {})}", moldura)
                rotulo.setObjectName("valor_da_metrica")
                rotulo.setWordWrap(True)
                # Nasce sem relatório, então nasce na ausência.
                rotulo.setStyleSheet(estilo_do_valor(nome, {}))
                self.metricas[nome] = rotulo
                dentro.addWidget(rotulo)
                self.legendas[nome] = self._legenda(nome, moldura, dentro)
                cartoes.append(moldura)
            # Até duas colunas, e uma quando não cabem: com duas fixas, o título mais longo, que
            # não quebra, punha rolagem horizontal na aba a 1280x760.
            grade = GradeResponsiva(
                cartoes,
                COLUNAS_DOS_CARTOES,
                moldura_do_tema,
                espaco=1,
                alinhar_ao_topo=True,
            )
            politica = grade.sizePolicy()
            politica.setVerticalPolicy(QSizePolicy.Policy.Maximum)
            grade.setSizePolicy(politica)
            dentro_do_tema.addWidget(grade)
            disposicao.addWidget(moldura_do_tema)

        # A seção do PLS-DA, com os widgets do par nascendo do mesmo laço que `mostra_plsda` usa.
        # O par já aparece na lista de cima, sob o tema da validação, e a seção o repete de
        # propósito: é aqui que o score e a validação ficam lado a lado. A seção tem moldura
        # própria para não se ler como dois quadros soltos depois dos temas. Por causa desta
        # repetição, 'toda métrica tem exatamente um tema' é falsa ao nível do widget; o oráculo
        # do agrupamento é sobre `TRADUCAO_DA_METRICA`, e não sobre molduras.
        self.secao_do_plsda = _MolduraDaQualidade(TITULO_DA_SECAO_DO_PLSDA, interno)
        self.secao_do_plsda.setObjectName("tema_da_qualidade")
        politica = self.secao_do_plsda.sizePolicy()
        politica.setVerticalPolicy(QSizePolicy.Policy.Maximum)
        self.secao_do_plsda.setSizePolicy(politica)
        dentro_da_secao = QVBoxLayout(self.secao_do_plsda)
        dentro_da_secao.setContentsMargins(0, 26, 0, 24)
        dentro_da_secao.setSpacing(16)
        # A repetição de Q² e p é dita na tela, e não só no código: quem via o par duas vezes lia
        # um defeito.
        self.nota_da_repeticao = QLabel(NOTA_DA_REPETICAO_DO_PLSDA, self.secao_do_plsda)
        self.nota_da_repeticao.setObjectName("nota_da_repeticao_do_plsda")
        self.nota_da_repeticao.setWordWrap(True)
        dentro_da_secao.addWidget(self.nota_da_repeticao)
        self.plsda: dict[str, QLabel] = {}
        self.legendas_do_plsda: dict[str, QLabel] = {}
        # O par fica empilhado: os dois títulos são longos (o do p de permutação passa dos
        # 600 px), e lado a lado a seção pediria mais largura que a aba tem a 1280 px.
        for nome in nomes_do_plsda():
            moldura_do_par = _MolduraDaQualidade(rotulo_da_metrica(nome), self.secao_do_plsda)
            moldura_do_par.setObjectName("metrica_da_qualidade")
            dentro_do_par = QVBoxLayout(moldura_do_par)
            dentro_do_par.setContentsMargins(16, 16, 16, 16)
            dentro_do_par.setSpacing(12)
            titulo = QLabel(rotulo_da_metrica(nome), moldura_do_par)
            titulo.setObjectName("titulo_da_metrica")
            titulo.setWordWrap(True)
            dentro_do_par.addWidget(titulo)
            rotulo = QLabel(f"{PREFIXO_DO_VALOR}{texto_da_metrica(nome, {})}", moldura_do_par)
            rotulo.setObjectName("valor_da_metrica")
            rotulo.setWordWrap(True)
            rotulo.setStyleSheet(estilo_do_valor(nome, {}))
            self.plsda[nome] = rotulo
            dentro_do_par.addWidget(rotulo)
            self.legendas_do_plsda[nome] = self._legenda(nome, moldura_do_par, dentro_do_par)
            dentro_da_secao.addWidget(moldura_do_par)

        # O aviso é semeado na construção, porque vale antes de qualquer execução: um aviso que só
        # aparecesse depois da primeira deixaria quem abre a aba procurando o score.
        self.aviso_do_plsda = QLabel(aviso_do_plsda(), self.secao_do_plsda)
        self.aviso_do_plsda.setWordWrap(True)
        dentro_da_secao.addWidget(self.aviso_do_plsda)
        disposicao.addWidget(self.secao_do_plsda)

        self.comparacao = QLabel(SEM_COMPARACAO, interno)
        self.comparacao.setObjectName("comparacao_da_qualidade")
        self.comparacao.setWordWrap(True)
        # `MarkdownText`, porque o texto concatena `RESSALVA_DO_ESTIMADOR`, que traz negrito e
        # crases, e `AutoText` não reconhece markdown. Por isso a ressalva vem depois de uma linha
        # em branco: com uma quebra só, o markdown a colapsaria num espaço.
        self.comparacao.setTextFormat(Qt.TextFormat.MarkdownText)
        disposicao.addWidget(self.comparacao)

        # A normalidade por variável e por grupo. O limite vem antes dos números, na mesma
        # moldura: o resumo diz quantas variáveis rejeitam a normal, e quem o lesse sem o limite
        # concluiria que as outras são normais. A tabela por variável sai em CSV, e não é navegada
        # aqui, porque são milhares de linhas por grupo. As contas e as frases estão em
        # `interface/fluxo.py`.
        self.secao_da_normalidade = _MolduraDaQualidade(TITULO_DA_NORMALIDADE, interno)
        self.secao_da_normalidade.setObjectName("tema_da_qualidade")
        politica = self.secao_da_normalidade.sizePolicy()
        politica.setVerticalPolicy(QSizePolicy.Policy.Maximum)
        self.secao_da_normalidade.setSizePolicy(politica)
        dentro_da_normalidade = QVBoxLayout(self.secao_da_normalidade)
        dentro_da_normalidade.setContentsMargins(0, 26, 0, 24)
        dentro_da_normalidade.setSpacing(16)
        self.limite_da_normalidade = QLabel(LIMITE_DA_NORMALIDADE, self.secao_da_normalidade)
        self.limite_da_normalidade.setObjectName("nota_do_campo")
        self.limite_da_normalidade.setWordWrap(True)
        self.limite_da_normalidade.setTextFormat(Qt.TextFormat.MarkdownText)
        dentro_da_normalidade.addWidget(self.limite_da_normalidade)
        # Nasce no repouso, um dos três estados de `mostra_normalidade`.
        self.resumo_da_normalidade = QLabel(NORMALIDADE_EM_REPOUSO, self.secao_da_normalidade)
        self.resumo_da_normalidade.setObjectName("resumo_da_normalidade")
        self.resumo_da_normalidade.setWordWrap(True)
        self.resumo_da_normalidade.setTextFormat(Qt.TextFormat.MarkdownText)
        dentro_da_normalidade.addWidget(self.resumo_da_normalidade)
        self.botao_da_tabela_da_normalidade = QPushButton(
            ROTULO_DA_TABELA_DA_NORMALIDADE, self.secao_da_normalidade
        )
        self.botao_da_tabela_da_normalidade.setObjectName("salva_tabela_da_normalidade")
        self.botao_da_tabela_da_normalidade.setCursor(Qt.CursorShape.PointingHandCursor)
        self.botao_da_tabela_da_normalidade.clicked.connect(self.salva_tabela_da_normalidade)
        dentro_da_normalidade.addWidget(
            self.botao_da_tabela_da_normalidade, 0, Qt.AlignmentFlag.AlignLeft
        )
        # Sem resultado não há tabela: o botão só acorda quando a raiz chega, e até lá a frase logo
        # abaixo dele diz como chegar a ela.
        desliga_com_motivo(self.botao_da_tabela_da_normalidade, MOTIVO_SEM_TABELA_DA_NORMALIDADE)
        # A confirmação, ou a falha, de gravar: a frase diz qual, porque as ações são opostas.
        self.aviso_da_tabela_da_normalidade = QLabel(self.secao_da_normalidade)
        self.aviso_da_tabela_da_normalidade.setObjectName("nota_do_campo")
        self.aviso_da_tabela_da_normalidade.setWordWrap(True)
        self.aviso_da_tabela_da_normalidade.setVisible(False)
        dentro_da_normalidade.addWidget(self.aviso_da_tabela_da_normalidade)
        disposicao.addWidget(self.secao_da_normalidade)

        # O resumo metodológico da Análise fica junto do que ele descreve: o mesmo `200` que enche
        # as métricas acima enche o resumo, no mesmo ponto da janela. As referências dos métodos
        # ficam na aba 'Métodos e limites'. O salvador é o mesmo da tabela de normalidade.
        self.secao_do_resumo = _MolduraDaQualidade(TITULO_DO_RESUMO_DA_ANALISE, interno)
        self.secao_do_resumo.setObjectName("tema_da_qualidade")
        politica = self.secao_do_resumo.sizePolicy()
        politica.setVerticalPolicy(QSizePolicy.Policy.Maximum)
        self.secao_do_resumo.setSizePolicy(politica)
        dentro_do_resumo = QVBoxLayout(self.secao_do_resumo)
        dentro_do_resumo.setContentsMargins(0, 26, 0, 24)
        dentro_do_resumo.setSpacing(16)
        self.resumo = PainelDoResumo(self.secao_do_resumo, salvador=salvador)
        dentro_do_resumo.addWidget(self.resumo)
        disposicao.addWidget(self.secao_do_resumo)

        # O espaço que sobra vai para o fim, e não repartido entre os widgets: sem esta linha o
        # `QVBoxLayout` distribui a folga por todos, e os rótulos flutuam no meio da tela com vãos
        # entre os controles.
        disposicao.addStretch(1)

        self.rolagem.setWidget(interno)

    @staticmethod
    def _legenda(nome: str, interno: QWidget, disposicao: QVBoxLayout) -> QLabel:
        """O rótulo legível de uma métrica, já no layout, recolhido sob 'Como ler'.

        `MarkdownText` e não `AutoText`: a prosa de `interface/fluxo.py` usa `**` e crases, e
        `AutoText`
        escolhe entre texto simples e HTML, sem markdown; sem esta linha a janela mostraria os
        asteriscos. Privada e estática, porque é só a construção de um widget, chamada dos dois
        laços do construtor.
        """
        # O nome em português é o título da moldura, e não entra na explicação: seria a mesma
        # frase duas vezes na mesma caixa.
        legenda = QLabel(legenda_da_metrica(nome), interno)
        legenda.setObjectName("legenda_da_metrica")
        legenda.setWordWrap(True)
        legenda.setTextFormat(Qt.TextFormat.MarkdownText)
        # 9 pt contra os 13 pt do valor: a hierarquia é do widget, e não do texto.
        legenda.setStyleSheet(ESTILO_DA_LEGENDA)
        # 'Como ler', recolhido: o rótulo fica no layout do cartão (é o que `texto_visivel` exige)
        # e nasce escondido; o botão por cima dele o abre.
        abre = QToolButton(interno)
        abre.setObjectName("alterna")
        abre.setText(ROTULO_DO_COMO_LER)
        abre.setCheckable(True)
        abre.setArrowType(Qt.ArrowType.RightArrow)
        abre.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        abre.setCursor(Qt.CursorShape.PointingHandCursor)
        expansao = ExpansaoSuave(legenda)
        abre.toggled.connect(expansao.alterna)
        abre.toggled.connect(
            lambda aberto, b=abre: b.setArrowType(
                Qt.ArrowType.DownArrow if aberto else Qt.ArrowType.RightArrow
            )
        )
        legenda.setVisible(False)
        disposicao.addWidget(abre, 0, Qt.AlignmentFlag.AlignRight)
        disposicao.addWidget(legenda)
        return legenda

    # ------------------------------------------------------------------
    # Nenhuma métrica única: o conjunto completo
    # ------------------------------------------------------------------

    def mostra_qualidade(self, relatorio: Any) -> dict[str, str]:
        """Escreve todas as métricas e devolve o que escreveu, por nome.

        O valor de retorno é conveniência; o oráculo do teste é o widget. Um `setText(f"{nome}:")`
        que deixasse o dicionário intacto passaria em toda asserção que lesse o retorno, com o
        painel mostrando nomes sem valores.
        """
        corpo = relatorio if isinstance(relatorio, dict) else {}
        escritos: dict[str, str] = {}
        for nome, rotulo in self.metricas.items():
            texto = texto_da_metrica(nome, corpo)
            rotulo.setText(f"{PREFIXO_DO_VALOR}{texto}")
            rotulo.setStyleSheet(estilo_do_valor(nome, corpo))
            escritos[nome] = texto
        return escritos

    # ------------------------------------------------------------------
    # PLS-DA nunca sem validação: o par, ou nada
    # ------------------------------------------------------------------

    def mostra_plsda(self, relatorio: Any) -> dict[str, str]:
        """O score do PLS-DA e a validação dele, do mesmo laço. Devolve o que escreveu.

        Sem score, recusa desenhar o par: os rótulos ficam no estado de ausência, que é o que eles
        já dizem, e o aviso explica. Com score, o par sai inteiro, porque o laço é sobre
        `nomes_do_plsda()` e não há caminho que escreva um sem o outro.

        O aviso vem de `aviso_do_plsda(relatorio=corpo)`: `SEM_PLSDA` quando o par de grupos não
        veio, e a procedência quando veio. O terceiro desfecho dele, `SEM_CAMINHO_PARA_PLSDA`,
        exige que `CAMINHO_DA_METRICA` dê 'nenhum' para os dois membros do par, e hoje dá `api`;
        este painel nunca injeta `caminhos`, então aquele ramo é inalcançável daqui, e quem o
        exercita é o teste de `aviso_do_plsda` com a tabela substituída.
        """
        corpo = relatorio if isinstance(relatorio, dict) else {}
        nomes = nomes_do_plsda()

        # O relatório entra na chamada: sem ele a linha não distingue 'esta execução não pediu' de
        # 'o produto não calcula'.
        self.aviso_do_plsda.setText(aviso_do_plsda(relatorio=corpo))

        escritos: dict[str, str] = {}
        for score, _exigidos in PARES_DE_VALIDACAO_OBRIGATORIA:
            if corpo.get(score) is None:
                continue
            for nome in nomes:
                texto = texto_da_metrica(nome, corpo)
                self.plsda[nome].setText(f"{PREFIXO_DO_VALOR}{texto}")
                self.plsda[nome].setStyleSheet(estilo_do_valor(nome, corpo))
                escritos[nome] = texto
        return escritos

    # ------------------------------------------------------------------
    # O antes-e-depois
    # ------------------------------------------------------------------

    def mostra_comparacao(self, comparacao: Any) -> str:
        """O par de medianas do RSD dos QCs, e devolve o texto escrito."""
        texto = texto_da_comparacao(comparacao)
        self.comparacao.setText(texto)
        return texto

    # ------------------------------------------------------------------
    # A normalidade por variável e por grupo
    # ------------------------------------------------------------------

    def mostra_normalidade(self, resposta: Any) -> str:
        """O resumo por grupo da execução, ou qual das duas ausências. Devolve o que escreveu.

        Recebe o corpo inteiro e não a raiz, porque a raiz a `None` não distingue os dois estados
        de ausência: sem leitura completa (repouso, erro) o corpo é vazio, e com a execução sem a
        família ele traz tudo menos ela. As ações são opostas: no primeiro a pessoa espera, no
        segundo marca a caixa.

        As chaves da raiz são lidas aqui, e é isso que o portão do contrato da janela confere
        contra `ResumoDaNormalidade`; o texto e o CSV saem de `interface/fluxo.py`.
        """
        corpo = resposta if isinstance(resposta, dict) else {}
        normalidade = corpo.get("normalidade")
        if not corpo:
            self._normalidade = None
            texto = NORMALIDADE_EM_REPOUSO
        elif not isinstance(normalidade, dict):
            self._normalidade = None
            texto = NORMALIDADE_NAO_PEDIDA
        else:
            self._normalidade = normalidade
            texto = prosa_dos_grupos_da_normalidade(
                normalidade.get("grupos"),
                alfa=normalidade.get("alfa"),
                nivel_do_fdr=normalidade.get("nivel_do_fdr"),
                lambda_de_storey=normalidade.get("lambda_de_storey"),
            )
        self.resumo_da_normalidade.setText(texto)
        if self._normalidade is None:
            desliga_com_motivo(
                self.botao_da_tabela_da_normalidade, MOTIVO_SEM_TABELA_DA_NORMALIDADE
            )
        else:
            religa(self.botao_da_tabela_da_normalidade)
        # A frase da gravação anterior fala de outra execução, e sai com ela.
        self.aviso_da_tabela_da_normalidade.setVisible(False)
        return texto

    def salva_tabela_da_normalidade(self) -> Path | None:
        """Grava a tabela por variável e por grupo da última execução, onde a pessoa escolher.

        Devolve o destino, ou `None` se ela cancelou, se não há tabela, ou se a escrita falhou; a
        frase diz qual: quem cancelou não precisa de nada, e quem não conseguiu gravar precisa
        escolher outro lugar.
        """
        normalidade = self._normalidade
        if normalidade is None:
            return None
        # O diálogo de produção é o de `arquivos.py`, com o título do botão.
        salvador = self._salvador or salvador_por_dialogo(
            self.window(), titulo=ROTULO_DA_TABELA_DA_NORMALIDADE.rstrip("…")
        )
        destino = salvador(NOME_DA_TABELA_DA_NORMALIDADE)
        if destino is None:
            return None
        texto = csv_da_normalidade(normalidade.get("features"), normalidade.get("grupos"))
        try:
            destino.write_text(texto, encoding="utf-8", newline="")
        except OSError as erro:
            self.aviso_da_tabela_da_normalidade.setText(
                f"Não foi possível salvar a tabela em {destino}: {type(erro).__name__}"
            )
            self.aviso_da_tabela_da_normalidade.setVisible(True)
            return None
        self.aviso_da_tabela_da_normalidade.setText(f"Tabela salva em {destino}")
        self.aviso_da_tabela_da_normalidade.setVisible(True)
        return destino

    # ------------------------------------------------------------------
    # O que está visível, o oráculo dos testes
    # ------------------------------------------------------------------

    def texto_visivel(self) -> dict[str, str]:
        """O que cada rótulo mostra agora, e só o que está no layout.

        É este o oráculo dos testes, e não o retorno dos `mostra_*`: um rótulo fora do layout, com
        o nome e sem o valor, ou com o valor de outra métrica, passaria em toda asserção que lesse
        o retorno. Um rótulo que existe e não está no layout é um painel em branco, e o dicionário
        de retorno não sabe disso.
        """
        visivel: dict[str, str] = {}
        for nome, rotulo in {**self.metricas, **self.plsda}.items():
            visivel[nome] = rotulo.text() if self._no_layout(rotulo) else ""
        # As legendas entram com prefixo, e o prefixo é contrato com o arnês: ele separa o que é
        # valor do que é moldura, e sem isso a contagem dos estados somaria legendas que não são
        # estado nenhum.
        for nome, rotulo in self.legendas.items():
            visivel[f"__legenda__{nome}"] = rotulo.text() if self._no_layout(rotulo) else ""
        for nome, rotulo in self.legendas_do_plsda.items():
            visivel[f"__legenda_plsda__{nome}"] = rotulo.text() if self._no_layout(rotulo) else ""
        # O título da moldura entra no que é visível, porque o nome em português da métrica é o
        # título do `QGroupBox` que contém o valor e a explicação. O prefixo `__moldura__` é
        # contrato com o arnês, como `__legenda__`. São dois prefixos, um para a lista e outro
        # para a seção do PLS-DA, porque `plsda_q2` e `plsda_permutation_p` estão nos dois
        # dicionários: com uma chave só por nome, a segunda moldura sobrescreveria a primeira.
        for nome, rotulo in self.metricas.items():
            visivel[f"__moldura__{nome}"] = self._titulo_da_moldura(rotulo)
        # `nomes_do_plsda()`, e não `self.plsda.items()`: o portão do PLS-DA permite uma forma só
        # de alcançar o par, o subscrito cuja chave é a variável do laço canônico, e uma
        # enumeração paralela seria um segundo caminho.
        for nome in nomes_do_plsda():
            visivel[f"__moldura_plsda__{nome}"] = self._titulo_da_moldura(self.plsda[nome])
        # Os cabeçalhos de tema entram no que é visível: sem eles o arnês não separa 'a partição
        # está na tabela' de 'a partição chegou à tela'. O prefixo `__tema__` é contrato com o
        # arnês, pela mesma razão dos outros.
        for chave, moldura in self.temas.items():
            visivel[f"__tema__{chave}"] = moldura.title() if self._no_layout(moldura) else ""
        for chave, rotulo_da_nota in self.notas_dos_temas.items():
            visivel[f"__nota_tema__{chave}"] = (
                rotulo_da_nota.text() if self._no_layout(rotulo_da_nota) else ""
            )
        # A moldura da seção dedicada do PLS-DA, que não é um tema: sem esta chave o arnês não
        # separa 'a seção existe' de 'o par está na lista'.
        visivel["__secao_plsda__"] = (
            self.secao_do_plsda.title() if self._no_layout(self.secao_do_plsda) else ""
        )
        visivel["__aviso__"] = (
            self.aviso_do_plsda.text() if self._no_layout(self.aviso_do_plsda) else ""
        )
        visivel["__comparacao__"] = (
            self.comparacao.text() if self._no_layout(self.comparacao) else ""
        )
        # A seção da normalidade em três chaves, porque o portão da ordem precisa separá-las: o
        # limite tem de estar no layout antes do resumo, e o título da moldura diz que a seção
        # existe.
        visivel["__secao_normalidade__"] = (
            self.secao_da_normalidade.title() if self._no_layout(self.secao_da_normalidade) else ""
        )
        visivel["__limite_normalidade__"] = (
            self.limite_da_normalidade.text() if self._no_layout(self.limite_da_normalidade) else ""
        )
        visivel["__normalidade__"] = (
            self.resumo_da_normalidade.text() if self._no_layout(self.resumo_da_normalidade) else ""
        )
        return visivel

    def _titulo_da_moldura(self, rotulo: QLabel) -> str:
        """O título do `QGroupBox` que contém este rótulo, ou vazio se ele não está desenhado.

        Vazio nos dois casos, de propósito: rótulo fora do layout e rótulo sem moldura se leem
        igual para quem pergunta 'o nome em português está na tela?', e a resposta é não nos dois.
        """
        pai = rotulo.parentWidget()
        if not self._no_layout(rotulo) or not isinstance(pai, QGroupBox):
            return ""
        return pai.title()

    @staticmethod
    def _no_layout(rotulo: QWidget) -> bool:
        """Se este widget está de fato num layout, e não só construído.

        Um widget com pai mas fora do layout não é desenhado; a distinção separa 'o painel mostra'
        de 'o painel construiu'. Recebe `QWidget`, e não só `QLabel`, porque a moldura de um tema
        é um `QGroupBox`, e a pergunta é a mesma.
        """
        pai = rotulo.parentWidget()
        disposicao = pai.layout() if pai is not None else None
        return disposicao is not None and disposicao.indexOf(rotulo) >= 0

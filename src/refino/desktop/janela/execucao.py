"""As páginas das etapas Correção, Normalização e Análise estatística.

Cada página tem três blocos numerados que rolam juntos, como os de 'Preparar o estudo': 1
'Ajustes', com 'Seguir sugestões do app', a matriz que entra e os parâmetros; 2 'Execução da
etapa', com o botão, o estado, a barra e o Resultado; e 3, os gráficos da etapa. Há uma rolagem
só, sem rodapé fixo. Sem os dois arquivos do estudo, a página inteira dá lugar a um estado vazio
que diz qual falta e leva a 'Preparar o estudo', em vez de um formulário cujo botão não leva a
lugar nenhum.

A justificativa das propostas vem do `PainelDeLeitura`, que constrói os rótulos e não os põe no
próprio layout; aqui ela abre num quadro, pelo '?' ao lado da sugestão, junto do controle que
explica. O controle repousa no que a máquina propôs, e mudar o valor é o gesto de sobrescrever
(`PainelDePipeline.eixos_divergentes`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QMouseEvent, QShowEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.etapas import ENTRADAS_DA_ETAPA, ETAPAS, EstadoDasEtapas
from refino.desktop.janela.arquivos import CAMPOS_DOS_ARQUIVOS, salvador_por_dialogo
from refino.desktop.janela.artefatos import (
    ARTEFATOS_DA_JANELA,
    CHAVE_DA_ALTERNATIVA,
    CHAVES_DO_RESULTADO,
    EM_CURSO,
    FORMATO_METABOANALYST,
    FORMATOS,
    NOMES_DOS_FORMATOS,
    SEM_ARTEFATO,
    PainelDeArtefatos,
)
from refino.desktop.janela.componentes import (
    Divisor,
    FaixaDeAviso,
    NivelDeAviso,
    Recolhivel,
    RotuloQueSome,
    Secao,
    com_maiuscula,
    desliga_com_motivo,
    religa,
)
from refino.desktop.janela.leitura import NOMES_DOS_CORRETORES, TITULO_DA_JUSTIFICATIVA
from refino.desktop.janela.pipeline import DICA_DO_LOTE
from refino.desktop.janela.tema import SegueAAltura
from refino.interface.referencias import SEM_RESUMO, SEM_RESUMO_DA_ETAPA, ResumoMetodologico

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from refino.desktop.janela.artefatos import EscolhedorDePasta

__all__ = [
    "AVISO_SEM_IMPUTACAO",
    "EXPLICACOES_DAS_SUGESTOES",
    "FILTRO_DO_RESUMO",
    "FRASES_DO_RESULTADO",
    "FRASE_DA_ENTRADA_DA_CORRECAO",
    "FRASE_DA_RECUSA_SABIDA",
    "MEDINDO_A_SUGESTAO",
    "MOTIVOS_SEM_ETAPA",
    "MOTIVO_DO_ESCALONAMENTO",
    "MOTIVO_SEM_RESUMO",
    "NOME_SUGERIDO_DO_RESUMO",
    "PERGUNTAS_DA_ENTRADA",
    "TEXTOS_DE_REFAZER",
    "TEXTOS_DO_DOWNLOAD",
    "TEXTO_DA_ALTERNATIVA",
    "TEXTO_DA_CAIXA_DO_RESUMO",
    "TEXTO_DO_BOTAO_DAS_SUGESTOES",
    "TEXTO_DO_BOTAO_DA_MAIS_TRATADA",
    "TEXTO_DO_BOTAO_DO_PREPARO",
    "TEXTO_DO_LOTE",
    "TITULOS_DA_FALHA",
    "TITULO_DO_BLOCO_DA_EXECUCAO",
    "TITULO_DO_BLOCO_DOS_AJUSTES",
    "TITULO_DO_BLOCO_DO_RESUMO",
    "TITULO_DO_DIALOGO_DO_RESUMO",
    "AbaDeEtapa",
    "BlocoDeSugestoes",
    "PainelDoResumo",
    "QuadroDaJustificativa",
    "ResultadoDaEtapa",
    "SeletorDaEntrada",
    "VistaDaEntrada",
    "aviso_sem_imputacao",
    "frase_da_entrada_fora_de_dia",
    "frase_da_entrada_unica",
    "frase_da_etapa_fora_de_dia",
    "frase_da_indicacao",
    "frase_das_sugestoes_da_correcao",
    "frase_do_estado_vazio",
    "frase_dos_valores_iniciais",
]

# O nome que o diálogo de salvar o resumo sugere.
NOME_SUGERIDO_DO_RESUMO: Final = "resumo-metodologico.txt"

SalvadorDeTexto = Callable[[str], Path | None]
"""O diálogo de salvar: recebe o nome sugerido e devolve o destino, ou `None` se cancelou.

É a assinatura de `figuras.EscolhedorDeArquivo` e de `arquivos.Salvador`, e a janela injeta o mesmo
duplo nos três botões que salvam.
"""

# O diálogo de produção do resumo é `arquivos.salvador_por_dialogo` com este título e este
# filtro.
TITULO_DO_DIALOGO_DO_RESUMO: Final = "Salvar o resumo metodológico"
FILTRO_DO_RESUMO: Final = "Texto (*.txt)"

# As frases das páginas de etapa nomeiam cada etapa pelo nome da aba, e não da rota: a etapa 2
# é o pré-processamento na rota e 'Normalização' na tela, e é a tela que a pessoa lê.
_NOMES_NAS_FRASES: Final[dict[int, str]] = {1: "a correção", 2: "a normalização", 3: "a análise"}

# A Entrada diz que matriz a etapa usa, e a pessoa a escolhe; não há escolha automática. A
# Correção parte sempre da matriz carregada. A Normalização e a Análise partem da entrada
# escolhida: com uma só, a frase diz qual e por que não há outra; com duas ou três, um seletor
# as nomeia pelo que são.
FRASE_DA_ENTRADA_DA_CORRECAO: Final = "Usando a matriz carregada em «Preparar o estudo»."
PERGUNTAS_DA_ENTRADA: Final[dict[int, str]] = {
    2: "Escolha a matriz que entra na normalização:",
    3: "Escolha a matriz que entra na análise:",
}
TEXTO_DO_BOTAO_DA_MAIS_TRATADA: Final = "Usar esta"
# O aviso dos zeros não desliga nada: a etapa roda, com um aviso curto. A original e a
# corrigida não passam pela imputação, e uma normalizada pode não ter passado; o que trata os
# zeros é normalizar com uma imputação.
AVISO_SEM_IMPUTACAO: Final = (
    "Esta matriz não passou pela imputação: os zeros entram como valor e podem distorcer a PCA, o "
    "PLS-DA e os testes."
)
_COMO_TRATAR_OS_ZEROS: Final[dict[str, str]] = {
    "original": "Para tratá-los, normalize antes.",
    "corrigida": "Para tratá-los, normalize antes.",
    "normalizada": "Para tratá-los, normalize de novo com uma imputação.",
}

# Os avisos de etapa são amarelos, e os críticos vermelhos. O verbo de cada etapa vai no
# infinitivo e no imperativo: o botão do aviso refaz a etapa, e o título da falha diz o que não
# foi possível.
_VERBOS: Final[dict[int, tuple[str, str]]] = {
    1: ("corrigir", "corrija"),
    2: ("normalizar", "normalize"),
    3: ("analisar", "analise"),
}
TEXTOS_DE_REFAZER: Final[dict[int, str]] = {
    etapa: f"{com_maiuscula(infinitivo)} de novo" for etapa, (infinitivo, _) in _VERBOS.items()
}
TITULOS_DA_FALHA: Final[dict[int, str]] = {
    etapa: infinitivo for etapa, (infinitivo, _) in _VERBOS.items()
}
# A etapa que cada entrada derivada é, e a matriz pelo que ela é, no vocabulário de
# `etapas.ENTRADAS`. Uma entrada fora dele levanta `KeyError` aqui, em vez de cair numa frase.
_ETAPA_DA_ENTRADA_NA_FRASE: Final[dict[str, str]] = {
    "corrigida": "a correção",
    "normalizada": "a normalização",
}
_MATRIZ_DA_ENTRADA: Final[dict[str, str]] = {
    "corrigida": "A matriz corrigida",
    "normalizada": "A matriz normalizada",
}
_NESTA_ETAPA: Final[dict[int, str]] = {2: "nesta normalização", 3: "nesta análise"}

# O Resultado diz, em cada estado, onde está o resultado ou o que falta para ele existir. Na
# Análise o lugar é o 'Próximo passo': ela não tem arquivo a baixar, e o resultado mora em
# outras abas. Os estados são o vocabulário fechado de `etapas.EstadoDaEtapa`: um estado fora
# dele levanta `KeyError` aqui, em vez de cair numa frase qualquer.
FRASES_DO_RESULTADO: Final[dict[int, dict[str, str]]] = {
    1: {
        "nao_feita": "Rode a correção para ver o resultado aqui.",
        "em_curso": "A correção está em curso: o resultado aparece aqui quando ela terminar.",
        "em_dia": "A correção está em dia. As figuras dela estão logo abaixo, nesta aba.",
        "desatualizada": (
            "Os parâmetros mudaram depois da correção: rode-a de novo para atualizar o resultado."
        ),
    },
    2: {
        "nao_feita": "Rode a normalização para ver o resultado aqui.",
        "em_curso": "A normalização está em curso: o resultado aparece aqui quando ela terminar.",
        "em_dia": "A normalização está em dia. As figuras dela estão logo abaixo, nesta aba.",
        "desatualizada": (
            "Os parâmetros mudaram depois da normalização: rode-a de novo para atualizar o "
            "resultado."
        ),
    },
    3: {
        "nao_feita": "Rode a análise para ver os resultados.",
        "em_curso": "A análise está em curso: os resultados aparecem aqui quando ela terminar.",
        "em_dia": "A análise terminou. Veja os resultados:",
        "desatualizada": (
            "Os resultados são da última análise que correu, e os parâmetros mudaram depois: "
            "rode-a de novo para atualizá-los."
        ),
    },
}
# Os títulos dos blocos 1 e 2, iguais nas três páginas. O do bloco 3 é o da vista dos gráficos
# (`graficos.TITULO_DOS_GRAFICOS_DA_ETAPA`), que traz o próprio bloco.
TITULO_DO_BLOCO_DOS_AJUSTES: Final = "Ajustes"
TITULO_DO_BLOCO_DA_EXECUCAO: Final = "Execução da etapa"
# O subtítulo do Resultado, dentro do bloco 2: é o desfecho da ação logo acima dele, e por isso
# não é um bloco à parte.
TITULO_DO_RESULTADO: Final = "Resultado"
TITULO_DO_PROXIMO_PASSO: Final = "Próximo passo"
TEXTO_DO_BOTAO_DA_QUALIDADE: Final = "Ver a qualidade do resultado"
TEXTO_DO_BOTAO_DOS_GRAFICOS: Final = "Abrir em Gráficos"
TEXTO_DO_BOTAO_DO_PREPARO: Final = "Ir para Preparar o estudo"

# 'Seguir sugestões do app', no topo do bloco 1 da Correção e da Normalização. A frase ao lado
# diz o que o botão preenche, e cada uma diz só o que é verdade:
#
# - na Correção a máquina tem opinião medida sobre dois eixos, e só dois (`PROPOSTAS_POR_ROTA`):
#   o corretor, que o Palpite mede em 200 features, e o número de lotes, que o 'Diagnóstico'
#   propõe;
# - na Normalização ela não tem: os eixos do pré-processamento e os cortes não são propostos
#   por rota nenhuma, e o botão volta os campos aos valores com que a tela abre. A frase os
#   chama de pontos de partida, e não de recomendação.
#
# A Análise não tem o botão: nenhum dos controles dela tem proposta, nem valor inicial que
# alguém precise repor com um clique. As chaves desta tabela decidem que páginas têm o bloco.
TEXTO_DO_BOTAO_DAS_SUGESTOES: Final = "Seguir sugestões do app"
EXPLICACOES_DAS_SUGESTOES: Final[dict[int, str]] = {
    1: (
        "Preenche o corretor com o do Palpite — o que deixou os QCs mais estáveis em 200 features "
        "desta matriz — e o número de lotes com o que o «Diagnóstico» propôs."
    ),
    2: (
        "Volta os métodos, as frações e os cortes aos valores iniciais do app. São pontos de "
        "partida, e não uma escolha medida para esta matriz."
    ),
}
# A frase ao lado do botão de sugestões enquanto o Palpite mede: diz por que ele está desligado
# (`desliga_com_motivo`). O Palpite é independente da aba, e o que a pessoa escolhe durante a
# espera fica (`frase_do_que_voce_mudou_durante_o_palpite`).
MEDINDO_A_SUGESTAO: Final = (
    "Medindo a sugestão… o Palpite corre os três corretores em 200 features, perto de um minuto; "
    "os campos são preenchidos quando ele chegar, menos o que você mudar até lá."
)
# A sugestão ao lado do botão diz só o corretor e os lotes indicados (`frase_da_indicacao`), e
# aparece quando as leituras de 'Preparar o estudo' já correram. O '?' ao lado dela abre o
# porquê (`QuadroDaJustificativa`).
TEXTO_DO_PORQUE: Final = "?"
# A largura máxima do quadro do porquê: o texto quebra linha nela, como numa dica de mouse larga.
LARGURA_DA_JUSTIFICATIVA: Final = 460
# O que o quadro guarda entre ele e a borda da tela, quando é deslocado para caber nela, e entre ele
# e o '?' que o abriu.
_FOLGA_DO_QUADRO: Final = 8
_ABAIXO_DO_PORQUE: Final = 4

# O Resultado da Correção e o da Normalização: o arquivo da etapa, nos dois formatos. O download
# leva a etapa feita, e não os controles: o que se baixa é o que a etapa mostrou (ver
# `artefatos.campos_do_download`).
TEXTOS_DO_DOWNLOAD: Final[dict[int, str]] = {
    1: "Baixar a matriz corrigida",
    2: "Baixar a matriz normalizada",
}
# Antes de a etapa ser feita não há o que baixar, e a frase diz o gesto que o liga.
MOTIVOS_SEM_ETAPA: Final[dict[int, str]] = {
    1: "Corrija a matriz para baixar o resultado.",
    2: "Normalize a matriz para baixar o resultado.",
}
# 'Gerar os três' corre os três corretores por conta própria, e por isso fica ligado assim que
# há os arquivos: não depende da correção feita.
TEXTO_DO_LOTE: Final = "Gerar os três"
TEXTO_DA_CAIXA_DO_RESUMO: Final = "Salvar também o resumo dos métodos (.txt)"
TITULO_DO_BLOCO_DO_RESUMO: Final = "Resumo dos métodos do arquivo"
MOTIVO_SEM_RESUMO: Final = "Gere um arquivo para ter o resumo."
# A recusa sabida antes do clique. Com escalonamento a matriz normalizada sempre tem negativo,
# porque todo método de escalonamento centra cada feature na média, e o formato MetaboAnalyst
# só aceita positivos. Nada foi pedido ainda, então o aviso é amarelo; o botão principal fica
# desligado com o motivo, e o da faixa baixa a alternativa, que se apresenta como alternativa
# à recusa.
FRASE_DA_RECUSA_SABIDA: Final = (
    "O MetaboAnalyst recebe só valores positivos e faz a própria transformação e o próprio "
    "escalonamento. Como alternativa, baixe a matriz antes dessas duas etapas."
)
TEXTO_DA_ALTERNATIVA: Final = "Baixar a alternativa (antes da transformação e do escalonamento)"
MOTIVO_DO_ESCALONAMENTO: Final = (
    "Com escalonamento, o formato MetaboAnalyst recusa a matriz: baixe a alternativa no aviso "
    "abaixo, ou escolha o formato do laboratório."
)
# Os estados em que há resultado, o da última corrida, em dia ou não: os dois em que o 'Próximo
# passo' da Análise mostra os botões que levam às abas dos resultados.
_COM_RESULTADO: Final = frozenset({"em_dia", "desatualizada"})
# Os estados de onde chegar a 'em dia' é uma corrida que acabou. A volta de 'desatualizada' é a
# de parâmetros que voltaram ao que eram, e não deve arrancar a página de quem os edita.
_DE_ONDE_ROLA: Final = frozenset({"nao_feita", "em_curso"})


def frase_do_estado_vazio(etapa: int, faltam: tuple[str, ...]) -> str:
    """O que falta para usar a etapa, pelos nomes que os cartões de 'Preparar o estudo' mostram.

    `faltam` são chaves de `CAMPOS_DOS_ARQUIVOS` (`PainelDeArquivos.metades_que_faltam`). Uma chave
    fora dele levanta `KeyError` em vez de cair num nome qualquer.
    """
    nomes = [f"«{CAMPOS_DOS_ARQUIVOS[chave][0]}»" for chave in faltam]
    uso = _NOMES_NAS_FRASES[etapa]
    if len(nomes) == 1:
        return f"Falta o arquivo {nomes[0]}: carregue-o em «Preparar o estudo» para usar {uso}."
    return (
        f"Faltam os arquivos {' e '.join(nomes)}: carregue-os em «Preparar o estudo» para usar "
        f"{uso}."
    )


def frase_da_entrada_unica(etapa: int, ausencias: Sequence[str]) -> str:
    """A Entrada da Normalização ou da Análise quando só a original existe, e por que não há outra.

    `ausencias` são as frases de `EstadoDasEtapas.ausencia` das entradas derivadas que faltam,
    como 'a correção não foi feita', que dizem o que fazer para a outra aparecer.
    """
    sem = "sem correção" if etapa == 2 else "sem correção nem normalização"
    return f"Usando a matriz {sem} — {' e '.join(ausencias)}."


def aviso_sem_imputacao(entrada: str) -> str:
    """O aviso dos zeros da Análise sobre `entrada`, com o que os trata."""
    return f"{AVISO_SEM_IMPUTACAO} {_COMO_TRATAR_OS_ZEROS[entrada]}"


def texto_dos_lotes(lotes: int) -> str:
    """'1 lote', '2 lotes': o número de lotes como as frases das sugestões o escrevem.

    Um lugar só para o plural: a frase ao lado do botão e a do desfecho estão na mesma tela, e
    uma mudança de redação feita numa só as faria discordar.
    """
    return f"{lotes} {'lote' if lotes == 1 else 'lotes'}"


def frase_das_sugestoes_da_correcao(corretor: str | None, lotes: int | None) -> str:
    """O que 'Seguir sugestões do app' pôs nos campos da Correção.

    `corretor` é o valor que o seletor aceitou do Palpite, e `lotes` o número que o
    'Diagnóstico' propôs; `None` quer dizer que não havia proposta para pôr, e a frase diz isso
    em vez de afirmar uma sugestão que não chegou. O nome do corretor é o que o seletor desenha:
    um valor fora do vocabulário levanta `KeyError`, em vez de cair num nome qualquer.
    """
    if corretor is None:
        return (
            "O Palpite chegou sem um corretor que o seletor ofereça, e os campos ficaram como "
            "estavam."
        )
    nome = NOMES_DOS_CORRETORES[corretor]
    if lotes is None:
        # 'não propôs', e não 'ainda não propôs': o Diagnóstico roda antes de as sugestões
        # entrarem, e um 'ainda' sugeriria que falta rodá-lo.
        return (
            f"Sugestões aplicadas: corretor «{nome}». O número de lotes ficou como estava: o "
            "«Diagnóstico» não propôs lotes para esta matriz."
        )
    return f"Sugestões aplicadas: corretor «{nome}», {texto_dos_lotes(lotes)}."


def frase_da_indicacao(corretor: str | None, lotes: int | None) -> str:
    """A sugestão ao lado de 'Seguir sugestões do app': o corretor e os lotes indicados.

    `corretor` é o do Palpite em dia para a entrada atual e `lotes` o número que o 'Diagnóstico'
    deixou guardado; quem decide quais valem é a janela, com a condição em que o clique aplica
    sem medir de novo. `None` é 'não há o que indicar'; sem os dois a frase é vazia, e o rótulo
    some.

    O nome é o que o seletor desenha, o mesmo das frases do desfecho: um corretor fora do
    vocabulário levanta `KeyError`, como em `frase_das_sugestoes_da_correcao`.
    """
    partes = []
    if corretor is not None:
        partes.append(NOMES_DOS_CORRETORES[corretor])
    if lotes is not None:
        partes.append(texto_dos_lotes(lotes))
    return f"Indicado: {', '.join(partes)}" if partes else ""


def frase_do_que_voce_mudou_durante_o_palpite(
    corretor: str | None,
    lotes: int | None,
    *,
    corretor_sugerido: str | None,
    corretor_mantido: bool,
    lotes_mantidos: bool,
) -> str:
    """As sugestões que entraram, e o que ficou como a pessoa deixou durante a espera.

    O Palpite é independente da aba 'Correção': mexer num campo enquanto ele mede não o para nem
    muda o que ele mede. O que a pessoa escolheu no intervalo também não é sobrescrito quando ele
    chega, porque ela escolheu depois de pedir as sugestões; a frase diz o que o Palpite sugeriu,
    para um segundo clique o seguir. `corretor` e `lotes` são o que de fato entrou, ou `None`.
    """
    aplicadas = []
    if corretor is not None:
        aplicadas.append(f"corretor «{NOMES_DOS_CORRETORES[corretor]}»")
    if lotes is not None:
        aplicadas.append(texto_dos_lotes(lotes))
    partes = []
    if aplicadas:
        partes.append(f"Sugestões aplicadas: {', '.join(aplicadas)}.")
    mudados = [
        nome
        for nome, mantido in (
            ("o corretor", corretor_mantido),
            ("o número de lotes", lotes_mantidos),
        )
        if mantido
    ]
    if mudados:
        sujeito = " e ".join(mudados)
        verbo = "ficaram" if len(mudados) > 1 else "ficou"
        partes.append(
            f"{sujeito[0].upper()}{sujeito[1:]} que você escolheu enquanto o Palpite media {verbo} "
            "como você deixou."
        )
    if corretor_mantido and corretor_sugerido in NOMES_DOS_CORRETORES:
        partes.append(
            f"O Palpite sugeriu «{NOMES_DOS_CORRETORES[corretor_sugerido]}»: clique de novo para "
            "segui-lo."
        )
    return " ".join(partes)


def frase_dos_valores_iniciais(mudados: int) -> str:
    """O que 'Seguir sugestões do app' fez nos campos da Normalização.

    `mudados` é quantos campos o clique de fato mudou. Zero diz que já estavam nos valores
    iniciais, e não 'aplicadas' sobre nada: um clique que não mexe em nada e diz que mexeu é
    indistinguível de um que não funciona.
    """
    if mudados == 0:
        return "Os métodos, as frações e os cortes já estavam nos valores iniciais do app."
    campos = "campo voltou" if mudados == 1 else "campos voltaram"
    # Sem 'Sugestões aplicadas': a frase de baixo do botão diz que estes valores são pontos de
    # partida e não uma escolha medida, e o desfecho não pode chamá-los de sugestões.
    return f"{mudados} {campos} ao valor inicial do app."


def frase_da_etapa_fora_de_dia(
    etapa: int, *, entrada: str, matriz_mudou: bool, so_a_entrada_refeita: bool
) -> str:
    """O aviso da etapa feita que já não vale, com a causa: a matriz, os parâmetros ou a entrada.

    `entrada` é a entrada com que ela rodou; `so_a_entrada_refeita`, se o único motivo de
    `EstadoDasEtapas.motivos` é a etapa de entrada ter sido refeita. Os parâmetros da Normalização
    e da Análise incluem a entrada escolhida: trocá-la é mudar a etapa.
    """
    infinitivo, imperativo = _VERBOS[etapa]
    if matriz_mudou:
        causa = "A matriz carregada mudou"
    elif so_a_entrada_refeita:
        causa = f"{com_maiuscula(_ETAPA_DA_ENTRADA_NA_FRASE[entrada])} foi refeita com outros "
        causa += "parâmetros"
    else:
        causa = (
            "Os parâmetros mudaram"
            if etapa == ETAPAS[0]
            else "Os parâmetros ou a matriz de entrada mudaram"
        )
    return f"{causa} depois de {infinitivo}; {imperativo} de novo para o resultado valer."


def frase_da_entrada_fora_de_dia(etapa: int, entrada: str, alvo: int, aba_do_alvo: str) -> str:
    """O aviso da etapa em dia cuja entrada está fora de dia, e por onde começar.

    `alvo` é a etapa mais a montante fora de dia: refazer a de baixo antes repetiria o trabalho.
    """
    _, imperativo = _VERBOS[etapa]
    return (
        f"{_MATRIZ_DA_ENTRADA[entrada]}, que entra {_NESTA_ETAPA[etapa]}, está fora de dia. Comece "
        f"por refazer {_NOMES_NAS_FRASES[alvo]} na aba «{aba_do_alvo}»; depois {imperativo} de "
        "novo."
    )


@dataclass(frozen=True)
class VistaDaEntrada:
    """O que a seção 'Entrada' mostra: a frase, as entradas com o nome de cada uma, a escolhida e os
    dois avisos dela.

    `opcoes` é `(valor, nome)` por entrada que existe; com uma só, não há seletor. `mais_tratada` é
    a entrada mais tratada que existe sem estar escolhida; `volta`, o porquê de a escolhida ter
    deixado de existir; `imputacao`, o aviso dos zeros.
    """

    frase: str
    opcoes: tuple[tuple[str, str], ...] = ()
    escolhida: str = "original"
    mais_tratada: tuple[str, str] | None = None
    volta: str | None = None
    imputacao: str | None = None


class SeletorDaEntrada(QFrame):
    """As entradas de uma etapa em segmentos, um por entrada, nomeados pelo que ela é.

    Os segmentos nascem com a página, um por entrada que a etapa admite, e depois só se escondem
    e se renomeiam, sem destruir nenhum. O valor de cada um é o de `etapas.ENTRADAS`; o texto é o
    nome que a janela dá, 'Corrigida — LOESS'. O clique emite `escolhida` com o valor, e quem
    guarda a escolha é a janela.
    """

    escolhida = Signal(str)

    def __init__(
        self, entradas: Sequence[str], pai: QWidget | None = None, *, assunto: str = "Entrada"
    ) -> None:
        """`assunto` abre o nome acessível de cada segmento, 'Entrada: …', e é o que deixa o
        mesmo seletor servir ao formato do arquivo no Resultado, com o estilo dos segmentos."""
        super().__init__(pai)
        self._assunto = assunto
        self.setObjectName("faixaDeEntradas")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        linha = QHBoxLayout(self)
        linha.setContentsMargins(4, 4, 4, 4)
        linha.setSpacing(4)
        self._grupo = QButtonGroup(self)
        self._grupo.setExclusive(True)
        self.botoes: dict[str, QPushButton] = {}
        for valor in entradas:
            botao = QPushButton(valor, self)
            botao.setObjectName("segmentoDaEntrada")
            botao.setCheckable(True)
            botao.setCursor(Qt.CursorShape.PointingHandCursor)
            botao.clicked.connect(
                lambda _marcado=False, entrada=valor: self.escolhida.emit(entrada)
            )
            self._grupo.addButton(botao)
            self.botoes[valor] = botao
            linha.addWidget(botao)

    def mostra(self, opcoes: Sequence[tuple[str, str]], escolhida: str) -> None:
        """Os segmentos das entradas que existem, com o nome de cada uma, e a escolhida marcada."""
        nomes = dict(opcoes)
        for valor, botao in self.botoes.items():
            nome = nomes.get(valor)
            botao.setVisible(nome is not None)
            if nome is not None and botao.text() != nome:
                botao.setText(nome)
                botao.setAccessibleName(f"{self._assunto}: {nome}")
        self._grupo.setExclusive(False)
        for valor, botao in self.botoes.items():
            botao.setChecked(valor == escolhida)
        self._grupo.setExclusive(True)


class PainelDoResumo(QWidget):
    """O resumo metodológico do que foi produzido, para copiar ou salvar.

    O texto vem pronto de `refino.interface.referencias`, montado pela janela no instante em que o
    resultado passa a existir: o `200` da Análise, que vai à aba 'Qualidade do resultado', ou a
    escrita de um arquivo do Resultado da Correção ou da Normalização, e do lote 'Gerar os três'.
    Este painel só o mostra, copia e salva; quem sabe qual pedido produziu qual resultado é a
    janela.

    Uma caixa de texto e não um rótulo, porque o texto é para copiar inteiro ou em parte: a
    seleção do `QPlainTextEdit` é a de um editor, e o rótulo selecionável perde a seleção ao
    rolar. As referências dos métodos não ficam aqui, e sim na aba 'Métodos e limites'.

    `vazio` é a frase de antes do primeiro resultado, que diz o que fazer naquela aba para o
    resumo existir.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        salvador: SalvadorDeTexto | None = None,
        vazio: str = SEM_RESUMO,
    ) -> None:
        super().__init__(parent)
        # O diálogo de produção nasce no clique, pela razão de `PainelDeArquivos`: um
        # `QFileDialog` real é modal, e um arnês que não o injete pendura.
        self._salvador = salvador
        self._resumo: ResumoMetodologico | None = None
        # A frase do repouso, para `esquece` voltar à desta caixa, e não à de outra.
        self._vazio = vazio
        coluna = QVBoxLayout(self)
        coluna.setContentsMargins(0, 0, 0, 0)
        coluna.setSpacing(10)
        self.caixa = QPlainTextEdit(self)
        self.caixa.setReadOnly(True)
        self.caixa.setPlainText(vazio)
        # A altura mínima vale pela folha do tema, que acha a caixa pelo nome: a regra geral
        # `min-height` dos campos sobrescreve `setMinimumHeight` no polimento, e a caixa ficaria
        # com uma linha na 'Qualidade do resultado', onde a rolagem dá a cada bloco a altura
        # mínima quando a página transborda.
        self.caixa.setObjectName("caixa_do_resumo")
        self.caixa.setMinimumHeight(160)
        coluna.addWidget(self.caixa)
        acoes = QHBoxLayout()
        acoes.setSpacing(8)
        self.copiar = QPushButton("Copiar", self)
        self.salvar = QPushButton("Salvar como .txt", self)
        for botao in (self.copiar, self.salvar):
            botao.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            acoes.addWidget(botao)
        self.desfecho = RotuloQueSome(pai=self)
        self.desfecho.setWordWrap(True)
        acoes.addWidget(self.desfecho, 1)
        # A folga é do fim da linha: com o desfecho e os dois motivos escondidos não sobra item
        # que estique, e a caixa espalharia os dois botões pela largura do bloco.
        acoes.addStretch(1)
        coluna.addLayout(acoes)
        # Sem resumo não há o que levar, e os dois botões ficam desligados com a frase do motivo
        # ao lado, e não em silêncio. Ligam no primeiro resumo (`mostra`).
        for botao in (self.copiar, self.salvar):
            desliga_com_motivo(botao, MOTIVO_SEM_RESUMO)
        self.copiar.clicked.connect(self.copia)
        self.salvar.clicked.connect(self.salva)

    def mostra(self, resumo: ResumoMetodologico) -> None:
        """Põe o resumo na caixa e liga copiar e salvar; o desfecho anterior sai com o texto."""
        self._resumo = resumo
        self.caixa.setPlainText(resumo.texto)
        religa(self.copiar)
        religa(self.salvar)
        self.desfecho.setText("")

    def esquece(self) -> None:
        """Volta ao repouso: sem resumo, e copiar e salvar desligados com o motivo.

        A janela o pede quando a matriz da tela deixa de ter a execução cujo resumo a caixa
        mostrava: o resumo é do arquivo que aquela execução produziu, e sob outra matriz ele seria
        lido como o desta. O repouso é o que o construtor deixou: a frase `vazio` desta caixa, e
        os botões desligados com a frase do motivo ao lado.
        """
        self._resumo = None
        self.caixa.setPlainText(self._vazio)
        for botao in (self.copiar, self.salvar):
            desliga_com_motivo(botao, MOTIVO_SEM_RESUMO)
        self.desfecho.setText("")

    def copia(self) -> None:
        """O texto inteiro para a área de transferência, como a caixa o mostra."""
        if self._resumo is None:
            return
        QGuiApplication.clipboard().setText(self._resumo.texto)
        self.desfecho.setText("Copiado para a área de transferência.")

    def salva(self) -> None:
        """O texto inteiro num `.txt`, em UTF-8, no arquivo que a pessoa escolher.

        `OSError` vira frase e não sobe: num executável `--windowed` não há `stderr`, e a exceção
        dentro do slot seria engolida, com o botão parecendo ter salvo. É a razão de
        `JanelaPrincipal._escreve_no_destino`.
        """
        if self._resumo is None:
            return
        salvador = self._salvador or salvador_por_dialogo(
            self, titulo=TITULO_DO_DIALOGO_DO_RESUMO, filtro=FILTRO_DO_RESUMO
        )
        destino = salvador(NOME_SUGERIDO_DO_RESUMO)
        if destino is None:
            self.desfecho.setText("Nada foi salvo: a escolha do arquivo foi cancelada.")
            return
        if not destino.suffix:
            destino = destino.with_suffix(".txt")
        try:
            destino.write_text(self._resumo.texto, encoding="utf-8")
        except OSError as erro:
            self.desfecho.setText(f"Não foi possível salvar em {destino}: {erro}")
            return
        self.desfecho.setText(f"Salvo em {destino}.")


class ResultadoDaEtapa(PainelDeArtefatos):
    """O Resultado da Correção ou da Normalização: o arquivo da etapa feita, nos dois formatos,
    com o resumo dos métodos logo abaixo.

    É o painel dos artefatos com outra disposição, e não uma cópia dele: os bytes, a identidade
    da entrada, a espera e o relatório do corte são os de `PainelDeArtefatos`, e a janela trata
    este como trata aquele. O que muda é que uma aba serve duas ou três chaves com um botão de
    download só (o seletor de formato escolhe a chave), e por isso todas elas escrevem no mesmo
    rótulo de estado.

    O download só liga com a etapa feita sobre a matriz carregada, e desligado diz o gesto que o
    liga (`mostra_etapa`, `desliga_com_motivo`). 'Gerar os três', na Correção, fica ligado assim
    que há os arquivos: ele corre os três corretores por conta própria.

    A Normalização no formato MetaboAnalyst recusa com explicação e oferece a alternativa. Com
    escalonamento a recusa é sabida antes do clique: faixa amarela, botão principal desligado
    com o motivo. Com `log` ela só se sabe pelo `422` do servidor: faixa vermelha, porque o
    pedido falhou. As duas oferecem o mesmo botão, 'Baixar a alternativa'.

    A falha de um download fica na faixa vermelha até deixar de valer: outro clique, outro
    formato, ou outra etapa feita. A falha de um pedido não descreve o pedido seguinte.
    """

    gerar_os_tres = Signal(str)

    def __init__(
        self,
        etapa: int,
        parent: QWidget | None = None,
        *,
        escolhedor: EscolhedorDePasta | None = None,
        salvador: SalvadorDeTexto | None = None,
    ) -> None:
        chaves = CHAVES_DO_RESULTADO[etapa]
        alternativa = CHAVE_DA_ALTERNATIVA.get(etapa)
        servidas = (*chaves.values(), *(() if alternativa is None else (alternativa,)))
        super().__init__(
            parent,
            escolhedor,
            artefatos=tuple(ARTEFATOS_DA_JANELA[chave].passo for chave in servidas),
            grade=False,
        )
        self.setObjectName("resultadoDaEtapa")
        self.etapa = etapa
        self._formato = FORMATOS[0]
        # O que a janela disse da etapa feita: por que o download não liga, e os campos dela.
        self._motivo: str | None = MOTIVOS_SEM_ETAPA[etapa]
        self._campos_da_feita: Mapping[str, Any] | None = None
        # A falha mostrada (a frase, se a alternativa cabe, e o título; `None` é o do download) e
        # os campos da etapa feita de quando ela chegou: outra etapa feita a apaga.
        self._falha: tuple[str, bool, str | None] | None = None
        self._campos_da_falha: Mapping[str, Any] | None = None
        # O que está desenhado, porque a janela chama `mostra_etapa` a cada meio segundo, e
        # redesenhar a faixa a cada volta a repintaria sem nada ter mudado.
        self._desenhado: tuple[Any, ...] | None = None

        coluna = QVBoxLayout(self)
        coluna.setContentsMargins(0, 12, 0, 0)
        coluna.setSpacing(10)

        linha_do_formato = QHBoxLayout()
        linha_do_formato.setSpacing(8)
        rotulo_do_formato = QLabel("Formato do arquivo:", self)
        self.formatos = SeletorDaEntrada(FORMATOS, self, assunto="Formato")
        self.formatos.mostra(tuple(NOMES_DOS_FORMATOS.items()), self._formato)
        self.formatos.escolhida.connect(self._escolhe_formato)
        linha_do_formato.addWidget(rotulo_do_formato)
        linha_do_formato.addWidget(self.formatos)
        linha_do_formato.addStretch(1)
        coluna.addLayout(linha_do_formato)

        linha_do_download = QHBoxLayout()
        linha_do_download.setSpacing(12)
        self.baixar = QPushButton(TEXTOS_DO_DOWNLOAD[etapa], self)
        self.baixar.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.baixar.setFixedHeight(32)
        self.baixar.clicked.connect(self._baixa)
        # Em repouso some, como os rótulos do painel dos artefatos: o estado aparece quando o
        # arquivo é pedido.
        self.estado = RotuloQueSome(SEM_ARTEFATO, self, vazios=frozenset({SEM_ARTEFATO}))
        self.estado.setWordWrap(True)
        linha_do_download.addWidget(self.baixar)
        linha_do_download.addWidget(self.estado, 1)
        # Com o estado e o motivo escondidos, a folga vai para o fim e o botão fica à esquerda; com
        # o estado à vista, é ele que a leva (fator 1 contra 0).
        linha_do_download.addStretch(0)
        coluna.addLayout(linha_do_download)

        self.faixa = FaixaDeAviso(self)
        self.faixa.acionada.connect(self._baixa_a_alternativa)
        coluna.addWidget(self.faixa)

        self.gerar_os_tres_botao: QPushButton | None = None
        if etapa == ETAPAS[0]:
            linha_do_lote = QHBoxLayout()
            linha_do_lote.setSpacing(12)
            botao = QPushButton(TEXTO_DO_LOTE, self)
            botao.setToolTip(DICA_DO_LOTE)
            botao.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            botao.setFixedHeight(32)
            botao.clicked.connect(self._gera_os_tres)
            explicacao = QLabel(
                "Corre os três corretores e grava um arquivo de cada, no formato escolhido, numa "
                "pasta só. Leva minutos: o SERRF domina o custo.",
                self,
            )
            explicacao.setWordWrap(True)
            explicacao.setProperty("textoAuxiliar", True)
            linha_do_lote.addWidget(botao)
            linha_do_lote.addWidget(explicacao, 1)
            coluna.addLayout(linha_do_lote)
            self.gerar_os_tres_botao = botao

        self.salvar_o_resumo = QCheckBox(TEXTO_DA_CAIXA_DO_RESUMO, self)
        self.salvar_o_resumo.setChecked(True)
        coluna.addWidget(self.salvar_o_resumo)

        self.bloco_do_resumo = Recolhivel(TITULO_DO_BLOCO_DO_RESUMO, self)
        dentro_do_bloco = QVBoxLayout(self.bloco_do_resumo.corpo)
        dentro_do_bloco.setContentsMargins(0, 0, 0, 0)
        self.resumo = PainelDoResumo(
            self.bloco_do_resumo.corpo, salvador=salvador, vazio=SEM_RESUMO_DA_ETAPA[etapa]
        )
        dentro_do_bloco.addWidget(self.resumo)
        coluna.addWidget(self.bloco_do_resumo)

        # Todas as chaves desta aba escrevem no mesmo rótulo, e a de cada formato aponta para o
        # botão de download, que é o nome que a frase da desistência cita. A da alternativa aponta
        # para o botão da faixa, que é onde ela se pede.
        for chave in chaves.values():
            self.botoes_de_geracao[chave] = self.baixar
            self.rotulos[chave] = self.estado
        if alternativa is not None:
            self.botoes_de_geracao[alternativa] = self.faixa.botao
            self.rotulos[alternativa] = self.estado
        self._desenha()

    def chave_do_formato(self) -> str:
        """A chave do download no formato escolhido, a que o botão principal e o lote pedem."""
        return CHAVES_DO_RESULTADO[self.etapa][self._formato]

    def mostra_etapa(self, motivo: str | None, campos: Mapping[str, Any] | None) -> None:
        """O que a janela sabe da etapa feita: por que o download não liga, e os campos dela.

        `motivo` é `None` quando há etapa feita sobre a matriz carregada; senão é a frase que diz
        o gesto que liga o download. `campos` são os que a etapa feita mandou: o escalonamento
        deles decide a recusa sabida, e a transformação e o escalonamento, se a alternativa cabe.
        Outros campos que os da falha mostrada apagam a falha: ela era de outra etapa feita.
        """
        if self._falha is not None and campos is not self._campos_da_falha:
            self._falha = None
        self._motivo = motivo
        self._campos_da_feita = campos
        self._desenha()

    def mostra_escrita(self, chave: str, frase: str) -> None:
        """Onde o arquivo foi escrito, e o resumo ao lado dele, no rótulo do estado."""
        if chave in self.rotulos:
            self.rotulos[chave].setText(frase)

    def recebe(self, chave: str, codigo: int, conteudo: bytes | None, mensagem: str) -> str:
        """O desfecho, como no painel dos artefatos, e a falha vai à faixa vermelha.

        A alternativa só é oferecida onde ela muda o pedido: o `422` do download MetaboAnalyst de
        uma normalização que transformou ou escalonou. Sem nenhum dos dois, o pedido da
        alternativa seria o mesmo que acabou de falhar.
        """
        texto = super().recebe(chave, codigo, conteudo, mensagem)
        if chave not in self.rotulos or codigo == EM_CURSO:
            return texto
        if conteudo:
            self._falha = None
        else:
            alternativa = (
                codigo == 422
                and ARTEFATOS_DA_JANELA[chave].formato == FORMATO_METABOANALYST
                and self.etapa in CHAVE_DA_ALTERNATIVA
                and self._a_alternativa_muda_o_pedido()
            )
            self._falha = (texto, alternativa, None)
            self._campos_da_falha = self._campos_da_feita
        self._desenha()
        return texto

    def mostra_falha_do_lote(self, frase: str) -> None:
        """O 'Gerar os três' que parou no meio, na faixa vermelha.

        O aviso de desfecho some; a faixa fica até o próximo clique de download, como a falha de
        um download avulso. Sem ela, quem voltasse à aba veria a correção em dia, com os arquivos
        de antes no rótulo, e não saberia que a pasta tem dois arquivos e não três.
        """
        self._falha = (frase, False, TEXTO_DO_LOTE[0].lower() + TEXTO_DO_LOTE[1:])
        self._campos_da_falha = self._campos_da_feita
        self._desenha()

    def _a_alternativa_muda_o_pedido(self) -> bool:
        campos = self._campos_da_feita
        if campos is None:
            return False
        return any(campos.get(eixo, "none") != "none" for eixo in ("transformation", "scaling"))

    def _escolhe_formato(self, formato: str) -> None:
        self._formato = formato
        self.formatos.mostra(tuple(NOMES_DOS_FORMATOS.items()), formato)
        self._falha = None
        self._desenha()

    def _baixa(self) -> None:
        self._falha = None
        self._desenha()
        self.pede_e_gera(self.chave_do_formato())

    def _baixa_a_alternativa(self) -> None:
        self._falha = None
        self._desenha()
        self.pede_e_gera(CHAVE_DA_ALTERNATIVA[self.etapa])

    def _gera_os_tres(self) -> None:
        self.gerar_os_tres.emit(self.chave_do_formato())

    def _desenha(self) -> None:
        """O botão de download e a faixa, a partir do motivo, do formato e da falha guardados."""
        campos = self._campos_da_feita
        escalonamento = None if campos is None else campos.get("scaling")
        recusa_sabida = (
            self._motivo is None
            and self._formato == FORMATO_METABOANALYST
            and self.etapa in CHAVE_DA_ALTERNATIVA
            and escalonamento not in (None, "none")
        )
        desenho = (self._motivo, recusa_sabida, self._falha)
        if desenho == self._desenhado:
            return
        self._desenhado = desenho
        if self._motivo is not None:
            desliga_com_motivo(self.baixar, self._motivo)
        elif recusa_sabida:
            desliga_com_motivo(self.baixar, MOTIVO_DO_ESCALONAMENTO)
        else:
            religa(self.baixar)
        if self._falha is not None:
            frase, com_alternativa, titulo = self._falha
            texto = TEXTOS_DO_DOWNLOAD[self.etapa]
            self.faixa.mostra(
                frase,
                nivel="critico",
                titulo=titulo if titulo is not None else texto[0].lower() + texto[1:],
                acao=TEXTO_DA_ALTERNATIVA if com_alternativa else None,
            )
        elif recusa_sabida:
            self.faixa.mostra(FRASE_DA_RECUSA_SABIDA, nivel="aviso", acao=TEXTO_DA_ALTERNATIVA)
        else:
            self.faixa.limpa()


class BlocoDeSugestoes(QWidget):
    """'Seguir sugestões do app', com a sugestão, o '?' do porquê e o desfecho do clique.

    Três peças, e cada uma responde uma pergunta: o botão é o gesto; o desfecho, logo ao lado
    dele, diz o que o último clique fez ou, com o botão desligado, por que ele espera
    (`desliga_com_motivo`); a frase de baixo diz o que o botão preenche antes de alguém clicar. O
    botão tem o peso dos botões secundários de propósito: a ação principal da página é a da
    etapa, mais abaixo.

    Na linha do botão, da esquerda para a direita: o botão, a sugestão (`indicacao`, que some
    vazia), o '?' (`porque`) e o desfecho. O bloco não sabe o que preenche: quem aplica as
    sugestões e escreve o desfecho e a sugestão (`mostra_a_indicacao`) é a janela, porque na
    Correção a sugestão pode esperar o Palpite, que é uma rota. Na Normalização, onde a máquina
    não propõe nada, a sugestão fica vazia e o '?' fica ao lado do botão. Quem abre o quadro do
    '?' é a página (`QuadroDaJustificativa`), que tem a justificativa.
    """

    pedida = Signal()

    def __init__(self, explicacao: str, pai: QWidget | None = None) -> None:
        super().__init__(pai)
        self.setObjectName("blocoDeSugestoes")
        coluna = QVBoxLayout(self)
        coluna.setContentsMargins(0, 0, 0, 0)
        coluna.setSpacing(6)
        linha = QHBoxLayout()
        linha.setSpacing(12)
        self.botao = QPushButton(TEXTO_DO_BOTAO_DAS_SUGESTOES, self)
        self.botao.setFixedHeight(32)
        self.botao.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.botao.setCursor(Qt.CursorShape.PointingHandCursor)
        self.botao.setToolTip(explicacao)
        self.botao.clicked.connect(self.pedida)
        linha.addWidget(self.botao)
        # A sugestão e o '?' ficam juntos, mais próximos entre si que do botão: o '?' explica a
        # sugestão, e a distância diz que os dois são um par.
        par = QHBoxLayout()
        par.setSpacing(6)
        self.indicacao = RotuloQueSome(pai=self)
        self.indicacao.setObjectName("indicacaoDasSugestoes")
        # A última frase escrita: a janela a pede a cada meio segundo, e um rótulo reescrito a
        # cada volta seria repintado sem nada ter mudado, como em `AbaDeEtapa.mostra_entrada`.
        self._indicacao = ""
        par.addWidget(self.indicacao)
        self.porque = QToolButton(self)
        self.porque.setObjectName("botaoDoPorque")
        self.porque.setText(TEXTO_DO_PORQUE)
        # O nome e a dica dizem o que ele abre: um '?' sozinho não diz nada a um leitor de tela.
        self.porque.setToolTip(TITULO_DA_JUSTIFICATIVA)
        self.porque.setAccessibleName(TITULO_DA_JUSTIFICATIVA)
        # No Tab e no clique, e Espaço o aciona como a qualquer botão focado.
        self.porque.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.porque.setCursor(Qt.CursorShape.PointingHandCursor)
        self.porque.setFixedSize(18, 18)
        par.addWidget(self.porque)
        linha.addLayout(par)
        # O desfecho some enquanto vazio: antes do primeiro clique não há o que dizer, e o motivo
        # de o botão esperar entra entre os dois, logo depois do botão.
        self.desfecho = RotuloQueSome(pai=self)
        self.desfecho.setWordWrap(True)
        linha.addWidget(self.desfecho, 1)
        linha.addStretch(0)
        coluna.addLayout(linha)
        self.explicacao = QLabel(explicacao, self)
        self.explicacao.setWordWrap(True)
        self.explicacao.setProperty("textoAuxiliar", True)
        coluna.addWidget(self.explicacao)

    def mostra_a_indicacao(self, frase: str) -> None:
        """A sugestão ao lado do botão, escrita só quando muda. Vazia, ela some."""
        if frase == self._indicacao:
            return
        self._indicacao = frase
        self.indicacao.setText(frase)

    def diz_o_desfecho(self, frase: str) -> None:
        """O que o último clique fez, ao lado do botão; e o botão volta a ligar."""
        religa(self.botao)
        self.desfecho.setText(frase)

    def espera(self, motivo: str) -> None:
        """O botão desligado enquanto a sugestão é medida, com o porquê ao lado dele."""
        self.desfecho.setText("")
        desliga_com_motivo(self.botao, motivo)

    def limpa(self) -> None:
        """O desfecho deixou de descrever os campos: a pessoa mudou um deles depois do clique."""
        self.desfecho.setText("")


class QuadroDaJustificativa(QFrame):
    """'Por que o programa propôs isto', num quadro que o '?' abre ao lado da sugestão.

    Um quadro que abre num clique, e não uma dica de mouse: a justificativa tem parágrafos em
    Markdown e cresce com as leituras, e uma dica some quando o cursor sai do '?', sem dar tempo
    de a ler nem de rolar o que não cabe.

    O conteúdo é a mesma parte que `PainelDeLeitura.reparte_a_justificativa` devolve para a
    página, com os rótulos que `mostra_justificativa` reescreve a cada leitura; ela muda de pai
    para a rolagem deste quadro. Nada é copiado: o que o quadro mostra é o que a leitura
    escreveu por último, aberto ou fechado. O quadro é filho da página para morrer com ela: com
    `Qt.WindowType.Popup` ele é uma janela própria, e sem pai sobreviveria à janela que o abriu.

    - abre logo abaixo do '?', e é deslocado só o que falta para caber na tela;
    - tem no máximo `LARGURA_DA_JUSTIFICATIVA` de largura, para o texto quebrar linha;
    - a altura é a do texto, limitada pela tela, e o que não cabe rola dentro dele;
    - fecha com um clique fora, que `Qt.WindowType.Popup` já faz, ou com Esc, que o `QWidget` já
      faz para um popup.

    O clique no próprio '?', com o quadro aberto, fecha-o e não o reabre. O sistema entrega a
    pressão ao quadro aberto, que fecha, e o Qt pode reenviá-la depois ao widget por baixo, o
    '?', que a leria como um clique novo. `WA_NoMouseReplay` pede ao Qt que não a reenvie, mas no
    Windows, com o Qt 6.11, ele a reenvia às vezes mesmo assim, com o mesmo instante da pressão
    original. Por isso o quadro também recusa, no '?', a pressão cujo instante é o da que o
    fechou: um clique novo tem instante próprio, e passa.
    """

    def __init__(self, parte: QWidget, ancora: QToolButton, pai: QWidget) -> None:
        super().__init__(pai, Qt.WindowType.Popup)
        self.setObjectName("quadroDaJustificativa")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoMouseReplay, True)
        self.setAccessibleName(TITULO_DA_JUSTIFICATIVA)
        self._ancora = ancora
        # O instante da pressão fora do quadro que o fechou, para recusar o reenvio dela ao '?'.
        self._instante_do_fecho: int | None = None
        coluna = QVBoxLayout(self)
        coluna.setContentsMargins(16, 14, 16, 14)
        coluna.setSpacing(10)
        self.titulo = QLabel(TITULO_DA_JUSTIFICATIVA, self)
        self.titulo.setObjectName("tituloDaJustificativa")
        coluna.addWidget(self.titulo)
        self.rolagem = QScrollArea(self)
        self.rolagem.setObjectName("rolagemDaJustificativa")
        self.rolagem.setWidgetResizable(True)
        self.rolagem.setFrameShape(QFrame.Shape.NoFrame)
        self.rolagem.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.parte = parte
        self.rolagem.setWidget(parte)
        coluna.addWidget(self.rolagem, 1)
        ancora.clicked.connect(self.abre)
        ancora.installEventFilter(self)

    def abre(self) -> None:
        """Abre logo abaixo do '?', da altura do texto e inteiro dentro da tela."""
        self._instante_do_fecho = None
        # A tela pelo ponto do '?', e não por `widget.screen()`: no PySide6 o objeto da tela pedido
        # a um widget vira filho dele e é invalidado quando ele morre (ver
        # `tests/test_desktop_tela.py`).
        canto_do_porque = self._ancora.mapToGlobal(QPoint(0, 0))
        tela_do_porque = QGuiApplication.screenAt(canto_do_porque)
        if tela_do_porque is None:
            tela_do_porque = QGuiApplication.primaryScreen()
        tela = tela_do_porque.availableGeometry()
        largura = min(LARGURA_DA_JUSTIFICATIVA, tela.width() - 2 * _FOLGA_DO_QUADRO)
        altura = min(self._altura_para(largura), tela.height() - 2 * _FOLGA_DO_QUADRO)
        self.resize(largura, altura)
        canto = self._ancora.mapToGlobal(QPoint(0, self._ancora.height() + _ABAIXO_DO_PORQUE))
        # Deslocado o que falta, e só isso: à esquerda se passaria da borda direita, e para cima se
        # passaria da de baixo.
        x = min(canto.x(), tela.right() + 1 - _FOLGA_DO_QUADRO - largura)
        y = min(canto.y(), tela.bottom() + 1 - _FOLGA_DO_QUADRO - altura)
        self.move(max(x, tela.left() + _FOLGA_DO_QUADRO), max(y, tela.top() + _FOLGA_DO_QUADRO))
        self.rolagem.verticalScrollBar().setValue(0)
        self.show()

    def _altura_para(self, largura: int) -> int:
        """A altura que mostra o título e a justificativa inteira, com o texto quebrado em
        `largura`.

        A barra de rolagem entra na conta mesmo invisível: a folha a deixa sempre reservada
        (`tema.veste_a_janela`), e sem ela a conta quebraria o texto mais largo do que ele é
        desenhado, e a última linha ficaria fora."""
        disposicao = self.layout()
        margens = disposicao.contentsMargins() if disposicao is not None else None
        horizontais = 0 if margens is None else margens.left() + margens.right()
        verticais = 0 if margens is None else margens.top() + margens.bottom()
        espaco = disposicao.spacing() if disposicao is not None else 0
        moldura = 2 * self.frameWidth()
        util = largura - horizontais - moldura - self.rolagem.verticalScrollBar().sizeHint().width()
        corpo = self.parte.heightForWidth(util)
        if corpo < 0:
            corpo = self.parte.sizeHint().height()
        return verticais + moldura + self.titulo.sizeHint().height() + espaco + corpo

    def mousePressEvent(self, evento: QMouseEvent) -> None:  # noqa: N802 - nome do Qt
        """A pressão fora do quadro o fecha (`QWidget` faz isso para um popup); o instante dela fica
        guardado, para o reenvio dela ao '?' não o reabrir."""
        if not self.rect().contains(evento.position().toPoint()):
            self._instante_do_fecho = evento.timestamp()
        super().mousePressEvent(evento)

    def eventFilter(self, observado: QObject, evento: QEvent) -> bool:  # noqa: N802 - nome do Qt
        """Recusa, no '?', a pressão que fechou o quadro: a reenviada, com o instante dela."""
        if (
            observado is self._ancora
            and evento.type() == QEvent.Type.MouseButtonPress
            and isinstance(evento, QMouseEvent)
            and self._instante_do_fecho is not None
            and evento.timestamp() == self._instante_do_fecho
        ):
            self._instante_do_fecho = None
            return True
        return super().eventFilter(observado, evento)


class AbaDeEtapa(QWidget):
    """Uma etapa do preparo, em três blocos numerados que rolam juntos: os ajustes, a execução da
    etapa com o resultado, e os gráficos.

    A composição acontece antes de inserir a página no QTabWidget. Nenhum controle é duplicado:
    o pipeline continua a ler e atualizar os mesmos objetos, inclusive os botões de cada etapa, o
    rótulo do estado e a barra; a página só os recebe e decide onde aparecem.

    Os três blocos, de cima para baixo:

    - 1 'Ajustes': 'Seguir sugestões do app' (Correção e Normalização), com a sugestão e o '?'
      do porquê ao lado; a matriz que entra e os avisos dela; e os grupos de parâmetros da etapa.
    - 2 'Execução da etapa': o aviso da etapa; a linha da ação, com o botão, o estado em palavras
      e a barra; as notas; e o 'Resultado' ou o 'Próximo passo'.
    - 3 'Gráficos desta etapa': a vista das figuras da etapa, que traz o próprio bloco numerado.

    Os blocos são os de 'Preparar o estudo' (`AbaDePreparo`): a mesma `Secao` numerada com a
    moldura de referência, e o mesmo respiro de 24 px entre eles na folha. Os grupos de
    parâmetros continuam planos dentro do bloco 1; a regra
    `QFrame#secao QGroupBox[painelPlano="true"]` do tema vale para eles dentro de qualquer seção.

    A ordem é a da causa ao efeito: o que se escolhe, no bloco 1; o que impede o resultado de
    valer, o gesto que executa, o estado dele e o que ele produziu, no bloco 2. O aviso fica logo
    acima do botão porque é o botão, ou o do próprio aviso, que o resolve; o resultado fica logo
    abaixo porque é o desfecho do gesto, e por isso é um subtítulo dentro do bloco 2 e não um
    bloco à parte.

    Não há rodapé fixo: a ação fica dentro da rolagem, entre os parâmetros que ela executa e o
    resultado que produz. A 1280x720 o botão da Correção fica abaixo da dobra, e chega-se a ele
    rolando pelos parâmetros.

    Uma pilha troca a página inteira pelo estado vazio enquanto falta um dos dois arquivos
    (`mostra_arquivos`): a ação vai junto, porque o botão da etapa sem arquivos só poderia nascer
    desligado. Quem decide quando trocar é a janela, nos sinais do painel de arquivos.

    O Resultado vem à vista quando a etapa passa a 'em dia' (`mostra_estado`), com a rolagem
    mínima que o mostra sem tirar a linha da ação da tela (`traz_o_resultado_a_vista`). Na
    Análise o lugar dele é o 'Próximo passo', com os dois botões que levam às abas dos
    resultados; quem troca de aba é a janela, pelos sinais desta página.

    A Entrada, no começo do bloco 1, diz que matriz a etapa usa (`mostra_entrada`): uma frase com
    uma só, o seletor com duas ou três, e os dois avisos que são dela, a escolha que deixou de
    existir e, na Análise, os zeros sem imputação. A escolha da entrada, o botão do aviso e o das
    sugestões são sinais; quem decide o que fazer é a janela.
    """

    ir_ao_preparo = Signal()
    ver_a_qualidade = Signal()
    abrir_os_graficos = Signal()
    entrada_escolhida = Signal(str)
    seguir_sugestoes = Signal()

    def __init__(
        self,
        grupos: tuple[QWidget, ...],
        parent: QWidget | None = None,
        *,
        etapa: int,
        acao: QPushButton,
        estado: QLabel,
        barra: QProgressBar,
        resultado: QWidget | None = None,
        notas: tuple[QWidget, ...] = (),
        justificativa: QWidget | None = None,
    ) -> None:
        """`resultado` é o que a etapa entrega no bloco 'Resultado': o download e o resumo dos
        métodos, na Correção e na Normalização. A Análise não tem: o resultado dela mora noutras
        abas, e o bloco dela é o 'Próximo passo'.

        `notas` são as linhas logo abaixo da linha da ação: a nota da etapa e, na Análise, o
        corretor e o resultado da execução. Somem enquanto vazias. `justificativa` é a parte de
        'Por que o programa propôs isto' que explica os controles desta página; ela abre pelo '?'
        do bloco das sugestões, num `QuadroDaJustificativa`.

        A parte e o bloco das sugestões vêm juntos, ou nenhum dos dois, e o desencontro levanta
        `ValueError`. Sem o bloco não há '?', e a justificativa desta página sumiria sem nada
        acusar, que é o que `reparte_a_justificativa` também recusa. Com o bloco e sem a parte, o
        '?' abriria um quadro vazio."""
        super().__init__(parent)
        if (justificativa is None) != (etapa not in EXPLICACOES_DAS_SUGESTOES):
            tem = "tem" if justificativa is None else "não tem"
            raise ValueError(
                f"a página da etapa {etapa} {tem} o bloco das sugestões, e a justificativa abre "
                "pelo «?» dele: as duas vêm juntas"
            )
        self.setObjectName("paginaDeEtapa")
        self.etapa = etapa
        self.resultado = resultado
        # O estado que a página mostra; nasce no de uma etapa que ninguém rodou.
        self._estado = "nao_feita"
        self._rolar_ao_mostrar = False
        # O que a Entrada e o aviso mostram agora: a janela chama isto a cada meio segundo, e um
        # aviso redesenhado a cada volta repintaria a faixa sem nada ter mudado.
        self._vista_da_entrada: VistaDaEntrada | None = None
        self._aviso: tuple[str, str, str | None, str | None] | None = None
        self._mais_tratada: str | None = None
        externo = QVBoxLayout(self)
        externo.setContentsMargins(0, 0, 0, 0)
        externo.setSpacing(0)
        # A pilha nasce dentro da página, antes de ela entrar nas abas: o que esteve na pilha das
        # abas não pode mudar de pai depois de composto (ver `tema.pagina_rolavel`).
        self.pilha = QStackedWidget(self)
        externo.addWidget(self.pilha)
        self.vazio = self._monta_o_vazio()
        self.pilha.addWidget(self.vazio)

        # O formulário é a rolagem: sem rodapé fixo, não sobra nada fora dela.
        self.rolagem = pagina = QScrollArea(self.pilha)
        self.formulario = pagina
        pagina.setWidgetResizable(True)
        pagina.setFrameShape(QFrame.Shape.NoFrame)
        pagina.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.pilha.addWidget(pagina)
        self._conteudo = conteudo = QWidget(pagina)
        pagina.setWidget(conteudo)
        coluna = QVBoxLayout(conteudo)
        coluna.setContentsMargins(0, 0, 0, 0)
        coluna.setSpacing(0)
        self.folha = QFrame(conteudo)
        self.folha.setProperty("folhaDeEtapa", True)
        self.folha.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.folha.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        # A altura mínima da folha acompanha a que o conteúdo pede na largura atual. Com a política
        # vertical `Maximum` a folha fica presa ao `sizeHint`, e o `sizeHint` não sabe a largura:
        # texto numa coluna estreita, como o motivo do 'Baixar' desligado, pede mais linhas do que
        # ele conta, e o layout tiraria a diferença de quem tem pouco mínimo, cortando blocos e
        # notas. `SegueAAltura` fecha o mesmo defeito em 'Preparar o estudo'. Filho da folha:
        # morre com ela.
        self._altura_da_folha = SegueAAltura(self.folha)
        paineis = QVBoxLayout(self.folha)
        paineis.setContentsMargins(20, 20, 20, 20)
        # O respiro entre os blocos é o de 'Preparar o estudo': 24 px entre os cartões numerados,
        # como em `AbaDePreparo`, e nenhum espaço avulso entre um e outro.
        paineis.setSpacing(24)

        self._monta_a_entrada(etapa, justificativa)
        self._poe_os_parametros(grupos)
        paineis.addWidget(self.secao_da_entrada)
        self._monta_a_secao_da_etapa(acao, estado, barra, notas)
        paineis.addWidget(self.secao_da_etapa)
        # O lugar dos gráficos da etapa, o bloco 3. Vazio, não ocupa altura; quem o preenche é a
        # vista das figuras da etapa, que vive com o painel dos gráficos e traz o bloco dela, '3',
        # 'Gráficos desta etapa'; é a janela que a põe aqui (`poe_os_graficos`).
        self.lugar_dos_graficos = QWidget(self.folha)
        disposicao_dos_graficos = QVBoxLayout(self.lugar_dos_graficos)
        disposicao_dos_graficos.setContentsMargins(0, 0, 0, 0)
        paineis.addWidget(self.lugar_dos_graficos)
        coluna.addWidget(self.folha)
        coluna.addStretch(1)

        # Filho da página, como os relógios da janela: morre com ela, e nunca dispara sobre uma
        # página destruída.
        self._relogio_da_rolagem = QTimer(self)
        self._relogio_da_rolagem.setSingleShot(True)
        self._relogio_da_rolagem.setInterval(0)
        self._relogio_da_rolagem.timeout.connect(self.traz_o_resultado_a_vista)

    def _monta_a_entrada(self, etapa: int, justificativa: QWidget | None) -> None:
        """O começo do bloco 1, 'Ajustes': as sugestões no topo, com o '?' da justificativa, e a
        matriz que entra com os avisos dela. Os parâmetros vêm depois, em `_poe_os_parametros`."""
        self.secao_da_entrada = Secao(
            TITULO_DO_BLOCO_DOS_AJUSTES, self.folha, numero=1, referencia=True
        )
        corpo_da_entrada = self.secao_da_entrada.corpo
        # As sugestões ficam no topo, antes da matriz, porque são o primeiro gesto de quem não quer
        # escolher campo a campo. A Análise não as tem (`EXPLICACOES_DAS_SUGESTOES`).
        self.sugestoes: BlocoDeSugestoes | None = None
        # A justificativa das propostas abre pelo '?' das sugestões, num quadro filho da página para
        # morrer com ela; o construtor já garantiu que a parte e o bloco vêm juntos.
        self.justificativa: QuadroDaJustificativa | None = None
        if etapa in EXPLICACOES_DAS_SUGESTOES:
            self.sugestoes = BlocoDeSugestoes(
                EXPLICACOES_DAS_SUGESTOES[etapa], self.secao_da_entrada
            )
            self.sugestoes.pedida.connect(self.seguir_sugestoes)
            corpo_da_entrada.addWidget(self.sugestoes)
            corpo_da_entrada.addWidget(Divisor(self.secao_da_entrada))
            if justificativa is not None:
                self.justificativa = QuadroDaJustificativa(
                    justificativa, self.sugestoes.porque, self
                )
        self.frase_da_entrada = QLabel(self.secao_da_entrada)
        self.frase_da_entrada.setWordWrap(True)
        corpo_da_entrada.addWidget(self.frase_da_entrada)
        self.seletor_da_entrada = SeletorDaEntrada(ENTRADAS_DA_ETAPA[etapa], self.secao_da_entrada)
        self.seletor_da_entrada.hide()
        self.seletor_da_entrada.escolhida.connect(self.entrada_escolhida)
        corpo_da_entrada.addWidget(self.seletor_da_entrada, 0, Qt.AlignmentFlag.AlignLeft)
        # A entrada mais tratada que surgiu sem trocar a escolha: a escolha da pessoa persiste
        # enquanto a entrada dela existir, e a página só avisa que há outra, com o botão que a usa.
        linha_da_mais_tratada = QHBoxLayout()
        linha_da_mais_tratada.setSpacing(8)
        self.frase_da_mais_tratada = QLabel(self.secao_da_entrada)
        self.frase_da_mais_tratada.setWordWrap(True)
        self.usar_a_mais_tratada = QPushButton(
            TEXTO_DO_BOTAO_DA_MAIS_TRATADA, self.secao_da_entrada
        )
        self.usar_a_mais_tratada.clicked.connect(self._usa_a_mais_tratada)
        self.frase_da_mais_tratada.hide()
        self.usar_a_mais_tratada.hide()
        linha_da_mais_tratada.addWidget(self.frase_da_mais_tratada, 1)
        linha_da_mais_tratada.addWidget(self.usar_a_mais_tratada)
        corpo_da_entrada.addLayout(linha_da_mais_tratada)
        self.faixa_da_volta = FaixaDeAviso(self.secao_da_entrada)
        corpo_da_entrada.addWidget(self.faixa_da_volta)
        self.faixa_da_imputacao = FaixaDeAviso(self.secao_da_entrada)
        corpo_da_entrada.addWidget(self.faixa_da_imputacao)
        # A página nasce com a Entrada de quem não fez etapa nenhuma, nas frases do estado.
        nada_feito = EstadoDasEtapas()
        ausencias = tuple(
            frase
            for entrada in ENTRADAS_DA_ETAPA[etapa][1:]
            if (frase := nada_feito.ausencia(entrada, "")) is not None
        )
        self.mostra_entrada(
            VistaDaEntrada(
                FRASE_DA_ENTRADA_DA_CORRECAO
                if etapa == ETAPAS[0]
                else frase_da_entrada_unica(etapa, ausencias)
            )
        )

    def _poe_os_parametros(self, grupos: tuple[QWidget, ...]) -> None:
        """O resto do bloco 1: os grupos de parâmetros da etapa, junto da matriz que entra, que é o
        outro ajuste da etapa. O bloco 2 fica só com o que executa e o que resulta.

        Numa coluna própria, com 20 px entre um grupo e o seguinte. No corpo do bloco o
        espaçamento é o da `Secao`, e somar a ele as folgas avulsas dobraria a distância entre os
        grupos. Uma divisória separa a matriz que entra dos parâmetros, como a que separa as
        sugestões da matriz.
        """
        corpo = self.secao_da_entrada.corpo
        corpo.addWidget(Divisor(self.secao_da_entrada))
        parametros = QVBoxLayout()
        parametros.setContentsMargins(0, 8, 0, 0)
        parametros.setSpacing(0)
        corpo.addLayout(parametros)
        # Cada grupo com a divisória dele; nenhum é o primeiro nem o último painel da folha.
        for indice, grupo in enumerate(grupos):
            grupo.setProperty("primeiroPainel", False)
            grupo.setProperty("ultimoPainel", False)
            if indice:
                parametros.addSpacing(20)
            parametros.addWidget(grupo)

    def _monta_a_secao_da_etapa(
        self,
        acao: QPushButton,
        estado: QLabel,
        barra: QProgressBar,
        notas: tuple[QWidget, ...],
    ) -> None:
        """O bloco 2, 'Execução da etapa': o aviso, a ação e o resultado dela."""
        self.secao_da_etapa = Secao(
            TITULO_DO_BLOCO_DA_EXECUCAO, self.folha, numero=2, referencia=True
        )
        corpo = self.secao_da_etapa.corpo
        corpo.setSpacing(0)

        # O aviso da etapa, logo acima do botão dela. Escondido, não ocupa altura; à vista, diz por
        # que o resultado não vale ou por que a etapa falhou, e o botão dele é a saída. Mora aqui,
        # e não no topo da página, porque o que o resolve é o botão logo abaixo.
        self.faixa_da_etapa = FaixaDeAviso(self.secao_da_etapa)
        corpo.addWidget(self.faixa_da_etapa)
        corpo.addSpacing(8)
        # A linha da ação: o botão, o estado em palavras e a barra, lado a lado. O painel faz nascer
        # os três escondidos, para não os desenhar soltos se alguém o montar sem a página (ver
        # `pipeline.py`). O motivo de o botão estar desligado, quando há, entra logo depois dele
        # (`desliga_com_motivo`).
        self.linha_da_acao = QWidget(self.secao_da_etapa)
        self.linha_da_acao.setObjectName("linhaDaAcao")
        linha = QHBoxLayout(self.linha_da_acao)
        linha.setContentsMargins(0, 0, 0, 0)
        linha.setSpacing(12)
        linha.addWidget(acao)
        linha.addWidget(estado)
        linha.addWidget(barra, 1)
        for parte in (acao, estado, barra):
            parte.show()
        corpo.addWidget(self.linha_da_acao)
        # As notas, por baixo da linha da ação: o progresso em texto e, na Análise, o corretor e o
        # resultado da execução, o que a barra e o estado em palavras não dizem. Somem enquanto
        # vazias, e não recebem `show()` aqui: quem decide se aparecem é o texto delas
        # (`RotuloQueSome`).
        self.notas = notas
        for nota in notas:
            corpo.addSpacing(6)
            corpo.addWidget(nota)

        # O resultado fecha o bloco 2, como desfecho da ação logo acima: um subtítulo dentro do
        # bloco, e não um bloco à parte. Na Análise é o 'Próximo passo', com os dois botões.
        ultima = self.etapa == ETAPAS[-1]
        self.bloco_do_resultado = QWidget(self.secao_da_etapa)
        self.bloco_do_resultado.setObjectName("blocoDoResultado")
        coluna_do_resultado = QVBoxLayout(self.bloco_do_resultado)
        coluna_do_resultado.setContentsMargins(0, 24, 0, 0)
        coluna_do_resultado.setSpacing(8)
        self.titulo_do_resultado = QLabel(
            TITULO_DO_PROXIMO_PASSO if ultima else TITULO_DO_RESULTADO, self.bloco_do_resultado
        )
        self.titulo_do_resultado.setObjectName("subtitulo_da_etapa")
        coluna_do_resultado.addWidget(self.titulo_do_resultado)
        self.frase_do_resultado = QLabel(
            FRASES_DO_RESULTADO[self.etapa][self._estado], self.bloco_do_resultado
        )
        self.frase_do_resultado.setWordWrap(True)
        coluna_do_resultado.addWidget(self.frase_do_resultado)
        if self.resultado is not None:
            coluna_do_resultado.addWidget(self.resultado)
        self.botao_da_qualidade: QPushButton | None = None
        self.botao_dos_graficos: QPushButton | None = None
        if ultima:
            linha_dos_passos = QHBoxLayout()
            linha_dos_passos.setSpacing(8)
            self.botao_da_qualidade = QPushButton(
                TEXTO_DO_BOTAO_DA_QUALIDADE, self.bloco_do_resultado
            )
            self.botao_dos_graficos = QPushButton(
                TEXTO_DO_BOTAO_DOS_GRAFICOS, self.bloco_do_resultado
            )
            self.botao_da_qualidade.clicked.connect(self.ver_a_qualidade)
            self.botao_dos_graficos.clicked.connect(self.abrir_os_graficos)
            for botao in (self.botao_da_qualidade, self.botao_dos_graficos):
                botao.setFixedHeight(32)
                linha_dos_passos.addWidget(botao)
            linha_dos_passos.addStretch(1)
            coluna_do_resultado.addLayout(linha_dos_passos)
            # Antes da análise não há resultado a que levar: os dois só aparecem depois do `200`.
            self.botao_da_qualidade.hide()
            self.botao_dos_graficos.hide()
        corpo.addWidget(self.bloco_do_resultado)

    def poe_os_graficos(self, vista: QWidget) -> None:
        """A vista das figuras da etapa, no bloco 3, o último da folha.

        A vista traz o próprio bloco, com o número 3 e o título 'Gráficos desta etapa': a página
        não lhe põe um segundo cabeçalho por cima, e não lhe mexe em nada. Os blocos numerados são
        cartões com moldura e respiro próprios, como em 'Preparar o estudo'.
        """
        lugar = self.lugar_dos_graficos.layout()
        if lugar is not None:
            lugar.addWidget(vista)
        vista.show()

    def _monta_o_vazio(self) -> QFrame:
        """O bloco que ocupa a página sem os arquivos: a frase do que falta e o botão ao preparo.

        O estilo é o do estado vazio dos gráficos (`QFrame#estado_vazio`): a frase centrada, com
        largura de leitura, num bloco que ocupa a página, e não solta no meio dela.
        """
        vazio = QFrame(self.pilha)
        vazio.setObjectName("estado_vazio")
        vazio.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dentro = QVBoxLayout(vazio)
        dentro.setSpacing(16)
        dentro.addStretch(1)
        self.frase_do_vazio = QLabel(vazio)
        self.frase_do_vazio.setObjectName("texto_do_vazio")
        self.frase_do_vazio.setWordWrap(True)
        self.frase_do_vazio.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        # Centrada por esticadores, como a dos gráficos: com alinhamento no `addWidget` o layout
        # dá ao rótulo a altura de uma linha e a frase sai cortada.
        linha_da_frase = QHBoxLayout()
        linha_da_frase.addStretch(1)
        linha_da_frase.addWidget(self.frase_do_vazio, 10)
        linha_da_frase.addStretch(1)
        dentro.addLayout(linha_da_frase)
        self.botao_do_preparo = QPushButton(TEXTO_DO_BOTAO_DO_PREPARO, vazio)
        self.botao_do_preparo.setFixedHeight(32)
        self.botao_do_preparo.clicked.connect(self.ir_ao_preparo)
        dentro.addWidget(self.botao_do_preparo, 0, Qt.AlignmentFlag.AlignHCenter)
        dentro.addStretch(1)
        self.frase_do_vazio.setText(frase_do_estado_vazio(self.etapa, ("medidas", "lista")))
        return vazio

    # ------------------------------------------------------------------
    # O que a janela diz à página
    # ------------------------------------------------------------------

    def mostra_arquivos(self, faltam: tuple[str, ...]) -> None:
        """O formulário quando não falta arquivo nenhum; o estado vazio, com o que falta, senão.

        `faltam` é `PainelDeArquivos.metades_que_faltam`, vazio exatamente quando há par.
        """
        if not faltam:
            self.pilha.setCurrentWidget(self.formulario)
            return
        self.frase_do_vazio.setText(frase_do_estado_vazio(self.etapa, faltam))
        self.pilha.setCurrentWidget(self.vazio)

    def mostra_entrada(self, vista: VistaDaEntrada) -> None:
        """A Entrada, no bloco 1: a frase, o seletor com duas ou três entradas, e os avisos dela."""
        if vista == self._vista_da_entrada:
            return
        self._vista_da_entrada = vista
        if self.frase_da_entrada.text() != vista.frase:
            self.frase_da_entrada.setText(vista.frase)
        com_escolha = len(vista.opcoes) > 1
        if com_escolha:
            self.seletor_da_entrada.mostra(vista.opcoes, vista.escolhida)
        self.seletor_da_entrada.setVisible(com_escolha)
        self._mais_tratada = None if vista.mais_tratada is None else vista.mais_tratada[0]
        if vista.mais_tratada is not None:
            self.frase_da_mais_tratada.setText(
                f"Existe uma matriz mais tratada: «{vista.mais_tratada[1]}»."
            )
        for parte in (self.frase_da_mais_tratada, self.usar_a_mais_tratada):
            parte.setVisible(vista.mais_tratada is not None)
        for faixa, frase in (
            (self.faixa_da_volta, vista.volta),
            (self.faixa_da_imputacao, vista.imputacao),
        ):
            if frase is None:
                faixa.limpa()
            else:
                faixa.mostra(frase, nivel="aviso")

    def _usa_a_mais_tratada(self) -> None:
        if self._mais_tratada is not None:
            self.entrada_escolhida.emit(self._mais_tratada)

    def mostra_aviso(
        self,
        frase: str,
        *,
        nivel: NivelDeAviso,
        titulo: str | None = None,
        acao: str | None = None,
    ) -> None:
        """O aviso da etapa, logo acima do botão dela; o botão do aviso emite
        `faixa_da_etapa.acionada`."""
        chave = (frase, nivel, titulo, acao)
        if chave == self._aviso:
            return
        self._aviso = chave
        self.faixa_da_etapa.mostra(frase, nivel=nivel, titulo=titulo, acao=acao)

    def limpa_aviso(self) -> None:
        """Nenhum aviso da etapa vale agora."""
        if self._aviso is None:
            return
        self._aviso = None
        self.faixa_da_etapa.limpa()

    def mostra_estado(self, estado: str) -> None:
        """O estado da etapa: a frase do Resultado, os botões do Próximo passo e a rolagem.

        A página traz o Resultado à vista quando a etapa passa a 'em dia' depois de uma corrida,
        vinda de 'não feita' ou de 'em curso'. Da volta de 'desatualizada' ela não rola: é a
        pessoa devolvendo um parâmetro ao que era, com o formulário na mão.
        """
        frase = FRASES_DO_RESULTADO[self.etapa][estado]
        anterior = self._estado
        self._estado = estado
        if self.frase_do_resultado.text() != frase:
            self.frase_do_resultado.setText(frase)
        com_resultado = estado in _COM_RESULTADO
        for botao in (self.botao_da_qualidade, self.botao_dos_graficos):
            if botao is not None:
                botao.setVisible(com_resultado)
        if estado == "em_dia" and anterior in _DE_ONDE_ROLA:
            self._rola_quando_puder()
        elif estado != "em_dia":
            # A rolagem guardada para quando a página aparecer era a de um resultado em dia.
            self._rolar_ao_mostrar = False

    # ------------------------------------------------------------------
    # O Resultado à vista
    # ------------------------------------------------------------------

    def traz_o_resultado_a_vista(self) -> None:
        """A rolagem mínima que mostra a linha da ação e o Resultado logo abaixo dela.

        A faixa que tem de caber vai do topo da linha da ação ao fim do Resultado, e não é só o
        Resultado: o estado em palavras e a barra dizem que a etapa acabou, e um salto que os
        tirasse da tela trocaria o contexto do gesto pelo desfecho dele. Quando a faixa não cabe na
        área visível, a linha da ação fica no topo e o Resultado começa logo abaixo dela.

        Mínima: com a faixa já à vista nada se move, e com ela abaixo da dobra a página desce o que
        falta. É o que `QScrollArea.ensureWidgetVisible` faz para um widget só, aqui para dois.
        """
        barra = self.rolagem.verticalScrollBar()
        altura = self.rolagem.viewport().height()
        topo = self.linha_da_acao.mapTo(self._conteudo, QPoint(0, 0)).y()
        resultado = self.bloco_do_resultado
        fim = resultado.mapTo(self._conteudo, QPoint(0, 0)).y() + resultado.height()
        fim = min(fim, topo + altura)
        atual = barra.value()
        if topo < atual:
            alvo = topo
        elif fim > atual + altura:
            alvo = fim - altura
        else:
            return
        barra.setValue(max(barra.minimum(), min(alvo, barra.maximum())))

    def _rola_quando_puder(self) -> None:
        """Rola depois do próximo passo de layout: a frase do Resultado acabou de mudar, e a
        geometria ainda é a de antes. Com a página escondida, rola quando ela aparecer: a geometria
        de uma página que nunca foi mostrada não diz onde o Resultado fica."""
        if self.isVisible():
            self._relogio_da_rolagem.start()
        else:
            self._rolar_ao_mostrar = True

    def showEvent(self, evento: QShowEvent) -> None:  # noqa: N802 - nome do Qt
        """A rolagem pendente de quando a página estava escondida."""
        super().showEvent(evento)
        if self._rolar_ao_mostrar:
            self._rolar_ao_mostrar = False
            self._relogio_da_rolagem.start()

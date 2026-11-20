"""O enriquecimento pelo MetaboAnalystR, o segundo modo da aba 'Vias metabólicas (beta)'.

O primeiro modo prediz vias a partir dos picos, sem identificação. Este parte da identificação
que o laboratório já fez no Progenesis: lê a tabela exportada, cruza com o teste univariado da
execução, monta a lista significativa e o fundo do ensaio, e roda a ORA do MetaboAnalystR num R
portátil que viaja com o programa. Cada passo tem estado próprio na tela:

- carregar a tabela e confirmar as colunas; a heurística só pré-preenche, e 'Enriquecer' fica
  desligado até a confirmação;
- escolher o corte (p, q ou as N melhores); cada mudança refaz as contagens sem chamar o R;
- enriquecer: `mapear`, as recusas sobre o mapeamento e a ORA, num trabalhador fora da thread da
  tela; a espera mostra a fase e o tempo, e 'Cancelar' encerra a árvore de processos do R;
- ler e exportar: a tabela por p crescente, os compostos de cada via na dica da linha, e o CSV
  com o recorte inteiro.

Toda mudança que invalida o que está em voo (execução nova, tabela, colunas, corte, cancelamento)
avança um token de geração, e um resultado que chega com token velho não pinta.

No modo externo, a ORA KEGG roda no servidor do MetaboAnalyst, porque o pacote só a tem lá. Em
segundo plano, a lista é mapeada no R local, a G1 confere que nenhum nome repete um rótulo da
execução ou da tabela, e a carga é montada por esquema fechado e conferida byte a byte (G2); na
thread da tela, o diálogo mostra os nomes que vão sair, e nada sai sem 'Enviar'; em segundo plano,
o envio, e a resposta passa pela mesma conferência do modo local (`ora_da_resposta`). O servidor
ignora o fundo do ensaio na ORA KEGG, então o universo é a biblioteca KEGG inteira, e a tela e o
CSV dizem que isso tende a falsos positivos.
"""

from __future__ import annotations

import csv
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from PySide6.QtCore import QObject, QRunnable, QSignalBlocker, Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.janela.aparencia import FiltroDaRoda, sem_roda_sem_foco
from refino.desktop.janela.componentes import desliga_com_motivo, religa
from refino.desktop.vias_r import MotorDoR, resolve_motor
from refino.interface.decorrido import agora, instante_utc
from refino.io import dados_vias
from refino.io import enriquecimento_externo as ext
from refino.io.enriquecimento_externo import DesfechoDoEnvio, RespostaDoServidor
from refino.io.enriquecimento_r import (
    TEMPO_LIMITE_PADRAO_S,
    RespostaDoR,
    corre_roteiro_de_vias,
)
from refino.io.identificacao import (
    IdentificacaoMalformadaError,
    TabelaDeIdentificacao,
    le_cabecalho,
    le_identificacao,
    sugere_colunas,
)
from refino.io.validators import CANONICAL_GROUPS
from refino.pathways import enriquecimento as en
from refino.plots.recorte import decimal

__all__ = [
    "COLUNAS",
    "MENSAGENS",
    "EnvioExterno",
    "EnvioPreparado",
    "PainelDeEnriquecimento",
    "conclui_envio_externo",
    "dialogo_de_envio",
    "escreve_csv",
    "executa_enriquecimento",
    "origem_das_bibliotecas",
    "prepara_envio_externo",
    "vocabulario_proibido",
]

COLUNAS: Final = ("Via", "Tamanho", "Esperado", "Acertos", "p", "FDR")
_NENHUMA: Final = "(nenhuma)"
_FASES: Final = {
    "mapeando": "Reconhecendo os nomes na base do MetaboAnalyst",
    "calculando": "Calculando a sobre-representação",
    "baixando": "Baixando as bibliotecas de dados",
    "enviando": "Enviando ao servidor do MetaboAnalyst e esperando a ORA",
}
# A fase em que quem trabalha é o servidor de terceiro: nela o aviso do antivírus, que é sobre o R
# deste computador, não se aplica. As outras fases mantêm o aviso que tinham antes do modo externo.
_FASES_DO_SERVIDOR: Final = frozenset({"enviando"})

# Os rótulos do modo externo. As frases do seletor, dos dois itens, da nota e do título do envio
# estão também em `docs/EXPLICAR-cada-tela-e-cada-campo.html`, e um teste confere que as duas dizem
# o mesmo.
ROTULO_DO_SELETOR: Final = "Biblioteca de vias"
ITEM_LOCAL: Final = "SMPDB — no computador (modo local)"
ITEM_EXTERNO: Final = "KEGG — pelo servidor do MetaboAnalyst (modo externo)"
NOTA_DO_MODO_EXTERNO: Final = (
    "O servidor do MetaboAnalyst não aplica o fundo do ensaio na ORA KEGG: o universo é a "
    "biblioteca KEGG inteira, e a análise tende a falsos positivos."
)
TITULO_DO_ENVIO: Final = "Enviar ao servidor do MetaboAnalyst?"
ORA_EXTERNA: Final = "ORA do MSEA pelo servidor do MetaboAnalyst sobre a KEGG"
UNIVERSO_EXTERNO: Final = "a biblioteca KEGG inteira"
MOTOR_EXTERNO: Final = f"servidor do MetaboAnalyst ({ext.HOST}), ORA do MSEA"
BIBLIOTECA_DO_MODO_EXTERNO: Final = (
    f"**{NOTA_DO_MODO_EXTERNO}** Ao «Enriquecer», a lista é mapeada neste computador e o programa "
    f"mostra os nomes que sairiam para {ext.HOST}; nada sai sem a sua confirmação."
)
AVISO_DA_PRIMEIRA: Final = (
    "A primeira análise depois de instalar pode demorar mais: o antivírus confere os arquivos do R."
)
SEM_EXECUCAO: Final = (
    "Rode uma execução com dois grupos para ter o teste univariado que dá o corte."
)
SEM_TABELA: Final = "Carregue a tabela de identificação exportada do Progenesis."

# As frases ao lado dos botões desligados: cada uma diz por que o botão está desligado e o que fazer
# para ligá-lo. Ficam na linha do botão, então são curtas; o detalhe fica nos degraus e no estado. A
# recusa antecipada não tem frase aqui: usa a de `MENSAGENS`, para o botão e o estado dizerem o
# mesmo.
MOTIVO_SEM_TABELA_PARA_CONFIRMAR: Final = (
    "Carregue a tabela do Progenesis: as colunas a confirmar saem dela."
)
MOTIVO_SEM_EXECUCAO: Final = (
    "Falta o teste univariado: rode a «Análise estatística» com «Grupo A do contraste» e "
    "«Grupo B do contraste» preenchidos."
)
MOTIVO_SEM_TABELA: Final = "Carregue a tabela do Progenesis com «Carregar tabela do Progenesis…»."
MOTIVO_COLUNAS_POR_CONFIRMAR: Final = "Confirme as colunas da tabela com «Confirmar colunas»."
MOTIVO_EM_CURSO: Final = "Em curso: espere o fim, ou clique em «Cancelar»."
MOTIVO_NADA_A_CANCELAR: Final = "Nada em curso: o botão liga enquanto o enriquecimento corre."
MOTIVO_SEM_RESULTADO: Final = (
    "Ainda não há resultado: clique em «Enriquecer» para ter o que exportar."
)
BIBLIOTECA_LOCAL: Final = (
    "Biblioteca de vias: **SMPDB, vias humanas**. Tudo roda nesta máquina; nada é enviado."
)
FUNDO_DA_BIBLIOTECA: Final = (
    "**Fundo não específico do ensaio: tende a falsos positivos.** O universo passou a ser a "
    "biblioteca inteira, e não o que o ensaio consegue medir; vias podem parecer enriquecidas só "
    "porque a plataforma detecta os compostos delas."
)


ClasseDaMensagem = Literal["entrada", "ambiente", "defeito"]


@dataclass(frozen=True)
class Mensagem:
    classe: ClasseDaMensagem
    frase: str
    acao: str


# A mesma recusa chega por dois caminhos, do R (`lista_igual_ao_fundo`) e do núcleo antes de
# chamar o R (`igual_ao_fundo`), e as duas oferecem o fundo da biblioteca inteira.
_IGUAL_AO_FUNDO: Final = Mensagem(
    "entrada",
    "A lista é igual ao fundo, e então todo p sai 1.",
    "Carregue a identificação de todas as features; ou use o fundo da biblioteca inteira, "
    "sabendo que ele tende a falsos positivos.",
)
_OFERECEM_O_FUNDO_DA_BIBLIOTECA: Final = frozenset({"lista_igual_ao_fundo", "igual_ao_fundo"})

# Classe, frase e ação por desfecho. A classe diz de quem é a próxima ação: da entrada, do ambiente,
# ou de quem mantém o programa. Sem ela, 'nada mapeou' sairia como defeito e 'resposta malformada'
# como culpa de quem usa.
MENSAGENS: Final[dict[str, Mensagem]] = {
    "motor_ausente": Mensagem(
        "ambiente",
        "O motor R não está na pasta do programa.",
        "Reinstale o programa: a pasta «R» ao lado do executável sumiu ou foi movida.",
    ),
    "motor_falhou": Mensagem(
        "defeito", "O motor R falhou.", "Isto é um defeito a reportar, com o texto abaixo."
    ),
    "tempo_esgotado": Mensagem(
        "ambiente", "O R passou do tempo limite.", f"Tente de novo. {AVISO_DA_PRIMEIRA}"
    ),
    "resposta_malformada": Mensagem(
        "defeito",
        "O R devolveu uma resposta fora do contrato.",
        "Isto é um defeito a reportar, com o texto abaixo.",
    ),
    "cancelado": Mensagem("entrada", "A análise foi cancelada.", "Nada foi calculado."),
    "pedido_invalido": Mensagem(
        "defeito", "O pedido ao R saiu malformado.", "Isto é um defeito a reportar."
    ),
    "biblioteca_ausente": Mensagem(
        "ambiente",
        "Faltam bibliotecas de dados.",
        "Na pasta entregue ao laboratório elas viajam com o programa: reinstale. Na versão "
        "pública, aceite o download quando ele for oferecido.",
    ),
    "biblioteca_divergente": Mensagem(
        "ambiente",
        "As bibliotecas de dados não são as que viajaram com o programa.",
        "A pasta «vias-dados» foi alterada: reinstale o programa.",
    ),
    "fundo_invalido": Mensagem(
        "entrada",
        "Um nome do fundo tem quebra de linha ou tabulação.",
        "Confira a coluna do nome do composto na tabela.",
    ),
    "fundo_recusado": Mensagem(
        "entrada",
        "Mais da metade do fundo não foi reconhecida pela base do MetaboAnalyst.",
        "Confira se a coluna escolhida é a do nome do composto.",
    ),
    # O roteiro emite `lista_recusada` só quando nenhum nome da lista casa com a base
    # (`acao_ora_local`); a lista curta calcula, com aviso.
    "lista_recusada": Mensagem(
        "entrada",
        "Nenhum nome da lista foi reconhecido pela base do MetaboAnalyst.",
        "Confira a coluna do nome do composto.",
    ),
    "lista_igual_ao_fundo": _IGUAL_AO_FUNDO,
    "sem_casamento_na_biblioteca": Mensagem(
        "entrada",
        "Nenhum composto da lista está nas vias humanas da SMPDB.",
        "Confira a coluna do nome do composto; nem todo composto tem via nas vias humanas da "
        "SMPDB.",
    ),
    "lista_vazia": Mensagem(
        "entrada", "Nenhum composto passa o corte escolhido.", "Afrouxe o corte."
    ),
    "igual_ao_fundo": _IGUAL_AO_FUNDO,
    "nada_mapeou": Mensagem(
        "entrada",
        "Nenhum composto da lista foi reconhecido pela base do MetaboAnalyst.",
        "Confira a coluna do nome do composto.",
    ),
    "download_recusado": Mensagem(
        "entrada",
        "O download das bibliotecas não foi aceito.",
        "Sem elas, o enriquecimento não roda nesta versão.",
    ),
    "sem_rede": Mensagem(
        "ambiente",
        "O download das bibliotecas falhou por rede.",
        "Confira a conexão e tente de novo.",
    ),
    "transporte_recusado": Mensagem(
        "ambiente",
        "O download foi recusado por um controle de segurança da conexão.",
        "Tente de novo noutra rede; se persistir, reporte.",
    ),
    # O modo externo. Chaves próprias, e não as do download, embora o `Literal` de
    # `DesfechoDoEnvio` repita dois nomes dele (`sem_rede`, `transporte_recusado`): as frases do
    # download falam de bibliotecas baixadas, e lidas depois de um envio diriam o fato errado. A
    # tradução é `_DESFECHO_DO_ENVIO`.
    "envio_sem_rede": Mensagem(
        "ambiente",
        "O envio ao servidor do MetaboAnalyst falhou por rede.",
        "Confira a conexão e tente de novo.",
    ),
    "envio_transporte_recusado": Mensagem(
        "ambiente",
        "A conexão com o servidor do MetaboAnalyst foi recusada por um controle de segurança.",
        "Tente de novo noutra rede; se persistir, reporte.",
    ),
    "servidor_recusou": Mensagem(
        "ambiente",
        "O servidor do MetaboAnalyst recusou o pedido.",
        "Tente de novo mais tarde: o servidor é de terceiro e pode estar fora do ar.",
    ),
    "servidor_resposta_malformada": Mensagem(
        "defeito",
        "O servidor do MetaboAnalyst devolveu uma resposta fora do contrato.",
        "Isto é um defeito a reportar, com o texto abaixo.",
    ),
    # A G1 recusa por um nome que repete um rótulo da execução ou da tabela: é a entrada que o
    # produziu, e quem age é quem a escolheu.
    "envio_recusado": Mensagem(
        "entrada",
        "Um nome da lista coincide com um nome de amostra, de grupo ou de coluna da tabela.",
        "Nada foi enviado. O nome está no texto abaixo; confira a coluna do nome do composto.",
    ),
    # A G2 recusa a carga que o próprio programa montou: esquema diferente do fechado, ou texto que
    # não é nome confirmado. Nenhuma entrada do usuário chega lá sem passar pela G1: é defeito.
    "carga_recusada": Mensagem(
        "defeito",
        "A guarda do envio recusou o pedido montado pelo programa.",
        "Nada foi enviado. Isto é um defeito a reportar, com o texto abaixo.",
    ),
    "envio_nao_confirmado": Mensagem("entrada", "O envio não foi confirmado.", "Nada foi enviado."),
    # O erro 500 de uma lista de um ou dois compostos é do pedido, e não do servidor fora do ar: um
    # composto cujo único acerto cai numa só via dá 500, e um composto em três vias calcula. A frase
    # de `servidor_recusou` mandaria tentar mais tarde.
    "servidor_falhou_lista_curta": Mensagem(
        "entrada",
        "O servidor do MetaboAnalyst não calculou esta lista curta.",
        "Com tão poucos compostos o servidor pode falhar por causa da própria lista: num teste, um "
        "composto que só aparece numa via deu este erro. Acrescente compostos, ou use o modo "
        "local.",
    ),
    # O 'Cancelar' depois do 'Enviar': o pedido pode já ter saído, e a frase de `cancelado` ('Nada
    # foi calculado') calaria isso. O detalhe do transporte diz em que ponto o envio estava.
    "envio_cancelado": Mensagem(
        "entrada",
        "O envio foi cancelado.",
        "Se o pedido já tinha saído, os nomes da lista chegaram ao servidor do MetaboAnalyst; a "
        "resposta foi descartada, e nada foi calculado aqui.",
    ),
}

# O desfecho do envio e a chave da mensagem. Fechada sobre `DesfechoDoEnvio` sem `ok`, e um teste a
# confere nas duas direções: um desfecho novo sem tradução reprova lá, em vez de cair no `.get`
# abaixo e sair como resposta malformada.
_DESFECHO_DO_ENVIO: Final[dict[DesfechoDoEnvio, str]] = {
    "sem_rede": "envio_sem_rede",
    "transporte_recusado": "envio_transporte_recusado",
    "servidor_recusou": "servidor_recusou",
    "resposta_malformada": "servidor_resposta_malformada",
    "cancelado": "envio_cancelado",
}
_TITULO_DA_CLASSE: Final[dict[ClasseDaMensagem, str]] = {
    "entrada": "Sobre a sua entrada",
    "ambiente": "Sobre o ambiente",
    "defeito": "Defeito a reportar",
}
_NEUTRALIZA: Final = ("=", "+", "-", "@", "\t", "\r")


@dataclass(frozen=True)
class Falha:
    """Uma falha a mostrar, e um fato do lançador que viaja com ela.

    `o_r_respondeu` diz se alguma chamada desta análise teve o R a arrancar e a responder; o painel
    decide por ela se a próxima chamada leva o teto. O fato é do lançador e não se infere da chave
    da mensagem: um `motor_falhou` de `OSError`, o antivírus a barrar o `Rscript.exe`, é justamente
    o caso frio que precisa da chamada sem teto.
    """

    chave: str
    detalhe: str = ""
    oferece_fundo_da_biblioteca: bool = False
    o_r_respondeu: bool = False


@dataclass(frozen=True)
class ArquivoDaBiblioteca:
    """Um arquivo de dados que o R leu, com o hash e a procedência que o CSV leva."""

    arquivo: str
    sha256: str
    procedencia: str


@dataclass(frozen=True)
class OrigemDasBibliotecas:
    pasta: Path
    descricao: str
    validada: bool
    # Por arquivo, e não só o agregado: 'validadas: sim' não diz sobre que versão de cada arquivo o
    # resultado foi calculado.
    arquivos: tuple[ArquivoDaBiblioteca, ...] = ()


# Os desfechos em que o R arrancou e respondeu. `motor_falhou` fica de fora porque cobre também o
# R que nem pôde ser lançado (o `OSError` do lançador); contá-lo como frio quando ele chegou a
# correr custa só uma chamada a mais sem teto, e o erro oposto tira o teto que falta a quem
# arranca frio pela primeira vez.
_O_R_RESPONDEU: Final = frozenset({"ok", "recusado", "resposta_malformada"})


def _falha_do_r(r: RespostaDoR, *, ja_respondeu: bool = False) -> Falha:
    chave = r.recusa if r.desfecho == "recusado" and r.recusa else r.desfecho
    detalhe = r.motivo
    if MENSAGENS[chave].classe == "defeito" and r.stderr:
        detalhe = f"{detalhe}\n{r.stderr}"
    return Falha(
        chave,
        detalhe,
        chave in _OFERECEM_O_FUNDO_DA_BIBLIOTECA,
        o_r_respondeu=ja_respondeu or r.desfecho in _O_R_RESPONDEU,
    )


Executor = Callable[..., RespostaDoR]


def executa_enriquecimento(
    recorte: en.RecorteDaIdentificacao,
    motor: MotorDoR,
    bibliotecas: Path,
    *,
    fundo_da_biblioteca: bool,
    executor: Executor,
    cancelado: Callable[[], bool],
    fase: Callable[[str], None],
    fria: bool = False,
) -> en.ResultadoDaOra | Falha:
    """As duas chamadas ao R e as recusas entre elas. Sem Qt: o painel a corre num trabalhador.

    `fria` é a primeira chamada ao R neste processo: ela paga o cache frio e o antivírus, e vai
    sem teto rígido; só o cancelamento a para. A segunda chamada da mesma análise já encontra o R
    aquecido e leva o teto.

    Toda `Falha` depois de um `mapear` com `ok` sai com `o_r_respondeu=True`: o R já arrancou e
    respondeu nesta análise, ainda que a segunda chamada tenha sido cancelada ou recusada.
    """
    comum = {"r_home": motor.r_home, "roteiro": motor.roteiro, "cancelado": cancelado}
    fase("mapeando")
    pedido_mapear = {
        "action": "mapear",
        "dados_dir": str(bibliotecas),
        "lista": list(recorte.lista),
        "fundo": list(recorte.fundo),
    }
    r = executor(pedido_mapear, tempo_limite_s=None if fria else TEMPO_LIMITE_PADRAO_S, **comum)
    if r.desfecho != "ok":
        return _falha_do_r(r)
    try:
        lista, fundo = en.mapeamento_da_resposta(r.dados)
    except en.RespostaInvalidaError as erro:
        return Falha("resposta_malformada", str(erro), o_r_respondeu=True)
    preparo = en.prepara_ora(lista, fundo, fundo_da_biblioteca=fundo_da_biblioteca)
    if isinstance(preparo, en.Recusa):
        return Falha(
            preparo.motivo, preparo.texto, preparo.oferece_fundo_da_biblioteca, o_r_respondeu=True
        )
    fase("calculando")
    # Montado de uma vez, sem escrita por chave: o teste de contrato da janela lê todo subscrito
    # dos módulos da janela como leitura de resposta, e este é um pedido nosso, não uma resposta.
    fundo_do_pedido = {} if preparo.fundo is None else {"fundo": list(preparo.fundo)}
    pedido_ora: dict[str, object] = {
        "action": "ora_local",
        "dados_dir": str(bibliotecas),
        "lista": list(preparo.lista),
        **fundo_do_pedido,
    }
    r2 = executor(pedido_ora, tempo_limite_s=TEMPO_LIMITE_PADRAO_S, **comum)
    if r2.desfecho != "ok":
        return _falha_do_r(r2, ja_respondeu=True)
    try:
        return en.ora_da_resposta(r2.dados)
    except en.RespostaInvalidaError as erro:
        return Falha("resposta_malformada", str(erro), o_r_respondeu=True)


# ---------------------------------------------------------------------------- o modo externo


# As chaves do corpo de `/execucoes` cujo valor de texto é nome de amostra ou de grupo, com o
# esquema de onde cada uma vem (`api/esquemas.py`, e `domain/spec.py` para as duas do
# `QualityReport`, que viaja no `dict` livre `qualidade`). Um teste confere a lista contra o OpenAPI
# nas duas direções: propriedade de texto com nome de amostra ou de grupo que não esteja aqui nem
# em `_CHAVES_FORA_DA_G1` reprova, e chave daqui que nenhum esquema declare também.
_CHAVES_QUE_NOMEIAM: Final = frozenset(
    {
        "grupo_a",  # ResumoDoVolcano
        "grupo_b",  # ResumoDoVolcano
        "grupos",  # ResumoDaAnova, ResumoDoKmeans, ResumoDoSom, ResumoDoClassificador
        "amostras",  # ResumoDoKmeans, ResumoDoSom
        "rotulos",  # ResumoDoKmeans: inteiros hoje, e por isso não contribui
        "amostra",  # CaixaPorAmostra, em distribuicoes.caixas_por_amostra_antes/_depois
        "grupos_por_amostra",  # ResumoDoPlsda
        "grupo_positivo",  # ResumoDoPlsda
        "grupo_negativo",  # ResumoDoPlsda
        "ordem_dos_grupos",  # ResumoDaBuscaPorPadrao
        "grupo",  # ClasseDaComposicao, em composicao_por_classe.classes
        "plsda_groups",  # QualityReport, em qualidade
        "plsda_multiclass_groups",  # QualityReport, em qualidade
    }
)
# As chaves que falam de amostra ou de grupo e não carregam nome de nenhum, cada uma com o motivo.
# Ficam fora da G1 por desenho, e estar aqui é o que as tira da reprovação do teste acima.
_CHAVES_FORA_DA_G1: Final[dict[str, str]] = {
    "amostragem": (
        "a frase que a figura escreve sobre o recorte (ResumoDaCorrelacao, ResumoDaRede, "
        "ResumoDoClassificador, ResumoDoKmeans, ResumoDoSom); a G1 compara o nome inteiro, e uma "
        "frase nunca é igual a um nome canônico"
    ),
    "tipos_de_amostra_antes": (
        "ParDePcas: o `SampleType` de cada injeção, vocabulário fechado `qc`/`sample`/`validate`, "
        "e não o nome dela"
    ),
    "tipos_de_amostra_depois": "ParDePcas: o mesmo `SampleType`, do lado depois da correção",
}


def _textos(valores: object) -> set[str]:
    """Os textos não vazios de um campo que deveria ser lista de textos, ou nenhum.

    Com guarda, item a item: um campo pode chegar `null` ou com outro tipo. Um campo fora da forma
    contribui com nada em vez de quebrar o painel. Hoje só o cabeçalho da tabela passa por aqui.
    """
    if isinstance(valores, str):
        valores = [valores]
    if not isinstance(valores, list):
        return set()
    return {v for v in valores if isinstance(v, str) and v.strip()}


def _nomes_do_corpo(corpo: object) -> set[str]:
    """Todo texto não vazio sob uma chave de `_CHAVES_QUE_NOMEIAM`, em qualquer profundidade.

    'Sob' é literal: abaixo de uma chave que nomeia, todo texto conta, numa lista, numa lista de
    listas ou num objeto aninhado nela, porque para uma guarda vetar a mais é a direção barata.
    Fora dessas chaves nada conta, nem as chaves dos objetos. Número, lógico e `null` são ignorados
    onde estiverem, e um corpo que não é dicionário só contribui com o que tiver sob uma dessas
    chaves.

    Iterativo, com pilha, e sem guarda de ciclo: o corpo vem de `json.loads`, que só produz árvore.
    As leituras são por iteração e não por chave literal, então o teste de contrato da janela não
    as vê; quem confere as chaves contra o esquema é o teste citado em `_CHAVES_QUE_NOMEIAM`.
    """
    achados: set[str] = set()
    pendentes: list[tuple[object, bool]] = [(corpo, False)]
    while pendentes:
        valor, sob_nome = pendentes.pop()
        if isinstance(valor, dict):
            pendentes.extend((v, sob_nome or k in _CHAVES_QUE_NOMEIAM) for k, v in valor.items())
        elif isinstance(valor, list | tuple):
            pendentes.extend((v, sob_nome) for v in valor)
        elif sob_nome and isinstance(valor, str) and valor.strip():
            achados.add(valor)
    return achados


def vocabulario_proibido(corpo: Any, cabecalho: Iterable[str] = ()) -> frozenset[str]:
    """O que a G1 recusa na carga: rótulos da execução e da tabela que não podem sair da máquina.

    A G1 recusa se algum nome bater com nome de amostra, rótulo de grupo, `CANONICAL_GROUPS` ou
    cabeçalho da tabela do Progenesis. Três fontes:

    - todo texto do corpo de `/execucoes` sob uma chave que nomeia amostra ou grupo, em qualquer
      profundidade (`_nomes_do_corpo`); as chaves e os esquemas de onde vêm estão em
      `_CHAVES_QUE_NOMEIAM`. Nomes de amostra entram em toda execução, porque `distribuicoes` e
      `composicao_por_classe` não são opcionais em `ExecucaoResposta`; as caixas por amostra, porém,
      são um sorteio de até `TETO_DE_CAIXAS` (50) amostras quando há mais, e a amostra fora do
      sorteio só entra pelo K-means, pelo SOM ou pelo cabeçalho da tabela, que traz a abundância
      por amostra;
    - os grupos canônicos do laboratório (`CANONICAL_GROUPS`);
    - o cabeçalho inteiro da tabela de identificação lida (`TabelaDeIdentificacao.cabecalho`).

    Fica de fora: as chaves de `_CHAVES_FORA_DA_G1`, cada uma com o motivo; nome de amostra ou de
    grupo sob uma chave cujo nome não o diga, porque o teste escolhe as candidatas pelo nome da
    propriedade no OpenAPI; as chaves de um objeto; e texto que contém um nome sem ser igual a ele,
    porque `confere_g1` compara a igualdade, com `casefold`.

    Corpo que não é dicionário, listas e valores que não são texto não quebram. Os grupos canônicos
    estão sempre, e é isso que impede o vocabulário vazio, que `confere_g1` recusa por aprovar
    tudo.
    """
    vetados: set[str] = set(CANONICAL_GROUPS) | _textos(list(cabecalho)) | _nomes_do_corpo(corpo)
    return frozenset(vetados)


@dataclass(frozen=True)
class EnvioPreparado:
    """O fim da etapa 1: os nomes conferidos pela G1 e a carga conferida pela G2.

    Viaja do trabalhador para a thread da tela, que mostra `nomes` no diálogo e, com 'Enviar',
    entrega `carga`, os mesmos bytes que a G2 leu, à etapa 3.
    """

    nomes: tuple[str, ...]
    carga: bytes


@dataclass(frozen=True)
class EnvioExterno:
    """O que o CSV do modo externo diz sobre o envio: os nomes que saíram e quando."""

    nomes: tuple[str, ...]
    consultado_em: datetime


@dataclass(frozen=True)
class EntradasDoRecorte:
    """O que, além do volcano, decide o recorte de um enriquecimento.

    A tabela vai por identidade: uma tabela relida, ou reconfirmada com outras colunas, é outro
    objeto, e um resultado calculado com a anterior não é o desta tela.
    """

    tabela: TabelaDeIdentificacao | None
    colunas: str
    corte: tuple[object, float]
    modo: object

    def mesmas(self, outras: EntradasDoRecorte) -> bool:
        return (
            self.tabela is outras.tabela
            and self.colunas == outras.colunas
            and self.corte == outras.corte
            and self.modo == outras.modo
        )


@dataclass(frozen=True)
class FotoDoEnriquecimento:
    """Um resultado do enriquecimento e o que o produziu, para a janela o repor.

    O que `_mostra_resultado` lê além do resultado: a origem das bibliotecas, o envio externo que o
    produziu, e se o universo foi a biblioteca inteira. E as entradas do recorte, porque a volta à
    matriz só o repõe se a tela ainda tiver as mesmas.
    """

    resultado: en.ResultadoDaOra
    origem: OrigemDasBibliotecas | None
    envio: EnvioExterno | None
    fundo_da_biblioteca: bool
    entradas: EntradasDoRecorte


@dataclass(frozen=True)
class _Voo:
    """O contexto de um cálculo em voo, pelo token dele: o corpo da execução e o que ele usou."""

    corpo: Any
    entradas: EntradasDoRecorte
    origem: OrigemDasBibliotecas | None
    envio: EnvioExterno | None
    fundo_da_biblioteca: bool
    cancelar: threading.Event


Enviador = Callable[..., RespostaDoServidor]


def prepara_envio_externo(
    recorte: en.RecorteDaIdentificacao,
    motor: MotorDoR,
    bibliotecas: Path,
    *,
    proibidos: Iterable[str],
    executor: Executor,
    cancelado: Callable[[], bool],
    fase: Callable[[str], None],
    fria: bool = False,
) -> EnvioPreparado | Falha:
    """A etapa 1 do modo externo, sem Qt: mapear no R local, G1, montar e conferir a carga (G2).

    O `mapear` é o mesmo pedido do modo local sem o fundo: o servidor não o aplica na ORA KEGG,
    então mapeá-lo seria trabalho do R para nada. A primeira chamada fria vai sem teto, pela mesma
    regra de `executa_enriquecimento`.

    `permitidos` da G1 são os próprios canônicos: só sai o que o mapeamento confirmou, e a recusa
    que resta à G1 é a de um canônico que repete um rótulo de `proibidos`.
    """
    comum = {"r_home": motor.r_home, "roteiro": motor.roteiro, "cancelado": cancelado}
    fase("mapeando")
    pedido_mapear = {
        "action": "mapear",
        "dados_dir": str(bibliotecas),
        "lista": list(recorte.lista),
    }
    r = executor(pedido_mapear, tempo_limite_s=None if fria else TEMPO_LIMITE_PADRAO_S, **comum)
    if r.desfecho != "ok":
        return _falha_do_r(r)
    try:
        lista, _ = en.mapeamento_da_resposta(r.dados)
    except en.RespostaInvalidaError as erro:
        return Falha("resposta_malformada", str(erro), o_r_respondeu=True)
    nomes = en.canonicos_do_envio(lista)
    if isinstance(nomes, en.Recusa):
        return Falha(nomes.motivo, nomes.texto, o_r_respondeu=True)
    try:
        en.confere_g1(nomes, permitidos=nomes, proibidos=proibidos)
    except en.EnvioRecusadoError as erro:
        return Falha("envio_recusado", str(erro), o_r_respondeu=True)
    except ValueError as erro:
        # A G1 sem vocabulário proibido: os grupos canônicos o tornam inalcançável pelo painel, e
        # quem chega aqui é um chamador que não o montou: defeito do programa, não da entrada.
        return Falha("carga_recusada", str(erro), o_r_respondeu=True)
    try:
        carga = ext.monta_carga(nomes)
        ext.confere_carga(carga, nomes)
    except ext.CargaRecusadaError as erro:
        return Falha("carga_recusada", str(erro), o_r_respondeu=True)
    return EnvioPreparado(nomes, carga)


def conclui_envio_externo(
    envio: EnvioPreparado,
    *,
    enviador: Enviador,
    cancelado: Callable[[], bool],
    relogio: Callable[[], float],
    fase: Callable[[str], None],
) -> en.ResultadoDaOra | Falha:
    """A etapa 3, sem Qt: enviar a carga já conferida e passar a resposta pela conferência local.

    `ora_da_resposta` é a mesma do modo local: cada p é recalculado pela hipergeométrica, e um p
    que não bate é resposta malformada. O universo tem de ser o declarado na tela, a biblioteca
    inteira; outro valor seria a tela a afirmar 'KEGG inteira' sobre um resultado que diz outra
    coisa.
    """
    fase("enviando")
    r = enviador(envio.carga, list(envio.nomes), cancelado=cancelado, relogio=relogio)
    if (
        r.desfecho == "servidor_recusou"
        and r.status == 500
        and len(envio.nomes) < en.POUCOS_COMPOSTOS
    ):
        return Falha("servidor_falhou_lista_curta", r.motivo, o_r_respondeu=True)
    if r.desfecho != "ok":
        chave = _DESFECHO_DO_ENVIO.get(r.desfecho, "servidor_resposta_malformada")
        return Falha(chave, r.motivo, o_r_respondeu=True)
    if r.dados is None:
        return Falha(
            "servidor_resposta_malformada", "o envio devolveu «ok» sem dados", o_r_respondeu=True
        )
    try:
        resultado = en.ora_da_resposta(r.dados)
    except en.RespostaInvalidaError as erro:
        return Falha("servidor_resposta_malformada", str(erro), o_r_respondeu=True)
    if resultado.tipo_do_universo != "biblioteca":
        return Falha(
            "servidor_resposta_malformada",
            f"o universo declarado é «{resultado.tipo_do_universo}», e não a biblioteca inteira",
            o_r_respondeu=True,
        )
    return resultado


def neutraliza(celula: str) -> str:
    """Uma célula de CSV que o Excel leria como fórmula ganha um apóstrofo na frente."""
    return "'" + celula if celula.startswith(_NEUTRALIZA) else celula


def _conta(n: int, singular: str, plural: str | None = None) -> str:
    """'1 composto', '2 compostos': a tela diz o número com a palavra que ele pede."""
    return f"{n} {singular}" if n == 1 else f"{n} {plural or singular + 's'}"


def _contraste(volcano: Any) -> str:
    """O par de grupos do teste univariado que deu o p, para o CSV.

    Lido com guarda, campo a campo: nada do fio passou pelo mypy, e um campo pode chegar `null`.
    O nome da variável é `volcano` de propósito: é a base que o teste de contrato da janela confere
    contra `ResumoDoVolcano`.
    """
    if not isinstance(volcano, dict):
        return "não declarado na execução"
    grupo_a, grupo_b = volcano.get("grupo_a"), volcano.get("grupo_b")
    if not (isinstance(grupo_a, str) and grupo_a and isinstance(grupo_b, str) and grupo_b):
        return "não declarado na execução"

    def com_n(grupo: str, n: object) -> str:
        valido = isinstance(n, int) and not isinstance(n, bool)
        return f"{grupo} (n = {n})" if valido else grupo

    return (
        f"{com_n(grupo_a, volcano.get('n_grupo_a'))} contra "
        f"{com_n(grupo_b, volcano.get('n_grupo_b'))}"
    )


class _Sinais(QObject):
    concluido = Signal(object, int)
    fase = Signal(str, int)


def despacha_trabalho(
    tarefa: Callable[[Callable[[str], None]], object], sinais: _Sinais, token: int
) -> None:
    """A lógica de `_Trabalho.run`, chamável da thread principal, no padrão de `despacha_vias`.

    O `coverage` não enxerga as threads nativas do `QThreadPool`, e com o corpo dentro de `run` o
    código apareceria com 0% mesmo quando corre. `BaseException` e não `Exception`: uma exceção que
    escapa de um `QRunnable` some.
    """
    try:
        resultado: object = tarefa(lambda f: sinais.fase.emit(f, token))
    except BaseException as erro:
        resultado = erro
    sinais.concluido.emit(resultado, token)


class _Trabalho(QRunnable):
    def __init__(
        self, tarefa: Callable[[Callable[[str], None]], object], sinais: _Sinais, token: int
    ) -> None:
        super().__init__()
        self._tarefa, self._sinais, self._token = tarefa, sinais, token

    def run(self) -> None:
        despacha_trabalho(self._tarefa, self._sinais, self._token)


def _abre_arquivo(pai: QWidget) -> Path | None:
    caminho, _ = QFileDialog.getOpenFileName(
        pai, "Tabela de identificação do Progenesis", "", "CSV (*.csv *.txt);;Todos (*)"
    )
    return Path(caminho) if caminho else None


def _salva_arquivo(pai: QWidget) -> Path | None:
    caminho, _ = QFileDialog.getSaveFileName(
        pai, "Exportar o enriquecimento", "enriquecimento.csv", "CSV (*.csv)"
    )
    return Path(caminho) if caminho else None


def _confirma(pai: QWidget, titulo: str, texto: str) -> bool:
    caixa = QMessageBox(QMessageBox.Icon.Question, titulo, texto, parent=pai)
    sim = caixa.addButton("Baixar", QMessageBox.ButtonRole.AcceptRole)
    nao = caixa.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
    caixa.setDefaultButton(nao)
    caixa.exec()
    return caixa.clickedButton() is sim


def texto_do_envio(n: int) -> str:
    """O que o diálogo diz antes da lista: a contagem no topo, o destino, e o que não sai."""
    contagem = (
        "1 nome de composto será enviado" if n == 1 else f"{n} nomes de compostos serão enviados"
    )
    return (
        f"{contagem} ao servidor do MetaboAnalyst ({ext.HOST}), que calcula a ORA sobre a "
        "biblioteca KEGG.\n\n"
        "Só saem os nomes de compostos que o mapeamento local confirmou — nem a tabela, nem "
        "amostras, nem o fundo. A lista inteira está abaixo."
    )


def dialogo_de_envio(pai: QWidget | None, nomes: list[str]) -> QDialog:
    """O diálogo de confirmação do envio, montado e não mostrado; `_confirma_envio` o executa.

    O foco padrão e o botão padrão são 'Cancelar': um Enter distraído não envia nada. A lista vai
    numa área rolável só de leitura porque ela é o objeto da decisão: quem confirma tem de poder ler
    cada nome que sai, e não um resumo.
    """
    dialogo = QDialog(pai)
    dialogo.setWindowTitle(TITULO_DO_ENVIO)
    corpo = QVBoxLayout(dialogo)
    texto = QLabel(texto_do_envio(len(nomes)), dialogo)
    texto.setTextFormat(Qt.TextFormat.PlainText)
    texto.setWordWrap(True)
    corpo.addWidget(texto)
    lista = QPlainTextEdit("\n".join(nomes), dialogo)
    lista.setObjectName("nomes_do_envio")
    lista.setReadOnly(True)
    corpo.addWidget(lista, 1)
    botoes = QDialogButtonBox(dialogo)
    enviar = botoes.addButton("Enviar", QDialogButtonBox.ButtonRole.AcceptRole)
    cancelar = botoes.addButton("Cancelar", QDialogButtonBox.ButtonRole.RejectRole)
    enviar.setObjectName("enviar")
    cancelar.setObjectName("cancelar")
    enviar.setAutoDefault(False)
    enviar.setDefault(False)
    cancelar.setDefault(True)
    botoes.accepted.connect(dialogo.accept)
    botoes.rejected.connect(dialogo.reject)
    corpo.addWidget(botoes)
    cancelar.setFocus()
    return dialogo


def _confirma_envio(pai: QWidget, nomes: list[str]) -> bool:
    dialogo = dialogo_de_envio(pai, nomes)
    try:
        return bool(dialogo.exec() == QDialog.DialogCode.Accepted)
    finally:
        # Filho do painel, ele viveria até o painel morrer: um por envio, acumulados.
        dialogo.deleteLater()


def origem_das_bibliotecas(motor: MotorDoR) -> OrigemDasBibliotecas | Falha | None:
    """A pasta das bibliotecas a usar; `None` quando é preciso baixar (versão pública).

    A pasta fixada, quando existe, é a da pasta do laboratório e só passa conferida pelo manifesto;
    divergente vence ausente, porque pede outra ação (reinstalar contra baixar).
    """
    manifesto = dados_vias.le_manifesto()
    if motor.bibliotecas_fixadas.exists():
        conferencia = dados_vias.confere_pasta(motor.bibliotecas_fixadas, manifesto)
        if conferencia.estado == "ok":
            # Conferida, a cópia tem o hash do manifesto, e a data da versão é a do manifesto.
            procedencia = (
                f"fixado, conferido contra o manifesto de {dados_vias.le_data_do_manifesto()}"
            )
            fixados = tuple(
                ArquivoDaBiblioteca(b.arquivo, b.sha256, procedencia) for b in manifesto
            )
            return OrigemDasBibliotecas(
                motor.bibliotecas_fixadas, "da pasta do laboratório, conferidas", True, fixados
            )
        chave = (
            "biblioteca_divergente" if conferencia.estado == "divergente" else "biblioteca_ausente"
        )
        return Falha(chave, ", ".join(conferencia.divergentes + conferencia.ausentes))
    baixadas = dados_vias.confere_baixadas(motor.bibliotecas_baixadas, manifesto)
    if baixadas is None:
        return None
    validada = all(b.validada for b in baixadas)
    quando = min(b.baixada_em for b in baixadas)
    descricao = f"baixadas em {quando}" + (
        "" if validada else ", **não validadas** (o servidor as trocou)"
    )
    arquivos = tuple(
        ArquivoDaBiblioteca(
            b.arquivo,
            b.sha256,
            f"baixado em {b.baixada_em}" + ("" if b.validada else ", não validado"),
        )
        for b in baixadas
    )
    return OrigemDasBibliotecas(motor.bibliotecas_baixadas, descricao, validada, arquivos)


_DESFECHO_DO_DOWNLOAD: Final = {
    "recusado": "download_recusado",
    "cancelado": "cancelado",
    "sem_rede": "sem_rede",
    "transporte_recusado": "transporte_recusado",
}


TEXTO_DO_DOWNLOAD: Final = (
    "Esta versão não traz as bibliotecas de dados do MetaboAnalyst. Elas podem ser baixadas agora "
    "do servidor do MetaboAnalyst (metaboanalyst.ca), cerca de 18 MB, e ficam guardadas nesta "
    "máquina para as próximas análises.\n\n"
    "São arquivos de terceiro, que o R vai carregar. Se o servidor tiver trocado algum deles desde "
    "a versão conferida, o resultado sai marcado como calculado com biblioteca não validada.\n\n"
    "Nada da sua análise é enviado: só os arquivos públicos são baixados."
)


class PainelDeEnriquecimento(QWidget):
    """O segundo modo da aba de vias. Ver a docstring do módulo."""

    # Cada resultado que chega, com o corpo da execução de que ele é e se foi pintado. A janela
    # guarda-o com a matriz dona desse corpo e repõe-no na volta a ela.
    resultado_concluido = Signal(object, object, bool)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        executor: Executor = corre_roteiro_de_vias,
        motor: Callable[[], MotorDoR] = resolve_motor,
        abre_arquivo: Callable[[QWidget], Path | None] = _abre_arquivo,
        salva_arquivo: Callable[[QWidget], Path | None] = _salva_arquivo,
        confirma: Callable[[QWidget, str, str], bool] = _confirma,
        bibliotecas: Callable[
            [MotorDoR], OrigemDasBibliotecas | Falha | None
        ] = origem_das_bibliotecas,
        relogio: Callable[[], float] = agora,
        instante: Callable[[], datetime] = instante_utc,
        confirma_envio: Callable[[QWidget, list[str]], bool] = _confirma_envio,
        enviador: Enviador = ext.envia_ao_servidor,
    ) -> None:
        """O executor, o motor e os diálogos são injetáveis para o teste.

        Os padrões são os de produção. O relógio e o instante vêm de `interface/decorrido`, porque
        nenhum
        arquivo da janela lê o relógio direto. `enviador` é o único caminho do painel para fora da
        máquina, e o teste o troca por um duplo: nenhum teste fala com a rede.
        """
        super().__init__(parent)
        self._executor = executor
        self._motor = motor
        self._abre_arquivo = abre_arquivo
        self._salva_arquivo = salva_arquivo
        self._confirma = confirma
        self._confirma_envio = confirma_envio
        self._enviador = enviador
        self._origem_das_bibliotecas = bibliotecas
        self._relogio_de_parede = instante
        self._agora = relogio
        self._depois_do_download = False
        # O R deste processo já respondeu uma vez? Até lá, a primeira chamada vai sem teto.
        self._r_aquecido = False
        # O que a G1 recusa, tirado do corpo da execução em `recebe_volcano`; o cabeçalho da tabela
        # entra no instante do 'Enriquecer', porque a tabela pode mudar depois do corpo.
        self._proibidos: frozenset[str] = vocabulario_proibido({})
        # O envio da etapa 3 em voo, e o do resultado que está na tela. Dois e não um: o segundo é
        # o que o CSV carimba, e não pode ser trocado por um envio que ainda não voltou.
        self._envio_em_voo: EnvioExterno | None = None
        self._envio_do_resultado: EnvioExterno | None = None
        self._volcano: Any = None
        self._tabela: TabelaDeIdentificacao | None = None
        # As colunas com que `_tabela` foi lida, por extenso, para o CSV dizer de onde veio a lista.
        self._colunas = ""
        self._recorte: en.RecorteDaIdentificacao | None = None
        self._resultado: en.ResultadoDaOra | None = None
        self._origem: OrigemDasBibliotecas | None = None
        self._caminho: Path | None = None
        self._token = 0
        self._cancelar = threading.Event()
        # O corpo da execução na tela, o contexto de cada cálculo em voo pelo token, e todos os
        # cancelamentos. Uma troca de matriz não cancela o R em voo: o resultado volta com a matriz
        # dele. Por isso o fechamento cancela todos, e não só o último; e na troca o cancelamento
        # corrente passa ao do cálculo da matriz que entra, ou a um que nenhum cálculo lê, para que
        # uma execução nova da que entra não cancele o da que saiu.
        self._corpo: Any = None
        self._voos: dict[int, _Voo] = {}
        self._cancelamentos: list[threading.Event] = []
        self._sem_calculo = threading.Event()
        self._fundo_da_biblioteca = False
        self._inicio = 0.0
        self._fase = ""
        self._piscina = QThreadPool()
        self._piscina.setMaxThreadCount(1)
        self._sinais = _Sinais(self)
        self._sinais.concluido.connect(self._recebe)
        self._sinais.fase.connect(self._recebe_fase)
        self._relogio = QTimer(self)
        self._relogio.setInterval(1000)
        self._relogio.timeout.connect(self._mostra_espera)

        # O filtro é do painel e vive com ele: ver a docstring de `sem_roda_sem_foco`.
        self._filtro_da_roda = FiltroDaRoda(self)
        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        # A biblioteca é escolhida a cada execução, e o padrão é o modo local: o primeiro item de um
        # `QComboBox` é o selecionado, então a ordem é o valor de quem não mexer nele.
        form_biblioteca = QFormLayout()
        self.biblioteca_de_vias = sem_roda_sem_foco(QComboBox(self), self._filtro_da_roda)
        self.biblioteca_de_vias.addItem(ITEM_LOCAL, "local")
        self.biblioteca_de_vias.addItem(ITEM_EXTERNO, "externo")
        self.biblioteca_de_vias.currentIndexChanged.connect(self._muda_biblioteca)
        form_biblioteca.addRow(ROTULO_DO_SELETOR, self.biblioteca_de_vias)
        raiz.addLayout(form_biblioteca)
        self.biblioteca = QLabel(BIBLIOTECA_LOCAL, self)
        self.biblioteca.setTextFormat(Qt.TextFormat.MarkdownText)
        self.biblioteca.setWordWrap(True)
        raiz.addWidget(self.biblioteca)

        linha_tabela = QHBoxLayout()
        self.botao_tabela = QPushButton("Carregar tabela do Progenesis…", self)
        self.botao_tabela.clicked.connect(self._carrega_tabela)
        self.arquivo = QLabel(SEM_TABELA, self)
        self.arquivo.setTextFormat(Qt.TextFormat.PlainText)
        linha_tabela.addWidget(self.botao_tabela)
        linha_tabela.addWidget(self.arquivo, 1)
        raiz.addLayout(linha_tabela)

        colunas = QGroupBox("Colunas da tabela", self)
        form = QFormLayout(colunas)
        self.coluna_feature = sem_roda_sem_foco(QComboBox(colunas), self._filtro_da_roda)
        self.coluna_composto = sem_roda_sem_foco(QComboBox(colunas), self._filtro_da_roda)
        self.coluna_aceito = sem_roda_sem_foco(QComboBox(colunas), self._filtro_da_roda)
        form.addRow("Feature (tempo_massa)", self.coluna_feature)
        form.addRow("Nome do composto", self.coluna_composto)
        form.addRow("Candidato aceito (opcional)", self.coluna_aceito)
        # Trocar uma coluna depois de confirmar desfaz a confirmação: a tabela lida é a das colunas
        # de antes, e sem isso 'Enriquecer' calcularia e exportaria com elas enquanto a caixa mostra
        # outra.
        for combo in (self.coluna_feature, self.coluna_composto, self.coluna_aceito):
            combo.currentIndexChanged.connect(self._muda_coluna)
        self.botao_confirma = QPushButton("Confirmar colunas", colunas)
        self.botao_confirma.clicked.connect(self._confirma_colunas)
        # Numa caixa, e não direto no formulário: a frase do botão desligado fica ao lado dele, e um
        # formulário não tem 'ao lado' sem mexer nas linhas dos outros (`desliga_com_motivo`).
        linha_confirma = QHBoxLayout()
        linha_confirma.addWidget(self.botao_confirma)
        linha_confirma.addStretch(1)
        form.addRow(linha_confirma)
        raiz.addWidget(colunas)
        desliga_com_motivo(self.botao_confirma, MOTIVO_SEM_TABELA_PARA_CONFIRMAR)

        corte = QGroupBox("Corte da lista significativa", self)
        form_corte = QFormLayout(corte)
        self.tipo_do_corte = sem_roda_sem_foco(QComboBox(corte), self._filtro_da_roda)
        for valor, rotulo in (("p", "p bruto"), ("q", "q de FDR"), ("n_melhores", "N melhores")):
            self.tipo_do_corte.addItem(rotulo, valor)
        self.valor_do_corte = sem_roda_sem_foco(QDoubleSpinBox(corte), self._filtro_da_roda)
        self.valor_do_corte.setDecimals(4)
        self.valor_do_corte.setRange(0.0001, 0.5)
        self.valor_do_corte.setSingleStep(0.01)
        self.valor_do_corte.setValue(0.05)
        self.tipo_do_corte.currentIndexChanged.connect(self._muda_tipo_do_corte)
        self.valor_do_corte.valueChanged.connect(self._invalida_e_recorta)
        form_corte.addRow("Corte", self.tipo_do_corte)
        form_corte.addRow("Limiar", self.valor_do_corte)
        raiz.addWidget(corte)

        self.degraus = QLabel(SEM_EXECUCAO, self)
        self.degraus.setTextFormat(Qt.TextFormat.MarkdownText)
        self.degraus.setWordWrap(True)
        raiz.addWidget(self.degraus)

        botoes = QHBoxLayout()
        self.botao_enriquece = QPushButton("Enriquecer", self)
        self.botao_enriquece.clicked.connect(lambda: self._enriquece(fundo_da_biblioteca=False))
        self.botao_fundo_biblioteca = QPushButton("Usar o fundo da biblioteca inteira", self)
        self.botao_fundo_biblioteca.setVisible(False)
        self.botao_fundo_biblioteca.clicked.connect(
            lambda: self._enriquece(fundo_da_biblioteca=True)
        )
        self.botao_cancela = QPushButton("Cancelar", self)
        self.botao_cancela.clicked.connect(self._cancela)
        self.botao_exporta = QPushButton("Exportar CSV…", self)
        self.botao_exporta.clicked.connect(self._exporta)
        for b in (
            self.botao_enriquece,
            self.botao_fundo_biblioteca,
            self.botao_cancela,
            self.botao_exporta,
        ):
            botoes.addWidget(b)
        botoes.addStretch(1)
        raiz.addLayout(botoes)
        # Os três nascem desligados, cada um com a sua frase, que só tem lugar depois de o botão
        # estar na caixa.
        desliga_com_motivo(self.botao_enriquece, MOTIVO_SEM_EXECUCAO)
        desliga_com_motivo(self.botao_cancela, MOTIVO_NADA_A_CANCELAR)
        desliga_com_motivo(self.botao_exporta, MOTIVO_SEM_RESULTADO)

        self.estado = QLabel("", self)
        self.estado.setTextFormat(Qt.TextFormat.MarkdownText)
        self.estado.setWordWrap(True)
        self.estado.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        raiz.addWidget(self.estado)

        self.tabela = QTableWidget(0, len(COLUNAS), self)
        self.tabela.setHorizontalHeaderLabels(list(COLUNAS))
        cabecalho = self.tabela.horizontalHeader()
        if cabecalho is not None:  # pragma: no cover - o Qt devolve-o num widget construído
            cabecalho.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            cabecalho.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tabela.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        raiz.addWidget(self.tabela, 1)

        self.rodape = QLabel("", self)
        self.rodape.setTextFormat(Qt.TextFormat.MarkdownText)
        self.rodape.setWordWrap(True)
        raiz.addWidget(self.rodape)

    # ------------------------------------------------------------------ entrada
    def recebe_volcano(self, corpo: Any, *, por_troca: bool = False) -> None:
        """Guarda as duas colunas do teste univariado. Avança o token e refaz o recorte.

        Do mesmo corpo sai o vocabulário que a G1 do modo externo recusa (`vocabulario_proibido`):
        uma segunda leitura da execução para a guarda poderia ser de outra execução.

        Com `por_troca`, a janela trocou de matriz, e o R em voo não é cancelado. Uma execução nova
        torna o cálculo da anterior inútil, e cancelá-lo é o certo; uma troca de matriz não, porque
        uma tecla no modo custaria um R de minutos. O token avança na mesma, e o resultado que chega
        com o token velho vai à janela pelo sinal, com o corpo de que é.
        """
        self._corpo = corpo
        corpo = corpo if isinstance(corpo, dict) else {}
        volcano = corpo.get("volcano")
        self._volcano = volcano
        self._proibidos = vocabulario_proibido(corpo)
        if por_troca:
            self._invalida_sem_cancelar()
            self._cancelar = next(
                (voo.cancelar for voo in self._voos.values() if voo.corpo is self._corpo),
                self._sem_calculo,
            )
        else:
            self._invalida_e_recorta()

    def _modo_externo(self) -> bool:
        return bool(self.biblioteca_de_vias.currentData() == "externo")

    def _muda_biblioteca(self) -> None:
        """Trocar a biblioteca invalida o que está em voo e refaz o recorte com as recusas dela."""
        self.biblioteca.setText(
            BIBLIOTECA_DO_MODO_EXTERNO if self._modo_externo() else BIBLIOTECA_LOCAL
        )
        self._invalida_e_recorta()

    def _recusa_previa(self, recorte: en.RecorteDaIdentificacao) -> en.Recusa | None:
        """A recusa antecipada do modo escolhido, sobre o recorte da tela.

        No modo externo só a lista vazia é recusada: não há fundo, e a lista igual ao fundo é uma
        lista como outra qualquer contra a biblioteca KEGG inteira.
        """
        recusa = en.recusa_previa(recorte)
        if self._modo_externo() and recusa is not None and recusa.motivo != "lista_vazia":
            return None
        return recusa

    def _carrega_tabela(self) -> None:
        caminho = self._abre_arquivo(self)
        if caminho is None:
            return
        try:
            cabecalho = le_cabecalho(caminho)
        except (IdentificacaoMalformadaError, OSError) as erro:
            self._mostra_mensagem_livre("entrada", f"A tabela não pôde ser lida: {erro}")
            return
        self._caminho = caminho
        self._tabela = None
        self._colunas = ""
        self.arquivo.setText(caminho.name)
        sugestao = sugere_colunas(cabecalho)
        # O mesmo bloqueio de `_muda_tipo_do_corte`: `clear`, `addItems` e `setCurrentText`
        # emitem `currentIndexChanged` cada um, e cada emissão refaria o recorte. Ele é refeito uma
        # vez, no fim, com as três caixas já preenchidas.
        caixas = (
            (self.coluna_feature, "feature", False),
            (self.coluna_composto, "composto", False),
            (self.coluna_aceito, "aceito", True),
        )
        bloqueios = [QSignalBlocker(combo) for combo, _, _ in caixas]
        for combo, chave, opcional in caixas:
            combo.clear()
            if opcional:
                combo.addItem(_NENHUMA)
            combo.addItems(list(cabecalho))
            alvo = sugestao.get(chave)
            if alvo:
                combo.setCurrentText(alvo)
        for bloqueio in bloqueios:
            bloqueio.unblock()
        religa(self.botao_confirma)
        self._invalida_e_recorta()

    def _confirma_colunas(self) -> None:
        if self._caminho is None:  # pragma: no cover - o botão está desligado sem tabela
            return
        feature = self.coluna_feature.currentText()
        composto = self.coluna_composto.currentText()
        aceito = self.coluna_aceito.currentText()
        try:
            self._tabela = le_identificacao(
                self._caminho,
                coluna_feature=feature,
                coluna_composto=composto,
                coluna_aceito=None if aceito in ("", _NENHUMA) else aceito,
            )
        except (IdentificacaoMalformadaError, OSError) as erro:
            self._tabela = None
            self._colunas = ""
            self._mostra_mensagem_livre("entrada", f"A tabela não pôde ser lida assim: {erro}")
            self._invalida_e_recorta()
            return
        self._colunas = f"feature: {feature}; composto: {composto}; aceito: {aceito or _NENHUMA}"
        self._invalida_e_recorta()

    def _muda_coluna(self) -> None:
        """Uma coluna trocada desfaz a confirmação: a tabela lida deixou de ser a da tela."""
        self._tabela = None
        self._colunas = ""
        self._invalida_e_recorta()

    def _muda_tipo_do_corte(self) -> None:
        # Sem o bloqueio, `setDecimals`, `setRange` e `setValue` emitem `valueChanged` cada um, e o
        # primeiro recorte sairia com `Corte("n_melhores", 0.0)`, que levanta dentro do slot. O
        # recorte é refeito uma vez, no fim, com o valor final.
        bloqueio = QSignalBlocker(self.valor_do_corte)
        if self.tipo_do_corte.currentData() == "n_melhores":
            self.valor_do_corte.setDecimals(0)
            self.valor_do_corte.setRange(1, 10000)
            self.valor_do_corte.setSingleStep(1)
            self.valor_do_corte.setValue(25)
        else:
            self.valor_do_corte.setDecimals(4)
            self.valor_do_corte.setRange(0.0001, 0.5)
            self.valor_do_corte.setSingleStep(0.01)
            self.valor_do_corte.setValue(0.05)
        bloqueio.unblock()
        self._invalida_e_recorta()

    # ------------------------------------------------------------------ recorte
    def _invalida_e_recorta(self) -> None:
        """Avança o token, limpa o resultado, e refaz o recorte com o que houver."""
        self._cancelar.set()
        self._invalida_sem_cancelar()

    def _invalida_sem_cancelar(self) -> None:
        """`_invalida_e_recorta` sem cancelar o cálculo em voo, para a troca de matriz."""
        self._token += 1
        self._resultado = None
        self._envio_em_voo = None
        self._envio_do_resultado = None
        self._fundo_da_biblioteca = False
        self.tabela.setRowCount(0)
        self.rodape.setText("")
        self.estado.setText("")
        desliga_com_motivo(self.botao_exporta, MOTIVO_SEM_RESULTADO)
        self.botao_fundo_biblioteca.setVisible(False)
        self._para_espera()
        self._recorte = None
        volcano = self._volcano
        if volcano is None:
            self.degraus.setText(SEM_EXECUCAO)
            self._libera_o_enriquecer()
            return
        if self._tabela is None:
            self.degraus.setText(
                SEM_TABELA if self._caminho is None else "Confirme as colunas da tabela."
            )
            self._libera_o_enriquecer()
            return
        nomes = [str(x) for x in volcano["features"]]
        pvalores = [float("nan") if v is None else float(v) for v in volcano["pvalue"]]
        corte = en.Corte(self.tipo_do_corte.currentData(), float(self.valor_do_corte.value()))
        linhas = [(ln.feature, ln.composto, ln.aceito) for ln in self._tabela.linhas]
        self._recorte = en.recorta(linhas, nomes, pvalores, corte)
        d = self._recorte.degraus
        # No modo externo não há fundo: dizer 'contra um fundo de N' afirmaria um universo que o
        # servidor não usa.
        contra = (
            f"contra {UNIVERSO_EXTERNO}"
            if self._modo_externo()
            else f"contra um fundo de {len(self._recorte.fundo)}"
        )
        self.degraus.setText(
            f"{_conta(d.linhas_lidas, 'linha lida', 'linhas lidas')} · "
            f"{_conta(d.features_identificadas, 'feature identificada', 'features identificadas')}"
            f" · {_conta(d.compostos_distintos, 'composto')} · "
            f"{_conta(d.features_em_comum, 'feature')} em comum com a "
            f"execução · {d.features_sem_p} sem p calculado"
            + (
                " · "
                + _conta(
                    d.candidatos_descartados, "candidato não escolhido", "candidatos não escolhidos"
                )
                if d.candidatos_descartados
                else ""
            )
            + (
                f" · {_conta(d.compostos_fora_da_execucao, 'composto')} fora da execução"
                if d.compostos_fora_da_execucao
                else ""
            )
            + f"\n\n**{_conta(len(self._recorte.lista), 'composto')} na lista**, "
            f"{contra}; {self._recorte.descricao_do_esperado}."
        )
        recusa = self._recusa_previa(self._recorte)
        if recusa is not None:
            self._mostra_falha(
                Falha(recusa.motivo, recusa.texto, recusa.oferece_fundo_da_biblioteca)
            )
        self._libera_o_enriquecer()

    def _motivo_para_nao_enriquecer(self) -> str | None:
        """Por que 'Enriquecer' não pode correr agora, ou `None` quando pode.

        Um predicado só, lido no recorte e na volta do cálculo: depois de um cálculo com o fundo da
        biblioteca o recorte continua recusado com o fundo do ensaio, e religar o botão ali levaria
        o usuário ao R só para ouvir a mesma recusa. A recusa diz a frase e a ação de `MENSAGENS`,
        as mesmas que o estado mostra abaixo.
        """
        if self._volcano is None:
            return MOTIVO_SEM_EXECUCAO
        if self._tabela is None:
            return MOTIVO_SEM_TABELA if self._caminho is None else MOTIVO_COLUNAS_POR_CONFIRMAR
        if self._recorte is None:
            return MOTIVO_COLUNAS_POR_CONFIRMAR
        recusa = self._recusa_previa(self._recorte)
        if recusa is None:
            return None
        mensagem = MENSAGENS[recusa.motivo]
        return f"{mensagem.frase} {mensagem.acao}"

    def _libera_o_enriquecer(self) -> None:
        """Liga 'Enriquecer', ou o desliga com o motivo de `_motivo_para_nao_enriquecer` ao lado."""
        motivo = self._motivo_para_nao_enriquecer()
        if motivo is None:
            religa(self.botao_enriquece)
        else:
            desliga_com_motivo(self.botao_enriquece, motivo)

    # ------------------------------------------------------------------ o cálculo
    def _enriquece(self, *, fundo_da_biblioteca: bool) -> None:
        if self._recorte is None:  # pragma: no cover - o botão está desligado sem recorte
            return
        motor = self._motor()
        origem = self._origem_das_bibliotecas(motor)
        if isinstance(origem, Falha):
            self._mostra_falha(origem)
            return
        if origem is None:
            if not self._confirma(self, "Baixar as bibliotecas de dados?", TEXTO_DO_DOWNLOAD):
                self._mostra_falha(Falha("download_recusado"))
                return
            self._baixa(motor, fundo_da_biblioteca)
            return
        self._origem = origem
        externo = self._modo_externo()
        # O fundo da biblioteca é opção do modo local; no externo ele não existe.
        self._fundo_da_biblioteca = fundo_da_biblioteca and not externo
        self._envio_em_voo = None
        recorte = self._recorte
        cancelar = threading.Event()
        self._cancelar = cancelar
        # O cabeçalho entra aqui, e não em `recebe_volcano`: é o da tabela do recorte que vai sair.
        proibidos = self._proibidos | vocabulario_proibido(
            {}, self._tabela.cabecalho if self._tabela is not None else ()
        )

        def local(fase: Callable[[str], None]) -> object:
            return executa_enriquecimento(
                recorte,
                motor,
                origem.pasta,
                fundo_da_biblioteca=fundo_da_biblioteca,
                executor=self._executor,
                cancelado=cancelar.is_set,
                fase=fase,
                fria=not self._r_aquecido,
            )

        def etapa_1_do_externo(fase: Callable[[str], None]) -> object:
            return prepara_envio_externo(
                recorte,
                motor,
                origem.pasta,
                proibidos=proibidos,
                executor=self._executor,
                cancelado=cancelar.is_set,
                fase=fase,
                fria=not self._r_aquecido,
            )

        self._comeca_espera()
        tarefa = etapa_1_do_externo if externo else local
        self._cancelamentos.append(cancelar)
        self._registra_o_voo(cancelar)
        self._piscina.start(_Trabalho(tarefa, self._sinais, self._token))

    def _confirma_e_envia(self, envio: EnvioPreparado) -> None:
        """A etapa 2, na thread da tela: o diálogo, e só com 'Enviar' a etapa 3 começa.

        O diálogo é modal e roda um laço de eventos próprio: uma execução nova pode chegar enquanto
        ele está aberto. Por isso o token é conferido depois dele: um 'Enviar' sobre um recorte que
        já não é o da tela não envia nada.
        """
        token = self._token
        if not self._confirma_envio(self, list(envio.nomes)):
            if token == self._token:
                self._mostra_falha(Falha("envio_nao_confirmado"))
            return
        if token != self._token:
            return
        cancelar = threading.Event()
        self._cancelar = cancelar
        self._envio_em_voo = EnvioExterno(envio.nomes, self._relogio_de_parede())

        def etapa_3_do_externo(fase: Callable[[str], None]) -> object:
            return conclui_envio_externo(
                envio,
                enviador=self._enviador,
                cancelado=cancelar.is_set,
                relogio=self._agora,
                fase=fase,
            )

        self._comeca_espera("enviando")
        self._cancelamentos.append(cancelar)
        self._registra_o_voo(cancelar)
        self._piscina.start(_Trabalho(etapa_3_do_externo, self._sinais, self._token))

    def _baixa(self, motor: MotorDoR, fundo_da_biblioteca: bool) -> None:
        cancelar = threading.Event()
        self._cancelar = cancelar
        manifesto = dados_vias.le_manifesto()

        def tarefa(fase: Callable[[str], None]) -> object:
            fase("baixando")
            return dados_vias.baixa_bibliotecas(
                motor.bibliotecas_baixadas, manifesto, confirmado=True, cancelado=cancelar.is_set
            )

        self._depois_do_download = fundo_da_biblioteca
        self._comeca_espera()
        self._cancelamentos.append(cancelar)
        self._piscina.start(_Trabalho(tarefa, self._sinais, self._token))

    def _cancela(self) -> None:
        self._cancelar.set()
        self.estado.setText("Cancelando…")

    def encerra(self, teto_ms: int = 10_000) -> bool:
        """No fechamento: cancela o R em voo (o Job leva a árvore) e só então drena a piscina.

        Cancela todos, e não só o último: uma troca de matriz deixa o R da matriz que saiu a correr,
        e um cálculo novo na que entrou põe outro cancelamento no lugar.
        """
        self._cancelar.set()
        for cancelar in self._cancelamentos:
            cancelar.set()
        self._piscina.clear()
        return bool(self._piscina.waitForDone(teto_ms))

    # ------------------------------------------------------------------ a volta à matriz
    def _entradas(self) -> EntradasDoRecorte:
        return EntradasDoRecorte(
            self._tabela,
            self._colunas,
            (self.tipo_do_corte.currentData(), float(self.valor_do_corte.value())),
            self.biblioteca_de_vias.currentData(),
        )

    def _registra_o_voo(self, cancelar: threading.Event) -> None:
        """O contexto do cálculo que começa, pelo token: o resultado pode chegar noutra tela."""
        self._voos[self._token] = _Voo(
            self._corpo,
            self._entradas(),
            self._origem,
            self._envio_em_voo,
            self._fundo_da_biblioteca,
            cancelar,
        )

    def corpo_na_tela(self) -> Any:
        """O corpo da execução que o painel mostra: o objeto que a janela passou."""
        return self._corpo

    def repoe(self, foto: FotoDoEnriquecimento) -> bool:
        """Pinta um resultado guardado do corpo da tela, na volta à matriz dele. Devolve se pintou.

        Só com as mesmas entradas do recorte: com outra tabela, outro corte ou outra biblioteca na
        tela, o resultado não é o desta tela, e fica o recorte. E não com um cálculo a correr: o
        relógio da espera reescreveria o estado por cima dele a cada segundo, e o resultado do
        cálculo corrente chega a seguir.
        """
        if self._recorte is None or self._relogio.isActive():
            return False
        if not foto.entradas.mesmas(self._entradas()):
            return False
        self._origem = foto.origem
        self._envio_em_voo = foto.envio
        self._fundo_da_biblioteca = foto.fundo_da_biblioteca
        self._mostra_resultado(foto.resultado)
        return True

    # ------------------------------------------------------------------ a espera
    def _comeca_espera(self, fase: str = "mapeando") -> None:
        self._inicio = self._agora()
        self._fase = fase
        desliga_com_motivo(self.botao_enriquece, MOTIVO_EM_CURSO)
        self.botao_fundo_biblioteca.setVisible(False)
        religa(self.botao_cancela)
        self.tabela.setRowCount(0)
        self._mostra_espera()
        self._relogio.start()

    def _para_espera(self) -> None:
        self._relogio.stop()
        desliga_com_motivo(self.botao_cancela, MOTIVO_NADA_A_CANCELAR)

    def _recebe_fase(self, fase: str, token: int) -> None:
        if token == self._token:
            self._fase = fase
            self._mostra_espera()

    def _mostra_espera(self) -> None:
        decorrido = int(self._agora() - self._inicio)
        # O aviso do antivírus é sobre o R desta máquina: na espera do servidor ele não se aplica.
        aviso = "" if self._fase in _FASES_DO_SERVIDOR else f" {AVISO_DA_PRIMEIRA}"
        self.estado.setText(f"**{_FASES.get(self._fase, self._fase)}…** {decorrido} s.{aviso}")

    # ------------------------------------------------------------------ o desfecho
    def _recebe(self, resultado: object, token: int) -> None:
        voo = self._voos.pop(token, None)
        if token != self._token:
            # Resultado de um recorte que já não é o da tela: não pinta. Um resultado da ORA, porém,
            # vai à janela com o corpo de que é, porque a troca de matriz não o cancelou e ele volta
            # com a matriz dele. As etapas intermédias, o download e o envio preparado, não
            # continuam fora da tela.
            if voo is not None and isinstance(resultado, en.ResultadoDaOra):
                self._r_aquecido = True
                self.resultado_concluido.emit(voo.corpo, self._foto(resultado, voo), False)
            return
        self._para_espera()
        # O mesmo predicado de `_invalida_e_recorta`, e não só 'há recorte': ver
        # `_motivo_para_nao_enriquecer`.
        self._libera_o_enriquecer()
        if isinstance(resultado, dados_vias.ResultadoDoDownload):
            if resultado.desfecho == "baixado":
                self._enriquece(fundo_da_biblioteca=self._depois_do_download)
            else:
                self._mostra_falha(
                    Falha(_DESFECHO_DO_DOWNLOAD[resultado.desfecho], resultado.motivo)
                )
            return
        # O fato vem do lançador, gravado na `Falha` por `executa_enriquecimento`, e não da chave
        # da mensagem: ver a docstring de `Falha`.
        if isinstance(resultado, en.ResultadoDaOra | EnvioPreparado) or (
            isinstance(resultado, Falha) and resultado.o_r_respondeu
        ):
            self._r_aquecido = True  # o R respondeu ao menos uma vez neste processo
        if self._cancelar.is_set() and not isinstance(resultado, en.ResultadoDaOra):
            # Na etapa 3 o pedido pode já ter saído, e a mensagem é a do envio, com o detalhe do
            # transporte quando ele chegou a dizer em que ponto estava.
            if self._envio_em_voo is None:
                self._mostra_falha(Falha("cancelado"))
            elif isinstance(resultado, Falha) and resultado.chave == "envio_cancelado":
                self._mostra_falha(resultado)
            else:
                self._mostra_falha(Falha("envio_cancelado"))
            return
        if isinstance(resultado, EnvioPreparado):
            self._confirma_e_envia(resultado)
            return
        if isinstance(resultado, Falha):
            self._mostra_falha(resultado)
            return
        if isinstance(resultado, en.ResultadoDaOra):
            self._mostra_resultado(resultado)
            if voo is not None:
                self.resultado_concluido.emit(voo.corpo, self._foto(resultado, voo), True)
            return
        self._mostra_falha(Falha("motor_falhou", f"{type(resultado).__name__}: {resultado}"))

    @staticmethod
    def _foto(resultado: en.ResultadoDaOra, voo: _Voo) -> FotoDoEnriquecimento:
        return FotoDoEnriquecimento(
            resultado, voo.origem, voo.envio, voo.fundo_da_biblioteca, voo.entradas
        )

    def _mostra_mensagem_livre(self, classe: ClasseDaMensagem, texto: str) -> None:
        self.estado.setText(f"**{_TITULO_DA_CLASSE[classe]}.** {texto}")

    def _mostra_falha(self, falha: Falha) -> None:
        msg = MENSAGENS[falha.chave]
        self.tabela.setRowCount(0)
        self._resultado = None
        self._envio_do_resultado = None
        desliga_com_motivo(self.botao_exporta, MOTIVO_SEM_RESULTADO)
        texto = f"**{_TITULO_DA_CLASSE[msg.classe]}: {msg.frase}** {msg.acao}"
        if falha.detalhe:
            texto += f"\n\n`{falha.detalhe.strip()[:600]}`"
        self.estado.setText(texto)
        # O fundo da biblioteca é opção do modo local: no externo o botão nunca aparece.
        self.botao_fundo_biblioteca.setVisible(
            falha.oferece_fundo_da_biblioteca and not self._modo_externo()
        )

    def _mostra_resultado(self, r: en.ResultadoDaOra) -> None:
        self._resultado = r
        # O envio carimbado é o que produziu este resultado: o em voo vira o do resultado aqui, e um
        # resultado local leva `None`; o CSV decide o cabeçalho por ele, e não pelo seletor.
        envio = self._envio_do_resultado = self._envio_em_voo
        origem = self._origem
        descricao = origem.descricao if origem else ""
        if envio is not None:
            texto = (
                f"**{_conta(len(r.vias), 'via')} com acerto**, {ORA_EXTERNA}. Universo: "
                f"{UNIVERSO_EXTERNO}, {_conta(r.tamanho_do_universo, 'composto')}; "
                f"{_conta(len(envio.nomes), 'nome enviado', 'nomes enviados')}, "
                f"{r.na_lista} em alguma via KEGG. O mapeamento local "
                f"usou as bibliotecas de dados {descricao}.\n\n**{NOTA_DO_MODO_EXTERNO}**"
            )
        else:
            universo = (
                "o fundo do ensaio"
                if r.tipo_do_universo == "fundo_do_ensaio"
                else "a biblioteca inteira"
            )
            texto = (
                f"**{_conta(len(r.vias), 'via')} com acerto**, ORA do MetaboAnalystR sobre a SMPDB "
                f"(vias humanas). Universo: {universo}, {_conta(r.tamanho_do_universo, 'composto')}"
                f"; {_conta(r.na_lista, 'composto')} da lista "
                f"no universo. Bibliotecas de dados {descricao}."
            )
        if self._fundo_da_biblioteca:
            texto += "\n\n" + FUNDO_DA_BIBLIOTECA
        # A lista curta calcula, e o aviso vai junto do resultado: sem ele, uma via com um único
        # acerto numa lista de dois compostos se lê como enriquecimento forte.
        aviso = en.aviso_de_poucos(r.na_lista)
        if aviso:
            texto += f"\n\n**Aviso:** {aviso}"
        self.estado.setText(texto)
        self.tabela.setRowCount(len(r.vias))
        for i, via in enumerate(r.vias):
            celulas = (
                via.via,
                str(via.total),
                decimal(via.esperado, 3),
                str(via.acertos),
                decimal(via.p, 3),
                decimal(via.fdr, 3),
            )
            dica = "Compostos que acertaram: " + ", ".join(via.compostos)
            if self._fundo_da_biblioteca:
                dica += "\n\nFundo não específico do ensaio: tende a falsos positivos."
            if envio is not None:
                dica += f"\n\nUniverso: {UNIVERSO_EXTERNO}; tende a falsos positivos."
            for j, texto_celula in enumerate(celulas):
                item = QTableWidgetItem(texto_celula)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(dica)
                self.tabela.setItem(i, j, item)
        self.rodape.setText(
            "Os compostos de cada via estão na dica da linha e no CSV, e vêm da **identificação "
            "feita por você no Progenesis**: este programa não identifica compostos. O p é o "
            "hipergeométrico da sobre-representação, conferido contra as contagens."
        )
        religa(self.botao_exporta)

    def _exporta(self) -> None:
        if self._resultado is None or self._recorte is None:  # pragma: no cover - botão desligado
            return
        destino = self._salva_arquivo(self)
        if destino is None:
            return
        try:
            escreve_csv(
                destino,
                self._recorte,
                self._resultado,
                self._origem,
                self._fundo_da_biblioteca,
                exportado_em=self._relogio_de_parede(),
                contraste=_contraste(self._volcano),
                colunas=self._colunas,
                externo=self._envio_do_resultado,
            )
        except OSError as erro:
            # Uma exceção que escapa de um slot de `clicked` vai para o stderr, que o executável
            # congelado não mostra. O CSV aberto no Excel e a pasta sem permissão são `OSError`.
            self.estado.setText(
                self.estado.text()
                + f"\n\nNão foi possível gravar `{destino.name}`: {type(erro).__name__}. "
                "Feche o arquivo se ele estiver aberto noutro programa, ou escolha outra pasta."
            )
            return
        self.estado.setText(self.estado.text() + f"\n\nExportado para `{destino.name}`.")

    # ------------------------------------------------------------------ para o teste
    def linhas(self) -> tuple[tuple[str, ...], ...]:
        """O que está visível na tabela, para o teste."""
        saida = []
        for i in range(self.tabela.rowCount()):
            linha = []
            for j in range(self.tabela.columnCount()):
                item = self.tabela.item(i, j)
                linha.append("" if item is None else item.text())
            saida.append(tuple(linha))
        return tuple(saida)


def escreve_csv(
    destino: Path,
    recorte: en.RecorteDaIdentificacao,
    resultado: en.ResultadoDaOra,
    origem: OrigemDasBibliotecas | None,
    fundo_da_biblioteca: bool,
    *,
    exportado_em: datetime,
    contraste: str,
    colunas: str,
    externo: EnvioExterno | None = None,
) -> None:
    """O CSV com o recorte junto. Nenhuma célula começa por caractere de fórmula.

    O cabeçalho diz o contraste, o hash de cada arquivo de dados e as colunas da tabela: sem eles o
    CSV não diria sobre qual par de grupos, que versão das bibliotecas e que coluna a lista foi
    calculada.

    `externo` é o envio que produziu o resultado, ou `None` no modo local, cujo cabeçalho não muda
    com ele. No externo, o motor é o servidor, a biblioteca é a dele, e a linha 'biblioteca do
    servidor' diz o que se sabe da versão: nada, porque o servidor não a informa, só o instante UTC
    da consulta. As bibliotecas de dados locais continuam no CSV, nomeadas como as do mapeamento,
    que é o que elas fizeram neste modo.
    """
    d = recorte.degraus
    por_arquivo = [
        (f"arquivo de dados {a.arquivo}", f"sha256 {a.sha256}; {a.procedencia}")
        for a in (origem.arquivos if origem else ())
    ]
    descricao = (origem.descricao if origem else "").replace("*", "")
    validadas = "sim" if origem and origem.validada else "não"
    if externo is None:
        procedencia = [
            ("motor", "MetaboAnalystR, ORA do MSEA"),
            ("modo", "local"),
            ("biblioteca de vias", resultado.biblioteca),
            ("bibliotecas de dados", descricao),
            ("bibliotecas validadas", validadas),
        ]
        fundo = [
            (
                "fundo não específico do ensaio",
                "sim, tende a falsos positivos" if fundo_da_biblioteca else "não",
            )
        ]
        compostos_no_fundo = str(len(recorte.fundo))
    else:
        consultada = externo.consultado_em.astimezone(UTC).isoformat(timespec="seconds")
        procedencia = [
            ("motor", MOTOR_EXTERNO),
            ("modo", "externo"),
            ("biblioteca de vias", resultado.biblioteca),
            (
                "biblioteca do servidor",
                f"versão não informada pelo servidor; consultada em {consultada}",
            ),
            ("bibliotecas de dados do mapeamento local", descricao),
            ("bibliotecas do mapeamento validadas", validadas),
        ]
        fundo = [
            (
                "fundo não específico do ensaio",
                "sim: o servidor não aplica o fundo do ensaio na ORA KEGG; tende a falsos "
                "positivos",
            ),
            ("nomes enviados", str(len(externo.nomes))),
            # O servidor conta só os que estão em alguma via KEGG, e a ORA usa esses.
            ("nomes enviados em alguma via KEGG", str(resultado.na_lista)),
        ]
        compostos_no_fundo = "não se aplica: o modo externo não envia fundo"
    cabecalho = [
        *procedencia,
        *por_arquivo,
        ("contraste", contraste),
        ("colunas da tabela", colunas),
        ("corte", f"{recorte.corte.tipo} {recorte.corte.valor:g}"),
        ("esperado por acaso", recorte.descricao_do_esperado),
        ("universo", f"{resultado.tipo_do_universo} ({resultado.tamanho_do_universo})"),
        ("aviso", en.aviso_de_poucos(resultado.na_lista) or "nenhum"),
        *fundo,
        ("linhas lidas", str(d.linhas_lidas)),
        ("features em comum com a execução", str(d.features_em_comum)),
        ("compostos na lista", str(len(recorte.lista))),
        ("compostos no fundo", compostos_no_fundo),
        ("exportado em", exportado_em.isoformat(timespec="seconds")),
    ]
    with destino.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for chave, valor in cabecalho:
            w.writerow([neutraliza(chave), neutraliza(valor)])
        w.writerow([])
        w.writerow(["via", "tamanho", "esperado", "acertos", "p", "fdr", "compostos"])
        for via in resultado.vias:
            w.writerow(
                [
                    neutraliza(via.via),
                    via.total,
                    f"{via.esperado:.6g}",
                    via.acertos,
                    f"{via.p:.6g}",
                    f"{via.fdr:.6g}",
                    neutraliza("; ".join(via.compostos)),
                ]
            )

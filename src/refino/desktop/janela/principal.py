"""A `QMainWindow` da aplicação e a composição dos seus painéis.

A janela monta os painéis em abas e trata todo desfecho de rota pelo vocabulário fechado de três
causas de `refino.interface.cliente`. Três escolhas de desenho:

- O transporte entra por injeção. `JanelaPrincipal` recebe um `Transporte`, o `Protocol` de
  `cliente.py` que o `TestClient` do FastAPI satisfaz; assim os testes verificam a janela sem
  subir `uvicorn`.
- Nenhuma mensagem nasce aqui. `causa_de` e `mensagem_de` são as fontes, e a janela só formata. A
  causa vai crua, no formato `[causa: X] mensagem`: o projeto não tem vocabulário de causa na
  língua do usuário, e criar um aqui daria uma segunda fonte de verdade sobre de quem é a causa.
- A chamada de rota sai do laço de eventos. `cliente.envia` é síncrono e uma execução leva
  minutos; chamado do laço, ele congela a janela. Todo `envia` corre num `QThreadPool` próprio e o
  resultado volta por sinal.

`_Trabalho` tem dois sinais. `envia` converte toda `Exception` em `Resposta`, mas uma exceção que
escapa de um `QRunnable` some sem aviso, e a janela ficaria à espera para sempre. `falhou` é esse
segundo desfecho.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QHideEvent, QIcon, QShowEvent
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from refino.api.progresso import TERMINAIS
from refino.desktop.etapas import (
    CHAVE_DA_ETAPA,
    ENTRADAS_DA_ETAPA,
    ETAPA_DA_ENTRADA,
    ETAPAS,
    IDENTIDADE_DA_MATRIZ,
    NOMES_DAS_ETAPAS,
    ROTA_DA_ETAPA,
    Entrada,
    EntradaDaEtapa,
    EntradaIndisponivelError,
    EstadoDasEtapas,
    Fotografia,
    campos_da_rota,
    fotografia_da_etapa,
)
from refino.desktop.janela.arquivos import (
    ModoAusenteError,
    PainelDeArquivos,
    escolhedor_por_dialogo,
)
from refino.desktop.janela.artefatos import (
    ARTEFATO_DO_RELATORIO,
    ARTEFATOS_DA_JANELA,
    CHAVES_DO_RELATORIO,
    PASSOS_DOS_ARTEFATOS,
    ArtefatoDaJanela,
    campos_do_download,
    escreve_o_corte_ao_lado,
    nome_do_arquivo,
    pasta_por_dialogo,
    relatorio_da_resposta,
)
from refino.desktop.janela.avisos import AvisoDeDesfecho
from refino.desktop.janela.componentes import com_maiuscula
from refino.desktop.janela.execucao import (
    FRASE_DA_ENTRADA_DA_CORRECAO,
    MEDINDO_A_SUGESTAO,
    MOTIVOS_SEM_ETAPA,
    PERGUNTAS_DA_ENTRADA,
    TEXTOS_DE_REFAZER,
    TITULO_DO_BLOCO_DO_RESUMO,
    TITULOS_DA_FALHA,
    AbaDeEtapa,
    BlocoDeSugestoes,
    ResultadoDaEtapa,
    VistaDaEntrada,
    aviso_sem_imputacao,
    frase_da_entrada_fora_de_dia,
    frase_da_entrada_unica,
    frase_da_etapa_fora_de_dia,
    frase_da_indicacao,
    frase_das_sugestoes_da_correcao,
    frase_do_que_voce_mudou_durante_o_palpite,
    frase_dos_valores_iniciais,
)
from refino.desktop.janela.graficos import FIGURA_DA_CURVA_DO_LOESS, PainelDeGraficos
from refino.desktop.janela.leitura import (
    CHAVE_DO_PALPITE,
    ROTA_DO_PALPITE,
    ROTAS_DA_LEITURA,
    ROTULOS_DA_LEITURA,
    PainelDeLeitura,
)
from refino.desktop.janela.multiverso import (
    CHAVE_DA_GRAVADA,
    CHAVE_DO_CANCELAMENTO,
    CHAVE_DO_DISPARO,
    CHAVE_DO_HISTORICO,
    CHAVE_DO_MULTIVERSO,
    CHAVE_DO_PROGRESSO,
    ROTA_DO_DISPARO,
    PainelDeMultiverso,
    rota_da_gravada,
    rota_do_cancelamento,
    rota_do_historico,
    rota_do_progresso_do_multiverso,
    rota_do_resultado,
)
from refino.desktop.janela.nomes import NOMES_DOS_CORRETORES
from refino.desktop.janela.pipeline import (
    ABA_DE_LIMITES,
    CAMPOS_POR_ARTEFATO,
    CORRETORES,
    EIXO_DO_CORRETOR,
    EIXO_DOS_LOTES,
    EM_CURSO,
    ETAPA_DA_CHAVE,
    LOTES_QUANDO_VAZIO,
    ROTA_DA_EXECUCAO,
    SEMENTE_DA_JANELA,
    CombinacaoProibidaError,
    FotoDoGrupoDaExecucao,
    PainelDePipeline,
    PropostaAusenteError,
    e_corpo_de_aceite,
    inventario_do_corpo,
    rota_de_cancelamento,
    rota_do_progresso,
    rota_do_relatorio,
)
from refino.desktop.janela.preparo import AbaDePreparo
from refino.desktop.janela.qualidade import PainelDeQualidade
from refino.desktop.janela.recorte import PainelDeRecorte
from refino.desktop.janela.tema import BarraLateral, pagina_rolavel
from refino.desktop.janela.trabalhos import (
    CHAVE_DO_CANCELAMENTO_DE_TRABALHO,
    CHAVE_DOS_TRABALHOS,
    PainelDeTrabalhos,
)
from refino.desktop.janela.vias import PainelDeVias
from refino.interface.cliente import Resposta, Transporte, causa_de, envia, mensagem_de
from refino.interface.decorrido import agora
from refino.interface.fluxo import (
    PONTOS_DE_DECISAO_AUTOMATICA,
    PROPOSTAS_POR_ROTA,
    chave_da_entrada,
)
from refino.interface.referencias import (
    ArquivoExportado,
    ResumoMetodologico,
    resumo_da_execucao,
    resumo_da_exportacao,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from refino.desktop.janela.arquivos import Escolhedor
    from refino.desktop.janela.artefatos import EscolhedorDePasta
    from refino.desktop.janela.enriquecimento import FotoDoEnriquecimento
    from refino.desktop.janela.figuras import EscolhedorDeArquivo
    from refino.pathways.analise import RelatorioDeVias

__all__ = [
    "ABAS",
    "INDICES_DAS_ABAS_DE_ETAPA",
    "INTERVALO_DA_ESPERA_MS",
    "PAINEIS",
    "PAINEL_DAS_ETAPAS",
    "SEM_DESFECHO",
    "SEPARADOR_DO_PEDIDO",
    "TETO_DA_ESPERA_MS",
    "TITULO",
    "Aba",
    "AgrupamentoSemComposerError",
    "JanelaPrincipal",
    "PainelVazio",
    "caminho_do_icone",
    "icone_da_janela",
    "texto_da_resposta",
    "texto_do_desfecho",
]

TITULO: Final = "Refino — Refino quimiométrico de metabolômica LC-MS"


def caminho_do_icone() -> Path | None:
    """Onde está `refino.ico`, ou `None` se não estiver lá.

    Devolve o caminho e não o `QIcon`: um `QIcon` vazio e um carregado são indistinguíveis por
    asserção barata, e um teste sobre a função que devolve `QIcon` só afirmaria que ela não
    levantou.

    Congelado, o PyInstaller expande os `datas` para `sys._MEIPASS` preservando o destino declarado
    no spec (`refino/desktop/recursos`); em código, o recurso está ao lado do pacote. Ler `__file__`
    no congelado daria um caminho dentro do PYZ, que não existe no sistema de arquivos.
    """
    if getattr(sys, "frozen", False):
        raiz = Path(getattr(sys, "_MEIPASS", "")) / "refino" / "desktop" / "recursos"
    else:
        raiz = Path(__file__).resolve().parents[1] / "recursos"
    achado = raiz / "refino.ico"
    return achado if achado.is_file() else None


def icone_da_janela() -> QIcon:
    """O ícone da barra de tarefas, ou um `QIcon` vazio.

    Não levanta: o ícone é cosmético, e recusar abrir porque um `.ico` faltou trocaria um defeito
    invisível por um bloqueio. O `QIcon()` vazio é o que o Windows já desenharia.
    """
    achado = caminho_do_icone()
    return QIcon(str(achado)) if achado else QIcon()


# Texto do rótulo antes de qualquer chamada; não é mensagem de desfecho. Um valor inicial distinto
# separa "a janela não mostrou erro" de "a janela mostrou string vazia".
SEM_DESFECHO: Final = "—"

# ---------------------------------------------------------------------------
# A espera pelo artefato
# ---------------------------------------------------------------------------
#
# O corpo do `202` de `GET /trabalhos/{id}` é `ProgressoResposta`, que traz `identificador`
# obrigatório, então `_encaminha` volta a ler a cada resposta; o intervalo estrangula esse laço. Com
# um segundo, o progresso atualiza a cada segundo, e uma leitura custa menos de 0,08 s. A constante
# é uma só para artefatos, medição e execução: um intervalo por chave seria uma segunda fonte sobre
# de quanto em quanto tempo a janela consulta.
INTERVALO_DA_ESPERA_MS: Final = 1_000

# `None` quer dizer que a janela não desiste de esperar. Um teto só fazia a janela parar de
# perguntar enquanto o trabalho continuava no servidor, e a tela dizia "parei de esperar" sobre um
# trabalho vivo. Esperar muito não perde o resultado: a poda de trabalhos de `rotas.py` só alcança
# estados terminais. `None` e não zero, porque zero teria de querer dizer "desiste já" e "nunca
# desiste". O mecanismo fica, opt-in, porque os testes precisam dele: um duplo que nunca conclui
# faria a espera girar para sempre.
TETO_DA_ESPERA_MS: Final[int | None] = None

# Os desfechos do lote: concluir, ser recusado por já haver outro em voo, ser cancelado no diálogo
# da pasta, abortar numa perna ou não conseguir escrever. Cada um pede uma ação diferente de quem
# lê, então cada um tem a sua frase.
LOTE_EM_CURSO: Final = "Já há um lote a correr; espere ou feche a janela"
# Uma correção por vez: duas ao mesmo tempo disputam os mesmos núcleos. Esta é a frase de quando a
# outra correção em curso não é o lote.
CORRECAO_EM_CURSO: Final = (
    "Já há uma correção a correr — espere ela terminar, ou cancele-a, antes de disparar outra: "
    "duas ao mesmo tempo disputam os mesmos núcleos"
)
# O multiverso também entra na trava: ele corrige a matriz com os corretores da grade, nos mesmos
# núcleos, e parar o SERRF em vários núcleos encerra os processos de todo trabalho do programa. A
# frase é própria porque o painel só sabe que o multiverso acabou quando alguém lê o progresso, e a
# ação que ela pede inclui essa leitura.
MULTIVERSO_EM_CURSO: Final = (
    "O teste de robustez está a correr — espere ele terminar, ou cancele-o, antes de disparar uma "
    "correção: os dois disputam os mesmos núcleos, e parar um pode derrubar o outro. Se ele já "
    "terminou, «Ler progresso», na aba do teste de robustez, atualiza o estado."
)
LOTE_CANCELADO: Final = "Lote cancelado: nenhuma pasta escolhida"
# O Palpite medido também entra na trava: ele corre os três corretores, o SERRF entre eles, nos
# mesmos núcleos. A frase é própria porque o que corre não é uma correção, e a ação pedida é
# esperar, e não cancelar.
PALPITE_EM_CURSO: Final = (
    "O Palpite está a ser medido — espere ele terminar, perto de um minuto com vários núcleos "
    "numa corrida de 7 QCs, e mais com mais QCs, antes de disparar uma correção: os dois "
    "disputariam os mesmos núcleos"
)
# As sugestões de 'Seguir sugestões do app' que não entram porque a entrada mudou enquanto o
# Palpite media.
ENTRADA_MUDOU_DURANTE_O_PALPITE: Final = (
    "O Palpite chegou, mas os arquivos, o modo ou os lotes que o «Diagnóstico» sugere mudaram "
    "enquanto ele media: o corretor sugerido não entrou — clique de novo, e o Palpite destes é "
    "medido antes."
)
LOTE_CONCLUIDO: Final = "Lote concluído: {quantos} arquivos em {pasta} — {nomes}"
# O fim e o aborto do lote nomeiam os CSV que ficaram sem o `-removidas.csv`; um relatório que
# volte depois de o lote acabar diz o arquivo na própria frase, no rótulo e no aviso.
LOTE_SEM_O_CORTE: Final = "{frase} O registro do corte NÃO foi salvo ao lado de: {nomes}."
CORTE_DA_PERNA_SEM_ESCRITA: Final = "{nome} — {falha}"
TITULO_DO_LOTE: Final = "Gerar os três"
# Título do aviso das frases do Palpite. O aviso flutuante abre por cima de todas as abas, e com o
# título genérico 'Mensagem do estudo' a pessoa não saberia que a frase é do Palpite, que corre na
# aba 'Preparar o estudo'.
TITULO_DO_PALPITE: Final = "Palpite"

# A cadeia das leituras: Validação, Diagnóstico e Palpite, nesta ordem, que é a única. Nenhuma é
# obrigatória para usar o app, mas pedir uma roda antes as anteriores que não estão em dia para a
# matriz da tela (`JanelaPrincipal._pede_a_cadeia`). Nenhuma correção, normalização ou análise pede
# leitura: a cadeia só existe quando alguém pede o Diagnóstico ou o Palpite.
ORDEM_DAS_LEITURAS: Final = ("validacao", "diagnostico", CHAVE_DO_PALPITE)
# O que cada etapa faz, na frase 'Etapa 2 de 3: diagnosticando o sinal' do cartão pedido.
PASSOS_DA_CADEIA: Final[dict[str, str]] = {
    "validacao": "validando os arquivos",
    "diagnostico": "diagnosticando o sinal",
    CHAVE_DO_PALPITE: "medindo o Palpite",
}
# A espera de 'Seguir sugestões do app' segue as mesmas etapas: antes do Palpite e durante ele.
ESPERA_DAS_SUGESTOES_ANTES_DO_PALPITE: Final = (
    "{etapa}. Depois vem o Palpite, perto de um minuto; os campos são preenchidos quando ele "
    "chegar, menos o que você mudar até lá."
)
ESPERA_DAS_SUGESTOES_NO_PALPITE: Final = (
    "{etapa} — os três corretores em 200 features, perto de um minuto; os campos são preenchidos "
    "quando ele chegar, menos o que você mudar até lá."
)
# As quatro paradas da cadeia, cada uma com a frase inteira (em 'Detalhes' do cartão pedido e no
# aviso) e a manchete curta que o cartão mostra à vista. Arquivos recusados, uma leitura que não
# completou, a matriz que mudou e o Palpite que não pôde sair pedem ações diferentes de quem lê.
CADEIA_VALIDACAO_RECUSADA: Final = (
    "A «Validação» recusou estes arquivos, e o app parou antes do «{alvo}»: o «Diagnóstico» e o "
    "«Palpite» só rodam sobre arquivos aceitos. Os problemas estão no cartão «Validação» — "
    "corrija-os e peça de novo."
)
MANCHETE_DA_VALIDACAO_RECUSADA: Final = "A Validação recusou os arquivos — veja o cartão dela"
CADEIA_LEITURA_NAO_COMPLETOU: Final = (
    "A leitura «{anterior}» não completou, e o app parou antes do «{alvo}»: {motivo}"
)
MANCHETE_DA_LEITURA_QUE_NAO_COMPLETOU: Final = "A leitura «{anterior}» não completou"
# A frase do cartão da própria leitura que não completou quando ele também pediu a cadeia: o
# Diagnóstico de uma cadeia estendida ao Palpite cuja chamada morreu.
LEITURA_DA_CADEIA_NAO_COMPLETOU: Final = "A leitura «{anterior}» não completou: {motivo}"
CADEIA_MATRIZ_MUDOU: Final = (
    "Os arquivos ou o modo mudaram enquanto as leituras corriam, e o «{alvo}» não foi concluído: "
    "peça de novo para os arquivos da tela."
)
MANCHETE_DA_MATRIZ_QUE_MUDOU: Final = "Os arquivos ou o modo mudaram — peça de novo"
CADEIA_PALPITE_NAO_SAIU: Final = "O «Palpite» não pôde ser pedido agora: {motivo}"
MANCHETE_DO_PALPITE_QUE_NAO_SAIU: Final = "O Palpite não pôde ser pedido agora"


def frase_da_etapa(chave: str, alvo: str) -> str:
    """'Etapa N de M: ...' da leitura `chave` numa cadeia que vai até `alvo`.

    O número é a posição em `ORDEM_DAS_LEITURAS`, e não a contagem do que de fato corre: com a
    Validação em dia, o Diagnóstico pedido é 'Etapa 2 de 2'. Uma etapa já em dia não aparece,
    porque não corre.
    """
    total = ORDEM_DAS_LEITURAS.index(alvo) + 1
    return f"Etapa {ORDEM_DAS_LEITURAS.index(chave) + 1} de {total}: {PASSOS_DA_CADEIA[chave]}"


LOTE_ABORTADO: Final = (
    "Lote ABORTADO na perna `{metodo}`: ela não produziu artefato. Os arquivos já escritos ficam "
    "na pasta; os seguintes não foram gerados"
)
LOTE_SEM_ESCRITA: Final = "Lote ABORTADO: não consegui escrever {alvo} — {causa}"
# A perna recusada na janela antes de chegar ao servidor: nada falhou, a perna nem saiu. O motivo é
# o da recusa (outra correção a correr, combinação proibida), e é ele que diz o que fazer antes de
# repetir o lote.
LOTE_RECUSADO: Final = (
    "Lote ABORTADO antes da perna `{metodo}`: {motivo}. Os arquivos já escritos ficam na pasta; "
    "os seguintes não foram gerados"
)

# Os desfechos do gesto unificado: cancelar o diálogo, não conseguir escrever e ter escrito pedem
# ações diferentes. O do sucesso nomeia o caminho e não só a pasta, porque o nome do arquivo é
# derivado e a pessoa não o escolheu.
ARTEFATO_ESCRITO: Final = "Pronto: {alvo}"
ARTEFATO_SEM_ESCRITA: Final = "O artefato ficou pronto, mas não consegui escrever {alvo} — {causa}"
# O resumo dos métodos ao lado do arquivo. A falha dele não desfaz o arquivo: o CSV já está em
# disco quando o `.txt` é escrito, e dizer 'não foi possível' sobre os dois mandaria a pessoa
# procurar o que está lá.
SUFIXO_DO_RESUMO: Final = "-metodos.txt"
RESUMO_ESCRITO: Final = "Salvo em {alvo}, com o resumo dos métodos ao lado ({resumo})."
RESUMO_SEM_ESCRITA: Final = (
    "Salvo em {alvo}; o resumo dos métodos NÃO foi salvo ao lado — {causa}. Copie-o ou salve-o no "
    f"bloco «{TITULO_DO_BLOCO_DO_RESUMO}», abaixo."
)
SO_O_ARQUIVO_ESCRITO: Final = "Salvo em {alvo}."
LOTE_COM_RESUMO: Final = "{frase}. O resumo dos métodos dos três está ao lado deles: {resumo}."
LOTE_SEM_RESUMO: Final = (
    "{frase}. O resumo dos métodos NÃO foi salvo ao lado deles — {causa}. Copie-o ou salve-o no "
    f"bloco «{TITULO_DO_BLOCO_DO_RESUMO}», abaixo."
)


def _com_os_cortes_que_faltam(lote: LoteDeExportacao, frase: str) -> str:
    """A frase de fim do lote, com os CSV que ficaram sem o registro do corte.

    Sem nenhum, devolve a frase como veio: o lote que gravou os três irmãos não ganha uma oração a
    dizer que nada faltou.
    """
    if not lote.sem_o_corte:
        return frase
    # As frases de fim acabam com e sem ponto (o resumo põe um, o aborto não), e a oração nova
    # começa depois de um só.
    fechada = frase if frase.endswith(".") else f"{frase}."
    return LOTE_SEM_O_CORTE.format(frase=fechada, nomes=", ".join(lote.sem_o_corte))


def _resumo_ao_lado(arquivo: Path) -> Path:
    """O `.txt` do resumo dos métodos ao lado de `arquivo`: `<nome>-metodos.txt`, na mesma pasta.

    Derivado do caminho escrito, como o irmão do corte (`nome_do_arquivo_irmao`): quem leva o CSV
    para outra pasta acha o resumo pelo nome dele.
    """
    return arquivo.with_name(f"{arquivo.stem}{SUFIXO_DO_RESUMO}")


@dataclass(frozen=True)
class DestinoPendente:
    """Onde escrever quando os bytes chegarem, capturado no instante do clique.

    Os campos são capturados juntos e nenhum é relido depois. Entre o clique e a chegada passam-se
    minutos de SERRF, e nesse intervalo a pessoa pode trocar o modo ou o corretor; reler o combo na
    chegada poria no nome do arquivo o corretor de agora sobre bytes produzidos pelo de antes.

    `frozen=True`, ao contrário de `LoteDeExportacao`, porque aqui não há estado a consumir: o
    destino é escrito uma vez e descartado.

    `metodo` é o corretor que produz os bytes: no download do Resultado de uma etapa, o da etapa
    feita, e não o do combo. `com_resumo` é a caixa 'Salvar também o resumo dos métodos', lida no
    clique pela mesma razão.
    """

    pasta: Path
    modo: str
    metodo: str
    com_resumo: bool = False


@dataclass
class LoteDeExportacao:
    """O estado de um lote dos três corretores em voo.

    Mutável, ao contrário das outras deste módulo: `restantes` é consumido por `pop(0)` a cada perna
    e `escritos` cresce; uma versão congelada obrigaria a recriar o objeto a cada perna, e o estado
    viveria em dois lugares durante a troca.

    `corretor_original` existe para o combo voltar ao que a pessoa tinha escolhido: o lote move-o
    para mostrar qual perna corre, e deixá-lo no `serrf` ao fim faria a próxima geração avulsa
    correr um corretor que ninguém pediu.
    """

    pasta: Path
    modo: str
    restantes: list[str]
    escritos: list[Path]
    corretor_original: str
    # A chave do artefato que este lote percorre. A condição de roteamento e o nome do arquivo leem
    # esta chave; com um literal em cada, dois lotes diferentes escreveriam os mesmos nomes na mesma
    # pasta.
    chave: str
    # A caixa 'Salvar também o resumo dos métodos' do Resultado, lida no clique: o resumo dos três
    # vai ao lado deles quando o lote termina.
    com_resumo: bool = False
    # Os CSV cujo registro do corte não se gravou. A frase da falha vai ao rótulo do Resultado na
    # hora, e a leitura de progresso da perna seguinte a reescreve; a frase de fim de lote é a única
    # que fica, e é ela que os nomeia.
    sem_o_corte: list[str] = field(default_factory=list)
    # Os pedidos de relatório das pernas que escreveram o CSV e cujo relatório ainda não voltou. O
    # lote só diz os registros em falta quando este conjunto esvazia, que é o último evento dele.
    relatorios_pendentes: set[str] = field(default_factory=set)


# O separador da chave composta. O `200` de um artefato não tem identificador: `cliente.envia` lê o
# corpo pelo `content-type`, e um artefato vem como `text/csv`, com os bytes em `conteudo` e `corpo`
# em `None`. Sem a chave composta, o ramo terminal de `_encaminha` saberia que bytes chegaram mas
# não de qual trabalho, e gravaria os de uma cadeia velha sob a identidade corrente do painel.
#
# As leituras encadeadas usam `<chave>@<identificador>`; `_recebe` separa uma vez, e daí para baixo
# tudo vê a chave base. `partition` corta na primeira ocorrência, então o que não pode conter o
# separador é a chave, e um teste o afirma sobre `ARTEFATOS` e `ROTAS_DA_LEITURA`. Um identificador
# com `@` truncaria a comparação, e o desfecho seria descartar, que é o lado seguro.
SEPARADOR_DO_PEDIDO: Final = "@"


def parte_a_chave(chave: str) -> tuple[str, str | None]:
    """`(chave base, identificador pedido)`; a única fonte da separação.

    Função de módulo, para que o teste do formato corra sem construir `QApplication`.

    O segundo é `None` e não `""`: `None` quer dizer que a resposta não é de uma leitura encadeada
    (é o `POST`, ou uma chamada de arnês), e o ramo terminal entrega. `""` misturaria esse caso com
    o de um identificador vazio.
    """
    base, _, pedido = chave.partition(SEPARADOR_DO_PEDIDO)
    return base, pedido or None


def chave_do_pedido(chave: str, identificador: str) -> str:
    """A chave composta de uma leitura encadeada; o par de `parte_a_chave`.

    Existe para que o formato não se repita numa `f`-string em `_rele` e em `_espera_pelo_artefato`.
    """
    return f"{chave}{SEPARADOR_DO_PEDIDO}{identificador}"


@dataclass
class _EsperaDoArtefato:
    """Um artefato pelo qual a janela está a esperar; o campo que muda é a contagem de leituras.

    `identificador` é o do trabalho, e compará-lo distingue uma leitura da espera que já corre do
    `202` de um disparo novo. As duas chegam a `_encaminha` com a mesma chave e o mesmo código, e
    sem a comparação a primeira leitura de cada disparo esperaria o intervalo sem razão.
    """

    identificador: str
    leituras: int
    inicio: float


@dataclass
class _CadeiaDasLeituras:
    """A cadeia Validação, Diagnóstico e Palpite em curso.

    - `alvo` é a leitura pedida, e só sobe: o Palpite pedido com o Diagnóstico a correr leva a
      mesma cadeia até ele, em vez de abrir uma segunda que repetiria as mesmas leituras.
    - `etiqueta` é a da matriz e da geração dos anexos no pedido (`_etiqueta_da_leitura`). Ela
      descarta as leituras em voo de outra entrada; com outra etiqueta na tela, a resposta que a
      cadeia espera nunca chega, e por isso a cadeia para (`_confere_a_etiqueta_da_cadeia`).
    - `em_voo` é a leitura que a cadeia disparou e espera, ou `None` antes do primeiro disparo. A
      cadeia só avança com uma resposta dessa leitura. Uma Validação pedida à parte, da mesma
      entrada, que chegue antes da da cadeia conta como a dela, por ser a mesma leitura dos mesmos
      arquivos; a segunda, quando chegar, só atualiza o cartão.
    - `anteriores` são os alvos que a cadeia tinha antes de ser estendida, na ordem. O cartão de
      cada um pediu a cadeia e mostra a etapa dela; quando ela para antes da leitura dele, ele
      recebe a parada também, senão ficaria em 'Em curso' sem nada a correr.
    """

    alvo: str
    etiqueta: str
    em_voo: str | None = None
    anteriores: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ExecucaoGuardada:
    """O último desfecho da execução de uma matriz e o que a tela escreveu com ele.

    Guarda a resposta inteira, a recusa incluída: com a matriz na tela, um erro tira as figuras e
    limpa a aba de Qualidade, e é isso que a volta àquela matriz repõe.

    - `resumo` é o resumo metodológico que a caixa de 'Qualidade do resultado' mostrava para esta
      matriz depois deste desfecho: o deste `200`, ou, num erro, o que estava guardado com o
      desfecho anterior dela. Um erro não substitui o resumo do último arquivo que existe.
    - `grupo` é o que o grupo da execução, na seção da Análise, escreveu com este desfecho.
    - `ordem_do_sucesso` é a ordem de chegada do último `200` desta matriz, `-1` sem nenhum: uma
      prévia de etapa mais nova que ele escreveu por cima dos dados das figuras, e um erro depois
      dela não os apaga (`graficos._da_execucao`). É contra ela que a volta repõe a prévia.
    - `escreve` são as etapas guardadas cujas figuras este desfecho escreveu ao chegar, as que a
      Análise dele reproduzia (`JanelaPrincipal._etapas_que_a_execucao_escreve`). Cada etapa é dona
      dos dados das próprias figuras, e a volta escreve nelas o que o desfecho escreveu, e não o que
      o estado das etapas diria na hora da troca, que já não descreve a Análise dele.
    """

    resposta: Resposta
    resumo: ResumoMetodologico | None
    grupo: FotoDoGrupoDaExecucao
    ordem_do_sucesso: int
    escreve: frozenset[int]


@dataclass(frozen=True)
class _PreviaGuardada:
    """A última prévia de uma etapa do passo a passo, a matriz do disparo dela e a ordem de chegada.

    Há uma por etapa, como os dados das figuras que ela escreve: é o que a volta à matriz dela repõe
    por cima de uma execução guardada mais velha.

    `entrada` é a da etapa feita, e é ela que diz a `graficos.recebe_etapa` se a Normalização
    escreve também nas figuras da Correção, o que só acontece quando ela partiu da Correção
    guardada. Sem ela, a prévia não se repõe como chegou.
    """

    etapa: int
    corpo: dict[str, Any]
    matriz: str | None
    ordem: int
    entrada: Entrada


@dataclass(frozen=True)
class PainelVazio:
    """Um painel da composição: a chave, o título e a classe que o preenche.

    `unidade` é o nome da classe do painel. Ela só aparece na marca de um painel ainda não
    composto, e um teste confere que nenhum painel fica sem ela.
    """

    chave: str
    titulo: str
    unidade: str


# Os painéis da janela, na ordem do desenho. `PAINEIS` é o registro dos painéis (quem existe, quem o
# preenche e como se chama); `ABAS` diz quais deles partilham uma aba e com que título.
#
# Os títulos não são numerados porque o servidor não impõe sequência entre os painéis: o único
# pré-requisito duro de `api/rotas.py` é o par de arquivos mais o modo, e as três leituras são
# independentes entre si. As dependências reais são a recomendação antes de sobrescrever um eixo, e
# o diagnóstico para o número de lotes e para os grupos contrastáveis.
PAINEIS: Final[tuple[PainelVazio, ...]] = (
    PainelVazio("arquivos", "Os seus arquivos", "PainelDeArquivos"),
    PainelVazio("leitura", "Validação, diagnóstico e palpite", "PainelDeLeitura"),
    PainelVazio(CHAVE_DOS_TRABALHOS, "Trabalhos em curso", "PainelDeTrabalhos"),
    PainelVazio("pipeline", "Pipeline com parâmetros explícitos", "PainelDePipeline"),
    PainelVazio(
        CHAVE_DO_MULTIVERSO, "Teste de robustez — o que sobrevive à grade", "PainelDeMultiverso"
    ),
    # Os downloads não têm painel próprio: moram no Resultado da Correção e no da Normalização, cada
    # um com o arquivo da etapa feita nos dois formatos e o resumo dos métodos ao lado. A classe
    # `PainelDeArtefatos` continua, como base de `execucao.ResultadoDaEtapa`.
    PainelVazio(
        "qualidade", "Qualidade — o conjunto completo, nunca um número", "PainelDeQualidade"
    ),
    PainelVazio("graficos", "Gráficos — as figuras do resultado", "PainelDeGraficos"),
    # Este painel abre escopo em vez de preencher um espaço declarado; o caminho local do
    # enriquecimento de vias cabe no executável.
    PainelVazio("vias", "Vias metabólicas — predição a partir dos picos", "PainelDeVias"),
    PainelVazio("recorte", ABA_DE_LIMITES, "PainelDeRecorte"),
)

# O painel cujos grupos as três abas de etapa compõem. O painel do pipeline não é desenhado inteiro
# em aba nenhuma: cada grupo dele vai à página da etapa a que pertence (`_compoe_a_aba`), e o widget
# do painel fica como o controlador que os criou. É por esta constante que a bijeção entre painéis e
# abas o acha nas abas de etapa, em vez de o declarar fora delas.
PAINEL_DAS_ETAPAS: Final = "pipeline"


class AgrupamentoSemComposerError(RuntimeError):
    """`ABAS` declara um agrupamento que nenhum composer sabe montar.

    Erro e não recuo: o recuo natural mostraria o primeiro painel do agrupamento e deixaria os
    outros construídos, com pai e fora do layout, e a aba mostraria menos do que a janela declara.
    """


@dataclass(frozen=True)
class Aba:
    """Uma aba da janela e os painéis que ela mostra.

    `ABAS` é uma camada por cima de `PAINEIS`, que continua a ser o registro dos painéis. A chave de
    painel é o eixo dos portões de `tests/test_desktop_recorte.py`, e o painel do recorte deriva
    dela o que declara ao usuário; fundir chaves para juntar painéis numa aba mexeria nesse eixo sem
    portão que o cubra. A composição itera `ABAS`, e a derivação do recorte continua a iterar
    `PAINEIS`.

    `paineis` é uma tupla porque uma aba pode mostrar dois painéis lado a lado.
    """

    titulo: str
    paineis: tuple[str, ...]
    etapa: int | None = None

    @property
    def nome_da_captura(self) -> str:
        """O pedaço do nome de arquivo da captura desta aba.

        As abas de etapa não têm painel, e nomeá-las pelos painéis gravaria `1-.png`, `2-.png` e
        `3-.png`. A aba de etapa é nomeada pela etapa, e as outras pelos painéis.
        """
        return "-".join(self.paineis) if self.paineis else f"etapa-{self.etapa}"


# As abas da janela, agrupadas por função e não numeradas.
#
# Cada peça mora onde é usada: os downloads no Resultado da Correção e da Normalização, os gráficos
# de cada etapa na aba dela, a justificativa das propostas num quadro que o '?' ao lado da sugestão
# abre, o resumo da execução em 'Qualidade do resultado' e as referências dos métodos na última aba.
# A Análise monta o próprio `/execucoes` a partir da entrada escolhida.
#
# `trabalhos` não está aqui: é a faixa fixa da janela, visível em qualquer aba (no rodapé da janela
# que `__init__` monta, e no topo da que o aplicativo abre, para onde
# `tema._monta_faixa_de_trabalhos` a move). Ele continua em `PAINEIS` e em `_preenchidos` porque
# continua a ser um painel composto; o que muda é onde ele é desenhado.
#
# 'Gráficos' fica ao lado da leitura de qualidade, e a aba de honestidade é a última por desenho.
#
# As abas com ' (beta)' ficam juntas, logo antes da última: a barra lateral as põe na gaveta
# 'Trabalhos futuros'.
#
# 'Multiverso (beta)' e 'Vias metabólicas (beta)' ficam no executável, e cada painel abre com a
# linha `FUNCIONALIDADE_EM_BETA`, de `desktop/recorte.py`, que as declara fora da entrega. Os
# títulos dos dois painéis em `PAINEIS` não levam o '(beta)': numa aba de um painel só a barra
# mostra o título da aba, e o do painel só chega à tela pela marca de painel não composto de
# `_compoe_a_aba`.
ABAS: Final[tuple[Aba, ...]] = (
    Aba("Preparar o estudo", ("arquivos", "leitura")),
    Aba("Correção", (), etapa=1),
    Aba("Normalização", (), etapa=2),
    Aba("Análise estatística", (), etapa=3),
    Aba("Qualidade do resultado", ("qualidade",)),
    Aba("Gráficos", ("graficos",)),
    Aba("Teste de robustez (beta)", (CHAVE_DO_MULTIVERSO,)),
    Aba("Vias metabólicas (beta)", ("vias",)),
    Aba(ABA_DE_LIMITES, ("recorte",)),
)

# As páginas de um painel só que rolam na vertical. Ver `_compoe_a_aba`.
PAINEIS_QUE_ROLAM: Final = frozenset({CHAVE_DO_MULTIVERSO, "vias"})

# A aba para onde 'Exportar esta figura em Gráficos', sob a vista de cada aba de etapa, leva.
# Derivada de `ABAS`, e não digitada, para seguir uma reordenação das abas.
INDICE_DA_ABA_DOS_GRAFICOS: Final = next(
    indice for indice, aba in enumerate(ABAS) if "graficos" in aba.paineis
)
# A aba para onde o 'Próximo passo' da Análise leva com 'Ver a qualidade do resultado'. Derivada de
# `ABAS` pela mesma razão da dos gráficos.
INDICE_DA_ABA_DA_QUALIDADE: Final = next(
    indice for indice, aba in enumerate(ABAS) if "qualidade" in aba.paineis
)

INDICES_DAS_ABAS_DE_ETAPA: Final = {
    aba.etapa: indice for indice, aba in enumerate(ABAS) if aba.etapa is not None
}


def texto_da_resposta(resposta: Resposta) -> str:
    """A linha que a janela mostra quando há uma `Resposta`; é esta que o fluxo usa.

    Formata os campos que `cliente.envia` já calculou, sem rederivá-los de `(código, corpo)`: as
    chaves `demora` e `rede` do `503` não vêm de resposta HTTP nenhuma, e `cliente.envia` as escolhe
    direto. Rederivar faria uma `ConnectionRefusedError` aparecer como 'o banco de dados não está de
    pé', mandando a pessoa subir um serviço que está saudável.
    """
    return f"[causa: {resposta.causa}] {resposta.mensagem}"


def texto_do_desfecho(codigo: int, corpo: Any = None) -> str:
    """A linha derivada de `(código, corpo)`, para quando não há `Resposta`.

    Função de módulo: não usa estado da janela, e o teste de que nenhuma mensagem nasce aqui corre
    sem construir `QApplication`.

    Não a use sobre uma `Resposta`; use `texto_da_resposta`. Esta perde em silêncio a distinção
    entre as três chaves de `503`, com uma frase plausível e errada.
    """
    return f"[causa: {causa_de(codigo, corpo)}] {mensagem_de(codigo, corpo)}"


class _Sinais(QObject):
    """O portador dos sinais; `QRunnable` não é `QObject` e não pode declará-los.

    Os dois carregam a chave da chamada. Sem ela, dois painéis ligados ao mesmo sinal não saberiam
    de quem é a resposta, e seria preciso um sinal por painel, uma segunda fonte de verdade sobre
    'uma chamada terminou'.
    """

    concluido = Signal(str, object)
    falhou = Signal(str, object)


def despacha(chamada: Callable[[], Resposta], sinais: _Sinais, chave: str) -> None:
    """A lógica de `_Trabalho.run`, numa função chamável da thread principal.

    O `coverage` não vê a thread do `QThreadPool`: ela é nativa e não passa pelo módulo
    `threading`, e o corpo dentro de `run` apareceria como não coberto mesmo com os testes a afirmar
    os dois sinais. `pragma: no cover` esconderia lógica que corre; aqui ela fica medida, e `run`
    fica com uma linha de despacho.

    Os dois desfechos são exercitáveis pelas duas vias: por esta função, na thread principal, com a
    cobertura a ver; e pelo caminho real, pela piscina, com `qtbot.waitSignal`.
    """
    try:
        resposta = chamada()
    except BaseException as erro:
        sinais.falhou.emit(chave, erro)
        return
    sinais.concluido.emit(chave, resposta)


class _Trabalho(QRunnable):
    """Uma chamada de rota fora do laço de eventos.

    Recebe a chamada já fechada em vez dos argumentos dela, para que o teste injete uma chamada
    lenta ou uma que levanta sem que esta classe saiba o que é uma rota. `run` tem uma linha só,
    porque o `coverage` não vê o que corre nesta thread (ver `despacha`).
    """

    def __init__(self, chamada: Callable[[], Resposta], chave: str) -> None:
        super().__init__()
        self.sinais = _Sinais()
        self._chamada = chamada
        self._chave = chave

    def run(self) -> None:
        # Chamar `run()` na thread principal instrumenta esta linha; a lógica vive em `despacha`
        # porque o que a piscina executa fica fora do alcance do `coverage`.
        despacha(self._chamada, self.sinais, self._chave)


# A chave e a rota da medição. A chave aparece na ligação, no disparo, no ramo de `_encaminha` e no
# despacho de mostrador, e uma constante evita quatro literais que divergem. A rota não entra em
# `ARTEFATOS`: aquelas devolvem `text/csv` e esta devolve JSON, e o ramo dos artefatos tentaria
# servir bytes de um corpo que não os tem.
CHAVE_DA_MEDICAO: Final = "medicao"

# A chave da execução, nomeada para que `parte_a_chave`, o agendamento e a sondagem automática
# falem da mesma string.
CHAVE_DA_EXECUCAO: Final = "execucao"
# As chaves de espera cujo trabalho corrige a matriz ou ocupa os mesmos núcleos, para a trava de uma
# correção por vez. A trava vale nos dois sentidos: valendo numa direção só, a outra disputaria os
# núcleos.
#
# Os artefatos são derivados de `CAMPOS_POR_ARTEFATO`, o mapa de `artefatos.py` com os downloads do
# Resultado da Correção e da Normalização, para que um artefato novo entre sozinho; as três rotas
# deles passam por `_corrige_reusando`. As prévias do passo a passo entram derivadas de
# `CHAVE_DA_ETAPA` pela mesma razão. A medição dos corretores e o Palpite medido não gravam no cache
# de correção, mas correm os três corretores, o SERRF entre eles, nos mesmos núcleos. As leituras
# ficam de fora.
#
# O multiverso não entra por este conjunto: ele nunca é uma espera (o progresso dele é lido a
# pedido), então quem sabe se ele corre é o painel, e `_motivo_para_nao_corrigir` pergunta-lhe.
CHAVES_QUE_CORRIGEM: Final = frozenset(
    {
        CHAVE_DA_EXECUCAO,
        CHAVE_DA_MEDICAO,
        CHAVE_DO_PALPITE,
        *CAMPOS_POR_ARTEFATO,
        *CHAVE_DA_ETAPA.values(),
    }
)
# O intervalo em que a janela recalcula o estado das etapas do passo a passo.
INTERVALO_DAS_ETAPAS_MS: Final = 500
ROTA_DA_MEDICAO: Final = "/matrizes/medir-corretor"

# A chave e a rota da curva do LOESS de uma feature, constantes pelo motivo das da medição. Ficam
# fora das chaves que corrigem: são 1 + n_QC ajustes de uma feature, e travar a correção por eles
# seria disputa que não existe. O custo está na docstring de `api/rotas.py::curva_do_loess`.
CHAVE_DA_CURVA_DO_LOESS: Final = "curva_do_loess"
ROTA_DA_CURVA_DO_LOESS: Final = "/matrizes/curva-do-loess"
# O que o rótulo do botão diz quando a resposta da curva (curva ou recusa) chega depois da troca de
# arquivos ou de modo e é descartada. Sem ela, o rótulo ficaria no 'Pedindo...' do disparo, e o
# botão pareceria morto.
FRASE_DA_CURVA_DESCARTADA: Final = (
    "A resposta da curva chegou depois de os arquivos ou o modo mudarem, e foi descartada: "
    "peça-a outra vez."
)
# As irmãs da frase da curva, para a medição, o Palpite e a execução. Dizem 'os arquivos ou o modo'
# porque a troca do modo também muda a matriz (`chave_da_entrada`), e é ela que descarta.
FRASE_DA_MEDICAO_DESCARTADA: Final = (
    "A medição chegou depois de os arquivos ou o modo mudarem, e foi descartada: peça-a outra vez."
)
FRASE_DO_PALPITE_DESCARTADO: Final = (
    "O Palpite chegou depois de os arquivos ou o modo mudarem, e foi descartado: peça-o outra vez."
)
# O desfecho da execução não é descartado: fica guardado com a matriz dele, e a frase diz onde ele
# aparece.
FRASE_DA_EXECUCAO_GUARDADA: Final = (
    "O desfecho da execução chegou depois de os arquivos ou o modo mudarem, e ficou guardado: "
    "ele aparece quando você voltar aos arquivos e ao modo dela."
)


def _identificador_do_aceite(resposta: Resposta) -> Any:
    """O identificador do trabalho num `202` que abre ou mantém uma espera, ou `None`.

    É a mesma expressão nos ramos de `_encaminha` e em `_chega_descartada`: antes da guarda de
    supersessão, o aceite encadeia a espera e passa.
    """
    if resposta.codigo == EM_CURSO and isinstance(resposta.corpo, dict):
        return resposta.corpo.get("identificador")
    return None


# Os ramos de `_encaminha` com guarda de supersessão, e os que, antes dela, deixam passar o aceite
# que encadeia a espera. A execução tem a guarda e não o atalho: nela a supersessão vem antes de
# tudo. `_chega_descartada` lê os dois conjuntos para perguntar o que os ramos perguntam.
#
# Os artefatos entram todos, os das abas de etapa inclusive, derivados do mapa de `artefatos.py`
# como as chaves que corrigem: uma chave de download nova fica sob a mesma guarda sem ninguém se
# lembrar dela.
_CHAVES_ENCADEADAS: Final = frozenset(
    {
        CHAVE_DO_PALPITE,
        CHAVE_DA_MEDICAO,
        *ETAPA_DA_CHAVE,
        *(passo.chave for passo in PASSOS_DOS_ARTEFATOS),
    }
)
_CHAVES_COM_SUPERSESSAO: Final = _CHAVES_ENCADEADAS | {CHAVE_DA_EXECUCAO}


class JanelaPrincipal(QMainWindow):
    """A janela. Compõe os painéis e exibe o desfecho de toda chamada.

    `resposta_recebida` e `chamada_falhou` são os dois sinais que o arnês espera com
    `qtbot.waitSignal`. O resultado chega por sinal, então um teste que lê o widget logo depois da
    ação passa ou reprova conforme o escalonamento.

    Os dois carregam `(chave, payload)`. A chave é a que o chamador passou a `chama` ou
    `enfileira`, e permite a um painel ligar-se ao sinal e ignorar as respostas que não são dele.
    """

    resposta_recebida = Signal(str, object)
    chamada_falhou = Signal(str, object)

    def __init__(
        self,
        transporte: Transporte,
        parent: QWidget | None = None,
        *,
        escolhedor: Escolhedor | None = None,
        pasta: EscolhedorDePasta | None = None,
        arquivo: EscolhedorDeArquivo | None = None,
        intervalo_da_espera_ms: int = INTERVALO_DA_ESPERA_MS,
        teto_da_espera_ms: int | None = TETO_DA_ESPERA_MS,
    ) -> None:
        """A janela. `escolhedor`, `pasta` e `arquivo` são os três diálogos, e entram por injeção.

        O padrão de cada um é o diálogo do sistema. Um `QFileDialog` modal não pode ser fechado por
        um teste, e sem a injeção ficariam inverificáveis a partir da janela: que os bytes escritos
        são os da resposta, que o seletor é invocado uma vez e as chamadas seguintes reusam os
        mesmos bytes, e que a figura gravada tem o enquadramento da tela e mais pixels.

        `intervalo_da_espera_ms` e `teto_da_espera_ms` são parâmetros pela mesma razão: com
        `teto=6, intervalo=2` o teste tem três leituras e um desfecho de desistência determinístico,
        sem o tempo de parede de uma espera real.
        """
        super().__init__(parent)
        self._transporte = transporte
        self._piscina = QThreadPool()
        self.setWindowTitle(TITULO)
        self.setWindowIcon(icone_da_janela())

        # A espera pelo artefato. Um relógio por chave e não um partilhado: `corrigir` e `exportar`
        # podem estar em voo ao mesmo tempo, e um relógio só faria a segunda espera reagendar a
        # primeira.
        self.intervalo_da_espera_ms = intervalo_da_espera_ms
        self.teto_da_espera_ms = teto_da_espera_ms
        self._esperas: dict[str, _EsperaDoArtefato] = {}
        self._relogios: dict[str, QTimer] = {}

        # Os trabalhos que esta janela disparou. `_esperas` é o que está a ser relido agora e se
        # esvazia quando a espera termina; isto é o histórico da sessão, de que o painel de
        # trabalhos vive.
        self._trabalhos: dict[str, str] = {}

        raiz = QWidget(self)
        disposicao = QVBoxLayout(raiz)

        self.abas = QTabWidget(raiz)
        # Os painéis de `PAINEIS`, todos compostos, de modo que `_vazios` fica vazio. Trocar uma
        # marca de painel vazio por um `QWidget` real não muda a contagem de abas, então a contagem
        # não serve de oráculo da composição.
        #
        # `salvador` é o `arquivo` injetado: os dois botões que salvam, a figura e o exemplo dos
        # fatores, usam o mesmo duplo nos testes; sem injeção, cada um abriria o seu.
        self.arquivos = PainelDeArquivos(
            escolhedor or escolhedor_por_dialogo(self), raiz, salvador=arquivo
        )
        self.leitura = PainelDeLeitura(raiz)
        self.trabalhos = PainelDeTrabalhos(raiz)
        self.pipeline = PainelDePipeline(raiz)
        # A importação dos fatores vai para a página 'Normalização', no lugar que o pipeline reserva
        # abaixo da referência da normalização. O estado continua no painel dos arquivos; só a
        # moldura muda de aba.
        layout_dos_fatores = self.pipeline.lugar_dos_fatores.layout()
        assert layout_dos_fatores is not None, "o pipeline nasce com o lugar dos fatores"
        layout_dos_fatores.addWidget(self.arquivos.importacao_dos_fatores)
        self.multiverso = PainelDeMultiverso(raiz)
        # O Resultado da Correção e o da Normalização: o download da etapa feita nos dois formatos,
        # 'Gerar os três' na Correção, e o resumo dos métodos dos arquivos daquela aba. Nascem aqui,
        # antes de as páginas das etapas serem compostas, porque é a página que os põe na seção
        # 'Resultado'. Os diálogos são os injetados, como os dos outros botões.
        self.resultado_da_correcao = ResultadoDaEtapa(1, raiz, escolhedor=pasta, salvador=arquivo)
        self.resultado_da_normalizacao = ResultadoDaEtapa(
            2, raiz, escolhedor=pasta, salvador=arquivo
        )
        self._resultados: dict[int, ResultadoDaEtapa] = {
            1: self.resultado_da_correcao,
            2: self.resultado_da_normalizacao,
        }
        # O pedido da última Análise (o resumo da execução diz o que ele levou), o de cada artefato,
        # o aceite de cada artefato, e os arquivos que o gesto corrente já escreveu no Resultado de
        # cada etapa, cada um com o seu resumo.
        self._pedido_da_execucao: dict[str, Any] | None = None
        self._pedidos_de_artefato: dict[str, dict[str, Any]] = {}
        self._aceites_de_artefato: dict[str, dict[str, Any]] = {}
        self._arquivos_do_resumo_da_etapa: dict[int, list[ArquivoExportado]] = {
            etapa: [] for etapa in self._resultados
        }
        # O que cada download do Resultado é, dito no disparo (a entrada da Normalização, por
        # exemplo), para o resumo do arquivo o dizer.
        self._descricoes_de_artefato: dict[str, str] = {}
        # O terceiro diálogo, injetado pela mesma razão dos dois primeiros: `QFileDialog` modal não
        # fecha por teste. Serve dois fluxos: o lote pede a pasta uma vez para os três corretores, e
        # o gesto unificado pede-a a cada clique de artefato.
        self._escolhe_pasta: EscolhedorDePasta = pasta or pasta_por_dialogo(self)
        # O destino de cada artefato em voo, por chave. Vive aqui e não no painel porque é
        # `principal` quem escreve: o painel não tem escritor nenhum (o conjunto sancionado pelo
        # portão por AST de `artefatos.py` é vazio), e a escrita não passa por diálogo, porque a
        # pasta veio do clique.
        self._destinos: dict[str, DestinoPendente] = {}
        self._escritos: dict[str, Path] = {}
        # O CSV de cada perna do lote, pelo identificador do trabalho dela. É por ele que o
        # relatório do corte de uma perna acha o arquivo ao lado do qual o registro vai: a chave do
        # artefato é a mesma nas três pernas, e a do painel já é a da perna seguinte quando o
        # relatório volta. A entrada sai quando o relatório chega.
        self._pernas_por_pedido: dict[str, Path] = {}
        # Os relatórios de pernas abortadas, que não têm destino. A perna cujo CSV não se escreveu
        # já pediu o relatório; sem o registro aqui ele cairia no caminho do clique avulso, e
        # `_escritos` poderia apontar para o CSV de uma exportação avulsa anterior, cujo irmão seria
        # reescrito com o corte de outro corretor.
        self._relatorios_descartados: set[str] = set()
        # O lote que encerrou com relatórios de perna ainda por voltar: ele, a frase de fim sem os
        # registros em falta, e se abortou. `_fecha_os_cortes_do_lote` o solta.
        self._lote_encerrado: tuple[LoteDeExportacao, str, bool] | None = None
        # O estado do lote em curso, ou `None`. Um só de cada vez, com a guarda em `_inicia_lote`:
        # dois lotes em voo escreveriam na mesma pasta com o combo a saltar.
        self._lote: LoteDeExportacao | None = None
        # O estado do passo a passo vive aqui, e não no servidor: o servidor é sem estado por
        # chamada, e quem sabe que etapa foi feita, e com que parâmetros, é a janela.
        self._etapas = EstadoDasEtapas()
        # A falha de cada etapa, até ela ser disparada de novo: o aviso vermelho da aba dela, com a
        # causa crua. E para onde o botão do aviso amarelo de cada aba leva: a própria etapa, que
        # ele refaz, ou a etapa mais a montante fora de dia, cuja aba ele abre.
        self._falhas_das_etapas: dict[int, str] = {}
        self._alvos_dos_avisos: dict[int, int] = {}
        self._identidade_memo: tuple[object, str, str] | None = None
        self._fatores_memo: tuple[bytes, str] | None = None
        # O Palpite medido, do lado da janela. `_palpite_pedido` é a identidade do Palpite em voo
        # (arquivos, modo, lotes e cortes), `_palpite_em_dia` a do último que chegou, e
        # `_depois_do_palpite` o que espera por ele, as sugestões da Correção. O servidor guarda o
        # Palpite pela mesma identidade; a janela só precisa de saber se já o pediu.
        self._palpite_pedido: tuple[Any, ...] | None = None
        self._palpite_em_dia: tuple[Any, ...] | None = None
        self._depois_do_palpite: Callable[[], object] | None = None
        # Os lotes que o último clique de 'Seguir sugestões do app' pôs, para a frase de uma
        # sugestão que não terminou dizer que eles já mudaram; e se o Palpite em voo, que o clique
        # encontrou a medir, é de outra partição que a do clique.
        self._lotes_da_sugestao: int | None = None
        self._remede_a_sugestao = False
        # A cadeia das leituras em curso, e as leituras em dia. `_leituras_em_dia` guarda, para a
        # Validação e o Diagnóstico, a matriz em que o último `200` chegou e se a Validação aceitou
        # os arquivos; está em dia a que é da matriz da tela (`_leitura_em_dia`). Um par novo ou
        # removido a esquece (`_esquece_as_amostras_dos_lotes`), como esquece os lotes que o
        # Diagnóstico propôs; sem isso o Palpite reusaria um Diagnóstico cuja sugestão já saiu.
        self._cadeia: _CadeiaDasLeituras | None = None
        self._leituras_em_dia: dict[str, tuple[str | None, bool]] = {}
        # A matriz do Palpite em voo, para o desfecho dele não se mostrar sob outra.
        # `_palpite_pedido` leva também os lotes e os cortes, e é a retomada que os compara; o que
        # se mostra segue a guarda das outras respostas, que é a da matriz.
        self._matriz_do_palpite: str | None = None
        # A matriz da execução em voo, pelo mesmo motivo: o desfecho de `/execucoes` enche a aba de
        # Qualidade, as figuras, as vias e o resumo. `None` sem disparo: o arnês que entrega o
        # desfecho direto não tem matriz.
        self._matriz_da_execucao: str | None = None
        # O último desfecho da execução de cada matriz, pela identidade dela. O desfecho de A que
        # chega com B na tela fica aqui e aparece quando a pessoa volta a A; a troca para B mostra o
        # de B, ou o repouso. Uma resposta por matriz, a última aceita.
        self._execucoes_guardadas: dict[str, _ExecucaoGuardada] = {}
        # A matriz cuja execução as abas mostram (guardada ou em voo), e `None` no repouso ou num
        # desfecho sem carimbo (o arnês), que não pertence a matriz nenhuma e a troca não toca.
        self._execucao_na_tela: str | None = None
        # O último `202` da execução em voo, para a volta à matriz dela mostrar 'ainda corre'.
        self._leitura_em_voo: Resposta | None = None
        # A ordem de chegada dos desfechos e das prévias, a última prévia de cada etapa com a matriz
        # do disparo dela, e o que a aba das vias mostrava de cada execução guardada: a predição e o
        # enriquecimento, que chegam depois dela e por outro caminho.
        self._ordem_dos_desfechos = 0
        self._previas: dict[int, _PreviaGuardada] = {}
        self._matriz_da_previa: dict[int, str | None] = {}
        self._predicoes_guardadas: dict[str, RelatorioDeVias] = {}
        self._enriquecimentos_guardados: dict[str, FotoDoEnriquecimento] = {}
        self._geracao_da_leitura = 0
        # O botão da tabela de normalidade salva pelo mesmo `arquivo` da figura e dos fatores, e o
        # do resumo da execução também: o resumo mora nesta aba, junto do que ele descreve.
        self.qualidade = PainelDeQualidade(raiz, salvador=arquivo)
        # Todo painel nasce com a chave em `_preenchidos`. Sem ela, o ramo de recuo de
        # `_compoe_a_aba` compõe a aba como marca em `_vazios`, com aparência de sucesso: a suíte
        # fica verde e a pessoa recebe uma aba vazia.
        self.graficos = PainelDeGraficos(raiz, escolhedor=arquivo)
        self.vias = PainelDeVias(raiz)
        self.recorte = PainelDeRecorte(raiz)
        self._preenchidos: dict[str, QWidget] = {
            "arquivos": self.arquivos,
            "leitura": self.leitura,
            CHAVE_DOS_TRABALHOS: self.trabalhos,
            "pipeline": self.pipeline,
            CHAVE_DO_MULTIVERSO: self.multiverso,
            "qualidade": self.qualidade,
            "graficos": self.graficos,
            "vias": self.vias,
            "recorte": self.recorte,
        }

        # ------------------------------------------------------------------
        # A ligação de saída: os sinais dos painéis disparam as rotas
        # ------------------------------------------------------------------
        self.leitura.pedir.connect(self._pede_leitura)
        self.pipeline.disparar_etapa.connect(self._dispara_etapa)
        # Mudar um campo sugerido apaga a frase 'Sugestões aplicadas' da página dele.
        self.pipeline.sugestao_desfeita.connect(self._esquece_o_desfecho_das_sugestoes)
        self.pipeline.ver_curva_do_loess.connect(self._dispara_curva_do_loess)
        # Trocar o par apaga a linha 'QCs por lote': ela é escrita só pela validação, e com outro
        # par na tela contaria os QCs da matriz anterior sob o número de lotes de agora.
        self.arquivos.par_escolhido.connect(self._esquece_as_amostras_dos_lotes)
        self.arquivos.arquivo_removido.connect(self._arquivo_removido)
        # A matriz dos gráficos segue a da tela, e o modo também a muda: sem isso, a medição e a
        # curva de A que chegassem com B escolhido seriam aceitas. O modo, que é parte da matriz
        # (`chave_da_entrada`), entra também na etiqueta das leituras (ver `_etiqueta_da_leitura`).
        self.arquivos.par_escolhido.connect(self._segue_a_matriz_da_tela)
        self.arquivos.modo.currentTextChanged.connect(self._segue_a_matriz_da_tela)
        # O frescor das etapas é recalculado, e não seguido por sinal. A máquina também escreve nos
        # controles (o pré-preenchimento da recomendação, o lote que move o combo do corretor), e
        # uma regra por sinal acompanharia gestos e não valores. A fotografia dos campos é comparada
        # com a guardada a cada meio segundo; a identidade dos arquivos é memorizada, então o custo
        # de uma volta é o de montar o dicionário do pedido.
        #
        # O relógio só corre com a janela visível (`showEvent`/`hideEvent`): janela que ninguém vê
        # não tem botão a habilitar, e os testes constroem centenas de janelas sem mostrar quase
        # nenhuma.
        self._relogio_das_etapas = QTimer(self)
        self._relogio_das_etapas.setInterval(INTERVALO_DAS_ETAPAS_MS)
        self._relogio_das_etapas.timeout.connect(self._atualiza_as_etapas)
        self.pipeline.medir_corretor.connect(self._dispara_medicao)
        # O download do Resultado de cada etapa passa por `_gera_artefato`, e o lote dos três é o
        # 'Gerar os três' do Resultado da Correção. O sinal carrega a chave do formato escolhido lá.
        for resultado in self._resultados.values():
            resultado.gerar.connect(self._gera_artefato)
        self.resultado_da_correcao.gerar_os_tres.connect(self._inicia_lote)
        # O modo viaja do painel de arquivos para o do multiverso, que mostra a estimativa do modo
        # aberto. A ligação vive aqui e não em nenhum dos dois painéis, como as outras travessias:
        # um painel que fosse buscar o estado do outro passaria a conhecer o resto da janela.
        # Liga-se a `currentTextChanged` e não ao disparo, porque a estimativa informa a decisão
        # antes do clique.
        self.arquivos.modo.currentTextChanged.connect(self.multiverso.define_modo)
        self.multiverso.define_modo(self.arquivos.modo.currentText())
        # O mesmo modo viaja para o painel das vias, onde decide a polaridade da ionização: `pos_*`
        # é positivo, `neg_*` é negativo. A chamada inicial não é redundante: sem ela, quem abrisse
        # o programa e corresse um `neg_hilic` direto receberia a polaridade positiva, que desloca
        # toda massa neutra em dois prótons, e o sintoma seria só 'menos vias'.
        self.arquivos.modo.currentTextChanged.connect(self.vias.sugere_polaridade)
        self.vias.sugere_polaridade(self.arquivos.modo.currentText())
        # A predição e o enriquecimento que terminam voltam com a matriz deles.
        self.vias.predicao_concluida.connect(self._guarda_a_predicao)
        self.vias.enriquecimento.resultado_concluido.connect(self._guarda_o_enriquecimento)
        self.multiverso.pedir.connect(self._le_conclusao_do_multiverso)
        # Sem estas ligações os botões do multiverso existiriam e não fariam nada; o portão de
        # ligação existe para acusar esse defeito.
        self.multiverso.disparar.connect(self._dispara_multiverso)
        self.multiverso.ler_progresso.connect(self._le_progresso_do_multiverso)
        self.multiverso.cancelar.connect(self._cancela_multiverso)
        self.multiverso.ler_historico.connect(self._le_historico_de_multiversos)
        self.multiverso.abrir_gravada.connect(self._abre_multiverso_gravado)
        # O painel de trabalhos também para trabalho, e não só o lista.
        self.trabalhos.cancelar.connect(self._cancela_o_trabalho)
        self.trabalhos.atualizar.connect(self._atualiza_os_trabalhos)

        # O recorte é calculado depois do dicionário e a partir dele: uma lista de painéis compostos
        # escrita à mão no módulo do recorte envelheceria em silêncio.
        #
        # `operaveis` nomeia os painéis cujos sinais são ligados acima, e nada mais. `qualidade`,
        # `graficos` e `recorte` ficam fora porque não têm controle que dispare rota: os dois
        # primeiros são mostradores e o terceiro é declaração. `graficos` tem combo e botão, mas
        # mudar a figura à vista não dispara rota nenhuma. É atributo e não literal dentro da
        # chamada para que o teste derive o conjunto em vez de o repetir.
        self.operaveis: frozenset[str] = frozenset(
            {
                "arquivos",
                "leitura",
                "pipeline",
                CHAVE_DO_MULTIVERSO,
                CHAVE_DOS_TRABALHOS,
            }
        )
        # Os painéis que calculam aqui, sem rota. O painel das vias liga o próprio botão a
        # `prediz_vias` dentro do processo (`vias.py`, `botao.clicked`), então não entra em
        # `operaveis`, que é bijetado contra quem chama `chama`/`enfileira`; sem este conjunto a
        # linha dele cairia no balde 'SÓ PARA LEITURA'. `tests/test_desktop_recorte.py` confere por
        # emissão que cada membro calcula de fato.
        self.calculam_aqui: frozenset[str] = frozenset({"vias"})
        self.recorte.define(
            compostos=frozenset(self._preenchidos),
            operaveis=self.operaveis,
            declarados={painel.chave: painel.unidade for painel in PAINEIS},
            calculam_aqui=self.calculam_aqui,
        )

        # ------------------------------------------------------------------
        # As abas, agrupadas por função
        # ------------------------------------------------------------------
        #
        # O laço itera `ABAS` e não `PAINEIS`. Cada aba pede o widget que a compõe a
        # `_compoe_a_aba`, que sabe qual composer cada agrupamento usa e devolve o painel sozinho
        # quando a aba tem um só.
        self._vazios: dict[str, QLabel] = {}
        self._paginas_de_etapa: dict[int, AbaDeEtapa] = {}
        # A justificativa das propostas, repartida pela aba do controle que cada parte explica; quem
        # sabe em que página mora cada controle é o painel do pipeline (`etapas_dos_eixos`). Em cada
        # página a parte abre pelo '?' ao lado da sugestão, num quadro
        # (`execucao.QuadroDaJustificativa`).
        self._partes_da_justificativa = self.leitura.reparte_a_justificativa(
            self.pipeline.etapas_dos_eixos()
        )
        # Todas as páginas nascem antes da inserção nas abas: a posse dos grupos já está
        # definida quando o QTabWidget passa a controlar sua visibilidade e seu ciclo de vida.
        paginas = [self._compoe_a_aba(aba) for aba in ABAS]
        for aba, pagina in zip(ABAS, paginas, strict=True):
            self.abas.addTab(pagina, aba.titulo)
        # O painel do pipeline é o controlador, e não é desenhado: os grupos e a linha da ação dele
        # já foram para as páginas das etapas. Escondido depois de as páginas o receberem, porque
        # esconder antes mudaria o estado dos filhos no instante de mudarem de pai. Sem isto ele
        # ficaria filho da raiz fora do layout, desenhado no canto por cima da barra das abas.
        self.pipeline.hide()
        # O estado vazio das etapas segue os sinais do painel de arquivos, e não o relógio das
        # etapas: o relógio só corre com a janela à vista, e a troca entre o vazio e o formulário é
        # o que a pessoa vê no instante em que escolhe ou remove um arquivo. `metade_escolhida`
        # cobre a metade sozinha, que muda a frase do que falta sem formar o par.
        self.arquivos.par_escolhido.connect(self._segue_os_arquivos_nas_etapas)
        self.arquivos.arquivo_removido.connect(self._segue_os_arquivos_nas_etapas)
        self.arquivos.metade_escolhida.connect(self._segue_os_arquivos_nas_etapas)
        self._segue_os_arquivos_nas_etapas()

        self._dimensiona()

        # A folga vertical da janela é das abas. Sem o fator, o `QVBoxLayout` reparte o que sobra
        # entre as abas, a faixa dos trabalhos e o rótulo de desfecho, e a faixa, que tem altura
        # própria e pequena, cresceria sozinha.
        disposicao.addWidget(self.abas, 1)

        # ------------------------------------------------------------------
        # A faixa dos trabalhos, visível em qualquer aba
        # ------------------------------------------------------------------
        #
        # Fica no rodapé da janela que este `__init__` monta, e no topo da que o aplicativo abre:
        # `tema._monta_faixa_de_trabalhos` tira-a daqui e põe-na acima das páginas. Ela responde 'já
        # ficou pronto?' sobre os trabalhos longos (uma correção com `serrf` leva minutos), e
        # durante esse tempo a pessoa está a olhar outra tela.
        #
        # O painel continua a ser painel, com a mesma chave e dentro de `_preenchidos`; só muda onde
        # ele é desenhado, e por isso `compostos` não significa 'tem aba'. São duas linhas: o título
        # e o aviso partilham a primeira, e os controles ficam na segunda. Numa linha só, o mínimo
        # da janela passaria de meia tela num monitor de 1920 px. O título é o do painel, lido de
        # `PAINEIS`.
        self.faixa_de_trabalhos = QGroupBox(raiz)
        self.faixa_de_trabalhos.setObjectName("faixa_de_trabalhos")
        dentro_da_faixa = QVBoxLayout(self.faixa_de_trabalhos)
        dentro_da_faixa.setContentsMargins(14, 8, 12, 8)
        dentro_da_faixa.setSpacing(6)
        linha_do_titulo = QHBoxLayout()
        linha_do_titulo.setSpacing(14)
        titulo_da_faixa = QLabel(
            next(p.titulo for p in PAINEIS if p.chave == CHAVE_DOS_TRABALHOS),
            self.faixa_de_trabalhos,
        )
        titulo_da_faixa.setObjectName("titulo_da_faixa")
        linha_do_titulo.addWidget(titulo_da_faixa)
        linha_do_titulo.addWidget(self.trabalhos.aviso_do_fecho)
        linha_do_titulo.addStretch(1)
        dentro_da_faixa.addLayout(linha_do_titulo)
        dentro_da_faixa.addWidget(self.trabalhos)
        disposicao.addWidget(self.faixa_de_trabalhos, 0)

        # O aviso recebe o rótulo de sempre: texto e contratos de rota continuam integrais.
        # Fechar recolhe a apresentação sem apagar esse último desfecho.
        self.aviso_de_desfecho = AvisoDeDesfecho(SEM_DESFECHO, raiz)
        self.rotulo_de_desfecho = self.aviso_de_desfecho.rotulo
        self.arquivos.par_escolhido.connect(self._oculta_o_aviso_de_arquivos)

        self.setCentralWidget(raiz)

    # ------------------------------------------------------------------
    # O desfecho
    # ------------------------------------------------------------------

    def _compoe_a_aba(self, aba: Aba) -> QWidget:
        """O widget de uma aba: o painel sozinho, ou o composer do agrupamento.

        Um `match` sobre a tupla de chaves, e não um mapa de composers, porque o agrupamento não é
        uniforme: 'Preparar o estudo' põe dois painéis lado a lado com uma moldura cada; Correção,
        Normalização e Análise estatística recebem os grupos do pipeline, o Resultado, a parte da
        justificativa que explica os controles delas e, na linha da ação, o botão, o estado, a barra
        e as notas de cada etapa. Os composers não têm a mesma assinatura.

        O `else` levanta em vez de recuar: um agrupamento novo sem composer produziria, num recuo,
        uma aba com o primeiro painel e os outros construídos, com pai e fora do layout, e a aba
        mostraria menos do que declara.
        """
        titulos = {painel.chave: painel.titulo for painel in PAINEIS}
        if aba.etapa is not None:
            # O botão da etapa vai à linha da ação, na seção do meio da página, com o estado, a
            # barra e as notas dela; na Análise, também o corretor e o resultado da execução. A
            # composição acontece antes de a página entrar na pilha das abas, que é quando mover é
            # seguro.
            notas: tuple[QWidget, ...] = (self.pipeline.notas_de_etapa[aba.etapa],)
            if aba.etapa == ETAPAS[-1]:
                notas = (
                    *notas,
                    self.pipeline.rotulo_do_corretor,
                    self.pipeline.rotulo_de_resultado,
                )
            pagina = AbaDeEtapa(
                self.pipeline.grupos_das_etapas[aba.etapa],
                self,
                etapa=aba.etapa,
                acao=self.pipeline.botoes_de_etapa[aba.etapa],
                estado=self.pipeline.rotulos_de_etapa[aba.etapa],
                barra=self.pipeline.barras_de_etapa[aba.etapa],
                resultado=self._resultados.get(aba.etapa),
                notas=notas,
                # A parte vai ao quadro que o '?' das sugestões abre; uma parte numa página sem as
                # sugestões levanta, em vez de sumir.
                justificativa=self._partes_da_justificativa.get(aba.etapa),
            )
            pagina.ir_ao_preparo.connect(self._vai_ao_preparo)
            pagina.ver_a_qualidade.connect(
                lambda: self.abas.setCurrentIndex(INDICE_DA_ABA_DA_QUALIDADE)
            )
            pagina.abrir_os_graficos.connect(
                lambda: self.abas.setCurrentIndex(INDICE_DA_ABA_DOS_GRAFICOS)
            )
            pagina.entrada_escolhida.connect(
                lambda entrada, etapa=aba.etapa: self._escolhe_a_entrada(etapa, entrada)
            )
            pagina.faixa_da_etapa.acionada.connect(
                lambda etapa=aba.etapa: self._age_no_aviso(etapa)
            )
            # 'Seguir sugestões do app', na Correção e na Normalização. Cada uma tem a sua ação: uma
            # página com o bloco e sem ação aqui levanta `KeyError` na composição, em vez de nascer
            # com um botão que não faz nada.
            if pagina.sugestoes is not None:
                acoes_das_sugestoes: dict[int, Callable[[], None]] = {
                    ETAPAS[0]: self._segue_as_sugestoes_da_correcao,
                    2: self._repoe_os_valores_iniciais,
                }
                pagina.seguir_sugestoes.connect(acoes_das_sugestoes[aba.etapa])
            # Os gráficos da etapa aparecem também na aba dela, além da aba 'Gráficos'. A vista
            # nasce com o painel dos gráficos, que é o dono do estado, e muda de pai aqui, antes de
            # a página entrar na pilha das abas.
            vista = self.graficos.vistas_das_etapas[aba.etapa]
            pagina.poe_os_graficos(vista)
            vista.exporta_em_graficos.connect(self._abre_a_figura)
            self._paginas_de_etapa[aba.etapa] = pagina
            nome = {1: "correcao", 2: "normalizacao", 3: "analise"}[aba.etapa]
            setattr(self, f"pagina_da_{nome}", pagina)
            return pagina
        if aba.paineis == ("arquivos", "leitura"):
            return AbaDePreparo(self.arquivos, self.leitura, titulos, self)
        if len(aba.paineis) == 1:
            unico = self._preenchidos.get(aba.paineis[0])
            if unico is not None:
                # As páginas que não cabiam rolam: sem tema, 'Vias metabólicas' e 'Multiverso'
                # pediam mais de 530 px de altura mínima e punham a janela inteira acima de um
                # portátil de 768 px. As outras páginas de um painel já cabem ou já rolam por dentro
                # (`qualidade`, `recorte`), e continuam a ser o painel; um teste afirma `isinstance`
                # sobre a da qualidade.
                if aba.paineis[0] in PAINEIS_QUE_ROLAM:
                    return pagina_rolavel(unico, self)
                return unico
            # A marca do painel ainda não composto. Diz de quem é o painel, sem afirmar quem o vai
            # ligar. Com todos os painéis compostos nenhuma aba chega aqui, e o ramo fica porque é
            # ele que torna `_vazios` verificável.
            declarado = next(p for p in PAINEIS if p.chave == aba.paineis[0])
            marca = QLabel(f"{declarado.titulo} — {declarado.unidade}, ainda não composto")
            self._vazios[declarado.chave] = marca
            return marca
        raise AgrupamentoSemComposerError(
            f"a aba «{aba.titulo}» agrupa {list(aba.paineis)} e nenhum composer sabe montar esse "
            f"conjunto. Um recuo razoável mostraria o primeiro painel e esconderia os outros — "
            f"construídos, com pai, e fora do layout"
        )

    def _dimensiona(self) -> None:
        """Abre num tamanho utilizável, e faz as abas encolherem em vez de sumirem.

        Sem isto o Qt dimensiona pela `sizeHint` do layout, e a janela abre estreita, com a barra a
        mostrar uma ou duas abas e as outras atrás das setas de rolagem; quem abre o aplicativo pela
        primeira vez não descobre que os outros painéis existem. Fixar uma largura que caiba todas
        as legendas quebraria numa tela de 1366 px, comum em portátil. Então há duas metades:

        - `ElideRight` na barra: todas as abas ficam sempre visíveis, encolhendo com reticências em
          vez de irem para trás de uma seta;
        - o teto pela tela disponível: a janela nunca nasce maior que o monitor de quem a abre.

        `resize` e não `setFixedSize`: a pessoa continua a poder redimensionar. O recuo existe
        porque a busca da tela devolve `None` quando não há tela (`offscreen`, sessão sem display),
        e um `AttributeError` aqui mataria a janela por causa de geometria.
        """
        self.abas.tabBar().setElideMode(Qt.TextElideMode.ElideRight)

        # 1440x900 porque a faixa dos trabalhos ocupa cerca de 90 px em todas as abas, e o que se
        # ganha em não trocar de aba cobra-se em altura. O `min` com 92% do disponível mantém a tela
        # pequena como era: quem tem 1366x768 não perde nada, e quem tem espaço deixa de rolar.
        preferida = (1440, 900)
        # Busca estática da tela, e não `self.screen()`. No PySide6 6.11 o objeto Python do
        # `QScreen` que `QWidget.screen()` devolve nasce filho do widget que chamou; quando esse
        # widget morre, o objeto é invalidado com o `QScreen` ainda vivo, e a consulta seguinte
        # corrompe o heap (`0xc0000374`). A busca estática não passa pela heurística do Shiboken.
        tela = QApplication.screenAt(self.geometry().center()) or QApplication.primaryScreen()
        if tela is None:  # pragma: no cover - só sem display; o teste cobre o caminho com tela
            self.resize(*preferida)
            return
        disponivel = tela.availableGeometry()
        self.resize(
            min(preferida[0], int(disponivel.width() * 0.92)),
            min(preferida[1], int(disponivel.height() * 0.92)),
        )

    def mostra_desfecho(self, codigo: int, corpo: Any = None) -> str:
        """Escreve o desfecho no painel e devolve o texto escrito.

        Devolve o texto para o chamador afirmar o que foi escrito sem reconstruir a expectativa, que
        seria uma segunda fonte da mesma frase.
        """
        texto = texto_do_desfecho(codigo, corpo)
        self.aviso_de_desfecho.mostra(
            texto,
            titulo="Não foi possível concluir" if codigo >= 400 else "Resultado do estudo",
            tipo="erro" if codigo >= 400 else "info",
            # Só a falha abre o aviso. O `2xx` fica registrado como último desfecho e recolhe o
            # aviso anterior; o resultado dele está no painel de quem pediu.
            aparece=codigo >= 400,
        )
        return texto

    def mostra_resposta(self, resposta: Resposta) -> str:
        """Escreve no painel o que `cliente.envia` classificou, sem rederivar nada."""
        texto = texto_da_resposta(resposta)
        self.aviso_de_desfecho.mostra(
            texto,
            titulo="Não foi possível concluir" if resposta.codigo >= 400 else "Resultado do estudo",
            tipo="erro" if resposta.codigo >= 400 else "info",
            aparece=resposta.codigo >= 400,
        )
        return texto

    def _recebe(self, chave: str, resposta: object) -> None:
        """A resposta chegou. Separa a chave composta uma vez, e daí para baixo é a base.

        O sinal `resposta_recebida` continua a emitir `"corrigir"` e não
        `"corrigir@<identificador>"`, então os painéis e os testes que afirmam a chave não veem
        diferença. Só `_encaminha` precisa do identificador, e recebe-o à parte.
        """
        assert isinstance(resposta, Resposta)
        base, pedido = parte_a_chave(chave)
        if self._leitura_desatualizada(base, pedido):
            return
        # O aviso de desfecho segue a guarda: a recusa de A descartada sob B não pode abrir 'Não foi
        # possível concluir', nem o `200` de A descartado recolher o aviso aberto de B. A pergunta é
        # feita aqui, antes, pelas mesmas contas que as guardas de `_encaminha` usam; o aviso
        # continua a abrir antes do ramo, porque os ramos escrevem no rótulo de desfecho depois
        # dele.
        if not self._chega_descartada(base, resposta, pedido):
            self.mostra_resposta(resposta)
        self._encaminha(base, resposta, pedido)
        self.resposta_recebida.emit(base, resposta)

    def _supersedida(self, chave: str, pedido: str | None) -> bool:
        """A releitura de uma cadeia que já não é a da espera desta chave.

        Clique novo, desistência ou fecho: a espera mudou de trabalho ou acabou. `pedido` só existe
        nas releituras automáticas, que `_rele` compõe com `chave_do_pedido`; o `POST` e o gesto
        manual chegam sem ele e passam. É uma função só porque os ramos de `_encaminha` e
        `_chega_descartada` fazem a mesma pergunta.
        """
        if pedido is None:
            return False
        espera = self._esperas.get(chave)
        return espera is None or espera.identificador != pedido

    def _chega_descartada(self, chave: str, resposta: Resposta, pedido: str | None) -> bool:
        """Se `_encaminha` vai descartar esta resposta sem a mostrar a ninguém.

        Faz a mesma pergunta que as guardas de `_encaminha`, na ordem delas e pelas mesmas contas:
        `_supersedida`, `_palpite_de_outra_matriz`, `_execucao_de_outra_matriz` e os dois
        predicados do painel dos gráficos que `recebe_medicao` e `recebe_curva_do_loess` usam para
        descartar. Uma conta reescrita aqui divergiria: o aviso abriria sobre o que o painel
        descartou, ou ficaria calado sobre o que ele mostrou.

        - O `202` que abre ou mantém uma espera passa nos ramos encadeados, como lá: antes da guarda
          de supersessão vem o aceite, e o de A, depois da troca, conta para a trava.
        - Na execução não há esse atalho: a supersessão vem antes de tudo, e só o desfecho que não
          é `202` passa pela guarda da matriz. O desfecho de outra matriz fica guardado mas não se
          mostra, nem nas abas nem no aviso; a volta à matriz dele repõe as abas e não reabre o
          aviso.
        - A validação e o diagnóstico não estão aqui: a guarda deles corre antes, em
          `_leitura_desatualizada`.
        """
        if chave in _CHAVES_ENCADEADAS and _identificador_do_aceite(resposta):
            return False
        if chave in _CHAVES_COM_SUPERSESSAO and self._supersedida(chave, pedido):
            return True
        if chave == CHAVE_DO_PALPITE:
            return self._palpite_de_outra_matriz()
        if chave == CHAVE_DA_MEDICAO:
            return self.graficos.medicao_de_outra_matriz()
        if chave == CHAVE_DA_CURVA_DO_LOESS:
            return self.graficos.curva_de_outra_matriz()
        if chave == CHAVE_DA_EXECUCAO:
            return resposta.codigo != EM_CURSO and self._execucao_de_outra_matriz()
        return False

    def _encaminha(self, chave: str, resposta: Resposta, pedido: str | None = None) -> None:
        """Entrega a resposta ao painel de quem ela é.

        `resposta_recebida` é emitido para toda chamada, e o roteamento é pela chave: sem ela, cada
        painel teria de adivinhar se a resposta é dele, ou haveria um sinal por painel.

        Uma chave que nenhum painel reconhece (uma chamada de arnês, uma sonda) cai no fim sem erro:
        levantar aqui mataria a resposta dentro de um slot de Qt, onde a exceção é engolida para o
        `stderr`.

        O `202` do progresso não alimenta a qualidade: o corpo de progresso não traz `qualidade`, e
        alimentá-la com ele escreveria 'não calculada' sobre valores que existem.
        """
        # As leituras. Sem este ramo, `validacao`, `diagnostico` e as outras cairiam no `return` do
        # fim, e o clique atravessaria até a rota sem a resposta aparecer.
        if chave in dict(ROTAS_DA_LEITURA):
            if chave == CHAVE_DO_PALPITE:
                # O Palpite é a única leitura longa, e encadeia a espera como a medição: `202` com
                # identificador agenda a leitura seguinte; o desfecho passa pela guarda de
                # supersessão e só então chega ao painel, como as outras leituras.
                identificador = _identificador_do_aceite(resposta)
                if identificador:
                    self.registra_trabalho(chave, str(identificador))
                    self._espera_pelo_artefato(chave, str(identificador), resposta.corpo)
                    return
                if self._supersedida(chave, pedido):
                    return
                self.para_a_espera(chave)
                # O Palpite de outra matriz não se mostra, não se guarda e não pré-preenche: o de A,
                # chegado com B na tela, viraria a proposta da máquina que `/execucoes` compara e o
                # corretor proposto de B. A espera das sugestões é largada com a frase de sempre,
                # senão ela ficaria guardada e a trava presa. E nenhum Palpite fica em dia: o cartão
                # já não mostra nenhum, e o próximo clique em 'Seguir sugestões do app' mede o da
                # tela, em vez de aplicar um que a pessoa não vê.
                if self._palpite_de_outra_matriz():
                    self.leitura.descarta(chave, FRASE_DO_PALPITE_DESCARTADO)
                    self._palpite_em_dia = None
                    self._larga_a_espera_das_sugestoes(
                        ENTRADA_MUDOU_DURANTE_O_PALPITE, com_os_lotes=False
                    )
                    # E a cadeia que o esperava termina com ele.
                    self._a_leitura_terminou(chave, resposta)
                    return
            self.leitura.mostra(chave, resposta.corpo)
            # Os nomes para o combo de referência. A validação é a única rota que os devolve, e
            # ligá-los no instante da exibição faz a execução usar o que a pessoa viu, e não uma
            # segunda chamada com outro resultado.
            if chave == "validacao" and isinstance(resposta.corpo, dict):
                # E a ordem de injeção com a marca de QC, para a linha 'QCs por lote' debaixo do
                # número de lotes. O painel confere as duas listas na entrada.
                self.pipeline.define_amostras(
                    resposta.corpo.get("ordens_de_injecao"), resposta.corpo.get("qc")
                )
                self.pipeline.povoa_referencias(
                    [
                        str(nome)
                        for nome in [
                            *(resposta.corpo.get("sample_names") or []),
                            *(resposta.corpo.get("feature_names") or []),
                        ]
                    ]
                )
                # E só as features para o combo da curva do LOESS: a mesma lista, sem os nomes de
                # amostra, porque a curva é de uma feature.
                self.pipeline.povoa_features_da_curva(
                    [str(nome) for nome in resposta.corpo.get("feature_names") or []]
                )
            # A justificativa de cada ponto de decisão automática. O ponto é casado pela chave:
            # `PontoDeDecisao.chave` é a mesma chave da rota que o produz (`diagnostico`,
            # `recomendacao`), então a correspondência é uma comparação e não um mapa a manter.
            for ponto in PONTOS_DE_DECISAO_AUTOMATICA:
                if ponto.chave == chave:
                    self.leitura.mostra_justificativa(ponto, resposta.corpo)
            # O palpite de duração sai da validação, a única das três leituras que traz
            # `n_features`, e a que corre primeiro. A travessia vive aqui e não no painel, pela
            # mesma razão do pré-preenchimento: quem vê os dois lados é a janela, e o painel do
            # pipeline não conhece rota nenhuma.
            if chave == "validacao" and isinstance(resposta.corpo, dict):
                self.pipeline.mostra_palpite_da_medicao(resposta.corpo.get("n_features"))
            if chave == "diagnostico" and isinstance(resposta.corpo, dict):
                # Os grupos do diagnóstico alimentam o painel do multiverso: a conclusão é medida
                # sobre um par de grupos, e `POST /multiversos` recusa com `422` um nome que não
                # exista. Vêm desta leitura, que já corre antes de qualquer decisão e já recebe o
                # par de arquivos; uma rota própria pediria um segundo envio dos mesmos dois CSV.
                #
                # O inventário é nome mais contagem, e os combos recebem só os nomes: o texto do
                # combo é o que `escolhas_do_disparo` envia à rota, e um `"ATLL (13)"` lá dentro
                # voltaria como `422` sobre um grupo que existe. Quem mostra a contagem é a linha do
                # painel do pipeline.
                #
                # `None` e `()` não são a mesma coisa: campo ausente não chama `define_grupos`, e
                # por isso não apaga o que a pessoa já tinha escolhido.
                inventario = inventario_do_corpo(resposta.corpo)
                if inventario is not None:
                    self.multiverso.define_grupos([grupo.nome for grupo in inventario])
                    self.pipeline.mostra_inventario_de_grupos(inventario)
                # A trajetória do QC atravessa da leitura para os gráficos. A travessia vive aqui, e
                # não num dos dois painéis, porque quem vê os dois lados é a janela. A figura é dos
                # gráficos porque é ali que a pessoa compara antes e depois, e a trajetória é a
                # régua contra a qual a leitura se lê. A chamada corre antes de
                # `figuras.define_execucao`, lá em baixo.
                self.graficos.recebe_diagnostico(resposta.corpo)
                # E a PCA inteira, do mesmo corpo e por um método próprio. As figuras de PCA saem do
                # mesmo `StructuralPCA` de que sai a trajetória; dois métodos e não um porque as
                # duas raízes respondem perguntas diferentes, e um que guardasse as duas faria a
                # ausência de uma implicar a da outra.
                self.graficos.recebe_pca(resposta.corpo)
                # E a figura escolhida é refeita: as do diagnóstico não esperam a execução, então a
                # chegada dele pode mudar o que a aba 'Gráficos' mostra, e o violino da ingerida nas
                # vistas das etapas.
                self.graficos.reavalia()
            # A proposta da máquina é guardada para os controles de valor. A divergência entre o
            # valor corrente e essa proposta é o que vira `campos_sobrescritos`; sem a proposta, a
            # divergência mediria a distância até `none`, o repouso dos combos, e toda execução
            # ficaria gravada como manual. As leituras guardam as propostas e não escrevem na aba
            # 'Correção'. A travessia vive aqui pela mesma razão das outras: o painel da leitura não
            # conhece os widgets do pipeline, e o do pipeline não chama rota.
            self._pre_preenche_com_a_proposta(chave, resposta.corpo)
            # A cadeia das leituras avança aqui, e a posição importa. Depois de a proposta ser
            # guardada, porque o Palpite que a cadeia dispara depois do Diagnóstico é medido nos
            # lotes que ele acabou de sugerir (`particao_sugerida`). E antes de `_palpite_chegou`,
            # porque a retomada das sugestões pode pedir outro Palpite pela cadeia, e a deste tem
            # de já ter terminado.
            self._a_leitura_terminou(chave, resposta)
            # E as sugestões que esperavam o Palpite entram agora, depois de o painel o mostrar e de
            # o corretor proposto estar pré-preenchido: é o que a tela vê.
            if chave == CHAVE_DO_PALPITE:
                self._palpite_chegou(resposta)
            return

        # O painel de trabalhos. A chave composta traz, no `pedido`, a chave do trabalho a que esta
        # leitura pertence (`trabalhos@corrigir`); sem estes ramos, uma atualização de `corrigir`
        # cairia no ramo dos artefatos e reabriria a espera daquele artefato. O cancelamento vem
        # antes, com base própria: lido no ramo da atualização, seria tomado por leitura de
        # progresso, que é outro corpo e outro significado.
        if chave == CHAVE_DO_CANCELAMENTO_DE_TRABALHO:
            if pedido is not None:
                self.trabalhos.mostra_cancelamento(pedido, resposta.codigo, resposta.corpo)
            self.trabalhos.mostra(self.trabalhos_conhecidos())
            return

        if chave == CHAVE_DOS_TRABALHOS:
            if pedido is not None:
                self.trabalhos.mostra_estado(pedido, resposta.codigo, resposta.corpo)
            self.trabalhos.mostra(self.trabalhos_conhecidos())
            return

        # A conclusão do multiverso chega com cinco códigos possíveis, e o painel trata os cinco: o
        # `200` com a tabela, o `202` com o sinal de vida, o `409` que manda redisparar declarando o
        # par de grupos, e os dois erros. Passar só o `200` deixaria o painel a exibir a conclusão
        # anterior ao lado de um desfecho novo.
        if chave == CHAVE_DO_MULTIVERSO:
            self.multiverso.mostra(resposta.codigo, resposta.corpo)
            return

        # As três respostas do ciclo do multiverso. Cada uma vai ao método que a sabe ler, e as três
        # chaves existem separadas por isso: o `202` do disparo traz `MultiversoAceito` e o do
        # cancelamento traz `ProgressoResposta`, e um `isinstance(corpo, dict)` não os distingue.
        if chave == CHAVE_DO_DISPARO:
            self.multiverso.aceita_disparo(resposta.codigo, resposta.corpo)
            return
        if chave == CHAVE_DO_PROGRESSO:
            self.multiverso.recebe_progresso(resposta.codigo, resposta.corpo)
            return
        if chave == CHAVE_DO_CANCELAMENTO:
            self.multiverso.aceita_cancelamento(resposta.codigo, resposta.corpo)
            return

        # As duas do histórico, separadas pelo mesmo argumento: a lista traz
        # `HistoricoDeMultiversos` e o detalhe traz `ConclusaoDoHistorico`, e eles vão para tabelas
        # diferentes.
        if chave == CHAVE_DO_HISTORICO:
            self.multiverso.mostra_historico(resposta.codigo, resposta.corpo)
            return
        if chave == CHAVE_DA_GRAVADA:
            self.multiverso.mostra_gravada(resposta.codigo, resposta.corpo)
            return

        # Os artefatos têm duas pernas: `POST /matrizes/corrigir` devolve `202` com identificador, e
        # os bytes vêm de `GET /trabalhos/{id}`. A mesma chave viaja na segunda perna, e é o que faz
        # o `200` com bytes voltar ao ramo dos artefatos e alimentar o painel certo. O `202` da
        # leitura traz `identificador`, então o mesmo ramo reencadeia a espera.
        if chave == CHAVE_DA_CURVA_DO_LOESS:
            # A curva do LOESS. A rota é síncrona: a resposta é a curva ou a recusa, sem espera a
            # encadear. O corpo vai sempre para o painel dos gráficos, para que um erro apague a
            # curva anterior em vez de a deixar sob outra feature. Com curva, a janela vai à figura,
            # como o atalho da moldura.
            if not self.graficos.recebe_curva_do_loess(resposta.corpo):
                # De outra matriz: descartada, e a janela fica onde está.
                self._diz_que_descartou(
                    self.graficos.resposta_da_curva_na_tela(),
                    self._escreve_o_rotulo_da_curva,
                    lambda: self.pipeline.mostra_curva_do_loess(FRASE_DA_CURVA_DESCARTADA),
                )
                return
            # O rótulo sai da resposta guardada, pela mesma função da troca de matriz.
            self._escreve_o_rotulo_da_curva(*self.graficos.resposta_da_curva_na_tela())
            if isinstance(resposta.corpo, dict) and "curva" in resposta.corpo:
                self._abre_a_figura(FIGURA_DA_CURVA_DO_LOESS)
            return
        if chave == CHAVE_DA_MEDICAO:
            # O mesmo laço dos artefatos, com JSON em vez de bytes: `202` com identificador encadeia
            # a espera, e `200` traz o corpo da medição. O ramo vem antes do dos artefatos porque a
            # chave não está em `ARTEFATOS`: a rota devolve `application/json`, e o ramo de baixo
            # serve `text/csv`.
            identificador = _identificador_do_aceite(resposta)
            if identificador:
                self.registra_trabalho(chave, str(identificador))
                self._espera_pelo_artefato(chave, str(identificador), resposta.corpo)
                return
            if self._supersedida(chave, pedido):
                # Cadeia supersedida: um clique novo trocou o trabalho, e a resposta velha é
                # descartada em silêncio, como no ramo dos artefatos; também no aviso de desfecho,
                # que `_recebe` deixa de abrir pela mesma pergunta (`_chega_descartada`).
                return
            self.para_a_espera(chave)
            # O corpo da medição vai para o painel dos gráficos. `figuras.define_execucao` não é
            # chamado com esta resposta: ela não traz `chave_de_invalidacao`, e o painel veria
            # 'execução nova' e retiraria a figura corrente por uma resposta que não é de
            # `/execucoes`.
            #
            # A carga corre para toda resposta, fora do `if`: o corpo de um erro, que não tem
            # `scores`, apaga a medição anterior em vez de a deixar viva sob a execução seguinte. O
            # rótulo segue a decisão do painel: a medição de A, chegada com B na tela, é descartada
            # também no rótulo.
            if not self.graficos.recebe_medicao(resposta.corpo):
                self._diz_que_descartou(
                    self.graficos.resposta_da_medicao_na_tela(),
                    self._escreve_o_rotulo_da_medicao,
                    lambda: self.pipeline.rotulo_da_medicao.setText(FRASE_DA_MEDICAO_DESCARTADA),
                )
                return
            # O desfecho vai para o rótulo da medição e não só para o aviso: uma recusa que não
            # aparece ao lado do botão recusado é indistinguível de botão morto. Vai o `detail` da
            # recusa, e não o dicionário, porque a recusa de lote com menos de três QCs é comum e a
            # frase dela diz o que fazer. A prosa e a recusa saem de `_escreve_o_rotulo_da_medicao`,
            # a mesma função que as reescreve quando a matriz muda e volta, e que lê a resposta que
            # o painel dos gráficos acabou de guardar.
            self._escreve_o_rotulo_da_medicao(*self.graficos.resposta_da_medicao_na_tela())
            return

        etapa = ETAPA_DA_CHAVE.get(chave)
        if etapa is not None:
            # As prévias do passo a passo, no molde do ramo da medição: `202` com identificador
            # encadeia a espera; o resto é o desfecho, depois da guarda de supersessão.
            identificador = _identificador_do_aceite(resposta)
            if identificador:
                self.registra_trabalho(chave, str(identificador))
                self._espera_pelo_artefato(chave, str(identificador), resposta.corpo)
                return
            if self._supersedida(chave, pedido):
                return
            self.para_a_espera(chave)
            self._etapa_terminou(etapa, resposta)
            return

        if chave in ARTEFATOS_DA_JANELA:
            painel = self._painel_do_artefato(chave)
            identificador = _identificador_do_aceite(resposta)
            if identificador:
                # O aceite do `POST` guarda a delegação: sob a escolha automática é ele que diz qual
                # corretor a política escolheu, e o arquivo de bytes não o diz.
                if e_corpo_de_aceite(resposta.codigo, resposta.corpo):
                    self._aceites_de_artefato[chave] = dict(resposta.corpo)
                # O painel de trabalhos vê os artefatos por aqui, o único ponto por onde o `202` de
                # um artefato passa. A faixa dos trabalhos o nomeia pela rota (`base`): um download
                # do Resultado é o mesmo trabalho de servidor que o artefato de que deriva, e o nome
                # legível de cada um já existe lá. Com uma correção por vez, dois downloads da mesma
                # rota não correm juntos; o identificador é o do trabalho, e é por ele que 'Parar' o
                # acha.
                self.registra_trabalho(ARTEFATOS_DA_JANELA[chave].base, str(identificador))
                self._espera_pelo_artefato(chave, str(identificador), resposta.corpo)
                return
            # Desfecho terminal: `200` com bytes, `200` sem bytes, ou erro.
            #
            # Um `200` de artefato não carrega identificador (é CSV), então o único jeito de saber
            # de qual trabalho ele é é a chave com que a leitura foi pedida, e só se entrega com uma
            # espera que case. Sem isto, o `200` de uma cadeia velha gravaria os bytes de A sob a
            # identidade de B. `pedido is None` entrega: é o `POST` (que devolve `202` com
            # identificador, ou um erro que o painel tem de mostrar) ou uma chamada de arnês.
            #
            # Espera ausente com pedido presente também descarta: um `200` que chegue depois de a
            # espera ter sido encerrada, por desistência, clique novo ou fecho, é descartado. Na
            # desistência é o que o rótulo promete, porque clicar outra vez reclama o mesmo
            # trabalho.
            if self._supersedida(chave, pedido):
                return
            # A espera encerra antes de o painel receber, para que um relógio já agendado não
            # relance uma leitura depois de o artefato ter chegado.
            self.para_a_espera(chave)
            painel.recebe(
                chave, resposta.codigo, resposta.conteudo, self._frase_do_artefato(chave, resposta)
            )
            # A escrita automática do gesto unificado, depois de `recebe` e não em vez dele: o
            # painel continua a guardar os bytes, e é deles que vivem o rótulo, o relatório do corte
            # e `salvaveis()`. Não há botão de baixar como rede, e uma escrita que falhe custa
            # regenerar. O destino sai por `pop` e não por `get`, porque vale para uma chegada só;
            # uma releitura reescreveria num caminho que a pessoa escolheu para o trabalho anterior.
            self._escreve_no_destino(chave, resposta.conteudo)
            # A terceira perna, só da exportação. O corte acontece dentro de `write_metaboanalyst`,
            # e o relatório dele não cabe na resposta do artefato, que é `text/csv` com os bytes que
            # sobem para o MetaboAnalyst. A rota irmã serve o registro, que alimenta o rótulo e o
            # arquivo irmão, escrito ao lado do CSV pelo ramo do relatório (no lote, pelo pedido de
            # cada perna).
            #
            # Fica condicionada aos bytes e não só à chave: sem artefato não há corte a relatar, e
            # pedir o relatório de um trabalho que falhou traria `409` para um rótulo que já mostra
            # a causa. Nada trata o desfecho aqui: `recebe_relatorio` ignora tudo o que não seja
            # `200` com corpo, e as outras rotas de artefato respondem `409` por desenho.
            relatorio = CHAVES_DO_RELATORIO.get(chave)
            # O pedido cujo relatório foi pedido, para o lote saber de qual perna ele virá.
            pedido_do_relatorio: str | None = None
            if pedido is not None and relatorio is not None and resposta.conteudo:
                self.chama(
                    "GET",
                    rota_do_relatorio(pedido),
                    chave=chave_do_pedido(relatorio, pedido),
                )
                pedido_do_relatorio = pedido
            # A quarta perna, só com lote vivo. Depois de `recebe` e não em vez dele: o painel
            # continua a guardar os bytes e a habilitar o 'Salvar' da perna corrente, então um lote
            # que aborte a meio deixa o último artefato ainda salvável à mão. Passa
            # `resposta.conteudo` e não o desfecho do painel, porque é o mesmo valor que decide lá:
            # `None` é 'a rota respondeu e não produziu artefato', e é ele que aborta.
            if self._lote is not None and chave == self._lote.chave:
                self._lote_recebe(resposta.conteudo, pedido_do_relatorio)
            return

        do_relatorio = ARTEFATO_DO_RELATORIO.get(chave)
        if do_relatorio is not None:
            # O relatório de uma perna abortada sai aqui, antes do rótulo. Sem CSV não há irmão a
            # escrever, e o rótulo mostra o aborto: o relatório não o troca pelo tamanho e o corte
            # de um arquivo que não está na pasta, nem desce ao caminho do clique avulso. A guarda
            # de `_escreve_no_destino` tira a chave de `_escritos` pela mesma razão.
            if pedido is not None and pedido in self._relatorios_descartados:
                self._relatorios_descartados.discard(pedido)
                return
            # O painel filtra por conta própria: chave sem bytes, ou código que não é `200`, não
            # tocam o rótulo. Ver `PainelDeArtefatos.recebe_relatorio`. O painel e a chave são os do
            # artefato que pediu o relatório: o download MetaboAnalyst do Resultado da Correção ou
            # da Normalização.
            painel = self._painel_do_artefato(do_relatorio)
            texto = painel.recebe_relatorio(do_relatorio, resposta.codigo, resposta.corpo)
            # O relatório de uma perna do lote escreve o irmão pelo pedido, e não pelo painel.
            # Quando ele volta, a perna seguinte já redefiniu a entrada do painel, e a guarda de
            # bytes de `recebe_relatorio` o recusa: `texto` vem vazio, e o caminho avulso abaixo
            # não escreveria nada para as pernas que não são a última.
            #
            # A guarda continua a governar o rótulo, que descreve só a perna à vista. O irmão sai do
            # corpo da resposta, ao lado do CSV que aquela perna escreveu, e uma falha de escrita
            # não aborta nada, porque o CSV já está na pasta: ela vai ao rótulo e à frase de fim do
            # lote (`_corte_da_perna_falhou`). Um relatório que não seja `200` não escreve, e a
            # perna sai do registro.
            perna = None if pedido is None else self._pernas_por_pedido.pop(pedido, None)
            if pedido is not None and perna is not None:
                dono = self._lote_do_pedido(pedido)
                do_lote = relatorio_da_resposta(resposta.codigo, resposta.corpo)
                falha = None if do_lote is None else escreve_o_corte_ao_lado(do_lote, perna)
                if falha is not None:
                    self._corte_da_perna_falhou(do_relatorio, perna, falha, dono)
                self._fecha_os_cortes_do_lote()
                return
            # O arquivo irmão do corte: escrevê-lo aqui impede a exportação de produzir o CSV
            # cortado sem o registro do que foi cortado. Fica condicionado ao texto e não ao código:
            # `recebe_relatorio` devolve vazio nos casos em que não há relatório a registrar (chave
            # sem bytes, código que não é `200`, corpo que não é dicionário), e é ele quem decide.
            escrito = self._escritos.get(do_relatorio)
            if texto and escrito is not None:
                painel.registra_corte_ao_lado(do_relatorio, escrito)
            return

        if chave != CHAVE_DA_EXECUCAO:
            return

        # A guarda de supersessão da execução. Com duas cadeias vivas (um segundo disparo enquanto a
        # primeira ainda é sondada), a resposta da cadeia velha escreveria o progresso por cima da
        # nova, e um `estado` terminal dela pararia a sondagem do trabalho que ainda corre. `pedido`
        # só existe nas releituras automáticas; o `POST` chega sem ele e passa, porque é ele que
        # abre a espera.
        if self._supersedida(CHAVE_DA_EXECUCAO, pedido):
            return

        # O desfecho de outra matriz não se mostra: executar A, escolher B e o `200` de A chegar
        # encheria a aba de Qualidade, as figuras, as vias e o resumo com os números de A sob os
        # arquivos de B. Ele fica guardado pela identidade de A e aparece quando a pessoa volta a A
        # (`_repoe_a_execucao`). O `202` de A sob B também não toca as abas, mas passa pela
        # contabilidade abaixo (a espera, a trava, o progresso no grupo da execução), porque o
        # trabalho de A corre de fato e a trava das correções tem de o contar.
        #
        # As etapas que o desfecho escreve são lidas aqui, antes da contabilidade:
        # `_etapas_que_a_execucao_escreve` lê a Análise em curso, e o `concluiu` de baixo tira-a de
        # lá. Ficam guardadas com o desfecho, para a volta à matriz dele escrever nas figuras das
        # etapas o que ele escreveu quando chegou.
        escreve = self._etapas_que_a_execucao_escreve()
        de_outra = self._execucao_de_outra_matriz()
        if not de_outra:
            self._mostra_a_execucao(resposta, escreve=escreve)
            self._execucao_na_tela = self._matriz_da_execucao
            # O resumo metodológico da execução, na aba 'Qualidade do resultado', ao lado das
            # métricas que este desfecho enche. Só no `200` com corpo (`_resumo_do_desfecho`): um
            # erro ou uma leitura de progresso não substituem o resumo do último resultado que
            # existe.
            resumo = self._resumo_do_desfecho(resposta)
            if resumo is not None:
                self.qualidade.resumo.mostra(resumo)

        # `aceita_disparo` é o único lugar que escreve o identificador da execução no pipeline, e é
        # ele que chama `carimba_disparo`; sem esta chamada, o identificador e o decorrido morreriam
        # juntos.
        #
        # `e_corpo_de_aceite` é o discriminador, o mesmo que a formatação usa. Dois corpos chegam
        # com `202`: o do `POST` (`TrabalhoAceito`, sem `estado`) e o de uma leitura de
        # `/trabalhos/`. Tratar uma leitura como aceite recarimbaria o disparo a cada consulta, e o
        # decorrido recomeçaria do zero.
        if e_corpo_de_aceite(resposta.codigo, resposta.corpo):
            identificador = self.pipeline.aceita_disparo(resposta.corpo, agora())
            if identificador:
                self.registra_trabalho(CHAVE_DA_EXECUCAO, identificador)
        elif pedido is None:
            # O disparo recusado também limpa: é por aqui que a limpeza de `aceita_disparo` é
            # alcançável. Sem isto, um `POST /execucoes` recusado deixaria o resultado e o
            # identificador da corrida anterior na tela. `pedido is None` é o discriminador, e não o
            # código: as releituras automáticas chegam sempre com `pedido`, então o que cai aqui é
            # resposta de disparo. As guardas de `aceita_disparo` devolvem `None` sem carimbar, e o
            # que se aproveita é só a limpeza dos três widgets.
            self.pipeline.aceita_disparo(resposta.corpo, agora())

        self.pipeline.recebe_leitura(resposta.codigo, resposta.corpo)
        self._agenda_a_releitura_da_execucao(resposta)
        # A etapa 3 do passo a passo termina com a execução: ela é o `/execucoes`, então o desfecho
        # dela é o desta chave. `200` conclui, e todo o resto que encerra a sondagem (`410`, `4xx`,
        # `5xx`, um `202` sem identificador) é falha. Sem etapa 3 em curso, nada.
        if ETAPAS[-1] in self._etapas.em_curso():
            if resposta.codigo == 200:
                self._etapas.concluiu(ETAPAS[-1])
            elif CHAVE_DA_EXECUCAO not in self._esperas:
                self._etapas.falhou(ETAPAS[-1])
                # A falha da Análise vira o aviso vermelho da aba dela, com a causa crua, e só com a
                # matriz dela na tela: a faixa sai quando a matriz da tela muda, e a de A posta sob
                # B ficaria sob B e sumiria na volta a A, ao contrário do desfecho, que fica
                # guardado.
                if not de_outra:
                    self._falhas_das_etapas[ETAPAS[-1]] = texto_da_resposta(resposta)
            # A linha da ação da Análise diz o estado já, e não no próximo meio segundo do relógio,
            # que nem corre com a janela escondida.
            self._atualiza_as_etapas()
        # Um `200` conclui só a Análise, que parte da entrada escolhida. O desfecho fica guardado
        # com a matriz dele, a recusa incluída: é o que a volta àquela matriz repõe. O `202` não é
        # desfecho (a espera continua), e fica só como a leitura em voo que a volta à matriz dele
        # mostra.
        if resposta.codigo == EM_CURSO:
            self._leitura_em_voo = resposta
            return
        self._leitura_em_voo = None
        self._guarda_a_execucao(resposta, escreve=escreve, de_outra=de_outra)

    def _mostra_a_execucao(
        self, resposta: Resposta | None, *, escreve: frozenset[int], por_troca: bool = False
    ) -> None:
        """Escreve nas abas que uma execução alimenta este desfecho, ou o repouso com `None`.

        `por_troca`: é a troca de matriz que repõe, e não um desfecho que chega; o enriquecimento
        em voo da matriz que sai não é cancelado (`PainelDeVias.recebe_volcano`).

        A troca de matriz passa por aqui para repor o desfecho guardado de uma matriz ou o repouso
        de outra, pelo mesmo caminho por que o desfecho chega. Nenhuma das abas lê o que a outra
        escreve. O resumo metodológico e o grupo da execução ficam de fora: um erro não substitui
        o resumo, e o grupo é escrito pela contabilidade, que com uma execução em voo mostra o
        progresso dela.

        `escreve` são as etapas guardadas cujas figuras este desfecho escreve, as que a Análise
        dele reproduz (`_etapas_que_a_execucao_escreve`), lidas na chegada e guardadas com ele.
        Cada etapa é dona dos dados das próprias figuras, e a volta a uma matriz escreve nelas o
        mesmo que o desfecho escreveu na chegada: nem mais (a Análise sobre a original não tem
        correção a mostrar), nem menos. O repouso não escreve em nenhuma.
        """
        corpo_da_execucao: Any = {} if resposta is None else resposta.corpo
        # Tudo isto corre em todo desfecho, e não só no `200`: senão a figura da execução anterior
        # atravessaria o `202` do progresso e o erro, a desenhar números que já não são os
        # correntes. `interface.cliente.chave_de_invalidacao` devolve a chave do servidor num `200`
        # com
        # chave e, no resto, um marcador irrepetível, para duas ausências nunca compararem iguais.
        #
        # Os dados de cada figura chegam antes de `define_execucao`, que constrói a figura no
        # instante em que reconhece a execução nova: guardar depois desenharia os dados da execução
        # anterior sob o rótulo da corrente. A ordem entre estas linhas é o invariante.
        #
        # Os dados das figuras das etapas 1 e 2 são das etapas, e a execução só os escreve quando
        # corresponde às etapas guardadas (`escreve`).
        self.graficos.recebe_distribuicoes(corpo_da_execucao, corresponde=2 in escreve)
        self.graficos.recebe_volcano(corpo_da_execucao)
        # As vias não recalculam estatística: a predição consome os p-valores e a estatística do
        # teste desta execução. Recalcular daria um segundo número, capaz de discordar do volcano.
        self.vias.recebe_volcano(corpo_da_execucao, por_troca=por_troca)
        # A ANOVA tem raiz própria: o volcano descreve o par de grupos declarado, e ela descreve
        # todos os grupos de estudo da matriz. As duas se ausentam por motivos diferentes.
        self.graficos.recebe_anova(corpo_da_execucao)
        self.graficos.recebe_correlacao(corpo_da_execucao)
        self.graficos.recebe_agrupamento_particional(corpo_da_execucao)
        # Aqui a ordem pesa mais: um `p` de permutação sob o rótulo errado engana justamente sobre
        # a validação do classificador.
        self.graficos.recebe_classificadores(corpo_da_execucao)
        # A rede tem método próprio porque `AnalysisSpec.dspc` e `AnalysisSpec.correlation` são
        # bandeiras separadas: a rede estima a precisão, o mapa estima a covariância.
        self.graficos.recebe_rede(corpo_da_execucao)
        # `ResumoDoPlsda` não admite score sem `q2` e `permutation_pvalue`.
        self.graficos.recebe_plsda(corpo_da_execucao)
        # O 'antes' do par vem daqui e não de `recebe_pca`, que guarda a PCA do diagnóstico: a
        # sequência arquivo A, diagnosticar, executar, arquivo B, diagnosticar poria o 'antes' de B
        # ao lado do 'depois' de A, porque o diagnóstico não viaja sob esta chave.
        self.graficos.recebe_pca_do_par(corpo_da_execucao, corresponde=1 in escreve)
        # O RSD por feature é medido sobre a matriz corrigida e não sobre a processada: entre as
        # duas está o corte por RSD de QC, que remove as features cuja dispersão o violino mostra.
        self.graficos.recebe_rsd_por_feature(corpo_da_execucao, corresponde=1 in escreve)
        # A composição por classe é medida sobre a matriz ingerida: as outras medem a correção, e
        # esta mede o estudo. A matriz corrigida tem células faltantes que o corretor criou.
        self.graficos.recebe_composicao(corpo_da_execucao, corresponde=1 in escreve)
        # A chave chega às quatro vistas: 'Gráficos' e as das abas de etapa.
        if resposta is None:
            self.graficos.esquece_a_execucao()
        else:
            self.graficos.define_execucao(resposta)
        # Um erro também limpa a aba de Qualidade. A figura, que vive noutra aba, já sai em todo
        # código que não é `200`; sem isto, depois de um `500` as métricas, o PLS-DA e o
        # antes-e-depois continuariam a mostrar a execução anterior sem nada a dizê-lo. Passar
        # `{}` faz os painéis caírem no estado de ausência que eles já produzem.
        #
        # O `202` é exceção: o corpo de progresso não traz qualidade a escrever, e uma leitura de
        # progresso não toca este painel. A figura, essa, sai com 'a execução ainda corre', então
        # no `202` as duas abas discordam; qual das duas está certa é decisão de quem opera a
        # ferramenta.
        codigo = None if resposta is None else resposta.codigo
        e_progresso = codigo == EM_CURSO
        leitura_completa = codigo == 200 and isinstance(corpo_da_execucao, dict)
        if not e_progresso:
            corpo: dict[str, Any] = corpo_da_execucao if leitura_completa else {}
            self.qualidade.mostra_qualidade(corpo.get("qualidade"))
            self.qualidade.mostra_plsda(corpo.get("qualidade"))
            self.qualidade.mostra_comparacao(corpo.get("rsd_antes_e_depois"))
            # O corpo inteiro e não a raiz: o vazio é o repouso, e a raiz ausente num corpo cheio
            # quer dizer 'não pedida'. Só o corpo separa os dois.
            self.qualidade.mostra_normalidade(corpo)

    # ------------------------------------------------------------------
    # A EXECUÇÃO DE CADA MATRIZ, GUARDADA E REPOSTA
    # ------------------------------------------------------------------

    def _resumo_do_desfecho(self, resposta: Resposta) -> ResumoMetodologico | None:
        """O resumo metodológico que este desfecho produz, ou `None`.

        Só no `200` com corpo, que é o que produziu a matriz: um erro ou uma leitura de progresso
        não substituem o resumo do último arquivo que existe. E só com o pedido guardado no
        disparo, porque o `200` não traz a semente, os lotes nem os eixos. O resumo aparece em
        'Qualidade do resultado', ao lado das métricas que o mesmo desfecho enche.
        """
        if resposta.codigo != 200 or not isinstance(resposta.corpo, dict):
            return None
        if self._pedido_da_execucao is None:
            return None
        return resumo_da_execucao(self._pedido_da_execucao, resposta.corpo)

    def _guardada_de(self, matriz: str | None) -> _ExecucaoGuardada | None:
        """O desfecho guardado da execução desta matriz, ou `None`; sem matriz, nenhum."""
        return None if matriz is None else self._execucoes_guardadas.get(matriz)

    def _guarda_a_execucao(
        self, resposta: Resposta, *, escreve: frozenset[int], de_outra: bool
    ) -> None:
        """Guarda o desfecho com a matriz do disparo; o de outra matriz não aparece na tela.

        A matriz é a do carimbo (`_matriz_da_execucao`), e não a da tela: é a que diz de que
        arquivos o desfecho é. Sem carimbo (o arnês que entrega o desfecho sem disparo) ele não
        pertence a matriz nenhuma, e não se guarda.

        O grupo da execução, o da página da Análise, guarda-se como a contabilidade de
        `_encaminha` acabou de o escrever, com o decorrido medido desde o disparo. Com outra matriz
        na tela, ele volta logo ao que é dela (o desfecho guardado, ou o repouso), e o progresso
        diz que o desfecho ficou guardado e onde aparece.

        `escreve` vai junto: as etapas cujas figuras este desfecho escreveu ao chegar, que a volta
        à matriz dele escreve outra vez.
        """
        dona = self._matriz_da_execucao
        if dona is None:
            return
        anterior = self._execucoes_guardadas.get(dona)
        ordem = self._proxima_ordem()
        sucesso = resposta.codigo == 200
        resumo = (
            self._resumo_do_desfecho(resposta)
            if sucesso
            else (None if anterior is None else anterior.resumo)
        )
        self._execucoes_guardadas[dona] = _ExecucaoGuardada(
            resposta,
            resumo,
            self.pipeline.foto_do_grupo_da_execucao(),
            ordem if sucesso else (-1 if anterior is None else anterior.ordem_do_sucesso),
            escreve,
        )
        # A predição e o enriquecimento eram da execução anterior: com a matriz na tela, o desfecho
        # novo já os tirou da aba das vias, e a volta à matriz não os repõe.
        self._predicoes_guardadas.pop(dona, None)
        self._enriquecimentos_guardados.pop(dona, None)
        if not de_outra:
            return
        na_tela = self._guardada_de(self._matriz_da_tela())
        self.pipeline.repoe_o_grupo_da_execucao(None if na_tela is None else na_tela.grupo)
        self.pipeline.rotulo_de_progresso.setText(FRASE_DA_EXECUCAO_GUARDADA)

    def _mostra_o_resumo(self, resumo: ResumoMetodologico | None) -> None:
        """O resumo da execução da matriz da tela, ou o repouso, na aba 'Qualidade do resultado'.

        O de uma execução é do arquivo que ela produziu, e sob outra matriz seria lido como o desta.
        Esta caixa só mostra o da execução, então segue sempre a matriz. Os resumos dos arquivos
        levados ficam no Resultado da etapa que os escreveu, e esses não seguem a matriz.
        """
        if resumo is None:
            self.qualidade.resumo.esquece()
            return
        self.qualidade.resumo.mostra(resumo)

    def _repoe_a_execucao(self, matriz: str | None) -> None:
        """A matriz da tela mudou: as abas da execução passam à desta (guardada, em voo, ou nada).

        Como a medição e a curva, esconde e repõe pela identidade, sem apagar. A troca para B
        mostra o desfecho guardado de B, ou o repouso se as abas mostravam o de outra matriz; a
        volta a A repõe o de A. Com a execução de A em voo, a volta a A mostra também, pelo último
        `202` dela, o que a tela de A mostrava enquanto ela corre: a figura em 'ainda corre', as
        vias à espera.

        O grupo da execução só segue a matriz quando não há execução em voo: com uma a correr, ele
        diz o progresso dela, qualquer que seja a matriz da tela, porque é ela que ocupa os
        núcleos.

        A execução reposta escreve nas figuras das etapas só o que escreveu quando chegou
        (`escreve`), e a do repouso não escreve em nenhuma. As prévias da matriz voltam na ordem em
        que chegaram: as mais velhas que o último `200` antes da execução, e as mais novas depois.
        A reposição da execução de outra matriz pode escrever nos campos da Correção; repor só as
        prévias mais novas deixaria no painel da correção de A, 'em dia', os dados de B. Na ordem
        de chegada, os campos terminam como estavam quando o desfecho chegou com a matriz na tela.
        """
        guardada = self._guardada_de(matriz)
        mudou = False
        if guardada is not None:
            self._repoe_as_previas(matriz, ate=guardada.ordem_do_sucesso)
            self._mostra_a_execucao(guardada.resposta, escreve=guardada.escreve, por_troca=True)
            self._mostra_o_resumo(guardada.resumo)
            self._repoe_a_aba_das_vias(matriz)
            self._execucao_na_tela = matriz
            mudou = True
        elif self._execucao_na_tela is not None:
            self._mostra_a_execucao(None, escreve=frozenset(), por_troca=True)
            self._mostra_o_resumo(None)
            self._execucao_na_tela = None
            mudou = True
        em_voo = CHAVE_DA_EXECUCAO in self._esperas
        if (
            em_voo
            and matriz is not None
            and matriz == self._matriz_da_execucao
            and self._leitura_em_voo is not None
        ):
            # A Análise de A ainda está em curso: as etapas que ela escreve são as de agora.
            self._mostra_a_execucao(
                self._leitura_em_voo, escreve=self._etapas_que_a_execucao_escreve(), por_troca=True
            )
            self._execucao_na_tela = matriz
        # As prévias mais novas por último, e só se esta troca escreveu nos campos das figuras: o
        # `202` em voo acima não apaga os dados de uma prévia (`graficos._da_execucao`).
        if mudou:
            self._repoe_as_previas(
                matriz, desde=-1 if guardada is None else guardada.ordem_do_sucesso
            )
        if mudou and not em_voo:
            self.pipeline.repoe_o_grupo_da_execucao(None if guardada is None else guardada.grupo)

    def _proxima_ordem(self) -> int:
        """A ordem de chegada do próximo desfecho ou prévia, por um contador da janela."""
        self._ordem_dos_desfechos += 1
        return self._ordem_dos_desfechos

    def _repoe_as_previas(
        self, matriz: str | None, *, desde: int | None = None, ate: int | None = None
    ) -> None:
        """Repõe as prévias desta matriz chegadas depois de `desde` e antes de `ate`, na ordem.

        As prévias das etapas escrevem nos campos das figuras que a execução escreve, e quem
        escreveu por último decide. Repor a execução de A por cima de uma prévia de A mais nova
        deixaria a figura com os dados de outro corretor enquanto a etapa 1 diz 'em dia'. Conta o
        último `200`, e não o último desfecho: um erro depois da prévia não a apaga, e é isso que
        `graficos._da_execucao` decide quando ele chega com a matriz na tela. As prévias de outra
        matriz não se repõem: o frescor das etapas já as diz desatualizadas.

        As mais velhas que o último `200` voltam antes da execução (`_repoe_a_execucao` diz por
        quê), e cada prévia volta com a entrada da etapa feita, que diz se ela escreve também nas
        figuras da Correção (`graficos.recebe_etapa`).
        """
        if matriz is None:
            return
        for previa in sorted(self._previas.values(), key=lambda p: p.ordem):
            if previa.matriz != matriz:
                continue
            if desde is not None and previa.ordem <= desde:
                continue
            if ate is not None and previa.ordem >= ate:
                continue
            self.graficos.recebe_etapa(previa.etapa, previa.corpo, entrada=previa.entrada)

    def _repoe_a_aba_das_vias(self, matriz: str | None) -> None:
        """A predição e o enriquecimento da execução guardada desta matriz, se ela os tinha."""
        if matriz is None:
            return
        predicao = self._predicoes_guardadas.get(matriz)
        if predicao is not None:
            self.vias.repoe_a_predicao(predicao)
        foto = self._enriquecimentos_guardados.get(matriz)
        if foto is not None:
            self.vias.enriquecimento.repoe(foto)

    def _dona_do_corpo(self, corpo: Any) -> str | None:
        """A matriz cuja execução guardada tem este corpo: o mesmo objeto, e não um igual."""
        for matriz, guardada in self._execucoes_guardadas.items():
            if guardada.resposta.corpo is corpo:
                return matriz
        return None

    def _guarda_a_predicao(self, corpo: Any, relatorio: RelatorioDeVias, pintada: bool) -> None:
        """Uma predição de vias terminou: fica com a matriz dona do corpo dela.

        Como na medição em voo, o resultado fica com a matriz de que é, para a volta a ela repor a
        tabela das vias preditas. A que chega já de volta à matriz dela pinta aqui: a do disparo
        não pintou, porque a troca avançou a geração do painel.
        """
        dona = self._dona_do_corpo(corpo)
        if dona is None:
            return
        self._predicoes_guardadas[dona] = relatorio
        if not pintada and self._execucao_na_tela == dona and self.vias.corpo_na_tela() is corpo:
            self.vias.repoe_a_predicao(relatorio)

    def _guarda_o_enriquecimento(
        self, corpo: Any, foto: FotoDoEnriquecimento, pintado: bool
    ) -> None:
        """Um enriquecimento terminou: fica com a matriz dona do corpo dele.

        O mesmo molde da predição. A troca não cancela o R, e o resultado que chega sob outra
        matriz volta com a dele.
        """
        dona = self._dona_do_corpo(corpo)
        if dona is None:
            return
        self._enriquecimentos_guardados[dona] = foto
        enriquecimento = self.vias.enriquecimento
        if (
            not pintado
            and self._execucao_na_tela == dona
            and enriquecimento.corpo_na_tela() is corpo
        ):
            enriquecimento.repoe(foto)

    # ------------------------------------------------------------------
    # O REGISTRO DE TRABALHOS DESTA JANELA
    # ------------------------------------------------------------------

    def registra_trabalho(self, chave: str, identificador: str) -> None:
        """Anota que esta janela disparou um trabalho longo, e alimenta o painel 3.

        O registro vive na janela porque é ela que vê todo `202` que passa por `_encaminha` (os
        três artefatos e a execução). Uma rota de índice (`GET /trabalhos`) exporia a lista de
        trabalhos retidos numa API sem autenticação, e o que interessa ao usuário são os trabalhos
        desta janela.

        Por chave, o último a escrever ganha: disparar `corrigir` outra vez substitui o trabalho de
        `corrigir`. A rota devolve o mesmo identificador para a mesma entrada, então redisparar sem
        mudar nada nem muda o valor.
        """
        self._trabalhos[chave] = identificador
        self.trabalhos.mostra(self.trabalhos_conhecidos())

    # A tabela `PROPOSTAS_POR_ROTA` vive em `interface/fluxo.py`: a travessia desta janela e a frase
    # que
    # o painel da leitura mostra precisam da mesma resposta, e duas listas à mão divergiriam.

    def _pre_preenche_com_a_proposta(self, chave: str, corpo: Any) -> tuple[str, ...]:
        """Põe no painel do pipeline o que a leitura propôs. Devolve os eixos que ela mudou.

        Devolver os eixos, e não `None`, deixa o arnês afirmar que o pré-preenchimento aconteceu
        em vez de o inferir do estado do widget.

        O corpo do diagnóstico traz a sugestão de lotes aninhada, em `lotes.n_lotes`, e não um
        `n_batches` de primeiro nível. A leitura desce um nível e não mais: `.get` encadeado sobre
        um corpo que não é dicionário devolveria `None` em silêncio, e o eixo sem proposta se
        leria como 'a máquina não propôs' quando a resposta só veio noutra forma.
        """
        if not isinstance(corpo, dict):
            return ()
        entrada = PROPOSTAS_POR_ROTA.get(chave)
        if entrada is None:
            return ()
        _eixo, campo = entrada
        proposto = corpo.get(campo)
        # O nome `lotes` diz a que esquema a base pertence: é a raiz aninhada que corresponde a
        # `SugestaoDeLotes`, o objeto que `prosa_dos_lotes` formata. O teste
        # `test_toda_base_de_leitura_esta_classificada` exige que toda base de leitura de chave
        # esteja classificada, e um nome genérico seria uma base nova sem esquema.
        if isinstance(proposto, dict):
            lotes = proposto
            valor = lotes.get("n_lotes")
        else:
            valor = proposto
        if valor is None:
            return ()
        if chave == "diagnostico":
            # O número de lotes fica guardado e não entra no campo: o padrão de quantos lotes
            # declarar é sempre 1, e quem põe a sugestão no campo é 'Seguir sugestões do app'.
            # Nada na tela muda, e por isso nenhum eixo volta.
            self.pipeline.guarda_a_sugestao_de_lotes(valor)
            if isinstance(proposto, dict):
                # Os cortes vêm junto com o número. Não têm controle na tela: valem enquanto o
                # número de lotes for o adotado, e é o painel que decide isso.
                self.pipeline.define_cortes_propostos(lotes.get("changepoints"))
            return ()
        # O corretor do Palpite também fica guardado: a página da Correção fica nos valores padrão
        # até o pesquisador a preencher ou escolher seguir as recomendações do app.
        self.pipeline.guarda_a_sugestao_do_corretor(valor)
        return ()

    def trabalhos_conhecidos(self) -> dict[str, str]:
        """`chave -> identificador` dos trabalhos desta janela, numa cópia.

        Devolver o dicionário vivo deixaria o painel, ou um teste, mutá-lo sem passar por
        `registra_trabalho`, e o painel deixaria de ser função do registro.
        """
        return dict(self._trabalhos)

    @property
    def leituras_maximas(self) -> int | None:
        """Quantas leituras o teto permite, ou `None` quando não há teto.

        Derivado do intervalo e do teto: uma constante à parte seria a terceira fonte de verdade
        sobre a mesma quantidade. `None` é o padrão e se propaga em vez de virar um número grande,
        que teria um ramo com cara de vivo que nunca corre. Não dá para derivar daqui o tempo de
        parede da última leitura, porque cada leitura tem latência própria.

        O piso de um existe porque `intervalo` de zero é legítimo num teste (faz o relógio
        disparar no próximo giro do laço), e uma divisão por ele levantaria.
        """
        if self.teto_da_espera_ms is None:
            return None
        return max(1, self.teto_da_espera_ms // max(1, self.intervalo_da_espera_ms))

    def esperas_em_curso(self) -> tuple[str, ...]:
        """As chaves pelas quais a janela está a esperar, em ordem. Vazio é o estado normal."""
        return tuple(sorted(self._esperas))

    def para_a_espera(self, chave: str) -> bool:
        """Encerra a releitura de uma chave. Devolve se havia espera para ela.

        O booleano distingue 'a espera parou' de 'não havia espera'.

        Quem impede a leitura nova é o `pop`: a guarda de `_rele` lê `self._esperas`. O `stop()`
        do relógio é só um cinto a mais, que evita entregar um `timeout` que já não serve; o
        desfecho é o mesmo com e sem ele.
        """
        relogio = self._relogios.get(chave)
        if relogio is not None:
            relogio.stop()
        return self._esperas.pop(chave, None) is not None

    def para_as_esperas(self) -> tuple[str, ...]:
        """Encerra todas. Devolve as chaves que estavam à espera, em ordem."""
        chaves = self.esperas_em_curso()
        for chave in chaves:
            self.para_a_espera(chave)
        return chaves

    def _espera_pelo_artefato(self, chave: str, identificador: str, corpo: dict[str, Any]) -> None:
        """Um `202` de artefato: a primeira leitura é imediata, as seguintes são agendadas.

        A primeira não espera o intervalo porque um artefato rápido fica pronto em fração de
        segundo, e o intervalo só protege a rota contra releitura em rajada; uma leitura não é
        rajada.

        O que distingue os ramos é a espera guardada, e não o formato do corpo (o `202` do `POST`
        e o do `GET` têm esquemas diferentes):
        - não existe: primeiro disparo, criado por um clique; cria e lê já;
        - existe com o mesmo identificador: uma leitura da espera que corre; conta e agenda;
        - existe com outro identificador: trabalho substituído por um clique; descarta. Sem isto,
          duas cadeias se reiniciariam uma à outra e produziriam rajada.

        `feito` ausente vira zero e o painel diz que não há contagem de passos, porque o `202` do
        `POST` não o traz. Escrever '0 de N' afirmaria que nenhum passo terminou.

        `corpo` não é revalidado: o único chamador só entra aqui depois de
        `isinstance(resposta.corpo, dict)`, que é a condição que produz o `identificador`.
        """
        feito = corpo.get("feito", 0)
        total = corpo.get("total", 0)
        espera = self._esperas.get(chave)

        if espera is None:
            # Primeiro disparo desta chave. Só o clique cria espera (`_gera_artefato` encerra a
            # anterior antes de disparar), e é isso que faz este ramo significar uma coisa só.
            self._esperas[chave] = _EsperaDoArtefato(identificador, leituras=1, inicio=agora())
            self._mostrador_da_espera(chave).mostra_espera(
                chave, feito=int(feito), total=int(total), leituras=1
            )
            self.chama(
                "GET",
                rota_do_progresso(identificador),
                chave=chave_do_pedido(chave, identificador),
            )
            return

        if espera.identificador != identificador:
            # Resposta de um trabalho substituído por um clique: a cadeia velha morre em silêncio.
            # Se caísse no ramo do primeiro disparo, duas cadeias vivas na mesma chave se
            # reiniciariam uma à outra, com leitura imediata e sem relógio, e o teto nunca chegaria.
            return

        espera.leituras += 1
        # A guarda de `None` vem antes da comparação, senão `int > None` levanta. Sem teto, que é
        # o padrão, este ramo não corre em produção; os testes põem teto para exercitá-lo.
        teto = self.leituras_maximas
        if teto is not None and espera.leituras > teto:
            # O teto passou. A espera é encerrada e o painel diz o que fazer: a rota devolve o
            # mesmo trabalho, então clicar outra vez no mesmo botão reclama em vez de recalcular.
            # Os segundos são medidos no relógio, e não derivados do intervalo, que ignora a
            # latência das próprias leituras.
            self.para_a_espera(chave)
            self._mostrador_da_espera(chave).mostra_desistencia(
                chave,
                leituras=espera.leituras - 1,
                segundos=agora() - espera.inicio,
            )
            if chave == CHAVE_DO_PALPITE:
                self._larga_a_espera_das_sugestoes(
                    "Parei de esperar o Palpite, e o corretor sugerido não entrou — a medição "
                    "continua no servidor: clique de novo, e ela é reclamada em vez de recomeçar."
                )
                # E a cadeia que o pediu termina: o próximo clique é outro pedido.
                self._a_leitura_terminou(chave, None, motivo="a espera do Palpite desistiu")
            return

        self._mostrador_da_espera(chave).mostra_espera(
            chave, feito=int(feito), total=int(total), leituras=espera.leituras
        )
        self._relogio_de(chave).start(self.intervalo_da_espera_ms)

    def _agenda_a_releitura_da_execucao(self, resposta: Resposta) -> None:
        """A sondagem automática de `/execucoes`.

        Reusa `_esperas`, `_relogio_de` e `_rele`, que sondam os artefatos e a medição. Não reusa
        `_espera_pelo_artefato`, que escreveria o sinal de vida no painel errado: quem escreve o
        progresso da execução é `pipeline.recebe_leitura`, que já move a barra.

        Para por estado e nunca por contagem, porque não há teto de leituras:
        - `200`: o resultado chegou, não há mais o que ler;
        - `202` com `estado` em `TERMINAIS`: concluído, falhou ou cancelado;
        - `202` sem identificador: não há o que consultar;
        - qualquer outro código: `4xx` e `5xx` não viram progresso sozinhos;
        - `202` a correr: agenda a próxima daqui a `intervalo_da_espera_ms`.

        `TERMINAIS` é importado e não enumerado aqui: um `estado == "concluido"` escrito à mão faria
        a janela sondar para sempre um trabalho que falhou ou um estado novo.

        Não há botão para ler o progresso à mão. Quando a sondagem para por código de erro, o
        rótulo diz o que aconteceu, e quem quiser insistir redispara.
        """
        corpo = resposta.corpo if isinstance(resposta.corpo, dict) else {}
        identificador = self.pipeline.identificador
        if resposta.codigo != 202 or not identificador or str(corpo.get("estado", "")) in TERMINAIS:
            self.para_a_espera(CHAVE_DA_EXECUCAO)
            return
        # A contagem e o instante do disparo são da espera e não da leitura, e sobrevivem a cada
        # resposta. Reiniciá-los faria o contador sugerir que o trabalho recomeçou.
        anterior = self._esperas.get(CHAVE_DA_EXECUCAO)
        self._esperas[CHAVE_DA_EXECUCAO] = _EsperaDoArtefato(
            identificador,
            leituras=1 + (anterior.leituras if anterior else 0),
            inicio=anterior.inicio if anterior else agora(),
        )
        self._relogio_de(CHAVE_DA_EXECUCAO).start(self.intervalo_da_espera_ms)

    def _mostrador_da_espera(self, chave: str) -> Any:
        """Quem mostra o sinal de vida desta chave.

        Existe para o laço de espera ser reusado em vez de duplicado: é nele que vivem o teto de
        leituras, o descarte de cadeia substituída e o relógio, e uma segunda cópia divergiria.

        Não há `else` silencioso. A chave é comparada contra as constantes, e uma chave fora das
        famílias conhecidas levanta `KeyError` em `_painel_do_artefato`; cair num painel qualquer
        escreveria a espera onde ninguém a pediu.
        """
        if chave == CHAVE_DA_MEDICAO or chave in ETAPA_DA_CHAVE:
            return self.pipeline
        if chave == CHAVE_DO_PALPITE:
            return self.leitura
        return self._painel_do_artefato(chave)

    def _painel_do_artefato(self, chave: str) -> ResultadoDaEtapa:
        """O Resultado da etapa dona desta chave, que mostra os bytes e o estado dela.

        O mapa de `artefatos.py` diz qual etapa é a dona.
        """
        return self._resultados[ARTEFATOS_DA_JANELA[chave].etapa]

    def _frase_do_artefato(self, chave: str, resposta: Resposta) -> str:
        """A frase do desfecho que o Resultado da chave escreve.

        É a mensagem de `cliente.py` e, quando veio, o `detail` da rota, que diz a causa e a
        alternativa: a recusa do MetaboAnalyst por valor negativo nomeia o eixo que o produziu e o
        pedido que a evita.
        """
        corpo = resposta.corpo
        detalhe = corpo.get("detail") if isinstance(corpo, dict) else None
        if isinstance(detalhe, str) and detalhe.strip():
            return f"{resposta.mensagem} {detalhe}"
        return resposta.mensagem

    def _relogio_de(self, chave: str) -> QTimer:
        """O relógio desta chave, criado na primeira vez como filho da janela.

        `QTimer(self)` põe-no na árvore de objetos da janela: quando ela é destruída, ele vai com
        ela. Um `QTimer()` sem pai sobreviveria à janela e dispararia num slot de um objeto morto.
        """
        relogio = self._relogios.get(chave)
        if relogio is None:
            relogio = QTimer(self)
            relogio.setSingleShot(True)
            relogio.timeout.connect(lambda: self._rele(chave))
            self._relogios[chave] = relogio
        return relogio

    def _rele(self, chave: str) -> None:
        """O disparo do relógio: uma leitura do progresso. Chave sem espera não faz nada.

        `stop()` e a remoção do dicionário não são atômicos com um `timeout` que já esteja na fila
        de eventos, então este método tolera disparar sobre uma espera encerrada. Uma exceção num
        slot de Qt vai para o `stderr`, que num executável `--windowed` não existe.
        """
        espera = self._esperas.get(chave)
        if espera is None:
            return
        self.chama(
            "GET",
            rota_do_progresso(espera.identificador),
            chave=chave_do_pedido(chave, espera.identificador),
        )

    def _falha(self, chave: str, erro: object) -> None:
        """Exceção que `envia` não trata: mostra a falha sem inventar causa.

        `envia` converte toda `Exception` em `Resposta`, então só `BaseException` fora dessa
        hierarquia chega aqui. Um `503` sintético confundiria 'a rota respondeu 503' com 'o
        processo está a morrer'.

        Ficar em silêncio seria pior: o rótulo continuaria com o desfecho da chamada anterior, e
        quem olha leria um sucesso de uma chamada que morreu.

        A frase não usa o vocabulário de causa, porque `causa_de` classifica desfecho de rota e
        isto não é um. Como `janela/arquivos.py` faz com `OSError`, escreve só a classe da exceção,
        que é diagnóstico.
        """
        # Separa a chave composta aqui também, senão a frase mostra o identificador do pedido,
        # que é detalhe de encadeamento e não diz nada a quem lê.
        base, pedido = parte_a_chave(chave)
        if self._leitura_desatualizada(base, pedido):
            return
        self.aviso_de_desfecho.mostra(
            f"A chamada `{base}` não chegou a produzir desfecho de rota: {type(erro).__name__}",
            titulo="A chamada não foi concluída",
            tipo="erro",
        )
        # A etapa também sai de 'em curso' aqui, senão travaria toda correção da janela.
        etapa = ETAPA_DA_CHAVE.get(base, ETAPAS[-1] if base == CHAVE_DA_EXECUCAO else None)
        if etapa is not None and etapa in self._etapas.em_curso():
            self._etapas.falhou(etapa)
            # E o aviso vermelho da aba, que é onde a pessoa espera o desfecho.
            self._falhas_das_etapas[etapa] = (
                f"a chamada não chegou a produzir desfecho de rota ({type(erro).__name__})"
            )
            self._atualiza_as_etapas()
        # E a espera das sugestões pelo Palpite não fica guardada.
        if base == CHAVE_DO_PALPITE:
            self._larga_a_espera_das_sugestoes(
                "O Palpite não completou, e o corretor sugerido não entrou: a chamada não chegou "
                f"a produzir desfecho de rota ({type(erro).__name__})"
            )
        # E a cadeia das leituras não fica à espera de uma leitura que morreu.
        if base in dict(ROTAS_DA_LEITURA):
            self._a_leitura_terminou(
                base,
                None,
                motivo=(
                    f"a chamada não chegou a produzir desfecho de rota ({type(erro).__name__})"
                ),
            )
        self.chamada_falhou.emit(base, erro)

    # ------------------------------------------------------------------
    # OS SLOTS QUE CHAMAM AS ROTAS
    # ------------------------------------------------------------------
    #
    # A composição do payload vive na janela, o único lugar que vê todos os painéis: cada painel
    # sabe o que escolheu, e nenhum sabe ao mesmo tempo o par de arquivos, o modo e os eixos.

    def _pede_leitura(self, chave: str) -> None:
        """Uma das três rotas de leitura, escolhida pela chave que o botão emitiu.

        A rota sai de `ROTAS_DA_LEITURA`, a mesma tupla com que o painel cria os rótulos e os
        botões; um `if` aqui seria uma segunda tabela.

        O Diagnóstico e o Palpite não saem direto: rodar o Palpite exige o Diagnóstico e a
        Validação, então eles vão por `_pede_a_cadeia`, que roda antes as leituras anteriores fora
        de dia.
        """
        if dict(ROTAS_DA_LEITURA).get(chave) is None:
            # Chave que a tupla não declara é defeito de ligação, e não erro do usuário; no painel
            # de desfecho ele fica visível em vez de morrer dentro de um slot de Qt.
            self.rotulo_de_desfecho.setText(f"Leitura desconhecida: {chave}")
            return
        # A Validação, primeira da ordem, não depende de nada e sai sozinha.
        if chave == ORDEM_DAS_LEITURAS[0]:
            self._dispara_a_leitura(chave)
            return
        self._pede_a_cadeia(chave)

    def _dispara_a_leitura(self, chave: str) -> bool:
        """`POST` da Validação ou do Diagnóstico, com o par da tela. Devolve se saiu.

        Sem par, `_pedido_de_arquivos` já pôs no aviso o que falta, e nada sai. O Palpite não
        passa por aqui: ele tem rota longa, lotes, núcleos e a trava das correções
        (`_dispara_o_palpite`).
        """
        pedido = self._pedido_de_arquivos()
        if pedido is None:
            return False
        arquivos, campos = pedido
        if chave == "diagnostico":
            # A matriz do disparo, para as figuras do diagnóstico valerem sem execução.
            self.graficos.carimba_o_diagnostico()
        rota = dict(ROTAS_DA_LEITURA)[chave]
        self.chama("POST", rota, chave=chave, arquivos=arquivos, campos=campos)
        return True

    # ------------------------------------------------------------------
    # A CADEIA DAS LEITURAS
    # ------------------------------------------------------------------
    #
    # Validação, Diagnóstico e Palpite, nessa ordem. Nenhuma é obrigatória para usar o app, mas
    # rodar o Palpite exige o Diagnóstico e a Validação em dia, e rodar o Diagnóstico exige a
    # Validação.
    #
    # Um estado e uma função que avança, em vez de condições espalhadas. `_cadeia` é a cadeia em
    # curso; `_pede_a_cadeia` a começa ou a estende; `_avanca_a_cadeia` dispara a próxima leitura,
    # no começo e a cada uma que chega (`_a_leitura_terminou`); `_para_a_cadeia` a encerra com a
    # frase do porquê. A ordem é `ORDEM_DAS_LEITURAS`, e ninguém mais a escreve.

    def _leitura_em_dia(self, chave: str) -> bool | None:
        """Se a leitura `chave` está em dia para a matriz da tela.

        Três valores: `True`, em dia e aceita; `False`, em dia e recusada (só a Validação recusa,
        e 'Aceita com avisos' é aceita); `None`, fora de dia ou nunca lida. `False` e `None` pedem
        ações opostas: o primeiro para a cadeia, o segundo corre a leitura outra vez.
        """
        matriz = self._matriz_da_tela()
        guardada = self._leituras_em_dia.get(chave)
        if matriz is None or guardada is None or guardada[0] != matriz:
            return None
        return guardada[1]

    def _palpite_em_voo(self) -> bool:
        """Se há um Palpite pedido e ainda sem desfecho: a espera dele, ou o `POST` da cadeia.

        A cadeia cobre o intervalo entre o `POST` e o primeiro `202`, em que a espera da rota ainda
        não nasceu.
        """
        cadeia = self._cadeia
        return CHAVE_DO_PALPITE in self._esperas or (
            cadeia is not None and cadeia.em_voo == CHAVE_DO_PALPITE
        )

    def _pede_a_cadeia(self, alvo: str) -> bool:
        """Pede `alvo` pela cadeia: roda antes as leituras anteriores fora de dia, e depois ele.

        Devolve se há uma cadeia a correr até `alvo`, esta ou a que já corria. O alvo corre
        sempre, porque a pessoa o pediu: um Diagnóstico pedido de novo é refeito. O que se reusa
        são as anteriores em dia para a matriz da tela.

        Um segundo pedido com a cadeia a correr não abre outra. Um alvo mais adiante estende a que
        corre (o Palpite pedido com o Diagnóstico a correr); um igual ou anterior não faz nada,
        porque a que corre já passa por ele.

        Para o Palpite, a trava das correções é perguntada aqui também, e não só ao dispará-lo:
        com uma correção a correr, a pessoa sabe na hora que ele não sai, em vez de validar e
        diagnosticar primeiro. Recusado o alvo novo, a cadeia que já corria segue até o dela.
        """
        self._confere_a_etiqueta_da_cadeia()
        cadeia = self._cadeia
        if cadeia is not None and ORDEM_DAS_LEITURAS.index(alvo) <= ORDEM_DAS_LEITURAS.index(
            cadeia.alvo
        ):
            return True
        if alvo == CHAVE_DO_PALPITE:
            motivo = self._motivo_para_nao_corrigir(alvo)
            if motivo is not None:
                frase = CADEIA_PALPITE_NAO_SAIU.format(motivo=motivo.rstrip("."))
                self.leitura.mostra_a_parada(alvo, frase, MANCHETE_DO_PALPITE_QUE_NAO_SAIU)
                self.aviso_de_desfecho.mostra(frase, titulo=TITULO_DO_PALPITE, tipo="aviso")
                return False
        if cadeia is not None:
            # O alvo anterior fica guardado: o cartão dele pediu esta cadeia, e continua a mostrar
            # a etapa dela até a leitura dele chegar ou a cadeia parar antes.
            cadeia.anteriores.append(cadeia.alvo)
            cadeia.alvo = alvo
            self._mostra_a_etapa()
            return True
        self._cadeia = _CadeiaDasLeituras(alvo, self._etiqueta_da_leitura())
        return self._avanca_a_cadeia()

    def _avanca_a_cadeia(self) -> bool:
        """Dispara a próxima leitura da cadeia, ou a para. Devolve se uma leitura saiu.

        A próxima é a primeira, na ordem, que não está em dia e aceita, ou o alvo, que corre
        sempre. A Validação em dia e recusada para a cadeia: nem o Diagnóstico nem o Palpite
        rodam sobre arquivos recusados. O Palpite pergunta à trava antes de sair, porque uma
        correção que começou enquanto as outras leituras corriam disputaria os núcleos com ele.
        """
        cadeia = self._cadeia
        if cadeia is None:
            return False
        ate_o_alvo = ORDEM_DAS_LEITURAS[: ORDEM_DAS_LEITURAS.index(cadeia.alvo) + 1]
        proxima = next(
            chave
            for chave in ate_o_alvo
            if chave == cadeia.alvo or self._leitura_em_dia(chave) is not True
        )
        if proxima != cadeia.alvo and self._leitura_em_dia(proxima) is False:
            self._para_a_cadeia(
                lambda cartao: CADEIA_VALIDACAO_RECUSADA.format(alvo=ROTULOS_DA_LEITURA[cartao]),
                MANCHETE_DA_VALIDACAO_RECUSADA,
            )
            return False
        if proxima == CHAVE_DO_PALPITE:
            motivo = self._motivo_para_nao_corrigir(proxima)
            if motivo is not None:
                frase = CADEIA_PALPITE_NAO_SAIU.format(motivo=motivo.rstrip("."))
                self._para_a_cadeia(lambda _cartao: frase, MANCHETE_DO_PALPITE_QUE_NAO_SAIU)
                return False
            saiu = self._dispara_o_palpite()
        else:
            saiu = self._dispara_a_leitura(proxima)
        if not saiu:
            # Sem arquivos, o único motivo de a leitura não sair depois da trava.
            # `_pedido_de_arquivos` já pôs no aviso o que falta, com a origem que o recolhe quando
            # o par se completa; reescrevê-lo aqui apagaria essa origem.
            self._cadeia = None
            self._larga_a_espera_das_sugestoes(self.rotulo_de_desfecho.text())
            return False
        cadeia.em_voo = proxima
        self._mostra_a_etapa()
        return True

    def _mostra_a_etapa(self) -> None:
        """A etapa em voo no cartão pedido, e na espera de 'Seguir sugestões do app', se há.

        Também nos cartões dos alvos anteriores que ainda a esperam, cada um com a etapa contada
        até ele, para o cartão do Diagnóstico de uma cadeia estendida ao Palpite não ficar parado
        na Validação com o Diagnóstico já a correr.
        """
        cadeia = self._cadeia
        if cadeia is None or cadeia.em_voo is None:
            return
        em_voo = ORDEM_DAS_LEITURAS.index(cadeia.em_voo)
        for anterior in cadeia.anteriores:
            if (
                ORDEM_DAS_LEITURAS.index(anterior) >= em_voo
                and self.leitura.etapa_em_curso(anterior) is not None
            ):
                self.leitura.mostra_etapa(anterior, frase_da_etapa(cadeia.em_voo, anterior))
        self.leitura.mostra_etapa(cadeia.alvo, frase_da_etapa(cadeia.em_voo, cadeia.alvo))
        if self._depois_do_palpite is not None:
            self._bloco_das_sugestoes(ETAPAS[0]).espera(self._motivo_da_espera_das_sugestoes())

    def _motivo_da_espera_das_sugestoes(self) -> str:
        """Por que 'Seguir sugestões do app' espera: a etapa da cadeia, ou o Palpite já a medir."""
        cadeia = self._cadeia
        if cadeia is None or cadeia.em_voo is None:
            return MEDINDO_A_SUGESTAO
        etapa = frase_da_etapa(cadeia.em_voo, cadeia.alvo)
        if cadeia.em_voo == CHAVE_DO_PALPITE:
            return ESPERA_DAS_SUGESTOES_NO_PALPITE.format(etapa=etapa)
        return ESPERA_DAS_SUGESTOES_ANTES_DO_PALPITE.format(etapa=etapa)

    def _a_leitura_terminou(
        self, chave: str, resposta: Resposta | None, *, motivo: str = ""
    ) -> None:
        """Uma leitura terminou: guarda se ela está em dia e avança a cadeia que a esperava.

        `resposta` é o desfecho da rota, ou `None` quando não houve desfecho de rota (a chamada que
        morreu, a espera do Palpite que desistiu), e então `motivo` diz o que houve.

        Em dia é só o `200` legível, e da matriz da tela no instante da chegada: a leitura que
        chega aqui passou pela etiqueta, então é desta matriz. Qualquer outro desfecho tira a
        leitura de dia, porque o cartão passou a mostrar a falha, e não uma leitura.

        O alvo que termina encerra a cadeia sem frase, porque o cartão já mostra o desfecho dele.
        Uma anterior que não completou para a cadeia, e a frase diz qual e por quê.
        """
        if chave in ORDEM_DAS_LEITURAS[:-1]:
            corpo = None if resposta is None else resposta.corpo
            if resposta is not None and resposta.codigo == 200 and isinstance(corpo, dict):
                aceita = chave != ORDEM_DAS_LEITURAS[0] or bool(corpo.get("aceita"))
                self._leituras_em_dia[chave] = (self._matriz_da_tela(), aceita)
            else:
                self._leituras_em_dia.pop(chave, None)
        cadeia = self._cadeia
        if cadeia is None or cadeia.em_voo != chave:
            return
        if chave == cadeia.alvo:
            self._cadeia = None
            return
        if self._leitura_em_dia(chave) is None:
            if resposta is not None:
                motivo = (
                    texto_da_resposta(resposta)
                    if resposta.codigo != 200
                    else "a resposta não pôde ser lida"
                )
            anterior = ROTULOS_DA_LEITURA[chave]
            self._para_a_cadeia(
                # O cartão da própria leitura que não completou (um alvo anterior da cadeia
                # estendida) diz só que ela não completou: 'parou antes do Diagnóstico' no cartão
                # do Diagnóstico não descreveria nada.
                lambda cartao: (
                    LEITURA_DA_CADEIA_NAO_COMPLETOU.format(anterior=anterior, motivo=motivo)
                    if cartao == chave
                    else CADEIA_LEITURA_NAO_COMPLETOU.format(
                        anterior=anterior, alvo=ROTULOS_DA_LEITURA[cartao], motivo=motivo
                    )
                ),
                MANCHETE_DA_LEITURA_QUE_NAO_COMPLETOU.format(anterior=anterior),
            )
            return
        # Os lotes do Diagnóstico entram agora se 'Seguir sugestões do app' espera e o clique, que
        # veio antes de haver Diagnóstico, não os pôs. É o que o clique teria feito com ele em dia:
        # os lotes primeiro, e o corretor quando o Palpite chegar. O que a pessoa mudar depois fica.
        # E só se ela não mudou os lotes desde o clique: a frase da espera promete respeitar o que
        # ela mudar até lá, e `segue_a_proposta` esquece o toque para poder escrever.
        if (
            chave == "diagnostico"
            and self._depois_do_palpite is not None
            and self._lotes_da_sugestao is None
        ):
            if self.pipeline.foi_tocado(EIXO_DOS_LOTES):
                # Os lotes mudados durante a espera contam como sobrescrita. O campo fica como a
                # pessoa o deixou, e o que o Diagnóstico propôs vira o lado da máquina do par,
                # mesmo se o Palpite não chegar.
                self.pipeline.registra_a_sugestao(EIXO_DOS_LOTES)
            else:
                self._poe_os_lotes_da_sugestao()
        self._avanca_a_cadeia()

    def _confere_a_etiqueta_da_cadeia(self) -> None:
        """A cadeia para se a matriz ou a geração dos anexos mudaram.

        Com outra etiqueta, a leitura que a cadeia espera é descartada ao chegar
        (`_leitura_desatualizada`), e a cadeia esperaria para sempre, travando todo pedido
        seguinte. O Palpite já disparado não para nada: ele fica fora da etiqueta, e a
        substituição e a guarda da matriz dele decidem o que fazer com o desfecho.
        """
        cadeia = self._cadeia
        if cadeia is None or cadeia.etiqueta == self._etiqueta_da_leitura():
            return
        if cadeia.em_voo == CHAVE_DO_PALPITE:
            self._cadeia = None
            return
        self._para_a_cadeia(
            lambda cartao: CADEIA_MATRIZ_MUDOU.format(alvo=ROTULOS_DA_LEITURA[cartao]),
            MANCHETE_DA_MATRIZ_QUE_MUDOU,
            com_os_lotes=False,
        )

    def _para_a_cadeia(
        self, frase_para: Callable[[str], str], manchete: str, *, com_os_lotes: bool = True
    ) -> None:
        """Encerra a cadeia: a frase no cartão pedido e no aviso.

        Se 'Seguir sugestões do app' espera, a frase vai também ao lado dele, pela mesma porta das
        outras sugestões que não terminam. `com_os_lotes` segue `_larga_a_espera_das_sugestoes`:
        sem os lotes quando a frase é sobre outra entrada, porque o número no campo já não é o do
        clique.

        `frase_para` recebe o cartão e devolve a frase dele. A cadeia estendida (o Palpite pedido
        com o Diagnóstico a correr) tem mais de um cartão à espera, e cada um diz antes de que
        leitura o app parou; sem isso o cartão do Diagnóstico ficaria 'Em curso' sem nada a correr.
        Um anterior cuja leitura já chegou fica como está; o aviso diz a frase do alvo final.
        """
        cadeia, self._cadeia = self._cadeia, None
        if cadeia is None:
            return
        frase = frase_para(cadeia.alvo)
        for anterior in cadeia.anteriores:
            if self.leitura.etapa_em_curso(anterior) is not None:
                self.leitura.mostra_a_parada(anterior, frase_para(anterior), manchete)
        self.leitura.mostra_a_parada(cadeia.alvo, frase, manchete)
        if self._depois_do_palpite is not None:
            self._larga_a_espera_das_sugestoes(frase, com_os_lotes=com_os_lotes)
            return
        self.aviso_de_desfecho.mostra(frase, titulo=ROTULOS_DA_LEITURA[cadeia.alvo], tipo="aviso")

    # ------------------------------------------------------------------
    # O PALPITE MEDIDO
    # ------------------------------------------------------------------

    def _identidade_do_palpite(self) -> tuple[Any, ...] | None:
        """O que identifica um Palpite medido: a matriz, o modo, os lotes e os cortes, ou `None`.

        É a identidade com que o servidor o guarda, lida do lado de cá: os bytes dos dois arquivos
        e o modo fazem a matriz, e os lotes e os cortes são a partição que o Diagnóstico sugeriu
        (`particao_sugerida`). O Palpite é independente da aba Correção: nada que se mude nela
        muda esta identidade. Sem arquivos não há identidade, e quem chama diz o que falta.
        """
        par = self.arquivos.par
        if par is None:
            return None
        # O modo sem os espaços das pontas, como `campos_para_envio` o manda e `_matriz_da_tela`
        # o conta. Com o modo cru, um espaço no fim faria duas contas da mesma identidade
        # discordarem, enquanto o servidor vê uma só.
        modo = self.arquivos.modo.currentText().strip()
        n_lotes, cortes = self.pipeline.particao_sugerida()
        return (
            self._matriz_de(par, modo),
            modo,
            n_lotes,
            None if cortes is None else tuple(cortes),
        )

    def _dispara_o_palpite(self) -> bool:
        """`POST /matrizes/palpite`, com os lotes e os núcleos da tela. Devolve se saiu.

        Os campos são os de `_dispara_medicao`, pelo mesmo motivo: o Palpite mede na partição em
        que a correção vai correr, e com a caixa dos núcleos. A trava vem antes, como lá: o
        Palpite corre os três corretores, e uma correção em voo disputaria os mesmos núcleos.

        Em produção só a cadeia o chama (`_avanca_a_cadeia`), depois da Validação e do Diagnóstico
        em dia.
        """
        motivo = self._motivo_para_nao_corrigir(CHAVE_DO_PALPITE)
        if motivo is not None:
            self.leitura.rotulos[CHAVE_DO_PALPITE].setText(motivo)
            self.rotulo_de_desfecho.setText(motivo)
            return False
        par = self._pedido_de_arquivos()
        if par is None:
            return False
        arquivos, campos = par
        campos["paralelo"] = self.pipeline.paralelo.isChecked()
        # A partição sugerida, e não a do campo da aba Correção: o Palpite é independente dela.
        n_lotes, cortes = self.pipeline.particao_sugerida()
        campos["n_batches"] = n_lotes
        if cortes is not None:
            campos["batch_cuts"] = json.dumps(list(cortes))
        self._palpite_pedido = self._identidade_do_palpite()
        # O carimbo, depois de `_pedido_de_arquivos` e pela mesma conta que o painel dos gráficos
        # recebeu.
        self._matriz_do_palpite = self._matriz_da_tela()
        self.para_a_espera(CHAVE_DO_PALPITE)
        self.leitura.rotulos[CHAVE_DO_PALPITE].setText(
            "Medindo o Palpite em 200 features... o primeiro corretor foi disparado."
        )
        # O mesmo, no que está à vista do cartão; o texto acima fica em 'Detalhes'.
        self.leitura.cartoes[CHAVE_DO_PALPITE].mostra_resumo(
            "info", "Medindo", "O primeiro corretor foi disparado"
        )
        self.chama(
            "POST", ROTA_DO_PALPITE, chave=CHAVE_DO_PALPITE, arquivos=arquivos, campos=campos
        )
        return True

    # ------------------------------------------------------------------
    # SEGUIR SUGESTÕES DO APP
    # ------------------------------------------------------------------

    def _bloco_das_sugestoes(self, etapa: int) -> BlocoDeSugestoes:
        """O bloco das sugestões da página da `etapa`. Levanta se ela não o tem.

        Um clique que chegasse aqui sem bloco seria um sinal ligado à página errada.
        """
        bloco = self._paginas_de_etapa[etapa].sugestoes
        if bloco is None:
            raise LookupError(f"a página da etapa {etapa} não tem «Seguir sugestões do app»")
        return bloco

    def _segue_as_sugestoes_da_correcao(self) -> None:
        """O clique de 'Seguir sugestões do app' na Correção: lotes e corretor propostos nos campos.

        Os lotes vêm do Diagnóstico e o corretor, do Palpite. O botão só põe nos campos o que a
        máquina propôs: a Correção leva sempre o corretor do seletor, sem delegar, e não espera o
        Palpite para sair.

        Os lotes primeiro, porque o Palpite é medido na partição em que a correção vai correr, e é
        por ela que o servidor o guarda: pôr a proposta de lotes depois de medir faria o Palpite
        ser de outra partição. Com o Palpite destes arquivos, modo, lotes e cortes já em dia, o
        corretor entra na hora; senão ele é medido (ou esperado, se já estiver a medir) e o
        corretor entra quando ele chegar (`_palpite_chegou`). Enquanto isso o botão fica desligado
        com a frase do porquê ao lado.
        """
        bloco = self._bloco_das_sugestoes(ETAPAS[0])
        bloco.limpa()
        self._remede_a_sugestao = False
        self._poe_os_lotes_da_sugestao()
        # O clique é a pessoa a pedir as sugestões: o toque de antes dele, no corretor ou nos
        # lotes, não conta contra elas. O de depois, durante a espera, conta como escolha dela
        # (`_aplica_as_sugestoes_da_correcao`), como a frase da espera promete.
        self.pipeline.esquece_o_toque(EIXO_DO_CORRETOR)
        self.pipeline.esquece_o_toque(EIXO_DOS_LOTES)
        identidade = self._identidade_do_palpite()
        if identidade is not None and identidade == self._palpite_em_dia:
            self._aplica_as_sugestoes_da_correcao()
            return
        if self._palpite_em_voo():
            # Um Palpite já mede (o do cartão, ou o `POST` da cadeia antes do primeiro `202`), e
            # pode ser de outra partição se a sugestão do Diagnóstico mudou depois de ele ser
            # pedido. Quando ele chegar, a sugestão é pedida outra vez, na partição de agora.
            self._remede_a_sugestao = self._palpite_pedido != identidade
        else:
            motivo = self._motivo_para_nao_corrigir(CHAVE_DO_PALPITE)
            if motivo is not None:
                bloco.diz_o_desfecho(
                    self._com_os_lotes(
                        f"O corretor não pôde ser sugerido agora: {motivo.rstrip('.')}."
                    )
                )
                return
            # O Palpite vai pela cadeia, que roda antes a Validação e o Diagnóstico que faltarem;
            # direto, ele mediria num lote só quando o Diagnóstico ainda não sugeriu a partição.
            if not self._pede_a_cadeia(CHAVE_DO_PALPITE):
                bloco.diz_o_desfecho(
                    self._com_os_lotes(
                        f"O corretor não pôde ser sugerido: {self.rotulo_de_desfecho.text()}"
                    )
                )
                return
        self._depois_do_palpite = self._aplica_as_sugestoes_da_correcao
        bloco.espera(self._motivo_da_espera_das_sugestoes())

    def _poe_os_lotes_da_sugestao(self) -> None:
        """Os lotes que o Diagnóstico propôs no campo, e o número guardado para as frases.

        Chamado pelo clique de 'Seguir sugestões do app' e pela chegada do Diagnóstico que a
        cadeia rodou para ele: o clique sem Diagnóstico em dia não tem lotes a pôr, e eles entram
        quando o Diagnóstico chega, antes de o Palpite sair.
        """
        lotes = self.pipeline.segue_a_proposta(EIXO_DOS_LOTES)
        self._lotes_da_sugestao = (
            lotes if isinstance(lotes, int) and not isinstance(lotes, bool) else None
        )

    def _recomendacao_adotada(self) -> dict[str, Any] | None:
        """O Palpite guardado, se o corretor dele foi adotado na Correção; senão `None`.

        O Palpite só é o lado da máquina do par quando a pessoa o seguiu. Ele fica guardado sem
        entrar no seletor, e guardado e não seguido ele não é uma proposta que a pessoa trocou:
        passá-lo a `proposta_do_pedido` gravaria na linha da máquina um corretor que ninguém pôs
        na tela. Adotado por 'Seguir sugestões do app', ele está em `_proposta` com o mesmo valor.

        Vale também quando a pessoa pediu as sugestões e trocou o corretor durante a espera: o
        Palpite foi registrado como o lado da máquina sem entrar no seletor
        (`registra_a_sugestao`), e a troca conta como sobrescrita.
        """
        recomendacao = self.leitura.recomendacao_guardada()
        adotado = self.pipeline.proposta_da_maquina().get(EIXO_DO_CORRETOR)
        if recomendacao is not None and adotado != (recomendacao or {}).get("correction_proposta"):
            return None
        return recomendacao

    def _com_os_lotes(self, frase: str) -> str:
        """A frase de uma sugestão que não terminou, com os lotes que o clique já pôs.

        Os lotes entram no clique, antes de medir, porque o Palpite é medido na partição em que a
        correção vai correr. Uma sugestão que para depois disso (a trava, o Palpite que falha, a
        espera que desiste) deixou os lotes mudados, e a frase precisa dizê-lo para a pessoa não
        ler que nada mudou ao lado de um número que o clique mudou.

        Sem os lotes quando a frase é sobre outra entrada (a matriz trocada, ou a partição
        sugerida que mudou), e é quem chama que o diz: ali o número no campo já não é o do clique.
        """
        lotes = self._lotes_da_sugestao
        base = frase.rstrip()
        if lotes is None:
            return base
        if not base.endswith((".", "!", "?")):
            base += "."
        return f"{base} O número de lotes já está em {lotes}, como o «Diagnóstico» propôs."

    def _aplica_as_sugestoes_da_correcao(self) -> None:
        """Os lotes e o corretor propostos nos campos, e a frase do que entrou ao lado do botão.

        Os lotes outra vez, e a repetição só devolve o valor que o clique pôs.

        O que a pessoa mudou durante a espera fica. O Palpite é independente da aba Correção:
        mexer nela não o larga nem muda o que ele mede. Quem escolheu um corretor ou um número de
        lotes depois de pedir as sugestões escolheu depois; o campo fica como ela deixou, e a
        frase diz o que o Palpite sugeriu, para um segundo clique o seguir.
        """
        corretor_mantido = self.pipeline.foi_tocado(EIXO_DO_CORRETOR)
        # O toque nos lotes conta mesmo sem lotes postos pelo clique: o clique esquece o toque de
        # antes (`_segue_as_sugestoes_da_correcao`), então um toque aqui é de depois.
        lotes_mantidos = self.pipeline.foi_tocado(EIXO_DOS_LOTES)
        lotes = None if lotes_mantidos else self.pipeline.segue_a_proposta(EIXO_DOS_LOTES)
        corretor = None if corretor_mantido else self.pipeline.segue_a_proposta(EIXO_DO_CORRETOR)
        # O campo mantido conta como sobrescrita da sugestão que a pessoa pediu: a sugestão vira
        # o lado da máquina do par sem entrar no campo, e a Correção sai `sobrescreveu_discordando`
        # em vez de gravar a escolha da pessoa na linha da máquina.
        if corretor_mantido:
            self.pipeline.registra_a_sugestao(EIXO_DO_CORRETOR)
        if lotes_mantidos:
            self.pipeline.registra_a_sugestao(EIXO_DOS_LOTES)
        lotes_int = lotes if isinstance(lotes, int) and not isinstance(lotes, bool) else None
        if not (corretor_mantido or lotes_mantidos):
            frase = frase_das_sugestoes_da_correcao(
                None if corretor is None else str(corretor), lotes_int
            )
        else:
            sugerido = self.pipeline.corretor_sugerido
            frase = frase_do_que_voce_mudou_durante_o_palpite(
                None if corretor is None else str(corretor),
                lotes_int,
                corretor_sugerido=None if sugerido is None else str(sugerido),
                corretor_mantido=corretor_mantido,
                lotes_mantidos=lotes_mantidos,
            )
        self._bloco_das_sugestoes(ETAPAS[0]).diz_o_desfecho(frase)

    def _repoe_os_valores_iniciais(self) -> None:
        """O clique de 'Seguir sugestões do app' na Normalização: volta aos valores iniciais.

        A máquina não propõe nenhum eixo desta página (`PROPOSTAS_POR_ROTA`), então o botão volta
        os campos aos valores com que a tela abre, sem travá-los, e diz quantos mudaram.
        """
        mudados = self.pipeline.repoe_os_valores_iniciais()
        self._bloco_das_sugestoes(2).diz_o_desfecho(frase_dos_valores_iniciais(mudados))

    def _esquece_o_desfecho_das_sugestoes(self, etapa: int) -> None:
        """A pessoa mudou um campo sugerido depois do clique: a frase deixou de os descrever.

        Sem isto a frase continuaria a dizer 'corretor LOESS' ao lado de um seletor em SERRF, um
        desfecho velho lido como estado atual. Só as páginas com o bloco recebem o sinal.
        """
        # O toque durante a espera não larga a espera: mexer na Correção enquanto o Palpite roda
        # não o para nem muda o resultado dele. O que a pessoa escolheu fica, e quem o respeita é
        # `_aplica_as_sugestoes_da_correcao` quando o Palpite chega.
        # `getattr` e não o atributo: a troca de matriz chama isto, e os sinais dela estão ligados
        # antes de as páginas das etapas existirem.
        pagina = getattr(self, "_paginas_de_etapa", {}).get(etapa)
        if pagina is not None and pagina.sugestoes is not None:
            pagina.sugestoes.limpa()

    def _palpite_chegou(self, resposta: Resposta) -> None:
        """O desfecho do Palpite, com a espera já encerrada: marca-o em dia e aplica as sugestões.

        A ordem importa: a espera sai antes desta chamada, e a retomada sai de
        `_depois_do_palpite` antes de correr, senão a trava continuaria a ver o Palpite 'a medir'.
        Um Palpite que falha não aplica nada; a frase dele é a que diz o que fazer.

        Só aplica sobre a entrada do clique: com os arquivos, o modo ou os lotes mudados durante a
        medição, o Palpite que chegou é de outra entrada, e o corretor dele não é a sugestão para
        esta.
        """
        corpo = resposta.corpo if isinstance(resposta.corpo, dict) else {}
        if not (resposta.codigo == 200 and "correction_proposta" in corpo):
            self._palpite_em_dia = None
            self._larga_a_espera_das_sugestoes(
                "O Palpite não completou, e o corretor sugerido não entrou: "
                f"{texto_do_desfecho(resposta.codigo, resposta.corpo)}"
            )
            return
        self._palpite_em_dia = self._palpite_pedido
        if self._depois_do_palpite is None:
            return
        if self._identidade_do_palpite() != self._palpite_pedido:
            if self._remede_a_sugestao:
                # A partição sugerida pelo Diagnóstico mudou com este Palpite já a medir: a
                # sugestão é pedida outra vez, na partição de agora. Outra matriz larga a espera
                # antes de chegar aqui.
                self._remede_a_sugestao = False
                self._depois_do_palpite = None
                self._segue_as_sugestoes_da_correcao()
                return
            self._larga_a_espera_das_sugestoes(ENTRADA_MUDOU_DURANTE_O_PALPITE, com_os_lotes=False)
            return
        retoma, self._depois_do_palpite = self._depois_do_palpite, None
        retoma()

    def _larga_a_espera_das_sugestoes(self, frase: str, *, com_os_lotes: bool = True) -> None:
        """As sugestões que esperavam o Palpite não entram, e a tela diz por quê.

        Chamado em todo desfecho que não é Palpite em dia: a resposta de erro, a chamada que não
        produziu desfecho de rota e a espera que desistiu. Sem isto a retomada ficaria guardada, a
        trava recusaria toda correção, e o próximo Palpite pedido à mão aplicaria um clique antigo.

        A frase vai ao lado de 'Seguir sugestões do app', que é onde a pessoa clicou, e o botão
        volta a ligar.
        """
        if self._depois_do_palpite is None:
            return
        self._depois_do_palpite = None
        self._remede_a_sugestao = False
        frase = self._com_os_lotes(frase) if com_os_lotes else frase
        self._bloco_das_sugestoes(ETAPAS[0]).diz_o_desfecho(frase)
        self.aviso_de_desfecho.mostra(frase, titulo=TITULO_DO_PALPITE, tipo="aviso")

    def _esquece_as_amostras_dos_lotes(self, _par: object) -> None:
        """Um par novo invalida a ordem de injeção e os QCs da validação anterior.

        Invalida também o que o diagnóstico anterior propôs. Os cortes são ordens de injeção de
        outra corrida, e mandá-los com o par novo cortaria a corrida em fronteiras que não são
        dela. O número de lotes voltaria ao campo pelo clique de 'Seguir sugestões do app'. E as
        leituras em dia: o Palpite pedido depois reusaria o Diagnóstico sem a sugestão de lotes
        que esta função apaga. Quem para a cadeia em curso é `_confere_a_etiqueta_da_cadeia`, que
        a troca e a remoção chamam depois desta.
        """
        self._geracao_da_leitura += 1
        self._leituras_em_dia.clear()
        self.pipeline.define_amostras(None, None)
        self.pipeline.esquece_as_propostas_do_par()
        # E o Palpite em dia, junto com o corretor que ele sugeriu, que
        # `esquece_as_propostas_do_par` apaga: o mesmo arquivo recolocado reconstruiria a mesma
        # identidade, e o clique aplicaria 'em dia' um Palpite sem corretor.
        self._palpite_em_dia = None

    # ------------------------------------------------------------------
    # A MATRIZ DA TELA
    # ------------------------------------------------------------------

    def _matriz_da_tela(self) -> str | None:
        """A identidade da matriz que a tela mostra agora (o par e o modo), ou `None` sem par.

        A conta é a do disparo, `_matriz_de`: o carimbo da medição, o da curva, o do Palpite e o
        da execução são ela no instante do disparo, e a matriz corrente do painel dos gráficos é
        ela agora. É a mesma que a retomada do Palpite usa (`_identidade_do_palpite`): duas contas
        diferentes fariam uma guarda aceitar o que a outra larga.
        """
        par = self.arquivos.par
        if par is None:
            return None
        return self._matriz_de(par, self.arquivos.modo.currentText())

    def _matriz_de(self, par: Any, modo: str) -> str:
        """`chave_da_entrada` de `par` e `modo`, com `metodo=None`: a conta única, memorizada.

        O modo entra sem os espaços das pontas, como `campos_para_envio` o manda: um espaço no fim
        não muda o que o servidor recebe, e não pode mudar a matriz de nenhuma guarda.

        O memo guarda o próprio objeto do par, e não o `id()`: o par guardado não é coletado,
        então outro par não pode nascer no mesmo endereço e herdar a chave dele. O relógio das
        etapas chama isto a cada meio segundo, a troca do modo a cada tecla, e o `sha256` das
        medidas custa milissegundos.
        """
        modo = modo.strip()
        memo = self._identidade_memo
        if memo is None or memo[0] is not par or memo[1] != modo:
            memo = (
                par,
                modo,
                chave_da_entrada(
                    modo=modo,
                    metodo=None,
                    medidas=par.bytes_das_medidas,
                    lista=par.bytes_da_lista,
                ),
            )
            self._identidade_memo = memo
        return memo[2]

    def _define_a_matriz(self, identidade: str | None) -> bool:
        """Diz ao painel dos gráficos qual é a matriz e, se ela mudou, reescreve os rótulos dele.

        O painel esconde a medição e a curva de outra matriz e repõe as desta, e os rótulos seguem
        a mesma identidade: senão o vencedor de A continuaria escrito ao lado do botão da medição
        depois de as barras de A saírem da figura. De volta a A, as duas metades voltam juntas.
        Devolve o que `define_matriz` devolve.
        """
        if not self.graficos.define_matriz(identidade):
            return False
        self._escreve_o_rotulo_da_medicao(*self.graficos.resposta_da_medicao_na_tela())
        self._escreve_o_rotulo_da_curva(*self.graficos.resposta_da_curva_na_tela())
        # A execução segue a mesma identidade, depois de o painel saber a matriz: a figura reposta
        # é avaliada contra o diagnóstico e o violino da matriz que chega.
        self._repoe_a_execucao(identidade)
        return True

    def _escreve_o_rotulo_da_medicao(self, ha: bool, corpo: Any) -> None:
        """O rótulo da medição a partir da resposta aceita da matriz da tela.

        Uma fonte para as três vezes em que ele se escreve: a chegada aceita, a troca de matriz e
        a volta a ela. Sem resposta, o repouso (`SEM_MEDICAO`); com vencedor, a prosa; senão, a
        recusa, pelo `detail` dela e não pelo dicionário.
        """
        if not ha:
            self.pipeline.mostra_medicao(None)
            return
        if isinstance(corpo, dict) and "vencedor" in corpo:
            self.pipeline.mostra_medicao(corpo)
            return
        detalhe = corpo.get("detail") if isinstance(corpo, dict) else None
        motivo = detalhe if isinstance(detalhe, str) else corpo
        self.pipeline.rotulo_da_medicao.setText(f"A medição não completou: {motivo}")

    def _escreve_o_rotulo_da_curva(self, ha: bool, corpo: Any) -> None:
        """O rótulo do botão da curva a partir da resposta aceita da matriz da tela.

        Vazio sem resposta e com curva (a curva está na figura); com recusa, o `detail` dela.
        """
        if not ha or (isinstance(corpo, dict) and "curva" in corpo):
            self.pipeline.mostra_curva_do_loess("")
            return
        detalhe = corpo.get("detail") if isinstance(corpo, dict) else corpo
        self.pipeline.mostra_curva_do_loess(f"A curva não veio: {detalhe}")

    @staticmethod
    def _diz_que_descartou(
        na_tela: tuple[bool, Any],
        escreve: Callable[[bool, Any], None],
        frase: Callable[[], object],
    ) -> None:
        """A resposta de outra matriz foi descartada: o que o rótulo diz.

        A frase do descarte só aparece quando a matriz da tela não tem resposta guardada. Com uma
        (a de B, reposta na volta a B enquanto a de A corria), as barras de B continuam na figura,
        e a frase ao lado delas diria que foram descartadas: o rótulo fica com a resposta de B.
        """
        if na_tela[0]:
            escreve(*na_tela)
            return
        frase()

    def _segue_a_matriz_da_tela(self, *_sinal: object) -> None:
        """A pessoa escolheu outros arquivos ou mudou o modo: os gráficos passam à matriz da tela.

        O carimbo da medição e o da curva são postos no disparo e comparados com a matriz da tela,
        e não com a do último disparo: medir A, escolher B e a resposta de A chegar não desenha as
        barras de A nem escreve o vencedor dela sob B.

        Ligado a `currentTextChanged`, que no combo editável dispara a cada tecla, e é por isso
        que a troca esconde em vez de apagar: corrigir o modo à mão passa por textos que não são
        modo nenhum, e a volta ao mesmo texto tem de encontrar a medição onde ela estava.
        """
        # A falha de uma etapa e a frase das sugestões eram sobre a matriz de antes, e saem com a
        # troca. Com o clique à espera, quem decide a frase é a chegada do Palpite.
        if self._define_a_matriz(self._matriz_da_tela()):
            self._falhas_das_etapas.clear()
            if self._depois_do_palpite is None:
                self._esquece_o_desfecho_das_sugestoes(ETAPAS[0])
        # E a cadeia das leituras para se a entrada dela já não é a da tela, depois de a frase das
        # sugestões ser apagada, para a frase da parada ficar ao lado do botão. A troca do par
        # chega aqui também, pelo mesmo sinal, depois de `_esquece_as_amostras_dos_lotes`.
        self._confere_a_etiqueta_da_cadeia()

    def _execucao_de_outra_matriz(self) -> bool:
        """Se a execução cujo desfecho chega foi disparada com outra matriz que a da tela.

        Sem carimbo a guarda não corre, como na medição e no Palpite: um desfecho entregue sem
        disparo (o arnês) não tem matriz a que pertencer. Vale também para o `202`: o de outra
        matriz não toca as abas, e o desfecho dela fica guardado (`_guarda_a_execucao`).
        """
        pedida = self._matriz_da_execucao
        return pedida is not None and pedida != self._matriz_da_tela()

    def _palpite_de_outra_matriz(self) -> bool:
        """Se o Palpite que chega foi pedido com outra matriz que a da tela.

        Sem carimbo a guarda não corre, como em `PainelDeGraficos.recebe_medicao`: um desfecho
        entregue sem disparo (o arnês) não tem matriz a que pertencer.
        """
        pedida = self._matriz_do_palpite
        return pedida is not None and pedida != self._matriz_da_tela()

    def _oculta_o_aviso_de_arquivos(self, _par: object) -> None:
        """Completar o par encerra só o aviso de dependência de arquivos."""
        self.aviso_de_desfecho.oculta_da_origem("arquivos")

    def _arquivo_removido(self, chave: str) -> None:
        """Recalcula a validade das etapas no mesmo gesto de desanexar.

        Remover fatores mantém a correção válida; retirar uma metade do par
        invalida também as amostras e os cortes derivados da matriz anterior.
        As execuções salvas e os trabalhos já disparados mantêm seus próprios dados.
        """
        if chave != "fatores":
            self._esquece_as_amostras_dos_lotes(None)
            self._identidade_memo = None
            self.leitura.limpa()
            self.pipeline.mostra_inventario_de_grupos(None)
            self.pipeline.povoa_referencias([])
            self.pipeline.mostra_palpite_da_medicao(None)
            self._define_a_matriz(None)
            self._falhas_das_etapas.clear()
            # Depois de `leitura.limpa`, para a frase da parada ficar no cartão pedido.
            self._confere_a_etiqueta_da_cadeia()
        self._fatores_memo = None
        self._atualiza_as_etapas()

    def _abre_a_figura(self, funcao: str) -> None:
        """Seleciona a figura na aba 'Gráficos' e vai para ela.

        Serve a 'Exportar esta figura em Gráficos', sob a vista de cada aba de etapa, e à curva do
        LOESS que chega.
        """
        if self.graficos.seleciona(funcao):
            self.abas.setCurrentIndex(INDICE_DA_ABA_DOS_GRAFICOS)

    def _etapas_que_a_execucao_escreve(self) -> frozenset[int]:
        """As etapas guardadas a que o desfecho da execução corresponde: as figuras que ele escreve.

        Cada etapa é dona dos dados das próprias figuras. A Análise parte da entrada escolhida e
        escreve só as etapas que ela reproduz (`EstadoDasEtapas.etapas_reproduzidas`): sobre a
        `original`, nenhuma, porque não há correção a mostrar; sobre a `corrigida`, a Correção, se
        a que ela usou é a guardada; sobre a `normalizada`, a Normalização e a Correção de que esta
        partiu, nas mesmas condições. Quem o diz é o estado das etapas, e não o corpo, que tem a
        mesma forma em todos os casos. Sem a Análise em voo, nenhuma.
        """
        return self._etapas.etapas_reproduzidas(ETAPAS[-1])

    def _motivo_para_nao_corrigir(self, chave: str) -> str | None:
        """A frase da recusa, ou `None` quando nenhuma correção está em curso.

        Uma correção por vez: o pipeline, as etapas, o lote dos três corretores e os artefatos
        avulsos corrigem, e dois deles ao mesmo tempo disputam os núcleos.

        - Com o lote vivo, só a perna dele passa, e mesmo ela só quando nenhuma outra correção
          está em espera. A perna seguinte do lote dispara depois de a anterior sair das esperas;
          um clique avulso na mesma chave, com a perna em voo, encontra a espera e é recusado.
        - A medição dos corretores entra nos dois sentidos: ela corre os três corretores nos
          mesmos núcleos.
        - O multiverso entra nos dois sentidos: a correção recusa enquanto o painel dele diz que
          ele corre, e `_dispara_multiverso` recusa enquanto uma correção corre.

        A trava não vê o intervalo entre o `POST` e o primeiro `202`, em que a espera ainda não
        nasceu (as etapas e o Palpite o cobrem por conta própria, abaixo), nem um trabalho de que
        a janela desistiu de ler mas que o servidor ainda corre. E vê com atraso o multiverso que
        terminou sem ninguém ler o progresso: ele continua 'a correr' no painel e trava até a
        leitura seguinte, e a frase da recusa diz isso.
        """
        if self._lote is not None and chave != self._lote.chave:
            return LOTE_EM_CURSO
        # O Palpite trava desde o `POST` (a cadeia cobre o intervalo até o primeiro `202`), e a
        # espera das sugestões trava toda correção menos o próprio Palpite: o clique de 'Seguir
        # sugestões do app' fica à espera antes de a cadeia o disparar, e travá-lo faria a cadeia
        # recusar o dela.
        if self._palpite_em_voo() or (
            self._depois_do_palpite is not None and chave != CHAVE_DO_PALPITE
        ):
            return PALPITE_EM_CURSO
        # O multiverso não entra em `CHAVES_QUE_CORRIGEM` porque nunca é uma espera: o progresso
        # dele é lido a pedido, e quem sabe se ele corre é o painel.
        if self.multiverso.a_correr():
            return MULTIVERSO_EM_CURSO
        if any(em_espera in CHAVES_QUE_CORRIGEM for em_espera in self._esperas):
            return CORRECAO_EM_CURSO
        # Uma etapa em curso também trava, e cobre o intervalo entre o `POST` e o primeiro `202`:
        # a etapa é marcada no clique, antes de a espera nascer.
        if self._etapas.em_curso():
            return CORRECAO_EM_CURSO
        return None

    # ------------------------------------------------------------------
    # O PASSO A PASSO
    # ------------------------------------------------------------------

    def _campos_atuais(self) -> tuple[dict[str, Any], dict[str, str]] | None:
        """O pedido que a tela daria agora e a identidade dos arquivos, ou `None` sem arquivos.

        É o `campos_base` do painel do pipeline, com as escolhas das três abas: cada etapa filtra
        deste dicionário o que a rota dela aceita, e a entrada escolhida substitui o que ela
        herda (`EstadoDasEtapas.campos_da_entrada`).
        """
        par = self.arquivos.par
        if par is None:
            return None
        modo = self.arquivos.modo.currentText()
        escolhas = self.pipeline.escolhas_do_disparo()
        campos = self.pipeline.campos_base(
            modo,
            escolhas["correcao"],
            escolhas["n_lotes"] or LOTES_QUANDO_VAZIO,
            eixos=self.pipeline.eixos_divergentes(),
            rsd_cutoff=escolhas["rsd_cutoff"],
        )
        return campos, self._identidade_dos_arquivos(par, modo)

    def _identidade_dos_arquivos(self, par: Any, modo: str) -> dict[str, str]:
        """A matriz (etapa 1) e o CSV de fatores (etapa 2), por conteúdo, memorizados.

        O memo guarda o próprio objeto, e não o `id()`, como em `_matriz_de`. O relógio chama isto
        a cada meio segundo, e o `sha256` das medidas custa milissegundos.

        A matriz sai de `_matriz_de`, a conta das guardas de matriz, onde o modo entra sem os
        espaços das pontas. Os campos da fotografia continuam com o modo como está no combo
        (`_campos_atuais`).
        """
        identidade = {"matriz@1": self._matriz_de(par, modo)}
        arquivos = self.arquivos.arquivos_da_execucao()
        fatores = arquivos.get("normalization_factors")
        if fatores is not None:
            conteudo: bytes = fatores[1]
            if self._fatores_memo is None or self._fatores_memo[0] is not conteudo:
                self._fatores_memo = (conteudo, hashlib.sha256(conteudo).hexdigest())
            identidade["fatores@2"] = self._fatores_memo[1]
        return identidade

    def _fotos_da_tela(
        self, tela: dict[str, Any], identidade: dict[str, str]
    ) -> tuple[dict[int, Fotografia], dict[int, EntradaDaEtapa]]:
        """A fotografia que a tela daria agora a cada etapa, com a entrada escolhida, e as entradas.

        A foto é a da etapa com a entrada dela (`fotografia_da_etapa`), e não a do pedido montado
        da tela inteira: os campos que a etapa herda vêm da etapa de entrada feita, e mudar os
        controles de outra aba não muda o que ela pede.
        """
        matriz = identidade[IDENTIDADE_DA_MATRIZ]
        entradas = {etapa: self._etapas.entrada(etapa, matriz) for etapa in ETAPAS}
        fotos = {
            etapa: fotografia_da_etapa(tela, etapa, identidade, entradas[etapa].escolhida)
            for etapa in ETAPAS
        }
        return fotos, entradas

    def _atualiza_as_etapas(self) -> tuple[str, ...]:
        """Recalcula o estado das três etapas e o que depende dele em cada aba.

        Liga os botões, a Entrada e os avisos de cada aba, o ponto de cada etapa na barra lateral
        e o frescor das figuras. Devolve os estados, para o arnês afirmar sem ler o rótulo. Sem
        arquivos, a fotografia é vazia: nenhuma etapa feita está em dia, e as páginas mostram o
        estado vazio. Uma etapa não pede a anterior em dia: só uma correção em curso desliga as
        três, e sempre com a frase que diz qual (`pipeline.mostra_etapas` não aceita botão
        desligado sem motivo).
        """
        atual = self._campos_atuais()
        entradas: dict[int, EntradaDaEtapa] | None = None
        fotos: dict[int, Fotografia] = {etapa: () for etapa in ETAPAS}
        if atual is not None:
            fotos, entradas = self._fotos_da_tela(*atual)
        estados = tuple(self._etapas.estado(etapa, fotos[etapa]) for etapa in ETAPAS)
        livre = self._motivo_para_nao_corrigir(CHAVE_DA_ETAPA[1])
        motivos = [
            livre if self._etapas.pode_disparar(etapa) else self._frase_da_etapa_em_curso()
            for etapa in ETAPAS
        ]
        # O frescor das figuras é o da etapa feita, e não o estado da linha: refazer a etapa 1 com
        # os mesmos parâmetros põe-na 'em curso', e a figura dela continua válida enquanto isso. A
        # entrada de cada etapa feita diz qual etapa a frase da figura desatualizada manda refazer.
        self.graficos.define_frescor_das_etapas(
            {etapa: self._etapas.em_dia(etapa, fotos[etapa]) for etapa in (1, 2)},
            entradas={
                etapa: feita.entrada
                for etapa in (1, 2)
                if (feita := self._etapas.feita(etapa)) is not None
            },
        )
        self.pipeline.mostra_etapas(estados, motivos)
        # O ponto de cada etapa na barra lateral vem destes mesmos estados. Sem par, a aba da
        # etapa mostra o estado vazio, e a barra não tem estado a dizer: uma etapa feita antes de
        # o par sair seria 'desatualizada' contra a fotografia vazia.
        barra = BarraLateral.da_janela(self)
        if barra is not None:
            barra.mostra_os_estados(
                {
                    INDICES_DAS_ABAS_DE_ETAPA[etapa]: None if atual is None else estado
                    for etapa, estado in zip(ETAPAS, estados, strict=True)
                }
            )
        for etapa, estado in zip(ETAPAS, estados, strict=True):
            pagina = self._paginas_de_etapa[etapa]
            pagina.mostra_estado(estado)
            if atual is not None and entradas is not None:
                pagina.mostra_entrada(self._vista_da_entrada(etapa, entradas[etapa], atual[1]))
            self._mostra_o_aviso_da_etapa(etapa, estado, fotos, atual, livre)
        # A sugestão ao lado de 'Seguir sugestões do app', na Correção, recalculada a cada volta
        # como a Entrada: a troca do par, do modo, da leitura ou do Palpite aparece na volta
        # seguinte sem gatilho enumerado, e o bloco só a reescreve quando ela muda.
        self._bloco_das_sugestoes(ETAPAS[0]).mostra_a_indicacao(self._indicacao_da_correcao())
        matriz = None if atual is None else atual[1][IDENTIDADE_DA_MATRIZ]
        for etapa, resultado in self._resultados.items():
            feita = self._etapas.feita(etapa)
            resultado.mostra_etapa(
                self._motivo_do_download(etapa, matriz),
                None if feita is None else feita.campos,
            )
        return estados

    def _indicacao_da_correcao(self) -> str:
        """O que 'Seguir sugestões do app' aplicaria na Correção, numa frase.

        Diz só qual corretor e quantos lotes são indicados, e sob a mesma condição do clique. Os
        lotes são os que o Diagnóstico deixou guardados, que o clique põe no campo. O corretor
        guardado só entra com o Palpite em dia para esta entrada, a mesma condição em que
        `_segue_as_sugestoes_da_correcao` o aplica sem medir de novo. Fora dela o clique remede, e
        o Palpite novo pode sugerir outro: mostrar o guardado prometeria o que o clique talvez não
        faça. Um Diagnóstico com outro número muda a partição, e a frase fica só com os lotes.
        """
        em_dia = (
            self._palpite_em_dia is not None
            and self._palpite_em_dia == self._identidade_do_palpite()
        )
        return frase_da_indicacao(
            self.pipeline.corretor_sugerido if em_dia else None, self.pipeline.lotes_sugeridos
        )

    def _motivo_do_download(self, etapa: int, matriz: str | None) -> str | None:
        """Por que o download do Resultado da `etapa` não liga, ou `None` quando ele liga.

        O download é o da etapa feita sobre a matriz carregada, a mesma condição que faz dela uma
        entrada das etapas seguintes (`EstadoDasEtapas.ausencia` daquela entrada). Sem a etapa
        feita, a frase diz o gesto que a faz; feita sobre outra matriz, diz isso e o gesto.
        """
        entrada = next(nome for nome, dona in ETAPA_DA_ENTRADA.items() if dona == etapa)
        feita = self._etapas.feita(etapa)
        if feita is None or matriz is None:
            return MOTIVOS_SEM_ETAPA[etapa]
        ausencia = self._etapas.ausencia(entrada, matriz)
        if ausencia is None:
            return None
        return (
            f"{com_maiuscula(ausencia)}: {MOTIVOS_SEM_ETAPA[etapa][0].lower()}"
            + (MOTIVOS_SEM_ETAPA[etapa][1:])
        )

    def _vista_da_entrada(
        self, etapa: int, entrada: EntradaDaEtapa, identidade: dict[str, str]
    ) -> VistaDaEntrada:
        """O que a seção 'Entrada' da `etapa` mostra, a partir das entradas que existem.

        Com uma só, a frase diz qual é e por que não há outra; com duas ou três, cada uma é nomeada
        pelo que é ('Corrigida - LOESS', 'Normalizada (sem correção)'). A escolha que deixou de
        existir e o aviso dos zeros são os dois avisos da Entrada.
        """
        if etapa == ETAPAS[0]:
            return VistaDaEntrada(FRASE_DA_ENTRADA_DA_CORRECAO)
        matriz = identidade[IDENTIDADE_DA_MATRIZ]
        nomes = {
            disponivel: com_maiuscula(
                self._etapas.nome_da_entrada(disponivel, corretores=NOMES_DOS_CORRETORES)
            )
            for disponivel in entrada.disponiveis
        }
        if len(entrada.disponiveis) == 1:
            ausencias = tuple(
                frase
                for derivada in ENTRADAS_DA_ETAPA[etapa][1:]
                if (frase := self._etapas.ausencia(derivada, matriz)) is not None
            )
            frase = frase_da_entrada_unica(etapa, ausencias)
        else:
            frase = PERGUNTAS_DA_ENTRADA[etapa]
        mais = entrada.mais_tratada
        sem_imputacao = etapa == ETAPAS[-1] and not self._etapas.passou_pela_imputacao(
            entrada.escolhida
        )
        return VistaDaEntrada(
            frase,
            opcoes=tuple(nomes.items()),
            escolhida=entrada.escolhida,
            mais_tratada=None if mais is None else (mais, nomes[mais]),
            volta=None
            if entrada.motivo_da_volta is None
            else com_maiuscula(f"{entrada.motivo_da_volta}."),
            imputacao=aviso_sem_imputacao(entrada.escolhida) if sem_imputacao else None,
        )

    def _mostra_o_aviso_da_etapa(
        self,
        etapa: int,
        estado: str,
        fotos: dict[int, Fotografia],
        atual: tuple[dict[str, Any], dict[str, str]] | None,
        livre: str | None,
    ) -> None:
        """O aviso fixo da aba da `etapa`, ou nenhum quando o que ela mostra vale.

        Vermelho com a falha; amarelo com o resultado fora de dia, dela ou da entrada dela. O botão
        do amarelo leva à etapa mais a montante fora de dia (`EstadoDasEtapas.etapa_a_refazer`):
        na própria, ele a refaz, e só quando a trava deixa (senão a frase basta e o botão da etapa
        diz por que espera); noutra, ele abre a aba dela. Durante a corrida da própria etapa não
        há aviso, porque o resultado que ele diria fora de dia está a ser refeito.
        """
        pagina = self._paginas_de_etapa[etapa]
        self._alvos_dos_avisos.pop(etapa, None)
        falha = self._falhas_das_etapas.get(etapa)
        if falha is not None:
            pagina.mostra_aviso(falha, nivel="critico", titulo=TITULOS_DA_FALHA[etapa])
            return
        alvo = (
            None
            if atual is None or estado == "em_curso"
            else self._etapas.etapa_a_refazer(etapa, fotos)
        )
        feita = self._etapas.feita(etapa)
        if alvo is None or atual is None or feita is None:
            pagina.limpa_aviso()
            return
        if alvo == etapa:
            frase = frase_da_etapa_fora_de_dia(
                etapa,
                entrada=feita.entrada,
                matriz_mudou=feita.matriz != atual[1][IDENTIDADE_DA_MATRIZ],
                so_a_entrada_refeita=self._etapas.motivos(etapa, fotos[etapa])
                == {"entrada_refeita"},
            )
            acao = TEXTOS_DE_REFAZER[etapa] if livre is None else None
        else:
            aba = ABAS[INDICES_DAS_ABAS_DE_ETAPA[alvo]].titulo
            frase = frase_da_entrada_fora_de_dia(etapa, feita.entrada, alvo, aba)
            acao = f"Ir para «{aba}»"
        if acao is not None:
            self._alvos_dos_avisos[etapa] = alvo
        pagina.mostra_aviso(frase, nivel="aviso", acao=acao)

    def _age_no_aviso(self, etapa: int) -> None:
        """O botão do aviso amarelo: refaz a etapa, ou abre a aba da mais a montante fora de dia."""
        alvo = self._alvos_dos_avisos.get(etapa)
        if alvo is None:
            return
        if alvo == etapa:
            self._dispara_etapa(etapa)
            return
        self.abas.setCurrentWidget(self._paginas_de_etapa[alvo])

    def _escolhe_a_entrada(self, etapa: int, entrada: str) -> None:
        """Guarda a entrada escolhida no seletor da aba da `etapa`; ela vale enquanto existir.

        Uma entrada que deixou de existir entre o desenho e o clique não é escolhida: o seletor se
        redesenha com as que existem, e a escolha anterior fica.
        """
        matriz = self._matriz_da_tela()
        if matriz is not None:
            with contextlib.suppress(EntradaIndisponivelError):
                self._etapas.escolhe_entrada(etapa, entrada, matriz)
        self._atualiza_as_etapas()

    def _frase_da_etapa_em_curso(self) -> str:
        """Por que os botões das etapas estão desligados enquanto uma delas corre.

        Mais exata que a frase da correção em curso (`CORRECAO_EM_CURSO`), que a trava daria aqui:
        diz qual etapa corre e onde pará-la. O nome do painel vem de `PAINEIS`, e não é digitado,
        porque é o título que a pessoa procura na barra lateral.
        """
        etapa = self._etapas.em_curso()[0]
        trabalhos = next(p.titulo for p in PAINEIS if p.chave == CHAVE_DOS_TRABALHOS)
        return (
            f"A etapa {etapa}, {NOMES_DAS_ETAPAS[etapa]}, está em curso — aguarde ela terminar, ou "
            f"pare-a em «{trabalhos}»."
        )

    def _segue_os_arquivos_nas_etapas(self, *_: object) -> None:
        """As páginas de etapa no formulário com o par, e no estado vazio sem ele.

        O estado vazio diz qual arquivo falta. Ligado aos sinais do painel de arquivos; ver
        `__init__`.
        """
        faltam = self.arquivos.metades_que_faltam
        for pagina in self._paginas_de_etapa.values():
            pagina.mostra_arquivos(faltam)

    def showEvent(self, evento: QShowEvent) -> None:  # noqa: N802 - nome do Qt
        """Liga o relógio das etapas e atualiza já. Ver o comentário em `__init__`."""
        super().showEvent(evento)
        self._relogio_das_etapas.start()
        self._atualiza_as_etapas()

    def hideEvent(self, evento: QHideEvent) -> None:  # noqa: N802 - nome do Qt
        """Para o relógio das etapas: janela escondida não tem botão a habilitar."""
        super().hideEvent(evento)
        self._relogio_das_etapas.stop()

    def _recusa_a_etapa(self, etapa: int, texto: str) -> None:
        """Mostra a recusa no aviso de desfecho e no aviso vermelho da aba da etapa.

        Uma recusa que não aparece ao lado do botão recusado é indistinguível de botão morto. A
        faixa vermelha fica logo acima do botão, na seção da etapa; a nota da etapa não a repete,
        para a mesma frase não aparecer duas vezes ao lado do mesmo botão.
        """
        self.rotulo_de_desfecho.setText(texto)
        self._falhas_das_etapas[etapa] = texto
        self._atualiza_as_etapas()

    def _dispara_etapa(self, etapa: int) -> None:
        """Um botão do passo a passo: dispara a etapa sobre a entrada escolhida.

        As etapas 1 e 2 são prévias sem estado: reenviam os arquivos e os campos, e nada gravam. A
        etapa 3 é o `/execucoes`, que grava o registro. Os campos saem da entrada
        (`EstadoDasEtapas.campos_da_entrada`): os da própria etapa vêm da tela, e os que ela herda
        vêm da etapa feita que a entrada é (ou neutros, na original), e não dos controles das
        outras abas. A Correção leva o par de decisão da tela, que as entradas derivadas dela levam
        à execução.

        Nenhuma etapa delega nem espera o Palpite: a Correção leva o corretor do seletor, e as
        outras o da Correção feita, ou nenhum.

        A fotografia guardada é a da mesma tela de onde sai o pedido, lida uma vez: uma segunda
        leitura abriria uma janela em que as duas divergem.
        """
        chave = CHAVE_DA_ETAPA.get(etapa, CHAVE_DA_EXECUCAO)
        motivo = self._motivo_para_nao_corrigir(chave)
        if motivo is not None:
            self._recusa_a_etapa(etapa, motivo)
            return
        atual = self._campos_atuais()
        if atual is None:
            # Sem arquivos: `_pedido_de_arquivos` escreve a frase de sempre no aviso de desfecho, e
            # a página da etapa já mostra o estado vazio que diz qual arquivo falta.
            self._pedido_de_arquivos()
            return
        tela, identidade = atual
        try:
            if etapa == ETAPAS[0]:
                # A linha da máquina do par, no clique da Correção: é ela que as entradas derivadas
                # desta Correção levam à execução, e a máquina pode propor outra coisa depois.
                tela = {
                    **tela,
                    **self.pipeline.proposta_do_pedido(
                        tela,
                        self.pipeline.eixos_divergentes(),
                        self._recomendacao_adotada(),
                    ),
                }
            elif etapa == 2:
                self.pipeline.confere_a_combinacao()
        except (PropostaAusenteError, CombinacaoProibidaError) as erro:
            self._recusa_a_etapa(etapa, str(erro))
            return
        matriz = identidade[IDENTIDADE_DA_MATRIZ]
        entrada = self._etapas.entrada(etapa, matriz).escolhida
        campos = self._etapas.campos_da_entrada(etapa, entrada, tela, matriz)
        pedido = self._pedido_de_arquivos()
        if pedido is None:
            return
        # O arquivo de fatores só viaja para quem normaliza (`/execucoes` também o declara). A
        # Análise sobre a 'normalizada' refaz a Normalização feita, e por isso leva os fatores que
        # ela enviou, e não os da tela de agora (ver `EtapaFeita.fatores`).
        if etapa == ETAPAS[0]:
            arquivos = pedido[0]
        elif etapa == ETAPAS[-1] and entrada == "normalizada":
            arquivos = self._com_os_fatores_da_normalizacao(pedido[0])
        else:
            arquivos = self.arquivos.arquivos_da_execucao()
        self.para_a_espera(chave)
        self._falhas_das_etapas.pop(etapa, None)
        # A matriz do disparo, como o carimbo da execução: a prévia que chegar fica com ela, e a
        # volta a ela a repõe por cima de uma execução guardada mais velha.
        self._matriz_da_previa[etapa] = self._matriz_da_tela()
        self._etapas.disparou(
            etapa,
            fotografia_da_etapa(tela, etapa, identidade, entrada),
            campos=campos,
            entrada=entrada,
            fatores=arquivos.get("normalization_factors") if etapa == 2 else None,
        )
        if etapa == ETAPAS[-1]:
            # A linha do corretor no resultado e o progresso dizem o corretor que a Análise leva:
            # o da Correção feita, ou nenhum, e nunca uma delegação.
            self.pipeline.anota_o_pedido(str(campos.get("correction", "none")), delegou=False)
            # O pedido fica para o resumo metodológico, e a matriz para a guarda de outra matriz.
            self._pedido_da_execucao = dict(campos)
            self._matriz_da_execucao = self._matriz_da_tela()
        # A nota anterior (o progresso da corrida passada) sai: o estado 'em curso' e a barra já
        # dizem que a etapa foi enviada, e o progresso dela escreve aqui.
        self.pipeline.anota_etapa(etapa, "")
        self._atualiza_as_etapas()
        self.chama(
            "POST",
            ROTA_DA_ETAPA.get(etapa, ROTA_DA_EXECUCAO),
            chave=chave,
            arquivos=arquivos,
            campos=campos_da_rota(campos, etapa),
        )

    def _etapa_terminou(self, etapa: int, resposta: Resposta) -> None:
        """O desfecho de uma prévia, com a espera já encerrada.

        A Correção guarda o corretor efetivo que voltou: é ele que a entrada 'corrigida' leva
        adiante, e não uma delegação, que mediria o Palpite outra vez. A falha vira o aviso
        vermelho da aba, com a causa crua (`texto_da_resposta`), e a etapa volta ao estado de
        antes do disparo.

        A nota da etapa sai nos dois desfechos: na conclusão, o estado 'em dia', a barra cheia e a
        frase do Resultado já dizem onde olhar; na falha, a faixa vermelha diz a causa, logo acima
        do botão. O progresso que a nota mostrava durante a corrida deixou de valer nos dois.
        """
        corpo = resposta.corpo
        self.pipeline.anota_etapa(etapa, "")
        if resposta.codigo == 200 and isinstance(corpo, dict):
            efetivo = corpo.get("correcao") if etapa == ETAPAS[0] else None
            self._etapas.concluiu(etapa, correcao=None if efetivo is None else str(efetivo))
            self._falhas_das_etapas.pop(etapa, None)
            # O frescor antes dos dados, para a figura redesenhar uma vez, já em dia. E a entrada
            # da etapa feita, porque a Normalização só escreve as figuras da Correção quando
            # partiu da Correção guardada.
            self._atualiza_as_etapas()
            feita = self._etapas.feita(etapa)
            entrada: Entrada = "original" if feita is None else feita.entrada
            # A prévia fica guardada com a matriz do disparo, a ordem de chegada e a entrada da
            # etapa feita: é o que a volta a essa matriz repõe em volta da execução guardada.
            self._previas[etapa] = _PreviaGuardada(
                etapa, corpo, self._matriz_da_previa.get(etapa), self._proxima_ordem(), entrada
            )
            self.graficos.recebe_etapa(etapa, corpo, entrada=entrada)
            return
        self._etapas.falhou(etapa)
        self._falhas_das_etapas[etapa] = texto_da_resposta(resposta)
        self._atualiza_as_etapas()

    def _dispara_medicao(self) -> None:
        """`POST /matrizes/medir-corretor`: a recomendação medida do corretor.

        Só os campos que a rota aceita: `measurements`, `sample_list`, `mode`, `orientation`,
        `n_sampled_features`, `paralelo`, `n_batches` e `batch_cuts`, e não `method` nem os eixos
        da grade. Reaproveitar `campos_do_artefato` mandaria campos que a rota não declara.

        Os lotes viajam para a medição ser sobre a partição da correção: com lote único, o
        vencedor medido podia ser de outra configuração que a corrigida. O teste que guarda os
        campos desta rota é
        `test_d376_o_botao_da_medicao_dispara_a_rota_certa_com_os_campos_que_ela_aceita`.

        A espera anterior é encerrada antes de disparar, como em `_gera_artefato`: duas cadeias de
        leitura sobre a mesma chave produzem rajada.
        """
        # A medição entra na trava. A recusa vem antes de `_pedido_de_arquivos`, que chama
        # `graficos.define_matriz`: uma medição recusada não pode trocar a matriz das figuras. A
        # frase vai para o rótulo da medição, onde o profissional olha depois de clicar, e para o
        # aviso de desfecho, como as outras recusas.
        motivo = self._motivo_para_nao_corrigir(CHAVE_DA_MEDICAO)
        if motivo is not None:
            self.pipeline.rotulo_da_medicao.setText(motivo)
            self.rotulo_de_desfecho.setText(motivo)
            return
        par = self._pedido_de_arquivos()
        if par is None:
            return
        arquivos, campos = par
        # A caixa dos núcleos importa aqui mais que em qualquer rota: são 21 ajustes de SERRF e
        # três corretores, e o tempo conta em horas.
        campos["paralelo"] = self.pipeline.paralelo.isChecked()
        # O intervalo por barra viaja sempre, marcado ou não: o falso é o padrão da rota, e
        # mandá-lo explícito faz o pedido dizer o que a tela mostra. Só aqui; o Palpite não o lê.
        campos["intervalo_por_barra"] = self.pipeline.intervalo_por_barra.isChecked()
        # Os lotes da correção, e os mesmos: o número vem do campo de lotes, com o vazio traduzido
        # pela constante que o rótulo promete, e os cortes, da mesma regra que o pedido da
        # execução usa (`cortes_em_uso`). Com lote de menos de três QCs, a rota recusa no disparo.
        escolhas = self.pipeline.escolhas_do_disparo()
        n_lotes = escolhas["n_lotes"] or LOTES_QUANDO_VAZIO
        campos["n_batches"] = n_lotes
        cortes = self.pipeline.cortes_em_uso(n_lotes)
        if cortes is not None:
            campos["batch_cuts"] = json.dumps(list(cortes))
        # O carimbo da matriz medida vem depois de `_pedido_de_arquivos`, que chama
        # `graficos.define_matriz`: a identidade vigente já é a desta medição. Com ele,
        # `recebe_medicao` descarta uma resposta que chegue depois de o profissional trocar os
        # arquivos, em vez de desenhar as barras de A sob a execução de B.
        self.graficos.carimba_a_medicao()
        self.para_a_espera(CHAVE_DA_MEDICAO)
        self.pipeline.rotulo_da_medicao.setText("Medindo... o primeiro corretor foi disparado.")
        self.chama(
            "POST", ROTA_DA_MEDICAO, chave=CHAVE_DA_MEDICAO, arquivos=arquivos, campos=campos
        )

    def _dispara_curva_do_loess(self, feature: str) -> None:
        """`POST /matrizes/curva-do-loess`: a curva de uma feature e o envelope sem um QC.

        Os lotes e a semente são os da correção desta janela, para a curva desenhada ser a que a
        correção usaria: o número de lotes e os cortes saem das mesmas regras do pedido da execução
        (`LOTES_QUANDO_VAZIO`, `cortes_em_uso`), e a semente é `SEMENTE_DA_JANELA`.

        O carimbo da matriz vem depois de `_pedido_de_arquivos`, que diz ao painel dos gráficos
        qual é a matriz corrente, como na medição: uma curva que chegue depois de o profissional
        trocar os arquivos é descartada, em vez de desenhada sob a matriz nova.
        """
        par = self._pedido_de_arquivos()
        if par is None:
            return
        arquivos, campos = par
        escolhas = self.pipeline.escolhas_do_disparo()
        n_lotes = escolhas["n_lotes"] or LOTES_QUANDO_VAZIO
        campos["feature"] = feature
        campos["n_batches"] = n_lotes
        campos["seed"] = SEMENTE_DA_JANELA
        cortes = self.pipeline.cortes_em_uso(n_lotes)
        if cortes is not None:
            campos["batch_cuts"] = json.dumps(list(cortes))
        # O corretor que vai correr é o do seletor, porque a janela não delega: a figura só diz
        # que a curva corrigiu o sinal quando ele é o LOESS.
        self.graficos.carimba_a_curva(self.pipeline.corretor.currentText())
        self.chama(
            "POST",
            ROTA_DA_CURVA_DO_LOESS,
            chave=CHAVE_DA_CURVA_DO_LOESS,
            arquivos=arquivos,
            campos=campos,
        )

    def _atualiza_os_trabalhos(self) -> None:
        """Uma leitura de `GET /trabalhos/{id}` por trabalho conhecido, para o painel 3.

        Uma chamada por trabalho e não uma rota de índice, pela razão que a docstring de
        `registra_trabalho` dá: `GET /trabalhos` exporia a lista de retidos numa API sem
        autenticação, e o painel só precisa dos trabalhos desta janela.

        A chave da chamada carrega o prefixo do painel, para `_encaminha` saber que a resposta é
        para o painel 3 e não para o painel que disparou aquele trabalho. Sem o prefixo, uma
        atualização de `corrigir` cairia no ramo dos artefatos e reabriria a espera.

        Sem trabalho nenhum, escreve a frase de repouso e não chama nada: botão que recusa em
        silêncio é indistinguível de botão morto.
        """
        conhecidos = self.trabalhos_conhecidos()
        if not conhecidos:
            self.trabalhos.mostra({})
            return
        for chave, identificador in sorted(conhecidos.items()):
            self.chama(
                "GET",
                rota_do_progresso(identificador),
                chave=chave_do_pedido(CHAVE_DOS_TRABALHOS, chave),
            )

    def _le_conclusao_do_multiverso(self, identificador: str) -> None:
        """`GET /multiversos/{id}/resultado`: a conclusão do multiverso.

        Sem arquivos, como `_le_progresso`: o identificador está na rota, e vem do painel, que é
        quem tem o campo.
        """
        # Os três limiares do veredito viajam na query e vêm do painel, pela mesma divisão: quem
        # tem o controle é o painel, e a janela pergunta. Vazio quando ninguém mexeu nos três, e
        # aí a rota usa os padrões.
        self.chama(
            "GET",
            rota_do_resultado(identificador),
            chave=CHAVE_DO_MULTIVERSO,
            campos=dict(self.multiverso.campos_da_conclusao()),
        )

    def _cancela_o_trabalho(self, chave: str) -> None:
        """`POST /trabalhos/{id}/cancelar`: a parada de um trabalho longo.

        A chave vem do painel e o identificador vem daqui, a mesma divisão de `_le_progresso`: o
        painel sabe qual linha o usuário escolheu, e a janela tem o mapa de chave para
        identificador.

        Trabalho desconhecido escreve e não chama. Pode acontecer entre a leitura da lista e o
        clique (poda, ou reinício da API), e um `POST` sobre um identificador que a janela não tem
        produziria um `404` que o usuário não teria como ligar ao que fez.
        """
        identificador = self.trabalhos_conhecidos().get(chave)
        if identificador is None:
            self.trabalhos.mostra_cancelamento(chave, 404, {"detail": "trabalho desconhecido"})
            self.trabalhos.mostra(self.trabalhos_conhecidos())
            return
        self.chama(
            "POST",
            rota_de_cancelamento(identificador),
            chave=chave_do_pedido(CHAVE_DO_CANCELAMENTO_DE_TRABALHO, chave),
        )

    def _dispara_multiverso(self, escolhas: dict[str, Any]) -> None:
        """`POST /multiversos`: dispara o multiverso a partir da janela.

        Junta os lados que só a janela vê: o par de arquivos e o modo vêm do painel 1, os
        corretores e o par de grupos vêm do painel 5. `_pedido_de_arquivos` recusa antes de
        chamar, com a mesma mensagem dos outros slots.

        O `estudo` vem do painel 3. Com o histórico em disco, ele é a única coluna que distingue
        duas execuções com o mesmo modo e os mesmos arquivos. É lido do painel 3 e não pedido de
        novo no painel 5: quem já digitou o nome não o digita outra vez, e dois campos com o mesmo
        rótulo divergiriam. Vazio não é mandado, e a rota decide o padrão (`"sem_identificacao"`,
        o mesmo de `janela/pipeline.py`).

        Entra na trava de uma correção por vez, nos dois sentidos: o multiverso corrige com os
        corretores da grade nos mesmos núcleos, e o SERRF em vários núcleos, quando é parado,
        encerra os processos de todo trabalho do programa. A recusa vem antes de
        `_pedido_de_arquivos`, como na medição, e a frase vai para o rótulo do progresso do
        multiverso, onde quem clicou olha, e para o aviso de desfecho.
        """
        motivo = self._motivo_para_nao_corrigir(CHAVE_DO_MULTIVERSO)
        if motivo is not None:
            self.multiverso.rotulo_de_progresso.setText(motivo)
            self.rotulo_de_desfecho.setText(motivo)
            return
        pedido = self._pedido_de_arquivos()
        if pedido is None:
            return
        arquivos, campos = pedido
        estudo = self.pipeline.estudo.text().strip()
        self.chama(
            "POST",
            ROTA_DO_DISPARO,
            chave=CHAVE_DO_DISPARO,
            arquivos=arquivos,
            campos={**campos, **escolhas, **({"estudo": estudo} if estudo else {})},
        )

    def _le_progresso_do_multiverso(self, identificador: str) -> None:
        """`GET /multiversos/{id}`: o sinal de vida, lido a pedido.

        Leitura manual e não laço automático. O laço existe para os artefatos, onde o trabalho
        dura segundos e o usuário espera olhando; a grade padrão do multiverso leva da ordem de
        quinze minutos no maior dos modos, e um laço de 2 s faria centenas de leituras. Quem quer
        saber clica, e o botão de cancelar torna a espera suportável sem o laço.
        """
        self.chama("GET", rota_do_progresso_do_multiverso(identificador), chave=CHAVE_DO_PROGRESSO)

    def _le_historico_de_multiversos(self, limite: int) -> None:
        """`GET /multiversos/historico`: o que sobreviveu ao fechamento do programa.

        A única rota do painel que não depende de haver execução nesta sessão: as outras falam de
        um identificador que este processo criou, e esta fala do banco, que existe entre sessões.
        """
        self.chama("GET", rota_do_historico(limite), chave=CHAVE_DO_HISTORICO)

    def _abre_multiverso_gravado(self, identificador: str) -> None:
        """`GET /multiversos/historico/{id}`: a conclusão gravada em disco, não a da memória.

        `_le_conclusao_do_multiverso` lê o registro deste processo, que some ao fechar, e antes
        disso quando a poda aperta. Esta lê o banco, e responde por execuções de qualquer sessão.
        """
        # Os limiares viajam aqui também: `le_conclusao_gravada` recalcula o veredito no serviço,
        # e sem eles esta leitura reclassificaria pelos padrões enquanto os campos na tela mostram
        # outro corte.
        self.chama(
            "GET",
            rota_da_gravada(identificador),
            chave=CHAVE_DA_GRAVADA,
            campos=dict(self.multiverso.campos_da_conclusao()),
        )

    def _cancela_multiverso(self, identificador: str) -> None:
        """`POST /multiversos/{id}/cancelar`: a parada cooperativa.

        `POST` sem corpo, porque o que se pede é uma transição e o identificador está na rota.
        """
        self.chama("POST", rota_do_cancelamento(identificador), chave=CHAVE_DO_CANCELAMENTO)

    def _escreve_no_destino(self, chave: str, conteudo: bytes | None) -> None:
        """Escreve o artefato na pasta que o clique escolheu.

        Há duas saídas silenciosas, ambas legítimas: sem destino pendente, o clique não escolheu
        pasta (por exemplo, um 'Baixar de novo'); sem bytes, a rota respondeu sem produzir
        artefato, e `recebe` já pôs a causa no rótulo, que uma mensagem de escrita apagaria.

        O nome inclui o método, e o do diálogo não. `salva` sugere `{modo}_{chave}.csv`, e lá o
        profissional vê e corrige o nome; aqui não há diálogo, então sem o método dois corretores
        no mesmo modo escreveriam um por cima do outro em silêncio.
        """
        destino = self._destinos.pop(chave, None)
        if destino is None or not conteudo:
            return
        alvo = destino.pasta / nome_do_arquivo(destino.modo, chave, destino.metodo)
        etapa = ARTEFATOS_DA_JANELA[chave].etapa
        try:
            alvo.write_bytes(conteudo)
        except OSError as erro:
            # A chave sai do registro do escrito, e não fica a apontar para a escrita anterior: o
            # irmão do corte iria parar ao lado de um CSV que não é o desta exportação.
            self._escritos.pop(chave, None)
            frase = ARTEFATO_SEM_ESCRITA.format(alvo=alvo, causa=erro)
            self.rotulo_de_desfecho.setText(frase)
            self._resultados[etapa].mostra_escrita(chave, frase)
            return
        self._escritos[chave] = alvo
        self.rotulo_de_desfecho.setText(ARTEFATO_ESCRITO.format(alvo=alvo))
        resumo = self._registra_o_arquivo(chave, alvo)
        # O resumo ao lado do arquivo, quando a caixa estava marcada no clique. A frase vai ao
        # rótulo do Resultado, que é onde a pessoa clicou.
        frase = SO_O_ARQUIVO_ESCRITO.format(alvo=alvo)
        if destino.com_resumo and resumo is not None:
            ao_lado = _resumo_ao_lado(alvo)
            causa = self._salva_o_resumo(ao_lado, resumo)
            frase = (
                RESUMO_ESCRITO.format(alvo=alvo, resumo=ao_lado.name)
                if causa is None
                else RESUMO_SEM_ESCRITA.format(alvo=alvo, causa=causa)
            )
        self._resultados[etapa].mostra_escrita(chave, frase)

    def _salva_o_resumo(self, destino: Path, resumo: ResumoMetodologico) -> str | None:
        """Grava o resumo em `destino`; devolve a causa da falha, ou `None` quando gravou.

        A falha vira frase, e o arquivo da matriz continua salvo: num executável `--windowed` não
        há `stderr`, e a exceção dentro do slot seria engolida. A causa vai também ao aviso de
        desfecho, que abre por cima da página. O texto é o que o bloco do resumo mostra, o mesmo
        objeto, e não uma segunda montagem dele.
        """
        try:
            destino.write_text(resumo.texto, encoding="utf-8")
        except OSError as erro:
            causa = f"{type(erro).__name__}: {erro}"
            self.rotulo_de_desfecho.setText(f"O resumo dos métodos não foi salvo — {causa}")
            return causa
        return None

    def _registra_o_arquivo(self, chave: str, alvo: Path) -> ResumoMetodologico | None:
        """O arquivo escrito entra no resumo metodológico, com o pedido que o produziu.

        Sem pedido guardado, nada entra: o resumo diz só o que foi registrado, e um arquivo cujo
        pedido esta janela não montou (um arnês que escreve direto) não tem o que dizer.

        Cada Resultado tem o seu resumo, que descreve só os arquivos que aquela aba escreveu. O
        arquivo leva a rota que o produziu e o que ele é. Devolve o resumo mostrado, que é o que
        vai ao `.txt`.
        """
        pedido = self._pedidos_de_artefato.get(chave)
        if pedido is None:
            return None
        artefato = ARTEFATOS_DA_JANELA[chave]
        arquivo = ArquivoExportado(
            alvo.name,
            artefato.base,
            dict(pedido),
            self._aceites_de_artefato.get(chave),
            o_que_e=self._descricoes_de_artefato.get(chave),
        )
        arquivos = self._arquivos_do_resumo_da_etapa[artefato.etapa]
        arquivos.append(arquivo)
        resumo = resumo_da_exportacao(arquivos)
        self._resultados[artefato.etapa].resumo.mostra(resumo)
        return resumo

    def _esquece_o_resumo_do_gesto(self, chave: str) -> None:
        """Um gesto novo (um clique com pasta escolhida, ou um lote) recomeça o resumo da aba dele.

        O resumo descreve os arquivos de um gesto só.
        """
        self._arquivos_do_resumo_da_etapa[ARTEFATOS_DA_JANELA[chave].etapa] = []

    def _inicia_lote(self, chave: str) -> None:
        """Pede a pasta uma vez e dispara o primeiro dos três corretores.

        No servidor, `_chave_do_trabalho` já inclui `"correction": method` nos campos, então três
        `POST /matrizes/exportar` com métodos diferentes produzem três identificadores
        coexistentes. No painel, porém, `define_entrada` descarta por identidade, e a identidade
        inclui o método: as três gerações se apagariam umas às outras. Por isso a saída é o disco,
        e cada arquivo é escrito assim que chega.

        A pasta é pedida antes de correr, e não ao fim: os três corretores levam minutos, e três
        diálogos ao longo deles fariam o profissional voltar ao computador três vezes.
        """
        if self._lote is not None:
            self.rotulo_de_desfecho.setText(LOTE_EM_CURSO)
            return
        motivo = self._motivo_para_nao_corrigir(chave)
        if motivo is not None:
            self.rotulo_de_desfecho.setText(motivo)
            return
        pedido = self._pedido_de_arquivos()
        if pedido is None:
            return
        pasta = self._escolhe_pasta()
        if pasta is None:
            self.rotulo_de_desfecho.setText(LOTE_CANCELADO)
            return
        # O modo sai do pedido e não do painel dos artefatos, cujo `_modo` é privado: alcançá-lo
        # seria uma segunda fonte da mesma coisa. Desempacotado, e não `pedido[1][...]`, porque
        # `test_toda_base_de_leitura_esta_classificada` exige base de leitura nomeada.
        _arquivos_do_lote, campos_do_lote = pedido
        # O lote é um gesto: o resumo metodológico descreve os arquivos dele, e só eles.
        self._esquece_o_resumo_do_gesto(chave)
        # A caixa do resumo é a do Resultado de onde o lote saiu.
        etapa = ARTEFATOS_DA_JANELA[chave].etapa
        self._lote = LoteDeExportacao(
            pasta=pasta,
            modo=str(campos_do_lote["mode"]),
            restantes=list(CORRETORES),
            escritos=[],
            corretor_original=self.pipeline.corretor.currentText(),
            chave=chave,
            com_resumo=self._resultados[etapa].salvar_o_resumo.isChecked(),
        )
        self._avanca_lote()

    def _avanca_lote(self) -> None:
        """Põe o combo no próximo corretor e dispara a exportação. Encerra quando não há próximo.

        Move o combo em vez de sobrepor o campo no `POST`, por causa da identidade: ela sai do
        corretor do combo em `_gera_artefato`, e um `POST` com um método que o combo não mostra
        faria os bytes chegarem sob a identidade errada. O combo movido também mostra ao
        profissional qual dos três está a correr.
        """
        lote = self._lote
        if lote is None:  # pragma: no cover - só chamado com lote vivo
            return
        if not lote.restantes:
            self.pipeline.corretor.setCurrentText(lote.corretor_original)
            self._lote = None
            nomes = ", ".join(caminho.name for caminho in lote.escritos)
            frase = LOTE_CONCLUIDO.format(quantos=len(lote.escritos), pasta=lote.pasta, nomes=nomes)
            self.rotulo_de_desfecho.setText(_com_os_cortes_que_faltam(lote, frase))
            self._fecha_o_lote_no_resultado(lote, frase)
            return
        self.pipeline.corretor.setCurrentText(lote.restantes[0])
        # A perna recusada aborta o lote (outra correção pode ter entrado em espera desde o diálogo
        # da pasta, ou entre duas pernas). Sem isto o lote ficaria vivo sem trabalho no servidor, e
        # todo disparo de outra chave receberia `LOTE_EM_CURSO`, com a janela travada sem nada que
        # o dissesse. Abortar é a mesma regra de `_lote_recebe`.
        if not self._gera_artefato(lote.chave):
            motivo = self.rotulo_de_desfecho.text()
            self._aborta_o_lote(lote, LOTE_RECUSADO.format(metodo=lote.restantes[0], motivo=motivo))

    def _aborta_o_lote(self, lote: LoteDeExportacao, frase: str) -> None:
        """O lote parou: o combo volta, o lote solta, e a frase vai ao desfecho e ao Resultado.

        As três formas de abortar passam por aqui: perna sem artefato, escrita que falhou e perna
        recusada antes de sair. O aviso de desfecho some; sem a frase no Resultado, ele
        continuaria a dizer 'em dia'.
        """
        self.pipeline.corretor.setCurrentText(lote.corretor_original)
        self._lote = None
        com_os_cortes = _com_os_cortes_que_faltam(lote, frase)
        self.rotulo_de_desfecho.setText(com_os_cortes)
        self._resultados[ARTEFATOS_DA_JANELA[lote.chave].etapa].mostra_falha_do_lote(com_os_cortes)
        self._lote_encerrado = (lote, frase, True)
        self._fecha_os_cortes_do_lote()

    def _lote_do_pedido(self, pedido: str) -> LoteDeExportacao | None:
        """O lote (vivo, ou encerrado com relatórios por voltar) que pediu este relatório.

        Tira o pedido dos pendentes dele: o relatório chegou. `None` quando nenhum dos dois o tem,
        o que só acontece se outro lote encerrou entretanto e tomou o lugar do encerrado.
        """
        encerrado = None if self._lote_encerrado is None else self._lote_encerrado[0]
        for lote in (self._lote, encerrado):
            if lote is not None and pedido in lote.relatorios_pendentes:
                lote.relatorios_pendentes.discard(pedido)
                return lote
        return None

    def _corte_da_perna_falhou(
        self, chave: str, perna: Path, falha: str, dono: LoteDeExportacao | None
    ) -> None:
        """O registro do corte de uma perna do lote não se gravou.

        A frase vai ao rótulo na hora, e isso não basta: com o lote vivo, a leitura de progresso da
        perna seguinte reescreve o rótulo antes de alguém o ler. O nome do arquivo fica no lote
        que pediu o relatório (o vivo, ou o encerrado cujos relatórios ainda voltam), e
        `_fecha_os_cortes_do_lote` o diz quando o último deles chegar.

        Sem lote a quem pertencer (`dono` é `None`: outro lote encerrou entretanto e tomou o lugar
        deste), a frase vai ao aviso de desfecho, como erro, para não ficar só num rótulo que outro
        escreve.
        """
        frase = CORTE_DA_PERNA_SEM_ESCRITA.format(nome=perna.name, falha=falha)
        self._painel_do_artefato(chave).mostra_escrita(chave, frase)
        if dono is None:
            self.aviso_de_desfecho.mostra(frase, titulo=TITULO_DO_LOTE, tipo="erro")
            return
        dono.sem_o_corte.append(perna.name)

    def _fecha_os_cortes_do_lote(self) -> None:
        """Lote encerrado e último relatório de perna chegado: diz os registros que faltaram.

        A frase vai aqui porque este é o último evento do lote. O relatório da última perna volta
        depois de o lote encerrar, e `recebe_relatorio` troca a frase de fim pelo corte daquela
        perna; o aviso de desfecho é recolhido por cada resposta `2xx`. Dita antes deste ponto, a
        falha ficaria por baixo de uma escrita posterior.

        Sem relatório pendente, roda já no fim do lote. Sem registro em falta, só solta o lote: o
        rótulo fica com o que o último relatório escreveu.

        Os pendentes são os do lote, e não o registro global das pernas: uma entrada que nunca
        volte, de um lote anterior, prenderia o fecho de todos os lotes seguintes.
        """
        encerrado = self._lote_encerrado
        if encerrado is None or encerrado[0].relatorios_pendentes:
            return
        self._lote_encerrado = None
        lote, frase, abortado = encerrado
        if not lote.sem_o_corte:
            return
        final = _com_os_cortes_que_faltam(lote, frase)
        resultado = self._resultados[ARTEFATOS_DA_JANELA[lote.chave].etapa]
        if abortado:
            resultado.mostra_falha_do_lote(final)
        else:
            resultado.mostra_escrita(lote.chave, final)
        self.aviso_de_desfecho.mostra(final, titulo=TITULO_DO_LOTE, tipo="erro")

    def _fecha_o_lote_no_resultado(self, lote: LoteDeExportacao, frase: str) -> None:
        """O lote do Resultado terminou: a frase vai ao rótulo dele, e o resumo, à pasta.

        Com a caixa marcada, vai um resumo só, ao lado dos três arquivos: o lote é um gesto, e o
        resumo dele já descreve os três, um bloco por arquivo com o seu pedido
        (`resumo_da_exportacao`). O nome é o do lote sem o método: `<modo>_<chave>-metodos.txt`.
        """
        etapa = ARTEFATOS_DA_JANELA[lote.chave].etapa
        arquivos = self._arquivos_do_resumo_da_etapa[etapa]
        if lote.com_resumo and arquivos:
            ao_lado = _resumo_ao_lado(lote.pasta / nome_do_arquivo(lote.modo, lote.chave))
            causa = self._salva_o_resumo(ao_lado, resumo_da_exportacao(arquivos))
            frase = (
                LOTE_COM_RESUMO.format(frase=frase, resumo=ao_lado.name)
                if causa is None
                else LOTE_SEM_RESUMO.format(frase=frase, causa=causa)
            )
        self._resultados[etapa].mostra_escrita(lote.chave, _com_os_cortes_que_faltam(lote, frase))
        self._lote_encerrado = (lote, frase, False)
        self._fecha_os_cortes_do_lote()

    def _lote_recebe(self, conteudo: bytes | None, pedido_do_relatorio: str | None = None) -> None:
        """Escreve o arquivo do corretor corrente e avança. Falha no meio aborta o lote.

        Seguir depois de uma perna falhar entregaria dois arquivos numa pasta pedida para três,
        sem nada na tela a dizer qual faltou. O rótulo nomeia o corretor que falhou, porque é ele
        que decide se vale repetir: `serrf` custa minutos e `none` custa zero.

        `pedido_do_relatorio` é o identificador da perna cujo relatório do corte `_encaminha`
        acabou de pedir, e `None` quando nenhum foi pedido: o formato do laboratório não corta, e
        o arnês que entrega bytes sem pedido não encadeia nada.
        """
        lote = self._lote
        if lote is None:  # pragma: no cover - só chamado com lote vivo
            return
        metodo = lote.restantes.pop(0)
        if not conteudo:
            self._aborta_o_lote(lote, LOTE_ABORTADO.format(metodo=metodo))
            return
        # O nome sai da chave do lote e não de um literal: com dois formatos, os dois lotes
        # escreveriam os mesmos nomes na mesma pasta.
        alvo = lote.pasta / nome_do_arquivo(lote.modo, lote.chave, metodo)
        try:
            alvo.write_bytes(conteudo)
        except OSError as erro:
            # O relatório desta perna já foi pedido, e não tem CSV ao lado do qual ir: `_encaminha`
            # o descarta quando ele voltar.
            if pedido_do_relatorio is not None:
                self._relatorios_descartados.add(pedido_do_relatorio)
            self._aborta_o_lote(lote, LOTE_SEM_ESCRITA.format(alvo=alvo, causa=erro))
            return
        # O lote também registra o escrito, para o relatório do corte de cada perna ir parar ao
        # lado do CSV dela. O registro é pelo identificador do trabalho da perna, que é o que a
        # resposta do relatório traz. `_escritos`, que é só do clique avulso, não serviria: ele é
        # lido junto com a guarda de bytes do painel, e a perna seguinte apaga esses bytes antes
        # de o relatório da anterior voltar.
        if pedido_do_relatorio is not None:
            self._pernas_por_pedido[pedido_do_relatorio] = alvo
            lote.relatorios_pendentes.add(pedido_do_relatorio)
        lote.escritos.append(alvo)
        self._registra_o_arquivo(lote.chave, alvo)
        self._avanca_lote()

    def _gera_artefato(self, chave: str) -> bool:
        """Um download do Resultado de uma etapa. Devolve se o pedido saiu.

        A rota sai de `PassoDoFluxo.rota`: o mapa de `artefatos.py` já liga cada chave à rota que
        a serve, e um dicionário aqui seria uma segunda fonte.

        `define_entrada` é chamado antes da rota, para o painel saber de qual entrada são os bytes
        que vão chegar. A identidade é `fluxo.chave_da_entrada` (`modo`, `metodo` e o digest dos
        bytes): só com o modo, trocar o par de arquivos ou o corretor no mesmo modo não
        descartaria nada.

        O lote precisa de saber se a perna foi recusada, para abortar em vez de ficar vivo sem
        trabalho.

        O download leva a etapa feita em três lugares: nos campos do pedido, na identidade que
        `define_entrada` dá aos bytes e no nome do arquivo (`DestinoPendente.metodo`). Os três
        saem da Correção ou da Normalização feita, e não dos controles: o arquivo baixado é o que
        a etapa mostrou, e o combo pode ter mudado desde então. Sem etapa feita sobre a matriz
        carregada, o pedido não sai, e a frase diz o gesto que o faz sair. A perna de um lote é a
        exceção: o lote corre os três corretores pelo combo, e a etapa feita não é dele.
        """
        artefato = ARTEFATOS_DA_JANELA.get(chave)
        if artefato is None:
            self.rotulo_de_desfecho.setText(f"Artefato desconhecido: {chave}")
            return False
        passo = artefato.passo
        painel = self._painel_do_artefato(chave)
        # Um clique avulso durante o lote é recusado. Se saísse, este método encerraria a espera
        # da perna e repostaria; o `200` da perna chegaria depois e seria descartado como de um
        # trabalho substituído, e o lote ficaria vivo para sempre, travando todo lote novo.
        # Cancelar o lote seria pior: o profissional perderia minutos de SERRF sem o ter pedido.
        motivo = self._motivo_para_nao_corrigir(chave)
        if motivo is not None:
            self.rotulo_de_desfecho.setText(motivo)
            # A recusa vai também ao lado do botão que a pessoa clicou, no Resultado.
            painel.rotulos[chave].setText(motivo)
            return False
        # Nenhum download espera o Palpite: o do Resultado leva o corretor efetivo da etapa feita,
        # e a perna de um lote não delega, porque o lote existe para gerar os três corretores, e
        # com `auto` no pedido a rota trocaria o `method` de cada perna pelo que a delegação
        # escolhesse.
        de_um_lote = self._lote is not None and self._lote.chave == chave
        # A etapa feita vem antes de qualquer efeito: sem ela o pedido não sai, e nada (o destino,
        # a espera, os bytes guardados) muda por um clique recusado.
        feita = None if de_um_lote else self._da_etapa_feita(artefato)
        if not de_um_lote and feita is None:
            return False
        pedido = self._pedido_de_arquivos()
        if pedido is None:
            return False
        arquivos, campos = pedido
        # O corretor que produz os bytes: o da etapa feita, e o do combo na perna de um lote.
        metodo = self.pipeline.corretor.currentText() if feita is None else feita[1]
        # O arquivo de fatores sobe para quem declara o campo: os downloads da Normalização levam
        # os eixos, com `sample_normalization` (que pode ser `supplied_factors`), e sem o arquivo a
        # rota recusaria com 409. Os da Correção não declaram o campo, e mandá-lo afirmaria uma
        # influência que a rota não tem. Os fatores são os da Normalização feita, e não os da
        # tela: com `supplied_factors` o resultado depende dos bytes, e trocar o arquivo depois de
        # normalizar faria o arquivo baixado divergir do que a etapa mostrou.
        if "normalization_factors" in artefato.campos:
            arquivos = self._com_os_fatores_da_normalizacao(arquivos)
        par = self.arquivos.par
        if par is None:  # pragma: no cover - `_pedido_de_arquivos` já teria devolvido `None`
            return False
        # O destino é recolhido, não pedido: quem perguntou a pasta foi o botão, em
        # `PainelDeArtefatos.pede_e_gera`, no instante do clique. Essa separação mantém `gerar` com
        # um significado só, e um arnês que o emita direto não abre diálogo nenhum.
        #
        # `modo` e `metodo` ficam congelados agora, e não relidos na chegada: entre o despacho e
        # os bytes podem passar minutos, e o combo pode mudar. Reler o corretor na chegada poria o
        # nome do corretor de agora num arquivo produzido pelo de antes.
        pasta_escolhida = painel.destinos.pop(chave, None)
        if pasta_escolhida is not None:
            self._destinos[chave] = DestinoPendente(
                pasta=pasta_escolhida,
                modo=str(campos["mode"]),
                metodo=metodo,
                com_resumo=self._resultados[artefato.etapa].salvar_o_resumo.isChecked(),
            )
        painel.define_entrada(
            identidade=chave_da_entrada(
                modo=campos["mode"],
                metodo=metodo,
                medidas=par.bytes_das_medidas,
                lista=par.bytes_da_lista,
            ),
            modo=campos["mode"],
        )
        # O clique encerra a espera anterior e recomeça a espera. Sem isto, o `202` do `POST`
        # (a rota devolve o mesmo identificador para a mesma entrada) cairia no ramo de uma espera
        # que corre e contaria como leitura, e `leituras` passaria a significar duas coisas. Nada
        # é recalculado, porque a rota reclama o mesmo trabalho.
        self.para_a_espera(chave)
        # Os campos do painel do pipeline viajam no pedido; sem `method`, a rota corrigiria com o
        # padrão dela, `none`, qualquer que fosse o corretor escolhido.
        #
        # A recusa local é traduzida e não propagada: num `--windowed` o `stderr` é `None`, e uma
        # exceção que subisse do slot deixaria o botão indistinguível de morto. A mensagem vai ao
        # rótulo de desfecho, que guarda o último e abre o aviso, e ao rótulo do artefato, onde a
        # ação foi e onde o texto fica à vista.
        if feita is not None:
            campos_do_painel = campos_do_download(
                artefato, feita[0], metodo=metodo, max_features=self.pipeline.max_features.value()
            )
        else:
            try:
                campos_do_painel = self.pipeline.campos_do_artefato(chave)
            except CombinacaoProibidaError as erro:
                self.rotulo_de_desfecho.setText(str(erro))
                painel.rotulos[chave].setText(str(erro))
                return False
        # O pedido de cada artefato fica guardado, como o da execução: os bytes que voltam não
        # dizem com que corretor, semente e lotes foram feitos. O aceite anterior sai, para a
        # delegação de outro disparo não ser lida como a deste. Um clique avulso com pasta
        # escolhida é um gesto novo, e o resumo passa a descrever os arquivos dele.
        enviados = {**campos, **campos_do_painel}
        self._pedidos_de_artefato[chave] = enviados
        self._aceites_de_artefato.pop(chave, None)
        self._descreve_o_download(artefato, None if feita is None else feita[0], metodo)
        if pasta_escolhida is not None and not de_um_lote:
            self._esquece_o_resumo_do_gesto(chave)
        self.chama("POST", passo.rota, chave=chave, arquivos=arquivos, campos=enviados)
        return True

    def _com_os_fatores_da_normalizacao(self, de_agora: Mapping[str, Any]) -> dict[str, Any]:
        """Os arquivos `de_agora` sem os fatores da tela, e com os que a Normalização feita enviou.

        Os bytes, e não o nome: o resultado de `supplied_factors` depende do conteúdo, e é ele que
        a etapa feita guardou (`EtapaFeita.fatores`). Sem Normalização feita, ou sem fatores nela,
        o pedido sai sem o arquivo, como a etapa saiu.
        """
        # `arquivos` é o `files` do multipart do pedido, a base que o portão de contrato classifica.
        arquivos = {
            nome: valor for nome, valor in de_agora.items() if nome != "normalization_factors"
        }
        feita = self._etapas.feita(2)
        if feita is not None and feita.fatores is not None:
            arquivos["normalization_factors"] = feita.fatores
        return arquivos

    def _da_etapa_feita(self, artefato: ArtefatoDaJanela) -> tuple[Mapping[str, Any], str] | None:
        """Os campos e o corretor da etapa feita que o download do Resultado leva, ou `None`.

        Com `None`, a frase já está escrita ao lado do botão e no aviso de desfecho; é a de
        `_motivo_do_download`, a mesma que o botão desligado mostra. O corretor é o que produziu a
        matriz da etapa: na Correção, o efetivo que ela devolveu (e não `auto`, que mediria o
        Palpite de novo); na Normalização, o da entrada dela, que é `none` sobre a original.
        """
        etapa = artefato.etapa
        feita = self._etapas.feita(etapa)
        motivo = self._motivo_do_download(etapa, self._matriz_da_tela())
        if motivo is not None or feita is None or feita.campos is None:
            frase = motivo or MOTIVOS_SEM_ETAPA[etapa]
            self.rotulo_de_desfecho.setText(frase)
            self._resultados[etapa].rotulos[artefato.chave].setText(frase)
            return None
        corretor = feita.correcao if etapa == ETAPAS[0] else feita.campos.get("correction")
        return feita.campos, str(corretor)

    def _descreve_o_download(
        self, artefato: ArtefatoDaJanela, campos: Mapping[str, Any] | None, metodo: str
    ) -> None:
        """O que o arquivo é, para o resumo dele: o do mapa, e na Normalização a matriz que entrou.

        A entrada sai dos campos da Normalização feita, e não do nome que a tela dá hoje à matriz
        corrigida: refazer a Correção depois de normalizar mudaria aquele nome, e o resumo diria
        um corretor que o arquivo não usou.
        """
        descricao = artefato.descricao
        if campos is not None and artefato.etapa != ETAPAS[0]:
            sobre = (
                "a matriz sem correção"
                if metodo == "none"
                else f"a matriz corrigida — {NOMES_DOS_CORRETORES.get(metodo, metodo)}"
            )
            descricao = f"{descricao}, sobre {sobre}"
        self._descricoes_de_artefato[artefato.chave] = descricao

    def _vai_ao_preparo(self) -> None:
        """Mostra a aba 'Preparar o estudo', onde mora o campo do modo."""
        for indice in range(self.abas.count()):
            if isinstance(self.abas.widget(indice), AbaDePreparo):
                self.abas.setCurrentIndex(indice)
                return

    def _pedido_de_arquivos(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """`(arquivos, campos)` do painel 1, ou `None` com a mensagem já escrita.

        Um lugar só, para a mesma falta ter uma mensagem só: todos os slots de disparo precisam do
        par, e `arquivos_para_envio` levanta `ValueError` quando não há.

        É também aqui que o painel dos gráficos aprende qual matriz é a corrente, pela mesma
        razão: é o funil por onde todo disparo passa para se comprometer com um par de arquivos, e
        um disparo novo não nasce sem a chamada. Acontece no disparo e não na chegada da resposta:
        a medição do corretor demora, e entre o clique e o `200` o profissional pode trocar os
        arquivos; ler a identidade na chegada tomaria a matriz nova como a medida. `metodo=None`,
        porque a medição corre os três corretores, e trocar o corretor entre medir e executar não
        pode descartá-la (o argumento de `CHAVES_SEM_METODO`).

        O painel aprende a matriz também na troca, por `_segue_a_matriz_da_tela`; no disparo fica
        o carimbo, que diz a que matriz a resposta pertence. Este funil continua a definir a
        matriz porque o par pode ter entrado sem passar pela escolha.
        """
        self.aviso_de_desfecho.oculta()
        try:
            arquivos, campos = (
                self.arquivos.arquivos_para_envio(),
                self.arquivos.campos_para_envio(),
            )
        except ValueError as erro:
            sem_modo = isinstance(erro, ModoAusenteError)
            if sem_modo:
                # A recusa leva ao campo: a aba do preparo, com 'Opções avançadas' aberto pelo
                # painel. Recusar numa aba e deixar o campo noutra, fechado, apontaria para um
                # lugar que ninguém vê.
                self._vai_ao_preparo()
            self.aviso_de_desfecho.mostra(
                str(erro),
                titulo="Falta o modo do estudo" if sem_modo else "Escolha os arquivos do estudo",
                tipo="aviso",
                origem="arquivos",
            )
            return None
        # A mesma conta de `_matriz_da_tela`, que os carimbos usam (aqui o par existe, senão os
        # dois construtores acima levantariam), e pelo funil que apaga os rótulos da medição e da
        # curva quando a matriz muda.
        self._define_a_matriz(self._matriz_da_tela())
        return arquivos, campos

    def chama(
        self,
        metodo: str,
        rota: str,
        *,
        chave: str,
        arquivos: dict[str, Any] | None = None,
        campos: dict[str, Any] | None = None,
    ) -> None:
        """Enfileira a chamada e devolve imediatamente; o resultado vem por sinal.

        `chave` é obrigatória e não tem padrão: `""` cobriria 'não passei chave' e 'a chave é
        vazia' com o mesmo valor, e quem filtra por chave não distinguiria os dois.
        """
        self.enfileira(
            lambda: envia(self._transporte, metodo, rota, arquivos=arquivos, campos=campos),
            chave=chave,
        )

    def enfileira(self, chamada: Callable[[], Resposta], *, chave: str) -> None:
        """O ponto de entrada da piscina, exposto para o arnês injetar chamada lenta ou que levanta.

        É público porque o arnês precisa afirmar que o laço de eventos continua a responder
        durante a chamada, e sem injetar demora teria de usar uma rota real e lenta.
        """
        base, pedido = parte_a_chave(chave)
        # O Palpite fica fora da etiqueta de geração: ele já tem identidade e substituição
        # próprias, e o `pedido` dele é o identificador do trabalho em curso. Etiquetado, o `422`
        # síncrono seria lido como resposta de outro trabalho e o `202` de depois de uma troca de
        # arquivo seria descartado, deixando `_depois_do_palpite` guardado e toda correção
        # seguinte recusada.
        if base in dict(ROTAS_DA_LEITURA) and base != CHAVE_DO_PALPITE and pedido is None:
            chave = chave_do_pedido(base, self._etiqueta_da_leitura())
        trabalho = _Trabalho(chamada, chave)
        trabalho.sinais.concluido.connect(self._recebe)
        trabalho.sinais.falhou.connect(self._falha)
        self._piscina.start(trabalho)

    def _etiqueta_da_leitura(self) -> str:
        """A etiqueta com que uma leitura sai: a geração dos anexos e a matriz da tela.

        A geração sobe a cada troca e remoção de arquivo, e é ela que descarta a validação
        atrasada depois de remover e reanexar os mesmos arquivos: a matriz é a mesma, e o pedido é
        outro. A matriz traz o modo, que a geração não conta: o diagnóstico usa um limiar por modo
        quando nenhum é mandado, e a validação confere o modo contra os nomes das amostras.

        O modo vai pelo valor e não por outra geração, porque o combo emite a cada tecla: corrigir
        o modo à mão e voltar ao mesmo texto não pode descartar a leitura em voo da mesma matriz.
        Um `@` no modo não parte a chave: `parte_a_chave` corta na primeira ocorrência, e a base
        vem antes dela.
        """
        return f"entrada-{self._geracao_da_leitura}-{self._matriz_da_tela()}"

    def _leitura_desatualizada(self, chave: str, pedido: str | None) -> bool:
        """Uma leitura iniciada antes da troca dos anexos não repõe os dados antigos.

        Nem uma iniciada antes da troca do modo: a etiqueta leva a matriz da tela, e não só a
        geração (`_etiqueta_da_leitura`).
        """
        return (
            chave in dict(ROTAS_DA_LEITURA)
            and pedido is not None
            and pedido.startswith("entrada-")
            and pedido != self._etiqueta_da_leitura()
        )

    def espera_a_piscina(self, teto_ms: int = 10_000) -> bool:
        """Espera a piscina esvaziar. Devolve se esvaziou, e o desfecho não é ignorável."""
        return bool(self._piscina.waitForDone(teto_ms))

    def drena_a_piscina(self, teto_ms: int = 10_000) -> bool:
        """Descarta o que está enfileirado e espera pelo que está a correr. Devolve se esvaziou.

        `clear()` antes de esperar: `waitForDone` espera pelo enfileirado mais o que está em
        execução, e uma fila de trabalhos longos estouraria o teto. O que se descarta nunca
        começou, então não há resultado a perder.

        É um método próprio, e não um parâmetro de `espera_a_piscina`, porque os usos são opostos:
        aquele serve ao arnês para esperar trabalho que ele quer ver terminar, e este é o
        encerramento, onde descartar é correto.

        `arranque.py` o chama quando a janela fecha, e as esperas param aqui também: senão um
        relógio agendado dispararia depois do descarte e enfileiraria uma leitura nova durante o
        desligamento. Um relógio agendado é, como o enfileirado, trabalho que ainda não começou.
        Quem impede a leitura é o `pop` de `_esperas` (ver `para_a_espera`). `espera_a_piscina`
        não para as esperas, para o arnês poder observar o laço.
        """
        self.para_as_esperas()
        # O painel das vias tem piscina própria e, no modo do MetaboAnalystR, um R em voo. Ele
        # cancela o R (o Job Object leva a árvore de processos) e só então drena; sem isto, o
        # destrutor da piscina dele esperaria sem teto por um R que ninguém mandou parar.
        vias_ok = self.vias.encerra(teto_ms)
        self._piscina.clear()
        return self.espera_a_piscina(teto_ms) and vias_ok

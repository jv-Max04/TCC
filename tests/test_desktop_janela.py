"""A janela Qt do aplicativo nativo.

Estes testes correm no portão de cada commit.
`QT_QPA_PLATFORM=offscreen` faz o Qt subir sem servidor gráfico, e o `Transporte` injetado é o
`TestClient` do FastAPI, que fala com a app ASGI **em processo**. Sem as duas coisas, verificar a
janela exigiria display e `uvicorn`, e os critérios sairiam do portão.

**O congelamento NÃO é verificado aqui e não pode ser:** PyInstaller no WSL produz binário Linux,
que parece funcionar e não serve. O que o verifica é `scripts/congela_windows.ps1`, à mão no
Windows.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Final, get_args

import pytest

from tests.pedido_da_tela import campos_da_tela
from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

# **O `skip` vem ANTES de qualquer import de Qt**, e a ordem é o conteúdo: um
# `from PySide6 import …` no topo levantaria `ModuleNotFoundError` na COLETA, que é erro e não
# `skip` — não casa regime nenhum, e faria a política reprovar a sessão pelo motivo errado.
#
# **A frase fica AQUI e não na guarda**, porque `tests/skip_policy.py` extrai a mensagem por AST
# colhendo constantes de string: `pytest.skip(variavel)` não entrega literal nenhum, e o regime
# declarado reprovaria por *"padrão sem emissão"*. A guarda decide **quem falta**; este sítio
# é que a fraseia.
_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a decisão é exercitada em `tests/test_qt_guard.py`, com planta
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()

import httpx2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from refino.api.app import build_app  # noqa: E402
from refino.desktop import arranque  # noqa: E402
from refino.desktop.etapas import (  # noqa: E402
    CHAVE_DA_ETAPA,
    ETAPAS,
    IDENTIDADE_DA_MATRIZ,
    fotografia_da_etapa,
)
from refino.desktop.janela.arquivos import (  # noqa: E402
    ROTULO_DA_LISTA,
    ROTULO_DAS_MEDIDAS,
    SEM_PAR,
    TIPO_DO_ARQUIVO,
    PainelDeArquivos,
    ParDeArquivos,
)
from refino.desktop.janela.artefatos import (  # noqa: E402
    ARTEFATO_DO_RELATORIO,
    ARTEFATOS,
    ARTEFATOS_DA_JANELA,
    CHAVE_DA_ALTERNATIVA,
    CHAVES_DO_RELATORIO,
    CHAVES_NA_ORDEM,
    FORMATO_METABOANALYST,
    MARCA_DE_GERADO,
    PASSOS_DOS_ARTEFATOS,
    SEM_ARTEFATO,
    SEM_BYTES,
    SEM_DESTINO,
    PainelDeArtefatos,
    nome_do_arquivo,
    nome_do_arquivo_irmao,
    pasta_por_dialogo,
    removidas_do_relatorio,
    texto_do_decorrido,
    texto_do_tamanho,
)
from refino.desktop.janela.componentes import com_maiuscula  # noqa: E402
from refino.desktop.janela.figuras import PainelDeFigura  # noqa: E402
from refino.desktop.janela.graficos import (  # noqa: E402
    PainelDeGraficos,
    frase_da_etapa_desatualizada,
)
from refino.desktop.janela.leitura import (  # noqa: E402
    ROTAS_DA_LEITURA,
    ROTULOS_DA_LEITURA,
    SEM_LEITURA,
    PainelDeLeitura,
    rotulo_do_eixo,
    texto_da_recomendacao,
    texto_do_diagnostico,
    texto_dos_problemas,
)
from refino.desktop.janela.multiverso import (  # noqa: E402
    CHAVE_DO_MULTIVERSO,
    COLUNAS,
    SEM_CONCLUSAO,
    SEM_IDENTIFICADOR_DO_MULTIVERSO,
    PainelDeMultiverso,
    rota_do_resultado,
)
from refino.desktop.janela.pipeline import (  # noqa: E402
    CAMPO_DA_ANALISE,
    CAMPO_DO_INVENTARIO,
    CAMPO_DOS_CLASSIFICADORES,
    CAMPOS_POR_ARTEFATO,
    CHAVE_DO_INICIO,
    CHAVE_DO_LOTE_CORRIGIDA,
    CHAVE_DO_LOTE_METABO,
    CORRETORES,
    EIXOS_DA_GRADE,
    EM_CURSO,
    FIXOS_DECLARADOS,
    INVENTARIO_EM_REPOUSO,
    INVENTARIO_SEM_GRUPO,
    LOTES_QUANDO_VAZIO,
    PROGRESSO_EM_REPOUSO,
    RESULTADO_EM_REPOUSO,
    ROTA_DA_EXECUCAO,
    SEM_EXECUCAO,
    SEM_RESSALVA,
    TEXTO_DO_ESTADO_DA_ETAPA,
    CombinacaoProibidaError,
    GrupoNoInventario,
    PainelDePipeline,
    PropostaAusenteError,
    e_corpo_de_aceite,
    inventario_do_corpo,
    rota_de_cancelamento,
    rota_do_progresso,
    rota_do_relatorio,
    texto_da_leitura,
    texto_do_resultado,
)
from refino.desktop.janela.principal import (  # noqa: E402
    ABAS,
    ARTEFATO_ESCRITO,
    ARTEFATO_SEM_ESCRITA,
    CHAVE_DA_EXECUCAO,
    CHAVE_DA_MEDICAO,
    CORRECAO_EM_CURSO,
    LOTE_ABORTADO,
    LOTE_CANCELADO,
    LOTE_EM_CURSO,
    LOTE_RECUSADO,
    PAINEIS,
    SEM_DESFECHO,
    SEPARADOR_DO_PEDIDO,
    TETO_DA_ESPERA_MS,
    TITULO,
    Aba,
    JanelaPrincipal,
    _EsperaDoArtefato,
    _Sinais,
    _Trabalho,
    chave_do_pedido,
    despacha,
    parte_a_chave,
    texto_da_resposta,
    texto_do_desfecho,
)
from refino.desktop.janela.trabalhos import (  # noqa: E402
    CHAVE_DOS_TRABALHOS,
    NAO_LIDO,
    NOMES_DOS_TRABALHOS,
    ROTULOS_DOS_ESTADOS,
    SEM_TRABALHOS,
    PainelDeTrabalhos,
)
from refino.domain.analysis import (  # noqa: E402
    PADRAO_DA_ANALISE,
    analise_para_json,
)
from refino.domain.labels import ClassifierKind  # noqa: E402
from refino.interface import cliente  # noqa: E402
from refino.interface.cliente import Resposta  # noqa: E402
from refino.interface.decorrido import (  # noqa: E402
    DepositoEmMemoria,
    agora,
    carimba_disparo,
    decorrido_da_leitura,
)
from refino.interface.fluxo import (  # noqa: E402
    CHAVES_DE_ARTEFATO,
    CONTROLES_DE_SOBRESCRITA,
    FAMILIAS_DE_ANALISE,
    MODOS_SUGERIDOS,
    PONTOS_DE_DECISAO_AUTOMATICA,
    ROTULOS_DE_DEFEITO,
    SEM_MEDICAO_DA_CLASSIFICACAO,
    chave_da_entrada,
    prosa_da_confianca,
    prosa_da_confianca_do_diagnostico,
)
from refino.io.writers import write_removidas  # noqa: E402
from tests.conftest import measurements_path, sample_list_path  # noqa: E402
from tests.contrato_de_rotas import chamadas_diretas_ao_transporte  # noqa: E402

FONTES_DA_JANELA = sorted(
    Path(__file__).resolve().parents[1].glob("src/refino/desktop/janela/*.py")
)
PIPELINE_FONTE = (
    Path(__file__).resolve().parents[1] / "src" / "refino" / "desktop" / "janela" / "pipeline.py"
)

# Um `422` cujo `detail` é **string** com a frase da nossa guarda: é o discriminador MECÂNICO de
# `cliente._forma_do_422`, e o que separa `422:conteudo` (causa `dado`) de `422:forma` (`usuario`).
CORPO_DE_422_DE_CONTEUDO: Any = {"detail": "o arquivo foi recusado pela validação"}


# ---------------------------------------------------------------------------
# O portão de AST — função de módulo, para os outros testes importarem ESTA
# ---------------------------------------------------------------------------


def envios_sincronos(fonte: str) -> list[str]:
    """As chamadas a `envia` que **não** estão dentro de um `lambda`, uma por linha.

    `cliente.envia` é síncrono e as rotas variam três ordens de grandeza —
    0,19 s para os 6,08 MB do `pos_hilic` contra minutos para uma execução. Chamado do laço de
    eventos, congela a janela; e janela congelada é indistinguível, para quem olha, de uma que
    travou.

    O critério é **estar dentro de um `lambda`**, e não "estar numa função com certo nome": é o
    `lambda` que difere a execução, e é ele que `enfileira` recebe. Proxy de nome é o defeito que
    a guarda irmã tinha: ela exigia que a variável se chamasse `transporte`, e
    `constroi_transporte().post(...)` passava por ela.

    Função de módulo devolvendo lista porque é o que torna o negativo honesto: a planta é
    argumento, então o negativo percorre este mesmo caminho e não pode ficar inerte. E os outros
    testes importam esta função, porque negativo que reimplementa a regra fica verde enquanto a
    regra apodrece.
    """
    arvore = ast.parse(fonte)

    diferidos = {
        filho for no in ast.walk(arvore) if isinstance(no, ast.Lambda) for filho in ast.walk(no)
    }

    achados: list[str] = []
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Call):
            continue
        alvo = no.func
        nome = (
            alvo.id
            if isinstance(alvo, ast.Name)
            else alvo.attr
            if isinstance(alvo, ast.Attribute)
            else None
        )
        if nome == "envia" and no not in diferidos:
            achados.append(f"linha {no.lineno} — {ast.unparse(no)[:70]}")
    return achados


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _TransporteQueLevanta:
    """Um `Transporte` cujas duas chamadas levantam. Serve o desfecho de causa `ambiente`.

    Não herda nada: `Transporte` é `Protocol`, e é isso que permite ao `TestClient` servir de duplo.
    """

    def __init__(self, erro: BaseException) -> None:
        self._erro = erro

    def post(self, url: str, **kwargs: Any) -> Any:
        raise self._erro

    def get(self, url: str, **kwargs: Any) -> Any:
        raise self._erro


# O Palpite medido, trocado por um duplo que decide na hora. A janela já pediu o Palpite antes de
# todo disparo delegado, e a caixa «Automático» nascia ticada: sem o duplo, todo teste que
# executava pela janela corria três corretores sobre 200 features no servidor em processo —
# minutos por teste, medido ao correr este arquivo com a mudança. Hoje só o cartão «Palpite» e
# «Seguir sugestões do app» o pedem, e o duplo continua a ser o Palpite de quem os clica. O que o
# duplo devolve sai da regra do produto, `decide_o_palpite`; a medição em si tem teste próprio
# contra os vetores gravados dos arquivos reais (`tests/unit/test_palpite_medido.py`). Avisa o
# gancho uma vez por corretor, como o oráculo, para o progresso da leitura andar pelo caminho de
# produção.
#
# **`none` vence, e de propósito.** Os testes do laço de espera foram escritos para o `neg_c18` com
# `method=none` — *«pronto em 0,20 s»* —, e a caixa ticada fazia a árvore delegar o SERRF, que leva
# mais de 120 s neste contêiner com a memória de correções fria: na árvore anterior a este diff,
# `test_u13_a_uma_leitura_basta_quando_o_trabalho_ja_terminou` sozinho reprova assim (medido em
# 23/10/2026). Com o Palpite a propor `none`, a correção delegada é a que eles descrevem.
ESCORES_DO_PALPITE_RAPIDO: Final = {"none": 0.10, "loess": 0.30, "serrf": 0.20}


@pytest.fixture(autouse=True)
def palpite_rapido(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Devolve os modos das matrizes medidas, para o teste contar as medições."""
    from collections import OrderedDict

    import refino.api.rotas as rotas
    from refino.decision.palpite import decide_o_palpite

    medidas: list[str] = []

    def medidor(
        matriz: Any,
        *,
        batches: Any = None,
        n_jobs: int = -1,
        depois_de_cada: Any = None,
        cancelled: Any = None,
        seed: int = 42,
    ) -> Any:
        medidas.append(str(matriz.mode))
        for corretor in ("none", "loess", "serrf"):
            if depois_de_cada is not None:
                depois_de_cada((corretor, 1))
        return decide_o_palpite(
            ESCORES_DO_PALPITE_RAPIDO, n_features_medidas=min(200, int(matriz.n_features))
        )

    monkeypatch.setattr(rotas, "mede_o_palpite", medidor)
    monkeypatch.setattr(rotas, "_PALPITES", OrderedDict())
    return medidas


def _le_a_rota_ate_o_fim(
    transporte: Any, rota: str, *, arquivos: dict[str, Any], campos: dict[str, Any]
) -> Resposta:
    """`POST` e, se a rota é longa, as leituras de `GET /trabalhos/{id}` até sair de `202`.

    As leituras eram três rotas síncronas; o Palpite passou a ser a medição em 200 features, que
    responde `202`. Os testes que percorrem `ROTAS_DA_LEITURA` pela API real recebem por aqui o
    corpo de cada uma, e a âncora do `202` é a do arnês da rota: sem identificador, reprova.
    """
    resposta = cliente.envia(transporte, "POST", rota, arquivos=arquivos, campos=campos)
    if resposta.codigo != 202:
        return resposta
    assert isinstance(resposta.corpo, dict) and resposta.corpo.get("identificador"), resposta.corpo
    identificador = str(resposta.corpo["identificador"])
    limite = time.monotonic() + 120
    while time.monotonic() < limite:
        resposta = cliente.envia(transporte, "GET", f"/trabalhos/{identificador}")
        if resposta.codigo != 202:
            return resposta
        time.sleep(0.02)
    raise AssertionError(f"{rota}: o trabalho {identificador} não terminou em 120 s")


def _com_a_correcao_feita(janela: JanelaPrincipal, corretor: str = "none") -> None:
    """A Correção feita sobre a matriz da tela, sem rota: é dela que partem os downloads dela.

    **Para os testes cujo assunto é o DOWNLOAD e não a Correção.** O download da Correção leva a
    etapa feita — os campos dela e o corretor efetivo —, e sem ela o pedido não sai. Até 09/11/2026
    estes testes clicavam o «Gerar» da aba «Executar e exportar», que corria o corretor do combo
    sem etapa nenhuma; a aba saiu, e o mesmo download mora no Resultado da Correção. A etapa é
    marcada pelo estado das etapas, com a fotografia da tela, como `_dispara_etapa` a marca.
    """
    atual = janela._campos_atuais()
    assert atual is not None, "âncora: sem arquivos não há Correção a dar por feita"
    tela, identidade = atual
    matriz = identidade[IDENTIDADE_DA_MATRIZ]
    campos = janela._etapas.campos_da_entrada(ETAPAS[0], "original", tela, matriz)
    janela._etapas.disparou(
        ETAPAS[0], fotografia_da_etapa(tela, ETAPAS[0], identidade, "original"), campos=campos
    )
    janela._etapas.concluiu(ETAPAS[0], correcao=corretor)
    assert janela._motivo_do_download(ETAPAS[0], matriz) is None, (
        "âncora: a Correção marcada não liga o download dela"
    )


@pytest.fixture
def transporte_real() -> TestClient:
    """O `TestClient` como `Transporte` — o padrão de `tests/test_ui_consumo.py`."""
    return TestClient(build_app(), raise_server_exceptions=False)


@pytest.fixture
def janela(qtbot: Any, transporte_real: TestClient) -> JanelaPrincipal:
    j = JanelaPrincipal(transporte_real)
    qtbot.addWidget(j)
    return j


# ---------------------------------------------------------------------------
# A janela abre e compõe
# ---------------------------------------------------------------------------


def test_a_janela_abre_e_o_titulo_e_o_declarado(janela: JanelaPrincipal) -> None:
    assert janela.windowTitle() == TITULO
    assert TITULO.strip(), "âncora: um título vazio passaria pela igualdade acima"


def _aba_que_mostra(janela: JanelaPrincipal, painel: Any) -> int:
    """O índice da aba em cuja árvore este painel está, ou `-1`.

    ## Por que ele existe, e por que `abas.widget(indice)` deixou de servir

    Até 03/09/2026 cada painel **era** o widget da sua aba, e o oráculo natural era
    `isinstance(janela.abas.widget(i), PainelDeX)` com `i` derivado da posição em `PAINEIS`. Com o
    agrupamento por função, dois painéis partilham uma aba e o widget dela é um composer — então
    aquele oráculo passou a comparar o painel errado, e reprovou dizendo *"o painel 4 ainda é uma
    marca: PainelDeQualidade"*, que é uma frase falsa sobre um estado correto.

    **O que se quer afirmar não mudou: o painel está na tela.** O que mudou é que ele pode estar
    aninhado. `isAncestorOf` percorre a árvore de widgets e responde exactamente isso.

    > **`-1` e não `None`, e não uma excepção:** o chamador afirma o índice, e um sentinela fora
    > do domínio dos índices distingue *"não está em aba nenhuma"* de *"está na primeira"* sem que
    > ninguém tenha de lembrar qual é qual.

    Isto NÃO substitui a pergunta *composto ≠ alimentado ≠ operável*. Ele responde
    à primeira, que é a mais fraca das três, e é por isso que os testes que o usam continuam a
    afirmar a classe e o estado ao lado dele.
    """
    for indice in range(janela.abas.count()):
        conteudo = janela.abas.widget(indice)
        if conteudo is painel or (conteudo is not None and conteudo.isAncestorOf(painel)):
            return indice
    return -1


def test_todo_painel_declarado_esta_composto_e_diz_de_quem_e(janela: JanelaPrincipal) -> None:
    """A composição, e cada painel declara **qual unidade** o preenche.

    Painel sem unidade é escopo perdido: ninguém o reclama, e o corte fecha sem ele sem que nada
    acuse.

    > Chamava-se `test_os_seis_paineis_...` com a âncora `== 6`, e um painel novo fê-lo reprovar
    > por acréscimo legítimo — que é o desfecho correto de uma âncora. A âncora **fica**: ela é o
    > que protege contra remoção silenciosa de painel. O que sai é o número **do nome**, porque
    > nome com contagem dentro envelhece num sítio onde nenhum portão a lê.

    **E a contagem de abas não é oráculo da composição**: trocar a marca de um painel vazio
    por um `QWidget` real não a muda. O que se afirma aqui é a bijeção entre declarado e composto.

    ## O agrupamento por função separou as duas contagens

    Até 03/09/2026 valia `abas.count() == len(PAINEIS)`, porque cada painel era uma aba. Os
    painéis passaram a ser agrupados por função, e agora `ABAS` diz quais painéis partilham aba: são
    **seis** abas para **nove** painéis, mais a faixa dos trabalhos, que é painel e não é aba.

    **As duas âncoras ficam, e são duas porque medem coisas diferentes:** `len(PAINEIS) == 9`
    protege contra painel removido em silêncio; `len(ABAS) == 6` protege contra aba removida. E a
    bijeção nova — **todo painel aparece em exactamente uma aba, ou é a faixa** — é o que impede
    um painel de sumir da tela continuando declarado, que é o defeito que o agrupamento
    introduziu como possível.

    > Os dois números subiram, e actualizar não é afrouxar. O painel `graficos` e a aba
    > «Gráficos» entraram em quinto lugar: o referente mudou por acréscimo legítimo, e a âncora
    > que o mede tem de o acompanhar — que é exactamente o desfecho que a docstring acima
    > descreve. O que se proíbe é baixar um número **medido** para fazer um teste passar, e aqui
    > não há número medido: há uma contagem de estrutura declarada no código.

    > E subiram outra vez, pela mesma razão: o painel `vias` e a aba «Vias metabólicas» entraram
    > em sexto lugar — antes da aba de honestidade, que é terminal por desenho, como «Gráficos».
    > O acréscimo é legítimo e a âncora acompanha-o.
    >
    > **A posição desloca UM ordinal — o da aba terminal — e isso é deliberado**:
    > a classe «contagem e ordinal de aba» tem afirmações vivas em documentos que
    > viajam para o usuário, e cada ordinal deslocado é uma edição a mais em prosa.

    > **E os dois DESCERAM UM em 09/11/2026, com a saída da aba «Executar e exportar».** A aba
    > saiu com o painel `artefatos`, que só ela desenhava, e o painel do
    > `pipeline` ficou sem aba: os grupos dele moram nas três abas de etapa. **Ele é a segunda
    > exceção da bijeção, declarada como a faixa** — `PAINEL_DAS_ETAPAS` —, e o que impede o
    > «fora das abas» de ser «não desenhado» é a asserção sobre os grupos, no fim deste teste.
    """
    from refino.desktop.janela.principal import INDICES_DAS_ABAS_DE_ETAPA, PAINEL_DAS_ETAPAS

    assert len(PAINEIS) == 9, f"a composição do corte mudou: {len(PAINEIS)} painéis"
    assert len(ABAS) == 9, f"o agrupamento das abas mudou: {len(ABAS)} abas"
    assert janela.abas.count() == len(ABAS)

    for indice, aba in enumerate(ABAS):
        assert janela.abas.tabText(indice) == aba.titulo
        assert aba.paineis or aba.etapa is not None, (
            f"a aba «{aba.titulo}» não declara painel nem etapa"
        )

    for painel in PAINEIS:
        assert painel.unidade.strip(), f"o painel `{painel.chave}` não diz quem o preenche"

    chaves = [p.chave for p in PAINEIS]
    assert len(set(chaves)) == len(chaves), f"chave de painel repetida: {chaves}"

    # **A bijeção painel↔aba, e as exceções são DUAS — declaradas e não silenciosas**: a faixa, que
    # está no rodapé, e o painel do pipeline, cujos grupos estão nas abas de etapa.
    em_abas = [chave for aba in ABAS for chave in aba.paineis]
    assert len(em_abas) == len(set(em_abas)), (
        f"um painel aparece em mais de uma aba: {em_abas}. Um `QWidget` só pode estar num layout, "
        f"então a segunda aba mostraria vazio"
    )
    fora = sorted(set(chaves) - set(em_abas))
    assert fora == sorted([CHAVE_DOS_TRABALHOS, PAINEL_DAS_ETAPAS]), (
        f"estes painéis são declarados e não aparecem em aba nenhuma: {fora}. Só a faixa do rodapé "
        f"— `{CHAVE_DOS_TRABALHOS}` — e o painel cujos grupos as abas de etapa compõem — "
        f"`{PAINEL_DAS_ETAPAS}` — podem estar fora das abas"
    )
    for etapa, grupos in janela.pipeline.grupos_das_etapas.items():
        pagina = janela.abas.widget(INDICES_DAS_ABAS_DE_ETAPA[etapa])
        assert grupos and all(pagina.isAncestorOf(grupo) for grupo in grupos), (
            f"os grupos da etapa {etapa} não estão na aba dela: o painel do pipeline saiu das abas "
            "e os controles dele não são desenhados em lado nenhum"
        )
    orfaos = sorted(set(em_abas) - set(chaves))
    assert not orfaos, f"estas abas nomeiam painel que `PAINEIS` não declara: {orfaos}"

    # A faixa existe, e o painel dos trabalhos está DENTRO dela — senão «fora das abas» seria
    # indistinguível de «não desenhado em lado nenhum».
    dentro_da_faixa = janela.faixa_de_trabalhos.layout()
    assert dentro_da_faixa is not None and dentro_da_faixa.indexOf(janela.trabalhos) >= 0, (
        "o painel dos trabalhos saiu das abas e não está na faixa do rodapé: ele foi construído, "
        "tem pai, e não é desenhado em lado nenhum"
    )


def test_u2_existe_exactamente_uma_aba_de_graficos_e_ela_mostra_o_painel_novo(
    janela: JanelaPrincipal,
) -> None:
    """A aba «Gráficos» existe, é única, e o painel dela está na tela.

    **O índice é DERIVADO por `next(...)` e nunca escrito.** Um literal aqui seria um portão
    pinado, e reprovaria no dia em que uma aba entrasse antes dela por razão legítima, sem que
    nada de errado tivesse acontecido.

    **A unicidade é afirmada e não suposta.** Duas abas com o mesmo conjunto de painéis fariam a
    segunda mostrar vazio — um `QWidget` só vive num layout —, e `next(...)` acharia a primeira e
    passaria em silêncio sobre a segunda.
    """
    das_figuras = [aba for aba in ABAS if aba.paineis == ("graficos",)]
    assert len(das_figuras) == 1, f"abas cujo conjunto de painéis é `('graficos',)`: {das_figuras}"
    assert das_figuras[0].titulo == "Gráficos", (
        f"o título da aba das figuras é «{das_figuras[0].titulo}»"
    )

    indice = next(i for i, aba in enumerate(ABAS) if aba.paineis == ("graficos",))
    assert janela.abas.tabText(indice) == "Gráficos"
    assert isinstance(janela.graficos, PainelDeGraficos)
    assert _aba_que_mostra(janela, janela.graficos) == indice, (
        "o painel dos gráficos não está na árvore da aba que o declara"
    )


def test_u2_a_aba_de_graficos_nao_e_marca_de_painel_vazio(janela: JanelaPrincipal) -> None:
    """**A categoria de ERRO que não levanta, e é a mais cara desta unidade.**

    `_compoe_a_aba` tem um ramo de recuo: uma aba de um painel só cuja chave não esteja em
    `_preenchidos` compõe-se como **marca** em `_vazios`. Ele não levanta, não escreve em `stderr`,
    e a suíte fica verde — o usuário é que recebe uma aba com uma linha de texto onde deviam
    estar treze figuras. É um desfecho com aparência de sucesso.

    **Par causal na mesma função, e a planta é a omissão que se teme:** tirar `graficos` de
    `_preenchidos` e recompor a mesma aba. Sem esta metade, a asserção de ausência abaixo —
    *"`graficos` não está em `_vazios`"* — passaria por vacuidade sobre um mecanismo que nunca
    produz `_vazios` para chave nenhuma.
    """
    from PySide6.QtWidgets import QLabel

    # (a) a árvore limpa: a chave está composta, e a aba NÃO é marca.
    assert janela._preenchidos.get("graficos") is janela.graficos
    assert "graficos" not in janela._vazios, (
        f"a aba «Gráficos» compôs-se como marca de painel vazio: {sorted(janela._vazios)}. A "
        "suíte fica verde e a aba chega vazia a quem abre o programa"
    )

    # (b) a planta: sem a chave em `_preenchidos`, o mesmo agrupamento vira marca — sem levantar.
    composto = janela._preenchidos.pop("graficos")
    try:
        recuo = janela._compoe_a_aba(Aba("Gráficos", ("graficos",)))
        assert isinstance(recuo, QLabel), (
            f"o ramo de recuo devolveu {type(recuo).__name__} e não uma marca — a asserção (a) "
            "não separa nada"
        )
        assert "graficos" in janela._vazios
    finally:
        janela._preenchidos["graficos"] = composto
        janela._vazios.pop("graficos", None)


def _paineis_de_figura_desenhados_em(widget: Any) -> list[PainelDeFigura]:
    """Todo `PainelDeFigura` da árvore deste widget que está de facto num layout.

    **O oráculo é `PainelDeFigura._dentro` e não uma terceira descida escrita aqui.** Aquele
    método desce por layouts aninhados, e a sua docstring mede por que isso importa:
    `QLayout.indexOf` procura só os itens diretos. Reimplementá-lo seria a segunda fonte de
    verdade sobre *"está desenhado"*, e o negativo que dependesse dela ficaria verde por olhar
    para o lado errado.

    **`findChildren` responde por PARENTESCO e o layout responde por DESENHO**, e as duas metades
    são precisas: um widget com pai e fora do layout não é desenhado — é a distinção que
    `PainelDeQualidade._no_layout` existe para medir.
    """
    achados: list[PainelDeFigura] = []
    for candidato in widget.findChildren(PainelDeFigura):
        pai = candidato.parentWidget()
        raiz = pai.layout() if pai is not None else None
        if raiz is not None and PainelDeFigura._dentro(raiz, candidato):
            achados.append(candidato)
    return achados


def test_u2_a_figura_saiu_da_aba_de_qualidade_e_esta_na_dos_graficos(
    janela: JanelaPrincipal,
) -> None:
    """A figura mudou de casa, e as duas metades estão na MESMA função.

    **A âncora de não-vacuidade vem primeiro**: a descida acha **um** `PainelDeFigura`
    desenhado na árvore do painel dos gráficos. Sem ela, a asserção de ausência sobre a aba de
    qualidade passaria por vacuidade — bastava a descida devolver lista vazia para tudo, que é o
    desfecho de uma descida partida.

    **É par causal, e as duas pontas divergem sobre o mesmo instrumento**: a mesma função, os dois
    painéis, um resultado não vazio e um vazio.
    """
    na_aba_dos_graficos = _paineis_de_figura_desenhados_em(janela.graficos)
    assert na_aba_dos_graficos == [janela.graficos.figuras], (
        f"âncora: a descida achou {na_aba_dos_graficos} na aba das figuras — se ela não acha o "
        "painel que ESTÁ lá, a asserção de ausência abaixo não mede nada"
    )

    na_aba_de_qualidade = _paineis_de_figura_desenhados_em(janela.qualidade)
    assert na_aba_de_qualidade == [], (
        f"a aba de qualidade continua a desenhar um painel de figura: {na_aba_de_qualidade}. Ele "
        "mudou-se para a aba «Gráficos» na U2, e duas cópias dariam duas fontes sobre qual figura "
        "está na tela"
    )


def test_antes_de_qualquer_chamada_nao_ha_desfecho(janela: JanelaPrincipal) -> None:
    """A âncora do painel de desfecho.

    Sem um valor inicial **distinto**, *"a janela não mostrou erro"* e *"a janela mostrou string
    vazia"* seriam o mesmo estado, na célula que decide se há erro na tela.
    """
    assert janela.rotulo_de_desfecho.text() == SEM_DESFECHO


# ---------------------------------------------------------------------------
# O vocabulário de causa
# ---------------------------------------------------------------------------


def test_a_janela_exibe_toda_causa_do_conjunto_fechado_de_cliente() -> None:
    """Paridade de vocabulário, com o conjunto **extraído** de `cliente.py` e não escrito à mão.

    **A não-vacuidade vem primeiro e na mesma função**: sem ela, um `get_args` que devolvesse
    tupla vazia faria o laço não iterar e o teste passaria sem olhar causa nenhuma.

    E a asserção é de **presença**: para cada causa do conjunto fechado, a janela produz um texto
    que a nomeia. Uma causa acrescentada a `cliente.Causa` amanhã entra neste laço sozinha.
    """
    causas = get_args(cliente.Causa)
    assert len(causas) == 3, f"o vocabulário fechado deixou de ter três valores: {causas}"

    # Um desfecho por causa, com o corpo que o discriminador mecânico exige.
    por_causa = {
        "usuario": (404, None),
        "dado": (422, CORPO_DE_422_DE_CONTEUDO),
        "ambiente": (500, None),
    }
    assert set(por_causa) == set(causas), (
        "este teste não cobre o conjunto fechado inteiro — falta desfecho para "
        f"{sorted(set(causas) - set(por_causa))}"
    )

    vistas: list[str] = []
    for causa in causas:
        codigo, corpo = por_causa[causa]
        assert cliente.causa_de(codigo, corpo) == causa, (
            f"o desfecho escolhido para `{causa}` não produz essa causa — o teste mediria outra"
        )
        texto = texto_do_desfecho(codigo, corpo)
        assert causa in texto, f"a janela não nomeia a causa `{causa}`: {texto!r}"
        vistas.append(causa)

    assert len(vistas) == 3, "o laço não percorreu as três causas"


def test_a_mensagem_e_a_de_cliente_e_nenhuma_nasce_na_janela(janela: JanelaPrincipal) -> None:
    """A janela formata, não decide.

    A asserção é de igualdade com `cliente.mensagem_de` — não *"contém alguma frase"*, que passaria
    com um texto reescrito que por acaso partilhasse uma palavra.
    """
    texto = janela.mostra_desfecho(422, CORPO_DE_422_DE_CONTEUDO)
    esperada = cliente.mensagem_de(422, CORPO_DE_422_DE_CONTEUDO)

    assert esperada.strip(), "âncora: mensagem vazia tornaria o `in` abaixo vácuo"
    assert esperada in texto
    assert texto == f"[causa: dado] {esperada}"
    assert janela.rotulo_de_desfecho.text() == texto, "o painel não recebeu o texto"


def test_um_200_nao_produz_desfecho_de_erro(janela: JanelaPrincipal) -> None:
    """O par causal do painel de desfecho: sucesso não tem causa.

    `causa_de(200)` é `None` por desenho — forçá-lo ao vocabulário seria sucesso classificado como
    falha, e a contagem de falhas por causa passaria a contar sucesso.
    """
    texto = janela.mostra_desfecho(200)
    assert "[causa: None]" in texto
    for causa in get_args(cliente.Causa):
        assert f"[causa: {causa}]" not in texto


def test_uma_falha_de_transporte_e_causa_ambiente(qtbot: Any) -> None:
    """Erro de transporte — sem resposta HTTP nenhuma — sai como `ambiente`.

    Exercita o caminho `except` de `cliente.envia` **atravessando a piscina**, que é o que separa
    este teste de um teste de `cliente.py`: prova que a excepção é convertida antes de chegar à
    janela, e não que o mapeamento existe.
    """
    janela = JanelaPrincipal(_TransporteQueLevanta(ConnectionRefusedError("recusado")))
    qtbot.addWidget(janela)

    with qtbot.waitSignal(janela.resposta_recebida, timeout=10_000) as capturado:
        janela.chama("GET", "/qualquer", chave="ancora")

    chave, resposta = capturado.args
    assert chave == "ancora", "o sinal não trouxe a chave da chamada de volta"
    assert isinstance(resposta, Resposta)
    assert resposta.codigo == 503
    assert resposta.causa == "ambiente"
    assert janela.rotulo_de_desfecho.text() == texto_da_resposta(resposta)


def test_um_503_de_transporte_e_um_503_da_api_dizem_coisas_diferentes() -> None:
    """A planta que fixa a perda, e a âncora vem **antes** de qualquer afirmação sobre a janela.

    **Esta é a reprovação que a primeira versão da janela levou, virada em portão.** As três chaves
    de `503` — `banco`, `demora`, `rede` — não são deriváveis de `(código, corpo)`, então rederivar
    de uma `Resposta` de transporte produz a frase do **banco**, que acusa um banco que está de pé.

    A primeira asserção é a que torna o teste honesto: se as duas formas dessem o mesmo texto, tudo
    abaixo passaria por coincidência e o portão não mediria nada.
    """
    de_rede = cliente.envia(
        _TransporteQueLevanta(ConnectionRefusedError("recusado")), "GET", "/qualquer"
    )

    derivado = texto_do_desfecho(de_rede.codigo, de_rede.corpo)
    da_resposta = texto_da_resposta(de_rede)

    assert derivado != da_resposta, (
        "ÂNCORA: as duas formas deram o mesmo texto, então esta planta não separa nada. "
        f"derivado={derivado!r}"
    )
    # Comparar com a frase da chave é o oráculo direto, e não um proxy como o nome de um produto
    # na mensagem: é a propriedade, não a forma que ela tinha numa instância.
    assert derivado == f"[causa: ambiente] {cliente.mensagem_por_chave('503:banco')}", (
        f"a forma derivada deixou de cair na chave do banco — a planta perdeu o alvo: {derivado!r}"
    )
    assert da_resposta == f"[causa: ambiente] {cliente.mensagem_por_chave('503:rede')}"


def test_uma_rota_real_atravessa_a_piscina_e_chega_ao_painel(
    janela: JanelaPrincipal, qtbot: Any
) -> None:
    """Ponta a ponta em processo: janela -> piscina -> `cliente.envia` -> app ASGI -> painel.

    `GET /multiversos/<inexistente>` dá **404** de verdade, então o desfecho vem da API e não de um
    duplo — o que faz este teste falhar se a rota mudar de código, que é o ponto.
    """
    with qtbot.waitSignal(janela.resposta_recebida, timeout=20_000) as capturado:
        janela.chama("GET", "/multiversos/nao-existe-mesmo", chave="leitura")

    chave, resposta = capturado.args
    assert chave == "leitura"
    assert resposta.codigo == 404, f"a rota devolveu {resposta.codigo}; o teste mediria outra coisa"
    assert resposta.causa == "usuario"
    assert resposta.mensagem == cliente.mensagem_de(404, resposta.corpo), (
        "a mensagem não é a de `cliente` — alguma coisa a reescreveu no caminho"
    )
    assert janela.rotulo_de_desfecho.text() == texto_da_resposta(resposta)


# ---------------------------------------------------------------------------
# A chamada sai do laço de eventos
# ---------------------------------------------------------------------------


def test_ktd9_a_chamada_sai_do_laco_e_a_janela_continua_a_responder(
    janela: JanelaPrincipal, qtbot: Any
) -> None:
    """A chamada sai do laço de eventos, com par causal que separa síncrono de assíncrono.

    **O que faz este teste separar as duas implementações** é a asserção do meio. Com `enfileira`
    síncrono, `_Trabalho.run` correria na thread principal, o `emit` seria conexão **direta** —
    mesma thread — e `_recebe` já teria corrido quando o controle voltasse: `recebidas` estaria
    cheia. Assíncrono, o `emit` atravessa thread, a conexão é **enfileirada**, e nada chega até o
    laço girar.

    Sem essa asserção o teste passaria nas duas, porque a única diferença observável seria o tempo —
    e afirmar tempo dentro do portão de commit é o que produz reprovação intermitente.
    """
    comecou = threading.Event()
    liberar = threading.Event()
    recebidas: list[tuple[str, object]] = []
    janela.resposta_recebida.connect(lambda c, r: recebidas.append((c, r)))

    def lenta() -> Resposta:
        comecou.set()
        liberar.wait(30)
        return Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=None)

    with qtbot.waitSignal(janela.resposta_recebida, timeout=35_000):
        janela.enfileira(lenta, chave="lenta")

        assert comecou.wait(10), "a chamada nunca começou: a piscina não a executou"
        assert recebidas == [], (
            "a resposta chegou antes de o controle voltar ao teste — a chamada correu no laço de "
            "eventos, e a janela estaria congelada por todo o tempo da rota"
        )

        # E o laço responde COM a chamada em curso: uma ação na janela toma efeito.
        durante = janela.mostra_desfecho(404)
        assert janela.rotulo_de_desfecho.text() == durante != SEM_DESFECHO

        liberar.set()

    assert len(recebidas) == 1, (
        "a lista ficou vazia até o fim: a asserção de ausência acima passava por vacuidade"
    )
    assert recebidas[0][0] == "lenta", "a chave da chamada não chegou ao slot"
    assert janela.rotulo_de_desfecho.text() == f"[causa: None] {cliente.mensagem_de(200)}"


def test_uma_excepcao_que_envia_nao_trata_sai_por_sinal_em_vez_de_sumir(
    janela: JanelaPrincipal, qtbot: Any
) -> None:
    """O segundo desfecho de `_Trabalho`.

    `uvicorn` chamava `sys.exit(3)` num `bind` falhado, e o `SystemExit` escapava da thread como
    `PytestUnhandledThreadExceptionWarning` — reprovação **pelo motivo errado**. `envia` converte
    toda `Exception`, então só `BaseException` fora dessa hierarquia chega aqui; sem o sinal
    `falhou`, ela **some** e a janela fica à espera para sempre.
    """

    class DefeitoQueNaoEExcepcao(BaseException):
        pass

    def explode() -> Resposta:
        raise DefeitoQueNaoEExcepcao("XYZZY-SUBIU-DA-THREAD")

    with qtbot.waitSignal(janela.chamada_falhou, timeout=10_000) as capturado:
        janela.enfileira(explode, chave="explosiva")

    chave, erro = capturado.args
    assert chave == "explosiva"
    assert isinstance(erro, DefeitoQueNaoEExcepcao)
    assert "XYZZY-SUBIU-DA-THREAD" in str(erro)

    # A asserção já foi o inverso, e as duas metades dela são o conteúdo.
    #
    # Esta linha já exigiu `== SEM_DESFECHO`, e o que ela protegia era **não fabricar causa**.
    # A proteção fica; o silêncio não: `chamada_falhou` **não tinha consumidor de
    # produção**, então uma falha deixava no rótulo o desfecho da chamada **anterior** — que afirma
    # sucesso sobre uma chamada que morreu. Silêncio é indistinguível de sucesso; desfecho stale é
    # pior, porque é sucesso positivamente afirmado.
    texto = janela.rotulo_de_desfecho.text()
    assert texto != SEM_DESFECHO, (
        "a falha não chegou ao rótulo: quem olha continua a ver o desfecho da chamada anterior"
    )
    assert "DefeitoQueNaoEExcepcao" in texto, (
        f"a classe da excepção é o diagnóstico, e não está na frase: {texto}"
    )
    assert "explosiva" in texto, f"a frase não diz QUAL chamada falhou: {texto}"
    assert "[causa:" not in texto, (
        "a janela inventou causa para uma falha que não é de rota — colapsaria «a rota respondeu "
        f"503» com «o processo está a morrer». Texto: {texto}"
    )


def test_a_piscina_esvazia_e_o_desfecho_da_espera_nao_e_ignoravel(
    janela: JanelaPrincipal, qtbot: Any
) -> None:
    """`espera_a_piscina` devolve **se** esvaziou.

    Um método que devolvesse `None` faria o chamador não distinguir *"esvaziou"* de *"o teto
    passou"*, no ponto em que alguém decide se pode desligar.
    """
    with qtbot.waitSignal(janela.resposta_recebida, timeout=20_000):
        janela.chama("GET", "/multiversos/nao-existe-mesmo", chave="drenagem")
    assert janela.espera_a_piscina(5_000) is True


# ---------------------------------------------------------------------------
# O despacho, medido na thread PRINCIPAL — o `coverage` é cego à da piscina
# ---------------------------------------------------------------------------


def test_despacha_emite_concluido_quando_a_chamada_devolve(qtbot: Any) -> None:
    """O primeiro desfecho de `despacha`, com a cobertura a ver.

    **Medido em 22/08/2026: com este corpo dentro de `_Trabalho.run`, o `coverage` reportava 0%** —
    e os testes de piscina provavam que ele corria, porque afirmavam os sinais com o payload certo.
    `QThreadPool` cria threads **nativas**, que não passam pelo módulo `threading`, e é por ali que
    o `coverage` instrumenta thread.
    """
    sinais = _Sinais()
    vistas: list[tuple[str, object]] = []
    sinais.concluido.connect(lambda c, r: vistas.append((c, r)))
    esperada = Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=None)

    despacha(lambda: esperada, sinais, "boa")

    assert vistas == [("boa", esperada)]


def test_despacha_emite_falhou_quando_a_chamada_levanta(qtbot: Any) -> None:
    """O segundo desfecho. Dois sinais, duas direções, as duas na thread principal."""
    sinais = _Sinais()
    quebras: list[tuple[str, object]] = []
    sinais.falhou.connect(lambda c, e: quebras.append((c, e)))

    class DefeitoDeTeste(BaseException):
        pass

    def explode() -> Resposta:
        raise DefeitoDeTeste("XYZZY-DESPACHO")

    despacha(explode, sinais, "ma")

    assert len(quebras) == 1
    assert quebras[0][0] == "ma", "a chave não viajou pelo desfecho de falha"
    assert isinstance(quebras[0][1], DefeitoDeTeste)


def test_o_run_do_trabalho_delega_ao_despacho(qtbot: Any) -> None:
    """A linha de `_Trabalho.run`, exercitada em vez de marcada com `pragma`.

    **O teste vem antes do `pragma`, e aqui ele é trivial:** chamar `run()` na
    thread principal instrumenta a linha e prova que o despacho é o que se pensa. Sem isto, um `run`
    que não chamasse nada ficaria coberto por um `pragma` e a piscina engoliria o silêncio.
    """
    esperada = Resposta(codigo=202, ok=True, causa=None, mensagem="OK", corpo=None)
    trabalho = _Trabalho(lambda: esperada, "delegada")
    vistas: list[tuple[str, object]] = []
    trabalho.sinais.concluido.connect(lambda c, r: vistas.append((c, r)))

    trabalho.run()

    assert vistas == [("delegada", esperada)], "`run` não delegou a `despacha`"


# ---------------------------------------------------------------------------
# O ponto de entrada de Qt do arranque
# ---------------------------------------------------------------------------


def _corre_a_janela_sem_deixar_o_quit(app: Any) -> int:
    """`arranque._corre_a_janela`, e depois tira da fila o `Quit` que o fecho da janela deixa nela.

    Medido: com a janela aberta, o `app.quit` que encerra o `exec()`
    deixa um `QEvent.Quit` **postado** na fila depois de o `exec()` voltar. O `processEvents` que o
    próprio `pytest-qt` chama ao fim da fase de chamada o entrega, e `QCoreApplication.exit` corre
    sem `exec()` que o desfaça. A partir daí todo `QEventLoop.exec()` do processo volta **-1 sem
    girar**: `qtbot.wait` e `qtbot.waitUntil` viram um laço que não processa evento nenhum e
    estoura o prazo, e `processEvents`, que não passa por ali, continua a entregar. Sem a janela, o
    mesmo `exec()` com `app.quit` não deixa nada.

    **A âncora vai junto e é o que torna a limpeza verificável:** depois dela, um `QEventLoop`
    aninhado tem de girar até o próprio temporizador. Sem a linha da limpeza, ela reprova nos três
    testes que sobem o laço principal.
    """
    from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer

    codigo = arranque._corre_a_janela()
    QCoreApplication.removePostedEvents(app, QEvent.Type.Quit)
    QCoreApplication.processEvents()
    laco = QEventLoop()
    QTimer.singleShot(20, laco.quit)
    assert laco.exec() == 0, (
        "depois do laço principal, um `QEventLoop` aninhado volta sem girar: ficou um `Quit` na "
        "fila, e `qtbot.wait` e `qtbot.waitUntil` deixam de processar eventos no resto do processo"
    )
    return int(codigo)


def test_corre_a_janela_sobe_o_laco_de_eventos_e_devolve_o_codigo(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O caminho de Qt REAL de `arranque`, e é o único que faltava.

    `QTimer.singleShot(0, app.quit)` faz `exec()` voltar assim que o laço gira — é a forma
    idiomática de exercitar um laço principal sem o deixar bloquear. Sem isto, `_corre_a_janela`
    ficaria coberto
    só por um `pragma`, e o que ele afirma — *a janela abre e o laço corre* — é a coisa que a
    janela existe para entregar.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    assert app is not None, "âncora: o `qtbot` devia ter construído a QApplication"

    monkeypatch.setenv(arranque.VARIAVEL_DA_BASE, "http://127.0.0.1:1")
    QTimer.singleShot(0, app.quit)

    assert _corre_a_janela_sem_deixar_o_quit(app) == 0


# ---------------------------------------------------------------------------
# Os dois portões por AST sobre `desktop/janela/`
# ---------------------------------------------------------------------------


def test_a_janela_nao_fala_com_o_transporte_direto() -> None:
    """A janela não monta transporte, e a guarda é a de `tests/contrato_de_rotas.py`.

    **Reusa a função daquele módulo em vez de a copiar.** Negativo que reimplementa a regra fica
    verde enquanto a regra apodrece — e aqui o risco é concreto: aquela guarda já foi
    corrigida uma vez por casar nome de variável.

    ## ESCOPO NEGATIVO, e ele não é o degrau mais baixo

    O alvo é `src/refino/desktop/janela/*.py` e **`arranque.py` fica de fora de propósito**: ele é a
    raiz de composição — o único lugar que legitimamente chama `constroi_transporte`, e chama-o em
    `_corre_a_janela`. Incluí-lo faria o portão acusar a única chamada correta que existe.

    **A consequência declarada:** nada confere que `arranque.py` não monte transporte por fora além
    desta frase. Ele tem **uma** chamada, num módulo de 200 linhas com 100% de cobertura, e o
    portão irmão da janela contra a API é onde essa superfície ganha mecanismo se precisar.

    E o critério herdado tem um limite conhecido: `constroi_transporte` dentro de uma função chamada
    `_transporte` passa, porque a guarda de origem resolve a fábrica por **nome**. É a mesma classe
    que já foi corrigida do outro lado, e fica registrada em vez de suposta ausente.
    """
    assert FONTES_DA_JANELA, "âncora: nenhum arquivo de `desktop/janela/` foi encontrado"
    for caminho in FONTES_DA_JANELA:
        achados = chamadas_diretas_ao_transporte(caminho.read_text(encoding="utf-8"))
        assert achados == [], f"{caminho.name} fala com o transporte direto: {achados}"


def test_nenhum_envio_corre_no_laco_de_eventos() -> None:
    """Nenhum envio no laço de eventos, como portão e não como intenção.

    Painéis novos entram neste pacote, e o `envia` síncrono é a coisa mais natural de
    escrever. O portão é o que faz a regra valer sem ninguém se lembrar dela.
    """
    assert FONTES_DA_JANELA, "âncora: nenhum arquivo de `desktop/janela/` foi encontrado"
    # **A segunda âncora, e ela vai NA MESMA FUNÇÃO e antes.** `achados == []` passa quando o
    # extractor deixa de casar — e um extractor mudo devolve lista vazia para tudo. O negativo que
    # prova o contrário vivia só num teste vizinho, que pode ser removido sem que nada ligue os
    # dois.
    assert envios_sincronos("def f(t):\n    return envia(t, 'GET', '/x')\n"), (
        "o extractor não acusa nem uma chamada síncrona plantada — as asserções abaixo passariam "
        "por vacuidade"
    )
    for caminho in FONTES_DA_JANELA:
        achados = envios_sincronos(caminho.read_text(encoding="utf-8"))
        assert achados == [], f"{caminho.name} chama `envia` fora de um `lambda`: {achados}"


def test_negativo_um_envio_sincrono_novo_e_acusado() -> None:
    """A planta, e ela cai sobre código que **difere** do aceito.

    Duas fontes na mesma asserção: uma com `envia` dentro de `lambda` — aceita — e outra com ele
    solto. Se a guarda casasse só o nome `envia`, as duas acusariam e o portão sempre reprovaria.
    """
    diferida = "def chama(t):\n    return enfileira(lambda: envia(t, 'GET', '/x'))\n"
    sincrona = "def chama(t):\n    resposta = envia(t, 'GET', '/x')\n    return resposta\n"

    assert envios_sincronos(diferida) == [], (
        "a forma ACEITA foi acusada — o portão sempre reprova e os negativos ficariam verdes"
    )

    acusada = envios_sincronos(sincrona)
    assert len(acusada) == 1, f"a planta síncrona não foi acusada: {acusada}"
    assert "linha 2" in acusada[0]


def test_negativo_a_forma_de_atributo_tambem_e_acusada() -> None:
    """`cliente.envia(...)` é a mesma chamada com outra sintaxe, e um portão que só casasse `Name`
    ficaria cego a ela — o proxy de forma, que é irmão do proxy de nome."""
    acusada = envios_sincronos("def f(t):\n    return cliente.envia(t, 'GET', '/x')\n")
    assert len(acusada) == 1, f"a forma de atributo escapou: {acusada}"


def test_a_janela_nao_importa_biblioteca_de_http() -> None:
    """Nem `httpx`, nem `httpx2`, nem `requests`. O transporte entra por injeção e mais nada.

    Asserção de ausência com **âncora de presença antes**: a lista de módulos importados tem de ser
    não vazia, senão uma extração que falhasse deixaria o `not in` passar por vacuidade.
    """
    proibidos = {"httpx", "httpx2", "requests", "urllib", "urllib3", "http"}
    for caminho in FONTES_DA_JANELA:
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        importados: set[str] = set()
        for no in ast.walk(arvore):
            if isinstance(no, ast.Import):
                importados.update(a.name.split(".")[0] for a in no.names)
            elif isinstance(no, ast.ImportFrom) and no.module:
                importados.add(no.module.split(".")[0])

        if caminho.name == "__init__.py":
            continue
        assert importados, f"{caminho.name}: nenhum import extraído — a asserção seria vácua"
        assert not (importados & proibidos), (
            f"{caminho.name} importa biblioteca de HTTP: {sorted(importados & proibidos)}"
        )


# ---------------------------------------------------------------------------
# Painéis 1 e 2
# ---------------------------------------------------------------------------
#
# **Os dois portões de AST acima já cobrem os módulos novos**, e é por isso que não há teste novo
# para eles aqui: `FONTES_DA_JANELA` é um `glob` sobre `desktop/janela/*.py`, então `arquivos.py` e
# `leitura.py` entraram na cobertura deles sozinhos.
#
# **E o que essa frase NÃO cobre vai declarado, porque a revisão mostrou que ela se lê mais larga do
# que é.** Os dois portões são o do transporte direto e o do `envia` fora de `lambda`. Nenhum dos
# dois atribui a uma rota as leituras de resposta que `leitura.py` faz (`codigo`, `severidade`,
# `mensagem`, `aceita`, `problemas`, `defect_type`, `confidence`, `correction_proposta`,
# `avaliavel`, `abstem`, `confianca`): isso é do portão irmão, `tests/test_contrato_janela_api.py`.


def _faz_divergir(pipeline: Any) -> tuple[str, ...]:
    """Encena o gesto que exerce a sobrescrita: a máquina propõe, e o humano muda. Devolve os eixos.

    **Deriva o eixo e os dois valores da ESTRUTURA e não os nomeia**, pela mesma razão que a versão
    com checkbox derivava a primeira caixa: um teste que escrevesse `"correction"` e `"loess"` à mão
    passaria a medir aquele eixo em vez de *"o gesto que exerce a sobrescrita"*, e ficaria mudo no
    dia em que ele mudasse de tipo de widget.

    **Levanta em vez de devolver vazio quando não consegue encenar.** Vazio seria lido pelo chamador
    como *"nada divergiu"*, que é o desfecho que os testes medem — o auxiliar deixaria de separar o
    que existe para separar. Um valor de retorno, um significado.
    """
    from refino.desktop.janela.pipeline import CORRETORES

    proposto, mudado = CORRETORES[0], CORRETORES[1]
    if not pipeline.pre_preenche("correction", proposto):
        pipeline.corretor.setCurrentText(proposto)
    pipeline.corretor.setCurrentText(mudado)
    divergentes = pipeline.eixos_divergentes()
    if not divergentes:
        raise AssertionError(
            f"o auxiliar não conseguiu encenar divergência: propôs {proposto!r}, mudou para "
            f"{mudado!r}, e `eixos_divergentes` devolveu vazio"
        )
    return divergentes


class _EscolhedorQueConta:
    """Um `Escolhedor` que CONTA invocações. É o que torna a propriedade central verificável.

    `QFileDialog` abre janela modal do sistema, e um teste não pode fechá-la: um painel que o
    chamasse direto tornaria *"invocado uma vez"* inverificável. Com o duplo, a asserção é sobre um
    número.
    """

    def __init__(self, medidas: Path, lista: Path) -> None:
        self._por_titulo = {ROTULO_DAS_MEDIDAS: medidas, ROTULO_DA_LISTA: lista}
        self.invocacoes = 0
        self.titulos: list[str] = []

    def __call__(self, titulo: str) -> Path | None:
        self.invocacoes += 1
        self.titulos.append(titulo)
        # **Levanta em vez de recuar para `None`.** Título que o duplo não conhece significa que o
        # painel pediu um arquivo que este arnês não sabe servir, e devolver `None` faria isso
        # passar por «o usuário cancelou» — o desfecho de sucesso do cancelamento a cobrir um
        # defeito de acoplamento. É a mesma decisão que `rotulo_do_eixo` regista.
        return self._por_titulo[titulo]


class _TransporteQueGrava:
    """Envolve o `TestClient` e GRAVA `(rota, files, data)` de cada chamada.

    **Existe para o cenário da sobrescrita, e a razão é o recorte dele:** o plano exige que a
    sobrescrita exercida mude *"o payload enviado — conferido no corpo da requisição, não na tela"*.
    Uma asserção sobre o widget mediria o que o profissional vê; esta mede o que o servidor recebe,
    e são coisas diferentes — uma sobrescrita que só mudasse a tela é sobrescrita que o servidor
    nunca vê.
    """

    def __init__(self, real: TestClient) -> None:
        self._real = real
        self.chamadas: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        self.chamadas.append({"metodo": "POST", "rota": url, **kwargs})
        return self._real.post(url, **kwargs)

    def get(self, url: str, **kwargs: Any) -> Any:
        self.chamadas.append({"metodo": "GET", "rota": url, **kwargs})
        return self._real.get(url, **kwargs)


@pytest.fixture
def painel_de_arquivos(qtbot: Any) -> Any:
    """O painel 1 com o par real do `neg_c18`, pelos resolvedores do `conftest`."""
    escolhedor = _EscolhedorQueConta(measurements_path("neg_c18"), sample_list_path("neg_c18"))
    painel = PainelDeArquivos(escolhedor)
    qtbot.addWidget(painel)
    return painel, escolhedor


def test_u4_o_seletor_e_invocado_uma_vez_e_seis_chamadas_reusam_os_bytes(
    painel_de_arquivos: Any,
) -> None:
    """A contagem, e não a impressão.

    Seis composições de payload contra **uma** invocação do seletor. A asserção é sobre
    `escolhedor.invocacoes`, que é um número: *"o painel parece guardar os bytes"* não é
    verificável, e é exactamente o que um teste sobre widget mediria.

    E a segunda metade é a que importa: os bytes têm de ser **os mesmos**. Um painel que relesse o
    disco a cada chamada passaria pela contagem e falharia aqui no dia em que o arquivo mudasse.
    """
    painel, escolhedor = painel_de_arquivos

    assert escolhedor.invocacoes == 0, "âncora: o painel invocou o seletor antes de lhe pedirem"
    assert painel.par is None
    assert painel.rotulo.text() == SEM_PAR

    assert painel.escolhe_medidas() is None, (
        "o painel montou um par com uma metade só: `arquivos_para_envio` enviaria `sample_list` "
        "vazio, e a rota culparia o arquivo"
    )
    assert escolhedor.invocacoes == 1
    assert painel.escolhe_lista() is not None
    assert escolhedor.invocacoes == 2

    assert escolhedor.titulos == [ROTULO_DAS_MEDIDAS, ROTULO_DA_LISTA], (
        f"o painel pediu os arquivos com estes títulos: {escolhedor.titulos}. Cada botão tem de "
        "dizer ao diálogo QUAL dos dois está a pedir, senão a ordem volta a ser sorte"
    )

    payloads = [painel.arquivos_para_envio() for _ in range(6)]
    assert escolhedor.invocacoes == 2, (
        f"o seletor foi invocado {escolhedor.invocacoes} vezes para seis chamadas — o par não está "
        "a ser reusado, e cada chamada pediria os arquivos de novo"
    )
    primeiro = payloads[0]["measurements"][1]
    assert all(p["measurements"][1] is primeiro for p in payloads), (
        "os bytes não são os MESMOS objetos: alguma chamada releu o disco, e duas chamadas podem "
        "enviar conteúdo diferente sem nada acusar"
    )


def test_u4_cancelar_o_seletor_nao_apaga_o_par_anterior(qtbot: Any) -> None:
    """O terceiro desfecho de `escolhe`, e ele dispara por AUSÊNCIA — logo ficaria inerte.

    Cancelar não é erro e não é escolha: é o usuário a não pedir nada. Um painel que apagasse o par
    ali faria a chamada seguinte levantar, e o profissional leria *"nenhum par escolhido"* depois de
    ter escolhido um.
    """

    class _QueCancelaDepois:
        """Serve cada título **uma** vez; a chamada seguinte ao mesmo título cancela."""

        def __init__(self, medidas: Path, lista: Path) -> None:
            self._por_titulo: dict[str, Path | None] = {
                ROTULO_DAS_MEDIDAS: medidas,
                ROTULO_DA_LISTA: lista,
            }

        def __call__(self, titulo: str) -> Path | None:
            atual = self._por_titulo[titulo]
            self._por_titulo[titulo] = None
            return atual

    painel = PainelDeArquivos(
        _QueCancelaDepois(measurements_path("neg_c18"), sample_list_path("neg_c18"))
    )
    qtbot.addWidget(painel)

    painel.escolhe_medidas()
    primeiro = painel.escolhe_lista()
    assert primeiro is not None, "âncora: a primeira escolha tem de ter dado um par completo"

    segundo = painel.escolhe_medidas()  # este cancela
    assert segundo is primeiro, "cancelar trocou o par"
    assert painel.par is primeiro
    assert painel.escolhe_lista() is primeiro, "cancelar na outra metade trocou o par"


def test_u4_sem_par_escolhido_o_payload_levanta_em_vez_de_ir_vazio(qtbot: Any) -> None:
    """Devolver `{}` faria a rota responder `422` a culpar um arquivo que ninguém escolheu."""
    painel = PainelDeArquivos(lambda _titulo: None)
    qtbot.addWidget(painel)

    for metodo in (painel.arquivos_para_envio, painel.campos_para_envio):
        with pytest.raises(ValueError, match="nenhum par de arquivos escolhido"):
            metodo()


def test_u4_o_conjunto_de_modos_fica_aberto(painel_de_arquivos: Any) -> None:
    """Fechar o combo restringiria o que a ferramenta pode fazer, e isso foi medido.

    `fluxo.MODOS_SUGERIDOS` é **sugestão e não domínio**: o núcleo trata o modo como
    rótulo aberto, e um combo fechado deixaria um estudo de lipidómica sem por onde
    entrar. A asserção é sobre `isEditable`, que é a propriedade que
    o permite.
    """
    painel, _ = painel_de_arquivos

    assert MODOS_SUGERIDOS, "âncora: a tupla de sugestões está vazia"
    assert painel.modo.count() == len(MODOS_SUGERIDOS)
    assert painel.modo.isEditable(), (
        "o combo do modo está FECHADO: um estudo fora dos quatro deste laboratório não teria por "
        "onde entrar, que é o defeito que `accept_new_options=True` consertou na tela web"
    )

    painel.modo.setEditText("pos_lipidomica")
    painel.escolhe_medidas()
    assert painel.escolhe_lista() is not None
    assert painel.campos_para_envio() == {"mode": "pos_lipidomica"}, (
        "o modo digitado não chegou ao payload"
    )


# ---------------------------------------------------------------------------
# O PRÉ-PREENCHIMENTO e a DIVERGÊNCIA, no lugar das checkboxes
# ---------------------------------------------------------------------------


def test_d378_a_leitura_pre_preenche_o_controle_de_valor_e_registra_a_proposta(
    qtbot: Any,
) -> None:
    """A travessia inteira: a resposta chega à janela, e o combo do pipeline repousa na proposta.

    **É o teste da metade que torna a divergência mensurável.** Sem pré-preenchimento, o combo
    repousa em `none` e *"o valor difere da proposta"* seria verdade em quase toda execução —
    inclusive na de quem nunca olhou. É essa a razão de a metade não poder ser adiada.

    **Duas asserções e não uma, porque o widget e o registro são coisas diferentes.** Um
    `pre_preenche` que mexesse o combo e esquecesse de guardar a proposta passaria pela primeira; a
    divergência seguinte compararia contra vazio e **nenhum** eixo entraria em
    `campos_sobrescritos`, em silêncio. `proposta_da_maquina` existe para essa segunda asserção.
    """
    janela = JanelaPrincipal(
        _TransporteQueLevanta(RuntimeError("nenhuma rota corre neste teste")),
        escolhedor=lambda _titulo: None,
    )
    qtbot.addWidget(janela)
    proposto = CORRETORES[-1]

    assert janela.pipeline.corretor.currentText() != proposto, (
        f"âncora: o combo já repousa em {proposto!r}, e o teste não separaria o pré-preenchimento "
        f"de não fazer nada"
    )

    mudados = janela._pre_preenche_com_a_proposta(
        "recomendacao", {"correction_proposta": proposto, "rationale": "r"}
    )

    # A leitura GUARDA e não escreve: a página de correção fica nos valores padrão até que o
    # pesquisador a preencha ou escolha seguir as recomendações do app.
    # O combo já repousou na proposta assim que ela chegava; agora quem a escreve é «Seguir
    # sugestões do app», e as duas asserções de baixo — o widget e o registro — passam a ser as da
    # adoção.
    assert mudados == (), f"a leitura escreveu na aba «Correção»: {mudados}"
    assert janela.pipeline.corretor.currentText() != proposto, "o combo recebeu a proposta sozinho"
    assert janela.pipeline.corretor_sugerido == proposto, "a sugestão não ficou guardada"
    assert "correction" not in janela.pipeline.proposta_da_maquina(), (
        "a proposta não adotada entrou no lado da máquina do par"
    )
    assert janela.pipeline.segue_a_proposta("correction") == proposto, "a adoção não escreveu"
    assert janela.pipeline.corretor.currentText() == proposto, "o combo não recebeu a proposta"
    assert janela.pipeline.proposta_da_maquina().get("correction") == proposto, (
        "o combo mudou e a proposta NÃO ficou registada — a divergência seguinte compararia "
        "contra vazio e nenhum eixo entraria em `campos_sobrescritos`"
    )
    assert janela.pipeline.eixos_divergentes() == (), (
        "aceitar a proposta foi contado como sobrescrita"
    )


def test_d378_o_pre_preenchimento_nao_apaga_o_que_o_humano_ja_escolheu(qtbot: Any) -> None:
    """As três leituras correm em PARALELO, e uma pode chegar depois da escolha.

    **O defeito que isto impede é silencioso e é o pior desta unidade:** o profissional escolhe o
    corretor, a recomendação volta segundos depois, o combo salta para a proposta, e a execução
    corre com um método que ninguém pediu. Nada levanta — o valor é legítimo, e a tela mostra-o.

    **Par causal:** a mesma chamada, sobre o mesmo painel, muda o combo quando ninguém tocou e
    **não** o muda depois de alguém ter tocado. Sem a primeira metade, um `pre_preenche` que nunca
    fizesse nada passaria pela segunda.
    """
    intocado = PainelDePipeline()
    qtbot.addWidget(intocado)
    tocado = PainelDePipeline()
    qtbot.addWidget(tocado)
    escolhido, proposto = CORRETORES[1], CORRETORES[-1]

    assert escolhido != proposto, "âncora: os dois valores do par causal são o mesmo"
    # **O valor escolhido tem de DIFERIR do repouso, e a razão é o mecanismo.** `_tocados` é
    # alimentado por `currentTextChanged`, e o Qt não o emite quando se escreve o valor que já lá
    # está. Um teste que «escolhesse» o repouso não marcaria o eixo como tocado e mediria o
    # mecanismo inerte em vez do mecanismo a respeitar a escolha.
    assert escolhido != tocado.corretor.currentText(), (
        "âncora: o valor escolhido é o de repouso do combo, e o sinal que marca «tocado» não "
        "dispara — este teste passaria a medir outra coisa"
    )

    assert intocado.pre_preenche("correction", proposto), (
        "âncora: o pré-preenchimento não mexeu um painel em que ninguém tocou, e a asserção de "
        "baixo passaria por o mecanismo estar inerte e não por ele respeitar a escolha"
    )

    tocado.corretor.setCurrentText(escolhido)
    assert not tocado.pre_preenche("correction", proposto), (
        "a proposta que chegou DEPOIS apagou a escolha do profissional"
    )
    assert tocado.corretor.currentText() == escolhido
    assert tocado.eixos_divergentes() == ("correction",), (
        "a escolha que difere da proposta não foi contada como sobrescrita"
    )


def test_d378_a_escrita_da_maquina_nao_conta_como_toque_do_humano(qtbot: Any) -> None:
    """O defeito que **anulava a unidade inteira**, achado por revisão adversarial.

    ## O mecanismo, e por que ele não se vê no código

    `_escreve_valor` chama `setCurrentText`, e o Qt emite `currentTextChanged` — o **mesmo** sinal
    que um clique emite. O Qt não distingue quem originou a escrita. Sem guarda, pré-preencher
    marcava o eixo como *tocado pelo humano*, e a leitura **seguinte** — clicar «Recomendação» outra
    vez, que é gesto legítimo porque os três botões são independentes — gravava
    `campos_sobrescritos=["correction"]` para quem não tinha mexido em nada.

    > **É exactamente o desfecho que o pré-preenchimento existe para evitar:** *"toda execução
    > ficaria gravada como manual, incluindo a de quem nunca olhou"*.

    **Medido no painel real antes do conserto**, com duas leituras seguidas:

        pre_preenche#1 -> True   combo: loess   tocados: {'correction'}   divergentes: ()
        pre_preenche#2 -> False  combo: loess   proposta: {'correction': 'serrf'}
        divergentes: ('correction',)   <-- e o humano nunca tocou

    **Par causal, e as duas metades correm na mesma função:** a segunda leitura da máquina **não**
    produz divergência; o toque do humano **produz**. Sem a segunda metade, uma guarda que
    ignorasse *toda* escrita — inclusive a do humano — passaria pela primeira.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    primeiro, segundo = CORRETORES[1], CORRETORES[-1]

    assert primeiro != segundo, "âncora: as duas propostas do par causal são o mesmo valor"

    assert painel.pre_preenche("correction", primeiro), (
        "âncora: o primeiro pré-preenchimento não mexeu o combo, e o resto do teste mediria um "
        "mecanismo inerte"
    )
    assert painel.eixos_divergentes() == (), "a primeira leitura já contou como sobrescrita"

    painel.pre_preenche("correction", segundo)

    assert painel.eixos_divergentes() == (), (
        "a SEGUNDA leitura da máquina contou como sobrescrita: o `setCurrentText` da própria "
        "máquina marcou o eixo como tocado pelo humano, e quem não mexeu em nada fica gravado "
        "como tendo decidido"
    )

    painel.corretor.setCurrentText(primeiro)

    assert painel.eixos_divergentes() == ("correction",), (
        "a guarda ficou larga demais e passou a ignorar o toque do HUMANO também — nenhuma "
        "escolha contaria como sobrescrita, que é o defeito oposto"
    )


def test_d378_a_proposta_que_o_widget_recusa_nao_fica_registada(qtbot: Any) -> None:
    """A recusa tinha docstring e não tinha efeito, e o desfecho era sobrescrita FANTASMA.

    `setCurrentText` sobre um `QComboBox` não editável com texto fora da lista **não levanta**:
    deixa o combo como estava. A primeira versão gravava `_proposta[eixo]` **antes** de escrever, e
    o estado ficava com um valor que a tela nunca mostrou — `eixos_divergentes` reportava
    sobrescrita contra ele, sem ninguém ter mexido em nada.

    ## E o par causal tem TRÊS casos, porque o valor de retorno significava duas coisas

    *"o widget não mudou"* cobria dois estados opostos: ele **recusou**, e ele **já estava** no
    valor. Colapsá-los fazia a máquina a propor o valor de repouso ser lida como recusa — a
    proposta não ficava registada, e a mudança seguinte do profissional **deixava de contar como
    sobrescrita**.

    | O que se propõe | `pre_preenche` devolve | A proposta fica registada? |
    |---|---|---|
    | valor que o combo não tem na lista | `False` | **não** — é recusa |
    | valor em que o combo já repousa | `False` | **sim** — aceitou sem mudar |
    | valor novo e válido | `True` | sim |
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    repouso = painel.corretor.currentText()
    outro = next(c for c in CORRETORES if c != repouso)

    # (a) recusa: o valor não está na lista do combo.
    assert not painel.pre_preenche("correction", "metodo_que_nao_existe")
    assert painel.corretor.currentText() == repouso, "o combo aceitou um valor fora da lista"
    assert "correction" not in painel.proposta_da_maquina(), (
        "um valor que o widget RECUSOU ficou registado como proposta da máquina — e "
        "`eixos_divergentes` passa a reportar sobrescrita contra algo que a tela nunca mostrou"
    )
    assert painel.eixos_divergentes() == (), "sobrescrita fantasma: ninguém mexeu em nada"

    # (b) aceitou SEM mudar: o combo já repousava no valor proposto.
    assert not painel.pre_preenche("correction", repouso), (
        "âncora: propor o valor de repouso mudou o combo, e este caso não é o que ele parece"
    )
    assert painel.proposta_da_maquina().get("correction") == repouso, (
        "a proposta igual ao repouso foi lida como RECUSA e não ficou registada — a escolha "
        "seguinte do profissional deixaria de contar como sobrescrita"
    )
    painel.corretor.setCurrentText(outro)
    assert painel.eixos_divergentes() == ("correction",), (
        "o profissional mudou o valor contra uma proposta registada, e isso não contou"
    )


def test_d378_eixo_sem_proposta_da_maquina_nunca_entra_em_campos_sobrescritos(
    qtbot: Any,
) -> None:
    """Os sete eixos que rota nenhuma propõe, e por que a ausência deles é DESENHO.

    Os quatro da grade, os dois cortes e a variante do RSD não são propostos por rota nenhuma. Sem o
    lado da máquina não há par de concordância a gravar, e `/execucoes` recusaria com `422` um eixo
    sobrescrito sem proposta com que comparar — o valor escolhido continua a viajar na spec, o que
    não viaja é a **marca** de que ele sobrescreveu algo.

    **A âncora é a não-vacuidade da classe**: se `CONTROLES_DE_SOBRESCRITA` só tivesse eixos
    propostos, o laço não correria e o teste passaria sem olhar eixo nenhum.
    """
    from refino.interface.fluxo import PROPOSTAS_POR_ROTA

    pipeline = PainelDePipeline()
    qtbot.addWidget(pipeline)

    propostos = {eixo for eixo, _campo in PROPOSTAS_POR_ROTA.values()}
    sem_proposta = [c.eixo for c in CONTROLES_DE_SOBRESCRITA if c.eixo not in propostos]

    assert sem_proposta, (
        "âncora: todo eixo de sobrescrita tem proposta da máquina, e este teste não separa nada"
    )

    widgets = pipeline._controles_de_valor()
    mexidos = 0
    for eixo in sem_proposta:
        widget = widgets[eixo]
        if hasattr(widget, "itemText"):
            outros = [
                widget.itemText(i)
                for i in range(widget.count())
                if widget.itemText(i) != widget.currentText()
            ]
            if not outros:
                continue
            widget.setCurrentText(outros[0])
        else:
            widget.setValue(widget.value() + 1)
        mexidos += 1

    # **Conta os que foram DE FACTO mexidos, e não os que a lista tem.** O `continue` acima salta
    # combo sem segunda opção; se todos caíssem nele, a asserção de ausência abaixo passaria sem
    # nada ter sido mexido — a vacuidade a entrar pela porta do laço em vez da do conjunto.
    assert mexidos, (
        f"nenhum dos {len(sem_proposta)} eixos sem proposta pôde ser mexido, e a asserção abaixo "
        f"passaria por vacuidade"
    )

    divergentes = pipeline.eixos_divergentes()
    assert divergentes == (), (
        f"eixos sem proposta da máquina entraram em `campos_sobrescritos`: {divergentes}. A rota "
        f"recusaria com `422` — não há proposta com que comparar"
    )


# ---------------------------------------------------------------------------
# O PALPITE de duração, e o teto de espera removido
# ---------------------------------------------------------------------------
#
# Os dois andam juntos, desde uma espera de 3.659 s sem resposta: o palpite de quanto cada ação
# vai levar, pelo tamanho da entrada, e o fim do teto de 3000 s.


def test_d380_o_palpite_escala_com_o_tamanho_da_entrada(qtbot: Any) -> None:
    """O palpite cresce com `n_features`, e a proporção é a que a medição declara.

    ## Por que a asserção é sobre PROPORÇÃO e não sobre o número

    Afirmar *"6.035 features dão 40 min"* congelaria a constante medida dentro do teste, e a
    constante é a coisa que se pode querer remedir noutra máquina. O que **não** pode mudar sem
    aviso é a forma: o custo é linear em `n_features`, e foi isso que a medição de 06/09/2026
    estabeleceu — no `pos_hilic`, 100 e 200 features dão 362,6 e 358,7 ms/feature.

    **A tolerância é frouxa de propósito.** Ela não mede a linearidade da implementação; mede que
    ninguém trocou a multiplicação por um degrau ou por um número fixo. Uma tolerância apertada
    reprovaria no dia em que a constante mudasse por remedição legítima.
    """
    from refino.desktop.janela.pipeline import EM_REGIME_SERIAL
    from refino.interface.fluxo import estimativa_da_medicao, prosa_de_duracao

    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    # A âncora é o repouso: sem `n_features` não há número, e há uma frase que diz onde ele vem.
    repouso = painel.rotulo_do_palpite.text()
    assert "palpite:" not in repouso, (
        f"o painel mostra um palpite antes de saber o tamanho da entrada: {repouso!r}"
    )
    assert "Validação" in repouso, f"o repouso não diz de onde vem o número: {repouso!r}"

    pequeno = painel.mostra_palpite_da_medicao(1_000)
    assert pequeno == painel.rotulo_do_palpite.text(), "o método devolveu texto que não escreveu"
    assert "1.000 features" in pequeno, f"o tamanho não aparece na frase: {pequeno!r}"

    grande = painel.mostra_palpite_da_medicao(4_000)
    assert grande != pequeno, "quadruplicar a entrada não mudou o palpite"

    # **A proporção é medida NOS DOIS REGIMES, e a primeira escrita media só um.**
    #
    # Ela comparava a frase com `4_000 * sum(CUSTO_POR_FEATURE_MS.values())`, que é a soma CRUA —
    # o custo do regime **paralelo**. O painel não escolhe: ele lê a máquina, e sob
    # `LOKY_MAX_CPU_COUNT=1` — o contorno que torna a suíte completa executável neste clone —
    # `EM_REGIME_SERIAL` é verdadeiro, o `serrf` leva o fator de 5,5x, e o painel mostra **2,3 h**
    # onde a asserção exigia **27 min**. O teste reprovava por comparar dois regimes, não por o
    # palpite estar errado.
    #
    # O número estava certo e a POPULAÇÃO era outra — aqui a
    # população é o regime de paralelismo em que o custo foi somado.
    #
    # A propriedade que o teste existe para defender **não depende do regime**: o custo é linear em
    # `n_features`. Medi-la nos dois fecha a porta a um degrau que só aparecesse num deles, e a
    # âncora afirma que os dois regimes de facto DIFEREM — sem isso o laço mediria a mesma coisa
    # duas vezes e a metade serial ficaria inerte.
    from refino.interface.fluxo import CUSTO_POR_FEATURE_MS, FATOR_SERIAL_DO_SERRF

    def _por_feature_ms(serial: bool) -> float:
        return sum(
            custo * (FATOR_SERIAL_DO_SERRF if serial and corretor == "serrf" else 1.0)
            for corretor, custo in CUSTO_POR_FEATURE_MS.items()
        )

    assert _por_feature_ms(False) > 0, "âncora: a tabela de custo está zerada e nada escalaria"
    assert _por_feature_ms(True) > _por_feature_ms(False), (
        "âncora: os dois regimes custam o mesmo, e então o laço abaixo mede um só"
    )
    for serial in (False, True):
        esperado = prosa_de_duracao(4_000 * _por_feature_ms(serial) / 1000)
        assert esperado in estimativa_da_medicao(4_000, serial=serial), (
            f"o palpite de 4.000 features não é o que a constante medida prevê no regime "
            f"serial={serial}: esperava {esperado!r}"
        )

    # **E a frase do PAINEL traz o número do regime em que ele está**, que é o que o profissional
    # lê. Separado do laço de propósito: acima mede-se a aritmética, aqui a ligação.
    # A janela sobe a primeira letra da frase de `interface/fluxo.py`, que abre a linha.
    assert grande == com_maiuscula(estimativa_da_medicao(4_000, serial=EM_REGIME_SERIAL)), (
        f"o painel não mostra o palpite do regime desta máquina (serial={EM_REGIME_SERIAL}): "
        f"{grande!r}"
    )

    # E volta ao repouso quando a entrada some — não fica a mostrar o número da matriz anterior.
    assert "palpite:" not in painel.mostra_palpite_da_medicao(None), (
        "o painel manteve o palpite da matriz anterior depois de a entrada sumir"
    )
    assert estimativa_da_medicao(0, serial=False) == estimativa_da_medicao(None, serial=False), (
        "zero features e ausência de contagem produzem frases diferentes, e as duas são a mesma "
        "ausência de número"
    )


def test_d380_o_palpite_do_executavel_e_maior_e_o_fator_e_o_medido() -> None:
    """Os dois regimes da mesma máquina, e a diferença é o fator medido no `serrf`.

    O executável congelado paraleliza; a série é o que se obtém desticando a caixa «usar vários
    núcleos», e nela o `serrf` custa **6,18x** mais, fator medido nos quatro modos dentro do
    binário. O teste afirma que existem dois regimes, que a frase diz em qual está, e que só o
    `serrf` recebe o fator.

    A frase diz *"o regime de UM núcleo, que é o da caixa desticada"*: a asserção procura
    `um núcleo` e `desticada`, que é o vocabulário da tela. Quem tem a caixa ticada e lesse um
    palpite do regime serial veria um número **6,2x alto** e concluiria que o programa é lento.

    **Só o `serrf` recebe o fator**, e a asserção afirma isso: o `none` não corre nada, e o LOESS,
    que lê a caixa, não tem fator medido dentro do executável — o custo dele na tabela é
    o da série.
    """
    from refino.interface.fluxo import (
        CUSTO_POR_FEATURE_MS,
        FATOR_SERIAL_DO_SERRF,
        estimativa_da_medicao,
    )

    assert FATOR_SERIAL_DO_SERRF > 1, "âncora: o fator não separa os regimes"
    assert set(CUSTO_POR_FEATURE_MS) == {"none", "loess", "serrf"}, (
        f"a tabela de custo mudou de população: {sorted(CUSTO_POR_FEATURE_MS)}. O fator serial é "
        f"aplicado por nome, e um corretor novo entraria sem ninguém decidir se ele o recebe"
    )

    paralelo = estimativa_da_medicao(6_035, serial=False)
    serial = estimativa_da_medicao(6_035, serial=True)

    assert paralelo != serial, "os dois regimes produzem o mesmo palpite"
    # **`desticada` e não `serial`, e a escolha da palavra é o conteúdo.** O que a frase precisa de
    # nomear é o regime em vocabulário de quem usa: a caixa que ele vê na tela. `serial` descrevia
    # o empacotamento — e o empacotamento deixou de decidir isto em 08/09/2026.
    assert "desticada" in serial and "desticada" not in paralelo, (
        f"a frase não diz em que regime o número foi calculado: {serial!r}"
    )

    # O fator entra SÓ no `serrf`: o total serial é o paralelo mais o acréscimo daquele corretor.
    esperado_ms = sum(CUSTO_POR_FEATURE_MS.values()) + CUSTO_POR_FEATURE_MS["serrf"] * (
        FATOR_SERIAL_DO_SERRF - 1
    )
    from refino.interface.fluxo import prosa_de_duracao

    assert prosa_de_duracao(6_035 * esperado_ms / 1000) in serial, (
        f"o palpite serial não é o paralelo com o fator só no `serrf`: {serial!r}"
    )


def test_d380_a_janela_nao_desiste_de_esperar_por_padrao(qtbot: Any) -> None:
    """O teto saiu, e o mecanismo FICA para quem o puser — as duas metades numa asserção.

    **Par causal:** a janela padrão não tem teto; a que recebe um tem-no derivado. Sem a segunda
    metade, remover o mecanismo inteiro passaria por esta — e os testes que medem a desistência
    perderiam o único caminho que a produz.
    """
    padrao = JanelaPrincipal(_TransporteQueLevanta(RuntimeError("não usado")))
    qtbot.addWidget(padrao)
    assert padrao.leituras_maximas is None, (
        f"a janela padrão desiste depois de {padrao.leituras_maximas} leituras: o teto voltou, e "
        f"com ele a tela a dizer «parei de esperar» sobre um trabalho vivo"
    )

    com_teto = JanelaPrincipal(
        _TransporteQueLevanta(RuntimeError("não usado")),
        intervalo_da_espera_ms=50,
        teto_da_espera_ms=150,
    )
    qtbot.addWidget(com_teto)
    assert com_teto.leituras_maximas == 3, (
        "o mecanismo do teto saiu junto com o padrão — os testes que medem a desistência ficam "
        "sem o único caminho que a produz"
    )


# ---------------------------------------------------------------------------
# A legenda das classes de defeito, na aba de preparo
# ---------------------------------------------------------------------------
#
# **A lacuna e o espaço vêm juntos:** a tela dava o diagnóstico como «errático» sem dizer o que
# isso é, e abaixo dos arquivos e da validação havia espaço vazio para uma tabela com os tipos de
# diagnóstico e o que cada um significa.


def _aba_de_preparo(qtbot: Any) -> Any:
    """A aba montada com os dois painéis reais, e o escolhedor que nunca é chamado."""
    from refino.desktop.janela.leitura import PainelDeLeitura
    from refino.desktop.janela.preparo import AbaDePreparo

    aba = AbaDePreparo(
        PainelDeArquivos(lambda _titulo: None),
        PainelDeLeitura(),
        {"arquivos": "Os seus arquivos", "leitura": "Validação, diagnóstico e palpite"},
    )
    qtbot.addWidget(aba)
    return aba


def test_u26_a_legenda_cobre_todas_as_classes_de_defeito_e_deriva_da_constante(
    qtbot: Any,
) -> None:
    """Bijeção contra `DefectType`, nas duas direções, e a legenda deriva da constante.

    A legenda existe para responder *"o que é errático?"* — e uma que cobrisse quatro dos cinco
    deixaria de fora exactamente a classe que o profissional não reconhece. **Derivar da constante
    é o que impede a divergência**; este portão é o que impede alguém de voltar a escrevê-la à mão
    e de a lista ficar para trás.

    A constante passou a ser `INTERPRETACAO_DO_SINAL`, com o texto do laboratório, e a
    bijeção é conferida pela função que a tela usa, com uma planta de cada lado.

    **A âncora de não-vacuidade vem antes da bijeção e na MESMA função**: dois conjuntos
    vazios são iguais, e conjunto vazio é o que se obtém de uma legenda que não desenhou linha
    nenhuma.
    """
    from refino.domain.labels import DefectType
    from refino.interface.fluxo import INTERPRETACAO_DO_SINAL, classes_sem_interpretacao

    aba = _aba_de_preparo(qtbot)

    assert INTERPRETACAO_DO_SINAL, "âncora: a constante da legenda está vazia"
    assert aba.nomes_da_legenda, "âncora: a legenda não desenhou linha nenhuma"
    assert classes_sem_interpretacao() == (frozenset(), frozenset())
    assert set(aba.nomes_da_legenda) == set(INTERPRETACAO_DO_SINAL) == set(get_args(DefectType))

    sem_erratico = {k: v for k, v in INTERPRETACAO_DO_SINAL.items() if k != "erratic"}
    assert classes_sem_interpretacao(sem_erratico) == (frozenset({"erratic"}), frozenset())
    com_orfa = {**INTERPRETACAO_DO_SINAL, "orfa": INTERPRETACAO_DO_SINAL["clean"]}
    assert classes_sem_interpretacao(com_orfa) == (frozenset(), frozenset({"orfa"}))


def test_u26_a_legenda_mostra_o_texto_do_laboratorio_na_ordem_dele(qtbot: Any) -> None:
    """O título, a introdução e, por classe, os campos que o laboratório escreveu.

    O que a tela desenha é comparado com a constante campo a campo, e a constante com os trechos
    do texto do laboratório que identificam cada parte: uma transcrição que trocasse uma resposta de
    lugar, ou pulasse um campo, reprova aqui.
    """
    from refino.desktop.janela.preparo import TITULO_DA_LEGENDA
    from refino.interface.fluxo import (
        INTERPRETACAO_DO_SINAL,
        INTRODUCAO_DA_INTERPRETACAO_DO_SINAL,
    )

    aba = _aba_de_preparo(qtbot)

    assert TITULO_DA_LEGENDA == "Como interpretar o comportamento do sinal analítico"
    assert aba.legenda.alterna.text() == TITULO_DA_LEGENDA
    assert [r.text() for r in aba.introducao_da_legenda] == list(
        INTRODUCAO_DA_INTERPRETACAO_DO_SINAL
    )
    assert len(INTRODUCAO_DA_INTERPRETACAO_DO_SINAL) == 3
    assert INTRODUCAO_DA_INTERPRETACAO_DO_SINAL[1].startswith("O Refino utiliza")

    assert list(aba.nomes_da_legenda) == list(INTERPRETACAO_DO_SINAL), "a ordem não é a do texto"
    for defeito, classe in INTERPRETACAO_DO_SINAL.items():
        assert aba.nomes_da_legenda[defeito].text() == f"{classe.nome} — {classe.resumo}"
        desenhados = [(q.text(), r.text()) for q, r in aba.campos_da_legenda[defeito]]
        assert desenhados == list(classe.campos), f"os campos de `{defeito}` divergem"
        assert desenhados[0][0] == "O que significa?" and desenhados[-1][0] == "O que fazer?"

    perguntas = {d: [q for q, _ in c.campos] for d, c in INTERPRETACAO_DO_SINAL.items()}
    assert "Isso significa que os dados estão bons?" in perguntas["clean"]
    assert "Por que isso pode acontecer?" in perguntas["unassessable"]
    assert "Exemplo prático" not in perguntas["clean"] + perguntas["unassessable"]
    assert (
        INTERPRETACAO_DO_SINAL["erratic"]
        .campos[1][1]
        .startswith("O mesmo composto apresenta intensidades de 1.000, 500, 1.200, 700 e 1.100")
    )


def test_u26_a_legenda_nao_poe_negrito_por_cima_do_que_ja_o_tem(qtbot: Any) -> None:
    """O defeito que a aba MONTADA revelou, e a forma que a legenda nova tomou.

    A primeira legenda envolvia todo nome em `**`, e o `unassessable`, que chega de
    `ROTULOS_DE_DEFEITO` já com ênfase, saía `****não avaliável****`. A legenda nova é
    texto simples: o negrito da pergunta vem do tema, e um asterisco no texto do laboratório seria
    desenhado literal. O que este teste guarda passou a ser isso — nenhum rótulo em markdown, e
    nenhum texto da constante com marca de markdown —, com a planta do segundo lado.
    """
    from PySide6.QtCore import Qt

    from refino.interface.fluxo import INTERPRETACAO_DO_SINAL, InterpretacaoDoSinal

    aba = _aba_de_preparo(qtbot)

    # **A âncora da LEGENDA vive nesta função**: sem campos desenhados, o laço não corre
    # e a asserção de ausência passa por vacuidade.
    rotulos = [r for campos in aba.campos_da_legenda.values() for par in campos for r in par]
    assert rotulos, "âncora: a legenda não desenhou campo nenhum"
    assert all(r.textFormat() == Qt.TextFormat.PlainText for r in rotulos)

    def com_marca(mapa: dict[str, InterpretacaoDoSinal]) -> list[str]:
        textos = [t for c in mapa.values() for par in c.campos for t in par]
        textos += [t for c in mapa.values() for t in (c.nome, c.resumo)]
        return [t for t in textos if "*" in t or "`" in t]

    assert com_marca(INTERPRETACAO_DO_SINAL) == []
    plantado = {
        "clean": InterpretacaoDoSinal("**Sem padrão**", "x", (("O que significa?", "y"),)),
    }
    assert com_marca(plantado) == ["**Sem padrão**"], "a planta não separou nada"


# ---------------------------------------------------------------------------
# A importação dos fatores na página «Normalização», e o arquivo de exemplo
# ---------------------------------------------------------------------------


def _janela_com_fatores(
    qtbot: Any, escolhido: Path | None, destino: Path | None = None
) -> JanelaPrincipal:
    """A janela com um escolhedor que devolve `escolhido` e um salvador que devolve `destino`."""
    janela = JanelaPrincipal(
        _TransporteQueLevanta(ConnectionRefusedError("recusado")),
        escolhedor=lambda _titulo: escolhido,
        arquivo=lambda _nome: destino,
    )
    qtbot.addWidget(janela)
    return janela


def test_d654_os_fatores_moram_na_normalizacao_e_sairam_do_preparo(
    qtbot: Any, tmp_path: Path
) -> None:
    """O botão sai de «Preparar o estudo» e vai à aba de normalização.

    **O estado não mudou de dono**: o arquivo escolhido na página nova chega a
    `arquivos_da_execucao`, que é o que `/execucoes` recebe. Sem isso, a moldura mudaria de aba e
    o fator deixaria de viajar sem nada acusar.
    """
    from PySide6.QtWidgets import QLabel

    from refino.desktop.janela.arquivos import (
        FORMATO_DOS_FATORES,
        ROTULO_DOS_FATORES,
        TEXTO_DOS_FATORES,
    )
    from refino.desktop.janela.preparo import AbaDePreparo

    fatores = tmp_path / "fatores.csv"
    fatores.write_bytes(b"amostra;fator\nS0;1,5\n")
    janela = _janela_com_fatores(qtbot, fatores)
    preparo = janela.findChild(AbaDePreparo)
    assert preparo is not None, "âncora: a aba de preparo não foi montada"
    botao = janela.arquivos.botao_fatores

    assert botao.text() == ROTULO_DOS_FATORES == "Importar fatores de normalização (opcional)"
    assert janela.pagina_da_normalizacao.isAncestorOf(botao), "o botão não está na normalização"
    assert not preparo.isAncestorOf(botao), "o botão continua em «Preparar o estudo»"
    assert janela.pagina_da_normalizacao.isAncestorOf(janela.arquivos.botao_do_exemplo)

    textos = [r.text() for r in janela.arquivos.importacao_dos_fatores.findChildren(QLabel)]
    assert TEXTO_DOS_FATORES in textos and FORMATO_DOS_FATORES in textos
    assert "No MetaboAnalyst: `Sample-specific normalization (i.e. weight, volume)`" in textos

    # O clique na página nova chega ao estado do painel dos arquivos, que é o que
    # `arquivos_da_execucao` envia; o envio com o par está em `tests/test_desktop_lovable.py`.
    botao.click()
    assert janela.arquivos._fatores == (fatores.name, fatores.read_bytes())
    assert janela.arquivos.estados["fatores"].text() == fatores.name


def test_os_textos_dos_fatores_nomeiam_a_opcao_como_o_seletor_a_mostra(qtbot: Any) -> None:
    """«Pelos fatores fornecidos», e não `supplied_factors`.

    O seletor de «Normalização por amostra» mostra o nome legível de `NOMES_DAS_OPCOES` por cima
    do valor do domínio, e o formato do arquivo e a dica do botão mandavam pôr o seletor
    *«em supplied_factors»*, uma palavra que a tela não mostra em lado nenhum. **O nome esperado
    sai do mapa que o seletor desenha**, e o par em processo é a frase de antes.
    """
    from refino.desktop.janela.arquivos import FORMATO_DOS_FATORES, PainelDeArquivos
    from refino.desktop.janela.nomes import NOMES_DAS_OPCOES

    nome = NOMES_DAS_OPCOES["supplied_factors"]
    painel = PainelDeArquivos(lambda _titulo: None)
    qtbot.addWidget(painel)
    for rotulo, texto in (
        ("formato", FORMATO_DOS_FATORES),
        ("dica", painel.botao_fatores.toolTip()),
    ):
        assert f"«{nome}»" in texto and "supplied_factors" not in texto, f"{rotulo}: {texto!r}"
    antiga = "Só age com «Normalização por amostra» em supplied_factors."
    assert f"«{nome}»" not in antiga and "supplied_factors" in antiga, "o par não separa nada"


def test_o_cartao_da_lista_de_amostras_descreve_o_que_o_leitor_le(
    qtbot: Any, tmp_path: Path
) -> None:
    """O subtítulo do cartão dizia *«Metadados: grupo, lote, ordem de injeção»*.

    `read_sample_list` lê DUAS colunas, a ordem de injeção e o nome da amostra; o tipo e o
    grupo vêm do arquivo de medições, e os lotes declaram-se em «Correção». **A âncora mostra que
    terceiras colunas chamadas tipo, grupo ou lote não chegam a lado nenhum:** o leitor devolve o
    mesmo com e sem elas. Um texto que promete tipo, grupo ou lote promete o que o arquivo não leva
    ao programa.

    **A população são os três textos do cartão, e não só o subtítulo** — a revisão achou a irmã na
    DICA: o subtítulo foi corrigido, e a dica do mesmo rótulo e a do botão «Selecionar arquivo»
    continuavam a dizer *«A ordem de injeção e o tipo de cada amostra»*. O que se confere é a
    primeira frase de cada um, que é a que diz o que o arquivo traz; a frase seguinte da dica pode
    dizer de onde vem o resto.
    """
    import re

    from PySide6.QtWidgets import QFrame, QLabel

    from refino.desktop.janela.arquivos import PainelDeArquivos
    from refino.io.readers import read_sample_list

    duas = tmp_path / "duas.csv"
    duas.write_text("ID,inj_order_name\n1,QC1\n2,A1\n", encoding="utf-8")
    cinco = tmp_path / "cinco.csv"
    cinco.write_text(
        "ID,inj_order_name,tipo,grupo,lote\n1,QC1,qc,QC,A\n2,A1,sample,X,A\n", encoding="utf-8"
    )
    assert read_sample_list(duas) == read_sample_list(cinco) == {"QC1": 1, "A1": 2}

    def promete(texto: str) -> list[str]:
        primeira = re.split(r"(?<=\.)\s+", " ".join(texto.split()))[0].lower()
        return [p for p in ("tipo", "grupo", "lote") if re.search(rf"\b{p}", primeira)]

    painel = PainelDeArquivos(lambda _titulo: None)
    qtbot.addWidget(painel)
    rotulos = [
        r
        for cartao in painel.findChildren(QFrame, "cartao_de_arquivo")
        if painel.botao_lista in cartao.findChildren(type(painel.botao_lista))
        for r in cartao.findChildren(QLabel, "descricao_do_arquivo")
    ]
    assert len(rotulos) == 1, f"âncora: o cartão da lista tem {[r.text() for r in rotulos]}"
    (rotulo,) = rotulos
    textos = {
        "subtítulo": rotulo.text(),
        "dica do subtítulo": rotulo.toolTip(),
        "dica do botão": painel.botao_lista.toolTip(),
    }
    for nome, texto in textos.items():
        assert "ordem de injeção" in texto.lower(), f"âncora: a {nome} não diz a ordem: {texto!r}"
        assert promete(texto) == [], (
            f"a {nome} do cartão da lista promete {promete(texto)}: {texto!r}"
        )

    assert promete("A ordem de injeção e o tipo de cada amostra.") == ["tipo"], (
        "a dica de antes não foi acusada"
    )
    assert promete("Metadados: grupo, lote, ordem de injeção") == ["grupo", "lote"], (
        "o subtítulo de antes não foi acusado"
    )


def test_d654_o_exemplo_e_lido_pelo_leitor_e_tem_fator_para_cada_nome() -> None:
    """O exemplo que viaja tem de passar pelo leitor que vai recebê-lo de volta.

    Nomes genéricos (nenhum nome de amostra do laboratório), um por linha, QCs incluídos, e todo
    fator positivo e finito, que é o que a guarda de `PipelineSpec` exige.
    """
    import math

    from refino.desktop.janela.arquivos import NOME_DO_EXEMPLO
    from refino.desktop.janela.tema import pasta_de_recursos
    from refino.io.fatores import le_fatores_de_bytes

    pares = le_fatores_de_bytes(
        (pasta_de_recursos() / NOME_DO_EXEMPLO).read_bytes(), origem=NOME_DO_EXEMPLO
    )
    nomes = [nome for nome, _ in pares]
    assert len(nomes) == len(set(nomes)) >= 4, "âncora: o exemplo tem poucas linhas ou repete nome"
    assert any(n.startswith("QC_") for n in nomes), "o exemplo não mostra que o QC precisa de fator"
    assert any(n.startswith("AMOSTRA_") for n in nomes)
    assert all(math.isfinite(f) and f > 0 for _, f in pares)


def test_d654_o_botao_do_exemplo_grava_uma_copia_igual_e_diz_onde(
    qtbot: Any, tmp_path: Path
) -> None:
    """A cópia é byte a byte a do recurso, e a frase diz o destino; cancelar não escreve nada."""
    from refino.desktop.janela.arquivos import NOME_DO_EXEMPLO
    from refino.desktop.janela.tema import pasta_de_recursos

    destino = tmp_path / "meu-modelo.csv"
    janela = _janela_com_fatores(qtbot, None, destino)
    janela.arquivos.botao_do_exemplo.click()
    assert destino.read_bytes() == (pasta_de_recursos() / NOME_DO_EXEMPLO).read_bytes()
    assert str(destino) in janela.arquivos.aviso_dos_fatores.text()

    cancelada = _janela_com_fatores(qtbot, None, None)
    assert cancelada.arquivos.salva_exemplo_dos_fatores() is None
    assert cancelada.arquivos.aviso_dos_fatores.isHidden(), "cancelar não é falha"


def test_d654_a_falha_de_ler_os_fatores_sai_na_linha_deles(qtbot: Any, tmp_path: Path) -> None:
    """A frase vai para a aba onde o botão foi clicado, e não para o rótulo do par no preparo."""
    sumido = tmp_path / "sumiu.csv"
    janela = _janela_com_fatores(qtbot, sumido)
    janela.arquivos.botao_fatores.click()
    assert "Não foi possível ler sumiu.csv" in janela.arquivos.aviso_dos_fatores.text()
    assert not janela.arquivos.aviso_dos_fatores.isHidden()
    assert janela.arquivos.rotulo.isHidden(), "a falha dos fatores saiu no rótulo do par"
    assert janela.arquivos._fatores is None


def test_d654_o_exemplo_que_nao_pode_ser_gravado_diz_por_que(qtbot: Any, tmp_path: Path) -> None:
    """A outra ação: a escrita falhou, e a pessoa precisa de escolher outro lugar."""
    destino = tmp_path / "nao-existe" / "modelo.csv"
    janela = _janela_com_fatores(qtbot, None, destino)
    assert janela.arquivos.salva_exemplo_dos_fatores() is None
    assert not destino.exists()
    assert "Não foi possível salvar o exemplo" in janela.arquivos.aviso_dos_fatores.text()


# ---------------------------------------------------------------------------
# O modo deduzido do nome do arquivo ou dos nomes das amostras, em «Opções avançadas»
# ---------------------------------------------------------------------------

_CABECALHO_NEG_C18: Final = b"NAME,X_NEG_C18_01,X_NEG_C18_02,QC_NEG_C18_01\nf1,1,2,3\n"
_CABECALHO_SEM_TOKEN: Final = b"NAME,amostra_01,amostra_02,qc_01\nf1,1,2,3\n"


def _painel_com(qtbot: Any, tmp_path: Path, nome: str, conteudo: bytes) -> Any:
    """O painel com o arquivo de medições `nome` e uma lista sem modo no nome, já escolhidos."""
    medidas = tmp_path / nome
    medidas.write_bytes(conteudo)
    lista = tmp_path / "lista.csv"
    lista.write_bytes(b"ID,inj_order_name\n1,x\n")
    por_titulo = {ROTULO_DAS_MEDIDAS: medidas, ROTULO_DA_LISTA: lista}
    painel = PainelDeArquivos(por_titulo.get)
    qtbot.addWidget(painel)
    painel.escolhe_medidas()
    assert painel.escolhe_lista() is not None, "âncora: o par não se formou"
    return painel


def test_d656_o_modo_no_nome_do_arquivo_vale_e_a_tela_diz_a_origem(
    qtbot: Any, tmp_path: Path
) -> None:
    """O nome do arquivo decide, mesmo quando as amostras dizem outro modo — e a tela diz qual."""
    from refino.interface.fluxo import ORIGEM_NO_NOME_DO_ARQUIVO

    painel = _painel_com(qtbot, tmp_path, "measurements_pos_hilic.csv", _CABECALHO_NEG_C18)
    assert painel.modo.currentText() == "pos_hilic"
    assert painel.origem_do_modo == ORIGEM_NO_NOME_DO_ARQUIVO
    assert painel.rotulo_do_modo.text() == "Modo: pos_hilic (do nome do arquivo)"
    assert painel.campos_para_envio() == {"mode": "pos_hilic"}


def test_d656_par_causal_o_mesmo_arquivo_com_dois_nomes_da_duas_origens(
    qtbot: Any, tmp_path: Path
) -> None:
    """Os mesmos bytes, com o modo no nome e sem ele: a origem e o modo mudam com o nome só.

    É a planta que separa a dedução pelo nome da dedução pelas amostras: com o mesmo conteúdo, só o
    nome do arquivo difere, e as duas linhas da tela têm de divergir.
    """
    from refino.interface.fluxo import ORIGEM_NO_NOME_DO_ARQUIVO, ORIGEM_NOS_NOMES_DAS_AMOSTRAS

    pelo_nome = _painel_com(qtbot, tmp_path, "measurements_pos_hilic.csv", _CABECALHO_NEG_C18)
    pelas_amostras = _painel_com(qtbot, tmp_path, "medidas.csv", _CABECALHO_NEG_C18)

    assert (pelo_nome.modo.currentText(), pelo_nome.origem_do_modo) == (
        "pos_hilic",
        ORIGEM_NO_NOME_DO_ARQUIVO,
    )
    assert (pelas_amostras.modo.currentText(), pelas_amostras.origem_do_modo) == (
        "neg_c18",
        ORIGEM_NOS_NOMES_DAS_AMOSTRAS,
    )
    assert pelo_nome.rotulo_do_modo.text() != pelas_amostras.rotulo_do_modo.text()


def test_d656_sem_deducao_o_campo_fica_vazio_e_a_chamada_e_recusada(
    qtbot: Any, tmp_path: Path
) -> None:
    """Nenhum padrão silencioso: sem modo, a chamada é recusada e «Opções avançadas» abre."""
    from refino.desktop.janela.arquivos import MODO_AUSENTE, ModoAusenteError

    painel = _painel_com(qtbot, tmp_path, "medidas.csv", _CABECALHO_SEM_TOKEN)
    assert painel.modo.currentText() == ""
    assert painel.origem_do_modo is None
    assert "não deduzido" in painel.rotulo_do_modo.text()
    assert not painel.opcoes_avancadas.alterna.isChecked(), "âncora: o bloco nasce fechado"

    with pytest.raises(ModoAusenteError, match="Falta o modo do estudo") as recusa:
        painel.campos_para_envio()
    assert str(recusa.value) == MODO_AUSENTE
    assert painel.opcoes_avancadas.alterna.isChecked(), "a recusa não abriu «Opções avançadas»"

    painel.segmentos_do_modo["neg_hilic"].click()
    assert painel.campos_para_envio() == {"mode": "neg_hilic"}
    assert painel.rotulo_do_modo.text() == "Modo: neg_hilic (escolhido à mão)"


def test_d656_a_escolha_a_mao_vence_a_deducao(qtbot: Any, tmp_path: Path) -> None:
    """Quem escolheu um modo não o perde por ter escolhido um arquivo depois."""
    from refino.interface.fluxo import ORIGEM_ESCOLHIDA_A_MAO

    medidas = tmp_path / "measurements_pos_hilic.csv"
    medidas.write_bytes(_CABECALHO_NEG_C18)
    lista = tmp_path / "lista.csv"
    lista.write_bytes(b"ID,inj_order_name\n1,x\n")
    painel = PainelDeArquivos({ROTULO_DAS_MEDIDAS: medidas, ROTULO_DA_LISTA: lista}.get)
    qtbot.addWidget(painel)
    painel.modo.setEditText("pos_lipidomica")
    painel.escolhe_medidas()
    painel.escolhe_lista()
    assert painel.modo.currentText() == "pos_lipidomica"
    assert painel.origem_do_modo == ORIGEM_ESCOLHIDA_A_MAO


def test_d656_a_janela_recusa_sem_modo_e_leva_ao_preparo(qtbot: Any, tmp_path: Path) -> None:
    """A recusa leva ao campo: a aba do preparo, com «Opções avançadas» aberto."""
    from refino.desktop.janela.preparo import AbaDePreparo

    medidas = tmp_path / "medidas.csv"
    medidas.write_bytes(_CABECALHO_SEM_TOKEN)
    lista = tmp_path / "lista.csv"
    lista.write_bytes(b"ID,inj_order_name\n1,x\n")
    janela = JanelaPrincipal(
        _TransporteQueLevanta(ConnectionRefusedError("recusado")),
        escolhedor={ROTULO_DAS_MEDIDAS: medidas, ROTULO_DA_LISTA: lista}.get,
    )
    qtbot.addWidget(janela)
    janela.arquivos.escolhe_medidas()
    janela.arquivos.escolhe_lista()
    janela.abas.setCurrentWidget(janela.pagina_da_correcao)

    assert janela._pedido_de_arquivos() is None
    assert isinstance(janela.abas.currentWidget(), AbaDePreparo), "a recusa não levou ao preparo"
    assert janela.arquivos.opcoes_avancadas.alterna.isChecked()
    assert "Falta o modo do estudo" in janela.aviso_de_desfecho.rotulo.text()
    assert janela.aviso_de_desfecho.titulo.text() == "Falta o modo do estudo"
    janela.espera_a_piscina(30_000)


def test_d656_a_deducao_pelos_nomes_e_a_maioria_estrita_em_cada_eixo() -> None:
    """Os tokens são observação: a dedução só fala com maioria, e cala no empate."""
    from refino.io.validators import deduce_mode_from_sample_names as deduz

    assert deduz(["A_POS_HILIC_1", "A_POS_HILIC_2", "QC_1"]) == "pos_hilic"
    assert deduz(["A_POS_HILIC_1", "B_NEG_HILIC_1"]) is None, "empate de polaridade"
    assert deduz(["A_POS_NEG_HILIC_1", "B_POS_NEG_HILIC_2"]) is None, "os dois lados não contam"
    assert deduz(["amostra_1", "amostra_2"]) is None
    assert deduz([]) is None


def test_d656_o_modo_no_nome_do_arquivo_exige_as_duas_partes_inteiras() -> None:
    """`pos_hilic` dentro de outra palavra não conta, e as quatro combinações são reconhecidas."""
    from refino.interface.fluxo import modo_do_nome_do_arquivo as do_nome

    assert do_nome("measurements_pos_hilic.csv") == "pos_hilic"
    assert do_nome("Sample_List_NEG_C18.csv") == "neg_c18"
    assert do_nome("estudo-pos_c18-2026.csv") == "pos_c18"
    assert do_nome("neg_hilic.csv") == "neg_hilic"
    assert do_nome("measurements.csv") is None
    assert do_nome("xpos_hilic.csv") is None
    assert do_nome("pos_hilicx.csv") is None


@pytest.mark.parametrize("modo", MODOS_SUGERIDOS)
def test_d656_nos_quatro_arquivos_reais_os_nomes_das_amostras_dao_o_modo(modo: str) -> None:
    """Contagem esperada sobre os quatro arquivos do laboratório, e não invariante.

    Medido em 06/11/2026: as 68 amostras de cada arquivo trazem os dois tokens do modo, e a
    dedução acerta os quatro. Um estudo com outra convenção de nomes cai no campo vazio, que é o
    desfecho desenhado.
    """
    from refino.desktop.janela.arquivos import _nomes_do_cabecalho
    from refino.io.validators import deduce_mode_from_sample_names

    nomes = _nomes_do_cabecalho(measurements_path(modo).read_bytes())
    assert len(nomes) == 68, "âncora: o cabeçalho não trouxe as 68 amostras"
    assert deduce_mode_from_sample_names(nomes) == modo


def test_u26_rotulo_de_defeito_sem_o_separador_levanta_em_vez_de_recuar() -> None:
    """A guarda de `partes_do_defeito`, e ela dispara por AUSÊNCIA — logo ficaria inerte.

    O recuo natural — devolver `("", texto)` — desenharia uma linha com a primeira coluna vazia, e
    célula vazia lê-se como *"ainda não preencheram"* e não como *"o rótulo está malformado"*. É a
    mesma decisão que `rotulo_do_eixo` regista para o mapa irmão.

    **Negativo EM PROCESSO**: a planta é um valor passado à função, e não um arquivo mutado que um
    subprocesso possa não alcançar. Esta forma não consegue ficar inerte.
    """
    from refino.interface import fluxo

    plantado = {"sem_travessao": "um rótulo sem separador nenhum"}

    assert fluxo.partes_do_defeito("erratic", dict(fluxo.ROTULOS_DE_DEFEITO)) == (
        "errático",
        "varia sem direção que a cascata reconheça",
    ), "âncora: a função nem sobre o mapa real devolve as duas partes, e a planta não separa nada"

    with pytest.raises(ValueError, match="não tem o travessão"):
        fluxo.partes_do_defeito("sem_travessao", plantado)


# ---------------------------------------------------------------------------
# Painel 2 — os dois desfechos, e a abstenção
# ---------------------------------------------------------------------------


def test_u4_validar_com_arquivo_bom_e_com_arquivo_ruim_no_mesmo_teste(
    transporte_real: TestClient, qtbot: Any
) -> None:
    """Os **dois** desfechos de `/matrizes/validar`, no mesmo teste.

    ## O plano dizia `422` e a rota responde `200`. A divergência foi MEDIDA e vai declarada

    O cenário do plano pede *"com arquivo inválido: `422`, a lista de problemas na tela"*. Medido em
    22/08/2026 com **sete** corrupções: seis devolvem `200` — `measurements` vazio, só o cabeçalho,
    lista de **outro modo**, lista vazia, CSV sem coluna de feature, lista de uma coluna — e a
    sétima devolve **`500`**, que é um defeito conhecido e não a validação a funcionar.

    > **A rota RELATA em vez de recusar, e é o desenho melhor.** `aceita: false` mais
    > `problemas: [{codigo, severidade, mensagem}]` diz **o quê** está errado; um `422` deixaria o
    > profissional a adivinhar. O teste afirma os dois desfechos que a rota tem, não os que o plano
    > supôs.

    Juntos de propósito: em testes separados, o do arquivo ruim poderia estar a acusar por qualquer
    motivo — arquivo ausente, rota mudada, campo esquecido — e continuaria verde. Lado a lado, a
    asserção é que os desfechos **divergem**.
    """
    painel = PainelDeLeitura()
    qtbot.addWidget(painel)

    bons = {
        "measurements": ("m.csv", measurements_path("neg_c18").read_bytes(), "text/csv"),
        "sample_list": ("s.csv", sample_list_path("neg_c18").read_bytes(), "text/csv"),
    }
    boa = cliente.envia(
        transporte_real, "POST", "/matrizes/validar", arquivos=bons, campos={"mode": "neg_c18"}
    )

    # **A planta é a lista de OUTRO modo**, e não bytes aleatórios: é o erro que o laboratório de
    # facto comete, e — medido — é um dos seis que a rota classifica em vez de explodir. Bytes com
    # linha curta cairiam naquele `500`, e a asserção mediria um defeito da ferramenta.
    ruins = {
        "measurements": ("m.csv", measurements_path("neg_c18").read_bytes(), "text/csv"),
        "sample_list": ("s.csv", sample_list_path("pos_hilic").read_bytes(), "text/csv"),
    }
    ma = cliente.envia(
        transporte_real, "POST", "/matrizes/validar", arquivos=ruins, campos={"mode": "neg_c18"}
    )

    assert boa.codigo == 200, f"o arquivo real foi recusado: {boa.codigo} {boa.corpo}"
    assert ma.codigo == 200, f"a rota deixou de RELATAR e passou a recusar: {ma.codigo}"
    assert boa.corpo["aceita"] is True, f"o par real não foi aceito: {boa.corpo}"
    assert ma.corpo["aceita"] is False, (
        f"o par trocado foi ACEITO — a planta não separou: {ma.corpo}"
    )
    assert ma.corpo["problemas"], "recusou sem dizer por quê: a lista de problemas veio vazia"

    texto_bom = painel.mostra("validacao", boa.corpo)
    texto_ruim = painel.mostra("validacao", ma.corpo)

    assert texto_bom != texto_ruim, "os dois desfechos produzem o MESMO texto na tela"
    assert texto_bom != SEM_LEITURA and texto_ruim != SEM_LEITURA, (
        f"algum desfecho caiu no texto de ausência: bom={texto_bom!r} ruim={texto_ruim!r}"
    )
    assert painel.rotulos["validacao"].text() == texto_ruim, "o painel não recebeu o último texto"


def test_u4_todo_ponto_de_decisao_exibe_justificativa_nao_vazia(
    transporte_real: TestClient, qtbot: Any
) -> None:
    """Todo ponto de decisão automática mostra a justificativa nesta superfície.

    **A enumeração vem da estrutura e a não-vacuidade vai ANTES**: sem a primeira asserção,
    uma `PONTOS_DE_DECISAO_AUTOMATICA` vazia faria o laço não iterar e o teste passar sem olhar
    ponto nenhum.

    **O EMPARELHAMENTO MUDOU DE PORTADOR em 15/09/2026, e o oráculo mudou com ele.** O critério
    exige justificativa *e* controle em CADA ponto, e o que liga a justificativa ao ponto é agora o
    título do `QGroupBox` — não um prefixo dentro do corpo. Ver a construção de `PainelDeLeitura`.
    """
    from PySide6.QtWidgets import QGroupBox

    painel = PainelDeLeitura()
    qtbot.addWidget(painel)

    assert PONTOS_DE_DECISAO_AUTOMATICA, "a estrutura dos pontos de decisão está vazia"

    arquivos = {
        "measurements": ("m.csv", measurements_path("neg_c18").read_bytes(), "text/csv"),
        "sample_list": ("s.csv", sample_list_path("neg_c18").read_bytes(), "text/csv"),
    }
    corpos: dict[str, Any] = {}
    for chave, rota in ROTAS_DA_LEITURA:
        resposta = _le_a_rota_ate_o_fim(
            transporte_real, rota, arquivos=arquivos, campos={"mode": "neg_c18"}
        )
        assert resposta.codigo == 200, f"{rota} respondeu {resposta.codigo}: {resposta.corpo}"
        corpos[chave] = resposta.corpo

    vistos = 0
    for ponto in PONTOS_DE_DECISAO_AUTOMATICA:
        corpo = corpos["diagnostico" if ponto.chave == "diagnostico" else "recomendacao"]
        texto = painel.mostra_justificativa(ponto, corpo)
        assert texto != SEM_LEITURA, (
            f"o ponto `{ponto.chave}` não exibiu justificativa: o campo "
            f"`{ponto.campo_da_justificativa}` veio vazio de {corpo!r}"
        )
        # **A ÂNCORA DE NÃO-VACUIDADE vem antes da asserção de ausência**: *"o corpo não
        # repete o título"* é satisfeito por um corpo VAZIO, e corpo vazio é o desfecho de um
        # formatador que falhou.
        assert texto.strip(), f"o ponto `{ponto.chave}` exibiu justificativa em branco"

        # **O EMPARELHAMENTO É LIDO NA MOLDURA, e até 15/09/2026 era lido no corpo.** A asserção
        # anterior era `ponto.o_que_decide in texto`, e ela nasceu quando o rótulo era filho do
        # painel e o texto dele era o único portador do nome. Desde que o `QGroupBox` por ponto
        # existe, ler só o corpo é verificar a instância errada do referente certo — e o desfecho
        # de sucesso era o que escondia a repetição na tela.
        moldura = painel.justificativas[ponto.chave].parent()
        assert isinstance(moldura, QGroupBox), (
            f"o ponto `{ponto.chave}` não está numa moldura: o pai é {type(moldura)}"
        )
        assert moldura.title() == com_maiuscula(ponto.o_que_decide), (
            f"a moldura do ponto `{ponto.chave}` não diz o que ele decide: {moldura.title()!r}"
        )
        # O título sobe a primeira letra; o corpo não repete nenhuma das duas grafias.
        for titulo in (ponto.o_que_decide, com_maiuscula(ponto.o_que_decide)):
            assert titulo not in texto, (
                f"o corpo repete o título da moldura, e o profissional lê-o duas vezes: {texto!r}"
            )
        vistos += 1

    assert vistos == len(PONTOS_DE_DECISAO_AUTOMATICA), "o laço não percorreu todos os pontos"


def test_u4_todo_ponto_de_decisao_oferece_controle_de_sobrescrita(qtbot: Any) -> None:
    """A outra metade do critério: justificativa **e** controle, em cada ponto.

    ## O CONTROLE mudou de widget, e o critério não mudou

    Até aqui o controle era um `QCheckBox` por eixo, no painel da leitura, e este teste media a
    presença dele. As caixas saíram, porque o valor já se muda sem elas, e o controle passou a ser
    o **widget de valor** do painel do
    pipeline: mudar o valor É assumir a escolha.

    O critério diz *"a interface DECLARA o ponto e OFERECE o controle"*, e não que o controle
    tenha de ser uma caixa de marcar. As duas metades continuam a ser afirmadas — a declaração no
    painel da leitura, o controle no do pipeline —, e o par continua indivisível: este teste falha
    se um ponto perder qualquer uma.

    **`eixos_do_ponto` continua a levantar sobre lista vazia**, então um ponto sem controle nenhum
    na estrutura já é impossível por construção. O que este teste afirma é o lado da **tela**.
    """
    leitura = PainelDeLeitura()
    qtbot.addWidget(leitura)
    pipeline = PainelDePipeline()
    qtbot.addWidget(pipeline)

    # **AS DUAS âncoras, e a segunda faltava.** A primeira versão ancorava só
    # `CONTROLES_DE_SOBRESCRITA` e iterava `PONTOS_DE_DECISAO_AUTOMATICA` — colecção **diferente**,
    # nunca afirmada não vazia. Com os pontos vazios o laço não corria e o teste passava sem olhar
    # ponto nenhum. A revisão apanhou-o, e o teste irmão acima já o fazia certo.
    assert CONTROLES_DE_SOBRESCRITA, "a estrutura dos controles está vazia"
    assert PONTOS_DE_DECISAO_AUTOMATICA, "a estrutura dos pontos de decisão está vazia"

    widgets = pipeline._controles_de_valor()
    # **A declaração tem duas casas desde 09/11/2026**: a justificativa saiu da aba «Executar e
    # exportar» repartida por eixo, e cada parte foi para a aba do controle que explica. O ponto
    # declara na moldura dele o que o programa propõe; o que fica por conta da pessoa é dito na
    # frase «não opina» da aba de cada eixo, que nasce na repartição.
    etapa_do_eixo = pipeline.etapas_dos_eixos()
    leitura.reparte_a_justificativa(etapa_do_eixo)
    assert leitura.eixos_sem_proposta, "âncora: a repartição não escreveu frase «não opina»"
    for ponto in PONTOS_DE_DECISAO_AUTOMATICA:
        controles = leitura.controles_do_ponto(ponto.chave)
        assert controles, f"o ponto `{ponto.chave}` não tem controle de sobrescrita nenhum"
        assert ponto.chave in leitura.eixos_do_ponto, (
            f"o ponto `{ponto.chave}` não DECLARA o que decide: sem essa linha, quem lê a "
            f"justificativa não sabe que mais aquele ponto escolhe por ele"
        )
        for controle in controles:
            assert controle.eixo in widgets, (
                f"o eixo `{controle.eixo}` está na estrutura e não tem controle de valor na aba "
                f"da correção — o critério pede o controle, e ele deixou de existir"
            )
            sem_proposta = leitura.eixos_sem_proposta.get(etapa_do_eixo[controle.eixo])
            declarados = leitura.eixos_do_ponto[ponto.chave].text() + (
                "" if sem_proposta is None else " " + sem_proposta.text()
            )
            assert rotulo_do_eixo(controle.eixo) in declarados, (
                f"o eixo `{controle.eixo}` tem controle e o ponto `{ponto.chave}` não o declara: "
                f"{declarados!r}"
            )


def test_u4_a_sobrescrita_exercida_muda_o_corpo_da_requisicao(qtbot: Any) -> None:
    """A sobrescrita exercida, e a asserção é sobre o payload — **não** sobre a tela.

    **Par causal, com a âncora primeiro:** sem controle marcado o payload não carrega o eixo; com
    ele marcado, carrega. Sem a primeira metade, um painel que enviasse o eixo **sempre** passaria
    pela segunda.

    ## A asserção mudou para o canal que a rota CONSOME

    A versão anterior media `campos_de_sobrescrita()` — um dicionário `{eixo: valor}` — enviado a
    `POST /matrizes/recomendar`. **Aquela rota declara quatro campos de formulário e o FastAPI
    descarta em silêncio o que não declara**, e a própria docstring do método o dizia. O teste
    afirmava uma diferença entre dois dicionários que o servidor **jogava fora**.

    > **O canal que a rota consome é `campos_sobrescritos`**, uma lista JSON de nomes de eixo em
    > `POST /execucoes`. É sobre ele que a asserção passa a ser, e `campos_de_sobrescrita` foi
    > **removido**.

    ## E o GATILHO mudou: divergência, e não marca

    Quem alimenta `campos_sobrescritos` deixou de ser `eixos_marcados()` — a checkbox — e passou a
    ser `PainelDePipeline.eixos_divergentes()`: o painel repousa no que a máquina propôs, e o eixo
    entra na lista quando o valor corrente **difere** dela.

    **Par causal, e as três metades importam:** com o valor igual à proposta a lista é vazia; com o
    valor mudado o eixo entra. A âncora do meio é a que separa esta versão da anterior — sem o
    pré-preenchimento, o combo repousa em `none` e **toda** execução divergiria da proposta.

    A propriedade guardada não mudou: exercer o controle muda o que o servidor recebe. O que mudou é
    qual gesto conta como exercer.
    """
    pipeline = PainelDePipeline()
    qtbot.addWidget(pipeline)

    def payload(eixos: tuple[str, ...]) -> dict[str, Any]:
        return campos_da_tela(
            pipeline,
            modo="neg_c18",
            correcao="none",
            n_lotes=1,
            eixos=eixos,
            recomendacao={"correction_proposta": "loess", "rationale": "r", "origem": "modelo"},
        )

    # (a) âncora: sem proposta nenhuma, não há divergência a medir — nem com o combo em `none`.
    assert pipeline.eixos_divergentes() == (), (
        "o painel reportou divergência antes de a máquina ter proposto seja o que for"
    )

    # (b) a proposta chega e o painel REPOUSA nela: aceitar não é sobrescrever.
    assert pipeline.pre_preenche("correction", "loess"), (
        "âncora: o pré-preenchimento não mudou o combo, e as duas asserções abaixo não separariam "
        "nada — o combo já estaria em `loess` por outro motivo"
    )
    assert pipeline.corretor.currentText() == "loess"
    assert pipeline.eixos_divergentes() == (), (
        "o painel contou como sobrescrita um valor que é EXACTAMENTE o que a máquina propôs"
    )
    sem = payload(pipeline.eixos_divergentes())
    assert json.loads(sem["campos_sobrescritos"]) == []
    assert "proposta_json" not in sem, (
        "a proposta entrou sem eixo nenhum exercido, e `_sobrescrita_ou_422` recusa esse par"
    )

    # (c) o profissional muda o valor: o eixo entra, e a proposta da máquina entra com ele.
    pipeline.corretor.setCurrentText("none")
    divergentes = pipeline.eixos_divergentes()
    assert divergentes == ("correction",), f"a divergência não chegou ao acessor: {divergentes}"
    com = payload(divergentes)
    assert json.loads(com["campos_sobrescritos"]) == ["correction"]
    assert "proposta_json" in com, "sem a linha da máquina o par de concordância fica sem metade"

    assert sem != com, "os dois corpos são iguais — a sobrescrita não muda o que o servidor recebe"


def test_u4_a_abstencao_nao_vira_numero_e_a_planta_acusa(qtbot: Any) -> None:
    """A abstenção não vira número, e a planta cai sobre dado em que as duas formas DIFEREM.

    **A âncora vem primeiro, e é ela que torna o resto honesto:** a forma ingénua —
    `prosa_da_confianca(0.0, …)` — produz uma percentagem.
    Se as duas dessem o mesmo texto, tudo abaixo passaria por coincidência.

    O defeito que isto impede está medido: `unassessable` traz `confidence=0.0`
    (`diagnosis/profile.py:269` e `:305`), e imprimir *"confiança da classificação: 0%"* ao lado do
    rótulo que diz *"não houve medição"* é uma contradição.
    """
    painel = PainelDeLeitura()
    qtbot.addWidget(painel)

    corpo = {"defect_type": "unassessable", "confidence": 0.0}

    ingenua = prosa_da_confianca(corpo["confidence"], do_que="a classificação")
    assert "%" in ingenua, (
        "ÂNCORA: a forma ingénua deixou de imprimir percentagem, então esta planta não separa nada"
    )

    texto = painel.mostra("diagnostico", corpo)

    assert "%" not in texto, f"a abstenção virou número: {texto!r}"
    assert ingenua not in texto, "o painel usou a forma ingénua"
    # A frase abre uma sentença na tela, então chega com a primeira letra em maiúscula.
    assert com_maiuscula(SEM_MEDICAO_DA_CLASSIFICACAO) in texto, (
        f"o painel não diz que não houve medição: {texto!r}"
    )
    # **O identificador CRU não aparece, e a inversão é o conserto de um achado da revisão.** A
    # primeira versão deste teste exigia `"unassessable" in texto`; o painel passou a traduzir o
    # rótulo por `ROTULOS_DE_DEFEITO`, e `gradual_drift`/`erratic` são nomes de código que quem
    # opera o equipamento não tem por que conhecer. O teste apanhou a própria melhoria.
    assert "unassessable" not in texto, f"o painel imprimiu o identificador cru: {texto!r}"
    assert com_maiuscula(ROTULOS_DE_DEFEITO["unassessable"]) in texto, (
        "o painel não traduziu o rótulo"
    )


def test_u4_um_diagnostico_medido_continua_a_mostrar_a_percentagem(qtbot: Any) -> None:
    """A outra direção, e sem ela o painel poderia esconder a confiança SEMPRE.

    Um painel que nunca imprimisse percentagem passaria pelo teste da abstenção e perderia a
    informação em todos os outros casos.
    """
    painel = PainelDeLeitura()
    qtbot.addWidget(painel)

    texto = painel.mostra("diagnostico", {"defect_type": "gradual_drift", "confidence": 0.87})

    assert "87%" in texto, f"a confiança medida não apareceu: {texto!r}"
    assert SEM_MEDICAO_DA_CLASSIFICACAO not in texto


def test_u4_a_recomendacao_tem_os_tres_estados_e_so_um_traz_numero(qtbot: Any) -> None:
    """`prosa_da_recomendacao` atravessando o painel, nos três estados.

    A asserção é que os três textos são **dois a dois distintos**: dois estados com a mesma frase
    seriam um colapso, e a contagem por causa passaria a contar coisa errada.
    """
    painel = PainelDeLeitura()
    qtbot.addWidget(painel)

    textos = {
        "nao_avaliavel": painel.mostra(
            "recomendacao",
            {"correction_proposta": "none", "avaliavel": False, "abstem": True, "confianca": 0.0},
        ),
        "abstem": painel.mostra(
            "recomendacao",
            {"correction_proposta": "none", "avaliavel": True, "abstem": True, "confianca": 0.0},
        ),
        "eleito": painel.mostra(
            "recomendacao",
            {"correction_proposta": "loess", "avaliavel": True, "abstem": False, "confianca": 0.87},
        ),
    }

    assert len(set(textos.values())) == 3, f"os três estados não são distintos: {textos}"
    assert "%" not in textos["nao_avaliavel"], "não avaliável virou número"
    assert "%" not in textos["abstem"], "abstenção virou número"
    assert "87%" in textos["eleito"], "o estado medido perdeu a percentagem"


@pytest.mark.parametrize("modo", MODOS_SUGERIDOS)
def test_u4_os_quatro_modos_reais_atravessam_a_leitura(
    modo: str, transporte_real: TestClient, qtbot: Any
) -> None:
    """Os quatro modos, pelos resolvedores do `conftest`.

    **Pelos resolvedores e não por caminho montado à mão** — eles resolvem nome lógico para caminho
    real e fazem `skip` com mensagem quando o arquivo falta. E os quatro
    porque um modo só não distingue *"a leitura funciona"* de *"a leitura funciona no `neg_c18`"*.
    """
    painel = PainelDeLeitura()
    qtbot.addWidget(painel)

    arquivos = {
        "measurements": ("m.csv", measurements_path(modo).read_bytes(), "text/csv"),
        "sample_list": ("s.csv", sample_list_path(modo).read_bytes(), "text/csv"),
    }
    for chave, rota in ROTAS_DA_LEITURA:
        resposta = _le_a_rota_ate_o_fim(
            transporte_real, rota, arquivos=arquivos, campos={"mode": modo}
        )
        assert resposta.codigo == 200, f"{modo} {rota}: {resposta.codigo} {resposta.corpo}"
        texto = painel.mostra(chave, resposta.corpo)
        assert texto != SEM_LEITURA, f"{modo} {chave} não produziu texto"


# ---------------------------------------------------------------------------
# As guardas dos formatadores — a cobertura as achou
# ---------------------------------------------------------------------------
#
# **Nenhuma destas é caminho feliz, e é por isso que elas estão aqui.** Caminho
# de erro que dá trabalho testar não justifica `pragma` — é justamente onde o teste importa. Aqui
# não dá trabalho nenhum: a planta é **argumento**, então o negativo percorre o mesmo caminho que
# o portão.


@pytest.mark.parametrize(
    "formatador",
    [texto_dos_problemas, texto_do_diagnostico, texto_da_recomendacao],
    ids=["problemas", "diagnostico", "recomendacao"],
)
@pytest.mark.parametrize("corpo", [None, "uma string", 42, []], ids=["none", "str", "int", "lista"])
def test_u4_os_formatadores_toleram_corpo_que_nao_e_dicionario(formatador: Any, corpo: Any) -> None:
    """`cliente.envia` devolve `corpo=None` quando o `content-type` não é JSON — e isso acontece.

    Medido no próprio cliente: a rota de exportação devolve `text/csv`, e ali `corpo` fica `None`
    com os bytes em `conteudo`. Um formatador que fizesse `corpo.get(...)` levantaria
    `AttributeError` **dentro do painel**, e o profissional veria um traceback em vez de uma frase.
    """
    assert formatador(corpo) == SEM_LEITURA


def test_u4_um_corpo_sem_aceita_cai_no_detail_das_outras_rotas() -> None:
    """O ramo do `detail`, e ele existe porque este painel exibe **qualquer** corpo.

    `/matrizes/validar` sempre traz `aceita`. As outras rotas, num `422`, trazem `detail` — string
    quando é guarda nossa, lista de dicionários quando é a validação do FastAPI. É o discriminador
    mecânico de `cliente._forma_do_422`, e as duas formas chegam aqui.
    """
    assert texto_dos_problemas({"detail": "recusado pela validação"}) == "recusado pela validação"
    assert texto_dos_problemas({"detail": [{"msg": "faltou mode"}, {"msg": "e o modo"}]}) == (
        "faltou mode; e o modo"
    )
    # E o terceiro desfecho: um dicionário que não tem nem `aceita` nem `detail`.
    assert texto_dos_problemas({"outra": "coisa"}) == SEM_LEITURA


def test_u4_a_validacao_aceita_relata_contagens_e_a_recusada_relata_codigos() -> None:
    """Os dois ramos de `aceita`, com planta por argumento e âncora de divergência.

    **A âncora é a última asserção:** se os dois ramos dessem o mesmo texto, este teste passaria por
    coincidência e não mediria a distinção que o painel existe para fazer.
    """
    aceita = texto_dos_problemas({"aceita": True, "n_features": 2184, "n_samples": 68})
    recusada = texto_dos_problemas(
        {
            "aceita": False,
            "problemas": [
                {"codigo": "SAMPLE_LIST_MISMATCH", "severidade": "error", "mensagem": "trocada"}
            ],
        }
    )

    assert "2184" in aceita and "68" in aceita
    assert "nenhum problema" in aceita
    assert "SAMPLE_LIST_MISMATCH" in recusada
    assert "error" in recusada
    assert aceita != recusada, "os dois vereditos produzem o mesmo texto"


def test_u4_o_modo_escolhido_vai_no_campo_mode(painel_de_arquivos: Any) -> None:
    """O caminho feliz de `campos_para_envio`, que só o levantamento estava a exercitar.

    Um teste que só vê a exceção deixa o retorno sem cobertura — e é o retorno que a rota consome.
    """
    painel, _ = painel_de_arquivos
    painel.modo.setEditText("neg_c18")
    painel.escolhe_medidas()
    painel.escolhe_lista()

    assert painel.campos_para_envio() == {"mode": "neg_c18"}


def test_u4_confianca_ausente_nao_e_o_mesmo_que_confianca_zero() -> None:
    """`prosa_da_confianca(None, …)` — o ramo que separa *"não informada"* de *"zero por cento"*.

    Colapsar os dois apagaria a diferença: *"o campo não veio"* e *"o campo veio e vale zero"*
    levam a ações diferentes.
    """
    ausente = prosa_da_confianca(None, do_que="a classificação")
    zero = prosa_da_confianca(0.0, do_que="a classificação")

    assert "não informada" in ausente
    assert "%" not in ausente
    assert "0%" in zero
    assert ausente != zero, "os dois estados produzem a mesma frase"


# ---------------------------------------------------------------------------
# As lacunas que a revisão de código nomeou
# ---------------------------------------------------------------------------


def test_u4_matriz_aceita_com_aviso_nao_e_exibida_como_recusada(
    transporte_real: TestClient, qtbot: Any
) -> None:
    """A célula do 2x2 que faltava, e é onde vivia a inversão do veredito.

    A validação tem quatro estados — aceita/recusada, com/sem problemas — e a primeira versão dos
    testes cobria só a **diagonal**. A célula `(aceita=True, problemas≠[])` é real: um
    `MODE_CONTENT_MISMATCH` é `severity="warning"` (`io/validators.py:732`) e a rota devolve `200`
    com `aceita=True` (`api/rotas.py:287`).

    **A planta é a que o laboratório de facto comete:** os arquivos do `neg_c18` declarados como
    `neg_hilic`. A primeira versão do painel ramificava na vacuidade de `problemas` e escrevia
    *"recusada"* sobre uma matriz aceita, descartando as contagens.
    """
    painel = PainelDeLeitura()
    qtbot.addWidget(painel)

    arquivos = {
        "measurements": ("m.csv", measurements_path("neg_c18").read_bytes(), "text/csv"),
        "sample_list": ("s.csv", sample_list_path("neg_c18").read_bytes(), "text/csv"),
    }
    resposta = cliente.envia(
        transporte_real,
        "POST",
        "/matrizes/validar",
        arquivos=arquivos,
        campos={"mode": "neg_hilic"},
    )

    assert resposta.codigo == 200, f"{resposta.codigo}: {resposta.corpo}"
    assert resposta.corpo["aceita"] is True, (
        f"a planta não produziu o estado que este teste mede — aceita com aviso: {resposta.corpo}"
    )
    assert resposta.corpo["problemas"], "a planta não gerou aviso nenhum: o teste seria vácuo"

    texto = painel.mostra("validacao", resposta.corpo)

    assert "recusada" not in texto, f"matriz ACEITA exibida como recusada: {texto!r}"
    assert texto.startswith("aceita"), f"o veredito não é o da resposta: {texto!r}"
    assert str(resposta.corpo["n_features"]) in texto, "as contagens foram descartadas"


def test_u4_o_modo_corrigido_depois_de_escolher_chega_ao_payload(painel_de_arquivos: Any) -> None:
    """A ordem inversa, e é a única em que o modo congelado aparecia.

    Os dois testes de modo da primeira versão punham o combo **antes** de escolher — a ordem
    em que
    congelar está correto. A revisão construiu a sequência oposta: escolher, notar o modo errado,
    corrigir o combo. Com o modo no par, todas as chamadas seguintes enviavam o antigo.
    """
    painel, _ = painel_de_arquivos

    painel.modo.setEditText("pos_hilic")
    painel.escolhe_medidas()
    painel.escolhe_lista()
    assert painel.campos_para_envio() == {"mode": "pos_hilic"}, "âncora: o primeiro modo não chegou"

    painel.modo.setEditText("neg_c18")

    assert painel.campos_para_envio() == {"mode": "neg_c18"}, (
        "o modo corrigido DEPOIS da escolha não chegou ao payload — está congelado no par"
    )
    assert "neg_c18" in painel.rotulo.text(), (
        f"o rótulo mostra um modo e o payload envia outro: {painel.rotulo.text()!r}"
    )


def test_u4_a_justificativa_do_diagnostico_nao_e_um_repr_de_dicionario(
    transporte_real: TestClient, qtbot: Any
) -> None:
    """O ponto `diagnostico` tem `campo_da_justificativa="lotes"`, e `lotes` é OBJETO.

    A primeira versão de `mostra_justificativa` fazia `str(bruto)` — exactamente o
    *"`else` razoável
    que se comporta como o defeito original"* que a docstring de `justificativa_em_prosa` nomeia. No
    `pos_hilic` isso são 68 números de `rotulos_por_amostra` despejados na tela.

    **E o teste irmão não conseguia acusar:** `ponto.o_que_decide in texto` e `texto != SEM_LEITURA`
    são satisfeitos por **qualquer** repr não vazio. Estas asserções são sobre os marcadores de
    estrutura, que é o que separa prosa de despejo.
    """
    painel = PainelDeLeitura()
    qtbot.addWidget(painel)

    arquivos = {
        "measurements": ("m.csv", measurements_path("neg_c18").read_bytes(), "text/csv"),
        "sample_list": ("s.csv", sample_list_path("neg_c18").read_bytes(), "text/csv"),
    }
    resposta = cliente.envia(
        transporte_real,
        "POST",
        "/matrizes/diagnosticar",
        arquivos=arquivos,
        campos={"mode": "neg_c18"},
    )
    assert resposta.codigo == 200, f"{resposta.codigo}: {resposta.corpo}"

    ponto = next(p for p in PONTOS_DE_DECISAO_AUTOMATICA if p.chave == "diagnostico")
    bruto = resposta.corpo[ponto.campo_da_justificativa]
    assert isinstance(bruto, dict), (
        f"a planta não separa: o campo `{ponto.campo_da_justificativa}` não é objeto "
        f"({type(bruto)})"
    )

    texto = painel.mostra_justificativa(ponto, resposta.corpo)

    assert "rotulos_por_amostra" not in texto, f"o painel despejou o objeto: {texto!r}"
    assert "{" not in texto and "'" not in texto, (
        f"o texto tem marcadores de estrutura de dicionário: {texto!r}"
    )
    assert "Lotes sugeridos" in texto, f"a prosa canónica não foi usada: {texto!r}"


def test_u4_o_sinal_par_escolhido_dispara_uma_vez_por_escolha(
    painel_de_arquivos: Any, qtbot: Any
) -> None:
    """O sinal que a docstring chama de *"o que os painéis seguintes esperam"*, afirmado.

    Sem este teste o contrato entre painéis existia só na prosa: nada provava que o sinal dispara,
    nem que carrega o par que `escolhe` devolveu. E cancelar **não** emite, então o sinal distingue
    os dois desfechos que o valor de retorno não distingue.
    """
    painel, _ = painel_de_arquivos
    vistos: list[object] = []
    painel.par_escolhido.connect(vistos.append)

    assert vistos == [], "âncora: o painel emitiu antes de lhe pedirem"

    painel.escolhe_medidas()
    assert vistos == [], (
        "o sinal disparou com METADE do par escolhido: quem o recebe compõe um payload a que falta "
        "um arquivo, e a rota responde `422` a culpar o profissional por um passo que ele ainda "
        "estava a dar"
    )

    par = painel.escolhe_lista()

    assert len(vistos) == 1, f"o sinal não disparou exactamente uma vez: {len(vistos)}"
    assert vistos[0] is par, "o sinal carrega objeto diferente do que a escolha devolveu"


def test_u4_o_payload_carrega_os_nomes_e_o_tipo_e_nao_so_os_bytes(
    painel_de_arquivos: Any,
) -> None:
    """A forma inteira do multipart, e não só a identidade dos bytes.

    A primeira versão afirmava `payloads[0]["measurements"][1] is primeiro` — o índice dos
    bytes — e nunca leu o nome nem o tipo. Uma troca de chave, `nome_da_lista` sob
    `"measurements"`, passaria.
    """
    painel, _ = painel_de_arquivos
    painel.escolhe_medidas()
    painel.escolhe_lista()

    envio = painel.arquivos_para_envio()
    par = painel.par
    assert par is not None, "âncora: não há par para comparar"

    assert envio["measurements"] == (
        par.nome_das_medidas,
        par.bytes_das_medidas,
        TIPO_DO_ARQUIVO,
    )
    assert envio["sample_list"] == (par.nome_da_lista, par.bytes_da_lista, TIPO_DO_ARQUIVO)
    assert par.nome_das_medidas != par.nome_da_lista, (
        "os dois nomes são iguais no dado de teste: uma troca de chave não seria detectável"
    )


def test_u4_uma_leitura_ilegivel_relata_e_preserva_o_par_anterior(qtbot: Any) -> None:
    """`read_bytes` dentro de um slot de `clicked`, com o arquivo a desaparecer entre a escolha e a
    leitura. A excepção não pode escapar: o desfecho seria silêncio.

    **Par causal:** primeiro uma escolha boa, depois uma que falha. O par bom tem de sobreviver.
    """
    sequencia = [
        measurements_path("neg_c18"),
        sample_list_path("neg_c18"),
        Path("/nao/existe/m.csv"),
    ]

    painel = PainelDeArquivos(lambda _titulo: sequencia.pop(0))
    qtbot.addWidget(painel)

    painel.escolhe_medidas()
    primeiro = painel.escolhe_lista()
    assert primeiro is not None, "âncora: a primeira escolha falhou, e o teste mediria outra coisa"

    segundo = painel.escolhe_medidas()

    assert segundo is primeiro, "a leitura falhada descartou o par que estava bom"
    assert "Não foi possível ler" in painel.rotulo.text(), (
        f"a falha de leitura não foi relatada: {painel.rotulo.text()!r}"
    )


def test_u4_a_fabrica_do_arquivo_traduz_o_cancelamento_e_leva_o_titulo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As DUAS direções de `escolhedor_por_dialogo`, e o título que diz QUAL arquivo se pede.

    ## Esta propriedade estava AFIRMADA e sem dono, e a afirmação era a da própria docstring

    `escolhedor_por_dialogo` declara — *"o arnês remenda `QFileDialog.getOpenFileName` e afirma o
    caminho devolvido. As duas direções ficam exercitadas"* —, e até 06/09/2026 **nenhum teste desta
    árvore o cumpria**: `grep` por `getOpenFileNames` sobre `tests/` devolvia zero. É a mesma
    ausência que `test_u6_a_fabrica_da_pasta_traduz_o_cancelamento_e_entrega_o_caminho` fechou para
    a fábrica irmã, e o molde é o dela.

    | O que o `QFileDialog` devolve | O que a fábrica faz |
    |---|---|
    | `("C:/…/m.csv", "CSV (*.csv)")` | devolve o `Path` dele |
    | `("", "")` — o cancelamento do Qt | devolve `None` |

    > **String vazia traduzida para `None` é o contrato do `Escolhedor`.** Deixá-la passar faria
    > `Path("")` — o diretório corrente —, e o painel tentaria `read_bytes()` sobre um diretório: um
    > `OSError` a dizer «não foi possível ler» sobre um arquivo que ninguém escolheu.

    **E a terceira asserção é a que os dois botões acrescentam:** o título viaja para o diálogo. Uma
    fábrica que o descartasse abriria as duas janelas com o mesmo cabeçalho, e a pessoa deixaria de
    saber qual dos dois arquivos está a escolher — que é metade do que a separação dos botões
    comprou.
    """
    from PySide6.QtWidgets import QFileDialog

    from refino.desktop.janela.arquivos import escolhedor_por_dialogo

    alvo = tmp_path / "measurements_neg_c18.csv"
    alvo.write_text("feature,QC1\n1,2\n", encoding="utf-8")
    vistos: list[tuple[Any, ...]] = []

    def dialogo_que_aceita(*args: Any, **kwargs: Any) -> tuple[str, str]:
        vistos.append(args)
        return str(alvo), "CSV (*.csv)"

    monkeypatch.setattr(QFileDialog, "getOpenFileName", dialogo_que_aceita)
    escolhido = escolhedor_por_dialogo()(ROTULO_DAS_MEDIDAS)

    assert escolhido == alvo, f"a fábrica não devolveu o caminho escolhido: {escolhido}"
    assert vistos, "âncora: o diálogo não foi chamado"
    assert ROTULO_DAS_MEDIDAS in vistos[0], (
        f"o título não chegou ao diálogo: {vistos[0]}. Sem ele as duas janelas abrem com o mesmo "
        f"cabeçalho, e quem escolhe não sabe qual dos dois arquivos está a dar"
    )

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    cancelado = escolhedor_por_dialogo()(ROTULO_DA_LISTA)

    assert cancelado is None, (
        f"o cancelamento não virou `None`: {cancelado!r}. `Path('')` é o diretório corrente, e o "
        f"painel tentaria lê-lo como se fosse o CSV"
    )
    assert cancelado != escolhido, (
        "âncora: as duas direções devolveram a mesma coisa, e o teste não separa nada"
    )


def test_u4_uma_metade_escolhida_nao_monta_par_e_o_rotulo_diz_o_que_falta(qtbot: Any) -> None:
    """O estado que os DOIS botões criam, e que com um botão só não existia.

    Com um seletor único havia dois estados — com par e sem par. Com dois botões há um terceiro:
    **uma metade escolhida**. Ele dispara por ausência de clique, então ficaria inerte sem teste
    próprio: nada o produz de propósito, e a suíte inteira passa sem nunca o alcançar.

    **As três asserções são as três consequências dele**, e são independentes:

    | O que se afirma | O que quebraria sem isso |
    |---|---|
    | `par is None` | `arquivos_para_envio` montaria `sample_list` vazio, e a rota culparia o CSV |
    | o payload **levanta**, dizendo o que falta | `422` a mandar conferir arquivo que está certo |
    | o rótulo mostra o nome que JÁ se tem | quem escolheu metade reescolheria a metade que tinha |

    **Par causal na última:** o rótulo do estado intermédio tem de diferir de `SEM_PAR`, senão
    *"não escolhi nada"* e *"escolhi um dos dois"* voltam a ser a mesma tela.
    """
    medidas = measurements_path("neg_c18")
    painel = PainelDeArquivos(lambda titulo: medidas if titulo == ROTULO_DAS_MEDIDAS else None)
    qtbot.addWidget(painel)

    assert painel.rotulo.text() == SEM_PAR, "âncora: o painel não repousa no texto de vazio"

    assert painel.escolhe_medidas() is None, "o painel montou um par com uma metade só"
    assert painel.par is None

    for metodo in (painel.arquivos_para_envio, painel.campos_para_envio):
        with pytest.raises(ValueError, match="falta Sample_List"):
            metodo()

    texto = painel.rotulo.text()
    assert texto != SEM_PAR, (
        "o rótulo do estado intermédio é igual ao do vazio: quem escolheu uma metade não tem como "
        "saber que ela ficou guardada"
    )
    assert medidas.name in texto, f"o rótulo não mostra a metade que já foi escolhida: {texto!r}"
    assert "Sample_List" in texto, f"o rótulo não diz qual metade falta: {texto!r}"


# ---------------------------------------------------------------------------
# O pipeline, o progresso, e o par de sobrescrita
# ---------------------------------------------------------------------------


def _painel_de_pipeline(qtbot: Any, instantes: list[float]) -> PainelDePipeline:
    """Um painel cujo relógio percorre `instantes`, um por leitura.

    **O relógio é substituído e não medido**: o arnês mede 0,01 s por
    interação, então duas leituras reais dariam o mesmo número de segundos e *"a segunda é maior"*
    passaria ou reprovaria conforme o escalonamento — a forma intermitente.
    """
    restantes = list(instantes)
    painel = PainelDePipeline(relogio=lambda: restantes.pop(0))
    qtbot.addWidget(painel)
    # A caixa «Automático» era DESTICADA aqui: ticada, o payload
    # levava `auto` e a política escolhia o corretor — o `correcao` que o teste passa deixava de ser
    # o que corre, e `test_u5_os_estados_alcancaveis_do_par_chegam_ao_banco` não terminava em 600
    # reclamações. A caixa saiu, e o payload leva sempre o corretor que o teste passa.
    return painel


def test_u5_o_painel_de_pipeline_esta_composto_e_nao_e_mais_uma_marca(
    janela: JanelaPrincipal,
) -> None:
    """O painel 4 deixou de ser espaço vazio, e os outros cinco continuam a ser.

    **A âncora é a segunda asserção**, e sem ela este teste passaria numa janela que trocasse
    *todas* as marcas por widgets vazios: o que se afirma é **um** painel, e a contagem dos que
    faltam é o que declara o escopo restante.
    """
    painel = janela.pipeline
    assert isinstance(painel, PainelDePipeline), f"não é o painel: {type(painel).__name__}"
    # **Desde 09/11/2026 o painel não tem aba própria**: a aba «Executar e exportar», que o
    # desenhava, saiu, e cada grupo dele mora na aba da etapa a que pertence. O
    # que se afirma é a mesma coisa — os controles estão na tela —, um grupo por vez, e a aba de
    # cada grupo é a da etapa dele.
    from refino.desktop.janela.principal import INDICES_DAS_ABAS_DE_ETAPA

    assert painel.grupos_das_etapas, "âncora: o painel não declara grupo nenhum"
    for etapa, grupos in painel.grupos_das_etapas.items():
        for grupo in grupos:
            indice = _aba_que_mostra(janela, grupo)
            abas = [janela.abas.tabText(i) for i in range(janela.abas.count())]
            assert indice == INDICES_DAS_ABAS_DE_ETAPA[etapa], (
                f"o grupo «{grupo.objectName() or type(grupo).__name__}» da etapa {etapa} está na "
                f"aba {indice}. Abas: {abas}"
            )
    assert _aba_que_mostra(janela, painel) == -1, (
        "o painel do pipeline voltou a ser desenhado inteiro numa aba, ao lado dos grupos dele"
    )
    # **DERIVADO de `_preenchidos` e não `len(PAINEIS) - 1`.** O numeral à mão reprovou no instante
    # em que o segundo painel foi composto — e reprovar era o desfecho certo, porque a partição
    # mudou; o que estava errado era a asserção afirmar a partição de ontem. Derivar faz a soma
    # continuar a valer e mantém o que ela protege: que os dois conjuntos sejam **disjuntos** e
    # cubram os seis.
    assert len(janela._vazios) == len(PAINEIS) - len(janela._preenchidos), (
        f"{len(janela._vazios)} marcas mais {len(janela._preenchidos)} preenchidos não somam os "
        f"{len(PAINEIS)} painéis: {sorted(janela._vazios)} / {sorted(janela._preenchidos)}"
    )
    assert not set(janela._vazios) & set(janela._preenchidos), (
        "um painel está nas duas partições ao mesmo tempo"
    )
    assert set(janela._vazios) | set(janela._preenchidos) == {p.chave for p in PAINEIS}
    assert "pipeline" not in janela._vazios


# ---------------------------------------------------------------------------
# A TERMINALIDADE vem do código HTTP — o defeito que a revisão de correção mediu
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_u5_a_rota_nunca_emite_terminou_verdadeiro_e_e_por_isso_que_o_codigo_decide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qtbot: Any
) -> None:
    """**O achado que derrubou a primeira versão do painel, virado em portão.**

    A primeira versão de `mostra_progresso` derivava a terminalidade de `corpo.get("terminou")`.
    Medido numa corrida real contra SQLite, `neg_c18`, `correction=none`:

    | Leituras | `terminou` |
    |---|---|
    | as em curso (HTTP 202) | `False` |
    | a terminal (HTTP 200) | **o campo não existe** |
    | qualquer uma | `True` **zero** vezes |

    A causa é da rota e é deliberada: ela serve `ProgressoResposta` **só** `if not
    leitura.terminou`, e o desfecho terminal é `200` com `ExecucaoResposta` — outro esquema, sem o
    campo. Então o congelamento do decorrido era **inalcançável em produção**, e os testes que o
    exercitavam alimentavam dicionários feitos à mão com `terminou=True` que a rota não sabe
    produzir.

    > *"A minha planta cai sobre dado em que as duas coisas que ela separa diferem?"* — e aqui
    > ela caía sobre dado que a fonte **não gera**.

    Este teste é o portão contra a reintrodução: se um dia a rota passar a emitir `terminou=True`,
    ele reprova e alguém revisita o roteamento por código.
    """
    from sqlalchemy import create_engine

    from refino.persistence.config import VARIAVEL_DA_URL, prepara_sqlite
    from refino.persistence.schema import METADATA

    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    prepara_sqlite(url)
    METADATA.create_all(create_engine(url))

    cliente_real = TestClient(build_app(), raise_server_exceptions=False)
    modo = "neg_c18"
    medidas, lista = measurements_path(modo), sample_list_path(modo)

    # **O payload sai do CONSTRUTOR DE PRODUÇÃO.**
    #
    # Ele montava o `data` à mão com `**FIXOS_DECLARADOS`, tratando aquela constante como *"tudo o
    # que o painel fixa"*. Os quatro eixos da grade saíram dela — passaram a ser combos —, e a
    # rota devolveu **422**: quatro campos obrigatórios em falta.
    #
    # > **A reprovação é o desfecho certo, e a docstring deste teste já dizia por quê:** ele existe
    # > porque *"os testes que o exercitavam alimentavam dicionários feitos à mão que a rota não
    # > sabe produzir"*. O conserto não é acrescentar os quatro à mão — é parar de montar à mão.
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    # A caixa «Automático» era DESTICADA aqui: ticada, o payload levava
    # `auto` e a política escolhia um corretor que custa minutos — o teste não terminava em 600
    # leituras. A caixa saiu, e o payload leva sempre o `correcao="none"` que o teste passa.
    disparo = cliente_real.post(
        "/execucoes",
        files={
            "measurements": (medidas.name, medidas.read_bytes(), "text/csv"),
            "sample_list": (lista.name, lista.read_bytes(), "text/csv"),
        },
        data=campos_da_tela(painel, modo=modo, correcao="none", n_lotes=1),
    )
    assert disparo.status_code == 202, f"âncora: o disparo deu {disparo.status_code}"
    identificador = disparo.json()["identificador"]

    leituras: list[tuple[int, object]] = []
    for _ in range(600):
        resposta = cliente_real.get(rota_do_progresso(identificador))
        corpo = resposta.json()
        leituras.append((resposta.status_code, corpo.get("terminou", "(ausente)")))
        if resposta.status_code != EM_CURSO:
            break

    assert len(leituras) >= 2, (
        f"âncora: só {len(leituras)} leitura(s) — sem uma em curso E uma terminal, este teste não "
        f"mede a distinção que existe para medir"
    )
    assert leituras[-1][0] == 200, f"a leitura terminal deu {leituras[-1][0]}"
    assert leituras[-1][1] == "(ausente)", (
        f"o corpo TERMINAL passou a trazer `terminou`: {leituras[-1][1]!r}. O painel roteia pelo "
        f"código HTTP por causa desta ausência — revisite `recebe_leitura`"
    )
    assert not any(valor is True for _, valor in leituras), (
        f"a rota passou a emitir `terminou=True`: {leituras}. A premissa do roteamento por código "
        f"mudou, e o painel tem de ser revisitado"
    )


def test_u5_recebe_leitura_roteia_os_quatro_desfechos_pelo_codigo(qtbot: Any) -> None:
    """Os **quatro** desfechos de `GET /trabalhos/{id}`, e nenhum sai igual a outro.

    A rota tem quatro formas de responder — em curso, concluído, identificador desconhecido, e
    falhou —, e *"os dois desfechos"* não é resposta para um mecanismo que classifica em quatro.

    **A âncora é a distinção mútua**: quatro textos, quatro valores distintos. Um roteador que
    colapsasse dois deles passaria em qualquer asserção individual.
    """
    painel = _painel_de_pipeline(qtbot, [130.0, 130.0, 130.0, 130.0])
    painel.aceita_disparo({"identificador": "t"}, instante=100.0)

    em_curso = painel.recebe_leitura(
        EM_CURSO, {"estado": "executando", "feito": 2, "total": 4, "terminou": False}
    )
    concluido = painel.recebe_leitura(200, {"run_id": 5, "estado_concordancia": "nao_sobrescreveu"})
    desconhecido = painel.recebe_leitura(404, {"detail": "não há trabalho com identificador 'x'"})
    falhou = painel.recebe_leitura(409, {"detail": "o trabalho falhou na etapa «pipeline»: vazia"})

    textos = {
        "em curso": em_curso,
        "concluído": concluido,
        "desconhecido": desconhecido,
        "falhou": falhou,
    }
    assert len(set(textos.values())) == 4, f"dois desfechos colapsaram: {textos}"

    assert "Executando: 2/4 etapas" in em_curso
    assert "desde o disparo" in em_curso, f"o decorrido de uma execução VIVA congelou: {em_curso}"

    assert "Concluído" in concluido
    assert "até a leitura do fim" in concluido, (
        f"o decorrido de um desfecho CONCLUÍDO não congelou: {concluido}"
    )

    assert "Identificador desconhecido" in desconhecido
    assert "404" in desconhecido
    assert "falhou" not in desconhecido, (
        f"um identificador desconhecido foi apresentado como falha do trabalho — as ações são "
        f"opostas, redisparar contra corrigir o dado: {desconhecido}"
    )

    assert "O trabalho falhou" in falhou
    assert "«pipeline»" in falhou, (
        f"a etapa não foi nomeada — é o que o `[TO-12.6]` cobra: {falhou}"
    )


def test_u5_um_200_leva_o_corpo_ao_rotulo_de_resultado_e_a_barra_a_cheio(qtbot: Any) -> None:
    """O `200` é o resultado, e ele vai aos DOIS rótulos.

    **A âncora é o estado antes:** o rótulo de resultado começa na string de ausência, e a barra
    numa leitura parcial. Sem isso, *"o resultado apareceu"* seria indistinguível de *"já estava
    lá"*, e *"a barra encheu"* de *"a barra já estava cheia"*.
    """
    painel = _painel_de_pipeline(qtbot, [130.0, 200.0])
    painel.aceita_disparo({"identificador": "t"}, instante=100.0)
    painel.recebe_leitura(EM_CURSO, {"estado": "executando", "feito": 3, "total": 4})

    # **`RESULTADO_EM_REPOUSO` e não `SEM_EXECUCAO`, desde 07/09/2026.** Os dois rótulos passaram a
    # dizer DE QUE são também depois de um disparo: a tela mostrava `pendente: 0/4 etapas` numa
    # linha e `nenhuma execução disparada` na seguinte, e quem olhou leu uma contradição.
    assert painel.rotulo_de_resultado.text() == RESULTADO_EM_REPOUSO, (
        "âncora: já havia resultado na tela"
    )
    assert (
        painel.barras_de_etapa[ETAPAS[-1]].value() != painel.barras_de_etapa[ETAPAS[-1]].maximum()
    ), "âncora: a barra já estava cheia"

    painel.recebe_leitura(
        200,
        {
            "run_id": 9,
            "estado_concordancia": "sobrescreveu_discordando",
            "qc_rsd_median": 11.0,
            "n_features_saida": 300,
            "rsd_antes_e_depois": {"mediana_antes": 20.0, "mediana_depois": 11.0},
        },
    )

    resultado = painel.rotulo_de_resultado.text()
    assert "execução 9" in resultado
    assert "sobrescreveu_discordando" in resultado
    assert (
        painel.barras_de_etapa[ETAPAS[-1]].value() == painel.barras_de_etapa[ETAPAS[-1]].maximum()
    ), "a barra ficou em 3/4 ao lado de «concluído» — duas superfícies a discordar"


def test_u5_o_progresso_e_monotonico_e_alcanca_o_total(qtbot: Any) -> None:
    """A barra sobe e chega ao total, e a leitura do meio é **intermediária**.

    **A asserção do meio é a que separa isto de um teste de dois pontos.** Uma implementação que
    escrevesse `0` até terminar e `total` no fim passaria numa asserção de extremos, e o que o
    profissional lê durante uma execução de minutos é justamente o meio.
    """
    painel = _painel_de_pipeline(qtbot, [110.0, 120.0, 130.0])
    painel.aceita_disparo({"identificador": "t1"}, instante=100.0)

    vistos: list[int] = []
    for feito in (0, 2, 4):
        painel.recebe_leitura(EM_CURSO, {"estado": "executando", "feito": feito, "total": 4})
        vistos.append(painel.barras_de_etapa[ETAPAS[-1]].value())

    assert vistos == sorted(vistos), f"o progresso não foi monotônico: {vistos}"
    assert vistos[1] not in (vistos[0], vistos[-1]), (
        f"a leitura do meio não é intermediária — os extremos passariam sozinhos: {vistos}"
    )
    assert (
        painel.barras_de_etapa[ETAPAS[-1]].value()
        == painel.barras_de_etapa[ETAPAS[-1]].maximum()
        == 4
    )


def test_u5_falha_no_meio_produz_estado_terminal_nomeado(qtbot: Any) -> None:
    """Falhar não é ficar preso, e o texto **nomeia** onde falhou.

    *"O trabalho falhou"* sem dizer onde obriga a refazer tudo. A âncora é o par: uma leitura
    saudável **não** traz a palavra.
    """
    painel = _painel_de_pipeline(qtbot, [130.0, 130.0])

    painel.aceita_disparo({"identificador": "t2"}, instante=100.0)
    saudavel = painel.recebe_leitura(EM_CURSO, {"estado": "executando", "feito": 1, "total": 4})
    assert "falhou" not in saudavel, f"âncora: a leitura saudável já dizia «falhou»: {saudavel}"

    quebrado = painel.recebe_leitura(
        500, {"detail": "o trabalho falhou na etapa «relatorio de qualidade»: EmptyMatrixError"}
    )
    assert "«relatorio de qualidade»" in quebrado, quebrado
    assert "EmptyMatrixError" in quebrado, quebrado


def test_u5_o_par_causal_de_d298_portado_a_segunda_corrida_nao_exibe_o_decorrido_da_primeira(
    qtbot: Any,
) -> None:
    """**O defeito do decorrido congelado, portado para a janela.**

    Duas corridas na mesma janela: a primeira de 30 s, a segunda de 500 s. Sem o funil a limpar o
    congelado, a segunda exibiria **30 s** — parado, plausível, e pertencente a outra execução.

    ## As agulhas discriminam, e é obrigatório que discriminem

    `"30 s"` é sufixo de `"130 s"` e de `"530 s"`, então uma implementação que exibisse a **leitura
    do relógio** em vez do intervalo passaria numa agulha ingénua. Os instantes são escolhidos para
    que os dois números não se contenham: 30 e 500.

    ## O RÓTULO do congelado é asserção obrigatória, e a revisão adversarial mediu por quê

    Ela construiu um painel que **nunca congela** — o defeito antigo, o contador a
    crescer ao lado de *"100% concluído"* — e as três asserções originais **passaram nas duas
    implementações**. A única diferença observável era o sufixo, e `grep` sobre `tests/` achava
    **zero** ocorrências de `"até a leitura do fim"`: o rótulo não era afirmado em lugar nenhum.

    > **Um par causal cujas duas metades produzem o mesmo texto não separa nada.** As duas
    > asserções de rótulo — e a leitura **não terminal** no fim — são o que fecha isso.
    """
    painel = _painel_de_pipeline(qtbot, [130.0, 1500.0, 1600.0])
    resultado = {"run_id": 1, "estado_concordancia": "nao_sobrescreveu"}

    painel.aceita_disparo({"identificador": "primeira"}, instante=100.0)
    primeira = painel.recebe_leitura(200, resultado)

    painel.aceita_disparo({"identificador": "segunda"}, instante=1000.0)
    segunda = painel.recebe_leitura(200, resultado)

    assert "decorrido: 30 s" in primeira, f"âncora: a primeira corrida não deu 30 s: {primeira}"
    assert "decorrido: 500 s" in segunda, f"a segunda corrida não deu 500 s: {segunda}"
    assert "30 s" not in segunda.split("decorrido: ")[1], (
        f"a segunda corrida exibiu o decorrido CONGELADO da primeira — defeito reaberto: {segunda}"
    )

    # **O RÓTULO, nas duas.** Sem estas duas linhas, um painel que nunca congelasse passaria em
    # tudo acima — medido pela revisão adversarial.
    for rotulo, texto in (("primeira", primeira), ("segunda", segunda)):
        assert "até a leitura do fim" in texto, (
            f"{rotulo}: o rótulo do congelado desapareceu, e «{texto}» lê-se como duração quando é "
            f"um teto — o defeito antigo"
        )

    # E a metade VIVA, que é o que torna o rótulo discriminante em vez de constante.
    viva = painel.recebe_leitura(EM_CURSO, {"estado": "executando", "feito": 1, "total": 4})
    assert "desde o disparo" in viva, f"uma leitura em curso saiu como congelada: {viva}"
    assert "até a leitura do fim" not in viva, (
        f"o rótulo do congelado é CONSTANTE — ele aparece numa leitura viva: {viva}"
    )


def test_u5_negativo_sem_o_funil_o_congelado_da_primeira_corrida_sobrevive(qtbot: Any) -> None:
    """A planta que prova que o teste acima não passa por acaso — e ela **é** o defeito.

    **Par causal de duas sequências idênticas que diferem só pelo funil.** O ramo (a) usa
    `carimba_disparo`, que descarta o congelado; o (b) escreve a chave de disparo **direto no
    depósito**, que é exactamente o que o portão do funil acusa por AST.

    > **Sem esta planta, a asserção `"30 s" not in segunda` seria satisfeita por qualquer
    > implementação que não congelasse nada** — inclusive uma que recalculasse o decorrido a cada
    > leitura, que é o defeito ANTERIOR. A planta é o que separa *"o congelado foi descartado"* de
    > *"não há congelado nenhum"*.
    """
    terminal = {"chave_do_inicio": CHAVE_DO_INICIO, "terminou": True}

    # (a) o caminho real: o funil descarta o congelado
    com_funil = DepositoEmMemoria()
    carimba_disparo(com_funil, CHAVE_DO_INICIO, 100.0)
    primeiro_a = decorrido_da_leitura(com_funil, relogio=lambda: 130.0, **terminal)
    carimba_disparo(com_funil, CHAVE_DO_INICIO, 1000.0)
    segundo_a = decorrido_da_leitura(com_funil, relogio=lambda: 1500.0, **terminal)

    # (b) a planta: a chave de disparo escrita POR FORA do funil
    sem_funil = DepositoEmMemoria()
    carimba_disparo(sem_funil, CHAVE_DO_INICIO, 100.0)
    primeiro_b = decorrido_da_leitura(sem_funil, relogio=lambda: 130.0, **terminal)
    sem_funil.guarda(CHAVE_DO_INICIO, 1000.0)
    segundo_b = decorrido_da_leitura(sem_funil, relogio=lambda: 1500.0, **terminal)

    assert primeiro_a == primeiro_b == (30.0, True), (
        f"âncora: as duas sequências divergiram já na PRIMEIRA leitura, então a planta não isola o "
        f"funil: {primeiro_a} contra {primeiro_b}"
    )
    assert segundo_a == (500.0, True), f"o funil não descartou o congelado: {segundo_a}"
    assert segundo_b == (30.0, True), (
        f"a planta NÃO reproduziu o defeito — sem o funil o congelado devia sobreviver com "
        f"30 s, e deu {segundo_b}. A asserção do teste irmão seria vácua"
    )


def test_u5_um_disparo_novo_limpa_progresso_barra_e_resultado(qtbot: Any) -> None:
    """**A limpeza alcança os TRÊS widgets, e a revisão de código mediu os dois defeitos.**

    A primeira versão limpava só o rótulo de resultado. Medido:

    | O que sobrevivia | O que o usuário via |
    |---|---|
    | o rótulo de progresso e a barra | *"concluído · decorrido 30 s até a leitura do fim"* da
    corrida **anterior**, a 100% |
    | num disparo **recusado**, o identificador e o resultado | «Ler progresso» consultava o
    trabalho **antigo** |

    O primeiro é a mesma classe no widget vizinho: o depósito era limpo e o **pixel** não. O
    segundo é pior, porque apresenta estado antigo como resposta a uma ação nova.

    **A âncora é o estado cheio antes de cada disparo**: sem ela, *"ficou limpo"* seria
    indistinguível de *"nunca teve nada"*.
    """
    painel = _painel_de_pipeline(qtbot, [130.0, 130.0])
    painel.aceita_disparo({"identificador": "primeira"}, instante=100.0)
    painel.recebe_leitura(EM_CURSO, {"estado": "executando", "feito": 3, "total": 4})
    painel.recebe_leitura(200, {"run_id": 7, "estado_concordancia": "nao_sobrescreveu"})

    assert painel.rotulo_de_progresso.text() != PROGRESSO_EM_REPOUSO, (
        "âncora: o progresso estava vazio"
    )
    assert painel.rotulo_de_resultado.text() != RESULTADO_EM_REPOUSO, (
        "âncora: o resultado estava vazio"
    )
    assert painel.barras_de_etapa[ETAPAS[-1]].maximum() != 0, "âncora: a barra estava indeterminada"

    # (a) um disparo ACEITO limpa tudo
    painel.aceita_disparo({"identificador": "segunda"}, instante=1000.0)
    # **Os dois rótulos de repouso voltaram a ser os PREFIXADOS em 07/09/2026**, e é essa a
    # diferença que o usuário viu: `nenhuma execução disparada` debaixo de um progresso vivo lê-se
    # como contradição; `Resultado — nenhuma execução disparada` lê-se como o que é.
    assert painel.rotulo_de_progresso.text() == PROGRESSO_EM_REPOUSO, (
        "o progresso da corrida anterior sobreviveu ao disparo novo, a 100% e com o decorrido dela"
    )
    assert painel.rotulo_de_resultado.text() == RESULTADO_EM_REPOUSO
    assert painel.barras_de_etapa[ETAPAS[-1]].maximum() == 0, (
        "a barra ficou cheia com a corrida nova a começar"
    )
    assert painel.identificador == "segunda"


def test_u5_um_disparo_recusado_nao_deixa_o_identificador_antigo_de_pe(qtbot: Any) -> None:
    """O disparo recusado limpa também — e é o defeito mais insidioso dos dois.

    Cenário medido: a corrida 1 dá certo; um segundo disparo é recusado (`422`/`503`, corpo sem
    `identificador`); o identificador antigo fica de pé e o progresso do trabalho **anterior** é
    reportado como se fosse o da tentativa nova.

    **O CONSUMIDOR do identificador mudou em 07/09/2026, e a propriedade não.** Era o botão «Ler
    progresso», que saiu; hoje é a sondagem automática, que lê `pipeline.identificador` a cada
    segundo. Um identificador stale passou de *"o usuário clica e vê o trabalho errado"* a *"a
    janela segue o trabalho errado sozinha"* — o mesmo defeito, sem ninguém a pedi-lo.

    **O par causal é sobre o `identificador`**, e a âncora é a primeira metade: sem ela, *"ficou
    `None`"* seria verdade num painel que nunca teve identificador nenhum.
    """
    painel = _painel_de_pipeline(qtbot, [130.0])
    assert painel.aceita_disparo({"identificador": "boa"}, instante=100.0) == "boa"
    painel.recebe_leitura(200, {"run_id": 3, "estado_concordancia": "nao_sobrescreveu"})
    assert painel.identificador == "boa", "âncora: o identificador bom não ficou guardado"

    for recusado in ({"detail": "faltou um campo"}, {}, None, "texto", {"identificador": ""}):
        assert painel.aceita_disparo(recusado, instante=200.0) is None, f"aceitou {recusado!r}"
        assert painel.identificador is None, (
            f"o disparo recusado {recusado!r} deixou o identificador antigo de pé: a sondagem "
            f"automática seguiria o trabalho anterior como se fosse o corrente"
        )
        assert painel.rotulo_de_resultado.text() == RESULTADO_EM_REPOUSO


# ---------------------------------------------------------------------------
# Os dois portões por AST do pipeline
# ---------------------------------------------------------------------------


def _chamadas_de_relogio(fonte: str) -> list[str]:
    """As leituras de relógio de um arquivo de janela. **Por CHAMADA e não por importação.**

    ## A versão anterior gateava `import time`, e era o módulo errado

    Medido pela revisão adversarial: `import time` era acusado, e **quatro** formas escapavam —
    `from datetime import datetime`, `from PySide6.QtCore import QElapsedTimer`, o mesmo com
    `QDateTime`, e `__import__('time')`. Num arquivo PySide6 a ferramenta idiomática de tempo
    decorrido é `QElapsedTimer`, que é exactamente a que não era vista.

    > **O `time` é o único módulo que um desenvolvedor de Qt NÃO usaria para isto.** Gatear o
    > módulo em vez da operação é gatear o caminho que ninguém tomaria.

    Então o portão casa o **nome chamado**, não a importação: `monotonic`, `perf_counter`,
    `process_time`, `time`, `now`, `today`, `currentMSecsSinceEpoch`, `currentDateTime`, `elapsed`,
    `nsecsElapsed`, `restart`. `agora` **não** está na lista: ele é a fonte única de
    `interface/decorrido`, e é o que o painel recebe por injeção.
    """
    nomes = {
        "monotonic",
        "monotonic_ns",
        "perf_counter",
        "perf_counter_ns",
        "process_time",
        "time",
        "time_ns",
        "now",
        "today",
        "utcnow",
        "currentMSecsSinceEpoch",
        "currentDateTime",
        "currentTime",
        "elapsed",
        "nsecsElapsed",
        "restart",
    }
    achados: list[str] = []
    for no in ast.walk(ast.parse(fonte)):
        if not isinstance(no, ast.Call):
            continue
        alvo = no.func
        chamado = (
            alvo.id
            if isinstance(alvo, ast.Name)
            else alvo.attr
            if isinstance(alvo, ast.Attribute)
            else ""
        )
        if chamado in nomes:
            achados.append(f"linha {no.lineno} — {ast.unparse(no)[:60]}")
    return sorted(achados)


def test_u5_nenhum_arquivo_de_janela_le_relogio_e_o_portao_e_por_chamada() -> None:
    """**Nenhuma leitura de relógio nasce em `desktop/janela/`.**

    O portão do funil cobre a **escrita** da chave de disparo. Este cobre a outra metade: um
    segundo relógio em qualquer painel produz decorridos que discordam sem nada acusar.

    **Corre sobre o GLOB e não sobre um arquivo nomeado**, e a razão é a mesma do portão do laço:
    `FONTES_DA_JANELA` já existe neste arquivo com âncora de não-vacuidade, e um portão com o nome
    de um arquivo dentro é o defeito que esta unidade passou inteira a fechar.

    **O `relogio` injetado NÃO é violação**, e a distinção é o conteúdo: ele é parâmetro que o
    chamador escolhe, e o padrão dele é `interface/decorrido.agora` — a fonte única.
    """
    assert FONTES_DA_JANELA, "âncora: o glob de fontes da janela está vazio"

    for caminho in FONTES_DA_JANELA:
        achados = _chamadas_de_relogio(caminho.read_text(encoding="utf-8"))
        assert achados == [], (
            f"{caminho.name} lê o relógio direto: {achados}. A fonte única é "
            f"`ui/decorrido.agora`, e o painel a recebe por injeção"
        )

    # As cinco plantas que a revisão adversarial mediu, **em processo**, logo não ficam inertes.
    # A primeira é o controle; as quatro seguintes escapavam do portão por importação.
    base = PIPELINE_FONTE.read_text(encoding="utf-8")
    plantas = {
        "time.monotonic": "import time\n\n\ndef _p() -> float:\n    return time.monotonic()\n",
        "datetime.now": (
            "from datetime import datetime\n\n\ndef _p() -> object:\n    return datetime.now()\n"
        ),
        "QElapsedTimer.elapsed": (
            "from PySide6.QtCore import QElapsedTimer\n\n\n"
            "def _p(t: QElapsedTimer) -> int:\n    return t.elapsed()\n"
        ),
        "QDateTime": (
            "from PySide6.QtCore import QDateTime\n\n\n"
            "def _p() -> int:\n    return QDateTime.currentMSecsSinceEpoch()\n"
        ),
        "perf_counter": (
            "from time import perf_counter\n\n\ndef _p() -> float:\n    return perf_counter()\n"
        ),
    }
    for rotulo, planta in plantas.items():
        plantado = f"{base}\n\n{planta}"
        assert plantado != base, f"a planta `{rotulo}` NÃO aterrou"
        assert _chamadas_de_relogio(plantado), (
            f"o portão não acusou `{rotulo}` — num arquivo PySide6 esta é a forma "
            f"idiomática de ler tempo decorrido, e a árvore limpa acima seria vácua"
        )


def _chaves_lidas(fonte: str, funcao: str) -> set[str]:
    """Os literais que uma função usa como chave de dicionário. **Derivado, não declarado.**

    Cobre `corpo.get("x")` e `corpo["x"]`, que são as duas formas que os formatadores usam.

    **A primeira versão deste portão era uma tupla escrita à mão**, e a revisão adversarial mediu o
    que ela valia: `CAMPOS_DO_RESULTADO` tinha **zero** referências de produção — os formatadores
    usavam literais —, e `texto_da_leitura` lia **cinco** nomes que nenhuma tupla declarava. Era
    declaração paralela, não derivação, e a docstring afirmava o contrário.

    > **Derivar é o que faz um literal novo entrar na conferência sozinho.** Uma tupla à mão exige
    > que o autor se lembre de a actualizar — e *atenção não é defesa*.
    """
    arvore = ast.parse(fonte)
    alvo = next(
        (
            no
            for no in ast.walk(arvore)
            if isinstance(no, ast.FunctionDef | ast.AsyncFunctionDef) and no.name == funcao
        ),
        None,
    )
    assert alvo is not None, f"a função `{funcao}` não existe em fonte nenhuma"

    lidas: set[str] = set()
    for no in ast.walk(alvo):
        if (
            isinstance(no, ast.Call)
            and isinstance(no.func, ast.Attribute)
            and no.func.attr == "get"
            and no.args
            and isinstance(no.args[0], ast.Constant)
            and isinstance(no.args[0].value, str)
        ):
            lidas.add(no.args[0].value)
        if (
            isinstance(no, ast.Subscript)
            and isinstance(no.slice, ast.Constant)
            and isinstance(no.slice.value, str)
        ):
            lidas.add(no.slice.value)
    return lidas


def test_u5_os_campos_que_o_painel_le_sao_derivados_e_existem_no_esquema() -> None:
    """**Bijeção contra o esquema, com o conjunto lido DERIVADO por AST.**

    A primeira versão de `texto_do_resultado` lia `execucao_id` e `comparacao` — nomes plausíveis
    que **não existem** em `ExecucaoResposta`. Ela devolveria `execução None` para toda resposta
    real, e **nenhum teste de comportamento a pegaria**, porque `dict.get` não levanta.

    > É o defeito que o painel da leitura já teve, com outros dois nomes — `codigos`, `n_qc` —, e
    > a frase *"nomes plausíveis que **não existem**"* é da docstring de
    > `desktop/janela/leitura.py`.

    **O primeiro conserto foi um par de tuplas à mão, e ele não valia o que a docstring dizia.**
    Medido: zero referências de produção, e cinco nomes lidos por `texto_da_leitura` que nenhuma
    tupla declarava. Este portão **extrai** o conjunto lido e o confere contra os três esquemas que
    as duas rotas devolvem.

    **As duas direções**, e a segunda é a que a tupla não tinha: nome lido que o esquema não
    declara reprova, e o conjunto lido **muda com o código** em vez de com a memória do autor.
    """
    from refino.api.esquemas import (
        DelegacaoResposta,
        ExecucaoResposta,
        ProgressoResposta,
        RsdAntesEDepois,
        TrabalhoAceito,
    )

    fonte = PIPELINE_FONTE.read_text(encoding="utf-8")

    # **Os três formatadores mais novos entram na mesma conferência.** A frase
    # da etapa lê o corpo do progresso; a do corretor lê o aceite e desce a `delegacao`. Derivados
    # pelo mesmo extrator, e não declarados: um literal novo escrito neles entra aqui sozinho.
    lidas_etapa = _chaves_lidas(fonte, "texto_da_etapa_em_curso") | _chaves_lidas(
        fonte, "_etapa_em_curso"
    )
    lidas_corretor = _chaves_lidas(fonte, "texto_do_corretor")
    assert lidas_etapa and lidas_corretor, "âncora: nenhuma chave extraída dos formatadores novos"
    assert not lidas_etapa - set(ProgressoResposta.model_fields), sorted(
        lidas_etapa - set(ProgressoResposta.model_fields)
    )
    do_aceite = set(TrabalhoAceito.model_fields) | set(DelegacaoResposta.model_fields)
    assert not lidas_corretor - do_aceite, sorted(lidas_corretor - do_aceite)

    # **`texto_da_leitura` formata DOIS corpos, e o portão deriva dos dois.** Ele afirmava que
    # a função só lê `ProgressoResposta`, e era verdade enquanto ela formatava um corpo só. Com o
    # conserto do `None: None/4 etapas` — o `202` do `POST /execucoes` traz `TrabalhoAceito`, que
    # tem `total` e `reclamado` e não tem `estado` nem `feito` — ele reprovou por acréscimo
    # legítimo, que é o desfecho certo de um portão derivado.
    do_progresso = set(ProgressoResposta.model_fields) | set(TrabalhoAceito.model_fields)
    do_resultado = set(ExecucaoResposta.model_fields) | set(RsdAntesEDepois.model_fields)
    assert do_progresso and do_resultado, "âncora: os esquemas não declaram campo nenhum"

    # **`detail` e `msg` são do FastAPI e não de esquema nosso**, e vão declarados em vez de
    # silenciados. `detail` é o corpo de toda `HTTPException`, e a rota o usa nos três desfechos de
    # erro; `msg` é a chave de cada item quando `detail` é a **lista** de erros de validação que o
    # FastAPI emite num `422`.
    #
    # **O portão acusou `msg` na primeira execução, e a acusação estava certa:** ele é lido por
    # `_detalhe` e não existe em `ProgressoResposta`. Declarar a origem é o conserto; pôr um
    # `noqa` na asserção seria supressão — *a métrica melhorou, ou parou de medir?*
    do_fastapi = {"detail", "msg"}

    lidas_leitura = _chaves_lidas(fonte, "texto_da_leitura")
    lidas_detalhe = _chaves_lidas(fonte, "_detalhe")
    lidas_resultado = _chaves_lidas(fonte, "texto_do_resultado")

    assert lidas_leitura, "âncora: nenhuma chave extraída de `texto_da_leitura`"
    assert lidas_resultado, "âncora: nenhuma chave extraída de `texto_do_resultado`"

    fora = (lidas_leitura | lidas_detalhe) - do_progresso - do_fastapi
    assert not fora, (
        f"`texto_da_leitura`/`_detalhe` leem campos que nem `ProgressoResposta` nem "
        f"`TrabalhoAceito` declaram: {sorted(fora)}"
    )
    fora_r = lidas_resultado - do_resultado
    assert not fora_r, (
        f"`texto_do_resultado` lê campos que `ExecucaoResposta`/`RsdAntesEDepois` não declaram: "
        f"{sorted(fora_r)}"
    )

    # A planta: um nome inventado é acusado. **Em processo**, logo não fica inerte.
    plantado = fonte.replace(
        "partes = [f\"execução {corpo.get('run_id')}\"]",
        "partes = [f\"execução {corpo.get('execucao_id')}\"]",
        1,
    )
    assert plantado != fonte, "a planta do nome inventado NÃO aterrou"
    assert _chaves_lidas(plantado, "texto_do_resultado") - do_resultado == {"execucao_id"}, (
        "o extrator não viu o nome inventado plantado: a bijeção acima seria vácua"
    )


# ---------------------------------------------------------------------------
# O payload de `/execucoes`, e o par de sobrescrita
# ---------------------------------------------------------------------------


def test_u5_eixos_sem_proposta_e_recusado_aqui_e_nao_pela_rota(qtbot: Any) -> None:
    """**A recusa é local, e é o conserto de um defeito que a revisão de código mediu.**

    A rota recusa eixos sem `proposta_json` com `422`, e a mensagem que o profissional veria é
    *"Faltou um campo, ou um valor não pertence ao vocabulário aceito"* — a tela a culpá-lo por um
    defeito dela. A primeira versão desta docstring afirmava que as **duas** direções de
    `_sobrescrita_ou_422` estavam fechadas por construção; medido, só uma estava.

    O precedente é `arquivos_para_envio`, que levanta em vez de enviar multipart vazio.

    **As duas direções**, e a segunda é a âncora: com proposta, o mesmo eixo **passa**.
    """
    painel = _painel_de_pipeline(qtbot, [])
    comuns = {"modo": "neg_c18", "correcao": "none", "n_lotes": 1}

    # **«o corretor» e não `correction`:** a mensagem passou a nomear o eixo na língua da
    # tela e o botão que produz a proposta. O que esta linha afirma não mudou — a recusa nomeia o
    # eixo sem proposta —; mudou a forma do nome.
    with pytest.raises(PropostaAusenteError, match="o corretor"):
        campos_da_tela(painel, **comuns, eixos=["correction"])

    # A âncora: sem eixos, o mesmo caminho NÃO levanta; e com proposta, também não.
    assert campos_da_tela(painel, **comuns)["campos_sobrescritos"] == "[]"
    com_proposta = campos_da_tela(
        painel,
        **comuns,
        eixos=["correction"],
        recomendacao={"correction_proposta": "none", "rationale": "r", "origem": "modelo"},
    )
    assert "proposta_json" in com_proposta, (
        "âncora: nem com proposta o payload a carrega, então a excepção acima não isola a causa"
    )

    # E uma recomendação SEM `correction_proposta` é o mesmo caso: o campo é o que importa, não o
    # dicionário existir.
    with pytest.raises(PropostaAusenteError):
        campos_da_tela(painel, **comuns, eixos=["correction"], recomendacao={"rationale": "r"})


def test_u5_a_proposta_nao_entra_sem_eixos_e_essa_direcao_e_por_construcao(qtbot: Any) -> None:
    """A outra direção que `_sobrescrita_ou_422` recusa: proposta **sem** eixos.

    Esta é fechada por construção — `proposta_json` só entra na conjunção `eixos and proposta` —, e
    a âncora é o par: com eixos, a mesma proposta entra.
    """
    painel = _painel_de_pipeline(qtbot, [])
    comuns = {"modo": "neg_c18", "correcao": "none", "n_lotes": 1}
    recomendacao = {"correction_proposta": "serrf", "rationale": "r", "origem": "modelo"}

    so_proposta = campos_da_tela(painel, **comuns, recomendacao=recomendacao)
    assert "proposta_json" not in so_proposta, (
        "proposta sem eixos entrou no payload: a rota devolveria `422` por par sem eixo que alguém "
        "soubesse nomear"
    )
    assert so_proposta["campos_sobrescritos"] == "[]", "âncora: o payload saiu sem os eixos vazios"

    com_eixos = campos_da_tela(painel, **comuns, eixos=["correction"], recomendacao=recomendacao)
    assert "proposta_json" in com_eixos, (
        "âncora: nem com eixos a proposta entra, então a asserção acima não separa nada"
    )


def test_u5_o_corte_de_rsd_entra_na_proposta_e_a_ordem_e_o_conteudo(qtbot: Any) -> None:
    """**`proposta_json` é montado DEPOIS de `rsd_cutoff` entrar, e a ordem é requisito.**

    A proposta da máquina é a spec do humano com o corretor substituído. Montada antes de o corte
    entrar, a linha A do par sairia **sempre sem corte**, e a comparação de texto diria
    *"discordando"* sobre duas specs que só diferem no que ninguém sobrescreveu.

    **A âncora é a primeira asserção:** sem corte no payload, esta planta não separaria nada.
    """
    painel = _painel_de_pipeline(qtbot, [])
    campos = campos_da_tela(
        painel,
        modo="neg_c18",
        correcao="none",
        n_lotes=1,
        eixos=["correction"],
        recomendacao={"correction_proposta": "none", "rationale": "r", "origem": "modelo"},
        rsd_cutoff=0.3,
    )

    assert campos["rsd_cutoff"] == 0.3, "âncora: o corte não entrou no payload"
    spec_proposta = json.loads(json.loads(campos["proposta_json"])["spec_json"])
    assert spec_proposta["rsd_cutoff"] == 0.3, (
        "a proposta da máquina saiu SEM o corte que o humano enviou: a comparação do par diria "
        "«discordando» sobre specs que só diferem no que ninguém sobrescreveu"
    )
    assert spec_proposta["correction"] == "none"


def test_u5_o_corte_ausente_nao_entra_no_payload(qtbot: Any) -> None:
    """`rsd_cutoff` omitido e `0.0` são coisas diferentes, e a rota recusa o segundo.

    A rota declara `float | None = None` e `0.0` cai fora de `(0, 1]`. Enviar zero é pedir um
    `422`; omitir é saltar a etapa.
    """
    painel = _painel_de_pipeline(qtbot, [])
    campos = campos_da_tela(painel, modo="neg_c18", correcao="none", n_lotes=1)
    # **A não-vacuidade vem antes e na mesma função**: `campos` vazio faria a asserção de
    # ausência passar, e payload vazio é o que se obtém de um método que devolvesse `{}`.
    assert campos.get("mode") == "neg_c18", f"âncora: o payload saiu vazio ou errado: {campos}"
    assert "rsd_cutoff" not in campos, f"o corte ausente entrou como {campos.get('rsd_cutoff')!r}"


def test_d434_o_corte_de_ausentes_viaja_no_payload(qtbot: Any) -> None:
    """O SÉTIMO EIXO chega a `POST /execucoes` — o par causal que faltava.

    ## O defeito que este teste fecha, e por que a suíte não o viu

    O widget entrou com a tradução do vazio em `escolhas_do_disparo` e
    **sem ligar o valor ao payload**. Havia teste — `test_u5_as_escolhas_do_disparo_*` afirma
    `"missing_threshold": None` no dicionário de escolhas —, e ele passava: exercitava a
    **leitura** do widget, não o **envio**, e as duas quantidades têm o mesmo nome.

    > **O eixo deste teste é o payload, e não o widget.** Ler o widget já tinha dono; o que
    > ninguém afirmava era que o número sai da janela.

    As duas direções, e a segunda é a que impede a primeira de passar por vacuidade: um corte
    real viaja como número, e o mínimo do widget — que é como o Qt representa vazio — **não
    viaja**, porque `0.0` cai fora da guarda `(0, 1]` da rota e pediria um `422` que ninguém quis.
    """
    painel = _painel_de_pipeline(qtbot, [])
    comuns = {"modo": "neg_c18", "correcao": "none", "n_lotes": 1}

    # (a) âncora de não-vacuidade, na MESMA função: payload vazio faria a asserção de
    # ausência de (b) passar, e é o que se obtém de um método que devolvesse `{}`.
    painel.ausentes.setValue(painel.ausentes.minimum())
    vazio = campos_da_tela(painel, **comuns)
    assert vazio.get("mode") == "neg_c18", f"âncora: o payload saiu vazio ou errado: {vazio}"

    # (b) o vazio NÃO viaja
    assert "missing_threshold" not in vazio, (
        f"o corte de ausentes vazio entrou como {vazio.get('missing_threshold')!r} — `0.0` é o que "
        "a rota recusa por ser fração em `(0, 1]`"
    )

    # (c) e um corte REAL viaja como número. Sem esta metade, (b) passaria sobre a implementação
    # defeituosa que este teste existe para fechar — ela nunca punha a chave no payload.
    painel.ausentes.setValue(0.25)
    cheio = campos_da_tela(painel, **comuns)
    assert cheio.get("missing_threshold") == 0.25, (
        "o corte de ausentes escolhido na tela NÃO chegou ao payload: o profissional põe o filtro "
        f"e a rota corre sem ele, em silêncio. payload={cheio}"
    )

    # (d) e o par causal explícito: os dois payloads DIFEREM exactamente nesta chave.
    assert set(cheio) - set(vazio) == {"missing_threshold"}, (
        "a planta não separa: mudar só o widget de ausentes mudou outra coisa no payload, ou não "
        f"mudou nada. só-em-cheio={set(cheio) - set(vazio)}"
    )


def test_u5_os_parametros_fixos_vao_no_payload_e_a_enumeracao_vem_da_estrutura(qtbot: Any) -> None:
    """Os fixos declarados chegam **todos** à rota, e a enumeração vem da estrutura.

    Asserção de **presença**, por chave nomeada. A não-vacuidade da própria lista vem primeiro:
    `FIXOS_DECLARADOS` vazio faria o laço não olhar nada e a asserção passar.

    *O nome deste teste dizia `os_seis_parametros_fixos` e a estrutura tem **sete** chaves — o
    numeral era contagem redigitada à mão, apanhada pela revisão de padrões. O corpo sempre iterou
    a estrutura, então o comportamento estava certo e só o rótulo mentia.*
    """
    assert FIXOS_DECLARADOS, "âncora: a lista de fixos está vazia e o laço abaixo não olha nada"
    campos = campos_da_tela(
        _painel_de_pipeline(qtbot, []), modo="neg_c18", correcao="none", n_lotes=1
    )
    for chave, valor in FIXOS_DECLARADOS.items():
        assert campos[chave] == valor, f"o fixo `{chave}` não chegou ao payload"


def test_u5_as_rotas_de_trabalho_sao_montadas_e_nao_redigitadas() -> None:
    """As três rotas de `/trabalhos/` são compostas por função, e nada as monta em linha.

    > **Este teste contava UM literal e passou a contar DOIS**, quando
    > `rota_de_cancelamento` nasceu ao lado de `rota_do_progresso`. A propriedade guardada não
    > mudou — *cada rota tem exactamente um lugar que a monta* —, e a asserção passou a ser sobre
    > a **bijeção** em vez de sobre a contagem: contar literais é a forma; o que importa é que cada
    > função exista uma vez e produza um caminho distinto.
    >
    > **E passou a TRÊS**, com `rota_do_relatorio`. O portão acusou a função
    > nova na primeira execução, que é o desfecho que ele existe para dar: subir a contagem é
    > deliberado e a bijeção abaixo é o que impede a subida de ser afrouxamento — três funções,
    > três caminhos **distintos**, e um conjunto com menos de três elementos reprova.
    """
    caminhos = {
        rota_do_progresso("abc123"),
        rota_de_cancelamento("abc123"),
        rota_do_relatorio("abc123"),
    }
    assert caminhos == {
        "/trabalhos/abc123",
        "/trabalhos/abc123/cancelar",
        "/trabalhos/abc123/relatorio",
    }, f"as três funções não produziram três caminhos distintos: {sorted(caminhos)}"

    # **E CONTRA A OpenAPI, e não só contra os literais acima.** Os literais guardam a bijeção —
    # três funções, três caminhos distintos —, e essa é outra propriedade. O que eles NÃO podem
    # guardar é que o caminho montado seja o que a API SERVE: renomeada a rota em `rotas.py`, a
    # janela e o literal daqui continuam de acordo entre si e divergem da API em silêncio.
    # É o argumento que `test_as_rotas_do_historico_batem_com_o_esquema_da_api` declara verbatim
    # em `tests/test_historico_na_janela.py`, e que quatro funções de rota da janela já seguem.
    servidos = set(build_app().openapi()["paths"])

    assert servidos, "âncora: a OpenAPI não trouxe caminho nenhum"
    for constroi in (rota_do_progresso, rota_de_cancelamento, rota_do_relatorio):
        montado = constroi("{identificador}")
        assert montado in servidos, (
            f"{constroi.__name__} monta um caminho que a API não serve: {montado}"
        )

    fonte = PIPELINE_FONTE.read_text(encoding="utf-8")
    montagens = fonte.count('f"/trabalhos/')
    assert montagens == len(caminhos), (
        f"são {montagens} lugares a montar rota de `/trabalhos/` neste arquivo, e as funções são "
        f"{len(caminhos)}. Um literal a mais é uma cópia que diverge no dia em que a rota mudar, "
        "e nada acusa"
    )


@pytest.mark.integration
def test_u5_os_estados_alcancaveis_do_par_chegam_ao_banco_pelo_payload_do_painel(
    qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**O payload do painel, contra banco real, e os estados que ele de facto alcança.**

    ## O plano dizia QUATRO e a medição diz TRÊS — a correção é aditiva

    O plano pedia *"Os quatro estados do par chegam ao banco"*. Medido contra
    SQLite real, com `correction=none` nas quatro configurações:

    | Configuração enviada | `estado_concordancia` MEDIDO |
    |---|---|
    | sem eixos, sem proposta | `nao_sobrescreveu` |
    | sem eixos, sem proposta *(a que devia dar `sem_proposta`)* | **`nao_sobrescreveu`** |
    | eixos + proposta igual | `sobrescreveu_concordando` |
    | eixos + proposta diferente | `sobrescreveu_discordando` |
    | eixos **sem** proposta | recusado **aqui**, e a rota nunca é chamada |

    > **`sem_proposta` é INALCANÇÁVEL por `POST /execucoes`, e é por construção e não por lacuna.**
    > `api/rotas.py` grava `origem="manual" if eixos else "modelo"`, e `classifica_concordancia`
    > devolve `sem_proposta` apenas para **uma** linha com `origem="manual"`. Uma linha só exige
    > `eixos` vazio → `origem="modelo"`; `origem="manual"` exige `eixos` → e eixos sem
    > `proposta_json` é recusado. As três condições não têm interseção.

    **O quarto estado existe e é exercitado — por outra porta.**
    `tests/integration/test_par_de_concordancia.py` grava direto no repositório, e o registro
    daquele arquivo diz o que ele é: *"a máquina nunca propôs — uso legítimo da CLI"*. **Não é
    desfecho de interface**, e afirmá-lo desta janela seria afirmar cobertura mais larga que o
    mecanismo.

    **A âncora é a primeira asserção**: os três estados medidos têm de ser **distintos**. Se dois
    colapsassem, o teste passaria com um classificador cego.

    ## São QUATRO, e o quarto vem do painel

    A tabela acima mediu a rota antiga, e a linha *«a que devia dar `sem_proposta`»* era o
    defeito: a aba «Correção» abre nos valores padrão, e a Correção feita sem
    «Seguir sugestões do app» caía nela. A rota passou a aceitar `sem_proposta`, e o painel o manda
    quando nenhuma proposta foi adotada (`proposta_do_pedido`). O `nao_sobrescreveu` passou a pedir
    uma proposta ADOTADA — o corretor em `proposta_da_maquina` —, que é o que ele quer dizer. A
    âncora passou a quatro estados distintos.
    """
    from sqlalchemy import create_engine

    from refino.persistence.config import VARIAVEL_DA_URL, prepara_sqlite
    from refino.persistence.schema import METADATA

    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    prepara_sqlite(url)
    METADATA.create_all(create_engine(url))

    cliente_real = TestClient(build_app(), raise_server_exceptions=False)
    painel = _painel_de_pipeline(qtbot, [])
    modo = "neg_c18"
    medidas, lista = measurements_path(modo), sample_list_path(modo)
    arquivos = {
        "measurements": (medidas.name, medidas.read_bytes(), "text/csv"),
        "sample_list": (lista.name, lista.read_bytes(), "text/csv"),
    }
    proposta_igual = {"correction_proposta": "none", "rationale": "r", "origem": "modelo"}
    proposta_outra = {"correction_proposta": "serrf", "rationale": "r", "origem": "modelo"}

    def executa(eixos: list[str], recomendacao: dict[str, Any] | None, estudo: str) -> Any:
        painel.estudo.setText(estudo)
        campos = campos_da_tela(
            painel, modo=modo, correcao="none", n_lotes=1, eixos=eixos, recomendacao=recomendacao
        )
        resposta = cliente_real.post("/execucoes", files=arquivos, data=campos)
        if resposta.status_code != 202:
            return resposta.status_code
        identificador = resposta.json()["identificador"]
        for _ in range(600):
            reclamada = cliente_real.get(rota_do_progresso(identificador))
            if reclamada.status_code == 200:
                return reclamada.json()["estado_concordancia"]
        raise AssertionError(f"a execução `{estudo}` não terminou em 600 reclamações")

    # Nada adotado: o painel declara que não houve proposta. Depois, a proposta de corretor é
    # adotada, e as outras três medições correm com ela.
    sem_proposta = executa([], None, "u5_sem_proposta")
    painel.pre_preenche("correction", "none")
    assert painel.proposta_da_maquina().get("correction") == "none", "âncora: nada foi adotado"
    medido = {
        "sem_proposta": sem_proposta,
        "nao_sobrescreveu": executa([], proposta_igual, "u5_sem_eixos"),
        "sobrescreveu_concordando": executa(["correction"], proposta_igual, "u5_concordando"),
        "sobrescreveu_discordando": executa(["correction"], proposta_outra, "u5_discordando"),
    }

    assert len(set(medido.values())) == 4, (
        f"os quatro estados colapsaram: {medido}. Um classificador cego passaria neste teste"
    )
    for esperado, obtido in medido.items():
        assert obtido == esperado, f"esperado `{esperado}` e o banco devolveu `{obtido}`"


# ---------------------------------------------------------------------------
# O encerramento — três achados da revisão
# ---------------------------------------------------------------------------


def test_u5_o_arranque_drena_a_piscina_antes_de_o_servidor_descer(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**`drena_a_piscina` ganhou chamador de PRODUÇÃO, e a ausência dele era o defeito.**

    `espera_a_piscina` existia e nada o chamava fora do arnês: `_corre_a_janela` chamava
    `app.exec()` e voltava, e `main` descia o servidor no `finally` **com chamadas em voo**. Uma
    execução de SERRF — 209,9 s medidos **no `pos_hilic`** — continuaria a falar com um
    `uvicorn` a descer, e o desfecho é excepção numa thread nativa que ninguém lê.

    **O par causal é sobre a ORDEM e não sobre o valor de retorno:** o arnês registra a sequência
    de eventos e afirma que a drenagem aparece **depois** do laço. Um teste que só verificasse *"foi
    chamado"* passaria numa implementação que drenasse **antes** de `exec`, onde não há nada em voo.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    assert app is not None, "âncora: o `qtbot` devia ter construído a QApplication"

    ordem: list[str] = []
    original = JanelaPrincipal.drena_a_piscina

    def espia(self: JanelaPrincipal, teto_ms: int = 10_000) -> bool:
        ordem.append("drenou")
        return bool(original(self, teto_ms))

    monkeypatch.setattr(JanelaPrincipal, "drena_a_piscina", espia)
    monkeypatch.setenv(arranque.VARIAVEL_DA_BASE, "http://127.0.0.1:1")
    QTimer.singleShot(0, lambda: ordem.append("laco_girou"))
    QTimer.singleShot(0, app.quit)

    assert _corre_a_janela_sem_deixar_o_quit(app) == 0

    assert ordem == ["laco_girou", "drenou"], (
        f"a drenagem não aconteceu DEPOIS do laço de eventos: {ordem}. Drenar antes de `exec` "
        "esvazia uma piscina que ainda não recebeu nada"
    )


def test_u5_a_drenagem_descarta_o_que_esta_enfileirado_antes_de_esperar(qtbot: Any) -> None:
    """**`clear()` antes de `waitForDone`, e a revisão adversarial mediu por quê.**

    `waitForDone` espera pelo enfileirado **mais** o que corre. Sem o descarte, uma fila de
    trabalhos longos garante o estouro do teto, e a mensagem de aviso passa a ser a norma.

    **Par causal com âncora**: a mesma fila, drenada com descarte e esperada sem ele. A primeira
    asserção prova que a fila de facto tinha trabalho pendente — sem ela, *"esvaziou depressa"*
    seria verdade para uma piscina vazia.
    """
    janela = JanelaPrincipal(_TransporteQueLevanta(ConnectionRefusedError("recusado")))
    qtbot.addWidget(janela)

    liberar = threading.Event()
    comecou = threading.Event()

    def lenta() -> Resposta:
        comecou.set()
        liberar.wait(30)
        return Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=None)

    def rapida() -> Resposta:
        return Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=None)

    janela._piscina.setMaxThreadCount(1)
    janela.enfileira(lenta, chave="lenta")
    assert comecou.wait(10), "âncora: a lenta nunca começou, e a fila abaixo não estaria atrás dela"
    for i in range(20):
        janela.enfileira(rapida, chave=f"fila_{i}")

    assert janela._piscina.activeThreadCount() >= 1, "âncora: a piscina está inativa"

    # Com descarte: só a que corre precisa de acabar. Sem ele, as vinte da fila também.
    liberar.set()
    assert janela.drena_a_piscina(20_000) is True, "a drenagem com descarte não esvaziou"


def test_u5_a_drenagem_que_estoura_o_teto_relata_em_vez_de_seguir_calada(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """A SEGUNDA direção da guarda de drenagem.

    O teste irmão vê a drenagem **acontecer**; este vê o desfecho em que ela **não conclui**.
    `waitForDone` devolve `False` no teto, e engoli-lo colapsaria *"esvaziou"* com *"desisti de
    esperar"*, no ponto em que se decide se é seguro descer o servidor.

    **O código de saída NÃO muda**, e isso é deliberado: a janela fechou normalmente, e transformar
    uma espera estourada em falha de arranque diria que o aplicativo quebrou quando ele só tinha
    trabalho em voo.

    **E a mensagem diz que o processo VAI BLOQUEAR, porque ele bloqueia** — medido: o destrutor de
    `QThreadPool` chama `waitForDone()` sem teto. A primeira versão dizia *"o servidor vai descer
    com trabalho em voo"*, que descreve o oposto do que acontece.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    assert app is not None, "âncora: o `qtbot` devia ter construído a QApplication"

    monkeypatch.setattr(JanelaPrincipal, "drena_a_piscina", lambda self, teto_ms=10_000: False)
    monkeypatch.setenv(arranque.VARIAVEL_DA_BASE, "http://127.0.0.1:1")
    QTimer.singleShot(0, app.quit)

    assert _corre_a_janela_sem_deixar_o_quit(app) == 0, (
        "a espera estourada virou falha de arranque: a janela fechou normalmente"
    )

    saida = capsys.readouterr().err
    assert "chamada em curso" in saida, f"a espera estourada passou calada: {saida!r}"
    assert "causa: ambiente" in saida, f"a mensagem não declara a causa: {saida!r}"
    assert "só termina quando ela voltar" in saida, (
        f"a mensagem não diz que o processo bloqueia, e ele bloqueia: {saida!r}"
    )


def test_u5_fechar_a_janela_cancela_o_r_em_voo_das_vias(qtbot: Any, tmp_path: Path) -> None:
    """**`drena_a_piscina` chega ao R das vias, e é a janela que o diz.**

    O painel das vias tem piscina **própria**, e o destrutor dela espera sem teto. Sem a chamada a
    `self.vias.encerra`, fechar a janela com um enriquecimento em voo prenderia o processo por um R
    que ninguém mandou parar. `PainelDeEnriquecimento.encerra` e `PainelDeVias.encerra` têm teste
    isolado; nenhum passava por `JanelaPrincipal.drena_a_piscina`, que é o único elo que o
    `arranque` chama — e trocar a linha por `vias_ok = True` deixava verdes os outros 266 testes
    deste arquivo (medido em 01/10/2026, com só estes dois a reprovar).

    **O par causal é o desfecho que o duplo observa, e não o valor de retorno:** o R de mentira só
    sai do laço se o `cancelado()` que recebeu virar verdadeiro. Sem o conserto, a janela devolve
    `True` — a piscina **dela** está vazia — e `viu_cancelamento` fica `False`; é por isso que a
    asserção decisiva é a segunda e não a primeira.
    """
    from refino.desktop.janela.enriquecimento import OrigemDasBibliotecas
    from refino.desktop.vias_r import MotorDoR
    from refino.io.enriquecimento_r import RespostaDoR

    comecou = threading.Event()
    viu_cancelamento: list[bool] = []

    def r_que_espera_o_cancelamento(pedido: dict[str, Any], **kw: Any) -> RespostaDoR:
        comecou.set()
        # O prazo faz uma regressão REPROVAR em vez de pendurar: sem ele, um `encerra` que não
        # cancelasse prenderia o destrutor da piscina do painel para sempre.
        prazo = time.monotonic() + 10
        while not kw["cancelado"]():
            if time.monotonic() > prazo:
                return RespostaDoR("motor_falhou", "o duplo esperou 10 s sem ver cancelamento")
            time.sleep(0.02)
        viu_cancelamento.append(True)
        return RespostaDoR("cancelado", "cancelado")

    janela = JanelaPrincipal(_TransporteQueLevanta(ConnectionRefusedError("recusado")))
    qtbot.addWidget(janela)
    painel = janela.vias.enriquecimento
    tabela = tmp_path / "identificacao.csv"
    tabela.write_text(
        "Compound,Description\n"
        + "".join(f"f{i},{n}\n" for i, n in enumerate(("Glycine", "Urea", "Citric acid")))
        + "f3,Taurine\n",
        encoding="utf-8",
    )
    painel._executor = r_que_espera_o_cancelamento
    painel._motor = lambda: MotorDoR(tmp_path, tmp_path / "r.R", tmp_path, tmp_path / "b")
    painel._abre_arquivo = lambda _pai: tabela
    painel._origem_das_bibliotecas = lambda _m: OrigemDasBibliotecas(tmp_path, "teste", True)
    janela.vias.recebe_volcano(
        {
            "volcano": {
                "features": ["f0", "f1", "f2", "f3"],
                "pvalue": [0.001, 0.002, 0.003, 0.5],
                "statistic": [0.0] * 4,
            }
        }
    )
    painel._carrega_tabela()
    painel._confirma_colunas()
    assert painel.botao_enriquece.isEnabled(), "âncora: o recorte não ligou o botão"
    painel.botao_enriquece.click()
    assert comecou.wait(10), "âncora: o R de mentira nunca começou, e não há nada em voo a cancelar"
    assert viu_cancelamento == [], "âncora: o cancelamento já tinha sido pedido antes do fecho"

    assert janela.drena_a_piscina(5_000) is True, (
        "a piscina das vias não drenou: o R em voo não foi cancelado"
    )
    assert viu_cancelamento == [True], (
        "a janela drenou sem pedir ao R das vias que parasse: o `vias.encerra` não foi chamado"
    )


def test_u5_a_drenagem_da_janela_relata_a_falha_das_vias_e_passa_o_teto(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O desfecho do painel das vias entra no da janela, **nas duas direções** e com o mesmo teto.

    `vias_ok` e `espera_a_piscina` são duas metades de um `and`: se o valor das vias fosse
    ignorado, a janela diria *"esvaziou"* com um R ainda em voo, e o `arranque` desceria o servidor
    sem o aviso que existe para esse desfecho. O par é `True`/`False` sobre a **mesma** janela, com
    a piscina dela vazia nos dois — só o que as vias respondem muda, e é isso que o separa de
    «a janela esvaziou».
    """
    janela = JanelaPrincipal(_TransporteQueLevanta(ConnectionRefusedError("recusado")))
    qtbot.addWidget(janela)
    tetos: list[int] = []
    resposta = {"valor": True}

    def encerra_das_vias(teto_ms: int = 10_000) -> bool:
        tetos.append(teto_ms)
        return resposta["valor"]

    monkeypatch.setattr(janela.vias, "encerra", encerra_das_vias)

    assert janela.drena_a_piscina(1234) is True, "âncora: com as vias a drenar, a janela drena"
    resposta["valor"] = False
    assert janela.drena_a_piscina(1234) is False, (
        "as vias não drenaram e a janela diz que esvaziou: o desfecho delas foi engolido"
    )
    assert tetos == [1234, 1234], f"o teto da janela não chegou às vias: {tetos}"


# ---------------------------------------------------------------------------
# Tolerância de corpo, e os ramos `else`
# ---------------------------------------------------------------------------


def test_u5_o_resultado_com_os_opcionais_ausentes_nao_imprime_none(qtbot: Any) -> None:
    """A direção de ausência de cada campo opcional, e ela é a que fica inerte.

    `if corpo.get(x)` tem dois ramos, e o de presença é o que todo teste exercita. O de
    ausência é onde `None` vaza para a tela: *"concordância: None"* e *"None features na
    saída"* leem como medição feita, e a distinção entre *"não veio"* e *"veio nulo"*
    desaparece, na célula que o profissional lê.

    **A âncora é a comparação com o texto completo**: se os dois fossem iguais, este teste não
    separaria a ausência da presença.
    """
    painel = _painel_de_pipeline(qtbot, [])
    minimo = painel.mostra_resultado({"run_id": 3})
    completo = painel.mostra_resultado(
        {
            "run_id": 3,
            "estado_concordancia": "nao_sobrescreveu",
            "qc_rsd_median": 9.0,
            "n_features_saida": 10,
            "rsd_antes_e_depois": {"mediana_antes": 1.0, "mediana_depois": 2.0},
        }
    )

    assert minimo != completo, "âncora: ausência e presença dão o mesmo texto, e nada é separado"
    assert minimo == "execução 3", f"algo mais saiu com os opcionais ausentes: {minimo!r}"
    assert "None" not in minimo, f"`None` vazou para a tela: {minimo!r}"
    # `rsd_antes_e_depois` presente mas NÃO dicionário: o terceiro estado daquele campo. A
    # asserção é de IGUALDADE e não de ausência — *"a frase não aparece"* passaria com o texto
    # vazio, e igualdade diz o que de facto saiu.
    assert painel.mostra_resultado({"run_id": 3, "rsd_antes_e_depois": "texto"}) == (
        "execução 3"
    ), "um `rsd_antes_e_depois` que não é dicionário produziu texto de comparação"


def test_u10_a_linha_de_desfecho_da_correcao_nao_mostra_float_cru(qtbot: Any) -> None:
    """Dezassete casas decimais na linha que o usuário lê ANTES da aba de qualidade.

    `RSD mediana dos QCs: 0.1799082558331037` não é o número que um profissional de metabolómica
    lê: é o `repr` de um `float` de dupla precisão, e ele **afirma uma precisão que a medida não
    tem** — sete QCs num lote. É o mesmo gênero dos nomes de campo em inglês, que saíram da
    tela.

    **A política não é inventada aqui:** `prosa_do_numero` é a de `interface/fluxo.py`, e
    `qualidade.py` já escreve este mesmo par de medianas por ela. O defeito era a
    política **faltar** neste painel, não haver duas.

    **A ÂNCORA DE NÃO-VACUIDADE é a contagem**: *"nenhum número com mais de seis casas"* é
    satisfeito por uma linha **sem número nenhum** — que é o desfecho de um corpo que não chegou,
    de uma guarda `is not None` invertida, ou de um formatador que devolveu a string de ausência.
    Exigir os **três** é o que separa «formatou» de «não imprimiu».

    **`n_features_saida` e `run_id` ficam FORA de propósito**, e é por isso que não têm ponto
    decimal: são contagens, e `prosa_do_numero` daria *"2102,0000 features na saída"*.
    """
    import re

    painel = _painel_de_pipeline(qtbot, [])
    texto = painel.mostra_resultado(
        {
            "run_id": 3,
            "estado_concordancia": "nao_sobrescreveu",
            "qc_rsd_median": 0.1799082558331037,
            "n_features_saida": 2102,
            "rsd_antes_e_depois": {
                "mediana_antes": 0.15251233876491234,
                "mediana_depois": 0.08501987654321098,
            },
        }
    )

    # Vírgula decimal, que é a forma de `prosa_do_numero`.
    decimais = re.findall(r"\d+,\d+", texto)
    assert len(decimais) == 3, (
        f"âncora: a linha devia trazer as TRÊS medidas com casas decimais e trouxe "
        f"{len(decimais)} — {texto!r}"
    )

    longos = [n for n in decimais if len(n.split(",")[1]) > 6]
    assert longos == [], (
        f"estes números saíram crus, com a precisão do `float` e não a da medida: {longos} "
        f"— {texto!r}"
    )

    # **As contagens NÃO foram arrastadas pelo formatador**, e a asserção é de igualdade e não de
    # ausência: sem ela, um `prosa_do_numero` aplicado a mais que os três passaria neste teste.
    assert "2102 features na saída" in texto, (
        f"a contagem de features saiu formatada como medida: {texto!r}"
    )


@pytest.mark.parametrize("corpo", [None, "texto", 42, [], ()])
def test_u5_os_formatadores_do_pipeline_toleram_corpo_que_nao_e_dicionario(
    corpo: Any, qtbot: Any
) -> None:
    """Corpo que não é dicionário devolve a string de ausência, e não levanta.

    A rota pode devolver `detail` string, e um `Resposta.corpo` de erro chega como `None`. Um
    `corpo.get` sobre qualquer um deles seria `AttributeError` dentro de um slot de `clicked` — o
    desfecho é traceback no console e nada na tela, que é um defeito de silêncio.

    **Os MÉTODOS do painel entram junto das funções**, e a razão é medida: com só as funções, o
    ramo `isinstance(corpo, dict)` de `recebe_leitura` — o que decide se a barra é tocada —
    nunca corria na direção falsa. A cobertura o apontou como ramo parcial, e o `pragma`
    seria a resposta errada.
    """
    assert texto_da_leitura(EM_CURSO, corpo, None) == SEM_EXECUCAO
    assert texto_do_resultado(corpo) == SEM_EXECUCAO

    painel = _painel_de_pipeline(qtbot, [])
    antes = (
        painel.barras_de_etapa[ETAPAS[-1]].minimum(),
        painel.barras_de_etapa[ETAPAS[-1]].maximum(),
        painel.barras_de_etapa[ETAPAS[-1]].value(),
    )
    assert painel.recebe_leitura(EM_CURSO, corpo) == SEM_EXECUCAO
    assert painel.mostra_resultado(corpo) == SEM_EXECUCAO
    assert (
        painel.barras_de_etapa[ETAPAS[-1]].minimum(),
        painel.barras_de_etapa[ETAPAS[-1]].maximum(),
        painel.barras_de_etapa[ETAPAS[-1]].value(),
    ) == antes, "um corpo que não é dicionário mexeu na barra: a percentagem passou a ser inventada"


def test_u5_o_botao_de_disparo_pede_e_nao_chama(
    qtbot: Any,
) -> None:
    """O sinal do painel: ele EMITE, e não chama rota nenhuma — e as escolhas são lidas dele.

    **Este teste tinha duas metades e perdeu a primeira em 07/09/2026.** Ela cobria o ramo `else`
    da releitura manual — *"sem identificador o painel não pede progresso"* —, e o botão «Ler
    progresso» saiu a pedido de quem usa. O ramo deixou de existir junto com o slot, e um teste
    sobre um `else` que já não é código seria verificação sobre nada.

    **A propriedade equivalente na sondagem automática está coberta e noutro arquivo:**
    `test_desktop_sondagem.py` mede que a janela só abre espera depois de um `202` com
    identificador, e que um erro a fecha.

    ## O botão que emitia as escolhas saiu em 09/11/2026, e as escolhas não

    O «Executar pipeline» emitia `disparar` com o dicionário de `escolhas_do_disparo`; saiu com a
    aba «Executar e exportar». **Os três botões das etapas emitem só o número
    da etapa** (`disparar_etapa`), e a janela lê as escolhas do painel no clique
    (`JanelaPrincipal._campos_atuais`), pelo mesmo `escolhas_do_disparo`. As duas metades ficam:
    o botão emite e não chama, e a tradução do vazio para `None` é a que a janela lê.
    """
    painel = _painel_de_pipeline(qtbot, [])

    # (c) cada botão de etapa emite a SUA etapa, e o painel continua a NÃO chamar rota nenhuma.
    etapas: list[int] = []
    painel.disparar_etapa.connect(etapas.append)
    assert len(painel.botoes_de_etapa) > 1, (
        "âncora: com um botão só, um fecho errado no `lambda` emitiria a etapa certa por acidente"
    )
    for etapa, botao in painel.botoes_de_etapa.items():
        botao.click()
        assert etapas[-1] == etapa, f"o botão da etapa {etapa} emitiu {etapas[-1]}"
    assert etapas == list(painel.botoes_de_etapa), f"os botões trocaram de etapa: {etapas}"

    # **O payload mudou, e a propriedade que este teste guarda não mudou.** Ele afirmava
    # `[{"estudo": "estudo_x"}]`, e o painel não tinha os controles que a execução precisa. Hoje
    # as escolhas carregam o que só este painel sabe, e o `estudo` é lido por `campos_base`, que é
    # quem monta a spec.
    painel.estudo.setText("estudo_x")
    painel.corretor.setCurrentText("loess")
    painel.n_lotes.setValue(3)
    # **`missing_threshold`** — o sétimo eixo passou a ser do usuário, e repousa em
    # vazio como os outros dois campos numéricos. `None` aqui é a asserção de que o vazio **não
    # viaja**: mandá-lo como `0.0` afirmaria «descarte feature nenhuma» onde ninguém escolheu.
    # **`paralelo` NÃO tem vazio a traduzir** — `QCheckBox` é
    # binário por construção, ao contrário dos três acima. É por isso que ele aparece com valor e
    # não com `None`: omiti-lo quando ticado faria a tela depender do padrão do servidor no caso
    # comum.
    # **As duas fracoes de posto entraram em 10/09/2026**, e repousam vazias pela mesma regra
    # dos tres campos numericos acima: `None` aqui e a asercao de que o vazio NAO viaja, porque
    # `0.0` cai fora de `(0, 1]` e a rota o recusaria com 422.
    assert painel.escolhas_do_disparo() == {
        "correcao": "loess",
        "n_lotes": 3,
        "rsd_cutoff": None,
        "ranking_keep_fraction": None,
        "ranking_keep_fraction_2": None,
        "normalization_reference": None,
        "missing_threshold": None,
        "paralelo": True,
    }, f"as escolhas não carregaram o que a tela mostra: {painel.escolhas_do_disparo()}"

    # (d) **o vazio dos dois campos numéricos vira `None` e não zero**, e é o par causal do
    # `specialValueText`: sem esta metade, um mapeamento que enviasse `0` passaria — e `0` é o que a
    # API recusa para o corte, por ser fração em `(0, 1]`.
    painel.n_lotes.setValue(painel.n_lotes.minimum())
    painel.corte.setValue(painel.corte.minimum())
    assert painel.escolhas_do_disparo() == {
        "correcao": "loess",
        "n_lotes": None,
        "rsd_cutoff": None,
        "ranking_keep_fraction": None,
        "ranking_keep_fraction_2": None,
        "normalization_reference": None,
        "missing_threshold": None,
        "paralelo": True,
    }, f"o vazio não virou `None`: {painel.escolhas_do_disparo()}"

    # (e) e um corte REAL viaja como número. Sem isto, (d) passaria com um mapeamento que
    # devolvesse `None` sempre.
    painel.corte.setValue(0.3)
    # **As duas fracoes de posto exercitam a MESMA direcao, e por isso entram aqui —
    # 10/09/2026.** Sem um valor real, a metade (d) acima passaria sobre um mapeamento que
    # devolvesse `None` sempre para elas, e a traducao do vazio nunca seria exercitada nos dois
    # sentidos — que e a planta que nao separa.
    painel.fracao_posto.setValue(0.6)
    painel.fracao_posto_2.setValue(0.9)
    painel.referencia.setCurrentText("QC1")
    escolhas = painel.escolhas_do_disparo()
    assert escolhas["rsd_cutoff"] == 0.3, f"o corte real não viajou: {escolhas}"
    assert escolhas["ranking_keep_fraction"] == 0.6, f"a fracao real nao viajou: {escolhas}"
    assert escolhas["ranking_keep_fraction_2"] == 0.9, f"a fracao 2 nao viajou: {escolhas}"
    assert escolhas["normalization_reference"] == "QC1", f"a referencia real nao viajou: {escolhas}"


def test_u11_o_corretor_esta_na_ordem_do_dominio_e_nao_na_alfabetica(qtbot: Any) -> None:
    """`none` primeiro, e a ordem é COMPORTAMENTO e não apresentação.

    O primeiro item é o que o combo pré-seleciona, então ele é o corretor que roda para quem não
    mexe no seletor. `sorted(AVAILABLE_METHODS)` daria `loess, none, serrf` e **corrigiria por
    omissão** a matriz de quem não pediu correção nenhuma.

    **A asserção é de ordem e não de conjunto**, porque um teste de conjunto passaria com a ordem
    alfabética, que é exactamente o defeito.
    """
    painel = _painel_de_pipeline(qtbot, [])
    itens = [painel.corretor.itemText(i) for i in range(painel.corretor.count())]
    assert itens, "o combo do corretor está vazio"
    assert itens[0] == "none", (
        f"o primeiro item é {itens[0]!r}, e é ele que roda para quem não mexe no seletor"
    )
    assert itens != sorted(itens), (
        f"a ordem é alfabética: {itens}. `sorted` trocaria o padrão de `none` para `loess`"
    )
    assert itens == list(CORRETORES), (
        f"o combo divergiu da fonte derivada: {itens} contra {list(CORRETORES)}"
    )


def test_u5_o_detalhe_cobre_as_tres_formas_que_o_fastapi_emite(qtbot: Any) -> None:
    """`detail` string, `detail` lista, e `detail` ausente. **Os três, e o terceiro é o `else`.**

    O FastAPI emite `detail` como **string** numa `HTTPException` construída por nós, e como
    **lista** de `{loc, msg, type}` num `422` de validação. E há o caso em que não há `detail`
    nenhum — corpo de progresso, ou dicionário vazio.

    **Comece pelo ramo `else`**: ele dispara por ausência e não há o que plantar nele. Aqui a
    ausência é o corpo sem `detail`, e o desfecho correto é a frase de falha **sem** detalhe — não
    a palavra `None`.

    A âncora é a distinção mútua: se dois dos três dessem o mesmo texto, nada seria separado.
    """
    painel = _painel_de_pipeline(qtbot, [130.0, 130.0, 130.0, 130.0])
    painel.aceita_disparo({"identificador": "t"}, instante=100.0)

    como_string = painel.recebe_leitura(409, {"detail": "a matriz ficou vazia"})
    como_lista = painel.recebe_leitura(
        422, {"detail": [{"loc": ["body", "mode"], "msg": "campo obrigatório", "type": "missing"}]}
    )
    sem_detalhe = painel.recebe_leitura(500, {})
    corpo_torto = painel.recebe_leitura(500, "nem dicionário")

    assert len({como_string, como_lista, sem_detalhe}) == 3, (
        f"duas das três formas de `detail` deram o mesmo texto: "
        f"{(como_string, como_lista, sem_detalhe)}"
    )
    assert "a matriz ficou vazia" in como_string
    assert "campo obrigatório" in como_lista, f"a lista de erros não foi lida: {como_lista}"
    assert "None" not in sem_detalhe, f"`None` vazou onde não havia `detail`: {sem_detalhe}"
    assert "O trabalho falhou" in sem_detalhe
    assert "O trabalho falhou" in corpo_torto, (
        f"um corpo que não é dicionário num desfecho de erro perdeu a frase de falha: {corpo_torto}"
    )


def test_u5_sem_disparo_conhecido_nenhuma_leitura_inventa_decorrido(qtbot: Any) -> None:
    """A linha do decorrido **não** é renderizada quando não há disparo — nos quatro desfechos.

    Contar a partir de um disparo alheio daria um número plausível sobre a coisa errada, e
    esta é a direção de ausência de `_texto_do_decorrido`: `None` significa *"não sei quando
    começou"*.

    **A âncora é o par:** com disparo, os mesmos quatro desfechos **trazem** a linha. Sem ela,
    *"não apareceu"* seria verdade num painel que nunca renderiza decorrido nenhum.
    """
    leituras = (
        (EM_CURSO, {"estado": "executando", "feito": 1, "total": 4}),
        (200, {"run_id": 1}),
        (404, {"detail": "desconhecido"}),
        (500, {"detail": "falhou na etapa «x»"}),
    )

    sem = _painel_de_pipeline(qtbot, [130.0, 130.0, 130.0, 130.0])
    for codigo, corpo in leituras:
        texto = sem.recebe_leitura(codigo, corpo)
        assert "decorrido" not in texto, (
            f"HTTP {codigo}: o painel inventou um decorrido sem saber quando o disparo aconteceu — "
            f"{texto}"
        )

    com = _painel_de_pipeline(qtbot, [130.0, 130.0, 130.0, 130.0])
    com.aceita_disparo({"identificador": "t"}, instante=100.0)
    for codigo, corpo in leituras:
        texto = com.recebe_leitura(codigo, corpo)
        assert "decorrido" in texto, (
            f"âncora: HTTP {codigo} não traz o decorrido nem COM disparo conhecido, então a "
            f"asserção de ausência acima é vácua — {texto}"
        )


def test_u5_a_barra_ignora_total_zero_em_vez_de_dividir_por_ele(qtbot: Any) -> None:
    """`total=0` mantém a barra indeterminada, e um `total` real passa a escala.

    Um `setRange(0, 0)` é indeterminado no Qt; escrever `setValue` contra um máximo de zero
    afirmaria uma percentagem que ninguém mediu. **Par causal com âncora:** o estado inicial
    é indeterminado, `total=0` não o muda, e `total=4` muda.
    """
    painel = _painel_de_pipeline(qtbot, [130.0, 130.0, 130.0])
    painel.aceita_disparo({"identificador": "t"}, instante=100.0)

    assert painel.barras_de_etapa[ETAPAS[-1]].maximum() == 0, (
        "âncora: a barra não nasce indeterminada"
    )

    painel.recebe_leitura(EM_CURSO, {"estado": "executando", "feito": 0, "total": 0})
    assert painel.barras_de_etapa[ETAPAS[-1]].maximum() == 0, (
        "`total=0` passou a escala real e a percentagem é inventada"
    )

    painel.recebe_leitura(EM_CURSO, {"estado": "executando", "feito": 2, "total": 4})
    assert (
        painel.barras_de_etapa[ETAPAS[-1]].maximum(),
        painel.barras_de_etapa[ETAPAS[-1]].value(),
    ) == (4, 2)


# ---------------------------------------------------------------------------
# Levar o resultado, com download se e somente se
# ---------------------------------------------------------------------------
#
# **A REVISÃO ADVERSARIAL DERRUBOU A PRIMEIRA VERSÃO DESTE BLOCO, e o número é o argumento:**
# **seis** mutantes passaram nos 103 testes do arquivo, e a propriedade central da unidade —
# *"clicar em Salvar escreve os bytes DAQUELE artefato"* — não tinha teste nenhum. Os seis:
#
# | Mutante | O que fazia | Antes | Sem o botão de salvar |
# |---|---|---|---|
# | m1 | `recebe` marcava os **três** em qualquer `200` com bytes | passava | expressável |
# | m2 | `salva` devolvia o primeiro valor de `_bytes` | passava | **não é expressável** |
# | m3 | erro de fecho no `lambda` do botão de **salvar** | passava | **não é expressável** |
# | m4 | os botões de salvar ligados a **nada** | passava | **não é expressável** |
# | m5 | o escritor passava `""` como nome sugerido ao diálogo | passava | **não é expressável** |
# | m6 | `define_modo` limpava os bytes e não mexia na tela | passava | expressável |
#
# **A causa comum era uma:** nenhum teste clicava num botão de salvar, e nenhum tinha dois
# artefatos com bytes ao mesmo tempo. O irmão m7 — o mesmo erro de fecho no botão de **gerar**,
# três linhas acima — **reprovava**, porque aquele botão era clicado. A cobertura de 100% não viu
# nada: a linha do `clicked.connect(lambda …)` é contada quando o `lambda` é **criado**.
#
# **O BOTÃO DE SALVAR SAIU, e quatro dos seis mutantes saíram com ele.** Havia dois botões
# redundantes — um para gerar e baixar a matriz, outro só para baixar —, ficou o de gerar e
# baixar, e com o botão saíram
# `salva`, o `Protocol` `Escritor` e `escritor_por_dialogo`, que existiam só para ele. **Os dois
# mutantes que continuam expressáveis são os que atacam o ESTADO do painel**, e é por eles que a
# segunda superfície da bicondicional continua a ser afirmada: ela deixou de ser o `isEnabled()`
# do botão de salvar e passou a ser o **texto do rótulo** — ver `_anunciados` abaixo.
#
# **O que os quatro mediam NÃO deixou de ser medido; mudou de casa.** A escrita em disco passou a
# ser de `JanelaPrincipal._escreve_no_destino`, e é lá que os testes `test_d367_*` afirmam
# *"os bytes escritos são os DAQUELE artefato, no caminho que o clique escolheu"*.


ENTRADA_A = "neg_c18|none|aaaaaaaaaaaaaaaa"
ENTRADA_B = "neg_c18|loess|aaaaaaaaaaaaaaaa"


def _painel_de_artefatos(
    qtbot: Any,
    modo: str = "neg_c18",
    identidade: str = ENTRADA_A,
    escolhedor: Any = None,
) -> PainelDeArtefatos:
    """O painel com a entrada definida e o escolhedor de pasta INJETADO.

    **Sem a injeção a suíte MENTE, e isto foi medido.** O botão de gerar pergunta a pasta
    antes de emitir. Com o escolhedor real, o `QFileDialog` devolve vazio sob `offscreen`, o clique
    deixa de emitir, e um teste de roteamento de sinal reprova por uma razão que nada tem a ver com
    o que ele mede — foi exactamente o que aconteceu com o teste do fecho do `lambda`.

    **Um caminho sem disco serve, e é ele que declara o alcance:** o painel apenas GUARDA a pasta.
    Quem escreve é `principal`, e isso tem arnês próprio nos testes `test_d367_*`.

    > **Devolve SÓ o painel.** Devolvia `(painel, escritor)`, e o segundo era o duplo
    > do `Escritor` que o botão «Baixar X» usava. Sem aquele botão não há escritor no painel: o
    > módulo não escreve em disco por nome nenhum, e há portão por AST a afirmá-lo.
    """
    painel = PainelDeArtefatos(escolhedor=escolhedor or (lambda: Path("destino")))
    qtbot.addWidget(painel)
    painel.define_entrada(identidade, modo)
    return painel


def _anunciados(painel: PainelDeArtefatos) -> tuple[str, ...]:
    """As chaves cujo RÓTULO anuncia artefato gerado. **É o segundo observável da bicondicional.**

    `salvaveis()` lê `_bytes`; isto lê o que a tela de facto mostra. Afirmar só o primeiro deixa
    passar o mutante m1 — que marcava os três na resposta de um —, porque aquele acessor lê o
    mesmo dicionário que o defeito corrompe.

    ## OS DOIS NÃO CONCORDAM SEMPRE, e a versão anterior desta nota dizia que sim

    Ela afirmava *"os dois têm de concordar"*, e isso era verdade do **botão** e não do rótulo. O
    `isEnabled()` antigo era estado DEDICADO: só `recebe` e `define_entrada` lhe tocavam. O texto
    do rótulo é o único slot do artefato, e **seis** métodos escrevem nele. Medido em 03/09/2026
    com o painel real:

    | Estado | `salvaveis()` | `_anunciados()` |
    |---|---|---|
    | logo depois de `recebe(200, bytes)` | `('exportar',)` | `('exportar',)` |
    | `pede_e_gera` cancelado (`SEM_DESTINO`) | `('exportar',)` | `()` |
    | depois de `mostra_espera` | `('exportar',)` | `()` |
    | depois de `mostra_desistencia` | `('exportar',)` | `()` |

    > **Os dois são a bicondicional logo depois de `recebe`, `recebe_relatorio` ou
    > `define_entrada` — e só aí.** Nos outros quatro caminhos o rótulo está a falar de outra
    > coisa, e `_anunciados() == ()` significa *"a tela não está a anunciar artefato agora"* e
    > **não** *"não há bytes"*. `mostra_espera` declara a omissão como deliberada na própria
    > docstring, em `artefatos.py`.

    **Consequência para quem escreve teste:** asserção de ausência sobre este helper é satisfeita
    por três métodos que nada têm a ver com a bicondicional. Onde a pergunta for *"há bytes?"*, o
    oráculo é `salvaveis()`; onde for *"a tela diz que há?"*, é este — e os dois juntos só são a
    bicondicional no instante em que a resposta acaba de chegar.

    ## A superfície mudou; a propriedade não

    Até 03/09/2026 este helper chamava-se `_habilitados` e lia
    `painel.botoes_de_salvar[c].isEnabled()`: o botão existia habilitado exactamente quando havia
    bytes. Com o botão removido, o que resta na tela é o texto do rótulo, e o nome antigo passaria
    a mentir sobre o que se lê.

    **`MARCA_DE_GERADO` é IMPORTADA e não redigitada**, pela razão que o comentário dela declara em
    `artefatos.py`: duas cópias da frase divergem no dia em que uma mudar, e o portão passaria a
    afirmar sobre uma frase que a tela já não escreve.

    > **A marca não colide com nenhuma das outras frases do painel**, e isso é escolha e não sorte:
    > `SEM_ARTEFATO` é *"nada gerado ainda"* — no singular —, e nem ele, nem `SEM_BYTES`, nem
    > `SEM_DESTINO`, nem as duas frases da espera contêm `"gerados"`.
    """
    return tuple(c for c in CHAVES_NA_ORDEM if MARCA_DE_GERADO in painel.rotulos[c].text())


def test_u6_os_tres_artefatos_sao_oferecidos_e_a_enumeracao_vem_da_estrutura(qtbot: Any) -> None:
    """Asserção de **presença**, por chave, com a enumeração derivada de `interface/fluxo.py`.

    A não-vacuidade vem primeiro e na mesma função: `ARTEFATOS` vazio faria o laço não olhar
    nada, e cada asserção de presença passaria por não ser executada.

    **A metade dos DOIS rótulos CAIU, e não foi afrouxada: ela perdeu o sujeito.** Ela
    afirmava `gerar != salvar`, e a razão era esta: o botão que **calcula** dizia *"Baixar"* e o que
    entrega os bytes dizia o mesmo, então quem clicava no primeiro esperava arquivo e recebia
    identificador. Com o botão de salvar removido há **um** rótulo de botão por artefato, e não
    existe par a comparar: manter a asserção exigiria inventar o segundo lado.

    O que sobrevive é a metade que ainda tem sujeito — o botão que dispara **anuncia que gera** —,
    e ela está abaixo com a sua própria justificativa.

    **E o rótulo de ESTADO entra na asserção de presença no lugar do botão que saiu**, porque é ele
    a segunda superfície da bicondicional (ver `_anunciados`).
    """
    assert ARTEFATOS, "âncora: a partição de artefatos está vazia e o laço não olha nada"
    assert set(CHAVES_DE_ARTEFATO) == set(CHAVES_NA_ORDEM), (
        "a derivação divergiu da partição de `fluxo.py`"
    )

    painel = _painel_de_artefatos(qtbot)
    for passo in ARTEFATOS:
        assert passo.chave in painel.botoes_de_geracao, f"`{passo.chave}` não tem botão de gerar"
        assert passo.chave in painel.rotulos, f"`{passo.chave}` não tem rótulo de estado"
        gerar = painel.botoes_de_geracao[passo.chave].text()
        assert gerar, f"`{passo.chave}`: o botão que dispara nasceu sem rótulo"
        # **A PREMISSA DESTA ASSERÇÃO INVERTEU, e ela não foi afrouxada.**
        #
        # Ela dizia `"Baixar" not in gerar`, e a razão era: *"quem clicasse no primeiro
        # esperaria arquivo e receberia identificador"*. Depois do gesto unificado o primeiro botão
        # **entrega o arquivo** — pede a pasta no clique e escreve lá quando os bytes chegam —,
        # então a premissa passou a ser falsa **por construção**, e mantê-la exigiria um rótulo que
        # mente sobre o que o botão faz.
        #
        # **Renomear o botão para escapar do portão era a saída errada.**
        # O que fica é a propriedade que sobrevive à mudança: o botão que dispara **anuncia que
        # gera**, para que ninguém o leia como entrega de algo que ainda não foi calculado. Um
        # rótulo *"Baixar a matriz corrigida"* no botão de disparo volta a reprovar aqui, que é o
        # defeito exacto que se mediu.
        assert gerar.startswith("Gerar"), (
            f"`{passo.chave}`: o botão que dispara não anuncia que gera: {gerar!r} — quem o lesse "
            f"como entrega esperaria um arquivo que ainda não foi calculado (R18)"
        )


def test_u6_cada_botao_de_gerar_emite_a_sua_chave(qtbot: Any) -> None:
    """**Cada botão emite a SUA chave** — o mutante que morre aqui é o erro de fecho no `lambda`.

    Os botões nascem num laço, com `lambda _=False, chave=passo.chave:`. Um fecho sem o argumento
    de valor por omissão faz os três emitirem a chave da **última** iteração, e o painel passa a
    gerar sempre o mesmo artefato — com a cobertura em 100%, porque a linha do
    `clicked.connect(lambda …)` é contada quando o `lambda` é **criado**.

    ## A metade «nos dois lados» caiu, e o que ela media MUDOU DE CASA

    O nome antigo era `…_nos_dois_lados`: gerar e salvar. Com o botão de salvar removido resta um
    lado, e a propriedade que a outra metade media — *"os bytes escritos são os DAQUELE artefato"* —
    **não deixou de ser medida**. Ela passou para onde a escrita passou a acontecer,
    `JanelaPrincipal._escreve_no_destino`: dois cliques com corretores diferentes deixam dois
    arquivos distintos (`test_d367_dois_corretores_no_mesmo_modo_nao_colidem_no_disco`), e a
    identidade byte a byte contra a rota real está em
    `test_u6_os_bytes_escritos_sao_identicos_aos_da_resposta_da_rota`.

    **A âncora é a lista de chaves e não a contagem**: um fecho errado produz três emissões
    — o número certo, com o conteúdo errado —, e só a ordem das chaves separa as duas coisas. Com
    uma chave só nem isso separaria, e é por isso que a cardinalidade é afirmada antes.

    **O clique passa por `pede_e_gera`, que pergunta a pasta antes de emitir** — é o escolhedor
    injetado por `_painel_de_artefatos` que faz o gesto chegar ao sinal sob `offscreen`.
    """
    assert len(CHAVES_NA_ORDEM) > 1, (
        "âncora: com uma chave só, um fecho errado emitiria a chave certa por acidente"
    )
    painel = _painel_de_artefatos(qtbot)

    pedidos: list[str] = []
    painel.gerar.connect(pedidos.append)
    for chave in CHAVES_NA_ORDEM:
        painel.botoes_de_geracao[chave].click()

    assert pedidos == list(CHAVES_NA_ORDEM), f"os botões de gerar trocaram de chave: {pedidos}"


def test_u6_a_bicondicional_do_anuncio_do_artefato_nos_tres_desfechos(qtbot: Any) -> None:
    """**O painel anuncia artefato se e somente se há bytes**, nos três desfechos.

    | Desfecho | Anúncio no rótulo |
    |---|---|
    | `202`, ainda a correr | **não** |
    | `200` com bytes | **sim** |
    | erro | **não**, e a mensagem de `cliente.py` no rótulo |

    **A asserção de ausência tem âncora na mesma função e antes**: sem o desfecho do meio,
    *"não anuncia nada"* seria verdade num painel que nunca anuncia coisa nenhuma. E a ordem é
    deliberada — o erro vem **depois** do sucesso, que é a direção em que o anúncio stale sobrevive.

    ## O observável mudou, e a bicondicional não

    Até 03/09/2026 ele era `botoes_de_salvar[c].isEnabled()`, escolhido contra `isVisible()` por
    medição: sob `QT_QPA_PLATFORM=offscreen` e sem `show()`, `isVisibleTo(painel)` é `True` para
    todos os botões e não distingue nada. Com o botão de salvar removido o que a tela mostra é o
    **texto do rótulo**, e é ele que `_anunciados` lê — ainda duas superfícies, ainda independentes
    de `salvaveis()`.

    ## E cada asserção olha OS TRÊS artefatos, não só o da chave

    A primeira versão olhava um. O mutante que marcava **todos** os três em qualquer `200` com
    bytes passava — e é a bicondicional violada verbatim: um anúncio sem bytes por trás.
    """
    painel = _painel_de_artefatos(qtbot)

    assert painel.salvaveis() == (), "âncora: o painel nasce com bytes"
    assert _anunciados(painel) == ()

    # (a) `202`: aceito e a correr — nada a levar ainda
    texto = painel.recebe("corrigir", EM_CURSO, None, "irrelevante")
    assert _anunciados(painel) == (), "o `202` anunciou artefato antes de haver bytes"
    assert "corre no servidor" in texto, texto

    # (b) `200` com bytes — a âncora que impede as ausências de serem vácuas
    painel.recebe("corrigir", 200, b"x" * 2048, "irrelevante")
    assert _anunciados(painel) == ("corrigir",), (
        f"a resposta de UM artefato mexeu no anúncio dos outros: {_anunciados(painel)}"
    )
    assert painel.salvaveis() == ("corrigir",)

    # (c) o erro DEPOIS do sucesso — o anúncio stale é o defeito já medido no painel vizinho
    mensagem = "Faltou um campo, ou um valor não pertence ao vocabulário aceito."
    texto_do_erro = painel.recebe("corrigir", 422, None, mensagem)
    assert _anunciados(painel) == (), (
        "o anúncio do artefato sobreviveu ao erro: o rótulo continuaria a prometer os bytes de uma "
        "tentativa que já não é a corrente"
    )
    assert painel.salvaveis() == ()
    assert texto_do_erro == mensagem, (
        f"a mensagem não é a de `cliente.py` — alguma coisa a reescreveu: {texto_do_erro!r}"
    )


def test_u6_o_desfecho_de_um_artefato_nao_toca_os_outros(qtbot: Any) -> None:
    """**Dois artefatos com bytes ao mesmo tempo, e um erro num deles.**

    Nenhum teste da primeira versão tinha dois artefatos vivos, então um `recebe` que fizesse
    `_bytes.clear()` passava — e no painel real gerar `processar` revogaria silenciosamente o
    artefato de `corrigir` que o usuário acabara de pagar.

    **A âncora é o estado com os dois vivos**, e a asserção é sobre a **partição**: o que muda é só
    a chave que respondeu.

    ## Os bytes que sobram são os certos, e é o TAMANHO no rótulo que o diz

    Antes esta metade era medida clicando no botão de salvar de `corrigir` e lendo o que o
    escritor recebeu. Sem aquele botão, o observável que resta é o rótulo — e ele carrega
    `texto_do_tamanho`, então dois artefatos de tamanhos **diferentes** produzem frases diferentes.
    A âncora dos tamanhos distintos é o que faz a asserção separar alguma coisa: com 32 e 32
    bytes, um `recebe` que trocasse os bytes de um pelo do outro passaria.
    """
    painel = _painel_de_artefatos(qtbot)

    de_corrigir, de_processar = b"C" * 32, b"P" * 64
    assert texto_do_tamanho(de_corrigir) != texto_do_tamanho(de_processar), (
        "âncora: os dois artefatos têm o mesmo tamanho, e o rótulo não separa qual sobreviveu"
    )

    painel.recebe("corrigir", 200, de_corrigir, "ok")
    painel.recebe("processar", 200, de_processar, "ok")
    assert _anunciados(painel) == ("corrigir", "processar"), (
        f"âncora: os dois não ficaram vivos ao mesmo tempo: {_anunciados(painel)}"
    )

    # Um erro em `processar` **não** toca `corrigir`.
    painel.recebe("processar", 500, None, "defeito da ferramenta")
    assert _anunciados(painel) == ("corrigir",), (
        f"o erro num artefato revogou o artefato de outro: {_anunciados(painel)}"
    )
    assert painel.salvaveis() == ("corrigir",)
    assert (
        painel.rotulos["corrigir"].text() == f"{texto_do_tamanho(de_corrigir)} {MARCA_DE_GERADO}"
    ), (
        f"o rótulo de `corrigir` foi reescrito pelo desfecho de `processar`: "
        f"{painel.rotulos['corrigir'].text()!r}"
    )


def test_u6_uma_chave_que_nao_e_de_artefato_e_ignorada(qtbot: Any) -> None:
    """`recebe` com a chave de outra rota **não levanta e não muda nada**.

    `JanelaPrincipal.resposta_recebida` é emitido para **toda** chamada — validação, diagnóstico,
    execução, progresso —, e este painel vai ligar-se a ele. Sem a guarda, a primeira resposta de
    `validacao` faria `self.rotulos["validacao"]` levantar `KeyError` **depois** de `_bytes.pop` já
    ter corrido; num slot de Qt a excepção é engolida para o `stderr`, e o painel simplesmente para
    de actualizar.

    **A âncora é o estado preservado**: um artefato vivo continua vivo depois de uma chave alheia.
    """
    painel = _painel_de_artefatos(qtbot)
    painel.recebe("corrigir", 200, b"C" * 16, "ok")
    assert _anunciados(painel) == ("corrigir",), "âncora: não havia artefato vivo a preservar"

    for alheia in ("validacao", "diagnostico", "execucao", "multiverso", "progresso"):
        assert painel.recebe(alheia, 200, b"lixo", "ok") == "", f"`{alheia}` produziu texto"

    assert _anunciados(painel) == ("corrigir",), (
        "uma chave alheia mexeu no estado do painel — o `_bytes.pop` correu antes da guarda"
    )
    assert painel.salvaveis() == ("corrigir",)


def test_u6_um_200_sem_bytes_nao_anuncia_artefato_e_nao_diz_ok(qtbot: Any) -> None:
    """`200` e `conteudo=None` é *"respondeu e não produziu artefato"*, e não há o que levar.

    **`Resposta.conteudo` é `bytes | None` e nunca `b""`**, e o comentário de `cliente.py` diz por
    quê. Este teste é a metade daquele desenho que vive no painel — e a âncora é o par com o `200`
    que **tem** bytes.

    **E o texto NÃO é a mensagem de `cliente.py`.** Medido: `cliente.mensagem_de(200)` é `"OK"`, e
    escrever `"OK"` ao lado de um controle ausente é uma palavra de sucesso a descrever a ausência
    do que se pediu, na célula que o profissional lê. A primeira versão deste teste
    passava mensagens escritas à mão que a rota nunca produz para um `200`, e o defeito ficava
    escondido.
    """
    painel = _painel_de_artefatos(qtbot)

    texto = painel.recebe("processar", 200, None, cliente.mensagem_de(200))
    assert _anunciados(painel) == ()
    assert texto == SEM_BYTES, f"o `200` sem bytes saiu como {texto!r}"
    assert texto != cliente.mensagem_de(200), (
        f"âncora: a mensagem de `cliente.py` para um 200 é {cliente.mensagem_de(200)!r}, e este "
        f"painel não a repete para descrever a ausência do artefato"
    )

    assert painel.recebe("processar", 200, b"", "vazio") == SEM_BYTES, (
        "bytes vazios saíram como artefato: um arquivo de zero bytes não é artefato"
    )
    assert _anunciados(painel) == ()

    painel.recebe("processar", 200, b"conteudo", "ok")
    assert _anunciados(painel) == ("processar",), (
        "âncora: nem com bytes o painel anuncia, e as asserções acima são vácuas"
    )


def test_u6_o_tamanho_nao_diz_zero_kb_para_artefato_que_existe() -> None:
    """`0 KB gerados` é o rótulo a dizer que nada foi produzido para um arquivo que existe.

    `len(conteudo) // 1024` dá zero para qualquer artefato abaixo de 1 KiB. Par causal com âncora:
    abaixo de 1 KiB sai em **bytes**, e acima sai em **KB**.

    > **A frase que este número acompanha mudou** — era *"0 KB prontos para salvar"* ao lado
    > de um botão de salvar habilitado, as duas superfícies a discordar; hoje é
    > `f"{tamanho} {MARCA_DE_GERADO}"`, e o rótulo é a única superfície. A razão de o número não
    > poder ser zero é a mesma: ele é o que diz que há artefato.
    """
    assert texto_do_tamanho(b"x" * 500) == "500 bytes"
    assert texto_do_tamanho(b"x" * 1023) == "1023 bytes"
    assert texto_do_tamanho(b"x" * 1024) == "1 KB", "âncora: nem 1 KiB sai em KB"
    assert texto_do_tamanho(b"x" * 5000) == "4 KB"
    assert "0 KB" not in texto_do_tamanho(b"x"), "um artefato de 1 byte saiu como `0 KB`"


# **DUAS PROPRIEDADES SAÍRAM DAQUI com o botão de salvar, e nenhuma delas foi afrouxada.**
#
# | O que era medido aqui | Onde passou a viver, e por quê |
# |---|---|
# | *"`salva` sem bytes devolve `None` e não abre diálogo"* | **em lugar nenhum, porque o
# referente sumiu**: `salva` e o `Escritor` saíram com o botão «Baixar X», e não há diálogo que um
# painel sem bytes pudesse abrir. A metade que sobrevive — *"sem bytes, `_escreve_no_destino` não
# escreve"* — é o ramo `else` de `test_d367_a_chegada_sem_destino_ou_sem_bytes_nao_escreve` |
# | *"um `OSError` de escrita não escapa do slot, e o rótulo diz o que aconteceu"* | **continua a
# existir, do outro lado da fronteira**: quem escreve é `JanelaPrincipal._escreve_no_destino`, que
# captura `OSError` e escreve `ARTEFATO_SEM_ESCRITA`. Ver
# `test_d367_uma_falha_de_escrita_relata_no_rotulo_e_nao_perde_os_bytes` |


def test_u6_cancelar_a_pasta_nao_descarta_os_bytes(qtbot: Any) -> None:
    """Cancelar **não** é erro, e não apaga o artefato que já foi pago.

    É o mesmo desenho de `arquivos.py`: *"cancelar não é erro, e um painel que tratasse os dois como
    o mesmo estado exibiria falha onde não houve"*. Aqui a consequência é mais direta — o artefato
    custou uma correção, e descartá-lo obrigaria a refazê-la.

    ## O diálogo que se cancela mudou, e a propriedade é a mesma

    Até 03/09/2026 este teste cancelava o diálogo de **salvar**, que era o segundo gesto. Com um
    botão só, o único diálogo do painel é o da **pasta**, pedido no instante do clique
    (`pede_e_gera`) — e cancelá-lo é exactamente o mesmo pedido: *"não quero isto agora"*. Os bytes
    que já chegaram não são de quem cancelou.

    **E o cancelamento NÃO dispara o trabalho**, que é a metade que impede o gesto de custar uma
    correção por engano. A âncora positiva vem junto: o mesmo painel, com pasta, emite.
    """
    painel = _painel_de_artefatos(qtbot, escolhedor=lambda: None)
    painel.recebe("exportar", 200, b"w" * 100, "ok")
    assert _anunciados(painel) == ("exportar",), "âncora: não havia artefato a preservar"

    pedidos: list[str] = []
    painel.gerar.connect(pedidos.append)
    painel.botoes_de_geracao["exportar"].click()

    assert pedidos == [], "cancelar a pasta disparou o trabalho na mesma"
    assert painel.destinos == {}, "o painel guardou uma pasta que ninguém escolheu"
    assert painel.salvaveis() == ("exportar",), (
        "cancelar o diálogo descartou os bytes: quem cancelou não pediu para perder o artefato"
    )
    assert painel.rotulos["exportar"].text() == SEM_DESTINO, (
        "cancelar não escreveu a frase do cancelamento no rótulo daquele artefato"
    )
    # **A DIVERGÊNCIA VAI AFIRMADA e não deixada em silêncio.** Este é um dos três estados em que
    # as duas superfícies discordam — os bytes continuam vivos e a tela deixou de os anunciar,
    # porque o rótulo é o único slot do artefato e a frase do cancelamento tomou-o. Afirmá-la aqui
    # é o que impede a próxima sessão de ler `_anunciados() == ()` como *"não há bytes"*, e é o
    # gatilho para decidir — noutra unidade — se `SEM_DESTINO` devia mesmo apagar o anúncio.
    assert _anunciados(painel) == (), (
        "a tela continua a anunciar o artefato depois de a frase do cancelamento tomar o rótulo — "
        "se isto passar a ser verdade, a tabela da docstring de `_anunciados` caducou"
    )

    # **A âncora do par causal:** o mesmo gesto, com pasta escolhida, emite e guarda o destino.
    com_pasta = _painel_de_artefatos(qtbot, escolhedor=lambda: Path("destino"))
    com_pasta.gerar.connect(pedidos.append)
    com_pasta.botoes_de_geracao["exportar"].click()
    assert pedidos == ["exportar"], (
        f"âncora: nem com pasta o clique emite, e a ausência acima é vácua: {pedidos}"
    )
    assert com_pasta.destinos == {"exportar": Path("destino")}, (
        f"âncora: nem com pasta o clique GUARDA o destino, e o `destinos == {{}}` acima é vácuo — "
        f"ele passaria com um `pede_e_gera` que nunca escrevesse ali: {com_pasta.destinos}"
    )


def test_u6_a_troca_de_entrada_descarta_os_artefatos_e_o_anuncio(qtbot: Any) -> None:
    """Um artefato de outra entrada não pode ser levado com o nome da entrada nova.

    **A identidade tem TRÊS eixos, e a primeira versão deste painel usava UM.**
    `interface/fluxo.chave_da_entrada` compõe `modo|metodo|digest`, e a docstring dela diz por que:
    *"Ignorar qualquer um deixa a porta aberta para o resultado de uma entrada sair sob outra"*, e
    **"Num contexto clínico, é CSV de paciente com o rótulo errado"**. Trocar o corretor, no mesmo
    modo, não descartava nada.

    **E a asserção é sobre a TELA e não só sobre o conjunto.** O mutante que limpava `_bytes` e
    esquecia o rótulo passava na primeira versão — o próprio docstring dela chamava isto de *"a
    segunda metade da bicondicional"* e omitia justamente o observável dela. Antes esse
    observável era o `setEnabled(False)` do botão de salvar; hoje é o rótulo voltar a
    `SEM_ARTEFATO`.
    """
    painel = _painel_de_artefatos(qtbot, identidade=ENTRADA_A)
    painel.recebe("corrigir", 200, b"a" * 50, "ok")
    assert _anunciados(painel) == ("corrigir",), "âncora: não havia artefato para descartar"

    # A mesma identidade não descarta — senão qualquer chamada limparia a tela.
    painel.define_entrada(ENTRADA_A, "neg_c18")
    assert _anunciados(painel) == ("corrigir",), (
        "redefinir a MESMA entrada descartou o artefato — qualquer chamada limparia a tela"
    )

    # A troca de CORRETOR — mesmo modo, mesmos bytes — descarta.
    assert ENTRADA_B != ENTRADA_A and ENTRADA_B.split("|")[0] == ENTRADA_A.split("|")[0], (
        "âncora: as duas entradas de teste não diferem só no corretor, e o teste mediria "
        "outra coisa"
    )
    painel.define_entrada(ENTRADA_B, "neg_c18")
    assert painel.salvaveis() == (), "a troca de entrada não descartou o artefato anterior"
    assert _anunciados(painel) == (), (
        "o rótulo continuou a anunciar artefato depois da troca de entrada, com zero bytes por "
        "trás — o `[CA-12.14]` violado verbatim"
    )
    assert painel.rotulos["corrigir"].text() == SEM_ARTEFATO


def test_u6_a_identidade_da_entrada_e_a_mesma_das_duas_interfaces() -> None:
    """A identidade é computada por `interface/fluxo.chave_da_entrada`, e é uma só.

    **Duas implementações da mesma identidade divergem sem levantar**: as duas produzem
    chave plausível, e a divergência aparece como artefato com o rótulo errado. Os três eixos têm
    par
    causal aqui, e a âncora é que os três produzem chaves **distintas**.
    """
    base = chave_da_entrada(modo="neg_c18", metodo="none", medidas=b"m", lista=b"l")
    variantes = {
        "modo": chave_da_entrada(modo="pos_hilic", metodo="none", medidas=b"m", lista=b"l"),
        "metodo": chave_da_entrada(modo="neg_c18", metodo="loess", medidas=b"m", lista=b"l"),
        "medidas": chave_da_entrada(modo="neg_c18", metodo="none", medidas=b"M", lista=b"l"),
        "lista": chave_da_entrada(modo="neg_c18", metodo="none", medidas=b"m", lista=b"L"),
        "sem metodo": chave_da_entrada(modo="neg_c18", metodo=None, medidas=b"m", lista=b"l"),
    }
    assert len(set(variantes.values()) | {base}) == len(variantes) + 1, (
        f"dois eixos produziram a mesma chave: {variantes}"
    )
    for eixo, chave in variantes.items():
        assert chave != base, f"a chave ignora o eixo `{eixo}`"


def test_u6_o_nome_sugerido_e_o_mesmo_das_duas_interfaces() -> None:
    """O nome sugerido para salvar um artefato é `f"{modo}_{chave}.csv"`."""
    assert nome_do_arquivo("pos_hilic", "corrigir") == "pos_hilic_corrigir.csv"


def test_u6_o_modulo_nao_escreve_em_disco_por_nome_nenhum() -> None:
    """**Este módulo não escreve em disco** — portão por AST, e o conjunto sancionado é VAZIO.

    É a metade que vive no cliente. A metade do **servidor** — *"nada novo fica em
    caminho durável depois da chamada"* — já tem portão com as duas direções em
    `tests/test_api_artefatos.py`, e repeti-la aqui seria segunda fonte sobre a mesma afirmação.

    ## A PRIMEIRA VERSÃO DEIXAVA ESCAPAR 11 DE 11 PLANTAS, e duas eram naturais

    Ela casava seis nomes de método e permitia por **nome simples** de função. A revisão adversarial
    correu onze plantas contra o portão real e **todas** passaram:

    | Escape | Por que |
    |---|---|
    | `def escreve(...)` NOVA, no topo do módulo | a permissão era por nome simples, e `escreve` é
    o verbo mais natural para um ajudante de escrita |
    | `escreve` como método de outra classe | idem |
    | `tempfile.NamedTemporaryFile(delete=False).write` | `write` e `NamedTemporaryFile` fora do
    conjunto |
    | `shutil.copyfile`, `os.write`, `os.replace`, `zipfile.writestr`, `logging.FileHandler`,
    `to_csv` | fora do conjunto |
    | `w = alvo.write_bytes` e `getattr(alvo, "write_bytes")()` | não são nó de chamada com nome |

    As duas primeiras são as que um desenvolvedor escreveria. **O conserto é permissão por nome
    QUALIFICADO** — era `escritor_por_dialogo.escreve` e nada mais — e um conjunto de nomes muito
    mais largo. Um `escreve` no topo do módulo, ou noutra classe, passa a ser acusado.

    ## O CONJUNTO SANCIONADO ESVAZIOU, e o portão ficou MAIS forte e não mais fraco

    `escritor_por_dialogo` existia só para o botão «Baixar X», e saiu com ele em 03/09/2026. Com o
    conjunto vazio o portão deixa de ter isenção nenhuma: ele passa a afirmar que **nenhum** escopo
    deste módulo chama nenhum dos nomes de escrita — que é a afirmação mais forte que esta forma de
    portão sabe fazer, e a que o módulo de hoje de facto satisfaz.

    > **A alternativa era apagar o portão**, e ela é a errada: o módulo continua a ser o lugar
    > natural para alguém voltar a pôr uma escrita — foi lá que ela viveu até ontem. O que mudou
    > é que hoje quem escreve é `JanelaPrincipal._escreve_no_destino`, do outro lado da fronteira.

    **O registro do corte NÃO é contra-exemplo, e a distinção é a mesma que o módulo declara:**
    `escreve_o_corte_ao_lado` — que `registra_corte_ao_lado` e o lote chamam — converte
    e chama `write_removidas`, que é o escritor **do núcleo** —
    o byte sai por `refino/io/writers.py`, não daqui. É por isso que `write_removidas` não está no
    conjunto de nomes de escrita: acusá-lo obrigaria o painel a formatar o arquivo por conta
    própria, que é exactamente o defeito que `removidas_do_relatorio` existe para não ter.

    **As plantas são em processo, logo não ficam inertes**, e há uma por classe de escape.
    """
    caminho = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "refino"
        / "desktop"
        / "janela"
        / "artefatos.py"
    )
    fonte = caminho.read_text(encoding="utf-8")

    # **VAZIO**, e a docstring diz por quê: o único escopo que já esteve aqui era
    # `escritor_por_dialogo.escreve`, e ele saiu com o botão «Baixar X». A permissão continua a ser
    # por nome QUALIFICADO — `escreve` sozinho é o verbo que qualquer ajudante de escrita levaria,
    # e permitir por nome simples é a heurística que este portão recusa.
    sancionados: frozenset[str] = frozenset()

    escritores = frozenset(
        {
            "open",
            "write",
            "writelines",
            "write_bytes",
            "write_text",
            "writestr",
            "mkdir",
            "touch",
            "rename",
            "replace",
            "copy",
            "copy2",
            "copyfile",
            "copyfileobj",
            "move",
            "savetxt",
            "savez",
            "to_csv",
            "to_parquet",
            "dump",
            "NamedTemporaryFile",
            "mkstemp",
            "FileHandler",
            "ZipFile",
        }
    )

    def escritas(texto: str, isentos: frozenset[str] = sancionados) -> list[str]:
        arvore = ast.parse(texto)
        # O escopo QUALIFICADO de cada nó: `a.b.c` para funções aninhadas.
        escopos: dict[int, str] = {id(arvore): "<modulo>"}

        def desce(no: ast.AST, atual: str) -> None:
            for filho in ast.iter_child_nodes(no):
                if isinstance(filho, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    proximo = filho.name if atual == "<modulo>" else f"{atual}.{filho.name}"
                elif isinstance(filho, ast.Lambda):
                    proximo = f"{atual}.<lambda>"
                else:
                    proximo = atual
                escopos[id(filho)] = proximo
                desce(filho, proximo)

        desce(arvore, "<modulo>")

        achados = []
        for no in ast.walk(arvore):
            escopo = escopos.get(id(no), "<modulo>")
            if escopo in isentos:
                continue
            if isinstance(no, ast.Call):
                alvo = no.func
                chamado = (
                    alvo.id
                    if isinstance(alvo, ast.Name)
                    else alvo.attr
                    if isinstance(alvo, ast.Attribute)
                    else ""
                )
                if chamado in escritores:
                    achados.append(f"{escopo}:{no.lineno} — {ast.unparse(no)[:45]}")
            # A forma sem chamada: o método capturado como valor.
            if isinstance(no, ast.Attribute) and no.attr in escritores:
                pai_e_chamada = any(
                    isinstance(outro, ast.Call) and outro.func is no for outro in ast.walk(arvore)
                )
                if not pai_e_chamada:
                    achados.append(f"{escopo}:{no.lineno} — captura de {no.attr}")
        return sorted(set(achados))

    assert escritas(fonte) == [], (
        f"há escrita em disco neste módulo, e o conjunto sancionado é {sorted(sancionados)}: "
        f"{escritas(fonte)}. Quem escreve o artefato é `JanelaPrincipal._escreve_no_destino`, e "
        f"quem escreve o registro do corte é `write_removidas`, no núcleo"
    )

    plantas = {
        "def escreve nova no módulo": (
            "\n\ndef escreve(alvo: Path, dados: bytes) -> None:\n    alvo.write_bytes(dados)\n"
        ),
        "escreve como método de outra classe": (
            "\n\nclass _Outra:\n    def escreve(self, alvo: Path, d: bytes) -> None:\n"
            "        alvo.write_bytes(d)\n"
        ),
        "NamedTemporaryFile": (
            "\n\ndef _p(d: bytes) -> None:\n    import tempfile\n\n"
            "    f = tempfile.NamedTemporaryFile(delete=False)\n    f.write(d)\n"
        ),
        "shutil.copyfile": (
            "\n\ndef _p(a: str, b: str) -> None:\n    import shutil\n\n    shutil.copyfile(a, b)\n"
        ),
        "os.replace": (
            "\n\ndef _p(a: str, b: str) -> None:\n    import os\n\n    os.replace(a, b)\n"
        ),
        "captura do método": (
            "\n\ndef _p(alvo: Path, d: bytes) -> None:\n    w = alvo.write_bytes\n    w(d)\n"
        ),
        "nível de módulo": '\n\nPath("/tmp/x").write_bytes(b"y")\n',
        "lambda de módulo": "\n\n_g = lambda alvo, d: alvo.write_bytes(d)\n",
    }
    for rotulo, planta in plantas.items():
        plantado = f"{fonte}{planta}"
        assert plantado != fonte, f"a planta `{rotulo}` NÃO aterrou"
        assert escritas(plantado), (
            f"o portão não acusou a planta `{rotulo}`: a asserção da árvore limpa é vácua para ela"
        )

    # ------------------------------------------------------------------
    # A ISENÇÃO POR NOME QUALIFICADO — exercitada, e não só construída
    # ------------------------------------------------------------------
    #
    # **Com `sancionados` vazio, `if escopo in isentos` NUNCA dispara**, e toda a maquinaria de
    # escopo — `escopos`, `desce`, o `f"{atual}.{filho.name}"`, o `<lambda>` — deixa de decidir
    # coisa nenhuma: ela passa a alimentar só a string da mensagem. A docstring continua a afirmar
    # *"a permissão é por nome QUALIFICADO"*, e essa metade ficou **inverificável** no dia em que
    # o `escritor_por_dialogo` saiu.
    #
    # Medido em 03/09/2026: trocar `escopos.get(id(no), "<modulo>")` por `"<modulo>"` fixo deixava
    # este teste **verde**: as oito plantas exercitam a acusação, e nada exercitava a isenção.
    #
    # As três direções, sobre a MESMA planta, com o conjunto passado por parâmetro:
    metodo_plantado = (
        "\n\nclass _Outra:"
        + "\n    def escreve(self, alvo: Path, d: bytes) -> None:"
        + "\n        alvo.write_bytes(d)\n"
    )
    com_metodo = f"{fonte}{metodo_plantado}"
    assert escritas(com_metodo, frozenset()), (
        "âncora: a planta do método não é acusada nem SEM isenção nenhuma, e as duas asserções "
        "abaixo não separariam nada"
    )
    assert escritas(com_metodo, frozenset({"_Outra.escreve"})) == [], (
        "a isenção por nome QUALIFICADO não isenta: a maquinaria de escopo está inerte e o "
        "`if escopo in isentos` nunca casa"
    )
    assert escritas(com_metodo, frozenset({"escreve"})), (
        "a isenção casou por nome SIMPLES — é exactamente a heurística que este portão recusa, e "
        "que deixou escapar 11 de 11 plantas na primeira versão"
    )


# **O TESTE DO `escritor_por_dialogo` CAIU, e o referente sumiu junto.** Ele exercitava as
# duas direções daquela fábrica — o caminho escolhido, e o `""` do cancelamento traduzido para
# `None` —, e a fábrica existia **só** para o botão «Baixar X». Removido o botão, não há função a
# testar. O que ocupa o lugar dele é o teste abaixo: a **outra** fábrica de diálogo deste módulo,
# `pasta_por_dialogo`, que é a única que sobrou e faz exactamente a mesma tradução.


def test_u6_a_fabrica_da_pasta_traduz_o_cancelamento_e_entrega_o_caminho(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As DUAS direções de `pasta_por_dialogo`, em vez de um `pragma`.

    O diálogo é remendado porque um modal do sistema não pode ser fechado por um teste — mas a
    **lógica** em volta dele é testável, e é ela que tem as duas direções:

    | O que o `QFileDialog` devolve | O que a fábrica faz |
    |---|---|
    | um caminho | devolve o `Path` dele |
    | `""` — o cancelamento do Qt | devolve `None` |

    > **String vazia traduzida para `None` é o contrato do `EscolhedorDePasta`.** Deixá-la passar
    > faria `Path("")`, que é o **diretório corrente** — e o desfecho seria escrever CSV de paciente
    > num caminho que ninguém escolheu.

    ## Este teste OCUPA O LUGAR do que caiu com o botão «Baixar X», e o argumento é o mesmo

    Até 03/09/2026 quem afirmava esta propriedade era
    `test_u6_o_escritor_de_producao_escreve_traduz_o_cancelamento_e_entrega_o_nome`, sobre
    `escritor_por_dialogo`. Aquela fábrica saiu com o botão, e a docstring de `pasta_por_dialogo`
    já declarava que o arnês remenda `QFileDialog.getExistingDirectory` e afirma as duas direções,
    sem `pragma` — uma afirmação que **nenhum teste desta árvore cumpria**: medido
    em 03/09/2026, `grep` por `getExistingDirectory` sobre `tests/` devolvia zero. A propriedade
    não deixou de existir; deixou de ter dono, e este teste é o dono.

    **A âncora é o par causal**: as duas direções correm no mesmo teste e produzem valores que
    DIFEREM, então uma fábrica que ignorasse o retorno do diálogo reprova numa delas.
    """
    from PySide6.QtWidgets import QFileDialog

    vistos: list[tuple[Any, ...]] = []

    def dialogo_que_aceita(*args: Any, **kwargs: Any) -> str:
        vistos.append(args)
        return str(tmp_path)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", dialogo_que_aceita)
    escolhida = pasta_por_dialogo()()

    assert escolhida == tmp_path, f"a fábrica não devolveu a pasta escolhida: {escolhida}"
    assert vistos, "âncora: o diálogo não foi chamado"

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    cancelada = pasta_por_dialogo()()

    assert cancelada is None, (
        f"o cancelamento não virou `None`: {cancelada!r}. `Path('')` é o diretório corrente, e "
        f"escrever lá é o defeito"
    )
    assert cancelada != escolhida, (
        "âncora: as duas direções devolveram a mesma coisa, e o teste não separa nada"
    )


def test_u6_o_painel_de_artefatos_esta_composto_na_janela(qtbot: Any, tmp_path: Path) -> None:
    """O painel 6 deixou de ser marca, e o diálogo de PASTA entra por injeção na janela.

    **A âncora é a partição inteira e não uma contagem à mão**: os dois conjuntos — marcas e
    preenchidos — têm de ser disjuntos e cobrir os painéis declarados.

    ## O que este teste afirma sobre a injeção mudou de diálogo

    Ele injetava um `Escritor` e clicava no botão «Baixar X» para provar que a janela usava o duplo
    e não o diálogo do sistema. Aquele botão saiu, e o único diálogo que este painel abre é o da
    **pasta** — pedido no instante do clique em «Gerar e baixar», e injetado por `pasta=`. A
    propriedade é a mesma: *o diálogo que a janela usa é o injetado*, e sem ela um modal do sistema
    penduraria o arnês.

    **O gesto para aqui de propósito.** `pede_e_gera` guarda a pasta e emite; `_gera_artefato` sai
    cedo porque não há etapa feita, e é por isso que nenhuma rota é chamada. O que se mede é a
    chegada da pasta ao painel que a janela construiu, e não o despacho — que tem portão próprio.

    ## Desde 09/11/2026 os downloads moram no Resultado de cada etapa

    O painel de três downloads vivia na aba «Executar e exportar», que saiu. O que este teste
    afirmava dele afirma-se do lugar
    novo da mesma coisa: os dois Resultados são painéis de artefatos, cada um na aba da etapa
    dele, e a pasta injetada é a que os dois usam.
    """
    from refino.desktop.janela.principal import INDICES_DAS_ABAS_DE_ETAPA

    janela = JanelaPrincipal(
        TestClient(build_app(), raise_server_exceptions=False), pasta=lambda: tmp_path
    )
    qtbot.addWidget(janela)

    resultados = {1: janela.resultado_da_correcao, 2: janela.resultado_da_normalizacao}
    for etapa, resultado in resultados.items():
        assert isinstance(resultado, PainelDeArtefatos), type(resultado).__name__
        indice = _aba_que_mostra(janela, resultado)
        assert indice == INDICES_DAS_ABAS_DE_ETAPA[etapa], (
            f"o Resultado da etapa {etapa} está na aba {indice} — "
            f"«{janela.abas.tabText(indice) if indice >= 0 else 'nenhuma'}»"
        )
    assert "artefatos" not in {p.chave for p in PAINEIS}, "o painel da aba que saiu voltou"
    assert set(janela._vazios) | set(janela._preenchidos) == {p.chave for p in PAINEIS}

    # E a pasta injetada é a que a janela usa — não o diálogo do sistema —, nos dois Resultados.
    for resultado in resultados.values():
        chave = resultado.chave_do_formato()
        assert resultado.destinos == {}, (
            "âncora: o painel já nasce com destino, e a igualdade abaixo passaria sem o clique"
        )
        resultado.pede_e_gera(chave)
        assert resultado.destinos == {chave: tmp_path}, (
            f"a janela não usou o escolhedor de pasta injetado: {resultado.destinos}"
        )


@pytest.mark.integration
def test_u6_os_bytes_escritos_sao_identicos_aos_da_resposta_da_rota(
    qtbot: Any, tmp_path: Path, transporte_real: TestClient
) -> None:
    """**Ponta a ponta contra a rota real, em DOIS artefatos, e a asserção é IDENTIDADE de bytes.**

    O caminho é o que a rota de facto tem, medido em 23/08/2026: `POST /matrizes/<passo>` devolve
    **`202`** com identificador, e os bytes vêm de `GET /trabalhos/{id}`, que responde **`200`** com
    `text/csv` — `cliente.envia` lê o corpo pelo `content-type` e preenche `conteudo`.

    > **A terminalidade é o CÓDIGO e nunca um campo do corpo.**

    **DOIS artefatos e não um**: com população de **um** não há proporção a
    ler, e uma escrita que ignorasse a chave passaria. Com dois, a identidade é afirmada por chave.

    **O painel não reprocessa nada**: outro teste já garante identidade de bytes entre a API e a
    CLI, e reprocessar aqui inventaria uma terceira fonte do mesmo arquivo.

    **O piso de tamanho é DERIVADO da entrada.** `assert conteudo` é verdade para dez bytes.
    Medido no `neg_c18` com `method=none`: a entrada tem 2.357.623 bytes e `corrigir` devolve
    2.367.696 — razão **1,004**.

    ## Quem escreve mudou, e é por isso que o arnês é a JANELA e não o painel

    Até 03/09/2026 este teste montava o painel sozinho e clicava no botão «Baixar X». Sem aquele
    botão não há escrita no painel: quem escreve é `JanelaPrincipal._escreve_no_destino`, na pasta
    que o clique escolheu. O arnês passou a ser a janela inteira, com `pasta=` injetada, e a
    propriedade afirmada é a mesma — **os bytes em disco são byte a byte os da resposta da rota**.

    **`chama` é remendado e o resto corre de verdade.** As rotas são exercitadas aqui, em processo,
    para que os bytes sejam os do servidor real; o que se remenda é só o despacho assíncrono, que
    tem portão próprio e traria a piscina de threads para dentro de um teste de identidade.
    """
    modo = "neg_c18"
    janela = JanelaPrincipal(transporte_real, pasta=lambda: tmp_path)
    qtbot.addWidget(janela)
    janela.arquivos._par = _par_real(modo)
    janela.arquivos.modo.setCurrentText(modo)
    janela.chama = lambda *a, **k: None  # type: ignore[method-assign]
    # A Correção dada por feita com `none`, que é o corretor das rotas exercitadas abaixo: desde
    # 09/11/2026 os dois downloads são os do Resultado da Correção, que leva a etapa feita e o
    # corretor dela — e é este que dá nome ao arquivo. Até ali o Palpite era também dado
    # por medido aqui, para a caixa «Automático» não o pedir antes do disparo; a caixa saiu.
    metodo = "none"
    _com_a_correcao_feita(janela, metodo)

    assert janela.arquivos.campos_para_envio()["mode"] == modo, (
        "âncora: o combo da janela não está no modo em que as rotas são exercitadas, e o nome do "
        "arquivo esperado sairia de outro modo"
    )

    medidas, lista = measurements_path(modo), sample_list_path(modo)
    arquivos = {
        "measurements": (medidas.name, medidas.read_bytes(), "text/csv"),
        "sample_list": (lista.name, lista.read_bytes(), "text/csv"),
    }
    entrada = medidas.stat().st_size

    colhidos: dict[str, bytes] = {}
    for chave in ("corrigida", "corrigida_metaboanalyst"):
        passo = ARTEFATOS_DA_JANELA[chave].passo
        disparo = cliente.envia(
            transporte_real,
            passo.metodo,
            passo.rota,
            arquivos=arquivos,
            campos={"mode": modo, "method": "none"},
        )
        assert disparo.codigo == 202, (
            f"{chave}: o disparo deu {disparo.codigo} — {disparo.mensagem}"
        )
        identificador = (disparo.corpo or {})["identificador"]

        for _ in range(600):
            reclamada = cliente.envia(transporte_real, "GET", rota_do_progresso(identificador))
            if reclamada.codigo != EM_CURSO:
                break
        assert reclamada.codigo == 200, f"{chave}: a reclamação deu {reclamada.codigo}"
        assert reclamada.conteudo, f"{chave}: âncora — a reclamação não trouxe bytes"
        assert len(reclamada.conteudo) > entrada // 2, (
            f"{chave}: o artefato tem {len(reclamada.conteudo)} bytes contra {entrada} da "
            f"entrada — a razão medida para `corrigir` no `neg_c18` é 1,004, então isto não é "
            f"o artefato"
        )
        colhidos[chave] = reclamada.conteudo

        # **O GESTO REAL PASSA PELO SLOT, e a versão anterior deste bloco NÃO passava.**
        #
        # Ela chamava `artefatos.recebe(...)` e `janela._escreve_no_destino(...)` à mão, uma a
        # seguir à outra, sob um comentário que dizia *"na ordem real"*. Em produção quem chama as
        # duas é `_encaminha`, a partir do slot `_recebe` — e a diferença foi MEDIDA em 03/09/2026
        # com uma planta: trocar a chamada de `_escreve_no_destino` dentro de `_encaminha` por
        # `pass` deixava **220 passed, EXIT=0**. Nenhum dos 220 ligava a chegada da resposta à
        # escrita, que é a única linha de que o botão único depende.
        #
        # `_recebe` com a chave NUA é caminho de produção e não atalho de arnês:
        # `parte_a_chave("corrigida")` devolve `("corrigida", None)`, e a própria docstring dela
        # diz que `None` significa *"não é leitura encadeada — é o `POST`"*, que é o ramo que
        # entrega.
        janela.resultado_da_correcao.pede_e_gera(chave)
        assert chave in janela._destinos, f"{chave}: o clique não guardou o destino"
        janela._recebe(chave, reclamada)

    assert len(set(colhidos.values())) == 2, (
        "âncora: os dois artefatos têm bytes IDÊNTICOS, então a identidade por chave abaixo não "
        "separa nada"
    )
    # **O Resultado tem um rótulo de estado para as duas chaves**, então o segundo observável da
    # bicondicional é o conjunto de chaves com bytes, e não um rótulo por chave.
    assert janela.resultado_da_correcao.salvaveis() == ("corrigida", "corrigida_metaboanalyst")

    for chave, esperados in colhidos.items():
        alvo = tmp_path / nome_do_arquivo(modo, chave, metodo)
        assert alvo.exists(), (
            f"{chave}: o artefato não foi escrito em {alvo} — o diretório tem "
            f"{sorted(p.name for p in tmp_path.iterdir())}"
        )
        assert alvo.read_bytes() == esperados, (
            f"{chave}: os bytes escritos DIFEREM dos da resposta — {len(alvo.read_bytes())} contra "
            f"{len(esperados)}"
        )


@pytest.mark.integration
def test_u6_o_erro_real_da_rota_nao_anuncia_artefato(
    qtbot: Any, transporte_real: TestClient
) -> None:
    """O desfecho de erro **da rota real**, e não de um duplo.

    Medido: `POST /matrizes/corrigir` sem os campos obrigatórios devolve `422` com causa `usuario` e
    `conteudo=None`. A mensagem é a de `cliente.py`, e o painel não a reescreve.

    **A âncora vem antes:** o mesmo painel anuncia artefato quando a rota responde com bytes — está
    no teste irmão. Aqui a asserção é de ausência, e o que a torna não-vácua é a causa **não** ser
    `None`: uma resposta que não classificou nada seria indistinguível de sucesso.
    """
    painel = _painel_de_artefatos(qtbot)
    modo = "neg_c18"
    medidas, lista = measurements_path(modo), sample_list_path(modo)

    resposta = cliente.envia(
        transporte_real,
        "POST",
        "/matrizes/corrigir",
        arquivos={
            "measurements": (medidas.name, medidas.read_bytes(), "text/csv"),
            "sample_list": (lista.name, lista.read_bytes(), "text/csv"),
        },
        campos={},
    )

    assert resposta.codigo == 422, f"a rota devolveu {resposta.codigo}; o teste mediria outra coisa"
    assert resposta.causa == "usuario", (
        f"âncora: a resposta não classificou a causa ({resposta.causa}), e a ausência abaixo seria "
        f"indistinguível de sucesso"
    )
    assert resposta.conteudo is None

    texto = painel.recebe("corrigir", resposta.codigo, resposta.conteudo, resposta.mensagem)

    assert _anunciados(painel) == ()
    assert painel.salvaveis() == ()
    assert texto == resposta.mensagem, "o painel reescreveu a mensagem de `cliente.py`"


# ---------------------------------------------------------------------------
# O FLUXO LIGADO, contra a rota REAL
# ---------------------------------------------------------------------------


def _par_real(modo: str = "neg_c18") -> ParDeArquivos:
    """O par do LABORATÓRIO, pelos resolvedores do `conftest`.

    ## A primeira versão fabricou o par, e a rota devolveu 500

    Ela construía uma matriz 2x4 à mão *"porque o que se mede aqui é a ligação"*. Medido: aquele par
    faz `/matrizes/validar` devolver **500** com `text/plain` — o defeito conhecido de
    `zip(..., strict=True)` em `io/validators.py:311`, que a docstring de `texto_dos_problemas`
    nomeia. Corpo não-JSON dá `corpo=None`, e o painel escreve o texto inicial: **o teste reprovava
    por defeito do meu dado, e a mensagem apontava para a ligação.**

    > *"Substituir por dado inventado destrói o valor do golden test"*. Aqui não havia golden a
    > destruir — havia uma rota real a exercitar, e dado inventado que ela recusa não exercita nada.

    Os resolvedores fazem `pytest.skip` com mensagem quando o arquivo falta, e é o que
    `tests/test_api_rotas.py` usa para esta rota.
    """
    medidas = measurements_path(modo)
    lista = sample_list_path(modo)
    return ParDeArquivos(medidas.name, medidas.read_bytes(), lista.name, lista.read_bytes())


def test_u11_o_clique_atravessa_ate_a_rota_real_e_a_resposta_volta_ao_painel(
    qtbot: Any, transporte_real: TestClient
) -> None:
    """**O clique atravessa até a rota real, e a resposta volta ao painel.**

    Antes, nenhum botão desta janela disparava rota. O binário congelado
    subia, servia e declarava — e não se operava. Este teste clica e afirma que a resposta da rota
    **real** chega ao painel.

    **O oráculo é o WIDGET**: a asserção é sobre o texto que o rótulo mostra, não sobre o
    valor devolvido por um método. E o texto tem de DIFERIR do estado inicial — sem isso, um painel
    que ignorasse a resposta passaria.
    """
    janela = JanelaPrincipal(transporte_real, escolhedor=lambda: None)
    qtbot.addWidget(janela)
    janela.arquivos._par = _par_real()
    # **O modo tem de ser o do par**, senão a rota avisa `MODE_CONTENT_MISMATCH` — que é um
    # aviso legítimo e não o que este teste mede.
    janela.arquivos.modo.setCurrentText("neg_c18")

    antes = janela.leitura.rotulos["validacao"].text()
    assert antes == SEM_LEITURA, f"o rótulo não repousa no estado inicial: {antes!r}"

    # **Duas esperas e não uma, e as duas foram medidas.**
    #
    # `drena_a_piscina` espera o `QRunnable` **terminar**; o sinal `concluido` é entregue pelo laço
    # de eventos da thread principal. Esperar só a piscina mede *"a chamada acabou"* e afirma *"a
    # resposta chegou ao painel"* — duas coisas, e a primeira versão deste teste reprovou por isso.
    #
    # **Mas `qtbot.waitSignal` também não serve aqui, e a razão é do ARNÊS.** Medido: ele levanta
    # `TimeoutError` **sem gastar os 30 s** — o arquivo inteiro corre em 15. Este arquivo exercita
    # `arranque`, que chama `app.exec()`, e uma `QApplication` que já saiu faz o `QEventLoop` de
    # `waitSignal` devolver na hora. Isolado o teste passa; na suíte reprova. **É a forma de sempre:
    # o mecanismo relatou um desfecho que ele não mediu** — um tempo-limite sem tempo decorrido.
    #
    # `processEvents` não abre laço: entrega a fila. Robusto nas duas ordens.
    #
    # A causa medida é mais estreita que «uma `QApplication` que já saiu»:
    # o fecho da janela deixa um `Quit` postado na fila, e `_corre_a_janela_sem_deixar_o_quit` o
    # tira. `processEvents` fica, porque não depende da limpeza.
    # Import local, como os outros dois sitios deste arquivo o fazem.
    from PySide6.QtWidgets import QApplication

    chaves: list[str] = []
    janela.resposta_recebida.connect(lambda chave, _resposta: chaves.append(chave))
    janela.leitura.botoes["validacao"].click()
    assert janela.drena_a_piscina(30_000), "a piscina não esvaziou em 30 s"
    QApplication.processEvents()

    assert chaves == ["validacao"], (
        f"as respostas entregues foram {chaves}, e o clique foi um, na validação"
    )

    depois = janela.leitura.rotulos["validacao"].text()
    assert depois != antes, (
        f"o rótulo não mudou depois da chamada: {depois!r}. O clique não atravessou, ou a resposta "
        "não chegou ao painel"
    )
    # **A resposta é da ROTA e não de um duplo**, e é o que `aceita`/`recusada` no texto prova: só
    # `/matrizes/validar` produz aquele vocabulário, em `texto_dos_problemas`.
    assert "aceita" in depois or "recusada" in depois, (
        f"o texto não é o de `/matrizes/validar`: {depois!r}"
    )


def test_u11_sem_par_de_arquivos_o_clique_diz_o_que_falta_e_nao_chama(
    qtbot: Any, transporte_real: TestClient
) -> None:
    """O ramo `else` da ligação, e ele dispara por AUSÊNCIA — comece por ele.

    Sem par escolhido, `arquivos_para_envio` levanta `ValueError`, e os quatro slots passam por
    `_pedido_de_arquivos`. **O desfecho tem de ser mensagem e não excepção**: dentro de um slot de
    Qt a excepção é engolida para o `stderr`, que num `--windowed` é `None` — o clique não faria
    nada e não diria nada.
    """
    janela = JanelaPrincipal(transporte_real, escolhedor=lambda: None)
    qtbot.addWidget(janela)
    assert janela.arquivos.par is None, "o painel nasceu com par"

    # **Aqui NÃO se espera sinal, e a ausência é a asserção.** Sem par, nenhuma chamada é feita,
    # então `resposta_recebida` nunca chega — um `waitSignal` esperaria até o timeout e reprovaria
    # por motivo errado. O que se afirma é que a mensagem foi escrita SEM ida ao servidor.
    janela.leitura.botoes["diagnostico"].click()
    assert janela.drena_a_piscina(5_000), (
        "a piscina não esvaziou, e nada devia ter sido enfileirado"
    )

    desfecho = janela.rotulo_de_desfecho.text()
    assert desfecho != SEM_DESFECHO, "o clique sem par não disse nada"
    assert "arquivo" in desfecho.lower(), f"a mensagem não nomeia o que falta: {desfecho!r}"
    # E o rótulo da leitura NÃO mudou: nada foi pedido.
    assert janela.leitura.rotulos["diagnostico"].text() == SEM_LEITURA, (
        "o painel escreveu leitura sem ter chamado rota"
    )


def test_u11_o_eixo_marcado_sem_recomendacao_e_recusado_antes_da_rota(
    qtbot: Any, transporte_real: TestClient
) -> None:
    """`PropostaAusenteError` traduzido em mensagem, e não propagado nem enviado.

    A rota recusa com `422` eixo sobrescrito sem proposta da máquina. Dizê-lo aqui é mais barato que
    uma ida ao servidor — e o que este teste guarda é que a recusa acontece **antes** da chamada:
    sem isso, o desfecho seria um `422` cujo texto o profissional não sabe ler.
    """
    janela = JanelaPrincipal(transporte_real, escolhedor=lambda: None)
    qtbot.addWidget(janela)
    janela.arquivos._par = _par_real()
    # O par entra direto, sem a escolha que deduz o modo: o modo do par vai à mão.
    janela.arquivos.modo.setEditText("neg_c18")
    # O Palpite já foi dado por medido aqui, porque a caixa «Automático» o pediria
    # antes do disparo; a caixa saiu, e a Correção sai no clique.

    # **Faz o eixo DIVERGIR, sem nomeá-lo: a proposta e o valor saem da estrutura.** Com as
    # checkboxes fora, o gesto que exerce a sobrescrita é mudar o valor depois de a máquina
    # ter proposto outro — e é isso que este par de linhas encena.
    assert _faz_divergir(janela.pipeline), "nenhum eixo ficou divergente — o teste seria vácuo"
    assert janela.pipeline.eixos_divergentes(), "a divergência não chegou ao acessor"
    assert janela.leitura.recomendacao_guardada() is None, "há recomendação guardada"

    # Também sem espera de sinal: a recusa acontece ANTES da chamada, que é a propriedade medida.
    # O botão é o da Correção desde 09/11/2026: é ela que leva a linha da máquina do par, e o
    # «Executar pipeline», que também a levava, saiu com a aba «Executar e exportar».
    janela.pipeline.botoes_de_etapa[ETAPAS[0]].click()
    assert janela.drena_a_piscina(5_000), (
        "a piscina não esvaziou, e nada devia ter sido enfileirado"
    )

    desfecho = janela.rotulo_de_desfecho.text()
    assert "proposta" in desfecho.lower(), f"a recusa não nomeia a proposta ausente: {desfecho!r}"


def test_u11_a_sondagem_da_execucao_chama_a_rota_do_trabalho(
    qtbot: Any, transporte_real: TestClient
) -> None:
    """A leitura do progresso chega à rota REAL — e desde 07/09/2026 sem ninguém clicar.

    **Este portão media o clique em «Ler progresso», e o botão saiu.** O que ele fechava —
    *"tirar o `connect` de `ler_progresso` não era acusado por nada"* — passou a ser *"tirar o
    agendamento da sondagem não é acusado por nada"*, e é a mesma fuga com outro emissor.

    **A distância para `test_desktop_sondagem.py` é o transporte:** lá as respostas são fabricadas,
    e o que se mede é a máquina de estados; aqui a chamada sai, atravessa o `TestClient` e volta
    com o `404` que a rota de facto devolve para um identificador que não existe.
    """
    from PySide6.QtWidgets import QApplication

    janela = JanelaPrincipal(transporte_real, escolhedor=lambda: None)
    qtbot.addWidget(janela)

    chaves: list[str] = []
    janela.resposta_recebida.connect(lambda chave, _r: chaves.append(chave))

    # O aceite entra pelo caminho de PRODUÇÃO — o `202` que `POST /execucoes` devolve —, e é ele
    # que guarda o identificador e arma o relógio. Atribuir o identificador à mão saltaria
    # exactamente a aresta que este teste existe para medir.
    janela._encaminha(
        "execucao",
        Resposta(
            codigo=EM_CURSO,
            ok=True,
            causa=None,
            mensagem="",
            corpo={"identificador": "inexistente-de-proposito", "total": 4, "reclamado": False},
            conteudo=None,
        ),
    )
    assert "execucao" in janela.esperas_em_curso(), "âncora: a espera não chegou a abrir"

    # **O TIQUE É DISPARADO À MÃO, e o projeto já tinha escrito por quê.**
    #
    # `test_desktop_sondagem.py` declara: *"um teste que dormisse um segundo para ver a releitura
    # acontecer mediria o `QTimer` do Qt, não a decisão desta janela — e seria lento e
    # intermitente"*. Duas versões deste teste ignoraram a frase e mediram-na: com `qtbot.wait` e
    # depois com `waitUntil`, ele **passava sozinho e reprovava dentro do arquivo**, onde o relógio
    # não entrega dentro de 30 s.
    #
    # `_rele` é o que o relógio chama — não é um atalho para o efeito, é a mesma linha de produção.
    # Que o relógio esteja ARMADO é a outra metade, e está medida em `test_desktop_sondagem.py`
    # com `isActive()`.
    janela._rele("execucao")
    assert janela.drena_a_piscina(30_000), "a piscina não esvaziou"
    QApplication.processEvents()

    # **A chave volta COMPOSTA, e é isso que se afirma.** A sondagem carrega
    # `chave_do_pedido(chave, identificador)` para que a guarda de supersessão saiba de qual
    # cadeia a resposta é; comparar contra `"execucao"` cru mediria a chave do caminho manual, que
    # já não existe.
    assert [parte_a_chave(c)[0] for c in chaves] == ["execucao"], (
        f"o progresso não chegou à rota: {chaves}"
    )
    # **A rota responde 404 para um identificador que não existe, e é o desfecho CERTO de medir:**
    # ele prova que a chamada saiu e voltou. O painel escreve o texto de desconhecido.
    leitura = janela.pipeline.rotulo_de_progresso.text()
    assert leitura != PROGRESSO_EM_REPOUSO, f"o painel não escreveu a leitura: {leitura!r}"


def test_u11_a_resposta_do_artefato_chega_ao_painel_dos_artefatos(
    qtbot: Any, transporte_real: TestClient
) -> None:
    """Fecha a fuga do mutante 6: tirar o roteamento dos artefatos não era acusado por nada.

    **O oráculo é `salvaveis()`**, e é a bicondicional do anúncio: o painel tem os bytes se e
    somente se a operação os produziu. Antes da chamada ele é vazio.

    > **A prosa dizia *"o controle de salvar … fica habilitado"* até 03/09/2026**, e o teste nunca
    > olhou botão nenhum — o botão de salvar saiu e a frase descrevia um observável que não
    > era o deste teste. A asserção não mudou.
    """
    from PySide6.QtWidgets import QApplication

    janela = JanelaPrincipal(transporte_real, escolhedor=lambda: None)
    qtbot.addWidget(janela)
    # **A faixa não se relê durante este teste.** Medido com a suíte da
    # `ui_consumo` a correr ao lado: a geração passou dos dez segundos, a faixa leu o trabalho, e
    # a asserção de chave abaixo viu `trabalhos` — uma resposta legítima de OUTRO mecanismo, que não
    # é o roteamento que este teste mede. A faixa tem teste próprio.
    janela.trabalhos.intervalo_da_releitura_ms = _SEM_RELEITURA_DA_FAIXA_MS
    janela.arquivos._par = _par_real()
    janela.arquivos.modo.setCurrentText("neg_c18")
    # A Correção dada por feita: o assunto é o roteamento do download, e é da Correção feita que
    # ele parte desde que a aba «Executar e exportar» saiu. O Palpite deixou de ser dado
    # por medido aqui: sem a caixa «Automático», nada o espera.
    _com_a_correcao_feita(janela)

    assert janela.resultado_da_correcao.salvaveis() == (), "o painel nasceu com artefato salvável"

    chaves: list[str] = []
    janela.resposta_recebida.connect(lambda chave, _r: chaves.append(chave))

    # **UM disparo, e o laço de espera faz o resto.**
    #
    # **A prosa anterior deste bloco era falsa, e fica registrada em vez de apagada.** Ela dizia
    # *"não há laço de espera — a decisão está declarada em `_encaminha`: quem clica de novo
    # reclama"*, e por isso este teste emitia `gerar` até doze vezes.
    #
    # **Medido sobre o código de `062db19`:** o corpo do `202` do
    # `GET` traz `identificador` obrigatório, então `_encaminha` **já** reencadeava sozinho. A
    # população é um clique, com três `202` forçados, no `neg_c18` com `method=none`:
    # **4 leituras** de `/trabalhos/`, com intervalos de **0,001 s, 0,000 s e 0,000 s**. Com a
    # planta removida, **1 leitura** em 0,20 s.
    #
    # Havia um laço sem intervalo que nenhum teste exercitava: com o `TestClient` o artefato do
    # `neg_c18` fica pronto em 0,20 s, e o ramo do `202` no `GET` não era alcançado. O
    # mecanismo estava verde e um ramo dele nunca corria.
    #
    # **`drena_a_piscina` NÃO serve mais aqui, e é desenho e não acidente:** ele é o caminho de
    # encerramento e cancela as esperas agendadas. Girar o laço com `espera_a_piscina` observa o
    # laço em vez de o matar.
    janela.resultado_da_correcao.gerar.emit("corrigida")
    inicio = time.monotonic()
    while time.monotonic() - inicio < 120:
        janela.espera_a_piscina(50)
        QApplication.processEvents()
        if janela.resultado_da_correcao.salvaveis():
            break

    assert chaves, f"a resposta do artefato não voltou nenhuma vez: {chaves}"
    assert set(chaves) == {"corrigida"}, f"voltou resposta de outra chave: {sorted(set(chaves))}"
    # **A asserção é sobre o painel e não sobre a chave.** Sem o roteamento, a chave chega — o sinal
    # `resposta_recebida` é emitido para toda chamada — e o painel não recebe nada. É a distância
    # entre *"a chamada voltou"* e *"o painel foi alimentado"*, que é o que o mutante 6 explora.
    assert janela.resultado_da_correcao.salvaveis() == ("corrigida",), (
        f"o painel não registrou os bytes depois de {len(chaves)} respostas: "
        f"salvaveis={janela.resultado_da_correcao.salvaveis()}"
    )


def test_u11_a_recomendacao_guardada_entra_no_payload_da_execucao(
    qtbot: Any, transporte_real: TestClient
) -> None:
    """Fecha a fuga do mutante 12: não guardar a recomendação não era acusado por nada.

    A cadeia inteira: pedir a recomendação → o painel a guarda → marcar o eixo → disparar
    produz `proposta_json` no payload. **Sem a guarda, o disparo é recusado com «proposta
    ausente»** — que é o teste vizinho, e é por isso que este afirma o caminho POSITIVO.

    O clique no Palpite roda antes a Validação e o Diagnóstico, de verdade,
    pela API em processo: as respostas entregues passaram de *só `recomendacao`* a *`validacao`,
    `diagnostico` e depois `recomendacao`*, nesta ordem — e é isso que a asserção afirma agora.
    """
    from PySide6.QtWidgets import QApplication

    janela = JanelaPrincipal(transporte_real, escolhedor=lambda: None)
    qtbot.addWidget(janela)
    janela.arquivos._par = _par_real()
    janela.arquivos.modo.setCurrentText("neg_c18")

    assert janela.leitura.recomendacao_guardada() is None, "nasceu com recomendação"

    chaves: list[str] = []
    janela.resposta_recebida.connect(lambda chave, _r: chaves.append(chave))
    janela.leitura.botoes["recomendacao"].click()
    # **O Palpite é rota longa**: o `202` do disparo e as leituras do trabalho voltam
    # todos pela mesma chave, e o fim é o corpo guardado — não o primeiro sinal.
    assert _gira_ate(janela, lambda: janela.leitura.recomendacao_guardada() is not None, 60), (
        "o Palpite não chegou em 60 s"
    )
    assert janela.drena_a_piscina(60_000), "a piscina não esvaziou em 60 s"
    QApplication.processEvents()
    assert chaves[:2] == ["validacao", "diagnostico"], f"a cadeia não veio antes: {chaves}"
    assert chaves[2:] and set(chaves[2:]) == {"recomendacao"}, (
        f"a recomendação não voltou: {chaves}"
    )

    guardada = janela.leitura.recomendacao_guardada()
    assert guardada is not None, "a recomendação voltou e o painel não a guardou"
    assert "correction_proposta" in guardada, (
        f"o corpo guardado não é o da recomendação: {sorted(guardada)}"
    )

    # E agora o eixo DIVERGENTE produz `proposta_json` em vez da recusa.
    assert _faz_divergir(janela.pipeline), "âncora: nenhum eixo divergiu, e o payload seria o vazio"
    campos = campos_da_tela(
        janela.pipeline,
        modo="neg_c18",
        correcao="none",
        n_lotes=1,
        eixos=janela.pipeline.eixos_divergentes(),
        recomendacao=janela.leitura.recomendacao_guardada(),
    )
    assert "proposta_json" in campos, (
        f"o eixo marcado com recomendação guardada não produziu proposta: {sorted(campos)}"
    )
    assert campos["campos_sobrescritos"] != "[]", (
        f"o eixo marcado não entrou em `campos_sobrescritos`: {campos['campos_sobrescritos']!r}"
    )


# ---------------------------------------------------------------------------
# O LAÇO DE ESPERA DO ARTEFATO: intervalo, teto e cancelamento
# ---------------------------------------------------------------------------
#
# **O laço já existia sem intervalo, e a medição é a razão de esta seção existir.** Sonda de
# 23/08/2026, três `202` forçados sobre o `neg_c18`: a janela fez **4 leituras** de `/trabalhos/`
# com intervalos de **0,001 s, 0,000 s e 0,000 s**. A docstring afirmava o contrário e nenhum teste
# a contradizia, porque com o `TestClient` o artefato fica pronto em **0,20 s** e o ramo do `202`
# no `GET` nunca era alcançado.
#
# **QUATRO direções e não duas** — dois é o piso, não o número:
#
# | Direção | O que ela mede |
# |---|---|
# | (a) `200` na primeira leitura | **uma** leitura, e nenhum relógio |
# | (b) `202` e depois `200` | N leituras **com** o intervalo, e a espera encerra |
# | (c) o teto passa | exactamente `leituras_maximas`, e nenhuma depois |
# | (d) o fecho da janela | a espera para, e nenhuma leitura nasce depois |
#
# **O par causal de cada uma é o mesmo dado com a planta removida:** `retidos=0` produz UMA leitura,
# e é o que prova que as leituras extra vêm do `202` forçado e não do arnês.


def _e_reclamacao_de_artefato(url: str) -> bool:
    """`GET /trabalhos/{id}` e **só** ela — não `/trabalhos/{id}/relatorio` nem `/cancelar`.

    **Existe porque o duplo lia o ÚLTIMO segmento da URL como identificador de trabalho**, e
    nasceu uma rota irmã com um segmento a mais. O desfecho foi
    `leituras_por_trabalho()` a devolver `(3, 3, 1)` — um terceiro «trabalho» chamado `relatorio`,
    e o teto do relógio a reprovar por uma população que não é a dele.

    O instrumento lia o referente certo — uma URL de
    `/trabalhos/` — e a **instância** errada. O gesto que a fecha: em vez
    de perguntar se a URL *contém* `/trabalhos/`, contar os segmentos e exigir a forma exata.
    """
    partes = url.split("?", 1)[0].rstrip("/").split("/")
    return len(partes) >= 2 and partes[-2] == "trabalhos"


class _TransporteQueRetem:
    """Envolve o `TestClient` e devolve `202` nas primeiras `retidos` leituras de `/trabalhos`.

    **A planta é o `202`, e ela cai sobre dado em que as duas coisas diferem**: com
    `retidos=0` o mesmo arnês produz **uma** leitura, e com `retidos=3` produz quatro. Sem essa
    diferença medida, um laço que ignorasse o código de resposta passaria.

    O corpo é o `ProgressoResposta` real — `identificador` é obrigatório nele, e é exactamente esse
    campo que fazia o laço original reencadear sozinho.
    """

    def __init__(self, real: TestClient, retidos: int) -> None:
        self._real = real
        self._retidos = retidos

        self.rotas: list[str] = []
        self.instantes: list[float] = []
        # **A retenção é POR TRABALHO e não global, e um mutante escapou por isso.** Com a contagem
        # global, dois artefatos em voo dividiam as retenções: o primeiro consumia-as todas e o
        # segundo recebia desfecho terminal na primeira leitura, **sem nunca usar o relógio**. O
        # mutante do relógio partilhado passava nos doze testes.
        self._vistos: dict[str, int] = {}

    def leituras(self) -> int:
        return sum(1 for rota in self.rotas if "/trabalhos/" in rota)

    def leituras_por_trabalho(self) -> tuple[int, ...]:
        """Quantas leituras cada trabalho recebeu, em ordem de chegada. **A população certa.**"""
        return tuple(self._vistos.values())

    def post(self, url: str, **kwargs: Any) -> Any:
        self.rotas.append(f"POST {url}")
        self.instantes.append(time.monotonic())
        return self._real.post(url, **kwargs)

    def get(self, url: str, **kwargs: Any) -> Any:
        self.rotas.append(f"GET {url}")
        self.instantes.append(time.monotonic())
        if _e_reclamacao_de_artefato(url):
            trabalho = url.rsplit("/", 1)[-1]
            self._vistos[trabalho] = self._vistos.get(trabalho, 0) + 1
            if self._vistos[trabalho] <= self._retidos:
                corpo = {
                    "identificador": trabalho,
                    "estado": "executando",
                    "feito": self._vistos[trabalho],
                    "total": self._retidos + 1,
                    "fracao": self._vistos[trabalho] / (self._retidos + 1),
                    "terminou": False,
                }
                return httpx2.Response(
                    202,
                    content=json.dumps(corpo).encode(),
                    headers={"content-type": "application/json"},
                )
        resposta = self._real.get(url, **kwargs)
        if not _e_reclamacao_de_artefato(url):
            return resposta
        # **Absorve os `202` do PRÓPRIO servidor, e sem isto a contagem é intermitente.** Medido:
        # com o `TestClient` o artefato do `neg_c18` às vezes está pronto na primeira leitura —
        # 0,20 s — e às vezes não, então a direção (a) reprovou com **2** leituras onde esperava
        # uma. O número da janela passava a depender do escalonamento.
        #
        # **As releituras daqui NÃO entram em `self.rotas`**, e é essa omissão que faz a contagem
        # medir o laço da janela em vez do relógio da máquina: a janela vê exactamente `retidos`
        # respostas `202` e depois uma terminal.
        limite = time.monotonic() + 120
        while resposta.status_code == EM_CURSO and time.monotonic() < limite:
            time.sleep(0.05)
            resposta = self._real.get(url, **kwargs)
        return resposta


_SEM_RELEITURA_DA_FAIXA_MS: Final = 3_600_000
"""Uma hora: mais que qualquer teste deste arquivo. Ver `_janela_que_retem`."""


def _janela_que_retem(
    qtbot: Any, *, retidos: int, **kw: Any
) -> tuple[JanelaPrincipal, _TransporteQueRetem]:
    """A janela com o par real do `neg_c18` e o transporte que retém. **`method` é o padrão.**"""
    transporte = _TransporteQueRetem(
        TestClient(build_app(), raise_server_exceptions=False), retidos
    )
    janela = JanelaPrincipal(transporte, **kw)
    qtbot.addWidget(janela)
    # **A releitura automática da faixa sai deste arnês.** Ela lê
    # `GET /trabalhos/{id}` a cada `INTERVALO_DA_RELEITURA_MS` enquanto há trabalho sem desfecho,
    # e `leituras()` conta TODO GET com `/trabalhos/`. Os testes deste arnês contam o laço da ESPERA
    # do artefato; sob carga uma espera passa dos dez segundos, e a faixa somaria leituras que não
    # são do mecanismo medido. A faixa tem teste próprio, em
    # `tests/test_desktop_execucao_em_curso.py`.
    janela.trabalhos.intervalo_da_releitura_ms = _SEM_RELEITURA_DA_FAIXA_MS
    janela.arquivos._par = _par_real()
    janela.arquivos.modo.setCurrentText("neg_c18")
    # Este arnês conta o laço da espera do ARTEFATO, e já deu o Palpite por medido
    # aqui: com a caixa «Automático» ticada o `corrigir` esperaria primeiro o Palpite, cuja rota
    # somaria leituras de `/trabalhos/` que não são do mecanismo medido. A caixa saiu.
    # **A Correção dada por feita, sem rota — 09/11/2026.** O artefato cuja espera este arnês conta
    # é o download do Resultado da Correção, que só sai com a etapa feita; correr a etapa de verdade
    # somaria a leitura dela às do artefato.
    _com_a_correcao_feita(janela)
    return janela, transporte


def _gira_ate(janela: JanelaPrincipal, condicao: Any, segundos: float) -> bool:
    """Gira o laço de eventos ATÉ a condição, e devolve se ela ocorreu.

    **Gira antes de testar, e a ordem custou dois números falsos.** A primeira versão desta sonda
    testava a condição no topo do laço, e `not esperas_em_curso()` é verdade **antes** de o `POST`
    voltar — então ela saía na primeira volta e imprimia zero leituras como se fosse desfecho. Sexta
    instância de *"o arnês relatou um desfecho que ele não mediu"*, e no próprio arnês.
    """
    from PySide6.QtWidgets import QApplication

    inicio = time.monotonic()
    while time.monotonic() - inicio < segundos:
        janela.espera_a_piscina(20)
        QApplication.processEvents()
        if condicao():
            return True
    return False


def test_u13_leituras_maximas_e_derivada_do_intervalo_e_do_teto(qtbot: Any) -> None:
    """**Derivada e não uma terceira constante.** Um `LEITURAS_MAXIMAS` divergiria dos outros dois.

    O piso de **um** existe porque intervalo zero é legítimo num teste — é o que faz o relógio
    disparar no giro seguinte — e a divisão levantaria.
    """
    padrao = JanelaPrincipal(_TransporteQueLevanta(RuntimeError("não usado")))
    qtbot.addWidget(padrao)
    # **O PADRÃO passou a ser «sem teto»**, e a derivação continua a existir
    # para quem põe um. `None` propaga em vez de virar número grande: um teto de 10**9 leituras
    # seria *"nunca desiste"* escrito de forma que alguém teria de calcular para descobrir.
    assert TETO_DA_ESPERA_MS is None, (
        f"o teto voltou a ser um número ({TETO_DA_ESPERA_MS}) — se isso for deliberado, este "
        f"portão e a docstring de `leituras_maximas` mudam juntos"
    )
    assert padrao.leituras_maximas is None, (
        f"a janela padrão desiste depois de {padrao.leituras_maximas} leituras. O teto foi "
        f"removido: ele parava de PERGUNTAR sobre um trabalho vivo, e a tela lia-se como falha"
    )

    apertada = JanelaPrincipal(
        _TransporteQueLevanta(RuntimeError("não usado")),
        intervalo_da_espera_ms=50,
        teto_da_espera_ms=150,
    )
    qtbot.addWidget(apertada)
    assert apertada.leituras_maximas == 3

    # **Intervalo zero é DECLARADO e não escondido atrás de um `>= 1`.** Com ele o piso da
    # divisão entra e o teto passa a ser `teto_ms` inteiro em leituras — 300.000 com os padrões.
    # Isso não limita parede nenhuma, e é aceitável porque só um teste passa zero: é o valor que
    # faz o relógio disparar no giro seguinte. **Afirmar o número em vez de `>= 1` é o que impede
    # a próxima sessão de o descobrir por acidente** (achado 8 da revisão adversarial).
    sem_intervalo = JanelaPrincipal(
        _TransporteQueLevanta(RuntimeError("não usado")),
        intervalo_da_espera_ms=0,
        teto_da_espera_ms=1_000,
    )
    qtbot.addWidget(sem_intervalo)
    assert sem_intervalo.leituras_maximas == 1_000, (
        f"o piso da divisão mudou: {sem_intervalo.leituras_maximas}"
    )


# **O maior custo MEDIDO de `serrf`, e é dele que o teto da janela é derivado.**
#
# 1.209,2 s no `pos_hilic` — o maior dos quatro modos, 6.035 features — no regime de paralelismo do
# executável congelado (`joblib` em threads sob `sys.frozen`), medido **fora** do binário.
# Procedência: `scripts/mede_serrf_congelado.ps1`, cabeçalho, e a auditoria de 30/08/2026. O número
# irmão é **451 s** no `neg_c18` DENTRO do binário.
PIOR_SERRF_MEDIDO_S = 1209.2

# **O piso de margem que o portão defende, e ele é MENOR que a margem escolhida — 2,98x.**
# O portão não pina a escolha: ele impede que o teto caia abaixo do dobro do pior custo medido, que
# é a faixa em que a desistência volta a acontecer no caminho que a aba 2 recomenda.
PISO_DA_MARGEM = 2

# **A pior requisição ISOLADA do caminho da janela, medida em 30/08/2026.**
#
# `POST /matrizes/diagnosticar` sobre o `pos_hilic` (6.082.501 bytes), em processo pelo
# `TestClient`. As outras: `POST /matrizes/corrigir` com `method=serrf` 0,219 s — ela devolve `202`
# e põe o trabalho numa thread —, `validar` 0,218 s, `recomendar` 0,656 s, e 44 leituras de
# `GET /trabalhos/{id}` entre 0,000 e 0,079 s com o `serrf` a correr.
PIOR_REQUISICAO_MEDIDA_S = 0.735


def _timeouts_declarados_no_cliente() -> list[float]:
    """Os `timeout=` de `interface/cliente.py`, por AST. **Extraídos e não redigitados.**"""
    fonte = Path(__file__).resolve().parents[1] / "src" / "refino" / "interface" / "cliente.py"
    arvore = ast.parse(fonte.read_text(encoding="utf-8"))
    return [
        float(palavra.value.value)
        for no in ast.walk(arvore)
        if isinstance(no, ast.Call)
        for palavra in no.keywords
        if palavra.arg == "timeout"
        and isinstance(palavra.value, ast.Constant)
        and isinstance(palavra.value.value, (int, float))
    ]


def test_u13_a_janela_nao_desiste_de_esperar_e_o_custo_medido_e_o_argumento() -> None:
    """A propriedade que SUBSTITUIU o teto, e o número que mostra por que ele não servia.

    ## Este portão trocou de referente, e a troca vai declarada em vez de silenciosa

    Ele afirmava *"o teto cobre o pior `serrf` medido"* — e o teto **deixou de existir**.
    Sobre uma janela sem teto essa asserção não é falsa: ela fica **sem
    sujeito**. O que ela pede não é corrigir a
    comparação, é dar-lhe o referente novo.

    **O referente novo é a propriedade que o teto tentava aproximar**, e é mais forte que ela: a
    janela não desiste. O teto era uma aposta sobre quanto o pior caso custa, e a história dele é
    a de a aposta ser perdida duas vezes — 300 s dava **5** desistências no `pos_hilic`, e 3.600 s
    não cobria a medição de corretor, que o usuário viu parar aos **3.659 s** com o trabalho vivo.

    **O número FICA no portão, e não como comparação — como argumento.** `PIOR_SERRF_MEDIDO_S` é
    o que torna *"não desiste"* uma decisão e não uma preguiça: qualquer teto plausível teria de
    ser maior que ele, e o custo de o errar para baixo é uma tela a dizer *"parei de esperar"*
    sobre um trabalho que continua a correr.
    """
    assert PIOR_SERRF_MEDIDO_S > 0 and PISO_DA_MARGEM >= 1, "âncora: o piso não é comparação vácua"

    assert TETO_DA_ESPERA_MS is None, (
        f"o teto voltou ({TETO_DA_ESPERA_MS} ms). Para ele valer teria de cobrir "
        f"{PIOR_SERRF_MEDIDO_S * PISO_DA_MARGEM} s — e o teto anterior, de 3.600 s, já foi visto "
        f"desistir aos 3.659 s de uma medição de corretor que continuava a correr"
    )

    # **O par causal é o teto de ONTEM contra o custo de hoje**, e ele fica porque é o que mostra
    # que a remoção não é comodidade: 300 s foi um teto real deste projeto, e ele reprovaria.
    teto_de_ontem_s = 300_000 / 1000
    assert teto_de_ontem_s < PIOR_SERRF_MEDIDO_S * PISO_DA_MARGEM, (
        f"âncora: o teto de ontem ({teto_de_ontem_s} s) cobriria o pior `serrf` medido, e então "
        f"este portão não separa o valor que causou o defeito do que o conserta"
    )


def test_u13_o_timeout_do_cliente_e_por_requisicao_e_nao_governa_a_espera() -> None:
    """O `timeout` do cliente cobre a pior requisição ISOLADA, e é só isso que ele mede.

    **Ele NÃO subiu quando o teto da janela subiu 12x, e a decisão é medição.** O `httpx` conta
    este valor contra cada requisição; a espera da janela são N requisições curtas. Subi-lo *por
    precaução* esconderia a única coisa que ele ainda mede — uma requisição que de facto travou.

    **Medido em 30/08/2026, `pos_hilic`:** a pior requisição isolada do caminho é 0,735 s, contra
    300 s aqui — **408x** de folga. O portão defende um piso de 10x sobre o medido, que é escolha
    reversível: o que ele impede é alguém *apertar* o `timeout` para um valor abaixo do que
    o caminho custa, não fixar o valor de hoje.

    **O número sai de `cliente.py` por AST e não é redigitado**, que é a diferença entre portão e
    segunda fonte de verdade.
    """
    piso_de_folga = 10
    tempos = _timeouts_declarados_no_cliente()
    assert tempos, (
        "âncora: não se achou `timeout=` nenhum em `cliente.py`, então a comparação abaixo é vácua"
    )
    exigido_s = PIOR_REQUISICAO_MEDIDA_S * piso_de_folga
    assert min(tempos) >= exigido_s, (
        f"o `timeout` do cliente é {min(tempos)} s e a pior requisição medida é "
        f"{PIOR_REQUISICAO_MEDIDA_S} s: abaixo de {exigido_s} s uma chamada normal do caminho "
        "passa a reportar `503:demora`"
    )
    # **Âncora da não-vacuidade, e ela é o registro da inversão.** Até 30/08/2026 o portão irmão
    # exigia `timeout >= teto da janela`; depois o `timeout` passou a ser MENOR que o teto, e desde
    # então **não há teto nenhum** — que é o desacoplamento máximo dos dois números.
    #
    # **A âncora seguiu o referente em vez de sumir com ele.** Enquanto o teto for `None`, o que
    # ela afirma é que ele continua a não existir; se alguém o repuser, a desigualdade volta a ser
    # a comparação de antes. Apagar a âncora deixaria o portão a medir só uma folga.
    assert TETO_DA_ESPERA_MS is None or min(tempos) < TETO_DA_ESPERA_MS / 1000, (
        f"o `timeout` do cliente ({min(tempos)} s) voltou a cobrir o teto da janela "
        f"({TETO_DA_ESPERA_MS / 1000} s): os dois foram reacoplados, e o acoplamento é o defeito"
    )


@pytest.mark.integration
def test_u13_a_uma_leitura_basta_quando_o_trabalho_ja_terminou(qtbot: Any) -> None:
    """DIREÇÃO (a), e é ela o par causal das outras três.

    Com `retidos=0` o artefato do `neg_c18` está pronto na primeira leitura — 0,20 s medidos —, e a
    janela faz **uma** e para. É este número que torna as quatro leituras da direção (b) atribuíveis
    à planta e não ao arnês.
    """
    janela, transporte = _janela_que_retem(qtbot, retidos=0)
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert _gira_ate(janela, lambda: bool(janela.resultado_da_correcao.salvaveis()), 120), (
        "os bytes não chegaram em 120 s"
    )
    assert transporte.leituras() == 1, (
        f"a janela leu `/trabalhos/` {transporte.leituras()} vezes para um trabalho que já tinha "
        f"terminado: {transporte.rotas}"
    )
    assert janela.esperas_em_curso() == (), (
        f"a espera ficou aberta depois do desfecho terminal: {janela.esperas_em_curso()}"
    )


@pytest.mark.integration
def test_u13_b_o_202_no_get_e_relido_com_intervalo(qtbot: Any) -> None:
    """DIREÇÃO (b): três `202` forçados produzem quatro leituras, e o INTERVALO é respeitado.

    **A asserção de tempo é sobre o mínimo e não sobre o valor**, porque o laço de eventos não dá
    garantia superior. Medido com intervalo de 300 ms: `0,154` até a primeira leitura — imediata,
    por desenho — e depois `0,296`, `0,300`, `0,300`.

    **A primeira leitura é imediata de propósito**, e o número diz por quê: o caminho rápido custa
    0,20 s, e fazê-lo esperar o intervalo tornaria-o duas vezes mais lento sem proteger nada.
    """
    intervalo_ms = 300
    janela, transporte = _janela_que_retem(qtbot, retidos=3, intervalo_da_espera_ms=intervalo_ms)
    inicio = time.monotonic()
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert _gira_ate(janela, lambda: bool(janela.resultado_da_correcao.salvaveis()), 120), (
        f"os bytes não chegaram em 120 s; leituras={transporte.leituras()}"
    )
    decorrido = time.monotonic() - inicio

    assert transporte.leituras() == 4, (
        f"três `202` forçados deviam dar quatro leituras e deram {transporte.leituras()}: "
        f"{transporte.rotas}"
    )
    # **É esta a asserção que separa laço estrangulado de laço em rajada.** O original media
    # 0,001 s entre leituras, e passaria em tudo acima menos aqui.
    assert decorrido >= 3 * intervalo_ms / 1000, (
        f"as quatro leituras couberam em {decorrido:.3f} s, e três intervalos de {intervalo_ms} ms "
        f"exigem {3 * intervalo_ms / 1000} s: o laço não esperou"
    )
    assert janela.esperas_em_curso() == (), "a espera ficou aberta depois dos bytes"


@pytest.mark.integration
def test_u13_c_o_teto_encerra_a_espera_e_o_rotulo_diz_o_que_fazer(qtbot: Any) -> None:
    """DIREÇÃO (c): o teto passa, a espera encerra, e NENHUMA leitura nasce depois.

    A segunda metade é a que importa: um teto que escrevesse a frase e deixasse o relógio a correr
    seria indistinguível desta implementação no instante da medição, e diferente um segundo depois.
    """
    janela, transporte = _janela_que_retem(
        qtbot, retidos=99, intervalo_da_espera_ms=50, teto_da_espera_ms=150
    )
    assert janela.leituras_maximas == 3
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert _gira_ate(janela, lambda: bool(janela.esperas_em_curso()), 60), "a espera não nasceu"
    assert _gira_ate(janela, lambda: not janela.esperas_em_curso(), 60), (
        f"a espera não encerrou; leituras={transporte.leituras()}"
    )

    assert transporte.leituras() == janela.leituras_maximas, (
        f"o teto é {janela.leituras_maximas} leituras e houve {transporte.leituras()}"
    )
    rotulo = janela.resultado_da_correcao.rotulos["corrigida"].text()
    assert "3 leituras" in rotulo and "CONTINUA a correr" in rotulo, (
        f"o rótulo não diz o que aconteceu nem que o trabalho segue vivo: {rotulo!r}"
    )
    # **`0 s` seria a frase a contradizer-se, e foi o que a primeira versão escreveu.** Com
    # intervalo de 50 ms o decorrido é 0,1 s, e `:.0f` arredondava para zero.
    assert " 0 s" not in rotulo, f"o rótulo afirma zero segundos de espera: {rotulo!r}"

    depois_do_teto = transporte.leituras()
    _gira_ate(janela, lambda: False, 0.6)
    assert transporte.leituras() == depois_do_teto, (
        f"nasceram {transporte.leituras() - depois_do_teto} leituras depois de a espera encerrar: "
        "o relógio continua a correr"
    )


@pytest.mark.integration
def test_u13_d_o_fecho_da_janela_para_a_espera(qtbot: Any) -> None:
    """DIREÇÃO (d): `drena_a_piscina` — o que `arranque` chama no fecho — encerra as esperas.

    **O par causal está dentro do teste:** antes do dreno as leituras crescem, depois dele não
    cresce nenhuma. Sem a primeira metade, um dreno que não fizesse nada passaria — bastaria que o
    trabalho tivesse terminado antes.
    """
    janela, transporte = _janela_que_retem(qtbot, retidos=99, intervalo_da_espera_ms=100)
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert _gira_ate(janela, lambda: transporte.leituras() >= 3, 60), (
        f"o laço não chegou a três leituras: {transporte.leituras()}"
    )
    assert janela.esperas_em_curso() == ("corrigida",), (
        f"a espera não estava aberta no instante do fecho: {janela.esperas_em_curso()}"
    )

    assert janela.drena_a_piscina(30_000), "a piscina não esvaziou"
    antes = transporte.leituras()
    assert janela.esperas_em_curso() == (), (
        f"o dreno não encerrou a espera: {janela.esperas_em_curso()}"
    )
    _gira_ate(janela, lambda: False, 0.8)
    assert transporte.leituras() == antes, (
        f"nasceram {transporte.leituras() - antes} leituras depois do fecho"
    )


@pytest.mark.integration
def test_u13_um_relogio_por_chave_e_nao_um_partilhado(qtbot: Any) -> None:
    """Dois artefatos em voo ao mesmo tempo, e o mutante é o relógio ÚNICO.

    Um relógio partilhado seria reagendado pela resposta do segundo artefato, e a leitura do
    primeiro nunca aconteceria — ele ficaria à espera para sempre. **O oráculo é «os dois recebem
    bytes»**, e não a contagem de relógios: contar relógios pinaria a implementação.

    **A âncora é a coexistência.** Se os dois não estivessem à espera ao mesmo tempo, a conclusão
    seria sobre execução sequencial e o mutante do relógio único passaria.
    """
    janela, transporte = _janela_que_retem(qtbot, retidos=2, intervalo_da_espera_ms=80)
    janela.resultado_da_correcao.gerar.emit("corrigida")
    janela.resultado_da_correcao.gerar.emit("corrigida_metaboanalyst")
    assert _gira_ate(janela, lambda: len(janela.esperas_em_curso()) == 2, 60), (
        f"as duas esperas não coexistiram: {janela.esperas_em_curso()}"
    )
    assert _gira_ate(janela, lambda: len(janela.resultado_da_correcao.salvaveis()) == 2, 120), (
        f"só {janela.resultado_da_correcao.salvaveis()} recebeu bytes; esperas abertas: "
        f"{janela.esperas_em_curso()}; leituras por trabalho: "
        f"{transporte.leituras_por_trabalho()}"
    )
    assert janela.esperas_em_curso() == (), "sobrou espera aberta depois dos dois desfechos"

    # **A ÂNCORA VERDADEIRA, e a primeira versão deste teste tinha a errada.** *"As duas esperas
    # coexistiram"* é satisfeito pelos dois `POST` sozinhos — a primeira leitura de cada um é
    # imediata —, e o mutante do relógio partilhado passava nos doze testes. O que separa é **cada
    # trabalho ter precisado de mais de uma leitura**: só a segunda em diante vem do relógio.
    por_trabalho = transporte.leituras_por_trabalho()
    assert len(por_trabalho) == 2, f"não houve dois trabalhos: {por_trabalho}"
    assert min(por_trabalho) > 1, (
        f"algum trabalho terminou na primeira leitura ({por_trabalho}), então o relógio não foi "
        "usado para ele e este teste não separa relógio partilhado de relógio por chave"
    )


def test_u13_para_a_espera_devolve_falso_quando_nao_havia_espera(janela: JanelaPrincipal) -> None:
    """A direção do «deixa de acusar»: sem espera, o encerramento devolve `False` e não levanta.

    **Sem esta, `para_a_espera` podia devolver `True` sempre** e os testes acima passariam todos —
    nenhum deles lê o valor de retorno.
    """
    assert janela.esperas_em_curso() == ()
    assert janela.para_a_espera("corrigir") is False
    assert janela.para_as_esperas() == ()


@pytest.mark.integration
def test_u13_para_a_espera_devolve_verdadeiro_quando_havia_espera(qtbot: Any) -> None:
    """A OUTRA direção, e a revisão adversarial mediu que ela faltava.

    **Um `para_a_espera` que devolvesse `False` sempre passava nos doze testes** — um mutante da
    revisão, `127 passed`. O teste irmão afirma a direção vazia e o seu próprio docstring dizia
    *"sem esta, `para_a_espera` podia devolver `True` sempre"*: o eixo oposto ficou aberto.
    """
    janela, _ = _janela_que_retem(qtbot, retidos=99, intervalo_da_espera_ms=200)
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert _gira_ate(janela, lambda: bool(janela.esperas_em_curso()), 60), "a espera não nasceu"

    assert janela.para_a_espera("corrigida") is True
    assert janela.esperas_em_curso() == (), "devolveu `True` e a espera ficou"
    # E a segunda chamada é a direção do «deixa de acusar», na mesma função.
    assert janela.para_a_espera("corrigida") is False


def test_u13_o_relogio_sobre_espera_encerrada_nao_chama_rota(qtbot: Any) -> None:
    """`_rele` sobre chave sem espera é **ausência e não falha**, e não toca o transporte.

    `stop()` e a remoção do dicionário não são atómicos com um `timeout` já na fila de eventos, e
    levantar dentro de um slot de Qt engole a excepção para um `stderr` que num `--windowed` não
    existe.

    ## ESTE TESTE ESTAVA INERTE, e o que o achou foi um mutante

    **A primeira versão lia `len(transporte.rotas)` no mesmo tique da chamada.** `chama`
    **enfileira** na `QThreadPool` e devolve já — então a contagem não tinha mudado nem no
    caminho defeituoso. A
    revisão adversarial plantou um `_rele` que chama a rota com `espera is None`, e a suíte inteira
    do arquivo passou: **127 passed**.

    Dois consertos, e os dois são necessários:

    1. **drenar antes de ler.** Medido com a planta: `imediato=0` e `depois_do_dreno=1`.
       Sem o dreno, o par causal não separa nada.
    2. **a âncora de não-vacuidade.** `antes` era **0**, então a igualdade passava com um
       transporte que nunca fosse chamado por razão nenhuma. Aqui a âncora é positiva: o
       contador **cresce** num caminho conhecido, e só depois se afirma que `_rele` não o
       move.
    """
    from PySide6.QtWidgets import QApplication

    janela, transporte = _janela_que_retem(qtbot, retidos=0)

    # **A ÂNCORA, e ela é positiva.** Uma chamada por um caminho que sabidamente dispara rota,
    # drenada, prova que este arnês **consegue** ver o contador crescer.
    janela.chama("GET", rota_do_progresso("ancora-positiva"), chave="sonda")
    janela.espera_a_piscina(30_000)
    QApplication.processEvents()
    antes = len(transporte.rotas)
    assert antes > 0, (
        "âncora: o transporte não registrou a chamada de sonda, então a igualdade abaixo passaria "
        "com um arnês que nunca vê chamada nenhuma"
    )

    assert janela.esperas_em_curso() == (), "há espera aberta, e este teste mede a ausência dela"
    janela._rele("corrigida")
    janela.espera_a_piscina(30_000)
    QApplication.processEvents()
    assert len(transporte.rotas) == antes, (
        f"o relógio chamou rota sem espera aberta: {transporte.rotas[antes:]}"
    )


def test_u13_o_rotulo_da_espera_distingue_contagem_ausente_de_zero_passos(
    qtbot: Any, tmp_path: Path
) -> None:
    """`total=0` é contagem AUSENTE — o `202` do `POST` não traz `feito` — e não zero passos.

    E a guarda de chave devolve `""` sem escrever nada, pela mesma razão que `recebe` a tem: este
    painel recebe chamadas de chaves que não são dele.
    """
    painel = _painel_de_artefatos(qtbot, modo="neg_c18", identidade=ENTRADA_A)

    com_contagem = painel.mostra_espera("corrigir", feito=1, total=4, leituras=2)
    assert "1 de 4" in com_contagem and "leitura 2" in com_contagem, com_contagem

    sem_contagem = painel.mostra_espera("corrigir", feito=0, total=0, leituras=1)
    assert "sem contagem de passos" in sem_contagem, sem_contagem
    assert "0 de 0" not in sem_contagem, (
        f"o rótulo afirma zero de zero passos, que é uma medida e não uma ausência: "
        f"{sem_contagem!r}"
    )

    assert painel.mostra_espera("validacao", feito=1, total=2, leituras=1) == ""
    assert painel.mostra_desistencia("validacao", leituras=3, segundos=1.0) == ""


def test_u13_a_desistencia_nao_arredonda_a_espera_para_zero(qtbot: Any, tmp_path: Path) -> None:
    """`:g` e não `:.0f`, e a sonda mediu por quê: com intervalo de 50 ms saía *"em 0 s"*.

    **A frase que se contradiz é pior que a frase vaga**, porque quem a lê conclui que o teto é
    zero e não que a unidade é pequena.
    """
    painel = _painel_de_artefatos(qtbot, modo="neg_c18", identidade=ENTRADA_A)

    curta = painel.mostra_desistencia("corrigir", leituras=3, segundos=0.1)
    assert "0.1 s" in curta, curta
    assert " 0 s" not in curta, f"a espera de 0,1 s foi escrita como zero: {curta!r}"

    longa = painel.mostra_desistencia("corrigir", leituras=150, segundos=298.0)
    assert "4 min 58 s" in longa and "150 leituras" in longa, longa
    # **`298 s` era a asserção até 30/08/2026, e ela caiu com o `:.3g`.** Com o teto em 3.600 s
    # aquele formato entrava em notação científica — `f"{3601.4:.3g}"` dá `3.6e+03` —, e o rótulo
    # diria a um profissional de metabolômica *"depois de 1800 leituras e 3.6e+03 s"*.
    assert "e+0" not in longa, f"o decorrido saiu em notação científica: {longa!r}"
    # **A frase diz o que fazer, e o que ela diz é verdade da rota:** o mesmo trabalho é reclamado.
    # **E o botão que ela manda clicar é o que existe**, lido dele: a frase dizia «Gerar», e
    # os botões deste painel dizem «Gerar e baixar …» (`rotulo_unificado`).
    nome = painel.botoes_de_geracao["corrigir"].text()
    assert f"«{nome}»" in longa and "não recomeça" in longa, longa


def test_u13_a_desistencia_diz_que_a_espera_e_esperada(qtbot: Any, tmp_path: Path) -> None:
    """As três coisas que a frase tem de dizer, e nenhuma delas estava lá — 30/08/2026.

    ## O defeito era de registro de voz, e cada metade da frase antiga era verdadeira

    Ela abria com *"o trabalho não terminou em X s, e a janela parou de reler"* e fechava mandando
    clicar outra vez. **Lê como falha:** nada nela dizia que a demora é o comportamento esperado do
    corretor que a aba 2 recomenda, e este é o primeiro contato do profissional com a ferramenta.

    1. **o trabalho continua a correr no servidor** — `JanelaPrincipal.para_a_espera` mexe em
       `_esperas` e num `QTimer`, e não fala com a API.
    2. **`serrf` custa dezenas de minutos, e isso é normal** — 1.209,2 s no `pos_hilic`, e 451 s no
       `neg_c18` dentro do congelado.
    3. **clicar outra vez não recomeça o cálculo** — `api/progresso.py::abre_ou_reclama`.

    **A âncora contra vacuidade é a frase de ontem**, e ela cai sobre dado em que as duas diferem:
    o texto de hoje satisfaz as três cláusulas e o de ontem falha as três primeiras.
    """
    painel = _painel_de_artefatos(qtbot, modo="neg_c18", identidade=ENTRADA_A)
    texto = painel.mostra_desistencia("corrigir", leituras=1800, segundos=3601.4)

    assert "CONTINUA a correr no servidor" in texto, f"(i) não diz que o trabalho segue: {texto!r}"
    assert "serrf" in texto and "normais" in texto, f"(ii) não diz que a demora é normal: {texto!r}"
    assert "não recomeça o cálculo" in texto, f"(iii) não diz que reclicar não recomeça: {texto!r}"

    # **As três de presença NÃO apanham a reintrodução do defeito, e isso foi MEDIDO.** A revisão
    # adversarial plantou só a abertura antiga — `"o trabalho não terminou depois de …"` — deixando
    # as três cláusulas novas intactas, e a suíte deu `8 passed`: a frase voltava a abrir com o
    # registro de falha que esta unidade existe para tirar, e nada acusava. Presença de cláusula não
    # é ausência de registro de falha; são dois eixos, e o portão só tinha um.
    for proibido in ("não terminou", "parou de reler"):
        assert proibido not in texto, (
            f"a redação de falha voltou à frase pelo fragmento {proibido!r}: {texto!r}"
        )

    de_ontem = (
        "o trabalho não terminou em 3.6e+03 s, e a janela parou de reler depois de 1800 leituras. "
        "Clique em «Gerar» outra vez para continuar a esperar: a rota devolve o mesmo trabalho e "
        "não recomeça nada"
    )
    faltando = [
        clausula
        for clausula in ("CONTINUA a correr no servidor", "serrf", "não recomeça o cálculo")
        if clausula not in de_ontem
    ]
    assert len(faltando) == 3, (
        f"âncora: a frase de ontem já satisfazia {3 - len(faltando)} das três cláusulas, então "
        "elas não separam a redação nova da antiga"
    )
    for proibido in ("não terminou", "parou de reler"):
        assert proibido in de_ontem, (
            f"âncora: {proibido!r} não está na frase de ontem, então proibi-lo na de hoje é "
            "asserção de ausência sobre um conjunto que nunca o teve — vacuidade"
        )


def test_texto_do_decorrido_nao_sai_em_notacao_cientifica_no_teto_novo() -> None:
    """O `:.3g` sozinho CADUCOU quando o teto passou de 300 s para 3.600 s.

    **Nada estava errado no `:.3g`**: ele foi escolhido com medição, contra `:.0f` que escrevia
    *"0 s"* para 50 ms e contra `:g` que dava `0.157123`. Ele deixou de ser verdade quando a
    quantidade que formata mudou de ordem de grandeza.

    **O par causal está no corpo do teste**, e é ele que impede a asserção de passar por acaso: o
    formato antigo aplicado ao mesmo valor produz notação científica, e o novo não.
    """
    # Abaixo de um minuto, o comportamento medido de ontem FICA — é a faixa que os testes do teto
    # exercitam, com `teto=150 ms` a dar 0,157 s.
    assert texto_do_decorrido(0.157123) == "0.157 s"
    assert texto_do_decorrido(0.1) == "0.1 s"
    assert " 0 s" not in texto_do_decorrido(0.1)

    # Acima, a unidade que uma pessoa lê. `1209.2` é o pior `serrf` medido, e `3601.4` é o teto.
    assert texto_do_decorrido(1209.2) == "20 min 9 s"
    assert texto_do_decorrido(3601.4) == "60 min 1 s"

    # **`divmod` sobre o valor ARREDONDADO, e não `%` sobre o float**: `119.7 % 60` dá 59,7, que
    # `:.0f` escreve como **60** — o rótulo diria "1 min 60 s", um relógio que não existe.
    assert texto_do_decorrido(119.7) == "2 min 0 s"

    # **A âncora: o formato de ontem falha exactamente onde o de hoje passa.** Sem ela, este teste
    # passaria com uma implementação que só reescrevesse os mesmos números por acaso.
    for grande in (1209.2, 3601.4):
        assert "e+0" in f"{grande:.3g}", (
            f"âncora: `:.3g` de {grande} não sai em notação científica, então não há defeito a "
            "separar e este teste é decorativo"
        )
        assert "e+0" not in texto_do_decorrido(grande)


@pytest.mark.integration
def test_u13_a_primeira_leitura_e_imediata_e_nao_espera_o_intervalo(qtbot: Any) -> None:
    """A afirmação está na docstring de `_espera_pelo_artefato`, e este é o portão dela.

    **Um mutante que agendasse também a primeira leitura seria invisível à contagem** — o número de
    leituras não muda, só o instante da primeira. O que o separa é o tempo de parede contra um
    intervalo grande.

    Medido: o caminho rápido custa 0,20 s. Com intervalo de 5 s, esperá-lo antes da primeira
    leitura tornaria este teste 25 vezes mais lento — e é essa diferença que a asserção lê.
    """
    intervalo_ms = 5_000
    janela, transporte = _janela_que_retem(qtbot, retidos=0, intervalo_da_espera_ms=intervalo_ms)
    inicio = time.monotonic()
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert _gira_ate(janela, lambda: bool(janela.resultado_da_correcao.salvaveis()), 60), (
        f"os bytes não chegaram; leituras={transporte.leituras()}"
    )
    decorrido = time.monotonic() - inicio
    # **Âncora: o teto da asserção tem de ser MENOR que o intervalo**, senão ela passa com a
    # primeira leitura agendada e não distingue nada.
    teto_s = 3.0
    assert teto_s < intervalo_ms / 1000, (
        f"âncora: o teto de {teto_s} s não é menor que o intervalo de {intervalo_ms / 1000} s"
    )
    assert decorrido < teto_s, (
        f"os bytes levaram {decorrido:.2f} s com um intervalo de {intervalo_ms / 1000} s: a "
        "primeira leitura esperou o intervalo em vez de sair já"
    )
    assert transporte.leituras() == 1, f"leituras={transporte.leituras()}"


@pytest.mark.integration
def test_u13_o_reclique_nao_adianta_o_teto(qtbot: Any) -> None:
    """ACHADO 1 da revisão: dez cliques davam UMA leitura, e a janela informava dez.

    A rota devolve o **mesmo** identificador para a mesma entrada — é o campo `reclamado` —,
    então o `202` do segundo `POST` caía no ramo do *"já estou à espera deste trabalho"* e
    **incrementava** o contador. Aquele `202` não é uma leitura de `/trabalhos/`.

    **Medido pela revisão, com teto de 10 leituras:** 1 clique dava 10 leituras reais, 3 cliques
    davam 8, e 10 cliques davam **1** — com o rótulo a dizer *"10 leituras"* nos três casos. Com os
    padrões, cada reclique impaciente tirava 2 s dos 300 s.

    **O oráculo é o número de leituras REAIS contra o teto**, e não o rótulo: o rótulo mentia
    justamente por concordar com o teto.
    """
    teto, intervalo = 800, 100
    janela, transporte = _janela_que_retem(
        qtbot, retidos=99, intervalo_da_espera_ms=intervalo, teto_da_espera_ms=teto
    )
    assert janela.leituras_maximas == 8

    # Cinco cliques em rajada, que é o que a própria frase de desistência convida a fazer.
    for _ in range(5):
        janela.resultado_da_correcao.gerar.emit("corrigida")
        assert _gira_ate(janela, lambda: bool(janela.esperas_em_curso()), 60), (
            "a espera não nasceu depois de um clique"
        )

    assert _gira_ate(janela, lambda: not janela.esperas_em_curso(), 120), (
        f"a espera não encerrou; leituras={transporte.leituras()}"
    )
    # **Cada clique recomeça a espera**, então a última corrida sozinha tem de gastar o teto
    # inteiro. O defeito consumia o teto com os cliques e desistia com 1 leitura.
    ultimas = transporte.leituras_por_trabalho()
    assert max(ultimas) >= janela.leituras_maximas, (
        f"o teto foi alcançado com {max(ultimas)} leituras contra as {janela.leituras_maximas} que "
        f"ele declara: os recliques adiantaram-no. Por trabalho: {ultimas}"
    )


@pytest.mark.integration
def test_u13_resposta_de_trabalho_supersedido_e_descartada(qtbot: Any) -> None:
    """ACHADO 2 da revisão: duas cadeias na mesma chave desligavam o estrangulamento E o teto.

    Com identificadores diferentes vivos na chave `corrigir`, **cada** resposta caía no ramo do
    primeiro disparo — `leituras=1`, chamada imediata, relógio nenhum. As duas resetavam-se
    mutuamente. Medido pela revisão, pelo caminho real com `GET` de 250 ms e intervalo de 10 ms:
    **64 leituras em 7,965 s**, `leituras` presa em **1**, e o teto declarado de 0,6 s nunca
    alcançado. **É a rajada de 0,001 s que esta unidade existe para remover.**

    O teste ataca o ramo direto, porque a corrida que o produz em produção é de janela estreita — a
    própria revisão não a observou com as latências do `TestClient`, e um teste que dependa dela
    seria intermitente.
    """
    janela, transporte = _janela_que_retem(qtbot, retidos=99, intervalo_da_espera_ms=200)
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert _gira_ate(janela, lambda: bool(janela.esperas_em_curso()), 60), "a espera não nasceu"

    corrente = janela._esperas["corrigida"]
    antes_leituras = corrente.leituras
    antes_rotas = len(transporte.rotas)

    # **A resposta de um trabalho que já não é o desta chave.** Corpo bem formado, código `202`,
    # identificador **outro**.
    janela._espera_pelo_artefato(
        "corrigida",
        "identificador-de-outro-trabalho",
        {"identificador": "identificador-de-outro-trabalho", "feito": 9, "total": 9},
    )

    # **Nada muda: nem a espera corrente, nem o contador, nem o transporte.**
    assert janela._esperas["corrigida"] is corrente, "a espera corrente foi substituída"
    assert janela._esperas["corrigida"].identificador == corrente.identificador
    assert janela._esperas["corrigida"].leituras == antes_leituras, (
        "a resposta supersedida mexeu no contador da espera corrente"
    )
    assert len(transporte.rotas) == antes_rotas, (
        f"a resposta supersedida disparou rota: {transporte.rotas[antes_rotas:]}"
    )

    # **Par causal, na mesma função:** a MESMA chamada com o identificador CORRENTE conta e agenda.
    janela._espera_pelo_artefato(
        "corrigida",
        corrente.identificador,
        {"identificador": corrente.identificador, "feito": 1, "total": 9},
    )
    assert janela._esperas["corrigida"].leituras == antes_leituras + 1, (
        "a resposta corrente NÃO contou, então a igualdade acima não separa descarte de inércia"
    )


@pytest.mark.integration
def test_u13_o_decorrido_da_desistencia_e_medido_e_nao_derivado(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ACHADO 7 da revisão: o teto declarado não era teto de PAREDE.

    A versão anterior calculava `(leituras_maximas - 1) x intervalo`, que ignora a latência das
    próprias leituras. **Medido pela revisão com `GET` de 250 ms:** o rótulo dizia *"0.5 s"* e a
    parede real da primeira à última leitura era **1,754 s** — 3,5x.

    **A ilustração com os padrões foi refeita em 30/08/2026 e é ARITMÉTICA, não medição.** Ela dizia
    *"a espera real passa de 298 s para ~328 s"*, e caducou com o teto: com 1.800 leituras de 2 s a
    derivação antiga daria **3.598 s**, e leituras de 0,2 s põem a parede em **~3.958 s**. O erro
    absoluto que a derivação comete cresce com o teto, que é o motivo de o rótulo medir.

    **O oráculo é o seam `agora()`, e ele existe exactamente para isto.** Substituído o
    relógio, o número relatado tem de ser a diferença de instantes — e a derivação antiga daria
    `(3 - 1) x 0,05 = 0,1 s`, que passa em toda asserção dos outros testes. **Sem este teste, um
    mutante que voltasse a derivar escaparia**, e foi medido que escaparia.
    """
    # **O relógio falso é uma FUNÇÃO da contagem de leituras, e não uma sequência.** A primeira
    # versão era `iter([1_000.0] + [1_042.5] * 20)`, e ela **passou isolada e reprovou na suíte**:
    # `monkeypatch.setattr` substitui o atributo do MÓDULO, e `QApplication.processEvents()` entrega
    # eventos de **todas** as janelas — inclusive as que testes anteriores deixaram vivas na mesma
    # `QApplication`. Cada uma delas, ao tratar um `202` pendente, chamava `agora()` e consumia uma
    # posição do iterador. Medido: o rótulo disse *"0 s"*, que é a frase que
    # `test_u13_a_desistencia_nao_arredonda_a_espera_para_zero` existe para proibir.
    #
    # > **Duplo com estado de sequência é recurso GLOBAL quando o seam é de módulo.** Uma função do
    # > estado observável não tem esse problema: chamadas alheias devolvem algum valor e não movem
    # > nada.
    #
    # A quantidade escolhida é a contagem de leituras, que as outras asserções desta seção já fixam:
    # a criação vê **zero** leituras — é o `202` do `POST` — e a desistência vê `leituras_maximas`.
    janela, transporte = _janela_que_retem(
        qtbot, retidos=99, intervalo_da_espera_ms=50, teto_da_espera_ms=150
    )
    assert janela.leituras_maximas == 3
    monkeypatch.setattr(
        "refino.desktop.janela.principal.agora",
        lambda: 1_000.0 + 10.0 * transporte.leituras(),
    )
    esperado_s = 10.0 * janela.leituras_maximas

    janela.resultado_da_correcao.gerar.emit("corrigida")
    # **Esperar que a espera NASÇA antes de esperar que ela morra.** `not esperas_em_curso()` é
    # verdade **antes** de o `POST` voltar, e a primeira versão deste teste saiu do laço na primeira
    # volta — lendo o rótulo inicial `nada gerado ainda` como se fosse desfecho. Terceira instância
    # da mesma forma nesta sessão, todas no arnês.
    assert _gira_ate(janela, lambda: bool(janela.esperas_em_curso()), 60), "a espera não nasceu"
    assert _gira_ate(janela, lambda: not janela.esperas_em_curso(), 60), (
        f"a espera não encerrou; leituras={transporte.leituras()}"
    )

    assert transporte.leituras() == janela.leituras_maximas, (
        f"o teto não foi alcançado com as leituras que ele declara: {transporte.leituras()}. O "
        "relógio falso é função desta contagem, então o decorrido esperado abaixo depende dela"
    )
    rotulo = janela.resultado_da_correcao.rotulos["corrigida"].text()
    assert f"{esperado_s:.3g} s" in rotulo, (
        f"o decorrido relatado não é a diferença de instantes do relógio: {rotulo!r}. Esperado "
        f"{esperado_s:.3g} s; a derivação antiga daria "
        f"{(janela.leituras_maximas - 1) * janela.intervalo_da_espera_ms / 1000} s"
    )
    # **Âncora: a derivação antiga e o medido DIFEREM neste dado.** Sem isto, o teste passaria com
    # as duas implementações e não separaria nada.
    derivado = (janela.leituras_maximas - 1) * janela.intervalo_da_espera_ms / 1000
    assert f"{derivado:.3g} s" != f"{esperado_s:.3g} s", (
        f"âncora: a derivação antiga dá {derivado:.3g} s, igual ao medido — este dado não separa "
        "as duas implementações"
    )


# ---------------------------------------------------------------------------
# A CHAVE COMPOSTA: de qual trabalho são estes bytes?
# ---------------------------------------------------------------------------
#
# **O `200` de um artefato é CSV e não carrega identificador nenhum** — `cliente.envia` lê o corpo
# pelo `content-type`, então `conteudo` fica com os bytes e `corpo` fica em `None`. O ramo terminal
# de `_encaminha` sabia **que** bytes chegaram e não **de qual trabalho**, e gravava sob a
# identidade corrente do painel.
#
# Medido pela revisão adversarial: clique, troca do par de arquivos, clique — e o `200` da
# cadeia **velha** deixava `salvaveis=('corrigir',)` com `b'BYTES_DE_A'` sob a identidade
# `neg_c18|none|ENTRADA_B`, com o botão de salvar habilitado e o rótulo a dizer *"a correr"*.
#
# O canal de volta é a chave, então é a chave que carrega o identificador.


def _resposta_com_bytes(conteudo: bytes = b"a,b\n1,2\n") -> Resposta:
    """Um `200` de artefato: bytes em `conteudo` e `corpo` em `None`, como `cliente.envia` faz."""
    return Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=None, conteudo=conteudo)


def test_u14_nenhuma_chave_de_rota_contem_o_separador() -> None:
    """`partition` corta na PRIMEIRA ocorrência, então é a CHAVE que não pode conter o separador.

    **Os dois conjuntos são derivados e não escritos à mão** — `ARTEFATOS` vem da partição de
    `interface/fluxo.py` e `ROTAS_DA_LEITURA` do painel —, então uma chave nova entra nesta
    verificação sem ninguém se lembrar dela.

    Identificador com o separador não quebra a base: truncaria a comparação, e o desfecho seria
    **descartar**. Falha para o lado seguro, e por isso não precisa de portão.
    """
    chaves = (
        {passo.chave for passo in PASSOS_DOS_ARTEFATOS}
        | set(ARTEFATO_DO_RELATORIO)
        | set(dict(ROTAS_DA_LEITURA))
    )
    assert len(chaves) >= 4, (
        f"âncora: só {len(chaves)} chave(s) para conferir — os conjuntos derivados não carregaram, "
        "e a asserção abaixo é vácua"
    )
    culpadas = sorted(c for c in chaves if SEPARADOR_DO_PEDIDO in c)
    assert culpadas == [], (
        f"estas chaves contêm o separador {SEPARADOR_DO_PEDIDO!r} e `parte_a_chave` cortaria "
        f"a base no meio delas: {culpadas}"
    )


def test_u14_o_par_de_funcoes_da_chave_e_redondo() -> None:
    """`chave_do_pedido` e `parte_a_chave` são o par, e `None` não é `""`.

    A distinção decide comportamento: `None` significa *"esta resposta não é de leitura
    encadeada"* — o `POST`, ou uma chamada de arnês — e o ramo terminal **entrega**. Um `""`
    cobriria isso e *"o identificador é vazio"* com o mesmo valor.
    """
    composta = chave_do_pedido("corrigir", "8993abc")
    assert composta == f"corrigir{SEPARADOR_DO_PEDIDO}8993abc"
    assert parte_a_chave(composta) == ("corrigir", "8993abc")
    assert parte_a_chave("corrigir") == ("corrigir", None)
    assert parte_a_chave(f"corrigir{SEPARADOR_DO_PEDIDO}") == ("corrigir", None)


@pytest.mark.integration
def test_u14_o_sinal_emite_a_chave_base_mesmo_na_leitura_encadeada(qtbot: Any) -> None:
    """É esta propriedade que mantém os seis painéis e o portão da janela contra a API intactos.

    **Sem a separação em `_recebe`, o sinal emitiria `corrigir@8993…`** — e o painel de artefatos
    filtra por chave (`chave not in self.rotulos`), então ele deixaria de reconhecer a
    própria resposta. A âncora mede que a leitura encadeada de facto aconteceu: sem ela, o teste
    passaria sobre uma execução em que só o `POST` voltou.
    """
    janela, transporte = _janela_que_retem(qtbot, retidos=2, intervalo_da_espera_ms=60)
    chaves: list[str] = []
    janela.resposta_recebida.connect(lambda chave, _r: chaves.append(chave))

    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert _gira_ate(janela, lambda: bool(janela.resultado_da_correcao.salvaveis()), 120), (
        f"os bytes não chegaram; leituras={transporte.leituras()}"
    )

    assert transporte.leituras() >= 2, (
        f"âncora: houve {transporte.leituras()} leitura(s), então o caminho encadeado com relógio "
        "não foi exercitado e a asserção abaixo não separa nada"
    )
    assert len(chaves) >= 3, f"âncora: só {len(chaves)} resposta(s) emitida(s): {chaves}"
    compostas = [c for c in chaves if SEPARADOR_DO_PEDIDO in c]
    assert compostas == [], (
        f"o sinal emitiu chave composta: {compostas}. Os painéis filtram por chave e deixariam de "
        "reconhecer a própria resposta"
    )
    assert set(chaves) == {"corrigida"}, f"emitiu chave de outra coisa: {sorted(set(chaves))}"


@pytest.mark.integration
def test_u14_os_bytes_de_um_trabalho_supersedido_sao_descartados(qtbot: Any) -> None:
    """O ACHADO 3, fechado. E o par causal está na mesma função.

    O cenário real é clique, troca de arquivos, clique — e o `200` da cadeia velha a chegar. Aqui a
    resposta terminal é entregue direto a `_encaminha` com o identificador que a pediu, porque a
    corrida que a produz em produção é de janela estreita e um teste que dependa dela seria
    intermitente.
    """
    janela, _ = _janela_que_retem(qtbot, retidos=99, intervalo_da_espera_ms=400)
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert _gira_ate(janela, lambda: bool(janela.esperas_em_curso()), 60), "a espera não nasceu"
    corrente = janela._esperas["corrigida"].identificador

    # ── os bytes de OUTRO trabalho: descartados ──
    janela._encaminha("corrigida", _resposta_com_bytes(b"BYTES_DE_A"), "identificador-de-outro")
    salvaveis = janela.resultado_da_correcao.salvaveis()
    assert salvaveis == (), f"os bytes de um trabalho supersedido chegaram ao painel: {salvaveis}"
    assert janela.esperas_em_curso() == ("corrigida",), (
        "a resposta supersedida encerrou a espera corrente"
    )

    # ── PAR CAUSAL: os mesmos bytes, com o identificador CORRENTE ──
    janela._encaminha("corrigida", _resposta_com_bytes(b"BYTES_DE_B"), corrente)
    assert janela.resultado_da_correcao.salvaveis() == ("corrigida",), (
        "os bytes do trabalho CORRENTE não chegaram ao painel — então o descarte acima não separa "
        "nada, e este teste é vácuo"
    )
    assert janela.esperas_em_curso() == (), "o desfecho terminal não encerrou a espera"


@pytest.mark.integration
def test_u14_o_200_atrasado_depois_de_a_espera_fechar_e_descartado(qtbot: Any) -> None:
    """Espera ausente com pedido presente também descarta, e a consequência vai declarada.

    Um `200` que chegue depois de a espera ter sido encerrada — por desistência, por clique novo ou
    pelo fecho da janela — é descartado. **Para a desistência isso é exactamente o que o rótulo
    promete:** clicar outra vez no botão do artefato reclama o mesmo trabalho, porque a rota devolve
    o mesmo identificador para a mesma entrada.

    **Sem esta direção, `espera is None` entregaria** — e é justamente o estado em que a
    cadeia velha chega quando o `POST` novo ainda não voltou.
    """
    janela, _ = _janela_que_retem(qtbot, retidos=99, intervalo_da_espera_ms=400)
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert _gira_ate(janela, lambda: bool(janela.esperas_em_curso()), 60), "a espera não nasceu"
    pedido = janela._esperas["corrigida"].identificador
    assert janela.para_a_espera("corrigida") is True

    janela._encaminha("corrigida", _resposta_com_bytes(b"ATRASADO"), pedido)
    salvaveis = janela.resultado_da_correcao.salvaveis()
    assert salvaveis == (), f"um `200` atrasado chegou ao painel sem espera aberta: {salvaveis}"

    # **PAR CAUSAL: `pedido is None` ENTREGA**, e tem de entregar — é o `POST` e as chamadas de
    # arnês. Sem esta metade, um filtro que descartasse tudo passaria no teste acima.
    janela._encaminha("corrigida", _resposta_com_bytes(b"DO POST"), None)
    assert janela.resultado_da_correcao.salvaveis() == ("corrigida",), (
        "resposta sem pedido foi descartada — o filtro está a barrar o `POST` e as chamadas "
        "de arnês"
    )


def test_u14_a_falha_mostra_a_chave_base_e_nao_a_composta(janela: JanelaPrincipal) -> None:
    """O identificador é detalhe de encadeamento e não diz nada a quem lê o rótulo."""
    vistos: list[str] = []
    janela.chamada_falhou.connect(lambda chave, _e: vistos.append(chave))

    janela._falha(chave_do_pedido("corrigir", "8993abcdef"), RuntimeError("morreu"))

    texto = janela.rotulo_de_desfecho.text()
    assert "`corrigir`" in texto, texto
    assert "8993abcdef" not in texto, f"o rótulo mostra o identificador ao profissional: {texto!r}"
    assert vistos == ["corrigir"], f"o sinal levou a chave composta: {vistos}"


# ---------------------------------------------------------------------------
# A TELA: o que se ve, medido nas capturas de 23/08/2026
# ---------------------------------------------------------------------------
#
# **A primeira abertura da janela, em tres capturas, mostrou o que a suite nao via.** Nenhum dos
# 2.842 testes
# verdes olhava aparencia nem se o valor chega ao destino, e por isso sete defeitos visiveis
# conviviam com a suite verde.
#
# | O que se via | A causa |
# |---|---|
# | rotulos a flutuar a meio da tela | `addStretch` nao aparecia em `desktop/janela/` em lugar
# nenhum |
# | a barra azul em loop em REPOUSO | `setRange(0, 0)` na construcao e o indeterminado do Qt |
# | `None: None/4 etapas` | o `202` do `POST` e formatado com o esquema do `GET` |
# | `**65%**` com os asteriscos | `AutoText` decide entre texto e HTML, e nao ve markdown |
# | «nenhuma leitura feita» cinco vezes | as justificativas nascem anonimas |
# | `Validacao`, `Diagnostico` | o rotulo era `chave.capitalize()` |
# | «Ler progresso» nao faz nada | sem identificador, o slot voltava em silencio |


PAINEIS_DA_JANELA = sorted(
    (Path(__file__).resolve().parents[1] / "src" / "refino" / "desktop" / "janela").glob("*.py")
)


def test_u18_todo_painel_declara_o_dono_da_folga_do_layout() -> None:
    """O PORTAO DA APARENCIA, e ele nasce de um defeito visto — **EMENDADO**.

    Um `QVBoxLayout` sem dono para a folga reparte-a ENTRE os widgets, e o resultado medido nas
    capturas de 23/08/2026 e um rotulo do par de arquivos a ~570 px do topo. Os seis paineis tinham
    o mesmo defeito ao mesmo tempo, o que e a assinatura de uma omissao sistematica.

    ## A PRIMEIRA EMENDA, e ela vem de o oraculo antigo pedir o defeito OPOSTO

    O oraculo era *"ha uma chamada a `addStretch` no modulo"*. O painel do multiverso satisfazia-o
    **e tinha o defeito contrario**: desenhado em 24/08/2026 com 14 linhas numa janela de 1180x800,
    a tabela ficou com **~180 px**, as linhas 6 a 14 cortadas, e **~400 px de vazio** abaixo — o
    `addStretch` no fim ficava com o espaco que a tabela devia absorver.

    > **A propriedade real nao e «ha um `addStretch`»: e «a folga tem DONO declarado».** Um painel
    > de rotulos declara-o com um esticador no fim; um painel cujo widget dominante deve crescer
    > declara-o com o **fator de esticamento** daquele widget — `addWidget(x, 1)`. As duas formas
    > satisfazem a propriedade, e so uma satisfazia o portao.

    **A fraqueza CONTINUA declarada:** ele nao confere que o dono seja o widget certo nem que o
    esticador esteja no fim. E fio de tropeco contra a omissao — a forma que de facto aconteceu —, e
    nao prova de layout correto. O que acha esta classe e desenhar a janela e olhar.
    """
    import ast as _ast

    # **A POPULACAO passou a ser DERIVADA em 07/09/2026, e a mudanca e de oraculo e nao de
    # isencao.** Ela era *"todo `.py` do pacote, menos dois nomeados"*, e o pacote ganhou
    # `aparencia.py` — um modulo de convencoes que nao desenha painel nenhum. Isenta-lo pelo nome
    # seria a terceira entrada numa lista que ja era o sintoma: **o que este portao pergunta e
    # «quem recebe a folga DESTE layout», e um modulo que nao constroi layout nenhum nao tem folga
    # a repartir.**
    #
    # O oraculo antigo codificava a FORMA da
    # populacao no dia em que ela foi vista — nove arquivos, todos paineis —, e nao a propriedade.
    # **Medido: com o filtro, oito modulos entram e `aparencia.py` fica de fora por construcao.**
    def _constroi_layout(caminho: Path) -> bool:
        """O modulo instancia algum `Q*Layout`? So esse tem folga para repartir."""
        for no in _ast.walk(_ast.parse(caminho.read_text(encoding="utf-8"))):
            if (
                isinstance(no, _ast.Call)
                and isinstance(no.func, _ast.Name)
                and no.func.id.startswith("Q")
                and no.func.id.endswith("Layout")
            ):
                return True
        return False

    fontes = [
        c
        for c in PAINEIS_DA_JANELA
        if c.name not in {"__init__.py", "principal.py"} and _constroi_layout(c)
    ]
    assert len(fontes) >= 5, f"ancora: so {len(fontes)} painel(eis) achado(s)"

    def _e_inteiro(no: _ast.expr) -> bool:
        """O argumento e um literal inteiro — qualquer um, inclusive zero.

        **`bool` NAO conta, e a distincao e do Python e nao do Qt:** `True` e instancia de `int`,
        e um `addWidget(w, 1, True)` — que ninguem escreve — cairia no ramo da grade por acidente.
        """
        return (
            isinstance(no, _ast.Constant)
            and isinstance(no.value, int)
            and not isinstance(no.value, bool)
        )

    def _fator_declarado(no: _ast.expr) -> bool:
        """O argumento e um inteiro literal maior que zero."""
        return _e_inteiro(no) and no.value >= 1  # type: ignore[attr-defined]

    def _declara_dono_da_folga(caminho: Path) -> bool:
        """Tres formas: `.addStretch(...)`, `.addWidget(x, n>=1)` com DOIS args, ou `set*Stretch`.

        **Chamada e nao a palavra no texto, e a distincao custou um mutante sobrevivente.** Tirar a
        linha `disposicao.addStretch(1)` de `arquivos.py` deixou o portao verde na primeira versao,
        porque o comentario que explica a linha contem a palavra. O instrumento leu a prosa SOBRE a
        chamada em vez da chamada.

        ## A SEGUNDA EMENDA, e ela tem as duas direcoes

        **A que ABRE: `QGridLayout` declara o dono por `setRowStretch` / `setColumnStretch`.** A
        reestruturacao de 03/09/2026 poe os paineis em grade — rotulo a esquerda, controle a
        direita —, e numa grade nao existe `addStretch`: quem recebe a folga e a linha ou a coluna
        nomeada. O oraculo antigo obrigaria a escrever um `addStretch` decorativo num layout que
        nao o consome. Portao que reprova pela FORMA esta a dizer que a
        forma e contrato, e aqui a propriedade («a folga tem dono declarado») e a mesma nas tres
        formas.

        **A que FECHA, e ela e um falso positivo MEDIDO e nao hipotetico:** `QGridLayout.addWidget`
        tem assinatura `(widget, linha, coluna)`, e o oraculo antigo lia a **linha** como fator de
        esticamento. Medido em 03/09/2026 sobre a arvore de entao, `multiverso.py:475` —
        `escolhas.addWidget(QLabel("Contraste (grupo A x grupo B):", self), 1, 0)` — satisfazia o
        portao por acidente: aquele `1` e o indice da segunda linha da grade, e nao declara folga
        nenhuma. O painel passava de todo modo pelos seus dois `addStretch`, entao o falso positivo
        nunca teve consequencia; a emenda tira-o antes que um painel novo passe a depender dele.

        **O que separa as duas assinaturas e o TERCEIRO argumento, e nao a contagem deles.** A
        primeira versao desta emenda exigia exactamente dois, com o argumento de que *"a forma de
        caixa tem exactamente dois e a de grade tem tres ou mais"*. **Falso, e medido com PySide6
        em 03/09/2026:** `QBoxLayout::addWidget(widget, stretch = 0, alignment = {})` aceita tres,
        e `lay.addWidget(a, 1, Qt.AlignmentFlag.AlignTop)` devolve `lay.stretch(0) == 1` — um
        painel que declarasse o dono com alinhamento seria REPROVADO por um portao que deveria
        aceita-lo. E o falso positivo que *"mata o portao no primeiro dia"*.

        O discriminador e o tipo: na **grade** o terceiro argumento e a coluna, um `int`; na
        **caixa** e um `Alignment`, que nunca e literal inteiro. Entao a regra e: fator literal
        `>= 1` na posicao 1, **e** ou nao ha terceiro argumento, ou o terceiro nao e um inteiro.

        > **A CONTAGEM ANTES E DEPOIS, e ela CORRIGE o que esta nota afirmava.** A versao anterior
        > dizia *"os oito passam, todos por `addStretch`; depois, os mesmos oito pelo mesmo
        > caminho — a emenda nao afrouxa nada hoje e acrescenta uma forma que ainda nao existe na
        > arvore"*. **Era verdade quando foi escrita e ficou falsa na mesma sessao**, porque a
        > mesma unidade converteu `artefatos.py` para `QGridLayout`. Medido em 03/09/2026:
        >
        > | painel | forma por que passa |
        > |---|---|
        > | arquivos, leitura, pipeline, qualidade, recorte, trabalhos | `addStretch` |
        > | multiverso | `addStretch` **e** `addWidget` de dois argumentos |
        > | **artefatos** | **so `set*Stretch`** — nao tem `addStretch` nenhum |
        >
        > E o oraculo ANTIGO sobre a arvore de hoje reprova `['artefatos.py']`. **A emenda nao e
        > neutra: e ela que mantem o portao verde**, e dize-lo e o que permite a proxima sessao
        > auditar a decisao em vez de a herdar. Nada estava errado na medicao, e o referente
        > mudou por baixo dela.
        """
        arvore = _ast.parse(caminho.read_text(encoding="utf-8"))
        for no in _ast.walk(arvore):
            if not (isinstance(no, _ast.Call) and isinstance(no.func, _ast.Attribute)):
                continue
            if no.func.attr == "addStretch":
                return True
            # `addWidget(widget, fator[, alinhamento])` com fator literal maior que zero: o widget
            # e o dono. O terceiro argumento, quando existe, tem de NAO ser inteiro — senao e
            # `addWidget(w, linha, coluna)` de uma grade, e o `1` ali e indice e nao esticamento.
            if (
                no.func.attr == "addWidget"
                and len(no.args) in (2, 3)
                and _fator_declarado(no.args[1])
                and not (len(no.args) == 3 and _e_inteiro(no.args[2]))
            ):
                return True
            # `setRowStretch(i, n)` / `setColumnStretch(i, n)` / `setStretch(i, n)`: a linha, a
            # coluna ou o item nomeado e o dono.
            if (
                no.func.attr in {"setRowStretch", "setColumnStretch", "setStretch"}
                and len(no.args) == 2
                and _fator_declarado(no.args[1])
            ):
                return True
        return False

    sem = [c.name for c in fontes if not _declara_dono_da_folga(c)]
    assert sem == [], (
        f"estes paineis nao declaram o dono da folga do layout: {sem}. Sem dono o `QVBoxLayout` "
        "reparte o espaco que sobra entre os widgets, e os rotulos flutuam no meio da tela. As "
        "tres formas valem: `addStretch(...)` no fim; `addWidget(x, n)` com DOIS argumentos e "
        "`n >= 1` no widget que deve crescer; ou `setRowStretch`/`setColumnStretch(i, n)` com "
        "`n >= 1` na linha ou coluna da grade que deve crescer"
    )


def test_u18_todo_metodo_de_payload_de_painel_tem_chamador_ou_e_declarado() -> None:
    """O PORTAO DA LIGACAO: um metodo `campos_*` que ninguem chama e um controle que MENTE.

    **Medido em 23/08/2026:** `PainelDeLeitura.campos_de_sobrescrita` existe, tem testes, e
    `principal.py` nunca o chama. O painel tem ZERO widgets de entrada, entao ele so pode devolver
    strings vazias — e quem marca o checkbox nao tem como dizer com que valor sobrescrever.

    > **O recorte e ESTREITO de proposito: so o prefixo `campos_`.** Nao ha como derivar «metodos
    > que
    > compoem payload» de coisa nenhuma, e um portao sobre todo metodo publico acusaria os acessores
    > de teste. Fio de tropeco sobre a classe que ja mordeu, e dize-lo e o que impede a proxima
    > sessao de o ler como cobertura.

    A isencao e DECLARADA e nao silenciosa: quem nao tem chamador entra na tabela com a razao.
    """
    import ast as _ast

    # Metodos `campos_*` definidos nos paineis, derivados por AST.
    definidos: dict[str, str] = {}
    for caminho in PAINEIS_DA_JANELA:
        if caminho.name in {"__init__.py", "principal.py"}:
            continue
        arvore = _ast.parse(caminho.read_text(encoding="utf-8"))
        for no in _ast.walk(arvore):
            if isinstance(no, _ast.FunctionDef) and no.name.startswith("campos_"):
                definidos[no.name] = caminho.name
    assert definidos, "ancora: nenhum metodo `campos_*` achado nos paineis"

    fonte_janela = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "refino"
        / "desktop"
        / "janela"
        / "principal.py"
    ).read_text(encoding="utf-8")

    # **A isencao, com a razao de cada uma.** Uma linha aqui e uma afirmacao auditavel.
    #
    # **A TABELA ESTAVA VAZIA, e o vazio era o desfecho e nao a ausencia de politica.** O
    # unico membro era `campos_de_sobrescrita`, e o metodo foi **removido** em vez de mantido
    # isento: os tres eixos marcaveis ja tem controle de valor no painel do pipeline, e os valores
    # ja viajam como campos de spec — liga-lo escreveria as mesmas tres chaves uma segunda vez.
    # A tabela fica, porque e ela que obriga a proxima isencao a ter razao escrita.
    # **O PRIMEIRO MEMBRO depois disso:**
    #
    # `campos_dos_limiares` NAO e metodo de painel: e funcao de MODULO em `multiverso.py`, pura,
    # exercitada pelo arnes sem construir a janela — o molde de `rota_do_resultado`. Ela casa o
    # prefixo `campos_` porque o portao anda a arvore inteira com `_ast.FunctionDef`, e nao so as
    # definicoes dentro de classe.
    #
    # **A propriedade que o portao protege CONTINUA a valer, e e por isso que a isencao e honesta:**
    # o valor sai do painel. `PainelDeMultiverso.campos_da_conclusao` chama esta funcao, e
    # `principal.py` chama `.campos_da_conclusao(` — que o portao ve. O que ele nao segue e a
    # travessia de um nivel.
    #
    # **Renomea-la para escapar do prefixo seria a resposta errada:** o nome descreve o que ela
    # devolve, e mudar o nome do codigo para agradar ao oraculo de um portao e afrouxa-lo com outro
    # gesto. A tabela existe exactamente para este caso, e a razao fica escrita aqui.
    sem_chamador: dict[str, str] = {
        "campos_dos_limiares": (
            "funcao de MODULO em `multiverso.py`, nao metodo de painel. E chamada por "
            "`PainelDeMultiverso.campos_da_conclusao`, que `principal.py` chama — o valor SAI do "
            "painel, e o portao nao segue a travessia de um nivel"
        ),
        # **O segundo membro, de 08/11/2026, e pela mesma razao de forma do primeiro.** Ela e
        # funcao de MODULO, pura, e `principal.py` a chama pelo NOME importado —
        # `campos_do_download(`, sem o ponto que o oraculo procura. O valor sai: e ela que monta os
        # campos do download do Resultado a partir da etapa feita, em `_gera_artefato`.
        "campos_do_download": (
            "funcao de MODULO em `artefatos.py`, pura: os campos do download do Resultado de uma "
            "etapa, tirados da etapa feita. `principal.py` a chama em `_gera_artefato` pelo nome "
            "importado, sem o ponto que este portao procura — o valor SAI para o pedido"
        ),
    }

    orfaos = [
        f"{nome} ({arquivo})"
        for nome, arquivo in sorted(definidos.items())
        if f".{nome}(" not in fonte_janela and nome not in sem_chamador
    ]
    assert orfaos == [], (
        f"estes metodos de payload nao tem chamador em `principal.py` e nao estao declarados: "
        f"{orfaos}. Um controle cujo valor nao sai do painel e um controle que mente"
    )
    # A segunda direcao: isencao que ganhou chamador tem de sair da tabela.
    obsoletas = [n for n in sem_chamador if f".{n}(" in fonte_janela]
    assert obsoletas == [], f"estas isencoes ja tem chamador e continuam declaradas: {obsoletas}"
    # **A TERCEIRA direcao:** isencao para metodo que JA NAO EXISTE.
    #
    # Ao remover `campos_de_sobrescrita` a entrada dele ficaria na tabela a apontar para nada, e
    # nenhuma das duas direcoes acima o veria — a primeira so olha o que `definidos` tem, e a
    # segunda pergunta se o metodo ganhou chamador. Uma isencao orfa e pior que ausente: ela afirma
    # que um metodo existe e esta isento, e a proxima sessao gasta atencao a procura-lo.
    fantasmas = sorted(set(sem_chamador) - set(definidos))
    assert fantasmas == [], (
        f"estas isencoes apontam para metodo que nao existe mais: {fantasmas}. Metodo removido "
        "tira a isencao dele junto"
    )


def test_u18_a_barra_em_repouso_nao_finge_trabalho(qtbot: Any) -> None:
    """`setRange(0, 0)` e o indeterminado do Qt, e em repouso ele afirma trabalho que nao ha.

    **O oraculo e `minimum == maximum`**, que e a definicao do indeterminado — nao um valor
    especifico. Um teste que exigisse `maximum == 1` pinaria a escolha em vez de medir a
    propriedade.
    """
    p = PainelDePipeline()
    qtbot.addWidget(p)
    # A barra da execução é a do rodapé da Análise desde 09/11/2026: a da moldura «Executar», que
    # este teste lia, saiu com a aba «Executar e exportar».
    barra = p.barras_de_etapa[ETAPAS[-1]]
    assert not (barra.minimum() == barra.maximum() == 0), (
        "a barra nasce indeterminada, e a tela parece estar a trabalhar quando nao esta"
    )
    assert barra.value() == barra.minimum(), (
        f"a barra nasce com valor {barra.value()} sobre [{barra.minimum()}, "
        f"{barra.maximum()}]: ela afirma progresso que ninguem mediu"
    )


def test_u18_o_corpo_de_aceite_nao_produz_none(qtbot: Any) -> None:
    """`None: None/4 etapas` — o `202` do `POST` formatado com o esquema do `GET`.

    Os dois corpos e o par causal estao na mesma funcao: o de aceite nao pode produzir `None`, e o
    de progresso tem de continuar a dizer `estado` e `feito`.
    """
    p = PainelDePipeline()
    qtbot.addWidget(p)

    aceite = p.recebe_leitura(202, {"identificador": "abc", "total": 4, "reclamado": False})
    assert "None" not in aceite, f"o corpo de aceite ainda produz `None`: {aceite!r}"
    assert "4" in aceite, f"o corpo de aceite perdeu o total: {aceite!r}"

    progresso = p.recebe_leitura(
        202,
        {"identificador": "abc", "estado": "executando", "feito": 2, "total": 4, "fracao": 0.5},
    )
    assert "Executando" in progresso and "2/4" in progresso, (
        f"o corpo de PROGRESSO deixou de ser formatado: {progresso!r}"
    )
    barra = p.barras_de_etapa[ETAPAS[-1]]
    assert barra.maximum() == 4 and barra.value() == 2, (
        f"a barra nao seguiu o progresso: [{barra.minimum()}, {barra.maximum()}] = {barra.value()}"
    )


def test_u18_os_rotulos_da_leitura_tem_acento_e_bijecao() -> None:
    """Os botoes diziam `Validacao`. E a bijecao e o que impede a tabela nova de envelhecer."""
    assert set(ROTULOS_DA_LEITURA) == set(dict(ROTAS_DA_LEITURA)), (
        f"as duas tabelas divergiram: so em rotulos="
        f"{set(ROTULOS_DA_LEITURA) - set(dict(ROTAS_DA_LEITURA))}, so em rotas="
        f"{set(dict(ROTAS_DA_LEITURA)) - set(ROTULOS_DA_LEITURA)}"
    )
    sem_acento = [k for k, v in ROTULOS_DA_LEITURA.items() if v == k.capitalize()]
    assert sem_acento == [], (
        f"estes rotulos sao a chave capitalizada, e a chave nao tem acento: {sem_acento}"
    )


def test_u18_a_prosa_da_leitura_e_renderizada_como_markdown(qtbot: Any) -> None:
    """`**65%**` aparecia com os asteriscos: `AutoText` ve HTML e nao ve markdown.

    **O oraculo e o formato do rotulo e nao o texto**, porque o `QLabel` guarda o markdown cru em
    `text()` e so o RENDERIZA diferente. Medir o texto nao separaria as duas implementacoes.
    """
    p = PainelDeLeitura()
    qtbot.addWidget(p)
    from PySide6.QtCore import Qt as _Qt

    crus = [c for c, r in p.rotulos.items() if r.textFormat() != _Qt.TextFormat.MarkdownText]
    assert crus == [], f"estes rotulos mostram markdown cru: {crus}"
    assert len(p.rotulos) >= 3, f"ancora: so {len(p.rotulos)} rotulo(s)"


def test_u18_as_justificativas_nascem_nomeadas(qtbot: Any) -> None:
    """Cinco rotulos com o mesmo texto inicial, dois deles sem dizer de que sao.

    O nome vem de `ponto.o_que_decide`, que e o campo que `PontoDeDecisao` declara para isso.

    **QUEM O CARREGA MUDOU em 15/09/2026, e o oráculo mudou com ele.** Este teste nasceu quando o
    rótulo era filho do painel e o texto dele era o único portador do nome; desde que existe um
    `QGroupBox` por ponto, quem nomeia é a moldura, e o prefixo dentro do corpo era o mesmo
    enunciado duas vezes na mesma tela. **A propriedade é a mesma — o rótulo em repouso diz de que
    ponto é —, e o que mudou é onde o instrumento a lê.**
    """
    from PySide6.QtWidgets import QGroupBox

    p = PainelDeLeitura()
    qtbot.addWidget(p)
    assert p.justificativas, "ancora: nenhuma justificativa"

    for chave, rotulo in p.justificativas.items():
        ponto = next(p_ for p_ in PONTOS_DE_DECISAO_AUTOMATICA if p_.chave == chave)
        # **A ÂNCORA vem antes da asserção de ausência**: *"o rótulo não repete o título"*
        # é satisfeito por um rótulo VAZIO, que é o desfecho de um rótulo que não foi semeado.
        assert rotulo.text().strip(), f"{chave}: o rotulo de repouso nasceu em branco"
        moldura = rotulo.parent()
        assert isinstance(moldura, QGroupBox), (
            f"{chave}: a justificativa nao esta numa moldura: o pai e {type(moldura)}"
        )
        assert moldura.title() == com_maiuscula(ponto.o_que_decide), (
            f"{chave}: a moldura nao nomeia o que o ponto decide: {moldura.title()!r}"
        )
        for titulo in (ponto.o_que_decide, com_maiuscula(ponto.o_que_decide)):
            assert titulo not in rotulo.text(), (
                f"{chave}: o rotulo repete o titulo da moldura: {rotulo.text()!r}"
            )


@pytest.mark.integration
def test_u18_a_recusa_do_disparo_aparece_no_painel_do_botao(qtbot: Any) -> None:
    """A recusa ia so para o rodape de uma linha, e quem clicou olha para o painel.

    **Medido:** marcar um eixo sem ter pedido a recomendacao produz ZERO rotas chamadas. A
    explicacao existia e estava no sitio errado.

    **O botao e o painel mudaram em 09/11/2026, e a propriedade nao.** Quem recusava era o
    «Executar pipeline», e a frase ia a linha de progresso ao lado dele; o botao saiu com a aba
    «Executar e exportar». Quem leva a linha da maquina do par agora e a Correcao, e a recusa vai
    ao aviso vermelho do rodape da aba dela, logo acima do botao clicado.
    """
    janela, transporte = _janela_que_retem(qtbot, retidos=0)
    assert _faz_divergir(janela.pipeline), "âncora: nada divergiu, e não haveria recusa a medir"
    antes = len(transporte.rotas)
    pagina = janela.pagina_da_correcao
    assert not pagina.faixa_da_etapa.frase.text(), "âncora: o aviso da Correção já tinha frase"

    janela.pipeline.botoes_de_etapa[ETAPAS[0]].click()
    janela.espera_a_piscina(30_000)

    assert len(transporte.rotas) == antes, (
        f"a rota foi chamada apesar da recusa: {transporte.rotas[antes:]}"
    )
    rodape = janela.rotulo_de_desfecho.text()
    aviso = pagina.faixa_da_etapa.frase.text()
    # O marcador era «sobrescrita» antes de a frase passar a nomear o botão; o que se
    # procura é a recusa de `PropostaAusenteError`, e esta é a parte fixa do molde dela.
    assert "a proposta dela não está disponível" in rodape, (
        f"o rodape perdeu a mensagem: {rodape!r}"
    )
    assert rodape in aviso, (
        f"o aviso da Correção nao mostra a recusa: {aviso!r}. Quem clicou no botão da Correção "
        "esta a olhar para ali, e nao para o rodape"
    )


# ---------------------------------------------------------------------------
# O PAINEL 5: a conclusao do multiverso
# ---------------------------------------------------------------------------
#
# O painel do recorte declarava a secao *"5. Multiverso"* FORA do corte, e a primeira das duas
# razoes era bloqueio: *"um painel de mesa nao teria o que exibir no fim"*. A rota passou a existir.
#
# **A segunda razao CONTINUA a valer, e e ela que mantem o disparo fora:** 4.320 celulas sem
# cancelamento. Este painel LE e nao dispara, e o identificador vem de fora da janela.


def _corpo_de_conclusao(n: int = 3, grupos: tuple[str, str] = ("ATLL", "NoHTLV")) -> dict[str, Any]:
    """Um corpo de `200` no formato que a rota serve, com fracoes que DIVERGEM.

    **As fracoes divergem de proposito.** Um corpo em que todas fossem iguais deixaria qualquer
    asercao sobre ordem passar sob qualquer ordenacao — e foi exactamente essa vacuidade que uma
    planta explorou do lado da API. Aqui o painel nao ordena — a rota ordena —, e o que se
    afirma e que ele **preserva** a ordem recebida; fracoes distintas tornam a preservacao
    observavel.
    """
    return {
        "identificador": "abc123",
        "grupos": list(grupos),
        "n_celulas": 12,
        "n_features": n,
        "estabilidade": [
            {
                "feature": f"{i}.0_{100 + i}m/z",
                "frac_significant": 1.0 - i * 0.25,
                "median_qvalue": None if i == n - 1 else 0.01 * (i + 1),
                "median_rank": None if i == n - 1 else float(i + 1),
                "rank_iqr": None if i == n - 1 else 0.0,
                "sign_consistency": None if i == n - 1 else 1.0,
                "n_cells": 0 if i == n - 1 else 12,
            }
            for i in range(n)
        ],
    }


def test_u19_o_painel_do_multiverso_nasce_sem_conclusao_e_declarando_o_recorte(
    qtbot: Any,
) -> None:
    """A ancora do painel, e o REPOUSO declara o que a janela nao faz.

    Sem a ancora, «nao mostrou conclusao» e «mostrou vazio» sao o mesmo estado, na celula
    que o profissional le como resultado.

    > **A segunda metade veio de DESENHAR a tela.** O repouso dizia so «nenhuma conclusao lida», e a
    > captura mostrou um campo, um botao e tres palavras: nada dizia de onde vem o identificador nem
    > que esta janela nao dispara. Escopo negativo ausente le-se como capacidade presente, e
    > a pessoa procuraria o botao de disparar.

    > **AS DUAS ASSERCOES DE ESCOPO NEGATIVO SAIRAM, e a razao e que elas passaram a afirmar o
    > CONTRARIO do estado.** Elas exigiam que o repouso dissesse que a janela nao dispara, porque
    > ela nao disparava; hoje ela dispara, e manter a exigencia
    > obrigaria o painel a declarar uma limitacao que ja nao existe. **Um portao que trava o defeito
    > que ele foi escrito para achar e pior que portao nenhum**, porque reprova a correcao.
    >
    > O que fica no lugar e a mesma propriedade um degrau acima: o repouso tem de dizer **o que
    > fazer** em vez de ficar em tres palavras. A ancora contra «mostrou vazio» — que e a razao
    > original desta funcao — nao muda.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)

    assert painel.rotulo_de_estado.text() == SEM_CONCLUSAO
    assert painel.linhas_mostradas() == ()
    assert painel.campo_do_identificador.text() == ""
    assert "Executar teste de robustez" in SEM_CONCLUSAO, (
        "o repouso nao diz o que fazer; a pessoa ve um campo, botoes e nenhuma instrucao"
    )
    assert "cancelada" in SEM_CONCLUSAO, (
        "o repouso nao diz que a execucao pode ser parada, e e essa a informacao que torna o "
        "clique de disparo aceitavel numa grade de milhares de celulas"
    )


def test_u19_clicar_sem_identificador_escreve_a_frase_e_nao_emite(qtbot: Any) -> None:
    """**O defeito antigo, na direcao certa.** Clicar nao mudava nada, porque o slot voltava
    em silencio; aqui a recusa aparece onde a acao foi.

    As duas metades: nada e emitido, e o painel DIZ por que.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)
    pedidos: list[str] = []
    painel.pedir.connect(pedidos.append)

    painel.botao_de_leitura.click()

    assert pedidos == [], "o painel pediu a conclusao de um identificador vazio"
    assert painel.rotulo_de_estado.text() == SEM_IDENTIFICADOR_DO_MULTIVERSO
    assert "identificador" in painel.rotulo_de_estado.text()


def test_u19_com_identificador_o_painel_emite_o_valor_aparado(qtbot: Any) -> None:
    """O par causal do teste acima: **com** identificador, emite — e o espaco em volta sai.

    Sem esta direcao, um painel que nunca emitisse passaria no teste anterior.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)
    pedidos: list[str] = []
    painel.pedir.connect(pedidos.append)

    painel.campo_do_identificador.setText("  abc123  ")
    painel.botao_de_leitura.click()

    assert pedidos == ["abc123"], (
        f"o painel emitiu {pedidos}: identificador com espaco em volta produz uma rota com espaco, "
        "e o `404` que vem de volta parece identificador errado"
    )


def test_u19_o_200_preenche_a_tabela_na_ordem_recebida_e_declara_os_grupos(qtbot: Any) -> None:
    """As tres propriedades do desfecho de sucesso.

    1. A tabela tem uma linha por feature, na **ordem em que a resposta a entregou** — o painel nao
       reordena, porque a rota ja ordena e duas fontes divergiriam.
    2. `n_cells = 0` chega como *nao medido* e nao como celula vazia: *"medido e deu zero"* e *"nao
       havia o que medir"* pedem acoes opostas.
    3. O par de grupos aparece no cabecalho, senao a tabela e uma fracao sem a populacao em que foi
       medida — e estes arquivos tem quatro classes, logo **seis** pares possiveis.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)
    corpo = _corpo_de_conclusao()

    texto = painel.mostra(200, corpo)

    esperado = [linha["feature"] for linha in corpo["estabilidade"]]
    assert painel.linhas_mostradas() == tuple(esperado), (
        f"a tabela saiu {painel.linhas_mostradas()} e a resposta trouxe {esperado}: o painel "
        "reordenou, e a ordem passa a ter duas fontes"
    )
    assert "ATLL" in texto and "NoHTLV" in texto, (
        f"o cabecalho nao diz o contraste: {texto!r}. Uma fracao sem o par de grupos nao "
        "identifica o que se esta a ler"
    )
    assert "12 células" in texto, f"o cabecalho nao diz sobre quantas celulas: {texto!r}"

    ultima = painel.tabela.rowCount() - 1
    coluna_do_q = [chave for chave, _ in COLUNAS].index("median_qvalue")
    item = painel.tabela.item(ultima, coluna_do_q)
    assert item is not None and item.text() == "não medido", (
        "a feature testada em celula nenhuma apareceu como numero ou como celula vazia; as duas "
        "leem-se como «o valor e zero»"
    )
    coluna_das_celulas = [chave for chave, _ in COLUNAS].index("n_cells")
    contagem = painel.tabela.item(ultima, coluna_das_celulas)
    assert contagem is not None and contagem.text() == "0"


def test_u19_o_painel_nunca_mostra_veredito_sem_o_corte_que_o_produziu(qtbot: Any) -> None:
    """**A propriedade que este portao guarda MUDOU, e a antiga fica escrita.**

    Ele exigia que o painel NAO mostrasse veredito nenhum, com este argumento:

    > *"A omissao e o conteudo. O dominio classifica em `robust`/`fragile`/`absent` com limiares
    > que ninguem validou, e o limiar e decisao em aberto."*

    **O que estava errado nele:** os limiares nao sao escolha do codigo — a especificacao os
    declara, na tabela de *Classificacao de robustez*, junto com o proprio campo `verdict` no
    modelo de dados. O painel passa a servi-los.

    **O que continua verdade, e e o que este portao passa a guardar:** ninguem calibrou `0,80` e
    `0,95` para ESTES dados. Entao a propriedade deixa de ser *"nao mostre o veredito"* e passa a
    ser **"nunca mostre o veredito sem o corte que o produziu"** — o que impede exactamente o
    defeito que a versao anterior temia, sem negar ao profissional a leitura que ele veio buscar.

    Duas direcoes: com `limiares` no corpo o cabecalho declara o corte; sem eles, ele declara a
    AUSENCIA em vez de inventar um numero.
    """
    from refino.multiverse.stability import Verdict

    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)
    rotulos = get_args(Verdict)
    assert len(rotulos) == 3, f"o vocabulario do veredito mudou: {rotulos}"

    # **A ancora de nao-vacuidade vem ANTES e na MESMA funcao**: com `COLUNAS` vazio, o
    # `any(...)` abaixo nao acharia a coluna e a mensagem culparia a peca errada.
    assert COLUNAS, "ancora: a tabela nao declara coluna nenhuma"
    assert any(chave == "verdict" for chave, _ in COLUNAS), (
        "o veredito deixou de ser coluna da tabela: a U7 o serve, e sem ele o profissional volta a "
        "ter de derivar a conclusao das cinco medidas"
    )

    corpo = _corpo_de_conclusao()
    corpo["limiares"] = {"robust_frac": 0.8, "robust_sign": 0.95, "absent_frac": 0.05}
    com_corte = painel.mostra(200, corpo)
    assert "0.8" in com_corte and "0.95" in com_corte, (
        f"o cabecalho mostra veredito e nao declara o corte que o produziu: {com_corte!r}"
    )

    sem_corte = painel.mostra(200, _corpo_de_conclusao())
    assert com_corte != sem_corte, "ancora: o cabecalho nao muda com e sem corte, e o par e vacuo"
    assert "0.8" not in sem_corte, (
        f"o cabecalho inventou um corte que a resposta nao trouxe: {sem_corte!r}"
    )


def test_u19_os_outros_desfechos_limpam_a_tabela_e_dizem_o_que_fazer(qtbot: Any) -> None:
    """**Quatro desfechos alem do `200`, e a tabela e limpa nos quatro.**

    Deixar a conclusao anterior na tela ao lado de uma mensagem de erro afirmaria que ela e desta
    leitura — o estado *stale* ja medido no rotulo de desfecho, onde uma falha deixava o painel
    a exibir o desfecho da chamada ANTERIOR. Silencio e indistinguivel de sucesso; stale **afirma**
    sucesso.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)

    # Primeiro uma conclusao de verdade, para haver o que limpar. Sem ela os quatro casos abaixo
    # passariam sobre uma tabela que ja estava vazia, por vacuidade.
    painel.mostra(200, _corpo_de_conclusao())
    assert painel.linhas_mostradas(), "ancora: a tabela nao foi preenchida, e limpar nao prova nada"

    em_curso = painel.mostra(202, {"feito": 7, "total": 12, "estado": "executando"})
    assert painel.linhas_mostradas() == (), "o `202` deixou a conclusao anterior na tela"
    assert "7" in em_curso and "12" in em_curso, f"o `202` nao diz onde vai: {em_curso!r}"

    painel.mostra(200, _corpo_de_conclusao())
    conflito = painel.mostra(409, {"detail": "o disparo não declarou `group_a` e `group_b`"})
    assert painel.linhas_mostradas() == ()
    assert "group_a" in conflito, (
        f"o `409` nao diz o que redisparar: {conflito!r}. Quem le tem de saber que falta "
        "declarar o par de grupos"
    )

    painel.mostra(200, _corpo_de_conclusao())
    ausente = painel.mostra(
        404, {"detail": "não há execução de teste de robustez com identificador 'x'"}
    )
    assert painel.linhas_mostradas() == ()
    assert "identificador" in ausente

    painel.mostra(200, _corpo_de_conclusao())
    sem_corpo = painel.mostra(500, None)
    assert painel.linhas_mostradas() == ()
    assert "500" in sem_corpo, (
        f"o desfecho sem corpo nao diz o codigo: {sem_corpo!r}. Sem ele o painel afirma «nenhuma "
        "conclusao» sobre uma falha, que e outra coisa"
    )


def test_u19_a_rota_do_resultado_e_uma_fonte_so(qtbot: Any) -> None:
    """A rota e composta por funcao e nao por `f`-string repetida — o molde de `rota_do_progresso`.

    E a asercao e sobre o que a JANELA chama, nao sobre a funcao: o que se quer saber e que o clique
    no painel produz `GET /multiversos/{id}/resultado` e nada mais.
    """
    assert rota_do_resultado("abc123") == "/multiversos/abc123/resultado"

    janela, transporte = _janela_que_retem(qtbot, retidos=0)
    janela.multiverso.campo_do_identificador.setText("abc123")
    janela.multiverso.botao_de_leitura.click()
    janela.espera_a_piscina(30_000)

    assert "GET /multiversos/abc123/resultado" in transporte.rotas, (
        f"a janela nao chamou a rota da conclusao: {transporte.rotas}"
    )


def test_u19_a_resposta_do_multiverso_chega_ao_painel_e_nao_a_outro(qtbot: Any) -> None:
    """**A ligacao de ENTRADA.**

    Antes, os tres `mostra_*` da qualidade existiam, tinham cobertura, e `principal.py` nao os
    chamava — mecanismo correto, verde, e o alvo nao existia. Aqui a asercao e que a resposta com a
    chave `multiverso` chega a ESTE painel: uma chave que caisse no `else` nao produziria erro
    nenhum, e a tabela ficaria vazia para sempre.
    """
    janela, _transporte = _janela_que_retem(qtbot, retidos=0)
    corpo = _corpo_de_conclusao()

    janela._encaminha(
        CHAVE_DO_MULTIVERSO,
        Resposta(codigo=200, ok=True, causa=None, mensagem="", corpo=corpo, conteudo=None),
    )

    assert janela.multiverso.linhas_mostradas() == tuple(
        linha["feature"] for linha in corpo["estabilidade"]
    ), "a resposta do multiverso nao chegou ao painel dele"


def test_u19_o_painel_do_multiverso_nao_importa_nem_chama_o_transporte() -> None:
    """O painel COMPOE e nao chama: quem chama e a janela, pela piscina.

    > **A primeira versao procurava a SUBSTRING `envia` na fonte, e ela REPROVOU** — porque a
    > docstring do modulo contem a frase *"o portao de AST reprova um `envia` fora de `lambda`"*. O
    > instrumento leu a prosa SOBRE a chamada em vez da chamada: o mesmo defeito que
    > o portao do `addStretch` teve. Terceira instancia desta forma no repositorio, e a
    > segunda neste arquivo.

    Entao o oraculo e o IMPORT e a CHAMADA, derivados por AST. Prosa nao conta.
    """
    import ast as _ast

    arvore = _ast.parse(
        (
            Path(__file__).resolve().parents[1]
            / "src"
            / "refino"
            / "desktop"
            / "janela"
            / "multiverso.py"
        ).read_text(encoding="utf-8")
    )

    proibidos = {"envia", "Transporte"}
    importados = {
        alias.asname or alias.name
        for no in _ast.walk(arvore)
        if isinstance(no, _ast.ImportFrom)
        for alias in no.names
    }
    # A ancora vem ANTES: sem ela, uma arvore vazia faria as duas asercoes passarem.
    assert importados, "ancora: nenhum import achado no modulo — a arvore nao foi percorrida"
    assert not (importados & proibidos), (
        f"o painel importa {sorted(importados & proibidos)}: a chamada de rota sairia do laco de "
        "eventos, e a KTD9 exige que ela corra na piscina"
    )

    chamados = {
        no.func.id
        for no in _ast.walk(arvore)
        if isinstance(no, _ast.Call) and isinstance(no.func, _ast.Name)
    }
    assert not (chamados & proibidos), f"o painel CHAMA {sorted(chamados & proibidos)}"


# ---------------------------------------------------------------------------
# OS EIXOS DA GRADE DEIXAM DE SER ESCOLHIDOS POR NOS
# ---------------------------------------------------------------------------
#
# **`FIXOS_DECLARADOS` fixava sete parametros, e quatro eram eixos da `GridDefinition`.** O
# comentario ao lado declarava o problema em vez de o resolver: *"o que faz toda execucao do usuario
# ser uma celula especifica da grade, escolhida por nos e nao por ele"*.
#
# **Servir a conclusao do multiverso tornou isto pior antes de melhor**: o usuario passou a
# poder MEDIR quanto o resultado depende da escolha dos eixos — e continuava sem poder escolhe-los.


def test_u20_os_eixos_da_grade_tem_combo_e_saem_do_dominio() -> None:
    """Os itens de cada combo sao o `Literal` do dominio, nas DUAS direcoes.

    **Rotulo do dominio que falte no combo reprova, e rotulo do combo que o dominio nao declare
    tambem.** A segunda direcao e a que apodrece sem sintoma: uma lista escrita a mao continuaria
    plausivel depois de o dominio mudar, e a janela ofereceria um valor que a rota recusa com 422.

    Este teste NAO constroi widget: ele mede a constante, e por isso corre sem `QApplication`.
    """
    from refino.domain.labels import (
        Imputation,
        RankingFilter,
        SampleNormalization,
        Scaling,
        Transformation,
    )

    esperado = {
        "ranking_filter": get_args(RankingFilter),
        "ranking_filter_2": get_args(RankingFilter),
        "imputation": get_args(Imputation),
        "sample_normalization": get_args(SampleNormalization),
        "transformation": get_args(Transformation),
        "scaling": get_args(Scaling),
    }
    assert len(EIXOS_DA_GRADE) == 6, (
        f"a constante deixou de ter SEIS eixos: {len(EIXOS_DA_GRADE)}. Eram quatro "
        "ate 07/09/2026, `ranking_filter` entrou com a paridade, e `ranking_filter_2` "
        "em 10/09/2026 com o filtro em dois eixos do MetaboAnalyst."
    )

    declarado = {chave: opcoes for chave, _rotulo, _padrao, opcoes in EIXOS_DA_GRADE}
    assert declarado == esperado, (
        f"o vocabulario do combo divergiu do dominio.\n  a mais: "
        f"{ {k: set(declarado.get(k, ())) - set(v) for k, v in esperado.items()} }\n  a menos: "
        f"{ {k: set(v) - set(declarado.get(k, ())) for k, v in esperado.items()} }"
    )
    for chave, rotulo, padrao, opcoes in EIXOS_DA_GRADE:
        assert rotulo.strip(), f"o eixo `{chave}` nao tem rotulo legivel"
        assert padrao in opcoes, f"o padrao de `{chave}` nao esta no vocabulario dele: {padrao}"


def test_u20_nenhum_eixo_da_grade_continua_em_fixos_declarados() -> None:
    """**Derivado de `GridDefinition` e nao de uma lista escrita aqui.**

    O que este teste guarda e a propriedade que a unidade entrega: nenhum eixo da grade e escolhido
    por nos. Um eixo que voltasse a `FIXOS_DECLARADOS` — ou um eixo NOVO na grade que nascesse fixo
    — reprova aqui.

    **`missing_threshold` e a excecao declarada**, com o motivo na constante: e `float | None` e
    precisaria do widget com `specialValueText` que o corte de RSD tem. Ele nao esta em
    `FIXOS_DECLARADOS` nenhum — viaja no padrao do servidor —, entao este teste passa sobre ele por
    construcao, e a declaracao vive na constante e em `docs/revisar-apos-a-noite.md`.
    """
    from dataclasses import fields

    from refino.multiverse.grid import GridDefinition

    # **Os eixos sao MEDIDOS e nao derivados do nome do campo, e a primeira versao deste teste
    # errou por isso.** Ela fazia `p[:-1]` sobre os campos de `GridDefinition` — cirurgia de string
    # —, e o mapa eixo->campo-da-spec nao e esse: `sample_norms` vira `sample_normalization`,
    # `batch_options` vira `n_batches`, `corrections` vira `correction`. O teste acusou
    # `sample_normalization` como «nao e eixo», e ele e.
    #
    # A medicao: uma grade com DOIS valores por eixo, enumerada, e os campos da `PipelineSpec` que
    # de facto variam. Isso e «o que a grade varia» sem passar por nome nenhum.
    grade = GridDefinition(
        corrections=("none", "loess"),
        batch_options=(1, 2),
        rsd_cutoffs=(None, 0.30),
        missing_thresholds=(0.5, 0.8),
        imputations=("one_fifth_min", "median"),
        sample_norms=("none", "sum"),
        transformations=("none", "log10"),
        scalings=("none", "auto"),
    )
    especificacoes = list(grade.enumerate())
    assert len(especificacoes) == grade.size() > 1, (
        "ancora: a grade nao enumerou mais de uma especificacao, e nada varia para ser medido"
    )

    nomes = [f.name for f in fields(especificacoes[0])]
    variam = {nome for nome in nomes if len({getattr(spec, nome) for spec in especificacoes}) > 1}
    assert len(variam) == 8, (
        f"a grade varia {len(variam)} campos da spec, e os eixos sao oito: {sorted(variam)}"
    )

    intrusos = sorted(set(FIXOS_DECLARADOS) & variam)
    assert intrusos == [], (
        f"estes eixos da grade continuam FIXOS: {intrusos}. Toda execucao do usuario seria uma "
        "celula especifica da grade, escolhida por nos"
    )
    # A ancora que impede o teste de passar por vacuidade: os que ganharam combo SAO eixos.
    #
    # **`ranking_filter` e a EXCECAO declarada, e ela e escolha e nao lacuna.**
    # A etapa existe no pipeline e a tela oferece o controle; a grade padrao mantem-na fixa em
    # `none` porque varre-la multiplicaria a grade por oito. Ter combo e ser varrido pela grade
    # deixaram de ser a mesma coisa, e a excecao vai NOMEADA — uma lista vazia aqui faria a
    # asercao voltar a ser «todo combo e eixo», que e falso desde entao.
    # **E `ranking_filter_2` entra na MESMA excecao em 10/09/2026, pelo mesmo argumento.** Os
    # metodos novos entram como escalares e nao como eixo: um segundo eixo de filtro
    # multiplicaria a grade por oito outra vez, e 9.600 celulas viravam 76.800.
    fora_da_grade_por_decisao = {"ranking_filter", "ranking_filter_2"}
    eixos_com_combo = {chave for chave, _r, _p, _o in EIXOS_DA_GRADE}
    assert eixos_com_combo >= fora_da_grade_por_decisao, (
        "a excecao nomeia um eixo que nao tem combo: ela caducou e esconde a asercao"
    )
    assert not fora_da_grade_por_decisao & variam, (
        f"{sorted(fora_da_grade_por_decisao & variam)} passou a ser varrido pela grade: a "
        "excecao deixou de ter razao de ser e tem de sair daqui"
    )
    assert eixos_com_combo - fora_da_grade_por_decisao <= variam, (
        "estes «eixos» com combo nao sao variados pela grade: "
        f"{sorted(eixos_com_combo - fora_da_grade_por_decisao - variam)}"
    )


def test_u20_o_padrao_de_cada_combo_e_o_valor_que_era_fixo(qtbot: Any) -> None:
    """**Quem nao mexer no controle executa o que executava antes desta unidade.**

    Isto e requisito e nao cortesia: mudar o padrao junto com o controle mudaria, em silencio, o
    resultado de quem nao pediu nada. Os valores vem da redacao anterior de `FIXOS_DECLARADOS`,
    citada aqui porque ela ja nao existe no codigo para ser lida.
    """
    antes_da_u20 = {
        "ranking_filter": "none",
        "ranking_filter_2": "none",
        "imputation": "one_fifth_min",
        "sample_normalization": "none",
        "transformation": "none",
        "scaling": "none",
    }
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    assert painel.valores_dos_eixos() == antes_da_u20, (
        f"o padrao dos combos divergiu do que era fixo: {painel.valores_dos_eixos()}"
    )
    # O combo tem de ter DE FACTO aquele item selecionado, e nao um texto qualquer: `setCurrentText`
    # com valor ausente da lista deixa o combo no primeiro item, em silencio.
    for chave, esperado in antes_da_u20.items():
        combo = painel.eixos[chave]
        assert combo.currentText() == esperado
        assert combo.findText(esperado) >= 0, (
            f"`{esperado}` nao esta entre os itens de `{chave}` — `setCurrentText` nao pegou"
        )


def test_u20_mudar_um_eixo_muda_o_payload_eixo_a_eixo(qtbot: Any) -> None:
    """**Par causal POR EIXO, e nao um so.** Quatro combos, quatro asercoes.

    Um teste que mexesse num eixo e afirmasse *"o payload mudou"* passaria com tres dos quatro
    desligados — e tres controles mortos ao lado de um vivo e exactamente o que a tela ja teve. O
    laco corre os quatro e afirma, para cada um, que o valor escolhido chega ao campo com o nome
    dele.

    **A ancora e que o valor novo DIFERE do padrao**: sem ela, escolher o mesmo valor
    produziria payloads iguais e a asercao passaria sem separar nada.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    for chave, _rotulo, padrao, opcoes in EIXOS_DA_GRADE:
        outros = [o for o in opcoes if o != padrao]
        assert outros, f"o eixo `{chave}` tem um valor so — nao ha o que variar"
        novo = outros[0]
        assert novo != padrao, "ancora: o valor novo tem de diferir do padrao"

        antes = campos_da_tela(painel, modo="neg_c18", correcao="none", n_lotes=1)
        assert antes[chave] == padrao, f"o payload nao trazia o padrao de `{chave}`: {antes[chave]}"

        painel.eixos[chave].setCurrentText(novo)
        depois = campos_da_tela(painel, modo="neg_c18", correcao="none", n_lotes=1)
        assert depois[chave] == novo, (
            f"mudar o combo de `{chave}` para `{novo}` nao mudou o payload: {depois[chave]}. O "
            "controle mexe na tela e nao no que o servidor recebe"
        )
        painel.eixos[chave].setCurrentText(padrao)


# **NAO HA teste aqui para «a rota aceita os quatro campos», e a ausencia e deliberada.**
#
# Eu escrevi um, e ele reprovou com `KeyError: 'properties'`: o esquema multipart da OpenAPI usa
# `$ref` para um componente em vez de `properties` em linha, e o meu teste andava pelo dicionario a
# mao. **`tests/test_contrato_janela_api.py` ja resolve isto bem** — `contratos_das_rotas` resolve o
# `$ref` — e a bijecao de la passou a incluir os quatro eixos sozinha, porque eles entram por
# `campos_da_execucao`, que aquele fixture CHAMA.
#
# > Reimplementar pior o que ja existe seria a segunda fonte de verdade sobre o contrato da rota. O
# > teste sai e a cobertura fica declarada onde ela vive.


def test_u20_campos_de_sobrescrita_foi_removido_e_o_que_ficou_esta_ligado() -> None:
    """A remocao, e a metade que FICA — as duas na mesma asercao.

    **`campos_de_sobrescrita` sai.** Ele devolvia os VALORES por eixo, e os eixos ja tem controle
    de valor no painel do pipeline: liga-lo escreveria as mesmas chaves uma segunda vez.

    **A asercao de ausencia vem com a de presenca ao lado**, senao um `hasattr` falso por qualquer
    razao — modulo que nao importa, classe renomeada — passaria como remocao bem feita.

    ## O ACESSOR LIGADO MUDOU DE PAINEL, e a ancora seguiu-o

    Ate aqui a metade de presenca era `PainelDeLeitura.eixos_marcados` — a leitura das checkboxes.
    Elas sairam, e quem alimenta `campos_sobrescritos` passou a ser
    `PainelDePipeline.eixos_divergentes`, que deriva a sobrescrita da diferenca entre o valor
    corrente e a proposta da maquina.

    **A ancora e a mesma propriedade noutro sitio:** se ela desaparecer, `campos_sobrescritos` fica
    sempre vazio e o par de concordancia perde o sinal. E `eixos_marcados` passa a ter de estar
    AUSENTE — deixa-lo la seria um acessor morto a ler um dicionario de widgets que ja nao existe.
    """
    from refino.desktop.janela.leitura import PainelDeLeitura as Painel
    from refino.desktop.janela.pipeline import PainelDePipeline as Pipeline

    assert hasattr(Pipeline, "eixos_divergentes"), (
        "ancora: `eixos_divergentes` desapareceu. Ele e o acessor LIGADO, e sem ele "
        "`campos_sobrescritos` fica sempre vazio e o par de concordancia perde o sinal"
    )
    assert not hasattr(Painel, "campos_de_sobrescrita"), (
        "`campos_de_sobrescrita` continua no painel. Ele nao tem chamador de producao e os valores "
        "dele ja viajam pelo painel do pipeline"
    )
    assert not hasattr(Painel, "eixos_marcados"), (
        "`eixos_marcados` continua no painel da leitura. Ele lia `self.controles`, que saiu com as "
        "checkboxes — um acessor que le um dicionario que ja nao existe"
    )


# ---------------------------------------------------------------------------
# O PAINEL 3, e a CHAMADA QUE FALTAVA
# ---------------------------------------------------------------------------
#
# **O defeito foi medido pelo caminho real antes de uma linha ser escrita.** Sonda de 24/08/2026,
# janela real e transporte que retem, dois cliques:
#
#     (a) «Executar pipeline»  ->  POST /execucoes, rotulo «aceita: 4 etapas»
#                                  pipeline.identificador = None
#     (b) «Ler progresso»      ->  ZERO rotas, e o rotulo:
#                                  «nao ha execucao para ler. Clique em «Executar pipeline»
# primeiro»
#
# A frase contradiz a acao que a precede. A causa: `aceita_disparo` — o UNICO sitio que escreve
# `pipeline._identificador` — tinha zero chamadores de producao, e com ele morriam TRES coisas: o
# identificador, o botao «Ler progresso», e o DECORRIDO, porque e `aceita_disparo` que carimba.


def test_u21_o_disparo_grava_o_identificador_e_sem_disparo_ele_e_none(qtbot: Any) -> None:
    """**O par causal da chamada que faltava**, e a ancora vem primeiro.

    Sem disparo, `identificador` e `None`; com o `202` do `POST` a atravessar `_encaminha`, ele
    passa
    a ser o do corpo. Sem a primeira metade, uma janela que gravasse qualquer coisa sempre passaria
    pela segunda.
    """
    janela, _transporte = _janela_que_retem(qtbot, retidos=0)

    assert janela.pipeline.identificador is None, "ancora: nasceu com identificador"
    assert janela.trabalhos_conhecidos() == {}

    janela._encaminha(
        "execucao",
        Resposta(
            codigo=EM_CURSO,
            ok=True,
            causa=None,
            mensagem="",
            corpo={"identificador": "t-u21", "total": 4, "reclamado": False},
            conteudo=None,
        ),
    )

    assert janela.pipeline.identificador == "t-u21", (
        "o `202` do POST atravessou `_encaminha` e o identificador nao foi gravado — e sem ele "
        "«Ler progresso» diz «nao ha execucao para ler» a quem acabou de disparar"
    )
    assert janela.trabalhos_conhecidos() == {"execucao": "t-u21"}


def test_u21_o_202_de_uma_leitura_nao_e_tratado_como_aceite(qtbot: Any) -> None:
    """O outro lado do discriminador: um `202` de `/trabalhos/` **nao** recarimba o disparo.

    **Tratar uma leitura como aceite faria o decorrido recomecar a cada consulta** — o numero
    plausivel medido do instante errado. O discriminador e `estado`, e
    `ProgressoResposta` declara-o obrigatorio.

    A ancora e que o corpo de leitura TEM `estado` e o de aceite nao: sem ela, os dois corpos podiam
    ser iguais e o teste passaria sob qualquer discriminador.
    """
    aceite = {"identificador": "t-u21", "total": 4, "reclamado": False}
    leitura = {
        "identificador": "OUTRO",
        "estado": "executando",
        "feito": 2,
        "total": 4,
        "fracao": 0.5,
        "terminou": False,
    }
    assert "estado" not in aceite and "estado" in leitura, (
        "ancora: os dois corpos nao diferem no discriminador, e o teste nao separa nada"
    )
    assert e_corpo_de_aceite(EM_CURSO, aceite)
    assert not e_corpo_de_aceite(EM_CURSO, leitura)

    janela, _transporte = _janela_que_retem(qtbot, retidos=0)

    def entrega(corpo: dict[str, Any]) -> None:
        janela._encaminha(
            "execucao",
            Resposta(codigo=EM_CURSO, ok=True, causa=None, mensagem="", corpo=corpo, conteudo=None),
        )

    entrega(aceite)
    assert janela.pipeline.identificador == "t-u21"
    entrega(leitura)
    assert janela.pipeline.identificador == "t-u21", (
        "a leitura foi tratada como aceite: o identificador passou a ser o do corpo de progresso, "
        "e o carimbo do decorrido foi reescrito"
    )


def test_u21_a_rota_do_progresso_sai_sozinha_depois_de_um_disparo(qtbot: Any) -> None:
    """**O defeito medido, virado em portao**, com o par causal do SILENCIO.

    Antes do disparo nao ha espera aberta e nenhuma rota de progresso sai — sondar sem trabalho
    seria rajada sobre um `404`. **Depois** do disparo, `GET /trabalhos/{id}` tem de sair, e desde
    07/09/2026 sem ninguem clicar: o botao «Ler progresso» saiu a pedido de quem usa, e quem chama
    e o relogio de um segundo.
    """
    janela, transporte = _janela_que_retem(qtbot, retidos=0)

    # (a) sem disparo: nao ha espera, e o relogio nao tem o que sondar.
    antes = len(transporte.rotas)
    assert "execucao" not in janela.esperas_em_curso(), (
        "ancora: ha espera aberta sem disparo nenhum"
    )
    # **A metade negativa dispara o tique TAMBEM**, e e ela que separa «nao sondou porque nao ha
    # espera» de «nao sondou porque o relogio nao tocou». Sem espera aberta, `_rele` devolve sem
    # chamar nada — e e isso que se afirma.
    janela._rele("execucao")
    janela.espera_a_piscina(30_000)
    assert len(transporte.rotas) == antes, (
        f"a janela sondou sem haver execucao: {transporte.rotas[antes:]}"
    )

    # (b) com disparo: a rota do progresso sai, com o identificador do aceite.
    janela._encaminha(
        "execucao",
        Resposta(
            codigo=EM_CURSO,
            ok=True,
            causa=None,
            mensagem="",
            corpo={"identificador": "t-u21", "total": 4, "reclamado": False},
            conteudo=None,
        ),
    )
    antes = len(transporte.rotas)
    # O tique vai a mao, pelo mesmo `_rele` que o relogio chama — ver a nota extensa em
    # `test_u11_a_sondagem_da_execucao_chama_a_rota_do_trabalho`. Que o relogio esteja armado
    # depois do aceite e medido em `test_desktop_sondagem.py`.
    janela._rele("execucao")
    janela.espera_a_piscina(30_000)

    assert "GET /trabalhos/t-u21" in transporte.rotas[antes:], (
        f"a janela nao sondou a rota depois do disparo: {transporte.rotas[antes:]}. E o defeito "
        "que a sonda da U21 mediu, agravado: nao ha mais botao nenhum para o usuario clicar"
    )


def test_u21_o_decorrido_aparece_depois_do_disparo(qtbot: Any) -> None:
    """A TERCEIRA coisa que morria com a chamada em falta.

    `aceita_disparo` chama `carimba_disparo`; sem carimbo, `decorrido_da_leitura` devolve `None` e
    `_texto_do_decorrido` devolve string vazia — a linha do decorrido simplesmente nao e
    renderizada.
    O usuario nunca via quanto tempo a execucao levava.
    """
    janela, _transporte = _janela_que_retem(qtbot, retidos=0)

    janela._encaminha(
        "execucao",
        Resposta(
            codigo=EM_CURSO,
            ok=True,
            causa=None,
            mensagem="",
            corpo={"identificador": "t-u21", "total": 4, "reclamado": False},
            conteudo=None,
        ),
    )

    assert "decorrido" in janela.pipeline.rotulo_de_progresso.text(), (
        f"o decorrido nao aparece: {janela.pipeline.rotulo_de_progresso.text()!r}. Sem o carimbo "
        "disparo, `decorrido_da_leitura` devolve `None` e a linha nao e renderizada"
    )


def test_u21_o_painel_lista_os_trabalhos_e_o_vazio_diz_que_nao_ha(qtbot: Any) -> None:
    """Os dois estados do painel, e o vazio e uma FRASE e nao uma lista de zero linhas.

    *"Nao disparei nada"* e *"o painel nao esta a mostrar"* leem-se de forma diferente, e uma lista
    vazia nao os separa.
    """
    painel = PainelDeTrabalhos()
    qtbot.addWidget(painel)

    assert painel.mostra({}) == SEM_TRABALHOS
    assert painel.estados_conhecidos() == {}

    texto = painel.mostra({"corrigir": "id-a", "execucao": "id-b"})
    # O nome legivel no lugar da chave, e o identificador na dica do nome — a linha mostra o
    # que o trabalho E, e o identificador aparece ao passar o ponteiro sobre ele.
    corrigir, execucao = NOMES_DOS_TRABALHOS["corrigir"], NOMES_DOS_TRABALHOS["execucao"]
    assert corrigir in texto and execucao in texto
    assert "id-a" in painel.linhas["corrigir"].nome.toolTip()
    assert "id-b" in painel.linhas["execucao"].nome.toolTip()
    assert NAO_LIDO in texto, (
        "o painel nao diz que ainda nao leu o estado — e «nao lido» e diferente de «em curso»"
    )
    # A ordem e a das chaves e nao a de insercao: duas sessoes com a mesma lista mostram a mesma
    # ordem, e a de insercao dependeria de quem clicou primeiro.
    assert texto.index(corrigir) < texto.index(execucao)


def test_u21_o_painel_declara_que_fechar_encerra_os_trabalhos(qtbot: Any) -> None:
    """**A metade que NAO se cumpre, declarada onde o usuario esta.**

    A API corre dentro do processo da janela, entao fechar mata os trabalhos. A instrucao
    «GUARDE ESTE IDENTIFICADOR» so seria verdadeira onde a API e outro processo, e seria falsa
    aqui. Um painel que a desse afirmaria recuperacao que nao existe.
    """
    painel = PainelDeTrabalhos()
    qtbot.addWidget(painel)

    aviso = painel.rotulo_do_aviso.text()
    assert "FECHAR O APLICATIVO ENCERRA OS TRABALHOS" in aviso
    assert "dentro deste processo" in aviso, (
        "o aviso nao diz POR QUE, e um aviso sem causa le-se como cautela em vez de fato"
    )
    assert "GUARDE ESTE IDENTIFICADOR" not in aviso, (
        "a instrucao da tela web foi portada, e aqui ela e falsa: nao ha o que recuperar depois de "
        "fechar"
    )


def test_u21_o_estado_de_cada_trabalho_vem_do_codigo_e_cobre_os_quatro(qtbot: Any) -> None:
    """**Quatro desfechos e nao dois**, e o `404` e separado do erro de proposito.

    *"Nao conheco este identificador"* e *"o trabalho falhou na etapa X"* pedem acoes opostas —
    redisparar contra corrigir o dado.
    """
    painel = PainelDeTrabalhos()
    qtbot.addWidget(painel)

    em_curso = painel.mostra_estado("a", 202, {"estado": "executando", "feito": 2, "total": 4})
    # Os rotulos legiveis, com a maiuscula.
    assert ROTULOS_DOS_ESTADOS["executando"] in em_curso and "2 de 4" in em_curso

    assert painel.mostra_estado("b", 200, {}) == ROTULOS_DOS_ESTADOS["concluido"] == "Concluído"

    ausente = painel.mostra_estado("c", 404, {"detail": "nao ha trabalho"})
    assert "desconhecido" in ausente.lower()
    assert "falhou" not in ausente.lower(), "o 404 foi colapsado em falha, e as acoes sao opostas"

    falhou = painel.mostra_estado("d", 409, {"detail": "o trabalho falhou na etapa «correção»"})
    assert "falhou" in falhou and "correção" in falhou

    assert len(painel.estados_conhecidos()) == 4, "os quatro estados nao foram todos anotados"


def test_u21_atualizar_sem_trabalho_nao_chama_rota_e_escreve_a_frase(qtbot: Any) -> None:
    """O ramo vazio, e ele e o `else` que dispara por AUSENCIA de condicao.

    Botao que recusa em silencio e indistinguivel de botao morto — o defeito que o usuario relatou.
    """
    janela, transporte = _janela_que_retem(qtbot, retidos=0)
    antes = len(transporte.rotas)

    janela.trabalhos.botao_de_atualizacao.click()
    janela.espera_a_piscina(30_000)

    assert len(transporte.rotas) == antes, (
        f"atualizar sem trabalho chamou rota: {transporte.rotas[antes:]}"
    )
    assert janela.trabalhos.rotulo_da_lista.text() == SEM_TRABALHOS


def test_u21_todo_metodo_publico_de_painel_tem_uso_em_producao_ou_e_declarado() -> None:
    """O PORTAO DA CLASSE, e ele nasce de DOIS metodos mortos achados na mesma varredura.

    ## Por que o portao do prefixo `campos_` nao bastava, e a medicao esta nos numeros

    Aquele olha o prefixo `campos_`, e a sua propria docstring declara a estreiteza: *"nao ha como
    derivar «metodos que compoem payload» de coisa nenhuma"*. **Uma varredura achou dois metodos
    mortos que nao casam o prefixo**, e os dois tinham consequencia visivel:

    | Metodo | O que morria com ele |
    |---|---|
    | `PainelDePipeline.aceita_disparo` | o identificador, «Ler progresso», e o DECORRIDO |
    | `PainelDeLeitura.mostra_justificativa` | as justificativas exigidas em CADA
    ponto |

    ## O oraculo foi MEDIDO em tres versoes, e as duas primeiras nao serviam

    | Oraculo | Acusa | Serve? |
    |---|---|---|
    | «sem chamador em `principal.py`» | **14 de 33** | nao: a maioria e chamada DENTRO do proprio
    painel |
    | «sem chamada `.nome(` em `src/`» | 9 | nao: perde ligacao de sinal, que e `self.nome` sem
    parenteses |
    | **«sem REFERENCIA `.nome` em `src/`»** | **8** | sim, e os 8 sao acessores declarados |

    A terceira conta `.nome` seguido de algo que nao seja identificador — o que pega `self.nome(`,
    `connect(self.nome)` e `self.nome,`. **`aceita_disparo` e `mostra_justificativa` estavam nos
    oito
    antes desta unidade**, e e por isso que a tabela abaixo os NAO contem.

    **A tabela e de acessores, e cada linha diz para que serve.** Acessor lido por teste e legitimo:
    ele existe para o arnes afirmar sobre o painel sem percorrer widget. O que a tabela impede e que
    um metodo com CONSEQUENCIA se esconda entre eles.
    """
    import ast as _ast
    import re as _re

    raiz = Path(__file__).resolve().parents[1] / "src"
    fontes = {
        caminho: caminho.read_text(encoding="utf-8")
        for caminho in sorted(raiz.rglob("*.py"))
        if "__pycache__" not in str(caminho)
    }
    assert fontes, "ancora: nenhuma fonte de producao lida"
    todo = "\n".join(fontes.values())

    # **A tabela de acessores, com a razao de cada um.** Uma linha aqui e afirmacao auditavel.
    acessores = {
        "CaixaDeAutomatico.valores_de_repouso": (
            "o que a caixa REPOE ao voltar a ticar, para o arnes afirmar o desfazer sem "
            "reescrever a travessia dos widgets em cada asercao. **A producao nao o le de "
            "proposito**: quem repoe e o proprio `_aplica`, e um leitor externo seria a segunda "
            "fonte de verdade sobre o valor de repouso"
        ),
        "PainelDeArquivos.origem_do_modo": (
            "de onde veio o modo corrente — o nome do arquivo, os nomes das amostras ou a mao "
            "A tela a mostra pela prosa de `rotulo_do_modo`; o arnes le a origem crua "
            "para separar as tres sem reescrever a prosa em cada asercao"
        ),
        "Aba.nome_da_captura": (
            "o pedaco do nome de arquivo da captura de cada aba. Quem o le e o "
            "script de captura, `scripts/captura_telas_do_guia.py`, que vive fora de `src/` e por "
            "isso nao conta como uso de producao; derivar o nome la, de novo, seria a segunda fonte"
        ),
        "PainelDeArtefatos.salvaveis": (
            "a propriedade SEMANTICA do `[CA-12.14]` — quais passos tem bytes para salvar. A U6 "
            "declarou que e ela, e nao a visibilidade do widget, que o teste afirma: sob "
            "`offscreen` e sem `show()`, `isVisibleTo` devolve `True` para todos os botoes"
        ),
        # **`PainelDeLeitura.controles_do_ponto` SAIU desta tabela**, e quem o tirou foi a
        # segunda direcao deste portao: ele passou a ter chamador de producao — o proprio painel o
        # usa para montar um sub-grupo de caixas por ponto de decisao — e um acessor com uso em
        # producao nao e acessor. **O portao acusou; ninguem se lembrou.**
        # **`PainelDeLeitura.controles_do_ponto` VOLTOU a esta tabela**, e foi
        # a segunda direção deste portão que o mandou de volta: ele saiu por ter ganho
        # chamador de produção — montava a grade de caixas por ponto —, e as caixas saíram. O
        # portão acusou na mesma sessão em que elas saíram.
        #
        # **Ele fica em vez de sair, e a razão é o que ele responde:** *quais eixos este ponto de
        # decisão move?* É a pergunta que `test_u4_todo_ponto_de_decisao_oferece_controle_de_
        # sobrescrita` faz para afirmar o par justificativa-mais-controle, e ela
        # continua a ser feita — o que mudou foi o widget em que o controle vive, não o par. A
        # produção lê a mesma estrutura por `eixos_propostos_do_ponto` e
        # `eixos_so_do_humano_do_ponto`, que são a PARTIÇÃO dela; este acessor é o conjunto
        # inteiro, e é dele que o arnês deriva a bijeção.
        "PainelDeLeitura.controles_do_ponto": (
            "os controles daquele ponto de decisão, o conjunto INTEIRO. A produção usa a partição "
            "dele — o que a máquina propõe e o que fica com o humano —, e o arnês usa o todo para "
            "afirmar que a soma das duas frases da tela é a estrutura, sem reescrever o filtro"
        ),
        # **`PainelDePipeline.proposta_da_maquina` saiu desta lista**: a
        # produção passou a lê-lo — `_aplica_as_sugestoes_da_correcao` diz, com ele, o corretor que
        # o Palpite sugeriu quando a pessoa escolheu outro durante a espera.
        "PainelDeMultiverso.linhas_mostradas": (
            "os nomes de feature que a tabela mostra, para o arnes nao reescrever a travessia de "
            "`QTableWidgetItem` em cada asercao"
        ),
        "PainelDeMultiverso.vereditos_mostrados": (
            "a coluna do VEREDITO, servida desde a U7 — irmao de `linhas_mostradas` e pelo mesmo "
            "argumento. Acessor proprio e nao um indice literal no teste: `COLUNAS` mudou de ordem "
            "na U7 (o veredito entrou em segundo), e um indice fixo no arnes teria passado a ler a "
            "coluna vizinha em silencio, devolvendo texto plausivel"
        ),
        "PainelDeMultiverso.linhas_do_historico": (
            "os identificadores que a tabela do HISTORICO mostra — outra tabela e outra pergunta "
            "que a de `linhas_mostradas`: aquela responde *quais features sobrevivem*, esta "
            "*que execucoes existiram*. Mesmo argumento: o arnes afirma o que o painel MOSTRA sem "
            "reescrever a travessia de `QTableWidgetItem` em cada asercao (item 58)"
        ),
        # **`PainelDeFigura.estado_visivel` e nao `texto_visivel` — o nome e ESCOLHA e nao gosto.**
        # A verificacao de orfaos e por NOME de metodo sobre `src/` inteiro, entao um segundo
        # `texto_visivel` neste pacote nao seria distinguivel do de `PainelDeQualidade`: bastaria
        # um painel chamar o do filho para que a linha do pai passasse a constar como «ganhou uso
        # em producao» e o portao reprovasse pelo motivo errado: o instrumento leria
        # o referente certo e a instancia errada.
        "PainelDeFigura.estado_visivel": (
            "o que o painel da figura MOSTRA — o texto de repouso, o item do seletor, a anotacao "
            "da identificacao, quantos eixos estao na tela e, desde a U3, o rotulo do botao de "
            "exportar e o aviso da exportacao —, cada um vazio quando o widget nao "
            "esta desenhado. A producao nao o le: quem escreve e `define_execucao` e `desenha`, e "
            "um leitor externo seria a segunda fonte sobre o que esta na tela. O arnes le-o para "
            "afirmar as duas metades que a revisao adversarial de `PainelDeQualidade` mediu — o "
            "texto E o layout —, sem reescrever a travessia de widget em cada asercao"
        ),
        "HistoricoMudo.push_current": (
            "nao e acessor: e o protocolo de toolbar do `matplotlib`. Quem a chama e "
            "`backend_bases.scroll_handler`, que `PainelDeFigura._amplia` invoca com um "
            "`HistoricoMudo`; a chamada vive na biblioteca, e esta varredura sobre `src/` nao a ve"
        ),
        "PainelDeQualidade.texto_visivel": (
            "o que cada metrica mostra, para o arnes afirmar os TRES estados de cada uma — valor, "
            "«sem caminho» e «nao calculada» — sem ler widget a widget"
        ),
        "PainelDeRecorte.textos": (
            "os pares que o painel desenha, e e o que o portao de bijecao de "
            "`test_desktop_recorte.py` compara contra `itens_para_a_janela`"
        ),
        "PainelDeTrabalhos.estados_conhecidos": (
            "o ultimo estado lido de cada trabalho, para o arnes afirmar os quatro desfechos sem "
            "voltar a formatar a linha"
        ),
        "JanelaPrincipal.mostra_desfecho": (
            "o caminho de «nao ha `Resposta`», e ele NAO ocorre hoje: `cliente.envia` converte "
            "toda `Exception` em `Resposta`, e o que escapa disso e tratado por `_falha`, com "
            "texto proprio. Fica como o par de `mostra_resposta` — e a docstring dele PROIBE "
            "usa-lo sobre uma `Resposta`, porque perde a distincao das tres chaves de 503"
        ),
    }

    janela = raiz / "refino" / "desktop" / "janela"
    orfaos: list[str] = []
    publicos = 0
    for caminho in sorted(janela.glob("*.py")):
        if caminho.name == "__init__.py":
            continue
        for no in _ast.walk(_ast.parse(fontes[caminho])):
            if not isinstance(no, _ast.ClassDef):
                continue
            for corpo in no.body:
                if not isinstance(corpo, _ast.FunctionDef) or corpo.name.startswith("_"):
                    continue
                # **SOBRESCRITA DO Qt: quem a chama e o laco de eventos, e o `grep` nao o ve.**
                #
                # A casa escreve `snake_case` — o `ruff` reprova o resto com `N802` —, entao um
                # nome em `camelCase` neste pacote **e** um metodo da API do Qt, marcado com
                # `# noqa: N802`. Medido em 07/09/2026: `eventFilter` de `FiltroDaRoda` e a UNICA
                # ocorrencia em `src/refino/`, e sem esta linha ela entraria como orfa.
                #
                # **Derivado e nao isento pelo nome**, que e a diferenca que importa: a proxima
                # sobrescrita — `paintEvent`, `closeEvent` — passa por aqui sozinha, e um metodo
                # NOSSO que ficasse sem chamador continua a reprovar, porque o nome dele e
                # `snake_case`.
                if corpo.name != corpo.name.lower():
                    continue
                publicos += 1
                nome = f"{no.name}.{corpo.name}"
                usos = _re.findall(rf"\.{_re.escape(corpo.name)}(?![A-Za-z0-9_])", todo)
                if not usos and nome not in acessores:
                    orfaos.append(f"{caminho.name}::{nome}")

    assert publicos >= 30, (
        f"ancora: so {publicos} metodos publicos achados — a varredura mediu nada"
    )
    assert orfaos == [], (
        f"estes metodos de painel nao tem sitio de uso em producao nenhum e nao estao declarados "
        f"como acessor: {orfaos}. Foi assim que `aceita_disparo` matou «Ler progresso» e o "
        "decorrido, e que `mostra_justificativa` deixou as justificativas do `[CA-12.6]` em branco"
    )
    # A direcao oposta: acessor que ganhou uso em producao sai da tabela.
    ligados = [
        nome
        for nome in acessores
        if _re.search(rf"\.{_re.escape(nome.split('.')[1])}(?![A-Za-z0-9_])", todo)
    ]
    assert ligados == [], (
        f"estes acessores ganharam uso em producao e continuam declarados como acessor: {ligados}"
    )


def test_u21_a_resposta_da_atualizacao_chega_ao_painel_3_e_nao_ao_do_artefato(qtbot: Any) -> None:
    """**A chave composta e o que impede a atualizacao de reabrir a espera de um artefato.**

    Sem o prefixo `trabalhos@`, uma leitura de `corrigir` cairia no ramo dos artefatos de
    `_encaminha` e reabriria a espera daquele artefato — o defeito de identidade que a chave
    composta existe para nao ter.
    """
    janela, _transporte = _janela_que_retem(qtbot, retidos=0)
    janela.registra_trabalho("corrigir", "id-corrigir")

    janela._encaminha(
        CHAVE_DOS_TRABALHOS,
        Resposta(
            codigo=EM_CURSO,
            ok=True,
            causa=None,
            mensagem="",
            corpo={"estado": "executando", "feito": 1, "total": 3},
            conteudo=None,
        ),
        "corrigir",
    )

    assert janela.trabalhos.estados_conhecidos()["corrigir"].startswith(
        ROTULOS_DOS_ESTADOS["executando"]
    )
    # A contagem esta na barra da linha, e nao mais no rotulo da lista.
    assert "1 de 3" in janela.trabalhos.linhas["corrigir"].barra.text()
    assert janela.esperas_em_curso() == (), (
        "a atualizacao do painel 3 reabriu a espera de um artefato — a chave composta nao separou"
    )


# ---------------------------------------------------------------------------
# AS RESSALVAS DE METODO CHEGAM AO PONTO DE ESCOLHA
# ---------------------------------------------------------------------------
#
# `METHOD_CAVEATS` chegava ao `stderr`, ao `--help`, ao log e a API, e **nao a janela**: as unicas
# ocorrencias dele em `src/refino/desktop/` eram a prosa do painel do recorte a dizer que ele nao
# chegava.
#
# **O `Caveat` separa `resumo` e `detalhe` de proposito**, e a docstring dele diz para que:
#
# > *"`resumo` e o que aparece no ponto de escolha; `detalhe` e o recorte inteiro"*


def test_a_janela_nao_delega_e_o_seletor_do_corretor_fica_editavel(qtbot: Any) -> None:
    """A janela não oferece a delegação, e o seletor do corretor aceita escolha desde a abertura.

    **Era `test_a_janela_oferece_a_delegacao_e_repousa_na_escolha_manual`** — o `--auto` na
    janela, que virou caixa «Automático» ticada por padrão e travava o seletor.
    A caixa saiu, e no lugar dela há um botão «Seguir sugestões do app» no alto da seção de
    entrada. O argumento
    contra delegar por omissão — trocar o corretor de quem não mexeu no combo — vale agora sem
    exceção: o pedido leva o que o seletor mostra. A linha de comando continua a delegar com
    `--auto`, e é lá que a delegação é medida.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    for nome in ("automatico", "quem_decide", "corretor_que_vai_correr"):
        assert not hasattr(painel, nome), f"`{nome}` voltou ao painel"
    assert painel.corretor.isEnabled(), "o seletor do corretor nasce travado"
    for corretor in CORRETORES:
        painel.corretor.setCurrentText(corretor)
        assert painel.corretor.currentText() == corretor, f"o seletor recusou `{corretor}`"


def test_a_ressalva_do_palpite_sai_de_auto_caveats_e_nao_e_reescrita(qtbot: Any) -> None:
    """Atrás de «Ver o recorte medido», a ressalva do Palpite é a de `AUTO_CAVEATS`, **por acesso**.

    Uma segunda redação da ressalva no arnês divergiria da primeira no dia em que o número mudasse,
    e o teste passaria a defender o texto velho. A âncora vem
    primeiro: com o `detalhe` vazio não há ressalva a mostrar, e tudo abaixo passaria por vacuidade.

    **Era `test_a_ressalva_da_politica_sai_de_auto_caveats_e_nao_e_reescrita`**, e a ressalva
    seguia a caixa «Automático» — a da política com ela ticada, `SEM_DELEGACAO` desticada. A caixa
    saiu, e o rótulo mostra sempre a ressalva do Palpite, que é a proposta que «Seguir sugestões do
    app» aplica. Absorveu `test_a_escolha_manual_diz_isso_em_vez_de_ficar_em_branco`: sem escolha
    manual, o rótulo nunca tem um estado em branco a evitar — tem sempre esta frase.
    """
    from refino.cli import AUTO_CAVEATS, AUTO_MEDIDA

    esperado = AUTO_CAVEATS[AUTO_MEDIDA]
    assert esperado.detalhe.strip(), "âncora: o Palpite não tem `detalhe` medido"

    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    assert painel.rotulo_do_detalhe_da_politica.text() == esperado.detalhe, (
        "o detalhe da ressalva do Palpite foi reescrito em vez de vir da estrutura"
    )


def test_mudar_o_corretor_muda_a_ressalva_dele_e_nao_a_do_palpite(qtbot: Any) -> None:
    """**Par causal:** trocar o corretor troca a ressalva do método, e a do Palpite fica.

    A âncora é que as duas escolhas DIFEREM na ressalva do método; sem ela, um rótulo parado
    passaria na segunda metade. **Era `test_mudar_quem_escolhe_muda_a_ressalva_e_o_par_causal
    _separa`**, que trocava a caixa «Automático» e via a ressalva da política mudar; sem a caixa,
    o que muda a ressalva é o corretor, e a do Palpite descreve a proposta, não a escolha.
    """
    from refino.cli import AUTO_CAVEATS, AUTO_MEDIDA

    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.corretor.setCurrentText("none")
    do_none = painel.rotulo_do_detalhe.text()
    painel.corretor.setCurrentText("serrf")
    do_serrf = painel.rotulo_do_detalhe.text()
    assert do_none != do_serrf, "âncora: as duas escolhas têm a mesma ressalva de método"
    assert painel.rotulo_do_detalhe_da_politica.text() == AUTO_CAVEATS[AUTO_MEDIDA].detalhe, (
        "a ressalva do Palpite passou a seguir o seletor"
    )


def _pedido_da_etapa(painel: PainelDePipeline, etapa: int) -> dict[str, Any]:
    """O que a rota da `etapa` recebe da tela sobre a matriz original — o recorte de
    `campos_da_entrada` e `campos_da_rota`, as duas funções que `_dispara_etapa` chama."""
    from refino.desktop.etapas import EstadoDasEtapas, campos_da_rota

    tela = campos_da_tela(painel, modo="neg_c18", correcao=painel.corretor.currentText(), n_lotes=1)
    campos = EstadoDasEtapas().campos_da_entrada(etapa, "original", tela, "matriz-x")
    return campos_da_rota(campos, etapa)


def test_o_payload_da_correcao_leva_o_corretor_do_seletor_e_nunca_auto(qtbot: Any) -> None:
    """**Par causal sobre o dicionário que vai para a rota**, corretor a corretor.

    Cada corretor do seletor chega ao pedido como ele mesmo, e nenhum leva `auto`. A primeira metade
    é a âncora da segunda: um pedido vazio não levaria `auto` e passaria.

    **Era `test_o_payload_do_corrigir_leva_auto_apenas_quando_se_delega`**, que afirmava de
    propósito o desenho da caixa: `auto` presente com ela ticada, ausente desticada. A caixa saiu, e
    a presença deixou de ser alcançável pela janela.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    for corretor in CORRETORES:
        painel.corretor.setCurrentText(corretor)
        pedido = _pedido_da_etapa(painel, ETAPAS[0])
        assert pedido.get("correction") == corretor, pedido
        assert "auto" not in pedido, f"a Correção delegou com `{corretor}` no seletor: {pedido}"


def test_nenhuma_rota_da_janela_leva_auto(qtbot: Any) -> None:
    """**A direção que a tabela `CAMPOS_POR_ARTEFATO` existe para ter**, agora sem exceção.

    Campo não declarado é descartado em silêncio pelo FastAPI, e um `auto` que voltasse a um pedido
    da janela delegaria sem a tela saber. As três etapas e os downloads dos Resultados: nenhum.
    A âncora é que cada pedido tem o corretor, para a ausência não passar sobre um dicionário vazio.

    **Era `test_so_o_corrigir_leva_auto_e_as_outras_duas_rotas_nao`**, com a Correção como a
    exceção que delegava; a exceção saiu com a caixa «Automático».
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    for etapa in ETAPAS:
        pedido = _pedido_da_etapa(painel, etapa)
        assert "correction" in pedido, f"âncora: o pedido da etapa {etapa} veio sem corretor"
        assert "auto" not in pedido, f"a etapa {etapa} recebeu `auto`, e mediria o Palpite"
    assert ARTEFATOS_DA_JANELA, "âncora: nenhum download a conferir"
    for chave in ARTEFATOS_DA_JANELA:
        assert "auto" not in CAMPOS_POR_ARTEFATO[chave], (
            f"`{chave}` declara `auto`, e o arquivo poderia sair de outro corretor"
        )
        assert "auto" not in painel.campos_do_artefato(chave), (
            f"`{chave}` recebeu `auto`, e a rota o descarta em silêncio"
        )


def test_u22_a_ressalva_do_metodo_escolhido_sai_de_method_caveats(qtbot: Any) -> None:
    """O texto e o de `METHOD_CAVEATS`, **por acesso** e nao redigitado.

    **A asercao compara contra a estrutura e nao contra uma frase escrita aqui.** Uma segunda
    redacao da ressalva no teste divergiria da primeira no dia em que o recorte mudasse, e o teste
    passaria a defender o texto velho.

    A ancora vem primeiro: se `METHOD_CAVEATS` estiver vazio, nao ha ressalva a mostrar e tudo
    abaixo
    passaria por vacuidade.

    **O `resumo` DEIXOU de ser desenhado**. Ele continua a sair de
    `cli.py` para o `--help`, o `stderr`, o `--log` e a API; o que esta janela desenha e o
    `detalhe`, atras do botao «ver o recorte medido».
    """
    from refino.cli import METHOD_CAVEATS

    assert METHOD_CAVEATS, "ancora: nao ha ressalva medida em metodo nenhum"
    com_ressalva = sorted(set(METHOD_CAVEATS) & set(CORRETORES))
    assert com_ressalva, (
        f"ancora: nenhum corretor OFERECIDO tem ressalva. Oferecidos: {CORRETORES}; com ressalva: "
        f"{sorted(METHOD_CAVEATS)}"
    )

    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    for metodo in com_ressalva:
        painel.corretor.setCurrentText(metodo)
        esperado = METHOD_CAVEATS[metodo]
        assert esperado.detalhe.strip(), (
            f"ancora: `{metodo}` nao tem `detalhe` medido, e a assercao abaixo passaria por "
            f"vacuidade contra o `resumo` de recurso"
        )
        assert painel.rotulo_do_detalhe.text() == esperado.detalhe, (
            "o detalhe da ressalva foi reescrito em vez de vir da estrutura"
        )


def test_u22_mudar_o_corretor_muda_a_ressalva_e_o_par_causal_separa(qtbot: Any) -> None:
    """**Par causal, e a ancora e que os dois metodos DIFEREM na ressalva.**

    Sem a ancora, escolher dois metodos que ambos nao tem ressalva — ou ambos a mesma — produziria
    textos iguais, e a asercao passaria sob um painel que nunca actualiza. A
    planta tem de cair sobre dado em que as duas coisas diferem.

    **O par passou a separar pelo `detalhe`**, porque o rotulo do `resumo`
    saiu do ponto de escolha a pedido de quem usa.
    """
    from refino.cli import METHOD_CAVEATS

    com = sorted(set(METHOD_CAVEATS) & set(CORRETORES))
    sem = sorted(set(CORRETORES) - set(METHOD_CAVEATS))
    assert com and sem, (
        f"ancora: preciso de um corretor COM ressalva e de um SEM. Com: {com}; sem: {sem}"
    )

    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    painel.corretor.setCurrentText(sem[0])
    texto_sem = painel.rotulo_do_detalhe.text()
    painel.corretor.setCurrentText(com[0])
    texto_com = painel.rotulo_do_detalhe.text()

    assert texto_sem != texto_com, (
        "trocar o corretor nao mudou a ressalva: ela descreve o metodo com que o painel nasceu, e "
        "um recorte medido apresentado sobre o corretor errado e pior que ressalva ausente"
    )
    assert texto_sem == SEM_RESSALVA
    assert texto_com == METHOD_CAVEATS[com[0]].detalhe


def test_u22_metodo_sem_ressalva_diz_isso_em_vez_de_ficar_em_branco(qtbot: Any) -> None:
    """**Rotulo em branco num layout fixo le-se como «nao carregou»**, e nao como «nao ha».

    Aqui o rotulo existe mesmo quando nao ha ressalva, e isso obriga a frase, na celula que o
    profissional le antes de escolher o corretor.

    **REPONTADO para o rotulo do `detalhe`, e nao afrouxado.** Ate aqui havia
    dois rotulos: o do `resumo`, que dizia a frase, e o do `detalhe`, que ficava VAZIO. O primeiro
    saiu do ponto de escolha a pedido de quem usa, e a frase desceu para o segundo — e por isso
    esta asercao passou de *«o detalhe esta vazio»* para *«o detalhe diz que nao ha»*. **A
    propriedade medida e a mesma**, e a superficie e a que resta.
    """
    from refino.cli import METHOD_CAVEATS

    sem = sorted(set(CORRETORES) - set(METHOD_CAVEATS))
    assert sem, f"ancora: todo corretor oferecido tem ressalva; oferecidos {CORRETORES}"

    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.corretor.setCurrentText(sem[0])

    assert painel.rotulo_do_detalhe.text() == SEM_RESSALVA
    assert painel.rotulo_do_detalhe.text().strip(), "o rotulo ficou em branco"


def test_u22_a_ressalva_nasce_com_o_painel_e_nao_espera_um_clique(qtbot: Any) -> None:
    """O painel nasce com a ressalva do metodo INICIAL.

    Sem a chamada na construcao, o rotulo ficaria vazio ate alguem tocar no combo — e quem nao o
    toca executa com o metodo padrao, que e exactamente quem mais precisa de ver a ressalva dele.

    **Le o rotulo do `detalhe`.** O do `resumo` saiu do ponto de escolha.
    """
    from refino.cli import METHOD_CAVEATS

    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    inicial = painel.corretor.currentText()

    assert painel.rotulo_do_detalhe.text().strip(), (
        f"o painel nasceu sem ressalva nenhuma para `{inicial}`"
    )
    esperado = METHOD_CAVEATS[inicial].detalhe if inicial in METHOD_CAVEATS else SEM_RESSALVA
    assert painel.rotulo_do_detalhe.text() == esperado


def test_d459_o_botao_do_recorte_e_o_que_revela_a_ressalva_inteira(qtbot: Any) -> None:
    """**Par causal do botao «ver o recorte medido»: fechado esconde, aberto revela.**

    ## Por que ele nasce agora, e o motivo e uma AUSENCIA medida

    O botao existe desde 03/09/2026 e tem duas ligacoes `toggled`. **Nenhum teste da suite as
    exercitava**: medido em 15/09/2026, `botao_do_detalhe` ocorria em `tests/` uma unica vez, em
    `test_desktop_aparencia.py`, e so para afirmar que ele tem moldura, icone e `isCheckable()`.
    Ver que um botao E alternavel nao e ver que alternar FAZ alguma coisa.

    **E ele passou a carregar sozinho o que dois rotulos carregavam.** Antes a ressalva estava
    tambem no ponto de escolha, sempre visivel; depois que ela saiu de la, este botao e a UNICA
    superficie da
    janela onde ela aparece. Se a ligacao partir, a ressalva desaparece do produto **em silencio**,
    e nada mais reprova.

    ## A ancora, na mesma funcao e antes

    Os dois rotulos tem texto nao vazio antes de o botao ser tocado. Sem ela, um painel que
    escrevesse `""` nos dois passaria o lado «fechado» por vacuidade, e o lado «aberto» mediria a
    visibilidade de um rotulo em branco — que e o desfecho de sucesso mais convidativo que existe.
    """
    from refino.cli import AUTO_CAVEATS, METHOD_CAVEATS

    com_ressalva = sorted(set(METHOD_CAVEATS) & set(CORRETORES))
    assert com_ressalva, f"ancora: nenhum corretor OFERECIDO tem ressalva; oferecidos {CORRETORES}"
    assert AUTO_CAVEATS, "ancora: nao ha ressalva medida em politica nenhuma"

    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.corretor.setCurrentText(com_ressalva[0])
    # A caixa «Automático» era ticada aqui para o rótulo da política ter texto; desde a saída
    # dela, ele tem sempre a ressalva do Palpite.

    rotulos = {
        "o metodo": painel.rotulo_do_detalhe,
        "a politica": painel.rotulo_do_detalhe_da_politica,
    }
    for nome, rotulo in rotulos.items():
        assert rotulo.text().strip(), (
            f"ancora: o rotulo do detalhe d{nome} veio VAZIO, e o par abaixo mediria a "
            f"visibilidade de um branco"
        )

    # **Fechado.** O painel nasce assim, e e o estado em que o profissional o encontra.
    assert not painel.botao_do_detalhe.isChecked(), "o botao devia nascer fechado"
    for nome, rotulo in rotulos.items():
        assert not rotulo.isVisibleTo(painel), (
            f"o detalhe d{nome} ja esta a mostrar com o botao fechado — o botao deixou de esconder"
        )

    # **Aberto.**
    painel.botao_do_detalhe.setChecked(True)
    for nome, rotulo in rotulos.items():
        assert rotulo.isVisibleTo(painel), (
            f"abrir «ver o recorte medido» nao revelou o detalhe d{nome} — este botao e "
            f"a UNICA superficie da janela onde a ressalva aparece, e a ligacao `toggled` partiu"
        )

    # **E fecha outra vez**, porque um revelador que so abre e um rotulo com um clique a mais.
    painel.botao_do_detalhe.setChecked(False)
    for nome, rotulo in rotulos.items():
        assert not rotulo.isVisibleTo(painel), f"o detalhe d{nome} nao voltou a esconder"


def test_u22_a_janela_nao_reescreve_o_texto_da_ressalva() -> None:
    """**O texto sai de `METHOD_CAVEATS` e nao e redigitado no modulo da janela.**

    Duas redacoes da mesma ressalva divergem, e o que viaja para o texto e a frase — o numero medido
    fica fora. O oraculo e que nenhum trecho longo do `resumo` ou do `detalhe` apareca
    literal na fonte do painel: ele os LE, nao os contem.
    """
    from refino.cli import METHOD_CAVEATS

    fonte = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "refino"
        / "desktop"
        / "janela"
        / "pipeline.py"
    ).read_text(encoding="utf-8")

    assert METHOD_CAVEATS, "ancora: nao ha ressalva a conferir"
    for nome, ressalva in METHOD_CAVEATS.items():
        for campo, texto in (("resumo", ressalva.resumo), ("detalhe", ressalva.detalhe)):
            # Os primeiros 40 caracteres bastam para apanhar uma copia, e nao apanham uma palavra
            # solta que a prosa do modulo use por coincidencia.
            trecho = texto[:40]
            assert trecho not in fonte, (
                f"o {campo} da ressalva de `{nome}` esta LITERAL em `pipeline.py`: ele foi "
                f"reescrito em vez de lido. Trecho: {trecho!r}"
            )


# ---------------------------------------------------------------------------
# O DOWNLOAD HONRA O CORRETOR, e ja nao honrou
# ---------------------------------------------------------------------------
#
# **Medido pela janela, no `neg_c18`, com o combo em `loess`:**
#
#     o que a janela enviava   : ['mode']
#     POST sem `method`        : 2.367.696 bytes   sha 198b1bf7...
#     POST com `method=loess`  : 2.330.291 bytes   sha 4c5b3782...
#
# O download entregava o arquivo de `none` qualquer que fosse o corretor escolhido — e 2.367.696 e
# exactamente o numero que o comentario de `LIMITE_DE_TRABALHOS_RETIDOS` registra para
# `refino corrigir --method none`.
#
# **O sintoma era invisivel por construcao:** `chave_da_entrada` INCLUI o metodo, entao trocar o
# corretor descartava os bytes e pedia de novo — e a rota devolvia o mesmo arquivo. A tela mostrava
# atividade e o conteudo nao mudava.


def test_u23_o_pedido_do_artefato_leva_o_corretor_escolhido(qtbot: Any) -> None:
    """**O par causal do defeito**, e a ancora e que os dois corretores diferem.

    Sem a ancora, escolher o mesmo valor duas vezes produziria payloads iguais e a asercao passaria
    sob uma janela que nunca envia o metodo — que era exactamente o estado anterior.

    **O corretor «escolhido» e o da Correcao FEITA desde 09/11/2026.** O download vivia na aba
    «Executar e exportar» e levava o do combo; a aba saiu, e o download da Correcao leva o corretor
    efetivo da etapa feita — o combo pode ter mudado desde entao, e o arquivo e o que a etapa
    mostrou. Que o combo NAO decide esta em
    `test_resultado_da_correcao_baixa_a_correcao_feita_e_nao_o_combo`; aqui fica o par causal de
    que o corretor viaja.
    """
    janela, _transporte = _janela_que_retem(qtbot, retidos=0)
    assert len(CORRETORES) >= 2, f"ancora: so ha {len(CORRETORES)} corretor(es)"

    enviados: list[dict[str, Any]] = []
    original = janela.chama

    def espia(metodo: str, rota: str, **kwargs: Any) -> None:
        enviados.append({"rota": rota, "campos": dict(kwargs.get("campos") or {})})
        original(metodo, rota, **kwargs)

    janela.chama = espia  # type: ignore[method-assign]

    for corretor in CORRETORES[:2]:
        _com_a_correcao_feita(janela, corretor)
        janela.resultado_da_correcao.gerar.emit("corrigida")
        janela.espera_a_piscina(30_000)

    pedidos = [e for e in enviados if e["rota"].endswith("/corrigir")]
    assert len(pedidos) == 2, f"esperava dois POST de corrigir: {[e['rota'] for e in enviados]}"
    metodos = [p["campos"].get("method") for p in pedidos]
    assert metodos == list(CORRETORES[:2]), (
        f"o pedido do artefato levou {metodos} e os corretores escolhidos foram "
        f"{list(CORRETORES[:2])}. Sem o `method`, a rota corrige com o padrao dela — `none` — e o "
        "profissional leva a matriz por corrigir a acreditar que esta corrigida"
    )


def test_u23_cada_artefato_leva_so_o_que_a_rota_dele_aceita(transporte_real: Any) -> None:
    """**Bijecao contra a OpenAPI, por artefato.** Campo nao declarado e descartado em silencio.

    As tres rotas aceitam conjuntos DIFERENTES — medido: `corrigir` nao aceita `imputation`, e
    `exportar` nao aceita nem `method`. Enviar a uniao as tres faria o payload afirmar uma
    influencia
    que a rota nao tem, e o FastAPI nao acusaria: ele descarta.

    A ancora vem primeiro: se o esquema nao trouxer campo nenhum, a contencao passaria por
    vacuidade.
    """
    esquema = transporte_real.get("/openapi.json").json()

    def aceitos(rota: str) -> set[str]:
        alvo = esquema["paths"][rota]["post"]["requestBody"]["content"]["multipart/form-data"][
            "schema"
        ]
        if "$ref" in alvo:
            alvo = esquema["components"]["schemas"][alvo["$ref"].rsplit("/", 1)[-1]]
        return set(alvo.get("properties", {}))

    assert CAMPOS_POR_ARTEFATO, "ancora: o mapa esta vazio"
    # Todos os passos, os do Resultado das etapas inclusive: o da Normalizacao manda os eixos ao
    # `exportar`, e e aqui que se ve que a rota os aceita.
    for passo in PASSOS_DOS_ARTEFATOS:
        declarados = set(CAMPOS_POR_ARTEFATO.get(passo.chave, ()))
        pela_rota = aceitos(passo.rota)
        assert pela_rota, f"ancora: a rota {passo.rota} nao declara campo nenhum"
        assert declarados <= pela_rota, (
            f"o painel manda a `{passo.rota}` campos que ela NAO declara: "
            f"{sorted(declarados - pela_rota)}. O FastAPI descarta em silencio, e o payload passa "
            "a afirmar uma influencia que a rota nao tem"
        )


def test_u23_o_mapa_cobre_os_tres_artefatos_e_o_exportar_leva_os_dois_dele() -> None:
    """A bijecao entre o mapa e `ARTEFATOS`, nas duas direcoes.

    > **A tupla do `exportar` era VAZIA e a assercao era `== ()`, com a razao — verdadeira a data —
    > de que *"ele aceita `rsd_variant` e `max_features`, e este painel nao controla nenhum dos
    > dois"*. **O painel passou a controlar os dois**, e o `max_features` e o limite de 5.000 do
    > MetaboAnalyst: fixa-lo fazia a ferramenta reproduzir, por omissao, o corte cego que ela veio
    > substituir.
    >
    > **E em 30/08/2026 entraram `method` e `n_batches`**, porque a rota passou a CORRIGIR antes de
    > exportar. Sem eles o profissional escolhia `serrf` no combo e levava a matriz crua — o
    > defeito ja medido para `corrigir` e `processar`, vivo no terceiro artefato.
    >
    > A propriedade guardada nao mudou — o mapa e os artefatos bijectam, e o `exportar` leva
    > exactamente o que a ROTA dele aceita. O que mudou e qual e esse conjunto.

    > **Desde 08/11/2026 a bijecao e com TODOS os passos de artefato**, os do Resultado da Correcao
    > e da Normalizacao inclusive: os dois sairam do mesmo mapa de `artefatos.py`, e `ARTEFATOS`
    > passou a ser so o recorte da aba «Executar e exportar». A propriedade continua a mesma —
    > nenhum passo sem campos, nenhum campo sem passo.

    > **E desde 09/11/2026 os passos sao SO os dos Resultados**: a aba «Executar e exportar» saiu,
    > e com ela as tres chaves dos passos de `interface/fluxo.py`. O «exportar» deste teste e o
    > download da Correcao no formato MetaboAnalyst, que serve-se de `/matrizes/exportar` com a
    > tupla dele;
    > e cada chave declara a etapa dona, que e a aba onde o download mora.
    """
    chaves = {passo.chave for passo in PASSOS_DOS_ARTEFATOS}
    assert chaves, "ancora: nenhum passo de artefato, e a bijecao abaixo passaria vazia"
    assert not {passo.chave for passo in ARTEFATOS} & chaves, (
        "uma chave dos passos de `ui/fluxo.py` voltou a ser download da janela"
    )
    assert {ARTEFATOS_DA_JANELA[chave].etapa for chave in chaves} == {1, 2}, (
        "os downloads deixaram de morar so nos Resultados da Correcao e da Normalizacao"
    )
    assert set(CAMPOS_POR_ARTEFATO) == chaves, (
        f"o mapa e os artefatos divergem: a mais {sorted(set(CAMPOS_POR_ARTEFATO) - chaves)}, "
        f"a menos {sorted(chaves - set(CAMPOS_POR_ARTEFATO))}"
    )
    # **A bijecao contra a OpenAPI de cada rota vive em `test_contrato_janela_api.py`**, que e
    # o portao de contrato; aqui o que se afirma e o CONJUNTO, para que ele nao mude sem
    # alguem olhar. Duplicar a leitura do esquema aqui criaria a segunda fonte que aquele
    # portao existe para nao ter.
    assert ARTEFATOS_DA_JANELA["corrigida_metaboanalyst"].base == "exportar"
    assert set(CAMPOS_POR_ARTEFATO["corrigida_metaboanalyst"]) == {
        "max_features",
        "rsd_variant",
        "method",
        "n_batches",
        # A exportação corrige antes de serializar, então ela
        # corre um corretor — e o regime de núcleos alcança-a como às outras duas.
        "paralelo",
        # Os cortes de lote que o diagnóstico propôs, pela mesma razão — a
        # rota os declara, e sem eles o arquivo exportado era corrigido em blocos iguais.
        "batch_cuts",
        # A semente da execução, para o resumo metodológico do arquivo a
        # dizer a partir do pedido. O valor é o padrão que a rota já usava, e o arquivo não muda.
        "seed",
    }, (
        "o `exportar` leva os campos que a rota dele aceita e este painel controla: "
        f"{CAMPOS_POR_ARTEFATO['corrigida_metaboanalyst']}"
    )
    # **`method` no `exportar` e o que fecha o fluxo**, e a assercao e de PRESENCA porque a de
    # ausencia — «nao manda campo a mais» — e satisfeita por tupla vazia.
    assert "method" in CAMPOS_POR_ARTEFATO["corrigida_metaboanalyst"], (
        "sem `method` o painel exporta a matriz CRUA com o corretor escolhido na tela, e o "
        "profissional leva o arquivo por corrigir a acreditar que esta corrigido"
    )


def test_u23_o_corte_e_os_eixos_so_vao_para_quem_os_aceita(qtbot: Any) -> None:
    """O corte de RSD e os quatro eixos vao ao `processar` e **nao** ao `corrigir`.

    `/matrizes/corrigir` corrige e mais nada; mandar-lhe `scaling` seria pedir uma etapa que ela nao
    corre. As duas direcoes na mesma asercao: presente num, ausente no outro.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.corte.setValue(0.30)
    painel.corretor.setCurrentText(CORRETORES[-1])

    do_corrigir = painel.campos_do_artefato("corrigida")
    do_processar = painel.campos_do_artefato("normalizada")

    assert do_corrigir.get("method") == CORRETORES[-1]
    assert "rsd_cutoff" not in do_corrigir, (
        "o corte foi para `corrigir`, que nao filtra nada — a rota descartaria em silencio"
    )
    assert "scaling" not in do_corrigir
    assert do_processar.get("rsd_cutoff") == 0.30
    assert do_processar.get("scaling") == "none"
    # **O `exportar` leva o que a rota dele aceita, e o corretor.** A
    # assercao ja foi `== {}` e depois `== {"max_features", "rsd_variant"}`; a propriedade guardada
    # nunca mudou — cada artefato leva exactamente o que a rota dele aceita. O que muda e o
    # conjunto, e por isso a assercao e sobre o CONJUNTO e nao sobre membros soltos.
    #
    # **`n_batches` e `seed` passaram a viajar SEMPRE**, para o resumo
    # metodológico do arquivo dizer a partir do pedido os lotes e a semente com que ele foi
    # corrigido. O vazio do número de lotes viaja como `LOTES_QUANDO_VAZIO`, que é o padrão da
    # rota e o que `/execucoes` já recebia; a semente é a de `FIXOS_DECLARADOS`, também o padrão
    # da rota. Até aqui esta nota dizia que `n_batches` só viajava quando escolhido.
    do_exportar = painel.campos_do_artefato("corrigida_metaboanalyst")
    # **`paralelo` aparece SEMPRE**, pelo mesmo motivo de `max_features` e
    # `rsd_variant`: a caixa tem valor por construção, e a rota tem padrão. Não há vazio a traduzir.
    assert set(do_exportar) == {
        "max_features",
        "rsd_variant",
        "method",
        "paralelo",
        "n_batches",
        "seed",
    }, do_exportar
    assert (do_exportar["n_batches"], do_exportar["seed"]) == (
        LOTES_QUANDO_VAZIO,
        FIXOS_DECLARADOS["seed"],
    ), do_exportar
    assert do_exportar["max_features"] == 5000, "o padrao deixou de ser o que a rota ja usava"
    # **A assercao INVERTEU em 30/08/2026, e a inversao e o achado.** Ela dizia `"method" not in
    # do_exportar`, com a razao *"o corretor foi para a exportacao, que nao corrige"* — verdadeira
    # a data. A exportacao passou a corrigir, porque o fluxo do laboratorio sobe o arquivo
    # CORRIGIDO; sem esta linha o painel voltaria a exportar a matriz crua com o corretor
    # escolhido na tela, que e o defeito ja medido, no terceiro artefato.
    assert do_exportar["method"] == CORRETORES[-1], (
        "o corretor escolhido na tela nao chega a exportacao: o profissional escolhe `serrf`, "
        "clica «Gerar o arquivo no formato MetaboAnalyst», e leva a matriz CRUA"
    )
    assert "rsd_cutoff" not in do_exportar, (
        "o corte foi para `exportar`, que nao filtra por RSD escolhido — ela corta por RSD "
        "crescente so quando excede `max_features`, que e outra coisa"
    )
    assert "scaling" not in do_exportar, (
        "um eixo do pipeline foi para a exportacao, que nao processa"
    )


def test_u23_o_vazio_dos_dois_campos_numericos_nao_viaja(qtbot: Any) -> None:
    """O mesmo argumento de `campos_da_execucao`: o minimo significa *"nao escolhi"*.

    Enviar `0` no corte pede um `422` — a rota exige fracao em `(0, 1]` —, e no numero de lotes
    significaria zero lotes. A ambiguidade e resolvida na apresentacao.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    # Em repouso os dois estao no minimo, que o Qt mostra como `specialValueText`.
    assert painel.corte.value() == painel.corte.minimum()
    assert painel.n_lotes.value() == painel.n_lotes.minimum()

    campos = painel.campos_do_artefato("normalizada")

    assert "rsd_cutoff" not in campos, f"o corte vazio viajou: {campos}"
    # **O número de lotes vazio viaja como o padrão declarado, e nunca como o mínimo do campo.**
    # Antes ele não viajava; passou a viajar para o resumo metodológico do
    # arquivo o dizer a partir do pedido. O `0` do mínimo continua a não sair daqui.
    assert campos.get("n_batches") == LOTES_QUANDO_VAZIO, f"o numero de lotes vazio: {campos}"
    assert painel.n_lotes.minimum() != LOTES_QUANDO_VAZIO, "âncora: o mínimo e o padrão coincidem"
    assert campos.get("method"), "ancora: o metodo tem de viajar sempre, e nao viajou"


# ---------------------------------------------------------------------------------------------
# A ressalva do EIXO da grade, e a recusa local da combinacao proibida.
#
# Os dois consertos vem de uma medicao sobre os quatro seletores: os 21 valores fazem efeito, e
# dois deles precisavam de coisa que a tela nao dava.
# ---------------------------------------------------------------------------------------------


def test_d335_o_eixo_com_ressalva_ganha_rotulo_e_os_outros_nao(qtbot: Any) -> None:
    """So os eixos com ressalva medida ganham rotulo, e a particao vem da CONSTANTE.

    **A ancora e dupla e as duas direcoes importam.** Se `RESSALVAS_POR_EIXO` tivesse ressalva em
    todos, a metade «os outros nao» seria vacua; se nao tivesse em nenhum, a metade «ganha rotulo»
    seria vacua. A asercao afirma a particao, nao um nome de eixo — eixo novo com ressalva entra
    sem tocar aqui.
    """
    from refino.desktop.janela.pipeline import RESSALVAS_POR_EIXO

    com = sorted(e for e, r in RESSALVAS_POR_EIXO.items() if r)
    sem = sorted(e for e, r in RESSALVAS_POR_EIXO.items() if not r)
    assert com and sem, (
        f"ancora: preciso de um eixo COM ressalva e de um SEM. Com: {com}; sem: {sem}"
    )

    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    assert set(painel.ressalvas_dos_eixos) == set(com), (
        "os eixos com rotulo de ressalva nao sao os que tem ressalva medida"
    )
    for eixo in sem:
        assert eixo not in painel.ressalvas_dos_eixos, (
            f"o eixo {eixo!r} nao tem ressalva medida e ganhou rotulo: um «sem ressalva medida» "
            "sob um seletor que nunca tera ressalva e ruido, e ruido esconde o rotulo do eixo "
            "que importa"
        )


def test_d335_mudar_o_valor_do_eixo_muda_a_ressalva_e_o_par_causal_separa(qtbot: Any) -> None:
    """**Par causal, com a ancora de que os dois valores DIFEREM na ressalva.**

    Sem a ancora, escolher dois valores ambos sem ressalva daria textos iguais e a asercao
    passaria sob um painel que nunca actualiza.
    """
    from refino.desktop.janela.pipeline import EIXOS_DA_GRADE, RESSALVAS_POR_EIXO

    eixo = next(e for e, r in RESSALVAS_POR_EIXO.items() if r)
    opcoes = next(o for chave, _, _, o in EIXOS_DA_GRADE if chave == eixo)
    com = sorted(set(RESSALVAS_POR_EIXO[eixo]) & set(opcoes))
    sem = sorted(set(opcoes) - set(RESSALVAS_POR_EIXO[eixo]))
    assert com and sem, (
        f"ancora: o eixo {eixo!r} precisa de um valor COM ressalva e de um SEM. "
        f"Com: {com}; sem: {sem}; oferecidos: {opcoes}"
    )

    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    rotulo = painel.ressalvas_dos_eixos[eixo]

    painel.eixos[eixo].setCurrentText(sem[0])
    texto_sem = rotulo.text()
    painel.eixos[eixo].setCurrentText(com[0])
    texto_com = rotulo.text()

    assert texto_sem != texto_com, (
        "trocar o valor do eixo nao mudou a ressalva: ela descreve o valor com que o painel "
        "nasceu, e um recorte medido apresentado sobre o valor errado e pior que ausente"
    )
    assert texto_sem == SEM_RESSALVA
    assert texto_com and texto_com != SEM_RESSALVA


def test_d335_a_ressalva_do_missforest_diz_o_custo_e_sai_da_constante(qtbot: Any) -> None:
    """O texto vem de `IMPUTATION_CAVEATS` **por acesso**, nao redigitado aqui.

    Uma segunda redacao no arnes divergiria da primeira no dia em que a medicao mudasse, e o teste
    passaria a defender o numero velho.
    """
    from refino.cli import IMPUTATION_CAVEATS

    assert "missforest" in IMPUTATION_CAVEATS, (
        "ancora: o `missforest` perdeu a ressalva de custo; ele e o unico valor da grade cujo "
        "tempo torna a janela indistinguivel de travada"
    )
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.eixos["imputation"].setCurrentText("missforest")

    escrito = painel.ressalvas_dos_eixos["imputation"].text()
    assert IMPUTATION_CAVEATS["missforest"].resumo in escrito
    assert IMPUTATION_CAVEATS["missforest"].detalhe in escrito


def test_d335_a_ressalva_nasce_com_o_painel_e_nao_espera_uma_troca(qtbot: Any) -> None:
    """O painel nasce com a ressalva do valor INICIAL do eixo.

    Ligar so o `currentTextChanged` deixaria o rotulo vazio ate a primeira troca, e vazio le-se
    como «nao ha ressalva» — que e falso para quem nascer com um valor que tem uma.
    """
    from refino.desktop.janela.pipeline import RESSALVAS_POR_EIXO

    eixo = next(e for e, r in RESSALVAS_POR_EIXO.items() if r)
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    assert painel.ressalvas_dos_eixos[eixo].text().strip(), (
        "o rotulo da ressalva nasceu em branco: ele so seria preenchido na primeira troca"
    )


def test_d436_a_combinacao_proibida_e_recusada_tambem_no_caminho_do_ARTEFATO(qtbot: Any) -> None:  # noqa: N802
    """A IRMÃ da guarda de baixo, que faltava.

    ## O buraco, medido

    `campos_da_execucao` recusava a combinação e `campos_do_artefato` **não**, e os dois mandam os
    mesmos seis eixos da grade: quem escolhesse `imputation=none` com um escalonamento ativo e
    clicasse em «Gerar e baixar a matriz processada» recebia o `422` da rota — correto, e na língua
    da rota, que cita identificadores internos e `sample_normalization` a quem clicou num botão.

    > **«Quem mais faz esta mesma coisa?»** A guarda nasceu num caminho e o outro caminho já
    > existia. Extraí-la para método é o que impede o terceiro de nascer sem ela.

    **A âncora é a última asserção:** sem ela, uma guarda que levantasse para QUALQUER estado
    passaria nas duas primeiras.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    # Até ali um laço desticava aqui as caixas «Automático» do painel, que travavam os eixos.
    painel.eixos["imputation"].setCurrentText("none")
    painel.eixos["scaling"].setCurrentText("auto")

    with pytest.raises(CombinacaoProibidaError) as capturado:
        painel.campos_do_artefato("normalizada")

    assert "Escalonamento" in str(capturado.value), (
        f"a recusa não nomeia o eixo que a causou: {capturado.value}"
    )

    # **A guarda é CONDICIONAL, e estes dois são a âncora do escopo.** A primeira versão recusava
    # os três artefatos, e `corrigir` e `exportar` não levam eixo nenhum: a combinação proibida é
    # sobre normalizar sem imputar, e onde não se normaliza ela não existe. Recusar ali bloquearia
    # um gesto legítimo com uma frase sobre um eixo que aquele pedido não carrega.
    for chave in ("corrigida", "corrigida_metaboanalyst"):
        assert painel.campos_do_artefato(chave), (
            f"«{chave}» não leva os eixos da grade e foi recusado na mesma — a guarda ficou mais "
            f"larga que a população dela"
        )

    # A âncora do outro lado: com a imputação escolhida, o MESMO caminho passa. Sem esta metade,
    # uma guarda que levantasse sempre satisfaria a asserção de cima.
    painel.eixos["imputation"].setCurrentText("one_fifth_min")
    campos = painel.campos_do_artefato("normalizada")
    assert campos.get("scaling") == "auto", (
        f"com a imputação escolhida o caminho tem de passar, e os eixos viajam: {campos}"
    )


def test_d335_imputacao_none_com_normalizacao_ativa_e_recusada_localmente(qtbot: Any) -> None:
    """A combinacao proibida recusada **antes** da rota, na lingua de quem clicou.

    A rota tambem a recusa e recusa bem; o que ela nao faz e dize-lo sem citar identificadores
    internos, `unassessable` e `sample_normalization`. Medido no congelado, era isso que aparecia
    na tela.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.eixos["imputation"].setCurrentText("none")
    painel.eixos["sample_normalization"].setCurrentText("sum")

    with pytest.raises(CombinacaoProibidaError) as capturado:
        campos_da_tela(
            painel, modo="neg_c18", correcao="none", n_lotes=1, eixos=(), recomendacao=None
        )

    mensagem = str(capturado.value)
    assert "Normalização por amostra" in mensagem, (
        "a mensagem nao nomeia o seletor que a pessoa tem de mexer, e ela ve quatro seletores"
    )
    assert re.search(r"\bD\d{2,3}\b", mensagem) is None, (
        f"a recusa local escreveu um código interno do repositório: {mensagem!r}"
    )
    for jargao in ("unassessable", "sample_normalization", "PipelineSpec"):
        assert jargao not in mensagem, (
            f"a recusa local escreveu {jargao!r} — e linguagem de repositorio na tela de quem "
            "opera, que e exactamente o que esta recusa existe para nao mostrar"
        )


def test_d335_a_recusa_local_nomeia_todos_os_seletores_ativos(qtbot: Any) -> None:
    """Com os tres normalizadores activos, os tres sao nomeados.

    Nomear so o primeiro faria a pessoa corrigi-lo, clicar outra vez e ser recusada de novo — a
    recusa a gotejar, que e pior que uma recusa so.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.eixos["imputation"].setCurrentText("none")
    painel.eixos["sample_normalization"].setCurrentText("sum")
    painel.eixos["transformation"].setCurrentText("log10")
    painel.eixos["scaling"].setCurrentText("auto")

    with pytest.raises(CombinacaoProibidaError) as capturado:
        campos_da_tela(
            painel, modo="neg_c18", correcao="none", n_lotes=1, eixos=(), recomendacao=None
        )

    mensagem = str(capturado.value)
    for rotulo in ("Normalização por amostra", "Transformação", "Escalonamento"):
        assert rotulo in mensagem, f"a recusa nao nomeou {rotulo!r}, que esta activo"


def test_a_recusa_local_nomeia_as_opcoes_como_os_seletores_as_mostram(qtbot: Any) -> None:
    """«Não aplicar» e «1/5 do menor valor positivo», e não `none` e `one_fifth_min`.

    Os seletores da grade mostram os nomes de `NOMES_DAS_OPCOES` por cima do valor, e a
    recusa mandava pôr um seletor *«em `none`»* e escolher *«`one_fifth_min`»*, duas palavras que a
    tela não mostra. **Os nomes esperados saem do mapa que os seletores desenham**, e o padrão da
    imputação sai de `EIXOS_DA_GRADE`, que é de onde o seletor nasce.
    """
    import re as _re

    from refino.desktop.janela.nomes import NOMES_DAS_OPCOES
    from refino.interface.fluxo import EIXOS_DA_GRADE

    padrao = next(p for chave, _r, p, _o in EIXOS_DA_GRADE if chave == "imputation")
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.eixos["imputation"].setCurrentText("none")
    painel.eixos["scaling"].setCurrentText("auto")
    with pytest.raises(CombinacaoProibidaError) as capturado:
        campos_da_tela(
            painel, modo="neg_c18", correcao="none", n_lotes=1, eixos=(), recomendacao=None
        )
    mensagem = str(capturado.value)
    assert f"«{NOMES_DAS_OPCOES['none']}»" in mensagem, mensagem
    assert f"«{NOMES_DAS_OPCOES[padrao]}»" in mensagem, mensagem
    crus = sorted(set(_re.findall(r"`([a-z_0-9]+)`", mensagem)))
    assert crus == [], f"a recusa ainda escreve valores crus que o seletor não mostra: {crus}"


def test_d335_as_combinacoes_legitimas_continuam_a_passar(qtbot: Any) -> None:
    """**A direccao oposta, e sem ela a recusa seria indistinguivel de «recusa sempre».**

    Tres ancoras, e a do meio e a que separa: `none` com tudo em `none` e o caminho legitimo que
    entrega a matriz corrigida sem normalizar, e recusa-lo seria tirar uma capacidade real.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    legitimas = (
        ("one_fifth_min", "sum", "log10", "auto"),
        ("none", "none", "none", "none"),
        ("knn", "pqn_group", "cbrt", "vast"),
    )
    for imputacao, normalizacao, transformacao, escalonamento in legitimas:
        painel.eixos["imputation"].setCurrentText(imputacao)
        painel.eixos["sample_normalization"].setCurrentText(normalizacao)
        painel.eixos["transformation"].setCurrentText(transformacao)
        painel.eixos["scaling"].setCurrentText(escalonamento)
        campos = campos_da_tela(
            painel, modo="neg_c18", correcao="none", n_lotes=1, eixos=(), recomendacao=None
        )
        assert campos["imputation"] == imputacao
        assert campos["sample_normalization"] == normalizacao


def test_d335_a_recusa_local_concorda_com_a_recusa_do_dominio(qtbot: Any) -> None:
    """**As duas guardas nao podem divergir**, e este e o teste que as amarra.

    A recusa local existe para dizer melhor o que o dominio ja diz. Se ela recusasse mais, tiraria
    uma capacidade que a CLI tem; se recusasse menos, o `422` voltaria com o jargao. Entao as duas
    sao corridas sobre a mesma matriz de casos e os desfechos comparados.
    """
    from refino.domain.spec import PipelineSpec

    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    casos = [
        (i, n, t, e)
        for i in ("none", "one_fifth_min")
        for n in ("none", "sum")
        for t in ("none", "log10")
        for e in ("none", "auto")
    ]
    divergentes = []
    for imputacao, normalizacao, transformacao, escalonamento in casos:
        painel.eixos["imputation"].setCurrentText(imputacao)
        painel.eixos["sample_normalization"].setCurrentText(normalizacao)
        painel.eixos["transformation"].setCurrentText(transformacao)
        painel.eixos["scaling"].setCurrentText(escalonamento)
        try:
            campos_da_tela(
                painel, modo="neg_c18", correcao="none", n_lotes=1, eixos=(), recomendacao=None
            )
            painel_recusou = False
        except CombinacaoProibidaError:
            painel_recusou = True
        try:
            PipelineSpec(
                correction="none",
                n_batches=1,
                rsd_cutoff=None,
                ranking_filter="none",
                ranking_keep_fraction=None,
                ranking_filter_2="none",
                ranking_keep_fraction_2=None,
                rsd_variant="outlier_removed",
                missing_threshold=None,
                imputation=imputacao,
                sample_normalization=normalizacao,
                normalization_reference=None,
                normalization_factors=None,
                transformation=transformacao,
                scaling=escalonamento,
            )
            dominio_recusou = False
        except ValueError:
            dominio_recusou = True
        if painel_recusou != dominio_recusou:
            divergentes.append(
                (
                    imputacao,
                    normalizacao,
                    transformacao,
                    escalonamento,
                    painel_recusou,
                    dominio_recusou,
                )
            )

    assert any(  # a ancora: se o dominio nunca recusasse, a concordancia seria vacua
        c[0] == "none" and (c[1], c[2], c[3]) != ("none", "none", "none") for c in casos
    ), "ancora: nenhum caso exercita a combinacao proibida"
    assert not divergentes, f"as duas guardas divergem em {len(divergentes)} caso(s): {divergentes}"


# ---------------------------------------------------------------------------
# O registro do corte na janela: o rótulo e o arquivo irmão
# ---------------------------------------------------------------------------

# O corpo que `GET /trabalhos/{id}/relatorio` serve, na forma mínima que exercita as três causas.
# **Escrito à mão e não capturado da rota de propósito:** o que estes testes afirmam é o que o
# painel FAZ com um corpo, e capturá-lo da rota acoplaria o teste do painel ao servidor — a
# paridade entre os dois é assunto de `tests/test_relatorio_do_corte.py`.
RELATORIO_COM_CORTE: dict[str, Any] = {
    "mantidas": 5000,
    "removidas": [
        {"nome": "0.71_180.0634m/z", "causa": "qcs_todos_zero", "rsd": None},
        {"nome": "1.02_244.0821m/z", "causa": "outliers_apagaram_qcs", "rsd": None},
        {"nome": "4.18_612.1907m/z", "causa": "rsd_acima_do_corte", "rsd": 1.8342},
    ],
    "zeros_como_vazias": 12,
    "nomes_sanitizados": 5000,
}

RELATORIO_SEM_CORTE: dict[str, Any] = {
    "mantidas": 2039,
    "removidas": [],
    "zeros_como_vazias": 7,
    "nomes_sanitizados": 2039,
}


def _resposta_de_artefato(conteudo: bytes) -> Resposta:
    """Um `200` com bytes, na forma que `cliente.envia` produz para `text/csv`."""
    return Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=None, conteudo=conteudo)


def test_d347_o_relatorio_acrescenta_o_corte_ao_rotulo_sem_apagar_o_tamanho(qtbot: Any) -> None:
    """As contagens por causa entram no rótulo, e a frase do download **fica**.

    A frase *"N KB gerados"* é a que diz que há artefato. Substituí-la pelo
    resumo do corte trocaria uma informação pela outra, e o painel passaria a descrever o que foi
    removido sem dizer que há o que levar.

    > **A frase era *"N KB prontos para salvar"* até 03/09/2026**, e mudou com o botão «Baixar X»:
    > sem um controle de salvar, prometer *"prontos para salvar"* descreveria um gesto que já não
    > existe. `MARCA_DE_GERADO` é importada em vez de redigitada — ver `_anunciados`.
    """
    painel = _painel_de_artefatos(qtbot)
    painel.recebe("exportar", 200, b"z" * 2048, "ok")

    texto = painel.recebe_relatorio("exportar", 200, RELATORIO_COM_CORTE)

    assert MARCA_DE_GERADO in texto, "a frase que anuncia o artefato saiu do rótulo"
    assert texto_do_tamanho(b"z" * 2048) in texto, "o tamanho do artefato saiu do rótulo"
    assert "5000 mantidas" in texto and "3 removidas" in texto
    assert "QCs todos zero: 1" in texto
    assert "a remoção de outliers apagou os QCs: 1" in texto
    assert "RSD acima do corte: 1" in texto
    assert painel.rotulos["exportar"].text() == texto


def test_d347_sem_remocao_o_rotulo_diz_que_nada_saiu(qtbot: Any) -> None:
    """Lista vazia escreve «nenhuma removida», e **não** omite a linha.

    Silêncio ali é indistinguível de *"o painel não sabe"*, e é o caso comum: três dos quatro
    modos reais não chegam a 5.000 features.
    """
    painel = _painel_de_artefatos(qtbot)
    painel.recebe("exportar", 200, b"z" * 2048, "ok")

    texto = painel.recebe_relatorio("exportar", 200, RELATORIO_SEM_CORTE)

    assert "nenhuma removida" in texto
    assert "2039 mantidas" in texto


def _janela_de_exportacao(transporte_real: Any, qtbot: Any, pasta: Path) -> tuple[Any, list[int]]:
    """A janela pronta para exportar, com o escolhedor de pasta CONTADO.

    A contagem é o que torna *"sem um segundo gesto"* verificável: um registro do corte que abrisse
    diálogo próprio faria a lista crescer para dois, e o profissional teria de voltar ao computador
    para um arquivo que ele não pediu.
    """
    aberturas: list[int] = []

    def escolhe() -> Path:
        aberturas.append(1)
        return pasta

    janela = JanelaPrincipal(transporte_real, pasta=escolhe)
    qtbot.addWidget(janela)
    janela.arquivos._par = _par_real()
    # O par entra direto, sem a escolha que deduz o modo: o modo do par vai à mão.
    janela.arquivos.modo.setEditText("neg_c18")
    janela.chama = lambda *a, **k: None  # type: ignore[method-assign]
    # O download que corta é o da Correção no formato MetaboAnalyst desde 09/11/2026, e ele parte
    # da Correção feita.
    _com_a_correcao_feita(janela)
    return janela, aberturas


def _resposta_de_relatorio(corpo: dict[str, Any]) -> Resposta:
    """Um `200` com o corpo do relatório, na forma que `cliente.envia` produz para JSON."""
    return Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=corpo, conteudo=None)


def test_d347_o_arquivo_irmao_nasce_ao_lado_do_csv_escrito(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """Um clique produz DOIS arquivos, e o segundo tem uma linha por feature removida.

    **É a propriedade que a unidade inteira existe para ter.** O CSV cortado é o que sobe para o
    MetaboAnalyst; o irmão é o registro do que saiu, e ele viaja com o artefato porque nasce do
    **mesmo** gesto — o profissional não pede o registro, e por isso ele existe.

    ## O caminho mudou, e a propriedade é a mesma

    Até 03/09/2026 quem escrevia o irmão era `PainelDeArtefatos.salva`, do botão «Baixar X». Com o
    botão removido, o CSV é escrito por `JanelaPrincipal._escreve_no_destino` e o irmão por
    `registra_corte_ao_lado`, chamado do ramo do relatório. **A ordem é o conteúdo:** os bytes
    chegam numa resposta e o relatório do corte noutra, DEPOIS — escrever o irmão junto com o CSV
    leria `_relatorios` antes de a chave existir e produziria um arquivo de cabeçalho para sempre.

    **A âncora de «sem um segundo gesto» é a CONTAGEM de aberturas do diálogo**, e não a ausência
    de uma janela que um teste não veria: ela vale um no fim, e valeria dois se o registro pedisse
    o caminho por conta própria.
    """
    janela, aberturas = _janela_de_exportacao(transporte_real, qtbot, tmp_path)

    janela.resultado_da_correcao.pede_e_gera("corrigida_metaboanalyst")
    assert aberturas == [1], f"âncora: o clique não abriu o diálogo da pasta: {aberturas}"
    janela.resultado_da_correcao.recebe("corrigida_metaboanalyst", 200, b"z" * 2048, "ok")
    janela._escreve_no_destino("corrigida_metaboanalyst", b"z" * 2048)

    escrito = janela._escritos.get("corrigida_metaboanalyst")
    assert escrito is not None and escrito.is_file(), (
        f"âncora: o CSV não chegou ao disco, e não há «ao lado» de quê: {janela._escritos}"
    )

    # A segunda resposta — o relatório do corte —, pelo ramo real da janela.
    janela._encaminha(
        CHAVES_DO_RELATORIO["corrigida_metaboanalyst"],
        _resposta_de_relatorio(RELATORIO_COM_CORTE),
    )

    irmao = nome_do_arquivo_irmao(escrito)
    assert irmao.is_file(), f"o registro do corte não foi escrito: {sorted(tmp_path.iterdir())}"
    # **O NOME INTEIRO e não o sufixo, e a diferença é uma tautologia.** `nome_do_arquivo_irmao` é
    # `escolhido.with_name(f"{stem}-removidas{suffix}")` e `escrito` vem sempre de
    # `nome_do_arquivo(...)`, que termina em `.csv` — então `endswith("-removidas.csv")` é
    # **verdadeiro por construção** para qualquer `escrito`, e não separa modo errado, chave
    # errada, método ausente nem pasta errada. Derivado do que a janela de facto escolheu, e não
    # redigitado, para o teste não pinar o corretor.
    #
    # **O modo sai do COMBO e não é redigitado**, e a primeira versão desta asserção escreveu
    # `"neg_c18"` de cabeça — a asserção nova reprovou na hora com `pos_hilic_exportar_none.csv`,
    # que é o padrão do combo. O nome estava à mão e media outra coisa.
    modo_corrente = janela.arquivos.modo.currentText()
    # O corretor é o da Correção feita, que é o que o download leva desde 09/11/2026.
    feita = janela._etapas.feita(ETAPAS[0])
    assert feita is not None and feita.correcao is not None, "âncora: a Correção feita"
    metodo_corrente = feita.correcao
    esperado = tmp_path / nome_do_arquivo(modo_corrente, "corrigida_metaboanalyst", metodo_corrente)
    assert escrito == esperado, f"o CSV foi escrito em {escrito}, e não em {esperado}"
    assert irmao.name == f"{esperado.stem}-removidas.csv", irmao.name

    linhas = list(csv.reader(irmao.read_text(encoding="utf-8").splitlines()))
    assert linhas[0] == ["feature", "causa", "rsd_fracao"]
    assert [linha[0] for linha in linhas[1:]] == [
        item["nome"] for item in RELATORIO_COM_CORTE["removidas"]
    ]
    assert [linha[1] for linha in linhas[1:]] == [
        item["causa"] for item in RELATORIO_COM_CORTE["removidas"]
    ]
    assert aberturas == [1], (
        f"o registro do corte pediu um caminho por conta própria: o diálogo abriu "
        f"{len(aberturas)} vezes"
    )


def test_d347_o_arquivo_irmao_e_escrito_pelo_escritor_do_nucleo(qtbot: Any, tmp_path: Path) -> None:
    """Byte-identidade contra `write_removidas`, o mesmo argumento da identidade API e CLI.

    Se a janela formatasse o arquivo por conta própria, esta asserção seria a única coisa a
    separar as duas formatações — e ela reprova no dia em que uma delas mudar. É por isso que
    `removidas_do_relatorio` existe: a janela converte, e **não** escreve.
    """
    painel = _painel_de_artefatos(qtbot)
    painel.recebe("exportar", 200, b"z" * 2048, "ok")
    painel.recebe_relatorio("exportar", 200, RELATORIO_COM_CORTE)

    csv_escrito = tmp_path / nome_do_arquivo("neg_c18", "exportar")
    csv_escrito.write_bytes(b"z" * 2048)
    irmao = painel.registra_corte_ao_lado("exportar", csv_escrito)
    assert irmao is not None and irmao.is_file()

    esperado = tmp_path / "pelo-nucleo.csv"
    write_removidas(removidas_do_relatorio(RELATORIO_COM_CORTE), esperado)

    assert irmao.read_bytes() == esperado.read_bytes()


def test_d347_par_causal_operacao_que_nao_corta_nao_produz_arquivo_irmao(
    qtbot: Any, tmp_path: Path
) -> None:
    """**O par causal**, e sem ele o teste acima seria satisfeito por um painel que escreve sempre.

    `corrigir` devolve a matriz inteira e não tem relatório. Um arquivo irmão ao lado dela — ainda
    que só com cabeçalho — afirmaria que houve um corte que nunca aconteceu, e o profissional
    leria *"0 removidas"* como garantia sobre um arquivo que não passou por corte nenhum.
    """
    painel = _painel_de_artefatos(qtbot)
    painel.recebe("corrigir", 200, b"z" * 2048, "ok")
    # A rota do relatório responde `409` para esta operação, e o painel ignora o que não é 200.
    painel.recebe_relatorio("corrigir", 409, {"detail": "não tem relatório de corte"})

    csv_escrito = tmp_path / nome_do_arquivo("neg_c18", "corrigir")
    csv_escrito.write_bytes(b"z" * 2048)
    assert painel.registra_corte_ao_lado("corrigir", csv_escrito) is None, (
        "o painel registrou um corte que a operação não fez"
    )

    escritos = sorted(p.name for p in tmp_path.iterdir())
    assert escritos == [nome_do_arquivo("neg_c18", "corrigir")], (
        f"apareceu arquivo a mais: {escritos}"
    )


def test_d347_o_relatorio_nao_toca_o_rotulo_quando_nao_ha_bytes(qtbot: Any) -> None:
    """A guarda dos bytes — resposta atrasada de um trabalho supersedido não escreve nada.

    Sem ela, as contagens de um corte apareceriam ao lado de um rótulo que já não anuncia artefato
    nenhum: o rótulo a descrever um arquivo que o painel já descartou. É o mesmo defeito de
    identidade que `define_entrada` fecha um nível acima.
    """
    painel = _painel_de_artefatos(qtbot)

    texto = painel.recebe_relatorio("exportar", 200, RELATORIO_COM_CORTE)

    assert texto == ""
    assert painel.rotulos["exportar"].text() == SEM_ARTEFATO


def test_d347_trocar_a_entrada_descarta_o_relatorio_junto_com_os_bytes(
    qtbot: Any, tmp_path: Path
) -> None:
    """O relatório segue o ciclo de vida dos bytes, e não o seu próprio.

    Mantê-lo faria a exportação seguinte escrever o registro do corte **anterior** ao lado do CSV
    novo — o defeito de identidade que o docstring do módulo mede, na superfície do registro.
    """
    painel = _painel_de_artefatos(qtbot)
    painel.recebe("exportar", 200, b"z" * 2048, "ok")
    painel.recebe_relatorio("exportar", 200, RELATORIO_COM_CORTE)

    painel.define_entrada(ENTRADA_B, "neg_c18")
    painel.recebe("exportar", 200, b"y" * 2048, "ok")

    csv_escrito = tmp_path / nome_do_arquivo("neg_c18", "exportar")
    csv_escrito.write_bytes(b"y" * 2048)
    assert painel.registra_corte_ao_lado("exportar", csv_escrito) is None, (
        "o relatório da entrada anterior sobreviveu à troca"
    )

    # **A forma POSITIVA e nao so a ausencia.** Uma asserção de que o caminho do irmão não existe
    # passaria por vacuidade se ele fosse escrito com outro nome, ou noutra pasta: o conjunto do
    # diretório é o que separa "não foi escrito" de "foi escrito onde eu não olhei".
    assert sorted(p.name for p in tmp_path.iterdir()) == [nome_do_arquivo("neg_c18", "exportar")], (
        f"o diretorio tem mais que o artefato: {sorted(p.name for p in tmp_path.iterdir())}"
    )
    assert not nome_do_arquivo_irmao(csv_escrito).exists(), (
        "o registro do corte da entrada anterior foi escrito ao lado do artefato novo"
    )


def test_d347_falha_ao_escrever_o_irmao_nao_perde_o_artefato(
    qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O CSV já está em disco quando o irmão falha, e o rótulo diz que só o registro se perdeu.

    Reportar falha sobre um arquivo que existe seria pior que perder o registro: a pessoa
    procuraria um CSV que está lá. **A âncora é o CSV no disco antes da falha** — sem ela, *"o
    artefato não foi perdido"* seria verdade sobre um arquivo que nunca existiu.
    """
    painel = _painel_de_artefatos(qtbot)
    painel.recebe("exportar", 200, b"z" * 2048, "ok")
    painel.recebe_relatorio("exportar", 200, RELATORIO_COM_CORTE)

    csv_escrito = tmp_path / nome_do_arquivo("neg_c18", "exportar")
    csv_escrito.write_bytes(b"z" * 2048)
    assert csv_escrito.is_file(), "âncora: não havia CSV a preservar"

    def explode(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("disco cheio")

    monkeypatch.setattr("refino.desktop.janela.artefatos.write_removidas", explode)

    assert painel.registra_corte_ao_lado("exportar", csv_escrito) is None
    assert csv_escrito.is_file(), "o artefato foi perdido por causa do irmão"
    assert "registro do corte NÃO" in painel.rotulos["exportar"].text()


def test_d347_a_janela_pede_o_relatorio_depois_dos_bytes_da_exportacao(
    qtbot: Any, transporte_real: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terceira perna existe: chegado o artefato da exportação, a janela chama a rota irmã.

    **É a ligação verificada na superfície certa.** O painel sabe receber o
    relatório e o servidor sabe servi-lo; sem esta chamada os dois mecanismos ficam corretos,
    verdes, e o profissional continua sem ver o corte — que é exactamente o defeito que já
    houve com `mostra_qualidade`, achado por `grep` e não por teste.

    **O par causal está no mesmo teste:** `corrigir` percorre o mesmo ramo e **não** pede nada.
    """
    janela = JanelaPrincipal(transporte_real)
    qtbot.addWidget(janela)
    pedidas: list[tuple[str, str]] = []
    monkeypatch.setattr(
        JanelaPrincipal,
        "chama",
        lambda self, metodo, rota, **kw: pedidas.append((rota, kw.get("chave", ""))),
    )
    janela._esperas["corrigida_metaboanalyst"] = _EsperaDoArtefato(
        "trab-1", leituras=1, inicio=agora()
    )
    janela._esperas["corrigida"] = _EsperaDoArtefato("trab-2", leituras=1, inicio=agora())

    janela._encaminha("corrigida_metaboanalyst", _resposta_de_artefato(b"csv"), "trab-1")
    relatorio = CHAVES_DO_RELATORIO["corrigida_metaboanalyst"]
    assert pedidas == [("/trabalhos/trab-1/relatorio", f"{relatorio}@trab-1")], pedidas

    pedidas.clear()
    janela._encaminha("corrigida", _resposta_de_artefato(b"csv"), "trab-2")
    assert pedidas == [], f"`corrigir` pediu relatório e não tem corte: {pedidas}"


# ---------------------------------------------------------------------------
# O lote dos tres corretores
# ---------------------------------------------------------------------------


def _janela_com_lote(transporte_real: Any, qtbot: Any, pasta: Path | None) -> Any:
    """Uma janela com o par de arquivos escolhido e o escolhedor de pasta REMENDADO.

    O par tem de estar escolhido porque `_inicia_lote` recusa sem ele — e recusar sem par e a
    primeira das cinco direcoes, exercitada por `test_d350_sem_par_o_lote_nao_pede_pasta`.
    """
    janela = JanelaPrincipal(transporte_real, pasta=lambda: pasta)
    qtbot.addWidget(janela)
    janela.arquivos._par = _par_real()
    # O par entra direto, sem a escolha que deduz o modo: o modo do par vai à mão.
    janela.arquivos.modo.setEditText("neg_c18")
    return janela


def test_d367_o_clique_unico_pede_a_pasta_e_escreve_quando_os_bytes_chegam(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """O caminho feliz do gesto unificado: um clique, e o arquivo aparece na pasta escolhida.

    **`_janela_com_lote` é reusado de propósito**, apesar do nome: ele injeta o escolhedor de pasta
    e escolhe o par de arquivos, que é exactamente o que este fluxo precisa. Um helper novo com o
    mesmo corpo seria segunda fonte sobre como se monta a janela.

    **`chama` é remendado e o resto de `_gera_artefato` corre de verdade.** O que se mede aqui é o
    que o método decide — pedir a pasta, capturar os três campos, e só então disparar —, e não o
    despacho, que tem portão próprio.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    _com_a_correcao_feita(janela)
    despachos: list[str] = []
    janela.chama = lambda *a, **k: despachos.append(k.get("chave", ""))  # type: ignore[method-assign]

    # **Pelo PAINEL e não por `_gera_artefato`**, porque é o painel que pergunta: o gesto que se
    # está a medir começa no botão, e chamar o método da janela saltaria a metade que pede a pasta.
    janela.resultado_da_correcao.pede_e_gera("corrigida")

    assert despachos, "o clique não disparou o trabalho"
    destino = janela._destinos.get("corrigida")
    assert destino is not None, (
        "o clique não guardou o destino, e a chegada não teria onde escrever"
    )
    modo = janela.arquivos.campos_para_envio()["mode"]
    assert (destino.pasta, destino.modo) == (tmp_path, modo), destino
    # O corretor da Correção feita, e não o do combo — o download leva o que a etapa mostrou.
    feita = janela._etapas.feita(ETAPAS[0])
    assert feita is not None and destino.metodo == feita.correcao, destino

    # A chegada dos bytes, que é o segundo tempo do gesto.
    janela._escreve_no_destino("corrigida", b"csv-corrigido")

    alvo = tmp_path / nome_do_arquivo(modo, "corrigida", destino.metodo)
    assert alvo.read_bytes() == b"csv-corrigido", f"o arquivo não apareceu em {alvo}"
    assert janela.rotulo_de_desfecho.text() == ARTEFATO_ESCRITO.format(alvo=alvo)
    assert "corrigida" not in janela._destinos, (
        "o destino não foi consumido: uma segunda chegada reescreveria num caminho antigo"
    )


_FATORES_DA_ETAPA = ("fatores_a.csv", b"amostra,fator\nA1,1.0\n")
_FATORES_DE_AGORA = ("fatores_b.csv", b"amostra,fator\nB1,2.0\n")


def _com_a_normalizacao_feita_com_fatores(janela: JanelaPrincipal) -> None:
    """A Normalização feita sobre a original, com `supplied_factors` e o arquivo A, e depois a tela
    com o arquivo B — o cenário que a revisão de código achou.

    A etapa é marcada como `_dispara_etapa` a marca: com os fatores que o pedido dela levou.
    """
    assert _FATORES_DA_ETAPA[1] != _FATORES_DE_AGORA[1], "âncora: os dois arquivos diferem"
    janela.arquivos._fatores = _FATORES_DA_ETAPA
    atual = janela._campos_atuais()
    assert atual is not None, "âncora: sem arquivos não há Normalização a dar por feita"
    tela, identidade = atual
    tela = {**tela, "sample_normalization": "supplied_factors"}
    matriz = identidade[IDENTIDADE_DA_MATRIZ]
    campos = janela._etapas.campos_da_entrada(2, "original", tela, matriz)
    enviados = janela.arquivos.arquivos_da_execucao()
    janela._etapas.disparou(
        2,
        fotografia_da_etapa(tela, 2, identidade, "original"),
        campos=campos,
        entrada="original",
        fatores=enviados["normalization_factors"],
    )
    janela._etapas.concluiu(2)
    janela.arquivos._fatores = _FATORES_DE_AGORA


def test_o_download_da_normalizacao_leva_os_fatores_da_etapa_feita_e_nao_os_de_agora(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """Trocar o arquivo de fatores depois de normalizar não muda o arquivo que se baixa.

    O download leva os campos da Normalização FEITA, e com `supplied_factors` o resultado depende
    dos bytes. Antes da correção ele subia o arquivo da tela de agora, e o arquivo baixado
    divergia do que a etapa mostrou. **Par causal**: os dois arquivos diferem (âncora no
    auxiliar), e a asserção separa um do outro pelos bytes.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    _com_a_normalizacao_feita_com_fatores(janela)
    enviados: list[dict[str, Any]] = []
    janela.chama = lambda *a, **k: enviados.append(k)  # type: ignore[method-assign]

    janela.resultado_da_normalizacao.pede_e_gera("normalizada")

    assert enviados, "o clique não disparou o download"
    fatores = enviados[-1]["arquivos"].get("normalization_factors")
    assert fatores is not None, "o download da Normalização saiu sem o arquivo de fatores"
    assert fatores[1] == _FATORES_DA_ETAPA[1], (
        f"o download subiu os fatores de agora ({fatores[0]}), e não os da etapa feita"
    )


def test_a_analise_sobre_a_normalizada_leva_os_fatores_da_etapa_feita(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """A Análise sobre a «normalizada» refaz a Normalização feita: leva os fatores dela.

    A irmã do download acima, em `_dispara_etapa`. Sem a correção, a Análise mandava os campos da
    etapa feita com o arquivo de agora, e ainda ficava «em dia».
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    _com_a_normalizacao_feita_com_fatores(janela)
    atual = janela._campos_atuais()
    assert atual is not None
    assert janela._etapas.entrada(3, atual[1][IDENTIDADE_DA_MATRIZ]).escolhida == "normalizada", (
        "âncora: a Análise parte da normalizada, que é a padrão quando ela existe"
    )
    enviados: list[dict[str, Any]] = []
    janela.chama = lambda *a, **k: enviados.append(k)  # type: ignore[method-assign]

    janela._dispara_etapa(3)

    assert enviados, "a Análise não foi disparada"
    fatores = enviados[-1]["arquivos"].get("normalization_factors")
    assert fatores is not None and fatores[1] == _FATORES_DA_ETAPA[1], (
        f"a Análise sobre a normalizada subiu {None if fatores is None else fatores[0]}, e não os "
        "fatores da Normalização feita"
    )


def test_d367_cancelar_a_pasta_aborta_sem_disparar_o_trabalho(
    transporte_real: Any, qtbot: Any
) -> None:
    """A outra direção, e ela é o par causal da de cima: as duas DIFEREM no despacho.

    **É a metade que impede o gesto de custar uma correção por engano.** Sem esta direção, cancelar
    o diálogo ainda dispararia até 209,9 s de SERRF para escrever em lugar nenhum — e o profissional
    que cancelou pediu justamente para não correr.

    **A asserção de que NADA foi despachado é de presença invertida**, então vem com a âncora do
    caminho feliz ao lado: `despachos` estar vazio só prova alguma coisa porque o mesmo arnês, com
    pasta, o enche.
    """
    janela = _janela_com_lote(transporte_real, qtbot, None)
    _com_a_correcao_feita(janela)
    despachos: list[str] = []
    janela.chama = lambda *a, **k: despachos.append(k.get("chave", ""))  # type: ignore[method-assign]

    janela.resultado_da_correcao.pede_e_gera("corrigida")

    assert despachos == [], f"cancelar a pasta disparou o trabalho na mesma: {despachos}"
    assert janela._destinos == {}, "ficou destino pendente de um trabalho que não corre"
    assert janela.resultado_da_correcao.destinos == {}, (
        "o painel guardou uma pasta que ninguém escolheu"
    )
    assert janela.resultado_da_correcao.rotulos["corrigida"].text() == SEM_DESTINO


def test_d367_dois_corretores_no_mesmo_modo_nao_colidem_no_disco(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """Dois cliques com corretores diferentes deixam DOIS arquivos — é o que o método compra.

    **A planta cai sobre dado em que as duas coisas diferem:** se `_escreve_no_destino`
    ignorasse `metodo`, os dois cliques escreveriam o mesmo caminho e a pasta ficaria com **um**
    arquivo — o do segundo corretor — enquanto o profissional acredita ter os dois. É o defeito
    do lote na superfície do disco, aqui no caminho avulso.

    **O corretor é lido no CLIQUE**, e é por isso que trocá-lo entre os dois cliques produz nomes
    distintos: `DestinoPendente` congela os três campos, e a chegada não os relê. **Desde
    09/11/2026 ele é o da Correção feita, e não o do combo**: o download mora no Resultado da
    Correção, e leva o que a etapa mostrou.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela.chama = lambda *a, **k: None  # type: ignore[method-assign]
    modo = janela.arquivos.campos_para_envio()["mode"]

    primeiro, segundo = CORRETORES[0], CORRETORES[1]
    assert primeiro != segundo, "âncora: os dois corretores são o mesmo e a colisão não é testável"

    _com_a_correcao_feita(janela, primeiro)
    janela.resultado_da_correcao.pede_e_gera("corrigida")
    janela._escreve_no_destino("corrigida", b"do-primeiro")

    _com_a_correcao_feita(janela, segundo)
    janela.resultado_da_correcao.pede_e_gera("corrigida")
    janela._escreve_no_destino("corrigida", b"do-segundo")

    escritos = sorted(caminho.name for caminho in tmp_path.glob("*.csv"))
    assert escritos == sorted(
        [nome_do_arquivo(modo, "corrigida", primeiro), nome_do_arquivo(modo, "corrigida", segundo)]
    ), f"os dois cliques colidiram no disco: {escritos}"
    assert (tmp_path / nome_do_arquivo(modo, "corrigida", primeiro)).read_bytes() == b"do-primeiro"


def test_d367_a_chegada_sem_destino_ou_sem_bytes_nao_escreve(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """As duas saídas silenciosas, e nenhuma delas pode escrever nem mexer no rótulo.

    **Sem destino** é o caminho antigo — um «Baixar de novo», ou uma chegada que não veio de um
    clique unificado. **Sem bytes** é a rota a responder sem produzir artefato, e aí `recebe` já pôs
    a causa no rótulo: escrever por cima trocaria o diagnóstico por uma mensagem de escrita.

    **É o ramo `else`**, o que dispara por ausência de condição e por isso é o mais
    fácil de ficar inerte sem ninguém notar.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela.chama = lambda *a, **k: None  # type: ignore[method-assign]
    _com_a_correcao_feita(janela)
    antes = janela.rotulo_de_desfecho.text()

    janela._escreve_no_destino("corrigida", b"bytes sem destino")
    assert list(tmp_path.glob("*")) == [], "escreveu sem destino pendente"
    assert janela.rotulo_de_desfecho.text() == antes, "mexeu no rótulo sem ter escrito"

    janela.resultado_da_correcao.pede_e_gera("corrigida")
    assert "corrigida" in janela._destinos, "âncora: sem destino pendente o segundo caso é vácuo"
    janela._escreve_no_destino("corrigida", None)
    assert list(tmp_path.glob("*")) == [], "escreveu um artefato que não veio"
    assert "corrigida" not in janela._destinos, "o destino ficou pendente depois de uma falha"


def test_d367_o_sinal_cru_nao_abre_dialogo_nem_escreve(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """Emitir `artefatos.gerar` direto continua a NÃO ter destino, e é o que preserva 14 testes.

    **Esta é a asserção que paga o desenho.** A primeira versão pedia a pasta dentro de
    `_gera_artefato`, e catorze testes que emitem este sinal passaram a abrir um `QFileDialog` real
    e a **pendurar** — `EXIT=124` medido. Mover a pergunta para o clique devolveu ao sinal um
    significado só, e este teste é o que impede alguém de a mover de volta.

    **A âncora é o par:** o mesmo arnês, pelo botão, escreve — está no teste do caminho feliz. Aqui,
    pelo sinal, não escreve. Sem essa outra metade, `_destinos == {}` passaria por vacuidade.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela.chama = lambda *a, **k: None  # type: ignore[method-assign]
    _com_a_correcao_feita(janela)

    janela.resultado_da_correcao.gerar.emit("corrigida")

    assert janela._destinos == {}, "o sinal cru guardou destino: o diálogo voltou para o despacho"
    janela._escreve_no_destino("corrigida", b"nao-devia-aparecer")
    assert list(tmp_path.glob("*")) == [], "o sinal cru escreveu em disco sem ninguém escolher onde"


def test_d367_uma_falha_de_escrita_relata_no_rotulo_e_nao_perde_os_bytes(
    transporte_real: Any, qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`OSError` na escrita **não escapa do slot**, e o rótulo de desfecho diz o que aconteceu.

    Uma excepção que escapa de um slot de `clicked` não mata o processo e não chega ao chamador: o
    traceback vai para o `stderr`, e num executável `--windowed` não há `stderr`. O desfecho seria
    **nenhum arquivo, nenhuma mensagem, e a janela a parecer ter escrito**.

    `janela/arquivos.py` captura `OSError` na **leitura** por esta razão exacta.

    ## Esta propriedade MUDOU DE CASA, e não foi afrouxada

    Ela era medida em `PainelDeArtefatos.salva`, pelo botão «Baixar X», por
    `test_u6_uma_falha_de_escrita_relata_e_preserva_os_bytes`. Com o botão removido quem escreve é
    `JanelaPrincipal._escreve_no_destino`, e é ele que captura `OSError` e escreve
    `ARTEFATO_SEM_ESCRITA`. O arnês mudou de lado da fronteira; as três asserções são as mesmas.

    **Os bytes ficam no painel**, como ficavam: quem tentou salvar e não conseguiu não pediu para
    perder o artefato, que custou uma correção — e o rótulo do painel continua a anunciá-lo.

    **E a chave sai de `_escritos`, que é a asserção que o par causal torna não-vácua.** Ela é
    semeada antes com um caminho de uma escrita anterior: sem o `pop`, o irmão do corte iria parar
    ao lado de um CSV que **não** é o desta exportação. Semear é o que separa *"a chave saiu"* de
    *"a chave nunca esteve lá"*.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela.chama = lambda *a, **k: None  # type: ignore[method-assign]
    _com_a_correcao_feita(janela)

    janela.resultado_da_correcao.pede_e_gera("corrigida")
    janela.resultado_da_correcao.recebe("corrigida", 200, b"C" * 4096, "ok")
    prometido = janela.resultado_da_correcao.rotulos["corrigida"].text()
    assert MARCA_DE_GERADO in prometido, (
        f"âncora: o painel não anunciou o artefato, e a preservação abaixo é vácua: {prometido!r}"
    )

    de_antes = tmp_path / "de-uma-escrita-anterior.csv"
    janela._escritos["corrigida"] = de_antes
    antes_do_rotulo = janela.rotulo_de_desfecho.text()

    def nega(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError(13, "negado")

    monkeypatch.setattr(Path, "write_bytes", nega)
    janela._escreve_no_destino("corrigida", b"C" * 4096)
    monkeypatch.undo()

    texto = janela.rotulo_de_desfecho.text()
    assert texto != antes_do_rotulo, (
        "o rótulo de desfecho não mudou depois de a escrita falhar — silêncio, que num executável "
        "`--windowed` é tudo o que o profissional recebe"
    )
    assert texto.startswith(ARTEFATO_SEM_ESCRITA.split("{")[0]), (
        f"a frase não é a do artefato sem escrita: {texto!r}"
    )
    assert "negado" in texto, f"a causa do erro não está na frase: {texto!r}"

    assert list(tmp_path.glob("*.csv")) == [], "a escrita falhou e ainda assim deixou arquivo"
    assert "corrigida" not in janela._escritos, (
        "a chave ficou a apontar para a escrita anterior: o irmão do corte iria parar ao lado de "
        "um CSV que não é o desta exportação"
    )
    assert janela.resultado_da_correcao.salvaveis() == ("corrigida",), (
        "a falha de escrita descartou os bytes: quem gerou não pediu para perder o artefato"
    )
    # **O rótulo do Resultado diz a falha, e isto INVERTEU de propósito em 09/11/2026.** O painel
    # da aba «Executar e exportar» tinha um rótulo por chave que descrevia os BYTES, e a asserção
    # era que a falha de escrita não o reescrevia. O Resultado da Correção tem um rótulo de estado
    # só, ao lado do único botão de download, e ele diz o desfecho do clique — onde o arquivo foi
    # escrito, ou por que não foi: é ali que a pessoa olha. Os bytes continuam no painel,
    # afirmado acima, e a frase é a mesma do aviso de desfecho.
    estado = janela.resultado_da_correcao.rotulos["corrigida"].text()
    assert estado != prometido and estado == texto, (
        f"a falha de escrita não chegou ao lado do botão que a pediu: {estado!r}"
    )


def test_d350_o_lote_escreve_um_arquivo_por_corretor_com_o_metodo_no_nome(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """O caminho feliz: tres pernas, tres arquivos, nomes distintos.

    **A ancora dos NOMES DISTINTOS e o conteudo do teste**, e nao decoracao. Se
    `nome_do_arquivo` ignorasse `metodo`, as tres pernas escreveriam o MESMO caminho e a pasta
    ficaria com um arquivo — o do ultimo corretor — enquanto o rotulo diria «tres». O
    profissional levaria um arquivo a acreditar que tem os tres.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    disparos: list[str] = []
    # `is None` devolve `True`: o duplo diz que a perna saiu, como `_gera_artefato`.
    janela._gera_artefato = lambda chave: (  # type: ignore[method-assign]
        disparos.append(janela.pipeline.corretor.currentText()) is None
    )

    janela._inicia_lote(CHAVE_DO_LOTE_METABO)
    for i, _ in enumerate(CORRETORES):
        janela._lote_recebe(f"csv-{i}".encode())

    assert disparos == list(CORRETORES), f"as pernas nao correram os tres na ordem: {disparos}"
    escritos = sorted(p.name for p in tmp_path.glob("*.csv"))
    assert len(escritos) == len(CORRETORES), (
        f"esperava um arquivo por corretor e vieram {escritos} — os nomes colidiram?"
    )
    modo = janela.arquivos.campos_para_envio()["mode"]
    assert escritos == sorted(nome_do_arquivo(modo, CHAVE_DO_LOTE_METABO, m) for m in CORRETORES), (
        escritos
    )
    assert (tmp_path / nome_do_arquivo(modo, CHAVE_DO_LOTE_METABO, CORRETORES[0])).read_bytes() == (
        b"csv-0"
    ), "os bytes de uma perna foram parar no arquivo de outra"
    assert janela._lote is None, "o lote nao encerrou"


def test_d350_o_combo_volta_ao_que_o_profissional_tinha_escolhido(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """O lote move o combo para mostrar a perna, e **devolve-o** ao fim.

    Sem isto, a geracao avulsa seguinte correria o ultimo corretor do lote — um metodo que
    ninguem pediu, escolhido por um lote que ja terminou.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela.pipeline.corretor.setCurrentText(CORRETORES[-1])
    escolhido = janela.pipeline.corretor.currentText()
    janela._gera_artefato = lambda chave: True  # type: ignore[method-assign]

    janela._inicia_lote(CHAVE_DO_LOTE_METABO)
    assert janela.pipeline.corretor.currentText() == CORRETORES[0], (
        "a primeira perna nao moveu o combo"
    )
    for i, _ in enumerate(CORRETORES):
        janela._lote_recebe(f"csv-{i}".encode())

    assert janela.pipeline.corretor.currentText() == escolhido, (
        "o combo ficou no corretor do lote em vez de voltar ao escolhido"
    )


def test_d350_perna_sem_artefato_aborta_e_o_rotulo_nomeia_o_corretor(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """**A segunda direcao**: um lote que seguisse deixaria dois arquivos numa pasta
    pedida para tres.

    O corretor tem de ser NOMEADO porque e ele que decide se vale repetir: `serrf` custa 220,3 s
    medidos e `none` custa zero.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela._gera_artefato = lambda chave: True  # type: ignore[method-assign]

    janela._inicia_lote(CHAVE_DO_LOTE_METABO)
    janela._lote_recebe(b"csv-0")
    janela._lote_recebe(None)

    assert janela._lote is None, "o lote continuou depois de uma perna sem artefato"
    modo = janela.arquivos.campos_para_envio()["mode"]
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == [
        nome_do_arquivo(modo, CHAVE_DO_LOTE_METABO, CORRETORES[0])
    ], "o lote abortado escreveu arquivo a mais ou apagou o que ja tinha escrito"
    assert LOTE_ABORTADO.format(metodo=CORRETORES[1]) == janela.rotulo_de_desfecho.text(), (
        f"o rotulo nao nomeia o corretor que falhou: {janela.rotulo_de_desfecho.text()}"
    )


def test_o_lote_abortado_fica_na_faixa_vermelha_do_resultado(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """O lote que parou no meio aparece no Resultado, e não só no aviso de desfecho, que some.

    Achado da revisão de código. Par causal: antes do lote o Resultado não tem
    falha; depois do aborto, tem a frase que nomeia o corretor.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela._gera_artefato = lambda chave: True  # type: ignore[method-assign]
    resultado = janela.resultado_da_correcao
    assert resultado._falha is None, "âncora: antes do lote não há falha no Resultado"

    janela._inicia_lote(CHAVE_DO_LOTE_METABO)
    janela._lote_recebe(b"csv-0")
    janela._lote_recebe(None)

    assert resultado._falha is not None, "o lote abortado não chegou ao Resultado"
    assert resultado._falha[0] == LOTE_ABORTADO.format(metodo=CORRETORES[1]), resultado._falha


def test_a_falha_de_uma_etapa_sai_quando_a_matriz_da_tela_muda(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """A faixa vermelha de uma etapa era sobre a matriz de antes: trocar o modo a tira.

    Achado da revisão de código. Par causal: com a mesma matriz a falha está
    na aba; com outra, sai.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela._falhas_das_etapas[2] = "a rota recusou"
    janela._atualiza_as_etapas()
    pagina = janela._paginas_de_etapa[2]
    assert pagina._aviso is not None and pagina._aviso[1] == "critico", (
        "âncora: a falha aparece na aba enquanto a matriz é a mesma"
    )

    janela.arquivos.modo.setEditText("pos_c18")
    janela._atualiza_as_etapas()

    assert pagina._aviso is None or pagina._aviso[1] != "critico", pagina._aviso


def test_d350_cancelar_a_pasta_nao_dispara_nada(transporte_real: Any, qtbot: Any) -> None:
    """**A terceira direcao**: `None` do dialogo e cancelamento, e nao a pasta corrente.

    `Path("")` seria o diretorio de trabalho, e escrever tres CSV de paciente num caminho que
    ninguem escolheu e o defeito que `pasta_por_dialogo` ja declara para o seu proprio `None`.
    """
    janela = _janela_com_lote(transporte_real, qtbot, None)
    disparos: list[str] = []
    janela._gera_artefato = lambda chave: disparos.append(chave) is None  # type: ignore[method-assign]

    janela._inicia_lote(CHAVE_DO_LOTE_METABO)

    assert disparos == [], "cancelar a pasta disparou a exportacao mesmo assim"
    assert janela._lote is None
    assert janela.rotulo_de_desfecho.text() == LOTE_CANCELADO


def test_d350_um_lote_de_cada_vez(transporte_real: Any, qtbot: Any, tmp_path: Path) -> None:
    """**A quarta direcao**: dois lotes em voo escreveriam na mesma pasta com o combo a saltar."""
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela._gera_artefato = lambda chave: True  # type: ignore[method-assign]

    janela._inicia_lote(CHAVE_DO_LOTE_METABO)
    primeiro = janela._lote
    janela._inicia_lote(CHAVE_DO_LOTE_METABO)

    assert janela._lote is primeiro, "o segundo clique substituiu o lote em curso"
    assert janela.rotulo_de_desfecho.text() == LOTE_EM_CURSO


def test_d544_o_pipeline_recusa_com_a_exportacao_dos_tres_em_curso(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """A trava dos irmãos, que faltava no «Executar pipeline».

    Medido em 08/10/2026 na máquina de um testador: o pipeline disparado 5 s depois da exportação
    dos três leu o combo que ela tinha movido para o `serrf`, e os dois SERRF disputaram os núcleos.
    **Par causal:** a mesma janela, com e sem o lote em curso, e só a segunda chega à rota.

    **O botão é o da Análise desde 09/11/2026**: o «Executar pipeline» saiu com a aba «Executar e
    exportar», e quem dispara o `/execucoes` agora é a Análise, pela mesma trava.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela._gera_artefato = lambda chave: True  # type: ignore[method-assign]
    chamadas: list[str] = []
    janela.chama = lambda metodo, rota, **_k: chamadas.append(rota)  # type: ignore[method-assign]

    janela._inicia_lote(CHAVE_DO_LOTE_METABO)
    assert janela._lote is not None, "ancora: o lote nao comecou"
    janela._dispara_etapa(ETAPAS[-1])
    assert chamadas == [], "o pipeline disparou com a exportacao dos tres em curso"
    assert janela.rotulo_de_desfecho.text() == LOTE_EM_CURSO

    janela._lote = None
    janela._dispara_etapa(ETAPAS[-1])
    assert chamadas == [ROTA_DA_EXECUCAO], "sem lote em curso, o pipeline tinha de disparar"


def test_d545_cada_atalho_nomeia_uma_figura_do_registo() -> None:
    """Um nome que saísse de `plots.CONSTRUTORES` faria o botão não abrir nada, em silêncio.

    **Os atalhos «Ver gráfico» saíram em 09/11/2026**, com a aba «Executar e exportar» que os
    tinha: os gráficos de cada etapa estão na aba dela. O caminho para a aba «Gráficos» que fica é
    «Exportar esta figura em Gráficos», sob a vista de cada etapa, e as figuras que ele pode levar
    são as de `VISTA_DA_FIGURA` — é delas que a mesma pergunta se faz.
    """
    from refino.desktop.janela.graficos import VISTA_DA_FIGURA
    from refino.plots import CONSTRUTORES

    registadas = {c.funcao for c in CONSTRUTORES}
    assert VISTA_DA_FIGURA, "ancora: nenhuma figura nas vistas das etapas"
    fora = [funcao for funcao in VISTA_DA_FIGURA if funcao not in registadas]
    assert fora == [], f"figuras das vistas sem construtor registado: {fora}"


def test_d545_o_atalho_escolhe_a_figura_e_vai_para_a_aba_dos_graficos(
    transporte_real: Any, qtbot: Any
) -> None:
    """Cada figura das vistas das etapas, levada a «Gráficos», deixa o combo de lá nela.

    **O emissor mudou em 09/11/2026**: eram os botões da moldura «Executar», que saíram com a aba
    «Executar e exportar»; é o sinal `exporta_em_graficos` da vista de cada aba de etapa, que a
    janela liga ao mesmo `_abre_a_figura`. O clique de verdade, com a figura desenhada, está em
    `test_a_vista_de_cada_etapa_esta_na_aba_dela_e_leva_a_figura_aos_graficos`; aqui é cada figura
    que a vista pode levar.

    **Par causal:** um nome fora do registo não troca de aba nem de figura.
    """
    from refino.desktop.janela.principal import INDICE_DA_ABA_DOS_GRAFICOS

    janela = _janela_com_lote(transporte_real, qtbot, None)
    figuras = janela.graficos.figuras
    assert janela.abas.tabText(INDICE_DA_ABA_DOS_GRAFICOS) == "Gráficos"
    vistas = janela.graficos.vistas_das_etapas
    assert vistas, "âncora: nenhuma vista de etapa"
    levadas = 0
    for vista in vistas.values():
        for _rotulo, construtor in vista.figuras._entradas:
            janela.abas.setCurrentIndex(0)
            vista.exporta_em_graficos.emit(construtor.funcao)
            assert janela.abas.currentIndex() == INDICE_DA_ABA_DOS_GRAFICOS, construtor.funcao
            assert figuras._entradas[figuras.seletor.currentIndex()][1].funcao == construtor.funcao
            levadas += 1
    assert levadas > 1, f"âncora: só {levadas} figura(s) nas vistas, e a troca não separa nada"

    janela.abas.setCurrentIndex(0)
    antes = figuras.seletor.currentIndex()
    janela._abre_a_figura("figura_que_nao_existe")
    assert janela.abas.currentIndex() == 0
    assert figuras.seletor.currentIndex() == antes


def _com_a_validacao_em_dia(
    janela: Any, transporte_real: Any, arquivos: dict[str, Any], campos: dict[str, Any]
) -> None:
    """A Validação real dos arquivos da tela, entregue por `_recebe`.

    O Diagnóstico pedido roda antes a Validação, se ela falta:
    com ela em dia, `_pede_leitura("diagnostico")` dispara o próprio Diagnóstico — com o carimbo —,
    como disparava. Os testes abaixo tratam do carimbo, e não da cadeia.
    """
    http = transporte_real.post("/matrizes/validar", files=arquivos, data=campos)
    assert http.status_code == 200, http.text
    janela._recebe(
        "validacao", Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=http.json())
    )
    assert janela._leitura_em_dia("validacao") is True, "âncora: a Validação não ficou em dia"


def test_d547_o_diagnostico_carimbado_no_disparo_desenha_a_deriva_sem_execucao(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """A figura do diagnóstico vale sem execução, e quem a liga à matriz é o carimbo do disparo.

    **Par causal:** a mesma resposta real de `/matrizes/diagnosticar`, entregue sem e com o carimbo
    de `_pede_leitura`; só a segunda desenha a deriva, e nenhuma execução corre nas duas.

    A Validação entra em dia antes do pedido (`_com_a_validacao_em_dia`): sem
    ela, o pedido rodaria a Validação primeiro, e o carimbo só sairia com o Diagnóstico.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela.chama = lambda *_a, **_k: None  # type: ignore[method-assign]
    pedido = janela._pedido_de_arquivos()
    assert pedido is not None, "ancora: o par de arquivos tem de estar escolhido"
    arquivos, campos = pedido
    http = transporte_real.post("/matrizes/diagnosticar", files=arquivos, data=campos)
    assert http.status_code == 200, http.text
    resposta = Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=http.json())
    figuras = janela.graficos.figuras
    assert figuras.seleciona("figura_da_deriva_do_qc")

    janela._encaminha("diagnostico", resposta)
    assert figuras.estado_visivel()["figura"] == "", "sem carimbo, o diagnostico nao e desta matriz"

    _com_a_validacao_em_dia(janela, transporte_real, arquivos, campos)
    janela._pede_leitura("diagnostico")
    graficos = janela.graficos
    assert graficos._identidade_do_diagnostico == graficos._identidade_da_matriz is not None
    janela._encaminha("diagnostico", resposta)
    assert figuras.estado_visivel()["figura"] != "", "com o carimbo, a deriva tinha de desenhar"


def test_d551_trocar_de_matriz_tira_o_diagnostico_da_anterior(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """#3 da revisão de 15/10/2026: a deriva de A ficava desenhada, e exportável, com B na tela.

    `define_matriz` trocava a identidade sem reavaliar as figuras, e um teste antigo fazia à mão a
    reavaliação que a janela nunca fazia. Aqui a troca passa pelo caminho de produção — o disparo da
    validação, que é quem chama `define_matriz`. **Par causal:** de volta à matriz do diagnóstico,
    a mesma deriva volta a desenhar.

    A Validação entra em dia antes do pedido do Diagnóstico
    (`_com_a_validacao_em_dia`), e o pedido dispara o Diagnóstico com o carimbo, como disparava.
    """
    from refino.desktop.janela.figuras import EstadoDaFigura, exporta_habilitado, texto_do_estado

    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela.chama = lambda *_a, **_k: None  # type: ignore[method-assign]
    pedido = janela._pedido_de_arquivos()
    assert pedido is not None, "ancora: o par de arquivos tem de estar escolhido"
    arquivos, campos = pedido
    http = transporte_real.post("/matrizes/diagnosticar", files=arquivos, data=campos)
    assert http.status_code == 200, http.text
    resposta = Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=http.json())
    figuras = janela.graficos.figuras
    assert figuras.seleciona("figura_da_deriva_do_qc")
    _com_a_validacao_em_dia(janela, transporte_real, arquivos, campos)
    janela._pede_leitura("diagnostico")
    janela._encaminha("diagnostico", resposta)
    assert figuras.estado_visivel()["figura"] != "", "ancora: a deriva nao desenhou"

    janela.arquivos._par = _par_real("pos_hilic")
    # O par entra direto, sem a escolha que deduz o modo: o modo do par vai à mão.
    janela.arquivos.modo.setEditText("pos_hilic")
    janela._pede_leitura("validacao")
    assert figuras.estado_visivel()["figura"] == "", "a deriva da matriz anterior ficou na tela"
    assert figuras.repouso.text() == texto_do_estado(EstadoDaFigura.SEM_DIAGNOSTICO)
    assert not exporta_habilitado(figuras._estado), "a figura de outra matriz ficou exportavel"

    janela.arquivos._par = _par_real()
    # O par entra direto, sem a escolha que deduz o modo: o modo do par vai à mão.
    janela.arquivos.modo.setEditText("neg_c18")
    janela._pede_leitura("validacao")
    assert figuras.estado_visivel()["figura"] != "", "de volta a matriz, a deriva tinha de voltar"


def test_d567_pela_rota_real_o_diagnostico_desenha_o_violino_da_INGERIDA_sem_execucao(  # noqa: N802
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """O violino antes da primeira correção, pelo caminho de produção.

    O corpo vem da rota real — é ela, e não a fixtura dos testes do painel, que diz que forma a
    carga tem — e entra por `_encaminha`, que é por onde a janela a recebe. **Par causal:**
    antes do diagnóstico o violino espera; com ele, desenha a ingerida, sem `POST /execucoes`.

    A Validação entra em dia antes do pedido do Diagnóstico
    (`_com_a_validacao_em_dia`), e o pedido dispara o Diagnóstico com o carimbo, como disparava.
    """
    from refino.desktop.janela.figuras import CHAVE_DE_REPOUSO, EstadoDaFigura, texto_do_estado
    from refino.plots.recorte import milhar

    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela.chama = lambda *_a, **_k: None  # type: ignore[method-assign]
    pedido = janela._pedido_de_arquivos()
    assert pedido is not None, "ancora: o par de arquivos tem de estar escolhido"
    arquivos, campos = pedido
    http = transporte_real.post("/matrizes/diagnosticar", files=arquivos, data=campos)
    assert http.status_code == 200, http.text
    resposta = Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=http.json())
    figuras = janela.graficos.figuras
    assert figuras.seleciona("figura_do_violino_do_rsd_por_feature")
    assert figuras.estado_visivel()["figura"] == "", "ancora: o violino desenhou sem diagnostico"
    assert figuras.repouso.text() == texto_do_estado(EstadoDaFigura.SEM_DIAGNOSTICO)

    _com_a_validacao_em_dia(janela, transporte_real, arquivos, campos)
    janela._pede_leitura("diagnostico")
    janela._encaminha("diagnostico", resposta)
    assert figuras.estado_visivel()["figura"] != "", "o diagnostico nao desenhou o violino"
    # Presença e não ausência: o painel continua na chave de repouso, que só uma resposta de
    # `/execucoes` troca — e `chama` está remendado, então nenhuma saiu da janela.
    assert figuras._chave == CHAVE_DE_REPOUSO, figuras._chave

    titulo = " ".join(figuras.canvas.figure.get_suptitle().split())
    assert "Matriz: uma — «ingerida», antes de qualquer correção" in titulo, titulo
    carga = http.json()["rsd_por_feature"]
    do_eixo = " ".join(figuras.canvas.figure.axes[0].get_title().split())
    for lado in ("dos_qcs", "das_amostras"):
        assert f"{milhar(carga[f'n_features_finitas_{lado}'])} features" in do_eixo, (lado, do_eixo)


def test_d551_trocar_o_par_apaga_a_linha_dos_qcs_por_lote(janela: JanelaPrincipal) -> None:
    """#8 da revisão de 15/10/2026: a linha seguia a validação ANTERIOR depois de outro par.

    `par_escolhido` não tinha consumidor, e só a resposta da validação reescrevia a linha. **Par
    causal:** a conta aparece com as amostras definidas, e o sinal do par novo a apaga.
    """
    janela.pipeline.define_amostras([1, 2, 3, 4], [True, False, True, True])
    janela.pipeline.n_lotes.setValue(2)
    assert janela.pipeline.qcs_por_lote.text().startswith("QCs por lote: "), "ancora: sem conta"
    janela.arquivos.par_escolhido.emit(object())
    assert janela.pipeline.qcs_por_lote.text() == "", "a conta da matriz anterior ficou na tela"


def test_d548_com_uma_correcao_em_curso_as_outras_recusam_e_a_medicao_tambem(qtbot: Any) -> None:
    """Uma correção por vez, nos dois sentidos — desde o caso dos 3.915 s.

    Um artefato avulso fica retido em curso pelo transporte; com ele em voo, o pipeline, o lote, o
    clique avulso na MESMA chave e a medição dos corretores recusam. **Par causal:**
    com a espera encerrada, o pipeline e a medição chegam às rotas.

    **Este teste já afirmou o contrário**, com o nome `..._e_a_medicao_nao`: a medição ficava fora
    da trava.
    """
    janela, transporte = _janela_que_retem(qtbot, retidos=99, intervalo_da_espera_ms=200)
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert _gira_ate(janela, lambda: bool(janela.esperas_em_curso()), 60), "a espera nao nasceu"
    antes = len(transporte.rotas)

    janela._dispara_etapa(ETAPAS[-1])
    assert janela.rotulo_de_desfecho.text() == CORRECAO_EM_CURSO
    janela._inicia_lote(CHAVE_DO_LOTE_METABO)
    assert janela._lote is None, "o lote comecou com uma correcao em curso"
    janela.resultado_da_correcao.gerar.emit("corrigida")
    posts = [r for r in transporte.rotas[antes:] if r.startswith("POST")]
    assert posts == [], f"com a correcao em curso, chegou a rota: {posts}"

    janela._dispara_medicao()
    assert janela.pipeline.rotulo_da_medicao.text() == CORRECAO_EM_CURSO
    # A recusa é síncrona, antes de `chama`; o giro dá tempo a um `POST` que tivesse saído.
    _gira_ate(janela, lambda: False, 1)
    medicoes = [r for r in transporte.rotas[antes:] if "medir-corretor" in r]
    assert medicoes == [], f"com a correcao em curso, a medicao chegou a rota: {medicoes}"

    # **A medição antes da Análise, e a ordem é o conteúdo desde 09/11/2026.** A Análise marca a
    # etapa em curso no clique, e ela continua em curso até o desfecho — que este transporte
    # retém —, então dispará-la primeiro travaria a medição pela razão certa. O «Executar
    # pipeline», que vinha primeiro até ali, não marcava etapa nenhuma.
    janela.para_as_esperas()
    janela._dispara_medicao()
    assert _gira_ate(
        janela, lambda: any("medir-corretor" in r for r in transporte.rotas[antes:]), 30
    ), "sem correcao em curso, a medicao tinha de disparar"
    janela.para_as_esperas()
    janela._dispara_etapa(ETAPAS[-1])
    assert _gira_ate(janela, lambda: f"POST {ROTA_DA_EXECUCAO}" in transporte.rotas[antes:], 30), (
        "sem correcao em curso, a Análise tinha de disparar"
    )


def test_d554_com_a_medicao_em_curso_as_correcoes_recusam(qtbot: Any) -> None:
    """O outro sentido da trava da medição, contra uma trava de mão única.

    Com a espera da medição viva, o artefato avulso, o pipeline e a própria medição recusam. **Par
    causal:** sem a espera, o artefato sai.
    """
    janela, _transporte = _janela_que_retem(qtbot, retidos=99, intervalo_da_espera_ms=200)
    chamadas: list[str] = []
    janela.chama = lambda metodo, rota, **_k: chamadas.append(rota)  # type: ignore[method-assign]
    janela._esperas[CHAVE_DA_MEDICAO] = _EsperaDoArtefato("id-da-medicao", leituras=1, inicio=0.0)

    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert janela.rotulo_de_desfecho.text() == CORRECAO_EM_CURSO
    janela._dispara_etapa(ETAPAS[-1])
    assert janela.rotulo_de_desfecho.text() == CORRECAO_EM_CURSO
    janela._dispara_medicao()
    assert janela.pipeline.rotulo_da_medicao.text() == CORRECAO_EM_CURSO
    assert chamadas == [], f"com a medicao em curso, chegou a rota: {chamadas}"

    janela._esperas.pop(CHAVE_DA_MEDICAO)
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert len(chamadas) == 1, "ancora: sem a medicao em curso, o artefato tinha de sair"


def test_d674_com_o_multiverso_a_correr_as_correcoes_e_o_palpite_recusam(qtbot: Any) -> None:
    """O multiverso na trava de uma correção por vez — o sentido em que ele corre primeiro.

    Parar o SERRF em vários núcleos encerra os processos de todo trabalho do programa, e o
    multiverso corre os corretores da grade ao lado. Com o painel dele a dizer que ele corre, o
    artefato avulso, o pipeline, a medição e o Palpite recusam com a frase própria. **Par causal:**
    a leitura do progresso que diz «terminou» destrava, e o artefato sai.
    """
    from refino.desktop.janela.principal import MULTIVERSO_EM_CURSO

    janela, _transporte = _janela_que_retem(qtbot, retidos=99, intervalo_da_espera_ms=200)
    chamadas: list[str] = []
    janela.chama = lambda metodo, rota, **_k: chamadas.append(rota)  # type: ignore[method-assign]
    janela.multiverso.aceita_disparo(202, {"identificador": "m-1", "total": 8, "persistido": True})
    assert janela.multiverso.a_correr(), "âncora: o disparo aceito não pôs o painel a correr"

    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert janela.rotulo_de_desfecho.text() == MULTIVERSO_EM_CURSO
    janela._dispara_etapa(ETAPAS[-1])
    assert janela.rotulo_de_desfecho.text() == MULTIVERSO_EM_CURSO
    janela._dispara_medicao()
    assert janela.pipeline.rotulo_da_medicao.text() == MULTIVERSO_EM_CURSO
    assert janela._dispara_o_palpite() is False
    assert chamadas == [], f"com o multiverso a correr, chegou a rota: {chamadas}"

    janela.multiverso.recebe_progresso(
        200, {"estado": "concluido", "terminou": True, "feito": 8, "total": 8}
    )
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert len(chamadas) == 1, "ancora: com o multiverso terminado, o artefato tinha de sair"


def test_d674_com_uma_correcao_em_curso_o_multiverso_recusa(qtbot: Any) -> None:
    """O outro sentido: uma trava que vale numa direção só deixa a outra passar.

    Com a espera do pipeline viva, o disparo do multiverso recusa antes de chegar à rota, e a frase
    vai para o rótulo do progresso dele. **Par causal:** sem a espera, o `POST /multiversos` sai.
    """
    janela, _transporte = _janela_que_retem(qtbot, retidos=99, intervalo_da_espera_ms=200)
    chamadas: list[str] = []
    janela.chama = lambda metodo, rota, **_k: chamadas.append(rota)  # type: ignore[method-assign]
    janela._esperas[CHAVE_DA_EXECUCAO] = _EsperaDoArtefato("id-da-execucao", leituras=1, inicio=0.0)

    janela.multiverso.disparar.emit(janela.multiverso.escolhas_do_disparo())
    assert janela.multiverso.rotulo_de_progresso.text() == CORRECAO_EM_CURSO
    assert chamadas == [], f"com a correcao em curso, o multiverso chegou a rota: {chamadas}"

    janela._esperas.pop(CHAVE_DA_EXECUCAO)
    janela.multiverso.disparar.emit(janela.multiverso.escolhas_do_disparo())
    assert chamadas == ["/multiversos"], (
        f"sem correcao em curso, o multiverso tinha de sair: {chamadas}"
    )


def _janela_da_trava_sem_o_par(qtbot: Any) -> tuple[JanelaPrincipal, list[str]]:
    """A janela da trava do multiverso SEM o par do laboratório: os dois testes acima pulam sem ele.

    **A recusa acontece antes de `_pedido_de_arquivos`**, então ela não depende do par: aqui o funil
    dos arquivos devolve «há arquivos» e a espera do Palpite «está em dia», e o que se mede é a
    fiação da trava. Nenhum arquivo de referência é fabricado — os duplos ficam no lugar de dois
    métodos da janela, e a âncora da rota real continua nos dois testes de cima.
    """
    janela = JanelaPrincipal(TestClient(build_app(), raise_server_exceptions=False))
    qtbot.addWidget(janela)
    janela.trabalhos.intervalo_da_releitura_ms = 3_600_000
    chamadas: list[str] = []
    janela.chama = lambda metodo, rota, **_k: chamadas.append(rota)  # type: ignore[method-assign]
    janela._pedido_de_arquivos = lambda: ({}, {})  # type: ignore[method-assign]
    return janela, chamadas


def test_d674_sem_o_par_o_multiverso_a_correr_trava_as_correcoes_e_o_palpite(qtbot: Any) -> None:
    """O sentido «multiverso primeiro», em toda máquina. **Par causal:** terminado, ele destrava.

    Sem o par o artefato iria ao funil dos arquivos duplicado, então a âncora de destrave é o motivo
    que a trava devolve, `None`, e não a rota — essa fica no teste com o par real, acima.

    **A primeira asserção é a do comportamento, antes de qualquer nome novo:** na árvore anterior à
    trava ela reprova porque o motivo é `None`, e não por `ImportError`. **E não há asserção sobre
    `chamadas` neste sentido, de propósito:** com o funil duplicado o artefato para em `par is None`
    antes da rota, e `chamadas == []` passaria sem trava nenhuma (medido na árvore anterior a ela) —
    asserção de ausência vácua. O que afirma a recusa é a frase em cada rótulo.
    """
    janela, _chamadas = _janela_da_trava_sem_o_par(qtbot)
    janela.multiverso.aceita_disparo(202, {"identificador": "m-1", "total": 8, "persistido": True})
    assert janela._motivo_para_nao_corrigir("corrigida") is not None, (
        "com o multiverso a correr, a trava de uma correção por vez não recusou"
    )
    assert janela.multiverso.a_correr(), "âncora: o disparo aceito não pôs o painel a correr"

    from refino.desktop.janela.principal import MULTIVERSO_EM_CURSO

    assert janela._motivo_para_nao_corrigir("corrigida") == MULTIVERSO_EM_CURSO
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert janela.rotulo_de_desfecho.text() == MULTIVERSO_EM_CURSO
    janela.rotulo_de_desfecho.setText("")
    janela._dispara_etapa(ETAPAS[-1])
    assert janela.rotulo_de_desfecho.text() == MULTIVERSO_EM_CURSO
    janela._dispara_medicao()
    assert janela.pipeline.rotulo_da_medicao.text() == MULTIVERSO_EM_CURSO
    assert janela._dispara_o_palpite() is False

    janela.multiverso.recebe_progresso(
        200, {"estado": "concluido", "terminou": True, "feito": 8, "total": 8}
    )
    assert janela._motivo_para_nao_corrigir("corrigida") is None, (
        "com o multiverso terminado, a trava continuou a recusar"
    )


def test_d674_sem_o_par_uma_correcao_em_curso_trava_o_multiverso(qtbot: Any) -> None:
    """O sentido «correção primeiro», corrido em toda máquina. **Par causal:** sem a espera, sai."""
    janela, chamadas = _janela_da_trava_sem_o_par(qtbot)
    janela._esperas[CHAVE_DA_EXECUCAO] = _EsperaDoArtefato("id-da-execucao", leituras=1, inicio=0.0)

    janela.multiverso.disparar.emit(janela.multiverso.escolhas_do_disparo())
    assert janela.multiverso.rotulo_de_progresso.text() == CORRECAO_EM_CURSO
    assert chamadas == [], f"com a correcao em curso, o multiverso chegou a rota: {chamadas}"

    janela._esperas.pop(CHAVE_DA_EXECUCAO)
    janela.multiverso.disparar.emit(janela.multiverso.escolhas_do_disparo())
    assert chamadas == ["/multiversos"], (
        f"sem correcao em curso, o multiverso tinha de sair: {chamadas}"
    )


def test_d548_com_o_pipeline_em_curso_o_artefato_avulso_recusa(qtbot: Any) -> None:
    """A direção que a primeira trava deixava livre: o pipeline primeiro, o artefato depois."""
    janela, _transporte = _janela_que_retem(qtbot, retidos=99, intervalo_da_espera_ms=200)
    # **O duplo REGISTRA as rotas.** Com `chama` mudo, a asserção
    # antiga, `"corrigida" not in janela._esperas`, passava por construção: só o `202` cria espera.
    chamadas: list[str] = []
    janela.chama = lambda metodo, rota, **_k: chamadas.append(rota)  # type: ignore[method-assign]
    janela._esperas[CHAVE_DA_EXECUCAO] = _EsperaDoArtefato("id-da-execucao", leituras=1, inicio=0.0)
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert janela.rotulo_de_desfecho.text() == CORRECAO_EM_CURSO
    assert chamadas == [], f"com o pipeline em curso, o artefato chegou a rota: {chamadas}"
    janela._esperas.pop(CHAVE_DA_EXECUCAO)
    janela.resultado_da_correcao.gerar.emit("corrigida")
    assert len(chamadas) == 1, "ancora: sem o pipeline em curso, o artefato tinha de sair"


def test_d551_a_perna_recusada_aborta_o_lote_em_vez_de_o_deixar_vivo(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """#5 da revisão de 15/10/2026: a perna recusada pela trava deixava o lote vivo.

    Uma execução que entra nas esperas entre duas pernas (ou durante o diálogo da pasta) faz
    `_gera_artefato` recusar a perna seguinte. Antes, o lote ficava vivo sem trabalho no servidor, o
    combo no corretor da perna recusada, e todo disparo de outra chave recebia `LOTE_EM_CURSO`.
    **Par causal:** a mesma janela, com a segunda perna livre e a terceira com a execução em espera;
    só a terceira é recusada, e é ela que aborta.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    chamadas: list[str] = []
    janela.chama = lambda metodo, rota, **_k: chamadas.append(rota)  # type: ignore[method-assign]
    escolhido = janela.pipeline.corretor.currentText()

    janela._inicia_lote(CHAVE_DO_LOTE_METABO)
    assert janela._lote is not None and len(chamadas) == 1, "ancora: a primeira perna nao saiu"
    janela._lote_recebe(b"csv-0")
    assert len(chamadas) == 2, "sem outra correcao em curso, a segunda perna tinha de sair"
    assert janela._lote is not None

    janela._esperas[CHAVE_DA_EXECUCAO] = _EsperaDoArtefato("id-da-execucao", leituras=1, inicio=0.0)
    janela._lote_recebe(b"csv-1")
    assert len(chamadas) == 2, "a terceira perna saiu com a execucao em curso"
    assert janela._lote is None, "o lote ficou vivo depois da perna recusada"
    assert janela.pipeline.corretor.currentText() == escolhido, "o combo nao voltou ao escolhido"
    assert janela.rotulo_de_desfecho.text() == LOTE_RECUSADO.format(
        metodo=CORRETORES[2], motivo=CORRECAO_EM_CURSO
    )
    assert len(list(tmp_path.glob("*.csv"))) == 2, "as duas pernas escritas tinham de ficar"
    janela._esperas.pop(CHAVE_DA_EXECUCAO)
    assert janela._motivo_para_nao_corrigir("processar") is None, "a janela ficou travada"


def test_d544_a_linha_de_qcs_por_lote_segue_o_numero_de_lotes(qtbot: Any) -> None:
    """Sete QCs do `pos_hilic` em três lotes dão três, dois e dois — e a tela passa a dizê-lo.

    As listas são as do arquivo real, pelo leitor; a mesma partição da correção é conferida contra
    as duas cópias dela em `tests/test_domain_lotes.py`.
    """
    from refino.io.readers import load_matrix

    matriz = load_matrix(measurements_path("pos_hilic"), sample_list_path("pos_hilic"), "pos_hilic")
    painel = _painel_de_pipeline(qtbot, [])
    assert painel.qcs_por_lote.text() == "", "sem validacao nao ha conta a mostrar"
    painel.define_amostras(matriz.injection_orders.tolist(), matriz.qc_mask.tolist())
    painel.n_lotes.setValue(3)
    assert painel.qcs_por_lote.text() == "QCs por lote: 3, 2 e 2."
    painel.n_lotes.setValue(painel.n_lotes.minimum())
    assert painel.qcs_por_lote.text() == "", "um lote so nao parte nada"
    painel.n_lotes.setValue(2)
    assert painel.qcs_por_lote.text().startswith("QCs por lote: ")
    # A fronteira: listas que nao conferem viram «sem dado», e nao uma conta sobre a lista errada.
    painel.define_amostras([1, 2, 3], [True, False])
    assert painel.qcs_por_lote.text() == ""


# ---------------------------------------------------------------------------
# Os CORTES do diagnóstico, da proposta até o pedido
# ---------------------------------------------------------------------------
#
# **Sem arquivo de referência, de propósito.** O que estes testes afirmam é o caminho dos cortes
# dentro da janela — guardados, usados só com o número que os propôs, e levados à linha da
# máquina —, e a resposta decorre das ordens dadas aqui. A partição sobre os arquivos reais é de
# `tests/test_domain_lotes.py`.

# Ordens de 68 injeções fora de ordem, como as colunas dos arquivos, e os QCs nas posições dos
# arquivos do laboratório — como ENTRADA, e não como afirmação sobre eles.
_ORDENS_D557 = list(range(68, 34, -1)) + list(range(1, 35))
_QC_D557 = [o in (1, 12, 23, 34, 45, 56, 68) for o in _ORDENS_D557]


def test_d557_os_cortes_viajam_so_com_o_numero_de_lotes_que_os_propos(qtbot: Any) -> None:
    """Três cortes não servem a dois lotes: mudado o número, a correção volta aos blocos iguais.

    **Par causal:** o mesmo painel manda os cortes com o número proposto e não os manda com outro.
    E a fronteira de confiança: o que não é lista de inteiros vira «sem cortes».
    """
    painel = _painel_de_pipeline(qtbot, [])
    assert painel.pre_preenche("n_batches", 2), "âncora: a proposta de dois lotes não entrou"
    painel.define_cortes_propostos([56])

    com_o_proposto = painel.campos_base("neg_hilic", "loess", 2)
    assert json.loads(com_o_proposto["batch_cuts"]) == [56], (
        f"os cortes não viajaram com o número que os propôs: {com_o_proposto.get('batch_cuts')}"
    )
    assert "batch_cuts" not in painel.campos_base("neg_hilic", "loess", 3), (
        "com outro número de lotes, os cortes da proposta viajaram — e não servem a ele"
    )

    painel.esquece_os_cortes_propostos()
    assert "batch_cuts" not in painel.campos_base("neg_hilic", "loess", 2), (
        "os cortes esquecidos continuaram a viajar"
    )
    # Cortes sem o número que os propôs — um painel cuja proposta de lotes não entrou — não viajam,
    # mesmo quando a contagem casa: é o número aceito que os torna da correção.
    sem_proposta = _painel_de_pipeline(qtbot, [])
    sem_proposta.define_cortes_propostos([56])
    assert sem_proposta.cortes_em_uso(2) is None, "cortes viajaram sem a proposta do número"
    for invalido in ([True], ["56"], "56", None, [56.0]):
        painel.define_cortes_propostos(invalido)
        assert painel.cortes_em_uso(2) is None, f"a forma {invalido!r} passou pela fronteira"


def test_d557_a_linha_dos_qcs_diz_qual_particao_a_correcao_vai_usar(qtbot: Any) -> None:
    """A conta da tela é a da partição que vai correr, e diz quando ela é a do diagnóstico.

    Com o corte em 56, cinco e dois QCs; com três lotes, que o diagnóstico não propôs, os blocos
    iguais dão três, dois e dois. **E um corte que não cabe nestas ordens** — de outro par — volta à
    conta dos blocos iguais em vez de derrubar a tela.
    """
    painel = _painel_de_pipeline(qtbot, [])
    painel.define_amostras(_ORDENS_D557, _QC_D557)
    assert painel.pre_preenche("n_batches", 2)
    painel.define_cortes_propostos([56])
    assert painel.qcs_por_lote.text() == (
        "QCs por lote: 5 e 2 — nas fronteiras que o diagnóstico propôs."
    )

    painel.n_lotes.setValue(3)
    assert painel.qcs_por_lote.text() == "QCs por lote: 3, 2 e 2."
    painel.n_lotes.setValue(2)
    assert painel.qcs_por_lote.text().endswith("nas fronteiras que o diagnóstico propôs."), (
        "de volta ao número proposto, a conta não voltou aos cortes dele"
    )

    painel.define_cortes_propostos([90])
    assert painel.qcs_por_lote.text() == "QCs por lote: 4 e 3.", (
        "um corte fora destas ordens tinha de cair nos blocos iguais"
    )

    # E os cortes sem o número proposto: a conta é a dos blocos iguais, que é o que vai correr —
    # `campos_base` não os manda (o teste vizinho), e a tela não pode contar sobre eles.
    sem_proposta = _painel_de_pipeline(qtbot, [])
    sem_proposta.define_amostras(_ORDENS_D557, _QC_D557)
    sem_proposta.define_cortes_propostos([56])
    sem_proposta.n_lotes.setValue(2)
    assert sem_proposta.qcs_por_lote.text() == "QCs por lote: 4 e 3.", (
        "a tela contou sobre cortes que o pedido não vai levar"
    )


def test_d557_a_linha_da_maquina_leva_os_lotes_que_ela_propos(qtbot: Any) -> None:
    """A linha da máquina já levou o número de lotes do HUMANO.

    Quem mudava os lotes propostos e mantinha o corretor ficava gravado como «concordando», porque
    só o corretor era trocado na spec da máquina. **Par causal:** a mesma escolha sem mexer nos
    lotes não tem eixo divergente, e não leva proposta nenhuma.
    """
    painel = _painel_de_pipeline(qtbot, [])
    assert painel.pre_preenche("n_batches", 2)
    painel.define_cortes_propostos([56])
    assert painel.eixos_divergentes() == (), "âncora: sem mexer, nada diverge"

    painel.n_lotes.setValue(3)
    eixos = painel.eixos_divergentes()
    assert eixos == ("n_batches",), f"mudar os lotes não divergiu: {eixos}"
    campos = campos_da_tela(painel, modo="neg_hilic", correcao="none", n_lotes=3, eixos=eixos)

    proposta = json.loads(campos["proposta_json"])
    spec_da_maquina = json.loads(proposta["spec_json"])
    assert campos["n_batches"] == 3 and "batch_cuts" not in campos
    assert spec_da_maquina["n_batches"] == 2, (
        f"a linha da máquina levou {spec_da_maquina['n_batches']} lotes — os do humano — e o par "
        "diria «concordando» sobre uma discordância"
    )
    assert spec_da_maquina["batch_cuts"] == [56]
    assert spec_da_maquina["correction"] == "none", "sem «Palpite», o corretor é o do humano"
    assert proposta["origem"] == "heuristica", "a proposta só do diagnóstico é a das regras"
    assert "56" in proposta["rationale"]


def test_d557_a_recusa_nomeia_o_botao_que_produz_a_proposta(qtbot: Any) -> None:
    """A mensagem antiga mandava «desmarcar os eixos», e as caixas saíram."""
    painel = _painel_de_pipeline(qtbot, [])
    comuns = {"modo": "neg_hilic", "correcao": "none", "n_lotes": 3}
    with pytest.raises(PropostaAusenteError) as corretor:
        campos_da_tela(painel, **comuns, eixos=["correction"])
    assert "«Palpite»" in str(corretor.value) and "o corretor" in str(corretor.value)
    with pytest.raises(PropostaAusenteError) as lotes:
        campos_da_tela(painel, **comuns, eixos=["n_batches"])
    assert "«Diagnóstico»" in str(lotes.value) and "o número de lotes" in str(lotes.value)
    assert "desmarque" not in str(corretor.value) + str(lotes.value)


def test_d557_a_recusa_da_medicao_chega_legivel_ao_lado_do_botao(janela: JanelaPrincipal) -> None:
    """A recusa de lote com menos de três QCs é o desfecho comum com os lotes propostos.

    No `pos_hilic`, os três lotes que o diagnóstico propõe dão 2, 2 e 3 QCs, e a rota recusa no
    disparo. A frase dela diz o que fazer, e o rótulo tem de a mostrar — não o dicionário em que ela
    viaja. **Par causal:** um corpo de erro sem `detail` continua a aparecer inteiro.
    """
    motivo = "com 3 lote(s), os QCs por lote são [2, 2, 3]"
    recusa = Resposta(
        codigo=422, ok=False, causa="usuario", mensagem="recusado", corpo={"detail": motivo}
    )
    janela._encaminha(CHAVE_DA_MEDICAO, recusa)
    assert janela.pipeline.rotulo_da_medicao.text() == f"A medição não completou: {motivo}"

    sem_detalhe = Resposta(
        codigo=500, ok=False, causa="ambiente", mensagem="falhou", corpo={"erro": "x"}
    )
    janela._encaminha(CHAVE_DA_MEDICAO, sem_detalhe)
    assert "erro" in janela.pipeline.rotulo_da_medicao.text(), "o corpo sem `detail` sumiu"


def test_d557_trocar_o_par_esquece_os_cortes_do_diagnostico(janela: JanelaPrincipal) -> None:
    """Os cortes são ordens de injeção de UMA corrida, e o par novo é outra.

    **Par causal:** com os cortes guardados e o número proposto, eles viajam; o sinal do par novo
    os apaga, e o mesmo pedido sai sem eles.
    """
    painel = janela.pipeline
    assert painel.pre_preenche("n_batches", 2)
    painel.define_cortes_propostos([56])
    assert "batch_cuts" in painel.campos_base("neg_hilic", "none", 2), "âncora: nada a esquecer"
    janela.arquivos.par_escolhido.emit(object())
    assert "batch_cuts" not in painel.campos_base("neg_hilic", "none", 2), (
        "os cortes da corrida anterior viajariam com o par novo"
    )


def test_d557_a_leitura_do_diagnostico_guarda_os_cortes_ao_lado_do_numero(
    janela: JanelaPrincipal,
) -> None:
    """O corpo do diagnóstico traz os cortes em `lotes.changepoints`, e a janela os passa ao painel.

    **Par causal:** um corpo sem `changepoints` deixa o painel sem cortes — e o pedido, sem eles.

    **Era o pré-preenchimento do número no campo**, que saiu: o padrão de «Quantos lotes
    declarar» é sempre 1. A leitura guarda o número e os cortes, e
    os cortes só valem depois de «Seguir sugestões do app» adotar o número — o segundo par causal.
    """
    corpo = {"lotes": {"n_lotes": 2, "changepoints": [56], "qcs_por_lote": [5, 2]}}
    assert janela._pre_preenche_com_a_proposta("diagnostico", corpo) == ()
    assert janela.pipeline.cortes_em_uso(2) is None, "os cortes valeram antes de a sugestão entrar"
    assert janela.pipeline.segue_a_proposta("n_batches") == 2, "a sugestão guardada não entrou"
    assert janela.pipeline.cortes_em_uso(2) == (56,)

    sem_cortes = {"lotes": {"n_lotes": 2, "qcs_por_lote": [4, 3]}}
    janela._pre_preenche_com_a_proposta("diagnostico", sem_cortes)
    assert janela.pipeline.cortes_em_uso(2) is None


def test_d350_o_botao_do_resultado_emite_o_sinal_do_lote_no_formato_escolhido(qtbot: Any) -> None:
    """A ligacao do botao, contra metodo vivo sem quem o chame.

    Sem esta assercao o botao existiria e nao faria nada, e a suite continuaria verde.

    **O sinal CARREGA a chave do artefato.** Eram dois botoes, um por formato; desde
    08/11/2026 e UM, o «Gerar os tres» do Resultado da Correcao, e o formato e o do seletor de la.
    Um sinal sem carga nao distinguiria os formatos: os dois lotes escreveriam os mesmos seis nomes
    na mesma pasta.
    """
    from refino.desktop.janela.execucao import ResultadoDaEtapa

    resultado = ResultadoDaEtapa(1, escolhedor=lambda: None)
    qtbot.addWidget(resultado)
    emitidos: list[str] = []
    resultado.gerar_os_tres.connect(emitidos.append)
    botao = resultado.gerar_os_tres_botao
    assert botao is not None, "ancora: o Resultado da Correcao nasce sem o «Gerar os tres»"

    botao.click()
    resultado.formatos.botoes[FORMATO_METABOANALYST].click()
    botao.click()

    assert emitidos == [CHAVE_DO_LOTE_CORRIGIDA, CHAVE_DO_LOTE_METABO], (
        f"o botao do lote emitiu {emitidos} — cada formato tem de nomear a SUA chave, e emitir a "
        f"mesma chave nos dois seria um seletor decorativo"
    )
    normalizacao = ResultadoDaEtapa(2, escolhedor=lambda: None)
    qtbot.addWidget(normalizacao)
    assert normalizacao.gerar_os_tres_botao is None, (
        "a Normalizacao nao tem lote: os tres corretores sao da Correcao"
    )


def test_d350_a_janela_liga_o_sinal_do_lote(transporte_real: Any, qtbot: Any) -> None:
    """A outra metade da ligacao: o sinal chega a `_inicia_lote`.

    As duas metades sao testes SEPARADOS de proposito — um botao ligado a nada e um metodo que
    ninguem liga sao defeitos diferentes, e uma assercao unica nao os distinguiria.
    """
    janela = _janela_com_lote(transporte_real, qtbot, None)
    chamadas: list[str] = []
    janela._inicia_lote = chamadas.append  # type: ignore[method-assign]
    janela.resultado_da_correcao.gerar_os_tres.connect(janela._inicia_lote)
    botao = janela.resultado_da_correcao.gerar_os_tres_botao
    assert botao is not None

    botao.click()

    # **`set` e nao lista, e a razao e do PySide6 e nao da producao — medido em 07/09/2026.** O
    # `connect` do `__init__` resolve `self._inicia_lote` no instante do sinal, entao remendar o
    # atributo faz as DUAS ligacoes chamarem o remendo: um clique produz duas entradas. O que este
    # teste afirma e que a CHAVE atravessa, e a contagem e artefato do arnes.
    assert set(chamadas) == {CHAVE_DO_LOTE_CORRIGIDA}, (
        f"o sinal do painel alcancou `_inicia_lote` com {chamadas} — a CHAVE tem de atravessar, "
        f"senao os dois botoes iniciam o mesmo lote"
    )


def test_item10_os_dois_lotes_escrevem_nomes_diferentes_na_mesma_pasta(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """O defeito que o segundo botao criaria, e ele estava a UMA string literal de distancia.

    `_lote_recebe` montava o nome com `nome_do_arquivo(lote.modo, "metaboanalyst", metodo)` — o
    formato ESCRITO A MAO. Com dois lotes, os dois escreveriam os mesmos tres nomes na mesma
    pasta: **tres arquivos onde deviam estar seis**, e o profissional levaria os do ultimo lote a
    acreditar que tem os dois formatos.

    **A ancora vem primeiro:** os dois conjuntos tem de ser nao vazios, senao a disjuncao abaixo
    passa por vacuidade.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela._gera_artefato = lambda chave: True  # type: ignore[method-assign]

    def _corre(chave: str) -> set[str]:
        janela._inicia_lote(chave)
        for i, _ in enumerate(CORRETORES):
            janela._lote_recebe(f"csv-{chave}-{i}".encode())
        return {p.name for p in tmp_path.glob("*.csv")}

    apos_o_primeiro = _corre(CHAVE_DO_LOTE_METABO)
    assert len(apos_o_primeiro) == len(CORRETORES), (
        f"ancora: o primeiro lote escreveu {sorted(apos_o_primeiro)} e devia escrever "
        f"{len(CORRETORES)} arquivos — sem eles, a comparacao abaixo nao tem o que comparar"
    )

    apos_os_dois = _corre(CHAVE_DO_LOTE_CORRIGIDA)
    assert len(apos_os_dois) == 2 * len(CORRETORES), (
        f"os dois lotes deixaram {len(apos_os_dois)} arquivos na pasta e deviam deixar "
        f"{2 * len(CORRETORES)}: {sorted(apos_os_dois)}. O segundo apagou o primeiro, e o "
        f"profissional leva tres arquivos a acreditar que tem os dois formatos"
    )


# ---------------------------------------------------------------------------
# O sétimo eixo do pipeline passa a ser do usuário
# ---------------------------------------------------------------------------


def test_u8_o_vazio_do_corte_de_ausentes_nao_viaja_e_o_zero_e_impossivel(qtbot: Any) -> None:
    """**O par causal do widget, e ele é a razão inteira de o widget existir.**

    `missing_threshold` é `float | None`. Um `0` que viajasse significaria duas coisas — *«corte de
    zero por ausentes»*, que descarta feature nenhuma, e *«sem corte»*, que deixa o padrão do
    servidor valer. O Qt resolve-as com `specialValueText` no lugar do mínimo.

    **As duas direções na mesma função:** em repouso o campo não entra no payload; com valor
    escolhido, entra com aquele valor.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    assert painel.escolhas_do_disparo()["missing_threshold"] is None, (
        "o campo em repouso mandou um número: o vazio está a viajar como zero"
    )

    painel.ausentes.setValue(0.35)
    assert painel.escolhas_do_disparo()["missing_threshold"] == pytest.approx(0.35), (
        "o valor escolhido não chegou às escolhas do disparo"
    )


def test_u8_o_setimo_eixo_chega_ao_payload_do_processar_e_so_dele(qtbot: Any) -> None:
    """**Só o `processar` o aceita**, e mandá-lo às outras duas rotas seria pior que nada.

    Campo não declarado é descartado em silêncio pelo FastAPI, e o payload passaria a afirmar uma
    influência que a rota não tem — o argumento que `CAMPOS_POR_ARTEFATO` já faz para os outros.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.ausentes.setValue(0.4)

    processar = painel.campos_do_artefato("normalizada")
    assert processar.get("missing_threshold") == pytest.approx(0.4), (
        f"o sétimo eixo não chegou ao payload do `processar`: {processar}"
    )
    for chave in ("corrigida", "corrigida_metaboanalyst"):
        assert "missing_threshold" not in painel.campos_do_artefato(chave), (
            f"o sétimo eixo vazou para o payload do `{chave}`, que não o aceita"
        )


def test_u8_em_repouso_o_setimo_eixo_nao_entra_em_payload_nenhum(qtbot: Any) -> None:
    """A outra direção do condicional. **Sem esta, o teste acima passaria com um `if` sempre
    verdadeiro**, e o padrão do servidor deixaria de valer sem ninguém o ter pedido.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    # **As chaves são as dos Resultados desde 09/11/2026**, e a âncora vai antes: com as três
    # chaves da aba que saiu, `campos_do_artefato` devolve `{}` para chave desconhecida, e a
    # ausência abaixo passaria sem olhar campo nenhum.
    assert "missing_threshold" in CAMPOS_POR_ARTEFATO["normalizada"], (
        "âncora: nenhum download declara o sétimo eixo, e a ausência abaixo seria vácua"
    )
    for chave in ARTEFATOS_DA_JANELA:
        assert "missing_threshold" not in painel.campos_do_artefato(chave), (
            f"o campo em repouso entrou no payload do `{chave}`"
        )


# ---------------------------------------------------------------------------
# O par «palpite antes / recomendação depois», e a troca de vocabulário
# ---------------------------------------------------------------------------


def test_d376_o_diagnostico_separa_as_duas_frases_com_ponto_e_nao_so_com_espaco() -> None:
    """A JUNÇÃO, e não as partes — o defeito vivia exactamente entre elas.

    **Os dois testes que já tocavam esta frase usam contenção de substring**, e substring passa
    igual com e sem o separador: `SEM_MEDICAO_DA_CLASSIFICACAO in texto` e `"87%" in texto` são
    verdadeiros nos dois mundos. Por isso este afirma o que eles não podem — que o rótulo do
    defeito **termina** e a frase da confiança **começa**.

    Cobre os CINCO tipos porque nenhum rótulo de `ROTULOS_DE_DEFEITO` termina em pontuação: o
    defeito não era do `erratic`, era da junção.
    """
    for tipo in ROTULOS_DE_DEFEITO:
        texto = texto_do_diagnostico({"defect_type": tipo, "confidence": 0.65})

        # O rótulo abre a primeira frase e chega em maiúscula; a colagem antiga é acusada
        # nas duas grafias da palavra seguinte.
        rotulo = com_maiuscula(ROTULOS_DE_DEFEITO[tipo])
        assert rotulo in texto, f"ÂNCORA: o rótulo de {tipo} nem aparece: {texto!r}"
        assert f"{rotulo}. " in texto, (
            f"as duas frases de {tipo} estão coladas sem ponto: {texto!r}"
        )
        for palavra in ("confiança", "Confiança"):
            assert f"{rotulo} {palavra}" not in texto, (
                f"a colagem antiga voltou em {tipo}: {texto!r}"
            )


def test_d376_o_par_causal_do_separador_a_forma_colada_seria_acusada() -> None:
    """A planta, e ela cai sobre dado em que as duas coisas DIFEREM.

    Sem esta, o teste acima poderia estar a afirmar algo que a forma antiga também satisfaz. A
    planta reconstrói a concatenação antiga — um espaço só — e afirma que a asserção
    do irmão **reprova** sobre ela.
    """
    tipo = "erratic"
    rotulo = com_maiuscula(ROTULOS_DE_DEFEITO[tipo])
    colado = f"{rotulo} " + com_maiuscula(prosa_da_confianca_do_diagnostico(tipo, 0.65))
    atual = texto_do_diagnostico({"defect_type": tipo, "confidence": 0.65})

    assert colado != atual, (
        "ÂNCORA: a forma colada e a actual são iguais, então esta planta não separa nada"
    )
    assert f"{rotulo}. " not in colado, "a planta não reproduz o defeito que diz reproduzir"
    assert f"{rotulo}. " in atual


def test_o_painel_do_palpite_mostra_o_palpite_e_a_confianca_e_mais_nada() -> None:
    """A DIREÇÃO desta asserção foi invertida em 06/09/2026, e a inversão é deliberada.

    ## O que este teste afirmava, e por que deixou de afirmar

    A ressalva medida já esteve **dentro desta frase**, com o argumento de que trocar o
    rótulo para «Palpite» *"é necessário e não é suficiente"*. Ficou o contrário: a frase traz só
    o palpite em si e a confiança da proposta.

    **O portão não foi APAGADO, foi virado** — e isso é deliberado. Um teste removido deixa a
    propriedade sem dono, e o próximo refactor que recolocasse o parágrafo não encontraria nada a
    reprovar. Aqui a propriedade nova tem a mesma força que a antiga tinha: o painel mostra **duas**
    frases, e o parágrafo de ressalva **não** volta por acidente.

    ## O que NÃO se perdeu, e é o que torna esta inversão aceitável

    A ressalva continua a viajar no corpo (`ressalva_da_politica`), continua a ser mostrada **no
    ponto em que se delega a escolha do corretor**, e continua com portão próprio —
    `test_a_ressalva_da_politica_sai_de_auto_caveats_e_nao_e_reescrita`, neste mesmo arquivo. O que
    mudou é o **lugar** dela, e o lugar é aquele em que alguém está a decidir.

    **Par causal:** o corpo carrega a ressalva, e o texto tem de sair sem ela. Sem a chave no corpo,
    a asserção de ausência passaria por vacuidade — não porque o painel a descarta, mas
    porque não havia o que descartar.
    """
    ressalva = "política APRENDIDA: mede 70,0% deixando um modo de fora"
    texto = texto_da_recomendacao(
        {
            "correction_proposta": "serrf",
            "avaliavel": True,
            "abstem": False,
            "confianca": 0.97,
            "ressalva_da_politica": ressalva,
        }
    )

    assert "serrf" in texto, f"ÂNCORA: o palpite nem aparece: {texto!r}"
    assert "97" in texto, f"a confiança da proposta não aparece: {texto!r}"
    assert ressalva not in texto, (
        f"a ressalva voltou para dentro da frase do palpite: {texto!r}. O lugar dela é o seletor "
        f"«Quem escolhe o corretor», que é onde alguém decide delegar"
    )
    assert "Isto é um palpite, não a recomendação." not in texto, (
        f"o parágrafo retirado voltou: {texto!r}"
    )
    assert "\n" not in texto, (
        f"o texto do palpite ganhou mais de um parágrafo: {texto!r}. São duas frases numa linha — "
        f"o corretor palpitado e a confiança dele"
    )


def test_d376_sem_ressalva_no_corpo_a_frase_sai_sem_ela_e_nao_com_none() -> None:
    """A outra direção, e sem ela o painel poderia imprimir `None` e ninguém notar.

    A ressalva é acréscimo honesto, não requisito de renderização: um corpo antigo — de um
    servidor que ainda não a serve — tem de produzir a frase de sempre.
    """
    texto = texto_da_recomendacao(
        {"correction_proposta": "loess", "avaliavel": True, "abstem": False, "confianca": 0.5}
    )

    assert "loess" in texto, f"ÂNCORA: o palpite nem aparece: {texto!r}"
    assert "None" not in texto, f"o campo ausente virou `None` na tela: {texto!r}"
    assert "Isto é um palpite, não a recomendação." not in texto


def test_d575_o_palpite_mostra_as_duas_opinioes_e_so_a_primeira_com_confianca() -> None:
    """As duas opiniões do Palpite, cada uma na sua linha, com o seu rótulo.

    **Par causal sobre a confiança:** a primeira linha traz a percentagem e a segunda não, e a
    âncora é que a percentagem EXISTE no texto — sem ela, a ausência na segunda linha passaria por
    vacuidade. A ausência é desenho: a margem da segunda é de erro contra a verdade, e a
    percentagem ao lado da da primeira mediria outra coisa com o mesmo nome.
    """
    from refino.cli import ROTULO_PELA_VERDADE, ROTULO_PELO_RSD

    texto = texto_da_recomendacao(
        {
            "rotulo_da_politica": ROTULO_PELO_RSD,
            "correction_proposta": "serrf",
            "avaliavel": True,
            "abstem": False,
            "confianca": 0.97,
            "segunda_opiniao": {
                "rotulo": ROTULO_PELA_VERDADE,
                "correction_proposta": "none",
                "caminho": "`is_gradual_drift` = 0.0000 ≤ 0.5000",
                "ressalva": "planta",
            },
        }
    )
    linhas = [linha for linha in texto.split("\n") if linha.strip()]
    assert len(linhas) == 2, f"esperava uma linha por opinião: {texto!r}"
    primeira, segunda = linhas
    # A primeira lembrava que era a que o «Automático» seguia; a caixa saiu, e quem a aplica
    # é «Seguir sugestões do app».
    assert ROTULO_PELO_RSD in primeira and "«Seguir sugestões do app» aplica" in primeira, primeira
    assert "serrf" in primeira and "97" in primeira, f"ÂNCORA: confiança da primeira: {primeira!r}"
    assert ROTULO_PELA_VERDADE in segunda and "none" in segunda, segunda
    assert "não decide" in segunda, segunda
    assert "%" not in segunda and "97" not in segunda, (
        f"a segunda opinião ganhou uma confiança na escala da primeira: {segunda!r}"
    )
    assert "planta" not in texto, "a ressalva da segunda opinião não é desta frase"


def test_d578_com_a_segunda_em_stand_by_o_palpite_tem_uma_linha_so() -> None:
    """A segunda opinião em stand-by: a rota manda `segunda_opiniao` nulo, e o painel dá a
    linha da primeira sozinha.

    **Par causal sobre o mesmo corpo**, que difere só pelo campo: com a segunda, duas linhas; com
    ela nula, uma — e a âncora de presença é a primeira opinião, com o rótulo e a confiança.
    """
    from refino.cli import ROTULO_PELA_VERDADE, ROTULO_PELO_RSD

    base = {
        "rotulo_da_politica": ROTULO_PELO_RSD,
        "correction_proposta": "serrf",
        "avaliavel": True,
        "abstem": False,
        "confianca": 0.97,
    }
    com = texto_da_recomendacao(
        base
        | {
            "segunda_opiniao": {
                "rotulo": ROTULO_PELA_VERDADE,
                "correction_proposta": "none",
                "caminho": "`is_gradual_drift` = 0.0000 ≤ 0.5000",
                "ressalva": "x",
            }
        }
    )
    sem = texto_da_recomendacao(base | {"segunda_opiniao": None})

    linhas = [linha for linha in sem.split("\n") if linha.strip()]
    assert len(linhas) == 1, sem
    assert ROTULO_PELO_RSD in sem and "serrf" in sem and "97" in sem, f"ÂNCORA: {sem!r}"
    assert ROTULO_PELA_VERDADE not in sem and "não decide" not in sem, sem
    assert ROTULO_PELA_VERDADE in com, "a planta não chegou ao texto: o par não separa nada"


def test_d376_o_rotulo_do_botao_e_palpite_e_nao_recomendacao() -> None:
    """A troca de vocabulário é condição, e ela tem portão.

    Sem isto, alguém restaura *«Recomendação»* num refactor e a ancoragem volta sem que nada
    acuse.
    """
    assert ROTULOS_DA_LEITURA["recomendacao"] == "Palpite", (
        f"o rótulo voltou a prometer mais do que a rota entrega: {ROTULOS_DA_LEITURA!r}"
    )
    assert "Recomendação" not in ROTULOS_DA_LEITURA.values()


# ---------------------------------------------------------------------------
# A RECOMENDAÇÃO MEDIDA na janela: o botão, a rota e o mostrador
# ---------------------------------------------------------------------------


def test_d376_o_botao_da_medicao_dispara_a_rota_certa_com_os_campos_que_ela_aceita(
    qtbot: Any,
) -> None:
    """O clique alcança `POST /matrizes/medir-corretor`, e **só** com os campos da assinatura.

    **A segunda metade é a que morde.** A rota aceita `measurements`, `sample_list`, `mode`,
    `orientation`, `n_sampled_features`, `paralelo`, `n_batches` e
    `batch_cuts`; e **não** `method` nem os eixos da grade.
    Reaproveitar `campos_do_artefato` mandaria campos que ela não declara, e o defeito chegaria
    à pessoa como `422` num botão que parece bom.

    **E há uma terceira metade, de PRESENÇA:** `n_batches` viaja sempre. Antes a
    medição corria com lote único mesmo quando a execução ia corrigir por lote, e o vencedor medido
    podia ser de outra configuração que a corrigida — sem nada a acusar.

    > **Esta docstring dizia *«e o portão de contrato acusa isso»*, e nenhum portão o faz.** A
    > medição está em `_dispara_medicao`, em `principal.py`, e a bijeção de
    > `test_contrato_janela_api.py` lê `contratos[("POST", ROTA_DA_EXECUCAO)]` e mais nada.
    > **Quem guarda esta rota é ESTE teste.**
    """
    janela, _transporte = _janela_que_retem(qtbot, retidos=0)
    enviados: list[dict[str, Any]] = []
    original = janela.chama

    def espia(metodo: str, rota: str, **kwargs: Any) -> None:
        enviados.append({"rota": rota, "campos": dict(kwargs.get("campos") or {})})
        original(metodo, rota, **kwargs)

    janela.chama = espia  # type: ignore[method-assign]
    # Dois lotes, e não o vazio: no mínimo do campo o valor do widget é `0` e o que viaja é `1`
    # (`LOTES_QUANDO_VAZIO`), e a asserção de presença abaixo compararia coisas diferentes.
    janela.pipeline.n_lotes.setValue(2)

    janela.pipeline.medir_botao.click()
    janela.espera_a_piscina(30_000)

    pedidos = [e for e in enviados if e["rota"] == "/matrizes/medir-corretor"]
    assert pedidos, (
        f"o clique não alcançou a rota da medição; as rotas chamadas foram "
        f"{[e['rota'] for e in enviados]}"
    )
    proibidos = {"method", "rsd_variant", "imputation", "scaling"}
    enviados_a_mais = proibidos & set(pedidos[0]["campos"])
    assert not enviados_a_mais, (
        f"o disparo levou campos que a rota não declara: {sorted(enviados_a_mais)}. Ela responde "
        "`422`, e o botão parece morto"
    )
    assert pedidos[0]["campos"].get("n_batches") == 2, (
        "a medição não levou o número de lotes do campo — ela voltaria a medir com lote único "
        f"enquanto a execução corrige por lote: {pedidos[0]['campos']}"
    )


def test_d376_o_mostrador_da_espera_da_medicao_e_o_painel_do_pipeline(qtbot: Any) -> None:
    """O despacho de uma linha, nas duas direções.

    Sem ele, o laço de espera escreveria no painel dos artefatos — cujo guarda devolve `""` para
    chave desconhecida e **falha em silêncio**. A barra da medição nunca apareceria, sem erro.
    """
    janela, _transporte = _janela_que_retem(qtbot, retidos=0)

    assert janela._mostrador_da_espera("medicao") is janela.pipeline, (
        "a espera da medição não vai para o painel que a mostra"
    )
    # O download da Correção vai ao Resultado dela desde que a aba «Executar e exportar» saiu.
    assert janela._mostrador_da_espera("corrigida") is janela.resultado_da_correcao, (
        "ÂNCORA: o despacho passou a mandar TUDO para o pipeline, e aí ele não separa nada"
    )


def test_d376_um_200_com_vencedor_escreve_a_medicao_no_painel(qtbot: Any) -> None:
    """O desfecho terminal: o corpo medido vira prosa no rótulo do pipeline.

    **A asserção é sobre o TEXTO ESCRITO e não sobre o retorno da função**, porque é o texto que a
    pessoa lê — e um método que devolvesse a frase certa sem a escrever passaria pelo outro.
    """
    janela, _transporte = _janela_que_retem(qtbot, retidos=0)
    corpo = {
        "vencedor": "none",
        "scores": {"none": 0.1399, "serrf": 0.1429, "loess": 0.1675},
        "margem": 0.0214,
        "avaliavel": True,
        "abstem": False,
        "n_features_medidas": 200,
        "palpite": "serrf",
        "palpite_acertou": False,
    }

    janela._encaminha(
        "medicao", Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=corpo), None
    )

    escrito = janela.pipeline.rotulo_da_medicao.text()
    assert "none" in escrito and "0,1399" in escrito, f"a medição não chegou à tela: {escrito!r}"
    assert "errou" in escrito, (
        f"o par palpite/medido não foi escrito, e é ele que torna o modelo aferível: {escrito!r}"
    )
    assert "%" not in escrito.split("margem")[0], (
        "os scores saíram com sinal de percentagem, e eles são FRAÇÕES, na célula que a "
        f"pessoa lê: {escrito!r}"
    )


def test_d376_o_par_causal_sem_o_corpo_o_painel_diz_que_ainda_nao_mediu(qtbot: Any) -> None:
    """A outra direção: rótulo em branco lê-se como «falhou», e a frase tem de dizer «ainda não».

    Sem este, o teste acima passaria com um painel que escrevesse a medição e nunca mais nada.
    """
    janela, _transporte = _janela_que_retem(qtbot, retidos=0)

    escrito = janela.pipeline.mostra_medicao(None)

    assert escrito.strip(), "o rótulo ficou em branco, e branco lê-se como falha"
    assert "ainda não correu" in escrito, f"a frase de espera não é a declarada: {escrito!r}"
    assert "minutos" in escrito, "a frase não diz o custo, e o botão dispara minutos de trabalho"


def test_d383_a_metade_numerica_da_proposta_desce_e_escreve_no_spin(janela: Any) -> None:
    """A entrada `diagnostico` de `PROPOSTAS_POR_ROTA`, que é a metade que nunca corria.

    ## Por que ela ficou de fora

    A tabela tem duas entradas e elas não se parecem: `recomendacao` traz `correction_proposta` no
    primeiro nível e vai a um `QComboBox`; `diagnostico` traz o número **aninhado** em
    `lotes.n_lotes` e vai a um `QSpinBox`. Todo o arnês exercitava a primeira — que é o valor
    natural, o que a tela mostra primeiro — e com ela ficavam por correr a descida de um nível, o
    ramo `setValue` de `_aplica_valor`, o ramo numérico de `_le_valor_pedido` e os três `return ()`.

    **O ramo numérico de `_le_valor_pedido` é o que mordia se quebrasse**, e em silêncio: sem a
    tradução, a comparação seria entre `"3"` e `3`, todo pré-preenchimento de número seria lido
    como recusa, e o eixo ficaria marcado como *"a máquina não conseguiu propor"* — que é o estado
    que a aba de justificativa mostra ao profissional.

    **A escrita no spin passou para o clique de «Seguir sugestões do app».** A
    leitura guarda o número sem o escrever — o campo fica sempre em 1 —, e é a adoção que
    percorre o ramo `setValue` e o ramo numérico da leitura de volta. A descida a `lotes.n_lotes` e
    os três `return ()` continuam a ser da leitura.
    """
    partida = janela.pipeline.n_lotes.value()
    proposto = partida + 2
    assert janela.pipeline.n_lotes.maximum() >= proposto, (
        f"âncora: o spin não alcança {proposto} e a proposta seria recusada pelo limite, não pelo "
        f"caminho que este teste mede"
    )

    mudados = janela._pre_preenche_com_a_proposta("diagnostico", {"lotes": {"n_lotes": proposto}})

    assert mudados == (), f"a leitura do diagnóstico escreveu na tela: {mudados}"
    assert janela.pipeline.n_lotes.value() == partida, "a leitura escreveu no spin"
    adotado = janela.pipeline.segue_a_proposta("n_batches")
    assert adotado == proposto, (
        f"a proposta aninhada não chegou ao eixo: {adotado}. Ou a descida a `lotes.n_lotes` "
        f"parou, ou `_escreve_valor` leu a escrita como recusa"
    )
    assert janela.pipeline.n_lotes.value() == proposto, (
        f"o spin ficou em {janela.pipeline.n_lotes.value()} e a proposta era {proposto}"
    )

    # **Os TRÊS `return ()`, e o `else` primeiro.** Ele dispara por ausência de condição, e é o mais
    # fácil de ficar inerte sem ninguém notar.
    assert janela._pre_preenche_com_a_proposta("diagnostico", ["nao e dicionario"]) == (), (
        "um corpo que não é dicionário passou pela primeira guarda"
    )
    assert janela._pre_preenche_com_a_proposta("validacao", {"lotes": {"n_lotes": 5}}) == (), (
        "uma rota que não propõe eixo nenhum foi tratada como se propusesse"
    )
    assert janela._pre_preenche_com_a_proposta("diagnostico", {"lotes": {}}) == (), (
        "um corpo sem `n_lotes` foi tratado como proposta"
    )
    assert janela.pipeline.n_lotes.value() == proposto, (
        f"uma das três guardas ESCREVEU no spin antes de desistir: "
        f"{janela.pipeline.n_lotes.value()}"
    )


def test_as_ressalvas_declaram_um_bloco_por_eixo_da_grade() -> None:
    """**Eixo sem entrada aqui é isenção INDECLARADA, e era o estado antes desta linha.**

    `RESSALVAS_POR_EIXO` é lido com `.get()`, então um eixo ausente cai em `None` e a tela
    desenha o combo sem ressalva — funcionalmente igual a um bloco vazio, e por isso
    indistinguível dele. A diferença é que o bloco vazio **declara** que ninguém achou
    recorte a mostrar, e a ausência não declara nada. É a mesma distinção que `SEM_REGISTRO`
    faz em `test_labels.py`, e a revisão apanhou o eixo novo do lado errado dela.
    """
    from refino.desktop.janela.pipeline import RESSALVAS_POR_EIXO
    from refino.interface.fluxo import EIXOS_DA_GRADE

    da_constante = {chave for chave, _r, _p, _o in EIXOS_DA_GRADE}
    assert da_constante, "âncora: a constante não declara eixo nenhum"
    assert set(RESSALVAS_POR_EIXO) == da_constante, (
        f"eixos sem bloco de ressalva: {sorted(da_constante - set(RESSALVAS_POR_EIXO))}; "
        f"blocos sem eixo: {sorted(set(RESSALVAS_POR_EIXO) - da_constante)}"
    )


def test_item10_um_clique_avulso_durante_o_lote_nao_o_estrangula(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """O lote sobrevive a um clique avulso, e antes NAO sobrevivia — 07/09/2026.

    ## A sequencia que a revisao adversarial MEDIU

    Com um lote em voo, clicar num artefato avulso fazia `_gera_artefato` correr
    `para_a_espera(chave)` e repostar. O `200` da perna do lote chegava depois, a guarda de
    supersessao descartava-o — e entao **`restantes` nao encolhia, nenhum arquivo era escrito, e
    `self._lote` ficava vivo para sempre**. A partir dali todo lote novo recebia *«ja ha um lote a
    correr»*, e fechar a janela era a unica recuperacao.

    ## Por que o segundo botao tornou isto grave

    O lote passou a ter DOIS formatos, e o novo e o `corrigir` — o artefato mais clicado. O gesto
    que estrangula e o gesto que se quer fazer passaram a ser o mesmo.

    ## Recusar e nao cancelar, e a escolha vai declarada

    Cancelar o lote por um clique avulso perderia quatro minutos de SERRF sem ninguem o pedir. A
    recusa e local, e o rotulo diz o que fazer.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    disparos: list[str] = []
    janela._gera_artefato_real = janela._gera_artefato  # type: ignore[attr-defined]

    janela._inicia_lote(CHAVE_DO_LOTE_CORRIGIDA)
    assert janela._lote is not None, "ancora: o lote nao abriu"
    restantes_antes = list(janela._lote.restantes)
    assert restantes_antes, "ancora: o lote abriu sem pernas"

    # O clique avulso NOUTRO artefato, com o lote em voo.
    janela._gera_artefato(CHAVE_DO_LOTE_METABO)
    del disparos

    assert janela._lote is not None, (
        "o clique avulso CANCELOU o lote — quatro minutos de SERRF perdidos sem ninguem o pedir"
    )
    assert janela._lote.restantes == restantes_antes, (
        f"o clique avulso consumiu uma perna do lote: {janela._lote.restantes} contra "
        f"{restantes_antes}"
    )
    assert janela.rotulo_de_desfecho.text() == LOTE_EM_CURSO, (
        f"a recusa nao foi dita a quem clicou: o rotulo diz {janela.rotulo_de_desfecho.text()!r}"
    )

    # **A ancora: o lote CONTINUA a andar depois da recusa.** Sem ela, uma implementacao que
    # travasse o lote inteiro passaria nas tres assercoes acima — ele ficaria vivo, intacto e
    # morto, que e exactamente o defeito que este teste existe para fechar.
    for i, _ in enumerate(restantes_antes):
        janela._lote_recebe(f"csv-{i}".encode())
    assert janela._lote is None, "o lote nao encerrou depois de receber as tres pernas"
    escritos = sorted(p.name for p in tmp_path.glob("*.csv"))
    assert len(escritos) == len(restantes_antes), (
        f"o lote sobreviveu a recusa mas nao escreveu os arquivos: {escritos}"
    )


def test_item5_o_palpite_da_execucao_segue_o_corretor_e_a_validacao(qtbot: Any) -> None:
    """O palpite depende de DUAS coisas, e o portao exercita as duas.

    `n_features` vem da validacao; o corretor vem do combo. Um palpite que so seguisse uma delas
    mostraria o numero do corretor errado, ou ficaria preso na frase de ausencia depois de a
    validacao ter voltado.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    # Sem validacao, a frase de ausencia — e ela e a ancora: sem esta linha, um painel que nunca
    # escrevesse nada passaria nas duas asercoes de diferenca abaixo.
    repouso = painel.rotulo_do_palpite_da_execucao.text()
    assert "Valida" in repouso, f"o repouso do palpite da execucao diz {repouso!r}"

    painel.mostra_palpite_da_medicao(2040)
    depois_da_validacao = painel.rotulo_do_palpite_da_execucao.text()
    assert depois_da_validacao != repouso, (
        "a validacao voltou e o palpite da execucao continua na frase de ausencia — ele nao ouve "
        "quem lhe traz `n_features`"
    )

    # A delegação era desligada aqui antes de mexer no combo; a caixa saiu, e o combo
    # é sempre o que decide.
    painel.corretor.setCurrentText("none")
    com_none = painel.rotulo_do_palpite_da_execucao.text()
    painel.corretor.setCurrentText("serrf")
    com_serrf = painel.rotulo_do_palpite_da_execucao.text()
    assert com_none != com_serrf, (
        f"trocar o corretor nao mudou o palpite: {com_none!r}. Ele descreve o corretor com que o "
        f"painel nasceu, e um numero apresentado sobre o corretor errado e pior que ausente"
    )


def test_d436_o_palpite_da_execucao_segue_o_seletor(qtbot: Any) -> None:
    """O palpite segue o corretor que a ROTA vai correr — e hoje ele é sempre o do seletor.

    ## O defeito antigo, e por que o teste mudou de lado

    A caixa «Automático» nascia ticada, e com ela o seletor ficava travado em `none`
    enquanto a política escolhia: o palpite lia o seletor e escrevia *«~3 s com `none`»* para uma
    execução que ia correr com `serrf`, **~4 min**. O conserto fez o palpite seguir a
    proposta sob delegação, e este teste afirmava três estados — delegada sem proposta, com
    proposta, e o seletor a NÃO desviar o palpite.

    **A caixa saiu, e a janela não delega mais**: o que vai correr é o que o seletor mostra.
    O teste afirma o lado novo — o palpite segue a proposta quando ela entra NO SELETOR, e segue o
    seletor quando a pessoa o muda —, e o par causal é o terceiro estado de antes, que exigia o
    contrário.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.mostra_palpite_da_medicao(6035)

    # (a) sem proposta: o palpite é o do corretor do seletor, e não diz que «não sabe»
    sem_proposta = painel.rotulo_do_palpite_da_execucao.text()
    assert "none" in sem_proposta and "delegada" not in sem_proposta, sem_proposta

    # (b) a proposta entra no seletor, e o palpite a segue — a ordem de grandeza muda
    assert painel.pre_preenche("correction", "serrf"), "âncora: a proposta não entrou no seletor"
    com_proposta = painel.rotulo_do_palpite_da_execucao.text()
    assert "serrf" in com_proposta and com_proposta != sem_proposta, com_proposta

    # (c) mexer no seletor desvia o palpite, porque é o seletor que vai correr
    painel.corretor.setCurrentText("loess")
    com_loess = painel.rotulo_do_palpite_da_execucao.text()
    assert "loess" in com_loess and com_loess != com_proposta, (
        f"o palpite não seguiu o seletor, que é o que a rota corre: {com_loess!r}"
    )

    # **Os DOIS palpites são rótulos DIFERENTES, e a asserção é sobre o widget e não sobre a
    # função.** Escrever os dois na mesma célula faria o profissional ler um pelo outro — são
    # populações diferentes, e a diferença é de uma ordem de grandeza.
    assert painel.rotulo_do_palpite.text() != painel.rotulo_do_palpite_da_execucao.text()


def test_u3_a_caixa_dos_nucleos_nasce_ticada_e_alcanca_os_QUATRO_payloads(qtbot: Any) -> None:  # noqa: N802
    """A caixa «usar vários núcleos» viaja nos quatro payloads que este painel monta.

    **Quatro e não um, e a contagem é o teste.** O painel alimenta `POST /execucoes` por
    `campos_da_execucao` e as três rotas de artefato por `campos_do_artefato`. Um painel que só
    ligasse a caixa a um deles daria o defeito mais confuso possível para quem usa: desticar mudaria
    o tempo de «Executar pipeline» e não o dos três botões de download, sem nada na tela a dizê-lo.

    **A âncora é o par causal:** a asserção afirma os dois estados, e não só o ticado. Sem o
    desticado, um `campos["paralelo"] = True` literal passaria em tudo.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    assert ARTEFATOS_DA_JANELA, "âncora: nenhum download, e a contagem abaixo perderia os três"

    assert painel.paralelo.isChecked(), (
        "a caixa «usar vários núcleos» nasceu desticada: o padrão é «Todos os núcleos», "
        "e um padrão serial poria o SERRF a custar 6,6x mais sem ninguém o pedir"
    )
    assert painel.paralelo.toolTip(), "a caixa nasceu sem dica — as irmãs deste painel têm"

    def _os_quatro() -> dict[str, Any]:
        return {
            "disparo": painel.escolhas_do_disparo()["paralelo"],
            "execucao": campos_da_tela(painel, modo="neg_c18", correcao="serrf", n_lotes=1)[
                "paralelo"
            ],
            # Os downloads dos Resultados, desde que os três da aba «Executar e exportar» saíram
            # com ela (09/11/2026); a âncora de que eles existem vem antes, abaixo.
            **{
                chave: painel.campos_do_artefato(chave)["paralelo"] for chave in ARTEFATOS_DA_JANELA
            },
        }

    ticada = _os_quatro()
    assert set(ticada.values()) == {True}, f"a caixa nasce ticada e o payload diz {ticada}"

    painel.paralelo.setChecked(False)
    desticada = _os_quatro()
    assert set(desticada.values()) == {False}, (
        f"desticar a caixa não alcançou todos os payloads: {desticada}. Quem ficou em `True` corre "
        f"o SERRF em paralelo depois de o usuário ter pedido o contrário"
    )


def test_u3_o_par_de_palpites_difere_por_mais_de_tres_vezes(qtbot: Any) -> None:
    """Ticar e desticar muda o NÚMERO na tela, e a âncora é a magnitude.

    **A asserção é sobre o fator e não sobre «são diferentes».** Dois textos diferentes seriam
    satisfeitos por uma vírgula: o que este teste protege é que o regime chegou à **conta**, e o
    fator medido entre série e paralelo no `serrf` é maior que 3 nas duas medições que este projeto
    tem — 5,5x fora do congelado e 6,6x dentro dele.

    **Só o `serrf`**, e o `loess` entra como âncora do lado oposto: ele lê a caixa, mas o
    fator dele dentro do executável não foi medido, e um palpite que mudasse com a caixa nele
    aplicaria um número que ninguém mediu.
    """
    from refino.interface.fluxo import (
        estimativa_da_execucao,  # noqa: F401  (documenta a origem do número)
    )

    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.mostra_palpite_da_medicao(2040)

    def _segundos(texto: str) -> float:
        import re

        achado = re.search(r"([\d.,]+)\s*(s|min|h)\b", texto)
        assert achado, f"o palpite não traz duração legível: {texto!r}"
        valor = float(achado.group(1).replace(".", "").replace(",", "."))
        return valor * {"s": 1.0, "min": 60.0, "h": 3600.0}[achado.group(2)]

    # A escolha passava a manual aqui antes de mexer no combo; a caixa saiu, e o
    # corretor é sempre o do seletor.
    painel.corretor.setCurrentText("serrf")
    painel.paralelo.setChecked(True)
    paralelo = _segundos(painel.rotulo_do_palpite_da_execucao.text())
    painel.paralelo.setChecked(False)
    serial = _segundos(painel.rotulo_do_palpite_da_execucao.text())

    assert serial / paralelo > 3.0, (
        f"o par de palpites do `serrf` difere por {serial / paralelo:.2f}x — menos de 3x. O regime "
        f"não chegou à conta: a tela promete o mesmo tempo nos dois estados da caixa"
    )

    painel.corretor.setCurrentText("loess")
    painel.paralelo.setChecked(True)
    loess_paralelo = painel.rotulo_do_palpite_da_execucao.text()
    painel.paralelo.setChecked(False)
    assert painel.rotulo_do_palpite_da_execucao.text() == loess_paralelo, (
        "o palpite do `loess` mudou com a caixa dos núcleos: o LOESS lê a caixa, mas "
        "o fator dele dentro do executável não foi medido, e um palpite que mudasse com ela "
        "aplicaria um número que ninguém mediu"
    )


# ---------------------------------------------------------------------------
# A linha do INVENTÁRIO de grupos, com a contagem
# ---------------------------------------------------------------------------

# **O corpo do diagnóstico na forma mínima que alimenta o inventário.** As quatro contagens são as
# medidas em 14/09/2026 sobre os `measurements_*.csv` do laboratório — 18, 16, 14 e 13 amostras
# não-QC, iguais nos quatro modos. **Elas são DISTINTAS entre si de propósito:** com quatro
# contagens iguais, uma implementação que escrevesse o número de grupos, ou uma constante, passaria
# — a planta não separaria.
_INVENTARIO_DO_ESTUDO: Final = (
    {"nome": "ATLL", "n": 13},
    {"nome": "Asymptomatic", "n": 18},
    {"nome": "HAMTSP", "n": 14},
    {"nome": "NoHTLV", "n": 16},
)


def test_u4_a_linha_do_inventario_traz_o_n_de_cada_grupo(qtbot: Any) -> None:
    """O painel diz quantas amostras tem cada grupo, antes de o contraste ser escolhido.

    **O oráculo é o rótulo do widget** e não o valor de retorno: um método que devolvesse o texto
    certo e não o escrevesse deixaria o painel em branco, que é o defeito que a revisão adversarial
    de `PainelDeQualidade` mediu em cinco mutantes.

    **A âncora de repouso vem ANTES:** sem ela, afirmar que o nome e o número estão na linha
    passaria sobre um rótulo semeado com o texto certo na construção.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    assert painel.rotulo_do_inventario.text() == INVENTARIO_EM_REPOUSO, (
        "âncora: o rótulo já nasce com o inventário, e o teste abaixo não mede a travessia"
    )

    inventario = inventario_do_corpo({CAMPO_DO_INVENTARIO: list(_INVENTARIO_DO_ESTUDO)})
    assert inventario is not None, "a travessia devolveu `None` sobre um corpo bem formado"
    escrito = painel.mostra_inventario_de_grupos(inventario)

    assert escrito == painel.rotulo_do_inventario.text(), "o método devolveu o que não escreveu"
    for entrada in _INVENTARIO_DO_ESTUDO:
        assert f"**{entrada['nome']}** ({entrada['n']})" in escrito, (
            f"o grupo {entrada['nome']!r} não aparece com o `n` dele: {escrito!r}"
        )
    assert "61 amostras" in escrito, f"o total não é a soma das quatro contagens: {escrito!r}"


def test_u4_a_linha_do_inventario_tem_TRES_desfechos_e_os_tres_sao_DISTINTOS(  # noqa: N802
    qtbot: Any,
) -> None:
    """*Quantos desfechos este mecanismo pode reportar, e eu vi todos?*

    São três e não dois, e o do meio é o que se perde: `None` é *"a leitura não correu"*, `()` é
    *"correu e não há grupo fora do QC"*, e com entradas é o inventário. Colapsar os dois primeiros
    mandaria quem ainda não leu o arquivo procurar um defeito que não existe.

    **Começa pelo ramo que dispara por AUSÊNCIA de condição**, que é o `None`: ele não tem o que
    plantar, e é o mais fácil de ficar inerte sem ninguém notar.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    desfechos = {
        "sem leitura": painel.mostra_inventario_de_grupos(None),
        "sem grupo": painel.mostra_inventario_de_grupos(()),
        "com grupos": painel.mostra_inventario_de_grupos(
            [GrupoNoInventario("ATLL", 13), GrupoNoInventario("NoHTLV", 16)]
        ),
    }

    assert all(texto.strip() for texto in desfechos.values()), (
        f"algum desfecho escreveu frase em branco: {desfechos}"
    )
    assert len(set(desfechos.values())) == 3, (
        f"dois desfechos escrevem a mesma frase: {desfechos}. «ainda não li» e «não há grupo» "
        "pedem ações opostas — clicar em «Leitura», ou conferir a linha `LABEL`"
    )
    assert desfechos["sem leitura"] == INVENTARIO_EM_REPOUSO
    assert desfechos["sem grupo"] == INVENTARIO_SEM_GRUPO


def test_u4_a_travessia_separa_CAMPO_AUSENTE_de_lista_vazia() -> None:  # noqa: N802
    """`None` e `()` são fatos diferentes, e o par causal cai sobre dado em que eles DIFEREM.

    Com um valor só para os dois, uma resposta que não traz o campo apagaria os combos do
    multiverso com a mesma naturalidade com que uma matriz sem grupos os apagaria — e `principal.py`
    guarda exactamente contra isso antes de chamar `define_grupos`.

    A última linha é a que impede a leitura otimista: **entrada malformada não levanta**, porque
    isto corre dentro de um slot de Qt, onde uma excepção é engolida para o `stderr` e o painel
    para de actualizar em silêncio.
    """
    assert inventario_do_corpo({}) is None, "campo ausente devia ser indistinguível de «não li»"
    assert inventario_do_corpo({CAMPO_DO_INVENTARIO: []}) == (), "lista vazia virou «não li»"
    assert inventario_do_corpo({CAMPO_DO_INVENTARIO: list(_INVENTARIO_DO_ESTUDO)}) == (
        GrupoNoInventario("ATLL", 13),
        GrupoNoInventario("Asymptomatic", 18),
        GrupoNoInventario("HAMTSP", 14),
        GrupoNoInventario("NoHTLV", 16),
    )

    for malformado in (
        None,
        {CAMPO_DO_INVENTARIO: "ATLL"},
        {CAMPO_DO_INVENTARIO: ["ATLL"]},
        {CAMPO_DO_INVENTARIO: [{"nome": "ATLL"}]},
        {CAMPO_DO_INVENTARIO: [{"nome": "ATLL", "n": "13"}]},
        {CAMPO_DO_INVENTARIO: [{"nome": 7, "n": 13}]},
    ):
        assert inventario_do_corpo(malformado) is None, (
            f"um corpo malformado passou como inventário: {malformado!r}"
        )


def test_u4_o_fio_do_diagnostico_alimenta_os_DOIS_consumidores_do_inventario(  # noqa: N802
    janela: JanelaPrincipal,
) -> None:
    """**O par que prova que a linha está ligada ao fio.**

    O inventário tem três consumidores, e dois deles vivem na janela: o combo do painel do
    multiverso, que precisa só dos **nomes**, e a linha do painel do pipeline, que precisa dos
    **pares**. Mudar o tipo satisfazendo um deles é um modo de falha já medido.

    **Os combos ficam com o NOME NU, e isso é afirmado e não suposto.** O texto do combo é o que
    `escolhas_do_disparo` envia à rota; um `"ATLL (13)"` lá dentro voltaria como `422` sobre um
    grupo que existe.

    **A âncora de repouso vem antes, nos dois painéis**: sem ela, as asserções de presença
    passariam sobre um estado que a construção já tinha produzido.
    """
    assert janela.pipeline.rotulo_do_inventario.text() == INVENTARIO_EM_REPOUSO, "âncora"
    assert janela.multiverso.grupo_a.count() == 0, "âncora: o combo já nasce com grupos"

    janela._encaminha(
        "diagnostico",
        Resposta(
            codigo=200,
            ok=True,
            causa=None,
            mensagem="",
            corpo={CAMPO_DO_INVENTARIO: list(_INVENTARIO_DO_ESTUDO)},
            conteudo=None,
        ),
    )

    escrito = janela.pipeline.rotulo_do_inventario.text()
    assert "**Asymptomatic** (18)" in escrito, escrito
    assert "61 amostras" in escrito, escrito

    nos_combos = [
        janela.multiverso.grupo_a.itemText(i) for i in range(janela.multiverso.grupo_a.count())
    ]
    assert nos_combos == ["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"], nos_combos
    assert not any("(" in item for item in nos_combos), (
        f"a contagem entrou no texto do combo: {nos_combos}. Ele viaja no disparo, e a rota "
        "recusaria com 422 um grupo que existe"
    )


# ---------------------------------------------------------------------------
# As caixas que pedem as análises, e o palpite que depende delas
# ---------------------------------------------------------------------------


def _analise_do_payload(painel: PainelDePipeline) -> dict[str, Any]:
    """O contrato de análise como ele viaja no corpo de `POST /execucoes`.

    **Lê o payload de PRODUÇÃO e não os widgets.** O que este bloco de testes tem de provar é que a
    caixa está ligada ao FIO; ler `analise_escolhida` afirmaria só que a caixa está ligada a si
    mesma — que é a distância exacta do defeito irmão deste painel, o corte de ausentes lido em
    `escolhas_do_disparo` e nunca posto no dicionário.
    """
    campos = campos_da_tela(painel, modo="pos_hilic", correcao="none", n_lotes=1)
    return dict(json.loads(campos[CAMPO_DA_ANALISE]))


def test_u23_o_bloco_das_analises_tem_uma_caixa_por_familia_e_por_classificador(
    qtbot: Any,
) -> None:
    """Bijeção entre as caixas e `FAMILIAS_DE_ANALISE` — e os classificadores por rótulo.

    Uma família nova na tabela sem caixa na tela seria oferecida à rota e invisível a quem opera;
    uma caixa sem família seria um controle que o pedido descarta em silêncio.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    bandeiras = {f.campo for f in FAMILIAS_DE_ANALISE if f.campo != CAMPO_DOS_CLASSIFICADORES}
    assert bandeiras, "ANCORA: a tabela não tem família nenhuma que seja bandeira"
    assert set(painel.analises) == bandeiras, (set(painel.analises), bandeiras)
    assert set(painel.classificadores) == set(get_args(ClassifierKind))


def test_u23_as_baratas_nascem_ligadas_e_as_caras_desligadas(qtbot: Any) -> None:
    """O estado de abertura, e o padrão vem do domínio — não de um literal nesta tela."""
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    ligadas = {campo for campo, caixa in painel.analises.items() if caixa.isChecked()}
    # A normalidade entrou ligada: custa milissegundos, e as baratas nascem ligadas.
    assert ligadas == {"anova", "correlation", "permanova", "normality"}, ligadas
    assert not any(caixa.isChecked() for caixa in painel.classificadores.values())
    # A âncora que separa *"o padrão é este"* de *"a tela ignora o padrão"*: o payload de abertura
    # tem de ser o padrão do domínio, e não uma reconstrução que coincida por acaso.
    #
    # **A comparação é contra o padrão SERIALIZADO e não contra o dataclass**, e a diferença é
    # real: `classifiers` é `()` no domínio e `[]` depois de atravessar o JSON. Comparar com a
    # tupla reprovaria por uma diferença que a fronteira introduz de propósito — e `_analise_ou_422`
    # faz a conversão de volta na outra ponta.
    esperado = json.loads(json.dumps(analise_para_json(PADRAO_DA_ANALISE)))
    assert _analise_do_payload(painel) == esperado


def test_u23_o_par_causal_do_fio_marcar_muda_o_corpo_e_desmarcar_desfaz(qtbot: Any) -> None:
    """**O par obrigatório desta unidade**: a caixa alcança o corpo da requisição.

    **Duas formas, porque o contrato tem duas.** Para `classifiers` o *"marcar acrescenta, desmarcar
    tira"* é literal — a lista cresce e encolhe. Para uma bandeira é o **valor** que troca, e tem de
    trocar: omitir a bandeira desligada faria a rota aplicar o padrão dela, e desticar «ANOVA»
    seria ignorado sem `422` e sem aviso.

    **A planta cai sobre dado em que as duas coisas DIFEREM:** `clustering` nasce desligado e
    `anova` nasce ligado, então cada metade move o campo na direção em que ele **não** já estava.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    abertura = _analise_do_payload(painel)
    assert abertura["clustering"] is False and abertura["anova"] is True, abertura
    assert abertura["classifiers"] == [], abertura

    painel.analises["clustering"].setChecked(True)
    painel.analises["anova"].setChecked(False)
    painel.classificadores["random_forest"].setChecked(True)
    depois = _analise_do_payload(painel)
    assert depois["clustering"] is True, depois
    assert depois["anova"] is False, depois
    assert depois["classifiers"] == ["random_forest"], depois

    # E o desfazer: sem ele, o teste provaria só que a tela sabe LIGAR.
    painel.analises["clustering"].setChecked(False)
    painel.analises["anova"].setChecked(True)
    painel.classificadores["random_forest"].setChecked(False)
    assert _analise_do_payload(painel) == abertura


def test_u23_o_palpite_cresce_na_direcao_quando_uma_familia_cara_e_marcada(qtbot: Any) -> None:
    """A DIREÇÃO e não o número.

    **Sem `n_features` o palpite é a frase de ausência**, então a validação é simulada pelo mesmo
    caminho que a janela usa: `mostra_palpite_da_medicao`, que é quem guarda o número.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.corretor.setCurrentText("serrf")
    painel.mostra_palpite_da_medicao(6035)

    antes = painel.rotulo_do_palpite_da_execucao.text()
    # **A agulha passou de «palpite:» a «estimativa» no merge de 17/09/2026, e o rótulo
    # é que manda.**
    # O rótulo foi renomeado, e este teste nasceu numa branch paralela, contra a palavra
    # anterior. É a forma «constante lida por dois mecanismos»: o rótulo mudou num
    # lado e a agulha ficou no outro, e nada os ligava. Quem decide é o rótulo, não o
    # teste — afrouxá-lo para casar os dois seria trocar o contrato pela conveniência do arnês.
    # **Sem distinguir maiúscula**: a janela passou a subir a primeira letra da frase,
    # e a agulha é a PALAVRA do rótulo, não a caixa dela.
    assert "estimativa" in antes.casefold() and "minutos" not in antes, antes

    painel.analises["clustering"].setChecked(True)
    depois = painel.rotulo_do_palpite_da_execucao.text()
    assert depois != antes, "o palpite não seguiu a caixa"
    assert "classe «minutos»" in depois and "agrupamento" in depois, depois

    painel.analises["clustering"].setChecked(False)
    assert painel.rotulo_do_palpite_da_execucao.text() == antes


def test_u23_o_rotulo_do_bloco_diz_o_que_a_barra_ganha_e_nao_repete_o_palpite(qtbot: Any) -> None:
    """Os TRÊS desfechos do rótulo, e o que ele **não** diz.

    O rótulo do bloco fala de progresso; o palpite, noutra coluna, fala de duração. Dois rótulos
    com o mesmo texto lado a lado é um defeito já medido neste painel.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    padrao = painel.rotulo_das_analises.text()
    assert "4 etapas a mais" in padrao and "anova" in padrao, padrao

    for caixa in painel.analises.values():
        caixa.setChecked(False)
    nenhuma = painel.rotulo_das_analises.text()
    assert "Nenhuma família pedida" in nenhuma, nenhuma

    painel.analises["dspc"].setChecked(True)
    uma = painel.rotulo_das_analises.text()
    assert "1 etapa a mais: rede" in uma, uma
    assert "palpite" not in uma.lower(), uma


def test_d663_o_teste_do_contraste_nasce_automatico_e_a_escolha_chega_ao_corpo(
    qtbot: Any,
) -> None:
    """«Automático», «Welch», «Mann-Whitney» — e o valor escolhido viaja em `analise_json`.

    **O oráculo é o payload de produção**, pela razão de `_analise_do_payload`: ler o widget
    afirmaria só que o seletor está ligado a si mesmo. **E a planta cai sobre dado em que as duas
    coisas diferem:** cada troca leva o campo para um valor em que ele não estava.

    **Ele vive no grupo do estudo e do contraste**, que é o da aba «Análise estatística». A frase
    terminava *«ao contrário das análises, que estão nas opções avançadas»*: aquele bloco era do
    painel montado sozinho e saiu em 09/11/2026 com a aba «Executar e exportar»; na janela, os dois
    grupos já moravam na página da Análise.
    """
    from refino.desktop.janela.componentes import ComboComNomes

    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    seletor = painel.teste_univariado
    assert isinstance(seletor, ComboComNomes)
    assert [seletor.itemText(i) for i in range(seletor.count())] == [
        "automatic",
        "welch",
        "mann_whitney",
    ]
    assert seletor.currentText() == "automatic" and seletor.nome_atual() == "Automático"
    assert painel.grupo_do_estudo.isAncestorOf(seletor)
    assert painel.grupo_do_estudo in painel.grupos_das_etapas[ETAPAS[-1]], (
        "o grupo do teste saiu da página da Análise"
    )
    assert _analise_do_payload(painel)["univariate_test"] == "automatic"

    nomes = {}
    for valor in ("welch", "mann_whitney"):
        seletor.setCurrentText(valor)
        nomes[valor] = seletor.nome_atual()
        assert _analise_do_payload(painel)["univariate_test"] == valor
    assert nomes == {"welch": "Welch", "mann_whitney": "Mann-Whitney"}

    seletor.setCurrentText("automatic")
    assert _analise_do_payload(painel)["univariate_test"] == "automatic"


def test_u23_as_analises_e_o_preprocessamento_vivem_nas_opcoes_avancadas(qtbot: Any) -> None:
    """A posição dos quatro grupos é o portão, e ela mudou duas vezes.

    **A primeira mudança:** a aba mostrava todos os controles de uma vez, e o painel passou a
    mostrar as duas decisões de toda corrida — o estudo e o contraste, e a correção do sinal — e a
    guardar o pré-processamento e as análises num bloco «Opções avançadas», fechado.

    **Desde 09/11/2026 o bloco não existe**: ele era a disposição do painel montado sozinho, que só
    a aba «Executar e exportar» usava, e ela saiu. Na janela os grupos já moravam nas abas das
    etapas, e é lá que se afirma
    a posição deles: cada grupo na página da etapa que o usa, e nenhum em duas.

    **A história da posição, que era de coluna:** as análises foram primeiro à direita por
    altura medida, e depois à esquerda pelo mesmo critério. As
    medições eram verdade quando foram feitas; o critério deixou de ser a altura.
    """
    from refino.desktop.janela.principal import INDICES_DAS_ABAS_DE_ETAPA

    janela = JanelaPrincipal(TestClient(build_app(), raise_server_exceptions=False))
    qtbot.addWidget(janela)
    grupos = {g.title(): g for gs in janela.pipeline.grupos_das_etapas.values() for g in gs}
    esperado = {
        "Correção do sinal": ETAPAS[0],
        "Pré-processamento e corte": 2,
        "Estudo e contraste": ETAPAS[-1],
        "Análises a correr": ETAPAS[-1],
    }
    paginas = {etapa: janela.abas.widget(i) for etapa, i in INDICES_DAS_ABAS_DE_ETAPA.items()}
    assert set(grupos) == set(esperado), f"os grupos das etapas mudaram: {sorted(grupos)}"
    for titulo, etapa in esperado.items():
        donas = [e for e, pagina in paginas.items() if pagina.isAncestorOf(grupos[titulo])]
        assert donas == [etapa], f"«{titulo}» está nas páginas {donas}, e não na da etapa {etapa}"


# ---------------------------------------------------------------------------
# O passo a passo na janela
# ---------------------------------------------------------------------------


def _janela_do_passo_a_passo(qtbot: Any) -> tuple[JanelaPrincipal, _TransporteQueGrava]:
    """A janela com o par real do `neg_c18`, o corretor `none` e a delegação desligada.

    **`none` e manual, e as duas escolhas são do recorte:** o que se mede é a ligação entre os
    botões, as rotas e as figuras, e o `none` faz a prévia correr em segundos. Com a caixa
    «Automático» ticada a política escolheria o corretor — e poderia escolher o SERRF.
    """
    transporte = _TransporteQueGrava(TestClient(build_app(), raise_server_exceptions=False))
    janela = JanelaPrincipal(transporte, intervalo_da_espera_ms=50)
    qtbot.addWidget(janela)
    janela.trabalhos.intervalo_da_releitura_ms = _SEM_RELEITURA_DA_FAIXA_MS
    janela.arquivos._par = _par_real()
    janela.arquivos.modo.setCurrentText("neg_c18")
    janela.pipeline.corretor.setCurrentText("none")
    return janela, transporte


def _habilitados(janela: JanelaPrincipal) -> list[bool]:
    return [botao.isEnabled() for botao in janela.pipeline.botoes_de_etapa.values()]


def test_d550_a_etapa_1_pela_rota_real_desenha_o_violino_sem_execucao_e_segue_os_parametros(
    qtbot: Any,
) -> None:
    """As propriedades abaixo pelo caminho real, com o `/etapas/correcao` do processo.

    - as três habilitam com ou sem etapa feita: desde as etapas independentes, a etapa k não pede
      mais a k-1 em dia, e a dependência entre elas é a da entrada escolhida;
    - o pedido da etapa 1 leva os campos da correção e mais nada;
    - o violino desenha sem nenhum `POST /execucoes`;
    - mudar a imputação não mexe na etapa 1; mudar o corretor a desatualiza e tira o violino.

    **Par causal do frescor:** voltar o corretor ao da etapa redesenha, sem pedido novo.
    """

    janela, transporte = _janela_do_passo_a_passo(qtbot)
    assert janela._atualiza_as_etapas() == ("nao_feita",) * 3
    assert _habilitados(janela) == [True, True, True]

    janela.pipeline.botoes_de_etapa[1].click()
    assert _gira_ate(janela, lambda: janela._atualiza_as_etapas()[0] == "em_dia", 120), (
        janela.rotulo_de_desfecho.text()
    )
    posts = [c for c in transporte.chamadas if c["metodo"] == "POST"]
    assert [c["rota"] for c in posts] == ["/etapas/correcao"]
    assert set(posts[0]["data"]) == {
        "mode",
        "correction",
        "n_batches",
        "rsd_variant",
        "seed",
        "paralelo",
    }
    assert set(posts[0]["files"]) == {"measurements", "sample_list"}
    assert _habilitados(janela) == [True, True, True]
    # O estado da Correção no rodapé dela — a linha combinada das três saiu com a aba
    # «Executar e exportar» (09/11/2026), e o estado de cada uma ficou no rodapé da sua.
    assert janela.pipeline.rotulos_de_etapa[1].text() == TEXTO_DO_ESTADO_DA_ETAPA["em_dia"]

    figuras = janela.graficos.figuras
    assert figuras.seleciona("figura_do_violino_do_rsd_por_feature")
    assert figuras.estado_visivel()["figura"] != "", "a etapa 1 nao desenhou o violino"

    imputacao = janela.pipeline.eixos["imputation"]
    imputacao.setCurrentIndex((imputacao.currentIndex() + 1) % imputacao.count())
    assert janela._atualiza_as_etapas()[0] == "em_dia", "a imputacao desatualizou a correcao"
    assert _habilitados(janela) == [True, True, True]

    janela.pipeline.corretor.setCurrentText("loess")
    assert janela._atualiza_as_etapas() == ("desatualizada", "nao_feita", "nao_feita")
    assert _habilitados(janela) == [True, True, True], "a etapa 1 fora de dia desligou as outras"
    assert figuras.repouso.text() == frase_da_etapa_desatualizada(1, 1)

    janela.pipeline.corretor.setCurrentText("none")
    assert janela._atualiza_as_etapas()[0] == "em_dia"
    assert figuras.estado_visivel()["figura"] != "", "de volta aos parametros e nao redesenhou"
    assert not any(c["rota"] == ROTA_DA_EXECUCAO for c in transporte.chamadas)
    assert [c["rota"] for c in transporte.chamadas if c["metodo"] == "POST"] == [
        "/etapas/correcao"
    ], "voltar aos parametros da etapa disparou pedido novo"


def test_d551_a_etapa_2_pela_rota_real_desenha_as_distribuicoes_e_recusa_a_combinacao_proibida(
    qtbot: Any,
) -> None:
    """#4 da revisão de 15/10/2026: o ramo da etapa 2 de `_dispara_etapa` não corria em teste.

    O clique no botão 2 com a etapa 1 em dia, contra o `/etapas/pre-processamento` real, e com o
    corte de RSD fora do mínimo — o ramo que o envia só corre com valor. **Par causal da
    recusa:** com a combinação proibida o clique não chega à rota nem deixa a etapa em
    curso; desfeita a combinação, o mesmo clique chega e as quatro distribuições desenham.
    """
    from refino.desktop.janela.graficos import ETAPA_DA_FIGURA

    janela, transporte = _janela_do_passo_a_passo(qtbot)
    janela.pipeline.corte.setValue(30.0)
    janela.pipeline.botoes_de_etapa[1].click()
    assert _gira_ate(janela, lambda: janela._atualiza_as_etapas()[0] == "em_dia", 120), (
        janela.rotulo_de_desfecho.text()
    )
    assert _habilitados(janela) == [True, True, True]

    imputacao = janela.pipeline.eixos["imputation"]
    original = imputacao.currentText()
    assert original != "none", "ancora: a imputacao padrao nao pode ser a da combinacao proibida"
    escalonamento = janela.pipeline.eixos["scaling"]
    itens = [escalonamento.itemText(i) for i in range(escalonamento.count())]
    escalonamento.setCurrentText(next(t for t in itens if t != "none"))
    imputacao.setCurrentText("none")
    antes = len(transporte.chamadas)
    janela.pipeline.botoes_de_etapa[2].click()
    assert [c for c in transporte.chamadas[antes:] if c["metodo"] == "POST"] == [], (
        "a combinacao proibida chegou a rota"
    )
    assert janela._etapas.em_curso() == (), "a recusa deixou a etapa 2 em curso"
    recusa = janela.rotulo_de_desfecho.text()
    assert recusa != "", "a recusa nao escreveu frase nenhuma"
    # A recusa vai ao aviso vermelho da aba da Normalização, logo acima do botão; a linha
    # combinada que a dizia saiu com a aba «Executar e exportar» em 09/11/2026.
    aviso = janela.pagina_da_normalizacao.faixa_da_etapa.frase.text()
    assert recusa in aviso, f"o aviso da Normalização não a diz: {aviso!r}"

    imputacao.setCurrentText(original)
    janela.pipeline.botoes_de_etapa[2].click()
    assert _gira_ate(janela, lambda: janela._atualiza_as_etapas()[1] == "em_dia", 120), (
        janela.rotulo_de_desfecho.text()
    )
    posts = [c for c in transporte.chamadas if c["metodo"] == "POST"]
    assert [c["rota"] for c in posts] == ["/etapas/correcao", "/etapas/pre-processamento"]
    enviados = set(posts[-1]["data"])
    assert {"imputation", "scaling", "rsd_cutoff"} <= enviados, sorted(enviados)
    assert not {"estudo", "analise_json", "campos_sobrescritos"} & enviados
    assert {"measurements", "sample_list"} <= set(posts[-1]["files"])
    assert _habilitados(janela) == [True, True, True]

    figuras = janela.graficos.figuras
    da_etapa_2 = sorted(f for f, etapa in ETAPA_DA_FIGURA.items() if etapa == 2)
    assert len(da_etapa_2) == 4, f"ancora: as figuras da etapa 2 sao {da_etapa_2}"
    for funcao in da_etapa_2:
        assert figuras.seleciona(funcao), funcao
        assert figuras.estado_visivel()["figura"] != "", f"a etapa 2 nao desenhou {funcao}"
    assert not any(c["rota"] == ROTA_DA_EXECUCAO for c in transporte.chamadas)


def _grava_as_chamadas(janela: JanelaPrincipal) -> list[tuple[str, dict[str, Any]]]:
    """`(rota, campos)` de cada `chama` da janela, no lugar da chamada: o que se afirma é o corpo
    que cruza a fronteira, e não o que a tela mostra."""
    chamadas: list[tuple[str, dict[str, Any]]] = []
    janela.chama = lambda _m, rota, **k: chamadas.append((rota, dict(k.get("campos") or {})))  # type: ignore[method-assign]
    return chamadas


def _janela_das_entradas(qtbot: Any) -> tuple[JanelaPrincipal, list[tuple[str, dict[str, Any]]]]:
    """A janela do passo a passo com as páginas no formulário, e as chamadas gravadas em vez de
    enviadas. O par entrou direto em `_par`, sem os arquivos escolhidos que o painel conta como
    metades, e a troca para o formulário é a que a página faz com o par completo."""
    janela, _transporte = _janela_do_passo_a_passo(qtbot)
    for pagina in janela._paginas_de_etapa.values():
        pagina.mostra_arquivos(())
    return janela, _grava_as_chamadas(janela)


def _desfecho(codigo: int, corpo: Any, *, causa: str | None = None, mensagem: str = "") -> Resposta:
    return Resposta(codigo=codigo, ok=codigo < 400, causa=causa, mensagem=mensagem, corpo=corpo)


def _conclui_a_etapa(janela: JanelaPrincipal, etapa: int, corpo: dict[str, Any]) -> None:
    """O `200` da etapa: o da prévia nas etapas 1 e 2, e o do `/execucoes` na 3."""
    janela._encaminha(CHAVE_DA_ETAPA.get(etapa, CHAVE_DA_EXECUCAO), _desfecho(200, corpo))


def _a_vista(widget: Any, pagina: Any) -> bool:
    return bool(widget.isVisibleTo(pagina))


def _marcada(seletor: Any) -> str | None:
    """O valor do segmento marcado no seletor da entrada, ou `None`."""
    return next((valor for valor, botao in seletor.botoes.items() if botao.isChecked()), None)


def test_etapa_3_sobre_a_original_manda_tudo_neutro_e_o_200_conclui_so_a_analise(
    qtbot: Any,
) -> None:
    """A Análise sem etapa nenhuma feita parte da matriz carregada, e o pedido diz isso.

    O corpo que vai ao `/execucoes` leva a correção e os sete eixos do pré-processamento neutros —
    os da tela não entram, e a imputação dela não é `none` —, o par de decisão vazio e nenhuma
    delegação. Com a 3 em curso, as três desligam. **O `200` conclui só a Análise**: a Correção e a
    Normalização continuam «não feita», porque não foram feitas. E um erro tira a Análise de «em
    curso», senão ela travaria toda correção da janela.
    """
    janela, chamadas = _janela_das_entradas(qtbot)
    assert janela.pipeline.eixos["imputation"].currentText() != "none", "âncora: a tela imputa"
    assert janela._atualiza_as_etapas() == ("nao_feita",) * 3, "ancora: nenhuma etapa feita"
    janela._dispara_etapa(3)
    assert [rota for rota, _ in chamadas] == [ROTA_DA_EXECUCAO], "a Análise esperou o Palpite"
    campos = chamadas[0][1]
    assert (campos["correction"], campos["n_batches"]) == ("none", 1)
    for eixo in ("imputation", "sample_normalization", "transformation", "scaling"):
        assert campos[eixo] == "none", eixo
    assert campos["campos_sobrescritos"] == "[]" and "proposta_json" not in campos
    assert "auto" not in campos, "a Análise delegou o corretor"
    assert janela.pipeline._pedido_do_disparo == (False, "none")
    assert janela._etapas.em_curso() == (3,)
    assert _habilitados(janela) == [False, False, False]

    _conclui_a_etapa(janela, 3, {"qualidade": None})
    assert janela._atualiza_as_etapas() == ("nao_feita", "nao_feita", "em_dia")

    janela._dispara_etapa(3)
    erro = _desfecho(422, {"detail": "x"}, causa="usuario", mensagem="x")
    janela._encaminha(CHAVE_DA_EXECUCAO, erro)
    assert janela._etapas.em_curso() == (), "o erro da execucao deixou a etapa 3 em curso"


def test_etapa_2_sem_correcao_mostra_a_frase_e_manda_a_correcao_neutra(qtbot: Any) -> None:
    """Uma entrada só: a frase diz qual e por que não há outra, e não há seletor.

    Par causal: com a Correção feita, o seletor aparece com a corrigida marcada.
    """
    janela, chamadas = _janela_das_entradas(qtbot)
    pagina = janela.pagina_da_normalizacao
    janela.pipeline.corretor.setCurrentText("loess")
    janela._atualiza_as_etapas()
    assert pagina.frase_da_entrada.text() == (
        "Usando a matriz sem correção — a correção não foi feita."
    )
    assert not _a_vista(pagina.seletor_da_entrada, pagina)
    janela._dispara_etapa(2)
    assert [rota for rota, _ in chamadas] == ["/etapas/pre-processamento"]
    assert (chamadas[0][1]["correction"], chamadas[0][1]["n_batches"]) == ("none", 1)
    _conclui_a_etapa(janela, 2, {})

    janela._dispara_etapa(1)
    _conclui_a_etapa(janela, 1, {"correcao": "loess"})
    janela._atualiza_as_etapas()
    assert _a_vista(pagina.seletor_da_entrada, pagina), "com a Correção feita não há seletor"
    assert pagina.frase_da_entrada.text() == "Escolha a matriz que entra na normalização:"


def test_etapa_2_depois_da_correcao_marca_a_corrigida_e_a_original_manda_none(qtbot: Any) -> None:
    """O seletor nomeia cada entrada pelo que ela é, com a padrão marcada; trocar para a original
    muda o pedido, e a escolha persiste — a página avisa que existe a mais tratada, com «Usar
    esta». A Correção guarda o corretor EFETIVO da resposta, e não o da tela.
    """
    janela, chamadas = _janela_das_entradas(qtbot)
    pagina = janela.pagina_da_normalizacao
    seletor = pagina.seletor_da_entrada
    janela._dispara_etapa(1)
    _conclui_a_etapa(janela, 1, {"correcao": "loess"})
    assert janela.pipeline.corretor.currentText() == "none", "âncora: a tela não pede o LOESS"
    janela._atualiza_as_etapas()
    nomes = {
        valor: botao.text() for valor, botao in seletor.botoes.items() if _a_vista(botao, pagina)
    }
    assert nomes == {"original": "Original", "corrigida": "Corrigida — LOESS"}
    assert _marcada(seletor) == "corrigida"
    janela._dispara_etapa(2)
    assert chamadas[-1][0] == "/etapas/pre-processamento"
    assert chamadas[-1][1]["correction"] == "loess", "a corrigida não levou o corretor efetivo"
    _conclui_a_etapa(janela, 2, {})

    seletor.botoes["original"].click()
    assert _marcada(seletor) == "original"
    assert _a_vista(pagina.usar_a_mais_tratada, pagina)
    assert "Corrigida — LOESS" in pagina.frase_da_mais_tratada.text()
    janela._dispara_etapa(2)
    assert chamadas[-1][1]["correction"] == "none", "a original levou a correção"
    _conclui_a_etapa(janela, 2, {})
    janela._atualiza_as_etapas()
    assert _marcada(seletor) == "original", "a escolha explícita não persistiu"

    pagina.usar_a_mais_tratada.click()
    assert _marcada(seletor) == "corrigida"
    assert not _a_vista(pagina.usar_a_mais_tratada, pagina)


def test_etapa_1_mudar_o_corretor_avisa_a_correcao_e_a_normalizacao_e_os_botoes_levam_a_refazer(
    qtbot: Any,
) -> None:
    """O aviso amarelo das duas abas, e o botão de cada um.

    Na Correção, os parâmetros mudaram e o botão a refaz. Na Normalização feita sobre a corrigida,
    o resultado dela continua «em dia», mas a entrada está fora de dia: o aviso diz qual, e o botão
    abre a aba da Correção. **Par causal:** antes de mudar o corretor, nenhuma das duas avisa.
    """
    from refino.desktop.janela.principal import ABAS, INDICES_DAS_ABAS_DE_ETAPA

    janela, chamadas = _janela_das_entradas(qtbot)
    correcao, normalizacao = janela.pagina_da_correcao, janela.pagina_da_normalizacao
    janela._dispara_etapa(1)
    _conclui_a_etapa(janela, 1, {"correcao": "none"})
    janela._dispara_etapa(2)
    _conclui_a_etapa(janela, 2, {})
    assert janela._atualiza_as_etapas() == ("em_dia", "em_dia", "nao_feita")
    assert correcao.faixa_da_etapa.isHidden() and normalizacao.faixa_da_etapa.isHidden()

    janela.pipeline.corretor.setCurrentText("loess")
    assert janela._atualiza_as_etapas() == ("desatualizada", "em_dia", "nao_feita")
    faixa = correcao.faixa_da_etapa
    assert not faixa.isHidden()
    assert faixa.property("nivel") == "aviso"
    assert faixa.texto == (
        "Atenção: Os parâmetros mudaram depois de corrigir; corrija de novo para o resultado valer."
    )
    assert faixa.botao.text() == "Corrigir de novo"
    dependente = normalizacao.faixa_da_etapa
    titulo = ABAS[INDICES_DAS_ABAS_DE_ETAPA[1]].titulo
    assert dependente.property("nivel") == "aviso"
    assert "A matriz corrigida, que entra nesta normalização, está fora de dia" in dependente.texto
    assert dependente.botao.text() == f"Ir para «{titulo}»"

    janela.abas.setCurrentWidget(normalizacao)
    dependente.botao.click()
    assert janela.abas.currentWidget() is correcao, "o aviso da dependente não abriu a Correção"
    antes = len(chamadas)
    faixa.botao.click()
    assert [rota for rota, _ in chamadas[antes:]] == ["/etapas/correcao"]
    assert chamadas[-1][1]["correction"] == "loess"
    assert faixa.isHidden(), "com a Correção a correr, o aviso dela continuou"

    # A Correção refeita com outro corretor: o resultado da Normalização deixou de valer pela
    # ENTRADA, e o aviso dela passa a refazê-la.
    _conclui_a_etapa(janela, 1, {"correcao": "loess"})
    assert janela._atualiza_as_etapas() == ("em_dia", "desatualizada", "nao_feita")
    assert faixa.isHidden()
    assert dependente.texto == (
        "Atenção: A correção foi refeita com outros parâmetros depois de normalizar; normalize de "
        "novo para o resultado valer."
    )
    assert dependente.botao.text() == "Normalizar de novo"
    dependente.botao.click()
    assert chamadas[-1][0] == "/etapas/pre-processamento"
    assert chamadas[-1][1]["correction"] == "loess"


@pytest.mark.parametrize(
    ("etapa", "matriz_mudou", "so_a_entrada", "comeco"),
    [
        (1, True, False, "A matriz carregada mudou depois de corrigir; corrija de novo"),
        (1, False, False, "Os parâmetros mudaram depois de corrigir; corrija de novo"),
        (3, False, False, "Os parâmetros ou a matriz de entrada mudaram depois de analisar"),
        (3, False, True, "A normalização foi refeita com outros parâmetros depois de analisar"),
    ],
)
def test_etapa_fora_de_dia_diz_a_causa_e_o_verbo_da_etapa(
    etapa: int, matriz_mudou: bool, so_a_entrada: bool, comeco: str
) -> None:
    from refino.desktop.janela.execucao import frase_da_etapa_fora_de_dia

    frase = frase_da_etapa_fora_de_dia(
        etapa,
        entrada="normalizada" if etapa == 3 else "original",
        matriz_mudou=matriz_mudou,
        so_a_entrada_refeita=so_a_entrada,
    )
    assert frase.startswith(comeco), frase
    assert frase.endswith("de novo para o resultado valer.")


def test_etapa_2_com_a_entrada_escolhida_de_outra_matriz_volta_a_original_e_diz_por_que(
    qtbot: Any,
) -> None:
    """Trocar o modo muda a matriz: a Correção feita deixa de ser entrada, a escolha explícita da
    Normalização volta à original com a faixa amarela da Entrada, e a Correção avisa que a matriz
    mudou. **Par causal:** de volta ao modo da Correção, a escolha volta e a faixa some.
    """
    janela, chamadas = _janela_das_entradas(qtbot)
    pagina = janela.pagina_da_normalizacao
    janela._dispara_etapa(1)
    _conclui_a_etapa(janela, 1, {"correcao": "loess"})
    janela._atualiza_as_etapas()
    pagina.seletor_da_entrada.botoes["corrigida"].click()
    assert pagina.faixa_da_volta.isHidden()

    janela.arquivos.modo.setCurrentText("pos_c18")
    janela._atualiza_as_etapas()
    assert not pagina.faixa_da_volta.isHidden()
    assert pagina.faixa_da_volta.texto == (
        "Atenção: A entrada «corrigida» deixou de existir (a correção feita é de outra matriz — o "
        "par de arquivos ou o modo mudou depois), e a seleção voltou à «original»."
    )
    assert not _a_vista(pagina.seletor_da_entrada, pagina), "a original sozinha não pede seletor"
    assert janela.pagina_da_correcao.faixa_da_etapa.texto.startswith(
        "Atenção: A matriz carregada mudou depois de corrigir"
    )
    janela._dispara_etapa(2)
    assert chamadas[-1][1]["correction"] == "none", "a entrada de outra matriz foi pedida"
    _conclui_a_etapa(janela, 2, {})

    janela.arquivos.modo.setCurrentText("neg_c18")
    janela._atualiza_as_etapas()
    assert pagina.faixa_da_volta.isHidden()
    assert _marcada(pagina.seletor_da_entrada) == "corrigida", "a escolha explícita não voltou"


def test_etapa_3_sobre_a_original_avisa_dos_zeros_e_sobre_a_normalizada_imputada_nao(
    qtbot: Any,
) -> None:
    """O aviso dos zeros não desliga nada: a Análise sobre a original dispara com ele à vista.

    **Par causal:** a mesma Análise sobre a normalizada com imputação não avisa; sobre uma
    normalizada sem imputação, avisa de novo, com o que trata os zeros.
    """
    janela, chamadas = _janela_das_entradas(qtbot)
    pagina = janela.pagina_da_analise
    faixa = pagina.faixa_da_imputacao
    janela._dispara_etapa(2)
    _conclui_a_etapa(janela, 2, {})
    janela._atualiza_as_etapas()
    assert _marcada(pagina.seletor_da_entrada) == "normalizada"
    assert faixa.isHidden(), "a normalizada imputada avisou dos zeros"

    pagina.seletor_da_entrada.botoes["original"].click()
    assert not faixa.isHidden()
    assert faixa.texto == (
        "Atenção: Esta matriz não passou pela imputação: os zeros entram como valor e podem "
        "distorcer a PCA, o PLS-DA e os testes. Para tratá-los, normalize antes."
    )
    assert janela.pipeline.botoes_de_etapa[3].isEnabled(), "o aviso dos zeros desligou a Análise"
    janela._dispara_etapa(3)
    assert chamadas[-1][0] == ROTA_DA_EXECUCAO
    assert chamadas[-1][1]["imputation"] == "none"
    _conclui_a_etapa(janela, 3, {"qualidade": None})

    janela.pipeline.eixos["sample_normalization"].setCurrentText("none")
    janela.pipeline.eixos["imputation"].setCurrentText("none")
    janela._dispara_etapa(2)
    _conclui_a_etapa(janela, 2, {})
    pagina.seletor_da_entrada.botoes["normalizada"].click()
    assert not faixa.isHidden(), "a normalizada sem imputação não avisou"
    assert faixa.texto.endswith("Para tratá-los, normalize de novo com uma imputação.")


def test_etapa_2_que_falha_mostra_a_causa_em_vermelho_e_volta_a_nao_feita(qtbot: Any) -> None:
    """A falha vira o aviso vermelho da aba, com a causa crua, e a etapa volta a «não feita».

    Par causal: disparar de novo tira o vermelho.
    """
    janela, _chamadas = _janela_das_entradas(qtbot)
    pagina = janela.pagina_da_normalizacao
    janela._dispara_etapa(2)
    assert janela._atualiza_as_etapas()[1] == "em_curso"
    recusa = _desfecho(422, {"detail": "o corte"}, causa="usuario", mensagem="o corte é inválido")
    janela._encaminha(CHAVE_DA_ETAPA[2], recusa)
    assert janela._atualiza_as_etapas()[1] == "nao_feita"
    faixa = pagina.faixa_da_etapa
    assert faixa.property("nivel") == "critico"
    assert faixa.texto == "Não foi possível: normalizar\n[causa: usuario] o corte é inválido"
    assert not faixa.botao.isVisibleTo(faixa)

    janela._dispara_etapa(2)
    assert faixa.isHidden(), "disparar de novo deixou a falha anterior à vista"


def test_nenhuma_etapa_espera_o_palpite_nem_delega(qtbot: Any) -> None:
    """A Normalização sobre a original recebe a correção neutra, e a Correção leva o corretor do
    seletor: nenhuma das duas pede o Palpite, e nenhuma manda `auto`.

    **Era `test_etapa_com_automatico_so_a_correcao_espera_o_palpite`**, e afirmava de
    propósito que a Correção, com a caixa «Automático», pedia o Palpite antes de sair. A caixa
    saiu, e o par causal é aquele desenho: o mesmo clique, que antes ia ao Palpite,
    vai agora à rota da Correção.
    """
    from refino.desktop.janela.leitura import ROTA_DO_PALPITE

    janela, chamadas = _janela_das_entradas(qtbot)
    janela._dispara_etapa(2)
    assert [rota for rota, _ in chamadas] == ["/etapas/pre-processamento"]
    assert "auto" not in chamadas[0][1]
    _conclui_a_etapa(janela, 2, {})
    janela._dispara_etapa(1)
    assert [rota for rota, _ in chamadas] == ["/etapas/pre-processamento", "/etapas/correcao"]
    assert ROTA_DO_PALPITE not in [rota for rota, _ in chamadas]
    assert "auto" not in chamadas[1][1], f"a Correção delegou: {chamadas[1][1]}"
    assert chamadas[1][1]["correction"] == janela.pipeline.corretor.currentText()


def test_etapa_3_leva_o_par_da_correcao_feita_na_corrigida_e_vazio_na_original(
    qtbot: Any,
) -> None:
    """O par humano-máquina que o `/execucoes` grava é a decisão da CORREÇÃO feita.

    A Correção é disparada com os lotes mudados em relação aos que o diagnóstico propôs; a Análise
    sobre a corrigida leva o eixo sobrescrito e a proposta montada no clique da Correção, com a
    spec reancorada no pedido da Análise. Sobre a original, os dois vão vazios.
    """
    janela, chamadas = _janela_das_entradas(qtbot)
    assert janela.pipeline.pre_preenche("n_batches", 2)
    janela.pipeline.n_lotes.setValue(3)
    assert janela.pipeline.eixos_divergentes() == ("n_batches",), "âncora: nenhum eixo sobrescrito"
    janela._dispara_etapa(1)
    _conclui_a_etapa(janela, 1, {"correcao": "none"})
    janela.pipeline.n_lotes.setValue(2)
    assert janela.pipeline.eixos_divergentes() == (), "âncora: a tela ainda sobrescreve"

    janela._dispara_etapa(3)
    campos = chamadas[-1][1]
    assert chamadas[-1][0] == ROTA_DA_EXECUCAO
    assert campos["campos_sobrescritos"] == '["n_batches"]'
    proposta = json.loads(campos["proposta_json"])
    spec = json.loads(proposta["spec_json"])
    assert (spec["n_batches"], spec["imputation"]) == (2, "none")
    assert campos["n_batches"] == 3 and proposta["origem"] == "heuristica"
    janela._encaminha(CHAVE_DA_EXECUCAO, _desfecho(422, {"detail": "x"}, causa="usuario"))

    janela.pagina_da_analise.seletor_da_entrada.botoes["original"].click()
    janela._dispara_etapa(3)
    campos = chamadas[-1][1]
    assert campos["campos_sobrescritos"] == "[]" and "proposta_json" not in campos


def test_d552_o_pipeline_inteiro_com_200_deixa_as_tres_etapas_em_dia(
    qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#10 da revisão de 15/10/2026.

    O «Executar pipeline» corria as três etapas; o `200` dele passava a deixá-las feitas como a
    cadeia que ele correu. **O botão saiu em 09/11/2026**, com a aba «Executar e exportar»; a
    mesma cadeia é a dos três botões, cada um sobre a entrada da etapa anterior — e o
    que este teste afirmava dela continua a valer. Pela rota real, com um SQLite temporário, como o
    `/execucoes` grava. **Par causal do frescor:** depois da cadeia, mudar o corretor desatualiza a
    Correção, e as duas que partem dela apontam para ela; voltar a ele as põe em dia outra vez.
    """
    from sqlalchemy import create_engine

    from refino.persistence.config import VARIAVEL_DA_URL, prepara_sqlite
    from refino.persistence.schema import METADATA

    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    prepara_sqlite(url)
    METADATA.create_all(create_engine(url))

    janela, transporte = _janela_do_passo_a_passo(qtbot)
    assert janela._atualiza_as_etapas() == ("nao_feita",) * 3
    for etapa, entrada in ((1, None), (2, "corrigida"), (3, "normalizada")):
        if entrada is not None:
            janela._escolhe_a_entrada(etapa, entrada)
        janela.pipeline.botoes_de_etapa[etapa].click()
        assert _gira_ate(
            janela, lambda e=etapa: janela._atualiza_as_etapas()[e - 1] == "em_dia", 180
        ), (etapa, janela.rotulo_de_desfecho.text(), [c["rota"] for c in transporte.chamadas])
    assert janela._atualiza_as_etapas() == ("em_dia",) * 3
    assert _marcada(janela.pagina_da_analise.seletor_da_entrada) == "normalizada"

    janela.pipeline.corretor.setCurrentText("loess")
    assert janela._atualiza_as_etapas() == ("desatualizada", "em_dia", "em_dia")
    assert janela._alvos_dos_avisos == {1: 1, 2: 1, 3: 1}
    janela.pipeline.corretor.setCurrentText("none")
    assert janela._atualiza_as_etapas() == ("em_dia",) * 3


def test_d550_com_uma_etapa_em_curso_o_pipeline_recusa_e_a_falha_da_chamada_solta(
    qtbot: Any,
) -> None:
    """O intervalo que a primeira trava não via: entre o clique e o primeiro `202`.

    **Par causal:** a mesma etapa em curso recusa o pipeline; depois de a chamada dela falhar sem
    desfecho de rota (`_falha`), o pipeline chega a `chama`. **O «pipeline» é a Análise desde
    09/11/2026**: é ela que dispara o `/execucoes`, e o botão que o fazia saiu com a aba «Executar e
    exportar».
    """
    janela, _transporte = _janela_do_passo_a_passo(qtbot)
    chamadas: list[str] = []
    janela.chama = lambda _m, rota, **_k: chamadas.append(rota)  # type: ignore[method-assign]
    atual = janela._campos_atuais()
    assert atual is not None
    janela._etapas.disparou(1, janela._fotos_da_tela(*atual)[0][1])

    janela._dispara_etapa(ETAPAS[-1])
    assert janela.rotulo_de_desfecho.text() == CORRECAO_EM_CURSO
    assert chamadas == []

    janela._falha("etapa_correcao", RuntimeError("sem desfecho"))
    assert janela._etapas.em_curso() == ()
    janela._dispara_etapa(ETAPAS[-1])
    assert chamadas == [ROTA_DA_EXECUCAO]


def test_d550_o_relogio_das_etapas_so_corre_com_a_janela_visivel(qtbot: Any) -> None:
    """Par causal do relógio: escondida, parado; mostrada, a correr e já atualizada; escondida
    outra vez, parado. Janela que ninguém vê não tem botão a habilitar."""
    janela, _transporte = _janela_do_passo_a_passo(qtbot)
    assert not janela._relogio_das_etapas.isActive(), "o relogio corre numa janela nunca mostrada"
    # O estado da Correção, no rodapé dela: a linha combinada «Etapas — …» saiu com a aba
    # «Executar e exportar» em 09/11/2026, e o que o relógio escreve é o estado de cada etapa.
    estado = janela.pipeline.rotulos_de_etapa[1]
    estado.setText("")
    janela.show()
    assert janela._relogio_das_etapas.isActive()
    assert estado.text() == TEXTO_DO_ESTADO_DA_ETAPA["nao_feita"], (
        "mostrar a janela nao atualizou o estado das etapas"
    )
    janela.hide()
    assert not janela._relogio_das_etapas.isActive()


def test_d550_todo_campo_que_a_tela_manda_tem_etapa(qtbot: Any) -> None:
    """A tabela de pertença contra o pedido REAL, com os campos opcionais todos presentes.

    Um campo novo do formulário fora de `ETAPA_DO_CAMPO` levanta dentro do relógio das etapas —
    e é este teste que o apanha antes, com a tela a mandar o máximo que sabe mandar.
    """
    from refino.desktop.etapas import ETAPAS, fotografia

    janela, _transporte = _janela_do_passo_a_passo(qtbot)
    janela.pipeline.grupo_a.setText("ATLL")
    janela.pipeline.grupo_b.setText("NoHTLV")
    campos = janela.pipeline.campos_base("neg_c18", "serrf", 2, rsd_cutoff=0.3)
    # `auto` saiu do conjunto: a janela não delega mais, e a tela não o manda.
    assert {"group_a", "group_b", "rsd_cutoff", "analise_json"} <= set(campos), sorted(campos)
    assert "auto" not in campos, "a tela voltou a mandar a delegação"
    for etapa in ETAPAS:
        fotografia(campos, etapa, {"matriz@1": "x"})


# ---------------------------------------------------------------------------
# O resumo metodológico e as referências dos métodos. Moravam na aba «Executar e
# exportar»; desde que ela saiu (09/11/2026) o resumo da Análise mora na «Qualidade do
# resultado» e as referências em «Métodos e limites».
# ---------------------------------------------------------------------------


def test_resumo_metodologico_sem_arquivo_diz_quando_aparece_e_mora_na_aba_de_execucao(
    janela: JanelaPrincipal,
) -> None:
    """O repouso: a frase diz quando o resumo aparece, e copiar e salvar ficam desligados.

    **A aba é a «Qualidade do resultado» desde 09/11/2026**, no mesmo ponto em que a qualidade da
    execução chega, e as referências estão em «Métodos e limites» — as duas moravam na aba
    «Executar e exportar», que saiu.
    """
    from refino.desktop.janela.principal import INDICE_DA_ABA_DA_QUALIDADE
    from refino.interface.referencias import SEM_RESUMO

    resumo = janela.qualidade.resumo
    assert resumo.caixa.toPlainText() == SEM_RESUMO
    assert not resumo.copiar.isEnabled()
    assert not resumo.salvar.isEnabled()
    assert _aba_que_mostra(janela, resumo) == INDICE_DA_ABA_DA_QUALIDADE
    # A lista das referências, com a conferência ao lado de cada obra, na aba dos métodos.
    indice = next(i for i, aba in enumerate(ABAS) if aba.paineis == ("recorte",))
    assert _aba_que_mostra(janela, janela.recorte.lista_das_referencias) == indice
    assert "R52 (só metadados)" in janela.recorte.lista_das_referencias.text()


def test_resumo_metodologico_de_uma_execucao_real_diz_o_que_o_pedido_enviou(
    qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOESS, semente 42, normalização pela soma e log10, pela rota real e com um SQLite temporário.

    O oráculo é o PEDIDO que o transporte gravou, e não o que a tela mostrava: o resumo tem de
    dizer o que viajou.
    """
    from sqlalchemy import create_engine

    from refino.interface.referencias import METODO_POR_CHAVE, SEM_RESUMO
    from refino.persistence.config import VARIAVEL_DA_URL, prepara_sqlite
    from refino.persistence.schema import METADATA

    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    prepara_sqlite(url)
    METADATA.create_all(create_engine(url))

    janela, transporte = _janela_do_passo_a_passo(qtbot)
    janela.pipeline.corretor.setCurrentText("loess")
    janela.pipeline.eixos["sample_normalization"].setCurrentText("sum")
    janela.pipeline.eixos["transformation"].setCurrentText("log10")
    # **A cadeia pelos três botões desde 09/11/2026**: o «Executar pipeline», que corria as três
    # etapas da tela de uma vez, saiu com a aba «Executar e exportar». A Análise leva a correção e a
    # normalização da entrada que a pessoa escolheu, e é por isso que ela parte da normalizada.
    for etapa, entrada in ((1, None), (2, "corrigida")):
        if entrada is not None:
            janela._escolhe_a_entrada(etapa, entrada)
        janela.pipeline.botoes_de_etapa[etapa].click()
        assert _gira_ate(
            janela, lambda e=etapa: janela._atualiza_as_etapas()[e - 1] == "em_dia", 180
        ), (etapa, janela.rotulo_de_desfecho.text())
    janela._escolhe_a_entrada(ETAPAS[-1], "normalizada")
    janela.pipeline.botoes_de_etapa[ETAPAS[-1]].click()
    resumo = janela.qualidade.resumo
    assert _gira_ate(janela, lambda: resumo.copiar.isEnabled(), 300), (
        janela.pipeline.rotulo_de_progresso.text()
    )

    enviado = next(
        c["data"]
        for c in transporte.chamadas
        if c["metodo"] == "POST" and c["rota"] == ROTA_DA_EXECUCAO
    )
    assert (enviado["correction"], enviado["seed"]) == ("loess", 42), enviado
    assert (enviado["sample_normalization"], enviado["transformation"]) == ("sum", "log10")
    texto = resumo.caixa.toPlainText()
    assert texto != SEM_RESUMO
    for chave in ("correction:loess", "sample_normalization:sum", "transformation:log10"):
        assert METODO_POR_CHAVE[chave].nome in texto, chave
    assert f"Semente: {enviado['seed']}." in texto
    assert f"modo {enviado['mode']}" in texto
    assert "[R7, R8, R9]" in texto
    assert "R52" not in texto, "sem par de grupos não há teste univariado a citar"


def test_resumo_metodologico_do_lote_dos_tres_traz_o_pedido_de_cada_correcao(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """Corrigir na etapa 1 com dois lotes e o corte proposto, e gerar os três: o resumo diz o
    corretor, a semente e os lotes que CADA pedido levou — e cada pedido os leva."""
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela.pipeline.pre_preenche("n_batches", 2)
    janela.pipeline.define_cortes_propostos([35])
    pedidos: list[dict[str, Any]] = []
    janela.chama = lambda _m, _r, **k: pedidos.append(dict(k["campos"]))  # type: ignore[method-assign]

    janela._inicia_lote(CHAVE_DO_LOTE_CORRIGIDA)
    for metodo in CORRETORES:
        assert pedidos and pedidos[-1]["method"] == metodo, pedidos
        janela._encaminha(CHAVE_DO_LOTE_CORRIGIDA, _resposta_de_artefato(f"csv-{metodo}".encode()))
    assert janela._lote is None, "o lote não encerrou"

    assert [(p["seed"], p["n_batches"], p["batch_cuts"]) for p in pedidos] == [
        (FIXOS_DECLARADOS["seed"], 2, "[35]")
    ] * len(CORRETORES), pedidos
    # **O lote é o «Gerar os três» do Resultado da Correção desde 08/11/2026**, e o resumo dele é o
    # daquela aba: o da Análise, na «Qualidade do resultado» desde 09/11/2026, descreve só a
    # execução.
    from refino.interface.referencias import SEM_RESUMO

    texto = janela.resultado_da_correcao.resumo.caixa.toPlainText()
    assert janela.qualidade.resumo.caixa.toPlainText() == SEM_RESUMO, (
        "o lote da Correção foi parar no resumo da Análise"
    )
    modo = janela.arquivos.campos_para_envio()["mode"]
    for metodo in CORRETORES:
        assert nome_do_arquivo(modo, CHAVE_DO_LOTE_CORRIGIDA, metodo) in texto, metodo
    assert texto.count(f"Semente: {FIXOS_DECLARADOS['seed']}.") == len(CORRETORES)
    corrigidos = [metodo for metodo in CORRETORES if metodo != "none"]
    assert texto.count("2 lotes, cortados antes das injeções 35") == len(corrigidos)
    assert "[R10, R26]" in texto and "[R7, R8, R9]" in texto


def test_resumo_metodologico_o_clique_avulso_descreve_o_arquivo_escrito(
    transporte_real: Any, qtbot: Any, tmp_path: Path
) -> None:
    """O gesto de um arquivo só: o resumo nasce na escrita, com o pedido daquele clique.

    **O clique é o do Resultado da Correção desde 09/11/2026**, sobre a Correção feita: o «Gerar»
    da aba «Executar e exportar» saiu com ela, e o resumo do arquivo é o da aba que o baixou.
    """
    janela = _janela_com_lote(transporte_real, qtbot, tmp_path)
    janela.chama = lambda *a, **k: None  # type: ignore[method-assign]
    _com_a_correcao_feita(janela, "loess")
    resultado = janela.resultado_da_correcao
    chave = "corrigida_metaboanalyst"
    resultado.pede_e_gera(chave)
    assert not resultado.resumo.copiar.isEnabled(), "âncora: antes da escrita não há resumo"

    janela._escreve_no_destino(chave, b"csv")

    texto = resultado.resumo.caixa.toPlainText()
    modo = janela.arquivos.campos_para_envio()["mode"]
    assert nome_do_arquivo(modo, chave, "loess") in texto
    assert "no formato do MetaboAnalyst" in texto
    assert "5.000 features" in texto
    assert resultado.resumo.copiar.isEnabled() and resultado.resumo.salvar.isEnabled()


def test_resumo_metodologico_copiar_e_salvar_levam_o_texto_inteiro(
    qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As três saídas do «Salvar como .txt» — escreve, cancela, falha — e o «Copiar»."""
    from PySide6.QtWidgets import QApplication

    from refino.desktop.janela.execucao import PainelDoResumo
    from refino.interface.referencias import ArquivoExportado, resumo_da_exportacao

    destinos: list[Path | None] = [tmp_path / "metodos", None]
    painel = PainelDoResumo(salvador=lambda _nome: destinos.pop(0))
    qtbot.addWidget(painel)
    resumo = resumo_da_exportacao(
        [ArquivoExportado("a.csv", "corrigir", {"mode": "neg_c18", "method": "loess", "seed": 42})]
    )
    painel.mostra(resumo)

    painel.copiar.click()
    assert QApplication.clipboard().text() == resumo.texto

    painel.salvar.click()
    escrito = tmp_path / "metodos.txt"
    assert escrito.read_text(encoding="utf-8") == resumo.texto, "o sufixo .txt não entrou"
    assert str(escrito) in painel.desfecho.text()

    painel.salvar.click()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["metodos.txt"], "cancelar escreveu"

    def nega(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError(13, "negado")

    destinos.append(tmp_path / "outro.txt")
    monkeypatch.setattr(Path, "write_text", nega)
    painel.salvar.click()
    monkeypatch.undo()
    assert "negado" in painel.desfecho.text(), painel.desfecho.text()
    assert not (tmp_path / "outro.txt").exists()


def test_resumo_metodologico_o_dialogo_de_producao_traduz_o_cancelamento(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """O diálogo de produção do «Salvar como .txt»: o caminho escolhido vira `Path`, e a string
    vazia do cancelamento vira `None` — `Path("")` seria o diretório corrente.

    **Pelo painel sem salvador injetado**: o diálogo é `arquivos.salvador_por_dialogo`,
    e o painel passa-lhe o título e o filtro de texto que a cópia apagada escrevia à mão.
    """
    from PySide6.QtWidgets import QFileDialog

    from refino.desktop.janela.execucao import (
        FILTRO_DO_RESUMO,
        NOME_SUGERIDO_DO_RESUMO,
        TITULO_DO_DIALOGO_DO_RESUMO,
        PainelDoResumo,
    )
    from refino.interface.referencias import ArquivoExportado, resumo_da_exportacao

    pedidos: list[tuple[str, str, str]] = []

    def dialogo(_pai: Any, titulo: str, sugerido: str, filtro: str) -> tuple[str, str]:
        pedidos.append((titulo, sugerido, filtro))
        return str(tmp_path / "metodos.txt"), filtro

    monkeypatch.setattr(QFileDialog, "getSaveFileName", dialogo)
    painel = PainelDoResumo()
    qtbot.addWidget(painel)
    resumo = resumo_da_exportacao(
        [ArquivoExportado("a.csv", "corrigir", {"mode": "neg_c18", "method": "loess", "seed": 42})]
    )
    painel.mostra(resumo)
    painel.salvar.click()
    assert (tmp_path / "metodos.txt").read_text(encoding="utf-8") == resumo.texto
    assert pedidos == [(TITULO_DO_DIALOGO_DO_RESUMO, NOME_SUGERIDO_DO_RESUMO, FILTRO_DO_RESUMO)]
    # As duas constantes nasceram no lugar de literais da cópia apagada; o que a mudança
    # declara preservar é o TEXTO daqueles literais, e só o literal o afirma.
    assert TITULO_DO_DIALOGO_DO_RESUMO == "Salvar o resumo metodológico", (
        "o título do diálogo do resumo deixou de ser o de antes"
    )
    assert FILTRO_DO_RESUMO == "Texto (*.txt)", "o filtro do resumo deixou de ser o de texto"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a: ("", ""))
    painel.salvar.click()
    assert painel.desfecho.text() == "Nada foi salvo: a escolha do arquivo foi cancelada."


# ---------------------------------------------------------------------------
# O Resultado da Correção e o da Normalização: o arquivo da etapa FEITA, nos dois formatos
# ---------------------------------------------------------------------------
#
# A alternativa está disponível, e o app deixa claro que é uma alternativa à recusa. O que estes
# testes afirmam é o que cruza a fronteira — a rota pedida, os
# campos que vão nela e o arquivo gravado na pasta —, e não as chamadas de dentro. Nenhuma rota
# corre: `chama` grava o pedido, e as respostas entram por `_encaminha` na forma que
# `cliente.envia` as produz.


class _NenhumaRota:
    """O transporte de uma janela cujo `chama` é gravado: chegar aqui é defeito do arnês."""

    def post(self, url: str, **_kwargs: Any) -> Any:
        raise AssertionError(f"o arnês grava os pedidos e não fala com rota nenhuma: {url}")

    def get(self, url: str, **_kwargs: Any) -> Any:
        raise AssertionError(f"o arnês grava os pedidos e não fala com rota nenhuma: {url}")


def _janela_dos_resultados(
    qtbot: Any, pasta: Path | None
) -> tuple[JanelaPrincipal, list[dict[str, Any]]]:
    """A janela com um par na tela, o corretor manual e os pedidos GRAVADOS.

    **Os bytes do par são escritos à mão porque nenhuma rota os lê**: o que se mede é o pedido
    montado e o arquivo escrito, e a identidade da matriz é o digest destes bytes. O modo vem do
    nome, como na escolha real.
    """
    janela = JanelaPrincipal(_NenhumaRota(), pasta=lambda: pasta)
    qtbot.addWidget(janela)
    janela.arquivos._par = ParDeArquivos(
        "measurements_neg_c18.csv", b"medidas", "Sample_List_neg_c18.csv", b"lista"
    )
    janela.arquivos.modo.setEditText("neg_c18")
    janela.pipeline.corretor.setCurrentText("none")
    pedidos: list[dict[str, Any]] = []
    janela.chama = lambda _metodo, rota, **k: pedidos.append({"rota": rota, **k})  # type: ignore[method-assign]
    return janela, pedidos


def _etapa_feita(janela: JanelaPrincipal, etapa: int, corpo: dict[str, Any]) -> None:
    """Dispara a etapa pelo botão dela e entrega o `200` da prévia. A âncora é a etapa feita."""
    janela._dispara_etapa(etapa)
    janela._encaminha(
        CHAVE_DA_ETAPA[etapa],
        Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=corpo),
    )
    assert janela._etapas.feita(etapa) is not None, f"âncora: a etapa {etapa} não ficou feita"
    janela._atualiza_as_etapas()


def _corrige(janela: JanelaPrincipal, corretor: str) -> None:
    janela.pipeline.corretor.setCurrentText(corretor)
    _etapa_feita(janela, 1, {"correcao": corretor})


def _normaliza(janela: JanelaPrincipal, **eixos: str) -> None:
    for eixo, valor in eixos.items():
        janela.pipeline.eixos[eixo].setCurrentText(valor)
    _etapa_feita(janela, 2, {})


def _motivo_ao_lado(botao: Any) -> str:
    """A frase do motivo do botão desligado — o rótulo cujo `buddy` é ele, como o portão a acha."""
    from PySide6.QtWidgets import QLabel

    pai = botao.parentWidget()
    return next((r.text() for r in pai.findChildren(QLabel) if r.buddy() is botao), "")


def _o_pedido(pedidos: list[dict[str, Any]], rota: str) -> dict[str, Any]:
    """Os campos do único pedido à `rota` — e reprova se não houve um, ou se houve mais."""
    achados = [p["campos"] for p in pedidos if p["rota"] == rota]
    assert len(achados) == 1, (
        f"esperava um pedido a {rota}, e vieram: {[p['rota'] for p in pedidos]}"
    )
    return dict(achados[0])


def test_resultado_da_correcao_antes_de_corrigir_diz_o_que_falta_e_o_lote_fica_ligado(
    qtbot: Any, tmp_path: Path
) -> None:
    """O download nasce desligado com a frase do gesto que o liga; «Gerar os três» não depende da
    correção feita. **As duas direções:** depois de corrigir, o download liga e a frase sai."""
    from refino.desktop.janela.execucao import MOTIVOS_SEM_ETAPA

    janela, _ = _janela_dos_resultados(qtbot, tmp_path)
    resultado = janela.resultado_da_correcao
    assert janela.pagina_da_correcao.bloco_do_resultado.isAncestorOf(resultado), (
        "o Resultado da Correção não está no bloco «Resultado» da aba dela"
    )
    janela._atualiza_as_etapas()
    assert not resultado.baixar.isEnabled()
    assert _motivo_ao_lado(resultado.baixar) == MOTIVOS_SEM_ETAPA[1]
    lote = resultado.gerar_os_tres_botao
    assert lote is not None and lote.isEnabled()

    _corrige(janela, "loess")
    assert resultado.baixar.isEnabled(), _motivo_ao_lado(resultado.baixar)
    assert _motivo_ao_lado(resultado.baixar) == ""


def test_resultado_da_correcao_baixa_a_correcao_feita_e_nao_o_combo(
    qtbot: Any, tmp_path: Path
) -> None:
    """Corrigido com `loess` e o combo depois em `serrf`: o download pede `loess`, o painel guarda a
    identidade de `loess`, e o arquivo gravado tem `loess` no nome.

    **O combo diferente é a planta, e ela cai sobre dado em que as duas leituras divergem**: um
    download que lesse o combo no clique pediria `serrf` nos três lugares.
    """
    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    _corrige(janela, "loess")
    janela.pipeline.corretor.setCurrentText("serrf")
    janela._atualiza_as_etapas()
    pedidos.clear()
    resultado = janela.resultado_da_correcao

    resultado.baixar.click()

    campos = _o_pedido(pedidos, "/matrizes/corrigir")
    assert campos["method"] == "loess", campos
    assert "auto" not in campos, "o download delegou de novo, e o Palpite poderia escolher outro"
    assert resultado._identidade is not None and "|loess|" in resultado._identidade
    janela._encaminha("corrigida", _resposta_de_artefato(b"csv da loess"))
    escrito = tmp_path / nome_do_arquivo("neg_c18", "corrigida", "loess")
    assert escrito.read_bytes() == b"csv da loess"
    assert not [p.name for p in tmp_path.glob("*serrf*")], "o nome levou o corretor do combo"


def test_resultado_da_correcao_os_dois_formatos_pedem_as_duas_rotas(
    qtbot: Any, tmp_path: Path
) -> None:
    """O do laboratório pede `/matrizes/corrigir`; o MetaboAnalyst, `/matrizes/exportar` — com o
    teto de features e sem eixo do pré-processamento nenhum, que é o arquivo de sempre."""
    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    _corrige(janela, "loess")
    resultado = janela.resultado_da_correcao
    pedidos.clear()
    resultado.baixar.click()
    resultado.formatos.botoes[FORMATO_METABOANALYST].click()
    resultado.baixar.click()

    assert [p["rota"] for p in pedidos] == ["/matrizes/corrigir", "/matrizes/exportar"]
    metabo = _o_pedido(pedidos, "/matrizes/exportar")
    assert metabo["method"] == "loess" and metabo["max_features"] == 5000, metabo
    eixos = {eixo for eixo, *_ in EIXOS_DA_GRADE} | {"rsd_cutoff", "missing_threshold"}
    assert not eixos & set(metabo), f"o arquivo da Correção levou eixos: {eixos & set(metabo)}"


def test_resultado_toda_chave_nova_esta_nas_duas_travas_e_o_download_trava_a_correcao(
    qtbot: Any, tmp_path: Path
) -> None:
    """Toda chave do Resultado está entre as que corrigem e entre as com guarda de resposta
    superada; e, com o download em espera, a etapa da Correção não começa."""
    from refino.desktop.janela.principal import _CHAVES_COM_SUPERSESSAO, CHAVES_QUE_CORRIGEM

    das_etapas = [chave for chave, a in ARTEFATOS_DA_JANELA.items() if a.etapa is not None]
    assert len(das_etapas) == 5, f"âncora: as chaves do Resultado são {das_etapas}"
    for chave in das_etapas:
        assert chave in CHAVES_QUE_CORRIGEM, chave
        assert chave in _CHAVES_COM_SUPERSESSAO, chave

    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    _corrige(janela, "none")
    janela.resultado_da_correcao.baixar.click()
    aceite = {"identificador": "t-1", "total": 2, "reclamado": False}
    janela._encaminha("corrigida", Resposta(202, True, None, "", aceite))
    assert "corrigida" in janela.esperas_em_curso(), "âncora: o download não ficou à espera"
    pedidos.clear()
    janela._dispara_etapa(1)
    assert pedidos == [], "a Correção disparou com o download a corrigir no servidor"
    assert janela.rotulo_de_desfecho.text() == CORRECAO_EM_CURSO
    # A resposta de uma espera que já não é a corrente é descartada: nada escrito na pasta.
    janela._encaminha("corrigida", _resposta_de_artefato(b"velho"), "t-0")
    assert not list(tmp_path.glob("*.csv")), "a resposta superada foi escrita"


def test_resultado_o_resumo_do_metaboanalyst_com_loess_nomeia_o_denominador_nulo(
    qtbot: Any, tmp_path: Path
) -> None:
    """O arquivo MetaboAnalyst sai com `uncorrected_feature`, e o do laboratório com `r_faithful`:
    a política que cada rota usa vai ao resumo do arquivo, ao lado do download."""
    janela, _ = _janela_dos_resultados(qtbot, tmp_path)
    _corrige(janela, "loess")
    resultado = janela.resultado_da_correcao
    resultado.formatos.botoes[FORMATO_METABOANALYST].click()
    resultado.baixar.click()
    janela._encaminha("corrigida_metaboanalyst", _resposta_de_artefato(b"csv"))
    texto = resultado.resumo.caixa.toPlainText()
    assert "`uncorrected_feature`" in texto, texto
    assert "matriz da etapa de correção, no formato do MetaboAnalyst" in texto

    resultado.formatos.botoes["laboratorio"].click()
    resultado.baixar.click()
    janela._encaminha("corrigida", _resposta_de_artefato(b"csv"))
    texto = resultado.resumo.caixa.toPlainText()
    assert "`r_faithful`" in texto and "`uncorrected_feature`" not in texto, texto


def test_resultado_a_caixa_do_resumo_grava_o_txt_ao_lado_e_desmarcada_nao(
    qtbot: Any, tmp_path: Path
) -> None:
    """Marcada por padrão: o CSV e `<nome>-metodos.txt`, com o texto do bloco. Desmarcada: só o
    CSV. Copiar e salvar o resumo dizem por que esperam, e ligam com o primeiro arquivo."""
    from refino.desktop.janela.execucao import MOTIVO_SEM_RESUMO

    janela, _ = _janela_dos_resultados(qtbot, tmp_path)
    _corrige(janela, "none")
    resultado = janela.resultado_da_correcao
    assert resultado.salvar_o_resumo.isChecked(), "a caixa nasce marcada"
    assert not resultado.resumo.copiar.isEnabled()
    assert _motivo_ao_lado(resultado.resumo.copiar) == MOTIVO_SEM_RESUMO
    resultado.baixar.click()
    janela._encaminha("corrigida", _resposta_de_artefato(b"csv"))
    csv_ = tmp_path / nome_do_arquivo("neg_c18", "corrigida", "none")
    txt = csv_.with_name(f"{csv_.stem}-metodos.txt")
    assert csv_.read_bytes() == b"csv"
    assert txt.read_text(encoding="utf-8") == resultado.resumo.caixa.toPlainText()
    assert resultado.resumo.copiar.isEnabled() and _motivo_ao_lado(resultado.resumo.copiar) == ""
    assert txt.name in resultado.estado.text()

    for escrito in tmp_path.iterdir():
        escrito.unlink()
    resultado.salvar_o_resumo.setChecked(False)
    resultado.baixar.click()
    janela._encaminha("corrigida", _resposta_de_artefato(b"csv"))
    assert sorted(p.name for p in tmp_path.iterdir()) == [csv_.name]


def test_resultado_o_lote_no_metaboanalyst_grava_tres_arquivos_e_um_resumo_dos_tres(
    qtbot: Any, tmp_path: Path
) -> None:
    """«Gerar os três» no formato escolhido: três CSV, um por corretor, e UM resumo que cita os
    três — o lote é um gesto só."""
    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    resultado = janela.resultado_da_correcao
    resultado.formatos.botoes[FORMATO_METABOANALYST].click()
    botao = resultado.gerar_os_tres_botao
    assert botao is not None
    botao.click()
    for metodo in CORRETORES:
        assert pedidos and pedidos[-1]["rota"] == "/matrizes/exportar", pedidos
        assert pedidos[-1]["campos"]["method"] == metodo
        janela._encaminha(CHAVE_DO_LOTE_METABO, _resposta_de_artefato(f"csv-{metodo}".encode()))
    assert janela._lote is None, "o lote não encerrou"

    nomes = sorted(nome_do_arquivo("neg_c18", CHAVE_DO_LOTE_METABO, m) for m in CORRETORES)
    assert sorted(p.name for p in tmp_path.glob("*.csv")) == nomes
    resumos = list(tmp_path.glob("*-metodos.txt"))
    assert [p.name for p in resumos] == [
        f"{Path(nome_do_arquivo('neg_c18', CHAVE_DO_LOTE_METABO)).stem}-metodos.txt"
    ]
    texto = resumos[0].read_text(encoding="utf-8")
    assert all(nome in texto for nome in nomes), texto


# ---------------------------------------------------------------------------
# O lote grava o registro do corte de CADA perna, e não só o da última
# ---------------------------------------------------------------------------


def _relatorio_da_perna(metodo: str, quantas: int) -> dict[str, Any]:
    """Um corpo de relatório que só esta perna tem: o nome de cada feature leva o corretor, e o
    número de removidas difere entre as pernas. É o que deixa a planta separar o irmão certo de um
    irmão escrito com o relatório de outra perna."""
    return {
        "mantidas": 5000,
        "removidas": [
            {"nome": f"{i + 1}.00_{metodo}_{i}m/z", "causa": "rsd_acima_do_corte", "rsd": 1.5 + i}
            for i in range(quantas)
        ],
        "zeros_como_vazias": 0,
        "nomes_sanitizados": 5000,
    }


def _corre_o_lote_na_ordem_real(
    janela: JanelaPrincipal,
    pedidos: list[dict[str, Any]],
    relatorios: dict[str, Resposta],
    *,
    prefixo: str = "trabalho",
    entrega_o_ultimo: bool = True,
) -> dict[str, str]:
    """«Gerar os três» no MetaboAnalyst, com as respostas na ordem que a piscina produz.

    Cada perna recebe o aceite do `POST`, a leitura de progresso que traz os bytes e, só DEPOIS de
    a perna seguinte ter começado, o relatório do corte dela; o da última chega com o lote já
    encerrado. É a ordem do passeio de 10/11/2026: o `GET` do relatório é assíncrono, e a perna
    seguinte redefine a entrada do painel antes de ele voltar.

    **As chaves das respostas são as que a janela gravou nos pedidos dela**, e não compostas aqui:
    é `_recebe` quem as separa, como na produção. Devolve o rótulo do Resultado lido logo depois de
    cada relatório chegar, por corretor.

    `prefixo` dá a cada lote os seus identificadores de trabalho, para dois lotes na mesma janela
    não partilharem pedidos; `entrega_o_ultimo=False` deixa o relatório da última perna por voltar.
    """
    resultado = janela.resultado_da_correcao
    resultado.formatos.botoes[FORMATO_METABOANALYST].click()
    botao = resultado.gerar_os_tres_botao
    assert botao is not None
    botao.click()
    rotulos: dict[str, str] = {}
    pendente: tuple[str, str] | None = None
    for metodo in CORRETORES:
        assert pedidos and pedidos[-1]["rota"] == "/matrizes/exportar", pedidos
        assert pedidos[-1]["campos"]["method"] == metodo, pedidos[-1]
        if pendente is not None:
            chave_do_relatorio, anterior = pendente
            # **A âncora da ordem**: a perna nova já redefiniu a entrada do painel, e os bytes da
            # anterior saíram. Sem isto o roteiro não reproduz o que o passeio mediu.
            assert resultado.salvaveis() == (), (
                f"âncora: a perna de {metodo} não redefiniu a entrada: {resultado.salvaveis()}"
            )
            janela._recebe(chave_do_relatorio, relatorios[anterior])
            rotulos[anterior] = resultado.estado.text()
        identificador = f"{prefixo}-{metodo}"
        aceite = {"identificador": identificador, "total": 2, "reclamado": False}
        janela._recebe(CHAVE_DO_LOTE_METABO, Resposta(202, True, None, "", aceite))
        leitura = pedidos[-1]
        assert leitura["rota"] == rota_do_progresso(identificador), leitura
        janela._recebe(leitura["chave"], _resposta_de_artefato(f"csv-{metodo}".encode()))
        do_relatorio = [p for p in pedidos if p["rota"] == rota_do_relatorio(identificador)]
        assert len(do_relatorio) == 1, f"a perna de {metodo} não pediu o relatório do corte"
        pendente = (do_relatorio[0]["chave"], metodo)
    assert janela._lote is None, "o lote não encerrou"
    assert pendente is not None
    if not entrega_o_ultimo:
        return rotulos
    chave_do_relatorio, ultimo = pendente
    janela._recebe(chave_do_relatorio, relatorios[ultimo])
    rotulos[ultimo] = resultado.estado.text()
    return rotulos


def _irmao_da_perna(pasta: Path, metodo: str) -> Path:
    return nome_do_arquivo_irmao(pasta / nome_do_arquivo("neg_c18", CHAVE_DO_LOTE_METABO, metodo))


def _features_do_irmao(irmao: Path) -> list[str]:
    linhas = list(csv.reader(irmao.read_text(encoding="utf-8").splitlines()))
    assert linhas and linhas[0] == ["feature", "causa", "rsd_fracao"], linhas[:1]
    return [linha[0] for linha in linhas[1:]]


def test_d709_o_lote_grava_o_registro_do_corte_de_cada_perna_na_ordem_real(
    qtbot: Any, tmp_path: Path
) -> None:
    """Três CSV e três `-removidas.csv`, cada irmão com o relatório da SUA perna.

    Medido no passeio de 10/11/2026: a pasta ficava só com `…_loess-removidas.csv`. A perna seguinte
    redefine a entrada do painel, e isso apagava os bytes da anterior antes de o relatório dela
    chegar; a guarda de bytes do painel recusava-o, e o irmão não era escrito.

    **Os relatórios diferem por perna**, então um irmão escrito com o relatório de outra perna
    reprova, e não só um irmão ausente. **E o rótulo continua a falar só da perna à vista**: o
    relatório de uma perna anterior não chega a ele.
    """
    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    corpos = {m: _relatorio_da_perna(m, i + 1) for i, m in enumerate(CORRETORES)}
    assert len({json.dumps(c, sort_keys=True) for c in corpos.values()}) == len(CORRETORES), (
        "âncora: os relatórios das pernas são iguais, e a planta não separaria um irmão trocado"
    )

    rotulos = _corre_o_lote_na_ordem_real(
        janela, pedidos, {m: _resposta_de_relatorio(c) for m, c in corpos.items()}
    )

    nomes = sorted(nome_do_arquivo("neg_c18", CHAVE_DO_LOTE_METABO, m) for m in CORRETORES)
    assert sorted(p.name for p in tmp_path.glob("*.csv") if "-removidas" not in p.name) == nomes
    irmaos = sorted(p.name for p in tmp_path.glob("*-removidas.csv"))
    assert irmaos == sorted(_irmao_da_perna(tmp_path, m).name for m in CORRETORES), (
        f"o lote não deixou um registro do corte por corretor: {irmaos}"
    )
    for metodo in CORRETORES:
        esperado = [item["nome"] for item in corpos[metodo]["removidas"]]
        assert _features_do_irmao(_irmao_da_perna(tmp_path, metodo)) == esperado, (
            f"o registro do corte de {metodo} não tem o relatório da perna dele"
        )
    # A guarda do painel continua a governar o rótulo: só a última perna está à vista.
    *anteriores, ultimo = CORRETORES
    for metodo in anteriores:
        assert f"{len(corpos[metodo]['removidas'])} removidas" not in rotulos[metodo], (
            f"o relatório de {metodo} chegou ao rótulo, que descreve outra perna: {rotulos[metodo]}"
        )
    assert f"{len(corpos[ultimo]['removidas'])} removidas" in rotulos[ultimo], rotulos[ultimo]


def test_d709_relatorio_de_perna_sem_200_nao_escreve_irmao_e_o_lote_segue(
    qtbot: Any, tmp_path: Path
) -> None:
    """O relatório da primeira perna volta `409` (o trabalho foi podado): nenhum irmão para ela, e
    o lote segue — as três pernas gravam o CSV, e as outras duas o registro do corte.

    **O par causal está no mesmo roteiro**: a mesma ordem, com `200` nas outras pernas, escreve os
    irmãos delas, então a ausência do da primeira é do código e não de um roteiro que não escreve.
    """
    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    primeiro, *outros = CORRETORES
    relatorios = {m: _resposta_de_relatorio(_relatorio_da_perna(m, 2)) for m in outros}
    relatorios[primeiro] = Resposta(
        codigo=409, ok=False, causa=None, mensagem="Conflito", corpo={"detail": "podado"}
    )

    _corre_o_lote_na_ordem_real(janela, pedidos, relatorios)

    nomes = sorted(nome_do_arquivo("neg_c18", CHAVE_DO_LOTE_METABO, m) for m in CORRETORES)
    assert sorted(p.name for p in tmp_path.glob("*.csv") if "-removidas" not in p.name) == nomes, (
        "o relatório sem 200 abortou o lote"
    )
    assert sorted(p.name for p in tmp_path.glob("*-removidas.csv")) == sorted(
        _irmao_da_perna(tmp_path, m).name for m in outros
    )


@pytest.mark.parametrize("que_falha", CORRETORES)
def test_d709_falha_ao_gravar_o_irmao_de_uma_perna_fica_a_vista_no_fim_do_lote(
    que_falha: str, qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O escritor do registro falha numa perna: a frase vai ao rótulo do Resultado na hora, como a
    do clique avulso, o lote não aborta — o CSV daquela perna já está gravado — e **a frase que
    fica no fim do lote nomeia o CSV que ficou sem o registro**, no Resultado e no aviso.

    **O fim é o que a revisão de código de 11/11/2026 pediu**: a leitura de progresso da perna
    seguinte reescrevia o rótulo, a frase de fim contava só os CSV, e o relatório da última perna
    trocava essa frase pelo corte dela — a falha de uma perna do meio sumia antes de alguém a ler.
    **Par causal feito à mão uma vez em 11/11/2026:** com `principal.py` de `4a8abd11`, as três
    variantes reprovam na asserção da frase de fim; as duas primeiras porque o rótulo termina com o
    corte da última perna, a terceira porque termina com a frase crua da falha.
    """
    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    escreve = write_removidas
    alvo_da_falha = _irmao_da_perna(tmp_path, que_falha)

    def falha_numa_perna(removidas: Any, destino: Path) -> None:
        if destino == alvo_da_falha:
            raise OSError("disco cheio")
        escreve(removidas, destino)

    monkeypatch.setattr("refino.desktop.janela.artefatos.write_removidas", falha_numa_perna)

    rotulos = _corre_o_lote_na_ordem_real(
        janela,
        pedidos,
        {m: _resposta_de_relatorio(_relatorio_da_perna(m, 2)) for m in CORRETORES},
    )

    csv_da_falha = nome_do_arquivo("neg_c18", CHAVE_DO_LOTE_METABO, que_falha)
    assert "registro do corte NÃO" in rotulos[que_falha], rotulos[que_falha]
    assert (tmp_path / csv_da_falha).is_file(), (
        "o CSV da perna cujo registro falhou não ficou na pasta"
    )
    assert sorted(p.name for p in tmp_path.glob("*-removidas.csv")) == sorted(
        _irmao_da_perna(tmp_path, m).name for m in CORRETORES if m != que_falha
    ), "a falha de uma perna parou o registro das outras"
    esperado = f"O registro do corte NÃO foi salvo ao lado de: {csv_da_falha}."
    final = janela.resultado_da_correcao.estado.text()
    assert final.startswith("Lote concluído"), f"o fim do lote não ficou no Resultado: {final}"
    assert esperado in final, f"o fim do lote não nomeia o CSV sem o registro: {final}"
    assert esperado in janela.rotulo_de_desfecho.text(), janela.rotulo_de_desfecho.text()
    assert janela.aviso_de_desfecho.property("tipo") == "erro"
    assert janela._lote_encerrado is None, "o lote encerrado não foi solto"


def test_d709_relatorio_que_nunca_volta_nao_prende_o_fim_do_lote_seguinte(
    qtbot: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O primeiro lote encerra com o relatório da última perna por voltar, e ele nunca volta. O
    segundo lote, noutra pasta, falha o registro da perna do meio: **a frase de fim dele diz isso**.

    Os pendentes são os de cada lote e não o registro global das pernas, onde o relatório perdido
    do primeiro continua. **Par causal feito à mão uma vez em 11/11/2026:** com o fecho a conferir
    o registro global (`self._pernas_por_pedido`), a asserção da frase de fim reprova — o rótulo
    termina com o corte da última perna do segundo lote.
    """
    primeira, segunda = tmp_path / "primeira", tmp_path / "segunda"
    primeira.mkdir()
    segunda.mkdir()
    pastas = iter((primeira, segunda))
    janela, pedidos = _janela_dos_resultados(qtbot, None)
    janela._escolhe_pasta = lambda: next(pastas)
    corpos = {m: _resposta_de_relatorio(_relatorio_da_perna(m, 2)) for m in CORRETORES}
    _corre_o_lote_na_ordem_real(janela, pedidos, corpos, prefixo="a", entrega_o_ultimo=False)
    assert janela._pernas_por_pedido, "âncora: o relatório perdido não ficou no registro global"

    meio = CORRETORES[1]
    alvo_da_falha = _irmao_da_perna(segunda, meio)
    escreve = write_removidas

    def falha_no_meio(removidas: Any, destino: Path) -> None:
        if destino == alvo_da_falha:
            raise OSError("disco cheio")
        escreve(removidas, destino)

    monkeypatch.setattr("refino.desktop.janela.artefatos.write_removidas", falha_no_meio)
    pedidos.clear()
    _corre_o_lote_na_ordem_real(janela, pedidos, corpos, prefixo="b")

    csv_da_falha = nome_do_arquivo("neg_c18", CHAVE_DO_LOTE_METABO, meio)
    final = janela.resultado_da_correcao.estado.text()
    assert f"O registro do corte NÃO foi salvo ao lado de: {csv_da_falha}." in final, final
    assert janela._lote_encerrado is None, "o segundo lote não foi solto"


def test_d709_lote_sem_falha_nao_ganha_a_oracao_do_registro_em_falta(
    qtbot: Any, tmp_path: Path
) -> None:
    """O lote que gravou os três irmãos não diz nada sobre registro em falta, e o rótulo fica com o
    que o relatório da última perna escreveu — o comportamento de antes da revisão."""
    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    _corre_o_lote_na_ordem_real(
        janela,
        pedidos,
        {m: _resposta_de_relatorio(_relatorio_da_perna(m, 2)) for m in CORRETORES},
    )
    assert len(list(tmp_path.glob("*-removidas.csv"))) == len(CORRETORES), "âncora: os irmãos"
    assert "NÃO foi salvo" not in janela.resultado_da_correcao.estado.text()
    assert "NÃO foi salvo" not in janela.rotulo_de_desfecho.text()
    assert janela._lote_encerrado is None, "o lote encerrado não foi solto"


def test_d709_relatorio_da_perna_abortada_nao_reescreve_o_irmao_de_um_clique_avulso(
    qtbot: Any, tmp_path: Path
) -> None:
    """Um clique avulso grava `X_loess.csv` e o irmão dele; depois, «Gerar os três» aborta na
    perna `serrf`, cujo CSV não se escreve. **O relatório dela já tinha sido pedido**, e quando
    volta não pode descer ao caminho do clique avulso: ali `_escritos` ainda aponta para o CSV do
    clique, e o irmão dele seria reescrito com o corte do `serrf` (revisão de código de
    11/11/2026, achado confirmado de perda de dado).

    **Par causal feito à mão uma vez em 11/11/2026:** com `principal.py` de `4a8abd11`, a asserção
    dos bytes do irmão do clique reprova — ele passa a listar as features do `serrf`.
    """
    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    _corrige(janela, "loess")
    resultado = janela.resultado_da_correcao
    resultado.formatos.botoes[FORMATO_METABOANALYST].click()
    chave = resultado.chave_do_formato()
    assert chave == CHAVE_DO_LOTE_METABO, "âncora: o clique e o lote não são da mesma chave"
    pedidos.clear()
    resultado.baixar.click()
    janela._recebe(chave, Resposta(202, True, None, "", {"identificador": "avulso", "total": 2}))
    janela._recebe(pedidos[-1]["chave"], _resposta_de_artefato(b"csv avulso"))
    (do_avulso,) = [p for p in pedidos if p["rota"] == rota_do_relatorio("avulso")]
    janela._recebe(do_avulso["chave"], _resposta_de_relatorio(_relatorio_da_perna("avulso", 2)))
    irmao_do_clique = nome_do_arquivo_irmao(tmp_path / nome_do_arquivo("neg_c18", chave, "loess"))
    assert irmao_do_clique.is_file(), "âncora: o clique avulso não gravou o irmão dele"
    bytes_do_clique = irmao_do_clique.read_bytes()

    primeiro, abortada, _ = CORRETORES
    (tmp_path / nome_do_arquivo("neg_c18", CHAVE_DO_LOTE_METABO, abortada)).mkdir()
    botao = resultado.gerar_os_tres_botao
    assert botao is not None
    botao.click()
    relatorios: dict[str, str] = {}
    for metodo in (primeiro, abortada):
        assert pedidos[-1]["campos"]["method"] == metodo, pedidos[-1]
        aceite = {"identificador": f"trabalho-{metodo}", "total": 2, "reclamado": False}
        janela._recebe(CHAVE_DO_LOTE_METABO, Resposta(202, True, None, "", aceite))
        janela._recebe(pedidos[-1]["chave"], _resposta_de_artefato(f"csv-{metodo}".encode()))
        (do_relatorio,) = [
            p for p in pedidos if p["rota"] == rota_do_relatorio(f"trabalho-{metodo}")
        ]
        relatorios[metodo] = do_relatorio["chave"]
    assert janela._lote is None, "âncora: o lote não abortou na perna sem escrita"
    assert resultado._falha is not None and "ABORTADO" in resultado._falha[0], resultado._falha

    removidas = {primeiro: 2, abortada: 3}
    for metodo in (primeiro, abortada):
        corpo = _relatorio_da_perna(metodo, removidas[metodo])
        janela._recebe(relatorios[metodo], _resposta_de_relatorio(corpo))

    assert irmao_do_clique.read_bytes() == bytes_do_clique, (
        "o relatório da perna abortada reescreveu o registro do corte do clique avulso"
    )
    assert _irmao_da_perna(tmp_path, primeiro).is_file(), "a perna que escreveu ficou sem irmão"
    assert not _irmao_da_perna(tmp_path, abortada).exists(), "a perna abortada ganhou irmão"
    assert "3 removidas" not in resultado.estado.text(), (
        f"o relatório da perna abortada chegou ao rótulo: {resultado.estado.text()}"
    )
    assert resultado._falha is not None and "ABORTADO" in resultado._falha[0], resultado._falha
    assert janela._relatorios_descartados == set(), "o relatório descartado ficou registrado"


def test_d709_o_clique_avulso_continua_a_gravar_o_seu_irmao(qtbot: Any, tmp_path: Path) -> None:
    """Fora do lote, o relatório chega com o pedido do clique, que não é de perna nenhuma, e o
    irmão sai pelo caminho do clique avulso, ao lado do CSV que a escrita automática guardou.

    **Existe porque o teste do caminho avulso na janela depende do dado de referência** e faz
    `skip` num clone sem ele; este usa bytes escritos à mão e corre em qualquer clone. As chaves são
    as que a janela gravou, como na produção: o relatório traz pedido, e é isso que o põe à prova
    contra o registro das pernas.
    """
    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    _corrige(janela, "loess")
    resultado = janela.resultado_da_correcao
    resultado.formatos.botoes[FORMATO_METABOANALYST].click()
    chave = resultado.chave_do_formato()
    pedidos.clear()
    resultado.baixar.click()
    assert [p["rota"] for p in pedidos] == ["/matrizes/exportar"], pedidos
    identificador = "trabalho-avulso"
    aceite = {"identificador": identificador, "total": 2, "reclamado": False}
    janela._recebe(chave, Resposta(202, True, None, "", aceite))
    janela._recebe(pedidos[-1]["chave"], _resposta_de_artefato(b"csv avulso"))
    do_relatorio = [p for p in pedidos if p["rota"] == rota_do_relatorio(identificador)]
    assert len(do_relatorio) == 1, f"o clique não pediu o relatório do corte: {pedidos}"
    corpo = _relatorio_da_perna("avulso", 2)

    janela._recebe(do_relatorio[0]["chave"], _resposta_de_relatorio(corpo))

    csv_ = tmp_path / nome_do_arquivo("neg_c18", chave, "loess")
    assert csv_.read_bytes() == b"csv avulso", "âncora: o CSV do clique não foi escrito"
    irmao = nome_do_arquivo_irmao(csv_)
    assert irmao.is_file(), "o clique avulso deixou de gravar o registro do corte"
    assert _features_do_irmao(irmao) == [item["nome"] for item in corpo["removidas"]], (
        "o registro do clique avulso não tem o relatório dele"
    )


def test_resultado_falha_ao_gravar_o_txt_vira_frase_e_o_csv_fica(
    qtbot: Any, tmp_path: Path
) -> None:
    """Um diretório no lugar do `.txt` faz a gravação dele falhar de verdade: a frase diz que o
    resumo não foi salvo, e o CSV já escrito continua na pasta."""
    janela, _ = _janela_dos_resultados(qtbot, tmp_path)
    _corrige(janela, "none")
    csv_ = tmp_path / nome_do_arquivo("neg_c18", "corrigida", "none")
    (tmp_path / f"{csv_.stem}-metodos.txt").mkdir()
    resultado = janela.resultado_da_correcao
    resultado.baixar.click()
    janela._encaminha("corrigida", _resposta_de_artefato(b"csv"))
    assert csv_.read_bytes() == b"csv"
    frase = resultado.estado.text()
    assert "NÃO foi salvo" in frase and str(csv_) in frase, frase


def test_resultado_da_normalizacao_com_escalonamento_avisa_antes_do_clique(
    qtbot: Any, tmp_path: Path
) -> None:
    """Escalonamento `auto` e o formato MetaboAnalyst: faixa amarela com a frase da recusa e o
    botão da alternativa, e o download principal desligado com o motivo. **No formato do
    laboratório nenhuma faixa** — é a âncora de que a faixa é do formato, e não do escalonamento."""
    from refino.desktop.janela.execucao import (
        FRASE_DA_RECUSA_SABIDA,
        MOTIVO_DO_ESCALONAMENTO,
        TEXTO_DA_ALTERNATIVA,
    )

    janela, _ = _janela_dos_resultados(qtbot, tmp_path)
    _normaliza(janela, scaling="auto")
    resultado = janela.resultado_da_normalizacao
    assert resultado.baixar.isEnabled() and resultado.faixa.isHidden()

    resultado.formatos.botoes[FORMATO_METABOANALYST].click()

    assert not resultado.baixar.isEnabled()
    assert _motivo_ao_lado(resultado.baixar) == MOTIVO_DO_ESCALONAMENTO
    assert not resultado.faixa.isHidden() and resultado.faixa.property("nivel") == "aviso"
    assert FRASE_DA_RECUSA_SABIDA in resultado.faixa.texto
    assert resultado.faixa.botao.text() == TEXTO_DA_ALTERNATIVA


def test_resultado_da_normalizacao_a_alternativa_pede_sem_transformacao_nem_escalonamento(
    qtbot: Any, tmp_path: Path
) -> None:
    """O botão da faixa pede `/matrizes/exportar` com a transformação e o escalonamento em
    `none` e o resto da Normalização feita; o arquivo e o resumo dizem que é a alternativa."""
    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    _normaliza(janela, sample_normalization="sum", transformation="log10", scaling="auto")
    resultado = janela.resultado_da_normalizacao
    resultado.formatos.botoes[FORMATO_METABOANALYST].click()
    pedidos.clear()

    resultado.faixa.botao.click()

    campos = _o_pedido(pedidos, "/matrizes/exportar")
    assert (campos["transformation"], campos["scaling"]) == ("none", "none"), campos
    assert (campos["sample_normalization"], campos["imputation"]) == ("sum", "one_fifth_min")
    chave = CHAVE_DA_ALTERNATIVA[2]
    janela._encaminha(chave, _resposta_de_artefato(b"csv"))
    escrito = tmp_path / nome_do_arquivo("neg_c18", chave, "none")
    assert escrito.exists() and "antes_da_transformacao_e_do_escalonamento" in escrito.name
    texto = resultado.resumo.caixa.toPlainText()
    assert (
        "ALTERNATIVA à recusa" in texto and "anterior à transformação e ao escalonamento" in texto
    )


def test_resultado_da_normalizacao_log_com_422_mostra_a_faixa_vermelha_com_a_alternativa(
    qtbot: Any, tmp_path: Path
) -> None:
    """Com `log10` a recusa só se sabe pelo `422`: antes do clique, nenhuma faixa; depois, a
    vermelha com a frase do servidor e o botão da alternativa. **Par causal:** o `422` do formato do
    laboratório não oferece a alternativa, que lá não muda nada."""
    from refino.desktop.janela.execucao import TEXTO_DA_ALTERNATIVA
    from refino.io.writers import _frase_da_recusa_por_negativo

    janela, _ = _janela_dos_resultados(qtbot, tmp_path)
    _normaliza(janela, sample_normalization="sum", transformation="log10")
    resultado = janela.resultado_da_normalizacao
    resultado.formatos.botoes[FORMATO_METABOANALYST].click()
    assert resultado.faixa.isHidden() and resultado.baixar.isEnabled(), "o log não é sabido antes"
    resultado.baixar.click()
    corpo = {"detail": _frase_da_recusa_por_negativo(12, ["transform(log10)"])}
    recusa = Resposta(
        422, False, cliente.causa_de(422, corpo), cliente.mensagem_de(422, corpo), corpo
    )

    janela._encaminha("normalizada_metaboanalyst", recusa)

    assert not resultado.faixa.isHidden() and resultado.faixa.property("nivel") == "critico"
    assert corpo["detail"] in resultado.faixa.texto
    assert resultado.faixa.botao.text() == TEXTO_DA_ALTERNATIVA
    assert not resultado.faixa.botao.isHidden()

    resultado.formatos.botoes["laboratorio"].click()
    assert resultado.faixa.isHidden(), "trocar o formato não apagou a falha do outro"
    resultado.baixar.click()
    janela._encaminha("normalizada", recusa)
    assert resultado.faixa.property("nivel") == "critico"
    assert resultado.faixa.botao.isHidden(), (
        "o laboratório ofereceu uma alternativa que não muda nada"
    )


def test_resultado_da_normalizacao_sobre_a_original_baixa_sem_correcao(
    qtbot: Any, tmp_path: Path
) -> None:
    """Sem correção feita, a Normalização parte da original: o download pede `method=none`, o
    resumo diz «sem correção», e o nome não leva o corretor que está no combo."""
    janela, pedidos = _janela_dos_resultados(qtbot, tmp_path)
    janela.pipeline.corretor.setCurrentText("serrf")
    _normaliza(janela)
    feita = janela._etapas.feita(2)
    assert feita is not None and feita.entrada == "original", "âncora: a entrada é a original"
    resultado = janela.resultado_da_normalizacao
    pedidos.clear()

    resultado.baixar.click()

    assert _o_pedido(pedidos, "/matrizes/processar")["method"] == "none"
    janela._encaminha("normalizada", _resposta_de_artefato(b"csv"))
    assert (tmp_path / nome_do_arquivo("neg_c18", "normalizada", "none")).exists()
    assert not [p.name for p in tmp_path.iterdir() if "serrf" in p.name]
    assert "sobre a matriz sem correção" in resultado.resumo.caixa.toPlainText()


# ---------------------------------------------------------------------------
# Os gráficos da etapa na aba da etapa — e cada etapa dona das próprias figuras
# ---------------------------------------------------------------------------


def test_a_vista_de_cada_etapa_esta_na_aba_dela_e_leva_a_figura_aos_graficos(
    qtbot: Any,
) -> None:
    """Pelo caminho real: cada vista vive no lugar dos gráficos da aba da etapa, e não na aba
    «Gráficos»; depois da etapa 1, a vista da Correção — com a aba à frente — mostra o painel da
    correção, e «Exportar esta figura em Gráficos» troca de aba com a mesma figura escolhida e
    desenhada lá."""
    from refino.desktop.janela.principal import INDICE_DA_ABA_DOS_GRAFICOS

    janela, _transporte = _janela_do_passo_a_passo(qtbot)
    # O par entrou por fora do seletor, e as abas de etapa continuam no estado vazio: o formulário
    # vem à frente como viria com o par escolhido.
    for pagina in janela._paginas_de_etapa.values():
        pagina.mostra_arquivos(())
    vistas = janela.graficos.vistas_das_etapas
    assert sorted(vistas) == [1, 2, 3], "âncora: as três vistas"
    for etapa, vista in vistas.items():
        pagina = janela._paginas_de_etapa[etapa]
        assert pagina.lugar_dos_graficos.isAncestorOf(vista), f"a vista {etapa} fora da aba dela"
        assert not janela.graficos.isAncestorOf(vista), f"a vista {etapa} ficou em «Gráficos»"
        assert vista.isVisibleTo(pagina), f"a vista {etapa} está escondida na aba"

    janela.resize(1280, 720)
    janela.show()
    janela.abas.setCurrentWidget(janela._paginas_de_etapa[1])
    janela.pipeline.botoes_de_etapa[1].click()
    assert _gira_ate(janela, lambda: janela._atualiza_as_etapas()[0] == "em_dia", 120), (
        janela.rotulo_de_desfecho.text()
    )
    vista = vistas[1]
    funcao = vista.figuras.funcao_escolhida()
    assert funcao == "figura_do_painel_da_correcao", funcao
    assert _gira_ate(janela, lambda: vista.figuras.estado_visivel()["figura"] != "", 10), (
        vista.figuras.repouso.text()
    )
    assert vista.botao_de_exportar.isEnabled(), "com a figura na tela o botão ficou desligado"

    vista.botao_de_exportar.click()
    assert janela.abas.currentIndex() == INDICE_DA_ABA_DOS_GRAFICOS
    assert janela.graficos.figuras.funcao_escolhida() == funcao
    assert janela.graficos.figuras.estado_visivel()["figura"] != "", "«Gráficos» sem a figura"


def _corpo_da_etapa_com(corretor: str) -> dict[str, Any]:
    """Um corpo de prévia em que só o nome do corretor do par importa: nada aqui é desenhado."""
    return {"correcao": corretor, "pcas_da_correcao": {"corretor": corretor}}


def _resposta(codigo: int, corpo: Any) -> Resposta:
    return Resposta(codigo=codigo, ok=codigo in (200, 202), causa=None, mensagem="", corpo=corpo)


@pytest.mark.parametrize(("entrada", "corretor"), [("original", "loess"), ("corrigida", "none")])
def test_a_janela_passa_a_entrada_da_etapa_feita_ao_painel_dos_graficos(
    qtbot: Any, transporte_real: TestClient, entrada: str, corretor: str
) -> None:
    """A Normalização escreve as figuras da Correção só quando a entrada dela é a Correção guardada,
    e é a janela que lê a entrada do estado das etapas. **Par causal:** a mesma sequência — uma
    Correção com `loess` e uma Normalização com `none` —, com a etapa 2 feita sobre a `original` e
    sobre a `corrigida`; o par guardado é o da etapa 1 no primeiro caso, e o da 2 no segundo."""
    from refino.desktop.etapas import CHAVE_DA_ETAPA

    janela = JanelaPrincipal(transporte_real)
    qtbot.addWidget(janela)
    janela._etapas.disparou(1, ())
    janela._encaminha(CHAVE_DA_ETAPA[1], _resposta(200, _corpo_da_etapa_com("loess")))
    assert janela.graficos._pcas_da_correcao == {"corretor": "loess"}, "âncora: a etapa 1"
    janela._etapas.disparou(2, (), entrada=entrada)
    janela._encaminha(CHAVE_DA_ETAPA[2], _resposta(200, _corpo_da_etapa_com("none")))
    feita = janela._etapas.feita(2)
    assert feita is not None and feita.entrada == entrada, "âncora: a etapa 2 não concluiu"
    assert janela.graficos._pcas_da_correcao == {"corretor": corretor}


def _corpo_da_execucao_com(corretor: str, chave: str) -> dict[str, Any]:
    """Um `200` de `/execucoes` em que só o par e o resumo das distribuições importam."""
    return {
        "matrix_id": 1,
        "apresentacao": "neg_c18|none",
        "chave_de_invalidacao": chave * 64,
        "pcas_da_correcao": {"corretor": corretor},
        "distribuicoes": {"rotulo": corretor},
    }


def test_a_execucao_so_escreve_as_figuras_das_etapas_que_ela_reproduz(qtbot: Any) -> None:
    """A janela lê do estado das etapas o que o `200` de `/execucoes` pode escrever: a Análise sobre
    a `original` nada; sobre a `corrigida`, a Correção guardada; sobre a `normalizada` que partiu
    da corrigida, as duas. **Par causal:** a mesma Análise, com o mesmo corpo, sobre a original e
    sobre a corrigida — o par da Correção só muda na segunda. E a Análise sobre a normalizada
    escreve o resumo das distribuições, que nenhuma das duas primeiras escreveu.

    **A terceira linha era o «Executar pipeline» até 09/11/2026**, que corria a cadeia da tela e
    escrevia as duas; ele saiu com a aba «Executar e exportar», e a mesma cadeia é a da Análise
    sobre a normalizada."""
    janela, _transporte = _janela_do_passo_a_passo(qtbot)
    janela.chama = lambda *_a, **_k: None  # type: ignore[method-assign]
    assert janela._etapas_que_a_execucao_escreve() == frozenset(), "âncora: nada em voo"
    janela._dispara_etapa(1)
    janela._encaminha(CHAVE_DA_ETAPA[1], _resposta(200, _corpo_da_etapa_com("none")))
    assert janela.graficos._pcas_da_correcao == {"corretor": "none"}, "âncora: a Correção"

    escreve: dict[str, frozenset[int]] = {}
    par: dict[str, Any] = {}
    for entrada, chave in (("original", "a"), ("corrigida", "b")):
        janela._escolhe_a_entrada(3, entrada)
        janela._dispara_etapa(3)
        em_curso = janela._etapas._em_curso.get(3)
        assert em_curso is not None and em_curso.entrada == entrada, f"âncora: {entrada}"
        escreve[entrada] = janela._etapas_que_a_execucao_escreve()
        # Um corretor de verdade no corpo, que o resumo metodológico lê; o valor só tem de diferir
        # do da Correção, para a escrita se ver.
        janela._encaminha(CHAVE_DA_EXECUCAO, _resposta(200, _corpo_da_execucao_com("loess", chave)))
        par[entrada] = janela.graficos._pcas_da_correcao
    assert escreve == {"original": frozenset(), "corrigida": frozenset({1})}
    assert par == {"original": {"corretor": "none"}, "corrigida": {"corretor": "loess"}}
    assert janela.graficos._distribuicoes is None, "uma Análise escreveu a figura da Normalização"

    janela._escolhe_a_entrada(2, "corrigida")
    janela._dispara_etapa(2)
    janela._encaminha(CHAVE_DA_ETAPA[2], _resposta(200, {}))
    feita = janela._etapas.feita(2)
    assert feita is not None and feita.entrada == "corrigida", "âncora: a Normalização"
    janela._escolhe_a_entrada(3, "normalizada")
    janela._dispara_etapa(3)
    em_curso = janela._etapas._em_curso.get(3)
    assert em_curso is not None and em_curso.entrada == "normalizada", "âncora: a normalizada"
    assert janela._etapas_que_a_execucao_escreve() == frozenset({1, 2})
    janela._encaminha(CHAVE_DA_EXECUCAO, _resposta(200, _corpo_da_execucao_com("serrf", "c")))
    assert janela.graficos._distribuicoes == {"rotulo": "serrf"}


# ---------------------------------------------------------------------------
# A sugestão ao lado de «Seguir sugestões do app»
# ---------------------------------------------------------------------------
#
# Feita a análise do estudo em «Preparar o estudo», as telas com «Seguir sugestões do app»
# mostram as sugestões ao lado do botão, de forma sucinta: qual corretor e quantos lotes são
# indicados. A frase é a que o clique aplicaria: os lotes que o «Diagnóstico»
# deixou guardados e, só com o Palpite em dia para esta entrada, o corretor dele.


def _janela_da_indicacao(
    qtbot: Any, pasta: Path, transporte: TestClient
) -> tuple[JanelaPrincipal, list[str]]:
    """A janela com o par escolhido pelo seletor e as rotas GRAVADAS, e não enviadas.

    **Os bytes não importam**: a indicação lê o que o pipeline guardou e a identidade do Palpite,
    que é a dos bytes e do modo, e nada aqui chega a uma rota. O modo vem do nome do arquivo.
    O relógio das etapas só corre com a janela à vista, e esta não é mostrada: quem o faz andar é o
    teste, chamando `_atualiza_as_etapas` — a mesma volta que o relógio dá a cada meio segundo.
    """
    medidas = pasta / "measurements_neg_c18.csv"
    lista = pasta / "Sample_List_neg_c18.csv"
    medidas.write_bytes(b"medicoes escolhidas")
    lista.write_bytes(b"ordem escolhida")
    janela = JanelaPrincipal(
        transporte, escolhedor={ROTULO_DAS_MEDIDAS: medidas, ROTULO_DA_LISTA: lista}.get
    )
    qtbot.addWidget(janela)
    rotas: list[str] = []
    janela.chama = lambda _metodo, rota, **_k: rotas.append(rota)  # type: ignore[method-assign]
    janela.arquivos.escolhe_medidas()
    janela.arquivos.escolhe_lista()
    assert janela.arquivos.par is not None, "âncora: o par não se formou"
    assert janela.arquivos.modo.currentText() == "neg_c18", "âncora: o modo não veio do nome"
    return janela, rotas


def _indicacao(janela: JanelaPrincipal, etapa: int = 1) -> Any:
    """O rótulo da indicação no bloco das sugestões da página da `etapa`."""
    bloco = janela._paginas_de_etapa[etapa].sugestoes
    assert bloco is not None, f"âncora: a etapa {etapa} ficou sem o bloco das sugestões"
    return bloco.indicacao


def _palpite_em_dia_com(janela: JanelaPrincipal, corretor: str) -> None:
    """O Palpite desta entrada chegou e sugeriu `corretor` — o estado que `_palpite_chegou` e
    `_pre_preenche_com_a_proposta` deixam, sem a rota."""
    assert janela.pipeline.guarda_a_sugestao_do_corretor(corretor), "âncora: o corretor recusado"
    janela._palpite_em_dia = janela._identidade_do_palpite()
    assert janela._palpite_em_dia is not None, "âncora: sem identidade não há Palpite em dia"


def test_d708_sem_leitura_a_indicacao_esta_vazia_e_o_porque_a_vista(
    qtbot: Any, tmp_path: Path, transporte_real: TestClient
) -> None:
    """Sem leitura, nada aparece ao lado do botão, e o «?» continua lá — ele abre o que hoje diz
    «Nenhuma leitura feita». Na Análise não há bloco das sugestões, nem «?»."""
    janela, _ = _janela_da_indicacao(qtbot, tmp_path, transporte_real)
    janela._atualiza_as_etapas()
    for etapa in (1, 2):
        bloco = janela._paginas_de_etapa[etapa].sugestoes
        assert bloco is not None, f"âncora: a etapa {etapa} ficou sem o bloco das sugestões"
        assert bloco.indicacao.text() == "", f"etapa {etapa}: {bloco.indicacao.text()!r}"
        assert bloco.indicacao.isHidden(), f"etapa {etapa}: a indicação vazia ocupa a linha"
        assert not bloco.porque.isHidden(), f"etapa {etapa}: o «?» sumiu sem leitura"
    assert janela._paginas_de_etapa[3].sugestoes is None, "a Análise ganhou as sugestões"
    assert janela._paginas_de_etapa[3].justificativa is None, "a Análise ganhou o «?»"


def test_d708_a_indicacao_segue_o_diagnostico_e_o_palpite_em_dia(
    qtbot: Any, tmp_path: Path, transporte_real: TestClient
) -> None:
    """Os lotes guardados aparecem sozinhos; com o Palpite em dia para esta entrada, o corretor dele
    vem junto. **Um Diagnóstico novo com outro número muda a partição**, e o Palpite guardado passa
    a ser de outra entrada: o clique mediria de novo, e a frase fica só com os lotes.

    A Normalização não ganha frase em nenhum momento: lá a máquina não propõe nada."""
    janela, _ = _janela_da_indicacao(qtbot, tmp_path, transporte_real)
    janela.pipeline.guarda_a_sugestao_de_lotes(2)
    janela._atualiza_as_etapas()
    assert _indicacao(janela).text() == "Indicado: 2 lotes"
    assert not _indicacao(janela).isHidden()

    _palpite_em_dia_com(janela, "serrf")
    janela._atualiza_as_etapas()
    assert _indicacao(janela).text() == "Indicado: SERRF, 2 lotes"

    janela._pre_preenche_com_a_proposta("diagnostico", {"lotes": {"n_lotes": 3}})
    janela._atualiza_as_etapas()
    assert janela._palpite_em_dia != janela._identidade_do_palpite(), "âncora: a partição ficou"
    assert _indicacao(janela).text() == "Indicado: 3 lotes"
    assert _indicacao(janela, 2).text() == "", "a Normalização ganhou uma indicação"
    assert _indicacao(janela, 2).isHidden()


def test_d708_um_lote_e_singular(qtbot: Any, tmp_path: Path, transporte_real: TestClient) -> None:
    """«1 lote», e não «1 lotes» — com e sem o corretor."""
    janela, _ = _janela_da_indicacao(qtbot, tmp_path, transporte_real)
    janela.pipeline.guarda_a_sugestao_de_lotes(1)
    janela._atualiza_as_etapas()
    assert _indicacao(janela).text() == "Indicado: 1 lote"
    _palpite_em_dia_com(janela, "loess")
    janela._atualiza_as_etapas()
    assert _indicacao(janela).text() == "Indicado: LOESS, 1 lote"


@pytest.mark.parametrize("em_dia", [True, False])
def test_d708_a_indicacao_e_o_que_o_clique_aplicaria(
    em_dia: bool, qtbot: Any, tmp_path: Path, transporte_real: TestClient
) -> None:
    """A frase ao lado do botão diz o que «Seguir sugestões do app» põe nos campos — lido nos
    widgets depois do clique, e não na função da frase.

    **Com o Palpite em dia**, o clique aplica na hora o corretor guardado e os lotes, e a frase os
    diz. **Com o Palpite de outra partição** — o corretor `serrf` guardado de um Palpite medido num
    lote, e o «Diagnóstico» que depois sugeriu dois —, o clique põe os lotes e mede o Palpite de
    novo, e o corretor guardado não entra: a frase não o diz. **Par causal, feito à mão uma vez em
    11/11/2026:** com a frase a mostrar o corretor guardado sem a condição do Palpite em dia, o
    ramo `em_dia=False` reprova na asserção da frase — ela diz «SERRF», que o clique não pôs.
    """
    janela, rotas = _janela_da_indicacao(qtbot, tmp_path, transporte_real)
    pipeline = janela.pipeline
    bloco = janela.pagina_da_correcao.sugestoes
    assert bloco is not None
    if em_dia:
        pipeline.guarda_a_sugestao_de_lotes(2)
        _palpite_em_dia_com(janela, "serrf")
    else:
        _palpite_em_dia_com(janela, "serrf")
        pipeline.guarda_a_sugestao_de_lotes(2)
        assert janela._palpite_em_dia != janela._identidade_do_palpite(), "âncora: em dia"
    corretor_antes = pipeline.corretor.currentText()
    assert corretor_antes != "serrf", "âncora: o seletor já estava no corretor sugerido"
    assert pipeline.n_lotes.value() != 2, "âncora: o campo já estava nos lotes sugeridos"
    janela._atualiza_as_etapas()
    frase = bloco.indicacao.text()
    antes = len(rotas)

    bloco.botao.click()

    assert pipeline.n_lotes.value() == 2, "o clique não pôs os lotes que a frase indica"
    if em_dia:
        assert frase == "Indicado: SERRF, 2 lotes"
        assert pipeline.corretor.currentText() == "serrf", "o clique não pôs o corretor indicado"
        assert rotas[antes:] == [], f"o Palpite em dia foi medido de novo: {rotas[antes:]}"
    else:
        assert frase == "Indicado: 2 lotes"
        assert pipeline.corretor.currentText() == corretor_antes, "o clique pôs um corretor velho"
        assert rotas[antes:], "âncora: com o Palpite de outra partição o clique não mediu"


def test_d708_trocar_o_modo_tira_o_corretor_da_indicacao(
    qtbot: Any, tmp_path: Path, transporte_real: TestClient
) -> None:
    """O modo é parte da entrada do Palpite: com outro modo, o Palpite em dia é de outra matriz, e
    o clique mediria de novo."""
    janela, _ = _janela_da_indicacao(qtbot, tmp_path, transporte_real)
    janela.pipeline.guarda_a_sugestao_de_lotes(2)
    _palpite_em_dia_com(janela, "serrf")
    janela._atualiza_as_etapas()
    assert _indicacao(janela).text() == "Indicado: SERRF, 2 lotes", "âncora: o corretor"

    janela.arquivos.modo.setCurrentText("pos_c18")
    janela._atualiza_as_etapas()

    assert janela.arquivos.modo.currentText() == "pos_c18", "âncora: o modo não mudou"
    assert _indicacao(janela).text() == "Indicado: 2 lotes"


def test_d708_trocar_o_par_esvazia_a_indicacao(
    qtbot: Any, tmp_path: Path, transporte_real: TestClient
) -> None:
    """Um par novo esquece os lotes e o corretor do anterior (`esquece_as_propostas_do_par`), e a
    frase some com eles — pelo caminho da troca, o seletor de arquivos."""
    janela, _ = _janela_da_indicacao(qtbot, tmp_path, transporte_real)
    janela.pipeline.guarda_a_sugestao_de_lotes(2)
    _palpite_em_dia_com(janela, "serrf")
    janela._atualiza_as_etapas()
    assert _indicacao(janela).text() == "Indicado: SERRF, 2 lotes", "âncora: a indicação"
    par_antes = janela.arquivos.par

    (tmp_path / "measurements_neg_c18.csv").write_bytes(b"outras medicoes")
    janela.arquivos.escolhe_medidas()
    janela._atualiza_as_etapas()

    assert janela.arquivos.par is not None and janela.arquivos.par != par_antes, "âncora: o par"
    assert _indicacao(janela).text() == ""
    assert _indicacao(janela).isHidden()


def test_recolocar_o_mesmo_arquivo_tira_o_palpite_de_dia(
    qtbot: Any, tmp_path: Path, transporte_real: TestClient
) -> None:
    """Remover o `measurements` e recolocá-lo apaga o corretor sugerido (`esquece_as_propostas_do_
    par`), e o Palpite deixa de estar em dia junto com ele. **A identidade volta igual** — os mesmos
    bytes, o mesmo modo, e um Diagnóstico com a mesma partição —, e antes da revisão de código de
    11/11/2026 o clique a tomava por em dia: aplicava sem medir, sem corretor nenhum, e dizia que o
    Palpite tinha chegado sem corretor.

    **Par causal feito à mão uma vez em 11/11/2026:** com `principal.py` de `4a8abd11`, a asserção
    do Palpite em dia reprova, e o clique não pede rota nenhuma.
    """
    janela, rotas = _janela_da_indicacao(qtbot, tmp_path, transporte_real)
    janela.pipeline.guarda_a_sugestao_de_lotes(2)
    _palpite_em_dia_com(janela, "serrf")
    identidade = janela._identidade_do_palpite()
    janela._atualiza_as_etapas()
    assert _indicacao(janela).text() == "Indicado: SERRF, 2 lotes", "âncora: a indicação"

    janela.arquivos.remover_arquivo("medidas")
    janela.arquivos.escolhe_medidas()
    janela.pipeline.guarda_a_sugestao_de_lotes(2)
    janela._atualiza_as_etapas()

    assert janela._identidade_do_palpite() == identidade, "âncora: a identidade não voltou igual"
    assert janela.pipeline.corretor_sugerido is None, "âncora: o corretor do par anterior ficou"
    assert janela._palpite_em_dia is None, "o Palpite ficou em dia sem o corretor que ele sugeriu"
    assert _indicacao(janela).text() == "Indicado: 2 lotes"
    bloco = janela.pagina_da_correcao.sugestoes
    assert bloco is not None
    antes = len(rotas)
    bloco.botao.click()
    assert rotas[antes:], "o clique não mediu o Palpite do par recolocado"


def test_d708_a_frase_da_indicacao_e_sucinta_e_recusa_o_corretor_desconhecido() -> None:
    """Só o corretor e os lotes, pelos nomes que o seletor desenha; nada sem sugestão; e um
    corretor fora do vocabulário levanta, como em `frase_das_sugestoes_da_correcao`, em vez de cair
    num nome qualquer."""
    from refino.desktop.janela.execucao import frase_da_indicacao

    assert frase_da_indicacao(None, None) == ""
    assert frase_da_indicacao("none", None) == "Indicado: Sem correção"
    assert frase_da_indicacao(None, 3) == "Indicado: 3 lotes"
    assert frase_da_indicacao("serrf", 1) == "Indicado: SERRF, 1 lote"
    with pytest.raises(KeyError):
        frase_da_indicacao("metodo_novo", 2)


def test_d708_a_indicacao_so_e_reescrita_quando_muda(
    qtbot: Any, tmp_path: Path, transporte_real: TestClient
) -> None:
    """O relógio pede a frase a cada meio segundo; o bloco só a escreve quando ela muda, como a
    Entrada (`_vista_da_entrada`). **Par causal:** duas voltas sem mudança escrevem uma vez; uma
    mudança escreve outra vez."""
    janela, _ = _janela_da_indicacao(qtbot, tmp_path, transporte_real)
    rotulo = _indicacao(janela)
    escritas: list[str] = []
    original = rotulo.setText

    def conta(texto: str) -> None:
        escritas.append(texto)
        original(texto)

    rotulo.setText = conta  # type: ignore[method-assign]
    janela.pipeline.guarda_a_sugestao_de_lotes(2)
    janela._atualiza_as_etapas()
    janela._atualiza_as_etapas()
    assert escritas == ["Indicado: 2 lotes"], escritas
    janela.pipeline.guarda_a_sugestao_de_lotes(3)
    janela._atualiza_as_etapas()
    assert escritas == ["Indicado: 2 lotes", "Indicado: 3 lotes"], escritas

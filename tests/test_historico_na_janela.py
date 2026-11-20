"""O histórico do multiverso na janela.

## O que esta suíte prova

| Camada | Desfechos exercitados |
|---|---|
| os dois pedidos | listar sem pré-condição; abrir sem seleção; abrir com seleção |
| a tabela do histórico | lista cheia; vazia com `200`; erro limpa a tabela; o teto declarado |
| a conclusão gravada | com tabela; sem tabela por falta de par; sem tabela por parada; erro |
| o disparo | declara `persistido=False` **com o motivo**, e não em silêncio |
| ligação à janela | os dois sinais chegam ao transporte, por EMISSÃO e não por introspecção |

## Escopo negativo

**Nada aqui toca o banco.** O que se mede é o painel a ler corpos de resposta; que o banco grave o
que devia é `tests/test_multiverso_em_disco.py`, e que a rota sirva o que o banco tem é
`tests/test_historico_de_multiversos.py`. As três camadas são separadas de propósito — foi a
costura entre elas que uma revisão adversarial achou com 307 testes verdes.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from refino.desktop.janela.multiverso import (
    LIMITE_DO_HISTORICO,
    SEM_HISTORICO,
    PainelDeMultiverso,
    rota_da_gravada,
    rota_do_historico,
)

pytestmark = pytest.mark.usefixtures("qtbot")


@pytest.fixture
def painel(qtbot: Any) -> PainelDeMultiverso:
    alvo = PainelDeMultiverso()
    qtbot.addWidget(alvo)
    return alvo


def _linha(identificador: str, **extra: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "identificador": identificador,
        "estado": "concluido",
        "estudo": "estudo-x",
        "mode": "neg_c18",
        "total": 96,
        "n_feitas": 96,
        "iniciado_em": "2026-08-28T12:00:00Z",
        "terminado_em": "2026-08-28T12:01:00Z",
        "n_features": 3,
        "grupos": None,
        "celula_que_falhou": None,
        "erro": None,
    }
    return base | extra


# ---------------------------------------------------------------------------
# AS DUAS ROTAS — uma fonte para o caminho, conferida contra o esquema
# ---------------------------------------------------------------------------


def test_as_rotas_do_historico_batem_com_o_esquema_da_api() -> None:
    """As funções de rota deste painel apontam para caminhos que a API de facto serve.

    **Contra a OpenAPI e não contra uma string escrita aqui**, pelo mesmo argumento de
    `test_as_duas_rotas_novas_batem_com_o_esquema_da_api`: uma comparação contra literal seria a
    segunda cópia do caminho, e as duas divergiriam juntas sem nada acusar.
    """
    from refino.api.app import build_app

    caminhos = set(build_app().openapi()["paths"])

    assert caminhos, "âncora: a OpenAPI não trouxe caminho nenhum"
    assert rota_do_historico().split("?")[0] in caminhos
    assert rota_da_gravada("{identificador}") in caminhos


def test_a_rota_da_gravada_nao_e_a_rota_do_resultado() -> None:
    """As duas conclusões vêm de sítios diferentes, e confundi-las seria um defeito.

    `rota_do_resultado` lê o registro deste processo — some ao fechar o programa, e some antes
    disso quando a poda aperta. `rota_da_gravada` lê o banco. Um nome só para as duas faria o
    painel prometer persistência ao servir memória, e o profissional descobriria a diferença ao
    reabrir o programa.
    """
    from refino.desktop.janela.multiverso import rota_do_resultado

    assert rota_da_gravada("abc") != rota_do_resultado("abc")


# ---------------------------------------------------------------------------
# OS DOIS PEDIDOS
# ---------------------------------------------------------------------------


def test_ver_historico_nao_exige_execucao_nesta_sessao(painel: PainelDeMultiverso) -> None:
    """**A única ação do painel sem pré-condição, e a ausência dela é o ponto.**

    As outras quatro falam de um identificador que este processo criou. Exigir um disparo antes de
    consultar o histórico negaria exactamente o caso de uso que ele serve: ver o que ficou de
    ontem.
    """
    pedidos: list[int] = []
    painel.ler_historico.connect(pedidos.append)

    painel.botao_do_historico.click()

    assert pedidos == [LIMITE_DO_HISTORICO]


def test_abrir_sem_selecao_escreve_a_frase_em_vez_de_emitir(painel: PainelDeMultiverso) -> None:
    """Sem linha selecionada, o botão **diz** o que fazer — não volta em silêncio."""
    pedidos: list[str] = []
    painel.abrir_gravada.connect(pedidos.append)

    painel.botao_de_abertura.click()

    assert pedidos == [], "emitiu um pedido sem haver seleção"
    assert "selecion" in painel.rotulo_do_historico.text().lower()


def test_abrir_com_selecao_emite_o_identificador_da_linha_selecionada(
    painel: PainelDeMultiverso,
) -> None:
    """**Lê a tabela do histórico, e NÃO o campo de texto.**

    Os dois existem e servem coisas diferentes: o campo é para colar um identificador desta sessão,
    a tabela é para escolher um de qualquer sessão. Ler o campo aqui faria o duplo clique numa
    linha do histórico abrir **outra** execução — e a planta abaixo separa as duas, porque põe
    valores diferentes nos dois sítios.
    """
    painel.mostra_historico(200, {"execucoes": [_linha("mv-do-disco")], "limite": 50})
    painel.campo_do_identificador.setText("mv-do-campo")
    painel.tabela_do_historico.setCurrentCell(0, 0)
    pedidos: list[str] = []
    painel.abrir_gravada.connect(pedidos.append)

    painel.botao_de_abertura.click()

    assert pedidos == ["mv-do-disco"], (
        f"o botão emitiu {pedidos}; ler o campo de texto em vez da linha selecionada faria o "
        "duplo clique no histórico abrir outra execução"
    )


# ---------------------------------------------------------------------------
# A TABELA DO HISTÓRICO
# ---------------------------------------------------------------------------


def test_a_lista_enche_a_tabela_na_ordem_em_que_veio(painel: PainelDeMultiverso) -> None:
    """A ordem é da rota — mais recente primeiro — e o painel **não** reordena."""
    corpo = {"execucoes": [_linha("mv-2"), _linha("mv-1"), _linha("mv-0")], "limite": 50}

    painel.mostra_historico(200, corpo)

    assert painel.linhas_do_historico() == ("mv-2", "mv-1", "mv-0")
    assert "3 execução" in painel.rotulo_do_historico.text()


def test_lista_vazia_com_200_nao_e_erro_e_diz_o_que_fazer(painel: PainelDeMultiverso) -> None:
    """**Banco em pé e nada gravado é estado legítimo**, e a mensagem é a de repouso.

    A distinção contra o `503` — banco que não responde — é a que separa *"ainda não corri nada"*
    de *"não consigo ler o que corri"*, e as duas pedem ações opostas.
    """
    painel.mostra_historico(200, {"execucoes": [], "limite": 50})

    assert painel.linhas_do_historico() == ()
    assert painel.rotulo_do_historico.text() == SEM_HISTORICO


def test_um_erro_limpa_a_tabela_do_historico(painel: PainelDeMultiverso) -> None:
    """**A tabela anterior não fica ao lado de uma mensagem de erro.**

    Deixá-la afirmaria que aquela lista é desta leitura — o estado *stale* que já apareceu no
    rótulo de desfecho. O par causal é o ponto: primeiro enche, depois erra, e a tabela tem de
    esvaziar.
    """
    painel.mostra_historico(200, {"execucoes": [_linha("mv-1")], "limite": 50})
    assert painel.linhas_do_historico() == ("mv-1",), "âncora: a tabela não encheu"

    texto = painel.mostra_historico(503, {"detail": "o banco não respondeu"})

    assert painel.linhas_do_historico() == ()
    assert "banco" in texto


def test_a_lista_que_alcanca_o_teto_declara_que_pode_haver_mais(
    painel: PainelDeMultiverso,
) -> None:
    """**É aí que a lista deixa de ser o histórico e passa a ser as N mais recentes.**

    Sem esta linha, quem recebe exactamente 50 conclui que viu tudo.
    """
    cheia = {"execucoes": [_linha(f"mv-{i}") for i in range(3)], "limite": 3}
    curta = {"execucoes": [_linha(f"mv-{i}") for i in range(2)], "limite": 3}

    texto_cheio = painel.mostra_historico(200, cheia)
    texto_curto = painel.mostra_historico(200, curta)

    assert "pode haver mais" in texto_cheio
    assert "pode haver mais" not in texto_curto, (
        "uma lista que não alcança o teto não está cortada, e dizer que pode estar é ruído"
    )


def test_o_estado_interrompido_aparece_na_tabela_como_qualquer_outro(
    painel: PainelDeMultiverso,
) -> None:
    """`interrompido` é o quinto estado, e ele existe só em disco.

    Sem ele o desfecho não seria sucesso nem falha, e sumiria do agregado sem reprovar. A tabela
    mostra-o como qualquer outro, e é isso que o torna contável.
    """
    corpo = {"execucoes": [_linha("mv-morta", estado="interrompido", n_feitas=7)], "limite": 50}

    painel.mostra_historico(200, corpo)

    item = painel.tabela_do_historico.item(0, 1)
    assert item is not None and item.text() == "interrompido"


# ---------------------------------------------------------------------------
# A CONCLUSÃO GRAVADA
# ---------------------------------------------------------------------------


def _conclusao(**extra: Any) -> dict[str, Any]:
    return {
        "execucao": _linha("mv-1", grupos=["ATLL", "HAM"], **extra),
        "estabilidade": [
            {
                "feature": "1.00_100.0m/z",
                "frac_significant": 0.5,
                "median_qvalue": 0.02,
                "median_rank": 3.0,
                "rank_iqr": 1.0,
                "sign_consistency": 0.9,
                "n_cells": 96,
            }
        ],
    }


def test_a_conclusao_gravada_enche_a_tabela_e_o_cabecalho_diz_de_onde_veio(
    painel: PainelDeMultiverso,
) -> None:
    """**A mesma tabela da conclusão em memória, e é o CABEÇALHO que as distingue.**

    Duas tabelas iguais lado a lado fariam o profissional comparar dois números que são o mesmo. O
    que muda é a procedência — e sem essa linha as duas ficariam de facto indistinguíveis.
    """
    texto = painel.mostra_gravada(200, _conclusao())

    assert painel.linhas_mostradas() == ("1.00_100.0m/z",)
    assert "histórico" in texto.lower(), f"o cabeçalho não diz de onde veio: {texto!r}"
    assert "ATLL x HAM" in texto


def test_execucao_sem_par_de_grupos_diz_a_causa_certa(painel: PainelDeMultiverso) -> None:
    """Sem par declarado, `resume_o_multiverso` nem é chamado — e isso é desfecho, não falta."""
    corpo = {"execucao": _linha("mv-1", grupos=None), "estabilidade": []}

    texto = painel.mostra_gravada(200, corpo)

    assert painel.linhas_mostradas() == ()
    assert "par de grupos" in texto


def test_execucao_que_parou_antes_diz_outra_causa(painel: PainelDeMultiverso) -> None:
    """**O par causal da anterior, e é ele que impede a mensagem de ser sempre a mesma.**

    Sem esta metade, um painel que escrevesse *"nenhum par foi declarado"* em todo caso sem tabela
    passaria no teste de cima — e diria a causa errada a quem cancelou uma execução que **tinha**
    par.
    """
    corpo = {
        "execucao": _linha("mv-1", grupos=["ATLL", "HAM"], estado="cancelado", n_feitas=7),
        "estabilidade": [],
    }

    texto = painel.mostra_gravada(200, corpo)

    assert "par de grupos" not in texto, (
        "a causa está errada: esta execução TINHA par declarado, e parou antes de medir"
    )
    assert "parou antes" in texto


def test_um_erro_na_conclusao_gravada_limpa_a_tabela(painel: PainelDeMultiverso) -> None:
    """`404` de identificador desconhecido não deixa a conclusão anterior na tela."""
    painel.mostra_gravada(200, _conclusao())
    assert painel.linhas_mostradas() == ("1.00_100.0m/z",), "âncora: a tabela não encheu"

    texto = painel.mostra_gravada(404, {"detail": "não há execução gravada"})

    assert painel.linhas_mostradas() == ()
    assert "não há execução gravada" in texto


# ---------------------------------------------------------------------------
# O DISPARO DECLARA A GRAVAÇÃO
# ---------------------------------------------------------------------------


def test_um_disparo_gravado_nao_avisa_nada(painel: PainelDeMultiverso) -> None:
    """A âncora: sem aviso quando não há o que avisar."""
    texto = painel.aceita_disparo(
        202,
        {"identificador": "mv-1", "total": 96, "persistido": True, "motivo_nao_gravado": None},
    )

    assert "NÃO será gravada" not in texto


def test_um_disparo_nao_gravado_avisa_com_o_motivo(painel: PainelDeMultiverso) -> None:
    """O silêncio seria o defeito.

    A persistência é melhor esforço: sem banco a grade corre e nada fica em disco. Não dizer isso
    faria o profissional fechar o programa a contar com um histórico que nunca existiu, e
    descobri-lo só ao reabrir.
    """
    texto = painel.aceita_disparo(
        202,
        {
            "identificador": "mv-1",
            "total": 96,
            "persistido": False,
            "motivo_nao_gravado": "REFINO_DB_HOST ausente",
        },
    )

    assert "NÃO será gravada" in texto
    assert "REFINO_DB_HOST ausente" in texto, "avisou sem dizer por quê, que é metade do defeito"
    assert "96 células" in texto, "o aviso comeu a mensagem de disparo em vez de a acompanhar"


# ---------------------------------------------------------------------------
# A COSTURA — os dois sinais chegam à rota PELA JANELA
# ---------------------------------------------------------------------------
#
# **Isto é o que nenhum dos dois lados vê sozinho.** O painel é testado isolado acima, a rota é
# testada isolada em `test_historico_de_multiversos.py`, e o fio entre os dois não — a revisão
# adversarial mediu que um `disconnect()` nas quatro ligações do multiverso não reprovava
# nada, com 307 testes verdes. Os portões existentes não viam: o da bijeção mede por **painel**, e
# `multiverso` já contava como ligado; o de método público **ignora nomes com `_`**, e os slots são
# privados.
#
# **A asserção é a ROTA que sai no transporte**, e não a existência do `connect`: introspecção diria
# *"há uma ligação"* sem dizer para onde ela vai.


def _janela(qtbot: Any) -> Any:
    from fastapi.testclient import TestClient

    from refino.api.app import build_app
    from refino.desktop.janela.principal import JanelaPrincipal

    j = JanelaPrincipal(TestClient(build_app(), raise_server_exceptions=False))
    qtbot.addWidget(j)
    return j


def _com_arquivos(janela: Any) -> Any:
    """Põe o par REAL do laboratório no painel 1, pelos resolvedores do `conftest`.

    **O par real e não um fabricado, pela regra 1 do README** — mais o que `_par_real` de
    `test_desktop_janela.py` mediu: uma matriz 2x4 à mão faz `/matrizes/validar` devolver `500`, e
    o teste reprovaria por defeito do dado com a mensagem a apontar para a ligação. Os resolvedores
    fazem `skip` com mensagem quando o arquivo falta.
    """
    from refino.desktop.janela.arquivos import ParDeArquivos
    from tests.conftest import measurements_path, sample_list_path

    medidas = measurements_path("neg_c18")
    lista = sample_list_path("neg_c18")
    janela.arquivos._par = ParDeArquivos(
        medidas.name, medidas.read_bytes(), lista.name, lista.read_bytes()
    )
    janela.arquivos.modo.setCurrentText("neg_c18")
    return janela


def _drena(janela: Any) -> None:
    """Esvazia a piscina **e** processa os eventos que a resposta enfileira.

    **As duas metades, e a segunda foi esquecida na primeira versão destes dois testes.** Drenar a
    piscina espera o trabalho de fundo; o sinal `resposta_recebida` chega à janela por evento de
    Qt, e sem `processEvents` ele fica na fila — a lista de chaves sai **vazia** e o teste acusa
    uma ligação partida que está inteira. É a mesma sequência que
    `test_u11_o_botao_de_progresso_chama_a_rota_do_trabalho` já usava.
    """
    from PySide6.QtWidgets import QApplication

    assert janela.drena_a_piscina(30_000), "a piscina não esvaziou"
    QApplication.processEvents()


def test_ver_historico_chega_a_rota_pela_janela(qtbot: Any) -> None:
    """`GET /multiversos/historico` sai de facto quando o botão é clicado.

    **O desfecho da rota não importa aqui** — sem banco ela responde `503`, e esse é o desfecho
    CERTO de medir: ele prova que a chamada saiu e voltou. O que se afirma é a chave roteada.
    """
    janela = _janela(qtbot)
    chaves: list[str] = []
    janela.resposta_recebida.connect(lambda chave, _r: chaves.append(chave))

    janela.multiverso.botao_do_historico.click()

    _drena(janela)
    assert chaves == ["multiverso_historico"], (
        f"«Ver histórico» não chegou à rota: {chaves}. Um `disconnect` no sinal `ler_historico` "
        "deixaria esta lista vazia, e nenhum outro portão o acusaria"
    )
    assert janela.multiverso.rotulo_do_historico.text() != SEM_HISTORICO, (
        "a resposta não chegou ao painel: a chave foi roteada e o método não foi chamado"
    )


def test_abrir_a_gravada_chega_a_rota_pela_janela(qtbot: Any) -> None:
    """`GET /multiversos/historico/{id}` sai com o identificador da LINHA selecionada.

    A seleção entra pelo caminho de produção — `mostra_historico` —, e não por atribuição: é ela
    que enche a tabela de onde o identificador é lido.
    """
    janela = _janela(qtbot)
    janela.multiverso.mostra_historico(200, {"execucoes": [_linha("mv-costura")], "limite": 50})
    janela.multiverso.tabela_do_historico.setCurrentCell(0, 0)
    chaves: list[str] = []
    janela.resposta_recebida.connect(lambda chave, _r: chaves.append(chave))

    janela.multiverso.botao_de_abertura.click()

    _drena(janela)
    assert chaves == ["multiverso_gravada"], f"«Abrir a selecionada» não chegou à rota: {chaves}"


def test_as_duas_chaves_do_historico_sao_distintas_das_outras_do_painel() -> None:
    """**Chave repetida faria a lista de execuções cair na tabela de features.**

    Os dois corpos são dicionários com `200` — `HistoricoDeMultiversos` e `ConclusaoDoHistorico` —,
    e um `isinstance(corpo, dict)` não os distingue. É o mesmo argumento que fez as outras chaves
    do painel existirem separadas.
    """
    from refino.desktop.janela.multiverso import (
        CHAVE_DA_GRAVADA,
        CHAVE_DO_CANCELAMENTO,
        CHAVE_DO_DISPARO,
        CHAVE_DO_HISTORICO,
        CHAVE_DO_MULTIVERSO,
        CHAVE_DO_PROGRESSO,
    )

    todas = [
        CHAVE_DO_MULTIVERSO,
        CHAVE_DO_DISPARO,
        CHAVE_DO_PROGRESSO,
        CHAVE_DO_CANCELAMENTO,
        CHAVE_DO_HISTORICO,
        CHAVE_DA_GRAVADA,
    ]

    assert len(set(todas)) == len(todas), f"chave repetida entre as seis do painel: {todas}"


def test_o_disparo_do_multiverso_leva_o_estudo_do_painel_3(qtbot: Any) -> None:
    """**A coluna que distingue duas execuções sobre o mesmo modo e os mesmos arquivos.**

    Enquanto o multiverso vivia só em memória, o nome do estudo não tinha para onde ir. Com o
    histórico em disco, ele é a única coisa que separa *"o que corri na terça"* de *"o que corri na
    quarta"* — e sem esta ligação toda linha do histórico diria `sem_identificacao`.
    """
    janela = _com_arquivos(_janela(qtbot))
    janela.pipeline.estudo.setText("estudo-do-usuario")
    enviados: list[dict[str, Any]] = []
    original = janela.chama
    janela.chama = lambda *a, **k: enviados.append(dict(k.get("campos") or {}))  # type: ignore[method-assign]

    janela.multiverso._pede_disparo()

    janela.chama = original  # type: ignore[method-assign]
    assert enviados, (
        "o disparo não chegou a `chama`: ou os arquivos foram recusados, ou a ligação está partida"
    )
    assert enviados[0].get("estudo") == "estudo-do-usuario", (
        f"o nome do estudo não viajou: {enviados[0]}"
    )


def test_estudo_em_branco_nao_e_enviado_e_a_rota_decide_o_padrao(qtbot: Any) -> None:
    """**O par causal, e ele impede a segunda cópia do padrão.**

    `POST /multiversos` declara `estudo="sem_identificacao"` como padrão. Mandar esse literal daqui
    quando o campo está vazio criaria a segunda fonte de verdade sobre ele — e as duas divergiriam
    no dia em que uma mudasse. O campo vazio simplesmente não viaja.
    """
    janela = _com_arquivos(_janela(qtbot))
    janela.pipeline.estudo.setText("   ")
    enviados: list[dict[str, Any]] = []
    original = janela.chama
    janela.chama = lambda *a, **k: enviados.append(dict(k.get("campos") or {}))  # type: ignore[method-assign]

    janela.multiverso._pede_disparo()

    janela.chama = original  # type: ignore[method-assign]
    assert enviados, "âncora: o disparo não chegou a `chama`"
    assert "estudo" not in enviados[0], (
        f"um estudo em branco viajou, e a rota vai gravá-lo como nome: {enviados[0]}"
    )

"""A execução passa a sondar-se sozinha, e PARA por estado.

A leitura de progresso atualiza-se a cada segundo, sem clicar «Ler progresso». Antes, a tela
ficava em `pendente: 0/4 etapas · decorrido: 44 s desde o disparo` e **0%** na barra até alguém
carregar outra vez para saber que nada mudara.

## O que este portão mede, e por que a metade que importa é a de PARAR

Ligar um relógio é a metade fácil. A que ninguém pede — e a que custa caro — é ele **parar**: não
há teto de leituras, então um relógio que só parasse por contagem sondaria para sempre uma rota
que já respondeu.

**São CINCO desfechos e o portão vê os cinco**: agenda; para no `200`; para em cada um dos três
estados terminais; para sem identificador; para em código de erro.

## O relógio é conferido pelo ESTADO da espera, e não por esperar tempo real

`esperas_em_curso()` diz por quais chaves a janela está a esperar. Um teste que dormisse um segundo
para ver a releitura acontecer mediria o `QTimer` do Qt, não a decisão desta janela — e seria lento
e intermitente. O que se afirma aqui é a **decisão**: continuar ou parar.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a decisão é exercitada em `tests/test_qt_guard.py`, com planta
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()

from fastapi.testclient import TestClient  # noqa: E402
from PySide6.QtWidgets import QPushButton  # noqa: E402

from refino.api.app import build_app  # noqa: E402
from refino.api.progresso import ESTADOS, TERMINAIS  # noqa: E402
from refino.desktop.janela.principal import (  # noqa: E402
    CHAVE_DA_EXECUCAO,
    INTERVALO_DA_ESPERA_MS,
    JanelaPrincipal,
)
from refino.interface.cliente import Resposta  # noqa: E402

IDENTIFICADOR = "abc123"
TOTAL = 4


def _resposta(codigo: int, corpo: Any, *, ok: bool = True) -> Resposta:
    """Uma `Resposta` como a que `cliente.envia` produz. **`causa` é `None` sse `ok`** — a
    bicondicional é contrato do módulo, e escrevê-la aqui é o que impede o arnês de construir um
    objeto que a produção nunca produziria."""
    return Resposta(
        codigo=codigo,
        ok=ok,
        causa=None if ok else "http",
        mensagem="OK" if ok else f"HTTP {codigo}",
        corpo=corpo,
    )


@pytest.fixture
def janela(qtbot: Any) -> JanelaPrincipal:
    j = JanelaPrincipal(TestClient(build_app(), raise_server_exceptions=False))
    qtbot.addWidget(j)
    return j


def _aceita(janela: JanelaPrincipal) -> None:
    """Põe a janela no estado de quem acabou de disparar a execução — «3. Analisar e registrar»."""
    janela._encaminha(
        CHAVE_DA_EXECUCAO,
        _resposta(202, {"identificador": IDENTIFICADOR, "total": TOTAL, "reclamado": False}),
    )


def _progresso(janela: JanelaPrincipal, estado: str, *, feito: int = 1) -> None:
    janela._encaminha(
        CHAVE_DA_EXECUCAO,
        _resposta(
            202,
            {
                "identificador": IDENTIFICADOR,
                "estado": estado,
                "feito": feito,
                "total": TOTAL,
                "fracao": feito / TOTAL,
            },
        ),
    )


def test_a_cadencia_e_de_um_segundo_e_o_numero_veio_de_uma_medicao() -> None:
    """A cadência é de um segundo, e a leitura mais cara já medida custa 0,079 s.

    **Piso e teto, e não igualdade**: o que este teste protege é a ordem de grandeza — um intervalo
    de 50 ms seria rajada sobre a rota, e um de 30 s desfaria a cadência.
    """
    assert 500 <= INTERVALO_DA_ESPERA_MS <= 2_000, (
        f"a cadência é de {INTERVALO_DA_ESPERA_MS} ms, e não de um segundo; as leituras de "
        f"`GET /trabalhos/{{id}}` custaram entre 0,000 e 0,079 s com o `serrf` a correr"
    )


def test_o_aceite_abre_a_espera_e_ela_nao_existia_antes(janela: JanelaPrincipal) -> None:
    """Primeira direção: depois do `202` do disparo, a janela passa a esperar sozinha.

    **A âncora é a linha de cima**, e sem ela este teste passaria numa janela que esperasse desde
    que nasce — o que seria sondar uma rota que ninguém chamou.
    """
    assert janela.esperas_em_curso() == (), (
        "âncora: a janela nasce à espera de alguma coisa, e não devia"
    )
    assert not janela._relogio_de(CHAVE_DA_EXECUCAO).isActive(), (
        "âncora: o relógio já está armado antes de qualquer disparo"
    )

    _aceita(janela)

    assert CHAVE_DA_EXECUCAO in janela.esperas_em_curso(), (
        "depois do aceite a janela devia sondar sozinha, e continua à espera de um clique"
    )
    # **O RELÓGIO ARMADO, e sem esta linha o portão não separa «sonda» de «nunca sonda».**
    #
    # A revisão adversarial não afirmou: **plantou**. `esperas_em_curso()` lê o dicionário
    # `_esperas`; quem de facto faz a releitura acontecer é `_relogio_de(...).start(...)`, e
    # **nenhuma asserção o tocava**. Com um relógio inerte plantado só para esta chave, os onze
    # testes deste arquivo passaram — e `test_desktop_janela.py` mais este deram **252 passed**
    # com a sondagem completamente morta.
    #
    # A metade «para por estado» estava coberta em cinco direções; a metade «anda», em nenhuma.
    assert janela._relogio_de(CHAVE_DA_EXECUCAO).isActive(), (
        "a espera abriu e o relógio NÃO foi armado: a janela regista que está à espera e nunca "
        "relê. É o desfecho que `esperas_em_curso()` sozinho não distingue de sondar"
    )


def test_o_progresso_a_correr_mantem_a_espera(janela: JanelaPrincipal) -> None:
    """Uma leitura com estado não terminal continua o laço em vez de o encerrar."""
    correntes = [e for e in ESTADOS if e not in TERMINAIS]
    assert correntes, "âncora: todo estado é terminal, e então não há laço nenhum a manter"

    _aceita(janela)
    for estado in correntes:
        _progresso(janela, estado)
        assert CHAVE_DA_EXECUCAO in janela.esperas_em_curso(), (
            f"a janela parou de sondar com o trabalho em `{estado}`, que não é terminal — quem "
            f"executa fica a olhar para uma barra que não anda"
        )


@pytest.mark.parametrize("estado", sorted(TERMINAIS))
def test_todo_estado_terminal_para_a_sondagem(janela: JanelaPrincipal, estado: str) -> None:
    """As três direções que fecham o laço. **`cancelado` e `falhou` contam tanto como `concluido`.**

    Um `estado == "concluido"` escrito à mão em vez de `TERMINAIS` faria a janela sondar para
    sempre um trabalho que falhou — o defeito que a docstring de `TERMINAIS` nomeia.
    """
    _aceita(janela)
    assert CHAVE_DA_EXECUCAO in janela.esperas_em_curso(), "âncora: a espera não chegou a abrir"

    _progresso(janela, estado, feito=TOTAL)
    assert CHAVE_DA_EXECUCAO not in janela.esperas_em_curso(), (
        f"a janela continua a sondar um trabalho em `{estado}`, que é terminal — ela consulta para "
        f"sempre uma rota que já respondeu"
    )


def test_o_resultado_em_200_para_a_sondagem(janela: JanelaPrincipal) -> None:
    """O `200` é o resultado a chegar: não há mais nada que ler."""
    _aceita(janela)
    assert CHAVE_DA_EXECUCAO in janela.esperas_em_curso(), "âncora: a espera não chegou a abrir"

    janela._encaminha(CHAVE_DA_EXECUCAO, _resposta(200, {"qualidade": None}))
    assert CHAVE_DA_EXECUCAO not in janela.esperas_em_curso(), (
        "o resultado chegou e a janela continua a sondar"
    )


def test_um_codigo_de_erro_para_a_sondagem(janela: JanelaPrincipal) -> None:
    """`4xx` e `5xx` não viram progresso sozinhos, e sondar em cima deles é rajada sobre uma rota
    que já disse que não.
    """
    _aceita(janela)
    assert CHAVE_DA_EXECUCAO in janela.esperas_em_curso(), "âncora: a espera não chegou a abrir"

    janela._encaminha(CHAVE_DA_EXECUCAO, _resposta(503, {"detail": "banco ausente"}, ok=False))
    assert CHAVE_DA_EXECUCAO not in janela.esperas_em_curso(), (
        "a rota devolveu 503 e a janela continua a insistir"
    )


def test_o_botao_de_ler_progresso_SAIU_a_pedido(janela: JanelaPrincipal) -> None:  # noqa: N802
    """**O botão de ler progresso saiu**: o progresso já é atualizado automaticamente.

    **O portão fica, virado**, em vez de ser apagado: um painel que voltasse a ganhar o botão
    passaria a ter dois caminhos para a mesma leitura, e o de baixo — o relógio — é o que a
    sondagem automática já cobre.

    **A varredura é por CLASSE e não pelo atributo**: qualquer `QPushButton` do painel cujo texto
    fale em ler ou progresso reprova, porque um botão novo com outro nome de atributo nasceria
    fora de uma asserção sobre `botao_progresso`.
    """
    assert not hasattr(janela.pipeline, "botao_progresso"), (
        "`botao_progresso` voltou ao painel do pipeline — ele saiu, e a "
        "sondagem automática é o caminho único desde então"
    )
    # **AS ÁRVORES DAS PÁGINAS, e a primeira versão deste portão lia só uma.**
    #
    # Os grupos e os botões do painel nascem com ele por pai e são **reparentados** pela composição
    # — até 09/11/2026 o grupo «Executar», pela aba «Executar e exportar»; desde então cada grupo e
    # cada botão de etapa, pela página da etapa dele. Depois disso `janela.pipeline.findChildren`
    # **não os vê**, e um `botao_progresso` que voltasse para o rodapé da Análise, que é onde a
    # leitura da execução vive hoje, passaria despercebido. A âncora abaixo é o que o apanha.
    botoes = [
        b.text()
        for pai in (janela.pipeline, *janela._paginas_de_etapa.values())
        for b in pai.findChildren(QPushButton)
    ]
    # **A âncora de não-vacuidade, na MESMA função.** `findChildren` sobre um painel que
    # não construiu botão nenhum devolve lista vazia, e a asserção de ausência abaixo passaria
    # sem tocar em nada — que é a forma de vácuo que este projeto nomeia. O botão da Análise é o
    # que dispara a execução cujo progresso a sondagem lê.
    assert janela.pipeline.botoes_de_etapa[3].text() in botoes, (
        f"âncora: a varredura não alcança o botão que dispara a execução, então ela não alcançaria "
        f"o de progresso tampouco. Botões vistos: {botoes}"
    )
    suspeitos = [t for t in botoes if "progresso" in t.casefold() or t.casefold().startswith("ler")]
    assert not suspeitos, f"o painel voltou a oferecer leitura manual do progresso: {suspeitos}"


def test_a_resposta_de_um_trabalho_supersedido_e_descartada(janela: JanelaPrincipal) -> None:
    """O sexto desfecho, e ele faltava — apanhado na revisão de código de 07/09/2026.

    ## O defeito que este teste fecha

    A sondagem automática reusou `_esperas`, `_relogio_de` e `_rele`, e **não** reusou a guarda de
    supersessão que o ramo dos artefatos tem. Com duas cadeias vivas — um segundo clique em
    «Executar pipeline» enquanto a primeira ainda é sondada — a resposta da cadeia VELHA escrevia o
    progresso por cima da nova, e um `estado` terminal dela **parava a sondagem do trabalho que
    ainda corre**.

    É o mesmo defeito que `_espera_pelo_artefato` regista para a sua própria chave, e a revisão
    citou aquele comentário como o precedente que eu não segui.

    ## Por que o `pedido` é o discriminador, e não o corpo

    O identificador chega nas duas metades — na chave composta que `_rele` monta e no corpo que a
    rota devolve. **A chave é a que sabe de qual CADEIA a resposta é**; o corpo diz de qual
    trabalho, que não é a mesma pergunta quando o mesmo trabalho é reclamado por duas cadeias.
    """
    _aceita(janela)
    assert CHAVE_DA_EXECUCAO in janela.esperas_em_curso(), "âncora: a espera não chegou a abrir"
    corrente = janela.pipeline.identificador
    assert corrente == IDENTIFICADOR, f"âncora: a espera abriu com {corrente!r}"

    # A resposta da cadeia VELHA — outro identificador na chave composta — chega terminal.
    janela._encaminha(
        CHAVE_DA_EXECUCAO,
        _resposta(
            202,
            {
                "identificador": "cadeia-velha",
                "estado": "concluido",
                "feito": TOTAL,
                "total": TOTAL,
            },
        ),
        pedido="cadeia-velha",
    )

    assert CHAVE_DA_EXECUCAO in janela.esperas_em_curso(), (
        "a resposta de um trabalho SUPERSEDIDO parou a sondagem do que ainda corre — é o defeito "
        "que `_espera_pelo_artefato` já regista para a sua própria chave"
    )
    assert janela.pipeline.identificador == corrente, (
        "a resposta velha reescreveu o identificador da cadeia viva"
    )


def test_a_releitura_da_cadeia_viva_continua_a_passar(janela: JanelaPrincipal) -> None:
    """A âncora do teste acima: a guarda descarta o velho e **deixa passar o corrente**.

    Sem esta metade, uma guarda que descartasse **tudo** o que traz `pedido` passaria no teste
    irmão — e a sondagem automática deixaria de funcionar por completo, em silêncio, porque é
    exatamente ela que traz `pedido`.
    """
    _aceita(janela)
    assert CHAVE_DA_EXECUCAO in janela.esperas_em_curso(), "âncora: a espera não chegou a abrir"

    janela._encaminha(
        CHAVE_DA_EXECUCAO,
        _resposta(
            202,
            {"identificador": IDENTIFICADOR, "estado": "concluido", "feito": TOTAL, "total": TOTAL},
        ),
        pedido=IDENTIFICADOR,
    )

    assert CHAVE_DA_EXECUCAO not in janela.esperas_em_curso(), (
        "a releitura da cadeia VIVA foi descartada pela guarda de supersessão — com ela a sondagem "
        "automática nunca termina, e o laço bate para sempre contra uma rota que já respondeu"
    )

"""O histórico do multiverso pela API — item 58, 28/08/2026.

O que `test_multiverso_em_disco.py` verifica é o **repositório**; o que este módulo verifica é a
**costura**: que o disparo grava a linha de abertura, que a thread fecha com o desfecho real, que
as duas rotas novas servem o que ficou em disco, e que o disparo **diz** quando não pôde gravar em
vez de o esconder.

**A costura é o que nenhum dos dois lados vê sozinho.** O repositório é testado isolado, a rota é
testada isolada, e o fio entre os dois não — é o defeito que foi pago na janela, e que a revisão
adversarial achou com 307 testes verdes.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.desktop.banco import migra
from refino.persistence.config import VARIAVEL_DA_URL

MODO = "neg_c18"


@pytest.fixture
def com_banco(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Um SQLite migrado, apontado por `REFINO_DB_URL` para o processo inteiro.

    **`monkeypatch.setenv` e não `os.environ` direto**, e a razão está em `persistence/config.py`:
    uma variável exportada de verdade vazaria para os testes que rodam depois, e a ordem deles
    passaria a decidir o resultado — a forma mais cara de portão intermitente.
    """
    url = f"sqlite:///{(tmp_path / 'refino.db').as_posix()}"
    migra(url)
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    yield url


@pytest.fixture
def cliente() -> Iterator[TestClient]:
    with TestClient(build_app()) as c:
        yield c


def _arquivos() -> dict[str, tuple[str, bytes, str]]:
    """Uma matriz mínima em memória, no formato que a rota aceita.

    **Construída à mão e não lida da referência**, e isto NÃO viola a regra 1 do README: o que se
    mede aqui é a costura entre a rota e o banco, não um número sobre os dados do laboratório. Usar
    os arquivos reais faria cada teste correr uma grade sobre 6.035 features para afirmar que uma
    linha foi gravada.
    """
    nomes = [f"HTLV_C18_PLASMA_MSe_NEG_QC{i}" for i in range(1, 4)] + [
        f"HTLV_C18_PLASMA_MSe_NEG_{i}_ID{i}_{g}"
        for i, g in enumerate(("ATLL", "ATLL", "HAM", "HAM"), start=1)
    ]
    rotulos = ["QC", "QC", "QC", "ATLL", "ATLL", "HAM/TSP", "HAM/TSP"]
    # **Os nomes carregam `C18` e `NEG`, e isso é requisito e não enfeite:**
    # `check_mode_matches_content` confere o modo declarado contra o CONTEÚDO, e um fixture com
    # nomes neutros seria recusado no disparo com `422` — que é a guarda a funcionar.
    linhas = [
        "NAME," + ",".join(nomes),
        "LABEL," + ",".join(rotulos),
        "1.00_100.0001m/z," + ",".join(["10", "11", "10", "20", "21", "30", "31"]),
        "2.00_200.0002m/z," + ",".join(["30", "31", "30", "40", "41", "50", "52"]),
        "3.00_300.0003m/z," + ",".join(["70", "68", "71", "12", "13", "90", "88"]),
    ]
    medidas = ("\n".join(linhas) + "\n").encode()
    lista = (
        "ID,inj_order_name\n" + "".join(f"{i},{nome}\n" for i, nome in enumerate(nomes, start=1))
    ).encode()
    return {
        "measurements": ("measurements_neg_c18.csv", medidas, "text/csv"),
        "sample_list": ("Sample_List_neg_c18.csv", lista, "text/csv"),
    }


def _dispara(cliente: TestClient, **extra: Any) -> dict[str, Any]:
    """Uma grade minúscula: um corretor, uma opção de lote, e os outros eixos num valor cada."""
    dados: dict[str, Any] = {
        "mode": MODO,
        "corrections": '["none"]',
        "batch_options": "[1]",
        "rsd_cutoffs": "[0.3]",
        "missing_thresholds": "[0.5]",
        "imputations": '["one_fifth_min"]',
        "sample_norms": '["none"]',
        "transformations": '["none"]',
        "scalings": '["none"]',
        **extra,
    }
    resposta = cliente.post("/multiversos", files=_arquivos(), data=dados)
    assert resposta.status_code == 202, resposta.text
    return dict(resposta.json())


def _espera_terminar(cliente: TestClient, identificador: str, *, segundos: float = 30.0) -> str:
    """Bloqueia até o estado ser terminal e devolve-o. **Falha com o estado, nunca em silêncio.**"""
    limite = time.monotonic() + segundos
    estado = "pendente"
    while time.monotonic() < limite:
        leitura = cliente.get(f"/multiversos/{identificador}")
        if leitura.status_code == 200:
            estado = leitura.json()["estado"]
            if estado in {"concluido", "falhou", "cancelado"}:
                return str(estado)
        time.sleep(0.05)
    pytest.fail(f"a execução {identificador} não terminou em {segundos}s; ficou em {estado!r}")


def _espera_no_disco(
    cliente: TestClient, identificador: str, *, segundos: float = 10.0
) -> dict[str, Any]:
    """Bloqueia até o **disco** ter o estado terminal, e devolve a execução gravada.

    **A espera é explícita e nomeada, porque a corrida é real e está declarada** na docstring de
    `executa_registrando`: a memória vira terminal **antes** de a escrita acontecer, então há uma
    janela — de uma transação de SQLite — em que `GET /multiversos/{id}` já diz `concluido` e
    `GET /multiversos/historico` ainda diz `executando`.

    **Esperar aqui não é esconder a corrida; é sincronizar com ela.** O que estes testes afirmam é
    *"o disco registra o desfecho"*, e não *"o disco registra o desfecho no mesmo instante"* — que
    é uma propriedade que o desenho deliberadamente não promete. Um `sleep` fixo esconderia; este
    laço, com o nome que tem, torna a janela visível a quem lê o teste.
    """
    limite = time.monotonic() + segundos
    ultimo: dict[str, Any] = {}
    while time.monotonic() < limite:
        resposta = cliente.get(f"/multiversos/historico/{identificador}")
        if resposta.status_code == 200:
            ultimo = dict(resposta.json()["execucao"])
            if ultimo["estado"] != "executando":
                return ultimo
        time.sleep(0.02)
    pytest.fail(
        f"o disco não recebeu o desfecho de {identificador} em {segundos}s; ficou em "
        f"{ultimo.get('estado')!r}"
    )


# ---------------------------------------------------------------------------
# A ORDEM DAS ROTAS — e ela é a única coisa aqui que se quebra em silêncio
# ---------------------------------------------------------------------------


def test_historico_nao_e_confundido_com_um_identificador(cliente: TestClient) -> None:
    """**`GET /multiversos/historico` NÃO pode cair em `le_progresso`.**

    O FastAPI casa por ordem de registro. Com `/multiversos/{identificador}` declarada primeiro,
    esta chamada entra em `le_progresso` com `identificador="historico"` e responde **404 «não há
    execução com identificador 'historico'»** — uma mensagem plausível sobre a rota errada, que
    ninguém lê como defeito de roteamento.

    **O oráculo é o CORPO e não o código**, e a diferença é o teste: sem banco as duas rotas podem
    responder erro, e só a mensagem diz qual delas respondeu.
    """
    resposta = cliente.get("/multiversos/historico")

    assert "identificador 'historico'" not in resposta.text, (
        "`/multiversos/historico` caiu em `le_progresso`: a rota do histórico está declarada "
        "DEPOIS do padrão variável, e o FastAPI casa por ordem de registro"
    )


def test_historico_de_um_identificador_tambem_tem_rota_propria(cliente: TestClient) -> None:
    """A segunda metade do mesmo defeito: `/multiversos/historico/abc`.

    ## O oráculo é o CÓDIGO, e a primeira versão usava uma ausência INFALSIFICÁVEL

    Ela afirmava que a frase `identificador 'historico'` **não** aparece na resposta — e essa
    frase nunca poderia aparecer: `/multiversos/{identificador}` tem **um** segmento e
    `/multiversos/historico/abc` tem **dois**, então nenhuma ordem de registro os faz colidir.
    Medido com a ordem invertida de propósito: o teste irmão reprovou e **este passou**.

    > **Asserção que nenhuma mutação pode derrubar não é teste, é prosa com sintaxe de asserção.**

    O que separa é *qual rota tratou o pedido*, e sem banco isso é o **código**: `503` significa
    que `le_conclusao_gravada` correu e foi ao repositório; qualquer `404` significa que o pedido
    caiu noutro sítio.
    """
    resposta = cliente.get("/multiversos/historico/abc")

    assert resposta.status_code == 503, (
        f"esperava 503 (a rota do histórico correu e não achou banco) e veio "
        f"{resposta.status_code}: {resposta.text[:160]}. Um 404 aqui significa que "
        "`/multiversos/historico/{id}` está declarada DEPOIS do padrão variável"
    )


# ---------------------------------------------------------------------------
# O CICLO: disparar, terminar, ler do disco
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_o_disparo_grava_e_o_historico_devolve_o_que_terminou(
    cliente: TestClient, com_banco: str
) -> None:
    """**O ciclo inteiro, e é ele que responde à pergunta do item 58.**

    Dispara, espera terminar, e afirma que o disco sabe o que aconteceu — o que é a única coisa que
    sobrevive ao fechamento do programa.
    """
    aceito = _dispara(cliente, estudo="estudo-do-historico")

    assert aceito["persistido"] is True, (
        f"o disparo não gravou a linha de abertura: {aceito['motivo_nao_gravado']}"
    )
    assert aceito["motivo_nao_gravado"] is None

    estado = _espera_terminar(cliente, aceito["identificador"])
    assert estado == "concluido"
    _espera_no_disco(cliente, aceito["identificador"])

    historico = cliente.get("/multiversos/historico")
    assert historico.status_code == 200, historico.text
    corpo = historico.json()
    assert corpo["limite"] == 50, "o teto viaja com a lista, senão 50 linhas são ambíguas"
    (linha,) = corpo["execucoes"]
    assert linha["identificador"] == aceito["identificador"]
    assert linha["estado"] == "concluido"
    assert linha["estudo"] == "estudo-do-historico"
    assert linha["mode"] == MODO
    assert linha["n_feitas"] == linha["total"] == aceito["total"]
    assert linha["terminado_em"] is not None
    assert linha["erro"] is None


@pytest.mark.slow
def test_sem_par_de_grupos_a_execucao_e_gravada_e_a_conclusao_fica_vazia(
    cliente: TestClient, com_banco: str
) -> None:
    """**Execução sem conclusão EXISTE, e a tabela vazia não é `404`.**

    Sem par declarado, `resume_o_multiverso` nem é chamado — e isso é desfecho legítimo, não falta.
    Um `404` aqui diria ao profissional que o trabalho nunca aconteceu.
    """
    aceito = _dispara(cliente)
    _espera_terminar(cliente, aceito["identificador"])
    _espera_no_disco(cliente, aceito["identificador"])

    detalhe = cliente.get(f"/multiversos/historico/{aceito['identificador']}")

    assert detalhe.status_code == 200, detalhe.text
    corpo = detalhe.json()
    assert corpo["estabilidade"] == []
    assert corpo["execucao"]["grupos"] is None
    assert corpo["execucao"]["estado"] == "concluido"


@pytest.mark.slow
def test_com_par_de_grupos_a_conclusao_e_gravada_feature_a_feature(
    cliente: TestClient, com_banco: str
) -> None:
    """**O PAR CAUSAL que impede o teste de cima de passar por vacuidade.**

    Sozinho, aquele passaria numa implementação que **nunca** gravasse estabilidade nenhuma: a
    asserção *"a tabela está vazia"* é satisfeita por ausência. Este é a outra metade — mesma
    rota, mesma matriz, com par de grupos declarado —, e aqui a tabela tem de deixar de estar
    vazia, com uma linha por feature.
    """
    aceito = _dispara(cliente, group_a="ATLL", group_b="HAM")
    assert _espera_terminar(cliente, aceito["identificador"]) == "concluido"
    _espera_no_disco(cliente, aceito["identificador"])

    detalhe = cliente.get(f"/multiversos/historico/{aceito['identificador']}")

    assert detalhe.status_code == 200, detalhe.text
    corpo = detalhe.json()
    assert corpo["execucao"]["grupos"] == ["ATLL", "HAM"]
    assert len(corpo["estabilidade"]) == 3, (
        "a matriz do fixture tem três features, e a conclusão tem uma linha por feature: "
        f"vieram {len(corpo['estabilidade'])}"
    )
    assert corpo["execucao"]["n_features"] == 3, (
        "`n_features` na rota de detalhe tem de contar a tabela que ela mesma serve; "
        f"disse {corpo['execucao']['n_features']} com {len(corpo['estabilidade'])} linhas"
    )
    # `n_cells` viaja com a fração ou a fração viaja sem população.
    assert all(m["n_cells"] == aceito["total"] for m in corpo["estabilidade"]), (
        f"alguma medida foi gravada com população errada: {corpo['estabilidade']}"
    )


@pytest.mark.slow
def test_um_cancelamento_fica_gravado_com_ate_onde_chegou(
    cliente: TestClient, com_banco: str
) -> None:
    """`cancelado` **é um fim**, e o disco conta-o em vez de o perder.

    Uma varredura que contasse `falhou` para medir fragilidade da ferramenta somaria cada clique no
    botão de parar; e uma execução cancelada que não fosse gravada desapareceria do histórico como
    se nunca tivesse corrido.
    """
    aceito = _dispara(cliente)
    cancelamento = cliente.post(f"/multiversos/{aceito['identificador']}/cancelar")
    estado = _espera_terminar(cliente, aceito["identificador"])
    gravado = _espera_no_disco(cliente, aceito["identificador"])

    # **O CÓDIGO DO CANCELAMENTO É AFIRMADO, e a primeira versão não o lia.**
    #
    # A grade de `_dispara` tem UMA célula: ela termina antes de o `POST` chegar, e a rota devolve
    # **409 «já terminou»**. Medido 3 de 3. Sem esta linha, as duas asserções abaixo eram
    # `'concluido' == 'concluido'` e `1 <= 1` — e uma planta que fizesse `cancela_multiverso`
    # **nunca levantar a bandeira** deixava a suíte inteira verde.
    #
    # O que fica aqui é o que este caminho consegue medir com honestidade: **o disco concorda com
    # a memória, qualquer que tenha sido o desfecho**. O estado `cancelado` propriamente dito é
    # medido pelo teste seguinte, que não corre uma corrida.
    assert cancelamento.status_code in {202, 409}, cancelamento.text
    assert estado in {"concluido", "cancelado"}, estado
    if cancelamento.status_code == 409:
        assert estado == "concluido", (
            "a rota disse «já terminou» e o estado não é `concluido`: os dois discordam"
        )
    assert gravado["estado"] == estado, (
        f"a memória diz {estado!r} e o disco diz {gravado['estado']!r}: o gancho de fecho não "
        "correu, ou correu com o estado errado"
    )
    assert gravado["n_feitas"] == gravado["total"] or estado == "cancelado"


@pytest.mark.slow
def test_o_estado_cancelado_atravessa_o_gancho_ate_o_disco(
    cliente: TestClient, com_banco: str
) -> None:
    """**`cancelado` chega ao disco, e isto é medido SEM corrida.**

    O teste acima dispara e cancela, e numa grade de uma célula a execução ganha — logo o estado
    `cancelado` **nunca era exercitado de ponta a ponta**. Aqui o desfecho é construído: alimenta-se
    o gancho de fecho com a leitura terminal que a thread produziria, e afirma-se que a rota do
    histórico o serve.

    **É a costura, e não o repositório.** Que `fecha_multiverso` aceite `cancelado` já é medido em
    `test_multiverso_em_disco.py`; o que só aqui se mede é a **tradução** de `ProgressoDaExecucao`
    para o que o banco guarda — e ela vive em `_fecha_no_disco`, que nenhum dos dois lados vê.
    """
    from refino.api.progresso import ProgressoDaExecucao
    from refino.api.rotas import _fecha_no_disco

    aceito = _dispara(cliente)
    _espera_terminar(cliente, aceito["identificador"])

    _fecha_no_disco(
        ProgressoDaExecucao(
            identificador=aceito["identificador"],
            estado="cancelado",
            feito=3,
            total=aceito["total"],
        )
    )

    detalhe = cliente.get(f"/multiversos/historico/{aceito['identificador']}")
    assert detalhe.status_code == 200, detalhe.text
    gravado = detalhe.json()["execucao"]
    assert gravado["estado"] == "cancelado", (
        f"o gancho traduziu `cancelado` para {gravado['estado']!r}: a tradução perdeu o estado"
    )
    assert gravado["n_feitas"] == 3, (
        "`n_feitas` não atravessou: a distância até `total` é o único dado que diz até onde a "
        "execução chegou"
    )


# ---------------------------------------------------------------------------
# SEM BANCO: a execução corre, e o disparo DIZ que não gravou
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sem_banco_a_grade_corre_e_o_disparo_declara_que_nao_gravou(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**Melhor esforço, e nunca em silêncio — item 58.**

    O banco é dependência nova de uma funcionalidade que sempre correu sem ele. Reprovar o disparo
    trocaria uma perda de conveniência por uma perda de capacidade; gravar nada sem dizer é *"a
    métrica parou de medir"*, e o profissional fecharia o programa a contar com um histórico
    que nunca existiu.
    """
    monkeypatch.delenv(VARIAVEL_DA_URL, raising=False)
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)

    aceito = _dispara(cliente)

    assert aceito["persistido"] is False
    assert aceito["motivo_nao_gravado"], "não gravou e não disse por quê"
    assert "senha" not in aceito["motivo_nao_gravado"].lower(), (
        "a mensagem chega à tela; ela nomeia a configuração, nunca o valor"
    )
    assert _espera_terminar(cliente, aceito["identificador"]) == "concluido", (
        "a grade tem de correr na mesma: a persistência é conveniência, não capacidade"
    )


def test_sem_banco_o_historico_e_503_e_nao_lista_vazia(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A assimetria com o disparo, e ela é deliberada.**

    No disparo a gravação é melhor esforço porque a alternativa seria recusar a grade inteira. Aqui
    o banco é a **única** fonte possível, e devolver lista vazia diria *"nunca correu nada"* sobre
    um histórico que existe e não pode ser lido — vazio por ausência e vazio por falha pedem ações
    opostas.
    """
    monkeypatch.delenv(VARIAVEL_DA_URL, raising=False)
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)

    resposta = cliente.get("/multiversos/historico")

    assert resposta.status_code == 503, resposta.text
    assert resposta.json()["detail"]


# ---------------------------------------------------------------------------
# As bordas das duas rotas
# ---------------------------------------------------------------------------


def test_identificador_desconhecido_no_historico_e_404_com_a_causa_provavel(
    cliente: TestClient, com_banco: str
) -> None:
    """`404` que **nomeia** a causa mais provável, em vez de só negar."""
    resposta = cliente.get("/multiversos/historico/nunca-existiu")

    assert resposta.status_code == 404
    assert "motivo_nao_gravado" in resposta.json()["detail"], (
        "o 404 tem de apontar a causa provável: a execução pode ter sido disparada sem banco"
    )


@pytest.mark.parametrize("limite", [0, -1])
def test_limite_que_nao_devolve_linha_nenhuma_e_recusado(
    cliente: TestClient, com_banco: str, limite: int
) -> None:
    """`limite=0` é indistinguível de *"o histórico está vazio"* para quem lê a resposta."""
    resposta = cliente.get("/multiversos/historico", params={"limite": limite})

    assert resposta.status_code == 422, resposta.text


def test_o_historico_vazio_e_lista_vazia_com_o_limite(cliente: TestClient, com_banco: str) -> None:
    """Banco em pé e nada gravado: `200` com lista vazia. **Não é `404` e não é `503`.**"""
    resposta = cliente.get("/multiversos/historico")

    assert resposta.status_code == 200
    assert resposta.json() == {"execucoes": [], "limite": 50}


@pytest.mark.slow
def test_a_conclusao_que_nao_pode_ser_gravada_nao_evapora_em_silencio(
    cliente: TestClient, com_banco: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """**O `False` de `fecha_multiverso` é LIDO, e a primeira versão descartava-o.**

    `_avisa_que_terminou` só relata quando o gancho **levanta**; aqui ele não levanta — devolve
    `False`, que significa *"a linha de abertura já não existe"*. Reproduzido pela revisão
    adversarial de 28/08/2026: com a linha podada enquanto a grade corria, a conclusão de uma
    grade inteira evaporava **sem sinal nenhum**, e o desfecho era indistinguível de sucesso.

    **O cenário é construído e não esperado:** apaga-se a linha pelo repositório — que é o que
    `esquece_alem_de` faz — e chama-se o gancho com a leitura terminal que a thread produziria.

    **`stderr` e não exceção**, e as duas metades são deliberadas: a execução terminou e o desfecho
    dela está correto na memória; o que falhou foi guardá-lo. Relançar numa thread de fundo mataria
    a thread **depois** de o estado estar certo, sem mudar nada.
    """
    from sqlalchemy import create_engine

    from refino.api.progresso import ProgressoDaExecucao
    from refino.api.rotas import _fecha_no_disco
    from refino.persistence.repositories import MultiversoRepository

    aceito = _dispara(cliente)
    _espera_terminar(cliente, aceito["identificador"])
    _espera_no_disco(cliente, aceito["identificador"])

    # A linha desaparece — exactamente o que a poda faz quando o teto aperta.
    motor = create_engine(com_banco)
    try:
        assert MultiversoRepository(motor).esquece_alem_de(0), "âncora: a poda não levou nada"
    finally:
        motor.dispose()
    capsys.readouterr()

    _fecha_no_disco(
        ProgressoDaExecucao(
            identificador=aceito["identificador"],
            estado="concluido",
            feito=aceito["total"],
            total=aceito["total"],
        )
    )

    erro = capsys.readouterr().err
    assert aceito["identificador"] in erro, (
        f"a gravação falhou e nada foi dito: stderr={erro!r}. Um `False` que ninguém lê é "
        "indistinguível de sucesso"
    )
    assert "ja nao existia no banco" in erro, erro

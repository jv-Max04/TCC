"""A parada das QUATRO ROTAS LONGAS.

## O que muda, e o número é o argumento

`POST /matrizes/corrigir` com `method=serrf` custa 209,9 s no `pos_hilic`. Antes da parada, quem
clicasse por engano esperava os 209,9 s ou fechava o programa.

**Cancelar é diferente de DESISTIR**, e a distinção é o assunto desta suíte:

| | O que faz | O que o trabalho faz |
|---|---|---|
| desistir | encerra a espera do cliente | **continua a correr** |
| cancelar | levanta a bandeira no registro | **para**, e o terminal é `cancelado` |

`rotas.py:623` continua verdadeiro — *"Python não interrompe thread de fora"*. O que mudou é que a
thread passa a **perguntar**.

## Escopo negativo

**Nada aqui mede latência**, e nada aqui corre um SERRF real: os pontos de checagem e os intervalos
que eles fecham são medidos em `tests/test_cancelamento.py` e registrados em
`domain/cancellation.py` com a população. O que se verifica é que a bandeira **chega ao corretor**,
que o estado terminal é `cancelado` e não `falhou`, e que a reclamação de um trabalho cancelado
devolve `410` e não uma mensagem de falha.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a decisão é exercitada em `tests/test_qt_guard.py`, com planta
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from refino.api.app import build_app  # noqa: E402
from refino.api.progresso import RegistroDeProgresso  # noqa: E402
from refino.api.rotas import TRABALHOS, Marca, _executa_trabalho  # noqa: E402
from refino.domain.cancellation import (  # noqa: E402
    CancelCheck,
    ExecutionCancelledError,
)


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


# ---------------------------------------------------------------------------
# O trabalhador — o par causal que separa CANCELADO de FALHOU
# ---------------------------------------------------------------------------


def test_a_bandeira_chega_ao_trabalho_e_responde_as_duas_coisas() -> None:
    """**A bandeira não é `None`, e ela muda de resposta quando alguém pede.**

    Sem esta, um `_executa_trabalho` que passasse `lambda: False` fixo satisfaria toda a suíte
    abaixo: as rotas continuariam a receber uma bandeira, e ela nunca diria sim.
    """
    vistas: list[bool] = []
    chave = TRABALHOS.abre(1, identificador="bandeira")

    def trabalho(marca: Marca, cancelado: CancelCheck) -> object:
        vistas.append(cancelado())
        TRABALHOS.pede_cancelamento("bandeira")
        vistas.append(cancelado())
        marca()
        return "pronto"

    _executa_trabalho(chave, ("unica",), trabalho)

    assert vistas == [False, True], (
        f"a bandeira não acompanhou o pedido: {vistas}. Se for `[False, False]`, o que chegou ao "
        "trabalho não consulta o registro"
    )


def test_cancelar_nao_e_falhar_e_a_etapa_e_gravada() -> None:
    """**O par causal.** Duas execuções idênticas menos na exceção levantada.

    Se `ExecutionCancelledError` fosse apanhada pelo `except Exception` largo, os dois braços
    dariam `falhou` com `codigo_http=500` — *"defeito da ferramenta"* sobre um botão que o
    profissional carregou de propósito. A asserção é que os estados **divergem**.
    """
    registro = TRABALHOS

    def cancela(marca: Marca, cancelado: CancelCheck) -> object:
        marca()
        raise ExecutionCancelledError("serrf: despacho de feature")

    def falha(marca: Marca, cancelado: CancelCheck) -> object:
        marca()
        raise ValueError("boom")

    cancelado_id = registro.abre(2, identificador="c-trab")
    _executa_trabalho(cancelado_id, ("correcao", "escrita"), cancela)
    falhado_id = registro.abre(2, identificador="f-trab")
    _executa_trabalho(falhado_id, ("correcao", "escrita"), falha)

    leitura_c = registro.le(cancelado_id)
    leitura_f = registro.le(falhado_id)
    assert leitura_c is not None and leitura_f is not None

    assert leitura_c.estado != leitura_f.estado, (
        "cancelar e falhar produziram o mesmo estado: o `except` largo absorveu o cancelamento"
    )
    assert leitura_c.estado == "cancelado"
    assert leitura_f.estado == "falhou"
    # **A etapa é gravada**, porque *onde parou* decide se vale redisparar: parar na escrita depois
    # de 209,9 s de SERRF e parar na correção são a mesma palavra e custos opostos.
    assert leitura_c.celula_que_falhou == "escrita"
    assert leitura_c.erro is None, "um cancelamento gravou mensagem de erro"
    assert leitura_c.codigo_http is None, "um cancelamento foi classificado como falha HTTP"


def test_o_trabalho_que_nao_e_cancelado_conclui() -> None:
    """**A metade negativa do par**, e sem ela o resto é vácuo.

    Um `raise_if_cancelled` que levantasse sempre, ou um `cancelado` que devolvesse sempre `True`,
    passaria em todos os testes acima.
    """
    chave = TRABALHOS.abre(1, identificador="normal")

    def trabalho(marca: Marca, cancelado: CancelCheck) -> object:
        marca()
        return "resultado"

    _executa_trabalho(chave, ("unica",), trabalho)

    leitura = TRABALHOS.le(chave)
    assert leitura is not None
    assert leitura.estado == "concluido"
    assert leitura.resultado == "resultado"


# ---------------------------------------------------------------------------
# A ROTA — três desfechos, a mesma partição da rota do multiverso
# ---------------------------------------------------------------------------


def test_cancelar_identificador_desconhecido_da_404(cliente: TestClient) -> None:
    resposta = cliente.post("/trabalhos/nao-existe/cancelar")

    assert resposta.status_code == 404
    assert "nunca existiu" in resposta.json()["detail"]


def test_cancelar_trabalho_ja_terminado_da_409_e_nomeia_o_estado(cliente: TestClient) -> None:
    chave = TRABALHOS.abre(1, identificador="t-terminado")
    TRABALHOS.conclui(chave, resultado="x")

    resposta = cliente.post(f"/trabalhos/{chave}/cancelar")

    assert resposta.status_code == 409
    assert "concluido" in resposta.json()["detail"]


def test_cancelar_trabalho_vivo_da_202(cliente: TestClient) -> None:
    """O `202` devolve o progresso de DEPOIS do pedido — o par que a tela usa para «a parar…»."""
    chave = TRABALHOS.abre(3, identificador="t-vivo")
    TRABALHOS.avanca(chave, 1, 3)

    resposta = cliente.post(f"/trabalhos/{chave}/cancelar")
    corpo = resposta.json()

    assert resposta.status_code == 202
    assert corpo["cancelamento_pedido"] is True
    assert corpo["estado"] == "executando", "a rota gravou o terminal em vez de pedir a parada"
    assert corpo["terminou"] is False


def test_a_particao_e_a_mesma_das_duas_rotas_de_cancelamento(cliente: TestClient) -> None:
    """**Simetria deliberada e não coincidência.**

    As duas rotas fazem a mesma coisa sobre registros diferentes. Dois desenhos de partição para a
    mesma pergunta obrigariam o cliente a aprender dois — e um deles envelheceria.
    """
    from refino.api.rotas import REGISTRO

    do_multiverso = REGISTRO.abre(1, identificador="m-simetria")
    REGISTRO.conclui(do_multiverso)
    do_trabalho = TRABALHOS.abre(1, identificador="t-simetria")
    TRABALHOS.conclui(do_trabalho, resultado="x")

    assert (
        cliente.post("/multiversos/nao-existe/cancelar").status_code
        == cliente.post("/trabalhos/nao-existe/cancelar").status_code
        == 404
    )
    assert (
        cliente.post(f"/multiversos/{do_multiverso}/cancelar").status_code
        == cliente.post(f"/trabalhos/{do_trabalho}/cancelar").status_code
        == 409
    )


# ---------------------------------------------------------------------------
# A RECLAMAÇÃO de um trabalho cancelado — o `410`
# ---------------------------------------------------------------------------


def test_reclamar_trabalho_cancelado_da_410_e_nao_mente_sobre_falha(cliente: TestClient) -> None:
    """**O guarda de ORDEM, e o defeito que ele impede é uma mensagem falsa.**

    Um trabalho cancelado tem `erro is None`. Sem o ramo do `410` **antes** do ramo de falha, o
    servidor diria *"o trabalho falhou na etapa «escrita»: None"* a quem carregou no botão de
    parar — sintaxe correta, conteúdo falso, e nada a acusar.

    **`410` e não `409`:** `409` significa *"do seu dado"* nesta API, e um trabalho que a pessoa
    mandou parar não é dado ruim.
    """
    chave = TRABALHOS.abre(2, identificador="t-cancelado")
    TRABALHOS.avanca(chave, 1, 2)
    TRABALHOS.cancela(chave, etapa="escrita")

    resposta = cliente.get(f"/trabalhos/{chave}")
    detalhe = resposta.json()["detail"]

    assert resposta.status_code == 410
    assert "CANCELADO a seu pedido" in detalhe
    assert "escrita" in detalhe
    assert "falhou" not in detalhe, "a mensagem de falha foi servida a um trabalho cancelado"


def test_o_cliente_classifica_o_410_como_acao_do_usuario() -> None:
    """A causa é `usuario` e não `dado`: mandar conferir o arquivo seria conselho falso.

    É a única causa em que a ação certa é **nenhuma**, e o vocabulário fechado de três valores
    continua a servir porque `usuario` já cobre *"a ação partiu de você"*.
    """
    from refino.interface.cliente import CAUSA_DE_FALHA, DESFECHOS, mensagem_por_chave

    assert 410 in DESFECHOS
    assert CAUSA_DE_FALHA["410"] == "usuario"
    mensagem = mensagem_por_chave("410")
    assert "cancelou" in mensagem
    assert "arquivo" in mensagem, "não tranquiliza sobre o dado, que é a dúvida imediata"


# **Cada teste usa um identificador PRÓPRIO, e não há limpeza entre eles — declarado.**
#
# `TRABALHOS` é global de módulo e por processo: é o mesmo registro que a aplicação usa. Limpá-lo
# entre testes exigiria um `esquece` por identificador ou um duplo, e as duas coisas mediriam outro
# registro que não o de produção. Identificadores distintos (`t-vivo`,
# `t-cancelado`, `c-trab`…) resolvem o mesmo problema sem trocar o referente.


# ---------------------------------------------------------------------------
# O PAINEL 3 — o controle de parada, e ele não existia
# ---------------------------------------------------------------------------
#
# **A guarda de Qt e o `qtbot` são obrigatórios, e a primeira versão desta seção não os tinha.**
# Construir um `QWidget` sem `QApplication` aborta o processo — `pytest` saiu com **127** e os nove
# testes acima apareceram como pontos verdes antes do aborto. É a saída indistinguível de sucesso
# na forma mais literal: a contagem de `passed` nunca chegou a ser impressa.


@pytest.fixture
def painel(qtbot: Any) -> Any:
    from refino.desktop.janela.trabalhos import PainelDeTrabalhos

    alvo = PainelDeTrabalhos()
    qtbot.addWidget(alvo)
    return alvo


def test_o_painel_recusa_em_voz_alta_quando_nao_ha_trabalho_selecionado(painel: Any) -> None:
    """Um slot mudo já produziu *"clicar nao muda nada"*."""
    from refino.desktop.janela.trabalhos import SEM_TRABALHO_A_PARAR

    vistos: list[str] = []
    painel.cancelar.connect(vistos.append)

    painel.pede_cancelamento()

    assert vistos == [], "emitiu sem trabalho selecionado"
    assert painel.rotulo_da_lista.text() == SEM_TRABALHO_A_PARAR


def test_o_seletor_acompanha_a_lista_e_preserva_a_escolha(painel: Any) -> None:
    """Repovoar do zero faria a escolha saltar para o primeiro item enquanto a pessoa lê.

    O botão de parar passaria a apontar para outro trabalho sem nada mudar na tela — que é a
    classe de defeito que este projeto chama de *saída indistinguível de sucesso*.
    """
    from refino.desktop.janela.trabalhos import NOMES_DOS_TRABALHOS

    painel.mostra({"corrigir": "id-1", "processar": "id-2"})
    # O item mostra o nome legível e guarda a chave como dado: escolhe-se pela chave.
    seletor = painel.seletor_de_trabalho
    seletor.setCurrentIndex(seletor.findData("processar"))

    painel.mostra({"corrigir": "id-1", "processar": "id-2", "exportar": "id-3"})

    assert seletor.currentData() == "processar"
    assert seletor.currentText() == NOMES_DOS_TRABALHOS["processar"]
    assert seletor.count() == 3


def test_o_painel_emite_a_chave_e_nunca_o_identificador(painel: Any) -> None:
    """Quem mapeia chave para identificador é a janela. Duas cópias do mapa divergem."""
    painel.mostra({"corrigir": "abc123"})
    vistos: list[str] = []
    painel.cancelar.connect(vistos.append)

    painel.pede_cancelamento()

    assert vistos == ["corrigir"], f"o painel emitiu {vistos}, e não a chave"


def test_o_410_no_painel_diz_parado_e_nunca_falhou(painel: Any) -> None:
    """**O ramo que uma implementação genérica erraria.**

    Sem ele, o `410` cairia no `else` e o painel escreveria *"falhou: o trabalho foi CANCELADO a
    seu pedido"* — uma frase que contradiz a mensagem que ela própria cita.
    """
    from refino.desktop.janela.trabalhos import ROTULOS_DOS_ESTADOS

    rotulo = painel.mostra_estado("corrigir", 410, {"detail": "o trabalho foi CANCELADO"})

    # O rótulo é o do estado `cancelado`, com maiúscula.
    assert rotulo == ROTULOS_DOS_ESTADOS["cancelado"] == "Parado a seu pedido"
    assert "falhou" not in rotulo.lower()


def test_o_painel_tem_os_quatro_desfechos_do_pedido_de_parada(painel: Any) -> None:
    """Quatro desfechos, e os quatro dizem coisas distintas.

    `409` é o que uma implementação apressada trataria como erro — e ele significa que o trabalho
    terminou bem, com a ação de reclamar o resultado.
    """
    rotulos = [
        painel.mostra_cancelamento("k", 202, {}),
        painel.mostra_cancelamento("k", 409, {}),
        painel.mostra_cancelamento("k", 404, {}),
        painel.mostra_cancelamento("k", 500, {"detail": "boom"}),
    ]

    from refino.desktop.janela.trabalhos import PARANDO

    assert len(set(rotulos)) == 4, f"dois desfechos colapsaram: {rotulos}"
    # Os rótulos abrem a linha e sobem a primeira letra.
    assert rotulos[0].startswith(PARANDO)
    assert rotulos[1].startswith("Já tinha terminado")
    assert "falhou" not in rotulos[1].lower(), "um 409 foi mostrado como falha"


# ---------------------------------------------------------------------------
# OS DOIS DEFEITOS DE CLASSE QUEBRA que a revisão adversarial achou
# ---------------------------------------------------------------------------


def test_redisparar_depois_de_cancelar_cria_trabalho_novo() -> None:
    """**A mensagem do `410` PROMETE isto, e sem esta cláusula ela mentia.**

    Ela diz, verbatim: *"Dispare outra vez se ainda o quiser — será um trabalho novo"*, e
    `interface/cliente.py` repete-o. `abre_ou_reclama` tratava qualquer entrada viva na mesma
    chave como *"já corre"*, **inclusive uma em estado terminal `cancelado`** — então o redisparo
    devolvia o mesmo identificador cancelado, sem thread nova, e o profissional que cancelasse por
    engano ficava preso no `410` para sempre.

    **O par causal está nas duas metades**: cancelado dá identificador novo, concluído dá o mesmo.
    Sem a segunda, uma implementação que ignorasse a chave por completo passaria.
    """
    registro = RegistroDeProgresso()

    cancelado, reclamou_c = registro.abre_ou_reclama(3, chave="mesma-entrada")
    assert reclamou_c is False, "âncora: o primeiro disparo não podia ser reclamação"
    registro.cancela(cancelado)
    depois_de_cancelar, reclamou_depois = registro.abre_ou_reclama(3, chave="mesma-entrada")

    assert reclamou_depois is False, (
        "o redisparo reclamou o trabalho cancelado; a mensagem do 410 promete um trabalho novo"
    )
    assert depois_de_cancelar != cancelado

    # **A outra metade: CONCLUÍDO continua a reclamar.** É o que R32 existe para dar — quem perdeu
    # o identificador reencontra o trabalho em vez de pagar o cálculo outra vez.
    concluido, _ = registro.abre_ou_reclama(3, chave="outra-entrada")
    registro.conclui(concluido, resultado="bytes")
    de_novo, reclamou_conc = registro.abre_ou_reclama(3, chave="outra-entrada")

    assert reclamou_conc is True, "um trabalho concluído deixou de ser reclamável"
    assert de_novo == concluido


def test_falhou_continua_a_ser_reclamavel_e_a_assimetria_e_deliberada() -> None:
    """**`falhou` NÃO entra na cláusula de `cancelado`.**

    Um trabalho que falhou tem causa a reportar, e reclamá-lo entrega a etapa e a mensagem;
    redisparar a mesma entrada daria a mesma falha, que é o que R32 existe para não pagar duas
    vezes. Sem este teste, alguém "simplificaria" a cláusula para *"todo terminal conta como
    ausente"* e a assimetria morreria em silêncio.
    """
    registro = RegistroDeProgresso()

    falhado, _ = registro.abre_ou_reclama(2, chave="que-falha")
    registro.falha(falhado, "correcao", "ValueError: x", codigo_http=500)
    de_novo, reclamou = registro.abre_ou_reclama(2, chave="que-falha")

    assert reclamou is True
    assert de_novo == falhado


def test_o_par_de_grupos_entra_na_chave_do_trabalho() -> None:
    """**Sem isto, pedir outro contraste servia o Q² do contraste anterior.**

    `group_a`/`group_b` mudam o resultado — o `PLSDAResult` entra no `QualityReport`, que vai para
    a resposta e para o `metrics_json` no banco. Fora da chave, dois pedidos com pares
    **diferentes** sobre a mesma matriz e a mesma spec produziam a mesma chave, e
    `abre_ou_reclama` devolvia o primeiro. Nada acusava: o corpo trazia `plsda_groups` do contraste
    antigo, que só quem o lê nota.

    **A planta cai sobre dado em que as duas coisas diferem**: os dois dicionários são
    idênticos menos no par, e a asserção é que as chaves **divergem**. Com o par fora da chave, os
    dois hashes seriam iguais e o teste reprovaria.
    """
    from pathlib import Path

    from refino.api.rotas import _chave_do_trabalho
    from refino.io.readers import load_matrix

    fixtures = Path(__file__).resolve().parent.parent / "data" / "fixtures"
    matriz = load_matrix(
        fixtures / "tiny_clean.csv", fixtures / "tiny_clean_sample_list.csv", "tiny"
    )
    base = {"correction": "none", "estudo": "e"}
    um = _chave_do_trabalho("execucoes", matriz, {**base, "group_a": "ATLL", "group_b": "NoHTLV"})
    outro = _chave_do_trabalho(
        "execucoes", matriz, {**base, "group_a": "Asymptomatic", "group_b": "ATLL"}
    )
    sem_par = _chave_do_trabalho("execucoes", matriz, base)

    assert um != outro, (
        "dois contrastes diferentes produzem a MESMA chave: o Q2 do primeiro seria servido ao "
        "segundo, com `plsda_groups` a nomear o contraste antigo"
    )
    assert um != sem_par and outro != sem_par


# ---------------------------------------------------------------------------
# A bandeira ATRAVESSA o estimador leave-one-QC-out até o corretor
# ---------------------------------------------------------------------------
#
# **Achado no uso, não por portão.** Mandar parar uma medição de corretor mostrava na tela *"a
# parada leva até cerca de um segundo"*, e o trabalho continuava — era preciso fechar o
# aplicativo. A causa: `leave_one_qc_out_rsd` não tinha parâmetro `cancelled`, então chamava
# `fit_transform` com o padrão `None`, e `raise_if_cancelled(None, ...)` é no-op por desenho.
#
# **Os três pontos de checagem do SERRF existiam e estavam inertes** neste caminho: mecanismo
# construído, revisado, e nunca exercitado nesta rota. O desfecho de um cancelamento que não pega é
# indistinguível de um trabalho lento, que é o silêncio que ninguém investiga.


def test_d379_o_estimador_leva_a_bandeira_ate_o_corretor() -> None:
    """A bandeira chega ao `fit_transform` de cada QC retido, e o par causal separa os dois estados.

    ## Por que o duplo, e não o SERRF real

    O que se mede aqui é a **travessia do argumento**, não o comportamento do SERRF — e o SERRF
    real custa minutos por ajuste, o que faria este teste inutilizável no portão de commit. O duplo
    regista o que recebeu, e a asserção passa a ser sobre um valor em vez de sobre um tempo.

    ## As TRÊS metades, e a do meio é a que faltava

    | O que se afirma | O que quebraria sem ela |
    |---|---|
    | sem bandeira, o corretor recebe `None` | o padrão deixaria de valer para quem não a passa |
    | com bandeira «não», o corretor **recebe**-a | era isto que faltava: o argumento não chegava |
    | com bandeira «sim», levanta ANTES do fim | atender só no fim não é cancelamento |

    A segunda é a que o defeito tinha: sem ela, um teste que só afirmasse *"levanta quando cancelo"*
    passaria com a checagem de fora do laço — e foi ela que existia, na rota, e não bastou.
    """
    import numpy as np

    from refino.domain.cancellation import ExecutionCancelledError
    from refino.metrics.crossval import leave_one_qc_out_rsd

    class _CorretorQueRegista:
        """Devolve a matriz como veio e guarda o que recebeu em `cancelled`."""

        name = "duplo"

        def __init__(self) -> None:
            self.recebidos: list[Any] = []

        def fit_transform(
            self,
            matrix: Any,
            *,
            batches: Any = None,
            seed: int = 42,
            target_features: Any = None,
            cancelled: Any = None,
        ) -> Any:
            from refino.domain.correction import CorrectionResult

            self.recebidos.append(cancelled)
            return CorrectionResult(matrix=matrix)

    matriz = _matriz_com_qcs()
    n_qc = int(np.count_nonzero(matriz.qc_mask))
    assert n_qc >= 2, f"âncora: a matriz do teste tem {n_qc} QC(s), e o laço não teria iterações"

    # (a) sem bandeira: o corretor recebe `None` em todas as iterações.
    sem = _CorretorQueRegista()
    leave_one_qc_out_rsd(matriz, sem)
    assert len(sem.recebidos) == n_qc, (
        f"o estimador correu {len(sem.recebidos)} ajustes para {n_qc} QCs — a população mudou e o "
        f"resto do teste mede outra coisa"
    )
    assert all(c is None for c in sem.recebidos), (
        "sem bandeira o corretor recebeu outra coisa que `None`, e o padrão deixou de ser "
        "compatível com quem não passa bandeira nenhuma"
    )

    # (b) com bandeira que diz «não»: corre igual, e o corretor RECEBE a bandeira.
    #     É esta metade que o defeito não tinha — o argumento não chegava.
    def nao_cancelar() -> bool:
        return False

    com = _CorretorQueRegista()
    leave_one_qc_out_rsd(matriz, com, cancelled=nao_cancelar)
    assert len(com.recebidos) == n_qc, "a bandeira que diz «não» interrompeu o estimador"
    assert all(c is nao_cancelar for c in com.recebidos), (
        f"a bandeira NÃO chegou ao corretor: {com.recebidos}. Os três pontos de checagem do SERRF "
        f"ficam inertes, e cancelar durante um corretor espera os n_qc ajustes dele"
    )

    # (c) com bandeira que diz «sim»: levanta, e ANTES de fazer os ajustes todos.
    parcial = _CorretorQueRegista()
    chamadas = {"n": 0}

    def cancelar_no_segundo() -> bool:
        chamadas["n"] += 1
        return chamadas["n"] > 2

    with pytest.raises(ExecutionCancelledError):
        leave_one_qc_out_rsd(matriz, parcial, cancelled=cancelar_no_segundo)

    assert len(parcial.recebidos) < n_qc, (
        f"o cancelamento só foi atendido depois dos {n_qc} ajustes todos — é a granularidade que o "
        f"defeito tinha, com a bandeira a chegar e a checagem a ficar fora do laço"
    )
    assert parcial.recebidos, (
        "âncora: nenhum ajuste correu antes do cancelamento, e a asserção acima passaria por o "
        "estimador ter levantado antes de começar — que é outro estado"
    )


def _matriz_com_qcs() -> Any:
    """Uma matriz pequena com QCs suficientes para o estimador iterar.

    **Construída à mão e não lida do laboratório**: o que se mede aqui é a travessia de um
    argumento, não uma contagem observada nos arquivos reais. Fabricar dado no lugar de dado real
    é proibido quando o número medido vem dele — aqui não vem número nenhum do dado.
    """
    import numpy as np

    from refino.domain.matrix import RefinoMatrix
    from refino.domain.metadata import SampleMetadata

    n_qc, n_amostras, n_features = 4, 3, 5
    valores = np.arange(1, (n_qc + n_amostras) * n_features + 1, dtype=np.float64).reshape(
        n_qc + n_amostras, n_features
    )
    amostras = tuple(
        SampleMetadata(
            name=f"QC{i + 1}" if i < n_qc else f"S{i - n_qc + 1}",
            injection_order=i + 1,
            sample_type="qc" if i < n_qc else "sample",
            group="QC" if i < n_qc else "A",
            batch="A",
        )
        for i in range(n_qc + n_amostras)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"1.0_{100 + j}m/z" for j in range(n_features)),
        samples=amostras,
        mode="pos_hilic",
    )


# ---------------------------------------------------------------------------
# A chave do trabalho digere o METADADO da amostra, e faltar era colisão
# ---------------------------------------------------------------------------


def test_d381_reetiquetar_uma_amostra_muda_a_chave_do_trabalho() -> None:
    """A colisão que `_chave_do_trabalho` deixava aberta, e ela era ALCANÇÁVEL.

    ## O caso, e por que nada o acusava

    O digest antigo levava o **nome** de cada amostra e mais nada dela. `sample_type` vem do rótulo
    `LABEL` do `measurements` e **não entra em `values`** — então duas matrizes com os mesmos
    números, os mesmos nomes de feature e os mesmos nomes de coluna, diferindo só numa amostra
    reetiquetada de `HAM` para `QC`, produziam a **mesma** chave.

    E a consequência não é sutil: `qc_mask` é a régua do sistema inteiro — decide o treino do SERRF
    e o eixo do LOESS. O segundo disparo receberia a correção do primeiro, com a tela a mostrar o
    nome certo. A própria docstring da função já nomeava o risco enquanto deixava a porta aberta.

    ## As QUATRO plantas, e cada uma isola um campo

    | Planta | O que ela muda no cálculo |
    |---|---|
    | `sample_type` | `qc_mask` e `sample_mask` — o treino do corretor |
    | `injection_order` reescalado | o `x` do LOESS e o que `_lotes` ordena |
    | `group` | o contraste do PLS-DA em `/execucoes` |
    | `batch` | os lotes quando `batches` não vem explícito |

    **A reescala de `injection_order` é a mais insidiosa e por isso está aqui:** `1,2,3` para
    `10,20,30` mantém a **ordem das linhas**, logo mantém o hash dos valores e o dos nomes. Nada
    além deste campo a vê.

    **A âncora é a igualdade:** duas matrizes idênticas têm de dar a mesma chave. Sem ela, um
    digest que misturasse o relógio passaria por todas as plantas e destruiria a deduplicação que a
    função existe para fazer.
    """
    from dataclasses import replace

    from refino.api.rotas import _chave_do_trabalho

    base = _matriz_com_qcs()
    campos = {"method": "serrf", "seed": 42}

    assert _chave_do_trabalho("corrigir", base, campos) == _chave_do_trabalho(
        "corrigir", base, campos
    ), "âncora: a mesma matriz dá chaves diferentes — a deduplicação de R32 deixou de funcionar"

    def com_amostra_trocada(**mudanca: object) -> object:
        amostras = list(base.samples)
        amostras[0] = replace(amostras[0], **mudanca)  # type: ignore[arg-type]
        return replace(base, samples=tuple(amostras))

    plantas = {
        "sample_type": com_amostra_trocada(sample_type="sample"),
        "group": com_amostra_trocada(group="OUTRO"),
        "batch": com_amostra_trocada(batch="B"),
        "injection_order": replace(
            base,
            samples=tuple(replace(a, injection_order=a.injection_order * 10) for a in base.samples),
        ),
    }

    original = _chave_do_trabalho("corrigir", base, campos)
    for campo, plantada in plantas.items():
        assert _chave_do_trabalho("corrigir", plantada, campos) != original, (  # type: ignore[arg-type]
            f"trocar `{campo}` numa amostra NÃO mudou a chave do trabalho: o segundo disparo "
            f"receberia o resultado do primeiro, com a tela a mostrar o nome certo"
        )

    # E os valores continuam a contar — a planta que sempre contou, para o teste não medir só os
    # campos novos.
    import numpy as np

    valores = np.array(base.values, dtype=np.float64, copy=True)
    valores[0, 0] += 1.0
    assert _chave_do_trabalho("corrigir", replace(base, values=valores), campos) != original

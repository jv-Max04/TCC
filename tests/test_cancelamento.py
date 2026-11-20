"""A parada cooperativa, ponta a ponta.

## O que esta suíte tem de provar, e por que não bastam «os dois desfechos»

O mecanismo novo classifica em mais de dois, e a pergunta é explícita: *"quantos
desfechos este mecanismo pode reportar, e eu vi todos?"*. São **nove**, e estão enumerados na
docstring de cada bloco:

| Camada | Desfechos |
|---|---|
| `raise_if_cancelled` | sem quem perguntar; pergunta e a resposta é não; pergunta e é sim |
| executor | cancelado antes do prefixo; cancelado entre células; **nunca cancelado** |
| corretor `serrf` | cancelado no despacho de feature |
| `executa_registrando` | vira `cancelado`; e uma falha comum continua a virar `falhou` |
| rota | `404`, `409` de terminal, `202` aceito, `202` repetido, `409` no resultado |

## A PLANTA TEM DE CAIR SOBRE DADO EM QUE AS DUAS COISAS DIFEREM

O par que mais importa é o último da quarta linha, e ele é a razão de esta suíte existir: uma
implementação que apanhasse `ExecutionCancelledError` no `except Exception` largo produziria
**exactamente** o desfecho que se quer ver — a execução para, o estado é terminal, o cliente deixa
de consultar — e só o **rótulo** estaria errado. `test_cancelar_nao_e_falhar` é o par causal: duas
execuções idênticas que diferem só na exceção levantada, e a asserção é que os estados **divergem**.

E o mesmo vale para o executor: `test_sem_bandeira_a_grade_inteira_corre` é a metade do par que
prova que a planta **não** está sempre a cair — sem ela, uma `raise_if_cancelled` que levantasse
sempre passaria em tudo o que fica acima.

## Escopo negativo

Nada aqui mede **latência**. Os números de `domain/cancellation.py` — 0,19 s por feature, ~0,75 s de
pior caso — foram medidos sobre o `pos_hilic` inteiro, e reproduzi-los num teste de unidade custaria
minutos por execução. O que se verifica aqui é que a checagem **existe no ponto declarado** e que a
exceção **atravessa**; o número vive no registro, com a população ao lado.

E nada aqui exercita `n_jobs > 1`. A garantia declarada naquele caminho é mais fraca — o `joblib`
usa processos e a bandeira do pai não os alcança —, e um teste que a afirmasse sobre o caminho
serial estaria a medir o corretor errado.

> **O parágrafo de cima já não valia, e fica com a nota ao
> lado.** `test_o_ramo_paralelo_do_executor_tem_ponto_de_parada` corre o executor com `n_jobs=2`, e
> o bloco do vizinho, no fim, corre os quatro agressores em dois núcleos. A latência continua fora:
> os números não estão num teste.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.api.progresso import (
    TERMINAIS,
    ProgressoDaExecucao,
    RegistroDeProgresso,
    executa_registrando,
)
from refino.domain.cancellation import ExecutionCancelledError, raise_if_cancelled
from refino.io.readers import load_matrix
from refino.multiverse.executor import run_multiverse
from refino.multiverse.grid import GridDefinition

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"


@pytest.fixture(scope="module")
def matriz() -> Any:
    """`tiny_drift` LIDO DO DISCO, e a escolha evita um defeito conhecido desta máquina.

    Os outros testes rendem o CSV em memória e escrevem-no com `Path.write_text`, que no Windows
    traduz `\\n` para `\\r\\n` sobre um texto que já vem com CRLF — e produz `\\r\\r\\n`. Aqui o
    arquivo já está em disco com o terminador certo, então o caminho de ingestão é o mesmo e o
    defeito de escrita não entra. Ver a correção separada em `tests/factories.py`.
    """
    return load_matrix(FIXTURES / "tiny_drift.csv", FIXTURES / "tiny_drift_sample_list.csv", "tiny")


def _grade_barata(n_celulas: int = 3) -> GridDefinition:
    """Uma grade pequena e sem `serrf`, variando o eixo que **não descarta feature**.

    `corrections=("none",)` é deliberado: o eixo caro é esse, e esta suíte conta paradas e não mede
    tempo. Uma grade com `serrf` faria cada teste custar minutos sem afirmar nada a mais.

    **O eixo que varia é `scalings`, e a primeira versão usava `rsd_cutoffs` — medido e trocado.**
    A `tiny_drift` tem deriva de 10% por injeção por construção, então um corte de RSD a 0,20
    descarta **as cinco** features e `process_matrix` levanta `EmptyMatrixError` antes de o
    cancelamento ter o que dizer. O escalonamento é invariante em contagem: as cinco atravessam
    sempre, e a grade mede o que ela existe para medir. Na escala de um fixture, o
    teste reprovou por um motivo que não era o dele.
    """
    escalonamentos: tuple[str, ...] = ("none", "auto", "pareto", "range")
    return GridDefinition(
        corrections=("none",),
        batch_options=(1,),
        rsd_cutoffs=(None,),
        missing_thresholds=(0.8,),
        imputations=("median",),
        sample_norms=("none",),
        transformations=("none",),
        scalings=escalonamentos[:n_celulas],  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# `raise_if_cancelled` — os três desfechos, começando pelo que dispara por AUSÊNCIA
# ---------------------------------------------------------------------------


def test_sem_quem_perguntar_nao_levanta() -> None:
    """`None` é o padrão de toda assinatura, e é o caminho de produção da CLI e dos testes.

    **Dispara por ausência de condição**, que é o ramo que se exercita primeiro: não há o
    que plantar nele, e é por isso que é o mais fácil de ficar inerte. Se a guarda de `None`
    sumisse, todo chamador sem cancelamento morreria com `TypeError` — no caminho normal.
    """
    raise_if_cancelled(None, "sítio irrelevante")


def test_pergunta_respondida_com_nao_nao_levanta() -> None:
    """Há quem pergunte, e a resposta é «continue». É o estado normal de uma execução viva."""
    raise_if_cancelled(lambda: False, "sítio irrelevante")


def test_pergunta_respondida_com_sim_levanta_e_preserva_o_sitio() -> None:
    """O terceiro desfecho, e a asserção sobre `where` não é decoração.

    Com cinco pontos de checagem, saber **qual** disparou é o que separa medir a latência do
    mecanismo de a supor. A asserção é sobre o **atributo** e não sobre a mensagem: teste que lê
    prosa quebra quando a prosa muda.
    """
    with pytest.raises(ExecutionCancelledError) as capturado:
        raise_if_cancelled(lambda: True, "serrf: despacho de feature")

    assert capturado.value.where == "serrf: despacho de feature"


# ---------------------------------------------------------------------------
# O executor — e a metade do par que prova que a planta não cai sempre
# ---------------------------------------------------------------------------


def test_sem_bandeira_a_grade_inteira_corre(matriz: Any) -> None:
    """**A metade negativa do par causal, e sem ela o resto desta suíte é vácuo.**

    Uma `raise_if_cancelled` que levantasse sempre — bandeira invertida, `check()` ignorado —
    passaria em todos os testes de cancelamento acima e abaixo. Este é o único que reprova nesse
    caso, e é ele que torna os outros significativos.
    """
    grade = _grade_barata()
    saida = run_multiverse(matriz, grade, cancelled=lambda: False)

    assert len(saida.cells) == grade.size()


def test_cancelado_antes_do_primeiro_prefixo_nao_paga_correcao_nenhuma(matriz: Any) -> None:
    """A checagem que fica ANTES de `corretor.fit_transform`, e é a que compra mais tempo.

    No `serrf` aquela chamada custa 1.076,6 s sobre o `pos_hilic`; cancelar depois dela seria
    esperar por ela inteira. A asserção é sobre o **sítio** que levantou, porque é o que distingue
    esta checagem da de baixo.
    """
    with pytest.raises(ExecutionCancelledError) as capturado:
        run_multiverse(matriz, _grade_barata(), cancelled=lambda: True)

    assert capturado.value.where.startswith("executor: antes da correcao do prefixo")


def test_cancelado_no_meio_para_entre_celulas_e_o_progresso_conta_o_que_correu(
    matriz: Any,
) -> None:
    """A checagem por célula, e a planta cai sobre dado em que as duas coisas DIFEREM.

    A bandeira sobe **depois** da primeira célula, então a execução tem de correr uma e parar na
    seguinte: uma implementação que só checasse entre prefixos correria a grade toda, e uma que
    checasse antes de tudo não correria nenhuma. A asserção fixa o meio, que é o único ponto em que
    as três implementações divergem.
    """
    vistos: list[int] = []
    grade = _grade_barata()

    def bandeira() -> bool:
        return len(vistos) >= 1

    with pytest.raises(ExecutionCancelledError) as capturado:
        run_multiverse(
            matriz,
            grade,
            progress=lambda feito, _total: vistos.append(feito),
            cancelled=bandeira,
        )

    assert capturado.value.where == "executor: entre celulas"
    assert vistos == [1], "parou noutro ponto que não a segunda célula"
    assert len(vistos) < grade.size(), "correu a grade inteira"


# ---------------------------------------------------------------------------
# O corretor `serrf` — o ponto que decide o pior caso
# ---------------------------------------------------------------------------


def test_o_serrf_para_no_despacho_de_feature(matriz: Any) -> None:
    """O ponto de 0,19 s, e o que se afirma aqui é que ele EXISTE e a exceção atravessa.

    **Atravessar é a parte não óbvia:** a checagem vive num gerador que o `joblib.Parallel`
    consome, e *"a exceção é engolida pelo pool"* era a hipótese que tornaria o desenho inútil.
    Ela propaga — medido antes de o código ser escrito, e afirmado aqui para que continue a ser
    verdade.

    A bandeira sobe já levantada porque o que se mede é o sítio, não a contagem: quantas features
    correm antes é propriedade do custo, e essa é medida sobre matriz real, não sobre `tiny_drift`.
    """
    from refino.correction.serrf import SerrfCorrector

    with pytest.raises(ExecutionCancelledError) as capturado:
        SerrfCorrector(n_jobs=1).fit_transform(matriz, cancelled=lambda: True)

    # O primeiro ponto que o `serrf` alcança é o laço do passo 1; o despacho vem depois. Afirmar
    # qualquer um dos dois seria frágil, então a asserção é que parou DENTRO do serrf.
    assert capturado.value.where.startswith("serrf: ")


def test_o_serrf_sem_bandeira_corrige_a_matriz_inteira(matriz: Any) -> None:
    """A metade negativa do par, ao nível do corretor.

    Sem ela, um `raise_if_cancelled` mal escrito dentro do `serrf` passaria no teste de cima e
    tornaria o corretor inutilizável no caminho normal — que é o caminho de toda a suíte.
    """
    from refino.correction.serrf import SerrfCorrector

    resultado = SerrfCorrector(n_jobs=1).fit_transform(matriz, cancelled=lambda: False)

    assert resultado.matrix.n_features == matriz.n_features


# ---------------------------------------------------------------------------
# `executa_registrando` — o par causal que separa CANCELADO de FALHOU
# ---------------------------------------------------------------------------


class _MatrizFalsa:
    """Basta para as chamadas que o executor injetado faz."""


def test_cancelar_nao_e_falhar() -> None:
    """**O par causal mais importante desta suíte**.

    Duas execuções idênticas em tudo menos na exceção que o executor levanta. Se
    `ExecutionCancelledError` fosse apanhada pelo `except Exception` largo, os dois braços dariam
    `falhou` e **todo o resto desta suíte continuaria verde**: a execução para, o estado é
    terminal, o cliente deixa de consultar. Só o rótulo estaria errado — e é o rótulo que separa
    «a ferramenta é frágil» de «o usuário carregou no botão».

    A asserção é que os estados **divergem**, e não que um deles é `cancelado`: é a forma que
    reprova o colapso, e não só a que confirma o rótulo.
    """
    registro = RegistroDeProgresso()
    grade = _grade_barata(2)

    def executor_que_cancela(*_a: object, **_k: object) -> object:
        raise ExecutionCancelledError("executor: entre celulas")

    def executor_que_falha(*_a: object, **_k: object) -> object:
        raise ValueError("matriz degenerada")

    cancelado = registro.abre(grade.size(), identificador="c")
    executa_registrando(
        _MatrizFalsa(),  # type: ignore[arg-type]
        grade,
        registro,
        cancelado,
        executor=executor_que_cancela,
    )
    falhado = registro.abre(grade.size(), identificador="f")
    executa_registrando(
        _MatrizFalsa(),  # type: ignore[arg-type]
        grade,
        registro,
        falhado,
        executor=executor_que_falha,
    )

    leitura_cancelada = registro.le(cancelado)
    leitura_falhada = registro.le(falhado)
    assert leitura_cancelada is not None
    assert leitura_falhada is not None

    assert leitura_cancelada.estado != leitura_falhada.estado, (
        "cancelar e falhar produziram o mesmo estado: o `except` largo absorveu o cancelamento"
    )
    assert leitura_cancelada.estado == "cancelado"
    assert leitura_falhada.estado == "falhou"


def test_cancelado_e_terminal_mas_nao_preenche_celula_nem_erro() -> None:
    """Os campos que existem **porque** houve falha ficam vazios num cancelamento.

    Preenchê-los afirmaria defeito onde houve obediência, e uma varredura que contasse
    `celula_que_falhou` para medir fragilidade somaria cada clique no botão de parar.
    """
    registro = RegistroDeProgresso()
    chave = registro.abre(10, identificador="x")
    registro.avanca(chave, 4, 10)
    registro.cancela(chave)

    leitura = registro.le(chave)
    assert leitura is not None
    assert leitura.estado in TERMINAIS
    assert leitura.terminou is True
    assert leitura.celula_que_falhou is None
    assert leitura.erro is None
    assert leitura.feito == 4, "`feito` foi forçado ao total e perdeu quanto chegou a correr"


def test_pede_cancelamento_distingue_desconhecido_de_ja_terminado() -> None:
    """A assinatura devolve a leitura e não um `bool`, e é este teste que diz por quê.

    `None` e «existe, já terminou» são `404` e `409` — respostas diferentes e ações opostas do
    cliente. Um `bool` colapsaria as duas.
    """
    registro = RegistroDeProgresso()

    assert registro.pede_cancelamento("nunca-registrado") is None

    chave = registro.abre(3, identificador="t")
    registro.conclui(chave)
    anterior = registro.pede_cancelamento(chave)

    assert isinstance(anterior, ProgressoDaExecucao)
    assert anterior.terminou is True


def test_quer_cancelar_e_falso_para_identificador_esquecido() -> None:
    """A pergunta do núcleo não pode explodir sobre um identificador podado.

    Levantar aqui transformaria uma poda de manutenção — que corre por contagem, sem olhar quem
    está a correr — num defeito de cálculo, no meio de uma grade paga.
    """
    registro = RegistroDeProgresso()

    assert registro.quer_cancelar("nunca-registrado") is False


# ---------------------------------------------------------------------------
# A rota — os cinco desfechos
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def test_cancelar_identificador_desconhecido_da_404(cliente: TestClient) -> None:
    resposta = cliente.post("/multiversos/nao-existe/cancelar")

    assert resposta.status_code == 404
    assert "nao-existe" in resposta.json()["detail"]


def test_cancelar_execucao_ja_terminada_da_409_e_nomeia_o_estado(cliente: TestClient) -> None:
    """`409` cobre os três terminais, e a **mensagem** é que diz qual era.

    Um código por estado terminal daria ao cliente três ramos para a mesma ação — nenhuma.
    """
    from refino.api.rotas import REGISTRO

    chave = REGISTRO.abre(5, identificador="ja-terminou")
    REGISTRO.conclui(chave)

    resposta = cliente.post(f"/multiversos/{chave}/cancelar")

    assert resposta.status_code == 409
    assert "concluido" in resposta.json()["detail"]


def test_cancelar_execucao_viva_da_202_com_a_bandeira_levantada(cliente: TestClient) -> None:
    """O `202` devolve o progresso **de depois**, e é o par que a tela usa para dizer «a parar…».

    `estado` ainda `executando` e `cancelamento_pedido` já `True`: sem esse par a interface veria
    `executando` antes e depois, e o botão pareceria morto — o relato exacto.
    """
    from refino.api.rotas import REGISTRO

    chave = REGISTRO.abre(5, identificador="viva")
    REGISTRO.avanca(chave, 1, 5)

    resposta = cliente.post(f"/multiversos/{chave}/cancelar")
    corpo = resposta.json()

    assert resposta.status_code == 202
    assert corpo["cancelamento_pedido"] is True
    assert corpo["estado"] == "executando", "a rota gravou o terminal em vez de pedir a parada"
    assert corpo["terminou"] is False


def test_cancelar_duas_vezes_nao_e_erro(cliente: TestClient) -> None:
    """Quem carrega duas vezes num botão que ainda não mudou de estado não recebe um erro.

    Um `409` na segunda faria a interface parecer partida exactamente quando a parada demora — que
    é quando a pessoa carrega outra vez.
    """
    from refino.api.rotas import REGISTRO

    chave = REGISTRO.abre(5, identificador="duas-vezes")
    REGISTRO.avanca(chave, 1, 5)

    primeira = cliente.post(f"/multiversos/{chave}/cancelar")
    segunda = cliente.post(f"/multiversos/{chave}/cancelar")

    assert (primeira.status_code, segunda.status_code) == (202, 202)
    assert segunda.json()["cancelamento_pedido"] is True


def test_resultado_de_execucao_cancelada_nao_mente_sobre_o_par_de_grupos(
    cliente: TestClient,
) -> None:
    """**O guarda de ORDEM, e o defeito que ele impede é silencioso.**

    Uma execução cancelada tem `resultado is None` por construção. Sem o ramo de `cancelado`
    **antes** da leitura do resultado, ela cairia no `409` de baixo e o servidor diria *«o disparo
    não declarou `group_a` e `group_b`»* a quem declarou os dois e carregou no botão de parar —
    mensagem correta na sintaxe e falsa no conteúdo, sem nada a acusar.

    A asserção é sobre as duas metades: que a mensagem certa está lá, e que a **errada não está**.
    """
    from refino.api.rotas import REGISTRO

    chave = REGISTRO.abre(9, identificador="cancelada-sem-conclusao")
    REGISTRO.avanca(chave, 4, 9)
    REGISTRO.cancela(chave)

    resposta = cliente.get(f"/multiversos/{chave}/resultado")
    detalhe = resposta.json()["detail"]

    assert resposta.status_code == 409
    assert "CANCELADA" in detalhe
    assert "4 de 9" in detalhe, "a mensagem não diz quanto chegou a correr"
    assert "group_a" not in detalhe, "serviu a mensagem do par de grupos a uma execução cancelada"


# ---------------------------------------------------------------------------
# OS PONTOS QUE EXISTIAM E NENHUM TESTE ALCANÇAVA — a revisão adversarial
# ---------------------------------------------------------------------------
#
# **A revisão provou, com planta, que três checagens nunca levantavam em teste nenhum.** A causa é
# a mesma nas três e é instrutiva: uma bandeira `lambda: True` faz o **primeiro** ponto do caminho
# ganhar sempre a corrida, e os pontos seguintes ficam inalcançáveis — com a suíte verde e a
# docstring a afirmar que eles estão cobertos.
#
# > **Bandeira sempre-verdadeira mede o primeiro ponto e mais nenhum.** Para alcançar o k-ésimo é
# > preciso uma bandeira que só diga sim DEPOIS de k-1 consultas, e é isso que `_apos` faz.


def _apos(n: int) -> Any:
    """Uma bandeira que responde `False` nas primeiras `n` consultas e `True` daí em diante.

    **É o que torna alcançável um ponto de checagem que não é o primeiro.** Sem ela,
    `cancelled=lambda: True` faz o laço do passo 1 do SERRF levantar sempre, e o despacho de
    feature — o ponto de 0,196 s que fecha o pior caso de 1.076,6 s — nunca corre.
    """
    contador = {"n": 0}

    def bandeira() -> bool:
        contador["n"] += 1
        return contador["n"] > n

    return bandeira


def test_o_despacho_de_feature_do_serrf_levanta_de_facto(matriz: Any) -> None:
    """**O ponto que decide o pior caso, e ele não tinha teste.**

    A revisão adversarial mediu: substituir `_indices_ate_cancelar` por um gerador **sem**
    checagem deixava a árvore verde, e o sítio foi chamado 2.184 vezes na suíte sem levantar uma
    única. O teste anterior afirmava `where.startswith("serrf: ")`, que o laço do passo 1 satisfaz.

    Aqui a bandeira só sobe depois de o passo 1 e o içamento terem passado, então o único ponto
    que resta é o despacho — e a asserção é de **igualdade**, não de prefixo.
    """
    from refino.correction.serrf import SerrfCorrector

    with pytest.raises(ExecutionCancelledError) as capturado:
        SerrfCorrector(n_jobs=1).fit_transform(matriz, cancelled=_apos(6))

    assert capturado.value.where == "serrf: despacho de feature", (
        f"levantou em {capturado.value.where!r}: o despacho continua sem ser alcançado"
    )


def test_o_icamento_das_correlacoes_do_serrf_levanta_de_facto(matriz: Any) -> None:
    """O segundo ponto que a revisão achou inerte. Mesmo mecanismo, `n` diferente.

    O laço do passo 1 corre uma vez por feature e por lote; sobre a `tiny_drift` são cinco
    consultas antes do içamento. A bandeira sobe na sexta.
    """
    from refino.correction.serrf import SerrfCorrector

    with pytest.raises(ExecutionCancelledError) as capturado:
        SerrfCorrector(n_jobs=1).fit_transform(matriz, cancelled=_apos(5))

    assert capturado.value.where == "serrf: antes do icamento das correlacoes"


def test_o_ramo_paralelo_do_executor_tem_ponto_de_parada(matriz: Any) -> None:
    """**A checagem de `_run_leaves` no caminho `n_jobs > 1`, que nunca correu.**

    `Parallel(...)` executa **tudo** antes do primeiro `yield`, então sem o gerador de despacho a
    checagem entre células só corria depois das folhas todas do prefixo. A revisão mediu que o
    sítio tinha zero chamadas.

    **`_apos(2)` e não `lambda: True`, e a contagem é o conteúdo**: antes do despacho correm dois
    pontos — «antes da correcao do prefixo» e «none: entrada» —, e uma bandeira sempre-verdadeira
    levantaria no primeiro. Foi assim que a primeira versão deste teste reprovou com
    `none: entrada`, que é o mesmo defeito de leitura que ele existe para fechar.
    """
    with pytest.raises(ExecutionCancelledError) as capturado:
        run_multiverse(matriz, _grade_barata(3), n_jobs=2, cancelled=_apos(2))

    assert capturado.value.where == "executor: despacho de celula", capturado.value.where


def test_o_plsda_para_no_laco_das_permutacoes() -> None:
    """**O bloco de 2,8 a 4,1 s que a interface prometia parar em menos de um segundo.**

    A revisão mediu que `plsda(...)` não recebia a bandeira: entre a checagem anterior e a
    seguinte corria o PLS-DA inteiro. Uma permutação custa ~15 ms, então este ponto fecha o
    intervalo em duas ordens de grandeza.
    """
    import numpy as np

    from refino.statistics.plsda import _permutation_pvalue

    x = np.random.default_rng(0).normal(size=(14, 5))
    y = np.array([0] * 7 + [1] * 7)

    with pytest.raises(ExecutionCancelledError) as capturado:
        _permutation_pvalue(
            x, y, 0.5, n_components=2, cv_folds=7, n=200, seed=42, cancelled=lambda: True
        )

    assert capturado.value.where == "plsda: laco das permutacoes"


def test_sem_bandeira_o_plsda_devolve_o_p_inteiro() -> None:
    """A metade negativa: sem quem pergunte, as 200 permutações correm e o `p` sai."""
    import numpy as np

    from refino.statistics.plsda import _permutation_pvalue

    x = np.random.default_rng(0).normal(size=(14, 5))
    y = np.array([0] * 7 + [1] * 7)

    p = _permutation_pvalue(x, y, 0.5, n_components=2, cv_folds=7, n=20, seed=42)

    assert 1 / 21 <= p <= 1.0, p


# ---------------------------------------------------------------------------
# A PARADA QUE DERRUBAVA O VIZINHO
# ---------------------------------------------------------------------------
#
# **O executor do `loky` é UM por processo**, e todo `Parallel(backend="loky")` do processo o
# partilha: a API corre cada trabalho numa thread, e a janela deixava o multiverso correr ao lado de
# uma correção. No `joblib` 1.5.3, uma exceção levantada DENTRO do gerador que o `Parallel` consome
# leva-o a `_abort`, e o `loky` chama `abort_everything`, que encerra esse executor partilhado com
# `kill_workers=True`. Todo futuro pendente de OUTRO `Parallel` vira `ShutdownExecutorError`, que o
# `joblib` traduz num `RuntimeError` — fora da família que o recuo para série apanha.
#
# > **Parar um trabalho matava outro que ninguém tinha parado.** O conserto é o gerador PARAR de
# > produzir e o cancelamento subir depois de o `Parallel` devolver (`CancellableDispatch`, em
# > `domain/cancellation.py`): o `Parallel` espera o que ESTE trabalho já despachou — e, com o
# > vizinho, as tarefas dele que estão à frente na fila do executor partilhado.
#
# **O vizinho é um `Parallel` sem cancelamento**, e não um corretor: a vítima é qualquer trabalho
# do processo, e as tarefas de meio segundo dão ao teste o controlo do tempo. **Duas âncoras antes
# da asserção sobre o vizinho:** o agressor parou no sítio do despacho, e o vizinho ainda não tinha
# devolvido quando ele parou — senão a sobrevivência dele é vácua pelos dois lados.
#
# **A prova de que o arnês consegue reprovar não depende do produto:** um agressor sintético, que
# levanta dentro do gerador como o desenho antigo fazia, tem de matar o vizinho. **O SERRF, que
# ainda levanta no despacho** porque `correction/serrf.py` é o caminho medido de dois portões de
# procedência, entra no mesmo teste parametrizado com `xfail(strict=True)` sobre a asserção certa:
# é pendência, e não comportamento a defender — quem o consertar vê o `XPASS` reprovar e tira o
# marcador com decisão, em vez de ver um teste reprovar por ter consertado.

_TAREFAS_DO_VIZINHO = 10
_SEGUNDOS_POR_TAREFA = 0.5


class _VizinhoDerrubadoError(AssertionError):
    """A asserção sobre o vizinho, com tipo próprio para o `xfail` do SERRF não absorver outra.

    `xfail(raises=...)` só aceita esta classe: uma âncora que falhasse — o agressor não parou, ou o
    vizinho já tinha terminado — levanta `AssertionError` simples e reprova, em vez de passar por
    pendência conhecida.
    """


def _com_um_vizinho_no_loky(
    agressor: Any,
) -> tuple[ExecutionCancelledError | None, bool, dict[str, Any]]:
    """Corre `agressor` enquanto um vizinho ocupa o executor do `loky`. Devolve os três desfechos.

    O vizinho corre numa thread: `_TAREFAS_DO_VIZINHO` tarefas de `_SEGUNDOS_POR_TAREFA` em dois
    processos, ao menos 2,5 s de trabalho. O agressor só começa depois de o vizinho pedir a
    terceira tarefa ao gerador — nesse instante as duas primeiras já foram submetidas ao executor,
    porque o `joblib` despacha um lote antes de puxar o seguinte.

    **O segundo desfecho é se o vizinho estava em VOO quando o agressor parou**: não ter devolvido
    a saída. Na árvore anterior ao conserto ele já pode ter morrido nesse instante, e a morte não o
    tira do voo — é o desfecho medido. Com tarefas de 0,5 s e dez delas, o agressor que espera os
    lotes dele, atrás das tarefas do vizinho na fila do executor, devolve perto de 1 s depois de
    começar, e o vizinho precisa de ao menos 2,5 s.

    **Dois núcleos dos dois lados, e é condição e não estilo:** com `n_jobs` diferente o `loky`
    SUBSTITUI o executor — `get_reusable_executor` vê os argumentos mudados e chama
    `shutdown(wait=True)` —, e com o vizinho vivo o processo TRAVA: o `shutdown` espera a thread de
    gestão, que está no callback do vizinho a submeter a tarefa seguinte, à espera da trava do
    redimensionamento. Medido, e o impasse não depende de cancelamento nenhum: existe com
    `cancelled=None`, e na árvore anterior ao conserto.

    > **A condição continua a valer AQUI, e o motivo é o vizinho:** ele é um
    > `Parallel` cru, que não pede a vez no pool (`domain/pool.py`), e a vez só protege quem a pede.
    > Entre os despachos do produto que a pedem — o LOESS, a régua, o jackknife e o executor —, o
    > de outro `n_jobs` espera, e não trava. E o `loky` tem um segundo caminho para a trava, sem
    > `shutdown`: quando o `env` dos workers é o mesmo (3 contra 4 num Linux de 4), ele redimensiona
    > o executor à espera das tarefas pendentes, com a mesma trava na mão.
    """
    import threading
    import time
    from collections.abc import Iterator

    from joblib import Parallel, delayed

    pediu_a_terceira = threading.Event()
    desfecho: dict[str, Any] = {}

    def tarefas() -> Iterator[Any]:
        for indice in range(_TAREFAS_DO_VIZINHO):
            if indice == 2:
                pediu_a_terceira.set()
            yield delayed(time.sleep)(_SEGUNDOS_POR_TAREFA)

    def vizinho() -> None:
        try:
            desfecho["saida"] = Parallel(n_jobs=2, backend="loky")(tarefas())
        except Exception as erro:
            desfecho["erro"] = erro

    linha = threading.Thread(target=vizinho, name="vizinho-no-loky")
    linha.start()
    assert pediu_a_terceira.wait(120), "âncora: o vizinho nunca chegou a despachar"
    parada: ExecutionCancelledError | None = None
    try:
        agressor()
    except ExecutionCancelledError as erro:
        parada = erro
    em_voo = "saida" not in desfecho
    linha.join(120)
    assert not linha.is_alive(), "o vizinho ficou pendurado depois da parada do agressor"
    return parada, em_voo, desfecho


def _matriz_com_sete_qcs() -> Any:
    """Sete QCs num lote só, para a régua fora da amostra ter o que reter — a `tiny_drift` tem 2."""
    import numpy as np

    from refino.domain.matrix import RefinoMatrix
    from refino.domain.metadata import SampleMetadata

    n_amostras = 14
    qcs = {0, 2, 4, 6, 8, 10, 13}
    return RefinoMatrix(
        values=np.random.default_rng(3).uniform(1e3, 1e5, size=(n_amostras, 4)),
        feature_names=tuple(f"f{j}" for j in range(4)),
        samples=tuple(
            SampleMetadata(
                name=f"s{i:02d}",
                injection_order=i + 1,
                sample_type="qc" if i in qcs else "sample",
                group="qc" if i in qcs else "caso",
                batch="A",
            )
            for i in range(n_amostras)
        ),
        mode="synthetic",
    )


def _agressor_loess(matriz: Any) -> Any:
    from refino.correction.loess import LoessCorrector

    return lambda: LoessCorrector(n_jobs=2).fit_transform(matriz, seed=42, cancelled=_apos(3))


def _agressor_da_regua(_matriz: Any) -> Any:
    from refino.correction.none import NoneCorrector
    from refino.metrics.crossval import leave_one_qc_out_rsd

    # Três consultas sem parar: a de antes do primeiro ajuste e os despachos de dois QCs.
    return lambda: leave_one_qc_out_rsd(
        _matriz_com_sete_qcs(), NoneCorrector(), n_jobs=2, cancelled=_apos(3)
    )


def _agressor_do_executor(matriz: Any) -> Any:
    # Duas consultas antes do despacho, como no teste do ramo paralelo do executor acima, e a
    # terceira é o despacho da primeira célula: uma sai, a segunda fica.
    return lambda: run_multiverse(matriz, _grade_barata(3), n_jobs=2, cancelled=_apos(3))


def _agressor_serrf(matriz: Any) -> Any:
    from refino.correction.serrf import SerrfCorrector

    # O passo 1 da `tiny_drift` consulta cinco vezes e o içamento uma; a sétima é o despacho da
    # primeira feature.
    return lambda: SerrfCorrector(n_jobs=2).fit_transform(matriz, cancelled=_apos(6))


_PENDENCIA_DO_SERRF = (
    "O despacho do SERRF ainda LEVANTA dentro do gerador, e o loky encerra o executor "
    "partilhado com o vizinho dentro. correction/serrf.py é o caminho medido da dispersão de "
    "serrf_dispersion_python.json e do custo do [CA-3.8], e mudá-lo obriga a remedir os dois no "
    "Windows (~147 min). Na janela, a trava de uma correção por vez com o multiverso dentro "
    "impede que os dois corram juntos; na API, nada impede. XPASS = o conserto chegou ao SERRF: "
    "tire o marcador."
)


@pytest.mark.parametrize(
    ("montador", "sitio"),
    [
        (_agressor_loess, "loess: despacho de feature"),
        (_agressor_da_regua, "leave-one-qc-out: despacho de um QC retido"),
        (_agressor_do_executor, "executor: despacho de celula"),
        pytest.param(
            _agressor_serrf,
            "serrf: despacho de feature",
            marks=pytest.mark.xfail(
                strict=True, raises=_VizinhoDerrubadoError, reason=_PENDENCIA_DO_SERRF
            ),
        ),
    ],
    ids=["loess", "regua-fora-da-amostra", "executor-do-multiverso", "serrf"],
)
def test_d674_parar_um_trabalho_paralelo_nao_derruba_o_vizinho_no_loky(
    matriz: Any, montador: Any, sitio: str
) -> None:
    """Parar o LOESS, a régua ou o executor em vários núcleos deixa o vizinho terminar.

    **As duas âncoras vêm antes da asserção sobre o vizinho:** o agressor tem de ter PARADO, no
    sítio do despacho, e o vizinho tem de estar em voo nesse instante. Sem a primeira, um agressor
    que nunca parasse deixaria o vizinho vivo por vacuidade; sem a segunda, um vizinho que já
    tivesse terminado.
    """
    parada, em_voo, vizinho = _com_um_vizinho_no_loky(montador(matriz))

    assert parada is not None, "âncora: o agressor não parou, e a sobrevivência do vizinho é vácua"
    assert parada.where == sitio, parada.where
    assert em_voo, (
        "âncora: o vizinho já tinha terminado quando o agressor parou, e a sobrevivência dele não "
        "prova nada"
    )
    if "erro" in vizinho:
        raise _VizinhoDerrubadoError(
            f"parar o agressor derrubou o vizinho no loky: {type(vizinho['erro']).__name__}: "
            f"{vizinho['erro']}"
        )
    assert vizinho["saida"] == [None] * _TAREFAS_DO_VIZINHO


def test_d674_o_arnes_ve_o_vizinho_morrer_quando_o_despacho_levanta(matriz: Any) -> None:
    """**A metade do par que reprova, sem depender de um defeito do produto.**

    O agressor é sintético: um gerador consumido por `Parallel(n_jobs=2, backend="loky")` que
    LEVANTA `ExecutionCancelledError` no terceiro item — o desenho que o LOESS, a régua e o
    executor tinham antes, e que o SERRF ainda tem. O vizinho tem de morrer com o executor
    encerrado. Se este teste reprovar, ou o `joblib` deixou de encerrar o executor partilhado, ou
    o arnês deixou de pôr o vizinho em voo — e nos dois casos o teste de cima deixou de poder
    reprovar.
    """
    from joblib import Parallel, delayed

    def agressor() -> None:
        def despacho() -> Any:
            for indice in range(4):
                if indice == 2:
                    raise ExecutionCancelledError("planta: despacho que levanta")
                yield delayed(abs)(indice)

        Parallel(n_jobs=2, backend="loky")(despacho())

    parada, em_voo, vizinho = _com_um_vizinho_no_loky(agressor)

    assert parada is not None and parada.where == "planta: despacho que levanta", parada
    assert em_voo, "âncora: o vizinho já tinha terminado quando a planta levantou"
    assert "erro" in vizinho, (
        "o vizinho sobreviveu a um despacho que levanta: o arnês já não vê o defeito, e o teste de "
        "cima deixou de poder reprovar"
    )
    assert "shutdown" in str(vizinho["erro"]), (
        f"o vizinho morreu de outra coisa: {type(vizinho['erro']).__name__}: {vizinho['erro']}"
    )

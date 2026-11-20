"""Progresso incremental do multiverso.

O que se verifica é a FORMA, e o limiar numérico fica declarado a derivar. *"Sem bloquear a
interface"* não compara contra nada: a grade padrão tem `GridDefinition().size()` especificações,
e o tempo de parede é desconhecido. Número escolhido agora seria limiar sem origem.

## Quatro desfechos, e "dois" é o piso e não o número

O registro classifica em `pendente`, `executando`, `concluido` e `falhou`. Um teste que afirmasse
*"os dois desfechos"* responderia **sim** para um arnês de quatro saídas que só sabe produzir duas.
Os quatro estão exercitados, e o `pendente` primeiro: ele dispara por **ausência** de condição —
nada aconteceu ainda —, não há o que plantar nele, e é por isso que é o mais fácil de ficar inerte
sem ninguém notar.

## Por que a leitura DURANTE a execução precisa de thread

*"Uma leitura feita durante a execução devolve valor intermediário"* não é verificável em série:
num teste sequencial, a execução termina antes de a leitura acontecer, e o que se mediria é
`concluido` — que passaria por "progresso" sem distinguir progresso real de progresso reportado no
fim. A barreira é um `threading.Event` que a célula do meio solta, e não um `sleep`: `sleep`
calibrado torna o teste intermitente na máquina de outra pessoa.

**Escopo negativo.** Estes testes não medem tempo de parede e não afirmam nada sobre ele. O que a
unidade entrega é a forma; o número entra quando houver multiverso medido ponta a ponta.
"""

import json
import re
import threading
from typing import Any, Final

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.api.progresso import ESTADOS, RegistroDeProgresso, executa_registrando
from refino.api.rotas import Marca, corre_as_analises
from refino.cli import UNAVAILABLE_METHODS, Unavailable
from refino.domain.analysis import PADRAO_DA_ANALISE, AnalysisSpec
from refino.domain.cancellation import ExecutionCancelledError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.interface.fluxo import FAMILIAS_DE_ANALISE, etapas_das_analises, familias_pedidas
from refino.multiverse.grid import GridDefinition
from tests.conftest import measurements_path, sample_list_path

MODO = "pos_hilic"


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


@pytest.fixture
def registro() -> RegistroDeProgresso:
    return RegistroDeProgresso()


def _grade(n: int) -> GridDefinition:
    """Uma grade de tamanho controlado, sempre sem `serrf`.

    O tamanho sai dos eixos baratos — `rsd_cutoffs` e `scalings` — porque `corrections` é o eixo
    caro: cada valor novo ali é uma correção inteira, e a grade existe aqui para contar células e
    não para medir custo.
    """
    cortes: tuple[float | None, ...] = (None, 0.20, 0.25, 0.30)
    return GridDefinition(
        corrections=("none",),
        batch_options=(1,),
        rsd_cutoffs=cortes[:n] if n <= 4 else cortes,
        missing_thresholds=(0.8,),
        imputations=("median",),
        sample_norms=("none",),
        transformations=("none",),
        scalings=("none",) if n <= 4 else ("none", "auto", "pareto"),
    )


class _MatrizFalsa:
    """Basta para as chamadas que o executor injetado faz. Não é usada com o executor real."""


# ---------------------------------------------------------------------------
# Os QUATRO estados, começando pelo `pendente`
# ---------------------------------------------------------------------------


def test_pendente_e_o_estado_de_quem_abriu_e_nao_comecou(registro: RegistroDeProgresso) -> None:
    """O ramo que dispara por AUSÊNCIA de condição.

    Não há o que plantar nele: ele é o que se vê quando nada aconteceu ainda. É o desfecho mais
    fácil de ficar inerte, porque nenhum teste de caminho feliz passa por ele.
    """
    chave = registro.abre(12, identificador="fixo")
    leitura = registro.le(chave)

    assert leitura is not None
    assert leitura.estado == "pendente"
    assert (leitura.feito, leitura.total) == (0, 12)
    assert leitura.terminou is False
    assert leitura.fracao == 0.0


def test_executando_concluido_falhou_e_cancelado_sao_os_outros_quatro(
    registro: RegistroDeProgresso,
) -> None:
    """Os quatro restantes, e a asserção é que os **cinco** rótulos são distintos.

    Sem esta, dois estados poderiam colidir e cada teste individual continuaria passando.

    > Esta função dizia «os outros TRÊS» e afirmava quatro rótulos. Quando `cancelado` entrou, a
    > última asserção — `set(ESTADOS) == {*vistos, "pendente"}` —
    > **reprovou**, que é o desfecho correto: ela existe exactamente para que um estado novo não
    > entre no vocabulário sem ser produzido por alguém. O portão fez o trabalho dele.
    """
    vistos = []
    for i, acao in enumerate(("avanca", "conclui", "falha", "cancela")):
        chave = registro.abre(4, identificador=f"e{i}")
        if acao == "avanca":
            registro.avanca(chave, 2, 4)
        elif acao == "conclui":
            registro.conclui(chave)
        elif acao == "falha":
            registro.falha(chave, "abc123", "ValueError: x")
        else:
            registro.cancela(chave)
        leitura = registro.le(chave)
        assert leitura is not None
        vistos.append(leitura.estado)

    assert vistos == ["executando", "concluido", "falhou", "cancelado"]
    assert len({*vistos, "pendente"}) == 5, "dois estados colidiram"
    assert set(ESTADOS) == {*vistos, "pendente"}, "o vocabulário e os estados produzidos divergem"


def test_identificador_desconhecido_devolve_none_e_nao_um_pendente_vazio(
    registro: RegistroDeProgresso,
) -> None:
    """*"Não conheço"* e *"conheço e não começou"* pedem respostas opostas.

    Colapsá-los faria a interface esperar para sempre por algo que nunca foi registrado.
    """
    assert registro.le("nunca-registrado") is None


# ---------------------------------------------------------------------------
# A forma: monotônico, alcança o total, leitura intermediária
# ---------------------------------------------------------------------------


def test_to_12_6_o_progresso_e_monotonico_e_alcanca_o_total(
    registro: RegistroDeProgresso,
) -> None:
    """Monotônico **e** chegando ao total: sem a segunda metade, um contador travado em 1 passaria.

    É a asserção que o critério pede em duas partes, e testar só a primeira é o defeito que a
    própria redação do critério antecipa.
    """
    grade = _grade(4)
    chave = registro.abre(grade.size(), identificador="mono")
    sequencia: list[int] = []

    def executor(
        _m: Any,
        g: GridDefinition,
        *,
        n_jobs: int,
        progress: Any,
        cancelled: Any = None,
        corrector_jobs: int = 1,
    ) -> None:
        for i in range(1, g.size() + 1):
            progress(i, g.size())

    def espia(*_: object) -> None:
        leitura = registro.le(chave)
        assert leitura is not None
        sequencia.append(leitura.feito)

    original = registro.avanca

    def avanca_e_espia(ident: str, feito: int, total: int) -> None:
        original(ident, feito, total)
        espia()

    registro.avanca = avanca_e_espia  # type: ignore[method-assign]
    executa_registrando(_MatrizFalsa(), grade, registro, chave, executor=executor)  # type: ignore[arg-type]

    assert sequencia == sorted(sequencia), f"o progresso recuou: {sequencia}"
    leitura = registro.le(chave)
    assert leitura is not None
    assert leitura.estado == "concluido"
    assert leitura.feito == leitura.total == grade.size(), (
        "o progresso terminou sem alcançar o total: a barra fica parada perto do fim, e "
        "'quase pronto para sempre' é a mesma leitura que contador preso"
    )
    assert leitura.fracao == 1.0


def test_avanco_para_tras_e_absorvido_e_a_barra_nao_recua(registro: RegistroDeProgresso) -> None:
    """`max` com o anterior, e a asserção é sobre o que se lê e não sobre o que se escreveu.

    Se o executor um dia paralelizar as folhas, chamadas fora de ordem deixam de ser hipótese —
    e recuo não é incremento observável.
    """
    chave = registro.abre(10, identificador="recuo")
    registro.avanca(chave, 7, 10)
    registro.avanca(chave, 3, 10)

    leitura = registro.le(chave)
    assert leitura is not None
    assert leitura.feito == 7


# ---------------------------------------------------------------------------
# A etapa em curso — o nome que a faixa «Trabalhos em curso» mostra
# ---------------------------------------------------------------------------


def test_avanca_guarda_a_etapa_anunciada_e_um_aviso_atrasado_nao_a_troca(
    registro: RegistroDeProgresso,
) -> None:
    """**A etapa acompanha o contador**, e as três leituras separam o que pode dar errado.

    `pendente` não anuncia nada; cada aviso troca a etapa; e um aviso ATRASADO — `feito` menor que o
    já lido — não move nem o contador nem a etapa. Sem a terceira, uma leitura podia dizer *«1 de 3
    etapas»* ao lado do nome da etapa anterior: os dois campos de uma leitura viriam de avisos
    diferentes, que é o que o `Lock` desta classe existe para não deixar acontecer.
    """
    chave = registro.abre(3, identificador="etapas")
    leitura = registro.le(chave)
    assert leitura is not None and leitura.etapa_em_curso is None, "pendente anunciou etapa"

    registro.avanca(chave, 0, 3, etapa_em_curso="correcao")
    registro.avanca(chave, 1, 3, etapa_em_curso="pipeline")
    leitura = registro.le(chave)
    assert leitura is not None
    assert (leitura.feito, leitura.etapa_em_curso) == (1, "pipeline")

    registro.avanca(chave, 0, 3, etapa_em_curso="correcao")
    leitura = registro.le(chave)
    assert leitura is not None
    assert (leitura.feito, leitura.etapa_em_curso) == (1, "pipeline"), (
        f"um aviso atrasado trocou a etapa sem mover o contador: {leitura}"
    )

    # **O multiverso não nomeia etapa**, e o aviso sem nome não inventa uma.
    celulas = registro.abre(4, identificador="celulas")
    registro.avanca(celulas, 2, 4)
    leitura = registro.le(celulas)
    assert leitura is not None and leitura.etapa_em_curso is None


def test_a_rota_de_progresso_devolve_como_etapa_corrente_a_ultima_que_a_rota_anunciou(
    cliente: TestClient,
) -> None:
    """`GET /trabalhos/{id}` durante a execução, em DOIS instantes, e a etapa muda entre eles.

    **Dois instantes e não um**, porque a planta tem de cair sobre dado em que as duas coisas
    diferem: uma etapa fixa, ou a primeira da tupla sempre, passaria numa leitura só. O
    trabalho pára em duas barreiras — antes da primeira marca e depois dela —, e a leitura em cada
    uma tem de nomear a etapa que **está a correr**: a primeira da tupla, e depois a segunda.
    """
    from refino.api.rotas import TRABALHOS, _executa_trabalho

    etapas = ("correcao", "pipeline", "escrita do artefato")
    chave = TRABALHOS.abre(len(etapas))
    na_primeira, solta_a_primeira = threading.Event(), threading.Event()
    na_segunda, solta_a_segunda = threading.Event(), threading.Event()

    def trabalho(marca: Marca, cancelado: Any) -> object:
        na_primeira.set()
        solta_a_primeira.wait(10)
        marca()
        na_segunda.set()
        solta_a_segunda.wait(10)
        marca()
        marca()
        return "pronto"

    linha = threading.Thread(target=_executa_trabalho, args=(chave, etapas, trabalho))
    linha.start()
    try:
        assert na_primeira.wait(10), "o trabalho não começou"
        primeira = cliente.get(f"/trabalhos/{chave}")
        solta_a_primeira.set()
        assert na_segunda.wait(10), "o trabalho não passou da primeira marca"
        segunda = cliente.get(f"/trabalhos/{chave}")
    finally:
        solta_a_primeira.set()
        solta_a_segunda.set()
        linha.join(10)
        TRABALHOS.esquece(chave)

    assert (primeira.status_code, segunda.status_code) == (202, 202)
    lidas = [(r.json()["feito"], r.json()["etapa_em_curso"]) for r in (primeira, segunda)]
    assert lidas == [(0, "correcao"), (1, "pipeline")], lidas


def test_to_12_6_uma_leitura_durante_a_execucao_devolve_valor_intermediario(
    registro: RegistroDeProgresso,
) -> None:
    """A asserção que distingue progresso real de progresso reportado no fim.

    **A barreira é um `Event` e não um `sleep`.** `sleep` calibrado passa nesta máquina e fica
    intermitente na de outra pessoa — e teste intermitente é pior que teste ausente, porque
    ensina a ignorar vermelho.
    """
    grade = _grade(4)
    chave = registro.abre(grade.size(), identificador="meio")
    no_meio = threading.Event()
    pode_seguir = threading.Event()

    def executor(
        _m: Any,
        g: GridDefinition,
        *,
        n_jobs: int,
        progress: Any,
        cancelled: Any = None,
        corrector_jobs: int = 1,
    ) -> None:
        progress(1, g.size())
        progress(2, g.size())
        no_meio.set()
        assert pode_seguir.wait(timeout=10), "a leitura intermediária não aconteceu"
        for i in range(3, g.size() + 1):
            progress(i, g.size())

    thread = threading.Thread(
        target=executa_registrando,
        args=(_MatrizFalsa(), grade, registro, chave),
        kwargs={"executor": executor},
    )
    thread.start()
    try:
        assert no_meio.wait(timeout=10), "a execução não chegou ao meio"
        durante = registro.le(chave)
    finally:
        pode_seguir.set()
        thread.join(timeout=20)

    assert durante is not None
    assert durante.estado == "executando", "a leitura no meio não viu a execução em curso"
    assert 0 < durante.feito < grade.size(), (
        f"a leitura no meio devolveu {durante.feito} de {grade.size()}, que não é intermediário"
    )
    depois = registro.le(chave)
    assert depois is not None
    assert depois.estado == "concluido"


def test_grade_de_um_elemento_vai_de_zero_a_um_sem_divisao_por_zero(
    registro: RegistroDeProgresso,
) -> None:
    """A borda da grade de um elemento."""
    grade = _grade(1)
    assert grade.size() == 1

    chave = registro.abre(grade.size(), identificador="unitaria")
    inicio = registro.le(chave)
    assert inicio is not None
    assert inicio.fracao == 0.0

    def executor(
        _m: Any,
        g: GridDefinition,
        *,
        n_jobs: int,
        progress: Any,
        cancelled: Any = None,
        corrector_jobs: int = 1,
    ) -> None:
        progress(1, g.size())

    executa_registrando(_MatrizFalsa(), grade, registro, chave, executor=executor)  # type: ignore[arg-type]

    fim = registro.le(chave)
    assert fim is not None
    assert (fim.feito, fim.total, fim.fracao) == (1, 1, 1.0)


def test_fracao_com_total_zero_e_zero_e_nao_nan(registro: RegistroDeProgresso) -> None:
    """`nan` faria a barra da interface desaparecer em vez de ficar em zero."""
    chave = registro.abre(0, identificador="vazia")
    leitura = registro.le(chave)

    assert leitura is not None
    assert leitura.fracao == 0.0


# ---------------------------------------------------------------------------
# Falha no meio produz estado terminal NOMEADO
# ---------------------------------------------------------------------------


def test_to_12_6_falha_no_meio_nomeia_a_celula_e_nao_deixa_o_contador_preso(
    registro: RegistroDeProgresso,
) -> None:
    """**Um progresso que ficasse em 2 de 4 para sempre é indistinguível de execução lenta.**

    A asserção tem três partes, e a terceira é a que o desenho óbvio esquece: o estado é
    terminal, o erro está lá, e a **célula** está nomeada — porque *"o multiverso falhou"* não diz
    onde reexecutar, e numa grade desse tamanho obriga a refazer tudo.
    """
    grade = _grade(4)
    chave = registro.abre(grade.size(), identificador="quebra")
    especificacoes = list(grade.enumerate())

    def executor(
        _m: Any,
        g: GridDefinition,
        *,
        n_jobs: int,
        progress: Any,
        cancelled: Any = None,
        corrector_jobs: int = 1,
    ) -> None:
        progress(1, g.size())
        progress(2, g.size())
        raise ZeroDivisionError("a terceira célula explodiu")

    executa_registrando(_MatrizFalsa(), grade, registro, chave, executor=executor)  # type: ignore[arg-type]

    leitura = registro.le(chave)
    assert leitura is not None
    assert leitura.estado == "falhou", "o contador ficou preso em vez de virar estado terminal"
    assert leitura.terminou is True, "a interface esperaria para sempre"
    assert leitura.erro is not None
    assert "ZeroDivisionError" in leitura.erro
    assert leitura.celula_que_falhou == especificacoes[2].fingerprint(), (
        "a célula nomeada não é a que falhou. O `progress` dispara DEPOIS de cada célula, então "
        "com dois avisos a que quebrou é a de índice 2 em base zero"
    )


def test_o_par_causal_da_falha_sem_a_planta_conclui(registro: RegistroDeProgresso) -> None:
    """Visto passar com a planta removida.

    Mesma grade, mesmo caminho, executor que **não** levanta — e o desfecho divergindo. Sem esta
    direção, um `executa_registrando` que marcasse `falhou` sempre passaria no teste acima.
    """
    grade = _grade(4)
    chave = registro.abre(grade.size(), identificador="sem-planta")

    def executor(
        _m: Any,
        g: GridDefinition,
        *,
        n_jobs: int,
        progress: Any,
        cancelled: Any = None,
        corrector_jobs: int = 1,
    ) -> None:
        progress(1, g.size())
        progress(2, g.size())

    executa_registrando(_MatrizFalsa(), grade, registro, chave, executor=executor)  # type: ignore[arg-type]

    leitura = registro.le(chave)
    assert leitura is not None
    assert leitura.estado == "concluido", "sem a planta o desfecho tem de divergir"
    assert leitura.celula_que_falhou is None


def test_falha_antes_da_primeira_celula_nao_inventa_uma_celula(
    registro: RegistroDeProgresso,
) -> None:
    """Nomear a célula errada é pior que dizer que não se sabe qual.

    Quem reexecutasse a errada veria sucesso e concluiria que o defeito era intermitente.
    """
    grade = _grade(4)
    chave = registro.abre(grade.size(), identificador="antes")

    def executor(
        _m: Any,
        _g: GridDefinition,
        *,
        n_jobs: int,
        progress: Any,
        cancelled: Any = None,
        corrector_jobs: int = 1,
    ) -> None:
        raise RuntimeError("morreu antes de qualquer célula")

    executa_registrando(_MatrizFalsa(), grade, registro, chave, executor=executor)  # type: ignore[arg-type]

    leitura = registro.le(chave)
    assert leitura is not None
    assert leitura.estado == "falhou"
    assert leitura.celula_que_falhou == next(iter(grade.enumerate())).fingerprint(), (
        "com zero avisos o índice é 0, que É a primeira célula — e ela é de facto a que falhou"
    )


def test_indice_fora_da_grade_diz_que_nao_sabe(registro: RegistroDeProgresso) -> None:
    """O ramo que devolve rótulo explícito em vez de a primeira especificação."""
    from refino.api.progresso import _celula_na_posicao

    grade = _grade(4)

    assert "fora da grade" in _celula_na_posicao(grade, 999)


# ---------------------------------------------------------------------------
# Determinismo da sequência de totais
# ---------------------------------------------------------------------------


def test_duas_execucoes_com_a_mesma_semente_dao_a_mesma_sequencia_de_totais() -> None:
    """Determinismo sobre a **sequência**, e não só sobre o resultado final.

    Duas execuções que chegassem ao mesmo total por caminhos diferentes reportariam progresso
    diferente à interface, e a sequência é o que ela mostra. `run_multiverse` sem semente na
    assinatura tira a fonte única de dúvida: ela vem de `PipelineSpec.seed`.
    """
    grade = _grade(4)
    sequencias: list[list[tuple[int, int]]] = []

    for volta in range(2):
        registro = RegistroDeProgresso()
        chave = registro.abre(grade.size(), identificador=f"det{volta}")
        vistos: list[tuple[int, int]] = []

        def executor(
            _m: Any,
            g: GridDefinition,
            *,
            n_jobs: int,
            progress: Any,
            cancelled: Any = None,
            corrector_jobs: int = 1,
        ) -> None:
            for i, _spec in enumerate(g.enumerate(), start=1):
                progress(i, g.size())

        original = registro.avanca

        def avanca_e_grava(
            ident: str,
            feito: int,
            total: int,
            _v: list[tuple[int, int]] = vistos,
            _orig: Any = original,
        ) -> None:
            # `_orig` e `_v` entram como argumento com padrao e nao por clausura: a clausura
            # capturaria a ULTIMA volta do laco, e as duas execucoes gravariam na mesma lista —
            # o teste de determinismo passaria comparando uma sequencia consigo mesma.
            _orig(ident, feito, total)
            _v.append((feito, total))

        registro.avanca = avanca_e_grava  # type: ignore[method-assign]
        executa_registrando(_MatrizFalsa(), grade, registro, chave, executor=executor)  # type: ignore[arg-type]
        sequencias.append(vistos)

    assert sequencias[0] == sequencias[1], f"as sequências divergiram: {sequencias}"
    assert sequencias[0], "nenhum aviso de progresso foi emitido — o teste passaria por vacuidade"


# ---------------------------------------------------------------------------
# As duas rotas
# ---------------------------------------------------------------------------


def _reais(mode: str = MODO) -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{mode}.csv",
            measurements_path(mode).read_bytes(),
            "text/csv",
        ),
        "sample_list": (f"Sample_List_{mode}.csv", sample_list_path(mode).read_bytes(), "text/csv"),
    }


def test_a_rota_de_leitura_devolve_404_para_identificador_desconhecido(
    cliente: TestClient,
) -> None:
    """404 e não um progresso vazio: corrigir o identificador e esperar são ações opostas."""
    resposta = cliente.get("/multiversos/nao-existe")

    assert resposta.status_code == 404, resposta.text
    assert "nao-existe" in resposta.json()["detail"]


def test_a_grade_do_multiverso_aceita_serrf_e_o_padrao_e_a_grade_completa() -> None:
    """Invertido, e o padrão da rota mudou junto.

    Este teste afirmava `422` para uma grade com `serrf`, pela assimetria de o multiverso
    executar pela porta de trás o que a execução recusava pela da frente. Com as duas portas
    abertas a assimetria desapareceu, e ela era o argumento inteiro.

    **Não passa pelo `POST`, e o motivo é mais forte que custo.** A rota devolve `202` e
    executa numa **thread daemon**: aceitar `serrf` ali dispararia uma grade de 4.320 células
    dominada por ele — ~97% do custo — a moer em segundo plano durante o resto da suíte, sem
    nada que a espere ou a mate. A propriedade é de `_grade_ou_422`, e é lá que se afirma.

    **A segunda asserção é sobre o PADRÃO**, e é a metade que quase ficou de fora: a rota
    tinha `corrections='["none", "loess"]'` escrito à mão, e um literal não teria mudado com a
    liberação — a grade padrão continuaria a excluir o `serrf` com o bloqueio já removido, em
    silêncio. Aqui o conserto foi derivar.
    """
    from refino.api.rotas import PADRAO_CORRECTIONS, _grade_ou_422

    grade = _grade_ou_422(json.dumps(["none", "serrf"]), "[1]", {})
    assert "serrf" in grade.corrections

    assert json.loads(PADRAO_CORRECTIONS) == list(GridDefinition().corrections), (
        "o padrão de `corrections` da rota divergiu de `GridDefinition` — se for literal, ele "
        "não segue a grade e a liberação de um método não o alcança"
    )
    assert "serrf" in json.loads(PADRAO_CORRECTIONS)


def test_a_grade_e_recusada_e_nao_filtrada_em_silencio() -> None:
    """Filtrar executaria uma grade menor que a pedida e devolveria um total que ninguém pediu.

    É o modo de falha insidioso na contagem de células: o cliente recebe um número
    plausível sobre um conjunto diferente do que pediu.

    A instância viva era o `serrf`, e a liberação dele a removeu. O teste passava porque o método
    era bloqueado; com ele liberado, uma grade só de `serrf` é **legítima** — e a versão anterior
    deste teste disparou uma de 1.440 células numa thread daemon ao ser executada, que é o
    desfecho que o `POST` esconde atrás do `202`.

    **A propriedade guardada é *recusar em vez de filtrar*, e ela não é sobre o `serrf`.**
    Exercitada sobre uma planta, com as duas direções: com a entrada bloqueada, `422`; sem
    ela, a grade sai com o método pedido dentro. Sem o segundo braço, uma guarda que recusasse
    toda grade passaria no primeiro.
    """
    from refino.api.rotas import _grade_ou_422

    planta = Unavailable(
        reason="planta de teste: bloqueado por [CA-9.9]",
        implemented=True,
        gate="[CA-9.9]",
        gate_unit="U99",
        corrector="refino.correction.serrf:SerrfCorrector",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(UNAVAILABLE_METHODS, "serrf", planta)
        with pytest.raises(HTTPException) as capturado:
            _grade_ou_422(json.dumps(["serrf"]), "[1]", {})
    assert capturado.value.status_code == 422
    assert "não é selecionável" in str(capturado.value.detail), (
        "a grade só com o método bloqueado virou grade vazia em silêncio"
    )

    sem_planta = _grade_ou_422(json.dumps(["serrf"]), "[1]", {})
    assert sem_planta.corrections == ("serrf",), "o braço de controle não exercitou nada"


def test_corrections_que_nao_e_vetor_e_recusado(cliente: TestClient) -> None:
    """`"none"` sozinho seria iterado como caracteres e produziria rótulos absurdos."""
    resposta = cliente.post(
        "/multiversos", files=_reais(), data={"mode": MODO, "corrections": '"none"'}
    )

    assert resposta.status_code == 422, resposta.text
    detalhe = resposta.json()["detail"]
    assert "`corrections`" in detalhe, (
        "a mensagem tem de NOMEAR o eixo: a rota aceita oito, e sem o nome o cliente adivinha"
    )
    assert "caracteres" in detalhe


def test_rotulo_de_correcao_invalido_e_recusado_na_fronteira(cliente: TestClient) -> None:
    """Rótulo fora do vocabulário cairia no ramo de outra política."""
    resposta = cliente.post(
        "/multiversos", files=_reais(), data={"mode": MODO, "corrections": json.dumps(["loes"])}
    )

    assert resposta.status_code == 422, resposta.text
    assert "`corrections`" in resposta.json()["detail"], "a mensagem não nomeia o eixo"
    assert "loes" in resposta.json()["detail"]


@pytest.mark.integration
def test_o_disparo_devolve_202_e_o_progresso_chega_ao_total(cliente: TestClient) -> None:
    """O caminho de produção inteiro: `run_multiverse` real sobre um modo real.

    **202 e não 200**, porque 200 afirmaria que o trabalho terminou e um cliente que o recebesse
    não teria motivo para consultar o progresso.

    A grade é a mínima — uma correção `none` e um corte —, porque o que se mede aqui é a forma do
    progresso e não o custo do multiverso.
    """
    resposta = cliente.post(
        "/multiversos",
        files=_reais(),
        data={
            "mode": MODO,
            # A grade e MINIMA de proposito, e a escolha vai declarada. Executar a grade padrao
            # da rota aqui mediria o CUSTO do multiverso, e nao a FORMA do progresso.
            "corrections": json.dumps(["none"]),
            "rsd_cutoffs": json.dumps([None, 0.20]),
            "missing_thresholds": json.dumps([0.8]),
            "imputations": json.dumps(["median"]),
            "sample_norms": json.dumps(["none"]),
            "transformations": json.dumps(["none"]),
            "scalings": json.dumps(["none"]),
        },
    )

    assert resposta.status_code == 202, resposta.text
    corpo = resposta.json()
    identificador = corpo["identificador"]
    assert corpo["total"] == 2, (
        f"a grade minima devolveu {corpo['total']} celulas; o teste mede forma e nao custo"
    )

    for _ in range(600):
        leitura = cliente.get(f"/multiversos/{identificador}").json()
        if leitura["terminou"]:
            break
        threading.Event().wait(0.5)

    assert leitura["estado"] == "concluido", leitura
    assert leitura["feito"] == leitura["total"] == corpo["total"]
    assert leitura["fracao"] == 1.0
    assert leitura["celula_que_falhou"] is None


def test_a_grade_do_teste_tem_o_tamanho_que_as_assercoes_supoem() -> None:
    """Guarda contra o teste de forma passar sobre grade degenerada.

    `_grade` constrói eixos à mão, e um eixo escrito errado daria uma grade de tamanho 1 onde o
    teste espera 4 — e as asserções de monotonicidade passariam por vacuidade sobre uma célula só.
    """
    assert _grade(4).size() == 4
    assert _grade(1).size() == 1


def test_esquece_remove_a_execucao_do_registro(registro: RegistroDeProgresso) -> None:
    """Existe para o registro não crescer sem limite, e **não** é chamado ao concluir.

    Apagar no fim faria `concluido` virar 404, e quem lê o progresso tem de poder ler o estado
    terminal ao menos uma vez.
    """
    chave = registro.abre(1, identificador="efemera")
    registro.conclui(chave)
    assert registro.le(chave) is not None, "o estado terminal tem de ser legível ao menos uma vez"

    registro.esquece(chave)
    assert registro.le(chave) is None


# ---------------------------------------------------------------------------
# A CONCLUSÃO do multiverso
# ---------------------------------------------------------------------------
#
# Medido antes de se escrever uma linha: `executa_registrando` chamava o executor sem atribuir o
# retorno, e `feature_stability` — *quais features sobrevivem à grade*, que é a pergunta que o
# trabalho existe para responder — tinha **zero** chamadas em produção.
#
# **A grade destes testes é MÍNIMA de propósito e o modo é o menor.** O que se mede aqui é a forma
# da conclusão e os desfechos da rota, nunca o custo do multiverso. `neg_c18` tem 2.155 features
# contra 5.929 do `pos_hilic`.


def _grade_de_uma_celula() -> GridDefinition:
    """Uma célula, para os testes que precisam de execução REAL e não de contagem."""
    return GridDefinition(
        corrections=("none",),
        batch_options=(1,),
        rsd_cutoffs=(None,),
        missing_thresholds=(0.8,),
        imputations=("median",),
        sample_norms=("none",),
        transformations=("none",),
        scalings=("none",),
    )


def test_u19_ou_nulo_traduz_nan_e_deixa_passar_inf_e_zero() -> None:
    """A tradução de *"não medido"* para JSON, e as **três** direções dela.

    `nan` vira `None` porque `_ausente` o usa para dizer *não medido*, e `NaN` cru não é JSON
    válido. **Zero passa**, e é a metade que importa: zero é *medido e deu zero* — colapsá-lo em
    `None` apagaria a informação que separa ações opostas. E **`inf` passa**,
    de propósito: se algum agregado devolver infinito isso é achado, e mandá-lo para `None` o
    esconderia atrás de *"não medido"*.
    """
    from refino.api.progresso import _ou_nulo

    assert _ou_nulo(float("nan")) is None
    assert _ou_nulo(0.0) == 0.0, "zero é MEDIDO e deu zero; virar `None` apaga a distinção"
    assert _ou_nulo(float("inf")) == float("inf")
    assert _ou_nulo(0.37) == 0.37


def test_u19_a_ordem_e_decrescente_por_fracao_com_o_nome_como_desempate() -> None:
    """A ordem da tabela, sobre frações que **divergem** — e é essa a metade que faltava.

    > **A primeira versão deste teste media a ordem só sobre execução real, e uma planta que a
    > invertesse PASSAVA.** Nos arquivos do laboratório nenhuma feature sobrevive ao FDR num
    > contraste de duas classes — zero em 24 de 24 pares —, então todas as frações valem
    > `0,0` e ascendente e descendente produzem a mesma tabela. A planta caía sobre dado em que as
    > duas coisas que ela separa **não** diferem.

    Aqui as frações são `0,2`, `0,9` e `0,9`, e os nomes estão em ordem inversa da esperada: as duas
    direções — inverter o sinal, tirar o desempate — mudam o desfecho.
    """
    from refino.api.progresso import ordem_de_estabilidade
    from refino.multiverse.stability import FeatureStability

    def medida(nome: str, frac: float) -> FeatureStability:
        return FeatureStability(
            feature=nome,
            frac_significant=frac,
            median_qvalue=0.5,
            median_rank=1.0,
            rank_iqr=0.0,
            sign_consistency=1.0,
            verdict="fragile",
            n_cells=10,
        )

    entrada = [medida("z", 0.2), medida("b", 0.9), medida("a", 0.9)]
    assert len({m.frac_significant for m in entrada}) == 2, (
        "âncora: as frações não divergem, e a asserção de ordem passaria sob qualquer sinal"
    )

    ordenadas = [m.feature for m in sorted(entrada, key=ordem_de_estabilidade)]

    assert ordenadas == ["a", "b", "z"], (
        f"a ordem saiu {ordenadas}: `a` e `b` empatam em 0,9 e vêm antes de `z` em 0,2, e entre "
        "elas decide o nome"
    )


def test_u19_resume_o_multiverso_levanta_em_vez_de_devolver_resumo_vazio() -> None:
    """Saída que não é `MultiverseRun` **levanta**, e não devolve tabela de zero features.

    O único caminho que chega aqui com outra coisa é um `executor` injetado. Devolver resumo vazio
    faria *"a grade não produziu conclusão"* — que seria uma afirmação científica — sair igual a *"o
    arnês devolveu `None`"*.
    """
    from refino.api.progresso import resume_o_multiverso

    with pytest.raises(TypeError) as capturado:
        resume_o_multiverso(None, ("ATLL", "NoHTLV"), "x")
    assert "NoneType" in str(capturado.value), "a mensagem não nomeia o tipo que chegou"


def test_u19_o_par_causal_do_resumo_sem_grupos_conclui_sem_resultado(
    registro: RegistroDeProgresso,
) -> None:
    """As duas direções sobre a MESMA planta, e a planta é o executor que devolve `None`.

    Com `grupos`, `resume_o_multiverso` levanta e o estado é `falhou` com o tipo nomeado. Sem
    `grupos`, o mesmo executor conclui e `resultado` é `None` — que é o caminho de compatibilidade
    que a rota preserva. Sem o segundo braço, um `executa_registrando` que falhasse sempre passaria
    no primeiro.
    """
    grade = _grade(4)

    def executor(
        _m: Any,
        g: GridDefinition,
        *,
        n_jobs: int,
        progress: Any,
        cancelled: Any = None,
        corrector_jobs: int = 1,
    ) -> None:
        for i in range(1, g.size() + 1):
            progress(i, g.size())

    com = registro.abre(grade.size(), identificador="com-grupos")
    executa_registrando(
        _MatrizFalsa(),  # type: ignore[arg-type]
        grade,
        registro,
        com,
        executor=executor,
        grupos=("ATLL", "NoHTLV"),
    )
    leitura_com = registro.le(com)
    assert leitura_com is not None
    assert leitura_com.estado == "falhou"
    assert leitura_com.erro is not None
    assert "TypeError" in leitura_com.erro

    sem = registro.abre(grade.size(), identificador="sem-grupos")
    executa_registrando(_MatrizFalsa(), grade, registro, sem, executor=executor)  # type: ignore[arg-type]
    leitura_sem = registro.le(sem)
    assert leitura_sem is not None
    assert leitura_sem.estado == "concluido", "sem par de grupos o desfecho tem de divergir"
    assert leitura_sem.resultado is None, (
        "sem par declarado não há conclusão a retar; um resultado aqui seria contraste inventado"
    )


@pytest.mark.integration
def test_u19_a_conclusao_e_medida_ordenada_e_carrega_a_populacao() -> None:
    """A conclusão sobre execução REAL, e as quatro propriedades que a tornam legível.

    1. **`n_cells` viaja com cada linha**, porque `frac_significant = 0,5` sobre 1 célula e sobre
       4.320 são a mesma fração e evidências incomparáveis.
    2. **A ordem é decrescente por fração, com o nome como desempate**, e o desempate garante o
       determinismo: sem ele duas execuções da mesma grade dariam a mesma tabela em ordens
       diferentes.
    3. **`verdict` NÃO é servido** — o limiar é decisão em aberto, e carimbar
       *"robusta"* seria afirmar ciência que o trabalho não mediu.
    4. **`grupos` está na resposta**, senão a tabela é uma fração sem a população em que foi medida.

    A contagem de features é OBSERVAÇÃO sobre este arquivo e não invariante: o que se
    afirma é que ela é a mesma que `feature_stability` devolveu, não um número.
    """
    from refino.api.progresso import resume_o_multiverso
    from refino.io.readers import load_matrix
    from refino.multiverse.executor import run_multiverse

    modo = "neg_c18"
    matriz = load_matrix(measurements_path(modo), sample_list_path(modo), mode=modo)
    disponiveis = sorted(set(matriz.groups))
    par = ("ATLL", "NoHTLV")
    assert set(par) <= set(disponiveis), f"o par do teste não existe no arquivo: {disponiveis}"

    saida = run_multiverse(matriz, _grade_de_uma_celula(), n_jobs=1)
    resultado = resume_o_multiverso(saida, par, "real")

    assert resultado.grupos == par
    assert resultado.n_celulas == 1
    assert resultado.n_features == len(resultado.estabilidade) > 0, (
        "tabela vazia: a asserção de ordem passaria por vacuidade"
    )
    # A propriedade que este portão guarda MUDOU. Ele exigia que o veredito NÃO chegasse à
    # fronteira, porque ele usa limiares que ninguém validou. Os limiares vêm da tabela de
    # *Classificação de robustez*, e o que continua verdade é que ninguém os calibrou para ESTES
    # dados. Então o veredito passa, e o corte passa com ele: é isso que impede o rótulo de ser
    # lido como sentença.
    linha = resultado.estabilidade[0].model_dump()
    assert "verdict" in linha, "o veredito deixou de chegar à fronteira, e a U7 existe para o levar"
    assert resultado.limiares is not None, (
        "o veredito chegou SEM o corte que o produziu: rótulo sem população, na fronteira"
    )

    # A ordem é afirmada aqui e o PODER dela está no teste da chave, porque neste arquivo
    # nenhuma feature sobrevive ao FDR e todas as frações valem `0,0`. Esta asserção é
    # sanidade de forma; quem separa ascendente de descendente é
    # `test_u19_a_ordem_e_decrescente_por_fracao_com_o_nome_como_desempate`.
    chaves = [(-e.frac_significant, e.feature) for e in resultado.estabilidade]
    assert chaves == sorted(chaves), "a tabela não está ordenada por estabilidade decrescente"

    outra = resume_o_multiverso(saida, par, "real")
    assert [e.feature for e in outra.estabilidade] == [e.feature for e in resultado.estabilidade], (
        "duas passagens sobre a MESMA execução deram ordens diferentes"
    )

    # **Feature testada em célula nenhuma: `n_cells == 0` e os agregados `None`.** Que existam
    # features assim neste arquivo é OBSERVAÇÃO — medido em 24/08/2026, duas —, então o que se
    # afirma é a propriedade sobre as que houver, e o ramo em si é exercitado em `_ou_nulo`.
    for linha in resultado.estabilidade:
        if linha.n_cells == 0:
            assert linha.median_qvalue is None, "«não medido» chegou como número"
            assert linha.frac_significant == 0.0


@pytest.mark.property
def test_u19_a_tabela_sai_ordenada_sobre_dado_em_que_a_fracao_diverge() -> None:
    """`resume_o_multiverso` **usa** a chave de ordem, medido sobre dado que a exercita.

    > **Este teste existe porque uma planta escapou.** Trocar a chave por `lambda m: m.feature`
    > dentro de `resume_o_multiverso` **passava** no teste de execução real: nos arquivos do
    > laboratório nenhuma feature sobrevive ao FDR, todas as frações valem `0,0`, e ordenar por nome
    > dá a mesma tabela que ordenar por fração. O teste da chave sozinho verifica a **alavanca**; o
    > sítio de uso ficava sem nada.

    **A matriz é a de `tests/unit/test_multiverse_stability.py` e não uma segunda igual.** Ela tem
    efeito plantado nas quatro primeiras features exactamente por esta razão, e a docstring de lá o
    diz: *"sem ele `frac_significant` seria zero em toda parte e os testes de ordem compararariam
    zeros"*. Uma cópia aqui seria a segunda fonte de verdade sobre o mesmo fixture.
    """
    from refino.api.progresso import resume_o_multiverso
    from refino.multiverse.executor import run_multiverse
    from tests.unit.test_multiverse_stability import _matriz

    grade = GridDefinition(
        corrections=("none",),
        batch_options=(1,),
        rsd_cutoffs=(None, 0.30),
        missing_thresholds=(0.8,),
        imputations=("one_fifth_min",),
        sample_norms=("none",),
        transformations=("none",),
        scalings=("none",),
    )
    saida = run_multiverse(_matriz(), grade, n_jobs=1)
    resultado = resume_o_multiverso(saida, ("A", "B"), "sintetico")

    fracoes = [e.frac_significant for e in resultado.estabilidade]
    assert len(set(fracoes)) >= 2, (
        f"ÂNCORA: as frações não divergem ({set(fracoes)}), e a asserção de ordem abaixo passaria "
        "sob qualquer chave — é a planta vácua que este teste existe para não ser"
    )
    assert fracoes == sorted(fracoes, reverse=True), (
        f"a tabela não saiu por estabilidade decrescente: {fracoes[:6]}…"
    )
    assert fracoes[0] > fracoes[-1], "a primeira linha não é mais estável que a última"


def test_u19_resultado_404_para_identificador_desconhecido(cliente: TestClient) -> None:
    """404 e não 200 vazio: corrigir o identificador e esperar são ações opostas.

    A mensagem nomeia as **duas** causas de desconhecimento — nunca existiu, ou foi podado —, porque
    a segunda pede ação diferente: redisparar.
    """
    resposta = cliente.get("/multiversos/nao-existe/resultado")

    assert resposta.status_code == 404, resposta.text
    assert "nao-existe" in resposta.json()["detail"]
    assert "esquecid" in resposta.json()["detail"], "a mensagem não declara a poda como causa"


def test_u19_resultado_202_enquanto_a_execucao_corre() -> None:
    """**202 e não 200 com corpo de progresso.** O código é a metade que diz *"isto não é o
    resultado"*.

    Um cliente que recebesse 200 guardaria o corpo do progresso como se fosse a conclusão.
    """
    from refino.api import rotas

    identificador = rotas.REGISTRO.abre(10, identificador="u19-em-curso")
    rotas.REGISTRO.avanca(identificador, 3, 10)
    try:
        resposta = TestClient(build_app()).get(f"/multiversos/{identificador}/resultado")
        assert resposta.status_code == 202, resposta.text
        corpo = resposta.json()
        assert corpo["estado"] == "executando"
        assert (corpo["feito"], corpo["total"]) == (3, 10)
        assert corpo["terminou"] is False
    finally:
        rotas.REGISTRO.esquece(identificador)


def test_u19_resultado_409_quando_o_disparo_nao_declarou_o_par_de_grupos() -> None:
    """**409, e não 404 nem 200 vazio** — e é o desfecho que a compatibilidade produz.

    O trabalho existe e terminou bem; falta uma declaração de entrada que só quem dispara pode dar.
    `404` diria *"não conheço"* sobre algo que se conhece, e `200` com tabela vazia faria *"nenhuma
    feature sobrevive à grade"* — uma conclusão científica — sair igual a *"ninguém disse sobre que
    contraste medir"*.
    """
    from refino.api import rotas

    identificador = rotas.REGISTRO.abre(1, identificador="u19-sem-par")
    rotas.REGISTRO.conclui(identificador)
    try:
        resposta = TestClient(build_app()).get(f"/multiversos/{identificador}/resultado")
        assert resposta.status_code == 409, resposta.text
        detalhe = resposta.json()["detail"]
        assert "group_a" in detalhe and "group_b" in detalhe, (
            "a mensagem não nomeia os campos que faltam: quem lê não sabe o que redisparar"
        )
    finally:
        rotas.REGISTRO.esquece(identificador)


def test_u19_resultado_500_quando_a_execucao_falhou_e_nomeia_a_celula() -> None:
    """Falha é 500 com a **célula** nomeada, e não 200 nem 404.

    *"O multiverso falhou"* não diz onde reexecutar, e numa grade desse tamanho obriga a refazer
    tudo.
    """
    from refino.api import rotas

    identificador = rotas.REGISTRO.abre(4, identificador="u19-falhou")
    rotas.REGISTRO.falha(identificador, "abc123", "ZeroDivisionError: x")
    try:
        resposta = TestClient(build_app()).get(f"/multiversos/{identificador}/resultado")
        assert resposta.status_code == 500, resposta.text
        assert "abc123" in resposta.json()["detail"], "a célula que falhou não chegou à mensagem"
    finally:
        rotas.REGISTRO.esquece(identificador)


def test_u19_resultado_500_quando_retem_tipo_que_nao_e_conclusao() -> None:
    """O quinto desfecho, e ele é **defeito nosso declarado** em vez de um `assert`.

    Devolver `200` com um corpo que o cliente não sabe ler seria pior: ele o guardaria como
    conclusão.
    """
    from refino.api import rotas

    identificador = rotas.REGISTRO.abre(1, identificador="u19-tipo-errado")
    rotas.REGISTRO.conclui(identificador, resultado="isto não é conclusão")
    try:
        resposta = TestClient(build_app()).get(f"/multiversos/{identificador}/resultado")
        assert resposta.status_code == 500, resposta.text
        assert "str" in resposta.json()["detail"]
    finally:
        rotas.REGISTRO.esquece(identificador)


def test_u19_resultado_200_serve_a_conclusao_retida() -> None:
    """O desfecho de sucesso, sobre um resultado posto no registro — a **mesma instância** que a
    rota lê.

    O teste escreve em `rotas.REGISTRO` e a rota lê `REGISTRO` do mesmo módulo. Um
    `RegistroDeProgresso()` novo aqui mediria outra instância e passaria a afirmar nada.
    """
    from refino.api import rotas
    from refino.api.esquemas import (
        EstabilidadeDeFeature,
        LimiaresDoVeredito,
        ResultadoDoMultiverso,
    )

    conclusao = ResultadoDoMultiverso(
        identificador="u19-ok",
        grupos=("ATLL", "NoHTLV"),
        n_celulas=2,
        n_features=2,
        limiares=LimiaresDoVeredito(robust_frac=0.8, robust_sign=0.95, absent_frac=0.05),
        estabilidade=(
            EstabilidadeDeFeature(
                feature="1.0_100m/z",
                frac_significant=1.0,
                median_qvalue=0.01,
                median_rank=1.0,
                rank_iqr=0.0,
                sign_consistency=1.0,
                n_cells=2,
                # O valor de `verdict` aqui e coerente com os numeros ao lado:
                # `frac_significant=1,0` e `sign_consistency=1,0` estao acima de `0,80` e `0,95`.
                # Um veredito que contradissesse as medidas da mesma linha faria este fixture
                # afirmar um estado que a classificacao nunca produz.
                verdict="robust",
            ),
            EstabilidadeDeFeature(
                feature="2.0_200m/z",
                frac_significant=0.0,
                median_qvalue=None,
                median_rank=None,
                rank_iqr=None,
                sign_consistency=None,
                n_cells=0,
                # `frac_significant=0,0` cai abaixo de `absent_frac=0,05`, entao `absent` — e e o
                # caso medido em 24 de 24 pares nos quatro modos reais.
                verdict="absent",
            ),
        ),
    )
    identificador = rotas.REGISTRO.abre(2, identificador="u19-ok")
    rotas.REGISTRO.conclui(identificador, resultado=conclusao)
    try:
        resposta = TestClient(build_app()).get(f"/multiversos/{identificador}/resultado")
        assert resposta.status_code == 200, resposta.text
        corpo = resposta.json()
        assert corpo["grupos"] == ["ATLL", "NoHTLV"], "a população do número não chegou à resposta"
        assert corpo["n_celulas"] == 2
        assert [linha["feature"] for linha in corpo["estabilidade"]] == [
            "1.0_100m/z",
            "2.0_200m/z",
        ]
        assert corpo["estabilidade"][1]["median_qvalue"] is None, (
            "«não medido» tem de atravessar como `null`; `NaN` não é JSON válido"
        )
        assert corpo["estabilidade"][1]["n_cells"] == 0
    finally:
        rotas.REGISTRO.esquece(identificador)


# ---------------------------------------------------------------------------
# A guarda do par de grupos, no DISPARO
# ---------------------------------------------------------------------------


def test_u19_grupo_inexistente_e_recusado_no_disparo_nomeando_os_disponiveis(
    cliente: TestClient,
) -> None:
    """422 no disparo e não `falhou` na thread.

    Um nome errado descoberto pela thread viraria estado terminal minutos depois, com a mensagem
    longe da causa e a grade já paga. A mensagem lista os disponíveis porque é a única coisa
    acionável.
    """
    resposta = cliente.post(
        "/multiversos",
        files=_reais(),
        data={"mode": MODO, "group_a": "NaoExiste", "group_b": "ATLL"},
    )

    assert resposta.status_code == 422, resposta.text
    detalhe = resposta.json()["detail"]
    assert "NaoExiste" in detalhe
    assert "ATLL" in detalhe, "a mensagem não lista os grupos disponíveis"


def test_u19_par_com_os_dois_grupos_iguais_e_recusado(cliente: TestClient) -> None:
    """Contraste sem contraste. `_blocos_dos_grupos` já o recusa; recusar aqui é fazê-lo antes da
    grade."""
    resposta = cliente.post(
        "/multiversos",
        files=_reais(),
        data={"mode": MODO, "group_a": "ATLL", "group_b": "ATLL"},
    )

    assert resposta.status_code == 422, resposta.text
    assert "não há contraste" in resposta.json()["detail"]


def test_u19_declarar_so_um_dos_dois_e_recusa_e_nao_omissao(cliente: TestClient) -> None:
    """Ignorar um `group_a` sozinho executaria uma grade de horas e devolveria progresso a quem
    pediu conclusão, sem nada acusar — o modo de falha insidioso num parâmetro engolido."""
    resposta = cliente.post("/multiversos", files=_reais(), data={"mode": MODO, "group_a": "ATLL"})

    assert resposta.status_code == 422, resposta.text
    assert "PAR" in resposta.json()["detail"]


def test_u19_a_guarda_dos_grupos_tem_o_braco_de_controle() -> None:
    """Visto passar com a planta removida: sem par, `None`; com par válido, o par.

    Sem esta direção, uma guarda que recusasse **todo** par passaria nos três testes acima.
    """
    from refino.api.rotas import _grupos_ou_422
    from refino.io.readers import load_matrix

    modo = "neg_c18"
    matriz = load_matrix(measurements_path(modo), sample_list_path(modo), mode=modo)

    assert _grupos_ou_422(matriz, None, None) is None, "sem par pedido não há par a validar"
    assert _grupos_ou_422(matriz, "ATLL", "NoHTLV") == ("ATLL", "NoHTLV")


@pytest.mark.integration
def test_u19_o_disparo_do_multiverso_poda_os_terminados(cliente: TestClient) -> None:
    """O que este teste mede é o **sítio da chamada**: o disparo do multiverso poda.

    Até 24/08/2026 `REGISTRO` não era podado — `esquece_terminados_alem_de` tinha um chamador só, e
    era sobre `TRABALHOS`. Com a conclusão retida, cada execução terminada guarda uma linha por
    feature: `pos_hilic` mede 946.905 bytes de JSON (24/08/2026).

    **ESCOPO, e ele é estreito de propósito.** A propriedade *"a poda nunca esquece o que está a
    correr"* é do **registro** e não deste sítio, e vive em
    `tests/test_api_execucoes.py::test_a_poda_esquece_os_terminados_mais_antigos_e_nunca_o_que_esta_a_correr`,
    onde é medida sobre um registro fresco e com ordem determinística. Afirmá-la aqui seria a
    segunda fonte de verdade sobre a mesma coisa — e **medido**: uma planta que fizesse a poda
    alcançar o que corre passava neste teste e morria naquele, porque aqui a execução viva é a
    última a entrar e o excedente nunca a alcança. A asserção sobre `vivo` fica como sanidade, com
    o poder declarado onde ele está.

    > **A primeira versão deste teste esquecia o disparo novo no `finally` ENQUANTO a thread dele
    > corria**, e o desfecho foi um autodefeito com consequência: `avanca` levantou `KeyError`, e
    > `falha` — dentro do `except` de `executa_registrando` — levantou o **segundo**, matando a
    > thread com excepção não tratada. O arnês limpava a mesa debaixo do trabalho que ele mesmo
    > disparou. Agora espera pelo estado terminal antes de limpar, que é o que a limpeza significa.
    """
    from refino.api import rotas

    # `_reais()` sai do `try` e vem para antes do estado, e a ordem é o conteúdo.
    # Ele lê os dados de referência e emite `skip` quando faltam. Chamado lá dentro, o `skip` subia
    # com `resposta` por atribuir, o `finally` a referenciava, e o teste morria com
    # `UnboundLocalError` — isto é, um `skip` legítimo chegava ao relatório como ERRO, que é a pior
    # das trocas porque some da contagem de pulados e entra na de quebrados.
    #
    # Chamado aqui, o `skip` acontece com o registro ainda limpo e não há o que desfazer.
    arquivos = _reais()

    plantados = [
        rotas.REGISTRO.abre(1, identificador=f"u19-poda-{i}")
        for i in range(rotas.LIMITE_DE_MULTIVERSOS_RETIDOS + 3)
    ]
    for identificador in plantados:
        rotas.REGISTRO.conclui(identificador)
    vivo = rotas.REGISTRO.abre(99, identificador="u19-poda-vivo")
    rotas.REGISTRO.avanca(vivo, 1, 99)

    resposta = None
    try:
        resposta = cliente.post(
            "/multiversos",
            files=arquivos,
            data={
                "mode": MODO,
                "corrections": json.dumps(["none"]),
                "rsd_cutoffs": json.dumps([None]),
                "missing_thresholds": json.dumps([0.8]),
                "imputations": json.dumps(["median"]),
                "sample_norms": json.dumps(["none"]),
                "transformations": json.dumps(["none"]),
                "scalings": json.dumps(["none"]),
            },
        )
        assert resposta.status_code == 202, resposta.text

        assert rotas.REGISTRO.le(plantados[0]) is None, (
            "o terminado mais antigo sobreviveu à poda: o registro cresce sem limite"
        )
        # Sanidade e não o portão daquela propriedade — ver o ESCOPO na docstring.
        assert rotas.REGISTRO.le(vivo) is not None
        novo = resposta.json()["identificador"]
        for _ in range(600):
            leitura = rotas.REGISTRO.le(novo)
            if leitura is not None and leitura.terminou:
                break
            threading.Event().wait(0.5)
    finally:
        # **Esperar pelo terminal ANTES de limpar, e a ordem é o conteúdo.** Esquecer uma execução
        # em curso faz a thread dela levantar `KeyError` em `avanca` — e `falha`, no `except`,
        # levantar o segundo, matando a thread com excepção não tratada.
        #
        # O `if resposta` é a segunda metade do conserto, e cobre o caso que a hoisting
        # do `_reais()` não cobre: se o `post` levantar, `resposta` fica `None` e a limpeza tem de
        # continuar a acontecer para os plantados. Sem ele, uma falha no disparo apagaria a causa
        # real por baixo de um `UnboundLocalError` no `finally`.
        disparado = [resposta.json()["identificador"]] if resposta is not None else []
        for identificador in [*plantados, vivo, *disparado]:
            rotas.REGISTRO.esquece(identificador)


@pytest.mark.integration
def test_u19_a_travessia_completa_do_disparo_a_conclusao(cliente: TestClient) -> None:
    """**O fluxo INTEIRO pelas rotas reais**, e é a evidência que os testes por peça não dão.

    Cada peça desta unidade tem teste próprio: o resumo, os cinco desfechos, a guarda dos grupos, a
    poda. Nenhum deles atravessa a costura, e é ali que vive o defeito que as verificações por peça
    não alcançam: *"o defeito vive na COSTURA entre esta camada e uma função que ela não escreve"*.

    Então: `POST /multiversos` com o par de grupos, espera pelo estado terminal, e
    `GET /multiversos/{id}/resultado` — com as três propriedades que o profissional lê.

    **A grade é de DUAS células e o modo é o menor.** O que se mede é a travessia, não o custo.
    """
    modo = "neg_c18"
    arquivos = {
        "measurements": (
            f"measurements_{modo}.csv",
            measurements_path(modo).read_bytes(),
            "text/csv",
        ),
        "sample_list": (
            f"Sample_List_{modo}.csv",
            sample_list_path(modo).read_bytes(),
            "text/csv",
        ),
    }
    disparo = cliente.post(
        "/multiversos",
        files=arquivos,
        data={
            "mode": modo,
            "corrections": json.dumps(["none"]),
            "batch_options": json.dumps([1]),
            "rsd_cutoffs": json.dumps([None, 0.30]),
            "missing_thresholds": json.dumps([0.8]),
            "imputations": json.dumps(["median"]),
            "sample_norms": json.dumps(["none"]),
            "transformations": json.dumps(["none"]),
            "scalings": json.dumps(["none"]),
            "group_a": "ATLL",
            "group_b": "NoHTLV",
        },
    )
    assert disparo.status_code == 202, disparo.text
    identificador = disparo.json()["identificador"]
    assert disparo.json()["total"] == 2, f"a grade minima deu {disparo.json()['total']} celulas"

    try:
        for _ in range(600):
            leitura = cliente.get(f"/multiversos/{identificador}").json()
            if leitura["terminou"]:
                break
            threading.Event().wait(0.5)
        assert leitura["estado"] == "concluido", leitura

        resposta = cliente.get(f"/multiversos/{identificador}/resultado")
        assert resposta.status_code == 200, resposta.text
        corpo = resposta.json()

        # 1. O par de grupos chega, senão a tabela é uma fração sem a população em que foi medida.
        assert corpo["grupos"] == ["ATLL", "NoHTLV"]
        # 2. As células que a grade correu, e são as duas que o disparo declarou.
        assert corpo["n_celulas"] == 2
        # 3. Uma linha por feature, com `n_cells` em cada — o denominador viaja com o numerador.
        assert corpo["n_features"] == len(corpo["estabilidade"]) > 0
        assert all("n_cells" in linha for linha in corpo["estabilidade"])
        # Mesma inversão do veredito, um nível acima: a asserção é sobre o JSON que sai da rota.
        assert "verdict" in corpo["estabilidade"][0], (
            "o veredito não atravessou a rota, e a U7 existe para o levar até a tela"
        )
        assert corpo.get("limiares"), (
            "a rota serviu veredito sem o corte que o produziu: quem lê «robust» não tem como "
            "saber contra o que"
        )
        # E a ordem, que é o que faz a tabela ser legível de cima para baixo.
        chaves = [(-linha["frac_significant"], linha["feature"]) for linha in corpo["estabilidade"]]
        assert chaves == sorted(chaves)
    finally:
        from refino.api import rotas

        rotas.REGISTRO.esquece(identificador)


@pytest.mark.integration
def test_u19_a_travessia_sem_grupos_chega_ao_409_e_nao_a_uma_tabela_vazia(
    cliente: TestClient,
) -> None:
    """**O par causal da travessia**, e é ele que impede o teste acima de passar por coincidência.

    Mesma grade, mesmo modo, mesmo caminho — e **sem** o par de grupos. O desfecho tem de DIVERGIR:
    `409` com a ação, e não `200` com uma tabela vazia. Sem este braço, uma implementação que
    ignorasse `group_a`/`group_b` e servisse sempre a conclusão passaria no teste de sucesso.
    """
    modo = "neg_c18"
    arquivos = {
        "measurements": (
            f"measurements_{modo}.csv",
            measurements_path(modo).read_bytes(),
            "text/csv",
        ),
        "sample_list": (
            f"Sample_List_{modo}.csv",
            sample_list_path(modo).read_bytes(),
            "text/csv",
        ),
    }
    disparo = cliente.post(
        "/multiversos",
        files=arquivos,
        data={
            "mode": modo,
            "corrections": json.dumps(["none"]),
            "batch_options": json.dumps([1]),
            "rsd_cutoffs": json.dumps([None]),
            "missing_thresholds": json.dumps([0.8]),
            "imputations": json.dumps(["median"]),
            "sample_norms": json.dumps(["none"]),
            "transformations": json.dumps(["none"]),
            "scalings": json.dumps(["none"]),
        },
    )
    assert disparo.status_code == 202, disparo.text
    identificador = disparo.json()["identificador"]

    try:
        for _ in range(600):
            leitura = cliente.get(f"/multiversos/{identificador}").json()
            if leitura["terminou"]:
                break
            threading.Event().wait(0.5)
        assert leitura["estado"] == "concluido", leitura

        resposta = cliente.get(f"/multiversos/{identificador}/resultado")
        assert resposta.status_code == 409, (
            f"sem par de grupos a rota devolveu {resposta.status_code} e não 409. Se devolveu 200, "
            "alguém escolheu um contraste por nós"
        )
        assert "group_a" in resposta.json()["detail"]
    finally:
        from refino.api import rotas

        rotas.REGISTRO.esquece(identificador)


def test_a_matriz_real_nao_e_usada_pelo_executor_injetado() -> None:
    """Declaração de escopo, como asserção.

    Os testes de forma passam `_MatrizFalsa` porque o executor injetado não a toca. Se alguém
    trocar a injeção por uma que a use, isto reprova — e o teste de forma deixaria de medir só a
    forma sem que nada acusasse.
    """
    assert not hasattr(_MatrizFalsa(), "values")
    assert not isinstance(_MatrizFalsa(), RefinoMatrix)


def test_toda_rota_avisa_o_progresso_tantas_vezes_quantas_etapas_declara() -> None:
    """A barra chega aos 100%, e ate 31/08/2026 duas rotas em quatro nao chegavam.

    **Medido antes de consertar:** `execucoes` avisava 4 de 4 e `exportar` 2 de 2; `corrigir`
    avisava **1 de 2** e `processar` **2 de 3**. As duas paravam em 50% e 67% e saltavam para
    «concluido», porque `_executa_trabalho` chama `conclui` independentemente do contador.

    **O sintoma era invisivel de dois lados ao mesmo tempo.** Na tela, a barra a saltar de 67% para
    pronto parece velocidade e nao defeito. No disco, `_com_registro_de_tempo` fecha a duracao de
    uma etapa no aviso dela, entao a etapa nunca avisada **nao tinha onde ser fechada**.

    O oraculo e o TEXTO da rota e nao uma lista mantida a mao, porque uma lista seria a segunda
    fonte: ela diverge no dia em que uma rota ganhar etapa, e o portao
    ficaria a afirmar sobre a rota de ontem.

    ## POR QUE O DEFEITO SOBREVIVEU, e e a razao de este portao ser ESTATICO

    `conclui` **forca `feito` a `total`**, e a docstring dela diz porque: *"uma execucao que
    terminasse com `feito < total` deixaria a barra parada perto do fim"*. O estado TERMINAL,
    portanto, sempre esteve certo — e um teste que afirmasse `feito == total` depois de a rota
    concluir passaria **com o defeito presente**.

    > **A asercao natural era vacua por construcao.** Nao por falta de cuidado de quem a
    > escrevesse: ela mede depois da forca, e a forca apaga exactamente a diferenca que ha para ver.

    O que ficava errado era o intermedio — a leitura DURANTE a ultima etapa —, e afirma-lo em teste
    exigiria apanhar a execucao no meio, que e a corrida que este arquivo evita noutros pontos pelo
    mesmo motivo. Dai a forma: contar `marca()` contra a tupla, no TEXTO, sem correr nada. E a
    segunda metade — que a duracao daquela etapa chega ao disco — vive em
    `test_persistencia_sqlite.py`, onde ha uma linha gravada para conferir.
    """
    fora = rotas_que_nao_avisam_todas_as_etapas(_fonte_das_rotas())
    assert fora == [], (
        "estas rotas nao avisam o progresso tantas vezes quantas etapas declaram: "
        + "; ".join(f"{n}: declara {d}, avisa {a}" for n, d, a in fora)
    )


def _fonte_das_rotas() -> str:
    from pathlib import Path

    caminho = Path(__file__).resolve().parents[1] / "src" / "refino" / "api" / "rotas.py"
    return caminho.read_text(encoding="utf-8")


# As isenções do proxy estático, e cada uma é declarada com o que a substitui.
#
# O proxy deste portão conta as chamadas TEXTUAIS da marca e compara com os literais da tupla de
# etapas. Ele vale enquanto cada etapa tiver a sua própria chamada escrita — que é o desenho das
# quatro rotas em que ele nasceu.
#
# `medir-corretor` quebra a premissa e **não por descuido**: ela mede três corretores num laço
# dentro de `LabelingOracle.label` e avisa por um gancho, `depois_de_cada`. A chamada aparece
# **uma** vez no texto e corre **três**. O proxy lê `declara 3, avisa 1` e acusa uma rota correta.
#
# As três saídas foram pesadas, e esta é a menos má:
#
# | Saída | Por que não |
# |---|---|
# | escrever três chamadas literais | duplicaria o laço do oráculo dentro da rota |
# | ensinar o proxy a seguir o gancho | análise de fluxo por `regex`, frágil e especial |
# | **isentar, declarando o substituto** | é o que está feito, e o substituto é mais forte |
#
# **O substituto NÃO é vácuo, e é essa a diferença que autoriza a isenção.** A docstring do teste
# acima explica por que uma asserção de runtime sobre `feito == total` não serve: `conclui` força
# o valor no estado terminal. Contar **invocações do gancho** não tem esse problema — ninguém as
# força —, e é o que `tests/test_api_medir_corretor.py` faz, com par causal.
#
# `palpite` entrou pelo mesmo motivo: é a mesma medição, numa amostra de
# 200 features, com o mesmo gancho. O substituto está em `tests/test_api_palpite.py`, no teste
# `test_o_gancho_avisa_uma_vez_por_corretor_e_o_progresso_anda_antes_do_fim`, também com par causal.
ISENTAS_DO_PROXY_ESTATICO: Final[frozenset[str]] = frozenset({"medir-corretor", "palpite"})


def rotas_que_nao_avisam_todas_as_etapas(fonte: str) -> list[tuple[str, int, int]]:
    """`(nome, etapas declaradas, avisos)` de cada rota em que os dois numeros divergem.

    Funcao pura recebendo a fonte como ARGUMENTO: o negativo abaixo planta
    numa copia em memoria e a planta alimenta a asercao, entao ele nao consegue ficar inerte. A
    primeira versao lia o arquivo dentro do teste, e nao havia onde plantar sem tocar na arvore.
    """
    corpos = re.findall(
        r"def trabalho\(marca: Marca, cancelado: CancelCheck\).*?return _despacha\(.*?\n    \)",
        fonte,
        re.S,
    )
    # A contagem e DERIVADA e nao pinada. Este `assert`
    # dizia `== 4`, e reprovou no dia em que nasceu a quinta rota longa — `medir-corretor`. A
    # mensagem dele estava certa e nomeava os dois casos: *"ou uma rota mudou de forma e este
    # portao cegou, ou nasceu uma quinta que ninguem contou"*. Era a segunda.
    #
    # Pinar o numero fazia o portao acusar CRESCIMENTO LEGITIMO com a mesma voz com que acusa
    # cegueira, e as duas pedem acoes opostas. O sinal independente que separa as duas e o
    # cabecalho: cada rota longa tem exactamente um `def trabalho(marca, cancelado)`, entao o
    # padrao so pode ter cegado se capturar MENOS blocos do que ha cabecalhos.
    #
    # Contar `_despacha(` seria o instrumento errado. Ha **seis** ocorrencias
    # no arquivo e so cinco sao chamadas: a sexta vive dentro de um comentario que cita o proprio
    # padrao. Prosa contamina a contagem; a assinatura da funcao nao.
    cabecalhos = len(re.findall(r"def trabalho\(marca: Marca, cancelado: CancelCheck\)", fonte))
    assert cabecalhos >= 4, (
        f"ANCORA: o arquivo declara {cabecalhos} rotas longas, e havia quatro em 31/08/2026. "
        "Rota longa removida sem decisao registrada, ou a assinatura de `trabalho` mudou"
    )
    assert len(corpos) == cabecalhos, (
        f"o padrao capturou {len(corpos)} blocos para {cabecalhos} rotas longas declaradas. O "
        "portao CEGOU: alguma rota mudou de forma e deixou de casar, e a etapa que ela nao avisa "
        "passa a nao ser conferida por ninguem"
    )
    fora: list[tuple[str, int, int]] = []
    for nome, corpo, tupla in _rotas_corpos_e_tuplas(corpos):
        if nome in ISENTAS_DO_PROXY_ESTATICO:
            continue
        # Conta os LITERAIS e nao as virgulas. Contar
        # `,` + 1 vale para tupla de dois ou mais e erra por um na de UM elemento, que leva
        # virgula final obrigatoria em Python: `("medicao",)` tem uma virgula e uma etapa. A
        # rota `medir-corretor` e a primeira com uma etapa so, e o portao teria reportado
        # `declara 2, avisa 1` — reprovacao pelo motivo errado.
        declaradas = len(re.findall(r'"[^"]*"', tupla))
        avisos = len(re.findall(r"(?<!def )marca\(\)", corpo))
        if avisos != declaradas:
            fora.append((nome, declaradas, avisos))
    return fora


def _rotas_corpos_e_tuplas(corpos: list[str]) -> list[tuple[str, str, str]]:
    """`(nome, corpo, texto da tupla de etapas)` de cada rota longa capturada.

    Extraida para ser UMA fonte, lida por dois portoes. O proxy estatico
    de contagem e o portao das tuplas dinamicas precisam exactamente da mesma tupla; dois regex
    iguais escritos em dois sitios divergiriam no dia em que a forma da chamada mudasse, e um dos
    dois passaria a ler outra coisa sem que nada acusasse.

    A tupla e localizada pela POSICAO e nao pelo conteudo. Este padrao exigia que ela
    comecasse por `"correcao"`, e isso era a forma das quatro instancias em que ele nasceu — nao a
    propriedade. As quatro corrigem; `medir-corretor` MEDE, e a etapa dela chama-se outra coisa.
    Portao que nasce contra uma instancia codifica a forma daquela instancia.

    O que e contrato de facto e a POSICAO — a tupla de etapas e o argumento imediatamente antes de
    `trabalho` em `_despacha` —, e e nela que o padrao ancora.
    """
    achados: list[tuple[str, str, str]] = []
    for corpo in corpos:
        nome = re.search(r'return _despacha\(\s*\n\s*"([\w-]+)"', corpo)
        etapas = re.search(r"(\([^()]*?\))\s*,\s*\n\s*trabalho,", corpo)
        assert nome is not None and etapas is not None, corpo[:200]
        achados.append((nome.group(1), corpo, etapas.group(1)))
    return achados


# AS ROTAS CUJA TUPLA DE ETAPAS E DINAMICA, e o substituto de runtime de cada uma.
#
# O proxy acima conta os literais da tupla contra os avisos escritos no corpo. Uma tupla com `*`
# quebra essa premissa de um lado so: os literais continuam contaveis, e a parte desempacotada
# **nao**, porque um laco avisa uma vez por volta e aparece uma vez no texto. O proxy fica entao a
# medir uma FRACAO da rota — e o desfecho e verde, que e a forma que ninguem investiga.
#
# > **Declarar e o que impede o silencio.** Uma rota que ganhe `*` e nao apareca aqui **reprova**,
# > e a reprovacao nomeia o que falta: o teste de runtime que conta os avisos da parte dinamica.
#
# O precedente e `ISENTAS_DO_PROXY_ESTATICO` e o argumento e o mesmo — *"o substituto NAO e vacuo, e
# e essa a diferenca que autoriza a isencao"*. A diferenca entre as duas tabelas e o que cada uma
# isenta: aquela isenta a rota INTEIRA, esta so a metade desempacotada, e os literais continuam
# conferidos.
DINAMICAS_COM_SUBSTITUTO: Final[dict[str, str]] = {
    "execucoes": "test_a_execucao_avisa_uma_vez_por_familia_pedida",
}


def rotas_dinamicas_sem_substituto(fonte: str) -> list[str]:
    """As rotas cuja tupla de etapas tem `*` e que nao declaram substituto. Funcao PURA."""
    corpos = re.findall(
        r"def trabalho\(marca: Marca, cancelado: CancelCheck\).*?return _despacha\(.*?\n    \)",
        fonte,
        re.S,
    )
    return [
        nome
        for nome, _corpo, tupla in _rotas_corpos_e_tuplas(corpos)
        if "*" in tupla and nome not in DINAMICAS_COM_SUBSTITUTO
    ]


def rotas_com_tupla_dinamica(fonte: str) -> list[str]:
    """As rotas cuja tupla de etapas tem `*`, declaradas ou nao."""
    corpos = re.findall(
        r"def trabalho\(marca: Marca, cancelado: CancelCheck\).*?return _despacha\(.*?\n    \)",
        fonte,
        re.S,
    )
    return [nome for nome, _corpo, tupla in _rotas_corpos_e_tuplas(corpos) if "*" in tupla]


def test_toda_tupla_de_etapas_dinamica_declara_o_substituto_de_runtime() -> None:
    """Tupla com `*` sem substituto declarado REPROVA: a clausula fail-closed.

    A ancora de nao-vacuidade vem PRIMEIRO e na MESMA funcao. `rotas_dinamicas_sem_
    substituto` devolve lista vazia tanto quando toda rota dinamica esta declarada quanto quando
    **nao ha rota dinamica nenhuma** — e os dois desfechos sao fatos diferentes. Sem a ancora, o
    dia em que alguem voltasse a tupla de `/execucoes` para um literal deixaria este portao verde a
    afirmar que confere algo que ja nao existe.

    **A segunda metade e a que impede a declaracao de ser promessa:** o nome do substituto tem de
    existir NESTE modulo. Uma chave apontando para um teste que ninguem escreveu isenta a rota e
    nao verifica nada.
    """
    fonte = _fonte_das_rotas()
    dinamicas = rotas_com_tupla_dinamica(fonte)
    assert dinamicas, (
        "ANCORA: nenhuma rota tem tupla de etapas dinamica, e esta tabela existe para as cobrir. "
        "Ou a U23 foi revertida, ou o padrao da tupla cegou — nos dois casos a asercao abaixo "
        "passaria por vacuidade"
    )
    assert rotas_dinamicas_sem_substituto(fonte) == [], (
        "estas rotas tem tupla de etapas com `*` e nao declaram o teste de runtime que conta os "
        "avisos da parte desempacotada: " + ", ".join(rotas_dinamicas_sem_substituto(fonte))
    )
    faltam = [
        f"{rota} -> {teste}"
        for rota, teste in DINAMICAS_COM_SUBSTITUTO.items()
        if teste not in globals()
    ]
    assert faltam == [], (
        f"o substituto declarado nao existe neste modulo: {faltam}. Declaracao sem teste isenta a "
        "rota e nao verifica nada"
    )


def test_o_portao_das_dinamicas_acusa_uma_rota_nao_declarada() -> None:
    """O negativo, com a planta em MEMORIA e a asercao de que ela aterrou.

    **A planta muda o NOME da rota e nao a tupla**, e a escolha e o conteudo: mexer na tupla
    testaria o extrator; mexer no nome testa a **tabela**, que e o mecanismo que esta cláusula
    acrescenta. Com o nome trocado, a rota continua dinamica e deixa de casar a declaracao.
    """
    fonte = _fonte_das_rotas()
    alvo = '        "execucoes",\n'
    assert fonte.count(alvo) == 1, (
        f"ancora: esperava UMA ocorrencia do nome da rota no despacho e ha {fonte.count(alvo)}"
    )
    mutada = fonte.replace(alvo, '        "execucoes-renomeada",\n', 1)
    assert mutada != fonte, "a planta NAO aterrou"
    assert '"execucoes-renomeada"' in mutada, "a planta NAO aterrou no texto esperado"

    assert rotas_dinamicas_sem_substituto(fonte) == [], "a arvore limpa nao pode acusar"
    assert rotas_dinamicas_sem_substituto(mutada) == ["execucoes-renomeada"]


# AS TRES CONFIGURACOES, e elas sao o eixo a exercitar. O valor natural
# aqui e `PADRAO_DA_ANALISE`, que pede quatro familias — passar so ele deixaria por exercitar tanto
# o caso de nenhuma (a tupla volta as quatro fixas) quanto o de todas (a tupla dobra de tamanho).
# Pedia tres ate a normalidade entrar ligada por padrao, por ser barata; a nenhuma a
# desliga tambem, senao ela deixaria de ser «nenhuma» sem que nada acusasse.
NENHUMA_ANALISE: Final = AnalysisSpec(
    anova=False, correlation=False, permanova=False, normality=False, classifiers=()
)
TODAS_AS_ANALISES: Final = AnalysisSpec(
    anova=True,
    correlation=True,
    permanova=True,
    normality=True,
    plsda_multiclass=True,
    clustering=True,
    kmeans=True,
    som=True,
    dspc=True,
    classifiers=("random_forest", "svm"),
    # **`3` e nao o padrao `5`, e o numero e DERIVADO da matriz abaixo e nao escolhido.** A guarda
    # de `statistics/classificadores.py` exige ao menos `cv_folds` membros de CADA classe — senao
    # alguma dobra fica sem uma delas e o recall dela passa a ser medido sobre menos dobras que o
    # das outras. `MATRIZ_DAS_ANALISES` tem tres por grupo por construcao, porque e o minimo que a
    # guarda da ANOVA aceita, e o padrao de cinco faria os classificadores levantarem aqui em vez
    # de avisar.
    #
    # **Alinhar o parametro a populacao preserva o que esta fixtura existe para medir** — um aviso
    # por familia PEDIDA — em vez de encolher o pedido, que e o que tiraria as duas familias caras
    # do laco e mediria a tupla curta com o nome da longa. E de passagem passa valor nao-padrao a
    # `classifier_folds`, que nenhum outro arnes deste arquivo exercita.
    classifier_folds=3,
    # OS DOIS ABAIXO, e o `1` e DERIVADO da guarda e nao escolhido. Ele e o minimo que
    # `AnalysisSpec.__post_init__` aceita nos dois campos, e o que autoriza descer ate la e a
    # quantidade que esta fixtura mede: **um aviso por familia pedida**. `corre_as_analises` chama
    # `marca()` uma vez por familia, e essa contagem e invariante ao numero de sorteios e ao tamanho
    # da floresta — os dois campos mudam quanto o classificador CUSTA, e nao quantas vezes ele
    # avisa. Descer aqui nao encolhe o pedido: as duas familias continuam no laco e continuam a
    # correr o caminho real, que e a propriedade que a docstring de `MATRIZ_DAS_ANALISES` defende.
    #
    # A medicao que o motivou: com os padroes de
    # 200 sorteios e 300 arvores, os quatro testes que tocam classificadores levavam este arquivo
    # de 6,82 s para 341,85 s — ~335 s no portao de cada commit, e a fixtura nao podia pedir menos
    # porque os dois numeros nao eram parametro.
    classifier_permutations=1,
    classifier_trees=1,
)


MATRIZ_DAS_ANALISES: Final = RefinoMatrix(
    values=np.array(
        [[1.0, 9.0], [2.0, 8.0], [3.0, 7.0], [4.0, 6.0], [5.0, 5.0], [6.0, 4.0]],
        dtype=np.float64,
    ),
    feature_names=("f0", "f1"),
    samples=tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="sample",
            group="A" if i < 3 else "B",
        )
        for i in range(6)
    ),
    mode="sintetico",
)
"""A populacao que as familias de analise recebem — DOIS grupos de tres, e nao uma matriz vazia.

Ela existe porque `corre_as_analises` passou a exigir a matriz, e a exigencia e deliberada. Um
`matriz=None` teria deixado estes tres arneses intactos — e seria exactamente o
valor natural que executa menos do caminho: eles contam avisos, o `None` saltaria todo calculo, e
os «avisos por familia» passariam a ser medidos sobre o ramo que nao faz nada. Com esta matriz, o
mesmo arnes que conta avisos corre a ANOVA de verdade.

**Dois grupos de tres, e os dois numeros sao o MINIMO que a guarda aceita** — `MIN_GRUPOS` e
`MIN_AMOSTRAS_POR_GRUPO`, importados e nao redigitados. Um grupo a menos, ou uma amostra a menos
num grupo, e `_a_populacao_sustenta_a_anova` devolve `False`: a familia marcaria a etapa sem
calcular, e a contagem de avisos passaria na mesma — que e o desfecho que esta matriz existe para
NAO produzir.
"""

# A mascara das celulas imputadas desta matriz, e ela e VAZIA porque nenhuma foi.
# `corre_as_analises` passou a exigi-la, pela normalidade; esta matriz nao passou por imputacao
# nenhuma, entao a mascara verdadeira dela e esta, e nao um padrao conveniente.
SEM_IMPUTADAS: Final = np.zeros(MATRIZ_DAS_ANALISES.values.shape, dtype=bool)


def _avisos_de(analise: AnalysisSpec) -> int:
    """Quantas vezes `corre_as_analises` avisa o progresso para esta analise.

    Conta EXECUCOES do codigo sob teste e nao testes que passam: o numero que
    denuncia um ramo percorrido sem ser exercitado e este, e nao o desfecho verde.
    """
    avisos = 0

    def marca() -> None:
        nonlocal avisos
        avisos += 1

    corre_as_analises(
        analise,
        matriz=MATRIZ_DAS_ANALISES,
        imputadas=SEM_IMPUTADAS,
        seed=42,
        marca=marca,
        cancelado=lambda: False,
    )
    return avisos


def test_a_execucao_avisa_uma_vez_por_familia_pedida() -> None:
    """O SUBSTITUTO de runtime do proxy estatico, para a metade dinamica da tupla.

    **Por que um teste de runtime, e nao uma extensao do proxy.** O proxy conta chamadas TEXTUAIS;
    a parte dinamica e um laco, que aparece uma vez no texto e corre N vezes. Ensinar o `regex` a
    seguir o laco seria analise de fluxo por expressao regular — o mesmo caminho que
    `ISENTAS_DO_PROXY_ESTATICO` ja pesou e recusou. Contar invocacoes nao tem esse problema:
    ninguem as forca, ao contrario do `feito == total` que `conclui` fixa no estado terminal.

    A ancora de nao-vacuidade vem primeiro. `0 == 0` passa: uma configuracao sem familia
    nenhuma satisfaz a igualdade sem exercitar nada, e se o laco nunca corresse **todas** as
    configuracoes passariam. A ancora afirma que ha configuracao com nove familias antes de a
    igualdade ser lida.
    """
    configuracoes = (NENHUMA_ANALISE, PADRAO_DA_ANALISE, TODAS_AS_ANALISES)
    tamanhos = [len(etapas_das_analises(a)) for a in configuracoes]
    assert tamanhos == [0, 4, len(FAMILIAS_DE_ANALISE)], (
        f"ANCORA: esperava as tres configuracoes a declarar 0, 4 e {len(FAMILIAS_DE_ANALISE)} "
        f"etapas e vieram {tamanhos}. Sem a de cima e a de baixo, a igualdade abaixo passa por "
        "vacuidade"
    )
    for analise, esperado in zip(configuracoes, tamanhos, strict=True):
        assert _avisos_de(analise) == esperado, (
            f"a rota anuncia {esperado} etapas de analise e avisa {_avisos_de(analise)} vezes. A "
            "barra para no meio e salta para «concluido»"
        )


def test_o_substituto_acusa_uma_familia_que_nao_avisa() -> None:
    """O negativo do substituto: um corredor que salta a ultima familia e acusado.

    **A planta e um DUPLO do corredor e nao uma mutacao do arquivo**, porque o que se quer provar e
    que a **asercao** consegue reprovar — e ela compara duas quantidades que vivem em memoria. Um
    duplo que marque `N-1` vezes reproduz exactamente o defeito que o portao existe para apanhar,
    e o par causal e a comparacao dos dois desfechos sobre a MESMA analise.
    """

    def corredor_defeituoso(analise: AnalysisSpec, *, marca: Marca) -> int:
        avisos = 0
        for _familia in familias_pedidas(analise)[:-1]:
            marca()
            avisos += 1
        return avisos

    esperado = len(etapas_das_analises(TODAS_AS_ANALISES))
    assert esperado > 1, "ANCORA: a planta so separa se houver mais de uma familia a saltar"
    assert _avisos_de(TODAS_AS_ANALISES) == esperado, "a arvore limpa nao pode divergir"
    assert corredor_defeituoso(TODAS_AS_ANALISES, marca=lambda: None) == esperado - 1


def test_o_cancelamento_para_entre_familias_e_nomeia_qual() -> None:
    """Cancelar no meio das analises levanta com o rotulo da familia.

    **Uma checagem por familia e nao uma antes do laco.** Cada familia ha de custar minutos; com a
    checagem sozinha na entrada, «Cancelar» nao teria efeito durante o conjunto inteiro. O rotulo
    nomeia a familia porque *"executar: antes de"* sem sujeito manda o profissional procurar onde.
    """
    with pytest.raises(ExecutionCancelledError) as erro:
        corre_as_analises(
            TODAS_AS_ANALISES,
            matriz=MATRIZ_DAS_ANALISES,
            imputadas=SEM_IMPUTADAS,
            seed=42,
            marca=lambda: None,
            cancelado=lambda: True,
        )
    assert FAMILIAS_DE_ANALISE[0].etapa in str(erro.value), str(erro.value)

    # O outro desfecho, e e ele que prova que a guarda nao esta sempre a levantar.
    assert _avisos_de(TODAS_AS_ANALISES) == len(FAMILIAS_DE_ANALISE)


def test_o_portao_das_etapas_acusa_um_aviso_removido() -> None:
    """O negativo, com a planta em MEMORIA e a asercao de que ela aterrou.

    A planta e a remocao de um `marca()` — o defeito exacto que estas rotas tinham ate 31/08/2026.
    Sem a asercao de que a fonte mudou, um `replace` que nao casasse deixaria o teste verde a
    afirmar que o portao acusa, quando ele nao teria tido o que acusar.
    """
    fonte = _fonte_das_rotas()
    alvo = "        marca()\n        return artefato"
    assert fonte.count(alvo) == 2, (
        f"ancora: esperava DUAS ocorrencias do fecho de etapa e ha {fonte.count(alvo)}"
    )
    mutada = fonte.replace(alvo, "        return artefato", 1)
    assert mutada != fonte, "a planta NAO aterrou"

    assert rotas_que_nao_avisam_todas_as_etapas(fonte) == [], "a arvore limpa nao pode acusar"
    acusadas = rotas_que_nao_avisam_todas_as_etapas(mutada)
    assert len(acusadas) == 1, f"esperava UMA rota acusada e vieram {acusadas}"
    assert acusadas[0][1] - acusadas[0][2] == 1, acusadas


# ---------------------------------------------------------------------------
# O portão dos dublês — a classe que ficou INERTE dezoito dias
# ---------------------------------------------------------------------------
#
# **Medido em 18/09/2026, e o custo foi de sete portões.** `corrector_jobs` entrou na chamada de
# `executa_registrando` em `5794f09` (31/08/2026) e os oito dublês deste arquivo não
# acompanharam. O `TypeError` da chamada caiu no `except Exception` largo — que existe por
# desenho, para traduzir *qualquer* falha em estado terminal — e virou `falhou` com
# `feito=0`. Os sete testes do progresso continuaram a reprovar e **deixaram de verificar o
# progresso**: nenhum deles chegava a executar o corpo do dublê.
#
# **O que absorveu não foi um marcador, foi a TABELA DE CONTAGENS.**
# `docs/linha-de-base-windows.md` cita `test_api_progresso.py | 7` em quatro lugares, sempre
# como número e **nunca com causa atribuída**: reprova pelo motivo errado, e a linha de base
# absorve no papel de marcador.
#
# O oráculo é a PROPRIEDADE e não a forma de hoje: os nomeados que a CHAMADA passa,
# lidos da fonte. Reflexão sobre `run_multiverse` seria o alvo errado — um parâmetro que exista
# lá e que ninguém passe nunca quebra dublê nenhum, e o portão acusaria o legítimo.


def _nomeados_que_executa_registrando_passa() -> frozenset[str]:
    """Os argumentos nomeados da chamada `alvo(...)`, lidos de `api/progresso.py`."""
    import ast
    from pathlib import Path

    fonte = (
        Path(__file__).resolve().parents[1] / "src" / "refino" / "api" / "progresso.py"
    ).read_text(encoding="utf-8")
    for no in ast.walk(ast.parse(fonte)):
        if isinstance(no, ast.Call) and isinstance(no.func, ast.Name) and no.func.id == "alvo":
            return frozenset(k.arg for k in no.keywords if k.arg is not None)
    return frozenset()


def _dubles_deste_arquivo() -> dict[str, frozenset[str]]:
    """Cada `def executor*` deste arquivo e os nomeados que ele aceita, por leitura da fonte.

    Reflexão não alcança: os dublês são funções ANINHADAS dentro dos testes, e não existem como
    atributos de módulo. A fonte é a única instância que se pode ler — e é a instância certa.
    """
    import ast
    from pathlib import Path

    arvore = ast.parse(Path(__file__).resolve().read_text(encoding="utf-8"))
    achados: dict[str, frozenset[str]] = {}
    for no in ast.walk(arvore):
        if isinstance(no, ast.FunctionDef) and no.name.startswith("executor"):
            aceita = {a.arg for a in no.args.kwonlyargs}
            if no.args.kwarg is not None:
                aceita.add("**")
            achados[f"{no.name}:{no.lineno}"] = frozenset(aceita)
    return achados


def test_todo_duble_de_executor_aceita_o_que_executa_registrando_passa() -> None:
    """O portão que teria apanhado o defeito de 31/08 no dia em que ele nasceu."""
    exigidos = _nomeados_que_executa_registrando_passa()
    assert exigidos, (
        "não achei a chamada `alvo(...)` em api/progresso.py — o portão está a medir o vazio, "
        "e uma asserção de ausência sobre conjunto vazio passa por vacuidade"
    )
    assert "corrector_jobs" in exigidos, (
        f"âncora conhecida ausente: a chamada passa {sorted(exigidos)}, e `corrector_jobs` "
        "é o nomeado cuja falta custou sete portões inertes"
    )

    dubles = _dubles_deste_arquivo()
    assert len(dubles) >= 8, f"esperava ao menos os oito dublês e achei {len(dubles)}"

    faltam = {
        onde: sorted(exigidos - aceita)
        for onde, aceita in dubles.items()
        if "**" not in aceita and (exigidos - aceita)
    }
    assert not faltam, (
        "estes dublês não aceitam tudo o que `executa_registrando` passa. O `TypeError` NÃO "
        "aparece: o `except Exception` largo transforma-o em estado `falhou` com `feito=0`, e o "
        f"teste reprova por um motivo que não é o que ele afirma medir. {faltam}"
    )

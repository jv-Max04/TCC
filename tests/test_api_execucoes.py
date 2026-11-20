"""A execução longa deixa de ser chamada HTTP síncrona.

**O que esta unidade NÃO faz: tornar o SERRF mais rápido.** 209,9 s continuam
209,9 s; o que muda é a espera ser observável e recuperável.

## A asserção que separa assíncrono de síncrono com outro nome

Um teste que disparasse com `correction=none` e lesse `202` **não distingue os dois regimes**: a
correção `none` termina antes de a resposta ser serializada, então uma rota síncrona também
devolveria depressa. A prova exige que o disparo volte **enquanto o corretor está provadamente a
correr**, e é por isso que o corretor é bloqueado num `threading.Event` — barreira e não `sleep`,
pelo mesmo argumento dos testes de progresso: `sleep` calibrado passa nesta máquina e fica
intermitente na de outra pessoa.

**Nenhum teste desta unidade corre SERRF real.** O duplo intercepta o corretor, sob pena de
repetir a regressão de 3.468 s de suíte já medida — sete disparos reais a moer em segundo
plano enquanto o resto da suíte competia pelos núcleos.
"""

from __future__ import annotations

import math
import threading
from dataclasses import replace
from typing import Any, Final

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from refino.api.app import build_app
from refino.api.esquemas import (
    ComposicaoPorClasse,
    ExecucaoResposta,
    ParDePcas,
    RsdAntesEDepois,
    RsdPorFeature,
)
from refino.api.progresso import RegistroDeProgresso
from refino.api.rotas import (
    TRABALHOS,
    Marca,
    _chave_do_trabalho,
    _composicao_por_classe,
    _etapa_na_posicao,
    _executa_trabalho,
    _par_de_pcas,
    _qualidade_para_json,
    _resumo_das_distribuicoes,
    _rsd_antes_e_depois,
    _rsd_por_feature,
)
from refino.domain.cancellation import CancelCheck
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.spec import REQUIRED_METRIC_FIELDS
from refino.interface.fluxo import METRICAS_DA_TELA_DE_QUALIDADE
from refino.io.readers import load_matrix
from refino.metrics.report import build_quality_report
from tests.conftest import measurements_path, sample_list_path

pytestmark = pytest.mark.integration

MODO = "neg_c18"


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def _arquivos(mode: str = MODO) -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{mode}.csv",
            measurements_path(mode).read_bytes(),
            "text/csv",
        ),
        "sample_list": (f"Sample_List_{mode}.csv", sample_list_path(mode).read_bytes(), "text/csv"),
    }


def _matriz_para_relatorio(n_samples: int = 12, n_features: int = 6) -> RefinoMatrix:
    """Matriz sintética com QC bastante para as métricas obrigatórias saírem.

    **Sintética e não de referência de propósito**: o que se afirma
    é propriedade da **função** — quais campos o relatório carrega e sobre que população as duas
    medianas são calculadas —, não uma contagem observada nos arquivos do laboratório. Contagem
    observada é testada sobre os arquivos reais e nunca sobre matriz de factory;
    propriedade de função é o contrário, e usar os 6,08 MB aqui mediria a mesma coisa mais devagar.
    """
    rng = np.random.default_rng(11)
    nivel = rng.uniform(1e4, 1e6, size=(1, n_features))
    deriva = np.linspace(1.0, 3.0, n_samples)[:, None]
    ruido = 1.0 + 0.05 * rng.standard_normal(size=(n_samples, n_features))
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            sample_type="qc" if i % 4 == 0 else "sample",
            injection_order=i + 1,
            batch="A",
            group="QC" if i % 4 == 0 else "G1",
        )
        for i in range(n_samples)
    )
    return RefinoMatrix(
        values=nivel * deriva * ruido,
        feature_names=tuple(f"1.0_{100 + j}m/z" for j in range(n_features)),
        samples=amostras,
        mode=MODO,
    )


def _resposta_de_execucao() -> ExecucaoResposta:
    """Um `ExecucaoResposta` completo, para os testes que afirmam o TRANSPORTE e não o cálculo.

    `qualidade` e `rsd_antes_e_depois` são obrigatórios no esquema, então construí-los aqui é o
    que impede o modelo de ganhar campo com padrão só para o teste não mudar — padrão nesta
    resposta deixaria a rota livre para esquecer de o preencher.

    Passaram a QUATRO, e este teste reprovou para o obrigar. A chave de
    invalidação e o resumo das distribuições entraram **sem padrão**, pelo mesmo argumento acima, e
    a `ValidationError` que este construtor levantou é o mecanismo a funcionar: dar-lhes `None` para
    o teste voltar ao verde é exactamente o que a frase de cima proíbe.

    Passaram a CINCO, e o mecanismo repetiu-se: o par de PCAs
    entrou sem padrão e este construtor voltou a levantar `ValidationError`.

    SEIS, e o mecanismo repetiu-se pela terceira vez: os dois vetores de RSD
    por feature entraram sem padrão e o construtor voltou a levantar.

    SETE, e foi a quarta: a composição por classe entrou sem padrão e o
    construtor levantou `Field required [type=missing]` — a mesma frase, pela quarta vez. É o
    desfecho certo, e o errado seria dar-lhe `None` para o teste voltar ao verde.
    """
    relatorio = build_quality_report(
        _matriz_para_relatorio(), measurement_point="post_pipeline", structural_pretransform="none"
    )
    return ExecucaoResposta(
        matrix_id=1,
        run_id=2,
        decision_id=3,
        decision_proposta_id=None,
        estado_concordancia="nao_sobrescreveu",
        campos_sobrescritos=[],
        apresentacao="exibida",
        qc_rsd_median=0.31,
        n_features_saida=2162,
        qualidade=_qualidade_para_json(relatorio),
        rsd_antes_e_depois=_rsd_antes_e_depois(
            _matriz_para_relatorio(),
            _matriz_para_relatorio(),
            correcao="none",
            variant="outlier_removed",
        ),
        # Os 64 hex de `_chave_do_trabalho`, com a forma real: o que estes testes afirmam é o
        # TRANSPORTE, e uma chave inventada com outra forma faria o corpo transportado divergir do
        # que a rota produz sem nada acusar.
        chave_de_invalidacao="0" * 64,
        distribuicoes=_resumo_das_distribuicoes(
            _matriz_para_relatorio(), _matriz_para_relatorio(), seed=42
        ),
        pcas_da_correcao=_par_de_pcas(
            _matriz_para_relatorio(), _matriz_para_relatorio(), correcao="none"
        ),
        rsd_por_feature=_rsd_por_feature(
            _matriz_para_relatorio(), correcao="none", variante="outlier_removed"
        ),
        composicao_por_classe=_composicao_por_classe(_matriz_para_relatorio()),
    )


def _matriz_minima() -> RefinoMatrix:
    """Matriz sintética mínima, para os testes que afirmam a CHAVE e não o cálculo.

    Sintética e não de referência de propósito: o que se afirma é que a chave distingue os valores,
    e isso é propriedade da função — os 6,08 MB do laboratório mediriam o mesmo mais devagar.
    """
    valores = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    amostras = tuple(
        SampleMetadata(name=f"S{i}", sample_type="qc", injection_order=i + 1, batch="A", group="QC")
        for i in range(2)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=("1.0_100m/z", "1.0_101m/z", "1.0_102m/z"),
        samples=amostras,
        mode=MODO,
    )


class _CorretorBloqueado:
    """O corretor `none` real, atrás de uma barreira que o teste solta.

    **Envolve o real em vez de o imitar**, para que o artefato reclamado no fim seja o mesmo que a
    árvore produz sem barreira: um duplo que devolvesse matriz fabricada mediria o arnês.
    """

    def __init__(self, real: Any, pode_seguir: threading.Event, entrou: threading.Event) -> None:
        self._real, self._pode_seguir, self._entrou = real, pode_seguir, entrou
        self.name = getattr(real, "name", "none")

    def __getattr__(self, nome: str) -> Any:
        """Tudo o que não é a barreira vai ao real.

        **A primeira escrita listava os atributos à mão**, e o protocolo cresceu por baixo dela:
        o corretor ganhou `identidade()`, para a chave de reuso de correção. Este duplo
        não a tinha, `chave_da_correcao` levantava `AttributeError` dentro da thread de trabalho, o
        trabalho terminava em `500` — e o teste reprovava a dizer *"o corretor nunca começou a
        correr"*, que é verdade e não é a causa.

        A mensagem acusava o produto, e o defeito era do
        arnês. Delegar tudo faz a próxima adição ao protocolo não repetir isto: o duplo declara-se
        como *"envolve o real"*, e agora envolve mesmo.
        """
        return getattr(self._real, nome)

    def fit_transform(self, *args: Any, **kwargs: Any) -> Any:
        self._entrou.set()
        assert self._pode_seguir.wait(timeout=30), "a barreira nunca foi solta"
        return self._real.fit_transform(*args, **kwargs)


def test_o_disparo_devolve_202_antes_de_o_corretor_terminar(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch, memoria_de_correcoes_privada: None
) -> None:
    """A barreira é o que torna a asserção falsificável.

    Com o corretor bloqueado, uma rota síncrona **não pode** responder: o `202` só existe se o
    disparo tiver voltado sem esperar pelo cálculo.

    **A semente é PRÓPRIA porque este caso exige um disparo NOVO, e sem ela ele reprovava em ordem
    aleatória.** O registro de trabalhos é de processo e a chave inclui os parâmetros: a tela envia
    `method=none` com `seed=42` sobre o mesmo `neg_c18`, então um teste de `test_ui_consumo.py` que
    corresse antes deixava este disparo ser uma **reclamação** — o corretor não corria, e `entrou`
    esgotava os 30 s. O mecanismo estava certo; o teste supunha ordem.
    """
    from refino.api import rotas

    entrou, pode_seguir = threading.Event(), threading.Event()
    original = rotas._corretor_ou_422

    # **`**kwds` e não `paralelo: bool = True`, e a escolha é deliberada.** O duplo repassa o que
    # a rota passar; fixar a assinatura aqui faria este teste reprovar no dia em que um eixo novo
    # entrasse na fábrica, com uma mensagem sobre `TypeError` e não sobre o que ele mede.
    def bloqueado(method: str, null_denominator: str, **kwds: Any) -> Any:
        return _CorretorBloqueado(original(method, null_denominator, **kwds), pode_seguir, entrou)

    monkeypatch.setattr(rotas, "_corretor_ou_422", bloqueado)
    try:
        resposta = cliente.post(
            "/matrizes/corrigir",
            files=_arquivos(),
            data={"mode": MODO, "method": "none", "seed": 20260817},
        )

        assert resposta.status_code == 202, resposta.text
        identificador = resposta.json()["identificador"]
        assert identificador, "o disparo não devolveu identificador"
        assert entrou.wait(timeout=30), "o corretor nunca começou a correr"

        durante = cliente.get(f"/trabalhos/{identificador}")
        assert durante.status_code == 202, durante.text
        # A DISJUNÇÃO SAIU.
        #
        # Ela dizia `in ("pendente", "executando")`, e as duas metades pedem leituras opostas de
        # quem espera: `pendente` é, na tabela de `api/progresso.py`, *"registrada, ainda não
        # começou"*. Aqui o corretor **já entrou** — a barreira acima prova-o —, então só uma das
        # duas é verdade, e a disjunção passava com o registro a mentir.
        #
        # **O custo do que ela deixou passar:** quem correu o congelado com `serrf` leu
        # `pendente: 0/4 etapas` durante **15 min 14 s**, com o processo a queimar um núcleo
        # inteiro, e concluiu que a execução não tinha começado.
        assert durante.json()["estado"] == "executando", durante.text
        # **A SEGUNDA METADE, e é ela que impede o conserto errado.** O estado tem de andar **sem**
        # o contador andar: nenhuma das etapas terminou. Um conserto que marcasse `feito=1` antes
        # da primeira faria a barra afirmar trabalho que não aconteceu.
        assert durante.json()["feito"] == 0, durante.text
        assert durante.json()["terminou"] is False
    finally:
        pode_seguir.set()

    for _ in range(600):
        reclamada = cliente.get(f"/trabalhos/{identificador}")
        if reclamada.status_code != 202:
            break
        threading.Event().wait(0.05)

    assert reclamada.status_code == 200, reclamada.text
    assert reclamada.headers["content-type"].startswith("text/csv")
    assert reclamada.content.startswith(b"NA,batch,"), "o artefato reclamado não é o CSV `toserrf`"


def test_o_par_causal_com_a_barreira_removida_o_disparo_tambem_devolve_202(
    cliente: TestClient,
) -> None:
    """**O outro lado do par, e a razão de ele existir é declarar o que ele NÃO prova.**

    Com a correção `none` — instantânea — o `202` sai igual, e é por isso que este caso **não
    distingue os dois regimes**: uma rota síncrona também responderia depressa. Ele prova que a
    barreira do teste acima não é o que produz o `202`, e nada mais. Na direção em que o desfecho
    é o mesmo, o valor está em ver o caminho **sem** a planta, não em ver o mesmo código.

    **A semente é PRÓPRIA, e a primeira versão deste teste reprovou por não ter uma.** O registro é
    de **processo** e a chave do trabalho é a mesma para chamadas idênticas, então a chamada acima —
    igual a esta em tudo — fez este disparo devolver `reclamado=True`. O mecanismo estava certo e o
    teste estava a supor ordem de execução: com `-p randomly` a suíte é reordenada, e `reclamado is
    False` só é afirmável quando a chave é **única na sessão**.
    """
    resposta = cliente.post(
        "/matrizes/corrigir",
        files=_arquivos(),
        data={"mode": MODO, "method": "none", "seed": 99},
    )

    assert resposta.status_code == 202, resposta.text
    corpo = resposta.json()
    assert corpo["total"] == 2, f"as etapas de `corrigir` são duas: {corpo}"
    assert corpo["reclamado"] is False, "o primeiro disparo desta entrada não é reclamação"


def test_servir_o_resultado_marca_reclamado_e_nao_servir_nao_marca(cliente: TestClient) -> None:
    """O par causal da marca de reclamado na rota, e ele liga a marca ao ato de entregar.

    Os testes do registro afirmam que a poda ordena por `reclamado`; este afirma **quem põe a
    marca**. Sem ele, a poda estaria correta sobre um campo que nada no caminho real preenche — o
    mecanismo são e o referente ausente.

    **Dois disparos que diferem só por um ser reclamado.** O primeiro é buscado até terminar, o
    que serve o artefato; o segundo é disparado e **não** é buscado. Se a marca estivesse no `le` do
    topo da rota em vez de no ponto onde o resultado é devolvido, os dois sairiam marcados e esta
    comparação não distinguiria nada.

    **Sementes próprias e distintas**, porque a chave do trabalho é de processo: sem isso a suíte em
    ordem aleatória faria um destes disparos ser reclamação de outro teste.
    """
    buscado = cliente.post(
        "/matrizes/corrigir",
        files=_arquivos(),
        data={"mode": MODO, "method": "none", "seed": 286001},
    )
    nao_buscado = cliente.post(
        "/matrizes/corrigir",
        files=_arquivos(),
        data={"mode": MODO, "method": "none", "seed": 286002},
    )
    assert buscado.status_code == 202 and nao_buscado.status_code == 202
    id_buscado = buscado.json()["identificador"]
    id_nao_buscado = nao_buscado.json()["identificador"]
    assert id_buscado != id_nao_buscado, "âncora: os dois disparos partilharam trabalho"

    servida = _reclama_ate_terminar(cliente, id_buscado)
    assert servida.status_code == 200, servida.text
    assert servida.content.startswith(b"NA,batch,"), "âncora: o artefato não foi de facto servido"

    # O segundo precisa de ter TERMINADO para a comparação ser sobre a marca e não sobre o estado.
    for _ in range(600):
        leitura = TRABALHOS.le(id_nao_buscado)
        if leitura is not None and leitura.terminou:
            break
        threading.Event().wait(0.05)

    depois_de_servir = TRABALHOS.le(id_buscado)
    sem_ninguem_buscar = TRABALHOS.le(id_nao_buscado)
    assert depois_de_servir is not None and sem_ninguem_buscar is not None, (
        "âncora: algum dos dois trabalhos já foi podado, e a comparação seria vácua"
    )
    assert sem_ninguem_buscar.terminou, (
        "âncora: o segundo trabalho não terminou, então `reclamado is False` mede o estado e não a "
        "marca"
    )

    assert depois_de_servir.reclamado is True, (
        "servir o artefato não marcou o trabalho; a poda ordenaria por um campo que nada preenche"
    )
    assert sem_ninguem_buscar.reclamado is False, (
        "um trabalho que ninguém buscou saiu marcado como reclamado — a marca está no lugar "
        "errado, e a poda sacrificaria o artefato que alguém ainda espera"
    )


def _reclama_ate_terminar(cliente: TestClient, identificador: str, tentativas: int = 600) -> Any:
    """Consulta até o trabalho sair de `202`, e devolve a resposta final.

    Laço com teto e não `sleep` calibrado, como nos testes de progresso: o teto existe para o teste
    reprovar em vez de pendurar quando o trabalho morre sem estado terminal, e o intervalo é curto
    porque o que se espera é milissegundos de `none` e não minutos de SERRF.
    """
    for _ in range(tentativas):
        resposta = cliente.get(f"/trabalhos/{identificador}")
        if resposta.status_code != 202:
            return resposta
        threading.Event().wait(0.05)
    raise AssertionError(f"o trabalho {identificador} não terminou em {tentativas} leituras")


def _dispara(cliente: TestClient, rota: str, campos: dict[str, Any]) -> dict[str, Any]:
    resposta = cliente.post(rota, files=_arquivos(), data={"mode": MODO, **campos})
    assert resposta.status_code == 202, resposta.text
    corpo: dict[str, Any] = resposta.json()
    return corpo


# ---------------------------------------------------------------------------
# Os CINCO desfechos da rota de reclamação — "os dois" não é resposta
# ---------------------------------------------------------------------------


def test_identificador_desconhecido_devolve_404_e_nao_um_progresso_vazio(
    cliente: TestClient,
) -> None:
    """**Desfecho 1 de 5.** *"Não conheço"* e *"conheço e não começou"* pedem ações opostas.

    Um progresso vazio em `pendente` faria a tela esperar para sempre por algo que nunca foi
    registrado — e a mensagem nomeia a **segunda** causa de desconhecimento, a poda, porque a ação
    dela é diferente: disparar outra vez.
    """
    resposta = cliente.get("/trabalhos/nao-existe")

    assert resposta.status_code == 404, resposta.text
    detalhe = str(resposta.json()["detail"])
    assert "nao-existe" in detalhe
    assert "esquecido" in detalhe, f"o 404 não declara a poda como causa possível: {detalhe}"


@pytest.mark.parametrize(
    ("rota", "campos", "etapas", "comeco"),
    [
        ("/matrizes/corrigir", {"method": "none"}, 2, b"NA,batch,"),
        # TRÊS, e eram duas. A rota passou a aceitar `method` e a correção é etapa
        # declarada — `("correcao", "pipeline", "escrita do artefato")`. O nome importa porque é ele
        # que `celula_que_falhou` reporta: uma falha do SERRF nomeada *"pipeline"* mandaria o
        # profissional reexecutar a etapa errada, e `_etapa_na_posicao` existe para nunca inventar.
        ("/matrizes/processar", {"imputation": "one_fifth_min"}, 3, b"NA,batch,"),
        # **DUAS desde 30/08/2026, e era uma.** A exportação passou a correr o corretor antes de
        # escrever — `("correcao", "exportacao")` —, pela mesma razão que a de cima ganhou a
        # terceira: a correção é etapa DECLARADA e não trabalho escondido. E o número tem de bater
        # com as chamadas de `marca()`, senão a barra da janela para antes do fim ou estoura.
        ("/matrizes/exportar", {"max_features": 5000}, 2, b"Sample"),
    ],
)
def test_as_tres_rotas_de_artefato_seguem_o_mesmo_padrao(
    cliente: TestClient, rota: str, campos: dict[str, Any], etapas: int, comeco: bytes
) -> None:
    """**Desfechos 2 e 3 de 5**, nas três rotas: `202` com progresso, e `200` com os bytes.

    **Um mecanismo aplicado a três rotas**, e é isso que a parametrização afirma: se uma delas
    tivesse contrato próprio, a tela precisaria de saber qual é qual.
    """
    corpo = _dispara(cliente, rota, campos)
    assert corpo["total"] == etapas, f"{rota}: etapas declaradas divergem de {etapas}"

    reclamada = _reclama_ate_terminar(cliente, corpo["identificador"])

    assert reclamada.status_code == 200, reclamada.text
    assert reclamada.headers["content-type"].startswith("text/csv")
    assert reclamada.content.startswith(comeco), f"{rota}: o artefato não tem a forma esperada"
    assert MODO in reclamada.headers["content-disposition"], (
        "o nome do arquivo não leva o modo; o download sairia com o rótulo do trabalho"
    )


def test_o_progresso_do_trabalho_alcanca_o_total_e_o_estado_e_terminal(
    cliente: TestClient,
) -> None:
    """Desfecho 2 de 5 lido com detalhe: o sinal de vida.

    O `202` traz o mesmo esquema que o multiverso serve, então a tela o renderiza pela mesma
    função. A asserção é sobre o **conteúdo** do sinal e não sobre o código: `feito`, `total` e
    `estado` nomeado.
    """
    corpo = _dispara(cliente, "/matrizes/exportar", {"max_features": 4000})
    vistos: list[dict[str, Any]] = []
    for _ in range(600):
        resposta = cliente.get(f"/trabalhos/{corpo['identificador']}")
        if resposta.status_code != 202:
            break
        vistos.append(resposta.json())
        threading.Event().wait(0.02)

    assert resposta.status_code == 200, resposta.text
    if vistos:
        # A leitura intermediária é OPORTUNISTA: `exportar` pode terminar antes da primeira
        # consulta, e exigir que ela apareça faria o teste intermitente. Quando ela acontece, o
        # que se afirma é a FORMA.
        assert vistos[0]["estado"] in ("pendente", "executando"), vistos[0]
        assert vistos[0]["terminou"] is False
        # **DOIS desde 30/08/2026, e era um.** A exportação corre o corretor antes de escrever, e
        # `_despacha` deriva o total de `len(etapas)`. Este número não é decoração: é o
        # denominador da barra que a janela desenha.
        assert vistos[0]["total"] == 2
        assert 0.0 <= vistos[0]["fracao"] <= 1.0


def test_o_resultado_de_execucao_e_reclamado_como_json_e_nao_como_csv(
    cliente: TestClient,
) -> None:
    """**Desfecho 4 de 5: o `200` que devolve corpo e não bytes.**

    **O registro é alimentado direto, e a escolha vai declarada.** O caminho ponta a ponta de
    `/execucoes` exige banco e vive em `tests/test_persistencia_sqlite.py` e
    `tests/test_api_sobrescrita.py`; o que se afirma **aqui** é o ramo da rota de reclamação que
    distingue `ArtefatoRetido` de `BaseModel` — e ele é função do que está retido, não de como o
    trabalho correu.
    """
    esperado = _resposta_de_execucao()
    identificador = TRABALHOS.abre(4, identificador="u10-json")
    TRABALHOS.conclui(identificador, resultado=esperado)
    try:
        resposta = cliente.get(f"/trabalhos/{identificador}")
    finally:
        TRABALHOS.esquece(identificador)

    assert resposta.status_code == 200, resposta.text
    assert resposta.headers["content-type"].startswith("application/json")
    assert resposta.json()["estado_concordancia"] == "nao_sobrescreveu"
    assert resposta.json()["n_features_saida"] == 2162
    # Os dois campos do relatório atravessam a reclamação, e é por aqui que eles chegam à tela: o
    # `POST` devolve `202` com identificador, então o corpo inteiro passa por esta rota ou por
    # nenhuma.
    assert set(resposta.json()["qualidade"]) >= METRICAS_DA_TELA_DE_QUALIDADE, (
        "o relatório reclamado não traz o conjunto que a tela de qualidade exige"
    )
    assert resposta.json()["rsd_antes_e_depois"]["rsd_estimator"] == "in_sample"


def test_a_falha_do_trabalho_preserva_o_codigo_que_o_disparo_teria_devolvido(
    cliente: TestClient,
) -> None:
    """**Desfecho 5 de 5**, e a asserção é sobre o CÓDIGO e não sobre haver erro.

    `409` e `500` levam o profissional a ações opostas — `interface/cliente.py` traduz um como
    *"do seu dado"* e o outro como *"defeito da ferramenta, não do seu arquivo"*. Colapsá-los
    apagaria a distinção na célula que decide se ele vai conferir o arquivo.

    **Par sobre a MESMA rota de reclamação**, para que a diferença seja só a do trabalho que
    falhou.
    """
    conflito = TRABALHOS.abre(2, identificador="u10-409")
    TRABALHOS.falha(conflito, "pipeline", "EmptyMatrixError: o corte removeu tudo", codigo_http=409)
    imprevisto = TRABALHOS.abre(2, identificador="u10-500")
    TRABALHOS.falha(imprevisto, "correcao", "ZeroDivisionError: x", codigo_http=500)
    try:
        do_dado = cliente.get(f"/trabalhos/{conflito}")
        do_ambiente = cliente.get(f"/trabalhos/{imprevisto}")
    finally:
        TRABALHOS.esquece(conflito)
        TRABALHOS.esquece(imprevisto)

    assert (do_dado.status_code, do_ambiente.status_code) == (409, 500), (
        f"os códigos não divergem: {do_dado.status_code} e {do_ambiente.status_code}"
    )
    assert "pipeline" in str(do_dado.json()["detail"]), "a etapa que falhou não é nomeada"
    assert "correcao" in str(do_ambiente.json()["detail"])


def test_falha_sem_codigo_classificado_cai_em_500_e_nao_em_200(cliente: TestClient) -> None:
    """O ramo `else` do código, que dispara por AUSÊNCIA e por isso é o mais fácil de ficar inerte.

    `codigo_http=None` é o que o multiverso grava, e um trabalho longo que o tivesse por defeito
    nosso não pode virar `200`: o cliente leria o corpo de erro como resultado.
    """
    identificador = TRABALHOS.abre(2, identificador="u10-sem-codigo")
    TRABALHOS.falha(identificador, "correcao", "erro sem classificação")
    try:
        resposta = cliente.get(f"/trabalhos/{identificador}")
    finally:
        TRABALHOS.esquece(identificador)

    assert resposta.status_code == 500, resposta.text


def test_concluir_sem_resultado_retido_e_declarado_e_nao_um_200_vazio(cliente: TestClient) -> None:
    """Defeito nosso, DITO em vez de servido como artefato de zero bytes.

    É o mesmo desfecho que a tela já trata com *"a rota respondeu com sucesso e sem artefato"*, e
    dizê-lo aqui é o que impede o cliente de guardar `b""` como se fosse o CSV.
    """
    identificador = TRABALHOS.abre(1, identificador="u10-sem-resultado")
    TRABALHOS.conclui(identificador)
    try:
        resposta = cliente.get(f"/trabalhos/{identificador}")
    finally:
        TRABALHOS.esquece(identificador)

    assert resposta.status_code == 500, resposta.text
    assert "não retém resultado" in str(resposta.json()["detail"])


def test_o_422_do_escritor_atravessa_a_thread_com_o_codigo_intacto(cliente: TestClient) -> None:
    """**O segundo ramo `HTTPException` do worker, na rota de exportação.**

    A docstring de `exportar` afirma que *"a recusa do escritor mudou de instante e não de código"*,
    e a medição de cobertura mostrou este `except` **não executado**, com a frase já escrita.

    `max_features=0` é a regra do próprio escritor — *"o corte mantém as N de menor RSD, e N abaixo
    de 1 não descreve nenhum arquivo"* —, e reimplementá-la na rota para a antecipar criaria a
    segunda fonte que divergiria dele.
    """
    corpo = _dispara(cliente, "/matrizes/exportar", {"max_features": 0})

    reclamada = _reclama_ate_terminar(cliente, corpo["identificador"])

    assert reclamada.status_code == 422, (
        f"a recusa do escritor não atravessou a thread como 422: {reclamada.status_code} — "
        f"{reclamada.text[:200]}"
    )
    detalhe = str(reclamada.json()["detail"])
    assert "max_features" in detalhe, f"o 422 não diz qual regra foi violada: {detalhe}"
    assert "exportacao" in detalhe, f"a etapa que falhou não é nomeada: {detalhe}"


def test_o_auxiliar_de_reclamacao_acusa_rota_que_volte_a_ser_sincrona() -> None:
    """**O auxiliar é ele próprio um mecanismo, e este é o desfecho que ninguém tinha visto.**

    `tests/reclamacao.py` substituiu `cliente.post(...)` em cinco arquivos, e é ele que decide o que
    as asserções de byte-identidade recebem. Se ele aceitasse um `200` do disparo, uma rota que
    voltasse a ser síncrona entregaria os bytes ali e **todas** aquelas asserções continuariam
    verdes — a propriedade da unidade desapareceria sem nada acusar.

    A âncora existe no código desde o início; o que faltava era vê-la **falhar**. As duas direções
    aqui, em processo, com duplos de três linhas — e a segunda cobre o teto de espera, que sem isto
    também nunca teria corrido.
    """
    import pytest as _pytest

    from tests.reclamacao import reclama

    class _Sincrona:
        """A regressão: o disparo devolve o artefato no corpo, como na rota síncrona."""

        def post(self, url: str, **kwargs: Any) -> Any:
            return _RespostaSimulada(200)

        def get(self, url: str, **kwargs: Any) -> Any:  # pragma: no cover - nunca alcançado
            raise AssertionError("o auxiliar não pode chegar ao GET com um disparo síncrono")

    class _NuncaTermina:
        """O trabalho que não sai de `202`: o teto de espera tem de reprovar, não pendurar."""

        def post(self, url: str, **kwargs: Any) -> Any:
            return _RespostaSimulada(202)

        def get(self, url: str, **kwargs: Any) -> Any:
            return _RespostaSimulada(202)

    with _pytest.raises(AssertionError, match="voltou a ser SÍNCRONA"):
        reclama(_Sincrona(), "/matrizes/corrigir", files={}, data={})

    with _pytest.raises(AssertionError, match="não terminou"):
        reclama(_NuncaTermina(), "/matrizes/corrigir", files={}, data={}, teto_s=0.0)


class _RespostaSimulada:
    """O mínimo que `tests/reclamacao.py` lê de uma resposta: código e corpo JSON."""

    def __init__(self, codigo: int) -> None:
        self.status_code = codigo

    def json(self) -> dict[str, Any]:
        return {"identificador": "u10-simulado", "estado": "executando", "total": 2}


# ---------------------------------------------------------------------------
# Disparo repetido é RECLAMADO, e nunca duplicado
# ---------------------------------------------------------------------------


def test_disparo_repetido_da_mesma_entrada_e_reclamado_e_o_corretor_corre_uma_vez(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch, memoria_de_correcoes_privada: None
) -> None:
    """A asserção que importa é a CONTAGEM de execuções e não a igualdade do id.

    Dois disparos com o mesmo identificador provam que o registro devolve a mesma chave; o que se
    proíbe é o **cálculo** correr duas vezes. Com `serrf` isso são 209,9 s a competir consigo mesmo
    pelos núcleos, e nada no servidor a dizer que há duas.

    **A escolha entre RECUSAR e RECLAMAR vai declarada** (ver a docstring de
    `abre_ou_reclama`): recusar com `409` deixaria quem perdeu o identificador sem o resultado e sem
    meio de o alcançar, que é o *"trabalho órfão"* que também se proíbe.

    A CONTAGEM É DE `fit_transform`, e a primeira versão deste teste contava outra coisa. Ela
    contava chamadas a `_corretor_ou_422` e deu **2**, que é o número certo e mede a **fronteira de
    vocabulário**: ela corre nos dois disparos por desenho, porque um rótulo inválido tem de ser
    recusado nos dois. O que se proíbe é o **cálculo** correr duas vezes, e sobre isso aquela
    contagem não dizia nada — o número estava à mão, tinha o nome certo, e media a quantidade
    errada.
    """
    from refino.api import rotas

    original = rotas._corretor_ou_422
    fronteiras: list[str] = []
    ajustes: list[str] = []

    class _Contando:
        """Envolve o corretor real e conta os AJUSTES, que é a quantidade restringida."""

        def __init__(self, real: Any) -> None:
            self._real = real
            self.name = getattr(real, "name", "none")

        def __getattr__(self, nome: str) -> Any:
            """Tudo o que não é a contagem vai ao real — pelo motivo de `_CorretorBloqueado`."""
            return getattr(self._real, nome)

        def fit_transform(self, *args: Any, **kwargs: Any) -> Any:
            ajustes.append(self.name)
            return self._real.fit_transform(*args, **kwargs)

    def contando(method: str, null_denominator: str, **kwds: Any) -> Any:
        fronteiras.append(method)
        return _Contando(original(method, null_denominator, **kwds))

    monkeypatch.setattr(rotas, "_corretor_ou_422", contando)
    primeiro = _dispara(cliente, "/matrizes/corrigir", {"method": "none", "seed": 7})
    _reclama_ate_terminar(cliente, primeiro["identificador"])
    segundo = _dispara(cliente, "/matrizes/corrigir", {"method": "none", "seed": 7})

    assert primeiro["reclamado"] is False, "o primeiro disparo foi lido como reclamação"
    assert segundo["reclamado"] is True, "o disparo repetido não avisou que é o mesmo trabalho"
    assert segundo["identificador"] == primeiro["identificador"]
    assert len(ajustes) == 1, (
        f"o mesmo cálculo correu {len(ajustes)} vezes; R32 proíbe duplicar o trabalho sobre a "
        "mesma entrada"
    )
    assert len(fronteiras) == 2, (
        "a fronteira de vocabulário deixou de correr no disparo repetido — ela tem de recusar "
        "rótulo inválido nas duas chamadas, e é por isso que a contagem dela NÃO mede R32"
    )

    reclamada = _reclama_ate_terminar(cliente, segundo["identificador"])
    assert reclamada.status_code == 200, reclamada.text


def test_o_409_do_pipeline_atravessa_a_thread_com_o_codigo_intacto(cliente: TestClient) -> None:
    """**O ramo `except HTTPException` do trabalho, exercitado PELO CAMINHO REAL.**

    O teste vizinho que afirma `409` planta o estado no registro à mão: ele prova que a rota de
    reclamação **replica** o código, e nada sobre o worker o **capturar**. Sem este caso, a tradução
    de `EmptyMatrixError` para `409` dentro da thread nunca corre, e um `except` que a perdesse
    daria `500` com a suíte verde — *"defeito da ferramenta"* sobre um corte que o profissional
    escolheu.

    O corte é minúsculo de propósito: `rsd_cutoff` é fração e mantém as features com RSD **abaixo**
    dele, então `1e-09` remove todas e `process_matrix` levanta.
    """
    corpo = _dispara(cliente, "/matrizes/processar", {"rsd_cutoff": 1e-09})

    reclamada = _reclama_ate_terminar(cliente, corpo["identificador"])

    assert reclamada.status_code == 409, (
        f"o `409` do filtro que esvazia a matriz não atravessou a thread: {reclamada.status_code} "
        f"— {reclamada.text[:200]}"
    )
    detalhe = str(reclamada.json()["detail"])
    assert "pipeline" in detalhe, f"a etapa que falhou não é nomeada: {detalhe}"


def test_entrada_diferente_produz_trabalho_diferente(cliente: TestClient) -> None:
    """**O par causal da reclamação**: sem ele, um registro que devolvesse sempre o mesmo trabalho
    passaria no teste acima — e o segundo pedido receberia o artefato do primeiro.

    Os três eixos são medidos um a um, porque cada um deles ignorado produz o mesmo defeito na
    camada do servidor: o resultado de uma entrada servido sob outra.
    """
    base = _dispara(cliente, "/matrizes/corrigir", {"method": "none", "seed": 1})
    outra_semente = _dispara(cliente, "/matrizes/corrigir", {"method": "none", "seed": 2})
    outra_operacao = _dispara(cliente, "/matrizes/exportar", {"max_features": 5000})

    assert base["reclamado"] is False, "âncora: o primeiro disparo é novo"
    assert outra_semente["identificador"] != base["identificador"], "a chave ignora os parâmetros"
    assert outra_operacao["identificador"] != base["identificador"], "a chave ignora a operação"


def test_a_chave_do_trabalho_distingue_a_matriz_e_nao_so_os_campos() -> None:
    """O eixo que a rota não consegue variar sem trocar de arquivo: a **matriz ingerida**.

    Em processo, porque a planta é argumento da função sob teste — negativo em processo não consegue
    ficar inerte.
    """
    uma = _matriz_minima()
    outra = uma.with_values(uma.values * 2.0, "planta do par causal")

    assert _chave_do_trabalho("corrigir", uma, {"a": 1}) == _chave_do_trabalho(
        "corrigir", uma, {"a": 1}
    ), "a mesma entrada deu chaves diferentes: nada seria nunca reclamado"
    assert _chave_do_trabalho("corrigir", uma, {"a": 1}) != _chave_do_trabalho(
        "corrigir", outra, {"a": 1}
    ), "a chave ignora os VALORES da matriz"
    assert _chave_do_trabalho("corrigir", uma, {"a": 1}) != _chave_do_trabalho(
        "processar", uma, {"a": 1}
    ), "a chave ignora a operação"


# ---------------------------------------------------------------------------
# O registro: reclamação atómica, poda com teto, e a etapa nomeada
# ---------------------------------------------------------------------------


def test_abre_ou_reclama_tem_os_dois_desfechos_e_a_chave_esquecida_conta_como_ausente() -> None:
    """Os TRÊS desfechos de `abre_ou_reclama`, e o terceiro é o que ninguém planta.

    Chave que aponta para trabalho já podado tem de contar como **ausente**: o resultado já não
    existe para ser reclamado, e devolvê-lo daria um identificador que a reclamação responde `404`.
    """
    registro = RegistroDeProgresso()

    primeiro, novo = registro.abre_ou_reclama(2, chave="k")
    segundo, reclamado = registro.abre_ou_reclama(2, chave="k")

    assert (novo, reclamado) == (False, True)
    assert primeiro == segundo

    registro.conclui(primeiro)
    registro.esquece(primeiro)
    terceiro, depois_de_esquecer = registro.abre_ou_reclama(2, chave="k")

    assert depois_de_esquecer is False, "reclamou um trabalho que já não existe"
    assert terceiro != primeiro


def test_a_poda_esquece_os_terminados_mais_antigos_e_nunca_o_que_esta_a_correr() -> None:
    """**Os dois desfechos da poda, mais a cláusula que a torna segura.**

    Um podador que esquecesse trabalho **vivo** deixaria a thread a correr com o registro a
    responder `404` — órfã no sentido exacto. E devolver o que esqueceu é o que
    distingue *"podou tudo"* de *"não podou nada"*, que num podador mudo saem iguais.
    """
    registro = RegistroDeProgresso()
    terminados = [registro.abre(1, identificador=f"t{i}") for i in range(3)]
    for identificador in terminados:
        registro.conclui(identificador, resultado=b"x")
    vivo = registro.abre(1, identificador="vivo")
    registro.avanca(vivo, 0, 1)

    assert registro.esquece_terminados_alem_de(5) == (), "podou com folga de sobra"

    esquecidos = registro.esquece_terminados_alem_de(1)

    assert esquecidos == ("t0", "t1"), f"a poda não começou pelos mais antigos: {esquecidos}"
    assert registro.le("t2") is not None, "o terminado mais recente tem de sobreviver"
    assert registro.le(vivo) is not None, "a poda matou um trabalho em curso"


def test_a_poda_esquece_o_reclamado_antes_do_que_ninguem_buscou() -> None:
    """O desfecho da marca de reclamado, e ele inverte a escolha da poda.

    A versão anterior podava só pela idade, e a docstring declarava o proxy: *"a política certa
    quando o critério é «quem já teve tempo de ser reclamado»"*. **Ter tempo não é ter sido
    buscado** — então ela descartava o artefato que ninguém pegou e mantinha o que já foi baixado.

    A planta é o **mais novo** ser o reclamado, para que idade e coleta apontem em direções
    **opostas**: com a política antiga sairia `t0`, o mais velho; com a nova sai `t2`, o que já foi
    levado. Se as duas apontassem para o mesmo, o teste passaria sob as duas políticas e não
    distinguiria conserto de coincidência.
    """
    registro = RegistroDeProgresso()
    for i in range(3):
        registro.conclui(registro.abre(1, identificador=f"t{i}"), resultado=b"x")

    assert registro.marca_reclamado("t2"), "âncora: a marca não encontrou o trabalho mais novo"
    assert registro.le("t2") is not None and registro.le("t2").reclamado, (  # type: ignore[union-attr]
        "âncora: a marca não ficou gravada, então a ordenação abaixo nada prova"
    )

    esquecidos = registro.esquece_terminados_alem_de(2)

    assert esquecidos == ("t2",), (
        f"a poda levou {esquecidos} em vez do reclamado `t2`: ela continua a escolher pela idade, "
        "e o artefato que ninguém buscou é que foi sacrificado"
    )
    assert registro.le("t0") is not None, "o mais antigo NÃO reclamado tem de sobreviver"


def test_sem_ninguem_reclamado_a_poda_e_identica_a_de_antes() -> None:
    """**A propriedade que torna a mudança segura, e ela é afirmada e não suposta.**

    A segunda chave da ordenação é a de inserção, então sem nenhum reclamado a política nova tem de
    devolver **exactamente** o que a antiga devolvia: o mais antigo primeiro. Sem este teste, uma
    ordenação que embaralhasse os não reclamados passaria despercebida — o teste acima só olha o
    grupo dos reclamados.
    """
    registro = RegistroDeProgresso()
    for i in range(4):
        registro.conclui(registro.abre(1, identificador=f"n{i}"), resultado=b"x")

    assert all(not registro.le(f"n{i}").reclamado for i in range(4)), (  # type: ignore[union-attr]
        "âncora: algum trabalho está marcado como reclamado, e a comparação abaixo mede outra coisa"
    )

    assert registro.esquece_terminados_alem_de(2) == ("n0", "n1"), (
        "sem reclamados, a ordem tem de ser a de inserção — o mais antigo primeiro"
    )


def test_a_marca_de_reclamado_distingue_identificador_conhecido_de_desconhecido() -> None:
    """Os **dois** desfechos de `marca_reclamado`, e por que ela devolve `bool`.

    Um marcador mudo tem os dois modos de falha indistinguíveis: marcar tudo e marcar
    nada saem iguais. `False` significa *"não há trabalho com este identificador"*, e é o único caso
    em que não anota.
    """
    registro = RegistroDeProgresso()
    identificador = registro.abre(1)
    registro.conclui(identificador, resultado=b"x")

    assert registro.marca_reclamado(identificador) is True
    assert registro.marca_reclamado("nunca-existiu") is False

    # Idempotente: a segunda chamada devolve `True` porque o trabalho existe e está marcado — o
    # valor de retorno é sobre o IDENTIFICADOR, não sobre ter havido mudança.
    assert registro.marca_reclamado(identificador) is True


def test_a_etapa_que_falhou_e_nomeada_e_o_indice_fora_da_lista_nao_inventa() -> None:
    """Nomear a etapa errada é pior que dizer que não se sabe (o argumento de `_celula_na_posicao`).

    Quem reexecutasse a errada veria sucesso e concluiria que o defeito era intermitente.
    """
    etapas = ("correcao", "pipeline")

    assert _etapa_na_posicao(etapas, 0) == "correcao"
    assert _etapa_na_posicao(etapas, 1) == "pipeline"
    assert "depois da ultima" in _etapa_na_posicao(etapas, 2)


def test_o_trabalho_que_falha_no_meio_nomeia_a_etapa_e_nao_a_primeira() -> None:
    """A contagem decide a etapa, e o aviso dispara **depois** de cada uma terminar.

    Em processo, com um trabalho que marca a primeira etapa e morre na segunda.
    """
    identificador = TRABALHOS.abre(2, identificador="u10-meio")

    def trabalho(marca: Marca, cancelado: CancelCheck) -> object:
        marca()
        raise ZeroDivisionError("morreu na segunda etapa")

    try:
        _executa_trabalho(identificador, ("primeira", "segunda"), trabalho)
        leitura = TRABALHOS.le(identificador)
    finally:
        TRABALHOS.esquece(identificador)

    assert leitura is not None
    assert leitura.estado == "falhou"
    assert leitura.celula_que_falhou == "segunda", (
        "com um aviso emitido, a que falhou é a de índice 1 em base zero"
    )
    assert leitura.codigo_http == 500
    assert leitura.resultado is None, "trabalho que falhou não pode retê resultado"


# ---------------------------------------------------------------------------
# A fronteira NÃO se moveu: o que era 4xx e 503 continua no disparo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rota", "campos"),
    [
        ("/matrizes/corrigir", {"mode": MODO, "method": "loes"}),
        ("/matrizes/corrigir", {"mode": "../fora", "method": "none"}),
        ("/matrizes/exportar", {"mode": MODO, "rsd_variant": "inventada"}),
    ],
)
def test_a_recusa_de_fronteira_continua_a_acontecer_no_disparo(
    cliente: TestClient, rota: str, campos: dict[str, Any]
) -> None:
    """**O `422` não pode virar `202` seguido de `422`.**

    Um erro de digitação que só aparecesse na reclamação faria o profissional esperar por ele — e a
    tela mostraria sinal de vida sobre um trabalho que já estava condenado quando nasceu.
    """
    resposta = cliente.post(rota, files=_arquivos(), data=campos)

    assert resposta.status_code == 422, f"{rota} com {campos} devolveu {resposta.status_code}"


def test_sem_banco_a_execucao_recusa_no_disparo_com_503_e_nao_202(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Descobrir que não há banco **depois** de 209,9 s de SERRF é o desfecho a evitar.

    É por isso que `_repositorio()` corre antes de a thread nascer: construir o motor é barato,
    usá-lo é que não é.
    """
    from refino.persistence.config import VARIAVEL_DA_URL

    monkeypatch.delenv("REFINO_DB_HOST", raising=False)
    monkeypatch.delenv(VARIAVEL_DA_URL, raising=False)

    resposta = cliente.post(
        "/execucoes",
        files=_arquivos(),
        data={
            "mode": MODO,
            "estudo": "u10",
            "correction": "none",
            "n_batches": 1,
            "rsd_variant": "plain",
            "imputation": "none",
            "sample_normalization": "none",
            "transformation": "none",
            "scaling": "none",
            "apresentacao": "exibida",
        },
    )

    assert resposta.status_code == 503, resposta.text
    assert VARIAVEL_DA_URL in str(resposta.json()["detail"])


# ---------------------------------------------------------------------------
# O relatório inteiro e o par antes/depois que a resposta passa a carregar
# ---------------------------------------------------------------------------
#
# **O caminho ponta a ponta de `/execucoes` exige banco e esta máquina não o tem** (o teste acima
# mede o `503`), então o que se afirma aqui são as duas funções que compõem a resposta. O negativo
# é **em processo**: a planta é argumento da função sob teste, e não há
# invocação intermediária que possa não a receber.


def test_u4_o_relatorio_que_a_resposta_carrega_traz_o_conjunto_que_a_tela_exige() -> None:
    """O conjunto completo de métricas atravessa a fronteira, e a ponte é o que se afirma.

    A tela deriva `METRICAS_DA_TELA_DE_QUALIDADE` de `QualityReport`, e a resposta trazia
    `qc_rsd_median` sozinho: as duas pontas eram derivadas da mesma verdade e **não se tocavam**.
    Um teste que só olhasse um lado passaria com as onze métricas a dizer *"não calculada"* na única
    camada em que alguém olha.

    **`>=` e não `==` de propósito.** O relatório tem 26 campos e a tela exibe onze — os outros
    quinze são contabilidade de feature e rótulos de medição, e exigir igualdade faria o portão
    reprovar por a resposta ser mais completa que a tela.
    """
    bruto = _qualidade_para_json(
        build_quality_report(
            _matriz_para_relatorio(),
            measurement_point="post_pipeline",
            structural_pretransform="none",
        )
    )

    assert METRICAS_DA_TELA_DE_QUALIDADE, "conjunto vazio: as asserções abaixo seriam vácuas"
    ausentes = sorted(METRICAS_DA_TELA_DE_QUALIDADE - set(bruto))
    assert ausentes == [], f"métricas que a tela exibe e a resposta não carrega: {ausentes}"
    # `computed` é o que responde PERTINÊNCIA antes de valor, e sem ele a tela não distingue
    # *"não calculei"* de *"calculei e é indefinido"*.
    assert set(bruto["computed"]) == REQUIRED_METRIC_FIELDS, (
        f"a chamada da rota põe em `computed` {sorted(bruto['computed'])}, e os obrigatórios são "
        f"{sorted(REQUIRED_METRIC_FIELDS)}"
    )
    assert bruto["computed"] == sorted(bruto["computed"]), "`computed` chega fora de ordem"


def test_u4_o_par_antes_e_depois_e_identico_sob_correcao_nenhuma_e_diverge_sob_correcao() -> None:
    """**Par causal, e as duas direções são o que impede o painel de ser decorativo.**

    Com a mesma matriz nos dois lados — que é o que `none` produz, porque `NoneCorrector` devolve a
    matriz inalterada em valor — as duas medianas são **iguais**. Se elas divergissem aí, o painel
    mostraria melhora onde nada foi feito, e ninguém teria como saber.

    A outra direção usa QCs constantes: RSD zero por construção, e é o desfecho que a ressalva do
    estimador existe para qualificar — melhora medida onde a variação foi apagada, não corrigida.
    """
    crua = _matriz_para_relatorio()
    igual = _rsd_antes_e_depois(crua, crua, correcao="none", variant="outlier_removed")

    assert igual.mediana_antes == igual.mediana_depois, (
        f"a mesma matriz nos dois lados deu medianas diferentes: {igual.mediana_antes} e "
        f"{igual.mediana_depois}"
    )
    assert igual.n_features_comparadas > 0, (
        "nenhuma feature comparável: a igualdade acima seria entre dois `nan`, por acidente"
    )

    achatada = RefinoMatrix(
        values=np.where(crua.qc_mask[:, None], 1.0, crua.values),
        feature_names=crua.feature_names,
        samples=crua.samples,
        mode=crua.mode,
    )
    divergente = _rsd_antes_e_depois(crua, achatada, correcao="loess", variant="outlier_removed")

    assert divergente.mediana_depois == 0.0, (
        f"QC constante tem RSD zero por construção, e saiu {divergente.mediana_depois}"
    )
    assert divergente.mediana_antes != divergente.mediana_depois, (
        "as duas medianas não divergem nem quando a variação dos QCs é apagada: o painel não "
        "distingue correção nenhuma de correção alguma"
    )


def test_u4_as_duas_medianas_saem_da_mesma_populacao_e_ela_e_declarada() -> None:
    """As duas medianas saem da mesma população, e ela é declarada.

    Corrigir muda quais features têm RSD definido. Com um denominador por lado, *"a mediana caiu"*
    seria comparação entre **conjuntos diferentes de features** apresentada como comparação
    temporal — o número certo, o nome certo, e a população que não é a declarada.

    A planta é uma feature cujo RSD é definido antes e **indefinido depois**: ela sai dos dois
    números, `n_features_comparadas` cai, e o total continua a viajar para que a distância seja
    visível.
    """
    crua = _matriz_para_relatorio()
    valores = crua.values.copy()
    # A primeira feature perde os QCs: um único QC não nulo não sustenta `sd` com denominador
    # `n - 1`, então o RSD dela fica indefinido **depois** e continua definido **antes**.
    valores[crua.qc_mask, 0] = 0.0
    perdida = RefinoMatrix(
        values=valores,
        feature_names=crua.feature_names,
        samples=crua.samples,
        mode=crua.mode,
    )

    par = _rsd_antes_e_depois(crua, perdida, correcao="loess", variant="outlier_removed")

    assert par.n_features == crua.n_features, "o total deixou de ser o total"
    assert par.n_features_comparadas == crua.n_features - 1, (
        f"a feature que perdeu o RSD continua na população comparada: "
        f"{par.n_features_comparadas} de {par.n_features}"
    )
    # Par causal: sem a planta, a população comparada é o total.
    inteira = _rsd_antes_e_depois(crua, crua, correcao="none", variant="outlier_removed")
    assert inteira.n_features_comparadas == crua.n_features, (
        "sem a planta a população comparada já não é o total: a asserção acima não mede a planta"
    )


def test_u4_sem_feature_comparavel_a_mediana_e_indefinida_e_nao_zero() -> None:
    """O ramo que dispara por AUSÊNCIA, e por isso o mais fácil de ficar inerte.

    Zero diria *"o RSD é perfeito"* sobre uma matriz em que ele não pôde ser medido. `nan` propaga
    a ausência.

    **Matriz sem QC nenhum**, que é o caso que `qc_rsd` já trata devolvendo `nan` para todas: aqui o
    que se afirma é o que o **agregador** faz com aquele vetor.
    """
    crua = _matriz_para_relatorio()
    sem_qc = RefinoMatrix(
        values=crua.values,
        feature_names=crua.feature_names,
        samples=tuple(
            SampleMetadata(
                name=a.name,
                sample_type="sample",
                injection_order=a.injection_order,
                batch=a.batch,
                group="G1",
            )
            for a in crua.samples
        ),
        mode=crua.mode,
    )

    par = _rsd_antes_e_depois(sem_qc, sem_qc, correcao="none", variant="outlier_removed")

    assert par.n_features_comparadas == 0, "âncora: havia feature comparável, e o ramo não rodou"
    assert math.isnan(par.mediana_antes), (
        f"sem QC a mediana tinha de ser `nan` e saiu {par.mediana_antes}"
    )
    assert math.isnan(par.mediana_depois), (
        f"sem QC a mediana tinha de ser `nan` e saiu {par.mediana_depois}"
    )


def test_u4_o_rotulo_do_estimador_nao_aceita_o_nome_da_funcao() -> None:
    """Vocabulário fechado no campo que existe para tornar o número comparável.

    `leave_one_qc_out_rsd` é o nome da função; `leave_one_qc_out` é o do estimador. Com `str` no
    esquema, o primeiro atravessaria e a tela imprimiria um rótulo que o vocabulário do domínio não
    reconhece — e o modo de falha seria o insidioso: não levanta, e o leitor supõe que alguém
    conferiu.

    **Par causal na mesma função:** o rótulo válido é aceito e o nome da função é recusado, e o que
    se afirma é que os desfechos divergem. Sem a primeira metade, um esquema que recusasse **tudo**
    passaria na segunda.
    """
    bom = _rsd_antes_e_depois(
        _matriz_para_relatorio(),
        _matriz_para_relatorio(),
        correcao="none",
        variant="outlier_removed",
    )
    campos = bom.model_dump()

    assert bom.estimador_alternativo == "leave_one_qc_out", "âncora: o rótulo válido passa"
    assert RsdAntesEDepois.model_validate(campos).estimador_alternativo == "leave_one_qc_out"
    with pytest.raises(ValidationError):
        RsdAntesEDepois.model_validate({**campos, "estimador_alternativo": "leave_one_qc_out_rsd"})


# ---------------------------------------------------------------------------
# O PAR de PCAs que a resposta passa a carregar
# ---------------------------------------------------------------------------
#
# Em processo, e pela razão do bloco acima: o caminho ponta a ponta de `/execucoes` exige
# banco e esta máquina não o tem. A planta é argumento de `_par_de_pcas`, e não há invocação
# intermediária que possa não a receber.


def _com_feature_estragada(matriz: RefinoMatrix, coluna: int = 0) -> RefinoMatrix:
    """A mesma matriz com **uma** feature não finita — o que um corretor de facto produz.

    Não é caso inventado: nas matrizes golden do `batch_loess` o descarte por não finito chega a
    15,2% das features, e é essa a metade que a figura tem de tornar visível — parte de
    uma queda aparente de dispersão pode vir do **descarte** e não do agrupamento.
    """
    valores = matriz.values.copy()
    valores[0, coluna] = np.inf
    return RefinoMatrix(
        values=valores,
        feature_names=matriz.feature_names,
        samples=matriz.samples,
        mode=matriz.mode,
    )


def test_u2_os_dois_lados_declaram_a_procedencia_e_a_pretransformacao_de_KTD4() -> None:  # noqa: N802
    """A pré-transformação é fidelidade medida, e o rótulo é o que a torna conferível.

    `tools.R:1577` — e a irmã do outro ramo, `:1580` — faz `prcomp(t(...), scale.=TRUE)` sobre a
    matriz **crua**, sem log, e a
    padronização que `structural_pca` já aplica **é** esse `scale.=TRUE`. O `pretransform` do
    diagnóstico é `log10p1` — outro número com o mesmo nome —, então o rótulo viaja no corpo para
    que `qc_dispersion_ratio` daqui nunca seja lido como o do `QualityReport`.

    **E os dois rótulos de matriz são «ingerida» e «corrigida», e não «processada»:** comparar com a
    matriz processada deixaria imputação, normalização e escalonamento apertarem os QCs mesmo com
    `correction=none`, e o leitor atribuiria ao corretor o que foi do resto do pipeline.
    """
    crua = _matriz_para_relatorio()
    par = _par_de_pcas(crua, crua, correcao="none")

    assert par.matriz_antes == "ingerida", f"a procedência do «antes» saiu {par.matriz_antes!r}"
    assert par.matriz_depois == "corrigida", (
        f"a procedência do «depois» saiu {par.matriz_depois!r}; «processada» seria outra matriz — "
        "a do pipeline inteiro, e não a do corretor"
    )
    assert par.pretransformacao == "none", (
        f"a pré-transformação saiu {par.pretransformacao!r}, e KTD4 fixa `none`: logar mudaria a "
        "forma da nuvem que o laboratório reconhece no PNG do R"
    )
    assert par.n_componentes == 2, (
        f"o par correu com {par.n_componentes} componentes; `qc_dispersion_ratio` é a distância "
        "média ao centroide sobre TODOS eles, e mudar o número move o número sem mudar o nome"
    )


def test_u2_os_dois_lados_trazem_a_mesma_populacao_de_amostras_com_os_tipos_alinhados() -> None:
    """Os dois painéis desenham as **mesmas** injeções, e cada lado carrega o próprio vetor.

    Um vetor de tipos só, partilhado, seria invariante escrito sobre observação:
    *"nenhuma etapa remove amostras"* é propriedade do pipeline de hoje, e a correção podê-lo-ia
    mudar. Com um vetor por lado, o dia em que as populações divergirem produz dois vetores
    diferentes em vez de um rótulo certo sobre o conteúdo do outro painel.
    """
    crua = _matriz_para_relatorio()
    par = _par_de_pcas(crua, crua, correcao="none")

    assert len(par.scores_pc1_antes) == crua.n_samples, (
        f"o «antes» trouxe {len(par.scores_pc1_antes)} scores para {crua.n_samples} injeções"
    )
    for nome, vetor in (
        ("scores_pc2_antes", par.scores_pc2_antes),
        ("tipos_de_amostra_antes", par.tipos_de_amostra_antes),
        ("scores_pc1_depois", par.scores_pc1_depois),
        ("scores_pc2_depois", par.scores_pc2_depois),
        ("tipos_de_amostra_depois", par.tipos_de_amostra_depois),
    ):
        assert len(vetor) == crua.n_samples, (
            f"`{nome}` tem {len(vetor)} entradas e a matriz tem {crua.n_samples} injeções: os "
            "vetores só significam alguma coisa alinhados"
        )

    esperados = [a.sample_type for a in crua.samples]
    assert set(esperados) == {"qc", "sample"}, (
        f"âncora: a matriz do teste tem os tipos {sorted(set(esperados))}, e sem QC **e** amostra "
        "a igualdade abaixo não separaria série nenhuma"
    )
    assert par.tipos_de_amostra_antes == esperados
    assert par.tipos_de_amostra_depois == esperados

    # Os dois lados declaram o denominador próprio, e ele não é `n_features`: o passo 3
    # descarta coluna constante e coluna não finita.
    assert par.n_features_totais_antes == crua.n_features
    assert par.n_features_totais_depois == crua.n_features
    assert 0 < par.n_features_usadas_antes <= crua.n_features
    assert 0 < par.n_features_usadas_depois <= crua.n_features


def test_u2_os_dois_qc_dispersion_ratio_sao_identicos_sob_none_e_divergem_sob_correcao() -> None:
    """**O par causal, e ele cai sobre dado em que as duas coisas DIFEREM.**

    Com a mesma matriz nos dois lados — que é o que `correction="none"` produz, porque
    `NoneCorrector` devolve a matriz inalterada em valor — as duas dispersões são **iguais**. Se
    divergissem aí, a figura mostraria agrupamento onde nada foi feito.

    A outra direção achata os QCs num ponto: a dispersão deles vai a zero por construção, e é o
    desfecho que a figura existe para mostrar. **As duas quantidades são medidas e afirmadas
    divergentes antes de a planta ser lida como sucesso**, e a âncora de finitude vem primeiro —
    sem ela, `nan != nan` faria a metade divergente passar por vacuidade.
    """
    crua = _matriz_para_relatorio()
    igual = _par_de_pcas(crua, crua, correcao="none")

    assert igual.qc_dispersion_ratio_antes is not None, (
        "a dispersão do «antes» veio `None`: a igualdade abaixo seria entre duas ausências"
    )
    assert math.isfinite(igual.qc_dispersion_ratio_antes)
    assert igual.qc_dispersion_ratio_antes > 0.0, (
        f"a dispersão medida é {igual.qc_dispersion_ratio_antes}, e uma âncora em zero tornaria a "
        "divergência abaixo indistinguível de ruído"
    )
    assert igual.qc_dispersion_ratio_antes == igual.qc_dispersion_ratio_depois, (
        f"a mesma matriz nos dois lados deu dispersões diferentes: "
        f"{igual.qc_dispersion_ratio_antes} e {igual.qc_dispersion_ratio_depois}"
    )

    agrupada = RefinoMatrix(
        values=np.where(crua.qc_mask[:, None], 1.0, crua.values),
        feature_names=crua.feature_names,
        samples=crua.samples,
        mode=crua.mode,
    )
    divergente = _par_de_pcas(crua, agrupada, correcao="none")

    assert divergente.qc_dispersion_ratio_antes == igual.qc_dispersion_ratio_antes, (
        "o lado «antes» mudou quando só o «depois» foi plantado: os dois lados não são "
        "independentes, e a divergência abaixo não é atribuível ao corretor"
    )
    assert divergente.qc_dispersion_ratio_depois is not None
    assert divergente.qc_dispersion_ratio_depois != divergente.qc_dispersion_ratio_antes, (
        "as duas dispersões não divergem nem com os QCs achatados num ponto: a figura não "
        "distingue correção nenhuma de correção alguma"
    )
    assert divergente.qc_dispersion_ratio_depois == pytest.approx(0.0, abs=1e-12), (
        f"QC num ponto só tem dispersão zero por construção, e saiu "
        f"{divergente.qc_dispersion_ratio_depois}"
    )


def test_u2_os_digests_sao_iguais_sob_none_e_divergem_quando_o_corretor_estraga_feature() -> None:
    """Parte de uma queda de dispersão pode vir do DESCARTE e não do agrupamento.

    Contagem igual não é conjunto igual, e é por isso que o digest viaja ao lado de
    `n_features_usadas`. Aqui as duas coisas se movem juntas — a feature estragada sai das duas —,
    e é o digest que sobrevive ao caso em que uma entra enquanto outra sai.

    **O par causal é o mesmo das duas metades:** sob `none` os dois espaços são o mesmo, e a
    igualdade dos digests é a âncora que torna a divergência atribuível à planta.
    """
    crua = _matriz_para_relatorio()
    igual = _par_de_pcas(crua, crua, correcao="none")

    assert igual.digest_das_colunas_retidas_antes, "digest vazio: a igualdade abaixo seria vácua"
    assert igual.digest_das_colunas_retidas_antes == igual.digest_das_colunas_retidas_depois, (
        "a mesma matriz nos dois lados reteve espaços diferentes"
    )

    estragada = _com_feature_estragada(crua)
    divergente = _par_de_pcas(crua, estragada, correcao="none")

    assert divergente.n_features_usadas_depois == divergente.n_features_usadas_antes - 1, (
        f"a feature não finita não foi descartada: {divergente.n_features_usadas_antes} antes e "
        f"{divergente.n_features_usadas_depois} depois"
    )
    assert (
        divergente.digest_das_colunas_retidas_depois != divergente.digest_das_colunas_retidas_antes
    ), (
        "os dois digests coincidem com espaços de tamanhos diferentes: o digest não distingue os "
        "conjuntos, e a figura não teria como declarar a divergência que R1 exige"
    )
    assert divergente.digest_das_colunas_retidas_antes == igual.digest_das_colunas_retidas_antes, (
        "o lado «antes» mudou com a planta no «depois»: a divergência não é atribuível ao descarte"
    )


def test_u2_um_sampletype_fora_do_dominio_e_recusado_na_fronteira() -> None:
    """`Literal` é análise estática, e o corpo vem do fio.

    `SampleMetadata` já recusa na ingestão, e é por isso que o valor inválido não chega aqui **por
    esta rota**. O que este campo protege é a outra fronteira: o corpo JSON que o painel lê, e que
    nunca passou pelo mypy. Um `str` aceitaria `"blank"`, e o modo de falha seria o insidioso —
    o desenho poria a amostra numa série por omissão e o QC deixaria de ser distinguível.

    **Par causal na mesma função:** o vetor válido atravessa e o inválido é recusado. Sem a primeira
    metade, um esquema que recusasse **tudo** passaria na segunda.
    """
    crua = _matriz_para_relatorio()
    campos = _par_de_pcas(crua, crua, correcao="none").model_dump()

    assert "qc" in campos["tipos_de_amostra_antes"], "âncora: o vetor válido não tem QC nenhum"
    assert (
        ParDePcas.model_validate(campos).tipos_de_amostra_antes == campos["tipos_de_amostra_antes"]
    ), "âncora: o vetor válido não sobrevive ao roundtrip, e a recusa abaixo mediria outra coisa"

    invalidos = list(campos["tipos_de_amostra_antes"])
    invalidos[0] = "blank"
    with pytest.raises(ValidationError):
        ParDePcas.model_validate({**campos, "tipos_de_amostra_antes": invalidos})
    with pytest.raises(ValidationError):
        ParDePcas.model_validate({**campos, "tipos_de_amostra_depois": invalidos})


# ---------------------------------------------------------------------------
# O NOME DO CORRETOR, que é o `xlab` do painel direito
# ---------------------------------------------------------------------------
#
# Isto foi declarado como limitação do fio e não como escolha, e a declaração está na docstring
# de `plots/correcao.py`: `tools.R:1597` escreve naquele eixo
# `names(qc_RSD_performance)[length(qc_RSD_performance)]` — o nome do corretor —, e a figura
# escrevia «corrigida», que é o **estágio** que `matriz_depois` traz. O campo que o serve nasce
# aqui, e a fonte é `spec.correction`: o único sítio que sabe qual corretor produziu aquela matriz.


def test_u10_o_corretor_vem_do_ARGUMENTO_e_nao_do_estagio_do_pipeline() -> None:  # noqa: N802
    """**O par causal com as duas metades, e a segunda é a que torna a primeira atribuível.**

    Trocar `correcao` tem de mudar `corretor` — e **nada mais**. A segunda metade é a que impede a
    leitura de que o campo veio de outro sítio do par: se qualquer outro campo se mexesse, a
    diferença deixaria de ser atribuível ao argumento.

    **E a âncora de que o corretor não é o estágio vai junto**, porque é o defeito que esta unidade
    fecha: `matriz_depois` continua a ser «corrigida», e um campo que devolvesse o estágio passaria
    na primeira metade sem consertar nada.
    """
    crua = _matriz_para_relatorio()
    sem = _par_de_pcas(crua, crua, correcao="none")
    com = _par_de_pcas(crua, crua, correcao="loess")

    assert sem.corretor == "none", sem.corretor
    assert com.corretor == "loess", com.corretor
    assert sem.corretor != com.corretor, "o argumento não move o campo: ele vem de outro sítio"

    assert com.matriz_depois == "corrigida", (
        f"âncora: o estágio deixou de ser «corrigida» ({com.matriz_depois!r}), e a distinção "
        "abaixo mediria outra coisa"
    )
    assert com.corretor != com.matriz_depois, (
        "o corretor saiu igual ao estágio do pipeline: o eixo direito voltaria a escrever «"
        f"{com.matriz_depois}» em vez do nome que `tools.R:1597` escreve"
    )

    resto_sem = sem.model_dump(exclude={"corretor"})
    resto_com = com.model_dump(exclude={"corretor"})
    divergiram = sorted(k for k in resto_sem if resto_sem[k] != resto_com[k])
    assert resto_sem == resto_com, (
        "trocar o corretor mexeu noutro campo do par: a diferença de `corretor` deixa de ser "
        f"atribuível ao argumento. Divergiram {divergiram}"
    )


def test_u10_um_corretor_fora_do_dominio_e_recusado_na_fronteira() -> None:
    """`Literal` é análise estática, e o corpo do painel vem do fio.

    `PipelineSpec` já recusa na construção da spec, e é por isso que o valor inválido não chega aqui
    **por esta rota**. O que este campo protege é a outra fronteira: o corpo JSON que a janela lê, e
    que nunca passou pelo mypy.

    **O valor plantado é `"SERRF"` e não lixo, e a escolha é deliberada:** é o nome que o `tools.R`
    escreve naquele eixo, e é exactamente o que alguém escreveria a copiar do R. Um `str` aceitá-lo-
    ia, o eixo imprimiria um rótulo que o vocabulário desta ferramenta não reconhece, e o leitor
    concluiria que correu o SERRF do R.

    **Par causal na mesma função:** o valor válido atravessa e o inválido é recusado. Sem a primeira
    metade, um esquema que recusasse **tudo** passaria na segunda.
    """
    crua = _matriz_para_relatorio()
    campos = _par_de_pcas(crua, crua, correcao="loess").model_dump()

    assert campos["corretor"] == "loess", f"âncora: o campo válido saiu {campos['corretor']!r}"
    assert ParDePcas.model_validate(campos).corretor == "loess", (
        "âncora: o valor válido não sobrevive ao roundtrip, e a recusa abaixo mediria outra coisa"
    )

    for invalido in ("SERRF", "batch_loess", ""):
        with pytest.raises(ValidationError):
            ParDePcas.model_validate({**campos, "corretor": invalido})


# ---------------------------------------------------------------------------
# Os dois vetores de RSD por feature, QCs contra amostras
# ---------------------------------------------------------------------------
#
# Em processo, pela razão dos blocos acima: o caminho ponta a ponta de
# `/execucoes` exige banco e esta máquina não o tem. A planta é argumento de `_rsd_por_feature`, e
# não há invocação intermediária que possa não a receber.
#
# **As contagens de indefinidos dos QCs são OBSERVAÇÃO sobre os quatro arquivos deste laboratório**
# e não invariante de desenho, e é por isso que o teste que as afirma nomeia o arquivo, o
# modo e a variante — e usa os arquivos reais, nunca uma factory.


def _sem_qc(matriz: RefinoMatrix) -> RefinoMatrix:
    """A mesma matriz com as linhas de QC **removidas** — «calculado e vazio» do lado dos QCs.

    **Remover as linhas e não reetiquetá-las como `sample`, e a diferença foi MEDIDA.** A primeira
    escrita trocava o `sample_type` das três linhas de QC para `sample`, e com isso a população das
    amostras passava de 9 para 12 injeções: o lado das amostras mexia-se legitimamente, e o par
    causal deixava de atribuir o vazio à ausência de QC. Com as linhas fora, o vetor das amostras
    sai **idêntico** — conferido no teste, e é essa igualdade que torna a divergência atribuível.
    """
    manter = ~matriz.qc_mask
    return RefinoMatrix(
        values=matriz.values[manter],
        feature_names=matriz.feature_names,
        samples=tuple(
            amostra for amostra, fica in zip(matriz.samples, manter, strict=True) if fica
        ),
        mode=matriz.mode,
    )


def _com_outlier_numa_amostra(matriz: RefinoMatrix, coluna: int = 0) -> RefinoMatrix:
    """Uma célula aberrante numa injeção de amostra — o dado que SEPARA as duas variantes.

    **A matriz limpa não as separa, e isso é medição e não suposição.** Com 3 QCs nenhuma das duas
    fórmulas remove ponto nenhum — as dobradiças de Tukey são estatísticas de ordem em posições
    interiores, e com `n = 3` os limites caem fora da faixa dos dados —, e nas 9 amostras o ruído
    log-normal não produz aberrante nenhum. Medido sobre a coluna 0 das amostras: `plain` e
    `outlier_removed` dão **0,317117** nas duas. Com esta célula, `plain` dá **2,99999** e
    `outlier_removed` dá **0,279654**.
    """
    valores = matriz.values.copy()
    valores[np.flatnonzero(matriz.sample_mask)[0], coluna] = 1e12
    return RefinoMatrix(
        values=valores,
        feature_names=matriz.feature_names,
        samples=matriz.samples,
        mode=matriz.mode,
    )


def _com_qc_todo_zero(matriz: RefinoMatrix, coluna: int = 0) -> RefinoMatrix:
    """Uma feature detectada em amostra nenhuma dos QCs — a causa MINORITÁRIA de RSD indefinido.

    Zera a coluna **só nas linhas de QC**: a média dos QCs vai a zero, `qc_rsd` devolve `nan` por
    desenho (*"RSD indefinido é informação sobre a feature, não defeito do cálculo"*), e o lado das
    amostras fica intacto. É a planta que separa as duas populações sem tocar em `n_features`.
    """
    valores = matriz.values.copy()
    valores[matriz.qc_mask, coluna] = 0.0
    return RefinoMatrix(
        values=valores,
        feature_names=matriz.feature_names,
        samples=matriz.samples,
        mode=matriz.mode,
    )


def test_u6_os_dois_vetores_declaram_a_matriz_o_corretor_e_os_DOIS_rotulos_de_rsd() -> None:  # noqa: N802
    """**«RSD carrega dois rótulos ou não é comparável»**, e este corpo carrega os dois.

    `rsd_variant` e `rsd_estimator`: o mesmo arquivo golden dá 3,78% ou 104,73% para o
    `batch_loess` conforme o estimador, e o ranking entre métodos **inverte**. Sem os dois rótulos,
    o número que a figura escreve não é comparável com nenhum outro RSD do projeto — nem com
    o `rsd_antes_e_depois` deste mesmo corpo.

    **O estimador é `in_sample` porque é o que `qc_rsd` e `rsd` implementam**, e a ressalva viaja na
    docstring do modelo: in-sample, um corretor que ajusta modelo através dos QCs mede nos próprios
    pontos que usou, e a mediana do LOESS é 1,9e-15 no `pos_c18`.

    **E a matriz é a CORRIGIDA e não a processada** — o escopo é a correção, como no par de PCAs.
    """
    corrigida = _matriz_para_relatorio()
    vetores = _rsd_por_feature(corrigida, correcao="loess", variante="outlier_removed")

    assert vetores.matriz == "corrigida", (
        f"a procedência saiu {vetores.matriz!r}; «processada» seria outra matriz — a do pipeline "
        "inteiro, com imputação, normalização e escalonamento a apertarem os QCs"
    )
    assert vetores.corretor == "loess", vetores.corretor
    assert vetores.rsd_variant == "outlier_removed", vetores.rsd_variant
    assert vetores.rsd_estimator == "in_sample", (
        f"o estimador saiu {vetores.rsd_estimator!r}: `qc_rsd` e `rsd` medem nos próprios pontos, "
        "e declarar outro rótulo faria a figura afirmar uma avaliação fora da amostra que não houve"
    )

    assert _rsd_por_feature(corrigida, correcao="loess", variante="plain").rsd_variant == "plain", (
        "o rótulo não segue o argumento: ele estaria a descrever uma escolha que não foi feita"
    )


def test_u6_a_variante_declarada_alcanca_o_CALCULO_e_nao_so_o_rotulo() -> None:  # noqa: N802
    """**A planta tem de cair sobre dado em que as duas variantes DIFEREM, e a limpa não difere.**

    A primeira escrita deste teste comparava os dois vetores sobre a matriz limpa e reprovava com
    *"as duas variantes deram o mesmo vetor"* — e a conclusão teria sido falsa. **Medido antes de
    plantar:** com 3 QCs nenhuma das duas fórmulas pode remover ponto — as dobradiças são
    estatísticas de ordem em posições interiores —, e nas 9 amostras o ruído não produz aberrante
    nenhum; a coluna 0 dá **0,317117** nas duas variantes. O dado não separava, e um teste sobre ele
    mediria o dado e não o código.

    Com uma célula aberrante numa injeção de amostra, `plain` dá **2,99999** e `outlier_removed` dá
    **0,279654** na mesma coluna: fator 10,7. É a diferença que `remove_outlier()` do `tools.R`
    produz, e é grande — no `pos_hilic` sem correção a mediana vai de 167,77% para 143,81%.

    **Âncora de que o lado dos QCs NÃO se move**, e é ela que torna a divergência atribuível: a
    planta caiu numa linha de amostra, e a população dos QCs é `qc_mask`.
    """
    corrigida = _com_outlier_numa_amostra(_matriz_para_relatorio())
    plain = _rsd_por_feature(corrigida, correcao="none", variante="plain")
    sem_outlier = _rsd_por_feature(corrigida, correcao="none", variante="outlier_removed")

    assert plain.rsd_das_amostras and sem_outlier.rsd_das_amostras, (
        "âncora: um dos vetores veio vazio, e a comparação abaixo seria entre duas ausências"
    )
    assert plain.rsd_das_amostras != sem_outlier.rsd_das_amostras, (
        "as duas variantes deram o mesmo vetor das amostras mesmo com a célula aberrante: "
        "`variante` não alcança o cálculo, e o rótulo estaria a descrever uma escolha que não foi "
        "feita"
    )
    # A comparação é na COLUNA QUE A PLANTA TOCOU, e o primeiro `assert` que escrevi comparava o
    # `max` dos dois vetores — número à mão, e o errado: o maior sob `outlier_removed`
    # é 0,3447, de uma coluna intacta, e a razão contra 2,99999 dá 8,7 em vez do fator real de
    # 10,7. A âncora de que os dois vetores têm todas as features torna o índice 0 a coluna 0.
    assert plain.n_features_finitas_das_amostras == plain.n_features == corrigida.n_features, (
        "alguma feature ficou de fora do vetor, e o índice 0 deixa de ser a coluna plantada"
    )
    assert sem_outlier.n_features_finitas_das_amostras == plain.n_features_finitas_das_amostras
    assert plain.rsd_das_amostras[0] > 10 * sem_outlier.rsd_das_amostras[0], (
        f"a remoção de outliers não baixou o RSD da coluna plantada: "
        f"{plain.rsd_das_amostras[0]} contra {sem_outlier.rsd_das_amostras[0]}. O efeito de "
        "`remove_outlier()` é para BAIXO"
    )
    assert plain.rsd_dos_qcs == sem_outlier.rsd_dos_qcs, (
        "o lado dos QCs mexeu-se com uma planta numa linha de amostra: a divergência acima não é "
        "atribuível à variante"
    )


def test_u6_as_duas_contagens_de_finitos_divergem_com_a_planta_e_nao_sem_ela() -> None:
    """**O par causal, e as duas quantidades são MEDIDAS e afirmadas iguais antes da planta.**

    Na matriz sintética as duas populações **coincidem** — toda feature tem RSD finito nos dois
    lados —, e é isso que torna a divergência de baixo atribuível à planta e não ao arnês. Sem esta
    metade o teste seria vácuo: duas contagens que já diferiam continuariam a diferir.

    A planta zera **uma** feature nas linhas de QC. O lado dos QCs perde exactamente uma feature da
    população; o lado das amostras não se mexe; e `n_features` fica igual nos dois desfechos — é a
    distância entre o total e a população que a figura tem de escrever.
    """
    corrigida = _matriz_para_relatorio()
    iguais = _rsd_por_feature(corrigida, correcao="none", variante="outlier_removed")

    assert iguais.n_features == corrigida.n_features, iguais.n_features
    assert iguais.n_features_finitas_dos_qcs > 0, (
        "âncora: nenhuma feature tem RSD finito nos QCs, e a comparação abaixo seria entre dois "
        "vazios"
    )
    assert iguais.n_features_finitas_dos_qcs == iguais.n_features_finitas_das_amostras, (
        f"as duas populações já divergem sem planta nenhuma — {iguais.n_features_finitas_dos_qcs} "
        f"e {iguais.n_features_finitas_das_amostras} —, e a divergência de baixo não seria "
        "atribuível"
    )

    plantada = _rsd_por_feature(
        _com_qc_todo_zero(corrigida), correcao="none", variante="outlier_removed"
    )

    assert plantada.n_features == iguais.n_features, (
        "o total mudou com a planta: ela devia mexer na população e não no denominador"
    )
    assert plantada.n_features_finitas_das_amostras == iguais.n_features_finitas_das_amostras, (
        "o lado das amostras mexeu-se com uma planta que só tocou as linhas de QC: os dois lados "
        "não são independentes, e a divergência abaixo não é atribuível"
    )
    assert plantada.n_features_finitas_dos_qcs == iguais.n_features_finitas_dos_qcs - 1, (
        f"a feature com QCs todos zero continua na população dos QCs: "
        f"{plantada.n_features_finitas_dos_qcs} contra {iguais.n_features_finitas_dos_qcs}"
    )
    assert plantada.n_features_finitas_dos_qcs != plantada.n_features_finitas_das_amostras, (
        "as duas contagens não divergem nem com uma feature apagada de um dos lados: uma contagem "
        "só serviria para os dois, e a figura declararia de uma população o que mediu da outra"
    )

    # **O vetor é a população, e a contagem é o rótulo dela** — comprimentos e contagens batem.
    assert len(plantada.rsd_dos_qcs) == plantada.n_features_finitas_dos_qcs
    assert len(plantada.rsd_das_amostras) == plantada.n_features_finitas_das_amostras
    # A não-vacuidade vem ANTES do `all`: `all()` sobre lista vazia devolve `True`, e um
    # vetor vazio — que é exactamente o que uma extração partida produz — passaria por aqui.
    assert plantada.rsd_dos_qcs, "o vetor dos QCs veio vazio, e o `all` abaixo seria vácuo"
    assert all(math.isfinite(v) for v in plantada.rsd_dos_qcs), (
        "há valor não finito no vetor dos QCs: `nan` e `inf` não são JSON válido, e o cliente que "
        "confiasse em `list[float]` partiria na primeira feature indefinida"
    )


# **As contagens de indefinidos dos QCs nos quatro arquivos deste laboratório, sob
# `outlier_removed`** — 234, 212, 60 e 33 são números já medidos, e medi-los pela composição
# desta unidade é a âncora barata: o arnês é alimentado com algo cuja resposta já se conhece,
# **antes** de o resultado novo ser lido.
INDEFINIDAS_NOS_QCS: Final = {
    "pos_hilic": 234,
    "neg_hilic": 212,
    "pos_c18": 60,
    "neg_c18": 33,
}


@pytest.mark.parametrize("modo", sorted(INDEFINIDAS_NOS_QCS))
def test_u6_as_duas_populacoes_diferem_no_corpus_real_e_a_dos_qcs_bate_com_AR12(  # noqa: N802
    modo: str,
) -> None:
    """A medição sobre os arquivos reais, e ela é OBSERVAÇÃO sobre estes quatro.

    Medido em 17/09/2026, sobre a matriz **ingerida** de cada modo, sob `outlier_removed`:

    | Modo | features | finitas nos QCs | finitas nas amostras |
    |---|---|---|---|
    | `pos_hilic` | 6.035 | 5.801 | 5.795 |
    | `neg_hilic` | 2.039 | 1.827 | 1.790 |
    | `pos_c18` | 2.572 | 2.512 | 2.521 |
    | `neg_c18` | 2.184 | 2.151 | 2.106 |

    **As duas populações diferem nos quatro, e o SINAL não é constante:** as amostras têm menos
    features com RSD definido em três dos modos e mais no `pos_c18`. Uma contagem só para os dois
    lados declararia de uma população o que mediu da outra.

    **O que esta unidade NÃO faz, e vai declarado porque a leitura natural sugeriria o contrário:**
    ela **não separa as duas causas** de RSD indefinido. As 234 do `pos_hilic`
    são *"feature sem RSD definido nos QCs"* e nada mais — 102 delas por QC **todo** zero, e as
    outras porque a remoção de outliers apagou os QCs não nulos, que é feature detectada num único
    QC e não feature não detectada. Separá-las está em `Deferred to Follow-Up Work`.
    """
    matriz = load_matrix(measurements_path(modo), sample_list_path(modo), mode=modo)
    vetores = _rsd_por_feature(matriz, correcao="none", variante="outlier_removed")

    assert vetores.n_features == matriz.n_features, vetores.n_features
    assert vetores.n_injecoes_dos_qcs == int(matriz.qc_mask.sum()) > 0, (
        "âncora: o modo não tem QC nenhum, e a contagem abaixo mediria outra coisa"
    )
    assert vetores.n_features - vetores.n_features_finitas_dos_qcs == INDEFINIDAS_NOS_QCS[modo], (
        f"no {modo} sob `outlier_removed` medi "
        f"{vetores.n_features - vetores.n_features_finitas_dos_qcs} features sem RSD nos QCs, e o "
        f"número esperado para estes arquivos é {INDEFINIDAS_NOS_QCS[modo]}. Ou a "
        "composição mudou, ou os arquivos de referência mudaram"
    )
    assert vetores.n_features_finitas_dos_qcs != vetores.n_features_finitas_das_amostras, (
        f"no {modo} as duas populações saíram iguais ({vetores.n_features_finitas_dos_qcs}): a "
        "medição de 17/09/2026 diz que diferem nos quatro modos, e uma contagem só passaria a "
        "descrever as duas"
    )


def test_u6_sem_qc_o_vetor_vem_VAZIO_e_a_contagem_ZERO_com_o_outro_lado_intacto() -> None:  # noqa: N802
    """«Calculado e vazio» e não «não calculado», e a distinção é o que a figura desenha.

    Com zero QCs o lado dos QCs não tem sobre o que decidir, e a resposta honesta é a lista
    **vazia** com a contagem em **zero** — nunca `null`, que a figura leria como *"a carga não
    chegou"*, e nunca um vetor de `null` do comprimento das features, que faria a figura filtrar
    por conta própria num painel que está sob portão de aritmética com lista de permissão vazia.

    E `n_injecoes_dos_qcs` é o que separa os dois fatos que a contagem de finitos colapsaria:
    *"não há QC nenhum nesta matriz"* e *"há sete QCs e nenhuma feature com RSD
    definido"* levam a frases diferentes na imagem, e uma contagem só torna-os indistinguíveis.

    **Par causal:** o lado das amostras é **idêntico** ao do desfecho com QCs, o que torna o vazio
    atribuível à ausência de QC e não ao arnês.
    """
    corrigida = _matriz_para_relatorio()
    com_qc = _rsd_por_feature(corrigida, correcao="none", variante="outlier_removed")
    sem_qc = _rsd_por_feature(_sem_qc(corrigida), correcao="none", variante="outlier_removed")

    assert com_qc.rsd_dos_qcs, "âncora: o desfecho com QCs já vinha vazio, e o par não separaria"
    assert com_qc.n_injecoes_dos_qcs > 0, com_qc.n_injecoes_dos_qcs

    assert sem_qc.rsd_dos_qcs == [], (
        f"o lado dos QCs não veio vazio sem QC nenhum: {sem_qc.rsd_dos_qcs[:5]}..."
    )
    assert sem_qc.n_features_finitas_dos_qcs == 0, sem_qc.n_features_finitas_dos_qcs
    assert sem_qc.n_injecoes_dos_qcs == 0, sem_qc.n_injecoes_dos_qcs
    assert sem_qc.n_features == corrigida.n_features, (
        "o total de features mudou com a ausência de QC: o denominador não é do lado dos QCs"
    )
    assert sem_qc.rsd_das_amostras == com_qc.rsd_das_amostras, (
        "o lado das amostras mexeu-se quando só os QCs foram retirados: o vazio não é atribuível "
        "à ausência de QC"
    )


def test_u6_os_tres_rotulos_de_rsd_sao_recusados_na_fronteira_quando_invalidos() -> None:
    """`Literal` é análise estática, e o corpo que a janela lê vem do fio.

    `PipelineSpec` recusa `rsd_variant` inválido na construção da spec, e é por isso que o valor
    errado não chega aqui **por esta rota**. O que estes campos protegem é a outra fronteira: o
    corpo JSON, que nunca passou pelo mypy.

    **Os valores plantados não são lixo, e a escolha é deliberada.** `"outlier-removed"` com hífen é
    o erro de grafia natural; `"leave_one_qc_out_rsd"` é o nome da **função** e não do estimador — o
    mesmo que `RsdAntesEDepois` já recusa; e `"SERRF"` é o que o `tools.R` escreve. Um `str`
    aceitaria os três, e a imagem imprimiria um rótulo que o vocabulário não reconhece.

    **Par causal na mesma função:** os valores válidos atravessam e os inválidos são recusados. Sem
    a primeira metade, um esquema que recusasse **tudo** passaria na segunda.
    """
    corrigida = _matriz_para_relatorio()
    campos = _rsd_por_feature(corrigida, correcao="loess", variante="plain").model_dump()

    aceito = RsdPorFeature.model_validate(campos)
    assert (aceito.rsd_variant, aceito.rsd_estimator, aceito.corretor) == (
        "plain",
        "in_sample",
        "loess",
    ), (
        "âncora: os rótulos válidos não sobrevivem ao roundtrip, e a recusa abaixo "
        "mediria outra coisa"
    )

    for campo, invalido in (
        ("rsd_variant", "outlier-removed"),
        ("rsd_variant", "inventada"),
        ("rsd_estimator", "leave_one_qc_out_rsd"),
        ("corretor", "SERRF"),
    ):
        with pytest.raises(ValidationError):
            RsdPorFeature.model_validate({**campos, campo: invalido})


# ---------------------------------------------------------------------------
# A composição por classe clínica
# ---------------------------------------------------------------------------
#
# Em processo, pela razão dos blocos acima: o caminho ponta a ponta de
# `/execucoes` exige banco e esta máquina não o tem. A planta é argumento da agregação da rota.
#
# **Os quatro grupos clínicos do corpus — `Asymptomatic` 18, `HAMTSP` 14, `NoHTLV` 16, `ATLL` 13 —
# são OBSERVAÇÃO sobre estes quatro arquivos** e não invariante, e é por isso que o teste
# que os afirma nomeia o modo e usa os arquivos reais, nunca uma factory.

# A abundância medida do `pos_hilic`, e os TRÊS números do fator. Reproduzidos sobre a matriz
# ingerida, grupo `ATLL`: média de todas as células, média das detectadas, e a mediana das
# detectadas em unidades brutas. O fator entre as duas últimas é **63,7**.
ABUNDANCIA_DO_ATLL: Final = {
    "media_de_todas": 597_695.6,
    "media_das_detectadas": 715_660.3,
    "mediana_das_detectadas": 11_238.6,
}

# As quatro classes clínicas do `pos_hilic`, com as injeções de cada uma. Observação sobre estes
# arquivos: o estudo seguinte traz outros rótulos e outras contagens.
CLASSES_DO_POS_HILIC: Final = {
    "ATLL": 13,
    "Asymptomatic": 18,
    "HAMTSP": 14,
    "NoHTLV": 16,
}


def _com_dois_grupos(matriz: RefinoMatrix) -> RefinoMatrix:
    """A mesma matriz com as injeções de amostra repartidas em **dois** grupos clínicos.

    **`_matriz_para_relatorio` tem um grupo clínico só — `G1` —, e sobre ele metade do que esta
    unidade afirma seria vácuo**: *"uma entrada por grupo"* passaria com uma entrada, e *"a soma das
    células bate com o total"* passaria com a soma de uma parcela. A repartição é por paridade da
    ordem de injeção, que é determinística.
    """
    amostras = tuple(
        amostra
        if amostra.is_qc
        else replace(amostra, group="G1" if amostra.injection_order % 2 else "G2")
        for amostra in matriz.samples
    )
    return RefinoMatrix(
        values=matriz.values,
        feature_names=matriz.feature_names,
        samples=amostras,
        mode=matriz.mode,
    )


def _com_grupo_nao_detectado(matriz: RefinoMatrix) -> RefinoMatrix:
    """Zera **todas** as células de um dos dois grupos — o grupo sem nenhuma célula detectada.

    É a planta do desfecho que `log10_mediana_das_detectadas is None` existe para representar: um
    grupo em que o equipamento não detectou nada não tem abundância típica, e a média de um vetor
    vazio é `nan`.
    """
    com_dois = _com_dois_grupos(matriz)
    valores = com_dois.values.copy()
    valores[np.asarray(com_dois.groups) == "G2"] = 0.0
    return RefinoMatrix(
        values=valores,
        feature_names=com_dois.feature_names,
        samples=com_dois.samples,
        mode=com_dois.mode,
    )


def test_u8_a_composicao_traz_uma_entrada_por_grupo_e_as_CELULAS_somam_a_populacao() -> None:  # noqa: N802
    """**Uma entrada por classe declarada, e a soma das células bate com a população.**

    A âncora de mais de um grupo vem primeiro e na mesma função: com um grupo só, *"uma
    entrada por grupo"* e *"a soma bate"* são satisfeitas por uma parcela única, e o teste mediria
    a existência da lista em vez da partição.

    **A soma é contra `sample_mask * n_features` e não contra `matriz.values.size`**, porque a
    população exclui os QCs — e é precisamente essa diferença que o `n_injecoes_fora_da_populacao`
    existe para declarar na imagem.
    """
    matriz = _com_dois_grupos(_matriz_para_relatorio())
    composicao = _composicao_por_classe(matriz)

    rotulos = [classe.grupo for classe in composicao.classes]
    assert len(set(rotulos)) > 1, (
        f"a matriz de prova tem um grupo clínico só ({rotulos}), e as asserções abaixo passariam "
        "com uma parcela única"
    )
    assert rotulos == sorted(rotulos), (
        f"as classes não vieram em ordem alfabética: {rotulos}. Sem ordem, a figura muda a ordem "
        "das barras conforme a ordem das linhas do arquivo"
    )
    assert rotulos == sorted(set(np.asarray(matriz.groups)[matriz.sample_mask].tolist())), rotulos

    da_populacao = int(matriz.sample_mask.sum())
    assert sum(c.n_injecoes for c in composicao.classes) == da_populacao, (
        f"as injeções das classes somam {sum(c.n_injecoes for c in composicao.classes)} e a "
        f"população tem {da_populacao}"
    )
    assert sum(c.n_celulas for c in composicao.classes) == da_populacao * matriz.n_features, (
        "a soma das células por classe não bate com a população vezes o total de features"
    )
    assert composicao.n_features == matriz.n_features
    assert composicao.n_injecoes_fora_da_populacao == matriz.n_samples - da_populacao > 0, (
        "âncora: nenhuma injeção ficou fora da população, e o campo que a declara não teria o que "
        "declarar — a matriz de prova tem QCs"
    )


def test_u8_o_QC_fica_FORA_da_populacao_e_a_planta_no_qc_nao_move_classe_nenhuma() -> None:  # noqa: N802
    """**A população é `sample_mask`, e o par causal cai sobre dado em que as duas coisas diferem.**

    Nos quatro arquivos deste laboratório o `group` das injeções de QC é a cadeia `"QC"` —
    `readers._resolve_group` devolve `QC_LABEL` quando o `LABEL` é o de QC —, então agrupar por
    `groups` **sem** máscara poria uma barra `QC` entre as clínicas. Isso é observação; a máscara é
    desenho, porque `SampleType` tem três valores.

    **A planta zera as linhas de QC inteiras**, e o que se afirma é que **nada** na composição se
    move. A âncora de que a planta aterrou vem primeiro: as duas matrizes têm de diferir nos
    valores, senão a igualdade abaixo seria entre duas leituras do mesmo objeto.
    """
    matriz = _com_dois_grupos(_matriz_para_relatorio())
    valores = matriz.values.copy()
    valores[matriz.qc_mask] = 0.0
    com_qc_zerado = RefinoMatrix(
        values=valores,
        feature_names=matriz.feature_names,
        samples=matriz.samples,
        mode=matriz.mode,
    )

    assert not np.array_equal(matriz.values, com_qc_zerado.values), (
        "âncora: a planta não aterrou — as duas matrizes têm valores idênticos"
    )
    assert "QC" in set(matriz.groups), (
        "âncora: nenhuma injeção tem o rótulo `QC`, e a asserção de que ele não vira classe seria "
        "vácua"
    )

    intacta = _composicao_por_classe(matriz)
    plantada = _composicao_por_classe(com_qc_zerado)

    assert "QC" not in {c.grupo for c in intacta.classes}, (
        f"o QC virou classe clínica: {[c.grupo for c in intacta.classes]}. Ele é uma mistura de "
        "TODAS as amostras, e a composição dele é uma média ponderada das outras"
    )
    assert intacta.populacao == "sample", intacta.populacao
    assert plantada.model_dump() == intacta.model_dump(), (
        "zerar as linhas de QUALIDADE mudou a composição por classe: a população não é "
        "`sample_mask`"
    )


def test_u8_com_ZERO_grupos_a_lista_vem_VAZIA_e_nunca_nula() -> None:  # noqa: N802
    """«Calculado e vazio» e não «não calculado», e a figura declara-o em texto.

    Uma matriz só de QCs produz zero classes clínicas, e isso é um fato sobre o estudo. `None` ali
    diria *"a carga não chegou"*, que é a leitura oposta.

    **Par causal:** com amostras a lista tem entradas; sem amostras vem vazia — e o
    `n_injecoes_fora_da_populacao` passa a ser o total, que é o que torna o vazio explicável.
    """
    matriz = _com_dois_grupos(_matriz_para_relatorio())
    com_amostras = _composicao_por_classe(matriz)
    assert com_amostras.classes, "âncora: a matriz de prova já não tinha classe nenhuma"

    so_qcs = RefinoMatrix(
        values=matriz.values[matriz.qc_mask],
        feature_names=matriz.feature_names,
        samples=tuple(a for a in matriz.samples if a.is_qc),
        mode=matriz.mode,
    )
    sem_classe = _composicao_por_classe(so_qcs)
    assert sem_classe.classes == [], [c.grupo for c in sem_classe.classes]
    assert sem_classe.n_injecoes_fora_da_populacao == so_qcs.n_samples > 0, (
        "com zero amostras, todas as injeções têm de estar declaradas como fora da população"
    )
    assert sem_classe.n_features == so_qcs.n_features, sem_classe.n_features


def test_u8_um_grupo_sem_celula_detectada_traz_None_e_nunca_nan() -> None:  # noqa: N802
    """`None` e nunca `nan`: o painel de abundância não pode desenhar zero.

    **Par causal:** com o grupo detectado a abundância é um número; com ele todo zerado é `None` — e
    o outro grupo fica **idêntico**, o que torna a ausência atribuível à planta e não ao arnês.

    **`nan` seria pior que errado.** O `jsonable_encoder` do FastAPI converte-o em `null` apesar de
    o campo declarar `float`, então o OpenAPI prometeria número onde chega `null`; e um `float(v or
    0)` do outro lado desenharia uma barra colada ao eixo, que se lê como *"abundância mínima"* — o
    fato oposto de *"não se detectou nada"*.
    """
    intacta = _composicao_por_classe(_com_dois_grupos(_matriz_para_relatorio()))
    plantada = _composicao_por_classe(_com_grupo_nao_detectado(_matriz_para_relatorio()))

    por_grupo = {c.grupo: c for c in intacta.classes}
    plantado = {c.grupo: c for c in plantada.classes}
    assert set(por_grupo) == set(plantado) == {"G1", "G2"}, (sorted(por_grupo), sorted(plantado))
    assert por_grupo["G2"].log10_mediana_das_detectadas is not None, (
        "âncora: o grupo já vinha sem abundância antes da planta, e a ausência não seria atribuível"
    )
    assert por_grupo["G2"].n_detectadas > 0, por_grupo["G2"].n_detectadas

    assert plantado["G2"].log10_mediana_das_detectadas is None, (
        f"um grupo sem célula detectada devolveu {plantado['G2'].log10_mediana_das_detectadas!r} "
        "em vez de `None`"
    )
    assert plantado["G2"].n_detectadas == 0, plantado["G2"].n_detectadas
    assert plantado["G2"].n_faltantes == plantado["G2"].n_celulas, plantado["G2"]
    assert plantado["G1"].model_dump() == por_grupo["G1"].model_dump(), (
        "o outro grupo mexeu-se com uma planta que só tocou o `G2`: a ausência acima não é "
        "atribuível"
    )


def test_u8_as_TRES_candidatas_de_faltante_e_a_uniao_e_a_soma_das_outras_duas() -> None:  # noqa: N802
    """**A união é DESENHO e as três contagens são conferíveis** — `is_missing` é `zero | ~finita`.

    Uma célula não pode ser `0.0` **e** não finita ao mesmo tempo, então `n_faltantes` é a soma
    exacta das outras duas. Isso é invariante de desenho e não observação, e afirmá-lo é o que
    impede um `n_faltantes` derivado de um terceiro predicado de divergir em silêncio.

    **A planta cai sobre dado em que as três DIFEREM**, que é o que a matriz limpa não dá: sem um
    `nan`, `n_zeros == n_faltantes` e a igualdade acima seria satisfeita por `n_nao_finitas == 0`.
    """
    matriz = _com_dois_grupos(_matriz_para_relatorio())
    limpa = _composicao_por_classe(matriz)
    assert all(c.n_nao_finitas == 0 for c in limpa.classes), (
        "a matriz de prova já tem célula não finita, e a planta abaixo não seria atribuível"
    )
    assert all(c.n_zeros == c.n_faltantes for c in limpa.classes), (
        "âncora: sem não finita as três candidatas têm de coincidir, e não coincidiram"
    )

    valores = matriz.values.copy()
    linhas = np.flatnonzero(matriz.sample_mask)
    valores[linhas[0], 0] = np.nan
    valores[linhas[0], 1] = 0.0
    plantada = _composicao_por_classe(
        RefinoMatrix(
            values=valores,
            feature_names=matriz.feature_names,
            samples=matriz.samples,
            mode=matriz.mode,
        )
    )

    tocado = next(c for c in plantada.classes if c.n_nao_finitas)
    assert (tocado.n_zeros, tocado.n_nao_finitas) == (1, 1), tocado
    assert tocado.n_faltantes == tocado.n_zeros + tocado.n_nao_finitas, tocado
    assert tocado.n_faltantes != tocado.n_zeros, (
        "com uma célula não finita as três candidatas têm de divergir, e `n_faltantes` saiu igual "
        "a `n_zeros` — a união não é `is_missing`"
    )
    assert tocado.n_detectadas == tocado.n_celulas - tocado.n_faltantes, (
        "nesta matriz não há célula finita negativa, então as duas contagens têm de fechar"
    )


def test_u8_a_mediana_do_log_das_detectadas_DIVERGE_da_media_de_todas_no_corpus_real() -> None:  # noqa: N802
    """Os três números e o fator 63,7, reproduzidos e não citados.

    Medido em 17/09/2026 sobre a matriz **ingerida** do `pos_hilic`, grupo `ATLL`:

    | Candidata | Valor |
    |---|---|
    | média de **todas** as células | 597.695,6 |
    | média das **detectadas** | 715.660,3 |
    | **mediana** das detectadas | 11.238,6 |

    **O par causal usa as duas e afirma que divergem** — fator **63,7** entre a média das detectadas
    e a mediana delas. A média é governada pela cauda, e por isso o painel construído sobre ela sai
    **plano**: os quatro grupos deste modo dão 597.695,6 / 563.549,4 / 705.344,8 / 621.592,1, num
    alcance de 1,25x contra 1,50x das medianas.

    **O que a função devolve é o LOG**, e a comparação é feita em unidades brutas por `10 ** v` —
    monótona, então a mediana do log é o log da mediana.
    """
    modo = "pos_hilic"
    matriz = load_matrix(measurements_path(modo), sample_list_path(modo), mode=modo)
    composicao = _composicao_por_classe(matriz)

    assert {c.grupo: c.n_injecoes for c in composicao.classes} == CLASSES_DO_POS_HILIC, (
        f"as classes do {modo} saíram {[(c.grupo, c.n_injecoes) for c in composicao.classes]}, e a "
        f"observação de 17/09/2026 sobre estes arquivos é {CLASSES_DO_POS_HILIC}"
    )

    atll = next(c for c in composicao.classes if c.grupo == "ATLL")
    bruto = np.asarray(matriz.values)[np.asarray(matriz.groups) == "ATLL"]
    detectadas = bruto[bruto > 0.0]
    assert atll.log10_mediana_das_detectadas is not None, atll

    media_de_todas = float(bruto.mean())
    media_das_detectadas = float(detectadas.mean())
    mediana_das_detectadas = 10.0**atll.log10_mediana_das_detectadas

    assert media_de_todas == pytest.approx(ABUNDANCIA_DO_ATLL["media_de_todas"], rel=1e-6)
    assert media_das_detectadas == pytest.approx(
        ABUNDANCIA_DO_ATLL["media_das_detectadas"], rel=1e-6
    )
    assert mediana_das_detectadas == pytest.approx(
        ABUNDANCIA_DO_ATLL["mediana_das_detectadas"], rel=1e-5
    )
    assert media_das_detectadas / mediana_das_detectadas > 50.0, (
        f"o fator entre a média das detectadas e a mediana delas caiu para "
        f"{media_das_detectadas / mediana_das_detectadas:.1f}: KTD8 mediu 63,7, e é ele que torna "
        "a escolha da mediana uma decisão e não gosto"
    )

    # **A consequência da escolha, medida:** com as médias o painel é plano e com as medianas não.
    medias = [
        float(np.asarray(matriz.values)[np.asarray(matriz.groups) == c.grupo].mean())
        for c in composicao.classes
    ]
    medianas = [10.0**c.log10_mediana_das_detectadas for c in composicao.classes]
    assert all(m is not None for m in medianas), medianas
    assert max(medias) / min(medias) < max(medianas) / min(medianas), (
        f"as médias espalham-se {max(medias) / min(medias):.2f}x e as medianas "
        f"{max(medianas) / min(medianas):.2f}x: a média NÃO é a que achata o painel neste corpus, "
        "e o argumento de KTD8 precisa de ser remedido"
    )


def test_u8_a_matriz_da_composicao_e_a_INGERIDA_e_o_rotulo_diz_isso() -> None:  # noqa: N802
    """**O escopo, e ele é o oposto do das três cargas irmãs** — elas medem a correção.

    **O custo de medir na corrigida está medido** (17/09/2026, `pos_hilic`, `batch_loess`, semente
    42): a corrigida tem 73.703 células faltantes contra 64.780 na ingerida — **8.923 criadas pelo
    corretor**, 5.431 zeros novos e 3.492 não finitas. A figura atribuiria à aquisição o que foi do
    corretor.

    Aqui o par causal é barato e é o que fecha a escolha: **a mesma matriz com um zero plantado**
    muda a composição, o que prova que a contagem sai dos valores que se passam — e a rota passa
    `matriz`, não `corrigida.matrix`. O rótulo `"ingerida"` é o que a figura imprime.
    """
    matriz = _com_dois_grupos(_matriz_para_relatorio())
    composicao = _composicao_por_classe(matriz)
    assert composicao.matriz == "ingerida", (
        f"a procedência saiu {composicao.matriz!r}; a composição mede o ESTUDO e não a correção, e "
        "o rótulo é o que a imagem imprime"
    )
    assert composicao.definicao_de_faltante == "zero_ou_nao_finita", composicao
    assert composicao.definicao_de_abundancia == "mediana_do_log10_das_detectadas", composicao

    valores = matriz.values.copy()
    valores[np.flatnonzero(matriz.sample_mask)[0], 0] = 0.0
    outra = _composicao_por_classe(
        RefinoMatrix(
            values=valores,
            feature_names=matriz.feature_names,
            samples=matriz.samples,
            mode=matriz.mode,
        )
    )
    assert outra.model_dump() != composicao.model_dump(), (
        "plantar um zero não mudou a composição: as contagens não saem da matriz que se passa"
    )


def test_u8_a_populacao_e_as_duas_definicoes_sao_recusadas_na_fronteira() -> None:
    """`Literal` é análise estática, e o corpo que a janela lê vem do fio.

    **Os valores plantados não são lixo.** `"todas"` é o que alguém escreveria a querer dizer
    *"sem máscara"*; `"zero"` e `"mediana_do_log10"` são os dois tokens **truncados**, que é a
    forma que um refactor descuidado produz — e é a forma que a figura traduziria para a frase
    errada se o campo fosse `str`.

    **Par causal na mesma função:** os valores válidos atravessam e os inválidos são recusados. Sem
    a primeira metade, um esquema que recusasse tudo passaria na segunda.

    > **ESCOPO NEGATIVO, e ele nasce de um defeito que este teste apanhou.** A primeira escrita
    > exigia que `populacao="qc"` fosse **recusado**, e ele não é: `"qc"` é valor legítimo de
    > `SampleType`, e a guarda de fronteira existe para recusar o que está **fora do vocabulário**,
    > não para afirmar qual dos três a rota escolheu. É o mesmo regime de `RsdPorFeature.corretor`,
    > que aceita os três corretores. Quem afirma que a população é `sample` é
    > `test_u8_o_QC_fica_FORA_da_populacao_...`, medindo a agregação — e não o esquema.
    """
    campos = _composicao_por_classe(_com_dois_grupos(_matriz_para_relatorio())).model_dump()

    aceito = ComposicaoPorClasse.model_validate(campos)
    assert (aceito.populacao, aceito.definicao_de_faltante, aceito.definicao_de_abundancia) == (
        "sample",
        "zero_ou_nao_finita",
        "mediana_do_log10_das_detectadas",
    ), (
        "âncora: os valores válidos não sobrevivem ao roundtrip, e a recusa abaixo mediria outra "
        "coisa"
    )

    for campo, invalido in (
        ("populacao", "todas"),
        ("populacao", "amostra"),
        ("definicao_de_faltante", "zero"),
        ("definicao_de_abundancia", "mediana_do_log10"),
    ):
        with pytest.raises(ValidationError):
            ComposicaoPorClasse.model_validate({**campos, campo: invalido})

    # **O que a fronteira NÃO guarda, afirmado para que a docstring acima não seja só prosa.** Um
    # `SampleType` diferente do que a rota usa atravessa, porque está no vocabulário — a guarda
    # recusa o que não é `SampleType`, e não o que é o `SampleType` errado.
    outro = ComposicaoPorClasse.model_validate({**campos, "populacao": "qc"})
    assert outro.populacao == "qc", outro.populacao


@pytest.mark.parametrize(
    ("rota", "enviado", "esperado"),
    (
        ("/matrizes/corrigir", "true", -1),
        ("/matrizes/corrigir", "false", 1),
        ("/matrizes/processar", "false", 1),
        ("/matrizes/exportar", "false", 1),
    ),
)
def test_a_caixa_dos_nucleos_atravessa_o_HTTP_ate_o_corretor(  # noqa: N802
    cliente: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    rota: str,
    enviado: str,
    esperado: int,
) -> None:
    """`paralelo` sai do formulário multipart e chega a `SerrfCorrector.n_jobs`.

    **A revisão de código de 08/09/2026 mediu que NENHUM teste fazia esta travessia.** Havia par
    causal para `n_jobs_efetivo` (em processo), para a fábrica (`_corretor_ou_422`) e para o payload
    da janela — e nada que atravessasse a fronteira do `multipart`. Ela tem uma conversão que
    ninguém nosso escreve: a string `"false"` tem de virar `False` antes de a fábrica a ver.

    **O `serrf` e não o `loess`, e a assimetria é o conteúdo:** só o SERRF usa `joblib`. Um teste
    que espiasse o `loess` não distinguiria os dois regimes, porque não há regime nenhum ali.
    """
    from refino.api import rotas

    original = rotas._corretor_ou_422
    vistos: list[int] = []

    def espia(method: str, null_denominator: str, **kwds: Any) -> Any:
        corretor = original(method, null_denominator, **kwds)
        if method == "serrf":
            vistos.append(int(corretor.n_jobs))
        return corretor

    monkeypatch.setattr(rotas, "_corretor_ou_422", espia)
    _dispara(cliente, rota, {"method": "serrf", "paralelo": enviado})

    assert vistos, (
        f"`{rota}` não construiu SERRF nenhum com `method=serrf`: o espião não viu a fábrica, e a "
        "asserção abaixo passaria por vacuidade"
    )
    assert vistos == [esperado], (
        f"`{rota}` com `paralelo={enviado}` construiu o SERRF com `n_jobs={vistos}`, e devia ser "
        f"`[{esperado}]`. A caixa «usar vários núcleos» não atravessa a fronteira do multipart"
    )

"""As prévias do passo a passo.

O oráculo é o OUTRO caminho: o `POST /execucoes` com os mesmos parâmetros. As prévias são
sem estado e não gravam nada; o que liga as etapas é o cache de correção por
conteúdo, e o contador de acertos dele é o que mostra que a etapa seguinte não corrigiu de novo.

O banco é um SQLite temporário migrado por `refino.desktop.banco.migra`, o mesmo caminho da janela
e o banco da entrega.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from refino.api import rotas
from refino.api.app import build_app
from refino.api.progresso import RegistroDeProgresso
from refino.api.reuso import MemoriaDeCorrecoes
from refino.desktop.banco import migra
from refino.persistence.config import VARIAVEL_DA_URL
from tests.conftest import encerra_trabalhos_vivos, measurements_path, sample_list_path
from tests.reclamacao import espera, reclama

pytestmark = pytest.mark.integration

MODO = "neg_c18"
# Os campos da etapa 1 e os da etapa 2, e a etapa 3 manda os dois mais os do registro. LOESS e não
# SERRF: 13 s contra 80 s no `neg_c18`, e a paridade é sobre o caminho e não sobre o corretor.
CORRECAO = {"mode": MODO, "correction": "loess", "n_batches": 1, "rsd_variant": "outlier_removed"}
PREPROCESSAMENTO = {
    "imputation": "one_fifth_min",
    "sample_normalization": "none",
    "transformation": "log10",
    "scaling": "pareto",
}
REGISTRO = {"estudo": "passo_a_passo", "apresentacao": "exibida"}
DA_CORRECAO = ("rsd_antes_e_depois", "pcas_da_correcao", "rsd_por_feature", "composicao_por_classe")


@pytest.fixture
def cliente(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    url = f"sqlite:///{(tmp_path / 'refino.db').as_posix()}"
    migra(url)
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    yield TestClient(build_app())


@pytest.fixture(autouse=True)
def registro_novo(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Um registro de trabalhos NOVO por teste: o do processo deduplica pedidos idênticos, e a
    etapa 1 de um teste devolveria o trabalho da etapa 1 do anterior sem correr nada.

    E os trabalhos vivos são encerrados AQUI, antes de o registro voltar. A teardown
    `conftest._sem_trabalho_vivo_no_fim_do_teste` corre depois do `monkeypatch` desfazer a
    troca, e pede o cancelamento ao registro do PROCESSO, onde os trabalhos deste teste não
    estão: o pedido não chegava, e um trabalho que sobrasse corria dentro do teste seguinte e
    gravava no cache dele. Observado: `test_cancelar_a_etapa_devolve_410` reprovou
    em `len(memoria) == 0`, com o aviso de um `KeyError` de um trabalho da etapa 1 que o registro
    daquele teste não conhecia — o LOESS que `test_a_etapa_1_nao_recusa...` deixava a correr.
    """
    monkeypatch.setattr(rotas, "TRABALHOS", RegistroDeProgresso())
    yield
    encerra_trabalhos_vivos(rotas)


@pytest.fixture
def memoria(monkeypatch: pytest.MonkeyPatch) -> MemoriaDeCorrecoes:
    """Um cache de correção NOVO: o do processo é partilhado, e outro teste pode já ter guardado a
    mesma correção — o acerto contado seria dele e não da etapa."""
    nova = MemoriaDeCorrecoes()
    monkeypatch.setattr(rotas, "CORRECOES", nova)
    return nova


def _arquivos() -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{MODO}.csv",
            measurements_path(MODO).read_bytes(),
            "text/csv",
        ),
        "sample_list": (f"Sample_List_{MODO}.csv", sample_list_path(MODO).read_bytes(), "text/csv"),
    }


def _linhas(tabela: str) -> int:
    motor = create_engine(os.environ[VARIAVEL_DA_URL])
    try:
        with motor.connect() as conexao:
            return int(conexao.execute(text(f"SELECT COUNT(*) FROM {tabela}")).scalar_one())
    finally:
        motor.dispose()


def test_as_tres_etapas_dao_as_figuras_da_execucao_gravam_uma_vez_e_corrigem_uma_vez(
    cliente: TestClient, memoria: MemoriaDeCorrecoes
) -> None:
    """R4, R5 e R9 pelo caminho real, com o `/execucoes` como oráculo.

    - os resumos da etapa 1 e da etapa 2 são os que a execução devolve com os mesmos parâmetros;
    - as prévias não gravam nada, e a etapa 3 grava uma execução;
    - a correção corre uma vez: as etapas 2 e 3 acertam o cache.
    """
    etapa_1 = reclama(cliente, "/etapas/correcao", files=_arquivos(), data=CORRECAO)
    assert etapa_1.status_code == 200, etapa_1.text
    corpo_1 = etapa_1.json()
    assert corpo_1["correcao"] == "loess"
    assert (memoria.acertos, memoria.erros) == (0, 1), "ancora: a etapa 1 tinha de corrigir"

    dados_2 = {**CORRECAO, **PREPROCESSAMENTO}
    etapa_2 = reclama(cliente, "/etapas/pre-processamento", files=_arquivos(), data=dados_2)
    assert etapa_2.status_code == 200, etapa_2.text
    corpo_2 = etapa_2.json()
    assert memoria.acertos == 1, "a etapa 2 corrigiu de novo em vez de usar o cache"
    assert _linhas("runs") == 0, "uma previa gravou no banco"

    dados_3 = {**CORRECAO, **PREPROCESSAMENTO, **REGISTRO}
    etapa_3 = reclama(cliente, "/execucoes", files=_arquivos(), data=dados_3)
    assert etapa_3.status_code == 200, etapa_3.text
    corpo_3 = etapa_3.json()
    assert memoria.acertos == 2, "a etapa 3 corrigiu de novo em vez de usar o cache"
    assert _linhas("runs") == 1

    for campo in DA_CORRECAO:
        assert corpo_1[campo] == corpo_3[campo], f"etapa 1 e execucao divergem em {campo}"
        assert corpo_2[campo] == corpo_3[campo], f"etapa 2 e execucao divergem em {campo}"
    assert corpo_2["distribuicoes"] == corpo_3["distribuicoes"]
    assert corpo_2["n_features_saida"] == corpo_3["n_features_saida"]


def test_a_mudanca_do_corretor_corrige_de_novo_e_a_da_imputacao_nao(
    cliente: TestClient, memoria: MemoriaDeCorrecoes
) -> None:
    """Par causal do cache: só o eixo da correção faz a etapa seguinte corrigir outra vez."""
    reclama(cliente, "/etapas/correcao", files=_arquivos(), data=CORRECAO)
    # Outra CORREÇÃO e não outro corretor qualquer: o `none` não passa pelo cache, porque não há
    # o que guardar. Dois lotes em vez de um são outra correção do mesmo LOESS.
    outra = {**CORRECAO, "n_batches": 2}
    dados = {**outra, **PREPROCESSAMENTO}
    reclama(cliente, "/etapas/pre-processamento", files=_arquivos(), data=dados)
    assert (memoria.acertos, memoria.erros) == (0, 2), "outro corretor acertou o cache do primeiro"

    mediana = {**outra, **PREPROCESSAMENTO, "imputation": "median"}
    reclama(cliente, "/etapas/pre-processamento", files=_arquivos(), data=mediana)
    assert (memoria.acertos, memoria.erros) == (1, 2), "mudar so a imputacao corrigiu de novo"


@pytest.mark.parametrize(
    ("rota", "dados", "trecho"),
    [
        ("/etapas/correcao", {**CORRECAO, "correction": "inexistente"}, "inexistente"),
        ("/etapas/correcao", {**CORRECAO, "rsd_variant": "outra"}, "outra"),
        (
            "/etapas/pre-processamento",
            {**CORRECAO, **PREPROCESSAMENTO, "imputation": "inexistente"},
            "inexistente",
        ),
        (
            "/etapas/pre-processamento",
            {**CORRECAO, **PREPROCESSAMENTO, "imputation": "none", "sample_normalization": "sum"},
            "escolha um metodo de imputacao",
        ),
    ],
)
def test_parametro_invalido_recusa_no_disparo(
    cliente: TestClient, rota: str, dados: dict[str, Any], trecho: str
) -> None:
    """A recusa é `422` no disparo, antes de o trabalho nascer — como no `/execucoes`."""
    resposta = cliente.post(rota, files=_arquivos(), data=dados)
    assert resposta.status_code == 422, resposta.text
    assert trecho in resposta.text


def test_a_etapa_1_nao_recusa_por_escolha_que_so_a_etapa_2_usa(cliente: TestClient) -> None:
    """A combinação proibida de escolhas é do pré-processamento, e a etapa 1 não a recebe.

    **Reclamado até o fim, e com o `none`**: o disparo sozinho deixava o trabalho a correr para
    dentro do teste seguinte (ver `registro_novo`), e o `none` termina em segundos.
    """
    resposta = reclama(
        cliente,
        "/etapas/correcao",
        files=_arquivos(),
        data={
            **CORRECAO,
            "correction": "none",
            "imputation": "none",
            "sample_normalization": "sum",
        },
    )
    assert resposta.status_code == 200, resposta.text


def test_cancelar_a_etapa_devolve_410(cliente: TestClient, memoria: MemoriaDeCorrecoes) -> None:
    """A parada das rotas longas alcança as prévias: elas são trabalhos como os outros.

    SERRF e não LOESS: o LOESS do `neg_c18` termina antes de o pedido de cancelamento chegar.
    """
    serrf = {**CORRECAO, "correction": "serrf"}
    disparo = cliente.post("/etapas/correcao", files=_arquivos(), data=serrf)
    assert disparo.status_code == 202, disparo.text
    identificador = disparo.json()["identificador"]
    cancelamento = cliente.post(f"/trabalhos/{identificador}/cancelar")
    if cancelamento.status_code == 409:
        pytest.fail("a previa terminou antes de o cancelamento chegar; o teste nao mediu nada")
    assert cancelamento.status_code == 202, cancelamento.text
    assert espera(cliente, identificador).status_code == 410
    assert len(memoria) == 0, "a correcao cancelada foi guardada no cache"
    # **A âncora do `== 0`, nesta função (refino-review de 15/10/2026):** sem ela, um `memoria` que
    # não fosse o cache que a rota escreve passaria a asserção acima por vacuidade. Uma correção que
    # termina, no mesmo cache, tem de ficar guardada.
    terminada = reclama(cliente, "/etapas/correcao", files=_arquivos(), data=CORRECAO)
    assert terminada.status_code == 200, terminada.text
    assert len(memoria) == 1, "ancora: a correcao terminada nao foi guardada no cache da rota"


@pytest.mark.slow
def test_com_o_serrf_a_etapa_3_acerta_o_cache_da_1_e_da_os_mesmos_resumos(
    cliente: TestClient, memoria: MemoriaDeCorrecoes
) -> None:
    """A paridade com o corretor que motivou o passo a passo.

    Aqui a caixa «Usar vários núcleos» vai marcada, e ela entra no corretor e não na chave do
    cache. Se ela entrasse, a etapa 3 não acharia a correção da 1 e o SERRF correria duas vezes —
    que é o custo que o passo a passo existe para não ter.

    O LOESS também lê a caixa, que no teste de cima fica no padrão da rota.
    """
    serrf = {**CORRECAO, "correction": "serrf", "paralelo": "true"}
    etapa_1 = reclama(cliente, "/etapas/correcao", files=_arquivos(), data=serrf)
    assert etapa_1.status_code == 200, etapa_1.text
    assert (memoria.acertos, memoria.erros) == (0, 1), "ancora: a etapa 1 tinha de corrigir"

    dados = {**serrf, **PREPROCESSAMENTO, **REGISTRO}
    etapa_3 = reclama(cliente, "/execucoes", files=_arquivos(), data=dados)
    assert etapa_3.status_code == 200, etapa_3.text
    assert (memoria.acertos, memoria.erros) == (1, 1), "a etapa 3 corrigiu o SERRF de novo"
    for campo in DA_CORRECAO:
        assert etapa_1.json()[campo] == etapa_3.json()[campo], campo


def test_com_o_automatico_as_tres_etapas_delegam_o_mesmo_corretor_e_acertam_o_cache(
    cliente: TestClient, memoria: MemoriaDeCorrecoes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A delegação `auto` das prévias, que antes não era exercitada.

    A caixa «Automático» nasce ligada na janela, então este é o caminho PADRÃO do usuário, e
    os outros testes deste arquivo mandam o corretor à mão. `_delega_na_spec` reescreve o corretor
    antes de formar a chave do cache: se a delegação da prévia e a do `/execucoes` divergissem, a
    etapa 3 erraria o cache e corrigiria de novo. O rótulo é o que a fronteira aceita para a
    caixa «Automático» (`cli.AUTO_MEDIDA`), e não um escrito aqui.

    O Palpite medido entra por um duplo. A caixa delega pela medição, e sem
    o duplo este teste, fora do `slow`, mediria os três corretores em 200 features antes da primeira
    etapa — o que o teste `slow` de `tests/unit/test_palpite_medido.py` já faz. O duplo decide pela
    regra do produto com o `loess` à frente, e a memória dos Palpites é deste teste: o que se mede
    aqui é o cache, e qualquer corretor que não seja `none` passa por ele.
    """
    from refino.cli import AUTO_MEDIDA
    from refino.decision.palpite import decide_o_palpite

    medidos: list[str] = []

    def medidor(matriz: Any, **_kwargs: Any) -> Any:
        medidos.append(str(matriz.mode))
        return decide_o_palpite({"none": 0.3, "loess": 0.1, "serrf": 0.2}, n_features_medidas=200)

    monkeypatch.setattr(rotas, "mede_o_palpite", medidor)
    monkeypatch.setattr(rotas, "_PALPITES", type(rotas._PALPITES)())

    auto = {**CORRECAO, "auto": AUTO_MEDIDA}

    disparo = cliente.post("/etapas/correcao", files=_arquivos(), data=auto)
    assert disparo.status_code == 202, disparo.text
    delegacao = disparo.json()["delegacao"]
    assert delegacao is not None, "o 202 da previa nao declarou a delegacao"
    etapa_1 = espera(cliente, disparo.json()["identificador"])
    assert etapa_1.status_code == 200, etapa_1.text
    corretor = etapa_1.json()["correcao"]
    assert corretor == delegacao["correction"], "a previa corrigiu com outro corretor"
    if corretor == "none":
        pytest.fail(
            f"a politica escolheu `none` no {MODO}, que nao passa pelo cache: o teste nao mede "
            "o acerto e precisa de outro modo"
        )
    assert (memoria.acertos, memoria.erros) == (0, 1), "ancora: a etapa 1 tinha de corrigir"

    dados_2 = {**auto, **PREPROCESSAMENTO}
    etapa_2 = reclama(cliente, "/etapas/pre-processamento", files=_arquivos(), data=dados_2)
    assert etapa_2.status_code == 200, etapa_2.text
    assert etapa_2.json()["correcao"] == corretor
    assert memoria.acertos == 1, "com o automatico, a etapa 2 corrigiu de novo"

    dados_3 = {**auto, **PREPROCESSAMENTO, **REGISTRO}
    etapa_3 = reclama(cliente, "/execucoes", files=_arquivos(), data=dados_3)
    assert etapa_3.status_code == 200, etapa_3.text
    assert (memoria.acertos, memoria.erros) == (2, 1), (
        "com o automatico, a etapa 3 corrigiu de novo"
    )
    for campo in DA_CORRECAO:
        assert etapa_1.json()[campo] == etapa_3.json()[campo], campo
    assert medidos == [MODO], (
        f"a delegação mediu o Palpite {len(medidos)} vez(es): ele é medido na primeira etapa e "
        "lido da memória nas outras duas"
    )

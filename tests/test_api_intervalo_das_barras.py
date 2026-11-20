"""O intervalo por barra na rota da medição dos corretores, por escolha.

O intervalo liga por escolha: uma caixa desmarcada por padrão, e quem não a marca não
paga nada. Na rota isso é um campo de pedido, `intervalo_por_barra`, falso por padrão, e um
campo de resposta, `intervalos`, `null` quando não pedido. Este arquivo afirma:

- **sem o campo, nenhum ajuste a mais** — conta-se o corretor: `n` por corretor, e `n(n+1)/2`
  com ele;
- **com o campo, um intervalo por corretor, e a eleição é a mesma** — escores, vencedor, margem e
  abstenção idênticos aos do pedido sem ele;
- **o pedido com o intervalo não é reclamado pelo trabalho sem ele** — a chave do trabalho muda;
- **o campo com tipo errado é `422` no disparo**;
- **lote com três QCs dá a barra e o motivo do intervalo ao `none`**, e não um número, e deixa o
  LOESS sem barra, com o motivo da recusa no intervalo dele.

**A matriz é da factory, e não um arquivo do laboratório** — 40 injeções, oito QCs em 1, 6, 11, 16,
21, 26, 31 e 36 —, e os corretores são servidos leves: `none` ele mesmo, e o LOESS em série no
lugar dos que ajustam. O que se afirma é a fiação e a conta, e não o SERRF nem número de estudo
nenhum.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.decision import labeling
from refino.domain.correction import CorrectionResult
from refino.domain.matrix import RefinoMatrix
from refino.metrics.crossval import JACKKNIFE_LEVEL, JACKKNIFE_METHOD
from tests.factories import base_spec, render_pair
from tests.reclamacao import reclama

# **Memória fria a cada teste**: o registro de trabalhos é do processo, e um trabalho de um teste
# anterior com os mesmos arquivos seria RECLAMADO em vez de medido — a contagem de ajustes
# mediria o resíduo.
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("memoria_de_correcoes_privada")]

MODO = "neg_hilic"
ROTA = "/matrizes/medir-corretor"
N_QC = 8


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


class _Contador:
    def __init__(self, interno: Any) -> None:
        self.interno = interno
        self.name = interno.name
        self.n_jobs = 1
        # Repassado: o estimador lê o mínimo de treino do corretor por atributo (o LOESS exige
        # três), e um envoltório que o escondesse mediria o piso em vez do do corretor que envolve.
        if hasattr(interno, "min_train_qc_per_batch"):
            self.min_train_qc_per_batch = interno.min_train_qc_per_batch
        self.chamadas = 0

    def fit_transform(self, matrix: RefinoMatrix, **kwargs: Any) -> CorrectionResult:
        self.chamadas += 1
        return self.interno.fit_transform(matrix, **kwargs)  # type: ignore[no-any-return]


@pytest.fixture
def corretores(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[_Contador]]:
    """Os corretores que o oráculo da rota constrói, contados — um por pedido e por prefixo."""
    construidos: dict[str, list[_Contador]] = {"none": [], "loess": [], "serrf": []}

    def constroi(nome: str, *, n_jobs: int = -1) -> Any:
        del n_jobs
        contador = _Contador(NoneCorrector() if nome == "none" else LoessCorrector(n_jobs=1))
        construidos[nome].append(contador)
        return contador

    monkeypatch.setattr(labeling, "build_corrector_for_labeling", constroi)
    return construidos


def _arquivos(n_features: int = 5) -> dict[str, tuple[str, bytes, str]]:
    medidas, lista = render_pair(base_spec(n_features=n_features, n_samples=40, n_qcs=N_QC))
    return {
        "measurements": ("measurements.csv", medidas.encode("utf-8"), "text/csv"),
        "sample_list": ("Sample_List.csv", lista.encode("utf-8"), "text/csv"),
    }


def _mede(cliente: TestClient, **campos: Any) -> dict[str, Any]:
    resposta = reclama(cliente, ROTA, files=_arquivos(), data={"mode": MODO, **campos}, teto_s=300)
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    assert isinstance(corpo, dict)
    return corpo


def _chamadas(corretores: dict[str, list[_Contador]]) -> dict[str, int]:
    return {nome: sum(c.chamadas for c in lista) for nome, lista in corretores.items()}


def test_sem_o_campo_a_medicao_nao_faz_ajuste_nenhum_a_mais_e_intervalos_e_null(
    cliente: TestClient, corretores: dict[str, list[_Contador]]
) -> None:
    """Quem não marca a caixa não paga nada: `n` ajustes por corretor, e o corpo sem
    intervalo nenhum — `null`, e não um dicionário vazio."""
    corpo = _mede(cliente)
    assert _chamadas(corretores) == {"none": N_QC, "loess": N_QC, "serrf": N_QC}, (
        f"sem o campo, a medição fez {_chamadas(corretores)} ajustes, e não {N_QC} por corretor"
    )
    assert "intervalos" in corpo and corpo["intervalos"] is None


def test_com_o_campo_o_corpo_traz_um_intervalo_por_corretor_e_a_eleicao_e_a_mesma(
    cliente: TestClient, corretores: dict[str, list[_Contador]]
) -> None:
    """`n(n+1)/2` ajustes por corretor, um intervalo por corretor que contém a barra, e a eleição
    — escores, vencedor, margem, abstenção — igual à do pedido sem o campo."""
    sem = _mede(cliente)
    antes = _chamadas(corretores)
    com = _mede(cliente, intervalo_por_barra="true")
    depois = _chamadas(corretores)

    esperados = N_QC * (N_QC + 1) // 2
    assert {k: depois[k] - antes[k] for k in depois} == {
        "none": esperados,
        "loess": esperados,
        "serrf": esperados,
    }
    for campo in ("scores", "vencedor", "margem", "abstem", "avaliavel", "estimador", "variante"):
        assert com[campo] == sem[campo], (
            f"o intervalo mudou `{campo}`: {sem[campo]} -> {com[campo]}"
        )

    intervalos = com["intervalos"]
    assert isinstance(intervalos, dict) and set(intervalos) == set(com["scores"])
    for corretor, intervalo in intervalos.items():
        assert intervalo["motivo"] is None, intervalo
        assert intervalo["metodo"] == JACKKNIFE_METHOD
        assert intervalo["nivel"] == JACKKNIFE_LEVEL and intervalo["n_qc"] == N_QC
        assert intervalo["erro_padrao"] is not None and intervalo["erro_padrao"] >= 0.0
        assert intervalo["inferior"] <= com["scores"][corretor] <= intervalo["superior"]


def test_o_pedido_com_intervalo_nao_e_reclamado_pelo_trabalho_sem_ele(
    cliente: TestClient, corretores: dict[str, list[_Contador]]
) -> None:
    """A chave do trabalho leva o campo: sem isso, o pedido com o intervalo devolveria o corpo do
    trabalho sem ele, e a figura sairia sem o que a pessoa esperou para ter."""
    assert _mede(cliente)["intervalos"] is None
    assert _mede(cliente, intervalo_por_barra="true")["intervalos"] is not None
    # E o inverso: o pedido sem o campo, depois, continua a ser o trabalho sem ele.
    assert _mede(cliente)["intervalos"] is None


@pytest.mark.parametrize("valor", ["talvez", "2", "[]"])
def test_o_campo_com_tipo_errado_e_422_no_disparo(
    cliente: TestClient, corretores: dict[str, list[_Contador]], valor: str
) -> None:
    """O campo atravessa a fronteira do formulário, e um valor fora do domínio não cai
    num `else` razoável — nem «falso», que desligaria em silêncio o que a pessoa pediu."""
    disparo = cliente.post(
        ROTA, files=_arquivos(), data={"mode": MODO, "intervalo_por_barra": valor}
    )
    assert disparo.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, disparo.text
    assert "intervalo_por_barra" in disparo.text
    assert _chamadas(corretores) == {"none": 0, "loess": 0, "serrf": 0}


def test_com_um_lote_de_tres_qcs_o_loess_fica_sem_barra_e_o_intervalo_traz_o_motivo(
    cliente: TestClient, corretores: dict[str, list[_Contador]]
) -> None:
    """Corte em 16: o primeiro lote fica com os QCs de 1, 6 e 11 — três, o piso da régua — e o
    segundo com cinco.

    O desfecho é a recusa com motivo. Antes, os três corretores tinham barra,
    e a do LOESS misturava os três ajustes do lote de três, que treinam com dois e voltam
    `fit_failed`, com os corrigidos. Agora o LOESS — e o «SERRF» deste arquivo, que é o LOESS em
    série e declara o mesmo mínimo — ficam sem barra, com o motivo em `recusas` e no intervalo, e
    sem um ajuste sequer; o `none` tem barra, e o intervalo dele traz o motivo do piso."""
    corpo = _mede(cliente, intervalo_por_barra="true", n_batches="2", batch_cuts="[16]")
    assert corpo["qcs_por_lote"] == [3, 5], corpo["qcs_por_lote"]
    assert corpo["scores"]["none"] is not None, corpo["scores"]
    assert corpo["scores"]["loess"] is None and corpo["scores"]["serrf"] is None, corpo["scores"]
    assert set(corpo["recusas"]) == {"loess", "serrf"}, corpo["recusas"]
    assert _chamadas(corretores) == {"none": N_QC, "loess": 0, "serrf": 0}
    for corretor, intervalo in corpo["intervalos"].items():
        assert intervalo["motivo"] is not None and "3 QCs" in intervalo["motivo"], intervalo
        assert intervalo["inferior"] is None and intervalo["superior"] is None
        assert intervalo["erro_padrao"] is None
        if corretor in corpo["recusas"]:
            assert intervalo["motivo"] == corpo["recusas"][corretor], (corretor, intervalo)

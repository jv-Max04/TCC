"""A recusa com motivo do corretor cujo lote menor não deixa QCs de treino, no fio.

O desfecho é a recusa com motivo. No núcleo isso é a recusa tipada da
régua (`metrics.crossval.BatchQcShortfallError`) e o `Label.refusals` do oráculo; aqui afirma-se o
que chega a quem lê a resposta:

- **na medição dos corretores**, o corretor recusado vem com `null` em `scores` e o motivo em
  `recusas`, não entra no vencedor nem na margem, e os outros continuam medidos;
- **no Palpite**, a mesma coisa em `escores` e `recusas`, e a justificativa diz o motivo — é ela que
  vai ao registro da decisão e à delegação do «Automático»;
- **e sem lote curto, `recusas` vem vazio** — o caminho de sempre não ganha nada.

**A matriz é da factory, e não um arquivo do laboratório** — 40 injeções, oito QCs em 1, 6, 11, 16,
21, 26, 31 e 36 —, e os corretores são servidos leves: `none` ele mesmo, o LOESS em série, e o
«SERRF» é o `none` sob outro nome, porque o que importa dele aqui é não declarar mínimo de treino,
como o SERRF real.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

import refino.api.rotas as rotas
from refino.api.app import build_app
from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.decision import labeling
from refino.domain.correction import CorrectionResult
from refino.domain.matrix import RefinoMatrix
from tests.factories import base_spec, render_pair
from tests.reclamacao import reclama

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("memoria_de_correcoes_privada")]

MODO = "neg_hilic"
N_QC = 8


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


@pytest.fixture(autouse=True)
def palpites_limpos(monkeypatch: pytest.MonkeyPatch) -> None:
    """A memória dos Palpites é do processo; um Palpite guardado de outro teste seria lido."""
    monkeypatch.setattr(rotas, "_PALPITES", OrderedDict())


class _Contado:
    def __init__(self, nome: str, interno: Any) -> None:
        self.interno = interno
        self.name = nome
        self.n_jobs = 1
        if hasattr(interno, "min_train_qc_per_batch"):
            self.min_train_qc_per_batch = interno.min_train_qc_per_batch
        self.chamadas = 0

    def fit_transform(self, matrix: RefinoMatrix, **kwargs: Any) -> CorrectionResult:
        self.chamadas += 1
        return self.interno.fit_transform(matrix, **kwargs)  # type: ignore[no-any-return]


@pytest.fixture
def corretores(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[_Contado]]:
    construidos: dict[str, list[_Contado]] = {"none": [], "loess": [], "serrf": []}

    def constroi(nome: str, *, n_jobs: int = -1) -> Any:
        del n_jobs
        interno = LoessCorrector(n_jobs=1) if nome == "loess" else NoneCorrector()
        contado = _Contado(nome, interno)
        construidos[nome].append(contado)
        return contado

    monkeypatch.setattr(labeling, "build_corrector_for_labeling", constroi)
    return construidos


def _chamadas(corretores: dict[str, list[_Contado]]) -> dict[str, int]:
    return {nome: sum(c.chamadas for c in lista) for nome, lista in corretores.items()}


def _arquivos() -> dict[str, tuple[str, bytes, str]]:
    medidas, lista = render_pair(base_spec(n_features=5, n_samples=40, n_qcs=N_QC))
    return {
        "measurements": ("measurements.csv", medidas.encode("utf-8"), "text/csv"),
        "sample_list": ("Sample_List.csv", lista.encode("utf-8"), "text/csv"),
    }


def _corpo(cliente: TestClient, rota: str, **campos: Any) -> dict[str, Any]:
    resposta = reclama(cliente, rota, files=_arquivos(), data={"mode": MODO, **campos}, teto_s=300)
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    assert isinstance(corpo, dict)
    return corpo


# Corte em 16: o primeiro lote fica com os QCs de 1, 6 e 11 — três, o piso da régua — e o segundo
# com cinco. O LOESS treinaria com dois no primeiro, e não ajusta com menos de três.
LOTE_DE_TRES = {"n_batches": "2", "batch_cuts": "[16]"}


def test_na_medicao_o_loess_vem_sem_numero_com_o_motivo_e_fora_da_eleicao(
    cliente: TestClient, corretores: dict[str, list[_Contado]]
) -> None:
    corpo = _corpo(cliente, "/matrizes/medir-corretor", **LOTE_DE_TRES)
    assert corpo["qcs_por_lote"] == [3, 5], corpo["qcs_por_lote"]

    assert set(corpo["recusas"]) == {"loess"}, corpo["recusas"]
    motivo = corpo["recusas"]["loess"]
    assert "0=3 QCs" in motivo and "menos de 4" in motivo and "loess" in motivo, motivo
    assert corpo["scores"]["loess"] is None, corpo["scores"]
    assert corpo["scores"]["none"] is not None and corpo["scores"]["serrf"] is not None
    assert corpo["vencedor"] in {"none", "serrf"} and corpo["avaliavel"] is True
    assert _chamadas(corretores) == {"none": N_QC, "loess": 0, "serrf": N_QC}

    # A margem é a dos dois medidos, recalculada aqui a partir dos escores do próprio corpo.
    medidos = sorted(v for v in corpo["scores"].values() if v is not None)
    if medidos[0] != 0.0:
        assert corpo["margem"] == pytest.approx((medidos[1] - medidos[0]) / abs(medidos[0]))


def test_sem_lote_curto_recusas_vem_vazio_e_o_loess_mede(
    cliente: TestClient, corretores: dict[str, list[_Contado]]
) -> None:
    """O par causal: a mesma matriz em lote único — oito QCs — mede os três, e `recusas` é `{}`."""
    corpo = _corpo(cliente, "/matrizes/medir-corretor")
    assert corpo["recusas"] == {}
    assert all(isinstance(v, float) and math.isfinite(v) for v in corpo["scores"].values())
    assert _chamadas(corretores) == {"none": N_QC, "loess": N_QC, "serrf": N_QC}


def test_no_palpite_o_loess_vem_sem_numero_e_a_justificativa_diz_o_motivo(
    cliente: TestClient, corretores: dict[str, list[_Contado]]
) -> None:
    corpo = _corpo(cliente, "/matrizes/palpite", **LOTE_DE_TRES)
    assert set(corpo["recusas"]) == {"loess"}, corpo["recusas"]
    assert corpo["escores"]["loess"] is None, corpo["escores"]
    assert corpo["correction_proposta"] in {"none", "serrf"}
    motivo = corpo["recusas"]["loess"]
    assert f"o loess não foi medido: {motivo}" in corpo["rationale"], corpo["rationale"]
    assert _chamadas(corretores)["loess"] == 0

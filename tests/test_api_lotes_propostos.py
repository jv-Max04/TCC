"""A sugestão de lotes de `POST /matrizes/diagnosticar` e o Palpite que mede nela.

Quando os cortes detectados deixariam um lote com QC de menos para medir fora da amostra, a
sugestão é um lote único e os cortes detectados vão como observação. O Palpite, disparado com a
sugestão, mede; disparado com os cortes detectados, continua recusado. **A medição verdadeira não
corre aqui**: o medidor é um duplo, como em `tests/test_api_palpite.py`, e o que se afirma é que a
rota aceitou a partição e mediu uma vez.

Os números são os dos arquivos reais: no `pos_hilic` a detecção corta em 23 e 45 (QCs 2, 2 e 3), no
`neg_hilic` em 56 (QCs 5 e 2) e no `neg_c18` em 34 (QCs 3 e 4).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

import refino.api.rotas as rotas
from refino.api.app import build_app
from refino.decision.palpite import decide_o_palpite
from refino.diagnosis.batches import batch_without_training
from tests.conftest import measurements_path, sample_list_path
from tests.reclamacao import reclama


@pytest.fixture
def cliente() -> TestClient:
    return TestClient(build_app())


@pytest.fixture
def medidos(monkeypatch: pytest.MonkeyPatch, memoria_de_correcoes_privada: None) -> list[Any]:
    """O duplo de `mede_o_palpite`. Guarda os lotes de cada chamada, `None` quando é lote único."""
    chamadas: list[Any] = []

    def medidor(
        matriz: Any,
        *,
        batches: Any = None,
        n_jobs: int = -1,
        depois_de_cada: Any = None,
        cancelled: Any = None,
        seed: int = 42,
    ) -> Any:
        chamadas.append(batches)
        for corretor in ("none", "loess", "serrf"):
            if depois_de_cada is not None:
                depois_de_cada((corretor, 1))
        return decide_o_palpite(
            {"none": 0.30, "loess": 0.10, "serrf": 0.20}, n_features_medidas=200
        )

    monkeypatch.setattr(rotas, "mede_o_palpite", medidor)
    monkeypatch.setattr(rotas, "_PALPITES", OrderedDict())
    return chamadas


def _arquivos(modo: str) -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{modo}.csv",
            measurements_path(modo).read_bytes(),
            "text/csv",
        ),
        "sample_list": (f"Sample_List_{modo}.csv", sample_list_path(modo).read_bytes(), "text/csv"),
    }


def _lotes_do_diagnostico(cliente: TestClient, modo: str) -> tuple[dict[str, Any], list[str]]:
    resposta = cliente.post("/matrizes/diagnosticar", files=_arquivos(modo), data={"mode": modo})
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    return corpo["lotes"], list(corpo["trajetoria_dos_qcs"]["lotes"])


@pytest.mark.parametrize(
    ("modo", "detectados", "qcs_detectados"),
    [("pos_hilic", [23, 45], [2, 2, 3]), ("neg_hilic", [56], [5, 2])],
)
def test_cortes_que_o_palpite_recusaria_viram_sugestao_de_lote_unico(
    cliente: TestClient, modo: str, detectados: list[int], qcs_detectados: list[int]
) -> None:
    lotes, lotes_dos_qcs = _lotes_do_diagnostico(cliente, modo)

    assert lotes["cortes_detectados"] == detectados
    assert lotes["qcs_por_lote_detectados"] == qcs_detectados
    assert lotes["n_lotes"] == 1
    assert lotes["changepoints"] == []
    assert lotes["qcs_por_lote"] == [7]
    assert set(lotes["rotulos_por_amostra"]) == {0}
    # A figura da trajetória pinta a proposta, e não os cortes detectados.
    assert set(lotes_dos_qcs) == {"0"}


def test_cortes_que_sustentam_a_medicao_continuam_na_sugestao(cliente: TestClient) -> None:
    """O par do teste de cima: no `neg_c18` o corte deixa três e quatro QCs, e entra na sugestão."""
    lotes, lotes_dos_qcs = _lotes_do_diagnostico(cliente, "neg_c18")

    assert lotes["changepoints"] == lotes["cortes_detectados"] == [34]
    assert lotes["qcs_por_lote"] == lotes["qcs_por_lote_detectados"] == [3, 4]
    assert lotes["n_lotes"] == 2
    assert set(lotes_dos_qcs) == {"0", "1"}


def test_o_palpite_mede_na_sugestao_e_recusa_os_cortes_detectados(
    cliente: TestClient, medidos: list[Any]
) -> None:
    """No `pos_hilic`, o Palpite com a sugestão mede em lote único; com os cortes crus, recusa.

    O segundo pedido é o que a janela mandava antes, quando a sugestão eram os cortes detectados.
    """
    lotes, _ = _lotes_do_diagnostico(cliente, "pos_hilic")
    sugestao: dict[str, Any] = {"mode": "pos_hilic", "n_batches": lotes["n_lotes"]}
    if lotes["changepoints"]:
        sugestao["batch_cuts"] = json.dumps(lotes["changepoints"])

    medido = reclama(cliente, "/matrizes/palpite", files=_arquivos("pos_hilic"), data=sugestao)

    assert medido.status_code == status.HTTP_200_OK, medido.text
    assert medidos == [None], f"o Palpite não mediu uma vez, em lote único: {medidos}"

    crus = cliente.post(
        "/matrizes/palpite",
        files=_arquivos("pos_hilic"),
        data={
            "mode": "pos_hilic",
            "n_batches": len(lotes["cortes_detectados"]) + 1,
            "batch_cuts": json.dumps(lotes["cortes_detectados"]),
        },
    )
    assert crus.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, crus.text
    assert "[2, 2, 3]" in crus.json()["detail"]
    assert medidos == [None], "a partição recusada chegou a medir"


@pytest.mark.parametrize("qcs", [(2, 2, 3), (5, 2), (3, 4), (7,), (3, 0, 4), (1,)])
def test_a_recusa_do_palpite_e_o_recuo_da_sugestao_usam_o_mesmo_predicado(
    qcs: tuple[int, ...],
) -> None:
    """Uma sugestão do diagnóstico nunca é recusada pelo Palpite: o predicado é um só."""
    assert rotas._algum_lote_sem_treino(qcs) is batch_without_training(qcs)

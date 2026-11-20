"""`POST /matrizes/curva-do-loess` — a curva do LOESS com o envelope sem um QC, de ponta a ponta.

**O caminho inteiro é o que se confere aqui:** arquivo do laboratório → rota → corpo JSON →
`curva_do_fio` → figura → a série da curva. A curva desenhada tem de ser a que a correção usou,
bit a bit, e o oráculo é a própria correção sobre a mesma matriz ingerida — dividir pela curva
desenhada reproduz a saída de `LoessCorrector.fit_transform`.

O modo é o `neg_c18`, o de menos features dos quatro: o que se mede aqui é a ligação, e a população
inteira de cada modo está em `tests/unit/test_loess_curva.py`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.correction.base import restore_zeros_and_missing
from refino.correction.loess import LoessCorrector, curve_sensitivity
from refino.decision.labeling import com_lotes
from refino.domain.lotes import lotes_declarados
from refino.domain.matrix import RefinoMatrix
from refino.io.readers import load_matrix
from refino.plots.curva_do_loess import (
    ROTULO_DA_CURVA,
    curva_do_fio,
    figura_da_curva_do_loess,
    recorte_da_curva_do_loess,
)
from tests.arnes_de_figura import series
from tests.conftest import measurements_path, sample_list_path
from tests.factories import base_spec, render_pair

pytestmark = pytest.mark.integration

MODO = "neg_c18"
ROTA = "/matrizes/curva-do-loess"

# Os campos do corpo que a figura lê, na forma de `curva_do_fio`. `n_features` vai para o rodapé.
CAMPOS_DA_FIGURA = (
    "feature",
    "ordens_de_injecao",
    "lotes",
    "qc",
    "valores",
    "curva",
    "curva_preenchida",
    "curva_sorteada",
    "qc_no_ajuste",
    "motivo_sem_curva",
    "predicao_sem_o_qc",
    "motivo_sem_predicao",
    "envelope_inferior",
    "envelope_superior",
    "envelope_nao_ajustado",
    "n_curvas_do_envelope",
    "motivo_sem_envelope",
    "semente",
    "spans_por_lote",
)


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


@pytest.fixture(scope="module")
def matriz() -> RefinoMatrix:
    return load_matrix(measurements_path(MODO), sample_list_path(MODO), MODO)


def _arquivos() -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{MODO}.csv",
            measurements_path(MODO).read_bytes(),
            "text/csv",
        ),
        "sample_list": (f"Sample_List_{MODO}.csv", sample_list_path(MODO).read_bytes(), "text/csv"),
    }


def _feature_com_curva(matriz: RefinoMatrix, seed: int = 42) -> str:
    """A primeira feature da matriz cuja correção usa uma curva — escolhida, não pinada."""
    corretor = LoessCorrector()
    for nome in matriz.feature_names:
        if corretor.fitted_curve(matriz, nome, seed=seed).cause is None:
            return nome
    raise AssertionError("nenhuma feature com curva no modo")  # pragma: no cover


def _curva_desenhada(corpo: dict[str, Any]) -> np.ndarray:
    # O corretor não vem da rota: é o da correção de quem pede, e estes testes comparam com a do
    # LOESS.
    curva = curva_do_fio(
        **{campo: corpo.get(campo) for campo in CAMPOS_DA_FIGURA}, corretor="loess"
    )
    figura = figura_da_curva_do_loess(
        curva=curva,
        recorte=recorte_da_curva_do_loess(curva=curva, n_features_totais=corpo.get("n_features")),
    )
    achadas = [s for s in series(figura.axes[0]) if s.rotulo.startswith(ROTULO_DA_CURVA)]
    assert achadas, "a figura não desenhou a curva"
    return np.concatenate([np.asarray(s.y, dtype=np.float64) for s in achadas])


def test_a_curva_desenhada_e_a_que_a_correcao_usou_bit_a_bit(
    cliente: TestClient, matriz: RefinoMatrix
) -> None:
    """Do arquivo à série da figura, e a correção da mesma matriz divide por ela.

    A conta que dá a saída da correção a partir da curva é a do passo 5 e a do passo 11; ela só
    deixa ver a curva onde a feature é finita e diferente de zero, e é lá que a âncora conta.
    """
    feature = _feature_com_curva(matriz)
    resposta = cliente.post(ROTA, files=_arquivos(), data={"mode": MODO, "feature": feature})
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    assert corpo["motivo_sem_curva"] is None and corpo["n_features"] == matriz.n_features

    desenhada = _curva_desenhada(corpo)
    j = matriz.feature_names.index(feature)
    coluna = np.asarray(matriz.values[:, j], dtype=np.float64)
    corretor = LoessCorrector()
    saida = corretor._normalize(coluna, desenhada).values
    refeita = restore_zeros_and_missing(coluna[:, None], saida[:, None]).values[:, 0]
    corrigida = corretor.fit_transform(matriz, seed=42, target_features=frozenset({feature}))
    assert refeita.tobytes() == np.asarray(corrigida.matrix.values[:, 0]).tobytes()
    assert int(np.sum(np.isfinite(coluna) & (coluna != 0.0))) >= 30, "âncora: posições visíveis"


def test_com_dois_lotes_a_curva_e_a_da_correcao_com_os_mesmos_lotes(
    cliente: TestClient, matriz: RefinoMatrix
) -> None:
    """Os lotes do formulário são os da correção: a curva é a de `fit_transform(batches=...)`.

    Em dois blocos iguais pela ordem de injeção, os sete QCs deste arquivo dividem-se em 4 e 3; sem
    um QC do lote de três sobram dois, e o ajuste sem ele falha — a rota diz isso por QC.
    """
    feature = _feature_com_curva(matriz)
    resposta = cliente.post(
        ROTA, files=_arquivos(), data={"mode": MODO, "feature": feature, "n_batches": 2}
    )
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    assert len(set(corpo["lotes"])) == 2 and len(corpo["spans_por_lote"]) == 2, corpo["lotes"]

    lotes = lotes_declarados(matriz.injection_orders, 2)
    esperada = LoessCorrector().fitted_curve(matriz, feature, batches=lotes, seed=42)
    assert esperada.cause is None, esperada.reason
    assert _curva_desenhada(corpo).tobytes() == esperada.curve.tobytes()
    sem_predicao = [
        m for q, m in zip(corpo["qc"], corpo["motivo_sem_predicao"], strict=True) if q and m
    ]
    assert sem_predicao and all("exige 3" in m for m in sem_predicao), sem_predicao


def test_lote_com_dois_qcs_responde_sem_curva_e_com_o_motivo(
    cliente: TestClient, matriz: RefinoMatrix
) -> None:
    """Três blocos dão um lote de dois QCs: a resposta é 200, sem curva, e o corpo diz por quê."""
    feature = _feature_com_curva(matriz)
    resposta = cliente.post(
        ROTA, files=_arquivos(), data={"mode": MODO, "feature": feature, "n_batches": 3}
    )
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    assert all(v is None for v in corpo["curva"])
    assert "exige 3" in corpo["motivo_sem_curva"], corpo["motivo_sem_curva"]
    assert corpo["motivo_sem_envelope"] and corpo["n_curvas_do_envelope"] == 0


def test_feature_que_a_matriz_nao_tem_e_422(cliente: TestClient) -> None:
    resposta = cliente.post(ROTA, files=_arquivos(), data={"mode": MODO, "feature": "0_0m/z"})
    assert resposta.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert "não existe nesta matriz" in resposta.json()["detail"]


# ---------------------------------------------------------------------------
# A mesma ligação sobre arquivos da fábrica — o que corre onde os do laboratório não estão
# ---------------------------------------------------------------------------
#
# **A matriz é da factory, e não um arquivo de referência, de propósito** — o argumento de
# `tests/test_api_cortes_de_lote.py`. O que se afirma aqui é a FIAÇÃO: o corpo da rota é a curva
# do caminho da correção, com os lotes do formulário, e as curvas sem cada QC são as de
# `curve_sensitivity` sobre a matriz com esses lotes. Nada aqui afirma o que os arquivos do
# laboratório contêm, e os testes acima continuam a ser os do dado real.
#
# 40 injeções e oito QCs, em 1, 6, 11, 16, 21, 26, 31 e 36 — as ordens que `base_spec` declara.
# Dois blocos dão quatro QCs por lote; quatro blocos dão dois, abaixo do mínimo do ajuste.

MODO_DA_FABRICA = "neg_hilic"


def _spec_da_fabrica() -> Any:
    """`base_spec` com deriva suave e ruído pequeno, para a curva não ser uma reta exacta."""
    spec = base_spec(n_features=6, n_samples=40, n_qcs=8)
    rng = np.random.default_rng(5)
    valores = tuple(
        tuple(
            float(
                1000.0
                * (f + 1)
                * (1.0 + 0.01 * s.injection_order + 0.05 * np.sin(s.injection_order / 5.0))
                * rng.normal(1.0, 0.02)
            )
            for s in spec.samples
        )
        for f in range(len(spec.feature_names))
    )
    return replace(spec, values=valores)


@pytest.fixture(scope="module")
def da_fabrica(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, Any], RefinoMatrix]:
    """Os dois arquivos da fábrica, como o formulário os manda, e a matriz que a ingestão dá."""
    medidas, lista = render_pair(_spec_da_fabrica())
    pasta = tmp_path_factory.mktemp("fabrica")
    caminho_m, caminho_s = pasta / "measurements.csv", pasta / "Sample_List.csv"
    caminho_m.write_bytes(medidas.encode("utf-8"))
    caminho_s.write_bytes(lista.encode("utf-8"))
    arquivos = {
        "measurements": ("measurements.csv", medidas.encode("utf-8"), "text/csv"),
        "sample_list": ("Sample_List.csv", lista.encode("utf-8"), "text/csv"),
    }
    return arquivos, load_matrix(caminho_m, caminho_s, MODO_DA_FABRICA)


def _sem_nulos(vetor: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in vetor], dtype=np.float64)


def test_na_fabrica_a_curva_da_rota_e_a_da_correcao_e_as_sem_um_qc_as_da_regua(
    cliente: TestClient, da_fabrica: tuple[dict[str, Any], RefinoMatrix]
) -> None:
    """Com dois lotes: a série desenhada é a curva por que a correção divide, bit a bit, e as
    predições e o envelope são os de `curve_sensitivity` sobre a matriz com os mesmos lotes."""
    arquivos, matriz = da_fabrica
    feature = _feature_com_curva(matriz)
    resposta = cliente.post(
        ROTA,
        files=arquivos,
        data={"mode": MODO_DA_FABRICA, "feature": feature, "n_batches": 2},
    )
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    assert len(set(corpo["lotes"])) == 2 and len(corpo["spans_por_lote"]) == 2, corpo["lotes"]

    lotes = lotes_declarados(matriz.injection_orders, 2)
    j = matriz.feature_names.index(feature)
    coluna = np.asarray(matriz.values[:, j], dtype=np.float64)
    corretor = LoessCorrector()
    saida = corretor._normalize(coluna, _curva_desenhada(corpo)).values
    refeita = restore_zeros_and_missing(coluna[:, None], saida[:, None]).values[:, 0]
    corrigida = corretor.fit_transform(
        matriz, batches=lotes, seed=42, target_features=frozenset({feature})
    )
    assert refeita.tobytes() == np.asarray(corrigida.matrix.values[:, 0]).tobytes()
    assert int(np.sum(np.isfinite(coluna) & (coluna != 0.0))) == matriz.n_samples, "âncora"

    sensibilidade = curve_sensitivity(com_lotes(matriz, lotes), feature, seed=42)
    esperada = np.full(matriz.n_samples, np.nan)
    for linha, curva in zip(sensibilidade.held_out_rows, sensibilidade.held_out, strict=True):
        if curva.cause is None:
            esperada[linha] = curva.curve[linha]
    predicao = _sem_nulos(corpo["predicao_sem_o_qc"])
    assert int(np.isfinite(predicao).sum()) == 8, "âncora: os oito QCs têm predição sem eles"
    assert predicao.tobytes() == esperada.tobytes()
    assert _sem_nulos(corpo["envelope_inferior"]).tobytes() == sensibilidade.lower.tobytes()
    assert _sem_nulos(corpo["envelope_superior"]).tobytes() == sensibilidade.upper.tobytes()
    assert corpo["n_curvas_do_envelope"] == sensibilidade.n_curves == 8


def test_na_fabrica_a_semente_do_formulario_e_a_da_correcao(
    cliente: TestClient, da_fabrica: tuple[dict[str, Any], RefinoMatrix]
) -> None:
    """Lote único e semente fora do padrão: a curva é a da correção com ESSA semente.

    **Os dois lotes do teste acima não separam sementes**, e foi medido: com quatro QCs por lote, a
    curva desta matriz é a mesma com 42 e com 43 nas seis features — uma rota que ignorasse a
    semente do formulário passaria por ele. Com o lote único, de oito QCs, a semente muda a curva
    em três das seis; a feature é escolhida entre essas, e a âncora diz que ali as duas diferem.
    """
    arquivos, matriz = da_fabrica
    corretor = LoessCorrector()
    feature = next(
        nome
        for nome in matriz.feature_names
        if corretor.fitted_curve(matriz, nome, seed=7).curve.tobytes()
        != corretor.fitted_curve(matriz, nome, seed=42).curve.tobytes()
    )
    resposta = cliente.post(
        ROTA, files=arquivos, data={"mode": MODO_DA_FABRICA, "feature": feature, "seed": 7}
    )
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    assert corpo["semente"] == 7 and len(set(corpo["lotes"])) == 1

    j = matriz.feature_names.index(feature)
    coluna = np.asarray(matriz.values[:, j], dtype=np.float64)
    saida = corretor._normalize(coluna, _curva_desenhada(corpo)).values
    refeita = restore_zeros_and_missing(coluna[:, None], saida[:, None]).values[:, 0]
    corrigida = corretor.fit_transform(matriz, seed=7, target_features=frozenset({feature}))
    assert refeita.tobytes() == np.asarray(corrigida.matrix.values[:, 0]).tobytes()


def test_na_fabrica_lote_com_dois_qcs_responde_sem_curva_e_com_o_motivo(
    cliente: TestClient, da_fabrica: tuple[dict[str, Any], RefinoMatrix]
) -> None:
    """Quatro blocos dão dois QCs por lote: 200, sem curva, sem envelope, e o corpo diz por quê."""
    arquivos, matriz = da_fabrica
    feature = matriz.feature_names[0]
    resposta = cliente.post(
        ROTA, files=arquivos, data={"mode": MODO_DA_FABRICA, "feature": feature, "n_batches": 4}
    )
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    assert all(v is None for v in corpo["curva"])
    assert "exige 3" in corpo["motivo_sem_curva"], corpo["motivo_sem_curva"]
    assert corpo["spans_por_lote"] == []
    assert corpo["motivo_sem_envelope"] and corpo["n_curvas_do_envelope"] == 0
    assert all(v is None for v in corpo["predicao_sem_o_qc"])


def test_na_fabrica_feature_que_a_matriz_nao_tem_e_422(
    cliente: TestClient, da_fabrica: tuple[dict[str, Any], RefinoMatrix]
) -> None:
    arquivos, _ = da_fabrica
    resposta = cliente.post(
        ROTA, files=arquivos, data={"mode": MODO_DA_FABRICA, "feature": "0_0m/z"}
    )
    assert resposta.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert "não existe nesta matriz" in resposta.json()["detail"]

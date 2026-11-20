"""O Palpite medido: a regra pura nas duas direções, e a medição que a alimenta.

A regra é testada sem medir nada, porque `decide_o_palpite` é função pura de três escores: as
duas direções do limiar, a fronteira exacta, e os dois casos degenerados, que não são o mesmo —
*«não houve medição»* e *«não houve segundo»*.

**A proposta no empate é o vencedor da amostra, e há teste que o prende.** A regra que trocaria o
vencedor pelo SERRF no empate foi pré-registrada e reprovou; se alguém a religar,
`test_no_empate_a_proposta_continua_o_vencedor_da_amostra` reprova.

**A medição é conferida por dois caminhos.** Em processo, um espião prova que o oráculo recebe as
200 *features*, os lotes e a bandeira de cancelamento; e, nos arquivos reais, `slow`, os escores do
Palpite são as medianas dos vetores da população gravados, restritos às 200 sorteadas, porque a
execução parcial é exata.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import pytest

import refino.decision.palpite as palpite
from refino.decision.labeling import Label
from refino.decision.palpite import (
    LIMIAR_DE_EMPATE_DO_PALPITE,
    N_FEATURES_DO_PALPITE,
    decide_o_palpite,
    mede_o_palpite,
)
from refino.decision.policy import ABSTENTION_MARGIN

pytestmark = pytest.mark.property


def test_os_numeros_do_palpite_sao_os_registrados() -> None:
    """200 é do usuário, e o limiar é o derivado do sorteio das 200 features (0,167027)."""
    assert N_FEATURES_DO_PALPITE == 200
    assert pytest.approx(0.167027, abs=5e-4) == LIMIAR_DE_EMPATE_DO_PALPITE
    # e não é o limiar da política, que vem do ruído da semente sobre a população inteira
    assert LIMIAR_DE_EMPATE_DO_PALPITE != ABSTENTION_MARGIN


def test_margem_acima_do_limiar_decide_pelo_menor() -> None:
    p = decide_o_palpite({"none": 0.30, "loess": 0.40, "serrf": 0.20}, n_features_medidas=200)
    assert p.correction == "serrf"
    assert p.segundo == "none"
    assert p.margem == pytest.approx(0.5)
    assert not p.abstem
    assert p.confianca > 0.5
    assert "serrf" in p.rationale and "50,0%" in p.rationale


def test_margem_abaixo_do_limiar_abstem_se() -> None:
    p = decide_o_palpite({"none": 0.21, "loess": 0.40, "serrf": 0.20}, n_features_medidas=200)
    assert p.margem == pytest.approx(0.05)
    assert p.abstem
    assert p.confianca < 0.5
    assert "não separa" in p.rationale


def test_no_empate_a_proposta_continua_o_vencedor_da_amostra() -> None:
    """A regra que trocaria o LOESS pelo SERRF reprovou pelo critério escrito antes."""
    p = decide_o_palpite({"none": 0.30, "loess": 0.100, "serrf": 0.105}, n_features_medidas=200)
    assert p.abstem
    assert p.correction == "loess"


def test_a_fronteira_exacta_nao_se_abstem() -> None:
    """Em cima do limiar a confiança é 0,5, e `abstem` é `confiança < 0,5`."""
    base = 0.2
    escores = {"none": base * (1 + LIMIAR_DE_EMPATE_DO_PALPITE), "loess": 0.9, "serrf": base}
    p = decide_o_palpite(escores, n_features_medidas=200)
    assert p.confianca == pytest.approx(0.5)
    assert not p.abstem
    abaixo = dict(escores, none=base * (1 + LIMIAR_DE_EMPATE_DO_PALPITE * 0.99))
    assert decide_o_palpite(abaixo, n_features_medidas=200).abstem


def test_sem_escore_finito_nao_e_avaliavel() -> None:
    p = decide_o_palpite(
        {"none": math.nan, "loess": math.nan, "serrf": math.nan}, n_features_medidas=200
    )
    assert not p.avaliavel
    assert p.margem is None
    assert p.abstem
    assert p.correction == "none"
    assert "indefinido" in p.rationale
    assert not p.como_recomendacao().assessable


def test_um_escore_so_nao_tem_margem_e_se_abstem() -> None:
    p = decide_o_palpite(
        {"none": math.nan, "loess": math.nan, "serrf": 0.2}, n_features_medidas=200
    )
    assert p.avaliavel
    assert p.margem is None
    assert p.correction == "serrf"
    assert p.abstem


def test_melhor_escore_zero_nao_tem_margem_e_a_justificativa_diz_por_que() -> None:
    """«Não houve segundo» e «o melhor é zero» abstêm-se os dois, e não são a mesma frase.

    A divisão da margem relativa não se define com o melhor em zero — a convenção do oráculo —, e a
    primeira versão dizia *«não houve segundo corretor»* com o `loess` medido logo ao lado.
    """
    zero = decide_o_palpite({"none": 0.0, "loess": 0.5, "serrf": 0.9}, n_features_medidas=200)
    assert zero.margem is None and zero.abstem
    assert zero.segundo == "loess", "o segundo existe, e a justificativa precisa de o nomear"
    assert "não se define" in zero.rationale, zero.rationale
    assert "não houve segundo" not in zero.rationale, zero.rationale
    so_um = decide_o_palpite(
        {"none": math.nan, "loess": math.nan, "serrf": 0.2}, n_features_medidas=200
    )
    assert so_um.segundo is None and "não houve segundo" in so_um.rationale, so_um.rationale


def test_empate_exacto_resolve_pela_ordem_do_oraculo() -> None:
    p = decide_o_palpite({"none": 0.2, "loess": 0.2, "serrf": 0.5}, n_features_medidas=200)
    assert p.correction == "none"
    assert p.margem == 0.0
    assert p.abstem


def test_a_recomendacao_carrega_a_mesma_decisao() -> None:
    p = decide_o_palpite({"none": 0.30, "loess": 0.40, "serrf": 0.20}, n_features_medidas=150)
    r = p.como_recomendacao()
    assert r.correction == p.correction
    assert r.confidence == p.confianca
    assert r.abstains == p.abstem
    assert r.rationale == p.rationale
    assert r.evidence["n_features_medidas"] == 150.0
    assert r.evidence["limiar_de_empate"] == LIMIAR_DE_EMPATE_DO_PALPITE
    assert r.evidence["rsd_serrf"] == 0.20


class _Matriz:
    """O mínimo que `mede_o_palpite` lê da matriz fora do oráculo."""

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features


def test_a_medicao_pede_200_features_os_lotes_e_a_bandeira(monkeypatch: Any) -> None:
    """Espião em processo: o oráculo recebe o que o Palpite diz que mede."""
    chamadas: list[dict[str, Any]] = []

    def label(self: Any, matrix: Any, **kwargs: Any) -> Label:
        chamadas.append(
            {
                "n_sampled_features": self.n_sampled_features,
                "prefixes": self.prefixes,
                "n_jobs": self.n_jobs,
                "seed": self.seed,
                **kwargs,
            }
        )
        n = self.prefixes[0][1]
        return Label(
            ("loess", n),
            {("none", n): 0.30, ("loess", n): 0.10, ("serrf", n): 0.20},
            "leave_one_qc_out_rsd",
            "outlier_removed",
        )

    monkeypatch.setattr(palpite.LabelingOracle, "label", label)
    cancelado = lambda: False  # noqa: E731
    lotes = np.array(["A", "A", "B", "B"])
    p = mede_o_palpite(_Matriz(6035), batches=lotes, n_jobs=3, cancelled=cancelado)  # type: ignore[arg-type]
    assert len(chamadas) == 1
    c = chamadas[0]
    assert c["n_sampled_features"] == 200
    assert c["prefixes"] == (("none", 2), ("loess", 2), ("serrf", 2))
    assert c["n_jobs"] == 3
    assert c["seed"] == 42, "o padrão da semente mudou"
    assert c["batches"] is lotes
    assert c["cancelled"] is cancelado
    assert p.correction == "loess"
    assert p.n_features_medidas == 200
    # matriz menor que a amostra é medida inteira, e o número diz isso
    assert mede_o_palpite(_Matriz(120)).n_features_medidas == 120  # type: ignore[arg-type]
    # a semente da execução chega ao oráculo (a outra direção do padrão 42 acima)
    mede_o_palpite(_Matriz(6035), seed=7)  # type: ignore[arg-type]
    assert chamadas[-1]["seed"] == 7


@pytest.fixture(scope="module")
def palpite_e_gravado_no_neg_c18() -> tuple[Any, dict[str, float]]:
    """O Palpite medido no `neg_c18` real, e as medianas dos vetores gravados nas mesmas 200.

    Uma medição para os dois testes de baixo: ela custa perto de um minuto com 12 núcleos.

    Os vetores são os da F2, gravados no Windows com o SERRF e o LOESS em uso.
    """
    from pathlib import Path

    from refino.io.readers import load_matrix
    from refino.metrics.crossval import sampled_feature_names
    from tests.conftest import measurements_path, sample_list_path

    modo = "neg_c18"
    matriz = load_matrix(measurements_path(modo), sample_list_path(modo), mode=modo)
    raiz = Path(__file__).resolve().parents[2]
    vetores = raiz / f"data/fixtures/medicoes/jh/f2_vetores/unidades/real-{modo}.npz"
    with np.load(vetores) as f2:
        gravado = {c: f2[c] for c in ("none", "loess", "serrf")}
    nomes = set(sampled_feature_names(matriz, N_FEATURES_DO_PALPITE, 42) or ())
    assert len(nomes) == 200
    idx = [i for i, n in enumerate(matriz.feature_names) if n in nomes]
    esperado = {c: float(np.nanmedian(v[idx])) for c, v in gravado.items()}
    return mede_o_palpite(matriz), esperado


@pytest.mark.slow
def test_nos_reais_o_palpite_e_a_populacao_restrita_as_200_sorteadas(
    palpite_e_gravado_no_neg_c18: tuple[Any, dict[str, float]],
) -> None:
    """O outro caminho como oráculo: os escores do Palpite são as medianas dos vetores gravados.

    A medição das 200 sorteadas dá o mesmo RSD por feature que a da população inteira, então a
    mediana do Palpite é a mediana do gravado nas mesmas 200 — igualdade exacta, e não aproximada.

    Este confere o `none`, que é o instrumento (a amostra de 200 e a mediana), em qualquer sistema;
    o de baixo confere os dois corretores. Uma reprovação diz qual dos dois quebrou.
    """
    p, esperado = palpite_e_gravado_no_neg_c18
    assert p.escores["none"] == esperado["none"]
    assert p.n_features_medidas == 200


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "win32",
    reason="só no Windows: os vetores da F2 foram gravados no Windows, e fora dele o LOESS "
    "fica a 1 ULP da mediana gravada",
)
def test_nos_reais_os_escores_do_serrf_e_do_loess_sao_os_dos_vetores_gravados(
    palpite_e_gravado_no_neg_c18: tuple[Any, dict[str, float]],
) -> None:
    """A parte do teste de cima que confere os dois corretores: o SERRF e o LOESS.

    O teste corre só no Windows, onde os vetores foram gravados e onde a igualdade é bit a bit.
    Num contêiner Linux, com o mesmo código, o LOESS dá 0.16451113535487225 contra os
    0.16451113535487222 gravados, 1 ULP, e o SERRF é igual. A igualdade é exacta, e não
    aproximada, porque a execução parcial é exata numa plataforma; entre duas, ela não vale, e uma
    tolerância trocaria o que o teste afirma.
    """
    p, esperado = palpite_e_gravado_no_neg_c18
    assert {c: p.escores[c] for c in ("loess", "serrf")} == {
        c: esperado[c] for c in ("loess", "serrf")
    }

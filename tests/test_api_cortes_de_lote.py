"""Os cortes de lote nas rotas que corrigem.

Antes, a correção recebia só `n_batches`, e partia a corrida em blocos de tamanho quase igual
pela ordem de injeção: as fronteiras que o diagnóstico propõe paravam na tela. O campo novo,
`batch_cuts`, leva as fronteiras até o corretor, e este arquivo afirma as três coisas que o campo
promete:

- os cortes **chegam** ao corretor, e o corte dos blocos iguais dá o mesmo artefato que nenhum
  corte — `test_os_cortes_chegam_ao_corretor`;
- a medição dos corretores recusa **no disparo** lote com menos de três QCs —
  `test_a_medicao_recusa_lote_com_menos_de_tres_qcs`;
- corte de forma errada, ou fora da corrida, é `422` em toda rota que corrige —
  `test_corte_invalido_e_422_em_toda_rota_que_corrige`.

**A matriz é da factory, e não um arquivo de referência, de propósito.** O que se afirma é a
fiação — o campo atravessa a rota e muda a partição —, e a resposta decorre das ordens de injeção
que a factory declara: 40 injeções, oito QCs em 1, 6, 11, 16, 21, 26, 31 e 36. Nada aqui afirma o
que os arquivos do laboratório contêm.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.decision.labeling import Label
from tests.factories import base_spec, render_pair
from tests.reclamacao import reclama

MODO = "neg_hilic"
# Os blocos iguais de dois lotes sobre 40 injeções cortam antes da 21: quatro QCs de cada lado.
CORTE_DOS_BLOCOS_IGUAIS = 21


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def _arquivos() -> dict[str, tuple[str, bytes, str]]:
    medidas, lista = render_pair(base_spec(n_features=5, n_samples=40, n_qcs=8))
    return {
        "measurements": ("measurements.csv", medidas.encode("utf-8"), "text/csv"),
        "sample_list": ("Sample_List.csv", lista.encode("utf-8"), "text/csv"),
    }


def _corrige(cliente: TestClient, **campos: Any) -> bytes:
    resposta = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos(),
        data={"mode": MODO, "method": "loess", "n_batches": 2, **campos},
        teto_s=120,
    )
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    return bytes(resposta.content)


def test_os_cortes_chegam_ao_corretor(cliente: TestClient) -> None:
    """Par causal: o corte dos blocos iguais reproduz o artefato sem cortes; outro corte o muda.

    **A primeira metade é a âncora**, e ela separa duas coisas que a segunda sozinha não separa: se
    os cortes fossem ignorados, o artefato com o corte em 16 seria igual ao sem cortes; se fossem
    lidos com outra convenção — o corte a fechar o lote em vez de o abrir —, o corte em 21 já não
    reproduziria os blocos iguais.
    """
    sem_cortes = _corrige(cliente)
    nos_blocos_iguais = _corrige(cliente, batch_cuts=f"[{CORTE_DOS_BLOCOS_IGUAIS}]")
    noutro_corte = _corrige(cliente, batch_cuts="[16]")

    assert sem_cortes, "âncora: a correção não devolveu artefato nenhum"
    assert nos_blocos_iguais == sem_cortes, (
        "o corte na fronteira dos blocos iguais deu outro artefato: a convenção do corte não é a "
        "dos blocos — cada corte é a PRIMEIRA injeção do lote novo"
    )
    assert noutro_corte != sem_cortes, (
        "o corte em 16 deu o mesmo artefato que os blocos iguais: os cortes não chegaram ao "
        "corretor"
    )


def _rotulo(n_lotes: int) -> Label:
    return Label(
        prefix=("none", n_lotes),  # type: ignore[arg-type]
        scores={("none", n_lotes): 0.10, ("loess", n_lotes): 0.12, ("serrf", n_lotes): 0.13},  # type: ignore[dict-item]
        estimator="leave_one_qc_out",
        variant="outlier_removed",
    )


@pytest.fixture
def oraculo(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """O oráculo trocado por um que REGISTA o que a rota lhe pediu, sem correr corretor nenhum.

    A medição real são 21 ajustes, e o que se afirma aqui é a fiação: os prefixos e o vetor de
    lote que a rota passa. O duplo devolve um rótulo com o número de lotes que recebeu.
    """
    from refino.api import rotas

    pedidos: list[dict[str, Any]] = []

    def falso(
        self: Any,
        matrix: Any,
        *,
        batches: Any = None,
        depois_de_cada: Any = None,
        cancelled: Any = None,
    ) -> Label:
        pedidos.append({"prefixos": tuple(self.prefixes), "lotes": batches})
        if depois_de_cada is not None:
            for prefixo in self.prefixes:
                depois_de_cada(prefixo)
        return _rotulo(self.prefixes[0][1])

    monkeypatch.setattr(rotas.LabelingOracle, "label", falso)
    return pedidos


def test_a_medicao_recusa_lote_com_menos_de_tres_qcs(
    cliente: TestClient, oraculo: list[dict[str, Any]]
) -> None:
    """A guarda do mínimo de QCs por lote dita no disparo, e não depois de minutos de SERRF.

    **Par causal:** os mesmos dois lotes, com o corte em 31, deixam dois QCs no segundo — `422`, e o
    oráculo não é chamado; sem cortes, quatro e quatro — `202`, e o oráculo recebe os três prefixos
    com dois lotes e o vetor da partição.
    """
    recusa = cliente.post(
        "/matrizes/medir-corretor",
        files=_arquivos(),
        data={"mode": MODO, "n_batches": 2, "batch_cuts": "[31]"},
    )
    assert recusa.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, recusa.text
    assert "[6, 2]" in recusa.text, f"a recusa não diz os QCs de cada lote: {recusa.text}"
    assert oraculo == [], "o oráculo correu num pedido recusado"

    resposta = reclama(
        cliente,
        "/matrizes/medir-corretor",
        files=_arquivos(),
        data={"mode": MODO, "n_batches": 2},
        teto_s=120,
    )
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    assert corpo["n_lotes"] == 2 and corpo["qcs_por_lote"] == [4, 4], corpo
    assert len(oraculo) == 1, f"o oráculo foi chamado {len(oraculo)} vezes"
    assert oraculo[0]["prefixos"] == (("none", 2), ("loess", 2), ("serrf", 2))
    lotes = oraculo[0]["lotes"]
    assert lotes is not None and sorted(set(np.asarray(lotes).tolist())) == [0, 1], (
        f"o oráculo não recebeu a partição em dois lotes: {lotes}"
    )


def test_a_medicao_com_um_lote_passa_none_ao_oraculo(
    cliente: TestClient, oraculo: list[dict[str, Any]]
) -> None:
    """Um lote é `None` e não um vetor de zeros: o caminho que a medição tinha antes dos cortes."""
    resposta = reclama(
        cliente, "/matrizes/medir-corretor", files=_arquivos(), data={"mode": MODO}, teto_s=120
    )
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    assert resposta.json()["qcs_por_lote"] == [8]
    assert len(oraculo) == 1 and oraculo[0]["lotes"] is None
    assert oraculo[0]["prefixos"] == (("none", 1), ("loess", 1), ("serrf", 1))


# Os campos que cada rota exige além dos arquivos, com dois lotes.
_CAMPOS_POR_ROTA: dict[str, dict[str, Any]] = {
    "/matrizes/corrigir": {"mode": MODO, "n_batches": 2},
    "/matrizes/processar": {"mode": MODO, "n_batches": 2},
    "/matrizes/exportar": {"mode": MODO, "n_batches": 2},
    "/matrizes/medir-corretor": {"mode": MODO, "n_batches": 2},
    "/etapas/correcao": {
        "mode": MODO,
        "correction": "none",
        "n_batches": 2,
        "rsd_variant": "outlier_removed",
    },
    "/etapas/pre-processamento": {
        "mode": MODO,
        "correction": "none",
        "n_batches": 2,
        "rsd_variant": "outlier_removed",
        "imputation": "one_fifth_min",
        "sample_normalization": "none",
        "transformation": "none",
        "scaling": "none",
    },
}


@pytest.mark.parametrize("rota", sorted(_CAMPOS_POR_ROTA))
@pytest.mark.parametrize(
    ("cortes", "trecho"),
    [
        ("[21, 31]", "pedem 1 corte"),
        ("[true]", "inteiras"),
        ("21", "lista JSON"),
        ("[21", "batch_cuts"),
        ("[99]", "fora do intervalo"),
        ("[1]", "fora do intervalo"),
    ],
)
def test_corte_invalido_e_422_em_toda_rota_que_corrige(
    cliente: TestClient, rota: str, cortes: str, trecho: str
) -> None:
    """Fronteira de confiança: o campo chega como texto, e cada forma errada é recusada.

    **E é recusada NO DISPARO**, que é a resposta `422` do `POST` e não a da reclamação: corte fora
    da corrida é dado do pedido, não falha do cálculo. As duas últimas formas só a matriz recusa —
    é a metade da guarda que corre depois da ingestão.
    """
    resposta = cliente.post(
        rota, files=_arquivos(), data={**_CAMPOS_POR_ROTA[rota], "batch_cuts": cortes}
    )
    assert resposta.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, (
        f"{rota} aceitou `batch_cuts={cortes}`: {resposta.status_code} {resposta.text[:300]}"
    )
    assert trecho in resposta.text, f"{rota}: a recusa não diz o motivo: {resposta.text[:300]}"


def test_a_execucao_recusa_a_forma_errada_antes_do_banco(cliente: TestClient) -> None:
    """`/execucoes` confere a forma dos cortes junto com a spec, antes de tocar o banco.

    Sem banco nesta máquina, a ordem é a propriedade: uma forma errada que chegasse ao repositório
    reprovaria como `503` — *«o serviço não subiu»* — sobre uma requisição malformada.
    """
    resposta = cliente.post(
        "/execucoes",
        files=_arquivos(),
        data={
            "mode": MODO,
            "estudo": "d557",
            "correction": "none",
            "n_batches": 2,
            "rsd_variant": "outlier_removed",
            "imputation": "one_fifth_min",
            "sample_normalization": "none",
            "transformation": "none",
            "scaling": "none",
            "apresentacao": "nao_exibida",
            "batch_cuts": "[21, 31]",
        },
    )
    assert resposta.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, resposta.text
    assert "pedem 1 corte" in resposta.text

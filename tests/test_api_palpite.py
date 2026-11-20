"""`POST /matrizes/palpite` e a delegação `medida`.

O Palpite passou a ser a medição do RSD dos QCs fora da amostra em 200 features. Estes testes
cobrem a rota que o mede, a memória que o guarda, e os dois leitores dela: a delegação do
«Automático» (`auto=medida`) e o par da medição completa (`palpite` em `medir-corretor`).

**A medição verdadeira não corre aqui.** Ela custa minutos por arquivo, e o que ela devolve está
provado em `tests/unit/test_palpite_medido.py` — nos arquivos reais, igual à população gravada
restrita às 200 sorteadas. Aqui o medidor é um duplo que conta as chamadas e avisa o gancho uma vez
por corretor, que é a forma do oráculo de verdade (o gancho é o mesmo `depois_de_cada`).

**A isenção da rota ao proxy estático de `test_api_progresso.py` tem o substituto aqui**:
`test_o_gancho_avisa_uma_vez_por_corretor_e_o_progresso_anda_antes_do_fim`, com par causal.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

import refino.api.rotas as rotas
from refino.api.app import build_app
from refino.cli import AUTO_CAVEATS, AUTO_MEDIDA
from refino.decision.labeling import Label
from refino.decision.palpite import LIMIAR_DE_EMPATE_DO_PALPITE, PalpiteMedido, decide_o_palpite
from refino.io.readers import load_matrix
from tests.conftest import measurements_path, sample_list_path
from tests.reclamacao import espera, reclama

MODO = "neg_c18"


@pytest.fixture
def cliente() -> TestClient:
    return TestClient(build_app())


@pytest.fixture(autouse=True)
def memoria_limpa(monkeypatch: Any, memoria_de_correcoes_privada: None) -> None:
    """Cada teste começa sem Palpite guardado e sem trabalho terminado no registro.

    **As duas memórias são do processo**, e os testes deste arquivo disparam a mesma matriz com os
    mesmos campos: com o registro sujo, a deduplicação reclamaria o trabalho do teste anterior, o
    medidor do teste corrente não correria, e as asserções sobre ele passariam pelo resíduo. A
    memória dos Palpites é trocada por uma nova com `monkeypatch`, que devolve a original ao fim —
    nada do que se guarda aqui vaza para os testes de outros arquivos.
    """
    monkeypatch.setattr(rotas, "_PALPITES", OrderedDict())


def _arquivos(mode: str = MODO) -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{mode}.csv",
            measurements_path(mode).read_bytes(),
            "text/csv",
        ),
        "sample_list": (f"Sample_List_{mode}.csv", sample_list_path(mode).read_bytes(), "text/csv"),
    }


class _Medidor:
    """O duplo de `mede_o_palpite`: conta as chamadas e avisa o gancho como o oráculo avisa."""

    def __init__(
        self,
        escores: dict[str, float],
        *,
        avisos: tuple[str, ...] = ("none", "loess", "serrf"),
        barreira: threading.Event | None = None,
    ) -> None:
        self.escores = escores
        self.avisos = avisos
        self.barreira = barreira
        self.chamadas: list[dict[str, Any]] = []

    def __call__(
        self,
        matriz: Any,
        *,
        batches: Any = None,
        n_jobs: int = -1,
        depois_de_cada: Any = None,
        cancelled: Any = None,
        seed: int = 42,
    ) -> PalpiteMedido:
        self.chamadas.append({"batches": batches, "n_jobs": n_jobs, "seed": seed})
        for corretor in self.avisos:
            if depois_de_cada is not None:
                depois_de_cada((corretor, 1))
        if self.barreira is not None:
            assert self.barreira.wait(30), "o teste não soltou a barreira"
        return decide_o_palpite(self.escores, n_features_medidas=200)


ESCORES_CLAROS = {"none": 0.30, "loess": 0.10, "serrf": 0.20}


def test_a_rota_e_assincrona_e_devolve_o_palpite_medido(
    cliente: TestClient, monkeypatch: Any
) -> None:
    medidor = _Medidor(ESCORES_CLAROS)
    monkeypatch.setattr(rotas, "mede_o_palpite", medidor)

    resposta = reclama(cliente, "/matrizes/palpite", files=_arquivos(), data={"mode": MODO})

    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    assert len(medidor.chamadas) == 1, "a rota não mediu, ou mediu mais de uma vez"
    assert corpo["correction_proposta"] == "loess"
    assert corpo["abstem"] is False
    assert corpo["avaliavel"] is True
    assert corpo["n_features_medidas"] == 200
    assert corpo["limiar_de_empate"] == LIMIAR_DE_EMPATE_DO_PALPITE
    assert corpo["escores"] == ESCORES_CLAROS
    assert corpo["n_lotes"] == 1 and corpo["qcs_por_lote"] == [7]
    assert corpo["origem"] == "modelo"
    assert corpo["ressalva_da_politica"] == AUTO_CAVEATS[AUTO_MEDIDA].resumo
    assert "loess" in corpo["rationale"]
    assert len(rotas._PALPITES) == 1, "o Palpite medido não ficou guardado para a delegação"


def test_o_gancho_avisa_uma_vez_por_corretor_e_o_progresso_anda_antes_do_fim(
    cliente: TestClient, monkeypatch: Any
) -> None:
    """O substituto da isenção ao proxy estático — e é par causal, contra um gancho que avisa uma.

    O progresso é lido com o trabalho PRESO na barreira, antes de `conclui` forçar `feito` ao total:
    é isso que faz a leitura não ser vácua.
    """

    def feito_preso(avisos: tuple[str, ...]) -> int:
        barreira = threading.Event()
        medidor = _Medidor(ESCORES_CLAROS, avisos=avisos, barreira=barreira)
        monkeypatch.setattr(rotas, "mede_o_palpite", medidor)
        rotas._PALPITES.clear()
        # O segundo disparo tem a MESMA chave do primeiro, e a deduplicação o reclamaria: a thread
        # não nasceria, o medidor novo não correria, e a leitura seria a do trabalho já concluído.
        rotas.TRABALHOS.esquece_terminados_alem_de(0)
        disparo = cliente.post("/matrizes/palpite", files=_arquivos(), data={"mode": MODO})
        assert disparo.status_code == status.HTTP_202_ACCEPTED, disparo.text
        identificador = disparo.json()["identificador"]
        feito = -1
        limite = time.monotonic() + 30
        while time.monotonic() < limite:
            leitura = cliente.get(f"/trabalhos/{identificador}").json()
            feito = leitura["feito"]
            if feito >= len(avisos):
                break
            time.sleep(0.02)
        barreira.set()
        assert espera(cliente, identificador).status_code == status.HTTP_200_OK
        assert len(medidor.chamadas) == 1, (
            "o disparo foi reclamado, e o medidor deste par não correu"
        )
        return feito

    assert feito_preso(("none", "loess", "serrf")) == 3
    assert feito_preso(("none",)) == 1, "o progresso andou sem o gancho avisar: a leitura é vácua"


def test_a_delegacao_medida_usa_o_palpite_guardado_sem_medir_de_novo(
    cliente: TestClient, monkeypatch: Any
) -> None:
    medidor = _Medidor({"none": 0.10, "loess": 0.30, "serrf": 0.20})
    monkeypatch.setattr(rotas, "mede_o_palpite", medidor)
    assert (
        reclama(cliente, "/matrizes/palpite", files=_arquivos(), data={"mode": MODO}).status_code
        == status.HTTP_200_OK
    )

    disparo = cliente.post(
        "/matrizes/corrigir", files=_arquivos(), data={"mode": MODO, "auto": AUTO_MEDIDA}
    )

    assert disparo.status_code == status.HTTP_202_ACCEPTED, disparo.text
    delegacao = disparo.json()["delegacao"]
    assert delegacao["politica"] == AUTO_MEDIDA
    assert delegacao["correction"] == "none"
    assert delegacao["caveat"]["resumo"] == AUTO_CAVEATS[AUTO_MEDIDA].resumo
    assert len(medidor.chamadas) == 1, "a delegação mediu de novo um Palpite que estava guardado"
    espera(cliente, disparo.json()["identificador"])


def test_sem_palpite_guardado_a_delegacao_mede_e_guarda(
    cliente: TestClient, monkeypatch: Any
) -> None:
    medidor = _Medidor({"none": 0.10, "loess": 0.30, "serrf": 0.20})
    monkeypatch.setattr(rotas, "mede_o_palpite", medidor)

    disparo = cliente.post(
        "/matrizes/corrigir", files=_arquivos(), data={"mode": MODO, "auto": AUTO_MEDIDA}
    )

    assert disparo.status_code == status.HTTP_202_ACCEPTED, disparo.text
    assert disparo.json()["delegacao"]["correction"] == "none"
    assert len(medidor.chamadas) == 1
    assert len(rotas._PALPITES) == 1
    espera(cliente, disparo.json()["identificador"])


def test_o_trabalho_usa_o_palpite_guardado_e_nao_mede_de_novo(
    cliente: TestClient, monkeypatch: Any
) -> None:
    """Com o Palpite guardado e o registro de trabalhos podado, o disparo novo não mede outra vez.

    É o que deixa a janela pedir o Palpite antes de todo disparo delegado: o registro de trabalhos
    poda os concluídos, a memória dos Palpites é outra, e medir de novo daria o mesmo número por
    mais um minuto. **Par causal:** com a memória vazia, o mesmo disparo mede.
    """
    medidor = _Medidor(ESCORES_CLAROS)
    monkeypatch.setattr(rotas, "mede_o_palpite", medidor)
    matriz = load_matrix(measurements_path(MODO), sample_list_path(MODO), mode=MODO)
    rotas._guarda_o_palpite(
        rotas._chave_do_palpite(matriz, 1, None),
        decide_o_palpite({"none": 0.1, "loess": 0.3, "serrf": 0.2}, n_features_medidas=200),
    )

    com = reclama(cliente, "/matrizes/palpite", files=_arquivos(), data={"mode": MODO}).json()
    assert com["correction_proposta"] == "none", "a resposta não é a do Palpite guardado"
    assert medidor.chamadas == [], "o trabalho mediu um Palpite que estava guardado"

    rotas._PALPITES.clear()
    rotas.TRABALHOS.esquece_terminados_alem_de(0)
    sem = reclama(cliente, "/matrizes/palpite", files=_arquivos(), data={"mode": MODO}).json()
    assert sem["correction_proposta"] == "loess"
    assert len(medidor.chamadas) == 1, "ÂNCORA: sem Palpite guardado o trabalho tinha de medir"


def test_a_segunda_opiniao_chega_nula_em_stand_by_e_volta_quando_religada(
    cliente: TestClient, monkeypatch: Any
) -> None:
    """A segunda opinião está em stand-by, e não apagada: religada, ela volta nesta resposta.

    O painel da leitura mostra o Palpite medido com o formatador que mostrava a árvore, e é ele que
    desenha a segunda opinião quando ela vem. **Par causal sobre o interruptor**, com o mesmo duplo.
    """
    from refino.io import politica

    monkeypatch.setattr(rotas, "mede_o_palpite", _Medidor(ESCORES_CLAROS))
    desligada = reclama(cliente, "/matrizes/palpite", files=_arquivos(), data={"mode": MODO})
    assert desligada.json()["segunda_opiniao"] is None
    assert desligada.json()["rotulo_da_politica"] == "Pela estabilidade dos QCs"

    monkeypatch.setattr(politica, "SEGUNDA_OPINIAO_ATIVA", True)
    rotas._PALPITES.clear()
    rotas.TRABALHOS.esquece_terminados_alem_de(0)
    ligada = reclama(cliente, "/matrizes/palpite", files=_arquivos(), data={"mode": MODO}).json()
    segunda = ligada["segunda_opiniao"]
    assert isinstance(segunda, dict), f"religada, a segunda opinião não veio: {segunda!r}"
    assert segunda["correction_proposta"] in {"none", "loess", "serrf"}
    assert segunda["caminho"].strip()


def test_lotes_diferentes_sao_palpites_diferentes(cliente: TestClient, monkeypatch: Any) -> None:
    """A chave leva os lotes: um Palpite de lote único não serve uma execução com dois lotes."""
    matriz = load_matrix(measurements_path(MODO), sample_list_path(MODO), mode=MODO)
    assert rotas._chave_do_palpite(matriz, 1, None) != rotas._chave_do_palpite(matriz, 2, None)
    assert rotas._chave_do_palpite(matriz, 2, None) != rotas._chave_do_palpite(matriz, 2, (30,))


def test_o_par_e_o_do_disparo_e_um_palpite_medido_depois_nao_o_reescreve(
    cliente: TestClient, monkeypatch: Any
) -> None:
    """O par da medição completa é um PRÉ-REGISTRO, e o reclamado o mostra.

    `medir-corretor` lê o Palpite guardado **no disparo**. Uma medição pedida antes de haver
    Palpite fica com `palpite = None`, e pedir a MESMA medição depois de o Palpite chegar devolve
    o trabalho já feito, com o par que ele tinha — o Palpite que chegou depois não vira predição
    de uma medição que já existia. Uma medição NOVA, com outros campos, leva o Palpite guardado.
    """

    def label(self: Any, matrix: Any, **kwargs: Any) -> Label:
        gancho = kwargs.get("depois_de_cada")
        for prefixo in self.prefixes:
            if gancho is not None:
                gancho(prefixo)
        return Label(
            ("serrf", 1),
            {("none", 1): 0.30, ("loess", 1): 0.40, ("serrf", 1): 0.20},
            "leave_one_qc_out_rsd",
            "outlier_removed",
        )

    monkeypatch.setattr(rotas.LabelingOracle, "label", label)
    dados = {"mode": MODO, "n_sampled_features": "40"}

    antes = reclama(cliente, "/matrizes/medir-corretor", files=_arquivos(), data=dados).json()
    assert antes["palpite"] is None and antes["palpite_acertou"] is None

    matriz = load_matrix(measurements_path(MODO), sample_list_path(MODO), mode=MODO)
    guardado = decide_o_palpite(
        {"none": 0.30, "loess": 0.10, "serrf": 0.20}, n_features_medidas=200
    )
    rotas._guarda_o_palpite(rotas._chave_do_palpite(matriz, 1, None), guardado)

    reclamado = reclama(cliente, "/matrizes/medir-corretor", files=_arquivos(), data=dados).json()
    assert reclamado["palpite"] is None, "o Palpite medido depois reescreveu o par da medição"

    nova = reclama(
        cliente,
        "/matrizes/medir-corretor",
        files=_arquivos(),
        data={"mode": MODO, "n_sampled_features": "41"},
    ).json()
    assert nova["palpite"] == "loess", "a medição nova não leu o Palpite guardado"
    assert nova["vencedor"] == "serrf"
    assert nova["palpite_acertou"] is False


def test_n_batches_menor_que_um_da_422(cliente: TestClient) -> None:
    resposta = cliente.post(
        "/matrizes/palpite", files=_arquivos(), data={"mode": MODO, "n_batches": "0"}
    )
    assert resposta.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_o_lote_sem_qc_nao_recusa_e_o_de_um_qc_recusa(
    cliente: TestClient, monkeypatch: Any
) -> None:
    """A recusa no disparo é a guarda do mínimo de QCs por lote, e ela não conta o lote sem QC.

    Os QCs do `neg_c18` estão nas injeções 1, 12, 23, 34, 45, 56 e 68. **Cortes em 24 e 34** dão
    `[3, 0, 4]` QCs por lote: o lote do meio não tem QC a reter, a medição o aceita, e a primeira
    escrita, `min(qcs) < 3`, recusava. **Corte em 60** dá `[6, 1]`: um QC só, e a recusa fica.
    """
    medidor = _Medidor(ESCORES_CLAROS)
    monkeypatch.setattr(rotas, "mede_o_palpite", medidor)
    aceito = cliente.post(
        "/matrizes/palpite",
        files=_arquivos(),
        data={"mode": MODO, "n_batches": "3", "batch_cuts": "[24, 34]"},
    )
    assert aceito.status_code == status.HTTP_202_ACCEPTED, aceito.text
    recusado = cliente.post(
        "/matrizes/palpite",
        files=_arquivos(),
        data={"mode": MODO, "n_batches": "2", "batch_cuts": "[60]"},
    )
    assert recusado.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT, recusado.text
    assert "[6, 1]" in recusado.json()["detail"], recusado.json()


def test_a_semente_da_execucao_chega_ao_palpite_e_separa_a_memoria(
    cliente: TestClient, monkeypatch: Any
) -> None:
    """A semente entra na medição e na identidade do Palpite guardado.

    **Par causal sobre a memória:** o Palpite medido com `seed=7` não responde pela delegação de uma
    correção com `seed=42` — o disparo mede de novo, com 42 —, e responde pela de `seed=7`, sem
    medir. Sem a semente na chave, as duas leituras dariam o mesmo Palpite.
    """
    medidor = _Medidor(ESCORES_CLAROS)
    monkeypatch.setattr(rotas, "mede_o_palpite", medidor)

    sete = reclama(
        cliente, "/matrizes/palpite", files=_arquivos(), data={"mode": MODO, "seed": "7"}
    )
    assert sete.status_code == status.HTTP_200_OK, sete.text
    assert [c["seed"] for c in medidor.chamadas] == [7]

    mesmo = cliente.post(
        "/matrizes/corrigir",
        files=_arquivos(),
        data={"mode": MODO, "auto": AUTO_MEDIDA, "seed": "7"},
    )
    assert mesmo.status_code == status.HTTP_202_ACCEPTED, mesmo.text
    espera(cliente, mesmo.json()["identificador"])
    assert [c["seed"] for c in medidor.chamadas] == [7], "com a mesma semente não mede de novo"

    outra = cliente.post(
        "/matrizes/corrigir", files=_arquivos(), data={"mode": MODO, "auto": AUTO_MEDIDA}
    )
    assert outra.status_code == status.HTTP_202_ACCEPTED, outra.text
    espera(cliente, outra.json()["identificador"])
    assert [c["seed"] for c in medidor.chamadas] == [7, 42], (
        "a delegação com a semente padrão reaproveitou o Palpite da semente 7"
    )

"""O `auto` de `POST /matrizes/corrigir`: a delegação automática pela rota.

## Por que a FRONTEIRA e não o `POST` inteiro, na maioria

É a mesma medida que `test_ca_12_12_a_rota_de_artefato_aceita_serrf_na_fronteira` já regista: com
`method=serrf` o mesmo `POST` executa um SERRF real — **86 s no `neg_c18`** —, e a delegação escolhe
`serrf` na maioria das matrizes porque ele é a classe majoritária. Um arquivo inteiro de testes de
rota custaria minutos por asserção.

**A propriedade que se verifica aqui é a DELEGAÇÃO**, e ela é de `_delegacao_ou_422`: o rótulo
recusado, a política escolhida, a justificativa montada e o recorte anexado. O caminho ponta a
ponta tem **um** teste, marcado `slow`, e ele existe porque a fronteira não prova que a rota a
chama.

## Os quatro desfechos de `_delegacao_ou_422`

| Entrada | Desfecho | Ação do cliente |
|---|---|---|
| `aprendida` | `DelegacaoResposta` com margem predita | segue |
| `constante` | `DelegacaoResposta` que **se abstém** | segue, sabendo que a margem não foi medida |
| `medida` | `DelegacaoResposta` com a margem MEDIDA em 200 features | segue |
| rótulo fora do vocabulário | `422` | corrige o pedido |
| artefato ilegível | `503` | tenta sem `auto`, ou avisa quem mantém |

O `422` e o `503` são separados de propósito: num o cliente errou, no outro não. Um
código só tornaria as duas causas indistinguíveis, e elas pedem ações opostas.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import refino.api.rotas as rotas
from refino.api.app import build_app
from refino.api.rotas import _delegacao_ou_422
from refino.cli import AUTO_APRENDIDA, AUTO_CAVEATS, AUTO_CONSTANTE, AUTO_MEDIDA, AUTO_VALIDOS
from refino.decision.palpite import decide_o_palpite
from refino.domain.matrix import RefinoMatrix
from refino.io.readers import load_matrix
from tests.conftest import measurements_path, sample_list_path

MODO = "neg_c18"


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


@pytest.fixture(scope="module")
def matriz() -> RefinoMatrix:
    """A matriz real do `neg_c18`, lida uma vez.

    Real e não fabricada: a política decide a partir de um `DefectProfile`, e um perfil de matriz
    gerada por factory mediria a factory.
    """
    return load_matrix(measurements_path(MODO), sample_list_path(MODO), MODO)


def _arquivos(mode: str = MODO) -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{mode}.csv",
            measurements_path(mode).read_bytes(),
            "text/csv",
        ),
        "sample_list": (f"Sample_List_{mode}.csv", sample_list_path(mode).read_bytes(), "text/csv"),
    }


# ---------------------------------------------------------------------------
# A fronteira
# ---------------------------------------------------------------------------


def test_rotulo_invalido_da_422_e_nomeia_os_aceitos(matriz: RefinoMatrix) -> None:
    """A guarda de fronteira nesta superfície: o `422` é a tradução do `exit 2` da CLI.

    Sem ela, `auto=constant` — sem o `e` — cairia num `else` razoável, correria a política
    aprendida, e a resposta declararia uma delegação que o cliente não pediu.
    """
    with pytest.raises(HTTPException) as erro:
        _delegacao_ou_422("constant", matriz)

    assert erro.value.status_code == 422
    for rotulo in AUTO_VALIDOS:
        assert rotulo in str(erro.value.detail)


def test_as_duas_politicas_produzem_delegacoes_que_divergem(matriz: RefinoMatrix) -> None:
    """A planta cai onde as duas diferem, e não onde elas coincidem.

    As duas escolhem `serrf` neste modo. Afirmar `correction` seria vácuo — passaria com uma
    implementação que ignorasse o rótulo por completo. O que diverge é a confiança: a aprendida
    prediz a margem pela segunda árvore do artefato, e a constante não tem estimador de margem.
    """
    aprendida = _delegacao_ou_422(AUTO_APRENDIDA, matriz)
    constante = _delegacao_ou_422(AUTO_CONSTANTE, matriz)

    assert aprendida.politica == AUTO_APRENDIDA
    assert constante.politica == AUTO_CONSTANTE

    assert constante.assessable is False
    assert constante.abstains is True
    assert aprendida.confidence != constante.confidence, (
        "as duas politicas relataram a MESMA confianca: o rotulo pode nao estar a ser lido"
    )


def test_a_delegacao_carrega_a_justificativa_e_o_recorte(matriz: RefinoMatrix) -> None:
    """A delegação traz a justificativa e, ao lado dela, o recorte que a limita.

    Uma resposta que dissesse *"escolhi `serrf` com confiança 0,94"* sem o número que a desautoriza
    (70,0% contra 55,0% da constante, sem significância) afirmaria mais do que o mecanismo
    sustenta, na superfície que a tela lê.
    """
    delegacao = _delegacao_ou_422(AUTO_APRENDIDA, matriz)

    assert delegacao.rationale.strip()
    assert delegacao.evidence, "a justificativa veio sem as caracteristicas que a sustentam"
    assert delegacao.caveat.resumo == AUTO_CAVEATS[AUTO_APRENDIDA].resumo, (
        "a ressalva foi REESCRITA aqui em vez de vir da fonte unica, e as duas vao divergir"
    )
    assert delegacao.caveat.item == f"--auto {AUTO_APRENDIDA}"


def test_a_ressalva_da_rota_e_o_mesmo_texto_da_cli(matriz: RefinoMatrix, monkeypatch: Any) -> None:
    """Um segundo texto para a mesma ressalva divergiria do primeiro.

    A CLI imprime e a rota devolve a mesma string. Sem esta asserção, alguém "melhora" o texto de
    um lado e a tela passa a contar uma história que o terminal não conta.

    `medida` mede o Palpite, e a medição de verdade custa minutos aqui: o medidor vira um
    duplo que devolve um Palpite decidido, e a memória dos Palpites começa vazia e volta ao fim.
    O que se afirma é o texto da ressalva, que não depende do número medido.
    """
    monkeypatch.setattr(rotas, "_PALPITES", OrderedDict())
    monkeypatch.setattr(
        rotas,
        "mede_o_palpite",
        lambda matriz, **_: decide_o_palpite(
            {"none": 0.3, "loess": 0.2, "serrf": 0.1}, n_features_medidas=200
        ),
    )
    for rotulo in AUTO_VALIDOS:
        delegacao = _delegacao_ou_422(rotulo, matriz)
        da_cli = AUTO_CAVEATS[rotulo]
        assert delegacao.caveat.detalhe == da_cli.detalhe
        assert delegacao.caveat.evidencia == da_cli.evidencia
        assert delegacao.caveat.portao_inerte == da_cli.portao_inerte


def test_medida_numa_matriz_de_dois_qcs_da_422_e_nao_500() -> None:
    """Sem duplo: a recusa é a da régua de verdade. Com dois QCs, esconder um deixa um
    só para ajustar, e a régua levanta; sem a tradução, a delegação viraria `500`. A mesma
    matriz com `constante` delega: o que se recusa é a decisão medida, não o arquivo."""
    from pathlib import Path

    fixtures = Path(__file__).resolve().parent.parent / "data" / "fixtures"
    pequena = load_matrix(
        fixtures / "tiny_drift.csv", fixtures / "tiny_drift_sample_list.csv", "tiny"
    )

    with pytest.raises(HTTPException) as erro:
        _delegacao_ou_422(AUTO_MEDIDA, pequena)

    assert erro.value.status_code == 422
    assert "QCs" in str(erro.value.detail), erro.value.detail
    assert AUTO_APRENDIDA in str(erro.value.detail), "a recusa não diz a saída que resta"
    assert _delegacao_ou_422(AUTO_CONSTANTE, pequena).politica == AUTO_CONSTANTE


# ---------------------------------------------------------------------------
# A rota inteira — UM teste, `slow`, porque a fronteira não prova que ela é chamada
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_a_rota_devolve_a_delegacao_no_202_e_sem_auto_nao_devolve(cliente: TestClient) -> None:
    """**Par causal na rota: dois `POST` que diferem só pelo campo `auto`.**

    Os testes acima verificam a função, e a pergunta é se a rota a chama. Afirmar só o caso com
    `auto` seria satisfeito por uma implementação que devolvesse a delegação sempre; afirmar só o
    caso sem, por uma que nunca a devolvesse.

    **`slow` porque este `POST` dispara um SERRF real** — 86 s medidos no `neg_c18`. O `202`
    volta na hora, e é isso que se afirma: a decisão é síncrona, só a correção é longa.
    """
    com = cliente.post(
        "/matrizes/corrigir", files=_arquivos(), data={"mode": MODO, "auto": AUTO_CONSTANTE}
    )
    sem = cliente.post(
        "/matrizes/corrigir", files=_arquivos(), data={"mode": MODO, "method": "none"}
    )

    assert (com.status_code, sem.status_code) == (202, 202), (com.text[:300], sem.text[:300])
    assert com.json()["delegacao"] is not None
    assert com.json()["delegacao"]["correction"] == "serrf"
    assert sem.json().get("delegacao") is None, "a rota devolveu delegacao sem ninguem ter delegado"


@pytest.mark.slow
def test_a_rota_recusa_rotulo_invalido_com_422(cliente: TestClient) -> None:
    """A guarda alcançada PELA ROTA, e não só pela função.

    `slow` porque a recusa acontece **depois** da ingestão: quem delega paga a leitura do arquivo
    para descobrir um `auto` inválido, e essa ordem é obrigatória — a política precisa da matriz
    lida. É o preço declarado de a decisão depender do dado.
    """
    resposta = cliente.post(
        "/matrizes/corrigir", files=_arquivos(), data={"mode": MODO, "auto": "constant"}
    )

    assert resposta.status_code == 422, resposta.text[:300]
    assert AUTO_CONSTANTE in resposta.text

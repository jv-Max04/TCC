"""O SERRF da revisão de 24/10/2026 — S1, S2, C04 e C08 —, cada um sobre o caminho de PRODUÇÃO.

**A revisão achou os dois primeiros num ponto que nenhum teste olhava: o argumento que a produção
passa.** `correlation_matrices` e `subtractive_outlier_repair` tinham testes, e passavam — testados
isolados, com o argumento certo. O que estava errado era a chamada dentro de `SerrfCorrector`. Por
isso estes testes espiam a CHAMADA, trocando a função no módulo do corretor por uma que registra o
que recebe e devolve o que a original devolveria.

`n_jobs=1` em todos, e é o que faz a espia funcionar: com `loky`, a correção por feature corre
noutro processo, onde a troca feita aqui não existe.

**A matriz é sintética e pequena, e é para a CHAMADA, nunca para contagem** (regra 1 do README): o
que se afirma é qual vetor chega a cada função, e isso é propriedade do código, não do dado.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import refino.correction.serrf as modulo
from refino.api.app import build_app
from refino.cli import EXIT_INPUT_REJECTED, main
from refino.correction._serrf_steps import subtractive_outlier_repair
from refino.correction.serrf import IntensidadeNegativaError, SerrfCorrector
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.interface import cliente
from tests.factories import base_spec, render_pair
from tests.reclamacao import espera


def _matriz(
    *, n_features: int = 24, validate: int = 0, semente: int = 5, zeros: bool = True
) -> RefinoMatrix:
    """Seis QCs, oito amostras de estudo e `validate` opcionais, com zeros espalhados."""
    tipos = ["qc" if i % 3 == 0 else "sample" for i in range(14)] + ["validate"] * validate
    gerador = np.random.default_rng(semente)
    valores = gerador.lognormal(6.0, 0.4, size=(len(tipos), n_features))
    if zeros:
        valores[gerador.random(valores.shape) < 0.06] = 0.0
    amostras = tuple(
        SampleMetadata(
            name=f"{tipo[0].upper()}{i:02d}",
            injection_order=i + 1,
            sample_type=tipo,  # type: ignore[arg-type]
            group="QC" if tipo == "qc" else "G1",
        )
        for i, tipo in enumerate(tipos)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"{i}.0_{100 + i}.0m/z" for i in range(n_features)),
        samples=amostras,
        mode="neg_c18",
    )


# ---------------------------------------------------------------------------
# S1 — o passo 2 recebe a matriz de ANTES do passo 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("icado", [True, False], ids=["icado", "por_feature"])
def test_s1_as_correlacoes_recebem_a_matriz_de_antes_do_passo_1(
    monkeypatch: pytest.MonkeyPatch, icado: bool
) -> None:
    """As duas rotas do passo 2 — a içada, que é o padrão, e a por feature.

    A primeira asserção é a âncora de que as duas matrizes DIFEREM: sem zeros, o passo 1 não muda
    nada, e o teste passaria com qualquer das duas.
    """
    matriz = _matriz()
    antes = np.asarray(matriz.values).T
    assert (antes == 0.0).any(), "âncora: sem zeros o passo 1 não muda nada"

    recebidas: list[np.ndarray] = []
    original = modulo.correlation_matrices

    def espia(matrix: np.ndarray, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        recebidas.append(np.array(matrix, copy=True))
        return original(matrix, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(modulo, "correlation_matrices", espia)
    SerrfCorrector(n_jobs=1, hoist_correlations=icado).fit_transform(matriz, seed=3)

    assert recebidas, "âncora: o passo 2 não foi chamado"
    for recebida in recebidas:
        np.testing.assert_array_equal(recebida, antes)


# ---------------------------------------------------------------------------
# S2 — o passo 9 usa o valor CRU do lote, e o normalizado onde não marca
# ---------------------------------------------------------------------------


def test_s2_a_formula_subtrativa_usa_o_cru_e_reverte_ao_cru() -> None:
    """`tools.R:1162-1163`, com números que separam o cru do normalizado.

    Posição 0: não marcada, positiva — fica o normalizado. Posição 1: marcada — `cru - desloca`.
    Posição 2: marcada e negativa depois do deslocamento — volta ao CRU, e não ao normalizado.
    """
    normalizado = np.array([10.0, 20.0, 30.0])
    cru = np.array([1.0, 500.0, 2.0])
    previsto = np.zeros(3)
    global_amostras = np.array([100.0, 100.0, 100.0])
    marcado = np.array([False, True, True])

    resultado = subtractive_outlier_repair(normalizado, cru, previsto, global_amostras, marcado)

    deslocamento = np.mean(cru) - np.median(global_amostras)
    assert resultado[0] == 10.0
    assert resultado[1] == pytest.approx(500.0 - deslocamento)
    assert 2.0 - deslocamento < 0, "âncora: a posição 2 tem de ficar negativa para reverter"
    assert resultado[2] == 2.0


def test_s2_a_producao_passa_o_cru_do_lote_ao_passo_9(monkeypatch: pytest.MonkeyPatch) -> None:
    """O vetor cru que chega é a coluna do lote depois do passo 1, e não o normalizado."""
    matriz = _matriz()
    chamadas: list[tuple[np.ndarray, np.ndarray]] = []
    original = modulo.subtractive_outlier_repair

    def espia(normalizado: np.ndarray, cru: np.ndarray, *resto: np.ndarray) -> np.ndarray:
        chamadas.append((np.array(normalizado, copy=True), np.array(cru, copy=True)))
        return original(normalizado, cru, *resto)

    passo1: list[np.ndarray] = []
    original_passo1 = SerrfCorrector._step1

    def espia_passo1(self: SerrfCorrector, values: np.ndarray, **kwargs: object) -> np.ndarray:
        saida = original_passo1(self, values, **kwargs)  # type: ignore[arg-type]
        passo1.append(saida)
        return saida

    monkeypatch.setattr(modulo, "subtractive_outlier_repair", espia)
    monkeypatch.setattr(SerrfCorrector, "_step1", espia_passo1)
    SerrfCorrector(n_jobs=1).fit_transform(matriz, seed=3)

    assert chamadas and len(passo1) == 1, "âncora: o passo 9 e o passo 1 correram"
    amostras = np.asarray(matriz.sample_mask)
    crus_esperados = [passo1[0][amostras, j] for j in range(matriz.n_features)]
    diferem = 0
    for normalizado, cru in chamadas:
        assert any(np.array_equal(cru, esperado) for esperado in crus_esperados), (
            "o passo 9 recebeu um vetor que não é a coluna do lote depois do passo 1"
        )
        diferem += not np.allclose(normalizado, cru)
    assert diferem, "âncora: o normalizado e o cru têm de diferir, senão o par não separa nada"


# ---------------------------------------------------------------------------
# C04 — `validate` volta inalterada e não vaza para as amostras
# ---------------------------------------------------------------------------


def test_c04_validate_volta_inalterada() -> None:
    matriz = _matriz(validate=2)
    validate = np.asarray([a.sample_type == "validate" for a in matriz.samples])

    saida = SerrfCorrector(n_jobs=1).fit_transform(matriz, seed=3).matrix

    np.testing.assert_array_equal(
        np.asarray(saida.values)[validate], np.asarray(matriz.values)[validate]
    )


def test_c04_mudar_so_as_validate_nao_muda_as_amostras() -> None:
    """O vazamento que a revisão mediu: 276 de 295 células de amostra mudavam."""
    matriz = _matriz(validate=2)
    validate = np.asarray([a.sample_type == "validate" for a in matriz.samples])
    outros = np.array(np.asarray(matriz.values), copy=True)
    outros[validate] = np.random.default_rng(99).lognormal(9.0, 1.0, size=outros[validate].shape)
    trocada = RefinoMatrix(
        values=outros, feature_names=matriz.feature_names, samples=matriz.samples, mode=matriz.mode
    )

    a = SerrfCorrector(n_jobs=1).fit_transform(matriz, seed=3).matrix
    b = SerrfCorrector(n_jobs=1).fit_transform(trocada, seed=3).matrix

    np.testing.assert_array_equal(np.asarray(a.values)[~validate], np.asarray(b.values)[~validate])


# ---------------------------------------------------------------------------
# C08 — intensidade abaixo de -0,1 ao lado de zero: recusa com o nome, e não `scale < 0`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("negativo", "recusa"), [(-12.0, True), (-0.05, False)], ids=["abaixo", "acima"]
)
def test_c08_intensidade_negativa_recusa_pelo_nome(negativo: float, recusa: bool) -> None:
    """-12 deixa o desvio do passo 1 negativo; -0,05 não, e corre — a direção oposta."""
    matriz = _matriz(zeros=False)
    valores = np.array(np.asarray(matriz.values), copy=True)
    amostras = np.flatnonzero(np.asarray(matriz.sample_mask))
    valores[amostras[0], 3] = 0.0
    valores[amostras[1], 3] = negativo
    plantada = RefinoMatrix(
        values=valores, feature_names=matriz.feature_names, samples=matriz.samples, mode=matriz.mode
    )

    if recusa:
        with pytest.raises(IntensidadeNegativaError, match=r"3\.0_103\.0m/z"):
            SerrfCorrector(n_jobs=1).fit_transform(plantada, seed=3)
    else:
        SerrfCorrector(n_jobs=1).fit_transform(plantada, seed=3)


# ---------------------------------------------------------------------------
# C08 nas fronteiras: `422` que o cliente lê como causa do dado, e o código 4 na CLI
# ---------------------------------------------------------------------------
#
# **As fronteiras são três, e cada uma tem o seu teste**: o trabalho de fundo, onde o SERRF corre
# quase sempre (`_executa_trabalho`); o DISPARO, onde `auto=medida` mede o Palpite antes de
# despachar (o tradutor de `api/app.py`); e os dois subcomandos da CLI que corrigem. O `processar`
# não corrige, e por isso não entra.

FEATURE_PLANTADA = 3


def _par_com_negativo(tmp_path: Path) -> tuple[Path, Path, str]:
    """Um par da factory com zero e -12 na mesma feature, nas amostras de estudo.

    `base_spec` põe os QCs primeiro; as colunas 6 e 7 são amostras. O nome volta junto, porque é
    ele que a recusa tem de dizer.
    """
    spec = base_spec(n_features=12, n_samples=14, n_qcs=6)
    valores = [list(linha) for linha in spec.values]
    valores[FEATURE_PLANTADA][6] = 0.0
    valores[FEATURE_PLANTADA][7] = -12.0
    plantada = replace(spec, values=tuple(tuple(linha) for linha in valores))
    medicoes, lista = render_pair(plantada)
    caminho_m, caminho_s = tmp_path / "measurements.csv", tmp_path / "Sample_List.csv"
    caminho_m.write_text(medicoes, encoding="utf-8", newline="")
    caminho_s.write_text(lista, encoding="utf-8", newline="")
    return caminho_m, caminho_s, spec.feature_names[FEATURE_PLANTADA]


def _arquivos(caminho_m: Path, caminho_s: Path) -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": ("m.csv", caminho_m.read_bytes(), "text/csv"),
        "sample_list": ("s.csv", caminho_s.read_bytes(), "text/csv"),
    }


@pytest.fixture(scope="module")
def cliente_http() -> TestClient:
    """`raise_server_exceptions=False`: o `500` da árvore anterior é o que o par compara."""
    return TestClient(build_app(), raise_server_exceptions=False)


@pytest.mark.parametrize(
    ("campos", "disparo_esperado"),
    [
        pytest.param({"method": "serrf", "paralelo": "false"}, 202, id="trabalho_de_fundo"),
        pytest.param({"auto": "medida", "paralelo": "false"}, 422, id="disparo_auto_medida"),
    ],
)
def test_c08_a_rota_recusa_com_422_que_o_cliente_le_como_dado(
    cliente_http: TestClient, tmp_path: Path, campos: dict[str, str], disparo_esperado: int
) -> None:
    """A pergunta é feita ao CLIENTE, como no C05: só a marca faz a tela dizer que é o arquivo.

    **O código do DISPARO é a âncora de que cada caso exercita a fronteira que nomeia:** `202` quer
    dizer que a recusa veio do trabalho de fundo, e `422` no próprio `POST`, que veio do tradutor do
    aplicativo — sem ela, os dois casos podiam passar pelo mesmo caminho.
    """
    caminho_m, caminho_s, nome = _par_com_negativo(tmp_path)

    disparo = cliente_http.post(
        "/matrizes/corrigir",
        files=_arquivos(caminho_m, caminho_s),
        data={"mode": "sintetico", **campos},
    )
    assert disparo.status_code == disparo_esperado, (disparo.status_code, disparo.text[:400])
    resposta = (
        espera(cliente_http, disparo.json()["identificador"])
        if disparo.status_code == 202
        else disparo
    )

    assert resposta.status_code == 422, (resposta.status_code, resposta.text[:400])
    assert cliente.causa_de(resposta.status_code, resposta.json()) == "dado", resposta.json()
    assert nome in str(resposta.json()["detail"]), resposta.json()


@pytest.mark.parametrize("subcomando", ["corrigir", "exportar"])
def test_c08_a_cli_sai_com_4_e_diz_a_feature(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], subcomando: str
) -> None:
    caminho_m, caminho_s, nome = _par_com_negativo(tmp_path)
    saida = tmp_path / ("saida.csv" if subcomando == "corrigir" else "exportado.csv")

    codigo = main(
        [
            subcomando,
            "--measurements",
            str(caminho_m),
            "--sample-list",
            str(caminho_s),
            "--mode",
            "sintetico",
            "--method",
            "serrf",
            "--output",
            str(saida),
        ]
    )

    erro = capsys.readouterr().err
    assert codigo == EXIT_INPUT_REJECTED, erro[-600:]
    assert "não sustenta o SERRF" in erro and nome in erro, erro[-600:]

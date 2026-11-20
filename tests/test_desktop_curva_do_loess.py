"""A curva do LOESS na janela — o botão, a figura, e a curva que chega depois da troca.

O corpo vem do produtor, e não de um dicionário escrito à mão: cada teste recebe o
que `POST /matrizes/curva-do-loess` devolve sobre os arquivos da fábrica — 40 injeções, oito QCs —,
pela `TestClient`. Um corpo montado à mão carregaria a minha leitura do esquema, e o leitor da
janela seria testado contra ela. Os arquivos da fábrica servem para o multipart e para a forma do
corpo; nenhum teste daqui afirma o que os arquivos do laboratório contêm.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a guarda é exercitada em tests/test_qt_guard.py
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()

from fastapi.testclient import TestClient  # noqa: E402
from PySide6.QtTest import QSignalSpy  # noqa: E402
from pytestqt.qtbot import QtBot  # noqa: E402

from refino.api.app import build_app  # noqa: E402
from refino.desktop.janela.figuras import EstadoDaFigura  # noqa: E402
from refino.desktop.janela.graficos import (  # noqa: E402
    FIGURA_DA_CURVA_DO_LOESS,
    FRASE_SEM_CURVA_DO_LOESS,
    PainelDeGraficos,
)
from refino.desktop.janela.pipeline import (  # noqa: E402
    FRASE_DA_CURVA_SEM_FEATURE,
    SEMENTE_DA_JANELA,
    PainelDePipeline,
)
from refino.interface.cliente import Resposta  # noqa: E402
from refino.plots.catalogo import CONSTRUTORES  # noqa: E402
from refino.plots.curva_do_loess import (  # noqa: E402
    ROTULO_DA_CURVA,
    ROTULO_DA_CURVA_DE_OUTRO_CORRETOR,
)
from tests.arnes_de_figura import series  # noqa: E402
from tests.factories import base_spec, render_pair  # noqa: E402

MODO = "neg_hilic"


class _TransporteQueRecusa:
    """Um `Transporte` cujas chamadas levantam: os testes que o usam substituem `chama`."""

    def post(self, url: str, **kwargs: object) -> object:
        raise AssertionError(f"estes testes não falam com a API, e alguém chamou POST {url}")

    def get(self, url: str, **kwargs: object) -> object:
        raise AssertionError(f"estes testes não falam com a API, e alguém chamou GET {url}")


def corpo_da_fabrica() -> dict[str, Any]:
    """O corpo da rota sobre os arquivos da fábrica, para a primeira feature.

    Pública porque a varredura de eixos de `tests/test_desktop_figuras.py` alimenta o painel com o
    mesmo corpo — uma figura com rota tem de entrar nela desenhada, e não pelo conjunto dos que
    não têm fio.
    """
    medidas, lista = render_pair(base_spec(n_features=3, n_samples=40, n_qcs=8))
    resposta = TestClient(build_app()).post(
        "/matrizes/curva-do-loess",
        files={
            "measurements": ("measurements.csv", medidas.encode("utf-8"), "text/csv"),
            "sample_list": ("Sample_List.csv", lista.encode("utf-8"), "text/csv"),
        },
        data={"mode": MODO, "feature": "1.00_100.0000m/z"},
    )
    assert resposta.status_code == 200, resposta.text
    devolvido: dict[str, Any] = resposta.json()
    assert devolvido["motivo_sem_curva"] is None, "âncora: a fábrica tem de dar uma curva"
    return devolvido


@pytest.fixture(scope="module")
def corpo() -> dict[str, Any]:
    return corpo_da_fabrica()


def _escolhe_a_curva(graficos: PainelDeGraficos) -> None:
    alvo = [c.funcao for c in CONSTRUTORES].index(FIGURA_DA_CURVA_DO_LOESS)
    graficos.figuras.seletor.setCurrentIndex(alvo)


def _curva_na_tela(graficos: PainelDeGraficos) -> np.ndarray:
    eixos = graficos.figuras.canvas.figure.axes
    assert eixos, "a figura da curva não tem eixo nenhum"
    # Os dois rótulos são a mesma série: qual deles depende do corretor carimbado, e o teste do
    # rótulo é outro.
    rotulos = (ROTULO_DA_CURVA, ROTULO_DA_CURVA_DE_OUTRO_CORRETOR)
    achadas = [s for s in series(eixos[0]) if s.rotulo.startswith(rotulos)]
    assert achadas, "a figura não desenhou a curva"
    return np.concatenate([np.asarray(s.y, dtype=np.float64) for s in achadas])


def test_sem_curva_pedida_a_figura_diz_onde_pedir_e_com_a_curva_desenha_a_do_corpo(
    qtbot: QtBot, corpo: dict[str, Any]
) -> None:
    """As duas direções do estado da figura, e a série na tela é a curva do corpo, bit a bit."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    construtor = next(c for c in CONSTRUTORES if c.funcao == FIGURA_DA_CURVA_DO_LOESS)

    assert graficos._validade_propria(construtor) is EstadoDaFigura.SEM_EXECUCAO
    _escolhe_a_curva(graficos)
    assert FRASE_SEM_CURVA_DO_LOESS in " ".join(graficos.figuras.estado_visivel().values()), (
        f"sem curva pedida, a figura não diz onde pedir: {graficos.figuras.estado_visivel()}"
    )
    assert not graficos.figuras.canvas.figure.axes, "sem curva pedida, desenhou alguma coisa"

    graficos.recebe_curva_do_loess(corpo)
    assert graficos._validade_propria(construtor) is EstadoDaFigura.DESENHADA
    esperada = np.array([np.nan if v is None else v for v in corpo["curva"]], dtype=np.float64)
    assert _curva_na_tela(graficos).tobytes() == esperada.tobytes()


def test_uma_resposta_de_erro_apaga_a_curva_anterior(qtbot: QtBot, corpo: dict[str, Any]) -> None:
    """Par causal da guarda de `recebe_curva_do_loess`: o corpo da rota guarda, o do 422 apaga."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_curva_do_loess(corpo)
    assert graficos._curva_do_loess is not None, "âncora: nem o corpo da rota ficou guardado"
    graficos.recebe_curva_do_loess({"detail": "`feature=x` não existe nesta matriz"})
    assert graficos._curva_do_loess is None, "o corpo de um 422 ficou guardado como curva"


def test_a_curva_que_chega_depois_da_troca_de_matriz_e_descartada(
    qtbot: QtBot, corpo: dict[str, Any]
) -> None:
    """Sem troca a curva fica; com troca entre o disparo e a chegada, ela é descartada.

    E a troca de matriz, sozinha, apaga a curva guardada — a curva de A não fica sob os arquivos de
    B à espera de outro pedido.

    > *«apaga a curva guardada»* deixou de valer: a troca ESCONDE a curva de A, que
    > volta com A. `_curva_do_loess` é a que se vê, e é ela que as asserções abaixo leem.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    graficos.define_matriz("matriz-A")
    graficos.carimba_a_curva("loess")
    graficos.recebe_curva_do_loess(corpo)
    assert graficos._curva_do_loess is not None, "âncora: a curva da matriz corrente foi descartada"

    graficos.define_matriz("matriz-B")
    assert graficos._curva_do_loess is None, "a troca de matriz não apagou a curva de A"

    graficos.define_matriz("matriz-A")
    graficos.carimba_a_curva("loess")
    graficos.define_matriz("matriz-B")
    graficos.recebe_curva_do_loess(corpo)
    assert graficos._curva_do_loess is None, "a curva de A ficou guardada sob a matriz B"


def test_o_botao_sem_feature_nao_emite_e_diz_por_que(qtbot: QtBot) -> None:
    """Nome vazio: nenhuma emissão, e a frase ao lado do botão. Nome escrito: uma, com ele."""
    pipeline = PainelDePipeline()
    qtbot.addWidget(pipeline)
    espiao = QSignalSpy(pipeline.ver_curva_do_loess)

    pipeline.feature_da_curva.setCurrentText("   ")
    pipeline.ver_curva_botao.click()
    assert espiao.count() == 0, "o botão emitiu sem feature"
    assert pipeline.rotulo_da_curva.text() == FRASE_DA_CURVA_SEM_FEATURE

    pipeline.povoa_features_da_curva(["F1", "F2"])
    pipeline.feature_da_curva.setCurrentText("F2")
    pipeline.ver_curva_botao.click()
    assert espiao.count() == 1 and espiao.at(0) == ["F2"], [
        espiao.at(i) for i in range(espiao.count())
    ]


def test_o_disparo_leva_os_lotes_e_a_semente_da_correcao_e_carimba_a_matriz(qtbot: QtBot) -> None:
    """Os campos do pedido são os da correção desta janela, e o carimbo é a matriz corrente."""
    from refino.desktop.janela.arquivos import ParDeArquivos
    from refino.desktop.janela.principal import (
        CHAVE_DA_CURVA_DO_LOESS,
        ROTA_DA_CURVA_DO_LOESS,
        JanelaPrincipal,
    )

    janela = JanelaPrincipal(_TransporteQueRecusa())
    qtbot.addWidget(janela)
    janela.arquivos._par = ParDeArquivos(
        "A.csv", b"feature,QC1\n1,2\n", "sA.csv", b"label,time\nQC1,1\n"
    )
    janela.arquivos.modo.setEditText("pos_hilic")
    enviadas: list[tuple[str, str, dict[str, Any]]] = []
    janela.chama = lambda metodo, rota, **kw: enviadas.append((rota, kw["chave"], kw["campos"]))  # type: ignore[method-assign]

    assert janela.graficos._identidade_da_curva is None, "âncora: já havia carimbo"
    janela._dispara_curva_do_loess("F7")

    assert [(r, c) for r, c, _ in enviadas] == [(ROTA_DA_CURVA_DO_LOESS, CHAVE_DA_CURVA_DO_LOESS)]
    campos = enviadas[0][2]
    assert campos["feature"] == "F7" and campos["seed"] == SEMENTE_DA_JANELA
    assert campos["mode"] == "pos_hilic", campos
    carimbo = janela.graficos._identidade_da_curva
    assert carimbo is not None and carimbo == janela.graficos._identidade_da_matriz

    # E os lotes são os da correção, com um valor que não é o padrão. Com o campo vazio
    # viaja um lote só, e uma janela que mandasse sempre 1 passaria pela metade de cima. Com dois
    # lotes e os cortes que o diagnóstico propôs, o pedido leva os dois, e são os que o pedido da
    # execução levaria — `campos_base`, o outro caminho.
    padrao = janela.pipeline.campos_base("pos_hilic", "loess", campos["n_batches"])
    assert campos["n_batches"] == padrao["n_batches"] == 1, campos
    assert "batch_cuts" not in campos and "batch_cuts" not in padrao, campos
    assert janela.pipeline.pre_preenche("n_batches", 2), "âncora: a proposta de lotes não entrou"
    janela.pipeline.define_cortes_propostos([40])
    janela._dispara_curva_do_loess("F7")
    com_lotes = enviadas[-1][2]
    da_execucao = janela.pipeline.campos_base("pos_hilic", "loess", 2)
    assert com_lotes["n_batches"] == da_execucao["n_batches"] == 2, com_lotes
    assert json.loads(com_lotes["batch_cuts"]) == json.loads(da_execucao["batch_cuts"]) == [40], (
        com_lotes
    )


@pytest.mark.parametrize(
    ("corretor", "rotulo", "outro"),
    [
        ("serrf", ROTULO_DA_CURVA_DE_OUTRO_CORRETOR, ROTULO_DA_CURVA),
        ("loess", ROTULO_DA_CURVA, ROTULO_DA_CURVA_DE_OUTRO_CORRETOR),
    ],
)
def test_o_rotulo_da_curva_segue_o_corretor_escolhido_no_disparo(
    qtbot: QtBot, corpo: dict[str, Any], corretor: str, rotulo: str, outro: str
) -> None:
    """Com o SERRF escolhido, a figura não diz que a curva do LOESS corrigiu o sinal.

    **Par causal:** o mesmo disparo e o mesmo corpo, e só o corretor do combo muda. O corretor é
    carimbado no disparo, como a matriz, e é o do combo: a janela não delega, e o
    corretor que a rota vai correr é sempre o do seletor (até ali, `corretor_que_vai_correr` lia a
    caixa «Automático», que este teste desmarcava).
    """
    from refino.desktop.janela.arquivos import ParDeArquivos
    from refino.desktop.janela.principal import CHAVE_DA_CURVA_DO_LOESS, JanelaPrincipal

    janela = JanelaPrincipal(_TransporteQueRecusa())
    qtbot.addWidget(janela)
    janela.arquivos._par = ParDeArquivos(
        "A.csv", b"feature,QC1\n1,2\n", "sA.csv", b"label,time\nQC1,1\n"
    )
    janela.arquivos.modo.setEditText(MODO)
    janela.chama = lambda metodo, rota, **kw: None  # type: ignore[method-assign]
    janela.pipeline.corretor.setCurrentText(corretor)
    assert janela.pipeline.corretor.currentText() == corretor, "âncora: o combo não mudou"

    janela._dispara_curva_do_loess(corpo["feature"])
    janela._encaminha(
        CHAVE_DA_CURVA_DO_LOESS,
        Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=corpo, conteudo=None),
    )
    figura = janela.graficos.figuras.canvas.figure
    assert figura.axes, "a figura da curva não foi desenhada"
    rotulos = {serie.rotulo for serie in series(figura.axes[0])}
    assert rotulo in rotulos and outro not in rotulos, rotulos
    rodape = " ".join(" ".join(t.get_text() for t in figura.texts).split())
    assert (f"«{corretor}»" in rodape) == (corretor != "loess"), rodape


def test_a_janela_leva_a_curva_a_figura_e_o_erro_ao_rotulo(
    qtbot: QtBot, corpo: dict[str, Any]
) -> None:
    """A resposta com curva abre a figura na aba dos gráficos; a recusa vai ao rótulo do botão."""
    from refino.desktop.janela.principal import (
        CHAVE_DA_CURVA_DO_LOESS,
        INDICE_DA_ABA_DOS_GRAFICOS,
        JanelaPrincipal,
    )

    janela = JanelaPrincipal(_TransporteQueRecusa())
    qtbot.addWidget(janela)
    janela.abas.setCurrentIndex(0)
    assert INDICE_DA_ABA_DOS_GRAFICOS != 0, "âncora: a aba dos gráficos não pode ser a inicial"

    janela._encaminha(
        CHAVE_DA_CURVA_DO_LOESS,
        Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=corpo, conteudo=None),
    )
    assert janela.graficos._curva_do_loess is not None
    assert janela.abas.currentIndex() == INDICE_DA_ABA_DOS_GRAFICOS
    assert janela.graficos.figuras.seletor.currentText() == next(
        rotulo
        for rotulo, construtor in janela.graficos.figuras._entradas
        if construtor.funcao == FIGURA_DA_CURVA_DO_LOESS
    )

    detalhe = "`feature=x` não existe nesta matriz"
    janela._encaminha(
        CHAVE_DA_CURVA_DO_LOESS,
        Resposta(
            codigo=422, ok=False, causa=None, mensagem="", corpo={"detail": detalhe}, conteudo=None
        ),
    )
    assert janela.graficos._curva_do_loess is None, "a recusa deixou a curva anterior guardada"
    assert detalhe in janela.pipeline.rotulo_da_curva.text()


def test_a_resposta_de_outra_matriz_nao_leva_a_janela_a_figura(
    qtbot: QtBot, corpo: dict[str, Any]
) -> None:
    """A curva descartada não abre a aba dos gráficos, e o rótulo do botão diz que foi descartada.

    **Par causal sobre o mesmo corpo:** com o carimbo da matriz corrente a janela vai à figura; com
    o de outra, fica onde está. `recebe_curva_do_loess` devolve se guardou, e é isso que separa os
    dois — `None` nos dois casos os tornava indistinguíveis para quem o chama.
    """
    from refino.desktop.janela.principal import (
        CHAVE_DA_CURVA_DO_LOESS,
        FRASE_DA_CURVA_DESCARTADA,
        INDICE_DA_ABA_DOS_GRAFICOS,
        JanelaPrincipal,
    )

    resposta = Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=corpo, conteudo=None)
    janela = JanelaPrincipal(_TransporteQueRecusa())
    qtbot.addWidget(janela)
    janela.abas.setCurrentIndex(0)
    janela.graficos.define_matriz("matriz-A")
    janela.graficos.carimba_a_curva("loess")
    janela.graficos.define_matriz("matriz-B")
    janela._encaminha(CHAVE_DA_CURVA_DO_LOESS, resposta)
    assert janela.graficos._curva_do_loess is None, "a curva de A ficou guardada sob B"
    assert janela.abas.currentIndex() == 0, "a janela foi à figura de uma curva descartada"
    assert janela.pipeline.rotulo_da_curva.text() == FRASE_DA_CURVA_DESCARTADA

    janela.graficos.carimba_a_curva("loess")
    janela._encaminha(CHAVE_DA_CURVA_DO_LOESS, resposta)
    assert janela.graficos._curva_do_loess is not None, "âncora: a curva de B não foi guardada"
    assert janela.abas.currentIndex() == INDICE_DA_ABA_DOS_GRAFICOS

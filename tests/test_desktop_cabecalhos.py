"""O título e a descrição formam um bloco compacto no centro vertical do cabeçalho."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

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

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout  # noqa: E402

from refino.desktop.janela.componentes import Secao  # noqa: E402
from refino.desktop.janela.tema import veste_a_aplicacao  # noqa: E402


@pytest.fixture
def tema_do_cabecalho(qtbot: Any) -> Iterator[None]:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    fonte, paleta, folha = app.font(), app.palette(), app.styleSheet()
    estilo = app.style().name()
    try:
        veste_a_aplicacao(app)
        yield
    finally:
        app.setStyleSheet(folha)
        app.setPalette(paleta)
        app.setFont(fonte)
        app.setStyle(estilo)


@pytest.mark.parametrize("apresentacao", ["padrao", "referencia", "plano"])
@pytest.mark.parametrize("multilinha", [False, True])
def test_textos_marcador_e_acao_ficam_no_mesmo_centro_vertical(
    apresentacao: str, multilinha: bool, qtbot: Any, tema_do_cabecalho: None
) -> None:
    subtitulo = (
        "Consulte os resultados de todas as combinações e compare as conclusões antes de "
        "escolher os parâmetros que serão usados no estudo."
        if multilinha
        else "Consulte a conclusão do estudo."
    )
    secao = Secao(
        "A conclusão",
        numero=2,
        subtitulo=subtitulo,
        referencia=apresentacao == "referencia",
        painel_plano=apresentacao == "plano",
    )
    qtbot.addWidget(secao)
    acao = QPushButton("Ver resultado", secao)
    acao.setFixedSize(100, 96)
    secao.acoes.addWidget(acao)
    secao.corpo.addWidget(QLabel("Conteúdo da seção", secao))
    secao.resize(520, 280)
    secao.show()
    QApplication.processEvents()

    externo = secao.layout()
    assert isinstance(externo, QVBoxLayout)
    cabecalho = externo.itemAt(0).layout()
    assert cabecalho is not None
    centro_da_linha = cabecalho.contentsRect().center().y()
    centro_do_bloco = (secao.titulo.geometry().top() + secao.subtitulo.geometry().bottom()) / 2
    assert abs(centro_do_bloco - centro_da_linha) <= 1
    assert abs(secao.numero.geometry().center().y() - centro_da_linha) <= 1
    assert abs(acao.geometry().center().y() - centro_da_linha) <= 1

    # A ação alta não estica os rótulos nem espalha título e descrição pela linha.
    for texto in (secao.titulo, secao.subtitulo):
        altura_do_texto = max(texto.minimumHeight(), texto.heightForWidth(texto.width()))
        assert texto.height() <= altura_do_texto + 1
    if multilinha:
        assert secao.subtitulo.height() > secao.subtitulo.fontMetrics().height()
    else:
        assert secao.subtitulo.height() <= secao.subtitulo.fontMetrics().height() + 1
    if apresentacao == "padrao":
        margens = cabecalho.contentsMargins()
        assert margens.top() == margens.bottom()


@pytest.mark.parametrize(
    ("pagina", "nome_anterior", "nome_atual"),
    [
        ("multiverso", "secao_da_grade", "secao_da_conclusao"),
        ("vias", "secao_dos_picos", "secao_das_vias"),
    ],
)
def test_cabecalho_da_pagina_real_fica_centrado_entre_as_divisorias(
    pagina: str,
    nome_anterior: str,
    nome_atual: str,
    qtbot: Any,
    tema_do_cabecalho: None,
) -> None:
    """O espaço externo entre seções também pertence à altura percebida do cabeçalho."""
    from refino.desktop.janela.principal import JanelaPrincipal
    from refino.desktop.janela.tema import veste_a_janela

    janela = JanelaPrincipal(None)
    qtbot.addWidget(janela)
    veste_a_janela(janela)
    janela.resize(1440, 1000)
    inicio_do_titulo = "Teste de robustez" if pagina == "multiverso" else "Vias metabólicas"
    indice = next(
        i for i in range(janela.abas.count()) if janela.abas.tabText(i).startswith(inicio_do_titulo)
    )
    janela.abas.setCurrentIndex(indice)
    janela.show()
    QApplication.processEvents()
    painel = getattr(janela, pagina)
    anterior = getattr(painel, nome_anterior)
    atual = getattr(painel, nome_atual)
    pai = atual.parentWidget()
    assert pai is anterior.parentWidget()
    divisor = atual.layout().itemAt(1).widget()
    assert divisor is not None and divisor.objectName() == "divisor"

    linha_superior = anterior.geometry().bottom()
    linha_inferior = divisor.mapTo(pai, QPoint(0, 0)).y()
    centro_entre_linhas = (linha_superior + linha_inferior) / 2
    centro_do_bloco = (
        atual.y() + (atual.titulo.geometry().top() + atual.subtitulo.geometry().bottom()) / 2
    )
    assert linha_superior < centro_do_bloco < linha_inferior
    assert abs(centro_do_bloco - centro_entre_linhas) <= 1

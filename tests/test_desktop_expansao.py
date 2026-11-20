"""As expansões têm quadros intermediários sem espremer textos ou prender alturas."""

from __future__ import annotations

from typing import Any

import pytest

from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )
garante_plataforma_offscreen()

from refino.desktop import arranque  # noqa: E402, F401, I001
from refino.desktop.janela.componentes import Recolhivel  # noqa: E402
from refino.desktop.janela.expansao import ExpansaoSuave  # noqa: E402

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QStackedWidget, QVBoxLayout, QWidget  # noqa: E402


TEXTO = (
    "A descrição continua completa e pode quebrar em várias linhas. "
    "Os controles preservam seu pai e o conteúdo fica imóvel enquanto a moldura revela o texto. "
) * 5


def _bloco(qtbot: Any) -> tuple[QWidget, Recolhivel, QLabel]:
    janela = QWidget()
    qtbot.addWidget(janela)
    coluna = QVBoxLayout(janela)
    bloco = Recolhivel("Como ler", janela)
    coluna.addWidget(bloco)
    coluna.addStretch(1)
    dentro = QVBoxLayout(bloco.corpo)
    texto = QLabel(TEXTO, bloco.corpo)
    texto.setWordWrap(True)
    texto.setAlignment(Qt.AlignmentFlag.AlignTop)
    dentro.addWidget(texto)
    janela.resize(760, 650)
    janela.show()
    QApplication.processEvents()
    return janela, bloco, texto


def test_revelacao_tem_quadros_intermediarios_e_nao_comprime_o_texto(qtbot: Any) -> None:
    _janela, bloco, texto = _bloco(qtbot)
    pai = texto.parentWidget()
    limites = bloco.corpo.minimumHeight(), bloco.corpo.maximumHeight()
    bloco.alterna.click()
    qtbot.waitUntil(lambda: bloco.corpo.height() > 0 and bloco.expansao.em_andamento)
    altura_intermediaria = bloco.corpo.height()
    assert texto.height() >= texto.heightForWidth(texto.width())
    qtbot.waitUntil(lambda: not bloco.expansao.em_andamento)
    assert bloco.corpo.height() > altura_intermediaria
    assert texto.parentWidget() is pai
    assert (bloco.corpo.minimumHeight(), bloco.corpo.maximumHeight()) == limites
    assert bloco.corpo.layout().isEnabled()

    altura_aberta = bloco.corpo.height()
    bloco.alterna.click()
    qtbot.waitUntil(lambda: 0 < bloco.corpo.height() < altura_aberta)
    assert bloco.corpo.isVisible()
    qtbot.waitUntil(lambda: not bloco.expansao.em_andamento)
    assert bloco.corpo.isHidden()
    assert (bloco.corpo.minimumHeight(), bloco.corpo.maximumHeight()) == limites


def test_cliques_rapidos_revertem_a_transicao_sem_altura_presa(qtbot: Any) -> None:
    _janela, bloco, texto = _bloco(qtbot)
    bloco.alterna.click()
    qtbot.waitUntil(lambda: bloco.corpo.height() > 5)
    bloco.alterna.click()
    assert bloco.expansao.em_andamento
    assert bloco.corpo.isVisible()
    bloco.alterna.click()
    qtbot.waitUntil(lambda: not bloco.expansao.em_andamento)
    assert bloco.corpo.isVisible()
    assert bloco.corpo.maximumHeight() == 16777215
    assert texto.height() >= texto.heightForWidth(texto.width())


def test_redimensionar_durante_a_abertura_preserva_as_linhas(qtbot: Any) -> None:
    janela, bloco, texto = _bloco(qtbot)
    bloco.alterna.click()
    qtbot.waitUntil(lambda: bloco.corpo.height() > 5)
    janela.resize(400, 650)
    qtbot.waitUntil(lambda: not bloco.expansao.em_andamento)
    assert bloco.corpo.isVisible()
    assert texto.height() >= texto.heightForWidth(texto.width())
    assert bloco.corpo.maximumHeight() == 16777215


def test_texto_atualizado_durante_a_transicao_continua_completo(qtbot: Any) -> None:
    _janela, bloco, texto = _bloco(qtbot)
    bloco.alterna.click()
    qtbot.waitUntil(lambda: bloco.corpo.height() > 5)
    texto.setText(TEXTO * 2)
    qtbot.waitUntil(lambda: not bloco.expansao.em_andamento)
    assert texto.text() == TEXTO * 2
    assert texto.height() >= texto.heightForWidth(texto.width())
    assert bloco.corpo.maximumHeight() == 16777215


def test_sair_da_pagina_durante_transicao_conclui_sem_prender_limites(qtbot: Any) -> None:
    janela, bloco, texto = _bloco(qtbot)
    janela.hide()
    pilha = QStackedWidget()
    qtbot.addWidget(pilha)
    pilha.addWidget(janela)
    pilha.addWidget(QWidget())
    pilha.resize(760, 650)
    pilha.show()
    QApplication.processEvents()
    bloco.alterna.click()
    qtbot.waitUntil(lambda: bloco.corpo.height() > 5)
    pilha.setCurrentIndex(1)
    assert not bloco.expansao.em_andamento
    assert bloco.corpo.maximumHeight() == 16777215
    pilha.setCurrentIndex(0)
    QApplication.processEvents()
    assert bloco.corpo.isVisible()
    assert texto.height() >= texto.heightForWidth(texto.width())


def test_limpar_ou_desabilitar_movimento_aplica_estado_imediato(qtbot: Any) -> None:
    _janela, bloco, _texto = _bloco(qtbot)
    bloco.alterna.click()
    qtbot.waitUntil(lambda: bloco.corpo.height() > 5)
    bloco.expansao.conclui(False)
    assert not bloco.expansao.em_andamento
    assert bloco.corpo.isHidden()
    assert bloco.corpo.maximumHeight() == 16777215
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    anterior = app.property("refino_animacoes")
    try:
        app.setProperty("refino_animacoes", False)
        bloco.expansao.alterna(True)
        assert bloco.corpo.isVisible()
        assert not bloco.expansao.em_andamento
        bloco.expansao.alterna(False)
        assert bloco.corpo.isHidden()
    finally:
        app.setProperty("refino_animacoes", anterior)


def test_rotulo_existente_revela_texto_e_restaura_alinhamento(qtbot: Any) -> None:
    janela = QWidget()
    qtbot.addWidget(janela)
    coluna = QVBoxLayout(janela)
    texto = QLabel(TEXTO, janela)
    texto.setWordWrap(True)
    coluna.addWidget(texto)
    coluna.addStretch(1)
    texto.hide()
    movimento = ExpansaoSuave(texto)
    alinhamento = texto.alignment()
    janela.resize(400, 650)
    janela.show()
    QApplication.processEvents()
    movimento.alterna(True)
    qtbot.waitUntil(lambda: texto.height() > 5)
    assert movimento.em_andamento
    assert texto.alignment() & Qt.AlignmentFlag.AlignTop
    qtbot.waitUntil(lambda: not movimento.em_andamento)
    assert texto.height() >= texto.heightForWidth(texto.width())
    assert texto.alignment() == alinhamento

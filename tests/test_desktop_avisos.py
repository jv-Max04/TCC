"""O aviso de desfecho é legível, fechável e não apaga os dados de uma resposta."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
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

from refino.desktop.janela.arquivos import ROTULO_DA_LISTA, ROTULO_DAS_MEDIDAS  # noqa: E402,I001
from refino.desktop.janela.principal import JanelaPrincipal, texto_do_desfecho  # noqa: E402
from refino.desktop.janela.tema import veste_a_aplicacao, veste_a_janela  # noqa: E402

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QToolButton, QWidget  # noqa: E402


class _SemRede:
    def post(self, url: str, **kwargs: Any) -> Any:
        raise AssertionError("um aviso local não deve chamar a API")

    def get(self, url: str, **kwargs: Any) -> Any:
        raise AssertionError("um aviso local não deve chamar a API")


@pytest.fixture
def janela(qtbot: Any, tmp_path: Path) -> Iterator[JanelaPrincipal]:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    fonte, paleta, folha = app.font(), app.palette(), app.styleSheet()
    estilo = app.style().name()
    # O nome traz o modo, que a janela deduz dele: sem modo, a chamada é recusada.
    arquivos = {
        ROTULO_DAS_MEDIDAS: tmp_path / "medidas_pos_hilic.csv",
        ROTULO_DA_LISTA: tmp_path / "amostras.csv",
    }
    for caminho in arquivos.values():
        caminho.write_bytes(b"anexo sintetico para testar o seletor")
    veste_a_aplicacao(app)
    montada = JanelaPrincipal(_SemRede(), escolhedor=lambda titulo: arquivos[titulo])
    qtbot.addWidget(montada)
    veste_a_janela(montada)
    montada.resize(1280, 800)
    montada.show()
    QApplication.processEvents()
    try:
        yield montada
    finally:
        montada.close()
        app.setStyleSheet(folha)
        app.setPalette(paleta)
        app.setFont(fonte)
        app.setStyle(estilo)


@pytest.mark.parametrize("largura", [1024, 1280])
def test_clique_sem_arquivos_mostra_aviso_com_titulo_icone_e_fechamento(
    largura: int,
    janela: JanelaPrincipal,
) -> None:
    janela.resize(largura, 800)
    janela.leitura.botoes["validacao"].click()
    QApplication.processEvents()
    aviso = janela.aviso_de_desfecho
    assert aviso.isVisibleTo(janela)
    assert aviso.property("tipo") == "aviso"
    assert aviso.titulo.text() == "Escolha os arquivos do estudo"
    assert "nenhum par de arquivos escolhido" in janela.rotulo_de_desfecho.text()
    assert aviso.rotulo is janela.rotulo_de_desfecho
    assert aviso.rotulo.textFormat() == Qt.TextFormat.MarkdownText
    assert aviso.rotulo.wordWrap()
    assert aviso.rotulo.height() >= aviso.rotulo.heightForWidth(aviso.rotulo.width())
    assert aviso.height() <= 128, "uma mensagem curta deve permanecer compacta"
    assert not aviso.icone.pixmap().isNull()
    assert aviso.fechar.accessibleName() == "Fechar aviso"


def test_fechar_preserva_ultimo_desfecho_e_nova_mensagem_reaparece(
    janela: JanelaPrincipal,
) -> None:
    janela.leitura.botoes["validacao"].click()
    aviso = janela.aviso_de_desfecho
    texto = janela.rotulo_de_desfecho.text()
    aviso.fechar.click()
    assert aviso.isHidden()
    assert janela.rotulo_de_desfecho.text() == texto

    janela.leitura.botoes["validacao"].click()
    assert aviso.isVisibleTo(janela)
    assert janela.rotulo_de_desfecho.text() == texto


def test_completar_o_par_resolve_o_aviso_de_arquivos_sem_apagar_o_texto(
    janela: JanelaPrincipal,
) -> None:
    janela.leitura.botoes["validacao"].click()
    aviso = janela.aviso_de_desfecho
    texto = janela.rotulo_de_desfecho.text()
    janela.arquivos.escolhe_medidas()
    assert aviso.isVisibleTo(janela), "a lista ainda falta"
    janela.arquivos.escolhe_lista()
    assert aviso.isHidden()
    assert janela.rotulo_de_desfecho.text() == texto
    assert janela.arquivos.par is not None


def test_anexar_arquivos_nao_esconde_falha_que_nao_pertence_ao_seletor(
    janela: JanelaPrincipal,
) -> None:
    corpo = {"detail": "falha de armazenamento"}
    texto = janela.mostra_desfecho(503, corpo)
    janela.arquivos.escolhe_medidas()
    janela.arquivos.escolhe_lista()
    aviso = janela.aviso_de_desfecho
    assert aviso.isVisibleTo(janela)
    assert aviso.property("tipo") == "erro"
    assert texto == texto_do_desfecho(503, corpo) == janela.rotulo_de_desfecho.text()


def test_nova_tentativa_valida_recolhe_o_aviso_anterior(janela: JanelaPrincipal) -> None:
    janela.arquivos.escolhe_medidas()
    janela.arquivos.escolhe_lista()
    texto = janela.mostra_desfecho(503, {"detail": "falha anterior"})
    assert janela.aviso_de_desfecho.isVisibleTo(janela)

    assert janela._pedido_de_arquivos() is not None

    assert janela.aviso_de_desfecho.isHidden()
    assert janela.rotulo_de_desfecho.text() == texto


def test_escrita_direta_no_rotulo_tambem_reabre_aviso(janela: JanelaPrincipal) -> None:
    aviso = janela.aviso_de_desfecho
    janela.mostra_desfecho(503, {"detail": "falha anterior"})
    aviso.fechar.click()
    janela.rotulo_de_desfecho.setText("Uma **nova** mensagem com `detalhes`.")
    assert aviso.isVisibleTo(janela)
    assert aviso.titulo.text() == "Mensagem do estudo"
    assert janela.rotulo_de_desfecho.text() == "Uma **nova** mensagem com `detalhes`."


@pytest.mark.parametrize("largura", [1024, 1440])
def test_notify_sobrepoe_a_pagina_sem_mover_o_formulario(
    largura: int, janela: JanelaPrincipal
) -> None:
    janela.resize(largura, 800)
    QApplication.processEvents()
    geometria_da_pagina = janela.abas.geometry()
    janela.leitura.botoes["validacao"].click()
    QApplication.processEvents()
    aviso = janela.aviso_de_desfecho
    area = janela.centralWidget()
    assert area is not None and area.layout() is not None
    assert aviso.parentWidget() is area
    assert area.layout().indexOf(aviso) == -1
    assert janela.abas.geometry() == geometria_da_pagina
    assert aviso.width() <= 400
    assert area.width() - aviso.geometry().right() - 1 == 16
    # Sem faixa no topo da área central, o aviso fica na margem do topo.
    assert aviso.y() == 16


def test_notify_segue_o_canto_superior_direito_ao_redimensionar(qtbot: Any) -> None:
    from refino.desktop.janela.avisos import AvisoDeDesfecho

    area = QWidget()
    qtbot.addWidget(area)
    area.resize(800, 600)
    aviso = AvisoDeDesfecho("—", area)
    area.show()
    aviso.mostra("Mensagem de teste.")
    QApplication.processEvents()
    assert aviso.y() == 16
    assert aviso.geometry().right() == 783
    area.resize(500, 400)
    QApplication.processEvents()
    assert aviso.y() == 16
    assert aviso.geometry().right() == 483


def test_notify_longo_tem_altura_limitada_e_mantem_todo_o_texto(
    janela: JanelaPrincipal, qtbot: Any
) -> None:
    texto = "Falha detalhada:\n\n" + "\n\n".join(
        f"Linha {i}: informação completa para verificar o erro do estudo." for i in range(60)
    )
    geometria_da_pagina = janela.abas.geometry()
    janela.aviso_de_desfecho.mostra(texto, tipo="erro", titulo="Não foi possível concluir")
    aviso = janela.aviso_de_desfecho
    qtbot.waitUntil(lambda: aviso.corpo.verticalScrollBar().maximum() > 0)
    assert aviso.height() <= 200
    assert aviso.corpo.height() <= 128
    assert janela.rotulo_de_desfecho.text() == texto
    assert janela.abas.geometry() == geometria_da_pagina
    assert aviso.rotulo.height() >= aviso.rotulo.heightForWidth(aviso.rotulo.width())
    barra = aviso.corpo.verticalScrollBar()
    barra.setValue(barra.maximum())
    assert barra.value() == barra.maximum()


@pytest.mark.parametrize("largura", [1024, 1440])
def test_notify_e_a_pagina_nao_se_movem_quando_a_lista_de_trabalhos_cresce(
    largura: int, janela: JanelaPrincipal, qtbot: Any
) -> None:
    """A faixa saiu da área central, e crescer a lista não move o aviso nem a página.

    Era o contrário até aqui: a faixa ficava no topo da área central, crescia com a lista, e o
    aviso a acompanhava. A âncora de que a lista CRESCEU vai antes — sem ela, *"não se moveu"*
    seria indistinguível de *"não havia o que mover"*.
    """
    from refino.desktop.janela.artefatos import ARTEFATOS

    janela.resize(largura, 800)
    janela.findChild(QToolButton, "alterna_dos_trabalhos").click()
    janela.leitura.botoes["validacao"].click()
    QApplication.processEvents()
    area = janela.centralWidget()
    assert area is not None
    aviso = janela.aviso_de_desfecho
    pagina = janela.abas.geometry()
    altura_vazia = janela.faixa_de_trabalhos.sizeHint().height()
    trabalhos = {"corrigir": "execucao-1"}
    trabalhos.update({a.chave: f"exportacao-{i}" for i, a in enumerate(ARTEFATOS)})
    janela.trabalhos.mostra(trabalhos)
    qtbot.waitUntil(lambda: janela.faixa_de_trabalhos.sizeHint().height() > altura_vazia)
    QApplication.processEvents()
    assert aviso.isVisibleTo(janela)
    assert aviso.y() == 16
    assert janela.abas.geometry() == pagina
    assert aviso.geometry().bottom() < area.height()


def test_aviso_simples_expira_sem_apagar_texto_e_falha_seria_permanece(
    janela: JanelaPrincipal, qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from refino.desktop.janela import avisos

    monkeypatch.setattr(avisos, "TEMPO_DO_AVISO_MS", 30)
    janela.leitura.botoes["validacao"].click()
    texto = janela.rotulo_de_desfecho.text()
    qtbot.waitUntil(janela.aviso_de_desfecho.isHidden, timeout=1000)
    assert janela.rotulo_de_desfecho.text() == texto

    janela.mostra_desfecho(503, {"detail": "falha de armazenamento"})
    qtbot.wait(60)
    assert janela.aviso_de_desfecho.isVisibleTo(janela)


def test_resposta_2xx_registra_o_desfecho_sem_abrir_o_aviso(janela: JanelaPrincipal) -> None:
    """Só a falha abre o aviso: o `2xx` saía como `[causa: None] OK` por cima das abas."""
    aviso = janela.aviso_de_desfecho
    janela.mostra_desfecho(409, {"detail": "conflito anterior"})
    assert aviso.isVisibleTo(janela), "âncora: a falha não abriu o aviso"

    texto = janela.mostra_desfecho(200, {"run_id": 1})
    assert aviso.isHidden(), "o 2xx abriu o aviso, ou deixou a falha anterior aberta"
    assert janela.rotulo_de_desfecho.text() == texto, "o 2xx não ficou registrado"

    janela.mostra_desfecho(503, {"detail": "falha nova"})
    assert aviso.isVisibleTo(janela), "depois de um 2xx, a falha seguinte deixou de abrir"


def test_aviso_que_nao_e_erro_expira_e_o_erro_fica(
    janela: JanelaPrincipal, qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Todo aviso que não é erro tem prazo; só o de arquivos tinha."""
    from refino.desktop.janela import avisos

    monkeypatch.setattr(avisos, "TEMPO_DO_AVISO_MS", 30)
    aviso = janela.aviso_de_desfecho
    aviso.mostra("Lote concluído", titulo="Lote", tipo="info")
    assert aviso.isVisibleTo(janela), "âncora: o aviso informativo não abriu"
    qtbot.waitUntil(aviso.isHidden, timeout=1000)
    assert janela.rotulo_de_desfecho.text() == "Lote concluído", "expirar apagou o desfecho"

    aviso.mostra("A gravação falhou", titulo="Erro", tipo="erro")
    qtbot.wait(90)
    assert aviso.isVisibleTo(janela), "o erro expirou como um aviso informativo"

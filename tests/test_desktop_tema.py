"""O tema da janela.

Cada propriedade aqui tem um par: o teste que passa na árvore limpa e o que planta a
violação e afirma a reprovação. O tema é cosmético, mas três dos seus modos de falha não são, e
nenhum levanta erro sozinho:

- **folha que não analisa**: o Qt descarta a folha INTEIRA com um aviso no `stderr` e desenha o
  estilo cru — a janela abre, nada acusa;
- **recurso ausente**: `url()` para arquivo que não existe deixa a seta ou o visto invisível — uma
  caixa de seleção desmarcada some da tela;
- **papel digitado errado**: o seletor `[papel="..."]` casa por igualdade de texto, e um papel
  inexistente cai no estilo comum em silêncio.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from tests.pedido_da_tela import campos_da_tela
from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a decisão é exercitada em `tests/test_qt_guard.py`, com planta
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()

# **O `refino` ANTES do PySide6, e a ordem não é estilo.** O gancho de import do PySide6 inspeciona
# módulos, e o `_SixMetaPathImporter` que o `dateutil` instala (via pandas) não suporta a inspeção:
# importar pandas DEPOIS do Qt levanta `AttributeError` na coleta. Medido em 19/10/2026; os outros
# módulos `test_desktop_*` escapam pela mesma ordem, e o arranque real também.
from refino.desktop import arranque  # noqa: E402, I001
from refino.desktop.janela import tema  # noqa: E402
from refino.desktop.janela.expansao import ExpansaoSuave  # noqa: E402
from refino.desktop.janela.principal import JanelaPrincipal  # noqa: E402

from PySide6.QtCore import QEvent, QObject, QtMsgType, qInstallMessageHandler  # noqa: E402
from PySide6.QtGui import QIcon, QImage, QPalette, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QApplication,
    QLabel,
    QPushButton,
    QStyle,
    QWidget,
)


class _Nada:
    def post(self, url: str, **kwargs: Any) -> Any:
        raise ConnectionRefusedError("teste")

    def get(self, url: str, **kwargs: Any) -> Any:
        raise ConnectionRefusedError("teste")


def _espera_transicoes(raiz: QWidget, qtbot: Any) -> None:
    """As asserções de geometria final aguardam os 180 ms do movimento visual."""
    QApplication.processEvents()
    qtbot.waitUntil(
        lambda: not any(e.em_andamento for e in raiz.findChildren(ExpansaoSuave)),
        timeout=2000,
    )
    qtbot.wait(30)
    QApplication.processEvents()


@contextmanager
def _avisos_do_qt() -> Iterator[list[str]]:
    """Captura as mensagens que o Qt manda ao manipulador — é por aí que folha inválida avisa."""
    recebidas: list[str] = []

    def _guarda(tipo: QtMsgType, contexto: Any, mensagem: str) -> None:
        recebidas.append(mensagem)

    anterior = qInstallMessageHandler(_guarda)
    try:
        yield recebidas
    finally:
        qInstallMessageHandler(anterior)


def _polir_com(folha: str, qtbot: Any) -> list[str]:
    """Aplica a folha a um widget com filhos e força o polimento, que é quando o Qt a analisa."""
    raiz = QWidget()
    qtbot.addWidget(raiz)
    QPushButton("x", raiz)
    with _avisos_do_qt() as avisos:
        raiz.setStyleSheet(folha)
        raiz.ensurePolished()
        for filho in raiz.findChildren(QWidget):
            filho.ensurePolished()
    return avisos


# ---------------------------------------------------------------------------
# A folha analisa — e o portão consegue acusar a que não analisa
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("esquema", ["claro", "escuro"])
def test_a_folha_de_cada_esquema_analisa_sem_aviso(esquema: str, qtbot: Any) -> None:
    folha = tema.folha_de_estilo(esquema)  # type: ignore[arg-type]
    assert "QPushButton" in folha, "folha vazia passaria por vacuidade"
    avisos = _polir_com(folha, qtbot)
    assert not [a for a in avisos if "parse" in a.lower()], avisos


def test_o_portao_da_analise_acusa_folha_quebrada(qtbot: Any) -> None:
    """A planta: uma chave aberta no fim. Sem este par o teste acima seria decorativo."""
    quebrada = tema.folha_de_estilo("claro") + "\nQPushButton { color: "
    avisos = _polir_com(quebrada, qtbot)
    assert [a for a in avisos if "parse" in a.lower()], (
        "o Qt não avisou sobre a folha quebrada — então o teste da folha válida não mede nada"
    )


def test_esquema_desconhecido_levanta_em_vez_de_recuar() -> None:
    with pytest.raises(KeyError):
        tema.folha_de_estilo("sepia")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Os recursos existem e carregam — e a folha aponta para eles
# ---------------------------------------------------------------------------


def test_cada_recurso_existe_e_carrega_como_imagem(qtbot: Any) -> None:
    pasta = tema.pasta_de_recursos()
    for nome in tema.SETAS:
        caminho = pasta / nome
        assert caminho.is_file(), caminho
        assert not QImage(str(caminho)).isNull(), f"{nome} não carrega — falta o plugin SVG?"
        # Cada recurso é usado por ALGUM esquema: o visto preto só pelo escuro.
        folhas = tema.folha_de_estilo("claro") + tema.folha_de_estilo("escuro")
        assert str(caminho).replace("\\", "/") in folhas


def test_o_portao_do_recurso_acusa_arquivo_que_nao_e_imagem(tmp_path: Path, qtbot: Any) -> None:
    """A planta: um `.svg` que não é SVG. `QImage` devolve nula, ou o teste acima é vácuo."""
    falso = tmp_path / "seta-baixo.svg"
    falso.write_text("isto não é svg", encoding="utf-8")
    assert QImage(str(falso)).isNull()


# ---------------------------------------------------------------------------
# A paleta segue os mesmos tokens da folha
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cores", [tema.CLARO, tema.ESCURO])
def test_a_paleta_usa_os_tokens_da_folha(cores: tema.Cores) -> None:
    p = tema.paleta(cores)
    assert p.color(QPalette.ColorRole.Window).name() == cores.fundo
    assert p.color(QPalette.ColorRole.Base).name() == cores.campo
    assert p.color(QPalette.ColorRole.Highlight).name() == cores.gradiente_b
    assert p.color(QPalette.ColorRole.WindowText).name() == cores.texto


def test_os_dois_esquemas_diferem_no_fundo() -> None:
    """Sem isto, um `ESCURO` copiado do `CLARO` passaria em todos os testes acima."""
    assert tema.CLARO.fundo != tema.ESCURO.fundo
    assert tema.CLARO.texto != tema.ESCURO.texto


# ---------------------------------------------------------------------------
# Os papéis dos botões
# ---------------------------------------------------------------------------


def test_papel_desconhecido_e_recusado(qtbot: Any) -> None:
    botao = QPushButton("x")
    qtbot.addWidget(botao)
    with pytest.raises(ValueError, match="papel de botão desconhecido"):
        tema.marca_papel(botao, "primário")  # com acento: o erro mais provável de digitação


def test_cada_papel_tem_regra_na_folha() -> None:
    folha = tema.folha_de_estilo("claro")
    for papel in tema.PAPEIS:
        assert f'QPushButton[papel="{papel}"]' in folha, papel


def test_veste_a_janela_marca_primarios_e_perigos(qtbot: Any) -> None:
    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    tema.veste_a_janela(janela)
    papeis = [b.property("papel") for b in janela.findChildren(QPushButton)]
    primarios = papeis.count("primario")
    perigos = papeis.count("perigo")
    # Contagem e não presença: «existe um» seria satisfeito por um laço que marcasse só o primeiro.
    # 3 leituras + 3 etapas + multiverso + vias; Parar e Cancelar continuam perigos. **Eram 9 até
    # 09/11/2026**, com o «Executar pipeline», que saiu com a aba «Executar e exportar»: o primário
    # que dispara a execução passou a ser o botão da Análise, «3. Analisar e registrar».
    assert (primarios, perigos) == (8, 2), (primarios, perigos)
    assert janela.pipeline.botoes_de_etapa[3].property("papel") == "primario"
    assert janela.trabalhos.botao_de_cancelamento.property("papel") == "perigo"


# ---------------------------------------------------------------------------
# A navegação lateral
# ---------------------------------------------------------------------------


def _vestida(qtbot: Any) -> tuple[JanelaPrincipal, tema.BarraLateral]:
    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    return janela, tema.veste_a_janela(janela)


def test_icones_e_descricoes_cobrem_exatamente_as_abas(qtbot: Any) -> None:
    """Bijeção nas duas direções: aba sem ícone levantaria `KeyError` no arranque, e ícone de aba
    que deixou de existir é texto morto. Derivado de `abas`, não de uma lista redigitada."""
    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    titulos = {janela.abas.tabText(i) for i in range(janela.abas.count())}
    assert titulos, "conjunto vazio passaria por vacuidade"
    assert set(tema.ICONES) == titulos
    assert set(tema.DESCRICOES) == titulos


# Os nomes de formato de imagem que uma descrição de aba poderia prometer, em maiúsculas.
_FORMATOS_CONHECIDOS = re.compile(r"\b(PNG|TIFF?|PDF|SVG|JPE?G|EPS|BMP)\b")


def _formatos_prometidos(descricao: str) -> set[str]:
    return {"TIFF" if f == "TIF" else f for f in _FORMATOS_CONHECIDOS.findall(descricao)}


def test_a_descricao_dos_graficos_promete_os_formatos_que_a_gravacao_escreve() -> None:
    """O cabeçalho de «Gráficos» diz os formatos de `FORMATOS_DA_EXPORTACAO`, nem mais nem menos.

    Ele dizia *«para ver e gravar em PNG»* desde a janela de sete abas, e a gravação passou a
    escrever PNG, TIFF e PDF. `DESCRICOES` declara que cada frase descreve o que a aba
    **já** mostra; o conjunto esperado sai do mapa que a gravação consulta, e não deste teste.
    """
    from refino.desktop.janela.figuras import FORMATOS_DA_EXPORTACAO

    esperados = {formato.upper() for formato in FORMATOS_DA_EXPORTACAO.values()}
    assert esperados, "âncora: a gravação não declara formato nenhum"
    assert _formatos_prometidos(tema.DESCRICOES["Gráficos"]) == esperados, tema.DESCRICOES[
        "Gráficos"
    ]
    # O par, em processo: a frase da janela de sete abas é acusada.
    antiga = "As figuras do diagnóstico e da correção, para ver e gravar em PNG."
    assert _formatos_prometidos(antiga) == {"PNG"} != esperados


def test_cada_icone_desenha(qtbot: Any) -> None:
    for nome, tracos in tema.ICONES.items():
        assert not tema.icone(tracos, "#ffffff").isNull(), nome
    assert tema.icone("<isto não é svg", "#ffffff").isNull(), "o par: traço inválido dá nulo"


def test_a_barra_comanda_as_abas_e_segue_a_troca_por_codigo(qtbot: Any) -> None:
    janela, barra = _vestida(qtbot)
    assert janela.abas.tabBar().isHidden()
    assert len(barra.botoes) == janela.abas.count()
    barra.botoes[3].click()
    assert janela.abas.currentIndex() == 3
    # A outra direção: `principal.py` troca de aba por código (o atalho para «Gráficos»).
    janela.abas.setCurrentIndex(1)
    assert barra.botoes[1].isChecked() and not barra.botoes[3].isChecked()
    titulo = janela.findChild(QLabel, "titulo_da_pagina")
    assert titulo is not None and titulo.text() == janela.abas.tabText(1)


def test_a_barra_e_sempre_larga_e_com_os_nomes(qtbot: Any) -> None:
    """A barra não vira mais trilho de ícones, em janela nenhuma."""
    janela, barra = _vestida(qtbot)
    janela.show()
    for largura in (1024, 1280, 1920):
        janela.resize(largura, 760)
        QApplication.processEvents()
        assert barra.width() == tema.LARGURA_DA_BARRA, largura
        assert all(botao.text() for botao in barra.botoes), largura


class _DwmFalso:
    """O `dwmapi` do Windows, de mentira: devolve as respostas dadas, por atributo, e conta."""

    def __init__(self, respostas: dict[int, int]) -> None:
        self.respostas = respostas
        self.pedidos: list[int] = []

    def DwmSetWindowAttribute(self, _janela: int, atributo: int, _valor: Any, _tamanho: int) -> int:  # noqa: N802
        self.pedidos.append(atributo)
        return self.respostas.get(atributo, 1)


def test_a_moldura_escura_tenta_o_atributo_novo_e_depois_o_antigo(qtbot: Any) -> None:
    """Par: o Windows aceita o 20 → um pedido; recusa o 20 e aceita o 19 → dois; recusa os dois →
    `False`, sem levantar. A cor real da barra foi medida à parte, num ecrã de verdade."""
    janela = QWidget()
    qtbot.addWidget(janela)
    aceita_novo = _DwmFalso({20: 0})
    assert tema.escurece_a_moldura(janela, dwm=aceita_novo) is True
    assert aceita_novo.pedidos == [20]
    so_antigo = _DwmFalso({19: 0})
    assert tema.escurece_a_moldura(janela, dwm=so_antigo) is True
    assert so_antigo.pedidos == [20, 19]
    recusa = _DwmFalso({})
    assert tema.escurece_a_moldura(janela, dwm=recusa) is False
    assert recusa.pedidos == [20, 19]


def test_a_moldura_escura_fora_do_windows_nao_faz_nada(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tema.sys, "platform", "linux")
    janela = QWidget()
    qtbot.addWidget(janela)
    assert tema.escurece_a_moldura(janela) is False


def test_o_padrao_e_o_escuro() -> None:
    """O tema padrão é o escuro, e não segue mais o Windows."""
    assert tema.ESQUEMA_PADRAO == "escuro"
    assert tema.folha_de_estilo() == tema.folha_de_estilo("escuro")


def test_sem_veste_nenhum_botao_tem_papel(qtbot: Any) -> None:
    """O par do anterior: a contagem 6/2 vem de `veste_a_janela`, não da construção da janela."""
    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    assert not [b for b in janela.findChildren(QPushButton) if b.property("papel")]


# ---------------------------------------------------------------------------
# O arranque veste a aplicação
# ---------------------------------------------------------------------------


def test_veste_a_aplicacao_aplica_estilo_paleta_e_folha(qtbot: Any) -> None:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    estilo_antes, paleta_antes, folha_antes = app.style().name(), app.palette(), app.styleSheet()
    fonte_antes = app.font()
    try:
        escolhido = tema.veste_a_aplicacao(app, "escuro")
        assert escolhido == "escuro"
        assert app.styleSheet() == tema.folha_de_estilo("escuro")
        assert app.palette().color(QPalette.ColorRole.Window).name() == tema.ESCURO.fundo
        # **Com folha ativa, `app.style()` é o estilo-de-folha do Qt, de nome VAZIO** — medido: a
        # primeira versão desta asserção leu `''` e acusou o tema, quando o instrumento lia a camada
        # errada. O Fusion está por baixo; desligar a folha o expõe.
        app.setStyleSheet("")
        assert app.style().name().lower() == "fusion"
    finally:
        # A `QApplication` é partilhada pela sessão inteira: deixar o tema aplicado mudaria o
        # tamanho dos widgets dos outros módulos, e portões de layout medem tamanho.
        app.setStyleSheet(folha_antes)
        app.setPalette(paleta_antes)
        app.setFont(fonte_antes)
        app.setStyle(estilo_antes)


def test_o_arranque_chama_o_tema_antes_da_janela() -> None:
    """Por leitura de fonte: `_corre_a_janela` chama `app.exec()`, e o teste penduraria.

    A ORDEM é o conteúdo — tema depois da janela é repolir a árvore inteira.
    """
    import inspect

    fonte = inspect.getsource(arranque._corre_a_janela)
    i_tema = fonte.index("veste_a_aplicacao(app")
    i_janela = fonte.index("JanelaPrincipal(constroi_transporte())")
    i_papeis = fonte.index("veste_a_janela(janela)")
    assert i_tema < i_janela < i_papeis


# ---------------------------------------------------------------------------
# A remontagem da aba de preparo
# ---------------------------------------------------------------------------


def test_as_frases_dos_cartoes_de_leitura_bijetam_com_os_rotulos() -> None:
    """O comentário de `FRASES_DA_LEITURA` afirma a bijeção por portão; este é o portão."""
    from refino.desktop.janela.leitura import FRASES_DA_LEITURA, ROTULOS_DA_LEITURA

    assert ROTULOS_DA_LEITURA, "âncora: rótulos vazios tornariam a igualdade vácua"
    assert set(FRASES_DA_LEITURA) == set(ROTULOS_DA_LEITURA)


def test_cada_leitura_mantem_botao_e_resultado_ligados_ao_mesmo_cartao(qtbot: Any) -> None:
    from refino.desktop.janela.leitura import PainelDeLeitura

    painel = PainelDeLeitura()
    qtbot.addWidget(painel)
    assert painel.cartoes, "âncora: nenhum cartão"
    for chave, cartao in painel.cartoes.items():
        # O resultado continua no bloco de detalhes do próprio cartão, mesmo quando esse bloco
        # ocupa uma linha inteira abaixo dos três cartões.
        assert cartao.isAncestorOf(painel.botoes[chave]), chave
        assert cartao.detalhes.isAncestorOf(painel.rotulos[chave]), chave


@pytest.mark.parametrize("chave", ["validacao", "diagnostico", "recomendacao"])
def test_detalhes_da_leitura_abrem_abaixo_dos_tres_cartoes_sem_executar(
    chave: str, qtbot: Any
) -> None:
    """A ajuda está disponível antes de qualquer retorno, numa faixa da largura da página."""
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QSignalSpy

    from refino.desktop.janela.leitura import ROTULOS_DA_LEITURA, SEM_LEITURA, PainelDeLeitura

    painel = PainelDeLeitura()
    qtbot.addWidget(painel)
    painel.resize(1080, 650)
    painel.show()
    pedidos = QSignalSpy(painel.pedir)
    detalhe = painel.paineis_de_detalhes[chave]
    pai_inicial = detalhe.parentWidget()
    assert detalhe is painel.cartoes[chave].detalhes
    assert not detalhe.isVisibleTo(painel)

    painel.cartoes[chave].alterna.click()
    _espera_transicoes(painel, qtbot)

    assert detalhe.isVisibleTo(painel)
    assert not painel.cartoes[chave].isAncestorOf(detalhe)
    assert detalhe.parentWidget() is pai_inicial, "abrir detalhes não deve reparentar widgets"
    assert pedidos.count() == 0, "consultar a explicação não pode executar a leitura"
    textos_estaticos = [
        rotulo.text()
        for rotulo in detalhe.findChildren(QLabel)
        if rotulo is not painel.rotulos[chave]
    ]
    titulo = f"{ROTULOS_DA_LEITURA[chave]} — em detalhe"
    assert titulo in textos_estaticos
    assert any(t.strip() and t not in (titulo, SEM_LEITURA) for t in textos_estaticos), (
        "o painel precisa explicar a leitura mesmo sem resultado"
    )
    fim_dos_cartoes = max(
        cartao.mapTo(painel, QPoint(0, cartao.height())).y() for cartao in painel.cartoes.values()
    )
    assert detalhe.mapTo(painel, QPoint(0, 0)).y() >= fim_dos_cartoes
    assert detalhe.width() > 2 * max(c.width() for c in painel.cartoes.values())

    painel.cartoes[chave].alterna.click()
    _espera_transicoes(painel, qtbot)
    assert not detalhe.isVisibleTo(painel)
    assert detalhe.parentWidget() is pai_inicial


def test_so_um_detalhe_abre_e_o_resultado_atualiza_no_painel_correto(qtbot: Any) -> None:
    """Trocar a ajuda fecha a anterior; uma resposta muda só o resultado da leitura que voltou."""
    from refino.desktop.janela.leitura import SEM_LEITURA, PainelDeLeitura

    painel = PainelDeLeitura()
    qtbot.addWidget(painel)
    painel.cartoes["validacao"].alterna.click()
    painel.cartoes["diagnostico"].alterna.click()
    assert not painel.cartoes["validacao"].alterna.isChecked()
    assert not painel.paineis_de_detalhes["validacao"].isVisibleTo(painel)
    assert painel.cartoes["diagnostico"].alterna.isChecked()
    assert painel.paineis_de_detalhes["diagnostico"].isVisibleTo(painel)
    assert not painel.paineis_de_detalhes["recomendacao"].isVisibleTo(painel)

    resultado = painel.rotulos["diagnostico"]
    pai_inicial = resultado.parentWidget()
    escrito = painel.mostra("diagnostico", {"defect_type": "gradual_drift", "confidence": 0.72})
    assert resultado is painel.rotulos["diagnostico"]
    assert resultado.text() == escrito and escrito != SEM_LEITURA
    assert resultado.parentWidget() is pai_inicial
    assert painel.rotulos["validacao"].text() == SEM_LEITURA
    assert painel.paineis_de_detalhes["diagnostico"].isVisibleTo(painel)
    assert not painel.cartoes["validacao"].alterna.isChecked()

    painel.cartoes["diagnostico"].alterna.click()
    assert not any(p.isVisibleTo(painel) for p in painel.paineis_de_detalhes.values())


def test_remover_arquivo_obrigatorio_fecha_os_detalhes_da_entrada_anterior(
    qtbot: Any, tmp_path: Path
) -> None:
    from refino.desktop.janela.leitura import SEM_LEITURA

    arquivo = tmp_path / "arquivo.csv"
    arquivo.write_bytes(b"arquivo anexado")
    janela = JanelaPrincipal(_Nada(), escolhedor=lambda _titulo: arquivo)
    qtbot.addWidget(janela)
    janela.arquivos.escolhe_medidas()
    janela.arquivos.escolhe_lista()
    janela.leitura.mostra(
        "validacao", {"aceita": True, "problemas": [], "n_features": 7, "n_samples": 9}
    )
    janela.leitura.cartoes["validacao"].alterna.click()
    assert janela.leitura.paineis_de_detalhes["validacao"].isVisibleTo(janela.leitura)

    janela.arquivos.remover_arquivo("medidas")

    assert not any(c.alterna.isChecked() for c in janela.leitura.cartoes.values())
    assert not any(
        p.isVisibleTo(janela.leitura) for p in janela.leitura.paineis_de_detalhes.values()
    )
    assert all(r.text() == SEM_LEITURA for r in janela.leitura.rotulos.values())


def test_fechar_detalhes_devolve_a_altura_da_secao_de_leitura(qtbot: Any) -> None:
    """A moldura cresce com a explicação e encolhe quando ela fecha, sem um mínimo preso."""
    from refino.desktop.janela.preparo import AbaDePreparo

    janela, _barra = _vestida(qtbot)
    janela.resize(1280, 760)
    janela.show()
    pagina = janela.findChild(AbaDePreparo)
    assert pagina is not None
    QApplication.processEvents()
    leitura = janela.leitura
    minimo_fechado = leitura.minimumHeight()
    altura_fechada = pagina.secao_da_leitura.height()

    leitura.cartoes["validacao"].alterna.click()
    _espera_transicoes(janela, qtbot)
    qtbot.waitUntil(lambda: leitura.minimumHeight() > minimo_fechado, timeout=2000)
    minimo_aberto = leitura.minimumHeight()
    qtbot.waitUntil(lambda: pagina.secao_da_leitura.height() > altura_fechada, timeout=2000)
    altura_aberta = pagina.secao_da_leitura.height()

    leitura.cartoes["validacao"].alterna.click()
    _espera_transicoes(janela, qtbot)

    qtbot.waitUntil(lambda: leitura.minimumHeight() < minimo_aberto, timeout=2000)
    qtbot.waitUntil(lambda: pagina.secao_da_leitura.height() < altura_aberta, timeout=2000)
    assert not leitura.paineis_de_detalhes["validacao"].isVisibleTo(leitura)


@pytest.fixture
def aplicativo_com_tema(qapp: QApplication) -> Iterator[None]:
    """Mede as fontes e a folha do arranque real, sem depender de outro teste."""
    fonte, paleta, folha = qapp.font(), qapp.palette(), qapp.styleSheet()
    estilo = qapp.style().name()
    try:
        tema.veste_a_aplicacao(qapp)
        yield
    finally:
        qapp.setStyleSheet(folha)
        qapp.setPalette(paleta)
        qapp.setFont(fonte)
        qapp.setStyle(estilo)


def test_a_caixa_do_resumo_guarda_a_altura_minima_sob_o_tema(
    qtbot: Any, aplicativo_com_tema: None
) -> None:
    """A caixa do resumo dos métodos não encolhe a uma linha quando a página transborda.

    **Medido em 09/11/2026, com o resumo da Análise na «Qualidade do resultado»:** a caixa ficava
    com 34 px, porque a regra geral `min-height` dos campos, aplicada no polimento, sobrescreve o
    `setMinimumHeight(160)` do código — e a rolagem da aba dá a cada bloco a altura mínima quando o
    conteúdo passa da janela. **Par causal na mesma função:** um `QPlainTextEdit` sem o nome, com o
    mesmo `setMinimumHeight(160)`, fica com a altura da regra geral.
    """
    from PySide6.QtWidgets import QPlainTextEdit

    from refino.desktop.janela.execucao import PainelDoResumo

    sem_nome = QPlainTextEdit()
    qtbot.addWidget(sem_nome)
    sem_nome.setMinimumHeight(160)
    sem_nome.show()
    painel = PainelDoResumo(salvador=lambda: None)
    qtbot.addWidget(painel)
    painel.show()
    QApplication.processEvents()
    assert sem_nome.minimumHeight() < 160, (
        f"âncora: a regra geral não sobrescreveu a altura ({sem_nome.minimumHeight()} px), e a "
        "regra pelo nome não separa nada"
    )
    assert painel.caixa.minimumHeight() >= 160, (
        f"a caixa do resumo encolheu para {painel.caixa.minimumHeight()} px sob o tema"
    )


@pytest.mark.parametrize("bloco", ["detalhes", "legenda"])
def test_recolher_um_bloco_nao_estica_os_cartoes_entre_quadros(
    bloco: str, qtbot: Any, aplicativo_com_tema: None
) -> None:
    """Recolher diminui a altura; não a redistribui temporariamente pelos cartões."""
    from refino.desktop.janela.preparo import AbaDePreparo

    class Alturas(QObject):
        def __init__(self, alvo: QWidget) -> None:
            super().__init__(alvo)
            self.medidas: list[int] = []
            alvo.installEventFilter(self)

        def eventFilter(self, watched: QObject, evento: QEvent) -> bool:  # noqa: N802
            if evento.type() == QEvent.Type.Resize:
                assert isinstance(watched, QWidget)
                self.medidas.append(watched.height())
            return False

    janela, _barra = _vestida(qtbot)
    janela.resize(1280, 800)
    janela.show()
    pagina = janela.findChild(AbaDePreparo)
    assert pagina is not None
    QApplication.processEvents()
    leitura = janela.leitura
    botao = leitura.cartoes["validacao"].alterna if bloco == "detalhes" else pagina.legenda.alterna
    altura_inicial = pagina.secao_da_leitura.height()
    botao.click()
    qtbot.waitUntil(lambda: pagina.secao_da_leitura.height() > altura_inicial)
    _espera_transicoes(janela, qtbot)
    vigia_dos_cartoes = Alturas(leitura)
    vigia_da_secao = Alturas(pagina.secao_da_leitura)
    alturas_abertas = (leitura.height(), pagina.secao_da_leitura.height())

    botao.click()
    qtbot.waitUntil(lambda: pagina.secao_da_leitura.height() == altura_inicial)
    _espera_transicoes(janela, qtbot)

    assert vigia_da_secao.medidas
    for altura_aberta, vigia in zip(
        alturas_abertas, (vigia_dos_cartoes, vigia_da_secao), strict=True
    ):
        alturas = [altura_aberta, *vigia.medidas]
        assert all(atual <= anterior for anterior, atual in pairwise(alturas)), alturas


def test_mostrar_uma_leitura_atualiza_o_selo_e_a_manchete_e_guarda_o_texto(qtbot: Any) -> None:
    """Par causal: o cartão da validação muda com a leitura dela, e o do diagnóstico NÃO muda."""
    from refino.desktop.janela.leitura import PainelDeLeitura

    painel = PainelDeLeitura()
    qtbot.addWidget(painel)
    assert painel.cartoes["validacao"].selo.tipo == "neutro"
    corpo = {"aceita": True, "problemas": [], "n_features": 2184, "n_samples": 68}
    texto = painel.mostra("validacao", corpo)
    assert painel.cartoes["validacao"].selo.tipo == "sucesso"
    assert "2184 features" in painel.cartoes["validacao"].manchete.text()
    assert painel.rotulos["validacao"].text() == texto, "o texto inteiro continua no rótulo"
    assert painel.cartoes["diagnostico"].selo.tipo == "neutro", "a outra leitura não foi feita"


def test_o_campo_de_cada_arquivo_mostra_o_estado_do_seu_arquivo(tmp_path: Path, qtbot: Any) -> None:
    """Par causal: o campo de «Medições» muda com a escolha dele, e o de «Lista» NÃO muda."""
    from refino.desktop.janela.arquivos import SEM_ARQUIVO, PainelDeArquivos

    medidas = tmp_path / "measurements_pos_hilic.csv"
    medidas.write_text("a,b\n1,2\n", encoding="utf-8")
    painel = PainelDeArquivos(lambda _titulo: medidas)
    qtbot.addWidget(painel)
    assert painel.estados["medidas"].text() == SEM_ARQUIVO
    assert painel.selo.text() == "0 / 2 obrigatórios"

    painel.escolhe_medidas()

    assert painel.estados["medidas"].text() == medidas.name
    assert painel.estados["medidas"].property("preenchido") is True
    assert painel.estados["lista"].text() == SEM_ARQUIVO, "a outra metade não foi escolhida"
    assert not painel.estados["lista"].property("preenchido")
    assert (painel.selo.text(), painel.selo.tipo) == ("1 / 2 obrigatórios", "aviso")


@pytest.mark.parametrize(
    ("segmento", "modo"),
    [
        ("pos_hilic", "pos_hilic"),
        ("neg_hilic", "neg_hilic"),
        ("pos_c18", "pos_c18"),
        ("neg_c18", "neg_c18"),
    ],
)
def test_o_segmento_do_modo_envia_o_identificador_original(
    segmento: str, modo: str, qtbot: Any, tmp_path: Path
) -> None:
    """Cada segmento mostra e envia o mesmo modo, e o contrato e os bytes ficam."""
    from PySide6.QtTest import QSignalSpy

    from refino.desktop.janela.arquivos import (
        ROTULO_DA_LISTA,
        ROTULO_DAS_MEDIDAS,
        PainelDeArquivos,
    )

    medidas = tmp_path / "measurements.csv"
    lista = tmp_path / "Sample_List.csv"
    medidas.write_bytes(b"medicoes escolhidas")
    lista.write_bytes(b"ordem escolhida")
    por_titulo = {ROTULO_DAS_MEDIDAS: medidas, ROTULO_DA_LISTA: lista}
    painel = PainelDeArquivos(por_titulo.get)
    qtbot.addWidget(painel)
    painel.escolhe_medidas()
    par = painel.escolhe_lista()
    assert par is not None
    painel.modo.setEditText("outro_modo")
    alteracoes = QSignalSpy(painel.modo.currentTextChanged)

    painel.segmentos_do_modo[segmento].click()

    assert painel.modo.currentText() == modo
    assert painel.campos_para_envio() == {"mode": modo}
    assert alteracoes.count() == 1, "o clique deve passar pelo sinal original do modo"
    assert alteracoes.at(0) == [modo]
    assert painel.par is par, "trocar modo não deve reabrir nem substituir os arquivos"
    assert par.bytes_das_medidas == medidas.read_bytes()
    assert par.bytes_da_lista == lista.read_bytes()
    marcados = [nome for nome, botao in painel.segmentos_do_modo.items() if botao.isChecked()]
    assert marcados == [segmento]


def test_os_segmentos_seguem_o_modo_programatico_e_permitem_um_modo_novo(
    qtbot: Any, tmp_path: Path
) -> None:
    """A sincronização de volta marca o segmento do modo e limpa sugestões para um rótulo novo."""
    from PySide6.QtCore import Qt

    from refino.desktop.janela.arquivos import PainelDeArquivos

    arquivo = tmp_path / "arquivo.csv"
    arquivo.write_bytes(b"arquivo escolhido")
    painel = PainelDeArquivos(lambda _titulo: arquivo)
    qtbot.addWidget(painel)
    painel.escolhe_medidas()
    painel.escolhe_lista()
    painel.show()
    # O seletor mora em «Opções avançadas», fechado: o gesto de edição pede-o aberto.
    painel.mostra_opcoes_avancadas()
    for modo, segmento in (("pos_c18", "pos_c18"), ("neg_c18", "neg_c18")):
        painel.modo.setCurrentText(modo)
        marcados = [nome for nome, b in painel.segmentos_do_modo.items() if b.isChecked()]
        assert marcados == [segmento]

    assert painel.modo.isEditable(), "o conjunto de modos continua aberto"
    qtbot.mouseDClick(painel.segmentos_do_modo["neg_c18"], Qt.MouseButton.LeftButton)
    assert painel.modo.isVisibleTo(painel), "o gesto de edição deve expor o campo personalizado"
    editor = painel.modo.lineEdit()
    assert editor is not None
    qtbot.keyClicks(editor, "pos_lipidomica")
    qtbot.keyClick(editor, Qt.Key.Key_Return)
    assert painel.modo.currentText() == "pos_lipidomica"
    assert painel.campos_para_envio() == {"mode": "pos_lipidomica"}
    assert not any(b.isChecked() for b in painel.segmentos_do_modo.values()), (
        "um modo novo não deve aparecer como um dos quatro sugeridos"
    )


def test_segue_a_altura_fixa_o_minimo_na_altura_que_o_texto_pede(qtbot: Any) -> None:
    """O par: sem o filtro, o mínimo fica abaixo de `heightForWidth`; com ele, igual."""
    from PySide6.QtWidgets import QVBoxLayout

    def _monta() -> QWidget:
        caixa = QWidget()
        disposicao = QVBoxLayout(caixa)
        rotulo = QLabel("palavra " * 200, caixa)
        rotulo.setWordWrap(True)
        disposicao.addWidget(rotulo)
        qtbot.addWidget(caixa)
        return caixa

    sem = _monta()
    sem.resize(300, 50)
    sem.show()
    QApplication.processEvents()
    # Âncora do par: SEM o filtro, o mínimo fica abaixo do que o texto pede. «Zero» foi a primeira
    # versão, e era falsa — janela de topo com layout ganha o mínimo do layout (34 px aqui).
    assert sem.minimumHeight() < sem.heightForWidth(sem.width())

    com = _monta()
    tema.SegueAAltura(com)
    com.resize(300, 50)
    com.show()
    QApplication.processEvents()
    pedida = com.heightForWidth(com.width())
    assert pedida > 50, "âncora: o texto tem de pedir mais que a altura dada, senão o par é vácuo"
    assert com.minimumHeight() == pedida


def test_a_folha_de_cada_etapa_tem_o_minimo_que_o_conteudo_pede(qtbot: Any) -> None:
    """A folha de cada página de etapa tem altura mínima igual à que o conteúdo pede.

    **O defeito, medido no passeio de 10/11/2026 a 1280x860:** com a Normalização feita e o formato
    MetaboAnalyst escolhido sob escalonamento, o Resultado ganha o motivo do «Baixar» desligado — um
    texto longo numa coluna estreita. A folha, de política vertical `Maximum`, ficava presa ao
    `sizeHint`, que não sabe a largura: num processo sem tela, 1896 px contra 2016 pedidos. O layout
    tirava a diferença de quem tem pouco mínimo: o lugar do bloco «Importar fatores de normalização»
    caía para 32 px, e a nota do escalonamento perdia a última linha.

    **O que se afirma é a guarda, e não o aperto, e a escolha é medida.** Sob o pytest a relação se
    inverte — o `sizeHint` do layout dá 2012 e o conteúdo pede 1947 —, e um teste que esperasse o
    aperto não o veria. A prova de ponta a ponta é a fase B do arnês do passeio, na plataforma
    nativa, com a captura olhada: o lugar do bloco com os 152 px que ele pede.

    **E a guarda mora na folha, que é quem aperta.** A primeira tentativa a pôs no próprio bloco:
    ele ficava com a altura certa, o teste dela passava, e o lugar dele no formulário continuava com
    32 px — o texto seguia invisível, e só a captura da janela o mostrou.

    O par causal foi visto à mão: sem `SegueAAltura` na folha, o mínimo é 0 e a asserção reprova.
    """
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    anteriores = app.font(), app.palette(), app.styleSheet()
    tema.veste_a_aplicacao(app)
    try:
        janela = JanelaPrincipal(_Nada())
        qtbot.addWidget(janela)
        tema.veste_a_janela(janela)
        janela.resize(1280, 860)
        janela.show()
        paginas = janela._paginas_de_etapa
        assert sorted(paginas) == [1, 2, 3], "âncora: as três páginas de etapa"
        for etapa, pagina in paginas.items():
            pagina.mostra_arquivos(())
            janela.abas.setCurrentWidget(pagina)
            _espera_transicoes(janela, qtbot)
            folha = pagina.folha
            pedida = folha.heightForWidth(folha.width())
            assert pedida > 0, f"âncora: a folha da etapa {etapa} pede altura pela largura"
            assert folha.minimumHeight() >= pedida, (
                f"a folha da etapa {etapa} tem mínimo {folha.minimumHeight()} px e o conteúdo pede "
                f"{pedida}: presa ao sizeHint, ela espreme os blocos quando a página cresce"
            )
    finally:
        app.setFont(anteriores[0])
        app.setPalette(anteriores[1])
        app.setStyleSheet(anteriores[2])


def test_as_paginas_que_nao_cabiam_rolam_desde_a_composicao(qtbot: Any) -> None:
    """«Corrigir», «Multiverso» e «Vias» punham a janela acima de 768 px. As três rolam
    SEM tema — a rolagem é da composição, não da casca —, e a da qualidade continua a ser o painel,
    que é o que `test_desktop_qualidade.py` afirma por `isinstance`.

    **«Corrigir» é hoje as três páginas de etapa**: a aba que levava o formulário inteiro —
    «Executar e exportar», na última forma — saiu em 09/11/2026, e cada etapa tem a rolagem dela."""
    from PySide6.QtWidgets import QScrollArea

    from refino.desktop.janela.principal import ABAS
    from refino.desktop.janela.qualidade import PainelDeQualidade

    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    por_painel = {aba.paineis: janela.abas.widget(i) for i, aba in enumerate(ABAS)}
    assert por_painel, "âncora: nenhuma aba"
    for chave in ("multiverso", "vias"):
        pagina = next(w for ps, w in por_painel.items() if ps == (chave,))
        assert isinstance(pagina, QScrollArea), chave
        assert pagina.widget() is getattr(janela, chave)
    etapas = [janela.abas.widget(i) for i, aba in enumerate(ABAS) if aba.etapa is not None]
    assert len(etapas) == 3, "âncora: as três abas de etapa"
    for pagina in etapas:
        assert pagina is not None and pagina.findChild(QScrollArea) is not None
    assert isinstance(por_painel[("qualidade",)], PainelDeQualidade)


def test_a_casca_nao_tira_a_pilha_de_abas_do_lugar(qtbot: Any) -> None:
    """Mover o `QTabWidget` de pai quebrava o processo na coleta de lixo. A casca põe a barra
    num painel acoplado e INSERE o título; as abas ficam no widget central de antes."""
    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    pai_antes = janela.abas.parentWidget()
    central_antes = janela.centralWidget()
    assert pai_antes is not None, "âncora: as abas têm de ter pai para o par significar algo"
    barra = tema.veste_a_janela(janela)
    assert janela.abas.parentWidget() is pai_antes
    assert janela.centralWidget() is central_antes
    assert barra.parentWidget() is not central_antes, "a barra vive na doca, fora do central"


def test_as_tres_etapas_tem_paginas_proprias_com_os_controles_e_a_acao(qtbot: Any) -> None:
    """Correção, normalização e análise são passos visíveis separados.

    O par de cada página é o controle de outra etapa: estar na página certa não pode significar
    continuar na página única antiga, com todos os quadros juntos.
    """
    from refino.desktop.janela.principal import ABAS

    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    pipeline = janela.pipeline
    declaradas = {aba.etapa: (i, aba) for i, aba in enumerate(ABAS) if aba.etapa is not None}
    assert set(declaradas) == {1, 2, 3}
    assert [declaradas[e][1].titulo for e in (1, 2, 3)] == [
        "Correção",
        "Normalização",
        "Análise estatística",
    ]
    controles = {
        1: (pipeline.corretor, pipeline.n_lotes, pipeline.variante_do_rsd),
        2: (pipeline.max_features, pipeline.corte, *pipeline.eixos.values()),
        3: (pipeline.estudo, pipeline.grupo_a, pipeline.grupo_b, *pipeline.analises.values()),
    }
    for etapa, (indice, _aba) in declaradas.items():
        pagina = janela.abas.widget(indice)
        assert pagina is not None
        assert all(pagina.isAncestorOf(controle) for controle in controles[etapa])
        assert pagina.isAncestorOf(pipeline.botoes_de_etapa[etapa])
        for outra, da_outra in controles.items():
            if outra != etapa:
                assert not any(pagina.isAncestorOf(controle) for controle in da_outra)
    # **Os downloads moram no Resultado da etapa que os produz desde 09/11/2026**: até ali a aba
    # «Executar e exportar» tinha o painel dos artefatos e o «Executar pipeline», e os dois saíram
    # com ela. O par: o Resultado de uma etapa não está na página da outra.
    resultados = {1: janela.resultado_da_correcao, 2: janela.resultado_da_normalizacao}
    for etapa, resultado in resultados.items():
        pagina = janela.abas.widget(declaradas[etapa][0])
        assert pagina is not None and pagina.isAncestorOf(resultado)
        outra = janela.abas.widget(declaradas[3 - etapa][0])
        assert outra is not None and not outra.isAncestorOf(resultado)
    assert len(ABAS) == 9 and not any(aba.paineis == ("pipeline", "artefatos") for aba in ABAS)


def test_os_parametros_das_tres_paginas_continuam_no_mesmo_pedido(qtbot: Any) -> None:
    """A separação visual conserva a seleção que será enviada pela execução integral."""
    import json

    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    pipeline = janela.pipeline
    # As caixas «Automático» da Correção e do pré-processamento eram desticadas aqui; os
    # campos nascem editáveis desde a saída delas.
    pipeline.corretor.setCurrentText("none")
    pipeline.eixos["sample_normalization"].setCurrentText("sum")
    pipeline.estudo.setText("Estudo da interface")
    pipeline.grupo_a.setText("Controle")
    pipeline.grupo_b.setText("Tratamento")
    pipeline.analises["clustering"].setChecked(False)
    campos = campos_da_tela(pipeline, modo="neg_c18", correcao="none", n_lotes=1)
    assert campos["estudo"] == "Estudo da interface"
    assert campos["correction"] == "none"
    assert campos["sample_normalization"] == "sum"
    assert (campos["group_a"], campos["group_b"]) == ("Controle", "Tratamento")
    assert json.loads(campos["analise_json"])["clustering"] is False


# ---------------------------------------------------------------------------
# Os componentes da remontagem
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chave", "corpo", "tipo", "estado"),
    [
        (
            "validacao",
            {"aceita": True, "problemas": [], "n_features": 9, "n_samples": 3},
            "sucesso",
            "Aceita",
        ),
        (
            "validacao",
            {"aceita": True, "problemas": [{"codigo": "X"}], "n_features": 9, "n_samples": 3},
            "aviso",
            "Aceita com avisos",
        ),
        ("validacao", {"aceita": False, "problemas": [{"codigo": "X"}]}, "erro", "Recusada"),
        (
            "diagnostico",
            {"defect_type": "unassessable", "confidence": 0.0},
            "neutro",
            "Sem medição",
        ),
        ("diagnostico", {"defect_type": "erratic", "confidence": 0.65}, "info", "65% de confiança"),
        (
            "recomendacao",
            {"correction_proposta": "serrf", "avaliavel": True, "abstem": True, "confianca": 0.9},
            "aviso",
            "Abstém-se",
        ),
        (
            "recomendacao",
            {"correction_proposta": "serrf", "avaliavel": True, "abstem": False, "confianca": 0.94},
            "info",
            "94% de confiança",
        ),
        # O Palpite MEDIDO: a abstenção é empate na amostra, e a recusa da rota é
        # «não medido» — o par das duas linhas de cima e o da linha genérica de erro.
        (
            "recomendacao",
            {
                "correction_proposta": "serrf",
                "avaliavel": True,
                "abstem": True,
                "confianca": 0.2,
                "n_features_medidas": 200,
            },
            "aviso",
            "Empate na amostra",
        ),
        (
            "recomendacao",
            {"detail": "com 2 lote(s), os QCs por lote são [5, 2]"},
            "erro",
            "Não medido",
        ),
        ("validacao", None, "erro", "Ilegível"),
        ("diagnostico", {"detail": "faltou um campo"}, "erro", "Erro"),
    ],
)
def test_o_resumo_de_cada_leitura_le_os_campos_do_corpo(
    chave: str, corpo: Any, tipo: str, estado: str
) -> None:
    """Cada linha é um desfecho distinto; juntas, elas são os pares umas das outras: o mesmo
    `chave` com corpos diferentes tem de dar selos diferentes, senão a função ignora o corpo."""
    from refino.desktop.janela.leitura import resumo_da_leitura

    assert resumo_da_leitura(chave, corpo)[:2] == (tipo, estado)


def test_o_diagnostico_sem_medicao_nao_mostra_confianca() -> None:
    """`unassessable` vem com `confidence=0.0`, e «0% de confiança» diria o oposto de «não houve
    medição» — o defeito que `prosa_da_confianca_do_diagnostico` existe para não ter."""
    from refino.desktop.janela.leitura import resumo_da_leitura

    resumo = resumo_da_leitura("diagnostico", {"defect_type": "unassessable", "confidence": 0.0})
    assert "%" not in resumo[1]


def test_selo_recusa_tipo_desconhecido(qtbot: Any) -> None:
    from refino.desktop.janela.componentes import Selo

    selo = Selo("x")
    qtbot.addWidget(selo)
    with pytest.raises(ValueError, match="tipo de selo desconhecido"):
        selo.define("y", "verde")
    assert selo.tipo == "neutro", "a recusa não pode deixar o selo meio trocado"


def test_cada_tipo_de_selo_tem_regra_na_folha() -> None:
    from refino.desktop.janela.componentes import TIPOS_DE_SELO

    folha = tema.folha_de_estilo()
    for tipo in TIPOS_DE_SELO:
        assert f'QLabel#selo_de_estado[tipo="{tipo}"]' in folha, tipo


def test_o_bloco_recolhivel_nasce_fechado_e_abre_num_clique(qtbot: Any) -> None:
    from refino.desktop.janela.componentes import Recolhivel

    bloco = Recolhivel("título")
    qtbot.addWidget(bloco)
    bloco.show()
    assert not bloco.corpo.isVisible()
    bloco.alterna.click()
    assert bloco.corpo.isVisible()


# ---------------------------------------------------------------------------
# O painel de recorte em grupos
# ---------------------------------------------------------------------------


def test_cada_prefixo_do_recorte_tem_categoria_e_selo_valido() -> None:
    """Bijeção prefixo ↔ categoria: item cujo prefixo não tem grupo não seria desenhado."""
    from refino.desktop import recorte
    from refino.desktop.janela.componentes import TIPOS_DE_SELO

    prefixos = {
        recorte.PREFIXO_FORA,
        recorte.PREFIXO_PENDENTE,
        recorte.PREFIXO_SO_LEITURA,
        recorte.PREFIXO_COBERTO,
    }
    categorias = recorte.CATEGORIAS_DO_RECORTE
    assert categorias, "âncora: nenhuma categoria"
    assert {c.prefixo for c in categorias} == prefixos
    assert all(c.tipo_do_selo in TIPOS_DE_SELO for c in categorias)


def test_o_item_do_recorte_nasce_fechado_e_abre_no_clique(qtbot: Any) -> None:
    from refino.desktop.janela.recorte import ItemDoRecorte

    pai = QWidget()
    qtbot.addWidget(pai)
    item = ItemDoRecorte("FORA DESTE PROGRAMA", "título", "o motivo inteiro", pai)
    pai.show()
    assert not item.corpo.isVisible()
    item.cabeca.click()
    assert item.corpo.isVisible()
    assert item.corpo.text() == "o motivo inteiro"


def test_com_maiuscula_sobe_a_primeira_letra_mesmo_depois_de_marcacao() -> None:
    """O par: texto já em maiúscula e texto sem letra ficam iguais."""
    from refino.desktop.janela.componentes import com_maiuscula

    assert com_maiuscula("deriva gradual — o sinal") == "Deriva gradual — o sinal"
    assert com_maiuscula("**não avaliável** — x") == "**Não avaliável** — x"
    assert com_maiuscula("Já em maiúscula") == "Já em maiúscula"
    assert com_maiuscula("— 12 %") == "— 12 %"


def test_a_manchete_do_palpite_mostra_o_nome_do_corretor_e_nao_o_identificador() -> None:
    """O par: um corretor fora da tabela aparece como veio, em vez de sumir."""
    from refino.desktop.janela.leitura import resumo_da_leitura

    corpo = {"correction_proposta": "serrf", "avaliavel": True, "abstem": False, "confianca": 0.9}
    assert resumo_da_leitura("recomendacao", corpo)[2] == "Corretor: SERRF"
    corpo["correction_proposta"] = "none"
    assert resumo_da_leitura("recomendacao", corpo)[2] == "Corretor: Sem correção"
    corpo["correction_proposta"] = "metodo_novo"
    assert resumo_da_leitura("recomendacao", corpo)[2] == "Corretor: metodo_novo"


# ---------------------------------------------------------------------------
# A aba «Corrigir» com uma rolagem só, e os rótulos que somem vazios
# ---------------------------------------------------------------------------


def _rolagens_acima(widget: QWidget) -> list[QWidget]:
    """As `QScrollArea` entre `widget` e a janela — quantas rolagens o envolvem."""
    from PySide6.QtWidgets import QScrollArea

    acima = []
    pai = widget.parentWidget()
    while pai is not None:
        if isinstance(pai, QScrollArea):
            acima.append(pai)
        pai = pai.parentWidget()
    return acima


def test_o_formulario_da_correcao_rola_com_a_pagina_e_so_com_ela(qtbot: Any) -> None:
    """O formulário tinha rolagem própria, de altura fixa, dentro da página que rola — duas
    barras, uma dentro da outra, e os grupos de baixo cortados a meio. A propriedade é «uma rolagem
    acima do formulário», e a ÂNCORA é que há pelo menos uma: sem página que rola, a página
    volta a esticar.

    **O formulário é cada grupo do pipeline, na página da etapa dele, desde 09/11/2026**: até ali
    ele era o painel inteiro, na aba «Executar e exportar», que saiu."""
    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    grupos = [g for grupos in janela.pipeline.grupos_das_etapas.values() for g in grupos]
    assert len(grupos) == 4, f"âncora: os grupos das etapas são {len(grupos)}"
    for grupo in grupos:
        acima = _rolagens_acima(grupo)
        assert acima, f"«{grupo.title()}» não está em rolagem nenhuma, e a página volta a esticar"
        assert len(acima) == 1, f"«{grupo.title()}» está em {len(acima)} rolagens encaixadas"


def test_o_portao_das_rolagens_acusa_a_rolagem_encaixada(qtbot: Any) -> None:
    """O par do portão acima: a mesma contagem sobre uma rolagem dentro de outra dá dois."""
    from PySide6.QtWidgets import QScrollArea

    de_fora = QScrollArea()
    qtbot.addWidget(de_fora)
    de_dentro = QScrollArea()
    folha = QLabel("x")
    de_dentro.setWidget(folha)
    de_fora.setWidget(de_dentro)
    assert len(_rolagens_acima(folha)) == 2


def test_a_pagina_da_etapa_rola_os_parametros_a_acao_e_o_resultado_juntos(qtbot: Any) -> None:
    """Os parâmetros, a ação e o Resultado de cada etapa rolam com a página, nessa ordem, numa
    rolagem só.

    Era «a página inteira rola com Executar dentro dela», e falava da aba «Executar e
    exportar»: lá a ação rolava junto, e «Levar o resultado» vinha depois
    dela. A aba saiu, e a mesma propriedade passou a cada página de etapa, com a ação
    num rodapé fixo. O rodapé saiu, e a propriedade antiga voltou inteira: a ação rola com o
    formulário, entre
    os parâmetros que ela executa e o Resultado que produz, sem rolagem encaixada.

    A ordem passou a ser lida na folha, e não numa seção só. Os parâmetros
    subiram para o bloco 1, «Ajustes», e a ação e o Resultado ficaram no bloco 2, «Execução da
    etapa». «Parâmetros, ação,
    Resultado» é agora: os grupos dentro do bloco 1, o bloco 1 antes do bloco 2 na folha, e a ação
    antes do Resultado dentro do bloco 2.
    """
    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    for etapa, pagina in janela._paginas_de_etapa.items():
        grupos = janela.pipeline.grupos_das_etapas[etapa]
        botao = janela.pipeline.botoes_de_etapa[etapa]
        da_pagina = [pagina.rolagem]
        for grupo in grupos:
            assert _rolagens_acima(grupo) == da_pagina, f"etapa {etapa}: o grupo não rola"
        assert _rolagens_acima(botao) == da_pagina, f"etapa {etapa}: a ação não rola com a página"
        assert _rolagens_acima(pagina.bloco_do_resultado) == da_pagina, (
            f"etapa {etapa}: o Resultado não rola com os parâmetros"
        )
        assert pagina.secao_da_entrada.isAncestorOf(grupos[-1]), (
            f"etapa {etapa}: os parâmetros saíram do bloco 1, «Ajustes»"
        )
        folha = pagina.folha.layout()
        assert folha is not None
        blocos = [folha.indexOf(pagina.secao_da_entrada), folha.indexOf(pagina.secao_da_etapa)]
        corpo = pagina.secao_da_etapa.corpo
        no_bloco_2 = [corpo.indexOf(pagina.linha_da_acao), corpo.indexOf(pagina.bloco_do_resultado)]
        assert min(blocos + no_bloco_2) >= 0, f"etapa {etapa}: {blocos}, {no_bloco_2}"
        assert blocos == sorted(blocos) and no_bloco_2 == sorted(no_bloco_2), (
            f"etapa {etapa}: a ordem parâmetros, ação, Resultado mudou: {blocos}, {no_bloco_2}"
        )


def test_o_rotulo_que_some_ocupa_a_linha_so_quando_diz_alguma_coisa(qtbot: Any) -> None:
    """Nas duas direções e com o companheiro: vazio some, texto aparece, e o texto de
    repouso declarado conta como vazio — o objeto e o `text()` são os de sempre."""
    from refino.desktop.janela.componentes import RotuloQueSome

    pai = QWidget()
    qtbot.addWidget(pai)
    companheiro = QLabel("rótulo da linha", pai)
    rotulo = RotuloQueSome("—", pai, vazios=frozenset({"—"}))
    rotulo.acompanha(companheiro)
    assert rotulo.isHidden() and companheiro.isHidden(), "o repouso declarado ocupa a linha"
    assert rotulo.text() == "—", "o texto de repouso mudou para quem o lê"
    rotulo.setText("Pronto: matriz.csv")
    assert not rotulo.isHidden() and not companheiro.isHidden(), "o texto escrito não aparece"
    rotulo.setText("   ")
    assert rotulo.isHidden() and companheiro.isHidden(), "espaço em branco ocupa a linha"


def test_a_linha_do_metaboanalyst_some_quando_nao_ha_frase(qtbot: Any) -> None:
    """Com `none`, cinco dos seis eixos não têm nome no MetaboAnalyst, e o rótulo vazio
    deixava um vão de uma linha no formulário. O par: a imputação tem frase e continua à vista."""
    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    rotulos = janela.pipeline.equivalencias_dos_eixos
    assert rotulos, "âncora: nenhum eixo tem rótulo de equivalência"
    vazios = [eixo for eixo, r in rotulos.items() if not r.text()]
    cheios = [eixo for eixo, r in rotulos.items() if r.text()]
    assert vazios and cheios, f"o par não separa: vazios={vazios}, cheios={cheios}"
    assert all(rotulos[e].isHidden() for e in vazios), "um rótulo vazio ainda ocupa a linha"
    assert not any(rotulos[e].isHidden() for e in cheios), "um rótulo com frase sumiu"


def test_o_desfecho_em_repouso_nao_ocupa_linha_e_aparece_quando_ha_o_que_ler(qtbot: Any) -> None:
    """O «—» de repouso tinha linha própria no fundo de todas as abas."""
    from refino.desktop.janela.principal import SEM_DESFECHO

    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    assert janela.rotulo_de_desfecho.text() == SEM_DESFECHO
    assert janela.rotulo_de_desfecho.isHidden()
    janela.rotulo_de_desfecho.setText("A medição não completou: x")
    assert not janela.rotulo_de_desfecho.isHidden()


# ---------------------------------------------------------------------------
# A ausência com corpo de nota, o estado vazio da figura e as frases que abrem
# ---------------------------------------------------------------------------


def test_o_corpo_do_numero_e_so_do_numero_calculado() -> None:
    """A frase de ausência pesa como nota; o número calculado, como número. O par sai do
    mesmo relatório, e a pergunta é `computed` — não o texto."""
    from refino.desktop.janela.qualidade import (
        ESTILO_DA_AUSENCIA,
        ESTILO_DO_VALOR,
        estilo_do_valor,
    )

    relatorio = {"computed": ["qc_rsd_median"], "qc_rsd_median": 0.25}
    assert ESTILO_DA_AUSENCIA != ESTILO_DO_VALOR, "âncora: os dois corpos são o mesmo"
    assert estilo_do_valor("qc_rsd_median", relatorio) == ESTILO_DO_VALOR
    assert estilo_do_valor("qc_rsd_mean", relatorio) == ESTILO_DA_AUSENCIA
    assert estilo_do_valor("qc_rsd_median", {}) == ESTILO_DA_AUSENCIA


def test_o_estado_vazio_da_figura_e_um_bloco_que_troca_de_lugar_com_o_canvas(qtbot: Any) -> None:
    """A frase de repouso vive no bloco do estado vazio, e o bloco e o canvas nunca estão à
    vista juntos. O oráculo de desenho do painel continua a ver a frase — é o mesmo rótulo."""
    from refino.desktop.janela.figuras import PainelDeFigura

    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    assert painel.repouso.parentWidget() is painel.estado_vazio
    assert painel.estado_vazio.isVisibleTo(painel), "sem figura, o estado vazio não aparece"
    assert not painel.canvas.isVisibleTo(painel), "o canvas vazio aparece ao lado do bloco"
    assert painel._desenhado(painel.repouso), "o oráculo de desenho deixou de ver a frase"
    painel.estado_vazio.setVisible(False)
    assert not painel._desenhado(painel.repouso), "o oráculo vê a frase com o bloco escondido"


def _abrem_em_minuscula(textos: dict[Any, str]) -> list[str]:
    """As frases não vazias cuja primeira letra é minúscula."""
    from refino.desktop.janela.componentes import com_maiuscula

    return [texto[:40] for texto in textos.values() if texto and com_maiuscula(texto) != texto]


def test_as_frases_de_estado_da_figura_abrem_em_maiuscula() -> None:
    """Estas frases aparecem SOZINHAS — no lugar da figura e na linha da gravação —, e a
    revisão das maiúsculas não as alcançou por serem valores de dicionário."""
    from refino.desktop.janela.figuras import TEXTO_DO_DESFECHO, TEXTO_DO_ESTADO
    from refino.desktop.janela.graficos import frase_da_etapa_desatualizada

    assert len(TEXTO_DO_ESTADO) > 5 and TEXTO_DO_DESFECHO, "âncora: as tabelas estão vazias"
    assert _abrem_em_minuscula(TEXTO_DO_ESTADO) == []
    assert _abrem_em_minuscula(TEXTO_DO_DESFECHO) == []
    assert _abrem_em_minuscula({1: frase_da_etapa_desatualizada(1, 1)}) == []


def test_o_portao_das_maiusculas_acusa_a_frase_que_abre_em_minuscula() -> None:
    """O par do portão acima: a mesma pergunta sobre uma tabela plantada acusa a minúscula."""
    assert _abrem_em_minuscula({1: "nenhuma figura", 2: "Já em maiúscula", 3: ""}) == [
        "nenhuma figura"
    ]


# ---------------------------------------------------------------------------
# «Multiverso» e «Vias» em seções numeradas
# ---------------------------------------------------------------------------


def test_o_multiverso_le_se_em_tres_secoes_e_cada_controle_mora_na_sua(qtbot: Any) -> None:
    """Escolher, ler e rever são três seções, nesta ordem, e cada controle vive na do seu
    passo. O par está na mesma função: a tabela da conclusão NÃO está na seção da grade."""
    from refino.desktop.janela.multiverso import PainelDeMultiverso

    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)
    secoes = [painel.secao_da_grade, painel.secao_da_conclusao, painel.secao_do_historico]
    assert [s.numero.text() for s in secoes] == ["1", "2", "3"]
    disposicao = painel.layout()
    assert disposicao is not None
    posicoes = [disposicao.indexOf(s) for s in secoes]
    assert posicoes == sorted(posicoes) and min(posicoes) > 0, posicoes
    moradas = {
        painel.secao_da_grade: [
            painel.botao_de_disparo,
            painel.grupo_a,
            painel.rotulo_de_progresso,
        ],
        painel.secao_da_conclusao: [
            painel.tabela,
            painel.botao_de_leitura,
            *painel.limiares.values(),
        ],
        painel.secao_do_historico: [painel.tabela_do_historico, painel.botao_do_historico],
    }
    for secao, controles in moradas.items():
        fora = [c for c in controles if not secao.isAncestorOf(c)]
        assert not fora, f"{secao.titulo.text()}: fora dela — {fora}"
    assert not painel.secao_da_grade.isAncestorOf(painel.tabela), "par: a âncora não separa"


def test_as_vias_leem_se_em_duas_secoes_e_cada_controle_mora_na_sua(qtbot: Any) -> None:
    """O que se escolhe e o que volta. O par: a tabela NÃO está na seção dos parâmetros."""
    from refino.desktop.janela.vias import PainelDeVias

    painel = PainelDeVias()
    qtbot.addWidget(painel)
    assert painel.secao_dos_picos.isAncestorOf(painel.polaridade)
    assert painel.secao_dos_picos.isAncestorOf(painel.botao)
    assert painel.secao_das_vias.isAncestorOf(painel.tabela)
    assert painel.secao_das_vias.isAncestorOf(painel.recorte)
    assert not painel.secao_dos_picos.isAncestorOf(painel.tabela), "par: a âncora não separa"


def test_os_dois_paineis_beta_abrem_com_a_mesma_faixa_e_ela_tem_regra_na_folha(qtbot: Any) -> None:
    """A frase de beta é uma faixa de aviso nos dois painéis, e a folha a desenha."""
    from refino.desktop.janela.multiverso import PainelDeMultiverso
    from refino.desktop.janela.vias import PainelDeVias

    paineis = [PainelDeMultiverso(), PainelDeVias()]
    for painel in paineis:
        qtbot.addWidget(painel)
    nomes = {painel.rotulo_de_beta.objectName() for painel in paineis}
    assert nomes == {"faixa_beta"}, nomes
    assert "QLabel#faixa_beta" in tema.folha_de_estilo("escuro")


def test_as_estimativas_da_correcao_abrem_em_maiuscula(qtbot: Any) -> None:
    """As duas frases de estimativa abrem a linha delas, no repouso e com número. O par do
    portão é `test_o_portao_das_maiusculas_acusa_a_frase_que_abre_em_minuscula`."""
    from refino.desktop.janela.pipeline import PainelDePipeline

    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    repouso = {1: painel.rotulo_do_palpite.text(), 2: painel.rotulo_do_palpite_da_execucao.text()}
    assert all(repouso.values()), f"âncora: uma estimativa nasceu vazia: {repouso}"
    assert _abrem_em_minuscula(repouso) == []
    com_numero = {1: painel.mostra_palpite_da_medicao(1000), 2: painel.mostra_palpite_da_execucao()}
    assert _abrem_em_minuscula(com_numero) == []


# ---------------------------------------------------------------------------
# A pílula que fica redonda, e a crase que não aparece literal
# ---------------------------------------------------------------------------


def _raio_do_selo(folha: str) -> int:
    """O `border-radius` da regra do selo na folha, em pixels."""
    import re

    bloco = folha.split("QLabel#selo_de_estado {", 1)[1].split("}", 1)[0]
    achado = re.search(r"border-radius:\s*(\d+)px", bloco)
    assert achado is not None, f"a regra do selo não declara raio: {bloco!r}"
    return int(achado.group(1))


def _pilula_redonda(altura: int, raio: int) -> bool:
    """O Qt só desenha o raio se ele couber na metade da altura; senão o canto sai reto."""
    return 2 * raio <= altura


def test_o_selo_tem_altura_para_o_raio_da_folha(qtbot: Any) -> None:
    """Os selos cheios de uma letra tinham ~19 px contra o raio de 10, e saíam quadrados."""
    from refino.desktop.janela.componentes import TIPOS_DE_SELO, Selo

    for esquema in ("claro", "escuro"):
        raio = _raio_do_selo(tema.folha_de_estilo(esquema))
        for tipo in TIPOS_DE_SELO:
            selo = Selo("5", tipo)
            qtbot.addWidget(selo)
            assert _pilula_redonda(selo.minimumHeight(), raio), (esquema, tipo, raio)


def test_o_portao_da_pilula_acusa_o_selo_baixo_demais() -> None:
    """O par do portão acima: a medida de antes do conserto — 19 px com raio 10 — é acusada."""
    assert not _pilula_redonda(19, 10)
    assert _pilula_redonda(20, 10)


def _crases_literais(raiz: QWidget) -> list[str]:
    """Os rótulos e botões que mostram crase literal: texto com «`» fora do formato markdown."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QAbstractButton

    achados = [
        r.text()[:60]
        for r in raiz.findChildren(QLabel)
        if "`" in r.text() and r.textFormat() != Qt.TextFormat.MarkdownText
    ]
    achados += [b.text()[:60] for b in raiz.findChildren(QAbstractButton) if "`" in b.text()]
    return achados


def test_nenhum_rotulo_da_janela_mostra_crase_literal(qtbot: Any) -> None:
    """A classe dos limiares do multiverso e do rótulo das vias: crase
    é marcação de código, e fora de `MarkdownText` ela chega à tela como o caractere. A ÂNCORA é
    que a janela TEM rótulos com crase, em markdown — sem eles a varredura não mediria nada."""
    from PySide6.QtCore import Qt

    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    com_marcacao = [
        r
        for r in janela.findChildren(QLabel)
        if "`" in r.text() and r.textFormat() == Qt.TextFormat.MarkdownText
    ]
    assert com_marcacao, "âncora: nenhum rótulo com crase — a varredura abaixo seria vácua"
    assert _crases_literais(janela) == []


def test_o_portao_das_crases_acusa_o_rotulo_em_texto_simples(qtbot: Any) -> None:
    """O par: o mesmo rótulo, em texto simples, é acusado; em markdown, não."""
    from PySide6.QtCore import Qt

    raiz = QWidget()
    qtbot.addWidget(raiz)
    rotulo = QLabel("Massas já deconvoluídas (`…n`)", raiz)
    rotulo.setTextFormat(Qt.TextFormat.PlainText)
    assert _crases_literais(raiz) == ["Massas já deconvoluídas (`…n`)"]
    rotulo.setTextFormat(Qt.TextFormat.MarkdownText)
    assert _crases_literais(raiz) == []


# ---------------------------------------------------------------------------
# Nomes legíveis, grade que se ajusta e a fonte do corpo
# ---------------------------------------------------------------------------


def test_o_seletor_mostra_o_nome_e_devolve_o_valor(qtbot: Any) -> None:
    """O campo desenha o nome legível e `currentText()` continua o valor técnico — é o que
    mantém o pedido e toda leitura antiga. O par: um valor fora da tabela aparece como veio."""
    from refino.desktop.janela.componentes import ComboComNomes

    combo = ComboComNomes({"serrf": "SERRF"})
    qtbot.addWidget(combo)
    combo.addItems(["serrf", "novo_valor"])
    assert combo.currentText() == "serrf", "o valor deixou de ser o texto do item"
    assert combo.nome_atual() == "SERRF"
    lista = combo.itemDelegate().displayText("serrf", combo.locale())
    assert lista == "SERRF", "a lista aberta mostra o valor e não o nome"
    combo.setCurrentIndex(1)
    assert combo.nome_atual() == "novo_valor", "valor sem nome tem de aparecer como veio"


def test_toda_opcao_dos_seletores_tem_nome_legivel() -> None:
    """Os seletores do pipeline, a variante do RSD e as figuras — nenhum valor do domínio
    sem nome. A âncora é que as populações não estão vazias."""
    from typing import get_args

    from refino.desktop.janela.figuras import entradas_do_seletor
    from refino.desktop.janela.nomes import (
        NOMES_DAS_FIGURAS,
        NOMES_DAS_OPCOES,
        NOMES_DOS_CORRETORES,
    )
    from refino.desktop.janela.pipeline import CORRETORES, EIXOS_DA_GRADE
    from refino.domain.labels import RsdVariant

    opcoes = {valor for _c, _r, _p, valores in EIXOS_DA_GRADE for valor in valores}
    opcoes |= set(get_args(RsdVariant))
    figuras = {rotulo for rotulo, _c in entradas_do_seletor()}
    assert opcoes and figuras and CORRETORES, "âncora: uma das populações está vazia"
    assert sorted(opcoes - set(NOMES_DAS_OPCOES)) == []
    assert sorted(figuras - set(NOMES_DAS_FIGURAS)) == []
    assert sorted(set(CORRETORES) - set(NOMES_DOS_CORRETORES)) == []


def test_a_grade_passa_a_uma_coluna_quando_duas_nao_cabem(qtbot: Any) -> None:
    """A grade da «Qualidade» punha 35 px de rolagem horizontal a 1280x760. O par: larga,
    duas colunas; estreita, uma — e o mínimo declarado é o de uma coluna, que a deixa encolher."""
    from refino.desktop.janela.componentes import GradeResponsiva

    cartoes = [QLabel("X" * 40) for _ in range(4)]
    grade = GradeResponsiva(cartoes, 2)
    qtbot.addWidget(grade)
    maior = max(c.minimumSizeHint().width() for c in cartoes)
    assert grade.colunas_que_cabem(3 * maior) == 2
    assert grade.colunas_que_cabem(maior + 1) == 1
    assert grade.minimumSizeHint().width() == maior, "o mínimo declarado é de duas colunas"
    grade.resize(maior + 1, 400)
    grade.show()
    qtbot.waitUntil(lambda: grade.colunas == 1, timeout=2000)


def test_a_grade_respeita_o_minimo_explicito_dos_cartoes(qtbot: Any) -> None:
    """Os anexos não podem manter três colunas à custa de sobrepor seus cartões."""
    from refino.desktop.janela.componentes import GradeResponsiva

    cartoes = [QLabel("Arquivo") for _ in range(3)]
    for cartao in cartoes:
        cartao.setMinimumWidth(220)
    grade = GradeResponsiva(cartoes, 3, espaco=12)
    qtbot.addWidget(grade)
    assert grade.colunas_que_cabem(684) == 3
    assert grade.colunas_que_cabem(654) == 2
    assert grade.minimumSizeHint().width() == 220
    grade.resize(654, 300)
    grade.show()
    qtbot.waitUntil(lambda: grade.colunas == 2)
    qtbot.waitUntil(lambda: all(grade.rect().contains(cartao.geometry()) for cartao in cartoes))


def test_o_corpo_do_texto_tem_o_tamanho_do_tema(qtbot: Any) -> None:
    """O corpo passa a 10 pt. O estado do app é restaurado no fim, para não vazar."""
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    antes = app.font()
    try:
        tema.veste_a_aplicacao(app)
        assert app.font().pointSizeF() == tema.TAMANHO_DA_FONTE_DO_CORPO
    finally:
        app.setFont(antes)


# ---------------------------------------------------------------------------
# Os seletores
# ---------------------------------------------------------------------------


def test_os_seletores_da_janela_ficam_prontos_para_a_lista_arredondada(qtbot: Any) -> None:
    """A lista aberta de cada seletor fica sem moldura e com
    fundo transparente (é o que deixa os cantos da folha aparecerem), e com um delegado que lê as
    regras `::item`. O par: um seletor fora da janela vestida não tem nada disto, e o seletor de
    nomes legíveis continua com o SEU delegado."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QComboBox, QStyledItemDelegate

    from refino.desktop.janela.componentes import ComboComNomes

    solto = QComboBox()
    qtbot.addWidget(solto)
    assert not solto.view().window().testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    janela, _barra = _vestida(qtbot)
    seletores = janela.findChildren(QComboBox)
    assert len(seletores) > 5, "âncora: a janela vestida quase não tem seletores"
    for seletor in seletores:
        lista = seletor.view().window()
        assert lista.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground), seletor
        assert lista.windowFlags() & Qt.WindowType.FramelessWindowHint, seletor
        assert isinstance(seletor.itemDelegate(), QStyledItemDelegate), seletor
        assert seletor.maxVisibleItems() == tema.ITENS_A_VISTA_NO_SELETOR
    com_nomes = [s for s in seletores if isinstance(s, ComboComNomes)]
    assert com_nomes, "âncora: nenhum seletor de nomes legíveis na janela"
    for seletor in com_nomes:
        chave = next(iter(seletor.nomes))
        assert (
            seletor.itemDelegate().displayText(chave, seletor.locale()) == seletor.nomes[chave]
        ), "o delegado dos nomes legíveis foi trocado pelo genérico"


def test_o_texto_de_cada_segmento_do_modo_e_o_valor_que_ele_envia(
    qtbot: Any, tmp_path: Path
) -> None:
    """Rótulo e valor não divergem: mostrava-se `pos_rp` e enviava-se `pos_c18`."""
    from refino.desktop.janela.arquivos import PainelDeArquivos

    arquivo = tmp_path / "arquivo.csv"
    arquivo.write_bytes(b"arquivo escolhido")
    painel = PainelDeArquivos(lambda _titulo: arquivo)
    qtbot.addWidget(painel)
    painel.escolhe_medidas()
    painel.escolhe_lista()
    assert painel.segmentos_do_modo, "âncora: o seletor segmentado não tem segmento nenhum"
    for nome, botao in painel.segmentos_do_modo.items():
        botao.click()
        assert botao.text() == nome == painel.campos_para_envio()["mode"], (nome, botao.text())


def _luminancia(cor: str) -> float:
    """A luminância relativa da WCAG 2.1, para o contraste entre duas cores."""
    canais = [int(cor.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lineares = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in canais]
    return 0.2126 * lineares[0] + 0.7152 * lineares[1] + 0.0722 * lineares[2]


def _contraste(a: str, b: str) -> float:
    claro, escuro = sorted((_luminancia(a), _luminancia(b)), reverse=True)
    return (claro + 0.05) / (escuro + 0.05)


def test_a_barra_de_progresso_cheia_mantem_o_percentual_legivel_no_escuro() -> None:
    """O preenchimento era o degradê quase branco dos botões, sob o texto claro."""
    from refino.desktop.janela.tema import ESCURO, folha_de_estilo

    folha = folha_de_estilo("escuro")
    bloco = folha[folha.index("QProgressBar::chunk") :].split("}", 1)[0]
    assert ESCURO.progresso in bloco, "o preenchimento da barra não usa o token próprio"
    assert ESCURO.gradiente_a not in bloco, "o preenchimento voltou ao degradê dos botões"
    assert _contraste(ESCURO.texto, ESCURO.progresso) >= 4.5, "o percentual some sobre a barra"
    # A planta: o valor de antes reprova o mesmo critério, então o teste mede alguma coisa.
    assert _contraste(ESCURO.texto, ESCURO.gradiente_a) < 4.5


def test_a_acao_de_cada_etapa_fica_na_secao_dela_e_nao_num_rodape(qtbot: Any) -> None:
    """A ação mora na seção do meio da página, dentro da área que rola, e não sobra nada fora.

    Era `test_a_acao_de_cada_etapa_fica_fora_da_area_que_rola`, e afirmava de propósito o
    desenho anterior — o formulário rola e a ação não, porque «1. Corrigir» ficava abaixo da
    dobra. O desenho foi revertido: o botão não fica fixo na parte de baixo. O
    custo da dobra vai medido em `test_a_acao_de_cada_etapa_rola_com_a_pagina_entre_os_parametros_e
    _o_resultado`; aqui fica a estrutura — o formulário É a rolagem, e o botão está nela.

    A seção do botão é o bloco 2, «Execução da etapa», e não o bloco 1, que leva os parâmetros.
    """
    from refino.desktop.janela.principal import ABAS

    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    etapas = {aba.etapa: i for i, aba in enumerate(ABAS) if aba.etapa is not None}
    assert set(etapas) == {1, 2, 3}, "âncora: as três abas de etapa"
    for etapa, indice in etapas.items():
        pagina = janela.abas.widget(indice)
        botao = janela.pipeline.botoes_de_etapa[etapa]
        assert pagina.isAncestorOf(botao), f"o botão da etapa {etapa} saiu da página"
        assert pagina.formulario is pagina.rolagem, f"etapa {etapa}: há algo fora da rolagem"
        assert pagina.rolagem.isAncestorOf(botao), f"o botão da etapa {etapa} não rola"
        assert pagina.secao_da_etapa.isAncestorOf(botao), (
            f"o botão da etapa {etapa} saiu do bloco «{pagina.secao_da_etapa.titulo.text()}»"
        )
        assert not pagina.secao_da_entrada.isAncestorOf(botao), (
            f"o botão da etapa {etapa} foi para o bloco dos parâmetros"
        )
        assert not hasattr(pagina, "rodape"), f"a etapa {etapa} voltou a ter rodapé"


# ---------------------------------------------------------------------------
# O esquema claro legível, e a troca de esquema na barra lateral
# ---------------------------------------------------------------------------
#
# A corrida da interface do colega achou o claro ilegível — «Preparar o estudo» e o item marcado
# da navegação — e sem controle na tela. A causa era a folha: as regras da referência do colega
# escreviam os cinzentos do escuro por extenso, e uma cor por extenso vale para os dois esquemas.
# Os portões abaixo são três: nenhuma cor da folha fora dos tokens; todo texto que as regras da
# folha derivam com contraste AA nos DOIS esquemas, com a razão de contraste da WCAG 2.1; e todo
# ícone desenhado por `poe_icone`, na cor de um token, que a troca de esquema repinta. **O ícone
# que não passa por `poe_icone` não está sob o portão de contraste**, e dois portões procuram-no:
# um na janela vestida, outro no código.

_COR_POR_EXTENSO = re.compile(r"#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)")
_REGRA = re.compile(r"([^{}]+?)\{\{(.*?)\}\}", re.S)
_TOKEN = re.compile(r"\{(\w+)\}")
_COMENTARIO = re.compile(r"/\*.*?\*/", re.S)
# Os dois mínimos da WCAG 2.1: texto corrido (AA), e ícone ou componente gráfico (1.4.11).
_MINIMO_DO_TEXTO = 4.5
_MINIMO_DO_GRAFICO = 3.0


def _cores_por_extenso(modelo: str) -> list[str]:
    """As cores que o modelo da folha escreve por extenso, fora dos tokens."""
    return _COR_POR_EXTENSO.findall(_COMENTARIO.sub("", modelo))


def test_a_folha_so_pinta_com_tokens() -> None:
    """A causa do claro ilegível, e o portão que a impede de voltar.

    Uma cor por extenso vale para os dois esquemas: ela é a do escuro, e o claro a herda. O par: a
    regra antiga — o branco do item marcado da navegação — plantada de volta é acusada.
    """
    assert "{fundo}" in tema._MODELO, "âncora: o modelo não tem tokens, e a busca nada mediria"
    assert _cores_por_extenso(tema._MODELO) == []
    planta = "QPushButton#nav:checked {{ background: {destaque_fundo}; color: #ffffff; }}"
    assert _cores_por_extenso(tema._MODELO + planta) == ["#ffffff"]


def _itens_de_cor(valor: str) -> list[str]:
    """Os tokens de um valor de cor da folha — os dois pontos, num degradê —, ou a cor escrita."""
    return _TOKEN.findall(valor) or _COR_POR_EXTENSO.findall(valor)


def _hex(item: str, cores: tema.Cores) -> str:
    """Um token, ou uma cor por extenso, em hexadecimal no esquema dado.

    **A cor translúcida é composta sobre a barra lateral**, e a razão é a folha: a única que ela tem
    é o «passar o rato» da navegação, que pinta sobre a barra.
    """
    valor = str(asdict(cores).get(item, item))
    if not valor.startswith("rgba("):
        return valor
    *canais, opacidade = (float(x) for x in valor[5:-1].split(","))
    base = (int(cores.topo[i : i + 2], 16) for i in (1, 3, 5))
    compostos = (
        round(opacidade * canal + (1 - opacidade) * baixo)
        for canal, baixo in zip(canais, base, strict=True)
    )
    return "#" + "".join(f"{c:02x}" for c in compostos)


# Os fundos que não levam texto por cima: a caixa da seleção, cujo visto é um SVG de `recursos/`
# medido à parte, e o ponto de estado, um quadrado de 6x6 px sem texto. São componente gráfico.
_SEM_TEXTO = ("::indicator", "QLabel#ponto_do_estado")
# O pseudo-estado de um seletor — `:checked`, `:hover`, `:!checked` —, e não o subcontrole `::`.
_PSEUDO_ESTADO = re.compile(r"(?<!:):(?!:)[\w!-]+")


def _declaracoes(corpo: str) -> dict[str, str]:
    declaracoes = {}
    for linha in corpo.split(";"):
        nome, _, valor = linha.partition(":")
        if valor:
            declaracoes[nome.strip()] = valor.strip()
    return declaracoes


def _regras(modelo: str) -> list[tuple[str, dict[str, str]]]:
    """Cada regra da folha, sem os comentários: o seletor, de espaço normalizado, e o corpo."""
    return [
        (" ".join(seletor.split()), _declaracoes(corpo))
        for seletor, corpo in _REGRA.findall(_COMENTARIO.sub("", modelo))
    ]


def _fundos(declaracoes: dict[str, str]) -> list[str]:
    return [
        item
        for propriedade in ("background", "background-color")
        if propriedade in declaracoes
        for item in _itens_de_cor(declaracoes[propriedade])
    ]


def _pares_de_texto(modelo: str) -> list[tuple[str, str, str]]:
    """`(seletor, texto, fundo)` de cada texto que a folha pinta, DERIVADOS das regras dela.

    A regra que dá cor e fundo forma o par. A que dá só a cor — ou o fundo transparente — pinta
    texto sobre o que estiver por baixo, e é conferida contra toda superfície neutra do tema. A
    seleção de texto forma o seu par. `:disabled` e o token `desabilitado` ficam fora: a WCAG isenta
    o texto desabilitado.

    E a regra que dá só o FUNDO também forma par, e é a forma de «Preparar o estudo». O
    cartão da referência tinha fundo e nenhuma cor, e o título por cima herdava o `texto` da paleta:
    em `4a448982`, `#19263d` sobre `#0f0f0f`, 1,27:1 — e a primeira versão deste portão não derivava
    par nenhum daquela regra. O texto sobre um fundo sem cor é, por ordem: a cor que outra regra
    dá ao MESMO seletor, que é a cascata do Qt entre regras (o item marcado da navegação tinha o
    branco numa regra e o fundo noutra); a do seletor sem o pseudo-estado, porque `:pressed` mantém
    o texto do botão; e senão o `texto`, que é a cor que a paleta dá a todo widget sem cor própria.
    As partes de `_SEM_TEXTO` ficam fora: não há texto sobre elas.
    """
    regras = _regras(modelo)
    cor_do_seletor: dict[str, list[str]] = {}
    for seletor, declaracoes in regras:
        if "color" in declaracoes:
            for parte in seletor.split(","):
                cor_do_seletor.setdefault(parte.strip(), []).extend(
                    _itens_de_cor(declaracoes["color"])
                )
    pares: list[tuple[str, str, str]] = []
    for seletor, declaracoes in regras:
        if ":disabled" in seletor:
            continue
        fundos_sem_cor = [] if "color" in declaracoes else _fundos(declaracoes)
        for parte in (p.strip() for p in seletor.split(",") if fundos_sem_cor):
            if any(marca in parte for marca in _SEM_TEXTO):
                continue
            sem_estado = " ".join(_PSEUDO_ESTADO.sub("", parte).split())
            herdadas = cor_do_seletor.get(parte) or cor_do_seletor.get(sem_estado) or ["texto"]
            pares.extend(
                (parte, texto, fundo)
                for texto in herdadas
                if texto != "desabilitado"
                for fundo in fundos_sem_cor
            )
        for frente, fundos in (
            ("color", ("background", "background-color")),
            ("selection-color", ("selection-background-color",)),
        ):
            if frente not in declaracoes:
                continue
            textos = [t for t in _itens_de_cor(declaracoes[frente]) if t != "desabilitado"]
            por_baixo = [
                i for f in fundos if f in declaracoes for i in _itens_de_cor(declaracoes[f])
            ]
            pares.extend(
                (seletor, texto, fundo)
                for texto in textos
                for fundo in por_baixo or tema.SUPERFICIES_NEUTRAS
            )
    return pares


def _cores_do(esquema: str) -> tema.Cores:
    return {"claro": tema.CLARO, "escuro": tema.ESCURO}[esquema]


@pytest.mark.parametrize("esquema", ["claro", "escuro"])
def test_todo_texto_da_folha_tem_contraste_aa_nos_dois_esquemas(esquema: str) -> None:
    """A razão de contraste da WCAG sobre todo texto que `_pares_de_texto`
    deriva das regras da folha. Não é a cascata do Qt: a cor que uma regra dá a um filho é
    medida contra as superfícies neutras, e não contra o fundo que o pai dele tem."""
    cores = _cores_do(esquema)
    pares = _pares_de_texto(tema._MODELO)
    assert len(pares) > 300, f"âncora: só {len(pares)} pares derivados da folha"
    baixos = sorted(
        {
            (seletor, texto, fundo, round(razao, 2))
            for seletor, texto, fundo in pares
            if (razao := _contraste(_hex(texto, cores), _hex(fundo, cores))) < _MINIMO_DO_TEXTO
        }
    )
    assert baixos == [], baixos


def test_o_portao_do_contraste_acusa_os_dois_defeitos_de_d634() -> None:
    """O par, com os dois defeitos do claro, na forma em que estavam na
    folha de `4a448982` — copiados dela, não reescritos.

    - **«Preparar o estudo»**: o cartão da referência com fundo e sem cor, e o título por cima a
      herdar o `texto` da paleta — `#19263d` sobre `#0f0f0f`;
    - **o item marcado da navegação**: o branco numa regra de `QPushButton#nav:checked` e o fundo
      `destaque_fundo` noutra do mesmo seletor, mais abaixo.

    As duas são regras que dão fundo sem dar cor, e a primeira versão deste portão não derivava par
    nenhum delas. **E a terceira planta é a primeira escrita com um token**, que o portão das cores
    por extenso não vê: o fundo escuro do claro no cartão, e o título a 1,00:1.
    """
    preparar = (
        'QFrame#secao[referencia="true"] {{\n'
        "    background: #0f0f0f; border-color: #262626; border-radius: 6px;\n}}\n"
        'QFrame#secao[referencia="true"] QLabel#titulo_da_secao {{ font-size: 16px; }}'
    )
    marcado = (
        "QPushButton#nav:checked {{\n    background: #242427;\n    color: #ffffff;\n}}\n"
        "QPushButton#nav:checked {{ background: {destaque_fundo}; border-color: {borda}; }}"
    )
    com_token = 'QFrame#secao[referencia="true"] {{ background: {invertido}; }}'
    for planta, par in (
        (preparar, ("texto", "#0f0f0f")),
        (marcado, ("#ffffff", "destaque_fundo")),
        (com_token, ("texto", "invertido")),
    ):
        derivados = {(t, f) for _s, t, f in _pares_de_texto(planta)}
        assert par in derivados, (planta, derivados)
        assert _contraste(_hex(par[0], tema.CLARO), _hex(par[1], tema.CLARO)) < _MINIMO_DO_TEXTO
    assert _cores_por_extenso(com_token) == [], "a planta com token escapa ao portão dos tokens"
    # O par: na folha desta decisão, o cartão da referência dá o `texto` sobre o `cartao`.
    assert ('QFrame#secao[referencia="true"]', "texto", "cartao") in _pares_de_texto(tema._MODELO)


@pytest.mark.parametrize("esquema", ["claro", "escuro"])
def test_cada_cor_de_icone_tem_contraste_com_a_superficie_dele(esquema: str) -> None:
    """Ícone é componente gráfico: 3:1 contra a superfície em que é desenhado (WCAG 1.4.11)."""
    cores = _cores_do(esquema)
    assert len(tema.SUPERFICIE_DOS_ICONES) >= 10, "âncora: quase nenhuma cor de ícone declarada"
    baixos = [
        (token, fundo, round(razao, 2))
        for token, fundos in tema.SUPERFICIE_DOS_ICONES.items()
        for fundo in fundos
        if (razao := _contraste(_hex(token, cores), _hex(fundo, cores))) < _MINIMO_DO_GRAFICO
    ]
    assert baixos == [], baixos
    # A planta: o cinzento fixo do ícone de envio, escolhido para o escuro, sobre o cartão claro.
    assert _contraste("#aaaaaa", tema.CLARO.cartao) < _MINIMO_DO_GRAFICO


@pytest.mark.parametrize("esquema", ["claro", "escuro"])
def test_a_barra_de_progresso_mantem_o_percentual_e_o_trilho_nos_dois_esquemas(
    esquema: str,
) -> None:
    """A régua da WCAG nos DOIS esquemas: o preenchimento contra o texto que passa por cima dele
    (4,5:1) e contra o trilho em volta (3:1, componente gráfico)."""
    cores = _cores_do(esquema)
    assert _contraste(cores.texto, cores.progresso) >= _MINIMO_DO_TEXTO
    assert _contraste(cores.campo, cores.progresso) >= _MINIMO_DO_GRAFICO
    # A planta: o preenchimento que o claro tinha antes desta decisão some no trilho.
    assert _contraste(tema.CLARO.campo, "#9dbcf5") < _MINIMO_DO_GRAFICO


# ---------------------------------------------------------------------------
# O estado das etapas na barra lateral
# ---------------------------------------------------------------------------

# Os fundos sobre os quais o ponto de uma etapa é desenhado: o botão da navegação é transparente
# sobre a barra, translúcido sob o rato e marcado com o destaque diluído.
_FUNDOS_DO_PONTO_NA_BARRA = ("topo", "nav_hover", "destaque_fundo")


# ---------------------------------------------------------------------------
# A gaveta 'Trabalhos futuros'
# ---------------------------------------------------------------------------


def _indices_beta(janela: JanelaPrincipal) -> list[int]:
    return [i for i in range(janela.abas.count()) if janela.abas.tabText(i).endswith(" (beta)")]


def test_as_abas_beta_ficam_juntas_logo_antes_da_ultima(qtbot: Any) -> None:
    """As duas abas beta são as penúltimas, e 'Métodos e limites' continua a última."""
    janela, _ = _vestida(qtbot)
    beta = _indices_beta(janela)
    ultima = janela.abas.count() - 1
    assert len(beta) == 2, beta
    assert beta == [ultima - 2, ultima - 1]
    assert janela.abas.tabText(beta[1]).startswith("Vias metabólicas")


def test_a_gaveta_abre_fechada_e_o_titulo_a_alterna(qtbot: Any) -> None:
    janela, barra = _vestida(qtbot)
    janela.show()
    beta = _indices_beta(janela)
    assert barra.cabecalho_da_gaveta.text() == "Trabalhos futuros"
    assert barra.cabecalho_da_gaveta.findChild(QLabel, "beta_nav") is not None
    assert all(not barra.botoes[i].isVisible() for i in beta), "a gaveta abriu aberta"
    barra.cabecalho_da_gaveta.click()
    assert all(barra.botoes[i].isVisible() for i in beta)
    barra.cabecalho_da_gaveta.click()
    assert all(not barra.botoes[i].isVisible() for i in beta)


def test_nenhum_texto_da_janela_vestida_diz_multiverso(qtbot: Any) -> None:
    janela, _barra = _vestida(qtbot)
    janela.show()
    for i in range(janela.abas.count()):
        janela.abas.setCurrentIndex(i)
    textos = [janela.abas.tabText(i) for i in range(janela.abas.count())]
    for w in janela.findChildren(QWidget):
        if isinstance(w, QLabel | QAbstractButton):
            textos.append(w.text())
        textos.append(w.toolTip())
    inspecionados = [t for t in textos if t]
    assert len(inspecionados) > 50, "âncora: a varredura quase não leu texto"
    # Os dois nomes da literatura ficam: o título de um artigo publicado (STEEGEN et al.) e o
    # nome do MÉTODO na lista de referências, para a referência ainda fazer sentido.
    da_literatura = (
        "Increasing transparency through a multiverse analysis",
        "Teste de robustez (análise de multiverso)",
    )
    sobra = []
    for t in inspecionados:
        t = " ".join(t.split())
        for nome in da_literatura:
            t = t.replace(nome, "")
        sobra.append(t)
    assert [t for t in sobra if "ultivers" in t.lower()] == []


def test_a_gaveta_fica_entre_graficos_e_metodos_e_limites(qtbot: Any) -> None:
    janela, barra = _vestida(qtbot)
    coluna = barra.layout()
    assert coluna is not None
    posicoes = [
        coluna.indexOf(w)
        for w in (
            barra.botoes[janela.abas.count() - 4],
            barra.cabecalho_da_gaveta,
            barra.botoes[janela.abas.count() - 1],
        )
    ]
    assert -1 not in posicoes and posicoes == sorted(posicoes), posicoes


def test_sem_aba_beta_o_titulo_da_gaveta_nao_aparece(qtbot: Any) -> None:
    barra = tema.BarraLateral(["Gráficos", "Métodos e limites"], "escuro", "Refino", "Refino")
    qtbot.addWidget(barra)
    barra.show()
    assert not barra.cabecalho_da_gaveta.isVisible()
    assert not barra.gaveta.isVisible()


def test_a_aba_da_gaveta_escolhida_por_codigo_abre_a_gaveta(qtbot: Any) -> None:
    janela, barra = _vestida(qtbot)
    janela.show()
    vias = _indices_beta(janela)[1]
    janela.abas.setCurrentIndex(vias)
    assert barra.botoes[vias].isChecked()
    assert barra.botoes[vias].isVisible(), "a aba ativa ficou com o botão escondido"
    assert barra.cabecalho_da_gaveta.isChecked()


def test_a_etiqueta_beta_e_dourada_e_legivel_nos_dois_esquemas() -> None:
    from refino.desktop.janela.tema import CLARO, ESCURO

    for cores in (ESCURO, CLARO):
        assert cores.beta != cores.texto_suave
        assert _contraste(cores.beta, cores.beta_fundo) >= 4.5, cores.beta
        folha = tema.folha_de_estilo("escuro" if cores is ESCURO else "claro")
        regra = folha[folha.index("QLabel#beta_nav") :].split("}", 1)[0]
        assert cores.beta in regra, regra


def _barra_solta(esquema: str, qtbot: Any) -> tema.BarraLateral:
    """A barra com as abas da janela, sem a janela, e a folha do esquema aplicada a ela."""
    from refino.desktop.janela.principal import ABAS

    titulos = [aba.titulo for aba in ABAS]
    barra = tema.BarraLateral(titulos, esquema, "Refino", "Refino LC-MS")  # type: ignore[arg-type]
    qtbot.addWidget(barra)
    barra.setStyleSheet(tema.folha_de_estilo(esquema))
    return barra


def _token_do_ponto(tipo: str) -> str:
    """O token que a folha dá ao fundo do ponto do `tipo`, DERIVADO das regras dela: a última regra
    do seletor que dá fundo é a que vale, como na cascata entre regras iguais."""
    seletor = f'QLabel#ponto_do_estado[tipo="{tipo}"]'
    fundos = [
        _fundos(declaracoes)
        for nome, declaracoes in _regras(tema._MODELO)
        if nome == seletor and _fundos(declaracoes)
    ]
    assert fundos, f"a folha não dá fundo ao ponto «{tipo}»"
    return fundos[-1][0]


def _cor_do_ponto(ponto: QLabel) -> str:
    """A cor no centro do ponto desenhado — o desenho, e não a propriedade que a folha leria."""
    imagem = ponto.grab().toImage()
    return str(imagem.pixelColor(imagem.width() // 2, imagem.height() // 2).name())


def test_os_estados_do_ponto_sao_os_da_etapa_e_cada_tipo_tem_regra_na_folha() -> None:
    """Bijeção com o vocabulário fechado das etapas: um estado novo sem ponto levantaria na barra
    no primeiro recálculo, e um ponto de estado que deixou de existir é texto morto."""
    from typing import get_args

    from refino.desktop.etapas import EstadoDaEtapa

    assert set(tema.PONTOS_DAS_ETAPAS) == set(get_args(EstadoDaEtapa))
    tipos = [tipo for tipo, _palavra in tema.PONTOS_DAS_ETAPAS.values() if tipo is not None]
    assert len(tipos) == len(set(tipos)) == 3, "dois estados com o mesmo ponto não se distinguem"
    for tipo in tipos:
        _token_do_ponto(tipo)


@pytest.mark.parametrize("esquema", ["claro", "escuro"])
def test_o_ponto_da_etapa_segue_o_estado_na_cor_da_folha_nos_dois_esquemas(
    esquema: str, qtbot: Any
) -> None:
    """O mesmo ponto troca de cor quando o estado da etapa muda, e some quando não há o que dizer.

    **A âncora vem antes:** as três cores distam umas das outras no esquema, senão um ponto que não
    se repintasse passaria pela sequência sem acusar.
    """
    from refino.desktop.janela.principal import INDICES_DAS_ABAS_DE_ETAPA

    cores = _cores_do(esquema)
    esperada = {
        estado: _hex(_token_do_ponto(tipo), cores)
        for estado, (tipo, _palavra) in tema.PONTOS_DAS_ETAPAS.items()
        if tipo is not None
    }
    assert len(set(esperada.values())) == 3, f"âncora: cores repetidas no {esquema}: {esperada}"
    barra = _barra_solta(esquema, qtbot)
    correcao = INDICES_DAS_ABAS_DE_ETAPA[1]
    barra.mostra_os_estados({correcao: "nao_feita"})
    assert correcao not in barra.pontos, "«não feita» criou um ponto"
    for estado in ("em_curso", "em_dia", "desatualizada", "em_dia"):
        barra.mostra_os_estados({correcao: estado})
        ponto = barra.pontos[correcao]
        assert not ponto.isHidden(), estado
        assert _mesma_cor(_cor_do_ponto(ponto), esperada[estado]), (estado, _cor_do_ponto(ponto))
    for sem_ponto in ("nao_feita", None):
        barra.mostra_os_estados({correcao: sem_ponto})
        assert barra.pontos[correcao].isHidden(), f"o ponto ficou à vista com {sem_ponto!r}"


@pytest.mark.parametrize("esquema", ["claro", "escuro"])
def test_o_ponto_da_etapa_tem_contraste_com_a_barra_nos_dois_esquemas(esquema: str) -> None:
    """O ponto é componente gráfico: 3:1 contra cada fundo em que o botão da barra o põe (WCAG
    1.4.11), com o token derivado da folha e não redigitado."""
    cores = _cores_do(esquema)
    baixos = [
        (tipo, fundo, round(razao, 2))
        for tipo, _palavra in tema.PONTOS_DAS_ETAPAS.values()
        if tipo is not None
        for fundo in _FUNDOS_DO_PONTO_NA_BARRA
        if (razao := _contraste(_hex(_token_do_ponto(tipo), cores), _hex(fundo, cores)))
        < _MINIMO_DO_GRAFICO
    ]
    assert baixos == [], baixos
    # A planta: o amarelo do escuro na barra clara.
    assert _contraste(tema.ESCURO.aviso, tema.CLARO.topo) < _MINIMO_DO_GRAFICO


def test_a_dica_da_etapa_diz_o_estado_em_palavras_e_as_outras_abas_nao_mudam(qtbot: Any) -> None:
    from refino.desktop.janela.principal import INDICES_DAS_ABAS_DE_ETAPA

    barra = _barra_solta("escuro", qtbot)
    dicas_antes = [botao.toolTip() for botao in barra.botoes]
    etapas = sorted(INDICES_DAS_ABAS_DE_ETAPA.values())
    barra.mostra_os_estados(dict(zip(etapas, ("em_dia", "em_curso", "desatualizada"), strict=True)))
    assert [barra.botoes[i].toolTip() for i in etapas] == [
        f"{barra.titulos[etapas[0]]} — em dia",
        f"{barra.titulos[etapas[1]]} — em curso",
        f"{barra.titulos[etapas[2]]} — desatualizada",
    ]
    outras = [i for i in range(len(barra.botoes)) if i not in etapas]
    assert outras, "âncora: nenhuma aba fora das etapas"
    assert [barra.botoes[i].toolTip() for i in outras] == [dicas_antes[i] for i in outras]
    sem_ponto = [i for i in outras if barra.botoes[i].findChildren(QLabel, "ponto_do_estado")]
    assert sem_ponto == [], "uma aba que não é etapa ganhou ponto"
    barra.mostra_os_estados({etapas[0]: None})
    assert barra.botoes[etapas[0]].toolTip() == barra.titulos[etapas[0]], "sem par, a dica ficou"


def test_estado_fora_do_vocabulario_levanta_e_a_barra_fica_como_estava(qtbot: Any) -> None:
    from refino.desktop.janela.principal import INDICES_DAS_ABAS_DE_ETAPA

    barra = _barra_solta("escuro", qtbot)
    correcao, normalizacao = INDICES_DAS_ABAS_DE_ETAPA[1], INDICES_DAS_ABAS_DE_ETAPA[2]
    barra.mostra_os_estados({correcao: "em_dia"})
    with pytest.raises(ValueError, match=r"desconhecido: feita\. Os válidos são nao_feita"):
        barra.mostra_os_estados({normalizacao: "em_dia", correcao: "feita"})  # type: ignore[dict-item]
    assert normalizacao not in barra.pontos, (
        "o estado válido foi aplicado antes de o inválido levantar"
    )
    assert barra.pontos[correcao].property("tipo") == "sucesso"


def test_a_barra_da_janela_e_a_que_veste_a_janela_acoplou(qtbot: Any) -> None:
    janela = JanelaPrincipal(_Nada())
    qtbot.addWidget(janela)
    assert tema.BarraLateral.da_janela(janela) is None, "a janela sem tema já tinha barra"
    janela._atualiza_as_etapas()
    barra = tema.veste_a_janela(janela)
    assert tema.BarraLateral.da_janela(janela) is barra


def _cor_do_traco(mapa: QPixmap) -> str:
    """A cor do traço de um ícone: a média dos pixels de opacidade de metade para cima.

    **Não a dos pixels opacos, e foi medido:** o traço de 1,8 de um ícone de 12 px não tem pixel
    opaco nenhum, e o rótulo guarda o desenho reduzido. A média dos quase opacos deu a cor pedida,
    exata, nos 62 ícones gravados da janela vestida no escuro; a cor de um pixel translúcido sai
    arredondada, e por isso a comparação é `_mesma_cor`, a duas unidades por canal.
    """
    imagem = mapa.toImage()
    fortes = [
        imagem.pixelColor(x, y)
        for y in range(imagem.height())
        for x in range(imagem.width())
        if imagem.pixelColor(x, y).alpha() >= 128
    ]
    assert fortes, "ícone sem pixel quase opaco nenhum: o instrumento nada leria"
    media = (
        round(sum(getattr(cor, canal)() for cor in fortes) / len(fortes))
        for canal in ("red", "green", "blue")
    )
    return "#" + "".join(f"{valor:02x}" for valor in media)


def _mesma_cor(a: str, b: str) -> bool:
    """Igual a duas unidades por canal — as cores dos dois esquemas distam bem mais que isso."""
    return all(abs(int(a[i : i + 2], 16) - int(b[i : i + 2], 16)) <= 2 for i in (1, 3, 5))


def _mapa_do_icone(icone: QIcon, estado: QIcon.State = QIcon.State.Off) -> QPixmap:
    """O desenho guardado no ícone, no tamanho em que foi guardado — pedir outro o reescalaria."""
    tamanho = icone.availableSizes(QIcon.Mode.Normal, estado)[0]
    return icone.pixmap(tamanho, QIcon.Mode.Normal, estado)


def _icones_gravados(raiz: QWidget) -> list[tuple[QWidget, str]]:
    """Cada widget com ícone desenhado por `tema.poe_icone`, e o token de cor que ele pediu."""
    achados = []
    for widget in raiz.findChildren(QWidget):
        receita = widget.property(tema.PROPRIEDADE_DO_ICONE)
        if receita:
            achados.append((widget, receita["cor"]))
    return achados


def _cor_desenhada(widget: QWidget) -> str:
    if isinstance(widget, QLabel):
        return _cor_do_traco(widget.pixmap())
    assert isinstance(widget, QAbstractButton), widget
    return _cor_do_traco(_mapa_do_icone(widget.icon()))


def _fora_do_tom(janela: QWidget, cores: tema.Cores) -> list[tuple[str, str]]:
    return [
        (widget.objectName() or type(widget).__name__, token)
        for widget, token in _icones_gravados(janela)
        if not _mesma_cor(_cor_desenhada(widget), asdict(cores)[token])
    ]


@contextmanager
def _aplicativo_restaurado() -> Iterator[QApplication]:
    """A `QApplication` é da sessão: estilo, paleta, folha, fonte e esquema voltam no fim."""
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    estilo, paleta, folha, fonte = app.style().name(), app.palette(), app.styleSheet(), app.font()
    esquema = app.property(tema.PROPRIEDADE_DO_ESQUEMA)
    try:
        yield app
    finally:
        app.setStyleSheet(folha)
        app.setPalette(paleta)
        app.setFont(fonte)
        app.setStyle(estilo)
        app.setProperty(tema.PROPRIEDADE_DO_ESQUEMA, esquema)


def test_cada_cor_de_icone_muda_de_um_esquema_para_o_outro() -> None:
    """A âncora do teste da troca: um token de ícone com o mesmo valor nos dois esquemas faria a
    troca que esquece de repintar passar sem acusar — a planta cairia onde as duas coisas não
    diferem."""
    assert tema.SUPERFICIE_DOS_ICONES, "âncora: com o mapa vazio, a lista vazia nada provaria"
    iguais = [
        token
        for token in tema.SUPERFICIE_DOS_ICONES
        if getattr(tema.CLARO, token) == getattr(tema.ESCURO, token)
    ]
    assert iguais == []


def test_o_controle_da_barra_troca_o_esquema_e_os_icones_nos_dois_sentidos(qtbot: Any) -> None:
    """O claro era inalcançável pela tela. A barra lateral ganha «Escuro | Claro»,
    e a troca alcança a folha, a paleta, os ícones da barra e todo ícone desenhado por `poe_icone`
    — nos dois sentidos, porque o ícone que fica com a cor do esquema de partida é o defeito que
    esta troca pode ter."""
    # O instrumento, sobre um caso de resposta conhecida: o traço sai na cor pedida, e o de
    # outra cor não passa por ela.
    assert _mesma_cor(_cor_do_traco(tema.icone(tema.ICONES["Gráficos"], "#123456")), "#123456")
    assert not _mesma_cor(_cor_do_traco(tema.icone(tema.ICONES["Gráficos"], "#163a5a")), "#123456")
    with _aplicativo_restaurado() as app:
        tema.veste_a_aplicacao(app, "escuro")
        janela, barra = _vestida(qtbot)
        escuro, claro = barra.botoes_do_esquema["escuro"], barra.botoes_do_esquema["claro"]
        assert escuro.isChecked() and not claro.isChecked()
        gravados = _icones_gravados(janela)
        assert len(gravados) >= 15, f"âncora: só {len(gravados)} ícones gravados na janela vestida"
        assert _fora_do_tom(janela, tema.ESCURO) == []
        # A planta: trocar só a folha e a paleta deixa os ícones no tom do escuro, e o portão acusa.
        app.setProperty(tema.PROPRIEDADE_DO_ESQUEMA, "claro")
        app.setPalette(tema.paleta(tema.CLARO))
        app.setStyleSheet(tema.folha_de_estilo("claro"))
        assert len(_fora_do_tom(janela, tema.CLARO)) == len(gravados)
        tema.veste_a_aplicacao(app, "escuro")
        for destino in ("claro", "escuro"):
            cores = _cores_do(destino)
            barra.botoes_do_esquema[destino].click()
            assert tema.esquema_em_vigor() == destino
            assert app.styleSheet() == tema.folha_de_estilo(destino)  # type: ignore[arg-type]
            assert app.palette().color(QPalette.ColorRole.Window).name() == cores.fundo
            assert barra.botoes_do_esquema[destino].isChecked()
            assert janela.abas.currentIndex() == 0, "a troca mudou de página"
            outra = _cor_do_traco(_mapa_do_icone(barra.botoes[1].icon()))
            assert _mesma_cor(outra, cores.topo_texto_suave), (destino, outra)
            marcada = _cor_do_traco(_mapa_do_icone(barra.botoes[0].icon(), QIcon.State.On))
            assert _mesma_cor(marcada, cores.topo_texto), (destino, marcada)
            assert _fora_do_tom(janela, cores) == [], destino


class _DwmQueGuardaOValor:
    """O `dwmapi` de mentira, que guarda o atributo E o valor pedido — o 1 da moldura escura, o 0
    da clara."""

    def __init__(self, respostas: dict[int, int]) -> None:
        self.respostas = respostas
        self.pedidos: list[tuple[int, int]] = []

    def DwmSetWindowAttribute(self, _janela: int, atributo: int, valor: Any, _tamanho: int) -> int:  # noqa: N802
        self.pedidos.append((atributo, valor._obj.value))
        return self.respostas.get(atributo, 1)


def test_a_moldura_pede_o_escuro_e_o_claro(qtbot: Any) -> None:
    """Na troca para o claro, a barra de título do Windows deixa de ser escura: o mesmo
    atributo da moldura escura, com 0 no lugar de 1."""
    janela = QWidget()
    qtbot.addWidget(janela)
    dwm = _DwmQueGuardaOValor({20: 0})
    assert tema.escurece_a_moldura(janela, escura=False, dwm=dwm) is True
    assert tema.escurece_a_moldura(janela, dwm=dwm) is True
    assert dwm.pedidos == [(20, 0), (20, 1)]


def test_a_troca_de_esquema_pede_a_moldura_do_esquema_novo(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A troca pede ao Windows a moldura do esquema de destino, nos dois sentidos."""
    pedidos: list[bool] = []

    def _guarda(_janela: QWidget, *, escura: bool = True, dwm: Any = None) -> bool:
        pedidos.append(escura)
        return False

    monkeypatch.setattr(tema, "escurece_a_moldura", _guarda)
    with _aplicativo_restaurado() as app:
        tema.veste_a_aplicacao(app, "escuro")
        _janela, barra = _vestida(qtbot)
        pedidos.clear()
        barra.botoes_do_esquema["claro"].click()
        barra.botoes_do_esquema["escuro"].click()
    assert pedidos == [False, True]


def test_poe_icone_recusa_a_cor_sem_superficie_declarada(qtbot: Any) -> None:
    """A cor do ícone é um token com superfície declarada, conferido em runtime: um nome
    errado — ou uma cor por extenso, a forma do defeito do claro — levanta em vez de desenhar."""
    rotulo = QLabel()
    qtbot.addWidget(rotulo)
    for cor in ("#aaaaaa", "texto_suav"):
        with pytest.raises(ValueError, match="cor de ícone"):
            tema.poe_icone(rotulo, tema.ICONES["Gráficos"], cor, 14)
    assert rotulo.property(tema.PROPRIEDADE_DO_ICONE) is None
    tema.poe_icone(rotulo, tema.ICONES["Gráficos"], "texto_suave", 14)
    assert rotulo.property(tema.PROPRIEDADE_DO_ICONE)["cor"] == "texto_suave"


def test_esquema_em_vigor_recusa_o_valor_que_nao_e_esquema(qtbot: Any) -> None:
    """A propriedade da aplicação com um valor que não é esquema levanta, e não recua no padrão:
    recuar pintaria os ícones no escuro sobre a folha que estiver aplicada, sem nada
    acusar, num `else` razoável. E `poe_icone`, que lê o esquema, levanta também, antes
    de gravar a receita."""
    rotulo = QLabel()
    qtbot.addWidget(rotulo)
    with _aplicativo_restaurado() as app:
        app.setProperty(tema.PROPRIEDADE_DO_ESQUEMA, "claro")
        assert tema.esquema_em_vigor() == "claro", "âncora: a propriedade é a que se lê"
        app.setProperty(tema.PROPRIEDADE_DO_ESQUEMA, "azul")
        with pytest.raises(ValueError, match="esquema desconhecido"):
            tema.esquema_em_vigor()
        with pytest.raises(ValueError, match="esquema desconhecido"):
            tema.poe_icone(rotulo, tema.ICONES["Gráficos"], "texto", 14)
    assert rotulo.property(tema.PROPRIEDADE_DO_ICONE) is None


def test_a_troca_para_um_esquema_desconhecido_levanta_sem_mudar_nada(qtbot: Any) -> None:
    """A irmã da guarda de `esquema_em_vigor`: `_aplica_cores` procura as cores ANTES de
    gravar o esquema na aplicação, então a troca para um esquema que não existe levanta e deixa a
    janela no esquema em que estava — a folha, a paleta e a propriedade. Na ordem inversa, a
    propriedade ficaria com o valor inválido, e o ícone seguinte levantaria longe da causa."""
    with _aplicativo_restaurado() as app:
        tema.veste_a_aplicacao(app, "escuro")
        janela, _barra = _vestida(qtbot)
        folha = app.styleSheet()
        assert folha == tema.folha_de_estilo("escuro"), "âncora: a folha é a do escuro"
        with pytest.raises(KeyError):
            tema.troca_o_esquema(janela, "sepia")  # type: ignore[arg-type]
        assert tema.esquema_em_vigor() == "escuro"
        assert app.styleSheet() == folha
        assert app.palette().color(QPalette.ColorRole.Window).name() == tema.ESCURO.fundo


def test_clicar_no_esquema_ja_marcado_nao_refaz_a_troca(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O grupo exclusivo emite `idClicked` também no botão que já está marcado, e a troca refaz a
    folha, repole a janela inteira e desenha os ícones de novo — a janela para pelo tempo de uma
    troca, e nada muda. O espião conta as trocas que o clique pede; a primeira, que muda
    de esquema, é a âncora de que ele as vê."""
    pedidas: list[str] = []
    troca = tema.troca_o_esquema

    def _espia(janela: QWidget, esquema: Any) -> int:
        pedidas.append(esquema)
        return troca(janela, esquema)

    monkeypatch.setattr(tema, "troca_o_esquema", _espia)
    with _aplicativo_restaurado() as app:
        tema.veste_a_aplicacao(app, "escuro")
        _janela, barra = _vestida(qtbot)
        barra.botoes_do_esquema["claro"].click()
        assert pedidas == ["claro"], "âncora: o espião vê a troca que o clique pede"
        barra.botoes_do_esquema["claro"].click()
        assert pedidas == ["claro"], "o clique no esquema marcado refez a troca"
        barra.botoes_do_esquema["escuro"].click()
        barra.botoes_do_esquema["escuro"].click()
        assert pedidas == ["claro", "escuro"]


def _do_estilo(widget: QWidget, desenho: QImage) -> bool:
    """O desenho é um dos ícones-padrão do estilo (`QStyle.standardIcon`), no tamanho dele."""
    estilo = widget.style()
    return any(
        estilo.standardIcon(padrao).pixmap(desenho.size()).toImage() == desenho
        for padrao in QStyle.StandardPixmap
    )


def _desenhos_da_janela(janela: QWidget) -> dict[str, list[str]]:
    """Cada widget com desenho — pixmap no rótulo, ícone no botão —, pela origem do desenho.

    `receita` é o que `poe_icone` desenhou e a troca repinta; `barra`, o que `BarraLateral.pinta`
    desenha; `estilo`, o ícone-padrão do estilo, que não é desenhado numa cor do esquema; e
    `sem_origem` o resto, que é a forma do defeito do claro — um desenho de cor fixa, que nenhuma
    troca refaz e nenhum portão de contraste vê.
    """
    barra = janela.findChild(tema.BarraLateral)
    origens: dict[str, list[str]] = {"receita": [], "barra": [], "estilo": [], "sem_origem": []}
    for widget in janela.findChildren(QWidget):
        if isinstance(widget, QLabel):
            mapa = widget.pixmap()
            if mapa is None or mapa.isNull():
                continue
            desenho = mapa.toImage()
        elif isinstance(widget, QAbstractButton) and not widget.icon().isNull():
            desenho = _mapa_do_icone(widget.icon()).toImage()
        else:
            continue
        nome = widget.objectName() or type(widget).__name__
        if widget.property(tema.PROPRIEDADE_DO_ICONE):
            origens["receita"].append(nome)
        elif barra is not None and barra.isAncestorOf(widget):
            origens["barra"].append(nome)
        elif _do_estilo(widget, desenho):
            origens["estilo"].append(nome)
        else:
            origens["sem_origem"].append(nome)
    return origens


def test_todo_desenho_da_janela_vestida_tem_origem_conhecida(qtbot: Any) -> None:
    """O portão de contraste dos ícones só alcança o que passa por `poe_icone`. Este
    alcança a janela vestida inteira: todo desenho dela vem de `poe_icone`, da barra lateral ou do
    estilo. O par é o defeito do claro plantado de volta num widget da janela: o ícone de
    envio desenhado com o `#aaaaaa` fixo, sem receita."""
    with _aplicativo_restaurado() as app:
        tema.veste_a_aplicacao(app, "escuro")
        janela, barra = _vestida(qtbot)
        origens = _desenhos_da_janela(janela)
        assert len(origens["receita"]) >= 15, f"âncora: só {origens['receita']} com receita"
        assert len(origens["barra"]) == len(barra.botoes) + 2, (
            "âncora: o frasco, a seta da gaveta e a navegação"
        )
        assert origens["estilo"], "âncora: nenhum ícone do estilo visto, e o ramo nada mediria"
        assert origens["sem_origem"] == []
        envio = janela.findChild(QPushButton, "seleciona_arquivo")
        assert envio is not None and envio.property(tema.PROPRIEDADE_DO_ICONE)
        envio.setIcon(QIcon(tema.icone_nitido(tema.ICONES["Gráficos"], "#aaaaaa", 14)))
        envio.setProperty(tema.PROPRIEDADE_DO_ICONE, None)
        assert _desenhos_da_janela(janela)["sem_origem"] == ["seleciona_arquivo"]


# Quem desenha um ícone na cor que recebe, e quem pode chamá-los: `poe_icone`, que dá a cor de um
# token, e a barra, que desenha os seus. Fora daqui, a cor é fixa — a forma do defeito do claro.
_DESENHAM_NUMA_COR = frozenset({"icone", "icone_nitido"})
_PODEM_DESENHAR = frozenset({"icone_nitido", "poe_icone", "BarraLateral.pinta"})


def _desenhos_fora_de_poe_icone(fonte: str, *, e_o_tema: bool) -> tuple[list[str], list[str]]:
    """`(permitidas, acusadas)`: as chamadas a `icone` e `icone_nitido` no código, por função.

    Pelo NOME da função chamada, com ou sem o módulo na frente (`tema.icone`). No `tema.py` são
    permitidas as de `_PODEM_DESENHAR`; noutro módulo, nenhuma.
    """
    permitidas: list[str] = []
    acusadas: list[str] = []

    def _visita(no: ast.AST, dono: str) -> None:
        for filho in ast.iter_child_nodes(no):
            if isinstance(filho, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                _visita(filho, f"{dono}.{filho.name}" if dono else filho.name)
                continue
            if isinstance(filho, ast.Call):
                chamada = filho.func
                nome = (
                    chamada.id
                    if isinstance(chamada, ast.Name)
                    else chamada.attr
                    if isinstance(chamada, ast.Attribute)
                    else None
                )
                if nome in _DESENHAM_NUMA_COR:
                    lugar = f"{dono or '<módulo>'}:{filho.lineno}"
                    (permitidas if e_o_tema and dono in _PODEM_DESENHAR else acusadas).append(lugar)
            _visita(filho, dono)

    _visita(ast.parse(fonte), "")
    return permitidas, acusadas


def test_nenhum_icone_e_desenhado_numa_cor_fora_de_poe_icone() -> None:
    """O portão do código, que alcança o que a janela vestida não mostra — os widgets que nascem
    quando há dado: `icone` e `icone_nitido` recebem a cor, e só `poe_icone` e a barra
    lateral os chamam. O par é o ícone do aviso desenhado com o `#aeb9c9` fixo, que no
    claro dava 1,85:1."""
    pasta = Path(tema.__file__).parent.parent
    permitidas: list[str] = []
    acusadas: list[str] = []
    arquivos = sorted(pasta.rglob("*.py"))
    assert len(arquivos) > 10, f"âncora: só {len(arquivos)} módulos em {pasta}"
    for arquivo in arquivos:
        sim, nao = _desenhos_fora_de_poe_icone(
            arquivo.read_text(encoding="utf-8"), e_o_tema=arquivo.name == "tema.py"
        )
        permitidas += [f"{arquivo.name}:{lugar}" for lugar in sim]
        acusadas += [f"{arquivo.name}:{lugar}" for lugar in nao]
    assert len(permitidas) >= 5, f"âncora: o percurso viu só {permitidas}"
    assert acusadas == []
    planta = (
        "class AvisoDoEstudo:\n"
        "    def __init__(self):\n"
        '        self.icone.setPixmap(icone_nitido(_TRACO_DO_AVISO, "#aeb9c9", 18))\n'
    )
    assert _desenhos_fora_de_poe_icone(planta, e_o_tema=False) == ([], ["AvisoDoEstudo.__init__:3"])
    assert _desenhos_fora_de_poe_icone(planta, e_o_tema=True)[1] == ["AvisoDoEstudo.__init__:3"]


def _fundos_dos_icones(janela: QWidget, modelo: str) -> list[tuple[str, str, str]]:
    """`(widget, cor do ícone, fundo)`: cada fundo que uma regra da folha dá, pelo NOME, a um widget
    com ícone gravado — o último seletor composto da regra nomeia o widget (`#remove_arquivo`).
    `:disabled` fica fora: o Qt desenha o ícone desabilitado a partir do normal, noutro tom."""
    cores_por_nome: dict[str, set[str]] = {}
    for widget, cor in _icones_gravados(janela):
        if widget.objectName():
            cores_por_nome.setdefault(widget.objectName(), set()).add(cor)
    achados = []
    for seletor, declaracoes in _regras(modelo):
        if ":disabled" in seletor:
            continue
        for parte in seletor.split(","):
            nome = re.search(r"#(\w+)", parte.split()[-1]) if parte.split() else None
            if nome is not None and nome.group(1) in cores_por_nome:
                achados.extend(
                    (nome.group(1), cor, fundo)
                    for cor in sorted(cores_por_nome[nome.group(1)])
                    for fundo in _fundos(declaracoes)
                )
    return achados


def test_o_mapa_dos_icones_tem_os_fundos_que_a_folha_da_aos_widgets_deles(qtbot: Any) -> None:
    """`SUPERFICIE_DOS_ICONES` é declarado, e o contraste dos ícones é medido contra ele.
    Este portão confere o mapa contra a folha: todo fundo que uma regra dá, pelo nome, a um widget
    com ícone gravado está entre as superfícies da cor desse ícone. O par é uma regra
    que dá ao ícone do aviso um fundo fora do mapa."""
    with _aplicativo_restaurado() as app:
        tema.veste_a_aplicacao(app, "escuro")
        janela, _barra = _vestida(qtbot)
        fundos = _fundos_dos_icones(janela, tema._MODELO)
        assert ("remove_arquivo", "acao_suave", "perigo_fundo") in fundos, (
            f"âncora: o fundo de «Remover» sob o rato não foi derivado de {fundos}"
        )
        fora = sorted({f for f in fundos if f[2] not in tema.SUPERFICIE_DOS_ICONES[f[1]]})
        assert fora == []
        planta = "QLabel#icone_do_aviso {{ background: {destaque}; }}"
        com_planta = _fundos_dos_icones(janela, tema._MODELO + planta)
        assert [f for f in com_planta if f[2] not in tema.SUPERFICIE_DOS_ICONES[f[1]]] == [
            ("icone_do_aviso", "nota_info_icone", "destaque")
        ]


# ---------------------------------------------------------------------------
# O aviso fixo na página e o botão desligado com motivo
# ---------------------------------------------------------------------------
#
# Botão desligado traz sempre a explicação, e a tela guia o usuário até um resultado. Avisos
# simples ficam em amarelo e avisos críticos em vermelho.
# A faixa pinta só com os tokens que o portão de contraste já mede nos dois esquemas, e o nível
# também se lê sem cor: pelo prefixo do texto e pelo nome acessível.

_TOKENS_DO_NIVEL = {"aviso": "nota_aviso", "critico": "nota_erro"}


def _faixa(qtbot: Any) -> Any:
    from refino.desktop.janela.componentes import FaixaDeAviso

    faixa = FaixaDeAviso()
    qtbot.addWidget(faixa)
    return faixa


def test_cada_nivel_da_faixa_tem_uma_regra_na_folha_com_os_tokens_dele() -> None:
    """A folha e a faixa falam do mesmo vocabulário: um nível sem regra cairia no estilo da base
    sem nada acusar, e uma regra sem nível seria cor que nunca aparece."""
    from refino.desktop.janela.componentes import PREFIXOS_DOS_NIVEIS

    assert set(PREFIXOS_DOS_NIVEIS) == set(_TOKENS_DO_NIVEL)
    regras = _regras(tema._MODELO)
    for nivel, base in _TOKENS_DO_NIVEL.items():
        seletor = f'QFrame#faixa_de_aviso[nivel="{nivel}"]'
        achadas = [declaracoes for s, declaracoes in regras if s == seletor]
        assert len(achadas) == 1, f"{seletor}: {len(achadas)} regras"
        assert {
            chave: achadas[0].get(chave)
            for chave in ("background", "border-color", "border-left-color")
        } == {
            "background": f"{{{base}_fundo}}",
            "border-color": f"{{{base}_borda}}",
            "border-left-color": f"{{{base}}}",
        }


@pytest.mark.parametrize("esquema", ["claro", "escuro"])
@pytest.mark.parametrize("nivel", ["aviso", "critico"])
def test_a_faixa_desenhada_sai_nas_cores_do_nivel_nos_dois_esquemas(
    nivel: str, esquema: str, qtbot: Any
) -> None:
    """O desenho, e não só a folha: o fundo, a borda e o traço à esquerda saem na cor do token do
    nível. É o que confere que a propriedade que a faixa marca é a que o seletor lê — um nome
    trocado deixaria a regra da folha certa e a faixa sem a cor dela."""
    faixa = _faixa(qtbot)
    faixa.setStyleSheet(tema.folha_de_estilo(esquema))  # type: ignore[arg-type]
    faixa.mostra("A matriz tem só três QCs.", nivel=nivel)
    faixa.resize(480, faixa.sizeHint().height())
    imagem = faixa.grab().toImage()
    meio = imagem.height() // 2
    cores = asdict(_cores_do(esquema))
    base = _TOKENS_DO_NIVEL[nivel]
    assert {
        "fundo": imagem.pixelColor(8, meio).name(),
        "traco": imagem.pixelColor(1, meio).name(),
        "borda": imagem.pixelColor(imagem.width() // 2, 0).name(),
    } == {"fundo": cores[f"{base}_fundo"], "traco": cores[base], "borda": cores[f"{base}_borda"]}


@pytest.mark.parametrize("esquema", ["claro", "escuro"])
def test_o_texto_da_faixa_tem_contraste_aa_sobre_o_fundo_de_cada_nivel(esquema: str) -> None:
    """Os pares da faixa estão entre os que o portão de contraste deriva da folha, e passam a
    régua da WCAG. A planta: o fundo do nível trocado pela cor do traço reprova nos dois."""
    cores = _cores_do(esquema)
    pares = [par for par in _pares_de_texto(tema._MODELO) if "faixa_de_aviso" in par[0]]
    assert {"nota_aviso_fundo", "nota_erro_fundo"} <= {fundo for _s, _t, fundo in pares}, (
        f"âncora: a faixa ficou fora do portão de contraste — {pares}"
    )
    baixos = [
        (seletor, texto, fundo, round(razao, 2))
        for seletor, texto, fundo in pares
        if (razao := _contraste(_hex(texto, cores), _hex(fundo, cores))) < _MINIMO_DO_TEXTO
    ]
    assert baixos == []
    planta = 'QFrame#faixa_de_aviso[nivel="aviso"] {{ background: {nota_aviso}; }}'
    ((_seletor, texto, fundo),) = _pares_de_texto(planta)
    assert _contraste(_hex(texto, cores), _hex(fundo, cores)) < _MINIMO_DO_TEXTO


def test_a_faixa_recusa_o_nivel_fora_do_vocabulario_com_a_lista_dos_validos(qtbot: Any) -> None:
    faixa = _faixa(qtbot)
    with pytest.raises(ValueError, match=r"'alerta'.*\['aviso', 'critico'\]"):
        faixa.mostra("A matriz tem só três QCs.", nivel="alerta")
    assert faixa.isHidden(), "a recusa mostrou a faixa"


def test_a_faixa_sem_texto_fica_escondida_e_nao_ocupa_altura(qtbot: Any) -> None:
    """Como `RotuloQueSome`: a mesma página com e sem a faixa vazia tem a mesma altura."""
    from PySide6.QtWidgets import QVBoxLayout

    from refino.desktop.janela.componentes import FaixaDeAviso

    def pagina(com_faixa: bool) -> tuple[QWidget, FaixaDeAviso | None]:
        raiz = QWidget()
        qtbot.addWidget(raiz)
        disposicao = QVBoxLayout(raiz)
        disposicao.addWidget(QLabel("acima", raiz))
        faixa = FaixaDeAviso(raiz) if com_faixa else None
        if faixa is not None:
            disposicao.addWidget(faixa)
        disposicao.addWidget(QLabel("abaixo", raiz))
        return raiz, faixa

    sem, _ = pagina(com_faixa=False)
    com, faixa = pagina(com_faixa=True)
    assert faixa is not None
    assert faixa.isHidden(), "a faixa nasce à vista sem nada a dizer"
    assert com.sizeHint().height() == sem.sizeHint().height()
    faixa.mostra("A matriz tem só três QCs.", nivel="aviso")
    assert not faixa.isHidden()
    assert com.sizeHint().height() > sem.sizeHint().height(), "o par não separa"
    faixa.mostra("   ", nivel="aviso")
    assert faixa.isHidden(), "espaço em branco mostrou a faixa"
    faixa.mostra("A matriz tem só três QCs.", nivel="critico")
    faixa.limpa()
    assert faixa.isHidden()
    assert com.sizeHint().height() == sem.sizeHint().height()


@pytest.mark.parametrize(
    ("nivel", "prefixo", "nome"),
    [("aviso", "Atenção:", "Aviso"), ("critico", "Não foi possível:", "Aviso crítico")],
)
@pytest.mark.parametrize("titulo", [None, "Poucos QCs"])
def test_o_texto_abre_com_o_prefixo_do_nivel_e_o_nome_acessivel_diz_o_nivel(
    nivel: str, prefixo: str, nome: str, titulo: str | None, qtbot: Any
) -> None:
    """O nível se lê sem a cor (WCAG 1.4.1): quem não distingue amarelo de vermelho lê o prefixo,
    e o leitor de tela lê o nome."""
    faixa = _faixa(qtbot)
    faixa.mostra("A matriz tem só três QCs.", nivel=nivel, titulo=titulo)
    assert faixa.texto.startswith(f"{prefixo} "), faixa.texto
    assert faixa.texto.count(prefixo) == 1, faixa.texto
    assert "A matriz tem só três QCs." in faixa.texto
    if titulo is not None:
        assert titulo in faixa.texto
    assert faixa.accessibleName().split(":")[0] == nome, faixa.accessibleName()
    assert faixa.accessibleDescription() == faixa.texto


def test_o_botao_da_faixa_emite_o_sinal_dela_e_some_sem_acao(qtbot: Any) -> None:
    faixa = _faixa(qtbot)
    faixa.mostra("A correção falhou.", nivel="critico", acao="Corrigir de novo")
    assert not faixa.botao.isHidden()
    assert faixa.botao.text() == "Corrigir de novo"
    with qtbot.waitSignal(faixa.acionada, timeout=1000):
        faixa.botao.click()
    faixa.mostra("A correção falhou.", nivel="critico")
    assert faixa.botao.isHidden(), "o botão da ação anterior ficou na faixa sem ação"


def _botao_numa_caixa(qtbot: Any) -> tuple[QWidget, QPushButton]:
    from PySide6.QtWidgets import QHBoxLayout

    raiz = QWidget()
    qtbot.addWidget(raiz)
    disposicao = QHBoxLayout(raiz)
    disposicao.addWidget(QLabel("antes", raiz))
    botao = QPushButton("Baixar", raiz)
    disposicao.addWidget(botao)
    disposicao.addWidget(QLabel("depois", raiz))
    return raiz, botao


@pytest.mark.parametrize("frase", ["", "   "])
def test_desligar_sem_motivo_levanta_e_o_botao_continua_ligado(frase: str, qtbot: Any) -> None:
    """Um motivo vazio é exatamente o botão desligado sem explicação, que não se admite."""
    from refino.desktop.janela.componentes import desliga_com_motivo

    raiz, botao = _botao_numa_caixa(qtbot)
    with pytest.raises(ValueError, match="motivo"):
        desliga_com_motivo(botao, frase)
    assert botao.isEnabled(), "a recusa desligou o botão sem frase"
    assert not [r for r in raiz.findChildren(QLabel) if r.buddy() is botao]


def test_o_motivo_aparece_ao_lado_do_botao_e_religar_o_esvazia(qtbot: Any) -> None:
    from refino.desktop.janela.componentes import desliga_com_motivo, religa

    raiz, botao = _botao_numa_caixa(qtbot)
    disposicao = raiz.layout()
    assert disposicao is not None
    rotulo = desliga_com_motivo(botao, "Corrija a matriz para baixar o resultado")
    assert not botao.isEnabled()
    assert rotulo.text() == "Corrija a matriz para baixar o resultado"
    assert not rotulo.isHidden()
    assert rotulo.buddy() is botao, "o rótulo não diz de qual botão é o motivo"
    assert disposicao.indexOf(rotulo) == disposicao.indexOf(botao) + 1, "o motivo não está ao lado"
    religa(botao)
    assert botao.isEnabled()
    assert rotulo.text() == "" and rotulo.isHidden(), "o motivo velho ficou ao lado do botão ligado"
    de_novo = desliga_com_motivo(botao, "Gere um arquivo para ter o resumo")
    assert de_novo is rotulo, "desligar outra vez criou um segundo rótulo"
    assert [r for r in raiz.findChildren(QLabel) if r.buddy() is botao] == [rotulo]
    assert rotulo.text() == "Gere um arquivo para ter o resumo"


def test_desligar_um_botao_fora_de_uma_caixa_levanta_em_vez_de_esconder_o_motivo(
    qtbot: Any,
) -> None:
    """Sem uma caixa de disposição não há «ao lado»: o rótulo ficaria solto no canto do pai."""
    from PySide6.QtWidgets import QGridLayout

    from refino.desktop.janela.componentes import desliga_com_motivo

    solto = QPushButton("Baixar")
    qtbot.addWidget(solto)
    raiz = QWidget()
    qtbot.addWidget(raiz)
    grade = QGridLayout(raiz)
    na_grade = QPushButton("Baixar", raiz)
    grade.addWidget(na_grade, 0, 0)
    for botao in (solto, na_grade):
        with pytest.raises(TypeError, match="caixa"):
            desliga_com_motivo(botao, "Corrija a matriz para baixar o resultado")
        assert botao.isEnabled()


def test_a_etiqueta_beta_e_compacta_e_nao_ocupa_a_altura_do_botao(qtbot: Any) -> None:
    """Sem o alinhamento vertical, o layout esticava a etiqueta até as bordas do botão."""
    barra = _barra_solta("escuro", qtbot)
    barra.cabecalho_da_gaveta.setChecked(True)
    barra.show()
    etiquetas = barra.findChildren(QLabel, "beta_nav")
    assert etiquetas
    for etiqueta in etiquetas:
        assert etiqueta.text() == "BETA"
        botao = etiqueta.parentWidget()
        assert etiqueta.height() < botao.height() * 0.6, (etiqueta.height(), botao.height())

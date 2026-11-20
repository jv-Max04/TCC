"""Os menus e as fontes locais da interface preservam os controles do aplicativo."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import pairwise
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

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt  # noqa: E402
from PySide6.QtGui import QFontDatabase, QFontInfo, QMouseEvent  # noqa: E402
from PySide6.QtTest import QSignalSpy, QTest  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QFrame,
    QLabel,
    QMenuBar,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QToolButton,
    QWidget,
)

from refino.desktop.etapas import CHAVE_DA_ETAPA, ETAPAS  # noqa: E402
from refino.desktop.janela import tema  # noqa: E402
from refino.desktop.janela.arquivos import (  # noqa: E402
    CAMPOS_DOS_ARQUIVOS,
    ROTULO_DA_LISTA,
    ROTULO_DAS_MEDIDAS,
    ROTULO_DOS_FATORES,
)
from refino.desktop.janela.execucao import AbaDeEtapa  # noqa: E402
from refino.desktop.janela.expansao import ExpansaoSuave  # noqa: E402
from refino.desktop.janela.pipeline import TEXTO_DO_ESTADO_DA_ETAPA  # noqa: E402
from refino.desktop.janela.principal import (  # noqa: E402
    ABAS,
    CHAVE_DA_EXECUCAO,
    INDICE_DA_ABA_DA_QUALIDADE,
    INDICE_DA_ABA_DOS_GRAFICOS,
    INDICES_DAS_ABAS_DE_ETAPA,
    JanelaPrincipal,
)
from refino.interface.cliente import Resposta  # noqa: E402


class _SemRede:
    def post(self, url: str, **kwargs: Any) -> Any:
        raise AssertionError("abrir arquivos e navegar não deve calcular nem chamar a API")

    def get(self, url: str, **kwargs: Any) -> Any:
        raise AssertionError("abrir arquivos e navegar não deve calcular nem chamar a API")


class _AlturasObservadas(QObject):
    def __init__(self, widget: QWidget) -> None:
        super().__init__(widget)
        self.alturas: list[int] = []
        widget.installEventFilter(self)

    def eventFilter(self, observado: QObject, evento: QEvent) -> bool:  # noqa: N802
        if evento.type() == QEvent.Type.Resize:
            assert isinstance(observado, QWidget)
            self.alturas.append(observado.height())
        return False


def _acomoda(janela: JanelaPrincipal, qtbot: Any) -> None:
    # O destino da transição substitui a antiga troca instantânea de visibilidade.
    qtbot.wait(30)
    qtbot.waitUntil(
        lambda: not any(e.em_andamento for e in janela.findChildren(ExpansaoSuave)),
        timeout=2000,
    )
    qtbot.wait(30)
    QApplication.processEvents()


@pytest.fixture
def janela_com_tema(qtbot: Any) -> Iterator[JanelaPrincipal]:
    """A geometria exercitada é a do aplicativo entregue, com a fonte e a folha reais."""
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    fonte_anterior, paleta_anterior, folha_anterior = app.font(), app.palette(), app.styleSheet()
    tema.veste_a_aplicacao(app)
    janela = JanelaPrincipal(_SemRede())
    qtbot.addWidget(janela)
    tema.veste_a_janela(janela)
    janela.resize(1280, 800)
    janela.show()
    _acomoda(janela, qtbot)
    try:
        yield janela
    finally:
        janela.close()
        app.setFont(fonte_anterior)
        app.setPalette(paleta_anterior)
        app.setStyleSheet(folha_anterior)


def test_sem_menu_a_barra_lateral_navega_as_mesmas_paginas(qtbot: Any) -> None:
    janela = JanelaPrincipal(_SemRede())
    qtbot.addWidget(janela)
    barra = tema.veste_a_janela(janela)
    assert janela.menuWidget() is None
    assert not janela.findChildren(QMenuBar)
    # «Executar e exportar» saiu em 09/11/2026; a aba do fim entra no lugar dela no percurso, com o
    # título novo, para a barra continuar a ser medida até a última página.
    titulos = [
        "Análise estatística",
        "Preparar o estudo",
        "Métodos e limites",
        "Normalização",
        "Correção",
    ]
    for titulo in titulos:
        indice = next(i for i, aba in enumerate(ABAS) if aba.titulo == titulo)
        pagina_original = janela.abas.widget(indice)
        barra.botoes[indice].click()
        assert janela.abas.currentIndex() == indice
        assert janela.abas.currentWidget() is pagina_original
        assert barra.botoes[indice].isChecked()
        assert sum(botao.isChecked() for botao in barra.botoes) == 1


def test_sem_menu_os_atalhos_e_botoes_preservam_seletores_e_bytes(
    qtbot: Any, tmp_path: Path
) -> None:
    medidas = tmp_path / "measurements.csv"
    lista = tmp_path / "Sample_List.csv"
    fatores = tmp_path / "fatores.csv"
    medidas.write_bytes(b"medicoes escolhidas")
    lista.write_bytes(b"ordem escolhida")
    fatores.write_bytes(b"fatores escolhidos")
    por_titulo = {
        ROTULO_DAS_MEDIDAS: medidas,
        ROTULO_DA_LISTA: lista,
        ROTULO_DOS_FATORES: fatores,
    }
    pedidos: list[str] = []

    def escolhe(titulo: str) -> Path:
        pedidos.append(titulo)
        return por_titulo[titulo]

    janela = JanelaPrincipal(_SemRede(), escolhedor=escolhe)
    qtbot.addWidget(janela)
    tema.veste_a_janela(janela)
    acoes = {acao.text(): acao for acao in janela.actions()}
    assert acoes["Abrir medições…"].shortcut().toString() == "Ctrl+O"
    assert acoes["Abrir lista de amostras…"].shortcut().toString() == "Ctrl+Shift+O"
    par_escolhido = QSignalSpy(janela.arquivos.par_escolhido)

    acoes["Abrir medições…"].trigger()
    assert janela.arquivos.par is None
    acoes["Abrir lista de amostras…"].trigger()
    par = janela.arquivos.par
    assert par is not None
    assert par_escolhido.count() == 1
    assert par.bytes_das_medidas == medidas.read_bytes()
    assert par.bytes_da_lista == lista.read_bytes()

    janela.arquivos.botao_fatores.click()
    assert janela.arquivos.par is par
    assert par_escolhido.count() == 1, "fatores opcionais não devem redisparar o par obrigatório"
    assert janela.arquivos.arquivos_da_execucao()["normalization_factors"][:2] == (
        fatores.name,
        fatores.read_bytes(),
    )
    assert pedidos == [ROTULO_DAS_MEDIDAS, ROTULO_DA_LISTA, ROTULO_DOS_FATORES]


def test_as_fontes_ibm_embarcadas_carregam_sem_instalacao_no_sistema(qtbot: Any) -> None:
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    pasta = tema.pasta_de_recursos() / "fontes"
    fontes = list(pasta.glob("IBMPlex*.ttf"))
    assert fontes, "as fontes devem viajar com o aplicativo"
    assert all(caminho.stat().st_size > 0 for caminho in fontes)
    fonte_anterior = app.font()
    paleta_anterior = app.palette()
    folha_anterior = app.styleSheet()
    try:
        tema.veste_a_aplicacao(app)
        familias = set(QFontDatabase.families())
        assert {"IBM Plex Sans", "IBM Plex Mono"} <= familias
        assert QFontInfo(app.font()).family() == "IBM Plex Sans"
    finally:
        app.setFont(fonte_anterior)
        app.setPalette(paleta_anterior)
        app.setStyleSheet(folha_anterior)


def _faixa_lateral(janela: JanelaPrincipal) -> tuple[tema.BarraLateral, QFrame, QToolButton]:
    barra = janela.findChild(tema.BarraLateral)
    lateral = janela.findChild(QFrame, "faixa_trabalhos_lateral")
    alterna = janela.findChild(QToolButton, "alterna_dos_trabalhos")
    assert barra is not None and lateral is not None and alterna is not None
    return barra, lateral, alterna


def _cabe_na_barra(barra: QWidget, widget: QWidget) -> bool:
    """O widget inteiro dentro da largura da barra, e não espremido abaixo do próprio mínimo."""
    x = widget.mapTo(barra, QPoint(0, 0)).x()
    return (
        x >= 0
        and x + widget.width() <= barra.width()
        and widget.width() >= widget.minimumSizeHint().width()
    )


@pytest.mark.parametrize("largura", [1024, 1440])
def test_d683_trabalhos_ficam_no_fim_da_barra_lateral_recolhidos_e_abrem_num_clique(
    largura: int, janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    """Os trabalhos retraem e ficam na parte inferior do menu lateral."""
    janela = janela_com_tema
    janela.resize(largura, 800)
    _acomoda(janela, qtbot)
    barra, lateral, alterna = _faixa_lateral(janela)
    assert janela.findChild(QFrame, "faixa_trabalhos_topo") is None, "a faixa do topo continua"
    assert barra.isAncestorOf(lateral)
    assert lateral.isAncestorOf(janela.trabalhos)
    assert not janela.findChildren(QStatusBar)
    # No fim: abaixo do último botão da navegação e acima do rodapé da barra.
    ultimo = barra.botoes[-1]
    topo_da_faixa = lateral.mapTo(barra, QPoint(0, 0)).y()
    assert topo_da_faixa > ultimo.mapTo(barra, QPoint(0, 0)).y() + ultimo.height()
    assert topo_da_faixa + lateral.height() <= barra.rodape.mapTo(barra, QPoint(0, 0)).y()
    # Recolhida ao nascer, com o selo à vista: ele responde «já ficou pronto?».
    assert not alterna.isChecked()
    assert not janela.faixa_de_trabalhos.isVisibleTo(janela)
    assert janela.trabalhos.selo.isVisibleTo(janela)
    posicao = lateral.mapTo(janela, QPoint(0, 0))
    for indice in (0, 4, 6):
        janela.abas.setCurrentIndex(indice)
        _acomoda(janela, qtbot)
        pagina = janela.abas.currentWidget()
        assert pagina is not None
        for area in pagina.findChildren(QScrollArea):
            area.verticalScrollBar().setValue(area.verticalScrollBar().maximum())
        QApplication.processEvents()
        assert lateral.isVisibleTo(janela)
        assert lateral.mapTo(janela, QPoint(0, 0)) == posicao
    # Abre num clique, com os três controles inteiros dentro da barra; recolhe no segundo.
    alterna.click()
    _acomoda(janela, qtbot)
    assert janela.faixa_de_trabalhos.isVisibleTo(janela)
    for controle in (
        janela.trabalhos.seletor_de_trabalho,
        janela.trabalhos.botao_de_atualizacao,
        janela.trabalhos.botao_de_cancelamento,
    ):
        assert controle.isVisibleTo(janela)
        assert _cabe_na_barra(barra, controle), f"{controle.objectName() or controle} sai da barra"
    alterna.click()
    _acomoda(janela, qtbot)
    assert not janela.faixa_de_trabalhos.isVisibleTo(janela)


def test_d683_as_linhas_empilhadas_cabem_na_barra_e_a_larga_nao_caberia(
    janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    """Toda chave de trabalho, com a faixa aberta, cabe nos 208 px da barra.

    **A âncora vem antes:** a linha larga, a da faixa do topo, não cabe — sem isso, *"cabe"*
    passaria também com as linhas largas, e o teste não separaria as duas formas.
    """
    from refino.desktop.janela.trabalhos import NOMES_DOS_TRABALHOS, LinhaDoTrabalho

    janela = janela_com_tema
    barra, _, alterna = _faixa_lateral(janela)
    larga = LinhaDoTrabalho("execucao")
    qtbot.addWidget(larga)
    assert larga.minimumSizeHint().width() > barra.width()
    alterna.click()
    janela.trabalhos.mostra({chave: f"id-{i}" for i, chave in enumerate(NOMES_DOS_TRABALHOS)})
    _acomoda(janela, qtbot)
    assert len(janela.trabalhos.linhas) == len(NOMES_DOS_TRABALHOS) >= 2
    for chave, linha in janela.trabalhos.linhas.items():
        assert linha.compacta, f"a linha {chave} nasceu larga"
        for parte in (linha.nome, linha.estado, linha.barra, linha.botao_de_parar):
            assert _cabe_na_barra(barra, parte), f"{chave}: {type(parte).__name__} sai da barra"


def test_d683_a_altura_minima_da_janela_nao_depende_de_quantos_trabalhos_ha(
    janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    """Medido antes do conserto: aberta e com as oito linhas, a barra pedia **979 px** e a altura
    mínima da janela subia junto. O corpo rola em vez de empurrar a janela.

    **A comparação é aberta-vazia contra aberta-cheia**, e não recolhida contra aberta: abrir
    acrescenta o mínimo da área rolável (72 px medidos, 569 para 641), que é o que garante uma
    fileira à vista, e é constante. O que não pode é o mínimo crescer com o CONTEÚDO.
    """
    from refino.desktop.janela.trabalhos import NOMES_DOS_TRABALHOS

    janela = janela_com_tema
    _, lateral, alterna = _faixa_lateral(janela)
    alterna.click()
    _acomoda(janela, qtbot)
    minimo = janela.minimumSizeHint().height()
    janela.trabalhos.mostra({chave: f"id-{i}" for i, chave in enumerate(NOMES_DOS_TRABALHOS)})
    _acomoda(janela, qtbot)
    corpo = lateral.findChild(QScrollArea, "corpo_dos_trabalhos")
    assert corpo is not None
    # A âncora: o conteúdo pede mais que o mínimo do corpo, então havia o que empurrar. Ela lê o
    # `minimumSizeHint` da área, que não depende de quanto a área recebeu.
    assert janela.faixa_de_trabalhos.sizeHint().height() > corpo.minimumSizeHint().height()
    assert janela.minimumSizeHint().height() == minimo
    assert corpo.verticalScrollBar().maximum() > 0, "o conteúdo maior que o corpo não rola"


def test_como_ler_fecha_sem_esticar_o_cartao_entre_duas_alturas(
    janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    janela = janela_com_tema
    janela.abas.setCurrentIndex(next(i for i, aba in enumerate(ABAS) if "qualidade" in aba.paineis))
    _acomoda(janela, qtbot)
    legenda = janela.qualidade.legendas["qc_rsd_median"]
    cartao = legenda.parentWidget()
    assert cartao is not None
    botao = cartao.findChild(QToolButton, "alterna")
    assert botao is not None
    altura_fechada = cartao.height()
    botao.click()
    _acomoda(janela, qtbot)
    assert cartao.height() > altura_fechada
    assert legenda.height() >= legenda.heightForWidth(legenda.width())
    altura_aberta = cartao.height()
    alturas = _AlturasObservadas(cartao)

    botao.click()
    _acomoda(janela, qtbot)

    assert cartao.height() == altura_fechada
    assert alturas.alturas, "o cartão aberto precisa realmente recolher"
    percurso = [altura_aberta, *alturas.alturas]
    assert any(altura_fechada < h < altura_aberta for h in percurso)
    assert all(atual <= anterior for anterior, atual in pairwise(percurso)), percurso


def test_limite_fecha_sem_esticar_a_linha_antes_de_recolher(
    janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    janela = janela_com_tema
    janela.abas.setCurrentIndex(next(i for i, aba in enumerate(ABAS) if "recorte" in aba.paineis))
    _acomoda(janela, qtbot)
    item = janela.recorte._itens[0]
    altura_fechada = item.height()
    item.cabeca.click()
    _acomoda(janela, qtbot)
    assert item.height() > altura_fechada
    altura_aberta = item.height()
    alturas = _AlturasObservadas(item)

    item.cabeca.click()
    _acomoda(janela, qtbot)

    assert item.height() == altura_fechada
    assert alturas.alturas
    percurso = [altura_aberta, *alturas.alturas]
    assert any(altura_fechada < h < altura_aberta for h in percurso)
    assert all(atual <= anterior for anterior, atual in pairwise(percurso)), percurso


def _pagina_com_formulario(janela: JanelaPrincipal, qtbot: Any, etapa: int = 1) -> AbaDeEtapa:
    """A página da `etapa` à vista, no formulário — onde mora o «?» da justificativa que explica os
    controles dela.

    **O formulário e não o estado vazio**: sem os arquivos a página mostra o vazio, e o «?» estaria
    fora da tela. O que a troca decide é `mostra_arquivos`, e é ele que se chama aqui, em vez de ler
    dois arquivos de disco que nada têm a ver com a medida.
    """
    pagina = _paginas_de_etapa(janela)[etapa]
    janela.abas.setCurrentWidget(pagina)
    pagina.mostra_arquivos(())
    _acomoda(janela, qtbot)
    assert pagina.justificativa is not None, f"a etapa {etapa} não recebeu a justificativa"
    assert pagina.sugestoes is not None, f"a etapa {etapa} ficou sem o bloco das sugestões"
    return pagina


def _tela_de(quadro: QWidget) -> Any:
    """A tela onde o quadro está, pelo ponto dele — e não por `widget.screen()`, que no PySide6
    vira filho do widget e morre com ele (`tests/test_desktop_tela.py`)."""
    tela = QApplication.screenAt(quadro.mapToGlobal(QPoint(0, 0)))
    return tela if tela is not None else QApplication.primaryScreen()


def _abre_o_porque(pagina: AbaDeEtapa, qtbot: Any) -> QFrame:
    """Clica no «?» e devolve o quadro que ele abriu, já à vista."""
    assert pagina.sugestoes is not None and pagina.justificativa is not None
    quadro = pagina.justificativa
    assert not quadro.isVisible(), "âncora: o quadro já estava aberto"
    qtbot.mouseClick(pagina.sugestoes.porque, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(quadro.isVisible, timeout=2000)
    return quadro


def _textos_a_vista(quadro: QWidget) -> list[str]:
    """O texto de cada rótulo que o quadro desenha — lido na tela, e não no dicionário que a
    leitura escreve."""
    return [r.text() for r in quadro.findChildren(QLabel) if r.isVisibleTo(quadro)]


# A justificativa saiu do bloco recolhível e passou a abrir num «?» pequeno ao lado da sugestão.
# Os dois testes que estavam aqui eram do bloco recolhível e perderam o assunto:
#
# - `test_fechar_justificativa_nao_estica_as_secoes_vizinhas` afirmava que fechar o bloco não
#   esticava o grupo «Correção do sinal» nem o bloco 2: o bloco ocupava altura na página, e a folga
#   tinha de ir ao fim dela. O quadro é uma janela própria, por cima da página, e abri-lo ou
#   fechá-lo não mexe em altura nenhuma da folha;
# - `test_expandir_nao_muda_a_largura_quando_passar_a_precisar_de_rolagem` afirmava que abrir o
#   bloco, que fazia a página passar a precisar de rolagem, não mudava a largura útil dela. O quadro
#   não entra na rolagem da página, e a página já não muda de altura ao abri-lo.
#
# O que os substitui é o que o quadro pode errar: abrir com o conteúdo da etapa, caber na tela com
# a rolagem dentro dele, e fechar.


def test_o_porque_abre_o_quadro_com_o_titulo_e_as_justificativas_da_etapa(
    janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    """O «?» da Correção abre o título e as justificativas dos dois pontos de decisão — os mesmos
    rótulos que `mostra_justificativa` escreve, e por isso uma leitura nova aparece nele. O da
    Normalização abre a frase dos eixos sobre os quais o programa não opina.

    **O título é lido na tela e escrito aqui por extenso**, e não importado do módulo que o desenha.
    """
    from refino.interface.fluxo import PONTOS_DE_DECISAO_AUTOMATICA

    janela = janela_com_tema
    pagina = _pagina_com_formulario(janela, qtbot)
    quadro = _abre_o_porque(pagina, qtbot)
    titulos = [r.text() for r in quadro.findChildren(QLabel, "tituloDaJustificativa")]
    assert titulos == ["Por que o programa propôs isto"], titulos
    assert PONTOS_DE_DECISAO_AUTOMATICA, "âncora: nenhum ponto de decisão"
    for ponto in PONTOS_DE_DECISAO_AUTOMATICA:
        rotulo = janela.leitura.justificativas[ponto.chave]
        assert quadro.isAncestorOf(rotulo), f"a justificativa de «{ponto.chave}» fora do quadro"
        assert rotulo.isVisibleTo(quadro), f"a justificativa de «{ponto.chave}» escondida"
    palpite = next(p for p in PONTOS_DE_DECISAO_AUTOMATICA if p.chave == "recomendacao")
    novo = "O SERRF deixou os QCs mais estáveis nas 200 features medidas."
    assert novo not in _textos_a_vista(quadro), "âncora: o texto novo já estava no quadro"

    janela.leitura.mostra_justificativa(palpite, {"rationale": novo})

    assert novo in _textos_a_vista(quadro), "a justificativa escrita não chegou ao quadro"
    qtbot.keyClick(quadro, Qt.Key.Key_Escape)
    assert not quadro.isVisible(), "âncora: Esc não fechou o quadro da Correção"

    normalizacao = _pagina_com_formulario(janela, qtbot, etapa=2)
    quadro = _abre_o_porque(normalizacao, qtbot)
    frase = janela.leitura.eixos_sem_proposta[2]
    assert quadro.isAncestorOf(frase) and frase.isVisibleTo(quadro), "a frase da Normalização"
    assert any(t.startswith("**Sobre estes o programa não opina") for t in _textos_a_vista(quadro))
    # A frase curta cabe: o quadro tem a altura dela, e nada fica para rolar nem cortado.
    _acomoda(janela, qtbot)
    rolagem = quadro.findChild(QScrollArea)
    assert rolagem is not None
    assert rolagem.verticalScrollBar().maximum() == 0, "a frase curta não coube no quadro"
    assert quadro.height() < _tela_de(quadro).availableGeometry().height() // 2, (
        "o quadro da frase curta ocupa a tela"
    )
    qtbot.keyClick(quadro, Qt.Key.Key_Escape)


def test_o_quadro_da_justificativa_cabe_na_tela_e_rola_por_dentro(
    janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    """Com as duas justificativas da Correção escritas, a 1280x860, o quadro abre logo abaixo do «?»
    — deslocado só o que falta para caber —, inteiro dentro da tela, com no máximo 460 px de largura
    e o texto que não cabe a rolar DENTRO dele.

    **A âncora é haver o que rolar**: um quadro mais baixo que a tela passaria pela asserção da tela
    sem nunca exercitar o limite. A justificativa do Palpite é a prosa real de `decide_o_palpite`, e
    vai repetida até passar da altura da tela de teste — o assunto é o limite, e não o texto.
    """
    from refino.decision.palpite import decide_o_palpite
    from refino.interface.fluxo import PONTOS_DE_DECISAO_AUTOMATICA

    janela = janela_com_tema
    janela.resize(1280, 860)
    pagina = _pagina_com_formulario(janela, qtbot)
    pontos = {p.chave: p for p in PONTOS_DE_DECISAO_AUTOMATICA}
    janela.leitura.mostra_justificativa(
        pontos["diagnostico"],
        {"lotes": {"n_lotes": 3, "changepoints": [23, 45], "qcs_por_lote": [2, 2, 3]}},
    )
    prosa = decide_o_palpite(
        {"none": 0.10, "loess": 0.30, "serrf": 0.20}, n_features_medidas=200
    ).rationale
    assert prosa.strip(), "âncora: a prosa do Palpite veio vazia"
    janela.leitura.mostra_justificativa(
        pontos["recomendacao"], {"rationale": "\n\n".join([prosa] * 12)}
    )

    quadro = _abre_o_porque(pagina, qtbot)
    _acomoda(janela, qtbot)

    tela = _tela_de(quadro).availableGeometry()
    assert tela.contains(quadro.frameGeometry()), (tela, quadro.frameGeometry())
    assert quadro.width() <= 460, quadro.width()
    porque = pagina.sugestoes.porque if pagina.sugestoes is not None else None
    assert porque is not None
    assert quadro.frameGeometry().top() >= tela.top()
    assert abs(quadro.frameGeometry().left() - porque.mapToGlobal(QPoint(0, 0)).x()) <= 460, (
        "o quadro abriu longe do «?»"
    )
    rolagem = quadro.findChild(QScrollArea)
    assert rolagem is not None, "o quadro não tem rolagem própria"
    barra = rolagem.verticalScrollBar()
    assert barra.maximum() > 0, "âncora: o texto coube, e o limite da tela não foi exercitado"
    conteudo = rolagem.widget()
    assert conteudo is not None
    topo = conteudo.mapTo(rolagem.viewport(), QPoint(0, 0)).y()
    barra.setValue(barra.maximum())
    QApplication.processEvents()
    assert conteudo.mapTo(rolagem.viewport(), QPoint(0, 0)).y() < topo, "o texto não rolou"
    assert conteudo.width() <= rolagem.viewport().width(), "o texto não quebra linha no quadro"
    qtbot.keyClick(quadro, Qt.Key.Key_Escape)


class _InstantesDasPressoes(QObject):
    """O instante de cada pressão que `alvo` recebe — lido fora do código sob teste."""

    def __init__(self, alvo: QWidget) -> None:
        super().__init__(alvo)
        self.instantes: list[int] = []
        alvo.installEventFilter(self)

    def eventFilter(self, observado: QObject, evento: QEvent) -> bool:  # noqa: N802
        if evento.type() == QEvent.Type.MouseButtonPress:
            assert isinstance(evento, QMouseEvent)
            self.instantes.append(evento.timestamp())
        return False


def test_o_clique_no_porque_com_o_quadro_aberto_o_fecha_e_esc_tambem(
    janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    """O clique no «?» com o quadro aberto deixa-o fechado; Esc fecha-o; e o clique seguinte, que é
    outro clique, abre-o outra vez.

    **O clique é entregue como o sistema o entrega**: a pressão vai ao quadro, que está aberto e a
    recebe, e ele fecha; no Windows o Qt reenvia depois a mesma pressão — com o mesmo instante — ao
    widget por baixo, o «?», e a soltura chega a ele. Medido em 11/11/2026 no Qt 6.11.2: no Windows
    o reenvio aconteceu com e sem `WA_NoMouseReplay`, em parte das corridas; sob `offscreen` não há
    reenvio nenhum. O teste entrega o reenvio à mão, para a propriedade valer nos dois casos.

    **Par causal da metade «o clique seguinte abre»:** sem ela, um quadro que recusasse todo clique
    depois de fechado passaria pela primeira metade.
    """
    from PySide6.QtCore import QPointF

    janela = janela_com_tema
    pagina = _pagina_com_formulario(janela, qtbot)
    assert pagina.sugestoes is not None
    porque = pagina.sugestoes.porque
    quadro = _abre_o_porque(pagina, qtbot)
    assert quadro.testAttribute(Qt.WidgetAttribute.WA_NoMouseReplay)
    pressoes = _InstantesDasPressoes(quadro)
    centro = porque.mapToGlobal(porque.rect().center())
    janela_do_quadro = quadro.windowHandle()
    assert janela_do_quadro is not None

    QTest.mousePress(
        janela_do_quadro,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        quadro.mapFromGlobal(centro),
    )
    QApplication.processEvents()
    assert not quadro.isVisible(), "a pressão fora do quadro não o fechou"
    assert len(pressoes.instantes) == 1, "âncora: a pressão não chegou ao quadro"
    local = QPointF(porque.mapFromGlobal(centro))
    reenvio = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        local,
        QPointF(centro),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    reenvio.setTimestamp(pressoes.instantes[0])
    QApplication.sendEvent(porque, reenvio)
    soltura = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        local,
        QPointF(centro),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    soltura.setTimestamp(pressoes.instantes[0] + 1)
    QApplication.sendEvent(porque, soltura)
    QApplication.processEvents()
    assert not quadro.isVisible(), "o clique que fechou o quadro reabriu-o"

    quadro = _abre_o_porque(pagina, qtbot)
    qtbot.keyClick(quadro, Qt.Key.Key_Escape)
    assert not quadro.isVisible(), "Esc não fechou o quadro"


def test_o_porque_abre_pelo_teclado_e_diz_o_que_abre(
    janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    """O «?» é alcançável pelo Tab, abre com Espaço, e o nome acessível e a dica de mouse dizem o
    que ele abre — o título do quadro, por extenso."""
    janela = janela_com_tema
    pagina = _pagina_com_formulario(janela, qtbot)
    assert pagina.sugestoes is not None and pagina.justificativa is not None
    porque = pagina.sugestoes.porque
    assert porque.text() == "?"
    assert porque.accessibleName() == "Por que o programa propôs isto"
    assert porque.toolTip() == "Por que o programa propôs isto"
    assert porque.focusPolicy() & Qt.FocusPolicy.TabFocus, "o «?» não entra no Tab"
    porque.setFocus(Qt.FocusReason.TabFocusReason)
    qtbot.keyClick(porque, Qt.Key.Key_Space)
    qtbot.waitUntil(pagina.justificativa.isVisible, timeout=2000)
    qtbot.keyClick(pagina.justificativa, Qt.Key.Key_Escape)
    assert not pagina.justificativa.isVisible()


def test_nenhuma_pagina_de_etapa_tem_o_bloco_recolhivel_da_justificativa(qtbot: Any) -> None:
    """O bloco «Por que o programa propôs isto» saiu das três páginas; o porquê abre pelo «?».

    **Asserção de ausência com duas âncoras**: a busca acha os blocos recolhíveis que as páginas
    têm — o do resumo dos métodos, no Resultado —, e as duas páginas com sugestões têm o «?». Sem a
    primeira, uma busca que não achasse bloco nenhum passaria por vacuidade.
    """
    from refino.desktop.janela.componentes import Recolhivel

    janela = JanelaPrincipal(_SemRede())
    qtbot.addWidget(janela)
    recolhiveis = 0
    for etapa, pagina in _paginas_de_etapa(janela).items():
        blocos = pagina.findChildren(Recolhivel)
        recolhiveis += len(blocos)
        titulos = [b.alterna.text() for b in blocos]
        assert "Por que o programa propôs isto" not in titulos, f"etapa {etapa}: {titulos}"
        if etapa in (1, 2):
            assert pagina.sugestoes is not None, f"âncora: a etapa {etapa} sem sugestões"
            assert not pagina.sugestoes.porque.isHidden(), f"etapa {etapa}: sem o «?»"
    assert recolhiveis > 0, "âncora: a busca não achou bloco recolhível nenhum nas páginas"


def test_a_justificativa_so_mora_numa_pagina_com_sugestoes(qtbot: Any) -> None:
    """A justificativa abre pelo «?», que mora no bloco das sugestões: uma página que recebesse a
    parte sem o bloco a perderia sem nada acusar, e uma
    com o bloco e sem a parte teria um «?» que abre um quadro vazio. As duas levantam.
    """
    from PySide6.QtWidgets import QProgressBar

    def pagina(etapa: int, justificativa: QWidget | None) -> AbaDeEtapa:
        return AbaDeEtapa(
            (),
            etapa=etapa,
            acao=QPushButton(),
            estado=QLabel(),
            barra=QProgressBar(),
            justificativa=justificativa,
        )

    certa = pagina(1, QLabel("a parte"))
    qtbot.addWidget(certa)
    assert certa.justificativa is not None, "âncora: a página com a parte não a recebeu"
    with pytest.raises(ValueError, match="sugestões"):
        pagina(3, QLabel("a parte"))
    with pytest.raises(ValueError, match="sugestões"):
        pagina(1, None)


@pytest.mark.parametrize("nome", ["frac_features_under_20", "frac_features_under_30"])
def test_legenda_longa_cabe_aberta_e_reflui_ao_redimensionar(
    nome: str, janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    """O texto inteiro cabe a 1024 px e devolve altura ao voltar para a janela mais larga."""
    janela = janela_com_tema
    janela.abas.setCurrentIndex(next(i for i, aba in enumerate(ABAS) if "qualidade" in aba.paineis))
    janela.resize(1440, 800)
    _acomoda(janela, qtbot)
    legenda = janela.qualidade.legendas[nome]
    cartao = legenda.parentWidget()
    assert cartao is not None
    botao = cartao.findChild(QToolButton, "alterna")
    assert botao is not None
    altura_fechada = cartao.height()
    botao.click()
    _acomoda(janela, qtbot)
    alturas: list[int] = []
    for largura in (1440, 1024, 1440):
        janela.resize(largura, 800)
        _acomoda(janela, qtbot)
        assert janela.width() == largura
        assert legenda.isVisibleTo(janela.qualidade)
        pedida = legenda.heightForWidth(legenda.width())
        assert legenda.height() >= pedida, (
            f"{nome}, janela de {largura} px: "
            f"a legenda recebeu {legenda.height()} px e pede {pedida}"
        )
        assert janela.qualidade.rolagem.horizontalScrollBar().maximum() == 0
        alturas.append(legenda.height())
    assert alturas[1] > alturas[0], "a janela estreita deve acomodar as linhas adicionais"
    assert alturas[2] == alturas[0], "a altura não deve ficar presa depois de alargar"

    botao.click()
    _acomoda(janela, qtbot)
    assert not legenda.isVisibleTo(janela.qualidade)
    assert cartao.height() == altura_fechada


def test_cartoes_da_mesma_linha_mantem_o_topo_com_legendas_abertas_e_janela_estreita(
    janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    janela = janela_com_tema
    janela.abas.setCurrentIndex(next(i for i, aba in enumerate(ABAS) if "qualidade" in aba.paineis))
    janela.resize(1440, 800)
    _acomoda(janela, qtbot)
    legendas = [
        janela.qualidade.legendas[nome]
        for nome in ("frac_features_under_20", "frac_features_under_30")
    ]
    cartoes = [legenda.parentWidget() for legenda in legendas]
    assert all(cartao is not None for cartao in cartoes)
    botoes = [cartao.findChild(QToolButton, "alterna") for cartao in cartoes]
    assert all(botao is not None for botao in botoes)

    def confere() -> None:
        assert cartoes[0].parentWidget() is cartoes[1].parentWidget()
        assert cartoes[0].y() == cartoes[1].y(), "o cartão fechado deve continuar no topo da linha"
        for legenda in legendas:
            if legenda.isVisibleTo(janela.qualidade):
                assert legenda.height() >= legenda.heightForWidth(legenda.width())
        assert janela.qualidade.rolagem.horizontalScrollBar().maximum() == 0

    botoes[0].click()
    _acomoda(janela, qtbot)
    confere()
    botoes[1].click()
    _acomoda(janela, qtbot)
    for largura in (1440, 1024, 1440):
        janela.resize(largura, 800)
        _acomoda(janela, qtbot)
        confere()
    botoes[0].click()
    _acomoda(janela, qtbot)
    confere()
    botoes[1].click()
    _acomoda(janela, qtbot)
    confere()


# ---------------------------------------------------------------------------
# O molde das abas de etapa, o estado vazio e o portão dos botões desligados
# ---------------------------------------------------------------------------
#
# Botão desligado traz sempre a explicação, e a tela guia o usuário até um resultado, sem que ele
# o tenha de descobrir sozinho. Cada aba de etapa tem três seções
# que rolam juntas — Entrada; a etapa e o resultado, com o botão, o estado e a barra;
# e os gráficos —, e não mais um rodapé fixo; sem os dois arquivos, a página inteira dá lugar a um
# estado vazio que diz o que falta e leva ao preparo.
#
# As três seções viraram os três blocos numerados de «Preparar o estudo»: 1
# «Ajustes», com a entrada e os parâmetros; 2 «Execução da etapa», com o aviso, o botão, o estado, a
# barra e o Resultado; 3 «Gráficos desta etapa».


def _paginas_de_etapa(janela: JanelaPrincipal) -> dict[int, AbaDeEtapa]:
    paginas: dict[int, AbaDeEtapa] = {}
    for etapa, indice in INDICES_DAS_ABAS_DE_ETAPA.items():
        pagina = janela.abas.widget(indice)
        assert isinstance(pagina, AbaDeEtapa), f"a aba da etapa {etapa} não é uma `AbaDeEtapa`"
        paginas[etapa] = pagina
    assert sorted(paginas) == list(ETAPAS), "âncora: as três abas de etapa"
    return paginas


def _arquivos_do_estudo(pasta: Path) -> dict[str, Path]:
    """Os dois arquivos, com o modo no nome para a janela o deduzir. Os bytes não importam:
    nada aqui chega a uma rota — o disparo é gravado, e não enviado."""
    medidas = pasta / "measurements_neg_c18.csv"
    lista = pasta / "Sample_List_neg_c18.csv"
    medidas.write_bytes(b"medicoes escolhidas")
    lista.write_bytes(b"ordem escolhida")
    return {ROTULO_DAS_MEDIDAS: medidas, ROTULO_DA_LISTA: lista}


def _janela_que_grava(qtbot: Any, pasta: Path) -> tuple[JanelaPrincipal, list[str]]:
    """A janela sem rede, com os arquivos à escolha e o corretor `none` escolhido à mão.

    Até ali a caixa «Automático» era desticada aqui, porque ticada faria a etapa esperar
    o Palpite, que é outra rota. A caixa saiu, e a Correção já não espera o Palpite.
    """
    janela = JanelaPrincipal(_SemRede(), escolhedor=_arquivos_do_estudo(pasta).get)
    qtbot.addWidget(janela)
    janela.pipeline.corretor.setCurrentText("none")
    rotas: list[str] = []
    janela.chama = lambda _metodo, rota, **_k: rotas.append(rota)  # type: ignore[method-assign]
    return janela, rotas


def _carrega_o_par(janela: JanelaPrincipal) -> None:
    janela.arquivos.escolhe_medidas()
    janela.arquivos.escolhe_lista()
    assert janela.arquivos.par is not None, "âncora: o par não se formou"


def _resposta(codigo: int, corpo: Any) -> Resposta:
    return Resposta(codigo=codigo, ok=True, causa=None, mensagem="OK", corpo=corpo)


def _inteiro_a_vista(pagina: AbaDeEtapa, widget: QWidget) -> bool:
    """`widget` inteiro dentro da área visível da rolagem — medido na geometria, e não pela conta
    que a página faz para decidir quanto rolar."""
    area = pagina.rolagem.viewport()
    return area.rect().contains(QRect(widget.mapTo(area, QPoint(0, 0)), widget.size()))


def _titulo_a_vista(pagina: AbaDeEtapa) -> bool:
    """O subtítulo do Resultado inteiro à vista — o bloco do resultado dentro da seção da etapa,
    que é o bloco 2, «Execução da etapa»."""
    return _inteiro_a_vista(pagina, pagina.titulo_do_resultado)


@pytest.fixture
def vestida_com_par(qtbot: Any, tmp_path: Path) -> Iterator[tuple[JanelaPrincipal, list[str]]]:
    """A janela do aplicativo entregue, a 1280x720, com o par carregado pelo seletor."""
    app = QApplication.instance()
    assert isinstance(app, QApplication)
    fonte_anterior, paleta_anterior, folha_anterior = app.font(), app.palette(), app.styleSheet()
    tema.veste_a_aplicacao(app)
    janela, rotas = _janela_que_grava(qtbot, tmp_path)
    tema.veste_a_janela(janela)
    janela.resize(1280, 720)
    janela.show()
    _carrega_o_par(janela)
    _acomoda(janela, qtbot)
    try:
        yield janela, rotas
    finally:
        janela.close()
        app.setFont(fonte_anterior)
        app.setPalette(paleta_anterior)
        app.setStyleSheet(folha_anterior)


def test_sem_par_as_tres_abas_de_etapa_mostram_o_estado_vazio_e_levam_ao_preparo(
    qtbot: Any, tmp_path: Path
) -> None:
    """Sem os dois arquivos, a página não oferece o formulário com o botão da etapa: ela diz o que
    falta e leva a «Preparar o estudo»."""
    from refino.desktop.janela.preparo import AbaDePreparo

    janela, _ = _janela_que_grava(qtbot, tmp_path)
    preparo = next(aba.titulo for aba in ABAS if aba.paineis == ("arquivos", "leitura"))
    for etapa, pagina in _paginas_de_etapa(janela).items():
        assert pagina.pilha.currentWidget() is pagina.vazio, f"a etapa {etapa} mostra o formulário"
        assert pagina.vazio.objectName() == "estado_vazio"
        assert not janela.pipeline.botoes_de_etapa[etapa].isVisibleTo(pagina), (
            f"o botão da etapa {etapa} continua à vista sem os arquivos"
        )
        frase = pagina.frase_do_vazio.text()
        for chave in ("medidas", "lista"):
            assert CAMPOS_DOS_ARQUIVOS[chave][0] in frase, (etapa, frase)
        assert preparo in pagina.botao_do_preparo.text()
        janela.abas.setCurrentWidget(pagina)
        assert janela.abas.currentWidget() is pagina, "âncora: a aba da etapa não ficou à frente"
        pagina.botao_do_preparo.click()
        assert isinstance(janela.abas.currentWidget(), AbaDePreparo), (
            f"o botão do estado vazio da etapa {etapa} não levou a «{preparo}»"
        )


@pytest.mark.parametrize(("escolhida", "falta"), [("lista", "medidas"), ("medidas", "lista")])
def test_o_estado_vazio_diz_qual_dos_dois_arquivos_falta(
    escolhida: str, falta: str, qtbot: Any, tmp_path: Path
) -> None:
    """Com uma metade escolhida, a frase pede a outra — e não a que já está carregada."""
    janela, _ = _janela_que_grava(qtbot, tmp_path)
    escolhe = {"medidas": janela.arquivos.escolhe_medidas, "lista": janela.arquivos.escolhe_lista}
    escolhe[escolhida]()
    assert janela.arquivos.par is None, "âncora: uma metade só não forma o par"
    for etapa, pagina in _paginas_de_etapa(janela).items():
        frase = pagina.frase_do_vazio.text()
        assert pagina.pilha.currentWidget() is pagina.vazio, etapa
        assert CAMPOS_DOS_ARQUIVOS[falta][0] in frase, (etapa, frase)
        assert CAMPOS_DOS_ARQUIVOS[escolhida][0] not in frase, (etapa, frase)


def test_escolher_o_par_troca_as_tres_para_o_formulario_e_remover_volta_ao_vazio(
    qtbot: Any, tmp_path: Path
) -> None:
    """A troca segue os sinais do painel de arquivos, e não o relógio das etapas: a janela nunca é
    mostrada aqui, e o relógio só corre com ela à vista."""
    janela, _ = _janela_que_grava(qtbot, tmp_path)
    paginas = _paginas_de_etapa(janela)
    assert not janela._relogio_das_etapas.isActive(), "âncora: o relógio não pode ser quem troca"
    _carrega_o_par(janela)
    for etapa, pagina in paginas.items():
        assert pagina.pilha.currentWidget() is pagina.formulario, etapa
        assert janela.pipeline.botoes_de_etapa[etapa].isVisibleTo(pagina), etapa
    janela.arquivos.remover_arquivo("lista")
    for etapa, pagina in paginas.items():
        assert pagina.pilha.currentWidget() is pagina.vazio, f"a etapa {etapa} ficou no formulário"
        assert CAMPOS_DOS_ARQUIVOS["lista"][0] in pagina.frase_do_vazio.text(), etapa


def _progresso_da_etapa(janela: JanelaPrincipal, etapa: int) -> None:
    """Uma leitura de progresso a meio — 2 de 5 —, pelo caminho que a etapa usa."""
    if etapa in CHAVE_DA_ETAPA:
        janela.pipeline.mostra_espera(CHAVE_DA_ETAPA[etapa], feito=2, total=5, leituras=1)
    else:
        janela.pipeline.recebe_leitura(
            202, {"identificador": "id", "estado": "executando", "feito": 2, "total": 5}
        )


def _conclui_a_etapa(janela: JanelaPrincipal, etapa: int) -> None:
    """O `200` da etapa: o da prévia nas etapas 1 e 2, e o da execução na 3."""
    if etapa in CHAVE_DA_ETAPA:
        janela._encaminha(CHAVE_DA_ETAPA[etapa], _resposta(200, {}))
    else:
        janela._encaminha(CHAVE_DA_EXECUCAO, _resposta(200, {"qualidade": None}))


@pytest.mark.parametrize("etapa", ETAPAS)
def test_a_linha_da_acao_diz_o_estado_da_etapa_e_leva_a_barra_dela(
    etapa: int, qtbot: Any, tmp_path: Path
) -> None:
    """«não feita» antes de rodar, «em curso» com a barra a correr, e «em dia» depois do `200` — na
    linha da ação da própria aba, dentro da seção do meio, com a nota da etapa logo abaixo.

    Era `test_o_rodape_diz_o_estado_da_etapa_e_leva_a_barra_dela`, e afirmava de
    propósito o desenho anterior: os quatro no rodapé fixo, fora da rolagem. O desenho foi
    revertido nas três telas, e o teste afirma o lugar novo inteiro: o botão, o estado e a barra
    lado a lado na linha da ação, a nota na mesma seção, e os quatro DENTRO da rolagem, que é a
    página inteira.

    A seção do meio é o bloco 2, «Execução da etapa»; o lugar não mudou."""
    janela, rotas = _janela_que_grava(qtbot, tmp_path)
    _carrega_o_par(janela)
    pagina = _paginas_de_etapa(janela)[etapa]
    rotulo = janela.pipeline.rotulos_de_etapa[etapa]
    barra = janela.pipeline.barras_de_etapa[etapa]
    acao = janela.pipeline.botoes_de_etapa[etapa]
    for widget in (acao, rotulo, barra):
        assert pagina.linha_da_acao.isAncestorOf(widget), f"{type(widget).__name__} fora da linha"
    for widget in (pagina.linha_da_acao, janela.pipeline.notas_de_etapa[etapa]):
        assert pagina.secao_da_etapa.isAncestorOf(widget), (
            f"{type(widget).__name__} fora do bloco 2"
        )
        assert pagina.rolagem.isAncestorOf(widget), f"{type(widget).__name__} não rola"
    janela._atualiza_as_etapas()
    assert rotulo.text() == TEXTO_DO_ESTADO_DA_ETAPA["nao_feita"]
    assert (barra.maximum(), barra.value()) == (1, 0), "a barra anda antes de a etapa rodar"

    janela._dispara_etapa(etapa)
    assert len(rotas) == 1, f"âncora: a etapa {etapa} não saiu ({rotas})"
    assert rotulo.text() == TEXTO_DO_ESTADO_DA_ETAPA["em_curso"]
    assert barra.maximum() == 0, (
        "em curso e sem leitura, a barra devia correr sem dizer quanto falta"
    )
    _progresso_da_etapa(janela, etapa)
    assert (barra.maximum(), barra.value()) == (5, 2), "a leitura de progresso não moveu a barra"

    _conclui_a_etapa(janela, etapa)
    assert janela._atualiza_as_etapas()[etapa - 1] == "em_dia", "âncora: o `200` não concluiu"
    assert rotulo.text() == TEXTO_DO_ESTADO_DA_ETAPA["em_dia"]
    assert barra.maximum() > 0 and barra.value() == barra.maximum()


def test_o_proximo_passo_da_analise_diz_o_que_fazer_e_depois_leva_as_abas(
    qtbot: Any, tmp_path: Path
) -> None:
    """A Análise não tem arquivo a baixar: o lugar do Resultado é o «Próximo passo», que diz o que
    fazer antes de ela rodar e leva às abas dos resultados depois do `200` da execução."""
    janela, _ = _janela_que_grava(qtbot, tmp_path)
    _carrega_o_par(janela)
    pagina = _paginas_de_etapa(janela)[ETAPAS[-1]]
    qualidade, graficos = pagina.botao_da_qualidade, pagina.botao_dos_graficos
    assert qualidade is not None and graficos is not None
    janela._atualiza_as_etapas()
    assert pagina.titulo_do_resultado.text() == "Próximo passo"
    assert pagina.frase_do_resultado.text().startswith("Rode a análise para ver os resultados")
    assert not qualidade.isVisibleTo(pagina) and not graficos.isVisibleTo(pagina)

    janela._dispara_etapa(ETAPAS[-1])
    _conclui_a_etapa(janela, ETAPAS[-1])
    assert janela._atualiza_as_etapas()[-1] == "em_dia", "âncora: o `200` não concluiu a análise"
    assert qualidade.isVisibleTo(pagina) and graficos.isVisibleTo(pagina)
    assert not pagina.frase_do_resultado.text().startswith("Rode a análise")
    janela.abas.setCurrentWidget(pagina)
    qualidade.click()
    assert janela.abas.currentIndex() == INDICE_DA_ABA_DA_QUALIDADE
    janela.abas.setCurrentWidget(pagina)
    graficos.click()
    assert janela.abas.currentIndex() == INDICE_DA_ABA_DOS_GRAFICOS


def test_depois_do_200_da_correcao_a_pagina_mostra_a_acao_e_o_resultado(
    vestida_com_par: tuple[JanelaPrincipal, list[str]], qtbot: Any
) -> None:
    """Quando a etapa passa a «em dia», o Resultado vem à vista — e a linha da ação continua nela.

    **A âncora vem antes:** a 1280x720 o Resultado da Correção está abaixo da dobra. Sem isso,
    «à vista depois do `200`» passaria com uma página que nunca rola.

    Até ali o rodapé fixo oferecia «Ir ao resultado» com o Resultado fora da vista, e a
    rolagem punha o topo do Resultado no topo da tela. O atalho saiu com o rodapé, e a rolagem
    passou a ser a MÍNIMA que mostra a linha da ação e o Resultado logo abaixo dela: o estado em
    palavras e a barra, que dizem que a etapa acabou, não podem sair da tela no mesmo gesto que
    mostra o desfecho.
    """
    janela, _ = vestida_com_par
    pagina = _paginas_de_etapa(janela)[1]
    janela.abas.setCurrentWidget(pagina)
    _acomoda(janela, qtbot)
    assert pagina.rolagem.verticalScrollBar().value() == 0
    assert not _titulo_a_vista(pagina), "âncora: o Resultado já estava à vista sem rolar"
    assert not hasattr(pagina, "ir_ao_resultado"), "o atalho do rodapé voltou"

    janela._dispara_etapa(1)
    _conclui_a_etapa(janela, 1)
    _acomoda(janela, qtbot)
    assert _titulo_a_vista(pagina), "o `200` da Correção não trouxe o Resultado à vista"
    assert _inteiro_a_vista(pagina, pagina.linha_da_acao), (
        "a rolagem tirou da tela a linha da ação — o botão, o estado e a barra"
    )


def test_a_rolagem_ate_o_resultado_e_minima(
    vestida_com_par: tuple[JanelaPrincipal, list[str]], qtbot: Any
) -> None:
    """Com a linha da ação e o Resultado já à vista, o `200` não move a página.

    **O par causal é o desenho de antes:** pôr o topo do Resultado no topo da tela moveria a página
    neste mesmo estado, e a âncora mede que moveria — o topo do Resultado está abaixo do topo da
    área visível, então o gesto antigo daria outro valor à barra.
    """
    janela, _ = vestida_com_par
    pagina = _paginas_de_etapa(janela)[1]
    janela.abas.setCurrentWidget(pagina)
    _acomoda(janela, qtbot)
    barra = pagina.rolagem.verticalScrollBar()
    topo_da_acao = pagina.linha_da_acao.mapTo(pagina._conteudo, QPoint(0, 0)).y()
    barra.setValue(topo_da_acao - 40)
    _acomoda(janela, qtbot)
    parada = barra.value()
    assert _inteiro_a_vista(pagina, pagina.linha_da_acao) and _titulo_a_vista(pagina), (
        "âncora: a ação e o Resultado não estavam à vista, e o teste não mediria a rolagem mínima"
    )
    topo_do_resultado = pagina.bloco_do_resultado.mapTo(pagina._conteudo, QPoint(0, 0)).y()
    assert min(topo_do_resultado, barra.maximum()) != parada, (
        "âncora: o gesto antigo, o topo do Resultado no topo da tela, também não moveria a página"
    )

    janela._dispara_etapa(1)
    _conclui_a_etapa(janela, 1)
    _acomoda(janela, qtbot)
    assert janela._atualiza_as_etapas()[0] == "em_dia", "âncora: o `200` não concluiu"
    assert barra.value() == parada, "a página rolou com o Resultado e a ação já à vista"


def test_a_correcao_feita_com_a_pagina_escondida_rola_quando_ela_aparece(
    vestida_com_par: tuple[JanelaPrincipal, list[str]], qtbot: Any
) -> None:
    """A geometria de uma página que nunca foi mostrada não diz onde o Resultado fica: a rolagem
    espera a página aparecer, e então traz o Resultado à vista."""
    janela, _ = vestida_com_par
    pagina = _paginas_de_etapa(janela)[1]
    assert janela.abas.currentWidget() is not pagina, "âncora: a Correção já estava à frente"
    janela._dispara_etapa(1)
    _conclui_a_etapa(janela, 1)
    _acomoda(janela, qtbot)
    janela.abas.setCurrentWidget(pagina)
    _acomoda(janela, qtbot)
    assert _titulo_a_vista(pagina), "a Correção feita com a página escondida não rolou ao aparecer"


def test_parametros_que_voltam_ao_que_eram_nao_arrancam_a_pagina_de_quem_os_edita(
    vestida_com_par: tuple[JanelaPrincipal, list[str]], qtbot: Any
) -> None:
    """De «desatualizada» de volta a «em dia» sem corrida nenhuma: a pessoa devolveu um parâmetro
    ao que era, com o formulário na mão, e a página fica onde ela está."""
    janela, _ = vestida_com_par
    pagina = _paginas_de_etapa(janela)[1]
    janela.abas.setCurrentWidget(pagina)
    _acomoda(janela, qtbot)
    janela._dispara_etapa(1)
    _conclui_a_etapa(janela, 1)
    _acomoda(janela, qtbot)
    barra = pagina.rolagem.verticalScrollBar()
    assert barra.value() > 0, "âncora: o `200` não rolou, e não haveria o que manter"
    barra.setValue(0)
    janela.pipeline.corretor.setCurrentText("loess")
    assert janela._atualiza_as_etapas()[0] == "desatualizada", "âncora: o corretor não desatualizou"
    janela.pipeline.corretor.setCurrentText("none")
    assert janela._atualiza_as_etapas()[0] == "em_dia", "âncora: a volta não pôs a etapa em dia"
    _acomoda(janela, qtbot)
    assert barra.value() == 0, "devolver o parâmetro rolou a página até o Resultado"


def _pontos_da_barra(janela: JanelaPrincipal) -> dict[str, str | None]:
    """O ponto de estado de cada botão da barra lateral que tem um, achado no PRÓPRIO botão — e não
    no registro da barra, que só conhece os que ela criou: o tipo, ou `None` com o ponto escondido.
    """
    barra = tema.BarraLateral.da_janela(janela)
    assert barra is not None, "âncora: a janela não foi vestida"
    achados: dict[str, str | None] = {}
    for titulo, botao in zip(barra.titulos, barra.botoes, strict=True):
        pontos = botao.findChildren(QLabel, "ponto_do_estado")
        assert len(pontos) <= 1, f"{titulo} tem {len(pontos)} pontos"
        if pontos:
            achados[titulo] = None if pontos[0].isHidden() else pontos[0].property("tipo")
    return achados


def _dica_na_barra(janela: JanelaPrincipal, etapa: int) -> str:
    barra = tema.BarraLateral.da_janela(janela)
    assert barra is not None
    return barra.botoes[INDICES_DAS_ABAS_DE_ETAPA[etapa]].toolTip()


def test_o_ponto_da_correcao_na_barra_segue_o_estado_da_etapa(
    vestida_com_par: tuple[JanelaPrincipal, list[str]],
) -> None:
    """Nenhum ponto antes de rodar; info em curso; verde e «em dia» depois do `200`; amarelo e
    «desatualizada» quando um campo da Correção muda — o mesmo recálculo que liga os botões."""
    janela, _ = vestida_com_par
    correcao = ABAS[INDICES_DAS_ABAS_DE_ETAPA[1]].titulo
    assert janela._atualiza_as_etapas() == ("nao_feita",) * 3
    assert _pontos_da_barra(janela) == {}, "um ponto apareceu antes de a etapa rodar"
    assert _dica_na_barra(janela, 1) == f"{correcao} — não feita"

    janela._dispara_etapa(1)
    assert janela._atualiza_as_etapas()[0] == "em_curso", "âncora: a correção não saiu"
    assert _pontos_da_barra(janela) == {correcao: "info"}
    _conclui_a_etapa(janela, 1)
    assert janela._atualiza_as_etapas()[0] == "em_dia", "âncora: o `200` não concluiu"
    assert _pontos_da_barra(janela) == {correcao: "sucesso"}
    assert _dica_na_barra(janela, 1) == f"{correcao} — em dia"

    janela.pipeline.corretor.setCurrentText("loess")
    assert janela._atualiza_as_etapas()[0] == "desatualizada", "âncora: o corretor não desatualizou"
    assert _pontos_da_barra(janela) == {correcao: "aviso"}
    assert _dica_na_barra(janela, 1) == f"{correcao} — desatualizada"


def test_so_as_tres_abas_de_etapa_tem_ponto_na_barra(
    vestida_com_par: tuple[JanelaPrincipal, list[str]],
) -> None:
    """As três etapas feitas põem um ponto em cada uma das três abas delas, e em mais nenhuma."""
    janela, _ = vestida_com_par
    for etapa in ETAPAS:
        janela._dispara_etapa(etapa)
        _conclui_a_etapa(janela, etapa)
    assert janela._atualiza_as_etapas() == ("em_dia",) * 3, "âncora: as três não ficaram em dia"
    assert _pontos_da_barra(janela) == {
        ABAS[indice].titulo: "sucesso" for indice in INDICES_DAS_ABAS_DE_ETAPA.values()
    }


def test_sem_par_as_tres_etapas_nao_tem_ponto_na_barra(
    vestida_com_par: tuple[JanelaPrincipal, list[str]],
) -> None:
    """Sem par a aba da etapa mostra o estado vazio, e a barra não tem estado a dizer — nem o da
    Correção feita antes de o par sair, que a fotografia vazia daria «desatualizada»."""
    janela, _ = vestida_com_par
    correcao = ABAS[INDICES_DAS_ABAS_DE_ETAPA[1]].titulo
    janela._dispara_etapa(1)
    _conclui_a_etapa(janela, 1)
    janela._atualiza_as_etapas()
    assert _pontos_da_barra(janela) == {correcao: "sucesso"}, "âncora: a Correção não ficou verde"
    janela.arquivos.remover_arquivo("lista")
    assert janela.arquivos.par is None, "âncora: o par não saiu"
    assert janela._atualiza_as_etapas()[0] == "desatualizada", (
        "âncora: sem par a Correção não é «desatualizada», e a planta não separaria"
    )
    assert _pontos_da_barra(janela) == {correcao: None}, "o ponto ficou à vista sem par"
    assert _dica_na_barra(janela, 1) == correcao


def test_a_janela_aberta_sem_par_nao_tem_ponto_nem_estado_na_dica(
    janela_com_tema: JanelaPrincipal,
) -> None:
    janela = janela_com_tema
    assert janela.arquivos.par is None, "âncora: a janela abriu com par"
    assert janela._atualiza_as_etapas() == ("nao_feita",) * 3
    assert _pontos_da_barra(janela) == {}
    assert [_dica_na_barra(janela, etapa) for etapa in ETAPAS] == [
        ABAS[INDICES_DAS_ABAS_DE_ETAPA[etapa]].titulo for etapa in ETAPAS
    ]


def test_a_acao_de_cada_etapa_rola_com_a_pagina_entre_os_parametros_e_o_resultado(
    vestida_com_par: tuple[JanelaPrincipal, list[str]], qtbot: Any
) -> None:
    """A linha da ação mora na seção do meio, abaixo do último grupo e acima do Resultado.

    Era `test_o_botao_de_cada_etapa_fica_a_vista_a_1280x720_sem_rolar`, e afirmava de
    propósito o desenho anterior: o formulário rola, e o botão da etapa não. O desenho foi
    revertido, e o custo vai medido em vez de
    escondido: a 1280x720 o botão da Correção fica abaixo da dobra (a âncora), e chega-se a ele
    rolando pelos parâmetros que ele executa, que é a ordem do formulário.

    O último grupo está no bloco 1, «Ajustes», e a linha da ação e o
    Resultado no bloco 2, «Execução da etapa», logo abaixo; a ordem na tela, medida aqui na
    geometria, é a mesma.
    """
    janela, _ = vestida_com_par
    assert (janela.width(), janela.height()) == (1280, 720), "âncora: o tamanho não é o medido"
    for etapa, pagina in _paginas_de_etapa(janela).items():
        janela.abas.setCurrentWidget(pagina)
        _acomoda(janela, qtbot)
        botao = janela.pipeline.botoes_de_etapa[etapa]
        assert pagina.formulario is pagina.rolagem, f"etapa {etapa}: há algo fora da rolagem"
        assert pagina.linha_da_acao.isAncestorOf(botao)
        ultimo_grupo = janela.pipeline.grupos_das_etapas[etapa][-1]

        def _topo(widget: QWidget, pagina: AbaDeEtapa = pagina) -> int:
            return widget.mapTo(pagina._conteudo, QPoint(0, 0)).y()

        ordem = [_topo(ultimo_grupo), _topo(pagina.linha_da_acao), _topo(pagina.bloco_do_resultado)]
        assert ordem == sorted(ordem) and len(set(ordem)) == 3, (
            f"etapa {etapa}: a ação deixou de ficar entre os parâmetros e o Resultado: {ordem}"
        )
        if etapa == ETAPAS[0]:
            assert pagina.rolagem.verticalScrollBar().value() == 0
            assert not _inteiro_a_vista(pagina, botao), (
                "âncora: o botão da Correção já cabia a 1280x720, e o custo declarado não existe"
            )

        # **Repetido até assentar**, e não uma vez: a vista dos gráficos da etapa constrói a figura
        # quando a aba aparece (adiada), e o layout pode mudar depois do primeiro pedido de rolagem.
        # A propriedade é «alcança-se rolando», e não o instante em que o layout assenta.
        def _alcancado(pagina: AbaDeEtapa = pagina, botao: QPushButton = botao) -> bool:
            pagina.rolagem.ensureWidgetVisible(botao)
            QApplication.processEvents()
            return botao.isVisible() and _inteiro_a_vista(pagina, botao)

        qtbot.waitUntil(_alcancado, timeout=3000)


# ---------------------------------------------------------------------------
# As três seções e «Seguir sugestões do app»
# ---------------------------------------------------------------------------
#
# Cada aba de etapa tem três seções — Entrada; Correção e Resultados; Gráficos relevantes para a
# etapa —, com o botão da etapa e a barra de progresso na do meio, nas três telas. Nas duas
# primeiras, as caixas «Automático» deram lugar a um botão «Seguir sugestões do app» no alto da
# seção de entrada.
#
# As três seções viraram três blocos numerados, na mesma divisão de «Preparar o estudo»: bloco 1,
# «Ajustes» (a entrada, com as configurações da etapa); bloco 2, «Execução da etapa»; bloco 3, os
# gráficos relevantes para a etapa.


def _numero_do_bloco(bloco: QWidget) -> str | None:
    """O número que o cabeçalho do bloco mostra, ou `None` se ele não tem selo numérico.

    Lido no rótulo da tela, e não no argumento com que o bloco foi construído: o que se afirma é o
    número que a pessoa vê."""
    numero = getattr(bloco, "numero", None)
    return numero.text() if isinstance(numero, QLabel) else None


def test_cada_aba_de_etapa_tem_as_tres_secoes_na_ordem(qtbot: Any, tmp_path: Path) -> None:
    """Os três blocos numerados de «Preparar o estudo», de cima para baixo, na folha que rola.

    **A ordem é medida no layout da folha**, e não pelos nomes: uma seção que existisse fora da
    folha passaria pelo nome e cairia no índice. Os títulos e os números são lidos na tela e
    escritos aqui por extenso, e não importados do módulo que os desenha. A vista dos gráficos traz
    o bloco 3 dela — um título só, e não um segundo por cima dela.

    **O que cada bloco leva:** os grupos de parâmetros no bloco 1, com a entrada, e não no 2; no
    bloco 2, o aviso da etapa, a linha da ação, as notas e o Resultado, nesta ordem — o Resultado
    fica no bloco 2, junto do botão. **Par causal, feito à mão uma vez em 11/11/2026:** com os
    grupos de volta ao bloco 2, a asserção «dentro do bloco 1» reprova.

    Antes as seções eram «Entrada», a da etapa — «Correção e resultado»,
    «Normalização e resultado» ou «Análise e próximo passo» — e «Gráficos desta etapa», sem número,
    e os grupos de parâmetros moravam na do meio, junto do botão.
    """
    from refino.desktop.janela.componentes import Secao

    janela, _ = _janela_que_grava(qtbot, tmp_path)
    _carrega_o_par(janela)
    for etapa, pagina in _paginas_de_etapa(janela).items():
        disposicao = pagina.folha.layout()
        assert disposicao is not None
        lugares = [pagina.secao_da_entrada, pagina.secao_da_etapa, pagina.lugar_dos_graficos]
        indices = [disposicao.indexOf(lugar) for lugar in lugares]
        assert min(indices) >= 0 and indices == sorted(indices), (etapa, indices)
        vista = janela.graficos.vistas_das_etapas[etapa]
        assert pagina.lugar_dos_graficos.isAncestorOf(vista), f"etapa {etapa}: a vista fora"
        titulos = [
            rotulo.text()
            for rotulo in pagina.lugar_dos_graficos.findChildren(QLabel, "titulo_da_secao")
        ]
        assert titulos == ["Gráficos desta etapa"], f"etapa {etapa}: {titulos}"
        terceiro = vista.secao
        assert isinstance(terceiro, Secao) and pagina.lugar_dos_graficos.isAncestorOf(terceiro)
        blocos = [pagina.secao_da_entrada, pagina.secao_da_etapa, terceiro]
        lidos = [(_numero_do_bloco(bloco), bloco.titulo.text()) for bloco in blocos]
        assert lidos == [
            ("1", "Ajustes"),
            ("2", "Execução da etapa"),
            ("3", "Gráficos desta etapa"),
        ], f"etapa {etapa}: {lidos}"
        # Os números à vista na folha são estes três e só estes: um quarto selo numerado, ou um
        # dos três escondido, seria outra divisão da página.
        numeros = [
            rotulo.text()
            for rotulo in pagina.folha.findChildren(QLabel, "numero_da_secao")
            if rotulo.isVisibleTo(pagina.folha)
        ]
        assert sorted(numeros) == ["1", "2", "3"], f"etapa {etapa}: {numeros}"
        # O estilo é o dos blocos de «Preparar o estudo» — a moldura de referência —, e não o
        # painel plano com a divisória por baixo que as três seções tinham.
        estilos = [(b.property("referencia"), b.property("painelPlano")) for b in blocos]
        assert estilos == [(True, False)] * 3, f"etapa {etapa}: {estilos}"

        grupos = janela.pipeline.grupos_das_etapas[etapa]
        assert grupos, f"âncora: a etapa {etapa} não tem grupo de parâmetros"
        for grupo in grupos:
            assert pagina.secao_da_entrada.isAncestorOf(grupo), (
                f"etapa {etapa}: «{grupo.title()}» fora do bloco 1, «Ajustes»"
            )
            assert not pagina.secao_da_etapa.isAncestorOf(grupo), (
                f"etapa {etapa}: «{grupo.title()}» no bloco 2, junto do botão"
            )

        assert janela.pipeline.notas_de_etapa[etapa] in pagina.notas, "âncora: a nota da etapa"
        corpo = pagina.secao_da_etapa.corpo
        sequencia = [
            pagina.faixa_da_etapa,
            pagina.linha_da_acao,
            *pagina.notas,
            pagina.bloco_do_resultado,
        ]
        posicoes = [corpo.indexOf(widget) for widget in sequencia]
        assert min(posicoes) >= 0 and posicoes == sorted(posicoes), (
            f"etapa {etapa}: o bloco 2 deixou de ser aviso, ação, notas e Resultado: {posicoes}"
        )
        assert not pagina.secao_da_entrada.isAncestorOf(pagina.linha_da_acao)


def test_o_botao_das_sugestoes_fica_no_topo_da_entrada_da_correcao_e_da_normalizacao(
    qtbot: Any, tmp_path: Path
) -> None:
    """Na Correção e na Normalização, o primeiro widget da Entrada; na Análise, nenhum.

    **E nenhuma caixa «Automático» nas três páginas**, que era a outra metade do pedido. A âncora
    é haver caixas nas páginas — os núcleos, o intervalo por barra, as análises —, para a ausência
    não passar sobre uma página vazia.

    A Entrada é o começo do bloco 1, «Ajustes», e o botão continua a ser o primeiro widget dele,
    acima da matriz que
    entra e dos parâmetros.
    """
    from PySide6.QtWidgets import QCheckBox

    from refino.desktop.janela.execucao import EXPLICACOES_DAS_SUGESTOES

    janela, _ = _janela_que_grava(qtbot, tmp_path)
    _carrega_o_par(janela)
    assert sorted(EXPLICACOES_DAS_SUGESTOES) == [1, 2]
    for etapa, pagina in _paginas_de_etapa(janela).items():
        caixas = pagina.findChildren(QCheckBox)
        assert caixas, f"âncora: a página da etapa {etapa} não tem caixa nenhuma"
        automaticas = [c.text() for c in caixas if "Automático" in c.text()]
        assert automaticas == [], f"etapa {etapa}: {automaticas}"
        if etapa not in EXPLICACOES_DAS_SUGESTOES:
            assert pagina.sugestoes is None, f"a etapa {etapa} ganhou o botão das sugestões"
            continue
        bloco = pagina.sugestoes
        assert bloco is not None, f"a etapa {etapa} ficou sem o botão das sugestões"
        assert bloco.botao.text() == "Seguir sugestões do app"
        assert bloco.explicacao.text() == EXPLICACOES_DAS_SUGESTOES[etapa]
        primeiro = pagina.secao_da_entrada.corpo.itemAt(0)
        assert primeiro is not None and primeiro.widget() is bloco, (
            f"etapa {etapa}: o botão das sugestões não abre o bloco 1, «Ajustes»"
        )
        assert bloco.botao.isEnabled(), "o botão das sugestões nasce desligado"


def test_os_campos_da_correcao_e_da_normalizacao_ficam_editaveis_sem_caixa(
    qtbot: Any, tmp_path: Path
) -> None:
    """Todo campo das duas páginas aceita edição desde a abertura.

    **O par causal é o desenho de antes:** com as caixas «Automático» ticadas, o corretor, os seis
    eixos, as frações e os três cortes nasciam DESLIGADOS. A âncora conta os campos, para a
    asserção não passar sobre uma lista vazia.
    """
    janela, _ = _janela_que_grava(qtbot, tmp_path)
    pipeline = janela.pipeline
    campos = [
        pipeline.corretor,
        pipeline.n_lotes,
        pipeline.corte,
        pipeline.ausentes,
        pipeline.max_features,
        *pipeline.eixos.values(),
        pipeline.fracao_posto,
        pipeline.fracao_posto_2,
    ]
    assert len(campos) >= 12, f"âncora: só {len(campos)} campos"
    desligados = [type(c).__name__ for c in campos if not c.isEnabled()]
    assert desligados == [], f"campos desligados sem caixa nenhuma: {desligados}"
    assert not hasattr(pipeline, "automatico"), "a caixa «Automático» da Correção voltou"
    assert not hasattr(pipeline, "automatico_do_corte")
    assert not hasattr(pipeline, "automatico_do_preproc")


def test_seguir_sugestoes_na_normalizacao_repoe_os_valores_iniciais(
    qtbot: Any, tmp_path: Path
) -> None:
    """O botão volta os métodos, as frações e os cortes ao valor inicial, e diz quantos.

    **Par causal:** os campos mudados à mão diferem do inicial (âncora), o clique os repõe, e a
    frase conta os que de facto mudaram; um segundo clique não muda nada e diz isso, em vez de
    «aplicadas» sobre nada. A referência da normalização fica como estava — o nome que a pessoa
    escolheu não é um valor inicial a repor.
    """
    janela, _ = _janela_que_grava(qtbot, tmp_path)
    _carrega_o_par(janela)
    pipeline = janela.pipeline
    bloco = janela.pagina_da_normalizacao.sugestoes
    assert bloco is not None
    iniciais = (
        pipeline.eixos["scaling"].currentText(),
        pipeline.max_features.value(),
        pipeline.corte.value(),
    )
    pipeline.eixos["scaling"].setCurrentText("auto")
    pipeline.max_features.setValue(1234)
    pipeline.corte.setValue(0.3)
    pipeline.referencia.setCurrentText("QC1")
    mexidos = (
        pipeline.eixos["scaling"].currentText(),
        pipeline.max_features.value(),
        pipeline.corte.value(),
    )
    assert all(a != b for a, b in zip(iniciais, mexidos, strict=True)), (
        f"âncora: a planta não mudou os campos {iniciais} -> {mexidos}"
    )

    bloco.botao.click()

    assert (
        pipeline.eixos["scaling"].currentText(),
        pipeline.max_features.value(),
        pipeline.corte.value(),
    ) == iniciais
    assert pipeline.referencia.currentText() == "QC1", "a referência escolhida foi apagada"
    assert bloco.desfecho.text() == "3 campos voltaram ao valor inicial do app."
    assert not bloco.desfecho.isHidden()
    bloco.botao.click()
    assert bloco.desfecho.text().startswith("Os métodos, as frações e os cortes já estavam")
    # Mudar à mão um campo sugerido apaga a frase: ela deixou de descrever a tela.
    pipeline.corte.setValue(0.25)
    assert bloco.desfecho.isHidden(), "a frase das sugestões ficou ao lado de um campo mudado"


def test_seguir_sugestoes_na_correcao_aplica_o_palpite_em_dia_e_os_lotes_propostos(
    qtbot: Any, tmp_path: Path
) -> None:
    """Com o Palpite destes arquivos em dia, o clique põe o corretor dele e os lotes do diagnóstico
    nos campos, por cima do que a pessoa escolheu, sem pedir rota nenhuma.

    **Par causal:** antes do clique os dois campos diferem das propostas (âncora); depois, são elas.
    E nenhuma rota sai: o Palpite já estava em dia para a partição dos lotes propostos.
    """
    janela, rotas = _janela_que_grava(qtbot, tmp_path)
    _carrega_o_par(janela)
    pipeline = janela.pipeline
    bloco = janela.pagina_da_correcao.sugestoes
    assert bloco is not None
    assert pipeline.pre_preenche("n_batches", 2), "âncora: a proposta de lotes não entrou"
    assert pipeline.pre_preenche("correction", "loess"), "âncora: a proposta do corretor não entrou"
    # O Palpite em dia é o da partição dos lotes propostos, que é onde ele é medido.
    janela._palpite_em_dia = janela._identidade_do_palpite()
    pipeline.corretor.setCurrentText("serrf")
    pipeline.n_lotes.setValue(3)
    assert (pipeline.corretor.currentText(), pipeline.n_lotes.value()) == ("serrf", 3), "âncora"
    antes = len(rotas)

    bloco.botao.click()

    assert (pipeline.corretor.currentText(), pipeline.n_lotes.value()) == ("loess", 2)
    assert rotas[antes:] == [], f"o clique pediu rotas com o Palpite em dia: {rotas[antes:]}"
    assert bloco.desfecho.text() == "Sugestões aplicadas: corretor «LOESS», 2 lotes."
    assert pipeline.eixos_divergentes() == (), "seguir as propostas ficou gravado como sobrescrita"
    pipeline.corretor.setCurrentText("none")
    assert bloco.desfecho.isHidden(), "a frase das sugestões ficou ao lado do corretor mudado"


def test_a_correcao_leva_o_corretor_do_seletor_e_nunca_auto(qtbot: Any, tmp_path: Path) -> None:
    """O pedido da Correção sai no clique, com o corretor do seletor e sem `auto`.

    **Par causal sobre o pedido gravado:** dois cliques com dois corretores diferentes mandam os
    dois, e nenhum leva `auto`; e o clique sai na hora, sem o Palpite medido nem pedido — antes,
    a caixa «Automático», ticada por padrão, mandava `auto=medida` e media o Palpite antes.
    """
    from refino.desktop.etapas import ROTA_DA_ETAPA, EstadoDasEtapas

    janela = JanelaPrincipal(_SemRede(), escolhedor=_arquivos_do_estudo(tmp_path).get)
    qtbot.addWidget(janela)
    pedidos: list[tuple[str, dict[str, Any]]] = []

    def grava(_metodo: str, rota: str, **kwargs: Any) -> None:
        pedidos.append((rota, dict(kwargs.get("campos") or {})))

    janela.chama = grava  # type: ignore[method-assign]
    _carrega_o_par(janela)
    assert janela._palpite_em_dia is None, "âncora: o Palpite já estava em dia"
    antes = len(pedidos)
    for corretor in ("loess", "serrf"):
        janela.pipeline.corretor.setCurrentText(corretor)
        # Cada clique sobre um estado sem etapa em curso: a trava recusaria o segundo.
        janela._etapas = EstadoDasEtapas()
        janela._dispara_etapa(1)
        rota, campos = pedidos[-1]
        assert rota == ROTA_DA_ETAPA[1], f"a Correção não saiu no clique: {pedidos}"
        assert campos.get("correction") == corretor, campos
        assert "auto" not in campos, f"a Correção delegou: {campos}"
    assert len(pedidos) - antes == 2, f"o clique pediu mais do que a Correção: {pedidos[antes:]}"


# O portão dos botões desligados. **Contado antes de reprovar**, em 08/11/2026, sobre a janela
# composta, vestida e não vestida, com e sem o par: **10** botões acusados nos dois casos. Dois eram
# das abas de etapa — «2. Pré-processar» e «3. Analisar e registrar», que nasciam desligados da
# trava em cadeia, retirada depois —, e saíram com o molde das abas de etapa. Os outros 8 eram do
# repouso, e o percurso de então não alcançava os que só se desligam em uso.
#
# **Recontado no mesmo dia sobre a janela em uso**, vestida e não vestida, nos estados que desligam
# botão — sem dado, em curso, terminado —, os mesmos que
# `test_nenhum_botao_da_janela_fica_desligado_sem_motivo_em_uso` percorre: **14** botões distintos,
# os mesmos nas duas composições. Os 10 de «Vias metabólicas (beta)», «Qualidade do resultado»,
# «Teste de robustez (beta)» e da faixa de trabalhos
# passaram a `desliga_com_motivo`, e o portão reprovava na janela inteira menos na aba «Gráficos»,
# cujos dois botões — «Voltar à vista original» e «Salvar figura…» — ainda se desligavam sem frase.
#
# «Copiar» e «Salvar como .txt» do resumo, na aba «Executar e exportar», também estiveram fora e
# saíram: o resumo passou a desligá-los com a frase, e a exceção da aba caducou — este
# teste a acusou, e ela saiu do conjunto.
#
# **A aba «Gráficos» saiu quando as figuras entraram nas abas de etapa**: os seus dois botões
# passaram a `desliga_com_motivo`, e «Voltar à vista original» está também na vista de cada aba de
# etapa. O conjunto ficou vazio, e o portão vale na janela inteira, sem exceção.
#
# **A exceção que caduca reprova**: a aba que entrar aqui e sair da janela, ou cujos botões
# passarem a ter frase, sai daqui — `test_o_portao_nao_tem_aba_de_fora` o confere.
_ABAS_FORA_DO_PORTAO: frozenset[str] = frozenset()
_TITULOS_DAS_ETAPAS: frozenset[str] = frozenset(
    ABAS[indice].titulo for indice in INDICES_DAS_ABAS_DE_ETAPA.values()
)
_TITULO_DOS_GRAFICOS: str = ABAS[INDICE_DA_ABA_DOS_GRAFICOS].titulo
_VIAS: str = "Vias metabólicas (beta)"
_MULTIVERSO: str = "Teste de robustez (beta)"
_QUALIDADE: str = ABAS[INDICE_DA_ABA_DA_QUALIDADE].titulo
_FORA_DAS_ABAS: str = "(fora das abas)"


def _aba_de(janela: JanelaPrincipal, widget: QWidget) -> tuple[str, QWidget]:
    """O título da aba cuja página contém `widget`, e a página; fora das abas, a janela."""
    for indice in range(janela.abas.count()):
        pagina = janela.abas.widget(indice)
        if pagina is not None and pagina.isAncestorOf(widget):
            return ABAS[indice].titulo, pagina
    return _FORA_DAS_ABAS, janela


def _desligados_sem_motivo(janela: JanelaPrincipal) -> list[tuple[str, str]]:
    """`(aba, texto)` de cada botão desligado e à vista na página dele sem uma frase de motivo à
    vista.

    **«À vista» é relativo à página da aba** (`isVisibleTo`), e não à tela: a pilha das abas
    esconde as outras páginas, e um portão que só olhasse a corrente veria uma de dez. Um botão
    escondido — o do formulário por trás do estado vazio, a ação de uma faixa sem ação — não está
    desligado aos olhos de ninguém. **O motivo é um rótulo cujo `buddy` é o botão**, a ligação que
    `desliga_com_motivo` faz e que o Qt dá também ao leitor de tela; qualquer rótulo assim conta, e
    não só os daquela função.
    """
    motivos = [(r.buddy(), r) for r in janela.findChildren(QLabel) if r.buddy() is not None]
    acusados = []
    for botao in janela.findChildren(QPushButton):
        if botao.isEnabled():
            continue
        titulo, pagina = _aba_de(janela, botao)
        if not botao.isVisibleTo(pagina):
            continue
        if any(
            alvo is botao and rotulo.text().strip() and rotulo.isVisibleTo(pagina)
            for alvo, rotulo in motivos
        ):
            continue
        acusados.append((titulo, botao.text()))
    return acusados


def _nas_etapas(acusados: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [acusado for acusado in acusados if acusado[0] in _TITULOS_DAS_ETAPAS]


def _no_portao(acusados: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Os acusados que reprovam: a janela inteira, faixa de trabalhos incluída, menos as abas de
    `_ABAS_FORA_DO_PORTAO` — hoje nenhuma."""
    return [acusado for acusado in acusados if acusado[0] not in _ABAS_FORA_DO_PORTAO]


def _desligados_a_vista(janela: JanelaPrincipal) -> set[tuple[str, str]]:
    """`(aba, texto)` de cada botão desligado e à vista na página dele, com frase ou sem — a âncora
    de que o portão olhou o botão que o estado desliga."""
    return {
        (aba[0], botao.text())
        for botao in janela.findChildren(QPushButton)
        if not botao.isEnabled() and botao.isVisibleTo((aba := _aba_de(janela, botao))[1])
    }


def _a_vista_nas_etapas(janela: JanelaPrincipal) -> list[str]:
    """Os botões à vista nas páginas de etapa — a âncora de que o portão olhou alguma coisa."""
    return [
        botao.text()
        for botao in janela.findChildren(QPushButton)
        if (aba := _aba_de(janela, botao))[0] in _TITULOS_DAS_ETAPAS and botao.isVisibleTo(aba[1])
    ]


def test_nenhum_botao_das_abas_de_etapa_fica_desligado_sem_motivo(
    qtbot: Any, tmp_path: Path
) -> None:
    """Nenhum botão desligado sem frase nas abas de etapa, em três estados: sem o par, com ele, e
    com uma etapa em curso — o único em que os botões das etapas se desligam, e então com a frase
    que diz por quê.

    A asserção é de AUSÊNCIA, e cada estado traz a âncora de que havia botão à vista para olhar;
    no terceiro, a de que os três estão de facto desligados.
    """
    janela, _ = _janela_que_grava(qtbot, tmp_path)
    assert _a_vista_nas_etapas(janela), "âncora: sem o par, nenhum botão à vista nas etapas"
    assert _nas_etapas(_desligados_sem_motivo(janela)) == []

    _carrega_o_par(janela)
    janela._atualiza_as_etapas()
    a_vista = _a_vista_nas_etapas(janela)
    assert all(b.text() in a_vista for b in janela.pipeline.botoes_de_etapa.values()), a_vista
    # E os da vista dos gráficos de cada etapa, que nascem desligados sem figura.
    from refino.desktop.janela.graficos import ROTULO_DA_IDA_AOS_GRAFICOS

    assert a_vista.count(ROTULO_DA_IDA_AOS_GRAFICOS) == len(ETAPAS), a_vista
    assert _nas_etapas(_desligados_sem_motivo(janela)) == []

    janela._dispara_etapa(1)
    botoes = janela.pipeline.botoes_de_etapa.values()
    assert not any(botao.isEnabled() for botao in botoes), "âncora: a etapa em curso não desligou"
    assert _nas_etapas(_desligados_sem_motivo(janela)) == []
    motivos = {r.text() for r in janela.findChildren(QLabel) if r.buddy() in list(botoes)}
    assert len(motivos) == 1 and "em curso" in next(iter(motivos)), motivos


def test_o_portao_acusa_o_botao_de_etapa_desligado_sem_motivo_e_passa_sem_a_planta(
    qtbot: Any, tmp_path: Path
) -> None:
    """A planta é o botão da Correção desligado por `setEnabled(False)` direto. Ela cai
    com o par carregado, porque sem ele o botão está escondido pelo estado vazio e as duas
    execuções não divergiriam."""
    janela, _ = _janela_que_grava(qtbot, tmp_path)
    _carrega_o_par(janela)
    janela._atualiza_as_etapas()
    botao = janela.pipeline.botoes_de_etapa[1]
    titulo = ABAS[INDICES_DAS_ABAS_DE_ETAPA[1]].titulo
    assert botao.isEnabled() and botao.text() in _a_vista_nas_etapas(janela), "âncora da planta"
    botao.setEnabled(False)
    assert _nas_etapas(_desligados_sem_motivo(janela)) == [(titulo, botao.text())]
    botao.setEnabled(True)
    assert _nas_etapas(_desligados_sem_motivo(janela)) == []


# O portão na janela inteira, e não só nas abas de etapa
# -------------------------------------------------------
#
# Os botões de fora das etapas só se desligam em uso: sem execução, com um cálculo em curso, depois
# de um trabalho terminar. O percurso abaixo passa por cada estado que desliga um botão, e em cada
# um afirma que o botão está de facto desligado e à vista antes de afirmar que nenhum está
# sem frase. Os cálculos não correm: a predição e o enriquecimento são levados ao estado «em curso»
# pelos mesmos métodos que o clique chama, sem o R e sem o modelo metabólico.


def _corpo_com_volcano() -> dict[str, Any]:
    """Um corpo de execução concluída com o par de grupos: três features, duas abaixo de 0,05."""
    return {
        "volcano": {
            "features": ["f0", "f1", "f2"],
            "pvalue": [0.001, 0.002, 0.5],
            "statistic": [1.0, 2.0, 0.1],
            "grupo_a": "A",
            "grupo_b": "B",
        }
    }


def _tabela_de_identificacao(pasta: Path) -> Path:
    caminho = pasta / "identificacao.csv"
    caminho.write_text(
        "Compound,Description\nf0,Glycine\nf1,Urea\nf2,Citric acid\n", encoding="utf-8"
    )
    return caminho


def test_o_portao_nao_tem_aba_de_fora(qtbot: Any, tmp_path: Path) -> None:
    """**O portão vale na janela inteira, sem exceção.** «Gráficos», a última aba de fora, saiu
    quando os seus dois botões passaram a desligar com a frase. A âncora é ela ter de facto os dois
    desligados e à vista no repouso, e nenhum sem frase; a planta que mostra que o portão os
    veria é `test_o_portao_acusa_o_botao_dos_graficos_desligado_sem_motivo_e_passa_sem_a_planta`.

    A exceção que caduca continua a reprovar: uma aba que voltar a entrar no conjunto tem
    de existir na janela e ainda ter botão desligado sem frase.
    """
    assert frozenset() == _ABAS_FORA_DO_PORTAO, (
        f"há abas fora do portão outra vez: {sorted(_ABAS_FORA_DO_PORTAO)}"
    )
    titulos = {aba.titulo for aba in ABAS}
    assert _TITULO_DOS_GRAFICOS in titulos, "âncora: a aba «Gráficos» mudou de nome"
    janela, _ = _janela_que_grava(qtbot, tmp_path)
    pagina = janela.abas.widget(INDICE_DA_ABA_DOS_GRAFICOS)
    assert pagina is not None
    for botao in (janela.graficos.figuras.botao_da_vista, janela.graficos.botao_de_exportar):
        assert botao.isVisibleTo(pagina) and not botao.isEnabled(), f"âncora: {botao.text()}"
    com_botao_sem_motivo = {aba for aba, _texto in _desligados_sem_motivo(janela)}
    assert _TITULO_DOS_GRAFICOS not in com_botao_sem_motivo
    assert titulos >= _ABAS_FORA_DO_PORTAO <= com_botao_sem_motivo


def test_nenhum_botao_da_janela_fica_desligado_sem_motivo_em_uso(
    qtbot: Any, tmp_path: Path
) -> None:
    """Nenhum botão desligado sem frase na janela inteira — menos `_ABAS_FORA_DO_PORTAO` —, em cada
    estado que desliga um botão nas vias, no enriquecimento, na qualidade, no multiverso e na faixa
    de trabalhos: sem dado, em curso e terminado.

    A asserção é de AUSÊNCIA, e cada estado traz antes a âncora de que os botões que ele desliga
    estão desligados e à vista: sem ela, um estado que não chegasse a desligar nada
    passaria.
    """
    from refino.desktop.janela.enriquecimento import Falha

    janela, _ = _janela_que_grava(qtbot, tmp_path)
    vistos: set[tuple[str, str]] = set()

    def confere(estado: str, *desligados: tuple[str, str]) -> None:
        faltam = set(desligados) - _desligados_a_vista(janela)
        assert not faltam, f"âncora ({estado}): {sorted(faltam)} não estão desligados e à vista"
        assert _no_portao(_desligados_sem_motivo(janela)) == [], estado
        vistos.update(desligados)

    repouso = (
        (_VIAS, "Predizer vias"),
        (_QUALIDADE, janela.qualidade.botao_da_tabela_da_normalidade.text()),
        (_MULTIVERSO, "Ler progresso"),
        (_MULTIVERSO, "Cancelar"),
        (_TITULO_DOS_GRAFICOS, janela.graficos.figuras.botao_da_vista.text()),
        (_TITULO_DOS_GRAFICOS, janela.graficos.botao_de_exportar.text()),
    )
    confere("repouso", *repouso)
    _carrega_o_par(janela)
    janela._atualiza_as_etapas()
    confere("com o par", *repouso)

    vias = janela.vias
    vias.recebe_volcano({"volcano": None})
    confere("execução sem o par de grupos", (_VIAS, "Predizer vias"))
    vias.recebe_volcano(_corpo_com_volcano())
    assert vias.botao.isEnabled(), "âncora: o volcano não ligou «Predizer vias»"
    # A predição não corre: o estado em curso é o que o clique deixa antes de a thread começar.
    vias._piscina.start = lambda _trabalho: None  # type: ignore[method-assign]
    vias._calcula_e_mostra()
    confere("predição em curso", (_VIAS, "Predizer vias"))
    vias._recebe_falha(RuntimeError("planta"), vias._geracao)
    assert vias.botao.isEnabled(), "âncora: o fim da predição não religou o botão"

    painel = vias.enriquecimento
    vias.modo.setCurrentIndex(vias.modo.findData("metaboanalystr"))
    confere(
        "enriquecimento sem tabela",
        (_VIAS, "Confirmar colunas"),
        (_VIAS, "Enriquecer"),
        (_VIAS, "Cancelar"),
        (_VIAS, "Exportar CSV…"),
    )
    tabela = _tabela_de_identificacao(tmp_path)
    painel._abre_arquivo = lambda _pai: tabela  # type: ignore[method-assign]
    painel._carrega_tabela()
    confere("colunas por confirmar", (_VIAS, "Enriquecer"))
    painel._confirma_colunas()
    painel.valor_do_corte.setValue(painel.valor_do_corte.minimum())
    confere("recusa antecipada: nenhum composto passa o corte", (_VIAS, "Enriquecer"))
    painel.valor_do_corte.setValue(0.05)
    assert painel.botao_enriquece.isEnabled(), "âncora: a lista recortada não ligou «Enriquecer»"
    painel._comeca_espera()
    confere("enriquecimento em curso", (_VIAS, "Enriquecer"), (_VIAS, "Exportar CSV…"))
    painel._recebe(Falha("motor_falhou", "planta"), painel._token)
    confere("enriquecimento terminado", (_VIAS, "Cancelar"), (_VIAS, "Exportar CSV…"))

    multiverso = janela.multiverso
    multiverso.aceita_disparo(202, {"identificador": "m-1", "total": 9600, "persistido": True})
    confere("multiverso a correr", (_MULTIVERSO, "Executar teste de robustez"))
    multiverso.botao_de_cancelamento.click()
    confere("parada pedida", (_MULTIVERSO, "Executar teste de robustez"), (_MULTIVERSO, "Cancelar"))
    multiverso.recebe_progresso(200, {"terminou": True, "estado": "concluido"})
    confere("multiverso terminado", (_MULTIVERSO, "Cancelar"))

    janela.trabalhos.mostra({"execucao": "id-1"})
    janela.trabalhos.mostra_estado("execucao", 200, {})
    janela.trabalhos.mostra({"execucao": "id-1"})
    confere("trabalho terminado", (_FORA_DAS_ABAS, "Parar"))

    janela.qualidade.mostra_normalidade({"run_id": 1, "normalidade": None})
    confere("execução sem a normalidade", repouso[1])

    assert {aba for aba, _texto in vistos} == {
        _VIAS,
        _QUALIDADE,
        _MULTIVERSO,
        _FORA_DAS_ABAS,
        _TITULO_DOS_GRAFICOS,
    }


def test_o_parar_de_um_trabalho_terminado_diz_por_que_e_cabe_na_barra_lateral(
    janela_com_tema: JanelaPrincipal, qtbot: Any
) -> None:
    """Na janela vestida a faixa de trabalhos vive na barra lateral, com as linhas empilhadas, e o
    «Parar» de um trabalho terminado leva a frase ao lado dele. Ela tem de caber na barra junto com
    o nome, como as outras peças da linha."""
    from refino.desktop.janela.trabalhos import MOTIVO_JA_TERMINOU, NOMES_DOS_TRABALHOS

    janela = janela_com_tema
    barra, _, alterna = _faixa_lateral(janela)
    alterna.click()
    trabalhos = {chave: f"id-{i}" for i, chave in enumerate(NOMES_DOS_TRABALHOS)}
    janela.trabalhos.mostra(trabalhos)
    for chave in trabalhos:
        janela.trabalhos.mostra_estado(chave, 200, {})
    janela.trabalhos.mostra(trabalhos)
    _acomoda(janela, qtbot)
    assert len(janela.trabalhos.linhas) == len(NOMES_DOS_TRABALHOS) >= 2
    assert _no_portao(_desligados_sem_motivo(janela)) == []
    for chave, linha in janela.trabalhos.linhas.items():
        assert linha.compacta, f"a linha {chave} nasceu larga"
        assert not linha.botao_de_parar.isEnabled(), f"âncora: o «Parar» de {chave} está ligado"
        motivos = [r for r in linha.findChildren(QLabel) if r.buddy() is linha.botao_de_parar]
        assert [r.text() for r in motivos] == [MOTIVO_JA_TERMINOU], chave
        assert motivos[0].isVisibleTo(janela), f"a frase do «Parar» de {chave} está escondida"
        for parte in (linha.nome, linha.estado, linha.barra, linha.botao_de_parar, motivos[0]):
            assert _cabe_na_barra(barra, parte), f"{chave}: {type(parte).__name__} sai da barra"


def _ligado_nas_vias(janela: JanelaPrincipal) -> QPushButton:
    janela.vias.recebe_volcano(_corpo_com_volcano())
    return janela.vias.botao


def _ligado_na_qualidade(janela: JanelaPrincipal) -> QPushButton:
    """O botão com a tabela da normalidade à mão: o mesmo `religa` que a chegada dela chama."""
    from refino.desktop.janela.componentes import religa

    botao = janela.qualidade.botao_da_tabela_da_normalidade
    religa(botao)
    return botao


def _ligado_no_multiverso(janela: JanelaPrincipal) -> QPushButton:
    return janela.multiverso.botao_de_disparo


def _ligado_na_faixa(janela: JanelaPrincipal) -> QPushButton:
    janela.trabalhos.mostra({"execucao": "id-1"})
    return janela.trabalhos.linhas["execucao"].botao_de_parar


@pytest.mark.parametrize(
    ("regiao", "ligado"),
    [
        (_VIAS, _ligado_nas_vias),
        (_QUALIDADE, _ligado_na_qualidade),
        (_MULTIVERSO, _ligado_no_multiverso),
        (_FORA_DAS_ABAS, _ligado_na_faixa),
    ],
    ids=["vias", "qualidade", "multiverso", "faixa"],
)
def test_o_portao_acusa_o_botao_desligado_sem_motivo_fora_das_etapas_e_passa_sem_a_planta(
    qtbot: Any, tmp_path: Path, regiao: str, ligado: Any
) -> None:
    """Uma planta em cada região que o portão passou a cobrir: um botão ligado e à vista,
    desligado por `setEnabled(False)` direto. As duas execuções diferem só pela planta. Na
    qualidade o botão nasceu com frase, esvaziada pelo `religa`: a planta mostra também que um
    rótulo sem texto não conta como motivo."""
    janela, _ = _janela_que_grava(qtbot, tmp_path)
    botao = ligado(janela)
    aba, pagina = _aba_de(janela, botao)
    assert aba == regiao and botao.isEnabled() and botao.isVisibleTo(pagina), "âncora da planta"
    assert _no_portao(_desligados_sem_motivo(janela)) == []
    botao.setEnabled(False)
    assert _no_portao(_desligados_sem_motivo(janela)) == [(regiao, botao.text())]
    botao.setEnabled(True)
    assert _no_portao(_desligados_sem_motivo(janela)) == []


def test_o_portao_ve_o_download_do_resultado_e_acusa_o_desligado_sem_motivo(
    qtbot: Any, tmp_path: Path
) -> None:
    """O download do Resultado da Correção está sob o portão: desligado com a frase antes de
    corrigir, ele passa; ligado depois, e desligado por `setEnabled(False)` direto — a planta —, o
    portão o acusa. A âncora é ele estar à vista na página, senão o portão não o olharia."""
    janela, _ = _janela_que_grava(qtbot, tmp_path)
    _carrega_o_par(janela)
    janela._atualiza_as_etapas()
    baixar = janela.resultado_da_correcao.baixar
    titulo = ABAS[INDICES_DAS_ABAS_DE_ETAPA[1]].titulo
    assert baixar.text() in _a_vista_nas_etapas(janela), "âncora: o download não está à vista"
    assert not baixar.isEnabled(), "âncora: antes de corrigir o download nasce desligado"
    assert _nas_etapas(_desligados_sem_motivo(janela)) == []

    janela._dispara_etapa(1)
    janela._encaminha(CHAVE_DA_ETAPA[1], _resposta(200, {"correcao": "none"}))
    janela._atualiza_as_etapas()
    assert baixar.isEnabled(), "âncora: a correção feita não ligou o download"
    baixar.setEnabled(False)
    assert _nas_etapas(_desligados_sem_motivo(janela)) == [(titulo, baixar.text())]
    baixar.setEnabled(True)
    assert _nas_etapas(_desligados_sem_motivo(janela)) == []


@pytest.mark.parametrize("qual", ["voltar", "salvar"])
def test_o_portao_acusa_o_botao_dos_graficos_desligado_sem_motivo_e_passa_sem_a_planta(
    qual: str, qtbot: Any, tmp_path: Path
) -> None:
    """A planta é um dos dois botões da aba «Gráficos» desligado por `setEnabled(False)`
    direto. Ela cai com uma figura desenhada, porque no repouso eles já estão desligados — com a
    frase —, e as duas execuções não divergiriam."""
    from matplotlib.figure import Figure

    janela, _ = _janela_que_grava(qtbot, tmp_path)
    graficos = janela.graficos
    figura = Figure()
    figura.add_subplot()
    graficos.figuras.desenha(figura)
    botao = {"voltar": graficos.figuras.botao_da_vista, "salvar": graficos.botao_de_exportar}[qual]
    assert botao.isEnabled(), "âncora da planta: com a figura na tela o botão está ligado"
    assert _no_portao(_desligados_sem_motivo(janela)) == []
    botao.setEnabled(False)
    assert _no_portao(_desligados_sem_motivo(janela)) == [(_TITULO_DOS_GRAFICOS, botao.text())]
    botao.setEnabled(True)
    assert _no_portao(_desligados_sem_motivo(janela)) == []

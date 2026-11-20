"""`PainelDeFigura`: tela, tema, escala, interação e recusa.

## O oráculo é o WIDGET, em duas metades, e não o valor de retorno

`estado_visivel()` lê o texto **e** confere que o widget está no layout e visível. A revisão
adversarial de `PainelDeQualidade` mediu o custo de não o fazer: cinco mutantes passavam porque
toda asserção lia o dicionário que os `mostra_*` devolvem, e um deles deixava o painel em branco.

**E o oráculo desta unidade tem uma metade a mais que o daquela**, porque a metade em falta seria
silenciosa: `QLayout.indexOf` **não** desce por layouts aninhados — medido em 10/09/2026 —, e o
seletor deste painel vive numa linha horizontal dentro da pilha. Um oráculo copiado tal e qual
diria *"o combo não está na tela"* sobre um combo desenhado, e a asserção que o afirmasse passaria
a ser vácua sem nada acusar. `test_o_oraculo_de_desenhado_desce_por_layout_aninhado` é o portão
dessa metade.

## Sobre as figuras destes testes: elas passam pelo arnês, não por asserção à mão

Toda asserção sobre o conteúdo de uma figura vai por `tests/arnes_de_figura` — as quatro classes
dele —, e cada planta traz a sua âncora dentro dela. Escrever `assert len(ax.lines) == 2` aqui
seria reimplementar a classe (a) sem as classes (c) e (d), que são as que apanham a figura
tecnicamente certa e visualmente vazia.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import pytest

from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a decisão é exercitada em `tests/test_qt_guard.py`, com planta
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()

from matplotlib.backend_bases import MouseEvent, PickEvent, scroll_handler  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QPalette  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from refino.api.esquemas import COEFICIENTE_DE_TUKEY  # noqa: E402
from refino.api.rotas import _chave_do_trabalho  # noqa: E402
from refino.desktop.janela import figuras as modulo_das_figuras  # noqa: E402
from refino.desktop.janela.figuras import (  # noqa: E402
    CHAVE_DE_REPOUSO,
    CORES_DA_FIGURA,
    DPI_DA_EXPORTACAO,
    DPIS_DA_EXPORTACAO,
    FORMATOS_DA_EXPORTACAO,
    MOTIVO_SEM_VISTA,
    POR_QUE_TODAS_SAEM,
    SUFIXO_DA_EXPORTACAO,
    TECLA_DO_ZOOM,
    TEXTO_DO_DESFECHO,
    TEXTO_DO_ESTADO,
    DesfechoDaExportacao,
    EstadoDaFigura,
    FonteAusenteError,
    HistoricoMudo,
    PainelDeFigura,
    aplica_o_tema,
    cores_do_tema,
    dpi_da_tela,
    entradas_do_seletor,
    estado_da_resposta,
    exporta_habilitado,
    motivo_da_chave,
    texto_do_desfecho,
    texto_do_estado,
)
from refino.desktop.janela.graficos import (  # noqa: E402
    ETAPA_DA_FIGURA,
    FIGURA_DA_CURVA_DO_LOESS,
    FIGURAS_DO_DIAGNOSTICO,
    MOTIVO_DA_EXPORTACAO_DESLIGADA,
    MOTIVO_DA_IDA_AOS_GRAFICOS_DESLIGADA,
    ROTULO_DA_IDA_AOS_GRAFICOS,
    SEM_EXPORTACAO,
    PainelDeGraficos,
    frase_da_etapa_desatualizada,
    frase_da_etapa_sem_resultado,
)
from refino.domain.matrix import RefinoMatrix  # noqa: E402
from refino.domain.metadata import SampleMetadata  # noqa: E402
from refino.interface.cliente import (  # noqa: E402
    PREFIXO_DE_DIVERGENCIA,
    Resposta,
    chave_de_invalidacao,
)
from refino.plots import CONSTRUTORES, MatrizVaziaError, Recorte, aplica, nova_figura  # noqa: E402
from refino.plots.base import (  # noqa: E402
    ALTURA_PADRAO,
    GID_DAS_FEATURES,
    GID_DO_RODAPE,
    LARGURA_PADRAO,
    RecusaDoConstrutorError,
)
from refino.plots.catalogo import ConstrutorDeFigura, Vinculo  # noqa: E402
from refino.plots.pca import (  # noqa: E402
    cargas_da_pca,
    figura_das_cargas,
    rotulo_da_regiao_do_grupo,
)
from refino.plots.recorte import decimal  # noqa: E402
from refino.plots.volcano import (  # noqa: E402
    MANN_WHITNEY,
    ROTULO_DA_SERIE,
    ROTULO_DAS_BARRAS,
    WELCH,
)
from tests.arnes_de_figura import (  # noqa: E402
    confere_numeros,
    confere_rotulos,
    confere_series,
    confere_vista,
    numeros_declarados,
    planta_corte_de_eixo,
    series,
    texto_corrido,
)
from tests.test_desktop_curva_do_loess import corpo_da_fabrica  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from pytestqt.qtbot import QtBot

# ---------------------------------------------------------------------------
# O material dos testes
# ---------------------------------------------------------------------------

# Os pontos das duas séries. **`x` difere de `y` e nenhum `y` é zero de propósito**: são as âncoras
# que `planta_eixos_trocados` e `planta_sinal_invertido` exigem, e uma figura de teste que não as
# satisfizesse tornaria as plantas do arnês vácuas sobre ela.
X_A = (1.0, 2.0, 3.0, 4.0)
Y_A = (2.0, 5.0, 3.0, 9.0)
X_B = (1.0, 2.0, 3.0, 4.0)
Y_B = (1.0, 1.5, 4.0, 6.0)

NOMES = ("2.25_810.7739m/z", "3.10_155.0345m/z", "4.02_301.1400m/z", "5.55_412.9001m/z")

# O recorte que a figura de teste declara. Os números entram na classe (d) do arnês, e o teste
# deriva o que espera **destes literais**, não do objeto que o construtor usou.
DESENHADAS = 4
DISPONIVEIS = 6035


def figura_de_teste() -> Figure:
    """Uma figura com duas séries, rótulos, legenda e rodapé de recorte.

    **Construída com `refino.plots` e não à mão**, porque é sobre a saída daquele pacote que o
    widget tem de funcionar: `nova_figura` não pinta nada — é isso que faz o tema deste painel ser
    a única opinião sobre cor — e `aplica` põe o rodapé de onde a classe (d) lê os números.

    **AS DUAS SÉRIES LEVAM `GID_DAS_FEATURES`, e a marca entrou em 11/09/2026 com o conserto da
    identificação.** Elas são **paralelas**: `antes` e `depois` descrevem as mesmas quatro features
    na mesma ordem, e o índice 3 é a mesma feature nas duas — que é exactamente o que a marca
    afirma. Antes disto a afirmação não existia em lado nenhum e o painel aplicava o vetor de nomes
    a **qualquer** artista pela posição; a fixtura passava porque aqui isso calha de estar certo, e
    o mesmo mecanismo devolvia o nome de uma feature ao clique na cruz da origem de uma figura de
    cargas. Marcar aqui é escrever o que a fixtura sempre quis dizer — não é afrouxar o teste: o
    par causal `test_o_artista_NAO_NOMEADO_cai_no_rotulo_e_o_nomeado_da_a_feature` planta a
    remoção da marca e vê a identificação mudar.
    """
    figura = nova_figura()
    eixo = figura.add_subplot()
    linha = eixo.plot(X_A, Y_A, marker="o", label="antes")[0]
    nuvem = eixo.scatter(X_B, Y_B, label="depois")
    for artista in (linha, nuvem):
        artista.set_gid(GID_DAS_FEATURES)
    eixo.set_xlabel("ordem de injeção")
    eixo.set_ylabel("intensidade")
    eixo.set_title("figura de arnês")
    eixo.legend()
    aplica(
        figura,
        Recorte(
            desenhadas=DESENHADAS,
            disponiveis=DISPONIVEIS,
            amostragem="as quatro primeiras",
            matriz="ingerida",
            metodo="sem teste estatístico",
        ),
    )
    return figura


def _matriz() -> RefinoMatrix:
    """Uma matriz mínima, só para `_chave_do_trabalho` ter sobre o que digerir.

    **Sintética e legítima**, porque nada aqui é contagem observada nos arquivos reais:
    o que se afirma é uma propriedade da **derivação da chave**, e ela não depende do dado.
    """
    amostras = (
        SampleMetadata(name="QC_01", injection_order=1, sample_type="qc", group="QC"),
        SampleMetadata(name="A_01", injection_order=2, sample_type="sample", group="A"),
    )
    return RefinoMatrix(
        values=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        feature_names=("1.0_100.0m/z", "2.0_200.0m/z"),
        samples=amostras,
        mode="pos_hilic",
    )


def _campos(seed: int) -> dict[str, Any]:
    """Os campos de uma execução. **Só `seed` varia entre as duas chamadas do par causal.**"""
    return {
        "mode": "pos_hilic",
        "correction": "none",
        "imputation": "half_min",
        "scaling": "pareto",
        "seed": seed,
    }


def _chave(seed: int) -> str:
    return _chave_do_trabalho("execucoes", _matriz(), _campos(seed))


def _resposta(codigo: int, corpo: object) -> Resposta:
    return Resposta(codigo=codigo, ok=codigo in (200, 202), causa=None, mensagem="", corpo=corpo)


def _corpo(seed: int) -> dict[str, Any]:
    """O corpo de `/execucoes` de uma execução, com tudo igual menos a chave.

    **O `modo`, a `apresentacao` e a mediana ficam IGUAIS entre as duas sementes de propósito**:
    são o que um atalho de três eixos leria, e é sobre eles que a âncora corre.
    """
    return {
        "matrix_id": 1,
        "apresentacao": "pos_hilic|none",
        "qc_rsd_median": 0.17,
        "chave_de_invalidacao": _chave(seed),
    }


def _projecao_pobre(corpo: dict[str, Any]) -> tuple[Any, ...]:
    """O atalho de três eixos que `api/reuso.py:174-176` mediu e recusou: modo, método, digest."""
    return (corpo["apresentacao"], corpo["matrix_id"], corpo["qc_rsd_median"])


def _um_construtor() -> ConstrutorDeFigura:
    return ConstrutorDeFigura(
        modulo="arnes", funcao="figura_de_arnes", vinculo=Vinculo.RELATORIO, chave="item-0"
    )


def _eixo(painel: PainelDeFigura) -> Axes:
    return painel.canvas.figure.axes[0]


# ---------------------------------------------------------------------------
# Caminho feliz e layout
# ---------------------------------------------------------------------------


def test_o_painel_monta_com_o_canvas_no_layout_e_o_oraculo_o_ve(qtbot: QtBot) -> None:
    """O caminho feliz, e o oráculo é o das duas metades.

    A âncora de não-vacuidade vem primeiro: um painel cujo `estado_visivel` devolvesse
    tudo vazio satisfaria por vacuidade toda asserção de ausência abaixo.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)

    em_repouso = painel.estado_visivel()
    assert set(em_repouso) == {
        "repouso",
        "seletor",
        "anotacao",
        "figura",
        # **AS DUAS ENTRADAS DA EXPORTAÇÃO SAÍRAM DAQUI NO MERGE, e a capacidade NÃO se perdeu.**
        # O bloco das análises pô-las neste dicionário porque a exportação que ele escreveu
        # não tinha estado próprio. A implementação que ficou é a do plano dos gráficos, que
        # expõe a mesma propriedade por `exporta_habilitado` e por `EstadoDaFigura` — e tem 51
        # testes próprios neste arquivo a afirmá-la. Mantê-las aqui exigiria duas fontes para o
        # mesmo fato, que é a irmã que diverge no primeiro dia.
    }, f"âncora: o oráculo mudou de forma e devolve {sorted(em_repouso)}"
    assert em_repouso["repouso"] == texto_do_estado(EstadoDaFigura.SEM_EXECUCAO)
    assert em_repouso["figura"] == "", "o canvas está na tela antes de haver figura nenhuma"

    painel.desenha(figura_de_teste(), nomes=NOMES)
    desenhado = painel.estado_visivel()
    assert desenhado["figura"] == "1 eixo(s)", desenhado
    assert desenhado["repouso"] == "", (
        "o texto de repouso continua na tela por cima da figura — duas fontes para a mesma verdade"
    )


def test_a_folga_do_layout_e_do_canvas_e_nao_ha_esticador_ao_fim(qtbot: QtBot) -> None:
    """Quem cresce é a figura, e um esticador ao fim roubar-lhe-ia a altura.

    O portão de `test_desktop_janela.py:6017-6175` confere que o módulo **declara** um dono da
    folga, por AST. Este confere o que o AST não vê: que o dono é o canvas, e que não há esticador
    a competir com ele.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)

    disposicao = painel.layout()
    assert isinstance(disposicao, QVBoxLayout)
    posicao = disposicao.indexOf(painel.canvas)
    assert posicao >= 0, "o canvas não está no layout do painel"
    assert disposicao.stretch(posicao) == 1, (
        f"o canvas tem fator de esticamento {disposicao.stretch(posicao)} — a folga não é dele"
    )

    esticadores = [
        i
        for i in range(disposicao.count())
        if (item := disposicao.itemAt(i)) is not None
        and item.spacerItem() is not None
        and item.widget() is None
    ]
    assert esticadores == [], (
        f"há esticador(es) na posição {esticadores} deste layout. Num painel cujo widget dominante "
        "é o canvas, um esticador ao fim fica com o espaço que a figura devia absorver — o defeito "
        "medido no painel do multiverso em 24/08/2026"
    )


def test_o_oraculo_de_desenhado_desce_por_layout_aninhado(qtbot: QtBot) -> None:
    """O instrumento deste arquivo: o veredito acusa o alvo ou o oráculo?

    `QLayout.indexOf` procura só os itens diretos. O par causal é sobre o **mesmo** widget: um
    `indexOf` cru diz `-1` e o oráculo deste painel diz que está na tela — e o segundo é o certo,
    porque o widget é desenhado.
    """
    # **Com registo, e não o painel vazio**, porque no painel vazio o seletor já nasce escondido:
    # a asserção de ausência lá abaixo passaria por vacuidade.
    painel = PainelDeFigura(
        construtores=(_um_construtor(),), fonte=lambda _c: (figura_de_teste(), NOMES)
    )
    qtbot.addWidget(painel)

    disposicao = painel.layout()
    assert isinstance(disposicao, QVBoxLayout)
    assert disposicao.indexOf(painel.seletor) < 0, (
        "âncora: o seletor deixou de estar aninhado, e este par causal deixou de separar nada"
    )
    assert painel.estado_visivel()["seletor"] != "", (
        "o oráculo diz que o seletor NÃO está na tela — o veredito acusaria o "
        "painel quando o defeito está no instrumento, que não desce por layouts aninhados"
    )
    painel.seletor.setVisible(False)
    assert painel.estado_visivel()["seletor"] == "", (
        "o oráculo diz que o seletor está na tela depois de ele ser escondido"
    )
    painel.seletor.setVisible(True)

    # **A metade NEGATIVA do oráculo, e é o mutante contra o qual ele existe:** um widget com pai
    # e **fora** do layout não é desenhado, e `PainelDeQualidade` mediu que cinco mutantes passavam
    # em toda asserção que lesse o valor de retorno em vez do widget.
    orfao = QLabel("nunca entrou no layout", painel)
    assert painel._desenhado(painel.repouso), "âncora: nem o rótulo real conta como desenhado"
    assert not painel._desenhado(orfao), (
        "um rótulo com pai e fora do layout foi lido como desenhado — é o painel em branco que o "
        "oráculo em duas metades existe para acusar"
    )

    # E a metade limpa do par, num layout construído aqui: sem descer, `indexOf` erra.
    hospedeiro = QWidget()
    qtbot.addWidget(hospedeiro)
    pilha = QVBoxLayout(hospedeiro)
    linha = QHBoxLayout()
    rotulo = QLabel("x", hospedeiro)
    linha.addWidget(rotulo)
    pilha.addLayout(linha)
    assert pilha.indexOf(rotulo) < 0 and linha.indexOf(rotulo) == 0, (
        "o Qt passou a procurar recursivamente, e a metade extra deste oráculo ficou sem motivo"
    )


# ---------------------------------------------------------------------------
# A recusa por obsolescência
# ---------------------------------------------------------------------------


def test_a_planta_da_semente_separa_o_que_o_atalho_de_tres_eixos_nao_separa() -> None:
    """A âncora da planta, e ela corre ANTES do par causal do painel.

    Plantar no modo ou no método passaria por coincidência: os dois estão *dentro* do atalho de
    `{modo, método, bytes}` que `api/reuso.py:174-176` recusou. A semente está **fora** dele, e é
    isso que este teste mede — se as duas quantidades não divergissem aqui, o par causal do painel
    passaria sobre uma chave que não separa nada.
    """
    a, b = _corpo(20260910), _corpo(20260911)
    assert _projecao_pobre(a) == _projecao_pobre(b), (
        "os dois corpos já diferem no que um atalho de três eixos leria — a planta deixou de cair "
        "sobre o eixo que ela existe para exercitar"
    )
    assert a["chave_de_invalidacao"] != b["chave_de_invalidacao"], (
        "duas execuções que diferem SÓ na semente receberam a mesma chave: a chave não cobre a "
        "semente, e a figura de uma sobreviveria à outra"
    )
    assert _chave(20260910) == _chave(20260910), (
        "a chave não é função dos argumentos, e toda figura ficaria obsoleta no instante seguinte"
    )


def test_par_causal_da_obsolescencia_a_mesma_chave_MANTEM_e_outra_semente_RETIRA(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """As duas direções, sobre o mesmo painel e a mesma figura — R4.

    **A direção da igualdade é a que impede o portão de ser satisfeito por «limpar sempre».** Um
    painel que descartasse a cada resposta passaria na segunda metade e destruiria o mecanismo: a
    figura piscaria a cada sondagem de progresso.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)

    a = _resposta(200, _corpo(20260910))
    painel.define_execucao(a)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    desenhada = painel.canvas.figure
    assert painel.estado_visivel()["figura"] == "1 eixo(s)", "âncora: nada foi desenhado"

    painel.define_execucao(_resposta(200, _corpo(20260910)))
    assert painel.canvas.figure is desenhada, (
        "a MESMA execução retirou a figura — o retorno antecipado na igualdade não disparou, e a "
        "figura piscaria a cada leitura de progresso"
    )
    assert painel.estado_visivel()["figura"] == "1 eixo(s)"

    texto = painel.define_execucao(_resposta(200, _corpo(20260911)))
    assert painel.canvas.figure is not desenhada, (
        "a figura da execução com a semente 20260910 sobreviveu a uma resposta da 20260911"
    )
    assert painel.estado_visivel()["figura"] == "", "o canvas obsoleto continua na tela"
    assert texto == painel.estado_visivel()["repouso"] != "", (
        "a figura saiu e não entrou texto nenhum no lugar — R4b proíbe espaço em branco"
    )


def test_par_causal_da_ausencia_sem_campo_e_erro_RETIRAM_a_figura_desenhada(qtbot: QtBot) -> None:  # noqa: N802
    """**O par que impede a falha aberta**, e as duas ausências não comparam iguais.

    Uma resposta sem o campo e uma resposta de erro, as duas depois de uma figura desenhada. Com um
    marcador constante para *"não veio chave"*, a segunda ausência seguida compararia igual à
    primeira, o retorno antecipado dispararia, e a figura obsoleta sobreviveria.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    for codigo, corpo, esperado in (
        (200, {"matrix_id": 1}, EstadoDaFigura.SEM_CAMPO),
        (500, {"detail": "boom"}, EstadoDaFigura.FALHOU),
        (202, {"estado": "executando"}, EstadoDaFigura.EM_CURSO),
        # O `410` do cancelamento a pedido. A figura sai como sai nas outras
        # quatro; o que muda é o texto, e é a linha de baixo que o confere.
        (410, {"detail": "cancelado a seu pedido"}, EstadoDaFigura.CANCELADA),
        (200, {"chave_de_invalidacao": 7}, EstadoDaFigura.CAMPO_ILEGIVEL),
    ):
        painel.define_execucao(_resposta(200, _corpo(1)))
        painel.desenha(figura_de_teste(), nomes=NOMES)
        assert painel.estado_visivel()["figura"] == "1 eixo(s)", "âncora: nada foi desenhado"

        painel.define_execucao(_resposta(codigo, corpo))
        visivel = painel.estado_visivel()
        assert visivel["figura"] == "", f"o {codigo} deixou a figura obsoleta na tela"
        assert visivel["repouso"] == texto_do_estado(esperado), (
            f"o {codigo} com corpo {corpo} escreveu o texto de outro estado: {visivel['repouso']!r}"
        )


def test_duas_ausencias_seguidas_NAO_sao_absorvidas_pelo_retorno_antecipado(qtbot: QtBot) -> None:  # noqa: N802
    """A metade que um marcador constante quebraria em silêncio — e ela é observável.

    Duas respostas sem campo, seguidas: a segunda tem de **voltar a agir**. Com um marcador
    constante as duas comparariam iguais e a segunda seria um `return` mudo.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)

    painel.define_execucao(_resposta(200, {"matrix_id": 1}))
    painel.desenha(figura_de_teste(), nomes=NOMES)
    assert painel.estado_visivel()["figura"] == "1 eixo(s)", "âncora: nada foi desenhado"

    painel.define_execucao(_resposta(200, {"matrix_id": 1}))
    assert painel.estado_visivel()["figura"] == "", (
        "a segunda ausência foi absorvida pelo retorno antecipado: as duas compararam iguais, e "
        "uma figura desenhada entre elas sobreviveria"
    )


def test_o_vocabulario_de_estado_cobre_todo_marcador_que_o_cliente_SABE_produzir() -> None:  # noqa: N802
    """O oráculo é a função real e não uma lista de literais.

    Um oráculo escrito a partir dos motivos que eu vi seria cego ao motivo que eu não vi. Aqui os
    marcadores saem de `chave_de_invalidacao` sobre os quatro desfechos que a docstring dela tabela,
    e cada um tem de cair num estado **distinto**.
    """
    chaves = {
        "com chave": chave_de_invalidacao(_resposta(200, {"chave_de_invalidacao": "0" * 64})),
        "sem campo": chave_de_invalidacao(_resposta(200, {"matrix_id": 1})),
        "ilegível": chave_de_invalidacao(_resposta(200, {"chave_de_invalidacao": ""})),
        "em curso": chave_de_invalidacao(_resposta(202, {"estado": "executando"})),
        # O `410` do cancelamento a pedido. Ele entra aqui e não numa lista
        # própria porque a asserção de distinção abaixo é o que prova que ele deixou de cair no
        # mesmo estado que o erro genuíno.
        "cancelado": chave_de_invalidacao(_resposta(410, {"detail": "cancelado a seu pedido"})),
        "erro": chave_de_invalidacao(_resposta(500, {"detail": "boom"})),
    }
    estados = {
        rotulo: estado_da_resposta(chave, ha_construtor=False) for rotulo, chave in chaves.items()
    }
    assert len(set(estados.values())) == len(estados), (
        f"dois desfechos diferentes caíram no mesmo estado: {estados}. Colapsá-los apagaria a "
        "distinção entre «ainda não há» e «nenhuma rota calcula isto»"
    )
    assert estados["em curso"] is EstadoDaFigura.EM_CURSO, estados
    assert estados["cancelado"] is EstadoDaFigura.CANCELADA, estados
    assert estados["erro"] is EstadoDaFigura.FALHOU, estados
    assert estado_da_resposta(CHAVE_DE_REPOUSO, ha_construtor=False) is EstadoDaFigura.SEM_EXECUCAO

    # E a totalidade: todo estado do vocabulário tem frase, e a frase de `DESENHADA` é vazia
    # porque nesse estado quem fala é a figura.
    for estado in EstadoDaFigura:
        assert estado in TEXTO_DO_ESTADO, f"o estado {estado} nasceu sem texto"
        if estado is not EstadoDaFigura.DESENHADA:
            assert texto_do_estado(estado).strip(), f"o estado {estado} tem frase em branco"

    assert motivo_da_chave("0" * 64) == "", "uma chave real foi lida como marcador"
    assert motivo_da_chave(f"{PREFIXO_DE_DIVERGENCIA}sem-campo:abc") == "sem-campo"


# ---------------------------------------------------------------------------
# O CANCELAMENTO deixa de ser falha, e a frase do esvaziamento simultâneo
# ---------------------------------------------------------------------------


def test_u4_a_planta_do_ANTES_o_410_e_o_500_COLAPSAVAM_no_mesmo_estado_e_no_mesmo_texto(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sem esta planta, o motivo do conserto é afirmação sobre o passado sem evidência.

    Dizia-se que *"o código de retorno do cancelamento a pedido cai hoje no estado de falha, e o
    texto diz que a chamada não devolveu resultado"*. Este cenário **mede** esse colapso em vez de
    o citar: com o mapa de motivos como era antes — sem a entrada do `410` —, o cancelamento
    e um `500` genuíno produzem o **mesmo** estado e a **mesma** frase.

    ## A planta cai sobre dado em que as duas coisas DIFEREM

    A metade de cima afirma a divergência **depois**: `410` e `500` dão estados diferentes e
    textos diferentes. Sem essa âncora, a metade de baixo passaria sobre uma implementação que
    nunca distinguiu nada, e o par causal seria vácuo.

    **A planta é em processo** (o mapa é argumento efectivo da função sob teste, pela ligação de
    módulo), então não há o caminho de inércia da tabela de triagem: ou ela aterra, ou o
    `monkeypatch` levanta. Ainda assim a âncora de que ela aterrou é afirmada antes de o desfecho
    ser lido.
    """
    cancelado = chave_de_invalidacao(_resposta(410, {"detail": "cancelado a seu pedido"}))
    erro = chave_de_invalidacao(_resposta(500, {"detail": "boom"}))

    # DEPOIS — e é esta metade que torna a de baixo não vácua.
    depois_cancelado = estado_da_resposta(cancelado, ha_construtor=True)
    depois_erro = estado_da_resposta(erro, ha_construtor=True)
    assert depois_cancelado is EstadoDaFigura.CANCELADA
    assert depois_erro is EstadoDaFigura.FALHOU
    assert texto_do_estado(depois_cancelado) != texto_do_estado(depois_erro), (
        "os dois estados existem e escrevem a mesma frase — a distinção morreu no texto"
    )

    # ANTES — o mapa como era, sem a entrada do cancelamento.
    como_era = {
        motivo: estado
        for motivo, estado in modulo_das_figuras._ESTADO_DO_MOTIVO.items()
        if estado is not EstadoDaFigura.CANCELADA
    }
    assert len(como_era) == len(modulo_das_figuras._ESTADO_DO_MOTIVO) - 1, (
        "a planta não tirou nada do mapa: ela não aterrou, e o desfecho abaixo seria o de sempre"
    )
    monkeypatch.setattr(modulo_das_figuras, "_ESTADO_DO_MOTIVO", como_era)
    assert EstadoDaFigura.CANCELADA not in modulo_das_figuras._ESTADO_DO_MOTIVO.values()

    antes_cancelado = estado_da_resposta(cancelado, ha_construtor=True)
    antes_erro = estado_da_resposta(erro, ha_construtor=True)
    assert antes_cancelado is antes_erro is EstadoDaFigura.FALHOU, (
        "o colapso que KTD11 descreve não se reproduz: a planta mede outra coisa"
    )
    assert texto_do_estado(antes_cancelado) == texto_do_estado(antes_erro)


def test_u4_o_410_desenha_CANCELAMENTO_e_o_texto_nao_diz_que_a_chamada_falhou(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """R15: análise cancelada pelo usuário é desenhada como cancelamento, nunca como falha.

    **O oráculo é o widget e não o valor de retorno** — `estado_visivel` lê o rótulo e confere que
    ele está no layout e visível.

    **A âncora de que havia figura ANTES** é o que impede o cenário de afirmar o esvaziamento
    sobre um painel que nunca desenhou nada: `"1 eixo(s)"` é asserção de presença.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.define_execucao(_resposta(200, _corpo(1)))
    painel.desenha(figura_de_teste(), nomes=NOMES)
    assert painel.estado_visivel()["figura"] == "1 eixo(s)", "âncora: nada foi desenhado"

    painel.define_execucao(_resposta(410, {"detail": "o trabalho foi CANCELADO a seu pedido"}))
    visivel = painel.estado_visivel()

    assert visivel["repouso"] == texto_do_estado(EstadoDaFigura.CANCELADA)
    assert visivel["figura"] == "", "a figura da execução cancelada ficou na tela"
    assert visivel["repouso"] != texto_do_estado(EstadoDaFigura.FALHOU), (
        "o cancelamento voltou a escrever a frase da falha — é o colapso que a U4 desfez"
    )
    # E o conteúdo, que é o que R15 pede: a frase não pode dizer que a chamada falhou nem mandar
    # conferir coisa nenhuma. Ela nomeia o autor da parada.
    assert "VOCÊ cancelou" in visivel["repouso"], visivel["repouso"]
    assert "não devolveu resultado" not in visivel["repouso"], visivel["repouso"]


def test_u4_estado_sem_entrada_no_mapa_de_textos_LEVANTA(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**A asserção é sobre o LEVANTAMENTO**, e é ela que prova que estado novo não nasce mudo.

    `texto_do_estado` usa indexação e não `.get` com recuo de propósito: um recuo faria um estado
    novo nascer com frase vazia, e vazio na tela lê-se como *"não há nada a dizer"*.

    **Par causal, e a direção que falta é a do sucesso:** primeiro com a entrada presente — devolve
    frase não vazia —, depois sem ela. Sem a primeira metade, um `texto_do_estado` que levantasse
    para **todo** estado passaria neste teste.
    """
    alvo = EstadoDaFigura.CANCELADA
    assert texto_do_estado(alvo).strip(), "âncora: o estado alvo já nasceria mudo sem a planta"

    sem_texto = {estado: frase for estado, frase in TEXTO_DO_ESTADO.items() if estado is not alvo}
    assert alvo not in sem_texto, "a planta não aterrou no mapa"
    monkeypatch.setattr(modulo_das_figuras, "TEXTO_DO_ESTADO", sem_texto)
    assert alvo not in modulo_das_figuras.TEXTO_DO_ESTADO

    with pytest.raises(KeyError):
        texto_do_estado(alvo)


def test_u4_a_frase_do_EM_CURSO_diz_que_TODAS_as_figuras_saem_juntas() -> None:  # noqa: N802
    """A tela diz **por que** o painel esvaziou.

    A chave de invalidação é escalar na raiz, então mudar um parâmetro de uma análise retira todas
    as figuras — a PCA e o volcano inclusive, que nada têm com ela. **Sem frase visível, o
    esvaziamento simultâneo lê-se como o aplicativo a partir.**

    **A cláusula é LIDA do módulo e não redigitada aqui**: uma cópia da prosa neste teste seria a
    segunda fonte da mesma verdade, e ficaria verde no dia em que a frase da tela mudasse.

    **E o escopo é afirmado nos dois sentidos**: o `EM_CURSO` a carrega porque é o estado que uma
    execução nova produz; o `SEM_EXECUCAO` não, porque ali nunca houve figura a sair.
    """
    assert POR_QUE_TODAS_SAEM.strip(), "a cláusula é vazia — as asserções abaixo seriam vácuas"
    assert POR_QUE_TODAS_SAEM in texto_do_estado(EstadoDaFigura.EM_CURSO)
    assert POR_QUE_TODAS_SAEM not in texto_do_estado(EstadoDaFigura.SEM_EXECUCAO), (
        "a cláusula entrou num estado em que nunca houve figura a retirar"
    )


def test_a_chave_de_repouso_nao_colide_com_nada_que_o_cliente_produza() -> None:
    """«Nasci e nada chegou» é um fato próprio, e não pode ser confundido com uma ausência."""
    assert not CHAVE_DE_REPOUSO.startswith(PREFIXO_DE_DIVERGENCIA)
    assert len(CHAVE_DE_REPOUSO) != 64 or not all(c in "0123456789abcdef" for c in CHAVE_DE_REPOUSO)


# ---------------------------------------------------------------------------
# O que a figura mostra — pelas quatro classes do arnês
# ---------------------------------------------------------------------------


def test_a_figura_desenhada_passa_pelas_quatro_classes_do_arnes(qtbot: QtBot) -> None:
    """As quatro classes do arnês sobre o que o painel de facto pôs na tela.

    Os números da classe (d) vêm **do teste**, derivados dos literais desta suíte, e não do
    `Recorte` que o construtor usou: se o teste chamasse o mesmo objeto, os dois errariam juntos.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)

    assert confere_series(eixo, {"antes": (X_A, Y_A), "depois": (X_B, Y_B)}) == []
    assert (
        confere_rotulos(
            eixo,
            x="ordem de injeção",
            y="intensidade",
            titulo="figura de arnês",
            legenda=("antes", "depois"),
        )
        == []
    )
    assert confere_vista(eixo) == []
    esperados = [str(DESENHADAS), "6.035"]
    assert confere_numeros(painel.canvas.figure, esperados) == [], (
        f"a figura declara números que o teste não derivou: "
        f"{confere_numeros(painel.canvas.figure, esperados)}"
    )


def test_a_planta_do_corte_de_eixo_ACUSA_na_figura_deste_painel(qtbot: QtBot) -> None:  # noqa: N802
    """A segunda direção do arnês, sobre esta figura — sem ela o teste acima é um portão inerte.

    `planta_corte_de_eixo` traz a sua âncora dentro dela: levanta se o `y` desenhado não tiver
    extensão. Aqui tem, e o corte põe a figura em branco **sem mudar dado nenhum**.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)
    assert confere_vista(eixo) == [], "âncora: a figura já reprovava antes da planta"

    limite = planta_corte_de_eixo(eixo)
    achados = confere_vista(eixo)
    assert achados, f"a planta {limite} não foi acusada — o portão da classe (c) está inerte"
    assert confere_series(eixo, {"antes": (X_A, Y_A), "depois": (X_B, Y_B)}) == [], (
        "a planta mudou dado, e devia mudar só a vista — a atribuição da reprovação deixaria de "
        "ser dela"
    )


# ---------------------------------------------------------------------------
# R2 — identificar o ponto, e voltar à vista original
# ---------------------------------------------------------------------------


def _dispara_o_pick(painel: PainelDeFigura, eixo: Axes, x: float, y: float) -> None:
    """Pica no ponto de dado `(x, y)`, pelo caminho real: coordenadas de tela e `Figure.pick`."""
    painel.canvas.draw()
    tela_x, tela_y = eixo.transData.transform((x, y))
    painel.canvas.figure.pick(MouseEvent("button_press_event", painel.canvas, tela_x, tela_y, 1))


def test_a_identificacao_diz_qual_feature_e_e_limpa_a_anterior(qtbot: QtBot) -> None:
    """R2: identificar é **dizer qual feature é**, e a anotação anterior sai."""
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)

    _dispara_o_pick(painel, eixo, X_A[1], Y_A[1])
    primeira = painel.estado_visivel()["anotacao"]
    assert NOMES[1] in primeira, f"a anotação não nomeia a feature: {primeira!r}"

    _dispara_o_pick(painel, eixo, X_A[3], Y_A[3])
    segunda = painel.estado_visivel()["anotacao"]
    assert NOMES[3] in segunda and NOMES[1] not in segunda, (
        f"a anotação anterior ficou na tela: {segunda!r}. Duas anotações ao mesmo tempo dizem que "
        "dois pontos foram identificados, e só um foi"
    )
    textos = [t.get_text() for t in eixo.texts]
    assert len(textos) == 1, f"há {len(textos)} anotações no eixo: {textos}"


def test_d437_a_anotacao_escreve_o_numero_com_VIRGULA(qtbot: QtBot) -> None:  # noqa: N802
    """A anotação segue a convenção decimal do resto da figura.

    ## O defeito, e ele é de CONVENÇÃO e não de valor

    A anotação saía de um `f"{x:.4g}"` — ponto decimal —, encostada a marcas de eixo que
    `plots.base.virgula_decimal` já convertera para **vírgula**. E o rodapé do recorte escreve
    `5.942` para cinco mil novecentos e quarenta e dois: na mesma imagem, a dois centímetros de
    distância, o mesmo símbolo significava separador decimal e separador de milhar.

    > **Nenhuma classe de defeito numérico o alcança, porque o número está CERTO.** O que está
    > errado é a convenção com que ele é escrito — e é a mesma observação que fez `recorte.decimal`
    > existir, aqui aplicada ao único texto que o painel escreve dentro da figura.

    ## A FIXTURA É PRÓPRIA, e a primeira versão deste teste era INERTE por causa disso

    `figura_de_teste` desenha em coordenadas **inteiras** — `X_A = (1.0, 2.0, 3.0, 4.0)` —, e
    `f"{2.0:.4g}"` é `"2"`: **sem separador nenhum**. A primeira versão afirmava *«não há ponto»*
    sobre isso e passava com a planta aterrada, conferida no arquivo. É um critério trivialmente
    satisfeito, cometido no teste escrito para fechar o defeito.

    **Aqui o ponto tem parte fraccionária**, e a asserção passa a separar as duas implementações:
    com `:.4g` sai `1.523`, com `decimal` sai `1,523`.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    figura = figura_de_teste()
    eixo_da_figura = figura.axes[0]
    # Uma série própria, com coordenadas que TÊM decimal — e marcada como portadora de features,
    # que é o que o painel exige para nomear o ponto.
    eixo_da_figura.scatter([1.523], [2.847], gid=GID_DAS_FEATURES)
    painel.desenha(figura, nomes=("feature-fraccionaria",))
    eixo = _eixo(painel)

    _dispara_o_pick(painel, eixo, 1.523, 2.847)
    anotacao = painel.estado_visivel()["anotacao"]

    # **A âncora, e ela é sobre a PARTE FRACCIONÁRIA e não sobre haver dígito.** Sem um separador
    # na anotação, a asserção de baixo passa por vacuidade — foi o que aconteceu na primeira versão.
    assert "523" in anotacao, (
        f"a anotação não traz o valor fraccionário, e a asserção abaixo seria vácua: {anotacao!r}"
    )
    assert "1,523" in anotacao, (
        f"a anotação escreve o decimal com PONTO, ao lado de marcas de eixo em vírgula e de um "
        f"rodapé que usa o ponto como separador de MILHAR: {anotacao!r}"
    )


def test_a_identificacao_SOBREVIVE_a_troca_de_figura(qtbot: QtBot) -> None:  # noqa: N802
    """**O registo de eventos é da FIGURA e não do canvas — e este defeito esteve escrito.**

    `canvas.callbacks` devolve `figura._canvas_callbacks`: trocar a figura troca o registo, e toda
    ligação feita por `mpl_connect` sobre a figura anterior desaparece **sem erro e sem aviso**. A
    primeira versão deste painel ligava a identificação uma vez, no construtor, e a linha seguinte
    punha a figura de repouso no canvas — a identificação nascia morta em todo painel, e o único
    sintoma era o cursor não responder.

    **O par causal é a segunda e a terceira figura**, e não a primeira: um painel que só ligasse na
    construção passaria numa asserção feita antes da primeira troca.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    registos = []

    for volta in range(3):
        painel.desenha(figura_de_teste(), nomes=NOMES)
        registos.append(id(painel.canvas.callbacks))
        _dispara_o_pick(painel, _eixo(painel), X_A[volta], Y_A[volta])
        assert NOMES[volta] in painel.estado_visivel()["anotacao"], (
            f"na volta {volta} a identificação deixou de responder — a ligação ficou na figura "
            "anterior"
        )
        painel.define_execucao(_resposta(500, {"detail": "boom"}))

    assert len(set(registos)) == 3, (
        f"âncora: as três figuras partilharam registo de eventos ({registos}), e este par causal "
        "não separa nada — o `matplotlib` deixou de guardar o registo na figura"
    )


def test_o_ponto_SOBREPOSTO_devolve_o_de_cima_e_o_teste_diz_qual_nome(qtbot: QtBot) -> None:  # noqa: N802
    """A borda: dois pontos no mesmo lugar, e o que fica anotado é o desenhado por último.

    ## A planta tem de cair onde as duas coisas DIFEREM

    A primeira versão sobrepunha o índice **0** do artista de baixo com o índice **0** do de cima,
    e a anotação nomeia a feature **pelo índice**: os dois desfechos escreviam a mesma frase, e o
    teste passaria qualquer que fosse o artista que ganhasse. Aqui o ponto comum é o índice `0` de
    baixo e o índice `1` de cima, então o nome escrito **diz qual dos dois ganhou**.

    **As duas âncoras vêm antes:** os pontos têm de coincidir, e os índices têm de diferir.
    """
    figura = nova_figura()
    eixo = figura.add_subplot()
    # As duas são NOMEADAS, e é o caso plural da marca: dois artistas cujos pontos são indexados
    # pelo mesmo vetor de nomes. Sem as duas marcadas o painel cairia no rótulo da série, e este
    # teste deixaria de poder dizer **qual feature** ganhou a sobreposição.
    de_baixo = eixo.scatter([1.0, 2.0], [1.0, 2.0], label="baixo")
    de_cima = eixo.scatter([5.0, 1.0], [5.0, 1.0], label="cima")
    for artista in (de_baixo, de_cima):
        artista.set_gid(GID_DAS_FEATURES)
    eixo.set_xlabel("x")
    eixo.set_ylabel("y")
    eixo.set_title("sobreposto")
    nomes = ("o índice zero", "o índice um")

    desenhadas = series(eixo)
    assert len(desenhadas) == 2, (
        f"âncora: {len(desenhadas)} série(s), e a sobreposição precisa de 2"
    )
    assert (desenhadas[0].x[0], desenhadas[0].y[0]) == (desenhadas[1].x[1], desenhadas[1].y[1]), (
        "âncora: os dois pontos não coincidem, e esta não é a borda do ponto sobreposto"
    )
    assert nomes[0] != nomes[1], (
        "âncora: os dois índices dariam o mesmo nome, e a asserção abaixo passaria qualquer que "
        "fosse o artista que ganhasse — a planta não separaria nada"
    )

    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura, nomes=nomes)

    vistos: list[str] = []
    painel.canvas.mpl_connect("pick_event", lambda e: vistos.append(str(e.artist.get_label())))
    _dispara_o_pick(painel, _eixo(painel), 1.0, 1.0)
    assert vistos == ["baixo", "cima"], (
        f"âncora: o `matplotlib` avisou {vistos} — se só um artista fosse alcançado, não haveria "
        "sobreposição a resolver"
    )

    anotacao = painel.estado_visivel()["anotacao"]
    assert anotacao, "nenhum dos dois artistas sobrepostos foi identificado"
    assert nomes[1] in anotacao and nomes[0] not in anotacao, (
        f"o ponto identificado não é o de cima: {anotacao!r}. O `matplotlib` dispara um evento por "
        "artista na ordem de desenho, e é o último a chegar que tem de ficar"
    )


def test_voltar_a_vista_original_depois_de_aproximar_e_deslocar(qtbot: QtBot) -> None:
    """O caminho de volta, agora depois de um zoom REAL.

    **Até 15/09/2026 este teste forjava o zoom com `set_xlim`**, porque o zoom não existia: o módulo
    afirmava que o `matplotlib` 3.11 o trazia, e a medição mostrou que não. O teste passava, e o que
    ele provava era que `set_xlim` seguido de `_volta_a_vista_original` devolve os limites — uma
    propriedade verdadeira sobre um gesto que nenhum profissional conseguia fazer.

    **Agora a aproximação vem da roda**, que é o que o usuário faz, e o botão desfaz o que a roda
    fez: o alvo certo não era `set_xlim`.

    **Os limites de chegada são conferidos pelo arnês e não contra literais**: `confere_vista`
    afirma que tudo o que está desenhado cabe dentro deles, e o que está desenhado ele lê da figura.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)
    original = (eixo.get_xlim(), eixo.get_ylim())

    for _ in range(12):
        _roda(painel, eixo, 2.0, 5.0, 1, ctrl=True)
    assert (eixo.get_xlim(), eixo.get_ylim()) != original, (
        "âncora: a roda não aproximou, e o retorno não teria o que desfazer"
    )
    assert confere_vista(eixo) != [], (
        "âncora: aproximar não excluiu ponto nenhum, e o retorno não teria o que provar"
    )

    painel.botao_da_vista.click()
    assert (eixo.get_xlim(), eixo.get_ylim()) == original, (
        f"os limites voltaram a {(eixo.get_xlim(), eixo.get_ylim())} e não a {original}"
    )
    assert confere_vista(eixo) == [], "a vista «original» não contém o que está desenhado"


def test_a_vista_original_NAO_e_recalculada_e_por_isso_ignora_a_anotacao(qtbot: QtBot) -> None:  # noqa: N802
    """Um `autoscale` no retorno faria a vista «original» depender de onde o profissional clicou."""
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)
    original = (eixo.get_xlim(), eixo.get_ylim())

    _dispara_o_pick(painel, eixo, X_A[3], Y_A[3])
    assert painel.estado_visivel()["anotacao"], "âncora: nenhuma anotação foi criada"
    eixo.set_xlim(1.9, 2.1)
    painel.botao_da_vista.click()

    assert (eixo.get_xlim(), eixo.get_ylim()) == original
    assert painel.estado_visivel()["anotacao"] == "", "o retorno à vista não limpou a anotação"


@pytest.mark.parametrize("gesto", ["identificar", "voltar"])
def test_com_o_widgetlock_TOMADO_o_manipulador_retorna_sem_efeito(qtbot: QtBot, gesto: str) -> None:  # noqa: N802
    """A PROPRIEDADE de ponta a ponta: com o cadeado tomado, o gesto não muda nada.

    ## O que este teste NÃO exercita, e dizê-lo é o que o impede de ser vácuo

    Para `identificar`, quem impede o efeito **não é a guarda deste painel**: `Figure.pick` abre com
    `if not self.canvas.widgetlock.locked()` (`matplotlib/figure.py:2666`), medido em 10/09/2026 com
    o manipulador instrumentado — **ele nem chega a ser chamado**. A cobertura o disse antes de a
    leitura o confirmar: a linha do `return` da guarda ficava por percorrer com este teste a passar.

    O veredito «passou» era sobre o alvo certo e vinha do mecanismo errado. A
    propriedade fica afirmada aqui, que é o que o profissional vê; a **guarda** tem par causal
    próprio, em processo, em
    `test_a_guarda_do_widgetlock_no_manipulador_da_identificacao_tem_as_duas_direcoes`.

    Para `voltar`, o clique chega ao slot por Qt e a guarda **é** de facto exercitada — os dois
    gestos ficam no mesmo teste porque a propriedade é a mesma, e a diferença de mecanismo é o que
    esta nota existe para não deixar suposta.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)
    if gesto == "voltar":
        eixo.set_xlim(1.9, 2.1)
    antes = (eixo.get_xlim(), eixo.get_ylim(), painel.estado_visivel()["anotacao"])

    dono = object()
    painel.canvas.widgetlock(dono)
    if gesto == "identificar":
        _dispara_o_pick(painel, eixo, X_A[1], Y_A[1])
    else:
        painel.botao_da_vista.click()
    tomado = (eixo.get_xlim(), eixo.get_ylim(), painel.estado_visivel()["anotacao"])
    assert tomado == antes, f"o gesto «{gesto}» agiu com o canvas tomado: {antes} -> {tomado}"

    painel.canvas.widgetlock.release(dono)
    if gesto == "identificar":
        _dispara_o_pick(painel, eixo, X_A[1], Y_A[1])
    else:
        painel.botao_da_vista.click()
    livre = (eixo.get_xlim(), eixo.get_ylim(), painel.estado_visivel()["anotacao"])
    assert livre != antes, (
        f"o gesto «{gesto}» não agiu nem com o canvas livre — a metade de cima seria vácua"
    )


# ----------------------------------------------------------------------
# O ZOOM POR Ctrl+RODA
# ----------------------------------------------------------------------


def _roda(
    painel: PainelDeFigura, eixo: Axes, x: float, y: float, passo: float, *, ctrl: bool
) -> None:
    """Despacha um `scroll_event` sobre o ponto de dado `(x, y)`, com ou sem Ctrl.

    **Pelo `_process()` e não chamando `_amplia` à mão**, porque é exactamente o que
    `FigureCanvasQT.wheelEvent` faz (`backend_qt.py:370-373`) — despachar um `MouseEvent` de
    `scroll_event` com `modifiers=` preenchido por `_mpl_modifiers()`.

    **`modifiers` e não `key`, e a razão é medida** — ver
    `test_u12_o_canvas_embutido_nao_recebe_key_e_e_por_isso_que_se_le_modifiers`.
    """
    painel.canvas.draw()
    tela_x, tela_y = eixo.transData.transform((x, y))
    MouseEvent(
        "scroll_event",
        painel.canvas,
        tela_x,
        tela_y,
        step=passo,
        modifiers=["ctrl"] if ctrl else [],
    )._process()


def _largura(eixo: Axes) -> float:
    return float(eixo.get_xlim()[1] - eixo.get_xlim()[0])


def test_u12_par_causal_do_zoom_COM_ctrl_aproxima_e_SEM_ctrl_nao_mexe(qtbot: QtBot) -> None:  # noqa: N802
    """O par causal do gesto, e as duas direções sobre a MESMA figura e o mesmo ponto.

    **A metade «sem Ctrl» é requisito e não simetria decorativa:** a aba «Qualidade do resultado»
    rola, e um zoom armado na roda nua roubaria o gesto de rolar a quem trocasse de aba sem pensar.

    **A âncora contra a vacuidade vai na MESMA função e ANTES**: sem afirmar que o par
    diverge, as duas metades passariam sobre um manipulador que nunca corre.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    # **A SEGUNDA causa da queixa «carreguei e não aconteceu nada», medida na mesma montagem.**
    # Em repouso o botão estava vivo sobre uma figura de zero eixos. As asserções vivem AQUI e não
    # num teste próprio por um motivo medido em 15/09/2026: acrescentar um nono teste que constrói
    # painel e desenha leva este arquivo de 0/3 a 3/3 quebras — ver `docs/linha-de-base-windows.md`.
    assert not painel.botao_da_vista.isEnabled(), (
        "em repouso o botão de voltar à vista está habilitado, sobre uma figura de zero eixos"
    )
    painel.desenha(figura_de_teste(), nomes=NOMES)
    assert painel.botao_da_vista.isEnabled(), (
        "com figura desenhada o botão ficou desabilitado — a metade de cima seria vácua"
    )
    eixo = _eixo(painel)
    original = (eixo.get_xlim(), eixo.get_ylim())

    _roda(painel, eixo, 2.0, 5.0, 1, ctrl=False)
    sem_ctrl = (eixo.get_xlim(), eixo.get_ylim())
    assert sem_ctrl == original, (
        f"a roda NUA mexeu nos limites: {original} -> {sem_ctrl}. A aba de qualidade rola, e o "
        f"hábito da roda é o mesmo nas duas abas"
    )

    _roda(painel, eixo, 2.0, 5.0, 1, ctrl=True)
    com_ctrl = (eixo.get_xlim(), eixo.get_ylim())
    assert com_ctrl != original, (
        "Ctrl+roda não mexeu em nada. Até 15/09/2026 este módulo AFIRMAVA que o `matplotlib` 3.11 "
        "trazia este zoom, e a medição mostrou que não — ver o cabeçalho de `figuras.py`"
    )
    assert _largura(eixo) < original[0][1] - original[0][0], (
        f"roda para CIMA tem de APROXIMAR, e a largura foi de {original[0][1] - original[0][0]} "
        f"para {_largura(eixo)}"
    )

    painel._retira(EstadoDaFigura.SEM_CAMPO)
    assert not painel.botao_da_vista.isEnabled(), (
        "a retirada da figura deixou o botão vivo — `_anuncia` não cobre este ramo"
    )


def test_u12_o_zoom_ancora_no_CURSOR_e_nao_no_centro_do_eixo(qtbot: QtBot) -> None:  # noqa: N802
    """A fração do eixo em que o cursor está tem de ficar a MESMA depois de aproximar.

    **É isto que separa «zoom» de «zoom no centro»**, e a diferença é o gesto inteiro: sem âncora no
    cursor, aproximar um canto de um volcano denso exigiria arrastar — e não há arrastar aqui.

    **O ponto escolhido NÃO é o centro**, e a asserção de que não é vai antes: ancorar no centro
    daria a mesma fração e o teste passaria sem medir nada.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)
    baixo, cima = eixo.get_xlim()
    alvo = 3.5
    fracao_antes = (alvo - baixo) / (cima - baixo)
    assert abs(fracao_antes - 0.5) > 0.1, (
        f"âncora: o ponto escolhido está a {fracao_antes:.3f} do eixo, perto demais do centro — "
        f"ancorar no centro daria o mesmo resultado e este teste seria vácuo"
    )

    _roda(painel, eixo, alvo, 5.0, 1, ctrl=True)
    baixo_dep, cima_dep = eixo.get_xlim()
    fracao_depois = (alvo - baixo_dep) / (cima_dep - baixo_dep)
    assert fracao_depois == pytest.approx(fracao_antes), (
        f"o ponto sob o cursor moveu-se: estava a {fracao_antes:.6f} do eixo e ficou a "
        f"{fracao_depois:.6f}. O zoom está a ancorar noutro sítio"
    )


def test_u12_afastar_DESFAZ_aproximar_e_a_roda_tem_as_duas_direcoes(qtbot: QtBot) -> None:  # noqa: N802
    """Um entalhe para cada lado, no mesmo ponto, devolve a vista. É a segunda direção do gesto."""
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)
    original = (eixo.get_xlim(), eixo.get_ylim())

    _roda(painel, eixo, 2.0, 5.0, 1, ctrl=True)
    aproximado = _largura(eixo)
    assert aproximado < original[0][1] - original[0][0], "âncora: não aproximou"

    _roda(painel, eixo, 2.0, 5.0, -1, ctrl=True)
    assert _largura(eixo) > aproximado, "a roda para BAIXO não afastou"
    assert eixo.get_xlim() == pytest.approx(original[0]), (
        f"um entalhe para cada lado não devolveu a vista: {original[0]} -> {eixo.get_xlim()}"
    )


def test_u12_a_ANOTACAO_do_clique_sobrevive_ao_zoom(qtbot: QtBot) -> None:  # noqa: N802
    """Uma decisão de desenho, e ela é o oposto da do botão de voltar — ver `_amplia`.

    A anotação está ancorada em **coordenadas de dado**, então o zoom leva-a consigo. Apagá-la a
    cada entalhe mataria o gesto que motivou o pedido: identificar uma feature num volcano denso e
    **depois** aproximar para ver a vizinhança dela.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)
    _dispara_o_pick(painel, eixo, X_A[3], Y_A[3])
    anotada = painel.estado_visivel()["anotacao"]
    assert anotada, "âncora: nenhuma anotação foi criada, e a asserção abaixo seria vácua"

    _roda(painel, eixo, X_A[3], Y_A[3], 1, ctrl=True)

    assert painel.estado_visivel()["anotacao"] == anotada, (
        "o zoom apagou a anotação. Ela é o contrário do botão de voltar à vista: aproximar é "
        "«continua a explorar», e a anotação é parte do que se está a explorar"
    )


@pytest.mark.parametrize("tomado", [True, False])
def test_u12_a_guarda_do_widgetlock_no_zoom_nas_DUAS_direcoes(qtbot: QtBot, tomado: bool) -> None:  # noqa: N802
    """Par causal da guarda: com o cadeado tomado não mexe, com ele livre mexe.

    **Aqui a guarda É de facto exercitada**, ao contrário do que acontece com `_identifica`:
    `Figure.pick` recusa o despacho com o cadeado tomado, mas `scroll_event` não passa por
    `Figure.pick` — chega ao manipulador, e quem o para é esta guarda.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)
    antes = (eixo.get_xlim(), eixo.get_ylim())

    dono = object()
    if tomado:
        painel.canvas.widgetlock(dono)
    _roda(painel, eixo, 2.0, 5.0, 1, ctrl=True)
    mexeu = (eixo.get_xlim(), eixo.get_ylim()) != antes

    assert mexeu is not tomado, (
        f"com o cadeado {'TOMADO' if tomado else 'LIVRE'} o zoom "
        f"{'agiu' if mexeu else 'não agiu'} — o contrário do esperado"
    )


def test_u12_o_canvas_embutido_nao_recebe_key_e_e_por_isso_que_se_le_modifiers(
    qtbot: QtBot,
) -> None:
    """A medição que DECIDIU ler `evento.modifiers` em vez de `evento.key` — 15/09/2026.

    `backend_bases.scroll_handler` compara `evento.key`, que o `matplotlib` enche a partir de
    `canvas._key` — escrito só por `key_press_event`, que **exige foco de teclado no canvas**. Num
    canvas embutido não há manager, logo não há o `setFocusPolicy(StrongFocus)` que
    `FigureManagerQT.__init__` faz (`backend_qt.py:618`).

    **Sem esta medição, um manipulador escrito contra `evento.key` nunca dispararia** — e o desfecho
    seria o silêncio: nada levanta, a roda apenas não faz nada.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)

    assert painel.canvas.focusPolicy() == Qt.FocusPolicy.NoFocus, (
        "o canvas ganhou política de foco. Se isso for deliberado, a razão de se ler `modifiers` "
        "em vez de `key` mudou e a docstring de `_amplia` precisa de ser remedida"
    )
    assert painel.canvas.manager is None, "o canvas embutido ganhou um manager"
    assert painel.canvas._key is None, "âncora: `canvas._key` já vem preenchido"


def test_u12_sem_o_HistoricoMudo_o_scroll_handler_do_matplotlib_RECUSA(qtbot: QtBot) -> None:  # noqa: N802
    """Par causal de `HistoricoMudo`: ele é carga, e não cerimônia.

    `scroll_handler` abre com `if toolbar is None: return`, e o comentário do próprio `matplotlib`
    diz que a restrição é **histórica e não técnica**. Este teste planta a ausência do toolbar e
    afirma que o desfecho DIVERGE — sem ele, `HistoricoMudo` poderia ser removido sem que nada
    acusasse.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)
    painel.canvas.draw()
    tela_x, tela_y = eixo.transData.transform((2.0, 5.0))
    evento = MouseEvent("scroll_event", painel.canvas, tela_x, tela_y, step=1, modifiers=["ctrl"])
    evento.key = TECLA_DO_ZOOM

    antes = (eixo.get_xlim(), eixo.get_ylim())
    scroll_handler(evento, toolbar=None)
    assert (eixo.get_xlim(), eixo.get_ylim()) == antes, (
        "`scroll_handler` aproximou SEM toolbar — a porta que `HistoricoMudo` existe para abrir "
        "deixou de estar fechada, e a classe passou a ser cerimônia"
    )

    scroll_handler(evento, toolbar=HistoricoMudo())
    assert (eixo.get_xlim(), eixo.get_ylim()) != antes, (
        "com `HistoricoMudo` também não aproximou — a planta não separa nada"
    )


def test_u12_nenhum_dos_eixos_das_figuras_REGISTADAS_e_logaritmico(qtbot: QtBot) -> None:  # noqa: N802
    """A medição que diz que a correção em escala de `scroll_handler` não é necessária HOJE.

    Ela vem junto e não custa — a conta do `matplotlib` passa por `ax.transScale` —, e este teste
    existe para que a afirmação *"nenhum é logarítmico"* deixe de ser leitura de `grep` e passe a
    ser medida sobre as figuras montadas.

    **Se algum construtor futuro usar escala logarítmica, este teste reprova** — e o desfecho certo
    aí é apagar a frase «não é necessária hoje» de `_amplia`, não afrouxar o teste.

    > Os DOIS `recebe_*` do painel da correção são alimentados aqui, e não é
    > arrumação. Este é o teste que exige eixo em **todos** os montadores, e a figura nova é a
    > única que lê duas cargas de rotas diferentes: sem o par de PCAs ela recusaria por população
    > vazia, o eixo não existiria, e a âncora `assert eixos` reprovaria a apontar para a fixtura em
    > vez de para a escala. **A tentação da fila de barras é real** — as barras vão de 1,9e-15 a 1,4
    > conforme o estimador —, então alimentar a medição é o que põe o eixo mais tentado de todos
    > dentro desta varredura.

    > E `recebe_rsd_por_feature` também é alimentado aqui, pela mesma razão e com a mesma
    > tentação. O
    > violino desenha RSD por feature, e sob o estimador in-sample a mediana do `batch_loess` é
    > 1,9e-15 no `pos_c18`: é o segundo eixo mais tentado do registo. Sem a carga, o construtor
    > recusaria por população vazia e a âncora `assert eixos` reprovaria a apontar para a fixtura.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_diagnostico(_com_pca())
    graficos.recebe_pca(_com_pca())
    graficos.recebe_distribuicoes(
        {**_corpo(5), "distribuicoes": dict(RESUMO_DE_TESTE)}, corresponde=True
    )
    graficos.recebe_volcano(_com_volcano(11))
    graficos.recebe_plsda(_com_plsda(7))
    graficos.recebe_pca_do_par(_com_par_de_pcas(5), corresponde=True)
    graficos.recebe_rsd_por_feature(_com_rsd_por_feature(5), corresponde=True)
    graficos.recebe_composicao(_com_composicao(5), corresponde=True)
    graficos.recebe_medicao(_medicao())
    # **E a ANOVA — merge do bloco das análises, pela mesma razão das duas notas acima.** Sem a
    # carga o construtor recusa por população vazia e a âncora reprovaria a apontar para a fixtura
    # em vez de para a escala.
    graficos.recebe_anova(_com_anova(12))
    # E as DUAS da correlação, pela mesma razão das notas acima. Sem a carga os dois
    # construtores recusam por população vazia e a âncora reprovaria a apontar para a fixtura em vez
    # de para a escala. **A tentação aqui é real:** o coeficiente vive em `[-1, 1]` e a busca
    # desenha
    # barras a partir de zero, que é a faixa em que um eixo logarítmico é impossível — é o caso em
    # que a varredura confirma em vez de acusar, e isso vale tanto quanto o contrário.
    graficos.recebe_correlacao(_com_correlacao(13))
    # E as DUAS particionais, pela mesma razão das notas acima. Sem a carga os dois
    # construtores recusam por população vazia e a âncora reprovaria a apontar para a fixtura em
    # vez de para a escala. **A tentação aqui é do eixo y do K-means:** ele conta amostras, e uma
    # contagem é a grandeza que mais convida a escala logarítmica quando os agrupamentos são muito
    # desiguais — é o caso em que a varredura confirma em vez de acusar.
    graficos.recebe_agrupamento_particional(_com_agrupamento_particional(17))
    # E as DUAS de classificação, pela mesma razão das notas acima. Sem a carga os dois
    # construtores recusam por população vazia e a âncora reprovaria a apontar para a fixtura em vez
    # de para a escala. **A tentação aqui é do eixo y:** ele é uma fração em `[0, 1]`, e uma figura
    # com barras a `0,0` é a faixa em que um eixo logarítmico é impossível — é o caso em que a
    # varredura confirma em vez de acusar, e isso vale tanto quanto o contrário.
    graficos.recebe_classificadores(_com_classificadores(19))
    # E a rede, pela mesma razão das notas acima. Sem a carga o construtor recusa por
    # população vazia e ela entraria em `sem_fio`, com a varredura a acusar a fixtura em vez da
    # escala. **A tentação aqui é o eixo sem marca:** os dois eixos desta figura não têm `tick`
    # nenhum, e uma escala logarítmica sem marca é invisível a olho — é exactamente o caso em que
    # uma varredura de propriedade vale mais do que uma leitura.
    graficos.recebe_rede(_com_rede(21))
    # E a curva do LOESS, pela mesma razão. Ela tem rota própria, então entra na
    # varredura desenhada; sem a carga cairia no conjunto dos que não têm fio, que é outra coisa. O
    # corpo é o da rota, sobre os arquivos da fábrica, e não um dicionário escrito aqui.
    graficos.recebe_curva_do_loess(corpo_da_fabrica())
    graficos.figuras.define_execucao(_resposta(200, _com_plsda(7)))

    montadores = graficos._montadores()
    assert len(montadores) == 32, sorted(montadores)

    # **OS CINCO SEM CAMINHO DE FIO SAEM DA VARREDURA, e a exclusão é DERIVADA por desfecho.**
    #
    # Nenhuma rota devolve o que `plsda_multiclasse` e `agrupamento` consomem — o primeiro não tem
    # sequer função `*_do_fio` —, então não há `recebe_*` que os alimente e eles recusam por
    # população vazia. Eles têm montador porque o portão de bijeção o exige, e porque sem ele a
    # entrada no combo levanta `FonteAusenteError`, que o painel descreve como defeito do programa.
    #
    # **A exclusão não é escrita à mão: é o conjunto dos que NÃO desenharam depois de todos os
    # `recebe_*` terem corrido**, e a contagem esperada vai ao lado como fonte independente. Um
    # construtor que deixe de desenhar por outro motivo faz este conjunto crescer, e a igualdade
    # abaixo acusa-o — que é a propriedade que a `assert eixos` sozinha dava.
    sem_fio: set[str] = set()
    medidos: list[tuple[str, str, str]] = []
    for funcao in sorted(montadores):
        _escolhe(graficos, funcao)
        eixos = graficos.figuras.canvas.figure.axes
        if not eixos:
            sem_fio.add(funcao)
            continue
        medidos += [(funcao, eixo.get_xscale(), eixo.get_yscale()) for eixo in eixos]

    assert sem_fio == {
        "figura_da_importancia_um_contra_resto",
        "figura_das_cargas_um_contra_resto",
        "figura_do_dendrograma",
        "figura_do_mapa_de_calor",
        "figura_dos_escores_um_contra_resto",
    }, (
        f"o conjunto dos que não desenham mudou: {sorted(sem_fio)}. Ou uma rota passou a alimentar "
        "um deles — e aí ele entra na varredura —, ou um construtor que desenhava deixou de o "
        "fazer, que é regressão"
    )
    assert len(medidos) >= len(montadores) - len(sem_fio), "âncora: menos eixos medidos que figuras"
    nao_lineares = [m for m in medidos if m[1:] != ("linear", "linear")]
    assert not nao_lineares, f"eixos não lineares: {nao_lineares}"


def _evento_de_pick(painel: PainelDeFigura, eixo: Axes, x: float, y: float, **extra: Any) -> Any:
    """Um `PickEvent` sobre a primeira linha do eixo, construído à mão.

    **À mão e não por `Figure.pick`**, porque é justamente `Figure.pick` que se quer contornar: ele
    não despacha com o cadeado tomado, e um par causal que passasse por ele mediria a guarda do
    `matplotlib` em vez da deste painel.
    """
    painel.canvas.draw()
    tela_x, tela_y = eixo.transData.transform((x, y))
    rato = MouseEvent("button_press_event", painel.canvas, tela_x, tela_y, 1)
    linha = eixo.lines[0]
    dentro, props = linha.contains(rato)
    assert dentro, "âncora: o ponto escolhido não está sob o cursor, e o evento não seria realista"
    return PickEvent("pick_event", painel.canvas, rato, linha, **{**props, **extra})


def test_a_guarda_do_widgetlock_no_manipulador_da_identificacao_tem_as_duas_direcoes(
    qtbot: QtBot,
) -> None:
    """O par causal DA GUARDA, em processo — a planta é o argumento, então não pode ficar inerte.

    O teste de ponta a ponta acima não a alcança: `Figure.pick` não despacha com o cadeado tomado.
    Aqui o manipulador é chamado **diretamente**, com o mesmo evento nas duas direções, e o que
    separa os desfechos é só o cadeado.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    evento = _evento_de_pick(painel, _eixo(painel), X_A[1], Y_A[1])

    dono = object()
    painel.canvas.widgetlock(dono)
    painel._identifica(evento)
    assert painel.estado_visivel()["anotacao"] == "", (
        "o manipulador anotou com o cadeado tomado — escrever no eixo por baixo de uma ferramenta "
        "interativa produz estado que ela não sabe desfazer"
    )

    painel.canvas.widgetlock.release(dono)
    painel._identifica(evento)
    assert NOMES[1] in painel.estado_visivel()["anotacao"], (
        "o manipulador não anotou nem com o cadeado livre — a metade de cima seria vácua"
    )


@pytest.mark.parametrize(
    ("extra", "porque"),
    [
        ({"ind": None}, "sem `ind` nenhum"),
        ({"ind": []}, "`ind` vazio"),
        ({"ind": [99]}, "`ind` fora do dado desenhado"),
    ],
)
def test_o_manipulador_recusa_o_evento_que_nao_aponta_ponto_nenhum(
    qtbot: QtBot, extra: dict[str, Any], porque: str
) -> None:
    """Os três desfechos de recusa, e a metade que os torna não vácuos vem depois.

    N desfechos, N direções. Um manipulador que recusasse **sempre** passaria nos três
    casos acima e não anotaria nunca — por isso o mesmo teste dispara, no fim, um evento bom.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)

    mau = _evento_de_pick(painel, eixo, X_A[1], Y_A[1], **extra)
    if extra["ind"] is None:
        del mau.ind
    painel._identifica(mau)
    assert painel.estado_visivel()["anotacao"] == "", f"o evento com {porque} produziu anotação"

    painel._identifica(_evento_de_pick(painel, eixo, X_A[1], Y_A[1]))
    assert NOMES[1] in painel.estado_visivel()["anotacao"], (
        "o manipulador não anota nem com evento bom — as recusas acima seriam vácuas"
    )


def test_a_recusa_por_indice_fora_do_dado_vale_para_COLECAO_e_nao_so_para_LINHA(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """A irmã da guarda de índice, na outra forma de artista.

    Linha e coleção guardam o dado por caminhos diferentes — `get_xdata` contra `get_offsets` —, e
    uma guarda de índice escrita num e esquecida no outro é o mesmo modo de `csv.Error`
    apanhado numa função e ausente na irmã.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    eixo = _eixo(painel)
    painel.canvas.draw()

    colecao = eixo.collections[0]
    assert len(colecao.get_offsets()) == len(X_B), "âncora: a coleção não tem o dado esperado"
    tela_x, tela_y = eixo.transData.transform((X_B[1], Y_B[1]))
    rato = MouseEvent("button_press_event", painel.canvas, tela_x, tela_y, 1)

    painel._identifica(PickEvent("pick_event", painel.canvas, rato, colecao, ind=[99]))
    assert painel.estado_visivel()["anotacao"] == "", (
        "a coleção com índice fora do dado produziu anotação — a guarda existe na linha e falta na "
        "irmã"
    )

    painel._identifica(PickEvent("pick_event", painel.canvas, rato, colecao, ind=[1]))
    assert NOMES[1] in painel.estado_visivel()["anotacao"], (
        "a coleção com índice válido não produziu anotação — a recusa acima seria vácua"
    )

    # **A terceira forma de artista: a que não tem dado de ponto nenhum.** Hoje `_artistas` só
    # devolve linhas e coleções, então este ramo não é alcançável pelo caminho normal — e é
    # exercitado de todo modo, porque o teste vem antes do `pragma`.
    assert PainelDeFigura._ponto_do_artista(eixo.title, 0) is None, (
        "um `Text` — artista sem dado de ponto — devolveu coordenadas; a identificação anotaria "
        "sobre um título"
    )


def test_o_seletor_sem_escolha_cai_em_SEM_CONSTRUTOR_em_vez_de_levantar(qtbot: QtBot) -> None:  # noqa: N802
    """O ramo de recuo de `_troca_a_figura`, pelo caminho público que o alcança.

    `setCurrentIndex(-1)` num combo com itens emite `currentIndexChanged` e deixa a escolha fora do
    intervalo. **Um `IndexError` aqui seria engolido para o `stderr` dentro de um slot de Qt** — o
    desfecho que este painel existe para não ter —, então o recuo é texto e não excepção.
    """
    painel = PainelDeFigura(
        construtores=(_um_construtor(),), fonte=lambda _c: (figura_de_teste(), NOMES)
    )
    qtbot.addWidget(painel)
    painel.define_execucao(_resposta(200, _corpo(1)))
    assert painel.estado_visivel()["figura"] == "1 eixo(s)", "âncora: nada foi desenhado"

    painel.seletor.setCurrentIndex(-1)
    assert painel.estado_visivel()["figura"] == "", "a figura ficou sem escolha que a justifique"
    assert painel.estado_visivel()["repouso"] == texto_do_estado(EstadoDaFigura.SEM_CONSTRUTOR)


# ---------------------------------------------------------------------------
# A SEGUNDA PORTA — o que a revisão de 11/09/2026 achou
#
# `desenha()` tem duas entradas: `define_execucao` e o slot do combo. A segunda nasceu sem as
# guardas da primeira, e os dois testes abaixo são o par causal de cada metade.
# ---------------------------------------------------------------------------


def _dois_construtores() -> tuple[ConstrutorDeFigura, ConstrutorDeFigura]:
    """Dois, porque com UM só não há troca de índice a exercitar.

    O valor natural — um construtor — é o que não executa o caminho. `setCurrentIndex`
    num combo de uma entrada não emite `currentIndexChanged`, e o teste passaria sem tocar no slot.
    """
    return (
        _um_construtor(),
        ConstrutorDeFigura(
            modulo="arnes", funcao="figura_de_arnes_2", vinculo=Vinculo.RELATORIO, chave="item-1"
        ),
    )


@pytest.mark.parametrize(
    ("codigo", "esperado"),
    [(202, EstadoDaFigura.EM_CURSO), (500, EstadoDaFigura.FALHOU)],
)
def test_trocar_de_figura_com_a_execucao_OBSOLETA_nao_ressuscita_o_canvas(  # noqa: N802
    qtbot: QtBot, codigo: int, esperado: EstadoDaFigura
) -> None:
    """Par causal do portão de obsolescência na porta do combo.

    A planta cai sobre dado em que as duas coisas diferem: a âncora afirma que a figura
    ESTAVA na tela antes do `202`/`500`, senão «continua retirada» seria vácuo — passaria com um
    painel que nunca desenhou nada.

    **E a segunda asserção é a que separa o defeito do seu primo.** Sem o portão, as figuras vindas
    de `/matrizes/diagnosticar` voltavam com número velho e as de `/execucoes` caíam em `SEM_DADO`
    — *"o construtor recebeu população vazia"* — sobre o fato *"a execução ainda corre"*. Afirmar só
    que o canvas está vazio deixaria o segundo passar: ele TAMBÉM esvazia o canvas, com a frase
    errada. Por isso o texto é comparado, e comparado contra o primo também.
    """
    painel = PainelDeFigura(
        construtores=_dois_construtores(), fonte=lambda _c: (figura_de_teste(), NOMES)
    )
    qtbot.addWidget(painel)
    painel.define_execucao(_resposta(200, _corpo(1)))
    assert painel.estado_visivel()["figura"] == "1 eixo(s)", "âncora: nada foi desenhado"

    painel.define_execucao(_resposta(codigo, {}))
    assert painel.estado_visivel()["figura"] == "", "âncora: a figura não saiu com a resposta ruim"

    painel.seletor.setCurrentIndex(1)

    visivel = painel.estado_visivel()
    assert visivel["figura"] == "", (
        "trocar de figura no seletor ressuscitou o canvas sobre uma execução que não é a corrente"
    )
    assert visivel["repouso"] == texto_do_estado(esperado)
    assert visivel["repouso"] != texto_do_estado(EstadoDaFigura.SEM_DADO), (
        "a troca no seletor trocou a frase certa pela de população vazia"
    )


def test_a_fonte_ausente_vira_ESTADO_e_nao_sobe_pelo_slot_de_qt(qtbot: QtBot) -> None:  # noqa: N802
    """`FonteAusenteError` atravessando a fronteira do slot vira texto, e texto PRÓPRIO.

    **O oráculo é o estado e não «não levantou».** Um teste que só afirmasse a ausência de excepção
    passaria com a recusa colapsada em `SEM_CONSTRUTOR` ou em `DADO_RECUSADO` — e as três frases
    dizem coisas diferentes: *"nenhuma figura registada"* é fato do projeto, *"o construtor recusou
    o que veio"* é fato do dado, e esta é defeito de ligação NOSSO. Por isso as duas negativas.

    **Por que o slot e não a chamada direta:** `test_a_construcao_sem_fonte_com_construtores_recusa`
    e o teste de `_constroi_figura` já cobrem os dois sítios que a levantam. O que não estava
    coberto — e é onde o silêncio acontecia — é ela CRUZAR a fronteira do Qt.
    """

    def fonte_mal_ligada(_c: ConstrutorDeFigura) -> tuple[Figure, tuple[str, ...]]:
        raise FonteAusenteError("este painel não sabe construí-lo")

    painel = PainelDeFigura(construtores=_dois_construtores(), fonte=fonte_mal_ligada)
    qtbot.addWidget(painel)

    painel.define_execucao(_resposta(200, _corpo(1)))

    visivel = painel.estado_visivel()
    assert visivel["figura"] == "", "a figura foi desenhada apesar de a fonte não saber construí-la"
    assert visivel["repouso"] == texto_do_estado(EstadoDaFigura.LIGACAO_QUEBRADA)
    assert visivel["repouso"] != texto_do_estado(EstadoDaFigura.SEM_CONSTRUTOR), (
        "defeito de ligação a mostrar a frase de «nenhuma figura registada»"
    )
    assert visivel["repouso"] != texto_do_estado(EstadoDaFigura.DADO_RECUSADO), (
        "defeito de ligação a mostrar a frase de recusa do dado"
    )


# ---------------------------------------------------------------------------
# A COR e fixa, a ESCALA continua a vir da tela
# ---------------------------------------------------------------------------


def _cor(valor: Any) -> str:
    """Uma cor do `matplotlib` em `#rrggbb`, seja ela tupla RGBA ou texto.

    **Um normalizador e não `QColor(valor)` direto**, e a razão é medida: `spine.get_edgecolor()`
    devolve uma tupla RGBA e `QColor` levanta `TypeError` sobre ela, enquanto `label.get_color()`
    devolve o texto que o painel escreveu. Comparar formas diferentes da mesma cor daria uma
    reprovação sobre o instrumento e não sobre o alvo.
    """
    return str(to_hex(valor))


def _paleta(fundo: str, area: str, tinta: str) -> QPalette:
    paleta = QPalette()
    paleta.setColor(QPalette.ColorRole.Window, QColor(fundo))
    paleta.setColor(QPalette.ColorRole.Base, QColor(area))
    paleta.setColor(QPalette.ColorRole.WindowText, QColor(tinta))
    return paleta


CLARA = ("#efefef", "#ffffff", "#000000")
ESCURA = ("#1f1f1f", "#2b2b2b", "#f0f0f0")


def test_par_causal_INVERTIDO_a_MESMA_figura_sob_as_duas_paletas_fica_BRANCA(qtbot: QtBot) -> None:  # noqa: N802
    """A figura pinta-se com uma trinca FIXA, e o tema do sistema deixou de a alcançar.

    ## Este par causal INVERTEU de sentido, e a reescrita é de propósito

    Até 14/09/2026 ele afirmava que a mesma figura sob duas paletas saía com dois fundos
    **diferentes**, cada um o da sua paleta. Hoje afirma o contrário: os dois desfechos são
    **idênticos e brancos**. Não é afrouxamento — ajustar número esperado para fazer teste
    passar é proibido, e o que mudou aqui foi a verdade sobre o que a tela mostra, com a
    decisão tomada antes de esta linha existir.

    **A âncora de não-vacuidade é sobre a ENTRADA**, e sem ela o cenário passaria com duas paletas
    iguais: afirma-se primeiro que `CLARA` e `ESCURA` divergem nos três papéis.

    **E o par corre pelo PAINEL e não pela função**, porque é o painel que decide o que alimenta
    `aplica_o_tema` — o ponto que a trinca fixa trocou. Chamar a função com uma trinca
    escolhida pelo teste responderia a pergunta errada: ela continua total sobre o que recebe.
    """
    assert all(a != b for a, b in zip(CLARA, ESCURA, strict=True)), (
        "âncora: as duas paletas de entrada não divergem nos três papéis, e o par seria vácuo"
    )

    desfechos: list[Figure] = []
    for cores in (CLARA, ESCURA):
        painel = PainelDeFigura(construtores=())
        qtbot.addWidget(painel)
        painel.setPalette(_paleta(*cores))
        painel.desenha(figura_de_teste(), nomes=NOMES)
        desfechos.append(painel.canvas.figure)

    clara, escura = desfechos
    assert _cor(clara.get_facecolor()) == _cor(escura.get_facecolor()), (
        f"a figura ainda segue a paleta do sistema: claro={_cor(clara.get_facecolor())}, "
        f"escuro={_cor(escura.get_facecolor())}"
    )
    assert _cor(clara.get_facecolor()) == CORES_DA_FIGURA[0]
    assert QColor(_cor(escura.get_facecolor())).lightnessF() > 0.9, (
        "o fundo fixo não é claro, e devia ser branco"
    )

    for figura in (clara, escura):
        eixo = figura.axes[0]
        assert _cor(eixo.get_facecolor()) == CORES_DA_FIGURA[1]
        assert _cor(eixo.xaxis.label.get_color()) == CORES_DA_FIGURA[2]
        assert _cor(eixo.yaxis.label.get_color()) == CORES_DA_FIGURA[2]
        assert _cor(eixo.title.get_color()) == CORES_DA_FIGURA[2]
        assert {_cor(e.get_edgecolor()) for e in eixo.spines.values()} == {CORES_DA_FIGURA[2]}
        marcas = {_cor(r.get_color()) for r in eixo.get_xticklabels()}
        assert marcas, "âncora: o eixo x não tem marca nenhuma e a asserção seria vácua"
        assert marcas == {CORES_DA_FIGURA[2]}, "as marcas do eixo x não levaram a tinta fixa"
        assert {_cor(x.get_color()) for x in figura.texts} == {CORES_DA_FIGURA[2]}, (
            "o rodapé do recorte não levou a tinta fixa — é o único texto da figura que outra "
            "pessoa pode desmentir, e ele desapareceria contra o fundo branco"
        )
        legenda = eixo.get_legend()
        assert legenda is not None, "âncora: a figura de teste não tem legenda"
        assert _cor(legenda.get_frame().get_facecolor()) == CORES_DA_FIGURA[1]
        assert {_cor(x.get_color()) for x in legenda.get_texts()} == {CORES_DA_FIGURA[2]}, (
            "os textos da legenda não levaram a tinta fixa"
        )


def test_a_TINTA_vai_junto_e_nada_CLARO_sobra_na_figura_sob_paleta_escura(qtbot: QtBot) -> None:  # noqa: N802
    """A armadilha da trinca fixa: fixar o fundo e deixar a tinta põe tudo branco sobre branco.

    `aplica_o_tema` usa **a mesma trinca** para o fundo e para tudo o que é texto ou moldura. Sob a
    paleta escura do sistema o `WindowText` é claro; se a tinta continuasse a vir dali, a figura
    ficaria branca com a rotulagem branca — e **nada levantaria**.

    **A âncora de não-vacuidade vem primeiro e na mesma função**, porque esta é uma asserção
    de ausência: *«nenhum artista de tinta ficou claro»* passa sobre um conjunto vazio, e conjunto
    vazio é o que se obtém de uma figura sem texto nenhum ou de uma recolha que falhou.
    """
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.setPalette(_paleta(*ESCURA))
    painel.desenha(figura_de_teste(), nomes=NOMES)
    figura = painel.canvas.figure
    eixo = figura.axes[0]
    legenda = eixo.get_legend()
    assert legenda is not None, "âncora: a figura de teste não tem legenda"

    de_tinta: list[tuple[str, str]] = [
        ("rotulo x", _cor(eixo.xaxis.label.get_color())),
        ("rotulo y", _cor(eixo.yaxis.label.get_color())),
        ("titulo", _cor(eixo.title.get_color())),
        *((f"espinha {n}", _cor(e.get_edgecolor())) for n, e in eixo.spines.items()),
        *((f"marca x {i}", _cor(r.get_color())) for i, r in enumerate(eixo.get_xticklabels())),
        *((f"marca y {i}", _cor(r.get_color())) for i, r in enumerate(eixo.get_yticklabels())),
        *((f"texto de figura {i}", _cor(x.get_color())) for i, x in enumerate(figura.texts)),
        *((f"texto de eixo {i}", _cor(x.get_color())) for i, x in enumerate(eixo.texts)),
        *((f"legenda {i}", _cor(x.get_color())) for i, x in enumerate(legenda.get_texts())),
        ("moldura da legenda", _cor(legenda.get_frame().get_edgecolor())),
    ]
    assert len(de_tinta) >= 6, (
        f"âncora: só {len(de_tinta)} artistas de tinta recolhidos — a asserção seria vácua"
    )

    claros = [(o, c) for o, c in de_tinta if QColor(c).lightnessF() > 0.5]
    assert claros == [], (
        f"sob a paleta ESCURA do sistema, {len(claros)} de {len(de_tinta)} artistas de tinta "
        f"ficaram claros sobre o fundo branco fixo — invisíveis: {claros[:6]}"
    )


def test_d437_os_textos_DENTRO_do_eixo_levam_a_tinta_FIXA(qtbot: QtBot) -> None:  # noqa: N802
    """`figura.texts` não alcança os `Text` do EIXO, e eles ficavam pretos.

    ## O buraco, medido em 13/09/2026

    `aplica_o_tema` percorria `figura.texts` — os textos de **nível de figura**, que aqui são só o
    rodapé do recorte — mais os rótulos, o título e a legenda de cada eixo. O que os construtores
    escrevem **dentro** do eixo é outra coleção, e ficava na cor dos `rcParams`: medido, o texto de
    figura saía `#eeeeee` e o do eixo saía `black`.

    **Quatro construtores escrevem ali, e o que se perde difere:** a frase do «vazio declarado» do
    volcano, o «sem elipse» dos scores do PCA e o «VIP indisponível» do PLS-DA — as três são a
    convenção de ausência declarada — e os **doze números das setas do biplot**, que são a única
    ligação entre a seta e a feature.

    ## O que a trinca fixa mudou nesta asserção, e por que o cenário FICA

    O oráculo deixou de ser *«a cor da paleta que entrou»* e passou a ser `CORES_DA_FIGURA[2]`.
    Com a tinta preta, estes textos **já sairiam pretos** pelos `rcParams` de origem — e o
    cenário deixou de provar que o laço os salva e passou a provar que a trinca é a **única**
    fonte: a planta abaixo troca a tinta e exige que eles a sigam. Sem ela, remover o laço não
    reprovaria nada, e a próxima trinca deixá-los-ia para trás.

    > **O teste vizinho não o apanhava**, e por um motivo de população: ele afirma sobre
    > `figura.texts`, que é exactamente a coleção certa para o rodapé e exactamente a errada para
    > estes. As duas asserções parecem a mesma e não são.

    A âncora vem primeiro: a figura de teste pode não ter texto de eixo nenhum, e aí a
    asserção passaria sobre um conjunto vazio.
    """
    for cores in (CLARA, ESCURA):
        painel = PainelDeFigura(construtores=())
        qtbot.addWidget(painel)
        painel.setPalette(_paleta(*cores))
        figura = figura_de_teste()
        eixo = figura.axes[0]
        eixo.text(0.5, 0.5, "ausência declarada")
        eixo.annotate("PC1", (0.2, 0.2))

        painel.desenha(figura, nomes=NOMES)

        do_eixo = {_cor(x.get_color()) for x in eixo.texts}
        assert do_eixo, "âncora: o eixo não tem texto nenhum e a asserção seria vácua"
        assert do_eixo == {CORES_DA_FIGURA[2]}, (
            f"os textos dentro do eixo não levaram a tinta fixa: {do_eixo} contra "
            f"{CORES_DA_FIGURA[2]}. Sobre o fundo branco eles somem se ficarem claros — a frase de "
            f"ausência desaparece, e o biplot perde os números que ligam cada seta à feature"
        )

    # A planta que separa o laço de uma coincidência com os `rcParams`: com a tinta trocada
    # por uma cor que o `matplotlib` nunca usaria, os textos do eixo têm de a seguir.
    outra = ("#ffffff", "#ffffff", "#7b1fa2")
    assert outra[2] != CORES_DA_FIGURA[2], "âncora: a planta não muda a tinta e seria vácua"
    figura = figura_de_teste()
    eixo = figura.axes[0]
    eixo.text(0.5, 0.5, "ausência declarada")
    aplica_o_tema(figura, outra)
    assert {_cor(x.get_color()) for x in eixo.texts} == {outra[2]}, (
        "o laço dos textos do eixo não segue a trinca — hoje passaria por coincidência com o preto "
        "dos `rcParams`, e a trinca seguinte deixá-los-ia para trás"
    )


def test_o_tema_NAO_repinta_as_series_e_a_distincao_entre_elas_fica(qtbot: QtBot) -> None:  # noqa: N802
    """A metade que o par causal acima não vê: pintar as séries apagaria a informação da figura."""
    figura = figura_de_teste()
    antes = {s.get_label(): s.get_color() for s in figura.axes[0].lines}
    assert antes, "âncora: a figura não tem linha nenhuma"
    aplica_o_tema(figura, CORES_DA_FIGURA)
    depois = {s.get_label(): s.get_color() for s in figura.axes[0].lines}
    assert depois == antes, f"o tema repintou as séries: {antes} -> {depois}"


def test_a_anotacao_do_clique_leva_a_tinta_FIXA_sob_paleta_escura(qtbot: QtBot) -> None:  # noqa: N802
    """A SEGUNDA leitura da paleta, que nenhum portão cobria.

    `_identifica` escolhia a cor da anotação com uma chamada própria a `cores_do_tema`, três métodos
    abaixo de `aplica_o_tema`. O que existia afirmava que a anotação **existe**, nunca de que cor
    ela é: ficasse por trocar, ela nasceria clara sobre o fundo branco fixo, e o defeito só
    apareceria a quem clicasse.

    **Par causal sobre a paleta**, e a âncora afirma primeiro que a anotação de facto apareceu — sem
    ela, `self._anotacao is None` faria as duas asserções passarem por vacuidade.
    """
    for cores in (CLARA, ESCURA):
        painel = PainelDeFigura(construtores=())
        qtbot.addWidget(painel)
        painel.setPalette(_paleta(*cores))
        painel.desenha(figura_de_teste(), nomes=NOMES)
        painel._identifica(_evento_de_pick(painel, _eixo(painel), X_A[1], Y_A[1]))

        anotacao = painel._anotacao
        assert anotacao is not None, "âncora: o clique não deixou anotação nenhuma"
        assert NOMES[1] in painel.estado_visivel()["anotacao"], (
            "âncora: a anotação está na tela mas não traz o nome esperado"
        )
        assert _cor(anotacao.get_color()) == CORES_DA_FIGURA[2], (
            f"a anotação do clique saiu em {_cor(anotacao.get_color())} sob a paleta "
            f"{cores[2]} — ela lê a trinca uma SEGUNDA vez, e sobre o fundo branco fixo uma tinta "
            f"clara torna-a invisível"
        )


def test_par_causal_da_escala_a_figura_desenha_no_dpi_DA_TELA(qtbot: QtBot) -> None:  # noqa: N802
    """`FigureCanvasQTAgg` não acompanha o fator de escala do Qt, e é medido.

    A 150% o Windows devolve `logicalDotsPerInch() == 144`, e sem esta leitura a figura ficaria nos
    100 de `rcParams` ao lado de widgets que cresceram metade.
    """
    for pedido in (96.0, 144.0):
        painel = PainelDeFigura(construtores=(), escala=lambda valor=pedido: valor)
        qtbot.addWidget(painel)
        painel.desenha(figura_de_teste(), nomes=NOMES)
        assert painel.canvas.figure.dpi == pedido, (
            f"pedi {pedido} pontos por polegada e a figura ficou em {painel.canvas.figure.dpi}"
        )

    # E o caminho de produção, sem injeção nenhuma: quem lê é `dpi_da_tela`, sobre a tela real.
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    da_tela = dpi_da_tela(painel)
    assert da_tela > 0, "a tela devolveu pontos por polegada não positivos"
    painel.desenha(figura_de_teste(), nomes=NOMES)
    assert painel.canvas.figure.dpi == da_tela, (
        f"o painel sem injeção desenhou a {painel.canvas.figure.dpi} e a tela diz {da_tela} — o "
        "padrão do parâmetro não está a ler a tela"
    )


def test_cores_do_tema_le_os_tres_papeis_e_nao_literais() -> None:
    """A função é total sobre a paleta que recebe: três papéis, três cores, nenhuma escrita aqui."""
    assert cores_do_tema(_paleta(*ESCURA)) == ESCURA
    assert cores_do_tema(_paleta(*CLARA)) == CLARA


# ---------------------------------------------------------------------------
# O seletor — derivado do registo, nunca redigitado
# ---------------------------------------------------------------------------


def test_d435_os_rotulos_do_seletor_sao_DISTINGUIVEIS_entre_si() -> None:  # noqa: N802
    """Dezasseis entradas, dezasseis rótulos — e antes de 12/09/2026 eram OITO para treze.

    ## O defeito, medido

    O rótulo era `f"{c.modulo} · {c.chave}"`, e esse par **não é único** no registo: quatro
    construtores de PCA partilham `("pca", "PCA")` e três de PLS-DA partilham
    `("plsda", "PLS-DA")`. O combo mostrava quatro linhas a ler *«pca · PCA»*, e quem quisesse o
    *scree* em vez do *biplot* não tinha como o pedir.

    > **Um rótulo derivado pode ser derivado do campo errado.** A derivação protege contra a lista
    > que caduca — que é o que `entradas_do_seletor` existe para ter — e não diz nada sobre a
    > **unicidade** do que se derivou. São duas propriedades, e só a primeira tinha portão.

    A âncora vem primeiro: com registo vazio a asserção de distinção passaria por
    vacuidade, e registo vazio é exactamente o estado que o painel teve no começo.
    """
    assert CONSTRUTORES, "âncora: o registo está vazio e a asserção abaixo seria vácua"

    rotulos = [rotulo for rotulo, _ in entradas_do_seletor(CONSTRUTORES)]
    repetidos = {r for r in rotulos if rotulos.count(r) > 1}

    assert not repetidos, (
        f"{len(rotulos) - len(set(rotulos))} entrada(s) do seletor não se distinguem de outra: "
        f"{sorted(repetidos)}. Quem abre o combo vê linhas iguais e não consegue pedir uma figura "
        f"específica"
    )
    assert len(set(rotulos)) == len(CONSTRUTORES), (
        f"{len(CONSTRUTORES)} construtores e {len(set(rotulos))} rótulos distintos"
    )


def test_as_entradas_do_seletor_sao_DERIVADAS_do_registo(qtbot: QtBot) -> None:  # noqa: N802
    """O padrão de `METRICAS_DA_TELA_DE_QUALIDADE`: a lista sai do registo.

    A âncora contra a vacuidade vai na mesma função: `CONSTRUTORES` está vazio hoje, e
    afirmar *"o combo tem tantas entradas quanto o registo"* sobre um registo vazio seria afirmar
    `0 == 0`. A metade com registo plantado é o que dá sentido à metade vazia.
    """
    assert entradas_do_seletor(()) == ()
    plantado = (_um_construtor(),)
    derivadas = entradas_do_seletor(plantado)
    assert len(derivadas) == 1 and derivadas[0][1] is plantado[0]
    # A afirmação é sobre o nome DERIVADO DA FUNÇÃO, e mudou. Ela dizia
    # `"arnes" in … and "item-0" in …`, o `modulo` e a `chave`, porque o rótulo era
    # `f"{modulo} · {chave}"`. Esse par colidia em sete das treze entradas do registo real, e o
    # rótulo passou a sair de `funcao`, que é o campo único.
    #
    # Esta linha tem uma IRMÃ: o formato do rótulo é lido por dois mecanismos — este
    # teste e o portão da distinção —, e mudá-lo satisfazendo só um deixaria o outro vermelho. O
    # `figura_de_arnes` do construtor plantado vira `arnes` pela regra de `_nome_legivel`, que tira
    # o prefixo `figura_` e o artigo; então `arnes` continua no rótulo, e `item-0` não.
    assert derivadas[0][0] == "arnes", derivadas
    assert "item-0" not in derivadas[0][0], (
        f"a `chave` voltou ao rótulo: ela é o campo que COLIDE, e é por isso que saiu. {derivadas}"
    )

    # `construtores=()` EXPLÍCITO. Antes `CONSTRUTORES` era vazio e o
    # padrão bastava; com a primeira figura registada, um painel sem `fonte` recusa nascer — que é
    # o comportamento certo. O que este teste mede é o estado **vazio**, e o estado vazio passou a
    # ter de ser pedido em vez de herdado.
    vazio = PainelDeFigura(construtores=())
    qtbot.addWidget(vazio)
    assert vazio.seletor.count() == 0, vazio.seletor.count()
    # E a outra direção: o registo real **não** está vazio, e um painel construído sobre
    # ele mostra tantas entradas quantas o registo tem. Sem esta metade, a asserção de cima seria
    # sobre um argumento e não sobre o registo do projeto.
    assert CONSTRUTORES, "o registo está vazio: a metade de baixo deste teste seria vácua"
    real = PainelDeFigura(fonte=lambda _c: (figura_de_teste(), NOMES))
    qtbot.addWidget(real)
    assert real.seletor.count() == len(CONSTRUTORES), (
        f"`CONSTRUTORES` tem {len(CONSTRUTORES)} entrada(s) e o combo tem {real.seletor.count()}"
    )
    # **ESCONDIDO e não DESABILITADO, e o par causal está no portão da roda.** Um `QComboBox`
    # desabilitado engole a roda pelo `setEnabled` em vez do filtro, e
    # `test_desktop_roda.py::test_a_roda_nao_muda_campo_nenhum_da_janela_sem_foco` varre **todos**
    # os combos da janela montada e reprova por isso — foi ele que acusou a primeira versão.
    assert vazio.seletor.isEnabled(), (
        "o combo vazio nasceu desabilitado: o portão da roda exige que nenhum campo rolável da "
        "janela esteja desabilitado, porque um campo assim engole a roda sem passar pelo filtro"
    )
    assert vazio.estado_visivel()["seletor"] == "", "o combo vazio continua na tela"

    cheio = PainelDeFigura(construtores=plantado, fonte=lambda _c: (figura_de_teste(), NOMES))
    qtbot.addWidget(cheio)
    assert cheio.seletor.count() == 1 and cheio.seletor.isEnabled()
    assert cheio.estado_visivel()["seletor"] == derivadas[0][0], (
        "com registo o combo tem de estar na tela — sem isto a asserção de ausência acima passa "
        "por vacuidade"
    )
    assert cheio.seletor.itemText(0) == derivadas[0][0], (
        "o rótulo do combo não é o que a derivação produz — há uma segunda redação da mesma lista"
    )


def test_com_registo_e_fonte_a_execucao_corrente_DESENHA(qtbot: QtBot) -> None:  # noqa: N802
    """O caminho que as unidades das figuras vão acender, exercitado por injeção.

    É o padrão de `texto_da_metrica(caminhos=…)`: um ramo cuja condição é estática hoje não tem par
    causal sem o parâmetro, e ramo sem par causal é o que fica inerte.
    """
    chamadas: list[ConstrutorDeFigura] = []

    def fonte(construtor: ConstrutorDeFigura) -> tuple[Figure, tuple[str, ...]]:
        chamadas.append(construtor)
        return figura_de_teste(), NOMES

    plantado = (_um_construtor(),)
    painel = PainelDeFigura(construtores=plantado, fonte=fonte)
    qtbot.addWidget(painel)
    assert painel.estado_visivel()["figura"] == "", "âncora: o painel nasceu com figura"

    painel.define_execucao(_resposta(200, _corpo(1)))
    assert chamadas == [plantado[0]], f"a fonte foi chamada {len(chamadas)} vez(es)"
    assert painel.estado_visivel()["figura"] == "1 eixo(s)"
    assert painel.estado_visivel()["repouso"] == ""


def test_sem_registo_a_execucao_corrente_diz_que_NAO_HA_FIGURA_e_nao_fica_em_branco(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """R4b na direção que não é obsolescência: há execução, e não há o que desenhar."""
    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    texto = painel.define_execucao(_resposta(200, _corpo(1)))
    assert texto == texto_do_estado(EstadoDaFigura.SEM_CONSTRUTOR)
    assert painel.estado_visivel()["repouso"] == texto
    assert texto != texto_do_estado(EstadoDaFigura.SEM_EXECUCAO), (
        "«há execução e não há figura» e «não há execução» dizem a mesma coisa na "
        "frase que o profissional lê"
    )


def test_a_populacao_vazia_do_construtor_vira_TEXTO_e_nao_excecao_num_slot(qtbot: QtBot) -> None:  # noqa: N802
    """`MatrizVaziaError` dentro de um slot de Qt é engolida para o `stderr`, e o painel congela.

    A recusa tipada de `plots.base` é um fato do dado — *"não há população para desenhar"* — e tem
    frase própria, distinta de *"não há figura registada"*.
    """

    def fonte_vazia(_construtor: ConstrutorDeFigura) -> tuple[Figure, tuple[str, ...]]:
        raise MatrizVaziaError("nada a desenhar")

    painel = PainelDeFigura(construtores=(_um_construtor(),), fonte=fonte_vazia)
    qtbot.addWidget(painel)
    texto = painel.define_execucao(_resposta(200, _corpo(1)))
    assert texto == texto_do_estado(EstadoDaFigura.SEM_DADO)
    assert painel.estado_visivel()["figura"] == ""
    assert texto != texto_do_estado(EstadoDaFigura.SEM_CONSTRUTOR), (
        "«o construtor recusou» e «não há construtor» leem-se igual, e são fatos opostos"
    )


def test_registo_sem_fonte_e_recusado_na_CONSTRUCAO(qtbot: QtBot) -> None:  # noqa: N802
    """As duas direções da guarda: com fonte constrói, sem fonte levanta.

    Deixar passar faria o painel escrever *"nenhuma figura registada"* sobre um registo cheio — a
    frase certa para o fato oposto, que é o modo de falha silenciosa que `_encaminha` já pagou.
    """
    with pytest.raises(FonteAusenteError, match="nenhuma fonte"):
        PainelDeFigura(construtores=(_um_construtor(),))

    com_fonte = PainelDeFigura(
        construtores=(_um_construtor(),), fonte=lambda _c: (figura_de_teste(), NOMES)
    )
    qtbot.addWidget(com_fonte)
    assert com_fonte.seletor.count() == 1

    sem_registo = PainelDeFigura(construtores=())
    qtbot.addWidget(sem_registo)
    assert sem_registo.seletor.count() == 0, "sem registo e sem fonte tem de construir"


# ---------------------------------------------------------------------------
# A ligação: o painel dos GRÁFICOS hospeda, e a janela alimenta
# ---------------------------------------------------------------------------
#
# O hospedeiro mudou no plano dos gráficos. Era `PainelDeQualidade`, e
# passou a ser `PainelDeGraficos` — que é dono também dos cinco corpos guardados.
# O que estes testes afirmam não mudou: o painel de figura está no layout, o corpo do fio chega, e
# o construtor certo é escolhido.


def test_o_painel_de_graficos_HOSPEDA_a_figura_e_o_widget_esta_no_layout(qtbot: QtBot) -> None:  # noqa: N802
    """A ligação de composição, e ela já foi esquecida uma vez.

    *"O painel foi composto"* é verdade estrutural e falsa funcionalmente quando ninguém o põe no
    layout, na célula que decide o que o usuário vê.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    assert isinstance(graficos.figuras, PainelDeFigura)
    pai = graficos.figuras.parentWidget()
    assert pai is not None
    disposicao = pai.layout()
    assert disposicao is not None and disposicao.indexOf(graficos.figuras) >= 0, (
        "o painel da figura foi construído e não entrou no layout — um widget assim não é desenhado"
    )
    assert graficos.figuras.estado_visivel()["repouso"] == texto_do_estado(
        EstadoDaFigura.SEM_EXECUCAO
    )


# ---------------------------------------------------------------------------
# A trajetória do QC chega ao painel, e as duas bijeções da ligação
# ---------------------------------------------------------------------------
#
# **O que este bloco prova, e o que fica em `tests/unit/test_plots_deriva.py`.** Ali prova-se o
# construtor — o que é desenhado, contra que eixo, com que recusa. Aqui prova-se a **ligação**: que
# o corpo de `/matrizes/diagnosticar` chega ao painel, que o painel escolhe o construtor certo, e
# que a lista de construtores que ele alimenta não diverge do registo.

TRAJETORIA_DE_TESTE: Final[dict[str, Any]] = {
    # As ordens NÃO são consecutivas de propósito: é a âncora que separa «desenhou contra a ordem
    # de injeção» de «desenhou contra a posição».
    "scores": [-12.0, -8.5, -4.0, 1.5, 6.0, 11.5, 18.0],
    "ordens_de_injecao": [1, 12, 23, 34, 45, 56, 68],
    "lotes": ["A"] * 7,
    "n_features_usadas": 40,
    "n_features_totais": 100,
}


def test_u9_a_lista_de_construtores_do_painel_BIJETA_com_o_registo(qtbot: QtBot) -> None:  # noqa: N802
    """A bijeção da ligação, nas duas direções — e ela não é a bijeção de R1.

    R1 liga o registo ao catálogo `[G-1]` e a `ITENS_OBRIGATORIOS`. Esta liga o registo aos
    **montadores** do painel: um montador que o registo não conheça nunca apareceria no combo, e um
    construtor do registo sem montador aqui apareceria no combo e levantaria ao ser escolhido.

    ## O ORÁCULO MUDOU no plano dos gráficos, e o anterior era trivialmente satisfeito

    Até 15/09/2026 o referente era `CONSTRUTORES_DESTE_PAINEL`, uma constante que enumerava *"que
    construtores do registo este painel sabe alimentar"*, e `construtores_deste_painel()` filtrava
    o registo por ela. **Medido nesse dia: 13 de 13**, com `registo - painel` e `painel - registo`
    os dois vazios — o filtro era a **identidade**, e a constante uma segunda cópia do registo.

    > **E a direção que faltava é a que custa.** A asserção que restava era
    > `{funções do registo} >= CONSTRUTORES_DESTE_PAINEL`, um **superset**: um construtor novo no
    > registo sem entrada na constante passava, e o desfecho era a figura nova **omitida do combo
    > em silêncio**: o oráculo acusava as instâncias já consertadas, e não as que
    > sobraram.

    O referente novo é `_montadores()`, que é a lista do que este painel sabe construir de facto, e
    a asserção é de **igualdade**: a bijeção reprova nas duas direções.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    montadores = graficos._montadores()
    assert montadores, "o painel não alimenta construtor nenhum: as asserções abaixo seriam vácuas"

    registados = {c.funcao for c in CONSTRUTORES}
    assert registados, "âncora: `plots.CONSTRUTORES` está vazio e a igualdade abaixo seria vácua"
    assert set(montadores) == registados, (
        f"montadores sem construtor no registo: {sorted(set(montadores) - registados)}; "
        f"construtores do registo sem montador: {sorted(registados - set(montadores))}. Os "
        "primeiros são nome que sobreviveu à função que o justificava; os segundos aparecem no "
        "combo e levantam `FonteAusenteError` ao serem escolhidos"
    )

    assert graficos.figuras.seletor.count() == len(registados)


def test_u9_o_painel_de_graficos_DESENHA_a_trajetoria_recebida(qtbot: QtBot) -> None:  # noqa: N802
    """O par causal da ligação inteira: com diagnóstico desenha, sem diagnóstico retira.

    **A metade de cima é a que ninguém escreve.** Um painel que desenhasse sempre passaria na
    metade de baixo, e o que ele mostraria sem diagnóstico nenhum seria um eixo montado e vazio.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    # (a) Sem diagnóstico: a execução corrente não produz figura, e o texto diz o porquê — que,
    # hoje, é a falta do DIAGNÓSTICO, e não população vazia: a figura é dele.
    assert graficos.figuras.define_execucao(_resposta(200, _corpo(1))) == texto_do_estado(
        EstadoDaFigura.SEM_DIAGNOSTICO
    )
    assert graficos.figuras.estado_visivel()["figura"] == ""

    # (b) Com diagnóstico: a mesma execução, agora, desenha.
    graficos.recebe_diagnostico({"trajetoria_dos_qcs": dict(TRAJETORIA_DE_TESTE)})
    assert graficos.figuras.define_execucao(_resposta(200, _corpo(2))) == texto_do_estado(
        EstadoDaFigura.DESENHADA
    )
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"

    eixo = _eixo(graficos.figuras)
    desenhada = series(eixo)[0]
    assert desenhada.x.tolist() == TRAJETORIA_DE_TESTE["ordens_de_injecao"]
    assert desenhada.y.tolist() == TRAJETORIA_DE_TESTE["scores"]
    assert confere_vista(eixo) == []
    assert confere_numeros(graficos.figuras.canvas.figure, ["1", "40", "100", "7"]) == []


def test_d547_a_figura_do_diagnostico_desenha_SEM_execucao_e_a_da_execucao_nao(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """Retorno de quem testou: ver as figuras que já têm dado antes de executar o pipeline.

    **Par causal:** o mesmo painel, sem e com diagnóstico, nenhuma execução nos dois; só o segundo
    desenha. E uma figura da execução continua à espera dela.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.define_matriz("matriz A")
    _escolhe(graficos, "figura_da_deriva_do_qc")
    assert graficos.figuras.reavalia() == texto_do_estado(EstadoDaFigura.SEM_DIAGNOSTICO)

    graficos.carimba_o_diagnostico()
    graficos.recebe_diagnostico(_com_pca())
    graficos.recebe_pca(_com_pca())
    assert graficos.figuras.reavalia() == texto_do_estado(EstadoDaFigura.DESENHADA)
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"
    for funcao in ("figura_do_scree", "figura_dos_scores", "figura_das_cargas", "figura_do_biplot"):
        _escolhe(graficos, funcao)
        assert graficos.figuras.estado_visivel()["figura"] != "", funcao

    # O exemplo da figura que espera a execução era o violino, e ele deixou de esperar:
    # antes do primeiro corretor ele é figura do diagnóstico. O painel da correção, que é da mesma
    # etapa 1, continua à espera — desde que cada etapa é dona das próprias figuras, à espera da
    # etapa dela, com a frase que a nomeia, e não do pipeline.
    _escolhe(graficos, "figura_do_painel_da_correcao")
    assert graficos.figuras.estado_visivel()["figura"] == ""
    assert graficos.figuras.repouso.text() == frase_da_etapa_sem_resultado(1)


def test_d547_a_figura_do_diagnostico_sai_com_a_MATRIZ_e_nao_com_a_execucao(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """A obsolescência dela é a da matriz: um `202` não a tira, trocar de matriz tira."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.define_matriz("matriz A")
    graficos.carimba_o_diagnostico()
    graficos.recebe_diagnostico(_com_pca())
    _escolhe(graficos, "figura_da_deriva_do_qc")
    assert graficos.figuras.reavalia() == texto_do_estado(EstadoDaFigura.DESENHADA)

    assert graficos.figuras.define_execucao(_resposta(202, {})) == texto_do_estado(
        EstadoDaFigura.DESENHADA
    ), "um 202 tirou a figura do diagnostico, que nao e da execucao"

    graficos.define_matriz("matriz B")
    assert graficos.figuras.reavalia() == texto_do_estado(EstadoDaFigura.SEM_DIAGNOSTICO)
    assert graficos.figuras.estado_visivel()["figura"] == ""

    # E o diagnóstico de B, quando chega, volta a desenhar.
    graficos.carimba_o_diagnostico()
    graficos.recebe_diagnostico(_com_pca())
    assert graficos.figuras.reavalia() == texto_do_estado(EstadoDaFigura.DESENHADA)


def test_u9_corpo_de_diagnostico_ILEGIVEL_retira_a_figura(qtbot: QtBot) -> None:  # noqa: N802
    """Guarda de fronteira: o corpo vem do fio e nada nele passou pelo mypy.

    Um `None`, uma lista, um dicionário sem a raiz — os três chegam à mesma frase, e nenhum deles
    pode fazer um *slot* do Qt levantar: excepção num slot é engolida para o `stderr` e o painel
    para de actualizar sem dizer nada.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    for semente, corpo in enumerate(
        (None, [], {}, {"trajetoria_dos_qcs": None}, {"trajetoria_dos_qcs": []}), start=10
    ):
        graficos.recebe_diagnostico(corpo)
        escrito = graficos.figuras.define_execucao(_resposta(200, _corpo(semente)))
        assert escrito == texto_do_estado(EstadoDaFigura.SEM_DADO), corpo


def test_u9_o_painel_RECUSA_um_construtor_que_nao_sabe_alimentar(qtbot: QtBot) -> None:  # noqa: N802
    """A direção oposta da bijeção, em processo: um construtor de fora não é servido em silêncio.

    Servir a deriva para um pedido de volcano seria a figura certa com o rótulo errado — e o combo
    mostra o rótulo do construtor pedido, então ninguém veria.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    with pytest.raises(FonteAusenteError, match="não sabe construí-lo"):
        graficos._constroi_figura(_um_construtor())


# ---------------------------------------------------------------------------
# As três distribuições chegam ao painel, e a ORDEM da chamada é o conteúdo
# ---------------------------------------------------------------------------
#
# **O que este bloco prova, e o que fica em `tests/unit/test_plots_distribuicao.py`.** Ali prova-se
# o construtor — o que é desenhado, com que recusa, com que declaração. Aqui prova-se a
# **ligação**: que o corpo de `/execucoes` chega ao painel, que cada montador escolhe os vetores
# certos do resumo, e que o resumo velho não sobrevive a uma resposta que não o traga.

CAIXAS_DO_FIO: Final[list[dict[str, Any]]] = [
    {
        "feature": nome,
        "bigode_inferior": base - 2.0,
        "q1": base - 1.0,
        "mediana": base,
        "q3": base + 1.0,
        "bigode_superior": base + 2.0,
        "n_discrepantes": 0,
    }
    for base, nome in zip((10.0, 20.0, 30.0, 40.0), NOMES, strict=True)
]

# O «depois» DIFERE do «antes», e a diferença é a âncora deste bloco. Um resumo em que
# os dois lados fossem iguais faria toda asserção sobre *"a figura mostra o efeito"* passar por
# coincidência — é o caso da matriz já normalizada que o plano nomeia.
CAIXAS_DEPOIS_DO_FIO: Final[list[dict[str, Any]]] = [
    {
        **caixa,
        "bigode_inferior": -2.0,
        "q1": -1.0,
        "mediana": 0.0,
        "q3": 1.0,
        "bigode_superior": 2.0,
    }
    for caixa in CAIXAS_DO_FIO
]

# As médias por eixo têm TAMANHOS DIFERENTES de propósito: é o que torna «trocar os dois vetores»
# uma mutação detetável em vez de um par que casa por acaso.
MEDIAS_POR_FEATURE_ANTES: Final = [11.0, 21.0, 31.0, 41.0]
MEDIAS_POR_FEATURE_DEPOIS: Final = [-1.0, -0.2, 0.4, 1.1]
MEDIAS_POR_AMOSTRA_ANTES: Final = [12.0, 24.0, 36.0]
MEDIAS_POR_AMOSTRA_DEPOIS: Final = [-0.7, 0.1, 0.9]

# **As caixas por AMOSTRA, e os números NÃO são os das médias por amostra.** Os dois campos vêm do
# mesmo eixo e têm a mesma cardinalidade, então trocá-los não levanta — a figura sairia bem formada
# com uma caixa construída de um número só. O intervalo que cada uma cobre é a planta que separa.
NOMES_DE_AMOSTRA: Final = ("QC_00_A", "ID99_Asymptomatic", "ID41_ATLL")

CAIXAS_POR_AMOSTRA_DO_FIO: Final[list[dict[str, Any]]] = [
    {
        "amostra": nome,
        "bigode_inferior": base - 40.0,
        "q1": base - 20.0,
        "mediana": base,
        "q3": base + 20.0,
        "bigode_superior": base + 40.0,
        "n_discrepantes": 0,
    }
    for base, nome in zip((300.0, 500.0, 700.0), NOMES_DE_AMOSTRA, strict=True)
]

# O «depois» difere do «antes» — a âncora também neste eixo.
CAIXAS_POR_AMOSTRA_DEPOIS_DO_FIO: Final[list[dict[str, Any]]] = [
    {
        **caixa,
        "bigode_inferior": -4.0,
        "q1": -2.0,
        "mediana": 0.0,
        "q3": 2.0,
        "bigode_superior": 4.0,
    }
    for caixa in CAIXAS_POR_AMOSTRA_DO_FIO
]

RESUMO_DE_TESTE: Final[dict[str, Any]] = {
    "matriz_antes": "ingerida",
    "matriz_depois": "processada",
    "caixas_antes": CAIXAS_DO_FIO,
    "caixas_depois": CAIXAS_DEPOIS_DO_FIO,
    "caixas_por_amostra_antes": CAIXAS_POR_AMOSTRA_DO_FIO,
    "caixas_por_amostra_depois": CAIXAS_POR_AMOSTRA_DEPOIS_DO_FIO,
    "medias_por_feature_antes": MEDIAS_POR_FEATURE_ANTES,
    "medias_por_feature_depois": MEDIAS_POR_FEATURE_DEPOIS,
    "medias_por_amostra_antes": MEDIAS_POR_AMOSTRA_ANTES,
    "medias_por_amostra_depois": MEDIAS_POR_AMOSTRA_DEPOIS,
    "n_caixas": 4,
    "n_caixas_por_amostra": 3,
    "teto_de_caixas": 50,
    "n_features_comparadas": 4,
    "n_features_antes": 9,
    "n_features_depois": 4,
    "n_amostras": 3,
    "n_amostras_comparadas": 3,
    "amostrada": False,
    "amostrada_por_amostra": False,
    "seed": 4242,
}


def _com_distribuicoes(seed: int) -> dict[str, Any]:
    """O corpo de `/execucoes` com o resumo dentro, que é como ele chega na produção."""
    return {**_corpo(seed), "distribuicoes": dict(RESUMO_DE_TESTE)}


def _escolhe(painel: PainelDeGraficos, funcao: str) -> None:
    """Põe o combo na figura chamada `funcao`. Levanta se ela não estiver lá — sem recuo."""
    alvo = [c.funcao for c in CONSTRUTORES].index(funcao)
    painel.figuras.seletor.setCurrentIndex(alvo)


def test_u4_a_lista_do_painel_ganhou_as_QUATRO_e_continua_a_bijetar(qtbot: QtBot) -> None:  # noqa: N802
    """A bijeção da ligação com as QUATRO desta unidade — e o combo mostra tudo o que o painel tem.

    **O TOTAL não se afirma aqui, e a ausência é decisão.** Este teste
    nasceu com `len(do_painel) == 5`, que era o total naquele dia; depois entrou a sexta figura
    e o literal passou a ser a irmã que ninguém procura. **Quem é dono do total é o teste da unidade
    mais recente** — hoje `test_u6_a_lista_do_painel_ganhou_o_VOLCANO_e_continua_a_bijetar` —, e o
    que fica aqui é o que esta unidade acrescentou: as quatro de distribuição, nomeadas.

    Nomear as quatro é **mais** forte que contar cinco: a contagem passa se uma delas sair e outra
    entrar, e a lista não.

    > **A bijeção contra o registo entrou aqui em 15/09/2026, e não é a contagem.** Ela é uma
    > igualdade de CONJUNTOS entre `_montadores()` e `plots.CONSTRUTORES`, derivada dos dois lados;
    > o literal do total continua a viver num sítio só, no teste das SETE. A asserção anterior era
    > um **superset** contra a constante que o plano dos gráficos removeu, e passava com um
    > construtor novo sem montador — a figura nova ficava fora do combo em silêncio.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    montadores = graficos._montadores()
    das_distribuicoes = {
        "figura_das_caixas_por_feature",
        "figura_das_caixas_por_amostra",
        "figura_da_densidade_por_feature",
        "figura_da_densidade_por_amostra",
    }
    assert das_distribuicoes <= set(montadores), sorted(montadores)
    assert set(montadores) == {c.funcao for c in CONSTRUTORES}
    assert graficos.figuras.seletor.count() == len(montadores)


@pytest.mark.parametrize(
    ("funcao", "eixos", "x_esperado"),
    [
        ("figura_das_caixas_por_feature", 2, None),
        ("figura_das_caixas_por_amostra", 2, None),
        ("figura_da_densidade_por_feature", 2, MEDIAS_POR_FEATURE_ANTES),
        ("figura_da_densidade_por_amostra", 2, MEDIAS_POR_AMOSTRA_ANTES),
    ],
)
def test_u4_cada_montador_desenha_do_VETOR_certo_do_resumo(  # noqa: N802
    qtbot: QtBot, funcao: str, eixos: int, x_esperado: list[float] | None
) -> None:
    """**A armadilha semântica da unidade, na camada da ligação: qual vetor alimenta qual figura.**

    Os dois vetores de médias são a mesma forma — uma lista de `float` —, então trocá-los não
    levanta nada: a figura sairia bem formada, com o título certo e a curva de outra população. É o
    rótulo certo sobre a quantidade errada, e o que a apanha é uma asserção sobre o
    **intervalo** que a curva cobre, que é derivado do vetor que devia entrar.

    Os dois vetores têm **tamanhos diferentes** no resumo de teste de propósito — 4 features contra
    3 amostras —, e é isso que torna a troca detetável em vez de um par que casa por acaso.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_distribuicoes(_com_distribuicoes(1), corresponde=True)
    assert graficos.figuras.define_execucao(_resposta(200, _com_distribuicoes(1))) is not None
    _escolhe(graficos, funcao)

    figura = graficos.figuras.canvas.figure
    assert graficos.figuras.estado_visivel()["figura"] == f"{eixos} eixo(s)"
    assert len(figura.axes) == eixos
    for eixo in figura.axes:
        assert confere_vista(eixo) == []

    if x_esperado is None:
        return
    curva = series(figura.axes[0])[0]
    assert float(curva.x.min()) < min(x_esperado), (
        f"a curva do «antes» cobre [{curva.x.min()}, {curva.x.max()}] e o vetor que devia "
        f"alimentá-la é {x_esperado} — o montador leu o vetor do outro eixo"
    )
    assert float(curva.x.max()) > max(x_esperado)
    outro = (
        MEDIAS_POR_AMOSTRA_ANTES
        if x_esperado is MEDIAS_POR_FEATURE_ANTES
        else MEDIAS_POR_FEATURE_ANTES
    )
    assert max(outro) != max(x_esperado), (
        "âncora: os dois vetores têm de diferir para a planta valer"
    )


def test_u4_a_SAMPLE_VIEW_desenha_das_CAIXAS_por_amostra_e_nao_das_medias(qtbot: QtBot) -> None:  # noqa: N802
    """**A armadilha desta figura, na camada da ligação — e ela não é a dos dois vetores.**

    `caixas_por_amostra_*` e `medias_por_amostra_*` vêm do MESMO eixo e têm a MESMA cardinalidade,
    então um montador que lesse o campo errado não levantaria: desenharia três caixas construídas
    de um número só, com o título certo. É o rótulo certo sobre a quantidade errada,
    e foi por causa disso que esta figura esperou pelo fio em vez de ser montada das médias.

    **A planta cai sobre dado em que as duas coisas diferem**: as médias por amostra do
    resumo de teste vivem em `[-0,7; 36]` e as caixas em `[260; 740]`, e os dois intervalos são
    disjuntos — afirmado aqui, sobre o próprio dicionário, antes de a figura ser lida.
    """
    medias = [*MEDIAS_POR_AMOSTRA_ANTES, *MEDIAS_POR_AMOSTRA_DEPOIS]
    extremos = [c["bigode_inferior"] for c in CAIXAS_POR_AMOSTRA_DO_FIO]
    assert max(medias) < min(extremos), (
        f"âncora: as médias {medias} e as caixas {extremos} têm de ocupar faixas disjuntas, ou o "
        "montador que lesse o campo errado desenharia no mesmo sítio"
    )
    assert len(MEDIAS_POR_AMOSTRA_ANTES) == len(CAIXAS_POR_AMOSTRA_DO_FIO), (
        "âncora ao contrário: os dois campos TÊM a mesma cardinalidade, e é por isso que a troca "
        "não levanta — a asserção tem de ser sobre o valor, nunca sobre o comprimento"
    )

    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_distribuicoes(_com_distribuicoes(7), corresponde=True)
    assert graficos.figuras.define_execucao(_resposta(200, _com_distribuicoes(7))) is not None
    _escolhe(graficos, "figura_das_caixas_por_amostra")

    figura = graficos.figuras.canvas.figure
    desenhadas = series(figura.axes[0])
    assert desenhadas, "âncora: o lado «antes» não desenhou artista nenhum"
    valores = [float(y) for s in desenhadas for y in s.y]
    assert min(valores) >= min(extremos), (
        f"o «antes» desenhou em [{min(valores)}, {max(valores)}] e as caixas do corpo começam em "
        f"{min(extremos)} — o montador leu `medias_por_amostra_antes`"
    )

    # E cada artista leva o nome da AMOSTRA, que é quem responde por R2 nesta figura.
    rotulos = {s.rotulo.split(" · ")[0] for s in desenhadas}
    assert rotulos == set(NOMES_DE_AMOSTRA), rotulos
    assert not (rotulos & set(NOMES)), "um artista leva nome de feature numa figura por amostra"


def test_u4_sem_resumo_a_figura_e_RETIRADA_com_texto_e_nao_fica_em_branco(qtbot: QtBot) -> None:  # noqa: N802
    """O par causal da ligação: com resumo desenha, sem resumo retira.

    **A metade de cima é a que ninguém escreve.** Um painel que desenhasse sempre passaria na
    metade de baixo, e o que ele mostraria sem resumo nenhum seria um eixo montado e vazio.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    # (a) sem resumo: a execução corrente não produz figura, e o texto diz o porquê. O `recebe_*`
    # corre como na janela, antes da chave: sem ele a figura não teria dono, e esperaria a etapa.
    graficos.recebe_distribuicoes(_corpo(1), corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _corpo(1)))
    _escolhe(graficos, "figura_das_caixas_por_feature")
    assert graficos.figuras.estado_visivel()["repouso"] == texto_do_estado(EstadoDaFigura.SEM_DADO)
    assert graficos.figuras.estado_visivel()["figura"] == ""

    # (b) com resumo: a mesma escolha, agora, desenha.
    graficos.recebe_distribuicoes(_com_distribuicoes(2), corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _com_distribuicoes(2)))
    assert graficos.figuras.estado_visivel()["figura"] == "2 eixo(s)"


@pytest.mark.parametrize(
    "corpo", [None, [], {}, {"distribuicoes": None}, {"distribuicoes": []}, {"distribuicoes": 7}]
)
def test_u4_corpo_de_execucao_ILEGIVEL_retira_a_figura(qtbot: QtBot, corpo: Any) -> None:  # noqa: N802
    """Guarda de fronteira: nada do corpo passou pelo mypy, e nada pode levantar no slot.

    Excepção dentro de um *slot* do Qt é engolida para o `stderr` e o painel para de actualizar sem
    dizer nada — que é o modo de falha silenciosa que este painel já pagou uma vez.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_distribuicoes(corpo, corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _corpo(3)))
    _escolhe(graficos, "figura_das_caixas_por_feature")
    assert graficos.figuras.estado_visivel()["repouso"] == texto_do_estado(EstadoDaFigura.SEM_DADO)


def test_u4_um_202_APAGA_o_resumo_da_execucao_anterior(qtbot: QtBot) -> None:  # noqa: N802
    """**R4 pela via do resumo: o número velho não pode sobreviver ao trabalho em curso.**

    A chave já retira a figura num `202`; esta asserção é sobre o que fica **guardado**. Um
    `recebe_distribuicoes` que só corresse no `200` deixaria o resumo anterior no atributo, e a
    figura voltaria a ser desenhada com ele assim que o combo mudasse.

    ## A expectativa da última linha MUDOU em 11/09/2026, e a intenção do teste não

    Ela era `SEM_DADO`, e `SEM_DADO` era o **sintoma do defeito** que a revisão daquele dia achou.
    O painel limpava o resumo guardado, o combo redesenhava sem portão de obsolescência, o
    construtor recebia população vazia e levantava — e a tela dizia *"o construtor recebeu
    população vazia"* sobre o fato *"a execução ainda corre"*. Esse é o colapso que
    `EstadoDaFigura` existe para não ter, e o teste afirmava-o.

    **O que este teste quer continua a ser o que a linha 1654 diz** — o resumo velho não volta —, e
    isso passou a ser verdade por uma razão mais forte: a figura não é reconstruída de todo. A
    negativa acrescentada é o que impede o colapso de regressar sem ninguém ver.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_distribuicoes(_com_distribuicoes(4), corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _com_distribuicoes(4)))
    _escolhe(graficos, "figura_das_caixas_por_feature")
    assert graficos.figuras.estado_visivel()["figura"] == "2 eixo(s)", "âncora: desenhou"

    graficos.recebe_distribuicoes({"estado": "em_curso"}, corresponde=True)
    graficos.figuras.define_execucao(_resposta(202, {"estado": "em_curso"}))
    assert graficos.figuras.estado_visivel()["figura"] == ""
    # E depois do `202`, escolher de novo NÃO ressuscita o resumo velho.
    _escolhe(graficos, "figura_da_densidade_por_feature")
    visivel = graficos.figuras.estado_visivel()
    assert visivel["figura"] == "", "escolher outra figura ressuscitou o canvas durante o `202`"
    assert visivel["repouso"] == texto_do_estado(EstadoDaFigura.EM_CURSO)
    assert visivel["repouso"] != texto_do_estado(EstadoDaFigura.SEM_DADO), (
        "voltou a dizer «população vazia» sobre uma execução que ainda corre"
    )


def test_u4_o_recorte_desenhado_declara_as_DUAS_pontas_que_vieram_do_FIO(qtbot: QtBot) -> None:  # noqa: N802
    """A classe (d) na camada da ligação: os números do rodapé são os do corpo, não outros.

    **Derivados do dicionário de teste e não do painel** — se o teste lesse o mesmo `Recorte` que o
    construtor usou, os dois errariam juntos.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_distribuicoes(_com_distribuicoes(5), corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _com_distribuicoes(5)))
    _escolhe(graficos, "figura_das_caixas_por_feature")

    rodapes = [t for t in graficos.figuras.canvas.figure.texts if t.get_va() == "bottom"]
    assert len(rodapes) == 1, rodapes
    escrito = " ".join(rodapes[0].get_text().split())
    comparadas = RESUMO_DE_TESTE["n_features_comparadas"]
    disponiveis = RESUMO_DE_TESTE["n_features_antes"]
    assert f"{comparadas} de {disponiveis} features" in escrito, escrito
    assert str(RESUMO_DE_TESTE["matriz_antes"]) in escrito
    assert str(RESUMO_DE_TESTE["matriz_depois"]) in escrito
    # O coeficiente de Tukey do rodapé é o do ESQUEMA, injetado pelo painel — não um literal do
    # lado do desenho, que seria a segunda fonte de uma verdade só.
    assert f"{COEFICIENTE_DE_TUKEY:g} x IQR" in escrito, escrito


# ---------------------------------------------------------------------------
# O volcano na ligação, e é a PRIMEIRA figura deste painel que devolve nomes
# ---------------------------------------------------------------------------
#
# **Os nomes são o que muda de gênero nesta unidade.** As cinco figuras anteriores devolvem `()` no
# segundo elemento: a trajetória põe um ponto por QC, e as caixas põem artistas cujo índice de
# `pick_event` é o do ponto **dentro** do artista. O volcano põe um ponto por feature num artista
# só, e é o primeiro caso em que `PainelDeFigura._identifica` consegue dizer *qual feature é* pelo
# vetor — que é literalmente o que R2 pede.

# **O material do fio: cinco features, e cada uma existe para um desfecho.** Duas significativas em
# sentidos opostos, uma não significativa, e duas de indefinido — uma por eixo.
#
# A âncora está nos VALORES e não na forma: os `log2fc` das duas significativas têm
# sinais opostos, senão a asserção sobre as classes passaria com um classificador que devolvesse
# sempre a mesma; e nenhum ponto tem `x == y`, senão a planta dos eixos trocados seria a identidade.
FEATURES_DO_VOLCANO: Final = (
    "0.50_111.1m/z",
    "1.50_222.2m/z",
    "2.50_333.3m/z",
    "3.50_444.4m/z",
    "4.50_555.5m/z",
)

VOLCANO_DE_TESTE: Final[dict[str, Any]] = {
    "features": list(FEATURES_DO_VOLCANO),
    "log2fc": [-2.0, 3.0, 0.75, None, 1.5],
    "pvalue": [0.0002, 0.0005, 0.30, 0.01, None],
    "qvalue_fdr": [0.004, 0.008, 0.40, 0.02, None],
    "grupo_a": "Asymptomatic",
    "grupo_b": "HAMTSP",
    "n_grupo_a": 18,
    "n_grupo_b": 14,
    "parametrico": False,
    "rota_da_normalidade": "shapiro",
    "shapiro_pvalue": 2.3664e-95,
    "n_residuos": 5000,
    "alfa_fdr": 0.05,
    "correcao": "Benjamini-Hochberg (FDR)",
    "matriz_do_pvalue": "processada",
    "matriz_do_fold_change": "pré-transformação",
    "matrizes_diferem": True,
    "n_features_testadas": 5,
}

# As TRÊS desenháveis, na ordem do fio — as duas últimas caem por indefinido.
DESENHAVEIS_DO_VOLCANO: Final = FEATURES_DO_VOLCANO[:3]


def _com_volcano(seed: int, **troca: Any) -> dict[str, Any]:
    """O corpo de `/execucoes` com a raiz `volcano` dentro, que é como ele chega na produção."""
    return {**_corpo(seed), "volcano": {**VOLCANO_DE_TESTE, **troca}}


def _pick(figura: Figure, artista: Any, posicao: int) -> PickEvent:
    """Um `PickEvent` sobre `posicao` daquele artista, na forma que o `matplotlib` produz."""
    rato = MouseEvent("button_press_event", figura.canvas, 10, 10)
    return PickEvent("pick_event", figura.canvas, rato, artista, ind=np.array([posicao]))


def _rodape_desenhado(painel: PainelDeFigura) -> str:
    """O rodapé da figura na tela, **com o espaço em branco colapsado**.

    `recorte.aplica` quebra a frase em linhas pela largura útil, e a quebra cai onde calhar. Buscar
    substring no texto cru dá **falso negativo** para qualquer frase que atravesse a quebra, e a
    direção é a perigosa: o veredito lê-se como *"a figura não declara isto"*.
    """
    rodapes = [t for t in painel.canvas.figure.texts if t.get_va() == "bottom"]
    assert len(rodapes) == 1, rodapes
    return " ".join(rodapes[0].get_text().split())


def test_u6_a_lista_do_painel_ganhou_o_VOLCANO_e_continua_a_bijetar(qtbot: QtBot) -> None:  # noqa: N802
    """A bijeção da ligação, com o volcano dentro — e o combo mostra todas.

    **A contagem literal saiu daqui e foi para
    `test_u8_a_lista_do_painel_ganhou_as_SETE_e_continua_a_bijetar`, que é onde ela é de hoje:**
    duas asserções sobre o mesmo total são irmãs que divergem, e aquele teste acrescentou sete
    figuras. O que
    fica aqui é o que este teste sempre provou — que o volcano está na lista e que a bijeção fecha.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    montadores = graficos._montadores()
    assert "figura_do_volcano" in montadores, sorted(montadores)
    assert set(montadores) == {c.funcao for c in CONSTRUTORES}
    assert graficos.figuras.seletor.count() == len(montadores)


def test_u6_o_painel_DESENHA_o_volcano_e_passa_os_NOMES_das_desenhadas(qtbot: QtBot) -> None:  # noqa: N802
    """**O que esta unidade acrescenta à ligação: o vetor de nomes deixa de ser vazio.**

    A âncora vem primeiro: as figuras anteriores passam `()`, então uma asserção de que os nomes
    chegaram tem de mostrar que aqui eles NÃO são vazios — senão ela passaria sobre a tupla vazia
    que todas as outras devolvem.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_volcano(_com_volcano(11))
    assert graficos.figuras.define_execucao(_resposta(200, _com_volcano(11))) is not None
    _escolhe(graficos, "figura_do_volcano")

    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"
    # (a) os nomes chegaram, são as DESENHÁVEIS, e estão na ordem do fio.
    assert graficos.figuras._nomes == DESENHAVEIS_DO_VOLCANO, graficos.figuras._nomes
    assert graficos.figuras._nomes != (), (
        "âncora: o painel passou a tupla vazia que as outras cinco figuras passam, e a "
        "identificação de R2 cairia no rótulo do artista"
    )
    # (b) e o artista tem exactamente esse número de pontos — senão o índice desalinha.
    figura = graficos.figuras.canvas.figure
    assert series(figura.axes[0])[0].x.size == len(DESENHAVEIS_DO_VOLCANO)
    assert confere_vista(figura.axes[0]) == []


def test_u6_a_identificacao_diz_QUAL_FEATURE_e_e_nao_o_rotulo_da_serie(qtbot: QtBot) -> None:  # noqa: N802
    """**R2 ponta a ponta, e é o par causal do vetor de nomes.**

    Sem o vetor, `_identifica` cai em `str(evento.artist.get_label())` e escreve o mesmo texto para
    todos os pontos. A ponta A: com nomes, cada índice dá a sua feature. A ponta B: sem nomes, dá o
    rótulo do artista — e é a mesma figura, o que torna a diferença atribuível ao vetor.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_volcano(_com_volcano(12))
    graficos.figuras.define_execucao(_resposta(200, _com_volcano(12)))
    _escolhe(graficos, "figura_do_volcano")

    painel = graficos.figuras
    figura = painel.canvas.figure
    artista = figura.axes[0].collections[0]

    for posicao, esperado in enumerate(DESENHAVEIS_DO_VOLCANO):
        painel._identifica(_pick(figura, artista, posicao))
        assert painel._anotacao is not None
        assert painel._anotacao.get_text().startswith(esperado), (
            f"o ponto {posicao} identificou-se como {painel._anotacao.get_text()!r} e a feature "
            f"dele é {esperado!r}"
        )

    # A ponta B: o MESMO desenho sem o vetor cai no rótulo do artista, para todos.
    painel.desenha(figura, nomes=())
    artista = painel.canvas.figure.axes[0].collections[0]
    painel._identifica(_pick(painel.canvas.figure, artista, 1))
    assert painel._anotacao is not None
    assert painel._anotacao.get_text().startswith(ROTULO_DA_SERIE), (
        "sem o vetor de nomes a identificação devia cair no rótulo do artista — se não caiu, o "
        "par causal não separa o vetor de coisa nenhuma"
    )


def test_u6_sem_a_raiz_volcano_a_figura_e_RETIRADA_com_texto(qtbot: QtBot) -> None:  # noqa: N802
    """O par causal da ligação: com a raiz desenha, sem ela retira.

    **`None` na raiz é *"a execução correu sem par de grupos"*** — o teste univariado é pareado, e
    sem `group_a`/`group_b` a rota não o corre. Isso é *"não calculado"* e não *"calculado e
    vazio"*, e o painel diz o que é verdade em vez de desenhar um eixo montado e vazio.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    # (a) sem a raiz: retira, com texto.
    graficos.recebe_volcano(_corpo(13))
    graficos.figuras.define_execucao(_resposta(200, _corpo(13)))
    _escolhe(graficos, "figura_do_volcano")
    assert graficos.figuras.estado_visivel()["repouso"] == texto_do_estado(EstadoDaFigura.SEM_DADO)
    assert graficos.figuras.estado_visivel()["figura"] == ""

    # (b) com a raiz: a mesma escolha, agora, desenha.
    graficos.recebe_volcano(_com_volcano(14))
    graficos.figuras.define_execucao(_resposta(200, _com_volcano(14)))
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"


@pytest.mark.parametrize(
    "corpo", [None, [], {}, {"volcano": None}, {"volcano": []}, {"volcano": 7}, {"volcano": {}}]
)
def test_u6_raiz_volcano_ILEGIVEL_retira_a_figura(qtbot: QtBot, corpo: Any) -> None:  # noqa: N802
    """Guarda de fronteira: nada do corpo passou pelo mypy, e nada pode levantar no slot.

    Excepção dentro de um *slot* do Qt é engolida para o `stderr` e o painel para de actualizar sem
    dizer nada. As sete formas incluem o dicionário **vazio**, que é o que um `volcano: {}` daria —
    e ali quem recusa é `pontos_do_volcano`, por população vazia.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_volcano(corpo)
    graficos.figuras.define_execucao(_resposta(200, _corpo(15)))
    _escolhe(graficos, "figura_do_volcano")
    assert graficos.figuras.estado_visivel()["repouso"] == texto_do_estado(EstadoDaFigura.SEM_DADO)


def test_u6_um_202_APAGA_o_volcano_da_execucao_anterior(qtbot: QtBot) -> None:  # noqa: N802
    """**R4 pela via da raiz: os vetores velhos não podem sobreviver ao trabalho em curso.**

    A chave já retira a figura num `202`; esta asserção é sobre o que fica **guardado**. Um
    `recebe_volcano` que só corresse no `200` deixaria os vetores anteriores no atributo.

    **A prova de que o atributo foi apagado é a execução SEGUINTE, e não a escolha do combo.**
    Reescolher a mesma figura não dispara `currentIndexChanged`, e o estado fica em `EM_CURSO` —
    que é o correcto e não prova nada sobre o atributo. O que prova é um `200` novo **sem** a raiz:
    com o atributo apagado ele cai em `SEM_DADO`; com os vetores velhos lá dentro, ele **desenha**
    o volcano da execução anterior, que é exactamente o desfecho que R4 existe para impedir.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_volcano(_com_volcano(16))
    graficos.figuras.define_execucao(_resposta(200, _com_volcano(16)))
    _escolhe(graficos, "figura_do_volcano")
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)", "âncora: desenhou"

    graficos.recebe_volcano({"estado": "em_curso"})
    graficos.figuras.define_execucao(_resposta(202, {"estado": "em_curso"}))
    assert graficos.figuras.estado_visivel()["figura"] == ""
    assert graficos.figuras.estado_visivel()["repouso"] == texto_do_estado(EstadoDaFigura.EM_CURSO)

    # O `200` seguinte, sem a raiz: o atributo apagado é o que o faz cair em `SEM_DADO`.
    graficos.recebe_volcano(_corpo(19))
    graficos.figuras.define_execucao(_resposta(200, _corpo(19)))
    assert graficos.figuras.estado_visivel()["figura"] == "", (
        "a figura da execução anterior sobreviveu ao `202` e ao `200` sem raiz"
    )
    assert graficos.figuras.estado_visivel()["repouso"] == texto_do_estado(EstadoDaFigura.SEM_DADO)


def test_u6_o_rodape_desenhado_declara_o_que_veio_do_FIO_e_nao_outra_coisa(qtbot: QtBot) -> None:  # noqa: N802
    """A classe (d) na camada da ligação: os números do rodapé são os do corpo — R5, R5b, R8.

    **Derivados do dicionário de teste e não do painel.** Se o teste lesse o mesmo `Contraste` que
    o construtor usou, os dois errariam juntos.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_volcano(_com_volcano(17))
    graficos.figuras.define_execucao(_resposta(200, _com_volcano(17)))
    _escolhe(graficos, "figura_do_volcano")
    escrito = _rodape_desenhado(graficos.figuras)

    # R3: quantas de quantas, com os dois números do dicionário deste módulo.
    desenhadas = len(DESENHAVEIS_DO_VOLCANO)
    testadas = VOLCANO_DE_TESTE["n_features_testadas"]
    assert f"{desenhadas} de {testadas} features" in escrito, escrito
    # R8: os dois `n` de grupo e a correção — o que separa «não há efeito» de «não há poder».
    for chave_grupo, chave_n in (("grupo_a", "n_grupo_a"), ("grupo_b", "n_grupo_b")):
        assert f"{VOLCANO_DE_TESTE[chave_grupo]} (n = {VOLCANO_DE_TESTE[chave_n]})" in escrito
    assert str(VOLCANO_DE_TESTE["correcao"]) in escrito
    # R5: as duas procedências, cada uma no eixo dela.
    assert str(VOLCANO_DE_TESTE["matriz_do_pvalue"]) in escrito
    assert str(VOLCANO_DE_TESTE["matriz_do_fold_change"]) in escrito
    # R5b: o teste que o fio declara, e NÃO uma constante do lado do desenho. `parametrico=False`
    # neste dicionário, e uma legenda que dissesse «Welch» estaria errada sem nada levantar.
    assert MANN_WHITNEY in escrito and WELCH not in escrito


def test_u6_o_teste_declarado_SEGUE_a_bandeira_do_fio(qtbot: QtBot) -> None:  # noqa: N802
    """**O par causal de R5b na ligação: a mesma figura, as duas bandeiras, dois rótulos.**

    É a asserção que uma constante `"Welch"` no painel passaria a falhar. No `pos_hilic` real a
    bandeira é `False` — Shapiro-Wilk rejeita a normalidade dos resíduos —, então o caminho errado
    seria o que a tela mostraria de facto.
    """
    vistos = []
    for parametrico in (True, False):
        graficos = PainelDeGraficos()
        qtbot.addWidget(graficos)
        corpo = _com_volcano(18, parametrico=parametrico)
        graficos.recebe_volcano(corpo)
        graficos.figuras.define_execucao(_resposta(200, corpo))
        _escolhe(graficos, "figura_do_volcano")
        vistos.append(_rodape_desenhado(graficos.figuras))

    assert WELCH in vistos[0] and MANN_WHITNEY not in vistos[0]
    assert MANN_WHITNEY in vistos[1] and WELCH not in vistos[1]


# ---------------------------------------------------------------------------
# As QUATRO de PCA no fio, e a dívida declarada antes
# ---------------------------------------------------------------------------
#
# As quatro figuras foram construídas antes e NÃO ligadas, e o motivo foi medido. Uma entrada de
# `RAIZES_ANINHADAS` que nenhum módulo de `desktop/janela/` leia reprova em DOIS portões de
# `tests/test_contrato_janela_api.py` — por classe sem uso e por base declarada e não lida. O par
# modelo/leitor nasce junto ou não nasce, e o modelo é desta unidade.

# **A PCA que atravessa o fio, com DOIS componentes e TRÊS tipos de amostra.** Os `n_features_*` são
# um par de propósito: o passo 3 descarta coluna constante e não finita, e servir só o total faria o
# rodapé declarar a população errada.
PCA_DE_TESTE: Final[dict[str, Any]] = {
    "scores_pc1": [-2.0, -1.0, 0.5, 1.0, 2.5],
    "scores_pc2": [1.0, -0.5, 0.25, -1.5, 0.75],
    "tipos_de_amostra": ["qc", "sample", "sample", "validate", "qc"],
    "features": ["2.25_810.7739m/z", "3.10_155.0345m/z", "4.02_301.1400m/z"],
    "loadings_pc1": [0.6, -0.2, 0.1],
    "loadings_pc2": [0.1, 0.5, -0.3],
    "variancia_explicada": [0.4723, 0.0912],
    "n_features_usadas": 3,
    "n_features_totais": 6035,
    "n_componentes": 2,
}

# As quatro funções de PCA, na ordem do registo.
FIGURAS_DE_PCA: Final = (
    "figura_do_scree",
    "figura_dos_scores",
    "figura_das_cargas",
    "figura_do_biplot",
)

# Os três vetores do PLS-DA, e o par de validação exigido ao lado deles. `vip` traz um
# `None`: é *"não foi possível calcular"* e não *"importância zero"*, e o rodapé tem de o contar.
PLSDA_DE_TESTE: Final[dict[str, Any]] = {
    "q2": -0.0125,
    "permutation_pvalue": 0.1095,
    "r2y": 0.3498,
    "n_permutations": 200,
    "cv_folds": 7,
    "n_components": 2,
    "seed": 42,
    "scores_componente_1": [-1.5, -0.5, 1.0, 2.0],
    "scores_componente_2": [0.5, -1.0, 1.5, -0.25],
    # Os de validação cruzada, DIFERENTES dos de calibração de propósito: é o que deixa
    # o teste do painel dizer qual dos dois pares ele desenhou.
    "scores_cv_componente_1": [-0.9, 0.3, 0.4, 1.1],
    "scores_cv_componente_2": [0.2, -0.6, 0.9, 0.1],
    "grupos_por_amostra": ["Asymptomatic", "Asymptomatic", "HAMTSP", "HAMTSP"],
    "grupo_positivo": "HAMTSP",
    "grupo_negativo": "Asymptomatic",
    "n_positivo": 2,
    "n_negativo": 2,
    "features": ["2.25_810.7739m/z", "3.10_155.0345m/z", "4.02_301.1400m/z"],
    "loadings_componente_1": [0.7, -0.2, 0.1],
    "loadings_componente_2": [0.1, 0.6, -0.3],
    "vip": [1.6, 0.8, None],
    "n_features": 3,
}

FIGURAS_DE_PLSDA: Final = (
    "figura_dos_scores_do_plsda",
    "figura_das_cargas_do_plsda",
    "figura_do_vip",
)


def _com_pca(**troca: Any) -> dict[str, Any]:
    """O corpo de `/matrizes/diagnosticar` com a raiz `pca`, que é como ele chega na produção."""
    return {"trajetoria_dos_qcs": dict(TRAJETORIA_DE_TESTE), "pca": {**PCA_DE_TESTE, **troca}}


def _com_plsda(seed: int, **troca: Any) -> dict[str, Any]:
    """O corpo de `/execucoes` com a raiz `plsda` dentro."""
    return {**_corpo(seed), "plsda": {**PLSDA_DE_TESTE, **troca}}


def test_u8_a_lista_do_painel_ganhou_as_SETE_e_continua_a_bijetar(qtbot: QtBot) -> None:  # noqa: N802
    """Seis das primeiras, quatro de PCA, três de PLS-DA, o painel da correção e o violino —
    **quinze**.

    A contagem é literal de propósito: é a fonte independente que apanharia um construtor que
    entrasse no registo e não no painel. **Derivá-la do registo destruiria a independência** — as
    duas pontas passariam a errar juntas —, e é por isso que ela sobe de um em cada unidade que
    regista figura: 13 → 14 com o painel da correção, 14 → 15 com o violino, e daí até **32**
    pelo merge do bloco das análises (24), pela correlação (26), pelas particionais (28), pela
    classificação (30), que traz os dois classificadores, pela rede de correlação parcial (31), e
    pela curva do LOESS com o envelope sem um QC (32).
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    montadores = graficos._montadores()
    assert len(montadores) == 32, sorted(montadores)
    assert set(FIGURAS_DE_PCA) | set(FIGURAS_DE_PLSDA) <= set(montadores)
    assert set(montadores) == {c.funcao for c in CONSTRUTORES}
    assert graficos.figuras.seletor.count() == len(montadores)


@pytest.mark.parametrize("funcao", FIGURAS_DE_PCA)
def test_u8_as_quatro_de_pca_DESENHAM_do_corpo_do_diagnostico(qtbot: QtBot, funcao: str) -> None:  # noqa: N802
    """O par causal da ligação de PCA: com a raiz `pca` cada uma das quatro desenha.

    **Uma por uma e não em bloco**, porque cada montador lê vetores diferentes: um que passasse o
    vetor errado — as cargas onde vão os scores — produziria uma figura bem formada, e um teste
    sobre «alguma delas desenhou» não o veria.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_pca(_com_pca())
    assert graficos.figuras.define_execucao(_resposta(200, _corpo(21))) is not None
    _escolhe(graficos, funcao)

    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"
    eixo = graficos.figuras.canvas.figure.axes[0]
    assert confere_vista(eixo) == []
    assert series(eixo), f"{funcao} desenhou zero séries"


def test_u8_os_scores_de_pca_separam_QC_de_amostra_e_a_legenda_conta_cada_uma(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """**A única leitura de PCA que o laboratório regista, ponta a ponta pelo fio.**

    *"Espera-se o QC agrupado no meio da nuvem de amostras"* (`[14:23]`). Sem o vetor de tipos a
    figura teria uma nuvem só; o teste afirma que as três séries são endereçáveis **em separado** e
    que a contagem de cada uma bate com o vetor que veio do corpo.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_pca(_com_pca())
    graficos.figuras.define_execucao(_resposta(200, _corpo(22)))
    _escolhe(graficos, "figura_dos_scores")

    eixo = graficos.figuras.canvas.figure.axes[0]
    por_rotulo = {s.rotulo: s for s in series(eixo)}
    tipos = PCA_DE_TESTE["tipos_de_amostra"]
    for rotulo, quantas in (
        ("QC — 2 amostras", tipos.count("qc")),
        ("amostra de estudo — 2 amostras", tipos.count("sample")),
        ("amostra de validação — 1 amostra", tipos.count("validate")),
    ):
        assert rotulo in por_rotulo, sorted(por_rotulo)
        assert por_rotulo[rotulo].x.size == quantas


def test_u8_um_tipo_de_amostra_INVALIDO_no_fio_retira_a_figura(qtbot: QtBot) -> None:  # noqa: N802
    """A fronteira valida o `Literal` em runtime, e a tela **diz** que recusou.

    **O modo de falha que a validação impede é o insidioso:** um quarto valor cairia fora das
    três séries, a amostra desapareceria da figura, e a legenda contaria menos do que a matriz tem
    — sem nada levantar.

    > **E ESTE TESTE ACHOU UM SEGUNDO DEFEITO, pré-existente.** A primeira versão dele afirmava que
    > a excepção **sobe**, e ela subia: `_troca_a_figura` apanhava só `MatrizVaziaError`, e as
    > outras sete recusas tipadas de `plots/` — as da deriva, das distribuições, do volcano e do
    > recorte, todas já ligadas — eram engolidas pelo `stderr` dentro do slot de Qt, com o painel a
    > ficar com a figura anterior. A correção é a base `RecusaDoConstrutorError`: a guarda
    > existia num sítio e faltava na classe inteira.

    **O estado é `DADO_RECUSADO` e não `SEM_DADO`, e a distinção importa:** *"não veio
    população"* e *"veio população que não descreve os mesmos objetos"* têm consertos opostos.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_pca(_com_pca(tipos_de_amostra=["qc", "blank", "sample", "sample", "qc"]))
    graficos.figuras.define_execucao(_resposta(200, _corpo(23)))
    _escolhe(graficos, "figura_dos_scores")

    visivel = graficos.figuras.estado_visivel()
    assert visivel["figura"] == "", "a figura ficou na tela sobre um corpo que o construtor recusou"
    assert visivel["repouso"] == TEXTO_DO_ESTADO[EstadoDaFigura.DADO_RECUSADO]
    assert visivel["repouso"] != TEXTO_DO_ESTADO[EstadoDaFigura.SEM_DADO], (
        "«não veio população» e «veio população desalinhada» colapsaram num texto só"
    )


def test_u8_a_recusa_por_populacao_vazia_continua_com_o_TEXTO_dela(qtbot: QtBot) -> None:  # noqa: N802
    """O outro lado do par: a base nova não engoliu a distinção que já existia.

    Sem esta metade, `except RecusaDoConstrutorError` poderia ter passado a mostrar o mesmo texto
    para os dois fatos e nada acusaria — é a metade que impede a correção de apagar o que ela devia
    preservar, porque modificar um mecanismo invalida a evidência dele.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_pca({"trajetoria_dos_qcs": dict(TRAJETORIA_DE_TESTE)})
    graficos.figuras.define_execucao(_resposta(200, _corpo(25)))
    _escolhe(graficos, "figura_dos_scores")

    visivel = graficos.figuras.estado_visivel()
    assert visivel["figura"] == ""
    assert visivel["repouso"] == TEXTO_DO_ESTADO[EstadoDaFigura.SEM_DADO]


def test_u8_a_recusa_do_construtor_e_uma_BASE_e_as_oito_herdam_dela() -> None:  # noqa: N802
    """**O portão é sobre a CLASSE, e é ele que faz a correção valer para a próxima recusa.**

    Consertar as sete que existiam não fecha nada: a oitava nasce fora da base e volta a ser
    engolida no slot. Este teste enumera todo `ValueError` declarado em `refino.plots.*` e exige
    que herde de `RecusaDoConstrutorError` — recusa nova reprova até ser classificada.
    """
    import inspect
    import pkgutil

    import refino.plots as pacote

    achadas: dict[str, type] = {}
    for info in pkgutil.iter_modules(pacote.__path__):
        modulo = importlib.import_module(f"refino.plots.{info.name}")
        for nome, cls in inspect.getmembers(modulo, inspect.isclass):
            if issubclass(cls, Exception) and cls.__module__.startswith("refino.plots."):
                achadas[nome] = cls

    assert len(achadas) >= 8, f"a extração achou {sorted(achadas)} — o portão seria quase vácuo"
    fora = sorted(
        nome
        for nome, cls in achadas.items()
        if nome != "RecusaDoConstrutorError" and not issubclass(cls, RecusaDoConstrutorError)
    )
    assert not fora, (
        f"recusas de `plots/` fora da base: {fora}. `_troca_a_figura` apanha a base, e o que "
        "ficar de fora é engolido para o `stderr` dentro de um slot de Qt"
    )
    # E a planta: uma recusa nova que nasça de `ValueError` é acusada pelo mesmo predicado.
    plantada = type("RecusaNovaError", (ValueError,), {"__module__": "refino.plots.inventado"})
    assert not issubclass(plantada, RecusaDoConstrutorError)


def test_u8_sem_a_raiz_pca_a_figura_e_RETIRADA_com_texto(qtbot: QtBot) -> None:  # noqa: N802
    """O par causal da ligação: com a raiz desenha, sem ela retira — e não fica em branco."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_pca({"trajetoria_dos_qcs": dict(TRAJETORIA_DE_TESTE)})
    graficos.figuras.define_execucao(_resposta(200, _corpo(24)))
    _escolhe(graficos, "figura_do_scree")
    assert graficos.figuras.estado_visivel()["figura"] == ""

    graficos.recebe_pca(_com_pca())
    _escolhe(graficos, "figura_dos_scores")
    _escolhe(graficos, "figura_do_scree")
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"


@pytest.mark.parametrize("funcao", FIGURAS_DE_PLSDA)
def test_u8_as_tres_de_plsda_DESENHAM_do_corpo_da_execucao(qtbot: QtBot, funcao: str) -> None:  # noqa: N802
    """O par causal da ligação de PLS-DA, uma figura de cada vez."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = _com_plsda(31)
    graficos.recebe_plsda(corpo)
    assert graficos.figuras.define_execucao(_resposta(200, corpo)) is not None
    _escolhe(graficos, funcao)

    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"
    eixo = graficos.figuras.canvas.figure.axes[0]
    assert confere_vista(eixo) == []
    assert series(eixo), f"{funcao} desenhou zero séries"


@pytest.mark.parametrize("funcao", FIGURAS_DE_PLSDA)
def test_u8_a_INVIOLAVEL_7_atravessa_o_fio_ate_a_tela(qtbot: QtBot, funcao: str) -> None:  # noqa: N802
    """**R7 ponta a ponta: o par está no rodapé DESENHADO das três figuras.**

    O oráculo são os números derivados do dicionário do fio por caminho independente do painel — é
    a classe (d) do arnês aplicada à ligação, e é o que separa *"o painel construiu a figura"* de
    *"a figura publica a validação"*.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = _com_plsda(32)
    graficos.recebe_plsda(corpo)
    graficos.figuras.define_execucao(_resposta(200, corpo))
    _escolhe(graficos, funcao)

    escrito = _rodape_desenhado(graficos.figuras)
    titulo = graficos.figuras.canvas.figure.axes[0].get_title()
    for valor in (PLSDA_DE_TESTE["q2"], PLSDA_DE_TESTE["permutation_pvalue"]):
        assert decimal(float(valor)) in escrito, f"{funcao}: o rodapé não traz {valor} — {escrito}"
        assert decimal(float(valor)) in titulo, f"{funcao}: o título não traz {valor} — {titulo!r}"


def test_u8_a_legenda_dos_scores_diz_QUAL_grupo_e_o_positivo(qtbot: QtBot) -> None:  # noqa: N802
    """**O que a ordem do par decide, depois de medida: só o rótulo.**

    Trocar a ordem não muda número nenhum — a troca de sinal é desfeita —, então a única
    coisa que a distingue é a legenda. O par causal é o mesmo desenho com os papéis invertidos.
    """
    vistos = []
    for positivo, negativo in (("HAMTSP", "Asymptomatic"), ("Asymptomatic", "HAMTSP")):
        graficos = PainelDeGraficos()
        qtbot.addWidget(graficos)
        corpo = _com_plsda(33, grupo_positivo=positivo, grupo_negativo=negativo)
        graficos.recebe_plsda(corpo)
        graficos.figuras.define_execucao(_resposta(200, corpo))
        _escolhe(graficos, "figura_dos_scores_do_plsda")
        legenda = graficos.figuras.canvas.figure.axes[0].get_legend()
        assert legenda is not None
        vistos.append(tuple(t.get_text() for t in legenda.get_texts()))

    assert vistos[0] != vistos[1], "trocar o papel não mudou a legenda"
    assert "HAMTSP — 2 amostras (classe positiva, y = 1)" in vistos[0]
    assert "Asymptomatic — 2 amostras (classe positiva, y = 1)" in vistos[1]


def test_os_scores_do_plsda_na_tela_sao_os_de_VALIDACAO_CRUZADA(qtbot: QtBot) -> None:  # noqa: N802
    """O painel desenha `scores_cv_componente_*`, e não `scores_componente_*`.

    O corpo traz os dois pares, com valores diferentes; o oráculo são os vetores do dicionário,
    lidos aqui por caminho independente do painel. **A âncora vem primeiro**: se os dois pares
    fossem iguais, desenhar o de calibração passaria. Com dois pontos por grupo não há região — o
    eixo diz porquê, e é isso que a última asserção lê.
    """
    cv1, cv2 = PLSDA_DE_TESTE["scores_cv_componente_1"], PLSDA_DE_TESTE["scores_cv_componente_2"]
    assert cv1 != PLSDA_DE_TESTE["scores_componente_1"], "âncora: os dois pares coincidem"

    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = _com_plsda(37)
    graficos.recebe_plsda(corpo)
    graficos.figuras.define_execucao(_resposta(200, corpo))
    _escolhe(graficos, "figura_dos_scores_do_plsda")
    eixo = graficos.figuras.canvas.figure.axes[0]

    grupos = PLSDA_DE_TESTE["grupos_por_amostra"]
    esperadas = {}
    for grupo, papel in (
        ("Asymptomatic", "classe de referência, y = 0"),
        ("HAMTSP", "classe positiva, y = 1"),
    ):
        indices = [i for i, g in enumerate(grupos) if g == grupo]
        esperadas[f"{grupo} — 2 amostras ({papel})"] = (
            [cv1[i] for i in indices],
            [cv2[i] for i in indices],
        )
    assert confere_series(eixo, esperadas) == []
    assert "validação cruzada" in eixo.get_title()
    no_eixo = " ".join(" ".join(t.get_text().split()) for t in eixo.texts)
    assert "sem região de HAMTSP: menos de 3 amostras (2)" in no_eixo, no_eixo


def test_sem_os_escores_de_VALIDACAO_CRUZADA_a_tela_RECUSA_e_nao_desenha_os_de_calibracao(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """Um corpo com os escores de calibração e sem os de validação cruzada: recusa, sem
    recuo, e a recusa diz a causa certa.

    Recuar para `scores_componente_*` desenharia a separação que os de calibração dão mesmo sobre
    rótulos sem informação, com uma região por grupo por cima — a figura persuasiva que a validação
    existe para impedir. **A âncora vem primeiro**: os de calibração estão no corpo, então a
    recusa não é a do PLS-DA que não correu — e é por isso que a frase não pode dizer que ele não
    correu: as duas ausências pedem consertos opostos.
    """
    corpo = _com_plsda(38)
    for chave in ("scores_cv_componente_1", "scores_cv_componente_2"):
        del corpo["plsda"][chave]
    assert corpo["plsda"]["scores_componente_1"], "âncora: os de calibração têm de estar no corpo"

    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_plsda(corpo)
    graficos.figuras.define_execucao(_resposta(200, corpo))
    _escolhe(graficos, "figura_dos_scores_do_plsda")

    visivel = graficos.figuras.estado_visivel()
    assert visivel["figura"] == "", "a tela desenhou um par de escores que o corpo não trazia"
    assert visivel["repouso"] == TEXTO_DO_ESTADO[EstadoDaFigura.SEM_DADO]
    with pytest.raises(MatrizVaziaError) as recusa:
        graficos._scores_do_plsda()
    assert "validação cruzada" in str(recusa.value), recusa.value
    assert "não correu" not in str(recusa.value), recusa.value


def test_u8_o_VIP_indefinido_do_fio_e_CONTADO_e_nao_vira_zero(qtbot: QtBot) -> None:  # noqa: N802
    """`null` no `vip` é ausência de medida, não medida zero, e isso atravessa o fio.

    O par causal é sobre a MESMA população: com um `null` o rodapé conta uma indefinida; com `0.0`
    no lugar dele conta nenhuma — e as duas figuras desenham números de pontos diferentes.
    """

    def rodape_e_pontos(vip: list[Any]) -> tuple[str, int]:
        graficos = PainelDeGraficos()
        qtbot.addWidget(graficos)
        corpo = _com_plsda(34, vip=vip)
        graficos.recebe_plsda(corpo)
        graficos.figuras.define_execucao(_resposta(200, corpo))
        _escolhe(graficos, "figura_do_vip")
        eixo = graficos.figuras.canvas.figure.axes[0]
        pontos = [s for s in series(eixo) if "maior VIP" in s.rotulo]
        return _rodape_desenhado(graficos.figuras), int(pontos[0].x.size)

    com_null, n_null = rodape_e_pontos([1.6, 0.8, None])
    com_zero, n_zero = rodape_e_pontos([1.6, 0.8, 0.0])

    assert com_null != com_zero, "os dois rodapés são iguais: a planta não separa nada"
    assert "1 de 3 features NÃO têm VIP calculável" in com_null
    assert "nenhuma feature ficou sem VIP calculável" in com_zero
    assert (n_null, n_zero) == (2, 3)


def test_u8_um_202_APAGA_a_pca_e_o_plsda_da_execucao_anterior(qtbot: QtBot) -> None:  # noqa: N802
    """As duas raízes seguem o regime das irmãs: correm para TODA resposta, não só no `200`.

    Um `recebe_plsda` que só corresse no `200` deixaria os vetores anteriores no atributo, e a
    figura da execução velha atravessaria o progresso — que é o desfecho que a chave existe para
    impedir.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = _com_plsda(35)
    graficos.recebe_plsda(corpo)
    graficos.figuras.define_execucao(_resposta(200, corpo))
    _escolhe(graficos, "figura_dos_scores_do_plsda")
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"

    graficos.recebe_plsda({"estado": "em_curso"})
    graficos.figuras.define_execucao(_resposta(202, {"estado": "em_curso"}))
    assert graficos.figuras.estado_visivel()["figura"] == ""

    graficos.recebe_pca(_com_pca())
    graficos.figuras.define_execucao(_resposta(200, _corpo(36)))
    _escolhe(graficos, "figura_do_scree")
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"
    graficos.recebe_pca({"estado": "em_curso"})
    # O scree é figura do DIAGNÓSTICO, e quem a refaz é a chegada dele — a janela
    # chama `reavalia` logo depois de `recebe_pca` —, e não o `202` da execução, que
    # não reconstrói figura de validade própria. A ordem é a de produção.
    graficos.figuras.reavalia()
    graficos.figuras.define_execucao(_resposta(202, {"estado": "em_curso"}))
    assert graficos.figuras.estado_visivel()["figura"] == ""


def test_u8_a_contagem_do_rodape_do_vip_BATE_com_os_pontos_desenhados(qtbot: QtBot) -> None:  # noqa: N802
    """**O rodapé e o desenho saem do MESMO número, e o painel não os deixa divergir.**

    `recorte_do_vip` recebe `n_desenhadas` de quem chama, então nada no construtor impede um rodapé
    que diga 20 sobre uma figura de 12 — aconteceu numa sonda desta unidade. O painel passa
    `FEATURES_DO_VIP` aos dois, e esta é a asserção que fecha a porta.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = _com_plsda(37, vip=[1.6, 0.8, 0.4])
    graficos.recebe_plsda(corpo)
    graficos.figuras.define_execucao(_resposta(200, corpo))
    _escolhe(graficos, "figura_do_vip")

    eixo = graficos.figuras.canvas.figure.axes[0]
    pontos = next(s for s in series(eixo) if "maior VIP" in s.rotulo)
    assert f"as {pontos.x.size} features de MAIOR VIP" in _rodape_desenhado(graficos.figuras)
    assert len(eixo.get_yticklabels()) == pontos.x.size


def test_o_artista_NAO_NOMEADO_cai_no_ROTULO_e_o_nomeado_da_a_FEATURE(qtbot: QtBot) -> None:  # noqa: N802
    """**O par causal do defeito de correção desta passada, sobre a figura REAL de cargas.**

    Antes de 11/09/2026 `desenha` tornava **todo** artista sensível ao cursor e `_identifica`
    resolvia o nome por `self._nomes[posicao]` **sem saber de qual artista o evento veio**. Numa
    nuvem de cargas há três artistas — a nuvem e as duas cruzes da origem —, e um clique na cruz
    devolvia um **nome de feature**: errado, com a figura perfeitamente bem formada, e invisível a
    todas as quatro classes do arnês, porque o dado, os rótulos, os limites e os números declarados
    estão todos certos.

    **As duas pontas correm sobre a mesma figura e o mesmo vetor de nomes**, e a única coisa que
    difere é em qual artista se pica. Sem a ponta da cruz, a ponta da nuvem passaria com o defeito
    intacto — era assim que ele estava a passar.

    E a âncora vem antes: a figura tem de ter mais de um artista, e o artista de
    referência **não** pode ser o nomeado. Numa figura de um artista só as duas pontas seriam a
    mesma, e a planta não separaria nada — que é exactamente por que o volcano nunca mostrou este
    defeito.
    """
    cargas = cargas_da_pca(
        pc1=[-0.02, 0.01, 0.03], pc2=[0.015, -0.01, 0.002], features=list(NOMES[:3])
    )
    figura = figura_das_cargas(
        cargas=cargas,
        variancia_explicada=[0.47, 0.09],
        recorte=Recorte(3, DISPONIVEIS, "as três primeiras", "ingerida", "PCA"),
    )
    eixo = figura.axes[0]
    artistas = [*eixo.lines, *eixo.collections]
    nomeados = [a for a in artistas if a.get_gid() == GID_DAS_FEATURES]
    assert len(artistas) > 1, (
        f"âncora: esta figura tem {len(artistas)} artista(s). Com um só, picar «a nuvem» e picar "
        "«outra coisa» seria o mesmo evento, e o par causal não separaria nada"
    )
    assert len(nomeados) == 1, f"âncora: {len(nomeados)} artista(s) nomeado(s), e a nuvem é um"
    nuvem = nomeados[0]
    cruz = next(a for a in artistas if a is not nuvem)

    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura, nomes=cargas.features)
    painel.canvas.draw()

    # Ponta A — a NUVEM: o nome da feature, que é o que R2 pede.
    painel._identifica(_pick(figura, nuvem, 0))
    da_nuvem = painel.estado_visivel()["anotacao"]
    assert cargas.features[0] in da_nuvem, (
        f"o clique na nuvem não nomeou a feature: {da_nuvem!r}. É a metade que R2 exige"
    )

    # Ponta B — a CRUZ DA ORIGEM: o rótulo dela, e NUNCA um nome de feature.
    painel._identifica(_pick(figura, cruz, 0))
    da_cruz = painel.estado_visivel()["anotacao"]
    assert da_cruz, "o clique na cruz não anotou nada — a ponta B não chegou a exercitar o caminho"
    assert not any(nome in da_cruz for nome in cargas.features), (
        f"o clique na CRUZ DA ORIGEM devolveu um nome de feature: {da_cruz!r}. A cruz não tem "
        "feature nenhuma: é geometria de referência, e o nome que ela recebeu é o de uma feature "
        "que está noutro sítio da figura — errado, e com aparência de certo"
    )
    assert str(cruz.get_label()) in da_cruz, (
        f"o clique na cruz não usou o rótulo dela: {da_cruz!r}. O recuo é o rótulo do artista, que "
        "é verdadeiro por construção"
    )


def test_a_marca_REMOVIDA_faz_a_identificacao_cair_no_rotulo(qtbot: QtBot) -> None:  # noqa: N802
    """A planta: tirada a marca da nuvem, o nome da feature deixa de aparecer.

    **É o negativo do mecanismo e não do defeito**, e ele responde à pergunta sobre a marca
    em si: *já vi este mecanismo produzir os dois desfechos?* Com a marca, feature; sem a marca,
    rótulo. Se `desenha` ignorasse o `gid` e aplicasse os nomes a toda a gente — que era o
    comportamento antigo —, esta planta não mudaria nada e passaria por vacuidade.
    """
    cargas = cargas_da_pca(
        pc1=[-0.02, 0.01, 0.03], pc2=[0.015, -0.01, 0.002], features=list(NOMES[:3])
    )
    figura = figura_das_cargas(
        cargas=cargas,
        variancia_explicada=[0.47, 0.09],
        recorte=Recorte(3, DISPONIVEIS, "as três primeiras", "ingerida", "PCA"),
    )
    eixo = figura.axes[0]
    nuvem = next(a for a in [*eixo.lines, *eixo.collections] if a.get_gid() == GID_DAS_FEATURES)
    nuvem.set_gid(None)

    painel = PainelDeFigura(construtores=())
    qtbot.addWidget(painel)
    painel.desenha(figura, nomes=cargas.features)
    painel.canvas.draw()
    painel._identifica(_pick(figura, nuvem, 0))

    anotacao = painel.estado_visivel()["anotacao"]
    assert anotacao, "a planta matou a identificação inteira em vez de trocar o nome pelo rótulo"
    assert not any(nome in anotacao for nome in cargas.features), (
        f"sem a marca a identificação continuou a nomear a feature: {anotacao!r}. O vetor de nomes "
        "está a ser aplicado a um artista que não se declarou dono dele, que é o defeito"
    )


# ---------------------------------------------------------------------------
# A janela roteia para o painel novo, e a ORDEM viajou
# ---------------------------------------------------------------------------


class _TransporteQueRecusa:
    """Um `Transporte` cujas duas chamadas levantam: estes testes não disparam rota nenhuma.

    Levantar em vez de devolver é deliberado — se algum caminho destes testes falasse com a API, o
    teste reprovaria em vez de passar sobre uma resposta inventada. É o molde do
    `_TransporteMudo` de `tests/test_desktop_recorte.py`, e não herda nada porque `Transporte` é
    `Protocol`.
    """

    def post(self, url: str, **kwargs: object) -> object:
        raise AssertionError(f"estes testes não disparam rota, e alguém chamou POST {url}")

    def get(self, url: str, **kwargs: object) -> object:
        raise AssertionError(f"estes testes não disparam rota, e alguém chamou GET {url}")


def _com_medias(seed: int, medias: list[float]) -> dict[str, Any]:
    """O corpo de `/execucoes` com o resumo dentro, e as médias por feature trocadas.

    É o material do par causal da ordem: duas execuções cujas curvas ocupam faixas **disjuntas**,
    afirmado na própria função de teste antes de qualquer figura ser lida.
    """
    return {
        **_corpo(seed),
        "distribuicoes": {**RESUMO_DE_TESTE, "medias_por_feature_antes": medias},
    }


def _janela_do_pipeline(qtbot: QtBot) -> Any:
    """A janela cujas respostas de `/execucoes` são as da Análise sobre a normalizada.

    **Sem clique nenhum**: estes testes entregam o corpo direto a `_encaminha`, e é a entrada da
    Análise em curso que diz que a execução conclui — e escreve — as figuras das etapas 1 e 2
    (`JanelaPrincipal._etapas_que_a_execucao_escreve`). Até 09/11/2026 era o clique do «Executar
    pipeline», que saiu com a aba «Executar e exportar». O que eles medem é o roteamento e a ordem
    dos `recebe_*`, e não a regra de quem escreve, que tem testes próprios na janela.
    """
    from refino.desktop.janela.principal import JanelaPrincipal

    janela = JanelaPrincipal(_TransporteQueRecusa())
    qtbot.addWidget(janela)
    janela._etapas_que_a_execucao_escreve = lambda: frozenset({1, 2})  # type: ignore[method-assign]
    return janela


def _faixa_do_antes(painel: PainelDeFigura) -> tuple[float, float]:
    """O intervalo em `x` da primeira série do eixo de cima — a curva do «antes»."""
    curva = series(painel.canvas.figure.axes[0])[0]
    return float(curva.x.min()), float(curva.x.max())


def test_u2_a_janela_ROTEIA_a_execucao_para_o_painel_de_graficos(qtbot: QtBot) -> None:  # noqa: N802
    """A resposta de `/execucoes` chega ao painel NOVO e desenha.

    **A âncora de não-vacuidade vem primeiro e na mesma função**: o painel dos gráficos
    desenhou de facto. Sem ela, a asserção de que o painel de qualidade não guarda corpo nenhum
    passaria com a janela inteira morta.
    """
    janela = _janela_do_pipeline(qtbot)

    corpo = _com_distribuicoes(1)
    janela._encaminha("execucao", _resposta(200, corpo))
    _escolhe(janela.graficos, "figura_da_densidade_por_feature")

    assert janela.graficos.figuras.estado_visivel()["figura"] == "2 eixo(s)", (
        "âncora: a execução não chegou a desenhar no painel novo, e as asserções abaixo seriam "
        "sobre uma janela morta"
    )
    assert janela.graficos._distribuicoes == corpo["distribuicoes"]

    # E o painel de qualidade deixou de ser dono deste estado. Duas fontes sobre
    # qual figura está na tela é exactamente o que a outra opção deixaria.
    assert not hasattr(janela.qualidade, "figuras"), (
        "o painel de qualidade continua a hospedar um painel de figura"
    )
    assert not hasattr(janela.qualidade, "_distribuicoes"), (
        "o painel de qualidade continua a guardar o resumo das distribuições — a segunda fonte "
        "que KTD3b existe para não ter"
    )


def test_u2_a_figura_da_SEGUNDA_execucao_mostra_o_corpo_da_SEGUNDA(qtbot: QtBot) -> None:  # noqa: N802
    """**O invariante de ORDEM, e ele é o que a mudança de casa mais facilmente perdia.**

    `JanelaPrincipal._encaminha` chama os cinco `recebe_*` **antes** de
    `figuras.define_execucao`, porque `define_execucao` constrói a figura no mesmo instante em que
    reconhece a execução nova. Guardar depois desenha o corpo da execução **anterior** sob o
    rótulo da corrente — quatro comentários de `principal.py` defendem essa ordem, e a mudança de
    painel moveu as quatro linhas de destino.

    ## O par causal, e a planta cai sobre dado em que as duas coisas diferem

    As duas execuções trazem médias por feature em faixas **disjuntas**, afirmado aqui antes de
    qualquer figura ser lida. Com a ordem certa, a segunda figura cai na faixa da segunda; com a
    ordem invertida — a metade (b), feita no painel e não na janela —, ela cai na da primeira. Sem
    a metade (b) esta asserção passaria com um painel que nunca redesenha.
    """
    primeiras = [11.0, 12.0, 13.0, 14.0]
    segundas = [911.0, 912.0, 913.0, 914.0]
    assert max(primeiras) < min(segundas), (
        f"âncora: as médias {primeiras} e {segundas} têm de ocupar faixas disjuntas, ou a figura "
        "da execução errada cairia no mesmo sítio e o cenário seria vácuo"
    )

    # (a) A ordem REAL, pela janela: duas execuções seguidas, e a segunda é a que se vê.
    janela = _janela_do_pipeline(qtbot)
    _escolhe(janela.graficos, "figura_da_densidade_por_feature")

    janela._encaminha("execucao", _resposta(200, _com_medias(1, primeiras)))
    inicio_da_primeira, fim_da_primeira = _faixa_do_antes(janela.graficos.figuras)
    assert inicio_da_primeira < min(primeiras) and fim_da_primeira > max(primeiras), (
        f"âncora: a primeira execução desenhou em [{inicio_da_primeira}, {fim_da_primeira}] e as "
        f"médias dela são {primeiras} — a comparação de faixas abaixo não mede nada"
    )

    janela._encaminha("execucao", _resposta(200, _com_medias(2, segundas)))
    inicio, fim = _faixa_do_antes(janela.graficos.figuras)
    assert inicio > max(primeiras), (
        f"a figura da SEGUNDA execução desenhou em [{inicio}, {fim}], que é a faixa da PRIMEIRA. "
        "Alguém pôs `figuras.define_execucao` antes dos `recebe_*` em `principal.py`"
    )
    assert fim > max(segundas)

    # (b) A ordem INVERTIDA, no painel: o mesmo material, e o desfecho diverge.
    invertido = PainelDeGraficos()
    qtbot.addWidget(invertido)
    _escolhe(invertido, "figura_da_densidade_por_feature")
    invertido.recebe_distribuicoes(_com_medias(1, primeiras), corresponde=True)
    invertido.figuras.define_execucao(_resposta(200, _com_medias(1, primeiras)))
    invertido.figuras.define_execucao(_resposta(200, _com_medias(2, segundas)))
    invertido.recebe_distribuicoes(_com_medias(2, segundas), corresponde=True)
    inicio_invertido, _fim_invertido = _faixa_do_antes(invertido.figuras)
    assert inicio_invertido < max(primeiras), (
        f"a planta não separou: com os `recebe_*` DEPOIS de `define_execucao` a figura desenhou "
        f"em [{inicio_invertido}, …], que já é a faixa da segunda. Então a asserção (a) passaria "
        "com a ordem trocada, e não mede o invariante"
    )


def _com_qualidade(mediana: float) -> dict[str, Any]:
    """Um corpo de `/execucoes` que enche a aba de qualidade e a linha de antes-e-depois."""
    return {
        **_corpo(1),
        "qualidade": {
            "qc_rsd_median": mediana,
            "n_features_rsd_defined": 2158,
            "rsd_variant": "plain",
            "rsd_estimator": "in_sample",
        },
        "rsd_antes_e_depois": {
            "mediana_antes": mediana,
            "mediana_depois": mediana / 2,
            "correcao": "loess",
            "rsd_variant": "plain",
            "rsd_estimator": "in_sample",
            "estimador_alternativo": "leave_one_qc_out",
            "n_features_comparadas": 400,
            "n_features": 500,
        },
    }


def test_u2_o_erro_da_segunda_execucao_LIMPA_a_aba_de_qualidade(qtbot: QtBot) -> None:  # noqa: N802
    """O que a mudança de painel quebrou ao levar a figura para outra aba, e nada media.

    `figuras.define_execucao` corre para **toda** resposta e retira a figura num `202` ou num erro;
    os três `mostra_*` corriam só sob `codigo == 200`. Enquanto a figura e as métricas partilhavam
    a aba, o rótulo de repouso da figura era a contradição **co-localizada** — quem visse a figura
    sumir sabia que os números ao lado eram da execução anterior. A troca levou a figura para a aba
    «Gráficos», e com ela saiu o único sinal de que a execução seguinte falhou.

    ## O par causal, e ele é entre DUAS janelas que só diferem pela segunda resposta

    A janela (a) recebe `200` e depois `500`; a (b) recebe só o `200`. A asserção é que os textos
    **divergem** — se os `mostra_*` voltassem para dentro do `if`, as duas ficariam byte a byte
    iguais e este cenário passaria a não medir nada.

    **E a âncora vem antes das duas**: o `200` tem de MUDAR a aba contra o painel em repouso, ou a
    igualdade final seria satisfeita por vacuidade — um painel que nunca escreveu nada
    também volta ao repouso.
    """
    from refino.desktop.janela.principal import JanelaPrincipal
    from refino.desktop.janela.qualidade import PainelDeQualidade

    em_repouso = PainelDeQualidade()
    qtbot.addWidget(em_repouso)
    repouso = dict(em_repouso.texto_visivel())

    janela = JanelaPrincipal(_TransporteQueRecusa())
    qtbot.addWidget(janela)
    janela._encaminha("execucao", _resposta(200, _com_qualidade(0.1799)))
    cheia = dict(janela.qualidade.texto_visivel())
    assert cheia != repouso, (
        "âncora: o corpo do `200` não mudou nada na aba de qualidade, então a comparação abaixo "
        "seria satisfeita por um painel que nunca escreveu"
    )

    janela._encaminha("execucao", _resposta(500, {"detail": "estourou"}))
    depois_do_erro = dict(janela.qualidade.texto_visivel())

    # (b) A planta: a MESMA primeira resposta, e nenhuma segunda.
    so_o_200 = JanelaPrincipal(_TransporteQueRecusa())
    qtbot.addWidget(so_o_200)
    so_o_200._encaminha("execucao", _resposta(200, _com_qualidade(0.1799)))
    sem_o_erro = dict(so_o_200.qualidade.texto_visivel())

    assert depois_do_erro != sem_o_erro, (
        "a aba de qualidade continuou a mostrar os números da execução ANTERIOR depois de um "
        "`500`. Alguém pôs os três `mostra_*` de volta dentro do `if resposta.codigo == 200` em "
        "`principal.py` — e a figura, que sai noutra aba, deixou de ser o sinal"
    )
    assert depois_do_erro == repouso, (
        "depois do erro a aba tinha de voltar ao estado de ausência com que nasce, e ficou em "
        f"{ {k: v for k, v in depois_do_erro.items() if repouso.get(k) != v} }"
    )


# ---------------------------------------------------------------------------
# A FIGURA ACOMPANHA O CANVAS
#
# **Os dois defeitos que esta seção guarda são PRÉ-EXISTENTES e foram vistos a olho**, na janela
# montada: a figura ficava no quarto superior esquerdo a partir do segundo desenho, e o
# título das figuras com `suptitle` recebia o texto do rodapé.
#
# **Medido na árvore ANTES do conserto, num `PainelDeFigura` de 1818x260 px de canvas** (população:
# um painel, duas figuras iguais desenhadas em seguida):
#
# | o que | figura | canvas | razão de área |
# |---|---|---|---|
# | um desenho só | 1818x260 | 1818x260 | **1,000** |
# | dois desenhos seguidos | 672x432 | 1818x260 | **0,614** |
#
# **É por isso que o par causal tem de ser DOIS desenhos.** Com um só, o arnês imprime `1,000` na
# árvore quebrada e na consertada — o mesmo veredito para estados opostos, e na forma mais
# convidativa que existe: o desfecho de sucesso. O primeiro desenho sai certo por acidente de ordem,
# porque `desenha` torna o canvas visível e **isso** é um redimensionamento de widget.
# ---------------------------------------------------------------------------


def _pixels_do_canvas(painel: PainelDeFigura) -> tuple[int, int]:
    """O tamanho do canvas em pixels FÍSICOS — a conta do `resizeEvent` do `matplotlib`."""
    proporcao = painel.canvas.device_pixel_ratio
    return (
        int(painel.canvas.width() * proporcao),
        int(painel.canvas.height() * proporcao),
    )


def _pixels_da_figura(figura: Figure) -> tuple[int, int]:
    largura, altura = figura.get_size_inches() * figura.dpi
    return round(float(largura)), round(float(altura))


def _ocupa_o_canvas(painel: PainelDeFigura) -> bool:
    """A figura do canvas tem o tamanho do canvas, com um pixel de folga para o arredondamento."""
    figura = _pixels_da_figura(painel.canvas.figure)
    canvas = _pixels_do_canvas(painel)
    return abs(figura[0] - canvas[0]) <= 1 and abs(figura[1] - canvas[1]) <= 1


def _painel_na_tela(
    qtbot: QtBot,
    largura: int,
    altura: int,
    *,
    construtores: Sequence[ConstrutorDeFigura] = (),
    escala: float | None = None,
) -> PainelDeFigura:
    """Um painel com tamanho de verdade. **Sem `show` o canvas fica nos 640x480 com que nasce.**"""
    argumentos: dict[str, Any] = {"construtores": construtores}
    if construtores:
        argumentos["fonte"] = lambda _c: (figura_de_teste(), NOMES)
    if escala is not None:
        argumentos["escala"] = lambda valor=escala: valor
    painel = PainelDeFigura(**argumentos)
    qtbot.addWidget(painel)
    painel.resize(largura, altura)
    painel.show()
    return painel


def test_par_causal_do_dimensionamento_DOIS_desenhos_seguidos_ocupam_o_canvas(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """O segundo desenho é o que carrega o sinal, e o primeiro é o acidente de ordem.

    A âncora vem primeiro: a figura NASCE com geometria
    diferente da do canvas. Sem essa divergência o cenário passaria por coincidência — uma figura
    que já nascesse do tamanho do widget satisfaria a asserção sem ninguém a dimensionar.
    """
    painel = _painel_na_tela(qtbot, 1200, 700)

    painel.desenha(figura_de_teste(), nomes=NOMES)
    canvas = _pixels_do_canvas(painel)
    assert _pixels_da_figura(nova_figura()) != canvas, (
        f"âncora: uma figura recém-nascida já mede {_pixels_da_figura(nova_figura())}, que é o "
        f"tamanho do canvas {canvas} — o cenário abaixo passaria sem dimensionamento nenhum"
    )

    assert _ocupa_o_canvas(painel), (
        f"PRIMEIRO desenho: figura {_pixels_da_figura(painel.canvas.figure)} num canvas {canvas}"
    )

    painel.desenha(figura_de_teste(), nomes=NOMES)
    assert _ocupa_o_canvas(painel), (
        f"SEGUNDO desenho: figura {_pixels_da_figura(painel.canvas.figure)} num canvas "
        f"{_pixels_do_canvas(painel)}. É este o desfecho que a árvore anterior dava — a figura "
        "ficava nas polegadas de nascença porque trocar a figura não redimensiona o widget"
    )


def test_o_oraculo_da_ocupacao_ACUSA_uma_figura_que_ficou_no_tamanho_DE_NASCENCA(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """Par causal **em processo** do oráculo: a planta repõe as polegadas com que a figura nasceu.

    **Sem esta metade, `_ocupa_o_canvas` podia ser uma asserção que nunca reprova** — e um oráculo
    que devolve sempre `True` dá exactamente o mesmo relatório que o conserto.
    """
    painel = _painel_na_tela(qtbot, 1200, 700)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    assert _ocupa_o_canvas(painel), "âncora: sem a planta a figura ocupa o canvas"

    painel.canvas.figure.set_size_inches(LARGURA_PADRAO, ALTURA_PADRAO, forward=False)

    assert not _ocupa_o_canvas(painel), (
        "o oráculo não acusou uma figura de volta às polegadas de nascença — ele não distingue os "
        "dois estados que esta unidade existe para separar"
    )


def test_trocar_a_figura_PELO_SELETOR_mantem_a_ocupacao(qtbot: QtBot) -> None:  # noqa: N802
    """A outra porta que chega a `desenha` — e é a que o profissional usa para trocar de figura."""
    painel = _painel_na_tela(qtbot, 1100, 640, construtores=_dois_construtores())
    painel.define_execucao(_resposta(200, _corpo(1)))
    assert painel.estado_visivel()["figura"] == "1 eixo(s)", "âncora: nada foi desenhado"
    assert _ocupa_o_canvas(painel), "âncora: o primeiro desenho já não ocupava o canvas"

    painel.seletor.setCurrentIndex(1)

    assert _ocupa_o_canvas(painel), (
        f"depois da troca no seletor a figura mede {_pixels_da_figura(painel.canvas.figure)} num "
        f"canvas {_pixels_do_canvas(painel)}"
    )


def test_redimensionar_DEPOIS_da_troca_continua_a_dar_a_figura_do_tamanho_do_canvas(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """O caminho do `matplotlib` e o nosso não brigam: quem redimensiona o widget continua a mandar.

    **A âncora afirma que o canvas MUDOU de tamanho**, senão o cenário mede o mesmo estado duas
    vezes e passa sem tocar no `resizeEvent`.
    """
    painel = _painel_na_tela(qtbot, 1100, 640)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    antes = _pixels_do_canvas(painel)

    painel.resize(1400, 820)
    qtbot.wait(0)
    depois = _pixels_do_canvas(painel)

    assert depois != antes, (
        f"âncora: o canvas mediu {antes} e mede {depois} — o redimensionamento não aconteceu e o "
        "cenário não exercita o caminho que diz exercitar"
    )
    assert _ocupa_o_canvas(painel), (
        f"depois de redimensionar, a figura mede {_pixels_da_figura(painel.canvas.figure)} num "
        f"canvas {depois}"
    )


def _folga_do_rodape(figura: Figure) -> float:
    """Pixels entre o topo do rodapé do recorte e o fundo do eixo mais baixo. Negativo = colide.

    **O rodapé é lido pelo `gid` e não por posição**, que é a mesma armadilha que
    `tests/unit/test_plots_distribuicao.py::rodape` documenta: `figura.texts[0]` é o `suptitle` nas
    figuras que têm título de figura, e o título seria medido como rodapé.
    """
    desenhador = figura.canvas.get_renderer()  # type: ignore[attr-defined]
    para_figura = figura.transFigure.inverted()
    achados = [t for t in figura.texts if t.get_gid() == GID_DO_RODAPE]
    assert len(achados) == 1, f"âncora: {len(achados)} rodapés marcados, e eu esperava um"
    topo = float(achados[0].get_window_extent(desenhador).transformed(para_figura).y1)
    fundo = min(
        float(eixo.get_tightbbox(desenhador).transformed(para_figura).y0) for eixo in figura.axes
    )
    return (fundo - topo) * float(figura.get_size_inches()[1]) * float(figura.dpi)


def test_ktd6_o_rodape_do_recorte_NAO_colide_com_o_eixo_depois_do_dimensionamento(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """A planta é o mesmo canvas SEM o reajuste — as duas geometrias são calculadas.

    **A faixa do rodapé é reservada como FRAÇÃO da altura e o texto mede-se em pontos**, que são
    absolutos. Dimensionar a figura sem refazer a faixa troca área morta por colisão — é o defeito
    que `plots.recorte.reajusta` mediu em -9,5 px a 1280x860.

    **A planta calcula a folga que a mesma figura teria sem o reajuste** e afirma que as duas
    divergem, antes de comparar — senão o cenário passaria num tamanho onde o reajuste não fosse
    necessário, e não mediria nada.
    """
    painel = _painel_na_tela(qtbot, 900, 340)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    com_reajuste = _folga_do_rodape(painel.canvas.figure)

    # A planta: a MESMA figura, posta no MESMO tamanho, sem ninguém refazer a faixa.
    sem_reajuste = figura_de_teste()
    largura, altura = _pixels_do_canvas(painel)
    sem_reajuste.set_dpi(painel.canvas.figure.dpi)
    sem_reajuste.set_size_inches(
        largura / sem_reajuste.dpi, altura / sem_reajuste.dpi, forward=False
    )
    folga_da_planta = _folga_do_rodape(sem_reajuste)

    assert folga_da_planta < com_reajuste, (
        f"a planta não separa: sem reajuste a folga é {folga_da_planta:.1f} px e com reajuste é "
        f"{com_reajuste:.1f} px. Neste tamanho o reajuste não muda nada, e o cenário é vácuo"
    )
    assert com_reajuste > 0.0, (
        f"o rodapé colide com o eixo por {-com_reajuste:.1f} px depois do dimensionamento"
    )


def test_as_marcas_do_eixo_X_e_o_rotulo_do_eixo_Y_ficam_DENTRO_da_figura(qtbot: QtBot) -> None:  # noqa: N802
    """O corte que apagava o `s` de «scores»: 432 px de figura num widget de 260 são 166%.

    A âncora afirma que a figura de facto tem rótulo nos dois eixos — sem ela a asserção de
    contenção correria sobre caixas vazias e passaria por vacuidade.
    """
    painel = _painel_na_tela(qtbot, 1000, 380)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    figura = painel.canvas.figure
    eixo = figura.axes[0]
    assert eixo.get_xlabel() and eixo.get_ylabel(), (
        "âncora: a figura de teste não tem rótulo nos dois eixos, e a contenção não mede nada"
    )

    desenhador = figura.canvas.get_renderer()  # type: ignore[attr-defined]
    para_figura = figura.transFigure.inverted()
    caixa = eixo.get_tightbbox(desenhador).transformed(para_figura)

    assert caixa.x0 >= 0.0 and caixa.x1 <= 1.0, (
        f"o eixo com as marcas e os rótulos vai de x={caixa.x0:.3f} a x={caixa.x1:.3f} em fração "
        "de figura — o que está fora de [0, 1] nunca chega à tela"
    )
    assert caixa.y0 >= 0.0 and caixa.y1 <= 1.0, (
        f"o eixo com as marcas e os rótulos vai de y={caixa.y0:.3f} a y={caixa.y1:.3f} em fração "
        "de figura — o que está fora de [0, 1] nunca chega à tela"
    )


def _figura_com_suptitle() -> tuple[Figure, str]:
    figura = figura_de_teste()
    titulo = "Distribuição por amostra, antes e depois"
    figura.suptitle(titulo)
    return figura, titulo


def test_o_TITULO_de_figura_sobrevive_ao_reajuste_do_rodape(qtbot: QtBot) -> None:  # noqa: N802
    """O segundo defeito desta seção, e ele é o que se via no topo da densidade por feature.

    `_reajusta_o_recorte` percorria `figura.texts` e reajustava **cada** `Text`; o `matplotlib` põe
    o `_suptitle` dentro dessa lista, então o título recebia o texto do rodapé. Quatro figuras do
    registo têm `suptitle` — as duas de caixas e as duas de densidade.

    **Duas âncoras, e as duas antes da asserção:** a figura TEM título de figura, e o texto
    dele DIFERE do do rodapé — sem a segunda, «o título continua o mesmo» passaria sobre dois textos
    iguais e não separaria nada.
    """
    painel = _painel_na_tela(qtbot, 900, 340)
    figura, titulo = _figura_com_suptitle()
    painel.desenha(figura, nomes=NOMES)

    desenhada = painel.canvas.figure
    suptitle = desenhada._suptitle
    assert suptitle is not None, "âncora: a figura de teste não tem `suptitle`"
    rodape = next(t for t in desenhada.texts if t.get_gid() == GID_DO_RODAPE)
    assert titulo != rodape.get_text(), (
        "âncora: o título e o rodapé têm o mesmo texto, e a asserção abaixo não distinguiria "
        "sobrescrever de preservar"
    )

    assert suptitle.get_text() == titulo, (
        f"o título da figura virou {suptitle.get_text()!r}. O reajuste do rodapé caiu sobre o "
        "`_suptitle`, que o `matplotlib` guarda dentro de `figura.texts`"
    )


def test_o_cenario_do_TITULO_ACUSA_quando_o_filtro_ADMITE_o_titulo(qtbot: QtBot) -> None:  # noqa: N802
    """Par causal **em processo**: a planta ALARGA o filtro, marcando o título como se fosse rodapé.

    **A primeira planta que escrevi foi REMOVER a marca do rodapé, e ela não separa** — apanhada na
    primeira execução. Sem marca nenhuma o laço não reajusta **nada**, e o título sobrevive pelo
    motivo errado: o cenário irmão passaria na mesma, e o negativo estaria a afirmar o contrário do
    que mede. É a planta que não separa, na forma mais convidativa — o desfecho de sucesso.

    A planta certa é a que reproduz o laço antigo: ele admitia **todo** `Text` de nível de figura,
    e admitir o título é exactamente o que fazia o defeito. Se o corpo do laço deixasse de
    sobrescrever, esta asserção reprovaria — e é isso que a torna o negativo do filtro.
    """
    painel = _painel_na_tela(qtbot, 900, 340)
    figura, titulo = _figura_com_suptitle()
    suptitle_nascido = figura._suptitle
    assert suptitle_nascido is not None, "âncora: a figura de teste não tem `suptitle`"
    suptitle_nascido.set_gid(GID_DO_RODAPE)
    marcados = [t.get_text() for t in figura.texts if t.get_gid() == GID_DO_RODAPE]
    assert titulo in marcados, (
        f"a planta não aterrou: os textos marcados como rodapé são {marcados} e o título não está "
        "entre eles — o laço não chegaria a ele e este negativo não mediria o filtro"
    )

    painel.desenha(figura, nomes=NOMES)

    suptitle = painel.canvas.figure._suptitle
    assert suptitle is not None and suptitle.get_text() != titulo, (
        "com o título ADMITIDO pelo filtro ele sobreviveu na mesma — então o cenário irmão não "
        "está a medir o filtro, e passaria com o laço de volta ao estado que reajustava tudo"
    )


def test_ktd5_a_figura_de_RECUSA_TIPADA_nao_passa_pelo_dimensionamento(qtbot: QtBot) -> None:  # noqa: N802
    """A guarda: dimensiona-se só figura COM eixos, e a recusa põe uma sem nenhum.

    **A âncora afirma que a figura da recusa de facto chega a `_poe_no_canvas`** — sem ela o cenário
    descreveria um caminho que ninguém percorre, e passaria com a guarda apagada.

    **A planta cai sobre dado em que as duas coisas diferem:** a segunda âncora afirma que o tamanho
    de nascença de uma `Figure()` DIVERGE do tamanho do canvas, senão «continua no tamanho de
    nascença» e «foi dimensionada» seriam a mesma medida.
    """
    recusa = ConstrutorDeFigura(
        modulo="arnes", funcao="figura_de_arnes_2", vinculo=Vinculo.RELATORIO, chave="item-1"
    )

    def fonte(construtor: ConstrutorDeFigura) -> tuple[Figure, tuple[str, ...]]:
        if construtor.funcao == recusa.funcao:
            raise MatrizVaziaError("o construtor recebeu população vazia")
        return figura_de_teste(), NOMES

    painel = PainelDeFigura(construtores=(_um_construtor(), recusa), fonte=fonte)
    qtbot.addWidget(painel)
    painel.resize(1100, 640)
    painel.show()

    vistas: list[int] = []
    original = painel._poe_no_canvas

    def espia(figura: Figure) -> None:
        vistas.append(len(figura.axes))
        original(figura)

    painel._poe_no_canvas = espia  # type: ignore[method-assign]

    painel.define_execucao(_resposta(200, _corpo(1)))
    assert painel.estado_visivel()["figura"] == "1 eixo(s)", "âncora: nada foi desenhado"

    painel.seletor.setCurrentIndex(1)

    assert 0 in vistas, (
        f"âncora: `_poe_no_canvas` foi chamado com {vistas} eixo(s) e nunca com zero — a figura da "
        "recusa não chega ao ponto de troca, e a guarda de KTD5 não é exercitada por este cenário"
    )
    assert painel.estado_visivel()["repouso"] == texto_do_estado(EstadoDaFigura.SEM_DADO)

    nascenca = _pixels_da_figura(Figure())
    assert nascenca != _pixels_do_canvas(painel), (
        f"âncora: uma `Figure()` nasce com {nascenca}, que é o tamanho do canvas "
        f"{_pixels_do_canvas(painel)} — as duas metades desta asserção não se distinguem"
    )
    assert _pixels_da_figura(painel.canvas.figure) == nascenca, (
        f"a figura vazia da recusa foi dimensionada para "
        f"{_pixels_da_figura(painel.canvas.figure)}: a guarda `if figura.axes` não está no caminho"
    )


def test_a_ocupacao_e_o_rodape_VALEM_A_144_pontos_por_polegada(qtbot: QtBot) -> None:  # noqa: N802
    """96 é o valor padrão, e é o que executa menos deste caminho.

    A 150% o Windows devolve `logicalDotsPerInch() == 144`, e é o monitor em que a janela vai ser
    demonstrada. A âncora afirma que a figura desenhou de facto nos 144 pedidos — sem ela o cenário
    seria uma segunda cópia do de 96.
    """
    painel = _painel_na_tela(qtbot, 1100, 640, escala=144.0)
    painel.desenha(figura_de_teste(), nomes=NOMES)
    painel.desenha(figura_de_teste(), nomes=NOMES)

    assert painel.canvas.figure.dpi == 144.0, (
        f"âncora: pedi 144 pontos por polegada e a figura desenhou a {painel.canvas.figure.dpi}"
    )
    assert _ocupa_o_canvas(painel), (
        f"a 144 a figura mede {_pixels_da_figura(painel.canvas.figure)} num canvas "
        f"{_pixels_do_canvas(painel)}"
    )
    folga = _folga_do_rodape(painel.canvas.figure)
    assert folga > 0.0, f"a 144 o rodapé colide com o eixo por {-folga:.1f} px"


# ---------------------------------------------------------------------------
# A duplicação do rótulo, FECHADA — o portão que a guardava saiu com ela
#
# O cenário óbvio — «as duas pontas devolvem o mesmo rótulo» — foi REJEITADO: depois da delegação
# qualquer mutação move as duas por igual, a igualdade continua a valer, e ele não acusa nada. A
# única mutação que separaria é repor a segunda tabela, e é essa a propriedade que se guarda aqui.
# ---------------------------------------------------------------------------

_FONTE_DAS_FIGURAS: Final = Path(modulo_das_figuras.__file__)


def _corpo_de_nome_legivel(fonte: str) -> ast.FunctionDef:
    achadas = [
        no
        for no in ast.parse(fonte).body
        if isinstance(no, ast.FunctionDef) and no.name == "_nome_legivel"
    ]
    assert len(achadas) == 1, (
        f"âncora: achei {len(achadas)} definições de `_nome_legivel` e esperava uma. Contar em vez "
        "de indexar: o `index` não levanta sobre a instância errada"
    )
    return achadas[0]


def _reimplementa_a_regra(fonte: str) -> bool:
    """A janela tem uma SEGUNDA implementação da regra do rótulo, e não uma delegação?

    **A docstring sai antes de contar**, e o defeito de a não tirar foi cometido e apanhado pela
    execução: uma delegação de uma linha tem `len(body) == 2`, e o portão acusava a árvore limpa.
    """
    corpo = _corpo_de_nome_legivel(fonte)
    instrucoes = [
        no
        for no in corpo.body
        if not (isinstance(no, ast.Expr) and isinstance(no.value, ast.Constant))
    ]
    if len(instrucoes) != 1 or not isinstance(instrucoes[0], ast.Return):
        return True
    chamada = instrucoes[0].value
    return not (
        isinstance(chamada, ast.Call)
        and isinstance(chamada.func, ast.Name)
        and chamada.func.id == "rotulo_do_construtor"
    )


def test_a_janela_DELEGA_o_rotulo_ao_catalogo_e_nao_o_reimplementa() -> None:  # noqa: N802
    """A irmã fechada: uma regra, uma implementação, e a tupla de artigos num sítio só.

    **Lido por AST do arquivo desta árvore**, e não do módulo que o `sys.path` calhou resolver — é
    a forma de `construtores_descobertos`.
    """
    fonte = _FONTE_DAS_FIGURAS.read_text(encoding="utf-8")
    assert not _reimplementa_a_regra(fonte), (
        "`desktop.janela.figuras._nome_legivel` voltou a implementar a regra do rótulo em vez de "
        "delegar a `plots.catalogo.rotulo_do_construtor`. Duas cópias concordam por coincidência "
        "de texto, e um artigo acrescentado de um lado não chega ao outro"
    )
    # **A âncora de não-vacuidade desta asserção de ausência é a linha ACIMA**: sobre uma
    # fonte vazia `"X" not in fonte` passa, e `_reimplementa_a_regra` levanta antes de chegar aqui
    # — ela exige exactamente uma `_nome_legivel` no que leu. Não separe as duas.
    assert "ARTIGOS_DO_CONSTRUTOR" not in fonte, (
        "a tupla de artigos voltou para o módulo da janela — a constante é do catálogo, que é "
        "quem o portão da unicidade lê"
    )


def test_o_portao_da_DELEGACAO_acusa_uma_segunda_implementacao_plantada() -> None:  # noqa: N802
    """Par causal **em processo**: a planta é a fonte que o extractor recebe.

    Repõe o corpo apagado — `removeprefix` mais o laço dos artigos — e exige que o portão o
    veja. Sem esta metade, um extractor que devolvesse sempre `False` daria um portão verde e mudo.
    """
    fonte = _FONTE_DAS_FIGURAS.read_text(encoding="utf-8")
    mutilada = fonte.replace(
        "    return rotulo_do_construtor(funcao)",
        '    resto = funcao.removeprefix("figura_")\n'
        '    for artigo in ("do_", "da_", "de_", "dos_", "das_"):\n'
        "        if resto.startswith(artigo):\n"
        "            resto = resto.removeprefix(artigo)\n"
        "            break\n"
        '    return resto.replace("_", " ")',
        1,
    )
    assert mutilada != fonte, (
        "a planta não aterrou: a linha da delegação mudou de forma, e este negativo passaria a "
        "correr sobre a fonte intacta"
    )

    assert _reimplementa_a_regra(mutilada), (
        "o portão não viu uma segunda implementação plantada dentro de `_nome_legivel` — ele não "
        "distingue delegar de reimplementar"
    )


# ---------------------------------------------------------------------------
# A FIGURA SAI PARA ARQUIVO, em PNG a 300 ppp
# ---------------------------------------------------------------------------
#
# **O que este bloco prova, e o que ele NÃO prova.** Ele prova o caminho de gravação: o que entra
# no arquivo, o que acontece em cada uma das cinco saídas, e que o controle só existe habilitado
# quando há figura. **Não** prova nada sobre o `savefig` do executável congelado — nenhuma prova
# daquele artefato passa por aqui, porque a sonda mede `buffer_rgba()`, que não atravessa o
# escritor de PNG. A lacuna está declarada no plano, com o mecanismo nomeado.
#
# **O escolhedor é INJETADO em todos os cenários**, e nunca se constrói um `QFileDialog` de
# verdade: o diálogo real é modal e o arnês que o abrisse penduraria a suíte inteira — foi medido
# no caminho irmão, com catorze testes a bloquear e `EXIT=124`.

# A escala da tela nos cenários de gravação. **Fixada e não lida da máquina**, porque o que os
# cenários afirmam é uma RAZÃO entre o arquivo e a tela, e uma tela que devolvesse 300 tornaria a
# âncora «o arquivo é maior que a tela» falsa sem nada estar errado.
ESCALA_DE_TESTE: Final = 96.0

# O tamanho do painel nos cenários de gravação. **Pequeno de propósito:** a 300 ppp cada polegada
# leva mais de três vezes os pixels da tela, e um painel do tamanho dos outros cenários deste
# arquivo produziria PNGs de vários megabytes em cada um dos treze testes abaixo.
LARGURA_DO_PAINEL: Final = 700
ALTURA_DO_PAINEL: Final = 520


def _pixels_do_png(caminho: Path) -> tuple[int, int]:
    """`(largura, altura)` lidas do cabeçalho IHDR — sem carregar a imagem para a memória.

    **Ler o cabeçalho e não decodificar**, porque decodificar um PNG de 2000x1500 em RGBA de vírgula
    flutuante custa dezenas de megabytes por cenário, e o que estes cenários perguntam é o TAMANHO.
    Quem decodifica é só o cenário da cor, e o painel dele é o mesmo tamanho dos outros.

    **A âncora da assinatura vem antes**, e não é zelo: um arquivo que não fosse PNG daria dois
    inteiros plausíveis lidos de bytes quaisquer, e a asserção seguinte compararia ruído.
    """
    dados = caminho.read_bytes()
    assert dados[:8] == b"\x89PNG\r\n\x1a\n", (
        f"âncora: `{caminho.name}` não começa pela assinatura de PNG, e os quatro bytes lidos "
        "abaixo não seriam dimensão nenhuma"
    )
    return int.from_bytes(dados[16:20], "big"), int.from_bytes(dados[20:24], "big")


def _pixels_da_construcao(painel: Any, dpi: float) -> tuple[int, int]:
    """O tamanho de construção, em mm, vezes aqueles pontos por polegada — o padrão."""
    largura_mm, altura_mm = painel.tamanho_de_construcao_mm
    return int(largura_mm / 25.4 * dpi), int(altura_mm / 25.4 * dpi)


def _pixels_esperados(figura: Figure, dpi: float) -> tuple[int, int]:
    """As polegadas da figura vezes aqueles pontos por polegada, truncadas como o Agg trunca.

    `RendererAgg` recebe `int(largura)` e `int(altura)` do `figure.bbox`, então a conta é de
    truncamento e não de arredondamento. As asserções toleram um pixel na mesma, porque o que elas
    afirmam é a razão e não o último bit.
    """
    largura, altura = figura.get_size_inches() * dpi
    return int(largura), int(altura)


def _escolhedor(destino: Path | None) -> tuple[Any, list[str]]:
    """Um escolhedor de caminho que devolve sempre `destino`, e a lista do que lhe foi pedido.

    A lista é o que separa *"o diálogo abriu e o profissional cancelou"* de *"o diálogo nem chegou
    a abrir"* — dois desfechos que não escrevem arquivo nenhum e que uma asserção sobre o disco não
    distingue.
    """
    pedidos: list[str] = []

    def escolhe(nome_sugerido: str) -> Path | None:
        pedidos.append(nome_sugerido)
        return destino

    return escolhe, pedidos


def _painel_para_exportar(
    qtbot: QtBot, destino: Path | None, *, escala: float = ESCALA_DE_TESTE
) -> tuple[PainelDeFigura, list[str]]:
    """Um painel com figura desenhada, tamanho de verdade e o escolhedor remendado."""
    escolhe, pedidos = _escolhedor(destino)
    painel = PainelDeFigura(
        construtores=(_um_construtor(),),
        fonte=lambda _c: (figura_de_teste(), NOMES),
        escala=lambda valor=escala: valor,  # type: ignore[misc]
        escolhedor=escolhe,
    )
    qtbot.addWidget(painel)
    painel.resize(LARGURA_DO_PAINEL, ALTURA_DO_PAINEL)
    painel.show()
    painel.desenha(figura_de_teste(), nomes=NOMES)
    return painel, pedidos


def test_d657_o_arquivo_tem_o_tamanho_de_CONSTRUCAO_e_nao_o_da_janela(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """O arquivo sai no tamanho com que a figura foi construída, e não no da janela.

    Antes era o contrário: só o `dpi` mudava, e o enquadramento era o da tela. O
    padrão passou a ser o tamanho da construção, que não muda
    com o tamanho da janela.

    **A âncora separa as duas quantidades antes de afirmar**: se a janela tivesse o tamanho
    da construção, a asserção passaria sem ninguém ter mudado nada.
    """
    destino = tmp_path / "figura.png"
    painel, pedidos = _painel_para_exportar(qtbot, destino)

    da_janela = _pixels_esperados(painel.canvas.figure, DPI_DA_EXPORTACAO)
    esperado = _pixels_da_construcao(painel, DPI_DA_EXPORTACAO)
    assert da_janela != esperado, (
        f"âncora: a janela e a construção dão os mesmos {esperado} px, e este cenário passaria "
        "por coincidência"
    )

    desfecho, frase = painel.exporta()

    assert desfecho is DesfechoDaExportacao.GRAVADA, frase
    assert pedidos == [painel.nome_sugerido()], (
        f"o escolhedor recebeu {pedidos} e o nome sugerido é {painel.nome_sugerido()!r}"
    )
    assert str(destino) in frase, f"a frase do sucesso não nomeia o caminho: {frase!r}"
    largura, altura = _pixels_do_png(destino)
    assert abs(largura - esperado[0]) <= 1 and abs(altura - esperado[1]) <= 1, (
        f"o PNG mede {(largura, altura)} e a construção a {DPI_DA_EXPORTACAO} ppp dá {esperado}"
    )


def test_u4_o_dpi_da_figura_VOLTA_ao_da_tela_e_a_figura_na_tela_nao_muda(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Gravar não pode deixar a figura da tela em 300 ppp — ela desenharia três vezes maior.

    **Quem repõe é o `matplotlib`**, que troca o `dpi` num gestor de contexto dentro de
    `print_figure`. O cenário existe na mesma: a propriedade é nossa, e quem a garante hoje é uma
    dependência pinada — se ela mudar, é aqui que se sabe.
    """
    painel, _pedidos = _painel_para_exportar(qtbot, tmp_path / "figura.png")

    antes = painel.canvas.figure.dpi
    polegadas = tuple(float(v) for v in painel.canvas.figure.get_size_inches())
    assert antes == ESCALA_DE_TESTE and antes != DPI_DA_EXPORTACAO, (
        f"âncora: a figura desenhou a {antes} ppp, e o cenário precisa de que ela difira de "
        f"{DPI_DA_EXPORTACAO} para poder separar alguma coisa"
    )

    assert painel.exporta()[0] is DesfechoDaExportacao.GRAVADA

    assert painel.canvas.figure.dpi == antes, (
        f"a figura da tela ficou a {painel.canvas.figure.dpi} ppp depois de gravar"
    )
    assert tuple(float(v) for v in painel.canvas.figure.get_size_inches()) == polegadas, (
        "as polegadas da figura mudaram com a gravação — KTD7 pede que só o `dpi` mude"
    )


def test_u4_o_dialogo_CANCELADO_nao_escreve_nada_e_nao_mexe_no_painel(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A segunda saída. **Cancelar não é erro**, e o painel fica exactamente como estava.

    **A âncora de não-vacuidade está na primeira metade:** *"nada foi escrito"*
    passa num painel que nunca escreve. Por isso o mesmo painel grava de facto antes de o
    escolhedor passar a cancelar — e a pasta é esvaziada entre as duas metades.
    """
    valido = tmp_path / "boa.png"
    painel, pedidos = _painel_para_exportar(qtbot, valido)
    assert painel.exporta()[0] is DesfechoDaExportacao.GRAVADA, (
        "âncora: este painel não grava nem no caminho feliz, e a ausência abaixo seria vácua"
    )
    assert valido.exists()
    valido.unlink()
    assert pedidos == [painel.nome_sugerido()], "âncora: o escolhedor do sucesso não foi chamado"

    cancela, pedidos_do_cancelamento = _escolhedor(None)
    painel._escolhedor = cancela  # type: ignore[assignment]
    antes = painel.estado_visivel()
    assert antes["figura"] == "1 eixo(s)", f"âncora: nada foi desenhado — {antes}"

    desfecho, frase = painel.exporta()

    assert desfecho is DesfechoDaExportacao.CANCELADA, frase
    assert pedidos_do_cancelamento == [painel.nome_sugerido()], (
        "o escolhedor não chegou a ser chamado, então este cenário está a medir o ramo do estado "
        "e não o do cancelamento"
    )
    assert list(tmp_path.iterdir()) == [], f"algo foi escrito num cancelamento: {list(tmp_path)}"
    assert painel.estado_visivel() == antes, "o cancelamento mexeu no que o painel mostra"


def test_u4_pasta_que_nao_existe_vira_TEXTO_e_a_excecao_NAO_sobe_do_slot(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A quarta saída: o `OSError` é apanhado, e a frase nomeia o caminho e a causa.

    **Levantar aqui seria o silêncio que este painel já pagou duas vezes:** uma excepção dentro de
    um slot do Qt é engolida para a saída de erro, e o painel para de actualizar sem dizer nada.
    """
    destino = tmp_path / "pasta-que-nao-existe" / "figura.png"
    painel, pedidos = _painel_para_exportar(qtbot, destino)
    assert not destino.parent.exists(), "âncora: a pasta existe, e a escrita não falharia"

    desfecho, frase = painel.exporta()

    assert desfecho is DesfechoDaExportacao.ESCRITA_FALHOU, frase
    assert pedidos == [painel.nome_sugerido()]
    assert str(destino) in frase, f"a frase não nomeia o caminho recusado: {frase!r}"
    assert "No such file" in frase or "não" in frase, (
        f"a frase não traz causa nenhuma junto do caminho: {frase!r}"
    )
    assert not destino.exists()
    assert painel.estado_visivel()["figura"] == "1 eixo(s)", (
        "o painel deixou de mostrar a figura depois de uma escrita recusada"
    )


def test_u4_o_sufixo_que_o_programa_nao_sabe_escrever_e_recusado_ANTES_de_gravar(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A terceira saída, e a ordem é o conteúdo: a guarda vem antes de qualquer escrita.

    **A âncora de não-vacuidade está na primeira metade**: o mesmo painel, com um destino
    `.png`, escreve de facto. Sem ela, *"nada foi escrito"* passaria num painel que nunca escreve.
    """
    valido = tmp_path / "boa.png"
    painel, _pedidos = _painel_para_exportar(qtbot, valido)
    assert painel.exporta()[0] is DesfechoDaExportacao.GRAVADA, (
        "âncora: este painel não grava nem no caminho feliz, e as asserções de ausência abaixo "
        "passariam por vacuidade"
    )
    valido.unlink()

    # `.pdf` passou a ser gravado, e a lista recusada é a dos formatos que continuam
    # fora: o SVG, o JPEG e o nome sem sufixo.
    for nome in ("figura.jpg", "figura.SVG", "figura"):
        alvo = tmp_path / nome
        painel._escolhedor = _escolhedor(alvo)[0]  # type: ignore[assignment]
        desfecho, frase = painel.exporta()
        assert desfecho is DesfechoDaExportacao.SUFIXO_RECUSADO, (nome, frase)
        assert all(sufixo in frase for sufixo in FORMATOS_DA_EXPORTACAO), (
            f"a frase não diz o que o programa sabe gravar: {frase!r}"
        )
        escritos = sorted(p.name for p in tmp_path.iterdir())
        assert escritos == [], f"a recusa de sufixo escreveu alguma coisa: {escritos}"

    # E a direção que a lista de sufixos NÃO pode perder: `.PNG` em maiúsculas é o mesmo formato.
    maiusculo = tmp_path / "figura.PNG"
    painel._escolhedor = _escolhedor(maiusculo)[0]  # type: ignore[assignment]
    assert painel.exporta()[0] is DesfechoDaExportacao.GRAVADA, (
        "`.PNG` foi recusado — a comparação do sufixo não está a ignorar a caixa"
    )


@pytest.mark.parametrize("estado", list(EstadoDaFigura), ids=lambda e: str(e.value))
def test_u4_exporta_habilitado_SO_no_estado_desenhado(estado: EstadoDaFigura) -> None:  # noqa: N802
    """O predicado é PURO, e é ele que o cenário percorre — os treze estados.

    **Parametrizado sobre o enum e nunca sobre uma lista escrita à mão**, para que um estado novo
    entre sozinho. A pergunta é *quantos desfechos este mecanismo pode reportar, e eu vi
    todos?* — e a resposta aqui é treze, porque `EstadoDaFigura` tem treze membros.

    **A direção que importa é a do estado NOVO.** Um predicado escrito como `not in {…}` passaria
    neste cenário hoje e daria `True` ao próximo estado de retirada que alguém acrescentasse: o
    botão nasceria habilitado sobre um canvas sem figura, e clicar gravaria a `Figure()` vazia.
    """
    assert len(EstadoDaFigura) == 13, (
        f"âncora: `EstadoDaFigura` tem {len(EstadoDaFigura)} membros e este cenário declara "
        "treze — a parametrização deixou de cobrir o que a docstring diz que cobre"
    )
    esperado = estado is EstadoDaFigura.DESENHADA
    assert exporta_habilitado(estado) is esperado, (
        f"`{estado.value}`: o predicado devolveu {exporta_habilitado(estado)} e o único estado em "
        "que há figura para gravar é `desenhada`"
    )


def test_u4_o_botao_SEGUE_o_estado_e_no_estado_indisponivel_clicar_nao_escreve(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """O sinal, o botão e a guarda — e a âncora de não-vacuidade vem PRIMEIRO.

    **A asserção de ausência é «clicar não escreveu nada», e ela é satisfeita por vacuidade** num
    escritor que nunca escreve. Por isso a primeira metade vê o mesmo escolhedor ser chamado
    **com sucesso** no estado desenhado, na mesma função, antes de o painel ir para a retirada.
    """
    destino = tmp_path / "figura.png"
    escolhe, pedidos = _escolhedor(destino)
    graficos = PainelDeGraficos(escolhedor=escolhe)
    qtbot.addWidget(graficos)
    graficos.resize(LARGURA_DO_PAINEL, ALTURA_DO_PAINEL)
    graficos.show()

    assert not graficos.botao_de_exportar.isEnabled(), (
        "o botão nasceu habilitado, e no repouso não há figura nenhuma para gravar"
    )
    assert graficos.exportacao.text() == SEM_EXPORTACAO

    # (a) A ÂNCORA: com figura na tela, o escolhedor é chamado e o arquivo aparece.
    graficos.recebe_diagnostico({"trajetoria_dos_qcs": dict(TRAJETORIA_DE_TESTE)})
    assert graficos.figuras.define_execucao(_resposta(200, _corpo(1))) == texto_do_estado(
        EstadoDaFigura.DESENHADA
    )
    assert graficos.botao_de_exportar.isEnabled(), "o botão não seguiu o sinal até o desenho"
    qtbot.mouseClick(graficos.botao_de_exportar, Qt.MouseButton.LeftButton)
    assert pedidos, "âncora: o escolhedor nunca foi chamado, e as ausências abaixo são vácuas"
    assert destino.exists(), "âncora: o caminho feliz não escreveu arquivo nenhum"
    assert str(destino) in graficos.exportacao.text(), graficos.exportacao.text()

    # (b) A ausência: trocar de matriz retira a figura do diagnóstico, o botão morre, e clicar não
    # escreve. Antes era um `202` — a figura do diagnóstico deixou de sair com a execução, e o
    # que a retira agora é a matriz deixar de ser a do diagnóstico.
    destino.unlink()
    pedidos.clear()
    graficos.define_matriz("outra matriz")
    assert graficos.figuras.reavalia() == texto_do_estado(EstadoDaFigura.SEM_DIAGNOSTICO)
    assert not graficos.botao_de_exportar.isEnabled(), (
        "o botão continua habilitado com a figura retirada — clicar gravaria a figura vazia"
    )
    qtbot.mouseClick(graficos.botao_de_exportar, Qt.MouseButton.LeftButton)
    assert pedidos == [], f"o clique num botão desabilitado chegou ao escolhedor: {pedidos}"
    assert list(tmp_path.iterdir()) == [], "o clique no estado indisponível escreveu alguma coisa"

    # E a chamada direta — o caminho que um botão desabilitado não fecha — recusa com texto.
    desfecho, frase = graficos.figuras.exporta()
    assert desfecho is DesfechoDaExportacao.SEM_FIGURA, frase
    assert pedidos == [], "o ramo do estado abriu o diálogo antes de recusar"


def test_u4_o_nome_sugerido_separa_as_FIGURAS_e_separa_as_EXECUCOES(qtbot: QtBot) -> None:  # noqa: N802
    """Sem a identidade da execução no nome, treze figuras em duas corridas colidem mudas.

    **As duas metades são necessárias e nenhuma basta.** Um nome só com a chave daria treze
    arquivos iguais dentro da mesma execução; um nome só com o rótulo daria a segunda corrida por
    cima da primeira. É o mesmo defeito já medido no caminho dos artefatos.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_diagnostico({"trajetoria_dos_qcs": dict(TRAJETORIA_DE_TESTE)})
    graficos.figuras.define_execucao(_resposta(200, _corpo(1)))

    _escolhe(graficos, "figura_da_deriva_do_qc")
    primeira = graficos.figuras.nome_sugerido()
    _escolhe(graficos, "figura_do_scree")
    segunda = graficos.figuras.nome_sugerido()
    assert primeira != segunda, (
        f"duas figuras da MESMA execução sugerem o mesmo nome: {primeira!r} — a segunda gravação "
        "escreveria por cima da primeira"
    )

    graficos.figuras.define_execucao(_resposta(200, _corpo(2)))
    outra_execucao = graficos.figuras.nome_sugerido()
    assert outra_execucao != segunda, (
        f"a MESMA figura em duas execuções sugere o mesmo nome: {segunda!r}"
    )
    assert all(n.endswith(SUFIXO_DA_EXPORTACAO) for n in (primeira, segunda, outra_execucao))
    assert " " not in outra_execucao, f"o nome sugerido tem espaço: {outra_execucao!r}"


def test_u4_par_causal_o_arquivo_fica_BRANCO_com_o_rcParams_global_ESCURO(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O fundo do arquivo vem da FIGURA, e não do `rcParams` global.

    **O par causal precisa de dado em que as duas coisas DIFIRAM**, e é aqui que o cenário
    do plano precisou de ser corrigido: a figura pinta-se com uma trinca FIXA, então a
    paleta do sistema já não muda cor nenhuma — pôr a paleta escura não separa mais nada. **O que
    separa é `savefig.facecolor`**, que é justamente o `rcParams` global: com ele escuro,
    a gravação explícita sai branca e a gravação pelo padrão global sai escura.

    A paleta escura fica activa na mesma, porque é o ambiente em que o profissional vai usar isto
    e porque é ela que torna a pergunta interessante — e porque um dia alguém pode voltar a ligar
    `cores_do_tema` ao desenho.
    """
    from matplotlib import image as imagem_do_mpl
    from matplotlib import rcParams

    aplicacao = QApplication.instance()
    assert isinstance(aplicacao, QApplication)
    paleta_anterior = aplicacao.palette()
    aplicacao.setPalette(_paleta(*ESCURA))
    try:
        monkeypatch.setitem(rcParams, "savefig.facecolor", ESCURA[0])
        monkeypatch.setitem(rcParams, "savefig.edgecolor", ESCURA[0])
        assert to_hex(rcParams["savefig.facecolor"]) != CORES_DA_FIGURA[0], (
            "âncora: o `rcParams` plantado é a mesma cor que a figura pinta, e a planta abaixo não "
            "separaria os dois caminhos"
        )

        explicito = tmp_path / "explicito.png"
        painel, _pedidos = _painel_para_exportar(qtbot, explicito)
        assert painel.exporta()[0] is DesfechoDaExportacao.GRAVADA

        # A PLANTA: a mesma figura, gravada pelo padrão global — o caminho recusado.
        global_ = tmp_path / "global.png"
        painel.canvas.figure.savefig(global_, dpi=DPI_DA_EXPORTACAO)
    finally:
        aplicacao.setPalette(paleta_anterior)

    canto_explicito = imagem_do_mpl.imread(explicito)[0][0]
    canto_global = imagem_do_mpl.imread(global_)[0][0]
    assert to_hex(canto_explicito) == CORES_DA_FIGURA[0], (
        f"o arquivo gravado por `exporta` tem o canto em {to_hex(canto_explicito)} e a figura "
        f"pinta-se em {CORES_DA_FIGURA[0]}"
    )
    assert to_hex(canto_global) != to_hex(canto_explicito), (
        "a planta não separou: gravar pelo padrão global deu a MESMA cor que a gravação explícita, "
        "e este cenário passaria com o `facecolor` removido de `exporta`"
    )


def test_u4_par_causal_o_enquadramento_NAO_segue_o_rcParams_de_bbox(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R4 pede *"o mesmo enquadramento que a tela"*, e `savefig.bbox` é o que o muda por fora.

    **E `bbox_inches=None` NÃO pina nada** — `print_figure` lê o `rcParams` quando o argumento é
    `None`, medido em 15/09/2026 na fonte da dependência. Quem pina é o `rc_context` de `exporta`,
    e é ele que esta planta separa.
    """
    from matplotlib import rcParams

    monkeypatch.setitem(rcParams, "savefig.bbox", "tight")

    destino = tmp_path / "figura.png"
    painel, _pedidos = _painel_para_exportar(qtbot, destino)
    esperado = _pixels_da_construcao(painel, DPI_DA_EXPORTACAO)

    assert painel.exporta()[0] is DesfechoDaExportacao.GRAVADA
    do_produto = _pixels_do_png(destino)

    # A PLANTA: a mesma figura, no MESMO tamanho do arquivo, gravada sem o `rc_context` —
    # o caminho que lê o `rcParams`. Sem o mesmo tamanho, a planta separaria pelo tamanho e não
    # pelo recorte.
    cortado = tmp_path / "cortado.png"
    figura = painel.canvas.figure
    tela = tuple(float(v) for v in figura.get_size_inches())
    largura_mm, altura_mm = painel.tamanho_de_construcao_mm
    figura.set_size_inches(largura_mm / 25.4, altura_mm / 25.4, forward=False)
    try:
        figura.savefig(
            cortado,
            dpi=DPI_DA_EXPORTACAO,
            facecolor=figura.get_facecolor(),
            edgecolor=figura.get_edgecolor(),
            bbox_inches=None,
        )
    finally:
        figura.set_size_inches(*tela, forward=False)
    da_planta = _pixels_do_png(cortado)

    assert da_planta != do_produto, (
        f"a planta não separou: com `savefig.bbox = 'tight'` a gravação pelo argumento `None` deu "
        f"{da_planta} e a do produto deu {do_produto} — este cenário passaria sem o `rc_context`"
    )
    assert abs(do_produto[0] - esperado[0]) <= 1 and abs(do_produto[1] - esperado[1]) <= 1, (
        f"o arquivo do produto mede {do_produto} e o tamanho de construção pede {esperado}"
    )


def test_u4_a_gravacao_a_144_tem_o_MESMO_enquadramento_que_a_96(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """96 é o valor padrão, e é o que executa menos deste caminho.

    A 150% o Windows devolve `logicalDotsPerInch() == 144`, e é o monitor em que a janela vai ser
    demonstrada. O arquivo passou a ter o tamanho de construção, e não o da janela: a
    escala da tela muda as polegadas da janela e não pode mudar o arquivo. A âncora é que as duas
    escalas desenham janelas diferentes; a asserção é que os dois arquivos são do mesmo tamanho.
    """
    medidos: dict[float, tuple[int, int]] = {}
    na_tela: dict[float, tuple[int, int]] = {}
    for escala in (96.0, 144.0):
        destino = tmp_path / f"figura-{escala:.0f}.png"
        painel, _pedidos = _painel_para_exportar(qtbot, destino, escala=escala)
        assert painel.canvas.figure.dpi == escala, (
            f"âncora: pedi {escala} ppp e a figura desenhou a {painel.canvas.figure.dpi}"
        )
        esperado = _pixels_da_construcao(painel, DPI_DA_EXPORTACAO)
        na_tela[escala] = _pixels_esperados(painel.canvas.figure, DPI_DA_EXPORTACAO)

        assert painel.exporta()[0] is DesfechoDaExportacao.GRAVADA
        largura, altura = _pixels_do_png(destino)
        assert abs(largura - esperado[0]) <= 1 and abs(altura - esperado[1]) <= 1, (
            f"a {escala} ppp o arquivo mede {(largura, altura)} e a construção pede {esperado}"
        )
        medidos[escala] = (largura, altura)

    assert na_tela[96.0] != na_tela[144.0], (
        f"âncora: as duas escalas deram a MESMA janela, {na_tela[96.0]} — a escala não está a "
        "chegar ao desenho, e este cenário seria uma segunda cópia do de 96"
    )
    assert medidos[96.0] == medidos[144.0], (
        f"os arquivos divergem entre as escalas: {medidos} — o tamanho do arquivo mudou com o "
        "fator de escala da tela"
    )


def test_u4_todo_desfecho_da_exportacao_TEM_texto_e_nenhum_nasce_mudo() -> None:  # noqa: N802
    """O vocabulário de desfecho é FECHADO, como o dos estados: desfecho sem texto levanta.

    Um `.get` com recuo faria um desfecho novo nascer mudo, e mudo na tela depois de um clique
    lê-se como *"gravou"* — que é o oposto de quatro dos cinco.
    """
    assert set(TEXTO_DO_DESFECHO) == set(DesfechoDaExportacao), (
        f"a tabela de textos cobre {sorted(TEXTO_DO_DESFECHO)} e o vocabulário é "
        f"{sorted(DesfechoDaExportacao)}"
    )
    # Sete e não cinco: o DPI fora do vocabulário e o tamanho em que o rodapé colide.
    # Oito: a recusa por colisão saiu — a figura sai sem o rodapé, que é uma gravação
    # —, e entraram essa gravação e a imagem grande demais para a memória ou para o Pillow.
    assert len(DesfechoDaExportacao) == 8, (
        f"âncora: o vocabulário tem {len(DesfechoDaExportacao)} membros e a exportação enumera "
        "oito saídas"
    )
    for desfecho in DesfechoDaExportacao:
        assert texto_do_desfecho(desfecho, "detalhe").strip(), desfecho

    with pytest.raises(KeyError):
        texto_do_desfecho(cast("DesfechoDaExportacao", "desfecho-que-nao-existe"), "")


# ---------------------------------------------------------------------------
# A exportação para publicação: PNG, TIFF e PDF, no DPI e no tamanho pedidos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sufixo", "assinaturas"),
    [(".png", (b"\x89PNG",)), (".tif", (b"II*\x00", b"MM\x00*")), (".pdf", (b"%PDF",))],
)
def test_d657_cada_formato_sai_com_a_assinatura_dele(
    qtbot: QtBot, tmp_path: Path, sufixo: str, assinaturas: tuple[bytes, ...]
) -> None:
    """O formato é o do sufixo do nome escolhido, e o arquivo começa pela assinatura dele."""
    destino = tmp_path / f"figura{sufixo}"
    painel, _pedidos = _painel_para_exportar(qtbot, destino)
    desfecho, frase = painel.exporta()
    assert desfecho is DesfechoDaExportacao.GRAVADA, frase
    assert destino.read_bytes()[:4] in assinaturas


def test_d657_o_tiff_e_RGB_com_LZW_e_o_dpi_pedido_e_o_savefig_padrao_nao(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """RGB, LZW e a tag de 600 ppp, no tamanho pedido; e o canto é branco (a trinca fixa).

    **A planta é o caminho padrão do `savefig` em TIFF, que grava RGBA** — o canal alfa que há
    revista que recusa. Ela tem de reprovar o mesmo critério, senão o teste não mede nada.
    """
    from PIL import Image

    destino = tmp_path / "figura.tif"
    painel, _pedidos = _painel_para_exportar(qtbot, destino)
    desfecho, frase = painel.exporta(dpi=600, tamanho_mm=(174.0, 110.0))
    assert desfecho is DesfechoDaExportacao.GRAVADA, frase
    assert "600 pontos por polegada" in frase, frase
    with Image.open(destino) as tiff:
        assert tiff.mode == "RGB"
        assert tiff.info.get("compression") == "tiff_lzw"
        assert tuple(round(v) for v in tiff.info["dpi"]) == (600, 600)
        esperado = (int(174.0 / 25.4 * 600), int(110.0 / 25.4 * 600))
        assert abs(tiff.size[0] - esperado[0]) <= 1 and abs(tiff.size[1] - esperado[1]) <= 1
        assert tiff.getpixel((0, 0)) == (255, 255, 255), "o canto não é branco"

    planta = tmp_path / "planta.tif"
    painel.canvas.figure.savefig(planta, format="tiff", dpi=600)
    with Image.open(planta) as rgba:
        assert rgba.mode != "RGB", "a planta não separa: o savefig padrão já gravaria RGB"


def test_d657_o_pdf_embute_truetype_sem_tipo3_e_sai_igual_duas_vezes(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Fontes TrueType embutidas, nenhuma Tipo 3, e nenhuma data: a mesma figura, o mesmo PDF."""
    primeiro, segundo = tmp_path / "a.pdf", tmp_path / "b.pdf"
    painel, _pedidos = _painel_para_exportar(qtbot, primeiro)
    assert painel.exporta()[0] is DesfechoDaExportacao.GRAVADA
    painel._escolhedor = _escolhedor(segundo)[0]  # type: ignore[assignment]
    assert painel.exporta()[0] is DesfechoDaExportacao.GRAVADA
    bruto = primeiro.read_bytes()
    assert b"/Type3" not in bruto
    assert b"/FontFile2" in bruto, "nenhuma fonte TrueType embutida"
    assert bruto == segundo.read_bytes(), "dois PDFs da mesma figura diferem"


@pytest.mark.parametrize(("largura_mm", "altura_mm"), [(200.0, 120.0), (174.0, None)])
def test_d657_o_tamanho_pedido_sai_e_o_rodape_nao_colide(
    qtbot: QtBot, tmp_path: Path, largura_mm: float, altura_mm: float | None
) -> None:
    """200 x 120 mm, e a coluna dupla da Springer (174 mm) com a altura de construção.

    A folga é medida na figura no tamanho do arquivo, pelo mesmo `folga_do_rodape` que a
    exportação consulta antes de escrever; e a janela fica com o tamanho que tinha.
    """
    from refino.plots.recorte import folga_do_rodape

    destino = tmp_path / "figura.png"
    painel, _pedidos = _painel_para_exportar(qtbot, destino)
    altura = altura_mm or painel.tamanho_de_construcao_mm[1]
    figura = painel.canvas.figure
    tela = tuple(float(v) for v in figura.get_size_inches())
    largura_do_widget = painel.canvas.width()

    desfecho, frase = painel.exporta(tamanho_mm=(largura_mm, altura))

    assert desfecho is DesfechoDaExportacao.GRAVADA, frase
    largura, altura_px = _pixels_do_png(destino)
    assert abs(largura - int(largura_mm / 25.4 * DPI_DA_EXPORTACAO)) <= 1
    assert abs(altura_px - int(altura / 25.4 * DPI_DA_EXPORTACAO)) <= 1
    assert tuple(float(v) for v in figura.get_size_inches()) == tela, "a figura da tela mudou"
    assert painel.canvas.width() == largura_do_widget, "o canvas mudou de tamanho"

    figura.set_size_inches(largura_mm / 25.4, altura / 25.4, forward=False)
    painel._reajusta_o_recorte(None)
    folga = folga_do_rodape(figura)
    figura.set_size_inches(*tela, forward=False)
    painel._reajusta_o_recorte(None)
    assert folga is not None and folga > 0.0, f"o rodapé colide em {largura_mm} mm: {folga}"


def _rodapes_crus(figura: Figure) -> list[str]:
    """Os textos do rodapé, com as quebras de linha: voltar «parecido» não é voltar como era."""
    return [t.get_text() for t in figura.texts if t.get_gid() == GID_DO_RODAPE]


def _transbordo_em_pontos(figura: Figure) -> float:
    """Quanto o gráfico — título, rótulos e marcas incluídos — passa das bordas da figura, em pt.

    **Zero quando tudo cabe.** A caixa é a apertada de cada eixo, a mesma que `folga_do_rodape`
    lê, medida no desenho sem pintar — e em pontos, para não depender do DPI.
    """
    figura.draw_without_rendering()
    desenhador = figura.canvas.get_renderer()  # type: ignore[attr-defined]
    largura, altura = figura.bbox.width, figura.bbox.height
    excesso = 0.0
    for eixo in figura.axes:
        caixa = eixo.get_tightbbox(desenhador)
        assert caixa is not None
        excesso = max(excesso, -caixa.x0, -caixa.y0, caixa.x1 - largura, caixa.y1 - altura)
    return float(excesso) * 72.0 / float(figura.dpi)


def _espia_a_gravacao(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, float]]:
    """Remenda `_grava` para anotar o rodapé e o transbordo NO INSTANTE da escrita.

    **A gravação continua a ser a real**: a espia chama a função original, e o arquivo é escrito.
    O que ela acrescenta é o que o disco não diz sem decodificar a imagem — o texto do rodapé que
    a figura levava e quanto do gráfico ficava fora das bordas.
    """
    original = modulo_das_figuras._grava
    vistos: list[tuple[str, float]] = []

    def espia(figura: Figure, destino: Path, formato: str, dpi: float) -> None:
        rodape = "".join(t.get_text() for t in figura.texts if t.get_gid() == GID_DO_RODAPE)
        vistos.append((rodape, _transbordo_em_pontos(figura)))
        original(figura, destino, formato, dpi)

    monkeypatch.setattr(modulo_das_figuras, "_grava", espia)
    return vistos


def test_D668_abaixo_do_minimo_a_figura_sai_SEM_rodape_e_o_texto_vai_para_o_txt(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Na coluna simples o rodapé não cabe: a figura sai sem ele, e o texto vai para um `.txt`.

    Antes era recusa: abaixo do mínimo nada era escrito. Agora a figura é gravada sem o rodapé, e
    o texto dele vai num `.txt` ao lado, para a legenda do artigo.

    **O par causal é a mesma figura na mesma largura, com a altura mínima e abaixo dela.** Na
    mínima o rodapé vai NA figura e nenhum `.txt` nasce — é a âncora que impede as asserções de
    ausência de passarem por vacuidade; abaixo, o rodapé sai da figura, a faixa dele volta
    ao gráfico, e o texto está no `.txt`, igual ao que a figura declara.
    """
    from refino.desktop.janela.figuras import caminho_do_rodape

    destino = tmp_path / "figura.png"
    rodape_em = caminho_do_rodape(destino)
    assert rodape_em == tmp_path / "figura-rodape.txt", rodape_em
    painel, _pedidos = _painel_para_exportar(qtbot, destino)
    vistos = _espia_a_gravacao(monkeypatch)
    figura = painel.canvas.figure
    tela = tuple(float(v) for v in figura.get_size_inches())
    rodape_da_tela = _rodapes_crus(figura)
    assert rodape_da_tela and rodape_da_tela[0].strip(), "âncora: a figura da tela não tem rodapé"
    eixos_da_tela = [float(v) for eixo in figura.axes for v in eixo.get_position().bounds]
    largura_do_widget = painel.canvas.width()
    minima = painel.altura_minima_mm(90.0)
    assert minima is not None and 20.0 < minima < 600.0, f"âncora: mínimo {minima}"

    desfecho, frase = painel.exporta(tamanho_mm=(90.0, minima))
    assert desfecho is DesfechoDaExportacao.GRAVADA, frase
    assert vistos[-1][0].strip(), "âncora: na altura mínima a figura foi escrita sem rodapé"
    assert not rodape_em.exists(), "o rodapé coube na figura e mesmo assim nasceu um `.txt`"
    destino.unlink()

    # **A planta: o rodapé só APAGADO**, com o leiaute que ele deixou no tamanho do arquivo. Ali
    # o `tight_layout` com a faixa reservada não se aplica, e o gráfico sai pelas bordas — medido
    # em 06/11/2026 a 90 x 39 mm: 9,6 pt abaixo, 17,8 pt à esquerda e 18,6 pt acima da figura.
    altura = minima - 2.0
    figura.set_size_inches(90.0 / 25.4, altura / 25.4, forward=False)
    painel._reajusta_e_mede()
    for texto in figura.texts:
        if texto.get_gid() == GID_DO_RODAPE:
            texto.set_text("")
    transbordo_da_planta = _transbordo_em_pontos(figura)
    figura.set_size_inches(*tela, forward=False)
    painel._reajusta_o_recorte(None)
    assert transbordo_da_planta > 1.0, (
        f"âncora: com o rodapé só apagado o gráfico transborda {transbordo_da_planta:.1f} pt, e a "
        "asserção da faixa devolvida passaria sem `retira`"
    )
    assert _rodapes_crus(figura) == rodape_da_tela, "âncora: a planta não foi desfeita"

    desfecho, frase = painel.exporta(tamanho_mm=(90.0, altura))

    assert desfecho is DesfechoDaExportacao.GRAVADA_SEM_RODAPE, frase
    largura_px, altura_px = _pixels_do_png(destino)
    assert abs(largura_px - int(90.0 / 25.4 * DPI_DA_EXPORTACAO)) <= 1
    assert abs(altura_px - int(altura / 25.4 * DPI_DA_EXPORTACAO)) <= 1
    rodape_gravado, transbordo_gravado = vistos[-1]
    assert rodape_gravado == "", f"a figura foi escrita com o rodapé: {rodape_gravado!r}"
    assert transbordo_gravado == 0.0, (
        f"o gráfico foi escrito {transbordo_gravado:.1f} pt para lá das bordas: a faixa do rodapé "
        "não voltou ao leiaute"
    )
    assert rodape_em.read_text(encoding="utf-8") == f"{TEXTO_DO_RODAPE_DE_TESTE}\n"
    assert str(destino) in frase and str(rodape_em) in frase, frase
    assert "sem o rodapé" in frase, frase

    assert tuple(float(v) for v in figura.get_size_inches()) == tela, "a figura da tela mudou"
    assert painel.canvas.width() == largura_do_widget, "o canvas mudou de tamanho"
    assert _rodapes_crus(figura) == rodape_da_tela, "o rodapé da tela não voltou como era"
    eixos_depois = [float(v) for eixo in figura.axes for v in eixo.get_position().bounds]
    assert eixos_depois == pytest.approx(eixos_da_tela, abs=1e-9), (
        f"os eixos da tela não voltaram ao lugar: {eixos_depois} contra {eixos_da_tela}"
    )


def test_D668_o_txt_que_FALHA_impede_a_figura_sem_rodape(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """O `.txt` vai primeiro: se ele não pode ser escrito, a figura sem rodapé também não é.

    **A ordem é o que impede a figura de perder a declaração do recorte em silêncio**: uma figura
    gravada sem rodapé e sem o `.txt` não diz de que população foi desenhada. A planta é um
    diretório no lugar do `.txt` — a escrita dele levanta `OSError` —, e a figura não pode estar
    no disco depois.
    """
    from refino.desktop.janela.figuras import caminho_do_rodape

    destino = tmp_path / "figura.png"
    caminho_do_rodape(destino).mkdir()
    painel, _pedidos = _painel_para_exportar(qtbot, destino)
    minima = painel.altura_minima_mm(90.0)
    assert minima is not None, "âncora: a figura de teste não tem mínimo em 90 mm"
    assert painel.exporta(tamanho_mm=(90.0, minima))[0] is DesfechoDaExportacao.GRAVADA, (
        "âncora: na altura mínima nem o diretório no caminho do `.txt` impede a gravação"
    )
    destino.unlink()

    desfecho, frase = painel.exporta(tamanho_mm=(90.0, minima - 2.0))

    assert desfecho is DesfechoDaExportacao.ESCRITA_FALHOU, frase
    assert str(caminho_do_rodape(destino)) in frase, f"a frase não nomeia o `.txt`: {frase!r}"
    assert not destino.exists(), "a figura foi escrita sem rodapé e sem o texto dele"
    assert painel.estado_visivel()["figura"] == "1 eixo(s)"


def test_d657_o_dpi_fora_dos_tres_e_recusado_antes_do_dialogo(qtbot: QtBot, tmp_path: Path) -> None:
    """O DPI é vocabulário fechado, e o valor fora dele não chega ao `savefig`."""
    destino = tmp_path / "figura.png"
    painel, pedidos = _painel_para_exportar(qtbot, destino)
    desfecho, frase = painel.exporta(dpi=450)
    assert desfecho is DesfechoDaExportacao.DPI_RECUSADO, frase
    assert pedidos == [] and not destino.exists()
    assert set(DPIS_DA_EXPORTACAO) == {300, 600, 1200}


def test_d657_a_aba_dos_graficos_leva_os_campos_ao_painel(qtbot: QtBot, tmp_path: Path) -> None:
    """Os campos da aba chegam à exportação, e o mínimo aparece antes de salvar."""
    from refino.desktop.janela.graficos import ROTULO_DA_EXPORTACAO, PainelDeGraficos

    destino = tmp_path / "figura.tif"
    graficos = PainelDeGraficos(escolhedor=_escolhedor(destino)[0])
    qtbot.addWidget(graficos)
    graficos.figuras.desenha(figura_de_teste(), nomes=NOMES)
    assert graficos.botao_de_exportar.text() == ROTULO_DA_EXPORTACAO == "Salvar figura…"
    construcao = graficos.figuras.tamanho_de_construcao_mm
    assert construcao is not None
    assert graficos.largura_da_exportacao.value() == round(construcao[0]), "o campo não seguiu"

    graficos.dpi_da_exportacao.setCurrentIndex(graficos.dpi_da_exportacao.findData(600))
    graficos.largura_da_exportacao.setValue(150)
    graficos.altura_da_exportacao.setValue(100)
    graficos._tamanho_editado()
    assert "altura mínima" in graficos.minimo_da_exportacao.text()

    graficos.botao_de_exportar.click()
    assert "600 pontos por polegada" in graficos.exportacao.text(), graficos.exportacao.text()
    from PIL import Image

    with Image.open(destino) as tiff:
        assert abs(tiff.size[0] - int(150 / 25.4 * 600)) <= 1


def test_d657_os_campos_da_exportacao_nao_alargam_a_aba_dos_graficos(qtbot: QtBot) -> None:
    """Os campos de gravar não pedem mais largura que a figura — nem com a frase do mínimo escrita.

    A primeira versão reprovou aqui por vizinhança, e é por isso que o teste existe.
    Resolução, largura e altura numa linha só pediam 536 px, e a aba passou de 486 para 576: com a
    legenda da «Qualidade» aberta, a janela deixou de descer a 1024 px
    (`test_legenda_longa_cabe_aberta_e_reflui_ao_redimensionar`, `1060 == 1024`). O teto é a área
    da figura mais as margens do painel, que é o que a aba pedia antes.
    """
    from refino.desktop.janela.graficos import PainelDeGraficos

    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    area = graficos.figuras.minimumSizeHint().width()
    assert area > 300, f"âncora: a área da figura pede só {area} px, e o teto seria vácuo"
    disposicao = graficos.layout()
    assert disposicao is not None
    margens = disposicao.contentsMargins()
    teto = area + margens.left() + margens.right()
    assert graficos.minimumSizeHint().width() <= teto, (
        f"os campos pedem {graficos.minimumSizeHint().width()} px, e a figura {teto}"
    )

    # A frase é a da produção, e a mais longa das duas — a da altura escolhida abaixo
    # do mínimo, que diz para onde vai o rodapé. Um literal escrito aqui mediria uma frase que a
    # tela não mostra.
    from refino.desktop.janela.figuras import frase_do_minimo

    frase = frase_do_minimo(123.0, 100.0)
    assert len(frase) > len(frase_do_minimo(123.0, 150.0)), "âncora: esta não é a mais longa"
    graficos.minimo_da_exportacao.setText(frase)
    disposicao.activate()
    assert graficos.minimumSizeHint().width() <= teto, (
        f"com a frase do mínimo, os campos pedem {graficos.minimumSizeHint().width()} px"
    )


# ---------------------------------------------------------------------------
# A figura sem o rodapé onde ele não cabe, o mínimo exato e a imagem grande demais
# ---------------------------------------------------------------------------

# **O texto que a figura de teste declara, refeito dos literais de `figura_de_teste`** — e não lido
# da figura, que é de onde o código sob teste o lê.
TEXTO_DO_RODAPE_DE_TESTE: Final = Recorte(
    desenhadas=DESENHADAS,
    disponiveis=DISPONIVEIS,
    amostragem="as quatro primeiras",
    matriz="ingerida",
    metodo="sem teste estatístico",
).texto()


def _bissecao_flutuante(cabe: Any) -> float:
    """A busca antiga, que parava num intervalo de 1 mm e devolvia o topo dele, sem arredondar."""
    baixo, alto = 20.0, 600.0
    while alto - baixo > 1.0:
        meio = (baixo + alto) / 2.0
        if cabe(meio):
            alto = meio
        else:
            baixo = meio
    return alto


def test_D668_o_minimo_e_o_MENOR_inteiro_que_cabe_e_round_e_ceil_nao_sao(  # noqa: N802
) -> None:
    """A altura mínima é o menor número inteiro de mm em que o rodapé cabe — nem menos, nem mais.

    **Os campos da aba são inteiros, e o número da tela é o que a pessoa digita.** A busca parava a
    bisseção num intervalo de 1 mm e devolvia `round(alto)`, que fica ABAIXO do limiar quando a
    parte fracionária é menor que meio: a tela dizia uma altura que a gravação recusava. O
    `math.ceil(alto)` da revisão de código nunca fica abaixo, mas fica UM mm acima quando o topo do
    intervalo passa do inteiro seguinte — e o mínimo separa duas saídas, então um mm a
    mais diz *"abaixo disto o rodapé vai para o .txt"* sobre uma altura em que ele cabe.

    **As plantas são as duas formas**, sobre limiares em que cada uma erra: 87,1 mm, em que o topo
    fica em 87,40 e o `round` dá 87; e 86,9 mm, em que o topo fica em 87,40 e o `ceil` dá 88.
    """
    import math

    from refino.desktop.janela.figuras import menor_altura_que_cabe

    def caso(limiar: float) -> tuple[Any, list[float]]:
        chamadas: list[float] = []

        def cabe(altura: float) -> bool:
            chamadas.append(altura)
            return altura >= limiar

        return cabe, chamadas

    planta_do_round, _ = caso(87.1)
    assert round(_bissecao_flutuante(planta_do_round)) == 87, "âncora: o `round` não erra aqui"
    planta_do_ceil, _ = caso(86.9)
    assert math.ceil(_bissecao_flutuante(planta_do_ceil)) == 88, "âncora: o `ceil` não erra aqui"

    for limiar, esperado in (
        (87.1, 88),
        (86.9, 87),
        (87.0, 87),
        (49.066, 50),
        (20.0, 20),
        (19.5, 20),
        (599.2, 600),
        (600.5, None),
    ):
        cabe, chamadas = caso(limiar)
        assert menor_altura_que_cabe(cabe) == esperado, limiar
        # O custo é o da busca antiga, doze medições no pior caso: as duas pontas e dez metades de
        # 580 mm. Uma busca linear acharia o mesmo número com quinhentas, e cada uma refaz o
        # rodapé da figura.
        assert len(chamadas) <= 12, (limiar, len(chamadas))


@pytest.mark.parametrize("largura_mm", [50.0, 60.0, 90.0, 174.0])
def test_D668_a_altura_minima_e_a_FRONTEIRA_entre_as_duas_saidas(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, largura_mm: float
) -> None:
    """Na altura mínima o rodapé vai na figura; um mm abaixo, vai para o `.txt` — em cada largura.

    **Várias larguras e não só 90 mm**, porque o limiar muda com o número de linhas do rodapé, e
    com uma largura só o teste passa ou falha conforme onde cai a fração do limiar. **50 mm é a
    planta medida**: o limiar ali é 49,066 mm, e o `round` antigo dava 49 — a tela dizia 49 e a
    gravação recusava (medido em 06/11/2026 sobre esta figura de teste, 33 larguras de 40 a 200 mm:
    o `round` ficou abaixo do limiar em 50 e 55 mm).
    """
    destino = tmp_path / "figura.png"
    painel, _pedidos = _painel_para_exportar(qtbot, destino)
    minima = painel.altura_minima_mm(largura_mm)
    assert minima is not None and 20.0 < minima < 600.0, f"âncora: mínimo {minima}"
    assert float(minima).is_integer(), f"o mínimo {minima} não é um número que o campo aceite"

    desfecho, frase = painel.exporta(tamanho_mm=(largura_mm, minima))
    assert desfecho is DesfechoDaExportacao.GRAVADA, (
        f"a tela diz {minima:.0f} mm como mínimo em {largura_mm:.0f} mm, e a gravação nessa "
        f"altura deu {desfecho}: {frase}"
    )
    desfecho, frase = painel.exporta(tamanho_mm=(largura_mm, minima - 1.0))
    assert desfecho is DesfechoDaExportacao.GRAVADA_SEM_RODAPE, (
        f"um mm abaixo do mínimo de {minima:.0f} mm, em {largura_mm:.0f} mm, deu {desfecho}: "
        f"{frase}"
    )


def test_D668_a_aba_diz_para_onde_vai_o_rodape_antes_e_depois_de_salvar(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A frase do mínimo diz o que a altura escolhida faz ao rodapé; a do desfecho, onde ele está.

    **O par é a mesma largura com duas alturas**, uma acima e outra abaixo do mínimo: a frase muda
    de *"abaixo dela"* para *"na altura escolhida"*, e as duas nomeiam o `.txt`.
    """
    from refino.desktop.janela.figuras import SUFIXO_DO_RODAPE

    destino = tmp_path / "figura.png"
    graficos = PainelDeGraficos(escolhedor=_escolhedor(destino)[0])
    qtbot.addWidget(graficos)
    graficos.figuras.desenha(figura_de_teste(), nomes=NOMES)
    graficos.largura_da_exportacao.setValue(90)
    minima = graficos.figuras.altura_minima_mm(90.0)
    assert minima is not None, "âncora: a figura de teste não tem mínimo em 90 mm"

    graficos.altura_da_exportacao.setValue(round(minima) + 10)
    graficos._tamanho_editado()
    acima = graficos.minimo_da_exportacao.text()
    graficos.altura_da_exportacao.setValue(round(minima) - 5)
    graficos._tamanho_editado()
    abaixo = graficos.minimo_da_exportacao.text()

    assert acima != abaixo, f"a frase não muda com a altura escolhida: {acima!r}"
    for frase in (acima, abaixo):
        assert f"{minima:.0f} mm" in frase and SUFIXO_DO_RODAPE in frase, frase
    assert "Abaixo dela" in acima and "Na altura escolhida" in abaixo, (acima, abaixo)

    graficos.botao_de_exportar.click()
    desfecho = graficos.exportacao.text()
    assert str(tmp_path / "figura-rodape.txt") in desfecho and "sem o rodapé" in desfecho, desfecho
    assert (tmp_path / "figura-rodape.txt").read_text(encoding="utf-8").strip() == (
        TEXTO_DO_RODAPE_DE_TESTE
    )


@pytest.mark.parametrize("causa", ["limite-do-pillow", "memoria"])
def test_D668_a_imagem_GRANDE_DEMAIS_vira_desfecho_e_nao_sobe_do_slot(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, causa: str
) -> None:
    """`MemoryError` e o limite de pixels do Pillow viram frase, e nada foi escrito.

    **Nenhum dos dois é `OSError`**, que era o único apanhado: subiam pelo slot do botão, e o Qt os
    engole para a saída de erro — que no executável `--windowed` não existe —, com a linha do
    desfecho a dizer a frase anterior. **Sem gravar arquivo gigante:** o limite do Pillow é
    reduzido, como o `MAX_IMAGE_PIXELS` permite, e a falta de memória é levantada por uma
    gravação remendada — a do tamanho real, A3 a 1.200 ppp, pediria gigabytes.
    """
    from PIL import Image

    from refino.plots.recorte import milhar

    destino = tmp_path / "figura.tif"
    graficos = PainelDeGraficos(escolhedor=_escolhedor(destino)[0])
    qtbot.addWidget(graficos)
    graficos.figuras.desenha(figura_de_teste(), nomes=NOMES)
    figura = graficos.figuras.canvas.figure
    tela = tuple(float(v) for v in figura.get_size_inches())
    graficos.largura_da_exportacao.setValue(150)
    graficos.altura_da_exportacao.setValue(100)
    graficos.dpi_da_exportacao.setCurrentIndex(graficos.dpi_da_exportacao.findData(600))
    if causa == "memoria":

        def sem_memoria(*_args: object) -> None:
            raise MemoryError

        monkeypatch.setattr(modulo_das_figuras, "_grava", sem_memoria)
    else:
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)

    graficos.botao_de_exportar.click()

    frase = graficos.exportacao.text()
    largura_px, altura_px = int(150 / 25.4 * 600), int(100 / 25.4 * 600)
    assert "grande demais" in frase, frase
    assert f"{milhar(largura_px)} x {milhar(altura_px)} pixels" in frase, frase
    assert not destino.exists(), f"algo foi escrito: {sorted(p.name for p in tmp_path.iterdir())}"
    assert tuple(float(v) for v in figura.get_size_inches()) == tela, "a figura da tela mudou"
    desfecho, _frase = graficos.figuras.exporta(dpi=600, tamanho_mm=(150.0, 100.0))
    assert desfecho is DesfechoDaExportacao.GRANDE_DEMAIS


# ---------------------------------------------------------------------------
# O corpo da medição chega ao painel, e nada mais muda
# ---------------------------------------------------------------------------
#
# **`figuras.define_execucao` NÃO é chamado com a resposta da medição.** Aquele corpo não tem
# `chave_de_invalidacao`, então `chave_de_invalidacao` devolveria um marcador irrepetível, o painel
# reconheceria «execução nova» e **retiraria** a figura corrente. `recebe_medicao` guarda e mais
# nada — é o molde dos seis `recebe_*` que já existem.
#
# **Esta seção não desenha.** Quem consome `_medicao` é o painel da correção; o que
# se afirma aqui é que o insumo chega, chega DISCRIMINÁVEL, e não sobrevive à troca de matriz.

MEDICAO_DE_TESTE: Final[dict[str, Any]] = {
    "vencedor": "serrf",
    "scores": {"none": 1.5219, "loess": 1.5534, "serrf": 1.4013},
    "margem": 0.0208,
    "avaliavel": True,
    "abstem": False,
    "estimador": "leave_one_qc_out",
    "variante": "outlier_removed",
    "n_features_medidas": None,
    "palpite": "serrf",
    "palpite_acertou": True,
}


def _medicao(**troca: Any) -> dict[str, Any]:
    """O corpo de `POST /matrizes/medir-corretor` que a janela recebe, com trocas pontuais."""
    return {**MEDICAO_DE_TESTE, **troca}


# ---------------------------------------------------------------------------
# O PAR de PCAs, que é o insumo do painel da correção
# ---------------------------------------------------------------------------
#
# **A raiz `pcas_da_correcao` de `/execucoes`, na forma achatada que `ParDePcas` declara.** Os dois
# lados levam populações IGUAIS em cardinalidade e conteúdo DIFERENTE, que é o que a figura precisa
# para os dois painéis não saírem idênticos por acidente de fixtura.
_TIPOS_DO_PAR: Final = ["qc", "sample", "sample", "sample", "qc", "sample"]

PAR_DE_PCAS_DE_TESTE: Final[dict[str, Any]] = {
    "matriz_antes": "ingerida",
    "matriz_depois": "corrigida",
    "pretransformacao": "none",
    "n_componentes": 2,
    # **O corretor que produziu o lado «depois», e é o `xlab` do painel direito** —
    # `tools.R:1597`. **`loess` e não `none` por omissão:** com `none` a fixtura não separaria o
    # nome do corretor do rótulo `raw data` do painel esquerdo, e a planta cairia sobre dado em que
    # as duas coisas não diferem.
    "corretor": "loess",
    "scores_pc1_antes": [-3.0, -1.5, 0.0, 1.5, 3.0, 4.5],
    "scores_pc2_antes": [1.0, -0.5, 2.0, -1.5, 0.5, -2.0],
    "tipos_de_amostra_antes": list(_TIPOS_DO_PAR),
    "variancia_explicada_antes": [0.34, 0.098],
    "qc_dispersion_ratio_antes": 0.977,
    "n_features_usadas_antes": 6035,
    "n_features_totais_antes": 6035,
    "digest_das_colunas_retidas_antes": "3151a4a6" * 8,
    "scores_pc1_depois": [-2.0, -0.5, 1.0, 2.5, 4.0, 5.5],
    "scores_pc2_depois": [0.5, -1.0, 1.5, -0.5, 1.5, -1.0],
    "tipos_de_amostra_depois": list(_TIPOS_DO_PAR),
    "variancia_explicada_depois": [0.245, 0.081],
    "qc_dispersion_ratio_depois": 0.2329,
    "n_features_usadas_depois": 6035,
    "n_features_totais_depois": 6035,
    "digest_das_colunas_retidas_depois": "3151a4a6" * 8,
}


def _com_par_de_pcas(seed: int, **troca: Any) -> dict[str, Any]:
    """O corpo de `/execucoes` com a raiz `pcas_da_correcao` dentro — o molde de `_com_plsda`."""
    return {**_corpo(seed), "pcas_da_correcao": {**PAR_DE_PCAS_DE_TESTE, **troca}}


# ---------------------------------------------------------------------------
# Os dois vetores de RSD por feature, insumo do violino
# ---------------------------------------------------------------------------
#
# **A raiz `rsd_por_feature` de `/execucoes`, na forma achatada que `RsdPorFeature` declara.** As
# duas contagens de features DIFEREM e nenhuma coincide com contagem de injeção nenhuma — é o que a
# figura precisa para *"o `n` de cada lado"* não passar com o mesmo número escrito duas vezes, e
# para a troca por `n_injecoes_*` não ser indistinguível.
_RSD_DOS_QCS: Final = [0.09, 0.11, 0.12, 0.13, 0.14, 0.16, 0.18, 0.21, 0.26]
_RSD_DAS_AMOSTRAS: Final = [0.22, 0.28, 0.31, 0.35, 0.39, 0.44, 0.52, 0.67]

RSD_POR_FEATURE_DE_TESTE: Final[dict[str, Any]] = {
    "matriz": "corrigida",
    "corretor": "loess",
    "rsd_variant": "outlier_removed",
    "rsd_estimator": "in_sample",
    "rsd_dos_qcs": list(_RSD_DOS_QCS),
    "rsd_das_amostras": list(_RSD_DAS_AMOSTRAS),
    "n_features": 12,
    "n_features_finitas_dos_qcs": len(_RSD_DOS_QCS),
    "n_features_finitas_das_amostras": len(_RSD_DAS_AMOSTRAS),
    "n_injecoes_dos_qcs": 7,
    "n_injecoes_das_amostras": 61,
}


def _com_rsd_por_feature(seed: int, **troca: Any) -> dict[str, Any]:
    """O corpo de `/execucoes` com a raiz `rsd_por_feature` dentro — o molde de `_com_plsda`."""
    return {**_corpo(seed), "rsd_por_feature": {**RSD_POR_FEATURE_DE_TESTE, **troca}}


# ---------------------------------------------------------------------------
# A composição por classe, insumo da figura D
# ---------------------------------------------------------------------------
#
# **A raiz `composicao_por_classe` de `/execucoes`, na forma que `ComposicaoPorClasse` declara.**
# As duas classes têm contagens DIFERENTES em tudo — injeções, células, faltantes e abundância —,
# porque é isso que impede uma leitura de chave trocada de passar com o mesmo número escrito duas
# vezes. E `n_celulas` NÃO é `n_injecoes * n_features` da fixtura: o `n_features` de teste é 12 e
# `13 * 12 = 156`, não 78.455 — a distância é deliberada, para que a figura que derivasse o
# denominador do par em vez de o ler da classe reprove.
_CLASSE_ATLL: Final[dict[str, Any]] = {
    "grupo": "ATLL",
    "n_injecoes": 13,
    "n_celulas": 78455,
    "n_zeros": 12930,
    "n_nao_finitas": 0,
    "n_faltantes": 12930,
    "n_detectadas": 65525,
    "log10_mediana_das_detectadas": 4.0507,
}

_CLASSE_NOHTLV: Final[dict[str, Any]] = {
    "grupo": "NoHTLV",
    "n_injecoes": 16,
    "n_celulas": 96560,
    "n_zeros": 15568,
    "n_nao_finitas": 0,
    "n_faltantes": 15568,
    "n_detectadas": 80992,
    "log10_mediana_das_detectadas": 4.0791,
}

COMPOSICAO_DE_TESTE: Final[dict[str, Any]] = {
    "matriz": "ingerida",
    "populacao": "sample",
    "n_injecoes_fora_da_populacao": 7,
    "n_features": 12,
    "definicao_de_faltante": "zero_ou_nao_finita",
    "definicao_de_abundancia": "mediana_do_log10_das_detectadas",
    "classes": [dict(_CLASSE_ATLL), dict(_CLASSE_NOHTLV)],
}


def _com_composicao(seed: int, **troca: Any) -> dict[str, Any]:
    """O corpo de `/execucoes` com a raiz `composicao_por_classe` — o molde de `_com_plsda`."""
    return {**_corpo(seed), "composicao_por_classe": {**COMPOSICAO_DE_TESTE, **troca}}


def test_u4_receber_a_medicao_nao_mexe_na_chave_nem_RETIRA_figura(qtbot: QtBot) -> None:  # noqa: N802
    """O que se impede, medido nas três superfícies que a medição poderia mexer.

    Chamar `figuras.define_execucao(resposta_da_medicao)` produziria chave divergente — aquele corpo
    não traz `chave_de_invalidacao` — e o painel retiraria a figura corrente. Aqui o oráculo é o
    estado ANTES e DEPOIS: a chave, o combo e a figura desenhada.

    **A âncora de não-vacuidade vem primeiro e na mesma função**: sem uma figura desenhada,
    *"nada foi retirado"* é satisfeito por um painel que nunca desenhou.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_pca(_com_pca())
    graficos.figuras.define_execucao(_resposta(200, _corpo(31)))
    _escolhe(graficos, "figura_dos_scores")

    antes = dict(graficos.figuras.estado_visivel())
    chave_antes = graficos.figuras._chave
    montadores_antes = sorted(graficos._montadores())
    assert antes["figura"] == "1 eixo(s)", (
        f"âncora: não há figura desenhada antes da medição — {antes}. Sem ela, «a medição não "
        "retirou nada» passaria sobre um painel vazio"
    )

    graficos.recebe_medicao(_medicao())

    assert graficos.figuras._chave == chave_antes, (
        "a medição mexeu na chave de invalidação: alguém a passou por `figuras.define_execucao`, "
        "que é exactamente o que KTD11 recusa"
    )
    assert dict(graficos.figuras.estado_visivel()) == antes, (
        f"a medição mudou o que o painel mostra: {antes} -> {graficos.figuras.estado_visivel()}"
    )
    assert sorted(graficos._montadores()) == montadores_antes, (
        "a medição mexeu na lista de figuras que o painel sabe construir"
    )
    assert len(montadores_antes) == 32, (
        f"receber a medição não regista construtor nenhum, e há {len(montadores_antes)}: "
        f"{montadores_antes}. O número sobe de um por unidade que regista figura (KTD14): 13 → 14 "
        "na U3 do plano dos gráficos, 14 → 15 na U7, 15 → 16 na U8, 24 → 26 na U12 do bloco das "
        "análises, que traz DUAS, 26 → 28 na U11, que traz as outras DUAS, 28 → 30 na U15, que "
        "traz os dois classificadores, 30 → 31 na U21, que traz a rede de correlação parcial, e "
        "31 → 32 com a curva do LOESS"
    )
    assert graficos._medicao == _medicao(), "e a medição não ficou guardada"


def test_u4_uma_resposta_SEM_scores_deixa_o_painel_sem_medicao_guardada(qtbot: QtBot) -> None:  # noqa: N802
    """O par causal da guarda: com `scores` guarda, sem `scores` não — e o velho é APAGADO.

    **Duas direções e não uma**: uma guarda que recusasse **sempre** passaria na metade
    de baixo e deixaria o painel sem barras para sempre, e uma que aceitasse sempre passaria na de
    cima guardando o corpo de erro como se fosse medição.

    **O ramo de recusa é o que corre para TODA resposta terminal da rota.** `_encaminha` chama isto
    antes de decidir entre a prosa e o texto de erro, então o corpo de um `409` — que não tem
    `scores` nenhum — apaga a medição anterior em vez de a deixar na tela sob a execução seguinte.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    graficos.recebe_medicao(_medicao())
    assert graficos._medicao is not None, (
        "âncora: nem com o corpo completo a medição ficou guardada, e a metade de baixo não "
        "mediria nada"
    )

    graficos.recebe_medicao({"detail": "a medição não completou"})
    assert graficos._medicao is None, (
        "um corpo sem `scores` ficou guardado como medição: o painel da correção desenharia a fila "
        "de barras de um corpo de erro"
    )

    for corpo, porque in (
        (None, "resposta sem corpo nenhum"),
        ({}, "corpo vazio"),
        (_medicao(scores={}), "`scores` presente e vazio"),
        (_medicao(avaliavel=False), "o servidor disse que a medição não é avaliável"),
    ):
        graficos.recebe_medicao(_medicao())
        assert graficos._medicao is not None, f"âncora: a reposição falhou antes de «{porque}»"
        graficos.recebe_medicao(corpo)
        assert graficos._medicao is None, f"ficou medição guardada com {porque}: {corpo!r}"


def test_u4_a_medicao_que_chega_DEPOIS_da_troca_de_matriz_e_descartada(qtbot: QtBot) -> None:  # noqa: N802
    """A corrida que estava declarada conhecida e não fechada, fechada.

    `define_matriz` só reescreve a identidade no **disparo**, e `recebe_medicao` não a comparava
    com nada. A cadeia é *medir A → trocar para B → a resposta de A chega*, e a fila de barras de
    **A** ficava a ser desenhada sob a execução de **B**, com o rótulo certo e o conteúdo de outra
    matriz, e nada acusava.

    **O par causal tem de DIVERGIR, e a metade de cima é a que o torna honesto**: a mesma
    sequência **sem** a troca guarda. Uma guarda que descartasse sempre passaria só na metade de
    baixo, e o painel ficaria sem barras para sempre — que é o defeito oposto e igualmente mudo.

    **A terceira metade é a ORDEM do carimbo**: carimbar **antes** de `define_matriz` descartaria a
    resposta certa. Aqui a ordem é a de `_dispara_medicao` — definir e só então carimbar.

    > *«`define_matriz` só reescreve a identidade no **disparo**»* deixou de valer: a
    > janela chama-o também na troca de arquivos e na do modo. E ele esconde a medição de A em vez
    > de a apagar — `_medicao` é a que se vê, e a guardada volta com A —, então as asserções abaixo
    > continuam a ler o que a figura desenharia sob B.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    # SEM troca: definir A, carimbar A, a resposta de A chega e é guardada.
    graficos.define_matriz("matriz-A")
    graficos.carimba_a_medicao()
    graficos.recebe_medicao(_medicao())
    sem_troca = graficos._medicao
    assert sem_troca is not None, (
        "âncora: a medição da matriz corrente foi descartada. A guarda recusa sempre, e a metade "
        "de baixo passaria sobre um painel que nunca guarda nada"
    )

    # COM troca: definir A, carimbar A, trocar para B, e só então a resposta de A chegar.
    graficos.define_matriz("matriz-A")
    graficos.carimba_a_medicao()
    graficos.define_matriz("matriz-B")
    assert graficos._medicao is None, "âncora: `define_matriz` já não descarta na troca"
    graficos.recebe_medicao(_medicao())

    assert graficos._medicao is None, (
        "a medição da matriz A ficou guardada depois de a tela passar para B: a fila de barras de "
        "uma matriz seria desenhada sob a execução de outra, com o rótulo certo"
    )
    assert sem_troca != graficos._medicao, (
        "os dois desfechos do par causal coincidem, e a troca de matriz não separa nada"
    )


def test_u4_o_disparo_da_medicao_CARIMBA_a_matriz_corrente(qtbot: QtBot) -> None:  # noqa: N802
    """O que impede a guarda acima de ficar inerte em silêncio.

    `recebe_medicao` não recusa o que chega **sem** carimbo, de propósito: recusar tornaria a
    ausência do carimbo uma fila de barras que nunca mais aparece, e ninguém veria porquê. O preço
    dessa escolha é que a guarda depende de `_dispara_medicao` carimbar — e é esta asserção que
    torna a retirada do carimbo uma reprovação em vez de um silêncio.

    **A ORDEM é o conteúdo**: o carimbo tem de ficar igual à identidade que `_pedido_de_arquivos`
    acabou de definir. Carimbar **antes** dela daria a matriz anterior, e a guarda passaria a
    descartar a resposta certa.
    """
    from refino.desktop.janela.arquivos import ParDeArquivos
    from refino.desktop.janela.principal import ROTA_DA_MEDICAO, JanelaPrincipal

    janela = JanelaPrincipal(_TransporteQueRecusa())
    qtbot.addWidget(janela)
    janela.arquivos._par = ParDeArquivos(
        "A.csv", b"feature,QC1\n1,2\n", "sA.csv", b"label,time\nQC1,1\n"
    )
    # O par entra direto, sem a escolha que deduz o modo: o modo vai à mão, senão a chamada é
    # recusada por falta dele.
    janela.arquivos.modo.setEditText("pos_hilic")
    enviadas: list[str] = []
    janela.chama = lambda metodo, rota, **kwargs: enviadas.append(rota)  # type: ignore[method-assign]

    assert janela.graficos._identidade_da_medicao is None, "âncora: já havia carimbo antes"
    janela._dispara_medicao()
    janela.espera_a_piscina(30_000)

    assert enviadas == [ROTA_DA_MEDICAO], (
        f"o disparo não alcançou a rota da medição: {enviadas}. Sem ele o carimbo abaixo seria "
        "afirmado sobre um caminho que não corre"
    )
    carimbo = janela.graficos._identidade_da_medicao
    assert carimbo is not None, (
        "`_dispara_medicao` não carimbou a matriz: a guarda de `recebe_medicao` fica INERTE, e a "
        "medição de uma matriz volta a poder ser desenhada sob a execução de outra"
    )
    assert carimbo == janela.graficos._identidade_da_matriz, (
        f"o carimbo {carimbo!r} não é a identidade corrente "
        f"{janela.graficos._identidade_da_matriz!r}: carimbar antes de `_pedido_de_arquivos` daria "
        "a matriz ANTERIOR, e a guarda passaria a descartar a resposta certa"
    )


def test_u4_um_score_None_com_avaliavel_NAO_vira_barra_de_altura_zero(qtbot: QtBot) -> None:  # noqa: N802
    """**`nan` serializa como `null`, e `null` não é zero** — a `Assumption` do plano, aqui medida.

    `MedicaoDoCorretorResposta.scores` é `dict[str, float]` e ainda assim uma barra chega `None`
    com `avaliavel=True`: o RSD ficou indefinido naquele corretor e nos outros não. Um painel que
    normalizasse o corpo — `float(v or 0)` — poria a barra do corretor **não medido** colada ao
    eixo, que é o retrato de uma correção perfeita.

    **O par causal cai sobre dado em que as duas coisas diferem:** os dois corpos são idênticos
    menos o valor do `serrf`, `None` num e `0.0` no outro, e o que se afirma é que o painel os
    mantém DISTINGUÍVEIS. Sem isso, o painel da correção não tem como escolher entre desenhar uma
    barra e declarar a ausência.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    com_nulo = _medicao(scores={"none": 1.5219, "loess": 1.5534, "serrf": None}, vencedor="none")
    com_zero = _medicao(scores={"none": 1.5219, "loess": 1.5534, "serrf": 0.0}, vencedor="serrf")
    assert com_nulo != com_zero, "âncora: os dois corpos do par são iguais, e não separam nada"

    graficos.recebe_medicao(com_nulo)
    guardado_nulo = graficos._medicao
    assert guardado_nulo is not None, (
        "uma medição com UM corretor indefinido foi descartada inteira: as outras duas barras "
        "existem, e `avaliavel` é verdadeiro"
    )
    assert guardado_nulo["scores"]["serrf"] is None, (
        f"o `None` do `serrf` virou {guardado_nulo['scores']['serrf']!r} — o painel normalizou o "
        "corpo, e «não medido» passou a ser indistinguível de «medido e deu zero»"
    )

    graficos.recebe_medicao(com_zero)
    guardado_zero = graficos._medicao
    assert guardado_zero is not None and guardado_zero["scores"]["serrf"] == 0.0
    assert guardado_nulo["scores"] != guardado_zero["scores"], (
        "os dois desfechos não divergem: o painel entrega a mesma coisa para «não medido» e para "
        "«zero», e a planta não separa nada"
    )


def test_u4_o_painel_e_a_PROSA_concordam_sobre_haver_medicao(qtbot: QtBot) -> None:  # noqa: N802
    """A irmã, em forma mecânica: os dois consumidores do mesmo corpo têm de concordar.

    `prosa_da_medicao` decide se há ranking a escrever com `not avaliavel or not valores`; este
    painel decide se há fila de barras a desenhar. **Se os dois critérios divergirem, a aba da
    correção escreve que a medição não foi avaliável enquanto a aba dos gráficos desenha três
    barras** — e nada acusa, porque cada um passa no seu próprio teste.

    O oráculo é a frase que a prosa reserva para o caso sem ranking, e a asserção é uma
    **equivalência** sobre cinco corpos, não uma implicação: as duas direções erram de formas
    diferentes.
    """
    from refino.interface.fluxo import prosa_da_medicao

    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    corpos = [
        _medicao(),
        _medicao(scores={"none": 1.5219, "loess": 1.5534, "serrf": None}),
        _medicao(avaliavel=False),
        _medicao(scores={}),
        _medicao(scores={"none": None, "loess": None, "serrf": None}, avaliavel=False),
    ]
    vistos: set[bool] = set()
    for corpo in corpos:
        graficos.recebe_medicao(corpo)
        ha_barras = graficos._medicao is not None
        prosa = prosa_da_medicao(
            vencedor=corpo.get("vencedor"),
            scores=corpo.get("scores"),
            margem=corpo.get("margem"),
            avaliavel=corpo.get("avaliavel"),
            abstem=corpo.get("abstem"),
            n_features_medidas=corpo.get("n_features_medidas"),
        )
        ha_ranking = "não foi avaliável" not in prosa
        assert ha_barras == ha_ranking, (
            f"o painel diz {ha_barras} e a prosa diz {ha_ranking} sobre o MESMO corpo {corpo!r}. "
            "Uma aba desenharia barras enquanto a outra escreve que não houve comparação"
        )
        vistos.add(ha_barras)

    assert vistos == {True, False}, (
        f"os cinco corpos deram sempre {vistos}: a equivalência acima é satisfeita por vacuidade "
        "num lado só (N desfechos, N direções)"
    )


def test_u4_a_medicao_NAO_sobrevive_a_troca_de_matriz(qtbot: QtBot) -> None:  # noqa: N802
    """**A medição não tem chave de invalidação própria, e é este o buraco que ela deixa.**

    As seis cargas anteriores viajam no corpo de `/execucoes` ou de `/matrizes/diagnosticar` e
    morrem com a chave da execução. A medição vem de uma rota própria, minutos antes, e o corpo dela
    não traz chave nenhuma: sem uma regra própria, a fila de barras de **`A`** aparece sob a
    execução de **`B`**, com o rótulo certo e o conteúdo de outra matriz.

    **A identidade é `interface/fluxo.chave_da_entrada` com `metodo=None`**, e o `None` é o
    conteúdo: a medição corre os TRÊS corretores, então ela não depende de qual deles se vai
    executar a seguir
    — é o argumento de `CHAVES_SEM_METODO`. Trocar o corretor entre medir e executar **não** pode
    descartar a medição, e trocar os arquivos tem de descartar.

    ## O par causal, e os dois braços diferem só pelo par de arquivos

    (a) `arquivo A → medir → arquivo B → executar` e (b) `arquivo A → medir → arquivo A →
    executar`. Tudo o mais é idêntico, inclusive o corpo da medição e o da execução. A asserção é
    que os desfechos **divergem** — sem a metade (b), *"a medição sumiu"* seria satisfeito por um
    painel que nunca a guardou.
    """
    from refino.desktop.janela.arquivos import ParDeArquivos
    from refino.desktop.janela.principal import JanelaPrincipal

    a = ParDeArquivos("A.csv", b"feature,QC1\n1,2\n", "sA.csv", b"label,time\nQC1,1\n")
    b = ParDeArquivos("B.csv", b"feature,QC1\n9,8\n", "sB.csv", b"label,time\nQC1,1\n")
    assert a.bytes_das_medidas != b.bytes_das_medidas, (
        "âncora: os dois pares de arquivos têm os mesmos bytes, e o braço (a) não trocaria matriz "
        "nenhuma"
    )

    desfechos: dict[str, Any] = {}
    for rotulo, segundo_par in (("trocou", b), ("manteve", a)):
        janela = JanelaPrincipal(_TransporteQueRecusa())
        qtbot.addWidget(janela)

        janela.arquivos._par = a
        # O par entra direto, sem a escolha que deduz o modo: o modo vai à mão.
        janela.arquivos.modo.setEditText("pos_hilic")
        assert janela._pedido_de_arquivos() is not None, "âncora: o par `A` não foi aceite"
        janela._encaminha("medicao", _resposta(200, _medicao()))
        assert janela.graficos._medicao is not None, (
            f"âncora ({rotulo}): a medição não chegou ao painel dos gráficos pelo caminho da "
            "janela, e o resto deste cenário seria vácuo"
        )

        janela.arquivos._par = segundo_par
        assert janela._pedido_de_arquivos() is not None, "âncora: o segundo par não foi aceite"
        janela._encaminha("execucao", _resposta(200, _corpo(41)))
        desfechos[rotulo] = janela.graficos._medicao

    assert desfechos["trocou"] is None, (
        "as barras da medição de `A` sobreviveram à execução de `B`: o painel da correção "
        "desenharia a fila da matriz ERRADA sob o rótulo da execução corrente, e nada acusaria"
    )
    assert desfechos["manteve"] is not None, (
        "a planta não separa: a medição também sumiu quando a matriz NÃO mudou, então o braço (a) "
        "passaria com um painel que descarta sempre — e a fila de barras nunca apareceria"
    )


# ---------------------------------------------------------------------------
# O painel da correção, ligado ao painel da janela
# ---------------------------------------------------------------------------
#
# **Esta figura é a primeira do registo que lê DUAS cargas, de DUAS rotas.** O par de PCAs vem de
# `/execucoes` e morre com a chave daquela execução; a medição vem de `/matrizes/medir-corretor` e
# só morre quando a matriz muda. Então a ligação tem quatro estados e não dois, e os testes abaixo
# percorrem os quatro: N desfechos, N direções.
#
# | par de PCAs | medição | o que a figura faz |
# |---|---|---|
# | ausente | qualquer | recusa por população vazia: sem execução não há o que desenhar |
# | presente | ausente | desenha as duas PCAs, e o lugar das barras leva texto declarado |
# | presente | presente | desenha as duas PCAs e a fila, uma barra por corretor |
# | presente | com abstenção | idem, e nenhuma barra leva o destaque de eleito |

_FUNCAO_DA_CORRECAO: Final = "figura_do_painel_da_correcao"


def _painel_da_correcao(qtbot: QtBot, **cargas: Any) -> PainelDeGraficos:
    """Um `PainelDeGraficos` com o painel da correção na tela, e só as cargas que se pedir."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    # O `recebe_*` do par corre sempre, como na janela — sem a carga, com o corpo da execução sem a
    # raiz. Sem ele a figura não teria dono, e esperaria a etapa em vez de recusar o vazio.
    graficos.recebe_pca_do_par(cargas.get("par", _corpo(77)), corresponde=True)
    if "medicao" in cargas:
        graficos.recebe_medicao(cargas["medicao"])
    graficos.figuras.define_execucao(_resposta(200, _corpo(77)))
    _escolhe(graficos, _FUNCAO_DA_CORRECAO)
    return graficos


def test_u3_o_painel_da_correcao_esta_no_seletor_e_DESENHA_do_par_de_pcas(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """A ligação ponta a ponta: a raiz de `/execucoes` vira três eixos na tela.

    **O oráculo é a contagem por série contra o vetor de tipos que veio no corpo**, e não o facto de
    o construtor não ter levantado: uma figura bem formada com o vetor errado tem exactamente a
    mesma aparência.

    **A linha da região de predição é afirmada à parte, e não somada aos pontos.** Ela é
    uma série com os vértices do contorno; a soma dos pontos continua exacta sobre as séries de
    ponto, e a região tem de lá estar, uma por painel.
    """
    from refino.plots.correcao import rotulo_da_regiao

    graficos = _painel_da_correcao(qtbot, par=_com_par_de_pcas(77), medicao=_medicao())
    assert graficos.figuras.estado_visivel()["figura"] == "3 eixo(s)"
    assert graficos.figuras.seletor.currentText() == "painel da correcao"

    tipos = PAR_DE_PCAS_DE_TESTE["tipos_de_amostra_antes"]
    quantos_qc = sum(1 for t in tipos if t == "qc")
    assert quantos_qc and quantos_qc < len(tipos), (
        f"âncora: a fixtura não separa QC de amostra: {tipos}"
    )

    _barras, antes, depois = graficos.figuras.canvas.figure.axes
    for eixo, lado in ((antes, "Before"), (depois, "After")):
        desenhadas = {s.rotulo: s for s in series(eixo)}
        assert desenhadas, f"o painel «{lado}» não desenhou série nenhuma"
        assert rotulo_da_regiao(lado) in desenhadas, sorted(desenhadas)
        pontos = {r: s for r, s in desenhadas.items() if r != rotulo_da_regiao(lado)}
        do_qc = pontos[f"QC — {lado}"]
        assert do_qc.x.size == quantos_qc, (do_qc.x.size, quantos_qc)
        assert sum(s.x.size for s in pontos.values()) == len(tipos)
        assert confere_vista(eixo) == []


def test_u3_SEM_o_par_de_pcas_a_figura_recusa_por_populacao_vazia(qtbot: QtBot) -> None:  # noqa: N802
    """A execução é a condição de existir figura, e a medição sozinha não a cria.

    É a verificação da medição, e ela só é verificável com o construtor do painel. O par causal
    compara os dois desfechos: com par há três eixos; sem par há texto. Sem a metade de cima,
    *"não desenhou"* passaria com um painel que nunca desenha.
    """
    com = _painel_da_correcao(qtbot, par=_com_par_de_pcas(77), medicao=_medicao())
    assert com.figuras.estado_visivel()["figura"] == "3 eixo(s)", (
        "âncora: nem com o par o painel desenhou, e a planta abaixo não separa nada"
    )

    sem = _painel_da_correcao(qtbot, medicao=_medicao())
    visivel = sem.figuras.estado_visivel()
    assert visivel["figura"] == "", visivel
    assert "população vazia" in visivel["repouso"], visivel
    assert visivel["repouso"] != com.figuras.estado_visivel()["repouso"]


def test_u3_COM_par_e_SEM_medicao_o_lugar_das_barras_leva_texto(qtbot: QtBot) -> None:  # noqa: N802
    """Ponta a ponta: sem medição não há fila, e o desfecho é legítimo e não estado partido.

    **O par causal compara os dois desfechos e exige que divirjam**: com medição o eixo de cima tem
    uma série por corretor; sem medição tem zero séries e um texto.
    """
    com = _painel_da_correcao(qtbot, par=_com_par_de_pcas(77), medicao=_medicao())
    sem = _painel_da_correcao(qtbot, par=_com_par_de_pcas(77))

    das_barras_com = com.figuras.canvas.figure.axes[0]
    das_barras_sem = sem.figuras.canvas.figure.axes[0]
    assert sem.figuras.estado_visivel()["figura"] == "3 eixo(s)", (
        "sem medição o painel deixou de desenhar: as duas PCAs continuam a responder à fala do "
        "laboratório, e a medição decide só se há fila de barras"
    )

    esperadas = set(MEDICAO_DE_TESTE["scores"])
    assert {s.rotulo for s in series(das_barras_com)} == esperadas, [
        s.rotulo for s in series(das_barras_com)
    ]
    assert not series(das_barras_sem), [s.rotulo for s in series(das_barras_sem)]
    assert "Medir qual corretor vence" in texto_corrido(das_barras_sem.texts[0].get_text())


def _o_que_a_dica_nao_nomeia(dica: str, *, botao: str, aba: str) -> list[str]:
    """O que falta à dica: o nome do botão de medir, e o da aba que o mostra, entre «».

    O botão chega com o sufixo do custo — *«Medir qual corretor vence (minutos)»* — e a dica cita o
    NOME, sem ele; então compara-se o texto antes do primeiro parêntese.
    """
    nome = botao.split(" (", 1)[0].strip()
    return [alvo for alvo in (f"«{nome}»", f"«{aba}»") if alvo not in dica]


def test_a_dica_nomeia_o_botao_e_a_aba_que_a_janela_tem(qtbot: QtBot) -> None:
    """A dica do painel da correção diz ONDE clicar, e o sítio tem de existir.

    **A amarra vive aqui e não em `plots/`**, porque `plots/` não importa Qt e o nome
    do botão é de `PainelDePipeline`. Os dois lados são DERIVADOS: o rótulo do botão lido do widget
    construído, e a aba lida de `ABAS` pela chave do painel que o tem. O texto de antes mandava
    correr «medir corretor» na «aba da correção», e nenhum dos dois nomes existia na janela.
    """
    from refino.desktop.janela.pipeline import PainelDePipeline
    from refino.desktop.janela.principal import ABAS
    from refino.plots.correcao import DICA_DE_MEDIR

    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    botao = painel.medir_botao.text()
    abas = [aba.titulo for aba in ABAS if aba.etapa == 1]
    assert botao and len(abas) == 1, f"âncora: botão {botao!r}, abas com o pipeline {abas}"

    assert _o_que_a_dica_nao_nomeia(DICA_DE_MEDIR, botao=botao, aba=abas[0]) == []

    # O par causal: a dica de `HEAD`, verbatim, não nomeia nenhum dos dois.
    antiga = (
        "sem fila de barras: a medição dos corretores não correu para esta matriz. Corra «medir "
        "corretor» na aba da correção e volte aqui"
    )
    assert len(_o_que_a_dica_nao_nomeia(antiga, botao=botao, aba=abas[0])) == 2

    # **E o custo não é um número estático — revisão de 25/09/2026.** A redação anterior dizia
    # «leva de 10 a 31 minutos nos quatro arquivos deste estudo, numa máquina de 12 núcleos», sem a
    # condição (`paralelo` ligado) e sobre arquivos que não são os de quem abre a figura. A dica
    # aponta para a estimativa AO VIVO, e ela tem de estar no mesmo grupo do botão.
    assert painel.rotulo_do_palpite.parent() is painel.medir_botao.parent(), (
        "a dica diz que a estimativa aparece logo abaixo do botão, e o rótulo mora noutro grupo"
    )
    minutos_fixos = re.compile(r"\d+ a \d+\s+minutos")
    assert not minutos_fixos.search(DICA_DE_MEDIR), DICA_DE_MEDIR
    anterior = (
        "leva de 10 a 31 minutos nos quatro arquivos deste estudo, numa máquina de 12 núcleos"
    )
    assert minutos_fixos.search(anterior), "âncora: o padrão não acusa a redação que o motivou"


def test_a_ressalva_do_estimador_nomeia_o_botao_que_mede_fora_da_amostra(qtbot: QtBot) -> None:
    """`RESSALVA_DO_ESTIMADOR` dizia que o `leave_one_qc_out` não era «exposto pela interface».

    **Era falso desde que a janela ganhou o botão de medir**, e a revisão de 25/09/2026 apanhou-o: a
    aba Qualidade concatena a ressalva ao par antes-e-depois, e a dica do painel da correção manda o
    usuário justamente para o botão que mede fora da amostra. A ressalva passa a nomeá-lo, e a
    amarra é a mesma da dica — os dois nomes lidos do widget e de `ABAS`, não repetidos.
    """
    from refino.api.esquemas import RESSALVA_DO_ESTIMADOR
    from refino.desktop.janela.pipeline import PainelDePipeline
    from refino.desktop.janela.principal import ABAS

    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    botao = painel.medir_botao.text()
    abas = [aba.titulo for aba in ABAS if aba.etapa == 1]
    assert botao and len(abas) == 1, f"âncora: botão {botao!r}, abas com o pipeline {abas}"

    assert _o_que_a_dica_nao_nomeia(RESSALVA_DO_ESTIMADOR, botao=botao, aba=abas[0]) == []
    assert "exposto pela interface" not in RESSALVA_DO_ESTIMADOR

    # O par causal: a última frase de `HEAD` (`118690c`), verbatim, não nomeia nenhum dos dois.
    antiga = (
        "A alternativa é o estimador fora da amostra `leave_one_qc_out` (`leave_one_qc_out_rsd`), "
        "que existe na biblioteca e não é servido por esta rota nem exposto pela interface."
    )
    assert len(_o_que_a_dica_nao_nomeia(antiga, botao=botao, aba=abas[0])) == 2


def test_u3_com_ABSTENCAO_nenhuma_barra_leva_o_destaque_de_eleito(qtbot: QtBot) -> None:  # noqa: N802
    """R8 ponta a ponta, pelo caminho da janela — e o par causal separa pelo `abstem` só."""
    from refino.plots.correcao import AMBAR_DO_MELHOR

    def cores(painel: PainelDeGraficos) -> list[str]:
        eixo = painel.figuras.canvas.figure.axes[0]
        return [str(to_hex(colecao.get_colors()[0])) for colecao in eixo.collections]

    sem = _painel_da_correcao(qtbot, par=_com_par_de_pcas(77), medicao=_medicao(abstem=False))
    com = _painel_da_correcao(qtbot, par=_com_par_de_pcas(77), medicao=_medicao(abstem=True))

    ambar = to_hex(AMBAR_DO_MELHOR)
    assert ambar in cores(sem), (
        f"sem abstenção também não há âmbar: {cores(sem)}. A planta não separa nada"
    )
    assert ambar not in cores(com), (
        f"com `abstem=True` uma barra continua em âmbar: {cores(com)}. A imagem afirma uma eleição "
        "que a política recusou, e a imagem diz mais alto que qualquer rodapé"
    )


def test_u10_o_eixo_DIREITO_leva_o_corretor_do_corpo_e_o_esquerdo_leva_raw_data(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """O nome do corretor atravessa `/execucoes` e chega ao `xlab` do painel direito.

    **O oráculo é o rótulo DESENHADO e não o campo do corpo**: a raiz podia crescer e ninguém a
    ler, que é o desfecho silencioso que este painel já pagou uma vez. É a ligação ponta a ponta,
    do corpo ao artista.

    **O par causal tem as duas metades.** Trocar o corretor no corpo muda o eixo **direito**; e o
    eixo **esquerdo** é `raw data` nos dois, porque a matriz ingerida não passou por corretor
    nenhum. Sem a segunda, *"o rótulo mudou"* passaria com uma janela que escrevesse o corretor nos
    dois eixos.
    """
    com_loess = _painel_da_correcao(
        qtbot, par=_com_par_de_pcas(77, corretor="loess"), medicao=_medicao()
    )
    com_none = _painel_da_correcao(
        qtbot, par=_com_par_de_pcas(77, corretor="none"), medicao=_medicao()
    )
    assert com_loess.figuras.estado_visivel()["figura"] == "3 eixo(s)", (
        "âncora: o painel não desenhou, e as leituras de rótulo abaixo seriam sobre nada"
    )

    def rotulos(painel: PainelDeGraficos) -> tuple[str, str]:
        _barras, antes, depois = painel.figuras.canvas.figure.axes
        return texto_corrido(antes.get_xlabel()), texto_corrido(depois.get_xlabel())

    esquerdo_loess, direito_loess = rotulos(com_loess)
    esquerdo_none, direito_none = rotulos(com_none)

    assert direito_loess.startswith("loess"), direito_loess
    assert direito_none.startswith("none"), direito_none
    assert direito_loess != direito_none, (
        f"o corretor do corpo não chega ao eixo direito: {direito_loess!r} nos dois. A raiz "
        "cresceu e o painel continua a escrever o estágio do pipeline"
    )
    assert "corrigida" not in direito_loess, (
        f"o eixo direito continua a levar o ESTÁGIO: {direito_loess!r}"
    )

    assert esquerdo_loess.startswith("raw data"), esquerdo_loess
    assert esquerdo_loess == esquerdo_none, (
        f"trocar o corretor mexeu no eixo ESQUERDO: {esquerdo_loess!r} contra {esquerdo_none!r}"
    )


def test_u3_o_par_de_pcas_NAO_sobrevive_a_execucao_seguinte(qtbot: QtBot) -> None:  # noqa: N802
    """O par viaja sob a chave da execução, e é isso que o separa da medição.

    A sequência é *executar A → executar B sem par*: o painel não pode desenhar o par de **A** sob a
    execução de **B**. `recebe_pca_do_par` corre para TODA resposta e apaga o campo quando a raiz
    não vem, que é o mesmo regime das outras cinco cargas de `/execucoes`.

    **O par causal separa**: com a raiz na segunda resposta há figura; sem ela, não há.
    """
    graficos = _painel_da_correcao(qtbot, par=_com_par_de_pcas(77), medicao=_medicao())
    assert graficos.figuras.estado_visivel()["figura"] == "3 eixo(s)", "âncora: A não desenhou"

    graficos.recebe_pca_do_par(_corpo(78), corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _corpo(78)))
    _escolhe(graficos, _FUNCAO_DA_CORRECAO)
    assert graficos.figuras.estado_visivel()["figura"] == "", graficos.figuras.estado_visivel()

    graficos.recebe_pca_do_par(_com_par_de_pcas(79), corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _corpo(79)))
    _escolhe(graficos, _FUNCAO_DA_CORRECAO)
    assert graficos.figuras.estado_visivel()["figura"] == "3 eixo(s)", (
        "com a raiz de volta o painel continua sem desenhar — a planta acima não é atribuível à "
        "ausência da raiz"
    )


# ---------------------------------------------------------------------------
# O violino de RSD por feature, ligado ao painel
# ---------------------------------------------------------------------------

_FUNCAO_DO_VIOLINO: Final = "figura_do_violino_do_rsd_por_feature"


def _painel_do_violino(qtbot: QtBot, corpo: Any) -> PainelDeGraficos:
    """Um `PainelDeGraficos` com o violino na tela, alimentado pelo corpo que se passar."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_rsd_por_feature(corpo, corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _corpo(81)))
    _escolhe(graficos, _FUNCAO_DO_VIOLINO)
    return graficos


def test_u7_a_lista_do_painel_ganhou_o_VIOLINO_e_continua_a_bijetar(qtbot: QtBot) -> None:  # noqa: N802
    """O montador entrou, e o combo mostra tudo o que o registo tem.

    **A contagem literal NÃO se afirma aqui** — ela vive num sítio só, em
    `test_u8_a_lista_do_painel_ganhou_as_SETE_e_continua_a_bijetar`. O que fica é o que esta
    unidade acrescentou, nomeado: nomear é mais forte que contar, porque a contagem passa se uma
    figura sair e outra entrar.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    montadores = graficos._montadores()
    assert _FUNCAO_DO_VIOLINO in montadores, sorted(montadores)
    assert set(montadores) == {c.funcao for c in CONSTRUTORES}
    assert graficos.figuras.seletor.count() == len(montadores)


def test_u7_o_violino_esta_no_seletor_e_DESENHA_dos_dois_vetores(qtbot: QtBot) -> None:  # noqa: N802
    """A ligação ponta a ponta: a raiz de `/execucoes` vira seis artistas na tela.

    **O oráculo é o que está desenhado contra o que veio no corpo**, e não o facto de o construtor
    não ter levantado: uma figura bem formada com o vetor errado tem exactamente a mesma aparência.
    O `n` de cada lado sai do corpo, e as duas contagens de features **diferem** — é o par que R3
    manda escrever.
    """
    graficos = _painel_do_violino(qtbot, _com_rsd_por_feature(81))
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"
    assert graficos.figuras.seletor.currentText() == "violino do rsd por feature"

    eixo = graficos.figuras.canvas.figure.axes[0]
    assert confere_vista(eixo) == []
    desenhadas = {s.rotulo: s for s in series(eixo)}
    assert len(desenhadas) == 6, sorted(desenhadas)

    # A mediana desenhada de cada lado, contra a mediana do vetor que veio no corpo — em
    # percentagem, que é a unidade do eixo, e derivada aqui e não do construtor.
    for rotulo, vetor in (("QCs", _RSD_DOS_QCS), ("amostras", _RSD_DAS_AMOSTRAS)):
        serie = desenhadas[f"{rotulo} — mediana"]
        assert np.allclose(serie.y, float(np.median(vetor)) * 100.0), (rotulo, serie.y)

    titulo = " ".join(eixo.get_title().split())
    assert f"{len(_RSD_DOS_QCS)} features" in titulo, titulo
    assert f"{len(_RSD_DAS_AMOSTRAS)} features" in titulo, titulo
    assert len(_RSD_DOS_QCS) != len(_RSD_DAS_AMOSTRAS), (
        "âncora: os dois lados têm a mesma contagem, e «o `n` de cada lado» passaria com o mesmo "
        "número escrito duas vezes"
    )


def test_u7_o_violino_le_a_contagem_de_FEATURES_e_nunca_a_de_injecoes(qtbot: QtBot) -> None:  # noqa: N802
    """**É na ligação que a troca de contagens aconteceria.** As nove leituras de chave vivem
    em `graficos.py`, e `n_injecoes_dos_qcs` está ao lado de `n_features_finitas_dos_qcs`.

    **O par causal cai sobre dado em que as duas coisas diferem:** o corpo plantado muda **só** as
    duas contagens de injeção, e o que a figura declara tem de sair idêntico. A âncora de
    não-vacuidade vem primeiro, na mesma função.
    """
    normal = _painel_do_violino(qtbot, _com_rsd_por_feature(81))
    plantado = _painel_do_violino(
        qtbot,
        _com_rsd_por_feature(81, n_injecoes_dos_qcs=444, n_injecoes_das_amostras=555),
    )
    do_normal = numeros_declarados(normal.figuras.canvas.figure)
    do_plantado = numeros_declarados(plantado.figuras.canvas.figure)
    assert do_normal, "âncora: a figura não declara número nenhum"
    assert do_normal == do_plantado, (
        f"trocar as contagens de INJEÇÃO mudou o que a figura declara: {do_normal} contra "
        f"{do_plantado}. O `n` de R3 é de FEATURES, e o painel leu a chave errada"
    )
    assert "444" not in do_normal and "555" not in do_plantado


def test_u7_o_rsd_por_feature_NAO_sobrevive_a_execucao_seguinte(qtbot: QtBot) -> None:  # noqa: N802
    """A carga viaja sob a chave da execução, como as outras cinco de `/execucoes`.

    A sequência é *executar A → executar B sem a raiz*: o painel não pode desenhar os vetores de
    **A** sob a execução de **B**. **O par causal separa**: com a raiz na segunda resposta há
    figura; sem ela, não há — e com a raiz de volta, volta a haver.
    """
    graficos = _painel_do_violino(qtbot, _com_rsd_por_feature(81))
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)", "âncora: A não desenhou"

    graficos.recebe_rsd_por_feature(_corpo(82), corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _corpo(82)))
    _escolhe(graficos, _FUNCAO_DO_VIOLINO)
    assert graficos.figuras.estado_visivel()["figura"] == "", graficos.figuras.estado_visivel()

    graficos.recebe_rsd_por_feature(_com_rsd_por_feature(83), corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _corpo(83)))
    _escolhe(graficos, _FUNCAO_DO_VIOLINO)
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)", (
        "com a raiz de volta o painel continua sem desenhar — a planta acima não é atribuível à "
        "ausência da raiz"
    )


def test_u7_com_ZERO_QCs_o_violino_declara_a_ausencia_e_desenha_as_amostras(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """A matriz sem QC é *"calculado e vazio"* e não *"não calculado"*, e a figura di-lo.

    **O par causal separa**: com QCs há seis artistas e nenhuma frase; sem QCs há três e a frase.
    Sem a metade de cima, *"a figura declara a ausência"* passaria com um painel que a escrevesse
    sempre.
    """
    com_qc = _painel_do_violino(qtbot, _com_rsd_por_feature(81))
    sem_qc = _painel_do_violino(
        qtbot,
        _com_rsd_por_feature(
            81, rsd_dos_qcs=[], n_features_finitas_dos_qcs=0, n_injecoes_dos_qcs=0
        ),
    )
    eixo_com = com_qc.figuras.canvas.figure.axes[0]
    eixo_sem = sem_qc.figuras.canvas.figure.axes[0]
    assert len(series(eixo_com)) == 6, [s.rotulo for s in series(eixo_com)]
    assert len(series(eixo_sem)) == 3, [s.rotulo for s in series(eixo_sem)]

    assert [t.get_text() for t in eixo_com.texts] == [], [t.get_text() for t in eixo_com.texts]
    declarado = " ".join(" ".join(t.get_text() for t in eixo_sem.texts).split())
    assert "nenhuma injeção" in declarado, declarado


# ---------------------------------------------------------------------------
# O violino da matriz INGERIDA, antes da primeira correção
# ---------------------------------------------------------------------------
#
# **Vetores DIFERENTES dos da execução, e a diferença é o oráculo de QUAL carga está desenhada.**
# Com os mesmos vetores, desenhar a ingerida no lugar da corrigida seria a identidade. **A
# forma da carga foi lida do PRODUTOR** — `POST /matrizes/diagnosticar` sobre os quatro arquivos
# devolve `matriz` «ingerida» e `corretor` nulo —, e o teste da janela pela rota real é a metade
# que confere.
_RSD_DOS_QCS_DA_INGERIDA: Final = [0.31, 0.35, 0.38, 0.42, 0.47, 0.55, 0.61]
_RSD_DAS_AMOSTRAS_DA_INGERIDA: Final = [0.41, 0.46, 0.5, 0.58, 0.63, 0.71]

RSD_DA_INGERIDA_DE_TESTE: Final[dict[str, Any]] = {
    **RSD_POR_FEATURE_DE_TESTE,
    "matriz": "ingerida",
    "corretor": None,
    "rsd_dos_qcs": list(_RSD_DOS_QCS_DA_INGERIDA),
    "rsd_das_amostras": list(_RSD_DAS_AMOSTRAS_DA_INGERIDA),
    "n_features_finitas_dos_qcs": len(_RSD_DOS_QCS_DA_INGERIDA),
    "n_features_finitas_das_amostras": len(_RSD_DAS_AMOSTRAS_DA_INGERIDA),
}
MEDIANA_DOS_QCS_DA_INGERIDA: Final = float(np.median(_RSD_DOS_QCS_DA_INGERIDA)) * 100.0
MEDIANA_DOS_QCS_DA_CORRIGIDA: Final = float(np.median(_RSD_DOS_QCS)) * 100.0


def _diagnostico_com_rsd() -> dict[str, Any]:
    """O corpo de `/matrizes/diagnosticar` com a raiz `rsd_por_feature` da ingerida."""
    return {**_com_pca(), "rsd_por_feature": dict(RSD_DA_INGERIDA_DE_TESTE)}


def _mediana_desenhada_dos_qcs(graficos: PainelDeGraficos) -> float:
    """A mediana que o violino na tela desenha do lado dos QCs, em percentagem."""
    desenhadas = {s.rotulo: s for s in series(graficos.figuras.canvas.figure.axes[0])}
    return float(desenhadas["QCs — mediana"].y[0])


def _painel_com_o_violino_da_ingerida(qtbot: QtBot) -> PainelDeGraficos:
    """Matriz A, diagnóstico de A, nenhuma execução, e o violino escolhido."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.define_matriz("matriz A")
    _escolhe(graficos, _FUNCAO_DO_VIOLINO)
    graficos.carimba_o_diagnostico()
    graficos.recebe_diagnostico(_diagnostico_com_rsd())
    graficos.figuras.reavalia()
    return graficos


def test_d567_o_violino_desenha_a_INGERIDA_com_o_diagnostico_e_sem_execucao(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """O violino visível antes da primeira correção.

    **Par causal:** o mesmo painel, sem e com diagnóstico, nenhuma execução nos dois; só o segundo
    desenha. **E o oráculo é o vetor desenhado, não o desfecho:** a mediana é a da ingerida, e o
    título diz o estágio em vez de nomear um corretor que não correu.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.define_matriz("matriz A")
    _escolhe(graficos, _FUNCAO_DO_VIOLINO)
    assert graficos.figuras.estado_visivel()["figura"] == ""
    assert graficos.figuras.repouso.text() == texto_do_estado(EstadoDaFigura.SEM_DIAGNOSTICO)

    graficos.carimba_o_diagnostico()
    graficos.recebe_diagnostico(_diagnostico_com_rsd())
    assert graficos.figuras.reavalia() == texto_do_estado(EstadoDaFigura.DESENHADA)
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"
    assert exporta_habilitado(graficos.figuras._estado), "o violino da ingerida nao e exportavel"
    assert MEDIANA_DOS_QCS_DA_INGERIDA != MEDIANA_DOS_QCS_DA_CORRIGIDA, (
        "âncora: as duas cargas têm a mesma mediana, e a asserção abaixo não diria qual delas "
        "está na tela"
    )
    assert np.isclose(_mediana_desenhada_dos_qcs(graficos), MEDIANA_DOS_QCS_DA_INGERIDA)

    # A linha da matriz vai para o título GERAL, que o violino tem.
    titulo = texto_corrido(graficos.figuras.canvas.figure.get_suptitle())
    assert "Matriz: uma — «ingerida», antes de qualquer correção" in titulo, titulo
    assert "corretor" not in titulo, titulo


def test_d567_o_202_e_o_erro_NAO_tiram_a_ingerida_e_o_primeiro_200_a_substitui(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """Na ordem em que a janela chama: os `recebe_*` e depois `define_execucao`.

    **Sem corretor nenhum ainda, a ingerida é o que há de verdade a desenhar** — o `202` e o erro
    não a tiram. O `200` traz a corrigida, e ela ocupa o lugar.

    **E a terceira metade é a que o desenho existe para impedir:** depois do primeiro corretor, o
    `202` da execução SEGUINTE retira o violino. Voltar à ingerida daria a figura a ir e vir; e se a
    regra fosse *«ingerida sempre que não há corrigida guardada»*, o painel nem a reconstruiria —
    o estado próprio seria `DESENHADA` antes e depois, e a reconstrução é pulada: o violino
    corrigido da execução anterior ficaria na tela durante a seguinte.
    """
    graficos = _painel_com_o_violino_da_ingerida(qtbot)
    assert np.isclose(_mediana_desenhada_dos_qcs(graficos), MEDIANA_DOS_QCS_DA_INGERIDA), (
        "âncora: a ingerida não está na tela"
    )

    for codigo, corpo in ((202, {"identificador": "x", "total": 5}), (500, {"detail": "falhou"})):
        graficos.recebe_rsd_por_feature(corpo, corresponde=True)
        graficos.figuras.define_execucao(_resposta(codigo, corpo))
        assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)", codigo
        assert np.isclose(_mediana_desenhada_dos_qcs(graficos), MEDIANA_DOS_QCS_DA_INGERIDA), codigo

    execucao = _com_rsd_por_feature(81)
    graficos.recebe_rsd_por_feature(execucao, corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, execucao))
    assert np.isclose(_mediana_desenhada_dos_qcs(graficos), MEDIANA_DOS_QCS_DA_CORRIGIDA), (
        "o 200 da primeira execução não substituiu a ingerida pela corrigida"
    )
    titulo = texto_corrido(graficos.figuras.canvas.figure.get_suptitle())
    assert "Matriz: uma — «corrigida», do corretor «loess»" in titulo, titulo

    seguinte = {"identificador": "y", "total": 5}
    graficos.recebe_rsd_por_feature(seguinte, corresponde=True)
    graficos.figuras.define_execucao(_resposta(202, seguinte))
    assert graficos.figuras.estado_visivel()["figura"] == "", (
        "o 202 da execução seguinte deixou um violino na tela"
    )
    assert graficos.figuras.repouso.text() == texto_do_estado(EstadoDaFigura.EM_CURSO)


def test_d567_trocar_de_matriz_volta_o_violino_a_INGERIDA_da_nova(qtbot: QtBot) -> None:  # noqa: N802
    """A matriz nova ainda não foi corrigida: o violino corrigido da anterior sai, e o diagnóstico
    da nova traz o dela. **Par causal:** antes da troca a corrigida está na tela.
    """
    graficos = _painel_com_o_violino_da_ingerida(qtbot)
    execucao = _com_rsd_por_feature(81)
    graficos.recebe_rsd_por_feature(execucao, corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, execucao))
    assert np.isclose(_mediana_desenhada_dos_qcs(graficos), MEDIANA_DOS_QCS_DA_CORRIGIDA), (
        "âncora: a corrigida de A não está na tela"
    )

    graficos.define_matriz("matriz B")
    assert graficos.figuras.estado_visivel()["figura"] == ""
    assert graficos.figuras.repouso.text() == texto_do_estado(EstadoDaFigura.SEM_DIAGNOSTICO)

    graficos.carimba_o_diagnostico()
    graficos.recebe_diagnostico(_diagnostico_com_rsd())
    graficos.figuras.reavalia()
    assert np.isclose(_mediana_desenhada_dos_qcs(graficos), MEDIANA_DOS_QCS_DA_INGERIDA)


def test_d567_sem_diagnostico_trocar_de_matriz_TAMBEM_tira_o_violino_corrigido(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """O ramo de `define_matriz` que reavalia sem diagnóstico guardado — o violino é o motivo dele.

    Antes a troca só reavaliava com diagnóstico, porque só as figuras dele dependiam da
    matriz. Sem este ramo, o violino corrigido da matriz anterior ficava na tela, e exportável, até
    a resposta seguinte.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.define_matriz("matriz A")
    execucao = _com_rsd_por_feature(81)
    graficos.recebe_rsd_por_feature(execucao, corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, execucao))
    _escolhe(graficos, _FUNCAO_DO_VIOLINO)
    assert not graficos._ha_diagnostico, "âncora: este painel não pode ter diagnóstico"
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)", "âncora: A não desenhou"

    graficos.define_matriz("matriz B")
    assert graficos.figuras.estado_visivel()["figura"] == ""
    assert not exporta_habilitado(graficos.figuras._estado), "o violino de A ficou exportável"


# ---------------------------------------------------------------------------
# A composição por classe, ligada ao painel
# ---------------------------------------------------------------------------

_FUNCAO_DA_COMPOSICAO: Final = "figura_da_composicao_por_classe"


def _painel_da_composicao(qtbot: QtBot, corpo: Any) -> PainelDeGraficos:
    """Um `PainelDeGraficos` com a composição na tela, alimentado pelo corpo que se passar."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_composicao(corpo, corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _corpo(91)))
    _escolhe(graficos, _FUNCAO_DA_COMPOSICAO)
    return graficos


def test_u8_a_lista_do_painel_ganhou_a_COMPOSICAO_e_continua_a_bijetar(qtbot: QtBot) -> None:  # noqa: N802
    """O montador entrou, e o combo mostra tudo o que o registo tem.

    **A contagem literal NÃO se afirma aqui** — ela vive num sítio só, em
    `test_u8_a_lista_do_painel_ganhou_as_SETE_e_continua_a_bijetar`. O que fica é o que esta
    unidade acrescentou, nomeado: nomear é mais forte que contar, porque a contagem passa se uma
    figura sair e outra entrar.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)

    montadores = graficos._montadores()
    assert _FUNCAO_DA_COMPOSICAO in montadores, sorted(montadores)
    assert set(montadores) == {c.funcao for c in CONSTRUTORES}
    assert graficos.figuras.seletor.count() == len(montadores)


def test_u8_a_composicao_esta_no_seletor_e_DESENHA_uma_barra_por_classe(qtbot: QtBot) -> None:  # noqa: N802
    """A ligação ponta a ponta: a raiz de `/execucoes` vira quatro artistas em dois eixos.

    **O oráculo é o que está desenhado contra o que veio no corpo**, e não o facto de o construtor
    não ter levantado: uma figura bem formada com a classe errada tem a mesma aparência. A altura de
    cada barra de faltantes é derivada AQUI, do corpo, e não pedida ao construtor.
    """
    graficos = _painel_da_composicao(qtbot, _com_composicao(91))
    assert graficos.figuras.estado_visivel()["figura"] == "2 eixo(s)"
    assert graficos.figuras.seletor.currentText() == "composicao por classe"

    desenhadas = {s.rotulo: s for eixo in graficos.figuras.canvas.figure.axes for s in series(eixo)}
    assert len(desenhadas) == 4, sorted(desenhadas)

    for bruto in (_CLASSE_ATLL, _CLASSE_NOHTLV):
        percentagem = bruto["n_faltantes"] / bruto["n_celulas"] * 100.0
        barra = desenhadas[f"{bruto['grupo']} — faltantes"]
        assert max(barra.y) == pytest.approx(percentagem), (bruto["grupo"], barra.y)
        abundancia = desenhadas[f"{bruto['grupo']} — abundância"]
        assert max(abundancia.y) == pytest.approx(bruto["log10_mediana_das_detectadas"])

    assert _CLASSE_ATLL["n_faltantes"] / _CLASSE_ATLL["n_celulas"] != pytest.approx(
        _CLASSE_NOHTLV["n_faltantes"] / _CLASSE_NOHTLV["n_celulas"]
    ), "âncora: as duas classes têm a mesma percentagem, e a leitura trocada passaria"


def test_u8_o_denominador_da_percentagem_e_N_CELULAS_DA_CLASSE(qtbot: QtBot) -> None:  # noqa: N802
    """**É na ligação que a troca de contagens aconteceria.** As seis leituras de chave vivem
    em `graficos.py`, e `n_injecoes` e `n_features` estão ao lado de `n_celulas`.

    **O par causal cai sobre dado em que as três coisas diferem:** o corpo plantado muda **só**
    `n_injecoes` e `n_features`, e a altura das barras tem de sair idêntica. Na fixtura
    `n_injecoes * n_features` é 156 e `n_celulas` é 78.455 — a distância é o que torna a troca
    visível. A âncora de não-vacuidade vem primeiro, na mesma função.
    """
    normal = _painel_da_composicao(qtbot, _com_composicao(91))
    plantado = _painel_da_composicao(
        qtbot,
        _com_composicao(
            91,
            n_features=999,
            classes=[
                {**_CLASSE_ATLL, "n_injecoes": 444},
                {**_CLASSE_NOHTLV, "n_injecoes": 555},
            ],
        ),
    )
    alturas = {
        s.rotulo: max(s.y) for eixo in normal.figuras.canvas.figure.axes for s in series(eixo)
    }
    plantadas = {
        s.rotulo: max(s.y) for eixo in plantado.figuras.canvas.figure.axes for s in series(eixo)
    }
    assert alturas, "âncora: a figura não desenhou artista nenhum"
    assert alturas == pytest.approx(plantadas), (
        f"trocar `n_injecoes` e `n_features` mudou a altura das barras: {alturas} contra "
        f"{plantadas}. O denominador da percentagem é `n_celulas` DA CLASSE"
    )


def test_u8_a_composicao_NAO_sobrevive_a_execucao_seguinte(qtbot: QtBot) -> None:  # noqa: N802
    """A carga viaja sob a chave da execução, como as outras de `/execucoes`.

    A sequência é *executar A → executar B sem a raiz*: o painel não pode desenhar a composição de
    **A** sob a execução de **B**. **O par causal separa**: com a raiz na segunda resposta há
    figura; sem ela, não há — e com a raiz de volta, voltam as barras.

    > **SEM A RAIZ o desfecho é `MatrizVaziaError` e NÃO o texto declarado, e a distinção
    > importa.** *"Nenhuma execução trouxe composição"* é **não calculado**, e é o que
    > `recorte_da_composicao_por_classe` recusa em `denominador_ou_recusa`: sem raiz não há
    > `n_features`, logo não há população de onde recortar. *"A rota calculou e não há classe
    > clínica"* é **calculado e vazio**, tem `n_features` e é o que a figura declara em texto —
    > medido em `test_u8_com_ZERO_classes_...`, que é onde esse desfecho se afirma.
    """
    graficos = _painel_da_composicao(qtbot, _com_composicao(91))
    assert len(series(graficos.figuras.canvas.figure.axes[0])) == 2, "âncora: A não desenhou barras"

    graficos.recebe_composicao(_corpo(92), corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _corpo(92)))
    _escolhe(graficos, _FUNCAO_DA_COMPOSICAO)
    assert graficos.figuras.estado_visivel()["figura"] == "", graficos.figuras.estado_visivel()

    graficos.recebe_composicao(_com_composicao(93), corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, _corpo(93)))
    _escolhe(graficos, _FUNCAO_DA_COMPOSICAO)
    assert len(series(graficos.figuras.canvas.figure.axes[0])) == 2, (
        "com a raiz de volta o painel continua sem barras — a planta acima não é atribuível à "
        "ausência da raiz"
    )


def test_u8_um_grupo_SEM_detectada_nao_vira_barra_e_a_ausencia_e_declarada(qtbot: QtBot) -> None:  # noqa: N802
    """**`None` no fio não vira barra no piso**: ausência de medida não é medida zero.

    **O par causal separa**: com abundância há duas barras no painel de baixo e nenhuma frase; sem
    ela há uma barra e a frase com o nome da classe. Sem a metade de cima, *"a figura declara a
    ausência"* passaria com um painel que a escrevesse sempre.
    """
    com = _painel_da_composicao(qtbot, _com_composicao(91))
    sem = _painel_da_composicao(
        qtbot,
        _com_composicao(
            91,
            classes=[
                dict(_CLASSE_ATLL),
                {
                    **_CLASSE_NOHTLV,
                    "n_detectadas": 0,
                    "n_faltantes": _CLASSE_NOHTLV["n_celulas"],
                    "n_zeros": _CLASSE_NOHTLV["n_celulas"],
                    "log10_mediana_das_detectadas": None,
                },
            ],
        ),
    )
    da_abundancia_com = series(com.figuras.canvas.figure.axes[1])
    da_abundancia_sem = series(sem.figuras.canvas.figure.axes[1])
    assert len(da_abundancia_com) == 2, [s.rotulo for s in da_abundancia_com]
    assert len(da_abundancia_sem) == 1, [s.rotulo for s in da_abundancia_sem]
    assert da_abundancia_sem[0].rotulo.startswith("ATLL"), da_abundancia_sem[0].rotulo

    # **O oráculo é a FRASE e não a existência de texto no eixo**, porque o valor anotado sobre
    # cada barra também é um `Text`: `assert not eixo.texts` reprovaria sobre a figura correta.
    def texto_de(painel: PainelDeGraficos) -> str:
        return " ".join(
            " ".join(
                t.get_text() for e in painel.figuras.canvas.figure.axes for t in e.texts
            ).split()
        )

    assert texto_de(com), "âncora: a figura completa não tem texto nenhum, nem as anotações"
    assert "sem abundância" not in texto_de(com), texto_de(com)
    assert "sem abundância típica: NoHTLV" in texto_de(sem), texto_de(sem)


def test_u8_com_ZERO_classes_a_figura_DECLARA_em_texto_e_nao_recusa(qtbot: QtBot) -> None:  # noqa: N802
    """**«Calculado e vazio» é ACHADO VÁLIDO e não população vazia** — R7, e é o que separa esta
    figura do violino.

    Uma matriz só de QCs produz zero classes clínicas, e isso é um fato sobre o desenho do estudo.
    R7 manda rotulá-lo — *"nunca espaço em branco"* —, e por isso a figura **desenha** um eixo com a
    causa escrita em vez de levantar `MatrizVaziaError`.

    **O par causal separa**: com classes há barras e nenhuma frase de ausência; com a lista vazia há
    **um** eixo, zero barras e a frase. E é **um** e não dois: com dois, a mesma frase sairia duas
    vezes, que se lê como figura duplicada.
    """
    com = _painel_da_composicao(qtbot, _com_composicao(91))
    sem = _painel_da_composicao(
        qtbot, _com_composicao(91, classes=[], n_injecoes_fora_da_populacao=68)
    )

    assert len(com.figuras.canvas.figure.axes) == 2, (
        "âncora: a figura com classes não tem dois eixos"
    )
    assert sum(len(series(e)) for e in com.figuras.canvas.figure.axes) == 4

    vazia = sem.figuras.canvas.figure
    assert sem.figuras.estado_visivel()["figura"] == "1 eixo(s)", (
        f"com zero classes a figura recusou em vez de declarar: {sem.figuras.estado_visivel()}"
    )
    assert len(vazia.axes) == 1, [e.get_title() for e in vazia.axes]
    assert sum(len(series(e)) for e in vazia.axes) == 0
    declarado = " ".join(" ".join(t.get_text() for e in vazia.axes for t in e.texts).split())
    assert "nenhuma injeção de estudo" in declarado, declarado


# **Os fixtures da ANOVA vieram com os quatro testes** — eles viviam no bloco da branch e
# seriam largados com ele. Ficam antes do cabeçalho, junto dos outros fixtures do arquivo.
ANOVA_DE_TESTE: Final[dict[str, Any]] = {
    "features": list(FEATURES_DO_VOLCANO),
    "statistic": [9.5, 7.1, 0.4, 0.2, None],
    "pvalue": [0.0002, 0.0005, 0.30, 0.80, None],
    "qvalue_fdr": [0.004, 0.008, 0.40, 0.90, None],
    "grupos": ["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"],
    "n_por_grupo": [13, 18, 14, 16],
    "metodo": "anova",
    "post_hoc_metodo": "tukey",
    "post_hoc_rota": "ran",
    "alfa_fdr": 0.05,
    "correcao": "Benjamini-Hochberg (FDR)",
    "min_por_grupo": 2,
    "matriz": "processada",
    "n_features_testadas": 5,
}

DESENHAVEIS_DA_ANOVA: Final = FEATURES_DO_VOLCANO[:4]


def _com_anova(seed: int, **troca: Any) -> dict[str, Any]:
    """O corpo de `/execucoes` com a raiz `anova` dentro, que é como ele chega na produção."""
    return {**_corpo(seed), "anova": {**ANOVA_DE_TESTE, **troca}}


# Os fixtures da correlação — as DUAS raízes, que vêm na mesma resposta.
#
# A matriz é simétrica com a diagonal a `1` porque é isso que uma correlação é; os valores fora da
# diagonal têm os dois sinais, para a paleta divergente exercitar as duas metades da escala.
CORRELACAO_DE_TESTE: Final[dict[str, Any]] = {
    "features": list(FEATURES_DO_VOLCANO[:4]),
    "valores": [
        [1.0, 0.82, -0.35, 0.10],
        [0.82, 1.0, -0.60, 0.05],
        [-0.35, -0.60, 1.0, -0.20],
        [0.10, 0.05, -0.20, 1.0],
    ],
    "metodo": "pearson",
    "matriz": "processada",
    "n_amostras": 61,
    "n_features_totais": 5942,
    "n_indefinidas": 2,
    "amostragem": "as 4 sorteadas de 5.942, com a semente 42",
}

# **Os dois sinais no coeficiente**, para as DUAS séries por sinal serem desenhadas: com um lado só
# a figura sairia com metade dos artistas que o construtor sabe pôr, e o teste da varredura de
# escala mediria uma figura que a produção não faz.
BUSCA_DE_TESTE: Final[dict[str, Any]] = {
    "features": list(FEATURES_DO_VOLCANO[:4]),
    "coeficiente": [1.0, 0.61, -0.44, -0.88],
    "referencia": FEATURES_DO_VOLCANO[0],
    "tipo_da_referencia": "feature",
    "perfil": [0.5, 1.5, 2.5, 3.5],
    "ordem_dos_grupos": ["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"],
    "metodo": "pearson",
    "matriz": "processada",
    "n_amostras": 61,
    "n_features_totais": 5942,
    "n_indefinidas": 1,
}


def _com_correlacao(seed: int, **troca: Any) -> dict[str, Any]:
    """O corpo de `/execucoes` com as DUAS raízes da correlação, como ele chega na produção."""
    return {
        **_corpo(seed),
        "correlacao": {**CORRELACAO_DE_TESTE, **troca},
        "busca_por_padrao": dict(BUSCA_DE_TESTE),
    }


# AS DUAS RAÍZES PARTICIONAIS. `k` é 3 e a grade é 2x3: nenhum dos dois é o valor de
# nascença do contrato (`kmeans_k` é 3 e `som_rows`/`som_cols` são 4), e a grade **não é quadrada**
# de propósito — numa quadrada, trocar linha por coluna é uma mutação que a figura absorve.
#
# **Os seis rótulos e as seis unidades cobrem os dois grupos**, para as duas séries por grupo serem
# desenhadas; e **o agrupamento 2 fica vazio e três unidades ficam vazias**, para o ramo de R12b do
# rodapé correr — uma grade cheia nunca o exercitaria.
KMEANS_DE_TESTE: Final[dict[str, Any]] = {
    "amostras": [f"S{i}" for i in range(6)],
    "rotulos": [0, 0, 0, 1, 1, 1],
    "grupos": ["ATLL", "ATLL", "ATLL", "NoHTLV", "NoHTLV", "NoHTLV"],
    "k": 3,
    "inercia": 225895.8,
    "semente": 1001697648,
    "n_iteracoes": 4,
    "matriz": "processada",
    "n_amostras": 6,
    "n_features_usadas": 5942,
    "n_features_totais": 5942,
    "amostragem": "todas as 5942 features utilizáveis, sem sorteio — a partição é sobre 6 amostras",
}

SOM_DE_TESTE: Final[dict[str, Any]] = {
    "amostras": [f"S{i}" for i in range(6)],
    "unidades": [0, 0, 1, 4, 5, 5],
    "grupos": ["ATLL", "ATLL", "ATLL", "NoHTLV", "NoHTLV", "NoHTLV"],
    "linhas": 2,
    "colunas": 3,
    "epocas": 100,
    "semente": 1172574093,
    "erro_de_quantizacao": 44.69,
    "matriz": "processada",
    "n_amostras": 6,
    "n_features_usadas": 5942,
    "n_features_totais": 5942,
    "amostragem": "todas as 5942 features utilizáveis, sem sorteio — o mapa é sobre 6 amostras",
}


def _com_agrupamento_particional(seed: int, **troca: Any) -> dict[str, Any]:
    """O corpo de `/execucoes` com as DUAS raízes particionais, como ele chega na produção."""
    return {
        **_corpo(seed),
        "kmeans": {**KMEANS_DE_TESTE, **troca},
        "som": dict(SOM_DE_TESTE),
    }


# AS DUAS RAÍZES DE CLASSIFICAÇÃO. Os dois `nome` são DIFERENTES de propósito: os dois
# construtores desenham o mesmo traço e o que os separa é a guarda de nome, então duas cargas com o
# mesmo rótulo fariam qualquer troca de montador passar despercebida.
#
# **O SVM tem uma classe com *recall* ZERO e fica ABAIXO do acaso**, e as duas coisas são o ramo de
# R12b: a classe sem *recall* não vira barra, e o veredito do rodapé sai como *"o modelo NÃO
# SEPARA"*. Uma carga com as quatro barras altas nunca exercitaria nenhum dos dois.
#
# **Os números são os MEDIDOS no `pos_hilic` real em 18/09/2026**, e não inventados — 61 amostras de
# estudo, 6.031 features utilizáveis de 6.035, quatro classes de 13/18/14/16, `classifier_folds=5` e
# 200 permutações. As contagens aqui somam 61.
_CLASSES_DE_TESTE: Final[list[str]] = ["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"]
_CONTAGENS_DE_TESTE: Final[list[int]] = [13, 18, 14, 16]

FLORESTA_DE_TESTE: Final[dict[str, Any]] = {
    "kind": "random_forest",
    "nome": "Random Forest",
    "grupos": list(_CLASSES_DE_TESTE),
    "contagens": list(_CONTAGENS_DE_TESTE),
    "recall": [0.23076923076923078, 0.5, 0.21428571428571427, 0.375],
    "balanced_accuracy": 0.3300137362637363,
    "accuracy": 0.3442622950819672,
    "macro_f1": 0.32684865444153366,
    "permutation_pvalue": 0.11940298507462686,
    "chance_level": 0.25,
    "confusao": [[3, 2, 4, 4], [3, 9, 2, 4], [3, 4, 3, 4], [2, 5, 3, 6]],
    "importancias": [0.4, 0.3, 0.2, 0.1],
    "importance_kind": "redução média de impureza (Gini), normalizada",
    "features": ["f0", "f1", "f2", "f3"],
    "cv_folds": 5,
    "n_permutations": 200,
    "semente": 1921827941,
    "matriz": "processada",
    "n_amostras": 61,
    "n_features_usadas": 6031,
    "n_features_totais": 6035,
    "amostragem": "todas as 6031 features utilizáveis, sem sorteio",
}

SVM_DE_TESTE: Final[dict[str, Any]] = {
    **FLORESTA_DE_TESTE,
    "kind": "svm",
    "nome": "SVM",
    "recall": [0.07692307692307693, 0.3333333333333333, 0.0, 0.1875],
    "balanced_accuracy": 0.14943910256410256,
    "accuracy": 0.16393442622950818,
    "macro_f1": 0.14978070175438596,
    "permutation_pvalue": 0.9701492537313433,
    "importance_kind": "magnitude média do peso linear entre os modelos um-contra-um",
    "semente": 979431272,
}


def _com_classificadores(seed: int, **troca: Any) -> dict[str, Any]:
    """O corpo de `/execucoes` com as DUAS raízes de classificação, como chega na produção."""
    return {
        **_corpo(seed),
        "random_forest": {**FLORESTA_DE_TESTE, **troca},
        "svm": dict(SVM_DE_TESTE),
    }


# ---------------------------------------------------------------------------
# A rede de correlação parcial desenha, e a raiz é PRÓPRIA
# ---------------------------------------------------------------------------

FEATURES_DA_REDE: Final[tuple[str, ...]] = ("r0", "r1", "r2", "r3", "r4")

REDE_DE_TESTE: Final[dict[str, Any]] = {
    "features": list(FEATURES_DA_REDE),
    # **Os DOIS sinais, e um vértice isolado.** Com um sinal só, metade dos artistas que o
    # construtor sabe pôr nunca é desenhada; sem isolado, o ramo do rodapé que declara o
    # isolamento nunca corre.
    "arestas": [
        {
            "feature_a": "r0",
            "feature_b": "r1",
            "correlacao_parcial": 0.62,
            "estatistica": 3.4,
            "pvalue": 0.0008,
            "qvalue_fdr": 0.021,
        },
        {
            "feature_a": "r1",
            "feature_b": "r2",
            "correlacao_parcial": -0.48,
            "estatistica": -2.9,
            "pvalue": 0.0041,
            "qvalue_fdr": 0.062,
        },
    ],
    "n_arestas": 2,
    "n_pares_testados": 10,
    "alfa": 0.05,
    "penalidade": 0.2431,
    "n_amostras_usadas": 61,
    "n_features_usadas": 5,
    "n_features_totais": 6035,
    "n_features_descartadas_constantes": 0,
    "n_features_descartadas_nao_finitas": 0,
    "metodo": "graphical lasso com correção de viés de Jankova e van de Geer (2015)",
    "amostragem": "as 5 sorteadas com semente 42",
}


def _com_rede(seed: int, **troca: Any) -> dict[str, Any]:
    """O corpo de `/execucoes` com a raiz `rede`, como chega na produção."""
    return {**_corpo(seed), "rede": {**REDE_DE_TESTE, **troca}}


def test_u21_o_painel_DESENHA_a_rede_da_raiz_PROPRIA(qtbot: QtBot) -> None:  # noqa: N802
    """**A raiz é própria e não uma leitura a mais de `correlacao`**, e o par causal está aqui.

    `AnalysisSpec.dspc` e `AnalysisSpec.correlation` são bandeiras separadas: a rede estima a
    **precisão** e o mapa estima a covariância, e pedir um sem o outro é o caso comum. Um painel que
    lesse a rede da raiz do mapa desenharia arestas que ninguém pediu — ou nenhuma, com o rótulo
    certo.

    **A outra ponta está na mesma função:** com o corpo a trazer **só** a correlação, o montador da
    rede recusa por população vazia e o painel di-lo.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = _com_rede(41)
    graficos.recebe_rede(corpo)
    assert graficos.figuras.define_execucao(_resposta(200, corpo)) is not None
    _escolhe(graficos, "figura_da_rede_de_correlacao_parcial")

    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"
    figura = graficos.figuras.canvas.figure
    rodape = " ".join(figura.texts[0].get_text().split())
    assert REDE_DE_TESTE["metodo"] in rodape, rodape
    assert "Corte no p CRU" in rodape, rodape
    assert confere_vista(figura.axes[0]) == []

    sem_rede = PainelDeGraficos()
    qtbot.addWidget(sem_rede)
    sem_rede.recebe_rede(_corpo(42))
    sem_rede.figuras.define_execucao(_resposta(200, _corpo(42)))
    _escolhe(sem_rede, "figura_da_rede_de_correlacao_parcial")
    assert sem_rede.figuras.estado_visivel()["figura"] != "1 eixo(s)", (
        "o painel desenhou uma rede sem raiz nenhuma — a recusa por população vazia é o que é "
        "verdade quando a família não foi pedida"
    )


def test_u21_o_montador_da_rede_devolve_o_vetor_de_NOMES_para_a_identificacao(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """R2 sem rótulo por feature: quem identifica o vértice é o clique, e ele precisa dos nomes.

    **É o primeiro montador desta aba que devolve o vetor por este motivo**, e a metade que o torna
    verdadeiro está do lado do construtor: a dispersão dos vértices é marcada com
    `GID_DAS_FEATURES`, e `PainelDeFigura.desenha` resolve o nome só nos artistas marcados. Devolver
    `()` aqui apagaria a única forma de dizer qual feature é qual nesta figura.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = _com_rede(43)
    graficos.recebe_rede(corpo)
    graficos.figuras.define_execucao(_resposta(200, corpo))
    _escolhe(graficos, "figura_da_rede_de_correlacao_parcial")

    assert graficos.figuras._nomes == FEATURES_DA_REDE, graficos.figuras._nomes
    assert graficos.figuras._nomeados, (
        "nenhum artista da rede está marcado com `GID_DAS_FEATURES`: o vetor de nomes chegou e não "
        "há a quem o aplicar, que é a identificação a existir só no papel"
    )


# ---------------------------------------------------------------------------
# A ANOVA de N grupos desenha, e a raiz é PRÓPRIA
# ---------------------------------------------------------------------------
#
# **Estes quatro vieram do merge, repontados de `PainelDeQualidade` para `PainelDeGraficos`.**
# Foram escritos contra a aba que hospedava figura antes do plano dos gráficos; o que muda é
# o painel, e nenhuma asserção foi tocada — se alguma passar a valer outra coisa, é achado e
# não ruído de merge.
#
# **Os outros DOZE testes daquele bloco não vieram, e a razão é que o alvo deles saiu:** eles
# exercitavam a exportação que o bloco das análises escreveu (`DPI_DE_PUBLICACAO`,
# `destino_por_dialogo`, `nome_sugerido`), e a implementação que ficou é a do plano dos
# gráficos, com 51 testes próprios neste arquivo. Portá-los seria medir duas vezes a mesma
# propriedade por dois vocabulários.


def test_u6a_o_painel_DESENHA_a_anova_da_raiz_PROPRIA_e_nao_da_do_volcano(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """**A raiz é própria, e a âncora é a população: as duas raízes têm grupos DIFERENTES.**

    `volcano` descreve o par declarado — dois grupos — e `anova` descreve os quatro da matriz. Um
    painel que lesse a ANOVA da raiz do volcano poria o `n` de dois grupos no rodapé de uma figura
    sobre quatro: número certo, população errada.

    **O par causal está na mesma função:** com a raiz `anova` a figura sai, e com o corpo a trazer
    **só** o volcano ela recusa por população vazia.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = {**_com_volcano(31), "anova": dict(ANOVA_DE_TESTE)}
    assert len(corpo["anova"]["grupos"]) != 2, (
        "âncora: a raiz da ANOVA tem dois grupos como o volcano, e a distinção deixaria de separar"
    )
    graficos.recebe_volcano(corpo)
    graficos.recebe_anova(corpo)
    assert graficos.figuras.define_execucao(_resposta(200, corpo)) is not None
    _escolhe(graficos, "figura_da_anova")

    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"
    assert graficos.figuras._nomes == DESENHAVEIS_DA_ANOVA, graficos.figuras._nomes
    figura = graficos.figuras.canvas.figure
    rodape = " ".join(figura.texts[0].get_text().split())
    for nome, n in zip(ANOVA_DE_TESTE["grupos"], ANOVA_DE_TESTE["n_por_grupo"], strict=True):
        assert f"{nome} (n = {n})" in rodape, rodape
    assert confere_vista(figura.axes[0]) == []

    # A outra ponta: sem a raiz `anova`, o painel recusa e diz que recusou.
    so_volcano = PainelDeGraficos()
    qtbot.addWidget(so_volcano)
    so_volcano.recebe_volcano(_com_volcano(32))
    so_volcano.recebe_anova(_com_volcano(32))
    so_volcano.figuras.define_execucao(_resposta(200, _com_volcano(32)))
    _escolhe(so_volcano, "figura_da_anova")
    assert so_volcano.figuras.estado_visivel()["figura"] != "1 eixo(s)"


# As desenháveis de cada figura univariada sobre `VOLCANO_DE_TESTE`. **São conjuntos com o MESMO
# tamanho e membros DIFERENTES, e é isso que as torna uma âncora:** a quarta feature não tem
# `log2fc` e a quinta não tem `p`. Uma asserção sobre a CONTAGEM passaria com os vetores trocados —
# foi o que aconteceu na primeira escrita deste teste, e o que a corrigiu foi comparar os nomes.
DESENHAVEIS_POR_FIGURA: Final[dict[str, tuple[str, ...]]] = {
    "figura_do_fold_change": FEATURES_DO_VOLCANO[:3] + FEATURES_DO_VOLCANO[4:],
    "figura_dos_testes_t": FEATURES_DO_VOLCANO[:4],
}


@pytest.mark.parametrize("funcao", sorted(DESENHAVEIS_POR_FIGURA))
def test_u6a_as_duas_univariadas_DESENHAM_da_raiz_do_volcano(  # noqa: N802
    qtbot: QtBot, funcao: str
) -> None:
    """**As três figuras univariadas leem a MESMA raiz, e cada uma perde features diferentes.**

    A âncora é a assimetria: o fold change perde a quarta (sem `log2fc`) e os testes t perdem a
    quinta (sem `p`). Um montador que lesse o vetor errado desenharia **o mesmo número** de pontos
    — as duas perdem uma — e um teste sobre a contagem não o veria; o que separa são os NOMES.
    """
    esperados = DESENHAVEIS_POR_FIGURA[funcao]
    assert set(DESENHAVEIS_POR_FIGURA["figura_do_fold_change"]) != set(
        DESENHAVEIS_POR_FIGURA["figura_dos_testes_t"]
    ), "âncora: as duas desenham a MESMA população, e a troca de vetores não separaria"

    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_volcano(_com_volcano(33))
    assert graficos.figuras.define_execucao(_resposta(200, _com_volcano(33))) is not None
    _escolhe(graficos, funcao)

    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"
    assert graficos.figuras._nomes == esperados, graficos.figuras._nomes
    figura = graficos.figuras.canvas.figure
    nuvem = next(s for s in series(figura.axes[0]) if s.rotulo == "features desenhadas")
    assert nuvem.x.size == len(esperados)
    assert confere_vista(figura.axes[0]) == []


def test_u6a_as_tres_figuras_da_mesma_execucao_declaram_a_MESMA_populacao(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """**R8b entre irmãs: o denominador que as três publicam é o mesmo número.**

    As três leem `n_features_testadas` de raízes diferentes — duas do volcano e uma da ANOVA —, e a
    rota deriva as duas do mesmo `len(feature)`. Se divergissem, duas abas da mesma execução
    diriam populações diferentes e nenhuma estaria obviamente errada.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = {**_com_volcano(34), "anova": dict(ANOVA_DE_TESTE)}
    graficos.recebe_volcano(corpo)
    graficos.recebe_anova(corpo)
    graficos.figuras.define_execucao(_resposta(200, corpo))

    esperado = VOLCANO_DE_TESTE["n_features_testadas"]
    assert esperado == ANOVA_DE_TESTE["n_features_testadas"], "âncora: as duas raízes já divergem"
    for funcao in (
        "figura_do_volcano",
        "figura_do_fold_change",
        "figura_dos_testes_t",
        "figura_da_anova",
    ):
        _escolhe(graficos, funcao)
        rodape = " ".join(graficos.figuras.canvas.figure.texts[0].get_text().split())
        assert f"de {esperado} features" in rodape, (funcao, rodape)


def test_u6a_recebe_anova_corre_para_TODA_resposta_e_apaga_a_anterior(qtbot: QtBot) -> None:  # noqa: N802
    """**O par causal do apagamento: um `202` tem de limpar os vetores da execução anterior.**

    `recebe_anova` que só corresse no `200` deixaria a ANOVA anterior no atributo, e a figura sairia
    com o rótulo da execução nova sobre o conteúdo da velha.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_anova(_com_anova(35))
    assert graficos._anova is not None, "âncora: a primeira resposta não guardou nada"

    graficos.recebe_anova({"estado": "em_curso"})
    assert graficos._anova is None, (
        "a ANOVA da execução anterior sobreviveu a um corpo sem a raiz: a figura sairia com o "
        "rótulo da nova e o conteúdo da velha"
    )


# ---------------------------------------------------------------------------
# As figuras de cada etapa do passo a passo
# ---------------------------------------------------------------------------


# As entradas das etapas feitas nestes testes — a cadeia que a Análise sobre a normalizada
# reproduz: a Correção sobre a original e a Normalização sobre a corrigida. A frase de figura
# desatualizada segue-as.
_ENTRADAS: Final[dict[int, Any]] = {1: "original", 2: "corrigida"}


def _corpo_da_etapa(etapa: int) -> dict[str, Any]:
    """O corpo de uma prévia na forma de `EtapaDaCorrecaoResposta`, com a raiz do violino.

    **Sintético e legítimo pelo argumento de `_matriz`**: o que se afirma é a regra de validade das
    figuras, e ela não depende do dado. A etapa 2 traz também as distribuições, como a rota.
    """
    corpo: dict[str, Any] = {
        "correcao": "none",
        "chave_da_etapa": f"etapa-{etapa}",
        "rsd_por_feature": dict(RSD_POR_FEATURE_DE_TESTE),
    }
    if etapa == 2:
        corpo["distribuicoes"] = dict(RESUMO_DE_TESTE)
    return corpo


def _painel_com_a_etapa(qtbot: QtBot, etapa: int) -> PainelDeGraficos:
    """O painel com a etapa feita; a 2 sobre a Correção guardada, que é a entrada em que o corpo
    dela escreve também as figuras da etapa 1."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_etapa(
        etapa, _corpo_da_etapa(etapa), entrada="corrigida" if etapa == 2 else "original"
    )
    graficos.define_frescor_das_etapas({1: True, 2: etapa == 2}, entradas=_ENTRADAS)
    return graficos


def test_d550_a_figura_da_etapa_desenha_sem_execucao_e_sai_quando_a_etapa_desatualiza(
    qtbot: QtBot,
) -> None:
    """R5 e R6: o violino da etapa 1 aparece sem execução, e sai com a frase de parâmetros antigos.

    **Par causal:** o mesmo painel, com a etapa em dia e desatualizada; só o primeiro desenha, e
    voltar a ficar em dia — os parâmetros voltaram aos da etapa — redesenha sem pedido novo. E a
    figura das análises continua à espera da execução.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    _escolhe(graficos, _FUNCAO_DO_VIOLINO)
    # Sem diagnóstico e sem etapa o violino espera o DIAGNÓSTICO, que desenha a matriz
    # ingerida; antes a frase era a de esperar a execução. O que o par causal mede não muda: nada
    # desenhado antes da etapa, e o violino da etapa depois dela.
    assert graficos.figuras.reavalia() == texto_do_estado(EstadoDaFigura.SEM_DIAGNOSTICO)
    assert graficos.figuras.estado_visivel()["figura"] == ""

    graficos.recebe_etapa(1, _corpo_da_etapa(1), entrada="original")
    graficos.define_frescor_das_etapas({1: True, 2: False}, entradas=_ENTRADAS)
    assert graficos.figuras.estado_visivel()["figura"] != "", "a etapa 1 nao desenhou o violino"

    graficos.define_frescor_das_etapas({1: False, 2: False}, entradas=_ENTRADAS)
    assert graficos.figuras.estado_visivel()["figura"] == ""
    assert graficos.figuras.repouso.text() == frase_da_etapa_desatualizada(1, 1)
    assert "refaça" in texto_do_estado(EstadoDaFigura.ETAPA_DESATUALIZADA)

    graficos.define_frescor_das_etapas({1: True, 2: False}, entradas=_ENTRADAS)
    assert graficos.figuras.estado_visivel()["figura"] != "", "em dia de novo e nao redesenhou"

    _escolhe(graficos, "figura_do_volcano")
    assert graficos.figuras.repouso.text() == texto_do_estado(EstadoDaFigura.SEM_EXECUCAO)


def test_d550_as_figuras_da_etapa_2_tem_frescor_proprio(qtbot: QtBot) -> None:
    """A etapa 2 sobre a Correção guardada desatualiza sozinha: mudar a imputação tira as caixas e
    deixa o violino, que é da correção.

    > Era *«a etapa 2 é cumulativa»*: ela escrevia sempre as figuras da etapa 1. Desde que cada
    > etapa é dona das próprias figuras, só a entrada `corrigida` as escreve — ver
    > `test_a_normalizacao_sobre_a_original_nao_escreve_as_figuras_da_correcao`."""
    graficos = _painel_com_a_etapa(qtbot, 2)
    _escolhe(graficos, "figura_das_caixas_por_feature")
    assert graficos.figuras.estado_visivel()["figura"] != "", "a etapa 2 nao desenhou as caixas"

    graficos.define_frescor_das_etapas({1: True, 2: False}, entradas=_ENTRADAS)
    assert graficos.figuras.repouso.text() == frase_da_etapa_desatualizada(2, 2)
    _escolhe(graficos, _FUNCAO_DO_VIOLINO)
    assert graficos.figuras.estado_visivel()["figura"] != "", (
        "a etapa 2 desatualizada tirou o violino, que e da etapa 1 e continua em dia"
    )


def test_d550_o_202_e_o_erro_da_execucao_nao_tiram_a_figura_da_etapa_e_o_200_a_substitui(
    qtbot: QtBot,
) -> None:
    """R7, na ordem em que a janela chama: os `recebe_*` e depois `define_execucao`.

    O `202` e o erro de uma execução que corresponde à etapa não tiram a figura dela; o `200`
    reescreve os dados. **E a etapa continua dona deles** — cada etapa é dona das próprias figuras,
    e a execução que a reproduz não a destitui —: o frescor dela continua a decidir. Até as etapas
    independentes, o `200` passava a figura à execução, e o frescor falso já não a tirava.
    """
    graficos = _painel_com_a_etapa(qtbot, 1)
    _escolhe(graficos, _FUNCAO_DO_VIOLINO)
    assert graficos.figuras.estado_visivel()["figura"] != "", "ancora: a etapa nao desenhou"
    # A ancora do `not in` la embaixo, na mesma funcao: sem ela, um painel que nunca guardasse a
    # fonte da etapa passaria a asserçao por vacuidade (refino-review de 15/10/2026).
    assert graficos._fonte_da_etapa.get(1) == "etapa", "ancora: a figura nao e da etapa"
    fonte = graficos.figuras._fonte
    assert fonte is not None
    chamadas: list[str] = []

    def contando(construtor: Any) -> Any:
        chamadas.append(construtor.funcao)
        return fonte(construtor)

    graficos.figuras._fonte = contando
    for codigo, corpo in ((202, {"identificador": "x", "total": 5}), (500, {"detail": "falhou"})):
        graficos.recebe_rsd_por_feature(corpo, corresponde=True)
        graficos.recebe_pca_do_par(corpo, corresponde=True)
        graficos.recebe_composicao(corpo, corresponde=True)
        graficos.recebe_distribuicoes(corpo, corresponde=True)
        graficos.define_execucao(_resposta(codigo, corpo))
        assert graficos.figuras.estado_visivel()["figura"] != "", (
            f"o {codigo} da execucao tirou a figura da etapa antes de o resultado chegar"
        )
    assert chamadas == [], f"o 202 e o erro refizeram a figura da etapa: {chamadas}"

    execucao = _com_rsd_por_feature(81)
    graficos.recebe_rsd_por_feature(execucao, corresponde=True)
    graficos.define_execucao(_resposta(200, execucao))
    assert graficos._rsd_por_feature == execucao["rsd_por_feature"]
    assert graficos._fonte_da_etapa.get(1) == "etapa", "a execucao destituiu a etapa dona"
    assert chamadas == [_FUNCAO_DO_VIOLINO], (
        f"os dados reescritos nao refizeram a figura da etapa uma vez: {chamadas}"
    )
    assert graficos.figuras.estado_visivel()["figura"] != "", "o 200 tirou a figura da etapa"
    graficos.define_frescor_das_etapas({1: False, 2: False}, entradas=_ENTRADAS)
    assert graficos.figuras.repouso.text() == frase_da_etapa_desatualizada(1, 1), (
        "depois do 200 a figura deixou de seguir o frescor da etapa"
    )


def test_d551_o_202_e_as_releituras_da_execucao_nao_reconstroem_a_figura_da_etapa(
    qtbot: QtBot,
) -> None:
    """#2 da revisão de 15/10/2026: cada não-200 de `/execucoes` reconstruía a figura própria.

    A chave de todo não-200 é irrepetível, e a figura da etapa (ou do diagnóstico) era refeita a
    cada `202` e a cada releitura do progresso — de segundo a segundo durante o SERRF —, apagando o
    zoom e a anotação. **Par causal:** os `202` não chamam a fonte; a reavaliação explícita, que é
    o caminho de quem muda os dados, chama uma vez.
    """
    graficos = _painel_com_a_etapa(qtbot, 1)
    _escolhe(graficos, _FUNCAO_DO_VIOLINO)
    figuras = graficos.figuras
    assert figuras.estado_visivel()["figura"] != "", "ancora: a etapa nao desenhou"
    fonte = figuras._fonte
    assert fonte is not None
    chamadas: list[str] = []

    def contando(construtor: Any) -> Any:
        chamadas.append(construtor.funcao)
        return fonte(construtor)

    figuras._fonte = contando
    for _ in range(5):
        figuras.define_execucao(_resposta(202, {"identificador": "x", "total": 5}))
    assert chamadas == [], f"os 202 reconstruiram a figura da etapa {len(chamadas)} vez(es)"
    assert figuras.estado_visivel()["figura"] != "", "o 202 tirou a figura da etapa"

    figuras.reavalia()
    assert chamadas == [_FUNCAO_DO_VIOLINO], "a reavaliacao explicita tinha de reconstruir"
    graficos.define_frescor_das_etapas({1: False, 2: False}, entradas=_ENTRADAS)
    figuras.define_execucao(_resposta(202, {"identificador": "x", "total": 5}))
    assert figuras.repouso.text() == frase_da_etapa_desatualizada(1, 1)


def test_d552_a_frase_da_figura_desatualizada_nomeia_a_etapa_a_refazer(qtbot: QtBot) -> None:
    """A frase diz QUAL etapa refazer.

    A frase do estado dizia *«refaça essa etapa»* sem dizer qual. **Par causal:** a mesma figura
    da etapa 2, com só a 2 fora de dia e com a 1 também fora; a frase pede a 2 no primeiro caso e
    a 1 no segundo, porque na cadeia padrão de `etapa_a_refazer` a figura da 2 depende da 1.
    """
    graficos = _painel_com_a_etapa(qtbot, 2)
    _escolhe(graficos, "figura_das_caixas_por_feature")
    assert graficos.figuras.estado_visivel()["figura"] != "", "ancora: a etapa 2 nao desenhou"

    graficos.define_frescor_das_etapas({1: True, 2: False}, entradas=_ENTRADAS)
    so_a_2 = graficos.figuras.repouso.text()
    assert so_a_2 == frase_da_etapa_desatualizada(2, 2)
    assert "refaça a etapa 2, normalização" in so_a_2, so_a_2

    graficos.define_frescor_das_etapas({1: False, 2: False}, entradas=_ENTRADAS)
    as_duas = graficos.figuras.repouso.text()
    assert as_duas == frase_da_etapa_desatualizada(2, 1)
    assert "refaça a etapa 1, correção" in as_duas, as_duas
    assert so_a_2 != as_duas
    assert graficos.figuras.estado_visivel()["figura"] == "", "a figura desatualizada ficou"


def test_a_frase_da_figura_desatualizada_segue_a_entrada_usada(qtbot: QtBot) -> None:
    """A etapa a refazer é a mais a montante fora de dia na cadeia da ENTRADA usada, e não a ordem
    1-2-3. **Par causal:** as mesmas duas etapas fora de dia, a Normalização feita sobre a
    `corrigida` e sobre a `original`; a frase das caixas pede a Correção no primeiro caso e a
    própria Normalização no segundo — que não depende da Correção."""
    frases = {}
    for entrada in ("corrigida", "original"):
        graficos = PainelDeGraficos()
        qtbot.addWidget(graficos)
        graficos.recebe_etapa(2, _corpo_da_etapa(2), entrada=entrada)  # type: ignore[arg-type]
        _escolhe(graficos, "figura_das_caixas_por_feature")
        graficos.define_frescor_das_etapas(
            {1: False, 2: False},
            entradas={1: "original", 2: entrada},  # type: ignore[dict-item]
        )
        frases[entrada] = graficos.figuras.repouso.text()
    assert frases == {
        "corrigida": frase_da_etapa_desatualizada(2, 1),
        "original": frase_da_etapa_desatualizada(2, 2),
    }


# ---------------------------------------------------------------------------
# Os gráficos da etapa na aba da etapa, e cada etapa dona das próprias figuras
# ---------------------------------------------------------------------------
#
# Os gráficos da etapa aparecem também na aba da etapa, além da aba «Gráficos». A vista de cada
# aba de etapa é um segundo
# `PainelDeFigura` sobre o mesmo dono de estado; ela adia a construção enquanto está escondida.


def _vistas(graficos: PainelDeGraficos) -> dict[int, PainelDeFigura]:
    return {etapa: vista.figuras for etapa, vista in graficos.vistas_das_etapas.items()}


# Os hospedeiros das vistas mostradas nos testes. **Guardados aqui** porque `qtbot.addWidget` não
# segura o objeto Python: sem referência, o hospedeiro sem pai é coletado e leva a vista com ele
# («Internal C++ object already deleted», visto na primeira corrida).
_HOSPEDEIROS: list[QWidget] = []


def _a_vista(qtbot: QtBot, graficos: PainelDeGraficos, etapa: int) -> PainelDeFigura:
    """A vista da etapa num hospedeiro à vista — a aba da etapa à frente."""
    vista = graficos.vistas_das_etapas[etapa]
    hospedeiro = QWidget()
    _HOSPEDEIROS.append(hospedeiro)
    qtbot.addWidget(hospedeiro, before_close_func=lambda _w: _HOSPEDEIROS.remove(hospedeiro))
    QVBoxLayout(hospedeiro).addWidget(vista)
    vista.show()
    hospedeiro.resize(LARGURA_DO_PAINEL, ALTURA_DO_PAINEL)
    hospedeiro.show()
    return vista.figuras


def _listadas(painel: PainelDeFigura) -> set[str]:
    return {construtor.funcao for _rotulo, construtor in painel._entradas}


def _construtor(funcao: str) -> ConstrutorDeFigura:
    return next(c for c in CONSTRUTORES if c.funcao == funcao)


def _motivo_de(botao: Any) -> str:
    """A frase à vista ao lado de um botão: o rótulo cujo `buddy` é ele, como o portão o acha."""
    pai = botao.parentWidget()
    rotulos = [r for r in pai.findChildren(QLabel) if r.buddy() is botao]
    return " ".join(r.text() for r in rotulos if not r.isHidden())


def _corpo_da_correcao(corretor: str) -> dict[str, Any]:
    """O corpo da etapa 1 com o par de PCAs e o violino de um corretor — o nome é o `xlab`."""
    return {
        **_corpo_da_etapa(1),
        "correcao": corretor,
        "pcas_da_correcao": {**PAR_DE_PCAS_DE_TESTE, "corretor": corretor},
        "rsd_por_feature": {**RSD_POR_FEATURE_DE_TESTE, "corretor": corretor},
    }


def _corretores_na_figura(painel: PainelDeFigura) -> str:
    return " ".join(eixo.get_xlabel() for eixo in painel.canvas.figure.axes)


def test_cada_vista_de_etapa_lista_so_as_figuras_dela(qtbot: QtBot) -> None:
    """A Correção tem as dela e a curva do LOESS; a Normalização, as quatro distribuições; a
    Análise, as das análises. As do diagnóstico só ficam em «Gráficos» — e nenhuma figura fica de
    fora das quatro, nem aparece em duas vistas de etapa."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    listadas = {etapa: _listadas(painel) for etapa, painel in _vistas(graficos).items()}
    assert sorted(listadas) == [1, 2, 3], "âncora: as três vistas"
    da_etapa = {
        etapa: {f for f, dona in ETAPA_DA_FIGURA.items() if dona == etapa} for etapa in (1, 2)
    }
    assert listadas[1] == da_etapa[1] | {FIGURA_DA_CURVA_DO_LOESS}
    assert listadas[2] == da_etapa[2] and len(listadas[2]) == 4
    assert "figura_do_volcano" in listadas[3], "âncora: a Análise sem o volcano"
    assert "figura_do_volcano" not in listadas[1], "a vista da Correção lista figura de análise"
    assert not listadas[3] & (set(ETAPA_DA_FIGURA) | FIGURAS_DO_DIAGNOSTICO)
    todas = {c.funcao for c in CONSTRUTORES}
    assert sum(len(f) for f in listadas.values()) + len(FIGURAS_DO_DIAGNOSTICO) == len(todas)
    assert listadas[1] | listadas[2] | listadas[3] | FIGURAS_DO_DIAGNOSTICO == todas
    assert _listadas(graficos.figuras) == todas, "«Gráficos» deixou de listar todas"


def test_a_vista_da_correcao_mostra_a_figura_e_a_frase_de_graficos(qtbot: QtBot) -> None:
    """Depois da etapa 1 a vista da Correção desenha o painel da correção, e «Gráficos» o mesmo; com
    a Correção desatualizada, as duas dizem a mesma frase. Cada uma tem a sua `Figure`: a mesma não
    pode ter dois canvas."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    vista = _a_vista(qtbot, graficos, 1)
    assert vista.funcao_escolhida() == _FUNCAO_DA_CORRECAO, "âncora: a vista abre noutra figura"
    graficos.recebe_etapa(1, _corpo_da_correcao("loess"), entrada="original")
    graficos.define_frescor_das_etapas({1: True, 2: False}, entradas=_ENTRADAS)
    _escolhe(graficos, _FUNCAO_DA_CORRECAO)
    for painel in (vista, graficos.figuras):
        assert painel.estado_visivel()["figura"] != "", "a etapa 1 não desenhou o painel"
        assert "loess" in _corretores_na_figura(painel)
    assert vista.canvas.figure is not graficos.figuras.canvas.figure

    graficos.define_frescor_das_etapas({1: False, 2: False}, entradas=_ENTRADAS)
    assert vista.repouso.text() == graficos.figuras.repouso.text()
    assert vista.repouso.text() == frase_da_etapa_desatualizada(1, 1)
    assert vista.estado_visivel()["figura"] == "", "a vista ficou com a figura desatualizada"


def test_a_vista_escondida_nao_constroi_e_constroi_ao_aparecer(qtbot: QtBot) -> None:
    """Com a aba da Normalização escondida, o corpo da etapa 2 não constrói figura na vista dela;
    mostrar a aba constrói. Voltar sem nada mudado não reconstrói, e com o estado mudado sim."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    painel = _a_vista(qtbot, graficos, 2)
    vista = graficos.vistas_das_etapas[2]
    vista.hide()
    fonte = painel._fonte
    assert fonte is not None
    chamadas: list[str] = []

    def contando(construtor: Any) -> Any:
        chamadas.append(construtor.funcao)
        return fonte(construtor)

    painel._fonte = contando
    graficos.recebe_etapa(2, _corpo_da_etapa(2), entrada="corrigida")
    graficos.define_frescor_das_etapas({1: True, 2: True}, entradas=_ENTRADAS)
    assert chamadas == [], f"a vista escondida construiu {chamadas}"

    vista.show()
    assert chamadas == ["figura_das_caixas_por_feature"], chamadas
    assert painel.estado_visivel()["figura"] != "", "mostrar a aba não desenhou"

    vista.hide()
    vista.show()
    assert len(chamadas) == 1, "voltar à aba sem nada mudado reconstruiu a figura"

    vista.hide()
    graficos.recebe_etapa(2, _corpo_da_etapa(2), entrada="corrigida")
    assert len(chamadas) == 1, "a vista escondida construiu"
    vista.show()
    assert len(chamadas) == 2, "voltar à aba com o estado mudado não reconstruiu"


def test_a_normalizacao_sobre_a_original_nao_escreve_as_figuras_da_correcao(qtbot: QtBot) -> None:
    """Corrigir com `loess` e normalizar sobre a `original` mantém na vista da Correção o painel do
    `loess`. **Par causal:** a mesma Normalização sobre a `corrigida` escreve as figuras da etapa 1
    com o corpo dela — o corretor `none` da planta, que difere do `loess` da etapa 1."""
    desfechos = {}
    for entrada in ("original", "corrigida"):
        graficos = PainelDeGraficos()
        qtbot.addWidget(graficos)
        vista = _a_vista(qtbot, graficos, 1)
        graficos.recebe_etapa(1, _corpo_da_correcao("loess"), entrada="original")
        normalizacao = {**_corpo_da_correcao("none"), "distribuicoes": dict(RESUMO_DE_TESTE)}
        graficos.recebe_etapa(2, normalizacao, entrada=entrada)
        graficos.define_frescor_das_etapas({1: True, 2: True}, entradas=_ENTRADAS)
        assert graficos._distribuicoes == normalizacao["distribuicoes"], entrada
        assert vista.estado_visivel()["figura"] != "", entrada
        desfechos[entrada] = (graficos._pcas_da_correcao["corretor"], _corretores_na_figura(vista))
    assert desfechos["original"][0] == "loess"
    assert "loess" in desfechos["original"][1]
    assert desfechos["corrigida"][0] == "none", "a planta não separou as duas entradas"


def test_normalizar_sem_correcao_deixa_a_vista_da_correcao_no_estado_vazio(qtbot: QtBot) -> None:
    """Sem Correção feita, a Normalização sobre a `original` deixa a vista da Correção no estado
    vazio dela — a frase de esperar a etapa —, e não em «etapa desatualizada». **Par causal:** a
    mesma Normalização dita `corrigida` escreveria as figuras da etapa 1, e elas ficariam fora de
    dia sem Correção nenhuma feita."""
    desfechos = {}
    for entrada in ("original", "corrigida"):
        graficos = PainelDeGraficos()
        qtbot.addWidget(graficos)
        vista = _a_vista(qtbot, graficos, 1)
        graficos.recebe_etapa(2, _corpo_da_etapa(2), entrada=entrada)
        graficos.define_frescor_das_etapas({1: False, 2: True}, entradas=_ENTRADAS)
        assert vista.funcao_escolhida() == _FUNCAO_DA_CORRECAO
        desfechos[entrada] = (
            vista.repouso.text(),
            graficos._validade_propria(_construtor(_FUNCAO_DA_CORRECAO)),
        )
    assert desfechos["original"] == (frase_da_etapa_sem_resultado(1), EstadoDaFigura.SEM_EXECUCAO)
    assert desfechos["corrigida"] == (
        frase_da_etapa_desatualizada(1, 1),
        EstadoDaFigura.ETAPA_DESATUALIZADA,
    )


def _execucao_completa(seed: int, corretor: str) -> dict[str, Any]:
    """Um `200` de `/execucoes` com as quatro raízes das figuras das etapas 1 e 2."""
    return {
        **_corpo(seed),
        "pcas_da_correcao": {**PAR_DE_PCAS_DE_TESTE, "corretor": corretor},
        "rsd_por_feature": {**RSD_POR_FEATURE_DE_TESTE, "corretor": corretor},
        "composicao_por_classe": dict(COMPOSICAO_DE_TESTE),
        "distribuicoes": {**RESUMO_DE_TESTE, "medias_por_feature_antes": [7.0, 8.0, 9.0, 10.0]},
    }


def _recebe_a_execucao(graficos: PainelDeGraficos, corpo: Any, *, corresponde: bool) -> None:
    """Os quatro `recebe_*` das figuras das etapas, na ordem da janela, e então a chave."""
    graficos.recebe_distribuicoes(corpo, corresponde=corresponde)
    graficos.recebe_pca_do_par(corpo, corresponde=corresponde)
    graficos.recebe_rsd_por_feature(corpo, corresponde=corresponde)
    graficos.recebe_composicao(corpo, corresponde=corresponde)
    codigo = 200 if isinstance(corpo, dict) and "distribuicoes" in corpo else 202
    graficos.define_execucao(_resposta(codigo, corpo))


def test_analisar_sobre_a_original_nao_muda_as_vistas_da_correcao_e_da_normalizacao(
    qtbot: QtBot,
) -> None:
    """Uma execução que não corresponde às etapas guardadas — a Análise sobre a `original` — não
    escreve os dados delas, nem no `202` nem no `200`, e as vistas continuam a desenhá-las.

    **Par causal:** o mesmo `200` dito correspondente escreve — e a etapa continua dona.
    """
    desfechos = {}
    for corresponde in (False, True):
        graficos = PainelDeGraficos()
        qtbot.addWidget(graficos)
        correcao, normalizacao = _a_vista(qtbot, graficos, 1), _a_vista(qtbot, graficos, 2)
        graficos.recebe_etapa(1, _corpo_da_correcao("loess"), entrada="original")
        sobre_a_correcao = {**_corpo_da_correcao("loess"), "distribuicoes": dict(RESUMO_DE_TESTE)}
        graficos.recebe_etapa(2, sobre_a_correcao, entrada="corrigida")
        graficos.define_frescor_das_etapas({1: True, 2: True}, entradas=_ENTRADAS)
        antes = (graficos._pcas_da_correcao, graficos._distribuicoes)
        assert antes[0]["corretor"] == "loess", "âncora: a etapa 1 não guardou o par"
        assert correcao.estado_visivel()["figura"] and normalizacao.estado_visivel()["figura"]

        _recebe_a_execucao(graficos, {"identificador": "x", "total": 5}, corresponde=corresponde)
        execucao = _execucao_completa(91, "none")
        _recebe_a_execucao(graficos, execucao, corresponde=corresponde)
        depois = (graficos._pcas_da_correcao, graficos._distribuicoes)
        desfechos[corresponde] = (depois == antes, dict(graficos._fonte_da_etapa))
        if not corresponde:
            assert "loess" in _corretores_na_figura(correcao)
            assert normalizacao.estado_visivel()["figura"] != ""
            graficos.define_frescor_das_etapas({1: False, 2: False}, entradas=_ENTRADAS)
            assert correcao.repouso.text() == frase_da_etapa_desatualizada(1, 1), (
                "a figura da Correção deixou de seguir o frescor da etapa"
            )
    assert desfechos[False] == (True, {1: "etapa", 2: "etapa"})
    assert desfechos[True] == (False, {1: "etapa", 2: "etapa"})


def test_a_execucao_que_nao_corresponde_entrega_a_etapa_os_dados_da_que_correspondia(
    qtbot: QtBot,
) -> None:
    """Depois do pipeline inteiro — que corresponde às etapas e as escreve —, uma execução que não
    corresponde não as reescreve, e a figura passa ao frescor da etapa em vez de seguir a chave
    dela. **Par causal:** sem a passagem, a figura de etapa seguiria a chave da execução nova e
    desenharia sob ela os dados do pipeline."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    pipeline = _execucao_completa(92, "loess")
    _recebe_a_execucao(graficos, pipeline, corresponde=True)
    _escolhe(graficos, _FUNCAO_DA_CORRECAO)
    assert graficos._fonte_da_etapa == {1: "execucao", 2: "execucao"}
    assert graficos.figuras.estado_visivel()["figura"] != "", "âncora: o pipeline não desenhou"

    graficos.define_frescor_das_etapas({1: True, 2: True}, entradas=_ENTRADAS)
    _recebe_a_execucao(graficos, _execucao_completa(93, "none"), corresponde=False)
    assert graficos._fonte_da_etapa == {1: "etapa", 2: "etapa"}
    assert graficos._pcas_da_correcao == pipeline["pcas_da_correcao"]
    assert "loess" in _corretores_na_figura(graficos.figuras)
    graficos.define_frescor_das_etapas({1: False, 2: True}, entradas=_ENTRADAS)
    assert graficos.figuras.repouso.text() == frase_da_etapa_desatualizada(1, 1)


def test_sem_dono_a_figura_da_etapa_espera_a_etapa_e_a_coorte_vem_do_corpo(qtbot: QtBot) -> None:
    """Sem etapa feita, uma execução que não corresponde não dá dono às figuras das etapas: elas
    dizem que esperam a etapa — e não «população vazia» sob a chave dela —, o violino continua o da
    ingerida, e a coorte da tabela dos estilos vem do corpo, porque é do estudo."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    assert graficos.tabela_dos_grupos.seletores == {}, "âncora: a tabela nasceu com grupos"
    _recebe_a_execucao(graficos, _execucao_completa(94, "none"), corresponde=False)
    _escolhe(graficos, "figura_das_caixas_por_feature")
    assert graficos.figuras.repouso.text() == frase_da_etapa_sem_resultado(2)
    assert graficos.figuras.estado_visivel()["figura"] == ""
    assert graficos._fonte_da_etapa == {}
    assert not graficos._violino_corrigido
    grupos = {classe["grupo"] for classe in COMPOSICAO_DE_TESTE["classes"]}
    assert set(graficos.tabela_dos_grupos.seletores) == grupos


@pytest.mark.parametrize(
    ("etapa", "entrada"), [(1, "corrigida"), (2, "normalizada"), (1, "qualquer")]
)
def test_recebe_etapa_recusa_entrada_que_nao_e_da_etapa(
    qtbot: QtBot, etapa: int, entrada: str
) -> None:
    """Uma entrada fora das da etapa levanta, e não cai no ramo que não escreve a correção.
    **Par:** as entradas válidas passam."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    with pytest.raises(ValueError, match="não é entrada da etapa"):
        graficos.recebe_etapa(etapa, _corpo_da_etapa(etapa), entrada=entrada)  # type: ignore[arg-type]
    for valida in ("original", "corrigida") if etapa == 2 else ("original",):
        graficos.recebe_etapa(etapa, _corpo_da_etapa(etapa), entrada=valida)  # type: ignore[arg-type]


def test_os_metodos_do_painel_alcancam_as_quatro_vistas(qtbot: QtBot) -> None:
    """A chave da execução, o esquecê-la e a escolha de figura chegam a «Gráficos» e às três vistas.
    A vista que não lista a figura escolhida fica como estava."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    paineis = [graficos.figuras, *_vistas(graficos).values()]
    assert len(paineis) == 4

    graficos.define_execucao(_resposta(200, _corpo(95)))
    assert {painel._chave for painel in paineis} == {_chave(95)}
    graficos.esquece_a_execucao()
    assert {painel._chave for painel in paineis} == {CHAVE_DE_REPOUSO}

    antes = {etapa: painel.funcao_escolhida() for etapa, painel in _vistas(graficos).items()}
    assert antes[3] != "figura_da_anova", "âncora: a vista da Análise já estava na ANOVA"
    assert graficos.seleciona("figura_da_anova")
    assert graficos.figuras.funcao_escolhida() == "figura_da_anova"
    assert _vistas(graficos)[3].funcao_escolhida() == "figura_da_anova"
    assert {etapa: _vistas(graficos)[etapa].funcao_escolhida() for etapa in (1, 2)} == {
        etapa: antes[etapa] for etapa in (1, 2)
    }


def test_os_botoes_das_figuras_desligam_com_a_frase_do_que_falta(qtbot: QtBot) -> None:
    """«Voltar à vista original», «Salvar figura…» e «Exportar esta figura em Gráficos» desligam
    sem figura na tela, cada um com a frase ao lado; com figura, ligam e a frase some."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    vista = graficos.vistas_das_etapas[1]
    _a_vista(qtbot, graficos, 1)
    botoes = {
        "voltar": (graficos.figuras.botao_da_vista, MOTIVO_SEM_VISTA),
        "salvar": (graficos.botao_de_exportar, MOTIVO_DA_EXPORTACAO_DESLIGADA),
        "voltar na vista": (vista.figuras.botao_da_vista, MOTIVO_SEM_VISTA),
        "levar aos gráficos": (vista.botao_de_exportar, MOTIVO_DA_IDA_AOS_GRAFICOS_DESLIGADA),
    }
    assert vista.botao_de_exportar.text() == ROTULO_DA_IDA_AOS_GRAFICOS
    for nome, (botao, motivo) in botoes.items():
        assert not botao.isEnabled(), f"{nome}: ligado sem figura"
        assert _motivo_de(botao) == motivo, nome

    graficos.recebe_etapa(1, _corpo_da_correcao("loess"), entrada="original")
    graficos.define_frescor_das_etapas({1: True, 2: False}, entradas=_ENTRADAS)
    _escolhe(graficos, _FUNCAO_DA_CORRECAO)
    for nome, (botao, _motivo) in botoes.items():
        assert botao.isEnabled(), f"{nome}: desligado com a figura na tela"
        assert _motivo_de(botao) == "", f"{nome}: a frase ficou ao lado do botão ligado"

    pedidos: list[str] = []
    vista.exporta_em_graficos.connect(pedidos.append)
    vista.botao_de_exportar.click()
    assert pedidos == [_FUNCAO_DA_CORRECAO]


# ---------------------------------------------------------------------------
# A cor e o símbolo de cada grupo, escolhidos na aba «Gráficos»
# ---------------------------------------------------------------------------
#
# A coorte vem da composição por classe, que toda resposta `200` de `/execucoes` traz; os nomes
# dos grupos aqui são os de `PLSDA_DE_TESTE`, para a figura do PLS-DA e a tabela falarem dos mesmos.

# As figuras que pintam por grupo clínico — eram nove, e são dez desde os scores de
# PCA. **Literal de propósito**: é a fonte independente contra a qual a assinatura dos construtores
# é conferida nas duas direções.
CONSTRUTORES_QUE_PINTAM_POR_GRUPO: Final = (
    "figura_dos_scores",
    "figura_dos_scores_do_plsda",
    "figura_dos_escores_um_contra_resto",
    "figura_do_dendrograma",
    "figura_do_mapa_de_calor",
    "figura_do_kmeans",
    "figura_do_mapa_auto_organizavel",
    "figura_do_random_forest",
    "figura_do_svm",
    "figura_da_composicao_por_classe",
)


def _corpo_com_coorte(seed: int, *grupos: str, **plsda: Any) -> dict[str, Any]:
    """Um corpo de `/execucoes` com o PLS-DA e a composição — os dois vêm na mesma resposta."""
    classes = [{**_CLASSE_ATLL, "grupo": grupo} for grupo in grupos]
    return {
        **_com_plsda(seed, **plsda),
        "composicao_por_classe": {**COMPOSICAO_DE_TESTE, "classes": classes},
    }


def _desenha_o_plsda(graficos: PainelDeGraficos, corpo: dict[str, Any]) -> None:
    """A ordem da janela: os `recebe_*` primeiro, e só então a chave da execução nova."""
    graficos.recebe_plsda(corpo)
    graficos.recebe_composicao(corpo, corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, corpo))
    _escolhe(graficos, "figura_dos_scores_do_plsda")


def _linha_do_grupo_na_tela(graficos: PainelDeGraficos, grupo: str) -> Any:
    """A série do grupo no gráfico de scores que está NA TELA, pelo nome no início do rótulo."""
    todas = graficos.figuras.canvas.figure.axes[0].lines
    linhas = [linha for linha in todas if str(linha.get_label()).startswith(f"{grupo} — ")]
    assert len(linhas) == 1, [str(linha.get_label()) for linha in todas]
    return linhas[0]


def _escolhe_na_tabela(
    graficos: PainelDeGraficos, grupo: str, *, cor: str = "", simbolo: str = ""
) -> None:
    """O gesto da pessoa: mudar o seletor da linha do grupo. Nada é chamado por fora dele."""
    seletor_da_cor, seletor_do_simbolo = graficos.tabela_dos_grupos.seletores[grupo]
    if cor:
        seletor_da_cor.setCurrentIndex(seletor_da_cor.findData(cor))
    if simbolo:
        seletor_do_simbolo.setCurrentIndex(seletor_do_simbolo.findData(simbolo))


def test_d662_a_tabela_fica_VAZIA_com_uma_frase_e_se_preenche_com_a_composicao(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """Os dois estados da tabela, e o cheio mostra o estilo que as figuras de facto recebem.

    **O que o seletor mostra é o que a figura recebe**: a linha de cada grupo nasce com o
    padrão da coorte, e a asserção compara os seletores com `estilos()`, que é o que os montadores
    passam aos construtores.
    """
    from refino.desktop.janela.graficos import SEM_GRUPOS_NA_TABELA

    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    tabela = graficos.tabela_dos_grupos
    assert tabela.seletores == {}
    assert tabela.frase.text() == SEM_GRUPOS_NA_TABELA
    assert tabela.estilos() is None

    graficos.recebe_composicao(_corpo_com_coorte(5, "HAMTSP", "Asymptomatic"), corresponde=True)
    assert list(tabela.seletores) == ["Asymptomatic", "HAMTSP"]
    assert tabela.frase.text() != SEM_GRUPOS_NA_TABELA
    estilos = tabela.estilos()
    assert estilos is not None
    for grupo, (cor, simbolo) in tabela.seletores.items():
        assert (cor.currentData(), simbolo.currentData()) == estilos.por_grupo[grupo].par
    assert estilos.por_grupo["Asymptomatic"].par == ("azul", "círculo")
    assert estilos.por_grupo["HAMTSP"].par == ("vermelho", "círculo")


def test_d662_a_tabela_RECUSA_o_mesmo_par_em_dois_grupos_e_aceita_meio_par(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """O par repetido volta atrás e é dito; o mesmo pedido com outro símbolo passa.

    **O par causal é de uma variável**: azul é recusado para o HAMTSP enquanto o Asymptomatic tem
    azul com círculo, e aceito depois de o HAMTSP passar ao quadrado. Uma recusa por cor sozinha
    proibiria a escolha que serve à impressão em cinza.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_composicao(_corpo_com_coorte(5, "Asymptomatic", "HAMTSP"), corresponde=True)
    tabela = graficos.tabela_dos_grupos
    mudancas: list[bool] = []
    tabela.mudou.connect(lambda: mudancas.append(True))
    antes = tabela.estilos()

    _escolhe_na_tabela(graficos, "HAMTSP", cor="azul")
    assert tabela.estilos() == antes, "a escolha repetida entrou nos estilos"
    assert tabela.seletores["HAMTSP"][0].currentData() == "vermelho", "o seletor não voltou atrás"
    assert "Asymptomatic" in tabela.frase.text() and "azul com círculo" in tabela.frase.text()
    assert mudancas == [], "a recusa redesenhou as figuras"

    _escolhe_na_tabela(graficos, "HAMTSP", simbolo="quadrado")
    _escolhe_na_tabela(graficos, "HAMTSP", cor="azul")
    depois = tabela.estilos()
    assert depois is not None
    assert depois.por_grupo["HAMTSP"].par == ("azul", "quadrado")
    assert "Asymptomatic" not in tabela.frase.text(), "a frase da recusa ficou depois de aceitar"
    assert len(mudancas) == 2


def test_d662_o_mesmo_grupo_tem_a_MESMA_cor_em_dois_contrastes_pela_janela(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """O HAMTSP é o `y = 1` do primeiro contraste e o `y = 0` do segundo, e a cor dele não muda.

    **Sem escolha nenhuma**: o padrão sai da coorte inteira, que a composição traz, e não dos dois
    grupos presentes — era isso que fazia o mesmo grupo mudar de cor conforme o contraste.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    coorte = ("Asymptomatic", "HAMTSP", "NoHTLV")

    _desenha_o_plsda(graficos, _corpo_com_coorte(5, *coorte))
    no_primeiro = to_hex(_linha_do_grupo_na_tela(graficos, "HAMTSP").get_color())
    assert "classe positiva" in str(_linha_do_grupo_na_tela(graficos, "HAMTSP").get_label())

    segundo = _corpo_com_coorte(
        6,
        *coorte,
        grupos_por_amostra=["HAMTSP", "HAMTSP", "NoHTLV", "NoHTLV"],
        grupo_positivo="NoHTLV",
        grupo_negativo="HAMTSP",
    )
    _desenha_o_plsda(graficos, segundo)
    no_segundo = to_hex(_linha_do_grupo_na_tela(graficos, "HAMTSP").get_color())
    assert "classe de referência" in str(_linha_do_grupo_na_tela(graficos, "HAMTSP").get_label())

    assert no_primeiro == no_segundo, (
        f"o HAMTSP saiu {no_primeiro} num contraste e {no_segundo} no outro: a cor ainda vem do "
        "lado do contraste, e não do grupo"
    )


def test_d662_mudar_a_cor_de_um_grupo_REDESENHA_a_figura_aberta(qtbot: QtBot) -> None:  # noqa: N802
    """A figura na tela é reconstruída com a cor nova, sem a pessoa trocar de figura."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    _desenha_o_plsda(graficos, _corpo_com_coorte(5, "Asymptomatic", "HAMTSP"))
    figura_antes = graficos.figuras.canvas.figure
    cor_antes = to_hex(_linha_do_grupo_na_tela(graficos, "HAMTSP").get_color())
    assert cor_antes != to_hex("#009E73"), "âncora: a cor escolhida já era a de antes"

    _escolhe_na_tabela(graficos, "HAMTSP", cor="verde-azulado", simbolo="triângulo")

    assert graficos.figuras.canvas.figure is not figura_antes, "a figura não foi reconstruída"
    linha = _linha_do_grupo_na_tela(graficos, "HAMTSP")
    assert (to_hex(linha.get_color()), linha.get_marker()) == (to_hex("#009E73"), "^")
    assert graficos.figuras.estado_visivel()["figura"] == "1 eixo(s)"


def test_d662_exportar_depois_de_escolher_GRAVA_a_cor_escolhida(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """O arquivo tem a cor escolhida, e o mesmo arquivo gravado antes da escolha não a tem.

    **O oráculo são os PIXELS do PNG**, e não a figura em memória: o que a pessoa leva para o artigo
    é o arquivo. O verde-azulado não é o padrão de nenhum dos dois grupos, então a planta separa.
    """
    from PIL import Image

    alvo = np.array([0x00, 0x9E, 0x73])
    destino = tmp_path / "scores.png"
    escolhe, _pedidos = _escolhedor(destino)
    graficos = PainelDeGraficos(escolhedor=escolhe)
    qtbot.addWidget(graficos)
    _desenha_o_plsda(graficos, _corpo_com_coorte(5, "Asymptomatic", "HAMTSP"))

    def tem_a_cor() -> bool:
        desfecho, frase = graficos.figuras.exporta(dpi=300)
        assert desfecho is DesfechoDaExportacao.GRAVADA, frase
        with Image.open(destino) as imagem:
            pixels = np.asarray(imagem.convert("RGB"))
        return bool(np.any(np.all(pixels == alvo, axis=-1)))

    assert not tem_a_cor(), "âncora: o verde-azulado já estava no arquivo antes da escolha"
    _escolhe_na_tabela(graficos, "HAMTSP", cor="verde-azulado")
    assert tem_a_cor(), "o arquivo gravado depois da escolha não tem a cor escolhida"


def test_d662_com_mais_grupos_do_que_pares_a_tabela_DIZ_e_as_figuras_ficam_como_antes(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """O terceiro desfecho da tabela: há grupos e não há pares distintos para todos."""
    from refino.plots.series import PARES_DE_ESTILO

    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    muitos = [f"g{i:02d}" for i in range(len(PARES_DE_ESTILO) + 1)]
    graficos.recebe_composicao(_corpo_com_coorte(5, *muitos), corresponde=True)
    tabela = graficos.tabela_dos_grupos
    assert tabela.estilos() is None
    assert tabela.seletores == {}
    assert (
        str(len(muitos)) in tabela.frase.text() and str(len(PARES_DE_ESTILO)) in tabela.frase.text()
    )


def _chamadas_sem_estilo(fonte: str, nomes: set[str]) -> tuple[list[str], int]:
    """As chamadas aos construtores `nomes` sem `estilos=`, e quantas chamadas houve ao todo."""
    sem: list[str] = []
    achadas = 0
    for no in ast.walk(ast.parse(fonte)):
        if isinstance(no, ast.Call) and isinstance(no.func, ast.Name) and no.func.id in nomes:
            achadas += 1
            if "estilos" not in {palavra.arg for palavra in no.keywords}:
                sem.append(f"{no.func.id}:{no.lineno}")
    return sem, achadas


def test_d662_todo_construtor_que_pinta_por_grupo_RECEBE_o_estilo_do_painel() -> None:  # noqa: N802
    """As figuras que pintam por grupo aceitam o estilo, e o painel passa-o a todas — dez.

    **Duas direções e uma contagem.** A assinatura confere a lista contra o registo — um construtor
    novo que aceite `estilos` sem entrar aqui reprova, e um que o perca também —; a varredura do
    `graficos.py` confere que cada montador o passa, e a contagem impede a asserção de ausência de
    passar sobre um arquivo onde a extração não achou chamada nenhuma.
    """
    import inspect

    from refino.desktop.janela import graficos as modulo_dos_graficos

    com_estilo = {
        construtor.funcao
        for construtor in CONSTRUTORES
        if "estilos"
        in inspect.signature(
            getattr(importlib.import_module(f"refino.plots.{construtor.modulo}"), construtor.funcao)
        ).parameters
    }
    assert com_estilo == set(CONSTRUTORES_QUE_PINTAM_POR_GRUPO)

    fonte = Path(modulo_dos_graficos.__file__).read_text(encoding="utf-8")
    sem, achadas = _chamadas_sem_estilo(fonte, set(CONSTRUTORES_QUE_PINTAM_POR_GRUPO))
    assert achadas == len(CONSTRUTORES_QUE_PINTAM_POR_GRUPO), achadas
    assert sem == [], f"estes montadores não passam o estilo dos grupos: {sem}"


def test_d662_o_portao_do_estilo_ACUSA_a_chamada_que_o_esquece() -> None:  # noqa: N802
    """O par do portão acima, em processo: a planta é o texto que a varredura recebe."""
    plantado = (
        "figura_do_kmeans(particao=p, recorte=r)\n"
        "figura_do_svm(classificacao=c, recorte=r, estilos=e)\n"
    )
    sem, achadas = _chamadas_sem_estilo(plantado, {"figura_do_kmeans", "figura_do_svm"})
    assert achadas == 2
    assert sem == ["figura_do_kmeans:1"]


# ---------------------------------------------------------------------------
# O intervalo do fold change nas 20 de menor p, lido da raiz `volcano`
# ---------------------------------------------------------------------------
#
# **Os vetores do intervalo para as cinco features de `VOLCANO_DE_TESTE`.** As quatro com `p` são
# as escolhidas — vinte é mais do que elas —; a quarta não tem `log2fc` e por isso não tem
# intervalo, com o motivo que o núcleo daria a uma mediana nula. A quinta tem intervalo e não tem
# `p`: a regra não a pode escolher, e uma barra nela seria a de uma feature fora da regra.
INTERVALO_DE_TESTE: Final[dict[str, Any]] = {
    "log2fc_ci_lower": [-3.0, 1.0, -0.5, None, 0.5],
    "log2fc_ci_upper": [-1.0, 5.0, 2.0, None, 2.5],
    "fold_change_ci_undefined": [None, None, None, "order_statistic_not_positive", None],
    "nivel_do_intervalo": 0.95,
}
# Três barras, sobre as quatro escolhidas e as cinco testadas: o nível de Benjamini-Yekutieli sai
# daqui e não do construtor.
BARRAS_DE_TESTE: Final = 3
ESCOLHIDAS_DE_TESTE: Final = 4


@pytest.mark.parametrize("funcao", ["figura_do_volcano", "figura_do_fold_change"])
def test_d669_o_painel_LE_o_intervalo_da_raiz_e_poe_as_barras_nas_DUAS_figuras(  # noqa: N802
    qtbot: QtBot, funcao: str
) -> None:
    """A leitura do intervalo pela janela, que ficou de fora antes: o corpo trazia e nada lia.

    As duas figuras desenham o mesmo `log2fc` da mesma raiz, e as duas ganham a barra das mesmas
    escolhidas, com o mesmo nível dito no rodapé — duas leituras seriam irmãs que divergem.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = _com_volcano(41, **INTERVALO_DE_TESTE)
    graficos.recebe_volcano(corpo)
    assert graficos.figuras.define_execucao(_resposta(200, corpo)) is not None
    _escolhe(graficos, funcao)

    eixo = graficos.figuras.canvas.figure.axes[0]
    barras = [s for s in series(eixo) if s.rotulo == ROTULO_DAS_BARRAS]
    assert len(barras) == 1, [s.rotulo for s in series(eixo)]
    assert int(np.isfinite(barras[0].x).sum()) == 2 * BARRAS_DE_TESTE
    assert confere_vista(eixo) == []

    rodape = _rodape_desenhado(graficos.figuras)
    nivel = 1.0 - BARRAS_DE_TESTE * (1.0 - 0.95) / VOLCANO_DE_TESTE["n_features_testadas"]
    for trecho in (
        f"nas {BARRAS_DE_TESTE} das {ESCOLHIDAS_DE_TESTE} de menor p",
        f"{decimal(nivel * 100.0, 5)}%",
        "Benjamini-Yekutieli",
    ):
        assert trecho in rodape, (trecho, rodape)


def test_d669_o_clique_no_ponto_COM_barra_diz_a_FEATURE_e_nao_a_barra(qtbot: QtBot) -> None:  # noqa: N802
    """**O par causal da ordem de inserção, pelo caminho real do clique (`Figure.pick`).**

    A barra passa pelo ponto. O `matplotlib` dispara um evento por artista na ordem em que eles
    entraram no eixo, e `_identifica` guarda o último: com a barra inserida antes da nuvem, o ponto
    responde pela feature. A planta reinsere a barra depois da nuvem, e a mesma coordenada passa a
    anotar a barra — o que um clique no ponto mais forte do volcano mostraria.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = _com_volcano(42, **INTERVALO_DE_TESTE)
    graficos.recebe_volcano(corpo)
    graficos.figuras.define_execucao(_resposta(200, corpo))
    _escolhe(graficos, "figura_do_volcano")
    painel = graficos.figuras
    eixo = painel.canvas.figure.axes[0]
    x = float(VOLCANO_DE_TESTE["log2fc"][0])
    y = float(-np.log10(VOLCANO_DE_TESTE["pvalue"][0]))

    _dispara_o_pick(painel, eixo, x, y)
    assert painel.estado_visivel()["anotacao"].startswith(FEATURES_DO_VOLCANO[0]), (
        painel.estado_visivel()["anotacao"]
    )

    barras = next(a for a in eixo.lines if a.get_label() == ROTULO_DAS_BARRAS)
    barras.remove()
    eixo.add_line(barras)
    _dispara_o_pick(painel, eixo, x, y)
    assert painel.estado_visivel()["anotacao"].startswith(ROTULO_DAS_BARRAS), (
        "a planta não separa: com a barra depois da nuvem o clique continuou a dizer a feature"
    )


def test_d669_SEM_os_campos_do_intervalo_o_rodape_DIZ_que_nao_vieram(qtbot: QtBot) -> None:  # noqa: N802
    """O corpo sem os campos não desenha barra — e diz porquê, em vez de calar."""
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    graficos.recebe_volcano(_com_volcano(43))
    graficos.figuras.define_execucao(_resposta(200, _com_volcano(43)))
    _escolhe(graficos, "figura_do_volcano")
    eixo = graficos.figuras.canvas.figure.axes[0]
    assert ROTULO_DAS_BARRAS not in {s.rotulo for s in series(eixo)}
    assert "não trouxe" in _rodape_desenhado(graficos.figuras)


# Os scores de PCA pintados por grupo, com uma região por grupo
# ---------------------------------------------------------------------------
#
# **Seis amostras de estudo em dois grupos de três**, com os nomes da coorte de `_corpo_com_coorte`
# para a tabela dos estilos e a figura falarem dos mesmos grupos. Três é o mínimo da região, então
# os dois grupos a têm; o QC e a validação chegam com `null`, como `rotas._resumo_da_pca` os serve.
PCA_POR_GRUPO: Final[dict[str, Any]] = {
    **PCA_DE_TESTE,
    "scores_pc1": [-2.0, -1.0, 0.5, 1.0, 2.5, -1.5, 0.8, 1.6, -0.6],
    "scores_pc2": [1.0, -0.5, 0.25, -1.5, 0.75, 0.4, -0.9, 1.2, -1.1],
    "tipos_de_amostra": [
        "qc",
        "sample",
        "sample",
        "validate",
        "qc",
        "sample",
        "sample",
        "sample",
        "sample",
    ],
    "grupos_por_amostra": [
        None,
        "Asymptomatic",
        "HAMTSP",
        None,
        None,
        "Asymptomatic",
        "HAMTSP",
        "Asymptomatic",
        "HAMTSP",
    ],
}


def test_os_scores_de_pca_PINTAM_por_grupo_com_a_cor_da_tabela_e_uma_regiao_por_grupo(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """Ponta a ponta pelo fio: o grupo de cada amostra chega, e a figura pinta por ele.

    **Três metades.** Com a chave `grupos_por_amostra` a nuvem é partida nos dois grupos, cada um
    com a cor e o símbolo que a tabela dos grupos mostra e com a região dele na mesma cor; a
    escolha feita na tabela chega à figura da tela; e **o par causal** — o mesmo corpo sem a chave
    — devolve a figura anterior, com a nuvem numa série só e sem região por grupo.
    """
    graficos = PainelDeGraficos()
    qtbot.addWidget(graficos)
    corpo = _corpo_com_coorte(33, "Asymptomatic", "HAMTSP")
    graficos.recebe_pca(
        {"trajetoria_dos_qcs": dict(TRAJETORIA_DE_TESTE), "pca": dict(PCA_POR_GRUPO)}
    )
    graficos.recebe_composicao(corpo, corresponde=True)
    graficos.figuras.define_execucao(_resposta(200, corpo))
    _escolhe(graficos, "figura_dos_scores")

    estilos = graficos.tabela_dos_grupos.estilos()
    assert estilos is not None, "âncora: a coorte não chegou à tabela, e não há estilo a conferir"
    linhas = {
        str(linha.get_label()): linha for linha in graficos.figuras.canvas.figure.axes[0].lines
    }
    for grupo in ("Asymptomatic", "HAMTSP"):
        serie, regiao = linhas[f"{grupo} — 3 amostras"], linhas[rotulo_da_regiao_do_grupo(grupo)]
        assert serie.get_color() == estilos.por_grupo[grupo].hex == regiao.get_color(), grupo
        assert serie.get_marker() == estilos.por_grupo[grupo].marcador, grupo
    assert "QC — 2 amostras" in linhas, sorted(linhas)
    assert not any(rotulo.startswith("amostra de estudo") for rotulo in linhas), sorted(linhas)

    _escolhe_na_tabela(graficos, "HAMTSP", cor="verde-azulado")
    assert _linha_do_grupo_na_tela(graficos, "HAMTSP").get_color() == "#009E73"

    sem_a_chave = {k: v for k, v in PCA_POR_GRUPO.items() if k != "grupos_por_amostra"}
    graficos.recebe_pca({"trajetoria_dos_qcs": dict(TRAJETORIA_DE_TESTE), "pca": sem_a_chave})
    graficos.figuras.reavalia()
    rotulos = {str(linha.get_label()) for linha in graficos.figuras.canvas.figure.axes[0].lines}
    assert "amostra de estudo — 6 amostras" in rotulos, sorted(rotulos)
    assert not any(rotulo.startswith(rotulo_da_regiao_do_grupo("")) for rotulo in rotulos)


@pytest.mark.parametrize(
    ("posicao", "valor", "scores_desenham"),
    [(1, "", True), (0, "Asymptomatic", False)],
    ids=["estudo-com-nome-em-branco", "qc-com-grupo"],
)
def test_o_biplot_NAO_le_o_grupo_e_o_nome_em_branco_NAO_retira_os_scores(  # noqa: N802
    qtbot: QtBot, posicao: int, valor: str, scores_desenham: bool
) -> None:
    """Um grupo que a fronteira recusava retirava da tela o BIPLOT.

    **O biplot pinta por tipo e não lê o grupo**, e lia os escores pelo mesmo leitor dos scores —
    que, com a chave nova, passou a conferir o vetor de grupos. Um corpo com um grupo recusado
    retirava as duas figuras, e a tela dizia *"os vetores ... não descrevem os mesmos objetos"*,
    que não era a causa. **Os dois casos, e eles têm desfechos opostos nos scores:** o nome em
    branco é fato do dado — a ingestão aceita a célula `LABEL` vazia — e dá a figura antiga, com
    o título a dizer porquê; um QC com grupo é corpo montado errado, e os scores continuam
    recusados. **Nos dois o biplot desenha**, como desenhava antes da chave existir.
    """
    grupos = list(PCA_POR_GRUPO["grupos_por_amostra"])
    grupos[posicao] = valor
    pca = {**PCA_POR_GRUPO, "grupos_por_amostra": grupos}

    def estado(funcao: str) -> dict[str, str]:
        graficos = PainelDeGraficos()
        qtbot.addWidget(graficos)
        graficos.recebe_pca({"trajetoria_dos_qcs": dict(TRAJETORIA_DE_TESTE), "pca": pca})
        graficos.figuras.define_execucao(_resposta(200, _corpo(43)))
        _escolhe(graficos, funcao)
        visivel = graficos.figuras.estado_visivel()
        if visivel["figura"]:
            eixo = graficos.figuras.canvas.figure.axes[0]
            visivel["titulo"] = " ".join(eixo.get_title().split())
            visivel["series"] = "|".join(str(linha.get_label()) for linha in eixo.lines)
        return visivel

    assert estado("figura_do_biplot")["figura"] == "1 eixo(s)"
    scores = estado("figura_dos_scores")
    if scores_desenham:
        assert scores["figura"] == "1 eixo(s)", scores
        assert "amostra de estudo — 6 amostras" in scores["series"].split("|"), scores
        assert "1 amostra de estudo com o grupo em branco" in scores["titulo"], scores
    else:
        assert scores["figura"] == "", scores
        assert scores["repouso"] == texto_do_estado(EstadoDaFigura.DADO_RECUSADO), scores

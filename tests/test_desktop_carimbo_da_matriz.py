"""A resposta de uma matriz que chega depois de a janela passar a outra.

**O defeito confirmado por sonda:** `recebe_medicao` descartava a medição carimbada
com outra matriz, e `_encaminha` escrevia a prosa dela mesmo assim — a medição de A, recebida
depois da troca para B, escrevia *«**loess** é o menor RSD de QC fora da amostra…»* sob B.
`recebe_medicao` devolvia `None` ao guardar e ao descartar, e quem a chamava não tinha como seguir
a decisão — a mesma forma já consertada em `recebe_curva_do_loess`.

**As irmãs**, cada uma com o seu teste aqui — o que a guardava, e o que faltava:

- **medição dos corretores**: o carimbo de `graficos`; faltava o rótulo da medição, no
  `200` e na recusa;
- **intervalo por barra**: o mesmo, porque viaja no corpo da medição;
- **curva do LOESS**: o carimbo de `graficos`, e nada faltava com um disparo em B;
- **as três acima com B só ESCOLHIDO**: nada, porque `graficos` só aprendia a matriz no disparo
  seguinte; faltava seguir a matriz da tela na troca;
- **o que já estava na tela ao trocar**: `graficos` descartava as barras no disparo, e o rótulo
  ficava com o vencedor de A;
- **Palpite medido**: nada guardava o que ele mostra — só a retomada lia a identidade;
- **validação e diagnóstico**: a geração dos arquivos (`entrada-N`), que não contava a troca de
  MODO, e o modo é parte da matriz;
- **a execução**: só a supersessão por `pedido` — o desfecho de A enchia a aba de Qualidade sob B.

**E a troca que VOLTA não custa nada.** O combo do modo emite a cada tecla, então corrigir o modo à
mão é uma sequência de trocas de matriz; a medição e a curva são escondidas e repostas pela
identidade, como o diagnóstico, e as leituras em voo levam a matriz na etiqueta, e não uma
segunda geração. A volta é afirmada aqui pelo editor do combo, tecla a tecla, e por A→B→A.

**Os arquivos são da factory, e nada corre corretor.** `chama` é espiado — o pedido não segue para
a API —, e as respostas entram por `_recebe`, com a chave composta que a releitura encadeada usa:
é o caminho de produção, e não `_encaminha` chamado à mão. Os corpos das leituras vêm da rota
sobre a factory, e não de um dicionário escrito aqui.

## Escopo negativo

**Não é o Windows nem o executável congelado**: Qt `offscreen`, com a API em processo. **E o que as
leituras de A já mostram quando B é escolhido não é afirmado aqui** — o texto da validação e do
diagnóstico, o cartão e a proposta do Palpite ficam na tela até a leitura seguinte.

> A execução de A já terminada saiu deste escopo negativo: a execução de cada matriz fica guardada
> pela identidade dela, sai na troca e volta com a matriz — a última seção deste arquivo. As
> leituras continuam como estavam.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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

from fastapi.testclient import TestClient  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QSignalSpy, QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402
from pytestqt.qtbot import QtBot  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

from refino.api.app import build_app  # noqa: E402
from refino.desktop.etapas import (  # noqa: E402
    CAMPO_SEM_PROPOSTA,
    CHAVE_DA_ETAPA,
    ETAPAS,
    ROTA_DA_ETAPA,
)
from refino.desktop.janela import enriquecimento as ej  # noqa: E402
from refino.desktop.janela.arquivos import ROTULO_DA_LISTA, ROTULO_DAS_MEDIDAS  # noqa: E402
from refino.desktop.janela.figuras import EstadoDaFigura, texto_do_estado  # noqa: E402
from refino.desktop.janela.leitura import (  # noqa: E402
    CHAVE_DO_PALPITE,
    ROTA_DO_PALPITE,
    ROTAS_DA_LEITURA,
    SEM_LEITURA,
)
from refino.desktop.janela.pipeline import (  # noqa: E402
    ROTA_DA_EXECUCAO,
    SEM_LOTES,
    SEM_MEDICAO,
    VARIANTE_DO_RSD_EM_REPOUSO,
    rota_do_progresso,
)
from refino.desktop.janela.preparo import SUBTITULO_DAS_LEITURAS, AbaDePreparo  # noqa: E402
from refino.desktop.janela.principal import (  # noqa: E402
    CHAVE_DA_CURVA_DO_LOESS,
    CHAVE_DA_EXECUCAO,
    CHAVE_DA_MEDICAO,
    ENTRADA_MUDOU_DURANTE_O_PALPITE,
    FRASE_DA_CURVA_DESCARTADA,
    FRASE_DA_EXECUCAO_GUARDADA,
    FRASE_DA_MEDICAO_DESCARTADA,
    FRASE_DO_PALPITE_DESCARTADO,
    INDICE_DA_ABA_DOS_GRAFICOS,
    ORDEM_DAS_LEITURAS,
    PALPITE_EM_CURSO,
    JanelaPrincipal,
    chave_do_pedido,
)
from refino.desktop.janela.qualidade import PainelDeQualidade  # noqa: E402
from refino.desktop.vias_r import MotorDoR  # noqa: E402
from refino.interface.cliente import Resposta  # noqa: E402
from refino.interface.referencias import SEM_RESUMO  # noqa: E402
from refino.persistence.config import VARIAVEL_DA_URL, prepara_sqlite  # noqa: E402
from refino.persistence.schema import METADATA  # noqa: E402
from refino.plots.curva_do_loess import (  # noqa: E402
    ROTULO_DA_CURVA,
    ROTULO_DA_CURVA_DE_OUTRO_CORRETOR,
)
from tests.arnes_de_figura import series  # noqa: E402
from tests.factories import base_spec, render_pair  # noqa: E402
from tests.test_desktop_curva_do_loess import corpo_da_fabrica  # noqa: E402
from tests.test_desktop_enriquecimento import SHA_DE_TESTE, Executor  # noqa: E402
from tests.test_desktop_figuras import _com_qualidade, _medicao  # noqa: E402
from tests.test_desktop_intervalo_das_barras import _intervalos_do_corpo  # noqa: E402
from tests.test_desktop_vias import _relatorio as _relatorio_de_vias  # noqa: E402
from tests.test_desktop_vias import _via  # noqa: E402

MODO = "neg_hilic"
FEATURE = "1.00_100.0000m/z"
IDENTIFICADOR = "trabalho-de-A"
PALPITE_DE_A = {
    "correction_proposta": "loess",
    "avaliavel": True,
    "abstem": False,
    "confianca": 0.9,
    "n_features_medidas": 200,
}


class _Escolha:
    """O diálogo de arquivos: devolve os da pasta corrente — `a` ou `b` —, pelo título do botão."""

    def __init__(self, raiz: Path) -> None:
        self.raiz = raiz
        self.pasta = "a"

    def __call__(self, titulo: str) -> Path:
        nome = {ROTULO_DAS_MEDIDAS: "measurements", ROTULO_DA_LISTA: "Sample_List"}[titulo]
        return self.raiz / self.pasta / f"{nome}_{MODO}.csv"


def _escreve_os_pares(raiz: Path) -> None:
    """Dois pares da factory com contagens de features diferentes, e por isso bytes diferentes.

    O modo vai no NOME, para a janela o deduzir pelo caminho de produção.
    """
    for pasta, n_features in (("a", 5), ("b", 6)):
        medidas, lista = render_pair(base_spec(n_features=n_features, n_samples=40, n_qcs=7))
        (raiz / pasta).mkdir(parents=True)
        (raiz / pasta / f"measurements_{MODO}.csv").write_text(
            medidas, encoding="utf-8", newline=""
        )
        (raiz / pasta / f"Sample_List_{MODO}.csv").write_text(lista, encoding="utf-8", newline="")


def _janela(qtbot: QtBot, tmp_path: Path) -> tuple[JanelaPrincipal, _Escolha, list[str]]:
    """A janela com o par A escolhido pelos botões e `chama` espiado. Devolve as rotas pedidas."""
    _escreve_os_pares(tmp_path)
    escolha = _Escolha(tmp_path)
    janela = JanelaPrincipal(
        TestClient(build_app(), raise_server_exceptions=False), escolhedor=escolha
    )
    qtbot.addWidget(janela)
    rotas: list[str] = []
    janela.chama = lambda _metodo, rota, **_kw: rotas.append(rota)  # type: ignore[method-assign]
    _escolhe(janela, escolha, "a")
    assert janela.arquivos.modo.currentText() == MODO, "âncora: o modo não foi deduzido do nome"
    return janela, escolha, rotas


def _escolhe(janela: JanelaPrincipal, escolha: _Escolha, pasta: str) -> None:
    """Os dois botões de arquivo, como a pessoa os clica — a troca passa por `par_escolhido`."""
    escolha.pasta = pasta
    janela.arquivos.escolhe_medidas()
    janela.arquivos.escolhe_lista()
    assert janela.arquivos.par is not None, "âncora: o par não ficou completo"


def _resposta(codigo: int, corpo: Any) -> Resposta:
    return Resposta(
        codigo=codigo, ok=codigo < 400, causa=None, mensagem="", corpo=corpo, conteudo=None
    )


def _aceita(janela: JanelaPrincipal, chave: str) -> None:
    """O `202` do `POST`, que chega logo depois do clique e abre a espera."""
    janela._recebe(chave, _resposta(202, {"identificador": IDENTIFICADOR, "total": 3}))
    assert chave in janela._esperas, f"âncora: o `202` de `{chave}` não abriu a espera"


def _termina(janela: JanelaPrincipal, chave: str, corpo: Any, codigo: int = 200) -> None:
    """O desfecho, pela releitura encadeada — a chave composta com o identificador do trabalho."""
    janela._recebe(chave_do_pedido(chave, IDENTIFICADOR), _resposta(codigo, corpo))


def _corpo_da_rota(rota: str) -> dict[str, Any]:
    """O corpo de uma leitura sobre o par A da factory, pela rota — e não escrito à mão."""
    medidas, lista = render_pair(base_spec(n_features=5, n_samples=40, n_qcs=7))
    resposta = TestClient(build_app()).post(
        rota,
        files={
            "measurements": (f"measurements_{MODO}.csv", medidas.encode("utf-8"), "text/csv"),
            "sample_list": (f"Sample_List_{MODO}.csv", lista.encode("utf-8"), "text/csv"),
        },
        data={"mode": MODO},
    )
    assert resposta.status_code == 200, resposta.text
    corpo: dict[str, Any] = resposta.json()
    return corpo


def _corpo_da_tela(janela: JanelaPrincipal, chave: str) -> dict[str, Any]:
    """O corpo da leitura `chave` sobre os arquivos e o modo da TELA, pela rota.

    Irmão de `_corpo_da_rota`, que lê sempre o par A: depois de uma troca, a leitura que a janela
    pediria é a do par que está na tela.
    """
    resposta = TestClient(build_app()).post(
        dict(ROTAS_DA_LEITURA)[chave],
        files=janela.arquivos.arquivos_para_envio(),
        data=janela.arquivos.campos_para_envio(),
    )
    assert resposta.status_code == 200, resposta.text
    corpo: dict[str, Any] = resposta.json()
    return corpo


def _leituras_em_dia(
    janela: JanelaPrincipal, chaves: tuple[str, ...] = ("validacao", "diagnostico")
) -> None:
    """A Validação e o Diagnóstico da matriz da tela em dia, pelo caminho de produção.

    O cartão do Palpite e «Seguir sugestões do app» rodam antes a Validação e o Diagnóstico que
    faltarem. Os testes cujo assunto é o Palpite, e não a
    cadeia, entregam as duas por `_recebe` antes do clique, e o Palpite sai no clique como saía.

    Os corpos vêm das rotas, sobre os arquivos da TELA — e não os do par A —, para valer depois de
    uma troca. **O do Diagnóstico vai sem `lotes`**: sem sugestão de lotes, nenhum número muda
    nos testes que afirmam os lotes; quando o teste quer uma sugestão, ele a põe.
    """
    for chave in chaves:
        corpo = {campo: valor for campo, valor in _corpo_da_tela(janela, chave).items()}
        corpo.pop("lotes", None)
        janela._recebe(chave, _resposta(200, corpo))
        assert janela._leitura_em_dia(chave) is True, f"âncora: `{chave}` não ficou em dia"


# ---------------------------------------------------------------------------
# A medição dos corretores, e o intervalo que viaja nela
# ---------------------------------------------------------------------------


def _mede_e_responde(
    qtbot: QtBot, tmp_path: Path, *, troca: str | None, dispara_em_b: bool, corpo: Any, codigo: int
) -> JanelaPrincipal:
    """Medir A; trocar (ou não) os arquivos; e só então o desfecho da medição de A chegar."""
    janela, escolha, rotas = _janela(qtbot, tmp_path)
    janela.pipeline.medir_botao.click()
    assert rotas, "âncora: o clique não disparou a medição"
    _aceita(janela, CHAVE_DA_MEDICAO)
    if troca is not None:
        _escolhe(janela, escolha, troca)
    if dispara_em_b:
        # **O caminho da sonda do revisor:** um disparo qualquer com B — aqui a validação — é o que
        # dizia ao painel dos gráficos que a matriz mudara, antes desta decisão.
        janela.leitura.botoes["validacao"].click()
    _termina(janela, CHAVE_DA_MEDICAO, corpo, codigo)
    return janela


@pytest.mark.parametrize("dispara_em_b", [True, False], ids=["com-disparo-em-B", "so-escolhido"])
def test_a_medicao_de_A_que_chega_depois_da_troca_nao_escreve_o_vencedor_sob_B(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, dispara_em_b: bool
) -> None:
    """A medição de A, com o intervalo por barra, chega com B na tela: nem barras, nem vencedor.

    **Par causal:** o mesmo desfecho sem a troca escreve o vencedor e guarda o corpo — e é a prosa
    dessa metade que a outra não pode conter. O corpo traz `intervalos`: o intervalo viaja
    na medição, e o painel da correção lê-o de `_medicao`, que fica vazio.

    **Os dois parâmetros são os dois caminhos da troca.** Com um disparo em B, o painel dos
    gráficos já descartava e o rótulo não — o defeito da sonda. Só com B escolhido, os dois
    aceitavam, porque o painel só aprendia a matriz no disparo seguinte.
    """
    corpo = {**_medicao(vencedor="loess"), "intervalos": _intervalos_do_corpo(_medicao()["scores"])}
    sem_troca = _mede_e_responde(
        qtbot, tmp_path / "sem", troca=None, dispara_em_b=False, corpo=corpo, codigo=200
    )
    prosa_de_a = sem_troca.pipeline.rotulo_da_medicao.text()
    assert "**loess**" in prosa_de_a, (
        f"âncora: sem a troca o vencedor não foi escrito: {prosa_de_a}"
    )
    assert sem_troca.graficos._medicao is not None, "âncora: sem a troca a medição foi descartada"

    com_troca = _mede_e_responde(
        qtbot, tmp_path / "com", troca="b", dispara_em_b=dispara_em_b, corpo=corpo, codigo=200
    )
    rotulo = com_troca.pipeline.rotulo_da_medicao.text()
    assert com_troca.graficos._medicao is None, (
        "a medição de A — com o intervalo de cada barra — ficou guardada com B na tela"
    )
    assert rotulo == FRASE_DA_MEDICAO_DESCARTADA, (
        f"o rótulo da medição sob B não diz que a de A foi descartada: {rotulo!r}"
    )
    assert prosa_de_a not in rotulo


@pytest.mark.parametrize("dispara_em_b", [True, False], ids=["com-disparo-em-B", "so-escolhido"])
def test_a_recusa_da_medicao_de_A_nao_vira_a_recusa_de_B(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, dispara_em_b: bool
) -> None:
    """A irmã no mesmo ramo: o `422` de A depois da troca não se escreve como recusa sob B."""
    corpo = {"detail": "RECUSA-DE-A: um lote com menos de três QCs"}
    sem_troca = _mede_e_responde(
        qtbot, tmp_path / "sem", troca=None, dispara_em_b=False, corpo=corpo, codigo=422
    )
    assert "RECUSA-DE-A" in sem_troca.pipeline.rotulo_da_medicao.text(), "âncora"

    com_troca = _mede_e_responde(
        qtbot, tmp_path / "com", troca="b", dispara_em_b=dispara_em_b, corpo=corpo, codigo=422
    )
    rotulo = com_troca.pipeline.rotulo_da_medicao.text()
    assert rotulo == FRASE_DA_MEDICAO_DESCARTADA, f"a recusa de A ficou escrita sob B: {rotulo!r}"


@pytest.mark.parametrize("pasta", ["b", "a"], ids=["outro-par", "o-mesmo-par"])
def test_trocar_os_arquivos_apaga_a_medicao_que_ja_estava_na_tela(
    qtbot: QtBot, tmp_path: Path, pasta: str
) -> None:
    """A medição de A já mostrada sai da tela quando B é escolhido — as barras e o vencedor juntos.

    **Par causal:** escolher outra vez os MESMOS arquivos não muda a matriz, e a medição fica. Antes
    desta decisão o painel dos gráficos descartava as barras só no disparo seguinte, e o rótulo
    nunca: o vencedor de A ficava escrito sob B até outra medição.
    """
    janela, escolha, _rotas = _janela(qtbot, tmp_path)
    janela.pipeline.medir_botao.click()
    _aceita(janela, CHAVE_DA_MEDICAO)
    _termina(janela, CHAVE_DA_MEDICAO, _medicao(vencedor="loess"))
    prosa_de_a = janela.pipeline.rotulo_da_medicao.text()
    assert "**loess**" in prosa_de_a and janela.graficos._medicao is not None, "âncora"

    _escolhe(janela, escolha, pasta)

    if pasta == "a":
        assert janela.pipeline.rotulo_da_medicao.text() == prosa_de_a, (
            "os mesmos arquivos apagaram a medição deles"
        )
        assert janela.graficos._medicao is not None
        return
    assert janela.pipeline.rotulo_da_medicao.text() == SEM_MEDICAO, (
        "o vencedor de A ficou escrito com B escolhido"
    )
    assert janela.graficos._medicao is None, "as barras de A ficaram com B escolhido"


# ---------------------------------------------------------------------------
# A curva do LOESS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pasta", ["b", "a"], ids=["outro-par", "o-mesmo-par"])
def test_a_curva_de_A_que_chega_com_B_so_escolhido_e_descartada(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, pasta: str
) -> None:
    """A curva era guardada pelo carimbo, e o painel só aprendia a matriz no disparo seguinte.

    **Par causal:** com os mesmos arquivos escolhidos outra vez, a curva é aceita e a janela vai à
    figura; com B, ela é descartada, a janela fica onde está e o rótulo do botão diz porquê.
    """
    janela, escolha, rotas = _janela(qtbot, tmp_path)
    janela.abas.setCurrentIndex(0)
    janela.pipeline.ver_curva_do_loess.emit(FEATURE)
    assert rotas, "âncora: o pedido da curva não saiu"
    _escolhe(janela, escolha, pasta)
    janela._recebe(CHAVE_DA_CURVA_DO_LOESS, _resposta(200, corpo_da_fabrica()))

    if pasta == "a":
        assert janela.graficos._curva_do_loess is not None, "âncora: a curva de A foi descartada"
        assert janela.abas.currentIndex() == INDICE_DA_ABA_DOS_GRAFICOS
        return
    assert janela.graficos._curva_do_loess is None, "a curva de A ficou guardada com B escolhido"
    assert janela.abas.currentIndex() == 0, "a janela foi à figura de uma curva descartada"
    assert janela.pipeline.rotulo_da_curva.text() == FRASE_DA_CURVA_DESCARTADA


# ---------------------------------------------------------------------------
# O Palpite medido
# ---------------------------------------------------------------------------


def _palpite_de_a(qtbot: QtBot, tmp_path: Path, pasta: str | None) -> tuple[JanelaPrincipal, str]:
    """Pedir o Palpite com A; escolher (ou não) outra pasta; e só então o desfecho chegar.

    A Validação e o Diagnóstico de A entram em dia antes do clique
    (`_leituras_em_dia`): sem eles o clique rodaria a Validação primeiro, e o assunto aqui é o
    desfecho do Palpite.
    """
    janela, escolha, rotas = _janela(qtbot, tmp_path)
    antes = janela.pipeline.corretor.currentText()
    assert antes != PALPITE_DE_A["correction_proposta"], "âncora: o repouso já era a proposta"
    _leituras_em_dia(janela)
    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    assert rotas == [ROTA_DO_PALPITE], f"âncora: o clique não pediu só o Palpite: {rotas}"
    _aceita(janela, CHAVE_DO_PALPITE)
    if pasta is not None:
        _escolhe(janela, escolha, pasta)
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))
    return janela, antes


def test_o_palpite_de_A_que_chega_depois_da_troca_nao_e_guardado_nem_pre_preenche(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """O Palpite de A, chegado com B na tela, não vira a proposta de B nem o corretor proposto.

    `leitura.mostra` GUARDA o corpo como a proposta que `/execucoes` compara — o argumento da
    docstring de `PainelDeLeitura.mostra` —, e o pré-preenchimento põe o corretor dele no painel do
    pipeline: os dois aconteciam sob B.

    **Par causal:** o mesmo desfecho sem a troca é guardado. O descarte é pela matriz da tela no
    instante da chegada: de volta aos arquivos do pedido, o Palpite deles é aceito — é a metade de
    baixo do teste da retomada, logo abaixo.

    Sem a troca, o Palpite era guardado E pré-preenchia o seletor; hoje nenhum Palpite escreve no
    seletor, e o lado aceito do par é a sugestão guardada.

    O pedido sai com a Validação e o Diagnóstico de A já em dia
    (`_palpite_de_a`): sem eles o clique rodaria a cadeia, e a troca cairia antes do Palpite sair.
    """
    sem_troca, antes = _palpite_de_a(qtbot, tmp_path / "sem", None)
    assert sem_troca.leitura.recomendacao_guardada() is not None, "âncora: nada foi guardado"
    assert sem_troca.pipeline.corretor_sugerido == PALPITE_DE_A["correction_proposta"]
    assert sem_troca.pipeline.corretor.currentText() == antes, "o Palpite escreveu no seletor"

    com_troca, antes = _palpite_de_a(qtbot, tmp_path / "com", "b")
    assert com_troca.leitura.recomendacao_guardada() is None, (
        "o Palpite de A ficou guardado como a proposta da máquina para B"
    )
    assert com_troca.pipeline.corretor.currentText() == antes, (
        "o corretor de B foi pré-preenchido com a proposta de A"
    )
    assert com_troca.pipeline.corretor_sugerido is None, "a sugestão de A ficou guardada para B"
    assert com_troca.leitura.rotulos[CHAVE_DO_PALPITE].text() == FRASE_DO_PALPITE_DESCARTADO
    assert com_troca.leitura.cartoes[CHAVE_DO_PALPITE].selo.text() == "Descartado", (
        "o selo do cartão, que é o que está à vista, não diz que o Palpite foi descartado"
    )


def test_o_palpite_descartado_apaga_a_proposta_de_um_pedido_anterior(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Um Palpite anterior, já guardado, não fica como proposta quando o seguinte é descartado.

    **A planta tem de cair onde as duas coisas diferem:** sem um Palpite anterior a proposta
    já era `None`, e um descarte que a deixasse estaria verde. Aqui o primeiro Palpite de A é aceito
    e guardado, o segundo é pedido, B é escolhido, e o segundo chega: o cartão diz «Descartado», e a
    proposta que `/execucoes` compararia não pode ser uma que ele já não mostra — o argumento da
    docstring de `PainelDeLeitura.mostra`. Pelo mesmo motivo nenhum Palpite fica «em dia», e o
    clique seguinte em «Seguir sugestões do app» mede antes de aplicar — antes, o disparo
    delegado seguinte.

    **Par causal:** a mesma sequência sem a troca aceita o segundo Palpite, que passa a ser a
    proposta guardada e o que está em dia.

    A Validação e o Diagnóstico de A entram em dia antes do primeiro clique
    (`_leituras_em_dia`), e os dois cliques pedem só o Palpite, como pediam: o assunto é o descarte.
    """
    for troca in (False, True):
        janela, escolha, _rotas = _janela(qtbot, tmp_path / str(troca))
        _leituras_em_dia(janela)
        janela.leitura.botoes[CHAVE_DO_PALPITE].click()
        _aceita(janela, CHAVE_DO_PALPITE)
        _termina(janela, CHAVE_DO_PALPITE, {**PALPITE_DE_A, "correction_proposta": "serrf"})
        anterior = janela.leitura.recomendacao_guardada()
        assert anterior is not None and anterior["correction_proposta"] == "serrf", "âncora"
        assert janela._palpite_em_dia is not None, "âncora: o Palpite aceito não ficou em dia"

        janela.leitura.botoes[CHAVE_DO_PALPITE].click()
        _aceita(janela, CHAVE_DO_PALPITE)
        if troca:
            _escolhe(janela, escolha, "b")
        _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))
        if not troca:
            segundo = janela.leitura.recomendacao_guardada()
            assert segundo is not None and segundo["correction_proposta"] == "loess", (
                "sem a troca, o segundo Palpite não virou a proposta guardada"
            )
            assert janela._palpite_em_dia is not None
            continue

    selo = janela.leitura.cartoes[CHAVE_DO_PALPITE].selo.text()
    assert selo == "Descartado", f"o selo do Palpite descartado não diz «Descartado»: {selo!r}"
    assert janela.leitura.recomendacao_guardada() is None, (
        "a proposta do Palpite anterior ficou guardada depois de o cartão dizer «Descartado»"
    )
    assert janela._palpite_em_dia is None, (
        "o Palpite anterior ficou «em dia»: de volta àqueles arquivos, as sugestões entrariam "
        "sem medir, com o cartão a dizer «Descartado»"
    )


@pytest.mark.parametrize("pasta", ["b", "a"], ids=["outro-par", "o-mesmo-par"])
def test_o_palpite_descartado_larga_a_espera_das_sugestoes_e_o_dos_mesmos_arquivos_aplica(
    qtbot: QtBot, tmp_path: Path, pasta: str
) -> None:
    """«Seguir sugestões do app» espera o Palpite; descartado, a espera não fica presa.

    **Par causal:** escolher os MESMOS arquivos durante o Palpite
    não muda a matriz — o Palpite é aceito e as sugestões entram no seletor. Com B, o Palpite de A é
    descartado, a espera é largada com a frase de sempre e a trava das correções solta.

    Era `test_o_palpite_descartado_larga_o_disparo_delegado_e_o_dos_mesmos_arquivos_retoma`: com
    «Automático», o clique da Correção esperava o Palpite e a correção retomada saía com os mesmos
    arquivos. A caixa saiu; quem espera é o botão das sugestões, e o que entra é o corretor no
    seletor, sem correção nenhuma.

    A Validação e o Diagnóstico de A entram em dia antes do clique
    (`_leituras_em_dia`): o clique mede o Palpite na hora, como media, e a troca cai durante ele.
    """
    janela, escolha, rotas = _janela(qtbot, tmp_path)
    bloco = janela.pagina_da_correcao.sugestoes
    assert bloco is not None
    janela.pipeline.corretor.setCurrentText("serrf")
    _leituras_em_dia(janela)
    bloco.botao.click()
    assert rotas == [ROTA_DO_PALPITE], f"âncora: o clique não pediu só o Palpite: {rotas}"
    assert janela._depois_do_palpite is not None, "âncora: o clique não ficou à espera do Palpite"
    _aceita(janela, CHAVE_DO_PALPITE)
    _escolhe(janela, escolha, pasta)
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))

    assert janela._depois_do_palpite is None, "a retomada ficou guardada"
    assert janela._motivo_para_nao_corrigir("execucao") is None, "a trava ficou presa"
    assert ROTA_DA_ETAPA[1] not in rotas, "seguir as sugestões disparou a Correção"
    assert bloco.botao.isEnabled(), "o botão das sugestões ficou desligado"
    if pasta == "a":
        assert janela.pipeline.corretor.currentText() == PALPITE_DE_A["correction_proposta"], (
            "com os mesmos arquivos, o corretor do Palpite não entrou"
        )
        assert bloco.desfecho.text().startswith("Sugestões aplicadas: corretor «LOESS»")
        return
    assert janela.pipeline.corretor.currentText() == "serrf", "o Palpite de A entrou sobre B"
    assert bloco.desfecho.text() == ENTRADA_MUDOU_DURANTE_O_PALPITE


# ---------------------------------------------------------------------------
# A validação e o diagnóstico: a etiqueta de geração, e o MODO que ela não contava
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chave", "rota"),
    [("validacao", "/matrizes/validar"), ("diagnostico", "/matrizes/diagnosticar")],
)
@pytest.mark.parametrize(
    "modo_na_chegada",
    ["pos_hilic", None, MODO],
    ids=["outro-modo", "mesmo-modo", "volta-ao-mesmo-modo"],
)
def test_a_leitura_pedida_num_modo_nao_chega_depois_da_troca_de_modo(
    qtbot: QtBot, tmp_path: Path, chave: str, rota: str, modo_na_chegada: str | None
) -> None:
    """O modo é parte da matriz (`chave_da_entrada`), e a etiqueta de geração não o contava.

    A leitura sai etiquetada com a geração dos arquivos (`entrada-N`), e a resposta de uma geração
    velha é descartada em `_recebe`. A geração subia na troca e na remoção de arquivo, e não na do
    modo — e o diagnóstico usa o modo: os limiares dele são por modo. **Par causal:** sem a
    troca de modo, a mesma resposta chega ao painel.

    **E a terceira metade é a VOLTA:** o combo emite a cada tecla, e passar por `neg_hili` e
    voltar a `neg_hilic` não é outra matriz na chegada. Com a troca do modo contada como outra
    geração, as duas trocas descartavam a leitura em voo da mesma matriz; a etiqueta leva a matriz
    pelo valor (`_etiqueta_da_leitura`).

    **A troca acontece entre a chamada e a entrega**, na thread principal, antes de o laço de
    eventos correr: o sinal da piscina é enfileirado para esta thread, e só a entrega lê a geração.
    """
    corpo = _corpo_da_rota(rota)
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    recebidas = QSignalSpy(janela.resposta_recebida)
    assert janela.leitura.rotulos[chave].text() == SEM_LEITURA, "âncora: nasceu com leitura"

    janela.enfileira(lambda: _resposta(200, corpo), chave=chave)
    if modo_na_chegada is not None:
        janela.arquivos.modo.setEditText(MODO[:-1])
        janela.arquivos.modo.setEditText(modo_na_chegada)
    assert janela.espera_a_piscina(5_000), "a chamada não voltou"
    QApplication.processEvents()

    if modo_na_chegada != "pos_hilic":
        assert recebidas.count() == 1, (
            f"a leitura da matriz da tela não chegou, com o modo {modo_na_chegada!r} na chegada"
        )
        assert janela.leitura.rotulos[chave].text() != SEM_LEITURA
        return
    assert recebidas.count() == 0, "a leitura do modo anterior foi entregue à interface"
    assert janela.leitura.rotulos[chave].text() == SEM_LEITURA


# ---------------------------------------------------------------------------
# A troca que VOLTA: a medição e a curva escondidas, e repostas pela identidade
# ---------------------------------------------------------------------------


def _mede(janela: JanelaPrincipal, corpo: Any = None) -> str:
    """Medir pelo botão, com o `202` e o desfecho pela releitura. Devolve o rótulo escrito."""
    janela.pipeline.medir_botao.click()
    _aceita(janela, CHAVE_DA_MEDICAO)
    _termina(janela, CHAVE_DA_MEDICAO, _medicao(vencedor="loess") if corpo is None else corpo)
    rotulo = janela.pipeline.rotulo_da_medicao.text()
    assert "**loess**" in rotulo, f"âncora: o vencedor não foi escrito: {rotulo!r}"
    assert janela.graficos._medicao is not None, "âncora: a medição não ficou guardada"
    return rotulo


@pytest.mark.parametrize("volta", [True, False], ids=["volta-ao-mesmo-modo", "fica-noutro-modo"])
def test_corrigir_o_modo_tecla_a_tecla_e_voltar_ao_mesmo_nao_custa_a_medicao(
    qtbot: QtBot, tmp_path: Path, volta: bool
) -> None:
    """O combo do modo emite a cada TECLA, e cada tecla é uma troca de matriz.

    Apagar a medição na troca fazia quem corrigisse o modo à mão perdê-la, com a tela de volta no
    mesmo modo: as barras e o vencedor saíam, e o rótulo dizia que a medição ainda não correu. A
    medição custa minutos. **Par causal:** parar noutro modo esconde as duas metades; voltar ao
    mesmo texto repõe as duas.

    **As teclas vão pelo editor do combo**, como a pessoa as escreve, e não por `setEditText`: é o
    caminho em que o sinal sai uma vez por tecla.
    """
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    prosa_de_a = _mede(janela)
    medicao_de_a = janela.graficos._medicao

    editor = janela.arquivos.modo.lineEdit()
    assert editor is not None, "âncora: o combo do modo não é editável"
    QTest.keyClick(editor, Qt.Key.Key_End)
    QTest.keyClick(editor, Qt.Key.Key_Backspace)
    assert janela.arquivos.modo.currentText() == MODO[:-1], "âncora: a tecla não chegou ao combo"
    if volta:
        QTest.keyClicks(editor, MODO[-1])
        assert janela.arquivos.modo.currentText() == MODO, "âncora: o modo não voltou"

    rotulo = janela.pipeline.rotulo_da_medicao.text()
    if volta:
        assert rotulo == prosa_de_a, f"de volta ao mesmo modo, o vencedor sumiu: {rotulo!r}"
        assert janela.graficos._medicao == medicao_de_a, "de volta ao mesmo modo, as barras sumiram"
        return
    assert rotulo == SEM_MEDICAO, f"noutro modo, o vencedor de {MODO} ficou escrito: {rotulo!r}"
    assert janela.graficos._medicao is None, f"noutro modo, as barras de {MODO} ficaram"


def test_escolher_B_e_voltar_a_A_repoe_a_medicao_de_A(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A→B→A: com B, nem barras nem vencedor de A; de volta a A, as duas metades voltam juntas.

    Antes desta decisão as barras de A eram aceitas sob B, porque o painel só aprendia a matriz no
    disparo seguinte; e a primeira versão do conserto apagava-as na troca, e a volta a A as perdia.
    """
    janela, escolha, _rotas = _janela(qtbot, tmp_path)
    prosa_de_a = _mede(janela)
    medicao_de_a = janela.graficos._medicao

    _escolhe(janela, escolha, "b")
    assert janela.pipeline.rotulo_da_medicao.text() == SEM_MEDICAO, "o vencedor de A ficou sob B"
    assert janela.graficos._medicao is None, "as barras de A ficaram sob B"

    _escolhe(janela, escolha, "a")
    assert janela.pipeline.rotulo_da_medicao.text() == prosa_de_a, "de volta a A, o vencedor sumiu"
    assert janela.graficos._medicao == medicao_de_a, "de volta a A, as barras sumiram"


def test_a_recusa_de_A_tambem_volta_com_A(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A recusa é a resposta da medição como o vencedor é: sai sob B e volta com A."""
    janela, escolha, _rotas = _janela(qtbot, tmp_path)
    janela.pipeline.medir_botao.click()
    _aceita(janela, CHAVE_DA_MEDICAO)
    _termina(janela, CHAVE_DA_MEDICAO, {"detail": "RECUSA-DE-A"}, codigo=422)
    recusa = janela.pipeline.rotulo_da_medicao.text()
    assert "RECUSA-DE-A" in recusa, f"âncora: a recusa não foi escrita: {recusa!r}"

    _escolhe(janela, escolha, "b")
    assert janela.pipeline.rotulo_da_medicao.text() == SEM_MEDICAO, "a recusa de A ficou sob B"
    _escolhe(janela, escolha, "a")
    assert janela.pipeline.rotulo_da_medicao.text() == recusa, "de volta a A, a recusa sumiu"


@pytest.mark.parametrize(
    ("codigo", "corpo", "rotulo_de_a"),
    [
        (200, None, ""),
        (422, {"detail": "FEATURE-AUSENTE"}, "A curva não veio: FEATURE-AUSENTE"),
    ],
    ids=["com-curva", "com-recusa"],
)
def test_o_rotulo_e_a_curva_seguem_a_matriz_e_voltam_com_ela(
    qtbot: QtBot, tmp_path: Path, codigo: int, corpo: Any, rotulo_de_a: str
) -> None:
    """O rótulo do botão da curva e a figura seguem a matriz da tela, e a volta repõe os dois.

    **Par causal, em três passos:** escolher outra vez os MESMOS arquivos deixa tudo como está;
    escolher B esvazia o rótulo e esconde a curva; voltar a A repõe os dois. Na recusa, o rótulo é
    a única metade — a curva não veio —, e é ele que a troca tem de esvaziar e a volta repor.
    """
    corpo = corpo_da_fabrica() if corpo is None else corpo
    janela, escolha, _rotas = _janela(qtbot, tmp_path)
    janela.pipeline.ver_curva_do_loess.emit(FEATURE)
    janela._recebe(CHAVE_DA_CURVA_DO_LOESS, _resposta(codigo, corpo))
    tem_curva = codigo == 200
    assert janela.pipeline.rotulo_da_curva.text() == rotulo_de_a, "âncora: o rótulo de A"
    assert (janela.graficos._curva_do_loess is not None) is tem_curva, "âncora: a curva de A"

    _escolhe(janela, escolha, "a")
    assert janela.pipeline.rotulo_da_curva.text() == rotulo_de_a, "os mesmos arquivos o mudaram"
    assert (janela.graficos._curva_do_loess is not None) is tem_curva

    _escolhe(janela, escolha, "b")
    assert janela.pipeline.rotulo_da_curva.text() == "", "o rótulo da curva de A ficou sob B"
    assert janela.graficos._curva_do_loess is None, "a curva de A ficou sob B"

    _escolhe(janela, escolha, "a")
    assert janela.pipeline.rotulo_da_curva.text() == rotulo_de_a, "de volta a A, o rótulo sumiu"
    assert (janela.graficos._curva_do_loess is not None) is tem_curva, "de volta a A, a curva sumiu"


@pytest.mark.parametrize("pede_em_b", [True, False], ids=["outro-pedido-em-B", "sem-pedido-em-B"])
def test_a_curva_reposta_diz_o_corretor_do_pedido_dela(
    qtbot: QtBot, tmp_path: Path, pede_em_b: bool
) -> None:
    """De volta a A, a curva reposta diz o corretor com que A a pediu, e não o do último pedido.

    O corretor é carimbado no disparo, e um pedido feito em B com outro corretor reescreve o
    carimbo. **Par causal:** sem pedido em B, a curva reposta diz o LOESS; com um pedido em B sob o
    SERRF, continua a dizer o LOESS — e a figura só se diz «a que corrigiu o sinal» com ele.
    """
    janela, escolha, _rotas = _janela(qtbot, tmp_path)
    janela.pipeline.corretor.setCurrentText("loess")
    janela.pipeline.ver_curva_do_loess.emit(FEATURE)
    janela._recebe(CHAVE_DA_CURVA_DO_LOESS, _resposta(200, corpo_da_fabrica()))
    figuras = janela.graficos.figuras

    def rotulos() -> set[str]:
        figura = figuras.canvas.figure
        assert figura.axes, "âncora: a figura da curva não foi desenhada"
        return {serie.rotulo for serie in series(figura.axes[0])}

    assert ROTULO_DA_CURVA in rotulos(), "âncora: a curva de A não se disse a do LOESS"

    _escolhe(janela, escolha, "b")
    if pede_em_b:
        janela.pipeline.corretor.setCurrentText("serrf")
        assert janela.pipeline.corretor.currentText() == "serrf", "âncora: o combo não mudou"
        janela.pipeline.ver_curva_do_loess.emit(FEATURE)
    _escolhe(janela, escolha, "a")
    assert janela.graficos._curva_do_loess is not None, "âncora: a curva de A não foi reposta"
    figuras.reavalia()
    vistos = rotulos()
    assert ROTULO_DA_CURVA in vistos and ROTULO_DA_CURVA_DE_OUTRO_CORRETOR not in vistos, (
        f"a curva reposta de A diz o corretor do pedido feito em B: {vistos}"
    )


@pytest.mark.parametrize("de_b", [True, False], ids=["B-tem-a-sua", "B-sem-nada"])
@pytest.mark.parametrize("qual", ["medicao", "curva"])
def test_a_resposta_descartada_de_A_nao_apaga_a_de_B_que_esta_na_tela(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, qual: str, de_b: bool
) -> None:
    """B tem a sua resposta na tela, e a de A — pedida enquanto A estava escolhido — chega depois.

    **Par causal:** sem resposta de B, o rótulo diz que a de A foi descartada; com a de B, as barras
    (ou a curva) de B continuam, e o rótulo também — a frase do descarte ao lado das barras de B
    diria que foram elas as descartadas.
    """
    janela, escolha, _rotas = _janela(qtbot, tmp_path)
    if qual == "medicao":
        rotulo, visivel, frase = (
            janela.pipeline.rotulo_da_medicao,
            lambda: janela.graficos._medicao,
            FRASE_DA_MEDICAO_DESCARTADA,
        )
    else:
        rotulo, visivel, frase = (
            janela.pipeline.rotulo_da_curva,
            lambda: janela.graficos._curva_do_loess,
            FRASE_DA_CURVA_DESCARTADA,
        )

    def pede() -> None:
        if qual == "medicao":
            janela.pipeline.medir_botao.click()
            _aceita(janela, CHAVE_DA_MEDICAO)
        else:
            janela.pipeline.ver_curva_do_loess.emit(FEATURE)

    def chega() -> None:
        if qual == "medicao":
            _termina(janela, CHAVE_DA_MEDICAO, _medicao(vencedor="loess"))
        else:
            janela._recebe(CHAVE_DA_CURVA_DO_LOESS, _resposta(200, corpo_da_fabrica()))

    if de_b:
        _escolhe(janela, escolha, "b")
        pede()
        chega()
        assert visivel() is not None, "âncora: a resposta de B não ficou na tela"
    texto_de_b, da_tela_de_b = rotulo.text(), visivel()
    _escolhe(janela, escolha, "a")
    pede()
    _escolhe(janela, escolha, "b")
    assert (visivel() is not None) is de_b, "âncora: a volta a B não repôs o que B tinha"
    chega()

    if not de_b:
        assert rotulo.text() == frase, (
            f"o rótulo não diz que a de A foi descartada: {rotulo.text()!r}"
        )
        assert visivel() is None
        return
    assert visivel() == da_tela_de_b, "a resposta descartada de A apagou a de B, que estava na tela"
    assert rotulo.text() == texto_de_b, (
        f"o rótulo de B foi trocado pela frase do descarte de A: {rotulo.text()!r}"
    )


@pytest.mark.parametrize("modo_na_chegada", ["pos_hilic", MODO], ids=["outro-modo", "mesmo-modo"])
def test_a_curva_pedida_num_modo_e_descartada_depois_da_troca_so_do_modo(
    qtbot: QtBot, tmp_path: Path, modo_na_chegada: str
) -> None:
    """Sem trocar arquivo nenhum: o modo muda, e a curva pedida no anterior é descartada.

    **E a frase diz o modo**, porque é ele que mudou: ela dizia *«depois da troca de arquivos»*
    com os mesmos arquivos na tela. **Par causal:** com o mesmo modo escrito outra vez, a curva é
    aceita.
    """
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    janela.pipeline.ver_curva_do_loess.emit(FEATURE)
    janela.arquivos.modo.setEditText(modo_na_chegada)
    janela._recebe(CHAVE_DA_CURVA_DO_LOESS, _resposta(200, corpo_da_fabrica()))
    if modo_na_chegada == MODO:
        assert janela.graficos._curva_do_loess is not None, "a curva do mesmo modo foi descartada"
        assert janela.pipeline.rotulo_da_curva.text() == ""
        return
    assert janela.graficos._curva_do_loess is None, "a curva do modo anterior foi guardada"
    assert janela.pipeline.rotulo_da_curva.text() == FRASE_DA_CURVA_DESCARTADA
    assert "modo" in FRASE_DA_CURVA_DESCARTADA, (
        f"a frase do descarte não diz que o modo mudou: {FRASE_DA_CURVA_DESCARTADA!r}"
    )


# ---------------------------------------------------------------------------
# A CONTA da matriz é uma só: a guarda do que se mostra e a da retomada concordam
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modo_na_chegada", [f"{MODO} ", "pos_hilic"], ids=["espaco", "outro"])
def test_um_espaco_no_fim_do_modo_nao_e_outra_matriz_para_nenhuma_guarda_do_palpite(
    qtbot: QtBot, tmp_path: Path, modo_na_chegada: str
) -> None:
    """`campos_para_envio` manda o modo sem os espaços das pontas: o servidor vê UMA matriz.

    A guarda do que se mostra contava o modo sem eles, e a da retomada com eles: com um espaço no
    fim, a primeira aceitava o Palpite — guardado, selo de confiança — e a segunda largava a espera
    por «entrada mudada», no mesmo instante. **Par causal:** outro modo de facto descarta o Palpite
    e larga a espera; o espaço não faz nem uma coisa nem outra. A espera é a de «Seguir
    sugestões do app», e não mais a do disparo delegado da Correção.

    A Validação e o Diagnóstico entram em dia antes do clique
    (`_leituras_em_dia`), e o Palpite sai no clique: a troca do modo cai durante ele.
    """
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    bloco = janela.pagina_da_correcao.sugestoes
    assert bloco is not None
    janela.pipeline.corretor.setCurrentText("serrf")
    _leituras_em_dia(janela)
    bloco.botao.click()
    assert janela._depois_do_palpite is not None, "âncora: o clique não ficou à espera do Palpite"
    _aceita(janela, CHAVE_DO_PALPITE)
    janela.arquivos.modo.setEditText(modo_na_chegada)
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))

    if modo_na_chegada.strip() == MODO:
        assert janela.leitura.recomendacao_guardada() is not None, "o Palpite foi descartado"
        assert janela.pipeline.corretor.currentText() == "loess", (
            f"a retomada largou a espera com um espaço no fim do modo: {bloco.desfecho.text()!r}"
        )
        return
    assert janela.leitura.recomendacao_guardada() is None, "o Palpite de outro modo foi guardado"
    assert janela.pipeline.corretor.currentText() == "serrf", "a sugestão entrou noutro modo"
    assert bloco.desfecho.text() == ENTRADA_MUDOU_DURANTE_O_PALPITE


# ---------------------------------------------------------------------------
# A execução (`/execucoes`): o desfecho de A não enche as abas sob B
# ---------------------------------------------------------------------------


def _executa_a(qtbot: QtBot, tmp_path: Path, pasta: str) -> tuple[JanelaPrincipal, dict[str, str]]:
    """Executar A pelo botão da Análise, o `202` do aceite, escolher `pasta`, e só então o `200`.

    **Pelo botão «3. Analisar e registrar»** desde 09/11/2026: o «Executar pipeline», que este
    arnês clicava, saiu com a aba «Executar e exportar». A Análise não espera o Palpite — só a
    Correção delega —, então o clique chega a `/execucoes` direto. Devolve a janela e a aba de
    Qualidade em repouso.

    **A figura é uma das da Análise**, e não o painel da correção que o pipeline também escrevia:
    sobre a matriz original a Análise não reproduz etapa nenhuma, e cada etapa é dona das próprias
    figuras — o painel da correção nem entraria em «ainda corre».
    """
    janela, escolha, rotas = _janela(qtbot, tmp_path)
    figuras = janela.graficos.figuras
    assert figuras.seleciona("figura_do_dendrograma"), "âncora: a figura não existe"
    repouso = dict(janela.qualidade.texto_visivel())
    assert repouso == dict(PainelDeQualidade().texto_visivel()), "âncora: a aba nasceu cheia"

    janela.pipeline.botoes_de_etapa[3].click()
    assert ROTA_DA_EXECUCAO in rotas, "âncora: o clique não chegou a `/execucoes`"
    janela._recebe(
        CHAVE_DA_EXECUCAO,
        _resposta(202, {"identificador": IDENTIFICADOR, "total": 3, "reclamado": False}),
    )
    assert CHAVE_DA_EXECUCAO in janela._esperas, "âncora: o `202` do aceite não abriu a espera"
    assert figuras.repouso.text() == texto_do_estado(EstadoDaFigura.EM_CURSO), (
        "âncora: o `202` não pôs a figura em «ainda corre», e o repouso abaixo seria vácuo"
    )
    _escolhe(janela, escolha, pasta)
    _termina(janela, CHAVE_DA_EXECUCAO, _com_qualidade(0.1799))
    return janela, repouso


@pytest.mark.parametrize("pasta", ["b", "a"], ids=["outro-par", "o-mesmo-par"])
def test_o_desfecho_da_execucao_de_A_nao_enche_a_qualidade_sob_B(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, pasta: str
) -> None:
    """A única guarda do ramo da execução era a de supersessão, e o `200` de A enchia as abas sob B.

    **Par causal:** com os mesmos arquivos escolhidos outra vez, o `200` enche a aba de Qualidade;
    com B, ela fica como estava, a figura sai de «a execução ainda corre» para o repouso, a espera e
    a trava soltam, e o rótulo do progresso diz porquê.

    > O porquê mudou: o desfecho de A fica guardado, e a frase diz onde ele aparece. A
    > figura sai de «ainda corre» na TROCA, e não na chegada: o `202` de A sob B já não a toca.
    """
    janela, repouso = _executa_a(qtbot, tmp_path, pasta)
    qualidade = dict(janela.qualidade.texto_visivel())
    figuras = janela.graficos.figuras
    if pasta == "a":
        assert qualidade != repouso, "com os mesmos arquivos, o `200` não encheu a aba de Qualidade"
        assert janela.pipeline.rotulo_de_progresso.text() != FRASE_DA_EXECUCAO_GUARDADA
        return
    assert qualidade == repouso, "os números da execução de A encheram a aba de Qualidade sob B"
    assert janela.pipeline.rotulo_de_progresso.text() == FRASE_DA_EXECUCAO_GUARDADA
    assert CHAVE_DA_EXECUCAO not in janela._esperas, "a espera da execução de A ficou aberta"
    assert janela._motivo_para_nao_corrigir("execucao") is None, "a trava ficou presa"
    assert figuras.repouso.text() == texto_do_estado(EstadoDaFigura.SEM_EXECUCAO), (
        f"a figura ficou em {figuras.repouso.text()!r} depois do descarte"
    )


# ---------------------------------------------------------------------------
# O aviso de desfecho segue a guarda: a resposta descartada não o abre nem o fecha
# ---------------------------------------------------------------------------
#
# `_recebe` chamava `mostra_resposta` ANTES de `_encaminha`, onde as guardas decidem: a recusa de A
# descartada sob B abria «Não foi possível concluir» com o texto dela, ao mesmo tempo que o painel
# de quem pediu dizia que ela fora descartada; e o `200` de A descartado recolhia o aviso aberto de
# B, porque um `2xx` recolhe o aviso anterior. **A escolha é a que o código sustenta: não
# abrir.** O painel já diz que a resposta foi descartada, e o código escolheu descartar, e não
# guardar e avisar.

_RECUSA_DE_A = "RECUSA-DE-A: a resposta é da matriz A"


def _recusa_de_a(codigo: int) -> Resposta:
    """Uma recusa com a mensagem que o aviso escreveria — e é ela que não pode aparecer sob B."""
    return Resposta(
        codigo=codigo,
        ok=False,
        causa="recusa",
        mensagem=_RECUSA_DE_A,
        corpo={"detail": _RECUSA_DE_A},
        conteudo=None,
    )


def _pede_com_a(janela: JanelaPrincipal, qual: str, rotas: list[str]) -> str:
    """O disparo de `qual` com A na tela, pelo caminho de produção; devolve a chave do desfecho."""
    if qual == "medicao":
        janela.pipeline.medir_botao.click()
        _aceita(janela, CHAVE_DA_MEDICAO)
        chave = chave_do_pedido(CHAVE_DA_MEDICAO, IDENTIFICADOR)
    elif qual == "curva":
        janela.pipeline.ver_curva_do_loess.emit(FEATURE)
        chave = CHAVE_DA_CURVA_DO_LOESS
    elif qual == "palpite":
        # Com a Validação e o Diagnóstico em dia, o clique pede só o Palpite,
        # que é o disparo carimbado com A; sem eles, a cadeia validaria primeiro.
        _leituras_em_dia(janela)
        janela.leitura.botoes[CHAVE_DO_PALPITE].click()
        _aceita(janela, CHAVE_DO_PALPITE)
        chave = chave_do_pedido(CHAVE_DO_PALPITE, IDENTIFICADOR)
    else:
        # A execução é a Análise, como em `_executa_a`: ela não espera o Palpite.
        janela.pipeline.botoes_de_etapa[3].click()
        janela._recebe(
            CHAVE_DA_EXECUCAO,
            _resposta(202, {"identificador": IDENTIFICADOR, "total": 3, "reclamado": False}),
        )
        chave = chave_do_pedido(CHAVE_DA_EXECUCAO, IDENTIFICADOR)
    assert rotas, f"âncora: o disparo de `{qual}` não saiu"
    return chave


_O_PAINEL_DIZ_QUE_DESCARTOU = {
    "medicao": lambda j: j.pipeline.rotulo_da_medicao.text() == FRASE_DA_MEDICAO_DESCARTADA,
    "curva": lambda j: j.pipeline.rotulo_da_curva.text() == FRASE_DA_CURVA_DESCARTADA,
    "palpite": lambda j: j.leitura.rotulos[CHAVE_DO_PALPITE].text() == FRASE_DO_PALPITE_DESCARTADO,
    "execucao": lambda j: j.pipeline.rotulo_de_progresso.text() == FRASE_DA_EXECUCAO_GUARDADA,
}


@pytest.mark.parametrize("pasta", ["b", "a"], ids=["outro-par", "o-mesmo-par"])
@pytest.mark.parametrize("qual", ["medicao", "curva", "palpite", "execucao"])
def test_d681_a_recusa_de_A_descartada_sob_B_nao_abre_o_aviso_de_desfecho(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, qual: str, pasta: str
) -> None:
    """A recusa de A, chegada com B na tela, não abre «Não foi possível concluir».

    **Par causal:** com os mesmos arquivos escolhidos outra vez, a recusa é aceita, e o aviso abre
    com o texto dela — é a metade que prova que a outra não passa por o aviso nunca abrir. A
    âncora da metade de B é o painel de quem pediu a dizer que a resposta foi descartada: sem ela,
    o aviso fechado podia ser só a recusa a não ter chegado.
    """
    janela, escolha, rotas = _janela(qtbot, tmp_path)
    chave = _pede_com_a(janela, qual, rotas)
    aviso = janela.aviso_de_desfecho
    assert aviso.isHidden(), "âncora: um aviso já estava aberto antes de a recusa chegar"
    _escolhe(janela, escolha, pasta)
    janela._recebe(chave, _recusa_de_a(500 if qual == "execucao" else 422))

    if pasta == "a":
        assert not aviso.isHidden(), f"âncora: a recusa de `{qual}` com A na tela não abriu o aviso"
        assert aviso.titulo.text() == "Não foi possível concluir"
        return
    assert _O_PAINEL_DIZ_QUE_DESCARTOU[qual](janela), (
        f"âncora: o painel de `{qual}` não diz que a resposta de A foi descartada"
    )
    assert aviso.isHidden(), (
        f"a recusa de `{qual}` de A, descartada sob B, abriu o aviso: {aviso.rotulo.text()!r}"
    )
    assert _RECUSA_DE_A not in aviso.rotulo.text(), (
        "a recusa descartada ficou registada como o último desfecho"
    )


@pytest.mark.parametrize("pasta", ["b", "a"], ids=["outro-par", "o-mesmo-par"])
def test_d681_o_200_de_A_descartado_nao_recolhe_o_aviso_aberto_de_B(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, pasta: str
) -> None:
    """O `2xx` recolhe o aviso anterior — e o de A, descartado sob B, não recolhe o de B.

    **Par causal:** com os mesmos arquivos, o `200` da medição é aceito e recolhe o aviso, como
    todo `2xx`; com B, o aviso que B abriu continua aberto com o texto dele.

    **O alcance é estreito, e vai dito:** o `202` de cada sondagem de A, que não é descartado,
    recolhe o mesmo aviso um segundo depois (o teste do `202`, abaixo). O que se protege aqui é a
    janela entre o último `202` de A e o desfecho dele.
    """
    janela, escolha, rotas = _janela(qtbot, tmp_path)
    chave = _pede_com_a(janela, "medicao", rotas)
    _escolhe(janela, escolha, pasta)
    janela.mostra_resposta(
        Resposta(codigo=422, ok=False, causa="recusa", mensagem="ERRO-DE-B", corpo=None)
    )
    aviso = janela.aviso_de_desfecho
    assert not aviso.isHidden() and "ERRO-DE-B" in aviso.rotulo.text(), "âncora: o aviso não abriu"
    janela._recebe(chave, _resposta(200, _medicao(vencedor="loess")))

    if pasta == "a":
        assert aviso.isHidden(), "âncora: o `200` aceito não recolheu o aviso"
        return
    assert janela.pipeline.rotulo_da_medicao.text() == FRASE_DA_MEDICAO_DESCARTADA, "âncora"
    assert not aviso.isHidden(), "o `200` de A, descartado sob B, recolheu o aviso aberto de B"
    assert "ERRO-DE-B" in aviso.rotulo.text(), (
        f"o último desfecho passou a ser o de A: {aviso.rotulo.text()!r}"
    )


@pytest.mark.parametrize("desistiu", [True, False], ids=["desistiu", "esperava"])
def test_d681_a_recusa_de_uma_espera_encerrada_nao_abre_o_aviso(
    qtbot: QtBot, tmp_path: Path, desistiu: bool
) -> None:
    """A irmã pela guarda de supersessão: a resposta de uma espera que já acabou chega calada.

    O ramo da medição diz que a resposta de uma cadeia que já não é a da espera *«é descartada em
    silêncio»* — clique novo, desistência ou fecho —, e o aviso abria-se com ela, porque
    `mostra_resposta` corria antes dessa guarda também. **Par causal:** com a espera aberta, a
    mesma recusa é aceita e abre o aviso.
    """
    janela, _escolha, rotas = _janela(qtbot, tmp_path)
    chave = _pede_com_a(janela, "medicao", rotas)
    if desistiu:
        assert janela.para_a_espera(CHAVE_DA_MEDICAO), "âncora: não havia espera a encerrar"
    janela._recebe(chave, _recusa_de_a(422))

    aviso = janela.aviso_de_desfecho
    if not desistiu:
        assert not aviso.isHidden(), "âncora: a recusa da espera aberta não abriu o aviso"
        return
    assert aviso.isHidden(), (
        f"a recusa de uma espera encerrada abriu o aviso: {aviso.rotulo.text()!r}"
    )


def test_d681_o_202_de_A_sob_B_continua_a_passar_pelo_aviso(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O `202` que mantém a espera de A passa pelo aviso como antes, com B na tela.

    `_encaminha` não o descarta — o trabalho de A corre de facto, e a trava conta-o —, e a
    pergunta feita antes do aviso tem de dar a mesma resposta: uma conta que o desse por descartado
    divergiria do ramo, e é essa divergência que a primeira metade acusa. **A âncora de que a matriz
    da tela é mesmo outra** é o painel da medição a dizer que o desfecho de A, a seguir, foi
    descartado; ela vale também na árvore anterior ao conserto, como a primeira metade.

    **A última asserção é a do conserto, e não âncora:** o desfecho de A sob B não passa pelo aviso.
    Na árvore anterior ao conserto o teste reprova nela, e só nela.

    **E passar pelo aviso tem um preço, que `mostradas == [202]` fixa junto.** Um `2xx` recolhe o
    aviso anterior, então o `202` de cada sondagem de A — um por segundo — recolhe também um
    aviso de erro aberto por B, ou por qualquer outro pedido, com ou sem troca de matriz. A
    proteção que o teste do `200` afirma vale, por isso, só entre o último `202` de A e o desfecho.
    """
    janela, escolha, rotas = _janela(qtbot, tmp_path)
    chave = _pede_com_a(janela, "medicao", rotas)
    _escolhe(janela, escolha, "b")
    mostradas: list[int] = []
    original = janela.mostra_resposta

    def espia(resposta: Resposta) -> str:
        mostradas.append(resposta.codigo)
        return original(resposta)

    monkeypatch.setattr(janela, "mostra_resposta", espia)
    janela._recebe(chave, _resposta(202, {"identificador": IDENTIFICADOR, "total": 3}))
    assert mostradas == [202], f"o `202` de A sob B deixou de passar pelo aviso: {mostradas}"
    assert CHAVE_DA_MEDICAO in janela._esperas, "o `202` de A fechou a espera dele"

    janela._recebe(chave, _recusa_de_a(422))
    assert _O_PAINEL_DIZ_QUE_DESCARTOU["medicao"](janela), (
        "âncora: o painel da medição não diz que o desfecho de A foi descartado"
    )
    assert mostradas == [202], f"o desfecho de A, descartado sob B, passou pelo aviso: {mostradas}"


# ---------------------------------------------------------------------------
# A execução de A guardada e reposta pela identidade da matriz
# ---------------------------------------------------------------------------
#
# **Guardar e repor ao voltar:** o desfecho de
# uma execução de A que chega com B na tela fica guardado pela identidade de A, e aparece quando a
# pessoa volta a A; e as abas que mostram a execução de A, na troca para B, passam a mostrar a de
# B guardada ou o repouso. **O oráculo é o outro caminho**: o que a janela mostra de volta
# a A é comparado com o que ela mostra quando o mesmo desfecho chega com A na tela, e não com
# valores escritos aqui. **Os corpos vêm da rota** `/execucoes` sobre o par da factory, com os
# arquivos e os campos que o clique mandou, e não de um dicionário.
#
# > Estes testes nasceram contra o botão «Executar
# > pipeline», que saiu com a aba «Executar e exportar». A execução é a Análise, disparada pelo
# > botão dela — «3. Analisar e registrar» —, e um `200` conclui só ela. **E a tela é lida em DUAS
# > figuras, e não numa.** A primeira versão lia o painel da correção, que o pipeline escrevia
# > sempre; sob as etapas independentes cada etapa é dona das próprias figuras, a Análise sobre a
# > original não o toca e ele nem entra em «ainda corre». A figura da execução
# > passa a ser uma das da Análise, que segue a chave dela; o painel da correção continua lido,
# > porque é nele que as prévias — e a Análise que reproduz a Correção — escrevem.

_CORPOS_REAIS: dict[str, dict[str, Any]] = {}
_FIGURA_DA_EXECUCAO = "figura_da_anova"
_FIGURA_DA_CORRECAO = "figura_do_painel_da_correcao"


def _janela_que_guarda_o_pedido(
    qtbot: QtBot, tmp_path: Path
) -> tuple[JanelaPrincipal, _Escolha, list[tuple[str, dict[str, Any]]]]:
    """`_janela`, com o espião a guardar também os arquivos e os campos de cada pedido."""
    janela, escolha, _rotas = _janela(qtbot, tmp_path)
    pedidos: list[tuple[str, dict[str, Any]]] = []
    janela.chama = lambda _metodo, rota, **kw: pedidos.append((rota, kw))  # type: ignore[method-assign]
    assert janela.graficos.figuras.seleciona(_FIGURA_DA_EXECUCAO), "âncora: a figura não existe"
    return janela, escolha, pedidos


def _corpo_real(kw: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """O `200` de `/execucoes` com os arquivos e os campos do clique, pela rota e um SQLite novo.

    Guardado por conteúdo: o mesmo pedido devolve o mesmo corpo, e cada um custa de meio a três
    segundos pela rota com um SQLite novo — medido na factory, e com o par de grupos o mais caro.
    """
    arquivos = kw["arquivos"]
    campos = kw["campos"]
    conteudo = repr((sorted((k, v[1]) for k, v in arquivos.items()), sorted(campos.items())))
    chave = hashlib.sha256(conteudo.encode("utf-8")).hexdigest()
    if chave not in _CORPOS_REAIS:
        with pytest.MonkeyPatch.context() as remendo:
            url = f"sqlite:///{tmp_path / f'refino-{chave[:8]}.db'}"
            remendo.delenv("REFINO_DB_HOST", raising=False)
            remendo.setenv(VARIAVEL_DA_URL, url)
            prepara_sqlite(url)
            METADATA.create_all(create_engine(url))
            servidor = TestClient(build_app(), raise_server_exceptions=False)
            aceite = servidor.post(ROTA_DA_EXECUCAO, files=arquivos, data=campos)
            assert aceite.status_code == 202, aceite.text
            identificador = aceite.json()["identificador"]
            for _ in range(2400):
                lida = servidor.get(rota_do_progresso(identificador))
                if lida.status_code != 202:
                    break
                time.sleep(0.05)
            assert lida.status_code == 200, lida.text
            _CORPOS_REAIS[chave] = lida.json()
    return copy.deepcopy(_CORPOS_REAIS[chave])


def _dispara(
    janela: JanelaPrincipal, pedidos: list[tuple[str, dict[str, Any]]], tmp_path: Path
) -> dict[str, Any]:
    """A Análise pelo botão dela e o `202` do aceite. Devolve o corpo real do `200`.

    > Era «Executar pipeline» pelo botão, com o Palpite posto em dia
    > antes, porque o «Automático» o esperava. O botão saiu com a aba «Executar e exportar»;
    > a Análise parte da entrada escolhida e não espera o Palpite — só a Correção delega.
    """
    pedidos.clear()
    janela.pipeline.botoes_de_etapa[ETAPAS[-1]].click()
    disparos = [kw for rota, kw in pedidos if rota == ROTA_DA_EXECUCAO]
    assert len(disparos) == 1, f"âncora: o clique não chegou a `/execucoes`: {pedidos}"
    janela._recebe(
        CHAVE_DA_EXECUCAO,
        _resposta(202, {"identificador": IDENTIFICADOR, "total": 8, "reclamado": False}),
    )
    assert CHAVE_DA_EXECUCAO in janela._esperas, "âncora: o `202` do aceite não abriu a espera"
    return _corpo_real(disparos[0], tmp_path)


def _executa(
    janela: JanelaPrincipal, pedidos: list[tuple[str, dict[str, Any]]], tmp_path: Path
) -> dict[str, Any]:
    """Executar a matriz da tela até o `200` chegar com ela na tela. Devolve o que a tela mostra."""
    corpo = _dispara(janela, pedidos, tmp_path)
    _termina(janela, CHAVE_DA_EXECUCAO, corpo)
    assert CHAVE_DA_EXECUCAO not in janela._esperas, "âncora: o `200` não fechou a espera"
    return _tela_da_execucao(janela)


def _pontos_da_figura(janela: JanelaPrincipal) -> tuple[tuple[float, ...], ...]:
    """As posições dos pontos e das linhas de cada eixo da figura no canvas — o conteúdo dela."""
    conteudo: list[tuple[float, ...]] = []
    for eixo in janela.graficos.figuras.canvas.figure.axes:
        for linha in eixo.lines:
            conteudo.append(tuple(float(v) for v in np.round(linha.get_xydata(), 9).ravel()))
        for colecao in eixo.collections:
            conteudo.append(tuple(float(v) for v in np.round(colecao.get_offsets(), 9).ravel()))
    return tuple(conteudo)


def _figura(janela: JanelaPrincipal, funcao: str) -> tuple[dict[str, str], tuple[Any, ...]]:
    """O estado e os pontos da figura `funcao` nos «Gráficos», escolhida para ser lida."""
    figuras = janela.graficos.figuras
    assert figuras.seleciona(funcao), f"âncora: a figura {funcao} não existe"
    return dict(figuras.estado_visivel()), _pontos_da_figura(janela)


def _motivo_do_botao(janela: JanelaPrincipal, botao: Any) -> str:
    """A frase do motivo de um botão desligado — o rótulo cujo `buddy` é ele —, ou vazio."""
    return "".join(r.text() for r in janela.findChildren(QLabel) if r.buddy() is botao)


def _tela_da_execucao(janela: JanelaPrincipal) -> dict[str, Any]:
    """O que as abas que uma execução alimenta mostram AGORA.

    A aba de Qualidade, duas figuras dos «Gráficos» — o estado e os pontos —, o recorte e o
    botão das vias, o resumo metodológico, e a linha do resultado e a do corretor no painel do
    pipeline. **O progresso não entra**: ele traz o decorrido em segundos, que muda entre dois
    caminhos sem nada mudar na execução.

    > As figuras são duas — a da Análise (`figura`, `pontos`) e o painel
    > da correção (`figura_da_correcao`, `pontos_da_correcao`) —, e a da Análise é lida por último,
    > para a escolhida nos «Gráficos» continuar a ser ela. O resumo é o de «Qualidade do
    > resultado», para onde ele foi quando «Executar e exportar» saiu, e vai com o botão
    > «Copiar» — ligado, ou desligado com a frase do motivo —, porque o repouso do resumo
    > é a caixa E os botões como a construção os deixa.
    """
    figura_da_correcao, pontos_da_correcao = _figura(janela, _FIGURA_DA_CORRECAO)
    figura, pontos = _figura(janela, _FIGURA_DA_EXECUCAO)
    return {
        "qualidade": dict(janela.qualidade.texto_visivel()),
        "figura": figura,
        "pontos": pontos,
        "figura_da_correcao": figura_da_correcao,
        "pontos_da_correcao": pontos_da_correcao,
        "vias": (janela.vias.recorte.text(), janela.vias.botao.isEnabled()),
        "resumo": (
            janela.qualidade.resumo.caixa.toPlainText(),
            janela.qualidade.resumo.copiar.isEnabled(),
            _motivo_do_botao(janela, janela.qualidade.resumo.copiar),
        ),
        "resultado": janela.pipeline.rotulo_de_resultado.text(),
        "corretor": janela.pipeline.rotulo_do_corretor.text(),
    }


def _difere(uma: dict[str, Any], outra: dict[str, Any]) -> list[str]:
    return sorted(chave for chave in uma if uma[chave] != outra[chave])


def _abas(tela: dict[str, Any]) -> dict[str, Any]:
    """`_tela_da_execucao` sem o grupo da execução, que com uma em voo diz o progresso dela.

    O resultado e o corretor são do trabalho que corre enquanto ele corre, qualquer que seja a
    matriz da tela: o `202` do aceite limpa-os, e só o desfecho os reescreve.
    """
    return {chave: valor for chave, valor in tela.items() if chave not in {"resultado", "corretor"}}


@pytest.mark.parametrize("codigo", [200, 500])
def test_o_desfecho_de_A_chegado_sob_B_e_guardado_e_volta_com_A(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, codigo: int
) -> None:
    """O desfecho de A que chega com B na tela não se mostra sob B, e aparece de volta a A.

    **Par causal, e o oráculo é o outro caminho:** uma janela de controle recebe o mesmo desfecho
    com A na tela; a outra escolhe B antes de ele chegar, e volta a A. O que as abas mostram de
    volta a A tem de ser o que o controle mostra — e sob B, o repouso. O `500` entra porque a
    resposta guardada é a última aceita, a recusa incluída, como a medição: com A na tela,
    um erro tira as figuras e limpa a aba de Qualidade, e é isso que a volta a A repõe.

    **E as etapas do passo a passo ficam em dia de volta a A**, porque o resultado está na tela;
    descartar o desfecho deixava-as por fazer.
    """
    controle, escolha_c, pedidos_c = _janela_que_guarda_o_pedido(qtbot, tmp_path / "controle")
    repouso = _tela_da_execucao(controle)
    corpo = _dispara(controle, pedidos_c, tmp_path)
    _escolhe(controle, escolha_c, "a")
    _termina(controle, CHAVE_DA_EXECUCAO, corpo if codigo == 200 else {"detail": "X"}, codigo)
    esperado = _tela_da_execucao(controle)
    assert _difere(esperado, repouso), "âncora: o desfecho com A na tela não mudou aba nenhuma"
    etapas_esperadas = controle._atualiza_as_etapas()
    # **A faixa vermelha da Análise é da matriz da tela**: com A na
    # tela o erro a põe — a âncora —, e o de A chegado sob B não a põe sob B, abaixo.
    assert (ETAPAS[-1] in controle._falhas_das_etapas) is (codigo != 200), (
        "âncora: com A na tela, a faixa vermelha da Análise não segue o desfecho"
    )

    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path / "troca")
    _dispara(janela, pedidos, tmp_path)
    _escolhe(janela, escolha, "b")
    _termina(janela, CHAVE_DA_EXECUCAO, corpo if codigo == 200 else {"detail": "X"}, codigo)

    sob_b = _tela_da_execucao(janela)
    assert sob_b == repouso, f"o desfecho de A mudou sob B: {_difere(sob_b, repouso)}"
    assert janela.pipeline.rotulo_de_progresso.text() == FRASE_DA_EXECUCAO_GUARDADA
    assert CHAVE_DA_EXECUCAO not in janela._esperas, "a espera da execução de A ficou aberta"
    assert janela._motivo_para_nao_corrigir("execucao") is None, "a trava ficou presa"
    assert ETAPAS[-1] not in janela._falhas_das_etapas, "a falha de A pôs a faixa vermelha sob B"

    _escolhe(janela, escolha, "a")
    de_volta = _tela_da_execucao(janela)
    assert de_volta == esperado, (
        f"de volta a A, o desfecho de A não voltou: difere em {_difere(de_volta, esperado)}"
    )
    assert janela._atualiza_as_etapas() == etapas_esperadas, (
        "de volta a A, as etapas não dizem o que dizem com o desfecho chegado com A na tela"
    )


@pytest.mark.parametrize("pasta", ["b", "a"], ids=["outro-par", "o-mesmo-par"])
def test_a_execucao_de_A_ja_mostrada_sai_na_troca_e_volta_com_A(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, pasta: str
) -> None:
    """A metade que era pendência: o que já estava na tela de A ficava sob B.

    **Par causal:** escolher os mesmos arquivos outra vez não muda a matriz, e as abas ficam com a
    execução de A; com B, elas passam ao repouso, e de volta a A, a execução de A volta inteira.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    repouso = _tela_da_execucao(janela)
    de_a = _executa(janela, pedidos, tmp_path)
    assert _difere(de_a, repouso), "âncora: a execução de A não mudou aba nenhuma"

    _escolhe(janela, escolha, pasta)
    depois = _tela_da_execucao(janela)
    if pasta == "a":
        assert depois == de_a, f"os mesmos arquivos tiraram a execução: {_difere(depois, de_a)}"
        return
    assert depois == repouso, f"a execução de A ficou sob B: {_difere(depois, repouso)}"

    _escolhe(janela, escolha, "a")
    de_volta = _tela_da_execucao(janela)
    assert de_volta == de_a, f"de volta a A, a execução não voltou: {_difere(de_volta, de_a)}"


def test_cada_matriz_mostra_a_sua_execucao_na_ida_e_na_volta(qtbot: QtBot, tmp_path: Path) -> None:
    """Uma execução guardada por matriz: A, B, A, B — cada uma volta com os seus arquivos.

    **A âncora é que as duas diferem** nas abas que o teste lê: com a mesma tela para A e
    para B, mostrar a errada passaria.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    de_a = _executa(janela, pedidos, tmp_path)
    _escolhe(janela, escolha, "b")
    de_b = _executa(janela, pedidos, tmp_path)
    assert {"qualidade", "pontos"} <= set(_difere(de_a, de_b)), (
        f"âncora: as execuções de A e de B só diferem em {_difere(de_a, de_b)}"
    )

    _escolhe(janela, escolha, "a")
    assert _tela_da_execucao(janela) == de_a, "de volta a A, a tela não é a da execução de A"
    _escolhe(janela, escolha, "b")
    assert _tela_da_execucao(janela) == de_b, "de volta a B, a tela não é a da execução de B"


@pytest.mark.parametrize("pasta", ["b", "a"], ids=["outro-par", "o-mesmo-par"])
def test_o_progresso_de_A_nao_apaga_as_abas_de_B(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, pasta: str
) -> None:
    """Com a execução de B na tela e a de A a correr, o `202` de A não toca as abas de B.

    **As abas, e não o grupo da execução** (`_abas`): enquanto A corre, o grupo diz o progresso de
    A, e é isso que a trava das correções conta. Cada sondagem de A traz um `202`, um por segundo,
    e ele punha as figuras em «a execução ainda corre» e esvaziava as vias — sob B, isso apagava a
    execução de B que a troca acabou de repor.
    **Par causal:** com A na tela, o mesmo `202` põe a figura em «ainda corre», como sempre. E o
    `202` continua a ler o progresso e a manter a espera, porque o trabalho de A corre.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    _escolhe(janela, escolha, "b")
    de_b = _executa(janela, pedidos, tmp_path)
    _escolhe(janela, escolha, "a")
    _dispara(janela, pedidos, tmp_path)
    _escolhe(janela, escolha, pasta)
    antes = _tela_da_execucao(janela)

    _termina(
        janela,
        CHAVE_DA_EXECUCAO,
        {"identificador": IDENTIFICADOR, "feito": 2, "total": 8, "estado": "executando"},
        codigo=202,
    )
    assert CHAVE_DA_EXECUCAO in janela._esperas, "o `202` de A fechou a espera dele"
    figura = janela.graficos.figuras.repouso.text()
    if pasta == "a":
        assert figura == texto_do_estado(EstadoDaFigura.EM_CURSO), (
            f"âncora: com A na tela, o `202` não pôs a figura em «ainda corre»: {figura!r}"
        )
        return
    assert _abas(antes) == _abas(de_b), (
        f"a troca para B não repôs a execução de B: {_difere(_abas(antes), _abas(de_b))}"
    )
    depois = _abas(_tela_da_execucao(janela))
    assert depois == _abas(de_b), f"o `202` de A apagou a execução de B: {_difere(depois, de_b)}"

    # **E a volta a A, com A ainda a correr, mostra o que a tela de A mostrava**: a figura em
    # «ainda corre», pelo último `202` de A — e não o repouso, que A não tem desfecho guardado.
    _escolhe(janela, escolha, "a")
    figura = janela.graficos.figuras.repouso.text()
    assert figura == texto_do_estado(EstadoDaFigura.EM_CURSO), (
        f"de volta a A em voo, a figura não diz que a execução ainda corre: {figura!r}"
    )


@pytest.mark.parametrize("pasta", ["a", "b"], ids=["os-mesmos-bytes", "outros-bytes"])
def test_remover_as_medidas_tira_a_execucao_e_os_mesmos_bytes_a_repoem(
    qtbot: QtBot, tmp_path: Path, pasta: str
) -> None:
    """Sem a matriz inteira não há execução a mostrar; os mesmos bytes de volta são a mesma matriz.

    **Par causal:** reanexar as medidas de A repõe a execução de A — a identidade é de conteúdo,
    como a da medição; reanexar as de B, com a lista de A, é outra matriz, e as abas ficam
    no repouso.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    repouso = _tela_da_execucao(janela)
    de_a = _executa(janela, pedidos, tmp_path)
    assert _difere(de_a, repouso), "âncora: a execução de A não mudou aba nenhuma"
    janela.arquivos.botoes_de_remocao["medidas"].click()
    assert janela.arquivos.par is None, "âncora: a remoção não desfez o par"
    sem = _tela_da_execucao(janela)
    assert sem == repouso, f"sem as medidas, a execução ficou: {_difere(sem, repouso)}"

    escolha.pasta = pasta
    janela.arquivos.escolhe_medidas()
    assert janela.arquivos.par is not None, "âncora: o par não voltou"
    de_volta = _tela_da_execucao(janela)
    if pasta == "b":
        assert de_volta == repouso, (
            f"outras medidas mostraram a execução de A: {_difere(de_volta, repouso)}"
        )
        return
    assert de_volta == de_a, f"os mesmos bytes não repuseram a execução: {_difere(de_volta, de_a)}"


def _leva_um_arquivo(
    janela: JanelaPrincipal, pedidos: list[tuple[str, dict[str, Any]]], tmp_path: Path
) -> str:
    """A Correção pela prévia e o download dela escrito na pasta. Devolve o resumo do Resultado.

    **O sucessor de «Levar o resultado»** desde que «Executar e exportar» saiu: os
    downloads moram no Resultado da Correção e da Normalização, cada um com o seu resumo, e
    o da Correção pede a Correção feita sobre a matriz da tela.
    """
    _entrega_a_previa(janela, _pede_a_previa(janela, pedidos, "loess"))
    (tmp_path / "levados").mkdir(exist_ok=True)
    resultado = janela.resultado_da_correcao
    resultado._escolhedor = lambda: tmp_path / "levados"
    resultado.pede_e_gera("corrigida")
    janela._escreve_no_destino("corrigida", b"csv")
    return resultado.resumo.caixa.toPlainText()


@pytest.mark.parametrize("levou", [True, False], ids=["com-arquivo-levado", "sem"])
def test_o_resumo_dos_arquivos_levados_nao_segue_a_matriz(
    qtbot: QtBot, tmp_path: Path, levou: bool
) -> None:
    """O resumo dos arquivos levados é de arquivos que existem, e a troca não o toca.

    **Par causal:** sem arquivo levado, a troca para B tira o resumo da execução de A, e o
    Resultado não tem resumo; com um arquivo levado depois, o resumo dele fica sob B.

    > **[Reescrito para as etapas independentes]** A primeira versão afirmava isto sobre
    > UMA caixa, a de «Executar e exportar», que mostrava ora o resumo da execução, ora o dos
    > arquivos levados — e com um arquivo levado a troca não tocava a caixa. A aba saiu: o
    > resumo da Análise mora em «Qualidade do resultado» e o dos arquivos levados no Resultado da
    > etapa que os escreveu. **O que este teste afirma passa a ser as duas metades em duas
    > caixas:** o resumo da execução segue a matriz com ou sem arquivo levado, e o dos levados fica.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    # O repouso do resumo do Resultado tem a frase dele, que não é a da Análise.
    repouso_do_resultado = janela.resultado_da_correcao.resumo.caixa.toPlainText()
    _executa(janela, pedidos, tmp_path)
    da_execucao = janela.qualidade.resumo.caixa.toPlainText()
    assert da_execucao != SEM_RESUMO, "âncora: a execução de A não escreveu o resumo"
    dos_levados = repouso_do_resultado
    if levou:
        dos_levados = _leva_um_arquivo(janela, pedidos, tmp_path)
    assert (dos_levados != repouso_do_resultado) is levou, (
        "âncora: o arquivo levado não escreveu o resumo"
    )
    assert janela.qualidade.resumo.caixa.toPlainText() == da_execucao, (
        "o arquivo levado tirou o resumo da execução da caixa dela"
    )

    _escolhe(janela, escolha, "b")
    assert janela.qualidade.resumo.caixa.toPlainText() == SEM_RESUMO, (
        "o resumo da execução de A ficou sob B"
    )
    sob_b = janela.resultado_da_correcao.resumo.caixa.toPlainText()
    assert sob_b == dos_levados, "a troca mexeu no resumo dos arquivos levados"


@pytest.mark.parametrize("pasta", ["b", "a"], ids=["outro-par", "o-mesmo-par"])
def test_o_erro_depois_do_sucesso_volta_com_o_resumo_do_sucesso(
    qtbot: QtBot, tmp_path: Path, pasta: str
) -> None:
    """Um erro não substitui o resumo do último arquivo que existe, e a volta também não.

    Com A na tela, a segunda execução de A falha: as figuras saem e a aba de Qualidade limpa, e o
    resumo continua o do `200` anterior, porque o arquivo dele existe. **A resposta guardada é o
    erro**, e o resumo guardado com ele é o do sucesso. **Par causal:** sem a ida a B, a tela é a
    mesma; com a ida e a volta, também — e a âncora é que o erro mudou a tela e não o resumo.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    sucesso = _executa(janela, pedidos, tmp_path)
    _dispara(janela, pedidos, tmp_path)
    _termina(janela, CHAVE_DA_EXECUCAO, {"detail": "X"}, codigo=500)
    erro = _tela_da_execucao(janela)
    assert {"qualidade", "figura"} <= set(_difere(erro, sucesso)), "âncora: o erro não mudou a tela"
    # O resumo é lido com o botão «Copiar» desde o merge de 09/11/2026: o texto é o primeiro item.
    assert erro["resumo"] == sucesso["resumo"], "âncora: o erro mudou o resumo"
    assert sucesso["resumo"][0] != SEM_RESUMO, "âncora: o sucesso não escreveu o resumo"

    _escolhe(janela, escolha, pasta)
    if pasta == "b":
        assert janela.qualidade.resumo.caixa.toPlainText() == SEM_RESUMO, (
            "o resumo de A ficou sob B"
        )
        _escolhe(janela, escolha, "a")
    de_volta = _tela_da_execucao(janela)
    assert de_volta == erro, f"a volta a A não repôs o erro com o resumo: {_difere(de_volta, erro)}"


# ---------------------------------------------------------------------------
# A prévia mais nova, as vias, o enriquecimento e duas transições
# ---------------------------------------------------------------------------
#
# Quatro achados de dois revisores, cada um com o seu teste. As prévias das etapas escrevem nos
# campos das figuras que a execução escreve; a tabela das vias preditas e o resultado do
# enriquecimento são o que a execução mostra na aba das vias; e duas transições de estado não
# tinham teste em direção nenhuma — as plantas delas passavam verdes.

_CORPOS_DAS_PREVIAS: dict[str, dict[str, Any]] = {}
_GRUPOS = ("NoHTLV", "ATLL")


def _corpo_da_previa(kw: dict[str, Any], etapa: int) -> dict[str, Any]:
    """O `200` da prévia da etapa pela rota, com os arquivos e os campos do clique.

    As prévias não gravam nada, e por isso não pedem um SQLite. Guardado por conteúdo, como o da
    execução.
    """
    arquivos = kw["arquivos"]
    campos = kw["campos"]
    conteudo = repr((etapa, sorted((k, v[1]) for k, v in arquivos.items()), sorted(campos.items())))
    chave = hashlib.sha256(conteudo.encode("utf-8")).hexdigest()
    if chave not in _CORPOS_DAS_PREVIAS:
        servidor = TestClient(build_app(), raise_server_exceptions=False)
        lida = servidor.post(ROTA_DA_ETAPA[etapa], files=arquivos, data=campos)
        if lida.status_code == 202:
            identificador = lida.json()["identificador"]
            for _ in range(2400):
                lida = servidor.get(rota_do_progresso(identificador))
                if lida.status_code != 202:
                    break
                time.sleep(0.05)
        assert lida.status_code == 200, lida.text
        _CORPOS_DAS_PREVIAS[chave] = lida.json()
    return copy.deepcopy(_CORPOS_DAS_PREVIAS[chave])


def _previa_com_none(
    janela: JanelaPrincipal, pedidos: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    """A prévia da etapa 1 com outro corretor — `none` —, pelo botão. Devolve o que a tela mostra.

    O corretor da tela é sempre o que vale. A prévia chega pela chave da etapa, com o
    `202` do aceite e o `200` pela releitura encadeada, como a execução.
    """
    _entrega_a_previa(janela, _pede_a_previa_com_none(janela, pedidos))
    return _tela_da_execucao(janela)


def _pede_a_previa_com_none(
    janela: JanelaPrincipal, pedidos: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    """O clique da etapa 1 com o corretor `none`, e o `202` do aceite. Devolve o corpo da rota."""
    return _pede_a_previa(janela, pedidos, "none")


def _pede_a_previa(
    janela: JanelaPrincipal, pedidos: list[tuple[str, dict[str, Any]]], corretor: str
) -> dict[str, Any]:
    """O clique da etapa 1 com `corretor`, e o `202` do aceite. Devolve o corpo da rota.

    *(Merge de 09/11/2026: era o corpo de `_pede_a_previa_com_none`, com `none` fixo; o corretor
    passou a parâmetro para a Correção com o LOESS de `_executa_sobre_a_correcao`.)*
    """
    janela.pipeline.corretor.setCurrentText(corretor)
    assert janela.pipeline.corretor.currentText() == corretor, "âncora: o corretor não mudou"
    pedidos.clear()
    janela.pipeline.botoes_de_etapa[1].click()
    da_etapa = [kw for rota, kw in pedidos if rota == ROTA_DA_ETAPA[1]]
    assert len(da_etapa) == 1, f"âncora: o clique não chegou à etapa 1: {pedidos}"
    janela._recebe(CHAVE_DA_ETAPA[1], _resposta(202, {"identificador": IDENTIFICADOR, "total": 3}))
    return _corpo_da_previa(da_etapa[0], 1)


def _entrega_a_previa(janela: JanelaPrincipal, corpo: dict[str, Any]) -> None:
    """O `200` da prévia pela releitura encadeada, e o relógio das etapas uma vez."""
    janela._recebe(chave_do_pedido(CHAVE_DA_ETAPA[1], IDENTIFICADOR), _resposta(200, corpo))
    janela._atualiza_as_etapas()


def _executa_sobre_a_correcao(
    janela: JanelaPrincipal, pedidos: list[tuple[str, dict[str, Any]]], tmp_path: Path
) -> dict[str, Any]:
    """A Correção com o LOESS pela prévia, e a Análise sobre a corrigida até o `200` com a matriz
    na tela. Devolve o que a tela mostra.

    > Os testes das prévias abaixo
    > nasceram com o «Executar pipeline», que escrevia SEMPRE os dados das figuras da Correção, com
    > o corretor da tela. Sob as etapas independentes a Análise só os escreve quando reproduz a
    > Correção guardada: sobre a original ela não os toca, e «a execução mais velha cobre a
    > prévia mais nova» não teria o que cobrir — a planta cairia sobre dado em que as duas coisas
    > não diferem. **Daí a Correção antes**: a Análise parte da corrigida, que é a entrada
    > padrão com a Correção feita, e escreve no painel da correção.
    """
    _entrega_a_previa(janela, _pede_a_previa(janela, pedidos, "loess"))
    corpo = _dispara(janela, pedidos, tmp_path)
    assert ETAPAS[0] in janela._etapas_que_a_execucao_escreve(), (
        "âncora: a Análise não reproduz a Correção, e não escreve no painel da correção"
    )
    _termina(janela, CHAVE_DA_EXECUCAO, corpo)
    assert CHAVE_DA_EXECUCAO not in janela._esperas, "âncora: o `200` não fechou a espera"
    return _tela_da_execucao(janela)


def _apaga_a_ultima_letra_do_modo(janela: JanelaPrincipal) -> None:
    """Uma tecla no editor do combo do modo, como a pessoa a escreve — uma troca de matriz."""
    editor = janela.arquivos.modo.lineEdit()
    assert editor is not None, "âncora: o combo do modo não é editável"
    QTest.keyClick(editor, Qt.Key.Key_End)
    QTest.keyClick(editor, Qt.Key.Key_Backspace)
    assert janela.arquivos.modo.currentText() == MODO[:-1], "âncora: a tecla não chegou ao combo"


def _reescreve_a_ultima_letra_do_modo(janela: JanelaPrincipal) -> None:
    editor = janela.arquivos.modo.lineEdit()
    assert editor is not None, "âncora: o combo do modo não é editável"
    QTest.keyClicks(editor, MODO[-1])
    assert janela.arquivos.modo.currentText() == MODO, "âncora: o modo não voltou"


@pytest.mark.parametrize("volta", ["modo", "B-com-a-sua"])
@pytest.mark.parametrize("previa", [True, False], ids=["com-previa", "sem-previa"])
def test_a_previa_mais_nova_de_A_volta_com_A_e_a_execucao_mais_velha_nao_a_cobre(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, volta: str, previa: bool
) -> None:
    """A reposição da execução guardada de A não escreve por cima de uma prévia de A mais nova.

    Os dados das figuras das etapas 1 e 2 vivem nos campos que a execução escreve, e quem escreveu
    por último decide. **Medido:** depois de executar A e de pedir a
    prévia da etapa 1 com outro corretor, uma tecla no modo e a volta — ou B e A — repunham a
    execução por cima da prévia, e a figura do painel da correção ficava desenhada com os dados do
    outro corretor enquanto a etapa 1 dizia «em dia».

    **Par causal:** sem a prévia, a volta mostra a execução; com ela, a prévia — o que a tela de A
    mostrava antes da troca. No caminho «B-com-a-sua», B tem execução guardada e a ida a B escreve
    nos mesmos campos: a prévia de A já não está neles quando se volta.

    > A execução é a Análise sobre a corrigida, depois de uma Correção
    > com o LOESS (`_executa_sobre_a_correcao`): só assim ela escreve no painel da correção, e a
    > figura que este teste lê é a dele (`pontos_da_correcao`). O resto do que ele afirma não
    > mudou.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    if volta == "B-com-a-sua":
        _escolhe(janela, escolha, "b")
        _executa_sobre_a_correcao(janela, pedidos, tmp_path)
        _escolhe(janela, escolha, "a")
    esperado = _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    if previa:
        de_a = esperado
        esperado = _previa_com_none(janela, pedidos)
        assert "pontos_da_correcao" in _difere(esperado, de_a), (
            "âncora: a prévia não mudou a figura"
        )
    etapas = janela._atualiza_as_etapas()

    if volta == "modo":
        _apaga_a_ultima_letra_do_modo(janela)
        _reescreve_a_ultima_letra_do_modo(janela)
    else:
        _escolhe(janela, escolha, "b")
        _escolhe(janela, escolha, "a")
    assert janela._atualiza_as_etapas() == etapas, "âncora: a volta mudou as etapas"
    de_volta = _tela_da_execucao(janela)
    assert de_volta == esperado, (
        f"de volta a A, a tela não é a de antes da troca: difere em {_difere(de_volta, esperado)}"
    )


def _executa_e_falha(janela: JanelaPrincipal, pedidos: list[tuple[str, dict[str, Any]]]) -> None:
    """A Análise pelo botão dela, sobre a entrada da tela, e um `500` como desfecho.

    *(Era «Executar pipeline» pelo botão, que saiu.)*
    """
    pedidos.clear()
    janela.pipeline.botoes_de_etapa[ETAPAS[-1]].click()
    assert [r for r, _kw in pedidos if r == ROTA_DA_EXECUCAO], f"âncora: sem disparo: {pedidos}"
    janela._recebe(
        CHAVE_DA_EXECUCAO,
        _resposta(202, {"identificador": IDENTIFICADOR, "total": 8, "reclamado": False}),
    )
    _termina(janela, CHAVE_DA_EXECUCAO, {"detail": "X"}, codigo=500)
    assert CHAVE_DA_EXECUCAO not in janela._esperas, "âncora: o `500` não fechou a espera"


def test_um_erro_depois_da_previa_nao_a_apaga_na_volta(qtbot: QtBot, tmp_path: Path) -> None:
    """Com A na tela, um erro de A depois da prévia deixa a prévia; a volta a A também a deixa.

    **A prévia vale contra o último SUCESSO de A, e não contra o último desfecho**: um erro não traz
    os dados das figuras, e com A na tela ele não apaga os de uma prévia (`graficos._da_execucao`).
    **Par causal com o outro caminho:** o que a tela mostra com o erro chegado com A na tela é o
    que ela tem de mostrar de volta a A — e B, com a sua execução, escreve nos mesmos campos entre
    as duas.

    > A execução é a Análise sobre a corrigida, depois de uma Correção
    > com o LOESS (`_executa_sobre_a_correcao`): só assim ela escreve no painel da correção, e a
    > figura que este teste lê é a dele (`pontos_da_correcao`). O resto do que ele afirma não
    > mudou.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    _escolhe(janela, escolha, "b")
    _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    _escolhe(janela, escolha, "a")
    de_a = _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    da_previa = _previa_com_none(janela, pedidos)
    _executa_e_falha(janela, pedidos)
    janela._atualiza_as_etapas()
    com_o_erro = _tela_da_execucao(janela)
    assert (
        com_o_erro["pontos_da_correcao"]
        == da_previa["pontos_da_correcao"]
        != de_a["pontos_da_correcao"]
    ), "âncora: com A na tela, o erro apagou a prévia, ou ela não diferia da execução"
    etapas = janela._atualiza_as_etapas()

    _escolhe(janela, escolha, "b")
    _escolhe(janela, escolha, "a")
    assert janela._atualiza_as_etapas() == etapas, "âncora: a volta mudou as etapas"
    de_volta = _tela_da_execucao(janela)
    assert de_volta == com_o_erro, (
        f"de volta a A, o erro apagou a prévia: difere em {_difere(de_volta, com_o_erro)}"
    )


def test_a_previa_de_B_nao_se_repoe_sobre_a_execucao_de_A(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A prévia que se repõe é a da matriz do disparo dela, e não a última que chegou.

    **Par causal com `test_a_previa_mais_nova_de_A_volta_com_A...`:** lá a prévia é de A e volta
    com A; aqui ela é de B — mais nova que a execução de A, e nos mesmos campos —, e a volta a A
    mostra a execução de A.

    > A execução é a Análise sobre a corrigida, depois de uma Correção
    > com o LOESS (`_executa_sobre_a_correcao`): só assim ela escreve no painel da correção, e a
    > figura que este teste lê é a dele (`pontos_da_correcao`). O resto do que ele afirma não
    > mudou.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    de_a = _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    _escolhe(janela, escolha, "b")
    da_previa_de_b = _previa_com_none(janela, pedidos)
    assert da_previa_de_b["pontos_da_correcao"] != de_a["pontos_da_correcao"], (
        "âncora: a prévia de B não mudou a figura"
    )
    _escolhe(janela, escolha, "a")
    de_volta = _tela_da_execucao(janela)
    assert de_volta["pontos_da_correcao"] == de_a["pontos_da_correcao"], (
        "a prévia de B foi reposta sobre a execução de A"
    )


def test_a_previa_de_A_chegada_sob_B_volta_com_A(qtbot: QtBot, tmp_path: Path) -> None:  # noqa: N802
    """A prévia é da matriz do disparo dela, e não da tela na chegada.

    **Par causal com o outro caminho:** uma janela de controle recebe a prévia com A na tela; a
    outra escolhe B antes de ela chegar, e volta a A. De volta a A, a tela é a do controle.

    > A execução é a Análise sobre a corrigida, depois de uma Correção
    > com o LOESS (`_executa_sobre_a_correcao`): só assim ela escreve no painel da correção, e a
    > figura que este teste lê é a dele (`pontos_da_correcao`). O resto do que ele afirma não
    > mudou.
    """
    controle, _escolha_c, pedidos_c = _janela_que_guarda_o_pedido(qtbot, tmp_path / "controle")
    _executa_sobre_a_correcao(controle, pedidos_c, tmp_path)
    esperado = _previa_com_none(controle, pedidos_c)

    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path / "troca")
    de_a = _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    assert esperado["pontos_da_correcao"] != de_a["pontos_da_correcao"], (
        "âncora: a prévia não muda a figura"
    )
    corpo = _pede_a_previa_com_none(janela, pedidos)
    _escolhe(janela, escolha, "b")
    _entrega_a_previa(janela, corpo)
    _escolhe(janela, escolha, "a")
    janela._atualiza_as_etapas()
    de_volta = _tela_da_execucao(janela)
    assert de_volta == esperado, (
        f"a prévia de A chegada sob B não voltou com A: difere em {_difere(de_volta, esperado)}"
    )


def test_a_execucao_de_B_reposta_nao_deixa_a_correcao_de_B_no_painel_de_A(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A prévia de A mais velha que a execução de A volta também, ANTES dela — merge de 09/11/2026.

    Sob as etapas independentes a Análise sobre a original não escreve no painel da correção, e a
    reposição da execução de B, que o escreve por reproduzir a Correção de B, escrevia nele os
    dados de B. A primeira versão repunha só as prévias de A mais NOVAS que a execução de A,
    porque a do «Executar pipeline» cobria sempre as mais velhas: de volta a A, o painel ficava com
    a correção de B, com a etapa 1 de A em dia.

    **Par causal com o outro caminho:** a tela de A antes da ida a B é a que tem de voltar. As
    âncoras são que a Análise de A não escreve no painel da correção, e que a tela de B difere da
    de A nele — sem isso a reposição da prévia não teria o que mostrar.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    _escolhe(janela, escolha, "b")
    de_b = _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    _escolhe(janela, escolha, "a")
    _previa_com_none(janela, pedidos)
    janela._escolhe_a_entrada(ETAPAS[-1], "original")
    corpo = _dispara(janela, pedidos, tmp_path)
    assert janela._etapas_que_a_execucao_escreve() == frozenset(), (
        "âncora: a Análise de A sobre a original escreve no painel da correção"
    )
    _termina(janela, CHAVE_DA_EXECUCAO, corpo)
    esperado = _tela_da_execucao(janela)
    assert "pontos_da_correcao" in _difere(esperado, de_b), (
        "âncora: o painel da correção de A não difere do de B"
    )
    etapas = janela._atualiza_as_etapas()

    _escolhe(janela, escolha, "b")
    _escolhe(janela, escolha, "a")
    assert janela._atualiza_as_etapas() == etapas, "âncora: a volta mudou as etapas"
    de_volta = _tela_da_execucao(janela)
    assert de_volta == esperado, (
        f"de volta a A, a tela não é a de antes da troca: difere em {_difere(de_volta, esperado)}"
    )


class _Soltura:
    """As travas e as piscinas de um teste, soltas e drenadas no fim dele, passe ou reprove.

    **Medido numa planta:** um teste que reprova a meio deixava um trabalhador preso numa trava até
    o prazo dela, e a exceção dele caía no laço de eventos do teste SEGUINTE, que reprovava por ela
    — a planta da predição velha acusou também o enriquecimento, sem nada a ver com ele. As travas
    abrem e as piscinas drenam com a janela ainda viva, antes de o `qtbot` a fechar.
    """

    def __init__(self) -> None:
        self.travas: list[threading.Event] = []
        self.janelas: list[JanelaPrincipal] = []

    def trava(self) -> threading.Event:
        trava = threading.Event()
        self.travas.append(trava)
        return trava

    def janela(self, janela: JanelaPrincipal) -> None:
        self.janelas.append(janela)


@pytest.fixture
def soltura(qtbot: QtBot) -> Iterator[_Soltura]:
    """Pede o `qtbot` para ser desmontada ANTES dele: a janela ainda existe quando as piscinas
    drenam, e o sinal do trabalhador não cai num objeto apagado."""
    _ = qtbot
    registro = _Soltura()
    yield registro
    for trava in registro.travas:
        trava.set()
    for janela in registro.janelas:
        janela.vias._piscina.waitForDone(20000)
        janela.vias.enriquecimento._piscina.waitForDone(20000)


def _com_par_de_grupos(janela: JanelaPrincipal) -> None:
    """O par de grupos do contraste, para a execução trazer o volcano que a aba das vias lê.

    > Com o par, os testes desta seção executam sobre a corrigida
    > (`_executa_sobre_a_correcao`), e não sobre a original. Medido no par da factory: a Análise
    > sobre a original — corretor `none` — com o par de grupos volta `500` da rota, *«o trabalho
    > falhou na etapa «relatorio de qualidade»: ValueError: slice 0 has a NaN entry»*; com a
    > Correção pelo LOESS antes, volta `200` com o volcano. O «Executar pipeline», com que estes
    > testes nasceram, corria o LOESS do Palpite.
    """
    janela.pipeline.grupo_a.setText(_GRUPOS[0])
    janela.pipeline.grupo_b.setText(_GRUPOS[1])


def _calculadora(nome: str, trava: threading.Event | None = None) -> Any:
    """A predição trocada por uma que devolve uma via com este nome; com `trava`, ela espera.

    O cálculo não é o que se testa aqui — é o que a tela faz com o resultado —, e o real recusa
    uma entrada de cinco features.
    """

    def calcula(*_a: object, **_k: object) -> Any:
        if trava is not None:
            trava.wait(15)
        return _relatorio_de_vias(vias=(_via(nome, 0.01),))

    return calcula


def _vias(janela: JanelaPrincipal) -> tuple[Any, ...]:
    """O que a aba das vias mostra da predição: as linhas da tabela, o recorte e o rodapé."""
    return (janela.vias.linhas(), janela.vias.recorte.text(), janela.vias.rodape.text())


def _prediz(qtbot: QtBot, janela: JanelaPrincipal, nome: str) -> tuple[Any, ...]:
    """«Predizer vias» pelo botão, e espera a tabela. Devolve o que a aba mostra."""
    janela.vias._calcula = _calculadora(nome)
    assert janela.vias.botao.isEnabled(), "âncora: a execução não trouxe o volcano"
    janela.vias.botao.click()
    qtbot.waitUntil(lambda: bool(janela.vias.linhas()), timeout=20000)
    return _vias(janela)


@pytest.mark.parametrize("volta", ["modo", "B-com-a-sua"])
def test_a_tabela_das_vias_preditas_de_A_sai_na_troca_e_volta_com_A(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, volta: str
) -> None:
    """A tabela das vias preditas é parte do que a execução de A mostra, e volta com A.

    **Medido:** uma tecla no modo apagava a tabela e a volta ao mesmo modo não
    a repunha, onde a árvore de partida a mantinha. Mantê-la sob outra matriz seria mostrar A
    sob B; a tabela é escondida e reposta.

    **Par causal:** sob a outra matriz, a tabela de A não está; de volta a A, ela volta inteira. No
    caminho «B-com-a-sua», B pediu a sua, e cada uma volta com a sua matriz.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    _com_par_de_grupos(janela)
    de_b: tuple[Any, ...] | None = None
    if volta == "B-com-a-sua":
        _escolhe(janela, escolha, "b")
        _executa_sobre_a_correcao(janela, pedidos, tmp_path)
        de_b = _prediz(qtbot, janela, "Via de B")
        _escolhe(janela, escolha, "a")
    _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    de_a = _prediz(qtbot, janela, "Via de A")
    assert de_a != de_b, "âncora: as tabelas de A e de B são iguais"

    if volta == "modo":
        _apaga_a_ultima_letra_do_modo(janela)
        assert janela.vias.linhas() == (), "a tabela das vias de A ficou sob outro modo"
        _reescreve_a_ultima_letra_do_modo(janela)
    else:
        _escolhe(janela, escolha, "b")
        assert _vias(janela) == de_b, f"sob B, a aba das vias não é a de B: {_vias(janela)}"
        _escolhe(janela, escolha, "a")
    assert _vias(janela) == de_a, f"de volta a A, a tabela das vias não voltou: {_vias(janela)}"


def test_uma_execucao_nova_de_A_tira_a_predicao_velha_tambem_na_volta(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A predição guardada é a da execução guardada: uma execução nova de A tira-a, na volta também.

    **Par causal com `test_a_tabela_das_vias_preditas_de_A...`:** lá a mesma execução volta com a
    sua tabela; aqui a execução nova de A, chegada com A na tela, já a tirou da aba, e a volta a A
    não a repõe por cima da execução nova.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    _com_par_de_grupos(janela)
    _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    _prediz(qtbot, janela, "Via de A")
    _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    assert janela.vias.linhas() == (), "âncora: a execução nova não tirou a tabela velha"
    _escolhe(janela, escolha, "b")
    _escolhe(janela, escolha, "a")
    assert janela.vias.linhas() == (), "a volta a A repôs a predição da execução anterior"


@pytest.mark.parametrize("volta_antes", [False, True], ids=["chega-sob-B", "chega-de-volta-a-A"])
def test_a_predicao_de_A_em_voo_na_troca_vai_para_A(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, volta_antes: bool, soltura: _Soltura
) -> None:
    """A predição de A que termina depois da troca não pinta sob B, e aparece com A.

    É o molde da medição em voo: o resultado fica com a matriz de que é, e não com a da tela
    na chegada. **Par causal:** a chegada sob B deixa B sem tabela, e a volta a A mostra a de A; a
    chegada já de volta a A pinta na hora.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    _com_par_de_grupos(janela)
    _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    soltura.janela(janela)
    trava = soltura.trava()
    janela.vias._calcula = _calculadora("Via de A", trava)
    chegou: list[int] = []
    janela.vias._sinais.concluido.connect(lambda *_a: chegou.append(1))
    janela.vias.botao.click()
    _escolhe(janela, escolha, "b")
    if volta_antes:
        _escolhe(janela, escolha, "a")
    trava.set()
    qtbot.waitUntil(lambda: bool(chegou), timeout=20000)
    if not volta_antes:
        assert janela.vias.linhas() == (), "a predição de A pintou sob B"
        _escolhe(janela, escolha, "a")
    linhas = janela.vias.linhas()
    assert [linha[0] for linha in linhas] == ["Via de A"], f"a predição de A não voltou: {linhas}"


def test_a_predicao_de_A_chegada_com_outra_a_correr_nao_pinta_por_cima(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, soltura: _Soltura
) -> None:
    """De volta a A com uma predição nova a correr, a velha que chega não pinta por cima dela.

    A velha fica guardada, e a nova chega a seguir: pintar a velha poria uma tabela sob o botão
    desligado e o «a predizer». **Par causal:** depois da velha, a tela continua a dizer que
    calcula; depois da nova, a tabela é a dela.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    _com_par_de_grupos(janela)
    _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    soltura.janela(janela)
    travas = [soltura.trava(), soltura.trava()]
    nomes = ["Via velha", "Via nova"]
    chamadas: list[int] = []

    def calcula(*_a: object, **_k: object) -> Any:
        indice = len(chamadas)
        chamadas.append(indice)
        travas[indice].wait(15)
        return _relatorio_de_vias(vias=(_via(nomes[indice], 0.01),))

    janela.vias._calcula = calcula
    chegou: list[int] = []
    janela.vias._sinais.concluido.connect(lambda *_a: chegou.append(1))
    janela.vias.botao.click()
    qtbot.waitUntil(lambda: chamadas == [0], timeout=20000)
    _escolhe(janela, escolha, "b")
    _escolhe(janela, escolha, "a")
    janela.vias.botao.click()
    a_calcular = janela.vias.recorte.text()
    travas[0].set()
    qtbot.waitUntil(lambda: len(chegou) == 1, timeout=20000)
    assert janela.vias.linhas() == (), "a predição velha pintou por cima da nova a correr"
    assert janela.vias.recorte.text() == a_calcular, "a tela deixou de dizer que calcula"
    travas[1].set()
    qtbot.waitUntil(lambda: len(chegou) == 2, timeout=20000)
    assert [linha[0] for linha in janela.vias.linhas()] == ["Via nova"]


class _ExecutorQueGuardaOCancelamento(Executor):
    """O duplo do R que guarda a pergunta de cancelamento que cada chamada recebeu.

    Lê-la depois da troca, e antes de a trava abrir, diz se a troca cancelou — sem a corrida entre
    a trava e a sondagem do cancelamento que o duplo faz a cada 50 ms.
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.cancelados: list[Any] = []

    def __call__(self, pedido: dict[str, Any], **kw: Any) -> Any:
        self.cancelados.append(kw["cancelado"])
        return super().__call__(pedido, **kw)


def _enriquecimento_pronto(
    janela: JanelaPrincipal, tmp_path: Path, executor: Executor
) -> ej.PainelDeEnriquecimento:
    """O segundo modo da aba das vias com o R e as bibliotecas trocados por duplos.

    O molde é o de `tests/test_desktop_enriquecimento.py`. A tabela de identificação nomeia as cinco
    features do par A da factory, e o corte é o dos três melhores: com os cinco, a lista seria o
    fundo, que é recusa antecipada, e com menos de três o duplo da ORA sai fora do contrato.
    """
    painel = janela.vias.enriquecimento
    motor = MotorDoR(
        tmp_path / "R",
        tmp_path / "R" / "vias_entrypoint.R",
        tmp_path / "vias-dados",
        tmp_path / "baixadas",
    )
    origem = ej.OrigemDasBibliotecas(
        tmp_path / "vias-dados",
        "da pasta do laboratório, conferidas",
        True,
        (
            ej.ArquivoDaBiblioteca(
                "syn_nms.qs", SHA_DE_TESTE, "fixado, conferido contra o manifesto"
            ),
        ),
    )
    nomes = ("Glycine", "L-Alanine", "Citric acid", "Pyruvic acid", "Urea")
    linhas = "".join(f"{i}.00_{i}00.0000m/z,{nome},1\n" for i, nome in enumerate(nomes, start=1))
    tabela = tmp_path / "identificacao.csv"
    tabela.write_text("Compound,Description,Score\n" + linhas, encoding="utf-8", newline="")
    painel._executor = executor
    painel._motor = lambda: motor
    painel._origem_das_bibliotecas = lambda _m: origem
    painel._abre_arquivo = lambda _pai: tabela
    painel._confirma = lambda *_a: True
    painel.tipo_do_corte.setCurrentIndex(painel.tipo_do_corte.findData("n_melhores"))
    painel.valor_do_corte.setValue(3)
    painel._carrega_tabela()
    painel._confirma_colunas()
    assert painel.botao_enriquece.isEnabled(), (
        f"âncora: o enriquecimento recusou: {painel.estado.text()}"
    )
    return painel


def _enriquecido(painel: ej.PainelDeEnriquecimento) -> tuple[Any, ...]:
    """O que o modo do enriquecimento mostra do resultado: a tabela, o estado e o rodapé."""
    linhas = tuple(
        tuple(
            "" if painel.tabela.item(i, j) is None else painel.tabela.item(i, j).text()
            for j in range(painel.tabela.columnCount())
        )
        for i in range(painel.tabela.rowCount())
    )
    return (linhas, painel.estado.text(), painel.rodape.text(), painel.botao_exporta.isEnabled())


@pytest.mark.parametrize("caso", ["terminado", "em-voo", "outro-corte"])
def test_o_enriquecimento_de_A_sai_na_troca_e_volta_com_A(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, caso: str, soltura: _Soltura
) -> None:
    """O resultado do enriquecimento de A sai na troca e volta com A; o R em voo não é cancelado.

    **A troca não cancela o R**: o enriquecimento leva minutos, e uma tecla no modo não pode
    custá-lo. O resultado que chega sob B fica com A, pelo molde da predição. **Par causal:**
    terminado antes da troca, ele volta com A; em voo, a troca não o cancela, ele não pinta sob B,
    e volta com A; e com o corte mudado entre a ida e a volta, ele não é o desta tela, e não volta.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    _com_par_de_grupos(janela)
    _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    soltura.janela(janela)
    trava = soltura.trava()
    executor = _ExecutorQueGuardaOCancelamento(trava=trava)
    painel = _enriquecimento_pronto(janela, tmp_path, executor)
    chegou: list[int] = []
    painel._sinais.concluido.connect(lambda *_a: chegou.append(1))
    if caso != "em-voo":
        trava.set()
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: bool(executor.cancelados), timeout=20000)
    if caso != "em-voo":
        qtbot.waitUntil(lambda: bool(chegou), timeout=20000)
        assert painel.tabela.rowCount() > 0, f"âncora: não pintou: {painel.estado.text()}"
    de_a = _enriquecido(painel)

    _escolhe(janela, escolha, "b")
    if caso == "em-voo":
        assert not executor.cancelados[0](), "a troca cancelou o R de A"
        trava.set()
        qtbot.waitUntil(lambda: bool(chegou), timeout=20000)
    assert painel.tabela.rowCount() == 0, "o enriquecimento de A ficou sob B"
    if caso == "outro-corte":
        painel.valor_do_corte.setValue(4)
    _escolhe(janela, escolha, "a")
    de_volta = _enriquecido(painel)
    if caso == "em-voo":
        assert [linha[0] for linha in de_volta[0]] == ["Urea Cycle"], (
            f"o enriquecimento de A não voltou com A: {de_volta}"
        )
    elif caso == "outro-corte":
        assert de_volta[0] == (), f"o resultado de outro corte voltou com A: {de_volta}"
        assert "**4 compostos na lista**" in painel.degraus.text(), "âncora: o corte não mudou"
    else:
        assert de_volta == de_a, f"o enriquecimento de A não voltou com A: {de_volta}"


class _ExecutorDeDuasTravas(Executor):
    """O duplo do R com uma trava por cálculo: as duas chamadas de cada um esperam a sua."""

    def __init__(self, travas: list[threading.Event]) -> None:
        super().__init__()
        self.travas = travas
        self.chamadas = 0

    def __call__(self, pedido: dict[str, Any], **kw: Any) -> Any:
        calculo = self.chamadas // 2
        self.chamadas += 1
        self.travas[calculo].wait(15)
        return super().__call__(pedido, **kw)


def test_o_enriquecimento_de_A_chegado_com_outro_a_correr_nao_pinta_por_cima(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, soltura: _Soltura
) -> None:
    """De volta a A com um enriquecimento novo a correr, o velho que chega não pinta por cima.

    O relógio da espera reescreveria o estado por cima dele a cada segundo, e o novo chega a
    seguir. **Par causal:** depois do velho, a tabela está vazia e a espera continua; depois do
    novo, a tabela é a dele.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    _com_par_de_grupos(janela)
    _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    soltura.janela(janela)
    executor = _ExecutorDeDuasTravas([soltura.trava(), soltura.trava()])
    painel = _enriquecimento_pronto(janela, tmp_path, executor)
    chegou: list[int] = []
    painel._sinais.concluido.connect(lambda *_a: chegou.append(1))
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: executor.chamadas == 1, timeout=20000)
    _escolhe(janela, escolha, "b")
    _escolhe(janela, escolha, "a")
    painel.botao_enriquece.click()
    assert painel.botao_cancela.isEnabled(), "âncora: o enriquecimento novo não está à espera"
    executor.travas[0].set()
    qtbot.waitUntil(lambda: len(chegou) == 1, timeout=20000)
    assert painel.tabela.rowCount() == 0, "o enriquecimento velho pintou por cima do novo a correr"
    assert painel.botao_cancela.isEnabled(), "a espera do enriquecimento novo parou"
    executor.travas[1].set()
    qtbot.waitUntil(lambda: len(chegou) == 2, timeout=20000)
    assert painel.tabela.rowCount() == 1, f"o novo não pintou: {painel.estado.text()}"


def test_o_fechamento_cancela_o_R_da_matriz_que_saiu(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, soltura: _Soltura
) -> None:
    """A troca não cancela o R de A; o fechamento da janela cancela-o, mesmo com outro depois.

    **Par causal** com o caso em voo do teste acima: lá a troca deixa o R de A correr; aqui o
    fechamento o para, e a piscina drena dentro do teto — com B a ter posto outro cancelamento no
    lugar do de A.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    _com_par_de_grupos(janela)
    _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    soltura.janela(janela)
    executor = _ExecutorQueGuardaOCancelamento(trava=soltura.trava())
    painel = _enriquecimento_pronto(janela, tmp_path, executor)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: bool(executor.cancelados), timeout=20000)
    _escolhe(janela, escolha, "b")
    _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    painel._carrega_tabela()
    painel._confirma_colunas()
    assert painel.botao_enriquece.isEnabled(), f"âncora: B recusou: {painel.estado.text()}"
    painel.botao_enriquece.click()
    assert not executor.cancelados[0](), "a troca ou a execução de B cancelou o R de A"
    assert painel.encerra(teto_ms=5000), "o fechamento não drenou a piscina dentro do teto"
    assert executor.cancelados[0](), "o fechamento não cancelou o R de A"


def test_uma_execucao_nova_de_A_depois_da_volta_cancela_o_R_de_A(  # noqa: N802
    qtbot: QtBot, tmp_path: Path, soltura: _Soltura
) -> None:
    """De volta a A com o R de A a correr, uma execução nova de A cancela-o, como sem a troca.

    **Par causal com o caso em voo de `test_o_enriquecimento_de_A_sai_na_troca...`:** lá a troca não
    cancela o R; aqui a execução nova da matriz dele, chegada com ela na tela, cancela — o cálculo
    da execução anterior deixou de ter uso.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    _com_par_de_grupos(janela)
    _executa_sobre_a_correcao(janela, pedidos, tmp_path)
    soltura.janela(janela)
    executor = _ExecutorQueGuardaOCancelamento(trava=soltura.trava())
    painel = _enriquecimento_pronto(janela, tmp_path, executor)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: bool(executor.cancelados), timeout=20000)
    _escolhe(janela, escolha, "b")
    _escolhe(janela, escolha, "a")
    assert not executor.cancelados[0](), "âncora: a troca cancelou o R de A"
    _dispara(janela, pedidos, tmp_path)
    assert executor.cancelados[0](), "a execução nova de A não cancelou o R da anterior"
    assert painel.encerra(teto_ms=5000), "a piscina não drenou"


def test_levar_o_resultado_e_executar_depois_devolve_o_resumo_a_matriz(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Depois de «Levar o resultado», a execução seguinte tem o seu resumo, e ele segue a matriz.

    **Por planta:** sem a linha que devolve a caixa à execução, o resumo da
    execução seguinte ficava sob as outras matrizes, e nenhum teste reprovava. **Par causal** com
    `test_o_resumo_dos_arquivos_levados_nao_segue_a_matriz[com-arquivo-levado]`: lá, sem execução
    depois de levar, o resumo dos levados fica sob B; aqui, com ela, sai.

    > A linha que esta planta
    > tirava — a que devolvia a caixa única à execução depois de um arquivo levado
    > (`_resumo_da_exportacao`) — saiu com a caixa: o resumo da Análise e o dos
    > arquivos levados moram em caixas diferentes. **O que fica é a mesma sequência e a mesma
    > asserção sobre a caixa da execução**: levar um arquivo e executar depois não pode deixar o
    > resumo da execução sob B — o que reprovaria se o download voltasse a escrever nela.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    repouso_do_resultado = janela.resultado_da_correcao.resumo.caixa.toPlainText()
    _executa(janela, pedidos, tmp_path)
    dos_levados = _leva_um_arquivo(janela, pedidos, tmp_path)
    assert dos_levados != repouso_do_resultado, "âncora: o arquivo levado não escreveu o resumo"

    _executa(janela, pedidos, tmp_path)
    da_execucao = janela.qualidade.resumo.caixa.toPlainText()
    assert da_execucao not in {dos_levados, SEM_RESUMO}, "âncora: a execução não escreveu o resumo"
    _escolhe(janela, escolha, "b")
    assert janela.qualidade.resumo.caixa.toPlainText() == SEM_RESUMO, (
        "o resumo da execução de A ficou sob B depois de um arquivo levado"
    )


def test_a_volta_a_A_em_voo_mostra_que_corre_e_a_ida_seguinte_a_B_o_repouso(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Com A em voo, a volta a A mostra «ainda corre», e a ida seguinte a B, o repouso.

    **Por planta:** sem marcar que as abas mostram a execução em voo de A, a
    sequência B → A → B deixava sob B a figura em «a execução ainda corre», e nenhum teste
    reprovava.
    **Par causal:** parar em A mostra «ainda corre»; seguir para B, o repouso.
    """
    janela, escolha, pedidos = _janela_que_guarda_o_pedido(qtbot, tmp_path)
    repouso = _tela_da_execucao(janela)
    _dispara(janela, pedidos, tmp_path)
    _escolhe(janela, escolha, "b")
    _escolhe(janela, escolha, "a")
    figura = janela.graficos.figuras.repouso.text()
    assert figura == texto_do_estado(EstadoDaFigura.EM_CURSO), (
        f"de volta a A em voo, a figura não diz que a execução ainda corre: {figura!r}"
    )
    _escolhe(janela, escolha, "b")
    sob_b = _abas(_tela_da_execucao(janela))
    assert sob_b == _abas(repouso), (
        f"sob B, a execução em voo de A ficou nas abas: {_difere(sob_b, _abas(repouso))}"
    )


# ----------------------------------------------------------------------------------------------
# «Seguir sugestões do app»
# ----------------------------------------------------------------------------------------------


def _bloco_da_correcao(janela: JanelaPrincipal) -> Any:
    bloco = janela.pagina_da_correcao.sugestoes
    assert bloco is not None, "âncora: a página da Correção não tem o botão das sugestões"
    return bloco


@pytest.mark.parametrize("troca_o_par", [False, True], ids=["mesmo-par", "par-novo"])
def test_os_lotes_propostos_para_um_par_nao_entram_no_seguinte(
    qtbot: QtBot, tmp_path: Path, troca_o_par: bool
) -> None:
    """O número de lotes do «Diagnóstico» de A não é a sugestão para B.

    **Par causal:** com o mesmo par, o clique põe os 2 lotes propostos; com o par B, o número
    escolhido à mão fica, e a frase diz que o «Diagnóstico» não propôs lotes para esta matriz.
    Antes, `_proposta` guardava os lotes de A, e o clique sobre B os escrevia.

    A Validação e o Diagnóstico do par da tela entram em dia antes do
    clique (`_leituras_em_dia`, sem sugestão de lotes): o clique mede o Palpite na hora, e a única
    sugestão de lotes é a que o teste põe.
    """
    janela, escolha, _rotas = _janela(qtbot, tmp_path)
    # A proposta pelo caminho de produção — a leitura do «Diagnóstico» —, que a guarda sem a
    # escrever no campo.
    janela._pre_preenche_com_a_proposta("diagnostico", {"lotes": {"n_lotes": 2}})
    minimo = janela.pipeline.n_lotes.value()
    assert minimo != 2, "âncora: o campo já está nos lotes propostos, e o par não separa nada"
    if troca_o_par:
        _escolhe(janela, escolha, "b")
    _leituras_em_dia(janela)
    bloco = _bloco_da_correcao(janela)
    bloco.botao.click()
    _aceita(janela, CHAVE_DO_PALPITE)
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))

    if not troca_o_par:
        assert janela.pipeline.n_lotes.value() == 2, "o clique não pôs os lotes propostos"
        assert bloco.desfecho.text() == "Sugestões aplicadas: corretor «LOESS», 2 lotes."
        return
    assert janela.pipeline.n_lotes.value() == minimo, "os lotes propostos para A entraram em B"
    assert "O número de lotes ficou como estava" in bloco.desfecho.text(), bloco.desfecho.text()


@pytest.mark.parametrize("depois", ["nada", "outro-par", "palpite-do-cartao"])
def test_a_frase_das_sugestoes_sai_quando_deixa_de_descrever_a_tela(
    qtbot: QtBot, tmp_path: Path, depois: str
) -> None:
    """«Sugestões aplicadas: corretor «LOESS»» não fica ao lado de um seletor em SERRF.

    **Par causal em três braços:** sem nada depois, a frase fica; trocar o par a tira; e o Palpite
    pedido pelo cartão, que escreve outro corretor no seletor como escrita da MÁQUINA — e por isso
    sem passar pelo sinal do toque —, também a tira. Antes só o toque humano a tirava.

    A Validação e o Diagnóstico entram em dia antes do clique
    (`_leituras_em_dia`), e o clique mede o Palpite na hora, como media.
    """
    janela, escolha, _rotas = _janela(qtbot, tmp_path)
    bloco = _bloco_da_correcao(janela)
    janela.pipeline.corretor.setCurrentText("serrf")
    _leituras_em_dia(janela)
    bloco.botao.click()
    _aceita(janela, CHAVE_DO_PALPITE)
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))
    assert bloco.desfecho.text().startswith("Sugestões aplicadas: corretor «LOESS»"), (
        f"âncora: as sugestões não entraram: {bloco.desfecho.text()!r}"
    )

    if depois == "outro-par":
        _escolhe(janela, escolha, "b")
    elif depois == "palpite-do-cartao":
        # **O Palpite do cartão já não escreve no seletor**: ele guarda a
        # sugestão nova e desfaz a adoção da anterior, e os campos — e a frase que os descreve —
        # ficam como estavam. Antes ele escrevia SERRF, e a frase saía.
        assert janela._dispara_o_palpite(), "âncora: o Palpite do cartão não saiu"
        _aceita(janela, CHAVE_DO_PALPITE)
        _termina(janela, CHAVE_DO_PALPITE, {**PALPITE_DE_A, "correction_proposta": "serrf"})
        assert janela.pipeline.corretor_sugerido == "serrf", "âncora: a sugestão nova não chegou"
        assert janela.pipeline.corretor.currentText() == "loess", "o Palpite escreveu no seletor"
        assert "correction" not in janela.pipeline.proposta_da_maquina(), (
            "a adoção da sugestão anterior continuou como o lado da máquina do par"
        )

    if depois in ("nada", "palpite-do-cartao"):
        assert bloco.desfecho.text().startswith("Sugestões aplicadas: corretor «LOESS»")
        return
    assert bloco.desfecho.text() == "", (
        f"a frase ficou a descrever outra tela: {bloco.desfecho.text()!r}"
    )


@pytest.mark.parametrize("toca", [False, True], ids=["sem-toque", "toque-no-corretor"])
def test_o_corretor_escolhido_durante_a_espera_vence_o_palpite(
    qtbot: QtBot, tmp_path: Path, toca: bool
) -> None:
    """O seletor fica editável durante a espera, e a escolha feita ali não é sobrescrita.

    **Par causal:** sem toque, o Palpite que chega põe o LOESS; com a pessoa a escolher SERRF no
    intervalo, o SERRF fica, e a frase diz o que o Palpite sugeriu. Antes,
    `segue_a_proposta` esquecia o toque e o LOESS escrevia por cima.

    **A espera continua até o Palpite chegar.** Uma versão anterior fazia o toque
    largá-la na hora; o Palpite é independente da aba: alterar algo na tela de correção enquanto
    ele roda não o para nem afeta o resultado.

    A Validação e o Diagnóstico entram em dia antes do clique
    (`_leituras_em_dia`), e o toque cai durante o Palpite, que é a espera de que o teste trata.
    """
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    bloco = _bloco_da_correcao(janela)
    _leituras_em_dia(janela)
    bloco.botao.click()
    _aceita(janela, CHAVE_DO_PALPITE)
    assert janela._depois_do_palpite is not None, "âncora: o clique não ficou à espera do Palpite"
    if toca:
        janela.pipeline.corretor.setCurrentText("serrf")
        assert janela._depois_do_palpite is not None, "o toque largou a espera do Palpite"
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))

    assert janela.pipeline.corretor.currentText() == ("serrf" if toca else "loess")
    assert bloco.botao.isEnabled(), "o botão das sugestões ficou desligado"
    if toca:
        texto = bloco.desfecho.text()
        assert "O corretor que você escolheu enquanto o Palpite media ficou" in texto, texto
        assert "O Palpite sugeriu «LOESS»" in texto, texto


@pytest.mark.parametrize("com_proposta", [False, True], ids=["sem-lotes", "com-lotes"])
def test_a_sugestao_recusada_pela_trava_diz_os_lotes_que_ja_pos(
    qtbot: QtBot, tmp_path: Path, com_proposta: bool
) -> None:
    """A trava recusa o Palpite DEPOIS de os lotes entrarem, e a frase diz que eles entraram.

    **Par causal:** com proposta de lotes, o campo muda e a frase termina dizendo o número; sem
    proposta, nada muda e a frase não fala de lotes. Antes a frase era *«A sugestão não
    pôde ser medida agora»* nos dois casos — e no primeiro a pessoa lia que nada mudara.
    """
    janela, _escolha, rotas = _janela(qtbot, tmp_path)
    if com_proposta:
        janela._pre_preenche_com_a_proposta("diagnostico", {"lotes": {"n_lotes": 2}})
        assert janela.pipeline.n_lotes.value() != 2, "âncora: o campo já estava no proposto"
    janela._esperas[CHAVE_DA_EXECUCAO] = object()  # type: ignore[assignment]
    try:
        _bloco_da_correcao(janela).botao.click()
    finally:
        janela._esperas.pop(CHAVE_DA_EXECUCAO)
    texto = _bloco_da_correcao(janela).desfecho.text()

    assert texto.startswith("O corretor não pôde ser sugerido agora:"), texto
    assert ROTA_DO_PALPITE not in rotas, "a trava recusou e o Palpite saiu mesmo assim"
    if com_proposta:
        assert janela.pipeline.n_lotes.value() == 2
        assert texto.endswith("O número de lotes já está em 2, como o «Diagnóstico» propôs."), texto
        return
    assert "lotes" not in texto, texto


@pytest.mark.parametrize(
    "a_sugestao_muda_em_voo", [False, True], ids=["sugestao-antes", "sugestao-em-voo"]
)
def test_o_palpite_em_voo_e_pedido_outra_vez_quando_a_sugestao_de_lotes_muda(
    qtbot: QtBot, tmp_path: Path, a_sugestao_muda_em_voo: bool
) -> None:
    """O Palpite do cartão já mede, e a sugestão de lotes mudou depois: ele é de outra partição.

    **Par causal:** com a sugestão de antes do Palpite, ele serve e as sugestões entram com um
    Palpite só; com o «Diagnóstico» a chegar durante a medição, a chegada pede o Palpite da partição
    sugerida agora, e as sugestões entram quando o segundo chega.

    Era `..._quando_o_clique_muda_os_lotes`: antes o Palpite era medido nos lotes do campo, e
    era o clique que, ao pô-los, o tornava de outra partição. Hoje ele mede na partição sugerida, e
    só a sugestão nova o torna velho.

    A Validação e o Diagnóstico entram em dia no começo
    (`_leituras_em_dia`, sem sugestão de lotes): o Palpite pedido outra vez sai pela cadeia, e com
    as duas em dia ele sai direto, como saía — a contagem de `ROTA_DO_PALPITE` é a mesma.
    """
    janela, _escolha, rotas = _janela(qtbot, tmp_path)
    _leituras_em_dia(janela)
    corpo = {"lotes": {"n_lotes": 2}}
    if not a_sugestao_muda_em_voo:
        janela._pre_preenche_com_a_proposta("diagnostico", corpo)
    assert janela._dispara_o_palpite(), "âncora: o Palpite do cartão não saiu"
    _aceita(janela, CHAVE_DO_PALPITE)
    if a_sugestao_muda_em_voo:
        janela._pre_preenche_com_a_proposta("diagnostico", corpo)
    bloco = _bloco_da_correcao(janela)
    bloco.botao.click()
    assert janela._depois_do_palpite is not None, "âncora: o clique não esperou o Palpite em voo"
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))

    if not a_sugestao_muda_em_voo:
        assert rotas.count(ROTA_DO_PALPITE) == 1, rotas
        assert bloco.desfecho.text() == "Sugestões aplicadas: corretor «LOESS», 2 lotes."
        return
    assert bloco.desfecho.text() != ENTRADA_MUDOU_DURANTE_O_PALPITE, (
        "o clique foi culpado ao chegar"
    )
    assert rotas.count(ROTA_DO_PALPITE) == 2, f"o Palpite da partição do clique não saiu: {rotas}"
    assert janela._depois_do_palpite is not None, "as sugestões não esperam o Palpite novo"
    _aceita(janela, CHAVE_DO_PALPITE)
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))
    assert bloco.desfecho.text() == "Sugestões aplicadas: corretor «LOESS», 2 lotes."


def test_seguir_a_proposta_sem_proposta_guarda_o_toque_da_pessoa(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Um clique sem proposta de lotes não apaga a marca de que a pessoa escolheu o número.

    **Par causal:** sem proposta, o «Diagnóstico» que chega depois não escreve por cima do número
    escolhido à mão; com o toque esquecido — o desenho anterior —, escreveria. A âncora é o mesmo
    `pre_preenche` sobre um campo NÃO tocado, que escreve.
    """
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    painel = janela.pipeline
    minimo = painel.n_lotes.minimum()
    painel.n_lotes.setValue(minimo + 1)
    escolhido = painel.n_lotes.value()
    assert escolhido != minimo, "âncora: o campo não aceitou o número escolhido à mão"
    assert painel.segue_a_proposta("n_batches") is None, "âncora: havia proposta de lotes"
    assert not painel.pre_preenche("n_batches", minimo), "o toque da pessoa foi esquecido"
    assert painel.n_lotes.value() == escolhido

    sem_toque, _e, _r = _janela(qtbot, tmp_path / "ancora")
    assert sem_toque.pipeline.n_lotes.value() != escolhido, "âncora: o campo novo já está lá"
    assert sem_toque.pipeline.pre_preenche("n_batches", escolhido), (
        "âncora: num campo não tocado, a proposta não escreveu — o par não separa nada"
    )


# ----------------------------------------------------------------------------------------------
# A aba «Correção» abre com «Com todos os QCs» e um lote
# ----------------------------------------------------------------------------------------------


def test_d702_a_correcao_abre_com_todos_os_qcs(qtbot: QtBot, tmp_path: Path) -> None:
    """A variante RSD dos QCs padrão é sempre «com todos os QCs».

    A âncora é o seletor oferecer as duas variantes: com uma só, o repouso não separaria nada.
    """
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    variante = janela.pipeline.variante_do_rsd
    opcoes = {variante.itemText(i) for i in range(variante.count())}
    assert opcoes == {"plain", "outlier_removed"}, f"âncora: o seletor oferece {opcoes}"
    assert variante.currentText() == VARIANTE_DO_RSD_EM_REPOUSO == "plain"
    campos = janela._campos_atuais()
    assert campos is not None, "âncora: a janela não tem o par para montar o pedido"
    assert campos[0]["rsd_variant"] == "plain", "o pedido não leva a variante da tela"


@pytest.mark.parametrize("adota", [False, True], ids=["digitado-a-mao", "pelo-botao"])
def test_d702_os_lotes_do_diagnostico_so_entram_pelo_botao(
    qtbot: QtBot, tmp_path: Path, adota: bool
) -> None:
    """O padrão de «Quantos lotes declarar» é sempre 1.

    O «Diagnóstico» chega e o campo fica em 1, sem contar como proposta trocada. **Par causal:**
    pelo botão, o número entra e leva os cortes dele; digitado à mão, o mesmo número corta
    em blocos iguais, porque só o botão adota a sugestão.
    """
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    painel = janela.pipeline
    # O repouso do campo é o mínimo com o texto especial «1 — um lote só», e o pedido manda 1.
    repouso = painel.n_lotes.value()
    assert repouso == painel.n_lotes.minimum(), f"âncora: o campo não nasce vazio: {repouso}"
    assert painel.n_lotes.specialValueText() == SEM_LOTES and SEM_LOTES.startswith("1 "), (
        f"âncora: o vazio não se lê como um lote: {SEM_LOTES!r}"
    )
    corpo = {"lotes": {"n_lotes": 2, "changepoints": [20]}}
    assert janela._pre_preenche_com_a_proposta("diagnostico", corpo) == ()
    assert painel.n_lotes.value() == repouso, "o «Diagnóstico» escreveu no campo"
    assert "n_batches" not in painel.eixos_divergentes(), (
        "o campo em 1 ficou gravado como proposta da máquina trocada, sem ninguém mexer"
    )
    assert painel.cortes_em_uso(2) is None, "os cortes valeram sem a sugestão adotada"

    if adota:
        _bloco_da_correcao(janela).botao.click()
        assert painel.n_lotes.value() == 2, "o botão não pôs os lotes do «Diagnóstico»"
        assert painel.cortes_em_uso(2) == (20,), "o botão adotou o número sem os cortes"
        assert "n_batches" not in painel.eixos_divergentes()
        return
    painel.n_lotes.setValue(2)
    assert painel.cortes_em_uso(2) is None, "o número digitado à mão adotou os cortes"


@pytest.mark.parametrize(("nova", "desfaz"), [(2, False), (3, True)], ids=["mesma", "outra"])
def test_d702_um_diagnostico_novo_com_outro_numero_desfaz_a_adocao(
    qtbot: QtBot, tmp_path: Path, nova: int, desfaz: bool
) -> None:
    """A adoção era da sugestão anterior; uma sugestão DIFERENTE a desfaz, a mesma não.

    **Par causal:** depois de adotar 2, um «Diagnóstico» que propõe 2 de novo deixa os cortes em
    uso; um que propõe 3 tira o número adotado do lado da máquina, e os cortes caem.
    """
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    painel = janela.pipeline
    janela._pre_preenche_com_a_proposta(
        "diagnostico", {"lotes": {"n_lotes": 2, "changepoints": [20]}}
    )
    _bloco_da_correcao(janela).botao.click()
    assert painel.cortes_em_uso(2) == (20,), "âncora: a adoção não aconteceu"

    cortes = [20] if nova == 2 else [15, 30]
    janela._pre_preenche_com_a_proposta(
        "diagnostico", {"lotes": {"n_lotes": nova, "changepoints": cortes}}
    )
    assert painel.n_lotes.value() == 2, "o «Diagnóstico» novo escreveu no campo"
    # **O que a adoção muda é o lado da máquina do par**, e é ele que se afirma: os cortes caem nos
    # dois braços quando o número muda, só pela contagem — três lotes trazem dois cortes —, e não
    # separariam nada (a planta que desliga o «desfaz» passava com eles).
    if desfaz:
        assert "n_batches" not in painel.proposta_da_maquina(), (
            "a adoção da sugestão anterior continuou como o lado da máquina do par"
        )
        return
    assert painel.proposta_da_maquina().get("n_batches") == 2, "a mesma sugestão desfez a adoção"
    assert painel.cortes_em_uso(2) == (20,), "a mesma sugestão de novo tirou os cortes"


# ----------------------------------------------------------------------------------------------
# «1 — Não dividir em lotes», e o Palpite independente da aba «Correção»
# ----------------------------------------------------------------------------------------------


def test_d703_o_campo_vazio_de_lotes_diz_nao_dividir_em_lotes(qtbot: QtBot, tmp_path: Path) -> None:
    """O padrão diz «1 — Não dividir em lotes». Antes, «1 — um lote só»."""
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    lotes = janela.pipeline.n_lotes
    assert lotes.value() == lotes.minimum(), "âncora: o campo não está no repouso"
    assert lotes.specialValueText() == SEM_LOTES == "1 — Não dividir em lotes"


@pytest.mark.parametrize(
    "com_diagnostico", [False, True], ids=["sem-diagnostico", "com-diagnostico"]
)
def test_d703_o_palpite_mede_na_particao_sugerida_e_nao_no_campo_da_correcao(
    qtbot: QtBot, tmp_path: Path, com_diagnostico: bool
) -> None:
    """O Palpite manda a partição que o «Diagnóstico» sugeriu, ou um lote — nunca a do campo.

    O Palpite é independente da tela de correção. **Par
    causal:** com o campo da aba mexido à mão para 3, o pedido leva 2 lotes e o corte do
    «Diagnóstico» quando ele sugeriu, e 1 lote sem corte quando não sugeriu. Antes levava os 3.
    """
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    pedidos: list[tuple[str, dict[str, Any]]] = []
    janela.chama = lambda _metodo, rota, **kw: pedidos.append((rota, dict(kw.get("campos") or {})))  # type: ignore[method-assign]
    janela.pipeline.n_lotes.setValue(3)
    assert janela.pipeline.escolhas_do_disparo()["n_lotes"] == 3, "âncora: o campo não ficou em 3"
    if com_diagnostico:
        janela._pre_preenche_com_a_proposta(
            "diagnostico", {"lotes": {"n_lotes": 2, "changepoints": [20]}}
        )
    assert janela._dispara_o_palpite(), "âncora: o Palpite não saiu"
    campos = [c for rota, c in pedidos if rota == ROTA_DO_PALPITE][-1]

    if com_diagnostico:
        assert campos["n_batches"] == 2, campos
        assert json.loads(campos["batch_cuts"]) == [20], campos
        return
    assert campos["n_batches"] == 1, campos
    assert "batch_cuts" not in campos, campos


@pytest.mark.parametrize("mexe", [False, True], ids=["sem-mexer", "mexe-nos-lotes"])
def test_d703_os_lotes_mudados_durante_a_espera_ficam(
    qtbot: QtBot, tmp_path: Path, mexe: bool
) -> None:
    """Os lotes que a pessoa muda depois de pedir as sugestões ficam quando o Palpite chega.

    **Par causal:** sem mexer, a chegada confirma os 2 lotes sugeridos e o corretor; mexendo para 3
    durante a espera, o corretor entra, os 3 ficam, e a frase diz que ficaram. A espera não sai com
    o toque, porque o Palpite é independente da aba.

    A Validação e o Diagnóstico entram em dia antes da sugestão de lotes
    (`_leituras_em_dia`, sem sugestão nenhuma): o clique mede o Palpite na hora, e a sugestão de
    lotes continua a ser só a que o teste põe.
    """
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    _leituras_em_dia(janela)
    janela._pre_preenche_com_a_proposta("diagnostico", {"lotes": {"n_lotes": 2}})
    bloco = _bloco_da_correcao(janela)
    bloco.botao.click()
    assert janela.pipeline.n_lotes.value() == 2, "âncora: o clique não pôs os lotes sugeridos"
    _aceita(janela, CHAVE_DO_PALPITE)
    if mexe:
        janela.pipeline.n_lotes.setValue(3)
        assert janela._depois_do_palpite is not None, "mexer nos lotes largou a espera"
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))

    assert janela.pipeline.corretor.currentText() == "loess", "o corretor do Palpite não entrou"
    if not mexe:
        assert janela.pipeline.n_lotes.value() == 2
        assert bloco.desfecho.text() == "Sugestões aplicadas: corretor «LOESS», 2 lotes."
        return
    assert janela.pipeline.n_lotes.value() == 3, "a chegada do Palpite escreveu por cima dos lotes"
    assert bloco.desfecho.text() == (
        "Sugestões aplicadas: corretor «LOESS». O número de lotes que você escolheu enquanto o "
        "Palpite media ficou como você deixou."
    )


# ----------------------------------------------------------------------------------------------
# A aba «Correção» nos valores padrão até a pessoa preencher ou seguir as sugestões
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("adota", [False, True], ids=["sem-adotar", "adota-e-troca"])
def test_d705_o_palpite_so_e_o_lado_da_maquina_depois_de_adotado(
    qtbot: QtBot, tmp_path: Path, adota: bool
) -> None:
    """O Palpite guardado e não seguido não é proposta que a pessoa trocou.

    A página de correção fica nos valores padrão até que o pesquisador a preencha ou escolha
    seguir as recomendações do app. **Par causal:** sem adotar, escolher
    SERRF à mão não é
    sobrescrita de nada — o Palpite não é a linha da máquina; adotando o LOESS pelo botão e trocando
    depois para SERRF, é sobrescrita do corretor, e a linha da máquina leva o LOESS.
    """
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    assert janela._dispara_o_palpite(), "âncora: o Palpite não saiu"
    _aceita(janela, CHAVE_DO_PALPITE)
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))
    assert janela.pipeline.corretor.currentText() == "none", "o Palpite escreveu no seletor"
    if adota:
        _bloco_da_correcao(janela).botao.click()
        assert janela.pipeline.corretor.currentText() == "loess", "âncora: o botão não adotou"
    janela.pipeline.corretor.setCurrentText("serrf")

    adotada = janela._recomendacao_adotada()
    divergentes = janela.pipeline.eixos_divergentes()
    if not adota:
        assert adotada is None, "o Palpite não seguido virou a linha da máquina"
        assert divergentes == (), f"escolher à mão foi gravado como sobrescrita: {divergentes}"
        return
    assert adotada is not None and adotada["correction_proposta"] == "loess"
    assert divergentes == ("correction",), divergentes
    tela = janela._campos_atuais()
    assert tela is not None, "âncora: sem par para montar o pedido"
    linha = janela.pipeline.proposta_do_pedido(tela[0], divergentes, adotada)
    assert "loess" in linha["proposta_json"], linha


def test_d705_trocar_o_par_esquece_o_corretor_sugerido(qtbot: QtBot, tmp_path: Path) -> None:
    """O corretor que o Palpite de A sugeriu não fica guardado para o par B.

    **Par causal:** antes da troca a sugestão está guardada; depois dela, não — e o botão sobre B
    mede o Palpite de B em vez de aplicar o de A.
    """
    janela, escolha, _rotas = _janela(qtbot, tmp_path)
    assert janela._dispara_o_palpite(), "âncora: o Palpite não saiu"
    _aceita(janela, CHAVE_DO_PALPITE)
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))
    assert janela.pipeline.corretor_sugerido == "loess", "âncora: a sugestão de A não ficou"

    _escolhe(janela, escolha, "b")

    assert janela.pipeline.corretor_sugerido is None, "a sugestão de A ficou guardada para B"


# ----------------------------------------------------------------------------------------------
# A cadeia Validação → Diagnóstico → Palpite, explícita
# ----------------------------------------------------------------------------------------------
#
# Nenhuma das três é obrigatória para usar o app, mas o Palpite exige o Diagnóstico e a
# Validação antes dele, nessa ordem, e a cadeia fica explícita na tela.
#
# `chama` é espiado e nada corre: cada resposta entra por `_recebe`, que é o caminho de produção, e
# a etapa seguinte só sai quando ela entra. Os corpos são os das rotas sobre os arquivos da tela.

ROTA_DA_VALIDACAO = dict(ROTAS_DA_LEITURA)["validacao"]
ROTA_DO_DIAGNOSTICO = dict(ROTAS_DA_LEITURA)["diagnostico"]


def _janela_que_guarda_os_campos(
    qtbot: QtBot, tmp_path: Path
) -> tuple[JanelaPrincipal, _Escolha, list[tuple[str, dict[str, Any]]]]:
    """`_janela`, com `chama` a guardar também os campos de cada pedido, na ordem."""
    janela, escolha, _rotas = _janela(qtbot, tmp_path)
    pedidos: list[tuple[str, dict[str, Any]]] = []
    janela.chama = lambda _metodo, rota, **kw: pedidos.append(  # type: ignore[method-assign]
        (rota, dict(kw.get("campos") or {}))
    )
    return janela, escolha, pedidos


def _rotas_de(pedidos: list[tuple[str, dict[str, Any]]]) -> list[str]:
    return [rota for rota, _campos in pedidos]


def test_d706_a_ordem_da_cadeia_e_a_dos_cartoes_e_a_tela_a_diz(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A ordem é uma só, e a tela a diz: o subtítulo da seção 2 e as notas dos cartões.

    A cadeia é explícita: a ordem da cadeia é a dos cartões na tela, e o subtítulo diz a
    ordem e a dependência. **Par causal nas notas:** a da Validação não nomeia nada que rode antes;
    as do Diagnóstico e do Palpite nomeiam o que roda antes de cada um.
    """
    assert tuple(chave for chave, _rota in ROTAS_DA_LEITURA) == ORDEM_DAS_LEITURAS
    janela, _escolha, _rotas = _janela(qtbot, tmp_path)
    preparo = janela.findChildren(AbaDePreparo)
    assert len(preparo) == 1, f"âncora: {len(preparo)} abas de preparo"
    assert preparo[0].secao_da_leitura.subtitulo.text() == SUBTITULO_DAS_LEITURAS
    assert SUBTITULO_DAS_LEITURAS.startswith("Validação → Diagnóstico → Palpite, nesta ordem")
    notas = {chave: janela.leitura.cartoes[chave].frase.text() for chave in ORDEM_DAS_LEITURAS}
    assert "Roda antes" not in notas["validacao"], notas["validacao"]
    assert notas["diagnostico"].endswith("Roda antes a Validação, se ela faltar."), notas
    assert notas[CHAVE_DO_PALPITE].endswith(
        "Roda antes a Validação e o Diagnóstico que faltarem."
    ), notas


@pytest.mark.parametrize("entregues", [0, 1, 2], ids=["nenhuma", "validacao", "as-duas"])
def test_d706_o_palpite_pede_as_tres_rotas_nesta_ordem_cada_uma_depois_da_anterior(
    qtbot: QtBot, tmp_path: Path, entregues: int
) -> None:
    """O clique no Palpite pede a Validação; a chegada dela, o Diagnóstico; a dele, o Palpite.

    **Par causal em três braços, que diferem só em quantas respostas chegaram:** cada rota sai
    depois da resposta da anterior, e nunca antes. E o Palpite sai na partição que o Diagnóstico
    acabou de sugerir.
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    corpos = {chave: _corpo_da_tela(janela, chave) for chave in ORDEM_DAS_LEITURAS[:2]}
    lotes = corpos["diagnostico"]["lotes"]
    assert lotes["n_lotes"] == 2, f"âncora: o Diagnóstico da factory não sugere lotes: {lotes}"

    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    for chave in ORDEM_DAS_LEITURAS[:entregues]:
        janela._recebe(chave, _resposta(200, corpos[chave]))

    esperadas = [ROTA_DA_VALIDACAO, ROTA_DO_DIAGNOSTICO, ROTA_DO_PALPITE][: entregues + 1]
    assert _rotas_de(pedidos) == esperadas
    if entregues < 2:
        return
    campos = pedidos[-1][1]
    assert campos["n_batches"] == lotes["n_lotes"], campos
    assert json.loads(campos["batch_cuts"]) == lotes["changepoints"], campos


@pytest.mark.parametrize(
    ("em_dia", "primeira", "etapa"),
    [
        ((), "validacao", "Etapa 1 de 3: validando os arquivos"),
        (("validacao",), "diagnostico", "Etapa 2 de 3: diagnosticando o sinal"),
        (("validacao", "diagnostico"), CHAVE_DO_PALPITE, "Etapa 3 de 3: medindo o Palpite"),
    ],
    ids=["nenhuma", "validacao", "as-duas"],
)
def test_d706_as_leituras_em_dia_nao_rodam_de_novo_e_o_cartao_diz_a_etapa(
    qtbot: QtBot, tmp_path: Path, em_dia: tuple[str, ...], primeira: str, etapa: str
) -> None:
    """As leituras em dia para esta matriz são reusadas, e o cartão do Palpite diz a etapa.

    **Par causal em três braços, que diferem só no que está em dia:** a primeira rota é a da
    primeira leitura fora de dia, e o cartão pedido mostra a etapa dela na ordem. No terceiro, o
    progresso do Palpite continua a dizer a etapa ao lado da contagem de corretores.
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    _leituras_em_dia(janela, em_dia)
    janela.leitura.botoes[CHAVE_DO_PALPITE].click()

    assert _rotas_de(pedidos) == [dict(ROTAS_DA_LEITURA)[primeira]]
    cartao = janela.leitura.cartoes[CHAVE_DO_PALPITE]
    assert cartao.manchete.text() == etapa
    assert not cartao.manchete.isHidden(), "a etapa não está à vista"
    if primeira != CHAVE_DO_PALPITE:
        return
    _aceita(janela, CHAVE_DO_PALPITE)
    assert cartao.manchete.text() == f"{etapa} — 0 de 3 corretores", cartao.manchete.text()
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))
    assert not cartao.manchete.text().startswith("Etapa"), "a etapa ficou depois do Palpite"
    assert janela._cadeia is None, "a cadeia ficou depois de o alvo chegar"


def _escreve_a_lista_cortada(raiz: Path) -> None:
    """A pasta `c`: o par A com a última linha da lista de amostras a menos.

    É uma recusa de verdade da rota — `SAMPLE_LIST_MISMATCH` —, e não um corpo escrito à mão.
    """
    (raiz / "c").mkdir()
    medidas = (raiz / "a" / f"measurements_{MODO}.csv").read_text(encoding="utf-8")
    lista = (raiz / "a" / f"Sample_List_{MODO}.csv").read_text(encoding="utf-8")
    (raiz / "c" / f"measurements_{MODO}.csv").write_text(medidas, encoding="utf-8", newline="")
    cortada = "".join(lista.splitlines(keepends=True)[:-1])
    (raiz / "c" / f"Sample_List_{MODO}.csv").write_text(cortada, encoding="utf-8", newline="")


@pytest.mark.parametrize("pasta", ["a", "c"], ids=["aceita", "recusada"])
def test_d706_a_validacao_recusada_para_a_cadeia_antes_do_diagnostico(
    qtbot: QtBot, tmp_path: Path, pasta: str
) -> None:
    """Arquivos recusados não vão ao Diagnóstico nem ao Palpite, e o cartão do Palpite diz por quê.

    **Par causal:** os mesmos passos sobre o par aceito pedem o Diagnóstico; sobre o recusado — a
    lista sem uma linha —, a cadeia para com a frase que aponta o cartão da Validação. E um segundo
    pedido do Palpite para na hora: a recusa está em dia, e validar de novo daria o mesmo.
    """
    janela, escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    _escreve_a_lista_cortada(tmp_path)
    _escolhe(janela, escolha, pasta)
    validacao = _corpo_da_tela(janela, "validacao")
    assert validacao["aceita"] is (pasta == "a"), f"âncora: {validacao.get('problemas')}"

    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    janela._recebe("validacao", _resposta(200, validacao))

    if pasta == "a":
        assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO, ROTA_DO_DIAGNOSTICO]
        return
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO], "a cadeia seguiu sobre arquivos recusados"
    frase = (
        "A «Validação» recusou estes arquivos, e o app parou antes do «Palpite»: o «Diagnóstico» "
        "e o «Palpite» só rodam sobre arquivos aceitos. Os problemas estão no cartão «Validação» — "
        "corrija-os e peça de novo."
    )
    cartao = janela.leitura.cartoes[CHAVE_DO_PALPITE]
    assert janela.leitura.rotulos[CHAVE_DO_PALPITE].text() == frase
    assert (cartao.selo.text(), cartao.manchete.text()) == (
        "Não rodou",
        "A Validação recusou os arquivos — veja o cartão dela",
    )
    assert janela.rotulo_de_desfecho.text() == frase, "o aviso não diz por que a cadeia parou"
    assert janela._cadeia is None
    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO], "a recusa em dia foi validada de novo"


@pytest.mark.parametrize("desfecho", ["200", "500", "sem-desfecho"])
def test_d706_o_diagnostico_que_nao_completa_para_a_cadeia_antes_do_palpite(
    qtbot: QtBot, tmp_path: Path, desfecho: str
) -> None:
    """Uma leitura anterior que não completa para a cadeia, e a frase diz qual e por quê.

    **Par causal em três braços:** o `200` do Diagnóstico pede o Palpite; um `500`, ou uma chamada
    que não produz desfecho de rota (`_falha`), param a cadeia, e o Diagnóstico sai de dia.
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    _leituras_em_dia(janela, ("validacao",))
    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    assert _rotas_de(pedidos) == [ROTA_DO_DIAGNOSTICO], (
        "âncora: a cadeia não começou no Diagnóstico"
    )

    if desfecho == "200":
        janela._recebe("diagnostico", _resposta(200, _corpo_da_tela(janela, "diagnostico")))
        assert _rotas_de(pedidos) == [ROTA_DO_DIAGNOSTICO, ROTA_DO_PALPITE]
        return
    if desfecho == "500":
        janela._recebe("diagnostico", _recusa_de_a(500))
        causa = _RECUSA_DE_A
    else:
        janela._falha("diagnostico", ConnectionError("recusado"))
        causa = "ConnectionError"
    assert _rotas_de(pedidos) == [ROTA_DO_DIAGNOSTICO], "o Palpite saiu sem o Diagnóstico"
    texto = janela.leitura.rotulos[CHAVE_DO_PALPITE].text()
    assert texto.startswith(
        "A leitura «Diagnóstico» não completou, e o app parou antes do «Palpite»: "
    ), texto
    assert causa in texto, texto
    assert janela.leitura.cartoes[CHAVE_DO_PALPITE].manchete.text() == (
        "A leitura «Diagnóstico» não completou"
    )
    assert janela.rotulo_de_desfecho.text() == texto
    assert janela._leitura_em_dia("diagnostico") is None
    assert janela._cadeia is None


@pytest.mark.parametrize("troca", [None, "par", "modo"], ids=["sem-troca", "par", "modo"])
def test_d706_a_troca_da_matriz_durante_a_cadeia_a_para(
    qtbot: QtBot, tmp_path: Path, troca: str | None
) -> None:
    """Outra matriz na tela com a Validação da cadeia em voo para a cadeia, com a frase.

    A resposta chega com a etiqueta do disparo, como a da piscina chegaria: com outra matriz, ela
    é descartada (`_leitura_desatualizada`), e sem a parada a cadeia esperaria para sempre — e com
    ela todo pedido seguinte. **Par causal em três braços:** sem a troca, a resposta pede o
    Diagnóstico; com o par ou o modo trocados, a cadeia para, e o pedido seguinte começa outra.
    """
    janela, escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    validacao = _corpo_da_tela(janela, "validacao")
    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    etiqueta = janela._etiqueta_da_leitura()
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO], "âncora"

    if troca == "par":
        _escolhe(janela, escolha, "b")
    elif troca == "modo":
        janela.arquivos.modo.setEditText("pos_hilic")
    janela._recebe(chave_do_pedido("validacao", etiqueta), _resposta(200, validacao))

    if troca is None:
        assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO, ROTA_DO_DIAGNOSTICO]
        return
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO]
    frase = (
        "Os arquivos ou o modo mudaram enquanto as leituras corriam, e o «Palpite» não foi "
        "concluído: peça de novo para os arquivos da tela."
    )
    assert janela.leitura.rotulos[CHAVE_DO_PALPITE].text() == frase
    assert janela.rotulo_de_desfecho.text() == frase
    assert janela._cadeia is None, "a cadeia ficou à espera de uma resposta descartada"
    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO, ROTA_DA_VALIDACAO], (
        "o pedido seguinte não começou outra cadeia"
    )


@pytest.mark.parametrize(
    ("em_dia", "primeira", "etapa"),
    [
        ((), "validacao", "Etapa 1 de 2: validando os arquivos"),
        (("validacao",), "diagnostico", "Etapa 2 de 2: diagnosticando o sinal"),
        (("validacao", "diagnostico"), "diagnostico", "Etapa 2 de 2: diagnosticando o sinal"),
    ],
    ids=["nenhuma", "validacao", "as-duas"],
)
def test_d706_o_diagnostico_roda_antes_a_validacao_que_falta_e_roda_sempre(
    qtbot: QtBot, tmp_path: Path, em_dia: tuple[str, ...], primeira: str, etapa: str
) -> None:
    """O cartão do Diagnóstico valida antes, se falta; e o Diagnóstico corre sempre.

    **Par causal em três braços:** sem nada em dia, a Validação sai primeiro e o Diagnóstico só
    depois da chegada dela; com a Validação em dia, o Diagnóstico sai no clique; e com ele também
    em dia sai de novo, porque a pessoa o pediu. Em nenhum dos três o Palpite sai.
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    _leituras_em_dia(janela, em_dia)
    janela.leitura.botoes["diagnostico"].click()

    cartao = janela.leitura.cartoes["diagnostico"]
    assert _rotas_de(pedidos) == [dict(ROTAS_DA_LEITURA)[primeira]]
    assert cartao.manchete.text() == etapa
    if primeira == "validacao":
        janela._recebe("validacao", _resposta(200, _corpo_da_tela(janela, "validacao")))
        assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO, ROTA_DO_DIAGNOSTICO]
        assert cartao.manchete.text() == "Etapa 2 de 2: diagnosticando o sinal"
    antes = _rotas_de(pedidos)
    janela._recebe("diagnostico", _resposta(200, _corpo_da_tela(janela, "diagnostico")))
    assert _rotas_de(pedidos) == antes, "o Diagnóstico pedido levou ao Palpite"
    assert janela._cadeia is None
    assert cartao.selo.text() != "Em curso", "o cartão ficou na etapa depois do Diagnóstico"


@pytest.mark.parametrize("cartao", ["validacao", "diagnostico"])
def test_d706_a_validacao_pedida_sozinha_nao_roda_mais_nada(
    qtbot: QtBot, tmp_path: Path, cartao: str
) -> None:
    """A Validação pedida pelo cartão dela é só a Validação, como antes da cadeia.

    **Par causal:** a mesma resposta da Validação, chegada depois do clique no cartão do
    Diagnóstico, pede o Diagnóstico; depois do clique no da Validação, não pede nada.
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    janela.leitura.botoes[cartao].click()
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO], "âncora: as duas começam na Validação"
    janela._recebe("validacao", _resposta(200, _corpo_da_tela(janela, "validacao")))

    if cartao == "diagnostico":
        assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO, ROTA_DO_DIAGNOSTICO]
        return
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO], "a Validação sozinha pediu outra leitura"
    assert janela._cadeia is None
    assert janela._leitura_em_dia("validacao") is True, (
        "a Validação pedida sozinha não ficou em dia"
    )


@pytest.mark.parametrize("segundo", [None, "diagnostico", CHAVE_DO_PALPITE])
def test_d706_um_segundo_pedido_com_a_cadeia_a_correr_nao_abre_outra(
    qtbot: QtBot, tmp_path: Path, segundo: str | None
) -> None:
    """Um segundo pedido com a cadeia a correr não abre outra; um alvo mais adiante a estende.

    **Par causal em três braços:** sem segundo pedido, a cadeia do Diagnóstico acaba nele; o mesmo
    pedido outra vez não muda nada; o Palpite pedido com o Diagnóstico a correr leva a mesma cadeia
    até ele — sem repetir a Validação —, e o cartão dele passa a mostrar a etapa.
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    corpos = {chave: _corpo_da_tela(janela, chave) for chave in ORDEM_DAS_LEITURAS[:2]}
    janela.leitura.botoes["diagnostico"].click()
    if segundo is not None:
        janela.leitura.botoes[segundo].click()
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO], "um segundo pedido abriu outra cadeia"
    palpite = janela.leitura.cartoes[CHAVE_DO_PALPITE]
    if segundo == CHAVE_DO_PALPITE:
        assert palpite.manchete.text() == "Etapa 1 de 3: validando os arquivos"

    janela._recebe("validacao", _resposta(200, corpos["validacao"]))
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO, ROTA_DO_DIAGNOSTICO]
    janela._recebe("diagnostico", _resposta(200, corpos["diagnostico"]))

    if segundo == CHAVE_DO_PALPITE:
        assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO, ROTA_DO_DIAGNOSTICO, ROTA_DO_PALPITE]
        assert palpite.manchete.text() == "Etapa 3 de 3: medindo o Palpite"
        return
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO, ROTA_DO_DIAGNOSTICO]
    assert janela._cadeia is None


@pytest.mark.parametrize("com_lotes", [False, True], ids=["sem-lotes", "com-lotes"])
def test_d706_seguir_sugestoes_roda_a_cadeia_e_poe_os_lotes_e_o_corretor(
    qtbot: QtBot, tmp_path: Path, com_lotes: bool
) -> None:
    """«Seguir sugestões do app» sem leitura nenhuma: valida, diagnostica, mede, e põe tudo.

    Um Palpite pedido sem o Diagnóstico roda o Diagnóstico antes. O botão espera com a etapa ao
    lado dele; os lotes entram quando o
    Diagnóstico chega — é o que o clique teria feito com ele em dia —, e o corretor quando o Palpite
    chega. **Par causal:** o Diagnóstico sem `lotes` deixa o campo em 1 e o Palpite num lote.
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    bloco = _bloco_da_correcao(janela)
    corpos = {chave: _corpo_da_tela(janela, chave) for chave in ORDEM_DAS_LEITURAS[:2]}
    if not com_lotes:
        corpos["diagnostico"].pop("lotes")
    painel = janela.pipeline
    # O repouso do campo é o mínimo, que se lê «1 — Não dividir em lotes».
    repouso = painel.n_lotes.minimum()
    assert (painel.n_lotes.value(), painel.corretor.currentText()) == (repouso, "none"), "âncora"
    lotes = 2 if com_lotes else repouso

    bloco.botao.click()
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO]
    assert not bloco.botao.isEnabled(), "o botão ficou ligado durante a cadeia"
    assert _motivo_do_botao(janela, bloco.botao).startswith(
        "Etapa 1 de 3: validando os arquivos. "
    ), _motivo_do_botao(janela, bloco.botao)
    janela._recebe("validacao", _resposta(200, corpos["validacao"]))
    assert _motivo_do_botao(janela, bloco.botao).startswith(
        "Etapa 2 de 3: diagnosticando o sinal. "
    )
    janela._recebe("diagnostico", _resposta(200, corpos["diagnostico"]))
    assert _motivo_do_botao(janela, bloco.botao).startswith("Etapa 3 de 3: medindo o Palpite — ")
    assert painel.n_lotes.value() == lotes, "os lotes do Diagnóstico não entraram na chegada dele"
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO, ROTA_DO_DIAGNOSTICO, ROTA_DO_PALPITE]
    assert pedidos[-1][1]["n_batches"] == (2 if com_lotes else 1), pedidos[-1][1]

    _aceita(janela, CHAVE_DO_PALPITE)
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))
    assert painel.corretor.currentText() == "loess", "o corretor do Palpite não entrou"
    assert painel.n_lotes.value() == lotes
    assert bloco.botao.isEnabled(), "o botão ficou desligado depois do Palpite"
    if com_lotes:
        assert bloco.desfecho.text() == "Sugestões aplicadas: corretor «LOESS», 2 lotes."
        return
    assert "O número de lotes ficou como estava" in bloco.desfecho.text(), bloco.desfecho.text()


def test_d706_a_espera_das_sugestoes_deixa_passar_o_palpite_da_cadeia_e_trava_a_correcao(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A trava com «Seguir sugestões do app» à espera: o Palpite DELA sai, a correção não.

    O botão espera antes de o Palpite sair, e a trava que via a espera recusaria o
    Palpite da própria cadeia. **Par causal sob a mesma espera:** a correção é recusada com a frase
    do Palpite, e o Palpite da cadeia sai; com ele em voo, um segundo Palpite é recusado; e com ele
    chegado, a correção sai — a âncora de que a recusa de antes não era vácua.
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    corpos = {chave: _corpo_da_tela(janela, chave) for chave in ORDEM_DAS_LEITURAS[:2]}
    _bloco_da_correcao(janela).botao.click()
    assert janela._depois_do_palpite is not None, "âncora: o clique não ficou à espera"

    assert janela._motivo_para_nao_corrigir(CHAVE_DA_ETAPA[1]) == PALPITE_EM_CURSO
    janela.pipeline.botoes_de_etapa[1].click()
    assert ROTA_DA_ETAPA[1] not in _rotas_de(pedidos), "a correção saiu durante a espera"
    assert janela._motivo_para_nao_corrigir(CHAVE_DO_PALPITE) is None

    janela._recebe("validacao", _resposta(200, corpos["validacao"]))
    janela._recebe("diagnostico", _resposta(200, corpos["diagnostico"]))
    assert _rotas_de(pedidos)[-1] == ROTA_DO_PALPITE, (
        f"a trava recusou o Palpite da cadeia: {janela.leitura.rotulos[CHAVE_DO_PALPITE].text()}"
    )
    assert janela._motivo_para_nao_corrigir(CHAVE_DO_PALPITE) == PALPITE_EM_CURSO, (
        "um segundo Palpite passaria com este no `POST`"
    )
    _aceita(janela, CHAVE_DO_PALPITE)
    assert janela._motivo_para_nao_corrigir(CHAVE_DO_PALPITE) == PALPITE_EM_CURSO
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))

    assert janela._motivo_para_nao_corrigir(CHAVE_DA_ETAPA[1]) is None, "a trava ficou presa"
    # A recusa de antes desligou o botão com o motivo ao lado; quem o religa é o relógio das etapas,
    # que só corre com a janela visível — aqui, uma volta dele à mão.
    janela._atualiza_as_etapas()
    janela.pipeline.botoes_de_etapa[1].click()
    assert ROTA_DA_ETAPA[1] in _rotas_de(pedidos), "âncora: sem a espera, a correção não saiu"


@pytest.mark.parametrize("mexe", [False, True], ids=["sem-mexer", "mexe-durante-a-cadeia"])
def test_d706_os_lotes_mudados_durante_a_cadeia_ficam(
    qtbot: QtBot, tmp_path: Path, mexe: bool
) -> None:
    """Os lotes que a pessoa muda entre o clique e a chegada do Diagnóstico ficam.

    A frase da espera promete *«os campos são preenchidos quando ele chegar, menos o que você mudar
    até lá»*. **Par causal:** sem mexer, os 2 lotes do Diagnóstico entram quando ele chega; mexendo
    para 3 depois da Validação e antes do Diagnóstico, os 3 ficam até o fim, e a frase diz que
    ficaram. A primeira versão da cadeia escrevia os 2 por cima.
    """
    janela, _escolha, _pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    bloco = _bloco_da_correcao(janela)
    corpos = {chave: _corpo_da_tela(janela, chave) for chave in ORDEM_DAS_LEITURAS[:2]}
    assert corpos["diagnostico"]["lotes"]["n_lotes"] == 2, "âncora: o Diagnóstico não sugere 2"
    painel = janela.pipeline

    bloco.botao.click()
    janela._recebe("validacao", _resposta(200, corpos["validacao"]))
    if mexe:
        painel.n_lotes.setValue(3)
    janela._recebe("diagnostico", _resposta(200, corpos["diagnostico"]))
    _aceita(janela, CHAVE_DO_PALPITE)
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))

    assert painel.corretor.currentText() == "loess", "o corretor do Palpite não entrou"
    if not mexe:
        assert painel.n_lotes.value() == 2
        assert bloco.desfecho.text() == "Sugestões aplicadas: corretor «LOESS», 2 lotes."
        return
    assert painel.n_lotes.value() == 3, "os lotes do Diagnóstico escreveram por cima da escolha"
    assert bloco.desfecho.text() == (
        "Sugestões aplicadas: corretor «LOESS». O número de lotes que você escolheu enquanto o "
        "Palpite media ficou como você deixou."
    )


# ----------------------------------------------------------------------------------------------
# A Correção feita sem «Seguir sugestões do app» é gravada como `sem_proposta`
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("segue", [False, True], ids=["sem-seguir", "segue-as-sugestoes"])
def test_d707_a_analise_declara_sem_proposta_so_quando_nada_foi_seguido(
    qtbot: QtBot, tmp_path: Path, segue: bool
) -> None:
    """O `/execucoes` da Análise leva `sem_proposta` quando a Correção não seguiu sugestão.

    A aba «Correção» fica nos valores padrão até a pessoa a preencher ou seguir
    as sugestões. Sem o clique não há proposta adotada, e a execução era gravada como
    `nao_sobrescreveu` — *«a proposta da máquina foi seguida»*. **Par causal pelo caminho de
    produção** (`_dispara_etapa`): com o Palpite guardado e não seguido, a Correção feita e a
    Análise sobre a corrigida declaram que não houve proposta; seguindo as sugestões, nenhuma das
    duas declara, e o par fica sem eixo — a proposta foi seguida.
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    assert janela._dispara_o_palpite(), "âncora: o Palpite não saiu"
    _aceita(janela, CHAVE_DO_PALPITE)
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))
    if segue:
        _bloco_da_correcao(janela).botao.click()
        assert janela.pipeline.corretor.currentText() == "loess", "âncora: o botão não adotou"

    janela._dispara_etapa(ETAPAS[0])
    janela._etapas.concluiu(ETAPAS[0], correcao=janela.pipeline.corretor.currentText())
    feita = janela._etapas.feita(ETAPAS[0])
    assert feita is not None and feita.campos is not None, "âncora: a Correção não ficou feita"
    del pedidos[:]
    janela._dispara_etapa(ETAPAS[-1])
    execucoes = [campos for rota, campos in pedidos if rota == "/execucoes"]
    assert len(execucoes) == 1, f"âncora: a Análise não pediu o `/execucoes` uma vez: {pedidos}"

    for onde, campos in (("na Correção feita", feita.campos), ("no `/execucoes`", execucoes[0])):
        assert "proposta_json" not in campos, f"{onde}: sem eixo trocado não há linha da máquina"
        if segue:
            assert CAMPO_SEM_PROPOSTA not in campos, (
                f"{onde}: a sugestão seguida virou sem proposta"
            )
        else:
            assert campos.get(CAMPO_SEM_PROPOSTA) is True, (
                f"{onde}: a Correção sem sugestão seguida não declarou que não houve proposta"
            )


# ----------------------------------------------------------------------------------------------
# O campo trocado durante a espera de «Seguir sugestões do app» é sobrescrita, e a cadeia
# estendida que para tira o «Em curso» de todo cartão que a pediu
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mexe", ["nada", "corretor", "lotes"])
def test_d710_o_campo_trocado_durante_a_espera_e_gravado_como_sobrescrita(
    qtbot: QtBot, tmp_path: Path, mexe: str
) -> None:
    """Quem pede as sugestões e troca um campo durante a espera sobrescreveu a proposta.

    Trocar um campo durante a espera, depois de clicar «Seguir sugestões do app», conta como
    sobrescrever a proposta da máquina. Antes o campo
    mantido ficava fora de `_proposta`, e a Correção era gravada como
    `nao_sobrescreveu` — *«a proposta da máquina foi seguida»* —, com a escolha da pessoa na linha
    da máquina.

    **Par causal em três braços, pelo caminho de produção** (`_dispara_etapa`): sem mexer, a
    proposta seguida vai sem eixo e sem linha da máquina; mexendo no corretor durante o Palpite, ou
    nos lotes antes de o Diagnóstico chegar, o eixo trocado vai em `campos_sobrescritos` e a linha
    da máquina leva o que ela sugeriu — o par que a rota grava como `sobrescreveu_discordando`
    (`test_u5_os_estados_alcancaveis_do_par_chegam_ao_banco_pelo_payload_do_painel`).
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    painel = janela.pipeline
    bloco = _bloco_da_correcao(janela)
    corpos = {chave: _corpo_da_tela(janela, chave) for chave in ORDEM_DAS_LEITURAS[:2]}
    assert corpos["diagnostico"]["lotes"]["n_lotes"] == 2, "âncora: o Diagnóstico não sugere 2"

    bloco.botao.click()
    janela._recebe("validacao", _resposta(200, corpos["validacao"]))
    if mexe == "lotes":
        painel.n_lotes.setValue(3)
    janela._recebe("diagnostico", _resposta(200, corpos["diagnostico"]))
    _aceita(janela, CHAVE_DO_PALPITE)
    if mexe == "corretor":
        painel.corretor.setCurrentText("serrf")
    _termina(janela, CHAVE_DO_PALPITE, dict(PALPITE_DE_A))
    esperado = {"nada": ("loess", 2), "corretor": ("serrf", 2), "lotes": ("loess", 3)}[mexe]
    assert (painel.corretor.currentText(), painel.n_lotes.value()) == esperado, (
        "âncora: os campos não ficaram como a pessoa os deixou"
    )

    janela._dispara_etapa(ETAPAS[0])
    janela._etapas.concluiu(ETAPAS[0], correcao=painel.corretor.currentText())
    feita = janela._etapas.feita(ETAPAS[0])
    assert feita is not None and feita.campos is not None, "âncora: a Correção não ficou feita"
    del pedidos[:]
    janela._dispara_etapa(ETAPAS[-1])
    execucoes = [campos for rota, campos in pedidos if rota == "/execucoes"]
    assert len(execucoes) == 1, f"âncora: a Análise não pediu o `/execucoes` uma vez: {pedidos}"

    for onde, campos in (("na Correção feita", feita.campos), ("no `/execucoes`", execucoes[0])):
        assert CAMPO_SEM_PROPOSTA not in campos, (
            f"{onde}: as sugestões pedidas viraram sem proposta"
        )
        trocados = json.loads(campos["campos_sobrescritos"])
        if mexe == "nada":
            assert trocados == [], f"{onde}: a proposta seguida saiu com eixo trocado: {trocados}"
            assert "proposta_json" not in campos, (
                f"{onde}: sem eixo trocado não há linha da máquina"
            )
            continue
        eixo = {"corretor": "correction", "lotes": "n_batches"}[mexe]
        assert trocados == [eixo], f"{onde}: o campo trocado na espera não contou: {trocados}"
        assert "proposta_json" in campos, f"{onde}: o eixo trocado foi sem a linha da máquina"
        spec = json.loads(json.loads(campos["proposta_json"])["spec_json"])
        assert (spec["correction"], spec["n_batches"]) == ("loess", 2), (
            f"{onde}: a linha da máquina não leva o que ela sugeriu: {spec}"
        )


@pytest.mark.parametrize("estende", [False, True], ids=["so-o-diagnostico", "estendida-ao-palpite"])
def test_d710_a_cadeia_que_para_tira_o_em_curso_de_todo_cartao_que_a_pediu(
    qtbot: QtBot, tmp_path: Path, estende: bool
) -> None:
    """O cartão do Diagnóstico não fica «Em curso» depois de a cadeia estendida parar.

    **Par causal:** pedido só o Diagnóstico, a Validação recusada deixa o cartão dele em «Não
    rodou»; pedido também o Palpite com a Validação ainda a correr, a cadeia passa a ir até ele —
    e antes parava escrevendo só no cartão do Palpite: o do Diagnóstico ficava em «Em curso»,
    sem nada a correr.
    """
    janela, escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    _escreve_a_lista_cortada(tmp_path)
    _escolhe(janela, escolha, "c")
    validacao = _corpo_da_tela(janela, "validacao")
    assert validacao["aceita"] is False, f"âncora: a lista cortada foi aceita: {validacao}"

    janela.leitura.botoes["diagnostico"].click()
    diagnostico = janela.leitura.cartoes["diagnostico"]
    assert diagnostico.selo.text() == "Em curso", (
        "âncora: o cartão do Diagnóstico não mostra a etapa"
    )
    if estende:
        janela.leitura.botoes[CHAVE_DO_PALPITE].click()
        assert janela._cadeia is not None and janela._cadeia.alvo == CHAVE_DO_PALPITE, (
            "âncora: o Palpite não estendeu a cadeia"
        )
    janela._recebe("validacao", _resposta(200, validacao))

    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO], "a cadeia seguiu sobre arquivos recusados"
    assert janela._cadeia is None, "a cadeia não parou"
    assert diagnostico.selo.text() == "Não rodou", (
        f"o cartão do Diagnóstico ficou em {diagnostico.selo.text()!r} — "
        f"{diagnostico.manchete.text()!r}, sem nada a correr"
    )
    assert "o app parou antes do «Diagnóstico»" in janela.leitura.rotulos["diagnostico"].text()
    if estende:
        palpite = janela.leitura.cartoes[CHAVE_DO_PALPITE]
        assert palpite.selo.text() == "Não rodou"
        assert "o app parou antes do «Palpite»" in janela.leitura.rotulos[CHAVE_DO_PALPITE].text()


@pytest.mark.parametrize("estende", [False, True], ids=["so-o-diagnostico", "estendida-ao-palpite"])
def test_d710_o_cartao_do_alvo_anterior_acompanha_a_etapa_da_cadeia_estendida(
    qtbot: QtBot, tmp_path: Path, estende: bool
) -> None:
    """O cartão do Diagnóstico diz «diagnosticando» quando a cadeia estendida chega a ele.

    **Par causal:** pedido só o Diagnóstico, o cartão dele é o alvo e mostra a etapa em voo; pedido
    também o Palpite durante a Validação, a etapa passava a ser escrita só no cartão do Palpite, e
    o do Diagnóstico ficava em «Etapa 1 de 2: validando os arquivos» com o Diagnóstico já a correr.
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    validacao = _corpo_da_tela(janela, "validacao")
    janela.leitura.botoes["diagnostico"].click()
    if estende:
        janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    janela._recebe("validacao", _resposta(200, validacao))
    assert _rotas_de(pedidos) == [ROTA_DA_VALIDACAO, ROTA_DO_DIAGNOSTICO], "âncora: sem Diagnóstico"

    diagnostico = janela.leitura.cartoes["diagnostico"]
    assert (diagnostico.selo.text(), diagnostico.manchete.text()) == (
        "Em curso",
        "Etapa 2 de 2: diagnosticando o sinal",
    ), "o cartão do Diagnóstico não acompanhou a etapa da cadeia"
    if estende:
        palpite = janela.leitura.cartoes[CHAVE_DO_PALPITE]
        assert palpite.manchete.text() == "Etapa 2 de 3: diagnosticando o sinal"


@pytest.mark.parametrize("desfecho", ["500", "sem-desfecho"])
def test_d710_o_alvo_anterior_que_nao_completa_diz_so_que_nao_completou(
    qtbot: QtBot, tmp_path: Path, desfecho: str
) -> None:
    """O Diagnóstico de uma cadeia estendida ao Palpite que não completa.

    **Par causal em dois braços:** com o `500`, a resposta chega ao cartão do Diagnóstico pela
    porta de sempre (`leitura.mostra`), e a parada não escreve por cima dela; com a chamada que
    morre sem desfecho de rota (`_falha`), nada chega ao cartão, e é a parada que o tira de «Em
    curso» — com a frase de que ele não completou, e não *«parou antes do «Diagnóstico»»*, que no
    cartão dele não descreveria nada.
    """
    janela, _escolha, pedidos = _janela_que_guarda_os_campos(qtbot, tmp_path)
    _leituras_em_dia(janela, ("validacao",))
    janela.leitura.botoes["diagnostico"].click()
    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    assert _rotas_de(pedidos) == [ROTA_DO_DIAGNOSTICO], "âncora: a cadeia não está no Diagnóstico"
    diagnostico = janela.leitura.cartoes["diagnostico"]
    assert diagnostico.selo.text() == "Em curso", "âncora: o cartão do Diagnóstico não espera"

    if desfecho == "500":
        janela._recebe("diagnostico", _recusa_de_a(500))
    else:
        janela._falha("diagnostico", ConnectionError("recusado"))

    assert janela._cadeia is None, "a cadeia não parou"
    assert diagnostico.selo.text() != "Em curso", "o cartão do Diagnóstico ficou em «Em curso»"
    texto = janela.leitura.rotulos["diagnostico"].text()
    assert "parou antes do «Diagnóstico»" not in texto, texto
    if desfecho == "500":
        assert not texto.startswith("A leitura «Diagnóstico» não completou"), (
            f"a parada escreveu por cima da resposta que chegou: {texto!r}"
        )
        return
    assert diagnostico.selo.text() == "Não rodou"
    assert texto.startswith("A leitura «Diagnóstico» não completou: "), texto
    assert "ConnectionError" in texto, texto

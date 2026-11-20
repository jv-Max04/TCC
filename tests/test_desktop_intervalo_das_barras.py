"""O intervalo por barra na janela, por escolha.

O intervalo liga por escolha: uma caixa desmarcada por padrão, junto do botão que pede
a medição dos corretores; quem não a marca não paga nada, e quem a marca recebe o jackknife-t com
reajuste em cada barra. Este arquivo afirma, do lado da janela:

- **a caixa nasce desmarcada, e o disparo manda o que ela diz** — falso sem ela, verdadeiro com ela;
- **o rótulo da caixa diz o custo pelos QCs do estudo carregado** — `(n+1)/2` vezes —, e a
  estimativa de tempo sob o botão multiplica-se pelo mesmo fator quando ela está marcada;
- **a caixa NÃO muda o Palpite** — o pedido do Palpite é o mesmo com ela marcada —, e a regra de
  empate dele continua a mesma (um teste da constante, que não exercita a caixa);
- **a dica diz o mínimo de QCs de cada corretor**, pelo núcleo, e que o fator é contagem;
- **o painel da correção desenha o intervalo que o corpo traz**, e sem ele a figura não tem traço.

**Os arquivos são da factory, e nada corre corretor:** o transporte espiado não encaminha o pedido,
e o painel dos gráficos recebe o corpo à mão. O que se afirma é a fiação da tela.

## Escopo negativo

**Não é o executável congelado nem o Windows**: Qt `offscreen`, com a API em processo.
"""

from __future__ import annotations

from typing import Any

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

from refino.api.app import build_app  # noqa: E402
from refino.correction.loess import LoessCorrector  # noqa: E402
from refino.correction.none import NoneCorrector  # noqa: E402
from refino.decision.palpite import LIMIAR_DE_EMPATE_DO_PALPITE, decide_o_palpite  # noqa: E402
from refino.desktop.janela.arquivos import ParDeArquivos  # noqa: E402
from refino.desktop.janela.componentes import com_maiuscula  # noqa: E402
from refino.desktop.janela.leitura import (  # noqa: E402
    CHAVE_DO_PALPITE,
    ROTA_DO_PALPITE,
    ROTAS_DA_LEITURA,
)
from refino.desktop.janela.pipeline import (  # noqa: E402
    PainelDePipeline,
    rotulo_da_caixa_do_intervalo,
)
from refino.desktop.janela.principal import ROTA_DA_MEDICAO, JanelaPrincipal  # noqa: E402
from refino.interface.cliente import Resposta  # noqa: E402
from refino.interface.fluxo import estimativa_da_medicao  # noqa: E402
from refino.metrics.crossval import jackknife_qc_minimum  # noqa: E402
from refino.plots.correcao import ROTULO_DO_INTERVALO  # noqa: E402
from refino.plots.recorte import decimal  # noqa: E402
from tests.arnes_de_figura import series, texto_corrido  # noqa: E402
from tests.factories import base_spec, render_pair  # noqa: E402
from tests.test_desktop_figuras import (  # noqa: E402
    _com_par_de_pcas,
    _medicao,
    _painel_da_correcao,
)

MODO = "neg_hilic"
N_QC = 7


def _janela_espiada(qtbot: Any) -> tuple[JanelaPrincipal, list[dict[str, Any]]]:
    """A janela com o par da factory e `chama` espiado — o pedido NÃO segue para a API.

    Não encaminhar é o que deixa estes testes correrem em milissegundos: o que se afirma é o que a
    janela MANDA, e a rota tem os testes dela em `tests/test_api_intervalo_das_barras.py`.
    """
    janela = JanelaPrincipal(
        TestClient(build_app(), raise_server_exceptions=False), escolhedor=lambda: None
    )
    qtbot.addWidget(janela)
    medidas, lista = render_pair(base_spec(n_features=5, n_samples=40, n_qcs=N_QC))
    janela.arquivos._par = ParDeArquivos(
        "measurements.csv", medidas.encode("utf-8"), "Sample_List.csv", lista.encode("utf-8")
    )
    janela.arquivos.modo.setCurrentText(MODO)
    enviados: list[dict[str, Any]] = []

    def espia(metodo: str, rota: str, **kwargs: Any) -> None:
        enviados.append(
            {"metodo": metodo, "rota": rota, "campos": dict(kwargs.get("campos") or {})}
        )

    janela.chama = espia  # type: ignore[method-assign]
    return janela, enviados


def _campos_para(enviados: list[dict[str, Any]], rota: str) -> list[dict[str, Any]]:
    return [e["campos"] for e in enviados if e["rota"] == rota]


def _sete_qcs(painel: PainelDePipeline, n_qc: int = N_QC) -> None:
    """A validação de uma corrida de `4 * n_qc` injeções com `n_qc` QCs, como a janela a recebe."""
    ordens = list(range(1, 4 * n_qc + 1))
    painel.define_amostras(ordens, [i % 4 == 0 for i in range(4 * n_qc)])


# ---------------------------------------------------------------------------
# A caixa e o disparo
# ---------------------------------------------------------------------------


def test_a_caixa_nasce_desmarcada_e_o_disparo_manda_o_que_ela_diz(qtbot: Any) -> None:
    """Desmarcada por padrão, e o campo viaja nos dois sentidos — o par causal é o mesmo
    clique com a caixa nos dois estados."""
    janela, enviados = _janela_espiada(qtbot)
    assert not janela.pipeline.intervalo_por_barra.isChecked(), "a caixa nasceu marcada"

    janela.pipeline.medir_botao.click()
    janela.pipeline.intervalo_por_barra.setChecked(True)
    janela.pipeline.medir_botao.click()

    pedidos = _campos_para(enviados, ROTA_DA_MEDICAO)
    assert len(pedidos) == 2, f"os cliques não chegaram à rota da medição: {enviados}"
    assert pedidos[0]["intervalo_por_barra"] is False
    assert pedidos[1]["intervalo_por_barra"] is True


def test_a_caixa_esta_no_grupo_da_correcao_junto_do_botao_da_medicao(qtbot: Any) -> None:
    """Ao lado de quem ela muda: no mesmo grupo do botão «Medir qual corretor vence»."""
    janela, _enviados = _janela_espiada(qtbot)
    caixa, botao = janela.pipeline.intervalo_por_barra, janela.pipeline.medir_botao
    assert caixa.parent() is botao.parent(), (caixa.parent(), botao.parent())


# ---------------------------------------------------------------------------
# O custo, dito pelos QCs do estudo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n_qc", "fator"), [(None, None), (7, "4"), (8, "4,5"), (21, "11")])
def test_o_rotulo_da_caixa_diz_o_custo_pelos_qcs(n_qc: int | None, fator: str | None) -> None:
    """`(n+1)/2` vezes, com o `n` do estudo carregado; sem a validação, a conta e não um número."""
    texto = rotulo_da_caixa_do_intervalo(n_qc)
    if fator is None:
        assert "(n+1)/2" in texto, texto
    else:
        assert f"{fator} vezes" in texto and f"{n_qc} QCs" in texto, texto


def test_a_dica_da_caixa_diz_o_minimo_de_cada_corretor_e_que_o_fator_e_contagem(
    qtbot: Any,
) -> None:
    """A dica diz quando não há intervalo pelo mínimo que o NÚCLEO aplica a cada corretor — quatro
    QCs por lote, cinco com o LOESS —, e que o fator do rótulo é contagem de ajustes, com o único
    relógio medido ao lado: o do LOESS. Os números saem de `jackknife_qc_minimum`, e não de uma
    cópia na tela que divergiria no dia em que o mínimo mudasse."""
    janela, _enviados = _janela_espiada(qtbot)
    dica = janela.pipeline.intervalo_por_barra.toolTip()
    piso = jackknife_qc_minimum(NoneCorrector())
    do_loess = jackknife_qc_minimum(LoessCorrector())
    assert piso != do_loess, "âncora: os dois mínimos coincidem, e a dica não os separaria"
    assert f"menos de {piso} QCs num lote" in dica, dica
    assert f"{do_loess} com o LOESS" in dica, dica
    assert "contagem de ajustes" in dica and "SERRF não foi medido" in dica, dica


def test_validada_a_matriz_a_caixa_diz_o_fator_dela(qtbot: Any) -> None:
    janela, _enviados = _janela_espiada(qtbot)
    painel = janela.pipeline
    assert "(n+1)/2" in painel.intervalo_por_barra.text(), "âncora: o repouso já dizia um fator"
    _sete_qcs(painel)
    assert painel.intervalo_por_barra.text() == rotulo_da_caixa_do_intervalo(N_QC)
    painel.define_amostras(None, None)
    assert "(n+1)/2" in painel.intervalo_por_barra.text(), "a validação apagada deixou o fator"


def test_marcada_a_estimativa_sob_o_botao_multiplica_pelo_mesmo_fator(qtbot: Any) -> None:
    """A estimativa de tempo é a de `estimativa_da_medicao` com o intervalo e os QCs da matriz, e
    volta à de antes quando a caixa é desmarcada — sem nova validação."""
    janela, _enviados = _janela_espiada(qtbot)
    painel = janela.pipeline
    _sete_qcs(painel)
    serial = painel._serial_agora()
    sem = painel.mostra_palpite_da_medicao(6035)
    assert sem == com_maiuscula(estimativa_da_medicao(6035, serial=serial))

    painel.intervalo_por_barra.setChecked(True)
    com = painel.rotulo_do_palpite.text()
    assert com == com_maiuscula(
        estimativa_da_medicao(6035, serial=serial, intervalo=True, n_qc=N_QC)
    ), com
    assert com != sem and "4 vezes" in com, com

    painel.intervalo_por_barra.setChecked(False)
    assert painel.rotulo_do_palpite.text() == sem


# ---------------------------------------------------------------------------
# A caixa não muda o Palpite
# ---------------------------------------------------------------------------


def _resposta(codigo: int, corpo: Any) -> Resposta:
    return Resposta(
        codigo=codigo, ok=codigo < 400, causa=None, mensagem="", corpo=corpo, conteudo=None
    )


def test_a_caixa_NAO_muda_o_pedido_do_palpite(qtbot: Any) -> None:  # noqa: N802
    """O Palpite mede em 200 features sem intervalo nenhum, com a caixa marcada ou não: o pedido
    é o mesmo, campo a campo, e não leva o do intervalo. A âncora é que o pedido saiu duas vezes.

    O clique no Palpite roda antes a Validação e o Diagnóstico que faltarem,
    e um segundo pedido com a cadeia a correr não abre outra. Os dois entram em dia por `_recebe`
    antes do primeiro clique — os corpos das rotas sobre o par da tela, o do Diagnóstico sem
    `lotes` —, e o primeiro Palpite termina antes do segundo clique: aqui `chama` não segue para a
    API, e sem o desfecho a cadeia do primeiro continuaria a correr.
    """
    janela, enviados = _janela_espiada(qtbot)
    cliente = TestClient(build_app())
    for chave in ("validacao", "diagnostico"):
        http = cliente.post(
            dict(ROTAS_DA_LEITURA)[chave],
            files=janela.arquivos.arquivos_para_envio(),
            data=janela.arquivos.campos_para_envio(),
        )
        assert http.status_code == 200, http.text
        corpo = http.json()
        corpo.pop("lotes", None)
        janela._recebe(chave, _resposta(200, corpo))
    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    janela._recebe(CHAVE_DO_PALPITE, _resposta(422, {"detail": "o primeiro Palpite terminou"}))
    janela.pipeline.intervalo_por_barra.setChecked(True)
    janela._palpite_pedido = None
    janela.leitura.botoes[CHAVE_DO_PALPITE].click()

    assert [e["rota"] for e in enviados] == [ROTA_DO_PALPITE, ROTA_DO_PALPITE], enviados
    pedidos = _campos_para(enviados, ROTA_DO_PALPITE)
    assert len(pedidos) == 2, f"o Palpite não foi pedido duas vezes: {enviados}"
    assert pedidos[0] == pedidos[1], pedidos
    assert "intervalo_por_barra" not in pedidos[0], pedidos[0]


def test_a_regra_de_empate_do_palpite_continua_a_de_D585() -> None:  # noqa: N802
    """τ = 0,167, derivado do ruído do sorteio de 200 features: abaixo dele o Palpite
    abstém-se, acima não.

    **Este teste fixa o VALOR de τ e a regra, e não exercita a caixa** — a regra é função pura e
    não lê a tela. Até a revisão ele era parametrizado pela caixa nos dois estados, e essa dimensão
    não tinha como reprovar: a caixa não entrava no que ele afirma. A prova de que a caixa não muda
    o Palpite é `test_a_caixa_NAO_muda_o_pedido_do_palpite`, que compara o pedido campo a campo.
    """
    assert LIMIAR_DE_EMPATE_DO_PALPITE == 0.167
    abaixo = decide_o_palpite(
        {"none": 1.0, "loess": 1.0 * (1 + 0.166), "serrf": 2.0}, n_features_medidas=200
    )
    acima = decide_o_palpite(
        {"none": 1.0, "loess": 1.0 * (1 + 0.168), "serrf": 2.0}, n_features_medidas=200
    )
    assert abaixo.abstem and not acima.abstem
    assert abaixo.limiar == acima.limiar == LIMIAR_DE_EMPATE_DO_PALPITE


# ---------------------------------------------------------------------------
# O painel da correção desenha o que o corpo traz
# ---------------------------------------------------------------------------


def _intervalos_do_corpo(scores: dict[str, float | None]) -> dict[str, dict[str, Any]]:
    return {
        nome: {
            "metodo": "jackknife-t com reajuste",
            "nivel": 0.95,
            "n_qc": N_QC,
            "inferior": valor - 0.02,
            "superior": valor + 0.03,
            "erro_padrao": 0.01,
            "motivo": None,
        }
        for nome, valor in scores.items()
        if valor is not None
    }


def test_o_painel_desenha_o_intervalo_que_o_corpo_traz_e_sem_ele_nao(qtbot: Any) -> None:
    """Par causal: o mesmo corpo com e sem `intervalos` — com ele, uma série de intervalo por
    barra e o rodapé a dizê-lo; sem ele, nenhuma."""
    sem = _painel_da_correcao(qtbot, par=_com_par_de_pcas(77), medicao=_medicao())
    corpo = _medicao()
    com = _painel_da_correcao(
        qtbot,
        par=_com_par_de_pcas(77),
        medicao={**corpo, "intervalos": _intervalos_do_corpo(corpo["scores"])},
    )
    barras_sem = sem.figuras.canvas.figure.axes[0]
    barras_com = com.figuras.canvas.figure.axes[0]
    assert not [s for s in series(barras_sem) if s.rotulo.startswith(ROTULO_DO_INTERVALO)]
    desenhados = [s.rotulo for s in series(barras_com) if s.rotulo.startswith(ROTULO_DO_INTERVALO)]
    finitos = [n for n, v in corpo["scores"].items() if v is not None]
    assert sorted(desenhados) == sorted(f"{ROTULO_DO_INTERVALO} {n}" for n in finitos), desenhados
    rodape = texto_corrido(com.figuras.canvas.figure.texts[-1].get_text())
    assert "Traço:" in rodape and "nominal" in rodape, rodape
    # **A margem do corpo chega ao rodapé**, e a janela é quem a lê: sem a leitura, o rodapé diria
    # «indefinida» com a margem presente no corpo — a planta que a tirava passava verde.
    margem = decimal(corpo["margem"] * 100.0, casas=3)
    assert f"Margem relativa dos dois melhores: {margem}% do melhor" in rodape, rodape


def test_um_corpo_com_intervalos_null_e_a_figura_de_antes(qtbot: Any) -> None:
    """`null` no campo é a medição sem a caixa: nenhum traço, e o rodapé de antes."""
    sem = _painel_da_correcao(qtbot, par=_com_par_de_pcas(77), medicao=_medicao())
    nulo = _painel_da_correcao(
        qtbot, par=_com_par_de_pcas(77), medicao={**_medicao(), "intervalos": None}
    )
    assert texto_corrido(sem.figuras.canvas.figure.texts[-1].get_text()) == texto_corrido(
        nulo.figuras.canvas.figure.texts[-1].get_text()
    )
    assert not [
        s
        for s in series(nulo.figuras.canvas.figure.axes[0])
        if s.rotulo.startswith(ROTULO_DO_INTERVALO)
    ]

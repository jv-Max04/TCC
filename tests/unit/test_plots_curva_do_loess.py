"""A figura da curva do LOESS com o envelope sem um QC.

**O oráculo da curva é o núcleo, e não um literal.** A carga dos testes sai de
`curve_sensitivity` sobre uma matriz sintética, convertida para a forma do fio por `_carga` — a
mesma conversão que a rota faz, com `None` no lugar de `nan` —, e o que se confere é que o artista
desenha os vetores do núcleo, em bytes. A conversão da própria rota tem o seu teste em
`tests/test_api_curva_do_loess.py`, que fecha o caminho inteiro sobre um modo real.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from refino.correction.loess import CurveSensitivity, curve_sensitivity
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.plots.base import MatrizVaziaError
from refino.plots.catalogo import CONSTRUTORES, Vinculo
from refino.plots.curva_do_loess import (
    MARCA_SEM_CURVA,
    ROTULO_DA_CURVA,
    ROTULO_DA_CURVA_DE_OUTRO_CORRETOR,
    ROTULO_DA_POSICAO_PREENCHIDA,
    ROTULO_DA_POSICAO_SORTEADA,
    ROTULO_DA_PREDICAO,
    ROTULO_DO_ENVELOPE,
    ROTULO_DO_ENVELOPE_NAO_AJUSTADO,
    ROTULO_DO_LIMITE_INFERIOR,
    ROTULO_DO_LIMITE_SUPERIOR,
    ROTULO_DO_QC,
    ROTULO_DO_QC_FORA_DO_AJUSTE,
    CurvaDesalinhadaError,
    CurvaDoLoess,
    curva_do_fio,
    figura_da_curva_do_loess,
    recorte_da_curva_do_loess,
)
from refino.reporting.model import ITENS_OBRIGATORIOS
from tests.arnes_de_figura import confere_vista, series

N = 68
QCS = (0, 11, 22, 33, 44, 55, 67)


def _matriz(
    *,
    qcs: tuple[int, ...] = QCS,
    lotes: tuple[str, ...] | None = None,
    coluna: np.ndarray | None = None,
) -> RefinoMatrix:
    """Deriva suave conhecida com ruído pequeno, numa feature só — ou a `coluna` dada."""
    rng = np.random.default_rng(3)
    tempo = np.arange(1, N + 1, dtype=np.float64)
    valores = 2000.0 * (1.0 + 0.004 * tempo + 0.05 * np.sin(tempo / 9.0))
    valores = valores * rng.normal(1.0, 0.02, size=N)
    if coluna is not None:
        valores = np.asarray(coluna, dtype=np.float64)
    return RefinoMatrix(
        values=valores[:, None],
        feature_names=("F1",),
        samples=tuple(
            SampleMetadata(
                name=f"S{i + 1}",
                injection_order=i + 1,
                sample_type="qc" if i in qcs else "sample",
                group="QC" if i in qcs else "Estudo",
                batch=lotes[i] if lotes else "A",
            )
            for i in range(N)
        ),
        mode="tiny",
    )


def _lista(vetor: np.ndarray) -> list[float | None]:
    return [float(x) if np.isfinite(x) else None for x in vetor]


def _carga(s: CurveSensitivity, m: RefinoMatrix) -> dict[str, Any]:
    """`CurveSensitivity` na forma do fio — o que a rota devolve, com `None` no lugar de `nan`.

    Mais o `corretor`, que não vem da rota e é o da correção de quem pede.
    """
    predicao = np.full(m.n_samples, np.nan)
    motivos: list[str | None] = [None] * m.n_samples
    for linha, curva in zip(s.held_out_rows, s.held_out, strict=True):
        if curva.cause is None:
            predicao[linha] = curva.curve[linha]
        else:
            motivos[linha] = curva.reason
    return {
        "feature": s.fitted.feature,
        "ordens_de_injecao": [float(o) for o in m.injection_orders],
        "lotes": [str(b) for b in m.batches],
        "qc": [bool(q) for q in m.qc_mask],
        "valores": _lista(np.asarray(m.values[:, 0], dtype=np.float64)),
        "curva": _lista(s.fitted.curve),
        "curva_preenchida": [bool(x) for x in s.fitted.filled],
        "curva_sorteada": [bool(x) for x in s.fitted.drawn],
        "qc_no_ajuste": [bool(x) for x in s.fitted.used_qc],
        "motivo_sem_curva": s.fitted.reason or None,
        "predicao_sem_o_qc": _lista(predicao),
        "motivo_sem_predicao": motivos,
        "envelope_inferior": _lista(s.lower),
        "envelope_superior": _lista(s.upper),
        "envelope_nao_ajustado": [bool(x) for x in s.not_fitted],
        "n_curvas_do_envelope": s.n_curves,
        "motivo_sem_envelope": s.degenerate,
        "semente": 11,
        "spans_por_lote": list(s.fitted.spans),
        # O corretor não vem da rota: é o da correção de quem pede, que a janela carimba no
        # disparo. Aqui é o LOESS, e o teste do rótulo troca-o.
        "corretor": "loess",
    }


def _figura(curva: CurvaDoLoess) -> Any:
    return figura_da_curva_do_loess(
        curva=curva, recorte=recorte_da_curva_do_loess(curva=curva, n_features_totais=6035)
    )


def _serie(figura: Any, rotulo: str) -> Any:
    achadas = [s for s in series(figura.axes[0]) if s.rotulo == rotulo]
    assert len(achadas) == 1, [s.rotulo for s in series(figura.axes[0])]
    return achadas[0]


@pytest.fixture(scope="module")
def sensibilidade() -> tuple[CurveSensitivity, RefinoMatrix]:
    matriz = _matriz()
    return curve_sensitivity(matriz, "F1", seed=11), matriz


# ---------------------------------------------------------------------------
# O que se desenha é o que o núcleo mediu
# ---------------------------------------------------------------------------


def test_a_curva_desenhada_e_a_do_nucleo_bit_a_bit(
    sensibilidade: tuple[CurveSensitivity, RefinoMatrix],
) -> None:
    """A série da curva tem as ordens de injeção no x e a curva do núcleo no y, em bytes."""
    s, m = sensibilidade
    assert s.fitted.cause is None, s.fitted.reason
    desenhada = _serie(_figura(curva_do_fio(**_carga(s, m))), ROTULO_DA_CURVA)
    assert np.asarray(desenhada.y, dtype=np.float64).tobytes() == s.fitted.curve.tobytes()
    assert np.asarray(desenhada.x, dtype=np.float64).tobytes() == (
        m.injection_orders.astype(np.float64).tobytes()
    )


def test_a_comparacao_da_curva_acusa_outra_curva_no_lugar(
    sensibilidade: tuple[CurveSensitivity, RefinoMatrix],
) -> None:
    """A planta: a carga com o mínimo do envelope no lugar da curva reprova a igualdade.

    Âncora de separação antes de plantar: o mínimo do envelope e a curva DIFEREM neste dado.
    """
    s, m = sensibilidade
    assert s.lower.tobytes() != s.fitted.curve.tobytes(), "a planta não separaria nada"
    carga = _carga(s, m)
    carga["curva"] = _lista(s.lower)
    desenhada = _serie(_figura(curva_do_fio(**carga)), ROTULO_DA_CURVA)
    assert np.asarray(desenhada.y, dtype=np.float64).tobytes() != s.fitted.curve.tobytes()


def test_as_predicoes_desenhadas_sao_as_curvas_sem_cada_qc_na_posicao_dele(
    sensibilidade: tuple[CurveSensitivity, RefinoMatrix],
) -> None:
    s, m = sensibilidade
    esperado_x = [float(m.injection_orders[r]) for r in s.held_out_rows]
    esperado_y = [c.curve[r] for r, c in zip(s.held_out_rows, s.held_out, strict=True)]
    assert len(esperado_y) == 7, "âncora: sete QCs, sete predições"
    desenhada = _serie(_figura(curva_do_fio(**_carga(s, m))), ROTULO_DA_PREDICAO)
    assert desenhada.x.tolist() == esperado_x
    assert np.asarray(desenhada.y).tobytes() == np.asarray(esperado_y).tobytes()


def test_o_envelope_desenhado_e_o_minimo_e_o_maximo_e_o_trecho_nao_ajustado_vai_hachurado(
    sensibilidade: tuple[CurveSensitivity, RefinoMatrix],
) -> None:
    """Os dois limites são linhas com o dado do núcleo; a faixa e a hachura vão por cima delas.

    A hachura cobre exactamente os trechos em que alguma curva sem um QC não foi ajustada, com um
    vizinho a mais de cada lado — o que se confere pelos `x` dos polígonos hachurados.
    """
    s, m = sensibilidade
    figura = _figura(curva_do_fio(**_carga(s, m)))
    eixo = figura.axes[0]
    inferior = _serie(figura, ROTULO_DO_LIMITE_INFERIOR)
    superior = _serie(figura, ROTULO_DO_LIMITE_SUPERIOR)
    assert np.asarray(inferior.y, dtype=np.float64).tobytes() == s.lower.tobytes()
    assert np.asarray(superior.y, dtype=np.float64).tobytes() == s.upper.tobytes()

    cheias = [p for p in eixo.patches if p.get_label() == ROTULO_DO_ENVELOPE]
    hachuradas = [p for p in eixo.patches if p.get_label() == ROTULO_DO_ENVELOPE_NAO_AJUSTADO]
    assert len(cheias) == 1 and not cheias[0].get_hatch()
    assert hachuradas and all(p.get_hatch() == "///" for p in hachuradas)
    xs_hachurados = {float(x) for p in hachuradas for x in p.get_xy()[:, 0]}
    ordens = m.injection_orders.astype(float)
    marcadas = s.not_fitted
    vizinhas = marcadas | np.roll(marcadas, 1) | np.roll(marcadas, -1)
    assert set(ordens[marcadas].tolist()) <= xs_hachurados <= set(ordens[vizinhas].tolist())

    legenda = [t.get_text() for t in eixo.get_legend().get_texts()]
    assert ROTULO_DO_ENVELOPE in legenda and ROTULO_DO_ENVELOPE_NAO_AJUSTADO in legenda
    assert not any(
        r.startswith((ROTULO_DO_LIMITE_INFERIOR, ROTULO_DO_LIMITE_SUPERIOR)) for r in legenda
    )


def test_o_rotulo_diz_sensibilidade_e_nao_intervalo(
    sensibilidade: tuple[CurveSensitivity, RefinoMatrix],
) -> None:
    """O que a figura existe para NÃO sugerir é uma banda de confiança — e o texto diz isso."""
    s, m = sensibilidade
    figura = _figura(curva_do_fio(**_carga(s, m)))
    legenda = [t.get_text() for t in figura.axes[0].get_legend().get_texts()]
    assert ROTULO_DO_ENVELOPE in legenda
    assert "sensibilidade" in ROTULO_DO_ENVELOPE and "não intervalo" in ROTULO_DO_ENVELOPE
    rodape = " ".join(" ".join(t.get_text() for t in figura.texts).split())
    assert "não intervalo de confiança" in rodape
    # O `span` eleito vai no rodapé, porque é ele que diz se a curva passa pelos QCs.
    assert len(s.fitted.spans) == 1, s.fitted.spans
    assert f"span eleito {s.fitted.spans[0]:.2f}".replace(".", ",") in rodape, rodape
    titulo = figura.axes[0].get_title()
    assert "7 QCs" in titulo and "7 com predição" in titulo and "envelope de 7" in titulo


@pytest.mark.parametrize("corretor", ["serrf", "none", None])
def test_com_outro_corretor_a_curva_nao_se_diz_a_que_corrigiu_o_sinal(
    sensibilidade: tuple[CurveSensitivity, RefinoMatrix], corretor: str | None
) -> None:
    """A curva do LOESS só é «a que corrigiu o sinal» quando o corretor do pedido é o LOESS.

    **O par causal tem a mesma curva dos dois lados**, e é o que faz o rótulo ser a única coisa
    que separa: com o SERRF, sem correção, ou sem corretor decidido, a série é a mesma, em bytes,
    e o rótulo e o rodapé dizem que é a curva que o LOESS usaria. O item 12 do relatório
    declara-se vazio nesse caso; a figura da janela é pedida antes de haver execução, e diz o que
    a curva é em vez de não a mostrar.
    """
    s, m = sensibilidade
    do_loess = _figura(curva_do_fio(**{**_carga(s, m), "corretor": "loess"}))
    de_outro = _figura(curva_do_fio(**{**_carga(s, m), "corretor": corretor}))

    pelo_loess = _serie(do_loess, ROTULO_DA_CURVA)
    pelo_outro = _serie(de_outro, ROTULO_DA_CURVA_DE_OUTRO_CORRETOR)
    assert np.asarray(pelo_outro.y).tobytes() == np.asarray(pelo_loess.y).tobytes()
    rotulos = {serie.rotulo for serie in series(de_outro.axes[0])}
    assert ROTULO_DA_CURVA not in rotulos, rotulos

    rodape_do_loess = " ".join(" ".join(t.get_text() for t in do_loess.texts).split())
    rodape = " ".join(" ".join(t.get_text() for t in de_outro.texts).split())
    assert "o LOESS usaria" not in rodape_do_loess, rodape_do_loess
    assert "o LOESS usaria" in rodape, rodape
    if corretor is not None:
        assert f"«{corretor}»" in rodape, rodape


def test_o_qc_fora_do_ajuste_vai_vazado_e_as_preenchidas_marcadas_onde_o_nucleo_diz() -> None:
    """O primeiro QC multiplicado por 20: o passo 1 tira-o, e a curva não alcança o começo.

    **O oráculo é o núcleo:** o QC vazado é o que `used_qc` deixou de fora, e os marcadores
    do passo 3 estão nas posições de `filled`, com a curva do núcleo no `y`. **Medido em 07/11/2026
    antes de escrever as âncoras:** com este dado, só o QC da injeção 1 fica fora, e as 11 posições
    antes do segundo QC são preenchidas pelo tempo mais próximo.
    """
    tempo = np.arange(1, N + 1, dtype=np.float64)
    coluna = 2000.0 * (1.0 + 0.004 * tempo + 0.05 * np.sin(tempo / 9.0))
    coluna = coluna * np.random.default_rng(3).normal(1.0, 0.02, size=N)
    coluna[QCS[0]] *= 20.0
    m = _matriz(coluna=coluna)
    s = curve_sensitivity(m, "F1", seed=11)
    fora = m.qc_mask & ~s.fitted.used_qc
    # As âncoras ANTES do desenho: as duas camadas têm população neste dado.
    assert s.fitted.cause is None, s.fitted.reason
    assert fora.any() and (m.qc_mask & s.fitted.used_qc).any(), np.flatnonzero(fora)
    assert s.fitted.filled.any(), "a curva alcançou o começo: o passo 3 não correu"

    figura = _figura(curva_do_fio(**_carga(s, m)))
    ordens = m.injection_orders.astype(np.float64)
    vazado = _serie(figura, ROTULO_DO_QC_FORA_DO_AJUSTE)
    assert vazado.x.tolist() == ordens[fora].tolist()
    assert np.asarray(vazado.y, dtype=np.float64).tobytes() == coluna[fora].tobytes()
    cheio = _serie(figura, ROTULO_DO_QC)
    assert cheio.x.tolist() == ordens[m.qc_mask & s.fitted.used_qc].tolist()
    preenchida = _serie(figura, ROTULO_DA_POSICAO_PREENCHIDA)
    assert preenchida.x.tolist() == ordens[s.fitted.filled].tolist()
    assert np.asarray(preenchida.y, dtype=np.float64).tobytes() == (
        s.fitted.curve[s.fitted.filled].tobytes()
    )


def test_as_posicoes_sorteadas_no_passo_4_vao_marcadas_onde_o_nucleo_diz() -> None:
    """Valores pequenos que caem ao longo da corrida: a curva fica negativa no fim, e sorteia-se.

    **Medido em 07/11/2026 antes de escrever as âncoras:** com este dado, as sete últimas
    posições têm curva negativa — menos de um quinto —, e são as sorteadas.
    """
    tempo = np.arange(1, N + 1, dtype=np.float64)
    coluna = np.maximum(0.0, 30.0 - 0.6 * tempo + np.random.default_rng(0).normal(0.0, 3.0, N))
    m = _matriz(coluna=coluna)
    s = curve_sensitivity(m, "F1", seed=11)
    assert s.fitted.cause is None, s.fitted.reason
    assert s.fitted.drawn.any(), "nenhuma posição sorteada: a camada não teria população"

    sorteada = _serie(_figura(curva_do_fio(**_carga(s, m))), ROTULO_DA_POSICAO_SORTEADA)
    ordens = m.injection_orders.astype(np.float64)
    assert sorteada.x.tolist() == ordens[s.fitted.drawn].tolist()
    assert np.asarray(sorteada.y, dtype=np.float64).tobytes() == (
        s.fitted.curve[s.fitted.drawn].tobytes()
    )


def test_nada_do_que_se_desenha_fica_fora_da_vista(
    sensibilidade: tuple[CurveSensitivity, RefinoMatrix],
) -> None:
    s, m = sensibilidade
    assert confere_vista(_figura(curva_do_fio(**_carga(s, m))).axes[0]) == []


# ---------------------------------------------------------------------------
# Sem curva, com o motivo; e os lotes
# ---------------------------------------------------------------------------


def test_lote_com_menos_de_tres_qcs_nao_tem_curva_e_a_figura_diz_por_que() -> None:
    """O mínimo de `loess.py` chega à figura por extenso: sem curva, sem envelope, e o porquê."""
    lotes = tuple("A" if i < 50 else "B" for i in range(N))
    m = _matriz(lotes=lotes)
    s = curve_sensitivity(m, "F1", seed=11)
    assert s.fitted.cause == "fit_failed" and s.degenerate is not None

    figura = _figura(curva_do_fio(**_carga(s, m)))
    rotulos = {serie.rotulo for serie in series(figura.axes[0])}
    assert not any(r.startswith(ROTULO_DA_CURVA) for r in rotulos), rotulos
    assert ROTULO_DO_QC in rotulos, "os QCs continuam desenhados: são a população da figura"
    # O porquê vai por extenso no rodapé; no eixo fica a marca curta que aponta para lá.
    no_eixo = [t.get_text() for t in figura.axes[0].texts]
    assert MARCA_SEM_CURVA in no_eixo, no_eixo
    rodape = " ".join(" ".join(t.get_text() for t in figura.texts).split())
    assert "Sem curva:" in rodape and "exige 3" in rodape, rodape
    assert "Sem curvas sem um QC:" in rodape and "nenhum span eleito" in rodape, rodape
    # **O motivo da régua termina em ponto, e a frase do rodapé também** — a figura renderizada e
    # lida mostrou «parecera medição..» no fim do rodapé. Um motivo que já traz o ponto final
    # não ganha outro.
    assert ".." not in rodape, rodape


def test_com_curva_o_eixo_nao_traz_a_marca_de_ausencia(
    sensibilidade: tuple[CurveSensitivity, RefinoMatrix],
) -> None:
    """O par da marca: com curva, nada no eixo diz que ela falta."""
    s, m = sensibilidade
    figura = _figura(curva_do_fio(**_carga(s, m)))
    assert MARCA_SEM_CURVA not in [t.get_text() for t in figura.axes[0].texts]


def test_dois_lotes_viram_duas_curvas_e_nenhuma_atravessa_a_fronteira() -> None:
    lotes = tuple("A" if i < 34 else "B" for i in range(N))
    m = _matriz(qcs=(0, 11, 22, 33, 34, 45, 56, 67), lotes=lotes)
    s = curve_sensitivity(m, "F1", seed=11)
    assert s.fitted.cause is None, s.fitted.reason
    figura = _figura(curva_do_fio(**_carga(s, m)))
    for lote, (inicio, fim) in (("A", (1, 34)), ("B", (35, 68))):
        serie = _serie(figura, f"{ROTULO_DA_CURVA} — lote {lote}")
        assert serie.x.min() == inicio and serie.x.max() == fim, (lote, serie.x)


# ---------------------------------------------------------------------------
# As recusas na fronteira
# ---------------------------------------------------------------------------


def test_vetor_desalinhado_e_recusa_propria(
    sensibilidade: tuple[CurveSensitivity, RefinoMatrix],
) -> None:
    s, m = sensibilidade
    carga = _carga(s, m)
    carga["curva"] = carga["curva"][:-1]
    with pytest.raises(CurvaDesalinhadaError, match="curva"):
        curva_do_fio(**carga)


def test_envelope_com_as_pontas_trocadas_e_recusado(
    sensibilidade: tuple[CurveSensitivity, RefinoMatrix],
) -> None:
    s, m = sensibilidade
    carga = _carga(s, m)
    carga["envelope_inferior"], carga["envelope_superior"] = (
        carga["envelope_superior"],
        carga["envelope_inferior"],
    )
    with pytest.raises(CurvaDesalinhadaError, match="mínimo acima do máximo"):
        curva_do_fio(**carga)


@pytest.mark.parametrize("campo", ["ordens_de_injecao", "qc"])
def test_sem_injecao_ou_sem_qc_e_populacao_vazia(
    sensibilidade: tuple[CurveSensitivity, RefinoMatrix], campo: str
) -> None:
    s, m = sensibilidade
    carga = _carga(s, m)
    carga[campo] = [] if campo == "ordens_de_injecao" else [False] * N
    with pytest.raises(MatrizVaziaError):
        curva_do_fio(**carga)


def test_o_construtor_bijeta_contra_o_item_12_do_relatorio() -> None:
    """O vínculo é o item que a emenda criou, e o título dele diz «não é intervalo»."""
    registo = [c for c in CONSTRUTORES if c.funcao == figura_da_curva_do_loess.__name__]
    assert len(registo) == 1 and registo[0].vinculo is Vinculo.RELATORIO, registo
    item = next(i for i in ITENS_OBRIGATORIOS if f"item-{i.numero}" == registo[0].chave)
    assert "LOESS" in item.titulo and "não é intervalo" in item.titulo, item.titulo

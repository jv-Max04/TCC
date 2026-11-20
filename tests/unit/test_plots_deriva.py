"""A trajetória do PC1 dos QCs, a primeira figura do projeto.

## O que este módulo prova, e o que ele deliberadamente NÃO prova

Quatro blocos:

| Bloco | O que prova |
|---|---|
| o construtor | os pontos desenhados são os que entraram, contra a **ordem de injeção** |
| as quatro plantas do arnês | as classes do arnês acusam sobre ESTA figura, e a limpa passa |
| as recusas tipadas | ordem ausente, desalinhada, não finita, decrescente; lotes desalinhados |
| o corpus real | o `neg_hilic`, com os sete QCs e a partição que o diagnóstico propõe |

**O que NÃO se afirma, e vai declarado porque a leitura natural sugeriria o contrário:**

1. **Nenhuma asserção sobre a DIREÇÃO da trajetória.** O laboratório espera que os QCs subam sem
   correção, e depois de SERRF ou LOESS que fiquem agrupados (`[25:22]`). Isso é o que a figura
   deixa ver — não é propriedade do desenho, e uma asserção sobre ela seria
   observação promovida a invariante.
2. **Nenhuma asserção sobre o SINAL do PC1.** Ele é arbitrário e a orientação canônica é obrigação
   de quem produz o componente; este construtor não a aplica, e afirmar o sinal aqui
   congelaria um número que `metrics.structure` pode legitimamente inverter.
3. **A guarda da ordem de injeção NÃO apanha um chamador que escreva `arange(n)`**, porque
   `0, 1, 2, …` é uma ordem de injeção legítima. Quem o apanha é
   `test_o_eixo_x_e_a_ORDEM_DE_INJECAO_e_nao_a_posicao_da_coluna`, sobre dado em que as duas
   **diferem** — que é a âncora de separação, e não a guarda.
4. **As contagens dos arquivos reais são observação, nunca invariante.** Sete QCs, as injeções 1,
   12, 23, 34, 45, 56 e 68 e o lote único `A` são o que os quatro `measurements` deste laboratório
   têm hoje. O teste do corpus real nomeia o modo e afirma
   sobre ele.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pytest

from refino.api.rotas import LIMIARES_INICIAIS
from refino.diagnosis.policy import DiagnosisPolicy
from refino.diagnosis.profile import PRETRANSFORM, diagnose
from refino.io.readers import load_matrix
from refino.metrics.structure import structural_metrics
from refino.plots.base import MatrizVaziaError, nova_figura
from refino.plots.deriva import (
    AMOSTRAGEM,
    MATRIZ,
    METODO,
    MINIMO_PARA_LINHA,
    ROTULO_DE_X,
    ROTULO_DE_Y,
    TITULO,
    LotesDesalinhadosError,
    SemOrdemDeInjecaoError,
    figura_da_deriva_do_qc,
    frase_da_incerteza,
    recorte_da_deriva,
    rotulo_do_lote,
)
from refino.plots.recorte import (
    ENTRELINHA,
    MARGEM,
    TAMANHO_DA_FONTE,
    Recorte,
    colunas_da_figura,
    quebra,
)
from tests.arnes_de_figura import (
    confere_numeros,
    confere_rotulos,
    confere_series,
    confere_vista,
    planta_corte_de_eixo,
    planta_eixos_trocados,
    planta_serie_removida,
    planta_sinal_invertido,
    series,
)
from tests.conftest import measurements_path, sample_list_path

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# O material: uma corrida sintética em que a ordem de injeção DIFERE da posição
# ---------------------------------------------------------------------------
#
# **As sete posições não são consecutivas, e essa é a âncora de separação deste módulo.** Se os QCs
# estivessem nas injeções 0 a 6, desenhar contra a posição da coluna e desenhar contra a ordem de
# injeção dariam a MESMA figura, e o par causal do eixo x não separaria nada. Os valores imitam a
# forma dos arquivos reais — QCs espaçados ao longo da corrida — sem copiar contagem nenhuma deles.
ORDENS: Final = (1, 12, 23, 34, 45, 56, 68)

# **Deriva monotônica, e a inclinação é o que o caminho feliz reproduz.** Monotônica de propósito:
# é a forma que o laboratório descreve, e é sobre ela que *"a trajetória a reproduz"* tem sentido.
# **Nenhum valor é zero** — âncora de `planta_sinal_invertido` — e a série **não é constante** —
# âncora de `planta_corte_de_eixo`.
SCORES: Final = (-12.0, -8.5, -4.0, 1.5, 6.0, 11.5, 18.0)

# Denominadores do rodapé. **Literais aqui de propósito:** são a fonte independente da classe (d),
# e o construtor recebe-os por um caminho que não passa por estes nomes.
USADAS: Final = 40
TOTAIS: Final = 100

LOTE_UNICO: Final = ("A",) * len(ORDENS)

# Dois lotes, com a fronteira depois do quinto QC. **Cinco e dois, e não três e quatro**, porque um
# lote com dois QCs é o caso que a legenda por lote existe para tornar visível.
DOIS_LOTES: Final = ("0", "0", "0", "0", "0", "1", "1")


def _recorte() -> Recorte:
    return recorte_da_deriva(n_features_usadas=USADAS, n_features_totais=TOTAIS)


def _figura(
    *,
    scores: tuple[float, ...] = SCORES,
    ordens: tuple[int, ...] = ORDENS,
    lotes: tuple[str, ...] = LOTE_UNICO,
) -> Figure:
    return figura_da_deriva_do_qc(
        scores=scores, ordens_de_injecao=ordens, lotes=lotes, recorte=_recorte()
    )


def _numeros_esperados(*extras: str) -> list[str]:
    """Todo número que a figura declara, derivado da ENTRADA e não do objeto que o construtor usou.

    `"1"` aparece por dois caminhos independentes — o `PC1` do título e o `PC1` do método —, e é
    isso que o torna inevitável na lista. Os denominadores vêm dos literais deste módulo.
    """
    return ["1", str(USADAS), str(TOTAIS), *extras]


# ---------------------------------------------------------------------------
# 1. O caminho feliz — deriva monotônica, e a trajetória a reproduz
# ---------------------------------------------------------------------------


def test_a_deriva_monotonica_e_reproduzida_ponto_a_ponto() -> None:
    """As quatro classes do arnês sobre a figura limpa. **É a outra ponta dos pares causais.**

    O dado esperado é construído aqui, dos literais do módulo, e não lido de volta do construtor.
    """
    figura = _figura()
    eixo = figura.axes[0]

    rotulo = rotulo_do_lote("A", len(SCORES))
    assert confere_series(eixo, {rotulo: (ORDENS, SCORES)}) == []
    assert (
        confere_rotulos(
            eixo,
            x=ROTULO_DE_X,
            y=ROTULO_DE_Y,
            titulo=f"{TITULO}\n{len(SCORES)} QCs, sem ajuste de tendência",
            legenda=[rotulo],
        )
        == []
    )
    assert confere_vista(eixo) == []
    assert confere_numeros(figura, _numeros_esperados(str(len(SCORES)))) == []


def test_a_monotonicidade_do_dado_chega_ao_artista() -> None:
    """A trajetória **reproduz** a deriva: o que foi desenhado sobe onde a entrada sobe.

    Não é afirmação sobre o domínio — é afirmação sobre o construtor não reordenar nem reamostrar.
    Nenhuma asserção aqui diz que QCs sobem; diz que ESTE dado, que sobe, chega assim ao artista.
    """
    # A âncora vem primeiro: a ENTRADA sobe. Sem ela, uma trajetória que chegasse constante ao
    # artista faria as duas asserções abaixo passarem por vacuidade.
    assert bool(np.all(np.diff(SCORES) > 0.0)), SCORES
    desenhada = series(_figura().axes[0])[0]
    assert bool(np.all(np.diff(desenhada.y) > 0.0)), desenhada.y
    assert bool(np.all(np.diff(desenhada.x) > 0.0)), desenhada.x


# ---------------------------------------------------------------------------
# 2. O eixo x é a ORDEM DE INJEÇÃO — o par causal que a guarda não alcança
# ---------------------------------------------------------------------------


def test_o_eixo_x_e_a_ORDEM_DE_INJECAO_e_nao_a_posicao_da_coluna() -> None:  # noqa: N802
    """**A planta é o defeito: desenhar contra a posição da coluna.**

    As colunas de `measurements_*.csv` estão agrupadas por classe (com os QCs primeiro em três dos
    quatro arquivos), então a
    posição de um QC é `0, 1, 2, …` — uma figura bem formada sobre o eixo errado, que não levanta.

    **A âncora de separação vem PRIMEIRO:** as duas quantidades têm de diferir sobre este dado,
    senão o
    par não separa nada. Ela é o motivo de `ORDENS` não ser `range(7)`.
    """
    posicoes = tuple(float(i) for i in range(len(ORDENS)))
    assert posicoes != tuple(float(o) for o in ORDENS), (
        "âncora: a ordem de injeção e a posição da coluna coincidem neste dado, e o par causal "
        "abaixo passaria com o defeito plantado"
    )

    rotulo = rotulo_do_lote("A", len(SCORES))
    eixo = _figura().axes[0]

    # A ponta boa: o que está desenhado é a ordem de injeção.
    assert confere_series(eixo, {rotulo: (ORDENS, SCORES)}) == []
    # A ponta da planta: contra a posição da coluna, o arnês acusa.
    assert confere_series(eixo, {rotulo: (posicoes, SCORES)}) != []


def test_a_ordem_de_injecao_nao_tem_valor_padrao() -> None:
    """Não há caminho neste construtor que substitua a ordem ausente. **Recusa, nunca recuo.**"""
    with pytest.raises(SemOrdemDeInjecaoError, match="agrupada por"):
        figura_da_deriva_do_qc(
            scores=SCORES, ordens_de_injecao=(), lotes=LOTE_UNICO, recorte=_recorte()
        )


@pytest.mark.parametrize(
    ("ordens", "trecho"),
    [
        ((1, 12, 23), "de outra população"),
        ((1, 12, 23, 34, 45, 56, float("nan")), "não são"),
        ((1, 12, 23, 34, 68, 56, 45), "não decrescentes"),
    ],
)
def test_ordem_de_injecao_invalida_produz_recusa_tipada(
    ordens: tuple[float, ...], trecho: str
) -> None:
    """Comprimento, finitude e monotonia. **A planta é o argumento — negativo em processo.**

    A monotonia é **não decrescente** e não estritamente crescente, e isso decorre do desenho:
    `RefinoMatrix` ordena as linhas por `injection_order` na construção, e a ingestão desempata
    ordens repetidas por nome em vez de as proibir. Exigir estritamente crescente seria afirmar
    como invariante o que é só o caso destes arquivos.
    """
    with pytest.raises(SemOrdemDeInjecaoError, match=trecho):
        figura_da_deriva_do_qc(
            scores=SCORES, ordens_de_injecao=ordens, lotes=LOTE_UNICO, recorte=_recorte()
        )


def test_ordem_de_injecao_com_EMPATE_e_aceite() -> None:  # noqa: N802
    """A outra ponta da monotonia: empate passa, e o teste acima não acusa por acaso."""
    com_empate = (1, 12, 12, 34, 45, 56, 68)
    eixo = _figura(ordens=com_empate).axes[0]
    assert confere_series(eixo, {rotulo_do_lote("A", len(SCORES)): (com_empate, SCORES)}) == []


# ---------------------------------------------------------------------------
# 3. Um QC só — nenhuma linha, e a figura diz que um ponto não é trajetória
# ---------------------------------------------------------------------------


def test_um_QC_so_nao_desenha_linha_e_a_figura_DIZ_o_porque() -> None:  # noqa: N802
    """A borda, e ela tem as duas metades: nada ligado, e a frase que explica.

    **Sem linha é a metade fácil; a frase é a que importa.** Um marcador solto sem texto lê-se como
    trajetória de um ponto, que é a leitura que a figura existe para não produzir.
    """
    figura = _figura(scores=SCORES[:1], ordens=ORDENS[:1], lotes=LOTE_UNICO[:1])
    eixo = figura.axes[0]

    assert len(eixo.lines) == 1, eixo.lines
    assert eixo.lines[0].get_linestyle() in {"None", "none", ""}, eixo.lines[0].get_linestyle()
    assert eixo.lines[0].get_marker() == "o", "sem marcador o único ponto seria invisível"

    assert "um ponto não é trajetória" in eixo.get_title()
    assert confere_numeros(figura, _numeros_esperados()) == []


def test_com_dois_QCs_a_linha_VOLTA() -> None:  # noqa: N802
    """A outra direção do mesmo mecanismo. Sem ela, um construtor que nunca ligasse nada passaria.

    `MINIMO_PARA_LINHA` é 2 porque dois pontos fazem um segmento — origem estrutural, não
    calibração.
    """
    figura = _figura(
        scores=SCORES[:MINIMO_PARA_LINHA],
        ordens=ORDENS[:MINIMO_PARA_LINHA],
        lotes=LOTE_UNICO[:MINIMO_PARA_LINHA],
    )
    eixo = figura.axes[0]
    assert eixo.lines[0].get_linestyle() not in {"None", "none", ""}
    assert "sem ajuste de tendência" in eixo.get_title()


@pytest.mark.parametrize(
    ("n_desenhados", "n_sem_score", "trecho"),
    [
        (1, 0, "1 QC — um ponto não é trajetória"),
        (7, 0, "7 QCs, sem ajuste de tendência"),
        (5, 2, "5 QCs, sem ajuste de tendência; 2 sem score finito"),
    ],
)
def test_a_frase_da_incerteza_tem_os_TRES_desfechos(  # noqa: N802
    n_desenhados: int, n_sem_score: int, trecho: str
) -> None:
    """**Três desfechos, e a verificação pede os três** — «dois» é o piso de um portão, não o
    número.

    O terceiro existe porque um QC sem score finito não é desenhado, e o `matplotlib` mostraria a
    ausência como um salto do traço: indistinguível de uma deriva abrupta.
    """
    assert frase_da_incerteza(n_desenhados, n_sem_score) == trecho


def test_QC_sem_score_finito_e_CONTADO_na_figura_e_nao_apagado() -> None:  # noqa: N802
    """O par causal do terceiro desfecho, sobre a figura e não sobre a função de texto."""
    com_buraco = (*SCORES[:3], float("nan"), *SCORES[4:])
    figura = _figura(scores=com_buraco)
    eixo = figura.axes[0]

    assert "6 QCs, sem ajuste de tendência; 1 sem score finito" in eixo.get_title()
    desenhada = series(eixo)[0]
    assert desenhada.y.size == len(SCORES) - 1, "o ponto não finito foi desenhado"
    assert confere_numeros(figura, _numeros_esperados("6")) == []

    # A outra ponta: sem buraco, a figura não fala de score ausente.
    assert "sem score finito" not in _figura().axes[0].get_title()


def test_scores_todos_nao_finitos_recusam_como_POPULACAO_VAZIA() -> None:  # noqa: N802
    """Eixo montado e nada dentro lê-se como figura quebrada — é o motivo de `MatrizVaziaError`."""
    with pytest.raises(MatrizVaziaError, match="nenhum com score finito"):
        _figura(scores=(float("nan"),) * len(SCORES))


def test_scores_vazios_recusam_antes_de_qualquer_desenho() -> None:
    with pytest.raises(MatrizVaziaError, match="`scores` está vazio"):
        _figura(scores=(), ordens=(), lotes=())


# ---------------------------------------------------------------------------
# 4. Mais de um lote — a figura separa, e o traço não atravessa
# ---------------------------------------------------------------------------


def test_dois_lotes_viram_duas_SERIES_e_nenhuma_linha_atravessa() -> None:  # noqa: N802
    """O número de lotes pode deixar de ser um; a figura tem de o mostrar.

    **A metade que se esquece é a segunda:** duas séries com um traço a ligar o último QC do lote 0
    ao primeiro do lote 1 desenhariam exactamente a continuidade que a separação nega. O arnês
    apanha-a porque o `x` de cada série é conferido ponto a ponto contra a partição esperada.
    """
    figura = _figura(lotes=DOIS_LOTES)
    eixo = figura.axes[0]

    do_zero = [i for i, lote in enumerate(DOIS_LOTES) if lote == "0"]
    do_um = [i for i, lote in enumerate(DOIS_LOTES) if lote == "1"]
    esperadas = {
        rotulo_do_lote("0", len(do_zero)): (
            [ORDENS[i] for i in do_zero],
            [SCORES[i] for i in do_zero],
        ),
        rotulo_do_lote("1", len(do_um)): (
            [ORDENS[i] for i in do_um],
            [SCORES[i] for i in do_um],
        ),
    }
    assert confere_series(eixo, esperadas) == []
    assert confere_vista(eixo) == []
    assert confere_numeros(figura, _numeros_esperados("7", "5", "2", "0")) == []

    # Nenhum artista contém pontos dos dois lotes: é isto que «o traço não atravessa» significa.
    for serie in series(eixo):
        lotes_tocados = {DOIS_LOTES[ORDENS.index(int(x))] for x in serie.x}
        assert len(lotes_tocados) == 1, (serie.rotulo, lotes_tocados)


def test_a_ordem_dos_lotes_e_a_de_APARICAO_e_nao_a_alfabetica() -> None:  # noqa: N802
    """`"10"` antes de `"2"` é o que `sorted` faria, e poria a corrida fora de ordem na legenda."""
    lotes = ("2", "2", "2", "10", "10", "10", "10")
    legenda = _figura(lotes=lotes).axes[0].get_legend()
    assert legenda is not None
    assert [t.get_text() for t in legenda.get_texts()] == [
        rotulo_do_lote("2", 3),
        rotulo_do_lote("10", 4),
    ]


def test_a_contagem_por_LOTE_entra_na_legenda() -> None:  # noqa: N802
    """Escopo: o total de 7 esconde que um dos lotes tem dois QCs."""
    assert rotulo_do_lote("A", 1) == "lote A — 1 QC"
    assert rotulo_do_lote("A", 7) == "lote A — 7 QCs"
    legenda = _figura(lotes=DOIS_LOTES).axes[0].get_legend()
    assert legenda is not None
    assert [t.get_text() for t in legenda.get_texts()] == ["lote 0 — 5 QCs", "lote 1 — 2 QCs"]


def test_lote_com_um_QC_SO_e_caso_normal_e_nao_erro() -> None:  # noqa: N802
    """Um QC por lote é o que uma sugestão de fronteira pode produzir, e não é recusa."""
    lotes = ("0", "0", "0", "0", "0", "0", "1")
    eixo = _figura(lotes=lotes).axes[0]
    solitario = next(s for s in series(eixo) if s.rotulo == rotulo_do_lote("1", 1))
    assert solitario.y.size == 1


def test_lote_com_TODOS_os_scores_nao_finitos_nao_vira_serie_vazia() -> None:  # noqa: N802
    """Uma entrada de legenda sem ponto nenhum promete um traço que não está lá.

    **E a outra ponta está no teste acima**: com pontos finitos, o lote vira série. Sem as duas, um
    construtor que nunca desenhasse o segundo lote passaria aqui.
    """
    scores = (*SCORES[:5], float("nan"), float("nan"))
    figura = _figura(scores=scores, lotes=DOIS_LOTES)
    desenhadas = series(figura.axes[0])
    assert [s.rotulo for s in desenhadas] == [rotulo_do_lote("0", 5)], desenhadas
    assert "5 QCs, sem ajuste de tendência; 2 sem score finito" in figura.axes[0].get_title()


def test_um_ESCALAR_no_lugar_do_vetor_produz_recusa_tipada() -> None:  # noqa: N802
    """`null` no fio chega como `nan` 0-dimensional, e um escalar não é uma trajetória."""
    with pytest.raises(SemOrdemDeInjecaoError, match="dimensão"):
        figura_da_deriva_do_qc(
            scores=np.asarray(1.0), ordens_de_injecao=ORDENS, lotes=LOTE_UNICO, recorte=_recorte()
        )
    with pytest.raises(SemOrdemDeInjecaoError, match="dimensão"):
        figura_da_deriva_do_qc(
            scores=SCORES,
            ordens_de_injecao=np.asarray(1.0),
            lotes=LOTE_UNICO,
            recorte=_recorte(),
        )


@pytest.mark.parametrize("lotes", [(), ("A",) * 3, ("A",) * 9])
def test_lotes_desalinhados_produzem_RECUSA_PROPRIA(lotes: tuple[str, ...]) -> None:  # noqa: N802
    """**Excepção diferente da ordem de injeção, porque o caso é outro.**

    «Não sei quando cada QC foi injetado» e «não sei a que lote pertence» têm conserto diferente —
    o `Sample_List` e a sugestão do diagnóstico. Um valor de erro para os dois tornaria as duas
    indistinguíveis em contagem.
    """
    with pytest.raises(LotesDesalinhadosError, match="rótulo"):
        _figura(lotes=lotes)


# ---------------------------------------------------------------------------
# 5. As quatro plantas do arnês, sobre ESTA figura
# ---------------------------------------------------------------------------


def _esperadas_de_dois_lotes() -> dict[str, tuple[Any, Any]]:
    do_zero = [i for i, lote in enumerate(DOIS_LOTES) if lote == "0"]
    do_um = [i for i, lote in enumerate(DOIS_LOTES) if lote == "1"]
    return {
        rotulo_do_lote("0", len(do_zero)): (
            [ORDENS[i] for i in do_zero],
            [SCORES[i] for i in do_zero],
        ),
        rotulo_do_lote("1", len(do_um)): ([ORDENS[i] for i in do_um], [SCORES[i] for i in do_um]),
    }


def test_planta_serie_removida_reprova() -> None:
    """**Duas séries são necessárias**, e é por isso que a planta corre sobre a figura de dois
    lotes: apagar a única deixaria eixos vazios, e a reprovação deixaria de ser atribuível."""
    eixo = _figura(lotes=DOIS_LOTES).axes[0]
    apagado = planta_serie_removida(eixo)
    achados = confere_series(eixo, _esperadas_de_dois_lotes())
    assert any(apagado in a for a in achados), achados


def test_planta_eixos_trocados_reprova() -> None:
    eixo = _figura(lotes=DOIS_LOTES).axes[0]
    mutilada = planta_eixos_trocados(eixo)
    achados = confere_series(eixo, _esperadas_de_dois_lotes())
    assert any(mutilada in a for a in achados), achados


def test_planta_sinal_invertido_reprova() -> None:
    eixo = _figura(lotes=DOIS_LOTES).axes[0]
    mutilada = planta_sinal_invertido(eixo)
    achados = confere_series(eixo, _esperadas_de_dois_lotes())
    assert any("eixo y" in a and mutilada in a for a in achados), achados


def test_planta_corte_de_eixo_reprova_e_o_dado_continua_CERTO() -> None:  # noqa: N802
    """**A planta que as três de cima não alcançam.** Dado nenhum muda e a tela fica em branco."""
    eixo = _figura(lotes=DOIS_LOTES).axes[0]
    planta_corte_de_eixo(eixo)
    assert confere_series(eixo, _esperadas_de_dois_lotes()) == [], (
        "âncora: a planta do corte não pode mexer em dado — se mexeu, não é esta a classe testada"
    )
    achados = confere_vista(eixo)
    assert any("A figura está em branco" in a for a in achados), achados


def test_planta_do_numero_declarado_reprova() -> None:
    """Classe (d): o rodapé é a única coisa da figura que outra pessoa pode desmentir."""
    figura = _figura()
    assert confere_numeros(figura, _numeros_esperados(str(len(SCORES)))) == []
    figura.texts[0].set_text("13 de 13 features")
    achados = confere_numeros(figura, _numeros_esperados(str(len(SCORES))))
    assert any("13" in a for a in achados), achados


# ---------------------------------------------------------------------------
# 6. O recorte que a figura declara
# ---------------------------------------------------------------------------


def test_o_rodape_declara_os_dois_denominadores_e_o_metodo() -> None:
    """R3 e R5b, pelo auxiliar partilhado. **Os dois números são um PAR**, e não um.

    A PCA descarta coluna de variância zero e coluna não finita, então o numerador não é derivável
    do total: servir só `n_features` faria o rodapé declarar a população errada, na única
    frase da imagem que se pode conferir contra fonte independente.
    """
    figura = _figura()
    rodape = figura.texts[0].get_text()
    assert f"{USADAS} de {TOTAIS} features" in rodape, rodape
    for campo in (AMOSTRAGEM, MATRIZ):
        assert campo.split(" — ")[0][:30] in rodape.replace("\n", " "), (campo, rodape)
    assert "sem teste estatístico" in rodape.replace("\n", " "), rodape


def test_o_metodo_declara_as_DUAS_ressalvas_que_a_figura_nao_pode_provar() -> None:  # noqa: N802
    """R5b, e o conteúdo é o que a imagem não consegue mostrar sozinha.

    O traço não é leitura entre dois QCs, e o sinal do PC1 é arbitrário. As duas são o que
    impede a figura de ser lida como mais forte do que é.
    """
    assert "NÃO é leitura entre eles" in METODO
    assert "sinal do componente é arbitrário" in METODO
    assert "nenhuma tendência é ajustada" in METODO
    assert "sinal arbitrário" in ROTULO_DE_Y


def test_nenhuma_serie_e_uma_TENDENCIA_ajustada() -> None:  # noqa: N802
    """**A prova de que a figura não sugere tendência é comportamental, não sintática.**

    As séries desenhadas são exactamente uma por lote, com exactamente os QCs daquele lote. Uma
    reta ajustada seria uma série a mais, com pontos que não estão na entrada, e o arnês acusa-a
    como *"série desenhada e não esperada"*.
    """
    eixo = _figura(lotes=DOIS_LOTES).axes[0]
    assert len(series(eixo)) == 2
    assert confere_series(eixo, _esperadas_de_dois_lotes()) == []


def _faixas(figura: Figure) -> tuple[float, float]:
    """`(topo do rodapé, fundo dos eixos COM as marcas e o rótulo)`, em fração de figura."""
    renderizador = figura.canvas.get_renderer()  # type: ignore[attr-defined]
    para_figura = figura.transFigure.inverted()
    topo_do_rodape = figura.texts[0].get_window_extent(renderizador).transformed(para_figura).y1
    fundo_dos_eixos = figura.axes[0].get_tightbbox(renderizador).transformed(para_figura).y0
    return float(topo_do_rodape), float(fundo_dos_eixos)


def test_o_rodape_NAO_e_impresso_por_baixo_do_rotulo_do_eixo_x() -> None:  # noqa: N802
    """**Defeito achado pela LEITURA da imagem montada, e nenhuma classe do arnês o via.**

    `recorte.aplica` sobe `subplotpars.bottom`, que é o fundo da **caixa** dos eixos; as marcas do
    eixo x e o `xlabel` são desenhados **por baixo** dela e caem dentro da faixa do rodapé. Na
    primeira renderização deste construtor, `ordem de injeção` saiu impresso por cima da primeira
    linha do rodapé — com as quatro classes do arnês verdes, porque dado, rótulo, limite e número
    declarado estavam todos certos.

    O par causal é a mesma figura **sem o afastamento**, montada aqui com os mesmos auxiliares:
    **é a planta, e ela separa.**

    ## A planta mudou de forma, e o motivo é que o conserto mudou de sítio

    Antes o afastamento era `_afasta_os_eixos_do_rodape`, privado deste módulo, e a planta era
    *"chamar `aplica` e não chamar o privado"*. Ele passou a viver **dentro** de `recorte.aplica`
    — para as quatro unidades de figura seguintes não terem de o copiar —, então `aplica`
    deixou de ser plantável por omissão: chamá-lo já afasta.

    A planta passou a **reconstruir o que `aplica` fazia antes**: quebrar a frase, reservar a faixa
    com `subplots_adjust` e escrever o `Text` no mesmo sítio, sem o `tight_layout`. É mais código,
    e é o preço de a propriedade ter subido de camada; o que ela continua a provar é o mesmo — que
    sem o afastamento os dois textos **de facto** colidem, e que a asserção de cima não passa por
    vacuidade.
    """
    topo_do_rodape, fundo_dos_eixos = _faixas(_figura())
    assert fundo_dos_eixos >= topo_do_rodape, (
        f"as marcas e o rótulo do eixo x começam em {fundo_dos_eixos:.3f} e o rodapé vai até "
        f"{topo_do_rodape:.3f}: os dois textos sobrepõem-se na imagem"
    )

    sem_afastamento = nova_figura()
    eixo = sem_afastamento.add_subplot(1, 1, 1)
    eixo.plot(ORDENS, SCORES, marker="o", linestyle="--", label=rotulo_do_lote("A", len(SCORES)))
    eixo.set_xlabel(ROTULO_DE_X)
    eixo.set_ylabel(ROTULO_DE_Y)
    _rodape_sem_afastamento(sem_afastamento, _recorte())
    topo_plantado, fundo_plantado = _faixas(sem_afastamento)
    assert fundo_plantado < topo_plantado, (
        "a planta não separa: sem o afastamento os dois textos já não colidiriam, e o teste acima "
        "estaria a passar por vacuidade"
    )


def _rodape_sem_afastamento(figura: Figure, recorte: Recorte) -> None:
    """O `aplica` antigo: reserva a faixa do rodapé e **não** afasta as marcas dela.

    Cada linha é a que `recorte.aplica` corre hoje, menos o `tight_layout` — e os quatro números de
    colocação saem de `recorte`, e não de literais repetidos aqui: uma planta com o seu próprio
    tamanho de fonte mediria outra figura.
    """
    linhas = quebra(recorte.texto(), colunas_da_figura(figura))
    altura = len(linhas) * TAMANHO_DA_FONTE * ENTRELINHA / (figura.get_figheight() * 72.0)
    figura.subplots_adjust(bottom=max(figura.subplotpars.bottom, altura + 3.0 * MARGEM))
    figura.text(
        MARGEM,
        MARGEM,
        "\n".join(linhas),
        ha="left",
        va="bottom",
        fontsize=TAMANHO_DA_FONTE,
        linespacing=ENTRELINHA,
    )


@pytest.mark.parametrize("totais", [None, 0, -1, "seis", float("nan"), True])
def test_sem_populacao_o_recorte_recusa_como_MATRIZ_VAZIA(totais: Any) -> None:  # noqa: N802
    """**`MatrizVaziaError` e não `RecorteInvalidoError`, e a escolha é sobre quem apanha.**

    `PainelDeFigura._troca_a_figura` apanha a primeira e mostra texto no lugar da figura; qualquer
    outra excepção levantada dentro de um *slot* do Qt é engolida para o `stderr`, com o painel a
    parar de actualizar sem dizer nada. Diagnóstico ausente é população vazia.
    """
    with pytest.raises(MatrizVaziaError, match="de quantas features"):
        recorte_da_deriva(n_features_usadas=10, n_features_totais=totais)


def test_com_populacao_o_recorte_SAI() -> None:  # noqa: N802
    """A outra ponta: um recorte válido não levanta, e carrega os três textos deste módulo."""
    recorte = recorte_da_deriva(n_features_usadas=USADAS, n_features_totais=TOTAIS)
    assert (recorte.desenhadas, recorte.disponiveis) == (USADAS, TOTAIS)
    assert (recorte.amostragem, recorte.matriz, recorte.metodo) == (AMOSTRAGEM, MATRIZ, METODO)


def test_numero_de_features_usadas_ilegivel_conta_como_ZERO() -> None:  # noqa: N802
    """Guarda de fronteira: `None` no fio vira `0`, e `0` é declarável — `0 de 100`."""
    recorte = recorte_da_deriva(n_features_usadas=None, n_features_totais=TOTAIS)
    assert recorte.desenhadas == 0


# ---------------------------------------------------------------------------
# 7. O corpus real — o `neg_hilic`, com os dois lotes sugeridos pelo diagnóstico
# ---------------------------------------------------------------------------

MODO_REAL: Final = "neg_hilic"


def test_o_neg_hilic_real_atravessa_da_ingestao_ate_a_figura() -> None:
    """**Ponta a ponta sobre o corpus real, sem número fabricado (regra 1 do README).**

    O que se afirma são propriedades da cadeia — alinhamento, ordem, população —, e as contagens que
    aparecem são **observação sobre este arquivo**, nomeada, nunca invariante. Os sete
    QCs e o lote único `A` são o que o `neg_hilic` tem hoje.

    **Os dois lotes desta unidade são os SUGERIDOS pelo diagnóstico**, e não os do arquivo: é essa
    a partição que a figura tem de separar.
    """
    matriz = load_matrix(
        measurements_path(MODO_REAL),
        sample_list_path(MODO_REAL),
        MODO_REAL,
        orientation="features-in-rows",
    )
    estrutura = structural_metrics(matriz, pretransform=PRETRANSFORM)
    qc = matriz.qc_mask
    ordens = tuple(int(v) for v in matriz.injection_orders[qc])
    scores = tuple(float(v) for v in estrutura.qc_pc1_trajectory)

    # Observação sobre ESTE arquivo, nomeado — não invariante.
    assert len(scores) == len(ordens) == int(qc.sum())
    assert sorted(set(matriz.batches)) == ["A"], (
        f"o {MODO_REAL} declara um lote único no arquivo; isto é observação sobre ele"
    )

    # (a) O lote do ARQUIVO: uma série só.
    figura = figura_da_deriva_do_qc(
        scores=scores,
        ordens_de_injecao=ordens,
        lotes=tuple(matriz.batches[i] for i in np.nonzero(qc)[0]),
        recorte=recorte_da_deriva(
            n_features_usadas=estrutura.n_features_used, n_features_totais=matriz.n_features
        ),
    )
    eixo = figura.axes[0]
    assert confere_series(eixo, {rotulo_do_lote("A", len(scores)): (ordens, scores)}) == []
    assert confere_vista(eixo) == []
    assert str(len(scores)) in eixo.get_title()

    # (b) A SUGESTÃO do diagnóstico: se ela partir a corrida, a figura separa.
    perfil = diagnose(matriz, policy=DiagnosisPolicy(**LIMIARES_INICIAIS))
    sugeridos = tuple(str(perfil.suggested_batches[i]) for i in np.nonzero(qc)[0])
    sugerida = figura_da_deriva_do_qc(
        scores=scores,
        ordens_de_injecao=ordens,
        lotes=sugeridos,
        recorte=recorte_da_deriva(
            n_features_usadas=estrutura.n_features_used, n_features_totais=matriz.n_features
        ),
    )
    desenhadas = series(sugerida.axes[0])
    assert len(desenhadas) == len(set(sugeridos)), (
        f"a sugestão do diagnóstico parte os QCs do {MODO_REAL} em {sorted(set(sugeridos))} e a "
        f"figura desenhou {[s.rotulo for s in desenhadas]}"
    )
    assert confere_vista(sugerida.axes[0]) == []
    # Todo QC entra em exactamente uma série — nenhum se perde na partição.
    assert sum(s.y.size for s in desenhadas) == len(scores)


def test_a_rota_de_diagnostico_SERVE_a_trajetoria_alinhada() -> None:  # noqa: N802
    """O fio: o que a rota põe em `trajetoria_dos_qcs` é o que a figura consome, alinhado.

    **Sobre a rota e não sobre um corpo fabricado**, porque a pergunta é se os quatro vetores saem
    da MESMA máscara — e um corpo escrito à mão responderia sim por construção.
    """
    from fastapi.testclient import TestClient

    from refino.api.app import build_app

    with (
        TestClient(build_app()) as cliente,
        measurements_path(MODO_REAL).open("rb") as medidas,
        sample_list_path(MODO_REAL).open("rb") as lista,
    ):
        resposta = cliente.post(
            "/matrizes/diagnosticar",
            files={"measurements": medidas, "sample_list": lista},
            data={"mode": MODO_REAL, "orientation": "features-in-rows"},
        )
    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    trajetoria = corpo["trajetoria_dos_qcs"]

    n = len(trajetoria["scores"])
    assert n > 0, "âncora: a rota serviu trajetória vazia e as asserções abaixo seriam vácuas"
    assert len(trajetoria["ordens_de_injecao"]) == n
    assert len(trajetoria["lotes"]) == n
    assert trajetoria["ordens_de_injecao"] == sorted(trajetoria["ordens_de_injecao"])
    assert 0 < trajetoria["n_features_usadas"] <= trajetoria["n_features_totais"]
    assert trajetoria["n_features_totais"] == corpo["n_features"]

    # E o corpo servido constrói a figura sem mais nada.
    figura = figura_da_deriva_do_qc(
        scores=trajetoria["scores"],
        ordens_de_injecao=trajetoria["ordens_de_injecao"],
        lotes=trajetoria["lotes"],
        recorte=recorte_da_deriva(
            n_features_usadas=trajetoria["n_features_usadas"],
            n_features_totais=trajetoria["n_features_totais"],
        ),
    )
    assert confere_vista(figura.axes[0]) == []

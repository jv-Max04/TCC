"""A figura da rede: os segmentos, o vazio como resultado, e as DUAS contagens de aresta.

## O que este arquivo afirma, e o que ele deliberadamente NÃO afirma

**Nenhuma asserção sobre um número de aresta do corpus real.** Quantas arestas a rede acha é
resultado do estudo — e neste corpus o desfecho esperado é **zero** (R12b). O que aqui se afirma é
que a figura desenha exactamente as arestas que recebeu, que declara a contagem e o limiar, e que o
vazio sai como resultado em vez de como tela em branco.

**Nenhuma asserção sobre legibilidade.** As classes de asserção do arnês leem o objeto `Figure`;
nenhuma delas sabe se sessenta vértices à volta de um círculo se distinguem. Isso vê-se na imagem
montada, e a área que sobra para o dado é o que `tests/unit/test_geometria_das_figuras.py`
mede.

## A planta que esta família tem e as outras não: os SEGMENTOS

Esta é a primeira figura de grafo do pacote, e as arestas entram como `LineCollection`. Antes da
`arnes_de_figura.series` lia uma coleção de segmentos pelos **deslocamentos** — e devolvia
*"uma série com um ponto na origem"*, o par nulo `[[0., 0.]]`, com o conteúdo que fosse. Sobre esta
figura, isso significaria que trocar as arestas todas não mudava asserção nenhuma.

`test_o_arnes_LE_os_vertices_dos_segmentos_e_nao_o_par_nulo` afirma que deixou de ser verdade, e é
a âncora de tudo o que este arquivo compara: sem ela, cada `confere_series` abaixo passaria por
vacuidade sobre um ponto que ninguém desenhou.

## A derivação das coordenadas é INDEPENDENTE do construtor

`posicoes_esperadas` abaixo reescreve a disposição circular em vez de importar
`rede.posicoes_do_circulo`. Se o teste chamasse a função do módulo sob teste, os dois errariam
juntos — é a metade que faz a classe (a) valer alguma coisa, e é a mesma razão de
`arnes_de_figura.milhar_do_teste` existir.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pytest

from refino.plots.base import FeaturesDemaisError, MatrizVaziaError
from refino.plots.recorte import RecorteInvalidoError
from refino.plots.rede import (
    METODO_NAO_DECLARADO,
    ROTULO_DAS_NEGATIVAS,
    ROTULO_DAS_POSITIVAS,
    ROTULO_DO_EIXO,
    TAMANHO_DO_VERTICE,
    TITULO_DA_REDE,
    Rede,
    RedeDesalinhadaError,
    figura_da_rede_de_correlacao_parcial,
    posicoes_do_circulo,
    recorte_da_rede,
    rede_do_fio,
    veredito_da_rede,
)
from refino.statistics.clustering import TETO_DE_FEATURES_PARA_DESENHAR
from tests.arnes_de_figura import (
    confere_rotulos,
    confere_series,
    confere_vista,
    milhar_do_teste,
    numeros_declarados,
    planta_eixos_trocados,
    planta_serie_removida,
    series,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.figure import Figure

# A população do `pos_hilic`, literal aqui de propósito: é a fonte independente da classe (d).
POPULACAO: Final = 6035
MATRIZ: Final = "a matriz corrigida"
AMOSTRAGEM: Final = "as 8 sorteadas com semente 42"

FEATURES: Final = tuple(f"1.{i}_10{i}.5m/z" for i in range(1, 9))

METODO: Final = (
    "graphical lasso com penalidade por validação cruzada, mais a correção de viés de "
    "Jankova e van de Geer (2015)"
)

# Três arestas, **os dois sinais**, e três features fora delas. Nenhuma das escolhas é decorativa:
#
# - os dois sinais são o que faz as DUAS coleções existirem, e sem isso metade dos artistas que o
#   construtor sabe pôr nunca é desenhada;
# - as três features isoladas fazem `frase_contada` correr o ramo do plural do isolamento;
# - o primeiro `q` está abaixo de 0,05 e os outros dois acima, o que põe o veredito no ramo que
#   escreve NÚMERO — o terceiro dos três.
ARESTAS: Final[tuple[dict[str, Any], ...]] = (
    {
        "feature_a": FEATURES[0],
        "feature_b": FEATURES[1],
        "correlacao_parcial": 0.62,
        "pvalue": 0.0008,
        "qvalue_fdr": 0.021,
    },
    {
        "feature_a": FEATURES[1],
        "feature_b": FEATURES[2],
        "correlacao_parcial": -0.48,
        "pvalue": 0.0041,
        "qvalue_fdr": 0.062,
    },
    {
        "feature_a": FEATURES[2],
        "feature_b": FEATURES[4],
        "correlacao_parcial": 0.31,
        "pvalue": 0.0192,
        "qvalue_fdr": 0.204,
    },
)


def posicoes_esperadas(quantos: int) -> np.ndarray:
    """A disposição circular, **reescrita aqui e não importada do módulo sob teste**.

    `theta_i = 2 pi i / k`, `x = cos`, `y = sin`. É a fonte independente da classe (a): com o teste
    a chamar `rede.posicoes_do_circulo`, os dois errariam juntos e a comparação abaixo seria a
    função a concordar consigo própria.
    """
    angulos = 2.0 * np.pi * np.arange(quantos, dtype=float) / float(quantos)
    return np.column_stack((np.cos(angulos), np.sin(angulos)))


def carga(
    *,
    features: tuple[str, ...] = FEATURES,
    arestas: tuple[dict[str, Any], ...] = ARESTAS,
    n_arestas: int | None = None,
    n_pares_testados: int = 28,
    alfa: float = 0.05,
    metodo: Any = METODO,
) -> Rede:
    """Uma `Rede` plausível. `n_arestas` a `None` significa *"coerente com a lista"*.

    **O padrão não é `len(arestas)` escrito no corpo por acaso:** os testes que plantam a
    divergência passam o número **explicitamente**, e é essa separação que deixa a guarda ser
    exercitada nas duas direções sem cada chamador ter de repetir a contagem.
    """
    return rede_do_fio(
        features=list(features),
        arestas=[dict(a) for a in arestas],
        n_arestas=len(arestas) if n_arestas is None else n_arestas,
        n_pares_testados=n_pares_testados,
        alfa=alfa,
        penalidade=0.2431,
        n_amostras_usadas=61,
        metodo=metodo,
    )


def recorte_de(rede: Rede, *, n_features_totais: Any = POPULACAO):
    return recorte_da_rede(
        rede=rede,
        n_features_totais=n_features_totais,
        matriz=MATRIZ,
        amostragem=AMOSTRAGEM,
    )


def figura_de(rede: Rede) -> Figure:
    return figura_da_rede_de_correlacao_parcial(rede=rede, recorte=recorte_de(rede))


def esperadas_de(rede: Rede) -> dict[str, tuple[list[float], list[float]]]:
    """As séries que a figura tem de ter, derivadas da entrada por caminho independente.

    Os vértices de cada segmento entram concatenados, que é o que `_vertices_dos_segmentos` devolve
    para uma `LineCollection` — a estrutura de aresta perde-se ali de propósito, e o que separa duas
    figuras com os mesmos vértices e arestas diferentes é a classe (d).
    """
    posicoes = posicoes_esperadas(len(rede.features))
    onde = {nome: i for i, nome in enumerate(rede.features)}
    esperadas: dict[str, tuple[list[float], list[float]]] = {}
    for rotulo, positiva in ((ROTULO_DAS_POSITIVAS, True), (ROTULO_DAS_NEGATIVAS, False)):
        escolhidas = [a for a in rede.arestas if a.positiva is positiva]
        if not escolhidas:
            continue
        xs: list[float] = []
        ys: list[float] = []
        for aresta in escolhidas:
            for nome in (aresta.feature_a, aresta.feature_b):
                xs.append(float(posicoes[onde[nome], 0]))
                ys.append(float(posicoes[onde[nome], 1]))
        esperadas[f"{rotulo} — {milhar_do_teste(len(escolhidas))}"] = (xs, ys)
    esperadas[f"{milhar_do_teste(len(rede.features))} features"] = (
        [float(v) for v in posicoes[:, 0]],
        [float(v) for v in posicoes[:, 1]],
    )
    return esperadas


# ---------------------------------------------------------------------------
# 1. A disposição circular — determinística, e sobre o círculo unitário
# ---------------------------------------------------------------------------


def test_a_disposicao_e_DETERMINISTICA_sem_semente_nenhuma() -> None:  # noqa: N802
    """Regra 3 do README. Duas chamadas com a mesma entrada dão o mesmo array, byte a byte.

    **É esta propriedade que uma disposição por forças não tem**, e é por ela que o círculo foi
    escolhido — ver o cabeçalho do módulo.
    """
    assert np.array_equal(posicoes_do_circulo(8), posicoes_do_circulo(8))
    assert not np.array_equal(posicoes_do_circulo(8), posicoes_do_circulo(7)[:8])


def test_os_vertices_caem_no_CIRCULO_UNITARIO_e_sao_igualmente_espacados() -> None:  # noqa: N802
    """O oráculo reproduzido à mão sobre um caso cuja resposta já se sabe.

    Raio 1 em todos, e a distância angular entre vizinhos consecutivos constante. Com um deles a
    falhar, a figura sairia bem formada e as posições seriam outra coisa.
    """
    posicoes = posicoes_do_circulo(6)
    raios = np.hypot(posicoes[:, 0], posicoes[:, 1])
    assert np.allclose(raios, 1.0)

    angulos = np.arctan2(posicoes[:, 1], posicoes[:, 0])
    passos = np.diff(np.unwrap(angulos))
    assert passos.size == 5, "âncora: sem passo nenhum a asserção abaixo seria vácua"
    assert np.allclose(passos, 2.0 * np.pi / 6.0)
    assert np.allclose(posicoes[0], (1.0, 0.0)), "a disposição deixou de começar em (1, 0)"


def test_a_disposicao_de_ZERO_vertices_e_recusa_tipada() -> None:  # noqa: N802
    """Não há círculo de nada. A recusa é da família que o painel apanha."""
    with pytest.raises(MatrizVaziaError, match="círculo de nada"):
        posicoes_do_circulo(0)


# ---------------------------------------------------------------------------
# 2. Classe (a): as arestas são SEGMENTOS, e o arnês lê os vértices deles
# ---------------------------------------------------------------------------


def test_o_arnes_LE_os_vertices_dos_segmentos_e_nao_o_par_nulo() -> None:  # noqa: N802
    """**A âncora de todo este arquivo.**

    Antes daquela unidade, `series` lia uma `LineCollection` pelos **deslocamentos** e devolvia uma
    série com **um** ponto em `(0, 0)`, qualquer que fosse o conteúdo. Sobre esta figura isso
    significaria que cada `confere_series` abaixo passava sobre um ponto que ninguém desenhou.

    Dois segmentos positivos dão **quatro** vértices, e é esse número que se afirma. A âncora de
    não-vacuidade está na mesma função: sem ela, uma figura sem coleção nenhuma satisfaria
    a asserção por conjunto vazio.
    """
    rede = carga()
    desenhadas = {s.rotulo: s for s in series(figura_de(rede).axes[0])}
    assert desenhadas, "a figura não desenhou artista nenhum — tudo abaixo seria vácuo"

    rotulo_positivas = f"{ROTULO_DAS_POSITIVAS} — {milhar_do_teste(rede.n_positivas)}"
    assert rotulo_positivas in desenhadas, sorted(desenhadas)
    serie = desenhadas[rotulo_positivas]
    assert len(serie.x) == 2 * rede.n_positivas, (
        f"a série das positivas traz {len(serie.x)} vértice(s) para {rede.n_positivas} aresta(s). "
        "Um único ponto aqui é o par nulo dos deslocamentos, que é a leitura que a U18 fechou"
    )
    assert not np.allclose(serie.x, 0.0), (
        "todos os `x` são zero: é a assinatura exacta do par nulo `[[0., 0.]]` a ser lido como dado"
    )


def test_cada_aresta_vira_UM_segmento_entre_as_posicoes_dos_dois_vertices() -> None:  # noqa: N802
    """Classe (a): o dado desenhado é o dado que entrou, série a série.

    A planta entra como **argumento** de `confere_series` — negativo em processo, que não consegue
    ficar inerte.
    """
    rede = carga()
    assert confere_series(figura_de(rede).axes[0], esperadas_de(rede)) == []


def test_par_causal_serie_removida() -> None:
    """Tirar uma série do traço tem de ser acusado — a metade que prova que a classe (a) age."""
    rede = carga()
    figura = figura_de(rede)
    eixo = figura.axes[0]
    esperadas = esperadas_de(rede)
    assert confere_series(eixo, esperadas) == [], "âncora: a árvore limpa não pode acusar"

    apagada = planta_serie_removida(eixo)
    achados = confere_series(eixo, esperadas)
    assert any(apagada in a for a in achados), achados


def test_par_causal_eixos_trocados_sobre_a_COLECAO_DE_ARESTAS() -> None:  # noqa: N802
    """A planta cai sobre o primeiro artista, e ele tem de ser o feixe de arestas.

    **A ordem de desenho é contrato:** `series` percorre `ax.collections` na ordem em que foram
    acrescentadas, e `planta_eixos_trocados` mutila `ax.collections[0]` enquanto **declara** ter
    mutilado `series(ax)[0]`. Com os vértices desenhados primeiro, a planta diria o rótulo de uma
    série e mexeria noutra — e a reprovação deixaria de ser atribuível a ela.

    A âncora de separação vive dentro da própria planta: ela recusa-se quando `x == y`.
    """
    rede = carga()
    figura = figura_de(rede)
    eixo = figura.axes[0]
    esperadas = esperadas_de(rede)
    assert confere_series(eixo, esperadas) == [], "âncora: a árvore limpa não pode acusar"

    mutilada = planta_eixos_trocados(eixo)
    assert mutilada == f"{ROTULO_DAS_POSITIVAS} — {milhar_do_teste(rede.n_positivas)}", (
        f"a planta caiu sobre {mutilada!r} e não sobre o feixe de arestas: a ordem de desenho "
        "mudou, e `planta_eixos_trocados` passou a declarar um artista e a mexer noutro"
    )
    achados = confere_series(eixo, esperadas)
    assert any(mutilada in a for a in achados), achados


# ---------------------------------------------------------------------------
# 3. As DUAS contagens de aresta, com a não-vacuidade na mesma função
# ---------------------------------------------------------------------------


def test_a_contagem_DESENHADA_e_a_contagem_DECLARADA_batem() -> None:  # noqa: N802
    """Os dois caminhos que produzem o número de arestas, comparados um contra o outro.

    | Caminho | De onde vem |
    |---|---|
    | **desenhado** | os vértices que `_vertices_dos_segmentos` lê das coleções, a dividir por 2 |
    | **declarado** | `n_arestas`, que o núcleo conta por `count_nonzero` sobre a máscara |

    **A asserção de não-vacuidade está na MESMA função e vem ANTES**: igualdade entre duas
    contagens que podem ambas ser zero é asserção de ausência, e ela passa numa figura sem coleção
    nenhuma — que é exactamente o desfecho de um construtor que tivesse deixado de desenhar arestas.
    """
    rede = carga()
    figura = figura_de(rede)

    vertices_de_aresta = sum(
        len(s.x)
        for s in series(figura.axes[0])
        if s.rotulo.startswith((ROTULO_DAS_POSITIVAS, ROTULO_DAS_NEGATIVAS))
    )
    assert vertices_de_aresta > 0, (
        "NÃO-VACUIDADE: nenhuma coleção de aresta foi desenhada, e a igualdade abaixo seria "
        "satisfeita por 0 == 0 sobre uma figura que não desenhou aresta nenhuma"
    )
    assert rede.n_arestas > 0, (
        "NÃO-VACUIDADE: a carga declara zero arestas, e a igualdade abaixo compararia zero com zero"
    )

    desenhadas = vertices_de_aresta // 2
    assert desenhadas == rede.n_arestas, (
        f"a imagem desenha {desenhadas} aresta(s) e o resultado declara {rede.n_arestas}. Os dois "
        "números saem de caminhos diferentes, e é a divergência entre eles que esta asserção "
        "existe para tornar visível"
    )
    assert 2 * desenhadas == vertices_de_aresta, (
        "algum segmento não tem exactamente dois vértices — uma aresta desenhada como polilinha "
        "contaria a mais nesta divisão"
    )


def test_a_contagem_DECLARADA_divergente_e_recusa_tipada() -> None:  # noqa: N802
    """O par causal da guarda: a mesma lista com `n_arestas` mentido tem de reprovar na fronteira.

    **É a metade que torna o teste acima capaz de falhar.** Sem esta guarda, uma carga com os dois
    números divergentes chegaria ao construtor, que desenharia o comprimento da lista e escreveria
    o outro número no título — e o teste acima só a apanharia se alguém a construísse.
    """
    with pytest.raises(RedeDesalinhadaError, match="caminhos DIFERENTES"):
        carga(n_arestas=len(ARESTAS) + 1)
    # A outra ponta: com o número coerente, a mesma carga não levanta.
    assert carga(n_arestas=len(ARESTAS)).n_arestas == len(ARESTAS)


def test_a_soma_dos_DOIS_SINAIS_e_a_contagem_de_arestas() -> None:  # noqa: N802
    """Toda aresta cai em exactamente uma das duas coleções — ver `>= 0` no cabeçalho.

    Uma aresta que não caísse em nenhuma seria contada no rodapé e ausente da imagem: rótulo
    certo sobre a população errada, aplicado a um artista.
    """
    rede = carga()
    assert rede.n_positivas > 0 and rede.n_negativas > 0, (
        "âncora: a carga tem de ter os dois sinais, senão esta soma não separa nada"
    )
    assert rede.n_positivas + rede.n_negativas == rede.n_arestas


def test_um_SINAL_SO_desenha_UMA_colecao_e_nao_duas_com_uma_vazia() -> None:  # noqa: N802
    """Uma coleção vazia entraria em `series` como série sem vértice e na legenda como *"0"*.

    **A planta cai sobre dado em que as duas coisas DIFEREM:** a mesma figura com os dois sinais tem
    três séries, e com um sinal só tem duas. As duas contagens estão na mesma função.
    """
    so_positivas = tuple(a for a in ARESTAS if a["correlacao_parcial"] >= 0)
    assert 0 < len(so_positivas) < len(ARESTAS), "âncora: a planta tem de tirar alguma aresta"

    com_os_dois = {s.rotulo for s in series(figura_de(carga()).axes[0])}
    com_um = {s.rotulo for s in series(figura_de(carga(arestas=so_positivas)).axes[0])}
    assert len(com_os_dois) == len(com_um) + 1, (
        f"com os dois sinais há {sorted(com_os_dois)} e com um só há {sorted(com_um)}: a planta "
        "não separa, ou a coleção vazia continuou a ser desenhada"
    )
    assert not any(r.startswith(ROTULO_DAS_NEGATIVAS) for r in com_um), sorted(com_um)


# ---------------------------------------------------------------------------
# 4. O VAZIO é desenhado como resultado — R12b
# ---------------------------------------------------------------------------


def test_a_rede_SEM_ARESTA_desenha_os_vertices_e_declara_a_contagem() -> None:  # noqa: N802
    """R12b: zero arestas é o desfecho **esperado** deste corpus, e não uma figura por acabar.

    **A planta cai sobre dado em que as duas coisas diferem:** a mesma população com arestas
    desenha três séries e sem arestas desenha uma. As duas contagens estão na mesma função, e o
    texto do desfecho só existe num dos dois lados.
    """
    vazia = carga(arestas=())
    com_arestas = carga()
    assert vazia.n_arestas == 0 and com_arestas.n_arestas > 0, "âncora: as duas cargas não diferem"

    figura = figura_de(vazia)
    eixo = figura.axes[0]
    desenhadas = {s.rotulo for s in series(eixo)}
    assert desenhadas == {f"{milhar_do_teste(len(FEATURES))} features"}, sorted(desenhadas)

    declarado = " ".join(" ".join(t.get_text().split()) for t in eixo.texts)
    assert declarado, "a figura vazia não escreveu nada dentro do eixo: é tela em branco"
    assert "Nenhuma aresta sobreviveu" in declarado, declarado
    assert milhar_do_teste(vazia.n_pares_testados) in declarado, declarado
    assert "A rede CORREU" in declarado, declarado

    com = {s.rotulo for s in series(figura_de(com_arestas).axes[0])}
    assert len(com) > len(desenhadas), "as duas figuras têm as mesmas séries: a planta não separa"
    assert not figura_de(com_arestas).axes[0].texts, (
        "a figura COM arestas escreveu o texto do desfecho vazio — ele é do outro ramo"
    )


# Os quatro tamanhos em que a sobreposição é medida, em (polegadas, polegadas, ppp, rótulo). O
# terceiro é o painel achatado medido, e é ele que decide `FAIXA_DO_DESFECHO`.
TAMANHOS: Final = (
    (7.0, 4.5, 100.0, "7,0x4,5 @100 — a nascença do construtor"),
    (10.0, 6.0, 100.0, "10x6 @100"),
    (1860.0 / 96.0, 302.0 / 96.0, 96.0, "1860x302 @96 — o painel achatado"),
    (1200.0 / 96.0, 700.0 / 96.0, 96.0, "1200x700 @96"),
)


def vertices_tapados(figura: Figure) -> list[int]:
    """Os índices dos vértices que a frase do desfecho cobre. **Lista vazia é o desfecho sadio.**

    A caixa do texto é inflada pelo **raio do marcador**, porque o que se compara é o glifo e não o
    centro dele: um vértice cujo centro fica a um pixel da borda da caixa está, na tela, metade por
    baixo do texto.

    **Devolve a lista e não um booleano**, que é o que torna o negativo honesto: a planta entra como
    argumento da figura, e o teste percorre exactamente o caminho que o portão percorre.
    """
    figura.canvas.draw()
    eixo = figura.axes[0]
    if not eixo.texts:
        return []
    caixa = eixo.texts[0].get_window_extent(figura.canvas.get_renderer())
    raio = math.sqrt(TAMANHO_DO_VERTICE) / 2.0 * float(figura.dpi) / 72.0
    tela = eixo.transData.transform(posicoes_esperadas(len(FEATURES)))
    dentro = (
        (tela[:, 0] >= caixa.x0 - raio)
        & (tela[:, 0] <= caixa.x1 + raio)
        & (tela[:, 1] >= caixa.y0 - raio)
        & (tela[:, 1] <= caixa.y1 + raio)
    )
    return [int(i) for i in np.flatnonzero(dentro)]


@pytest.mark.parametrize(
    ("largura", "altura", "ppp", "rotulo"), TAMANHOS, ids=lambda x: str(x)[:28]
)
def test_a_frase_do_desfecho_NAO_TAPA_vertice_nenhum(  # noqa: N802
    largura: float, altura: float, ppp: float, rotulo: str
) -> None:
    """**Geometria, e é a classe que as quatro do arnês não alcançam.**

    O dado está desenhado, dentro do limite, visível e opaco; o que o esconde é **outro artista**, e
    sobreposição é propriedade da imagem montada. A primeira renderização desta figura escreveu a
    frase de R12b por cima de três vértices, e nenhuma asserção deste arquivo o viu — foi preciso
    olhar. Isto mecaniza o que se viu, para não ser preciso olhar outra vez.

    **A comparação é entre a caixa REAL do texto e a posição REAL dos vértices**, medidas com o
    renderizador depois de a figura estar desenhada: qualquer estimativa aqui seria a segunda fonte
    de uma quantidade que o `matplotlib` já sabe.

    **A âncora de não-vacuidade está na mesma função**: uma figura sem texto e uma sem
    vértice satisfariam a lista vazia sem haver nada a medir.
    """
    figura = figura_de(carga(arestas=()))
    figura.set_size_inches(largura, altura)
    figura.set_dpi(ppp)
    figura.canvas.draw()
    eixo = figura.axes[0]

    assert eixo.texts, "a figura vazia não escreveu a frase do desfecho — não há o que medir"
    caixa = eixo.texts[0].get_window_extent(figura.canvas.get_renderer())
    assert caixa.width > 0 and caixa.height > 0, (
        f"a caixa do texto mede {caixa.width}x{caixa.height}: sem área, nada se sobrepõe a ela e "
        "a asserção abaixo passaria por vacuidade"
    )
    tapados = vertices_tapados(figura)
    assert tapados == [], (
        f"a frase do desfecho tapa {len(tapados)} vértice(s) em {rotulo}: {tapados}. REMEÇA "
        "`FAIXA_DO_DESFECHO` ou encurte a frase — as duas metades do conserto estão "
        "medidas na docstring da constante"
    )


def test_o_predicado_da_sobreposicao_ACUSA_a_faixa_encolhida() -> None:  # noqa: N802
    """O par causal do teste acima, com a planta no parâmetro que governa a geometria.

    **Sem ele, `vertices_tapados` podia devolver lista vazia sempre** — por ler o texto errado, por
    inflar a caixa ao contrário, por a transformação estar em coordenadas de eixo — e os quatro
    tamanhos acima seriam quatro verdes vazios.

    **A planta é `FAIXA_DO_DESFECHO` encolhida para a margem comum**, que é exactamente o valor com
    que o defeito foi visto em 18/09/2026. Os dois desfechos estão na mesma função.
    """
    import refino.plots.rede as modulo

    original = modulo.FAIXA_DO_DESFECHO
    try:
        sem_planta = vertices_tapados(figura_de(carga(arestas=())))
        modulo.FAIXA_DO_DESFECHO = modulo.MARGEM_DO_CIRCULO  # type: ignore[misc]
        assert original != modulo.FAIXA_DO_DESFECHO, "A PLANTA NÃO ATERROU: a faixa não mudou"
        com_planta = vertices_tapados(figura_de(carga(arestas=())))
    finally:
        modulo.FAIXA_DO_DESFECHO = original  # type: ignore[misc]

    assert sem_planta == [], f"a árvore limpa já tapa vértices: {sem_planta}"
    assert com_planta, (
        "com a faixa encolhida à margem comum NENHUM vértice fica tapado: ou a planta não aterrou, "
        "ou o predicado não consegue acusar — e os quatro tamanhos acima seriam verdes vazios"
    )


def test_o_rodape_da_rede_vazia_diz_a_contagem_e_o_limiar() -> None:
    """A figura vazia continua a declarar o recorte e o método (R3, R5b, R12b).

    Uma imagem sem aresta e sem frase lê-se como defeito do aplicativo; com a contagem, o limiar e
    o denominador de pares, lê-se como o resultado que é.
    """
    rodape = " ".join(" ".join(t.get_text().split()) for t in figura_de(carga(arestas=())).texts)
    for obrigatorio in (
        "nenhuma aresta sobreviveu ao corte",
        "Corte no p CRU",
        "NENHUM par sobreviveu ao corte",
        "É ACHADO e não defeito",
    ):
        assert obrigatorio in rodape, f"o rodapé não declara {obrigatorio!r}: {rodape}"


# ---------------------------------------------------------------------------
# 5. O veredito — TRÊS desfechos, e todos vistos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arestas", "trecho"),
    [
        ((), "NENHUM par sobreviveu"),
        (tuple(a for a in ARESTAS if a["qvalue_fdr"] >= 0.05), "NENHUMA delas sobrevive"),
        (ARESTAS, "sobrevivem também"),
    ],
)
def test_o_veredito_tem_os_TRES_desfechos_e_cada_um_e_alcancavel(  # noqa: N802
    arestas: tuple[dict[str, Any], ...], trecho: str
) -> None:
    """A verificação pede que se tenham visto todos, e «dois» é o piso e não o número.

    O primeiro é o deste corpus; o segundo é o que separa *"há candidatos"* de *"há achados"*.
    """
    assert trecho in veredito_da_rede(carga(arestas=arestas))


def test_os_TRES_vereditos_sao_TEXTOS_DISTINTOS() -> None:  # noqa: N802
    """Âncora do teste acima: com duas frases iguais, um dos `trecho` casaria por acaso."""
    vereditos = {
        veredito_da_rede(carga(arestas=a))
        for a in ((), tuple(x for x in ARESTAS if x["qvalue_fdr"] >= 0.05), ARESTAS)
    }
    assert len(vereditos) == 3, vereditos


def test_o_corte_no_q_usa_o_MESMO_alfa_que_viajou_e_nao_um_literal() -> None:  # noqa: N802
    """Nenhum limiar novo nasce do lado do desenho.

    **A planta cai sobre dado em que as duas coisas diferem:** os três `q` da carga são `0,021`,
    `0,062` e `0,204`, então com `alfa` a 0,05 passa **uma** e com `alfa` a 0,3 passam **três**. Um
    `0,05` escrito no módulo daria o mesmo número nos dois.
    """
    apertado = carga(alfa=0.05).n_sob_o_mesmo_corte_no_q
    frouxo = carga(alfa=0.3).n_sob_o_mesmo_corte_no_q
    assert apertado == 1, apertado
    assert frouxo == len(ARESTAS), frouxo
    assert apertado != frouxo, (
        "os dois alfas dão a mesma contagem: ou a planta não separa, ou o corte no `q` está preso "
        "a um literal do módulo em vez de ler o alfa que viajou"
    )


# ---------------------------------------------------------------------------
# 6. Classes (b), (c) e (d)
# ---------------------------------------------------------------------------


def test_os_rotulos_do_eixo_avisam_que_a_POSICAO_nao_e_medida() -> None:  # noqa: N802
    """Classe (b), e aqui o rótulo é uma ressalva e não um nome de quantidade.

    A coordenada de um vértice é inventada por `posicoes_do_circulo`; sem a ressalva e sem marcas,
    quem olha não tem como saber que aquele eixo não mede nada.
    """
    rede = carga()
    figura = figura_de(rede)
    eixo = figura.axes[0]
    achados = confere_rotulos(
        eixo,
        x=ROTULO_DO_EIXO,
        y="",
        titulo=eixo.get_title(),
        legenda=list(esperadas_de(rede)),
    )
    assert achados == [], achados
    assert eixo.get_xticks().size == 0 and eixo.get_yticks().size == 0, (
        "o eixo tem marca numérica: uma coordenada inventada ao lado de uma marca lê-se como medida"
    )


def test_o_titulo_nomeia_o_METODO_a_contagem_e_o_limiar() -> None:  # noqa: N802
    """R12 na imagem. O nome traz os DOIS passos — sem a correção de viés não é o método."""
    titulo = figura_de(carga()).axes[0].get_title()
    assert TITULO_DA_REDE in titulo
    assert "desenviesada" in titulo, "o título não nomeia a correção de viés, que é R8"
    assert "DSPC" in titulo
    assert "p <" in titulo


def test_a_vista_esta_sa() -> None:
    """Classe (c): nada invisível, nada fora do enquadramento, nada em escala trocada."""
    assert confere_vista(figura_de(carga()).axes[0]) == []
    assert confere_vista(figura_de(carga(arestas=())).axes[0]) == []


def test_os_numeros_declarados_SAEM_do_dado_e_nao_de_um_literal() -> None:  # noqa: N802
    """Classe (d): cada número escrito na figura é derivado da entrada por caminho independente.

    **A âncora de não-vacuidade está na mesma função**: uma figura sem número nenhum
    satisfaria a asserção de pertinência abaixo por conjunto vazio.
    """
    rede = carga()
    escritos = numeros_declarados(figura_de(rede))
    assert escritos, "a figura não declara número nenhum — a asserção abaixo seria vácua"

    for esperado in (
        milhar_do_teste(rede.n_arestas),
        milhar_do_teste(rede.n_pares_testados),
        milhar_do_teste(len(FEATURES)),
        milhar_do_teste(POPULACAO),
        "0,05",
        "0,2431",
        "61",
    ):
        assert esperado in escritos, f"{esperado!r} não está entre os números escritos: {escritos}"


def test_o_rodape_conta_as_isoladas_nos_TRES_ramos() -> None:  # noqa: N802
    """`frase_contada` tem três ramos, e `uma` é obrigatório — os três são alcançáveis aqui.

    O plural aritmético — *"as 1 features ficaram isoladas"* — é o defeito que aquela função existe
    para fechar, e uma frase que parece defeituosa põe em dúvida o número ao lado dela.
    """
    # As arestas tocam quatro vértices — `FEATURES[0]`, `[1]`, `[2]` e `[4]` —, então o número de
    # isoladas é `len(features) - 4`. É a mesma conta em três populações, e é ela que põe cada ramo.
    quatro_isoladas = recorte_de(carga()).metodo
    uma_isolada = recorte_de(carga(features=FEATURES[:5])).metodo
    nenhuma_isolada = recorte_de(carga(features=FEATURES[:3], arestas=ARESTAS[:2])).metodo
    assert "as 4 features ficaram isoladas" in quatro_isoladas, quatro_isoladas
    assert "1 feature ficou isolada" in uma_isolada, uma_isolada
    assert "nenhuma feature ficou isolada" in nenhuma_isolada, nenhuma_isolada


def test_o_rodape_conta_as_arestas_nos_TRES_ramos() -> None:  # noqa: N802
    """O mesmo para a contagem de arestas, que é o número que viaja para fora da imagem."""
    assert "nenhuma aresta sobreviveu ao corte" in recorte_de(carga(arestas=())).metodo
    assert "1 aresta sobreviveu ao corte" in recorte_de(carga(arestas=ARESTAS[:1])).metodo
    assert "as 3 arestas sobreviveram ao corte" in recorte_de(carga()).metodo


# ---------------------------------------------------------------------------
# 7. A fronteira: as recusas tipadas
# ---------------------------------------------------------------------------


def test_acima_do_teto_e_recusa_TIPADA_com_os_dois_numeros() -> None:  # noqa: N802
    """Sem ela a janela congela sem nada a acusar — a construção corre na thread da interface.

    **A planta cai sobre dado em que as duas coisas diferem:** o teto mais um levanta e o teto
    exacto não. As duas chamadas estão na mesma função.
    """
    demais = tuple(f"f{i}" for i in range(TETO_DE_FEATURES_PARA_DESENHAR + 1))
    no_teto = demais[:TETO_DE_FEATURES_PARA_DESENHAR]
    assert len(demais) == len(no_teto) + 1, "âncora: as duas populações não diferem"

    rede = carga(features=demais, arestas=())
    with pytest.raises(FeaturesDemaisError, match=str(TETO_DE_FEATURES_PARA_DESENHAR)):
        figura_de(rede)
    assert figura_de(carga(features=no_teto, arestas=())).axes, (
        "o construtor recusou no teto exacto: a recusa é para ACIMA dele"
    )


@pytest.mark.parametrize(
    ("campos", "excecao", "trecho"),
    [
        ({"features": ()}, MatrizVaziaError, "sem feature nenhuma"),
        ({"features": (FEATURES[0], FEATURES[0], FEATURES[1])}, RedeDesalinhadaError, "repetido"),
        (
            {
                "arestas": (
                    {
                        "feature_a": "não existe",
                        "feature_b": FEATURES[1],
                        "correlacao_parcial": 0.5,
                        "pvalue": 0.01,
                        "qvalue_fdr": 0.02,
                    },
                )
            },
            RedeDesalinhadaError,
            "fora dos",
        ),
        (
            {
                "arestas": (
                    {
                        "feature_a": FEATURES[0],
                        "feature_b": FEATURES[0],
                        "correlacao_parcial": 0.5,
                        "pvalue": 0.01,
                        "qvalue_fdr": 0.02,
                    },
                )
            },
            RedeDesalinhadaError,
            "com ela própria",
        ),
        (
            {
                "arestas": (
                    {
                        "feature_a": FEATURES[0],
                        "feature_b": FEATURES[1],
                        "correlacao_parcial": None,
                        "pvalue": 0.01,
                        "qvalue_fdr": 0.02,
                    },
                )
            },
            RedeDesalinhadaError,
            "não finito",
        ),
    ],
)
def test_a_carga_INCOERENTE_e_recusa_tipada(  # noqa: N802
    campos: dict[str, Any], excecao: type[Exception], trecho: str
) -> None:
    """As cinco formas que produziriam uma imagem bem formada e errada.

    A planta entra como **argumento** — negativo em processo, que não consegue ficar inerte.
    """
    with pytest.raises(excecao, match=trecho):
        carga(**campos)


def test_a_carga_SA_nao_levanta() -> None:  # noqa: N802
    """A outra ponta: sem isto, uma fronteira que recusasse tudo passaria nas cinco plantas."""
    rede = carga()
    assert rede.n_vertices == len(FEATURES)
    assert rede.n_arestas == len(ARESTAS)
    # Quatro: as três arestas tocam `FEATURES[0]`, `[1]`, `[2]` e `[4]`, e a população tem oito.
    assert rede.n_isoladas == 4
    assert rede.metodo == METODO


def test_o_metodo_AUSENTE_vira_o_recuo_e_nunca_a_frase_do_metodo_certo() -> None:  # noqa: N802
    """Escrever *"com correção de viés"* por omissão seria afirmar que a correção correu.

    É a afirmação que um atalho apagaria sem deixar marca, e é por isso que o recuo é uma declaração
    de ausência e não o rótulo mais plausível.
    """
    sem_metodo = carga(metodo=None)
    assert sem_metodo.metodo == METODO_NAO_DECLARADO
    rodape = " ".join(" ".join(t.get_text().split()) for t in figura_de(sem_metodo).texts)
    assert METODO_NAO_DECLARADO in rodape, rodape
    assert "correção de viés de Jankova" not in rodape, rodape


def test_o_recorte_recusa_denominador_impossivel() -> None:
    """`desenhadas > disponiveis` é numerador e denominador de populações diferentes."""
    with pytest.raises(RecorteInvalidoError):
        recorte_de(carga(), n_features_totais=len(FEATURES) - 1)

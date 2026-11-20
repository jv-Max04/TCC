"""A figura da ANOVA de N grupos, e **neste corpus ela é vazia, o que é o RESULTADO**.

## O que este módulo prova, e o que ele deliberadamente NÃO prova

| Bloco | O que prova |
|---|---|
| o construtor | os pontos desenhados são os que entraram, e a classe de cada um |
| as DUAS partições | classificação e desenho fecham, cada uma, sobre `n_da_matriz` |
| o `q = 0` | ele conta como sobrevivente, e a figura declara que ele não tem ponto |
| o cruzamento | `contagens.sobreviventes == AnovaCounts.survivors`, sobre o mesmo resultado |
| o corte como ALTURA | a contagem da legenda é o número de pontos acima da linha |
| as plantas do arnês | as classes acusam sobre ESTA figura, e a limpa passa |
| o par causal do desenho | trocar o vetor de `p` pelo de `q` muda a figura |
| as recusas tipadas | vetor desalinhado, par de grupos desalinhado, e o rodapé que não cabe |
| o corpus real | o `pos_hilic` com os quatro grupos, com as âncoras na mesma função |

**O que NÃO se afirma, e vai declarado porque a leitura natural sugeriria o contrário:**

1. **Nenhuma asserção sobre `constant` e `insufficient_groups`.** Eles são a partição de
   `AnovaCounts.statistic_undefined` e **não viajam** — a figura não os recebe e não os declara.
   Quem os afirma é `tests/unit/test_anova.py`, sobre o resultado do núcleo.
2. **Nenhuma asserção de que `0,6684` se reproduza.** Esse número é do `pos_hilic` sob a cascata
   nula, medido em 14/09/2026; o que o teste do corpus real afirma é a **igualdade** entre o que a
   figura declara e o que ele mediu, por caminho independente do construtor.
3. **Nenhuma asserção sobre a escolha entre ANOVA e Kruskal-Wallis.**
   `AnalysisSpec.anova_parametric` continua sem consumidor, e o portão que o guarda vive em
   `tests/unit/test_anova.py`. Aqui a figura lê `metodo` e escreve o que ele disser, e
   `test_o_nome_do_teste_e_DERIVADO_do_metodo` afirma os três desfechos — inclusive o rótulo
   desconhecido.
4. **Nenhuma asserção de geometria.** Rodapé por cima do rótulo, legenda sobre a nuvem e altura de
   eixo passam por todas as classes com cobertura total. O que cobre isso é a leitura da
   imagem montada.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pytest

from refino.plots.anova import (
    ANOVA,
    KRUSKAL_WALLIS,
    ROTULO_DA_SERIE,
    ROTULO_DE_X,
    ROTULO_DE_Y,
    TITULO,
    AnovaDesalinhadaError,
    Classe,
    ContextoDaAnova,
    GrupoDaAnova,
    PontosDaAnova,
    contexto_do_fio,
    figura_da_anova,
    frase_da_populacao,
    frase_do_post_hoc,
    frase_do_vazio,
    frase_dos_grupos,
    grupos_do_fio,
    nome_do_teste,
    pontos_da_anova,
    recorte_da_anova,
    rotulo_da_classe,
)
from refino.plots.base import MatrizVaziaError, RecusaDoConstrutorError
from refino.plots.catalogo import CONSTRUTORES, rotulo_do_construtor
from refino.plots.recorte import RecorteInvalidoError, decimal, milhar
from refino.statistics.univariate import benjamini_hochberg
from tests.arnes_de_figura import (
    PlantaQueNaoSeparaError,
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

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# O material sintético — e cada valor tem a âncora que o justifica
# ---------------------------------------------------------------------------
#
# **Seis features, e cada linha existe para um desfecho.** Duas sobrevivem ao FDR, três não, e a
# sexta é a forma de indefinido que tira a feature do desenho.
#
# **`PVALUE` e `QVALUE` DIFEREM em toda posição** — é a âncora do par causal que troca um pelo
# outro. Sem ela a troca seria a identidade e o teste passaria com a planta aplicada.
NOMES: Final = (
    "1.00_100.0m/z",  # q abaixo do corte  -> SOBREVIVE
    "2.00_200.0m/z",  # q abaixo do corte  -> SOBREVIVE
    "3.00_300.0m/z",  # q acima do corte   -> NAO
    "4.00_400.0m/z",  # q acima do corte   -> NAO
    "5.00_500.0m/z",  # q acima do corte   -> NAO
    "6.00_600.0m/z",  # q indefinido       -> fora do desenho
)
PVALUE: Final = (0.0001, 0.001, 0.30, 0.80, 0.05, None)
QVALUE: Final = (0.01, 0.04, 0.42, 0.88, 0.11, None)

ALFA: Final = 0.05
N_TESTADAS: Final = len(NOMES)

# Os quatro grupos clínicos deste estudo com os `n` reais do `pos_hilic`. **Literais aqui de
# propósito:** são a fonte independente da classe (d), e o construtor recebe-os por um caminho que
# não passa por estes nomes. **E os quatro `n` DIFEREM entre si**, senão uma troca de posição entre
# nome e contagem não separaria nada.
GRUPOS: Final = ("ATLL", "Asymptomatic", "HAMTSP", "NoHTLV")
N_POR_GRUPO: Final = (13, 18, 14, 16)

CORRECAO: Final = "Benjamini-Hochberg (FDR)"
MATRIZ: Final = "processada"
MIN_POR_GRUPO: Final = 2


def _contexto(**troca: Any) -> ContextoDaAnova:
    campos: dict[str, Any] = {
        "grupos": list(GRUPOS),
        "n_por_grupo": list(N_POR_GRUPO),
        "metodo": "anova",
        "post_hoc_metodo": "tukey",
        "post_hoc_rota": "ran",
        "alfa_fdr": ALFA,
        "correcao": CORRECAO,
        "matriz": MATRIZ,
        "min_por_grupo": MIN_POR_GRUPO,
    }
    campos.update(troca)
    return contexto_do_fio(**campos)


def _pontos(**troca: Any) -> PontosDaAnova:
    campos: dict[str, Any] = {
        "features": list(NOMES),
        "pvalue": list(PVALUE),
        "qvalue_fdr": list(QVALUE),
        "n_features_testadas": N_TESTADAS,
        "alfa": ALFA,
    }
    campos.update(troca)
    return pontos_da_anova(**campos)


def _pelo_fio_real(pvalue: list[float]) -> PontosDaAnova:
    """Os pontos com o `q` **derivado** do `p` por `benjamini_hochberg` — o fio da produção.

    ## A lacuna que isto fecha, e ela deixou passar um defeito publicado

    `_pontos` recebe `pvalue` e `qvalue_fdr` de forma **independente**, e isso é conveniência de
    arnês que o fio real **não pode** oferecer: quem produz `AnovaResult.qvalue_fdr` é
    `benjamini_hochberg` sobre o vetor de `p`, e ela devolve `0` **exactamente** onde o `p` é `0` —
    o menor `p` ajustado é `p₍₁₎·n/1`, que é zero quando `p₍₁₎` é zero.

    Consequência: o par `(p = 0, q = 0)` era **inalcançável** por construção neste módulo enquanto
    todo cenário escrevesse os dois vetores à mão, e por isso nenhuma asserção o tocava — enquanto
    `pontos_da_anova` perdia esse exacto caso da contagem de sobreviventes. **O valor
    natural do arnês era o que executava menos do caminho.**

    Ela é o fio e não um atalho: as features chamam-se como as de `NOMES` truncadas ao comprimento
    do `p`, e o resto dos argumentos é o mesmo de `_pontos`.
    """
    q = np.asarray(benjamini_hochberg(np.asarray(pvalue, dtype=float)), dtype=float)
    return pontos_da_anova(
        features=[f"{i + 1}.00_{(i + 1) * 100}.0m/z" for i in range(len(pvalue))],
        pvalue=list(pvalue),
        qvalue_fdr=q.tolist(),
        n_features_testadas=len(pvalue),
        alfa=ALFA,
    )


def _figura(pontos: PontosDaAnova | None = None, contexto: ContextoDaAnova | None = None) -> Figure:
    p = pontos if pontos is not None else _pontos()
    c = contexto if contexto is not None else _contexto()
    return figura_da_anova(pontos=p, contexto=c, recorte=recorte_da_anova(p, c))


def _y(q: float) -> float:
    """O `y` esperado, derivado da ENTRADA e por um caminho que não passa pelo construtor."""
    return float(-np.log10(q))


def _rodape(figura: Figure) -> str:
    """O texto do rodapé **com o espaço em branco colapsado**.

    `recorte.aplica` quebra a frase em linhas pela largura útil, e uma busca de substring sobre o
    texto cru devolve **falso negativo** para qualquer frase que atravesse uma quebra. O gesto é o
    de `scripts/verifica_citacao.py`: `" ".join(texto.split())` dos **dois** lados.
    """
    return " ".join(figura.texts[0].get_text().split())


X_DESENHADOS: Final = (0.0, 1.0, 2.0, 3.0, 4.0)
Y_DESENHADOS: Final = (_y(0.01), _y(0.04), _y(0.42), _y(0.88), _y(0.11))
ROTULO_DO_CORTE_ESPERADO: Final = f"corte do FDR (q = {decimal(ALFA)})"


def _esperado() -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    """As duas séries desenhadas, derivadas dos literais deste módulo."""
    corte = _y(ALFA)
    return {
        ROTULO_DA_SERIE: (X_DESENHADOS, Y_DESENHADOS),
        ROTULO_DO_CORTE_ESPERADO: ((X_DESENHADOS[0], X_DESENHADOS[-1]), (corte, corte)),
    }


# ---------------------------------------------------------------------------
# 1. O caminho feliz — as duas classes, e o ponto de cada uma
# ---------------------------------------------------------------------------


def test_as_duas_classes_sao_as_plantadas_e_o_ponto_de_cada_uma_e_o_que_entrou() -> None:
    """**A planta é o dado: seis features com desfecho conhecido.**

    A âncora vem primeiro: as duas classes têm de estar representadas, senão a asserção sobre a
    partição passaria sobre uma população de uma classe só.
    """
    pontos = _pontos()
    assert set(pontos.classes) == {Classe.SOBREVIVE, Classe.NAO}, (
        f"âncora: as duas classes têm de estar na planta e vieram {set(pontos.classes)}"
    )
    assert pontos.nomes == NOMES[:5]
    assert pontos.x.tolist() == list(X_DESENHADOS)
    assert pontos.y == pytest.approx(np.asarray(Y_DESENHADOS))
    assert pontos.classes == (
        Classe.SOBREVIVE,
        Classe.SOBREVIVE,
        Classe.NAO,
        Classe.NAO,
        Classe.NAO,
    )
    assert pontos.menor_q == pytest.approx(0.01)
    assert pontos.menor_p == pytest.approx(0.0001)


def test_as_DUAS_particoes_somam_a_populacao_da_matriz() -> None:  # noqa: N802
    """Classificação e desenho são DOIS eixos, e cada um fecha sobre `n_da_matriz`.

    Sem estas igualdades uma feature podia desaparecer entre o fio e o desenho sem nada acusar — e
    foi exactamente o que aconteceu com o `q = 0` enquanto houve uma partição só.
    """
    c = _pontos().contagens
    assert c.sobreviventes == 2
    assert c.nao_significativas == 3
    assert c.sem_qvalue == 1
    assert c.sobreviventes + c.nao_significativas + c.sem_qvalue == N_TESTADAS

    assert c.desenhadas == 5
    assert c.nao_desenhadas == 1
    assert c.desenhadas + c.nao_desenhadas == N_TESTADAS

    # E a terceira: as duas classes DENTRO do desenho somam o desenho, que é o que a legenda conta.
    assert c.sobreviventes_desenhados + c.nao_significativas_desenhadas == c.desenhadas


def test_o_p_indefinido_e_contado_e_NAO_tira_a_feature_do_desenho() -> None:  # noqa: N802
    """**O `p` não está em eixo nenhum desta figura, e por isso não decide o desenho.**

    A âncora separa as duas quantidades: a feature 4 tem `p` definido e `q` acima do corte, e a 5
    não tem nenhum dos dois. Uma implementação que filtrasse pelo `p` desenharia quatro pontos.
    """
    pontos = _pontos(pvalue=[0.0001, 0.001, None, None, 0.05, None])
    assert pontos.contagens.sem_pvalue == 3, "âncora: a planta tem de mudar a contagem do `p`"
    assert pontos.contagens.desenhadas == 5, (
        "o `p` indefinido tirou features do desenho — o eixo desta figura é o `q`"
    )
    assert pontos.menor_p == pytest.approx(0.0001)


def test_p_exactamente_zero_e_contado_a_parte_e_nao_e_indefinido() -> None:
    """`p = 0` é o desfecho EXACTO de uma feature constante em cada grupo e diferente entre eles.

    É medida e não ausência dela — `np.isfinite(0.0)` é verdadeiro —, e contá-la como indefinida
    apagaria a distinção que `AnovaCounts.pvalue_zero` pagou para separar.
    """
    pontos = _pontos(pvalue=[0.0, 0.001, 0.30, 0.80, 0.05, None])
    assert pontos.contagens.pvalue_zero == 1
    assert pontos.contagens.sem_pvalue == 1, "o zero não pode contar como indefinido"
    assert pontos.menor_p == pytest.approx(0.0)


def test_o_menor_q_e_sobre_a_MATRIZ_e_nao_sobre_as_desenhadas() -> None:  # noqa: N802
    """Rótulo certo sobre a quantidade errada: a feature de menor `q` pode estar fora do desenho.

    **E o caso extremo é o `q = 0`, que é o menor `q` possível e não tem ponto** — ver
    `test_o_qvalue_ZERO_conta_como_sobrevivente_e_a_figura_declara_que_ele_nao_tem_ponto`. Aqui a
    âncora é sobre o `p`, porque o menor `p` **pode** ser de uma feature sem `q`, e o rodapé declara
    a população dos dois.
    """
    pontos = _pontos(pvalue=[0.30, 0.80, 0.42, 0.88, 0.11, 1e-9])
    assert pontos.menor_p == pytest.approx(1e-9), "o mínimo tem de ver a feature fora do desenho"
    assert pontos.contagens.desenhadas == 5, "âncora: a feature de menor `p` NÃO está desenhada"
    metodo = recorte_da_anova(pontos, _contexto()).metodo
    assert f"Sobre as {milhar(N_TESTADAS)} features da matriz" in metodo, metodo
    assert "testadas" not in metodo, (
        "«testadas» é rótulo errado sobre este número: ele é `len(resultado.feature)`, e a ANOVA "
        f"não testa feature com grupo insuficiente nem constante — {metodo}"
    )
    assert f"o menor p bruto é {decimal(1e-9)}" in metodo, metodo


def test_sem_q_finito_a_figura_DECLARA_em_vez_de_ficar_em_branco() -> None:  # noqa: N802
    """R4b: gráfico vazio sem legenda é indistinguível de gráfico quebrado.

    **É a segunda das três formas de vazio** — vieram features e nenhuma é desenhável —,
    e ela sai como figura com texto, não como recusa.
    """
    pontos = _pontos(qvalue_fdr=[None] * N_TESTADAS)
    assert pontos.contagens.desenhadas == 0
    assert pontos.menor_q is None, "nenhum q finito é fato diferente de «o menor é 1,0»"
    figura = _figura(pontos)
    eixo = figura.axes[0]
    assert series(eixo) == (), "nenhum ponto e nenhuma linha de corte: não há x para a desenhar"
    assert frase_do_vazio(pontos) in eixo.texts[0].get_text()
    assert eixo.get_xticks().size == 0 and eixo.get_yticks().size == 0, (
        "sem ponto nenhum as marcas de origem afirmam um intervalo que ninguém mediu"
    )
    assert "nenhum q é definido" in _rodape(figura)


def test_com_pontos_e_ZERO_sobreviventes_a_nuvem_INTEIRA_e_desenhada() -> None:  # noqa: N802
    """**O desfecho NORMAL deste corpus, e ele é RESULTADO e não estado vazio.**

    A figura sai com a contagem, o menor `q`, o `n` de cada grupo e a correção aplicada — os quatro
    itens que R8 exige, e que são o que separa *"não há efeito"* de *"não há poder"*.
    """
    pontos = _pontos(qvalue_fdr=[0.42, 0.55, 0.60, 0.88, 0.11, None])
    assert pontos.contagens.sobreviventes == 0
    assert pontos.contagens.desenhadas == 5, "a nuvem inteira continua desenhada"
    figura = _figura(pontos)

    titulo = " ".join(figura.axes[0].get_title().split())
    assert f"0 de {milhar(N_TESTADAS)} sobrevive(m) ao FDR" in titulo, titulo
    legenda = [t.get_text() for t in figura.axes[0].get_legend().get_texts()]
    assert rotulo_da_classe(Classe.SOBREVIVE, 0, ALFA) in legenda, legenda

    rodape = _rodape(figura)
    for nome, n in zip(GRUPOS, N_POR_GRUPO, strict=True):
        assert f"{nome} (n = {milhar(n)})" in rodape, rodape
    assert CORRECAO in rodape
    assert (
        f"Sobre as {milhar(N_TESTADAS)} features da matriz, o menor q é {decimal(0.11)}" in rodape
    ), rodape
    assert f"0 de {milhar(N_TESTADAS)} sobrevive(m)" in rodape, rodape
    assert "sem ponto na nuvem" not in rodape, (
        "âncora: sem sobrevivente escondido a cláusula NÃO sai — a ausência dela é a informação"
    )


def test_a_contagem_da_legenda_e_o_numero_de_pontos_ACIMA_do_corte() -> None:  # noqa: N802
    """**O corte é uma ALTURA nesta figura, e é isso que esta igualdade prova.**

    O `y` é `-log10(q)` e o corte é `-log10(alfa)`: `q < alfa` é exactamente *"acima desta linha"*.
    A contagem é derivada do `y` DESENHADO — lido do artista —, e não do objeto que o construtor
    usou; o número da legenda é lido do texto. Se as duas discordassem, a legenda descreveria outro
    corte.
    """
    figura = _figura()
    eixo = figura.axes[0]
    nuvem = next(s for s in series(eixo) if s.rotulo == ROTULO_DA_SERIE)
    corte = _y(ALFA)
    acima = int((nuvem.y > corte).sum())
    assert acima > 0, "âncora: sem ponto acima do corte a igualdade passaria por vacuidade"
    assert acima < nuvem.y.size, "âncora: com todos acima ela também passaria sem separar"

    legenda = [t.get_text() for t in eixo.get_legend().get_texts()]
    assert rotulo_da_classe(Classe.SOBREVIVE, acima, ALFA) in legenda, legenda


# ---------------------------------------------------------------------------
# 1b. O `q = 0` — CLASSIFICAR não é DESENHAR, e o par causal da planta
# ---------------------------------------------------------------------------
#
# **O defeito que estes quatro fecham tinha número.** `pontos_da_anova` derivava `sobreviventes` das
# classes, e as classes só existiam para as **desenháveis**: `-log10(0)` é `+inf`, então a feature
# que separa os grupos perfeitamente saía do desenho E da contagem. A figura publicava *«0 de 4
# sobrevive(m)»* ao lado de *«o menor q é 0»*, com `AnovaCounts.survivors` a dizer **1**.


def test_o_qvalue_ZERO_conta_como_sobrevivente_e_a_figura_declara_que_ele_nao_tem_ponto() -> None:  # noqa: N802
    """**A PLANTA: uma feature com `q = 0`, pelo fio real.**

    É o par `(p = 0, q = 0)` que `benjamini_hochberg` produz e que o arnês de vetores independentes
    não conseguia alcançar. As três âncoras de não-vacuidade vêm antes do desfecho:

    1. o `q` derivado tem de ter um zero **exacto** — senão não há planta;
    2. ele tem de ficar **fora** do desenho — senão a classificação e o desenho não divergem, e a
       asserção passaria sem separar nada;
    3. tem de haver ponto desenhado — senão *"1 sobrevivente"* passaria sobre uma nuvem vazia.
    """
    pontos = _pelo_fio_real([0.0, 0.4, 0.6, 0.9])
    c = pontos.contagens

    assert pontos.menor_q == 0.0, "âncora 1: o fio real tem de produzir q = 0 exacto"
    assert c.qvalue_zero == 1
    assert c.desenhadas == 3 and c.nao_desenhadas == 1, (
        f"âncora 2: a feature de q = 0 tem de ficar fora do desenho — {c}"
    )
    assert c.desenhadas > 0, "âncora 3: sem ponto desenhado a contagem seria sobre nuvem vazia"

    # ---- O DESFECHO: ela conta, e a contagem diz sobre que população ----
    assert c.sobreviventes == 1, (
        f"a feature de q = 0 é o sobrevivente mais forte que existe e saiu da contagem — {c}"
    )
    assert c.sobreviventes_desenhados == 0, (
        "e ela não tem ponto: o desenho e a classificação diferem"
    )
    assert c.sobreviventes_sem_ponto == 1
    assert c.sem_qvalue == 0, "`q = 0` é MEDIDA e não ausência dela — é a irmã de `pvalue_zero`"
    assert c.nao_significativas == 3
    assert c.sobreviventes + c.nao_significativas + c.sem_qvalue == 4


def test_a_figura_do_qvalue_ZERO_diz_a_verdade_no_titulo_no_rodape_e_na_legenda() -> None:  # noqa: N802
    """R6 nas três superfícies: o número que muda tem de aparecer onde alguém o lê.

    **O rodapé é a única coisa da figura que outra pessoa pode desmentir**, e é onde a causa vai
    nomeada: `q` exactamente zero é separação exacta, não ausência de medida.
    """
    pontos = _pelo_fio_real([0.0, 0.4, 0.6, 0.9])
    contexto = _contexto()
    figura = figura_da_anova(
        pontos=pontos, contexto=contexto, recorte=recorte_da_anova(pontos, contexto)
    )

    titulo = " ".join(figura.axes[0].get_title().split())
    assert "3 ponto(s) de 4 features da matriz" in titulo, titulo
    assert "1 de 4 sobrevive(m) ao FDR" in titulo, titulo
    assert "1 sobrevivente(s) sem ponto" in titulo, titulo
    assert "1 com q exactamente zero" in titulo, titulo

    rodape = _rodape(figura)
    assert "1 de 4 sobrevive(m)" in rodape, rodape
    assert "1 não tem ponto na nuvem" in rodape, rodape
    assert "1 com q exactamente zero" in rodape, rodape

    # A legenda conta a NUVEM, e declara a distância para a população na própria entrada.
    legenda = [t.get_text() for t in figura.axes[0].get_legend().get_texts()]
    assert rotulo_da_classe(Classe.SOBREVIVE, 0, ALFA, sem_ponto=1) in legenda, legenda
    assert "0 desenhada(s), mais 1 sem ponto" in " | ".join(legenda), legenda
    # E o `[0]` continua a ser contável na imagem: nenhum ponto está acima do corte.
    nuvem = next(s for s in series(figura.axes[0]) if s.rotulo == ROTULO_DA_SERIE)
    assert int((nuvem.y > _y(ALFA)).sum()) == 0, "a legenda diz 0 desenhados e a nuvem tem de bater"


def test_PAR_CAUSAL_com_a_planta_do_q_zero_REMOVIDA_a_figura_deixa_de_acusar() -> None:  # noqa: N802
    """**A outra ponta: sem `p = 0` nenhuma cláusula nova sai, e as duas partições coincidem.**

    Duas execuções idênticas que diferem só pela planta. Sem esta metade, as asserções acima seriam
    satisfeitas por uma figura que escreve a cláusula **sempre** — o mecanismo
    que nunca deixa de acusar.
    """
    com = _pelo_fio_real([0.0, 0.4, 0.6, 0.9])
    sem = _pelo_fio_real([0.001, 0.4, 0.6, 0.9])

    assert sem.menor_q is not None and sem.menor_q > 0.0, (
        f"âncora: a planta removida não pode continuar a produzir q = 0 — {sem.menor_q}"
    )
    assert sem.contagens.qvalue_zero == 0
    assert sem.contagens.sobreviventes_sem_ponto == 0
    assert sem.contagens.desenhadas == 4 and sem.contagens.nao_desenhadas == 0
    assert sem.contagens.sobreviventes == sem.contagens.sobreviventes_desenhados, (
        "sem a planta as duas partições coincidem, e é isso que torna a divergência atribuível"
    )

    contexto = _contexto()
    limpa = figura_da_anova(pontos=sem, contexto=contexto, recorte=recorte_da_anova(sem, contexto))
    titulo = " ".join(limpa.axes[0].get_title().split())
    rodape = _rodape(limpa)
    for frase in ("sobrevivente(s) sem ponto", "com q exactamente zero"):
        assert frase not in titulo, f"a cláusula saiu sem a planta — {titulo}"
        assert frase not in rodape, f"a cláusula saiu sem a planta — {rodape}"
    assert "não tem ponto na nuvem" not in rodape, rodape
    assert "sem ponto]" not in " | ".join(
        t.get_text() for t in limpa.axes[0].get_legend().get_texts()
    )

    # E o desfecho DIVERGE entre as duas execuções — é isso que faz disto um par causal.
    assert com.contagens.sobreviventes_sem_ponto != sem.contagens.sobreviventes_sem_ponto
    assert com.contagens.desenhadas != sem.contagens.desenhadas


def test_a_contagem_da_figura_BATE_com_AnovaCounts_survivors_do_nucleo() -> None:  # noqa: N802
    """**O alvo certo é o número do NÚCLEO, e não a coerência interna da figura.**

    `statistics.anova` decide quem vai ao post-hoc com `isfinite(q) & (q < fdr_alpha)` —
    estrito —, e
    grava a contagem em `AnovaCounts.survivors`. A figura tem de chegar ao mesmo número a partir dos
    **vetores**, que é tudo o que atravessa o fio (`ResumoDaAnova` declara que as contagens não
    viajam). Enquanto ninguém cruzou os dois, a figura dizia `0` e o núcleo dizia `1`.

    **A planta é a feature constante dentro de cada grupo e diferente entre eles** — `F = +inf`,
    `p = 0`, `q = 0` —, e a âncora é que o núcleo a conte como sobrevivente: sobre um dado em que
    `survivors` fosse zero a igualdade passaria por vacuidade.
    """
    from refino.domain.matrix import RefinoMatrix
    from refino.domain.metadata import SampleMetadata
    from refino.statistics.anova import one_way_anova

    # Uma feature com separação exacta, e uma segunda que não separa nada — a segunda existe para
    # o vetor ter mais de um membro e o BH ter denominador.
    valores = np.array(
        [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [9.0, 1.0], [9.0, 2.0], [9.0, 3.0]], dtype=float
    )
    grupos = ["A", "A", "A", "B", "B", "B"]
    matriz = RefinoMatrix(
        values=valores,
        feature_names=("exacta", "nula"),
        samples=tuple(
            SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type="sample", group=g)
            for i, g in enumerate(grupos)
        ),
        mode="sintetico",
    )
    resultado = one_way_anova(matriz)

    assert resultado.pvalue[0] == 0.0, "âncora: a planta tem de dar p = 0 exacto"
    assert resultado.qvalue_fdr[0] == 0.0, "âncora: o BH leva o p = 0 a q = 0"
    assert resultado.counts.survivors == 1, (
        f"âncora: o núcleo tem de contar a feature exacta — {resultado.counts}"
    )

    pontos = pontos_da_anova(
        features=list(resultado.feature),
        pvalue=[None if np.isnan(v) else float(v) for v in resultado.pvalue],
        qvalue_fdr=[None if np.isnan(v) else float(v) for v in resultado.qvalue_fdr],
        n_features_testadas=len(resultado.feature),
        alfa=resultado.fdr_alpha,
    )
    assert pontos.contagens.sobreviventes == resultado.counts.survivors, (
        f"a figura conta {pontos.contagens.sobreviventes} sobreviventes e o núcleo "
        f"{resultado.counts.survivors}: são o mesmo predicado sobre o mesmo vetor"
    )
    assert pontos.contagens.pvalue_zero == resultado.counts.pvalue_zero
    assert pontos.contagens.sem_qvalue == resultado.counts.qvalue_undefined
    # E a divergência entre desenho e classificação está declarada, não apagada.
    assert pontos.contagens.desenhadas == 1 and pontos.contagens.sobreviventes_sem_ponto == 1


# ---------------------------------------------------------------------------
# 2. Os DOIS artistas, e só um é de dado
# ---------------------------------------------------------------------------


def test_ha_um_artista_de_DADO_e_a_linha_do_corte_e_marcada_a_parte() -> None:  # noqa: N802
    """`PainelDeFigura._identifica` indexa o vetor de nomes pela posição dentro do artista marcado.

    Sem `GID_DAS_FEATURES` na dispersão, um clique na linha do corte devolveria o nome da primeira
    feature — com a figura perfeitamente bem formada. É o defeito que a marca existe para fechar.
    """
    from refino.plots.base import GID_DAS_FEATURES

    eixo = _figura().axes[0]
    assert len(eixo.collections) == 1, [c.get_label() for c in eixo.collections]
    assert len(eixo.lines) == 1, [ln.get_label() for ln in eixo.lines]
    assert eixo.collections[0].get_gid() == GID_DAS_FEATURES
    assert eixo.lines[0].get_gid() != GID_DAS_FEATURES
    assert len(_pontos().nomes) == len(eixo.collections[0].get_offsets())


def test_a_linha_do_corte_esta_em_coordenadas_de_DADO_e_nao_de_eixo() -> None:  # noqa: N802
    """`axhline` põe o `x` em fração do eixo, e o arnês lê `get_xdata()` sem saber disso.

    Uma linha em `[0, 1]` sobre um eixo que vai a 5.942 lê-se como uma série de dois pontos
    encostada à origem — que é a classe (a) a medir o instrumento em vez do alvo.
    """
    eixo = _figura().axes[0]
    linha = eixo.lines[0]
    assert tuple(linha.get_xdata()) == (X_DESENHADOS[0], X_DESENHADOS[-1])
    assert X_DESENHADOS[-1] > 1.0, "âncora: com o eixo a ir só até 1 a distinção não apareceria"


def test_a_cor_do_corte_NAO_e_a_da_classe_que_sobrevive() -> None:  # noqa: N802
    """**Visto na imagem montada:** sem `color=`, a linha herda a primeira cor do ciclo.

    E a primeira cor do ciclo é a de `Classe.SOBREVIVE`. Com zero sobreviventes — o caso normal
    deste corpus — a única coisa daquela cor na figura passa a ser a linha, e quem olha lê a
    referência como a classe.
    """
    import matplotlib

    from refino.plots.anova import cores_das_classes

    eixo = _figura().axes[0]
    da_linha = matplotlib.colors.to_rgba(eixo.lines[0].get_color())
    da_classe = matplotlib.colors.to_rgba(cores_das_classes()[Classe.SOBREVIVE])
    assert da_linha != da_classe, f"a linha do corte e a classe partilham a cor {da_linha}"


# ---------------------------------------------------------------------------
# 3. As quatro classes, e os pares causais
# ---------------------------------------------------------------------------


def test_a_figura_limpa_passa_nas_quatro_classes_de_ktd12() -> None:
    """As quatro classes de asserção sobre a figura sã. **É a outra ponta dos pares causais.**

    O dado esperado é construído aqui, dos literais do módulo, e não lido de volta do construtor.
    """
    figura = _figura()
    eixo = figura.axes[0]

    assert confere_series(eixo, _esperado()) == []
    assert (
        confere_rotulos(
            eixo,
            x=ROTULO_DE_X,
            y=ROTULO_DE_Y,
            titulo=f"{TITULO}\n{frase_da_populacao(_pontos())}",
            legenda=[
                rotulo_da_classe(Classe.SOBREVIVE, 2, ALFA),
                rotulo_da_classe(Classe.NAO, 3, ALFA),
                ROTULO_DO_CORTE_ESPERADO,
            ],
        )
        == []
    )
    assert confere_vista(eixo) == []
    assert confere_numeros(figura, _numeros_esperados()) == []


def _numeros_esperados() -> list[str]:
    """Todo número que a figura declara, derivado da ENTRADA e não do objeto que o construtor usou.

    **É a classe (d), e a metade que a torna útil é a fonte independente.** Nada aqui lê o
    `Recorte` nem o `PontosDaAnova`: as contagens são contadas dos literais do módulo, e os
    denominadores são os literais.

    `confere_numeros` compara por **pertença** e não por multiplicidade, então a lista é o conjunto
    dos números que aparecem — cada um uma vez.
    """
    sobrevivem = sum(1 for q in QVALUE if q is not None and q < ALFA)
    sem_q = sum(1 for q in QVALUE if q is None)
    sem_p = sum(1 for p in PVALUE if p is None)
    desenhadas = N_TESTADAS - sem_q
    return [
        milhar(desenhadas),  # `N de M features`, e o título
        milhar(N_TESTADAS),  # o denominador, em quatro sítios
        milhar(sem_q),  # o indefinido, no rodapé e no título
        milhar(sem_p),  # o `p` indefinido, no título
        *[milhar(n) for n in N_POR_GRUPO],  # os quatro `n` de grupo
        milhar(MIN_POR_GRUPO),  # o mínimo por grupo, no método
        decimal(ALFA),  # o corte, no método e nas duas entradas de legenda
        milhar(sobrevivem),  # as sobreviventes, no título, no rodapé e na legenda
        milhar(N_TESTADAS - sobrevivem - sem_q),  # as não significativas, na legenda
        decimal(min(q for q in QVALUE if q is not None)),
        decimal(min(p for p in PVALUE if p is not None)),
    ]


def test_planta_eixos_trocados_reprova() -> None:
    eixo = _figura().axes[0]
    mutilada = planta_eixos_trocados(eixo)
    achados = confere_series(eixo, _esperado())
    assert any(mutilada in a for a in achados), achados


def test_planta_sinal_invertido_reprova() -> None:
    eixo = _figura().axes[0]
    mutilada = planta_sinal_invertido(eixo)
    achados = confere_series(eixo, _esperado())
    assert any("eixo y" in a and mutilada in a for a in achados), achados


def test_planta_da_serie_removida_reprova() -> None:
    """**Esta figura tem duas séries, então a planta APLICA-SE** — ao contrário do volcano.

    A segunda série é a linha do corte, e é ela que a planta apaga: `series` devolve as coleções
    antes das linhas, e a planta remove a última.
    """
    eixo = _figura().axes[0]
    apagada = planta_serie_removida(eixo)
    achados = confere_series(eixo, _esperado())
    assert any(apagada in a for a in achados), achados


def test_planta_corte_de_eixo_reprova_e_o_dado_continua_CERTO() -> None:  # noqa: N802
    """**A planta que as outras não alcançam.** Dado nenhum muda e a tela fica em branco."""
    eixo = _figura().axes[0]
    planta_corte_de_eixo(eixo)
    assert confere_series(eixo, _esperado()) == [], (
        "âncora: a planta do corte não pode mexer em dado — se mexeu, não é esta a classe testada"
    )
    achados = confere_vista(eixo)
    assert any("A figura está em branco" in a for a in achados), achados


def test_planta_do_numero_declarado_reprova() -> None:
    """Classe (d): o rodapé é a única coisa da figura que outra pessoa pode desmentir."""
    figura = _figura()
    assert confere_numeros(figura, _numeros_esperados()) == []
    # **999 e não 13, e a escolha é uma âncora:** `13` é o `n` do primeiro grupo, então a planta
    # seria absorvida — o número continuaria a ser esperado e a figura continuaria a declará-lo.
    assert "999" not in _numeros_esperados(), "âncora: a planta tem de trazer número NOVO"
    figura.texts[0].set_text("999 de 999 features")
    achados = confere_numeros(figura, _numeros_esperados())
    assert any("999" in a for a in achados), achados


def test_PAR_CAUSAL_trocar_o_vetor_de_p_pelo_de_q_MUDA_a_figura() -> None:  # noqa: N802
    """**A planta que um construtor que ignorasse o eixo passaria.**

    As classes afirmam que o dado desenhado é o que se pediu; nenhuma afirma **qual dos
    vetores do fio** foi para o eixo. Um construtor que pusesse o `p` em `y` desenharia uma figura
    bem formada, com a legenda a contar pelo `q` e a altura a mostrar outra coisa.

    **A âncora vem primeiro, e é uma divisão:** as duas quantidades têm de divergir sobre
    este dado. Sobre um vetor em que `p == q` — que acontece quando nenhuma feature é ajustada — a
    troca seria a identidade e o teste passaria com a planta aplicada.

    **E o restauro é num `finally`**, porque o objeto plantado é local mas a leitura do desfecho
    tem de acontecer com a planta ainda posta: sem o `finally`, uma reprovação a meio deixaria o
    par por fechar.
    """
    p = [v for v in PVALUE if v is not None]
    q = [v for v in QVALUE if v is not None]
    assert p != q, f"âncora: `p` e `q` não divergem sobre esta planta — {p} contra {q}"

    limpa = _figura().axes[0]
    do_limpo = confere_series(limpa, _esperado())
    assert do_limpo == [], f"âncora: a árvore limpa já diverge — {do_limpo}"

    plantada = _pontos(qvalue_fdr=list(PVALUE), pvalue=list(QVALUE))
    assert plantada.y.size == len(X_DESENHADOS), (
        "âncora: a planta mudou quantos pontos são desenháveis, e a reprovação deixaria de ser "
        "atribuível ao eixo"
    )
    achados = confere_series(_figura(plantada).axes[0], _esperado())
    assert any("eixo y" in a for a in achados), achados


# ---------------------------------------------------------------------------
# 4. As recusas tipadas — as três, e as três descem da base do pacote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vazio",
    [
        {"features": []},
        {"features": None},
        {"features": "não é lista"},
    ],
)
def test_sem_feature_NENHUMA_a_recusa_e_MATRIZ_VAZIA(vazio: dict[str, Any]) -> None:  # noqa: N802
    """População vazia é defeito de quem chama, e não o mesmo fato que «nenhuma é desenhável»."""
    with pytest.raises(MatrizVaziaError, match="não recebeu feature nenhuma"):
        _pontos(**vazio, pvalue=[], qvalue_fdr=[], n_features_testadas=0)


def test_vetores_de_comprimentos_DIFERENTES_produzem_recusa_TIPADA() -> None:  # noqa: N802
    """**Aqui a recusa, e no volcano a contagem — a diferença é o eixo x, e vai declarada.**

    Lá `x` é o `log2fc` e um valor que falta não é desenhável; aqui `x` é o **índice**, que existe
    sempre, e completar com `nan` desenharia posições que nenhuma feature ocupa.

    **A recusa desce de `RecusaDoConstrutorError`, e isso é o requisito e não o tipo exacto:**
    `PainelDeFigura._troca_a_figura` apanha essa base, e qualquer outra excepção dentro de um slot
    do Qt é engolida para o `stderr` com o painel a parar de actualizar sem dizer nada.
    """
    assert issubclass(AnovaDesalinhadaError, RecusaDoConstrutorError)
    with pytest.raises(AnovaDesalinhadaError, match="comprimentos diferentes"):
        _pontos(qvalue_fdr=list(QVALUE[:-1]))
    with pytest.raises(AnovaDesalinhadaError, match="features e trouxe"):
        _pontos(n_features_testadas=N_TESTADAS + 1)


def test_nome_de_grupo_sem_contagem_produz_recusa_TIPADA() -> None:  # noqa: N802
    """Os pares `(nome, n)` não se emparelham pelo que houver.

    **A âncora é a assimetria:** com `zip` sem `strict` o quarto grupo perderia o `n` em silêncio, e
    o rodapé sairia com o `n` de um grupo ao lado do nome de outro — a figura fica bem formada.
    """
    assert len(GRUPOS) != len(N_POR_GRUPO[:-1]), "âncora: a planta tem de desalinhar de facto"
    with pytest.raises(AnovaDesalinhadaError, match="contagem"):
        _contexto(n_por_grupo=list(N_POR_GRUPO[:-1]))


def test_sem_grupo_declarado_o_recorte_recusa_como_MATRIZ_VAZIA() -> None:  # noqa: N802
    """Sem grupos a ANOVA não correu, e isso é «não calculado» e não «calculado e vazio»."""
    with pytest.raises(MatrizVaziaError, match="não declara grupo nenhum"):
        recorte_da_anova(_pontos(), _contexto(grupos=[], n_por_grupo=[]))


def test_o_RODAPE_CHEIO_recusa_em_vez_de_desenhar_por_cima_dos_eixos() -> None:  # noqa: N802
    """**Caminho de erro VIVO, e não hipotético: este é o rodapé mais cheio do pacote.**

    Ele carrega quatro nomes de grupo com os quatro `n`, o teste, o post-hoc com a rota, a correção,
    o alfa, os dois mínimos com a população de cada um e o mínimo por grupo. `recorte.aplica` recusa
    quando nem quebrado ele cabe em `RESERVA_MAXIMA` da altura — e com nomes de grupo longos ele
    chega lá.

    **O par causal está na mesma função:** com os nomes reais deste estudo a figura sai, e é
    essa a metade que prova que a recusa não é a condição normal.
    """
    figura = figura_da_anova(
        pontos=_pontos(), contexto=_contexto(), recorte=recorte_da_anova(_pontos(), _contexto())
    )
    assert figura.texts, "âncora: com os nomes reais o rodapé TEM de caber, senão não há par"

    longos = [f"{nome} {'coorte clínica com nome muito extenso ' * 8}" for nome in GRUPOS]
    contexto = _contexto(grupos=longos)
    recorte = recorte_da_anova(_pontos(), contexto)
    assert len(recorte.texto()) > 2 * len(recorte_da_anova(_pontos(), _contexto()).texto()), (
        "âncora: a planta tem de fazer o rodapé crescer de facto"
    )
    with pytest.raises(RecorteInvalidoError, match="acima do teto"):
        figura_da_anova(pontos=_pontos(), contexto=contexto, recorte=recorte)


# ---------------------------------------------------------------------------
# 5. O que a figura diz sobre si mesma — R3, R5, R5b, R6, R8
# ---------------------------------------------------------------------------


def test_o_recorte_declara_quantas_de_quantas_qual_matriz_e_qual_teste() -> None:
    """R3 e R5b **por extenso**, e os números vêm dos literais deste módulo."""
    recorte = recorte_da_anova(_pontos(), _contexto())
    assert recorte.desenhadas == 5 and recorte.disponiveis == N_TESTADAS
    assert f"5 de {milhar(N_TESTADAS)} features" in recorte.texto()
    assert recorte.matriz == MATRIZ
    assert ANOVA in recorte.metodo
    assert f"mínimo de {MIN_POR_GRUPO} observações utilizáveis por grupo" in recorte.metodo
    assert frase_dos_grupos(_contexto().grupos) in recorte.amostragem


@pytest.mark.parametrize(
    ("metodo", "esperado"),
    [("anova", ANOVA), ("kruskal_wallis", KRUSKAL_WALLIS)],
)
def test_o_nome_do_teste_e_DERIVADO_do_metodo(metodo: str, esperado: str) -> None:  # noqa: N802
    """R5b: uma constante `"ANOVA"` na legenda diria `F` sobre um campo que pode carregar `H`."""
    assert nome_do_teste(metodo) == esperado
    assert esperado in recorte_da_anova(_pontos(), _contexto(metodo=metodo)).metodo


def test_o_metodo_desconhecido_devolve_o_rotulo_CRU_e_nao_cai_num_else() -> None:  # noqa: N802
    """Guarda de fronteira num vocabulário que atravessa o fio: **o terceiro desfecho existe.**"""
    assert "não reconhecido" in nome_do_teste("mann_whitney")
    assert "'mann_whitney'" in nome_do_teste("mann_whitney")


@pytest.mark.parametrize(
    ("rota", "trecho"),
    [
        ("ran", "nas 2 sobrevivente(s)"),
        ("no_survivors", "nenhuma feature sobreviveu ao FDR"),
        ("nothing_testable", "nenhuma feature era testável"),
        ("inventada", "não reconhecida"),
    ],
)
def test_a_frase_do_post_hoc_tem_as_TRES_rotas_mais_o_desconhecido(  # noqa: N802
    rota: str, trecho: str
) -> None:
    """**Quantos desfechos este mecanismo pode reportar, e eu vi todos?**

    Três rotas e o recuo: *"medi todas e nenhuma sobreviveu"* e *"nenhuma era testável"* produzem a
    mesma nuvem baixa e são fatos opostos.
    """
    frase = frase_do_post_hoc(rota, "tukey", 2)
    assert trecho in frase, frase
    assert "tukey" in frase or rota == "inventada", frase


def test_o_post_hoc_sem_metodo_declarado_nao_escreve_nome_em_branco() -> None:
    assert "não declarado" in frase_do_post_hoc("no_survivors", "", 0)


def test_a_frase_dos_grupos_traz_o_n_de_CADA_um_e_nao_o_total() -> None:  # noqa: N802
    """R8, e com quatro grupos é o MENOR deles que decide o poder — um total esconderia isso."""
    frase = frase_dos_grupos(_contexto().grupos)
    for nome, n in zip(GRUPOS, N_POR_GRUPO, strict=True):
        assert f"{nome} (n = {milhar(n)})" in frase, frase
    assert frase_dos_grupos(()) == "nenhum grupo declarado"
    assert "(sem nome)" in frase_dos_grupos((GrupoDaAnova(nome="", n=3),))


def test_o_titulo_conta_o_que_ficou_de_FORA_e_omite_a_clausula_quando_e_zero() -> None:  # noqa: N802
    """A ausência da cláusula **é** a informação, e é a forma de `volcano.frase_da_populacao`."""
    com_indefinido = frase_da_populacao(_pontos())
    assert "1 sem q definido" in com_indefinido and "1 sem p definido" in com_indefinido

    sem_indefinido = frase_da_populacao(
        _pontos(pvalue=[0.1] * N_TESTADAS, qvalue_fdr=[0.2] * N_TESTADAS)
    )
    assert "sem q" not in sem_indefinido and "sem p" not in sem_indefinido, sem_indefinido
    assert "com p exactamente zero" not in sem_indefinido
    assert "com q exactamente zero" not in sem_indefinido
    assert "sem ponto" not in sem_indefinido


def test_as_duas_populacoes_do_titulo_trazem_o_DENOMINADOR_colado() -> None:  # noqa: N802
    """*«N ponto(s)»* e *«M sobrevive(m)»* são populações diferentes com um `de X` cada.

    A âncora é a divergência: sobre esta planta os dois numeradores **diferem** — 5 desenhados e 2
    sobreviventes —, senão a asserção passaria sobre uma frase em que as duas leituras coincidem.
    """
    pontos = _pontos()
    c = pontos.contagens
    assert c.desenhadas != c.sobreviventes, (
        f"âncora: os dois numeradores têm de divergir sobre esta planta — {c}"
    )
    frase = frase_da_populacao(pontos)
    assert f"{milhar(c.desenhadas)} ponto(s) de {milhar(N_TESTADAS)} features da matriz" in frase
    assert f"{milhar(c.sobreviventes)} de {milhar(N_TESTADAS)} sobrevive(m) ao FDR" in frase, frase
    assert "testadas" not in frase, (
        f"«testadas» é o rótulo que a medição de 17/09/2026 derrubou — {frase}"
    )


# ---------------------------------------------------------------------------
# 6. A fronteira: o que chega do fio não passou pelo mypy
# ---------------------------------------------------------------------------


def test_o_contexto_recua_em_cada_campo_ilegivel_sem_levantar() -> None:
    """Campo que chega `null` ou de outro tipo cai no recuo declarado, e a figura não parte."""
    contexto = _contexto(
        metodo=None,
        post_hoc_metodo=123,
        post_hoc_rota=None,
        alfa_fdr="não é número",
        correcao=None,
        matriz="",
        min_por_grupo=True,
    )
    assert contexto.metodo == "" and contexto.post_hoc_metodo == ""
    assert np.isnan(contexto.alfa)
    assert contexto.correcao == "matriz não declarada"
    assert contexto.matriz == "matriz não declarada"
    assert contexto.min_por_grupo == 0, "`True` é um `int` em Python, e não é uma medida"


def test_grupo_ilegivel_vira_VAZIO_e_nao_e_descartado() -> None:  # noqa: N802
    """Descartar mudaria o comprimento e a recusa de desalinhamento deixaria de disparar."""
    grupos = grupos_do_fio(["ATLL", None, 7, "NoHTLV"], [13, 18, 14, 16])
    assert [g.nome for g in grupos] == ["ATLL", "", "", "NoHTLV"]
    assert [g.n for g in grupos] == [13, 18, 14, 16]


# ---------------------------------------------------------------------------
# 7. O registo — a entrada nova, e o rótulo do combo
# ---------------------------------------------------------------------------


def test_a_entrada_do_catalogo_existe_e_o_rotulo_do_combo_e_UNICO() -> None:  # noqa: N802
    """**A bijeção de R1 corre em `test_plots_arnes.py`; o que se afirma aqui é o rótulo.**

    Foram medidos **8 rótulos distintos para 13 entradas** antes de o rótulo passar a sair da
    função, e o portão da unicidade corre sobre o registo inteiro. Aqui a asserção é local: a
    entrada da ANOVA existe, com a chave verbatim do `[G-1]`, e o rótulo dela não colide com nenhum
    outro.
    """
    daqui = [c for c in CONSTRUTORES if c.modulo == "anova"]
    assert len(daqui) == 1, daqui
    assert daqui[0].funcao == "figura_da_anova"
    assert daqui[0].chave == "One-way ANOVA"

    rotulos = [rotulo_do_construtor(c.funcao) for c in CONSTRUTORES]
    assert rotulo_do_construtor(daqui[0].funcao) == "anova"
    assert len(rotulos) == len(set(rotulos)), (
        f"rótulos duplicados no seletor: {sorted({r for r in rotulos if rotulos.count(r) > 1})}"
    )


def test_a_planta_do_arnes_recusa_dado_em_que_ela_nao_separa() -> None:
    """Os negativos do arnês são eles próprios mecanismos."""
    pontos = _pontos(qvalue_fdr=[ALFA] * N_TESTADAS)
    eixo = _figura(pontos).axes[0]
    with pytest.raises(PlantaQueNaoSeparaError, match="constante"):
        planta_corte_de_eixo(eixo)


# ---------------------------------------------------------------------------
# 8. O CORPUS REAL — o `pos_hilic` com os quatro grupos, e as âncoras na mesma função
# ---------------------------------------------------------------------------

MODO_REAL: Final = "pos_hilic"

CAMPOS_DA_CASCATA_NULA: Final[dict[str, Any]] = {
    "correction": "none",
    "n_batches": 1,
    "rsd_cutoff": None,
    "ranking_filter": "none",
    "ranking_keep_fraction": None,
    "ranking_filter_2": "none",
    "ranking_keep_fraction_2": None,
    "rsd_variant": "plain",
    "missing_threshold": None,
    "imputation": "none",
    "sample_normalization": "none",
    "normalization_reference": None,
    "normalization_factors": None,
    "transformation": "none",
    "scaling": "none",
}


@pytest.mark.slow
def test_o_pos_hilic_real_atravessa_a_rota_ate_a_figura_SEM_SOBREVIVENTES() -> None:  # noqa: N802
    """**Ponta a ponta pelo caminho da ROTA, e a figura é vazia — é o resultado (regra 1 do
    README).**

    Zero features sobrevivem ao FDR num estudo de quatro grupos de 13 a 18 amostras depois da
    correção de múltiplos testes. Isso é o que este estudo diz, não um defeito do desenho.

    ## As TRÊS âncoras de não-vacuidade vivem NESTA função

    *"Zero sobreviventes"* é satisfeito por um array vazio, por um vetor todo `nan`, e por um laço
    que nunca corre:

    1. **a partição soma a população testada** — todo ponto desenhado mais todo indefinido dá o
       número de features da matriz processada, derivado da matriz e não do objeto que o construtor
       usou. Um array vazio dá `0 != 5.942`;
    2. **o menor `q` declarado é o que o teste mediu**, derivado de `AnovaResult.qvalue_fdr` por
       caminho independente do construtor. Um vetor todo `nan` dá `None` e não bate;
    3. **os quatro grupos são os da matriz**, e o QC não está entre eles.

    ## O caminho é o da ROTA e não uma chamada directa a `one_way_anova`

    `corre_as_analises` é quem a produção corre, e é ela que decide **se** a família corre e com que
    post-hoc. Chamar o núcleo directamente mediria o alvo certo por um caminho que a produção não
    usa, com o desfecho de sucesso.

    **As contagens são observação sobre ESTE arquivo, nomeadas, nunca invariante.**
    """
    from refino.api.rotas import corre_as_analises
    from refino.domain.analysis import AnalysisSpec
    from refino.domain.spec import PipelineSpec
    from refino.io.readers import load_matrix
    from refino.preprocessing.pipeline import process_matrix_detailed
    from refino.statistics.anova import one_way_anova
    from tests.conftest import measurements_path, sample_list_path

    matriz = load_matrix(
        measurements_path(MODO_REAL),
        sample_list_path(MODO_REAL),
        MODO_REAL,
        orientation="features-in-rows",
    )
    # **A máscara de imputadas vem do MESMO executor, como na rota** — `corre_as_analises` exige-a
    # desde a normalidade, e a matriz e a máscara saem da mesma chamada.
    detalhada = process_matrix_detailed(matriz, PipelineSpec(**CAMPOS_DA_CASCATA_NULA))
    processada = detalhada.matrix
    resultados = corre_as_analises(
        AnalysisSpec(),
        matriz=processada,
        imputadas=detalhada.imputed_cells,
        seed=42,
        marca=lambda: None,
        cancelado=lambda: False,
    )
    assert resultados.anova is not None, "a família vem ligada por padrão (KD11) e não correu"
    bruto = resultados.anova.model_dump()

    contexto = contexto_do_fio(
        grupos=bruto["grupos"],
        n_por_grupo=bruto["n_por_grupo"],
        metodo=bruto["metodo"],
        post_hoc_metodo=bruto["post_hoc_metodo"],
        post_hoc_rota=bruto["post_hoc_rota"],
        alfa_fdr=bruto["alfa_fdr"],
        correcao=bruto["correcao"],
        matriz=bruto["matriz"],
        min_por_grupo=bruto["min_por_grupo"],
    )
    pontos = pontos_da_anova(
        features=bruto["features"],
        pvalue=bruto["pvalue"],
        qvalue_fdr=bruto["qvalue_fdr"],
        n_features_testadas=bruto["n_features_testadas"],
        alfa=contexto.alfa,
    )
    figura = figura_da_anova(
        pontos=pontos, contexto=contexto, recorte=recorte_da_anova(pontos, contexto)
    )

    # ---- ÂNCORA 1: a partição soma a população testada, e ela é a da matriz processada ----
    da_matriz = processada.n_features
    assert da_matriz > 0, "âncora: a matriz processada ficou sem feature nenhuma"
    c = pontos.contagens
    assert pontos.n_da_matriz == da_matriz, (
        f"a figura declara {pontos.n_da_matriz} features e a matriz processada tem {da_matriz}"
    )
    assert c.desenhadas > 0, (
        "âncora: nenhum ponto foi desenhado — «zero sobreviventes» passaria sobre um laço que "
        "nunca correu"
    )
    assert c.sobreviventes + c.nao_significativas + c.sem_qvalue == da_matriz, (
        f"a partição da classificação não fecha: {c} contra {da_matriz} features da matriz"
    )
    assert c.desenhadas + c.nao_desenhadas == da_matriz, (
        f"a partição do desenho não fecha: {c} contra {da_matriz} features da matriz"
    )
    nuvem = next(s for s in series(figura.axes[0]) if s.rotulo == ROTULO_DA_SERIE)
    assert nuvem.x.size == c.desenhadas

    # ---- ÂNCORA 1b: a contagem da figura é a do NÚCLEO, e o rótulo «testadas» é MEDIDO ----
    #
    # **A chamada directa é obrigatória aqui e não lê a instância errada.** `ResumoDaAnova` declara
    # por extenso que *"as contagens de `AnovaCounts` não viajam"*: `counts` não existe no corpo,
    # então o único caminho para o número do núcleo é recalculá-lo. A ANOVA é determinística e custa
    # ~17 ms com zero sobreviventes, sobre a **mesma** matriz processada que a rota recebeu.
    do_nucleo = one_way_anova(processada, post_hoc=AnalysisSpec().post_hoc).counts
    assert c.sobreviventes == do_nucleo.survivors, (
        f"a figura conta {c.sobreviventes} sobreviventes e o núcleo {do_nucleo.survivors}"
    )
    assert c.sem_qvalue == do_nucleo.qvalue_undefined
    assert c.pvalue_zero == do_nucleo.pvalue_zero
    # **`insufficient_groups` e `constant` medidos a ZERO nos quatro modos em 17/09/2026**, e é por
    # isso que o denominador deste corpus é ao mesmo tempo «features da matriz» e «testadas» — o
    # rótulo largo está certo nos dois casos e o estreito só por acidente. No dia em que deixarem de
    # ser zero, é esta linha que reprova e manda remedir a docstring de `PontosDaAnova`.
    assert (do_nucleo.insufficient_groups, do_nucleo.constant) == (0, 0), (
        f"medido a (0, 0) nos quatro modos em 17/09/2026 e agora é "
        f"({do_nucleo.insufficient_groups}, {do_nucleo.constant}): o denominador deixou de ser "
        "«todas testadas», e a docstring de `PontosDaAnova` tem de ser remedida"
    )

    # ---- ÂNCORA 2: o menor `q` declarado é o MEDIDO, por caminho independente ----
    q = np.asarray([v for v in bruto["qvalue_fdr"] if v is not None], dtype=float)
    assert q.size > 0, "âncora: nenhum `q` é finito — o mínimo seria `None` por vacuidade"
    medido = float(q.min())
    assert pontos.menor_q == pytest.approx(medido)
    assert decimal(medido) in _rodape(figura), _rodape(figura)

    # ---- ÂNCORA 3: os quatro grupos são os da matriz, e o QC fica FORA ----
    de_estudo = sorted(
        {a.group for a, e in zip(processada.samples, processada.sample_mask, strict=True) if e}
    )
    assert list(contexto.grupos and [g.nome for g in contexto.grupos]) == de_estudo, (
        f"a figura declara {[g.nome for g in contexto.grupos]} e a matriz tem {de_estudo}"
    )
    assert "QC" not in de_estudo, "âncora: o QC entrou na máscara de estudo, e a comparação é vácua"

    # ---- E o resultado: ZERO sobrevivem, e o menor `q` está MUITO acima do corte ----
    assert c.sobreviventes == 0, (
        f"{c.sobreviventes} features sobrevivem ao FDR no {MODO_REAL} sob a cascata nula — uma "
        "divergência aqui é achado e não bug"
    )
    assert medido > ALFA, (
        f"o menor q é {medido} e o corte é {ALFA}: com zero sobreviventes o mínimo tem de estar "
        "acima do corte, e se não estiver a classificação e a contagem discordam"
    )
    assert c.sobreviventes_sem_ponto == 0, (
        "com o menor q acima do corte nenhum sobrevivente pode estar escondido; se estiver, a "
        "figura tem de o declarar em vez de o apagar"
    )
    assert f"0 de {milhar(da_matriz)} sobrevive(m) ao FDR" in " ".join(
        figura.axes[0].get_title().split()
    )

    # ---- R8 inteiro na figura: os quatro `n` e a correção, que separam efeito de poder ----
    rodape = _rodape(figura)
    for grupo in de_estudo:
        n = sum(
            1
            for a, e in zip(processada.samples, processada.sample_mask, strict=True)
            if e and a.group == grupo
        )
        assert f"{grupo} (n = {milhar(n)})" in rodape, rodape
    assert bruto["correcao"] in rodape
    assert nome_do_teste(bruto["metodo"]) in rodape

    assert confere_vista(figura.axes[0]) == []

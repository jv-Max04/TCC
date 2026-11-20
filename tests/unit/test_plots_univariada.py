"""As duas figuras univariadas por feature: `Fold Change` e `T-tests`.

## O que este módulo prova, e o que ele deliberadamente NÃO prova

| Bloco | O que prova |
|---|---|
| os dois construtores | o `y` de cada um é o vetor que ele promete, e não o do irmão |
| as DUAS partições | classificação e desenho, cada uma total sobre as testadas, nas duas figuras |
| o sobrevivente sem ponto | ele CONTA, e a figura declara-o no título, no rodapé e na legenda |
| o núcleo | a contagem da figura é a de `rotas.executar` sobre o mesmo `UnivariateResult` |
| as plantas do arnês | as classes de asserção acusam, e cada figura tem as que se aplicam |
| o par causal do desenho | trocar o vetor do eixo muda a figura |
| a raiz partilhada | as três figuras univariadas leem o MESMO contraste, e ele é um só |
| o intervalo nas escolhidas | a barra vertical só nas escolhidas, e a regra e o nível no rodapé |

**O que NÃO se afirma, e vai declarado porque a leitura natural sugeriria o contrário:**

1. **Nenhuma asserção de que estas duas concordem com o VOLCANO ponto a ponto.** O volcano desenha
   `log2fc` contra `-log10(p)` no mesmo par de eixos; estas desenham cada um contra o índice da
   feature. O que se afirma é que as três leem a mesma raiz e o mesmo contraste — a coincidência
   dos valores é consequência disso, e afirmá-la aqui seria medir o construtor do volcano num
   módulo que não é o dele.
2. **Nenhuma asserção sobre o par `(grupo_a, grupo_b)` estar certo.** Quem o produz é
   `rotas._resumo_do_volcano`, e quem o guarda é `tests/unit/test_plots_volcano.py`. Aqui ele é
   entrada.
3. **Nenhuma asserção de geometria.** O rótulo do eixo y do fold change saiu do eixo e foi para o
   título **por causa** de geometria — ele transbordava a figura na vertical —, e nenhuma classe de
   A verificação não vê isso. O que cobre é a leitura da imagem montada.
4. **Nenhuma contagem de `q` exactamente zero, e a ausência é decisão.** Em `plots/anova.py` ela
   existe porque lá o eixo vertical **é** o `-log10(q)` e `q = 0` é a causa do ponto que falta.
   Aqui o `q` não está em eixo nenhum: a causa é *"ficou sem o valor do eixo"*, e é essa que a
   figura nomeia. Uma contagem de `q = 0` seria um correlato a passar por causa.
5. **Nenhuma asserção sobre o CORPUS REAL neste módulo.** Ele corre em
   `tests/unit/test_plots_volcano.py`, que lê a mesma raiz e o mesmo `UnivariateResult` — e onde
   ele mede, em 17/09/2026, `sobreviventes_sem_ponto = 0` nas três figuras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pytest

from refino.api.rotas import FDR_ALPHA
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.plots.base import MatrizVaziaError
from refino.plots.catalogo import CONSTRUTORES, rotulo_do_construtor
from refino.plots.recorte import decimal, milhar
from refino.plots.univariada import (
    FOLD_CHANGE,
    LOG2FC,
    NEG_LOG10_P,
    PVALOR,
    ROTULO_DA_SERIE,
    ROTULO_DE_X,
    ROTULO_DO_ZERO,
    TITULO_DOS_TESTES_T,
    Classe,
    PontosUnivariados,
    figura_do_fold_change,
    figura_dos_testes_t,
    frase_da_populacao,
    frase_do_vazio,
    pontos_do_fold_change,
    pontos_dos_testes_t,
    recorte_do_fold_change,
    recorte_dos_testes_t,
    rotulo_da_classe,
    titulo_do_fold_change,
)
from refino.plots.volcano import (
    ROTULO_DAS_BARRAS,
    Contraste,
    IntervalosEscolhidos,
    contraste_do_fio,
    intervalos_escolhidos,
    rotulo_das_barras,
)
from refino.statistics.univariate import benjamini_hochberg, univariate_test
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
# **Seis features, e as duas figuras ficam sem features DIFERENTES.** A quinta não tem fold change
# e a sexta não tem `p`: é a âncora que impede as duas partições de coincidirem por acaso, e sem
# ela um construtor que lesse o vetor errado passaria nas duas.
#
# **`FC_ENTRADA` e `-log10(PVALUE)` divergem em toda posição desenhável** — é a âncora do par causal
# que troca um pelo outro.
NOMES: Final = (
    "1.00_100.0m/z",
    "2.00_200.0m/z",
    "3.00_300.0m/z",
    "4.00_400.0m/z",
    "5.00_500.0m/z",  # sem fold change
    "6.00_600.0m/z",  # sem p
)
FC_ENTRADA: Final = (-3.0, 2.5, 0.75, 1.25, None, 4.0)
PVALUE: Final = (0.0001, 0.001, 0.002, 0.4, 0.03, None)
QVALUE: Final = (0.001, 0.01, 0.02, 0.5, 0.04, None)

ALFA: Final = 0.05
N_TESTADAS: Final = len(NOMES)

N_A: Final = 18
N_B: Final = 14
GRUPO_A: Final = "Asymptomatic"
GRUPO_B: Final = "HAMTSP"

CORRECAO: Final = "Benjamini-Hochberg (FDR)"
MATRIZ_DO_P: Final = "processada"
MATRIZ_DO_FC: Final = "pré-transformação"


def _contraste(**troca: Any) -> Contraste:
    """O contraste, com a rota `forced` de propósito.

    **`forced` e não `shapiro`, e a escolha é sobre a classe (d):** a prosa de Shapiro traz um
    p-valor e um `n` que teriam de entrar na lista de números esperados sem descreverem nada desta
    figura. A rota forçada não escreve número nenhum, e o que sobra no rodapé é o que a figura
    declara sobre si.
    """
    campos: dict[str, Any] = {
        "grupo_a": GRUPO_A,
        "grupo_b": GRUPO_B,
        "n_grupo_a": N_A,
        "n_grupo_b": N_B,
        "parametrico": True,
        "alfa_fdr": ALFA,
        "correcao": CORRECAO,
        "rota_da_normalidade": "forced",
        "shapiro_pvalue": None,
        "n_residuos": None,
        "matriz_do_pvalue": MATRIZ_DO_P,
        "matriz_do_fold_change": MATRIZ_DO_FC,
        "matrizes_diferem": True,
    }
    campos.update(troca)
    return contraste_do_fio(**campos)


def _fc(**troca: Any) -> PontosUnivariados:
    campos: dict[str, Any] = {
        "features": list(NOMES),
        "log2fc": list(FC_ENTRADA),
        "qvalue_fdr": list(QVALUE),
        "n_features_testadas": N_TESTADAS,
        "alfa": ALFA,
    }
    campos.update(troca)
    return pontos_do_fold_change(**campos)


def _tt(**troca: Any) -> PontosUnivariados:
    campos: dict[str, Any] = {
        "features": list(NOMES),
        "pvalue": list(PVALUE),
        "qvalue_fdr": list(QVALUE),
        "n_features_testadas": N_TESTADAS,
        "alfa": ALFA,
    }
    campos.update(troca)
    return pontos_dos_testes_t(**campos)


def _figura_fc(pontos: PontosUnivariados | None = None) -> Figure:
    p = pontos if pontos is not None else _fc()
    c = _contraste()
    return figura_do_fold_change(pontos=p, contraste=c, recorte=recorte_do_fold_change(p, c))


def _figura_tt(pontos: PontosUnivariados | None = None) -> Figure:
    p = pontos if pontos is not None else _tt()
    c = _contraste()
    return figura_dos_testes_t(pontos=p, contraste=c, recorte=recorte_dos_testes_t(p, c))


def _rodape(figura: Figure) -> str:
    """O rodapé **com o espaço em branco colapsado**."""
    return " ".join(figura.texts[0].get_text().split())


X_FC: Final = (0.0, 1.0, 2.0, 3.0, 5.0)
Y_FC: Final = (-3.0, 2.5, 0.75, 1.25, 4.0)
X_TT: Final = (0.0, 1.0, 2.0, 3.0, 4.0)
Y_TT: Final = tuple(float(-np.log10(p)) for p in PVALUE[:5])


# ---------------------------------------------------------------------------
# 1. Cada figura desenha o SEU vetor, e não o do irmão
# ---------------------------------------------------------------------------


def test_o_eixo_y_de_cada_figura_e_o_vetor_que_ela_promete() -> None:
    """**A âncora vem primeiro e é uma comparação:** os dois vetores têm de divergir.

    Sobre um dado em que `log2fc` e `-log10(p)` coincidissem, um construtor que lesse o vetor
    errado passaria nas duas — e a asserção seria satisfeita com a troca aplicada.
    """
    assert Y_FC != Y_TT, f"âncora: os dois eixos não divergem — {Y_FC} contra {Y_TT}"

    fc, tt = _fc(), _tt()
    assert fc.quantidade == FOLD_CHANGE and tt.quantidade == PVALOR
    assert fc.x.tolist() == list(X_FC)
    assert fc.y == pytest.approx(np.asarray(Y_FC))
    assert tt.x.tolist() == list(X_TT)
    assert tt.y == pytest.approx(np.asarray(Y_TT))


def test_as_DUAS_particoes_somam_a_populacao_testada_nas_duas_figuras() -> None:  # noqa: N802
    """Classificação e desenho são DOIS eixos, e cada um fecha sobre `n_testadas`.

    Sem estas igualdades uma feature podia desaparecer entre o fio e o desenho sem nada acusar — e
    foi exactamente o que aconteceu ao sobrevivente sem valor de eixo enquanto houve uma partição
    só.

    **E as duas populações desenhadas NÃO são a mesma**, que é a âncora: a quinta feature não tem
    fold change e a sexta não tem `p`. Sem essa diferença, uma partição que fechasse por acaso nas
    duas não provaria nada sobre qual vetor cada uma leu.

    **A CLASSIFICAÇÃO é a mesma nas duas, e isso é R8b:** as duas leem o mesmo vetor de `q`, logo
    contam o mesmo número de sobreviventes — mesmo desenhando populações diferentes.
    """
    fc, tt = _fc(), _tt()
    assert fc.nomes != tt.nomes, "âncora: as duas desenham a MESMA população, e a planta é vácua"

    for pontos in (fc, tt):
        c = pontos.contagens
        assert c.sem_valor == 1
        assert c.sobreviventes + c.nao_significativas + c.sem_qvalue == N_TESTADAS, c
        assert c.desenhadas + c.sem_valor == N_TESTADAS, c
        # E a terceira: as duas classes DENTRO do desenho somam o desenho, que é a legenda.
        assert c.sobreviventes_desenhados + c.nao_sobreviventes_desenhados == c.desenhadas, c

    assert fc.contagens.sobreviventes == tt.contagens.sobreviventes == 4, (
        "as duas leem o MESMO vetor de q e têm de contar o mesmo número de sobreviventes (R8b)"
    )
    assert fc.contagens.sem_qvalue == 1 and tt.contagens.sem_qvalue == 1
    # E as duas desenham números DIFERENTES de sobreviventes, que é o que torna a igualdade acima
    # uma afirmação e não uma coincidência: no fold change a quinta feature sobrevive e não tem
    # ponto; nos testes t ela tem.
    assert fc.contagens.sobreviventes_desenhados == 3
    assert tt.contagens.sobreviventes_desenhados == 4
    assert fc.contagens.sobreviventes_sem_ponto == 1
    assert tt.contagens.sobreviventes_sem_ponto == 0


def test_o_q_indefinido_NAO_tira_a_feature_do_desenho_e_conta_na_CLASSIFICACAO() -> None:  # noqa: N802
    """**A feature sem `q` é DESENHADA e é classificada à parte — dois eixos, dois desfechos.**

    Ela continua na nuvem, na cor de `Classe.NAO`, porque o `q` não está em eixo nenhum destas duas
    figuras; e na partição da classificação ela conta em `sem_qvalue`, e não em
    `nao_significativas`, porque *"não há q"* não é *"medi e não passou"*.

    **A consequência é que a contagem da legenda tem de ser `nao_sobreviventes_desenhados` e não
    `nao_significativas`**: a segunda exclui esta feature, e o `[N]` deixaria de ser contável na
    imagem.
    """
    fc = _fc()
    c = fc.contagens
    assert c.sem_qvalue == 1
    assert "6.00_600.0m/z" in fc.nomes, "a feature sem `q` tem de continuar desenhada"
    assert fc.classes[-1] is Classe.NAO
    assert c.nao_sobreviventes_desenhados == 2, c
    assert c.nao_significativas == 1, "a feature sem `q` não conta como «medi e não passou»"
    assert c.nao_sobreviventes_desenhados > c.nao_significativas, (
        "âncora: as duas coincidem neste dado, e a distinção que este teste separa é invisível"
    )
    assert c.nao_sobreviventes_sem_ponto == 0, (
        "o minuendo tem de ser `nao_significativas + sem_qvalue`; com só o primeiro daria negativo"
    )


def test_o_menor_q_e_sobre_as_TESTADAS_e_nao_sobre_as_desenhadas() -> None:  # noqa: N802
    """Rótulo certo sobre a população errada: a feature de menor `q` pode estar fora do desenho.

    **A âncora é a planta:** o menor `q` vai para a feature **sem fold change**, e o rodapé da
    figura de fold change tem de continuar a declará-lo.
    """
    q = [0.5, 0.5, 0.5, 0.5, 1e-9, 0.5]
    pontos = _fc(qvalue_fdr=q)
    assert "5.00_500.0m/z" not in pontos.nomes, "âncora: a feature de menor `q` NÃO está desenhada"
    assert pontos.menor_q == pytest.approx(1e-9)
    assert f"testadas é {decimal(1e-9)}" in recorte_do_fold_change(pontos, _contraste()).metodo


def test_sem_valor_de_eixo_NENHUM_a_figura_DECLARA_em_vez_de_ficar_em_branco() -> None:  # noqa: N802
    """R4b, e a frase nomeia a quantidade — as duas ficam vazias por motivos diferentes."""
    pontos = _fc(log2fc=[None] * N_TESTADAS)
    assert pontos.contagens.desenhadas == 0
    figura = _figura_fc(pontos)
    eixo = figura.axes[0]
    assert series(eixo) == ()
    assert frase_do_vazio(pontos) in eixo.texts[0].get_text()
    assert FOLD_CHANGE in eixo.texts[0].get_text()
    assert eixo.get_xticks().size == 0 and eixo.get_yticks().size == 0


# ---------------------------------------------------------------------------
# 1b. O SOBREVIVENTE SEM PONTO — classificar não é desenhar, e o par causal da planta
# ---------------------------------------------------------------------------
#
# **O defeito que estes quatro fecham tinha número, e foi medido nas duas figuras.** `_pontos`
# derivava `sobreviventes` das classes, e as classes só existiam para as desenháveis. Nos `T-tests`
# isso perdia todo `p = 0` exacto — `-log10(0)` é `+inf` —, e no `Fold Change` perdia todo
# sobrevivente sem `log2fc`. Medido em 17/09/2026 com `p = [0, 0,4, 0,6, 0,9]` pelo fio real: o
# núcleo contava **1** e as duas figuras publicavam **0 sobrevive(m) ao FDR**.


def _q_do_fio(pvalue: list[float | None]) -> list[float | None]:
    """O `q` **derivado** do `p` por `benjamini_hochberg` — o fio da produção, e não dois vetores.

    ## A lacuna que isto fecha, e ela deixou passar o defeito que este bloco conserta

    `_fc` e `_tt` recebem `qvalue_fdr` de forma **independente** do `pvalue`, e isso é conveniência
    de arnês que o fio real **não pode** oferecer: quem produz `UnivariateResult.qvalue_fdr` é
    `benjamini_hochberg` sobre o vetor de `p`, e ela devolve `0` **exactamente** onde o `p` é `0` —
    o menor `p` ajustado é `p₍₁₎·n/1`, que é zero quando `p₍₁₎` é zero.

    Consequência: o par `(p = 0, q = 0)` era **inalcançável** por construção neste módulo enquanto
    todo cenário escrevesse os dois vetores à mão, e por isso nenhuma asserção o tocava. **O
    valor natural do arnês era o que executava menos do caminho.**
    """
    valores = np.asarray([np.nan if v is None else v for v in pvalue], dtype=float)
    q = benjamini_hochberg(valores)
    return [None if not np.isfinite(v) else float(v) for v in q]


_P_DA_PLANTA: Final[list[float | None]] = [0.0, 0.4, 0.6, 0.9]
_P_SEM_PLANTA: Final[list[float | None]] = [0.001, 0.4, 0.6, 0.9]
_NOMES_DA_PLANTA: Final = ("a", "b", "c", "d")


def _tt_do_fio(pvalue: list[float | None]) -> PontosUnivariados:
    return pontos_dos_testes_t(
        features=list(_NOMES_DA_PLANTA),
        pvalue=list(pvalue),
        qvalue_fdr=_q_do_fio(pvalue),
        n_features_testadas=len(pvalue),
        alfa=ALFA,
    )


def test_o_pvalue_ZERO_conta_como_sobrevivente_nos_testes_t_e_NAO_tem_ponto() -> None:  # noqa: N802
    """**A PLANTA: uma feature com `p = 0` exacto, pelo fio real.**

    É o par `(p = 0, q = 0)` que `benjamini_hochberg` produz e que o arnês de vetores independentes
    não conseguia alcançar. As três âncoras de não-vacuidade vêm antes do desfecho:

    1. o `q` derivado tem de ter um zero **exacto** — senão não há planta;
    2. a feature tem de ficar **fora** do desenho — senão a classificação e o desenho não divergem,
       e a asserção passaria sem separar nada;
    3. tem de haver ponto desenhado — senão *"1 sobrevivente"* passaria sobre uma nuvem vazia.
    """
    pontos = _tt_do_fio(_P_DA_PLANTA)
    c = pontos.contagens

    assert pontos.menor_q == 0.0, "âncora 1: o fio real tem de produzir q = 0 exacto"
    assert c.desenhadas == 3 and c.sem_valor == 1, (
        f"âncora 2: a feature de p = 0 tem de ficar fora do desenho — {c}"
    )
    assert c.desenhadas > 0, "âncora 3: sem ponto desenhado a contagem seria sobre nuvem vazia"

    # ---- O DESFECHO: ela conta, e a contagem diz sobre que população ----
    assert c.sobreviventes == 1, (
        f"a feature de p = 0 é o sobrevivente mais forte que existe e saiu da contagem — {c}"
    )
    assert c.sobreviventes_desenhados == 0, "e ela não tem ponto: desenho e classificação diferem"
    assert c.sobreviventes_sem_ponto == 1
    assert c.sem_qvalue == 0, "`q = 0` é MEDIDA e não ausência dela"
    assert c.sobreviventes + c.nao_significativas + c.sem_qvalue == 4


def test_o_sobrevivente_SEM_FOLD_CHANGE_conta_e_NAO_tem_ponto() -> None:  # noqa: N802
    """A mesma classe na outra figura, e aqui quem tira do desenho é o `log2fc` indefinido.

    **São 868 de 5.942 no `pos_hilic`**, no par e na cascata que `plots/univariada.py` declara,
    e qualquer uma delas podia ser a de menor `q`
    — a diferença entre as duas figuras não é hipotética.
    """
    q = [0.001, 0.4, 0.6, 0.9]
    pontos = pontos_do_fold_change(
        features=list(_NOMES_DA_PLANTA),
        log2fc=[None, 2.5, -1.25, 1.75],
        qvalue_fdr=q,
        n_features_testadas=4,
        alfa=ALFA,
    )
    c = pontos.contagens
    assert c.desenhadas == 3 and c.sem_valor == 1, f"âncora: a planta não tirou do desenho — {c}"
    assert c.sobreviventes == 1, f"o sobrevivente sem fold change saiu da contagem — {c}"
    assert c.sobreviventes_desenhados == 0
    assert c.sobreviventes_sem_ponto == 1


def test_a_figura_do_sobrevivente_SEM_PONTO_diz_a_verdade_no_titulo_rodape_e_legenda() -> None:  # noqa: N802
    """R6 nas três superfícies: o número que muda tem de aparecer onde alguém o lê.

    **O rodapé é a única coisa da figura que outra pessoa pode desmentir**, e é onde a causa vai
    nomeada: falta o valor do EIXO, e isso é fato diferente de falta de significância.
    """
    pontos = _tt_do_fio(_P_DA_PLANTA)
    contraste = _contraste()
    figura = figura_dos_testes_t(
        pontos=pontos, contraste=contraste, recorte=recorte_dos_testes_t(pontos, contraste)
    )

    titulo = " ".join(figura.axes[0].get_title().split())
    assert "3 ponto(s) de 4 features testadas" in titulo, titulo
    assert "1 de 4 sobrevive(m) ao FDR" in titulo, titulo
    assert "1 sobrevivente(s) sem ponto" in titulo, titulo

    rodape = _rodape(figura)
    assert "1 de 4 sobrevive(m)" in rodape, rodape
    assert "1 não tem ponto na nuvem" in rodape, rodape
    assert "ausência de valor de EIXO" in rodape, rodape

    legenda = [t.get_text() for t in figura.axes[0].get_legend().get_texts()]
    assert rotulo_da_classe(Classe.SOBREVIVE, 0, ALFA, sem_ponto=1) in legenda, legenda
    assert "0 desenhada(s), mais 1 sem ponto" in " | ".join(legenda), legenda
    # E o `[0]` continua a ser contável na imagem: não há ponto nenhum da classe que sobrevive.
    assert series(figura.axes[0])[0].x.size == 3


def test_PAR_CAUSAL_com_a_planta_do_p_zero_REMOVIDA_a_figura_deixa_de_acusar() -> None:  # noqa: N802
    """**A outra ponta: sem `p = 0` nenhuma cláusula nova sai, e as duas partições coincidem.**

    Duas execuções idênticas que diferem só pela planta. Sem esta metade, as asserções acima seriam
    satisfeitas por uma figura que escreve a cláusula **sempre** — o mecanismo
    que nunca deixa de acusar.
    """
    com = _tt_do_fio(_P_DA_PLANTA)
    sem = _tt_do_fio(_P_SEM_PLANTA)

    assert sem.menor_q is not None and sem.menor_q > 0.0, (
        f"âncora: a planta removida não pode continuar a produzir q = 0 — {sem.menor_q}"
    )
    assert sem.contagens.desenhadas == 4 and sem.contagens.sem_valor == 0
    assert sem.contagens.sobreviventes == sem.contagens.sobreviventes_desenhados, (
        "sem a planta as duas partições coincidem, e é isso que torna a divergência atribuível"
    )
    assert sem.contagens.sobreviventes_sem_ponto == 0

    contraste = _contraste()
    limpa = figura_dos_testes_t(
        pontos=sem, contraste=contraste, recorte=recorte_dos_testes_t(sem, contraste)
    )
    titulo = " ".join(limpa.axes[0].get_title().split())
    rodape = _rodape(limpa)
    # **A âncora de não-vacuidade vem ANTES das ausências:** as três asserções abaixo são de
    # ausência, e ausência é satisfeita por texto vazio.
    assert "1 de 4 sobrevive(m) ao FDR" in titulo, titulo
    assert "1 de 4 sobrevive(m)" in rodape, rodape
    assert "sobrevivente(s) sem ponto" not in titulo, f"a cláusula saiu sem a planta — {titulo}"
    assert "não tem ponto na nuvem" not in rodape, f"a cláusula saiu sem a planta — {rodape}"
    assert "sem ponto]" not in " | ".join(
        t.get_text() for t in limpa.axes[0].get_legend().get_texts()
    )

    # E o desfecho DIVERGE entre as duas execuções — é isso que faz disto um par causal.
    assert com.contagens.sobreviventes_sem_ponto != sem.contagens.sobreviventes_sem_ponto
    assert com.contagens.desenhadas != sem.contagens.desenhadas
    assert com.contagens.sobreviventes == sem.contagens.sobreviventes == 1, (
        "a CLASSIFICAÇÃO não muda com a planta, e é isso que a separa do desenho"
    )


# A matriz da planta do núcleo, e cada coluna existe para um desfecho.
#
# **`exacta`** é constante dentro de cada grupo e diferente entre eles: Welch dá `t = -inf` e
# `p = 0` exacto, e o BH leva o `p = 0` a `q = 0`. O fold change dela é `log2(1/9)`, **finito** —
# ela tem ponto no `Fold Change` e não tem nos `T-tests`.
# **`exacta_sem_fc`** é a mesma coisa com a mediana do primeiro grupo em zero: a razão não é
# positiva, `log2fc` é `nan`, e ela não tem ponto em figura nenhuma.
# **`nula`** não separa nada, e existe para o BH ter denominador e a nuvem ter ponto.
_VALORES_DA_PLANTA: Final = (
    (1.0, 0.0, 1.0),
    (1.0, 0.0, 2.0),
    (1.0, 0.0, 3.0),
    (9.0, 9.0, 1.0),
    (9.0, 9.0, 2.0),
    (9.0, 9.0, 3.0),
)
_GRUPOS_DA_PLANTA: Final = ("A", "A", "A", "B", "B", "B")


def _resultado_do_nucleo() -> Any:
    """`univariate_test` sobre a matriz da planta, com Welch FORÇADO.

    **`parametric=True` e não `None`**, porque a decisão automática corre Shapiro-Wilk sobre seis
    resíduos e a rota escolhida deixaria de ser a que a planta precisa: só Welch devolve `p = 0`
    exacto para a coluna que separa os grupos perfeitamente.
    """
    matriz = RefinoMatrix(
        values=np.asarray(_VALORES_DA_PLANTA, dtype=float),
        feature_names=("exacta", "exacta_sem_fc", "nula"),
        samples=tuple(
            SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type="sample", group=g)
            for i, g in enumerate(_GRUPOS_DA_PLANTA)
        ),
        mode="sintetico",
    )
    return univariate_test(matriz, ("A", "B"), parametric=True)


def _opcionais(valores: np.ndarray) -> list[float | None]:
    """O que o fio faz com `nan`: vira `null`. É o que `rotas._opcionais` produz."""
    return [None if not np.isfinite(v) else float(v) for v in valores]


def test_a_contagem_das_DUAS_figuras_BATE_com_a_do_NUCLEO() -> None:  # noqa: N802
    """**O alvo certo é o número do NÚCLEO, e não a coerência interna da figura.**

    `rotas.executar` grava `n_significant_fdr` de `q[definidos] < FDR_ALPHA`, com
    `definidos = np.isfinite(q)` — e é essa a expressão reproduzida aqui, verbatim, sobre o mesmo
    `UnivariateResult`. As figuras têm de chegar ao mesmo número a partir dos **vetores**, que é
    tudo o que atravessa o fio (`_resumo_do_volcano` declara que as contagens de `UnivariateCounts`
    não viajam). Enquanto ninguém cruzou os dois, as figuras diziam `0` e o núcleo dizia `2`.

    **As âncoras vêm antes do desfecho:** o núcleo tem de contar mais de zero, senão a igualdade
    passa por vacuidade; e as duas figuras têm de desenhar populações **diferentes**, senão uma
    delas não está a ser exercitada.
    """
    resultado = _resultado_do_nucleo()
    q = np.asarray(resultado.qvalue_fdr, dtype=np.float64)
    definidos = np.isfinite(q)
    do_nucleo = int((q[definidos] < FDR_ALPHA).sum())

    assert do_nucleo == 2, (
        f"âncora: o núcleo tem de contar os dois sobreviventes exactos — q = {q.tolist()}"
    )
    assert float(resultado.pvalue[0]) == 0.0, "âncora: a planta tem de dar p = 0 exacto"
    assert not np.isfinite(resultado.log2fc[1]), "âncora: a segunda coluna tem de ficar sem fc"

    fold_change = pontos_do_fold_change(
        features=list(resultado.feature),
        log2fc=_opcionais(resultado.log2fc),
        qvalue_fdr=_opcionais(resultado.qvalue_fdr),
        n_features_testadas=len(resultado.feature),
        alfa=FDR_ALPHA,
    )
    testes_t = pontos_dos_testes_t(
        features=list(resultado.feature),
        pvalue=_opcionais(resultado.pvalue),
        qvalue_fdr=_opcionais(resultado.qvalue_fdr),
        n_features_testadas=len(resultado.feature),
        alfa=FDR_ALPHA,
    )
    assert fold_change.contagens.desenhadas != testes_t.contagens.desenhadas, (
        "âncora: as duas figuras desenham a mesma população, e uma delas não está a ser exercitada"
    )

    for nome, pontos in (("fold change", fold_change), ("testes t", testes_t)):
        assert pontos.contagens.sobreviventes == do_nucleo, (
            f"a figura de {nome} conta {pontos.contagens.sobreviventes} sobreviventes e o núcleo "
            f"{do_nucleo}: são o mesmo predicado sobre o mesmo vetor"
        )
    # E a divergência entre desenho e classificação está DECLARADA, não apagada.
    assert fold_change.contagens.sobreviventes_sem_ponto == 1
    assert testes_t.contagens.sobreviventes_sem_ponto == 2


def test_o_denominador_das_figuras_univariadas_conta_TODA_feature_da_matriz() -> None:  # noqa: N802
    """**O rótulo *«features testadas»* é honesto aqui, ao contrário de `plots/anova.py`.**

    Lá o número é `len(resultado.feature)` e a ANOVA **não testa** feature com grupo abaixo do
    mínimo nem feature constante — `AnovaCounts` tem `insufficient_groups` e `constant` para as
    contar, e a palavra *«testadas»* mentia. `univariate_test` aplica `ttest_ind`/`mannwhitneyu`
    com `axis=0` sobre **todas** as colunas, sem filtro por feature, e `UnivariateCounts` não tem
    nenhuma contagem de não testada — a ausência é a consequência, e não a causa.

    **A planta é a feature CONSTANTE**, que é exactamente a que a ANOVA exclui: ela continua a ter
    linha no resultado, e o denominador conta-a.
    """
    from dataclasses import fields

    resultado = _resultado_do_nucleo()
    assert len(resultado.feature) == len(_VALORES_DA_PLANTA[0]), (
        "uma coluna ficou sem linha no resultado, e o denominador declararia população a mais"
    )
    assert resultado.counts.n_features == len(resultado.feature)
    nomes = {c.name for c in fields(resultado.counts)}
    assert not (nomes & {"constant", "insufficient_groups"}), (
        f"`UnivariateCounts` ganhou contagem de não testada ({nomes}) — a partir daí «testadas» é "
        "rótulo errado sobre este denominador, e o número tem de ser descontado como em "
        "`plots/anova.py`"
    )
    pontos = pontos_do_fold_change(
        features=list(resultado.feature),
        log2fc=_opcionais(resultado.log2fc),
        qvalue_fdr=_opcionais(resultado.qvalue_fdr),
        n_features_testadas=len(resultado.feature),
        alfa=FDR_ALPHA,
    )
    assert pontos.n_testadas == len(resultado.feature)
    assert "features testadas" in frase_da_populacao(pontos)


# ---------------------------------------------------------------------------
# 2. Os artistas — dois no fold change, UM nos testes t
# ---------------------------------------------------------------------------


def test_o_fold_change_tem_a_referencia_do_zero_e_os_testes_t_NAO(  # noqa: N802
) -> None:
    """**A ausência da linha nos testes t é decisão, e não esquecimento.**

    O `q` depende do POSTO do `p` na tabela inteira e não do valor dele: não há altura naquela
    figura que corresponda ao corte, e desenhar uma como se houvesse seria mentira de rótulo — é o
    argumento que `plots/volcano.py` já regista para o mesmo eixo.
    """
    from refino.plots.base import GID_DAS_FEATURES

    fc = _figura_fc().axes[0]
    assert len(fc.collections) == 1 and len(fc.lines) == 1
    assert fc.collections[0].get_gid() == GID_DAS_FEATURES
    assert fc.lines[0].get_gid() != GID_DAS_FEATURES
    assert tuple(fc.lines[0].get_ydata()) == (0.0, 0.0)
    assert tuple(fc.lines[0].get_xdata()) == (X_FC[0], X_FC[-1]), (
        "a referência tem de estar em coordenadas de DADO — `axhline` põe o x em fração do eixo"
    )

    tt = _figura_tt().axes[0]
    assert len(tt.collections) == 1 and len(tt.lines) == 0
    assert tt.collections[0].get_gid() == GID_DAS_FEATURES


def test_a_cor_da_referencia_NAO_e_a_da_classe_que_sobrevive() -> None:  # noqa: N802
    """**Visto na imagem montada:** sem `color=`, a linha herda a primeira cor do ciclo.

    E a primeira cor do ciclo é a de `Classe.SOBREVIVE`. No `pos_hilic` zero features sobrevivem, e
    a única coisa daquela cor na figura passava a ser a linha.
    """
    import matplotlib

    from refino.plots.univariada import cores_das_classes

    eixo = _figura_fc().axes[0]
    da_linha = matplotlib.colors.to_rgba(eixo.lines[0].get_color())
    da_classe = matplotlib.colors.to_rgba(cores_das_classes()[Classe.SOBREVIVE])
    assert da_linha != da_classe, f"a referência e a classe partilham a cor {da_linha}"


# ---------------------------------------------------------------------------
# 3. As quatro classes de asserção do arnês, e os pares causais
# ---------------------------------------------------------------------------


def _esperado_fc() -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    return {
        ROTULO_DA_SERIE: (X_FC, Y_FC),
        ROTULO_DO_ZERO: ((X_FC[0], X_FC[-1]), (0.0, 0.0)),
    }


def _esperado_tt() -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    return {ROTULO_DA_SERIE: (X_TT, Y_TT)}


def test_as_duas_figuras_limpas_passam_nas_quatro_classes_de_ktd12() -> None:
    """As quatro classes sobre as duas figuras sãs. **É a outra ponta dos pares causais.**"""
    fc = _figura_fc()
    assert confere_series(fc.axes[0], _esperado_fc()) == []
    assert (
        confere_rotulos(
            fc.axes[0],
            x=ROTULO_DE_X,
            y=LOG2FC,
            titulo=f"{titulo_do_fold_change(_contraste())}\n{frase_da_populacao(_fc())}",
            legenda=[
                # **A contagem é a da NUVEM, e a distância para a população sai na entrada.** Três
                # sobreviventes têm ponto e o quarto — a feature sem fold change — não tem.
                rotulo_da_classe(Classe.SOBREVIVE, 3, ALFA, sem_ponto=1),
                rotulo_da_classe(Classe.NAO, 2, ALFA),
                ROTULO_DO_ZERO,
            ],
        )
        == []
    )
    assert confere_vista(fc.axes[0]) == []
    assert confere_numeros(fc, _numeros_do_fold_change()) == []

    tt = _figura_tt()
    assert confere_series(tt.axes[0], _esperado_tt()) == []
    assert (
        confere_rotulos(
            tt.axes[0],
            x=ROTULO_DE_X,
            y=NEG_LOG10_P,
            titulo=f"{TITULO_DOS_TESTES_T}\n{frase_da_populacao(_tt())}",
            legenda=[
                rotulo_da_classe(Classe.SOBREVIVE, 4, ALFA),
                # A feature sem `p` também não sobrevive, e não tem ponto: a cláusula sai deste
                # lado nesta figura, e do outro no fold change.
                rotulo_da_classe(Classe.NAO, 1, ALFA, sem_ponto=1),
            ],
        )
        == []
    )
    assert confere_vista(tt.axes[0]) == []
    assert confere_numeros(tt, _numeros_dos_testes_t()) == []


def _sobrevivem() -> int:
    """Os sobreviventes sobre a população TODA — o predicado do núcleo, e nada mais.

    **Nenhuma cláusula sobre o eixo aqui, e a ausência é o conserto.** A versão anterior desta
    derivação trazia `fc is not None` colada, e por isso o teste derivava **o mesmo número errado**
    que o construtor: a classe (d) comparava um defeito consigo próprio.
    """
    return sum(1 for q in QVALUE if q is not None and q < ALFA)


def _numeros_do_fold_change() -> list[str]:
    """Todo número que a figura declara, derivado da ENTRADA e não do objeto do construtor.

    `confere_numeros` compara por **pertença** e não por multiplicidade, então a lista é o conjunto.
    """
    sem_fc = sum(1 for v in FC_ENTRADA if v is None)
    sobrevivem = _sobrevivem()
    desenhados = sum(
        1
        for fc, q in zip(FC_ENTRADA, QVALUE, strict=True)
        if fc is not None and q is not None and q < ALFA
    )
    return [
        milhar(N_TESTADAS - sem_fc),
        milhar(N_TESTADAS),
        milhar(sem_fc),
        milhar(sum(1 for q in QVALUE if q is None)),
        milhar(N_A),
        milhar(N_B),
        decimal(ALFA),
        milhar(sobrevivem),
        milhar(desenhados),
        milhar(sobrevivem - desenhados),
        milhar(N_TESTADAS - sem_fc - desenhados),
        decimal(min(q for q in QVALUE if q is not None)),
        # `log2 fc = 0` na entrada de legenda da referência: o `2` e o `0` são números no texto.
        "2",
        "0",
    ]


def _numeros_dos_testes_t() -> list[str]:
    sem_p = sum(1 for v in PVALUE if v is None)
    sobrevivem = _sobrevivem()
    desenhados = sum(
        1
        for p, q in zip(PVALUE, QVALUE, strict=True)
        if p is not None and q is not None and q < ALFA
    )
    return [
        milhar(N_TESTADAS - sem_p),
        milhar(N_TESTADAS),
        milhar(sem_p),
        milhar(sum(1 for q in QVALUE if q is None)),
        milhar(N_A),
        milhar(N_B),
        decimal(ALFA),
        milhar(sobrevivem),
        milhar(desenhados),
        milhar(N_TESTADAS - sem_p - desenhados),
        decimal(min(q for q in QVALUE if q is not None)),
    ]


def test_planta_eixos_trocados_reprova_nas_duas() -> None:
    for figura, esperado in ((_figura_fc(), _esperado_fc()), (_figura_tt(), _esperado_tt())):
        eixo = figura.axes[0]
        mutilada = planta_eixos_trocados(eixo)
        achados = confere_series(eixo, esperado)
        assert any(mutilada in a for a in achados), achados


def test_planta_sinal_invertido_reprova_nos_testes_t() -> None:
    eixo = _figura_tt().axes[0]
    mutilada = planta_sinal_invertido(eixo)
    achados = confere_series(eixo, _esperado_tt())
    assert any("eixo y" in a and mutilada in a for a in achados), achados


def test_planta_sinal_invertido_RECUSA_no_fold_change_porque_a_referencia_e_ZERO() -> None:  # noqa: N802
    """**O negativo que NÃO se aplica, e a ausência é afirmada em vez de omitida.**

    `planta_sinal_invertido` mutila a **primeira** série, e `series` devolve as linhas antes das
    coleções: a primeira desta figura é a referência do zero. Inverter o sinal de uma série toda a
    zeros é a identidade, e a planta recusa-se em vez de passar por vacuidade — que é exactamente o
    que ela existe para fazer.
    """
    with pytest.raises(PlantaQueNaoSeparaError, match="todo zero"):
        planta_sinal_invertido(_figura_fc().axes[0])


def test_planta_da_serie_removida_reprova_no_fold_change_e_RECUSA_nos_testes_t() -> None:  # noqa: N802
    """As duas direções, e a segunda é a ausência declarada.

    O fold change tem duas séries e a planta aplica-se; os testes t têm **um** artista de dado —
    requisito da identificação de R2 —, e acrescentar um segundo para agradar ao arnês partiria a
    indexação por posição.
    """
    eixo = _figura_fc().axes[0]
    apagada = planta_serie_removida(eixo)
    achados = confere_series(eixo, _esperado_fc())
    assert any(apagada in a for a in achados), achados

    with pytest.raises(PlantaQueNaoSeparaError, match="ao menos duas séries"):
        planta_serie_removida(_figura_tt().axes[0])


def test_planta_corte_de_eixo_reprova_e_o_dado_continua_CERTO() -> None:  # noqa: N802
    for figura, esperado in ((_figura_fc(), _esperado_fc()), (_figura_tt(), _esperado_tt())):
        eixo = figura.axes[0]
        planta_corte_de_eixo(eixo)
        assert confere_series(eixo, esperado) == [], (
            "âncora: a planta do corte não pode mexer em dado"
        )
        achados = confere_vista(eixo)
        assert any("A figura está em branco" in a for a in achados), achados


def test_planta_do_numero_declarado_reprova() -> None:
    figura = _figura_fc()
    assert confere_numeros(figura, _numeros_do_fold_change()) == []
    assert "999" not in _numeros_do_fold_change(), "âncora: a planta tem de trazer número NOVO"
    figura.texts[0].set_text("999 de 999 features")
    achados = confere_numeros(figura, _numeros_do_fold_change())
    assert any("999" in a for a in achados), achados


def test_PAR_CAUSAL_trocar_o_vetor_do_eixo_MUDA_a_figura() -> None:  # noqa: N802
    """**A planta que um construtor que ignorasse o eixo passaria.**

    As classes de asserção do arnês afirmam que o dado desenhado é o que se pediu; nenhuma afirma
    **qual dos vetores do fio** foi para o eixo. Alimentar o construtor do fold change com o vetor
    de `p` produziria uma figura bem formada, com a legenda a contar pelo `q` e a altura a mostrar
    outra coisa.

    **A âncora vem primeiro:** os dois vetores têm de divergir sobre este dado, e o número
    de pontos desenháveis tem de continuar o mesmo — senão a reprovação seria atribuível à
    população e não ao eixo.

    **O restauro é num `finally`** porque a leitura do desfecho acontece com a planta posta.
    """
    assert Y_FC != Y_TT, "âncora: os dois eixos não divergem"
    limpa = _figura_fc().axes[0]
    assert confere_series(limpa, _esperado_fc()) == [], "âncora: a árvore limpa já diverge"

    plantados = [None if v is None else float(v) for v in PVALUE]
    plantados[4], plantados[5] = 0.9, None  # a mesma população desenhável do fold change
    pontos = _fc(log2fc=plantados)
    try:
        assert pontos.contagens.desenhadas == len(X_FC), (
            "âncora: a planta mudou quantos pontos são desenháveis, e a reprovação deixaria de ser "
            "atribuível ao eixo"
        )
        achados = confere_series(_figura_fc(pontos).axes[0], _esperado_fc())
        assert any("eixo y" in a for a in achados), achados
    finally:
        del pontos


# ---------------------------------------------------------------------------
# 4. O que as figuras dizem sobre si mesmas — R3, R5, R5b, R6, R8
# ---------------------------------------------------------------------------


def test_o_recorte_declara_quantas_de_quantas_qual_matriz_e_qual_teste() -> None:
    """R3 e R5b **por extenso**, e as DUAS matrizes são as de cada eixo."""
    fc = recorte_do_fold_change(_fc(), _contraste())
    assert fc.desenhadas == 5 and fc.disponiveis == N_TESTADAS
    assert f"5 de {milhar(N_TESTADAS)} features" in fc.texto()
    assert fc.matriz == MATRIZ_DO_FC, "o eixo do fold change sai da PRÉ-TRANSFORMAÇÃO (KTD8)"
    assert "o eixo vertical é o fold change" in fc.metodo

    tt = recorte_dos_testes_t(_tt(), _contraste())
    assert tt.matriz == MATRIZ_DO_P, "o eixo do `p` sai da PROCESSADA"
    assert "o eixo vertical é o p BRUTO" in tt.metodo
    assert MATRIZ_DO_FC != MATRIZ_DO_P, "âncora: as duas procedências não diferem"


def test_o_rodape_traz_os_DOIS_n_de_grupo_e_a_correcao() -> None:  # noqa: N802
    """R8: é o que separa *"não há efeito"* de *"não há poder"*."""
    for recorte in (
        recorte_do_fold_change(_fc(), _contraste()),
        recorte_dos_testes_t(_tt(), _contraste()),
    ):
        texto = recorte.texto()
        assert f"{GRUPO_A} (n = {milhar(N_A)})" in texto, texto
        assert f"{GRUPO_B} (n = {milhar(N_B)})" in texto, texto
        assert CORRECAO in texto
        assert f"q < {decimal(ALFA)}" in texto


def test_o_rodape_conta_o_indefinido_dos_DOIS_eixos_sem_erro_de_concordancia() -> None:  # noqa: N802
    """**Visto na imagem montada: *«as 1 sem valor de eixo»*.**

    A frase antiga escrevia `as {k} sem valor de eixo são contadas`, e com `k = 1` — que é o que o
    `pos_hilic` produz na figura dos testes t — ela saía com o artigo no plural e o número no
    singular. Uma frase que parece defeituosa põe em dúvida o número ao lado dela, que é a única
    coisa da figura que outra pessoa pode conferir contra fonte independente.
    """
    texto = recorte_dos_testes_t(_tt(), _contraste()).amostragem
    assert "as que ficam sem valor de eixo" in texto, texto
    assert f"1 sem {PVALOR}" in texto and "1 sem q" in texto, texto
    assert "as 1 " not in texto, texto


def test_o_titulo_do_fold_change_NOMEIA_o_numerador() -> None:  # noqa: N802
    """A ordem do par é carga: `(A, B)` e `(B, A)` desenham imagens espelhadas em y.

    **E a frase está no TÍTULO e não no eixo y**, porque na vertical ela transbordava a figura —
    ver a docstring de `titulo_do_fold_change`.
    """
    titulo = titulo_do_fold_change(_contraste())
    assert f"mediana de {GRUPO_A} ÷ mediana de {GRUPO_B}" in titulo
    invertido = titulo_do_fold_change(_contraste(grupo_a=GRUPO_B, grupo_b=GRUPO_A))
    assert invertido != titulo, "trocar a ordem do par tem de mudar a frase"
    assert _figura_fc().axes[0].get_ylabel() == LOG2FC


def test_o_titulo_conta_o_que_ficou_de_FORA_e_omite_a_clausula_quando_e_zero() -> None:  # noqa: N802
    """A ausência da cláusula **é** a informação.

    **E a cláusula do `q` fica FORA do título** — ela vive no rodapé, porque o `q` indefinido tira
    a feature da classe e não do desenho. Com as duas, o título do `pos_hilic` passou a três linhas.
    """
    com = frase_da_populacao(_fc())
    assert f"1 sem {FOLD_CHANGE} desenhável" in com, com
    assert "sem q" not in com, com

    sem = frase_da_populacao(_fc(log2fc=[1.0] * N_TESTADAS, qvalue_fdr=[0.9] * N_TESTADAS))
    assert "sem" not in sem, sem


def test_os_dois_titulos_sao_DISTINTOS_do_da_anova() -> None:  # noqa: N802
    """**Um andar acima do combo:** o rótulo distinguia-as e a IMAGEM não distinguia.

    As três vivem no mesmo seletor da mesma aba, e duas capturas lado a lado seriam indistinguíveis
    pelo texto. O que separa os títulos é a POPULAÇÃO: o par declarado nos testes t, todos os
    grupos da matriz na ANOVA.
    """
    from refino.plots.anova import TITULO as TITULO_DA_ANOVA

    titulos = {
        titulo_do_fold_change(_contraste()),
        TITULO_DOS_TESTES_T,
        TITULO_DA_ANOVA,
    }
    assert len(titulos) == 3, sorted(titulos)


# ---------------------------------------------------------------------------
# 5. A fronteira e o registo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vazio", [{"features": []}, {"features": None}, {"features": 7}])
def test_sem_feature_NENHUMA_a_recusa_e_MATRIZ_VAZIA(vazio: dict[str, Any]) -> None:  # noqa: N802
    with pytest.raises(MatrizVaziaError, match="não recebeu feature nenhuma"):
        _fc(**vazio, log2fc=[], qvalue_fdr=[], n_features_testadas=0)


def test_sem_PAR_DE_GRUPOS_o_recorte_recusa_como_MATRIZ_VAZIA() -> None:  # noqa: N802
    """Sem par não houve teste univariado: é «não calculado» e não «calculado e vazio»."""
    for construtor in (recorte_do_fold_change, recorte_dos_testes_t):
        with pytest.raises(MatrizVaziaError, match="não declara o par de grupos"):
            construtor(_fc(), _contraste(grupo_a=None))


def test_vetor_mais_CURTO_vira_indefinido_e_NAO_levanta() -> None:  # noqa: N802
    """**Aqui a contagem, e em `plots/anova.py` a recusa — a diferença vai declarada.**

    Aqui a população é definida pelos **nomes**, e um valor que falta é exactamente o que R6 manda
    contar e declarar. Levantar trocaria uma figura que diz *"2 sem fold change"* por um painel
    mudo. Lá o eixo x é o índice, e completar desenharia posições que nenhuma feature ocupa.
    """
    pontos = _fc(log2fc=list(FC_ENTRADA[:4]))
    assert pontos.contagens.sem_valor == 2, "a que falta e a que já era `None`"
    assert pontos.n_testadas == N_TESTADAS


def test_o_denominador_DECLARADO_estica_a_populacao_e_nao_a_encolhe() -> None:  # noqa: N802
    """Um `n_features_testadas` maior que os nomes é população a mais, não a menos.

    O recuo é o maior dos dois: derivar do comprimento faria um vetor truncado no fio passar por
    uma população menor sem nada acusar.
    """
    pontos = _fc(n_features_testadas=N_TESTADAS + 3)
    assert pontos.n_testadas == N_TESTADAS + 3
    assert pontos.contagens.sem_valor == 4, "as três que não vieram mais a que já era `None`"


def test_as_entradas_do_catalogo_existem_e_os_rotulos_do_combo_sao_UNICOS() -> None:  # noqa: N802
    """**A bijeção de R1 corre em `test_plots_arnes.py`; o que se afirma aqui é o rótulo.**"""
    daqui = [c for c in CONSTRUTORES if c.modulo == "univariada"]
    assert {c.funcao for c in daqui} == {"figura_do_fold_change", "figura_dos_testes_t"}
    assert {c.chave for c in daqui} == {"Fold Change", "T-tests"}
    assert {rotulo_do_construtor(c.funcao) for c in daqui} == {"fold change", "testes t"}

    rotulos = [rotulo_do_construtor(c.funcao) for c in CONSTRUTORES]
    assert len(rotulos) == len(set(rotulos)), (
        f"rótulos duplicados: {sorted({r for r in rotulos if rotulos.count(r) > 1})}"
    )


# ---------------------------------------------------------------------------
# 6. O intervalo do fold change nas escolhidas
# ---------------------------------------------------------------------------
#
# **`k = 2`: as duas de menor `p` são a 0 e a 1**, e a 2, a 3 e a 5 também têm intervalo — a âncora
# de que a barra não vai a toda feature com intervalo. A seleção e o nível são os de
# `volcano.intervalos_escolhidos`, provados em `test_plots_volcano.py`; aqui prova-se o DESENHO.
INF_FC: Final = (-4.0, 1.5, -0.25, 0.25, None, 3.0)
SUP_FC: Final = (-2.0, 3.5, 1.75, 2.25, None, 5.0)
MOT_FC: Final = (None, None, None, None, "order_statistic_not_positive", None)
K_FC: Final = 2


def _intervalos(**troca: Any) -> Any:
    campos: dict[str, Any] = {
        "pvalue": list(PVALUE),
        "log2fc_ci_lower": list(INF_FC),
        "log2fc_ci_upper": list(SUP_FC),
        "fold_change_ci_undefined": list(MOT_FC),
        "nivel_do_intervalo": 0.95,
        "n_testadas": N_TESTADAS,
        "k": K_FC,
    }
    campos.update(troca)
    return intervalos_escolhidos(**campos)


def _figura_fc_com_barras(intervalos: Any) -> Figure:
    p = _fc()
    c = _contraste()
    return figura_do_fold_change(
        pontos=p,
        contraste=c,
        recorte=recorte_do_fold_change(p, c, intervalos=intervalos),
        intervalos=intervalos,
    )


def test_o_fold_change_poe_a_barra_VERTICAL_so_nas_escolhidas_e_passa_nas_quatro_classes() -> None:  # noqa: N802
    """A barra fica no índice da feature e vai de limite a limite, em `log2`, no nível ajustado."""
    i = _intervalos()
    assert isinstance(i, IntervalosEscolhidos)
    com_intervalo = [j for j, v in enumerate(INF_FC) if v is not None]
    assert len(com_intervalo) > i.com_intervalo > 0, (
        "âncora: toda feature com intervalo seria escolhida, e a seleção não separaria"
    )
    x: list[float] = []
    y: list[float] = []
    for posicao, j in enumerate(i.indices.tolist()):
        x += [float(j), float(j), float("nan")]
        y += [float(i.inferior[posicao]), float(i.superior[posicao]), float("nan")]

    figura = _figura_fc_com_barras(i)
    eixo = figura.axes[0]
    assert confere_series(eixo, {**_esperado_fc(), ROTULO_DAS_BARRAS: (tuple(x), tuple(y))}) == []
    assert confere_vista(eixo) == []
    legenda = [t.get_text() for t in eixo.get_legend().get_texts()]
    assert rotulo_das_barras(i.escolhidas, i.com_intervalo) in legenda, legenda

    rodape = _rodape(figura)
    for trecho in (
        f"nas {i.com_intervalo} das {i.escolhidas} de menor p",
        f"{decimal(i.nivel * 100.0, 5)}%",
        "Benjamini-Yekutieli",
        f"sobre a {MATRIZ_DO_P}",
    ):
        assert trecho in rodape, (trecho, rodape)
    novos = [
        milhar(i.com_intervalo),
        milhar(i.escolhidas),
        decimal(i.nivel * 100.0, 5),
        decimal(1.0 - 0.95),
    ]
    assert confere_numeros(figura, [*_numeros_do_fold_change(), *novos]) == []


def test_a_barra_do_fold_change_entra_ANTES_da_nuvem_e_fica_POR_CIMA() -> None:  # noqa: N802
    """Inserida antes, o clique no ponto responde pela nuvem; desenhada por cima, ela se vê.

    **As duas metades são necessárias aqui, e a segunda é desta figura:** cinco mil pontos no eixo
    do índice cobririam uma barra desenhada por baixo — os vizinhos de abscissa ficam por cima dela.
    """
    eixo = _figura_fc_com_barras(_intervalos()).axes[0]
    filhos = eixo.get_children()
    barras = next(a for a in eixo.lines if a.get_label() == ROTULO_DAS_BARRAS)
    nuvem = next(a for a in eixo.collections if a.get_label() == ROTULO_DA_SERIE)
    assert filhos.index(barras) < filhos.index(nuvem)
    assert barras.get_zorder() > nuvem.get_zorder()


def test_o_fold_change_SEM_intervalo_pedido_e_o_de_antes() -> None:  # noqa: N802
    """`intervalos=None` é a outra ponta do par: sem barra, e o rodapé sem a frase delas."""
    com = _figura_fc_com_barras(_intervalos())
    assert ROTULO_DAS_BARRAS in {s.rotulo for s in series(com.axes[0])}, "âncora: o par não separa"
    assert "Barras" in _rodape(com)
    sem = _figura_fc()
    assert ROTULO_DAS_BARRAS not in {s.rotulo for s in series(sem.axes[0])}
    assert "Barras" not in _rodape(sem)

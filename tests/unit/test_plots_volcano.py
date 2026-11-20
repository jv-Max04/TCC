"""O volcano, e **a figura deste corpus é vazia por construção**.

## O que este módulo prova, e o que ele deliberadamente NÃO prova

| Bloco | O que prova |
|---|---|
| o construtor | os pontos desenhados são os que entraram, e a classe de cada um |
| as DUAS partições | classificação e desenho, cada uma total sobre as testadas |
| o sobrevivente sem ponto | ele CONTA, e a figura declara-o no título e no rodapé |
| o núcleo | a contagem da figura é a de `rotas.executar`, sobre o mesmo `UnivariateResult` |
| as plantas do arnês | as classes de asserção do arnês acusam sobre ESTA figura, e a limpa passa |
| a ordem do par | `(A,B)` e `(B,A)` são espelhadas, e a figura diz qual é o numerador |
| os vazios | os três desfechos distintos, e nenhum fica em branco |
| o corpus real | o `pos_hilic`, com as DUAS âncoras de não-vacuidade na mesma função |
| o intervalo nas escolhidas | as `k` de menor `p`, o nível contra o NÚCLEO, e as barras |

**O que NÃO se afirma, e vai declarado porque a leitura natural sugeriria o contrário:**

1. **Nenhuma asserção de que `0,9976` se reproduza.** Esse número é de `rotas.py` e é o mínimo sobre
   **96 células do multiverso** do `pos_hilic`; o teste do corpus real corre **uma** célula, com a
   cascata em `none`, e mede `0,7791`. São populações diferentes, e usar um pelo outro seria
   rótulo certo sobre a população errada. O que o teste afirma é que o número **declarado na
   figura** é o mínimo que o teste derivou por caminho independente do construtor.
2. **Nenhuma asserção sobre a DIREÇÃO de feature nenhuma no corpus real.** Zero sobrevivem ao FDR:
   não há efeito a afirmar, e afirmá-lo seria fabricar resultado.
3. **A contagem de `sem_pvalue_desenhavel` da figura NÃO é `UnivariateCounts.pvalue_undefined`.** A
   da figura é condicional — conta só as que **têm** fold change —, porque a partição tem de somar a
   população e uma feature sem os dois valores não pode contar duas vezes. No `pos_hilic` a
   diferença é real: o servidor conta 1 e a figura conta 0, porque essa feature também não tem fold
   change. **Quem é comparável é `pvalue_indefinido`**, que é incondicional e nasceu para fechar
   exactamente esta lacuna — o teste cruza os dois e afirma a desigualdade esperada no outro.
4. **`planta_serie_removida` NÃO corre sobre esta figura**, e a ausência é afirmada em vez de
   omitida: há **um** artista de dado por desenho, e a planta do arnês recusa uma figura de uma
   série só. O teste afirma a recusa — que é o desfecho honesto — em vez de a figura ganhar um
   segundo artista para lhe agradar.
5. **Nenhuma asserção de que a LEGENDA declare o sobrevivente sem ponto.** Ela conta a nuvem, e a
   distância vive no título e no rodapé: as três classes significativas desta figura são de
   **direção**, e um sobrevivente sem `log2fc` não tem direção nenhuma. `plots/anova.py` põe a
   cláusula na legenda porque lá as duas classes são pelo `q`, e a atribuição é total.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Final, get_args

import matplotlib
import numpy as np
import pytest

from refino.api.rotas import (
    CORRECAO_DE_MULTIPLOS_TESTES,
    FDR_ALPHA,
    MATRIZ_DO_FOLD_CHANGE,
    MATRIZ_DO_PVALUE,
    _resumo_do_volcano,
)
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.spec import PipelineSpec
from refino.domain.statistics import FoldChangeIntervalUndefined
from refino.io.readers import load_matrix
from refino.plots.base import MatrizVaziaError
from refino.plots.recorte import Recorte, decimal, milhar
from refino.plots.volcano import (
    DIVERGENCIA,
    K_DAS_ESCOLHIDAS,
    MANN_WHITNEY,
    MOTIVOS_SEM_INTERVALO,
    NAO_DECLARADA,
    ROTULO_DA_SERIE,
    ROTULO_DAS_BARRAS,
    ROTULO_DE_Y,
    SEM_MOTIVO_NO_FIO,
    TITULO,
    WELCH,
    Classe,
    Contraste,
    IntervaloAusente,
    IntervalosEscolhidos,
    PontosDoVolcano,
    contraste_do_fio,
    cores_das_classes,
    figura_do_volcano,
    frase_da_populacao,
    frase_do_vazio,
    intervalos_escolhidos,
    nome_do_teste,
    nomes_do_fio,
    pontos_do_volcano,
    prosa_da_normalidade,
    recorte_do_volcano,
    rotulo_da_classe,
    rotulo_das_barras,
    rotulo_de_x,
)
from refino.preprocessing.pipeline import process_matrix_detailed
from refino.statistics import univariate as nucleo_univariado
from refino.statistics.univariate import benjamini_hochberg, fold_change_interval, univariate_test
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
from tests.conftest import measurements_path, sample_list_path

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# O material sintético — e cada valor tem a âncora que o justifica
# ---------------------------------------------------------------------------
#
# **Seis features, e cada linha existe para um desfecho.** As quatro primeiras são desenháveis e
# cobrem as quatro classes; as duas últimas são as duas formas de indefinido.
#
# **`x != y` em todas** — âncora de `planta_eixos_trocados`. **Nenhum `y` é zero e a série não é
# constante** — âncoras de `planta_sinal_invertido` e de `planta_corte_de_eixo`.
NOMES: Final = (
    "1.00_100.0m/z",  # q abaixo do corte, fc negativo  -> ABAIXO
    "2.00_200.0m/z",  # q abaixo do corte, fc positivo  -> ACIMA
    "3.00_300.0m/z",  # q abaixo do corte, fc == 0      -> SEM_DIRECAO
    "4.00_400.0m/z",  # q acima do corte                -> NAO
    "5.00_500.0m/z",  # fold change indefinido          -> fora do desenho
    "6.00_600.0m/z",  # p indefinido, fold change bom   -> fora do desenho
)
LOG2FC: Final = (-3.0, 2.5, 0.0, 1.25, None, 4.0)
PVALUE: Final = (0.0001, 0.001, 0.002, 0.4, 0.03, None)
QVALUE: Final = (0.001, 0.01, 0.02, 0.5, 0.04, None)

ALFA: Final = 0.05
N_TESTADAS: Final = len(NOMES)

# Os dois `n` de grupo. **Literais aqui de propósito:** são a fonte independente da classe (d), e o
# construtor recebe-os por um caminho que não passa por estes nomes. **E DIFEREM entre si**, senão
# uma troca das duas pontas do par não separaria nada.
N_A: Final = 18
N_B: Final = 14

GRUPO_A: Final = "Asymptomatic"
GRUPO_B: Final = "HAMTSP"


def _contraste(**troca: Any) -> Contraste:
    campos: dict[str, Any] = {
        "grupo_a": GRUPO_A,
        "grupo_b": GRUPO_B,
        "n_grupo_a": N_A,
        "n_grupo_b": N_B,
        "parametrico": True,
        "alfa_fdr": ALFA,
        "correcao": CORRECAO_DE_MULTIPLOS_TESTES,
        "rota_da_normalidade": "forced",
        "shapiro_pvalue": None,
        "n_residuos": None,
        "matriz_do_pvalue": MATRIZ_DO_PVALUE,
        "matriz_do_fold_change": MATRIZ_DO_FOLD_CHANGE,
        "matrizes_diferem": True,
    }
    campos.update(troca)
    return contraste_do_fio(**campos)


def _pontos(**troca: Any) -> PontosDoVolcano:
    campos: dict[str, Any] = {
        "features": list(NOMES),
        "log2fc": list(LOG2FC),
        "pvalue": list(PVALUE),
        "qvalue_fdr": list(QVALUE),
        "n_features_testadas": N_TESTADAS,
        "alfa": ALFA,
    }
    campos.update(troca)
    return pontos_do_volcano(**campos)


def _figura(pontos: PontosDoVolcano | None = None, contraste: Contraste | None = None) -> Figure:
    p = pontos if pontos is not None else _pontos()
    c = contraste if contraste is not None else _contraste()
    return figura_do_volcano(pontos=p, contraste=c, recorte=recorte_do_volcano(p, c))


def _y(p: float) -> float:
    """O `y` esperado, derivado da ENTRADA e por um caminho que não passa pelo construtor."""
    return float(-np.log10(p))


def _rodape(figura: Figure) -> str:
    """O texto do rodapé **com o espaço em branco colapsado**, e isto não é conveniência.

    `recorte.aplica` quebra a frase em linhas pela largura útil da figura, e a quebra cai onde
    calhar. Uma busca de substring sobre o texto cru devolve **falso negativo** para qualquer frase
    que atravesse uma quebra — é o modo, medido sobre esta prosa, e a direção é a perigosa:
    o veredito lê-se como *"a figura não declara isto"* e a reação é acrescentar o que já lá está.

    O gesto é o de `scripts/verifica_citacao.py`: `" ".join(texto.split())` dos **dois** lados.
    """
    return " ".join(figura.texts[0].get_text().split())


# As quatro desenháveis, na ordem em que o fio as traz.
X_DESENHADOS: Final = (-3.0, 2.5, 0.0, 1.25)
Y_DESENHADOS: Final = (_y(0.0001), _y(0.001), _y(0.002), _y(0.4))


# ---------------------------------------------------------------------------
# 1. O caminho feliz — as quatro classes, e o ponto de cada uma
# ---------------------------------------------------------------------------


def test_as_quatro_classes_sao_as_plantadas_e_o_ponto_de_cada_uma_e_o_que_entrou() -> None:
    """**A planta é o dado: quatro features com desfecho conhecido, uma por classe.**

    A âncora vem primeiro: as quatro classes têm de estar representadas, senão a asserção sobre a
    partição cairia sobre dado em que as classes não diferem.
    """
    pontos = _pontos()
    assert set(pontos.classes) == set(Classe), (
        f"âncora: o material sintético cobre {sorted(set(pontos.classes))} e não as quatro classes "
        "— o par causal abaixo não separaria a classificação de um construtor que devolvesse "
        "sempre a mesma"
    )
    assert pontos.classes == (Classe.ABAIXO, Classe.ACIMA, Classe.SEM_DIRECAO, Classe.NAO)
    assert pontos.nomes == NOMES[:4]
    assert pontos.x.tolist() == list(X_DESENHADOS)
    assert pontos.y.tolist() == pytest.approx(list(Y_DESENHADOS))


def test_a_figura_limpa_passa_nas_quatro_classes_de_ktd12() -> None:
    """As quatro classes de asserção sobre a figura sã. **É a outra ponta dos pares causais.**

    O dado esperado é construído aqui, dos literais do módulo, e não lido de volta do construtor.
    """
    figura = _figura()
    eixo = figura.axes[0]

    assert confere_series(eixo, {ROTULO_DA_SERIE: (X_DESENHADOS, Y_DESENHADOS)}) == []
    assert (
        confere_rotulos(
            eixo,
            x=rotulo_de_x(GRUPO_A, GRUPO_B),
            y=ROTULO_DE_Y,
            titulo=f"{TITULO} — {GRUPO_A} contra {GRUPO_B}\n{frase_da_populacao(_pontos())}",
            legenda=[
                rotulo_da_classe(Classe.ABAIXO, 1, ALFA),
                rotulo_da_classe(Classe.SEM_DIRECAO, 1, ALFA),
                rotulo_da_classe(Classe.ACIMA, 1, ALFA),
                rotulo_da_classe(Classe.NAO, 1, ALFA),
            ],
        )
        == []
    )
    assert confere_vista(eixo) == []
    assert confere_numeros(figura, _numeros_esperados()) == []


def _sobrevivem() -> int:
    """Os sobreviventes sobre a população TODA — o predicado do núcleo, e nada mais.

    **Nenhuma cláusula sobre os eixos aqui, e a ausência é o conserto.** Uma derivação com
    `fc is not None and p is not None` colada produziria no teste **o mesmo número errado** que o
    construtor produzia, e a classe (d) compararia um defeito consigo próprio.
    """
    return sum(1 for q in QVALUE if q is not None and q < ALFA)


def _sobrevivem_com_ponto() -> int:
    """Os sobreviventes que TÊM ponto: precisam dos dois eixos finitos. É o que a legenda conta."""
    return sum(
        1
        for fc, p, q in zip(LOG2FC, PVALUE, QVALUE, strict=True)
        if fc is not None and p is not None and q is not None and q < ALFA
    )


def _numeros_esperados() -> list[str]:
    """Todo número que a figura declara, derivado da ENTRADA e não do objeto que o construtor usou.

    **É a classe (d), e a metade que a torna útil é a fonte independente.** Nada aqui lê o
    `Recorte` nem o `PontosDoVolcano`: as contagens são contadas dos literais do módulo, e os
    denominadores são os literais.
    """
    desenhadas = len(X_DESENHADOS)
    sem_fc = sum(1 for v in LOG2FC if v is None)
    sem_p = sum(1 for fc, p in zip(LOG2FC, PVALUE, strict=True) if fc is not None and p is None)
    menor_q = min(q for q in QVALUE if q is not None)
    sobrevivem = _sobrevivem()
    sem_ponto = sobrevivem - _sobrevivem_com_ponto()
    sem_ponto_por_fc = sum(
        1 for fc, q in zip(LOG2FC, QVALUE, strict=True) if fc is None and q is not None and q < ALFA
    )
    return [
        # O rodapé: `N de M features`, os dois `n` de grupo, o corte e o menor `q`.
        milhar(desenhadas),
        milhar(N_TESTADAS),
        milhar(N_TESTADAS),
        milhar(sem_fc + sem_p),
        milhar(N_A),
        milhar(N_B),
        decimal(ALFA),
        milhar(sobrevivem),
        milhar(N_TESTADAS),
        milhar(N_TESTADAS),
        decimal(menor_q),
        # E a decomposição da distância, que só sai quando ela não é zero.
        milhar(sem_ponto),
        milhar(sem_ponto_por_fc),
        milhar(sem_ponto - sem_ponto_por_fc),
        # O título: pontos, população com o denominador colado, os sem ponto, e os dois indefinidos.
        milhar(desenhadas),
        milhar(N_TESTADAS),
        milhar(sobrevivem),
        milhar(sem_ponto),
        milhar(sem_fc),
        milhar(sem_p),
        # A legenda: o corte quatro vezes e a contagem de cada classe, que é a da NUVEM.
        decimal(ALFA),
        "1",
        decimal(ALFA),
        "1",
        decimal(ALFA),
        "1",
        "1",
        # `log2` e `-log10` do rótulo dos eixos NÃO entram: `numeros_declarados` lê o rodapé, o
        # título e a legenda, e nenhum deles os contém.
    ]


def test_ha_UM_artista_de_dado_e_os_nomes_indexam_por_posicao() -> None:  # noqa: N802
    """**O requisito que a identificação de R2 depende, e ele é sobre a CONTAGEM de artistas.**

    `PainelDeFigura._identifica` faz `self._nomes[posicao]`, onde `posicao` é o índice do ponto
    dentro do artista que o `pick_event` trouxe. Com dois artistas de dado o índice passa a ser
    relativo a cada um, e a identificação devolve o nome de outra feature — com a figura bem
    formada. A legenda tem quatro entradas e nenhuma delas é um artista do eixo.
    """
    figura = _figura()
    eixo = figura.axes[0]
    assert len(eixo.collections) == 1, [c.get_label() for c in eixo.collections]
    assert len(eixo.lines) == 0, [ln.get_label() for ln in eixo.lines]
    assert len(series(eixo)) == 1

    pontos = _pontos()
    desenhado = series(eixo)[0]
    assert len(pontos.nomes) == desenhado.x.size, (
        "o vetor de nomes e o de pontos têm comprimentos diferentes — a identificação devolveria o "
        "nome do vizinho a partir da primeira divergência"
    )
    for i, nome in enumerate(pontos.nomes):
        assert (desenhado.x[i], desenhado.y[i]) == pytest.approx(
            (X_DESENHADOS[i], Y_DESENHADOS[i])
        ), f"o ponto {i} não é o da feature {nome}"


# ---------------------------------------------------------------------------
# 2. A partição é TOTAL — nenhuma feature desaparece entre o fio e o desenho
# ---------------------------------------------------------------------------


def test_as_DUAS_particoes_somam_a_populacao_testada() -> None:  # noqa: N802
    """**A propriedade que impede o desaparecimento silencioso — R6, e são DOIS eixos.**

    Os sentinelas `fc.log = ±99` do MetaboAnalyst empurram a feature indefinida para a borda; aqui
    ela sai do desenho, e a única coisa que garante que ela não sumiu é esta soma. A partição da
    **classificação** é a segunda, e é a que faltava: sem ela o sobrevivente sem ponto desaparecia
    da contagem sem nada acusar, que é o que acontecia até 17/09/2026.

    **A âncora vem primeiro:** as duas contagens de indefinido têm de ser diferentes de zero, senão
    a soma valeria por vacuidade sobre uma população sem nenhum indefinido.
    """
    c = _pontos().contagens
    assert c.sem_fold_change > 0 and c.sem_pvalue_desenhavel > 0, (
        f"âncora: o material não tem indefinido nos dois eixos ({c}) — a soma abaixo não separaria "
        "um construtor que descartasse features em silêncio"
    )
    assert c.sobreviventes + c.nao_significativas + c.sem_qvalue == N_TESTADAS, c
    assert c.desenhadas + c.nao_desenhadas == N_TESTADAS, c
    # As quatro classes DENTRO do desenho somam o desenho, que é o que a legenda conta.
    assert c.abaixo + c.acima + c.sem_direcao + c.nao_significativas_desenhadas == c.desenhadas, c
    # E as duas razões de não desenhar somam o complemento do desenho.
    assert c.sem_fold_change + c.sem_pvalue_desenhavel == c.nao_desenhadas, c


def test_o_p_indefinido_e_contado_SO_quando_ha_fold_change() -> None:  # noqa: N802
    """**O escopo da segunda contagem — o par causal tem as duas direções.**

    Uma feature sem `p` **e** sem fold change não pode contar nas duas contagens, senão a soma
    excede a população. A ponta A: com fold change, conta em `sem_pvalue_desenhavel`. A ponta B: sem
    os dois, conta só em `sem_fold_change`.

    **E `pvalue_indefinido` é INCONDICIONAL, que é o que o separa do outro:** ele conta
    todas as features sem `p`, tenham ou não fold change, e é o que torna a figura comparável com
    `UnivariateCounts.pvalue_undefined`.
    """
    so_p = _pontos(log2fc=[1.0] * N_TESTADAS, pvalue=[None] * N_TESTADAS)
    assert (so_p.contagens.sem_fold_change, so_p.contagens.sem_pvalue_desenhavel) == (
        0,
        N_TESTADAS,
    )

    nenhum = _pontos(log2fc=[None] * N_TESTADAS, pvalue=[None] * N_TESTADAS)
    assert (nenhum.contagens.sem_fold_change, nenhum.contagens.sem_pvalue_desenhavel) == (
        N_TESTADAS,
        0,
    ), "uma feature sem os DOIS valores contou duas vezes: a partição excede a população"
    assert nenhum.contagens.pvalue_indefinido == N_TESTADAS, (
        "a contagem INCONDICIONAL do `p` seguiu a condicional — as duas são a mesma quantidade com "
        "escopos diferentes, e colapsá-las troca uma pela outra"
    )
    assert _pontos().contagens.pvalue_indefinido == 1


def test_p_exactamente_zero_e_contado_a_parte_e_nao_e_desenhavel() -> None:
    """**`-log10(0)` é `+inf`: não é desenhável, e é fato DIFERENTE de `p` indefinido.**

    `np.isfinite(0.0)` é verdadeiro, então a feature **entra** no ajuste de Benjamini-Hochberg e
    conta no `n` dele; o que ela não sobrevive é a `-log10`. As duas coisas são verdadeiras ao
    mesmo tempo e nenhum filtro de finitude as separa.
    """
    com_zero = _pontos(pvalue=[0.0, *PVALUE[1:]])
    assert com_zero.contagens.pvalue_zero == 1
    assert com_zero.nomes == NOMES[1:4], "a feature de `p` zero foi desenhada em `+inf`"
    assert "1 com p exactamente zero" in frase_da_populacao(com_zero)
    # A outra ponta: sem `p` zero, a figura não fala dele.
    assert "p exactamente zero" not in frase_da_populacao(_pontos())


def test_o_menor_q_e_sobre_as_TESTADAS_e_nao_sobre_as_desenhadas() -> None:  # noqa: N802
    """**Rótulo certo sobre a população errada, e a planta cai sobre dado em que as duas DIFEREM.**

    A feature `5.00_500.0m/z` tem `q = 0,04` — o segundo menor — e fold change indefinido, logo não
    é desenhada. Um `menor_q` calculado sobre as desenhadas devolveria `0,001`; sobre as testadas
    devolve `0,001` na mesma... então a planta tem de pôr o mínimo NA feature que não se desenha.
    """
    fora_do_desenho = [0.9, 0.9, 0.9, 0.9, 0.0007, None]
    pontos = _pontos(qvalue_fdr=fora_do_desenho)
    # O mínimo sobre as DESENHADAS: as que têm `log2fc` **e** `pvalue`, que são as quatro
    # primeiras. A feature de `q = 0,0007` é a quinta, e é precisamente a que não se desenha.
    das_desenhadas = min(
        q
        for q, x, p in zip(fora_do_desenho, LOG2FC, PVALUE, strict=True)
        if q is not None and x is not None and p is not None
    )
    assert das_desenhadas != 0.0007, (
        "âncora: o mínimo das desenhadas e o das testadas coincidem neste dado, e a asserção "
        "abaixo passaria com a população errada"
    )
    assert pontos.menor_q == pytest.approx(0.0007)


def test_sem_q_finito_o_menor_e_None_e_a_figura_DIZ_que_nenhum_e_definido() -> None:  # noqa: N802
    """**`None` é *"nenhum q é definido"* e não *"o menor é 1,0"*, e o rodapé di-lo.**"""
    pontos = _pontos(qvalue_fdr=[None] * N_TESTADAS)
    assert pontos.menor_q is None
    assert pontos.contagens.sem_qvalue == N_TESTADAS
    assert "nenhum q é definido" in recorte_do_volcano(pontos, _contraste()).metodo
    # A outra ponta: com `q` definido, o rodapé traz o número e não a frase.
    assert "nenhum q é definido" not in recorte_do_volcano(_pontos(), _contraste()).metodo


# ---------------------------------------------------------------------------
# 2b. O SOBREVIVENTE SEM PONTO — classificar não é desenhar, e o par causal da planta
# ---------------------------------------------------------------------------
#
# **O defeito que estes quatro fecham tinha número.** `pontos_do_volcano` trazia `& desenhavel`
# colado ao predicado da significância, e as classes só existiam para as desenháveis: uma feature
# com `p = 0` exacto — `-log10(0)` é `+inf` — ou sem `log2fc` definido saía do desenho **e da
# contagem**. Medido em 17/09/2026 com `p = [0, 0,4, 0,6, 0,9]` pelo fio real: o núcleo contava
# **1** e a figura publicava *«0 sobrevive(m) ao FDR»* ao lado de *«1 com p exactamente zero»*.


def _q_do_fio(pvalue: list[float | None]) -> list[float | None]:
    """O `q` **derivado** do `p` por `benjamini_hochberg` — o fio da produção, e não dois vetores.

    ## A lacuna que isto fecha, e ela deixou passar o defeito que este bloco conserta

    `_pontos` recebe `pvalue` e `qvalue_fdr` de forma **independente**, e isso é conveniência de
    arnês que o fio real **não pode** oferecer: quem produz `UnivariateResult.qvalue_fdr` é
    `benjamini_hochberg` sobre o vetor de `p`, e ela devolve `0` **exactamente** onde o `p` é `0` —
    o menor `p` ajustado é `p₍₁₎·n/1`, que é zero quando `p₍₁₎` é zero.

    Consequência: o par `(p = 0, q = 0)` era **inalcançável** por construção neste módulo enquanto
    todo cenário escrevesse os dois vetores à mão. **O valor natural do arnês era o que
    executava menos do caminho.**
    """
    valores = np.asarray([np.nan if v is None else v for v in pvalue], dtype=float)
    q = benjamini_hochberg(valores)
    return [None if not np.isfinite(v) else float(v) for v in q]


_P_DA_PLANTA: Final[list[float | None]] = [0.0, 0.4, 0.6, 0.9]
_P_SEM_PLANTA: Final[list[float | None]] = [0.001, 0.4, 0.6, 0.9]
# **Finitos e não nulos**, para que a planta caia sobre o eixo do `p` e não sobre o do fold change.
_FC_DA_PLANTA: Final = [-3.0, 2.5, -1.25, 1.75]


def _pelo_fio_real(pvalue: list[float | None], log2fc: list[float | None]) -> PontosDoVolcano:
    return pontos_do_volcano(
        features=[f"{i + 1}.00_{(i + 1) * 100}.0m/z" for i in range(len(pvalue))],
        log2fc=list(log2fc),
        pvalue=list(pvalue),
        qvalue_fdr=_q_do_fio(pvalue),
        n_features_testadas=len(pvalue),
        alfa=ALFA,
    )


def test_o_pvalue_ZERO_conta_como_sobrevivente_e_a_figura_declara_que_ele_nao_tem_ponto() -> None:  # noqa: N802
    """**A PLANTA: uma feature com `p = 0` exacto, pelo fio real.**

    As três âncoras de não-vacuidade vêm antes do desfecho:

    1. o `q` derivado tem de ter um zero **exacto** — senão não há planta;
    2. ela tem de ficar **fora** do desenho — senão a classificação e o desenho não divergem, e a
       asserção passaria sem separar nada;
    3. tem de haver ponto desenhado — senão *"1 sobrevivente"* passaria sobre uma nuvem vazia.
    """
    pontos = _pelo_fio_real(_P_DA_PLANTA, _FC_DA_PLANTA)
    c = pontos.contagens

    assert pontos.menor_q == 0.0, "âncora 1: o fio real tem de produzir q = 0 exacto"
    assert c.pvalue_zero == 1
    assert c.desenhadas == 3 and c.nao_desenhadas == 1, (
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
    # E a decomposição atribui a causa certa: aqui o fold change está definido.
    assert c.sobreviventes_sem_fold_change == 0
    assert c.sobreviventes_sem_pvalue_desenhavel == 1


def test_o_sobrevivente_SEM_FOLD_CHANGE_conta_e_a_decomposicao_nomeia_a_OUTRA_causa() -> None:  # noqa: N802
    """A mesma classe pela outra porta: aqui quem tira do desenho é o `log2fc` indefinido.

    **São 868 de 5.942 no `pos_hilic`**, no par e na cascata que `plots/volcano.py` declara, e é
    esta a causa que domina naquele corpus —
    a decomposição do rodapé existe para não atribuir a distância à causa errada.
    """
    pontos = _pelo_fio_real(_P_SEM_PLANTA, [None, 2.5, -1.25, 1.75])
    c = pontos.contagens
    assert c.desenhadas == 3 and c.sem_fold_change == 1, (
        f"âncora: a planta não tirou do desenho {c}"
    )
    assert c.sobreviventes == 1, f"o sobrevivente sem fold change saiu da contagem — {c}"
    assert c.sobreviventes_sem_ponto == 1
    assert c.sobreviventes_sem_fold_change == 1
    assert c.sobreviventes_sem_pvalue_desenhavel == 0
    assert (
        c.sobreviventes_sem_fold_change + c.sobreviventes_sem_pvalue_desenhavel
        == c.sobreviventes_sem_ponto
    ), "a decomposição tem de ser TOTAL, senão uma terceira causa esconde-se entre as duas"


def test_a_figura_do_sobrevivente_SEM_PONTO_diz_a_verdade_no_titulo_e_no_rodape() -> None:  # noqa: N802
    """R6 nas duas superfícies que a carregam, e a legenda continua a contar a NUVEM.

    **O rodapé é a única coisa da figura que outra pessoa pode desmentir**, e é onde a causa vai
    nomeada e decomposta. A legenda não recebe a cláusula porque as três classes significativas
    desta figura são de **direção**, e um sobrevivente sem `log2fc` não tem direção — ver o
    cabeçalho de `plots/volcano.py`.
    """
    pontos = _pelo_fio_real(_P_DA_PLANTA, _FC_DA_PLANTA)
    contraste = _contraste()
    figura = figura_do_volcano(
        pontos=pontos, contraste=contraste, recorte=recorte_do_volcano(pontos, contraste)
    )

    titulo = " ".join(figura.axes[0].get_title().split())
    assert "3 ponto(s) de 4 features testadas" in titulo, titulo
    assert "1 de 4 sobrevive(m) ao FDR" in titulo, titulo
    assert "1 sobrevivente(s) sem ponto" in titulo, titulo
    assert "1 com p exactamente zero" in titulo, titulo

    rodape = _rodape(figura)
    assert "1 de 4 sobrevive(m)" in rodape, rodape
    assert "1 não tem ponto na nuvem" in rodape, rodape
    assert "0 sem fold change definido e 1 com p não desenhável" in rodape, rodape
    assert "ausência de valor de EIXO" in rodape, rodape

    # A legenda conta a nuvem: nenhuma das três classes significativas tem ponto.
    legenda = [t.get_text() for t in figura.axes[0].get_legend().get_texts()]
    assert rotulo_da_classe(Classe.ABAIXO, 0, ALFA) in legenda, legenda
    assert rotulo_da_classe(Classe.ACIMA, 0, ALFA) in legenda, legenda
    assert rotulo_da_classe(Classe.NAO, 3, ALFA) in legenda, legenda
    assert series(figura.axes[0])[0].x.size == 3


def test_PAR_CAUSAL_com_a_planta_do_p_zero_REMOVIDA_a_figura_deixa_de_acusar() -> None:  # noqa: N802
    """**A outra ponta: sem `p = 0` nenhuma cláusula nova sai, e as duas partições coincidem.**

    Duas execuções idênticas que diferem só pela planta. Sem esta metade, as asserções acima seriam
    satisfeitas por uma figura que escreve a cláusula **sempre** — o mecanismo
    que nunca deixa de acusar.
    """
    com = _pelo_fio_real(_P_DA_PLANTA, _FC_DA_PLANTA)
    sem = _pelo_fio_real(_P_SEM_PLANTA, _FC_DA_PLANTA)

    assert sem.menor_q is not None and sem.menor_q > 0.0, (
        f"âncora: a planta removida não pode continuar a produzir q = 0 — {sem.menor_q}"
    )
    assert sem.contagens.pvalue_zero == 0
    assert sem.contagens.desenhadas == 4 and sem.contagens.nao_desenhadas == 0
    assert sem.contagens.sobreviventes == sem.contagens.sobreviventes_desenhados, (
        "sem a planta as duas partições coincidem, e é isso que torna a divergência atribuível"
    )
    assert sem.contagens.sobreviventes_sem_ponto == 0

    contraste = _contraste()
    limpa = figura_do_volcano(
        pontos=sem, contraste=contraste, recorte=recorte_do_volcano(sem, contraste)
    )
    titulo = " ".join(limpa.axes[0].get_title().split())
    rodape = _rodape(limpa)
    # **A âncora de não-vacuidade vem ANTES das ausências:** as três asserções abaixo são de
    # ausência, e ausência é satisfeita por texto vazio. Estas duas dizem que há texto, e que ele é
    # o da execução sem planta.
    assert "1 de 4 sobrevive(m) ao FDR" in titulo, titulo
    assert "1 de 4 sobrevive(m)" in rodape, rodape
    for frase in ("sobrevivente(s) sem ponto", "com p exactamente zero"):
        assert frase not in titulo, f"a cláusula saiu sem a planta — {titulo}"
    assert "não tem ponto na nuvem" not in rodape, f"a cláusula saiu sem a planta — {rodape}"

    # E o desfecho DIVERGE entre as duas execuções — é isso que faz disto um par causal.
    assert com.contagens.sobreviventes_sem_ponto != sem.contagens.sobreviventes_sem_ponto
    assert com.contagens.desenhadas != sem.contagens.desenhadas
    assert com.contagens.sobreviventes == sem.contagens.sobreviventes == 1, (
        "a CLASSIFICAÇÃO não muda com a planta, e é isso que a separa do desenho"
    )


# A matriz da planta do núcleo, e cada coluna existe para um desfecho.
#
# **`exacta`** é constante dentro de cada grupo e diferente entre eles: Welch dá `t = -inf` e
# `p = 0` exacto, e o BH leva o `p = 0` a `q = 0`. O `log2fc` dela é finito, então ela sai do
# desenho pelo eixo do `p`. **`exacta_sem_fc`** é a mesma coisa com a mediana do primeiro grupo em
# zero: a razão não é positiva, `log2fc` é `nan`, e ela sai pelos dois eixos. **`nula`** não separa
# nada, e existe para o BH ter denominador e a nuvem ter ponto.
_VALORES_DA_PLANTA: Final = (
    (1.0, 0.0, 1.0),
    (1.0, 0.0, 2.0),
    (1.0, 0.0, 3.0),
    (9.0, 9.0, 1.0),
    (9.0, 9.0, 2.0),
    (9.0, 9.0, 3.0),
)
_GRUPOS_DA_PLANTA: Final = ("A", "A", "A", "B", "B", "B")


def test_a_contagem_da_figura_BATE_com_a_do_NUCLEO() -> None:  # noqa: N802
    """**O alvo certo é o número do NÚCLEO, e não a coerência interna da figura.**

    `rotas.executar` grava `n_significant_fdr` de `q[definidos] < FDR_ALPHA`, com
    `definidos = np.isfinite(q)` — e é essa a expressão reproduzida aqui, verbatim, sobre o mesmo
    `UnivariateResult`. A figura tem de chegar ao mesmo número a partir dos **vetores**, que é tudo
    o que atravessa o fio: `_resumo_do_volcano` declara que as contagens de `UnivariateCounts` não
    viajam. Enquanto ninguém cruzou os dois, a figura dizia `0` e o núcleo dizia `2`.

    **As âncoras vêm antes do desfecho:** o núcleo tem de contar mais de zero, senão a igualdade
    passa por vacuidade; e as duas causas de ponto ausente têm de estar representadas, senão a
    decomposição não é exercitada.

    **`parametric=True` e não `None`**, porque a decisão automática corre Shapiro-Wilk sobre seis
    resíduos: só Welch devolve `p = 0` exacto para a coluna que separa os grupos perfeitamente.
    """
    from refino.domain.matrix import RefinoMatrix
    from refino.domain.metadata import SampleMetadata

    matriz = RefinoMatrix(
        values=np.asarray(_VALORES_DA_PLANTA, dtype=float),
        feature_names=("exacta", "exacta_sem_fc", "nula"),
        samples=tuple(
            SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type="sample", group=g)
            for i, g in enumerate(_GRUPOS_DA_PLANTA)
        ),
        mode="sintetico",
    )
    resultado = univariate_test(matriz, ("A", "B"), parametric=True)
    q = np.asarray(resultado.qvalue_fdr, dtype=np.float64)
    definidos = np.isfinite(q)
    do_nucleo = int((q[definidos] < FDR_ALPHA).sum())

    assert do_nucleo == 2, f"âncora: o núcleo tem de contar os dois exactos — q = {q.tolist()}"
    assert float(resultado.pvalue[0]) == 0.0, "âncora: a planta tem de dar p = 0 exacto"
    assert not np.isfinite(resultado.log2fc[1]), "âncora: a segunda coluna tem de ficar sem fc"

    opcional = [None if not np.isfinite(v) else float(v) for v in resultado.log2fc]
    pontos = pontos_do_volcano(
        features=list(resultado.feature),
        log2fc=opcional,
        pvalue=[None if not np.isfinite(v) else float(v) for v in resultado.pvalue],
        qvalue_fdr=[None if not np.isfinite(v) else float(v) for v in q],
        n_features_testadas=len(resultado.feature),
        alfa=FDR_ALPHA,
    )
    c = pontos.contagens
    assert c.sobreviventes == do_nucleo, (
        f"a figura conta {c.sobreviventes} sobreviventes e o núcleo {do_nucleo}: são o mesmo "
        "predicado sobre o mesmo vetor"
    )
    assert c.pvalue_zero == resultado.counts.pvalue_zero
    assert c.sem_fold_change == resultado.counts.log2fc_undefined
    assert c.pvalue_indefinido == resultado.counts.pvalue_undefined
    assert c.sem_qvalue == resultado.counts.qvalue_undefined
    # E a divergência entre desenho e classificação está DECLARADA, com as DUAS causas.
    assert c.sobreviventes_sem_ponto == 2
    assert c.sobreviventes_sem_fold_change == 1
    assert c.sobreviventes_sem_pvalue_desenhavel == 1


# ---------------------------------------------------------------------------
# 3. A ORDEM do par de grupos — imagens espelhadas, e a figura diz qual é o numerador
# ---------------------------------------------------------------------------


def test_trocar_a_ordem_do_par_ESPELHA_o_eixo_x_e_a_figura_diz_qual_e_o_numerador() -> None:  # noqa: N802
    """**`(A,B)` e `(B,A)` são leituras legítimas; não dizer qual é qual não é.**

    A âncora vem primeiro: o `x` tem de ser diferente do simétrico dele sobre este dado, senão o
    espelho é a identidade e o par não separa nada.
    """
    direto = _pontos()
    espelhado = _pontos(log2fc=[None if v is None else -v for v in LOG2FC])
    assert direto.x.tolist() != espelhado.x.tolist(), (
        "âncora: o fold change deste material é simétrico, e espelhá-lo é a identidade"
    )
    assert espelhado.x.tolist() == [-v for v in X_DESENHADOS]

    # E a metade que importa: os dois rótulos de eixo dizem coisas DIFERENTES.
    ida = rotulo_de_x(GRUPO_A, GRUPO_B)
    volta = rotulo_de_x(GRUPO_B, GRUPO_A)
    assert ida != volta, "os dois sentidos do par produzem o mesmo rótulo de eixo"
    assert GRUPO_A in ida and GRUPO_B in ida
    assert ida.index(GRUPO_A) < ida.index(GRUPO_B), "o numerador tem de vir primeiro no rótulo"
    assert volta.index(GRUPO_B) < volta.index(GRUPO_A)


def test_a_classe_de_um_ponto_INVERTE_com_a_ordem_do_par() -> None:  # noqa: N802
    """A consequência da ordem na classificação: `ABAIXO` e `ACIMA` trocam de lugar."""
    direto = _pontos().contagens
    espelhado = _pontos(log2fc=[None if v is None else -v for v in LOG2FC]).contagens
    assert (direto.abaixo, direto.acima) == (1, 1)
    assert (espelhado.abaixo, espelhado.acima) == (1, 1)
    # E o ponto de cada classe é o outro — a asserção que a contagem simétrica não faz.
    assert _pontos().classes[0] is Classe.ABAIXO
    assert _pontos(log2fc=[None if v is None else -v for v in LOG2FC]).classes[0] is Classe.ACIMA


# ---------------------------------------------------------------------------
# 4. Os TRÊS vazios, e cada um tem desfecho próprio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vazio",
    [
        {"features": [], "log2fc": [], "pvalue": [], "qvalue_fdr": [], "n_features_testadas": 0},
        {
            "features": None,
            "log2fc": None,
            "pvalue": None,
            "qvalue_fdr": None,
            "n_features_testadas": None,
        },
    ],
)
def test_sem_feature_NENHUMA_a_recusa_e_MATRIZ_VAZIA(vazio: dict[str, Any]) -> None:  # noqa: N802
    """**Primeiro vazio: não veio população.** É defeito de quem chama, e o painel troca por texto.

    `MatrizVaziaError` é a única excepção que `PainelDeFigura._troca_a_figura` apanha; qualquer
    outra levantada dentro de um *slot* do Qt é engolida para o `stderr`, com o painel a parar de
    actualizar sem dizer nada.
    """
    with pytest.raises(MatrizVaziaError, match="não recebeu feature nenhuma"):
        _pontos(**vazio)


def test_sem_PAR_DE_GRUPOS_o_recorte_recusa_como_MATRIZ_VAZIA() -> None:  # noqa: N802
    """**O vazio que não é do dado: a execução correu sem `group_a`/`group_b`.**

    *"Não calculado"* e *"calculado e vazio"* são fatos opostos, e é este corpus que produz
    o segundo. Um recorte que desenhasse com o par em branco afirmaria que o teste correu.
    """
    with pytest.raises(MatrizVaziaError, match="não declara o par de grupos"):
        recorte_do_volcano(_pontos(), _contraste(grupo_a="", grupo_b=GRUPO_B))
    with pytest.raises(MatrizVaziaError, match="não declara o par de grupos"):
        recorte_do_volcano(_pontos(), _contraste(grupo_a=GRUPO_A, grupo_b=None))
    # A outra ponta: com o par declarado, o recorte sai.
    assert recorte_do_volcano(_pontos(), _contraste()).desenhadas == len(X_DESENHADOS)


def test_com_NENHUM_ponto_desenhavel_a_figura_DECLARA_em_vez_de_ficar_em_branco() -> None:  # noqa: N802
    """**Segundo vazio: veio população e nenhuma é desenhável — R4b, e é fato do dado.**

    É a convenção de `ItemCorpo.vazio_declarado`: gráfico vazio sem legenda é indistinguível de
    gráfico quebrado. A figura sai, com a frase no lugar dos pontos e a legenda com os zeros.
    """
    pontos = _pontos(log2fc=[None] * N_TESTADAS)
    assert pontos.nomes == (), "âncora: ainda há ponto desenhável, e o caso vazio não é este"

    figura = figura_do_volcano(
        pontos=pontos, contraste=_contraste(), recorte=recorte_do_volcano(pontos, _contraste())
    )
    eixo = figura.axes[0]
    assert series(eixo) == (), "nenhum artista de dado devia existir"
    assert len(eixo.texts) == 1, [t.get_text() for t in eixo.texts]
    declarado = eixo.texts[0].get_text()
    assert declarado == frase_do_vazio(pontos)
    assert milhar(N_TESTADAS) in declarado and "não descartada" not in declarado

    # **E as marcas de eixo SAEM — visto na imagem montada, e as duas direções.** Sem ponto
    # nenhum o `matplotlib` desenha `0,0` a `1,0` nos dois eixos, e a figura passa a afirmar que o
    # fold change deste contraste vai de zero a um. Nenhuma das quatro classes de asserção do arnês
    # o vê: não há série sobre a qual `confere_vista` possa correr.
    assert list(eixo.get_xticks()) == [] and list(eixo.get_yticks()) == [], (
        f"a figura vazia desenhou marcas em x={list(eixo.get_xticks())} e "
        f"y={list(eixo.get_yticks())} — números que ninguém mediu, com a autoridade de um eixo"
    )
    com_pontos = _figura().axes[0]
    assert list(com_pontos.get_xticks()) and list(com_pontos.get_yticks()), (
        "âncora: a figura COM pontos também ficou sem marcas, e a asserção acima passaria sobre "
        "um construtor que nunca as desenha"
    )

    # A legenda continua a existir: `[0]` é a resposta, e não a ausência da entrada.
    assert [t.get_text() for t in eixo.get_legend().get_texts()] == [
        rotulo_da_classe(Classe.ABAIXO, 0, ALFA),
        rotulo_da_classe(Classe.ACIMA, 0, ALFA),
        rotulo_da_classe(Classe.NAO, 0, ALFA),
    ]


def test_com_pontos_e_ZERO_significativas_a_nuvem_INTEIRA_e_desenhada() -> None:  # noqa: N802
    """**Terceiro vazio, e é o caso NORMAL deste corpus: há nuvem e ninguém sobrevive ao FDR.**

    A confusão que este teste separa é entre *"a figura está vazia"* e *"a resposta é zero"*. A
    segunda desenha tudo — o que é zero é a contagem na legenda.
    """
    pontos = _pontos(qvalue_fdr=[0.9] * N_TESTADAS)
    assert pontos.contagens.sobreviventes == 0
    assert pontos.contagens.desenhadas == len(X_DESENHADOS) > 0, (
        "âncora: sem ponto desenhado este teste seria o caso vazio, e não o de zero significativas"
    )
    figura = figura_do_volcano(
        pontos=pontos, contraste=_contraste(), recorte=recorte_do_volcano(pontos, _contraste())
    )
    assert series(figura.axes[0])[0].x.size == len(X_DESENHADOS)
    assert f"0 de {milhar(N_TESTADAS)} sobrevive(m) ao FDR" in figura.axes[0].get_title()
    assert f"0 de {milhar(N_TESTADAS)} sobrevive(m)" in _rodape(figura)
    assert "sem ponto" not in figura.axes[0].get_title(), (
        "âncora: sem sobrevivente escondido a cláusula NÃO sai — a ausência dela é a informação"
    )


def test_a_classe_SEM_DIRECAO_so_aparece_na_legenda_quando_EXISTE() -> None:  # noqa: N802
    """**A ausência da entrada é a informação, e a presença dela também — as duas direções.**

    `ABAIXO`, `ACIMA` e `NAO` saem sempre com `[0]` explícito, porque *"quantas para cada lado?"* é
    a leitura que o profissional procura. `SEM_DIRECAO` é o caso raro que é preciso
    separar, e uma entrada `[0]` para ela ensinaria a procurar uma classe que quase nunca ocorre.
    """
    com = [t.get_text() for t in _figura().axes[0].get_legend().get_texts()]
    assert any("sem direção" in t for t in com), com

    sem = _pontos(log2fc=[-3.0, 2.5, 1.5, 1.25, None, 4.0])
    assert Classe.SEM_DIRECAO not in sem.classes, "âncora: ainda há ponto sem direção"
    rotulos_sem = [
        t.get_text()
        for t in figura_do_volcano(
            pontos=sem, contraste=_contraste(), recorte=recorte_do_volcano(sem, _contraste())
        )
        .axes[0]
        .get_legend()
        .get_texts()
    ]
    assert not any("sem direção" in t for t in rotulos_sem), rotulos_sem
    assert len(rotulos_sem) == 3


# ---------------------------------------------------------------------------
# 5. As plantas do arnês sobre ESTA figura
# ---------------------------------------------------------------------------


def _esperado() -> dict[str, tuple[Any, Any]]:
    return {ROTULO_DA_SERIE: (X_DESENHADOS, Y_DESENHADOS)}


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


def test_planta_corte_de_eixo_reprova_e_o_dado_continua_CERTO() -> None:  # noqa: N802
    """**A planta que as outras não alcançam.** Dado nenhum muda e a tela fica em branco."""
    eixo = _figura().axes[0]
    planta_corte_de_eixo(eixo)
    assert confere_series(eixo, _esperado()) == [], (
        "âncora: a planta do corte não pode mexer em dado — se mexeu, não é esta a classe testada"
    )
    achados = confere_vista(eixo)
    assert any("A figura está em branco" in a for a in achados), achados


def test_planta_da_serie_removida_RECUSA_porque_ha_um_artista_so() -> None:  # noqa: N802
    """**O negativo que NÃO se aplica, e a ausência é afirmada em vez de omitida.**

    `planta_serie_removida` exige duas séries: apagar a única deixa eixos vazios, e a reprovação
    deixaria de ser atribuível à planta. Um artista de dado é **requisito** desta figura — ver
    `test_ha_UM_artista_de_dado_e_os_nomes_indexam_por_posicao` —, e acrescentar um segundo para
    agradar ao arnês partiria a identificação de R2.
    """
    with pytest.raises(PlantaQueNaoSeparaError, match="ao menos duas séries"):
        planta_serie_removida(_figura().axes[0])


def test_planta_do_numero_declarado_reprova() -> None:
    """Classe (d): o rodapé é a única coisa da figura que outra pessoa pode desmentir."""
    figura = _figura()
    assert confere_numeros(figura, _numeros_esperados()) == []
    figura.texts[0].set_text("13 de 13 features")
    achados = confere_numeros(figura, _numeros_esperados())
    assert any("13" in a for a in achados), achados


# ---------------------------------------------------------------------------
# 6. O que a figura declara — R3, R5, R5b, R6, R8
# ---------------------------------------------------------------------------


def test_o_rodape_declara_os_QUATRO_itens_que_R8_exige() -> None:  # noqa: N802
    """**R8: contagem, menor `q`, o `n` de CADA grupo, e a correção aplicada.**

    Os dois últimos são o que separa *"não há efeito"* de *"não há poder"*. Sem eles, um volcano
    vazio de um estudo de 18 contra 14 amostras lê-se como ausência de biologia quando é ausência
    de amostra — e a figura falhou no trabalho dela mesmo tendo desenhado certo.

    **Os números vêm dos literais deste módulo**, por caminho independente do construtor.
    """
    recorte = recorte_do_volcano(_pontos(), _contraste())
    texto = recorte.texto()

    assert f"{_sobrevivem()} de {milhar(N_TESTADAS)} sobrevive(m)" in texto, texto
    assert f"menor q sobre as {milhar(N_TESTADAS)} testadas é {decimal(0.001)}" in texto
    assert f"{GRUPO_A} (n = {N_A})" in texto and f"{GRUPO_B} (n = {N_B})" in texto
    assert CORRECAO_DE_MULTIPLOS_TESTES in texto
    assert f"q < {decimal(ALFA)}" in texto


def test_o_rodape_declara_as_DUAS_procedencias_e_o_teste() -> None:  # noqa: N802
    """R5 e R5b, e as duas metades saem do fio em vez de serem escritas no construtor."""
    recorte = recorte_do_volcano(_pontos(), _contraste())
    assert MATRIZ_DO_FOLD_CHANGE in recorte.matriz and MATRIZ_DO_PVALUE in recorte.matriz
    assert "eixo x" in recorte.matriz and "eixo y" in recorte.matriz
    assert WELCH in recorte.metodo
    assert DIVERGENCIA in recorte.metodo


def test_quando_as_matrizes_NAO_diferem_o_rodape_diz_que_sao_a_mesma() -> None:  # noqa: N802
    """**A ressalva de `transform_moved_the_matrix`, com as duas direções.**

    Com `transformation` e `scaling` em `none` a pré-transformação e a processada são literalmente
    a mesma matriz; as tuplas de `provenance` diferem porque a segunda regista as duas etapas que
    não fizeram nada. Declarar duas procedências ali afirmaria uma distinção que a execução não
    tem.
    """
    iguais = recorte_do_volcano(_pontos(), _contraste(matrizes_diferem=False)).matriz
    diferentes = recorte_do_volcano(_pontos(), _contraste(matrizes_diferem=True)).matriz
    assert "as DUAS são a mesma matriz" in iguais
    assert "as DUAS são a mesma matriz" not in diferentes
    # E os dois rótulos continuam nas duas.
    for texto in (iguais, diferentes):
        assert MATRIZ_DO_FOLD_CHANGE in texto and MATRIZ_DO_PVALUE in texto


def test_procedencia_ausente_no_fio_NAO_inventa_um_nome() -> None:  # noqa: N802
    """Guarda de fronteira: o recuo é declarado, nunca um rótulo plausível."""
    recorte = recorte_do_volcano(
        _pontos(), _contraste(matriz_do_pvalue=None, matriz_do_fold_change="")
    )
    assert recorte.matriz.count(NAO_DECLARADA) == 2, recorte.matriz


@pytest.mark.parametrize(
    ("parametrico", "esperado"),
    [(True, WELCH), (False, MANN_WHITNEY)],
)
def test_o_nome_do_teste_e_DERIVADO_da_bandeira(parametrico: bool, esperado: str) -> None:  # noqa: N802
    """**As duas direções, e a errada custaria uma legenda falsa no corpus real.**

    Uma constante `"Welch"` no painel estaria errada sobre o `pos_hilic`: Shapiro-Wilk rejeita a
    normalidade dos resíduos e a decisão automática vai para Mann-Whitney.
    """
    assert nome_do_teste(parametrico) == esperado
    recorte = recorte_do_volcano(_pontos(), _contraste(parametrico=parametrico))
    assert esperado in recorte.metodo


@pytest.mark.parametrize(
    ("rota", "p", "n", "trecho"),
    [
        ("forced", None, None, "FORÇADO pelo pedido"),
        ("shapiro", 2.3664e-95, 5000, "p = 2,37e-95, n = 5.000"),
        ("shapiro", None, None, "p não declarado, n não declarado"),
        ("shapiro_unavailable", None, None, "sem resíduos que chegassem"),
        ("marciano", None, None, "não reconhecida"),
    ],
)
def test_a_prosa_da_normalidade_tem_as_TRES_rotas_mais_o_desconhecido(  # noqa: N802
    rota: str, p: float | None, n: int | None, trecho: str
) -> None:
    """**Três rotas e não duas** — `NormalityDecision` separa *"medi e rejeitei"* de *"não medi"*.

    A quinta linha é a guarda de fronteira num vocabulário que atravessa o fio: rota desconhecida
    devolve o rótulo cru em vez de cair num `else` silencioso que a faria passar por uma das três.
    """
    assert trecho in prosa_da_normalidade(rota, p, n)


def test_o_titulo_conta_o_que_ficou_de_FORA_e_omite_a_clausula_quando_e_zero() -> None:  # noqa: N802
    """R6 no título, com as duas direções de cada cláusula condicional."""
    com = frase_da_populacao(_pontos())
    assert "1 sem fold change definido" in com and "1 sem p desenhável" in com

    sem = frase_da_populacao(_pontos(log2fc=[1.0] * N_TESTADAS, pvalue=[0.5] * N_TESTADAS))
    assert "sem fold change" not in sem and "sem p desenhável" not in sem
    # E as duas cláusulas incondicionais saem nas duas.
    for texto in (com, sem):
        assert "features testadas" in texto and "sobrevive(m) ao FDR" in texto


# ---------------------------------------------------------------------------
# 7. A fronteira: nada do que vem do fio passou pelo mypy
# ---------------------------------------------------------------------------


def test_nome_ilegivel_vira_VAZIO_e_nao_e_descartado() -> None:  # noqa: N802
    """**A posição tem de sobreviver.** Nome vazio identifica-se mal; nome deslocado, ERRADO."""
    lidos = nomes_do_fio(["a", None, 3, True, "b"])
    assert lidos == ("a", "", "", "", "b"), lidos
    assert nomes_do_fio("não é lista") == ()


def test_vetores_de_comprimentos_DIFERENTES_viram_indefinido_e_nao_levantam() -> None:  # noqa: N802
    """**A guarda que NÃO recusa, e a escolha é declarada.**

    O que falta vira indefinido, e indefinido é contado e declarado. Levantar aqui trocaria uma
    figura que diz *"1 sem p"* por um painel mudo.
    """
    curto = _pontos(pvalue=[0.01, 0.02])
    assert curto.n_testadas == N_TESTADAS
    assert curto.contagens.sem_pvalue_desenhavel + curto.contagens.sem_fold_change == N_TESTADAS - 2


def test_o_denominador_e_o_DECLARADO_e_nao_o_comprimento_do_vetor() -> None:  # noqa: N802
    """**As duas direções: com o campo, ele manda; sem ele, o recuo é o maior comprimento.**

    Derivá-lo de um comprimento faria um vetor truncado no fio passar por população menor sem nada
    acusar.
    """
    declarado = _pontos(n_features_testadas=9999)
    assert declarado.n_testadas == 9999
    recuo = _pontos(n_features_testadas=None)
    assert recuo.n_testadas == N_TESTADAS


def test_o_contraste_recua_em_cada_campo_ilegivel_sem_levantar() -> None:
    """Guarda de fronteira em cada um dos treze campos, com a ausência a ter valor próprio."""
    vazio = contraste_do_fio(
        grupo_a=None,
        grupo_b=None,
        n_grupo_a="dezoito",
        n_grupo_b=True,
        parametrico="sim",
        alfa_fdr=None,
        correcao=None,
        rota_da_normalidade=None,
        shapiro_pvalue="baixo",
        n_residuos=-1,
        matriz_do_pvalue=None,
        matriz_do_fold_change=None,
        matrizes_diferem="sim",
    )
    assert (vazio.grupo_a, vazio.grupo_b) == ("", "")
    assert (vazio.n_a, vazio.n_b) == (0, 0), "`True` não é uma contagem"
    assert vazio.parametrico is False, "`'sim'` não é `True`"
    assert np.isnan(vazio.alfa)
    assert vazio.shapiro_pvalue is None and vazio.n_residuos is None
    assert vazio.matrizes_diferem is False


# ---------------------------------------------------------------------------
# 8. A cor sai do CICLO e não deste módulo
# ---------------------------------------------------------------------------


def test_as_cores_saem_do_ciclo_dos_rcparams_e_seguem_o_tema() -> None:
    """**O par causal do tema: o MESMO construtor, dois ciclos, cores opostas.**

    É a prova de que nenhuma cor está escrita neste módulo — um literal daria a mesma figura nos
    dois contextos, e o tema passaria a ser uma correção posterior sobre cada artista.
    """
    with matplotlib.rc_context({"axes.prop_cycle": matplotlib.cycler(color=["#111111"])}):
        escuro = cores_das_classes()
    with matplotlib.rc_context({"axes.prop_cycle": matplotlib.cycler(color=["#EEEEEE"])}):
        claro = cores_das_classes()
    assert set(escuro.values()) == {"#111111"}
    assert set(claro.values()) == {"#EEEEEE"}
    assert escuro != claro

    # E com um ciclo de quatro, as quatro classes recebem cores distintas.
    quatro = ["#000001", "#000002", "#000003", "#000004"]
    with matplotlib.rc_context({"axes.prop_cycle": matplotlib.cycler(color=quatro)}):
        distintas = cores_das_classes()
    assert sorted(distintas.values()) == quatro
    assert set(distintas) == set(Classe)


def test_com_UMA_classe_so_a_dispersao_leva_uma_cor_e_nao_um_vetor() -> None:  # noqa: N802
    """**A mesma informação escrita de duas maneiras, e a medição está na docstring da função.**

    Medido em 11/09/2026 no `pos_hilic`: 50,9 ms com o vetor contra 14,1 ms com uma só, sobre 5.074
    pontos — e o caso de uma classe só é o **normal** deste corpus, porque zero sobrevivem ao FDR.
    O que este teste afirma é que as duas figuras são a mesma: as cores desenhadas são as da
    paleta, seja qual for o caminho.
    """
    paleta = cores_das_classes()

    uma = _pontos(qvalue_fdr=[0.9] * N_TESTADAS)
    assert set(uma.classes) == {Classe.NAO}, "âncora: há mais de uma classe, e o ramo não é este"
    cores_de_uma = (
        figura_do_volcano(
            pontos=uma, contraste=_contraste(), recorte=recorte_do_volcano(uma, _contraste())
        )
        .axes[0]
        .collections[0]
        .get_facecolor()
    )
    assert cores_de_uma.shape[0] == 1, "uma classe só e o `matplotlib` guardou uma cor por ponto"
    assert tuple(cores_de_uma[0]) == matplotlib.colors.to_rgba(paleta[Classe.NAO])

    varias = _pontos()
    assert len(set(varias.classes)) > 1, "âncora: há uma classe só, e o ramo não é este"
    cores_de_varias = _figura().axes[0].collections[0].get_facecolor()
    assert cores_de_varias.shape[0] == len(X_DESENHADOS)
    for i, classe in enumerate(varias.classes):
        assert tuple(cores_de_varias[i]) == matplotlib.colors.to_rgba(paleta[classe])


# ---------------------------------------------------------------------------
# 9. O CORPUS REAL — o `pos_hilic`, com as DUAS âncoras de não-vacuidade
# ---------------------------------------------------------------------------

MODO_REAL: Final = "pos_hilic"
PAR_REAL: Final = ("Asymptomatic", "HAMTSP")

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
def test_o_pos_hilic_real_atravessa_da_ingestao_ate_a_figura_VAZIA() -> None:  # noqa: N802
    """**Ponta a ponta sobre o corpus real, e a figura é vazia — é o resultado (regra 1 do
    README).**

    Zero features sobrevivem ao FDR num estudo de 18 contra 14 amostras depois da correção de
    múltiplos testes. Isso é o que este estudo diz, não um defeito do desenho.

    ## As DUAS âncoras de não-vacuidade vivem NESTA função

    *"Zero significativas"* é satisfeito por um array vazio, por um vetor todo `nan`, e por um laço
    que nunca corre. As duas âncoras separam os três:

    1. **a partição soma a população testada** — todo ponto desenhado mais todo indefinido dá o
       número de features da matriz processada, derivado da matriz e não do objeto que o construtor
       usou. Um array vazio dá `0 != 5.942`;
    2. **o menor `q` declarado na figura é o que o teste mediu** — derivado de
       `univariate_test().qvalue_fdr` por caminho independente do construtor. Um vetor todo `nan`
       dá `None` e não bate.

    ## O número NÃO é `0,9976`, e a distinção é de população

    `rotas.py` regista *"o menor `q` em 0,9976"* sobre **96 células do multiverso**; esta execução é
    **uma** célula, com a cascata em `none`, e mede outro valor. São populações diferentes, e
    congelar o primeiro aqui seria rótulo certo sobre a população errada. O que se afirma é a
    **igualdade entre o que a figura declara e o que o teste mede**, e que o valor está acima do
    corte.

    **As contagens são observação sobre ESTE arquivo, nomeadas, nunca invariante.**
    """
    matriz = load_matrix(
        measurements_path(MODO_REAL),
        sample_list_path(MODO_REAL),
        MODO_REAL,
        orientation="features-in-rows",
    )
    detalhada = process_matrix_detailed(matriz, PipelineSpec(**CAMPOS_DA_CASCATA_NULA))
    resultado = univariate_test(
        detalhada.matrix, PAR_REAL, fold_change_matrix=detalhada.pre_transform
    )
    bruto = _resumo_do_volcano(resultado, detalhada, PAR_REAL).model_dump()

    contraste = contraste_do_fio(**{c: bruto[c] for c in _CAMPOS_DO_CONTRASTE})
    pontos = pontos_do_volcano(
        features=bruto["features"],
        log2fc=bruto["log2fc"],
        pvalue=bruto["pvalue"],
        qvalue_fdr=bruto["qvalue_fdr"],
        n_features_testadas=bruto["n_features_testadas"],
        alfa=contraste.alfa,
    )
    figura = figura_do_volcano(
        pontos=pontos, contraste=contraste, recorte=recorte_do_volcano(pontos, contraste)
    )

    # ---- ÂNCORA 1: a partição soma a população testada, e ela é a da matriz processada ----
    testadas = detalhada.matrix.n_features
    assert testadas > 0, "âncora: a matriz processada ficou sem feature nenhuma"
    c = pontos.contagens
    assert pontos.n_testadas == testadas, (
        f"a figura declara {pontos.n_testadas} testadas e a matriz processada tem {testadas}"
    )
    assert c.desenhadas > 0, (
        "âncora: nenhum ponto foi desenhado — «zero significativas» passaria sobre um laço que "
        "nunca correu"
    )
    assert c.desenhadas + c.sem_fold_change + c.sem_pvalue_desenhavel == testadas, (
        f"a partição do DESENHO não fecha: {c} contra {testadas} features testadas. Alguma feature "
        "desapareceu entre o fio e o desenho, que é o que os sentinelas ±99 fazem ao contrário"
    )
    assert c.sobreviventes + c.nao_significativas + c.sem_qvalue == testadas, (
        f"a partição da CLASSIFICAÇÃO não fecha: {c} contra {testadas} features testadas"
    )
    assert series(figura.axes[0])[0].x.size == c.desenhadas, (
        "o número de pontos no artista não é o que a figura declara"
    )

    # ---- ÂNCORA 2: o menor `q` declarado é o MEDIDO, por caminho independente ----
    q = np.asarray(resultado.qvalue_fdr, dtype=float)
    finitos = q[np.isfinite(q)]
    assert finitos.size > 0, "âncora: nenhum `q` é finito — o mínimo seria `None` por vacuidade"
    medido = float(finitos.min())
    assert pontos.menor_q == pytest.approx(medido)
    assert decimal(medido) in _rodape(figura), (
        f"o rodapé não declara o menor q medido ({decimal(medido)}): {_rodape(figura)}"
    )

    # ---- E o resultado: ZERO sobrevivem, e o menor `q` está MUITO acima do corte ----
    #
    # **A contagem da figura é agora a da POPULAÇÃO e não a da nuvem**, e é por isso que ela pode
    # ser cruzada com a do núcleo: `isfinite(q) & (q < FDR_ALPHA)` é a expressão que `rotas.py`
    # grava em `n_significant_fdr`. Medido em 17/09/2026 neste corpus, o tamanho da distância é
    # **zero** — 868 features sem fold change, 1 sem `p`, e nenhuma delas sobrevive.
    definidos = np.isfinite(q)
    do_nucleo = int((q[definidos] < FDR_ALPHA).sum())
    assert c.sobreviventes == do_nucleo, (
        f"a figura conta {c.sobreviventes} sobreviventes e o núcleo {do_nucleo}"
    )
    assert c.sobreviventes == 0, (
        f"{c.sobreviventes} features sobrevivem ao FDR no {MODO_REAL} sob a cascata nula — a "
        "medição de `rotas.py` diz zero, e uma divergência aqui é achado e não bug"
    )
    assert c.sobreviventes_sem_ponto == 0, (
        "há sobrevivente sem ponto neste corpus, e a medição de 17/09/2026 diz zero — a distância "
        "entre a classificação e o desenho mudou, e isso é achado"
    )
    assert medido > FDR_ALPHA, (
        f"o menor q é {medido} e o corte é {FDR_ALPHA}: com zero significativas o mínimo tem de "
        "estar acima do corte, e se não estiver a classificação e a contagem discordam"
    )
    assert f"0 de {milhar(testadas)} sobrevive(m) ao FDR" in figura.axes[0].get_title()

    # ---- R8 inteiro na figura: os dois `n` e a correção, que separam efeito de poder ----
    rodape = _rodape(figura)
    rotulos = [amostra.group for amostra in detalhada.matrix.samples]
    for grupo in PAR_REAL:
        assert f"{grupo} (n = {milhar(rotulos.count(grupo))})" in rodape, rodape
    assert CORRECAO_DE_MULTIPLOS_TESTES in rodape

    # ---- R5b: o teste que a figura declara é o que de facto correu ----
    assert nome_do_teste(resultado.parametric) in rodape
    assert (MANN_WHITNEY if not resultado.parametric else WELCH) in rodape

    assert confere_vista(figura.axes[0]) == []


_CAMPOS_DO_CONTRASTE: Final = (
    "grupo_a",
    "grupo_b",
    "n_grupo_a",
    "n_grupo_b",
    "parametrico",
    "alfa_fdr",
    "correcao",
    "rota_da_normalidade",
    "shapiro_pvalue",
    "n_residuos",
    "matriz_do_pvalue",
    "matriz_do_fold_change",
    "matrizes_diferem",
)


@pytest.mark.slow
def test_as_contagens_da_FIGURA_batem_com_as_do_servidor_onde_sao_comparaveis() -> None:  # noqa: N802
    """**A âncora barata: alimentar o arnês com algo cuja resposta já se conhece.**

    `UnivariateCounts` é calculado no servidor e **não viaja** — quem decide o que é desenhável é a
    figura, e uma contagem vinda pronta seria uma segunda fonte que pode discordar do que está na
    tela sem nada acusar. Este teste é o que fecha essa lacuna.

    **E o que NÃO é comparável vai declarado:** `pvalue_undefined` do servidor conta TODAS as
    features sem `p`; `sem_pvalue_desenhavel` da figura conta só as que **têm** fold change, porque
    uma feature sem os dois não pode contar duas vezes sem a partição exceder a população. As duas
    são diferentes por desenho, e no `pos_hilic` a diferença é real — o teste afirma a desigualdade
    esperada em vez de a esconder.

    **`pvalue_indefinido` nasceu para fechar essa lacuna e é comparável**, e o teste afirma a
    igualdade: ele é incondicional, como o do servidor.
    """
    matriz = load_matrix(
        measurements_path(MODO_REAL),
        sample_list_path(MODO_REAL),
        MODO_REAL,
        orientation="features-in-rows",
    )
    detalhada = process_matrix_detailed(matriz, PipelineSpec(**CAMPOS_DA_CASCATA_NULA))
    resultado = univariate_test(
        detalhada.matrix, PAR_REAL, fold_change_matrix=detalhada.pre_transform
    )
    bruto = _resumo_do_volcano(resultado, detalhada, PAR_REAL).model_dump()
    pontos = pontos_do_volcano(
        features=bruto["features"],
        log2fc=bruto["log2fc"],
        pvalue=bruto["pvalue"],
        qvalue_fdr=bruto["qvalue_fdr"],
        n_features_testadas=bruto["n_features_testadas"],
        alfa=bruto["alfa_fdr"],
    )
    do_servidor = resultado.counts

    assert do_servidor.log2fc_undefined > 0, (
        "âncora: nenhuma feature tem fold change indefinido, e as comparações abaixo seriam vácuas"
    )
    assert pontos.contagens.sem_fold_change == do_servidor.log2fc_undefined
    assert pontos.contagens.sem_qvalue == do_servidor.qvalue_undefined
    assert pontos.contagens.pvalue_zero == do_servidor.pvalue_zero
    assert pontos.contagens.pvalue_indefinido == do_servidor.pvalue_undefined, (
        "a contagem INCONDICIONAL do `p` divergiu da do servidor — as duas são a mesma quantidade "
        "sobre a mesma população, e é para isso que ela existe"
    )
    assert pontos.n_testadas == do_servidor.n_features

    # A que NÃO é comparável, e a desigualdade é o que o desenho impõe.
    assert pontos.contagens.sem_pvalue_desenhavel <= do_servidor.pvalue_undefined, (
        "a contagem condicional da figura excedeu a incondicional do servidor — o escopo inverteu"
    )
    assert do_servidor.pvalue_undefined > pontos.contagens.sem_pvalue_desenhavel, (
        "âncora: as duas coincidem neste corpus, e a desigualdade acima passaria por vacuidade"
    )


@pytest.mark.slow
def test_a_rota_de_execucao_NAO_move_a_contagem_de_FDR_ao_passar_a_matriz_do_fold_change() -> None:  # noqa: N802
    """**O par causal na rota: `fold_change_matrix` alcança o x e NÃO alcança o `q`.**

    `n_significant_fdr` está **persistido** em `metrics_json`; se o parâmetro novo o movesse, esta
    unidade teria deslocado um número já gravado. A âncora é que as duas chamadas produzam
    `log2fc` **diferentes** — senão o par não separa nada.
    """
    matriz = load_matrix(
        measurements_path(MODO_REAL),
        sample_list_path(MODO_REAL),
        MODO_REAL,
        orientation="features-in-rows",
    )
    detalhada = process_matrix_detailed(matriz, PipelineSpec(**CAMPOS_DA_CASCATA_NULA))
    sem = univariate_test(detalhada.matrix, PAR_REAL)
    com = univariate_test(detalhada.matrix, PAR_REAL, fold_change_matrix=detalhada.pre_transform)

    assert not np.array_equal(sem.log2fc, com.log2fc, equal_nan=True) or not (
        detalhada.transform_moved_the_matrix
    ), "âncora: as duas chamadas deram o mesmo `log2fc` e a transformação MOVEU a matriz"
    assert np.array_equal(sem.pvalue, com.pvalue, equal_nan=True), (
        "`fold_change_matrix` moveu o `pvalue` — `n_significant_fdr` está persistido e sairia do "
        "lugar"
    )
    assert np.array_equal(sem.qvalue_fdr, com.qvalue_fdr, equal_nan=True), (
        "`fold_change_matrix` moveu o `q` — a contagem de FDR da aba de qualidade mudaria"
    )


# ---------------------------------------------------------------------------
# 10. O recorte é um `Recorte` como os outros — o rodapé cabe
# ---------------------------------------------------------------------------


def test_o_rodape_do_volcano_CABE_na_figura_padrao() -> None:  # noqa: N802
    """**`aplica` recusa o rodapé que precise de mais de `RESERVA_MAXIMA` da altura.**

    O método deste construtor é a frase mais longa do pacote — teste, rota de normalidade, correção,
    contagem, menor `q` e a divergência declarada contra o MetaboAnalyst. Que ele caiba na figura
    padrão não é óbvio, e a alternativa de `aplica` é levantar em vez de truncar.
    """
    figura = _figura()
    assert isinstance(recorte_do_volcano(_pontos(), _contraste()), Recorte)
    assert figura.texts, "o rodapé não foi desenhado"
    assert figura.subplotpars.bottom < 0.6, (
        f"o rodapé empurrou o fundo dos eixos para {figura.subplotpars.bottom:.2f} — sobra "
        "menos de metade da figura para o dado"
    )


# ---------------------------------------------------------------------------
# 11. O intervalo do fold change nas escolhidas
# ---------------------------------------------------------------------------
#
# **Oito features e `k = 3`, e cada valor existe para um desfecho.** As três de menor `p` são as
# posições 3, 1 e 5 — NÃO as três primeiras da matriz, que é a âncora contra uma seleção pela ordem
# em vez de pelo `p`. A 5 e a 6 EMPATAM no terceiro `p`, e a 5 entra pelo desempate da ordem; a 5
# não tem intervalo, então duas das três ganham barra; a 4 não tem `p` e não pode ser escolhida; e a
# 6 tem intervalo e fica de fora — uma barra nela seria a de uma feature que a regra não escolheu.
P_SEL: Final = (0.04, 0.001, 0.2, 0.0005, None, 0.003, 0.003, 0.5)
INF_SEL: Final = (-1.0, 0.5, -2.0, -3.5, None, None, 0.0, -1.0)
SUP_SEL: Final = (1.0, 2.5, 1.0, -0.5, None, None, 2.0, 1.0)
MOT_SEL: Final = (
    None,
    None,
    None,
    None,
    "order_statistic_not_positive",
    "median_over_imputed_values",
    None,
    None,
)
K_SEL: Final = 3
NIVEL_MARGINAL: Final = 0.95


def _escolhidas(**troca: Any) -> IntervalosEscolhidos | IntervaloAusente:
    campos: dict[str, Any] = {
        "pvalue": list(P_SEL),
        "log2fc_ci_lower": list(INF_SEL),
        "log2fc_ci_upper": list(SUP_SEL),
        "fold_change_ci_undefined": list(MOT_SEL),
        "nivel_do_intervalo": NIVEL_MARGINAL,
        "n_testadas": len(P_SEL),
        "k": K_SEL,
    }
    campos.update(troca)
    return intervalos_escolhidos(**campos)


def _z(nivel: float) -> float:
    """O quantil bilateral do nível, por `statistics.NormalDist` — não é o caminho do construtor."""
    return NormalDist().inv_cdf(1.0 - (1.0 - nivel) / 2.0)


def test_o_vocabulario_dos_motivos_e_o_do_NUCLEO_nas_duas_direcoes() -> None:  # noqa: N802
    """A fronteira da figura traduz os três códigos, e só eles, do lado de cá."""
    assert set(MOTIVOS_SEM_INTERVALO) == set(get_args(FoldChangeIntervalUndefined))


def test_as_escolhidas_sao_as_k_de_MENOR_P_e_so_as_com_intervalo_ganham_barra() -> None:  # noqa: N802
    """A regra do usuário — as `k` de menor `p` — e a partição das escolhidas, que fecha.

    **A âncora vem antes:** as escolhidas não são as `k` primeiras da matriz, senão uma seleção pela
    ordem passaria; e há escolhida com barra, senão as asserções sobre ela seriam vácuas.
    """
    i = _escolhidas()
    assert isinstance(i, IntervalosEscolhidos), i
    por_p = sorted((p, j) for j, p in enumerate(P_SEL) if p is not None)[:K_SEL]
    assert {j for _, j in por_p} != set(range(K_SEL)), "âncora: as de menor p são as primeiras"
    assert i.com_intervalo > 0, "âncora (AR208): nenhuma escolhida tem intervalo"

    assert i.indices.tolist() == [3, 1], "a barra vai às escolhidas COM intervalo, na ordem do p"
    assert i.escolhidas == K_SEL
    assert i.m == len(P_SEL)
    assert i.sem_intervalo == ((MOTIVOS_SEM_INTERVALO["median_over_imputed_values"], 1),)
    assert i.com_intervalo + sum(n for _, n in i.sem_intervalo) == i.escolhidas, (
        "a partição das escolhidas não fecha"
    )
    # A 6 empata com a 5 no terceiro p e fica de fora pelo desempate da ordem da matriz.
    assert i.empatadas_fora == 1
    assert 6 not in i.indices.tolist()


def test_o_nivel_e_o_de_BENJAMINI_YEKUTIELI_e_a_barra_e_a_marginal_ESTICADA() -> None:  # noqa: N802
    """`1 - R·fcr/m`, com `R` as barras e `m` as testadas; a meia-largura cresce pela razão dos `z`.

    **A planta é o próprio intervalo marginal:** se o ajuste não se aplicasse, os limites seriam os
    do fio, e a âncora afirma que os dois diferem neste dado.
    """
    i = _escolhidas()
    assert isinstance(i, IntervalosEscolhidos)
    fcr = 1.0 - NIVEL_MARGINAL
    assert i.nivel_marginal == NIVEL_MARGINAL
    assert i.nivel == pytest.approx(1.0 - 2 * fcr / len(P_SEL), abs=1e-15)
    fator = _z(i.nivel) / _z(NIVEL_MARGINAL)
    assert fator > 1.0, "âncora: o ajuste não alarga, e a planta do marginal não separaria"
    for posicao, j in enumerate(i.indices.tolist()):
        inferior, superior = INF_SEL[j], SUP_SEL[j]
        assert inferior is not None and superior is not None
        centro, meia = (inferior + superior) / 2.0, (superior - inferior) / 2.0 * fator
        assert i.inferior[posicao] == pytest.approx(centro - meia, abs=1e-12)
        assert i.superior[posicao] == pytest.approx(centro + meia, abs=1e-12)
        assert i.inferior[posicao] < inferior and i.superior[posicao] > superior


def _par_sintetico() -> RefinoMatrix:
    """Dois grupos lognormais de 18 e 14 amostras, 40 features, valores todos positivos."""
    gerador = np.random.default_rng(20261106)
    a = gerador.lognormal(mean=1.0, sigma=0.6, size=(18, 40))
    b = gerador.lognormal(mean=0.8, sigma=0.6, size=(14, 40))
    grupos = ["A"] * 18 + ["B"] * 14
    return RefinoMatrix(
        values=np.vstack([a, b]),
        feature_names=tuple(f"f{j}" for j in range(40)),
        samples=tuple(
            SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type="sample", group=g)
            for i, g in enumerate(grupos)
        ),
        mode="sintetico",
    )


def test_a_barra_ajustada_e_o_intervalo_que_o_NUCLEO_calcula_no_nivel_ajustado(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**O oráculo é `fold_change_interval` corrido no nível ajustado**, não a minha conta.

    A figura não recalcula o intervalo: recebe o de 95% do fio e alarga-o. O núcleo, com o nível
    trocado, calcula-o de novo a partir dos dados — e os dois têm de dar o mesmo número, porque o
    intervalo de Bonett-Price é simétrico em log e o erro-padrão não depende do nível.
    """
    matriz = _par_sintetico()
    imputadas = np.zeros(matriz.values.shape, dtype=bool)
    marginal = fold_change_interval(matriz, ("A", "B"), imputadas)
    resultado = univariate_test(matriz, ("A", "B"), parametric=True)
    i = intervalos_escolhidos(
        pvalue=resultado.pvalue,
        log2fc_ci_lower=marginal.log2_lower,
        log2fc_ci_upper=marginal.log2_upper,
        fold_change_ci_undefined=list(marginal.undefined),
        nivel_do_intervalo=marginal.confidence,
        n_testadas=matriz.n_features,
    )
    assert isinstance(i, IntervalosEscolhidos)
    assert i.com_intervalo == K_DAS_ESCOLHIDAS, "âncora: as 20 escolhidas têm intervalo"

    monkeypatch.setattr(nucleo_univariado, "FOLD_CHANGE_CONFIDENCE", i.nivel)
    ajustado = fold_change_interval(matriz, ("A", "B"), imputadas)
    assert ajustado.confidence == i.nivel, "a planta do nível não aterrou no núcleo"
    assert not np.allclose(marginal.log2_lower[i.indices], ajustado.log2_lower[i.indices]), (
        "âncora: o nível ajustado não mexe no intervalo, e a igualdade abaixo seria vácua"
    )
    np.testing.assert_allclose(i.inferior, ajustado.log2_lower[i.indices], rtol=0, atol=1e-9)
    np.testing.assert_allclose(i.superior, ajustado.log2_upper[i.indices], rtol=0, atol=1e-9)


def test_o_motivo_de_FORA_do_vocabulario_sai_CRU_e_o_ausente_sai_dito() -> None:  # noqa: N802
    """Um código que a figura não conhece não cai num `else` com cara de outro motivo."""
    motivos: list[str | None] = list(MOT_SEL)
    motivos[5] = "codigo_novo"
    i = _escolhidas(fold_change_ci_undefined=motivos)
    assert isinstance(i, IntervalosEscolhidos)
    assert len(i.sem_intervalo) == 1 and "codigo_novo" in i.sem_intervalo[0][0], i.sem_intervalo
    assert i.sem_intervalo[0][0] not in MOTIVOS_SEM_INTERVALO.values()

    motivos[5] = None
    i = _escolhidas(fold_change_ci_undefined=motivos)
    assert isinstance(i, IntervalosEscolhidos)
    assert i.sem_intervalo == ((SEM_MOTIVO_NO_FIO, 1),)


def test_com_menos_p_que_k_escolhe_TODAS_as_que_tem_p() -> None:  # noqa: N802
    i = _escolhidas(k=K_DAS_ESCOLHIDAS)
    assert isinstance(i, IntervalosEscolhidos)
    com_p = sum(1 for p in P_SEL if p is not None)
    assert com_p < K_DAS_ESCOLHIDAS, "âncora: este dado tem p para mais que k features"
    assert i.escolhidas == com_p
    assert i.empatadas_fora == 0


@pytest.mark.parametrize("nivel", [None, 0.0, 1.0, 1.5, -0.1, True, "0.95", float("nan")])
def test_sem_nivel_VALIDO_o_intervalo_e_AUSENTE_e_nao_inventado(nivel: Any) -> None:  # noqa: N802
    """O nível que calculou os limites viaja no fio; sem ele não há como os alargar.

    A outra direção é o controle: o mesmo dado com `0,95` produz as escolhidas.
    """
    assert isinstance(_escolhidas(), IntervalosEscolhidos), "âncora: o controle não escolhe"
    ausente = _escolhidas(nivel_do_intervalo=nivel)
    assert isinstance(ausente, IntervaloAusente), ausente
    assert ausente.motivo.strip()


# --- as barras NO VOLCANO ---------------------------------------------------
#
# **`k = 2` sobre as seis features do módulo:** as duas de menor `p` são a 0 e a 1, e a 2, a 3 e a 5
# também têm intervalo — a âncora de que a barra não vai a toda feature com intervalo.
INF_V: Final = (-4.0, 1.5, -1.0, 0.25, None, 3.0)
SUP_V: Final = (-2.0, 3.5, 1.0, 2.25, None, 5.0)
MOT_V: Final = (None, None, None, None, "order_statistic_not_positive", None)
K_V: Final = 2


def _intervalos_v(**troca: Any) -> IntervalosEscolhidos | IntervaloAusente:
    campos: dict[str, Any] = {
        "pvalue": list(PVALUE),
        "log2fc_ci_lower": list(INF_V),
        "log2fc_ci_upper": list(SUP_V),
        "fold_change_ci_undefined": list(MOT_V),
        "nivel_do_intervalo": NIVEL_MARGINAL,
        "n_testadas": N_TESTADAS,
        "k": K_V,
    }
    campos.update(troca)
    return intervalos_escolhidos(**campos)


def _figura_com_barras(
    intervalos: IntervalosEscolhidos | IntervaloAusente, pontos: PontosDoVolcano | None = None
) -> Figure:
    p = pontos if pontos is not None else _pontos()
    c = _contraste()
    return figura_do_volcano(
        pontos=p,
        contraste=c,
        recorte=recorte_do_volcano(p, c, intervalos=intervalos),
        intervalos=intervalos,
    )


def _barras_esperadas(
    i: IntervalosEscolhidos, pvalue: tuple[float | None, ...]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Os segmentos horizontais, um por escolhida com barra, separados por `nan` — da ENTRADA."""
    x: list[float] = []
    y: list[float] = []
    for posicao, j in enumerate(i.indices.tolist()):
        p = pvalue[j]
        assert p is not None
        x += [float(i.inferior[posicao]), float(i.superior[posicao]), float("nan")]
        y += [_y(p), _y(p), float("nan")]
    return tuple(x), tuple(y)


def test_o_volcano_poe_a_barra_SO_nas_escolhidas_e_passa_nas_quatro_classes() -> None:  # noqa: N802
    i = _intervalos_v()
    assert isinstance(i, IntervalosEscolhidos)
    com_intervalo = [j for j, v in enumerate(INF_V) if v is not None]
    assert len(com_intervalo) > i.com_intervalo > 0, (
        "âncora: toda feature com intervalo seria escolhida, e a seleção não separaria"
    )
    figura = _figura_com_barras(i)
    eixo = figura.axes[0]
    esperado = {**_esperado(), ROTULO_DAS_BARRAS: _barras_esperadas(i, PVALUE)}
    assert confere_series(eixo, esperado) == []
    assert confere_vista(eixo) == []
    legenda = [t.get_text() for t in eixo.get_legend().get_texts()]
    assert rotulo_das_barras(i.escolhidas, i.com_intervalo) in legenda, legenda


def test_a_planta_da_barra_numa_NAO_escolhida_reprova() -> None:  # noqa: N802
    """O par causal da seleção na figura: as mesmas barras com `k` maior deixam de bater."""
    i = _intervalos_v()
    assert isinstance(i, IntervalosEscolhidos)
    plantada = _intervalos_v(k=K_V + 2)
    assert isinstance(plantada, IntervalosEscolhidos)
    assert plantada.com_intervalo > i.com_intervalo, "âncora: a planta não acrescenta barra"
    eixo = _figura_com_barras(plantada).axes[0]
    esperado = {**_esperado(), ROTULO_DAS_BARRAS: _barras_esperadas(i, PVALUE)}
    achados = confere_series(eixo, esperado)
    assert any(ROTULO_DAS_BARRAS in a for a in achados), achados


def test_o_rodape_do_volcano_diz_a_REGRA_o_NIVEL_e_o_AJUSTE() -> None:  # noqa: N802
    i = _intervalos_v()
    assert isinstance(i, IntervalosEscolhidos)
    figura = _figura_com_barras(i)
    rodape = _rodape(figura)
    fcr = decimal(1.0 - NIVEL_MARGINAL)
    for trecho in (
        f"nas {i.com_intervalo} das {i.escolhidas} de menor p",
        f"{decimal(i.nivel * 100.0, 5)}%",
        f"FCR de Benjamini-Yekutieli a {fcr}",
        "Bonett-Price da razão de medianas",
        f"outro teste, sobre a {MATRIZ_DO_PVALUE}",
    ):
        assert trecho in rodape, (trecho, rodape)
    # Classe (d): todo número da figura tem procedência, os novos inclusive.
    novos = [milhar(i.com_intervalo), milhar(i.escolhidas), decimal(i.nivel * 100.0, 5), fcr]
    assert confere_numeros(figura, [*_numeros_esperados(), *novos]) == []


def test_a_escolhida_de_p_ZERO_tem_intervalo_e_NAO_tem_barra_no_volcano() -> None:  # noqa: N802
    """`-log10(0)` não tem altura: a barra não tem onde ficar, e a falta é contada e dita."""
    pvalue: list[float | None] = list(PVALUE)
    pvalue[0] = 0.0
    i = _intervalos_v(pvalue=pvalue)
    assert isinstance(i, IntervalosEscolhidos)
    assert 0 in i.indices.tolist(), "âncora: a feature de p zero não foi escolhida"
    figura = _figura_com_barras(i, _pontos(pvalue=pvalue))
    barras = next(s for s in series(figura.axes[0]) if s.rotulo == ROTULO_DAS_BARRAS)
    assert int(np.isfinite(barras.x).sum()) == 2 * (i.com_intervalo - 1)
    assert "1 sem altura (p zero)" in _rodape(figura)
    legenda = [t.get_text() for t in figura.axes[0].get_legend().get_texts()]
    assert rotulo_das_barras(i.escolhidas, i.com_intervalo - 1) in legenda, legenda


def test_a_barra_entra_ANTES_da_nuvem_e_o_clique_no_ponto_e_da_nuvem() -> None:  # noqa: N802
    """O `pick` do `matplotlib` corre os filhos na ordem de inserção, e o último a responder fica.

    A barra passa pelo ponto: inserida depois da nuvem, um clique no ponto mais forte anotaria a
    barra em vez da feature. A ordem de DESENHO vem do `zorder`, e a barra fica por cima.
    """
    i = _intervalos_v()
    assert isinstance(i, IntervalosEscolhidos)
    eixo = _figura_com_barras(i).axes[0]
    filhos = eixo.get_children()
    barras = next(a for a in eixo.lines if a.get_label() == ROTULO_DAS_BARRAS)
    nuvem = next(a for a in eixo.collections if a.get_label() == ROTULO_DA_SERIE)
    assert filhos.index(barras) < filhos.index(nuvem)
    assert barras.get_zorder() > nuvem.get_zorder(), "a barra fica por baixo dos pontos"


def test_sem_intervalo_PEDIDO_o_volcano_e_o_de_antes_e_AUSENTE_e_dito() -> None:  # noqa: N802
    """`None` é «esta chamada não pede barras»; `IntervaloAusente` é «pediu e o fio não trouxe»."""
    antes = _rodape(_figura())
    ausente = _rodape(_figura_com_barras(_intervalos_v(nivel_do_intervalo=None)))
    assert ausente.startswith(antes) and ausente != antes and "Barras" in ausente, ausente
    assert ROTULO_DAS_BARRAS not in {s.rotulo for s in series(_figura().axes[0])}

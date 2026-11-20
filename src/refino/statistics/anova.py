"""ANOVA de uma via e Kruskal-Wallis sobre N grupos, com post-hoc.

A correção é em duas etapas, como no MetaboAnalystR: Benjamini-Hochberg sobre o vetor de `p` de
todas as features, filtro pelo corte, e só então o post-hoc nas sobreviventes. O post-hoc
paramétrico não recebe FDR outra vez, porque Tukey e Fisher já têm política própria de erro por
família: Tukey a controla sobre todos os pares pela distribuição do intervalo estudentizado, e
Fisher declaradamente não a controla. Por isso `PostHocComparison` não tem campo de `q`, e
`AnalysisSpec.post_hoc` tem `tukey` como padrão.

O Fisher usa o quadrado médio residual agrupado: sobre todos os grupos da ANOVA daquela feature,
com `df = N - k`, e não recalculado a partir dos dois grupos do par, com `df = n_a + n_b - 2`.
Sobre dado homocedástico os dois coincidem; com variâncias diferentes entre grupos, os vereditos
podem ser opostos.

A população são as linhas de `sample_mask`, as amostras de estudo. QC fica fora porque é a régua
e não um grupo, e `validate` fica fora porque não é amostra de estudo.

`kruskal_wallis` é o caminho não paramétrico. O ramo não paramétrico do MetaboAnalystR
(`R/stats_univariates.R`) não roda post-hoc nenhum; aqui o post-hoc é o de Dunn. A correção de
empates do `H`, o fator `1 - Σ(t³-t)/(N³-N)`, é necessária neste tipo de dado: zero é o valor
que o equipamento deixa quando não detecta, então empate é o caso normal. Sem o fator o `p`
sairia maior do que deve, porque o fator é menor ou igual a 1 e dividir por ele só aumenta o
`H`. Quem o aplica é `scipy.stats.kruskal`.

Dunn é um `z` normal por par, sem proteção por família, então recebe Benjamini-Hochberg sobre as
`k(k-1)/2` comparações de cada feature, em `DunnComparison.qvalue_fdr`. É outra família que a de
`AnovaResult.qvalue_fdr`, que é entre features, e não um segundo ajuste da mesma.

Não há Dunn no `scipy` nem no `sklearn`. A implementação é conferida por uma identidade: com
dois grupos, o `z` de Dunn com correção de empates é o `z` assintótico de Mann-Whitney sem
correção de continuidade, porque a diferença das médias dos postos é
`(N/(n1*n2))*(U - n1*n2/2)` e o termo de empates é o mesmo.
`test_o_dunn_com_dois_grupos_bate_com_mann_whitney_assintotico` confere a igualdade.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import f_oneway, kruskal, rankdata, tukey_hsd
from scipy.stats import norm as distribuicao_normal
from scipy.stats import t as distribuicao_t

from refino.domain.cancellation import CancelCheck, raise_if_cancelled
from refino.domain.labels import PostHoc, validate_label
from refino.domain.matrix import RefinoMatrix
from refino.domain.statistics import (
    AnovaCounts,
    AnovaResult,
    DunnComparison,
    PostHocComparison,
    PostHocDecision,
    PostHocRoute,
)
from refino.statistics.univariate import MIN_AMOSTRAS_POR_GRUPO, benjamini_hochberg

__all__ = ["FDR_ALPHA", "MIN_GRUPOS", "kruskal_wallis", "one_way_anova"]

FDR_ALPHA = 0.05
"""Corte do `q` de Benjamini-Hochberg que decide quais features vão ao post-hoc.

`api/rotas.py` tem uma constante com o mesmo valor e conta significativas com o mesmo operador
estrito, `q < FDR_ALPHA`. O número vive duas vezes porque o núcleo puro não importa da
fronteira. `0,05` é a convenção da área, e não uma derivação.
"""

MIN_GRUPOS = 2
"""Com menos de dois grupos não há contraste."""


def one_way_anova(
    matrix: RefinoMatrix,
    *,
    post_hoc: PostHoc = "tukey",
    fdr_alpha: float = FDR_ALPHA,
    cancelled: CancelCheck | None = None,
) -> AnovaResult:
    """ANOVA de uma via por feature sobre todos os grupos da matriz, com post-hoc.

    O conjunto de grupos não é parâmetro: é derivado da matriz. Quem quiser um par usa
    `univariate_test`.

    `cancelled` é consultado dentro do laço do post-hoc, que corre por feature e não é
    vetorizado. A ANOVA é uma chamada vetorizada sobre a tabela inteira e não tem laço onde
    parar, então sem sobrevivente `cancelled` nunca é consultado.

    O custo está no post-hoc de `tukey`: `scipy.stats.tukey_hsd` integra numericamente a
    distribuição do intervalo estudentizado e custa cerca de 300 vezes um `f_oneway` sobre os
    mesmos blocos. Com 214 sobreviventes e quatro grupos, o post-hoc de `tukey` leva cerca de
    28 s e o de `fisher` menos de 0,1 s.

    Levanta `ValueError` quando a população não sustenta o teste (menos de dois grupos, ou
    grupo com menos de `MIN_AMOSTRAS_POR_GRUPO` amostras) e `InvalidLabelError` quando
    `post_hoc` não pertence ao vocabulário.
    """
    # A guarda é necessária porque `_comparacoes_da_feature` trata como Fisher tudo o que não é
    # `tukey`: sem ela, um rótulo inválido correria Fisher e o resultado declararia o nome
    # pedido.
    validate_label("PostHoc", post_hoc)
    if not 0.0 < fdr_alpha <= 1.0:
        raise ValueError(
            f"`fdr_alpha` é {fdr_alpha} e tem de estar em (0, 1]. Em zero nenhuma feature "
            "sobrevive e o post-hoc nunca corre; acima de 1 o corte deixa de ser um corte."
        )

    grupos, blocos = _populacao(matrix)
    n_features = matrix.n_features

    # Quantas observações utilizáveis cada grupo tem em cada feature, em `(n_features, n_grupos)`.
    utilizaveis = np.stack([np.isfinite(bloco).sum(axis=0) for bloco in blocos], axis=1)
    # O limiar por feature é o mesmo do global. Uma feature em que um grupo não chega ao mínimo
    # teria `F` sobre uma população diferente da declarada; ela não é testada, fica `nan` e é
    # contada em `insufficient_groups`.
    testavel = np.asarray((utilizaveis >= MIN_AMOSTRAS_POR_GRUPO).all(axis=1), dtype=bool)

    statistic = np.full(n_features, np.nan, dtype=np.float64)
    pvalue = np.full(n_features, np.nan, dtype=np.float64)
    if bool(testavel.any()):
        # As colunas não testáveis ficam fora da chamada, em vez de mascaradas depois: com elas
        # o `scipy` emitiria `SmallSampleWarning` por amostra vazia. Com o pré-filtro nenhum
        # aviso dispara, e só o `errstate` fica.
        with np.errstate(invalid="ignore", divide="ignore"):
            bruto = f_oneway(*(bloco[:, testavel] for bloco in blocos), axis=0, nan_policy="omit")
        statistic[testavel] = np.atleast_1d(np.asarray(bruto.statistic, dtype=np.float64))
        pvalue[testavel] = np.atleast_1d(np.asarray(bruto.pvalue, dtype=np.float64))

    qvalue = benjamini_hochberg(pvalue)
    sobreviveu = np.isfinite(qvalue) & (qvalue < fdr_alpha)

    comparacoes = _post_hoc(
        feature_names=matrix.feature_names,
        grupos=grupos,
        blocos=blocos,
        sobreviveu=sobreviveu,
        metodo=post_hoc,
        cancelled=cancelled,
    )

    return AnovaResult(
        feature=matrix.feature_names,
        statistic=statistic,
        pvalue=pvalue,
        qvalue_fdr=qvalue,
        groups=grupos,
        n_per_group=tuple(int(bloco.shape[0]) for bloco in blocos),
        n_usable_per_group=utilizaveis,
        method="anova",
        fdr_alpha=fdr_alpha,
        min_per_group=MIN_AMOSTRAS_POR_GRUPO,
        provenance=matrix.provenance,
        counts=_contagens(statistic, pvalue, qvalue, testavel, sobreviveu),
        post_hoc=PostHocDecision(route=_rota(pvalue, sobreviveu), method=post_hoc),
        comparisons=comparacoes,
    )


# ---------------------------------------------------------------------------
# População
# ---------------------------------------------------------------------------


def _populacao(
    matrix: RefinoMatrix,
    *,
    teste: str = "a ANOVA",
    porque_dois: str = (
        "com uma observação não há dispersão dentro do grupo, e o quadrado médio residual "
        "fica sem contribuição dela"
    ),
) -> tuple[tuple[str, ...], list[np.ndarray]]:
    """Os grupos em ordem canônica e os blocos de linhas de cada um, com as duas recusas.

    A ordem é `sorted`, e não a de aparição, porque ela decide a ordem das comparações do
    post-hoc e a das cores da figura: a de aparição faria o mesmo grupo mudar de cor entre duas
    matrizes que diferem só na ordem das injeções.

    `teste` e `porque_dois` são texto da mensagem de recusa. `kruskal_wallis` partilha a guarda
    e não a justificativa: o argumento do quadrado médio residual vale para a ANOVA e não para o
    Kruskal-Wallis.
    """
    estudo = matrix.sample_mask
    rotulos = np.asarray(matrix.groups)[estudo]
    valores = np.asarray(matrix.values, dtype=np.float64)[estudo]

    grupos = tuple(sorted(set(rotulos.tolist())))
    if len(grupos) < MIN_GRUPOS:
        raise ValueError(
            f"{teste} precisa de ao menos {MIN_GRUPOS} grupos e esta matriz tem {len(grupos)}: "
            f"{list(grupos)}. A população são as amostras de estudo — QC e `validate` ficam de "
            "fora, e a matriz pode ter rótulos que não chegaram aqui por isso."
        )

    blocos = [valores[rotulos == nome] for nome in grupos]
    for nome, bloco in zip(grupos, blocos, strict=True):
        if bloco.shape[0] < MIN_AMOSTRAS_POR_GRUPO:
            raise ValueError(
                f"grupo {nome!r} tem {bloco.shape[0]} amostra(s) de estudo; {teste} exige ao "
                f"menos {MIN_AMOSTRAS_POR_GRUPO} por grupo — {porque_dois}."
            )
    return grupos, blocos


# ---------------------------------------------------------------------------
# Contagens e rota
# ---------------------------------------------------------------------------


def _contagens(
    statistic: np.ndarray,
    pvalue: np.ndarray,
    qvalue: np.ndarray,
    testavel: np.ndarray,
    sobreviveu: np.ndarray,
) -> AnovaCounts:
    """A partição das features indefinidas, exaustiva por construção.

    `constant` é derivado de `testavel & isnan(statistic)`, e não de uma segunda medição da
    variância. Entre as colunas testáveis, `nan` só aparece por `0/0`, com quadrado médio entre
    grupos e residual ambos nulos, que é feature constante. Variância residual nula com
    diferença entre grupos dá `+inf`, que é medida e conta em `pvalue_zero`.
    """
    indefinido = np.isnan(statistic)
    return AnovaCounts(
        n_features=int(statistic.size),
        statistic_undefined=int(indefinido.sum()),
        pvalue_undefined=int(np.isnan(pvalue).sum()),
        # `== 0.0` e não `< eps`, como em `UnivariateCounts`: só o zero exato vai para `+inf` no
        # `-log10`, e um limiar contaria features que a figura desenha sem problema.
        pvalue_zero=int((pvalue == 0.0).sum()),
        qvalue_undefined=int((~np.isfinite(qvalue)).sum()),
        constant=int((testavel & indefinido).sum()),
        insufficient_groups=int((~testavel).sum()),
        survivors=int(sobreviveu.sum()),
    )


def _rota(pvalue: np.ndarray, sobreviveu: np.ndarray) -> PostHocRoute:
    """Os três desfechos do post-hoc. O terceiro distingue 'nada testável' de 'nada sobreviveu'."""
    if bool(sobreviveu.any()):
        return "ran"
    # `isfinite` e não `~isnan`: é o mesmo predicado que `benjamini_hochberg` usa para montar o
    # denominador do ajuste, então "nada testável" aqui significa "o `n` do BH era zero".
    if not bool(np.isfinite(pvalue).any()):
        return "nothing_testable"
    return "no_survivors"


# ---------------------------------------------------------------------------
# Post-hoc
# ---------------------------------------------------------------------------


def _post_hoc(
    *,
    feature_names: tuple[str, ...],
    grupos: tuple[str, ...],
    blocos: list[np.ndarray],
    sobreviveu: np.ndarray,
    metodo: PostHoc,
    cancelled: CancelCheck | None,
) -> tuple[PostHocComparison, ...]:
    """As comparações par a par, só nas features que sobreviveram ao FDR.

    O laço é por feature e não é vetorizado. A checagem de cancelamento vai no topo de cada
    iteração, para o intervalo entre dois pontos de parada ser o custo de uma feature.
    """
    comparacoes: list[PostHocComparison] = []
    for indice in np.flatnonzero(sobreviveu).tolist():
        raise_if_cancelled(cancelled, "anova: laco do post-hoc")
        amostras = _amostras_utilizaveis(blocos, indice)
        comparacoes.extend(_comparacoes_da_feature(feature_names[indice], grupos, amostras, metodo))
    return tuple(comparacoes)


def _amostras_utilizaveis(blocos: list[np.ndarray], indice: int) -> list[np.ndarray]:
    """As observações utilizáveis de uma feature, grupo a grupo.

    `nan` sai aqui, e não por política de biblioteca: `tukey_hsd` não tem `nan_policy`, e
    `rankdata` ordena `nan` no topo sem se queixar.

    Os dois post-hoc, o paramétrico e o de Dunn, usam este mesmo recorte, e o critério de
    utilizável é o mesmo que conta `n_usable_per_group`.
    """
    return [bloco[np.isfinite(bloco[:, indice]), indice].astype(np.float64) for bloco in blocos]


def _comparacoes_da_feature(
    nome: str, grupos: tuple[str, ...], amostras: list[np.ndarray], metodo: PostHoc
) -> list[PostHocComparison]:
    """Os `k(k-1)/2` pares de uma feature, na ordem canônica dos grupos.

    O quadrado médio residual é calculado uma vez, sobre todos os grupos, e serve os dois
    métodos. É a definição de agrupado; recalculá-lo por par daria outro número.
    """
    medias = np.array([amostra.mean() for amostra in amostras], dtype=np.float64)
    tamanhos = np.array([amostra.size for amostra in amostras], dtype=np.int64)
    soma_quadrados = float(sum(float(((a - a.mean()) ** 2).sum()) for a in amostras))
    graus = int(tamanhos.sum()) - len(amostras)
    quadrado_medio = soma_quadrados / graus if graus > 0 else np.nan

    pares = list(itertools.combinations(range(len(grupos)), 2))
    diferencas = [float(medias[i] - medias[j]) for i, j in pares]

    if not np.isfinite(quadrado_medio) or quadrado_medio <= 0.0:
        # Variância residual nula é separação exata, e não erro: é a feature cujo `F` é `+inf` e
        # cujo `p` da ANOVA é zero. `tukey_hsd` dividiria por zero aqui. Diferença não nula
        # separa com `p = 0`; diferença nula é `0/0` e não tem `p`.
        pvalores = [0.0 if diferenca != 0.0 else np.nan for diferenca in diferencas]
    elif metodo == "tukey":
        with np.errstate(invalid="ignore", divide="ignore"):
            estudentizado = tukey_hsd(*amostras)
        matriz_p = np.asarray(estudentizado.pvalue, dtype=np.float64)
        pvalores = [float(matriz_p[i, j]) for i, j in pares]
    else:
        pvalores = [
            _fisher_lsd(
                diferenca=diferenca,
                quadrado_medio=quadrado_medio,
                graus=graus,
                n_a=int(tamanhos[i]),
                n_b=int(tamanhos[j]),
            )
            for diferenca, (i, j) in zip(diferencas, pares, strict=True)
        ]

    return [
        PostHocComparison(
            feature=nome,
            group_a=grupos[i],
            group_b=grupos[j],
            difference=diferenca,
            pvalue=p,
        )
        for (i, j), diferenca, p in zip(pares, diferencas, pvalores, strict=True)
    ]


def _fisher_lsd(
    *, diferenca: float, quadrado_medio: float, graus: int, n_a: int, n_b: int
) -> float:
    """`p` bilateral do `t` par a par, com o quadrado médio residual agrupado da ANOVA inteira.

    `quadrado_medio` vem de todos os `k` grupos e `graus` é `N - k`. Recalcular a variância só a
    partir de `a` e `b`, com `n_a + n_b - 2` graus, daria outro número, e sobre grupos de
    variâncias diferentes os vereditos divergem.
    """
    erro = float(np.sqrt(quadrado_medio * (1.0 / n_a + 1.0 / n_b)))
    t = diferenca / erro
    return float(2.0 * distribuicao_t.sf(abs(t), graus))


# ---------------------------------------------------------------------------
# Kruskal-Wallis
# ---------------------------------------------------------------------------


def kruskal_wallis(
    matrix: RefinoMatrix,
    *,
    fdr_alpha: float = FDR_ALPHA,
    cancelled: CancelCheck | None = None,
) -> AnovaResult:
    """Kruskal-Wallis por feature sobre todos os grupos da matriz, com post-hoc de Dunn.

    É a alternativa não paramétrica a `one_way_anova`, com a mesma população, a mesma política
    de grupo insuficiente por feature e as mesmas duas etapas. Muda o que se mede, `H` em vez de
    `F`, sobre postos em vez de valores, e o post-hoc, que é Dunn com Benjamini-Hochberg sobre
    as comparações da feature.

    Não há parâmetro `post_hoc`: Tukey e Fisher trabalham sobre o quadrado médio residual da
    ANOVA, que este caminho não calcula. `PostHocDecision.method` declara que foi Dunn.

    O resultado é `AnovaResult` com `method="kruskal_wallis"`. O campo `statistic` carrega o
    `H`, cuja distribuição nula é qui-quadrado com `k-1` graus, e não `F(k-1, N-k)`; é preciso
    ler `method` para saber qual dos dois é.

    `cancelled` é consultado dentro do laço do post-hoc, como em `one_way_anova`: sem
    sobrevivente, nunca é consultado.

    Levanta `ValueError` quando a população não sustenta o teste (menos de dois grupos, grupo
    com menos de `MIN_AMOSTRAS_POR_GRUPO` amostras) ou quando `fdr_alpha` está fora de `(0, 1]`.
    """
    if not 0.0 < fdr_alpha <= 1.0:
        raise ValueError(
            f"`fdr_alpha` é {fdr_alpha} e tem de estar em (0, 1]. Em zero nenhuma feature "
            "sobrevive e o post-hoc nunca corre; acima de 1 o corte deixa de ser um corte."
        )

    grupos, blocos = _populacao(
        matrix,
        teste="o Kruskal-Wallis",
        porque_dois=(
            "com uma observação o grupo contribui um posto e nenhuma dispersão de postos, e a "
            "soma de postos dele não distingue posição de acaso"
        ),
    )
    # Não finito vira `nan` antes de medir. `utilizaveis` conta por `isfinite`, e
    # `nan_policy="omit"` só descarta `nan`: um `+inf` ficaria fora da contagem e dentro da
    # ordenação, onde `rankdata` o põe no topo. `_amostras_utilizaveis` já filtra por `isfinite`,
    # então sem isto a estatística e o post-hoc correriam sobre populações diferentes.
    blocos = [np.where(np.isfinite(bloco), bloco, np.nan) for bloco in blocos]
    n_features = matrix.n_features

    utilizaveis = np.stack([np.isfinite(bloco).sum(axis=0) for bloco in blocos], axis=1)
    testavel = np.asarray((utilizaveis >= MIN_AMOSTRAS_POR_GRUPO).all(axis=1), dtype=bool)
    # A coluna constante é excluída antes da chamada. Com todas as observações empatadas o fator
    # de empates é zero e o `scipy` divide o `H` por ele: devolve `nan`, que é o desfecho certo,
    # mas com `RuntimeWarning`.
    constante = _colunas_constantes(blocos, testavel)
    alvo = testavel & ~constante

    hstat = np.full(n_features, np.nan, dtype=np.float64)
    pvalue = np.full(n_features, np.nan, dtype=np.float64)
    if bool(alvo.any()):
        bruto = kruskal(*(bloco[:, alvo] for bloco in blocos), axis=0, nan_policy="omit")
        hstat[alvo] = np.atleast_1d(np.asarray(bruto.statistic, dtype=np.float64))
        pvalue[alvo] = np.atleast_1d(np.asarray(bruto.pvalue, dtype=np.float64))

    qvalue = benjamini_hochberg(pvalue)
    sobreviveu = np.isfinite(qvalue) & (qvalue < fdr_alpha)

    comparacoes = _post_hoc_de_dunn(
        feature_names=matrix.feature_names,
        grupos=grupos,
        blocos=blocos,
        sobreviveu=sobreviveu,
        cancelled=cancelled,
    )

    return AnovaResult(
        feature=matrix.feature_names,
        statistic=hstat,
        pvalue=pvalue,
        qvalue_fdr=qvalue,
        groups=grupos,
        n_per_group=tuple(int(bloco.shape[0]) for bloco in blocos),
        n_usable_per_group=utilizaveis,
        method="kruskal_wallis",
        fdr_alpha=fdr_alpha,
        min_per_group=MIN_AMOSTRAS_POR_GRUPO,
        provenance=matrix.provenance,
        # `testavel` e não `alvo`: `AnovaCounts` deriva `constant` de `testavel & isnan`, e a
        # coluna constante é justamente a que `alvo` tirou.
        counts=_contagens(hstat, pvalue, qvalue, testavel, sobreviveu),
        post_hoc=PostHocDecision(route=_rota(pvalue, sobreviveu), method="dunn"),
        comparisons=comparacoes,
    )


def _colunas_constantes(blocos: list[np.ndarray], testavel: np.ndarray) -> np.ndarray:
    """Máscara por feature: todas as observações utilizáveis têm o mesmo valor.

    Medida só nas colunas testáveis. Numa coluna sem observação utilizável nenhuma, `np.nanmin`
    emitiria `All-NaN slice encountered`; toda coluna testável tem ao menos
    `MIN_AMOSTRAS_POR_GRUPO` observações finitas por grupo. O retorno antecipado só evita montar
    o `vstack` à toa.

    É o equivalente do `0/0` que a ANOVA produz, com quadrado médio entre grupos e residual
    ambos nulos, e é contado no mesmo campo, `AnovaCounts.constant`.
    """
    constante = np.zeros(testavel.size, dtype=bool)
    if not bool(testavel.any()):
        return constante
    empilhado = np.vstack([bloco[:, testavel] for bloco in blocos])
    constante[testavel] = np.nanmin(empilhado, axis=0) == np.nanmax(empilhado, axis=0)
    return constante


def _post_hoc_de_dunn(
    *,
    feature_names: tuple[str, ...],
    grupos: tuple[str, ...],
    blocos: list[np.ndarray],
    sobreviveu: np.ndarray,
    cancelled: CancelCheck | None,
) -> tuple[DunnComparison, ...]:
    """As comparações de Dunn, só nas features que sobreviveram ao FDR entre features.

    Mesma estrutura de `_post_hoc`, com o cancelamento no topo de cada iteração.
    """
    comparacoes: list[DunnComparison] = []
    for indice in np.flatnonzero(sobreviveu).tolist():
        raise_if_cancelled(cancelled, "kruskal_wallis: laco do post-hoc de Dunn")
        amostras = _amostras_utilizaveis(blocos, indice)
        comparacoes.extend(_dunn_da_feature(feature_names[indice], grupos, amostras))
    return tuple(comparacoes)


def _dunn_da_feature(
    nome: str, grupos: tuple[str, ...], amostras: list[np.ndarray]
) -> list[DunnComparison]:
    """Os `k(k-1)/2` pares de uma feature pelo `z` de Dunn, com `q` de Benjamini-Hochberg.

    Os postos são os da ordenação conjunta, sobre todas as observações utilizáveis da feature, e
    não uma ordenação por par. É o que faz de Dunn um post-hoc do Kruskal-Wallis em vez de uma
    bateria de Mann-Whitney: só os postos conjuntos são coerentes com o `H` que decidiu correr o
    post-hoc.

    A variância é `N(N+1)/12` corrigida pelo mesmo somatório de empates que corrige o `H`,
    dividido por `12(N-1)`.

    O `p` é bilateral e sem ajuste. O `q` é Benjamini-Hochberg sobre estes `k(k-1)/2` p-valores
    e só sobre eles; a família entre features está em `AnovaResult.qvalue_fdr`.
    """
    todos = np.concatenate(amostras)
    n_total = int(todos.size)
    postos = rankdata(todos)
    tamanhos = np.array([int(amostra.size) for amostra in amostras], dtype=np.int64)
    medias = np.array(
        [float(fatia.mean()) for fatia in np.split(postos, np.cumsum(tamanhos)[:-1])],
        dtype=np.float64,
    )

    contagens = np.unique(todos, return_counts=True)[1].astype(np.float64)
    empates = float((contagens**3 - contagens).sum())
    variancia = n_total * (n_total + 1.0) / 12.0 - empates / (12.0 * (n_total - 1.0))

    pares = list(itertools.combinations(range(len(grupos)), 2))
    esquerda = np.array([i for i, _ in pares], dtype=np.int64)
    direita = np.array([j for _, j in pares], dtype=np.int64)
    diferencas = medias[esquerda] - medias[direita]
    # `variancia` só é zero quando todas as observações estão empatadas, e essa coluna foi
    # excluída por `_colunas_constantes`, não tem `q` e nunca sobrevive. O `errstate` fica para o
    # desfecho ser `nan`, e não um aviso, se alguma vez chegar aqui.
    with np.errstate(invalid="ignore", divide="ignore"):
        z = diferencas / np.sqrt(variancia * (1.0 / tamanhos[esquerda] + 1.0 / tamanhos[direita]))
    pvalores = np.asarray(2.0 * distribuicao_normal.sf(np.abs(z)), dtype=np.float64)
    qvalores = benjamini_hochberg(pvalores)

    return [
        DunnComparison(
            feature=nome,
            group_a=grupos[i],
            group_b=grupos[j],
            mean_rank_difference=float(diferenca),
            pvalue=float(p),
            qvalue_fdr=float(q),
        )
        for (i, j), diferenca, p, q in zip(pares, diferencas, pvalores, qvalores, strict=True)
    ]

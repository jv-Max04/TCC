"""Rede de correlação parcial esparsa com correção de viés (DSPC).

O DSPC de Basu et al. (2017, Bioinformatics 33(10):1545-1553) aplica a uma matriz de precisão
estimada por graphical lasso o estimador de-esparsificado de Jankova e van de Geer (2015,
Electronic Journal of Statistics 9(1):1205-1229), que tem forma fechada e distribuição
assintótica fechada:

    T = 2 Θ̂ - Θ̂ Σ̂ Θ̂
    √n (T_ij - Θ⁰_ij) → N(0, Θ⁰_ii Θ⁰_jj + (Θ⁰_ij)²)

É daí que sai um `p` por aresta. O graphical lasso sozinho é enviesado para zero e põe zeros
exatos fora da diagonal, que não têm erro-padrão nem distribuição: sem a correção a rede sai
igualmente desenhável, e sem inferência. A contagem de arestas significativas pode ser a mesma
pelos dois caminhos. O que a correção muda é que todo par passa a ter estimativa, inclusive os
que a penalização zerou, que sem ela saem com `p = 1`.

Duas âncoras exatas conferem a fórmula em `tests/unit/test_dspc.py`: com `Θ̂ = Σ̂⁻¹` a correção
é nula (`T = Θ̂`), e com `Θ̂ = I` e `Σ̂ = [[1, ½], [½, 1]]` ela dá `T = [[1, -½], [-½, 1]]`.

O erro-padrão usa `Θ̂`, e não `T`. Os dois são estimadores consistentes de `Θ⁰`, mas só `Θ̂` é
definida positiva por construção. `T` não tem garantia de positividade, e uma diagonal não
positiva daria raiz de negativo.

As colunas são padronizadas, como em Basu et al., e `Σ̂` é a matriz de correlação amostral. Uma
penalização única `λ` sobre `|Θ_ij|` só é comparável entre pares com colunas na mesma escala;
sem padronizar, a topologia da rede seria a da abundância. A correlação parcial é invariante a
escala.

O corte é no `p` cru, por `AnalysisSpec.dspc_alpha`, como na rede do MetaboAnalyst. O `q` de
Benjamini-Hochberg sobre os `k(k-1)/2` pares viaja em cada aresta e não corta. Quem publica a
contagem de arestas publica uma contagem a `p` cru sobre `k(k-1)/2` testes simultâneos.

O ajuste do graphical lasso é uma chamada única ao `scikit-learn`, sem ponto de parada: o
cancelamento é consultado antes dele, depois dele e no laço que monta as arestas. O módulo não
impõe teto de features próprio. Ver `dspc_network` para o custo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from scipy.stats import norm
from sklearn.covariance import GraphicalLassoCV

from refino.domain.analysis import PADRAO_DA_ANALISE, AnalysisSpec
from refino.domain.cancellation import CancelCheck, raise_if_cancelled
from refino.metrics.structure import column_usability, retained_column_digest
from refino.statistics.clustering import TETO_DE_FEATURES_PARA_COMPUTAR
from refino.statistics.univariate import benjamini_hochberg

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Iterable, Sequence

    from refino.domain.matrix import RefinoMatrix

__all__ = [
    "DOBRAS_DA_VALIDACAO_CRUZADA",
    "MIN_AMOSTRAS_PARA_DSPC",
    "PASSO_DA_PARADA",
    "DspcEdge",
    "DspcResult",
    "RedeAcimaDoTetoError",
    "RedeSemPopulacaoError",
    "desenviesa_precisao",
    "dspc_network",
    "p_por_aresta",
    "precisao_esparsa",
]


DOBRAS_DA_VALIDACAO_CRUZADA: Final = 5
"""Quantas dobras escolhem a penalidade `λ` do graphical lasso.

A penalidade é escolhida, e não fixada, porque ela governa quantas arestas existem.
`GraphicalLassoCV` percorre uma grade de `λ` e fica com a que maximiza a verossimilhança fora
da dobra. Cinco é o mesmo número de `classifier_folds`.

`GraphicalLassoCV` com `cv` inteiro usa `KFold` sem embaralhar, então a partição depende só de
`n` e a saída só da entrada. Não há sorteio, e por isso `dspc_network` não recebe semente.
"""

MIN_AMOSTRAS_PARA_DSPC: Final = DOBRAS_DA_VALIDACAO_CRUZADA
"""Abaixo disto a penalidade não pode ser escolhida.

Com menos amostras que dobras, `KFold` não particiona e o `scikit-learn` levanta no meio do
ajuste. A guarda existe para a recusa falar o vocabulário do domínio e para a rota fazer a mesma
pergunta antes de chamar.

É piso de exequibilidade, e não de validade. A distribuição de Jankova e van de Geer é
assintótica em `n`: com poucas amostras o cálculo corre, e o `p` vale pouco. O `n` a partir do
qual o `p` é confiável não está determinado; exigiria medir a taxa de erro tipo I por simulação
sob a nula.
"""

PASSO_DA_PARADA: Final = 64
"""De quantas em quantas linhas da precisão o laço das arestas consulta o cancelamento.

Mesmo valor e mesmo argumento de `correlacao.PASSO_DA_PARADA`.
"""


class RedeSemPopulacaoError(ValueError):
    """A matriz não sustenta a rede. Não é defeito de quem chamou.

    É classe à parte, como `PermanovaSemPopulacaoError`, para quem chama distinguir um fato sobre
    o dado, como uma matriz com quatro amostras de estudo, de um defeito de programação, que
    sobe como `ValueError` comum.
    """


class RedeAcimaDoTetoError(ValueError):
    """Recusa: pediram mais features do que o teto de cálculo permite.

    É classe à parte porque quem a apanha pode agir: recortar. A mensagem nomeia o número pedido
    e o teto.
    """


# ---------------------------------------------------------------------------
# 1. A precisão esparsa
# ---------------------------------------------------------------------------


def precisao_esparsa(
    observacoes: np.ndarray,
    *,
    dobras: int = DOBRAS_DA_VALIDACAO_CRUZADA,
) -> tuple[np.ndarray, np.ndarray, float]:
    """`(Θ̂, Σ̂, λ)`: a precisão esparsa, a correlação amostral e a penalidade escolhida.

    `observacoes` é `(n, k)` e já vem padronizada por `_padroniza`. Esta função não padroniza,
    para o `Σ̂` devolvido ser exatamente o que entrou no ajuste que produziu `Θ̂`.

    A penalidade sai da validação cruzada e é devolvida para o resultado a publicar: ela diz quão
    esparsa a rede foi forçada a ser.
    """
    estimador = GraphicalLassoCV(cv=dobras, assume_centered=True).fit(observacoes)
    theta = np.asarray(estimador.precision_, dtype=np.float64)
    sigma = np.asarray(estimador.covariance_, dtype=np.float64)
    return theta, sigma, float(estimador.alpha_)


# ---------------------------------------------------------------------------
# 2. A correção de viés
# ---------------------------------------------------------------------------


def desenviesa_precisao(precisao: np.ndarray, covariancia: np.ndarray) -> np.ndarray:
    """`T = 2Θ̂ - Θ̂ Σ̂ Θ̂`, o estimador de-esparsificado de Jankova e van de Geer (2015).

    `Θ̂` tem zeros exatos fora da diagonal, postos pela penalização e não pelo dado; `T` não tem
    nenhum. Cada par volta a ter uma estimativa com erro-padrão, e é isso que dá `p`.

    É pública para o teste comparar o resultado com e sem a correção sobre a mesma entrada. Uma
    bandeira em `dspc_network` que a desligasse permitiria obter a rede sem a correção com o
    nome do método.

    Não simetriza. `Θ̂` e `Σ̂` são simétricas, logo `Θ̂Σ̂Θ̂` também é, a menos do épsilon de
    máquina, e a rede lê só o triângulo superior. Simetrizar esconderia um `Θ̂` assimétrico, que
    seria defeito.
    """
    precisao = np.asarray(precisao, dtype=np.float64)
    covariancia = np.asarray(covariancia, dtype=np.float64)
    if precisao.shape != covariancia.shape or precisao.ndim != 2:
        raise ValueError(
            f"a precisão tem forma {precisao.shape} e a covariância {covariancia.shape}: a "
            "correção é `2Θ - ΘΣΘ` e as duas têm de ser a mesma matriz quadrada `(k, k)`"
        )
    return np.asarray(2.0 * precisao - precisao @ covariancia @ precisao, dtype=np.float64)


def p_por_aresta(
    desenviesada: np.ndarray,
    precisao: np.ndarray,
    n_amostras: int,
) -> tuple[np.ndarray, np.ndarray]:
    """`(z, p)` por par, pela distribuição assintótica. Duas matrizes `(k, k)` simétricas.

    `z_ij = √n · T_ij / √(Θ̂_ii Θ̂_jj + Θ̂²_ij)` e `p = 2·Φ(-|z|)`, bilateral.

    `desenviesada` e `precisao` são argumentos separados para o teste poder passar `Θ̂` nos dois
    e medir o que acontece sem a correção: `p = 1` exato em todo par que a penalização zerou.

    A diagonal sai `nan`, porque `Θ_ii` não é aresta. Ver `DspcResult.pvalue`.
    """
    desenviesada = np.asarray(desenviesada, dtype=np.float64)
    precisao = np.asarray(precisao, dtype=np.float64)
    if n_amostras < 1:
        raise ValueError(f"`n_amostras` é {n_amostras}: o `z` tem `√n` no numerador")

    diagonal = np.diag(precisao)
    # `Θ̂_ii Θ̂_jj`, e não `T_ii T_jj`: `Θ̂` é definida positiva por construção, então o produto
    # é estritamente positivo e a raiz está sempre definida.
    variancia = np.outer(diagonal, diagonal) + precisao**2
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.sqrt(n_amostras) * desenviesada / np.sqrt(variancia)
    np.fill_diagonal(z, np.nan)
    p = np.asarray(2.0 * norm.sf(np.abs(z)), dtype=np.float64)
    return np.asarray(z, dtype=np.float64), p


# ---------------------------------------------------------------------------
# 3. O resultado
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DspcEdge:
    """Uma aresta sobrevivente, com os quatro números que a tornam citável.

    `partial_correlation` e `pvalue` vão juntos e sem valor padrão: uma aresta é uma associação
    e a confiança nela.
    """

    feature_a: str
    feature_b: str

    partial_correlation: float
    """`-T_ij / √(Θ̂_ii Θ̂_jj)`: numerador desenviesado, denominador do `Θ̂` esparso.

    O sinal negativo é da definição: a correlação parcial é o simétrico da precisão normalizada.
    Trocá-lo desenharia toda aresta na cor oposta.

    Partilha o numerador com `statistic`, então `partial_correlation == 0` se e só se
    `pvalue == 1`."""

    statistic: float
    """O `z` assintótico. O nome é neutro, como `PermanovaResult.statistic`."""

    pvalue: float
    qvalue_fdr: float
    """Benjamini-Hochberg sobre os `k(k-1)/2` pares. Viaja e não corta."""


@dataclass(frozen=True)
class DspcResult:
    """A rede de correlação parcial esparsa desenviesada, e o que a torna interpretável."""

    features: tuple[str, ...]
    """Os nomes retidos, na ordem das linhas e colunas das três matrizes."""

    partial_correlation: np.ndarray
    """`(k, k)` simétrica, diagonal `nan`. Imutável. Desenviesada."""

    pvalue: np.ndarray
    """`(k, k)` simétrica, diagonal `nan`.

    `nan` na diagonal, e não `0`, porque `Θ_ii` não é aresta: um zero ali seria contado por
    qualquer varredura por `p < alpha` que não excluísse a diagonal."""

    statistic: np.ndarray
    """`(k, k)` dos `z` assintóticos. Diagonal `nan`."""

    edges: tuple[DspcEdge, ...]
    """As arestas com `pvalue < alpha`, ordenadas por `(pvalue, feature_a, feature_b)`.

    Pode estar vazia, e vazio é resultado: quem desenha escreve a contagem e o limiar."""

    n_edges: int
    """Quantas arestas sobreviveram.

    Sai de `count_nonzero` sobre a máscara do triângulo superior, e não de `len(edges)`, que
    percorre um laço em Python. Um teste compara os dois."""

    n_pairs_tested: int
    """`k(k-1)/2`. O denominador do `q`, e o que torna a contagem de arestas legível."""

    alpha: float
    penalty: float
    """O `λ` que a validação cruzada escolheu. É ele que governa a esparsidade: o mesmo dado dá
    redes com poucas ou muitas arestas conforme o `λ`."""

    n_samples_used: int
    """Quantas amostras de estudo entraram. QC e `validate` ficam de fora: ver `_observacoes`."""

    n_features_total: int
    n_features_used: int
    n_features_dropped_constant: int
    n_features_dropped_non_finite: int

    n_pairs_undefined: int
    """Pares acima da diagonal cujo `p` saiu não finito.

    Hoje é zero por construção: `Θ̂` é definida positiva, então a variância é positiva e o `z` é
    finito. O campo separa 'não há aresta' de 'não deu para medir', como
    `CorrelationResult.n_pairs_undefined`."""

    column_digest: str


# ---------------------------------------------------------------------------
# 4. A rede
# ---------------------------------------------------------------------------


def _observacoes(matrix: RefinoMatrix) -> np.ndarray:
    """As linhas de estudo da matriz. Levanta quando não sustentam a rede.

    A população é a de `sample_mask`, como na correlação e na ANOVA. Os QCs são réplicas da mesma
    mistura e inflam a associação aparente entre features. Aqui o efeito é pior que na correlação
    simples: a precisão inverte a covariância, e uma covariância quase singular produz arestas
    enormes onde não há nada.
    """
    estudo = matrix.sample_mask
    n_estudo = int(estudo.sum())
    if n_estudo < MIN_AMOSTRAS_PARA_DSPC:
        raise RedeSemPopulacaoError(
            f"a rede precisa de ao menos {MIN_AMOSTRAS_PARA_DSPC} amostras de estudo e a matriz "
            f"tem {n_estudo}. A penalidade é escolhida por validação cruzada de "
            f"{DOBRAS_DA_VALIDACAO_CRUZADA} dobras, e com menos amostras que dobras não há como "
            "particionar. QC e `validate` ficam de fora de propósito: são réplicas da mesma "
            "mistura, e incluí-los mediria a estabilidade do instrumento com o nome da associação."
        )
    return np.asarray(matrix.values[estudo, :], dtype=np.float64)


def _padroniza(observacoes: np.ndarray) -> np.ndarray:
    """Centra e divide pelo desvio de cada coluna.

    O desvio é o populacional (`ddof=0`), para `Xᵀ X / n` ser a matriz de correlação com uns
    exatos na diagonal, que é o que o estimador consome com `assume_centered=True`. Com `ddof=1`
    a diagonal deixaria de ser 1, e a penalidade passaria a depender de `n`.

    As colunas constantes já saíram em `_utilizaveis`, então não há divisão por zero.
    """
    centradas = observacoes - observacoes.mean(axis=0)
    return np.asarray(centradas / centradas.std(axis=0, ddof=0), dtype=np.float64)


def _utilizaveis(
    valores: np.ndarray, escolhidas: np.ndarray, nomes: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...], int, int]:
    """As colunas escolhidas que dá para usar, com as duas contagens de descarte.

    Usa `column_usability`, porque o desvio de uma coluna constante não é exatamente zero em
    ponto flutuante. É igual ao `_utilizaveis` de `correlacao`.
    """
    recortadas = valores[:, escolhidas]
    usaveis, n_constantes, n_nao_finitas = column_usability(recortadas)
    retidas = recortadas[:, usaveis]
    nomes_retidos = tuple(
        nomes[int(indice)]
        for indice, mantem in zip(escolhidas.tolist(), usaveis.tolist(), strict=True)
        if mantem
    )
    return retidas, nomes_retidos, n_constantes, n_nao_finitas


def _colunas_ou_recusa(colunas: Iterable[int] | None, n_features: int) -> np.ndarray:
    """Os índices pedidos, conferidos. `None` é 'todas'.

    Igual ao de `correlacao`. Aqui um índice repetido poria duas colunas idênticas na rede, e a
    covariância ficaria singular.
    """
    if colunas is None:
        return np.arange(n_features, dtype=int)
    indices = np.asarray(list(colunas), dtype=int)
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= n_features):
        raise ValueError(
            f"há índice de coluna fora da matriz: pedidos entre {int(indices.min())} e "
            f"{int(indices.max())}, e a matriz tem {n_features} features"
        )
    if len(set(indices.tolist())) != indices.size:
        raise ValueError(
            "há índice de coluna repetido no recorte: a mesma feature entraria duas vezes na rede, "
            "e duas colunas idênticas tornam a covariância singular"
        )
    return indices


def dspc_network(
    matrix: RefinoMatrix,
    analise: AnalysisSpec = PADRAO_DA_ANALISE,
    *,
    colunas: Iterable[int] | None = None,
    teto: int | None = TETO_DE_FEATURES_PARA_COMPUTAR,
    cancelled: CancelCheck | None = None,
) -> DspcResult:
    """A rede de correlação parcial esparsa desenviesada, sobre as amostras de estudo.

    Recebe `AnalysisSpec`, e não o alfa solto, porque `dspc_alpha` é validado em
    `AnalysisSpec.__post_init__`. `colunas` é parâmetro de chamada, como em
    `feature_correlation`: não é escolha metodológica, é quantas features quem chama consegue
    pagar.

    `teto` vem de `TETO_DE_FEATURES_PARA_COMPUTAR`, que hoje é `None`, então nada é recusado por
    tamanho. O custo cresce mais depressa que o quadrado de `k`, porque o número de iterações do
    ajuste também cresce. Medido com `n = 60` amostras: 2,20 s com `k = 60`, 4,20 s com
    `k = 100`, 11,27 s com `k = 150` e 29,47 s com `k = 200`. Sobre as milhares de features de
    uma matriz inteira o cálculo não termina em tempo útil, e o recorte é de quem chama.

    Levanta `RedeSemPopulacaoError` quando a população não sustenta a rede, `RedeAcimaDoTetoError`
    quando `teto` é um número e o recorte o excede, e `ValueError` quando o recorte é inválido.
    """
    escolhidas = _colunas_ou_recusa(colunas, matrix.n_features)
    if teto is not None and escolhidas.size > teto:
        raise RedeAcimaDoTetoError(
            f"a rede foi pedida sobre {escolhidas.size} features e o teto de cálculo é {teto}. O "
            "ajuste esparso é superlinear em k e a penalidade é escolhida por validação cruzada; "
            "recorte as features antes de pedir a rede."
        )

    observacoes = _observacoes(matrix)
    raise_if_cancelled(cancelled, "rede: antes de escolher as colunas utilizáveis")
    retidas, nomes, n_constantes, n_nao_finitas = _utilizaveis(
        observacoes, escolhidas, matrix.feature_names
    )
    # São precisas duas colunas para haver um par.
    if len(nomes) < 2:
        raise RedeSemPopulacaoError(
            f"restaram {len(nomes)} feature(s) utilizável(eis) das {escolhidas.size} escolhidas: "
            f"{n_constantes} são constantes e {n_nao_finitas} têm valor não finito. Uma rede "
            "precisa de ao menos duas colunas para haver um par, e uma coluna constante torna a "
            "correlação `0/0` — devolver uma rede vazia seria indistinguível de «não há aresta»."
        )

    raise_if_cancelled(cancelled, "rede: antes do ajuste esparso da precisão")
    theta, sigma, penalidade = precisao_esparsa(_padroniza(retidas))
    raise_if_cancelled(cancelled, "rede: depois do ajuste esparso, antes da correção de viés")

    # A correção de viés é o que dá `p` aos pares que a penalização zerou. Sem ela, `theta` iria
    # direto para `p_por_aresta` e esses pares sairiam com `p = 1`.
    desenviesada = desenviesa_precisao(theta, sigma)
    n_usadas = int(retidas.shape[0])
    z, p = p_por_aresta(desenviesada, theta, n_usadas)

    diagonal = np.diag(theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        parcial = -desenviesada / np.sqrt(np.outer(diagonal, diagonal))
    np.fill_diagonal(parcial, np.nan)

    acima = np.triu(np.ones_like(p, dtype=bool), k=1)
    q = np.full_like(p, np.nan)
    q[acima] = benjamini_hochberg(p[acima])
    q = np.asarray(np.where(acima, q, q.T), dtype=np.float64)
    np.fill_diagonal(q, np.nan)

    sobreviveu = acima & np.isfinite(p) & (p < analise.dspc_alpha)
    arestas = _monta_arestas(nomes, parcial, z, p, q, sobreviveu, cancelled)

    for matriz in (parcial, z, p):
        matriz.flags.writeable = False
    return DspcResult(
        features=nomes,
        partial_correlation=parcial,
        pvalue=p,
        statistic=z,
        edges=arestas,
        # Contado pela máscara, e não por `len(arestas)`. Ver `DspcResult.n_edges`.
        n_edges=int(np.count_nonzero(sobreviveu)),
        n_pairs_tested=int(np.count_nonzero(acima)),
        alpha=analise.dspc_alpha,
        penalty=penalidade,
        n_samples_used=n_usadas,
        n_features_total=matrix.n_features,
        n_features_used=len(nomes),
        n_features_dropped_constant=n_constantes,
        n_features_dropped_non_finite=n_nao_finitas,
        n_pairs_undefined=int(np.count_nonzero(acima & ~np.isfinite(p))),
        column_digest=_digest_das_retidas(matrix, nomes),
    )


def _monta_arestas(
    nomes: tuple[str, ...],
    parcial: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    sobreviveu: np.ndarray,
    cancelled: CancelCheck | None,
) -> tuple[DspcEdge, ...]:
    """As arestas sobreviventes, ordenadas por `(pvalue, feature_a, feature_b)`.

    O desempate é pelos nomes, para a ordem não depender da posição das colunas. O cancelamento
    é consultado dentro do laço: com `k` na casa das centenas são dezenas de milhares de pares.
    """
    encontradas: list[DspcEdge] = []
    for i in range(len(nomes)):
        if i % PASSO_DA_PARADA == 0:
            raise_if_cancelled(cancelled, f"rede: aresta da linha {i} de {len(nomes)}")
        for j in range(i + 1, len(nomes)):
            if not sobreviveu[i, j]:
                continue
            encontradas.append(
                DspcEdge(
                    feature_a=nomes[i],
                    feature_b=nomes[j],
                    partial_correlation=float(parcial[i, j]),
                    statistic=float(z[i, j]),
                    pvalue=float(p[i, j]),
                    qvalue_fdr=float(q[i, j]),
                )
            )
    encontradas.sort(key=lambda a: (a.pvalue, a.feature_a, a.feature_b))
    return tuple(encontradas)


def _digest_das_retidas(matrix: RefinoMatrix, retidos: Sequence[str]) -> str:
    """`sha256` dos nomes retidos, na ordem das colunas da matriz. Contagem igual não é conjunto
    igual."""
    mantidos = set(retidos)
    mascara = np.array([nome in mantidos for nome in matrix.feature_names], dtype=bool)
    return retained_column_digest(matrix.feature_names, mascara)

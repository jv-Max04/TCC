"""Correlação entre features e busca por padrão contra um perfil de referência.

`feature_correlation` responde 'estas features sobem e descem juntas?' com uma matriz `k x k`.
`pattern_search` responde 'quais features seguem este perfil?' com um coeficiente por feature
contra um vetor de referência. As duas partilham o coeficiente: `CORRELATION_IMPLEMENTATIONS`
recebe `(observações, referência)` e devolve um coeficiente por coluna, e a matriz é montada
chamando-o uma vez por coluna.

A matriz é quadrática em `k`, e quem chama a recorta por `colunas`: com 6.035 features a saída
teria `6035² x 8 bytes = 291 MiB`, e o mapa de calor desenha no máximo
`TETO_DE_FEATURES_PARA_DESENHAR` delas. A busca é uma chamada de `O(n·k)`, devolve um número por
feature e corre sobre todas.

A população é a de `sample_mask`: QC e `validate` ficam de fora, como na ANOVA. Os QCs são
réplicas da mesma mistura, e linhas quase idênticas inflam a correlação aparente entre features
que nada têm a ver uma com a outra.

Coluna constante sai por `ptp == 0`, via `column_usability`, e é contada. Correlação com uma
constante é `0/0`, e escrever `0,0` na célula diria 'não há associação' sobre o que não foi
possível medir.

A ordem dos grupos de um perfil é a de `ordem_dos_grupos`, por ponto de código, a mesma regra de
`plots.series.ordem_canonica`, que decide a cor de cada grupo. O resultado carrega a ordem para
a figura a declarar, e
`tests/unit/test_correlacao.py::test_a_ordem_canonica_do_NUCLEO_e_a_do_DESENHO` confere que as
duas coincidem.

Não há passo estocástico neste módulo, e por isso não há semente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from scipy.stats import kendalltau, rankdata

from refino.domain.analysis import PADRAO_DA_ANALISE, AnalysisSpec
from refino.domain.cancellation import CancelCheck, raise_if_cancelled
from refino.domain.labels import CorrelationMethod
from refino.metrics.structure import column_usability, retained_column_digest

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Callable, Iterable, Sequence

    from refino.domain.matrix import RefinoMatrix

__all__ = [
    "CORRELATION_IMPLEMENTATIONS",
    "MIN_AMOSTRAS_PARA_CORRELACAO",
    "PASSO_DA_PARADA",
    "SEPARADOR_DO_PERFIL",
    "CorrelationResult",
    "PatternReferenceKind",
    "PatternSearchResult",
    "ReferenciaDesconhecidaError",
    "feature_correlation",
    "matriz_de_correlacao",
    "ordem_dos_grupos",
    "pattern_search",
]


PatternReferenceKind = Literal["feature", "profile"]
"""O que `AnalysisSpec.pattern_reference` era: o nome de uma feature, ou um perfil por grupo.

São dois estados nomeados, e não um booleano, porque a figura escreve uma frase diferente para
cada um. O valor vai no resumo da API, e `plots.correlacao.frase_da_referencia` recusa um valor
fora do vocabulário.
"""

SEPARADOR_DO_PERFIL: Final = ","
"""Como os números de um perfil vêm separados dentro de `pattern_reference`.

Vírgula para separar e ponto decimal. O campo é valor de formulário, e não texto: `float("1,5")`
não existe em Python, e aceitar vírgula decimal tornaria `1,5` ambíguo.

O nome da feature é tentado primeiro, então uma feature cujo nome contenha vírgula continua
alcançável pela correspondência exata.
"""

MIN_AMOSTRAS_PARA_CORRELACAO: Final = 3
"""Abaixo disto o coeficiente é consequência da geometria, e não medida.

Com duas observações, dois pontos definem uma reta: o `r` de Pearson vale `+1` ou `-1` para
qualquer par de features não constantes. Com uma, o desvio é zero e o coeficiente é `0/0`.

É constante importável para a rota fazer a mesma pergunta antes de chamar, em vez de apanhar um
`ValueError` largo depois.
"""

PASSO_DA_PARADA: Final = 64
"""De quantas em quantas colunas a matriz consulta o cancelamento.

A cada coluna, a consulta dominaria o custo de uma matriz pequena. Uma vez só, cancelar durante
a construção de uma matriz de milhares de colunas não teria efeito.
"""


class ReferenciaDesconhecidaError(ValueError):
    """Recusa: o perfil de referência não é feature desta matriz nem perfil válido para ela.

    A mensagem nomeia o que existe, porque as causas possíveis são três: um nome errado, um
    perfil com o número errado de grupos ou um separador diferente. É `ValueError`, e quem o
    traduz para a tela é a rota, que devolve 422.
    """


# ---------------------------------------------------------------------------
# Os três coeficientes: vetor contra matriz
# ---------------------------------------------------------------------------


def _pearson(observacoes: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    """`r` de Pearson entre cada coluna e a referência. Uma passagem, sem laço.

    Centra e divide pelas normas, em vez de usar `np.corrcoef`, que devolveria a matriz inteira
    ao custo de `k²` de memória.
    """
    centradas = observacoes - observacoes.mean(axis=0)
    centrada = referencia - referencia.mean()
    numerador = centradas.T @ centrada
    denominador = np.sqrt((centradas * centradas).sum(axis=0) * float(centrada @ centrada))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(numerador / denominador, dtype=float)


def _spearman(observacoes: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    """`rho`, o `r` de Pearson sobre os postos de cada coluna.

    Os postos são tomados dentro de cada coluna (`axis=0`), porque a variável é a coluna: a
    associação entre duas features é medida ao longo das amostras. Em
    `clustering._distancia_de_spearman` o eixo é o outro, porque lá o ponto é a linha.
    """
    return _pearson(
        np.asarray(rankdata(observacoes, axis=0), dtype=float),
        np.asarray(rankdata(referencia), dtype=float),
    )


def _kendall(observacoes: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    """`tau-b` de Kendall entre cada coluna e a referência.

    É um laço sobre o `kendalltau` do `scipy`, e não uma conta vetorizada, porque a correção de
    empates é onde as variantes `tau-a`, `tau-b` e `tau-c` diferem, e a do `scipy` é a
    referência.
    """
    return np.array(
        [
            float(kendalltau(observacoes[:, j], referencia).statistic)
            for j in range(observacoes.shape[1])
        ],
        dtype=float,
    )


CORRELATION_IMPLEMENTATIONS: Final[
    dict[CorrelationMethod, Callable[[np.ndarray, np.ndarray], np.ndarray]]
] = {
    "pearson": _pearson,
    "spearman": _spearman,
    "kendall": _kendall,
}
"""Despacho por nome de coeficiente. `tests/unit/test_labels.py` confere as chaves com o `Literal`.

É registro, e não `if/elif`, para um nome sem implementação, ou o contrário, reprovar no teste
de paridade.
"""


def matriz_de_correlacao(
    observacoes: np.ndarray,
    metodo: CorrelationMethod,
    *,
    cancelled: CancelCheck | None = None,
) -> np.ndarray:
    """A matriz `k x k` de correlação entre as colunas de `observacoes`. Imutável.

    É uma chamada do coeficiente por coluna, o que mantém uma implementação por método. O custo
    é quadrático em `k`; por isso quem chama recorta, e o ponto de parada fica dentro do laço.

    O resultado é aparado em `[-1, 1]`: a soma de produtos em ponto flutuante devolve
    `1,0000000000000002` para uma coluna contra si própria, e o mapa de calor desenha com
    limites fixos em `±1`.
    """
    coeficiente = CORRELATION_IMPLEMENTATIONS[metodo]
    colunas: list[np.ndarray] = []
    for j in range(observacoes.shape[1]):
        if j % PASSO_DA_PARADA == 0:
            raise_if_cancelled(cancelled, f"correlação: coluna {j} de {observacoes.shape[1]}")
        colunas.append(coeficiente(observacoes, observacoes[:, j]))
    # `asarray` explícito, e não `cast`: `np.clip` é tipado como `Any` nos stubs, e isto garante
    # o tipo também em runtime.
    matriz: np.ndarray = np.asarray(np.clip(np.column_stack(colunas), -1.0, 1.0), dtype=float)
    matriz.flags.writeable = False
    return matriz


# ---------------------------------------------------------------------------
# A população, e o recorte de quem chama
# ---------------------------------------------------------------------------


def _observacoes_de_estudo(matrix: RefinoMatrix) -> np.ndarray:
    """As linhas de estudo da matriz. Levanta quando são menos que o mínimo."""
    estudo = matrix.sample_mask
    n_estudo = int(estudo.sum())
    if n_estudo < MIN_AMOSTRAS_PARA_CORRELACAO:
        raise ValueError(
            f"a correlação precisa de ao menos {MIN_AMOSTRAS_PARA_CORRELACAO} amostras de estudo e "
            f"a matriz tem {n_estudo}. "
            "QC e `validate` ficam de fora de propósito: os QCs são réplicas da mesma mistura, e "
            "incluí-los mediria a estabilidade do instrumento com o nome da associação biológica. "
            "Com menos de três pontos o coeficiente ou é indefinido ou vale ±1 por construção."
        )
    return np.asarray(matrix.values[estudo, :], dtype=float)


def _colunas_ou_recusa(colunas: Iterable[int] | None, n_features: int) -> np.ndarray:
    """Os índices de coluna pedidos, conferidos. `None` é 'todas'.

    Índice fora do intervalo e índice repetido têm mensagens separadas. Um índice repetido poria
    a mesma feature duas vezes na matriz, com uns fora da diagonal.
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
            "há índice de coluna repetido no recorte: a mesma feature entraria duas vezes na "
            "matriz, e a correlação deixaria de ter a diagonal que a define"
        )
    return indices


def _utilizaveis(
    valores: np.ndarray, escolhidas: np.ndarray, nomes: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...], int, int]:
    """As colunas escolhidas que dá para correlacionar, com as duas contagens de descarte.

    Devolve `(observações retidas, nomes retidos, n constantes, n não finitas)`. As contagens são
    separadas porque constância é informação sobre o dado, e não finito é o que o processamento
    produziu.
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


def _digest_das_retidas(matrix: RefinoMatrix, retidos: Sequence[str]) -> str:
    """`sha256` dos nomes retidos, na ordem das colunas da matriz. Contagem igual não é conjunto
    igual."""
    mantidos = set(retidos)
    mascara = np.array([nome in mantidos for nome in matrix.feature_names], dtype=bool)
    return retained_column_digest(matrix.feature_names, mascara)


# ---------------------------------------------------------------------------
# 1. O mapa de correlação
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrelationResult:
    """A matriz de correlação entre features, e o que a torna interpretável.

    As contagens de descarte fazem parte do resultado porque não são deriváveis dele: uma
    feature constante não está na matriz, e nada em `matrix` diz que ela existiu. A figura
    declara quantas ficaram de fora e por quê.
    """

    features: tuple[str, ...]
    """Os nomes das features retidas, na ordem das linhas e colunas de `matrix`."""

    matrix: np.ndarray
    """`(k, k)`, simétrica, diagonal 1, aparada em `[-1, 1]`. Imutável."""

    method: CorrelationMethod

    n_samples_used: int
    """Quantas amostras de estudo entraram. QC e `validate` ficam de fora."""

    n_features_total: int
    """Quantas features a matriz inteira tem. É o denominador do rodapé, e não é recortado."""

    n_features_used: int
    n_features_dropped_constant: int
    n_features_dropped_non_finite: int

    n_pairs_undefined: int
    """Pares acima da diagonal cujo coeficiente saiu não finito.

    Hoje é zero por construção, porque as colunas constantes e não finitas já saíram. O campo
    separa 'não há associação' de 'não deu para medir' se um método novo produzir indefinido."""

    column_digest: str


def feature_correlation(
    matrix: RefinoMatrix,
    analise: AnalysisSpec = PADRAO_DA_ANALISE,
    *,
    colunas: Iterable[int] | None = None,
    cancelled: CancelCheck | None = None,
) -> CorrelationResult:
    """A matriz de correlação entre as features escolhidas, sobre as amostras de estudo.

    Recebe `AnalysisSpec`, e não o método solto, porque `correlation_method` é validado em
    `AnalysisSpec.__post_init__`.

    `colunas` é parâmetro de chamada, e não campo da especificação, como `cluster_features`: não
    é escolha metodológica, é quantas features quem chama consegue desenhar.

    A matriz chega processada. Esta função não imputa, não normaliza e não transforma.
    """
    escolhidas = _colunas_ou_recusa(colunas, matrix.n_features)
    observacoes = _observacoes_de_estudo(matrix)

    raise_if_cancelled(cancelled, "correlação: antes de escolher as colunas utilizáveis")
    retidas, nomes, n_constantes, n_nao_finitas = _utilizaveis(
        observacoes, escolhidas, matrix.feature_names
    )
    if not nomes:
        raise ValueError(
            f"nenhuma das {len(escolhidas)} features escolhidas é utilizável para correlacionar: "
            f"{n_constantes} são constantes e {n_nao_finitas} têm valor não finito. Correlação com "
            "uma constante é `0/0`, e devolver uma matriz vazia seria indistinguível de «não há "
            "associação nenhuma»."
        )

    matriz = matriz_de_correlacao(retidas, analise.correlation_method, cancelled=cancelled)
    acima = np.triu(np.ones_like(matriz, dtype=bool), k=1)
    return CorrelationResult(
        features=nomes,
        matrix=matriz,
        method=analise.correlation_method,
        n_samples_used=int(observacoes.shape[0]),
        n_features_total=matrix.n_features,
        n_features_used=len(nomes),
        n_features_dropped_constant=n_constantes,
        n_features_dropped_non_finite=n_nao_finitas,
        n_pairs_undefined=int(np.count_nonzero(acima & ~np.isfinite(matriz))),
        column_digest=_digest_das_retidas(matrix, nomes),
    )


# ---------------------------------------------------------------------------
# 2. A busca por padrão
# ---------------------------------------------------------------------------


def ordem_dos_grupos(grupos: Iterable[str]) -> tuple[str, ...]:
    """Os grupos distintos, ordenados por ponto de código. É a ordem que o perfil assume.

    É a mesma regra de `plots.series.ordem_canonica`, e um teste confere que as duas coincidem.

    Não é ordem alfabética: `ATLL` vem antes de `Asymptomatic` porque `T` (U+0054) vem antes de
    `s` (U+0073). Ganha-se determinismo independente de localidade, e por isso a figura escreve
    a ordem em vez de a deixar implícita.
    """
    return tuple(sorted(set(grupos)))


@dataclass(frozen=True)
class PatternSearchResult:
    """As features ordenadas pela aderência ao perfil, e o perfil que elas seguiram.

    `features` e `coefficient` são listas paralelas, em ordem decrescente do coeficiente. A
    ordem é decrescente com sinal, e não por módulo: 'sobe onde o perfil sobe' e 'desce onde o
    perfil sobe' são achados opostos, e ordenar por `|r|` poria os dois juntos no topo.
    """

    features: tuple[str, ...]
    coefficient: tuple[float, ...]

    reference: str
    """O valor de `AnalysisSpec.pattern_reference`, como veio. É o que a figura escreve."""

    reference_kind: PatternReferenceKind
    profile: tuple[float, ...]
    """O vetor de referência por grupo, na ordem de `group_order`.

    Preenchido nos dois casos: contra uma feature, é a média dela em cada grupo. É o que permite
    à figura mostrar contra o quê a busca correu."""

    group_order: tuple[str, ...]
    """A ordem assumida entre os grupos. A figura a declara."""

    method: CorrelationMethod

    n_samples_used: int
    n_features_total: int
    n_features_used: int
    n_features_dropped_constant: int
    n_features_dropped_non_finite: int

    n_coefficient_undefined: int
    """Features retidas cujo coeficiente saiu não finito, e que por isso não estão nas listas.

    São contadas, e não zeradas: `0,0` diria 'não segue o padrão', que é outro fato."""

    column_digest: str


def _perfil_ou_recusa(
    referencia: str, grupos: tuple[str, ...], nomes: Sequence[str]
) -> tuple[float, ...]:
    """Os números de um perfil por grupo, ou a recusa que nomeia as três causas possíveis."""
    pedacos = [p.strip() for p in referencia.split(SEPARADOR_DO_PERFIL)]
    disponiveis = ", ".join(list(nomes)[:5]) + ("…" if len(nomes) > 5 else "")
    recusa = ReferenciaDesconhecidaError(
        f"`pattern_reference` é {referencia!r} e não é nem uma feature desta matriz nem um perfil "
        f"por grupo. Um perfil são {len(grupos)} números separados por {SEPARADOR_DO_PERFIL!r}, "
        f"com ponto decimal, na ordem {list(grupos)}. As features começam por: {disponiveis}"
    )
    if len(pedacos) != len(grupos):
        raise recusa
    try:
        valores = tuple(float(p) for p in pedacos)
    except ValueError:
        raise recusa from None
    if not all(np.isfinite(valores)):
        raise recusa
    if float(np.ptp(valores)) == 0.0:
        raise ReferenciaDesconhecidaError(
            f"o perfil {list(valores)} é constante: um perfil sem variação não tem correlação "
            "definida com feature nenhuma, e a busca devolveria a lista inteira indefinida. "
            "Declare ao menos dois valores diferentes entre os grupos."
        )
    return valores


def pattern_search(
    matrix: RefinoMatrix,
    analise: AnalysisSpec = PADRAO_DA_ANALISE,
    *,
    cancelled: CancelCheck | None = None,
) -> PatternSearchResult:
    """As features ordenadas pela correlação com o perfil de referência.

    Corre sobre todas as features, ao contrário do mapa: o custo é uma chamada de `O(n·k)` e a
    saída é um número por feature.

    `pattern_reference` é lido primeiro como nome de feature e depois como perfil: a
    correspondência exata com uma feature existente não é ambígua.

    `analise.correlation` não é consultado aqui. `pattern_reference is None` é o que significa
    'não peça a busca', para ser possível pedir o mapa sem ela.
    """
    referencia = analise.pattern_reference
    if referencia is None:
        raise ValueError(
            "`pattern_reference` é `None`: a busca por padrão não foi pedida. `None` **salta** a "
            "busca sem desligar o mapa de correlação, e quem decide se a chama é quem lê o "
            "contrato — não esta função, que não tem contra o que correlacionar."
        )

    observacoes = _observacoes_de_estudo(matrix)
    grupos_por_amostra = [
        amostra.group
        for amostra, estudo in zip(matrix.samples, matrix.sample_mask.tolist(), strict=True)
        if estudo
    ]
    grupos = ordem_dos_grupos(grupos_por_amostra)

    raise_if_cancelled(cancelled, "busca por padrão: antes de montar o vetor de referência")
    todas = np.arange(matrix.n_features, dtype=int)
    retidas, nomes, n_constantes, n_nao_finitas = _utilizaveis(
        observacoes, todas, matrix.feature_names
    )
    if not nomes:
        raise ValueError(
            f"nenhuma das {matrix.n_features} features é utilizável para a busca por padrão: "
            f"{n_constantes} são constantes e {n_nao_finitas} têm valor não finito"
        )

    if referencia in matrix.feature_names:
        tipo: PatternReferenceKind = "feature"
        coluna = matrix.feature_names.index(referencia)
        vetor = observacoes[:, coluna].astype(float, copy=True)
        if float(np.ptp(vetor)) == 0.0 or not np.isfinite(vetor).all():
            raise ReferenciaDesconhecidaError(
                f"a feature {referencia!r} é constante ou tem valor não finito nas amostras de "
                "estudo: a correlação dela com qualquer outra é indefinida, e a busca devolveria a "
                "lista inteira indefinida"
            )
        perfil = tuple(
            float(np.mean(vetor[[g == nome for g in grupos_por_amostra]])) for nome in grupos
        )
    else:
        tipo = "profile"
        perfil = _perfil_ou_recusa(referencia, grupos, matrix.feature_names)
        por_grupo = dict(zip(grupos, perfil, strict=True))
        vetor = np.array([por_grupo[g] for g in grupos_por_amostra], dtype=float)

    raise_if_cancelled(cancelled, "busca por padrão: antes de correlacionar com a referência")
    coeficientes = np.clip(
        CORRELATION_IMPLEMENTATIONS[analise.correlation_method](retidas, vetor), -1.0, 1.0
    )

    definidos = np.isfinite(coeficientes)
    # O desempate é pelo nome, para a ordem não depender da posição das colunas na matriz.
    ordenados = sorted(
        ((float(coeficientes[i]), nomes[i]) for i in range(len(nomes)) if bool(definidos[i])),
        key=lambda par: (-par[0], par[1]),
    )
    return PatternSearchResult(
        features=tuple(nome for _r, nome in ordenados),
        coefficient=tuple(r for r, _nome in ordenados),
        reference=referencia,
        reference_kind=tipo,
        profile=perfil,
        group_order=grupos,
        method=analise.correlation_method,
        n_samples_used=int(observacoes.shape[0]),
        n_features_total=matrix.n_features,
        n_features_used=len(ordenados),
        n_features_dropped_constant=n_constantes,
        n_features_dropped_non_finite=n_nao_finitas,
        n_coefficient_undefined=int(np.count_nonzero(~definidos)),
        column_digest=_digest_das_retidas(matrix, [nome for _r, nome in ordenados]),
    )

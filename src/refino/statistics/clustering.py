"""Agrupamento das amostras: ligação hierárquica, K-means e mapa auto-organizável.

As três famílias respondem 'que amostras se parecem?' sobre a mesma matriz em z-score por
feature, e por isso partilham aqui o filtro de colunas e o escalonamento. Sem o z-score, uma
feature de intensidade alta decidiria sozinha a distância euclidiana entre duas amostras; com
ele, multiplicar uma feature por mil não move o agrupamento. Coluna constante sai por
`ptp == 0`, e não por `sd == 0`, via `column_usability`: o desvio de uma coluna constante não é
exatamente zero em ponto flutuante.

A população é a matriz inteira, QC incluído. `statistics.correlacao` faz o contrário, porque lá
os QCs, réplicas da mesma mistura, inflariam a correlação entre features. Aqui a unidade é a
amostra, e um QC que caia no agrupamento das amostras de estudo é um achado técnico legítimo.

Nenhuma das três tem métrica única de qualidade nem escolhe `k`. A inércia do K-means e o erro
de quantização do SOM descrevem o ajuste e não ordenam alternativas: caem monotonicamente com
`k` e com o tamanho da grade.

No hierárquico o padrão é `ward` com `euclidean`, o do MetaboAnalyst, e não `average`. `ward`
exige `euclidean` porque minimiza a variância intra-grupo, que só está definida sobre distância
euclidiana; a guarda vive em `AnalysisSpec.__post_init__`. O agrupamento das amostras é barato.
O das features é quadrático em tempo e em memória, e por isso `cluster_features` é parâmetro.
`pdist` e `linkage` são chamadas únicas do `scipy`, sem ponto de parada por dentro: o
cancelamento é consultado entre as etapas, e a última consulta fica logo antes da ligação das
features, que leva cerca de 1,3 s com 6.035 features.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from refino.domain.analysis import PADRAO_DA_ANALISE, AnalysisSpec
from refino.domain.cancellation import CancelCheck, raise_if_cancelled
from refino.domain.labels import ClusterLinkage, ClusterMetric
from refino.metrics.structure import column_usability, retained_column_digest

if TYPE_CHECKING:
    from refino.domain.matrix import RefinoMatrix

__all__ = [
    "CLUSTER_LINKAGE_IMPLEMENTATIONS",
    "CLUSTER_METRIC_IMPLEMENTATIONS",
    "EPOCAS_DO_SOM",
    "IDENTIFICADOR_DO_KMEANS",
    "IDENTIFICADOR_DO_SOM",
    "INICIALIZACOES_DO_KMEANS",
    "RAIO_FINAL_DO_SOM",
    "TAXA_FINAL_DO_SOM",
    "TAXA_INICIAL_DO_SOM",
    "TETO_DE_FEATURES_PARA_COMPUTAR",
    "TETO_DE_FEATURES_PARA_DESENHAR",
    "THREADS_DO_KMEANS",
    "ClusteringResult",
    "KMeansResult",
    "ParticaoImpossivelError",
    "SomResult",
    "hierarchical_clustering",
    "kmeans_clustering",
    "self_organizing_map",
]


# ---------------------------------------------------------------------------
# Os dois tetos
# ---------------------------------------------------------------------------
#
# São duas grandezas: os 100 ms de interação, que é quanto um redesenho pode custar sem a janela
# parecer presa, e a espera de segundos do cálculo, que é o que o cancelamento endereça. Os dois
# foram medidos sobre uma matriz de 6.035 features x 68 amostras.

TETO_DE_FEATURES_PARA_DESENHAR: Final = 60
"""O maior número de features cujo redesenho fica abaixo dos 100 ms de interação.

Medido com rótulo de feature no eixo, em máquina ociosa: 92,7 ms com 60 features e 120,8 ms com
80. O custo é dos rótulos, e não do dado: sem rótulos o redesenho fica abaixo de 80 ms até 6.035
features. Quem quiser desenhar mais features que este teto pode apagar os rótulos do eixo. Com
a CPU a 100%, a travessia dos 100 ms cai para entre 20 e 30 features.
"""

TETO_DE_FEATURES_PARA_COMPUTAR: Final[int | None] = None
"""Sem limiar: nenhuma matriz medida leva o agrupamento das features aos 10 s.

10 s é onde Nielsen (1993, p. 135) põe a perda de atenção do usuário na tarefa. Com 6.035
features a ligação leva 1,33 s e 159,5 MiB. O custo é quadrático, e o pico de memória é a matriz
de distâncias condensada que o `scipy` constrói densa. Um número aqui exigiria medir uma matriz
maior que essa.
"""


def _distancia_euclidiana(observacoes: np.ndarray) -> np.ndarray:
    """A raiz da soma dos quadrados. A única compatível com `ward`."""
    return pdist(observacoes, metric="euclidean")


def _distancia_de_correlacao(observacoes: np.ndarray) -> np.ndarray:
    """`1 - r` de Pearson entre duas observações, sobre as variáveis.

    Mede forma, e não magnitude: dois perfis paralelos com intensidades diferentes ficam à
    distância zero.
    """
    return pdist(observacoes, metric="correlation")


def _distancia_de_spearman(observacoes: np.ndarray) -> np.ndarray:
    """`1 - rho`, a correlação sobre os postos de cada observação.

    Os postos são tomados dentro de cada observação (`axis=1`), e não por variável, porque o
    ponto é a observação. Difere de `correlation` sobre relação monotônica não linear; sobre
    dado linear as duas coincidem.
    """
    postos: np.ndarray = rankdata(observacoes, axis=1).astype(np.float64, copy=False)
    return pdist(postos, metric="correlation")


CLUSTER_METRIC_IMPLEMENTATIONS: Final[dict[ClusterMetric, Callable[[np.ndarray], np.ndarray]]] = {
    "euclidean": _distancia_euclidiana,
    "correlation": _distancia_de_correlacao,
    "spearman": _distancia_de_spearman,
}
"""Despacho por nome de distância. `tests/unit/test_labels.py` confere as chaves com o `Literal`.

É registro, e não `if/elif`, para um nome sem implementação, ou o contrário, reprovar no teste
de paridade.
"""

CLUSTER_LINKAGE_IMPLEMENTATIONS: Final[dict[ClusterLinkage, ClusterLinkage]] = {
    "ward": "ward",
    "average": "average",
    "complete": "complete",
    "single": "single",
}
"""O nome da ligação no vocabulário do domínio e o nome dela no `scipy`.

É um mapa de identidade para `tests/unit/test_labels.py` comparar as chaves com
`get_args(ClusterLinkage)` e acusar uma ligação que entre no `Literal` sem chegar aqui. Passar o
rótulo direto ao `scipy` faria o conjunto implementado ser o que o `scipy` aceita.
"""


@dataclass(frozen=True)
class ClusteringResult:
    """As duas ligações, as duas ordens de folha e a matriz escalonada que as produziu.

    `scaled` faz parte do resultado porque o mapa de calor desenha exatamente a matriz sobre a
    qual a distância foi calculada. Recalcular o z-score na figura poria a mesma conta em dois
    lugares.

    `feature_linkage` é `None`, e não um array vazio, quando não há o que ligar: uma matriz de
    uma feature só tem agrupamento de amostras e não tem agrupamento de features.
    """

    sample_linkage: np.ndarray
    """A matriz de ligação das amostras, `(n_samples - 1, 4)` no formato do `scipy`."""

    sample_order: tuple[int, ...]
    """Índices de amostra na ordem das folhas, da esquerda para a direita do dendrograma."""

    feature_linkage: np.ndarray | None
    """A ligação das features retidas, ou `None` quando não foi pedida ou não há duas."""

    feature_order: tuple[int, ...] | None
    """Índices dentro das retidas, na ordem das folhas. `None` junto com `feature_linkage`."""

    scaled: np.ndarray
    """`(n_samples, n_features_used)`: z-score por coluna, só as colunas retidas. Imutável."""

    feature_names: tuple[str, ...]
    """Os nomes das colunas retidas, na ordem das colunas de `scaled`."""

    metric: ClusterMetric
    linkage_method: ClusterLinkage

    n_features_used: int
    n_features_dropped_constant: int
    n_features_dropped_non_finite: int

    column_digest: str
    """`sha256` dos nomes retidos, porque contagem igual não é conjunto igual."""


def _escalona_por_feature(values: np.ndarray) -> np.ndarray:
    """z-score por coluna. As colunas já vêm filtradas: aqui nenhum desvio é zero."""
    media = values.mean(axis=0)
    desvio = values.std(axis=0, ddof=0)
    return (values - media) / desvio


@dataclass(frozen=True)
class _Escalonada:
    """A matriz pronta para medir distância entre amostras, e o que ficou de fora dela.

    As três famílias usam este mesmo pré-processamento. As duas contagens de descarte ficam
    separadas porque são fatos diferentes: constância é informação sobre o dado, e não finito é
    o que o processamento produziu.
    """

    valores: np.ndarray
    """`(n_amostras, n_features_usadas)`: z-score por coluna, só as colunas retidas. Imutável."""

    nomes: tuple[str, ...]
    usaveis: np.ndarray
    n_constantes: int
    n_nao_finitas: int


def _escalonada_ou_recusa(matrix: RefinoMatrix, *, de: str) -> _Escalonada:
    """O z-score por feature das colunas utilizáveis, ou a recusa que nomeia as duas causas.

    `de` é o nome da família por extenso, como `"agrupar"` ou `"o K-means"`, e só entra na
    mensagem.
    """
    usaveis, n_constantes, n_nao_finitas = column_usability(matrix.values)
    n_usadas = int(usaveis.sum())
    if n_usadas == 0:
        raise ValueError(
            f"nenhuma das {matrix.n_features} features é utilizável para {de}: "
            f"{n_constantes} são constantes e {n_nao_finitas} têm valor não finito. Coluna "
            "constante não tem z-score definido, e coluna não finita contamina toda distância "
            "que a inclua."
        )
    escalonada = _escalona_por_feature(matrix.values[:, usaveis])
    escalonada.flags.writeable = False
    return _Escalonada(
        valores=escalonada,
        nomes=tuple(
            nome
            for nome, mantem in zip(matrix.feature_names, usaveis.tolist(), strict=True)
            if mantem
        ),
        usaveis=usaveis,
        n_constantes=n_constantes,
        n_nao_finitas=n_nao_finitas,
    )


def _exige_duas_amostras(matrix: RefinoMatrix, *, de: str) -> None:
    """Recusa a matriz de uma linha só, antes do escalonamento.

    Com uma observação não há par, distância nem partição. Além disso todas as colunas seriam
    constantes, e a recusa viria adiante como 'nenhuma feature utilizável', que é o diagnóstico
    errado.
    """
    if matrix.n_samples < 2:
        raise ValueError(
            f"{de} precisa de ao menos duas amostras e a matriz tem {matrix.n_samples}. Com uma "
            "observação não há par, não há distância e não há partição — e o z-score por feature "
            "seria uma divisão por zero em todas as colunas, o que faria isto reprovar mais "
            "adiante como «nenhuma feature utilizável», que é diagnóstico errado."
        )


def _liga(
    observacoes: np.ndarray,
    *,
    metric: ClusterMetric,
    linkage_method: ClusterLinkage,
    eixo: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Distância condensada e ligação sobre um eixo. Devolve a ligação e a ordem das folhas."""
    distancias = CLUSTER_METRIC_IMPLEMENTATIONS[metric](observacoes)
    if not np.isfinite(distancias).all():
        raise ValueError(
            f"a distância `{metric}` entre {eixo} produziu valor não finito. Com `correlation` ou "
            f"`spearman` isto acontece quando alguma observação é constante ao longo das "
            f"variáveis: a correlação dela com qualquer outra é indefinida, e a ligação recusaria "
            f"a matriz mais adiante com uma mensagem do `scipy`. Use `euclidean`, ou remova a "
            f"observação constante."
        )
    ligacao = linkage(distancias, method=CLUSTER_LINKAGE_IMPLEMENTATIONS[linkage_method])
    ordem = tuple(int(i) for i in leaves_list(ligacao))
    return ligacao, ordem


def hierarchical_clustering(
    matrix: RefinoMatrix,
    analise: AnalysisSpec = PADRAO_DA_ANALISE,
    *,
    cluster_features: bool = True,
    cancelled: CancelCheck | None = None,
) -> ClusteringResult:
    """Agrupa amostras e, se pedido, features, sobre a matriz em z-score por feature.

    Recebe `AnalysisSpec`, e não a distância e a ligação soltas, porque a combinação de `ward`
    com distância não euclidiana é recusada em `AnalysisSpec.__post_init__`. Com parâmetros
    soltos a recusa viria do `scipy`, no meio do cálculo.

    `cluster_features` é parâmetro de chamada, e não campo da especificação, porque não é
    escolha metodológica: o dendrograma das amostras não precisa do eixo das features, e o mapa
    de calor precisa.

    A matriz chega processada. Esta função não imputa, não normaliza e não transforma. O único
    tratamento é o z-score por coluna, que é da distância.
    """
    _exige_duas_amostras(matrix, de="agrupamento hierárquico")

    raise_if_cancelled(cancelled, "agrupamento: antes do escalonamento por feature")

    preparada = _escalonada_ou_recusa(matrix, de="agrupar")
    escalonada = preparada.valores
    retidos = preparada.nomes
    n_usadas = len(retidos)

    raise_if_cancelled(cancelled, "agrupamento: antes da ligação das amostras")
    ligacao_das_amostras, ordem_das_amostras = _liga(
        escalonada,
        metric=analise.cluster_metric,
        linkage_method=analise.cluster_linkage,
        eixo="amostras",
    )

    ligacao_das_features: np.ndarray | None = None
    ordem_das_features: tuple[int, ...] | None = None
    if cluster_features and n_usadas >= 2:
        # Último ponto de parada antes do trecho caro: a ligação das features é quadrática em
        # tempo e em memória, e corre em duas chamadas do `scipy` que não se interrompem.
        raise_if_cancelled(cancelled, "agrupamento: antes da ligação das features")
        ligacao_das_features, ordem_das_features = _liga(
            escalonada.T,
            metric=analise.cluster_metric,
            linkage_method=analise.cluster_linkage,
            eixo="features",
        )

    return ClusteringResult(
        sample_linkage=ligacao_das_amostras,
        sample_order=ordem_das_amostras,
        feature_linkage=ligacao_das_features,
        feature_order=ordem_das_features,
        scaled=escalonada,
        feature_names=retidos,
        metric=analise.cluster_metric,
        linkage_method=analise.cluster_linkage,
        n_features_used=n_usadas,
        n_features_dropped_constant=preparada.n_constantes,
        n_features_dropped_non_finite=preparada.n_nao_finitas,
        column_digest=retained_column_digest(matrix.feature_names, preparada.usaveis),
    )


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------

IDENTIFICADOR_DO_KMEANS: Final = 11
IDENTIFICADOR_DO_SOM: Final = 12
"""Os dois identificadores de unidade que derivam o gerador aleatório de `(seed, unidade)`.

As duas famílias correm na mesma execução, com a mesma `PipelineSpec.seed`. Com a semente crua,
o K-means e o SOM consumiriam a mesma sequência.

A derivação é por `SeedSequence`, e não por soma, porque somar componentes colide:
`(seed=1, unidade=12)` e `(seed=2, unidade=11)` dariam o mesmo gerador.
"""

INICIALIZACOES_DO_KMEANS: Final = 10
"""Quantos arranques independentes o K-means faz, ficando com o de menor inércia.

É explícito porque o padrão do `scikit-learn` mudou entre versões, de `10` até a 1.2 para
`"auto"` depois, e um padrão implícito faria o resultado depender da versão instalada. São dez,
e não um, porque o K-means converge para um mínimo local que depende de onde os centros caem.
"""


THREADS_DO_KMEANS: Final = 1
"""Quantas threads de OpenMP o `KMeans` pode usar durante o ajuste.

O `KMeans` do `scikit-learn` soma a inércia por redução paralela em OpenMP, e a ordem em que as
parcelas de cada thread se juntam depende do escalonamento. Com as threads livres, duas
execuções com a mesma semente e a mesma entrada podem dar inércias que diferem no último bit, e
a figura mostra esse número. `random_state` fixo não garante igualdade bit a bit quando o
estimador reduz em paralelo.

O limite vale só em volta do ajuste, e não no processo inteiro: o SERRF e os classificadores
paralelizam por `joblib`, que não tem esse defeito.
"""


class ParticaoImpossivelError(ValueError):
    """Recusa: o que se pediu não cabe na população.

    A mensagem nomeia os dois números em conflito, para quem pediu `k` saber contra que limite
    bateu. É classe própria, e não `ValueError` comum, para quem chama distinguir um pedido não
    satisfazível, que a rota devolve como 422, de outro erro de valor, que é defeito.
    """


@dataclass(frozen=True)
class KMeansResult:
    """A partição das amostras em `k` agrupamentos, e o que a torna interpretável.

    `groups` viaja com a partição porque a pergunta é se a partição que o método achou sem ver
    os rótulos coincide com os grupos clínicos. Sem o vetor clínico ao lado, a figura mostraria
    `k` barras sem nada com que as comparar.
    """

    labels: tuple[int, ...]
    """O agrupamento de cada amostra, em `[0, k)`, na ordem das linhas da matriz."""

    sample_names: tuple[str, ...]
    groups: tuple[str, ...]
    """O grupo clínico por amostra, na mesma ordem. `QC` é um grupo como os outros: ver o
    cabeçalho do módulo."""

    k: int
    inertia: float
    """A soma dos quadrados das distâncias de cada amostra ao centro dela.

    Descreve o ajuste e não ordena alternativas: cai monotonicamente com `k`, então escolher `k`
    por ela é escolher `k = n`. É a quantidade que o método minimiza."""

    n_iter: int
    """Quantas iterações o melhor arranque levou. É diagnóstico."""

    seed: int
    """A semente derivada que produziu esta partição, e não a global. A figura a declara, porque
    sem ela a partição não é reproduzível por quem lê."""

    n_samples: int
    n_features_used: int
    n_features_total: int
    n_features_dropped_constant: int
    n_features_dropped_non_finite: int

    column_digest: str
    """`sha256` dos nomes retidos, porque contagem igual não é conjunto igual."""

    @property
    def cluster_sizes(self) -> tuple[int, ...]:
        """Quantas amostras em cada agrupamento, na ordem dos rótulos. Derivado de `labels`."""
        contagem = [0] * self.k
        for rotulo in self.labels:
            contagem[rotulo] += 1
        return tuple(contagem)


def _semente_derivada(seed: int, unidade: int) -> int:
    """Um inteiro de fluxo próprio para `(seed, unidade)`. Ver `IDENTIFICADOR_DO_KMEANS`."""
    return int(np.random.default_rng([seed, unidade]).integers(2**31))


def kmeans_clustering(
    matrix: RefinoMatrix,
    analise: AnalysisSpec = PADRAO_DA_ANALISE,
    *,
    seed: int,
    cancelled: CancelCheck | None = None,
) -> KMeansResult:
    """Parte as amostras em `analise.kmeans_k` agrupamentos, sobre a matriz em z-score por feature.

    Recebe `AnalysisSpec`, e não `k` solto, porque `kmeans_k >= 1` é validado em
    `AnalysisSpec.__post_init__`.

    `seed` é obrigatório e sem valor padrão: toda operação estocástica recebe semente declarada.
    Ele é a `PipelineSpec.seed` da execução, e o gerador desta família é derivado dela.

    `k > n_amostras` é recusado, e não reduzido: reduzir daria uma figura bem formada com um
    número de agrupamentos que ninguém pediu. `k == n` é aceito: cada amostra é o seu próprio
    agrupamento e a inércia é zero.

    A matriz chega processada. Esta função não imputa, não normaliza e não transforma.
    """
    _exige_duas_amostras(matrix, de="o K-means")
    if analise.kmeans_k > matrix.n_samples:
        raise ParticaoImpossivelError(
            f"`kmeans_k` é {analise.kmeans_k} e a matriz tem {matrix.n_samples} amostras: não há "
            "como partir uma população em mais agrupamentos do que ela tem unidades. Reduzir o "
            "pedido em silêncio daria uma figura bem formada com um número de agrupamentos que "
            "ninguém pediu."
        )

    raise_if_cancelled(cancelled, "k-means: antes do escalonamento por feature")
    preparada = _escalonada_ou_recusa(matrix, de="o K-means")

    # Último ponto de parada: o `KMeans` do `scikit-learn` é uma chamada única, e cancelar
    # depois daqui espera os dez arranques.
    raise_if_cancelled(cancelled, "k-means: antes do ajuste")
    semente = _semente_derivada(seed, IDENTIFICADOR_DO_KMEANS)
    modelo = KMeans(
        n_clusters=analise.kmeans_k,
        n_init=INICIALIZACOES_DO_KMEANS,
        random_state=semente,
    )
    # OpenMP a uma thread, só em volta do ajuste. Ver `THREADS_DO_KMEANS`.
    with threadpool_limits(limits=THREADS_DO_KMEANS, user_api="openmp"):
        rotulos = modelo.fit_predict(preparada.valores)

    return KMeansResult(
        labels=tuple(int(r) for r in rotulos.tolist()),
        sample_names=tuple(amostra.name for amostra in matrix.samples),
        groups=tuple(amostra.group for amostra in matrix.samples),
        k=analise.kmeans_k,
        inertia=float(modelo.inertia_),
        n_iter=int(modelo.n_iter_),
        seed=semente,
        n_samples=matrix.n_samples,
        n_features_used=len(preparada.nomes),
        n_features_total=matrix.n_features,
        n_features_dropped_constant=preparada.n_constantes,
        n_features_dropped_non_finite=preparada.n_nao_finitas,
        column_digest=retained_column_digest(matrix.feature_names, preparada.usaveis),
    )


# ---------------------------------------------------------------------------
# Mapa auto-organizável
# ---------------------------------------------------------------------------

TAXA_INICIAL_DO_SOM: Final = 0.5
TAXA_FINAL_DO_SOM: Final = 0.01
RAIO_FINAL_DO_SOM: Final = 0.5
"""As três pontas do decaimento da taxa e do raio, que é geométrico.

A forma clássica, `taxa(t) = taxa0 · exp(-t/escala)` com `escala = épocas / log(raio0)`, falha
numa grade de uma unidade: o raio inicial vale `0,5`, o logaritmo é negativo e o raio cresce em
vez de decrescer. A forma geométrica `v(t) = v0 · (v_fim/v0)^(t/(T-1))` interpola entre dois
números escolhidos e é monotônica por construção.

O raio final é meia unidade, para a fase final ser competitiva: a vizinhança de um vizinho
imediato vale `exp(-1/(2·0,25)) = 0,135` e a de um vizinho a duas casas vale `3,4e-4`, então o
ajuste do fim é quase só da unidade vencedora.
"""

EPOCAS_DO_SOM: Final = 100
"""Quantas passagens sobre as amostras o mapa faz, por omissão.

É parâmetro de algoritmo, e não limiar. Medido na grade 4x4 com 68 amostras e 6.035 features,
com duas sementes: o erro de quantização ainda cai até 400 épocas, sem ponto de achatamento, e o
custo é linear nas épocas (1,23 s em 100, 4,88 s em 400). Com 100 fica um resíduo de cerca de 2%
do erro em relação a 400.

O número vale para a grade em que foi medido. Uma grade maior tem mais unidades para ordenar e
precisa de mais épocas, e por isso `epochs` é parâmetro de `self_organizing_map`.
"""


@dataclass(frozen=True)
class SomResult:
    """Em que unidade da grade cada amostra caiu, e a grade que as recebeu.

    `units` é o índice plano `linha · colunas + coluna`, e não um par de coordenadas: um par
    teria de ser recombinado por cada consumidor, e trocar linha por coluna numa grade não
    quadrada daria uma figura bem formada com as amostras no lugar errado.

    O erro de quantização descreve o ajuste e não ordena grades: cai monotonicamente com o
    número de unidades.
    """

    units: tuple[int, ...]
    """A unidade de cada amostra, em `[0, rows·cols)`, na ordem das linhas da matriz."""

    sample_names: tuple[str, ...]
    groups: tuple[str, ...]

    rows: int
    cols: int
    epochs: int
    seed: int
    """A semente derivada, como em `KMeansResult.seed`."""

    quantization_error: float
    """A distância média de cada amostra à unidade que a venceu, na escala do z-score."""

    n_samples: int
    n_features_used: int
    n_features_total: int
    n_features_dropped_constant: int
    n_features_dropped_non_finite: int

    column_digest: str

    @property
    def n_units(self) -> int:
        return self.rows * self.cols

    @property
    def unit_sizes(self) -> tuple[int, ...]:
        """Quantas amostras em cada unidade, na ordem do índice plano. Derivado de `units`."""
        contagem = [0] * self.n_units
        for unidade in self.units:
            contagem[unidade] += 1
        return tuple(contagem)

    @property
    def n_units_empty(self) -> int:
        """Quantas unidades não receberam amostra nenhuma.

        Não é defeito: as unidades vazias são a fronteira entre agrupamentos vizinhos, e a
        figura as declara.
        """
        return sum(1 for tamanho in self.unit_sizes if tamanho == 0)


def _coordenadas_da_grade(rows: int, cols: int) -> np.ndarray:
    """`(rows·cols, 2)` com a linha e a coluna de cada unidade, na ordem do índice plano."""
    linhas, colunas = np.divmod(np.arange(rows * cols, dtype=int), cols)
    return np.column_stack((linhas, colunas)).astype(float)


def _decaimento(inicial: float, final: float, passo: int, total: int) -> float:
    """A interpolação geométrica entre `inicial` e `final`. Ver `RAIO_FINAL_DO_SOM`."""
    if total <= 1:
        return inicial
    return float(inicial * (final / inicial) ** (passo / (total - 1)))


def self_organizing_map(
    matrix: RefinoMatrix,
    analise: AnalysisSpec = PADRAO_DA_ANALISE,
    *,
    seed: int,
    epochs: int = EPOCAS_DO_SOM,
    cancelled: CancelCheck | None = None,
) -> SomResult:
    """Ordena as amostras numa grade `som_rows x som_cols`. Implementação própria.

    O `scikit-learn` não tem mapa auto-organizável. O algoritmo: para cada amostra, achar a
    unidade mais próxima por distância euclidiana e puxar para ela a vencedora e a vizinhança
    dela na grade, com taxa e raio que decrescem ao longo das épocas. A vizinhança é medida na
    grade, e não no espaço das features, e é ela que põe amostras parecidas em unidades
    vizinhas. É o que distingue o mapa de um K-means.

    Os pesos começam em amostras sorteadas, e não em ruído. Pesos aleatórios num espaço de
    milhares de dimensões ficam todos à mesma distância de tudo, e a primeira época se gastaria
    a trazer a grade para perto da nuvem. O sorteio é com reposição quando há menos amostras do
    que unidades, que é o caso de uma grade maior que o número de amostras.

    A ordem de apresentação das amostras faz parte da entrada: duas ordens dão dois mapas. Cada
    época permuta as amostras com o mesmo gerador semeado, consumido em sequência e sem
    paralelismo, então a mesma entrada e a mesma semente dão o mesmo mapa.

    O cancelamento é consultado a cada época, uma passagem sobre as amostras.

    `epochs` tem o padrão `EPOCAS_DO_SOM`, e não é campo de `AnalysisSpec`.
    """
    _exige_duas_amostras(matrix, de="o mapa auto-organizável")
    if epochs < 1:
        raise ParticaoImpossivelError(
            f"`epochs` é {epochs} e o mínimo é 1: sem uma passagem sobre as amostras a grade fica "
            "com os pesos da inicialização, e o resultado seria o sorteio com o nome de um mapa."
        )

    raise_if_cancelled(cancelled, "som: antes do escalonamento por feature")
    preparada = _escalonada_ou_recusa(matrix, de="o mapa auto-organizável")
    observacoes = np.asarray(preparada.valores, dtype=float)

    rows, cols = analise.som_rows, analise.som_cols
    n_unidades = rows * cols
    coordenadas = _coordenadas_da_grade(rows, cols)
    gerador = np.random.default_rng([seed, IDENTIFICADOR_DO_SOM])
    semente = _semente_derivada(seed, IDENTIFICADOR_DO_SOM)

    escolhidas = gerador.choice(
        matrix.n_samples, size=n_unidades, replace=n_unidades > matrix.n_samples
    )
    pesos = observacoes[np.asarray(escolhidas, dtype=int), :].astype(float, copy=True)

    # O raio inicial é metade do maior lado, para a vizinhança da época zero alcançar a grade
    # inteira, com piso no raio final: um raio inicial menor que o final inverteria o
    # decaimento.
    raio_inicial = max(float(max(rows, cols)) / 2.0, RAIO_FINAL_DO_SOM)

    for epoca in range(epochs):
        raise_if_cancelled(cancelled, f"som: época {epoca + 1} de {epochs}")
        taxa = _decaimento(TAXA_INICIAL_DO_SOM, TAXA_FINAL_DO_SOM, epoca, epochs)
        raio = _decaimento(raio_inicial, RAIO_FINAL_DO_SOM, epoca, epochs)
        for indice in gerador.permutation(matrix.n_samples).tolist():
            amostra = observacoes[int(indice), :]
            vencedora = int(np.argmin(((pesos - amostra) ** 2).sum(axis=1)))
            distancia = ((coordenadas - coordenadas[vencedora]) ** 2).sum(axis=1)
            vizinhanca = np.exp(-distancia / (2.0 * raio * raio))
            pesos += taxa * vizinhanca[:, None] * (amostra - pesos)

    quadrados = ((observacoes[:, None, :] - pesos[None, :, :]) ** 2).sum(axis=2)
    unidades = np.argmin(quadrados, axis=1)
    erro = float(np.sqrt(quadrados[np.arange(matrix.n_samples), unidades]).mean())

    return SomResult(
        units=tuple(int(u) for u in unidades.tolist()),
        sample_names=tuple(amostra.name for amostra in matrix.samples),
        groups=tuple(amostra.group for amostra in matrix.samples),
        rows=rows,
        cols=cols,
        epochs=epochs,
        seed=semente,
        quantization_error=erro,
        n_samples=matrix.n_samples,
        n_features_used=len(preparada.nomes),
        n_features_total=matrix.n_features,
        n_features_dropped_constant=preparada.n_constantes,
        n_features_dropped_non_finite=preparada.n_nao_finitas,
        column_digest=retained_column_digest(matrix.feature_names, preparada.usaveis),
    )

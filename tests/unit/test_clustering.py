"""O agrupamento hierárquico: a topologia, o escalonamento, e as duas pontas da recusa.

## O que este módulo afirma, e o que ele deliberadamente NÃO afirma

Toda asserção sobre o resultado do agrupamento é sobre **topologia** — a partição que um corte da
árvore produz, e a ordem das folhas —, nunca sobre coordenadas. Distância e altura de ramo mudam
com o escalonamento, com a versão do `scipy` e com a ordem das colunas; a partição de um dado
separado por construção não muda, e é ela que responde à pergunta que o pesquisador faz.

**Nenhuma asserção de tempo de parede.** Os dois tetos são medidos por
`scripts/mede_custo_do_agrupamento.py` e **declarados** nas constantes de `statistics/clustering.py`
— é o que o portão de `tests/test_orcamento_contrato.py`
exige: comparação de tempo de parede escrita à mão reprova sob contenção sem distinguir regressão
de máquina ocupada. O que fica aqui é a âncora de que a constante não derivou sem ninguém a
remedir.

## A planta do escalonamento, e por que ela cai sobre dado em que as duas coisas DIFEREM

*"Multiplicar uma feature por mil não move o agrupamento"* é verdade sobre quase qualquer matriz —
inclusive sobre uma em que o escalonamento não aconteça, se a feature multiplicada for irrelevante.
A planta que separa é construída: uma feature de escala enorme cuja variação é **ortogonal** ao
grupo, e que parte as amostras por paridade. Com escalonamento por feature a partição sai por
**grupo**; sem ele sai por **paridade**, e as duas são disjuntas por construção. O teste afirma as
duas metades, e é a segunda que faz a primeira significar alguma coisa.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from refino.domain.analysis import AnalysisSpec
from refino.domain.cancellation import CancelCheck, ExecutionCancelledError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.io.readers import load_matrix
from refino.plots.base import (
    FeaturesDemaisError,
    MatrizVaziaError,
    RecusaDoConstrutorError,
    exige_abaixo_do_teto,
)
from refino.statistics.clustering import (
    CLUSTER_LINKAGE_IMPLEMENTATIONS,
    CLUSTER_METRIC_IMPLEMENTATIONS,
    EPOCAS_DO_SOM,
    IDENTIFICADOR_DO_KMEANS,
    IDENTIFICADOR_DO_SOM,
    INICIALIZACOES_DO_KMEANS,
    TETO_DE_FEATURES_PARA_COMPUTAR,
    TETO_DE_FEATURES_PARA_DESENHAR,
    THREADS_DO_KMEANS,
    ClusteringResult,
    ParticaoImpossivelError,
    hierarchical_clustering,
    kmeans_clustering,
    self_organizing_map,
)
from tests.conftest import measurements_path, sample_list_path

MODO = "pos_hilic"

# A subamostra do corpus real usada onde o teste precisa de dado real e não do corpus inteiro.
# **É a mesma semente do arnês de custo**, para que a população deste módulo e a da medição dos
# tetos sejam a mesma sequência encaixada — dois cortes diferentes do mesmo corpus dariam dois
# números com o mesmo nome.
SEMENTE_DA_PERMUTACAO = 20260913
FEATURES_DO_CORTE_REAL = 300


# ---------------------------------------------------------------------------
# As fixturas sintéticas
# ---------------------------------------------------------------------------


def matriz_de(valores: np.ndarray, *, grupos: Sequence[str] | None = None) -> RefinoMatrix:
    """Uma `RefinoMatrix` sintética a partir de um array, com grupos opcionais por linha."""
    array = np.ascontiguousarray(valores, dtype=np.float64)
    array.flags.writeable = False
    n_amostras, n_features = array.shape
    nomes_de_grupo = list(grupos) if grupos is not None else ["G"] * n_amostras
    amostras = tuple(
        SampleMetadata(
            name=f"s{i:03d}",
            injection_order=i + 1,
            sample_type="sample",
            group=nomes_de_grupo[i],
        )
        for i in range(n_amostras)
    )
    return RefinoMatrix(
        values=array,
        feature_names=tuple(f"f{j:03d}" for j in range(n_features)),
        samples=amostras,
        mode="sintetico",
    )


def dois_grupos(
    *, por_grupo: int = 6, n_features: int = 5, separacao: float = 4.0, seed: int = 20260913
) -> RefinoMatrix:
    """Dois grupos separados por construção: `A` em torno de 0, `B` em torno de `separacao`."""
    rng = np.random.default_rng(seed)
    valores = np.vstack(
        [
            rng.normal(0.0, 1.0, (por_grupo, n_features)),
            rng.normal(separacao, 1.0, (por_grupo, n_features)),
        ]
    )
    return matriz_de(valores, grupos=["A"] * por_grupo + ["B"] * por_grupo)


def com_feature_dominante(*, por_grupo: int = 6, seed: int = 20260913) -> RefinoMatrix:
    """Dois grupos, mais UMA feature de escala enorme que parte as amostras por PARIDADE.

    A feature dominante tem desvio de 1e5 contra o 1,0 das informativas: sem escalonamento por
    feature ela decide sozinha toda distância euclidiana. E o que ela codifica — o índice par ou
    ímpar da amostra — corta os dois grupos ao meio, de modo que as duas partições possíveis são
    **disjuntas**: não há como confundir «o escalonamento aconteceu» com «o dado era fácil».
    """
    base = dois_grupos(por_grupo=por_grupo, seed=seed)
    n = base.n_samples
    dominante = 1.0e6 + 1.0e5 * (np.arange(n) % 2)
    return matriz_de(
        np.column_stack([base.values, dominante]),
        grupos=[s.group for s in base.samples],
    )


def com_feature_de_ruido(*, por_grupo: int = 6, seed: int = 20260913) -> RefinoMatrix:
    """Dois grupos, mais UMA feature de ruído que não carrega grupo nenhum.

    É a feature que o teste do fator mil multiplica. Multiplicar uma feature **informativa** não
    provaria nada: amplificar o sinal que já separa mantém a partição com ou sem escalonamento.
    """
    base = dois_grupos(por_grupo=por_grupo, seed=seed)
    ruido = np.random.default_rng(seed + 1).normal(0.0, 1.0, base.n_samples)
    return matriz_de(
        np.column_stack([base.values, ruido]),
        grupos=[s.group for s in base.samples],
    )


def particao_canonica(rotulos: Sequence[int]) -> tuple[int, ...]:
    """A partição com os rótulos renumerados pela ordem da primeira ocorrência.

    `fcluster` devolve rótulos arbitrários: a mesma partição pode sair `[1,1,2,2]` ou `[2,2,1,1]`.
    Comparar os vetores crus acusaria diferença onde não há nenhuma.
    """
    mapa: dict[int, int] = {}
    for rotulo in rotulos:
        mapa.setdefault(int(rotulo), len(mapa))
    return tuple(mapa[int(rotulo)] for rotulo in rotulos)


def particao_das_amostras(resultado: ClusteringResult, k: int = 2) -> tuple[int, ...]:
    """O corte da árvore das amostras em `k` grupos, canonizado."""
    return particao_canonica(fcluster(resultado.sample_linkage, k, criterion="maxclust"))


def particao_crua(valores: np.ndarray, k: int = 2) -> tuple[int, ...]:
    """A mesma pergunta SEM escalonamento por feature — a outra metade do par causal."""
    return particao_canonica(fcluster(linkage(pdist(valores), method="ward"), k, "maxclust"))


# ---------------------------------------------------------------------------
# 1. A topologia
# ---------------------------------------------------------------------------


def test_dois_grupos_separados_por_construcao_caem_nos_dois_ramos() -> None:
    """O corte em dois da árvore das amostras reproduz os grupos. **Topologia, não coordenada.**"""
    matriz = dois_grupos()
    resultado = hierarchical_clustering(matriz)
    esperada = particao_canonica([0 if s.group == "A" else 1 for s in matriz.samples])
    assert particao_das_amostras(resultado) == esperada


def test_a_ordem_das_folhas_poe_cada_grupo_num_bloco_contiguo() -> None:
    """O dendrograma não cruza os dois ramos — é o que o desenho vai mostrar."""
    matriz = dois_grupos()
    resultado = hierarchical_clustering(matriz)
    grupos_na_ordem = [matriz.samples[i].group for i in resultado.sample_order]
    trocas = sum(1 for a, b in pairwise(grupos_na_ordem) if a != b)
    assert trocas == 1, f"a ordem das folhas alterna entre grupos: {grupos_na_ordem}"


def test_as_features_tambem_sao_agrupadas_e_a_ordem_e_uma_permutacao() -> None:
    matriz = dois_grupos()
    resultado = hierarchical_clustering(matriz)
    assert resultado.feature_linkage is not None
    assert resultado.feature_linkage.shape == (matriz.n_features - 1, 4)
    assert resultado.feature_order is not None
    assert sorted(resultado.feature_order) == list(range(matriz.n_features))


def test_pedir_so_as_amostras_nao_liga_as_features() -> None:
    """`cluster_features=False` é o caminho do dendrograma, e ele não paga o eixo quadrático."""
    resultado = hierarchical_clustering(dois_grupos(), cluster_features=False)
    assert resultado.feature_linkage is None
    assert resultado.feature_order is None


# ---------------------------------------------------------------------------
# 2. O escalonamento por feature — o par causal
# ---------------------------------------------------------------------------


def test_o_escalonamento_por_feature_acontece() -> None:
    """Com a feature dominante presente, a partição sai por GRUPO. Metade um."""
    matriz = com_feature_dominante()
    resultado = hierarchical_clustering(matriz)
    por_grupo = particao_canonica([0 if s.group == "A" else 1 for s in matriz.samples])
    assert particao_das_amostras(resultado) == por_grupo


def test_a_planta_do_escalonamento_cai_sobre_dado_em_que_as_duas_coisas_DIFEREM() -> None:  # noqa: N802
    """Metade dois: SEM escalonamento a mesma matriz parte por PARIDADE, e as duas são disjuntas.

    **Sem esta metade a primeira seria vácua.** Um agrupamento que ignorasse o escalonamento
    passaria no teste de cima sempre que o dado fosse fácil o bastante — e o dado de teste é,
    por construção, fácil. O que torna a planta uma planta é ela cair sobre uma matriz em que as
    duas políticas produzem partições que não se parecem.
    """
    matriz = com_feature_dominante()
    por_grupo = particao_canonica([0 if s.group == "A" else 1 for s in matriz.samples])
    por_paridade = particao_canonica([i % 2 for i in range(matriz.n_samples)])
    assert por_grupo != por_paridade, "a planta não separa: as duas partições coincidem"

    assert particao_crua(matriz.values) == por_paridade, (
        "sem escalonamento por feature, a matriz da planta deveria partir por paridade — se não "
        "parte, a feature dominante deixou de dominar e a planta parou de separar"
    )
    assert particao_das_amostras(hierarchical_clustering(matriz)) == por_grupo


def test_multiplicar_uma_feature_por_mil_nao_move_o_agrupamento() -> None:
    """A consequência verificável do z-score por coluna."""
    matriz = com_feature_de_ruido()
    escalada = matriz.values.copy()
    escalada[:, -1] *= 1000.0
    outra = matriz_de(escalada, grupos=[s.group for s in matriz.samples])

    antes = hierarchical_clustering(matriz)
    depois = hierarchical_clustering(outra)
    assert depois.sample_order == antes.sample_order
    assert particao_das_amostras(depois) == particao_das_amostras(antes)


def test_a_planta_do_fator_mil_separa_sem_o_escalonamento() -> None:
    """E sem z-score, multiplicar por mil MOVE a partição — senão o teste de cima é vácuo.

    **A primeira versão deste teste REPROVOU, e a reprovação é o achado.** Ela multiplicava a
    feature **0**, que é informativa: amplificar o sinal que já separa os grupos não move partição
    nenhuma, com ou sem escalonamento. O teste irmão passava, e passaria também sobre uma
    implementação sem z-score — vacuidade que só apareceu porque o par causal foi escrito.

    A feature que tem de ser multiplicada é a que **não** carrega o grupo: com z-score ela continua
    a valer um desvio como todas as outras, e sem z-score ela passa a valer mil.
    """
    matriz = com_feature_de_ruido()
    escalada = matriz.values.copy()
    escalada[:, -1] *= 1000.0
    assert particao_crua(escalada) != particao_crua(matriz.values)


def test_a_matriz_escalonada_e_imutavel() -> None:
    """`scaled` viaja para o mapa de calor; matriz de saída que se pode escrever não é
    resultado, é estado partilhado (regra 7 do README)."""
    resultado = hierarchical_clustering(dois_grupos())
    assert not resultado.scaled.flags.writeable
    with pytest.raises(ValueError):
        resultado.scaled[0, 0] = 1.0


# ---------------------------------------------------------------------------
# 3. Ward exige euclidiana — a guarda é do MÉTODO, e é UMA só
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distancia", ["correlation", "spearman"])
def test_ward_com_distancia_nao_euclidiana_e_recusado(distancia: str) -> None:
    """A recusa vem do contrato, e a mensagem diz que é restrição do método.

    **O teste é sobre `AnalysisSpec` e não sobre uma segunda guarda em `clustering.py`.** Não há
    segunda guarda: `hierarchical_clustering` recebe o contrato já construído, e um contrato que
    não se pode construir inválido é mais barato que duas cópias da mesma verificação.
    Este teste é a evidência de que a única guarda cobre o caminho.
    """
    with pytest.raises(ValueError) as capturado:
        AnalysisSpec(cluster_linkage="ward", cluster_metric=distancia)  # type: ignore[arg-type]
    mensagem = str(capturado.value)
    assert "ward" in mensagem
    assert "variância intra-grupo" in mensagem
    assert "Não é escolha nossa" in mensagem


def test_ward_com_euclidiana_e_o_padrao_e_nao_levanta() -> None:
    """A outra direção: a combinação padrão constrói, e é a do MetaboAnalyst lido na fonte."""
    padrao = AnalysisSpec()
    assert padrao.cluster_linkage == "ward"
    assert padrao.cluster_metric == "euclidean"


@pytest.mark.parametrize("ligacao", ["average", "complete", "single"])
def test_as_outras_ligacoes_aceitam_distancia_nao_euclidiana(ligacao: str) -> None:
    contrato = AnalysisSpec(cluster_linkage=ligacao, cluster_metric="correlation")  # type: ignore[arg-type]
    resultado = hierarchical_clustering(dois_grupos(), contrato)
    assert resultado.linkage_method == ligacao
    assert resultado.metric == "correlation"


# ---------------------------------------------------------------------------
# 4. Os dois registos de despacho
# ---------------------------------------------------------------------------


def test_as_tres_distancias_produzem_matriz_condensada_do_tamanho_certo() -> None:
    observacoes = dois_grupos().values
    esperado = observacoes.shape[0] * (observacoes.shape[0] - 1) // 2
    for nome, funcao in CLUSTER_METRIC_IMPLEMENTATIONS.items():
        assert funcao(observacoes).shape == (esperado,), nome


def test_spearman_e_correlation_divergem_sobre_dado_monotonico_nao_linear() -> None:
    """As duas medem associação e **não são intercambiáveis** — e a planta cai onde diferem.

    Sobre dado linear as duas coincidem, e um teste ali seria vácuo: é a planta que
    tem de separar. `exp` sobre uma rampa é monotônico e não linear, que é exactamente o regime em
    que o posto preserva a relação e o produto-momento não.
    """
    rampa = np.linspace(0.0, 3.0, 12)
    linear = np.vstack([rampa, 2.0 * rampa + 1.0, -rampa])
    nao_linear = np.vstack([rampa, np.exp(3.0 * rampa), -rampa])

    de_pearson = CLUSTER_METRIC_IMPLEMENTATIONS["correlation"]
    de_spearman = CLUSTER_METRIC_IMPLEMENTATIONS["spearman"]
    assert np.allclose(de_pearson(linear), de_spearman(linear), atol=1e-12), (
        "sobre dado linear as duas têm de coincidir — se não coincidem, uma delas não é o que diz"
    )
    assert not np.allclose(de_pearson(nao_linear), de_spearman(nao_linear), atol=1e-3)


def test_o_mapa_das_ligacoes_e_de_identidade_e_cobre_o_literal() -> None:
    """A paridade com o `Literal` é de `tests/unit/test_labels.py`; aqui é a forma do mapa."""
    assert all(nome == valor for nome, valor in CLUSTER_LINKAGE_IMPLEMENTATIONS.items())


# ---------------------------------------------------------------------------
# 5. As duas pontas da população — e são DOIS desfechos distintos
# ---------------------------------------------------------------------------


def test_uma_feature_so_agrupa_as_amostras_e_nao_as_features() -> None:
    """Desfecho um: há agrupamento de amostras, e o das features **não existe**."""
    matriz = dois_grupos(n_features=1)
    resultado = hierarchical_clustering(matriz)
    assert resultado.n_features_used == 1
    assert resultado.sample_linkage.shape == (matriz.n_samples - 1, 4)
    assert resultado.feature_linkage is None, (
        "uma feature não tem par, e `None` é o que distingue «não há o que ligar» de «ligou e deu "
        "vazio» — array vazio tornaria as duas indistinguíveis em contagem"
    )


def test_uma_amostra_so_e_recusada_nomeando_o_eixo_das_amostras() -> None:
    """Desfecho dois, e é outro: sem par não há árvore nenhuma."""
    matriz = matriz_de(np.arange(5.0).reshape(1, 5))
    with pytest.raises(ValueError) as capturado:
        hierarchical_clustering(matriz)
    mensagem = str(capturado.value)
    assert "duas amostras" in mensagem
    assert "1" in mensagem


def test_matriz_toda_constante_recusa_nomeando_as_duas_causas() -> None:
    matriz = matriz_de(np.ones((4, 3)))
    with pytest.raises(ValueError) as capturado:
        hierarchical_clustering(matriz)
    assert "nenhuma das 3 features" in str(capturado.value)


def test_feature_constante_sai_e_e_contada_sem_derrubar_o_resto() -> None:
    matriz = dois_grupos()
    com_constante = np.column_stack([matriz.values, np.full(matriz.n_samples, 7.0)])
    resultado = hierarchical_clustering(
        matriz_de(com_constante, grupos=[s.group for s in matriz.samples])
    )
    assert resultado.n_features_used == matriz.n_features
    assert resultado.n_features_dropped_constant == 1
    assert resultado.n_features_dropped_non_finite == 0
    assert len(resultado.feature_names) == matriz.n_features


def test_distancia_indefinida_e_recusada_no_vocabulario_do_dominio() -> None:
    """Features perfeitamente colineares: depois do z-score toda linha fica constante.

    **A primeira versão deste teste NÃO levantou, e a razão é a ordem das operações.** Ela punha
    uma amostra constante ao longo das features e esperava correlação indefinida — mas o z-score
    é **por coluna** e corre antes da distância, então uma linha constante deixa de o ser.
    A construção que sobrevive ao escalonamento é `a_j * x + b_j`: cada coluna é a mesma rampa
    reescalada, todas ficam idênticas depois do z-score, e aí **toda** linha é constante.

    Sem esta guarda a recusa viria do `scipy`, a meio da ligação e no vocabulário dele.
    """
    x = np.arange(4.0)
    valores = np.outer(x, np.array([1.0, 2.0, 3.0, 4.0])) + np.array([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError) as capturado:
        hierarchical_clustering(
            matriz_de(valores),
            AnalysisSpec(cluster_linkage="average", cluster_metric="correlation"),
        )
    assert "não finito" in str(capturado.value)
    assert "amostras" in str(capturado.value)


def test_a_planta_da_distancia_indefinida_nao_dispara_sobre_matriz_sa() -> None:
    """A outra ponta: a mesma chamada sobre dado não colinear passa. Sem isto a guarda podia
    estar a recusar tudo."""
    resultado = hierarchical_clustering(
        dois_grupos(), AnalysisSpec(cluster_linkage="average", cluster_metric="correlation")
    )
    assert resultado.feature_linkage is not None


# ---------------------------------------------------------------------------
# 6. A recusa tipada do teto de desenho
# ---------------------------------------------------------------------------


def test_acima_do_teto_a_recusa_traz_a_contagem_E_o_teto(  # noqa: N802
) -> None:
    with pytest.raises(FeaturesDemaisError) as capturado:
        exige_abaixo_do_teto(
            TETO_DE_FEATURES_PARA_DESENHAR + 1, TETO_DE_FEATURES_PARA_DESENHAR, de="o mapa de calor"
        )
    mensagem = str(capturado.value)
    assert str(TETO_DE_FEATURES_PARA_DESENHAR + 1) in mensagem
    assert str(TETO_DE_FEATURES_PARA_DESENHAR) in mensagem
    assert "o mapa de calor" in mensagem


@pytest.mark.parametrize("quantas", [0, 1, TETO_DE_FEATURES_PARA_DESENHAR])
def test_no_teto_ou_abaixo_dele_nao_recusa(quantas: int) -> None:
    """A outra direção — e o limite é INCLUSIVO, que é o que a mensagem promete."""
    assert exige_abaixo_do_teto(quantas, TETO_DE_FEATURES_PARA_DESENHAR, de="o mapa de calor") == (
        quantas
    )


def test_a_recusa_de_features_demais_e_apanhavel_pelo_painel() -> None:
    """Desce de `RecusaDoConstrutorError` — é o que `PainelDeFigura._troca_a_figura` apanha.

    Sem a herança, a excepção seria engolida para o `stderr` dentro de um *slot* do Qt e o painel
    ficaria com a figura anterior, mudo — que é o defeito que a base existe para fechar.
    """
    assert issubclass(FeaturesDemaisError, RecusaDoConstrutorError)
    assert not issubclass(FeaturesDemaisError, MatrizVaziaError)
    assert not issubclass(MatrizVaziaError, FeaturesDemaisError)


# ---------------------------------------------------------------------------
# 7. O cancelamento
# ---------------------------------------------------------------------------


def para_na_chamada(n: int) -> CancelCheck:
    """Um verificador que responde «sim» só na `n`-ésima pergunta. Conta a partir de 1."""
    contador = {"n": 0}

    def verifica() -> bool:
        contador["n"] += 1
        return contador["n"] == n

    return verifica


@pytest.mark.parametrize(
    ("chamada", "onde"),
    [
        (1, "agrupamento: antes do escalonamento por feature"),
        (2, "agrupamento: antes da ligação das amostras"),
        (3, "agrupamento: antes da ligação das features"),
    ],
)
def test_cada_ponto_de_parada_dispara_e_diz_onde(chamada: int, onde: str) -> None:
    """Os três pontos, um a um. **`where` não é decoração**: com três checagens, saber qual
    disparou é a diferença entre medir a latência do mecanismo e supô-la."""
    with pytest.raises(ExecutionCancelledError) as capturado:
        hierarchical_clustering(dois_grupos(), cancelled=para_na_chamada(chamada))
    assert capturado.value.where == onde


def test_sem_cancelamento_o_caminho_normal_nao_levanta() -> None:
    """A outra ponta do par causal: a mesma chamada com a planta removida."""
    resultado = hierarchical_clustering(dois_grupos(), cancelled=lambda: False)
    assert resultado.feature_linkage is not None


def test_o_ponto_de_parada_das_features_nao_e_alcancado_quando_elas_nao_sao_pedidas() -> None:
    """Com `cluster_features=False` há duas perguntas e não três — a terceira não existe."""
    resultado = hierarchical_clustering(
        dois_grupos(), cluster_features=False, cancelled=para_na_chamada(3)
    )
    assert resultado.feature_linkage is None


# ---------------------------------------------------------------------------
# 8. Determinismo
# ---------------------------------------------------------------------------


def test_duas_execucoes_da_mesma_matriz_dao_o_mesmo_resultado() -> None:
    matriz = com_feature_dominante()
    a = hierarchical_clustering(matriz)
    b = hierarchical_clustering(matriz)
    assert a.sample_order == b.sample_order
    assert a.feature_order == b.feature_order
    assert np.array_equal(a.sample_linkage, b.sample_linkage)
    assert a.feature_linkage is not None and b.feature_linkage is not None
    assert np.array_equal(a.feature_linkage, b.feature_linkage)
    assert a.column_digest == b.column_digest


# ---------------------------------------------------------------------------
# 9. O corpus real
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matriz_real() -> RefinoMatrix:
    return load_matrix(measurements_path(MODO), sample_list_path(MODO), mode=MODO)


def test_o_corte_real_agrupa_e_declara_a_populacao(matriz_real: RefinoMatrix) -> None:
    """Sobre 300 das 6.035 features do `pos_hilic`, 68 amostras — a população vai declarada.

    **O corte e não o corpus inteiro**, e o motivo é o custo medido: a ligação das 6.035 custa
    ~1,2 s por execução neste clone, e um teste unitário que a pague é um teste que ninguém corre.
    O corpus inteiro é exercitado pelo eixo das amostras, abaixo.
    """
    ordem = np.random.default_rng(SEMENTE_DA_PERMUTACAO).permutation(matriz_real.n_features)
    colunas = np.sort(ordem[:FEATURES_DO_CORTE_REAL])
    valores = np.ascontiguousarray(matriz_real.values[:, colunas])
    valores.flags.writeable = False
    corte = RefinoMatrix(
        values=valores,
        feature_names=tuple(matriz_real.feature_names[int(j)] for j in colunas),
        samples=matriz_real.samples,
        mode=matriz_real.mode,
    )

    resultado = hierarchical_clustering(corte)
    assert resultado.n_features_used <= FEATURES_DO_CORTE_REAL
    assert resultado.sample_linkage.shape == (matriz_real.n_samples - 1, 4)
    assert sorted(resultado.sample_order) == list(range(matriz_real.n_samples))
    assert resultado.feature_linkage is not None
    assert len(resultado.column_digest) == 64


def test_o_eixo_das_amostras_do_corpus_inteiro_e_barato_e_bem_formado(
    matriz_real: RefinoMatrix,
) -> None:
    """6.035 features x 68 amostras, só o eixo das amostras: 2.278 pares, trivial."""
    resultado = hierarchical_clustering(matriz_real, cluster_features=False)
    assert resultado.n_features_used == matriz_real.n_features
    assert resultado.n_features_dropped_constant == 0
    assert resultado.n_features_dropped_non_finite == 0
    assert sorted(resultado.sample_order) == list(range(matriz_real.n_samples))


# ---------------------------------------------------------------------------
# 10. As âncoras dos dois tetos — declaradas, nunca aferidas por relógio
# ---------------------------------------------------------------------------


def test_o_teto_de_desenho_nao_derivou_sem_ninguem_o_remedir() -> None:
    """A âncora da constante. **Não é asserção de tempo de parede.**

    Mudar o número aqui sem correr `scripts/mede_custo_do_agrupamento.py` é caducidade em
    forma pura: número medido que deixa de ser verdade porque o referente mudou por baixo dele.
    """
    assert TETO_DE_FEATURES_PARA_DESENHAR == 60


def test_o_teto_de_computacao_e_declarado_como_NAO_ALCANCADO() -> None:  # noqa: N802
    """`None` e não um número redondo. **Limiar sem origem não serve.**

    Nenhum ponto da série pré-registada — 250 a 6.035 — atravessa o limiar de atenção de 10 s: o
    pior caso medido é 1,2 s no corpus inteiro. Escrever um teto aqui seria inventar a travessia;
    o que existe é dívida declarada, e ela vence quando alguém trouxer matriz maior.
    """
    assert TETO_DE_FEATURES_PARA_COMPUTAR is None


# ---------------------------------------------------------------------------
# 10. K-means
# ---------------------------------------------------------------------------
#
# **Toda asserção continua a ser sobre TOPOLOGIA, e no K-means isso é ainda mais estreito do que na
# árvore:** os rótulos que o método devolve são arbitrários — a mesma partição sai `[0,0,1,1]` ou
# `[1,1,0,0]` conforme o arranque que venceu —, e comparar os vetores crus acusaria diferença onde
# não há nenhuma. `particao_canonica` é o que torna as comparações honestas, e ela já existia para a
# árvore.


def ruido_sem_estrutura(
    *, n_amostras: int = 20, n_features: int = 8, seed: int = 7
) -> RefinoMatrix:
    """Uma matriz SEM agrupamento por construção — e ela existe por causa da regra da planta.

    **A metade *"sementes diferentes divergem"* não é testável sobre dado separado**, e isso é
    propriedade do método e não do teste: com dois grupos afastados, o K-means converge para o mesmo
    mínimo a partir de qualquer arranque, e as cinco sementes dão a **mesma** partição. Plantar ali
    seria uma planta que não separa — o teste passaria com um `random_state` ignorado.

    Sobre ruído não há mínimo global óbvio, os arranques caem em mínimos locais diferentes, e a
    divergência entre sementes é o que se pode de facto observar.
    """
    rng = np.random.default_rng(seed)
    return matriz_de(rng.normal(0.0, 1.0, (n_amostras, n_features)))


def test_o_kmeans_com_k_grupos_por_construcao_recupera_os_k_grupos() -> None:
    """Dois grupos separados por construção caem nas duas partes. **Partição, não rótulo.**"""
    matriz = dois_grupos()
    resultado = kmeans_clustering(matriz, AnalysisSpec(kmeans=True, kmeans_k=2), seed=42)
    esperada = particao_canonica([0 if s.group == "A" else 1 for s in matriz.samples])
    assert particao_canonica(resultado.labels) == esperada
    assert resultado.k == 2
    assert sorted(resultado.cluster_sizes) == [6, 6]


def test_o_kmeans_tambem_recupera_TRES_grupos_e_o_k_nao_e_so_dois() -> None:  # noqa: N802
    """**A planta do `k`**: com `k=2` codificado em silêncio, este teste reprova.

    Sem ele, toda a família estaria verificada sobre o único valor em que um `k` ignorado e um `k`
    obedecido dão o mesmo desfecho: o valor que não exercita, num parâmetro numérico.
    """
    rng = np.random.default_rng(20260918)
    valores = np.vstack(
        [rng.normal(centro, 0.6, (5, 4)) for centro in (0.0, 8.0, 16.0)],
    )
    matriz = matriz_de(valores, grupos=["A"] * 5 + ["B"] * 5 + ["C"] * 5)
    resultado = kmeans_clustering(matriz, AnalysisSpec(kmeans=True, kmeans_k=3), seed=42)
    esperada = particao_canonica([0] * 5 + [1] * 5 + [2] * 5)
    assert particao_canonica(resultado.labels) == esperada
    assert resultado.cluster_sizes == (5, 5, 5) or sorted(resultado.cluster_sizes) == [5, 5, 5]


def test_o_kmeans_da_MESMA_semente_reproduz_e_o_de_OUTRA_diverge() -> None:  # noqa: N802
    """As duas metades, e a segunda corre sobre ruído pela razão de `ruido_sem_estrutura`."""
    matriz = ruido_sem_estrutura()
    analise = AnalysisSpec(kmeans=True, kmeans_k=4)

    primeira = kmeans_clustering(matriz, analise, seed=1)
    segunda = kmeans_clustering(matriz, analise, seed=1)
    assert primeira.labels == segunda.labels
    assert primeira.inertia == segunda.inertia
    assert primeira.seed == segunda.seed

    particoes = {
        particao_canonica(kmeans_clustering(matriz, analise, seed=s).labels) for s in range(1, 6)
    }
    assert len(particoes) > 1, (
        "cinco sementes produziram a MESMA partição sobre ruído: ou a semente não chega ao "
        "`random_state`, ou a planta caiu sobre dado em que as duas coisas não diferem"
    )


def test_d555_o_ajuste_do_kmeans_corre_com_o_openmp_a_UMA_thread(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O OpenMP está limitado a `THREADS_DO_KMEANS` DURANTE o `fit_predict`, e só durante.

    O teste de determinismo acima não apanha o defeito das threads: ele reprova por TAXA — 34 de 100
    num Linux de 12 núcleos sob carga, 0 de 100 neste clone Windows em 200 corridas — e uma corrida
    não estabelece nada. O que se afirma aqui é o mecanismo: lê-se `threadpool_info()` de
    dentro do ajuste, pelo mesmo instrumento que o limite usa.

    **A segunda metade é a âncora contra a vacuidade.** Numa máquina de um núcleo, ou sem runtime
    de OpenMP carregado, a primeira asserção passa sem o `with` — então o teste só vale onde o
    OpenMP, fora do ajuste, tem mais de uma thread, e declara o `skip` quando não tem. Com o `with`
    removido de `kmeans_clustering`, a primeira asserção reprova.
    """
    import threadpoolctl

    def threads_de_openmp() -> list[int]:
        info = threadpoolctl.threadpool_info()
        return [d["num_threads"] for d in info if d["user_api"] == "openmp"]

    fora = threads_de_openmp()
    if not fora or max(fora) <= 1:
        pytest.skip("sem OpenMP com mais de uma thread fora do ajuste: a planta não separaria")

    dentro: list[int] = []
    original = KMeans.fit_predict

    def observa(self: KMeans, x: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        dentro.extend(threads_de_openmp())
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(KMeans, "fit_predict", observa)
    kmeans_clustering(ruido_sem_estrutura(), AnalysisSpec(kmeans=True, kmeans_k=4), seed=1)

    assert dentro, "o duplo não viu o ajuste: a asserção abaixo seria vácua"
    assert set(dentro) == {THREADS_DO_KMEANS}, dentro
    assert threads_de_openmp() == fora, "o limite vazou para fora do ajuste"


def test_a_semente_do_kmeans_e_DERIVADA_e_nunca_a_global() -> None:  # noqa: N802
    """`(seed, unidade)` e não `seed`. **A planta é o SOM correr na mesma execução.**

    Sem a derivação, as duas famílias consumiriam a mesma sequência a partir da mesma
    `PipelineSpec.seed` — e a inicialização de uma passaria a ser previsível a partir da da outra.
    A asserção que separa é que os dois fluxos derivados **diferem entre si** e **nenhum** é a
    semente global.
    """
    matriz = ruido_sem_estrutura()
    resultado = kmeans_clustering(matriz, AnalysisSpec(kmeans=True, kmeans_k=3), seed=42)
    mapa = self_organizing_map(matriz, AnalysisSpec(som=True), seed=42)
    assert resultado.seed != 42
    assert mapa.seed != 42
    assert resultado.seed != mapa.seed
    assert IDENTIFICADOR_DO_KMEANS != IDENTIFICADOR_DO_SOM


def test_o_numero_de_arranques_do_kmeans_e_EXPLICITO_e_nao_o_da_biblioteca() -> None:  # noqa: N802
    """Regra 3 do README lida ao contrário: o padrão da biblioteca mudou entre versões."""
    assert INICIALIZACOES_DO_KMEANS == 10


def test_k_IGUAL_ao_numero_de_amostras_e_legitimo_e_a_inercia_e_zero() -> None:  # noqa: N802
    """A primeira borda: cada amostra é o seu próprio agrupamento."""
    matriz = dois_grupos(por_grupo=3)
    resultado = kmeans_clustering(
        matriz, AnalysisSpec(kmeans=True, kmeans_k=matriz.n_samples), seed=42
    )
    assert sorted(resultado.labels) == list(range(matriz.n_samples))
    assert resultado.inertia == pytest.approx(0.0, abs=1e-9)
    assert resultado.cluster_sizes == (1,) * matriz.n_samples


def test_k_ACIMA_do_numero_de_amostras_e_recusa_TIPADA_com_os_dois_numeros() -> None:  # noqa: N802
    """A segunda metade da borda, e o desfecho é **oposto** ao de `k == n`."""
    matriz = dois_grupos(por_grupo=3)
    with pytest.raises(ParticaoImpossivelError) as capturado:
        kmeans_clustering(matriz, AnalysisSpec(kmeans=True, kmeans_k=matriz.n_samples + 1), seed=42)
    mensagem = str(capturado.value)
    assert str(matriz.n_samples + 1) in mensagem and str(matriz.n_samples) in mensagem


def test_o_kmeans_de_UMA_amostra_nao_existe_e_a_recusa_diz_isso() -> None:  # noqa: N802
    matriz = matriz_de(np.ones((1, 4)))
    with pytest.raises(ValueError, match="ao menos duas amostras"):
        kmeans_clustering(matriz, AnalysisSpec(kmeans=True, kmeans_k=1), seed=42)


def test_o_kmeans_sem_feature_utilizavel_recusa_e_CONTA_as_duas_causas() -> None:  # noqa: N802
    """A recusa do escalonamento partilhado, vista pela porta do K-means."""
    matriz = matriz_de(np.ones((4, 3)))
    with pytest.raises(ValueError, match="utilizável para o K-means"):
        kmeans_clustering(matriz, AnalysisSpec(kmeans=True, kmeans_k=2), seed=42)


@pytest.mark.parametrize(
    ("chamada", "onde"),
    [
        (1, "k-means: antes do escalonamento por feature"),
        (2, "k-means: antes do ajuste"),
    ],
)
def test_cada_ponto_de_parada_do_kmeans_dispara_e_diz_onde(chamada: int, onde: str) -> None:
    with pytest.raises(ExecutionCancelledError) as capturado:
        kmeans_clustering(
            dois_grupos(),
            AnalysisSpec(kmeans=True, kmeans_k=2),
            seed=42,
            cancelled=para_na_chamada(chamada),
        )
    assert capturado.value.where == onde


def test_sem_cancelamento_o_kmeans_nao_levanta() -> None:
    """A outra ponta do par causal: a mesma chamada com a planta removida."""
    resultado = kmeans_clustering(
        dois_grupos(), AnalysisSpec(kmeans=True, kmeans_k=2), seed=42, cancelled=lambda: False
    )
    assert resultado.k == 2


# ---------------------------------------------------------------------------
# 11. O mapa auto-organizável
# ---------------------------------------------------------------------------
#
# **A asserção é sobre a TOPOLOGIA da grade e nunca sobre a unidade em que um grupo caiu.** Qual
# ponta da linha cada grupo ocupa depende do sorteio da inicialização — medido: com doze sementes,
# o grupo `A` ficou em `{0,1}` em cinco delas e em `{2,3}` nas outras sete. O que **não** depende do
# sorteio é a propriedade que faz disto um mapa: cada grupo ocupa um bloco **contíguo** de unidades,
# e os dois blocos não se tocam. Doze de doze sementes.


def blocos_por_grupo(unidades: Sequence[int], grupos: Sequence[str]) -> dict[str, set[int]]:
    """As unidades ocupadas por cada grupo."""
    por_grupo: dict[str, set[int]] = {}
    for unidade, grupo in zip(unidades, grupos, strict=True):
        por_grupo.setdefault(grupo, set()).add(int(unidade))
    return por_grupo


def e_contiguo_na_linha(unidades: set[int]) -> bool:
    """Se o conjunto de unidades é um intervalo sem buraco — a contiguidade numa grade `1 x N`."""
    return max(unidades) - min(unidades) + 1 == len(unidades)


def test_o_som_poe_amostras_parecidas_em_unidades_VIZINHAS() -> None:  # noqa: N802
    """Dois grupos separados ocupam dois blocos contíguos e disjuntos de uma grade `1 x 4`.

    **A grade é uma LINHA de propósito**, e é o que torna a contiguidade uma pergunta de uma
    dimensão em vez de uma de conectividade de conjunto — que exigiria uma segunda implementação
    dentro do teste.
    """
    matriz = dois_grupos(por_grupo=8, n_features=6, separacao=6.0)
    mapa = self_organizing_map(matriz, AnalysisSpec(som=True, som_rows=1, som_cols=4), seed=42)
    blocos = blocos_por_grupo(mapa.units, mapa.groups)
    assert not blocos["A"] & blocos["B"], "os dois grupos partilham uma unidade"
    assert e_contiguo_na_linha(blocos["A"]) and e_contiguo_na_linha(blocos["B"])
    assert min(blocos["A"]) > max(blocos["B"]) or min(blocos["B"]) > max(blocos["A"])


def test_a_planta_da_TOPOLOGIA_separa_de_um_metodo_SEM_grade() -> None:  # noqa: N802
    """A outra metade, e sem ela a de cima seria vácua.

    **Separar os dois grupos não é a propriedade** — qualquer método particional o faz, e o K-means
    também. O que a grade acrescenta é que as partes saem **contíguas**, e a planta é a mesma
    partição com as unidades RENOMEADAS: é exactamente o que um método sem topologia produziria, e
    a contiguidade tem de deixar de valer sobre ela.
    """
    matriz = dois_grupos(por_grupo=8, n_features=6, separacao=6.0)
    mapa = self_organizing_map(matriz, AnalysisSpec(som=True, som_rows=1, som_cols=4), seed=42)
    renomeia = {0: 0, 1: 2, 2: 1, 3: 3}
    plantadas = [renomeia[u] for u in mapa.units]
    assert plantadas != list(mapa.units), "a planta não aterrou: a renomeação foi a identidade"

    blocos = blocos_por_grupo(plantadas, mapa.groups)
    contiguos = all(e_contiguo_na_linha(unidades) for unidades in blocos.values())
    assert not contiguos, (
        "a partição RENOMEADA continua contígua: a planta não separa, e a asserção de cima "
        "passaria sobre um método que ignorasse a grade"
    )


def test_o_som_da_MESMA_semente_reproduz_e_o_de_OUTRA_diverge() -> None:  # noqa: N802
    """Regra 3 do README nas duas direções.

    **A divergência é medida sobre a mesma matriz separada.**
    """
    matriz = dois_grupos(por_grupo=8, n_features=6, separacao=6.0)
    analise = AnalysisSpec(som=True)

    primeira = self_organizing_map(matriz, analise, seed=3)
    segunda = self_organizing_map(matriz, analise, seed=3)
    assert primeira.units == segunda.units
    assert primeira.quantization_error == segunda.quantization_error

    mapas = {self_organizing_map(matriz, analise, seed=s).units for s in range(5)}
    assert len(mapas) > 1, (
        "cinco sementes produziram o MESMO mapa: ou a semente não chega ao gerador, ou a planta "
        "caiu sobre dado em que as duas coisas não diferem"
    )


def test_a_grade_de_UMA_unidade_poe_tudo_nela_e_nao_levanta() -> None:  # noqa: N802
    """A primeira borda: `1 x 1`. **É o caso em que a forma exponencial clássica inverteria o
    raio.**"""
    matriz = dois_grupos(por_grupo=4)
    mapa = self_organizing_map(matriz, AnalysisSpec(som=True, som_rows=1, som_cols=1), seed=42)
    assert set(mapa.units) == {0}
    assert mapa.n_units == 1
    assert mapa.n_units_empty == 0
    assert mapa.quantization_error > 0.0


def test_a_grade_MAIOR_que_a_coorte_deixa_unidades_vazias_e_isso_e_ACHADO() -> None:  # noqa: N802
    """A segunda borda, e o desfecho é **oposto** ao da grade de uma unidade: sobram unidades.

    **Não é recusa**, e a escolha está declarada em `self_organizing_map`: recusar proibiria
    exactamente a configuração em que as unidades vazias são informação (R12b).
    """
    matriz = dois_grupos(por_grupo=3)
    mapa = self_organizing_map(matriz, AnalysisSpec(som=True, som_rows=4, som_cols=4), seed=42)
    assert mapa.n_units == 16
    assert mapa.n_units_empty >= 16 - matriz.n_samples
    assert sum(mapa.unit_sizes) == matriz.n_samples


def test_o_som_de_ZERO_epocas_recusa_em_vez_de_devolver_a_INICIALIZACAO() -> None:  # noqa: N802
    matriz = dois_grupos(por_grupo=3)
    with pytest.raises(ParticaoImpossivelError, match="`epochs` é 0"):
        self_organizing_map(matriz, AnalysisSpec(som=True), seed=42, epochs=0)


def test_o_som_de_UMA_amostra_nao_existe_e_a_recusa_diz_isso() -> None:  # noqa: N802
    matriz = matriz_de(np.ones((1, 4)))
    with pytest.raises(ValueError, match="ao menos duas amostras"):
        self_organizing_map(matriz, AnalysisSpec(som=True), seed=42)


def test_o_cancelamento_do_som_dispara_DENTRO_do_laco_das_epocas():  # noqa: N802
    """No SOM o ponto de parada é **por época**, e a latência é uma passagem.

    **A asserção é sobre QUAL época disparou**, e não só sobre ter disparado: um ponto de parada
    posto antes do laço satisfaria «levantou» e deixaria «Cancelar» sem efeito durante o cálculo
    inteiro — que é a planta que não separa.
    """
    matriz = dois_grupos(por_grupo=3)
    with pytest.raises(ExecutionCancelledError) as capturado:
        self_organizing_map(
            matriz, AnalysisSpec(som=True), seed=42, epochs=8, cancelled=para_na_chamada(4)
        )
    assert capturado.value.where == "som: época 3 de 8"


def test_sem_cancelamento_o_som_nao_levanta() -> None:
    """A outra ponta do par causal: a mesma chamada com a planta removida."""
    mapa = self_organizing_map(
        dois_grupos(por_grupo=3),
        AnalysisSpec(som=True),
        seed=42,
        epochs=8,
        cancelled=lambda: False,
    )
    assert mapa.epochs == 8


def test_as_epocas_padrao_sao_a_CONSTANTE_medida_e_nao_um_literal() -> None:  # noqa: N802
    """Âncora contra a constante derivar sem ninguém a remedir, que é caducidade.

    O número **não** é afirmado aqui: o que se afirma é que o padrão da função **é** a constante, e
    que ela está na faixa que a medição percorreu. A tabela e a população vivem na docstring dela.
    """
    mapa = self_organizing_map(dois_grupos(por_grupo=3), AnalysisSpec(som=True), seed=42)
    assert mapa.epochs == EPOCAS_DO_SOM
    assert 12 <= EPOCAS_DO_SOM <= 400


def test_as_contagens_derivadas_do_som_e_do_kmeans_sao_PROPRIEDADE_e_nao_campo() -> None:  # noqa: N802
    """Contagem igual não é conjunto igual, e contagem em dois sítios diverge."""
    matriz = dois_grupos(por_grupo=4)
    particao = kmeans_clustering(matriz, AnalysisSpec(kmeans=True, kmeans_k=2), seed=42)
    assert sum(particao.cluster_sizes) == matriz.n_samples

    mapa = self_organizing_map(matriz, AnalysisSpec(som=True, som_rows=2, som_cols=3), seed=42)
    assert sum(mapa.unit_sizes) == matriz.n_samples
    assert mapa.n_units_empty == sum(1 for t in mapa.unit_sizes if t == 0)


def test_as_DUAS_familias_novas_veem_a_matriz_INTEIRA_e_nao_so_o_estudo() -> None:  # noqa: N802
    """A população é declarada no cabeçalho do módulo, e esta é a âncora dela.

    **É o contraste com `statistics.correlacao`, que recorta pelo `sample_mask`** — e a diferença
    tem consequência visível: o dendrograma desenha 68 folhas, e uma partição sobre 61 na aba
    ao lado seriam duas figuras a descrever populações diferentes com o mesmo nome.
    """
    matriz = dois_grupos(por_grupo=4)
    qcs = [
        SampleMetadata(name="qc1", injection_order=99, sample_type="qc", group="QC"),
    ]
    com_qc = RefinoMatrix(
        values=matriz.values,
        feature_names=matriz.feature_names,
        samples=(*matriz.samples[:-1], *qcs),
        mode=matriz.mode,
    )
    assert int(com_qc.sample_mask.sum()) == com_qc.n_samples - 1

    particao = kmeans_clustering(com_qc, AnalysisSpec(kmeans=True, kmeans_k=2), seed=42)
    mapa = self_organizing_map(com_qc, AnalysisSpec(som=True), seed=42)
    assert len(particao.labels) == com_qc.n_samples
    assert len(mapa.units) == com_qc.n_samples
    assert "QC" in particao.groups and "QC" in mapa.groups

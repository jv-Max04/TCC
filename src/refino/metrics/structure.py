"""Métricas estruturais de PCA.

O RSD mede repetibilidade feature por feature. Estas medem se a estrutura global
da matriz ainda é organizada pela ordem de injeção, que é o que a deriva
instrumental produz e o que a correção de sinal deve desfazer.

`pretransform` é obrigatório e sem valor padrão, porque depende de onde a medição
acontece:

- pós-correção, `"log10p1"`: a matriz é de intensidade bruta e precisa de
  compressão de escala;
- pós-pipeline, `"none"`: a transformação da especificação já foi aplicada.

Qualquer padrão estaria errado em metade das chamadas, sem levantar: `log10`
sobre matriz já transformada é transformação dupla, e sobre matriz normalizada
por soma os valores ficam muito abaixo de 1, onde `log10(1+x) ~ x/ln 10` é quase
a identidade.
"""

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from refino.domain.labels import PreTransform, validate_label
from refino.domain.matrix import RefinoMatrix

__all__ = [
    "PRETRANSFORM_IMPLEMENTATIONS",
    "EstruturaSemPopulacaoError",
    "StructuralMetrics",
    "StructuralPCA",
    "column_usability",
    "pc1_injection_correlation",
    "qc_dispersion_ratio",
    "qc_pc1_trajectory",
    "retained_column_digest",
    "structural_metrics",
    "structural_pca",
]

# Mínimo de linhas para que exista dispersão em torno de um centroide.
MIN_ROWS_FOR_DISPERSION = 2


class EstruturaSemPopulacaoError(ValueError):
    """A matriz não tem população que sustente a medida estrutural.

    É fato sobre o dado, separado do `ValueError` comum para que a fronteira o
    distinga de argumento errado de quem chamou: a rota traduz esta recusa em
    `422`, e o outro é defeito de programação e tem de subir. Uma matriz com um
    QC só, que a validação aceita com aviso, cai aqui.

    Herda de `ValueError` para que quem não precisa da distinção continue a
    apanhá-la.
    """


# Réplicas e semente do indicador de estabilidade. A perturbação é estocástica,
# então a semente é explícita e configurável.
STABILITY_REPLICATES = 32
STABILITY_SEED = 0


def _log10p1(values: np.ndarray) -> np.ndarray:
    """`log10(x + 1)`.

    Por causa do `+1` a pré-transformação não é equivariante a escala:
    `log10(cx + 1)` não é `log10(x + 1) + log10(c)`. Para intensidades de LC-MS
    entre 1e2 e 1e9 o `+1` é desprezível; para matriz já normalizada por soma, não.
    """
    return np.asarray(np.log10(values + 1.0), dtype=np.float64)


def _identity(values: np.ndarray) -> np.ndarray:
    """`"none"`: a especificação a jusante já transformou."""
    return np.asarray(values, dtype=np.float64)


# O teste de paridade compara as chaves deste registro com
# `get_args(PreTransform)`, nas duas direções: chave sem valor no `Literal`, ou o
# inverso, reprova.
PRETRANSFORM_IMPLEMENTATIONS: dict[PreTransform, Callable[[np.ndarray], np.ndarray]] = {
    "log10p1": _log10p1,
    "none": _identity,
}


@dataclass(frozen=True)
class StructuralMetrics:
    """As três métricas estruturais, mais o rótulo de como foram medidas.

    O rótulo vai com o número porque `qc_dispersion_ratio` medido com `"log10p1"`
    e com `"none"` são grandezas diferentes com o mesmo nome.
    """

    qc_dispersion_ratio: float
    pc1_injection_correlation_abs: float
    qc_pc1_trajectory: tuple[float, ...]
    # Fração da variância total explicada por cada componente, na ordem deles.
    # Comprimento igual a `n_components`.
    explained_variance_ratio: tuple[float, ...]
    pretransform: PreTransform
    n_components: int
    # Quantas colunas o passo 3 descartou, por motivo: variância zero é feature
    # que não varia, não finito é feature que o método de correção estragou.
    n_features_dropped_zero_variance: int
    n_features_dropped_non_finite: int
    n_features_used: int
    # Digest do conjunto de colunas retidas. Contagem igual não é conjunto igual:
    # duas especificações podem descartar o mesmo número de features e descartar
    # features diferentes. Com o digest se compara se duas métricas foram
    # calculadas sobre o mesmo espaço.
    retained_column_digest: str
    # Amplitude das duas métricas escalares sob perturbação de `eps x amplitude`.
    # As métricas não são contínuas, e um limiar sobre valor instável decidiria
    # por ruído de último bit.
    qc_dispersion_ratio_spread: float
    pc1_injection_correlation_spread: float
    stability_replicates: int
    stability_seed: int

    def is_assessable(self, *, max_spread: float) -> bool:
        """`False` quando as métricas estruturais não sustentam classificação.

        Métrica indefinida ou instável é ausência de medida. Sem este teste,
        `nan < 0,30` é `False` por semântica de ponto flutuante, e a matriz seria
        classificada por acidente aritmético.

        `max_spread` não tem padrão: quem define o que é instável demais é a
        política de classificação, que tem os dados para calibrar.
        """
        if max_spread < 0:
            raise ValueError(f"max_spread não pode ser negativo; recebi {max_spread}")
        valores = (self.qc_dispersion_ratio, self.pc1_injection_correlation_abs)
        amplitudes = (self.qc_dispersion_ratio_spread, self.pc1_injection_correlation_spread)
        if any(math.isnan(v) for v in valores):
            return False
        return all(math.isfinite(a) and a <= max_spread for a in amplitudes)


def _pca_scores(
    matrix: RefinoMatrix,
    *,
    pretransform: PreTransform,
    n_components: int = 2,
) -> np.ndarray:
    """Scores `(n_samples, n_components)`.

    Existe como função separada para o teste de sinal: o sinal de um componente
    vem do SVD e nenhuma entrada do chamador o controla, então a invariância a
    ele só é verificável por aqui.

    São cinco passos, e alterá-los muda os números. Sem o `log10`, a razão de
    dispersão do `neg_hilic` com `batch_loess` dá 0,114 em vez de 0,682; sem o
    autoescalonamento, 0,952.

    1. Todas as injeções, QCs e amostras juntos. Um espaço ajustado só nas
       amostras não tem os QCs para medir dispersão.
    2. A pré-transformação declarada.
    3. Descartar colunas inutilizáveis, sobre as mesmas linhas do passo 1: as
       constantes e as que têm valor não finito.
    4. Autoescalonamento por coluna. O denominador do desvio é indiferente:
       `n` e `n - 1` escalonam todas as colunas pelo mesmo fator, e nem razão de
       distâncias nem correlação de postos mudam com isso.
    5. SVD; `scores = U[:, :k] * S[:k]`.
    """
    return _pca_scores_with_drops(matrix, pretransform=pretransform, n_components=n_components)[0]


def column_usability(values: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Máscara das colunas utilizáveis no passo 3, e a contagem de cada motivo.

    Os dois motivos ficam separados. Constância é feature que não varia entre
    injeções, informação legítima sobre o dado. Não finito é feature que o método
    de correção produziu com `Inf` ou `NaN`; nas matrizes golden do `batch_loess`
    são de 2,7% a 15,2% das features.

    Constância se testa por `ptp == 0`, e não por `sd == 0`. O desvio padrão de
    uma coluna constante não é exatamente zero em ponto flutuante: a média de doze
    cópias de `log10(6)` não fecha bit a bit, e o desvio mede 1,11e-16. Com
    `sd > 0.0` a coluna passaria como utilizável, o passo 4 a dividiria por
    1,11e-16 e ela dominaria o SVD.

    Nos quatro arquivos de referência os dois critérios concordam, porque as
    colunas constantes deles são constantes em zero, onde `log10(0 + 1)` é
    exatamente `0.0`. Coluna constante num valor não nulo (feature saturada, matriz
    normalizada) é o caso em que `sd > 0.0` falha.

    É pública para que os módulos de `statistics` usem o mesmo critério em vez de
    uma cópia.
    """
    finite_column = np.isfinite(values).all(axis=0)
    constant = np.zeros(values.shape[1], dtype=bool)
    if finite_column.any():
        constant[finite_column] = np.ptp(values[:, finite_column], axis=0) == 0.0
    usable = finite_column & ~constant
    return usable, int((finite_column & constant).sum()), int((~finite_column).sum())


def retained_column_digest(feature_names: tuple[str, ...], usable: np.ndarray) -> str:
    """`sha256` dos nomes das colunas retidas, na ordem das colunas da matriz.

    É calculado sobre os nomes, e não sobre a máscara: duas matrizes com features
    diferentes e a mesma máscara não podem dar o mesmo digest. A ordem entra
    porque faz parte do espaço em que o SVD roda.

    Guarda-se o digest, e não a máscara, porque ele basta para comparar espaços
    entre muitos relatórios. Para saber quais colunas saíram, recalcula-se.
    """
    retidos = [name for name, keep in zip(feature_names, usable.tolist(), strict=True) if keep]
    payload = "\n".join(retidos).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class StructuralPCA:
    """O que os cinco passos produzem numa execução do SVD: scores, cargas e variância explicada.

    Não é a PCA de `refino.statistics.pca.pca`, que serve à matriz já processada
    e não pré-transforma nem autoescalona. Esta aplica a pré-transformação
    declarada e autoescalona por coluna, e é a que alimenta `diagnose`,
    `qc_dispersion_ratio`, `pc1_injection_correlation` e a trajetória que a janela
    desenha. As duas dão decomposições diferentes da mesma matriz, então uma
    figura tirada da outra discordaria dos números da aba de qualidade.

    `n_features_used` conta as colunas que sobraram do passo 3, e o digest vai ao
    lado porque contagem igual não é conjunto igual.

    O sinal de cada componente é arbitrário, e esta classe não o corrige: orientar
    aqui mudaria a `qc_pc1_trajectory` que `diagnose` e a janela já usam. Quem
    consome declara a arbitrariedade, como `plots/deriva.py` faz no rótulo do eixo.
    """

    scores: np.ndarray
    """`(n_samples, n_components)`. Linhas na ordem de injeção, que `RefinoMatrix` garante."""

    loadings: np.ndarray
    """`(n_features_used, n_components)`, alinhado a `feature_names_used`, e não a
    `matrix.feature_names`, porque o passo 3 descarta colunas."""

    feature_names_used: tuple[str, ...]
    """Os nomes das colunas retidas, na ordem das linhas de `loadings`."""

    explained_variance_ratio: tuple[float, ...]
    singular_values: tuple[float, ...]
    pretransform: PreTransform
    n_components: int
    n_features_dropped_zero_variance: int
    n_features_dropped_non_finite: int
    n_features_used: int
    retained_column_digest: str


def structural_pca(
    matrix: RefinoMatrix,
    *,
    pretransform: PreTransform,
    n_components: int = 2,
) -> StructuralPCA:
    """Os cinco passos de `_pca_scores`, com scores, cargas e variância explicada numa execução.

    É o caminho que `structural_metrics` corre: `_pca_scores_with_drops` delega
    aqui, então as figuras de PCA e as métricas saem do mesmo SVD.

    `n_components` não muda os componentes. O SVD é o mesmo, e pedir `k` fatia os
    `k` primeiros: a variância explicada de PC1 é a mesma para `k = 2` e `k = 10`.
    """
    validate_label("PreTransform", pretransform)
    if n_components < 1:
        raise ValueError(f"n_components deve ser >= 1; recebi {n_components}")

    with np.errstate(invalid="ignore"):
        values = PRETRANSFORM_IMPLEMENTATIONS[pretransform](matrix.values)

    usable, dropped_zero, dropped_non_finite = column_usability(values)
    kept = values[:, usable]
    if kept.shape[1] == 0:
        raise EstruturaSemPopulacaoError(
            "nenhuma coluna utilizável depois do passo 3: "
            f"{dropped_zero} com variância zero e {dropped_non_finite} com desvio não "
            "finito, de "
            f"{matrix.n_features}. Não há estrutura a medir"
        )

    scaled = (kept - kept.mean(axis=0)) / kept.std(axis=0)

    # `full_matrices=False` porque só os primeiros `k` componentes interessam e a
    # matriz é larga: dezenas de linhas por milhares de colunas.
    u, s, vt = np.linalg.svd(scaled, full_matrices=False)
    if s.size < n_components:
        raise EstruturaSemPopulacaoError(
            f"pedi {n_components} componentes e a matriz só admite {s.size}: "
            f"{matrix.n_samples} injeções e {kept.shape[1]} features úteis"
        )
    scores: np.ndarray = u[:, :n_components] * s[:n_components]
    n_used = int(usable.sum())
    # O mesmo denominador de `_explained_variance_ratio`: o passo 4 autoescalona
    # com `ddof=0`, então cada coluna retida tem variância 1 e a soma dos quadrados
    # da matriz escalada é `n_samples * n_features_used`.
    total = float(matrix.n_samples * n_used)
    return StructuralPCA(
        scores=scores,
        loadings=vt[:n_components].T,
        feature_names_used=tuple(
            nome for nome, keep in zip(matrix.feature_names, usable.tolist(), strict=True) if keep
        ),
        explained_variance_ratio=(
            tuple(float(s[k] ** 2 / total) for k in range(n_components))
            if total
            else tuple(0.0 for _ in range(n_components))
        ),
        singular_values=tuple(float(v) for v in s[:n_components]),
        pretransform=pretransform,
        n_components=n_components,
        n_features_dropped_zero_variance=dropped_zero,
        n_features_dropped_non_finite=dropped_non_finite,
        n_features_used=n_used,
        retained_column_digest=retained_column_digest(matrix.feature_names, usable),
    )


def _pca_scores_with_drops(
    matrix: RefinoMatrix,
    *,
    pretransform: PreTransform,
    n_components: int = 2,
) -> tuple[np.ndarray, int, int, str]:
    """Os scores, as duas contagens do passo 3 e o digest. Ver `_pca_scores`.

    Delega a `structural_pca` para que haja uma implementação só dos cinco passos.
    """
    completa = structural_pca(matrix, pretransform=pretransform, n_components=n_components)
    return (
        completa.scores,
        completa.n_features_dropped_zero_variance,
        completa.n_features_dropped_non_finite,
        completa.retained_column_digest,
    )


def _pca_ou_recalcula(
    matrix: RefinoMatrix,
    *,
    pretransform: PreTransform,
    n_components: int,
    pca: StructuralPCA | None,
) -> StructuralPCA:
    """A PCA que o chamador já tem, conferida, ou uma execução nova.

    `pretransform` e `n_components` são o que distingue duas PCAs do mesmo tipo:
    o primeiro decide se a matriz foi comprimida em escala, e o segundo, sobre
    quantas dimensões `qc_dispersion_ratio` mede distância ao centroide. Aceitar
    sem conferir devolveria números plausíveis com o rótulo errado.
    """
    if pca is None:
        return structural_pca(matrix, pretransform=pretransform, n_components=n_components)
    if pca.pretransform != pretransform or pca.n_components != n_components:
        raise ValueError(
            f"a PCA recebida foi calculada com pretransform={pca.pretransform!r} e "
            f"n_components={pca.n_components}, e esta chamada declara {pretransform!r} e "
            f"{n_components}. São medidas diferentes com o mesmo tipo: `qc_dispersion_ratio` "
            "depende de todos os componentes, e a pré-transformação decide a escala"
        )
    return pca


def _explained_variance_ratio(
    scores: np.ndarray, matrix: RefinoMatrix, n_features_used: int
) -> tuple[float, ...]:
    """Fração da variância total que cada componente explica.

    Não precisa dos valores singulares. O passo 4 autoescalona com `std` de
    `ddof=0`, então cada coluna retida tem variância 1 e a soma dos quadrados da
    matriz escalada é `n_samples * n_features_used`. E `scores[:, k] = u[:, k] * s_k`
    com `u` ortonormal, então `Σ scores[:, k]²` é `s_k²`.

    Devolve tupla de comprimento `n_components`, e não dois escalares, porque o
    número de componentes é parâmetro.
    """
    total = float(matrix.n_samples * n_features_used)
    if total == 0.0:
        return tuple(0.0 for _ in range(scores.shape[1]))
    return tuple(float((scores[:, k] ** 2).sum() / total) for k in range(scores.shape[1]))


def _mean_distance_to_centroid(points: np.ndarray) -> float:
    """Distância euclidiana média das linhas ao centroide delas."""
    centroid = points.mean(axis=0)
    return float(np.mean(np.linalg.norm(points - centroid, axis=1)))


def _dispersion_from_scores(scores: np.ndarray, matrix: RefinoMatrix) -> float:
    qc = scores[matrix.qc_mask]
    samples = scores[matrix.sample_mask]
    for name, block in (("qc_mask", qc), ("sample_mask", samples)):
        if block.shape[0] < MIN_ROWS_FOR_DISPERSION:
            raise EstruturaSemPopulacaoError(
                f"{name} tem {block.shape[0]} linha(s); dispersão em torno de um "
                "centroide exige ao menos duas"
            )
    denominator = _mean_distance_to_centroid(samples)
    if denominator == 0.0:
        # Amostras todas no mesmo ponto: a razão não existe. `nan`, e não `inf`,
        # porque o numerador também pode ser zero.
        return float("nan")
    return _mean_distance_to_centroid(qc) / denominator


def _abs_spearman_with_order(pc1: np.ndarray, orders: np.ndarray) -> float:
    if pc1.size < MIN_ROWS_FOR_DISPERSION:
        raise EstruturaSemPopulacaoError(
            f"correlação exige ao menos duas injeções; recebi {pc1.size}"
        )
    rho = float(spearmanr(pc1, orders).statistic)
    # `nan` quando PC1 é constante: sem variação não há posto. Zero afirmaria
    # ausência de correlação medida.
    return float("nan") if math.isnan(rho) else abs(rho)


def _spread(valores: list[float]) -> float:
    """Amplitude dos valores replicados, com três desfechos distinguíveis.

    - Todos indefinidos: `nan`. Não há medida cuja estabilidade avaliar, e o
      valor em si também é `nan`.
    - Alguns definidos e outros não: `inf`. A perturbação atravessou a fronteira
      da definibilidade, e qualquer tolerância finita reprova.
    - Todos definidos: `max - min`.

    Amplitude, e não desvio padrão: a pergunta é quanto o valor pode se mover sob
    ruído que não deveria importar. O desvio padrão de uma distribuição bimodal,
    que é o que uma descontinuidade produz, subestima o salto.
    """
    array = np.asarray(valores, dtype=np.float64)
    definidos = np.isfinite(array)
    if not definidos.any():
        return float("nan")
    if not definidos.all():
        return float("inf")
    return float(array.max() - array.min())


def _stability_spreads(
    scores: np.ndarray,
    matrix: RefinoMatrix,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Amplitude das duas métricas sob perturbação de `eps x amplitude`.

    As duas métricas não são funções contínuas da entrada.
    `pc1_injection_correlation` salta porque postos amplificam ruído entre valores
    empatados, e `qc_dispersion_ratio` salta em torno do denominador nulo. Um
    limiar sobre quantidade descontínua decide por último bit justamente onde ela
    é mal-condicionada.

    Perturba os scores, e não a matriz: PC1 tem uma entrada por injeção, então
    replicar é barato, e perturbar a matriz mediria também a estabilidade do SVD,
    que é outra pergunta.

    A escala do ruído é o `eps` da precisão dupla vezes a amplitude dos scores, o
    ruído que a própria aritmética já introduz. Se a resposta muda com ele, não
    era sustentada pelo dado.
    """
    rng = np.random.default_rng(seed)
    amplitude = float(np.max(np.abs(scores))) if scores.size else 0.0
    escala = float(np.finfo(np.float64).eps) * amplitude
    if escala == 0.0:
        # Scores todos nulos: não há ruído relativo a introduzir, e a métrica é
        # estável.
        return 0.0, 0.0

    dispersoes: list[float] = []
    correlacoes: list[float] = []
    for _ in range(replicates):
        perturbado = scores + rng.standard_normal(scores.shape) * escala
        dispersoes.append(_dispersion_from_scores(perturbado, matrix))
        correlacoes.append(_abs_spearman_with_order(perturbado[:, 0], matrix.injection_orders))
    return _spread(dispersoes), _spread(correlacoes)


def qc_dispersion_ratio(
    matrix: RefinoMatrix,
    *,
    pretransform: PreTransform,
    n_components: int = 2,
) -> float:
    """Dispersão dos QCs relativa à das amostras, nos `k` primeiros componentes.

    Menor é melhor: os QCs são a mesma mistura, então idealmente formam um ponto
    apertado dentro da nuvem das amostras biológicas, que variam de verdade.

    O denominador usa `sample_mask`, e não o complemento de `qc_mask`: amostras
    `validate` ficam fora das duas, como ficam fora da correção. Incluí-las
    tornaria a medida incomparável entre matrizes com e sem `validate`.

    É o análogo multivariado do D-ratio de Broadhurst et al. (2018), definido lá
    como `s_qc / s_amostra * 100%` por feature, com critério de aceitação `< 50%`.
    O mesmo artigo recomenda projetar nos dois primeiros componentes e comparar
    a dispersão dos QCs com a total. O limiar de 50% não vale para este número:
    o D-ratio é por feature, no espaço original, e este é sobre os scores de `k`
    componentes.
    """
    scores = _pca_scores(matrix, pretransform=pretransform, n_components=n_components)
    return _dispersion_from_scores(scores, matrix)


def pc1_injection_correlation(matrix: RefinoMatrix, *, pretransform: PreTransform) -> float:
    """Spearman entre PC1 e a ordem de injeção, em valor absoluto.

    O sinal de um componente principal é arbitrário (o SVD pode devolver `-PC1`),
    então o sinal da correlação não carrega informação.

    Valor alto significa que a maior fonte de variação da matriz ainda é quando a
    amostra foi injetada, e não o que havia nela. É o que a correção de sinal
    existe para desfazer.

    Correlação de postos, e não de Pearson, porque deriva instrumental não é
    linear no tempo: no `neg_hilic` ela aparece como degrau, que é monotônico sem
    ser linear.

    Broadhurst et al. (2018) e QComics (2024) recomendam examinar a resposta
    contra a ordem de injeção, e nenhum dos dois dá limiar numérico. O corte de
    0,30 em `pc1_injection_correlation_max` é deste trabalho.
    """
    scores = _pca_scores(matrix, pretransform=pretransform, n_components=1)
    return _abs_spearman_with_order(scores[:, 0], matrix.injection_orders)


def qc_pc1_trajectory(matrix: RefinoMatrix, *, pretransform: PreTransform) -> np.ndarray:
    """Scores de PC1 dos QCs, em ordem de injeção, em unidades de score.

    Não subtrai o primeiro elemento. Mostrar 'quanto andou desde o início' é
    escolha de apresentação, e quem apresenta subtrai.

    As linhas já vêm em ordem de injeção, que `RefinoMatrix` garante na
    construção, então filtrar por `qc_mask` preserva a ordem temporal.
    """
    scores = _pca_scores(matrix, pretransform=pretransform, n_components=1)
    trajectory: np.ndarray = scores[matrix.qc_mask, 0]
    return trajectory


def structural_metrics(
    matrix: RefinoMatrix,
    *,
    pretransform: PreTransform,
    n_components: int = 2,
    stability_replicates: int = STABILITY_REPLICATES,
    stability_seed: int = STABILITY_SEED,
    pca: StructuralPCA | None = None,
) -> StructuralMetrics:
    """As três métricas com o rótulo, calculando os scores uma vez.

    As três funções individuais servem ao uso isolado, e cada uma recalcula o SVD.
    Esta é o caminho do multiverso, que a executa milhares de vezes. PC1 é o mesmo
    vetor para qualquer `k >= 1`, então a trajetória e a correlação tiradas da
    primeira coluna dos `k` scores são as que as funções individuais dariam.

    É o único caminho que reporta quantas features o passo 3 descartou. Nas
    matrizes golden do `batch_loess` o descarte chega a 20,3% das colunas, e a
    razão de dispersão tem de vir com a contagem sobre a qual foi medida.

    `pca` aceita uma execução já feita, para que as cargas e a variância
    explicada das figuras saiam do mesmo objeto que `qc_pc1_trajectory`, sem um
    segundo SVD. O par de rótulos é conferido: um `StructuralPCA` de outra
    pré-transformação ou de outro número de componentes mede outra coisa, e a
    divergência levanta.
    """
    completa = _pca_ou_recalcula(
        matrix, pretransform=pretransform, n_components=n_components, pca=pca
    )
    scores = completa.scores
    dropped_zero = completa.n_features_dropped_zero_variance
    dropped_non_finite = completa.n_features_dropped_non_finite
    digest = completa.retained_column_digest
    dispersao_spread, correlacao_spread = _stability_spreads(
        scores, matrix, replicates=stability_replicates, seed=stability_seed
    )
    n_usadas = matrix.n_features - dropped_zero - dropped_non_finite
    return StructuralMetrics(
        qc_dispersion_ratio=_dispersion_from_scores(scores, matrix),
        explained_variance_ratio=_explained_variance_ratio(scores, matrix, n_usadas),
        pc1_injection_correlation_abs=_abs_spearman_with_order(
            scores[:, 0], matrix.injection_orders
        ),
        qc_pc1_trajectory=tuple(float(v) for v in scores[matrix.qc_mask, 0]),
        pretransform=pretransform,
        n_components=n_components,
        n_features_dropped_zero_variance=dropped_zero,
        n_features_dropped_non_finite=dropped_non_finite,
        n_features_used=n_usadas,
        retained_column_digest=digest,
        qc_dispersion_ratio_spread=dispersao_spread,
        pc1_injection_correlation_spread=correlacao_spread,
        stability_replicates=stability_replicates,
        stability_seed=stability_seed,
    )

"""PLS-DA binário com validação cruzada e teste de permutação.

Nenhum caminho devolve scores sem `q2` e `permutation_pvalue`. `PLSDAResult` é congelado e
todos os campos são obrigatórios, e a única função pública do módulo é `plsda`, que os preenche
todos. PLS-DA sem validação mostra separação convincente mesmo quando não há o que separar.

Os escores de validação cruzada saem do mesmo laço do `Q²`. Os de calibração tendem a separar os
grupos, porque o ajuste escolhe a direção que os separa; com muito mais features que amostras,
como no LC-MS, separam até rótulos sem informação. Por isso o resultado traz também cada amostra
projetada pelo modelo da dobra que não a viu, girada por Procrustes ortogonal para o referencial
do modelo de calibração: é a separação que o modelo prediz. São um campo do resultado, e não uma
segunda função, para não existir escore de PLS-DA longe do `q2`.

Cada permutação recebe um gerador derivado de `(seed, índice da permutação)`. Uma semente global
consumida em ordem tornaria a saída dependente da ordem de execução e do paralelismo.

O p-valor de permutação é `(k + 1) / (n + 1)`. Com `k/n` puro, um resultado que nenhuma
permutação supera daria `p = 0`. O `+1` conta o observado entre os sorteios, que é o que ele é
sob a nula, e o menor p-valor possível passa a ser `1/(n+1)`, a resolução do teste.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold

from refino.domain.cancellation import CancelCheck, raise_if_cancelled
from refino.domain.matrix import RefinoMatrix
from refino.statistics.orientation import orient_canonically

__all__ = ["PLSDAResult", "plsda"]

MIN_AMOSTRAS_POR_GRUPO = 2


@dataclass(frozen=True)
class PLSDAResult:
    """Resultado do PLS-DA binário. Todos os campos são obrigatórios.

    Dataclass congelado sem valor padrão não admite construção parcial, então não há como
    produzir `scores` sem `q2` e `permutation_pvalue`.

    `scores` são os de calibração: os do modelo ajustado sobre todas as amostras do contraste.
    `cv_scores` têm a mesma forma e as mesmas linhas, e cada linha vem do modelo de uma dobra
    que não a viu, girado para o referencial de `scores`. É a separação que o modelo prediz, e
    não a que ele ajusta.
    """

    scores: np.ndarray
    cv_scores: np.ndarray
    loadings: np.ndarray
    vip: np.ndarray
    r2y: float
    q2: float
    permutation_pvalue: float
    n_components: int
    n_permutations: int
    seed: int


def plsda(
    matrix: RefinoMatrix,
    groups: tuple[str, str],
    n_components: int = 2,
    cv_folds: int = 7,
    n_permutations: int = 200,
    seed: int = 42,
    cancelled: CancelCheck | None = None,
) -> PLSDAResult:
    """PLS-DA entre dois grupos, com `Q²` por validação cruzada e p-valor de permutação.

    `Q² = 1 - PRESS/TSS` com `cv_folds` dobras estratificadas por grupo. Com uma dobra que
    contenha um só grupo, o modelo prediz constante e o `PRESS` mede outra coisa; por isso cada
    grupo precisa de ao menos `cv_folds` amostras. O padrão `cv_folds=7` não vem do número de QCs
    do estudo, que por coincidência também é 7: QC não entra em PLS-DA.

    A população é a de `sample_mask`, e não a do rótulo. O QC fica fora porque é a régua e não um
    grupo, e a amostra `validate` fica fora porque não é amostra de estudo. Pedir o rótulo dos
    QCs como grupo é recusado. Ver `_matriz_e_rotulos`.

    `vip` é não negativo onde é definido e não recebe orientação, porque importância não tem
    sinal. Só `scores` e `loadings` passam pela orientação canônica. Onde a soma de quadrados
    total é zero, `vip` sai `nan` e não zeros. Ver `_vip`.
    """
    x, y = _matriz_e_rotulos(matrix, groups, cv_folds)

    modelo = PLSRegression(n_components=n_components, scale=False).fit(x, y)
    escores, cargas = orient_canonically(
        np.asarray(modelo.x_scores_), np.asarray(modelo.x_loadings_)
    )

    predito = np.asarray(modelo.predict(x)).ravel()
    tss = float(((y - y.mean()) ** 2).sum())
    r2y = 1.0 - float(((y - predito) ** 2).sum()) / tss if tss > 0.0 else float("nan")

    q2, escores_cv = _validacao_cruzada(
        x,
        y,
        n_components=n_components,
        cv_folds=cv_folds,
        referencial=_rotacoes_no_referencial_dos_escores(modelo),
    )
    return PLSDAResult(
        scores=escores,
        cv_scores=escores_cv,
        loadings=cargas,
        vip=_vip(modelo),
        r2y=r2y,
        q2=q2,
        permutation_pvalue=_permutation_pvalue(
            x,
            y,
            q2,
            n_components=n_components,
            cv_folds=cv_folds,
            n=n_permutations,
            seed=seed,
            cancelled=cancelled,
        ),
        n_components=n_components,
        n_permutations=n_permutations,
        seed=seed,
    )


def _matriz_e_rotulos(
    matrix: RefinoMatrix, groups: tuple[str, str], cv_folds: int
) -> tuple[np.ndarray, np.ndarray]:
    """As linhas dos dois grupos e o vetor `0/1`, com as guardas de entrada.

    `sample_mask` é lida uma vez e governa as duas pontas: a lista de grupos disponíveis que a
    guarda de pertinência confere e a seleção das linhas que entram no modelo. O QC fica fora
    porque é a régua e não um grupo, e `validate` fica fora porque não é amostra de estudo.
    `sample_mask` não é o complemento de `qc_mask`, porque `SampleType` tem três valores.

    A mensagem de recusa não diz que o grupo não existe: o rótulo dos QCs existe na matriz, só
    não é grupo de estudo.

    Levanta quando um grupo tem menos amostras que `cv_folds`: uma dobra sem membro de uma
    classe produziria `PRESS` que mede outra coisa, e o `Q²` sairia plausível.
    """
    primeiro, segundo = groups
    if primeiro == segundo:
        raise ValueError(f"os dois grupos são o mesmo ({primeiro!r}); não há contraste a modelar")

    por_linha = np.array(matrix.groups)
    estudo = np.asarray(matrix.sample_mask)
    disponiveis = set(por_linha[estudo].tolist())
    for nome in groups:
        if nome not in disponiveis:
            raise ValueError(
                f"grupo {nome!r} não está entre as amostras de estudo desta matriz. "
                f"Disponíveis: {sorted(disponiveis)}"
            )

    # O `& estudo` impede que uma amostra `validate` com o rótulo de um dos dois grupos entre no
    # modelo e na contagem que a guarda de `cv_folds` confere.
    mascara = np.isin(por_linha, groups) & estudo
    x = np.asarray(matrix.values, dtype=np.float64)[mascara]
    y = (por_linha[mascara] == segundo).astype(np.float64)

    for nome, contagem in zip(groups, (int((y == 0.0).sum()), int((y == 1.0).sum())), strict=True):
        if contagem < max(MIN_AMOSTRAS_POR_GRUPO, cv_folds):
            raise ValueError(
                f"grupo {nome!r} tem {contagem} amostra(s) e cv_folds={cv_folds}: dobras "
                "estratificadas exigem ao menos `cv_folds` membros de cada classe, senão "
                "alguma dobra fica sem um dos grupos e o PRESS mede outra coisa"
            )
    return x, y


def _rotacoes_no_referencial_dos_escores(modelo: PLSRegression) -> np.ndarray:
    """As rotações do modelo de calibração com a mesma virada de sinal que os escores levaram.

    `x_rotations_` leva a matriz centrada aos escores (`transform` é
    `(X - média) @ x_rotations_`), então é o referencial dos escores e o alvo do Procrustes. Os
    escores de calibração saem de `orient_canonically`; sem a mesma virada no alvo, cada
    componente virado apareceria com o sinal trocado nos escores de validação cruzada.

    O sinal sai das cargas, pela mesma função e pela mesma regra dos escores: ela vira as
    colunas da primeira matriz pelo sinal do maior elemento da segunda, e as colunas das
    rotações são os mesmos componentes.
    """
    rotacoes, _ = orient_canonically(
        np.asarray(modelo.x_rotations_), np.asarray(modelo.x_loadings_)
    )
    return rotacoes


def _dobras(
    x: np.ndarray, y: np.ndarray, *, n_components: int, cv_folds: int
) -> Iterator[tuple[np.ndarray, PLSRegression, float]]:
    """As dobras, o modelo de cada uma e o PRESS das retidas.

    É o único laço de validação cruzada do módulo. O `Q²` observado, os escores de validação
    cruzada e o `Q²` de cada permutação passam por ele, para o `p` comparar quantidades medidas
    pelo mesmo procedimento. Na permutação a partição muda, porque o `StratifiedKFold`
    estratifica pelo `y` permutado.

    `shuffle=False` no `StratifiedKFold`: a ordem das amostras é a da matriz, que é
    determinística, então a partição é a mesma em toda execução e os escores de validação
    cruzada não dependem da semente. Ela governa só as permutações.
    """
    for treino, retido in StratifiedKFold(n_splits=cv_folds).split(x, y):
        ajuste = PLSRegression(n_components=n_components, scale=False).fit(x[treino], y[treino])
        predito = np.asarray(ajuste.predict(x[retido])).ravel()
        yield retido, ajuste, float(((y[retido] - predito) ** 2).sum())


def _q2_do_press(press: float, y: np.ndarray) -> float:
    """`1 - PRESS/TSS`, ou `nan` quando os rótulos não variam e não há o que predizer."""
    tss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - press / tss if tss > 0.0 else float("nan")


def _q2(x: np.ndarray, y: np.ndarray, *, n_components: int, cv_folds: int) -> float:
    """`1 - PRESS/TSS`, com o PRESS somado sobre as predições fora da dobra.

    É o caminho das permutações, que só precisam do número: não projeta nem gira nada.
    """
    press = 0.0
    for _retido, _ajuste, da_dobra in _dobras(x, y, n_components=n_components, cv_folds=cv_folds):
        press += da_dobra
    return _q2_do_press(press, y)


def _validacao_cruzada(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_components: int,
    cv_folds: int,
    referencial: np.ndarray,
) -> tuple[float, np.ndarray]:
    """O `Q²` e os escores de validação cruzada, do mesmo laço.

    Cada dobra ajusta o seu modelo sem as amostras retidas, e `transform` as projeta no
    referencial desse modelo, que pode ter um componente virado ou os dois rodados em relação ao
    de calibração. O giro é o Procrustes ortogonal das rotações da dobra para as de calibração
    (`referencial`, já com a virada canônica): a matriz ortogonal `Q` que torna `R_dobra @ Q` o
    mais próximo possível de `R_calibração`, aplicada aos escores das retidas. É ortogonal e sem
    escala: gira e reflete, não estica.

    As dobras estratificadas partem as linhas, então cada amostra é retida uma vez e recebe
    exatamente um escore. O `Q²` é o mesmo número que `_q2` daria, somado na mesma ordem.
    """
    press = 0.0
    escores = np.full((x.shape[0], n_components), np.nan)
    for retido, ajuste, da_dobra in _dobras(x, y, n_components=n_components, cv_folds=cv_folds):
        press += da_dobra
        giro, _ = orthogonal_procrustes(np.asarray(ajuste.x_rotations_), referencial)
        escores[retido] = np.asarray(ajuste.transform(x[retido])) @ giro
    return _q2_do_press(press, y), escores


def _permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    q2_observado: float,
    *,
    n_components: int,
    cv_folds: int,
    n: int,
    seed: int,
    cancelled: CancelCheck | None = None,
) -> float:
    """Fração de permutações cujo `Q²` iguala ou supera o observado, por `(k + 1) / (n + 1)`.

    Cada permutação tem gerador derivado de `(seed, índice)`, o que torna o resultado
    independente da ordem de execução e do paralelismo.
    """
    if not np.isfinite(q2_observado):
        return float("nan")

    superaram = 0
    for indice in range(n):
        # Ponto de cancelamento a cada permutação. Sem ele o laço inteiro, de alguns segundos,
        # correria sem parada.
        raise_if_cancelled(cancelled, "plsda: laco das permutacoes")
        gerador = np.random.default_rng([seed, indice])
        embaralhado = gerador.permutation(y)
        if not (0.0 < embaralhado.mean() < 1.0):
            # Permutação com todos num grupo só não é comparável; conta como não superando.
            continue
        if _q2(x, embaralhado, n_components=n_components, cv_folds=cv_folds) >= q2_observado:
            superaram += 1
    return (superaram + 1) / (n + 1)


def _vip(modelo: PLSRegression) -> np.ndarray:
    """Importância na projeção (VIP), por feature. Não negativa, e por isso sem orientação.

    `VIP_j² = p · Σ_a (w_ja² · SS_a) / Σ_a SS_a`, com `SS_a` a soma de quadrados explicada pelo
    componente `a`. A definição agrega o quadrado do peso, então o sinal do componente não a
    alcança.

    Quando a soma de quadrados total é zero, devolve `nan` em todas as features, e não zeros.
    Zero diria que nenhuma feature tem importância, quando o que houve foi a impossibilidade de
    calcular. Esse é o único ramo que produz `nan`: no resto, a raiz é de uma razão de somas não
    negativas por um total positivo. Quem consome conta as não finitas e as declara.
    """
    pesos = np.asarray(modelo.x_weights_)
    escores = np.asarray(modelo.x_scores_)
    cargas_y = np.asarray(modelo.y_loadings_)

    ss = np.array(
        [
            float((escores[:, a] ** 2).sum() * (cargas_y[:, a] ** 2).sum())
            for a in range(escores.shape[1])
        ]
    )
    total = float(ss.sum())
    if total == 0.0:
        return np.full(pesos.shape[0], np.nan, dtype=np.float64)

    normalizados = pesos**2 / np.maximum(
        (pesos**2).sum(axis=0, keepdims=True), np.finfo(float).tiny
    )
    vip: np.ndarray = np.sqrt(pesos.shape[0] * (normalizados * ss).sum(axis=1) / total)
    return vip

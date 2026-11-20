"""Descarte por ausência por grupo e as imputações.

`drop_by_group_missingness` remove a feature cuja fração de ausentes excede o
limiar em todos os grupos, e mantém a que está bem medida em ao menos um, porque
ausência específica de grupo é informação biológica. Trocar 'em todos' por 'em
algum' dá o conjunto oposto.

Zero é o ausente neste domínio: o equipamento deixa a célula vazia quando não
detecta, e no `pos_hilic` 15,8% das células são zero. O caso dominante é MNAR,
abaixo do limite de detecção, o que justifica imputar por valor pequeno, e não
pela média. Por isso `one_fifth_min` é o padrão do laboratório.

`none` é opção legítima, mas deixa a ausência passar. A combinação dela com
qualquer normalização é recusada na construção da especificação do pipeline,
porque a matriz sairia parecendo processada sem ser.
"""

import warnings
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from refino.domain.labels import Imputation, validate_label
from refino.domain.matrix import RefinoMatrix
from refino.preprocessing.stage import StageResult

__all__ = [
    "AUSENTE_EM_TODOS_OS_GRUPOS",
    "AUSENTE_NA_FEATURE",
    "drop_by_group_missingness",
    "impute",
    "is_missing",
]

AUSENTE_EM_TODOS_OS_GRUPOS = "ausente_acima_do_limiar_em_todos_os_grupos"

# Motivo do descarte de `impute("exclude")`.
AUSENTE_NA_FEATURE = "ausente_na_feature"


def is_missing(values: np.ndarray) -> np.ndarray:
    """Máscara de célula ausente: zero ou não finita.

    Zero é o ausente neste domínio, porque o equipamento deixa a célula vazia
    quando não detecta. `nan` chega de `impute("none")` ou de correção que
    produziu não finito.

    É pública para que todo o pipeline use uma definição só de ausência.
    """
    # Anotação local porque o `numpy` devolve `Any`.
    ausentes: np.ndarray = (values == 0.0) | ~np.isfinite(values)
    return ausentes


def drop_by_group_missingness(
    matrix: RefinoMatrix, threshold: float = 0.8
) -> tuple[RefinoMatrix, np.ndarray, StageResult]:
    """Remove feature cuja fração de ausentes excede `threshold` em todos os grupos.

    Mantém a feature bem medida em ao menos um grupo: ausência específica de
    grupo é informação biológica, como o metabólito presente numa condição
    clínica e ausente noutra. Descartar por 'em algum grupo' removeria justamente
    essas features.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(
            f"`threshold` é FRAÇÃO de ausentes e tem de estar em (0, 1]; recebi {threshold!r}. "
            "Limiar em percentual aqui nunca é excedido e o descarte vira identidade em "
            "silêncio."
        )

    valores = np.asarray(matrix.values)
    ausente = is_missing(valores)
    grupos = np.asarray(matrix.groups)

    excede_em_todos = np.ones(matrix.n_features, dtype=bool)
    for grupo in np.unique(grupos):
        fracao = ausente[grupos == grupo].mean(axis=0)
        excede_em_todos &= fracao > threshold

    manter = ~excede_em_todos
    resultado = StageResult(
        stage="drop_by_group_missingness",
        n_features_in=matrix.n_features,
        n_features_out=int(manter.sum()),
        discarded_by_reason={AUSENTE_EM_TODOS_OS_GRUPOS: int(excede_em_todos.sum())},
        labels={"threshold": f"{threshold:g}", "n_grupos": str(int(np.unique(grupos).size))},
    )
    nomes = tuple(n for n, fica in zip(matrix.feature_names, manter, strict=True) if fica)
    reduzida = replace(matrix, values=valores[:, manter], feature_names=nomes)
    return (
        reduzida.with_provenance(f"drop_by_group_missingness(threshold={threshold:g})"),
        manter,
        resultado,
    )


def impute(matrix: RefinoMatrix, method: Imputation, seed: int = 42) -> RefinoMatrix:
    """Substitui as células ausentes pelo método pedido. Determinístico dada a semente.

    O que cada método assume:

    - `one_fifth_min`: MNAR, o não detectado está abaixo do menor valor visto;
    - `col_min`: o mesmo MNAR, sem o quinto; o piso é o menor positivo observado;
    - `median`: ausência não informativa, e o centro da feature é o melhor palpite;
    - `mean`: o mesmo, com a média, que é sensível a extremos;
    - `knn`: amostras parecidas têm valores parecidos; o vizinho é outra amostra;
    - `knn_features`: features parecidas têm valores parecidos; o vizinho é outra
      feature;
    - `missforest`: estrutura não linear entre features, aprendida iterativamente;
    - `exclude`: a feature com qualquer ausente não é aproveitável;
    - `none`: nada. Não imputa, deixa a ausência como está, e a decisão é de quem
      consome.

    `one_fifth_min` é o padrão do laboratório e o coerente com MNAR, o caso
    dominante nestes dados. `median`, `mean`, `knn` e `missforest` imputam para o
    centro, o que sob MNAR puxa o não detectado para cima e reduz a variância
    medida. Por isso o método vai no `--log`.

    `col_min` e `one_fifth_min` diferem por um fator 5 e usam o mesmo auxiliar. O
    `ReplaceMin` do MetaboAnalyst é o `one_fifth_min` daqui, e o `min` deles é o
    `col_min`.

    `exclude` é o único que muda a forma da matriz. Os outros preservam o conjunto
    de features.
    """
    validate_label("Imputation", method)
    if method == "none":
        return matrix.with_provenance("impute(none)")

    valores = np.asarray(matrix.values, dtype=np.float64).copy()
    ausente = is_missing(valores)
    if not ausente.any():
        return matrix.with_provenance(f"impute({method}:nada-ausente)")

    if method == "exclude":
        return _exclui_features_com_ausente(matrix, ausente)

    # Despacho explícito, com `raise` no fim em vez de `else`: um valor novo no
    # `Literal` não cai em silêncio no ramo de outro método.
    if method == "one_fifth_min":
        valores = _preenche_por_coluna(valores, ausente, _um_quinto_do_menor_positivo)
    elif method == "col_min":
        valores = _preenche_por_coluna(valores, ausente, _menor_positivo)
    elif method == "median":
        valores = _preenche_por_coluna(valores, ausente, _mediana_das_presentes)
    elif method == "mean":
        valores = _preenche_por_coluna(valores, ausente, _media_das_presentes)
    elif method in ("knn", "knn_features", "missforest"):
        valores = _imputa_por_modelo(valores, ausente, method=method, seed=seed)
    else:  # pragma: no cover - inalcançável: a guarda acima fecha o vocabulário
        raise AssertionError(
            f"imputação {method!r} está no vocabulário e não tem ramo aqui. "
            "Valor novo no `Literal` exige ramo próprio: sem ele o método cairia noutra "
            "política e produziria número medido de forma diferente da pedida."
        )

    return matrix.with_values(valores, f"impute({method},seed={seed})")


def _exclui_features_com_ausente(matrix: RefinoMatrix, ausente: np.ndarray) -> RefinoMatrix:
    """Remove toda feature com qualquer célula ausente, o `exclude` do MetaboAnalyst.

    Fica entre as imputações, e não entre os filtros, porque é onde o
    MetaboAnalyst o oferece e porque o pressuposto é sobre a ausência.

    Não levanta com matriz vazia: quem levanta é `process_matrix`, com
    `EmptyMatrixError` nomeando a etapa.
    """
    manter = ~ausente.any(axis=0)
    valores = np.asarray(matrix.values, dtype=np.float64)[:, manter]
    nomes = tuple(n for n, fica in zip(matrix.feature_names, manter, strict=True) if fica)
    reduzida = replace(matrix, values=valores, feature_names=nomes)
    return reduzida.with_provenance("impute(exclude)")


def _menor_positivo(presentes: np.ndarray) -> np.ndarray:
    """O menor valor positivo de cada coluna; zero quando não há nenhum.

    Coluna sem positivo fica zero, e não `nan`: com `nan` a matriz sairia da
    imputação com ausência dentro.
    """
    # `catch_warnings` além de `errstate`: coluna sem positivo dá fatia toda `nan`,
    # e o `nanmin` avisa pelo módulo `warnings`, que `errstate` não alcança. O
    # `where` abaixo converte o `nan` em zero.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        minimos = np.nanmin(np.where(presentes > 0.0, presentes, np.nan), axis=0)
    return np.where(np.isfinite(minimos), minimos, 0.0)


def _resgata_amostras_sem_medida(
    resultado: np.ndarray, entrada: np.ndarray, *, alvo_ausente: np.ndarray
) -> np.ndarray:
    """Preenche a amostra sem nenhuma detecção, que o `KNNImputer` devolve como zeros.

    No eixo transposto uma amostra inteira vazia é uma coluna vazia, e
    `keep_empty_features=True` a mantém preenchida com zero. Zero aqui é ausência
    (`is_missing` o conta), então a imputação devolveria a mesma ausência.

    Essas linhas recebem a política de coluna `_um_quinto_do_menor_positivo`, e
    não uma recusa: `imputation` é eixo do multiverso, e um método que recusasse
    uma matriz que os outros aceitam tornaria as células não comparáveis. Um
    quinto do menor positivo, e não a média, porque amostra sem detecção é o caso
    extremo de MNAR.

    Não trata a feature vazia do caminho não transposto: ali o `sklearn` descarta
    a coluna, o que muda `n_features`, e não `n_samples`.
    """
    # A amostra sem medida é a linha inteira ausente na máscara. Não se lê do
    # resultado, onde um zero pode ter sido produzido pela imputação.
    sem_medida = alvo_ausente.all(axis=1)
    if not sem_medida.any():
        return resultado
    # A cópia é necessária: `_preenche_por_coluna` altera o array que recebe, e
    # `entrada` deriva dos valores da matriz, que é imutável.
    #
    # A estatística de coluna sai das amostras que têm medida: a linha sem medida
    # já é `nan` em `entrada`, e `nanmin` a ignora.
    preenchido = _preenche_por_coluna(entrada.copy(), alvo_ausente, _um_quinto_do_menor_positivo)
    return np.where(sem_medida[:, None], preenchido, resultado)


def _um_quinto_do_menor_positivo(presentes: np.ndarray) -> np.ndarray:
    """Um quinto do `_menor_positivo`. Coluna sem positivo continua a dar zero."""
    return _menor_positivo(presentes) / 5.0


def _mediana_das_presentes(presentes: np.ndarray) -> np.ndarray:
    """Mediana das células presentes de cada coluna; zero quando não há nenhuma."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        medianas = np.nanmedian(presentes, axis=0)
    return np.asarray(np.nan_to_num(medianas, nan=0.0), dtype=np.float64)


def _media_das_presentes(presentes: np.ndarray) -> np.ndarray:
    """Média das células presentes de cada coluna; zero quando não há nenhuma."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        medias = np.nanmean(presentes, axis=0)
    return np.asarray(np.nan_to_num(medias, nan=0.0), dtype=np.float64)


def _preenche_por_coluna(
    valores: np.ndarray,
    ausente: np.ndarray,
    por_coluna: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Aplica um preenchimento por coluna nas células ausentes."""
    presentes = np.where(ausente, np.nan, valores)
    preenchimento = por_coluna(presentes)
    valores[ausente] = np.broadcast_to(preenchimento, valores.shape)[ausente]
    return valores


def _imputa_por_modelo(
    valores: np.ndarray,
    ausente: np.ndarray,
    *,
    method: Imputation,
    seed: int,
    n_neighbors: int | None = None,
) -> np.ndarray:
    """`knn`, `knn_features` e `missforest`, via `sklearn`, com `nan` marcando a ausência.

    A ausência vira `nan` antes de entrar, que é como o `sklearn` a representa.
    Passar zero faria o modelo aprender que o não detectado é zero.

    `k = round(sqrt(n_colunas))` para os dois `knn`. É regra, e não valor fixo,
    porque `n_features` varia por modo.

    `n_neighbors` existe para medições de atribuição, e a produção não o usa. As
    duas variantes de `knn` diferem em quem é o vizinho e em quantos vizinhos há,
    porque transpor troca o que a regra conta. Fixar `k` de fora permite comparar
    as duas com só a transposição diferindo.
    """
    entrada = np.where(ausente, np.nan, valores)

    if method in ("knn", "knn_features"):
        from sklearn.impute import KNNImputer

        # A transposição decide quem é vizinho. O `KNNImputer` imputa cada linha a
        # partir das linhas mais próximas. Na orientação interna as linhas são
        # amostras, e `knn` é o `KNN (sample-wise)` do MetaboAnalyst. Sobre a
        # transposta as linhas são features, o `KNN (feature-wise)`.
        #
        # O `k` muda junto, porque a regra é `round(sqrt(n_colunas))` e transpor
        # troca o que são as colunas.
        por_feature = method == "knn_features"
        alvo = entrada.T if por_feature else entrada
        k = n_neighbors if n_neighbors is not None else max(1, round(float(np.sqrt(alvo.shape[1]))))
        # `weights="uniform"` é a média dos vizinhos.
        #
        # `keep_empty_features` só no caminho transposto: ali uma coluna vazia é
        # uma amostra inteira sem medida, e o padrão do `sklearn` a descartaria,
        # mudando o número de amostras, que `with_values` recusa.
        imputador = (
            KNNImputer(n_neighbors=k, weights="uniform", keep_empty_features=True)
            if por_feature
            else KNNImputer(n_neighbors=k, weights="uniform")
        )
        resultado = np.asarray(imputador.fit_transform(alvo), dtype=np.float64)
        if por_feature:
            resultado = resultado.T
            resultado = _resgata_amostras_sem_medida(resultado, entrada, alvo_ausente=ausente)
        return resultado

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    floresta = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    imputador = IterativeImputer(estimator=floresta, random_state=seed, max_iter=10)
    return np.asarray(imputador.fit_transform(entrada), dtype=np.float64)

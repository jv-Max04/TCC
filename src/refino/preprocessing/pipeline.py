"""O executor da ordem canônica, com contagem por etapa.

A ordem é dado: uma tupla nomeada de etapas, em vez de chamadas aninhadas. Assim
a sequência é inspecionável por teste, e o multiverso troca métodos sem
reescrever o executor.

A ordem:

    filtro de RSD -> filtro por posto (dois passes) -> variância quase nula
    -> ausentes por grupo -> imputação -> normalização por amostra
    -> transformação -> escalonamento

As cinco primeiras etapas filtram: reduzem features e reportam por quê. As três
últimas preservam a forma e mudam os valores. `impute` fica no meio: preserva a
forma em todo método menos `exclude`, e por isso a contagem dele é medida.
`PipelineSpec` recusa na construção a combinação de `imputation="none"` com
qualquer normalização, então o executor não a confere.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from refino.domain.labels import Imputation
from refino.domain.matrix import RefinoMatrix
from refino.domain.spec import PipelineSpec
from refino.preprocessing.filtering import (
    filter_by_qc_rsd,
    filter_by_ranking,
    filter_near_zero_variance,
)
from refino.preprocessing.missing import (
    AUSENTE_NA_FEATURE,
    drop_by_group_missingness,
    impute,
    is_missing,
)
from refino.preprocessing.normalization import (
    normalize_samples_declarada,
    scale_features,
    transform,
)
from refino.preprocessing.stage import StageResult

__all__ = [
    "CANONICAL_ORDER",
    "EmptyMatrixError",
    "ProcessedMatrix",
    "chain_is_consistent",
    "process_matrix",
    "process_matrix_detailed",
    "requests_preprocessing",
    "stage_counts",
]

# A ordem canônica como valor. Os nomes são os que aparecem no `--log` e no
# relatório, e o que se afirma é a sequência, e não o conjunto.
# `tests/integration/test_cli_processar.py` compara o `stage_order` do log e
# esta constante com a mesma lista.
CANONICAL_ORDER: tuple[str, ...] = (
    "filter_by_qc_rsd",
    "filter_by_ranking",
    "filter_by_ranking_2",
    "filter_near_zero_variance",
    "drop_by_group_missingness",
    "impute",
    "normalize_samples",
    "transform",
    "scale_features",
)


class EmptyMatrixError(ValueError):
    """Uma etapa descartou todas as features.

    Levanta em vez de seguir: as etapas seguintes produziriam estatísticas sobre
    conjunto vazio, com aparência de medida. A mensagem nomeia a etapa porque a
    ação depende dela: corte de RSD agressivo demais pede outro corte, ausência
    em todos os grupos pede olhar o dado.
    """


@dataclass(frozen=True)
class ProcessedMatrix:
    """O que o pipeline produziu, incluindo a matriz anterior à transformação.

    O volcano precisa de duas matrizes: o `p` sai da processada, e o fold change
    sai de antes da transformação, porque depois do `log10` e de um escalonamento
    que centra a mediana de um grupo pode ser negativa, e a razão fica
    indefinida.

    `pre_transform` e `matrix` têm as mesmas features e as mesmas amostras: as
    etapas que removem features correm todas antes da normalização, e nenhuma
    remove amostras. Quem usa as duas em par as passa a `univariate_test`, que
    confere a compatibilidade.
    """

    matrix: RefinoMatrix
    stages: tuple[StageResult, ...]
    pre_transform: RefinoMatrix
    # As células de `pre_transform` cujo valor a imputação escolheu, com a mesma
    # forma de `pre_transform` e de `matrix`. O intervalo do fold change precisa
    # delas: uma mediana sobre valores imputados daria um intervalo estreito em
    # volta de um número que a imputação escolheu.
    imputed_cells: np.ndarray

    @property
    def transform_moved_the_matrix(self) -> bool:
        """Se as duas matrizes diferem. São iguais com `transformation` e `scaling` em `none`.

        Serve para a figura não anunciar 'fold change da pré-transformação' quando
        as duas matrizes são a mesma.
        """
        return not np.array_equal(
            np.asarray(self.matrix.values), np.asarray(self.pre_transform.values), equal_nan=True
        )


def requests_preprocessing(spec: PipelineSpec) -> bool:
    """Se a especificação pede alguma etapa além do filtro de variância quase nula.

    Serve a quem pode não chamar o executor, como a exportação para o
    MetaboAnalyst. `process_matrix` corre `filter_near_zero_variance` sempre, então
    mesmo com todos os eixos neutros a saída não é a entrada. Quem não pede
    nenhuma etapa tem de receber a matriz corrigida inteira, e para isso o
    executor não pode correr.

    Entram só os campos que ligam uma etapa. As frações de posto sem o método, e
    a referência ou os fatores sem a normalização que os lê, não mudam nada. A CLI
    e a rota usam esta mesma função, para decidirem igual.
    """
    return (
        spec.rsd_cutoff is not None
        or spec.missing_threshold is not None
        or spec.ranking_filter != "none"
        or spec.ranking_filter_2 != "none"
        or spec.imputation != "none"
        or spec.normalizes()
    )


def process_matrix(
    matrix: RefinoMatrix, spec: PipelineSpec
) -> tuple[RefinoMatrix, list[StageResult]]:
    """Aplica as etapas de `CANONICAL_ORDER`, na ordem. Devolve a matriz e uma contagem por etapa.

    Chama `process_matrix_detailed`, que devolve também a matriz anterior à
    transformação.
    """
    detalhado = process_matrix_detailed(matrix, spec)
    return detalhado.matrix, list(detalhado.stages)


def process_matrix_detailed(matrix: RefinoMatrix, spec: PipelineSpec) -> ProcessedMatrix:
    """Aplica as etapas de `CANONICAL_ORDER`, na ordem, e devolve também a matriz intermediária.

    A matriz de entrada não é alterada: cada etapa devolve nova `RefinoMatrix`
    com `provenance` acrescida, e a lista de `StageResult` sai na ordem em que as
    etapas rodaram.

    `rsd_cutoff=None` e `missing_threshold=None` saltam a etapa correspondente,
    e o salto é registrado com contagem igual na entrada e na saída. Uma etapa
    ausente da lista quebraria o encadeamento de contagens.
    """
    atual = matrix
    resultados: list[StageResult] = []

    # A presença da referência é decidida aqui, antes do primeiro filtro. Depois
    # dos filtros, 'o nome nunca esteve aqui' e 'um filtro o removeu' são
    # indistinguíveis, e pedem mensagens diferentes.
    #
    # O veredito é por eixo: nome de feature se procura entre as features, e nome
    # de amostra, entre as amostras. Só o ramo `feature_reference` lê a bandeira.
    referencia_na_entrada: bool | None = None
    if spec.normalization_reference is not None:
        alvo = spec.normalization_reference
        if spec.sample_normalization == "feature_reference":
            referencia_na_entrada = alvo in matrix.feature_names
        elif spec.sample_normalization == "sample_reference":
            referencia_na_entrada = alvo in {amostra.name for amostra in matrix.samples}

    if spec.rsd_cutoff is None:
        resultados.append(_saltada("filter_by_qc_rsd", atual, "rsd_cutoff=None"))
    else:
        atual, _, resultado = filter_by_qc_rsd(atual, spec.rsd_cutoff, variant=spec.rsd_variant)
        resultados.append(resultado)
        _exige_features(atual, resultado)

    # O filtro por posto vem depois do corte por RSD dos QCs e opera sobre a
    # população que sobreviveu, na ordem em que o MetaboAnalyst as apresenta. Com
    # `ranking_filter="none"` a etapa é saltada.
    atual, _, resultado = filter_by_ranking(
        atual, spec.ranking_filter, keep_fraction=spec.ranking_keep_fraction
    )
    resultados.append(resultado)
    _exige_features(atual, resultado)

    # O segundo passe opera sobre a população que o primeiro deixou. O nome da
    # etapa é distinto; ver a docstring de `filter_by_ranking`.
    atual, _, resultado = filter_by_ranking(
        atual,
        spec.ranking_filter_2,
        keep_fraction=spec.ranking_keep_fraction_2,
        stage="filter_by_ranking_2",
        campo_do_contrato="ranking_filter_2",
    )
    resultados.append(resultado)
    _exige_features(atual, resultado)

    atual, _, resultado = filter_near_zero_variance(atual)
    resultados.append(resultado)
    _exige_features(atual, resultado)

    if spec.missing_threshold is None:
        resultados.append(_saltada("drop_by_group_missingness", atual, "missing_threshold=None"))
    else:
        atual, _, resultado = drop_by_group_missingness(atual, spec.missing_threshold)
        resultados.append(resultado)
        _exige_features(atual, resultado)

    # A contagem do `impute` é medida: `exclude` remove a feature com qualquer
    # ausente, e os outros métodos preservam o conjunto de features.
    antes_do_impute = atual.n_features
    entrou_na_imputacao = atual
    atual = impute(atual, spec.imputation, seed=spec.seed)
    imputadas = _celulas_imputadas(entrou_na_imputacao, atual, spec.imputation)
    resultado = StageResult(
        stage="impute",
        n_features_in=antes_do_impute,
        # O motivo vai nomeado quando a etapa descarta, para que o
        # `EmptyMatrixError` tenha o que mostrar.
        discarded_by_reason=(
            {AUSENTE_NA_FEATURE: antes_do_impute - atual.n_features}
            if spec.imputation == "exclude" and atual.n_features < antes_do_impute
            else {}
        ),
        n_features_out=atual.n_features,
        labels={"method": spec.imputation},
    )
    resultados.append(resultado)
    _exige_features(atual, resultado)

    declarada = normalize_samples_declarada(
        atual,
        spec.sample_normalization,
        reference=spec.normalization_reference,
        reference_present_at_entry=referencia_na_entrada,
        factors=spec.normalization_factors,
    )
    atual = declarada.matrix
    # As linhas sem denominador entram no relatório. A chave só aparece quando há
    # o que declarar.
    if declarada.linhas_sem_denominador:
        resultados.append(
            _transformadora(
                "normalize_samples",
                atual,
                {
                    "method": spec.sample_normalization,
                    "linhas_sem_denominador": str(len(declarada.linhas_sem_denominador)),
                    "amostras_sem_denominador": ", ".join(declarada.linhas_sem_denominador[:8]),
                },
            )
        )
    else:
        resultados.append(
            _transformadora("normalize_samples", atual, {"method": spec.sample_normalization})
        )

    # A matriz pré-transformação é capturada aqui, depois da normalização por
    # amostra. O MetaboAnalyst calcula o fold change sobre o dado normalizado e
    # não transformado: a normalização já corrigiu a escala entre injeções, e o
    # `log` ainda não produziu medianas negativas.
    pre_transformacao = atual

    atual = transform(atual, spec.transformation)
    resultados.append(_transformadora("transform", atual, {"method": spec.transformation}))

    atual = scale_features(atual, spec.scaling)
    resultados.append(_transformadora("scale_features", atual, {"method": spec.scaling}))

    return ProcessedMatrix(
        matrix=atual,
        stages=tuple(resultados),
        pre_transform=pre_transformacao,
        imputed_cells=imputadas,
    )


def _celulas_imputadas(
    entrada: RefinoMatrix, saida: RefinoMatrix, metodo: Imputation
) -> np.ndarray:
    """As células cujo valor a imputação escolheu: as ausentes da matriz que entrou nela.

    Ausente é `is_missing`, a mesma definição que `impute` usa. A máscara é lida
    da entrada porque depois da imputação não há como reconhecer essas células.

    Com `none` e com `exclude` a máscara fica vazia, porque nenhum valor foi
    escolhido: `none` deixa a ausência como está, e `exclude` remove a feature
    com ausente.

    As colunas seguem a saída. Só quando a lista de features mudou a máscara é
    recortada pelo nome; recortar sempre pelo nome levaria duas colunas de nome
    repetido à mesma posição.
    """
    if metodo == "none":
        mascara = np.zeros((saida.n_samples, saida.n_features), dtype=bool)
    else:
        mascara = is_missing(np.asarray(entrada.values, dtype=np.float64))
        if saida.feature_names != entrada.feature_names:
            posicao = {nome: j for j, nome in enumerate(entrada.feature_names)}
            mascara = mascara[:, [posicao[nome] for nome in saida.feature_names]]
    # Só leitura: o `dataclass` congelado não protege o conteúdo do array.
    mascara.setflags(write=False)
    return mascara


def _saltada(etapa: str, matrix: RefinoMatrix, motivo: str) -> StageResult:
    """Etapa saltada por parâmetro nulo. Entra na lista para o encadeamento não quebrar."""
    return StageResult(
        stage=etapa,
        n_features_in=matrix.n_features,
        n_features_out=matrix.n_features,
        labels={"saltada": motivo},
    )


def _transformadora(etapa: str, matrix: RefinoMatrix, rotulos: dict[str, str]) -> StageResult:
    """Etapa que preserva a forma. Entrada e saída iguais por construção.

    A contagem igual entra na lista para manter o encadeamento: a saída de cada
    etapa tem de ser a entrada da seguinte.
    """
    return StageResult(
        stage=etapa,
        n_features_in=matrix.n_features,
        n_features_out=matrix.n_features,
        labels=rotulos,
    )


def _exige_features(matrix: RefinoMatrix, resultado: StageResult) -> None:
    """Levanta quando a etapa esvaziou a matriz."""
    if matrix.n_features == 0:
        motivos = ", ".join(f"{k}={v}" for k, v in sorted(resultado.discarded_by_reason.items()))
        raise EmptyMatrixError(
            f"a etapa {resultado.stage!r} descartou TODAS as {resultado.n_features_in} "
            f"features ({motivos}). Seguir produziria estatisticas sobre conjunto vazio, com "
            "aparencia de medida; a acao depende da etapa - corte de RSD agressivo demais "
            "pede outro corte, ausencia em todos os grupos pede olhar o dado."
        )


def stage_counts(resultados: list[StageResult]) -> dict[str, dict[str, object]]:
    """As contagens por etapa, no formato que o `--log` carrega.

    Fica aqui, e não no módulo da CLI, porque o multiverso também as consome.
    """
    return {
        resultado.stage: {
            "n_features_in": resultado.n_features_in,
            "n_features_out": resultado.n_features_out,
            "discarded_by_reason": dict(resultado.discarded_by_reason),
            "labels": dict(resultado.labels),
        }
        for resultado in resultados
    }


def chain_is_consistent(resultados: list[StageResult]) -> bool:
    """Se a saída de cada etapa é a entrada da seguinte.

    Contagem que não encadeia significa que alguma etapa mediu matriz diferente
    da que passou por ela. O `--log` permite reconferir depois.
    """
    return all(
        anterior.n_features_out == seguinte.n_features_in
        for anterior, seguinte in pairwise(resultados)
    )


def as_matrix_of_counts(resultados: list[StageResult]) -> np.ndarray:
    """Array com o par `(entrada, saída)` de cada etapa, para relatório e teste."""
    return np.array([[r.n_features_in, r.n_features_out] for r in resultados], dtype=np.int64)

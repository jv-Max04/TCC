"""Corretor SERRF: os onze passos de `serrfR`, do `tools.R`, cientes de lote.

No `tools.R`, `e[j, …]` é o lote corrente e `all[j, …]` é a matriz inteira depois do passo 1. Os
passos 5, 6, 7 e 9 usam âncora global, e com um lote as duas coincidem numericamente. Aqui os dois
vetores vão explicitamente para as funções de `_serrf_steps`, e `tests/unit/test_serrf_scope.py`
constrói dois lotes em que as âncoras divergem.

Cada sorteio tem gerador próprio, derivado de `(seed, feature, lote, passo)`. Uma semente global
consumida na ordem em que o escalonador entrega as features faria a saída depender do
paralelismo. O número do passo entra para que os passos 1, 8 e 10 da mesma feature não partilhem
gerador.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Final, Literal, get_args

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

from refino.correction._floresta_ranger import FlorestaRanger
from refino.correction._forest_speedup import cached_param_names
from refino.correction._serrf_steps import (
    RANGER_MAX_FEATURES,
    RANGER_MIN_SAMPLES_SPLIT,
    RANGER_N_ESTIMATORS,
    correlation_matrices,
    normalize_qc,
    normalize_samples,
    repair_assembled_row,
    replace_non_finite,
    rescale_by_median,
    scale_by_columns_of,
    select_predictors,
    step1_params,
    subtractive_outlier_repair,
    usable_predictor_columns,
)
from refino.correction.base import (
    feature_columns,
    resolve_target_features,
    restore_zeros_and_missing,
    select_features,
)
from refino.domain.cancellation import CancelCheck, raise_if_cancelled
from refino.domain.correction import CorrectionResult
from refino.domain.labels import UncorrectedCause
from refino.domain.matrix import RefinoMatrix
from refino.metrics.rsd import boxplot_outlier_mask

# As exceções que o `loky` levanta quando um worker morre. Elas vêm do módulo embutido no
# `joblib`, então a importação é defensiva: se falhar, o tuplo fica vazio, `except ()` nunca
# casa e a exceção sobe, sem recuo para série.
try:  # pragma: no cover - o ramo de exceção só corre num `joblib` com outra estrutura
    from joblib.externals.loky.process_executor import (
        BrokenProcessPool as _BrokenProcessPool,
    )
    from joblib.externals.loky.process_executor import (
        TerminatedWorkerError as _TerminatedWorkerError,
    )

    _MORTE_DO_POOL: tuple[type[BaseException], ...] = (
        _TerminatedWorkerError,
        _BrokenProcessPool,
    )
except Exception:  # pragma: no cover - idem
    _MORTE_DO_POOL = ()

_LOGGER = logging.getLogger(__name__)

__all__ = ["IntensidadeNegativaError", "SerrfCorrector"]

# `coef = 3` no passo 9, contra `1.5` do `remove_outlier()`/`RSD()`. Os dois valores estão
# no `tools.R` e não são intercambiáveis.
STEP9_OUTLIER_COEF = 3.0

# Quantas features correlacionadas o passo 3 pede.
N_CORRELATED = 10

# Componente de passo na semente derivada. São constantes nomeadas porque trocar um valor
# muda a saída sem erro nenhum.
_STREAM_STEP1_ZEROS = 1
_STREAM_STEP1_NA = 2
_STREAM_FOREST = 4
_STREAM_STEP8 = 8
_STREAM_STEP10 = 10

# As duas florestas do passo 4. `scikit_learn` é o `RandomForestRegressor`, cujo
# `min_samples_split` conta as linhas distintas do bootstrap: com sete QCs, cerca de 41% das
# árvores não dividem. `ranger` é `FlorestaRanger`, que repete as linhas sorteadas e conta como o
# `ranger` conta. O padrão é `ranger`, a floresta que mais se aproxima do SERRF do R, que é o
# que o laboratório usa.
Floresta = Literal["scikit_learn", "ranger"]
FLORESTA_PADRAO: Final[Floresta] = "ranger"


class IntensidadeNegativaError(ValueError):
    """Intensidade abaixo de -0,1 numa coluna com zeros ou ausentes.

    O passo 1 sorteia os zeros e os ausentes de cada `(feature, lote)` com desvio
    `0,1 · (mínimo + 0,1)` (`tools.R:1042-1043`), que fica negativo com mínimo abaixo de -0,1. No
    R, `sd < 0` faz o `rnorm` devolver `NaN`; o `numpy` levanta `ValueError: scale < 0`, sem dizer
    onde. Esta recusa diz qual feature e qual lote. Intensidade negativa é fisicamente impossível,
    e a ingestão já a avisa com `NEGATIVE_INTENSITY`.

    Desce de `ValueError` para que quem não queira a distinção continue a apanhá-la.
    """


def _recusa_desvio_negativo(
    desvio: float, index: int, lote: object, nomes: tuple[str, ...] | None
) -> None:
    if desvio < 0.0:
        nome = nomes[index] if nomes is not None else f"coluna {index}"
        raise IntensidadeNegativaError(
            f"a feature {nome!r} tem, no lote {lote!r}, zeros ou ausentes ao lado de intensidade "
            "abaixo de -0,1: o passo 1 do SERRF sorteia com desvio 0,1·(mínimo + 0,1), que fica "
            "negativo. Intensidade negativa é impossível numa medição; confira a exportação"
        )


@dataclass(frozen=True)
class _Outcome:
    """Resultado por feature, com a causa quando não houve correção."""

    values: np.ndarray
    cause: UncorrectedCause | None
    tree_depths: list[int]
    # Quantos preditores o passo 3 reteve de fato, por lote. Pode ser menos que `n_correlated`:
    # com menos features que o pedido o laço do R não termina, e o teto de `select_predictors`
    # devolve o que houver.
    n_predictors: list[int]


class SerrfCorrector:
    """SERRF ciente de lote, com a floresta nos hiperparâmetros do `ranger`.

    Os hiperparâmetros são os do R para que uma divergência contra ele aponte defeito de
    porte, e não efeito de hiperparâmetro.
    """

    name = "serrf"

    def __init__(
        self,
        *,
        n_correlated: int = N_CORRELATED,
        fast_param_names: bool = True,
        n_jobs: int = 1,
        hoist_correlations: bool = True,
        floresta: Floresta | None = None,
    ) -> None:
        # Guarda em tempo de execução, além do `Literal`: um valor fora do domínio cairia no
        # `else` da escolha da floresta e correria a outra sem erro. `None` é o padrão do
        # módulo, resolvido aqui.
        escolhida = FLORESTA_PADRAO if floresta is None else floresta
        if escolhida not in get_args(Floresta):
            raise ValueError(
                f"floresta desconhecida: {floresta!r}; as válidas são {get_args(Floresta)}"
            )
        self.floresta: Floresta = escolhida
        self.n_correlated = n_correlated
        # A correlação do passo 2 não depende da feature-alvo, só do lote. Calculá-la uma vez
        # por lote, e não uma vez por feature, dá o mesmo resultado; a flag existe para
        # `test_forest_speedup.py` comparar os dois caminhos.
        self.hoist_correlations = hoist_correlations
        # Liga o cache de `_forest_speedup`. Muda o tempo e não o resultado; a flag existe
        # para `test_forest_speedup.py` comparar os dois caminhos.
        self.fast_param_names = fast_param_names
        # Padrão 1: em matriz pequena, como as dos testes, criar processos custa mais que o
        # trabalho, e o ganho de `-1` só existe em matriz real. A saída é idêntica entre `1` e
        # `-1`, porque a semente é derivada de `(seed, feature, lote, passo)`.
        self.n_jobs = n_jobs

    def identidade(self) -> str:
        """Identidade de construtor: `n_correlated` e `floresta`, os eixos que mudam valores.

        `n_correlated` decide quantos preditores o passo 3 pede, e `floresta` escolhe a
        implementação do passo 4.

        `n_jobs`, `hoist_correlations` e `fast_param_names` ficam de fora porque mudam o tempo e
        não a saída. `tests/unit/test_forest_speedup.py` compara cada par, e o de `n_jobs` confere
        valores, `uncorrected` e `provenance`, que carrega a profundidade média das árvores. Sem
        esses testes o reuso de correção deixa de ser fiel. O par de `n_jobs` pula quando
        `joblib.effective_n_jobs(-1)` devolve 1; nesse caso `-1` e `1` são o mesmo regime, o que
        `tests/unit/test_identidade_do_corretor.py` afirma.

        Eixo novo no construtor tem de entrar aqui ou na tabela de exclusão declarada.
        """
        return f"{self.name}(n_correlated={self.n_correlated},floresta={self.floresta})"

    def fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        cancelled: CancelCheck | None = None,
    ) -> CorrectionResult:
        """Corrige feature por feature e lote por lote, e marca o que não corrigiu.

        O conjunto de features da saída é o da entrada: nada é descartado. Feature que não pôde
        ser corrigida volta marcada com a causa; o `tools.R` a devolve sem correção e sem marcar.

        Execução parcial é exata. `target_features` restringe quais features recebem floresta; o
        passo 1 e a seleção de preditores continuam vendo a matriz inteira, porque o conjunto de
        preditores do passo 3 é global e reduzi-lo mudaria a seleção. Os passos 1 e 10 são por
        feature, o passo 11 é por célula, e a semente carrega o índice original da feature.

        `cancelled` é lido em três pontos: no laço do passo 1, antes do cálculo das correlações e
        no despacho de cada feature ao `Parallel`. O terceiro fica no gerador que alimenta o
        `Parallel`: com `n_jobs=1` o `joblib` o consome item a item, no mesmo processo, e a
        checagem cai entre duas features. Com `n_jobs > 1` a bandeira do pai não alcança os
        processos filhos; a checagem só deixa de enviar trabalho novo, e as features já
        despachadas terminam.
        """
        return self._fit_transform(
            matrix,
            batches=batches,
            seed=seed,
            target_features=target_features,
            cancelled=cancelled,
        )

    def _fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        cancelled: CancelCheck | None = None,
    ) -> CorrectionResult:
        """Os onze passos. O cache da floresta é instalado em `_correct_feature`, no processo
        que ajusta."""
        values = np.asarray(matrix.values, dtype=np.float64)
        labels = self._batch_labels(matrix, batches)
        is_qc = np.asarray(matrix.qc_mask, dtype=bool)
        # `sample_mask` e não `~qc_mask`: há três valores em `SampleType`, e `validate` não
        # é amostra de estudo.
        is_sample = np.asarray(matrix.sample_mask, dtype=bool)

        # Passo 1 uma vez, para a matriz inteira: é ele que define `all[j, …]`, e recalculá-lo
        # por lote daria âncora global diferente por lote. Só QCs e amostras de estudo entram:
        # no R o passo 1 altera `all = cbind(train, target)`, com o alvo `e_sample` (linha 1235),
        # e `validate` não está lá. Ela volta inalterada: o passo 11 repõe os zeros e ausentes
        # dela, e nada mais a toca.
        after_step1 = self._step1(
            values,
            labels=labels,
            seed=seed,
            cancelled=cancelled,
            linhas=is_qc | is_sample,
            nomes=matrix.feature_names,
        )
        batch_order = _ordered_unique(labels)

        # O pedido decide quais features recebem floresta. `after_step1` continua sendo a
        # matriz inteira, porque a seleção de preditores do passo 3 é global.
        requested = resolve_target_features(matrix, target_features)
        columns = feature_columns(matrix, requested)

        corrected = np.array(after_step1[:, columns], copy=True)
        causes: dict[UncorrectedCause, set[str]] = defaultdict(set)
        depths: list[int] = []
        retidos: list[int] = []

        # Uma correlação por lote, e não por (lote, feature). A chave é a posição do lote, que
        # é o que `_correct_feature` tem em mãos.
        hoisted: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        # `correlation_matrices` é uma chamada única por lote, sem ponto interno onde checar o
        # cancelamento. A checagem vem antes dela.
        raise_if_cancelled(cancelled, "serrf: antes do icamento das correlacoes")
        if self.hoist_correlations:
            hoisted = {
                # A matriz de antes do passo 1 e o alvo sem `validate`; o porquê está na
                # docstring de `correlation_matrices`.
                batch_position: correlation_matrices(
                    values.T, is_qc=is_qc, in_batch=labels == batch, is_target=is_sample
                )
                for batch_position, batch in enumerate(batch_order)
                if ((labels == batch) & is_qc).any() and ((labels == batch) & is_sample).any()
            }

        # Paralelismo sobre features. `_correct_feature` recebe `after_step1` inteiro e a
        # semente é derivada de `(seed, index, lote, passo)`: não há estado compartilhado nem
        # dependência da ordem de execução, e `n_jobs` muda o tempo e não o resultado.
        #
        # `Parallel` preserva a ordem da entrada, então `depths` acumula na mesma sequência
        # e a profundidade média do `provenance` é a mesma.
        def _ajusta(n_jobs: int) -> list[Any]:
            return list(
                Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(self._correct_feature)(
                        after_step1,
                        before_step1=values,
                        index=index,
                        labels=labels,
                        batch_order=batch_order,
                        is_qc=is_qc,
                        is_sample=is_sample,
                        seed=seed,
                        hoisted=hoisted,
                    )
                    for index in _indices_ate_cancelar(columns, cancelled)
                )
            )

        # Recuo para série. Se o pool de workers morre no meio (falta de memória, o sistema
        # matando processos), o ajuste é refeito em série, uma vez: o resultado é o mesmo e só
        # o tempo muda. Se a série também falhar, o erro sobe. Este módulo não importa nada de
        # `desktop` e não rebaixa a capacidade do processo; apenas refaz este ajuste.
        if self.n_jobs == 1:
            outcomes = _ajusta(1)
        else:
            try:
                outcomes = _ajusta(self.n_jobs)
            except _MORTE_DO_POOL:
                _LOGGER.warning(
                    "o pool de workers morreu com n_jobs=%s; refazendo o ajuste em SERIE. "
                    "O resultado e identico — o que muda e o tempo",
                    self.n_jobs,
                )
                outcomes = _ajusta(1)

        for column, (feature, outcome) in enumerate(zip(requested, outcomes, strict=True)):
            corrected[:, column] = outcome.values
            depths.extend(outcome.tree_depths)
            retidos.extend(outcome.n_predictors)
            if outcome.cause is not None:
                causes[outcome.cause].add(feature)

        # Passo 10: fora do laço de features e de lotes, sobre a linha montada.
        corrected = self._step10(
            corrected, is_qc=is_qc, is_sample=is_sample, seed=seed, feature_indices=columns
        )

        # Passo 11: compara com a matriz de entrada, e não com a de depois do passo 1, em
        # que os zeros já foram substituídos. Recortada às colunas pedidas, para comparar
        # o mesmo espaço.
        restoration = restore_zeros_and_missing(values[:, columns], corrected)

        # A profundidade média das árvores é medida e vai no `provenance`. Com sete QCs o
        # `ranger` faz uma a duas divisões por árvore; profundidade zero denunciaria o
        # mapeamento errado de `min.node.size` (ver `RANGER_MIN_SAMPLES_SPLIT`).
        mean_depth = float(np.mean(depths)) if depths else float("nan")
        note = (
            f"correction:serrf(seed={seed},n_correlated={self.n_correlated},"
            f"floresta={self.floresta},"
            f"trees={RANGER_N_ESTIMATORS},n_trees_measured={len(depths)},"
            f"mean_tree_depth={mean_depth:.4f},"
            # O número efetivo de preditores ao lado do pedido: com `n_correlated` maior
            # que o número de features o passo 3 devolve menos do que se pediu. `min` e
            # `max` porque a retenção varia por feature e por lote.
            f"n_predictors_min={min(retidos) if retidos else 0},"
            f"n_predictors_max={max(retidos) if retidos else 0},"
            f"zeros_restored={restoration.restored},zeros_created={restoration.created},"
            f"uncorrected={sum(len(f) for f in causes.values())})"
        )
        return CorrectionResult(
            matrix=select_features(matrix, requested).with_values(restoration.values, note),
            uncorrected={cause: frozenset(features) for cause, features in causes.items()},
            requested=requested,
        )

    # -----------------------------------------------------------------------

    @staticmethod
    def _batch_labels(matrix: RefinoMatrix, batches: np.ndarray | None) -> np.ndarray:
        if batches is None:
            return np.asarray(matrix.batches, dtype=object)
        labels = np.asarray(batches, dtype=object)
        if labels.shape != (matrix.n_samples,):
            raise ValueError(
                f"`batches` tem de estar alinhado a `samples`: recebi {labels.shape} "
                f"para {matrix.n_samples} amostras"
            )
        return labels

    def _step1(
        self,
        values: np.ndarray,
        *,
        labels: np.ndarray,
        seed: int,
        cancelled: CancelCheck | None = None,
        linhas: np.ndarray | None = None,
        nomes: tuple[str, ...] | None = None,
    ) -> np.ndarray:
        """Passo 1 por `(feature, lote)`, zeros antes de ausentes.

        A ordem é a do R: o ramo dos zeros vê o vetor original e o dos ausentes vê o vetor
        com os zeros já substituídos, onde o mínimo não é mais zero.
        """
        result = np.array(np.asarray(values, dtype=np.float64), copy=True)
        # `linhas` restringe o passo às amostras que o SERRF corrige; ausente, são todas.
        no_passo = np.ones(result.shape[0], dtype=bool) if linhas is None else linhas
        for batch_position, batch in enumerate(_ordered_unique(labels)):
            rows = np.flatnonzero((labels == batch) & no_passo)
            if rows.size == 0:
                continue
            for index in range(result.shape[1]):
                raise_if_cancelled(cancelled, "serrf: laco do passo 1")
                column = np.array(result[rows, index], copy=True)

                zeros = column == 0.0
                if zeros.any():
                    params = step1_params(column, phase="zeros")
                    _recusa_desvio_negativo(params.sd, index, batch, nomes)
                    rng = np.random.default_rng([seed, index, batch_position, _STREAM_STEP1_ZEROS])
                    column[zeros] = rng.normal(params.mean, params.sd, size=int(zeros.sum()))

                missing = np.isnan(column)
                if missing.any():
                    # `phase="na"` sobre a coluna já substituída, que é a ordem do R.
                    params = step1_params(column, phase="na")
                    _recusa_desvio_negativo(params.sd, index, batch, nomes)
                    rng = np.random.default_rng([seed, index, batch_position, _STREAM_STEP1_NA])
                    column[missing] = rng.normal(params.mean, params.sd, size=int(missing.sum()))

                result[rows, index] = column
        return result

    def _correct_feature(
        self,
        after_step1: np.ndarray,
        *,
        before_step1: np.ndarray,
        index: int,
        labels: np.ndarray,
        batch_order: list[object],
        is_qc: np.ndarray,
        is_sample: np.ndarray,
        seed: int,
        hoisted: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> _Outcome:
        """Corrige uma feature com o cache de nomes de parâmetro instalado.

        O cache é instalado aqui porque a troca vale por processo, e este método roda nos
        workers do `loky`: instalado em `fit_transform`, no processo pai, ele não alcançaria
        as florestas. Custa uma instalação e uma restauração por feature.
        """
        with cached_param_names(enabled=self.fast_param_names):
            return self._correct_feature_uncached(
                after_step1,
                before_step1=before_step1,
                index=index,
                labels=labels,
                batch_order=batch_order,
                is_qc=is_qc,
                is_sample=is_sample,
                seed=seed,
                hoisted=hoisted,
            )

    def _predict(
        self,
        *,
        x_train: np.ndarray,
        x_test: np.ndarray,
        usable: np.ndarray,
        y: np.ndarray,
        seed: int,
        index: int,
        batch: object,
        batch_position: int,
        depths: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Passo 4: ajusta a floresta nos QCs e prediz QCs e amostras.

        É método à parte para que um arnês de medição possa trocar a predição pela do R e
        passar pelos passos 5 a 9 reais, em vez de reimplementá-los. `batch` e `batch_position`
        viajam por isso: o despejo do R indexa por rótulo de lote, e a posição deriva a semente.
        """
        semente = _stream_seed(seed, index, batch_position, _STREAM_FOREST)
        forest: RandomForestRegressor | FlorestaRanger
        if self.floresta == "ranger":
            forest = FlorestaRanger(
                n_estimators=RANGER_N_ESTIMATORS,
                max_features=RANGER_MAX_FEATURES,
                min_samples_split=RANGER_MIN_SAMPLES_SPLIT,
                random_state=semente,
            )
        else:
            # O nome é lido do módulo na hora da chamada, e não capturado: os arnêses de
            # medição trocam `serrf.RandomForestRegressor` só no processo deles.
            forest = RandomForestRegressor(
                n_estimators=RANGER_N_ESTIMATORS,
                max_features=RANGER_MAX_FEATURES,
                min_samples_split=RANGER_MIN_SAMPLES_SPLIT,
                random_state=semente,
            )
        forest.fit(x_train[usable].T, y)
        depths.extend(int(tree.get_depth()) for tree in forest.estimators_)
        return (
            np.asarray(forest.predict(x_train[usable].T), dtype=np.float64),
            np.asarray(forest.predict(x_test[usable].T), dtype=np.float64),
        )

    def _correct_feature_uncached(
        self,
        after_step1: np.ndarray,
        *,
        before_step1: np.ndarray,
        index: int,
        labels: np.ndarray,
        batch_order: list[object],
        is_qc: np.ndarray,
        is_sample: np.ndarray,
        seed: int,
        hoisted: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> _Outcome:
        column = after_step1[:, index]
        out = np.array(column, copy=True)
        depths: list[int] = []
        retidos: list[int] = []
        cause: UncorrectedCause | None = None

        # As âncoras globais, depois do passo 1, sobre todos os lotes: o `all[j, …]` do R.
        global_qc = column[is_qc]
        global_sample = column[is_sample]

        for batch_position, batch in enumerate(batch_order):
            in_batch = labels == batch
            qc_here = in_batch & is_qc
            sample_here = in_batch & is_sample
            if not qc_here.any() or not sample_here.any():
                continue

            if hoisted is not None:
                corrs_train, corrs_target = hoisted[batch_position]
            else:
                corrs_train, corrs_target = correlation_matrices(
                    before_step1.T, is_qc=is_qc, in_batch=in_batch, is_target=is_sample
                )
            selected = select_predictors(corrs_train, corrs_target, index + 1, self.n_correlated)
            retidos.append(len(selected))
            if not selected:
                cause = "empty_predictor_intersection"
                continue

            predictors = [s - 1 for s in selected]
            # Passo 4: treino e teste escalonados independentemente, cada um sobre o
            # próprio conjunto, como no R. Um `StandardScaler` ajustado no treino e
            # aplicado ao teste mudaria toda predição.
            x_train = scale_by_columns_of(after_step1[np.ix_(qc_here, predictors)].T)
            x_test = scale_by_columns_of(after_step1[np.ix_(sample_here, predictors)].T)
            usable = usable_predictor_columns(x_train, x_test)
            if not usable.any():
                cause = "no_usable_predictors"
                continue

            local_qc = column[qc_here]
            local_sample = column[sample_here]
            # `y` centrado: subtrai a média, sem dividir pelo desvio.
            y = local_qc - np.mean(local_qc)

            predicted_qc, predicted_sample = self._predict(
                x_train=x_train,
                x_test=x_test,
                usable=usable,
                y=y,
                seed=seed,
                index=index,
                batch=batch,
                batch_position=batch_position,
                depths=depths,
            )

            norm_qc = normalize_qc(local_qc, predicted_qc, global_qc)
            norm_sample = normalize_samples(local_sample, predicted_sample, global_sample)
            norm_qc = rescale_by_median(norm_qc, global_qc)
            norm_sample = rescale_by_median(norm_sample, global_sample)

            # Passos 8 e 9 operam sobre o vetor do lote inteiro, QCs e amostras de estudo
            # juntos, na ordem das colunas do lote. `validate` não entra: como `NaN`, o passo
            # 8 a preencheria com ruído, que entraria nas dobradiças do boxplot do passo 9.
            positions = np.flatnonzero(qc_here | sample_here)
            qc_slot = np.isin(positions, np.flatnonzero(qc_here))
            sample_slot = np.isin(positions, np.flatnonzero(sample_here))
            combined = np.full(positions.size, np.nan)
            combined[qc_slot] = norm_qc
            combined[sample_slot] = norm_sample

            combined = replace_non_finite(
                combined, np.random.default_rng([seed, index, batch_position, _STREAM_STEP8])
            )

            outlier = boxplot_outlier_mask(combined, coef=STEP9_OUTLIER_COEF)
            # O valor do lote sem normalizar vai junto do normalizado: é dele que o passo 9
            # tira as posições marcadas e a reversão dos negativos.
            combined[sample_slot] = subtractive_outlier_repair(
                combined[sample_slot],
                local_sample,
                predicted_sample,
                global_sample,
                outlier[sample_slot],
            )

            out[positions] = combined

        return _Outcome(values=out, cause=cause, tree_depths=depths, n_predictors=retidos)

    @staticmethod
    def _step10(
        corrected: np.ndarray,
        *,
        is_qc: np.ndarray,
        is_sample: np.ndarray,
        seed: int,
        feature_indices: np.ndarray,
    ) -> np.ndarray:
        """Passo 10, blocos de QC e de amostra separadamente, por feature.

        `feature_indices` traz o índice original de cada coluna de `corrected`, e não a
        posição dentro dela. A semente é derivada de `(seed, index, bloco, passo)`; com a
        posição no subconjunto, a execução parcial daria sorteio diferente do da completa.
        """
        result = np.array(corrected, copy=True)
        for column, index in enumerate(int(i) for i in feature_indices):
            for block_position, block in enumerate((is_qc, is_sample)):
                positions = np.flatnonzero(block)
                if positions.size == 0:
                    continue
                rng = np.random.default_rng([seed, index, block_position, _STREAM_STEP10])
                result[positions, column] = repair_assembled_row(result[positions, column], rng)
        return result


def _indices_ate_cancelar(columns: np.ndarray, cancelled: CancelCheck | None) -> Iterator[int]:
    """Os índices das features a despachar, parando quando o cancelamento é pedido.

    É gerador, e não lista, porque o `Parallel` consome o iterável à medida que despacha. Com
    `backend="loky"` e `n_jobs=1` ele calcula cada feature antes de pedir a próxima, e a checagem
    cai entre duas features. Numa lista pronta, todas as checagens aconteceriam antes de a
    primeira feature ser calculada. A exceção levantada aqui propaga pelo `Parallel`.
    """
    for column in columns:
        raise_if_cancelled(cancelled, "serrf: despacho de feature")
        yield int(column)


def _stream_seed(seed: int, index: int, batch_position: int, step: int) -> int:
    """Semente inteira para consumidor que não aceita `Generator`.

    O `RandomForestRegressor` pede `int`. A conversão passa pelo mesmo `SeedSequence` dos
    outros passos, em vez de somar os componentes: na soma, `(seed=1, index=2)` e
    `(seed=2, index=1)` dariam a mesma floresta.
    """
    return int(np.random.default_rng([seed, index, batch_position, step]).integers(2**31))


def _ordered_unique(labels: np.ndarray) -> list[object]:
    """Rótulos de lote na ordem de primeira aparição.

    Ordem de aparição e não `sorted`: o lote é rótulo de domínio, pode não ser
    ordenável, e a ordem decide o fluxo derivado de cada `(feature, lote)`.
    """
    seen: list[object] = []
    for label in labels.tolist():
        if label not in seen:
            seen.append(label)
    return seen

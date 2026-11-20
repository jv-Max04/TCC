"""Corretor `batch_loess`: porte da função de mesmo nome do `tools.R`, por feature e por lote.

- Passo 1: descarta os QCs aberrantes antes do ajuste, com `coef = 1.5` (linha 914).
- Passo 2: ajusta a curva contra a ordem de injeção, com `span` eleito por validação cruzada
  (`_loess_curve`).
- Passo 3: avalia a curva em todas as amostras do lote; onde a predição é indefinida, usa a do
  tempo mais próximo que tenha predição (linhas 923 a 929), e não o valor da borda.
- Passo 4: se mais de um quinto da curva é negativo, devolve a feature sem correção, marcada;
  senão, cada posição negativa recebe um sorteio numa banda em torno da mediana (linhas 933 a
  937).
- Passo 5: normaliza por `e[j] / (curva / median(e[j]))`. No modo padrão, denominador nulo
  produz `Inf` ou `NaN`, como no R.
- Passo 6: restaura zeros e ausentes (`base.py`).

O objetivo é a curva do R, e não uma curva melhor. Onde o original é frágil (sem reponderação
robusta, com divisão que produz infinito), o porte é frágil do mesmo jeito, e os testes afirmam
isso. Cada feature tem gerador próprio, semeado com `seed + índice`, e nada mais é partilhado
entre duas features; por isso a saída em vários núcleos é a da série, bit a bit.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed

from refino.correction._loess_curve import predict_curve, select_span
from refino.correction.base import (
    feature_columns,
    resolve_target_features,
    restore_zeros_and_missing,
    select_features,
)
from refino.domain.cancellation import CancelCheck, CancellableDispatch, raise_if_cancelled
from refino.domain.correction import CorrectionResult
from refino.domain.labels import NullDenominatorPolicy, UncorrectedCause, validate_label
from refino.domain.matrix import RefinoMatrix
from refino.domain.pool import MORTE_DO_POOL, vez_no_pool
from refino.metrics.crossval import DegenerateCrossValidationError, held_out_matrices
from refino.metrics.rsd import boxplot_outlier_mask

_LOGGER = logging.getLogger(__name__)

__all__ = ["CurveSensitivity", "FittedCurve", "LoessCorrector", "curve_sensitivity"]

# `coef = 1.5` é o do `remove_outlier()`/`RSD()` do `tools.R`; o passo 9 do SERRF usa 3.
OUTLIER_COEF = 1.5

# Linha 933 do `tools.R`: `sum(line < 0) > length(line) / 5`. Estritamente maior.
NEGATIVE_FRACTION_LIMIT = 0.2

# A banda do sorteio da linha 936, em unidades de desvio da própria feature.
NEGATIVE_BAND_SD_FACTOR = 0.1

MIN_QC_FOR_FIT = 3


@dataclass(frozen=True)
class _FeatureOutcome:
    """O que aconteceu com uma feature: valores e, se houver, a causa."""

    values: np.ndarray
    cause: UncorrectedCause | None


@dataclass(frozen=True)
class _BatchFit:
    """Os passos 1 a 3 num lote: a curva nas posições dele, o que o passo 3 preencheu, e os QCs
    do lote que entraram no ajuste."""

    curve: np.ndarray
    filled: np.ndarray
    used: np.ndarray
    span: float


@dataclass(frozen=True)
class _BatchFailure:
    """O lote em que o ajuste não aconteceu, com a causa e o porquê em português."""

    cause: UncorrectedCause
    reason: str


@dataclass(frozen=True)
class FittedCurve:
    """A curva com que o LOESS corrige uma feature, alinhada às amostras da matriz.

    Os quatro vetores têm uma posição por amostra, na ordem das linhas da matriz, que é a
    ordem de injeção:

    - `curve`: o divisor da correção antes da âncora de mediana; `nan` onde não há curva.
    - `filled`: o passo 3 pôs ali a predição do tempo mais próximo; a curva não foi ajustada ali.
    - `drawn`: o passo 4 sorteou ali, porque a curva ajustada era negativa.
    - `used_qc`: o QC entrou no ajuste; nenhum entra quando não há curva.

    `reason` diz o porquê quando `cause` existe, e é `""` quando há curva. `spans` traz o `span`
    eleito em cada lote, na ordem de primeira aparição, e é vazio sem curva; ele decide quão de
    perto a curva segue os QCs que a ajustaram.
    """

    feature: str
    curve: np.ndarray
    filled: np.ndarray
    drawn: np.ndarray
    used_qc: np.ndarray
    cause: UncorrectedCause | None
    reason: str
    spans: tuple[float, ...] = ()


@dataclass(frozen=True)
class CurveSensitivity:
    """A curva que corrigiu e as curvas refeitas sem cada QC.

    O envelope (`lower` e `upper`) é o mínimo e o máximo, posição a posição, das curvas sem um QC
    que existem. É sensibilidade à remoção de um QC, e não intervalo de confiança: não carrega
    probabilidade nenhuma.

    `held_out` traz uma curva por QC retido, na ordem de `held_out_rows`; a predição de cada QC pela
    curva que não o viu é `held_out[i].curve[held_out_rows[i]]`. `not_fitted` marca as posições em
    que alguma curva do envelope não foi ajustada: foi preenchida pelo tempo mais próximo, o caso
    das curvas sem o primeiro ou sem o último QC, ou sorteada no passo 4. `degenerate` traz a
    recusa da guarda de `leave_one_qc_out_rsd`, quando ela recusa, e então não há curva sem um QC.
    """

    fitted: FittedCurve
    held_out_rows: tuple[int, ...]
    held_out: tuple[FittedCurve, ...]
    lower: np.ndarray
    upper: np.ndarray
    not_fitted: np.ndarray
    degenerate: str | None

    @property
    def n_curves(self) -> int:
        """Quantas curvas sem um QC existem, isto é, as que o envelope cobre."""
        return sum(1 for curva in self.held_out if curva.cause is None)


class LoessCorrector:
    """`batch_loess` por feature e por lote, com marcação das features não corrigidas.

    `null_denominator` tem padrão `r_faithful` porque a saída de referência do R tem `Inf`
    e `NaN` onde o denominador é nulo, e os testes golden comparam contra ela.
    """

    name = "loess"

    # Quantos QCs de treino por lote o ajuste exige. O passo 1 recusa o lote com menos QCs
    # utilizáveis e devolve a feature sem correção, marcada `fit_failed`. `metrics.crossval` lê
    # este atributo para recusar, com motivo, a medida fora da amostra que deixaria um lote
    # abaixo disso ao reter QCs: sem a recusa, ela misturaria dado cru com corrigido. Corretor
    # que não o declara fica com o mínimo de `metrics.crossval`.
    min_train_qc_per_batch = MIN_QC_FOR_FIT

    def __init__(
        self,
        *,
        null_denominator: NullDenominatorPolicy = "r_faithful",
        n_jobs: int = 1,
    ) -> None:
        validate_label("NullDenominatorPolicy", null_denominator)
        self.null_denominator = null_denominator
        # Padrão 1: em matriz de poucas features, como as dos testes, o arranque dos processos
        # custa mais que o trabalho. Quem processa matriz real pede vários núcleos. O
        # vocabulário é o do `joblib`: `1` é série, `-1` são todos os núcleos.
        self.n_jobs = n_jobs

    def identidade(self) -> str:
        """Identidade de construtor: só `null_denominator` muda valores.

        Ele decide o que sai quando o denominador ajustado é zero, e as duas políticas divergem:
        sob `r_faithful` o `batch_loess` produz `Inf` nos quatro modos reais.

        `n_jobs` fica de fora porque muda o tempo e não o valor: a matriz de dois núcleos é a da
        série, bit a bit, como afirma
        `test_dois_nucleos_dao_a_matriz_da_serie_bit_a_bit_num_modo_real`.
        """
        return f"{self.name}(null_denominator={self.null_denominator})"

    # -----------------------------------------------------------------------

    def fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        cancelled: CancelCheck | None = None,
    ) -> CorrectionResult:
        """Corrige feature por feature, lote por lote, e marca o que não corrigiu.

        O conjunto de features da saída é o da entrada: nada é descartado. Feature que não
        pôde ser corrigida volta idêntica à entrada e marcada com a causa. O `tools.R`
        também a devolve sem correção, mas não marca.

        Com `n_jobs=1` a bandeira `cancelled` é lida antes de cada feature. Com `n_jobs != 1`
        as features correm em processos do `loky`, que a bandeira do pai não alcança, e a
        checagem fica no despacho (ver `_em_paralelo`): a parada espera os lotes já entregues.
        """
        values = np.asarray(matrix.values, dtype=np.float64)
        labels = self._batch_labels(matrix, batches)
        is_qc = matrix.qc_mask
        times = matrix.injection_orders.astype(np.float64)

        # Execução parcial é exata porque `_correct_feature` lê uma coluna e nada mais. O
        # índice da semente é o da coluna na matriz original, senão a mesma feature
        # receberia sorteio diferente entre execução completa e parcial.
        requested = resolve_target_features(matrix, target_features)
        columns = [int(c) for c in feature_columns(matrix, requested)]

        outcomes = self._ajusta(
            values, columns, labels=labels, is_qc=is_qc, times=times, seed=seed, cancelled=cancelled
        )

        corrected = np.array(values[:, columns], copy=True)
        causes: dict[UncorrectedCause, set[str]] = defaultdict(set)
        for column, (feature, outcome) in enumerate(zip(requested, outcomes, strict=True)):
            corrected[:, column] = outcome.values
            if outcome.cause is not None:
                causes[outcome.cause].add(feature)

        restoration = restore_zeros_and_missing(values[:, columns], corrected)
        note = (
            f"correction:loess(null_denominator={self.null_denominator},seed={seed},"
            f"zeros_restored={restoration.restored},zeros_created={restoration.created},"
            f"uncorrected={sum(len(f) for f in causes.values())})"
        )
        return CorrectionResult(
            matrix=select_features(matrix, requested).with_values(restoration.values, note),
            uncorrected={cause: frozenset(features) for cause, features in causes.items()},
            requested=requested,
        )

    def fitted_curve(
        self,
        matrix: RefinoMatrix,
        feature: str,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
    ) -> FittedCurve:
        """A curva pela qual `fit_transform` dividiria `feature`, com os mesmos argumentos.

        É a curva da correção, e não uma reconstrução dela: a semente é `seed` mais o índice
        original da coluna, e o caminho é `_trace_feature`, o mesmo que `_correct_feature` corre
        antes de normalizar. Feature que a matriz não tem levanta `ValueError`.
        """
        requested = resolve_target_features(matrix, frozenset({feature}))
        index = int(feature_columns(matrix, requested)[0])
        values = np.asarray(matrix.values, dtype=np.float64)
        return self._trace_feature(
            values[:, index],
            feature=feature,
            labels=self._batch_labels(matrix, batches),
            is_qc=matrix.qc_mask,
            times=matrix.injection_orders.astype(np.float64),
            seed=seed + index,
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

    def _ajusta(
        self,
        values: np.ndarray,
        columns: list[int],
        *,
        labels: np.ndarray,
        is_qc: np.ndarray,
        times: np.ndarray,
        seed: int,
        cancelled: CancelCheck | None,
    ) -> list[_FeatureOutcome]:
        """Um desfecho por feature, na ordem de `columns`, em série ou em vários núcleos.

        Se o pool de workers morre no meio (falta de memória, o sistema matando processos), o
        ajuste é refeito em série, uma vez. O resultado é o mesmo e só o tempo muda; se a série
        também falhar, o erro sobe.
        """
        if self.n_jobs == 1:
            return self._em_serie(
                values,
                columns,
                labels=labels,
                is_qc=is_qc,
                times=times,
                seed=seed,
                cancelled=cancelled,
            )
        try:
            return self._em_paralelo(
                values,
                columns,
                labels=labels,
                is_qc=is_qc,
                times=times,
                seed=seed,
                cancelled=cancelled,
            )
        except MORTE_DO_POOL:
            _LOGGER.warning(
                "o pool de workers morreu com n_jobs=%s; refazendo o LOESS em SERIE. "
                "O resultado e identico — o que muda e o tempo",
                self.n_jobs,
            )
            return self._em_serie(
                values,
                columns,
                labels=labels,
                is_qc=is_qc,
                times=times,
                seed=seed,
                cancelled=cancelled,
            )

    def _em_serie(
        self,
        values: np.ndarray,
        columns: list[int],
        *,
        labels: np.ndarray,
        is_qc: np.ndarray,
        times: np.ndarray,
        seed: int,
        cancelled: CancelCheck | None,
    ) -> list[_FeatureOutcome]:
        """Laço no processo do chamador, com a bandeira lida antes de cada feature."""
        outcomes: list[_FeatureOutcome] = []
        for index in columns:
            raise_if_cancelled(cancelled, "loess: laco por feature")
            outcomes.append(
                self._correct_feature(
                    values[:, index],
                    labels=labels,
                    is_qc=is_qc,
                    times=times,
                    # Semente por feature, `seed + índice`. Uma semente global consumida
                    # na ordem em que o escalonador entrega as features faria a saída
                    # depender do paralelismo.
                    seed=seed + index,
                )
            )
        return outcomes

    def _em_paralelo(
        self,
        values: np.ndarray,
        columns: list[int],
        *,
        labels: np.ndarray,
        is_qc: np.ndarray,
        times: np.ndarray,
        seed: int,
        cancelled: CancelCheck | None,
    ) -> list[_FeatureOutcome]:
        """As mesmas chamadas de `_em_serie`, em processos do `loky`, de volta na ordem da entrada.

        Cada tarefa leva uma coluna e a semente `seed + índice`, e `_correct_feature` não lê nada
        além disso. `Parallel` devolve na ordem da entrada, então a montagem em `fit_transform`
        é a mesma. O backend `loky` é explícito porque é o que `desktop/paralelismo.py` ajusta
        no executável congelado.

        Na parada, o despacho deixa de produzir em vez de levantar: levantar de dentro do gerador
        faria o `loky` encerrar o executor partilhado do processo, e com ele o trabalho de quem
        estivesse ao lado. A exceção sobe depois de o `Parallel` devolver (ver
        `CancellableDispatch`). O `Parallel` corre dentro da vez no pool (`vez_no_pool`, em
        `domain/pool.py`), que espera enquanto outro trabalho usa o executor com outro `n_jobs`.
        """
        despacho = CancellableDispatch(cancelled, "loess: despacho de feature")
        with vez_no_pool(self.n_jobs, cancelled, "loess: espera da vez no pool"):
            saida = list(
                Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed(self._correct_feature)(
                        values[:, index],
                        labels=labels,
                        is_qc=is_qc,
                        times=times,
                        seed=seed + index,
                    )
                    for index in despacho.over(columns)
                )
            )
        despacho.raise_if_stopped()
        return saida

    def _correct_feature(
        self,
        column: np.ndarray,
        *,
        labels: np.ndarray,
        is_qc: np.ndarray,
        times: np.ndarray,
        seed: int,
    ) -> _FeatureOutcome:
        """Corrige uma feature; a primeira causa encontrada em qualquer lote a devolve sem correção.

        A feature é a unidade de decisão, e não o lote, como no R, onde o `tryCatch` é por
        feature (linhas 910 e 953 do `tools.R`). Corrigir um lote e não o outro deixaria a
        feature mista.
        """
        trace = self._trace_feature(column, labels=labels, is_qc=is_qc, times=times, seed=seed)
        if trace.cause is not None:
            return _FeatureOutcome(values=np.array(column, copy=True), cause=trace.cause)
        return self._normalize(column, trace.curve)

    def _trace_feature(
        self,
        column: np.ndarray,
        *,
        labels: np.ndarray,
        is_qc: np.ndarray,
        times: np.ndarray,
        seed: int,
        feature: str = "",
    ) -> FittedCurve:
        """Os passos 1 a 4 da feature inteira: a curva por que o passo 5 divide, ou a causa.

        Há um gerador por feature. Dele saem, nesta ordem, a semente da seleção de `span` de
        cada lote e o sorteio do passo 4; mudar a ordem muda a correção.
        """
        curve = np.full(column.shape, np.nan)
        filled = np.zeros(column.shape, dtype=bool)
        drawn = np.zeros(column.shape, dtype=bool)
        used_qc = np.zeros(column.shape, dtype=bool)
        spans: list[float] = []
        rng = np.random.default_rng(seed)

        def sem_curva(cause: UncorrectedCause, reason: str) -> FittedCurve:
            # Sem curva não há ajuste: `used_qc`, `filled` e `drawn` voltam vazios.
            vazio = np.zeros(column.shape, dtype=bool)
            return FittedCurve(
                feature=feature,
                curve=np.full(column.shape, np.nan),
                filled=vazio,
                drawn=vazio.copy(),
                used_qc=vazio.copy(),
                cause=cause,
                reason=reason,
            )

        for label in _ordered_unique(labels):
            in_batch = labels == label
            fit = self._fit_batch(column, in_batch, is_qc, times, rng, label=label)
            if isinstance(fit, _BatchFailure):
                return sem_curva(fit.cause, fit.reason)
            curve[in_batch] = fit.curve
            filled[in_batch] = fit.filled
            used_qc[in_batch & is_qc] = fit.used
            spans.append(fit.span)

        # Passos 4 e 5 sobre a linha inteira, e não por lote. No `tools.R` o laço por lote fecha
        # na linha 931, e o corte de um quinto de negativos (933), a banda do sorteio (936) e a
        # normalização (946) vêm depois dele, sobre `line` e `e[i,]` inteiros. Com um lote só as
        # duas leituras coincidem. Com dois, normalizar pela mediana de cada lote devolveria à
        # saída o degrau entre eles.
        negative = curve < 0.0
        n_negative = int(np.sum(negative))
        if n_negative > NEGATIVE_FRACTION_LIMIT * curve.size:
            return sem_curva(
                "negative_fitted_curve",
                f"{n_negative} das {curve.size} posições da curva ajustada são negativas — mais de "
                "um quinto —, e a correção devolve a feature sem correção (linha 933 do tools.R)",
            )
        if negative.any():
            curve = _replace_negatives(curve, negative, column, rng)
            drawn = negative
        return FittedCurve(
            feature=feature,
            curve=curve,
            filled=filled,
            drawn=drawn,
            used_qc=used_qc,
            cause=None,
            reason="",
            spans=tuple(spans),
        )

    def _fit_batch(
        self,
        column: np.ndarray,
        in_batch: np.ndarray,
        is_qc: np.ndarray,
        times: np.ndarray,
        rng: np.random.Generator,
        *,
        label: object = "",
    ) -> _BatchFit | _BatchFailure:
        """Passos 1 a 3 num lote. Devolve a curva e o que a montou, ou a causa que a impediu."""
        qc_in_batch = in_batch & is_qc
        qc_values = column[qc_in_batch]
        qc_times = times[qc_in_batch]

        # Passo 1: descarta os QCs aberrantes antes do ajuste (linha 914). Com `n <= 2` a
        # máscara não marca nada, porque as dobradiças são os próprios extremos.
        keep = ~boxplot_outlier_mask(qc_values, coef=OUTLIER_COEF) & np.isfinite(qc_values)
        if int(np.sum(keep)) < MIN_QC_FOR_FIT:
            return _BatchFailure(
                cause="fit_failed",
                reason=(
                    f"o lote {label} tem {qc_values.size} QC(s), {int(np.sum(keep))} "
                    "utilizável(is) depois do descarte de aberrantes, e o ajuste quadrático exige "
                    f"{MIN_QC_FOR_FIT}"
                ),
            )

        # Passo 2: ajuste com `span` por validação cruzada.
        try:
            selection = select_span(qc_times[keep], qc_values[keep], seed=int(rng.integers(2**31)))
            fitted = predict_curve(qc_times[keep], qc_values[keep], times[in_batch], selection.span)
        except (ValueError, FloatingPointError) as erro:
            return _BatchFailure(
                cause="fit_failed", reason=f"o ajuste do lote {label} falhou: {erro}"
            )

        # Passo 3: preenche o indefinido com a predição do tempo mais próximo que tenha
        # predição (linhas 923 a 929), e não com o valor da borda. Acontece nas amostras
        # do lote que ficam fora do intervalo dos QCs usados no ajuste.
        curve = _fill_by_nearest_time(fitted, times[in_batch])
        if not np.all(np.isfinite(curve)):
            return _BatchFailure(
                cause="fit_failed",
                reason=f"a curva do lote {label} não ficou finita em todas as amostras dele",
            )
        # O passo 4 não fica aqui: ele é da linha inteira, depois de todos os lotes.
        return _BatchFit(curve=curve, filled=~np.isfinite(fitted), used=keep, span=selection.span)

    def _normalize(self, column: np.ndarray, curve: np.ndarray) -> _FeatureOutcome:
        """Passo 5: `e[j] / (curva / median(e[j]))`, na forma do R.

        A mediana é a da feature em todas as amostras, e não a do lote: no `tools.R` a linha 946
        está fora do laço por lote, que fecha na 931. A curva é local, de cada lote, e a âncora
        de escala é global; é o que deixa os lotes comparáveis depois da correção.
        """
        finite = column[np.isfinite(column)]
        median = float(np.median(finite)) if finite.size else float("nan")
        # Há dois casos degenerados na mesma divisão, com efeitos opostos:
        #
        #   curva == 0   -> divisor == 0   -> valor / 0   = Inf
        #   mediana == 0 -> divisor == Inf -> valor / Inf = 0
        #
        # O primeiro é o denominador nulo, que o modo `uncorrected_feature` marca. O
        # segundo é como o `batch_loess` do R cria zeros. Ele é reproduzido de propósito
        # e não é causa marcada: o método produziu zero, e não deixou de corrigir.
        #
        # `errstate` nas duas divisões porque os dois casos são esperados.
        with np.errstate(divide="ignore", invalid="ignore"):
            divisor = curve / median
            if self.null_denominator == "uncorrected_feature" and np.any(divisor == 0.0):
                return _FeatureOutcome(values=np.array(column, copy=True), cause="null_denominator")
            out = column / divisor
        return _FeatureOutcome(values=out, cause=None)


# ---------------------------------------------------------------------------
# A curva sem cada QC
# ---------------------------------------------------------------------------


def curve_sensitivity(matrix: RefinoMatrix, feature: str, *, seed: int = 42) -> CurveSensitivity:
    """A curva que corrigiu `feature` e as curvas refeitas sem cada QC.

    As curvas sem um QC saem dos passos de `leave_one_qc_out_rsd` (a mesma guarda, os mesmos QCs
    retidos, a mesma remarcação), com o mesmo corretor e a mesma semente. Assim o valor fora da
    amostra de cada QC é o dele dividido pela predição da curva que não o viu, com a âncora de
    mediana do passo 5, que é o número que a métrica agrega.

    A partição de lote é a de `matrix.samples`, o campo que a régua lê. `null_denominator` não
    entra: ele decide o que a divisão do passo 5 faz com um zero, e a curva é anterior à divisão.
    """
    loess = LoessCorrector()
    fitted = loess.fitted_curve(matrix, feature, seed=seed)
    try:
        retidos = held_out_matrices(matrix)
    except DegenerateCrossValidationError as recusa:
        # A régua recusa este caso (num lote de dois QCs, reter um deixa um só no treino), e
        # a função devolve a recusa em vez de curvas que a régua não mediria.
        return CurveSensitivity(
            fitted=fitted,
            held_out_rows=(),
            held_out=(),
            lower=np.full(matrix.n_samples, np.nan),
            upper=np.full(matrix.n_samples, np.nan),
            not_fitted=np.zeros(matrix.n_samples, dtype=bool),
            degenerate=str(recusa),
        )

    held_out = tuple(loess.fitted_curve(remarcada, feature, seed=seed) for _, remarcada in retidos)
    existentes = [curva for curva in held_out if curva.cause is None]
    if existentes:
        pilha = np.vstack([curva.curve for curva in existentes])
        lower = np.min(pilha, axis=0)
        upper = np.max(pilha, axis=0)
        not_fitted = np.any(np.vstack([c.filled | c.drawn for c in existentes]), axis=0)
    else:
        lower = np.full(matrix.n_samples, np.nan)
        upper = np.full(matrix.n_samples, np.nan)
        not_fitted = np.zeros(matrix.n_samples, dtype=bool)
    return CurveSensitivity(
        fitted=fitted,
        held_out_rows=tuple(row for row, _ in retidos),
        held_out=held_out,
        lower=lower,
        upper=upper,
        not_fitted=not_fitted,
        degenerate=None,
    )


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------


def _ordered_unique(labels: np.ndarray) -> list[object]:
    """Rótulos distintos na ordem de primeira aparição.

    Não são ordenados: a ordem dos lotes decide o consumo do gerador, e ordenar mudaria o
    sorteio do passo 4.
    """
    seen: list[object] = []
    for label in labels.tolist():
        if label not in seen:
            seen.append(label)
    return seen


def _fill_by_nearest_time(fitted: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Preenche posições indefinidas com a predição do tempo mais próximo.

    É o que o `tools.R` faz nas linhas 923 a 929. Difere de repetir o valor da borda quando as
    posições indefinidas não são as extremas.
    """
    out = np.array(fitted, dtype=np.float64, copy=True)
    known = np.isfinite(out)
    if not known.any() or known.all():
        return out
    known_times = times[known]
    known_values = out[known]
    for index in np.flatnonzero(~known):
        nearest = int(np.argmin(np.abs(known_times - times[index])))
        out[index] = known_values[nearest]
    return out


def _replace_negatives(
    curve: np.ndarray,
    negative: np.ndarray,
    observed: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sorteia a curva nas posições negativas, na banda da linha 936 do `tools.R`.

    `U(max(median - 0,1·sd, 0), max(median + 0,1·sd, 1))`, com a mediana e o desvio da feature
    observada na linha inteira, porque a linha 936 está fora do laço por lote. Os dois `max` são
    os do R e não são simétricos: o inferior é contra zero e o superior contra um.
    """
    finite = observed[np.isfinite(observed)]
    if finite.size == 0:  # pragma: no cover
        # Não chega aqui: sem três QCs finitos o ajuste falha antes. A guarda impede que a
        # mediana de vetor vazio vire `nan` dentro do sorteio.
        return curve
    median = float(np.median(finite))
    deviation = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    low = max(median - NEGATIVE_BAND_SD_FACTOR * deviation, 0.0)
    high = max(median + NEGATIVE_BAND_SD_FACTOR * deviation, 1.0)
    out = np.array(curve, copy=True)
    out[negative] = rng.uniform(low, high, size=int(np.sum(negative)))
    return out

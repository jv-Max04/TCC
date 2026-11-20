"""Estimadores de RSD fora da amostra.

São dois, e a diferença é deliberada. `leave_one_qc_out_rsd` é o estimador deste
projeto: mantém a matriz inteira, remarca o QC retido como `sample` e mede o
corretor como ele roda em produção. `qc_rsd_r_cv` é de compatibilidade: reajusta
sobre a submatriz de QC e mais nada, porque é o que o `tools.R` faz, e existe só
para reproduzir as colunas do `QC - RSD.csv`.

A remarcação é o que dá acesso ao valor fora da amostra através de um `Protocol`
que só expõe `fit_transform`, e é a mesma convenção do R (`tools.R:1267`). Por
causa dela o QC retido recebe a fórmula do passo 6 do SERRF, com âncora de
mediana, e não a do passo 5.

O `Protocol` é local porque `metrics/` não depende de `correction/`; o tipo de
retorno mora em `domain`, que as duas camadas importam. As funções levantam antes
de ajustar quando a configuração é impossível, com a causa nomeada.

Com `n_jobs != 1` os ajustes de `leave_one_qc_out_rsd` correm em processos do
`loky` e voltam na ordem dos QCs, com a mesma saída da execução em série. Só um
nível de paralelismo por vez: com a régua em vários núcleos o corretor tem de vir
com um só, senão cada worker abriria o seu pool.
"""

import logging
import math
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Final, Protocol

import numpy as np
from joblib import Parallel, delayed
from scipy import stats

from refino.domain.cancellation import CancelCheck, CancellableDispatch, raise_if_cancelled
from refino.domain.correction import CorrectionResult
from refino.domain.labels import RsdVariant, validate_label
from refino.domain.matrix import RefinoMatrix
from refino.domain.pool import MORTE_DO_POOL, vez_no_pool
from refino.metrics.rsd import rsd

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "BATCH_CONSTRAINED_CORRECTORS",
    "DRAW_SEED_GAP",
    "JACKKNIFE_LEVEL",
    "JACKKNIFE_METHOD",
    "MAX_DRAW_ATTEMPTS",
    "MIN_QC_PER_BATCH_FOR_JACKKNIFE",
    "MIN_TRAIN_QC_PER_BATCH",
    "BatchQcShortfallError",
    "DegenerateCrossValidationError",
    "JackknifeInterval",
    "SupportsCorrection",
    "draw_seed",
    "held_out_matrices",
    "held_out_pairs",
    "jackknife_cost_ratio",
    "jackknife_qc_minimum",
    "leave_one_qc_out_rsd",
    "leave_one_qc_out_rsd_with_jackknife",
    "out_of_sample_qc_minimum",
    "pair_out_of_sample_values",
    "qc_rsd_r_cv",
    "remark_pair_as_sample",
]

# Com menos de dois QCs de treino num lote o ajuste seria sobre um ponto. O
# estimador levanta em vez de devolver número.
MIN_TRAIN_QC_PER_BATCH = 2

# No `tools.R` só o caminho do SERRF exige todo lote representado no retido: o
# laço da linha 1263 resorteia enquanto algum lote faltar. O caminho do
# `batch_loess` sorteia cinco vezes sem condição (linhas 893-903). Com lote único
# a diferença não aparece, e por isso nenhum arquivo de referência a distingue.
#
# É a lista dos caminhos do R que têm o laço. Ela se atualiza quando outro método
# ganhar o estimador de compatibilidade.
BATCH_CONSTRAINED_CORRECTORS = frozenset({"serrf"})

# Teto de tentativas do resorteio. Só alcança os casos satisfazíveis, porque os
# impossíveis levantam antes. Existe porque o laço do R não tem teto e poderia
# não terminar. Com 5 QCs num lote e 2 no outro a probabilidade de um sorteio
# válido é 10/21, cerca de 2,1 tentativas esperadas, então 64 é folga larga.
MAX_DRAW_ATTEMPTS = 64

# Quantas features o teste de compatibilidade com o R amostra, para que os 800
# ajustes dele caibam no orçamento sem mudar o desenho estatístico (o número de
# sorteios e de sementes). O mesmo `n` vale para os quatro modos. É calibrado nos
# quatro arquivos de referência: o erro-padrão da mediana, por bootstrap, fica
# entre 1,62% e 6,45% conforme o modo. Um modo novo recebe a mesma regra e tem o
# seu erro-padrão medido e reportado.
SAMPLED_FEATURE_COUNT = 500

# Espaçamento entre as sementes dos sorteios de um mesmo `qc_rsd_r_cv`. Cada
# sorteio passa ao corretor uma semente própria: com a mesma em todos, só a
# partição variaria, e a média seria sobre replicatas correlacionadas.
#
# As faixas `[s, s + n_features)` de dois sorteios não podem se tocar, porque o
# LOESS usa `seed + index` por feature. O maior `n_features` dos quatro modos é
# 6.035, então 10.000 deixa folga. O espaçamento entre as sementes do
# pré-registro, 50.000, tem de comportar `n_draws` faixas destas, e o gerador
# confere as duas constantes juntas.
DRAW_SEED_GAP = 10_000


def draw_seed(seed: int, draw: int) -> int:
    """Semente do corretor no sorteio `draw`, com faixa disjunta das demais.

    Aritmética explícita, e não `SeedSequence`: assim a disjunção das faixas do
    LOESS é uma desigualdade sobre inteiros, que se confere. Com `SeedSequence`
    ela seria só provável.
    """
    return seed + draw * DRAW_SEED_GAP


class DegenerateCrossValidationError(ValueError):
    """Configuração em que a validação cruzada não é avaliável.

    Classe própria para que quem chama distinga 'esta configuração não admite o
    estimador', que é resultado a reportar, de outro erro de valor, que é defeito.
    """


class BatchQcShortfallError(DegenerateCrossValidationError):
    """O corretor não mede com esta partição: o lote menor não lhe deixa QCs de treino.

    É subclasse de `DegenerateCrossValidationError`, então quem já apanha a recusa
    da régua continua a apanhá-la. A diferença é o sujeito: `_guard_train_size`
    recusa a matriz para todo corretor, e esta recusa um corretor e deixa os
    outros medir. Quem elege entre vários (`decision.labeling.LabelingOracle`)
    põe o corretor fora da eleição e mostra o motivo no lugar do número.

    `reason` é a frase curta, que vai ao rodapé da figura e à justificativa do
    Palpite. `corrector`, `minimum` e `short` são os números que a compõem.
    """

    def __init__(self, *, corrector: str, minimum: int, short: dict[str, int], reason: str) -> None:
        super().__init__(f"o corretor `{corrector}` não mede fora da amostra: {reason}")
        self.corrector = corrector
        self.minimum = minimum
        self.short = dict(short)
        self.reason = reason


class SupportsCorrection(Protocol):
    """Protocolo local, satisfeito estruturalmente por qualquer `Corrector`.

    `target_features` restringe quais features recebem correção sem fatiar a
    matriz, porque o pool de preditores do passo 3 do SERRF é global.
    """

    name: str

    def fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        cancelled: CancelCheck | None = None,
    ) -> CorrectionResult: ...


# ---------------------------------------------------------------------------
# Os passos do laço, como funções nomeadas
# ---------------------------------------------------------------------------
#
# Os testes plantam defeitos nestes passos (vazamento, índice trocado, iteração a
# menos) e conferem que `leave_one_qc_out_rsd` os acusa. Com os passos embutidos
# no laço, o teste teria de reimplementar o laço para plantar o defeito, e
# provaria algo sobre a reimplementação.


def sampled_feature_names(
    matrix: RefinoMatrix, n_sampled: int | None, seed: int
) -> tuple[str, ...] | None:
    """Nomes das features amostradas, ou `None` quando não há o que amostrar.

    Devolve `None`, e não a tupla completa, quando `n_sampled` cobre a matriz:
    assim o pedido chega ao corretor como execução completa.

    O sorteio deriva de `seed`, e o resultado volta na ordem das colunas da matriz.
    """
    if n_sampled is None or n_sampled >= matrix.n_features:
        return None
    if n_sampled < 1:
        raise ValueError(f"`n_sampled_features` tem de ser >= 1; recebi {n_sampled}")
    chosen = np.random.default_rng([seed, matrix.n_features]).choice(
        matrix.n_features, size=n_sampled, replace=False
    )
    keep = set(int(i) for i in chosen)
    return tuple(name for index, name in enumerate(matrix.feature_names) if index in keep)


def _scatter(
    values: np.ndarray, requested: tuple[str, ...] | None, matrix: RefinoMatrix
) -> np.ndarray:
    """Espalha o RSD das features pedidas num vetor de comprimento `n_features`.

    As não amostradas ficam `nan`: o retorno continua sendo um valor por feature
    da matriz, e toda agregação de RSD já declara o que faz com `nan`. Um vetor
    curto obrigaria cada consumidor a saber qual subconjunto foi pedido.
    """
    if requested is None:
        return values
    full = np.full(matrix.n_features, np.nan)
    position = {name: index for index, name in enumerate(matrix.feature_names)}
    full[[position[name] for name in requested]] = values
    return full


def held_out_rows(matrix: RefinoMatrix) -> np.ndarray:
    """Índices de linha dos QCs, cada um retido exatamente uma vez."""
    return np.flatnonzero(matrix.qc_mask)


def remark_as_sample(matrix: RefinoMatrix, row: int) -> RefinoMatrix:
    """Nova matriz com a amostra `row` reetiquetada como `sample`.

    O `__post_init__` revalida a matriz nova: as máscaras continuam cobrindo
    todas as linhas, porque o `qc` virou `sample` e não desapareceu.
    """
    samples = list(matrix.samples)
    samples[row] = replace(samples[row], sample_type="sample")
    remarked = replace(matrix, samples=tuple(samples))
    return remarked.with_provenance(f"crossval:held_out(row={row},name={matrix.samples[row].name})")


# ---------------------------------------------------------------------------
# O estimador deste projeto
# ---------------------------------------------------------------------------


def _batch_qc_counts(matrix: RefinoMatrix) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample, is_qc in zip(matrix.samples, matrix.qc_mask, strict=True):
        if is_qc:
            counts[sample.batch] = counts.get(sample.batch, 0) + 1
    return counts


def _guard_train_size(matrix: RefinoMatrix) -> None:
    """Levanta antes de qualquer ajuste, nomeando os lotes ofensores.

    Reter um QC de um lote com `k` QCs deixa `k - 1` no treino, então a condição é
    `k - 1 >= 2`. Lote sem QC não entra: retirar um QC não o afeta, e o corretor
    já o salta.
    """
    counts = _batch_qc_counts(matrix)
    offenders = {
        batch: count for batch, count in counts.items() if count - 1 < MIN_TRAIN_QC_PER_BATCH
    }
    if offenders:
        detail = ", ".join(f"{batch}={count} QCs" for batch, count in sorted(offenders.items()))
        raise DegenerateCrossValidationError(
            f"leave_one_qc_out exige ao menos {MIN_TRAIN_QC_PER_BATCH + 1} QCs por lote; "
            f"reter um deixaria menos de {MIN_TRAIN_QC_PER_BATCH} no treino em: {detail}. "
            "Levantar é o contrato: número sobre treino de um ponto pareceria medição."
        )


def _guard_corrector_minimum(matrix: RefinoMatrix, corrector: SupportsCorrection) -> None:
    """Levanta quando o corretor exige mais QCs de treino por lote do que o lote menor deixa.

    Corre depois de `_guard_train_size`, então só dispara acima do piso geral: é o
    caso do corretor que declara `min_train_qc_per_batch` maior, como o LOESS com
    `MIN_QC_FOR_FIT`, diante de um lote de três QCs. Sem a guarda, os ajustes que
    retêm um QC desse lote treinam com dois, o LOESS devolve as features sem
    correção (`fit_failed`) e essas linhas entrariam no vetor como se fossem
    corrigidas.

    Levanta `BatchQcShortfallError`, antes de qualquer ajuste. Lote sem QC não
    conta, como em `_guard_train_size`.
    """
    minimum = out_of_sample_qc_minimum(corrector)
    if minimum <= MIN_TRAIN_QC_PER_BATCH + 1:
        return
    short = {batch: count for batch, count in _batch_qc_counts(matrix).items() if count < minimum}
    if not short:
        return
    detail = ", ".join(f"{batch}={count} QCs" for batch, count in sorted(short.items()))
    # Frase curta, porque vai ao rodapé da figura e à justificativa do Palpite.
    reason = (
        f"menos de {minimum} QCs num lote ({detail}): cada ajuste fora da amostra treina com um a "
        f"menos, e o {corrector.name} não ajusta com menos de {_train_minimum(corrector)}"
    )
    raise BatchQcShortfallError(
        corrector=corrector.name, minimum=minimum, short=short, reason=reason
    )


def held_out_matrices(matrix: RefinoMatrix) -> tuple[tuple[int, RefinoMatrix], ...]:
    """Cada QC retido e a matriz com ele remarcado: os passos de `leave_one_qc_out_rsd`.

    Serve a quem precisa de mais que o valor corrigido do retido, como a figura
    da curva do LOESS, que desenha a curva refeita sem cada QC. A figura passa
    pelos mesmos passos (a guarda de três QCs por lote, `held_out_rows` e
    `remark_as_sample`) para que os pontos dela sejam os que a métrica agrega.

    Levanta `DegenerateCrossValidationError` pela mesma guarda. Matriz sem QC
    devolve vazio.

    A guarda por corretor (`_guard_corrector_minimum`) não entra aqui: esta função
    não recebe corretor, e a figura não reduz as curvas a um número. Com um lote
    de três, as curvas sem um QC daquele lote não existem (`fit_failed`), e a
    figura conta as que existem (`n_curves`) e diz o motivo de cada uma que falta.
    """
    _guard_train_size(matrix)
    return tuple((int(row), remark_as_sample(matrix, int(row))) for row in held_out_rows(matrix))


def _out_of_sample_rows(
    matrix: RefinoMatrix,
    corrector: SupportsCorrection,
    seed: int,
    requested: tuple[str, ...] | None,
    cancelled: CancelCheck | None = None,
) -> Iterator[np.ndarray]:
    """Valor corrigido do QC retido, uma linha por iteração e na ordem dos QCs.

    `cancelled` é repassado ao corretor: sem isso os pontos de checagem dentro do
    SERRF ficam inertes, e a parada só seria lida depois dos `n_qc` ajustes. A
    checagem de fora responde antes do primeiro ajuste, e a de dentro do laço é
    atingida a cada QC retido.
    """
    target = frozenset(requested) if requested is not None else None
    raise_if_cancelled(cancelled, "leave-one-qc-out: antes do primeiro ajuste")
    for row in held_out_rows(matrix):
        raise_if_cancelled(cancelled, "leave-one-qc-out: antes do ajuste de um QC retido")
        result = corrector.fit_transform(
            remark_as_sample(matrix, int(row)),
            seed=seed,
            target_features=target,
            cancelled=cancelled,
        )
        yield np.asarray(result.matrix.values[int(row), :], dtype=np.float64)


def _one_held_out_value(
    matrix: RefinoMatrix,
    corrector: SupportsCorrection,
    row: int,
    seed: int,
    target: frozenset[str] | None,
) -> np.ndarray:
    """O ajuste de um QC retido e a linha corrigida dele: a unidade que vai a um worker.

    É a mesma chamada que `_out_of_sample_rows` faz por iteração, com os mesmos
    argumentos. Não recebe `cancelled`: a bandeira do pai não alcança outro
    processo, e neste caminho quem a lê é o despacho. Fica no nível do módulo
    porque o `loky` a serializa por referência.
    """
    result = corrector.fit_transform(
        remark_as_sample(matrix, row), seed=seed, target_features=target
    )
    return np.asarray(result.matrix.values[row, :], dtype=np.float64)


def _guard_single_level(corrector: SupportsCorrection, n_jobs: int) -> None:
    """Recusa a régua de vários núcleos com um corretor de vários núcleos, antes de ajustar.

    Dentro de um worker do `loky`, o `Parallel` do corretor abre outro pool, sem
    aviso do `joblib`: seriam até `n_jobs` pools de `n_jobs` processos disputando
    os mesmos núcleos. `n_jobs` não está no `SupportsCorrection`, então é lido por
    atributo, e corretor sem ele conta como série.
    """
    corrector_jobs = getattr(corrector, "n_jobs", 1)
    if n_jobs != 1 and corrector_jobs != 1:
        raise ValueError(
            f"dois níveis de paralelismo: a régua com n_jobs={n_jobs} e o corretor "
            f"`{corrector.name}` com n_jobs={corrector_jobs}. Dentro de um worker do `loky` o "
            "corretor abriria outro pool, um por QC retido; baixe um dos dois para 1."
        )


def out_of_sample_values(
    matrix: RefinoMatrix,
    corrector: SupportsCorrection,
    *,
    seed: int = 42,
    requested: tuple[str, ...] | None = None,
    cancelled: CancelCheck | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Os valores fora da amostra: uma linha por QC retido, na ordem dos QCs.

    Com `n_jobs=1` é o laço de `_out_of_sample_rows`, no processo do chamador e
    com a bandeira lida dentro do corretor. Com `n_jobs != 1` os ajustes correm em
    processos do `loky`, e `Parallel` os devolve na ordem da entrada. A ordem das
    linhas faz parte do contrato: o RSD muda poucos ULPs quando elas são trocadas,
    então o teste compara as linhas, e não só o RSD.

    Se o pool morre no meio (worker sem memória, o sistema matando processos), os
    ajustes são refeitos em série, uma vez, com a mesma saída.

    A parada é grossa: a bandeira é lida no despacho de cada QC retido, no
    processo pai (`CancellableDispatch`). Parar deixa de enviar ajuste novo, e os
    que já saíram terminam. A checagem não levanta dentro do gerador, porque o
    `loky` encerraria o executor, que serve ao processo inteiro.
    """
    if n_jobs == 1:
        return np.vstack(list(_out_of_sample_rows(matrix, corrector, seed, requested, cancelled)))

    raise_if_cancelled(cancelled, "leave-one-qc-out: antes do primeiro ajuste")
    target = frozenset(requested) if requested is not None else None
    despacho = CancellableDispatch(cancelled, "leave-one-qc-out: despacho de um QC retido")
    try:
        # Espera a vez no pool antes de pedir o executor: pedi-lo com outro
        # `n_jobs` enquanto outro trabalho ainda despacha lotes trava o processo.
        with vez_no_pool(n_jobs, cancelled, "leave-one-qc-out: espera da vez no pool"):
            linhas = list(
                # Despacho preguiçoso: o padrão do `joblib` despacha de início
                # `2 * n_jobs` tarefas, o que com poucos QCs são todas, e a bandeira
                # não seria mais lida. Com `pre_dispatch="n_jobs"` e um QC por tarefa,
                # cada ajuste que termina devolve o despacho ao gerador.
                Parallel(n_jobs=n_jobs, backend="loky", pre_dispatch="n_jobs", batch_size=1)(
                    delayed(_one_held_out_value)(matrix, corrector, row, seed, target)
                    for row in despacho.over(int(r) for r in held_out_rows(matrix))
                )
            )
    except MORTE_DO_POOL:
        _LOGGER.warning(
            "o pool de workers morreu com n_jobs=%s; refazendo os ajustes fora da amostra em "
            "SERIE. O resultado e identico — o que muda e o tempo",
            n_jobs,
        )
        return np.vstack(list(_out_of_sample_rows(matrix, corrector, seed, requested, cancelled)))
    # Antes do `vstack`: depois de uma parada as linhas são só as dos QCs despachados.
    despacho.raise_if_stopped()
    return np.vstack(linhas)


def leave_one_qc_out_rsd(
    matrix: RefinoMatrix,
    corrector: SupportsCorrection,
    *,
    variant: RsdVariant = "plain",
    seed: int = 42,
    n_sampled_features: int | None = None,
    cancelled: CancelCheck | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """Estimador deste projeto: partição completa, `n_qc` iterações.

    Para cada QC: remarca-o como `sample`, ajusta a correção sobre a matriz
    inteira, com as amostras biológicas presentes, e guarda o valor corrigido do
    retido, lido de `CorrectionResult.matrix`. Ao fim calcula o RSD por feature
    sobre os `n_qc` valores fora da amostra, cada QC medido uma vez. A partição é
    completa e determinística, diferente da subamostragem repetida do R.

    `n_sampled_features` em `None` mede todas as features. Quem amostra recebe
    `nan` nas não amostradas, indistinguível de RSD indefinido, e por isso
    amostrar não é o padrão.

    Levanta `DegenerateCrossValidationError` se algum lote ficar com menos de
    `MIN_TRAIN_QC_PER_BATCH` QCs no treino, e a subclasse `BatchQcShortfallError`
    se o lote menor tiver menos QCs do que este corretor exige
    (`out_of_sample_qc_minimum`, quatro com o LOESS).

    `cancelled` é repassado ao corretor; ver `_out_of_sample_rows`. Com `n_jobs`
    diferente de 1 os ajustes correm em vários núcleos, com o mesmo vetor da
    série. O corretor tem de vir com `n_jobs=1` (ver `_guard_single_level`), e a
    bandeira só é lida no despacho de cada QC.
    """
    validate_label("RsdVariant", variant)
    rows = held_out_rows(matrix)
    if rows.size == 0:
        return np.full(matrix.n_features, np.nan)

    _guard_train_size(matrix)
    _guard_corrector_minimum(matrix, corrector)
    _guard_single_level(corrector, n_jobs)

    requested = sampled_feature_names(matrix, n_sampled_features, seed)
    out_of_sample = out_of_sample_values(
        matrix, corrector, seed=seed, requested=requested, cancelled=cancelled, n_jobs=n_jobs
    )
    return _scatter(rsd(out_of_sample, axis=0, variant=variant), requested, matrix)


# ---------------------------------------------------------------------------
# O intervalo da barra: jackknife-t sobre os QCs, com reajuste
# ---------------------------------------------------------------------------
#
# A barra de cada corretor no painel da correção é a mediana, sobre as features,
# do RSD dos QCs fora da amostra (`leave_one_qc_out_rsd`). O intervalo diz quanto
# essa mediana depende de quais QCs foram injetados: o replicado `j` é a régua
# refeita sem o QC `j`. Ele não diz quanto a mediana depende das features, que
# são fixas no estudo.
#
# Cada replicado exige reajuste. Reusar as `n` linhas da régua tirando uma de
# cada vez não custaria ajuste nenhum, mas nelas o QC tirado continua no treino
# dos outros, e o erro-padrão sai bem menor que o do jackknife com reajuste.
#
# Um ajuste serve aos dois QCs do par, e por isso o custo é `n(n-1)/2`, e não
# `n(n-1)`. O QC retido e o QC `j` do replicado são ambos remarcados como
# `sample`, então o ajuste que dá o valor de `i` no replicado `j` é o mesmo que dá
# o valor de `j` no replicado `i`, com a mesma semente. Isso supõe corretor
# determinístico, que não lê o `provenance`.
# `tests/unit/test_intervalo_das_barras.py` confere contra a conta por extenso,
# com `n(n-1)` ajustes.

JACKKNIFE_LEVEL: Final = 0.95
"""Nível nominal do intervalo. A cobertura não foi medida, e a figura imprime '95% nominal'."""

JACKKNIFE_METHOD: Final = "jackknife-t com reajuste"

MIN_QC_PER_BATCH_FOR_JACKKNIFE: Final = MIN_TRAIN_QC_PER_BATCH + 2
"""Quatro, o piso. Cada replicado tira um QC e refaz a régua, que precisa de
`MIN_TRAIN_QC_PER_BATCH + 1` QCs por lote. Com três num lote há barra e não há intervalo.

O corretor pode exigir mais. Um ajuste de par treina com dois QCs a menos no lote, e o LOESS
não ajusta com menos de três no treino (`MIN_QC_FOR_FIT`), então com ele o mínimo é cinco. Ver
`jackknife_qc_minimum`."""


@dataclass(frozen=True)
class JackknifeInterval:
    """O intervalo de uma barra, ou o motivo de não haver um. Nunca os dois, nunca nenhum.

    `reason` é `None` exatamente quando os três números existem, e o construtor
    recusa a mistura. Um `nan` ou um zero no lugar dos limites confundiria 'não
    houve intervalo' com 'o intervalo tem largura zero'.

    `estimate` é a barra, a mediana que o oráculo põe em `Label.scores`. Pode ser
    `nan` quando o RSD ficou indefinido em todas as features, e aí há `reason`.
    """

    estimate: float
    level: float
    n_qc: int
    lower: float | None = None
    upper: float | None = None
    standard_error: float | None = None
    replicates: tuple[float, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        numeros = (self.lower, self.upper, self.standard_error)
        if self.reason is None and any(v is None for v in numeros):
            raise ValueError(
                "intervalo sem motivo e sem os três números: ou há limites e erro-padrão, ou há "
                "o motivo de não os haver"
            )
        if self.reason is not None and any(v is not None for v in numeros):
            raise ValueError(
                f"intervalo com motivo ({self.reason!r}) e com números: um dos dois é mentira"
            )


def _train_minimum(corrector: SupportsCorrection) -> int:
    """Quantos QCs de treino por lote o corretor precisa: o piso geral, ou o que ele declara.

    Lido por atributo, porque o `SupportsCorrection` não o tem. Corretor que não o
    declara, como o `none` e o SERRF, fica com o piso.
    """
    declarado = getattr(corrector, "min_train_qc_per_batch", MIN_TRAIN_QC_PER_BATCH)
    return max(MIN_TRAIN_QC_PER_BATCH, int(declarado))


def out_of_sample_qc_minimum(corrector: SupportsCorrection) -> int:
    """Quantos QCs por lote a régua exige com este corretor: o mínimo de treino mais um.

    Cada ajuste retém um QC e treina com os outros do lote. Com o piso isso dá
    três; com o LOESS, que não ajusta com menos de três no treino
    (`MIN_QC_FOR_FIT`), dá quatro.
    """
    return _train_minimum(corrector) + 1


def jackknife_qc_minimum(corrector: SupportsCorrection) -> int:
    """Quantos QCs por lote o intervalo exige com este corretor: o mínimo de treino mais dois.

    Um ajuste de par treina com dois QCs a menos no lote. Com o piso isso dá quatro
    (`MIN_QC_PER_BATCH_FOR_JACKKNIFE`); com o LOESS dá cinco.
    """
    return out_of_sample_qc_minimum(corrector) + 1


def jackknife_cost_ratio(n_qc: int) -> float:
    """Quantas vezes a medição com intervalo custa a medição sem ele: `(n+1)/2`.

    A régua são `n` ajustes, e o intervalo acrescenta um por par de QCs,
    `n(n-1)/2`. A razão é `(n + n(n-1)/2) / n`. Conta ajustes, e não tempo de
    relógio.
    """
    if n_qc < 1:
        raise ValueError(f"`n_qc` tem de ser >= 1; recebi {n_qc}")
    return (n_qc + 1) / 2


def held_out_pairs(matrix: RefinoMatrix) -> tuple[tuple[int, int], ...]:
    """Cada par de QCs uma vez, `(menor, maior)`, na ordem das linhas, que é a do despacho."""
    rows = [int(r) for r in held_out_rows(matrix)]
    return tuple((rows[a], rows[b]) for a in range(len(rows)) for b in range(a + 1, len(rows)))


def remark_pair_as_sample(matrix: RefinoMatrix, pair: tuple[int, int]) -> RefinoMatrix:
    """A matriz com os dois QCs do par remarcados como `sample`.

    Usa `remark_as_sample` duas vezes, e não uma remarcação própria, para que os
    dois caminhos não divirjam.
    """
    first, second = pair
    return remark_as_sample(remark_as_sample(matrix, first), second)


def _pair_fit_rows(
    matrix: RefinoMatrix,
    corrector: SupportsCorrection,
    pair: tuple[int, int],
    seed: int,
    target: frozenset[str] | None,
    cancelled: CancelCheck | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Um ajuste com os dois QCs do par remarcados, e as linhas corrigidas dos dois.

    Fica no nível do módulo porque o `loky` a serializa por referência. No caminho
    de vários núcleos vai sem `cancelled`: a bandeira do pai não alcança outro
    processo, e quem a lê ali é o despacho.
    """
    result = corrector.fit_transform(
        remark_pair_as_sample(matrix, pair),
        seed=seed,
        target_features=target,
        cancelled=cancelled,
    )
    values = np.asarray(result.matrix.values, dtype=np.float64)
    first, second = pair
    return values[first, :].copy(), values[second, :].copy()


def _one_pair_fit(
    matrix: RefinoMatrix,
    corrector: SupportsCorrection,
    pair: tuple[int, int],
    seed: int,
    target: frozenset[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """A unidade que vai a um worker: `_pair_fit_rows` sem a bandeira."""
    return _pair_fit_rows(matrix, corrector, pair, seed, target)


def _pairs_in_series(
    matrix: RefinoMatrix,
    corrector: SupportsCorrection,
    pairs: tuple[tuple[int, int], ...],
    seed: int,
    target: frozenset[str] | None,
    cancelled: CancelCheck | None,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Os ajustes de par no processo do chamador, com a bandeira lida antes de cada um e dentro."""
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for pair in pairs:
        raise_if_cancelled(cancelled, "jackknife: antes do ajuste de um par de QCs")
        out[pair] = _pair_fit_rows(matrix, corrector, pair, seed, target, cancelled)
    return out


def pair_out_of_sample_values(
    matrix: RefinoMatrix,
    corrector: SupportsCorrection,
    *,
    seed: int = 42,
    requested: tuple[str, ...] | None = None,
    cancelled: CancelCheck | None = None,
    n_jobs: int = 1,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """As linhas corrigidas dos dois QCs de cada par, de um ajuste por par: `n(n-1)/2` ajustes.

    Com `n_jobs != 1` os ajustes correm em processos do `loky` e voltam na ordem
    dos pares, com a mesma saída da série. A bandeira é lida no despacho de cada
    par, no processo pai: parar deixa de enviar ajuste novo, os que saíram
    terminam, e a exceção vem depois. Nada levanta dentro do `Parallel`, o que
    derrubaria o executor compartilhado com outros trabalhos do processo. Se o
    pool morre no meio, os ajustes são refeitos em série.
    """
    _guard_single_level(corrector, n_jobs)
    pairs = held_out_pairs(matrix)
    target = frozenset(requested) if requested is not None else None
    raise_if_cancelled(cancelled, "jackknife: antes do primeiro ajuste de par")
    if n_jobs == 1:
        return _pairs_in_series(matrix, corrector, pairs, seed, target, cancelled)

    despacho = CancellableDispatch(cancelled, "jackknife: despacho de um par de QCs")
    try:
        # Espera a vez no pool, como em `out_of_sample_values`.
        with vez_no_pool(n_jobs, cancelled, "jackknife: espera da vez no pool"):
            linhas = list(
                # Despacho preguiçoso, como em `out_of_sample_values`: com o padrão do
                # `joblib` o gerador se esgotaria antes de o primeiro ajuste terminar.
                Parallel(n_jobs=n_jobs, backend="loky", pre_dispatch="n_jobs", batch_size=1)(
                    delayed(_one_pair_fit)(matrix, corrector, pair, seed, target)
                    for pair in despacho.over(pairs)
                )
            )
    except MORTE_DO_POOL:
        _LOGGER.warning(
            "o pool de workers morreu com n_jobs=%s; refazendo os ajustes de par do jackknife em "
            "SERIE. O resultado e identico — o que muda e o tempo",
            n_jobs,
        )
        return _pairs_in_series(matrix, corrector, pairs, seed, target, cancelled)
    despacho.raise_if_stopped()
    return dict(zip(pairs, linhas, strict=True))


def _row_of_pair(
    pairs: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], i: int, j: int
) -> np.ndarray:
    """A linha do QC `i` no ajuste com `i` e `j` remarcados: o valor de `i` no replicado `j`.

    É função nomeada para que o teste possa plantar o defeito de ler a linha do
    outro QC do par e conferir que o estimador o acusa.
    """
    first, second = (i, j) if i < j else (j, i)
    rows = pairs[(first, second)]
    return rows[0] if i == first else rows[1]


def _median_of_rsd(rows: np.ndarray, variant: RsdVariant) -> float:
    """A barra sobre estas linhas: a mediana, sobre as features, do RSD de cada uma.

    O aviso `All-NaN slice` é esperado quando o RSD é indefinido em toda feature.
    Esse desfecho vai no `reason` do intervalo, então o aviso é suprimido.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmedian(rsd(rows, axis=0, variant=variant)))


def _reason_before_pairs(
    matrix: RefinoMatrix, estimate: float, corrector: SupportsCorrection
) -> str | None:
    """O motivo de não haver intervalo que se conhece antes de gastar um ajuste de par, ou `None`.

    O mínimo por lote é o do corretor (`jackknife_qc_minimum`), e a frase diz o
    que o fixou: a régua refeita sem um QC, ou o corretor que não ajusta com dois
    QCs a menos no treino.
    """
    minimum = jackknife_qc_minimum(corrector)
    counts = _batch_qc_counts(matrix)
    short = {batch: count for batch, count in counts.items() if count < minimum}
    if short:
        detail = ", ".join(f"{batch}={count} QCs" for batch, count in sorted(short.items()))
        # Frase curta, porque vai ao rodapé da figura.
        if minimum == MIN_QC_PER_BATCH_FOR_JACKKNIFE:
            return (
                f"menos de {minimum} QCs num lote ({detail}): cada replicado "
                f"tira um QC, e a régua precisa de {MIN_TRAIN_QC_PER_BATCH + 1} por lote"
            )
        return (
            f"menos de {minimum} QCs num lote ({detail}): o ajuste de par treina com dois a "
            f"menos, e o {corrector.name} não ajusta com menos de {_train_minimum(corrector)}"
        )
    if not math.isfinite(estimate):
        return (
            "a barra é indefinida — o RSD ficou indefinido em todas as features —, e não há o que "
            "cercar"
        )
    return None


def leave_one_qc_out_rsd_with_jackknife(
    matrix: RefinoMatrix,
    corrector: SupportsCorrection,
    *,
    variant: RsdVariant = "plain",
    seed: int = 42,
    n_sampled_features: int | None = None,
    level: float = JACKKNIFE_LEVEL,
    cancelled: CancelCheck | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, JackknifeInterval]:
    """A régua de `leave_one_qc_out_rsd` e o intervalo jackknife-t da mediana dela.

    O vetor devolvido é o de `leave_one_qc_out_rsd`, pelos mesmos passos, e a
    estimativa do intervalo é o `nanmedian` dele, que é o número que o oráculo põe
    na barra.

    O replicado `j` é a mediana do RSD dos outros `n - 1` QCs, cada um lido do
    ajuste em que ele e `j` foram remarcados. O erro-padrão é o do jackknife,
    `sqrt((n-1)/n * soma (t_j - média)^2)`, e o intervalo é `estimativa ± t * EP`,
    com `t` de Student com `n - 1` graus de liberdade. O nível é nominal: a
    cobertura não foi medida. Os limites não são cortados em zero; um limite
    inferior negativo diz que o erro-padrão é grande para a barra.

    Sem intervalo, devolve o motivo e não gasta ajuste a mais: menos QCs num lote
    que `jackknife_qc_minimum` (quatro, ou cinco com o LOESS), barra indefinida
    ou, já depois dos pares, um replicado indefinido. A régua de base levanta
    `DegenerateCrossValidationError` quando nem a barra existe, e
    `BatchQcShortfallError` quando o lote menor não deixa a este corretor os QCs
    de treino da barra.

    A guarda por lote não cobre o corretor que, com o lote acima do mínimo, deixa
    sem correção features que os ajustes de base corrigiram. É o caso do LOESS
    quando o descarte de aberrantes ou os QCs não detectados deixam menos de três
    QCs utilizáveis numa feature. A régua já tem essa mistura entre os seus `n`
    ajustes, e o intervalo não a separa.
    """
    validate_label("RsdVariant", variant)
    if not (0.0 < level < 1.0):
        raise ValueError(
            f"o nível do intervalo tem de estar entre 0 e 1, exclusivo; recebi {level}"
        )
    rows = held_out_rows(matrix)
    if rows.size == 0:
        return np.full(matrix.n_features, np.nan), JackknifeInterval(
            estimate=float("nan"),
            level=level,
            n_qc=0,
            reason="sem QC nesta matriz: não há barra nem intervalo",
        )

    _guard_train_size(matrix)
    _guard_corrector_minimum(matrix, corrector)
    _guard_single_level(corrector, n_jobs)

    requested = sampled_feature_names(matrix, n_sampled_features, seed)
    out_of_sample = out_of_sample_values(
        matrix, corrector, seed=seed, requested=requested, cancelled=cancelled, n_jobs=n_jobs
    )
    vector = _scatter(rsd(out_of_sample, axis=0, variant=variant), requested, matrix)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        estimate = float(np.nanmedian(vector))
    n_qc = int(rows.size)

    reason = _reason_before_pairs(matrix, estimate, corrector)
    if reason is not None:
        return vector, JackknifeInterval(estimate=estimate, level=level, n_qc=n_qc, reason=reason)

    pairs = pair_out_of_sample_values(
        matrix, corrector, seed=seed, requested=requested, cancelled=cancelled, n_jobs=n_jobs
    )
    qc_rows = [int(r) for r in rows]
    replicates = tuple(
        _median_of_rsd(np.vstack([_row_of_pair(pairs, i, j) for i in qc_rows if i != j]), variant)
        for j in qc_rows
    )
    undefined = [
        matrix.samples[j].name
        for j, value in zip(qc_rows, replicates, strict=True)
        if not math.isfinite(value)
    ]
    if undefined:
        return vector, JackknifeInterval(
            estimate=estimate,
            level=level,
            n_qc=n_qc,
            replicates=replicates,
            reason=(
                "o replicado sem o QC «" + "», «".join(undefined) + "» ficou indefinido: sem ele o "
                "RSD é indefinido em todas as features, e o erro-padrão não se calcula"
            ),
        )

    values = np.asarray(replicates, dtype=np.float64)
    standard_error = float(np.sqrt((n_qc - 1) / n_qc * np.sum((values - values.mean()) ** 2)))
    t = float(stats.t.ppf(0.5 + level / 2.0, n_qc - 1))
    return vector, JackknifeInterval(
        estimate=estimate,
        level=level,
        n_qc=n_qc,
        lower=estimate - t * standard_error,
        upper=estimate + t * standard_error,
        standard_error=standard_error,
        replicates=replicates,
    )


# ---------------------------------------------------------------------------
# O estimador de compatibilidade
# ---------------------------------------------------------------------------


def train_count(n_qc: int, train_ratio: float) -> int:
    """`round(n_qc * train_ratio)`, como o `tools.R` nas linhas 895 e 1259.

    `round(7 * 0,7) = 5`: cinco de treino e dois no retido. `int()` truncaria para
    4 e deixaria três no retido, sem levantar. O `round` do R e o do Python
    arredondam meio para par, então a transcrição coincide.
    """
    # `round(float)` sem `ndigits` já devolve `int`.
    return round(n_qc * train_ratio)


def qc_submatrix(matrix: RefinoMatrix, train: np.ndarray, target: np.ndarray) -> RefinoMatrix:
    """Submatriz de QC e mais nada, com `train` como `qc` e `target` como `sample`.

    As amostras biológicas não entram. O R passa `train = e_qc[, treino]` e
    `target = e_qc[, retido]` (`tools.R:1267`). Manter a matriz inteira e só
    remarcar os retidos, como `leave_one_qc_out_rsd` faz, mudaria o alvo, a matriz
    de correlação e a seleção de preditores, e não reproduziria a coluna golden.

    As linhas ficam em ordem de injeção, e não `train` depois `target`: o
    `__post_init__` exige `injection_order` crescente, e é o rótulo que distingue
    os dois conjuntos.
    """
    rows = np.sort(np.concatenate([train, target]))
    in_target = set(target.tolist())
    samples = tuple(
        replace(matrix.samples[int(row)], sample_type="sample")
        if int(row) in in_target
        else matrix.samples[int(row)]
        for row in rows
    )
    subset = replace(
        matrix,
        values=np.array(matrix.values[rows, :], copy=True),
        samples=samples,
    )
    return subset.with_provenance(
        f"crossval:r_cv(train={len(train)},target={len(target)},cols={len(rows)})"
    )


def _guard_batch_representation(matrix: RefinoMatrix, n_target: int, *, enforce: bool) -> None:
    """As duas configurações insatisfazíveis, verificadas antes do laço.

    São decidíveis por aritmética, então não gastam o teto de tentativas nem um
    ajuste. As causas ficam separadas porque pedem ações diferentes: 'o retido é
    pequeno' se resolve com mais QCs, 'um lote não tem QC' não.
    """
    if not enforce:
        return

    batches_present = {sample.batch for sample in matrix.samples}
    qc_counts = _batch_qc_counts(matrix)

    # A condição do R compara os lotes dos QCs retidos com os lotes de todas as
    # amostras (`tools.R:1015`), então lote com amostras e nenhum QC nunca é
    # alcançado. Basta haver dois lotes para isso acontecer.
    without_qc = sorted(batches_present - set(qc_counts))
    if without_qc:
        raise DegenerateCrossValidationError(
            f"lotes presentes na matriz sem nenhum QC: {without_qc}. O laço da linha "
            "1263 do `tools.R` giraria para sempre, porque o retido só alcança lotes "
            "que têm QC. Nenhuma quantidade de QCs remove isto — só todo lote "
            "ter ao menos um."
        )

    # Mais lotes que posições no retido: impossível por contagem, e se resolve com mais QCs.
    if len(batches_present) > n_target:
        raise DegenerateCrossValidationError(
            f"{len(batches_present)} lotes não cabem num retido de {n_target} posições. "
            "Com `train_ratio` o retido cresce como ~0,3·n_qc, então isto "
            "dissolve com mais QCs — é insatisfazibilidade conjunta entre contagem de "
            "lotes e de QCs, não propriedade de nenhuma das duas."
        )


def _draw(
    matrix: RefinoMatrix,
    rng: np.random.Generator,
    n_train: int,
    *,
    enforce: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Um sorteio de treino/retido, resorteando até todo lote estar no retido."""
    qc_rows = held_out_rows(matrix)
    batches = np.array([matrix.samples[int(row)].batch for row in qc_rows])
    n_batches = len({sample.batch for sample in matrix.samples})

    for _ in range(MAX_DRAW_ATTEMPTS):
        permuted = rng.permutation(qc_rows.size)
        train = qc_rows[np.sort(permuted[:n_train])]
        target = qc_rows[np.sort(permuted[n_train:])]
        if not enforce:
            return train, target
        if len(set(batches[np.sort(permuted[n_train:])].tolist())) >= n_batches:
            return train, target

    raise DegenerateCrossValidationError(
        f"nenhum sorteio com todos os lotes no retido em {MAX_DRAW_ATTEMPTS} tentativas. "
        "A configuração é satisfazível por contagem — as impossíveis levantam antes —, "
        "então isto é sorteio azarado ou desbalanceamento extremo, e admite outra "
        "semente. Mensagem distinta das de lotes que não cabem no retido e de lote sem QC, de "
        "propósito."
    )


def qc_rsd_r_cv(
    matrix: RefinoMatrix,
    corrector: SupportsCorrection,
    *,
    n_draws: int = 5,
    train_ratio: float = 0.7,
    variant: RsdVariant = "plain",
    seed: int = 42,
    enforce_batch_representation: bool | None = None,
    n_sampled_features: int | None = None,
) -> np.ndarray:
    """Estimador de compatibilidade: replica a subamostragem repetida do `tools.R`.

    `n_draws` sorteios independentes de `train_ratio` dos QCs, reajuste do
    corretor sobre as colunas de QC e mais nada, RSD só nos retidos, média dos
    sorteios.

    `enforce_batch_representation` em `None` deriva do corretor: a restrição da
    linha 1263 do `tools.R` existe no caminho do SERRF e não no do `batch_loess`.
    `n_sampled_features` em `None` mede todas as features, como em
    `leave_one_qc_out_rsd`.

    Existe só para reproduzir as colunas `SERRF` e `batch_loess` do
    `QC - RSD.csv`. É ruidoso por construção, com dois pontos por sorteio. No R é
    irreprodutível, porque o RNG não é semeado; aqui é determinístico. Não deve
    decidir nada.
    """
    validate_label("RsdVariant", variant)
    qc_rows = held_out_rows(matrix)
    if qc_rows.size == 0:
        return np.full(matrix.n_features, np.nan)

    enforce = (
        corrector.name in BATCH_CONSTRAINED_CORRECTORS
        if enforce_batch_representation is None
        else enforce_batch_representation
    )
    n_train = train_count(qc_rows.size, train_ratio)
    n_target = qc_rows.size - n_train
    if n_target < 1:
        raise DegenerateCrossValidationError(
            f"`train_ratio={train_ratio}` deixa {n_target} QCs no retido com "
            f"{qc_rows.size} disponíveis; sem retido não há medição fora da amostra."
        )

    _guard_batch_representation(matrix, n_target, enforce=enforce)

    # Cada sorteio tem fluxo próprio, derivado de `(seed, sorteio)`. Com um fluxo
    # só, as tentativas gastas por um sorteio deslocariam os seguintes.
    requested = sampled_feature_names(matrix, n_sampled_features, seed)
    target_features = frozenset(requested) if requested is not None else None

    per_draw: list[np.ndarray] = []
    for draw in range(n_draws):
        rng = np.random.default_rng([seed, draw])
        train, target = _draw(matrix, rng, n_train, enforce=enforce)
        result = corrector.fit_transform(
            qc_submatrix(matrix, train, target),
            seed=draw_seed(seed, draw),
            target_features=target_features,
        )
        rows = np.sort(np.concatenate([train, target]))
        held = np.isin(rows, target)
        per_draw.append(rsd(np.asarray(result.matrix.values)[held, :], axis=0, variant=variant))

    # `catch_warnings` além de `errstate`: feature com RSD indefinido em todos os
    # sorteios dá coluna toda `nan`, e o `nanmean` avisa pelo módulo `warnings`,
    # que `errstate` não alcança. O desfecho `nan` é o correto.
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        media = np.asarray(np.nanmean(np.vstack(per_draw), axis=0), dtype=np.float64)
    return _scatter(media, requested, matrix)

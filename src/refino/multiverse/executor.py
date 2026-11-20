"""Executor da grade do multiverso, pela árvore de prefixos.

As especificações formam uma árvore em que a correção é a raiz e o resto do pipeline
são as folhas. A correção é a etapa cara (minutos no SERRF, contra frações de segundo
por folha), então ela roda uma vez por prefixo `(correction, n_batches)` e todas as
especificações do prefixo reaproveitam a matriz corrigida. `CorrectorCallCounter`
conta as chamadas do corretor e permite conferir que foi isso que aconteceu.

O paralelismo fica nas folhas, que são muitas chamadas baratas. O corretor roda com
`n_jobs=1` por padrão: paralelizar os dois níveis ao mesmo tempo cria mais processos
que núcleos.
"""

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import groupby

import numpy as np

from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.correction.serrf import SerrfCorrector
from refino.correction.serrf_na_vez import SerrfNaVez
from refino.domain.cancellation import CancelCheck, CancellableDispatch, raise_if_cancelled
from refino.domain.correction import CorrectionResult
from refino.domain.labels import Correction
from refino.domain.matrix import RefinoMatrix
from refino.domain.pool import vez_no_pool
from refino.domain.spec import PipelineSpec
from refino.multiverse.grid import GridDefinition
from refino.preprocessing.pipeline import process_matrix
from refino.preprocessing.stage import StageResult

__all__ = [
    "CellResult",
    "CorrectorCallCounter",
    "MultiverseRun",
    "build_corrector",
    "run_multiverse",
    "run_naive",
]


@dataclass
class CorrectorCallCounter:
    """Contador de invocações do corretor, por chave de prefixo.

    Conta por prefixo e não só o total: um total certo com um prefixo chamado duas
    vezes e outro nenhuma esconderia a árvore quebrada.
    """

    per_prefix: dict[tuple[Correction, int], int] = field(default_factory=dict)

    def record(self, prefix: tuple[Correction, int]) -> None:
        self.per_prefix[prefix] = self.per_prefix.get(prefix, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.per_prefix.values())

    def defects_against(self, grid: GridDefinition) -> list[str]:
        """Lista os desvios em relação a uma chamada do corretor por prefixo da grade.

        Devolve lista vazia quando cada prefixo foi invocado exatamente uma vez.
        """
        found: list[str] = []
        esperadas = set(grid.prefix_keys())
        if self.total != len(esperadas):
            found.append(
                f"o corretor foi invocado {self.total} vez(es) e a grade tem {len(esperadas)} "
                f"prefixo(s) distinto(s). A árvore existe para que sejam iguais; {grid.size()} "
                "invocações significaria que ela não amortiza nada"
            )
        for prefixo in sorted(esperadas):
            vezes = self.per_prefix.get(prefixo, 0)
            if vezes != 1:
                found.append(
                    f"prefixo {prefixo} invocado {vezes} vez(es), esperado exactamente 1. Zero "
                    "significa subárvore não executada; mais de um, correção repetida que o total "
                    "agregado poderia esconder"
                )
        for prefixo in sorted(set(self.per_prefix) - esperadas):
            found.append(f"prefixo {prefixo} invocado e não pertence à grade")
        return found


@dataclass(frozen=True)
class CellResult:
    """Uma célula da grade: a especificação e o que o caminho dela produziu.

    `corrected` é a matriz logo depois da correção, e `matrix` a do fim do pipeline.
    As métricas são medidas nos dois pontos, e sem `corrected` só o segundo seria
    alcançável a partir da célula. Ela é a mesma `RefinoMatrix`, imutável, partilhada
    por todas as folhas do prefixo; guarda-se só a referência.
    """

    spec: PipelineSpec
    matrix: RefinoMatrix
    corrected: RefinoMatrix
    stages: tuple[StageResult, ...]


@dataclass(frozen=True)
class MultiverseRun:
    """O resultado da execução: as células, as correções por prefixo e o contador.

    `counter` viaja no resultado para que a amortização seja conferível em qualquer
    execução, e não só dentro de um teste.
    """

    cells: tuple[CellResult, ...]
    corrections: dict[tuple[Correction, int], CorrectionResult]
    counter: CorrectorCallCounter
    elapsed_seconds: float
    n_jobs: int


def build_corrector(
    name: Correction, corrector_jobs: int = 1
) -> NoneCorrector | LoessCorrector | SerrfCorrector:
    """O corretor pelo nome, com o `n_jobs` do SERRF passado explicitamente.

    O padrão de `corrector_jobs` é 1: em grade pequena, criar processos custa mais que
    o trabalho, e as folhas já podem estar em paralelo. Na grade completa a correção
    domina o tempo, e quem quer paralelismo dentro dela o pede aqui.
    """
    if name == "none":
        return NoneCorrector()
    if name == "loess":
        return LoessCorrector()
    if name == "serrf":
        return SerrfNaVez(n_jobs=corrector_jobs)
    raise ValueError(f"corretor desconhecido: {name!r}")


def run_multiverse(
    matrix: RefinoMatrix,
    grid: GridDefinition,
    *,
    n_jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
    counter: CorrectorCallCounter | None = None,
    cancelled: CancelCheck | None = None,
    corrector_jobs: int = 1,
) -> MultiverseRun:
    """Executa a grade pela árvore de prefixos, corrigindo uma vez por prefixo.

    `n_jobs` distribui as folhas; `corrector_jobs` distribui as features dentro de cada
    correção. Os dois têm padrão 1: em grade pequena, criar processos custa mais que o
    trabalho. Não há `seed` na assinatura; a única fonte de semente é `PipelineSpec.seed`.

    `cancelled` é checada em três pontos: antes da correção de cada prefixo, dentro do
    corretor (que a recebe e a lê no laço por feature) e entre células. O primeiro
    evita pagar uma correção inteira por um pedido que chegou entre prefixos; o segundo
    existe porque a correção é uma chamada só e pode levar muitos minutos.

    `ExecutionCancelledError` propaga. Devolver um `MultiverseRun` parcial obrigaria
    todo chamador a distinguir completo de parcial.
    """
    contador = counter if counter is not None else CorrectorCallCounter()
    inicio = time.perf_counter()

    especificacoes = list(grid.enumerate())
    correcoes: dict[tuple[Correction, int], CorrectionResult] = {}
    celulas: list[CellResult] = []
    total = len(especificacoes)

    for prefixo, grupo in groupby(especificacoes, key=lambda s: (s.correction, s.n_batches)):
        # Raiz: corrige uma vez por prefixo e guarda o `CorrectionResult` inteiro, não só
        # a matriz. As contagens de features não corrigidas, por causa, valem para todas
        # as folhas do prefixo. A checagem de cancelamento vem antes de pagar a correção.
        raise_if_cancelled(cancelled, f"executor: antes da correcao do prefixo {prefixo}")
        contador.record(prefixo)
        corretor = build_corrector(prefixo[0], corrector_jobs)
        resultado = corretor.fit_transform(
            matrix, batches=_batches_for(matrix, prefixo[1]), cancelled=cancelled
        )
        correcoes[prefixo] = resultado

        # Folhas: baratas, e é onde o paralelismo fica.
        for celula in _run_leaves(
            resultado.matrix, list(grupo), n_jobs=n_jobs, cancelled=cancelled
        ):
            raise_if_cancelled(cancelled, "executor: entre celulas")
            celulas.append(celula)
            if progress is not None:
                progress(len(celulas), total)

    return MultiverseRun(
        cells=tuple(celulas),
        corrections=correcoes,
        counter=contador,
        elapsed_seconds=time.perf_counter() - inicio,
        n_jobs=n_jobs,
    )


def run_naive(
    matrix: RefinoMatrix,
    grid: GridDefinition,
    *,
    counter: CorrectorCallCounter | None = None,
    cancelled: CancelCheck | None = None,
) -> MultiverseRun:
    """Execução ingênua: corrige uma vez por especificação.

    Serve de referência para os testes: a saída da árvore de prefixos tem de ser
    idêntica à desta, e o contador daqui dá `size()` invocações. `cancelled` existe
    para que a assinatura seja compatível com a de `run_multiverse`, já que
    `api/progresso.py` aceita um executor injetável.
    """
    contador = counter if counter is not None else CorrectorCallCounter()
    inicio = time.perf_counter()
    celulas: list[CellResult] = []
    correcoes: dict[tuple[Correction, int], CorrectionResult] = {}

    for spec in grid.enumerate():
        raise_if_cancelled(cancelled, "run_naive: entre especificacoes")
        prefixo = (spec.correction, spec.n_batches)
        contador.record(prefixo)
        corretor = build_corrector(spec.correction)
        resultado = corretor.fit_transform(
            matrix, batches=_batches_for(matrix, spec.n_batches), cancelled=cancelled
        )
        correcoes[prefixo] = resultado
        processada, etapas = process_matrix(resultado.matrix, spec)
        celulas.append(
            CellResult(
                spec=spec, matrix=processada, corrected=resultado.matrix, stages=tuple(etapas)
            )
        )

    return MultiverseRun(
        cells=tuple(celulas),
        corrections=correcoes,
        counter=contador,
        elapsed_seconds=time.perf_counter() - inicio,
        n_jobs=1,
    )


def _run_leaves(
    corrected: RefinoMatrix,
    specs: list[PipelineSpec],
    *,
    n_jobs: int,
    cancelled: CancelCheck | None = None,
) -> Iterator[CellResult]:
    """As folhas de um prefixo: serial quando `n_jobs == 1`, por `joblib` acima disso.

    Os dois caminhos têm de produzir saída idêntica.

    No ramo serial a função cede uma célula de cada vez, e a checagem de cancelamento
    de `run_multiverse` corre entre células. No ramo paralelo o `Parallel` executa tudo
    antes do primeiro `yield`, então a checagem fica no despacho: `CancellableDispatch`
    para de entregar trabalho novo quando pedem, o que já foi entregue termina, e a
    exceção sobe depois de o `Parallel` devolver. Levantar dentro do gerador faria o
    `loky` encerrar o executor do processo, com os trabalhos de qualquer outro
    `Parallel`. A exceção sobe antes do `zip` estrito, que acusaria a saída parcial.

    O `Parallel` corre dentro de `vez_no_pool`, que espera até o executor partilhado
    do `loky` poder servir este `n_jobs` sem mudar por baixo de outro trabalho.
    """
    if n_jobs == 1:
        for spec in specs:
            processada, etapas = process_matrix(corrected, spec)
            yield CellResult(
                spec=spec, matrix=processada, corrected=corrected, stages=tuple(etapas)
            )
        return

    from joblib import Parallel, delayed

    despacho = CancellableDispatch(cancelled, "executor: despacho de celula")
    with vez_no_pool(n_jobs, cancelled, "executor: espera da vez no pool"):
        resultados = Parallel(n_jobs=n_jobs)(
            delayed(process_matrix)(corrected, s) for s in despacho.over(specs)
        )
    despacho.raise_if_stopped()
    for spec, (processada, etapas) in zip(specs, resultados, strict=True):
        yield CellResult(spec=spec, matrix=processada, corrected=corrected, stages=tuple(etapas))


def _batches_for(matrix: RefinoMatrix, n_batches: int) -> np.ndarray | None:
    """Vetor de lote por amostra, ou `None` quando é lote único.

    Com mais de um lote, divide as amostras em blocos contíguos de tamanho quase igual
    na ordem de injeção. Devolve `None` e não um vetor de zeros porque o corretor
    distingue 'sem declaração de lote' de 'um lote declarado'.
    """
    if n_batches <= 1:
        return None
    ordem = np.array([a.injection_order for a in matrix.samples])
    lotes = np.empty(matrix.n_samples, dtype=np.int64)
    for indice, bloco in enumerate(np.array_split(np.argsort(ordem), n_batches)):
        lotes[bloco] = indice
    return lotes

"""Fronteira dos corretores: o protocolo `Corrector` e o que é comum aos três.

O corretor devolve `CorrectionResult`, e não só a matriz, para que cada feature não
corrigida saia com a sua causa (ver `refino.domain.correction`).
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from refino.domain.correction import CorrectionResult
from refino.domain.matrix import RefinoMatrix


class Corrector(Protocol):
    """O que todo corretor de deriva satisfaz estruturalmente.

    `batches` é o vetor de rótulos de lote alinhado a `matrix.samples`, ou `None` para
    lote único. Todo corretor recebe lote porque, no SERRF, os passos 5 a 9 usam âncora
    global, e com um lote só a âncora global e a local coincidem numericamente: testado
    só com lote único, o corretor não distingue os dois desenhos.

    `target_features` restringe quais features recebem correção, e `None` é execução
    completa. O corretor continua vendo todas as colunas, porque o passo 1 e a seleção de
    preditores do SERRF precisam do conjunto inteiro. Muda só o conjunto que sai, e o
    `CorrectionResult` declara o pedido.

    `cancelled` responde 'devo parar?' e é lido nos laços por feature. O corretor só o lê.
    É parâmetro e não atributo da classe porque `serrf.py` despacha
    `self._correct_feature` pelo `joblib` com backend `loky`, que serializa `self`, e um
    `threading.Event` não é serializável. O tipo vem por extenso, em vez do alias
    `CancelCheck` de `domain/cancellation.py`, para este módulo depender só de
    `domain.correction` e `domain.matrix`.
    """

    name: str

    def identidade(self) -> str:
        """O que, no construtor, faz duas instâncias produzirem valores diferentes.

        É a parte do corretor na chave de reuso de correções entre rotas. `fit_transform`
        já recebe a matriz, os lotes, a semente e o alvo; faltam os eixos que cada
        corretor fixa na construção, e só o corretor sabe quais são.

        Ficam de fora os eixos que mudam o tempo e não o valor: `n_jobs`,
        `hoist_correlations` e `fast_param_names`. `tests/unit/test_forest_speedup.py`
        afirma que a saída é idêntica com e sem cada um; sem esses testes a chave deixa
        de ser fiel.

        A forma é `nome(chave=valor,chave=valor)`, estável e legível, porque entra num
        `sha256` e em mensagens de diagnóstico.
        """
        ...

    def fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> CorrectionResult: ...


def resolve_target_features(
    matrix: RefinoMatrix, target_features: frozenset[str] | None
) -> tuple[str, ...]:
    """Normaliza o pedido e o devolve na ordem das colunas da matriz.

    A ordem vem da matriz porque `frozenset` não tem ordem: iterá-lo faria a ordem das
    colunas de saída depender do hash do interpretador. Pedido vazio, ou que nomeia
    feature que a matriz não tem, levanta `ValueError`.
    """
    if target_features is None:
        return tuple(matrix.feature_names)
    desconhecidas = sorted(target_features - set(matrix.feature_names))
    if desconhecidas:
        raise ValueError(
            f"`target_features` nomeia feature que não existe na matriz: "
            f"{desconhecidas[:5]}. Pedido tem de ser subconjunto da entrada."
        )
    if not target_features:
        raise ValueError("`target_features` vazio: pedido sem features não é execução parcial")
    return tuple(name for name in matrix.feature_names if name in target_features)


def feature_columns(matrix: RefinoMatrix, requested: tuple[str, ...]) -> np.ndarray:
    """Índices de coluna das features pedidas, na ordem do pedido."""
    index = {name: position for position, name in enumerate(matrix.feature_names)}
    return np.array([index[name] for name in requested], dtype=np.intp)


def select_features(matrix: RefinoMatrix, requested: tuple[str, ...]) -> RefinoMatrix:
    """Matriz com só as colunas pedidas, preservando ordem e metadados.

    Quando o pedido é o conjunto completo devolve a própria matriz, sem cópia.
    """
    if requested == tuple(matrix.feature_names):
        return matrix
    return replace(
        matrix,
        values=np.array(matrix.values[:, feature_columns(matrix, requested)], copy=True),
        feature_names=requested,
    )


@dataclass(frozen=True)
class ZeroRestoration:
    """O que o passo 11 fez, com as contagens separadas.

    `restored` conta as posições que eram zero na entrada e voltaram a zero. `created`
    conta as que não eram zero e viraram zero pela correção; no `batch_loess` isso
    acontece na divisão por `Inf`. Somar as duas faria um corretor que cria zeros parecer
    um que os preserva.
    """

    values: np.ndarray
    restored: int
    created: int
    missing_restored: int


def restore_zeros_and_missing(
    original: np.ndarray,
    corrected: np.ndarray,
) -> ZeroRestoration:
    """Passo 11: zero volta a zero, ausente volta a ausente.

    Fica aqui, e não dentro do SERRF, porque no `tools.R` quem restaura é o escritor
    (linhas 1610 e 1611), para os três métodos. O equipamento deixa a célula vazia quando
    não detecta, e nenhuma correção transforma não detecção em medida.

    Os zeros que a correção criou (posições que não eram zero e passaram a ser) são
    contados e ficam: são resultado do método, e não ausência de detecção.
    """
    before = np.asarray(original, dtype=np.float64)
    after = np.asarray(corrected, dtype=np.float64)
    if before.shape != after.shape:
        raise ValueError(
            f"o passo 11 compara posição a posição; recebi {before.shape} e {after.shape}"
        )

    was_zero = before == 0.0
    was_missing = np.isnan(before)
    created = int(np.sum((after == 0.0) & ~was_zero & ~was_missing))

    values = np.array(after, copy=True)
    values[was_zero] = 0.0
    values[was_missing] = np.nan
    return ZeroRestoration(
        values=values,
        restored=int(np.sum(was_zero)),
        created=created,
        missing_restored=int(np.sum(was_missing)),
    )

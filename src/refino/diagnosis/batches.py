"""Proposta de fronteiras de lote a partir dos pontos de mudança detectados.

Os pontos de mudança chegam em ordem de injeção, a mesma unidade que
`detect_changepoints` devolve. O retorno é alinhado a `matrix.samples`, que não está
em ordem de injeção: nos arquivos do laboratório as colunas vêm agrupadas por classe.

`propose_batches` traduz os cortes em rótulos, sem julgar se a partição serve.
`propose_supported_batches` é a proposta que vai a quem opera: adota os cortes só quando
todo lote fica com QCs bastantes para medir um corretor fora da amostra, e senão propõe
um lote único, guardando os cortes detectados como observação.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from refino.domain.matrix import RefinoMatrix
from refino.metrics.crossval import MIN_TRAIN_QC_PER_BATCH

__all__ = [
    "MIN_QC_PER_PROPOSED_BATCH",
    "BatchProposal",
    "batch_without_training",
    "propose_batches",
    "propose_supported_batches",
]

# Um QC retido e `MIN_TRAIN_QC_PER_BATCH` para ajustar: o mínimo para medir um corretor fora
# da amostra dentro de um lote. É o mesmo limiar com que o Palpite e a medição completa recusam
# uma partição, e por isso sai da mesma constante: uma proposta que passa aqui não é recusada lá.
#
# É o mínimo matemático, e fica abaixo do que a literatura usa para corrigir deriva por lote: a
# implementação de referência do QC-RSC (pacote `pmp`, Kirwan et al. 2013) não corrige uma feature
# num lote com menos de cinco QCs, por padrão.
MIN_QC_PER_PROPOSED_BATCH: Final = MIN_TRAIN_QC_PER_BATCH + 1


def batch_without_training(qcs_per_batch: Sequence[int]) -> bool:
    """Diz se algum lote tem QC, mas menos que `MIN_QC_PER_PROPOSED_BATCH`.

    Lote sem QC não conta: reter um QC não o afeta, e o corretor o salta. É o predicado que
    a API usa para recusar o Palpite e a medição completa, e o que a proposta usa para não
    sugerir uma partição que eles recusariam.
    """
    return any(0 < quantos < MIN_QC_PER_PROPOSED_BATCH for quantos in qcs_per_batch)


@dataclass(frozen=True)
class BatchProposal:
    """A partição proposta e a que a detecção achou, que diferem quando a proposta recuou.

    `labels` e `changepoints` são a proposta: o que o Palpite mede e o que a sugestão põe na
    correção. `detected_changepoints` e `detected_qcs_per_batch` são a observação: os cortes
    que a detecção devolveu e quantos QCs cada lote teria com eles.
    """

    labels: tuple[int, ...]
    changepoints: tuple[int, ...]
    detected_changepoints: tuple[int, ...]
    detected_qcs_per_batch: tuple[int, ...]

    @property
    def fell_back(self) -> bool:
        """Se a detecção achou cortes que a proposta não adotou."""
        return self.changepoints != self.detected_changepoints


def propose_batches(matrix: RefinoMatrix, changepoints: list[int]) -> np.ndarray:
    """Rótulos de lote por amostra, alinhados a `matrix.samples`.

    Cada ponto de mudança é a ordem de injeção que inicia um lote novo: amostra com
    `injection_order` menor que o primeiro corte fica no lote 0, e assim por diante.
    Com `k` cortes saem `k + 1` lotes, rotulados de 0 a `k`. Traduzir para as letras
    que os arquivos do laboratório usam é apresentação e fica com quem exibe.

    Levanta quando um corte cai fora do intervalo de ordens de injeção: ele criaria
    um lote sem amostra, que o corretor saltaria em silêncio por não ter QC.
    """
    ordens = np.asarray(matrix.injection_orders)
    if changepoints:
        menor, maior = int(ordens.min()), int(ordens.max())
        fora = [c for c in changepoints if not menor < c <= maior]
        if fora:
            raise ValueError(
                f"pontos de mudança fora do intervalo de ordens de injeção "
                f"({menor} a {maior}): {fora}. Corte na primeira ordem ou além da última "
                "produziria lote sem amostra nenhuma, que o corretor saltaria em silêncio "
                "por não ter QC."
            )

    fronteiras = np.asarray(sorted(int(c) for c in changepoints), dtype=np.int64)
    return np.searchsorted(fronteiras, ordens, side="right").astype(np.int64)


def _qcs_per_label(matrix: RefinoMatrix, labels: np.ndarray) -> tuple[int, ...]:
    """Quantos QCs caem em cada lote, na ordem dos rótulos; lote sem QC aparece como zero."""
    dos_qcs = labels[matrix.qc_mask]
    return tuple(int(np.count_nonzero(dos_qcs == lote)) for lote in sorted(set(labels.tolist())))


def propose_supported_batches(matrix: RefinoMatrix, changepoints: list[int]) -> BatchProposal:
    """A proposta de lotes que vai a quem opera, alinhada a `matrix.samples`.

    Adota os cortes detectados quando todo lote fica com pelo menos `MIN_QC_PER_PROPOSED_BATCH`
    QCs. Senão propõe um lote único: a detecção corre sobre a trajetória dos QCs e aceita trecho
    de dois, e com sete QCs ela pode sugerir lotes de dois QCs, que nem o Palpite nem a medição
    completa conseguem medir, e que nenhum método de correção por QC sustenta. Os cortes
    detectados continuam no retorno, para quem exibe dizer o que foi achado e por que não entrou.

    O perfil do diagnóstico não muda: os cortes detectados e os rótulos de `propose_batches`
    continuam a alimentar a classificação, as características da política e o banco.
    """
    detectados = tuple(sorted(int(c) for c in changepoints))
    rotulos = propose_batches(matrix, list(detectados))
    qcs = _qcs_per_label(matrix, rotulos)
    if batch_without_training(qcs):
        return BatchProposal(
            labels=(0,) * matrix.n_samples,
            changepoints=(),
            detected_changepoints=detectados,
            detected_qcs_per_batch=qcs,
        )
    return BatchProposal(
        labels=tuple(int(r) for r in rotulos),
        changepoints=detectados,
        detected_changepoints=detectados,
        detected_qcs_per_batch=qcs,
    )

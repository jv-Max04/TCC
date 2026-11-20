"""A proposta de lotes do diagnóstico adota os cortes detectados só quando todo lote sustenta medir.

Com sete QCs, a detecção sobre a trajetória dos QCs aceita trecho de dois e pode cortar onde o
Palpite e a medição completa recusam medir: um QC retido e dois para ajustar é o mínimo. Nesses
casos a proposta é um lote único, e os cortes detectados ficam como observação.

Os QCs dos quatro modos estão nas injeções 1, 12, 23, 34, 45, 56 e 68. Os cortes de cada caso são
os que a detecção devolve nos arquivos reais: 23 e 45 no `pos_hilic`, 56 no `neg_hilic` e 34 no
`neg_c18`. O `pos_c18` não tem corte.
"""

from __future__ import annotations

import numpy as np
import pytest

from refino.diagnosis.batches import (
    MIN_QC_PER_PROPOSED_BATCH,
    batch_without_training,
    propose_batches,
    propose_supported_batches,
)
from refino.domain.matrix import RefinoMatrix
from refino.io.readers import load_matrix
from refino.metrics.crossval import MIN_TRAIN_QC_PER_BATCH
from tests.conftest import measurements_path, sample_list_path


def _matriz(modo: str) -> RefinoMatrix:
    return load_matrix(measurements_path(modo), sample_list_path(modo), mode=modo)


def test_o_limiar_e_tres_e_sai_do_minimo_da_medicao_fora_da_amostra() -> None:
    """Três QCs: um retido e dois para ajustar, a mesma constante com que o Palpite recusa."""
    assert MIN_QC_PER_PROPOSED_BATCH == MIN_TRAIN_QC_PER_BATCH + 1 == 3


@pytest.mark.parametrize(
    ("qcs", "sem_treino"),
    [
        ((2, 2, 3), True),
        ((5, 2), True),
        ((3, 4), False),
        ((7,), False),
        # Lote sem QC não conta: o corretor o salta, e reter um QC não o afeta.
        ((3, 0, 4), False),
        ((1,), True),
    ],
)
def test_o_predicado_olha_so_os_lotes_com_qc(qcs: tuple[int, ...], sem_treino: bool) -> None:
    assert batch_without_training(qcs) is sem_treino


@pytest.mark.parametrize(
    ("modo", "cortes", "qcs_detectados"),
    [("pos_hilic", [23, 45], (2, 2, 3)), ("neg_hilic", [56], (5, 2))],
)
def test_cortes_que_deixam_lote_com_dois_qcs_viram_lote_unico(
    modo: str, cortes: list[int], qcs_detectados: tuple[int, ...]
) -> None:
    """A proposta recua para um lote, e os cortes e as contagens detectados continuam no retorno."""
    matriz = _matriz(modo)

    proposta = propose_supported_batches(matriz, cortes)

    assert proposta.detected_qcs_per_batch == qcs_detectados
    assert proposta.detected_changepoints == tuple(cortes)
    assert proposta.changepoints == ()
    assert proposta.fell_back is True
    assert proposta.labels == (0,) * matriz.n_samples


def test_cortes_que_sustentam_a_medicao_entram_como_vieram() -> None:
    """O par do teste de cima: a mesma função, com cortes que deixam três e quatro QCs, não recua.

    E os rótulos são os de `propose_batches`, que continua a ser a tradução dos cortes.
    """
    matriz = _matriz("neg_c18")

    proposta = propose_supported_batches(matriz, [34])

    assert proposta.detected_qcs_per_batch == (3, 4)
    assert proposta.changepoints == proposta.detected_changepoints == (34,)
    assert proposta.fell_back is False
    assert proposta.labels == tuple(int(r) for r in propose_batches(matriz, [34]))
    assert len(set(proposta.labels)) == 2


def test_sem_corte_detectado_a_proposta_e_um_lote_sem_recuo() -> None:
    matriz = _matriz("pos_c18")

    proposta = propose_supported_batches(matriz, [])

    assert proposta.changepoints == proposta.detected_changepoints == ()
    assert proposta.detected_qcs_per_batch == (int(np.count_nonzero(matriz.qc_mask)),)
    assert proposta.fell_back is False
    assert set(proposta.labels) == {0}

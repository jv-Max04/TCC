"""Os lotes declarados e os QCs de cada um.

A partição vive em três lugares: `domain/lotes.py`, que a janela usa para contar, e as duas cópias
que a correção usa (`api/rotas.py:_lotes` e `multiverse/executor.py:_batches_for`). O oráculo é a
IGUALDADE entre os três, sobre os arquivos reais: se uma cópia mudar, a conta que a tela mostra
deixa de ser a da correção, e é isso que este arquivo acusa.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from refino.api.rotas import _lotes
from refino.domain.lotes import confere_cortes, lotes_declarados, qcs_por_lote_declarado
from refino.io.readers import load_matrix
from refino.multiverse.executor import _batches_for
from tests.conftest import MODES, measurements_path, sample_list_path


@pytest.mark.parametrize("modo", MODES)
def test_a_particao_e_a_mesma_das_duas_copias_da_correcao(modo: str) -> None:
    matriz = load_matrix(measurements_path(modo), sample_list_path(modo), modo)
    for n in range(1, 9):
        nossa = lotes_declarados(matriz.injection_orders, n)
        rota, multiverso = _lotes(matriz, n), _batches_for(matriz, n)
        if n == 1:
            assert nossa is None and rota is None and multiverso is None
            continue
        # A âncora: a partição existe e tem `n` lotes, senão a igualdade abaixo seria de vazios.
        assert nossa is not None and len(set(nossa.tolist())) == n, (modo, n)
        assert rota is not None and multiverso is not None
        assert np.array_equal(nossa, rota), f"{modo}, {n} lotes: a janela e a rota divergem"
        assert np.array_equal(nossa, multiverso), f"{modo}, {n} lotes: a janela e o multiverso"


def test_os_qcs_por_lote_no_pos_hilic_com_tres_lotes() -> None:
    """O caso do testador de 08/10/2026: 3 lotes declarados sobre os 7 QCs do `pos_hilic`.

    `(3, 2, 2)` é contagem esperada sobre ESTE arquivo (regra 1), e a soma é a âncora: os sete QCs
    da máscara têm de estar todos em algum lote.
    """
    matriz = load_matrix(measurements_path("pos_hilic"), sample_list_path("pos_hilic"), "pos_hilic")
    contagens = qcs_por_lote_declarado(matriz.injection_orders, matriz.qc_mask, 3)
    assert sum(contagens) == int(matriz.qc_mask.sum())
    assert contagens == (3, 2, 2)


def test_lote_sem_qc_aparece_como_zero_e_um_lote_da_o_total() -> None:
    ordens = np.array([6, 1, 2, 3, 4, 5], dtype=np.int64)
    qc = np.array([False, True, False, False, False, False])
    assert qcs_por_lote_declarado(ordens, qc, 3) == (1, 0, 0)
    assert qcs_por_lote_declarado(ordens, qc, 1) == (1,)
    assert qcs_por_lote_declarado(ordens, qc, 0) == (1,)


# ---------------------------------------------------------------------------
# Os CORTES declarados
# ---------------------------------------------------------------------------
#
# **Sem arquivo de referência, de propósito:** a pergunta é de convenção — onde cai a fronteira —, e
# a resposta decorre da entrada. As ordens de injeção são dadas aqui, e as contagens esperadas saem
# delas; nada abaixo afirma o que os arquivos do laboratório contêm (regra 1).

# As posições dos sete QCs são as dos quatro arquivos do laboratório, usadas aqui como ENTRADA: o
# que se afirma é a partição destas ordens, e não que um arquivo as tenha.
QCS = (1, 12, 23, 34, 45, 56, 68)
# Colunas fora da ordem de injeção, como nos arquivos: a partição tem de seguir as ordens e
# não a posição da coluna.
ORDENS = np.array(list(range(68, 34, -1)) + list(range(1, 35)), dtype=np.int64)
QC = np.isin(ORDENS, QCS)


def test_d557_os_cortes_dao_a_particao_da_proposta_do_diagnostico() -> None:
    """O outro caminho como oráculo: `propose_batches` é de onde os cortes vêm.

    A janela manda os cortes que o diagnóstico devolveu, e a correção parte a corrida com
    `lotes_declarados`. As duas convenções têm de ser a mesma — cada corte é a primeira injeção do
    lote novo —, senão a amostra da fronteira muda de lote entre a proposta e a correção.
    """
    from types import SimpleNamespace

    from refino.diagnosis.batches import propose_batches

    for cortes in ([56], [23, 45], [12, 34, 56], [2], [68]):
        nossa = lotes_declarados(ORDENS, len(cortes) + 1, cortes)
        matriz = SimpleNamespace(injection_orders=ORDENS)
        proposta = propose_batches(matriz, cortes)  # type: ignore[arg-type]
        # A âncora: a partição existe e tem um lote por corte mais um, ou a igualdade seria vácua.
        assert nossa is not None and len(set(nossa.tolist())) == len(cortes) + 1, cortes
        assert np.array_equal(nossa, proposta), f"cortes {cortes}: a correção e a proposta divergem"
        # E a convenção, por extenso: a ordem do corte abre o lote, a anterior fica no de antes.
        for k, corte in enumerate(cortes):
            assert nossa[corte == ORDENS][0] == k + 1, (cortes, corte)
            assert nossa[corte - 1 == ORDENS][0] == k, (cortes, corte)


def test_d557_os_qcs_por_lote_nos_cortes_e_nos_blocos_iguais() -> None:
    """A diferença que o teste existe para levar à correção, contada sobre as ordens acima.

    Dois lotes: o corte em 56 dá 5 e 2 QCs, e os blocos iguais dão 4 e 3 — o degrau entre 45 e 56
    fica DENTRO do segundo bloco. Três lotes: os cortes em 23 e 45 dão 2, 2 e 3, e os blocos iguais
    3, 2 e 2. A soma é a âncora: os sete QCs estão todos em algum lote.
    """
    assert qcs_por_lote_declarado(ORDENS, QC, 2, [56]) == (5, 2)
    assert qcs_por_lote_declarado(ORDENS, QC, 2) == (4, 3)
    assert qcs_por_lote_declarado(ORDENS, QC, 3, [23, 45]) == (2, 2, 3)
    assert qcs_por_lote_declarado(ORDENS, QC, 3) == (3, 2, 2)
    assert sum(qcs_por_lote_declarado(ORDENS, QC, 3, [23, 45])) == len(QCS)


@pytest.mark.parametrize(
    ("cortes", "n_lotes", "trecho"),
    [
        ([56], 3, "pedem 2 corte(s)"),
        ([45, 23], 3, "estritamente crescente"),
        ([23, 23], 3, "estritamente crescente"),
        ([True], 2, "inteiras"),
        ([56.0], 2, "inteiras"),
        (["56"], 2, "inteiras"),
    ],
)
def test_d557_a_forma_errada_dos_cortes_levanta(
    cortes: list[object], n_lotes: int, trecho: str
) -> None:
    """Fronteira de confiança (regra 2): os cortes chegam de JSON, e `True` é `int` em Python."""
    with pytest.raises(ValueError, match=re.escape(trecho)):
        confere_cortes(cortes, n_lotes)  # type: ignore[arg-type]


def test_d557_a_forma_certa_vira_tupla_de_int_nativo() -> None:
    """`np.int64` vira `int`: a linha da máquina e a do humano são comparadas como TEXTO JSON."""
    assert confere_cortes([np.int64(23), 45], 3) == (23, 45)
    assert all(type(c) is int for c in confere_cortes([np.int64(23), 45], 3))
    assert confere_cortes([], 1) == ()


def test_d557_corte_fora_da_corrida_levanta_e_a_corrida_nao_precisa_comecar_em_1() -> None:
    """A recusa é a de `propose_batches` — na primeira injeção ou além da última —, e é RELATIVA.

    O leitor aceita qualquer inteiro na linha `time`; começar em 1 é observação sobre os arquivos
    do laboratório. Numa corrida que começa em 0, o corte em 1 deixa a injeção 0 sozinha no
    primeiro lote, e isso é partição válida.
    """
    with pytest.raises(ValueError, match="fora do intervalo"):
        lotes_declarados(ORDENS, 2, [69])
    with pytest.raises(ValueError, match="fora do intervalo"):
        lotes_declarados(ORDENS, 2, [1])
    with pytest.raises(ValueError, match="lote único não tem corte"):
        lotes_declarados(ORDENS, 1, [56])

    desde_zero = np.arange(0, 10, dtype=np.int64)
    rotulos = lotes_declarados(desde_zero, 2, [1])
    assert rotulos is not None and rotulos.tolist() == [0] + [1] * 9


def test_d557_a_spec_confere_os_cortes_e_o_padrao_nao_muda_a_impressao_digital() -> None:
    """Os dois lados do campo novo em `PipelineSpec`.

    **Com `None`, a impressão digital é a de antes do campo:** `e1914b114ddb8c59` foi medida na
    árvore anterior (`29e8aeb`) para esta mesma especificação, e mudar todas as impressões
    invalidaria o cache do multiverso por uma mudança sem efeito nenhum. **Com cortes, ela muda**:
    duas correções com fronteiras diferentes não são a mesma célula.
    """
    from refino.domain.spec import PipelineSpec

    campos: dict[str, object] = {
        "correction": "loess",
        "n_batches": 2,
        "rsd_cutoff": None,
        "ranking_filter": "none",
        "ranking_keep_fraction": None,
        "ranking_filter_2": "none",
        "ranking_keep_fraction_2": None,
        "rsd_variant": "outlier_removed",
        "missing_threshold": None,
        "imputation": "one_fifth_min",
        "sample_normalization": "none",
        "normalization_reference": None,
        "normalization_factors": None,
        "transformation": "none",
        "scaling": "none",
    }
    sem = PipelineSpec(**campos)  # type: ignore[arg-type]
    assert sem.batch_cuts is None
    assert sem.fingerprint() == "e1914b114ddb8c59"

    com = PipelineSpec(**campos, batch_cuts=[np.int64(56)])  # type: ignore[arg-type]
    assert com.batch_cuts == (56,) and type(com.batch_cuts[0]) is int
    assert com.fingerprint() != sem.fingerprint()
    assert PipelineSpec(**campos, batch_cuts=(45,)).fingerprint() != com.fingerprint()  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="pedem 1 corte"):
        PipelineSpec(**campos, batch_cuts=(23, 45))  # type: ignore[arg-type]

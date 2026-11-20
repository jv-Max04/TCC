"""Corretor `none` e a fronteira comum.

`none` é opção legítima de correção, não caso degenerado (achado 4.5): no
`neg_c18` corrigir tem tanta chance de piorar quanto de melhorar. O multiverso da
etapa seguinte o consome como ramo de primeira classe, então ele satisfaz o mesmo
protocolo que os outros dois.

**A asserção com dente é o mapa de causas vazio.** Um corretor que marcasse
**todas** as features como não corrigidas passaria em tudo o mais deste módulo —
valores idênticos, objeto novo, procedência estendida — e estaria errado sobre a
única coisa que o critério cobra dele.
"""

import numpy as np
import pytest

from refino.correction.base import restore_zeros_and_missing
from refino.correction.none import NoneCorrector
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata


def _samples(n: int = 4) -> tuple[SampleMetadata, ...]:
    return tuple(
        SampleMetadata(
            name=f"S{i + 1}",
            injection_order=i + 1,
            sample_type="qc" if i in (0, 3) else "sample",
            group="QC" if i in (0, 3) else "NoHTLV",
        )
        for i in range(n)
    )


def _matrix() -> RefinoMatrix:
    """Amostras em linhas, features em colunas — a orientação canônica interna."""
    values = np.array(
        [
            [10.0, 0.0, 30.0],
            [np.nan, 20.0, 0.0],
            [5.0, 6.0, 7.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float64,
    )
    return RefinoMatrix(
        values=values,
        feature_names=("F1", "F2", "F3"),
        samples=_samples(),
        mode="tiny",
    )


# ---------------------------------------------------------------------------
# O corretor
# ---------------------------------------------------------------------------


def test_ca_3_5_valores_sao_bit_identicos() -> None:
    """`none` não corrige: nem uma célula muda."""
    matrix = _matrix()
    result = NoneCorrector().fit_transform(matrix)
    assert np.array_equal(result.matrix.values, matrix.values, equal_nan=True)
    assert result.matrix.feature_names == matrix.feature_names
    assert result.matrix.samples == matrix.samples


def test_ca_3_5_o_mapa_de_causas_e_vazio() -> None:
    """**A asserção com dente.**

    Vazio, não `None` e não a matriz inteira. `none` não *tenta e desiste* — ele
    não tenta, e "tentou e desistiu" é o que `uncorrected` significa. Um
    corretor que marcasse tudo passaria em todos os outros testes daqui.
    """
    result = NoneCorrector().fit_transform(_matrix())
    assert result.uncorrected == {}
    assert not result.uncorrected_mask.any()
    assert result.uncorrected_mask.size == result.matrix.n_features


def test_o_objeto_devolvido_e_novo_e_a_entrada_nao_e_mutada() -> None:
    """Regra 7 do README: a imutabilidade alcança o array, não só o atributo.

    As duas metades importam. Objeto novo com o **mesmo** array deixaria um
    corretor a jusante alcançar a matriz de quem chamou; e mutar a entrada é o
    defeito que a regra nomeia.
    """
    matrix = _matrix()
    snapshot = np.array(matrix.values, copy=True)
    result = NoneCorrector().fit_transform(matrix)

    assert result.matrix is not matrix
    assert result.matrix.values is not matrix.values
    assert np.array_equal(matrix.values, snapshot, equal_nan=True)

    # E escrever no resultado não alcança a entrada.
    result.matrix.values.setflags(write=True)
    result.matrix.values[0, 0] = -999.0
    assert np.array_equal(matrix.values, snapshot, equal_nan=True)


def test_a_procedencia_ganha_exatamente_uma_entrada() -> None:
    """Uma, e nomeando o método — procedência é o que torna o multiverso legível."""
    matrix = _matrix()
    result = NoneCorrector().fit_transform(matrix)
    assert len(result.matrix.provenance) == len(matrix.provenance) + 1
    assert "none" in result.matrix.provenance[-1]


def test_duas_execucoes_sao_bit_identicas() -> None:
    """Regra 3 do README, e aqui é barato: `none` não tem sorteio."""
    matrix = _matrix()
    a = NoneCorrector().fit_transform(matrix, seed=1)
    b = NoneCorrector().fit_transform(matrix, seed=999)
    assert np.array_equal(a.matrix.values, b.matrix.values, equal_nan=True)
    assert a.matrix.provenance == b.matrix.provenance


def test_o_protocolo_aceita_lote_sem_mudar_o_resultado() -> None:
    """`batches` está na assinatura por contrato do protocolo, e `none` o ignora.

    Ignorar tem de ser **verificado**: um `none` que ramificasse por lote sem
    querer produziria resultados diferentes conforme o chamador, e a comparação
    entre métodos do multiverso deixaria de ter linha de base fixa.
    """
    matrix = _matrix()
    sem = NoneCorrector().fit_transform(matrix)
    com = NoneCorrector().fit_transform(matrix, batches=np.array(["A", "B", "A", "B"]))
    assert np.array_equal(sem.matrix.values, com.matrix.values, equal_nan=True)


# ---------------------------------------------------------------------------
# A fronteira: passo 11
# ---------------------------------------------------------------------------


def test_o_passo_11_devolve_zero_a_zero_e_ausente_a_ausente() -> None:
    """O equipamento deixa a célula vazia quando não detecta.

    Correção nenhuma transforma não-detecção em medida — uma feature que fosse de
    zero a valor positivo estaria inventando sinal.
    """
    original = np.array([[0.0, 10.0, np.nan], [5.0, 0.0, 2.0]])
    corrected = np.array([[7.0, 12.0, 3.0], [6.0, 8.0, 2.5]])
    out = restore_zeros_and_missing(original, corrected)

    assert out.values[0, 0] == 0.0
    assert np.isnan(out.values[0, 2])
    assert out.values[1, 1] == 0.0
    assert out.values[0, 1] == 12.0
    assert out.values[1, 0] == 6.0
    assert out.restored == 2
    assert out.missing_restored == 1


def test_o_passo_11_preserva_os_zeros_que_a_correcao_criou() -> None:
    """**Restaurado e criado são contagens diferentes, e a distinção decide ação.**

    Zero criado é resultado do método — no `batch_loess`, divisão por `Inf`
    —, não ausência de detecção. Apagá-lo reescreveria a saída do corretor; contá-lo
    junto com os restaurados faria um método que **cria** zeros parecer um que os
    preserva: um par de números que a leitura natural colapsa.
    """
    original = np.array([[10.0, 20.0, 0.0]])
    corrected = np.array([[0.0, 20.0, 0.0]])
    out = restore_zeros_and_missing(original, corrected)

    assert out.values[0, 0] == 0.0, "zero criado pela correção tem de sobreviver"
    assert out.created == 1
    assert out.restored == 1


def test_o_passo_11_nao_toca_posicao_que_era_positiva() -> None:
    """Sem zeros nem ausentes, o passo é identidade — o controle negativo."""
    original = np.array([[1.0, 2.0], [3.0, 4.0]])
    corrected = np.array([[1.5, 2.5], [3.5, 4.5]])
    out = restore_zeros_and_missing(original, corrected)
    assert np.array_equal(out.values, corrected)
    assert out.restored == 0
    assert out.created == 0
    assert out.missing_restored == 0


def test_o_passo_11_nao_muta_a_entrada() -> None:
    """Regra 7 do README também aqui: o array corrigido de quem chamou não é reescrito."""
    original = np.array([[0.0, 10.0]])
    corrected = np.array([[7.0, 12.0]])
    snapshot = np.array(corrected, copy=True)
    restore_zeros_and_missing(original, corrected)
    assert np.array_equal(corrected, snapshot)


def test_o_passo_11_reprova_formas_diferentes() -> None:
    """Comparação posição a posição exige a mesma forma.

    Sem a guarda, o *broadcast* do numpy compararia matrizes de tamanhos
    diferentes em silêncio e devolveria uma terceira forma — que é o defeito de
    eixo trocado chegando pela porta do passo 11.
    """
    with pytest.raises(ValueError, match="posição a posição"):
        restore_zeros_and_missing(np.zeros((2, 3)), np.zeros((3, 2)))

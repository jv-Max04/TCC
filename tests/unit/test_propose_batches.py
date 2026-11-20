"""`propose_batches` sobre matrizes construídas, onde POSIÇÃO e ORDEM divergem.

**Existe porque o teste de integração não pode separar as duas coisas.** A especificação declara
`RefinoMatrix.samples` ordenado por `injection_order`, e a ingestão ordena: nos quatro
arquivos reais `injection_orders` é `1..68`, então "rotular pela ordem da amostra" e
"rotular pela posição dela" dão o mesmo vetor e nenhum dado real os distingue.

Aqui as ordens são **não contíguas** - 10, 20, 30... -, e aí um corte em 35 rotula pela
ordem e não pela posição. É o mesmo cuidado do negativo de `detect_changepoints`: um
fixture com ordens `0..n-1` faria as duas convenções coincidir e o teste passaria sem
provar nada.
"""

import numpy as np
import pytest

from refino.diagnosis.batches import propose_batches
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata


def _matriz(ordens: tuple[int, ...]) -> RefinoMatrix:
    """Matriz mínima com as ordens de injeção pedidas."""
    amostras = tuple(
        SampleMetadata(
            name=f"S{ordem}",
            injection_order=ordem,
            sample_type="qc" if i % 3 == 0 else "sample",
            group="QC" if i % 3 == 0 else "G",
        )
        for i, ordem in enumerate(ordens)
    )
    return RefinoMatrix(
        values=np.arange(len(ordens) * 4, dtype=np.float64).reshape(len(ordens), 4) + 1.0,
        feature_names=("f0", "f1", "f2", "f3"),
        samples=amostras,
        mode="sintetico",
    )


ORDENS_ESPACADAS = (10, 20, 30, 40, 50, 60)


def test_sem_cortes_ha_um_lote_so() -> None:
    """Nenhum ponto de mudança: partição trivial, todas as amostras juntas."""
    lotes = propose_batches(_matriz(ORDENS_ESPACADAS), [])
    assert lotes.tolist() == [0, 0, 0, 0, 0, 0]


def test_um_corte_da_dois_lotes_pela_ordem_e_nao_pela_posicao() -> None:
    """**O caso que separa as duas convenções, e é a razão deste arquivo existir.**

    Ordens 10..60 e corte em 35. Pela ordem, as amostras de ordem 10, 20 e 30 ficam no
    lote 0 e as de 40, 50 e 60 no lote 1 - divisão 3/3. Pela **posição**, um corte no
    índice 35 não existe, e uma implementação que confundisse os eixos devolveria tudo
    num lote só ou levantaria.
    """
    lotes = propose_batches(_matriz(ORDENS_ESPACADAS), [35])
    assert lotes.tolist() == [0, 0, 0, 1, 1, 1]


def test_o_corte_e_a_primeira_injecao_do_lote_novo() -> None:
    """Corte na ordem 30: a amostra de ordem 30 fica no lote NOVO, não no anterior.

    **A convenção vem do que o índice de corte significa**, e não de preferência.
    `detect_changepoints` devolve `ordens[k]` onde `k` parte a trajetória em
    `valores[:k]` e `valores[k:]` - então o QC de índice `k` é o **primeiro elemento do
    segundo segmento**, e a ordem de injeção dele pertence ao lote novo.

    No `neg_hilic` isso é verificável contra os dados: o QC da injeção 45 está no nível
    antigo (+7) e o da 56 no novo (+49), e o corte devolvido é 56. Se a convenção fosse
    a oposta, a injeção 56 cairia no lote de baixo junto com os pontos de nível antigo -
    e o segundo lote começaria no meio do degrau.

    A primeira versão deste teste afirmava o contrário, por convenção que eu descrevi
    errado no próprio docstring dele enquanto a implementação estava certa. Fixar a
    convenção em teste é o que impede o off-by-one de virar escolha implícita da próxima
    pessoa - e o valor está em ela ser derivada, não escolhida.
    """
    lotes = propose_batches(_matriz(ORDENS_ESPACADAS), [30])
    assert lotes.tolist() == [0, 0, 1, 1, 1, 1]


def test_dois_cortes_dao_tres_lotes() -> None:
    """`k` cortes dão `k + 1` lotes."""
    lotes = propose_batches(_matriz(ORDENS_ESPACADAS), [25, 45])
    assert lotes.tolist() == [0, 0, 1, 1, 2, 2]
    assert len(set(lotes.tolist())) == 3


def test_cortes_fora_de_ordem_sao_ordenados() -> None:
    """A lista de entrada não precisa vir ordenada; o resultado é o mesmo."""
    assert propose_batches(_matriz(ORDENS_ESPACADAS), [45, 25]).tolist() == [0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize("fora", [[5], [10], [61], [100], [25, 999]])
def test_corte_fora_do_intervalo_levanta(fora: list[int]) -> None:
    """Lote sem amostra nenhuma não é partição, e o corretor o saltaria em silêncio.

    A borda inferior é exclusiva: corte na **primeira** ordem deixaria o lote 0 vazio.
    """
    with pytest.raises(ValueError, match="fora do intervalo"):
        propose_batches(_matriz(ORDENS_ESPACADAS), fora)


def test_o_comprimento_e_o_numero_de_amostras() -> None:
    """Um rótulo por amostra, alinhado a `matrix.samples`."""
    matriz = _matriz(ORDENS_ESPACADAS)
    assert propose_batches(matriz, [35]).size == matriz.n_samples


def test_os_rotulos_sao_deterministicos() -> None:
    """Regra 3 do README - duas chamadas com a mesma entrada dão o mesmo vetor."""
    matriz = _matriz(ORDENS_ESPACADAS)
    assert np.array_equal(propose_batches(matriz, [35]), propose_batches(matriz, [35]))

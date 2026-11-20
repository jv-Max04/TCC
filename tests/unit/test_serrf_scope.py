"""A âncora dos passos 5 a 9 é GLOBAL, e é isto que dois lotes tornam visível.

**Por que este arquivo é obrigatório e não zeloso.** No `tools.R`, `e[j, …]` é o
lote corrente e `all[j, …]` é a matriz inteira depois do passo 1. Com **um** lote
os dois são o mesmo vetor, então nenhum teste sobre os quatro arquivos do
laboratório — todos de lote único — poderia distinguir as duas leituras. Foi assim
que o defeito de escopo sobreviveu a duas revisões da especificação: não faltou
revisão, faltou **configuração em que a diferença aparecesse**.

**As duas metades, e a segunda é a que tem dente.** A primeira afirma que as
funções puras dão números **diferentes** quando alimentadas com âncora local em vez
de global — sem ela, o resto poderia passar sobre uma matriz em que as duas
coincidem por acidente. A segunda é a assinatura observável da âncora global no
corretor inteiro: **mexer no lote B muda a saída do lote A**. Se a âncora fosse
local, isso seria impossível — e é uma afirmação sobre o corretor, não sobre as
funções que ele chama.
"""

import numpy as np
import pytest

from refino.correction._serrf_steps import (
    normalize_qc,
    normalize_samples,
    rescale_by_median,
)
from refino.correction.serrf import SerrfCorrector
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata

# Dois lotes com magnitudes deliberadamente distintas: é a assimetria entre eles
# que separa a âncora global da local. Com lotes de mesma escala o teste passaria
# nas duas leituras, que é exatamente o defeito que ele existe para pegar.
BATCH_A_SCALE = 100.0
BATCH_B_SCALE = 900.0


def _samples(batch_of: list[str], qc_at: set[int]) -> tuple[SampleMetadata, ...]:
    return tuple(
        SampleMetadata(
            name=f"S{i + 1}",
            injection_order=i + 1,
            sample_type="qc" if i in qc_at else "sample",
            group="QC" if i in qc_at else "NoHTLV",
            batch=batch,
        )
        for i, batch in enumerate(batch_of)
    )


def _two_batch_matrix(*, b_scale: float = BATCH_B_SCALE, n_features: int = 14) -> RefinoMatrix:
    """Lote A pequeno, lote B grande. 6 QCs por lote para a floresta poder dividir."""
    rng = np.random.default_rng(11)
    per_batch = 16
    batch_of = ["A"] * per_batch + ["B"] * per_batch
    qc_at = {0, 3, 6, 9, 12, 15, 16, 19, 22, 25, 28, 31}

    scales = np.array([BATCH_A_SCALE] * per_batch + [b_scale] * per_batch)
    values = scales[:, None] * (1.0 + 0.08 * rng.standard_normal((2 * per_batch, n_features)))
    return RefinoMatrix(
        values=values,
        feature_names=tuple(f"F{i + 1}" for i in range(n_features)),
        samples=_samples(batch_of, qc_at),
        mode="tiny",
    )


# ---------------------------------------------------------------------------
# Primeira metade: local e global dão números DIFERENTES
# ---------------------------------------------------------------------------


def test_a_matriz_de_dois_lotes_separa_as_duas_ancoras() -> None:
    """Sem esta asserção o resto do arquivo poderia passar vacuamente.

    Se âncora local e global coincidissem nesta matriz, todo teste abaixo passaria
    nas duas leituras e o arquivo seria decorativo — o defeito aplicado a
    um fixture em vez de a um portão.
    """
    matrix = _two_batch_matrix()
    values = np.asarray(matrix.values, dtype=np.float64)
    is_qc = np.asarray(matrix.qc_mask, dtype=bool)
    in_a = np.asarray([s.batch == "A" for s in matrix.samples], dtype=bool)

    column = values[:, 0]
    local_qc = column[in_a & is_qc]
    global_qc = column[is_qc]

    assert not np.isclose(np.mean(local_qc), np.mean(global_qc), rtol=0.2), (
        "a média dos QCs do lote A é próxima da dos QCs de todos os lotes; "
        "esta matriz não separa as âncoras e o arquivo não prova nada"
    )


@pytest.mark.parametrize(
    ("function", "name"),
    [(normalize_qc, "passo 5"), (normalize_samples, "passo 6")],
)
def test_passos_5_e_6_dependem_da_ancora_global(function: object, name: str) -> None:
    """Alimentar com âncora local em vez de global muda o resultado.

    É a asserção que dá conteúdo ao contrato: as funções recebem os dois vetores
    **separadamente** justamente para que a troca seja impossível de cometer sem
    editar a chamada.
    """
    local = np.array([100.0, 105.0, 98.0, 102.0, 99.0, 101.0])
    predicted = np.array([1.0, -2.0, 0.5, 1.5, -1.0, 0.0])
    global_vector = np.concatenate([local, np.array([900.0, 890.0, 910.0, 905.0])])

    with_global = function(local, predicted, global_vector)  # type: ignore[operator]
    with_local = function(local, predicted, local)  # type: ignore[operator]

    assert not np.allclose(with_global, with_local), (
        f"{name}: âncora global e local deram o mesmo resultado; a distinção entre as duas "
        "não está sendo exercitada"
    )


def test_passo_7_depende_da_ancora_global() -> None:
    """O divisor de referência do passo 7 é global e **pós-passo-1**."""
    normalized = np.array([100.0, 105.0, 98.0, 102.0])
    global_vector = np.concatenate([normalized, np.array([900.0, 890.0, 910.0])])

    assert not np.allclose(
        rescale_by_median(normalized, global_vector),
        rescale_by_median(normalized, normalized),
    ), "passo 7: a referência local e a global coincidiram"


# ---------------------------------------------------------------------------
# Segunda metade: a assinatura observável no corretor inteiro
# ---------------------------------------------------------------------------


def test_mexer_no_lote_b_muda_a_saida_do_lote_a() -> None:
    """**A metade com dente, e ela é sobre o corretor e não sobre as funções.**

    Com âncora **global**, a saída do lote A depende dos valores do lote B, porque
    `mean(all[j, qc])` e `median(all[j, sample])` somam os dois. Com âncora
    **local**, a saída de A não poderia mudar de jeito nenhum ao mexer só em B.

    Uma implementação que passasse `local` no lugar de `global` produziria saída
    **idêntica** nas duas matrizes aqui, e este teste reprovaria — que é o único
    modo de verificar o escopo sem reimplementar os passos dentro do teste.
    """
    corrector = SerrfCorrector()
    in_a = None

    first = _two_batch_matrix(b_scale=BATCH_B_SCALE)
    second = _two_batch_matrix(b_scale=BATCH_B_SCALE * 3.0)
    in_a = np.asarray([s.batch == "A" for s in first.samples], dtype=bool)

    # O lote A é bit-idêntico nas duas matrizes: só B mudou.
    assert np.array_equal(np.asarray(first.values)[in_a], np.asarray(second.values)[in_a]), (
        "a construção mudou o lote A; o teste perdeu o controle"
    )

    out_first = corrector.fit_transform(first, seed=5).matrix.values[in_a]
    out_second = corrector.fit_transform(second, seed=5).matrix.values[in_a]

    assert not np.allclose(out_first, out_second, equal_nan=True), (
        "mudar SÓ o lote B não alterou a saída do lote A. Isso significa que a "
        "âncora dos passos 5 a 9 está local, e é o defeito da âncora local de volta — "
        "invisível em qualquer teste de lote único."
    )


def test_lote_unico_nao_distingue_as_ancoras_e_e_por_isso_que_isto_existe() -> None:
    """O controle que explica o arquivo: com um lote, local **é** global.

    Documenta mecanicamente por que os quatro arquivos do laboratório não podiam
    pegar o defeito — e por que remover este módulo devolveria a cegueira.
    """
    column = np.array([100.0, 105.0, 98.0, 102.0, 99.0, 101.0])
    predicted = np.zeros_like(column)

    assert np.allclose(
        normalize_qc(column, predicted, column),
        normalize_qc(column, predicted, column),
    )
    # E o ponto: com lote único o vetor global É o local, então as duas chamadas
    # que o teste anterior distingue tornam-se a mesma chamada.
    assert np.allclose(
        normalize_qc(column, predicted, column),
        normalize_qc(column, predicted, np.array(column, copy=True)),
    )

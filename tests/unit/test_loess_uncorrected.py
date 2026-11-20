"""Marcação por feature do `batch_loess`: causas, modos e exaustividade.

**O `negative_fitted_curve` é exercitado sobre DADO REAL, não sintético.** Medido
no `pos_hilic`: de 25 features amostradas entre as **1.561 que o R devolveu sem
correção**, 12 disparam a causa; de 25 amostradas entre as corrigidas, 1 dispara.
O cenário construído à mão seria mais barato e provaria menos — aqui há oráculo
por trás da população amostrada, e a concordância direcional com o R é ela própria
uma asserção.

**Os dois caminhos degenerados produzem artefatos opostos, e só um é causa.**

- `curva == 0` -> divisor `0` -> saída `Inf`. **É causa marcada**:
  `null_denominator`, só no modo corrigido.
- `mediana == 0` -> divisor `Inf` -> saída `0`. **Não é causa**: é o mecanismo
  pelo qual o `batch_loess` **cria** zeros.

Confundi-los é fácil porque a divisão é uma só, e o `pos_hilic` real dispara o
segundo. Um teste que só afirmasse "há não finitos" passaria com os dois trocados.
"""

from typing import get_args

import numpy as np
import pytest

from refino.correction.loess import LoessCorrector
from refino.domain.correction import CorrectionResult
from refino.domain.labels import UncorrectedCause
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from tests.conftest import matrix_from_normalized

# As duas causas que esta unidade introduz, mais a do modo corrigido. A lista
# existe para ser comparada contra `get_args(UncorrectedCause)` em vez de contra
# uma expectativa reescrita à mão.
CAUSES_FROM_LOESS: tuple[UncorrectedCause, ...] = (
    "negative_fitted_curve",
    "fit_failed",
    "null_denominator",
)


def _samples(n: int, *, qc_at: tuple[int, ...]) -> tuple[SampleMetadata, ...]:
    return tuple(
        SampleMetadata(
            name=f"S{i + 1}",
            injection_order=i + 1,
            sample_type="qc" if i in qc_at else "sample",
            group="QC" if i in qc_at else "NoHTLV",
        )
        for i in range(n)
    )


def _matrix(values: np.ndarray, *, qc_at: tuple[int, ...]) -> RefinoMatrix:
    n_samples, n_features = values.shape
    return RefinoMatrix(
        values=values,
        feature_names=tuple(f"F{i + 1}" for i in range(n_features)),
        samples=_samples(n_samples, qc_at=qc_at),
        mode="tiny",
    )


def _subset(matrix: RefinoMatrix, indices: np.ndarray) -> RefinoMatrix:
    return RefinoMatrix(
        values=np.array(matrix.values[:, indices], copy=True),
        feature_names=tuple(matrix.feature_names[j] for j in indices),
        samples=matrix.samples,
        mode=matrix.mode,
    )


@pytest.fixture(scope="module")
def real_sample() -> tuple[RefinoMatrix, np.ndarray, np.ndarray]:
    """`pos_hilic`, mais os índices que o R corrigiu e os que ele não corrigiu.

    A amostragem é **semeada** e pequena de propósito: o corretor faz dezesseis
    seleções de `span` por feature, e as 6.035 do modo inteiro não cabem num teste
    unitário. O que importa é a população de onde se amostra.
    """
    none_matrix = matrix_from_normalized("pos_hilic", "none")
    loess_matrix = matrix_from_normalized("pos_hilic", "batch_loess")
    untouched = np.array(
        [
            np.array_equal(none_matrix.values[:, j], loess_matrix.values[:, j], equal_nan=True)
            for j in range(none_matrix.n_features)
        ]
    )
    rng = np.random.default_rng(0)
    uncorrected = rng.choice(np.flatnonzero(untouched), size=25, replace=False)
    corrected = rng.choice(np.flatnonzero(~untouched), size=25, replace=False)
    return none_matrix, uncorrected, corrected


# ---------------------------------------------------------------------------
# negative_fitted_curve, sobre dado real
# ---------------------------------------------------------------------------


@pytest.mark.golden
def test_ca_3_13_negative_fitted_curve_dispara_no_dado_real(
    real_sample: tuple[RefinoMatrix, np.ndarray, np.ndarray],
) -> None:
    """A causa existe no `pos_hilic` e não precisa de cenário construído.

    Afirma que **dispara**, e não quantas vezes: a contagem depende da semente da
    amostragem e do `span` eleito, e congelá-la seria número sem significado com
    aparência de expectativa.
    """
    matrix, uncorrected_indices, _ = real_sample
    result = LoessCorrector().fit_transform(_subset(matrix, uncorrected_indices), seed=42)
    marked = result.uncorrected.get("negative_fitted_curve", frozenset())
    assert marked, (
        "nenhuma das 25 features que o R deixou sem correção disparou "
        "`negative_fitted_curve`. A causa existe no dado real (medido: 12 de 25) — "
        "se sumiu, o limite de um quinto ou o sinal da curva mudaram."
    )


@pytest.mark.golden
def test_a_causa_e_mais_frequente_onde_o_r_nao_corrigiu(
    real_sample: tuple[RefinoMatrix, np.ndarray, np.ndarray],
) -> None:
    """**Concordância direcional com o oráculo, e é ela que dá dente ao teste.**

    O `negative_fitted_curve` é a nossa marcação para o mesmo gatilho que fez o R
    devolver a feature sem correção (linha 933). Então ele tem de ser mais
    frequente entre as features que o R deixou intactas do que entre as que ele
    corrigiu.

    **Não é igualdade de conjunto**, e não pode ser: o conjunto do R
    é ele próprio estocástico — 32 e 26 contra 20 no `neg_c18`. Direção é o que
    resta verificável, e é mais que zero.
    """
    matrix, uncorrected_indices, corrected_indices = real_sample
    corrector = LoessCorrector()

    left = corrector.fit_transform(_subset(matrix, uncorrected_indices), seed=42)
    right = corrector.fit_transform(_subset(matrix, corrected_indices), seed=42)
    n_left = len(left.uncorrected.get("negative_fitted_curve", frozenset()))
    n_right = len(right.uncorrected.get("negative_fitted_curve", frozenset()))

    assert n_left > n_right, (
        f"{n_left} marcadas entre as que o R não corrigiu contra {n_right} entre as "
        "que ele corrigiu. Sem a diferença de direção, a nossa causa não está "
        "detectando o mesmo gatilho da linha 933."
    )


@pytest.mark.golden
def test_feature_marcada_volta_bit_identica_a_entrada(
    real_sample: tuple[RefinoMatrix, np.ndarray, np.ndarray],
) -> None:
    """Marcada é **devolvida sem correção**, não descartada nem alterada.

    É a metade que distingue "marcou e devolveu o original"
    de "marcou e devolveu qualquer coisa".
    """
    matrix, uncorrected_indices, _ = real_sample
    subset = _subset(matrix, uncorrected_indices)
    result = LoessCorrector().fit_transform(subset, seed=42)

    marked = {f for features in result.uncorrected.values() for f in features}
    assert marked, "o cenário não marcou nada; o teste perdeu o dente"
    for feature in marked:
        index = subset.feature_names.index(feature)
        assert np.array_equal(
            result.matrix.values[:, index], subset.values[:, index], equal_nan=True
        ), f"{feature} marcada mas alterada"


# ---------------------------------------------------------------------------
# Os dois modos de null_denominator
# ---------------------------------------------------------------------------


def _null_curve_matrix() -> RefinoMatrix:
    """Feature cuja curva ajustada é identicamente zero: QCs todos zero.

    Zero em todos os QCs faz a curva ser zero em toda posição, então o divisor é
    zero — o gatilho do denominador nulo, sem depender de sorteio.
    """
    values = np.array(
        [
            [0.0, 10.0],
            [50.0, 12.0],
            [0.0, 11.0],
            [60.0, 15.0],
            [0.0, 14.0],
            [70.0, 18.0],
            [0.0, 17.0],
            [80.0, 20.0],
        ],
        dtype=np.float64,
    )
    return _matrix(values, qc_at=(0, 2, 4, 6))


def test_null_denominator_em_r_faithful_produz_nao_finito() -> None:
    """O modo padrão divide e produz `Inf`/`NaN` onde o R produz.

    **Nenhuma asserção de finitude reprova aqui**, e isso é o ponto: o golden do
    laboratório **tem** os não finitos, e quem os "consertasse" quebraria a
    paridade. A feature **não** é marcada — ela foi corrigida, com o resultado que
    o oráculo dá.
    """
    matrix = _null_curve_matrix()
    result = LoessCorrector(null_denominator="r_faithful").fit_transform(matrix, seed=1)

    assert "null_denominator" not in result.uncorrected
    column = result.matrix.values[:, 0]
    assert not np.all(np.isfinite(column)), (
        f"o modo `r_faithful` tem de produzir não finito onde o divisor é zero; saiu {column}"
    )


def test_null_denominator_em_uncorrected_feature_marca_e_devolve_finito() -> None:
    """O outro modo devolve a feature sem correção, marcada.

    E a saída é **finita em toda célula** dada entrada finita — é o que distingue
    este modo do outro, e a asserção que reprovaria se ele apenas marcasse sem
    devolver o original.
    """
    matrix = _null_curve_matrix()
    result = LoessCorrector(null_denominator="uncorrected_feature").fit_transform(matrix, seed=1)

    assert "null_denominator" in result.uncorrected
    assert "F1" in result.uncorrected["null_denominator"]
    assert np.array_equal(result.matrix.values[:, 0], matrix.values[:, 0], equal_nan=True)


def test_os_dois_modos_dao_resultados_diferentes() -> None:
    """**O par que prova que o modo importa.**

    Sem esta asserção, uma implementação que ignorasse `null_denominator` passaria
    nos dois testes anteriores se por acaso o cenário não disparasse o gatilho.
    """
    matrix = _null_curve_matrix()
    faithful = LoessCorrector(null_denominator="r_faithful").fit_transform(matrix, seed=1)
    marked = LoessCorrector(null_denominator="uncorrected_feature").fit_transform(matrix, seed=1)
    assert not np.array_equal(faithful.matrix.values, marked.matrix.values, equal_nan=True), (
        "os dois modos produziram a mesma matriz; o modo não está sendo usado"
    )
    assert faithful.uncorrected != marked.uncorrected


def test_r_faithful_com_curva_zero_da_infinito_so_onde_havia_medida() -> None:
    """Curva zero com mediana positiva: infinito onde o valor medido não é zero, e ZERO onde ele
    já era zero.

    O passo 5 divide `0/0` nas células zeradas, e o passo 11 (`restore_zeros_and_missing`)
    as devolve a zero: o arquivo não traz célula vazia ali. É o que o resumo dos métodos passou a
    dizer sobre `r_faithful`, e a frase antiga — *«a divisão dá infinito»* — valia só para a
    metade medida.
    """
    matrix = _null_curve_matrix()
    result = LoessCorrector(null_denominator="r_faithful").fit_transform(matrix, seed=1)

    entrada = matrix.values[:, 0]
    saida = result.matrix.values[:, 0]
    medidas = entrada > 0
    assert medidas.any() and (~medidas).any(), "âncora: a feature tem células medidas e zeradas"
    assert np.all(np.isposinf(saida[medidas])), f"a célula medida não deu infinito: {saida}"
    assert np.all(saida[~medidas] == 0.0), f"a célula que era zero não voltou a zero: {saida}"


def _mediana_zero_matrix() -> RefinoMatrix:
    """F1 com os QCs todos zero e uma só amostra medida: curva zero e mediana da linha zero.

    O divisor do passo 5 é `0/0 = NaN` em toda posição, e a guarda de `uncorrected_feature`
    (`divisor == 0.0`) não o vê.
    """
    values = np.array(
        [
            [0.0, 10.0],
            [50.0, 12.0],
            [0.0, 11.0],
            [0.0, 15.0],
            [0.0, 14.0],
            [0.0, 18.0],
            [0.0, 17.0],
            [0.0, 20.0],
        ],
        dtype=np.float64,
    )
    return _matrix(values, qc_at=(0, 2, 4, 6))


@pytest.mark.parametrize("politica", ["r_faithful", "uncorrected_feature"])
def test_curva_zero_com_mediana_zero_sai_vazia_nas_duas_politicas(politica: str) -> None:
    """Curva zero e mediana zero: célula vazia (NaN) onde havia medida, sem causa marcada.

    **Nas duas políticas, e é isso que a frase de `uncorrected_feature` dizia ao contrário:**
    *«a feature cuja curva ajustada vale zero sai sem correção»*. Medido no passeio de
    10/11/2026: 38 features do `neg_c18` saem assim, com células vazias onde havia medida.
    """
    matrix = _mediana_zero_matrix()
    result = LoessCorrector(null_denominator=politica).fit_transform(  # type: ignore[arg-type]
        matrix, seed=1
    )

    saida = result.matrix.values[:, 0]
    assert np.isnan(saida[1]), f"a célula medida não saiu vazia sob {politica}: {saida}"
    assert np.all(saida[np.arange(8) != 1] == 0.0), f"os zeros não voltaram a zero: {saida}"
    assert all("F1" not in nomes for nomes in result.uncorrected.values()), (
        f"sob {politica} a feature foi marcada, e a guarda não a vê: {result.uncorrected}"
    )


def test_modo_invalido_levanta_na_construcao() -> None:
    """`Literal` que atravessa fronteira tem guarda em runtime.

    O modo pode chegar da CLI ou desserializado do `metrics_json`, e ali nada
    passou pelo mypy. Um valor inválido que caísse num `else` razoável se
    comportaria como o outro modo, e o defeito apareceria só como resultado
    diferente do pedido.
    """
    with pytest.raises(ValueError):
        LoessCorrector(null_denominator="r_fiel")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Exaustividade e disjunção — as três asserções
# ---------------------------------------------------------------------------


def test_ca_3_13_as_causas_sao_disjuntas() -> None:
    """Cada feature marcada tem **exatamente uma** causa."""
    matrix = _null_curve_matrix()
    result = LoessCorrector(null_denominator="uncorrected_feature").fit_transform(matrix, seed=1)

    seen: set[str] = set()
    for features in result.uncorrected.values():
        overlap = seen & set(features)
        assert not overlap, f"feature em duas causas: {overlap}"
        seen |= set(features)


def test_ca_3_13_a_uniao_das_causas_e_igual_a_mascara() -> None:
    """A máscara é **derivada** do mapa, e as duas têm de concordar.

    Duas representações da mesma informação derivam, e a máscara é a que se
    reconstrói sem perda — o contrário não vale, porque ela não carrega causa.
    """
    matrix = _null_curve_matrix()
    result = LoessCorrector(null_denominator="uncorrected_feature").fit_transform(matrix, seed=1)
    union = {f for features in result.uncorrected.values() for f in features}
    from_mask = {
        name
        for name, marked in zip(result.matrix.feature_names, result.uncorrected_mask, strict=True)
        if marked
    }
    assert union == from_mask


def test_ca_3_13_toda_causa_emitida_pertence_ao_vocabulario() -> None:
    """Paridade na direção "emitido sem declarar".

    A outra direção — declarado e nunca emitido — tem isenção nomeada: as duas
    causas do SERRF nascem com o SERRF.
    """
    declared = set(get_args(UncorrectedCause))
    assert set(CAUSES_FROM_LOESS) <= declared, (
        f"causas do LOESS fora do vocabulário: {set(CAUSES_FROM_LOESS) - declared}"
    )

    matrix = _null_curve_matrix()
    for policy in ("r_faithful", "uncorrected_feature"):
        result = LoessCorrector(null_denominator=policy).fit_transform(matrix, seed=1)
        for cause in result.uncorrected:
            assert cause in declared, f"causa emitida fora do vocabulário: {cause!r}"


def test_o_detector_de_sobreposicao_consegue_reprovar() -> None:
    """**O teste que prova que a disjunção é verificada**.

    A guarda está em `CorrectionResult`, e sem plantar a violação as três
    asserções acima passariam vacuamente sobre um corretor que nunca marca nada.
    """
    matrix = _matrix(np.ones((4, 2)), qc_at=(0, 2))
    with pytest.raises(ValueError, match="duas causas"):
        CorrectionResult(
            matrix=matrix,
            uncorrected={
                "fit_failed": frozenset({"F1"}),
                "negative_fitted_curve": frozenset({"F1"}),
            },
        )

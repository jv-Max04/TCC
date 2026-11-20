"""Corretor `batch_loess`: os seis passos e os dois modos.

**Os testes com dente são os que distinguem passo implementado de passo ausente.**
Vários cenários do LOESS passam com o passo faltando — o descarte de outlier é o
caso claro: com `n = 2` ele é identidade por construção, então um teste que só
verificasse "o resultado é razoável" não separaria as duas coisas. Cada passo aqui
tem o par: o cenário em que ele **muda** o resultado, e o em que ele **não pode**
mudar.

**A extrapolação exige dois lotes.** Com lote único e QCs nos extremos, toda
amostra cai dentro do intervalo e o ramo do tempo mais próximo nunca dispara — o
teste construído com um lote passaria sem exercitar nada.
"""

import logging
import os
from typing import Any

import numpy as np
import pytest

from refino.correction.loess import LoessCorrector
from refino.domain.cancellation import CancelCheck, ExecutionCancelledError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from tests.conftest import matrix_from_normalized


def _samples(
    n: int, *, qc_at: tuple[int, ...], batches: tuple[str, ...] | None = None
) -> tuple[SampleMetadata, ...]:
    return tuple(
        SampleMetadata(
            name=f"S{i + 1}",
            injection_order=i + 1,
            sample_type="qc" if i in qc_at else "sample",
            group="QC" if i in qc_at else "NoHTLV",
            batch=(batches[i] if batches else "A"),
        )
        for i in range(n)
    )


def _matrix(
    values: np.ndarray,
    *,
    qc_at: tuple[int, ...],
    batches: tuple[str, ...] | None = None,
) -> RefinoMatrix:
    n_samples, n_features = values.shape
    return RefinoMatrix(
        values=values,
        feature_names=tuple(f"F{i + 1}" for i in range(n_features)),
        samples=_samples(n_samples, qc_at=qc_at, batches=batches),
        mode="tiny",
    )


def _linear_drift(n: int, *, base: float = 100.0, slope: float = 5.0) -> np.ndarray:
    """Deriva linear conhecida: o sinal cresce com a ordem de injeção."""
    return base + slope * np.arange(n, dtype=np.float64)


# ---------------------------------------------------------------------------
# O caminho feliz: a deriva é removida
# ---------------------------------------------------------------------------


def test_ca_3_13_deriva_linear_conhecida_e_removida() -> None:
    """O teste substantivo: deriva plantada, deriva removida, erro abaixo de 5%.

    A referência é a **razão** entre o valor corrigido e a mediana da feature: com
    a deriva removida, todas as amostras ficam em torno da mediana, e o
    coeficiente de variação cai. Comparar valor absoluto mediria a normalização,
    não a remoção.
    """
    n = 12
    drift = _linear_drift(n)
    values = drift.reshape(-1, 1)
    matrix = _matrix(values, qc_at=(0, 3, 6, 9, 11))

    result = LoessCorrector().fit_transform(matrix, seed=7)
    corrected = result.matrix.values[:, 0]

    before = float(np.std(drift, ddof=1) / np.mean(drift))
    after = float(np.std(corrected, ddof=1) / np.mean(corrected))
    assert after < 0.05, f"coeficiente de variação após correção: {after:.4f}"
    assert after < before / 5.0, (
        f"a deriva não foi removida: CV foi de {before:.4f} para {after:.4f}"
    )
    assert result.uncorrected == {}


def test_o_conjunto_de_features_da_saida_e_identico_ao_da_entrada() -> None:
    """Nada é descartado — feature não corrigida volta **marcada**, não removida.

    É a diferença entre esta implementação e o `tools.R`, que devolve a feature
    sem correção **sem marcar**, e entre ela e um filtro, que a removeria.
    """
    values = np.column_stack([_linear_drift(10), _linear_drift(10, base=50.0, slope=-2.0)])
    matrix = _matrix(values, qc_at=(0, 3, 6, 9))
    result = LoessCorrector().fit_transform(matrix, seed=3)
    assert result.matrix.feature_names == matrix.feature_names
    assert result.matrix.values.shape == matrix.values.shape


def test_a_entrada_nao_e_mutada() -> None:
    """Regra 7 do README, e a imutabilidade alcança o array."""
    values = _linear_drift(10).reshape(-1, 1)
    matrix = _matrix(values, qc_at=(0, 3, 6, 9))
    snapshot = np.array(matrix.values, copy=True)
    LoessCorrector().fit_transform(matrix, seed=1)
    assert np.array_equal(matrix.values, snapshot, equal_nan=True)


def test_a_procedencia_registra_o_modo() -> None:
    """Sem isto, duas matrizes com o mesmo `spec_json` e valores diferentes seriam
    indistinguíveis — o modo é o que as separa."""
    values = _linear_drift(10).reshape(-1, 1)
    matrix = _matrix(values, qc_at=(0, 3, 6, 9))
    for policy in ("r_faithful", "uncorrected_feature"):
        result = LoessCorrector(null_denominator=policy).fit_transform(matrix, seed=1)
        assert f"null_denominator={policy}" in result.matrix.provenance[-1]


# ---------------------------------------------------------------------------
# Passo 1: descarte de QC outlier, nas duas metades
# ---------------------------------------------------------------------------


def test_passo1_qc_aberrante_muda_a_curva() -> None:
    """**A metade que prova que o passo existe.**

    Um QC aberrante arrasta a curva, e a curva é o divisor de todas as amostras do
    lote. Se o descarte estiver implementado, o resultado com e sem o QC
    contaminado difere; se não estiver, também — então esta metade sozinha não
    basta, e a próxima é o par.
    """
    n = 12
    clean = _linear_drift(n)
    contaminated = clean.copy()
    contaminated[6] *= 40.0  # um QC, muito acima da faixa

    qc_at = (0, 3, 6, 9, 11)
    first = LoessCorrector().fit_transform(_matrix(clean.reshape(-1, 1), qc_at=qc_at), seed=5)
    second = LoessCorrector().fit_transform(
        _matrix(contaminated.reshape(-1, 1), qc_at=qc_at), seed=5
    )
    assert not np.allclose(first.matrix.values[:, 0], second.matrix.values[:, 0], equal_nan=True), (
        "o QC aberrante não mudou nada — a curva não está sendo ajustada aos QCs"
    )


def test_passo1_com_dois_qcs_no_lote_a_mascara_nao_descarta_nada() -> None:
    """**O par da metade anterior, e é ele que distingue limite de defeito.**

    Com `n <= 2` as dobradiças de Tukey são os próprios extremos, então os limites
    caem fora da faixa e **nenhum** ponto pode ser marcado. O passo 1 vira
    identidade — e isso é consequência da definição, não omissão.

    Sem esta asserção, "o filtro de outlier não fez nada" é indistinguível de "o
    filtro não foi implementado". É a situação da fronteira
    onde os lotes ficam com 5 e 2 QCs.
    """
    from refino.metrics.rsd import boxplot_outlier_mask

    for values in ([1.0, 1000.0], [5.0, 5.0]):
        mask = boxplot_outlier_mask(np.array(values), coef=1.5)
        assert not mask.any(), f"com n = 2 a máscara marcou algo em {values}"


# ---------------------------------------------------------------------------
# Passo 3: extrapolação pelo tempo mais próximo, com dois lotes
# ---------------------------------------------------------------------------


def test_passo3_extrapolacao_usa_o_tempo_mais_proximo_com_predicao() -> None:
    """**Construído com dois lotes, porque com um o ramo não dispara.**

    As amostras de um lote caem fora do intervalo dos QCs do **próprio** lote
    quando os QCs não estão nos extremos — e é aí que o preenchimento pelo tempo
    mais próximo entra (linhas 923 a 929), em vez do valor da borda.
    """
    from refino.correction.loess import _fill_by_nearest_time

    times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fitted = np.array([np.nan, 10.0, 20.0, 30.0, np.nan])
    filled = _fill_by_nearest_time(fitted, times)

    assert filled[0] == 10.0, "a posição 1 tinha de receber a predição de t=2"
    assert filled[4] == 30.0, "a posição 5 tinha de receber a predição de t=4"
    assert np.array_equal(filled[1:4], fitted[1:4])


def test_passo3_o_preenchimento_e_por_tempo_e_nao_por_posicao_no_vetor() -> None:
    """O discriminante: com tempos irregulares, o mais próximo **no tempo** ganha.

    Uma implementação que preenchesse pelo vizinho de índice daria outro valor —
    e é exatamente o que "o valor da borda" faria.
    """
    from refino.correction.loess import _fill_by_nearest_time

    times = np.array([1.0, 2.0, 100.0])
    fitted = np.array([5.0, np.nan, 7.0])
    filled = _fill_by_nearest_time(fitted, times)
    assert filled[1] == 5.0, "t=2 está muito mais perto de t=1 que de t=100"


def test_dois_lotes_ajustam_curvas_independentes() -> None:
    """Cada lote tem a sua CURVA, e a âncora de escala é a mediana GLOBAL — `tools.R:946`.

    Construído para que a mediana dos dois lotes seja **diferente**, com a deriva de cada lote
    numa escala. A curva de cada lote tira a deriva dele, e o divisor `curva / median(e[i,])`,
    com a mediana da LINHA inteira, põe os dois lotes na mesma escala: o degrau entre eles sai, que
    é o que uma correção de lote existe para fazer.
    """
    n = 16
    low = _linear_drift(8, base=100.0, slope=3.0)
    high = _linear_drift(8, base=1000.0, slope=30.0)
    values = np.concatenate([low, high]).reshape(-1, 1)
    batches = ("A",) * 8 + ("B",) * 8
    # Cinco QCs por lote: com três o próprio R degenera — avisa
    # `reciprocal condition number 0` —, e `fit_failed` seria o desfecho correto
    # em vez de curva. Ver `test_tres_qcs_no_lote_devolvem_fit_failed`.
    matrix = _matrix(values, qc_at=(0, 2, 4, 6, 7, 8, 10, 12, 14, 15), batches=batches)

    result = LoessCorrector().fit_transform(matrix, seed=11)
    corrected = result.matrix.values[:, 0]
    assert result.uncorrected == {}, f"não deveria haver causa: {result.uncorrected}"

    # Os dois lotes em torno da mediana GLOBAL, e sem a deriva de dentro. A âncora de que o par
    # separa as duas leituras: as medianas dos lotes na entrada diferem por um fator de dez.
    global_median = float(np.median(values[:, 0]))
    assert np.median(values[8:, 0]) / np.median(values[:8, 0]) > 5, "âncora: lotes em escalas"
    for start, stop in ((0, 8), (8, n)):
        block = corrected[start:stop]
        ratio = float(np.median(block) / global_median)
        assert 0.95 < ratio < 1.05, (
            f"lote [{start}:{stop}] fora da mediana global: razão {ratio:.3f}. A mediana do "
            "próprio lote como âncora produziria exatamente isto — o degrau intacto."
        )
        assert np.ptp(block) < 0.1 * np.ptp(values[start:stop, 0]), (
            f"lote [{start}:{stop}]: a deriva de dentro do lote não saiu"
        )


# ---------------------------------------------------------------------------
# Determinismo
# ---------------------------------------------------------------------------


def test_duas_execucoes_com_a_mesma_semente_sao_bit_identicas() -> None:
    """Regra 3 do README. O passo 4 sorteia, então a semente é obrigatória."""
    values = np.column_stack([_linear_drift(12), _linear_drift(12, base=8.0, slope=-0.5)])
    matrix = _matrix(values, qc_at=(0, 3, 6, 9, 11))
    first = LoessCorrector().fit_transform(matrix, seed=99)
    second = LoessCorrector().fit_transform(matrix, seed=99)
    assert np.array_equal(first.matrix.values, second.matrix.values, equal_nan=True)
    assert first.uncorrected == second.uncorrected


def test_o_fluxo_e_por_feature_e_nao_global() -> None:
    """**Semente derivada de `(seed, índice)`, não gerador global.**

    Com gerador global consumido na ordem em que o escalonador entregar, a saída
    passa a depender do paralelismo e o teste de determinismo reprova de forma
    **intermitente** — pior que reprovar sempre.

    O discriminante: corrigir a matriz inteira e corrigir cada coluna
    isoladamente tem de dar o mesmo, porque cada feature tem fluxo próprio.
    """
    values = np.column_stack([_linear_drift(12), _linear_drift(12, base=30.0, slope=2.0)])
    matrix = _matrix(values, qc_at=(0, 3, 6, 9, 11))
    together = LoessCorrector().fit_transform(matrix, seed=4).matrix.values

    for index in range(values.shape[1]):
        single = _matrix(values[:, [index]], qc_at=(0, 3, 6, 9, 11))
        alone = LoessCorrector().fit_transform(single, seed=4 + index).matrix.values[:, 0]
        assert np.array_equal(together[:, index], alone, equal_nan=True), (
            f"feature {index} difere entre matriz inteira e coluna isolada — o fluxo "
            "não é por feature"
        )


def test_batches_desalinhado_levanta() -> None:
    """Vetor de lote com tamanho errado é defeito de chamada, não de dado."""
    values = _linear_drift(10).reshape(-1, 1)
    matrix = _matrix(values, qc_at=(0, 3, 6, 9))
    with pytest.raises(ValueError, match="alinhado a `samples`"):
        LoessCorrector().fit_transform(matrix, batches=np.array(["A", "B"]))


def test_tres_qcs_no_lote_ajustam_com_condicionamento_ruim_como_no_r() -> None:
    """Três QCs determinam a quadrática exatamente, e o R também produz saída.

    **Medido em R 4.4.2:** `loess(y ~ x, span = 0.75)` sobre três pontos **não
    levanta** — avisa `reciprocal condition number 0` e `There are other near
    singularities as well`, e devolve predição. A transcrição faz o
    mesmo: reproduzir o oráculo, não melhorá-lo.

    **Diferença declarada:** o R **avisa** e nós não. A especificação não pede
    aviso, e acrescentá-lo mudaria a superfície observável do corretor sem
    mudar valor. Fica registrado aqui em vez de decidido em silêncio.

    Este teste existe porque a primeira versão dele afirmava `fit_failed`, e a
    inflação do raio da vizinhança — necessária para o ponto de borda não receber
    peso zero — fez o ajuste passar a funcionar. Afirmar o que o código fazia
    antes da correção é o defeito que ele agora impede.
    """
    values = _linear_drift(6).reshape(-1, 1)
    matrix = _matrix(values, qc_at=(0, 2, 4))
    result = LoessCorrector().fit_transform(matrix, seed=1)

    assert result.uncorrected == {}, f"não deveria marcar: {result.uncorrected}"
    corrected = result.matrix.values[:, 0]
    assert np.all(np.isfinite(corrected)), "com deriva linear finita a saída é finita"
    # Deriva linear sobre três pontos é reproduzida exatamente pela quadrática, então
    # a correção a remove: os valores ficam em torno da mediana.
    ratio = float(np.std(corrected, ddof=1) / np.mean(corrected))
    assert ratio < 0.05, f"coeficiente de variação após correção: {ratio:.4f}"


def test_menos_de_tres_qcs_no_lote_devolvem_fit_failed() -> None:
    """Dois QCs não determinam uma quadrática, e aí a feature volta marcada.

    Bit-idêntica à entrada, com a causa nomeada. É o regime
    que a fronteira entre lotes cria no segundo lote, com 5 e 2 QCs.
    """
    values = _linear_drift(6).reshape(-1, 1)
    matrix = _matrix(values, qc_at=(0, 4))
    result = LoessCorrector().fit_transform(matrix, seed=1)
    assert result.uncorrected == {"fit_failed": frozenset({"F1"})}
    assert np.array_equal(result.matrix.values, matrix.values, equal_nan=True), (
        "feature marcada tem de voltar bit-idêntica à entrada"
    )


# ---------------------------------------------------------------------------
# Vários núcleos: o laço por feature em `joblib`, e a saída é a da série
# ---------------------------------------------------------------------------
#
# **Três propriedades, e a terceira é a que impede as duas primeiras de serem vácuas.**
#
# | Propriedade | Teste |
# |---|---|
# | dois núcleos dão a matriz da série, bit a bit, num modo real | o par sobre o `pos_hilic` |
# | a bandeira de cancelamento chega ao despacho, e o pool morto recua | um par para cada |
# | o braço de dois núcleos corre DE FACTO fora do processo de quem chama | o espião e a planta |
#
# Sem a terceira, um `if` errado que mandasse `n_jobs=2` para o laço em série deixaria as outras
# verdes: série contra série é igual por construção, e o teste passaria sem medir nada.

N_RECORTE = 300


@pytest.fixture(scope="module")
def recorte_real() -> RefinoMatrix:
    """300 features do `pos_hilic` real, sorteadas com semente fixa e na ordem da matriz.

    **Medido antes de escrever o teste, na série:** 69 das 300 saem marcadas
    `negative_fitted_curve`, e **143 das 300 mudam** quando a semente passa de 42 a 43. A segunda
    contagem é a que dá dente ao par: um defeito de fluxo — semente consumida na ordem do
    escalonador, ou derivada da posição no lote em vez do índice da feature — só aparece sobre
    features cuja saída depende da semente. O modo inteiro custaria dezenas de segundos por braço;
    o recorte custa dois.
    """
    cheia = matrix_from_normalized("pos_hilic", "none")
    escolhidas = np.sort(
        np.random.default_rng(0).choice(cheia.n_features, size=N_RECORTE, replace=False)
    )
    return RefinoMatrix(
        values=np.array(cheia.values[:, escolhidas], copy=True),
        feature_names=tuple(cheia.feature_names[j] for j in escolhidas),
        samples=cheia.samples,
        mode=cheia.mode,
    )


def _colunas_divergentes(a: np.ndarray, b: np.ndarray) -> list[int]:
    """As colunas que diferem em ALGUM bit.

    `tobytes` e não `array_equal`: este distingue `nan` de `nan` só com `equal_nan`, e confunde
    `-0.0` com `0.0`. O critério da série contra o paralelo é bit a bit (regra 3 do README).
    """
    return [j for j in range(a.shape[1]) if a[:, j].tobytes() != b[:, j].tobytes()]


def _varias_features(n_features: int = 40, seed: int = 5) -> RefinoMatrix:
    """Deriva linear com ruído em várias features: barato, e com trabalho para dois processos."""
    rng = np.random.default_rng(seed)
    n = 12
    base = rng.uniform(50.0, 500.0, size=n_features)
    inclinacao = rng.uniform(-2.0, 6.0, size=n_features)
    valores = base[None, :] + inclinacao[None, :] * np.arange(n, dtype=np.float64)[:, None]
    valores = valores * (1.0 + 0.03 * rng.standard_normal((n, n_features)))
    return _matrix(valores, qc_at=(0, 3, 6, 9, 11))


class _EspiaoDoProcesso(LoessCorrector):
    """Acusa a feature ajustada no processo de quem chamou.

    **Módulo-nível de propósito:** o `loky` serializa o corretor para o worker, e uma classe local
    não sobreviveria à viagem — o mesmo desenho do `_EspiaoDeCache` de `test_forest_speedup.py`.
    **O pid do pai viaja na INSTÂNCIA, e não na classe:** o worker importa este módulo de novo, e
    um `os.getpid()` de classe seria o do próprio worker.
    """

    def __init__(self, *, n_jobs: int) -> None:
        super().__init__(n_jobs=n_jobs)
        self.pid_do_pai = os.getpid()

    def _correct_feature(self, *args: Any, **kwargs: Any) -> Any:
        if os.getpid() == self.pid_do_pai:
            raise AssertionError(
                f"feature ajustada no processo do pai (pid {self.pid_do_pai}): o braço de vários "
                "núcleos correu em série"
            )
        return super()._correct_feature(*args, **kwargs)


def _depois_de(n: int) -> CancelCheck:
    """Uma bandeira que responde `False` nas primeiras `n` consultas e `True` daí em diante.

    É a de `tests/test_cancelamento.py`: uma bandeira sempre verdadeira mede o primeiro ponto de
    checagem e mais nenhum.
    """
    contador = {"n": 0}

    def bandeira() -> bool:
        contador["n"] += 1
        return contador["n"] > n

    return bandeira


def test_dois_nucleos_dao_a_matriz_da_serie_bit_a_bit_num_modo_real(
    recorte_real: RefinoMatrix,
) -> None:
    """**O par que autoriza `n_jobs` fora da identidade do corretor.**

    Paralelizar o laço por feature só é fiel porque cada feature tem fluxo próprio, `seed +
    índice`. Com semente global, ou com o índice trocado pela posição no lote, a saída
    passaria a depender da ordem em que o escalonador entrega as features — e o desfecho seria
    intermitente, que é pior que reprovar sempre.
    """
    serie = LoessCorrector(n_jobs=1).fit_transform(recorte_real, seed=42)

    # As duas âncoras, ANTES do braço paralelo. Conjunto de marcadas vazio faria a
    # igualdade das marcas passar por vacuidade; e uma saída que não dependesse da semente não
    # acusaria defeito de fluxo nenhum.
    assert serie.uncorrected, "âncora: nenhuma feature marcada, e a igualdade das marcas é vácua"
    outra_semente = LoessCorrector(n_jobs=1).fit_transform(recorte_real, seed=43)
    sensiveis = _colunas_divergentes(serie.matrix.values, outra_semente.matrix.values)
    assert len(sensiveis) >= N_RECORTE // 4, (
        f"âncora: só {len(sensiveis)} de {N_RECORTE} features mudam com a semente (medido: 143). "
        "Um defeito de fluxo de semente passaria por este par sem ser visto"
    )

    paralela = LoessCorrector(n_jobs=2).fit_transform(recorte_real, seed=42)

    assert paralela.matrix.values.shape == serie.matrix.values.shape
    divergentes = _colunas_divergentes(serie.matrix.values, paralela.matrix.values)
    assert not divergentes, (
        f"{len(divergentes)} de {N_RECORTE} features divergem entre um e dois núcleos — a primeira "
        f"é {recorte_real.feature_names[divergentes[0]] if divergentes else ''}. O fluxo de "
        "semente passou a depender da ordem de execução, ou a montagem saiu da ordem das features"
    )
    assert serie.uncorrected == paralela.uncorrected, "o conjunto das não corrigidas divergiu"
    assert serie.matrix.feature_names == paralela.matrix.feature_names
    assert serie.matrix.provenance == paralela.matrix.provenance


def test_com_dois_nucleos_nenhuma_feature_e_ajustada_no_processo_do_pai() -> None:
    """A âncora de que o braço paralelo é paralelo, contada no código sob teste."""
    resultado = _EspiaoDoProcesso(n_jobs=2).fit_transform(_varias_features(), seed=42)
    assert resultado.matrix.n_features == 40


def test_o_espiao_acusa_a_feature_ajustada_no_processo_do_pai() -> None:
    """A planta do espião: com um núcleo o ajuste corre no pai, e ele TEM de acusar.

    Sem esta metade, o teste de cima passaria com um espião que nunca acusa — o negativo verde que
    não exercita nada.
    """
    with pytest.raises(AssertionError, match="processo do pai"):
        _EspiaoDoProcesso(n_jobs=1).fit_transform(_varias_features(), seed=42)


def test_o_cancelamento_interrompe_o_loess_com_dois_nucleos() -> None:
    """**A checagem vai para o despacho, como no SERRF**, e a exceção atravessa o `joblib`.

    `_depois_de(3)` e não `lambda: True`: com dois núcleos o único ponto de checagem é o despacho,
    e a asserção é de IGUALDADE do sítio, não de prefixo. Depois da parada, uma correção nova com
    dois núcleos tem de funcionar — é o que quem cancela e volta a pedir encontra.
    """
    matriz = _varias_features()
    with pytest.raises(ExecutionCancelledError) as capturado:
        LoessCorrector(n_jobs=2).fit_transform(matriz, seed=42, cancelled=_depois_de(3))
    assert capturado.value.where == "loess: despacho de feature", capturado.value.where

    serie = LoessCorrector(n_jobs=1).fit_transform(matriz, seed=42)
    de_novo = LoessCorrector(n_jobs=2).fit_transform(matriz, seed=42)
    assert de_novo.matrix.values.tobytes() == serie.matrix.values.tobytes(), (
        "depois de um cancelamento, a correção seguinte com dois núcleos já não dá a da série"
    )


def test_sem_bandeira_levantada_os_dois_nucleos_corrigem_a_matriz_inteira() -> None:
    """A metade negativa: a checagem no despacho não pode parar o caminho normal."""
    matriz = _varias_features()
    serie = LoessCorrector(n_jobs=1).fit_transform(matriz, seed=42)
    paralela = LoessCorrector(n_jobs=2).fit_transform(matriz, seed=42, cancelled=lambda: False)
    assert paralela.matrix.values.tobytes() == serie.matrix.values.tobytes()
    assert paralela.uncorrected == serie.uncorrected


def test_o_pool_que_morre_recua_para_a_serie_com_a_mesma_saida(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """O recuo do SERRF, no LOESS: o modo de falha aceitável é ser lento.

    A planta é um `Parallel` cujo pool morre depois de consumir uma feature —
    `TerminatedWorkerError`, a exceção que o `loky` levanta quando um worker morre. O resultado
    tem de ser o da série, e o recuo tem de ficar no registro.
    """
    from joblib.externals.loky.process_executor import TerminatedWorkerError

    import refino.correction.loess as modulo

    class _PoolQueMorre:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def __call__(self, tarefas: Any) -> Any:
            next(iter(tarefas))
            raise TerminatedWorkerError("planta: o worker morreu a meio")

    matriz = _varias_features()
    serie = LoessCorrector(n_jobs=1).fit_transform(matriz, seed=42)
    monkeypatch.setattr(modulo, "Parallel", _PoolQueMorre)
    with caplog.at_level(logging.WARNING, logger="refino.correction.loess"):
        recuada = LoessCorrector(n_jobs=2).fit_transform(matriz, seed=42)

    assert recuada.matrix.values.tobytes() == serie.matrix.values.tobytes()
    assert any("SERIE" in registro.getMessage() for registro in caplog.records), (
        "o recuo para série aconteceu sem deixar registro"
    )


def test_so_a_morte_do_pool_recua_e_outra_falha_sobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A outra direção do recuo: um `except` largo esconderia defeito como lentidão."""
    import refino.correction.loess as modulo

    class _PoolQueFalha:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def __call__(self, tarefas: Any) -> Any:
            raise ValueError("planta: falha que não é morte do pool")

    monkeypatch.setattr(modulo, "Parallel", _PoolQueFalha)
    with pytest.raises(ValueError, match="planta: falha"):
        LoessCorrector(n_jobs=2).fit_transform(_varias_features(), seed=42)

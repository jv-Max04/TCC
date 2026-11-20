"""O espelho com `none` e os negativos que lhe dão dente.

**O critério é verdadeiro por construção, e é isso que o torna útil.** Com o
corretor `none` nada é alterado, então o vetor dos valores fora da amostra é o
próprio vetor dos QCs e o RSD sobre ele é o RSD comum. É espelho de calibração:
qualquer desvio dele denuncia defeito de laço, não defeito de correção.

**Um achado que muda a construção dos
negativos.** O cenário lista três negativos plantados — vazamento, índice
trocado e iteração a menos — como se os três quebrassem o espelho **com `none`**.
Dois não quebram, e o motivo é o mesmo que torna o critério verdadeiro por
construção: `none` é **identidade**, então o valor lido da linha retida é o
original independentemente de a linha estar marcada `qc` ou `sample`. Vazamento é
literalmente invisível a ele, e índice trocado só aparece se o índice apontar para
fora do conjunto de QCs.

Então os negativos vêm em duas famílias, e a divisão é sobre o que cada corretor
consegue detectar:

| Negativo | Detectável com `none`? | Corretor usado |
|---|---|---|
| iteração a menos | **sim** — 6 valores em vez de 7 | `none` |
| leitura fora do conjunto de QCs | **sim** — entra valor de amostra biológica | `none` |
| **vazamento** do retido para o treino | **não** | `_QcAnchorCorrector` |
| **índice trocado** entre retido e lido | **não** | `_QcAnchorCorrector` |

O `_QcAnchorCorrector` existe só para isso: um corretor mínimo cuja saída
**depende do conjunto de linhas marcadas `qc`**. Sem ele os dois últimos negativos
seriam verdes sem exercitar nada.

**Os quatro plantam nos passos nomeados de `crossval` e a asserção corre pelo
`leave_one_qc_out_rsd` de verdade** — forma "em processo", sem
intermediário que possa não receber a planta.
"""

import logging
import os
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.domain.cancellation import CancelCheck, ExecutionCancelledError
from refino.domain.correction import CorrectionResult
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.metrics import crossval
from refino.metrics.crossval import (
    DegenerateCrossValidationError,
    leave_one_qc_out_rsd,
    qc_rsd_r_cv,
)
from refino.metrics.rsd import qc_rsd
from tests.conftest import matrix_from_normalized

N_QC = 7


def _matrix(
    n_samples: int = 24,
    n_features: int = 6,
    seed: int = 11,
    qc_batches: tuple[str, ...] | None = None,
) -> RefinoMatrix:
    """Sete QCs espalhados, intensidades realistas, um lote por padrão."""
    rng = np.random.default_rng(seed)
    qc_at = tuple(round(i) for i in np.linspace(0, n_samples - 1, N_QC))
    values = rng.uniform(1e3, 1e5, size=(n_samples, n_features))

    qc_index = {row: position for position, row in enumerate(qc_at)}
    samples = tuple(
        SampleMetadata(
            name=f"s{i:03d}",
            injection_order=i + 1,
            sample_type="qc" if i in qc_index else "sample",
            group="qc" if i in qc_index else "case",
            batch=(qc_batches[qc_index[i]] if qc_batches and i in qc_index else "A"),
        )
        for i in range(n_samples)
    )
    return RefinoMatrix(
        values=values,
        feature_names=tuple(f"f{j}" for j in range(n_features)),
        samples=samples,
        mode="synthetic",
    )


class _QcAnchorCorrector:
    """Corretor mínimo cuja saída DEPENDE de quais linhas estão marcadas `qc`.

    Divide cada coluna pela média das linhas `qc`. É o mais simples que satisfaz a
    propriedade de que os negativos de vazamento e de índice trocado precisam: se o
    QC retido continuar marcado `qc`, ele entra na âncora e a saída muda.

    Não é um método de correção proposto, e não pretende ser — é instrumento de
    teste, e está aqui porque `none` não consegue detectar aqueles dois defeitos.
    """

    name = "qcanchor"

    def fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        # `cancelled` entrou no `SupportsCorrection` em 06/09/2026, porque a bandeira
        # precisava de atravessar o leave-one-QC-out até o corretor. A âncora não tem laço em
        # que a checar: recebe e descarta.
        cancelled: Any = None,
    ) -> CorrectionResult:
        del batches, seed, target_features, cancelled
        anchor = np.mean(matrix.values[matrix.qc_mask], axis=0)
        return CorrectionResult(
            matrix=matrix.with_values(matrix.values / anchor, "correction:qcanchor"),
            uncorrected={},
        )


class _CountingCorrector:
    """Conta chamadas. Existe para dar dente aos testes de recusa prévia."""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        # Aceito e ignorado: o protocolo passou a declarar `cancelled`, e
        # aqui a única quantidade medida é a CONTAGEM de chamadas — reagir à bandeira mudaria o
        # que o teste de recusa prévia conta.
        cancelled: Any = None,
    ) -> CorrectionResult:
        del batches, seed, target_features, cancelled
        self.calls += 1
        return CorrectionResult(
            matrix=matrix.with_values(np.array(matrix.values, copy=True), "correction:counting"),
            uncorrected={},
        )


# ---------------------------------------------------------------------------
# O espelho
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["plain", "outlier_removed"])
def test_ca_3_10_espelho_com_none_e_numericamente_igual(variant: str) -> None:
    """Igualdade EXATA, não aproximada.

    `array_equal` sobre os não-`nan`: o critério diz "numericamente igual", e o
    caminho é o mesmo array de QCs passando pela mesma função de RSD. Tolerância
    aqui esconderia justamente o desvio de laço que o espelho existe para pegar.
    """
    matrix = _matrix()
    espelho = leave_one_qc_out_rsd(matrix, NoneCorrector(), variant=variant)  # type: ignore[arg-type]
    referencia = qc_rsd(matrix, variant=variant)  # type: ignore[arg-type]

    assert espelho.shape == referencia.shape == (matrix.n_features,)
    assert np.array_equal(espelho, referencia, equal_nan=True), (
        "o espelho de calibração divergiu do RSD in-sample com o corretor `none`: "
        "é defeito de laço, e é exatamente o que o [CA-3.10] existe para pegar"
    )


def test_o_espelho_mede_cada_qc_exatamente_uma_vez() -> None:
    """A propriedade que os negativos de contagem atacam, afirmada direto.

    Sem esta asserção, "sete iterações" seria inferido do espelho passar — e o
    espelho passa também com um multiconjunto de QCs repetidos, porque o RSD não
    vê ordem.
    """
    matrix = _matrix()
    assert crossval.held_out_rows(matrix).size == N_QC
    assert sorted(crossval.held_out_rows(matrix).tolist()) == sorted(
        np.flatnonzero(matrix.qc_mask).tolist()
    )


# ---------------------------------------------------------------------------
# Os negativos plantados — família detectável com `none`
# ---------------------------------------------------------------------------


def test_iteracao_a_menos_quebra_o_espelho(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seis iterações em vez de sete: o vetor fora da amostra perde um valor."""
    matrix = _matrix()
    limpo = leave_one_qc_out_rsd(matrix, NoneCorrector())  # type: ignore[arg-type]

    monkeypatch.setattr(crossval, "held_out_rows", lambda m: np.flatnonzero(m.qc_mask)[:-1])
    plantado = leave_one_qc_out_rsd(matrix, NoneCorrector())  # type: ignore[arg-type]

    assert not np.allclose(limpo, plantado), (
        "com uma iteração a menos o espelho continuou fechando: o portão do "
        "[CA-3.10] está inerte quanto à contagem de iterações"
    )


def test_leitura_fora_do_conjunto_de_qcs_quebra_o_espelho(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Índice apontando para amostra biológica: entra valor que não é de QC."""
    matrix = _matrix()
    limpo = leave_one_qc_out_rsd(matrix, NoneCorrector())  # type: ignore[arg-type]

    nao_qc = int(np.flatnonzero(~matrix.qc_mask)[0])
    rows = np.flatnonzero(matrix.qc_mask)
    monkeypatch.setattr(crossval, "held_out_rows", lambda m: np.array([nao_qc, *rows[1:].tolist()]))
    plantado = leave_one_qc_out_rsd(matrix, NoneCorrector())  # type: ignore[arg-type]

    assert not np.allclose(limpo, plantado), (
        "trocar um QC por amostra biológica não mudou o RSD: o espelho não vê de "
        "onde vem o valor que mede"
    )


# ---------------------------------------------------------------------------
# Os negativos plantados — família que `none` NÃO detecta
# ---------------------------------------------------------------------------


def test_vazamento_do_retido_para_o_treino_quebra_com_corretor_sensivel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**O negativo que `none` não consegue pegar, e por isso muda de corretor.**

    A planta é `remark_as_sample` virar identidade: o QC retido continua marcado
    `qc` e entra na âncora do corretor, que é a definição de vazar para o treino.
    Com `none` isto seria verde sem exercitar nada.
    """
    matrix = _matrix()
    corretor = _QcAnchorCorrector()
    limpo = leave_one_qc_out_rsd(matrix, corretor)  # type: ignore[arg-type]

    monkeypatch.setattr(crossval, "remark_as_sample", lambda m, row: m)
    plantado = leave_one_qc_out_rsd(matrix, corretor)  # type: ignore[arg-type]

    assert not np.allclose(limpo, plantado), (
        "o QC retido vazou para o treino e o resultado não mudou: a remarcação não "
        "está no caminho, e o valor medido não é fora da amostra"
    )


def test_indice_trocado_quebra_com_corretor_sensivel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remarcar um QC e ler outro: o valor lido esteve no treino da iteração."""
    matrix = _matrix()
    corretor = _QcAnchorCorrector()
    limpo = leave_one_qc_out_rsd(matrix, corretor)  # type: ignore[arg-type]

    rows = np.flatnonzero(matrix.qc_mask).tolist()
    vizinho = dict(zip(rows, [*rows[1:], rows[0]], strict=True))
    original = crossval.remark_as_sample
    monkeypatch.setattr(crossval, "remark_as_sample", lambda m, row: original(m, vizinho[int(row)]))
    plantado = leave_one_qc_out_rsd(matrix, corretor)  # type: ignore[arg-type]

    assert not np.allclose(limpo, plantado), (
        "remarcar um QC e ler outro não mudou o resultado: o índice do retido e o "
        "do lido não estão amarrados"
    )


# ---------------------------------------------------------------------------
# A recusa prévia, e ela levanta em vez de devolver número
# ---------------------------------------------------------------------------


def test_levanta_quando_um_lote_fica_com_menos_de_dois_qcs_de_treino() -> None:
    """**A executabilidade é contrato, e o teste afirma que LEVANTA.**

    Dois QCs num lote deixam **um** no treino ao reter um — número sobre treino de
    um ponto pareceria medição. "Sete QCs" é configuração de hoje,
    e o diagnóstico existe para variar isso.
    """
    matrix = _matrix(qc_batches=("A", "A", "A", "A", "A", "B", "B"))

    with pytest.raises(DegenerateCrossValidationError, match="B=2 QCs"):
        leave_one_qc_out_rsd(matrix, NoneCorrector())  # type: ignore[arg-type]


def test_a_recusa_acontece_antes_de_qualquer_ajuste() -> None:
    """A asserção que dá dente à recusa prévia: zero chamadas ao corretor.

    Sem ela, "levantou" não distingue recusa por aritmética de recusa depois de
    gastar ajustes — e a distinção é a razão de a guarda existir antes do laço.
    """
    matrix = _matrix(qc_batches=("A", "A", "A", "A", "A", "B", "B"))
    corretor = _CountingCorrector()

    with pytest.raises(DegenerateCrossValidationError):
        leave_one_qc_out_rsd(matrix, corretor)  # type: ignore[arg-type]

    assert corretor.calls == 0, (
        f"a guarda gastou {corretor.calls} ajustes num caso decidível por contagem"
    )


def test_tres_qcs_por_lote_e_o_limite_e_ele_passa() -> None:
    """O par da recusa: com três QCs por lote, reter um deixa dois e executa.

    Sem este teste a guarda poderia estar do tipo "sempre reprova" — recusando
    toda configuração multi-lote — e o teste acima passaria igual.
    """
    matrix = _matrix(n_samples=24, qc_batches=("A", "A", "A", "A", "B", "B", "B"))
    resultado = leave_one_qc_out_rsd(matrix, NoneCorrector())  # type: ignore[arg-type]
    assert resultado.shape == (matrix.n_features,)
    assert np.isfinite(resultado).any(), "nenhum RSD definido: o caso limite degenerou"


def test_variante_invalida_levanta_na_fronteira() -> None:
    """`RsdVariant` é `Literal`, e a guarda em runtime vale na fronteira."""
    with pytest.raises((ValueError, KeyError)):
        leave_one_qc_out_rsd(_matrix(), NoneCorrector(), variant="nao_existe")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# `qc_rsd_r_cv` — o estimador de compatibilidade
# ---------------------------------------------------------------------------


class _SpyCorrector:
    """Registra a forma de cada matriz recebida e os lotes de cada retido.

    O `name` é parâmetro porque a restrição de lote é **condicional ao corretor**,
    e é o nome que a seleciona.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.shapes: list[tuple[int, int]] = []
        self.target_batches: list[frozenset[str]] = []

    def fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        # Conformidade com o `SupportsCorrection`: hoje quem passa a
        # bandeira é o `leave_one_qc_out_rsd`, e o `qc_rsd_r_cv` — o único que usa este espião —
        # ainda não a repassa. O espião declara o argumento para não depender disso, e ignora-o.
        cancelled: Any = None,
    ) -> CorrectionResult:
        del batches, seed, target_features, cancelled
        self.shapes.append((matrix.n_samples, matrix.n_features))
        self.target_batches.append(
            frozenset(
                sample.batch
                for sample, is_qc in zip(matrix.samples, matrix.qc_mask, strict=True)
                if not is_qc
            )
        )
        return CorrectionResult(
            matrix=matrix.with_values(np.array(matrix.values, copy=True), "correction:spy"),
            uncorrected={},
        )


def test_o_reajuste_e_sobre_a_submatriz_de_qc_e_mais_nada() -> None:
    """**A guarda, e sem ela a implementação natural passa em tudo menos no golden.**

    O espião conta as colunas que o corretor recebeu e afirma que são `n_qc` — sete —
    e **não** `n_samples`. São dois números diferentes no mesmo mecanismo: a entrada
    do corretor são 7 colunas de QC, e o **alvo** dentro dela são 2. Citar um no lugar
    do outro é rótulo certo sobre a quantidade errada, com os dois números certos.
    """
    matrix = _matrix()
    spy = _SpyCorrector("serrf")
    qc_rsd_r_cv(matrix, spy, n_draws=3)  # type: ignore[arg-type]

    assert len(spy.shapes) == 3
    for n_samples, n_features in spy.shapes:
        assert n_samples == N_QC, (
            f"o corretor recebeu {n_samples} colunas; o r_cv reajusta sobre a "
            f"submatriz de QC e mais nada, que são {N_QC}"
        )
        assert n_features == matrix.n_features
    assert matrix.n_samples != N_QC, "o fixture não distingue os dois números"


def test_a_contagem_de_treino_e_round_e_nao_truncamento() -> None:
    """`round(7 · 0,7) = 5`, logo 5 de treino e DOIS no retido.

    O negativo é o que importa: `int(7 · 0,7) = 4` daria 4/3, três colunas no alvo,
    sem levantar e sem golden disponível que acuse.
    """
    assert crossval.train_count(7, 0.7) == 5
    assert 7 - crossval.train_count(7, 0.7) == 2
    assert int(7 * 0.7) == 4, "a premissa do negativo mudou: int() deixaria de truncar"
    assert crossval.train_count(7, 0.7) != int(7 * 0.7)


def test_o_retido_tem_duas_colunas_e_e_o_que_o_rsd_mede() -> None:
    """O alvo é 2, e é sobre ele que o RSD de cada sorteio é calculado."""
    matrix = _matrix()
    spy = _SpyCorrector("serrf")
    qc_rsd_r_cv(matrix, spy, n_draws=2)  # type: ignore[arg-type]
    assert all(len(b) >= 1 for b in spy.target_batches)


def test_par_causal_a_restricao_de_lote_vale_para_serrf_e_nao_para_loess() -> None:
    """**O teste que a restrição de lote existe para exigir.**

    Fixture de 5 QCs num lote e 2 no outro, com espião registrando os lotes de cada
    retido. Com o corretor `serrf`, **todo** sorteio tem os dois lotes; com
    `batch_loess`, **existe** sorteio que não tem. A segunda asserção é a que uma
    implementação incondicional reprova.

    Com lote único as duas seriam vacuamente verdadeiras — o fixture de dois lotes é
    o requisito e não o detalhe, e é por isso que nenhum dos quatro arquivos do
    laboratório distingue as duas políticas.
    """
    matrix = _matrix(n_samples=24, qc_batches=("A", "A", "A", "A", "A", "B", "B"))

    serrf = _SpyCorrector("serrf")
    qc_rsd_r_cv(matrix, serrf, n_draws=8)  # type: ignore[arg-type]
    assert all(b == frozenset({"A", "B"}) for b in serrf.target_batches), (
        "algum sorteio do caminho SERRF não representou os dois lotes no retido: a "
        "restrição da linha 1263 não está no caminho"
    )

    loess = _SpyCorrector("batch_loess")
    qc_rsd_r_cv(matrix, loess, n_draws=8)  # type: ignore[arg-type]
    assert any(b != frozenset({"A", "B"}) for b in loess.target_batches), (
        "todo sorteio do caminho batch_loess representou os dois lotes: a restrição "
        "está sendo aplicada onde o oráculo NÃO a tem, e o [CA-3.12] "
        "divergiria na metade do batch_loess"
    )


def test_lote_sem_qc_levanta_antes_de_ajustar() -> None:
    """**A mais grave: dispara com dois lotes e não dissolve com mais QCs.**

    O laço do oráculo compara lotes dos QCs retidos contra lotes de TODAS as
    amostras, então lote com amostras e nenhum QC é inalcançável por construção.
    """
    matrix = _matrix()
    samples = list(matrix.samples)
    nao_qc = int(np.flatnonzero(~matrix.qc_mask)[0])
    samples[nao_qc] = replace(samples[nao_qc], batch="Z")
    com_lote_orfao = replace(matrix, samples=tuple(samples))

    spy = _SpyCorrector("serrf")
    with pytest.raises(DegenerateCrossValidationError, match="sem nenhum QC"):
        qc_rsd_r_cv(com_lote_orfao, spy, n_draws=5)  # type: ignore[arg-type]
    assert spy.shapes == [], "a recusa gastou ajuste num caso decidível por contagem"


def test_mais_lotes_que_posicoes_no_retido_levanta_antes_de_ajustar() -> None:
    """**Impossibilidade por contagem — e ela dissolve com mais QCs.**

    Três lotes não cabem num retido de duas posições. Causa **distinta** da do lote sem QC de
    propósito: esta admite mais QCs como remédio, aquela não admite nenhum.
    """
    matrix = _matrix(qc_batches=("A", "A", "A", "B", "B", "C", "C"))
    spy = _SpyCorrector("serrf")

    with pytest.raises(DegenerateCrossValidationError, match="não cabem num retido"):
        qc_rsd_r_cv(matrix, spy, n_draws=5)  # type: ignore[arg-type]
    assert spy.shapes == [], "a recusa gastou ajuste num caso decidível por contagem"


def test_as_duas_recusas_tem_mensagens_distintas() -> None:
    """Remédios opostos pedem mensagens diferentes.

    Sem esta asserção as duas causas poderiam convergir para o mesmo texto numa
    refatoração, e "impossível porque o retido é pequeno" — que admite mais QCs —
    ficaria indistinguível de "um lote não tem régua", que não admite nada.
    """
    tres_lotes = _matrix(qc_batches=("A", "A", "A", "B", "B", "C", "C"))
    matrix = _matrix()
    samples = list(matrix.samples)
    nao_qc = int(np.flatnonzero(~matrix.qc_mask)[0])
    samples[nao_qc] = replace(samples[nao_qc], batch="Z")
    orfao = replace(matrix, samples=tuple(samples))

    with pytest.raises(DegenerateCrossValidationError) as por_contagem:
        qc_rsd_r_cv(tres_lotes, _SpyCorrector("serrf"), n_draws=1)  # type: ignore[arg-type]
    with pytest.raises(DegenerateCrossValidationError) as por_lote_sem_qc:
        qc_rsd_r_cv(orfao, _SpyCorrector("serrf"), n_draws=1)  # type: ignore[arg-type]

    assert str(por_contagem.value) != str(por_lote_sem_qc.value)
    assert "lotes não cabem num retido" in str(por_contagem.value)
    assert "sem nenhum QC" in str(por_lote_sem_qc.value)


def test_a_amostragem_e_exata_e_as_nao_amostradas_sao_nan() -> None:
    """**O par causal da amostragem: o valor não muda, a cobertura muda.**

    É o que separa amostrar de cortar. Se o RSD das amostradas divergisse do da
    execução completa, o critério compararia outra quantidade — e a justificativa
    inteira da amostragem é preservar a quantidade e declarar a precisão.
    """
    matrix = _matrix(n_features=12)

    completo = leave_one_qc_out_rsd(matrix, NoneCorrector(), n_sampled_features=None)  # type: ignore[arg-type]
    amostrado = leave_one_qc_out_rsd(matrix, NoneCorrector(), n_sampled_features=5)  # type: ignore[arg-type]

    medidas = ~np.isnan(amostrado)
    assert medidas.sum() == 5, f"amostrou {medidas.sum()} em vez de 5"
    assert amostrado.shape == completo.shape == (12,), "a forma do retorno mudou"
    assert np.array_equal(amostrado[medidas], completo[medidas]), (
        "o RSD das features amostradas divergiu da execução completa: a amostragem "
        "mudou a quantidade medida, e deixou de ser amostragem"
    )


def test_a_amostragem_e_no_op_quando_cobre_a_matriz() -> None:
    """`n >= n_features` devolve execução completa, e é o que deixa fixture pequeno intocado.

    Sem isto todo teste de matriz de 6 ou 12 features passaria a exercitar o caminho
    parcial, e os que reafirmam o comportamento preservado deixariam de reafirmá-lo.
    """
    matrix = _matrix(n_features=6)
    assert crossval.sampled_feature_names(matrix, 500, seed=42) is None
    assert crossval.sampled_feature_names(matrix, 6, seed=42) is None
    assert crossval.sampled_feature_names(matrix, None, seed=42) is None
    assert crossval.sampled_feature_names(matrix, 3, seed=42) is not None


def test_a_amostra_e_deterministica_e_ordenada_pela_matriz() -> None:
    """Mesma semente, mesma amostra; e a ordem é a das colunas, não a do sorteio."""
    matrix = _matrix(n_features=20)
    a = crossval.sampled_feature_names(matrix, 6, seed=3)
    b = crossval.sampled_feature_names(matrix, 6, seed=3)
    c = crossval.sampled_feature_names(matrix, 6, seed=4)
    assert a == b, "mesma semente deu amostra diferente"
    assert a != c, "sementes diferentes deram a mesma amostra"
    assert a is not None
    posicoes = [matrix.feature_names.index(n) for n in a]
    assert posicoes == sorted(posicoes), "a ordem veio do sorteio e não da matriz"


def test_o_r_cv_e_deterministico_e_o_original_nao_e() -> None:
    """Replica um original sem semente, e por isso **é** determinístico aqui.

    A diferença é nossa e não do oráculo, e está declarada no contrato: o bloco do
    `tools.R` não chama `set.seed`, então nem ele se reproduz.
    """
    matrix = _matrix()
    a = qc_rsd_r_cv(matrix, NoneCorrector(), seed=7)  # type: ignore[arg-type]
    b = qc_rsd_r_cv(matrix, NoneCorrector(), seed=7)  # type: ignore[arg-type]
    c = qc_rsd_r_cv(matrix, NoneCorrector(), seed=8)  # type: ignore[arg-type]

    assert np.array_equal(a, b, equal_nan=True), "mesma semente deu resultado diferente"
    assert not np.array_equal(a, c, equal_nan=True), (
        "sementes diferentes deram resultado idêntico: o sorteio não está semeado, "
        "e o determinismo seria vacuamente verdadeiro"
    )


# ---------------------------------------------------------------------------
# Vários núcleos: os ajustes fora da amostra em `joblib`, com a saída da série
# ---------------------------------------------------------------------------
#
# **Os `n_qc` ajustes são unidades independentes** — cada um remarca um QC e corrige a matriz
# inteira, com a mesma semente —, então correm em processos separados e voltam na ordem dos QCs.
# Os negativos plantados acima continuam a correr pela série: o `monkeypatch` é do processo, e um
# worker importa o módulo sem a planta.

N_RECORTE_REAL = 60


@pytest.fixture(scope="module")
def recorte_real() -> RefinoMatrix:
    """60 features do `pos_hilic` real, sorteadas com semente fixa e na ordem da matriz.

    Sete ajustes de LOESS por braço: a população inteira custaria minutos, e o que o par afirma —
    a mesma régua com um e com dois núcleos — não depende do tamanho dela.
    """
    cheia = matrix_from_normalized("pos_hilic", "none")
    escolhidas = np.sort(
        np.random.default_rng(1).choice(cheia.n_features, size=N_RECORTE_REAL, replace=False)
    )
    return RefinoMatrix(
        values=np.array(cheia.values[:, escolhidas], copy=True),
        feature_names=tuple(cheia.feature_names[j] for j in escolhidas),
        samples=cheia.samples,
        mode=cheia.mode,
    )


class _CorretorQueAcusaOPai:
    """`none` que acusa o ajuste feito no processo de quem chamou a régua.

    Módulo-nível porque o `loky` o serializa; o pid do pai viaja na instância, porque o worker
    importa este módulo de novo e um `os.getpid()` de classe seria o dele.
    """

    name = "acusa-o-pai"

    def __init__(self) -> None:
        self.pid_do_pai = os.getpid()

    def fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        cancelled: Any = None,
    ) -> CorrectionResult:
        del cancelled
        if os.getpid() == self.pid_do_pai:
            raise AssertionError(
                f"ajuste fora da amostra no processo do pai (pid {self.pid_do_pai}): a régua de "
                "vários núcleos correu em série"
            )
        return NoneCorrector().fit_transform(
            matrix, batches=batches, seed=seed, target_features=target_features
        )


def _depois_de(n: int) -> CancelCheck:
    """`False` nas primeiras `n` consultas e `True` depois — a de `tests/test_cancelamento.py`."""
    contador = {"n": 0}

    def bandeira() -> bool:
        contador["n"] += 1
        return contador["n"] > n

    return bandeira


def test_a_medicao_fora_da_amostra_da_o_vetor_da_serie_com_dois_nucleos(
    recorte_real: RefinoMatrix,
) -> None:
    """**O par do estimador, sobre um modo real.** Bit a bit, e não `allclose`.

    O corretor é o LOESS em série: quem paraleliza aqui é a régua, e os dois níveis juntos
    abririam um pool dentro de cada worker.
    """
    corretor = LoessCorrector()
    serie = leave_one_qc_out_rsd(recorte_real, corretor, variant="outlier_removed", n_jobs=1)
    finitos = int(np.isfinite(serie).sum())
    assert finitos >= N_RECORTE_REAL // 2, (
        f"âncora: só {finitos} de {N_RECORTE_REAL} RSDs finitos, e a igualdade de dois vetores "
        "quase todos `nan` passaria sem medir nada"
    )

    paralelo = leave_one_qc_out_rsd(recorte_real, corretor, variant="outlier_removed", n_jobs=2)

    assert paralelo.shape == serie.shape
    # A contagem da mensagem não conta `nan` contra `nan` como divergência: `!=` contaria, e a
    # primeira versão desta mensagem dizia 31 onde eram 23 (8 RSDs indefinidos nos dois braços).
    divergem = ~((serie == paralelo) | (np.isnan(serie) & np.isnan(paralelo)))
    assert paralelo.tobytes() == serie.tobytes(), (
        f"{int(divergem.sum())} RSDs divergem entre um e dois núcleos: os ajustes fora da "
        "amostra deixaram de ser independentes, ou voltaram fora da ordem dos QCs"
    )


def test_as_linhas_fora_da_amostra_voltam_na_ordem_dos_qcs_com_dois_nucleos() -> None:
    """A ORDEM, que o RSD quase não vê: inverter as linhas muda poucos ULPs, ou nada.

    Medido sobre 60 features do `pos_hilic` com o LOESS: 23 dos 52 RSDs finitos mudaram na
    variante `outlier_removed`, e 27 dos 51 na `plain` — nenhum em mais de 3 ULPs.

    Por isso o par compara as linhas fora da amostra e não só o vetor. A âncora é que as sete
    linhas diferem entre si — sem ela, uma permutação seria invisível também aqui.
    """
    matriz = _matrix()
    corretor = _QcAnchorCorrector()
    serie = crossval.out_of_sample_values(matriz, corretor, seed=42, n_jobs=1)
    assert len({linha.tobytes() for linha in serie}) == N_QC, (
        "âncora: há linhas fora da amostra repetidas, e uma troca de ordem passaria"
    )

    paralelo = crossval.out_of_sample_values(matriz, corretor, seed=42, n_jobs=2)

    assert paralelo.tobytes() == serie.tobytes(), "as linhas fora da amostra mudaram de ordem"


def test_com_dois_nucleos_nenhum_ajuste_fora_da_amostra_corre_no_processo_do_pai() -> None:
    """A âncora de que o braço de dois núcleos é paralelo."""
    resultado = leave_one_qc_out_rsd(_matrix(), _CorretorQueAcusaOPai(), n_jobs=2)  # type: ignore[arg-type]
    assert resultado.shape == (_matrix().n_features,)


def test_o_espiao_da_regua_acusa_o_ajuste_no_processo_do_pai() -> None:
    """A planta do espião: com um núcleo o ajuste corre no pai, e ele tem de acusar."""
    with pytest.raises(AssertionError, match="processo do pai"):
        leave_one_qc_out_rsd(_matrix(), _CorretorQueAcusaOPai(), n_jobs=1)  # type: ignore[arg-type]


def test_o_cancelamento_para_o_despacho_dos_ajustes_fora_da_amostra() -> None:
    """A checagem no despacho de cada QC retido, alcançada de facto.

    `_depois_de(1)`: a primeira consulta é a de antes do primeiro ajuste, que corre nos dois
    caminhos; a segunda já é o despacho do primeiro QC, e é ela que tem de levantar.
    """
    with pytest.raises(ExecutionCancelledError) as capturado:
        leave_one_qc_out_rsd(_matrix(), NoneCorrector(), n_jobs=2, cancelled=_depois_de(1))  # type: ignore[arg-type]
    assert capturado.value.where == "leave-one-qc-out: despacho de um QC retido", (
        capturado.value.where
    )


def test_sem_bandeira_levantada_a_regua_de_dois_nucleos_mede_tudo() -> None:
    """A metade negativa: a checagem no despacho não para o caminho normal."""
    matriz = _matrix()
    serie = leave_one_qc_out_rsd(matriz, NoneCorrector(), n_jobs=1)  # type: ignore[arg-type]
    paralelo = leave_one_qc_out_rsd(matriz, NoneCorrector(), n_jobs=2, cancelled=lambda: False)  # type: ignore[arg-type]
    assert paralelo.tobytes() == serie.tobytes()


def test_os_dois_niveis_de_paralelismo_juntos_sao_recusados() -> None:
    """**Um pool dentro de cada worker, medido e não suposto.** Dentro de um worker do `loky`, um
    `Parallel(n_jobs=-1, backend="loky")` abre outro pool — o processo não é daemon e o `joblib`
    não avisa. A régua recusa antes de ajustar, e a recusa diz qual dos dois baixar.
    """
    with pytest.raises(ValueError, match="dois níveis"):
        leave_one_qc_out_rsd(_matrix(), LoessCorrector(n_jobs=2), n_jobs=2)  # type: ignore[arg-type]


def test_um_nivel_so_de_paralelismo_e_aceito_nos_dois_sentidos() -> None:
    """A outra direção da recusa: o corretor paralelo com a régua em série é o caminho do SERRF."""
    matriz = _matrix()
    dentro = leave_one_qc_out_rsd(matriz, LoessCorrector(n_jobs=2), n_jobs=1)  # type: ignore[arg-type]
    fora = leave_one_qc_out_rsd(matriz, LoessCorrector(n_jobs=1), n_jobs=2)  # type: ignore[arg-type]
    assert dentro.tobytes() == fora.tobytes()


def test_o_pool_da_regua_que_morre_recua_para_a_serie(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """O recuo do SERRF, na régua: o modo de falha aceitável é ser lento."""
    from joblib.externals.loky.process_executor import TerminatedWorkerError

    class _PoolQueMorre:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def __call__(self, tarefas: Any) -> Any:
            next(iter(tarefas))
            raise TerminatedWorkerError("planta: o worker morreu a meio")

    matriz = _matrix()
    corretor = _QcAnchorCorrector()
    serie = leave_one_qc_out_rsd(matriz, corretor, n_jobs=1)  # type: ignore[arg-type]
    monkeypatch.setattr(crossval, "Parallel", _PoolQueMorre)
    with caplog.at_level(logging.WARNING, logger="refino.metrics.crossval"):
        recuada = leave_one_qc_out_rsd(matriz, corretor, n_jobs=2)  # type: ignore[arg-type]

    assert recuada.tobytes() == serie.tobytes()
    assert any("SERIE" in registro.getMessage() for registro in caplog.records)


def test_so_a_morte_do_pool_da_regua_recua(monkeypatch: pytest.MonkeyPatch) -> None:
    """A outra direção: falha que não é morte do pool sobe."""

    class _PoolQueFalha:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def __call__(self, tarefas: Any) -> Any:
            raise ValueError("planta: falha que não é morte do pool")

    monkeypatch.setattr(crossval, "Parallel", _PoolQueFalha)
    with pytest.raises(ValueError, match="planta: falha"):
        leave_one_qc_out_rsd(_matrix(), NoneCorrector(), n_jobs=2)  # type: ignore[arg-type]

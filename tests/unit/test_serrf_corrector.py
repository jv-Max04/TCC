"""`SerrfCorrector` e os passos que têm discriminante próprio.

**O teste mais importante daqui é `test_a_floresta_divide`**, e ele não estava na
a especificação até então. A lista de testes obrigatórios pedia que os hiperparâmetros fossem
afirmados como instrumentação — e um teste de instrumentação sobre
`min_samples_leaf=5` teria **passado**, congelando o mapeamento errado como
expectativa versionada. Sob aquele mapeamento a floresta não fazia divisão nenhuma
e predizia constante zero, e todos os outros testes desta lista continuariam
verdes: determinismo, zeros, marcação. O que separa as duas leituras é medir a
profundidade, não afirmar o parâmetro.

Parar antes de escrever o teste é
diferente de parar antes de implementar, porque o teste **fixa**.
"""

import numpy as np
import pytest

from refino.correction._serrf_steps import (
    RANGER_MAX_FEATURES,
    RANGER_MIN_SAMPLES_SPLIT,
    RANGER_N_ESTIMATORS,
    repair_assembled_row,
    replace_non_finite,
)
from refino.correction.serrf import SerrfCorrector
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata

# Sete QCs, como os quatro arquivos do estudo. O número é **configuração** e não
# invariante — está aqui porque é o regime em que o defeito mordeu, e é sobre
# ele que a afirmação de profundidade vale.
N_QC = 7


def _matrix(
    values: np.ndarray, *, qc_at: tuple[int, ...], batches: tuple[str, ...] | None = None
) -> RefinoMatrix:
    n_samples, n_features = values.shape
    samples = tuple(
        SampleMetadata(
            name=f"S{i + 1}",
            injection_order=i + 1,
            sample_type="qc" if i in qc_at else "sample",
            group="QC" if i in qc_at else "NoHTLV",
            batch=(batches[i] if batches else "A"),
        )
        for i in range(n_samples)
    )
    return RefinoMatrix(
        values=values,
        feature_names=tuple(f"F{i + 1}" for i in range(n_features)),
        samples=samples,
        mode="tiny",
    )


def _realistic(n_samples: int = 30, n_features: int = 14, seed: int = 3) -> RefinoMatrix:
    """Sete QCs espalhados, deriva linear leve e magnitudes de intensidade."""
    rng = np.random.default_rng(seed)
    qc_at = tuple(round(i) for i in np.linspace(0, n_samples - 1, N_QC))
    drift = 1.0 + 0.004 * np.arange(n_samples)
    base = rng.uniform(1e3, 1e5, size=n_features)
    values = (
        base[None, :] * drift[:, None] * (1.0 + 0.06 * rng.standard_normal((n_samples, n_features)))
    )
    return _matrix(values, qc_at=qc_at)


def _depth_from_note(matrix: RefinoMatrix) -> float:
    note = matrix.provenance[-1]
    field = next(p for p in note.split(",") if p.startswith("mean_tree_depth="))
    return float(field.split("=", 1)[1].rstrip(")"))


# ---------------------------------------------------------------------------
# O discriminante
# ---------------------------------------------------------------------------


def test_a_floresta_divide() -> None:
    """**Profundidade média maior que zero com sete QCs.** É o defeito em um assert.

    Sob `min_samples_leaf = 5` — o mapeamento que a especificação trazia — qualquer
    corte deixaria um lado com no máximo 3 observações, então nenhuma árvore
    dividiria: profundidade **zero**, predição constante, e como o `y` do passo 4 é
    centrado a constante é **zero**. Os passos 5 e 6 colapsariam num
    reescalonamento por média, e o SERRF deixaria de ser SERRF **com a suíte
    inteira verde**.

    O `ranger` divide com sete linhas — 3 a 5 nós por árvore, medido no contêiner —
    e é isso que `min_samples_split` reproduz.
    """
    result = SerrfCorrector().fit_transform(_realistic(), seed=42)
    depth = _depth_from_note(result.matrix)
    assert depth > 0.0, (
        f"profundidade média {depth}: a floresta não está dividindo. Se "
        "`min_samples_split` voltou a ser `min_samples_leaf`, o corretor virou "
        "reescalonamento por média."
    )


def test_os_hiperparametros_espelham_o_ranger() -> None:
    """Instrumentação rotulada: o que ela pega é regressão para os padrões
    do `scikit-learn`, que mudam entre versões.

    **E ela sozinha não bastaria**, que é a lição: esta asserção passava
    igualmente bem sobre `min_samples_leaf=5`. Quem distingue é
    `test_a_floresta_divide`.
    """
    assert RANGER_N_ESTIMATORS == 500
    assert RANGER_MAX_FEATURES == "sqrt"
    assert RANGER_MIN_SAMPLES_SPLIT == 5


def test_o_nome_do_parametro_e_split_e_nao_leaf() -> None:
    """A troca é de **nome**, e nome é o que uma refatoração troca sem medir.

    Afirmado sobre a fonte porque o valor `5` é o mesmo nas duas leituras: um teste
    que só olhasse o número não distinguiria `min_samples_leaf=5` de
    `min_samples_split=5`, e é justamente essa indistinguibilidade que custou o
    método inteiro.
    """
    import inspect

    from refino.correction import serrf

    source = inspect.getsource(serrf)
    assert "min_samples_split=RANGER_MIN_SAMPLES_SPLIT" in source
    # Proibido o **uso**, não a palavra: o arquivo menciona `min_samples_leaf` no
    # comentário que explica o defeito, e um teste que proibisse a string proibiria a
    # própria documentação do defeito que ele existe para impedir.
    assert "min_samples_leaf=" not in source, (
        "`min_samples_leaf=` reapareceu como argumento em serrf.py"
    )


# ---------------------------------------------------------------------------
# Os critérios do corretor
# ---------------------------------------------------------------------------


def test_ca_3_4_mesma_semente_produz_saida_bit_identica() -> None:
    """Determinismo é propriedade **nossa**, e o oráculo não a tem (regra 3 do README)."""
    matrix = _realistic()
    corrector = SerrfCorrector()
    first = corrector.fit_transform(matrix, seed=42).matrix.values
    second = corrector.fit_transform(matrix, seed=42).matrix.values
    assert np.array_equal(first, second, equal_nan=True)


def test_sementes_diferentes_produzem_saida_diferente() -> None:
    """A outra direção: sem ela, um corretor que ignorasse a semente passaria acima."""
    matrix = _realistic()
    corrector = SerrfCorrector()
    first = corrector.fit_transform(matrix, seed=42).matrix.values
    other = corrector.fit_transform(matrix, seed=7).matrix.values
    assert not np.array_equal(first, other, equal_nan=True), (
        "trocar a semente não mudou nada; o fluxo derivado não está sendo consumido"
    )


def test_ca_3_6_zeros_e_ausentes_sao_unidirecionais() -> None:
    """Onde a entrada era zero a saída é zero; onde era ausente, ausente.

    Correção nenhuma transforma não-detecção em medida: uma feature que fosse de
    zero a valor positivo estaria inventando sinal.
    """
    matrix = _realistic()
    values = np.array(matrix.values, copy=True)
    values[2, 0] = 0.0
    values[5, 0] = 0.0
    values[8, 3] = np.nan
    planted = _matrix(
        values, qc_at=tuple(i for i, s in enumerate(matrix.samples) if s.sample_type == "qc")
    )

    out = SerrfCorrector().fit_transform(planted, seed=42).matrix.values
    assert np.array_equal(out == 0.0, values == 0.0), "o conjunto de zeros mudou"
    assert np.array_equal(np.isnan(out), np.isnan(values)), "o conjunto de ausentes mudou"


def test_o_conjunto_de_features_e_preservado() -> None:
    """Nada é descartado: feature não corrigida volta **marcada**."""
    matrix = _realistic()
    result = SerrfCorrector().fit_transform(matrix, seed=42)
    assert result.matrix.feature_names == matrix.feature_names
    assert result.matrix.values.shape == matrix.values.shape
    for cause, features in result.uncorrected.items():
        assert features <= set(matrix.feature_names), f"{cause} nomeia feature inexistente"


# ---------------------------------------------------------------------------
# Passos com discriminante próprio
# ---------------------------------------------------------------------------


def test_passo_8_substitui_nao_finitos_e_nao_toca_os_finitos() -> None:
    """`!is.finite` cobre `±Inf`, `NaN` e ausente; as posições finitas ficam."""
    vector = np.array([1.0, np.inf, 2.0, np.nan, 3.0, -np.inf, 4.0])
    finite_before = np.isfinite(vector)
    out = replace_non_finite(vector, np.random.default_rng(0))

    assert np.isfinite(out).all(), "sobrou não finito depois do passo 8"
    assert np.array_equal(out[finite_before], vector[finite_before]), (
        "o passo 8 mexeu em posição finita; é substituição, não suavização"
    )


def test_passo_10_usa_um_unico_sorteio_por_linha_nos_negativos() -> None:
    """`runif(1) * min(positivos)` — **um** sorteio, reutilizado na linha inteira.

    Um sorteio por posição daria dispersão que o original não tem, e a igualdade
    entre as posições negativas é o único jeito de distinguir as duas.
    """
    row = np.array([-5.0, 10.0, -3.0, 20.0, -1.0])
    out = repair_assembled_row(row, np.random.default_rng(0))
    repaired = out[row < 0]

    assert np.allclose(repaired, repaired[0]), (
        f"as posições negativas receberam valores diferentes ({repaired}); "
        "o passo 10 está sorteando por posição em vez de por linha"
    )
    assert np.array_equal(out[row > 0], row[row > 0]), "o passo 10 mexeu em positivo"
    assert (repaired >= 0).all()


def test_passo_10_substitui_ausentes_antes_dos_negativos() -> None:
    """A ordem é do original: ausentes primeiro, negativos depois."""
    row = np.array([np.nan, 10.0, -3.0, 20.0])
    out = repair_assembled_row(row, np.random.default_rng(0))
    assert np.isfinite(out).all()
    assert out[1] == 10.0
    assert out[3] == 20.0


def test_feature_sem_preditor_utilizavel_e_marcada() -> None:
    """Passo 4, zero colunas após o descarte: devolve sem correção e **marca**.

    Com duas features, o passo 3 não tem de onde tirar dez preditores e a
    interseção sai vazia — o que exercita o caminho de marcação sem precisar
    construir degeneração artificial em matriz grande.
    """
    rng = np.random.default_rng(1)
    values = 1e4 * (1.0 + 0.05 * rng.standard_normal((16, 2)))
    matrix = _matrix(values, qc_at=(0, 3, 6, 9, 12, 15))

    result = SerrfCorrector().fit_transform(matrix, seed=42)
    marked = set().union(*result.uncorrected.values()) if result.uncorrected else set()
    assert marked, (
        "nenhuma feature marcada numa matriz de duas features, onde dez preditores "
        "são impossíveis. A marcação não está sendo emitida."
    )
    assert set(result.uncorrected) <= {"empty_predictor_intersection", "no_usable_predictors"}


@pytest.mark.parametrize("factor", [2.0, 10.0])
def test_invariancia_de_escala_o_contraste_entre_com_e_sem_zeros(factor: float) -> None:
    """O discriminante é o contraste, e o discriminante é o **contraste**, não a tolerância.

    Pede-se que o teste afirme a **não** proporcionalidade para feature com
    zeros, porque o passo 1 injeta `N(1, 0.01)` **independentemente da escala**.
    Isso é verdade, e a redação convida a escrever `not np.allclose(...)` — que
    **não tem poder** nas intensidades reais. Medido aqui, o desvio relativo
    máximo escala como `1/magnitude`, porque a injeção é absoluta:

    | Magnitude | Desvio relativo máximo |
    |---|---|
    | 1e4 | 4,5e-06 |
    | 1e2 | 4,5e-04 |
    | 1e0 | 2,9e-02 |

    Nas intensidades do estudo — 1e3 a 1e5 — o desvio fica em ~1e-6, que é
    **abaixo do `rtol` padrão de `np.allclose`**. Um teste com a tolerância padrão
    passaria a acusar proporcionalidade e alguém o "consertaria" afrouxando.

    O que tem poder é o **contraste contra o controle**: feature sem zeros sai
    proporcional a `1e-16` — precisão de máquina —, e feature com zeros a `1e-6`.
    Dez ordens de grandeza separam as duas, e é isso que se afirma.
    """
    rng = np.random.default_rng(2)
    values = 1e4 * (1.0 + 0.05 * rng.standard_normal((16, 12)))
    # Feature 0 recebe zeros; a 1 fica limpa e é o controle.
    values[4, 0] = 0.0
    values[9, 0] = 0.0
    qc_at = (0, 3, 6, 9, 12, 15)

    corrector = SerrfCorrector()
    base = corrector.fit_transform(_matrix(values, qc_at=qc_at), seed=42).matrix.values
    other = corrector.fit_transform(_matrix(values * factor, qc_at=qc_at), seed=42).matrix.values

    def max_relative_deviation(index: int) -> float:
        expected = base[:, index] * factor
        keep = expected != 0.0
        return float(np.max(np.abs(other[keep, index] - expected[keep]) / np.abs(expected[keep])))

    with_zeros = max_relative_deviation(0)
    without_zeros = max_relative_deviation(1)

    assert without_zeros < 1e-12, (
        f"feature SEM zeros desviou {without_zeros:.3e} da proporcionalidade; "
        "deveria ser precisão de máquina, e se não é o defeito não é do passo 1"
    )
    assert with_zeros > 1e-9, (
        f"feature COM zeros desviou apenas {with_zeros:.3e}. A proporcionalidade é "
        "falsa por construção — o passo 1 injeta N(1, 0.01) independente da escala. "
        "Se ela ficou verdadeira, o passo 1 passou a escalar a substituição e o "
        "discriminante de F1 morreu."
    )
    assert with_zeros > without_zeros * 1e3, (
        f"o contraste sumiu: com zeros {with_zeros:.3e}, sem zeros {without_zeros:.3e}"
    )


# ---------------------------------------------------------------------------
# Ramos que os quatro modos não alcançam — valor construído
# ---------------------------------------------------------------------------


def test_ramo_4_zero_colunas_utilizaveis_marca_no_usable_predictors() -> None:
    """Ramo 4 — `tools.R:1130`, e ele **não** é alcançado pelos quatro modos.

    **O caminho para cá é mais estreito do que parece, e descobri isso medindo.**
    Espalhar `NaN` na entrada não serve: o passo 1 substitui **todo** ausente antes
    do passo 2, então nenhum `NaN` da entrada chega ao passo 4. O único `NaN` que
    chega vem da **padronização** — preditor constante no bloco tem desvio zero, e
    `0/0` é indefinido.

    Então a construção é: preditores constantes **entre os QCs**. Todos são
    descartados por `usable_predictor_columns`, sobra zero coluna, e a feature volta
    marcada `no_usable_predictors` em vez de `empty_predictor_intersection` — que é
    a outra causa e diz outra coisa.
    """
    n_samples, n_features = 16, 14
    qc_at = (0, 3, 6, 9, 12, 15)
    is_qc = np.zeros(n_samples, dtype=bool)
    is_qc[list(qc_at)] = True

    rng = np.random.default_rng(4)
    values = 1e4 * (1.0 + 0.05 * rng.standard_normal((n_samples, n_features)))
    # Todas as features menos a 0 ficam CONSTANTES entre os QCs; fora dos QCs
    # continuam variando, para que a seleção do passo 3 ainda as escolha.
    for j in range(1, n_features):
        values[is_qc, j] = 5e3 + j

    result = SerrfCorrector().fit_transform(_matrix(values, qc_at=qc_at), seed=42)

    assert "no_usable_predictors" in result.uncorrected, (
        f"causas emitidas: {sorted(result.uncorrected)}. O ramo 4 não foi alcançado — "
        "e sem ele o [CA-3.1] não diz nada sobre o que acontece quando a seleção "
        "não deixa coluna utilizável."
    )
    marked = result.uncorrected["no_usable_predictors"]
    assert marked, "a causa foi emitida com conjunto vazio"


def test_ramo_3_um_unico_preditor_retido_nao_degenera() -> None:
    """Ramo 3 — os três `is.null(dim(...))` de `tools.R:1105`, 1118 e 1125.

    No R, indexar matriz de uma coluna **derruba a dimensão** e o vetor resultante
    quebra o `ranger`; por isso o original testa `is.null(dim(...))` em três
    lugares. Em `numpy` a máscara booleana preserva o eixo, então o ramo não existe
    do nosso lado — mas *"não existe"* é afirmação sobre o nosso código, e ela
    precisa de teste, senão é suposição.

    Com `n_correlated = 1` sobra exatamente um preditor. A asserção é que a
    correção **acontece** — nenhuma causa emitida — e que a saída não é a entrada.
    """
    rng = np.random.default_rng(5)
    values = 1e4 * (1.0 + 0.05 * rng.standard_normal((16, 6)))
    qc_at = (0, 3, 6, 9, 12, 15)
    matrix = _matrix(values, qc_at=qc_at)

    result = SerrfCorrector(n_correlated=1).fit_transform(matrix, seed=42)

    assert not result.uncorrected, (
        f"com um único preditor a correção falhou: {sorted(result.uncorrected)}. "
        "Se `numpy` passou a derrubar o eixo, o ramo 3 do R passou a existir aqui."
    )
    assert result.matrix.values.shape == values.shape
    assert not np.allclose(result.matrix.values, values), (
        "a saída é igual à entrada; com um preditor a correção não fez nada"
    )

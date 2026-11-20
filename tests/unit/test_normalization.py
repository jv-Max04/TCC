"""As três operações de normalização.

**O critério é sobre as QUINZE combinações e não sobre as três funções.** "Nenhuma
operação de normalização produz `NaN` ou `Inf` a partir de matriz sem `NaN`, incluindo com
zeros presentes" — e cada uma das quinze opções tem um denominador próprio que pode
degenerar. Testar três funções deixaria doze caminhos sem cobertura.

E o fixture é construído para **conter** os casos degenerados que o critério exige tolerar:
linha toda zero, coluna constante, coluna com amplitude zero. Fixture bem-comportado passaria
por vacuidade.
"""

from typing import get_args

import numpy as np
import pytest

from refino.domain.labels import VOCABULARY, InvalidLabelError, SampleNormalization
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.preprocessing.normalization import (
    SOMA_CONSTANTE,
    normalize_samples,
    normalize_samples_declarada,
    scale_features,
    transform,
)

# **DERIVADA de `get_args` e nao escrita a mao — 10/09/2026.** A lista era pinada e nao cresceu
# Quando `sample_reference` entrou no `Literal`: o critério cobra a ausencia de nao-finito sobre
# TODAS as operacoes, e uma lista parada deixaria a nova de fora sem que nada reprovasse. E a
# enumeracao incompleta que erra sempre para MAIS — o que falta de uma particao vira residual, e
# residual le-se como pronto.
POR_AMOSTRA = get_args(SampleNormalization)

# Os metodos que exigem `normalization_reference`, com o nome que cada um usa no fixture de
# degenerados. Enumerado e nao inferido do nome: os dois chamam-se `*_reference`, mas adivinhar
# pelo sufixo seria o oraculo pela forma.
REFERENCIA_EXIGIDA: dict[str, str] = {
    "sample_reference": "S0",
    "feature_reference": "f4",
}

# `supplied_factors` nao recebe um NOME e sim um MAPA, entao ele tem tabela propria: a chave e o
# metodo e o valor e o fator uniforme aplicado a todas as amostras do fixture. Uniforme porque o
# Que o critério mede aqui e a ausencia de nao-finito, e nao a conta.
FATORES_EXIGIDOS: frozenset[str] = frozenset({"supplied_factors"})
TRANSFORMACOES = ("none", "log10", "log2", "sqrt", "cbrt")
ESCALONAMENTOS = ("none", "mean_center", "auto", "pareto", "range", "vast")


def _matriz(valores: np.ndarray, n_qc: int = 3) -> RefinoMatrix:
    n = valores.shape[0]
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i < n_qc else "sample",
            group="QC" if i < n_qc else "G",
        )
        for i in range(n)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(valores.shape[1])),
        samples=amostras,
        mode="sintetico",
    )


def _com_degenerados() -> RefinoMatrix:
    """Matriz que **contém** os casos que o critério exige tolerar.

    Sem eles o critério passaria por vacuidade: uma implementação que dividisse sem guarda
    daria o mesmo resultado numa matriz bem-comportada. A construção é declarada e afirmada
    logo abaixo, porque fixture que promete conter um caso tem de ser verificado quanto a
    contê-lo.
    """
    valores = np.array(
        [
            [10.0, 5.0, 7.0, 0.0],
            [20.0, 5.0, 3.0, 0.0],
            [30.0, 5.0, 9.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],  # linha toda zero nas colunas 0 e 2, e zeros presentes
            [40.0, 5.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    valores[3] = 0.0  # linha INTEIRAMENTE zero: soma e mediana zero
    return _matriz(valores)


def _com_degenerados_e_referencia() -> RefinoMatrix:
    """O fixture acima MAIS uma feature utilizavel como referencia — 10/09/2026.

    **Os metodos de referencia nao podem correr sobre `_com_degenerados` sozinho, e a recusa e
    correta**: ele tem uma linha inteiramente zero, entao toda feature tem zero em alguma
    amostra e nenhuma serve de denominador.

    Isso torna o critério impossivel de medir sobre eles — e a resposta certa nao e isenta-los,
    e sim dar-lhes uma matriz que satisfaz as DUAS condicoes do criterio: contem zeros, E tem por
    onde normalizar. `f4` e positiva em toda amostra, e a linha `S3` continua inteiramente zero
    nas outras quatro features.
    """
    base = np.asarray(_com_degenerados().values)
    referencia = np.array([[3.0], [4.0], [5.0], [6.0], [7.0]])
    return _matriz(np.hstack([base, referencia]))


def test_o_fixture_de_referencia_contem_zeros_E_uma_referencia_util() -> None:  # noqa: N802
    """Fixture que promete conter dois casos e verificado quanto aos DOIS.

    Sem os zeros o criterio passaria por vacuidade; sem a referencia utilizavel o metodo
    recusaria e o criterio nao seria medido de todo.
    """
    valores = np.asarray(_com_degenerados_e_referencia().values)
    assert (valores == 0.0).any(), "o fixture perdeu os zeros e o [CA-5.3] passaria por vacuidade"
    assert (valores[:, 4] > 0).all(), "a coluna de referencia tem zero e o metodo recusaria"
    assert (valores[3, :4] == 0.0).all(), "a linha inteiramente zero deixou de o ser"


def test_o_fixture_contem_os_casos_degenerados() -> None:
    """**A verificação do insumo, antes de usá-lo para afirmar tolerância**."""
    valores = np.asarray(_com_degenerados().values)
    assert (valores.sum(axis=1) == 0.0).any(), "falta linha de soma zero"
    assert (np.median(valores, axis=1) == 0.0).any(), "falta linha de mediana zero"
    assert (valores.std(axis=0) == 0.0).any(), "falta coluna de desvio zero"
    amplitude = valores.max(axis=0) - valores.min(axis=0)
    assert (amplitude == 0.0).any(), "falta coluna de amplitude zero"
    assert (valores == 0.0).any(), "falta zero presente, que o critério nomeia"
    assert np.isfinite(valores).all(), "a entrada tem de ser finita — é a premissa do CA-5.3"


# ---------------------------------------------------------------------------
# As quinze combinações
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metodo", POR_AMOSTRA)
def test_ca_5_3_normalize_samples_nao_produz_nao_finito(metodo: str) -> None:
    """Para TODAS as opções por amostra, com degenerados presentes.

    A lista é derivada, então um método novo entra aqui sem que ninguém se lembre — e os que
    exigem referência recebem-na de `REFERENCIA_EXIGIDA`, porque sem ela recusariam e o critério
    ficaria por medir sobre eles.
    """
    # Os metodos de referencia recebem o fixture que tem zeros E uma referencia utilizavel; ver
    # `_com_degenerados_e_referencia`. Isenta-los seria deixar o criterio por medir sobre eles.
    precisa_de_referencia = metodo in REFERENCIA_EXIGIDA
    entrada = _com_degenerados_e_referencia() if precisa_de_referencia else _com_degenerados()
    resultado = normalize_samples(
        entrada,
        metodo,  # type: ignore[arg-type]
        reference=REFERENCIA_EXIGIDA.get(metodo),
        reference_present_at_entry=True if precisa_de_referencia else None,
        factors=(
            tuple((nome, 2.0) for nome in entrada.sample_names)
            if metodo in FATORES_EXIGIDOS
            else None
        ),
    )
    assert np.isfinite(np.asarray(resultado.values)).all(), f"{metodo} produziu não-finito"


def test_ancora_a_lista_por_amostra_cobre_o_vocabulario_inteiro() -> None:
    """Sem esta âncora, uma derivação que devolvesse vazio faria o critério passar sobre nada.

    E ela é a que acusa o caso oposto ao da lista pinada: `get_args` a encolher — por um
    `Literal` mal editado — deixaria o critério verde com menos métodos medidos.
    """
    assert set(POR_AMOSTRA) == set(VOCABULARY["SampleNormalization"])
    assert len(POR_AMOSTRA) >= 6, f"o vocabulário encolheu para {len(POR_AMOSTRA)}"
    assert set(REFERENCIA_EXIGIDA) <= set(POR_AMOSTRA), (
        f"`REFERENCIA_EXIGIDA` nomeia método que não existe: "
        f"{sorted(set(REFERENCIA_EXIGIDA) - set(POR_AMOSTRA))}"
    )
    assert set(POR_AMOSTRA) >= FATORES_EXIGIDOS, (
        f"`FATORES_EXIGIDOS` nomeia método que não existe: "
        f"{sorted(FATORES_EXIGIDOS - set(POR_AMOSTRA))}"
    )
    assert not (set(REFERENCIA_EXIGIDA) & FATORES_EXIGIDOS), (
        "um método está nas duas tabelas: ele receberia nome E mapa, e só uma delas é o "
        "parâmetro que ele lê"
    )


@pytest.mark.parametrize("metodo", TRANSFORMACOES)
def test_ca_5_3_transform_nao_produz_nao_finito(metodo: str) -> None:
    """Para as cinco transformações."""
    resultado = transform(_com_degenerados(), metodo)  # type: ignore[arg-type]
    assert np.isfinite(np.asarray(resultado.values)).all(), f"{metodo} produziu não-finito"


@pytest.mark.parametrize("metodo", ESCALONAMENTOS)
def test_ca_5_3_scale_features_nao_produz_nao_finito(metodo: str) -> None:
    """Para os cinco escalonamentos."""
    resultado = scale_features(_com_degenerados(), metodo)  # type: ignore[arg-type]
    assert np.isfinite(np.asarray(resultado.values)).all(), f"{metodo} produziu não-finito"


@pytest.mark.parametrize("metodo", TRANSFORMACOES)
def test_transformacao_tolera_valor_negativo(metodo: str) -> None:
    """Correção pode produzir negativo, e `sqrt` comum daria `nan`.

    A raiz **com sinal** é a escolha declarada, e este teste é o que a torna verificável em
    vez de comentário.
    """
    valores = np.array([[-5.0, 0.0, 5.0], [-1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    resultado = transform(_matriz(valores, n_qc=1), metodo)  # type: ignore[arg-type]
    assert np.isfinite(np.asarray(resultado.values)).all(), f"{metodo} não tolerou negativo"


# ---------------------------------------------------------------------------
# Os valores
# ---------------------------------------------------------------------------


def test_ca_5_4_log10_com_zeros_e_finito_e_bate_com_a_forma_estabilizada() -> None:
    """O valor no zero é conferido contra a fórmula calculada à mão.

    Afirmar só finitude deixaria passar qualquer forma estabilizada; o que se verifica é que
    é **a do R** — `log10((x + sqrt(x² + m²)) / 2)` com `m = min(|x| não nulo) / 10`.
    """
    valores = np.array([[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]], dtype=np.float64)
    resultado = transform(_matriz(valores, n_qc=1), "log10")
    saida = np.asarray(resultado.values)

    assert np.isfinite(saida).all()
    m = 10.0 / 10.0  # menor módulo não nulo é 10
    esperado_no_zero = float(np.log10((0.0 + np.sqrt(0.0**2 + m**2)) / 2.0))
    assert float(saida[0, 0]) == pytest.approx(esperado_no_zero)


def test_ca_5_5_auto_da_media_zero_e_desvio_um() -> None:
    """Dentro de 1e-10, com `ddof` **declarado**."""
    gerador = np.random.default_rng(3)
    valores = gerador.normal(100.0, 15.0, size=(8, 4))
    resultado = scale_features(_matriz(valores, n_qc=2), "auto", ddof=0)
    saida = np.asarray(resultado.values)

    assert np.allclose(saida.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(saida.std(axis=0, ddof=0), 1.0, atol=1e-10)


def test_ca_5_5_o_ddof_muda_o_resultado() -> None:
    """**O par causal do parâmetro**: com `ddof` ignorado os dois dariam o mesmo.

    Em fixture pequeno a diferença entre `n` e `n - 1` é grande — 8 amostras dão fator
    `sqrt(8/7)` ≈ 1,069 —, que é justamente o motivo de o parâmetro ser explícito.
    """
    gerador = np.random.default_rng(3)
    matriz = _matriz(gerador.normal(100.0, 15.0, size=(8, 4)), n_qc=2)
    com_zero = np.asarray(scale_features(matriz, "auto", ddof=0).values)
    com_um = np.asarray(scale_features(matriz, "auto", ddof=1).values)
    assert not np.allclose(com_zero, com_um), "o `ddof` não está sendo consultado"


def test_ca_5_6_pareto_divide_pela_raiz_do_desvio() -> None:
    """O desvio resultante é a **raiz** do desvio original.

    Comparado contra o desvio original **medido** e não contra constante: se a matriz mudar,
    a asserção continua sendo sobre a propriedade.
    """
    gerador = np.random.default_rng(5)
    valores = gerador.normal(50.0, 9.0, size=(10, 3))
    matriz = _matriz(valores, n_qc=2)
    original = valores.std(axis=0, ddof=0)

    saida = np.asarray(scale_features(matriz, "pareto", ddof=0).values)
    assert np.allclose(saida.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(saida.std(axis=0, ddof=0), np.sqrt(original), rtol=1e-10)


def test_sum_faz_a_linha_somar_a_constante() -> None:
    """A propriedade obrigatória, no caso não degenerado.

    **O teste afirmava «a linha soma 1,0» até a paridade com o MetaboAnalyst.**
    O `SumNorm` deles é `1000*x/sum(x)`, e adotá-lo mudou a constante — a
    especificação foi emendada junto, porque um teste obrigatório que contradiz o documento é
    divergência de contrato e não de implementação.

    O que a propriedade afirma continua a ser o mesmo: a soma da linha é uma **constante
    conhecida**, igual para todas as linhas. É isso que a normalização por soma promete, e o
    valor da constante é convenção do formato de destino.
    """
    valores = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    saida = np.asarray(normalize_samples(_matriz(valores, n_qc=1), "sum").values)
    assert np.allclose(saida.sum(axis=1), SOMA_CONSTANTE)


def test_linha_de_soma_zero_fica_intacta_em_vez_de_virar_inf() -> None:
    """A política de denominador degenerado, afirmada onde ela age.

    Sem ela a linha viraria `inf` e o critério reprovaria — e propagar seria a escolha que
    recusada para `imputation="none"`.
    """
    valores = np.array([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
    saida = np.asarray(normalize_samples(_matriz(valores, n_qc=1), "sum").values)
    assert saida[1].tolist() == [0.0, 0.0], "linha sem soma tem de ficar como está"
    assert np.allclose(saida[0].sum(), SOMA_CONSTANTE), (
        "e as outras linhas normalizam normalmente — para a CONSTANTE, que passou de 1 a "
        "1000 na paridade com o `SumNorm` do MetaboAnalyst"
    )


def test_quantile_da_o_mesmo_valor_a_empates() -> None:
    """A escolha nos empates é declarada, e é onde as implementações divergem."""
    valores = np.array([[1.0, 1.0, 5.0], [2.0, 3.0, 4.0]], dtype=np.float64)
    saida = np.asarray(normalize_samples(_matriz(valores, n_qc=1), "quantile").values)
    assert saida[0, 0] == pytest.approx(saida[0, 1]), "valores iguais recebem o mesmo resultado"


def test_pqn_sem_qc_e_identidade() -> None:
    """Sem QC não há espectro de referência, e a operação declara isso em vez de dividir por nada.

    A proposta de lotes pode vir sem QC, e o corretor já o salta — então é configuração real.
    """
    valores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    amostras = tuple(
        SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type="sample", group="G")
        for i in range(2)
    )
    matriz = RefinoMatrix(
        values=valores, feature_names=("a", "b"), samples=amostras, mode="sintetico"
    )
    saida = np.asarray(normalize_samples(matriz, "pqn_group").values)
    assert np.array_equal(saida, valores)


# ---------------------------------------------------------------------------
# Identidade exata e fronteira
# ---------------------------------------------------------------------------


def test_none_e_identidade_exata_nas_tres() -> None:
    """Igualdade sem tolerância.

    Identidade com tolerância esconde operação que não deveria existir.
    """
    matriz = _com_degenerados()
    original = np.asarray(matriz.values)
    for funcao in (normalize_samples, transform, scale_features):
        saida = np.asarray(funcao(matriz, "none").values)  # type: ignore[operator,arg-type]
        assert np.array_equal(saida, original), f"{funcao.__name__}('none') não é identidade exata"


@pytest.mark.parametrize(
    ("funcao", "vocabulario", "invalido"),
    [
        (normalize_samples, "SampleNormalization", "soma"),
        (transform, "Transformation", "log"),
        (scale_features, "Scaling", "minmax"),
    ],
)
def test_metodo_invalido_levanta_nomeando_o_vocabulario(
    funcao: object, vocabulario: str, invalido: str
) -> None:
    """Os valores inválidos são **próximos** dos válidos, que é o caso realista.

    `"log"` para `"log10"` e `"minmax"` para `"range"` são o erro que alguém comete; um
    `"xyz"` seria pego por qualquer guarda.
    """
    with pytest.raises(InvalidLabelError, match=vocabulario):
        funcao(_com_degenerados(), invalido)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("funcao", "metodo"),
    [(normalize_samples, "sum"), (transform, "log10"), (scale_features, "auto")],
)
def test_a_matriz_de_entrada_nao_e_alterada(funcao: object, metodo: str) -> None:
    """Regra 7 do README — a imutabilidade alcança o array."""
    matriz = _com_degenerados()
    antes = np.array(matriz.values, copy=True)
    funcao(matriz, metodo)  # type: ignore[operator]
    assert np.array_equal(np.asarray(matriz.values), antes)


# ---------------------------------------------------------------------------
# As três contas que a paridade com o MetaboAnalyst mudou
# ---------------------------------------------------------------------------

# A matriz das três asserções abaixo. Valores escolhidos para que a soma das linhas
# DIFIRA entre elas — soma igual faria a normalização por soma virar identidade a menos
# de escala, e o teste do fator 1000 passaria por acidente.
_BASE = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], dtype=np.float64)


def _srnorm_do_metaboanalyst(x: np.ndarray) -> np.ndarray:
    """`((x + sqrt(x^2 + m^2))/2)^(1/2)` com `m = min|x != 0| / 10`.

    Transcrito do `SrNorm` de `R/general_norm_utils.R` do MetaboAnalystR e calculado aqui
    de forma independente da implementação sob teste — é o oráculo, e reusar o auxiliar
    do módulo faria o teste comparar a função consigo mesma.
    """
    m = float(np.abs(x[x != 0.0]).min()) / 10.0
    return np.asarray(((x + np.sqrt(x**2 + m**2)) / 2.0) ** 0.5, dtype=np.float64)


def test_sqrt_e_a_forma_estabilizada_do_metaboanalyst() -> None:
    """O `sqrt` deixou de ser raiz com sinal e passou a ser o `SrNorm` deles."""
    obtido = np.asarray(transform(_matriz(_BASE), "sqrt").values)
    np.testing.assert_allclose(obtido, _srnorm_do_metaboanalyst(_BASE), rtol=1e-12)


def test_sqrt_estabilizado_difere_da_raiz_com_sinal() -> None:
    """A planta cai sobre dado em que as duas contas DIFEREM.

    Sem esta asserção o teste acima poderia passar com a implementação antiga em alguma
    matriz onde as duas coincidissem, e não haveria como saber qual delas correu.
    """
    com_sinal = np.sign(_BASE) * np.sqrt(np.abs(_BASE))
    assert not np.allclose(com_sinal, _srnorm_do_metaboanalyst(_BASE)), (
        "a raiz com sinal e a forma estabilizada coincidem nesta matriz: o fixture não "
        "separa as duas implementações e o teste acima seria vácuo"
    )


def test_sqrt_continua_finito_sobre_negativo() -> None:
    """O critério continua satisfeito — era a justificativa da raiz com sinal.

    A justificativa antiga argumentava contra a raiz COMUM, que dá `nan` em negativo, e
    nunca contra a forma estabilizada, que também não dá. Ela era incompleta, não errada.
    """
    negativa = np.array([[-3.0, 2.0], [1.0, -5.0], [4.0, 6.0]], dtype=np.float64)
    obtido = np.asarray(transform(_matriz(negativa), "sqrt").values)
    assert np.all(np.isfinite(obtido)), "a forma estabilizada produziu não-finito sobre negativo"


def test_sum_carrega_o_fator_mil_do_metaboanalyst() -> None:
    """`SumNorm` deles é `1000*x/sum(x)`; o nosso era `x/sum(x)`."""
    obtido = np.asarray(normalize_samples(_matriz(_BASE), "sum").values)
    esperado = 1000.0 * _BASE / _BASE.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(obtido, esperado, rtol=1e-12)


# **`pareto` NAO esta aqui, e a exclusao foi MEDIDA e nao suposta.** Ele divide por
# `sqrt(desvio)`, entao um fator `c` na matriz sobrevive como `sqrt(c)` — a lista natural
# "os escalonamentos que dividem por escala" o incluiria, e ela estaria errada.
@pytest.mark.parametrize("escalonamento", ["auto", "range", "vast"])
def test_o_fator_mil_e_inerte_sob_escalonamento_invariante(escalonamento: str) -> None:
    """Metade um do par causal: onde o fator não muda nada.

    Os quatro escalonamentos dividem por uma estatística de escala, então um fator
    constante na linha some. É por isso que o RSD e as métricas de QC não mudaram.
    """
    matriz = _matriz(_BASE)
    com_fator = scale_features(normalize_samples(matriz, "sum"), escalonamento)
    sem_fator = scale_features(
        matriz.with_values(_BASE / _BASE.sum(axis=1, keepdims=True), "manual"), escalonamento
    )
    np.testing.assert_allclose(
        np.asarray(com_fator.values), np.asarray(sem_fator.values), atol=1e-10
    )


@pytest.mark.parametrize("escalonamento", ["none", "mean_center", "pareto"])
def test_o_fator_mil_sobrevive_ao_escalonamento_nao_invariante(escalonamento: str) -> None:
    """Metade dois do par causal, e é ela que impede o teste acima de ser vácuo.

    `none` não divide por nada, `mean_center` só subtrai a média, e `pareto` divide pela
    **raiz** do desvio — nos três o fator sobrevive. Sem esta metade, «o fator é inerte» seria
    afirmação sobre a matriz e não sobre o escalonamento, e a mudança de conta passaria sem
    nada acusar.

    **`pareto` está deste lado por medição, não por dedução.** A leitura natural — «divide por
    uma estatística de escala, logo é invariante» — o poria do outro lado, e o teste teria
    passado a afirmar invariância onde não há.
    """
    matriz = _matriz(_BASE)
    com_fator = scale_features(normalize_samples(matriz, "sum"), escalonamento)
    sem_fator = scale_features(
        matriz.with_values(_BASE / _BASE.sum(axis=1, keepdims=True), "manual"), escalonamento
    )
    assert not np.allclose(np.asarray(com_fator.values), np.asarray(sem_fator.values)), (
        f"o fator 1000 sumiu sob `{escalonamento}`, que não divide por escala: se sumiu "
        "aqui, a metade invariante do par não está a medir invariância nenhuma"
    )


def test_mean_center_centra_e_preserva_o_desvio() -> None:
    """É o que o distingue do `auto`, que centra E divide."""
    obtido = np.asarray(scale_features(_matriz(_BASE), "mean_center").values)
    np.testing.assert_allclose(obtido.mean(axis=0), np.zeros(_BASE.shape[1]), atol=1e-12)
    np.testing.assert_allclose(obtido.std(axis=0), _BASE.std(axis=0), rtol=1e-12)


def test_mean_center_nao_e_o_ramo_do_vast() -> None:
    """Âncora do despacho: `vast` era o `else` de `scale_features`.

    Um valor novo no `Literal` de `Scaling` caía nele em silêncio, exatamente como `mean`
    caía no `missforest` em `impute`. As duas contas têm de dar matrizes diferentes.
    """
    centrado = np.asarray(scale_features(_matriz(_BASE), "mean_center").values)
    vast = np.asarray(scale_features(_matriz(_BASE), "vast").values)
    assert not np.allclose(centrado, vast), (
        "`mean_center` produziu a mesma matriz que `vast`: o despacho pode estar a cair no "
        "ramo final em vez de no ramo próprio"
    )


# ---------------------------------------------------------------------------
# A guarda irmã de `_seguro`: a linha sem denominador é DECLARADA
# ---------------------------------------------------------------------------


def _com_linha_de_soma_zero() -> RefinoMatrix:
    """Uma linha toda zero entre linhas normais. A soma dela é zero e o denominador degenera."""
    valores = np.array(
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
    )
    return _matriz(valores, n_qc=1)


def test_par_causal_a_linha_sem_denominador_e_DECLARADA_e_a_outra_nao() -> None:  # noqa: N802
    """**O par causal**: duas linhas, o mesmo método, e só uma é declarada.

    Sem a guarda a linha de soma zero saía por normalizada: mesmos valores de entrada, mesma nota
    de procedência das outras, e nada a dizer que a operação não teve por onde acontecer. A figura
    a desenharia como uma amostra cuja distribuição não mudou — que é o que se lê como *"não foi
    normalizada"*, e não como *"não havia denominador"*. As duas pedem ações opostas.

    **A planta cai sobre dado em que as duas coisas diferem**: `S1` é a única linha degenerada, e
    a asserção afirma as duas metades — ela é declarada, e as outras três não.
    """
    matriz = _com_linha_de_soma_zero()
    declarada = normalize_samples_declarada(matriz, "sum")

    assert declarada.linhas_sem_denominador == ("S1",), (
        f"esperava só `S1` declarada e vieram {declarada.linhas_sem_denominador}. Se vier vazio, "
        "a guarda não está a contar; se vierem todas, ela está a contar o eixo errado"
    )
    saida = np.asarray(declarada.matrix.values)
    assert np.array_equal(saida[1], np.asarray(matriz.values)[1]), (
        "a política do `[CA-5.3]` não muda: a linha degenerada continua a sair intacta"
    )
    assert not np.array_equal(saida[0], np.asarray(matriz.values)[0]), (
        "as outras linhas têm de ter sido normalizadas, senão a planta não separa nada"
    )


def test_a_declaracao_vai_na_PROCEDENCIA_e_nao_so_no_par_de_retorno() -> None:  # noqa: N802
    """Quem chama `normalize_samples` por fora do pipeline leva a matriz e mais nada.

    E a nota mantém o prefixo antes do parêntese, que é o que o teste da ordem canónica lê.
    """
    saida = normalize_samples(_com_linha_de_soma_zero(), "sum")
    nota = saida.provenance[-1]

    assert nota.split("(")[0] == "normalize_samples"
    assert "sem denominador" in nota and "S1" in nota, (
        f"a procedência é {nota!r} e não declara a linha deixada como estava"
    )


def test_sem_linha_degenerada_a_declaracao_e_VAZIA_e_a_nota_e_a_de_sempre() -> None:  # noqa: N802
    """A outra direção do par: com a planta removida, a guarda não acusa.

    É o que separa guarda ativa de guarda que acusa sempre.
    """
    valores = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    declarada = normalize_samples_declarada(_matriz(valores, n_qc=1), "sum")

    assert declarada.linhas_sem_denominador == ()
    assert declarada.matrix.provenance[-1] == "normalize_samples(sum)"


def test_pqn_sem_qc_declara_a_matriz_INTEIRA_como_sem_denominador() -> None:  # noqa: N802
    """O caso extremo: a identidade que a docstring declarava e a matriz não.

    `test_pqn_sem_qc_e_identidade` afirma que os valores não mudam, e continuam a não mudar. O que
    muda é a matriz passar a **dizer** que não mudou, em vez de sair com a etiqueta
    `normalize_samples(pqn_group)` como se tivesse normalizado.
    """
    valores = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    amostras = tuple(
        SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type="sample", group="G")
        for i in range(2)
    )
    matriz = RefinoMatrix(
        values=valores, feature_names=("a", "b"), samples=amostras, mode="sintetico"
    )
    declarada = normalize_samples_declarada(matriz, "pqn_group")

    assert declarada.linhas_sem_denominador == ("S0", "S1")
    assert np.array_equal(np.asarray(declarada.matrix.values), valores)


def test_o_registo_recusa_denominador_QUE_NAO_E_POR_LINHA() -> None:  # noqa: N802
    """A guarda de eixo do próprio instrumento.

    `_seguro` também serve o escalonamento, que divide por COLUNA. Ligar o registo ali daria uma
    contagem sobre o eixo errado, com aparência de contagem de amostras. O instrumento recusa em
    vez de responder o número errado.
    """
    from refino.preprocessing.normalization import _LinhasIntactas, _seguro

    numerador = np.ones((4, 3))
    with pytest.raises(AssertionError, match="conta LINHAS"):
        _seguro(numerador, np.ones((1, 3)), registro=_LinhasIntactas())


def test_o_escalonamento_com_coluna_constante_sai_a_ZERO_e_nao_por_escalonar() -> None:  # noqa: N802
    """Por que o registo NÃO é ligado no escalonamento, medido em vez de argumentado.

    `_seguro(centrados, desvio)` com desvio zero devolve `centrados` — e desvio zero implica
    coluna constante, que implica centrado **exactamente zero**. A coluna degenerada sai a zero
    como qualquer outra coluna centrada, e não "com o valor bruto". O caso do `normalize_samples`
    é outro: ali o numerador é o valor cru, e ele sai cru.
    """
    valores = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]], dtype=np.float64)
    saida = np.asarray(scale_features(_matriz(valores, n_qc=1), "auto").values)

    assert np.array_equal(saida[:, 1], np.zeros(3)), (
        "a coluna constante saiu com o valor bruto e não a zero; então o argumento acima está "
        "errado e o registo precisa de alcançar o escalonamento também"
    )


# ---------------------------------------------------------------------------
# OQ3 — o caminho para a matriz pré-transformação
# ---------------------------------------------------------------------------


def _spec_de_oq3(pipeline_spec: object) -> object:
    """A especificação dos dois testes de OQ3, com TODOS os campos escritos.

    `log10` e `auto` não são detalhe: com `transformation="none"` e `scaling="none"` as duas
    matrizes seriam a mesma, e os testes passariam sem exercitar nada.
    """
    return pipeline_spec(  # type: ignore[operator]
        correction="none",
        n_batches=1,
        rsd_cutoff=None,
        ranking_filter="none",
        ranking_keep_fraction=None,
        ranking_filter_2="none",
        ranking_keep_fraction_2=None,
        rsd_variant="outlier_removed",
        missing_threshold=None,
        imputation="one_fifth_min",
        sample_normalization="sum",
        normalization_reference=None,
        normalization_factors=None,
        transformation="log10",
        scaling="auto",
    )


def test_oq3_a_matriz_pre_transformacao_e_a_de_DEPOIS_da_normalizacao() -> None:  # noqa: N802
    """OQ3: `process_matrix_detailed` devolve o ponto em que o MetaboAnalyst calcula o FC.

    A âncora é exacta e não estatística: `pre_transform` tem de ser **bit a bit** o que
    `normalize_samples` devolve sobre a matriz que chegou às etapas transformadoras. Com
    `transformation="none"` e `scaling="none"` as duas matrizes coincidem, o que não prova nada —
    por isso o fixture usa `log10` e `auto`, e o teste afirma que elas **diferem**.
    """
    from refino.domain.spec import PipelineSpec
    from refino.preprocessing.pipeline import process_matrix, process_matrix_detailed

    gerador = np.random.default_rng(7)
    matriz = _matriz(gerador.lognormal(8.0, 0.3, size=(9, 5)), n_qc=3)
    spec = _spec_de_oq3(PipelineSpec)

    detalhado = process_matrix_detailed(matriz, spec)

    assert detalhado.transform_moved_the_matrix, (
        "a pré-transformação e a final são iguais, então este fixture não exercita OQ3"
    )
    assert detalhado.pre_transform.feature_names == detalhado.matrix.feature_names, (
        "os dois eixos do volcano dependem do índice partilhado; se ele partir, o `p` de uma "
        "feature fica ao lado do fold change de outra"
    )
    assert detalhado.pre_transform.sample_names == detalhado.matrix.sample_names
    assert detalhado.pre_transform.provenance[-1].startswith("normalize_samples(")
    assert detalhado.matrix.provenance[-1].startswith("scale_features(")

    # E o contrato de sempre continua a ser o de sempre: uma implementação, duas vistas.
    final, etapas = process_matrix(matriz, spec)
    assert np.array_equal(np.asarray(final.values), np.asarray(detalhado.matrix.values))
    assert [e.stage for e in etapas] == [e.stage for e in detalhado.stages]


def test_oq3_o_par_de_matrizes_e_aceite_por_univariate_test() -> None:
    """A ponta a ponta: o que `process_matrix_detailed` devolve serve o parâmetro.

    Sem isto, as duas metades desta unidade poderiam estar certas e não encaixar — a fronteira de
    `univariate_test` recusaria o par que o pipeline produz, e ninguém saberia até a análise
    univariada.
    """
    from refino.domain.spec import PipelineSpec
    from refino.preprocessing.pipeline import process_matrix_detailed
    from refino.statistics.univariate import univariate_test

    gerador = np.random.default_rng(23)
    valores = gerador.lognormal(8.0, 0.3, size=(12, 5))
    valores[6:, :2] *= 3.0
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i < 3 else "sample",
            group="QC" if i < 3 else ("A" if i < 8 else "B"),
        )
        for i in range(12)
    )
    matriz = RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(5)),
        samples=amostras,
        mode="sintetico",
    )
    spec = _spec_de_oq3(PipelineSpec)

    detalhado = process_matrix_detailed(matriz, spec)
    resultado = univariate_test(
        detalhado.matrix, ("A", "B"), fold_change_matrix=detalhado.pre_transform
    )

    assert resultado.fold_change_from_other_matrix is True
    padrao = univariate_test(detalhado.matrix, ("A", "B"))
    assert resultado.counts.log2fc_undefined < padrao.counts.log2fc_undefined, (
        f"com a pré-transformação ficaram {resultado.counts.log2fc_undefined} fold changes "
        f"indefinidos e sem ela {padrao.counts.log2fc_undefined}; se forem iguais, este fixture "
        "não exercita o motivo de KTD8 existir"
    )

"""O intervalo de 95% do fold change — Bonett-Price para a razão de medianas.

**Toda matriz aqui é CONSTRUÍDA, e dizê-lo é requisito (regra 1 do README).** O oráculo é o exemplo
documentado do `statpsych` — dez e treze números do manual do pacote, não dado do laboratório —, e a
cobertura é medida em simulação com a verdade conhecida, que é o único sítio onde cobertura existe.
O que estes testes afirmam é propriedade da função e do executor do pipeline, não contagem observada
nos quatro arquivos: essa vai para a decisão, medida sobre eles.

**De onde vem o oráculo, conferido na fonte e não no dossiê que o trouxe:**
`man/ci.ratio.median2.Rd` e `R/statpsych1.R` do `statpsych` 2.0.0, no espelho do CRAN no GitHub,
lidos em 06/11/2026. O manual traz o exemplo e a saída, *«Should return: 43 37 1.162162 0.927667
1.455933»*, e o código traz a fórmula que `fold_change_interval` transcreve. Não há R nesta máquina:
o pacote não correu, e é o número documentado que serve de vetor fixo.
"""

import numpy as np
import pytest
from scipy.stats import gamma, ttest_ind

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.spec import PipelineSpec
from refino.preprocessing.missing import is_missing
from refino.preprocessing.pipeline import process_matrix_detailed
from refino.statistics.univariate import fold_change_interval, univariate_test

pytestmark = pytest.mark.property

SEMENTE = 20261106

# O exemplo do manual do `statpsych`, na ordem em que o manual os escreve.
Y1_DO_STATPSYCH = (36, 44, 47, 42, 49, 39, 46, 31, 33, 48)
Y2_DO_STATPSYCH = (32, 39, 26, 35, 43, 27, 40, 37, 34, 29, 49, 42, 40)

# Os tamanhos dos dois grupos do par que a tela usa por padrão, `Asymptomatic` e `HAMTSP`, neste
# estudo. Observação sobre estes arquivos e não invariante: servem para a simulação ter os
# `n` em que o intervalo vai correr de facto.
N_A, N_B = 18, 14

# **O critério da cobertura, escrito ANTES de medir.** Unilateral, porque o defeito que
# importa é o intervalo cobrir MENOS do que promete: 0,95 menos três erros-padrão binomiais de 2.000
# réplicas. O dossiê do plano mediu 0,961 a 0,967 com estes tamanhos — o intervalo é um pouco
# conservador —, e a exploração desta unidade, 0,9615 a 0,9650 em três sementes.
REPLICAS = 2_000
PISO_DA_COBERTURA = 0.95 - 3.0 * np.sqrt(0.95 * 0.05 / REPLICAS)


def _matriz(grupos: list[str], valores: np.ndarray) -> RefinoMatrix:
    amostras = tuple(
        SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type="sample", group=grupo)
        for i, grupo in enumerate(grupos)
    )
    return RefinoMatrix(
        values=np.asarray(valores, dtype=np.float64),
        feature_names=tuple(f"f{j}" for j in range(valores.shape[1])),
        samples=amostras,
        mode="sintetico",
    )


def _par(a: np.ndarray, b: np.ndarray) -> RefinoMatrix:
    """Os dois blocos empilhados, `A` no numerador e `B` no denominador."""
    return _matriz(["A"] * a.shape[0] + ["B"] * b.shape[0], np.vstack([a, b]))


def _nenhuma_imputada(matriz: RefinoMatrix) -> np.ndarray:
    return np.zeros(np.asarray(matriz.values).shape, dtype=bool)


def _spec(imputation: str, transformation: str = "log10", scaling: str = "auto") -> PipelineSpec:
    """A especificação com TODOS os campos escritos, e nenhuma etapa que remova feature.

    `log10` e `auto` por omissão pelo motivo de `_spec_de_oq3` em `test_normalization.py`: com as
    duas em `none` a pré-transformação seria a matriz final, e o teste não separaria as duas.
    """
    return PipelineSpec(
        correction="none",
        n_batches=1,
        rsd_cutoff=None,
        ranking_filter="none",
        ranking_keep_fraction=None,
        ranking_filter_2="none",
        ranking_keep_fraction_2=None,
        rsd_variant="outlier_removed",
        missing_threshold=None,
        imputation=imputation,  # type: ignore[arg-type]
        sample_normalization="none",
        normalization_reference=None,
        normalization_factors=None,
        transformation=transformation,  # type: ignore[arg-type]
        scaling=scaling,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# O oráculo
# ---------------------------------------------------------------------------


def test_o_exemplo_documentado_do_statpsych_a_seis_casas() -> None:
    """Razão 1,162162 e limites 0,927667 e 1,455933 — a saída que o manual do pacote documenta.

    **`abs=5e-7` é «arredonda para o número documentado»**, meia unidade da sexta casa, e não uma
    tolerância escolhida: o manual imprime seis casas, e é isso que se pode afirmar dele.
    """
    valores = np.array(Y1_DO_STATPSYCH + Y2_DO_STATPSYCH, dtype=np.float64)[:, None]
    matriz = _matriz(["A"] * 10 + ["B"] * 13, valores)

    intervalo = fold_change_interval(matriz, ("A", "B"), _nenhuma_imputada(matriz))
    ponto = univariate_test(matriz, ("A", "B")).fold_change

    assert intervalo.undefined == (None,), intervalo.undefined
    assert ponto[0] == pytest.approx(1.162162, abs=5e-7)
    assert intervalo.lower[0] == pytest.approx(0.927667, abs=5e-7)
    assert intervalo.upper[0] == pytest.approx(1.455933, abs=5e-7)
    # As duas escalas são o mesmo intervalo: em log2 é o logaritmo dos limites na razão.
    assert intervalo.log2_lower[0] == pytest.approx(np.log2(intervalo.lower[0]), abs=1e-12)
    assert intervalo.log2_upper[0] == pytest.approx(np.log2(intervalo.upper[0]), abs=1e-12)
    assert intervalo.confidence == 0.95
    assert intervalo.groups == ("A", "B")
    assert intervalo.feature == ("f0",)


# ---------------------------------------------------------------------------
# A cobertura, e a planta que prova que o critério consegue reprovar
# ---------------------------------------------------------------------------


def _sorteia(cenario: str, gerador: np.random.Generator, n: int, mediana: float) -> np.ndarray:
    """`REPLICAS` colunas de `n` valores cuja mediana POPULACIONAL é `mediana`.

    Na lognormal a mediana é `exp(mu)`; na assimétrica em log, `log Y = 0,5 (G - mediana(G))` com
    `G ~ Gama(2)` põe a mediana de `log Y` em zero, e a de `Y` em `mediana`. As duas lognormais têm
    as larguras das linhas do dossiê; o intervalo é quase equivariante à escala do log — não
    exactamente, porque a mediana de `n` par é média na escala linear —, e por isso as duas dão
    coberturas próximas e não iguais.
    """
    if cenario == "lognormal_0.5":
        return mediana * np.exp(gerador.normal(0.0, 0.5, (n, REPLICAS)))
    if cenario == "lognormal_1.0":
        return mediana * np.exp(gerador.normal(0.0, 1.0, (n, REPLICAS)))
    if cenario == "assimetrica_em_log":
        g = gerador.gamma(2.0, 1.0, (n, REPLICAS))
        return mediana * np.exp(0.5 * (g - gamma.median(2.0)))
    raise AssertionError(f"cenário {cenario!r} sem sorteio")


@pytest.mark.synthetic
@pytest.mark.parametrize("cenario", ["lognormal_0.5", "lognormal_1.0", "assimetrica_em_log"])
def test_a_cobertura_em_simulacao_passa_o_piso_e_a_planta_estreita_nao(cenario: str) -> None:
    """**Cada réplica é uma FEATURE**, e as 2.000 correm numa chamada só da função de produção.

    É a vetorização que torna a simulação barata — milissegundos — e é a mesma função que a rota
    chama, não uma transcrição à parte.

    **O par causal é a planta estreita.** O intervalo encolhido por `√2` na escala log é o que daria
    esquecer o erro-padrão de um dos dois grupos quando os dois são parecidos — o defeito de
    transcrição mais provável na soma `√(SE₁² + SE₂²)`. Medido na exploração, em três cenários e
    três sementes: 0,8635 a 0,8825. Se ele passasse o piso, o piso não teria poder para acusar o
    defeito que existe para acusar.
    """
    gerador = np.random.default_rng(SEMENTE)
    matriz = _par(_sorteia(cenario, gerador, N_A, 1.5), _sorteia(cenario, gerador, N_B, 1.0))

    intervalo = fold_change_interval(matriz, ("A", "B"), _nenhuma_imputada(matriz))

    assert all(m is None for m in intervalo.undefined), (
        "âncora: a cobertura abaixo seria sobre menos réplicas do que as declaradas"
    )
    alvo = np.log(1.5)
    inferior, superior = np.log(intervalo.lower), np.log(intervalo.upper)
    cobertura = float(np.mean((inferior <= alvo) & (alvo <= superior)))

    centro = (inferior + superior) / 2.0
    meia_da_planta = (superior - inferior) / 2.0 / np.sqrt(2.0)
    cobre_a_planta = (centro - meia_da_planta <= alvo) & (alvo <= centro + meia_da_planta)
    cobertura_da_planta = float(np.mean(cobre_a_planta))

    assert cobertura >= PISO_DA_COBERTURA, (
        f"{cenario}: cobertura {cobertura:.4f} abaixo do piso {PISO_DA_COBERTURA:.4f} em "
        f"{REPLICAS} réplicas — o intervalo promete 95% e cobre menos"
    )
    assert cobertura_da_planta < PISO_DA_COBERTURA, (
        f"{cenario}: a planta estreita por √2 cobriu {cobertura_da_planta:.4f}, acima do piso — o "
        "critério não acusaria o defeito de esquecer o erro-padrão de um grupo"
    )


# ---------------------------------------------------------------------------
# O centro: o ponto desenhado, e a planta de Welch em log
# ---------------------------------------------------------------------------


def test_o_centro_e_o_ponto_desenhado_e_o_de_welch_em_log_sai_descentrado() -> None:
    """**A razão de escolher Bonett-Price, como par causal.** O ponto da figura é a razão de
    MEDIANAS; o intervalo de Welch na escala log é centrado na razão de médias GEOMÉTRICAS, que só
    coincide com aquela quando os dois grupos são lognormais da mesma forma.

    **A planta cai sobre dado em que as duas quantidades diferem**: o grupo `A` é
    assimétrico na escala log e o `B` é lognormal — medido na exploração, o centro de Welch fica a
    0,40 em log2 do ponto na mediana das 40 features, e o de Bonett-Price a 4e-16.
    """
    gerador = np.random.default_rng(SEMENTE)
    n_features = 40
    a = 1.5 * np.exp(0.8 * (gerador.gamma(2.0, 1.0, (N_A, n_features)) - gamma.median(2.0)))
    b = np.exp(gerador.normal(0.0, 0.8, (N_B, n_features)))
    matriz = _par(a, b)

    ponto = univariate_test(matriz, ("A", "B")).log2fc
    intervalo = fold_change_interval(matriz, ("A", "B"), _nenhuma_imputada(matriz))
    welch = ttest_ind(np.log2(a), np.log2(b), axis=0, equal_var=False).confidence_interval(0.95)

    assert np.isfinite(ponto).all(), "âncora: sem ponto definido a comparação seria vácua"
    centro_bp = (intervalo.log2_lower + intervalo.log2_upper) / 2.0
    centro_welch = (np.asarray(welch.low) + np.asarray(welch.high)) / 2.0

    assert np.max(np.abs(centro_bp - ponto)) < 1e-12
    assert np.median(np.abs(centro_welch - ponto)) > 0.05, (
        "o intervalo de Welch em log ficou centrado no ponto — este dado não separa as duas "
        "quantidades, e o teste não provaria nada"
    )


# ---------------------------------------------------------------------------
# Os três motivos de intervalo indefinido
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("valor", [0.0, -1.0])
def test_valor_de_ordem_nao_positivo_da_intervalo_indefinido_e_nunca_inf(valor: float) -> None:
    """**O ponto existe e o intervalo não, e é aqui que as duas coisas diferem.**

    Com 14 valores, `a = round(7 - √14) = 3`: três valores ≤ 0 no grupo `B` põem `y(3)` em zero ou
    abaixo, e `ln` dele é `-inf` ou indefinido. A mediana dos 14 continua positiva, então o fold
    change é finito — e a fórmula sem guarda devolveria os limites `0` e `inf`, que se leem como
    medida. `ln 0` e a recusa do `statpsych` para valor negativo são os dois desfechos que a guarda
    substitui pelo motivo.
    """
    gerador = np.random.default_rng(SEMENTE)
    valores = np.exp(gerador.normal(3.0, 0.4, (N_A + N_B, 2)))
    valores[N_A : N_A + 3, 0] = valor
    matriz = _matriz(["A"] * N_A + ["B"] * N_B, valores)

    intervalo = fold_change_interval(matriz, ("A", "B"), _nenhuma_imputada(matriz))
    ponto = univariate_test(matriz, ("A", "B")).fold_change

    assert intervalo.undefined[1] is None and np.isfinite(intervalo.lower[1]), (
        "âncora: a feature de controlo tem de ter intervalo finito"
    )
    assert np.isfinite(ponto[0]), "o ponto tinha de existir — senão este é outro caso"
    assert intervalo.undefined[0] == "order_statistic_not_positive", intervalo.undefined
    for limites in (
        intervalo.lower,
        intervalo.upper,
        intervalo.log2_lower,
        intervalo.log2_upper,
    ):
        assert np.isnan(limites[0]), limites
        assert not np.isinf(limites).any(), limites


@pytest.mark.parametrize("valor", [np.nan, np.inf])
def test_celula_nao_finita_no_grupo_da_intervalo_indefinido(valor: float) -> None:
    """**O terceiro motivo, e o plano previa dois.** `imputation="none"` é aceita quando as três
    normalizações são `none`, e então a célula não finita que o corretor deixa chega
    à matriz do fold change. A mediana com `nan` é `nan` — não há ponto em volta do qual centrar —,
    e chamar-lhe «valor de ordem ≤ 0» seria um motivo com dois significados.
    """
    gerador = np.random.default_rng(SEMENTE)
    valores = np.exp(gerador.normal(3.0, 0.4, (N_A + N_B, 2)))
    valores[2, 0] = valor
    matriz = _matriz(["A"] * N_A + ["B"] * N_B, valores)

    intervalo = fold_change_interval(matriz, ("A", "B"), _nenhuma_imputada(matriz))

    assert intervalo.undefined[1] is None and np.isfinite(intervalo.upper[1]), "âncora"
    assert intervalo.undefined[0] == "non_finite_value", intervalo.undefined
    assert np.isnan(intervalo.lower[0]) and np.isnan(intervalo.upper[0])
    assert not np.isinf(intervalo.upper).any()


def test_um_grupo_de_18_com_10_imputados_da_mediana_sobre_valor_imputado() -> None:
    """**Pelo pipeline e não por máscara feita à mão**: dez zeros entram, `one_fifth_min` os troca
    pelo quinto do menor positivo, e o executor marca quais foram.

    Sem a guarda o intervalo existiria, e estreito: a mediana dos 18 cai entre o nono e o décimo
    valores, os dois imputados e iguais, e o intervalo seria uma medida sobre um número que a
    imputação escolheu.
    """
    gerador = np.random.default_rng(SEMENTE)
    valores = np.exp(gerador.normal(8.0, 0.4, (N_A + N_B, 3)))
    valores[:10, 0] = 0.0
    matriz = _matriz(["A"] * N_A + ["B"] * N_B, valores)

    detalhada = process_matrix_detailed(matriz, _spec("one_fifth_min"))
    intervalo = fold_change_interval(detalhada.pre_transform, ("A", "B"), detalhada.imputed_cells)

    assert detalhada.pre_transform.feature_names == matriz.feature_names, (
        "uma etapa removeu feature, e as posições abaixo deixariam de ser as do fixture"
    )
    assert int(detalhada.imputed_cells[:, 0].sum()) == 10
    assert intervalo.undefined[1] is None and np.isfinite(intervalo.lower[1]), "âncora"
    assert intervalo.undefined[0] == "median_over_imputed_values", intervalo.undefined

    # E o ponto é o da rota: `univariate_test` com a pré-transformação como `fold_change_matrix`.
    # Mesma matriz, mesmas features, e o intervalo centrado no `log2fc` que ela devolve.
    ponto = univariate_test(
        detalhada.matrix, ("A", "B"), fold_change_matrix=detalhada.pre_transform
    )
    assert intervalo.provenance == ponto.fold_change_provenance
    assert intervalo.feature == ponto.feature
    centro = (intervalo.log2_lower[1:] + intervalo.log2_upper[1:]) / 2.0
    assert np.max(np.abs(centro - ponto.log2fc[1:])) < 1e-12
    assert np.isnan(intervalo.lower[0]) and np.isnan(intervalo.log2_upper[0])


@pytest.mark.parametrize(("imputadas", "motivo"), [(9, "median_over_imputed_values"), (8, None)])
def test_a_fronteira_e_metade_dos_valores_do_grupo(imputadas: int, motivo: str | None) -> None:
    """«Metade ou mais»: com 18, nove imputados já põem um deles na mediana, e oito não."""
    gerador = np.random.default_rng(SEMENTE)
    matriz = _matriz(["A"] * N_A + ["B"] * N_B, np.exp(gerador.normal(3.0, 0.4, (N_A + N_B, 1))))
    mascara = _nenhuma_imputada(matriz)
    mascara[:imputadas, 0] = True

    assert fold_change_interval(matriz, ("A", "B"), mascara).undefined == (motivo,)


def test_cada_feature_recebe_o_PRIMEIRO_motivo_da_ordem_declarada() -> None:  # noqa: N802
    """Os motivos são uma PARTIÇÃO — a contagem por motivo vai para a decisão —, então a feature
    que cai em dois recebe um só, pela ordem não finito, valor de ordem, imputação.
    """
    gerador = np.random.default_rng(SEMENTE)
    valores = np.exp(gerador.normal(3.0, 0.4, (N_A + N_B, 2)))
    valores[0, 0] = np.nan  # não finito E metade imputada
    valores[:5, 1] = 0.0  # valor de ordem ≤ 0 (a = 5 com 18) E metade imputada
    matriz = _matriz(["A"] * N_A + ["B"] * N_B, valores)
    mascara = _nenhuma_imputada(matriz)
    mascara[:9, :] = True

    assert fold_change_interval(matriz, ("A", "B"), mascara).undefined == (
        "non_finite_value",
        "order_statistic_not_positive",
    )


def test_a_mascara_de_imputadas_com_outra_forma_e_recusada() -> None:
    """A máscara indexa as MESMAS células da matriz; com outra forma ela marcaria outras."""
    matriz = _matriz(["A"] * 3 + ["B"] * 3, np.ones((6, 4)))
    with pytest.raises(ValueError, match="forma"):
        fold_change_interval(matriz, ("A", "B"), np.zeros((6, 3), dtype=bool))


# ---------------------------------------------------------------------------
# As células imputadas, marcadas pelo executor do pipeline
# ---------------------------------------------------------------------------


def _com_ausentes() -> RefinoMatrix:
    gerador = np.random.default_rng(SEMENTE)
    valores = np.exp(gerador.normal(8.0, 0.4, (12, 4)))
    valores[[0, 3, 7], 1] = 0.0
    valores[5, 2] = np.nan
    return _matriz(["A"] * 6 + ["B"] * 6, valores)


def test_as_celulas_imputadas_sao_as_ausentes_da_matriz_que_entrou_na_imputacao() -> None:
    """Zero ou não finita na entrada da imputação — `is_missing`, a definição que o executor já usa.

    **E a máscara é só leitura**, pela regra 7 do README: ela viaja num `dataclass` congelado, e o
    congelamento do atributo não alcança o conteúdo do array.
    """
    matriz = _com_ausentes()
    detalhada = process_matrix_detailed(matriz, _spec("one_fifth_min"))

    assert detalhada.pre_transform.feature_names == matriz.feature_names
    esperada = is_missing(np.asarray(matriz.values))
    assert esperada.sum() == 4, "âncora: o fixture tem de ter as quatro ausentes"
    assert np.array_equal(detalhada.imputed_cells, esperada)
    assert not is_missing(np.asarray(detalhada.pre_transform.values)).any(), (
        "a imputação deixou ausência, e a máscara descreveria células que não foram preenchidas"
    )
    assert not detalhada.imputed_cells.flags.writeable


def test_sem_imputacao_nenhuma_celula_e_imputada() -> None:
    """**O par causal do método:** a mesma entrada, com `none`, não tem célula imputada nenhuma.

    As ausentes continuam ausentes, e o intervalo delas cai por valor de ordem ou por não finito —
    o motivo verdadeiro —, e não por «mediana sobre valor imputado», que seria falso.
    """
    matriz = _com_ausentes()
    detalhada = process_matrix_detailed(matriz, _spec("none", "none", "none"))

    assert is_missing(np.asarray(detalhada.pre_transform.values)).sum() == 4, "âncora"
    assert detalhada.imputed_cells.shape == np.asarray(detalhada.pre_transform.values).shape
    assert not detalhada.imputed_cells.any()


def test_exclude_remove_as_features_com_ausente_e_a_mascara_acompanha() -> None:
    """`exclude` é o único método que muda a forma: a máscara tem a forma da matriz que SAIU."""
    matriz = _com_ausentes()
    detalhada = process_matrix_detailed(matriz, _spec("exclude"))

    assert detalhada.pre_transform.feature_names == ("f0", "f3")
    assert detalhada.imputed_cells.shape == (12, 2)
    assert not detalhada.imputed_cells.any()

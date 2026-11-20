"""`univariate_test` — a tabela por feature, as guardas de fronteira, e o par da decisão.

O critério — poder acima de 80% — vive em `tests/integration/test_statistics_modos.py`, porque
poder é propriedade de um procedimento sobre **repetições** e exige dado real. Aqui ficam as
propriedades da função.

**A matriz com amostra `validate` é SINTÉTICA, e a declaração é requisito (regra 1 do README).**
Medido em 17/09/2026 sobre os quatro `measurements_*.csv` do laboratório: 68 linhas, 61 `sample`, 7
`qc` e **zero** `validate` nos quatro modos. A amostra `validate` de rótulo clínico **não existe no
corpus** — fabricá-la sobre os arquivos reais e apresentá-la como observação seria o que a
regra 1 do README proíbe, e por isso ela é construída à mão e dita como construída. O que os testes
abaixo afirmam é a propriedade de desenho, que não depende do corpus de hoje.
"""

from collections.abc import Callable

import numpy as np
import pytest
from scipy.stats import shapiro

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.statistics.univariate import (
    MAX_PARA_SHAPIRO,
    NORMALITY_ALPHA,
    _amostra_espalhada,
    univariate_test,
)

pytestmark = pytest.mark.property


def _matriz(valores: np.ndarray, grupos: list[str], tipos: list[str] | None = None) -> RefinoMatrix:
    tipos = tipos if tipos is not None else ["sample"] * len(grupos)
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type=tipo,  # type: ignore[arg-type]
            group=grupo,
        )
        for i, (grupo, tipo) in enumerate(zip(grupos, tipos, strict=True))
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(valores.shape[1])),
        samples=amostras,
        mode="sintetico",
    )


def _com_efeito(n_por_grupo: int, n_features: int, alteradas: int, fator: float, semente: int):
    gerador = np.random.default_rng(semente)
    valores = gerador.normal(10.0, 1.0, size=(2 * n_por_grupo, n_features))
    valores[n_por_grupo:, :alteradas] *= fator
    grupos = ["A"] * n_por_grupo + ["B"] * n_por_grupo
    return _matriz(valores, grupos)


def test_as_seis_colunas_da_especificacao_existem_como_campos() -> None:
    """O contrato, campo a campo — o resultado é dataclass, não `DataFrame`."""
    resultado = univariate_test(_com_efeito(6, 5, 2, 1.5, 7), ("A", "B"))

    for campo in ("feature", "statistic", "pvalue", "qvalue_fdr", "fold_change", "log2fc"):
        assert hasattr(resultado, campo), f"a coluna {campo!r} da §7 Fase 6 desapareceu"
    assert len(resultado.feature) == 5
    for campo in ("statistic", "pvalue", "qvalue_fdr", "fold_change", "log2fc"):
        assert getattr(resultado, campo).shape == (5,)


def test_os_rotulos_da_medicao_viajam_com_os_numeros() -> None:
    """`statistic` de Welch e de Mann-Whitney têm o mesmo nome e são grandezas diferentes."""
    matriz = _com_efeito(8, 4, 2, 2.0, 11)
    assert univariate_test(matriz, ("A", "B"), parametric=True).parametric is True
    assert univariate_test(matriz, ("A", "B"), parametric=False).parametric is False
    assert univariate_test(matriz, ("A", "B"), parametric=True).groups == ("A", "B")


def test_o_parametro_parametric_e_consultado() -> None:
    """O par causal do parâmetro: os dois valores dão `pvalue` diferente.

       Sem esta asserção, `parametric` poderia ser ignorado e a suíte inteira passaria — é a forma
    do lado do parâmetro, e ela exige entrada em que os dois testes divergem.
    """
    matriz = _com_efeito(10, 6, 3, 2.0, 13)
    welch = univariate_test(matriz, ("A", "B"), parametric=True)
    mann = univariate_test(matriz, ("A", "B"), parametric=False)

    assert not np.allclose(welch.pvalue, mann.pvalue), (
        "Welch e Mann-Whitney deram os mesmos p-valores; então `parametric` não está sendo "
        "consultado, ou a entrada não os distingue"
    )


def test_log2fc_e_o_log2_do_fold_change() -> None:
    """Coerência entre as duas colunas, a 1e-12 — duas colunas que discordam são defeito."""
    resultado = univariate_test(_com_efeito(7, 8, 4, 3.0, 17), ("A", "B"))
    finitos = np.isfinite(resultado.fold_change) & (resultado.fold_change > 0)
    assert np.allclose(
        resultado.log2fc[finitos], np.log2(resultado.fold_change[finitos]), atol=1e-12
    )


def test_fold_change_com_denominador_zero_e_nan_e_nao_inf() -> None:
    """A política declarada: razão sem denominador não tem valor de identidade."""
    valores = np.ones((8, 3))
    valores[4:, 1] = 0.0  # mediana do grupo B zerada na feature 1
    resultado = univariate_test(_matriz(valores, ["A"] * 4 + ["B"] * 4), ("A", "B"))

    assert np.isnan(resultado.fold_change[1])
    assert np.isnan(resultado.log2fc[1])
    assert not np.isinf(resultado.fold_change).any()


def test_qvalue_vem_do_benjamini_hochberg_e_e_monotonico() -> None:
    """O `q` não é segunda implementação: é `benjamini_hochberg`, e a monotonicidade prova."""
    resultado = univariate_test(_com_efeito(9, 40, 10, 2.5, 19), ("A", "B"))
    ordem = np.argsort(resultado.pvalue, kind="stable")
    assert np.all(np.diff(resultado.qvalue_fdr[ordem]) >= -1e-15)


def test_grupo_inexistente_levanta_nomeando_os_disponiveis() -> None:
    """Fronteira: nome de grupo vem de arquivo do laboratório e o domínio é aberto."""
    with pytest.raises(ValueError, match="não está entre as amostras de estudo"):
        univariate_test(_com_efeito(5, 3, 1, 2.0, 23), ("A", "Z"))


def test_os_dois_grupos_iguais_levanta() -> None:
    with pytest.raises(ValueError, match="são o mesmo"):
        univariate_test(_com_efeito(5, 3, 1, 2.0, 29), ("A", "A"))


def test_grupo_com_uma_amostra_levanta() -> None:
    """`n = 1` não tem estatística, em nenhum dos dois testes."""
    valores = np.ones((4, 3))
    with pytest.raises(ValueError, match="exige ao menos"):
        univariate_test(_matriz(valores, ["A", "A", "A", "B"]), ("A", "B"))


def test_determinismo() -> None:
    """Sem sorteio no caminho: duas execuções dão vetor idêntico."""
    matriz = _com_efeito(8, 10, 3, 2.0, 31)
    a, b = univariate_test(matriz, ("A", "B")), univariate_test(matriz, ("A", "B"))
    assert np.array_equal(a.pvalue, b.pvalue, equal_nan=True)
    assert np.array_equal(a.qvalue_fdr, b.qvalue_fdr, equal_nan=True)


# ---------------------------------------------------------------------------
# `fold_change_matrix`, o parâmetro que diz de qual matriz sai o fold change
# ---------------------------------------------------------------------------


def _par_de_matrizes() -> tuple[RefinoMatrix, RefinoMatrix]:
    """A pré-transformação e a processada, com o desfecho **medido e não suposto**.

    O caso é o normal e não o de borda: `log10` seguido de centragem põe medianas negativas, e a
    razão de medianas fica indefinida. A função afirma as duas metades antes de devolver, porque
    um par em que as duas matrizes dessem o mesmo fold change faria todo teste abaixo passar por
    coincidência — é a planta que não separa.
    """
    gerador = np.random.default_rng(101)
    brutos = gerador.lognormal(8.0, 0.4, size=(12, 6))
    brutos[6:, :3] *= 4.0  # sinal plantado: o grupo B é 4x o grupo A nas três primeiras
    grupos = ["A"] * 6 + ["B"] * 6

    pre = _matriz(brutos, grupos).with_values(brutos, "normalize_samples(none)")
    valores = np.log10(brutos)
    processada = pre.with_values(valores - valores.mean(axis=0, keepdims=True), "scale(mean)")

    antes = univariate_test(pre, ("A", "B"))
    depois = univariate_test(processada, ("A", "B"))
    assert antes.counts.log2fc_undefined == 0, (
        "a pré-transformação tinha de ter fold change em tudo"
    )
    assert depois.counts.log2fc_undefined > 0, (
        "a matriz processada deste fixture não tem mediana negativa nenhuma, então ela não separa "
        "as duas políticas e nenhum teste abaixo mede alguma coisa"
    )
    return pre, processada


def test_o_fold_change_sai_da_matriz_do_PARAMETRO_e_o_p_da_matriz_do_teste() -> None:  # noqa: N802
    """Caminho feliz: com o parâmetro o fold change é finito e bate com a conta à mão."""
    pre, processada = _par_de_matrizes()
    resultado = univariate_test(processada, ("A", "B"), fold_change_matrix=pre)

    bruto = np.asarray(pre.values)
    esperado = np.median(bruto[:6], axis=0) / np.median(bruto[6:], axis=0)
    assert np.allclose(resultado.fold_change, esperado, rtol=1e-12)

    # E o `p` continua a ser o da matriz processada, byte a byte.
    assert np.array_equal(
        resultado.pvalue, univariate_test(processada, ("A", "B")).pvalue, equal_nan=True
    )


def test_par_causal_do_parametro_as_duas_contagens_de_indefinido_diferem() -> None:
    """A mesma matriz pelos dois valores do parâmetro, e as duas contagens vão no relatório.

       Sem este par, `fold_change_matrix` poderia ser ignorado e a suíte inteira passaria — é a
       forma
    do lado do parâmetro.
    """
    pre, processada = _par_de_matrizes()
    padrao = univariate_test(processada, ("A", "B"))
    com_parametro = univariate_test(processada, ("A", "B"), fold_change_matrix=pre)

    assert com_parametro.counts.log2fc_undefined < padrao.counts.log2fc_undefined, (
        f"o parâmetro não mudou nada: {com_parametro.counts.log2fc_undefined} indefinidos com ele "
        f"e {padrao.counts.log2fc_undefined} sem"
    )
    assert com_parametro.counts.log2fc_undefined == 0
    # O padrão é a semântica de hoje, e ela não se move.
    assert np.array_equal(
        padrao.fold_change, univariate_test(processada, ("A", "B")).fold_change, equal_nan=True
    )


def test_as_duas_PROCEDENCIAS_viajam_e_dizem_quando_os_eixos_diferem() -> None:  # noqa: N802
    """R5: cada eixo declara a matriz sobre a qual foi calculado, e a figura lê a diferença."""
    pre, processada = _par_de_matrizes()

    mesmo = univariate_test(processada, ("A", "B"))
    assert mesmo.pvalue_provenance == mesmo.fold_change_provenance
    assert mesmo.fold_change_from_other_matrix is False

    cruzado = univariate_test(processada, ("A", "B"), fold_change_matrix=pre)
    assert cruzado.pvalue_provenance == processada.provenance
    assert cruzado.fold_change_provenance == pre.provenance
    assert cruzado.fold_change_from_other_matrix is True


def test_a_matriz_do_fold_change_com_OUTRAS_FEATURES_e_recusada() -> None:  # noqa: N802
    """A fronteira: índice diferente poria o `p` de uma feature ao lado do FC de outra."""
    _, processada = _par_de_matrizes()
    valores = np.asarray(processada.values)
    outra = _matriz(valores[:, :4], [a.group for a in processada.samples])

    with pytest.raises(ValueError, match="mesma ordem"):
        univariate_test(processada, ("A", "B"), fold_change_matrix=outra)


def test_a_matriz_do_fold_change_com_OUTRAS_AMOSTRAS_e_recusada() -> None:  # noqa: N802
    """A segunda metade da fronteira: FC sobre uma população e `p` sobre outra."""
    _, processada = _par_de_matrizes()
    valores = np.asarray(processada.values)
    trocada = _matriz(valores, ["A"] * 5 + ["B"] * 7)

    with pytest.raises(ValueError, match="outras amostras"):
        univariate_test(processada, ("A", "B"), fold_change_matrix=trocada)


# ---------------------------------------------------------------------------
# R6 — o que fica indefinido é CONTADO, nos dois eixos
# ---------------------------------------------------------------------------


def test_pvalue_ZERO_exato_e_contado_e_ENTRA_no_ajuste_de_BH() -> None:  # noqa: N802
    """Achado 1: constante dentro de cada grupo e diferente entre eles dá `p = 0,0` EXACTO.

    Welch devolve `statistic = ±inf` e `pvalue = 0.0`, e `np.isfinite(0.0)` é verdadeiro — então
    a feature **entra** no `n` de Benjamini-Hochberg, e o que ela não sobrevive é a `-log10`, que
    a manda para `+inf`. As duas coisas são verdadeiras ao mesmo tempo, e nenhum filtro de
    finitude as separa: é por isso que `pvalue_zero` é uma contagem própria e não parte de
    `pvalue_undefined`.

    **E ela é alcançável em produção, conferido no código do filtro e não suposto.**
    `filter_near_zero_variance` não mede variância: replica `caret::nearZeroVar`, e descarta a
    coluna só quando as DUAS condições valem — razão entre o valor mais comum e o segundo acima
    de `95/5`, **e** menos de 10% de valores distintos. Uma feature com um valor por grupo e
    grupos equilibrados tem razão `1,0`, falha a primeira condição, e **sobrevive** ao filtro.
    """
    valores = np.ones((10, 3))
    valores[:, 0] = np.concatenate([np.full(5, 1.0), np.full(5, 2.0)])  # constante em cada grupo
    valores[:, 1] = np.arange(10, dtype=np.float64)
    valores[:, 2] = np.arange(10, dtype=np.float64)[::-1]
    matriz = _matriz(valores, ["A"] * 5 + ["B"] * 5)

    resultado = univariate_test(matriz, ("A", "B"), parametric=True)
    assert resultado.pvalue[0] == 0.0, (
        "o zero é EXACTO; se não for, este achado precisa de remedida"
    )
    assert np.isinf(-np.log10(resultado.pvalue[0]))
    assert resultado.counts.pvalue_zero == 1
    assert resultado.counts.pvalue_undefined == 0, "zero é finito: não é indefinido"
    # A prova de que ele conta no denominador do BH: `q = p * n / i` com `n` a incluir esta.
    assert np.isfinite(resultado.qvalue_fdr[0]) and resultado.qvalue_fdr[0] == 0.0


@pytest.mark.parametrize("parametrico", [True, False])
def test_pvalue_NAO_FINITO_e_alcancavel_nas_duas_familias_e_e_contado(  # noqa: N802
    parametrico: bool,
) -> None:
    """Achado 2: feature sem variância nos DOIS grupos dá `p = nan` — em Welch e em Mann-Whitney.

    Até 11/09/2026 ela era descartada em silêncio por `q[np.isfinite(q)]` em `rotas.py`. Agora é
    contada nos dois eixos, e a contagem de `q` é **medida** e não derivada da de `p`.
    """
    valores = np.tile(np.arange(1.0, 4.0), (8, 1))
    valores[:, 1] = 5.0  # constante em toda a matriz: sem variância em nenhum dos dois grupos
    matriz = _matriz(valores, ["A"] * 4 + ["B"] * 4)

    resultado = univariate_test(matriz, ("A", "B"), parametric=parametrico)
    assert not np.isfinite(resultado.pvalue[1])
    assert resultado.counts.pvalue_undefined >= 1
    assert resultado.counts.qvalue_undefined == resultado.counts.pvalue_undefined, (
        "hoje as duas coincidem porque `benjamini_hochberg` devolve `nan` exactamente onde o `p` "
        "não é finito. Se divergirem, uma das duas mudou de política e a figura tem de saber"
    )


def test_denominador_ZERO_continua_indefinido_e_agora_e_CONTADO() -> None:  # noqa: N802
    """Borda: a política de `_fold_change` não muda, e o que ela exclui passa a ter número."""
    valores = np.ones((8, 3))
    valores[4:, 1] = 0.0
    resultado = univariate_test(_matriz(valores, ["A"] * 4 + ["B"] * 4), ("A", "B"))

    assert np.isnan(resultado.fold_change[1])
    assert resultado.counts.log2fc_undefined >= 1
    assert resultado.counts.n_features == 3


# ---------------------------------------------------------------------------
# Achado 4 — a decisão de normalidade é UMA, e a figura tem de poder dizer qual foi
# ---------------------------------------------------------------------------


def test_a_decisao_de_normalidade_declara_a_ROTA_e_a_EVIDENCIA() -> None:  # noqa: N802
    """Três rotas e não duas — o `False` cobria dois fatos.

    `forced` e `shapiro_unavailable` levam os dois a Mann-Whitney, e são fatos diferentes: no
    segundo **nenhum teste de normalidade correu**, e uma figura que escrevesse *"decidido por
    Shapiro-Wilk"* ali estaria a inventar procedência.
    """
    matriz = _com_efeito(8, 5, 2, 2.0, 41)

    forcado = univariate_test(matriz, ("A", "B"), parametric=False)
    assert forcado.normality.route == "forced"
    assert forcado.normality.shapiro_pvalue is None

    automatico = univariate_test(matriz, ("A", "B"))
    assert automatico.normality.route == "shapiro"
    assert automatico.normality.shapiro_pvalue is not None
    assert automatico.normality.n_residuals == 16 * 5, (
        "a população de Shapiro é os resíduos de TODAS as features concatenados — 16 amostras x 5 "
        "features. É o escopo que a figura tem de declarar, e ele não é por feature"
    )
    # A rota concorda com o desfecho: acima de alfa é paramétrico.
    assert automatico.parametric is (automatico.normality.shapiro_pvalue > NORMALITY_ALPHA)


def test_a_rota_shapiro_unavailable_existe_e_e_o_terceiro_desfecho() -> None:
    """O desfecho que o `bool` apagava, e ele é alcançado sem forçar nada.

    A mediana de `[1, nan]` é `nan`, então **todos** os resíduos saem não finitos, `achatados`
    fica vazio e Shapiro não tem o que medir. `parametric=None`, e mesmo assim nenhum teste de
    normalidade correu: é exactamente o caso em que a versão anterior devolvia `False` e a figura
    escreveria *"decidido por teste de normalidade"*.
    """
    valores = np.array([[1.0], [np.nan], [3.0], [np.nan]])
    matriz = _matriz(valores, ["A", "A", "B", "B"])

    resultado = univariate_test(matriz, ("A", "B"))
    assert resultado.normality.route == "shapiro_unavailable"
    assert resultado.normality.shapiro_pvalue is None
    assert resultado.parametric is False, "sem Shapiro a escolha é a conservadora"


# ---------------------------------------------------------------------------
# A amostra do Shapiro cobre os dois grupos, e o resíduo é em relação à média
# ---------------------------------------------------------------------------


def _residuos(a: np.ndarray, b: np.ndarray, centro: Callable[..., np.ndarray]) -> np.ndarray:
    """Os resíduos dentro de cada grupo, empilhados como `_decide_normalidade` os empilha."""
    return np.concatenate([x - centro(x, axis=0, keepdims=True) for x in (a, b)], axis=0)


def test_a_amostra_do_shapiro_toca_TODAS_as_linhas() -> None:  # noqa: N802
    """O vetor chega achatado linha a linha, e a amostra de 5.000 tem de alcançar cada linha.

    Cada valor aqui é o número da linha de onde veio, então o conjunto da amostra diz quais linhas
    ela alcançou. Os 5.000 primeiros, que era o corte antigo, alcançam só as 13 primeiras de 31.
    """
    linhas, colunas = 31, 400
    valores = np.repeat(np.arange(linhas, dtype=np.float64), colunas)

    assert set(valores[:MAX_PARA_SHAPIRO].tolist()) == set(range(13)), (
        "âncora: o corte nos primeiros parava na linha 12 — sem isto o teste não separa nada"
    )
    amostra = _amostra_espalhada(valores, MAX_PARA_SHAPIRO)
    assert amostra.size == MAX_PARA_SHAPIRO
    assert set(amostra.tolist()) == set(range(linhas))

    curto = np.arange(10, dtype=np.float64)
    assert np.array_equal(_amostra_espalhada(curto, MAX_PARA_SHAPIRO), curto), (
        "abaixo do teto a amostra é o vetor inteiro, sem perda"
    )


def test_o_SEGUNDO_grupo_entra_na_decisao_de_normalidade() -> None:  # noqa: N802
    """Par causal do defeito: grupo A normal, grupo B com caudas pesadas.

    Com 15 amostras e 400 features, o grupo A sozinho tem 6.000 resíduos, e os 5.000 primeiros são
    todos dele. A âncora confere que, sobre eles, o Shapiro-Wilk não rejeita — a planta cai sobre
    dado em que as duas leituras divergem. A decisão de verdade tem de ver o grupo B.

    **Com a implementação anterior INTEIRA este teste passa, e não por mérito dela.** Ela também
    centrava pela mediana, e isso fazia o teste rejeitar qualquer dado — o defeito do teste
    seguinte. O da amostragem só aparece isolado: com a média no centro e os 5.000 primeiros, este
    teste reprova na asserção do `shapiro_pvalue`, conferido em 17/10/2026.
    """
    gerador = np.random.default_rng(2)
    a = gerador.normal(10.0, 1.0, size=(15, 400))
    b = 10.0 + gerador.standard_t(2, size=(15, 400))

    residuos = _residuos(a, b, np.mean)
    primeiros = residuos[np.isfinite(residuos)].ravel()[:MAX_PARA_SHAPIRO]
    assert a.size > MAX_PARA_SHAPIRO, "âncora: o grupo A sozinho enche a amostra antiga"
    assert shapiro(primeiros).pvalue > NORMALITY_ALPHA, (
        "âncora: os primeiros 5.000, só do grupo A, não rejeitam — sem isto a planta não separa"
    )

    resultado = univariate_test(_matriz(np.vstack([a, b]), ["A"] * 15 + ["B"] * 15), ("A", "B"))
    assert resultado.normality.route == "shapiro"
    assert resultado.normality.shapiro_pvalue is not None
    assert resultado.normality.shapiro_pvalue <= NORMALITY_ALPHA
    assert resultado.parametric is False


def test_dado_NORMAL_nao_e_rejeitado_por_causa_do_centro_do_residuo() -> None:  # noqa: N802
    """O segundo defeito: centrar pela mediana rejeitava dado exactamente normal.

    Num grupo de 13 a mediana é uma das observações, e cada feature deixa um resíduo zero; com
    milhares de resíduos o Shapiro-Wilk vê o acúmulo. A âncora mede isso sobre o mesmo dado; a
    decisão, que centra pela média, tem de aceitar a normalidade.
    """
    gerador = np.random.default_rng(1)
    valores = gerador.normal(10.0, 1.0, size=(31, 400))
    a, b = valores[:18], valores[18:]

    pela_mediana = _amostra_espalhada(_residuos(a, b, np.median).ravel(), MAX_PARA_SHAPIRO)
    assert shapiro(pela_mediana).pvalue <= NORMALITY_ALPHA, (
        "âncora: centrar pela mediana rejeita este dado normal — sem isto o teste não separa nada"
    )

    resultado = univariate_test(_matriz(valores, ["A"] * 18 + ["B"] * 13), ("A", "B"))
    assert resultado.normality.shapiro_pvalue is not None
    assert resultado.normality.shapiro_pvalue > NORMALITY_ALPHA
    assert resultado.parametric is True


# ---------------------------------------------------------------------------
# A população é a MÁSCARA e não o rótulo — a irmã de `plsda_multiclasse`
# ---------------------------------------------------------------------------

# O desenho é o do estudo real reduzido — sete controles, e o `ATLL` como o grupo que recebe as
# linhas `validate`. A matriz é SINTÉTICA e dita como tal: nenhum dos quatro arquivos do
# laboratório tem amostra `validate` (medido em 17/09/2026, 0 de 68 nos quatro modos).
N_QC_DO_ESTUDO = 7


def _com_validate(tipo_das_duas: str) -> RefinoMatrix:
    """8 `ATLL` e 8 `NoHTLV` de estudo, DUAS linhas de rótulo `ATLL`, e 7 controles de rótulo `QC`.

    `tipo_das_duas` é a **única** coisa que o par causal move: valores da mesma semente, mesmos
    rótulos, mesma ordem de injeção.

    **Sete controles, e o número não é decoração:** com menos de `MIN_AMOSTRAS_POR_GRUPO` a recusa
    de `groups=("QC", ...)` viria da guarda de **tamanho**, e o teste não distinguiria as duas
    guardas nem reprovaria contra a implementação anterior, que aceitava o pedido.
    """
    rotulos = ["ATLL"] * 8 + ["NoHTLV"] * 8 + ["ATLL"] * 2 + ["QC"] * N_QC_DO_ESTUDO
    tipos = ["sample"] * 16 + [tipo_das_duas] * 2 + ["qc"] * N_QC_DO_ESTUDO
    gerador = np.random.default_rng([1212, 7])
    return _matriz(gerador.normal(size=(len(rotulos), 12)), rotulos, tipos)


def test_a_amostra_validate_de_rotulo_clinico_NAO_entra_no_bloco() -> None:  # noqa: N802
    """**`sample_mask` não é o complemento de `qc_mask`**, e é a seleção que tem de o saber.

    Uma amostra `validate` existe para medir a correção **fora** da amostra de treino, e o rótulo
    clínico dela é o do paciente de onde veio — então ele coincide com um grupo de estudo por
    construção, e não por acidente. Deixá-la entrar dá o `p` certo sobre a população errada.

    **A planta cai sobre dado em que as duas seleções DIFEREM, e isso é afirmado ANTES**:
    o bloco `ATLL` tem **10** linhas por rótulo e **8** por máscara. Com os dois iguais o teste
    passaria por coincidência — que é o caso dos quatro arquivos reais, onde não há `validate`
    nenhuma.

    **O par causal é o TIPO e nada mais:** as mesmas duas linhas, os mesmos valores e o mesmo
    rótulo, tipadas `validate` numa ponta e `sample` na outra.
    """
    matriz = _com_validate("validate")
    por_linha = np.array(matriz.groups)
    estudo = np.asarray(matriz.sample_mask)
    por_rotulo = int((por_linha == "ATLL").sum())
    por_mascara = int(((por_linha == "ATLL") & estudo).sum())
    assert (por_rotulo, por_mascara) == (10, 8), (
        f"a planta não separa: o bloco `ATLL` tem {por_rotulo} linhas por rótulo e {por_mascara} "
        "por máscara. Com os dois iguais, a asserção abaixo passaria sobre qualquer das duas"
    )

    resultado = univariate_test(matriz, ("ATLL", "NoHTLV"))
    como_estudo = univariate_test(_com_validate("sample"), ("ATLL", "NoHTLV"))

    assert not np.allclose(resultado.pvalue, como_estudo.pvalue, equal_nan=True), (
        "tipar as duas linhas como `validate` ou como `sample` deu os MESMOS p-valores: "
        "`sample_type` não está a ser lido de todo"
    )


def test_o_rotulo_dos_QC_pedido_como_GRUPO_e_recusado() -> None:  # noqa: N802
    """**O QC é a régua, e pedi-lo como grupo tem de ser recusado.**

    A guarda de pertinência lia os rótulos de **todas** as linhas, então `("QC", "ATLL")` era
    aceito e devolvia a tabela por feature para um contraste entre a mistura de controle e um
    diagnóstico. Como o QC é sempre a mesma mistura, toda variação nele é técnica, e um `q` baixo
    ali não é achado biológico nenhum.

    **Medido no `pos_hilic` REAL em 17/09/2026, antes do conserto:** aceito, com blocos de 7
    controles contra 13 `ATLL`, e **7** das 6.035 features a `q < 0,05` de 5.994 definidas.

    **A âncora vem primeiro:** um par de grupos de estudo TEM de ser aceito sobre o mesmo fixture,
    senão o teste passaria por a função recusar tudo.

    **A mensagem é afirmada e não só o tipo.** Sob a máscara o grupo `"QC"` fica com
    zero linhas, e sem a guarda de pertinência a recusa viria da guarda de **tamanho** — um
    `pytest.raises(ValueError)` nu passaria pelo motivo errado.
    """
    matriz = _com_validate("validate")

    aceito = univariate_test(matriz, ("ATLL", "NoHTLV"))
    assert aceito.groups == ("ATLL", "NoHTLV"), "âncora: um par de grupos de estudo é legítimo"

    n_controles = int(np.asarray(matriz.qc_mask).sum())
    assert n_controles >= 2, (
        f"há {n_controles} controles: com menos de dois a recusa abaixo viria do TAMANHO, e o "
        "teste passaria pelo motivo errado"
    )

    with pytest.raises(ValueError) as erro:
        univariate_test(matriz, ("QC", "ATLL"))
    texto = str(erro.value)
    assert "'QC'" in texto and "não está entre as amostras de estudo" in texto, texto
    assert "exige ao menos" not in texto, (
        f"a recusa veio da guarda de TAMANHO e não da de população: {texto}"
    )
    assert "não existe nesta matriz" not in texto, (
        f"a mensagem afirma que o rótulo não existe, e ele existe — é o rótulo dos controles "
        f"desta matriz. Quem a lê vai procurar um erro de escrita que não há: {texto}"
    )


def test_a_matriz_do_fold_change_com_OUTRO_SAMPLE_TYPE_e_recusada() -> None:  # noqa: N802
    """Na fronteira, o tipo decide a população.

    Desde que `_blocos_dos_grupos` seleciona por `sample_mask`, duas matrizes com os mesmos nomes
    e os mesmos rótulos, e uma delas com linhas tipadas `validate`, produzem o `p` sobre uma
    população e o fold change sobre outra — exactamente o desfecho que a segunda linha da tabela
    de `_confere_matriz_do_fold_change` nomeia. A guarda comparava só `(nome, grupo)` e deixava-o
    passar.

    **A planta separa**: as duas matrizes têm as mesmas 18 linhas e os mesmos rótulos, e a
    asserção afirma-o antes de esperar a recusa — se diferissem em nome ou em grupo, a recusa
    viria da metade antiga da guarda e este teste não mediria nada.
    """
    do_teste = _com_validate("sample")
    outra = _com_validate("validate")

    assert [a.name for a in do_teste.samples] == [a.name for a in outra.samples]
    assert [a.group for a in do_teste.samples] == [a.group for a in outra.samples]
    assert do_teste.feature_names == outra.feature_names
    assert [a.sample_type for a in do_teste.samples] != [a.sample_type for a in outra.samples], (
        "as duas matrizes têm o mesmo `sample_type`: a planta não separa"
    )

    with pytest.raises(ValueError, match="sample_type"):
        univariate_test(do_teste, ("ATLL", "NoHTLV"), fold_change_matrix=outra)

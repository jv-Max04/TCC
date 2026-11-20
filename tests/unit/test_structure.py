"""Unidades de `metrics/structure.py`.

O que não precisa de dado de referência: as guardas, o vocabulário, as
propriedades que decorrem do desenho, e os casos de borda que os quatro arquivos
reais não alcançam. É o mesmo problema, aplicado antes de haver divergência — golden test só
verifica a função no domínio que o oráculo visitou, e nenhuma matriz de referência
tem amostra `validate`, feature constante ou uma única amostra biológica.
"""

import numpy as np
import pytest

from refino.domain.labels import InvalidLabelError, PreTransform
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.metrics.structure import (
    PRETRANSFORM_IMPLEMENTATIONS,
    _pca_scores,
    pc1_injection_correlation,
    qc_dispersion_ratio,
    qc_pc1_trajectory,
    structural_metrics,
)

PRETRANSFORMS: tuple[PreTransform, ...] = ("log10p1", "none")


def _matrix(values: np.ndarray, tipos: tuple[str, ...]) -> RefinoMatrix:
    samples = tuple(
        SampleMetadata(
            name=f"QC{i + 1}" if tipo == "qc" else f"amostra{i + 1}",
            injection_order=i + 1,
            sample_type=tipo,  # type: ignore[arg-type]
            group="QC" if tipo == "qc" else "NoHTLV",
        )
        for i, tipo in enumerate(tipos)
    )
    return RefinoMatrix(
        values=np.asarray(values, dtype=np.float64),
        feature_names=tuple(f"f{j + 1}" for j in range(values.shape[1])),
        samples=samples,
        mode="teste",
    )


def _drifting(n_samples: int = 12, n_features: int = 6, seed: int = 7) -> RefinoMatrix:
    """Matriz sintética com deriva monotônica dominante e QCs intercalados.

    Sintética e não de referência de propósito: aqui a verdade é **conhecida** por
    construção, e o que se testa são propriedades do desenho. A regra 1 do
    README proíbe o inverso — testar contagem observada sobre factory.

    O modelo é `nível_da_feature x deriva x ruído`, e a estrutura importa: a
    primeira versão sorteava cada célula independentemente entre 1e4 e 1e6 e
    multiplicava pela deriva, o que deixava a variação aleatória com duas ordens de
    grandeza contra um fator de três da deriva. PC1 capturava o ruído e a
    correlação com a ordem de injeção era 0,52 — invariância de um número que não
    mede o que o nome diz. Cada feature tem **nível próprio e estável**, como uma
    feature real tem, e a deriva é o que varia ao longo das injeções.
    """
    rng = np.random.default_rng(seed)
    level = rng.uniform(1e4, 1e6, size=(1, n_features))
    drift = np.linspace(1.0, 3.0, n_samples)[:, None]
    noise = 1.0 + 0.05 * rng.standard_normal(size=(n_samples, n_features))
    tipos = tuple("qc" if i % 4 == 0 else "sample" for i in range(n_samples))
    return _matrix(level * drift * noise, tipos)


# ---------------------------------------------------------------------------
# Vocabulário e a guarda de fronteira
# ---------------------------------------------------------------------------


def test_registro_de_pretransform_cobre_exatamente_o_literal() -> None:
    """Paridade local; a global está em `test_labels.py`."""
    from typing import get_args

    assert set(PRETRANSFORM_IMPLEMENTATIONS) == set(get_args(PreTransform))


@pytest.mark.parametrize(
    "funcao", [qc_dispersion_ratio, pc1_injection_correlation, qc_pc1_trajectory]
)
def test_pretransform_invalido_levanta_em_vez_de_cair_em_outro_ramo(funcao: object) -> None:
    """O modo de falha insidioso é o que **não** levanta.

    Um `pretransform` desconhecido caindo num `else` mediria a matriz sem
    transformar e devolveria número plausível — a tabela 4.4 daria 0,114 em vez de
    0,682 no `neg_hilic`/`batch_loess`, e nada acusaria.
    """
    matrix = _drifting()
    with pytest.raises(InvalidLabelError, match="PreTransform"):
        funcao(matrix, pretransform="log10")  # type: ignore[operator, arg-type]


@pytest.mark.parametrize(
    "funcao", [qc_dispersion_ratio, pc1_injection_correlation, qc_pc1_trajectory]
)
def test_pretransform_e_obrigatorio_e_nomeado(funcao: object) -> None:
    """Sem valor padrão e keyword-only: é o remédio, cobrado pelo runtime.

    O `mypy` cobra estaticamente; este teste cobra em runtime, porque quem chama
    de um `getattr` ou de uma tabela de despacho não passa pelo `mypy`.
    """
    matrix = _drifting()
    with pytest.raises(TypeError):
        funcao(matrix)  # type: ignore[operator, call-arg]
    with pytest.raises(TypeError):
        funcao(matrix, "log10p1")  # type: ignore[operator, misc]


def test_structural_metrics_tambem_exige_pretransform() -> None:
    with pytest.raises(TypeError):
        structural_metrics(_drifting())  # type: ignore[call-arg]
    with pytest.raises(InvalidLabelError):
        structural_metrics(_drifting(), pretransform="pareto")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# `sample_mask` não é o complemento de `qc_mask`
# ---------------------------------------------------------------------------


def test_validate_fica_fora_do_denominador_e_o_resultado_difere() -> None:
    """`sample_mask` e o complemento de `qc_mask` são coisas diferentes.

    Nenhum dos quatro arquivos de referência tem amostra `validate`, então **nenhum
    golden test alcança este caminho** — é o cego estrutural. O teste
    constrói o caso e afirma as duas coisas: que `validate` não entra, e que
    incluí-la daria número diferente. A segunda metade é a que tem dente; sem ela,
    uma implementação que usasse o complemento passaria.
    """
    rng = np.random.default_rng(3)
    n_features = 5
    tipos = ("qc", "sample", "sample", "validate", "validate", "qc", "sample", "qc")
    valores = rng.uniform(1e4, 1e6, size=(len(tipos), n_features))
    # As `validate` ficam longe, para que incluí-las mude a dispersão de verdade.
    valores[3] *= 40.0
    valores[4] *= 60.0
    matrix = _matrix(valores, tipos)

    obtido = qc_dispersion_ratio(matrix, pretransform="none")

    # O mesmo cálculo com o complemento de `qc_mask` no denominador.
    scores = _pca_scores(matrix, pretransform="none", n_components=2)

    def dispersao(pontos: np.ndarray) -> float:
        centroide = pontos.mean(axis=0)
        return float(np.mean(np.linalg.norm(pontos - centroide, axis=1)))

    com_complemento = dispersao(scores[matrix.qc_mask]) / dispersao(scores[~matrix.qc_mask])

    assert obtido == pytest.approx(
        dispersao(scores[matrix.qc_mask]) / dispersao(scores[matrix.sample_mask]), rel=1e-12
    )
    assert obtido != pytest.approx(com_complemento, rel=1e-6), (
        "usar o complemento de qc_mask daria o mesmo número neste fixture, então ele "
        "não distingue as duas implementações e o teste seria decorativo"
    )


# ---------------------------------------------------------------------------
# Passo 3: descarte de coluna, pelos dois motivos
# ---------------------------------------------------------------------------


def test_feature_constante_e_descartada_e_contada() -> None:
    """Coluna de desvio zero não levanta, é descartada, e a contagem aparece."""
    matrix = _drifting(n_features=5)
    valores = matrix.values.copy()
    valores[:, 2] = 7.0
    com_constante = matrix.__class__(
        values=valores,
        feature_names=matrix.feature_names,
        samples=matrix.samples,
        mode=matrix.mode,
    )
    metricas = structural_metrics(com_constante, pretransform="none")
    assert metricas.n_features_dropped_zero_variance == 1
    assert metricas.n_features_dropped_non_finite == 0
    assert metricas.n_features_used == 4


def test_feature_constante_em_valor_nao_nulo_tambem_e_descartada() -> None:
    """O caso que `sd > 0.0` **não** pegava, e que os arquivos reais não têm.

    O desvio padrão de uma coluna constante não é exatamente zero em ponto
    flutuante: doze cópias de `log10(6)` dão desvio 1,11e-16. Sob o critério antigo
    essa coluna seria retida, o passo 4 a dividiria por 1,11e-16, e uma feature sem
    informação nenhuma dominaria o SVD.

    Nos quatro arquivos de referência as colunas constantes são constantes **em
    zero**, onde a aritmética é exata e os dois critérios coincidem — é o mesmo problema outra
    vez, guarda que funciona por propriedade do dado e não por desenho.
    """
    matrix = _drifting(n_features=5)
    valores = matrix.values.copy()
    valores[:, 2] = 5.0  # constante num valor NÃO nulo
    com_constante = matrix.__class__(
        values=valores,
        feature_names=matrix.feature_names,
        samples=matrix.samples,
        mode=matrix.mode,
    )
    metricas = structural_metrics(com_constante, pretransform="log10p1")
    assert metricas.n_features_dropped_zero_variance == 1
    assert metricas.n_features_used == 4


def test_o_criterio_antigo_de_desvio_positivo_reprovaria_nesse_caso() -> None:
    """A prova de que a troca de critério não é preciosismo.

    Mede o desvio padrão da coluna constante e afirma que ele é **positivo** — isto
    é, que `sd > 0.0` de fato a reteria. Se um dia o `numpy` passar a devolver zero
    exato ali, este teste reprova e a justificativa precisa ser relida em vez
    de continuar sendo citada.
    """
    matrix = _drifting(n_features=5)
    valores = matrix.values.copy()
    valores[:, 2] = 5.0
    transformada = np.log10(valores + 1.0)
    sd_da_constante = float(transformada[:, 2].std())
    assert sd_da_constante > 0.0, (
        "o desvio da coluna constante deu zero exato; o critério antigo funcionaria "
        "e o argumento a favor do critério novo precisa ser reexaminado"
    )
    assert float(np.ptp(transformada[:, 2])) == 0.0, (
        "o critério novo depende de ptp ser exatamente zero para coluna constante"
    )


@pytest.mark.parametrize("ruim", [np.inf, -np.inf, np.nan])
def test_feature_com_valor_nao_finito_e_descartada_pelo_motivo_certo(ruim: float) -> None:
    """Os dois motivos de descarte contam separado.

    O caminho existe porque as matrizes golden do `batch_loess` têm `Inf` **e**
    `NaN` — de 2,7% a 15,2% das features. Antes esta coluna era descartada
    por acidente de `nan > 0.0` ser `False`, e contada junto com variância zero.
    """
    matrix = _drifting(n_features=5)
    valores = matrix.values.copy()
    valores[3, 1] = ruim
    sujo = matrix.__class__(
        values=valores,
        feature_names=matrix.feature_names,
        samples=matrix.samples,
        mode=matrix.mode,
    )
    metricas = structural_metrics(sujo, pretransform="none")
    assert metricas.n_features_dropped_non_finite == 1
    assert metricas.n_features_dropped_zero_variance == 0
    assert metricas.n_features_used == 4


def test_matriz_sem_nenhuma_coluna_utilizavel_levanta_nomeando_os_dois_motivos() -> None:
    """Zero colunas após o passo 3 — o caso que a correção vai reencontrar no SERRF.

    Levanta em vez de devolver `nan` porque não é métrica indefinida: é matriz sem
    estrutura, e seguir daria SVD de array vazio três camadas acima.
    """
    matrix = _matrix(np.full((6, 3), 5.0), ("qc", "sample", "sample", "qc", "sample", "qc"))
    with pytest.raises(ValueError, match="nenhuma coluna utilizável"):
        qc_dispersion_ratio(matrix, pretransform="none")


# ---------------------------------------------------------------------------
# Propriedades do desenho
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [1e-6, 1e-3, 0.5, 2.0, 1e3, 1e6])
def test_pretransform_none_invariante_exato_a_fator_positivo(factor: float) -> None:
    """Igualdade exata: o autoescalonamento cancela o fator antes do SVD."""
    matrix = _drifting()
    base = pc1_injection_correlation(matrix, pretransform="none")
    escalada = matrix.__class__(
        values=matrix.values * factor,
        feature_names=matrix.feature_names,
        samples=matrix.samples,
        mode=matrix.mode,
    )
    assert pc1_injection_correlation(escalada, pretransform="none") == base


def test_dispersao_e_invariante_a_escala_mas_nao_bit_a_bit() -> None:
    """A mesma propriedade na outra métrica, e a diferença entre as duas é real.

    A invariância é do passo 4 e vale para as duas métricas em aritmética exata.
    Em ponto flutuante, só a correlação é **bit a bit** invariante, e o motivo é a
    correlação ser de **postos**: `(cx - mean(cx))/sd(cx)` difere de
    `(x - mean(x))/sd(x)` no último bit, os postos absorvem essa diferença, e uma
    razão de distâncias euclidianas não. Medido: erro relativo de 6,5e-16 — um
    epsilon de máquina.

    Afirmar igualdade exata aqui reprovaria por motivo nenhum; afirmar tolerância
    solta na correlação perderia a informação de que ali a igualdade **é** exata.
    Por isso os dois testes são separados e dizem coisas diferentes.
    """
    matrix = _drifting()
    base = qc_dispersion_ratio(matrix, pretransform="none")
    escalada = matrix.__class__(
        values=matrix.values * 1e5,
        feature_names=matrix.feature_names,
        samples=matrix.samples,
        mode=matrix.mode,
    )
    obtido = qc_dispersion_ratio(escalada, pretransform="none")
    assert obtido == pytest.approx(base, rel=1e-12)


@pytest.mark.parametrize("pretransform", PRETRANSFORMS)
def test_correlacao_e_sempre_nao_negativa(pretransform: PreTransform) -> None:
    """Instrumentação: pega o `abs` esquecido, com deriva decrescente também."""
    matrix = _drifting()
    decrescente = matrix.__class__(
        values=matrix.values[::-1].copy(),
        feature_names=matrix.feature_names,
        samples=matrix.samples,
        mode=matrix.mode,
    )
    assert pc1_injection_correlation(matrix, pretransform=pretransform) >= 0.0
    assert pc1_injection_correlation(decrescente, pretransform=pretransform) >= 0.0


@pytest.mark.parametrize("pretransform", PRETRANSFORMS)
def test_deriva_monotonica_forte_produz_correlacao_alta(pretransform: PreTransform) -> None:
    """Verdade conhecida por construção: a matriz sintética TEM deriva.

    Sem esta asserção, todos os testes de invariância acima passariam sobre uma
    matriz cuja correlação fosse zero — invariância de um número morto.
    """
    assert pc1_injection_correlation(_drifting(), pretransform=pretransform) > 0.8


@pytest.mark.parametrize("pretransform", PRETRANSFORMS)
def test_trajetoria_tem_um_score_por_qc_na_ordem_de_injecao(
    pretransform: PreTransform,
) -> None:
    matrix = _drifting()
    trajetoria = qc_pc1_trajectory(matrix, pretransform=pretransform)
    assert trajetoria.shape == (int(matrix.qc_mask.sum()),)
    scores = _pca_scores(matrix, pretransform=pretransform, n_components=1)
    np.testing.assert_array_equal(trajetoria, scores[matrix.qc_mask, 0])
    # A ordem é temporal porque `RefinoMatrix` ordena por `injection_order`.
    ordens = matrix.injection_orders[matrix.qc_mask]
    assert list(ordens) == sorted(ordens)


def test_trajetoria_nao_subtrai_o_primeiro_elemento() -> None:
    """O contrato diz explicitamente que a subtração é da apresentação.

    Uma implementação que subtraísse daria primeiro elemento sempre zero, o que é
    fácil de escrever por engano ao reproduzir a tabela 4.1.
    """
    trajetoria = qc_pc1_trajectory(_drifting(), pretransform="none")
    assert trajetoria[0] != 0.0


# ---------------------------------------------------------------------------
# `n_components`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_dispersao_opera_em_qualquer_numero_de_componentes(k: int) -> None:
    """A definição vale para qualquer `k`, e o padrão de 2 não é caso especial."""
    valor = qc_dispersion_ratio(_drifting(n_features=8), pretransform="none", n_components=k)
    assert np.isfinite(valor)
    assert valor > 0.0


def test_pc1_nao_depende_de_n_components() -> None:
    """A suposição que permite `structural_metrics` calcular o SVD uma vez.

    Os componentes principais não dependem de quantos se pede — PC1 com `k=1` é o
    mesmo vetor que a primeira coluna com `k=5`. Se dependesse, o caminho agregado
    daria número diferente do caminho isolado, e o multiverso mediria outra coisa.
    """
    matrix = _drifting(n_features=8)
    um = _pca_scores(matrix, pretransform="none", n_components=1)[:, 0]
    cinco = _pca_scores(matrix, pretransform="none", n_components=5)[:, 0]
    np.testing.assert_allclose(um, cinco, rtol=1e-12)


def test_n_components_invalido_levanta() -> None:
    matrix = _drifting()
    with pytest.raises(ValueError, match="n_components deve ser >= 1"):
        qc_dispersion_ratio(matrix, pretransform="none", n_components=0)
    with pytest.raises(ValueError, match="só admite"):
        qc_dispersion_ratio(matrix, pretransform="none", n_components=999)


# ---------------------------------------------------------------------------
# Casos degenerados que os arquivos reais não alcançam
# ---------------------------------------------------------------------------


def test_um_unico_qc_levanta_nomeando_a_mascara() -> None:
    """Dispersão em torno de um centroide exige duas linhas.

    Com um QC a distância ao centroide é zero por construção, e a razão seria 0,0 —
    número que afirma "QCs perfeitamente agrupados" onde não há informação nenhuma.
    """
    matrix = _matrix(
        np.random.default_rng(1).uniform(1e4, 1e6, size=(4, 4)),
        ("qc", "sample", "sample", "sample"),
    )
    with pytest.raises(ValueError, match="qc_mask tem 1 linha"):
        qc_dispersion_ratio(matrix, pretransform="none")


def test_uma_unica_amostra_biologica_levanta_nomeando_a_mascara() -> None:
    matrix = _matrix(
        np.random.default_rng(2).uniform(1e4, 1e6, size=(4, 4)),
        ("qc", "qc", "qc", "sample"),
    )
    with pytest.raises(ValueError, match="sample_mask tem 1 linha"):
        qc_dispersion_ratio(matrix, pretransform="none")


def test_amostras_identicas_caem_de_um_dos_dois_lados_da_descontinuidade() -> None:
    """Amostras idênticas: `nan` **ou** razão enorme, e não cabe a nós escolher.

    **Duas correções empilhadas, e a segunda desfaz metade da primeira.**

    A versão original deste teste esperava `nan` e reprovou com 4,4e+31. A leitura
    foi que a expectativa estava errada: amostras idênticas na entrada não saem do
    SVD bit-idênticas, sobra resíduo, o denominador é minúsculo e positivo, e razão
    enorme descreve corretamente "os QCs se espalham muito mais que as amostras".
    Essa parte segue verdadeira.

    O que estava errado era afirmar o **lado**. O teste passou duas fases nesta
    máquina e reprovou na primeira execução do CI, em `assert np.isfinite(obtido)`.
    Medido aqui: a dispersão das amostras idênticas é **exatamente 0,0 em PC1** e
    ~1e-27 nos componentes seguintes. Qual resíduo o LAPACK deixa — e se deixa — não
    decorre do nosso desenho; decorre da implementação de álgebra linear do host. É
    a observação promovida a invariante na forma que mais engana, porque o número observado era
    **reprodutível localmente**: rodar mil vezes na mesma máquina dá 4,4e+31 mil vezes.

    O que **decorre do desenho** e é o que este teste agora afirma:

    1. a razão é `nan` (denominador exatamente zero — `unassessable`) **ou**
       enorme; nunca um finito pequeno, que significaria "QCs e amostras se
       espalham parecido" e seria falso;
    2. as duas portas de entrada concordam — `structural_metrics` e o ajudante não
       podem discordar, e isso é desenho, não observação;
    3. quando o valor é finito, o indicador acusa instabilidade. É a razão de
       ele existir: já se viu 1,2e+16 de amplitude aqui, e a etapa seguinte não
       classifica a partir do valor sozinho.

    O ramo de `nan` para denominador exatamente zero continua tendo teste direto no
    ajudante, onde o caso é construtível em vez de sorteado pelo LAPACK — ver o
    teste seguinte.
    """
    valores = np.tile(np.array([1e5, 2e5, 3e5, 4e5]), (6, 1))
    # QCs variam; as amostras são idênticas entre si.
    valores[0] *= 1.3
    valores[3] *= 0.7
    valores[5] *= 1.1
    matrix = _matrix(valores, ("qc", "sample", "sample", "qc", "sample", "qc"))
    obtido = qc_dispersion_ratio(matrix, pretransform="none")
    relatorio = structural_metrics(matrix, pretransform="none")

    assert np.isnan(obtido) or obtido > 1e6, (
        f"razão finita e pequena ({obtido}) com amostras idênticas: significaria que "
        f"QCs e amostras se espalham parecido, o que é falso por construção."
    )

    if np.isnan(obtido):
        assert np.isnan(relatorio.qc_dispersion_ratio), (
            "o ajudante devolveu nan e o relatório não — as duas portas discordaram"
        )
    else:
        assert relatorio.qc_dispersion_ratio == pytest.approx(obtido, rel=1e-12)
        assert relatorio.qc_dispersion_ratio_spread > 1e6, (
            f"valor finito de {obtido:.3e} sem instabilidade acusada "
            f"(amplitude {relatorio.qc_dispersion_ratio_spread:.3e}). A acusação existe para "
            f"que limiar não seja aplicado a valor mal-condicionado sem aviso."
        )


def test_denominador_exatamente_zero_da_nan_pelo_ajudante() -> None:
    """O ramo de indefinido, alcançado onde ele é construtível.

    Pela API pública o caso é inalcançável — o SVD sempre deixa resíduo —, e a
    escolha então seria `pragma: no cover` ou teste pelo ajudante. Manda-se tentar
    o teste antes do `pragma`, e aqui o teste existe: `_dispersion_from_scores`
    recebe scores construídos com as linhas de `sample_mask` bit-idênticas.

    `nan` e não `inf` porque o numerador também pode ser zero, e `0/0` não é
    "dispersão relativa infinita" — é ausência de informação.
    """
    from refino.metrics.structure import _dispersion_from_scores

    matrix = _matrix(
        np.ones((6, 2)) * np.arange(1, 3), ("qc", "sample", "sample", "qc", "sample", "qc")
    )
    scores = np.array([[1.0, 0.0], [5.0, 5.0], [5.0, 5.0], [-1.0, 0.0], [5.0, 5.0], [0.0, 1.0]])
    assert np.isnan(_dispersion_from_scores(scores, matrix))


def test_uma_unica_injecao_levanta_pelo_ajudante_da_correlacao() -> None:
    """A guarda de "menos de duas injeções", alcançada onde é alcançável.

    Pela API pública ela é inatingível, e a razão é encadeamento: uma matriz de uma
    linha tem desvio zero em toda coluna, então o passo 3 descarta tudo e
    `_pca_scores` levanta antes. A guarda fica porque `_abs_spearman_with_order`
    tem dois chamadores e nada garante que o próximo venha depois do passo 3 —
    e manda-se tentar o teste antes de pôr `pragma`, o que aqui deu certo.
    """
    from refino.metrics.structure import _abs_spearman_with_order

    with pytest.raises(ValueError, match="ao menos duas injeções"):
        _abs_spearman_with_order(np.array([1.0]), np.array([1]))


def test_pc1_constante_da_correlacao_indefinida_e_nao_zero() -> None:
    """`nan` e não `0.0`: sem variação não há posto.

    Zero afirmaria "medi e não há correlação", que é conclusão sobre o dado. `nan`
    diz "não há o que medir", que é o que de fato acontece — e a diferença importa
    para o `DefectProfile`, cujo limiar de `clean` compara
    `pc1_injection_correlation < 0,30`: um `nan` não passa no limiar e um zero passa.
    """
    from scipy.stats import ConstantInputWarning

    from refino.metrics.structure import _abs_spearman_with_order

    # O aviso do scipy é afirmado, não silenciado: ele descreve corretamente o que
    # aconteceu, e deixá-lo solto na saída da suíte treina o leitor a ignorar aviso.
    with pytest.warns(ConstantInputWarning):
        resultado = _abs_spearman_with_order(np.full(6, 3.0), np.arange(1, 7))
    assert np.isnan(resultado)


def test_structural_metrics_carrega_os_rotulos_de_como_mediu() -> None:
    """O rótulo viaja com o número: sem ele o multiverso compara grandezas diferentes."""
    for pretransform in PRETRANSFORMS:
        metricas = structural_metrics(_drifting(), pretransform=pretransform, n_components=3)
        assert metricas.pretransform == pretransform
        assert metricas.n_components == 3


def test_structural_metrics_e_imutavel() -> None:
    metricas = structural_metrics(_drifting(), pretransform="none")
    with pytest.raises(AttributeError):
        metricas.qc_dispersion_ratio = 0.0  # type: ignore[misc]


def test_as_metricas_nao_mutam_a_matriz_de_entrada() -> None:
    """Regra 7 do README: nenhuma operação altera a matriz de entrada."""
    matrix = _drifting()
    antes = matrix.values.copy()
    structural_metrics(matrix, pretransform="log10p1")
    qc_dispersion_ratio(matrix, pretransform="log10p1")
    pc1_injection_correlation(matrix, pretransform="log10p1")
    qc_pc1_trajectory(matrix, pretransform="log10p1")
    np.testing.assert_array_equal(matrix.values, antes)


def test_todas_as_colunas_nao_finitas_levanta_nomeando_o_motivo() -> None:
    """Zero colunas utilizáveis por não finito em **todas** — um dos ramos.

    É o terceiro dos ramos que a correção tem de cobrir no SERRF ("zero colunas após
    descartar NaN") e que nenhum arquivo de referência alcança: nas quatro matrizes
    de `batch_loess` o pior caso é 15,2% das features não finitas, longe de 100%.
    Aqui o caso é construído, e a mensagem nomeia as duas contagens para que quem a
    ler saiba qual dos dois motivos esvaziou a matriz.
    """
    valores = np.full((6, 3), np.nan)
    matrix = _matrix(valores, ("qc", "sample", "sample", "qc", "sample", "qc"))
    with pytest.raises(ValueError, match="nenhuma coluna utilizável"):
        structural_metrics(matrix, pretransform="none")
    with pytest.raises(ValueError, match="3 com desvio não finito"):
        qc_dispersion_ratio(matrix, pretransform="none")


# ---------------------------------------------------------------------------
# Estabilidade
# ---------------------------------------------------------------------------


def test_amplitude_e_praticamente_zero_em_matriz_bem_condicionada() -> None:
    """O indicador não pode acusar instabilidade onde não há.

    Medido nas doze células reais: pior amplitude de 1,33e-15 na dispersão e
    **exatamente zero** na correlação. Aqui a asserção é sobre a matriz sintética,
    que é bem-condicionada por construção.
    """
    metricas = structural_metrics(_drifting(), pretransform="log10p1")
    assert metricas.qc_dispersion_ratio_spread < 1e-9
    assert metricas.pc1_injection_correlation_spread < 1e-9
    assert metricas.is_assessable(max_spread=0.01)


def test_amplitude_acusa_o_empate_de_pc1_que_o_hypothesis_achou() -> None:
    """A metade que tem dente: sem ela o indicador seria decorativo.

    Este é o contraexemplo literal — matriz cuja única coluna utilizável tem
    seis valores iguais. A correlação vale 0,7572 e a amplitude sob perturbação de
    `eps x amplitude` é **0,8553**: o valor pode ir a praticamente qualquer lugar de
    `[0, 1]`, e um limiar em 0,30 sobre ele decidiria por ruído.
    """
    valores = np.array(
        [
            [148.0, 148.0, 148.0],
            [100.0, 148.0, 148.0],
            [148.0, 148.0, 148.0],
            [148.0, 148.0, 148.0],
            [148.0, 148.0, 148.0],
            [148.0, 148.0, 148.0],
            [148.0, 148.0, 148.0],
        ]
    )
    matrix = _matrix(valores, ("qc", "qc", "sample", "sample", "qc", "sample", "sample"))
    metricas = structural_metrics(matrix, pretransform="none", n_components=1)

    assert metricas.pc1_injection_correlation_abs == pytest.approx(0.7572, abs=1e-3)
    assert metricas.pc1_injection_correlation_spread > 0.5, (
        "a amplitude não acusou o empate; o indicador não distingue o caso que "
        "existe para distinguir"
    )
    assert not metricas.is_assessable(max_spread=0.01)


def test_amplitude_acusa_o_denominador_degenerado() -> None:
    """A outra descontinuidade: razão em torno de denominador nulo."""
    valores = np.tile(np.array([1e5, 2e5, 3e5, 4e5]), (6, 1))
    valores[0] *= 1.3
    valores[3] *= 0.7
    valores[5] *= 1.1
    matrix = _matrix(valores, ("qc", "sample", "sample", "qc", "sample", "qc"))
    metricas = structural_metrics(matrix, pretransform="none")

    assert metricas.qc_dispersion_ratio_spread > 1.0
    assert not metricas.is_assessable(max_spread=0.01)


def test_estabilidade_e_deterministica_dada_a_semente() -> None:
    """Regra 3 do README, e a comparação é campo a campo por um motivo.

    `==` de dataclass devolve `False` para dois objetos **idênticos** quando algum
    campo é `nan`, porque `nan != nan`. Comparar por `==` aqui daria falso negativo e
    a conclusão errada seria "a perturbação não é determinística". O multiverso compara
    milhares de relatórios e precisa saber disso.
    """
    from dataclasses import fields

    matrix = _drifting()
    a = structural_metrics(matrix, pretransform="none", stability_seed=7)
    b = structural_metrics(matrix, pretransform="none", stability_seed=7)
    for campo in fields(a):
        x, y = getattr(a, campo.name), getattr(b, campo.name)
        if isinstance(x, float) and np.isnan(x):
            assert np.isnan(y), campo.name
        else:
            assert x == y, campo.name


def test_semente_diferente_pode_dar_amplitude_diferente() -> None:
    """O outro lado: amplitude que ignora a semente não estaria perturbando nada."""
    matrix = _matrix(
        np.array(
            [
                [148.0, 148.0, 148.0],
                [100.0, 148.0, 148.0],
                [148.0, 148.0, 148.0],
                [148.0, 148.0, 148.0],
                [148.0, 148.0, 148.0],
                [148.0, 148.0, 148.0],
                [148.0, 148.0, 148.0],
            ]
        ),
        ("qc", "qc", "sample", "sample", "qc", "sample", "sample"),
    )
    uma = structural_metrics(matrix, pretransform="none", n_components=1, stability_seed=7)
    outra = structural_metrics(matrix, pretransform="none", n_components=1, stability_seed=99)
    assert (
        uma.pc1_injection_correlation_spread != outra.pc1_injection_correlation_spread
        or uma.qc_dispersion_ratio_spread != outra.qc_dispersion_ratio_spread
    )


def test_o_valor_reportado_nao_muda_com_a_semente() -> None:
    """A perturbação mede a estabilidade e **não contamina** a medida.

    Se a semente movesse o valor, o indicador teria trocado uma métrica instável por
    uma métrica estocástica, e as vinte células publicadas deixariam de reproduzir.
    """
    matrix = _drifting()
    uma = structural_metrics(matrix, pretransform="log10p1", stability_seed=1)
    outra = structural_metrics(matrix, pretransform="log10p1", stability_seed=2)
    assert uma.qc_dispersion_ratio == outra.qc_dispersion_ratio
    assert uma.pc1_injection_correlation_abs == outra.pc1_injection_correlation_abs
    assert uma.qc_pc1_trajectory == outra.qc_pc1_trajectory


def test_metrica_indefinida_nao_e_avaliavel_qualquer_que_seja_a_tolerancia() -> None:
    """`nan` é ausência de medida, e nenhuma tolerância a torna avaliável.

    É a diferença entre este desenho e o anterior: sob `nan < 0,30` a matriz caía no
    `erratic` da regra 4 por acidente aritmético.
    """
    valores = np.tile(np.array([1e5, 2e5, 3e5, 4e5]), (6, 1))
    valores[0] *= 1.3
    valores[3] *= 0.7
    valores[5] *= 1.1
    metricas = structural_metrics(
        _matrix(valores, ("qc", "sample", "sample", "qc", "sample", "qc")), pretransform="none"
    )
    for tolerancia in (0.0, 0.01, 1.0, 1e9):
        assert not metricas.is_assessable(max_spread=tolerancia)


def test_nan_ainda_reprova_o_limiar_de_clean_por_semantica_de_float() -> None:
    """O acidente que a guarda existe para impedir, afirmado como fato do Python.

    `nan < 0,30` é `False`, então a matriz não casa a regra 1 e escorrega
    até o `erratic` da regra 4. Este teste não verifica o nosso código: verifica que
    a **premissa** é verdadeira, para que a decisão não repouse em suposição
    sobre semântica de ponto flutuante.
    """
    assert not (float("nan") < 0.30)
    assert not (float("nan") > 0.30)
    assert float("nan") != 0.30


def test_is_assessable_exige_tolerancia_nomeada_e_nao_negativa() -> None:
    metricas = structural_metrics(_drifting(), pretransform="none")
    with pytest.raises(TypeError):
        metricas.is_assessable(0.01)  # type: ignore[misc]
    with pytest.raises(ValueError, match="max_spread não pode ser negativo"):
        metricas.is_assessable(max_spread=-1.0)


def test_spread_distingue_os_tres_desfechos() -> None:
    """`nan` sem medida, `inf` atravessando a definibilidade, amplitude finita."""
    from refino.metrics.structure import _spread

    assert np.isnan(_spread([float("nan"), float("nan")]))
    assert _spread([0.5, float("nan")]) == float("inf")
    assert _spread([0.2, 0.5, 0.3]) == pytest.approx(0.3)
    assert _spread([0.4, 0.4]) == 0.0


def test_scores_nulos_dao_amplitude_zero_sem_dividir_por_nada() -> None:
    """Ruído relativo a amplitude zero é zero, e afirmar estabilidade ali é correto."""
    from refino.metrics.structure import _stability_spreads

    matrix = _matrix(
        np.ones((6, 2)) * np.arange(1, 3), ("qc", "sample", "sample", "qc", "sample", "qc")
    )
    assert _stability_spreads(np.zeros((6, 2)), matrix, replicates=4, seed=0) == (0.0, 0.0)

"""As vinte células numéricas e a trajetória.

As duas tabelas de PCA da especificação são **critério e não expectativa**: a especificação declara
que são reproduzíveis pela pré-transformação fixada na especificação, com três casas
nas doze células de dispersão e duas nas oito de correlação. É a diferença entre
este arquivo e o das medianas in-sample — aqui não há âncora a estabelecer, porque o número
publicado já foi verificado contra a implementação que o produziu.

O salto da trajetória é rotulado **critério-hipótese sobre um arquivo**: o limiar de 4 foi
calibrado no `neg_hilic`, e a segunda metade — os outros três **não** detectam — é
obrigatória. Um detector que sempre detecta não detecta nada.
"""

import numpy as np
import pytest

from refino.metrics.structure import (
    pc1_injection_correlation,
    qc_dispersion_ratio,
    qc_pc1_trajectory,
    structural_metrics,
)
from tests.conftest import matrix_from_normalized

pytestmark = pytest.mark.golden

METHODS = ("none", "batch_loess", "SERRF")

# Tabela de dispersão da especificação, três casas publicadas. Doze células.
PUBLISHED_DISPERSION: dict[tuple[str, str], float] = {
    ("pos_hilic", "none"): 0.930,
    ("pos_hilic", "batch_loess"): 0.683,
    ("pos_hilic", "SERRF"): 0.395,
    ("pos_c18", "none"): 0.796,
    ("pos_c18", "batch_loess"): 0.224,
    ("pos_c18", "SERRF"): 0.599,
    ("neg_hilic", "none"): 1.173,
    ("neg_hilic", "batch_loess"): 0.682,
    ("neg_hilic", "SERRF"): 0.788,
    ("neg_c18", "none"): 0.261,
    ("neg_c18", "batch_loess"): 0.077,
    ("neg_c18", "SERRF"): 0.078,
}

# Tabela de correlação da especificação, duas casas publicadas. **Oito** células, não doze:
# a tabela não traz `batch_loess`, e inventar as quatro faltantes seria fabricar
# expectativa onde não há publicação.
PUBLISHED_CORRELATION: dict[tuple[str, str], float] = {
    ("pos_hilic", "none"): 0.48,
    ("pos_hilic", "SERRF"): 0.48,
    ("pos_c18", "none"): 0.53,
    ("pos_c18", "SERRF"): 0.52,
    ("neg_hilic", "none"): 0.70,
    ("neg_hilic", "SERRF"): 0.62,
    ("neg_c18", "none"): 0.24,
    ("neg_c18", "SERRF"): 0.16,
}

TOLERANCE = 0.05

# Razão entre o salto do quinto para o sexto QC e o maior salto entre
# posições consecutivas anteriores. Observação sobre estes quatro arquivos.
PUBLISHED_STEP_RATIO: dict[str, float] = {
    "neg_hilic": 5.21,
    "pos_hilic": 0.08,
    "neg_c18": 0.29,
    "pos_c18": 0.93,
}
STEP_THRESHOLD = 4.0


def _step_ratio(trajectory: np.ndarray) -> float:
    """Razão entre o salto 5→6 e o maior salto anterior, em módulo.

    Em módulo porque o sinal de PC1 é arbitrário: um degrau para cima e um para
    baixo são o mesmo fenômeno visto por dois SVDs com sinais opostos.
    """
    jumps = np.abs(np.diff(trajectory))
    return float(jumps[4] / jumps[:4].max())


# ---------------------------------------------------------------------------
# As doze células de dispersão
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "method"), list(PUBLISHED_DISPERSION))
def test_ca_2_3_dispersao_reproduz_a_tabela_4_4(mode: str, method: str) -> None:
    matrix = matrix_from_normalized(mode, method)
    obtido = qc_dispersion_ratio(matrix, pretransform="log10p1")
    esperado = PUBLISHED_DISPERSION[(mode, method)]
    assert abs(obtido - esperado) <= TOLERANCE, (
        f"{mode}/{method}: §4.4 publica {esperado:.3f} e medi {obtido:.4f}"
    )


def test_ca_2_3_fecha_nas_tres_casas_publicadas_e_nao_apenas_na_tolerancia() -> None:
    """A especificação afirma reprodução **nas três casas**, e a tolerância é 0,05.

    Uma implementação errada por 0,04 passaria no teste anterior e contradiria a
    procedência da especificação. Este mede a folga real e reprova se ela encolher: o
    máximo observado é da ordem de 3e-4, mais de cem vezes abaixo da tolerância.
    """
    desvios = {
        (mode, method): abs(
            qc_dispersion_ratio(matrix_from_normalized(mode, method), pretransform="log10p1")
            - esperado
        )
        for (mode, method), esperado in PUBLISHED_DISPERSION.items()
    }
    pior = max(desvios.values())
    assert pior < 5e-4, (
        f"a §4.4 declara reprodução nas três casas decimais e o pior desvio é {pior:.2e}; "
        f"por célula: { {k: round(v, 5) for k, v in desvios.items()} }"
    )


# ---------------------------------------------------------------------------
# As oito células de correlação
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "method"), list(PUBLISHED_CORRELATION))
def test_ca_2_4_correlacao_reproduz_a_tabela_4_4(mode: str, method: str) -> None:
    matrix = matrix_from_normalized(mode, method)
    obtido = pc1_injection_correlation(matrix, pretransform="log10p1")
    esperado = PUBLISHED_CORRELATION[(mode, method)]
    assert abs(obtido - esperado) <= TOLERANCE, (
        f"{mode}/{method}: §4.4 publica {esperado:.2f} e medi {obtido:.4f}"
    )


@pytest.mark.parametrize(
    ("mode", "method"), [(m, met) for m in PUBLISHED_STEP_RATIO for met in METHODS]
)
def test_instrumentacao_correlacao_nunca_e_negativa(mode: str, method: str) -> None:
    """Verdadeira por construção — o contrato manda devolver valor absoluto.

    Fica porque é o que pega o `abs` esquecido, e o rótulo de instrumentação fica
    porque ela não pode reprovar uma implementação que faça outra coisa errada.
    Cobre as doze células, inclusive as quatro de `batch_loess` que a tabela não
    publica: a propriedade não depende de haver número publicado.
    """
    matrix = matrix_from_normalized(mode, method)
    assert pc1_injection_correlation(matrix, pretransform="log10p1") >= 0.0


# ---------------------------------------------------------------------------
# As duas metades
# ---------------------------------------------------------------------------


def test_ca_2_5_neg_hilic_mostra_o_degrau() -> None:
    """Metade 1: o `neg_hilic` sem correção tem razão > 4. Mede 5,21."""
    trajectory = qc_pc1_trajectory(
        matrix_from_normalized("neg_hilic", "none"), pretransform="log10p1"
    )
    razao = _step_ratio(trajectory)
    assert razao > STEP_THRESHOLD, f"esperava razão > {STEP_THRESHOLD} e medi {razao:.4f}"
    assert razao == pytest.approx(PUBLISHED_STEP_RATIO["neg_hilic"], abs=0.05)


@pytest.mark.parametrize("mode", ["pos_hilic", "pos_c18", "neg_c18"])
def test_ca_2_5_os_outros_tres_modos_nao_mostram_o_degrau(mode: str) -> None:
    """Metade 2, **obrigatória**: detector que sempre detecta não detecta nada.

    Sem esta metade, o critério ficaria verde com
    uma implementação que devolvesse razão constante de 10.
    """
    trajectory = qc_pc1_trajectory(matrix_from_normalized(mode, "none"), pretransform="log10p1")
    razao = _step_ratio(trajectory)
    assert razao < STEP_THRESHOLD, f"{mode}: esperava razão < {STEP_THRESHOLD} e medi {razao:.4f}"
    assert razao == pytest.approx(PUBLISHED_STEP_RATIO[mode], abs=0.05)


def test_o_limiar_de_quatro_tem_folga_nas_duas_direcoes() -> None:
    """O limiar separa 5,21 de 0,93, e a folga é o que o torna calibração e não sorte.

    Se o maior dos três modos negativos subisse acima de ~2, ou o `neg_hilic`
    caísse abaixo de ~2,5, o limiar de 4 passaria a ser escolha frágil e isso teria
    de ser dito. Medir a folga é como se sabe.
    """
    positivo = PUBLISHED_STEP_RATIO["neg_hilic"]
    maior_negativo = max(v for k, v in PUBLISHED_STEP_RATIO.items() if k != "neg_hilic")
    assert positivo / STEP_THRESHOLD > 1.25, "o modo que detecta está perto demais do limiar"
    assert STEP_THRESHOLD / maior_negativo > 4.0, (
        f"o maior não-detector mede {maior_negativo}, perto demais do limiar {STEP_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# A agregação, e a contagem de descarte que só ela reporta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "method"), list(PUBLISHED_DISPERSION))
def test_structural_metrics_concorda_com_as_tres_funcoes_individuais(
    mode: str, method: str
) -> None:
    """A suposição que justifica `structural_metrics` existir, medida.

    Ela calcula o SVD uma vez e extrai as três; as individuais recalculam. Se PC1
    dependesse de `n_components` — não depende —, os dois caminhos divergiriam e o
    caminho do multiverso daria número diferente do caminho isolado.
    """
    matrix = matrix_from_normalized(mode, method)
    agregado = structural_metrics(matrix, pretransform="log10p1")

    assert agregado.qc_dispersion_ratio == pytest.approx(
        qc_dispersion_ratio(matrix, pretransform="log10p1"), rel=1e-12
    )
    assert agregado.pc1_injection_correlation_abs == pytest.approx(
        pc1_injection_correlation(matrix, pretransform="log10p1"), rel=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(agregado.qc_pc1_trajectory),
        qc_pc1_trajectory(matrix, pretransform="log10p1"),
        rtol=1e-12,
    )


@pytest.mark.parametrize("mode", list(PUBLISHED_STEP_RATIO))
def test_so_o_batch_loess_descarta_coluna_e_a_contagem_e_visivel(mode: str) -> None:
    """O descarte existe, é grande, e é reportado.

    `none` e `SERRF` não descartam nada — as vinte células publicadas dessas duas
    colunas são medidas sobre **todas** as features. O `batch_loess` descarta de
    3,6% a 20,3%, e é por isso que a contagem viaja no resultado: uma razão de
    dispersão medida sobre 79,7% das features, reportada como se fosse sobre
    todas, é número que engana sem errar.
    """
    for method in ("none", "SERRF"):
        limpo = structural_metrics(matrix_from_normalized(mode, method), pretransform="log10p1")
        assert limpo.n_features_dropped_non_finite == 0
        assert limpo.n_features_dropped_zero_variance == 0
        assert limpo.n_features_used == matrix_from_normalized(mode, method).n_features

    sujo = structural_metrics(matrix_from_normalized(mode, "batch_loess"), pretransform="log10p1")
    assert sujo.n_features_dropped_non_finite > 0, (
        f"{mode}/batch_loess deveria ter feature não finita por AR11 e não tem"
    )
    total = matrix_from_normalized(mode, "batch_loess").n_features
    assert (
        sujo.n_features_used
        + sujo.n_features_dropped_non_finite
        + sujo.n_features_dropped_zero_variance
        == total
    )
    assert sujo.n_features_used < total


# ---------------------------------------------------------------------------
# Invariância de escala, nas duas pré-transformações
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factor", [1e-6, 1e-3, 0.5, 2.0, 1e3, 1e6])
def test_pretransform_none_e_exatamente_invariante_a_escala(factor: float) -> None:
    """Igualdade **exata**, sem tolerância — é consequência do desenho.

    `(cx - mean(cx))/sd(cx) = (x - mean(x))/sd(x)` para `c > 0`: o
    autoescalonamento cancela o fator antes do SVD. Asserção de tolerância aqui
    esconderia a diferença entre "cancela exatamente" e "cancela quase".
    """
    from dataclasses import replace

    matrix = matrix_from_normalized("pos_hilic", "none")
    base = pc1_injection_correlation(matrix, pretransform="none")
    escalada = replace(matrix, values=matrix.values * factor)
    assert pc1_injection_correlation(escalada, pretransform="none") == base


def test_pretransform_log10p1_e_invariante_apenas_na_faixa_declarada() -> None:
    """A outra metade, e a assimetria dela é derivável — não é ruído.

    `log10(cx + 1) != log10(x + 1) + log10(c)`, então a equivariância é aproximada.
    Mas a aproximação **só quebra para baixo**: quando `cx >> 1` o `+1` é
    desprezível e `log10(cx + 1) ~ log10(x) + log10(c)`, que o autoescalonamento
    cancela; quando `cx` se aproxima de 1 ou cai abaixo, o `+1` domina e a
    transformação deixa de comprimir escala.

    Medido no `pos_hilic`: desvio 0,0305 em fator 1e-6, 0,0002 em 1e-3 e 0,0005 em
    1e+6. **O plano pedia "desvio maior fora da faixa" nas duas pontas,
    e isso está errado para cima** — a faixa útil é limitada por baixo pela escala
    dos dados, não simétrica em log.
    """
    from dataclasses import replace

    matrix = matrix_from_normalized("pos_hilic", "none")
    base = pc1_injection_correlation(matrix, pretransform="log10p1")

    def desvio(factor: float) -> float:
        escalada = replace(matrix, values=matrix.values * factor)
        return abs(pc1_injection_correlation(escalada, pretransform="log10p1") - base)

    for factor in (1e-3, 0.5, 2.0, 1e3):
        assert desvio(factor) < 0.01, f"dentro da faixa declarada, fator {factor:g}"

    # Abaixo da faixa a invariância quebra, e é o que dá conteúdo à declaração.
    assert desvio(1e-6) > 0.01, (
        "em fator 1e-6 o desvio tem de exceder 0,01; se não excede, a faixa "
        "declarada não corresponde a limite real nenhum e a asserção acima é vácua"
    )
    # Acima da faixa NÃO quebra, e afirmar o contrário seria mais forte que o dado.
    assert desvio(1e6) < 0.01


@pytest.mark.parametrize("pretransform", ["log10p1", "none"])
def test_inverter_o_sinal_dos_scores_nao_altera_a_correlacao(pretransform: str) -> None:
    """Pela costura declarada, porque pela API pública é inexecutável.

    O sinal de um componente vem do SVD e nenhuma entrada o controla. Ou se testa
    por `_pca_scores`, ou o teste é decorativo — é a razão de a costura estar no
    contrato em vez de ser detalhe de implementação.
    """
    from scipy.stats import spearmanr

    from refino.metrics.structure import _pca_scores

    matrix = matrix_from_normalized("neg_hilic", "none")
    scores = _pca_scores(matrix, pretransform=pretransform, n_components=2)  # type: ignore[arg-type]
    pc1 = scores[:, 0]
    orders = matrix.injection_orders

    direto = abs(float(spearmanr(pc1, orders).statistic))
    invertido = abs(float(spearmanr(-pc1, orders).statistic))
    assert direto == invertido
    assert direto == pytest.approx(
        pc1_injection_correlation(matrix, pretransform=pretransform),  # type: ignore[arg-type]
        rel=1e-12,
    )

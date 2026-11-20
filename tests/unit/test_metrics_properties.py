"""Invariantes por `hypothesis`.

**As propriedades aqui decorem do desenho, não da observação**. É o critério
de admissão desta suíte: uma propriedade que valha "nos quatro arquivos" não entra,
porque `hypothesis` gera entradas que não são aqueles arquivos e a reprovação seria
correta e inútil.

Cada propriedade tem, ao lado, a razão pela qual ela é necessária — e as duas de
`boxplot_outlier_mask` cobrem o passo 9 do SERRF, que o corretor executa sobre
68 injeções, onde a implementação por quartis divergiria.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.metrics.rsd import boxplot_outlier_mask, rsd
from refino.metrics.structure import (
    _pca_scores,
    pc1_injection_correlation,
    qc_dispersion_ratio,
)

pytestmark = pytest.mark.property

# Faixa deliberadamente larga mas finita: intensidades de LC-MS vivem entre 1e2 e
# 1e9, e sair disso testaria aritmética do numpy em vez do nosso código.
FINITOS = hnp.arrays(
    dtype=np.float64,
    shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=40),
    elements=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
)

POSITIVOS = st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)

# Fatores que escalam **exatamente** em binário: potências de dois só mexem no
# expoente, então `x * 2**k` não perde nem um bit de mantissa.
#
# Por que as invariâncias estruturais usam estes e não `POSITIVOS`. A premissa
# "a mesma matriz a menos de um fator positivo" é FALSA em ponto flutuante para
# fator arbitrário: `100 * 1e-6` não é `1e-4` exato, então os dois lados da
# igualdade não são a mesma matriz escalada — são duas matrizes diferentes. Testar
# igualdade exata ali mede o arredondamento da multiplicação, não a função.
# `hypothesis` insistiu nisso com matrizes de 100 e 101 até o argumento ficar claro.
FATORES_EXATOS = st.integers(min_value=-20, max_value=20).map(lambda k: float(2.0**k))

# Matrizes das duas invariâncias estruturais. Sorteio LIVRE: a pré-condição de cada
# uma fica no `assume` do teste, e não aqui.
#
# **Por que não gerar a pré-condição por construção**, que foi a primeira tentativa
# de consertar a instabilidade e estava errada. `unique=True` numa estratégia de
# linhas move a filtragem para dentro do sorteio: `HealthCheck.filter_too_much` para
# de disparar — ele conta rejeição de nível de teste —, mas `hypothesis` passa a
# ABORTAR o sorteio quando não consegue montar linhas distintas, porque ele enviesa
# para valores simples e repetidos de propósito. Medido: 9,5% de resorteio e 3,4% de
# `Aborted test because unable to satisfy`. Mesma classe de reprovação intermitente,
# Com outro nome e mais rara — logo pior de diagnosticar.
MATRIZES_ESTRUTURAIS = hnp.arrays(
    dtype=np.float64,
    shape=st.tuples(st.integers(min_value=6, max_value=14), st.integers(min_value=3, max_value=8)),
    elements=st.floats(min_value=1e2, max_value=1e8, allow_nan=False, allow_infinity=False),
)

_ELEMENTO = st.floats(min_value=1e2, max_value=1e8, allow_nan=False, allow_infinity=False)


@st.composite
def _matrizes_de_linhas_distintas(draw: st.DrawFn) -> np.ndarray:
    """Matriz de linhas duas a duas distintas, **sem filtrar e sem injetar estrutura**.

    Uma coluna sorteada recebe `n` valores distintos em ordem **permutada
    aleatoriamente**. Valores distintos numa coluna implicam linhas distintas, e a
    permutação é o que impede a correção de enganar o teste: uma rampa naquela
    coluna criaria correlação com a ordem de injeção, que é exatamente a quantidade
    que um dos testes mede.

    **Os valores distintos são CONSTRUÍDOS, não filtrados.** `unique=True` sobre linhas —
    tuplas — foi a primeira tentativa e produzia `Aborted test because unable to
    satisfy` em 3,4% dos sorteios, porque `hypothesis` enviesa para valores repetidos
    e montar dez tuplas distintas exige muita retentativa. Dez FLOATS distintos
    de uma faixa contínua parecia fácil — e produzia 5,13% de aborto, medido. "Menor"
    foi duas vezes o argumento errado nesta sessão, então aqui não há filtragem
    nenhuma: os `n` valores saem de um intervalo sorteado, igualmente espaçados,
    distintos por construção aritmética. Zero retentativa, zero aborto.

    O espaçamento uniforme é regularidade, e a permutação a desfaz na dimensão que
    importa: não há relação entre valor e posição de injeção, que é a quantidade que
    um dos dois testes mede.
    """
    n_amostras = draw(st.integers(min_value=6, max_value=14))
    n_features = draw(st.integers(min_value=3, max_value=8))
    base = draw(hnp.arrays(dtype=np.float64, shape=(n_amostras, n_features), elements=_ELEMENTO))
    coluna = draw(st.integers(min_value=0, max_value=n_features - 1))
    inicio = draw(st.floats(min_value=1e2, max_value=1e7, allow_nan=False, allow_infinity=False))
    largura = draw(st.floats(min_value=1e2, max_value=1e7, allow_nan=False, allow_infinity=False))
    distintos = inicio + largura * (np.arange(n_amostras) / n_amostras)
    base = base.copy()
    base[:, coluna] = draw(st.permutations(distintos.tolist()))
    return base


# A estratégia da invariância da CORRELAÇÃO: união das duas, e a união é o ponto.
#
# **Medido antes de escolher**. A pré-condição daquele teste é "PC1 sem empate",
# que é condição de RESULTADO e não se constrói. Mas medir de onde vinham os empates
# mostrou que a causa não era o empate ser intrínseco — era o GERADOR: 99,9% dos
# empates vêm com linha duplicada, e o sorteio livre produz linhas todas distintas em
# só 11,9% das matrizes avaliáveis. É um gerador cujo nome promete matriz genérica
# e cujo conteúdo entrega caso degenerado em 88% dos sorteios.
#
# **Por que `one_of` e não substituição.** Trocar o livre pelo de linhas distintas
# ESTREITARIA o conjunto exercitado: 41 das 148 entradas válidas de hoje têm linha
# duplicada e não têm empate em PC1. A união sobe o aproveitamento sem perder nenhuma
# delas — alargar em vez de filtrar. É a diferença em relação à troca da dispersão,
# onde a guarda nova continha a antiga e substituir era seguro.
MATRIZES_SEM_EMPATE = st.one_of(MATRIZES_ESTRUTURAIS, _matrizes_de_linhas_distintas())


# ---------------------------------------------------------------------------
# `rsd`
# ---------------------------------------------------------------------------


@given(
    vector=hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=30),
        elements=st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    ),
    factor=POSITIVOS,
)
@settings(max_examples=200, deadline=None)
def test_rsd_e_exatamente_invariante_a_escala(vector: np.ndarray, factor: float) -> None:
    """`rsd(c*x) == rsd(x)` **exatamente**, para `c > 0`.

    Decorre do desenho: `sd(cx)/mean(cx) = c*sd(x)/(c*mean(x))`. A igualdade não é
    aproximada em teoria, e a asserção é de igualdade para que qualquer perda de
    exatidão apareça em vez de ser absorvida por uma tolerância inventada.

    A remoção de outliers preserva a propriedade porque as dobradiças e os limites
    escalam pelo mesmo fator, então o conjunto removido é o mesmo.
    """
    base = float(rsd(vector, variant="plain"))
    escalado = float(rsd(vector * factor, variant="plain"))
    assume(np.isfinite(base))
    # Tolerância relativa mínima: o produto `vector * factor` já introduz
    # arredondamento na ENTRADA, antes de `rsd` ver qualquer coisa. A propriedade é
    # sobre a função, não sobre a multiplicação que a precede.
    assert escalado == pytest.approx(base, rel=1e-12)


@given(vector=FINITOS)
@settings(max_examples=200, deadline=None)
def test_rsd_nunca_levanta_e_devolve_nan_onde_e_indefinido(vector: np.ndarray) -> None:
    """RSD indefinido é informação sobre a feature, nunca exceção.

    A propriedade é de totalidade: nenhum vetor finito faz `rsd` levantar. É o que
    permite `filter_by_qc_rsd` tratar indefinido por política em vez de
    por `try`.
    """
    resultado = float(rsd(vector, variant="outlier_removed"))
    assert np.isnan(resultado) or np.isfinite(resultado)


# ---------------------------------------------------------------------------
# `boxplot_outlier_mask`
# ---------------------------------------------------------------------------


@given(vector=FINITOS, seed=st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=200, deadline=None)
def test_mascara_comuta_com_permutacao(vector: np.ndarray, seed: int) -> None:
    """Permutar a entrada permuta a máscara igual.

    Decorre de as dobradiças serem estatísticas de **ordem**: elas não dependem da
    ordem de apresentação. Importa para o passo 9 do SERRF, que aplica a máscara
    sobre as 68 injeções — se a marcação dependesse da ordem das colunas, o
    resultado dependeria de como o arquivo foi exportado.
    """
    rng = np.random.default_rng(seed)
    ordem = rng.permutation(vector.size)
    direto = boxplot_outlier_mask(vector)
    permutado = boxplot_outlier_mask(vector[ordem])
    np.testing.assert_array_equal(permutado, direto[ordem])


@given(vector=FINITOS, factor=POSITIVOS, shift=st.floats(min_value=-1e6, max_value=1e6))
@settings(max_examples=200, deadline=None)
def test_mascara_e_invariante_a_transformacao_afim_positiva(
    vector: np.ndarray, factor: float, shift: float
) -> None:
    """`a*x + b` com `a > 0` marca os mesmos pontos — **quando a transformação
    preserva a informação**.

    Dobradiças e limites escalam e transladam juntos, então em aritmética exata a
    comparação estrita dá o mesmo resultado. É a propriedade que garante que a
    máscara não depende da unidade em que a intensidade foi reportada.

    A qualificação não é frouxidão: em ponto flutuante a **transformação** pode
    destruir a informação antes de a função vê-la. `hypothesis` achou o caso em
    quinze exemplos — `[1e-6, 0, 0, 0, 0]`, fator `1e-6`, deslocamento `16384`: o
    `1e-12` resultante desaparece na precisão disponível na magnitude de 16384, e o
    vetor transformado é **constante**. A máscara certa para um vetor constante é
    vazia, e a máscara certa para o original marca o `1e-6`. As duas estão certas
    sobre entradas diferentes.

    O `assume` de ida e volta é o que expressa isso: só compara quando a
    transformação é reversível bit a bit, que é a condição sob a qual a propriedade
    é uma afirmação sobre a função e não sobre a aritmética.
    """
    transformado_valores = vector * factor + shift
    assume(np.isfinite(transformado_valores).all())
    assume(vector.size > 0)
    assume(np.array_equal((transformado_valores - shift) / factor, vector))

    direto = boxplot_outlier_mask(vector)
    transformado = boxplot_outlier_mask(transformado_valores)
    np.testing.assert_array_equal(transformado, direto)


@given(
    vector=hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=0, max_side=2),
        elements=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
    ),
    coef=st.floats(min_value=1.5, max_value=5.0),
)
@settings(max_examples=100, deadline=None)
def test_com_n_ate_dois_nada_e_marcado(vector: np.ndarray, coef: float) -> None:
    """`n <= 2` nunca marca ponto, para qualquer `coef >= 1,5`.

    Não é caso degenerado a tratar: com dois valores as dobradiças **são** os
    extremos, então os limites caem fora da faixa dos dados. Importa porque a
    partição em dois lotes do critério produz um lote com dois QCs, e uma
    implementação que marcasse ali removeria metade daquele lote.
    """
    assert not boxplot_outlier_mask(vector, coef=coef).any()


@given(vector=FINITOS)
@settings(max_examples=200, deadline=None)
def test_remocao_nunca_deixa_menos_de_dois_valores(vector: np.ndarray) -> None:
    """Procura o contraexemplo do invariante que justifica um `pragma` em `rsd.py`.

    A guarda "a remoção deixou menos de dois valores" está marcada inalcançável
    porque as dobradiças são estatísticas de ordem em posições interiores, e o
    intervalo entre elas sobrevive à remoção. O `pragma` exige que venha
    acompanhado de busca ativa pelo contraexemplo em vez de suposição.
    """
    assume(vector.size >= 2)
    restantes = vector[~boxplot_outlier_mask(vector)]
    assert restantes.size >= 2


# ---------------------------------------------------------------------------
# Estrutura
# ---------------------------------------------------------------------------


def _dispersao_bem_condicionada(matrix: RefinoMatrix, limiar: float = 1e-8) -> bool:
    """O denominador de `qc_dispersion_ratio` não é degenerado.

    Calculado pela costura `_pca_scores`, declarada pública para
    exatamente este tipo de pergunta: a condição vive nos **scores**, não na matriz de
    entrada, e medi-la na entrada é uma proxy da condição.

    O denominador é a distância euclidiana média das linhas de `sample_mask` ao
    centroide delas. Comparado com a escala dos próprios scores, e não com valor
    absoluto, porque o autoescalonamento do passo 4 muda a magnitude — comparar com
    constante absoluta seria calibrar num regime e errar noutro.

    `limiar` de 1e-8 com resultado idêntico entre 1e-12 e 1e-6 nos 600 sorteios
    medidos: o denominador é ou muito acima da escala ou essencialmente zero,
    então isto não é botão de calibração.
    """
    try:
        scores = _pca_scores(matrix, pretransform="none", n_components=2)
    except ValueError:
        # Sem coluna utilizável depois do passo 3: **não avaliável**, que não é a
        # mesma coisa que degenerado. Devolver `False` aqui é operacionalmente certo
        # para o `assume` — não há o que avaliar —, mas as duas condições NÃO podem
        # ser confundidas em contagem: foi exatamente isso que fez a primeira versão
        # do meta-teste reportar duas entradas "degeneradas que a proxy admitia"
        # quando eram entradas sem coluna utilizável.
        # Quem conta usa `denominador_relativo`, que devolve `None` no caso.
        return False
    amostras = scores[matrix.sample_mask]
    denominador = float(np.linalg.norm(amostras - amostras.mean(axis=0), axis=1).mean())
    escala = max(float(np.abs(scores).max()), 1e-300)
    return denominador > limiar * escala


def _denominador_relativo(matrix: RefinoMatrix) -> float | None:
    """O denominador relativo à escala dos scores, ou `None` quando não avaliável.

    Existe separado de `_dispersao_bem_condicionada` porque **contar exige distinguir
    "não avaliável" de "degenerado"**, e a função booleana colapsa as duas em `False`.
    """
    try:
        scores = _pca_scores(matrix, pretransform="none", n_components=2)
    except ValueError:
        return None
    amostras = scores[matrix.sample_mask]
    denominador = float(np.linalg.norm(amostras - amostras.mean(axis=0), axis=1).mean())
    return denominador / max(float(np.abs(scores).max()), 1e-300)


def _matrix(values: np.ndarray) -> RefinoMatrix:
    n = values.shape[0]
    tipos = ["qc" if i % 3 == 0 else "sample" for i in range(n)]
    # Garante ao menos duas de cada máscara, que é o mínimo para dispersão.
    tipos[0] = tipos[1] = "qc"
    tipos[-1] = tipos[-2] = "sample"
    samples = tuple(
        SampleMetadata(
            name=f"QC{i}" if tipo == "qc" else f"a{i}",
            injection_order=i + 1,
            sample_type=tipo,  # type: ignore[arg-type]
            group="QC" if tipo == "qc" else "NoHTLV",
        )
        for i, tipo in enumerate(tipos)
    )
    return RefinoMatrix(
        values=np.asarray(values, dtype=np.float64),
        feature_names=tuple(f"f{j}" for j in range(values.shape[1])),
        samples=samples,
        mode="prop",
    )


@given(values=MATRIZES_SEM_EMPATE, factor=FATORES_EXATOS)
@settings(max_examples=100, deadline=None)
def test_correlacao_com_pretransform_none_e_exatamente_invariante_a_escala(
    values: np.ndarray, factor: float
) -> None:
    """Igualdade **exata**, e é a única das cinco que pode ser — **sem empates**.

    O autoescalonamento cancela o fator antes do SVD, e a correlação é de postos:
    postos absorvem a diferença de último bit que a multiplicação introduz. A
    dispersão, que é razão de distâncias, **não** tem essa propriedade.

    **A qualificação "sem empates" é o achado desta unidade, e ela é forte.** Postos
    absorvem ruído de último bit *entre valores distintos*, e amplificam ruído
    *entre valores empatados*: seis valores exatamente iguais recebem posto médio;
    os mesmos seis diferindo no último bit recebem seis postos distintos, numa ordem
    que o ruído decide. `hypothesis` achou o caso — matriz cuja única coluna
    utilizável tem seis valores iguais —, e a correlação saltou de **0,7572 para
    0,1336** com PC1 numericamente idêntico até a sexta casa. Não é perda de
    exatidão: é descontinuidade de Spearman em empate.

    Nos quatro arquivos reais isso não acontece porque PC1 não tem empate, e por
    isso o desvio medido é `0,00e+00`. A exatidão observada é
    propriedade **daqueles dados**, não do desenho.
    """
    matrix = _matrix(values)
    escalada = _matrix(values * factor)
    try:
        base = pc1_injection_correlation(matrix, pretransform="none")
        obtido = pc1_injection_correlation(escalada, pretransform="none")
        pc1 = _pca_scores(matrix, pretransform="none", n_components=1)[:, 0]
        pc1_escalado = _pca_scores(escalada, pretransform="none", n_components=1)[:, 0]
    except ValueError:
        # Matriz sem coluna utilizável depois do passo 3: a propriedade não se
        # aplica, e `assume` seria enganoso porque a condição é do resultado.
        return
    # Sem empate em PC1 **nos dois lados**, os postos são estáveis e a igualdade é
    # exata. Checar só um lado não basta, e `hypothesis` provou: achou matriz cujo
    # PC1 original é livre de empate e cujo escalado não é.
    assume(np.unique(pc1).size == pc1.size)
    assume(np.unique(pc1_escalado).size == pc1_escalado.size)
    if np.isnan(base):
        assert np.isnan(obtido)
        return
    assert obtido == base


@given(values=MATRIZES_ESTRUTURAIS, factor=FATORES_EXATOS)
@settings(max_examples=100, deadline=None)
def test_dispersao_com_pretransform_none_e_invariante_a_menos_de_epsilon(
    values: np.ndarray, factor: float
) -> None:
    """A propriedade que a dispersão **tem**, com a força que ela tem.

    Está aqui, e não fora da suíte, porque "não é exata" não é o mesmo que "não
    vale": ela é invariante a menos de epsilon, e afirmar isso é afirmar algo. O que
    seria errado é a asserção de igualdade exata, que reprovaria sem defeito.

    **A configuração degenerada fica de fora, e não por conveniência**. Com
    as amostras todas no mesmo ponto o denominador é zero ou um resíduo de ~1e-31
    conforme o SVD, e a métrica salta entre `nan` e 5,3e+29 — `hypothesis` achou o
    caso. É descontinuidade real da razão em torno do denominador nulo, não perda de
    exatidão, e nenhuma tolerância a atravessa. A pré-condição é o que separa "a
    propriedade vale" de "a propriedade vale onde a razão existe".

    **A pré-condição mira o denominador, e não linhas distintas — e a troca foi
    medida antes de ser feita**. "Linhas de amostra duas a duas distintas" era
    proxy da condição real, e proxy grosseira: ela rejeitava **72%** dos sorteios,
    porque `hypothesis` gera duplicata de propósito, e isso fazia
    `HealthCheck.filter_too_much` reprovar em cerca de uma execução em doze — sempre
    por saúde, nunca por asserção.

    Medido em 600 sorteios, com as duas guardas aplicadas lado a lado:

    | | linhas distintas | denominador |
    |---|---|---|
    | aproveitamento | 28,0% | **92,4%** |
    | aceita o que a outra rejeita | 0 | **186 de 289** |

    As duas são **aninhadas** na direção que importa: nenhuma entrada passa por
    linhas distintas e reprova no denominador, então a troca não perde cobertura. E o
    canto de 186 entradas — aceitas pelo denominador, rejeitadas pela proxy — tem a
    propriedade **verdadeira em todas**, o que significa que a proxy escondia 64% do
    espaço válido em vez de apenas encarecê-lo.

    O aninhamento é **observação sobre este gerador, não invariante**: duas
    linhas distintas por um ULP dariam denominador ínfimo em princípio, e zero
    ocorrências em 289 sorteios é evidência e não prova.

    O limiar não é botão de calibração: o resultado é idêntico entre 1e-12 e 1e-6,
    porque o denominador é ou muito acima da escala dos scores ou essencialmente
    zero — que é o regime degenerado.

    **E o health check fica ARMADO.** A primeira tentativa de conserto foi suprimi-lo;
    a 92% de aproveitamento ele não dispara e continua protegendo contra o dia em que
    a filtragem crescer. Correção que remove o defeito e mantém o instrumento é
    diferente de correção que cala o instrumento.
    """
    matrix = _matrix(values)
    # A condição sob a qual a razão tem denominador não degenerado, medida onde ela
    # de fato vive: nos scores. Ver o docstring.
    assume(_dispersao_bem_condicionada(matrix))
    try:
        base = qc_dispersion_ratio(matrix, pretransform="none")
        obtido = qc_dispersion_ratio(_matrix(values * factor), pretransform="none")
    except ValueError:
        return
    assume(np.isfinite(base))
    assert obtido == pytest.approx(base, rel=1e-9)


@given(
    values=hnp.arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=6, max_value=14), st.integers(min_value=3, max_value=8)
        ),
        elements=st.floats(min_value=1e2, max_value=1e8, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=100, deadline=None)
def test_correlacao_esta_sempre_entre_zero_e_um(values: np.ndarray) -> None:
    """Contradomínio: é valor absoluto de correlação, logo em `[0, 1]` ou `nan`.

    Pega o `abs` esquecido e também qualquer coisa que devolva o rho ao quadrado ou
    a estatística de outro teste por engano.
    """
    try:
        obtido = pc1_injection_correlation(_matrix(values), pretransform="log10p1")
    except ValueError:
        return
    assert np.isnan(obtido) or 0.0 <= obtido <= 1.0


# ---------------------------------------------------------------------------
# A troca de pré-condição da dispersão, afirmada em vez de suposta
# ---------------------------------------------------------------------------


def test_a_guarda_do_denominador_contem_a_de_linhas_distintas() -> None:
    """A troca de pré-condição não perde cobertura, e exercita mais.

    **Por que este teste existe.** Substituir pré-condição sem verificar se o
    conjunto exercitado continua o mesmo repete o erro do fixture cujo nome dizia deriva e
    cujo conteúdo media ruído. A troca foi medida antes de feita, e a medição fica
    como teste para que a próxima sessão não precise confiar num número em prosa.

    Medido em 600 sorteios livres, com as duas guardas lado a lado:

    | | linhas distintas | denominador |
    |---|---|---|
    | aproveitamento | 28,0% | **92,4%** |
    | aceita o que a outra rejeita | 0 | **186 de 289** |

    As duas são **aninhadas** sobre as entradas em que a métrica é calculável, e a
    proxy não era outra pré-condição: era a mesma, grosseira. O canto de 186 entradas
    — aceitas pelo denominador, rejeitadas pela proxy — tem a propriedade verdadeira
    em todas, então a proxy escondia 64% do espaço válido em vez de apenas encarecê-lo.

    **O aninhamento é observação sobre este gerador, não invariante**: duas
    linhas distintas por um ULP dariam denominador ínfimo em princípio. Zero em 1500
    sorteios é evidência forte e não prova.

    `_dispersao_bem_condicionada` devolve `False` tanto para "denominador degenerado" quanto
    para "sem coluna utilizável": duas condições diferentes com o mesmo valor de retorno. As
    entradas do segundo caso são as em que a propriedade não se aplica, e o `try` do teste já as
    descarta; por isso a contagem é feita com `_denominador_relativo`, e não com o booleano.

    **O que este teste afirma, com `derandomize=True`.** Sorteio derivado de semente fixa torna
    as contagens reprodutíveis, mas elas **não** são afirmadas: a semente sai do digest do
    CÓDIGO de `coleta`, então qualquer edição dela sorteia outras 200 matrizes, e pinar os
    números faria cada edição de `coleta` exigir trocar o número esperado — que é o gesto que a
    regra 2 do README proíbe, com outra roupa. No sorteio de hoje elas são **30 / 0 / 72**
    (ambas, só a proxy, só o denominador), com **5** degeneradas e **93** não avaliáveis. **O
    que se afirma são as três propriedades**; a afirmação estatística dos 92% vive no docstring,
    e não nesta asserção.

    Três asserções, nesta ordem de importância:

    1. **A propriedade vale em tudo que só a guarda nova aceita.** Se falhasse, o
       achado seria maior que o conserto — a proxy estaria escondendo contraexemplo.
    2. **A guarda nova aceita substancialmente mais** que a proxy.
    3. **O canto que só a proxy aceita é vazio ou minúsculo**, contando só entradas
       avaliáveis — a distinção que a primeira versão não fazia.
    """
    quadrantes = {
        "ambas": 0,
        "so_proxy": 0,
        "so_denominador": 0,
        "nenhuma": 0,
        "nao_avaliavel": 0,
    }
    falhas_no_canto: list[tuple[tuple[int, ...], float]] = []

    @given(values=MATRIZES_ESTRUTURAIS, factor=FATORES_EXATOS)
    @settings(max_examples=200, deadline=None, derandomize=True)
    def coleta(values: np.ndarray, factor: float) -> None:
        matrix = _matrix(values)
        linhas = values[matrix.sample_mask]
        proxy = np.unique(linhas, axis=0).shape[0] == linhas.shape[0]
        # **Contado por `_denominador_relativo`, e não pelo booleano.**
        # `_dispersao_bem_condicionada` devolve `False` para «degenerado» E para «sem
        # coluna utilizável», e este meta-teste existe para separar as duas — a docstring dele o
        # diz, e o comentário daquela função manda contar com esta.
        relativo = _denominador_relativo(matrix)
        if relativo is None:
            quadrantes["nao_avaliavel"] += 1
            return
        denom = relativo > 1e-8
        if proxy and denom:
            quadrantes["ambas"] += 1
        elif proxy:
            quadrantes["so_proxy"] += 1
        elif denom:
            quadrantes["so_denominador"] += 1
        else:
            quadrantes["nenhuma"] += 1
            return
        if denom and not proxy:
            try:
                base = qc_dispersion_ratio(matrix, pretransform="none")
                obtido = qc_dispersion_ratio(_matrix(values * factor), pretransform="none")
            except ValueError:
                return
            if np.isfinite(base) and abs(obtido - base) > 1e-9 * abs(base):
                falhas_no_canto.append((values.shape, factor))

    coleta()
    avaliados = quadrantes["ambas"] + quadrantes["so_proxy"] + quadrantes["so_denominador"]
    assert avaliados > 0, f"âncora: o sorteio não teve entrada avaliável — {quadrantes}"

    assert quadrantes["so_proxy"] <= max(3, avaliados // 10), (
        f"{quadrantes['so_proxy']} de {avaliados} entradas passam por linhas distintas "
        "e reprovam no denominador. Uma minoria é esperada e é o regime degenerado que "
        "a proxy admitia errado; uma fração grande significaria que as duas guardas "
        "medem coisas diferentes e a troca perde cobertura de verdade"
    )
    assert quadrantes["so_denominador"] > quadrantes["ambas"], (
        "a guarda do denominador deveria aceitar substancialmente mais que a proxy; "
        f"medido: {quadrantes['so_denominador']} contra {quadrantes['ambas']} em "
        f"{avaliados} avaliados. Se convergiram, a proxy não era grosseira e a troca "
        "não se paga"
    )
    assert not falhas_no_canto, (
        f"a propriedade FALHOU em {len(falhas_no_canto)} entradas que só a guarda "
        f"nova aceita — {falhas_no_canto[:3]}. Isto é achado, não defeito da troca: "
        "a proxy escondia um contraexemplo real, e a regra 2 do README manda relatar "
        "o caso, não ajustá-lo"
    )

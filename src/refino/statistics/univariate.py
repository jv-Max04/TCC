"""Teste univariado por feature, com controle de taxa de falsa descoberta.

O `q` de Benjamini-Hochberg é conferido contra `scipy.stats.false_discovery_control`, que é a
mesma definição do `p.adjust(method="BH")` do R, em processo e determinística. O R entra como
referência congelada em fixture.

O passo que as implementações de BH costumam esquecer é o último. Ordenar e aplicar `p * n / i`
produz uma sequência que não é monotônica: um p-valor grande em posto baixo pode gerar `q` menor
que o do vizinho anterior. A definição exige monotonicidade, obtida por mínimo cumulativo de
trás para frente sobre a sequência ordenada. Sem esse passo o resultado parece plausível e
discorda de qualquer referência. `tests/unit/test_benjamini_hochberg.py` planta essa versão e
confere que ela reprova.
"""

import numpy as np
from scipy.stats import binom, mannwhitneyu, norm, shapiro, ttest_ind

from refino.domain.matrix import RefinoMatrix
from refino.domain.statistics import (
    FoldChangeInterval,
    FoldChangeIntervalUndefined,
    NormalityDecision,
    UnivariateCounts,
    UnivariateResult,
)

__all__ = [
    "FOLD_CHANGE_CONFIDENCE",
    "NORMALITY_ALPHA",
    "benjamini_hochberg",
    "fold_change_interval",
    "univariate_test",
]

# Mínimo por grupo para existir contraste: com uma amostra de um lado, nem Welch nem
# Mann-Whitney têm estatística.
MIN_AMOSTRAS_POR_GRUPO = 2

# Shapiro-Wilk exige ao menos três pontos e degrada acima de 5.000 (o `scipy` avisa). Abaixo do
# mínimo a decisão automática cai no não paramétrico, que é a escolha conservadora: Mann-Whitney
# vale também sob normalidade, com menos poder. Acima do máximo o teste corre sobre uma amostra.
MIN_PARA_SHAPIRO = 3
MAX_PARA_SHAPIRO = 5_000


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """`q` de Benjamini-Hochberg, na mesma ordem da entrada.

    Três passos:

    - ordenar os p-valores;
    - escalar por `n / i`, com `i` o posto de 1 a `n`;
    - impor monotonicidade por mínimo cumulativo de trás para frente, e cortar em 1.

    Empates recebem `q` igual: o `argsort` estável mantém a ordem original entre empatados, e o
    mínimo cumulativo os iguala pelo menor valor do bloco.

    `NaN` não entra no ajuste e volta como `NaN`. Tratá-lo como 1 o contaria no denominador do
    posto e inflaria todo `q` do vetor; tratá-lo como 0 o faria significativo. O `n` do ajuste é
    a contagem de p-valores definidos, e não o comprimento do vetor. Feature sem p-valor
    definido, por variância zero num grupo, é caso real depois do filtro.

    Vetor vazio devolve vazio, sem levantar: matriz sem feature nenhuma é entrada legítima
    depois de filtro agressivo.
    """
    valores = np.asarray(pvalues, dtype=np.float64)
    if valores.ndim != 1:
        raise ValueError(f"benjamini_hochberg recebe vetor; recebi forma {valores.shape}")
    if valores.size == 0:
        return np.empty(0, dtype=np.float64)

    definidos = np.isfinite(valores)
    saida = np.full(valores.shape, np.nan, dtype=np.float64)
    if not definidos.any():
        return saida

    p = valores[definidos]
    n = p.size
    ordem = np.argsort(p, kind="stable")
    ordenados = p[ordem]

    escalados = ordenados * n / np.arange(1, n + 1)
    # O passo que a definição exige e que a versão ingênua omite: o `q` de um posto nunca
    # excede o do posto seguinte.
    monotonicos = np.minimum.accumulate(escalados[::-1])[::-1]

    ajustados = np.empty(n, dtype=np.float64)
    ajustados[ordem] = np.minimum(monotonicos, 1.0)
    saida[definidos] = ajustados
    return saida


NORMALITY_ALPHA = 0.05
"""Alfa do teste de normalidade que decide `parametric=None`.

O teste é um só, sobre os resíduos de todas as features juntos, até 5.000 deles. Com essa
população ele tem muito poder: rejeita desvios pequenos da normalidade, inclusive o acúmulo que
a imputação por valor fixo deixa numa feature com muitos zeros. Por isso `parametric` é
parâmetro, e não só automático.
"""


def univariate_test(
    matrix: RefinoMatrix,
    groups: tuple[str, str],
    parametric: bool | None = None,
    *,
    fold_change_matrix: RefinoMatrix | None = None,
) -> UnivariateResult:
    """Teste por feature entre dois grupos, com `q` de Benjamini-Hochberg.

    O resultado traz `feature`, `statistic`, `pvalue`, `qvalue_fdr`, `fold_change` e `log2fc`
    como campos, e não como `DataFrame`: `pandas` fica nas bordas de I/O. O contraste é sempre
    entre dois grupos; com mais de dois na matriz, quem chama escolhe o par.

    `parametric=None` decide por Shapiro-Wilk sobre os resíduos dentro de cada grupo, a
    `NORMALITY_ALPHA`. `True` força Welch, `False` força Mann-Whitney.

    `fold_change` é razão de medianas, e não de médias: intensidade de LC-MS tem cauda longa, e
    a média de um grupo pequeno é arrastada por uma injeção aberrante. É a razão entre as
    medianas dos dois grupos, que não é a mediana das razões.

    `fold_change_matrix` diz de qual matriz sai o fold change. Para o `p`, a matriz processada é
    a certa. Para a razão de medianas não é: depois de `log10` e de um escalonamento que centra,
    a mediana de um grupo é negativa com frequência, e o `log2fc` de uma razão não positiva é
    `nan`. Com `None`, o padrão, o fold change sai da mesma matriz do `p`. É o que
    `multiverse/stability.py` usa, e mudar o padrão deslocaria a `sign_consistency` já
    persistida. A rota das figuras passa a matriz pré-transformação, o mesmo ponto em que o
    MetaboAnalyst calcula o FC.

    A matriz alternativa tem de ter as mesmas features, na mesma ordem, e as mesmas amostras:
    `pvalue[i]` e `log2fc[i]` têm de ser da mesma feature.
    """
    a, b = _blocos_dos_grupos(matrix, groups)
    usa_parametrico, decisao = _decide_normalidade(a, b, parametric)

    with np.errstate(invalid="ignore", divide="ignore"):
        if usa_parametrico:
            estatisticas, pvalores = ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit")
        else:
            estatisticas, pvalores = mannwhitneyu(a, b, axis=0, alternative="two-sided")

    estatisticas = np.asarray(estatisticas, dtype=np.float64)
    pvalores = np.asarray(pvalores, dtype=np.float64)

    origem = matrix
    if fold_change_matrix is not None:
        _confere_matriz_do_fold_change(matrix, fold_change_matrix)
        origem = fold_change_matrix
    fa, fb = (a, b) if fold_change_matrix is None else _blocos_dos_grupos(origem, groups)
    razao = _fold_change(fa, fb)
    log2fc = np.where(razao > 0.0, np.log2(np.where(razao > 0.0, razao, 1.0)), np.nan)
    qvalores = benjamini_hochberg(pvalores)

    return UnivariateResult(
        feature=matrix.feature_names,
        statistic=estatisticas,
        pvalue=pvalores,
        qvalue_fdr=qvalores,
        fold_change=razao,
        log2fc=log2fc,
        parametric=usa_parametrico,
        groups=groups,
        pvalue_provenance=matrix.provenance,
        fold_change_provenance=origem.provenance,
        counts=UnivariateCounts(
            n_features=len(matrix.feature_names),
            log2fc_undefined=int((~np.isfinite(log2fc)).sum()),
            pvalue_undefined=int((~np.isfinite(pvalores)).sum()),
            # `== 0.0` e não `< eps`: só o zero exato vai para `+inf` no `-log10`, e um limiar
            # contaria features que a figura desenha sem problema.
            pvalue_zero=int((pvalores == 0.0).sum()),
            qvalue_undefined=int((~np.isfinite(qvalores)).sum()),
        ),
        normality=decisao,
    )


def _confere_matriz_do_fold_change(matrix: RefinoMatrix, alternativa: RefinoMatrix) -> None:
    """Confere que a matriz do fold change tem as mesmas features e as mesmas amostras.

    Os dois desalinhamentos dariam números finitos e plausíveis. Com outra lista de features, ou
    outra ordem, `pvalue[i]` seria de uma feature e `log2fc[i]` de outra. Com outras amostras em
    cada grupo, o fold change sairia de uma população e o `p` de outra.

    No pipeline atual as duas matrizes coincidem, porque nenhuma etapa remove amostras e as que
    removem features correm antes da transformação. A guarda existe para uma etapa futura que
    filtre depois dela.

    O `sample_type` entra na comparação porque `_blocos_dos_grupos` seleciona por `sample_mask`:
    duas matrizes com os mesmos nomes e rótulos, uma delas com uma linha `validate`, teriam
    populações diferentes.
    """
    if matrix.feature_names != alternativa.feature_names:
        comuns = len(set(matrix.feature_names) & set(alternativa.feature_names))
        raise ValueError(
            f"`fold_change_matrix` tem {alternativa.n_features} features e a matriz do teste tem "
            f"{matrix.n_features}, com {comuns} nome(s) em comum — e a lista tem de ser a MESMA, "
            "na mesma ordem. O resultado indexa as duas pela posição, então uma diferença aqui "
            "põe o `p` de uma feature ao lado do fold change de outra, e os dois saem finitos."
        )
    seu = tuple((a.name, a.group, a.sample_type) for a in matrix.samples)
    outro = tuple((a.name, a.group, a.sample_type) for a in alternativa.samples)
    if seu != outro:
        raise ValueError(
            "`fold_change_matrix` tem outras amostras, ou os mesmos nomes noutros grupos ou com "
            "outro `sample_type`, que a matriz do teste. O fold change sairia de uma população e "
            "o `p` de outra, e a figura declararia um par de grupos só."
        )


def _blocos_dos_grupos(
    matrix: RefinoMatrix, groups: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """As duas submatrizes, pelas linhas que `_linhas_dos_grupos` escolhe.

    A seleção das linhas é função à parte porque o intervalo do fold change precisa das mesmas
    linhas para ler a máscara de imputadas.
    """
    linhas_a, linhas_b = _linhas_dos_grupos(matrix, groups)
    valores = np.asarray(matrix.values, dtype=np.float64)
    return valores[linhas_a], valores[linhas_b]


def _linhas_dos_grupos(
    matrix: RefinoMatrix, groups: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """As linhas de cada grupo, como máscaras, com as guardas de entrada.

    Grupo inexistente levanta nomeando os disponíveis. O nome de grupo vem do arquivo do
    laboratório e o domínio é aberto, então a guarda é de pertinência e não de valor fechado.

    `sample_mask` é lida uma vez e governa as duas pontas: a lista de disponíveis que a guarda
    confere e a seleção das linhas de cada bloco. O QC fica fora porque é a régua e não um
    grupo, e `validate` fica fora porque não é amostra de estudo. `sample_mask` não é o
    complemento de `qc_mask`, porque `SampleType` tem três valores.

    A mensagem de recusa não diz que o grupo não existe: o rótulo dos QCs existe na matriz, só
    não é grupo de estudo.
    """
    primeiro, segundo = groups
    if primeiro == segundo:
        raise ValueError(f"os dois grupos são o mesmo ({primeiro!r}); não há contraste a testar")

    por_linha = np.array(matrix.groups)
    estudo = np.asarray(matrix.sample_mask)
    disponiveis = set(por_linha[estudo].tolist())
    for nome in groups:
        if nome not in disponiveis:
            raise ValueError(
                f"grupo {nome!r} não está entre as amostras de estudo desta matriz. "
                f"Disponíveis: {sorted(disponiveis)}"
            )

    # O `& estudo` impede que uma amostra `validate` com o rótulo de um dos dois grupos entre no
    # bloco e o `p` saia sobre uma população diferente da que `groups` declara.
    linhas = tuple((por_linha == nome) & estudo for nome in groups)
    for nome, selecionadas in zip(groups, linhas, strict=True):
        if int(selecionadas.sum()) < MIN_AMOSTRAS_POR_GRUPO:
            raise ValueError(
                f"grupo {nome!r} tem {int(selecionadas.sum())} amostra(s); teste entre grupos "
                f"exige ao menos {MIN_AMOSTRAS_POR_GRUPO} de cada lado"
            )
    return linhas[0], linhas[1]


def _decide_normalidade(
    a: np.ndarray, b: np.ndarray, parametric: bool | None
) -> tuple[bool, NormalityDecision]:
    """Shapiro-Wilk sobre os resíduos dentro de cada grupo, a `NORMALITY_ALPHA`.

    A decisão é uma para a matriz toda, e não uma por feature. Decidir por feature misturaria
    duas estatísticas na coluna `statistic` e dois nulos na coluna `pvalue`, e o `q` de BH seria
    calculado sobre p-valores de testes diferentes.

    Testam-se resíduos, e não valores brutos: a hipótese é normalidade dentro do grupo, e a
    diferença de posição entre grupos apareceria como não normalidade no conjunto.

    O resíduo é em relação à média do grupo, que é o do modelo que o teste t ajusta. Centrar
    pela mediana faz o teste rejeitar dado normal: num grupo de tamanho ímpar a mediana é uma
    das observações, cada feature contribui com um resíduo exatamente zero, e com milhares de
    resíduos o Shapiro-Wilk vê esse acúmulo.

    A amostragem dos 5.000 é igualmente espaçada, e não os primeiros: ver `_amostra_espalhada`.

    Limitação: os resíduos de todas as features entram juntos, e features com escalas muito
    diferentes formam uma mistura que não é normal mesmo quando cada uma é. Padronizar cada
    feature pelo próprio desvio não resolve, porque o desvio estimado com poucas amostras deixa
    caudas de t que o teste vê.

    Devolve dois valores: o `bool` que escolhe o teste e a `NormalityDecision` que diz como se
    chegou a ele. Sem resíduos suficientes para o Shapiro-Wilk o `bool` é `False`, como quando o
    teste rejeita, e a decisão registra que nenhum teste correu.
    """
    if parametric is not None:
        return parametric, NormalityDecision(route="forced", shapiro_pvalue=None, n_residuals=None)

    residuos = np.concatenate(
        [bloco - np.mean(bloco, axis=0, keepdims=True) for bloco in (a, b)], axis=0
    )
    achatados = residuos[np.isfinite(residuos)].ravel()
    if achatados.size < MIN_PARA_SHAPIRO:
        return False, NormalityDecision(
            route="shapiro_unavailable", shapiro_pvalue=None, n_residuals=int(achatados.size)
        )

    amostra = _amostra_espalhada(achatados, MAX_PARA_SHAPIRO)
    p = float(shapiro(amostra).pvalue)
    return p > NORMALITY_ALPHA, NormalityDecision(
        route="shapiro", shapiro_pvalue=p, n_residuals=int(amostra.size)
    )


def _amostra_espalhada(valores: np.ndarray, teto: int) -> np.ndarray:
    """Até `teto` valores, em posições igualmente espaçadas do vetor inteiro.

    O vetor chega achatado linha a linha, isto é, amostra a amostra, com o primeiro grupo do par
    antes do segundo. Cortar nos `teto` primeiros entregaria ao Shapiro-Wilk as primeiras
    amostras do primeiro grupo e nada do segundo: com cerca de 6 mil features, os 5.000
    primeiros resíduos não cobrem nem uma amostra.

    Posições igualmente espaçadas percorrem todas as linhas, nos dois grupos. Não há sorteio
    porque `univariate_test` não recebe semente, e toda operação estocástica precisa de uma.
    """
    if valores.size <= teto:
        return valores
    posicoes = np.linspace(0, valores.size - 1, teto).round().astype(np.int64)
    return valores[posicoes]


def _fold_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Razão das medianas, `nan` onde o denominador é zero ou não finito.

    `inf` numa coluna que entrou finita pareceria medida. O resultado é `nan`, e não o
    numerador, porque razão sem denominador não tem valor de identidade.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        mediana_a = np.median(a, axis=0)
        mediana_b = np.median(b, axis=0)
        utilizavel = np.isfinite(mediana_b) & (mediana_b != 0.0)
        return np.where(utilizavel, mediana_a / np.where(utilizavel, mediana_b, 1.0), np.nan)


FOLD_CHANGE_CONFIDENCE = 0.95
"""Nível do intervalo do fold change. Viaja no resultado e na resposta da rota, porque os
limites sem o nível ao lado não dizem o que cobrem."""


def fold_change_interval(
    matrix: RefinoMatrix,
    groups: tuple[str, str],
    imputed: np.ndarray,
) -> FoldChangeInterval:
    """Intervalo de 95% da razão de medianas, por feature, pelo método de Bonett-Price.

    `matrix` é a matriz de onde o fold change sai, a que a rota passa a `univariate_test` como
    `fold_change_matrix` (a pré-transformação). `imputed` marca as células dela cujo valor a
    imputação escolheu (`ProcessedMatrix.imputed_cells`). As linhas de cada grupo saem da mesma
    função que `univariate_test` usa, então o centro do intervalo é o ponto desenhado.

    O fold change aqui é razão de medianas (`_fold_change`). O intervalo de Welch em log é
    centrado na razão de médias geométricas, que só coincide com a de medianas quando os dois
    grupos são lognormais da mesma forma. O de Bonett-Price é centrado em `ln(m̃₁/m̃₂)`, tem
    forma fechada, não usa semente e corre sobre todas as features de uma vez.

    A fórmula é a de `statpsych::ci.ratio.median2` 2.0.0. Para cada grupo, com os valores
    ordenados `y(1) … y(n)`:

        a   = max(round(n/2 - √n), 1)
        SE  = [ln y(n-a+1) - ln y(a)] / (2 z),   z = Φ⁻¹(1 - P[Bin(n, ½) ≤ a - 1])
        IC  = exp[ ln(m̃₁/m̃₂) ± z₀,₉₇₅ · √(SE₁² + SE₂²) ]

    O `round` do R e o do `numpy` arredondam metade para par, e a convenção não muda `a`: a
    metade exata só aparece com `n` quadrado perfeito ímpar (9, 25, 49…), e nesses casos metade
    para par e metade para cima dão o mesmo número. O exemplo do manual do pacote é reproduzido
    a seis casas em `tests/unit/test_fold_change_interval.py`.

    Duas diferenças do `statpsych`. Ele descarta `NA` antes de calcular; aqui uma célula não
    finita num dos grupos torna o intervalo indefinido. E ele aceita zero e devolve os limites
    `0` e `Inf`; aqui um valor de ordem menor ou igual a zero torna o intervalo indefinido,
    porque `inf` se lê como medida.

    O intervalo também é indefinido quando metade ou mais de um grupo é imputada. Com
    `one_fifth_min` os imputados de uma feature são iguais entre si e os menores do grupo, e a
    mediana sobre eles daria um intervalo estreito em volta de um número que a imputação
    escolheu. O manual do `statpsych` avisa que supõe empates raros dentro de cada grupo. Ver
    `FoldChangeIntervalUndefined`.

    O nível vale para cada feature olhada sozinha, e não para as escolhidas por `p` ou por fold
    change. O `p` sai de outra matriz, a processada, por Welch ou Mann-Whitney, então os dois
    podem discordar: o intervalo exclui o 1 e o `q` fica acima do corte.

    É função à parte de `univariate_test` porque o multiverso chama aquela sem
    `fold_change_matrix` e persiste `sign_consistency` construída com ela.
    """
    valores = np.asarray(matrix.values, dtype=np.float64)
    mascara = np.asarray(imputed, dtype=bool)
    if mascara.shape != valores.shape:
        raise ValueError(
            f"a máscara de imputadas tem forma {mascara.shape} e a matriz {valores.shape}: ela "
            "marca as células desta matriz uma a uma, e com outra forma marcaria outras"
        )
    linhas_a, linhas_b = _linhas_dos_grupos(matrix, groups)
    a, b = valores[linhas_a], valores[linhas_b]

    erro_a, ordem_a = _erro_do_log_da_mediana(a)
    erro_b, ordem_b = _erro_do_log_da_mediana(b)
    motivos = _motivos(
        nao_finito=~(np.isfinite(a).all(axis=0) & np.isfinite(b).all(axis=0)),
        ordem_nao_positiva=~((ordem_a > 0.0) & (ordem_b > 0.0)),
        # Metade ou mais: com `n` par, metade já põe um imputado entre os dois do meio.
        mediana_imputada=(2 * mascara[linhas_a].sum(axis=0) >= a.shape[0])
        | (2 * mascara[linhas_b].sum(axis=0) >= b.shape[0]),
    )
    definido = np.array([motivo is None for motivo in motivos], dtype=bool)

    z = float(norm.ppf(1.0 - (1.0 - FOLD_CHANGE_CONFIDENCE) / 2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        centro = np.log(np.where(definido, _fold_change(a, b), 1.0))
        meia = z * np.hypot(erro_a, erro_b)
        inferior = np.where(definido, centro - meia, np.nan)
        superior = np.where(definido, centro + meia, np.nan)
    return FoldChangeInterval(
        feature=matrix.feature_names,
        lower=np.exp(inferior),
        upper=np.exp(superior),
        log2_lower=inferior / np.log(2.0),
        log2_upper=superior / np.log(2.0),
        undefined=motivos,
        confidence=FOLD_CHANGE_CONFIDENCE,
        groups=groups,
        provenance=matrix.provenance,
    )


def _erro_do_log_da_mediana(bloco: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """O erro-padrão de `ln(mediana)` de cada coluna e o valor de ordem `y(a)` de que ele depende.

    `a` e `z` dependem só de `n`, que é o mesmo em todas as colunas do bloco. Por isso a fórmula
    corre sobre todas as features de uma vez, com uma ordenação por coluna.
    """
    n = bloco.shape[0]
    a = max(int(np.round(n / 2.0 - np.sqrt(n))), 1)
    ordenado = np.sort(bloco, axis=0)
    inferior, superior = ordenado[a - 1], ordenado[n - a]
    z = float(norm.ppf(1.0 - binom.cdf(a - 1, n, 0.5)))
    with np.errstate(divide="ignore", invalid="ignore"):
        erro = (np.log(superior) - np.log(inferior)) / (2.0 * z)
    return erro, inferior


def _motivos(
    *, nao_finito: np.ndarray, ordem_nao_positiva: np.ndarray, mediana_imputada: np.ndarray
) -> tuple[FoldChangeIntervalUndefined | None, ...]:
    """O primeiro motivo de cada feature, na ordem de `FoldChangeIntervalUndefined`.

    Uma feature pode cair em mais de um motivo (imputados iguais a zero, por exemplo) e recebe
    um só, para a contagem por motivo ser uma partição.
    """
    motivos: list[FoldChangeIntervalUndefined | None] = []
    for sem_valor, sem_log, sobre_imputado in zip(
        nao_finito.tolist(), ordem_nao_positiva.tolist(), mediana_imputada.tolist(), strict=True
    ):
        if sem_valor:
            motivos.append("non_finite_value")
        elif sem_log:
            motivos.append("order_statistic_not_positive")
        elif sobre_imputado:
            motivos.append("median_over_imputed_values")
        else:
            motivos.append(None)
    return tuple(motivos)

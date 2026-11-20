"""Gerador de matrizes sintéticas com verdade conhecida.

A base é reamostragem dos resíduos de uma matriz real, e não um modelo paramétrico:
cada linha sintética é o vetor de resíduos de uma amostra real, o que preserva a
correlação entre features. As linhas são sorteadas em ordem aleatória, o que desfaz a
deriva que a matriz real contém. O defeito é então injetado com magnitude conhecida,
e a verdade é a matriz antes da injeção. O custo é que as sintéticas derivam dos
dados reais, e o que for calibrado nelas generaliza menos. `generate` recebe a matriz
real como argumento porque este pacote não lê arquivo.

A ordem dos passos importa: o efeito biológico entra antes dos QCs, para que o pool
o contenha; a deriva entra depois dos QCs, para que eles a sofram; os zeros entram por
último, para que a deriva não multiplique zeros. O QC é uma réplica do pool com ruído
técnico por injeção, e a magnitude da deriva é uma razão sobre a variação de fundo.
"""

import numpy as np

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata, SampleType
from refino.domain.synthetic import SyntheticDataset, SyntheticSpec

__all__ = ["PERFIL_REAL", "generate"]

# Propriedades medidas nos quatro arquivos de referência, por modo. São o alvo contra
# o qual os testes comparam as sintéticas; o gerador não as usa no cálculo.
#
#   rsd_qc_mediano          RSD mediano dos QCs, variante `outlier_removed`
#   dispersao_qc_amostra    `qc_dispersion_ratio` com pré-transformação log10p1
#   fracao_de_zeros         células nulas sobre o total
#   correlacao_mediana      |r| mediano par a par entre features, sobre 400 sorteadas
PERFIL_REAL: dict[str, dict[str, float]] = {
    "pos_hilic": {
        "rsd_qc_mediano": 1.4381,
        "dispersao_qc_amostra": 0.9302,
        "fracao_de_zeros": 0.1579,
        "correlacao_mediana": 0.2936,
    },
    "pos_c18": {
        "rsd_qc_mediano": 0.2415,
        "dispersao_qc_amostra": 0.7957,
        "fracao_de_zeros": 0.0339,
        "correlacao_mediana": 0.3779,
    },
    "neg_hilic": {
        "rsd_qc_mediano": 0.6718,
        "dispersao_qc_amostra": 1.1728,
        "fracao_de_zeros": 0.2538,
        "correlacao_mediana": 0.1771,
    },
    "neg_c18": {
        "rsd_qc_mediano": 0.1550,
        "dispersao_qc_amostra": 0.2609,
        "fracao_de_zeros": 0.0879,
        "correlacao_mediana": 0.1422,
    },
}


def generate(spec: SyntheticSpec, source: RefinoMatrix) -> SyntheticDataset:
    """Gera a matriz sintética e a verdade que a acompanha.

    `source` é a matriz real do modo imitado; a base sai da reamostragem dos resíduos
    dela. A razão deriva/ruído devolvida em `effective_drift_to_noise` é medida no
    resultado, e não a pedida.
    """
    if source.mode != spec.imitated_mode:
        raise ValueError(
            f"a fonte é do modo {source.mode!r} e a especificação imita {spec.imitated_mode!r}: "
            "reamostrar resíduos de um modo e rotular como outro produziria matriz cuja estrutura "
            "não é a do modo que ela declara imitar"
        )

    if spec.n_features > source.n_features:
        # Os resíduos são sorteados sem reposição entre as colunas da fonte; pedir mais
        # features do que ela tem falharia adiante com `IndexError` sem explicação.
        raise ValueError(
            f"a especificação pede {spec.n_features} features e a fonte tem "
            f"{source.n_features}: os resíduos são sorteados sem reposição entre as features da "
            "fonte, e não há como reamostrar mais do que ela tem"
        )
    gerador = np.random.default_rng(spec.seed)
    posicoes_qc = _posicoes_de_qc(spec)
    n_amostras_bio = spec.n_samples - len(posicoes_qc)

    niveis, residuos, residuos_qc, colunas, ausente = _residuos_reais(
        source, spec.n_features, gerador
    )
    base, mascara_bio = _matriz_base(spec, gerador, n_amostras_bio, residuos, niveis, ausente)
    blocos = _blocos_por_correlacao(residuos, spec.n_feature_blocks, gerador)
    grupos = _grupos_randomizados(spec, gerador, n_amostras_bio)
    base, diferenciais = _efeito_biologico(spec, gerador, base, grupos)

    valores, tipos, rotulos = _intercala_qc(spec, gerador, base, grupos, posicoes_qc, residuos_qc)
    verdade = valores.copy()

    deriva = _fator_de_deriva(spec, gerador, valores, blocos, None)
    valores = valores * deriva
    razao_efetiva = _razao_efetiva(verdade, deriva)

    valores = _introduz_ausencia(spec, gerador, valores, mascara_bio, posicoes_qc)

    amostras = tuple(
        SampleMetadata(
            name=f"{'QC' if tipo == 'qc' else 'S'}{indice + 1:03d}",
            injection_order=indice + 1,
            sample_type=tipo,
            group=rotulo,
            batch="A",
        )
        for indice, (tipo, rotulo) in enumerate(zip(tipos, rotulos, strict=True))
    )
    matriz = RefinoMatrix(
        values=valores,
        feature_names=tuple(source.feature_names[j] for j in colunas),
        samples=amostras,
        mode=f"sintetico_{spec.imitated_mode}",
    )
    return SyntheticDataset(
        matrix=matriz,
        ground_truth=verdade,
        true_differential=tuple(source.feature_names[colunas[j]] for j in diferenciais),
        applied_drift=deriva,
        effective_drift_to_noise=razao_efetiva,
        spec=spec,
    )


def _posicoes_de_qc(spec: SyntheticSpec) -> tuple[int, ...]:
    """Índices de injeção dos QCs: 0, `qc_every`, `2·qc_every`, ... e a última injeção.

    A última posição periódica é substituída pela última injeção, e não acrescida de
    mais uma. Nos arquivos de referência os QCs estão nas injeções 1, 12, 23, 34, 45,
    56 e 68: sete QCs, com o último intervalo mais longo, e a corrida termina em QC,
    que ancora a deriva do fim. Com `n_samples=68` e `qc_every=11` isso dá sete QCs;
    acrescentar daria oito, com dois QCs adjacentes no fim.

    Quando há uma única posição periódica, a última injeção é acrescentada, para a
    corrida não ficar sem QC no início.
    """
    posicoes = list(range(0, spec.n_samples, spec.qc_every))
    if posicoes[-1] != spec.n_samples - 1:
        if len(posicoes) > 1:
            posicoes[-1] = spec.n_samples - 1
        else:
            posicoes.append(spec.n_samples - 1)
    return tuple(posicoes)


def _residuos_reais(
    fonte: RefinoMatrix, n_features: int, gerador: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Níveis, resíduos de amostra e de QC, colunas escolhidas e a máscara de zeros da fonte.

    Trabalha em `log10` das células não nulas: zero é ausência de detecção, e entrar
    como intensidade contaminaria o nível da feature. Célula nula vira `nan` no resíduo
    e é preenchida por sorteio entre os resíduos definidos da mesma feature; os zeros
    voltam no fim, pela máscara.

    Os resíduos de QC saem dos QCs reais, em torno do nível da feature: são ruído
    técnico medido.
    """
    valores = np.asarray(fonte.values, dtype=np.float64)
    colunas = np.sort(
        gerador.choice(valores.shape[1], size=min(n_features, valores.shape[1]), replace=False)
    )
    recorte = valores[:, colunas]

    with np.errstate(divide="ignore"):
        log = np.where(recorte > 0, np.log10(np.maximum(recorte, 1e-12)), np.nan)

    niveis = np.nanmean(log, axis=0)
    niveis = np.where(np.isfinite(niveis), niveis, 0.0)
    residuo = log - niveis

    def _preenche(bloco: np.ndarray) -> np.ndarray:
        saida = bloco.copy()
        for coluna in range(saida.shape[1]):
            faltando = ~np.isfinite(saida[:, coluna])
            if not faltando.any():
                continue
            definidos = saida[~faltando, coluna]
            saida[faltando, coluna] = (
                gerador.choice(definidos, size=int(faltando.sum())) if definidos.size else 0.0
            )
        return saida

    # A máscara de zeros sai junto com o resíduo. O padrão real de zeros é por linha
    # (amostra de baixa abundância tem muitos), e uma regra global que zerasse os
    # menores valores apagaria features inteiras e reduziria a correlação entre colunas.
    ausente = ~np.isfinite(residuo)
    return (
        niveis,
        _preenche(residuo[fonte.sample_mask]),
        _preenche(residuo[fonte.qc_mask]),
        colunas,
        ausente[fonte.sample_mask],
    )


def _matriz_base(
    spec: SyntheticSpec,
    gerador: np.random.Generator,
    n_linhas: int,
    residuos: np.ndarray,
    niveis: np.ndarray,
    ausente: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Matriz base por reamostragem: cada linha sintética é um vetor de resíduo real inteiro.

    O vetor vem inteiro para que a covariância entre colunas seja a do dado real;
    sortear célula a célula a destruiria. A ordem do sorteio é aleatória, o que desfaz
    a deriva que a fonte contém.
    """
    escolhidas = gerador.choice(residuos.shape[0], size=n_linhas, replace=True)
    # A máscara vem do mesmo índice sorteado, para ficar coerente com o resíduo.
    return 10.0 ** (niveis + residuos[escolhidas]), ausente[escolhidas]


def _grupos_randomizados(
    spec: SyntheticSpec, gerador: np.random.Generator, n_linhas: int
) -> np.ndarray:
    """Grupos distribuídos aleatoriamente na ordem de injeção, nunca em blocos.

    Em blocos, o efeito biológico ficaria confundido com a deriva, e qualquer corretor
    de deriva removeria o sinal biológico junto.
    """
    rotulos = np.array([f"G{indice % spec.n_groups + 1}" for indice in range(n_linhas)])
    return gerador.permutation(rotulos)


def _efeito_biologico(
    spec: SyntheticSpec, gerador: np.random.Generator, base: np.ndarray, grupos: np.ndarray
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Deslocamento em uma fração das features, para as amostras do grupo `G1`.

    É fração e não contagem para que o total de features não mude a proporção de
    sinal. Devolve índices de coluna; `generate` os traduz para os nomes reais das
    features, que são os que a matriz carrega.
    """
    n_diferenciais = max(1, round(spec.differential_fraction * spec.n_features))
    indices = np.sort(gerador.choice(spec.n_features, size=n_diferenciais, replace=False))
    afetados = grupos == "G1"

    valores = base.copy()
    # Deslocamento multiplicativo de `effect_size` desvios do `log10` da própria
    # feature: em unidades do desvio, como a deriva, e não absoluto.
    desvio_log = np.log10(np.maximum(valores[:, indices], 1e-12)).std(axis=0)
    valores[np.ix_(afetados, indices)] *= 10.0 ** (spec.effect_size * desvio_log)
    return valores, tuple(int(j) for j in indices)


def _intercala_qc(
    spec: SyntheticSpec,
    gerador: np.random.Generator,
    base: np.ndarray,
    grupos: np.ndarray,
    posicoes_qc: tuple[int, ...],
    residuos_qc: np.ndarray,
) -> tuple[np.ndarray, list[SampleType], list[str]]:
    """Intercala os QCs: cada um é o pool das amostras com ruído técnico de uma injeção real.

    O pool é a média das amostras, que é o que o QC é fisicamente. Cada injeção de QC
    recebe um vetor de resíduos sorteado entre os QCs reais da fonte. Sem esse ruído os
    QCs seriam idênticos, e é a variação entre eles que o RSD de QC mede.
    """
    pool = base.mean(axis=0)

    valores = np.empty((spec.n_samples, spec.n_features), dtype=np.float64)
    tipos: list[SampleType] = []
    rotulos: list[str] = []
    proxima_bio = 0
    for indice in range(spec.n_samples):
        if indice in posicoes_qc:
            escolhido = gerador.integers(0, residuos_qc.shape[0])
            valores[indice] = pool * 10.0 ** residuos_qc[escolhido]
            tipos.append("qc")
            rotulos.append("QC")
        else:
            valores[indice] = base[proxima_bio]
            tipos.append("sample")
            rotulos.append(str(grupos[proxima_bio]))
            proxima_bio += 1
    return valores, tipos, rotulos


def _blocos_por_correlacao(
    residuos: np.ndarray, n_blocos: int, gerador: np.random.Generator
) -> np.ndarray:
    """Bloco de cada feature, sorteado, para que a deriva seja correlacionada dentro dele.

    A atribuição é aleatória e não por agrupamento dos resíduos: agrupar por correlação
    real faria a deriva injetada seguir a estrutura biológica do dado, e ela deixaria
    de ser separável dela.
    """
    return np.asarray(gerador.integers(0, n_blocos, size=residuos.shape[1]))


def _fator_de_deriva(
    spec: SyntheticSpec,
    gerador: np.random.Generator,
    valores: np.ndarray,
    blocos: np.ndarray,
    _perfil: dict[str, float] | None,
) -> np.ndarray:
    """Fator multiplicativo de deriva por (injeção, feature), conforme o tipo de defeito.

    A amplitude é `drift_to_noise_ratio` vezes o desvio do `log10` de cada feature na
    matriz sem deriva, que é a variação de fundo contra a qual a deriva compete. Um
    número absoluto não diria se a deriva é grande ou pequena.

    O sinal é comum às features de um bloco: deriva instrumental afeta features
    coeluentes juntas.
    """
    n_injecoes = valores.shape[0]
    if spec.defect_type == "clean":
        return np.ones_like(valores)

    residuo = np.log10(np.maximum(valores, 1e-12)).std(axis=0)
    amplitude = spec.drift_to_noise_ratio * residuo

    perfil_temporal = _perfil_temporal(spec, gerador, n_injecoes)
    # Sinal por bloco: parte das features deriva para cima e parte para baixo. Deriva no
    # mesmo sentido em todas seria removível por normalização por soma.
    sinal = gerador.choice((-1.0, 1.0), size=spec.n_feature_blocks)[blocos]
    return 10.0 ** (perfil_temporal[:, None] * amplitude[None, :] * sinal[None, :])


def _perfil_temporal(
    spec: SyntheticSpec, gerador: np.random.Generator, n_injecoes: int
) -> np.ndarray:
    """A forma da deriva ao longo da ordem de injeção, normalizada a média 0 e desvio 1.

    A normalização faz `drift_to_noise_ratio` significar a mesma coisa nos três tipos:
    rampa em `gradual_drift`, degrau em `step_change` e passeio aleatório no caso
    restante.
    """
    if spec.defect_type == "gradual_drift":
        forma = np.linspace(0.0, 1.0, n_injecoes)
    elif spec.defect_type == "step_change":
        posicao = spec.changepoint_position
        assert posicao is not None  # garantido pelo __post_init__
        corte = round(posicao * n_injecoes)
        forma = np.concatenate([np.zeros(corte), np.ones(n_injecoes - corte)])
    else:
        passos = gerador.normal(0.0, 1.0, size=n_injecoes)
        forma = np.cumsum(passos)

    desvio = forma.std()
    return (forma - forma.mean()) / desvio if desvio > 0 else forma


def _razao_efetiva(verdade: np.ndarray, deriva: np.ndarray) -> float:
    """Razão deriva/ruído medida no resultado, em mediana entre features.

    Por feature, é o desvio do `log10` da deriva sobre o desvio do `log10` da verdade.
    A razão pedida é parâmetro; esta é a que a matriz gerada tem.
    """
    residuo = np.log10(np.maximum(verdade, 1e-12)).std(axis=0)
    amplitude = np.log10(np.maximum(deriva, 1e-12)).std(axis=0)
    utilizavel = residuo > 0
    if not utilizavel.any():
        return float("nan")
    return float(np.median(amplitude[utilizavel] / residuo[utilizavel]))


def _introduz_ausencia(
    spec: SyntheticSpec,
    gerador: np.random.Generator,
    valores: np.ndarray,
    mascara_bio: np.ndarray,
    posicoes_qc: tuple[int, ...],
) -> np.ndarray:
    """Aplica os zeros: o padrão de ausência reamostrado da fonte, junto com o resíduo.

    O padrão real de zeros é por linha: amostra de baixa abundância tem muitos zeros
    espalhados, e não features inteiras ausentes. Reamostrar a máscara do mesmo índice
    do resíduo reproduz isso. Por consequência, `zero_fraction` e `mnar_fraction` da
    especificação não controlam a fração de zeros; ela é a do modo imitado.

    As linhas de QC não recebem máscara de amostra: o QC é a mesma mistura em toda
    injeção, e só tem zero onde a feature é zero em todas as amostras.
    """
    saida = valores.copy()
    indice_bio = 0
    for indice in range(saida.shape[0]):
        if indice in posicoes_qc:
            continue
        saida[indice, mascara_bio[indice_bio]] = 0.0
        indice_bio += 1

    # O QC fica com zero só onde a coluna é nula em todas as amostras, que é o que
    # sobrevive à média do pool.
    nunca_detectada = mascara_bio.all(axis=0)
    for indice in posicoes_qc:
        saida[indice, nunca_detectada] = 0.0
    return saida

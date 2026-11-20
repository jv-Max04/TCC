"""As três operações de normalização: por amostra, transformação e escalonamento.

Toda operação aqui divide por algo derivado do dado (soma da linha, desvio da
coluna, amplitude, quociente de PQN), e o denominador pode ser zero ou não
finito. Quando é, a operação deixa aquela linha ou coluna como está, e a
contagem vai no relatório. Dividir produziria `inf` numa matriz que entrou
finita, e nenhuma normalização pode produzir `NaN` ou `Inf` a partir de matriz
que não os tem.

As três não comutam, e por isso a ordem é fixa: normalização por amostra,
transformação, escalonamento. `tests/unit/test_normalization_properties.py`
mostra que trocar a ordem muda o resultado.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from refino.domain.labels import SampleNormalization, Scaling, Transformation, validate_label
from refino.domain.matrix import RefinoMatrix

__all__ = [
    "NormalizacaoDeclarada",
    "normalize_samples",
    "normalize_samples_declarada",
    "scale_features",
    "transform",
]


# A constante para a qual `sum` normaliza. É 1000 porque é o que o `SumNorm` do
# MetaboAnalyst usa (`R/general_norm_utils.R`: `1000*x/sum(x)`).
SOMA_CONSTANTE = 1000.0


@dataclass
class _LinhasIntactas:
    """Acumula as linhas que `_seguro` deixou como estavam.

    Uma linha com denominador zero ou não finito sai da normalização como entrou.
    Sem registro ela seria lida como 'não foi normalizada', quando o certo é 'não
    tinha por onde ser normalizada': a primeira leitura é defeito do pipeline, a
    segunda é propriedade do dado.

    Acumula em vez de levantar, porque a política é deixar a linha como está. O
    que se acrescenta é a declaração de quais linhas foram.
    """

    mascaras: list[np.ndarray] = field(default_factory=list)

    def anota(self, degenerado_por_linha: np.ndarray) -> None:
        self.mascaras.append(np.asarray(degenerado_por_linha, dtype=bool).ravel())

    def nomes(self, matrix: RefinoMatrix) -> tuple[str, ...]:
        """Os nomes das amostras deixadas intactas.

        Nomes, e não só a contagem: o operador precisa do nome para olhar a injeção.
        """
        if not self.mascaras:
            return ()
        junto = np.zeros(matrix.n_samples, dtype=bool)
        for mascara in self.mascaras:
            junto |= mascara
        return tuple(
            nome for nome, ruim in zip(matrix.sample_names, junto, strict=True) if bool(ruim)
        )


def _utilizavel(denominador: np.ndarray) -> np.ndarray:
    """Máscara dos denominadores utilizáveis: finitos e não nulos."""
    # Anotação local, e não `cast`: o `numpy` devolve `Any`, e o modo estrito recusa.
    mascara: np.ndarray = np.isfinite(denominador) & (denominador != 0.0)
    return mascara


def _seguro(
    numerador: np.ndarray,
    denominador: np.ndarray,
    *,
    registro: _LinhasIntactas | None = None,
) -> np.ndarray:
    """Divide, deixando o numerador intacto onde o denominador é zero ou não finito.

    É a política de denominador degenerado num lugar só, para que nenhuma das
    opções de normalização propague `inf`.

    `registro` não muda valor nenhum e só vale para denominador por linha, a
    forma que `normalize_samples` usa. O escalonamento divide por coluna, e ali o
    numerador é o valor centrado: desvio zero implica coluna constante e centrado
    zero, então a coluna degenerada sai zerada.
    """
    utilizavel = _utilizavel(denominador)
    if registro is not None:
        if denominador.shape != (numerador.shape[0], 1):
            raise AssertionError(
                f"`registro` conta LINHAS e o denominador tem forma {denominador.shape} para um "
                f"numerador {numerador.shape}. Só denominador por linha — `(n, 1)` — produz "
                "contagem de amostras; noutra forma o número sairia sobre o eixo errado."
            )
        registro.anota(~utilizavel)
    return np.where(utilizavel, numerador / np.where(utilizavel, denominador, 1.0), numerador)


@dataclass(frozen=True)
class NormalizacaoDeclarada:
    """A matriz normalizada e as amostras que a normalização deixou como estavam.

    `normalize_samples` devolve só a `RefinoMatrix`. Quem monta o `StageResult`
    precisa também dos nomes, e este par os leva sem uma segunda implementação
    dos denominadores.

    `linhas_sem_denominador` vazio é o caso normal: todas as linhas foram
    normalizadas.
    """

    matrix: RefinoMatrix
    linhas_sem_denominador: tuple[str, ...]


def normalize_samples(
    matrix: RefinoMatrix,
    method: SampleNormalization,
    *,
    reference: str | None = None,
    reference_present_at_entry: bool | None = None,
    factors: tuple[tuple[str, float], ...] | None = None,
) -> RefinoMatrix:
    """Normalização por amostra, devolvendo só a matriz.

    Chama `normalize_samples_declarada`. As linhas deixadas intactas ficam
    registradas no `provenance` da matriz devolvida.
    """
    return normalize_samples_declarada(
        matrix,
        method,
        reference=reference,
        reference_present_at_entry=reference_present_at_entry,
        factors=factors,
    ).matrix


def normalize_samples_declarada(
    matrix: RefinoMatrix,
    method: SampleNormalization,
    *,
    reference: str | None = None,
    reference_present_at_entry: bool | None = None,
    factors: tuple[tuple[str, float], ...] | None = None,
) -> NormalizacaoDeclarada:
    """Normalização por amostra: opera nas linhas.

    O que divide a linha em cada método:

    - `none`: nada; identidade exata;
    - `sum`: a soma total da linha, e multiplica por `SOMA_CONSTANTE`;
    - `median`: a mediana da linha;
    - `pqn_group`: o quociente probabilístico contra o espectro de referência dos
      QCs;
    - `sample_reference`: o quociente probabilístico contra a amostra nomeada;
    - `feature_reference`: o valor da linha na feature nomeada;
    - `supplied_factors`: o fator fornecido para a amostra;
    - `quantile`: não divide; substitui cada valor pela média dos valores de mesmo
      posto.

    `pqn_group` usa como referência a mediana das amostras de QC por feature, e é
    o `GroupPQN` do MetaboAnalyst com o grupo fixo nos QCs. Com poucos QCs a
    referência é fina.

    `reference` é o nome que o usuário escolheu: uma amostra ou uma feature,
    conforme o método.

    `reference_present_at_entry` diz se o nome estava na matriz ingerida. Esta
    função recebe a matriz depois dos filtros, e daqui não há como separar 'o nome
    nunca existiu' de 'estava e um filtro o removeu', que pedem mensagens
    diferentes. `process_matrix` decide antes do primeiro filtro e passa o
    veredito; `None` significa que ninguém conferiu.
    """
    validate_label("SampleNormalization", method)
    if method == "none":
        return NormalizacaoDeclarada(matrix.with_provenance("normalize_samples(none)"), ())

    valores = np.asarray(matrix.values, dtype=np.float64)
    registro = _LinhasIntactas()

    if method == "sum":
        # A constante entra no denominador, e não multiplica o resultado, por
        # causa da linha degenerada: `_seguro` deixa o numerador intacto onde a
        # soma é zero, e multiplicar depois o ampliaria 1000 vezes.
        #
        # O fator 1000 some sob `auto`, `range` e `vast`, e sobrevive sob `none`,
        # `mean_center` e `pareto`, que divide pela raiz do desvio e deixa
        # `sqrt(c)`.
        soma = valores.sum(axis=1, keepdims=True) / SOMA_CONSTANTE
        novos = _seguro(valores, soma, registro=registro)
    elif method == "median":
        novos = _seguro(valores, np.median(valores, axis=1, keepdims=True), registro=registro)
    elif method == "pqn_group":
        novos = _pqn(valores, matrix, registro=registro)
    elif method == "sample_reference":
        novos = _por_amostra_de_referencia(valores, matrix, reference, registro=registro)
    elif method == "feature_reference":
        # Sem `registro`: este ramo recusa o denominador degenerado (zero,
        # negativo ou não finito), então não há linha intacta a declarar.
        novos = _por_feature_de_referencia(valores, matrix, reference, reference_present_at_entry)
    elif method == "supplied_factors":
        novos = _por_fatores_fornecidos(valores, matrix, factors, registro=registro)
    elif method == "quantile":
        # `quantile` não divide por nada, então não há linha a declarar.
        novos = _quantile(valores)
    else:  # pragma: no cover - inalcançável: `validate_label` fecha o vocabulário
        raise AssertionError(
            f"normalizacao por amostra {method!r} esta no vocabulario e nao tem ramo aqui. "
            "Valor novo no `Literal` exige ramo proprio: sem ele cairia noutra conta."
        )

    _exige_saida_finita(matrix.values, novos, method)
    intactas = registro.nomes(matrix)
    # A declaração vai também no `provenance`, porque quem chama
    # `normalize_samples` fora do pipeline não monta `StageResult` e só leva a
    # matriz. O prefixo antes do parêntese fica `normalize_samples`, que é o que o
    # teste da ordem canônica lê.
    nota = f"normalize_samples({method})"
    if intactas:
        nota = (
            f"normalize_samples({method}; {len(intactas)} de {matrix.n_samples} linha(s) sem "
            f"denominador, deixadas como estavam: {', '.join(intactas[:5])}"
            f"{'…' if len(intactas) > 5 else ''})"
        )
    return NormalizacaoDeclarada(matrix.with_values(novos, nota), intactas)


class NormalizacaoInvalidaError(ValueError):
    """Base das recusas desta camada, e é ela que as fronteiras traduzem.

    `rotas.py` e `cli.py` apanham esta classe, então uma subclasse nova chega
    traduzida sem mexer neles: a rota devolve `409`, e a linha de comando, o
    código de indisponibilidade. `tests/test_leitura_de_fatores.py` confere.
    """


class ReferenciaAusenteError(NormalizacaoInvalidaError):
    """A referência nomeada não está na matriz, ou não serve como denominador.

    Classe própria para que as camadas de cima a traduzam pelo tipo, e não
    casando o texto da mensagem.
    """


class NormalizacaoNaoFinitaError(NormalizacaoInvalidaError):
    """A conta produziu `inf` ou `NaN` a partir de matriz que não os tinha."""


def _exige_saida_finita(antes: np.ndarray, depois: np.ndarray, method: SampleNormalization) -> None:
    """Pós-condição: a normalização não produz não finito a partir de matriz finita.

    `_seguro` protege contra denominador zero ou não finito, mas não contra
    transbordo: um fator positivo, finito e subnormal, como `5e-324`, passa por
    toda guarda de valor, e a divisão devolve `inf`. Conferir a saída cobre todos
    os ramos, inclusive os que vierem depois.

    Matriz que já chega com `NaN` não é recusada aqui: a política de ausência
    pertence a `impute`.
    """
    if not np.isfinite(antes).all():
        return
    ruins = ~np.isfinite(depois)
    if not ruins.any():
        return
    raise NormalizacaoNaoFinitaError(
        f"a normalização {method!r} produziu {int(ruins.sum())} valor(es) não finito(s) a "
        f"partir de uma matriz que não tinha nenhum. Isto é proibido. A causa provável "
        f"é um denominador de magnitude extrema — um fator subnormal, ou uma feature de "
        f"referência quase nula — que passa por ser positivo e finito e ainda assim transborda."
    )


def _por_amostra_de_referencia(
    valores: np.ndarray,
    matrix: RefinoMatrix,
    reference: str | None,
    *,
    registro: _LinhasIntactas | None = None,
) -> np.ndarray:
    """Quociente probabilístico contra a linha da amostra nomeada, o `SamplePQN` do MetaboAnalyst.

    É a mesma conta de `pqn_group`: calcula a razão por feature contra a
    referência e divide a linha pela mediana dessas razões, um escalar. Dividir
    feature a feature seria outro método e apagaria a variação relativa entre
    features. Com exatamente um QC, este método e `pqn_group` têm o mesmo espectro
    de referência e dão a mesma matriz.

    A conta é delegada a `_pqn_contra`, o mesmo corpo que `pqn_group` usa.
    """
    if reference is None:
        raise ReferenciaAusenteError(
            "`sample_normalization='sample_reference'` exige `normalization_reference` com o "
            "nome da amostra. Sem ele não há denominador, e devolver a matriz intacta seria "
            "dizer que normalizou."
        )

    nomes = list(matrix.sample_names)
    if reference not in nomes:
        raise ReferenciaAusenteError(
            f"a amostra de referência {reference!r} não está na matriz. As amostras são "
            f"{nomes[:8]}{'…' if len(nomes) > 8 else ''} ({len(nomes)} no total)."
        )

    espectro = valores[nomes.index(reference)]
    utilizavel = np.isfinite(espectro) & (espectro != 0.0)
    if not utilizavel.any():
        # Recusa, e não identidade. `pqn_group` devolve a matriz intacta neste
        # caso porque a referência dele é derivada dos QCs. Aqui o usuário nomeou
        # a amostra, e devolvê-la intacta registraria o nome na proveniência sem
        # que nada tivesse acontecido.
        raise ReferenciaAusenteError(
            f"a amostra de referência {reference!r} não tem uma única feature com valor "
            "positivo e finito: não há razão a calcular, e devolver a matriz intacta seria "
            "dizer que normalizou."
        )
    return _pqn_contra(valores, espectro, utilizavel, registro=registro)


def _por_fatores_fornecidos(
    valores: np.ndarray,
    matrix: RefinoMatrix,
    factors: tuple[tuple[str, float], ...] | None,
    *,
    registro: _LinhasIntactas | None = None,
) -> np.ndarray:
    """Divide cada amostra pelo fator fornecido para ela, o `SpecNorm` do MetaboAnalyst.

    É a normalização por peso seco, volume de urina, número de células: uma
    quantidade que o laboratório mediu fora do equipamento e que não está na
    matriz.

    A cobertura tem de ser total. Um mapa que cubra parte das linhas deixaria a
    matriz com dois regimes misturados. Os QCs entram na exigência, mesmo sem
    peso seco: o RSD de QC é a métrica que diz se a correção funcionou, e
    normalizar as amostras sem normalizar os QCs faria a comparação de antes e
    depois deixar de ser sobre a mesma escala. Não há fator neutro automático
    para eles, porque essa escolha é do laboratório.

    A guarda de valor (positivo e finito, sem duplicados) já correu em
    `PipelineSpec`. Aqui só se confere o que depende da matriz.
    """
    if factors is None:
        raise ReferenciaAusenteError(
            "`sample_normalization='supplied_factors'` exige `normalization_factors` com um "
            "fator por amostra. Sem eles não há denominador, e devolver a matriz intacta seria "
            "dizer que normalizou."
        )

    mapa = dict(factors)
    nomes = list(matrix.sample_names)
    faltam = [nome for nome in nomes if nome not in mapa]
    if faltam:
        qcs = {n for n, e_qc in zip(nomes, matrix.qc_mask, strict=True) if e_qc}
        so_qc = set(faltam) <= qcs
        raise ReferenciaAusenteError(
            f"{len(faltam)} de {len(nomes)} amostras não têm fator: "
            f"{faltam[:8]}{'…' if len(faltam) > 8 else ''}."
            + (
                " Todas elas são QCs, e eles contam: o RSD de QC é a régua que diz se a "
                "correção funcionou, e normalizar as amostras sem normalizar os QCs faz a "
                "comparação de antes e depois deixar de ser sobre a mesma escala."
                if so_qc
                else " Normalizar parte das linhas deixaria a matriz com dois regimes "
                "misturados, sem nada a dizê-lo."
            )
        )

    denominador = np.array([mapa[nome] for nome in nomes], dtype=np.float64)
    # `_seguro`, e não divisão crua: um fator zero daria `inf`. O transbordo por
    # fator subnormal é apanhado por `_exige_saida_finita`.
    return _seguro(valores, denominador[:, None], registro=registro)


def _por_feature_de_referencia(
    valores: np.ndarray,
    matrix: RefinoMatrix,
    reference: str | None,
    presente_na_entrada: bool | None,
) -> np.ndarray:
    """Divide cada amostra pelo valor dela numa feature nomeada, o `CompNorm` do MetaboAnalyst.

    É a normalização por padrão interno: escolhe-se uma feature, tipicamente um
    composto adicionado em quantidade conhecida, e cada linha é dividida pelo
    valor que tem ali. O denominador é um escalar por linha, e não a mediana de
    razões como em `sample_reference`.

    Se o nome não está na matriz, a mensagem depende de `presente_na_entrada`, o
    veredito que `process_matrix` decidiu antes do primeiro filtro: o nome não
    existe no arquivo, ou existia e um filtro o descartou. Com `None` a mensagem
    declara que não dá para saber.

    Depois da divisão a feature de referência fica constante em 1. A etapa que
    remove variância quase nula já correu, então ela continua na matriz
    processada.
    """
    if reference is None:
        raise ReferenciaAusenteError(
            "`sample_normalization='feature_reference'` exige `normalization_reference` com o "
            "nome da feature. Sem ele não há denominador, e devolver a matriz intacta seria "
            "dizer que normalizou."
        )

    nomes = list(matrix.feature_names)
    if reference not in nomes:
        if presente_na_entrada is True:
            raise ReferenciaAusenteError(
                f"a feature de referência {reference!r} estava na matriz que você enviou e foi "
                "DESCARTADA por uma das etapas de filtragem antes da normalização. Afrouxe o "
                "filtro que a removeu, ou escolha outra referência."
            )
        if presente_na_entrada is False:
            raise ReferenciaAusenteError(
                f"a feature de referência {reference!r} não existe no arquivo que você enviou. "
                f"O arquivo tem {matrix.n_features} features nesta altura do processamento."
            )
        raise ReferenciaAusenteError(
            f"a feature de referência {reference!r} não está na matriz. Não dá para dizer se ela "
            "nunca existiu ou se um filtro a removeu, porque a presença na entrada não foi "
            "conferida — quem chama esta função por fora do pipeline não a confere."
        )

    coluna = valores[:, nomes.index(reference)]
    # `> 0.0`, e não `!= 0.0`: um valor negativo é finito e não nulo, e dividir
    # por ele inverte o sinal da amostra inteira sem produzir não finito.
    # `PipelineSpec` recusa fator negativo em `normalization_factors` pelo mesmo
    # motivo. O caso é alcançável: `NEGATIVE_INTENSITY` é só aviso na validação,
    # então uma exportação com subtração de linha de base chega até aqui.
    degenerados = ~(np.isfinite(coluna) & (coluna > 0.0))
    if degenerados.any():
        quais = [n for n, ruim in zip(matrix.sample_names, degenerados, strict=True) if ruim]
        raise ReferenciaAusenteError(
            f"a feature de referência {reference!r} tem valor zero, negativo ou não finito em "
            f"{len(quais)} amostra(s) — {quais[:5]}{'…' if len(quais) > 5 else ''}. Zero e não "
            "finito produziriam não finito, que é proibido. Negativo é pior porque não "
            "produz: ele INVERTE o sinal da amostra inteira e sai finito, sem nada a acusar."
        )
    dividido: np.ndarray = valores / coluna[:, None]
    return dividido


def _pqn(
    valores: np.ndarray, matrix: RefinoMatrix, *, registro: _LinhasIntactas | None = None
) -> np.ndarray:
    """Quociente probabilístico contra a mediana dos QCs.

    Para cada amostra: razão por feature contra a referência, e a linha é
    dividida pela mediana das razões definidas.

    Sem QC, ou sem feature utilizável na referência, a operação é identidade, e
    todas as linhas são anotadas no `registro` como deixadas intactas.
    """
    qcs = valores[matrix.qc_mask]
    if qcs.size == 0:
        if registro is not None:
            registro.anota(np.ones(valores.shape[0], dtype=bool))
        return valores
    referencia = np.median(qcs, axis=0)
    utilizavel = np.isfinite(referencia) & (referencia != 0.0)
    if not utilizavel.any():
        if registro is not None:
            registro.anota(np.ones(valores.shape[0], dtype=bool))
        return valores

    return _pqn_contra(valores, referencia, utilizavel, registro=registro)


def _pqn_contra(
    valores: np.ndarray,
    referencia: np.ndarray,
    utilizavel: np.ndarray,
    *,
    registro: _LinhasIntactas | None = None,
) -> np.ndarray:
    """A conta do PQN contra um espectro de referência, usada por `pqn_group` e `sample_reference`.

    Um corpo só, para que os dois métodos deem o mesmo número quando a referência
    é a mesma.

    Razão indefinida (referência zero ou não finita) não entra na mediana:
    incluí-la como `inf` faria a mediana saltar, e como zero a puxaria para baixo.
    """
    razoes = valores[:, utilizavel] / referencia[utilizavel]
    with np.errstate(invalid="ignore"):
        quocientes = np.nanmedian(np.where(np.isfinite(razoes), razoes, np.nan), axis=1)
    return _seguro(valores, quocientes[:, None], registro=registro)


def _quantile(valores: np.ndarray) -> np.ndarray:
    """Normalização por quantis entre amostras.

    Cada linha é ordenada, a média por posto é calculada entre todas as linhas, e
    cada valor é substituído pela média do posto dele. Empates recebem o mesmo
    valor, pela média dos postos empatados, que é onde as implementações
    divergem.
    """
    ordenados = np.sort(valores, axis=1)
    media_por_posto = ordenados.mean(axis=0)

    saida = np.empty_like(valores)
    for linha in range(valores.shape[0]):
        # `average` nos empates: valores iguais recebem o mesmo resultado.
        postos = rankdata(valores[linha], method="average")
        saida[linha] = np.interp(postos, np.arange(1, valores.shape[1] + 1), media_por_posto)
    return saida


def transform(matrix: RefinoMatrix, method: Transformation) -> RefinoMatrix:
    """Transformação célula a célula.

    `log10` e `log2` usam a forma estabilizada que o código R emprega, que evita
    indefinição em zero:

        log((x + sqrt(x² + m²)) / 2),  com m = min(|x| para x ≠ 0) / 10

    Em `x = 0` a forma dá `log(m/2)`, que é finito.

    `sqrt` usa a mesma forma estabilizada, `((x + sqrt(x² + m²))/2)^½`, com o
    mesmo `m`: é o `SrNorm` do MetaboAnalyst, e reusa `_estabiliza_para_log` para
    que haja uma definição só de `m`. `cbrt` trata negativo nativamente e
    coincide com o `CrNorm`.
    """
    validate_label("Transformation", method)
    if method == "none":
        return matrix.with_provenance("transform(none)")

    valores = np.asarray(matrix.values, dtype=np.float64)

    if method in ("log10", "log2"):
        estabilizados = _estabiliza_para_log(valores)
        novos = np.log10(estabilizados) if method == "log10" else np.log2(estabilizados)
    elif method == "sqrt":
        novos = _estabiliza_para_log(valores) ** 0.5
    elif method == "cbrt":
        novos = np.cbrt(valores)
    else:  # pragma: no cover - inalcançável: `validate_label` fecha o vocabulário
        raise AssertionError(
            f"transformacao {method!r} esta no vocabulario e nao tem ramo aqui. "
            "Valor novo no `Literal` exige ramo proprio: sem ele cairia noutra conta."
        )

    return matrix.with_values(novos, f"transform({method})")


def _estabiliza_para_log(valores: np.ndarray) -> np.ndarray:
    """`(x + sqrt(x² + m²)) / 2` com `m` derivado do menor módulo não nulo.

    Matriz inteiramente nula não tem `m`: `m` vira 1, a forma dá 0,5 em toda
    célula, e o log disso é finito. Matriz toda nula é entrada legítima depois de
    um filtro agressivo.
    """
    nao_nulos = np.abs(valores[(valores != 0.0) & np.isfinite(valores)])
    m = float(nao_nulos.min()) / 10.0 if nao_nulos.size else 1.0
    return (valores + np.sqrt(valores**2 + m**2)) / 2.0


def scale_features(matrix: RefinoMatrix, method: Scaling, *, ddof: int = 0) -> RefinoMatrix:
    """Escalonamento por feature: opera nas colunas.

    Todos menos `none` centram a coluna. O que cada um divide:

    - `none`: nada; identidade exata;
    - `mean_center`: nada, só centra;
    - `auto`: o desvio;
    - `pareto`: a raiz do desvio;
    - `range`: a amplitude, `max - min`;
    - `vast`: o desvio, e multiplica pela razão `média / desvio`.

    `ddof` é explícito e o padrão é 0. A diferença entre `n` e `n - 1` é pequena
    em matriz grande e visível em fixture pequeno.

    `vast` é escalonamento de estabilidade de variável: feature estável, com
    média alta em relação ao desvio, ganha peso.
    """
    validate_label("Scaling", method)
    if method == "none":
        return matrix.with_provenance("scale_features(none)")

    valores = np.asarray(matrix.values, dtype=np.float64)
    medias = valores.mean(axis=0, keepdims=True)
    centrados = valores - medias

    if method == "mean_center":
        novos = centrados
    elif method == "auto":
        novos = _seguro(centrados, valores.std(axis=0, ddof=ddof, keepdims=True))
    elif method == "pareto":
        novos = _seguro(centrados, np.sqrt(valores.std(axis=0, ddof=ddof, keepdims=True)))
    elif method == "range":
        amplitude = valores.max(axis=0, keepdims=True) - valores.min(axis=0, keepdims=True)
        novos = _seguro(centrados, amplitude)
    elif method == "vast":
        desvios = valores.std(axis=0, ddof=ddof, keepdims=True)
        auto = _seguro(centrados, desvios)
        novos = auto * _seguro(medias, desvios)
    else:  # pragma: no cover - inalcançável: `validate_label` fecha o vocabulário
        raise AssertionError(
            f"escalonamento {method!r} esta no vocabulario e nao tem ramo aqui. "
            "Valor novo no `Literal` exige ramo proprio: sem ele cairia noutra conta."
        )

    return matrix.with_values(novos, f"scale_features({method},ddof={ddof})")

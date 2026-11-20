"""Normalidade por variável e por grupo: Shapiro-Wilk depois da correção e da normalização.

É diagnóstico, e não decisão: um Shapiro-Wilk por variável dentro de cada grupo e um resumo por
grupo que diz quantas variáveis rejeitam a normal, como apoio à escolha dos procedimentos
estatísticos seguintes. O teste do contraste continua a ser escolhido pela decisão única de
`statistics/univariate.py`, um Shapiro-Wilk sobre os resíduos de todas as features juntos, ou à
mão. Escolher o teste por variável a partir desta tabela mudaria o erro tipo I do teste seguinte
(Rochon, Gondan e Kieser 2012) e misturaria duas estatísticas sob um BH só.

A matriz é a que as análises recebem, a processada. O escalonamento não muda o resultado: ele é
afim por variável, e o W de Shapiro-Wilk não muda quando a variável é multiplicada e somada por
constantes. Os grupos saem de `sample_mask`, e não de `groups`: o QC é a mistura de todas as
amostras e não é grupo, e excluí-lo pelo nome `"QC"` deixaria entrar o estudo que chame o pool
de outra coisa. A amostra `validate` também fica fora. A ordem é `sorted`, a mesma da ANOVA.

Em cada grupo, cada variável cai na primeira destas classes que a alcança:

- `fewer_than_three_values`: menos de 3 valores finitos no grupo. Sem W, p e q.
- `constant_in_group`: os valores finitos são todos iguais. Sem W, p e q.
- com valor imputado: avaliável, com alguma célula do grupo imputada. Tem W e p, sem q.
- testada: avaliável e sem valor imputado no grupo. Tem W, p e q, e é a população do resumo.

Constância é `ptp == 0`, o predicado de `metrics/structure.column_usability`, porque o desvio de
uma coluna constante não é exatamente zero em ponto flutuante. Sem os dois motivos, o `scipy`
devolveria `nan` para a coluna constante e levantaria com menos de três valores.

As variáveis com valor imputado ficam à parte porque o empate que a imputação cria faz o
Shapiro-Wilk rejeitar. Com os 3 menores de 18 valores normais trocados por uma constante abaixo
do mínimo, que é o que `one_fifth_min` faz a três ausentes, ele rejeita em 35% das vezes, contra
5% sem a troca. Depois de uma normalização por amostra os imputados deixam de ser iguais, então
a pergunta é feita à máscara de imputadas (`ProcessedMatrix.imputed_cells`), e não aos empates
da matriz.

O resumo é sobre a população testada de cada grupo: a fração com `p < REJECTION_LEVEL`, a
contagem com o `q` de Benjamini-Hochberg abaixo de `FDR_LEVEL`, e `1 - π̂₀` de Storey, com
`λ = STOREY_LAMBDA`, como piso da fração que não segue a normal. π̂₀ não é reportado como a
fração normal: com grupos de poucas dezenas de amostras o teste tem pouco poder, e π̂₀ sai alto
mesmo quando nenhuma variável é normal. Ele superestima a fração nula quando as alternativas não
têm `p` pequeno, então `1 - π̂₀` subestima a que rejeita. O BH e o Storey correm só sobre a
população testada, porque as imputadas têm `p` artificialmente pequeno; o `q` delas é `nan`.

`scipy.stats.shapiro(x, axis=0)` é vetorizado, então o caso comum, sem não finitos, é uma
chamada por grupo. Acima de 5.000 valores num grupo o `p` do `scipy` deixa de ser exato (ele
avisa). Grupos desse tamanho não existem neste domínio, e não há guarda para isso.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from scipy.stats import shapiro

from refino.domain.matrix import RefinoMatrix
from refino.statistics.univariate import MIN_PARA_SHAPIRO, benjamini_hochberg

__all__ = [
    "FDR_LEVEL",
    "REJECTION_LEVEL",
    "STOREY_LAMBDA",
    "GroupNormality",
    "NormalityResult",
    "NormalityUnassessable",
    "normality_by_group",
]

REJECTION_LEVEL: Final = 0.05
"""Nível de cada teste por variável, e a fração que se espera abaixo dele sob a normal.

Vale o mesmo que o `NORMALITY_ALPHA` de `statistics/univariate.py` e é outra quantidade: aquele
decide o teste do contraste a partir de um Shapiro-Wilk sobre todos os resíduos juntos."""

FDR_LEVEL: Final = 0.05
"""Corte do `q` de Benjamini-Hochberg na contagem do resumo.

Separado de `REJECTION_LEVEL` porque é um nível de taxa de falsa descoberta, e não o nível de um
teste."""

STOREY_LAMBDA: Final = 0.5
"""O `λ` do π̂₀ de Storey: `#{p > λ} / (m (1 - λ))`. 0,5 é o valor de Storey e Tibshirani
(2003) para um λ fixo."""

NormalityUnassessable = Literal["fewer_than_three_values", "constant_in_group"]
"""Por que uma variável não foi avaliada num grupo: um motivo por variável, ou nenhum.

Vai na resposta de `/execucoes`, e quem o valida em runtime é o `pydantic`, ao montar o esquema
da resposta com este mesmo `Literal`."""


@dataclass(frozen=True)
class GroupNormality:
    """O Shapiro-Wilk de cada variável num grupo, e o resumo dele.

    Os vetores são a fonte, e o resumo é derivado deles por propriedades, então não há um
    segundo número guardado que possa discordar da tabela por variável. Os vetores estão na
    ordem de `NormalityResult.feature` e são só leitura.
    """

    group: str
    n_samples: int
    """Quantas amostras de estudo o grupo tem. São as linhas, e não os valores finitos de cada
    variável, que estão em `n_values`."""
    statistic: np.ndarray
    """O W de cada variável; `nan` exatamente onde `unassessable` tem motivo."""
    pvalue: np.ndarray
    """O `p` de cada variável; `nan` exatamente onde `unassessable` tem motivo."""
    qvalue: np.ndarray
    """O `q` de Benjamini-Hochberg, só na população testada; `nan` fora dela."""
    n_values: np.ndarray
    """Quantos valores finitos cada variável tem no grupo. É o `n` do teste."""
    unassessable: tuple[NormalityUnassessable | None, ...]
    has_imputed: np.ndarray
    """Se alguma célula da variável, neste grupo, foi preenchida pela imputação."""

    @property
    def assessable(self) -> np.ndarray:
        """As variáveis que têm W e `p`: as duas últimas classes da partição."""
        return np.array([motivo is None for motivo in self.unassessable], dtype=bool)

    @property
    def tested(self) -> np.ndarray:
        """A população do resumo: avaliável e sem valor imputado no grupo."""
        populacao: np.ndarray = self.assessable & ~self.has_imputed
        return populacao

    @property
    def n_fewer_than_three(self) -> int:
        return sum(motivo == "fewer_than_three_values" for motivo in self.unassessable)

    @property
    def n_constant(self) -> int:
        return sum(motivo == "constant_in_group" for motivo in self.unassessable)

    @property
    def n_with_imputed(self) -> int:
        """As avaliáveis com valor imputado, contadas à parte e fora do resumo."""
        return int(np.sum(self.assessable & self.has_imputed))

    @property
    def n_with_imputed_rejected(self) -> int:
        """Delas, quantas rejeitam a normal. É o que o empate da imputação produz sozinho."""
        dentro = self.assessable & self.has_imputed
        return int(np.sum(self.pvalue[dentro] < REJECTION_LEVEL))

    @property
    def n_tested(self) -> int:
        return int(np.sum(self.tested))

    @property
    def n_rejected(self) -> int:
        return int(np.sum(self.pvalue[self.tested] < REJECTION_LEVEL))

    @property
    def fraction_rejected(self) -> float | None:
        """A fração da população testada com `p < REJECTION_LEVEL`. `None` sem população.

        `None` e não `0`: um grupo sem variável testada não tem fração nenhuma, e zero diria
        que nenhuma rejeitou.
        """
        if self.n_tested == 0:
            return None
        return self.n_rejected / self.n_tested

    @property
    def n_rejected_fdr(self) -> int:
        return int(np.sum(self.qvalue[self.tested] < FDR_LEVEL))

    @property
    def min_fraction_non_normal(self) -> float | None:
        """`1 - π̂₀` de Storey, cortado em zero: o piso da fração que não segue a normal.

        O corte em zero equivale ao teto de π̂₀ em 1. Sem população testada é `None`, pelo mesmo
        motivo de `fraction_rejected`.
        """
        if self.n_tested == 0:
            return None
        p = self.pvalue[self.tested]
        pi0 = float(np.sum(p > STOREY_LAMBDA)) / (int(p.size) * (1.0 - STOREY_LAMBDA))
        return 1.0 - min(1.0, pi0)


@dataclass(frozen=True)
class NormalityResult:
    """A normalidade de cada grupo de estudo, na ordem dos grupos, sobre as mesmas variáveis.

    Os três níveis viajam no resultado, porque as contagens do resumo sem eles não dizem contra
    o que foram contadas.
    """

    feature: tuple[str, ...]
    groups: tuple[GroupNormality, ...]
    rejection_level: float
    fdr_level: float
    storey_lambda: float
    provenance: tuple[str, ...]


def normality_by_group(matrix: RefinoMatrix, imputed: np.ndarray) -> NormalityResult:
    """O Shapiro-Wilk de cada variável em cada grupo de estudo, com a partição e o resumo.

    `matrix` é a que as análises recebem, a processada, e `imputed` marca as células dela cujo
    valor a imputação escolheu (`ProcessedMatrix.imputed_cells`). A forma das duas é conferida:
    elas coincidem hoje porque nenhuma etapa depois da imputação remove feature ou amostra.

    Não levanta por população. Grupo pequeno, variável constante e matriz sem amostra de estudo
    são respostas; a última é um resultado sem grupos.
    """
    valores = np.asarray(matrix.values, dtype=np.float64)
    mascara = np.asarray(imputed, dtype=bool)
    if mascara.shape != valores.shape:
        raise ValueError(
            f"a máscara de imputadas tem forma {mascara.shape} e a matriz {valores.shape}: ela "
            "marca as células desta matriz uma a uma, e com outra forma marcaria outras"
        )

    estudo = np.asarray(matrix.sample_mask, dtype=bool)
    rotulos = np.asarray(matrix.groups)
    resultados: list[GroupNormality] = []
    for nome in sorted(set(rotulos[estudo].tolist())):
        linhas = (rotulos == nome) & estudo
        resultados.append(_do_grupo(nome, valores[linhas], mascara[linhas]))
    return NormalityResult(
        feature=matrix.feature_names,
        groups=tuple(resultados),
        rejection_level=REJECTION_LEVEL,
        fdr_level=FDR_LEVEL,
        storey_lambda=STOREY_LAMBDA,
        provenance=matrix.provenance,
    )


def _do_grupo(nome: str, bloco: np.ndarray, imputadas: np.ndarray) -> GroupNormality:
    """O teste de cada variável de um grupo, agrupando as colunas pelo número de valores finitos.

    Os não finitos vão para o fim da coluna ordenada, e os `n` primeiros de uma coluna com `n`
    finitos são exatamente os valores dela. Assim uma chamada do `scipy` testa de uma vez todas
    as colunas com o mesmo `n`, sem um laço por coluna.

    O bloco tem sempre ao menos uma linha, porque o grupo nasce dos rótulos presentes, e por
    isso a primeira linha do ordenado é lida sem guarda.
    """
    n_linhas, n_colunas = bloco.shape
    finitos = np.isfinite(bloco)
    n_valores = finitos.sum(axis=0).astype(np.int64)
    ordenado = np.sort(np.where(finitos, bloco, np.inf), axis=0)

    poucos = n_valores < MIN_PARA_SHAPIRO
    # O maior valor finito contra o menor: `ptp == 0` sobre os finitos.
    maior = ordenado[np.maximum(n_valores - 1, 0), np.arange(n_colunas)]
    constante = ~poucos & (maior == ordenado[0])
    motivos: tuple[NormalityUnassessable | None, ...] = tuple(
        "fewer_than_three_values" if p else "constant_in_group" if c else None
        for p, c in zip(poucos.tolist(), constante.tolist(), strict=True)
    )

    estatistica = np.full(n_colunas, np.nan)
    pvalor = np.full(n_colunas, np.nan)
    avaliavel = ~poucos & ~constante
    for n in np.unique(n_valores[avaliavel]).tolist():
        colunas = np.flatnonzero(avaliavel & (n_valores == n))
        resultado = shapiro(ordenado[:n, colunas], axis=0)
        estatistica[colunas] = resultado.statistic
        pvalor[colunas] = resultado.pvalue

    com_imputado = imputadas.any(axis=0)
    testada = avaliavel & ~com_imputado
    qvalor = benjamini_hochberg(np.where(testada, pvalor, np.nan))

    for vetor in (estatistica, pvalor, qvalor, n_valores, com_imputado):
        vetor.setflags(write=False)
    return GroupNormality(
        group=nome,
        n_samples=n_linhas,
        statistic=estatistica,
        pvalue=pvalor,
        qvalue=qvalor,
        n_values=n_valores,
        unassessable=motivos,
        has_imputed=com_imputado,
    )

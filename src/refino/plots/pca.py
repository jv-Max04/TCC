"""As quatro figuras de PCA: scree, scores com região de predição, cargas e biplot.

A PCA desenhada é a do diagnóstico, `metrics.structure.structural_pca`: `log10(x+1)`,
autoescalonamento por coluna e SVD sobre a matriz ingerida. Não é `statistics.pca.pca`, que
só centra e espera a matriz processada; na mesma matriz as duas dão decomposições diferentes,
com os mesmos nomes de campo, e uma figura feita da segunda discordaria do número que a aba
de qualidade mostra. As quatro figuras descrevem uma matriz só. A PCA antes e depois do
relatório é a do painel da correção.

A leitura esperada pelo laboratório é o QC agrupado no meio da nuvem de amostras. Por isso
os scores e o biplot separam `qc`, `sample` e `validate` em séries próprias, com a contagem
na legenda; `sample` não é o complemento de `qc`, porque há três tipos. A região desenhada é
a de predição de Hotelling T² para uma nova observação, calculada sobre as amostras de
estudo, exata sob normalidade multivariada dos scores (que a figura não testa). Quando cada
amostra vem com o grupo, a nuvem é pintada por grupo e há uma região por grupo, recortada ao
quadro dos pontos, com o `n` de cada grupo na legenda. O sinal de cada componente é
arbitrário: o módulo não o corrige e o declara no rótulo dos eixos e no rodapé, porque só
distância, agrupamento e variância explicada não dependem dele.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from matplotlib.colors import same_color
from matplotlib.lines import Line2D
from scipy.stats import f as distribuicao_f

from refino.plots.base import (
    GID_DAS_FEATURES,
    GID_REQUEBRAVEL,
    MatrizVaziaError,
    RecusaDoConstrutorError,
    cores_do_ciclo,
    denominador_ou_recusa,
    inteiro_do_fio,
    nova_figura,
    numero_do_fio,
    vetor_do_fio_ou_nucleo,
)
from refino.plots.recorte import (
    TAMANHO_DA_FONTE,
    Recorte,
    aplica,
    decimal,
    frase_contada,
    legenda_fora,
    milhar,
    quebra,
)
from refino.plots.series import (
    MARCADOR_SEM_ESTILO,
    PARES_DE_ESTILO,
    SIMBOLOS_DOS_GRUPOS,
    EstiloDoGrupo,
    GrupoInvalidoError,
    ordem_canonica,
    rotulo_do_grupo,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from refino.plots.series import EstilosDosGrupos

__all__ = [
    "ALFA_DA_ELIPSE",
    "COLUNAS_DO_MOTIVO",
    "DIMENSAO_DA_ELIPSE",
    "MINIMO_PARA_ELIPSE",
    "NOTA_MAIOR_QUE_O_QUADRO",
    "NOTA_SEM_REGIAO",
    "PERMANOVA_AUSENTE",
    "PONTOS_DA_ELIPSE",
    "PRETRATAMENTO",
    "ROTULO_DAS_SETAS",
    "ROTULO_DA_ORIGEM",
    "SETAS_DO_BIPLOT",
    "TITULO_DAS_CARGAS",
    "TITULO_DOS_SCORES",
    "TITULO_DO_BIPLOT",
    "TITULO_DO_SCREE",
    "Cargas",
    "Elipse",
    "ElipseAusente",
    "Escores",
    "PcaDesalinhadaError",
    "Permanova",
    "RegiaoDoGrupo",
    "RegioesDosGrupos",
    "TipoDeAmostra",
    "cargas_da_pca",
    "cores_das_series",
    "elipse_das_amostras",
    "elipse_de_hotelling",
    "escores_da_pca",
    "figura_das_cargas",
    "figura_do_biplot",
    "figura_do_scree",
    "figura_dos_scores",
    "frase_da_elipse",
    "frase_da_permanova",
    "frase_das_regioes",
    "legenda_das_setas",
    "permanova_do_fio",
    "recorte_das_cargas",
    "recorte_do_biplot",
    "recorte_do_scree",
    "recorte_dos_scores",
    "regiao_dos_scores",
    "regioes_dos_grupos",
    "rotulo_da_regiao_do_grupo",
    "rotulo_do_componente",
    "setas_do_biplot",
]

TITULO_DO_SCREE: Final = "Scree — variância explicada por componente"

TITULO_DOS_SCORES: Final = "Scores de PCA — PC1 contra PC2"

TITULO_DAS_CARGAS: Final = "Cargas de PCA — PC1 contra PC2"

TITULO_DO_BIPLOT: Final = "Biplot de PCA — amostras e as features de maior carga"

# O pré-tratamento vai por extenso no rodapé. É ele que distingue esta PCA da de
# `statistics.pca`.
PRETRATAMENTO: Final = (
    "PCA do diagnóstico: log10(x+1), descarte de coluna constante ou não finita, "
    "autoescalonamento por coluna, e SVD"
)

# A ressalva do sinal, que as quatro figuras repetem. O sinal de um componente vem do SVD e
# não é controlado por quem chama; sem a frase, 'à direita' seria lido como fato do dado.
SINAL_ARBITRARIO: Final = (
    "o sinal de cada componente é arbitrário e não é comparável entre execuções — leia distância e "
    "agrupamento, nunca o lado"
)

# Dimensão do espaço da elipse. Uma figura de PC1 contra PC2 é bidimensional, então é
# constante e não parâmetro.
DIMENSAO_DA_ELIPSE: Final = 2

# `n >= p + 1` é a condição de existência da elipse. Com `n = p` a covariância amostral é
# singular, e com `n = p + 1` os graus de liberdade de `F_{p, n-p}` chegam a um. Abaixo
# disso a figura diz por extenso que não há região.
MINIMO_PARA_ELIPSE: Final = DIMENSAO_DA_ELIPSE + 1

# O nível que a figura declara. Nada neste módulo classifica por ele.
ALFA_DA_ELIPSE: Final = 0.05

# Quantos vértices desenham a elipse: o bastante para a curva não parecer um polígono.
PONTOS_DA_ELIPSE: Final = 200

ROTULO_DA_ORIGEM: Final = "origem"

ROTULO_DAS_SETAS: Final = "features de maior carga"

# Quantas setas o biplot desenha por padrão. Cada seta leva um número na ponta e uma entrada
# na lista numerada, e o texto domina o custo de redesenho: acima de cerca de 22 setas o
# redesenho passa de 100 ms. É constante porque o desenho e o rodapé usam o mesmo número.
SETAS_DO_BIPLOT: Final = 12

# Tracejado, para a elipse única e para a região de cada grupo: uma linha contínua seria lida
# como dado. É constante porque a linha no eixo e o traço na legenda usam o mesmo valor.
_TRACO_DA_REGIAO: Final = "--"

# Abaixo dos pontos. As séries são `Line2D` de `zorder` 2, e a região de um grupo é desenhada
# depois delas, quando o quadro dos pontos já existe. Sem este valor o tracejado ficaria por
# cima do QC.
_ORDEM_DAS_REGIOES: Final = 1.5

# Os marcadores do QC e da validação na figura por grupo sem estilo, quando o ciclo de cores
# dá a volta. São diferentes entre si e do círculo dos grupos, para o par de cor e marcador
# de cada série ser único mesmo que a cor repita a de um grupo. Ver `_tracos_por_grupo`.
_MARCADOR_DO_QC_NA_VOLTA: Final = SIMBOLOS_DOS_GRUPOS["quadrado"]
_MARCADOR_DA_VALIDACAO_NA_VOLTA: Final = SIMBOLOS_DOS_GRUPOS["triângulo"]

# As duas frases que a legenda escreve, entre parênteses, na entrada de um grupo: a primeira
# quando o grupo não tem região, a segunda quando a região existe e não aparece no quadro.
NOTA_SEM_REGIAO: Final = "sem região"
NOTA_MAIOR_QUE_O_QUADRO: Final = "região maior que o quadro"

# O teto de cada linha do motivo na legenda: o comprimento do mais longo dos dois motivos
# curtos, com o parêntese. Impede um motivo longo, como o da covariância singular, de alargar
# a legenda e estreitar o eixo. É derivado para acompanhar os motivos curtos se eles mudarem.
COLUNAS_DO_MOTIVO: Final = max(
    len(f"({NOTA_MAIOR_QUE_O_QUADRO})"),
    len(f"({NOTA_SEM_REGIAO}: menos de {MINIMO_PARA_ELIPSE})"),
)


class TipoDeAmostra(StrEnum):
    """As três séries do gráfico de scores.

    Os valores são os de `refino.domain.metadata.SampleType`, e um teste confere a igualdade.
    O enum não é derivado de lá porque o rótulo em português e a ordem de desenho são
    decisões desta figura. Um valor desconhecido é recusado na fronteira, em vez de a amostra
    sumir das três séries.
    """

    QC = "qc"
    AMOSTRA = "sample"
    VALIDACAO = "validate"


# A ordem de desenho e da legenda. O QC vem por último para ficar por cima: a pergunta da
# figura é se o QC está no meio da nuvem, e ele não pode ficar escondido sob as amostras.
ORDEM_DAS_SERIES: Final = (TipoDeAmostra.VALIDACAO, TipoDeAmostra.AMOSTRA, TipoDeAmostra.QC)

_ROTULO_DA_SERIE: Final[dict[TipoDeAmostra, str]] = {
    TipoDeAmostra.QC: "QC",
    TipoDeAmostra.AMOSTRA: "amostra de estudo",
    TipoDeAmostra.VALIDACAO: "amostra de validação",
}


class PcaDesalinhadaError(RecusaDoConstrutorError):
    """Os vetores que descrevem os mesmos objetos têm comprimentos diferentes.

    Fica separada de `MatrizVaziaError` porque o conserto é outro (população ausente é uma
    execução que não correu; população desalinhada é um corpo montado errado) e porque o
    painel mostra uma frase por subtipo.
    """


# ---------------------------------------------------------------------------
# As duas populações: uma por amostra, outra por feature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Escores:
    """Os scores de PC1 e PC2 de cada injeção, com o tipo de cada uma.

    Uma entrada por amostra, na ordem das linhas da matriz, que `RefinoMatrix` garante ser a
    ordem de injeção.
    """

    x: np.ndarray
    y: np.ndarray
    tipos: tuple[TipoDeAmostra, ...]
    grupos: tuple[str | None, ...] | None = None
    """O grupo de cada amostra de estudo, com `None` nas outras injeções; ou `None` inteiro.

    `None` inteiro quer dizer que o grupo não veio, e a figura é a por tipo de amostra, com
    uma região só. `None` numa posição quer dizer que a injeção é QC ou validação.
    `escores_da_pca` garante que o grupo existe exatamente onde o tipo é `sample`.
    """
    grupos_em_branco: int = 0
    """Quantas amostras de estudo chegaram com o nome de grupo em branco.

    Quando há alguma, `grupos` é `None` e a figura é a por tipo de amostra, com o título
    dizendo quantas. Não é recusa porque é fato do dado: a ingestão aceita o rótulo de grupo
    vazio. Pintar por grupo deixaria essas amostras fora das séries, e um grupo de nome
    inventado criaria uma população. É contador, e não booleano, porque uma amostra e metade
    do estudo são leituras diferentes.
    """

    @property
    def n(self) -> int:
        return int(self.x.size)

    def mascara(self, tipo: TipoDeAmostra) -> np.ndarray:
        """Booleano por amostra: `True` onde o tipo casa e as duas coordenadas são finitas.

        As duas condições ficam na mesma máscara para a contagem da legenda ser a dos pontos
        desenhados.
        """
        do_tipo = np.array([t is tipo for t in self.tipos], dtype=bool)
        return do_tipo & np.isfinite(self.x) & np.isfinite(self.y)

    @property
    def grupos_de_estudo(self) -> tuple[str, ...]:
        """Os grupos das amostras de estudo, na ordem canônica. Vazio sem vetor de grupos.

        Inclui todos os grupos do vetor, e não só os desenháveis: o grupo cujas amostras têm
        todas score não finito não tem série, mas entra na legenda.
        """
        if self.grupos is None:
            return ()
        return ordem_canonica(g for g in self.grupos if g is not None)

    def mascara_do_grupo(self, grupo: str) -> np.ndarray:
        """Booleano por amostra: de estudo, do `grupo`, e com as duas coordenadas finitas.

        Parte da máscara das amostras de estudo, e assim o QC fica de fora por construção.
        """
        if self.grupos is None:
            raise ValueError(
                f"estes escores não trazem o grupo de cada amostra, e a máscara do grupo {grupo!r} "
                "não existe: quem pede regiões por grupo tem de passar `grupos` a `escores_da_pca`"
            )
        do_grupo = np.array([g == grupo for g in self.grupos], dtype=bool)
        mascara: np.ndarray = self.mascara(TipoDeAmostra.AMOSTRA) & do_grupo
        return mascara


@dataclass(frozen=True)
class Cargas:
    """A carga de cada feature em PC1 e PC2, com o nome de cada uma.

    Uma entrada por feature retida: a PCA descarta coluna constante e coluna não finita, e o
    vetor de nomes tem de ser o das retidas. Com os nomes da matriz inteira, cada ponto seria
    rotulado com o nome de outra feature.
    """

    x: np.ndarray
    y: np.ndarray
    features: tuple[str, ...]

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def desenhaveis(self) -> np.ndarray:
        """Booleano por feature: `True` onde as duas cargas são finitas."""
        return np.isfinite(self.x) & np.isfinite(self.y)


def escores_da_pca(
    *,
    pc1: Sequence[float] | np.ndarray | Any,
    pc2: Sequence[float] | np.ndarray | Any,
    tipos: Sequence[str] | Any,
    grupos: Sequence[str | None] | Any = None,
) -> Escores:
    """Os scores vindos do corpo, com as guardas de fronteira.

    O tipo de amostra é validado aqui. Um valor fora dos três tipos é recusado: fora das três
    séries, a amostra sumiria da figura e a legenda contaria menos do que a matriz tem.

    `grupos` ausente dá a figura por tipo de amostra. Uma lista é conferida contra os tipos,
    posição a posição, em `_grupos_ou_recusa`. Nome de grupo em branco numa amostra de estudo
    não é recusa; ver `Escores.grupos_em_branco`.
    """
    x = vetor_do_fio_ou_nucleo(pc1)
    y = vetor_do_fio_ou_nucleo(pc2)
    if x.size == 0:
        raise MatrizVaziaError(
            "não veio score nenhum: a PCA não correu para esta execução. Figura em branco seria "
            "indistinguível de figura quebrada"
        )
    if x.size != y.size:
        raise PcaDesalinhadaError(
            f"{x.size} score(s) de PC1 para {y.size} de PC2. Os dois vetores descrevem as mesmas "
            "injeções, e comprimentos diferentes significam que um deles é de outra população"
        )
    crus = list(tipos) if isinstance(tipos, (list, tuple)) else []
    if len(crus) != x.size:
        raise PcaDesalinhadaError(
            f"{len(crus)} tipo(s) de amostra para {x.size} score(s). O tipo é o que separa o QC da "
            "nuvem, e um vetor desalinhado poria o QC no meio das amostras por acidente de índice"
        )
    tipados = tuple(_tipo_ou_recusa(t) for t in crus)
    lidos, em_branco = _grupos_ou_recusa(grupos, tipados)
    return Escores(x=x, y=y, tipos=tipados, grupos=lidos, grupos_em_branco=em_branco)


def _grupos_ou_recusa(
    grupos: Sequence[str | None] | Any, tipos: tuple[TipoDeAmostra, ...]
) -> tuple[tuple[str | None, ...] | None, int]:
    """O vetor de grupos conferido contra os tipos, e quantos nomes vieram em branco.

    O servidor envia o grupo onde o tipo é `sample` e `None` nas outras injeções. As duas
    violações são recusadas. Uma amostra de estudo sem grupo sairia das séries e da região do
    grupo dela. Um QC com grupo indica corpo montado errado, e aceitá-lo desenharia uma
    região em torno do QC.

    Nome em branco numa amostra de estudo é fato do dado: o vetor sai como `None`, com a
    contagem. As recusas vêm antes, então um corpo com QC agrupado é recusado mesmo que
    também traga um nome em branco.
    """
    if grupos is None:
        return None, 0
    crus = list(grupos) if isinstance(grupos, (list, tuple)) else None
    if crus is None or len(crus) != len(tipos):
        recebidos = f"{len(crus)} grupo(s)" if crus is not None else f"um {type(grupos).__name__}"
        raise PcaDesalinhadaError(
            f"{recebidos} para {len(tipos)} amostra(s). O grupo é o que pinta cada amostra de "
            "estudo, e um vetor desalinhado poria uma amostra na cor e na região de outro grupo"
        )
    lidos = tuple(
        _grupo_da_amostra(posicao, tipo, valor)
        for posicao, (tipo, valor) in enumerate(zip(tipos, crus, strict=True))
    )
    em_branco = sum(1 for g in lidos if g is not None and not g.strip())
    return (None, em_branco) if em_branco else (lidos, 0)


def _grupo_da_amostra(posicao: int, tipo: TipoDeAmostra, valor: Any) -> str | None:
    """O grupo de uma injeção: texto na amostra de estudo, `None` nas outras.

    Aceita texto vazio; o nome em branco é contado por quem chama.
    """
    if tipo is TipoDeAmostra.AMOSTRA:
        if isinstance(valor, str):
            return valor
        raise GrupoInvalidoError(
            f"a amostra de estudo na posição {posicao} chegou com o grupo {valor!r}. Toda amostra "
            "de estudo tem grupo, e uma sem ele sairia das séries e da região do grupo dela — com "
            "a legenda a contar menos do que a matriz tem, e nada a levantar"
        )
    if valor is None:
        return None
    raise GrupoInvalidoError(
        f"a injeção na posição {posicao} é {tipo.value!r} e chegou com o grupo {valor!r}. Só a "
        "amostra de estudo tem grupo (`sample_mask`): o QC é a régua e não um grupo clínico, "
        "e aceitá-lo aqui desenharia uma região à volta dele"
    )


def _tipo_ou_recusa(valor: Any) -> TipoDeAmostra:
    """Um `TipoDeAmostra`, ou a recusa. Valor desconhecido não vira amostra de estudo."""
    try:
        return TipoDeAmostra(valor)
    except ValueError as erro:
        conhecidos = ", ".join(sorted(t.value for t in TipoDeAmostra))
        raise PcaDesalinhadaError(
            f"tipo de amostra {valor!r} não é um dos três: {conhecidos}. Um valor desconhecido "
            "aqui não pode virar «amostra de estudo» por omissão — a legenda contaria uma "
            "população que não existe, e a elipse seria calculada sobre ela"
        ) from erro


def cargas_da_pca(
    *,
    pc1: Sequence[float] | np.ndarray | Any,
    pc2: Sequence[float] | np.ndarray | Any,
    features: Sequence[str] | Any,
) -> Cargas:
    """As cargas vindas do corpo, alinhadas ao vetor de nomes das features retidas.

    Os nomes acompanham as cargas porque a figura põe um ponto por feature e a identificação
    por clique precisa dizer qual é.
    """
    x = vetor_do_fio_ou_nucleo(pc1)
    y = vetor_do_fio_ou_nucleo(pc2)
    if x.size == 0:
        raise MatrizVaziaError(
            "não veio carga nenhuma: a PCA não correu, ou o passo 3 descartou todas as colunas. "
            "Figura em branco seria indistinguível de figura quebrada"
        )
    if x.size != y.size:
        raise PcaDesalinhadaError(
            f"{x.size} carga(s) de PC1 para {y.size} de PC2. Os dois vetores descrevem as mesmas "
            "features, e comprimentos diferentes significam que um deles é de outra população"
        )
    nomes = [str(f) for f in features] if isinstance(features, (list, tuple)) else []
    if len(nomes) != x.size:
        raise PcaDesalinhadaError(
            f"{len(nomes)} nome(s) de feature para {x.size} carga(s). O nome é o que a "
            "identificação de R2 mostra, e um vetor desalinhado devolveria o nome do vizinho — com "
            "a figura bem formada"
        )
    return Cargas(x=x, y=y, features=tuple(nomes))


# ---------------------------------------------------------------------------
# A elipse de Hotelling T²
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Elipse:
    """A região de predição, em parâmetros de desenho.

    `c2` é o quantil. Vai junto porque o rodapé o declara, e é ele que permite conferir a
    região.
    """

    centro: tuple[float, float]
    semieixos: tuple[float, float]
    angulo_graus: float
    n: int
    alfa: float
    c2: float

    def contorno(self, vertices: int = PONTOS_DA_ELIPSE) -> tuple[np.ndarray, np.ndarray]:
        """O polígono fechado da elipse, em coordenadas de score.

        O último vértice repete o primeiro, para o contorno não ter uma fenda.
        """
        t = np.linspace(0.0, 2.0 * math.pi, vertices)
        a, b = self.semieixos
        theta = math.radians(self.angulo_graus)
        base_x = a * np.cos(t)
        base_y = b * np.sin(t)
        x = self.centro[0] + base_x * math.cos(theta) - base_y * math.sin(theta)
        y = self.centro[1] + base_x * math.sin(theta) + base_y * math.cos(theta)
        return x, y

    def contem(self, x: float, y: float) -> bool:
        """Se o ponto `(x, y)` cai dentro da região, com a borda incluída.

        Usa a geometria do desenho (centro, semieixos e ângulo), com a rotação inversa à de
        `contorno`, porque o objeto não guarda a covariância.
        """
        theta = math.radians(self.angulo_graus)
        dx, dy = x - self.centro[0], y - self.centro[1]
        ao_longo = dx * math.cos(theta) + dy * math.sin(theta)
        atraves = -dx * math.sin(theta) + dy * math.cos(theta)
        a, b = self.semieixos
        return (ao_longo / a) ** 2 + (atraves / b) ** 2 <= 1.0


@dataclass(frozen=True)
class ElipseAusente:
    """A elipse não foi calculada, e o motivo.

    É um tipo próprio, e não `None`, porque 'poucos pontos' e 'covariância singular' são
    fatos diferentes. O motivo é o texto que a figura escreve no lugar da elipse.
    """

    motivo: str


def elipse_de_hotelling(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alfa: float = ALFA_DA_ELIPSE,
) -> Elipse | ElipseAusente:
    """A região de predição de Hotelling T² a `1 - alfa` para uma nova observação.

        (x - xbar)' S^-1 (x - xbar)  <=  c2,   c2 = p(n+1)(n-1) / (n(n-p)) * F(p, n-p; 1-alfa)

    É a região de predição, e não a da média. A da média encolhe com `n` e fica em torno do
    centroide, o que faria quase todo ponto parecer discrepante. A de predição diz onde
    cairia a próxima amostra, que é a pergunta de quem olha se o QC está dentro da nuvem.

    É exata sob normalidade multivariada. A aproximação de qui-quadrado que o MetaboAnalyst
    usa supõe média e covariância conhecidas e subestima a região com poucas amostras: com
    `n = 30` cobre cerca de 92%, e não 95%. O teste unitário mede a cobertura por simulação.

    A covariância é a de `np.cov`, com `ddof = 1`, como a fórmula exige. Com `ddof = 0` a
    elipse sairia menor por um fator `(n-1)/n`.

    Devolve `ElipseAusente` em vez de levantar quando há poucos pontos ou a covariância é
    singular, para a figura declarar o motivo.
    """
    p = DIMENSAO_DA_ELIPSE
    finitos = np.isfinite(x) & np.isfinite(y)
    pontos = np.column_stack((x[finitos], y[finitos]))
    n = int(pontos.shape[0])
    if n < MINIMO_PARA_ELIPSE:
        return ElipseAusente(
            f"a elipse não se calcula com {n} ponto(s) finito(s): são precisos ao menos "
            f"{MINIMO_PARA_ELIPSE}, porque com n = p a covariância amostral é singular por "
            "construção"
        )
    if not 0.0 < alfa < 1.0:
        raise ValueError(f"alfa tem de estar em (0, 1); recebi {alfa}")

    centro = pontos.mean(axis=0)
    covariancia = np.cov(pontos, rowvar=False)
    autovalores, autovetores = np.linalg.eigh(covariancia)
    if not np.all(np.isfinite(autovalores)) or float(autovalores.min()) <= 0.0:
        return ElipseAusente(
            "a covariância destes pontos é singular ou não finita: eles caem sobre uma reta (ou "
            "num ponto só), e a região degenera num segmento. Um segmento desenhado como elipse "
            "lê-se como elipse muito achatada, que é outro fato"
        )

    c2 = p * (n + 1) * (n - 1) / (n * (n - p)) * float(distribuicao_f.ppf(1.0 - alfa, p, n - p))
    # `eigh` devolve os autovalores em ordem crescente. O semieixo maior é o do último, e o
    # ângulo sai dele; trocar a ordem rodaria a elipse em 90 graus.
    maior = int(np.argmax(autovalores))
    menor = 1 - maior
    direcao = autovetores[:, maior]
    return Elipse(
        centro=(float(centro[0]), float(centro[1])),
        semieixos=(
            math.sqrt(c2 * float(autovalores[maior])),
            math.sqrt(c2 * float(autovalores[menor])),
        ),
        angulo_graus=math.degrees(math.atan2(float(direcao[1]), float(direcao[0]))),
        n=n,
        alfa=alfa,
        c2=c2,
    )


def elipse_das_amostras(
    escores: Escores, *, alfa: float = ALFA_DA_ELIPSE
) -> Elipse | ElipseAusente:
    """A elipse sobre as amostras de estudo.

    A população é só a das amostras de estudo porque a pergunta é se o QC cai dentro da nuvem
    de amostras. Calculada sobre todas as injeções, a elipse cresceria para conter os QCs, e
    a resposta seria sempre sim. A janela e os testes usam esta função para a seleção ser
    feita num lugar só.
    """
    mascara = escores.mascara(TipoDeAmostra.AMOSTRA)
    return elipse_de_hotelling(escores.x[mascara], escores.y[mascara], alfa=alfa)


def frase_da_elipse(regiao: Elipse | ElipseAusente) -> str:
    """O que o rodapé escreve sobre a região: o método e o `n`.

    O `n` entra porque o quantil depende dele: uma elipse a 95% sobre sete pontos é muito
    maior que sobre sessenta.
    """
    if isinstance(regiao, ElipseAusente):
        return f"sem elipse: {regiao.motivo}"
    confianca = f"{(1.0 - regiao.alfa) * 100:.0f}%".replace(".", ",")
    return (
        f"elipse de predição de Hotelling T² a {confianca} sobre as {milhar(regiao.n)} amostras de "
        f"estudo (p = {DIMENSAO_DA_ELIPSE}, F de {DIMENSAO_DA_ELIPSE} e "
        f"{milhar(regiao.n - DIMENSAO_DA_ELIPSE)} graus, c² = {decimal(regiao.c2)}), exacta sob "
        "normalidade multivariada dos scores — que esta figura NÃO testa. Não é a aproximação de "
        "qui-quadrado do MetaboAnalyst, que subestima a região com poucas amostras"
    )


# ---------------------------------------------------------------------------
# Uma região por grupo de estudo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegiaoDoGrupo:
    """A região de predição de um grupo de estudo, com a contagem do grupo.

    `n` é o número de amostras do grupo com as duas coordenadas finitas, a população sobre a
    qual a região é calculada. Fica fora da `Elipse` porque `ElipseAusente` não o tem, e a
    legenda o escreve nos dois casos.
    """

    grupo: str
    n: int
    regiao: Elipse | ElipseAusente


@dataclass(frozen=True)
class RegioesDosGrupos:
    """Uma região por grupo de estudo, na ordem canônica.

    O tipo diz à figura e ao rodapé qual forma desenhar: `Elipse | ElipseAusente` é a região
    única sobre as amostras de estudo, e este é o caso de uma região por grupo. Os dois
    recebem o mesmo objeto, e assim o rodapé não declara regiões que a figura não desenhou.
    """

    por_grupo: tuple[RegiaoDoGrupo, ...]

    @property
    def grupos(self) -> tuple[str, ...]:
        return tuple(item.grupo for item in self.por_grupo)

    def de(self, grupo: str) -> RegiaoDoGrupo:
        """A região do `grupo`. Levanta `KeyError` para um grupo que não está aqui."""
        for item in self.por_grupo:
            if item.grupo == grupo:
                return item
        raise KeyError(grupo)


def regioes_dos_grupos(escores: Escores, *, alfa: float = ALFA_DA_ELIPSE) -> RegioesDosGrupos:
    """A região de Hotelling T² de cada grupo de estudo, sobre os pontos daquele grupo.

    A forma da região é a mesma da região única, calculada por `elipse_de_hotelling`; muda a
    população, e com ela o `n`. Com grupos pequenos o quantil cresce muito (um grupo de três
    amostras tem raio de mais de trinta desvios), e por isso a figura recorta a região ao
    quadro dos pontos e escreve o `n` de cada grupo na legenda, em vez de exigir um mínimo.

    O QC nunca entra, porque `mascara_do_grupo` parte das amostras de estudo. Um grupo com
    menos de três pontos finitos fica com `ElipseAusente` e continua na lista.
    """
    if escores.grupos is None:
        raise ValueError(
            "estes escores não trazem o grupo de cada amostra, e sem ele não há região por grupo: "
            "a região é única, a que `regiao_dos_scores` devolve neste caso"
        )
    regioes = []
    for grupo in escores.grupos_de_estudo:
        mascara = escores.mascara_do_grupo(grupo)
        regioes.append(
            RegiaoDoGrupo(
                grupo=grupo,
                n=int(mascara.sum()),
                regiao=elipse_de_hotelling(escores.x[mascara], escores.y[mascara], alfa=alfa),
            )
        )
    return RegioesDosGrupos(por_grupo=tuple(regioes))


def regiao_dos_scores(
    escores: Escores, *, alfa: float = ALFA_DA_ELIPSE
) -> Elipse | ElipseAusente | RegioesDosGrupos:
    """A região que o gráfico de scores desenha.

    Com o vetor de grupos, uma por grupo; sem ele, a região única das amostras de estudo. A
    janela e os testes usam esta função para a escolha ser feita num lugar só.
    """
    if escores.grupos is None:
        return elipse_das_amostras(escores, alfa=alfa)
    return regioes_dos_grupos(escores, alfa=alfa)


def rotulo_da_regiao_do_grupo(grupo: str) -> str:
    """O rótulo da linha da região de um grupo no eixo: `"região de predição — A"`.

    Não entra na legenda, onde a entrada do grupo já mostra o tracejado. É o texto pelo qual
    o arnês de teste encontra a região de cada grupo e o que a identificação por clique
    mostra.
    """
    return f"região de predição — {grupo}"


def frase_das_regioes(regioes: RegioesDosGrupos) -> str:
    """O que o rodapé escreve sobre as regiões dos grupos: o método, a faixa de c² e o recorte.

    A frase tem o tamanho da frase da região única, porque este rodapé é o mais cheio do
    pacote e `recorte.aplica` recusa a figura acima de `RESERVA_MAXIMA`. Por isso ela traz a
    faixa de c², e não um valor por grupo, e não nomeia os grupos sem região: o `n` e o motivo
    de cada grupo estão na legenda.
    """
    if not regioes.por_grupo:
        return f"{NOTA_SEM_REGIAO}: não há amostra de estudo com grupo nesta matriz"
    com = [item.regiao for item in regioes.por_grupo if isinstance(item.regiao, Elipse)]
    if not com:
        return (
            "nenhum grupo de estudo sustenta a região de predição de Hotelling T², e a legenda diz "
            "porquê"
        )
    menor, maior = min(r.c2 for r in com), max(r.c2 for r in com)
    faixa = (
        f"c² = {decimal(menor)}" if menor == maior else f"c² de {decimal(menor)} a {decimal(maior)}"
    )
    confianca = f"{(1.0 - com[0].alfa) * 100:.0f}%".replace(".", ",")
    return (
        f"regiões de predição de Hotelling T² a {confianca}, uma por grupo, recortadas ao quadro "
        f"(p = {DIMENSAO_DA_ELIPSE}, {faixa} pelo n), exactas sob normalidade multivariada dos "
        "scores — que esta figura NÃO testa. Não são as de qui-quadrado do MetaboAnalyst, que as "
        "subestimam com poucas amostras"
    )


# ---------------------------------------------------------------------------
# Os rótulos e as cores
# ---------------------------------------------------------------------------


def rotulo_do_componente(indice: int, fracao: float, *, prefixo: str = "") -> str:
    """O rótulo de eixo de um componente, em duas linhas: a variância e `"(sinal arbitrário)"`.

    A fração da variância vai no eixo para se ver que um componente explica muito mais que o
    outro, o que torna a distância horizontal comparável com a vertical. A ressalva do sinal
    vai onde o eixo é lido.

    São duas linhas porque numa só, com o prefixo das cargas, o rótulo do eixo y seria cortado
    na borda da figura.
    """
    corpo = (
        f"{prefixo}PC{indice + 1} — variância não declarada"
        if not math.isfinite(fracao)
        else f"{prefixo}PC{indice + 1} — {fracao * 100:.1f}% da variância".replace(".", ",")
    )
    return f"{corpo}\n(sinal arbitrário)"


def cores_das_series() -> dict[TipoDeAmostra, Any]:
    """Uma cor por tipo de amostra, do ciclo dos `rcParams`.

    A atribuição segue `ORDEM_DAS_SERIES`, e assim a mesma série tem a mesma cor nos scores e
    no biplot.
    """
    return dict(zip(ORDEM_DAS_SERIES, cores_do_ciclo(len(ORDEM_DAS_SERIES)), strict=True))


def _rotulo_da_serie(tipo: TipoDeAmostra, quantas: int) -> str:
    """A entrada de legenda de uma série por tipo, com a contagem: `"QC — 7 amostras"`.

    A contagem é por série porque é ela que decide o que se pode ler: o total esconde a
    diferença entre sete QCs e dezenas de amostras.
    """
    plural = "amostra" if quantas == 1 else "amostras"
    return f"{_ROTULO_DA_SERIE[tipo]} — {milhar(quantas)} {plural}"


# ---------------------------------------------------------------------------
# Os quatro recortes
# ---------------------------------------------------------------------------


def _amostragem_completa(n_desenhadas: int, n_totais: int) -> str:
    """A frase do recorte quando não há amostragem, com a contagem das features descartadas.

    `frase_contada` cuida do zero e do singular.
    """
    return (
        f"nenhuma — as {milhar(n_desenhadas)} features que a PCA reteve entram todas; "
        + frase_contada(
            n_totais - n_desenhadas,
            nenhuma="nenhuma foi descartada pelo passo 3",
            uma="1 descartada pelo passo 3 é constante ou não finita",
            algumas="descartadas pelo passo 3 são constantes ou não finitas",
        )
    )


def _frase_das_indefinidas(fora: int) -> str:
    """A frase das features de carga não finita, com o zero e o singular tratados."""
    return frase_contada(
        fora,
        nenhuma="nenhuma tem carga não finita",
        uma="1 de carga não finita é contada, não descartada",
        algumas="de carga não finita são contadas, não descartadas",
    )


def recorte_do_scree(*, n_features_usadas: Any, n_features_totais: Any) -> Recorte:
    """O rodapé do scree: de que população saíram os componentes, e por que método.

    O construtor recebe só o vetor de frações. Quantas features a PCA reteve e quantas a
    matriz tinha vêm do corpo da resposta, onde `null` é possível, e por isso passam por
    `inteiro_do_fio`.
    """
    usadas = inteiro_do_fio(n_features_usadas)
    totais = inteiro_do_fio(n_features_totais)
    return Recorte(
        desenhadas=usadas,
        disponiveis=denominador_ou_recusa(totais, de="a PCA"),
        amostragem=_amostragem_completa(usadas, totais),
        matriz="a matriz ingerida, antes de qualquer correção",
        metodo=(
            f"{PRETRATAMENTO}. Sem teste estatístico. A fração é da variância TOTAL da matriz "
            "autoescalonada — soma 1 sobre todos os componentes —, e não da variância retida pelos "
            "componentes desenhados"
        ),
    )


@dataclass(frozen=True)
class Permanova:
    """O par PERMANOVA e dispersão vindo do corpo, já validado.

    Os quatro números ficam num objeto só para não existir chamada com metade do par.

    `grupos` e `n_por_grupo` são a população do teste, que não é a da nuvem desenhada: o teste
    corre sobre as amostras de estudo, e a figura desenha todas as injeções. O rodapé declara
    as duas.
    """

    estatistica: float
    pvalue: float
    estatistica_da_dispersao: float
    pvalue_da_dispersao: float
    permutacoes: int
    semente: int
    grupos: tuple[str, ...]
    n_por_grupo: tuple[int, ...]


def permanova_do_fio(bruto: Any) -> Permanova | None:
    """O par vindo do corpo de `/matrizes/diagnosticar`, ou `None`.

    O campo pode faltar porque a população não sustentou o teste ou porque o servidor não o
    produz. Nenhum dos dois casos é 'correu e não separou', que é um p alto.

    O par é tudo ou nada: um corpo com `pvalue` e sem `pvalue_da_dispersao` dá `None`, e não
    uma figura com metade do par. O sinal é o `nan` que `numero_do_fio` devolve para o que
    não é número.
    """
    if not isinstance(bruto, dict):
        return None
    numeros = {
        chave: numero_do_fio(bruto.get(chave))
        for chave in ("estatistica", "pvalue", "estatistica_da_dispersao", "pvalue_da_dispersao")
    }
    # As chaves são enumeradas acima, em vez de lidas do corpo, porque `all` sobre um
    # dicionário vazio seria `True`.
    if not all(math.isfinite(valor) for valor in numeros.values()):
        return None
    crus = bruto.get("grupos")
    nomes = tuple(str(nome) for nome in crus) if isinstance(crus, (list, tuple)) else ()
    contagens = bruto.get("n_por_grupo")
    ns = tuple(inteiro_do_fio(n) for n in contagens) if isinstance(contagens, (list, tuple)) else ()
    if len(nomes) != len(ns) or not nomes:
        # População desalinhada é corpo montado errado. Com metade dela, o rodapé declararia
        # um `n` que não é o do teste.
        return None
    return Permanova(
        estatistica=numeros["estatistica"],
        pvalue=numeros["pvalue"],
        estatistica_da_dispersao=numeros["estatistica_da_dispersao"],
        pvalue_da_dispersao=numeros["pvalue_da_dispersao"],
        permutacoes=inteiro_do_fio(bruto.get("permutacoes")),
        semente=inteiro_do_fio(bruto.get("semente")),
        grupos=nomes,
        n_por_grupo=ns,
    )


# A frase usada quando o par não foi calculado. Não diz que não houve separação, que é outro
# fato: a nuvem é desenhada do mesmo jeito, e o que falta é o teste.
PERMANOVA_AUSENTE: Final = (
    "PERMANOVA não calculada: a matriz não declara dois grupos de estudo com amostras que cheguem, "
    "ou o corpo veio de um servidor que ainda não a produz"
)


def frase_da_permanova(par: Permanova | None) -> str:
    """O par de números por extenso, ou a ausência declarada.

    Os dois p vão na mesma frase. O p da PERMANOVA é confundido por heterogeneidade de
    dispersão: duas nuvens concêntricas, uma muito mais larga, dariam um p que não distingue
    centroides de dispersão.

    A frase declara a população do teste, as amostras de estudo, que não é a da nuvem. Sem
    isso os dois números seriam lidos sobre o `n` do título.

    Um p alto é descrito como achado válido, para não ser lido como defeito do aplicativo.
    """
    if par is None:
        return PERMANOVA_AUSENTE
    populacao = ", ".join(
        f"{nome} {milhar(n)}" for nome, n in zip(par.grupos, par.n_por_grupo, strict=True)
    )
    return (
        f"PERMANOVA nos componentes, {milhar(par.permutacoes)} permutações "
        f"(semente {par.semente}): pseudo-F = {decimal(par.estatistica)}, "
        f"p = {decimal(par.pvalue)}; dispersão, sempre junto: "
        f"F = {decimal(par.estatistica_da_dispersao)}, p = {decimal(par.pvalue_da_dispersao)}. "
        f"Sobre {milhar(sum(par.n_por_grupo))} amostras de estudo ({populacao}) — QC e validação "
        "fora, ao contrário da nuvem. Um sem o outro não separa centroides de dispersão, e p alto "
        "é achado válido e não ausência de medida"
    )


def recorte_dos_scores(
    *,
    n_features_usadas: Any,
    n_features_totais: Any,
    regiao: Elipse | ElipseAusente | RegioesDosGrupos,
    permanova: Permanova | None = None,
) -> Recorte:
    """O rodapé do gráfico de scores: população, método, a região e o par da PERMANOVA.

    As contagens de features vêm do corpo de `/matrizes/diagnosticar` e passam por
    `inteiro_do_fio`.

    `permanova` é opcional porque só esta figura tem o par. Com ele, este rodapé é o mais
    cheio do pacote e fica perto de `RESERVA_MAXIMA`; numa figura baixa `recorte.aplica` o
    recusa.

    Com `RegioesDosGrupos` o método descreve as regiões por grupo (`frase_das_regioes`); com
    `Elipse` ou `ElipseAusente`, a região única. É o mesmo objeto que a figura recebe.
    """
    usadas = inteiro_do_fio(n_features_usadas)
    totais = inteiro_do_fio(n_features_totais)
    sobre_a_regiao = (
        frase_das_regioes(regiao)
        if isinstance(regiao, RegioesDosGrupos)
        else frase_da_elipse(regiao)
    )
    return Recorte(
        desenhadas=usadas,
        disponiveis=denominador_ou_recusa(totais, de="a PCA"),
        amostragem=_amostragem_completa(usadas, totais),
        matriz="a matriz ingerida, antes de qualquer correção",
        metodo=(
            f"{PRETRATAMENTO}. {sobre_a_regiao}. E {SINAL_ARBITRARIO}. "
            f"{frase_da_permanova(permanova)}"
        ),
    )


def recorte_das_cargas(*, cargas: Cargas, n_features_totais: Any) -> Recorte:
    """O rodapé das cargas: as desenháveis sobre as features da matriz inteira.

    Entre as duas pontas ficam as features que a PCA descartou e as de carga não finita.
    """
    desenhaveis = int(cargas.desenhaveis.sum())
    return Recorte(
        desenhadas=desenhaveis,
        disponiveis=denominador_ou_recusa(n_features_totais, de="a PCA"),
        amostragem=(
            f"nenhuma — as {milhar(cargas.n)} features retidas pela PCA entram todas, e "
            f"{_frase_das_indefinidas(cargas.n - desenhaveis)}"
        ),
        matriz="a matriz ingerida, antes de qualquer correção",
        metodo=(
            f"{PRETRATAMENTO}. Sem teste estatístico: a carga é a projeção da feature no "
            f"componente, e não uma medida de significância. E {SINAL_ARBITRARIO}"
        ),
    )


def recorte_do_biplot(
    *, escores: Escores, cargas: Cargas, n_setas: int, n_features_totais: Any
) -> Recorte:
    """O rodapé do biplot: a lista numerada das setas, o fator de escala e a população.

    A seleção e o fator são recalculados aqui com os mesmos argumentos do construtor. Como
    `setas_do_biplot` e `_fator_das_setas` são determinísticas, o rodapé e o desenho coincidem.

    A lista numerada fica no rodapé porque as setas de um biplot apontam quase todas na mesma
    direção, e os nomes escritos nas pontas se sobreporiam. Na ponta vai só o número.

    O fator de escala é declarado porque os eixos são de score e as setas são de carga
    multiplicada. Sem ele, o comprimento de uma seta seria lido em unidades de score.
    """
    indices = setas_do_biplot(cargas, n_setas)
    return Recorte(
        desenhadas=int(indices.size),
        disponiveis=denominador_ou_recusa(n_features_totais, de="a PCA"),
        amostragem=(
            f"as {milhar(int(indices.size))} features de MAIOR módulo de carga, de "
            f"{milhar(cargas.n)} retidas — escolha determinística, com empate resolvido pelo nome; "
            f"nenhuma amostragem aleatória. Setas: {legenda_das_setas(cargas, indices)}"
        ),
        matriz="a matriz ingerida, antes de qualquer correção",
        metodo=(
            f"{PRETRATAMENTO}. Os pontos estão em unidades de score e as setas em unidades de "
            f"carga multiplicadas por {decimal(_fator_das_setas(escores, cargas, indices))} para "
            "caberem no mesmo eixo: o comprimento delas é comparável entre si e NUNCA com os "
            f"pontos. E {SINAL_ARBITRARIO}"
        ),
    )


def legenda_das_setas(cargas: Cargas, indices: np.ndarray) -> str:
    """A correspondência entre o número de cada seta e a feature: `"1 = 2.25_810.7739m/z; ..."`.

    A ordem é a de `setas_do_biplot`, decrescente por módulo de carga: a seta 1 é a da feature
    de maior carga.
    """
    return "; ".join(
        f"{posicao + 1} = {cargas.features[int(indice)]}" for posicao, indice in enumerate(indices)
    )


# ---------------------------------------------------------------------------
# 1. O scree
# ---------------------------------------------------------------------------


def figura_do_scree(
    *,
    variancia_explicada: Sequence[float] | np.ndarray,
    recorte: Recorte,
) -> Figure:
    """A fração da variância por componente, com a acumulada.

    Desenha todos os componentes que recebe. A série individual mostra se vale olhar o próximo
    componente; a acumulada mostra quanto do dado os componentes desenhados cobrem. As duas
    são frações da mesma variância total e usam o mesmo eixo.

    Nenhum limiar é desenhado: uma linha em 80% seria lida como critério, e não há critério
    medido para quantos componentes bastam.
    """
    fracoes = np.asarray(variancia_explicada, dtype=float)
    if fracoes.ndim != 1 or fracoes.size == 0:
        raise MatrizVaziaError(
            f"`variancia_explicada` tem forma {fracoes.shape} e a figura precisa de um vetor com "
            "ao menos um componente. Vetor vazio chega aqui quando a PCA não correu"
        )
    componentes = np.arange(1, fracoes.size + 1, dtype=float)
    acumulada = np.cumsum(np.nan_to_num(fracoes, nan=0.0))

    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    eixo.plot(componentes, fracoes, marker="o", label="variância do componente")
    eixo.plot(componentes, acumulada, marker="s", linestyle="--", label="variância acumulada")

    # Marcas inteiras, uma por componente: o `matplotlib` usaria marcas fracionárias com mais
    # de dez componentes.
    eixo.set_xticks(componentes.tolist())
    eixo.set_xlabel("componente principal")
    eixo.set_ylabel("fração da variância total")
    eixo.set_title(
        f"{TITULO_DO_SCREE}\n{_frase_do_scree(fracoes, acumulada)}", fontsize=TAMANHO_DA_FONTE
    )
    legenda_fora(eixo)
    aplica(figura, recorte)
    return figura


def _frase_do_scree(fracoes: np.ndarray, acumulada: np.ndarray) -> str:
    """A segunda linha do título: quanto os componentes desenhados somam.

    Uma curva que cai depressa parece cobrir o dado mesmo quando a soma é baixa.
    """
    total = float(acumulada[-1]) if acumulada.size else 0.0
    plural = "componente desenhado" if fracoes.size == 1 else "componentes desenhados"
    return (
        f"{milhar(int(fracoes.size))} {plural}, somando "
        f"{total * 100:.1f}% da variância total".replace(".", ",")
    )


# ---------------------------------------------------------------------------
# 2. Os scores, com a elipse e o QC como série própria
# ---------------------------------------------------------------------------


def figura_dos_scores(
    *,
    escores: Escores,
    variancia_explicada: Sequence[float] | np.ndarray,
    regiao: Elipse | ElipseAusente | RegioesDosGrupos,
    recorte: Recorte,
    estilos: EstilosDosGrupos | None = None,
) -> Figure:
    """PC1 contra PC2, com `qc`, `sample` e `validate` em séries distintas e a região de predição.

    São três séries, e não um artista com cor por ponto como no volcano, porque os pontos são
    amostras: não há vetor de nomes, e a identificação por clique devolve o rótulo da série.
    Série sem amostras não é desenhada nem entra na legenda.

    A elipse é uma linha, e não um `patch`: `patches.Ellipse` não entra em `ax.lines` nem em
    `ax.collections`, e ficaria invisível ao arnês de teste e à identificação por clique.

    Com `RegioesDosGrupos` a figura é pintada por grupo; ver `_desenha_por_grupo`. Nessa forma
    todo grupo de estudo tem entrada na legenda, inclusive o que não tem ponto finito. Quem
    escolhe a forma é `regiao_dos_scores`. `estilos` só vale para a figura por grupo.
    """
    fracoes = np.asarray(variancia_explicada, dtype=float)
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)

    maos: list[Artist] | None = None
    if isinstance(regiao, RegioesDosGrupos):
        desenhadas, maos = _desenha_por_grupo(eixo, escores, regiao, estilos)
    else:
        desenhadas = _desenha_por_tipo(eixo, escores, regiao)

    eixo.set_xlabel(rotulo_do_componente(0, _fracao(fracoes, 0)))
    eixo.set_ylabel(rotulo_do_componente(1, _fracao(fracoes, 1)))
    eixo.set_title(
        f"{TITULO_DOS_SCORES}\n{_frase_das_amostras(escores, desenhadas)}",
        fontsize=TAMANHO_DA_FONTE,
    )
    legenda_fora(eixo, maos)
    aplica(figura, recorte)
    return figura


def _sem_ponto(escores: Escores) -> MatrizVaziaError:
    """A recusa da figura sem ponto desenhável, igual nas duas formas da figura."""
    return MatrizVaziaError(
        f"há {escores.n} amostra(s) e nenhuma com PC1 e PC2 finitos: não há ponto a desenhar. "
        "Uma figura com o eixo montado e nada dentro lê-se como figura quebrada"
    )


def _desenha_por_tipo(eixo: Axes, escores: Escores, regiao: Elipse | ElipseAusente) -> int:
    """As três séries por tipo e a região única sobre as amostras de estudo."""
    paleta = cores_das_series()
    desenhadas = 0
    for tipo in ORDEM_DAS_SERIES:
        mascara = escores.mascara(tipo)
        quantas = int(mascara.sum())
        if quantas == 0:
            continue
        desenhadas += quantas
        eixo.plot(
            escores.x[mascara],
            escores.y[mascara],
            marker="o",
            linestyle="none",
            color=paleta[tipo],
            label=_rotulo_da_serie(tipo, quantas),
        )

    if desenhadas == 0:
        raise _sem_ponto(escores)

    _desenha_a_elipse(eixo, regiao)
    return desenhadas


def _desenha_por_grupo(
    eixo: Axes,
    escores: Escores,
    regioes: RegioesDosGrupos,
    estilos: EstilosDosGrupos | None,
) -> tuple[int, list[Artist]]:
    """Uma série por grupo de estudo, o QC e a validação por tipo, e a região de cada grupo.

    A ordem de desenho é a validação, os grupos na ordem canônica e o QC por último, para o QC
    ficar por cima.

    O quadro é fixado com os pontos, e só depois vêm as regiões. O `matplotlib` autoescala
    pelos dados de todas as linhas do eixo, e a região de um grupo pequeno, que pode ser
    muito maior que a nuvem, deixaria os pontos num canto. As regiões ficam inteiras no
    artista, e o eixo recorta o que passa do quadro.

    A entrada de cada grupo na legenda traz o `n`, com o tracejado ao lado do marcador quando
    o grupo tem região. Nenhum grupo some: o grupo sem região e o grupo cuja região contém o
    quadro inteiro dizem o motivo entre parênteses, numa segunda linha; o grupo sem ponto
    finito tem entrada sem marcador.

    O motivo vai na legenda, e não num texto dentro do eixo: o quadro é o dos pontos, e um
    texto ali os cobriria.
    """
    if escores.grupos is None or set(regioes.grupos) != set(escores.grupos_de_estudo):
        raise ValueError(
            f"as regiões são dos grupos {list(regioes.grupos)} e estes escores têm os grupos "
            f"{list(escores.grupos_de_estudo)}: a figura desenharia regiões de uma população que "
            "ela não mostra. As duas pontas saem de `regiao_dos_scores` sobre os MESMOS escores"
        )
    com_pontos = tuple(g for g in escores.grupos_de_estudo if escores.mascara_do_grupo(g).any())
    tracos, do_qc, da_validacao = _tracos_por_grupo(com_pontos, estilos)

    desenhadas = 0
    da_serie_de_validacao: Artist | None = None
    validacao = escores.mascara(TipoDeAmostra.VALIDACAO)
    if validacao.any():
        desenhadas += int(validacao.sum())
        da_serie_de_validacao = _serie(
            eixo, escores, validacao, da_validacao, TipoDeAmostra.VALIDACAO
        )
    for grupo in com_pontos:
        mascara = escores.mascara_do_grupo(grupo)
        cor, marcador = tracos[grupo]
        eixo.plot(
            escores.x[mascara],
            escores.y[mascara],
            marker=marcador,
            linestyle="none",
            color=cor,
            label=rotulo_do_grupo(grupo, int(mascara.sum())),
        )
        desenhadas += int(mascara.sum())
    da_serie_do_qc: Artist | None = None
    qc = escores.mascara(TipoDeAmostra.QC)
    if qc.any():
        desenhadas += int(qc.sum())
        da_serie_do_qc = _serie(eixo, escores, qc, do_qc, TipoDeAmostra.QC)

    if desenhadas == 0:
        raise _sem_ponto(escores)

    # Fixa o quadro dos pontos. `get_xlim` aplica o autoescalonamento pendente, que só viu
    # pontos, e `set_xlim` o desliga; as regiões desenhadas depois não mudam a vista.
    eixo.set_xlim(eixo.get_xlim())
    eixo.set_ylim(eixo.get_ylim())
    motivos = _desenha_as_regioes(eixo, regioes, tracos)

    # A legenda é montada depois das regiões, porque só então se sabe qual região contém o
    # quadro inteiro. A ordem é a do desenho. O motivo vai numa segunda linha da entrada: na
    # mesma linha ele alargaria a legenda e estreitaria o eixo. Nenhuma linha de motivo passa
    # de `COLUNAS_DO_MOTIVO`; o motivo longo é quebrado em várias.
    maos: list[Artist] = [] if da_serie_de_validacao is None else [da_serie_de_validacao]
    for item in regioes.por_grupo:
        rotulo = rotulo_do_grupo(item.grupo, item.n)
        if item.grupo in motivos:
            linhas_do_motivo = quebra(f"({motivos[item.grupo]})", COLUNAS_DO_MOTIVO)
            rotulo = "\n".join((rotulo, *linhas_do_motivo))
        cor, marcador = tracos.get(item.grupo, ("none", "none"))
        maos.append(
            Line2D(
                [],
                [],
                color=cor,
                marker=marcador,
                linestyle=_TRACO_DA_REGIAO if isinstance(item.regiao, Elipse) else "none",
                label=rotulo,
            )
        )
    if da_serie_do_qc is not None:
        maos.append(da_serie_do_qc)
    return desenhadas, maos


def _serie(
    eixo: Axes,
    escores: Escores,
    mascara: np.ndarray,
    traco: tuple[Any, str],
    tipo: TipoDeAmostra,
) -> Artist:
    """A série do QC ou da validação na figura por grupo, com a cor e o marcador recebidos."""
    cor, marcador = traco
    (linha,) = eixo.plot(
        escores.x[mascara],
        escores.y[mascara],
        marker=marcador,
        linestyle="none",
        color=cor,
        label=_rotulo_da_serie(tipo, int(mascara.sum())),
    )
    return linha


def _tracos_por_grupo(
    grupos: tuple[str, ...], estilos: EstilosDosGrupos | None
) -> tuple[dict[str, tuple[Any, str]], tuple[Any, str], tuple[Any, str]]:
    """A cor e o marcador de cada grupo, e os do QC e da validação, que nenhum grupo usa.

    Sem `estilos`, as cores saem do ciclo dos `rcParams`: os grupos levam as primeiras, na
    ordem canônica, e a validação e o QC as duas seguintes. As cores de `cores_das_series`
    não servem aqui, porque são as três primeiras do mesmo ciclo. Quando o ciclo dá a volta
    (com dez cores, a partir de nove grupos), a série de tipo que repetir a cor de um grupo
    troca de marcador. Entre os grupos, a cor se repete a partir do décimo primeiro.

    Com `estilos`, os grupos levam a escolha da pessoa, resolvida sobre a coorte, e o QC e a
    validação levam os dois primeiros pares que nem a coorte nem os grupos da figura usam. O
    ciclo não entra: o azul dele é quase igual ao azul de Okabe e Ito, e a validação sairia
    quase na cor do primeiro grupo. Com 63 ou 64 grupos na coorte não sobram dois pares
    livres, e o QC e a validação repetem um par.
    """
    if estilos is None:
        cores = cores_do_ciclo(len(grupos) + 2)
        tracos = {grupo: (cores[i], MARCADOR_SEM_ESTILO) for i, grupo in enumerate(grupos)}
        dos_grupos = cores[: len(grupos)]

        def sem_repetir(cor: Any, marcador_na_volta: str) -> tuple[Any, str]:
            repete = any(same_color(cor, outra) for outra in dos_grupos)
            return cor, marcador_na_volta if repete else MARCADOR_SEM_ESTILO

        return (
            tracos,
            sem_repetir(cores[len(grupos) + 1], _MARCADOR_DO_QC_NA_VOLTA),
            sem_repetir(cores[len(grupos)], _MARCADOR_DA_VALIDACAO_NA_VOLTA),
        )
    do_grupo = estilos.para(grupos)
    tomados = {e.par for e in estilos.por_grupo.values()} | {e.par for e in do_grupo.values()}
    livres = [par for par in PARES_DE_ESTILO if par not in tomados] or list(PARES_DE_ESTILO)
    do_qc = EstiloDoGrupo(*livres[0])
    da_validacao = EstiloDoGrupo(*livres[1 % len(livres)])
    return (
        {grupo: (estilo.hex, estilo.marcador) for grupo, estilo in do_grupo.items()},
        (do_qc.hex, do_qc.marcador),
        (da_validacao.hex, da_validacao.marcador),
    )


def _desenha_as_regioes(
    eixo: Axes, regioes: RegioesDosGrupos, tracos: dict[str, tuple[Any, str]]
) -> dict[str, str]:
    """Desenha a região de cada grupo, na cor dele e por baixo dos pontos.

    Corre com o quadro já fixado. Devolve, por grupo, a frase que a entrada da legenda leva
    entre parênteses: por que não há região, ou que ela contém o quadro inteiro. O grupo cuja
    região aparece não entra no dicionário.
    """
    (x0, x1), (y0, y1) = eixo.get_xlim(), eixo.get_ylim()
    cantos = ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
    motivos: dict[str, str] = {}
    for item in regioes.por_grupo:
        if isinstance(item.regiao, ElipseAusente):
            motivos[item.grupo] = _motivo_sem_regiao_do_grupo(item.n, item.regiao)
            continue
        x, y = item.regiao.contorno()
        eixo.plot(
            x,
            y,
            linestyle=_TRACO_DA_REGIAO,
            color=tracos[item.grupo][0],
            zorder=_ORDEM_DAS_REGIOES,
            label=rotulo_da_regiao_do_grupo(item.grupo),
        )
        # A região é convexa: se contém os quatro cantos, contém o quadro inteiro, e o
        # contorno não aparece.
        if all(item.regiao.contem(cx, cy) for cx, cy in cantos):
            motivos[item.grupo] = NOTA_MAIOR_QUE_O_QUADRO
    return motivos


def _motivo_sem_regiao_do_grupo(n: int, ausente: ElipseAusente) -> str:
    """O motivo resumido que a legenda põe sob o `n` do grupo, como `"sem região: menos de 3"`.

    Não repete o `n`, que já está na entrada. Fora do caso de poucos pontos, usa o texto do
    motivo de `elipse_de_hotelling` até os dois pontos, sem adivinhar a causa.
    """
    if n < MINIMO_PARA_ELIPSE:
        return f"{NOTA_SEM_REGIAO}: menos de {MINIMO_PARA_ELIPSE}"
    return f"{NOTA_SEM_REGIAO}: {ausente.motivo.split(':', 1)[0]}"


def _desenha_a_elipse(eixo: Axes, regiao: Elipse | ElipseAusente) -> None:
    """Desenha a elipse, ou o texto que diz por que ela não existe.

    O texto leva `GID_REQUEBRAVEL`: o motivo é uma frase longa, que numa linha só sairia da
    figura. A quebra não é feita aqui porque a largura final do eixo só é conhecida em
    `recorte.aplica`, depois de a legenda de fora e o `tight_layout` o estreitarem.
    """
    if isinstance(regiao, ElipseAusente):
        texto = eixo.text(
            0.02,
            0.98,
            f"sem elipse:\n{regiao.motivo}",
            transform=eixo.transAxes,
            ha="left",
            va="top",
            fontsize=TAMANHO_DA_FONTE,
        )
        texto.set_gid(GID_REQUEBRAVEL)
        return
    x, y = regiao.contorno()
    confianca = f"{(1.0 - regiao.alfa) * 100:.0f}%".replace(".", ",")
    eixo.plot(
        x,
        y,
        linestyle=_TRACO_DA_REGIAO,
        label=f"Hotelling T² {confianca} — n = {milhar(regiao.n)}",
    )


def _frase_das_amostras(escores: Escores, desenhadas: int) -> str:
    """A contagem de injeções no título, com as não desenháveis à parte.

    Quando a figura é por tipo porque algum nome de grupo veio em branco, a frase diz quantas
    amostras; ver `Escores.grupos_em_branco`. Fica no título porque fala da população da
    figura, e o título é quebrado pela largura final.
    """
    fora = escores.n - desenhadas
    corpo = f"{milhar(desenhadas)} de {milhar(escores.n)} injeções"
    if fora:
        corpo = f"{corpo}; {milhar(fora)} sem PC1 ou PC2 finito, contada(s) e não descartada(s)"
    if escores.grupos_em_branco:
        quantas = escores.grupos_em_branco
        amostras = "amostra" if quantas == 1 else "amostras"
        corpo = (
            f"{corpo}; por tipo e não por grupo: {milhar(quantas)} {amostras} de estudo com o "
            "grupo em branco"
        )
    return corpo


def _fracao(fracoes: np.ndarray, indice: int) -> float:
    """A fração do componente `indice`, ou `nan` quando o vetor não a traz.

    Zero afirmaria que o componente não explica nada. Com `nan`, `rotulo_do_componente` diz
    que a variância não foi declarada.
    """
    return float(fracoes[indice]) if fracoes.size > indice else float("nan")


# ---------------------------------------------------------------------------
# 3. As cargas
# ---------------------------------------------------------------------------


def figura_das_cargas(
    *,
    cargas: Cargas,
    variancia_explicada: Sequence[float] | np.ndarray,
    recorte: Recorte,
) -> Figure:
    """Um ponto por feature: a carga em PC1 contra a carga em PC2.

    Desenha todas as features retidas de carga finita. Há três artistas de dado, a nuvem e as
    duas cruzes da origem, e só a nuvem leva `GID_DAS_FEATURES`: é ela que a identificação por
    clique indexa pelo vetor de nomes. Um clique numa cruz devolve o rótulo da cruz.

    As cruzes da origem são desenhadas em coordenadas de dado, com `plot`, e não com
    `axhline`. A linha de `axhline` tem transformação mista, com `get_xdata()` em fração do
    eixo, e o arnês de teste, que lê o dado do artista, acusaria pontos fora do limite.
    """
    fracoes = np.asarray(variancia_explicada, dtype=float)
    desenhaveis = cargas.desenhaveis
    n_desenhadas = int(desenhaveis.sum())
    if n_desenhadas == 0:
        raise MatrizVaziaError(
            f"há {cargas.n} feature(s) e nenhuma com carga finita nos dois componentes: não há "
            "ponto a desenhar. Uma figura com o eixo montado e nada dentro lê-se como quebrada"
        )

    px = cargas.x[desenhaveis]
    py = cargas.y[desenhaveis]
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    # As duas cruzes têm rótulos diferentes porque o arnês de teste indexa as séries por
    # rótulo: com o mesmo texto, uma delas ficaria invisível para ele.
    #
    # As duas cores são pedidas ao ciclo: `scatter` e `plot` consomem cicladores separados
    # que começam na mesma cor, e a cruz da origem sumiria dentro da nuvem.
    da_nuvem, da_origem = cores_do_ciclo(2)
    origem = eixo.plot(
        [float(px.min()), float(px.max())],
        [0.0, 0.0],
        linewidth=0.8,
        color=da_origem,
        label=f"{ROTULO_DA_ORIGEM} de PC2",
    )[0]
    eixo.plot(
        [0.0, 0.0],
        [float(py.min()), float(py.max())],
        linewidth=0.8,
        color=da_origem,
        label=f"{ROTULO_DA_ORIGEM} de PC1",
    )
    pontos = eixo.scatter(px, py, s=8, color=da_nuvem, label=f"{milhar(n_desenhadas)} features")
    # Dos três artistas de dado, só este tem um ponto por feature.
    pontos.set_gid(GID_DAS_FEATURES)

    eixo.set_xlabel(rotulo_do_componente(0, _fracao(fracoes, 0), prefixo="carga em "))
    eixo.set_ylabel(rotulo_do_componente(1, _fracao(fracoes, 1), prefixo="carga em "))
    eixo.set_title(
        f"{TITULO_DAS_CARGAS}\n{_frase_das_cargas(cargas, n_desenhadas)}",
        fontsize=TAMANHO_DA_FONTE,
    )
    # Uma entrada de legenda para as duas cruzes: os rótulos distintos servem ao arnês, e a
    # legenda não precisa repeti-los.
    origem.set_label(ROTULO_DA_ORIGEM)
    legenda_fora(eixo, [pontos, origem])
    aplica(figura, recorte)
    return figura


def _frase_das_cargas(cargas: Cargas, n_desenhadas: int) -> str:
    fora = cargas.n - n_desenhadas
    corpo = f"{milhar(n_desenhadas)} de {milhar(cargas.n)} features retidas pela PCA"
    if fora:
        return f"{corpo}; {milhar(fora)} com carga não finita, contada(s) e não descartada(s)"
    return corpo


# ---------------------------------------------------------------------------
# 4. O biplot
# ---------------------------------------------------------------------------


def setas_do_biplot(cargas: Cargas, n_setas: int) -> np.ndarray:
    """Os índices das `n_setas` features de maior módulo de carga, em ordem decrescente.

    O empate é resolvido pelo nome, para a figura não mudar entre execuções da mesma matriz.
    Só entram as features de carga finita.
    """
    if n_setas < 1:
        raise ValueError(
            f"`n_setas` é {n_setas}: um biplot sem seta nenhuma é um gráfico de scores, e há um. "
            "O número é recorte de quem chama e não tem padrão — um padrão aqui seria número "
            "inventado por quem não pode justificá-lo"
        )
    desenhaveis = np.nonzero(cargas.desenhaveis)[0]
    modulos = np.hypot(cargas.x[desenhaveis], cargas.y[desenhaveis])
    ordem = sorted(
        range(desenhaveis.size),
        key=lambda i: (-float(modulos[i]), cargas.features[int(desenhaveis[i])]),
    )
    escolhidos: np.ndarray = desenhaveis[ordem[:n_setas]]
    return escolhidos


def figura_do_biplot(
    *,
    escores: Escores,
    cargas: Cargas,
    n_setas: int,
    variancia_explicada: Sequence[float] | np.ndarray,
    recorte: Recorte,
) -> Figure:
    """As amostras e as `n_setas` features de maior carga, no mesmo par de eixos.

    As cargas chegam todas, e o construtor escolhe as `n_setas` de maior módulo. O rodapé diz
    quantas de quantas.

    Esta figura não usa identificação por clique: com amostras e features no mesmo eixo, o
    índice de um ponto de amostra cairia no vetor de nomes de feature. O vetor de nomes vai
    vazio, cada seta leva um número, e a lista de números e nomes fica no rodapé.

    As setas partem da origem, que é o zero da carga, e não do centroide das amostras.

    `n_setas` tem custo de redesenho, dominado pelo texto: sem rótulos e legenda o redesenho
    não depende do número de setas, e com eles passa de 100 ms a partir de cerca de 22
    setas. O limite não é imposto aqui, porque o número é escolha do chamador.
    """
    fracoes = np.asarray(variancia_explicada, dtype=float)
    indices = setas_do_biplot(cargas, n_setas)
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    paleta = cores_das_series()

    das_amostras: list[Artist] = []
    desenhadas = 0
    for tipo in ORDEM_DAS_SERIES:
        mascara = escores.mascara(tipo)
        quantas = int(mascara.sum())
        if quantas == 0:
            continue
        desenhadas += quantas
        das_amostras.append(
            eixo.plot(
                escores.x[mascara],
                escores.y[mascara],
                marker="o",
                linestyle="none",
                color=paleta[tipo],
                label=_rotulo_da_serie(tipo, quantas),
            )[0]
        )

    if desenhadas == 0:
        raise MatrizVaziaError(
            f"há {escores.n} amostra(s) e nenhuma com PC1 e PC2 finitos: um biplot sem amostra "
            "nenhuma é um gráfico de cargas, e há um"
        )

    fator = _fator_das_setas(escores, cargas, indices)
    cor_da_seta = cores_do_ciclo(len(ORDEM_DAS_SERIES) + 1)[-1]
    for posicao, indice in enumerate(indices):
        ponta_x = float(cargas.x[indice]) * fator
        ponta_y = float(cargas.y[indice]) * fator
        # Cada seta tem o nome da feature como rótulo, porque o arnês de teste indexa as
        # séries por rótulo.
        eixo.plot(
            [0.0, ponta_x],
            [0.0, ponta_y],
            linewidth=0.9,
            color=cor_da_seta,
            label=cargas.features[int(indice)],
        )
        # O número fica sobre a seta, a uma fração do comprimento que desce com a ordem. As
        # setas apontam quase todas na mesma direção, e com todos os números nas pontas eles
        # se sobreporiam. A de maior carga fica na ponta e as seguintes mais para dentro.
        fracao = 1.0 - (1.0 - FRACAO_MINIMA_DO_ROTULO) * posicao / max(1, indices.size - 1)
        eixo.annotate(
            str(posicao + 1),
            xy=(ponta_x * fracao, ponta_y * fracao),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=TAMANHO_DA_FONTE,
        )

    eixo.set_xlabel(rotulo_do_componente(0, _fracao(fracoes, 0)))
    eixo.set_ylabel(rotulo_do_componente(1, _fracao(fracoes, 1)))
    eixo.set_title(
        f"{TITULO_DO_BIPLOT}\n{milhar(int(indices.size))} de {milhar(cargas.n)} features "
        f"retidas, e {milhar(desenhadas)} de {milhar(escores.n)} injeções",
        fontsize=TAMANHO_DA_FONTE,
    )
    # Uma entrada de legenda para todas as setas, por um artista que não entra no eixo. Uma
    # entrada por seta daria uma legenda maior que a figura.
    prova = Line2D([], [], color=cor_da_seta, label=f"{ROTULO_DAS_SETAS} ({milhar(indices.size)})")
    legenda_fora(eixo, [*das_amostras, prova])
    aplica(figura, recorte)
    return figura


# Onde fica o número da última seta, em fração do comprimento dela. Os números se distribuem
# entre essa fração e a ponta, em ordem decrescente de carga. Com 0,5 eles se separam sem
# chegar perto da origem, onde as setas se cruzam.
FRACAO_MINIMA_DO_ROTULO: Final = 0.5

# Fração do alcance dos scores que a seta mais longa ocupa. É menor que 1 para o número da
# seta não sair do eixo.
FRACAO_DA_SETA_MAIOR: Final = 0.8


def _fator_das_setas(escores: Escores, cargas: Cargas, indices: np.ndarray) -> float:
    """O fator que multiplica as cargas para caberem no eixo dos scores.

    É derivado do dado: a razão entre o maior módulo de score desenhado e o maior módulo de
    carga desenhada, vezes `FRACAO_DA_SETA_MAIOR`. Um fator fixo deixaria as setas invisíveis
    numa matriz e fora do eixo em outra. O rodapé declara o valor.

    Com carga máxima nula o fator é 1.
    """
    finitos = np.isfinite(escores.x) & np.isfinite(escores.y)
    alcance = (
        float(np.hypot(escores.x[finitos], escores.y[finitos]).max()) if finitos.any() else 0.0
    )
    maior_carga = (
        float(np.hypot(cargas.x[indices], cargas.y[indices]).max()) if indices.size else 0.0
    )
    if maior_carga <= 0.0:
        return 1.0
    return alcance * FRACAO_DA_SETA_MAIOR / maior_carga

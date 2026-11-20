"""O volcano: fold change contra significância entre dois grupos.

O eixo x é o `log2` da razão de medianas e o eixo y é `-log10` do p bruto. O primeiro grupo
do par é o numerador, então `(A, B)` e `(B, A)` dão imagens espelhadas em x, e o rótulo do
eixo nomeia as duas medianas na ordem da razão. Três coisas diferem do MetaboAnalyst, de
propósito. O eixo y é sempre o p bruto e é rotulado assim, com a cor classificando pelo q;
lá o eixo passa a `-log10(FDR)` e o rótulo continua dizendo p. Feature de fold change
indefinido não é desenhada e é contada, em vez de ir para a borda pelos sentinelas
`fc.log = ±99`, onde pareceria um efeito enorme. E o teste é Welch ou Mann-Whitney, conforme
a normalidade dos resíduos, enquanto os dois modos do volcano deles usam um p moderado; o
rodapé declara a divergência.

Há uma única dispersão, com todos os pontos desenháveis na ordem em que chegam, porque a
identificação por clique indexa o vetor de nomes pela posição do ponto dentro do artista. A
classe está na cor por ponto e nas entradas de legenda. Classificar e desenhar são
separados: a significância é contada sobre todas as features testadas, pelo q, e só tem
ponto quem tem `log2fc` e `-log10(p)` finitos; `ContagensDoVolcano` traz as duas partições.
Num estudo pequeno é comum nenhuma feature sobreviver à correção de múltiplos testes: a
figura mostra a nuvem inteira com `0` na legenda, e o rodapé traz o `n` de cada grupo e a
correção, que é o que separa 'não há efeito' de 'não há poder'.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import norm

from refino.plots.base import (
    GID_DAS_FEATURES,
    NAO_DECLARADA,
    MatrizVaziaError,
    cores_do_ciclo,
    inteiro_do_fio,
    nova_figura,
    numero_do_fio,
    texto_do_fio,
    vetor_do_fio,
    vetor_do_fio_ou_nucleo,
)
from refino.plots.recorte import TAMANHO_DA_FONTE, Recorte, aplica, decimal, milhar

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "DIVERGENCIA",
    "K_DAS_ESCOLHIDAS",
    "MANN_WHITNEY",
    "MOTIVOS_SEM_INTERVALO",
    "NAO_DECLARADA",
    "ROTULO_DAS_BARRAS",
    "ROTULO_DA_SERIE",
    "ROTULO_DE_Y",
    "SEM_MOTIVO_NO_FIO",
    "TITULO",
    "WELCH",
    "Classe",
    "ContagensDoVolcano",
    "Contraste",
    "IntervaloAusente",
    "IntervalosEscolhidos",
    "PontosDoVolcano",
    "contraste_do_fio",
    "cor_das_barras",
    "cores_das_classes",
    "desenha_as_barras",
    "figura_do_volcano",
    "frase_da_populacao",
    "frase_das_barras",
    "frase_do_vazio",
    "intervalos_escolhidos",
    "nome_do_teste",
    "nomes_do_fio",
    "pontos_do_volcano",
    "prosa_da_normalidade",
    "prova_das_barras",
    "recorte_do_volcano",
    "rotulo_da_classe",
    "rotulo_das_barras",
    "rotulo_de_x",
]

TITULO: Final = "Volcano"

# O eixo é o p bruto, e o rótulo diz isso. O q de Benjamini-Hochberg decide a cor, e o rodapé
# o declara.
ROTULO_DE_Y: Final = "-log10(p-valor bruto)"

# O rótulo do artista da dispersão. Não vai à legenda, que mostra as classes; é o texto que a
# identificação por clique mostra quando o vetor de nomes não cobre a posição.
ROTULO_DA_SERIE: Final = "features desenhadas"

WELCH: Final = "t de Welch (variâncias desiguais), bilateral"

MANN_WHITNEY: Final = "Mann-Whitney U, bilateral"

# A divergência em relação ao MetaboAnalyst, escrita no rodapé. É curta porque
# `recorte.aplica` recusa o rodapé que precise de mais de `RESERVA_MAXIMA` da altura.
DIVERGENCIA: Final = (
    "nenhum modo do volcano do MetaboAnalyst reproduz este p: os dois usam um p moderado"
)

# Quantas features ganham a barra do intervalo do fold change: as `k` de menor p. Com barra
# em todas, elas cobririam a figura.
K_DAS_ESCOLHIDAS: Final = 20

# O rótulo do artista das barras, sem contagem: é a chave pela qual o arnês de teste e a
# identificação o encontram. A contagem fica na entrada de legenda.
ROTULO_DAS_BARRAS: Final = "intervalo do fold change"

# Os motivos de intervalo indefinido, com os códigos de `FoldChangeIntervalUndefined` e o
# texto que o rodapé escreve. Um código fora deste vocabulário é escrito cru.
MOTIVOS_SEM_INTERVALO: Final[dict[str, str]] = {
    "non_finite_value": "com célula não finita num dos grupos",
    "order_statistic_not_positive": "com valor de ordem não positivo num dos grupos",
    "median_over_imputed_values": "com mediana sobre valor imputado",
}

# Limite ausente sem motivo ao lado: a resposta não diz por quê, e a figura não inventa.
SEM_MOTIVO_NO_FIO: Final = "sem limite e sem motivo na resposta"


class Classe(StrEnum):
    """A classe de um ponto desenhado.

    `SEM_DIRECAO` é a feature que sobrevive ao FDR com `log2fc` exatamente zero. Ela não é
    `NAO`, porque é significativa, nem `ABAIXO` ou `ACIMA`, porque a razão de medianas não tem
    sentido. A legenda só a mostra quando existe.
    """

    ABAIXO = "abaixo"
    ACIMA = "acima"
    SEM_DIRECAO = "sem direção"
    NAO = "não significativa"


# A ordem da legenda, que é a do MetaboAnalyst (`Sig.Down | Sig.Up | Unsig.`), com a classe
# sem direção entre as duas significativas.
ORDEM_DAS_CLASSES: Final = (Classe.ABAIXO, Classe.SEM_DIRECAO, Classe.ACIMA, Classe.NAO)


@dataclass(frozen=True)
class ContagensDoVolcano:
    """Quantas features caíram em cada desfecho, em duas partições de `n_testadas`.

    - classificação: `sobreviventes + nao_significativas + sem_qvalue`, decidida por q finito
      e `q < alfa`;
    - desenho: `desenhadas + nao_desenhadas`, decidida por `log2fc` e `-log10(p)` finitos.

    `sobreviventes` conta sobre todas as features, com o mesmo predicado estrito que a API usa
    para `n_significant_fdr`, e os dois números têm de coincidir. Contado só sobre as
    desenháveis, ele perderia o p igual a zero e o sobrevivente sem `log2fc`.

    `abaixo + acima + sem_direcao + nao_significativas_desenhadas` soma `desenhadas`, e são
    essas quatro que vão à legenda, porque a legenda descreve a nuvem. A última inclui a
    feature de q indefinido, que é desenhada com a cor de `Classe.NAO`.

    `sem_fold_change` e `sem_pvalue_desenhavel` somam `nao_desenhadas`. A segunda conta só as
    que têm fold change, para uma feature sem os dois valores não entrar duas vezes.

    `pvalue_indefinido` e `pvalue_zero` são contados à parte, sobre todas as features. São as
    duas causas de p não desenhável e são fatos opostos: o teste não devolveu nada, ou
    devolveu exatamente zero.

    `sobreviventes_sem_fold_change` permite ao rodapé decompor os sobreviventes sem ponto por
    causa, sem usar contagens que são da matriz inteira.
    """

    # A partição da classificação.
    sobreviventes: int
    nao_significativas: int
    sem_qvalue: int
    # A partição do desenho.
    desenhadas: int
    nao_desenhadas: int
    # As quatro classes dentro do desenho, que a legenda conta.
    abaixo: int
    acima: int
    sem_direcao: int
    nao_significativas_desenhadas: int
    # As duas razões de não desenhar, que somam `nao_desenhadas`.
    sem_fold_change: int
    sem_pvalue_desenhavel: int
    # Contadas à parte, sobre todas as features; nenhuma é classe.
    pvalue_indefinido: int
    pvalue_zero: int
    sobreviventes_sem_fold_change: int

    @property
    def sobreviventes_desenhados(self) -> int:
        """Os sobreviventes que têm ponto: a soma das três classes de direção."""
        return self.abaixo + self.acima + self.sem_direcao

    @property
    def sobreviventes_sem_ponto(self) -> int:
        """Quantas sobreviveram ao FDR e não têm ponto na nuvem."""
        return self.sobreviventes - self.sobreviventes_desenhados

    @property
    def sobreviventes_sem_pvalue_desenhavel(self) -> int:
        """Os sobreviventes sem ponto que têm fold change e cujo p não é desenhável.

        É derivada, para as duas parcelas somarem exatamente `sobreviventes_sem_ponto`.
        """
        return self.sobreviventes_sem_ponto - self.sobreviventes_sem_fold_change


@dataclass(frozen=True)
class Contraste:
    """O par de grupos e como ele foi medido: o que a figura declara e não consegue derivar.

    `grupo_a` é o numerador do fold change.

    As duas procedências são campos separados porque os eixos vêm de matrizes diferentes: o p
    sai da matriz processada e o fold change da matriz anterior à transformação.
    `matrizes_diferem` diz se a transformação de fato alterou a matriz. Com `transformation` e
    `scaling` em `none` as duas são a mesma, e a figura não declara duas procedências.
    """

    grupo_a: str
    grupo_b: str
    n_a: int
    n_b: int
    parametrico: bool
    alfa: float
    correcao: str
    rota_da_normalidade: str
    shapiro_pvalue: float | None
    n_residuos: int | None
    matriz_do_pvalue: str
    matriz_do_fold_change: str
    matrizes_diferem: bool


@dataclass(frozen=True)
class PontosDoVolcano:
    """Os pontos desenháveis, na ordem em que as features chegam, com as contagens da população.

    `nomes`, `x` e `y` têm o mesmo comprimento e descrevem as features com `log2fc` e
    `-log10(p)` finitos. `nomes` é o vetor que o painel recebe para a identificação por clique.

    `menor_q` é o mínimo sobre as features testadas, e não sobre as desenhadas: a feature de
    menor q pode ser uma das que ficaram sem ponto. É `None` quando nenhum q é finito.
    """

    nomes: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    classes: tuple[Classe, ...]
    n_testadas: int
    contagens: ContagensDoVolcano
    menor_q: float | None


@dataclass(frozen=True)
class IntervalosEscolhidos:
    """As `k` features de menor p, com o intervalo de cada uma no nível ajustado pela seleção.

    O intervalo que chega é marginal: vale para cada feature olhada sozinha. Olhado só nas
    escolhidas por p ele é otimista, porque a escolha favorece as features em que o acaso foi
    extremo. O nível que mantém em `fcr` a taxa de intervalos que não cobrem, entre as `R`
    escolhidas de `m`, é `1 - R·fcr/m` (Benjamini e Yekutieli, 2005), com `fcr` igual ao
    complemento do nível marginal.

    `R` conta as escolhidas com intervalo, e não as `k`, porque a seleção efetiva é 'das `k` de
    menor p, as que têm intervalo'. `m` conta as testadas, inclusive as sem p.

    O intervalo de Bonett-Price é simétrico em log, `centro ± z·√(SE₁² + SE₂²)`, e o
    erro-padrão não depende do nível. Para mudar de nível basta multiplicar a meia-largura
    pela razão dos dois `z`, sem recalcular.

    `com_intervalo + soma(sem_intervalo) == escolhidas`: cada escolhida tem barra ou tem um
    motivo. `empatadas_fora` conta as features com p igual ao da última escolhida que ficaram
    de fora. O p de Mann-Whitney é discreto, o empate na fronteira é frequente, e o desempate
    é a ordem da matriz.
    """

    k: int
    m: int
    escolhidas: int
    indices: np.ndarray
    """As posições, entre as testadas, das escolhidas com intervalo, na ordem do p."""
    pvalue: np.ndarray
    centro: np.ndarray
    inferior: np.ndarray
    superior: np.ndarray
    """Os limites no nível ajustado, em `log2`, alinhados com `indices`."""
    nivel_marginal: float
    nivel: float
    sem_intervalo: tuple[tuple[str, int], ...]
    empatadas_fora: int

    @property
    def com_intervalo(self) -> int:
        """Quantas escolhidas têm intervalo. É o `R` da fórmula do nível."""
        return int(self.indices.size)

    @property
    def fcr(self) -> float:
        """A taxa de intervalos que não cobrem, entre as escolhidas: 1 menos o nível marginal."""
        return 1.0 - self.nivel_marginal


@dataclass(frozen=True)
class IntervaloAusente:
    """A resposta não trouxe o que permite ajustar o intervalo.

    É diferente de nenhuma feature ter intervalo: a figura declara por que não há barra.
    """

    motivo: str


# ---------------------------------------------------------------------------
# A fronteira: o que chega do corpo JSON não passou pelo mypy
# ---------------------------------------------------------------------------


def nomes_do_fio(valores: Any) -> tuple[str, ...]:
    """A lista de nomes de feature do corpo, com o que não é texto convertido em `""`.

    Não descarta: um nome ilegível tem de continuar na posição dele, senão o vetor se
    desalinha dos numéricos e a identificação devolve o nome do vizinho.
    """
    if not isinstance(valores, list):
        return ()
    return tuple("" if isinstance(v, bool) or not isinstance(v, str) else v for v in valores)


def contraste_do_fio(
    *,
    grupo_a: Any,
    grupo_b: Any,
    n_grupo_a: Any,
    n_grupo_b: Any,
    parametrico: Any,
    alfa_fdr: Any,
    correcao: Any,
    rota_da_normalidade: Any,
    shapiro_pvalue: Any,
    n_residuos: Any,
    matriz_do_pvalue: Any,
    matriz_do_fold_change: Any,
    matrizes_diferem: Any,
) -> Contraste:
    """O `Contraste` a partir dos campos da raiz `volcano`, com guarda em cada um.

    Um campo pode chegar `null`, e a raiz inteira pode faltar quando a execução correu sem par
    de grupos. O que falta vira `NAO_DECLARADA` nas procedências e `0` nas contagens, e
    `recorte_do_volcano` recusa por população vazia quando não há grupo.

    Recebe os campos um a um, e não o dicionário, para que a leitura de cada chave do corpo
    fique na janela, onde o teste de contrato com a API confere as chaves contra o esquema.
    """
    return Contraste(
        grupo_a=texto_do_fio(grupo_a, ""),
        grupo_b=texto_do_fio(grupo_b, ""),
        n_a=inteiro_do_fio(n_grupo_a),
        n_b=inteiro_do_fio(n_grupo_b),
        parametrico=parametrico is True,
        alfa=numero_do_fio(alfa_fdr),
        correcao=texto_do_fio(correcao, NAO_DECLARADA),
        rota_da_normalidade=texto_do_fio(rota_da_normalidade, ""),
        shapiro_pvalue=_opcional(shapiro_pvalue),
        n_residuos=_inteiro_opcional(n_residuos),
        matriz_do_pvalue=texto_do_fio(matriz_do_pvalue, NAO_DECLARADA),
        matriz_do_fold_change=texto_do_fio(matriz_do_fold_change, NAO_DECLARADA),
        matrizes_diferem=matrizes_diferem is True,
    )


def _opcional(valor: Any) -> float | None:
    """Um `float` finito, ou `None`. O `None` não vira `nan` nem zero.

    `shapiro_pvalue` é `None` quando o teste de Shapiro não correu, e um número ali faria 'não
    medi' parecer 'medi e deu isto'.
    """
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return None
    return None if not np.isfinite(valor) else float(valor)


def _inteiro_opcional(valor: Any) -> int | None:
    """Um `int` não negativo, ou `None`, pela mesma razão de `_opcional`."""
    if isinstance(valor, bool) or not isinstance(valor, int):
        return None
    return valor if valor >= 0 else None


# ---------------------------------------------------------------------------
# A partição da população, usada pelo construtor e pelo painel
# ---------------------------------------------------------------------------


def pontos_do_volcano(
    *,
    features: Any,
    log2fc: Any,
    pvalue: Any,
    qvalue_fdr: Any,
    n_features_testadas: Any,
    alfa: float,
) -> PontosDoVolcano:
    """Transforma os quatro vetores do corpo nos pontos desenháveis e nas contagens.

    O painel usa o mesmo resultado para obter os nomes desenhados, e assim a seleção dos
    pontos é feita num lugar só.

    O `y` é `-log10(p)` e a classificação é pelo q. Nenhuma altura corresponde ao corte,
    porque o q depende do posto do p na tabela inteira; por isso não há linha de corte.

    Levanta `MatrizVaziaError` quando não veio feature nenhuma. Nenhuma feature desenhável é
    outro caso: é fato do dado, e a figura o declara.

    O denominador é o `n_features_testadas` declarado, e não o comprimento dos vetores: é o
    número que a figura publica, e um vetor truncado publicaria uma população menor. Quando
    não vem, vale o maior dos comprimentos. 'Testadas' é o nome certo aqui, ao contrário da
    ANOVA: o teste univariado corre sobre todas as colunas, sem filtro por feature.

    A significância é calculada sobre o vetor inteiro, e o que é desenhável sobre os dois
    eixos. Ver `ContagensDoVolcano`.
    """
    nomes = nomes_do_fio(features)
    fc = vetor_do_fio(log2fc)
    p = vetor_do_fio(pvalue)
    q = vetor_do_fio(qvalue_fdr)

    declarado = inteiro_do_fio(n_features_testadas)
    n_testadas = declarado if declarado > 0 else max(len(nomes), fc.size, p.size, q.size)
    if n_testadas == 0:
        raise MatrizVaziaError(
            "o volcano não recebeu feature nenhuma: não há população para desenhar nem para "
            "declarar. Sem par de grupos a execução não corre o teste univariado, e é isso que é "
            "verdade — figura em branco seria indistinguível de figura quebrada"
        )
    nomes = _alinha(nomes, n_testadas)
    fc, p, q = (_estica(v, n_testadas) for v in (fc, p, q))

    with np.errstate(divide="ignore", invalid="ignore"):
        y = -np.log10(p)

    com_fold_change = np.isfinite(fc)
    com_qvalue = np.isfinite(q)
    # Para haver ponto, os dois eixos têm de ter valor finito.
    desenhavel = com_fold_change & np.isfinite(y)
    # A classificação corre sobre o vetor inteiro, com o mesmo predicado estrito que a API usa
    # para `n_significant_fdr`.
    significativa = com_qvalue & (q < alfa)
    # Conta só as que têm fold change, para uma feature sem p e sem fold change não entrar nas
    # duas contagens.
    sem_pvalue_desenhavel = int((com_fold_change & ~np.isfinite(y)).sum())

    classes = tuple(
        _classe(bool(significativa[i]), float(fc[i]))
        for i in range(n_testadas)
        if bool(desenhavel[i])
    )
    contagens = ContagensDoVolcano(
        sobreviventes=int(significativa.sum()),
        nao_significativas=int((com_qvalue & ~significativa).sum()),
        sem_qvalue=int((~com_qvalue).sum()),
        desenhadas=int(desenhavel.sum()),
        nao_desenhadas=int((~desenhavel).sum()),
        abaixo=sum(1 for c in classes if c is Classe.ABAIXO),
        acima=sum(1 for c in classes if c is Classe.ACIMA),
        sem_direcao=sum(1 for c in classes if c is Classe.SEM_DIRECAO),
        nao_significativas_desenhadas=sum(1 for c in classes if c is Classe.NAO),
        sem_fold_change=int((~com_fold_change).sum()),
        sem_pvalue_desenhavel=sem_pvalue_desenhavel,
        pvalue_indefinido=int((~np.isfinite(p)).sum()),
        pvalue_zero=int((p == 0.0).sum()),
        sobreviventes_sem_fold_change=int((significativa & ~com_fold_change).sum()),
    )
    return PontosDoVolcano(
        nomes=tuple(nomes[i] for i in range(n_testadas) if bool(desenhavel[i])),
        x=fc[desenhavel],
        y=y[desenhavel],
        classes=classes,
        n_testadas=n_testadas,
        contagens=contagens,
        menor_q=float(np.min(q[np.isfinite(q)])) if bool(np.isfinite(q).any()) else None,
    )


def _classe(significativa: bool, fold_change: float) -> Classe:
    """A classe de um ponto desenhável."""
    if not significativa:
        return Classe.NAO
    if fold_change < 0.0:
        return Classe.ABAIXO
    if fold_change > 0.0:
        return Classe.ACIMA
    return Classe.SEM_DIRECAO


def _alinha(nomes: tuple[str, ...], n: int) -> tuple[str, ...]:
    """Completa ou corta o vetor de nomes para o comprimento da população.

    Completa com `""` em vez de levantar: um nome ausente degrada a identificação, e não a
    figura.
    """
    if len(nomes) == n:
        return nomes
    return (*nomes[:n], *("",) * max(0, n - len(nomes)))


def _estica(valores: np.ndarray, n: int) -> np.ndarray:
    """Completa com `nan` ou corta um vetor numérico para o comprimento da população.

    O que falta vira indefinido, e indefinido é contado e declarado. Levantar trocaria uma
    figura que diz '1 sem p' por um painel sem figura.
    """
    if valores.size == n:
        return valores
    saida = np.full(n, np.nan, dtype=float)
    saida[: min(n, valores.size)] = valores[:n]
    return saida


# ---------------------------------------------------------------------------
# O intervalo do fold change nas escolhidas, comum ao volcano e ao fold change por feature
# ---------------------------------------------------------------------------


def intervalos_escolhidos(
    *,
    pvalue: Any,
    log2fc_ci_lower: Any,
    log2fc_ci_upper: Any,
    fold_change_ci_undefined: Any,
    nivel_do_intervalo: Any,
    n_testadas: int,
    k: int = K_DAS_ESCOLHIDAS,
) -> IntervalosEscolhidos | IntervaloAusente:
    """As `k` de menor p, com o intervalo de cada uma levado ao nível ajustado pela seleção.

    É usada pelas duas figuras que desenham o `log2fc`, para as duas abas da mesma execução
    porem barras nas mesmas features, com o mesmo nível.

    `n_testadas` vem da figura: é a população que ela declara e o `m` da fórmula, e os vetores
    são ajustados a esse comprimento.

    `nivel_do_intervalo` é o nível com que os limites recebidos foram calculados. Sem ele, ou
    fora de `(0, 1)`, o resultado é `IntervaloAusente`: ajustar a partir de uma constante
    daqui afirmaria um nível que ninguém calculou.

    O desempate é a ordem da matriz (`argsort` estável), e as empatadas que ficam de fora são
    contadas.
    """
    marginal = _opcional(nivel_do_intervalo)
    if marginal is None or not 0.0 < marginal < 1.0:
        return IntervaloAusente(
            "a resposta não trouxe o nível do intervalo de Bonett-Price, e sem ele os limites não "
            "têm como ser ajustados à seleção"
        )
    p = _estica(vetor_do_fio_ou_nucleo(pvalue), n_testadas)
    inferior = _estica(vetor_do_fio_ou_nucleo(log2fc_ci_lower), n_testadas)
    superior = _estica(vetor_do_fio_ou_nucleo(log2fc_ci_upper), n_testadas)
    codigos = _codigos_do_fio(fold_change_ci_undefined, n_testadas)

    com_p = np.isfinite(p)
    ordem = np.argsort(np.where(com_p, p, np.inf), kind="stable")
    escolhidas = ordem[: min(k, int(com_p.sum()))]
    definido = np.isfinite(inferior) & np.isfinite(superior)
    indices = escolhidas[definido[escolhidas]]

    fcr = 1.0 - marginal
    nivel = 1.0 - indices.size * fcr / n_testadas if n_testadas else marginal
    # Os dois `z` pela mesma expressão do núcleo (`fold_change_interval`), para o intervalo
    # ajustado coincidir com o recalculado.
    fator = float(norm.ppf(1.0 - (1.0 - nivel) / 2.0) / norm.ppf(1.0 - fcr / 2.0))
    centro = (inferior[indices] + superior[indices]) / 2.0
    meia = (superior[indices] - inferior[indices]) / 2.0 * fator

    ultima = p[escolhidas[-1]] if escolhidas.size else np.nan
    return IntervalosEscolhidos(
        k=k,
        m=n_testadas,
        escolhidas=int(escolhidas.size),
        indices=indices,
        pvalue=p[indices],
        centro=centro,
        inferior=centro - meia,
        superior=centro + meia,
        nivel_marginal=marginal,
        nivel=nivel,
        sem_intervalo=_motivos_das_escolhidas(
            [codigos[int(j)] for j in escolhidas if not bool(definido[j])]
        ),
        empatadas_fora=int((com_p & (p == ultima)).sum() - (p[escolhidas] == ultima).sum()),
    )


def _codigos_do_fio(valores: Any, n: int) -> tuple[str | None, ...]:
    """Os códigos de motivo, um por feature, com o que não é texto convertido em `None`.

    São ajustados ao comprimento da população, como os vetores numéricos, para o motivo de
    uma feature não cair na vizinha.
    """
    lidos = list(valores) if isinstance(valores, (list, tuple)) else []
    codigos = tuple(v if isinstance(v, str) else None for v in lidos[:n])
    return (*codigos, *(None,) * (n - len(codigos)))


def _motivos_das_escolhidas(codigos: list[str | None]) -> tuple[tuple[str, int], ...]:
    """A contagem por motivo das escolhidas sem intervalo, na ordem dos motivos do núcleo.

    Um código fora do vocabulário é escrito cru, em vez de receber o nome de outro motivo.
    """
    prosas = [
        SEM_MOTIVO_NO_FIO
        if codigo is None
        else MOTIVOS_SEM_INTERVALO.get(codigo, f"com motivo não reconhecido ({codigo})")
        for codigo in codigos
    ]
    ordem = [
        *MOTIVOS_SEM_INTERVALO.values(),
        *sorted(set(prosas) - set(MOTIVOS_SEM_INTERVALO.values())),
    ]
    return tuple((prosa, prosas.count(prosa)) for prosa in ordem if prosa in prosas)


# ---------------------------------------------------------------------------
# O que a figura diz sobre si mesma
# ---------------------------------------------------------------------------


def rotulo_de_x(grupo_a: str, grupo_b: str) -> str:
    """O rótulo do eixo x: o `log2` da razão, com o numerador nomeado.

    `(A, B)` e `(B, A)` desenham imagens espelhadas em x, e um `log2fc` lido ao contrário
    inverte a conclusão biológica.
    """
    return f"log2(fold change) — mediana de {grupo_a} ÷ mediana de {grupo_b}"


def nome_do_teste(parametrico: bool) -> str:
    """O nome do teste que correu, a partir de `parametrico`.

    Não é constante: quando Shapiro-Wilk rejeita a normalidade dos resíduos, a decisão
    automática vai para Mann-Whitney.
    """
    return WELCH if parametrico else MANN_WHITNEY


def prosa_da_normalidade(rota: str, shapiro_pvalue: float | None, n_residuos: int | None) -> str:
    """Como o teste foi escolhido. São as três rotas de `NormalityDecision`.

    A decisão é uma para a matriz inteira: Shapiro-Wilk corre uma vez sobre os resíduos de
    todas as features juntos. Uma decisão por feature seria outro método, com o q ajustando p
    de testes diferentes. Rota desconhecida devolve o rótulo cru.
    """
    if rota == "forced":
        return "FORÇADO pelo pedido; nenhum teste de normalidade correu"
    if rota == "shapiro":
        n = "n não declarado" if n_residuos is None else f"n = {milhar(n_residuos)}"
        p = "p não declarado" if shapiro_pvalue is None else f"p = {decimal(shapiro_pvalue, 3)}"
        return (
            f"escolhido por Shapiro-Wilk sobre os resíduos de TODAS as features juntos "
            f"({p}, {n}) — uma decisão para a matriz inteira, não por feature"
        )
    if rota == "shapiro_unavailable":
        return (
            "sem resíduos que chegassem para Shapiro-Wilk correr, caiu na opção conservadora — "
            "que é fato diferente de «medi e a normalidade foi rejeitada»"
        )
    return f"rota de normalidade não reconhecida ({rota!r})"


def rotulo_da_classe(classe: Classe, quantas: int, alfa: float) -> str:
    """A entrada de legenda de uma classe: o nome e a contagem.

    É a forma do MetaboAnalyst, `Sig.Down [1] | Sig.Up [2] | Unsig. [60]`.
    """
    if classe is Classe.NAO:
        return f"não significativa [{milhar(quantas)}]"
    return f"significativa {classe.value} (q < {decimal(alfa)}) [{milhar(quantas)}]"


def frase_da_populacao(pontos: PontosDoVolcano) -> str:
    """A segunda linha do título: quantos pontos, de quantas features, e o que ficou de fora.

    Os dois primeiros números levam o denominador, porque 'pontos' é sobre o desenho e
    'sobrevivem' é sobre a classificação. A cláusula dos sobreviventes sem ponto é o que
    reconcilia a soma das entradas da legenda com o número de sobreviventes.

    As duas primeiras cláusulas aparecem sempre, e as demais só quando não são zero.
    """
    c = pontos.contagens
    total = milhar(pontos.n_testadas)
    partes = [
        f"{milhar(c.desenhadas)} ponto(s) de {total} features testadas",
        f"{milhar(c.sobreviventes)} de {total} sobrevive(m) ao FDR",
    ]
    if c.sobreviventes_sem_ponto:
        partes.append(f"{milhar(c.sobreviventes_sem_ponto)} sobrevivente(s) sem ponto")
    if c.sem_fold_change:
        partes.append(f"{milhar(c.sem_fold_change)} sem fold change definido")
    if c.sem_pvalue_desenhavel:
        partes.append(f"{milhar(c.sem_pvalue_desenhavel)} sem p desenhável")
    if c.pvalue_zero:
        partes.append(f"{milhar(c.pvalue_zero)} com p exactamente zero")
    return "; ".join(partes)


def frase_do_vazio(pontos: PontosDoVolcano) -> str:
    """O texto escrito no lugar dos pontos quando nenhum é desenhável.

    Diz qual valor de eixo falta e traz a contagem de sobreviventes: sem ponto na tela, só
    ela separa 'nada significativo' de 'há, e não se consegue desenhar'.

    `pvalue_indefinido` e `pvalue_zero` são escritos como contagens sobre a matriz inteira, e
    não como decomposição das features sem p desenhável.
    """
    c = pontos.contagens
    total = milhar(pontos.n_testadas)
    return (
        f"nenhum dos {total} pontos é desenhável:\n"
        f"{milhar(c.sem_fold_change)} sem fold change definido e "
        f"{milhar(c.sem_pvalue_desenhavel)} com fold change e sem p desenhável.\n"
        f"Sobre as {total} testadas: {milhar(c.pvalue_indefinido)} sem p definido, "
        f"{milhar(c.pvalue_zero)} com p exactamente zero, e "
        f"{milhar(c.sobreviventes)} sobrevive(m) ao FDR.\n"
        "O teste correu — o que falta é o valor de eixo, e ele é contado e não descartado."
    )


def frase_das_barras(
    intervalos: IntervalosEscolhidos | IntervaloAusente,
    contraste: Contraste,
    *,
    sem_ponto: int = 0,
) -> str:
    """A frase das barras no rodapé: a regra, o nível, o ajuste e o que ficou sem barra.

    É a mesma nas duas figuras que desenham o `log2fc`. `sem_ponto` só existe no volcano, onde
    uma escolhida de p zero tem intervalo e não tem altura.

    A frase sempre avisa que o p é de outro teste. O intervalo é da razão de medianas, sobre a
    matriz do fold change, e o p vem da matriz que `matriz_do_pvalue` nomeia; a barra pode
    excluir o zero com o q acima do corte.

    É curta porque o rodapé do volcano é o mais longo do pacote e `recorte.aplica` recusa o
    que passa de `RESERVA_MAXIMA`. A fórmula do nível por extenso não cabe.
    """
    if isinstance(intervalos, IntervaloAusente):
        return f"Barras do intervalo do fold change: nenhuma — {intervalos.motivo}."
    i = intervalos
    if not i.escolhidas:
        return (
            "Barras do intervalo do fold change: nenhuma — nenhuma feature tem p definido, e a "
            "regra escolhe as de menor p."
        )
    # Nenhum número desta frase é seguido de vírgula ou ponto: com a vírgula decimal, '6, a'
    # seria lido como o número `6,`.
    if i.com_intervalo:
        partes = [
            f"Barras: intervalo de Bonett-Price da razão de medianas nas "
            f"{milhar(i.com_intervalo)} das {milhar(i.escolhidas)} de menor p, a "
            f"{decimal(i.nivel * 100.0, 5)}% (FCR de Benjamini-Yekutieli a {decimal(i.fcr)})"
        ]
    else:
        partes = [f"Barras: nenhuma das {milhar(i.escolhidas)} de menor p tem intervalo"]
    sem_intervalo = sum(n for _, n in i.sem_intervalo)
    if sem_intervalo:
        motivos = ", ".join(f"{milhar(n)} {prosa}" for prosa, n in i.sem_intervalo)
        partes.append(f"{milhar(sem_intervalo)} sem intervalo ({motivos})")
    if sem_ponto:
        partes.append(f"{milhar(sem_ponto)} sem altura (p zero)")
    if i.empatadas_fora:
        partes.append(
            f"{milhar(i.empatadas_fora)} de fora empatam no último p (desempate: ordem da matriz)"
        )
    outra = ", outra matriz" if contraste.matrizes_diferem else ""
    return (
        "; ".join(partes) + f". O p é de outro teste, sobre a {contraste.matriz_do_pvalue}{outra}."
    )


def rotulo_das_barras(escolhidas: int, desenhadas: int) -> str:
    """A entrada de legenda das barras: a regra e quantas barras estão na imagem.

    O nível fica no rodapé, junto da regra que o explica.
    """
    return f"intervalo das {milhar(escolhidas)} de menor p [{milhar(desenhadas)}]"


def recorte_do_volcano(
    pontos: PontosDoVolcano,
    contraste: Contraste,
    intervalos: IntervalosEscolhidos | IntervaloAusente | None = None,
) -> Recorte:
    """O rodapé: população, as duas procedências, método, correção e o `n` de cada grupo.

    O `n` de cada grupo e a correção ficam no rodapé, e não na legenda, porque o rodapé é o
    texto da figura que quebra em linhas.

    `menor_q` é declarado sobre as features testadas, e não sobre os pontos.

    A cláusula dos sobreviventes sem ponto só aparece quando há algum, porque este rodapé é o
    mais longo do pacote e `recorte.aplica` recusa o que passa de `RESERVA_MAXIMA` da altura.
    Ela decompõe os sobreviventes sem ponto por causa, com `sobreviventes_sem_fold_change`.

    Sem par de grupos declarado levanta `MatrizVaziaError`, que o painel mostra como texto.
    """
    if not contraste.grupo_a or not contraste.grupo_b:
        raise MatrizVaziaError(
            "o volcano não declara o par de grupos: a execução correu sem `group_a`/`group_b`, "
            "então nenhum teste univariado foi calculado. Isso é «não calculado», e não «calculado "
            "e vazio» — figura em branco seria indistinguível de figura quebrada"
        )
    c = pontos.contagens
    total = milhar(pontos.n_testadas)
    menor = "nenhum q é definido" if pontos.menor_q is None else decimal(pontos.menor_q)
    normalidade = prosa_da_normalidade(
        contraste.rota_da_normalidade, contraste.shapiro_pvalue, contraste.n_residuos
    )
    sem_ponto = (
        f" Destas, {milhar(c.sobreviventes_sem_ponto)} não tem ponto na nuvem: "
        f"{milhar(c.sobreviventes_sem_fold_change)} sem fold change definido e "
        f"{milhar(c.sobreviventes_sem_pvalue_desenhavel)} com p não desenhável — é ausência de "
        "valor de EIXO, e não de significância."
        if c.sobreviventes_sem_ponto
        else ""
    )
    return Recorte(
        desenhadas=c.desenhadas,
        disponiveis=pontos.n_testadas,
        amostragem=(
            f"nenhuma — as {total} testadas entram, e as "
            f"{milhar(c.nao_desenhadas)} sem valor de eixo são contadas, não descartadas. "
            f"Contraste: {contraste.grupo_a} (n = {milhar(contraste.n_a)}) contra "
            f"{contraste.grupo_b} (n = {milhar(contraste.n_b)})"
        ),
        matriz=_procedencia(contraste),
        metodo=(
            f"{nome_do_teste(contraste.parametrico)} por feature, {normalidade}. "
            f"{contraste.correcao} a q < {decimal(contraste.alfa)}: "
            f"{milhar(c.sobreviventes)} de {total} sobrevive(m), e o menor q "
            f"sobre as {total} testadas é {menor} — a COR classifica por q, e "
            f"o eixo vertical é o p BRUTO, então o corte não é uma altura desta figura."
            f"{sem_ponto} {DIVERGENCIA}"
            f"{_frase_das_barras_do_volcano(intervalos, contraste)}"
        ),
    )


def _frase_das_barras_do_volcano(
    intervalos: IntervalosEscolhidos | IntervaloAusente | None, contraste: Contraste
) -> str:
    """A frase das barras com a contagem de escolhidas sem altura desta figura.

    `None` devolve `""`: a chamada não pediu barras. `IntervaloAusente` é outro caso, o de
    barras pedidas sem os dados para desenhá-las, e esse é declarado.
    """
    if intervalos is None:
        return ""
    sem_ponto = _barras_do_volcano(intervalos)[2]
    return f". {frase_das_barras(intervalos, contraste, sem_ponto=sem_ponto)}"


def _barras_do_volcano(
    intervalos: IntervalosEscolhidos | IntervaloAusente,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Os segmentos horizontais das barras no volcano, e quantas escolhidas ficaram sem altura.

    Cada barra vai de um limite ao outro na altura do ponto, `-log10(p)`. Uma escolhida de p
    zero tem intervalo e não tem altura, então fica sem barra e é contada.

    As barras são uma linha só, com `nan` entre elas, para a identificação do painel e o arnês
    de teste as lerem como uma série.
    """
    if isinstance(intervalos, IntervaloAusente):
        return np.empty(0), np.empty(0), 0
    with np.errstate(divide="ignore"):
        altura = -np.log10(intervalos.pvalue)
    com_altura = np.isfinite(altura)
    n = int(com_altura.sum())
    x = np.column_stack(
        [intervalos.inferior[com_altura], intervalos.superior[com_altura], np.full(n, np.nan)]
    ).ravel()
    y = np.column_stack([altura[com_altura], altura[com_altura], np.full(n, np.nan)]).ravel()
    return x, y, int((~com_altura).sum())


def _procedencia(contraste: Contraste) -> str:
    """A frase da matriz de cada eixo, com a ressalva de quando as duas são a mesma.

    Com `transformation` e `scaling` em `none`, a matriz anterior à transformação e a
    processada são a mesma, e declarar duas procedências afirmaria uma distinção que a
    execução não tem.
    """
    corpo = (
        f"eixo x (fold change) da {contraste.matriz_do_fold_change}, "
        f"eixo y (p) da {contraste.matriz_do_pvalue}"
    )
    if contraste.matrizes_diferem:
        return corpo
    return f"{corpo} — e nesta execução as DUAS são a mesma matriz: a transformação não a moveu"


# ---------------------------------------------------------------------------
# A cor, do ciclo dos `rcParams`
# ---------------------------------------------------------------------------


def cores_das_classes() -> dict[Classe, Any]:
    """Uma cor por classe, tirada do ciclo dos `rcParams`.

    A leitura do ciclo é de `base.cores_do_ciclo`; aqui se decide que classe fica com que
    posição. O valor é `Any` porque o ciclo pode trazer uma cadeia hexadecimal ou uma tupla
    RGB.
    """
    return dict(zip(ORDEM_DAS_CLASSES, cores_do_ciclo(len(ORDEM_DAS_CLASSES)), strict=True))


def cor_das_barras() -> Any:
    """A cor das barras do intervalo: a posição do ciclo logo depois das quatro classes.

    É a mesma no volcano e no fold change por feature, porque as duas figuras desenham o mesmo
    intervalo da mesma feature.
    """
    return cores_do_ciclo(len(ORDEM_DAS_CLASSES) + 1)[-1]


# ---------------------------------------------------------------------------
# O construtor
# ---------------------------------------------------------------------------


def figura_do_volcano(
    *,
    pontos: PontosDoVolcano,
    contraste: Contraste,
    recorte: Recorte,
    intervalos: IntervalosEscolhidos | IntervaloAusente | None = None,
) -> Figure:
    """O volcano: a dispersão, as contagens por classe na legenda e o recorte no rodapé.

    Desenha todas as features desenháveis, sem amostragem, numa única dispersão. Ver o
    cabeçalho do módulo.

    Com `intervalos`, as barras são um segundo artista, sem nomes. Elas entram no eixo antes
    da dispersão: o clique dispara um evento por artista na ordem de inserção e a
    identificação fica com o último, que tem de ser a nuvem. O `zorder` as desenha por cima.
    `recorte` tem de ter sido montado com os mesmos `intervalos`, porque é o rodapé que diz a
    regra.

    As entradas de legenda são artistas que não entram no eixo.
    """
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    paleta = cores_das_classes()
    barras = 0

    if pontos.nomes:
        if intervalos is not None:
            x, y, _sem_ponto = _barras_do_volcano(intervalos)
            barras = desenha_as_barras(eixo, x, y)
        # `color=` e não `c=`: com `c=`, uma sequência do comprimento de `x` é lida como
        # valores a mapear por colormap, e a tupla RGB de uma classe só pode cair nesse caso.
        nuvem = eixo.scatter(
            pontos.x, pontos.y, color=_cor_dos_pontos(pontos, paleta), label=ROTULO_DA_SERIE
        )
        # A marca diz ao painel que é este o artista indexado pelo vetor de nomes.
        nuvem.set_gid(GID_DAS_FEATURES)
    else:
        # Sem pontos, a figura diz por quê em texto: gráfico vazio se lê como gráfico quebrado.
        eixo.text(
            0.5,
            0.5,
            frase_do_vazio(pontos),
            transform=eixo.transAxes,
            ha="center",
            va="center",
            fontsize=TAMANHO_DA_FONTE,
        )
        # E as marcas saem: o `matplotlib` desenharia os limites padrão de 0 a 1, e a figura
        # afirmaria que o fold change vai de zero a um.
        eixo.set_xticks([])
        eixo.set_yticks([])

    eixo.set_xlabel(rotulo_de_x(contraste.grupo_a, contraste.grupo_b))
    eixo.set_ylabel(ROTULO_DE_Y)
    eixo.set_title(
        f"{TITULO} — {contraste.grupo_a} contra {contraste.grupo_b}\n{frase_da_populacao(pontos)}",
        fontsize=TAMANHO_DA_FONTE,
    )
    provas = _provas_da_legenda(pontos, contraste, paleta)
    if barras and isinstance(intervalos, IntervalosEscolhidos):
        provas.append(prova_das_barras(intervalos, barras))
    eixo.legend(handles=provas, fontsize=TAMANHO_DA_FONTE)
    aplica(figura, recorte)
    return figura


def desenha_as_barras(eixo: Axes, x: np.ndarray, y: np.ndarray) -> int:
    """Põe as barras no eixo, como uma linha só com `nan` entre elas, e devolve quantas desenhou.

    Serve às duas figuras: a geometria muda (horizontal no volcano, vertical no fold change)
    e o artista não. Tem de ser chamada antes da dispersão, para a nuvem ser o último artista
    inserido e responder ao clique; o `zorder` acima do da dispersão desenha a barra por cima.

    Sem barra nenhuma não cria artista, para não haver uma série vazia na legenda.
    """
    if not x.size:
        return 0
    eixo.plot(
        x,
        y,
        color=cor_das_barras(),
        linewidth=1.2,
        solid_capstyle="butt",
        zorder=2.5,
        label=ROTULO_DAS_BARRAS,
    )
    return int(np.isfinite(x).sum() // 2)


def prova_das_barras(intervalos: IntervalosEscolhidos, desenhadas: int) -> Line2D:
    """A entrada de legenda das barras, com a contagem das que estão na imagem."""
    return Line2D(
        [],
        [],
        color=cor_das_barras(),
        linewidth=1.2,
        label=rotulo_das_barras(intervalos.escolhidas, desenhadas),
    )


def _cor_dos_pontos(pontos: PontosDoVolcano, paleta: dict[Classe, Any]) -> Any:
    """A cor da dispersão: uma só quando há uma classe só, e o vetor quando há mais.

    A imagem é a mesma, e o `matplotlib` deixa de resolver N cores a cada redesenho. Com
    milhares de pontos isso pesa: deslocar e aproximar é manipulação direta, com limite de
    cerca de 100 ms (Nielsen 1993), e o canvas repinta a figura inteira a cada deslocamento.
    Uma classe só é o caso comum, quando nenhuma feature sobrevive ao FDR.
    """
    distintas = set(pontos.classes)
    if len(distintas) == 1:
        return paleta[next(iter(distintas))]
    return [paleta[classe] for classe in pontos.classes]


def _provas_da_legenda(
    pontos: PontosDoVolcano, contraste: Contraste, paleta: dict[Classe, Any]
) -> list[Line2D]:
    """As entradas de legenda, uma por classe.

    `ABAIXO`, `ACIMA` e `NAO` aparecem sempre, inclusive com `[0]`, porque o zero responde
    'quantas para cada lado?'. `SEM_DIRECAO` só aparece quando existe.

    As contagens são as do desenho, para o número ser contável na imagem. O total de
    sobreviventes fica no título e no rodapé. Os sobreviventes sem ponto não entram em nenhuma
    entrada, ao contrário da ANOVA, porque não têm direção.
    """
    quantas = {
        Classe.ABAIXO: pontos.contagens.abaixo,
        Classe.SEM_DIRECAO: pontos.contagens.sem_direcao,
        Classe.ACIMA: pontos.contagens.acima,
        Classe.NAO: pontos.contagens.nao_significativas_desenhadas,
    }
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=paleta[classe],
            label=rotulo_da_classe(classe, quantas[classe], contraste.alfa),
        )
        for classe in ORDEM_DAS_CLASSES
        if classe is not Classe.SEM_DIRECAO or quantas[classe]
    ]

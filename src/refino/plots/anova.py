"""A figura da ANOVA de N grupos: a entrada `One-way ANOVA` do MetaboAnalyst.

Responde se alguma feature separa os grupos clínicos. O eixo x é a feature, na ordem da
matriz, porque com mais de dois grupos não há razão entre grupos para pôr ali. O eixo y é
`-log10(q)`, com o corte do FDR desenhado como uma altura: `q < alfa` é estar acima da
linha. No volcano não há essa linha porque lá o eixo vertical é o p bruto. Num estudo
pequeno, depois da correção de múltiplos testes, é comum nenhuma feature sobreviver; a
figura mostra a nuvem inteira abaixo da linha, e o rodapé traz o `n` de cada grupo e a
correção aplicada, que é o que separa 'não há efeito' de 'não há poder'.

Sem feature nenhuma a figura é recusada com `MatrizVaziaError`; com features e nenhum q
desenhável ela sai com texto no lugar dos pontos. Classificar e desenhar são separados: um
q igual a zero vira `+inf` e fica sem ponto, mas continua contado como sobrevivente, e
`ContagensDaAnova` traz as duas partições. Vetores por feature de comprimentos diferentes
são recusados, em vez de completados com `nan` como no volcano: aqui o x é o índice, e
completar desenharia pontos em posições que nenhuma feature ocupa.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from matplotlib.lines import Line2D

from refino.plots.base import (
    GID_DAS_FEATURES,
    NAO_DECLARADA,
    MatrizVaziaError,
    RecusaDoConstrutorError,
    cores_do_ciclo,
    inteiro_do_fio,
    nova_figura,
    numero_do_fio,
    texto_do_fio,
    vetor_do_fio_ou_nucleo,
)
from refino.plots.recorte import TAMANHO_DA_FONTE, Recorte, aplica, decimal, milhar

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.figure import Figure

__all__ = [
    "ANOVA",
    "KRUSKAL_WALLIS",
    "ROTULO_DA_SERIE",
    "ROTULO_DE_X",
    "ROTULO_DE_Y",
    "ROTULO_DO_CORTE",
    "TITULO",
    "AnovaDesalinhadaError",
    "Classe",
    "ContagensDaAnova",
    "ContextoDaAnova",
    "GrupoDaAnova",
    "PontosDaAnova",
    "contexto_do_fio",
    "cores_das_classes",
    "figura_da_anova",
    "frase_da_populacao",
    "frase_do_post_hoc",
    "frase_do_vazio",
    "frase_dos_grupos",
    "grupos_do_fio",
    "nome_do_teste",
    "nomes_do_fio",
    "pontos_da_anova",
    "recorte_da_anova",
    "rotulo_da_classe",
]

# 'todos os grupos' distingue este título do da figura dos testes t, que fica no mesmo seletor
# e também fala de significância por feature. O que separa as duas é a população: todos os
# grupos da matriz aqui, o par declarado lá.
TITULO: Final = "Significância por feature — todos os grupos"

ROTULO_DE_X: Final = "feature, na ordem da matriz"

# O eixo é o q, e não o p. É por isso que o corte pode ser uma linha.
ROTULO_DE_Y: Final = "-log10(q de Benjamini-Hochberg)"

# O rótulo do único artista de dado. Não vai à legenda, que mostra as classes; é o texto que
# a identificação por clique mostra quando o vetor de nomes não cobre a posição.
ROTULO_DA_SERIE: Final = "features desenhadas"

ROTULO_DO_CORTE: Final = "corte do FDR"

ANOVA: Final = "ANOVA de uma via (F), sobre os valores"

KRUSKAL_WALLIS: Final = "Kruskal-Wallis (H), sobre os postos"


class AnovaDesalinhadaError(RecusaDoConstrutorError):
    """Vetores que descrevem a mesma população têm comprimentos diferentes.

    Vale para dois casos. Os nomes de grupo e as contagens de grupo formam pares, e emparelhar
    o que houver poria o `n` de um grupo ao lado do nome de outro. Os vetores por feature são
    indexados pelo índice, e completar com `nan` desenharia posições que nenhuma feature ocupa.
    """


class Classe(StrEnum):
    """A classe de um ponto desenhado. São duas.

    O volcano tem quatro porque o `log2fc` dá direção ao que é significativo. A ANOVA só diz
    que algum grupo difere de algum outro; quais diferem é assunto do post-hoc, que não está
    nesta figura.
    """

    SOBREVIVE = "sobrevive ao FDR"
    NAO = "não sobrevive"


# A ordem é a da legenda: primeiro o que se procura, depois o resto.
ORDEM_DAS_CLASSES: Final = (Classe.SOBREVIVE, Classe.NAO)


@dataclass(frozen=True)
class GrupoDaAnova:
    """Um grupo da ANOVA e o `n` dele.

    O par é um tipo para que nome e contagem não se desalinhem em silêncio, como aconteceria
    com duas listas paralelas. `grupos_do_fio` recusa as listas que não emparelham.
    """

    nome: str
    n: int


@dataclass(frozen=True)
class ContagensDaAnova:
    """Quantas features caíram em cada desfecho, em duas partições de `n_da_matriz`.

    - classificação: `sobreviventes + nao_significativas + sem_qvalue`, decidida por q finito
      e `q < alfa`;
    - desenho: `desenhadas + nao_desenhadas`, decidida por `-log10(q)` ser finito.

    `sobreviventes` conta sobre todas as features, com o mesmo predicado estrito que
    `statistics.anova` usa para decidir quem vai ao post-hoc, e por isso é igual a
    `AnovaCounts.survivors`. Contado só sobre as desenháveis, ele perderia a feature de q
    igual a zero, que vai a `+inf` e sai do desenho.

    `sobreviventes_desenhados` e `nao_significativas_desenhadas` somam `desenhadas` e são as
    que vão à legenda, porque a legenda descreve a nuvem.

    `sem_pvalue`, `pvalue_zero` e `qvalue_zero` são contados à parte e não entram em partição
    nenhuma. O p não está em eixo desta figura, então a ausência dele não tira a feature da
    nuvem. Um p ou um q exatamente zero é medida, e não ausência; `qvalue_zero` é a causa
    usual de um sobrevivente sem ponto.
    """

    # A partição da classificação.
    sobreviventes: int
    nao_significativas: int
    sem_qvalue: int
    # A partição do desenho.
    desenhadas: int
    nao_desenhadas: int
    # As duas classes dentro do desenho, que a legenda conta.
    sobreviventes_desenhados: int
    nao_significativas_desenhadas: int
    # Contadas à parte; nenhuma é classe.
    sem_pvalue: int
    pvalue_zero: int
    qvalue_zero: int

    @property
    def sobreviventes_sem_ponto(self) -> int:
        """Quantas sobreviveram ao FDR e não têm ponto na nuvem. A causa usual é q igual a zero."""
        return self.sobreviventes - self.sobreviventes_desenhados

    @property
    def nao_significativas_sem_ponto(self) -> int:
        """Quantas não significativas não têm ponto na nuvem.

        Com `alfa` válido é zero, porque `-log10` de um q positivo e finito é finito. Com
        `alfa` ilegível ele chega como `nan`, `q < nan` é falso, e um q igual a zero cai aqui
        em vez de em `sobreviventes_sem_ponto`.
        """
        return self.nao_significativas - self.nao_significativas_desenhadas


@dataclass(frozen=True)
class ContextoDaAnova:
    """Como a ANOVA foi medida: o que a figura declara e não consegue derivar dos vetores.

    `metodo` é o rótulo de `AnovaMethod`. O mesmo campo de estatística carrega o F da ANOVA de
    uma via ou o H de Kruskal-Wallis, que têm distribuições nulas diferentes.

    `post_hoc_rota` tem três valores, os de `PostHocRoute`: o post-hoc correu, nenhuma feature
    sobreviveu, ou nenhuma era testável. Os dois últimos dão a mesma nuvem e são fatos opostos.

    `matriz` é uma só, ao contrário do volcano: sem razão entre grupos não há a segunda matriz
    de onde o fold change sairia.
    """

    grupos: tuple[GrupoDaAnova, ...]
    metodo: str
    post_hoc_metodo: str
    post_hoc_rota: str
    alfa: float
    correcao: str
    matriz: str
    min_por_grupo: int


@dataclass(frozen=True)
class PontosDaAnova:
    """Os pontos desenháveis, na ordem em que as features chegam, com as contagens da população.

    `nomes`, `x` e `y` têm o mesmo comprimento e descrevem as features de q finito. `nomes` é o
    vetor que o painel recebe para a identificação por clique.

    `x` é o índice original da feature na matriz, e não a posição dentro do desenho. Uma
    feature sem q deixa um vão, e a abscissa continua comparável com a tabela.

    `menor_q` e `menor_p` são o mínimo sobre todas as features da matriz, e não sobre as
    desenhadas: a feature de menor q pode estar fora do desenho. São `None` quando nenhum valor
    é finito.

    `n_da_matriz` é o número de features da matriz processada, e não o de testadas. A ANOVA
    não testa feature com grupo abaixo do mínimo nem feature constante, e as duas continuam na
    tabela com estatística, p e q em `nan`. O parâmetro do corpo se chama `n_features_testadas`,
    mas o que a API envia é o comprimento da tabela.
    """

    nomes: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    classes: tuple[Classe, ...]
    n_da_matriz: int
    contagens: ContagensDaAnova
    menor_q: float | None
    menor_p: float | None


# ---------------------------------------------------------------------------
# A fronteira: o que chega do corpo JSON não passou pelo mypy
# ---------------------------------------------------------------------------


def nomes_do_fio(valores: Any) -> tuple[str, ...]:
    """A lista de nomes de feature do corpo, com o que não é texto convertido em `""`.

    Não descarta: um nome ilegível tem de continuar na posição dele, senão o vetor se
    desalinha dos numéricos e a identificação devolve o nome do vizinho. É igual a
    `volcano.nomes_do_fio`.
    """
    if not isinstance(valores, list):
        return ()
    return tuple("" if isinstance(v, bool) or not isinstance(v, str) else v for v in valores)


def grupos_do_fio(nomes: Any, contagens: Any) -> tuple[GrupoDaAnova, ...]:
    """Os pares de nome e `n` dos grupos, ou a recusa quando as duas listas não emparelham.

    Emparelhar o que houver, com `zip` sem `strict`, poria o `n` de um grupo ao lado do nome de
    outro. Nome ilegível vira `""` em vez de ser descartado, para o comprimento não mudar e a
    recusa disparar.
    """
    lidos = nomes if isinstance(nomes, list) else []
    contados = contagens if isinstance(contagens, list) else []
    if len(lidos) != len(contados):
        raise AnovaDesalinhadaError(
            f"a ANOVA declara {len(lidos)} nome(s) de grupo e {len(contados)} contagem(ns): as "
            "duas "
            "listas são PARES e não coleções independentes. Emparelhar pelo que houver poria o `n` "
            "de um grupo ao lado do nome de outro, e a figura sairia bem formada"
        )
    return tuple(
        GrupoDaAnova(nome=texto_do_fio(nome, ""), n=inteiro_do_fio(n))
        for nome, n in zip(lidos, contados, strict=True)
    )


def contexto_do_fio(
    *,
    grupos: Any,
    n_por_grupo: Any,
    metodo: Any,
    post_hoc_metodo: Any,
    post_hoc_rota: Any,
    alfa_fdr: Any,
    correcao: Any,
    matriz: Any,
    min_por_grupo: Any,
) -> ContextoDaAnova:
    """O `ContextoDaAnova` a partir dos campos da raiz `anova`, com guarda em cada um.

    Recebe os campos um a um, e não o dicionário, para que a leitura de cada chave do corpo
    fique na janela, onde o teste de contrato com a API confere as chaves contra o esquema.
    """
    return ContextoDaAnova(
        grupos=grupos_do_fio(grupos, n_por_grupo),
        metodo=texto_do_fio(metodo, ""),
        post_hoc_metodo=texto_do_fio(post_hoc_metodo, ""),
        post_hoc_rota=texto_do_fio(post_hoc_rota, ""),
        alfa=numero_do_fio(alfa_fdr),
        correcao=texto_do_fio(correcao, NAO_DECLARADA),
        matriz=texto_do_fio(matriz, NAO_DECLARADA),
        min_por_grupo=inteiro_do_fio(min_por_grupo),
    )


# ---------------------------------------------------------------------------
# A partição da população, usada pelo construtor e pelo painel
# ---------------------------------------------------------------------------


def pontos_da_anova(
    *,
    features: Any,
    pvalue: Any,
    qvalue_fdr: Any,
    n_features_testadas: Any,
    alfa: float,
) -> PontosDaAnova:
    """Transforma os vetores do corpo nos pontos desenháveis e nas contagens.

    O painel usa o mesmo resultado para obter os nomes desenhados, e assim a seleção dos
    pontos é feita num lugar só.

    Levanta `MatrizVaziaError` quando não veio feature nenhuma. Nenhuma feature desenhável é
    outro caso: é fato do dado, e a figura o declara. Levanta `AnovaDesalinhadaError` quando os
    vetores não têm o mesmo comprimento.

    `n_features_testadas` é conferido contra o comprimento dos vetores, em vez de derivado
    dele: é o denominador que a figura publica, e um vetor truncado publicaria uma população
    menor. Quando não vem, vale o comprimento. Ele alimenta `n_da_matriz`.

    A significância é calculada sobre o vetor inteiro, e o que é desenhável sobre `-log10(q)`.
    Ver `ContagensDaAnova`.
    """
    nomes = nomes_do_fio(features)
    p = vetor_do_fio_ou_nucleo(pvalue)
    q = vetor_do_fio_ou_nucleo(qvalue_fdr)

    comprimentos = {"features": len(nomes), "pvalue": int(p.size), "qvalue_fdr": int(q.size)}
    if len(set(comprimentos.values())) > 1:
        raise AnovaDesalinhadaError(
            f"os vetores por feature da ANOVA têm comprimentos diferentes: {comprimentos}. Eles "
            "indexam as MESMAS features, e o eixo x desta figura é o índice — completar o que "
            "falta desenharia pontos em posições que nenhuma feature ocupa, e a contagem "
            "chamar-lhes-ia «sem q»"
        )

    n_testadas = len(nomes)
    declarado = inteiro_do_fio(n_features_testadas)
    if declarado and declarado != n_testadas:
        raise AnovaDesalinhadaError(
            f"a ANOVA declara {declarado} features e trouxe {n_testadas} valores. O "
            "denominador é o que a figura publica; com um vetor truncado ela publicaria uma "
            "população que não é a que está desenhada"
        )
    if n_testadas == 0:
        raise MatrizVaziaError(
            "a ANOVA não recebeu feature nenhuma: não há população para desenhar nem para "
            "declarar. Sem a família pedida a execução não corre o teste, e é isso que é verdade — "
            "figura em branco seria indistinguível de figura quebrada"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        y = -np.log10(q)

    indices = np.arange(n_testadas, dtype=float)
    # O que é desenhável: `-log10(q)` é infinito em q igual a zero e indefinido em q não finito.
    desenhavel = np.isfinite(y)
    # A classificação corre sobre o vetor inteiro, com o mesmo predicado estrito de
    # `statistics.anova`.
    significativa = np.isfinite(q) & (q < alfa)
    com_qvalue = np.isfinite(q)

    classes = tuple(
        Classe.SOBREVIVE if bool(significativa[i]) else Classe.NAO
        for i in range(n_testadas)
        if bool(desenhavel[i])
    )
    contagens = ContagensDaAnova(
        sobreviventes=int(significativa.sum()),
        nao_significativas=int((com_qvalue & ~significativa).sum()),
        sem_qvalue=int((~com_qvalue).sum()),
        desenhadas=int(desenhavel.sum()),
        nao_desenhadas=int((~desenhavel).sum()),
        sobreviventes_desenhados=sum(1 for c in classes if c is Classe.SOBREVIVE),
        nao_significativas_desenhadas=sum(1 for c in classes if c is Classe.NAO),
        sem_pvalue=int((~np.isfinite(p)).sum()),
        pvalue_zero=int((p == 0.0).sum()),
        # Zero exato, e não abaixo de um limiar: só o zero exato vai a `+inf` na `-log10`.
        qvalue_zero=int((q == 0.0).sum()),
    )
    return PontosDaAnova(
        nomes=tuple(nomes[i] for i in range(n_testadas) if bool(desenhavel[i])),
        x=indices[desenhavel],
        y=y[desenhavel],
        classes=classes,
        n_da_matriz=n_testadas,
        contagens=contagens,
        menor_q=_minimo_finito(q),
        menor_p=_minimo_finito(p),
    )


def _minimo_finito(valores: np.ndarray) -> float | None:
    """O menor valor finito, ou `None` quando não há nenhum.

    `None`, e não `nan`, porque o rodapé escreve uma frase própria para 'nenhum valor é
    definido'.
    """
    finitos = valores[np.isfinite(valores)]
    return float(finitos.min()) if finitos.size else None


# ---------------------------------------------------------------------------
# O que a figura diz sobre si mesma
# ---------------------------------------------------------------------------


def nome_do_teste(metodo: str) -> str:
    """O nome do teste que correu, a partir do rótulo de `AnovaMethod`.

    Não é uma constante porque o mesmo campo carrega F ou H, com distribuições nulas
    diferentes: `F(k-1, N-k)` e `χ²(k-1)`. Rótulo desconhecido devolve o texto cru.
    """
    if metodo == "anova":
        return ANOVA
    if metodo == "kruskal_wallis":
        return KRUSKAL_WALLIS
    return f"teste de N grupos não reconhecido ({metodo!r})"


def frase_do_post_hoc(rota: str, metodo: str, sobreviventes: int) -> str:
    """Se o post-hoc correu e, quando não correu, por quê. São as três rotas de `PostHocRoute`.

    'Nenhuma feature sobreviveu ao FDR' e 'nenhuma feature era testável' dão a mesma nuvem
    baixa. A primeira é resultado; a segunda indica matriz degenerada.

    O método aparece mesmo quando o post-hoc não correu, porque faz parte da configuração que
    produziu a figura.
    """
    nome = metodo or "não declarado"
    if rota == "ran":
        return f"post-hoc de {nome} nas {milhar(sobreviventes)} sobrevivente(s)"
    if rota == "no_survivors":
        return f"post-hoc de {nome} pedido e NÃO corrido: nenhuma feature sobreviveu ao FDR"
    if rota == "nothing_testable":
        return (
            f"post-hoc de {nome} pedido e NÃO corrido: nenhuma feature era testável, e o "
            "denominador da correção era zero — que é fato diferente de «nenhuma sobreviveu»"
        )
    return f"rota de post-hoc não reconhecida ({rota!r})"


def frase_dos_grupos(grupos: tuple[GrupoDaAnova, ...]) -> str:
    """Os grupos com o `n` de cada um.

    Com vários grupos é o menor deles que decide o poder do teste, e o total esconderia isso.
    """
    if not grupos:
        return "nenhum grupo declarado"
    return ", ".join(f"{g.nome or '(sem nome)'} (n = {milhar(g.n)})" for g in grupos)


def rotulo_da_classe(classe: Classe, quantas: int, alfa: float, *, sem_ponto: int = 0) -> str:
    """A entrada de legenda de uma classe: o nome, o corte e a contagem da nuvem.

    A forma é a do MetaboAnalyst, `Sig. [1] | Unsig. [60]`. `quantas` conta os pontos
    desenhados, para o número ser contável na imagem. Quando `sem_ponto` não é zero, a entrada
    diz também quantas features da classe ficaram sem ponto.
    """
    nome = (
        f"sobrevive ao FDR (q < {decimal(alfa)})" if classe is Classe.SOBREVIVE else "não sobrevive"
    )
    if sem_ponto:
        return f"{nome} [{milhar(quantas)} desenhada(s), mais {milhar(sem_ponto)} sem ponto]"
    return f"{nome} [{milhar(quantas)}]"


def frase_da_populacao(pontos: PontosDaAnova) -> str:
    """A segunda linha do título: quantos pontos, quantos sobrevivem e sobre que população.

    Os dois primeiros números levam o denominador, porque 'pontos' é sobre o desenho e
    'sobrevivem' é sobre a classificação, e as duas populações divergem quando há sobrevivente
    sem ponto. O denominador é 'features da matriz', e não 'testadas'; ver `PontosDaAnova`.

    As duas primeiras cláusulas aparecem sempre, e as demais só quando não são zero.
    """
    c = pontos.contagens
    total = milhar(pontos.n_da_matriz)
    partes = [
        f"{milhar(c.desenhadas)} ponto(s) de {total} features da matriz",
        f"{milhar(c.sobreviventes)} de {total} sobrevive(m) ao FDR",
    ]
    if c.sobreviventes_sem_ponto:
        partes.append(f"{milhar(c.sobreviventes_sem_ponto)} sobrevivente(s) sem ponto")
    if c.qvalue_zero:
        partes.append(f"{milhar(c.qvalue_zero)} com q exactamente zero")
    if c.sem_qvalue:
        partes.append(f"{milhar(c.sem_qvalue)} sem q definido")
    if c.sem_pvalue:
        partes.append(f"{milhar(c.sem_pvalue)} sem p definido")
    if c.pvalue_zero:
        partes.append(f"{milhar(c.pvalue_zero)} com p exactamente zero")
    return "; ".join(partes)


def frase_do_vazio(pontos: PontosDaAnova) -> str:
    """O texto escrito no lugar dos pontos quando nenhum é desenhável.

    Separa 'sem q' de 'q exatamente zero': o primeiro é ausência de medida, e o segundo é a
    feature que mais separa os grupos.
    """
    c = pontos.contagens
    return (
        f"nenhuma das {milhar(pontos.n_da_matriz)} features tem q desenhável:\n"
        f"{milhar(c.sem_qvalue)} sem q, {milhar(c.qvalue_zero)} com q exactamente zero, "
        f"{milhar(c.sem_pvalue)} sem p.\n"
        f"Sobrevivem ao FDR: {milhar(c.sobreviventes)}.\n"
        "O teste foi pedido — o que falta é o valor de eixo, e ele é contado e não descartado."
    )


def recorte_da_anova(pontos: PontosDaAnova, contexto: ContextoDaAnova) -> Recorte:
    """O rodapé: população, procedência da matriz, método, correção e o `n` de cada grupo.

    Os `n` de grupo e a correção ficam no rodapé, e não na legenda, porque o rodapé é o texto
    da figura que quebra em linhas. Este é o rodapé mais cheio do pacote: com nomes de grupo
    longos ele pode passar de `RESERVA_MAXIMA` da altura, e então `recorte.aplica` levanta
    `RecorteInvalidoError`, que o painel apanha. Por isso a cláusula do sobrevivente sem ponto
    só aparece quando há algum.

    Os dois mínimos são declarados sobre as features da matriz, e não sobre os pontos: um q
    igual a zero é o menor possível e não tem ponto.

    Sem grupo declarado levanta `MatrizVaziaError`: a ANOVA não correu.
    """
    if not contexto.grupos:
        raise MatrizVaziaError(
            "a ANOVA não declara grupo nenhum: a execução correu sem a família pedida, ou a matriz "
            "não tem grupos contrastáveis. Isso é «não calculado», e não «calculado e vazio» — "
            "figura em branco seria indistinguível de figura quebrada"
        )
    c = pontos.contagens
    sobrevivem = c.sobreviventes
    total = milhar(pontos.n_da_matriz)
    menor_q = "nenhum q é definido" if pontos.menor_q is None else decimal(pontos.menor_q)
    menor_p = "nenhum p é definido" if pontos.menor_p is None else decimal(pontos.menor_p)
    sem_ponto = (
        f" Destas, {milhar(c.sobreviventes_sem_ponto)} não tem ponto na nuvem — "
        f"{milhar(c.qvalue_zero)} com q exactamente zero, que é separação exacta e não ausência "
        "de medida."
        if c.sobreviventes_sem_ponto
        else ""
    )
    return Recorte(
        desenhadas=c.desenhadas,
        disponiveis=pontos.n_da_matriz,
        amostragem=(
            f"nenhuma — as {total} features da matriz entram, e as que ficam sem valor "
            f"de eixo são contadas e não descartadas: {milhar(c.nao_desenhadas)} sem ponto. "
            f"Grupos: {frase_dos_grupos(contexto.grupos)}"
        ),
        matriz=contexto.matriz,
        metodo=(
            f"{nome_do_teste(contexto.metodo)} por feature, com o mínimo de "
            f"{milhar(contexto.min_por_grupo)} observações utilizáveis por grupo. "
            f"{contexto.correcao} a q < {decimal(contexto.alfa)}: "
            f"{milhar(c.sobreviventes)} de {total} sobrevive(m); "
            f"{frase_do_post_hoc(contexto.post_hoc_rota, contexto.post_hoc_metodo, sobrevivem)}"
            f".{sem_ponto} Sobre as {total} features da matriz, o menor q é {menor_q} e o menor p "
            f"bruto é {menor_p}"
        ),
    )


# ---------------------------------------------------------------------------
# A cor, do ciclo dos `rcParams`
# ---------------------------------------------------------------------------


def cores_das_classes() -> dict[Classe, Any]:
    """Uma cor por classe, tirada do ciclo dos `rcParams`.

    São as duas primeiras posições do ciclo, e não as do volcano: as duas figuras têm
    partições diferentes (quatro classes lá, duas aqui), e o que elas têm em comum é o texto
    da legenda.
    """
    return dict(zip(ORDEM_DAS_CLASSES, cores_do_ciclo(len(ORDEM_DAS_CLASSES)), strict=True))


# ---------------------------------------------------------------------------
# O construtor
# ---------------------------------------------------------------------------


def figura_da_anova(
    *,
    pontos: PontosDaAnova,
    contexto: ContextoDaAnova,
    recorte: Recorte,
) -> Figure:
    """A ANOVA: a dispersão, a linha do corte, as contagens na legenda e o recorte no rodapé.

    Desenha todas as features com q finito, sem amostragem. A dispersão leva
    `GID_DAS_FEATURES` e é o único artista que a identificação por clique indexa; a linha do
    corte é referência.

    A linha do corte é desenhada em coordenadas de dado, com `plot`, e não com `axhline`, cujo
    x fica em fração do eixo.
    """
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    paleta = cores_das_classes()

    if pontos.nomes:
        # `color=` e não `c=`: com `c=`, uma sequência do comprimento de `x` é lida como
        # valores a mapear por colormap, e a tupla RGB de uma classe só pode cair nesse caso.
        nuvem = eixo.scatter(
            pontos.x, pontos.y, color=_cor_dos_pontos(pontos, paleta), label=ROTULO_DA_SERIE
        )
        nuvem.set_gid(GID_DAS_FEATURES)
        eixo.plot(
            [float(pontos.x.min()), float(pontos.x.max())],
            [_altura_do_corte(contexto.alfa)] * 2,
            color=_cor_do_corte(),
            linestyle="--",
            label=_rotulo_do_corte(contexto.alfa),
        )
    else:
        # Sem pontos, a figura diz por quê em texto.
        eixo.text(
            0.5,
            0.5,
            frase_do_vazio(pontos),
            transform=eixo.transAxes,
            ha="center",
            va="center",
            fontsize=TAMANHO_DA_FONTE,
        )
        # E as marcas saem: o `matplotlib` desenharia os limites padrão de 0 a 1, um intervalo
        # que ninguém mediu.
        eixo.set_xticks([])
        eixo.set_yticks([])

    eixo.set_xlabel(ROTULO_DE_X)
    eixo.set_ylabel(ROTULO_DE_Y)
    eixo.set_title(f"{TITULO}\n{frase_da_populacao(pontos)}", fontsize=TAMANHO_DA_FONTE)
    eixo.legend(handles=_provas_da_legenda(pontos, contexto, paleta), fontsize=TAMANHO_DA_FONTE)
    aplica(figura, recorte)
    return figura


def _altura_do_corte(alfa: float) -> float:
    """`-log10(alfa)`, a altura acima da qual um ponto sobrevive ao FDR.

    Vem do mesmo `alfa` que classifica os pontos, e assim a contagem da legenda é o número de
    pontos acima da linha.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.log10(alfa))


def _rotulo_do_corte(alfa: float) -> str:
    return f"{ROTULO_DO_CORTE} (q = {decimal(alfa)})"


def _cor_dos_pontos(pontos: PontosDaAnova, paleta: dict[Classe, Any]) -> Any:
    """A cor da dispersão: uma só quando há uma classe só, e o vetor quando há mais.

    A imagem é a mesma, e o `matplotlib` deixa de resolver N cores a cada redesenho. Uma classe
    só é o caso comum, quando nenhuma feature sobrevive.
    """
    distintas = set(pontos.classes)
    if len(distintas) == 1:
        return paleta[next(iter(distintas))]
    return [paleta[classe] for classe in pontos.classes]


def _provas_da_legenda(
    pontos: PontosDaAnova, contexto: ContextoDaAnova, paleta: dict[Classe, Any]
) -> list[Line2D]:
    """As entradas de legenda: as duas classes, mais a linha do corte.

    As duas classes aparecem sempre, inclusive com `[0]`, porque o zero responde 'quantas
    sobreviveram?'. As contagens são as do desenho. Ver `rotulo_da_classe`.
    """
    c = pontos.contagens
    quantas = {
        Classe.SOBREVIVE: c.sobreviventes_desenhados,
        Classe.NAO: c.nao_significativas_desenhadas,
    }
    sem_ponto = {
        Classe.SOBREVIVE: c.sobreviventes_sem_ponto,
        Classe.NAO: c.nao_significativas_sem_ponto,
    }
    provas = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=paleta[classe],
            label=rotulo_da_classe(
                classe, quantas[classe], contexto.alfa, sem_ponto=sem_ponto[classe]
            ),
        )
        for classe in ORDEM_DAS_CLASSES
    ]
    if pontos.nomes:
        provas.append(
            Line2D(
                [], [], color=_cor_do_corte(), linestyle="--", label=_rotulo_do_corte(contexto.alfa)
            )
        )
    return provas


def _cor_do_corte() -> Any:
    """A cor da linha do corte: a terceira posição do ciclo.

    Sem `color=`, o `matplotlib` daria à linha a primeira cor do ciclo, que é a de
    `Classe.SOBREVIVE`, e o corte seria lido como a classe.
    """
    return cores_do_ciclo(len(ORDEM_DAS_CLASSES) + 1)[-1]

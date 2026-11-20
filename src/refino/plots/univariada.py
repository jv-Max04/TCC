"""As duas figuras univariadas por feature: `Fold Change` e `T-tests` do MetaboAnalyst.

O volcano cruza efeito e significância. Estas duas mostram cada metade separada, com as
features na ordem da matriz, que é como se procura uma feature nomeada numa tabela. Nenhuma
estatística corre aqui: `log2fc`, `pvalue` e `qvalue_fdr` saem do mesmo resultado univariado
que o volcano desenha, lido da mesma raiz `volcano` do corpo, para as três figuras de uma
execução mostrarem o mesmo número.

O eixo y dos testes t é o p bruto, como no volcano: o q depende do posto do p na tabela
inteira, então nenhuma altura corresponde ao corte e não há linha de corte. A figura da
ANOVA difere de propósito; lá o eixo é `-log10(q)`. Classificar e desenhar são separados: a
significância é contada sobre todas as features testadas, pelo q, e o desenho depende de o
valor do eixo y ser finito. `ContagensUnivariadas` traz as duas partições.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from matplotlib.lines import Line2D

from refino.plots.base import (
    GID_DAS_FEATURES,
    MatrizVaziaError,
    cores_do_ciclo,
    inteiro_do_fio,
    nova_figura,
    vetor_do_fio_ou_nucleo,
)
from refino.plots.recorte import TAMANHO_DA_FONTE, Recorte, aplica, decimal, milhar
from refino.plots.volcano import (
    Contraste,
    IntervaloAusente,
    IntervalosEscolhidos,
    desenha_as_barras,
    frase_das_barras,
    nome_do_teste,
    nomes_do_fio,
    prosa_da_normalidade,
    prova_das_barras,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.figure import Figure

__all__ = [
    "FOLD_CHANGE",
    "LOG2FC",
    "NEG_LOG10_P",
    "PVALOR",
    "ROTULO_DA_SERIE",
    "ROTULO_DE_X",
    "ROTULO_DO_ZERO",
    "TITULO_DOS_TESTES_T",
    "TITULO_DO_FOLD_CHANGE",
    "Classe",
    "ContagensUnivariadas",
    "PontosUnivariados",
    "cores_das_classes",
    "figura_do_fold_change",
    "figura_dos_testes_t",
    "frase_da_populacao",
    "frase_do_vazio",
    "pontos_do_fold_change",
    "pontos_dos_testes_t",
    "recorte_do_fold_change",
    "recorte_dos_testes_t",
    "rotulo_da_classe",
    "titulo_do_fold_change",
]

TITULO_DO_FOLD_CHANGE: Final = "Fold change por feature"

# 'o par de grupos' distingue este título do da figura da ANOVA, que fica no mesmo seletor e
# também fala de significância por feature. O que separa as duas é a população: o par
# declarado aqui, todos os grupos da matriz lá.
TITULO_DOS_TESTES_T: Final = "Significância por feature — o par de grupos"

ROTULO_DE_X: Final = "feature, na ordem da matriz"

ROTULO_DA_SERIE: Final = "features desenhadas"

ROTULO_DO_ZERO: Final = "sem alteração (log2 fc = 0)"

# Os dois nomes de quantidade acompanham o `y` dentro de `PontosUnivariados`, porque o mesmo
# campo carrega grandezas diferentes conforme a figura. São curtos porque entram na segunda
# linha do título, que já tem duas contagens; o nome completo fica no eixo.
FOLD_CHANGE: Final = "fold change"

PVALOR: Final = "p"

# Os rótulos do eixo y. O do fold change não nomeia o numerador: a frase com as duas medianas
# não cabe na vertical e cortaria o título e o rodapé. A ordem do par vai no título, que
# quebra em linhas.
LOG2FC: Final = "log2(fold change)"

NEG_LOG10_P: Final = "-log10(p-valor bruto)"


class Classe(StrEnum):
    """A classe de um ponto. São duas, e a classificação é pelo q nas duas figuras.

    O volcano tem quatro porque cruza direção com significância. Aqui a direção é o próprio
    eixo vertical de uma figura e não existe na outra.
    """

    SOBREVIVE = "sobrevive ao FDR"
    NAO = "não sobrevive"


ORDEM_DAS_CLASSES: Final = (Classe.SOBREVIVE, Classe.NAO)


@dataclass(frozen=True)
class ContagensUnivariadas:
    """Quantas features caíram em cada desfecho, em duas partições de `n_testadas`.

    - classificação: `sobreviventes + nao_significativas + sem_qvalue`, decidida por q finito
      e `q < alfa`;
    - desenho: `desenhadas + sem_valor`, decidida por o valor do eixo y ser finito.

    `sobreviventes` conta sobre todas as features, com o mesmo predicado que a API usa para
    `n_significant_fdr`, e não só sobre as desenháveis. Assim um sobrevivente sem ponto (um p
    igual a zero nos testes t, um `log2fc` indefinido no fold change) não some da contagem.

    `sobreviventes_desenhados + nao_sobreviventes_desenhados` soma `desenhadas`, e são essas
    duas que vão à legenda, porque a legenda descreve a nuvem. A segunda inclui a feature de q
    indefinido: ela é desenhada, com a cor de `Classe.NAO`, porque o q não está em nenhum eixo.
    """

    # A partição da classificação.
    sobreviventes: int
    nao_significativas: int
    sem_qvalue: int
    # A partição do desenho.
    desenhadas: int
    sem_valor: int
    # As duas classes dentro do desenho, que a legenda conta.
    sobreviventes_desenhados: int
    nao_sobreviventes_desenhados: int

    @property
    def sobreviventes_sem_ponto(self) -> int:
        """Quantas sobreviveram ao FDR e não têm ponto na nuvem."""
        return self.sobreviventes - self.sobreviventes_desenhados

    @property
    def nao_sobreviventes_sem_ponto(self) -> int:
        """Quantas não sobreviveram e não têm ponto na nuvem.

        O minuendo soma `nao_significativas` e `sem_qvalue` porque a classe desenhada `NAO`
        recebe as duas: quem não tem q também não sobrevive.
        """
        return self.nao_significativas + self.sem_qvalue - self.nao_sobreviventes_desenhados


@dataclass(frozen=True)
class PontosUnivariados:
    """Os pontos desenháveis de uma das duas figuras, na ordem em que as features chegam.

    `quantidade` diz o que o `y` carrega: o `log2(fold change)` ou o `-log10(p)`.

    `x` é o índice original da feature na matriz, e não a posição dentro do desenho. Uma
    feature sem valor deixa um vão, e a abscissa continua comparável com a tabela.

    `nomes` é o vetor que o painel recebe para a identificação por clique, filtrado do mesmo
    modo que `x` e `y`.

    `menor_q` é o mínimo sobre as features testadas, e não sobre as desenhadas.
    """

    quantidade: str
    nomes: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    classes: tuple[Classe, ...]
    n_testadas: int
    contagens: ContagensUnivariadas
    menor_q: float | None


# ---------------------------------------------------------------------------
# A partição da população, comum às duas figuras
# ---------------------------------------------------------------------------


def _pontos(
    *,
    quantidade: str,
    features: Any,
    valores: Any,
    qvalue_fdr: Any,
    n_features_testadas: Any,
    alfa: float,
) -> PontosUnivariados:
    """Monta os pontos e as contagens. As duas figuras só diferem no vetor que vai ao eixo y.

    Levanta `MatrizVaziaError` quando não veio feature nenhuma. Nenhuma feature desenhável é
    outro caso: é fato do dado, e a figura o declara.

    Um vetor mais curto que os nomes é completado com indefinidos, em vez de recusado como na
    figura da ANOVA: a população é definida pelos nomes, e valor ausente é contado e declarado.

    A significância é calculada sobre o vetor inteiro, e o que é desenhável sobre o vetor do
    eixo y. Ver `ContagensUnivariadas`.
    """
    nomes = nomes_do_fio(features)
    n_testadas = len(nomes)
    declarado = inteiro_do_fio(n_features_testadas)
    if declarado > n_testadas:
        n_testadas = declarado
        nomes = (*nomes, *("",) * (n_testadas - len(nomes)))
    if n_testadas == 0:
        raise MatrizVaziaError(
            f"a figura de {quantidade} não recebeu feature nenhuma: não há população para desenhar "
            "nem para declarar. Sem par de grupos a execução não corre o teste univariado, e é "
            "isso que é verdade — figura em branco seria indistinguível de figura quebrada"
        )

    v = _estica(vetor_do_fio_ou_nucleo(valores), n_testadas)
    q = _estica(vetor_do_fio_ou_nucleo(qvalue_fdr), n_testadas)

    indices = np.arange(n_testadas, dtype=float)
    # O que é desenhável é a única coisa que muda entre as duas figuras.
    desenhavel = np.isfinite(v)
    com_qvalue = np.isfinite(q)
    # A classificação corre sobre o vetor inteiro, com o mesmo predicado estrito que a API usa
    # para `n_significant_fdr`.
    significativa = com_qvalue & (q < alfa)

    classes = tuple(
        Classe.SOBREVIVE if bool(significativa[i]) else Classe.NAO
        for i in range(n_testadas)
        if bool(desenhavel[i])
    )
    contagens = ContagensUnivariadas(
        sobreviventes=int(significativa.sum()),
        nao_significativas=int((com_qvalue & ~significativa).sum()),
        sem_qvalue=int((~com_qvalue).sum()),
        desenhadas=int(desenhavel.sum()),
        sem_valor=int((~desenhavel).sum()),
        sobreviventes_desenhados=sum(1 for c in classes if c is Classe.SOBREVIVE),
        nao_sobreviventes_desenhados=sum(1 for c in classes if c is Classe.NAO),
    )
    finitos = q[np.isfinite(q)]
    return PontosUnivariados(
        quantidade=quantidade,
        nomes=tuple(nomes[i] for i in range(n_testadas) if bool(desenhavel[i])),
        x=indices[desenhavel],
        y=v[desenhavel],
        classes=classes,
        n_testadas=n_testadas,
        contagens=contagens,
        menor_q=float(finitos.min()) if finitos.size else None,
    )


def _estica(valores: np.ndarray, n: int) -> np.ndarray:
    """Completa com `nan` ou corta um vetor numérico para o comprimento da população.

    O que falta vira indefinido, e indefinido é contado e declarado.
    """
    if valores.size == n:
        return valores
    saida = np.full(n, np.nan, dtype=float)
    saida[: min(n, valores.size)] = valores[:n]
    return saida


def pontos_do_fold_change(
    *,
    features: Any,
    log2fc: Any,
    qvalue_fdr: Any,
    n_features_testadas: Any,
    alfa: float,
) -> PontosUnivariados:
    """Os pontos da figura de fold change: `log2fc` no eixo y, classificados pelo q.

    Usa o `log2`, como o MetaboAnalyst, porque nele dobrar e cair pela metade ficam simétricos
    em torno de zero.
    """
    return _pontos(
        quantidade=FOLD_CHANGE,
        features=features,
        valores=log2fc,
        qvalue_fdr=qvalue_fdr,
        n_features_testadas=n_features_testadas,
        alfa=alfa,
    )


def pontos_dos_testes_t(
    *,
    features: Any,
    pvalue: Any,
    qvalue_fdr: Any,
    n_features_testadas: Any,
    alfa: float,
) -> PontosUnivariados:
    """Os pontos da figura dos testes por feature: `-log10(p bruto)` no eixo y.

    A transformação é feita aqui, e não no servidor. Um p igual a zero é legítimo (feature
    constante dentro de cada grupo e diferente entre eles) e dá `+inf`, que não se desenha.
    Convertido no servidor ele chegaria como `null`, igual a um teste que não devolveu nada;
    aqui a feature é contada em `sem_valor`.
    """
    p = vetor_do_fio_ou_nucleo(pvalue)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = -np.log10(p)
    return _pontos(
        quantidade=PVALOR,
        features=features,
        valores=y,
        qvalue_fdr=qvalue_fdr,
        n_features_testadas=n_features_testadas,
        alfa=alfa,
    )


# ---------------------------------------------------------------------------
# O que as figuras dizem sobre si mesmas
# ---------------------------------------------------------------------------


def rotulo_da_classe(classe: Classe, quantas: int, alfa: float, *, sem_ponto: int = 0) -> str:
    """A entrada de legenda de uma classe: o nome, o corte e a contagem da nuvem.

    `quantas` conta os pontos desenhados, para o número ser contável na imagem. Quando
    `sem_ponto` não é zero, a entrada diz também quantas features da classe ficaram sem ponto.
    """
    nome = (
        f"sobrevive ao FDR (q < {decimal(alfa)})" if classe is Classe.SOBREVIVE else "não sobrevive"
    )
    if sem_ponto:
        return f"{nome} [{milhar(quantas)} desenhada(s), mais {milhar(sem_ponto)} sem ponto]"
    return f"{nome} [{milhar(quantas)}]"


def frase_da_populacao(pontos: PontosUnivariados) -> str:
    """A segunda linha do título: quantos pontos, de quantas features, e o que ficou de fora.

    Os dois primeiros números levam o denominador, porque 'pontos' é sobre o desenho e
    'sobrevivem' é sobre a classificação, e as duas populações divergem quando há sobrevivente
    sem ponto. As cláusulas seguintes só aparecem quando não são zero.

    `sem_qvalue` fica no rodapé, e não aqui, para o título não passar de duas linhas: a
    feature sem q continua desenhada, só perde a classe.
    """
    c = pontos.contagens
    total = milhar(pontos.n_testadas)
    partes = [
        f"{milhar(c.desenhadas)} ponto(s) de {total} features testadas",
        f"{milhar(c.sobreviventes)} de {total} sobrevive(m) ao FDR",
    ]
    if c.sobreviventes_sem_ponto:
        partes.append(f"{milhar(c.sobreviventes_sem_ponto)} sobrevivente(s) sem ponto")
    if c.sem_valor:
        partes.append(f"{milhar(c.sem_valor)} sem {pontos.quantidade} desenhável")
    return "; ".join(partes)


def frase_do_vazio(pontos: PontosUnivariados) -> str:
    """O texto escrito no lugar dos pontos quando nenhum é desenhável.

    Nomeia a quantidade que falta, porque as duas figuras ficam vazias por motivos diferentes,
    e traz a contagem de sobreviventes: sem ponto na tela, só ela separa 'nada significativo'
    de 'há, e não se consegue desenhar'.
    """
    c = pontos.contagens
    return (
        f"nenhuma das {milhar(pontos.n_testadas)} features tem {pontos.quantidade} desenhável.\n"
        f"Sobrevivem ao FDR: {milhar(c.sobreviventes)}.\n"
        "O teste correu — o que falta é o valor de eixo, e ele é contado e não descartado."
    )


def _rotulo_de_x() -> str:
    return ROTULO_DE_X


def titulo_do_fold_change(contraste: Contraste) -> str:
    """O título com o numerador nomeado.

    `(A, B)` e `(B, A)` desenham imagens espelhadas em y, e um `log2fc` lido ao contrário
    inverte a conclusão biológica. A frase fica no título, e não no eixo y, porque na vertical
    ela não cabe; o título quebra em linhas pela largura do eixo.
    """
    return (
        f"{TITULO_DO_FOLD_CHANGE} — mediana de {contraste.grupo_a} ÷ mediana de {contraste.grupo_b}"
    )


def _recorte(
    pontos: PontosUnivariados,
    contraste: Contraste,
    *,
    matriz: str,
    o_que_o_eixo_e: str,
    barras: str = "",
) -> Recorte:
    """O rodapé das duas figuras: população, procedência da matriz, método e correção.

    Entre as duas só mudam a matriz de onde o eixo saiu e a frase que descreve o eixo. A
    cláusula do sobrevivente sem ponto só aparece quando há algum.

    Sem par de grupos declarado levanta `MatrizVaziaError`: o teste univariado não correu, e
    o painel mostra a frase de população ausente.
    """
    if not contraste.grupo_a or not contraste.grupo_b:
        raise MatrizVaziaError(
            "a figura univariada não declara o par de grupos: a execução correu sem "
            "`group_a`/`group_b`, então nenhum teste univariado foi calculado. Isso é «não "
            "calculado», e não «calculado e vazio» — figura em branco seria indistinguível de "
            "figura quebrada"
        )
    c = pontos.contagens
    total = milhar(pontos.n_testadas)
    menor = "nenhum q é definido" if pontos.menor_q is None else decimal(pontos.menor_q)
    normalidade = prosa_da_normalidade(
        contraste.rota_da_normalidade, contraste.shapiro_pvalue, contraste.n_residuos
    )
    sem_ponto = (
        f". Destas, {milhar(c.sobreviventes_sem_ponto)} não tem ponto na nuvem: ficou sem "
        f"{pontos.quantidade} desenhável, que é ausência de valor de EIXO e não de significância"
        if c.sobreviventes_sem_ponto
        else ""
    )
    return Recorte(
        desenhadas=c.desenhadas,
        disponiveis=pontos.n_testadas,
        amostragem=(
            f"nenhuma — as {total} testadas entram, e as que ficam sem valor "
            f"de eixo são contadas e não descartadas: {milhar(c.sem_valor)} sem "
            f"{pontos.quantidade}, {milhar(c.sem_qvalue)} sem q. Contraste: "
            f"{contraste.grupo_a} (n = {milhar(contraste.n_a)}) contra "
            f"{contraste.grupo_b} (n = {milhar(contraste.n_b)})"
        ),
        matriz=matriz,
        metodo=(
            f"{nome_do_teste(contraste.parametrico)} por feature, {normalidade}. "
            f"{contraste.correcao} a q < {decimal(contraste.alfa)}: "
            f"{milhar(c.sobreviventes)} de {total} sobrevive(m), e o menor q "
            f"sobre as {total} testadas é {menor} — {o_que_o_eixo_e}{sem_ponto}{barras}"
        ),
    )


def recorte_do_fold_change(
    pontos: PontosUnivariados,
    contraste: Contraste,
    intervalos: IntervalosEscolhidos | IntervaloAusente | None = None,
) -> Recorte:
    """O rodapé da figura de fold change.

    A matriz declarada é a do fold change, e não a do p: o p sai da matriz processada e o
    fold change da matriz anterior à transformação, que é onde o MetaboAnalyst o calcula.

    Com `intervalos`, acrescenta a frase de `volcano.frase_das_barras`, a mesma das duas
    figuras que desenham o `log2fc`.
    """
    return _recorte(
        pontos,
        contraste,
        matriz=contraste.matriz_do_fold_change,
        o_que_o_eixo_e=(
            "a COR classifica por q e o eixo vertical é o fold change, então o corte não é uma "
            "altura desta figura"
        ),
        barras="" if intervalos is None else f". {frase_das_barras(intervalos, contraste)}",
    )


def recorte_dos_testes_t(pontos: PontosUnivariados, contraste: Contraste) -> Recorte:
    """O rodapé da figura dos testes por feature. A matriz declarada é a do p."""
    return _recorte(
        pontos,
        contraste,
        matriz=contraste.matriz_do_pvalue,
        o_que_o_eixo_e=(
            "a COR classifica por q e o eixo vertical é o p BRUTO — o q depende do POSTO do p na "
            "tabela inteira e não do valor dele, então o corte não é uma altura desta figura"
        ),
    )


# ---------------------------------------------------------------------------
# A cor, do ciclo dos `rcParams`
# ---------------------------------------------------------------------------


def cores_das_classes() -> dict[Classe, Any]:
    """Uma cor por classe, tirada do ciclo dos `rcParams`.

    As duas figuras usam a mesma paleta, para a mesma feature não mudar de cor entre duas abas
    da mesma execução.
    """
    return dict(zip(ORDEM_DAS_CLASSES, cores_do_ciclo(len(ORDEM_DAS_CLASSES)), strict=True))


# ---------------------------------------------------------------------------
# Os construtores
# ---------------------------------------------------------------------------


def figura_do_fold_change(
    *,
    pontos: PontosUnivariados,
    contraste: Contraste,
    recorte: Recorte,
    intervalos: IntervalosEscolhidos | IntervaloAusente | None = None,
) -> Figure:
    """O fold change por feature, com a referência do zero e as contagens na legenda.

    Desenha todas as features com valor, sem amostragem. A dispersão leva `GID_DAS_FEATURES`
    e é o único artista que a identificação por clique indexa; a linha do zero é referência.

    Com `intervalos`, desenha também as barras das features escolhidas, verticais, no índice
    de cada uma, de um limite ao outro. `recorte` tem de ter sido montado com os mesmos
    `intervalos`, porque é o rodapé que diz a regra e o nível.

    A linha do zero é desenhada em coordenadas de dado, com `plot`, e não com `axhline`, cujo
    x fica em fração do eixo.
    """
    return _desenha(
        pontos=pontos,
        recorte=recorte,
        titulo=titulo_do_fold_change(contraste),
        rotulo_de_y=LOG2FC,
        alfa=contraste.alfa,
        referencia=(0.0, ROTULO_DO_ZERO),
        intervalos=intervalos,
    )


def _barras_do_fold_change(intervalos: IntervalosEscolhidos) -> tuple[np.ndarray, np.ndarray]:
    """Os segmentos verticais das barras: no índice da feature, de um limite ao outro.

    Usa o índice, como o `x` dos pontos, para a barra ficar sobre o ponto da mesma feature.
    """
    n = intervalos.com_intervalo
    x = intervalos.indices.astype(float)
    return (
        np.column_stack([x, x, np.full(n, np.nan)]).ravel(),
        np.column_stack([intervalos.inferior, intervalos.superior, np.full(n, np.nan)]).ravel(),
    )


def figura_dos_testes_t(
    *,
    pontos: PontosUnivariados,
    contraste: Contraste,
    recorte: Recorte,
) -> Figure:
    """O `-log10(p bruto)` por feature, com as contagens por classe na legenda.

    Não tem linha de corte: o q depende do posto do p na tabela inteira, e nenhuma altura
    desta figura corresponde ao corte. Tem um único artista de dado.
    """
    return _desenha(
        pontos=pontos,
        recorte=recorte,
        titulo=TITULO_DOS_TESTES_T,
        rotulo_de_y=NEG_LOG10_P,
        alfa=contraste.alfa,
        referencia=None,
    )


def _desenha(
    *,
    pontos: PontosUnivariados,
    recorte: Recorte,
    titulo: str,
    rotulo_de_y: str,
    alfa: float,
    referencia: tuple[float, str] | None,
    intervalos: IntervalosEscolhidos | IntervaloAusente | None = None,
) -> Figure:
    """O desenho comum às duas figuras.

    Elas diferem no título, no rótulo do eixo y e na linha de referência. `intervalos` só é
    passado pelo fold change.
    """
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    paleta = cores_das_classes()
    barras = 0

    if pontos.nomes:
        if isinstance(intervalos, IntervalosEscolhidos):
            barras = desenha_as_barras(eixo, *_barras_do_fold_change(intervalos))
        nuvem = eixo.scatter(
            pontos.x, pontos.y, color=_cor_dos_pontos(pontos, paleta), label=ROTULO_DA_SERIE
        )
        nuvem.set_gid(GID_DAS_FEATURES)
        if referencia is not None:
            altura, rotulo = referencia
            eixo.plot(
                [float(pontos.x.min()), float(pontos.x.max())],
                [altura] * 2,
                color=_cor_da_referencia(),
                linestyle="--",
                label=rotulo,
            )
    else:
        eixo.text(
            0.5,
            0.5,
            frase_do_vazio(pontos),
            transform=eixo.transAxes,
            ha="center",
            va="center",
            fontsize=TAMANHO_DA_FONTE,
        )
        # Sem pontos, as marcas saem: o `matplotlib` desenharia os limites padrão de 0 a 1, um
        # intervalo que ninguém mediu.
        eixo.set_xticks([])
        eixo.set_yticks([])

    eixo.set_xlabel(_rotulo_de_x())
    eixo.set_ylabel(rotulo_de_y)
    eixo.set_title(f"{titulo}\n{frase_da_populacao(pontos)}", fontsize=TAMANHO_DA_FONTE)
    provas = _provas_da_legenda(pontos, alfa, paleta, referencia)
    if barras and isinstance(intervalos, IntervalosEscolhidos):
        provas.append(prova_das_barras(intervalos, barras))
    eixo.legend(handles=provas, fontsize=TAMANHO_DA_FONTE)
    aplica(figura, recorte)
    return figura


def _cor_dos_pontos(pontos: PontosUnivariados, paleta: dict[Classe, Any]) -> Any:
    """A cor da dispersão: uma só quando há uma classe só, e o vetor quando há mais.

    A imagem é a mesma, e o `scatter` com uma cor é mais rápido. Uma classe só é o caso comum,
    quando nenhuma feature sobrevive ao FDR.
    """
    distintas = set(pontos.classes)
    if len(distintas) == 1:
        return paleta[next(iter(distintas))]
    return [paleta[classe] for classe in pontos.classes]


def _provas_da_legenda(
    pontos: PontosUnivariados,
    alfa: float,
    paleta: dict[Classe, Any],
    referencia: tuple[float, str] | None,
) -> list[Line2D]:
    """As entradas de legenda: as duas classes, mais a referência quando existe.

    As duas classes aparecem sempre, inclusive com `[0]`, porque o zero responde 'quantas
    sobreviveram?'. As contagens são as do desenho. Ver `rotulo_da_classe`.
    """
    c = pontos.contagens
    quantas = {
        Classe.SOBREVIVE: c.sobreviventes_desenhados,
        Classe.NAO: c.nao_sobreviventes_desenhados,
    }
    sem_ponto = {
        Classe.SOBREVIVE: c.sobreviventes_sem_ponto,
        Classe.NAO: c.nao_sobreviventes_sem_ponto,
    }
    provas = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=paleta[classe],
            label=rotulo_da_classe(classe, quantas[classe], alfa, sem_ponto=sem_ponto[classe]),
        )
        for classe in ORDEM_DAS_CLASSES
    ]
    if referencia is not None and pontos.nomes:
        provas.append(
            Line2D([], [], color=_cor_da_referencia(), linestyle="--", label=referencia[1])
        )
    return provas


def _cor_da_referencia() -> Any:
    """A cor da linha de referência: a terceira posição do ciclo.

    Sem `color=`, o `matplotlib` daria à linha a primeira cor do ciclo, que é a de
    `Classe.SOBREVIVE`, e a referência seria lida como a classe.
    """
    return cores_do_ciclo(len(ORDEM_DAS_CLASSES) + 1)[-1]

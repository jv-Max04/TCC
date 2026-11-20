"""O que o pré-processamento fez com a distribuição: quatro figuras de antes e depois.

Respondem o que a imputação, a normalização, a transformação e o escalonamento fizeram com a
distribuição dos dados. É a tela 'Normalization → View Result' do MetaboAnalyst, com as vistas
por feature e por amostra, cada uma com densidade e caixas. A caixa é a de Tukey (corpo do 1.º
ao 3.º quartil, mediana no traço, bigodes até `1,5 x IQR` do quartil), que é o que o
MetaboAnalyst e `Axes.bxp` desenham, para as duas telas mostrarem a mesma imagem do mesmo dado.

A densidade é sobre as médias de eixo, não sobre as células. A distribuição das células
mistura a variação entre features com a variação entre amostras e responde outra pergunta.

As caixas são uma amostra, e a figura o declara. O MetaboAnalyst mostra no máximo 50 por
limitação de espaço; aqui o teto é o mesmo nos dois eixos, o sorteio é semeado e feito pelo
servidor, e a semente vem no corpo para o rodapé escrevê-la. Este módulo não sorteia. O teto
também mantém o redesenho abaixo de 100 ms, o limite clássico de resposta instantânea à
manipulação direta (Miller 1968; Card, Robertson e Mackinlay 1991; Nielsen 1993): o custo de
redesenho cresce com o número de caixas, e a população inteira ficaria muito acima disso.

A população é a interseção dos nomes de feature das duas matrizes, porque etapas do pipeline
removem features e a comparação só faz sentido sobre o mesmo conjunto. O rodapé das quatro
figuras escreve 'N de M features' com `n_features_comparadas` e `n_features_antes`, os dois
vindos do corpo, e a cláusula 'as que sobreviveram ao corte'.

As caixas por amostra são a transposta das caixas por feature, e não uma redução delas: cada
caixa resume uma amostra sobre todas as features comuns. O vetor de médias por amostra tem um
número por amostra e serve só à densidade.

O 'antes' é a matriz ingerida e o 'depois' a processada. Os dois lados têm eixos separados,
com limites próprios, porque as escalas diferem por ordens de grandeza depois de um `log10`
seguido de `auto`; num eixo só, a curva de maior escala esmagaria a outra contra o zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, TypedDict

import numpy as np
from scipy.stats import gaussian_kde

from refino.plots.base import (
    NAO_DECLARADA,
    MatrizVaziaError,
    RecusaDoConstrutorError,
    inteiro_do_fio,
    nova_figura,
    numero_do_fio,
    objetos_do_fio,
    texto_do_fio,
)
from refino.plots.recorte import (
    FATOR_DE_LARGURA,
    PONTOS_POR_POLEGADA,
    RESERVA_MAXIMA,
    TAMANHO_DA_FONTE,
    Recorte,
    aplica,
    largura_util_em_pontos,
    marcas_que_cabem,
    milhar,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "ANTES",
    "CORTE_DA_DENSIDADE",
    "DEPOIS",
    "NAO_DECLARADA",
    "PONTOS_DA_CURVA",
    "ROTULO_DAS_CAIXAS",
    "ROTULO_DA_DENSIDADE",
    "SOBREVIVERAM",
    "SUBSTANTIVO_DAS_AMOSTRAS",
    "SUBSTANTIVO_DAS_FEATURES",
    "TITULO_DAS_CAIXAS",
    "TITULO_DAS_CAIXAS_POR_AMOSTRA",
    "TITULO_DA_DENSIDADE_POR_AMOSTRA",
    "TITULO_DA_DENSIDADE_POR_FEATURE",
    "UNIDADE_POR_AMOSTRA",
    "UNIDADE_POR_FEATURE",
    "Caixa",
    "CaixaDeAmostra",
    "CorpoDaCaixa",
    "DensidadeIndefinidaError",
    "ParDesalinhadoError",
    "caixas_do_fio",
    "caixas_por_amostra_do_fio",
    "curva_da_densidade",
    "figura_da_densidade_por_amostra",
    "figura_da_densidade_por_feature",
    "figura_das_caixas_por_amostra",
    "figura_das_caixas_por_feature",
    "frase_da_densidade",
    "frase_das_caixas",
    "recorte_da_densidade_por_amostra",
    "recorte_da_densidade_por_feature",
    "recorte_das_caixas",
    "recorte_das_caixas_por_amostra",
    "rotulo_do_artista",
    "rotulos_cabem",
    "rotulos_cabem_na_altura",
]

TITULO_DAS_CAIXAS: Final = "Distribuição por feature, antes e depois"

TITULO_DAS_CAIXAS_POR_AMOSTRA: Final = "Distribuição por amostra, antes e depois"

# Substantivo do eixo x de cada figura de caixas. As duas desenham o mesmo objeto sobre eixos
# opostos da matriz, e o substantivo é o que diz qual pergunta a figura responde.
SUBSTANTIVO_DAS_FEATURES: Final = "features"

SUBSTANTIVO_DAS_AMOSTRAS: Final = "amostras"

TITULO_DA_DENSIDADE_POR_FEATURE: Final = "Densidade das médias por feature, antes e depois"

TITULO_DA_DENSIDADE_POR_AMOSTRA: Final = "Densidade das médias por amostra, antes e depois"

ROTULO_DAS_CAIXAS: Final = "valor medido"

ROTULO_DA_DENSIDADE: Final = "densidade"

UNIDADE_POR_FEATURE: Final = "média da feature sobre as amostras"

UNIDADE_POR_AMOSTRA: Final = "média da amostra sobre as features"

# A cláusula que declara a interseção. É uma constante porque as quatro figuras a repetem.
SOBREVIVERAM: Final = "as que sobreviveram ao corte"

# Os dois lados. O rótulo da matriz ('ingerida', 'processada') diz de onde o número veio, mas
# não qual dos lados é o de partida; quem não conhece o pipeline precisa de 'antes' e 'depois'.
ANTES: Final = "antes"

DEPOIS: Final = "depois"

# O `density()` do R, que desenha as curvas do MetaboAnalyst, avalia numa grelha de `n = 512`
# por omissão. O mesmo número torna as duas curvas comparáveis à vista.
PONTOS_DA_CURVA: Final = 512

# Margem em que a curva é avaliada além do dado, em larguras de banda. `density()` do R usa
# `cut = 3`: a grelha se estende três bandas para cada lado e a cauda fecha.
CORTE_DA_DENSIDADE: Final = 3.0

# Mínimo de valores distintos para haver densidade. `gaussian_kde` estima a banda pela
# dispersão da amostra; com um valor só, ou todos iguais, a covariância é singular.
MINIMO_PARA_DENSIDADE: Final = 2


class DensidadeIndefinidaError(RecusaDoConstrutorError):
    """Não há dispersão sobre a qual estimar uma densidade.

    Com um valor só, ou com todos iguais, `gaussian_kde` levanta `LinAlgError`, que fala de
    matriz singular e não do dado. Desenhar uma reta no lugar afirmaria distribuição uniforme
    sobre uma população que não tem distribuição.

    É separada de `MatrizVaziaError` porque 'não veio população' é defeito de quem chama, e 'a
    população veio e é constante' é fato do dado: uma matriz de uma amostra só, ou de uma
    feature só, chega aqui legitimamente.
    """


class ParDesalinhadoError(RecusaDoConstrutorError):
    """As caixas do 'antes' e do 'depois' não descrevem as mesmas unidades.

    O defeito que a guarda apanha é invisível na imagem: com o número certo de caixas de cada
    lado e a ordem errada, a caixa `i` da esquerda e a da direita seriam de features diferentes
    numa figura bem formada. O servidor casa os lados por nome, e esta guarda impede que um
    chamador desfaça isso sem nada acusar.
    """


class CorpoDaCaixa:
    """Os cinco números de uma caixa de Tukey e o que se faz com eles.

    É uma base sem campos de dataclass. As duas caixas diferem só no campo do nome (`feature`
    ou `amostra`); declarar os outros aqui como campos reordenaria o construtor das subclasses,
    e implementar `desenhavel` e `estatisticas` em cada uma duplicaria a conta. A base anota os
    campos para o mypy e implementa o comportamento, e cada subclasse é o dataclass.

    `n_discrepantes` é `int | None`. `None` quer dizer que não havia valor nenhum sobre o qual
    decidir, o que é diferente de 'nenhum ponto ficou fora dos bigodes'. As duas contagens saem
    separadas na frase da figura.
    """

    bigode_inferior: float
    q1: float
    mediana: float
    q3: float
    bigode_superior: float
    n_discrepantes: int | None

    @property
    def nome(self) -> str:
        """O rótulo da unidade que esta caixa resume: a feature ou a amostra.

        O desenho, o rótulo dos artistas e a guarda do par leem `nome`, e assim servem às duas
        figuras sem pôr nome de amostra num campo chamado `feature`.
        """
        raise NotImplementedError  # pragma: no cover - as duas subclasses respondem

    @property
    def desenhavel(self) -> bool:
        """Se os cinco números são finitos. Caixa com `nan` não é desenhada, é contada."""
        return bool(
            np.isfinite(
                [self.bigode_inferior, self.q1, self.mediana, self.q3, self.bigode_superior]
            ).all()
        )

    def estatisticas(self) -> dict[str, Any]:
        """O dicionário que `Axes.bxp` consome, com os nomes que ele espera.

        `fliers` vai vazio porque o corpo traz a contagem dos discrepantes e não os valores. A
        contagem é declarada na frase da figura.
        """
        return {
            "label": self.nome,
            "whislo": self.bigode_inferior,
            "q1": self.q1,
            "med": self.mediana,
            "q3": self.q3,
            "whishi": self.bigode_superior,
            "fliers": [],
        }


@dataclass(frozen=True)
class Caixa(CorpoDaCaixa):
    """A caixa de Tukey de uma feature: a distribuição dela sobre as amostras.

    Os nomes dos campos são os de `CaixaDaDistribuicao` em `api/esquemas.py`. O pacote não pode
    importar a API, então a forma é repetida; manter os mesmos nomes evita um segundo
    vocabulário na conversão.
    """

    feature: str
    bigode_inferior: float
    q1: float
    mediana: float
    q3: float
    bigode_superior: float
    n_discrepantes: int | None

    @property
    def nome(self) -> str:
        return self.feature


@dataclass(frozen=True)
class CaixaDeAmostra(CorpoDaCaixa):
    """A caixa de Tukey de uma amostra: a distribuição dela sobre as features.

    É a transposta da caixa por feature, e não uma redução dela, por isso é um tipo próprio.
    Montá-la a partir de `medias_por_amostra_*` daria uma caixa sobre um número por amostra com
    o nome de uma caixa sobre todas as features.

    Os nomes dos campos são os de `CaixaPorAmostra` em `api/esquemas.py`.
    """

    amostra: str
    bigode_inferior: float
    q1: float
    mediana: float
    q3: float
    bigode_superior: float
    n_discrepantes: int | None

    @property
    def nome(self) -> str:
        return self.amostra


# ---------------------------------------------------------------------------
# A fronteira: o que chega do fio não é tipado
# ---------------------------------------------------------------------------


class _CamposDaCaixa(TypedDict):
    """Os seis campos que as duas caixas partilham, já convertidos."""

    bigode_inferior: float
    q1: float
    mediana: float
    q3: float
    bigode_superior: float
    n_discrepantes: int | None


def _campos_da_caixa(bruto: dict[str, Any]) -> _CamposDaCaixa:
    """Os seis números com guarda. É a conversão das duas caixas."""
    return {
        "bigode_inferior": numero_do_fio(bruto.get("bigode_inferior")),
        "q1": numero_do_fio(bruto.get("q1")),
        "mediana": numero_do_fio(bruto.get("mediana")),
        "q3": numero_do_fio(bruto.get("q3")),
        "bigode_superior": numero_do_fio(bruto.get("bigode_superior")),
        "n_discrepantes": _contagem(bruto.get("n_discrepantes")),
    }


def caixas_do_fio(valores: Any) -> tuple[Caixa, ...]:
    """Lê a lista de caixas por feature do corpo de `/execucoes`, com guarda em cada campo.

    Nada do que vem do fio é tipado: um campo pode chegar `null`, a lista pode chegar `None` e
    um elemento pode não ser objeto. O que não é número vira `nan`, que torna a caixa não
    desenhável; o construtor a conta na frase da figura.

    Lista que não é lista devolve tupla vazia. Quem recusa é o construtor, com
    `MatrizVaziaError`, que o painel troca por texto.
    """
    return tuple(
        Caixa(feature=str(bruto.get("feature", "")), **_campos_da_caixa(bruto))
        for bruto in objetos_do_fio(valores)
    )


def caixas_por_amostra_do_fio(valores: Any) -> tuple[CaixaDeAmostra, ...]:
    """Lê a lista de caixas por amostra do corpo de `/execucoes`, com a mesma guarda.

    A única diferença é a chave lida, `amostra`. Ler `feature` aqui devolveria todas as caixas
    com nome em branco, e a guarda do par alinhado passaria por serem todas igualmente vazias.
    """
    return tuple(
        CaixaDeAmostra(amostra=str(bruto.get("amostra", "")), **_campos_da_caixa(bruto))
        for bruto in objetos_do_fio(valores)
    )


def _contagem(valor: Any) -> int | None:
    """Um inteiro não negativo, ou `None`. O `None` do fio é preservado.

    Trocar `None` por `0` apagaria a distinção entre 'nenhum ponto ficou fora' e 'não havia
    sobre o que decidir'.
    """
    if isinstance(valor, bool) or not isinstance(valor, int):
        return None
    return valor if valor >= 0 else None


def curva_da_densidade(
    finitos: np.ndarray, *, margem_em_bandas: float = CORTE_DA_DENSIDADE
) -> tuple[np.ndarray, np.ndarray]:
    """A grelha e a densidade estimada sobre `finitos`.

    É usada pelas figuras de densidade e pelo violino de RSD (`dispersao.py`), para as duas
    famílias desenharem curvas com a mesma banda e o mesmo número de pontos.

    O padrão de `margem_em_bandas` é o do `density()` do R, que estende a grelha três larguras
    de banda para cada lado. O violino passa zero, porque o RSD não é negativo e uma cauda
    abaixo do mínimo observado desenharia dispersão que ninguém mediu.

    Não há guarda contra população degenerada; quem chama decide o que fazer. A pré-condição é
    `np.unique(finitos).size >= MINIMO_PARA_DENSIDADE`, sem a qual `gaussian_kde` levanta
    `LinAlgError`.
    """
    estimativa = gaussian_kde(finitos)
    margem = margem_em_bandas * float(estimativa.factor) * float(finitos.std(ddof=1))
    grelha = np.linspace(
        float(finitos.min()) - margem, float(finitos.max()) + margem, PONTOS_DA_CURVA
    )
    return grelha, np.asarray(estimativa(grelha), dtype=float)


# ---------------------------------------------------------------------------
# O que a figura diz sobre a própria população
# ---------------------------------------------------------------------------


def frase_das_caixas(
    n_desenhadas: int, n_fora: int, n_nao_avaliaveis: int, n_sem_caixa: int
) -> str:
    """A segunda linha do título das caixas, derivada da população.

    - `n_fora`: quantos pontos ficaram fora dos bigodes e não estão desenhados.
    - `n_nao_avaliaveis`: em quantas unidades não havia valor nenhum sobre que decidir.
    - `n_sem_caixa`: em quantas os cinco números não chegaram finitos, logo não há caixa.

    A primeira sai sempre, inclusive zero, porque 'nenhum ponto fora dos bigodes' é um fato. As
    outras duas só saem quando não são zero.

    As duas últimas coincidem no servidor atual, que põe `n_discrepantes = None` na coluna
    toda-`nan`, a mesma que dá cinco `nan`. Não são a mesma coisa, e a frase conta cada uma.
    """
    plural = "caixa" if n_desenhadas == 1 else "caixas"
    partes = [
        f"{milhar(n_desenhadas)} {plural}",
        (
            "nenhum ponto fora dos bigodes"
            if n_fora == 0
            else f"{milhar(n_fora)} ponto(s) fora dos bigodes, contados e não desenhados"
        ),
    ]
    if n_nao_avaliaveis:
        partes.append(f"{milhar(n_nao_avaliaveis)} sem valor em amostra nenhuma")
    if n_sem_caixa:
        partes.append(f"{milhar(n_sem_caixa)} sem caixa desenhável")
    return "; ".join(partes)


def frase_da_densidade(n_usados: int, n_nao_finitos: int) -> str:
    """A segunda linha do título da densidade, com o não finito contado.

    Um valor não finito no vetor de médias não entra na estimativa, e a frase diz quantos
    saíram para a curva não descrever uma população menor que a declarada.
    """
    corpo = f"{milhar(n_usados)} valor(es)"
    if n_nao_finitos:
        return f"{corpo}; {milhar(n_nao_finitos)} não finito(s), fora da estimativa"
    return corpo


def rotulo_do_artista(caixas: Sequence[CorpoDaCaixa], chave: str, posicao: int) -> str:
    """O nome que cada artista de `bxp` leva, para o cursor dizer qual unidade é.

    `Axes.bxp` devolve os artistas agrupados (`boxes`, `medians`, `whiskers`, `caps`) com
    rótulos internos como `_child7`. Numa figura de caixas a identificação do painel usa o
    rótulo do artista, porque o índice do `pick_event` é o do ponto dentro do artista, não o da
    unidade.

    Os bigodes e as pontas vêm aos pares, dois por caixa na ordem inferior-superior, por isso o
    índice é dividido por dois.
    """
    aos_pares = chave in {"whiskers", "caps"}
    indice = posicao // 2 if aos_pares else posicao
    if not 0 <= indice < len(caixas):  # pragma: no cover - `bxp` devolve o que recebeu
        return chave
    nome = caixas[indice].nome
    if not aos_pares:
        return f"{nome} · {'caixa' if chave == 'boxes' else 'mediana'}"
    ponta = "inferior" if posicao % 2 == 0 else "superior"
    return f"{nome} · {'bigode' if chave == 'whiskers' else 'ponta'} {ponta}"


def rotulos_cabem(figura: Figure, n_caixas: int) -> bool:
    """Se os nomes das unidades cabem como marcas do eixo, pela largura útil da figura.

    Um rótulo rodado a 90 graus ocupa a altura de uma linha de texto, e cabem tantos quantos a
    largura útil comportar. Não há número fixo de caixas: a resposta depende da largura da
    figura e do tamanho da fonte. Na figura padrão as 50 caixas do teto não cabem, e as duas
    figuras de caixas saem sem marcas; a identificação por clique responde qual unidade é.

    A conta é `recorte.marcas_que_cabem`. O que esta função decide é o comprimento disponível:
    a largura útil da figura inteira, porque a figura de caixas não tem barra de cor.
    """
    return marcas_que_cabem(largura_util_em_pontos(figura), n_caixas)


def rotulos_cabem_na_altura(figura: Figure, nomes: Sequence[str]) -> bool:
    """Se os nomes rodados cabem na altura, que `rotulos_cabem` não verifica.

    Rodado a 90 graus, um rótulo ocupa a altura do seu comprimento. Nomes de amostra são bem
    mais longos que um `RT_mz`, e sem este teste as marcas seriam impressas por cima do rodapé.
    São dois predicados porque as duas perguntas têm respostas independentes.

    O limite é `RESERVA_MAXIMA`, a fração máxima da altura que o texto no fundo da figura pode
    ocupar, e a largura de caractere é a estimativa de `FATOR_DE_LARGURA`. Na figura padrão de
    4,5 polegadas isso dá 33 caracteres.

    Sem nome nenhum a resposta é `False`: marcas vazias reservariam espaço para texto que não
    existe.
    """
    if not nomes:  # pragma: no cover - o construtor recusa a lista vazia antes de chegar aqui
        return False
    util = float(figura.get_figheight()) * PONTOS_POR_POLEGADA * RESERVA_MAXIMA
    mais_longo = max(len(nome) for nome in nomes)
    return bool(mais_longo * TAMANHO_DA_FONTE * FATOR_DE_LARGURA <= util)


# ---------------------------------------------------------------------------
# Os recortes: os números que só o servidor tem
# ---------------------------------------------------------------------------


def _procedencia(matriz_antes: Any, matriz_depois: Any) -> str:
    """O campo `matriz` do rodapé para uma figura de dois lados: as duas matrizes, nomeadas."""
    return (
        f"«antes» {texto_do_fio(matriz_antes, NAO_DECLARADA)} → "
        f"«depois» {texto_do_fio(matriz_depois, NAO_DECLARADA)}"
    )


def _exige_populacao_declarada(n_antes: int, o_que: str) -> None:
    """Levanta `MatrizVaziaError` quando a matriz de origem não tem população declarada.

    É `MatrizVaziaError`, e não `RecorteInvalidoError`, porque é a que o painel apanha para
    mostrar texto no lugar da figura. Outra exceção levantada dentro de um slot do Qt vai para
    o `stderr` e o painel para de atualizar.
    """
    if n_antes < 1:
        raise MatrizVaziaError(
            f"{o_que} não declara de quantas features a matriz ingerida saiu: `n_features_antes` "
            "veio ilegível ou zero. Sem execução não há população, e figura em branco seria "
            "indistinguível de figura quebrada"
        )


def recorte_das_caixas(
    *,
    n_caixas: Any,
    n_comparadas: Any,
    n_antes: Any,
    amostrada: Any,
    seed: Any,
    teto_de_caixas: Any,
    coeficiente_de_tukey: float,
    matriz_antes: Any = None,
    matriz_depois: Any = None,
) -> Recorte:
    """O `Recorte` das caixas: quantas de quantas, com que semente, de que matrizes e o método.

    `coeficiente_de_tukey` é parâmetro. O servidor calcula o bigode com esse número e a figura
    o escreve; como o pacote não pode importar a API, quem passa o valor é o painel.

    `amostrada` vem do fio e não é derivada de `n_caixas < n_comparadas`: com exatamente 50
    features comparáveis não há sorteio, e a comparação diria o mesmo que diz para 6.035.
    """
    desenhadas, comparadas, disponiveis = (
        inteiro_do_fio(n_caixas),
        inteiro_do_fio(n_comparadas),
        inteiro_do_fio(n_antes),
    )
    _exige_populacao_declarada(disponiveis, "a figura das caixas")
    houve_sorteio = amostrada is True
    amostragem = (
        f"{milhar(desenhadas)} de {milhar(comparadas)} features comuns às duas matrizes, "
        f"sorteadas com a semente {inteiro_do_fio(seed)} — teto de "
        f"{inteiro_do_fio(teto_de_caixas)} caixas, o do MetaboAnalyst; {SOBREVIVERAM}"
        if houve_sorteio
        else f"nenhuma — as {milhar(comparadas)} features comuns às duas matrizes, {SOBREVIVERAM}"
    )
    return Recorte(
        desenhadas=comparadas,
        disponiveis=disponiveis,
        amostragem=amostragem,
        matriz=_procedencia(matriz_antes, matriz_depois),
        metodo=(
            f"caixa de Tukey, sem teste estatístico: corpo do 1.º ao 3.º quartil, mediana no "
            f"traço, e bigodes até o ponto mais afastado ainda dentro de "
            f"{coeficiente_de_tukey:g} x IQR do quartil. Os discrepantes NÃO são desenhados — o "
            f"corpo traz a contagem deles, não os valores — e a contagem está no título de cada "
            f"lado. Os dois lados têm eixos próprios: as escalas diferem por ordens de grandeza"
        ),
    )


def recorte_das_caixas_por_amostra(
    *,
    n_caixas: Any,
    n_amostras_comparadas: Any,
    n_comparadas: Any,
    n_antes: Any,
    amostrada: Any,
    seed: Any,
    teto_de_caixas: Any,
    coeficiente_de_tukey: float,
    matriz_antes: Any = None,
    matriz_depois: Any = None,
) -> Recorte:
    """O `Recorte` das caixas por amostra, que tem duas populações.

    Cada caixa é sobre as features comuns, e há uma caixa por amostra. O par 'N de M features'
    do rodapé conta features, porque `Recorte.texto()` escreve essa palavra, e descreve o que
    está dentro de cada caixa. O número de amostras vai na cláusula de amostragem.

    O denominador do sorteio é `n_amostras_comparadas`, a população de que o servidor sorteou.
    Hoje ela é igual a `n_amostras`, porque nenhuma etapa remove amostras, mas isso não é
    invariante.

    `amostrada` vem do fio e é a do eixo das amostras; os dois eixos decidem em separado.
    """
    desenhadas, comparadas, disponiveis = (
        inteiro_do_fio(n_caixas),
        inteiro_do_fio(n_comparadas),
        inteiro_do_fio(n_antes),
    )
    _exige_populacao_declarada(disponiveis, "a figura das caixas por amostra")
    amostras = inteiro_do_fio(n_amostras_comparadas)
    houve_sorteio = amostrada is True
    amostragem = (
        f"{milhar(desenhadas)} de {milhar(amostras)} amostras comuns às duas matrizes, sorteadas "
        f"com a semente {inteiro_do_fio(seed)} — teto de {inteiro_do_fio(teto_de_caixas)} caixas, "
        f"o do MetaboAnalyst; cada caixa é sobre as {milhar(comparadas)} features comuns, "
        f"{SOBREVIVERAM}"
        if houve_sorteio
        else (
            f"nenhuma — as {milhar(amostras)} amostras comuns às duas matrizes, cada caixa sobre "
            f"as {milhar(comparadas)} features comuns, {SOBREVIVERAM}"
        )
    )
    return Recorte(
        desenhadas=comparadas,
        disponiveis=disponiveis,
        amostragem=amostragem,
        matriz=_procedencia(matriz_antes, matriz_depois),
        # A frase é curta de propósito: `aplica` recusa o rodapé que precise de mais de
        # `RESERVA_MAXIMA` da altura.
        metodo=(
            f"caixa de Tukey por AMOSTRA — a distribuição de cada amostra sobre as features, a "
            f"transposta da vista por feature e não uma redução dela —, sem teste estatístico: "
            f"corpo do 1.º ao 3.º quartil, mediana no traço, e bigodes até o ponto mais afastado "
            f"ainda dentro de {coeficiente_de_tukey:g} x IQR do quartil. Os discrepantes NÃO são "
            f"desenhados — o corpo traz a contagem deles — e ela está no título de cada lado. Os "
            f"dois lados têm eixos próprios: as escalas diferem por ordens de grandeza"
        ),
    )


def recorte_da_densidade_por_feature(
    *,
    n_comparadas: Any,
    n_antes: Any,
    matriz_antes: Any = None,
    matriz_depois: Any = None,
) -> Recorte:
    """O `Recorte` da densidade por feature.

    Sem amostragem: a curva usa a interseção inteira dos nomes.
    """
    comparadas, disponiveis = inteiro_do_fio(n_comparadas), inteiro_do_fio(n_antes)
    _exige_populacao_declarada(disponiveis, "a densidade por feature")
    return Recorte(
        desenhadas=comparadas,
        disponiveis=disponiveis,
        amostragem=(
            f"nenhuma — a curva é sobre as médias das {milhar(comparadas)} features comuns "
            f"às duas matrizes, {SOBREVIVERAM}"
        ),
        matriz=_procedencia(matriz_antes, matriz_depois),
        metodo=_METODO_DA_DENSIDADE.format(unidade=UNIDADE_POR_FEATURE),
    )


def recorte_da_densidade_por_amostra(
    *,
    n_comparadas: Any,
    n_antes: Any,
    n_amostras: Any,
    matriz_antes: Any = None,
    matriz_depois: Any = None,
) -> Recorte:
    """O `Recorte` da densidade por amostra. O denominador continua a ser em features.

    A curva tem um ponto por amostra, e cada ponto é uma média sobre as features comuns. O par
    'N de M features' do rodapé conta features, porque `Recorte.texto()` escreve essa palavra;
    o número de amostras vai na cláusula de amostragem.
    """
    comparadas, disponiveis = inteiro_do_fio(n_comparadas), inteiro_do_fio(n_antes)
    _exige_populacao_declarada(disponiveis, "a densidade por amostra")
    return Recorte(
        desenhadas=comparadas,
        disponiveis=disponiveis,
        amostragem=(
            f"nenhuma — a curva é sobre as médias das {milhar(inteiro_do_fio(n_amostras))} "
            f"amostras, cada média sobre as {milhar(comparadas)} features comuns às duas "
            f"matrizes, {SOBREVIVERAM}"
        ),
        matriz=_procedencia(matriz_antes, matriz_depois),
        metodo=_METODO_DA_DENSIDADE.format(unidade=UNIDADE_POR_AMOSTRA),
    )


_METODO_DA_DENSIDADE: Final = (
    "estimativa de densidade por núcleo gaussiano (`scipy.stats.gaussian_kde`, banda pela regra de "
    "Scott), sem teste estatístico. A curva é sobre a {unidade} — e NÃO sobre as células, que são "
    "outra distribuição com o mesmo desenho. Os dois lados têm eixos próprios: as escalas diferem "
    "por ordens de grandeza."
)


# ---------------------------------------------------------------------------
# Os quatro construtores
# ---------------------------------------------------------------------------


def figura_das_caixas_por_feature(
    *,
    antes: Sequence[Caixa],
    depois: Sequence[Caixa],
    recorte: Recorte,
    matriz_antes: Any,
    matriz_depois: Any,
) -> Figure:
    """As caixas de Tukey por feature, o 'antes' à esquerda e o 'depois' à direita.

    Desenha a população inteira que recebe. O sorteio é do servidor, com a semente que o rodapé
    declara.

    Os dois lados têm de trazer as mesmas features, na mesma ordem; ver `ParDesalinhadoError`.
    """
    return _figura_de_caixas(
        antes=antes,
        depois=depois,
        recorte=recorte,
        matriz_antes=matriz_antes,
        matriz_depois=matriz_depois,
        titulo=TITULO_DAS_CAIXAS,
        substantivo=SUBSTANTIVO_DAS_FEATURES,
    )


def figura_das_caixas_por_amostra(
    *,
    antes: Sequence[CaixaDeAmostra],
    depois: Sequence[CaixaDeAmostra],
    recorte: Recorte,
    matriz_antes: Any,
    matriz_depois: Any,
) -> Figure:
    """As caixas de Tukey por amostra, o 'antes' à esquerda e o 'depois' à direita.

    É a 'Sample View' do MetaboAnalyst e a transposta da figura por feature: cada caixa resume
    uma amostra sobre as features comuns, e responde se as amostras ficaram comparáveis entre
    si. Não é o vetor de médias por amostra desenhado como caixa.

    Desenha a população inteira que recebe. O sorteio é do servidor, com o mesmo teto.
    """
    return _figura_de_caixas(
        antes=antes,
        depois=depois,
        recorte=recorte,
        matriz_antes=matriz_antes,
        matriz_depois=matriz_depois,
        titulo=TITULO_DAS_CAIXAS_POR_AMOSTRA,
        substantivo=SUBSTANTIVO_DAS_AMOSTRAS,
    )


def figura_da_densidade_por_feature(
    *,
    antes: Sequence[float] | np.ndarray,
    depois: Sequence[float] | np.ndarray,
    recorte: Recorte,
    matriz_antes: Any,
    matriz_depois: Any,
) -> Figure:
    """A densidade das médias por feature, antes e depois. Uma média por coluna, não por célula."""
    return _figura_de_densidade(
        antes=antes,
        depois=depois,
        recorte=recorte,
        matriz_antes=matriz_antes,
        matriz_depois=matriz_depois,
        titulo=TITULO_DA_DENSIDADE_POR_FEATURE,
        unidade=UNIDADE_POR_FEATURE,
    )


def figura_da_densidade_por_amostra(
    *,
    antes: Sequence[float] | np.ndarray,
    depois: Sequence[float] | np.ndarray,
    recorte: Recorte,
    matriz_antes: Any,
    matriz_depois: Any,
) -> Figure:
    """A densidade das médias por amostra, antes e depois. Uma média por linha, não por célula."""
    return _figura_de_densidade(
        antes=antes,
        depois=depois,
        recorte=recorte,
        matriz_antes=matriz_antes,
        matriz_depois=matriz_depois,
        titulo=TITULO_DA_DENSIDADE_POR_AMOSTRA,
        unidade=UNIDADE_POR_AMOSTRA,
    )


# ---------------------------------------------------------------------------
# O que os quatro partilham
# ---------------------------------------------------------------------------


def _figura_de_caixas(
    *,
    antes: Sequence[CorpoDaCaixa],
    depois: Sequence[CorpoDaCaixa],
    recorte: Recorte,
    matriz_antes: Any,
    matriz_depois: Any,
    titulo: str,
    substantivo: str,
) -> Figure:
    """Os dois lados de caixas, em eixos próprios. Os dois construtores públicos delegam aqui.

    Eles diferem só no título e no substantivo do eixo x.
    """
    _exige_par_alinhado(antes, depois, substantivo)
    figura = nova_figura()
    esquerda, direita = figura.subplots(1, 2)
    for eixo, caixas, lado, rotulo in (
        (esquerda, antes, ANTES, texto_do_fio(matriz_antes, NAO_DECLARADA)),
        (direita, depois, DEPOIS, texto_do_fio(matriz_depois, NAO_DECLARADA)),
    ):
        _desenha_as_caixas(figura, eixo, caixas, lado, rotulo, substantivo)
    figura.suptitle(titulo)
    aplica(figura, recorte)
    return figura


def _exige_par_alinhado(
    antes: Sequence[CorpoDaCaixa], depois: Sequence[CorpoDaCaixa], substantivo: str
) -> None:
    """Recusa quando as duas listas não descrevem as mesmas unidades, na mesma ordem."""
    nomes_antes = [c.nome for c in antes]
    nomes_depois = [c.nome for c in depois]
    if nomes_antes == nomes_depois:
        return
    if len(nomes_antes) != len(nomes_depois):
        raise ParDesalinhadoError(
            f"o «antes» traz {len(nomes_antes)} caixa(s) e o «depois» {len(nomes_depois)}. Os dois "
            "lados descrevem a mesma interseção de nomes, e comprimentos diferentes significam que "
            "um deles é de outra população"
        )
    divergentes = [(a, d) for a, d in zip(nomes_antes, nomes_depois, strict=True) if a != d]
    raise ParDesalinhadoError(
        f"{len(divergentes)} de {len(nomes_antes)} posições trazem {substantivo} diferentes nos "
        f"dois lados; a primeira é {divergentes[0]!r}. Lado a lado, isso desenha o rótulo certo "
        f"sobre o conteúdo de outra unidade, e a imagem fica bem formada"
    )


def _desenha_as_caixas(
    figura: Figure,
    eixo: Axes,
    caixas: Sequence[CorpoDaCaixa],
    lado: str,
    matriz: str,
    substantivo: str,
) -> None:
    """Um lado da figura das caixas: `bxp`, os rótulos dos artistas e a contagem no título."""
    desenhaveis = [c for c in caixas if c.desenhavel]
    if not desenhaveis:
        raise MatrizVaziaError(
            f"o lado «{matriz}» tem {len(caixas)} caixa(s) e nenhuma com os cinco números finitos: "
            "não há o que desenhar. Um eixo montado e vazio ao lado de um cheio lê-se como figura "
            "quebrada, e é o outro lado que ficaria a parecer errado"
        )
    desenhados = eixo.bxp(
        [c.estatisticas() for c in desenhaveis],
        showfliers=False,
        # Com `patch_artist=False` a caixa é uma `Line2D` em `ax.lines`, que os testes leem e o
        # cursor do painel alcança. Com `True` ela viraria `Patch` e sairia das duas.
        patch_artist=False,
    )
    for chave, artistas in desenhados.items():
        for posicao, artista in enumerate(artistas):
            artista.set_label(rotulo_do_artista(desenhaveis, chave, posicao))

    eixo.set_ylabel(ROTULO_DAS_CAIXAS)
    nomes = [c.nome for c in desenhaveis]
    # A largura decide quantos rótulos cabem lado a lado, e o comprimento decide se cada um
    # cabe na altura depois de rodado. As duas condições têm de valer.
    if rotulos_cabem(figura, len(desenhaveis)) and rotulos_cabem_na_altura(figura, nomes):
        eixo.tick_params(axis="x", labelrotation=90, labelsize=TAMANHO_DA_FONTE)
    else:
        # Sem marca, em vez de marca ilegível: dezenas de nomes empilhados viram uma barra
        # preta. A identificação por clique diz qual unidade é, pelo rótulo de cada artista.
        eixo.set_xticks([])
    eixo.set_xlabel(f"{milhar(len(desenhaveis))} {substantivo}, uma caixa cada")
    eixo.set_title(
        f"{lado} — {matriz}\n"
        + frase_das_caixas(
            n_desenhadas=len(desenhaveis),
            n_fora=sum(c.n_discrepantes or 0 for c in desenhaveis),
            n_nao_avaliaveis=sum(1 for c in caixas if c.n_discrepantes is None),
            n_sem_caixa=len(caixas) - len(desenhaveis),
        ),
        fontsize=TAMANHO_DA_FONTE,
    )


def _figura_de_densidade(
    *,
    antes: Sequence[float] | np.ndarray,
    depois: Sequence[float] | np.ndarray,
    recorte: Recorte,
    matriz_antes: Any,
    matriz_depois: Any,
    titulo: str,
    unidade: str,
) -> Figure:
    """As duas curvas, em eixos próprios. Os dois construtores públicos delegam aqui.

    Eles diferem só no título e na unidade do eixo.
    """
    figura = nova_figura()
    esquerda, direita = figura.subplots(1, 2)
    for eixo, valores, lado, rotulo in (
        (esquerda, antes, ANTES, texto_do_fio(matriz_antes, NAO_DECLARADA)),
        (direita, depois, DEPOIS, texto_do_fio(matriz_depois, NAO_DECLARADA)),
    ):
        _desenha_a_densidade(eixo, valores, lado, rotulo, unidade)
    figura.suptitle(titulo)
    aplica(figura, recorte)
    return figura


def _desenha_a_densidade(
    eixo: Axes,
    valores: Sequence[float] | np.ndarray,
    lado: str,
    matriz: str,
    unidade: str,
) -> None:
    """Um lado da densidade: filtra o não finito, estima, avalia numa grelha e conta o que caiu."""
    todos = np.asarray(valores, dtype=float)
    if todos.size == 0:
        raise MatrizVaziaError(
            f"o lado «{matriz}» não trouxe média nenhuma: não há população para estimar densidade. "
            "Uma curva em branco ao lado de uma cheia lê-se como figura quebrada"
        )
    finitos = todos[np.isfinite(todos)]
    if finitos.size < MINIMO_PARA_DENSIDADE or np.unique(finitos).size < MINIMO_PARA_DENSIDADE:
        raise DensidadeIndefinidaError(
            f"o lado «{matriz}» tem {finitos.size} valor(es) finito(s) e "
            f"{np.unique(finitos).size} distinto(s): não há dispersão sobre a qual estimar uma "
            "densidade. Uma reta desenhada aqui afirmaria distribuição uniforme sobre uma "
            "população que não tem distribuição nenhuma"
        )

    grelha, densidade = curva_da_densidade(finitos)
    eixo.plot(grelha, densidade, label=f"{lado} — {matriz}")

    eixo.set_xlabel(unidade)
    eixo.set_ylabel(ROTULO_DA_DENSIDADE)
    eixo.set_title(
        f"{lado} — {matriz}\n{frase_da_densidade(finitos.size, int(todos.size - finitos.size))}",
        fontsize=TAMANHO_DA_FONTE,
    )

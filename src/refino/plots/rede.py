"""A figura da rede de correlação parcial esparsa desenviesada (DSPC).

É uma figura de grafo: o que se desenha é uma relação entre pares de features, e a posição
dos vértices é inventada por este módulo. A disposição é circular, na ordem em que as
features chegam, porque é determinística; uma disposição por forças é iterativa e dependeria
de semente. Os eixos não têm marcas, e o rótulo avisa que a posição não é medida. As arestas
são duas coleções de segmentos, uma por sinal da correlação parcial, cada uma com a contagem
na legenda. Positiva inclui o zero, para toda aresta cair em exatamente uma das duas.

Rede sem aresta é resultado, e não figura em branco: os vértices são desenhados todos, a
frase do desfecho vai dentro do eixo e a contagem com o limiar vai no rodapé. O corte de
sobrevivência é no p cru de cada aresta; o q de Benjamini-Hochberg sobre os `k(k-1)/2`
pares acompanha cada aresta sem ser o critério, e o rodapé diz quantas arestas passariam
com o mesmo `alfa` aplicado ao q. O teto de features é o de `statistics.clustering`, medido
para figuras com um rótulo por feature; esta não desenha rótulos, então ele é conservador.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from matplotlib.collections import LineCollection

from refino.plots.base import (
    GID_DAS_FEATURES,
    GID_REQUEBRAVEL,
    MatrizVaziaError,
    RecusaDoConstrutorError,
    cores_do_ciclo,
    denominador_ou_recusa,
    exige_abaixo_do_teto,
    inteiro_do_fio,
    nova_figura,
    numero_do_fio,
    objetos_do_fio,
    texto_do_fio,
)
from refino.plots.recorte import (
    TAMANHO_DA_FONTE,
    Recorte,
    aplica,
    decimal,
    frase_contada,
    legenda_fora,
    milhar,
)
from refino.statistics.clustering import TETO_DE_FEATURES_PARA_DESENHAR

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "ESPESSURA_DA_ARESTA",
    "FAIXA_DO_DESFECHO",
    "MARGEM_DO_CIRCULO",
    "METODO_NAO_DECLARADO",
    "ROTULO_DAS_NEGATIVAS",
    "ROTULO_DAS_POSITIVAS",
    "ROTULO_DO_EIXO",
    "TAMANHO_DO_VERTICE",
    "TITULO_DA_REDE",
    "Aresta",
    "Rede",
    "RedeDesalinhadaError",
    "figura_da_rede_de_correlacao_parcial",
    "posicoes_do_circulo",
    "recorte_da_rede",
    "rede_do_fio",
    "veredito_da_rede",
]

TITULO_DA_REDE: Final = "Rede de correlação parcial esparsa desenviesada (DSPC)"
"""O título nomeia os dois passos do método por extenso antes da sigla.

'Rede de correlação parcial' sozinho descreveria também o graphical lasso sem a correção de
viés. A sigla é a da entrada `Correlation Networks (DSPC)` do catálogo.
"""

ROTULO_DAS_POSITIVAS: Final = "arestas positivas"
ROTULO_DAS_NEGATIVAS: Final = "arestas negativas"
"""Os rótulos das duas séries de aresta.

Positiva é associação direta condicionada às outras features, e negativa é inversa. O sinal
da correlação parcial é o oposto do sinal da precisão, por definição.
"""

ROTULO_DO_EIXO: Final = "disposição circular — a posição NÃO é medida"
"""O rótulo de x é uma ressalva, e não o nome de uma quantidade.

A posição de um vértice é inventada por este módulo, e sem o aviso ela seria lida como
coordenada.
"""

METODO_NAO_DECLARADO: Final = "método não declarado"
"""O texto usado quando o corpo não traz `metodo`.

Escrever o nome do método por omissão faria a figura afirmar que a correção de viés correu.
"""

TAMANHO_DO_VERTICE: Final = 26.0
"""Área do marcador de vértice, em pontos quadrados.

Um pouco maior que o do dendrograma porque aqui o marcador é o alvo do clique: não há rótulo
de feature.
"""

ESPESSURA_DA_ARESTA: Final = 1.0
"""Espessura de toda aresta, em pontos.

É constante de propósito. Escalar pela força da correlação parcial poria na imagem uma
quantidade sem escala nem legenda, que seria lida por comparação entre arestas.
"""

MARGEM_DO_CIRCULO: Final = 0.09
"""Folga de cada lado, em fração da extensão dos dados.

Com o limite exatamente no raio, o `matplotlib` desenha metade do marcador dos vértices
extremos fora da área de dado.

É fração, e não limite fixo, por causa de `set_aspect(..., adjustable="datalim")`: com
limites fixados à mão o `matplotlib` os ignora para manter a proporção, e avisa.
"""


FAIXA_DO_DESFECHO: Final = 0.55
"""A margem vertical do eixo quando não há aresta, em fração da extensão dos dados.

A frase do desfecho é escrita dentro da área de dado, e com `MARGEM_DO_CIRCULO` ela cobriria
os vértices mais baixos do círculo. 0,55 é o valor em que a frase curta de `_declara` fica
afastada dos vértices nos tamanhos de painel medidos, inclusive no painel largo e baixo da
janela estreita, onde 0,40 deixa menos que o raio do marcador. Valores maiores só encolhem o
círculo.

Vale só no caso sem arestas: com arestas não há frase a acomodar, e como a escala é igual
nos dois eixos a faixa encolheria o desenho.
"""


class RedeDesalinhadaError(RecusaDoConstrutorError):
    """A carga da rede não é coerente consigo mesma.

    Cobre quatro formas que dariam uma imagem bem formada e errada:

    - `n_arestas` diferente do tamanho da lista: a figura desenharia um número de segmentos e
      escreveria outro no título;
    - aresta que nomeia feature fora de `features`: a aresta não teria posição;
    - aresta de uma feature com ela mesma: um segmento de comprimento zero;
    - aresta com p ou correlação parcial não finitos: não entraria em nenhuma das duas
      coleções de sinal.

    `n_arestas` vem com o resultado porque o núcleo conta as arestas por um caminho diferente
    do que monta a lista. Recalculá-lo aqui a partir da lista apagaria essa conferência.
    """


@dataclass(frozen=True)
class Aresta:
    """Uma aresta desenhável: o par, a correlação parcial com sinal, o p e o q.

    `pvalue` e `qvalue_fdr` não têm valor padrão: uma aresta não é desenhada sem a confiança
    nela.
    """

    feature_a: str
    feature_b: str
    correlacao_parcial: float
    pvalue: float
    qvalue_fdr: float

    @property
    def positiva(self) -> bool:
        """Se a aresta entra na coleção das positivas. O zero conta como positivo."""
        return self.correlacao_parcial >= 0.0


@dataclass(frozen=True)
class Rede:
    """A rede desenhável, com o que a torna interpretável na própria imagem.

    Não é o resultado do núcleo: traz só as arestas que passaram o corte, sem as matrizes
    `(k, k)`, porque é isso que a figura desenha.
    """

    features: tuple[str, ...]
    """Os nomes das features que entraram na rede, na ordem da disposição circular.

    É a população dos vértices, e não a das arestas: uma feature sem aresta continua sendo um
    vértice."""

    arestas: tuple[Aresta, ...]
    """As que sobreviveram ao corte no p cru. Pode estar vazia, e vazio é resultado."""

    n_arestas: int
    """Quantas sobreviveram, pela contagem do núcleo. Ver `RedeDesalinhadaError`."""

    n_pares_testados: int
    """`k(k-1)/2`. É o denominador da contagem de arestas: sete arestas sobre 28 pares e sobre
    1.770 são achados de tamanhos diferentes."""

    alfa: float
    penalidade: float
    n_amostras_usadas: int
    metodo: str
    """O estimador e a correção, por extenso, como vieram do corpo.

    É campo, e não constante deste módulo, para a figura declarar o que a execução fez. Um
    rótulo escrito aqui diria 'com correção de viés' mesmo que ela não tivesse corrido."""

    @property
    def n_vertices(self) -> int:
        return len(self.features)

    @property
    def n_positivas(self) -> int:
        """Contadas sobre as arestas. Somadas às negativas dão `len(arestas)`."""
        return sum(1 for a in self.arestas if a.positiva)

    @property
    def n_negativas(self) -> int:
        return len(self.arestas) - self.n_positivas

    @property
    def n_isoladas(self) -> int:
        """Features que não aparecem em nenhuma aresta, contadas pelo nome."""
        tocadas = {a.feature_a for a in self.arestas} | {a.feature_b for a in self.arestas}
        return sum(1 for nome in self.features if nome not in tocadas)

    @property
    def n_sob_o_mesmo_corte_no_q(self) -> int:
        """Quantas arestas sobreviveriam com o mesmo `alfa` aplicado ao q de BH.

        Responde 'e com correção de múltiplos testes?' sem criar um segundo limiar: o corte é
        o `alfa` que veio com o resultado.
        """
        abaixo = [
            a for a in self.arestas if math.isfinite(a.qvalue_fdr) and a.qvalue_fdr < self.alfa
        ]
        return len(abaixo)


def rede_do_fio(
    *,
    features: Any,
    arestas: Any,
    n_arestas: Any,
    n_pares_testados: Any,
    alfa: Any,
    penalidade: Any,
    n_amostras_usadas: Any,
    metodo: Any,
) -> Rede:
    """Uma `Rede` a partir do que o corpo JSON ou o núcleo trazem, com as guardas de fronteira.

    Recebe os campos um a um, e não o dicionário, para que a leitura de cada chave do corpo
    fique na janela, onde o teste de contrato com a API confere as chaves contra o esquema.

    Confere `n_arestas` contra o tamanho da lista: os dois números vêm de caminhos diferentes
    no núcleo, e divergência significa que um deles está errado.
    """
    nomes = tuple(str(f) for f in features) if isinstance(features, (list, tuple)) else ()
    if not nomes:
        raise MatrizVaziaError(
            "a rede chegou sem feature nenhuma: não há vértice para desenhar, e figura em branco "
            "seria indistinguível de figura quebrada. Rede que CORREU e não achou aresta é outro "
            "fato, e chega com as features e com a lista de arestas vazia"
        )
    if len(set(nomes)) != len(nomes):
        raise RedeDesalinhadaError(
            f"há nome de feature repetido nos {len(nomes)} vértices da rede: a posição de um "
            "vértice é resolvida pelo nome, e com um nome repetido as arestas de uma das cópias "
            "iriam parar à outra — com a figura bem formada e as ligações trocadas"
        )

    lidas = tuple(
        Aresta(
            feature_a=str(bruta.get("feature_a", "")),
            feature_b=str(bruta.get("feature_b", "")),
            correlacao_parcial=numero_do_fio(bruta.get("correlacao_parcial")),
            pvalue=numero_do_fio(bruta.get("pvalue")),
            qvalue_fdr=numero_do_fio(bruta.get("qvalue_fdr")),
        )
        for bruta in objetos_do_fio(arestas)
    )
    declaradas = inteiro_do_fio(n_arestas)
    _exige_coerencia(lidas, nomes, declaradas)

    return Rede(
        features=nomes,
        arestas=lidas,
        n_arestas=declaradas,
        n_pares_testados=inteiro_do_fio(n_pares_testados),
        alfa=numero_do_fio(alfa),
        penalidade=numero_do_fio(penalidade),
        n_amostras_usadas=inteiro_do_fio(n_amostras_usadas),
        metodo=texto_do_fio(metodo, METODO_NAO_DECLARADO),
    )


def _exige_coerencia(arestas: Sequence[Aresta], nomes: Sequence[str], declaradas: int) -> None:
    """As quatro conferências de `RedeDesalinhadaError`."""
    if declaradas != len(arestas):
        raise RedeDesalinhadaError(
            f"o resultado declara {declaradas} aresta(s) e a lista traz {len(arestas)}. Os dois "
            "números saem de caminhos DIFERENTES no núcleo — `count_nonzero` sobre a máscara e o "
            "laço que monta a lista —, e divergirem significa que um deles está errado. A figura "
            "desenharia o comprimento da lista e escreveria o outro número no título"
        )
    conhecidos = set(nomes)
    fora = sorted({n for a in arestas for n in (a.feature_a, a.feature_b) if n not in conhecidos})
    if fora:
        raise RedeDesalinhadaError(
            f"há aresta que nomeia feature fora dos {len(nomes)} vértices da rede: {fora[:5]}. "
            "Uma aresta sem vértice não tem posição, e descartá-la em silêncio deixaria a contagem "
            "do rodapé a descrever mais ligações do que a imagem mostra"
        )
    laco = sorted({a.feature_a for a in arestas if a.feature_a == a.feature_b})
    if laco:
        raise RedeDesalinhadaError(
            f"há aresta de uma feature com ela própria: {laco[:5]}. A diagonal da precisão sai "
            "`nan` de propósito porque não é aresta, e um laço aqui seria um segmento de "
            "comprimento zero — dois vértices coincidentes que o arnês conta e ninguém vê"
        )
    indefinidas = sorted(
        {
            f"{a.feature_a}--{a.feature_b}"
            for a in arestas
            if not math.isfinite(a.correlacao_parcial) or not math.isfinite(a.pvalue)
        }
    )
    if indefinidas:
        raise RedeDesalinhadaError(
            f"há aresta com correlação parcial ou `p` não finito: {indefinidas[:5]}. Uma aresta "
            "está na lista porque sobreviveu ao corte no `p`, e um `p` indefinido é incoerente com "
            "isso — além de a deixar fora das duas coleções de sinal, contada no rodapé e ausente "
            "da imagem"
        )


def posicoes_do_circulo(quantos: int) -> np.ndarray:
    """Os vértices igualmente espaçados no círculo unitário, em um array `(quantos, 2)`.

    `theta_i = 2 pi i / k`, com `x = cos(theta)` e `y = sin(theta)`: começa em `(1, 0)` e anda
    no sentido anti-horário. A posição depende só do índice e do total, então a mesma entrada
    dá a mesma imagem sem semente.
    """
    if quantos < 1:
        raise MatrizVaziaError(
            f"pedi a disposição de {quantos} vértice(s): não há círculo de nada para desenhar"
        )
    angulos = 2.0 * np.pi * np.arange(quantos, dtype=float) / float(quantos)
    return np.column_stack((np.cos(angulos), np.sin(angulos)))


def veredito_da_rede(rede: Rede) -> str:
    """A frase que diz o que a contagem de arestas significa. São três casos:

    - nenhuma aresta: a rede correu e nada sobreviveu, o que é achado e não defeito;
    - arestas a p cru e nenhuma sob o mesmo corte no q: a contagem é só a p cru;
    - arestas, algumas também sob o corte no q: quantas sobrevivem à correção.

    O corte aplicado ao q é o mesmo `alfa` que veio com o resultado.
    """
    if rede.n_arestas == 0:
        return (
            "NENHUM par sobreviveu ao corte: a rede correu e não achou associação parcial "
            "nenhuma. É ACHADO e não defeito — com este número de amostras, e com a penalidade a "
            "zerar a maior parte dos pares antes do teste, a ausência de aresta é o desfecho que "
            "este corpus produz. Os vértices desenhados são as features que entraram no ajuste"
        )
    if rede.n_sob_o_mesmo_corte_no_q == 0:
        return (
            "As arestas desenhadas estão a p CRU, e NENHUMA delas sobrevive ao mesmo corte "
            "aplicado ao q de Benjamini-Hochberg. Sobre este número de pares simultâneos, a "
            "leitura honesta é que a rede mostra candidatos e não achados — o q de cada aresta "
            "viaja ao lado, e quem lê decide"
        )
    return (
        f"As arestas desenhadas estão a p cru, e {milhar(rede.n_sob_o_mesmo_corte_no_q)} de "
        f"{milhar(rede.n_arestas)} sobrevivem também ao mesmo corte aplicado ao q de "
        "Benjamini-Hochberg. As duas contagens vão lado a lado de propósito: a primeira é o que a "
        "imagem mostra, e a segunda é o que resta depois da correção de múltiplos testes"
    )


def recorte_da_rede(
    *,
    rede: Rede,
    n_features_totais: Any,
    matriz: str,
    amostragem: str,
) -> Recorte:
    """O rodapé: quantas features entraram, com que método, que penalidade e que corte.

    O número de features da matriz vem de quem chama, porque não está nos vértices nem nas
    arestas.

    A penalidade entra no texto do método porque governa a esparsidade: a mesma matriz com
    outra penalidade dá uma rede de tamanho muito diferente. O rodapé também diz que o corte
    é no p cru, sobre `k(k-1)/2` testes simultâneos.
    """
    das_arestas = frase_contada(
        rede.n_arestas,
        nenhuma="nenhuma aresta sobreviveu ao corte",
        uma="1 aresta sobreviveu ao corte",
        algumas="arestas sobreviveram ao corte",
    )
    # A contagem sob o corte no q fica no veredito, e não aqui: num campo fixo ela escreveria
    # '0 arestas' numa figura que já disse não ter aresta.
    das_isoladas = frase_contada(
        rede.n_isoladas,
        nenhuma="nenhuma feature ficou isolada",
        uma="1 feature ficou isolada, sem aresta nenhuma",
        algumas="features ficaram isoladas, sem aresta nenhuma",
    )
    return Recorte(
        desenhadas=rede.n_vertices,
        disponiveis=denominador_ou_recusa(n_features_totais, de="a rede de correlação parcial"),
        amostragem=amostragem,
        matriz=matriz,
        metodo=(
            f"{rede.metodo}, sobre as {milhar(rede.n_amostras_usadas)} amostras de estudo, com "
            f"penalidade {decimal(rede.penalidade)} escolhida por validação cruzada. "
            f"Corte no p CRU a {decimal(rede.alfa)}, sobre "
            f"{milhar(rede.n_pares_testados)} pares testados ao mesmo tempo — o q de "
            "Benjamini-Hochberg viaja em cada aresta e NÃO é o critério de sobrevivência. "
            f"{das_arestas}; {das_isoladas}. {veredito_da_rede(rede)}"
        ),
    )


def _declara(eixo: Axes, frase: str) -> None:
    """Escreve a frase do desfecho dentro da área de dado.

    A marca `GID_REQUEBRAVEL` faz `recorte.aplica` requebrar a frase pela largura final do
    eixo, que só é conhecida depois de `tight_layout` e da legenda de fora.
    """
    texto = eixo.text(
        0.0,
        0.01,
        frase,
        transform=eixo.transAxes,
        ha="left",
        va="bottom",
        fontsize=TAMANHO_DA_FONTE,
    )
    texto.set_gid(GID_REQUEBRAVEL)


def figura_da_rede_de_correlacao_parcial(*, rede: Rede, recorte: Recorte) -> Figure:
    """A rede de correlação parcial esparsa desenviesada, em disposição circular.

    As arestas são desenhadas antes dos vértices. Assim nenhum vértice fica escondido por uma
    aresta, e a ordem das coleções no eixo é a que o arnês de teste das figuras espera. Pelo
    mesmo motivo nenhuma `Line2D` é desenhada.

    A escala é igual nos dois eixos: sem isso o círculo sairia como elipse e o comprimento
    aparente de uma aresta dependeria da direção.

    Os vértices levam `GID_DAS_FEATURES`, um ponto por feature na ordem de `rede.features`, e
    a identificação é por clique. Sessenta rótulos em volta de um círculo seriam ilegíveis, e
    é o texto que domina o custo de redesenho.
    """
    exige_abaixo_do_teto(
        rede.n_vertices, TETO_DE_FEATURES_PARA_DESENHAR, de="a rede de correlação parcial"
    )
    posicoes = posicoes_do_circulo(rede.n_vertices)
    onde = {nome: indice for indice, nome in enumerate(rede.features)}

    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    # Três cores: as duas séries de aresta e a dos vértices. Com a mesma cor, vértice e aresta
    # positiva não se distinguiriam.
    cores = cores_do_ciclo(3)

    maos: list[Artist] = []
    for rotulo, cor, positiva in (
        (ROTULO_DAS_POSITIVAS, cores[0], True),
        (ROTULO_DAS_NEGATIVAS, cores[1], False),
    ):
        segmentos = [
            [
                (float(posicoes[onde[a.feature_a], 0]), float(posicoes[onde[a.feature_a], 1])),
                (float(posicoes[onde[a.feature_b], 0]), float(posicoes[onde[a.feature_b], 1])),
            ]
            for a in rede.arestas
            if a.positiva is positiva
        ]
        if not segmentos:
            # Coleção vazia não é desenhada: entraria na legenda como '0'. A ausência daquele
            # sinal está contada no rodapé.
            continue
        feixe = LineCollection(
            segmentos,
            colors=cor,
            linewidths=ESPESSURA_DA_ARESTA,
            label=f"{rotulo} — {milhar(len(segmentos))}",
        )
        eixo.add_collection(feixe)
        maos.append(feixe)

    vertices = eixo.scatter(
        posicoes[:, 0],
        posicoes[:, 1],
        s=TAMANHO_DO_VERTICE,
        color=cores[2],
        label=f"{milhar(rede.n_vertices)} features",
        zorder=3,
    )
    vertices.set_gid(GID_DAS_FEATURES)
    maos.append(vertices)

    if not rede.arestas:
        _declara(
            eixo,
            # Curta de propósito: a frase divide a área de dado com os vértices, e cada linha
            # a mais cobre uma fileira deles. A explicação completa vai no rodapé.
            f"Nenhuma aresta sobreviveu ao corte em {milhar(rede.n_pares_testados)} pares "
            "testados. A rede CORREU: a ausência de ligação é o resultado.",
        )

    eixo.set_xlabel(ROTULO_DO_EIXO)
    eixo.set_ylabel("")
    eixo.set_title(
        f"{TITULO_DA_REDE} · {milhar(rede.n_arestas)} de "
        f"{milhar(rede.n_pares_testados)} pares a p < {decimal(rede.alfa)}",
        fontsize=TAMANHO_DA_FONTE,
    )
    # Sem marcas nos eixos: a coordenada de um vértice é inventada, e uma marca numérica
    # convidaria a lê-la como medida.
    eixo.set_xticks([])
    eixo.set_yticks([])
    # Sem moldura: com `datalim` a caixa do eixo é o retângulo inteiro e o círculo fica no
    # meio, então a moldura pareceria uma figura encolhida dentro de uma caixa.
    eixo.set_frame_on(False)
    # `add_collection` não ajusta os limites, ao contrário de `plot`. Hoje a dispersão dos
    # vértices já os ajusta; a chamada garante a vista se ela deixar de ser desenhada.
    eixo.autoscale_view()
    # A faixa de baixo é maior quando não há aresta, para a frase do desfecho não cobrir os
    # vértices mais baixos.
    eixo.margins(
        x=MARGEM_DO_CIRCULO,
        y=FAIXA_DO_DESFECHO if not rede.arestas else MARGEM_DO_CIRCULO,
    )
    # `datalim` e não `box`: com `box` o eixo encolhe até ficar quadrado, `tight_layout` não
    # consegue acomodar as decorações e o rodapé se sobrepõe ao eixo. Com `datalim` quem se
    # estica é o limite de dado, que aqui é derivado do círculo.
    eixo.set_aspect("equal", adjustable="datalim")
    legenda_fora(eixo, maos)
    aplica(figura, recorte)
    return figura

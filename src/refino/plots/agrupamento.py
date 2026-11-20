"""Figuras de agrupamento: dendrograma das amostras, mapa de calor, K-means e mapa auto-organizável.

O mapa de calor desenha a matriz que `statistics.clustering` escalonou por feature. Como o
z-score é centrado em zero, a paleta é a divergente de `base.PALETA_DIVERGENTE`, com o zero no
tom neutro, e os limites são simétricos (`vmin = -L`, `vmax = +L`). Com os limites assimétricos
que o `imshow` escolhe por padrão, o tom neutro cairia na média dos extremos e não no zero.

A barra de cor é um eixo numérico com `ScalarFormatter` e recebe a vírgula decimal de
`recorte.aplica`. Os rótulos de feature e de amostra são postos por `set_*ticklabels`, que
instala um `FuncFormatter`, e não são reescritos: `21,37_508,2876m/z` não é o nome de nada.

O teto de features do mapa de calor é importado de `statistics.clustering`, onde foi medido.
Ele vale só para o eixo das features. O dendrograma é das amostras, e o que há sobre o eixo
dele está em `TETO_DE_FOLHAS_PARA_DESENHAR`.

O desenho do dendrograma e do mapa de calor não tem passo estocástico: a ligação, a ordem das
folhas e o z-score são determinísticos. Amostragem, quando há, é de quem chama, que a declara
no `Recorte`. Nenhum artista leva `GID_DAS_FEATURES`, porque o dendrograma tem um ponto por
amostra e o mapa tem células; a identificação usa o rótulo da série.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import dendrogram, is_valid_linkage

from refino.plots.base import (
    PALETA_DIVERGENTE,
    MatrizVaziaError,
    RecusaDoConstrutorError,
    denominador_ou_recusa,
    exige_abaixo_do_teto,
    inteiro_do_fio,
    nova_figura,
    numero_do_fio,
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
from refino.plots.series import series_por_grupo
from refino.statistics.clustering import (
    INICIALIZACOES_DO_KMEANS,
    TETO_DE_FEATURES_PARA_DESENHAR,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from refino.plots.series import EstilosDosGrupos

__all__ = [
    "LARGURA_DA_BARRA_DO_KMEANS",
    "NOME_DA_LIGACAO",
    "NOME_DA_METRICA",
    "ROTULO_DA_ARVORE",
    "ROTULO_DA_ESCALA",
    "ROTULO_DO_MAPA",
    "SEPARADOR_DO_NOME",
    "TETO_DE_FOLHAS_PARA_DESENHAR",
    "TITULO_DO_DENDROGRAMA",
    "TITULO_DO_KMEANS",
    "TITULO_DO_MAPA",
    "TITULO_DO_SOM",
    "Arvore",
    "ArvoreInvalidaError",
    "Grade",
    "MapaDeCalor",
    "MapaDesalinhadoError",
    "MetodoDesconhecidoError",
    "Particao",
    "ParticaoDesalinhadaError",
    "arvore_do_fio",
    "figura_do_dendrograma",
    "figura_do_kmeans",
    "figura_do_mapa_auto_organizavel",
    "figura_do_mapa_de_calor",
    "frase_da_grade",
    "frase_da_semente",
    "frase_do_metodo",
    "frase_do_prefixo",
    "kmeans_do_fio",
    "mapa_do_fio",
    "posicoes_na_grade",
    "recorte_do_dendrograma",
    "recorte_do_kmeans",
    "recorte_do_mapa_auto_organizavel",
    "recorte_do_mapa_de_calor",
    "segmentos_do_dendrograma",
    "sem_prefixo_comum",
    "som_do_fio",
]

TITULO_DO_DENDROGRAMA: Final = "Agrupamento hierárquico das amostras"
TITULO_DO_MAPA: Final = "Mapa de calor das features agrupadas"

ROTULO_DA_ARVORE: Final = "árvore de ligação"
"""Rótulo do feixe de segmentos. É distinto do de toda série de grupo porque os testes indexam
as séries por rótulo, e dois artistas com o mesmo texto colapsariam num só."""

ROTULO_DO_MAPA: Final = "z-score por feature"
"""Rótulo do `AxesImage` e da barra de cor."""

ROTULO_DA_ESCALA: Final = ROTULO_DO_MAPA

TETO_DE_FOLHAS_PARA_DESENHAR: Final[int | None] = None
"""Teto para o eixo das folhas do dendrograma. Não há número.

O teto de features não se aplica aqui: foi medido com rótulos de feature, e este eixo tem nomes
de amostra. Na medição com o construtor real, o redesenho passa de 100 ms já com dez folhas, e
também sem rótulo nenhum, porque o custo dominante é o texto fixo (título, rodapé e legenda).
Um teto sobre o número de folhas não governaria essa quantidade. O custo marginal é de cerca
de 2,6 ms por rótulo de amostra, mais que o de um rótulo de feature porque estes são rodados
a 90°.

Um número só faria sentido com uma coorte maior que as 68 amostras do maior modo deste estudo.
"""

# Nome legível de cada valor do vocabulário fechado do domínio. Um teste compara estas chaves
# com `get_args(ClusterMetric)` e `get_args(ClusterLinkage)`, para não haver método sem nome
# nem nome sem método.
NOME_DA_METRICA: Final[dict[str, str]] = {
    "euclidean": "distância euclidiana",
    "correlation": "distância de correlação: 1 menos o r de Pearson",
    "spearman": "distância de Spearman: 1 menos o rho sobre os postos",
}

NOME_DA_LIGACAO: Final[dict[str, str]] = {
    "ward": "ligação de Ward",
    "average": "ligação por média (UPGMA)",
    "complete": "ligação completa",
    "single": "ligação simples",
}


class ArvoreInvalidaError(RecusaDoConstrutorError):
    """A matriz de ligação não é uma ligação.

    Herda de `RecusaDoConstrutorError` porque é a única família que o painel apanha; outra
    exceção levantada dentro de um slot do Qt vai para o `stderr` e o painel fica com a figura
    anterior.

    A validação é a de `scipy.cluster.hierarchy.is_valid_linkage`, porque quem consome a matriz
    é o `scipy`. Uma matriz inválida que passasse levantaria de dentro do `dendrogram`, a meio
    do desenho e fora da família que o painel apanha.
    """


class MapaDesalinhadoError(RecusaDoConstrutorError):
    """A matriz do mapa não tem a forma que os nomes declaram.

    Um vetor de nomes com comprimento diferente do eixo que ele nomeia produz uma figura bem
    formada com os rótulos deslocados.
    """


class MetodoDesconhecidoError(RecusaDoConstrutorError):
    """A distância ou a ligação não estão no vocabulário fechado.

    Os dois nomes atravessam o fio. Um nome desconhecido escrito cru no rodapé, ou omitido,
    poria a figura a declarar um método que ninguém correu.
    """


def frase_do_metodo(metrica: str, ligacao: str) -> str:
    """`("euclidean", "ward")` vira `"distância euclidiana · ligação de Ward"`, ou recusa.

    O separador é o mesmo dos campos do rodapé, que é onde esta frase vai.
    """
    faltam = [
        f"{rotulo} {valor!r}"
        for rotulo, valor, registo in (
            ("a distância", metrica, NOME_DA_METRICA),
            ("a ligação", ligacao, NOME_DA_LIGACAO),
        )
        if valor not in registo
    ]
    if faltam:
        raise MetodoDesconhecidoError(
            f"{' e '.join(faltam)} não está(ão) no vocabulário fechado deste módulo — "
            f"distâncias: {sorted(NOME_DA_METRICA)}; ligações: {sorted(NOME_DA_LIGACAO)}. "
            "A figura declara o método na própria imagem (R5b), e declarar um nome que ninguém "
            "correu é inventar procedência"
        )
    return f"{NOME_DA_METRICA[metrica]} · {NOME_DA_LIGACAO[ligacao]}"


@dataclass(frozen=True)
class Arvore:
    """Uma ligação hierárquica desenhável: a matriz, os rótulos das folhas e o que a produziu.

    Não é `ClusteringResult`, que é o resultado do núcleo estatístico, com a matriz escalonada
    inteira. Esta é a carga que atravessa o fio para desenhar um eixo. A janela recebe JSON, e
    um contrato moldado pelo objeto do núcleo obrigaria a janela e o relatório a caminhos
    diferentes.
    """

    ligacao: np.ndarray
    """`(n_folhas - 1, 4)`, no formato do `scipy`. Imutável."""

    rotulos: tuple[str, ...]
    """Um nome por folha, na ordem original das observações, não na ordem das folhas."""

    metrica: str
    ligacao_metodo: str

    @property
    def n_folhas(self) -> int:
        return int(self.ligacao.shape[0]) + 1


@dataclass(frozen=True)
class MapaDeCalor:
    """A matriz desenhável, os nomes dos dois eixos e a ordem em que cada eixo é lido.

    Os valores ficam na orientação interna do projeto, amostras em linhas e features em
    colunas, que é a de `ClusteringResult.scaled`. O construtor transpõe uma vez, para desenhar.

    As duas ordens fazem parte do objeto: sem a ordem das folhas, o mapa seria uma matriz
    qualquer. Ordem vazia significa a ordem das linhas, que é caso legítimo.
    """

    valores: np.ndarray
    """`(n_amostras, n_features)`. Imutável."""

    amostras: tuple[str, ...]
    features: tuple[str, ...]

    grupos: tuple[str, ...]
    """O grupo clínico por amostra, ou `()` quando o estudo não o declara."""

    ordem_das_amostras: tuple[int, ...]
    ordem_das_features: tuple[int, ...]


def _matriz_ou_recusa(valores: Any, *, de: str) -> np.ndarray:
    """Converte o que vem do fio ou do núcleo num `ndarray` 2-D de `float`, ou recusa."""
    try:
        array = np.asarray(valores, dtype=float)
    except (TypeError, ValueError) as erro:
        raise MapaDesalinhadoError(
            f"{de} não converte para matriz de números: {erro}. Nada que venha de JSON passou pelo "
            "mypy, e uma lista de listas irregular chega aqui como objeto"
        ) from erro
    # O teste de vazio vem antes do de forma: `[]` tem uma dimensão e tamanho zero, e o fato é
    # 'não veio população' (`MatrizVaziaError`), não 'tem 1 dimensão'.
    if array.size == 0:
        raise MatrizVaziaError(
            f"{de} chegou vazia: não há célula para desenhar. Figura em branco seria "
            "indistinguível de figura quebrada"
        )
    if array.ndim != 2:
        raise MapaDesalinhadoError(
            f"{de} tem {array.ndim} dimensão(ões) e o mapa de calor precisa de duas. Um vetor "
            "achatado desenharia uma faixa de um pixel de altura, que é figura bem formada sobre "
            "a matriz errada"
        )
    return array


def _permutacao_ou_recusa(ordem: Any, n: int, *, de: str) -> tuple[int, ...]:
    """Uma permutação de `range(n)`, ou a identidade quando `ordem` é vazia; senão recusa.

    Exige permutação porque índices repetidos desenhariam a mesma amostra duas vezes e
    omitiriam outra, numa figura bem formada cujo rodapé declara uma contagem que a imagem não
    cumpre.
    """
    if ordem is None or (hasattr(ordem, "__len__") and len(ordem) == 0):
        return tuple(range(n))
    if not isinstance(ordem, (list, tuple, np.ndarray)):
        raise MapaDesalinhadoError(f"{de} não é uma sequência de índices: {ordem!r}")
    try:
        indices = tuple(int(i) for i in ordem)
    except (TypeError, ValueError) as erro:
        raise MapaDesalinhadoError(f"{de} tem índice que não é inteiro: {erro}") from erro
    if sorted(indices) != list(range(n)):
        raise MapaDesalinhadoError(
            f"{de} tem {len(indices)} índice(s) e não é uma permutação de {n}: um índice repetido "
            "desenha a mesma linha duas vezes e omite outra, com a figura bem formada e o rodapé "
            "a declarar uma contagem que a imagem não cumpre"
        )
    return indices


def _porque_nao_e_percorrivel(matriz: np.ndarray) -> str:
    """O que impede o `dendrogram` de percorrer esta ligação, ou `""`.

    Complementa `is_valid_linkage`, que devolve `True` para qualquer ligação de uma linha só:
    o `scipy` sai cedo com `if n < 2: return True`, sem conferir índices, alturas nem
    contagens. Uma ligação de uma junção com índice fora do intervalo passaria, e o
    `dendrogram` levantaria `IndexError`, fora da família que o painel apanha.

    As verificações são as do `scipy`, aplicadas a qualquer número de linhas. A altura é
    conferida por finitude e não só por `< 0`, porque `nan < 0` é `False` e o `nan` passaria.
    """
    n_juncoes = int(matriz.shape[0])
    n_folhas = n_juncoes + 1
    if not np.isfinite(matriz).all():
        return "há valor não finito na ligação, e a altura da junção é o eixo y inteiro da figura"
    filhos = matriz[:, :2]
    if (filhos < 0).any():
        return "há índice de filho negativo"
    # O nó formado pela junção `i` recebe o identificador `n_folhas + i`, então a junção `i` só
    # pode referir filhos com identificador menor que esse. É a conferência 'non-singleton
    # cluster before it is formed' do `scipy`, aqui sem o corte de `n < 2`.
    limite = np.arange(n_folhas, n_folhas + n_juncoes, dtype=float)
    if (filhos.max(axis=1) >= limite).any():
        return (
            f"há junção que refere um nó ainda não formado: com {n_folhas} folhas, a junção `i` "
            f"só pode usar identificadores abaixo de {n_folhas} + i"
        )
    if (matriz[:, 2] < 0).any():
        return "há altura de junção negativa"
    contagens = matriz[:, 3]
    if ((contagens < 2) | (contagens > n_folhas)).any():
        return f"há contagem de folhas fora de [2, {n_folhas}] numa junção"
    return ""


def arvore_do_fio(
    *,
    ligacao: Any,
    rotulos: Any,
    metrica: Any,
    ligacao_metodo: Any,
) -> Arvore:
    """Monta uma `Arvore` a partir do que o fio ou o núcleo trazem, com as guardas de fronteira.

    A ligação é validada antes do alinhamento dos rótulos. Com uma matriz que não é ligação, o
    número de folhas não faz sentido, e a mensagem sairia sobre os rótulos quando o defeito é a
    matriz.
    """
    try:
        matriz = np.asarray(ligacao, dtype=float)
    except (TypeError, ValueError) as erro:
        raise ArvoreInvalidaError(
            f"a ligação não converte para matriz de números: {erro}"
        ) from erro
    if matriz.ndim != 2 or matriz.shape[0] < 1 or matriz.shape[1] != 4:
        raise ArvoreInvalidaError(
            f"a ligação tem forma {matriz.shape} e o formato do `scipy` é `(n_folhas - 1, 4)`, com "
            "ao menos uma junção. Uma observação só não tem árvore: não há par, não há distância "
            "e não há o que desenhar"
        )
    if not is_valid_linkage(matriz) or (motivo := _porque_nao_e_percorrivel(matriz)):
        raise ArvoreInvalidaError(
            f"a matriz tem a forma de uma ligação e não é uma: {motivo or 'o `scipy` recusa-a'}. "
            "Índices fora do intervalo, alturas não finitas ou contagens incoerentes produzem, "
            "dentro do `dendrogram`, uma excepção crua no vocabulário do `scipy` — e fora da "
            "família que o painel apanha"
        )
    matriz = matriz.copy()
    matriz.flags.writeable = False

    if not isinstance(rotulos, (list, tuple)):
        raise ArvoreInvalidaError(f"os rótulos das folhas não são uma sequência: {rotulos!r}")
    nomes = tuple(str(r) for r in rotulos)
    esperadas = int(matriz.shape[0]) + 1
    if len(nomes) != esperadas:
        raise ArvoreInvalidaError(
            f"a ligação tem {esperadas} folhas e chegaram {len(nomes)} rótulo(s). Rótulo deslocado "
            "produz figura bem formada com o nome errado em cada folha, que é a única classe de "
            "defeito de figura que ninguém vê olhando para ela"
        )
    if any(not n.strip() for n in nomes):
        raise ArvoreInvalidaError(
            "há rótulo de folha em branco. Marca sem texto lê-se como folha sem identidade, e a "
            "identidade da folha é o que liga a árvore às amostras do estudo"
        )

    # O método é conferido aqui, e não só no rodapé: sem isso uma árvore com método
    # desconhecido seria desenhada, e a recusa só apareceria ao montar o `Recorte`.
    frase_do_metodo(str(metrica), str(ligacao_metodo))
    return Arvore(
        ligacao=matriz,
        rotulos=nomes,
        metrica=str(metrica),
        ligacao_metodo=str(ligacao_metodo),
    )


def mapa_do_fio(
    *,
    valores: Any,
    amostras: Any,
    features: Any,
    grupos: Any = (),
    ordem_das_amostras: Any = (),
    ordem_das_features: Any = (),
) -> MapaDeCalor:
    """Um `MapaDeCalor` a partir do que o fio ou o núcleo trazem, com as guardas de fronteira."""
    matriz = _matriz_ou_recusa(valores, de="a matriz do mapa de calor")
    nomes_de_amostra = (
        tuple(str(a) for a in amostras) if isinstance(amostras, (list, tuple)) else ()
    )
    nomes_de_feature = (
        tuple(str(f) for f in features) if isinstance(features, (list, tuple)) else ()
    )
    if (len(nomes_de_amostra), len(nomes_de_feature)) != matriz.shape:
        raise MapaDesalinhadoError(
            f"a matriz é {matriz.shape} — amostras em linhas, features em colunas — e chegaram "
            f"{len(nomes_de_amostra)} nome(s) de amostra e {len(nomes_de_feature)} de feature. "
            "Nomes deslocados produzem figura bem formada com os rótulos trocados"
        )
    nomes_de_grupo = tuple(str(g) for g in grupos) if isinstance(grupos, (list, tuple)) else ()
    if nomes_de_grupo and len(nomes_de_grupo) != len(nomes_de_amostra):
        raise MapaDesalinhadoError(
            f"chegaram {len(nomes_de_grupo)} grupo(s) para {len(nomes_de_amostra)} amostra(s). O "
            "grupo é por amostra, e um vetor mais curto pintaria os rótulos por posição — cada "
            "amostra com a cor da vizinha"
        )
    matriz = matriz.copy()
    matriz.flags.writeable = False
    return MapaDeCalor(
        valores=matriz,
        amostras=nomes_de_amostra,
        features=nomes_de_feature,
        grupos=nomes_de_grupo,
        ordem_das_amostras=_permutacao_ou_recusa(
            ordem_das_amostras, matriz.shape[0], de="a ordem das amostras"
        ),
        ordem_das_features=_permutacao_ou_recusa(
            ordem_das_features, matriz.shape[1], de="a ordem das features"
        ),
    )


# ---------------------------------------------------------------------------
# 1. O dendrograma das amostras
# ---------------------------------------------------------------------------

SEPARADOR_DO_NOME: Final = "_"
"""Onde o prefixo comum de nomes de amostra pode ser cortado. Ver `sem_prefixo_comum`."""


def sem_prefixo_comum(nomes: Sequence[str]) -> tuple[tuple[str, ...], str]:
    """Os nomes sem o prefixo que todos partilham, e o prefixo.

    Os nomes de amostra do laboratório são longos, como
    `HTLV_HILIC_PLASMA_MSe_POS_0_ID99_Asymptomatic`, e começam todos pelo mesmo trecho (o modo,
    a matriz biológica e a polaridade), que o rodapé já declara. Rodados 90° num eixo, os nomes
    inteiros consomem a altura da figura: o `tight_layout` desiste e a área de dado some.

    Nada é reescrito. Um prefixo constante é fatorado e o rodapé o publica por inteiro, então o
    nome completo se reconstrói da imagem.

    O corte é no separador, não no caractere. Sem isso, `..._POS_1_ID33` e `..._POS_11_ID7`
    partilhariam `_POS_1`, e o campo seria cortado em lugares diferentes nos dois nomes.

    Devolve os nomes intactos quando não há o que fatorar: menos de dois nomes, prefixo comum
    vazio ou sem separador, ou um prefixo que deixaria algum nome vazio.
    """
    if len(nomes) < 2:
        return tuple(nomes), ""
    prefixo = nomes[0]
    for nome in nomes[1:]:
        while prefixo and not nome.startswith(prefixo):
            prefixo = prefixo[:-1]
        if not prefixo:
            return tuple(nomes), ""
    corte = prefixo.rfind(SEPARADOR_DO_NOME)
    if corte < 0:
        return tuple(nomes), ""
    prefixo = prefixo[: corte + 1]
    curtos = tuple(nome[len(prefixo) :] for nome in nomes)
    if any(not c.strip() for c in curtos):
        return tuple(nomes), ""
    return curtos, prefixo


def frase_do_prefixo(prefixo: str) -> str:
    """O que o rodapé diz sobre o prefixo fatorado, com o prefixo por inteiro."""
    if not prefixo:
        return "os rótulos das amostras são os nomes completos"
    return (
        f"os rótulos das amostras saem sem o prefixo comum a todas elas, «{prefixo}» — fatorado "
        "e não truncado: o nome completo é o prefixo mais o rótulo"
    )


_PASSO_DA_FOLHA: Final = 10.0
"""A distância entre folhas na convenção do `scipy`: a folha `i` fica em `10·i + 5`.

Um teste afirma que as posições derivadas daqui são as que os segmentos do `icoord` tocam, e
reprova se a convenção do `scipy` mudar.
"""


def segmentos_do_dendrograma(arvore: Arvore) -> tuple[np.ndarray, tuple[int, ...]]:
    """Os segmentos da árvore, em `(4, 2)` cada, e a ordem das folhas da esquerda para a direita.

    A chamada usa `no_plot=True` e só pede a geometria. Desenhado pelo `scipy`, o dendrograma
    seria um `Line2D` por junção, com transformação própria, e os testes de figura leriam uma
    série por junção e acusariam pontos fora dos limites. Quem chama desenha os segmentos como
    uma `LineCollection`, que é lida como uma série só.
    """
    desenho = dendrogram(arvore.ligacao, no_plot=True)
    segmentos = np.array(
        [
            np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
            for x, y in zip(desenho["icoord"], desenho["dcoord"], strict=True)
        ],
        dtype=float,
    )
    # O `scipy` tipa `leaves` como `list[int] | None`; `None` só ocorre com `truncate_mode`,
    # que esta função não usa. A recusa cobre o caso em vez de um `cast`.
    folhas = desenho["leaves"]
    if folhas is None:  # pragma: no cover - `no_plot=True` sem truncagem devolve sempre a lista
        raise ArvoreInvalidaError(
            "o `scipy` não devolveu a ordem das folhas para esta ligação: sem ela não há como pôr "
            "cada amostra na posição dela, e a figura sairia com os rótulos deslocados"
        )
    return segmentos, tuple(int(i) for i in folhas)


def recorte_do_dendrograma(
    *,
    arvore: Arvore,
    n_features_usadas: int,
    n_features_totais: Any,
    matriz: str,
    amostragem: str,
) -> Recorte:
    """O rodapé do dendrograma: de que população as distâncias saíram e por que distância.

    O `Recorte` é argumento do construtor porque a matriz de ligação não diz quantas features
    entraram na distância nem quantas a matriz tinha.

    A distância entra no `metodo`: o mesmo dendrograma com outra distância é outra árvore, e
    sem o rodapé as duas figuras seriam indistinguíveis.
    """
    return Recorte(
        desenhadas=n_features_usadas,
        disponiveis=denominador_ou_recusa(n_features_totais, de="o agrupamento"),
        amostragem=amostragem,
        matriz=matriz,
        metodo=(
            f"{frase_do_metodo(arvore.metrica, arvore.ligacao_metodo)}, sobre a matriz "
            f"z-scoreada por feature. Sem teste estatístico: a árvore não tem p-valor, e a "
            f"altura de uma junção é uma distância e não uma evidência. "
            f"{milhar(arvore.n_folhas)} folhas; "
            f"{frase_do_prefixo(sem_prefixo_comum(arvore.rotulos)[1])}"
        ),
    )


def figura_do_dendrograma(
    *,
    arvore: Arvore,
    grupos: Sequence[str] = (),
    recorte: Recorte,
    estilos: EstilosDosGrupos | None = None,
) -> Figure:
    """A árvore das amostras, com as folhas coloridas pelo grupo clínico.

    Desenha a população inteira que recebe. A cor do grupo vem de `series.series_por_grupo`,
    para a mesma coorte ter a mesma paleta em qualquer aba.

    As folhas são uma dispersão por grupo, e não a cor do rótulo de marca. Assim há um artista
    de dado por série, que os testes leem, e a legenda traz a contagem por grupo.

    Sem grupos, uma dispersão só, com o rótulo a dizer quantas amostras.
    """
    if arvore.n_folhas < 2:  # pragma: no cover - `arvore_do_fio` já recusa uma junção a menos
        raise MatrizVaziaError(
            f"a árvore tem {arvore.n_folhas} folha(s): não há agrupamento a desenhar"
        )
    if grupos and len(grupos) != arvore.n_folhas:
        raise MapaDesalinhadoError(
            f"chegaram {len(grupos)} grupo(s) para {arvore.n_folhas} folha(s). O grupo é por "
            "amostra, e um vetor mais curto pintaria as folhas por posição"
        )

    segmentos, ordem = segmentos_do_dendrograma(arvore)
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)

    # A árvore é desenhada antes das folhas, para o feixe ficar por baixo da dispersão.
    feixe = LineCollection(
        [s for s in segmentos],
        linewidths=0.9,
        label=ROTULO_DA_ARVORE,
    )
    eixo.add_collection(feixe)

    posicoes = np.arange(arvore.n_folhas, dtype=float) * _PASSO_DA_FOLHA + _PASSO_DA_FOLHA / 2.0
    # `ordem[j]` é a observação que está na posição `j`. O inverso põe cada observação na sua
    # posição sem reordenar as máscaras de grupo, que vêm na ordem original das amostras.
    por_observacao = np.empty(arvore.n_folhas, dtype=float)
    por_observacao[np.asarray(ordem, dtype=int)] = posicoes
    zeros = np.zeros(arvore.n_folhas, dtype=float)

    maos: list[Artist] = [feixe]
    if grupos:
        # Com `estilos`, a cor e o símbolo de cada folha são os do grupo escolhido.
        for serie in series_por_grupo(list(grupos), estilos=estilos):
            maos.append(
                eixo.scatter(
                    por_observacao[serie.mascara],
                    zeros[serie.mascara],
                    s=18,
                    color=serie.cor,
                    marker=serie.marcador,
                    label=serie.rotulo,
                    zorder=3,
                )
            )
    else:
        maos.append(
            eixo.scatter(
                por_observacao,
                zeros,
                s=18,
                label=f"{milhar(arvore.n_folhas)} amostras",
                zorder=3,
            )
        )

    curtos, _prefixo = sem_prefixo_comum(arvore.rotulos)
    eixo.set_xticks(posicoes.tolist())
    eixo.set_xticklabels([curtos[i] for i in ordem], rotation=90, fontsize=TAMANHO_DA_FONTE)
    eixo.set_xlabel("amostra")
    eixo.set_ylabel(f"altura da junção — {NOME_DA_METRICA[arvore.metrica]}")
    eixo.set_title(TITULO_DO_DENDROGRAMA, fontsize=TAMANHO_DA_FONTE)
    # `add_collection` não ajusta os limites, ao contrário de `plot`; sem isto a árvore fica
    # fora da vista e a figura sai em branco.
    eixo.autoscale_view()
    eixo.set_xlim(0.0, float(arvore.n_folhas) * _PASSO_DA_FOLHA)
    # As folhas estão em `y = 0`. Com o limite inferior em zero, metade de cada marcador
    # ficaria abaixo da linha do eixo, por isso há uma folga embaixo.
    altura = float(segmentos[:, :, 1].max())
    eixo.set_ylim(-altura * 0.03, altura * 1.05)
    legenda_fora(eixo, maos)
    aplica(figura, recorte)
    return figura


# ---------------------------------------------------------------------------
# 2. O mapa de calor
# ---------------------------------------------------------------------------


def recorte_do_mapa_de_calor(
    *,
    mapa: MapaDeCalor,
    metrica: str,
    ligacao_metodo: str,
    n_features_totais: Any,
    matriz: str,
    amostragem: str,
) -> Recorte:
    """O rodapé do mapa: quantas features desenhou, de quantas, e quantas células são indefinidas.

    A contagem de não finitas é publicada porque uma célula `nan` sai transparente, e sobre um
    fundo claro ela se confunde com um z-score perto de zero na paleta divergente. Sem a
    contagem, 'não foi medido' e 'está na média' sairiam iguais.
    """
    nao_finitas = int(np.count_nonzero(~np.isfinite(mapa.valores)))
    indefinidas = frase_contada(
        nao_finitas,
        nenhuma="nenhuma célula é indefinida",
        uma=f"1 de {milhar(mapa.valores.size)} células é indefinida e sai sem cor",
        algumas=f"de {milhar(mapa.valores.size)} células são indefinidas e saem sem cor",
    )
    return Recorte(
        desenhadas=len(mapa.features),
        disponiveis=denominador_ou_recusa(n_features_totais, de="o agrupamento"),
        amostragem=amostragem,
        matriz=matriz,
        metodo=(
            f"{frase_do_metodo(metrica, ligacao_metodo)}; as duas ordens são as das folhas. A "
            f"escala é o z-score por feature, com limites simétricos em torno de zero. Sem teste "
            # `frase_contada` devolve a oração em minúscula, então ela vem depois de ponto e
            # vírgula e não de ponto final.
            f"estatístico; {indefinidas}; "
            f"{frase_do_prefixo(sem_prefixo_comum(mapa.amostras)[1])}"
        ),
    )


def _limite_simetrico(conteudo: np.ndarray) -> float:
    """O maior desvio absoluto finito, ou `1,0` quando não há nenhum. Nunca zero.

    `vmin == vmax` quebra a norma do `matplotlib`. O recuo de `1,0` cobre a matriz toda a
    zeros, que é toda neutra e pode ser desenhada.
    """
    finitos = conteudo[np.isfinite(conteudo)]
    if finitos.size == 0:
        return 1.0
    maior = float(np.abs(finitos).max())
    return maior if maior > 0.0 else 1.0


def figura_do_mapa_de_calor(
    *, mapa: MapaDeCalor, recorte: Recorte, estilos: EstilosDosGrupos | None = None
) -> Figure:
    """A matriz z-scoreada, features em linhas e amostras em colunas, nas ordens do agrupamento.

    Desenha a população inteira que recebe. Quantas features desenhar é decisão de quem chama;
    acima do teto a figura é recusada.

    As features vão em linhas porque `figura_do_vip` já põe nomes de feature no eixo y, e as
    duas abas são comparadas lado a lado. A matriz chega na orientação interna, amostras em
    linhas, e a transposição acontece aqui.

    O teto é o de `statistics.clustering`, e a recusa é de apresentação, não de cálculo: o que
    passa dos 100 ms de interação é o desenho dos rótulos. A mensagem traz a contagem e o teto.

    A barra de cor recebe a vírgula decimal e os rótulos de feature não.
    """
    exige_abaixo_do_teto(len(mapa.features), TETO_DE_FEATURES_PARA_DESENHAR, de="o mapa de calor")

    ordenada = mapa.valores[
        np.ix_(
            np.asarray(mapa.ordem_das_amostras, dtype=int),
            np.asarray(mapa.ordem_das_features, dtype=int),
        )
    ]
    conteudo = np.ascontiguousarray(ordenada.T)
    limite = _limite_simetrico(conteudo)

    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    imagem = eixo.imshow(
        conteudo,
        aspect="auto",
        interpolation="nearest",
        cmap=PALETA_DIVERGENTE,
        vmin=-limite,
        vmax=limite,
        label=ROTULO_DO_MAPA,
    )
    barra = figura.colorbar(imagem, ax=eixo)
    barra.set_label(ROTULO_DA_ESCALA, fontsize=TAMANHO_DA_FONTE)
    # As marcas e o rótulo da barra vão à esquerda dela. `legenda_fora` ancora a legenda em
    # `(1.01, 1.0)` da caixa do eixo que recebe, e com as marcas à direita a legenda cairia por
    # cima dos números da escala.
    barra.ax.yaxis.set_ticks_position("left")
    barra.ax.yaxis.set_label_position("left")

    eixo.set_yticks(np.arange(len(mapa.features), dtype=float).tolist())
    eixo.set_yticklabels(
        [mapa.features[i] for i in mapa.ordem_das_features], fontsize=TAMANHO_DA_FONTE
    )
    curtos, _prefixo = sem_prefixo_comum(mapa.amostras)
    eixo.set_xticks(np.arange(len(mapa.amostras), dtype=float).tolist())
    eixo.set_xticklabels(
        [curtos[i] for i in mapa.ordem_das_amostras],
        rotation=90,
        fontsize=TAMANHO_DA_FONTE,
    )
    eixo.set_xlabel("amostra")
    eixo.set_ylabel("feature")
    eixo.set_title(TITULO_DO_MAPA, fontsize=TAMANHO_DA_FONTE)

    maos = _maos_dos_grupos(eixo, mapa, estilos)
    if maos:
        # A legenda é ancorada na barra de cor, e não no eixo do mapa: à direita do eixo já
        # está a barra, e a legenda a cobriria. Com `barra.ax` a âncora fica depois da barra,
        # e a legenda continua filha de um eixo da figura, dentro do `tightbbox` de `aplica`.
        legenda_fora(barra.ax, maos)
    aplica(figura, recorte)
    return figura


def _maos_dos_grupos(
    eixo: Axes, mapa: MapaDeCalor, estilos: EstilosDosGrupos | None = None
) -> list[Artist]:
    """Pinta o rótulo de cada amostra com a cor do grupo dela e devolve as mãos da legenda.

    Aqui a cor vai no rótulo de marca, ao contrário do dendrograma, porque o mapa de calor não
    tem onde pôr um ponto; uma faixa de cor por amostra seria uma segunda imagem na figura.

    Com `estilos`, a cor é a do grupo escolhido. O símbolo não entra: o mapa não desenha ponto
    nenhum, e a mão da legenda continua a ser um quadrado.

    As mãos da legenda são `Line2D` avulsos, que não entram em `ax.lines`, e o eixo continua a
    ter só a imagem. A legenda dá o nome e a contagem de cada grupo.
    """
    if not mapa.grupos:
        return []
    maos: list[Artist] = []
    por_amostra: dict[int, Any] = {}
    for serie in series_por_grupo(list(mapa.grupos), estilos=estilos):
        maos.append(
            Line2D([], [], marker="s", linestyle="none", color=serie.cor, label=serie.rotulo)
        )
        for indice in np.flatnonzero(serie.mascara).tolist():
            por_amostra[int(indice)] = serie.cor
    for posicao, indice in enumerate(mapa.ordem_das_amostras):
        eixo.get_xticklabels()[posicao].set_color(por_amostra[indice])
    return maos


# ---------------------------------------------------------------------------
# 3. O K-means e o mapa auto-organizável
# ---------------------------------------------------------------------------
#
# As duas figuras mostram em que parte cada amostra caiu: `k` agrupamentos sem relação entre si
# no K-means, uma grade com vizinhança no SOM. A pergunta é se a partição que o método achou
# sem ver os rótulos coincide com os grupos clínicos. Por isso as duas põem a cor do grupo
# sobre a geometria da partição, em vez de pintar cada agrupamento de uma cor: o agrupamento 0
# de uma execução não é o agrupamento 0 da seguinte.
#
# Nenhuma das duas usa `Axes.bar`, que cria `Rectangle` em `ax.patches`, fora do que os testes
# de figura leem. `vlines` devolve uma `LineCollection`, legível pelos segmentos.

TITULO_DO_KMEANS: Final = "Agrupamento particional das amostras (K-means)"
TITULO_DO_SOM: Final = "Mapa auto-organizável das amostras"

LARGURA_DA_BARRA_DO_KMEANS: Final = 14.0
"""A espessura do segmento que faz a barra empilhada parecer uma barra.

Está em pontos, que é a unidade de `linewidth`, e não em unidades de dado. Uma largura em
unidades de dado encolheria com `k`. Com `k` grande as barras se encostam, e o eixo continua a
dizer quantos agrupamentos há.
"""

_FOLGA_DA_CELULA: Final = 0.78
"""Que fração do lado de uma célula da grade os pontos de uma unidade podem ocupar.

Fica abaixo de 1 para os pontos de unidades vizinhas não se tocarem, o que sugeriria uma
vizinhança que a grade não tem.
"""

_MARGEM_DA_GRADE: Final = 0.12
"""Quanto a vista se estende para lá da última célula, em unidades de grade.

As unidades da borda ficam a meia célula do limite, e o desvio dentro da célula leva um ponto
até `0,39` dela. Com o limite exatamente em `±0,5`, metade do marcador de um ponto de borda
ficaria fora da caixa e seria recortada.
"""


def frase_da_semente(semente: Any) -> str:
    """A semente como ela se digita, sem separador de milhar.

    A semente é um identificador, não uma quantidade. Quem quiser reproduzir a figura cola o
    número, e `1.001.697.648` teria de perder os pontos antes.
    """
    return str(inteiro_do_fio(semente))


class ParticaoDesalinhadaError(RecusaDoConstrutorError):
    """Um vetor por amostra não indexa o conjunto que o mesmo objeto declara.

    Serve ao K-means e ao SOM, porque o fato é o mesmo nos dois: um inteiro por amostra que
    aponta para um agrupamento ou unidade fora do conjunto declarado, ou um vetor de grupos de
    outro comprimento.

    Um rótulo `7` numa partição de três poria a amostra numa barra que não existe, e um rótulo
    trocado a poria na barra errada, numa figura bem formada.
    """


# ---------------------------------------------------------------------------
# 3.1 O K-means
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Particao:
    """A partição desenhável: em que agrupamento cada amostra caiu e de que grupo clínico ela é.

    Não é `KMeansResult`, que é o resultado do núcleo, com as contagens de descarte e o digest
    das colunas. Esta é a carga que atravessa o fio para desenhar uma figura.

    `inercia` vem junto porque o rodapé a publica. Ela não ordena `k`: cai monotonicamente com
    `k`, então escolher por ela seria escolher `k = n`.
    """

    rotulos: tuple[int, ...]
    """O agrupamento de cada amostra, em `[0, k)`, na ordem das linhas da matriz."""

    grupos: tuple[str, ...]
    """O grupo clínico por amostra, ou `()` quando o estudo não o declara."""

    k: int
    inercia: float

    @property
    def n_amostras(self) -> int:
        return len(self.rotulos)

    @property
    def tamanhos(self) -> tuple[int, ...]:
        """Quantas amostras em cada agrupamento, contadas de `rotulos`."""
        contagem = [0] * self.k
        for rotulo in self.rotulos:
            contagem[rotulo] += 1
        return tuple(contagem)


def _inteiros_por_amostra(valores: Any, *, limite: int, de: str) -> tuple[int, ...]:
    """Um inteiro por amostra, todos dentro de `[0, limite)`, ou recusa.

    O intervalo é conferido aqui, onde o vetor e o limite estão juntos. No construtor, um
    índice fora do intervalo seria um `IndexError` a meio do desenho, fora da família que o
    painel apanha.
    """
    if not isinstance(valores, (list, tuple, np.ndarray)):
        raise ParticaoDesalinhadaError(f"{de} não é uma sequência de inteiros: {valores!r}")
    try:
        indices = tuple(int(v) for v in valores)
    except (TypeError, ValueError) as erro:
        raise ParticaoDesalinhadaError(f"{de} tem valor que não é inteiro: {erro}") from erro
    if not indices:
        raise MatrizVaziaError(
            f"{de} chegou vazio: não há amostra para desenhar. Figura em branco seria "
            "indistinguível de figura quebrada"
        )
    fora = sorted({i for i in indices if not 0 <= i < limite})
    if fora:
        raise ParticaoDesalinhadaError(
            f"{de} traz o(s) índice(s) {fora} e o conjunto declarado tem {limite} posições. Um "
            "índice fora do intervalo põe a amostra numa parte que não existe, e um índice trocado "
            "põe-na na parte errada — com a figura perfeitamente bem formada"
        )
    return indices


def _grupos_alinhados(grupos: Any, quantas: int, *, de: str) -> tuple[str, ...]:
    """O vetor de grupos por amostra, ou `()`. Recusa comprimento diferente do das amostras."""
    nomes = tuple(str(g) for g in grupos) if isinstance(grupos, (list, tuple)) else ()
    if nomes and len(nomes) != quantas:
        raise ParticaoDesalinhadaError(
            f"chegaram {len(nomes)} grupo(s) para {quantas} amostra(s) em {de}. O grupo é por "
            "amostra, e um vetor mais curto pintaria as amostras por posição — cada uma com a cor "
            "da vizinha"
        )
    return nomes


def kmeans_do_fio(*, rotulos: Any, grupos: Any = (), k: Any, inercia: Any = 0.0) -> Particao:
    """Uma `Particao` a partir do que o fio ou o núcleo trazem, com as guardas de fronteira."""
    quantos = inteiro_do_fio(k)
    if quantos < 1:
        raise MatrizVaziaError(
            f"o K-means declara {k!r} agrupamentos: sem ao menos um não há partição e não há o que "
            "desenhar"
        )
    indices = _inteiros_por_amostra(rotulos, limite=quantos, de="o agrupamento de cada amostra")
    return Particao(
        rotulos=indices,
        grupos=_grupos_alinhados(grupos, len(indices), de="o K-means"),
        k=quantos,
        inercia=numero_do_fio(inercia),
    )


def recorte_do_kmeans(
    *,
    particao: Particao,
    n_features_usadas: int,
    n_features_totais: Any,
    matriz: str,
    amostragem: str,
    semente: Any,
) -> Recorte:
    """O rodapé do K-means: quantas features entraram na distância, com que `k` e com que semente.

    A semente entra no rodapé porque a partição sai de um processo estocástico com vários
    arranques, e sem ela ninguém a reproduz.

    A inércia é declarada como ajuste, não como qualidade. Ela cai com `k` por construção, e
    lê-la como 'este `k` é melhor' levaria a escolher sempre o maior.
    """
    vazios = frase_contada(
        sum(1 for tamanho in particao.tamanhos if tamanho == 0),
        nenhuma="nenhum agrupamento ficou vazio",
        uma="1 parte da partição não recebeu amostra nenhuma",
        algumas="partes da partição não receberam amostra nenhuma",
    )
    arranques = milhar(INICIALIZACOES_DO_KMEANS)
    return Recorte(
        desenhadas=n_features_usadas,
        disponiveis=denominador_ou_recusa(n_features_totais, de="o K-means"),
        amostragem=amostragem,
        matriz=matriz,
        metodo=(
            f"K-means em {milhar(particao.k)} agrupamentos sobre a matriz z-scoreada por feature, "
            f"com {arranques} arranques e semente {frase_da_semente(semente)}. "
            f"Sem teste estatístico: a partição não tem p-valor. A inércia de "
            f"{decimal(particao.inercia)} descreve o AJUSTE e não a qualidade — ela cai com o "
            f"número de agrupamentos por construção. "
            f"{milhar(particao.n_amostras)} amostras; {vazios}"
        ),
    )


def figura_do_kmeans(
    *, particao: Particao, recorte: Recorte, estilos: EstilosDosGrupos | None = None
) -> Figure:
    """Uma barra empilhada por agrupamento, segmentada pelo grupo clínico de cada amostra.

    Desenha a população inteira que recebe.

    Um agrupamento quase todo de um grupo clínico diz que a partição não supervisionada
    reencontrou o fator do estudo; um agrupamento repartido entre os grupos diz que não. As
    duas leituras dependem da cor dentro da barra, que vem de `series.series_por_grupo`.

    Sem grupos declarados, uma série só com a contagem. Uma série por agrupamento pintaria um
    rótulo arbitrário.
    """
    posicoes = np.arange(particao.k, dtype=float)
    tamanhos = particao.tamanhos
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)

    rotulos = np.asarray(particao.rotulos, dtype=int)
    base = np.zeros(particao.k, dtype=float)
    maos: list[Artist] = []
    # Com `estilos`, o segmento de cada grupo leva a cor escolhida. Só a cor: segmento de barra
    # não tem marcador. O QC, que o K-means desenha e a coorte clínica não tem, recebe um par
    # que a coorte não usa (`EstilosDosGrupos.para`).
    series = series_por_grupo(list(particao.grupos), estilos=estilos) if particao.grupos else ()
    if series:
        for serie in series:
            contagem = np.array(
                [int(np.count_nonzero(rotulos[serie.mascara] == c)) for c in range(particao.k)],
                dtype=float,
            )
            # Só os agrupamentos em que este grupo tem amostras. Um segmento de comprimento
            # zero teria dois vértices coincidentes, que contam como pontos e ninguém vê.
            presentes = contagem > 0.0
            if not presentes.any():
                continue
            maos.append(
                eixo.vlines(
                    posicoes[presentes],
                    base[presentes],
                    (base + contagem)[presentes],
                    colors=serie.cor,
                    linewidth=LARGURA_DA_BARRA_DO_KMEANS,
                    label=serie.rotulo,
                )
            )
            base = base + contagem
    else:
        alturas = np.asarray(tamanhos, dtype=float)
        presentes = alturas > 0.0
        maos.append(
            eixo.vlines(
                posicoes[presentes],
                0.0,
                alturas[presentes],
                linewidth=LARGURA_DA_BARRA_DO_KMEANS,
                label=f"{milhar(particao.n_amostras)} amostras",
            )
        )

    eixo.set_xticks(posicoes.tolist())
    eixo.set_xticklabels([str(c + 1) for c in range(particao.k)], fontsize=TAMANHO_DA_FONTE)
    # O eixo y conta amostras, então as marcas são inteiras. Com contagens pequenas o
    # `matplotlib` escolheria passos como `0,25`, e '0,25 amostras' não existe.
    eixo.yaxis.set_major_locator(MaxNLocator(integer=True))
    eixo.set_xlabel("agrupamento")
    eixo.set_ylabel("amostras")
    eixo.set_title(TITULO_DO_KMEANS, fontsize=TAMANHO_DA_FONTE)
    # `vlines` não ajusta os limites, como `add_collection`; sem isto a figura sai em branco.
    eixo.set_xlim(-0.6, float(particao.k) - 0.4)
    eixo.set_ylim(0.0, float(max(tamanhos)) * 1.08)
    eixo.grid(visible=True, axis="y", linewidth=0.4, alpha=0.4)
    eixo.set_axisbelow(True)
    legenda_fora(eixo, maos)
    aplica(figura, recorte)
    return figura


# ---------------------------------------------------------------------------
# 3.2 O mapa auto-organizável
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grade:
    """Em que unidade da grade cada amostra caiu, e a forma da grade.

    A unidade é o índice plano, como em `SomResult`, e não um par de coordenadas. Com o par,
    trocar linha por coluna numa grade não quadrada daria uma figura bem formada com as
    amostras no lugar errado.
    """

    unidades: tuple[int, ...]
    grupos: tuple[str, ...]
    linhas: int
    colunas: int
    epocas: int
    erro_de_quantizacao: float

    @property
    def n_unidades(self) -> int:
        return self.linhas * self.colunas

    @property
    def n_amostras(self) -> int:
        return len(self.unidades)

    @property
    def tamanhos(self) -> tuple[int, ...]:
        """Quantas amostras em cada unidade, na ordem do índice plano."""
        contagem = [0] * self.n_unidades
        for unidade in self.unidades:
            contagem[unidade] += 1
        return tuple(contagem)

    @property
    def n_vazias(self) -> int:
        return sum(1 for tamanho in self.tamanhos if tamanho == 0)


def som_do_fio(
    *,
    unidades: Any,
    grupos: Any = (),
    linhas: Any,
    colunas: Any,
    epocas: Any = 0,
    erro_de_quantizacao: Any = 0.0,
) -> Grade:
    """Uma `Grade` a partir do que o fio ou o núcleo trazem, com as guardas de fronteira."""
    n_linhas, n_colunas = inteiro_do_fio(linhas), inteiro_do_fio(colunas)
    if n_linhas < 1 or n_colunas < 1:
        raise MatrizVaziaError(
            f"a grade declara {linhas!r} x {colunas!r} unidades: sem ao menos uma linha e uma "
            "coluna não há grade e não há o que desenhar"
        )
    indices = _inteiros_por_amostra(
        unidades, limite=n_linhas * n_colunas, de="a unidade de cada amostra"
    )
    return Grade(
        unidades=indices,
        grupos=_grupos_alinhados(grupos, len(indices), de="o mapa auto-organizável"),
        linhas=n_linhas,
        colunas=n_colunas,
        epocas=inteiro_do_fio(epocas),
        erro_de_quantizacao=numero_do_fio(erro_de_quantizacao),
    )


def frase_da_grade(grade: Grade) -> str:
    """O que o rodapé diz sobre a grade e sobre as unidades que ficaram vazias.

    Unidade vazia é achado: numa grade de dezesseis unidades com 68 amostras, as vazias são a
    fronteira entre agrupamentos vizinhos. Sem a frase, a célula em branco pareceria defeito.
    """
    vazias = frase_contada(
        grade.n_vazias,
        nenhuma="nenhuma unidade ficou vazia",
        uma=(
            f"1 de {milhar(grade.n_unidades)} unidades não recebeu amostra nenhuma — é a "
            "fronteira entre unidades vizinhas, e não um buraco"
        ),
        algumas=(
            f"de {milhar(grade.n_unidades)} unidades não receberam amostra nenhuma — são a "
            "fronteira entre unidades vizinhas, e não um buraco"
        ),
    )
    return f"grade de {milhar(grade.linhas)} x {milhar(grade.colunas)} unidades; {vazias}"


def posicoes_na_grade(grade: Grade) -> np.ndarray:
    """`(n_amostras, 2)` com a coluna e a linha de cada amostra, já com o desvio dentro da célula.

    Várias amostras caem na mesma unidade, e no mesmo ponto só uma seria vista. O desvio é
    determinístico, e não aleatório, para a figura ser reprodutível a partir do corpo do fio
    sem semente. As `m` amostras de uma unidade ocupam uma malha `lado x lado`, com
    `lado = ceil(sqrt(m))`, na ordem em que aparecem no vetor, que é a das linhas da matriz.

    O eixo x é a coluna e o y é a linha. Trocá-los transporia a grade.
    """
    posicoes = np.zeros((grade.n_amostras, 2), dtype=float)
    vistos: dict[int, int] = {}
    tamanhos = grade.tamanhos
    for indice, unidade in enumerate(grade.unidades):
        ordem = vistos.get(unidade, 0)
        vistos[unidade] = ordem + 1
        lado = max(1, int(np.ceil(np.sqrt(tamanhos[unidade]))))
        desvio_x = ((ordem % lado) + 0.5) / lado - 0.5
        desvio_y = ((ordem // lado) + 0.5) / lado - 0.5
        posicoes[indice, 0] = float(unidade % grade.colunas) + _FOLGA_DA_CELULA * desvio_x
        posicoes[indice, 1] = float(unidade // grade.colunas) + _FOLGA_DA_CELULA * desvio_y
    return posicoes


def recorte_do_mapa_auto_organizavel(
    *,
    grade: Grade,
    n_features_usadas: int,
    n_features_totais: Any,
    matriz: str,
    amostragem: str,
    semente: Any,
) -> Recorte:
    """O rodapé do SOM: a grade, as épocas, a semente e o erro de quantização.

    As épocas entram porque um mapa com poucas épocas ainda não ordenou, e a figura dele se
    confunde com a de um mapa ordenado sobre dado sem estrutura.

    O erro de quantização é declarado como ajuste, não como qualidade, como a inércia do
    K-means: ele cai com o número de unidades por construção.
    """
    return Recorte(
        desenhadas=n_features_usadas,
        disponiveis=denominador_ou_recusa(n_features_totais, de="o mapa auto-organizável"),
        amostragem=amostragem,
        matriz=matriz,
        metodo=(
            f"mapa auto-organizável sobre a matriz z-scoreada por feature, com "
            f"{milhar(grade.epocas)} épocas e semente {frase_da_semente(semente)}. Sem teste "
            f"estatístico: a posição na grade não tem p-valor. O erro de quantização de "
            f"{decimal(grade.erro_de_quantizacao)} descreve o AJUSTE e não a qualidade — ele cai "
            f"com o número de unidades por construção. {milhar(grade.n_amostras)} amostras; "
            f"{frase_da_grade(grade)}"
        ),
    )


def figura_do_mapa_auto_organizavel(
    *, grade: Grade, recorte: Recorte, estilos: EstilosDosGrupos | None = None
) -> Figure:
    """A grade do mapa, com uma dispersão por grupo clínico sobre as unidades que os receberam.

    Desenha a população inteira que recebe.

    É uma dispersão por grupo, e não uma imagem de contagens. Um `imshow` da contagem diria
    quantas amostras caíram em cada unidade e não quais: uma unidade com quatro `ATLL` e outra
    com uma amostra de cada grupo sairiam iguais. A dispersão dá um artista de dado por série
    clínica e a legenda com a contagem por grupo.

    O eixo y é invertido para a grade se ler como matriz, com a unidade `0` no canto superior
    esquerdo, de acordo com o índice plano `linha · colunas + coluna`.

    Sem grupos declarados, uma dispersão só com a contagem.
    """
    if grade.n_amostras == 0:  # pragma: no cover - `som_do_fio` já recusa o vetor vazio
        raise MatrizVaziaError("a grade não tem amostra nenhuma: não há o que desenhar")

    posicoes = posicoes_na_grade(grade)
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)

    maos: list[Artist] = []
    if grade.grupos:
        # Com `estilos`, a cor e o símbolo de cada dispersão são os do grupo escolhido.
        for serie in series_por_grupo(list(grade.grupos), estilos=estilos):
            maos.append(
                eixo.scatter(
                    posicoes[serie.mascara, 0],
                    posicoes[serie.mascara, 1],
                    s=22,
                    color=serie.cor,
                    marker=serie.marcador,
                    label=serie.rotulo,
                    zorder=3,
                )
            )
    else:
        maos.append(
            eixo.scatter(
                posicoes[:, 0],
                posicoes[:, 1],
                s=22,
                label=f"{milhar(grade.n_amostras)} amostras",
                zorder=3,
            )
        )

    eixo.set_xticks([float(c) for c in range(grade.colunas)])
    eixo.set_xticklabels([str(c + 1) for c in range(grade.colunas)], fontsize=TAMANHO_DA_FONTE)
    eixo.set_yticks([float(r) for r in range(grade.linhas)])
    eixo.set_yticklabels([str(r + 1) for r in range(grade.linhas)], fontsize=TAMANHO_DA_FONTE)
    # As linhas da grade são marcas menores do eixo, e não um `axhline` por fronteira, que
    # entraria em `ax.lines` e seria lido pelos testes como uma série de dado.
    eixo.set_xticks([c + 0.5 for c in range(grade.colunas - 1)], minor=True)
    eixo.set_yticks([r + 0.5 for r in range(grade.linhas - 1)], minor=True)
    eixo.grid(visible=True, which="minor", linewidth=0.6, alpha=0.5)
    eixo.set_axisbelow(True)

    eixo.set_xlabel("coluna da grade")
    eixo.set_ylabel("linha da grade")
    eixo.set_title(TITULO_DO_SOM, fontsize=TAMANHO_DA_FONTE)
    eixo.set_xlim(-0.5 - _MARGEM_DA_GRADE, float(grade.colunas) - 0.5 + _MARGEM_DA_GRADE)
    eixo.set_ylim(float(grade.linhas) - 0.5 + _MARGEM_DA_GRADE, -0.5 - _MARGEM_DA_GRADE)
    legenda_fora(eixo, maos)
    aplica(figura, recorte)
    return figura

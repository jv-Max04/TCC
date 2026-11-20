"""Mapa de correlação entre features e busca por padrão.

A escala do mapa é fixa em ±1. Uma correlação vive em `[-1, 1]` por definição, e um limite
derivado do dado pintaria um `0,4` com a cor do extremo numa figura e um `0,95` com a mesma
cor em outra; além disso a diagonal é 1, então o limite derivado seria sempre 1. O zero é o
tom neutro da paleta divergente e significa ausência de associação linear. A paleta é a de
`base.PALETA_DIVERGENTE`.

As barras da busca são `hlines` e não `bar`: `Axes.bar` cria `Rectangle` em `ax.patches`, que
os testes de figura não leem, e `hlines` devolve uma `LineCollection` legível pelos segmentos.
A busca desenha duas séries, separadas pelo sinal do coeficiente, porque acompanhar o perfil e
contrariá-lo são achados opostos. Entram as features de maior `|r|`, e a ordem é a do
coeficiente com sinal.

O teto de features do mapa é `TETO_DE_FEATURES_PARA_DESENHAR`, importado de
`statistics.clustering`. Ele foi medido com nomes de feature num eixo só e aqui os nomes vão
nos dois, então no teto o redesenho é mais lento que o do mapa de calor; a travessia dos 100 ms
fica entre 20 e 25 features. O teto não é apertado aqui porque isso restringiria o que a
ferramenta desenha.

Não há passo estocástico. Nenhum artista leva `GID_DAS_FEATURES`: o mapa tem células e não
pontos, e a busca põe um segmento por feature na ordem do coeficiente, não na da matriz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from refino.plots.base import (
    GID_REQUEBRAVEL,
    PALETA_DIVERGENTE,
    MatrizVaziaError,
    RecusaDoConstrutorError,
    cores_do_ciclo,
    denominador_ou_recusa,
    exige_abaixo_do_teto,
    inteiro_do_fio,
    nova_figura,
)
from refino.plots.recorte import (
    PONTOS_POR_POLEGADA,
    TAMANHO_DA_FONTE,
    Recorte,
    aplica,
    decimal,
    frase_contada,
    legenda_fora,
    marcas_que_cabem,
    milhar,
)
from refino.statistics.clustering import TETO_DE_FEATURES_PARA_DESENHAR

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "FEATURES_DA_BUSCA",
    "NOME_DA_REFERENCIA",
    "NOME_DO_COEFICIENTE",
    "ROTULO_DAS_CONTRARIAS",
    "ROTULO_DAS_SEGUIDORAS",
    "ROTULO_DA_ESCALA_DA_CORRELACAO",
    "ROTULO_DO_EIXO_DO_MAPA",
    "SEM_NOMES_EM_EIXO_NENHUM",
    "SEM_NOMES_NUM_EIXO",
    "TITULO_DA_BUSCA",
    "TITULO_DO_MAPA_DE_CORRELACAO",
    "BuscaDesalinhadaError",
    "BuscaPorPadrao",
    "CorrelacaoDesalinhadaError",
    "MapaDeCorrelacao",
    "MetodoDeCorrelacaoDesconhecidoError",
    "busca_do_fio",
    "figura_da_busca_por_padrao",
    "figura_do_mapa_de_correlacao",
    "frase_da_ordem",
    "frase_da_referencia",
    "frase_do_coeficiente",
    "mapa_de_correlacao_do_fio",
    "recorte_da_busca_por_padrao",
    "recorte_do_mapa_de_correlacao",
]


TITULO_DO_MAPA_DE_CORRELACAO: Final = "Correlação entre features"
TITULO_DA_BUSCA: Final = "Busca por padrão"

# Quantas features a busca desenha por omissão. O custo de redesenho é dominado pelo texto dos
# rótulos do eixo y, e ele cruza os 100 ms entre 25 e 40 rótulos; 20 deixa margem para uma
# máquina mais lenta. É recorte de quem chama, não teto. O valor difere de `FEATURES_DO_VIP`
# porque esta figura tem custo de base menor (sem legenda na área de dado e sem nuvem de pontos).
FEATURES_DA_BUSCA: Final = 20

ROTULO_DA_ESCALA_DA_CORRELACAO: Final = "coeficiente de correlação"
"""Rótulo do `AxesImage` e da barra de cor."""

ROTULO_DAS_SEGUIDORAS: Final = "acompanham o perfil"
ROTULO_DAS_CONTRARIAS: Final = "contrariam o perfil"
"""Os dois rótulos da busca. São distintos entre si porque os testes indexam as séries por
rótulo, e dois artistas com o mesmo texto colapsariam num só."""

# Nome legível de cada valor do vocabulário fechado. Um teste compara estas chaves com
# `get_args(CorrelationMethod)`, para não haver método sem nome nem nome sem método.
NOME_DO_COEFICIENTE: Final[dict[str, str]] = {
    "pearson": "r de Pearson, sobre os valores",
    "spearman": "rho de Spearman, sobre os postos",
    "kendall": "tau-b de Kendall, sobre pares concordantes",
}

# O mesmo para o tipo de referência da busca. Valor desconhecido é recusado.
NOME_DA_REFERENCIA: Final[dict[str, str]] = {
    "feature": "a feature",
    "profile": "o perfil por grupo",
}


class CorrelacaoDesalinhadaError(RecusaDoConstrutorError):
    """A matriz de correlação não tem a forma que os nomes declaram.

    Além de os nomes terem de casar com o eixo, a matriz tem de ser quadrada, porque indexa o
    mesmo conjunto nas duas direções. Uma matriz `k x m` desenharia um retângulo com os rótulos
    de um eixo a nomear o outro.

    É uma classe própria, e não `MapaDesalinhadoError` do agrupamento, porque aquela guarda
    'amostras em linhas, features em colunas' e esta figura não tem amostras.
    """


class BuscaDesalinhadaError(RecusaDoConstrutorError):
    """Os vetores paralelos da busca não têm o mesmo comprimento.

    `features` e `coefficient` só significam alguma coisa emparelhados. Um vetor mais curto
    desenharia o coeficiente de uma feature ao lado do nome de outra, numa figura bem formada.
    """


class MetodoDeCorrelacaoDesconhecidoError(RecusaDoConstrutorError):
    """O coeficiente ou o tipo de referência não estão no vocabulário fechado.

    Os dois nomes atravessam o fio. Um nome desconhecido escrito cru no rodapé, ou omitido,
    poria a figura a declarar um método que ninguém correu.
    """


def frase_do_coeficiente(metodo: str) -> str:
    """`"pearson"` vira `"r de Pearson, sobre os valores"`. Nome desconhecido é recusado."""
    if metodo not in NOME_DO_COEFICIENTE:
        raise MetodoDeCorrelacaoDesconhecidoError(
            f"o coeficiente {metodo!r} não está no vocabulário fechado deste módulo — "
            f"{sorted(NOME_DO_COEFICIENTE)}. A figura declara o método na própria imagem (R12), e "
            "declarar um nome que ninguém correu é inventar procedência"
        )
    return NOME_DO_COEFICIENTE[metodo]


def frase_da_referencia(tipo: str, referencia: str) -> str:
    """`("feature", "x")` vira `"contra a feature «x»"`. Tipo desconhecido é recusado."""
    if tipo not in NOME_DA_REFERENCIA:
        raise MetodoDeCorrelacaoDesconhecidoError(
            f"o tipo de referência {tipo!r} não está no vocabulário fechado deste módulo — "
            f"{sorted(NOME_DA_REFERENCIA)}. Ele diz se a busca correu contra uma feature ou contra "
            "um perfil por grupo, e as duas leem-se de forma diferente"
        )
    if not referencia.strip():
        raise MetodoDeCorrelacaoDesconhecidoError(
            "a referência da busca chegou em branco: sem ela a figura não diz contra o que "
            "correlacionou, e o rodapé passaria a afirmar uma busca sem alvo"
        )
    return f"contra {NOME_DA_REFERENCIA[tipo]} «{referencia}»"


# ---------------------------------------------------------------------------
# 1. O mapa de correlação
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MapaDeCorrelacao:
    """A matriz quadrada de correlação, os nomes que indexam os dois eixos e o método.

    Os três andam juntos para a conferência de alinhamento correr uma vez, na fronteira.
    """

    valores: np.ndarray
    """`(k, k)`, simétrica por construção. Imutável."""

    features: tuple[str, ...]
    """Um nome por linha e o mesmo por coluna: é a mesma população nos dois eixos."""

    metodo: str
    """O `CorrelationMethod` que produziu os coeficientes."""


def _matriz_quadrada_ou_recusa(valores: Any, nomes: Sequence[str]) -> np.ndarray:
    """Converte o que vem do fio ou do núcleo num `ndarray` quadrado de `float`, ou recusa."""
    try:
        array = np.asarray(valores, dtype=float)
    except (TypeError, ValueError) as erro:
        raise CorrelacaoDesalinhadaError(
            f"a matriz de correlação não converte para matriz de números: {erro}. Nada que venha "
            "de JSON passou pelo mypy, e uma lista de listas irregular chega aqui como objeto"
        ) from erro
    # O teste de vazio vem antes do de forma: `[]` tem uma dimensão e tamanho zero, e o fato é
    # 'não veio população' (`MatrizVaziaError`), não 'tem 1 dimensão'.
    if array.size == 0:
        raise MatrizVaziaError(
            "a matriz de correlação chegou vazia: não há célula para desenhar. Figura em branco "
            "seria indistinguível de figura quebrada"
        )
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise CorrelacaoDesalinhadaError(
            f"a matriz de correlação tem forma {array.shape} e precisa de ser quadrada: ela indexa "
            "a MESMA população nos dois eixos, e um retângulo poria os rótulos de um eixo a nomear "
            "o outro"
        )
    if len(nomes) != array.shape[0]:
        raise CorrelacaoDesalinhadaError(
            f"a matriz é {array.shape} e chegaram {len(nomes)} nome(s) de feature. Nomes "
            "deslocados produzem figura bem formada com os rótulos trocados, que é a única classe "
            "de defeito "
            "de figura que ninguém vê olhando para ela"
        )
    return array


def mapa_de_correlacao_do_fio(
    *,
    valores: Any,
    features: Any,
    metodo: Any,
) -> MapaDeCorrelacao:
    """Monta um `MapaDeCorrelacao` a partir do que o fio ou o núcleo trazem, com as guardas.

    O método é conferido aqui, e não só no rodapé: sem isso um mapa com método desconhecido
    seria desenhado, e a recusa só apareceria ao montar o `Recorte`.
    """
    nomes = tuple(str(f) for f in features) if isinstance(features, (list, tuple)) else ()
    matriz = _matriz_quadrada_ou_recusa(valores, nomes)
    if any(not n.strip() for n in nomes):
        raise CorrelacaoDesalinhadaError(
            "há nome de feature em branco. Marca sem texto lê-se como feature sem identidade, e a "
            "identidade da feature é o que liga a célula ao sinal medido"
        )
    frase_do_coeficiente(str(metodo))
    matriz = matriz.copy()
    matriz.flags.writeable = False
    return MapaDeCorrelacao(valores=matriz, features=nomes, metodo=str(metodo))


def recorte_do_mapa_de_correlacao(
    *,
    mapa: MapaDeCorrelacao,
    n_features_totais: Any,
    n_amostras: Any,
    n_indefinidas: Any,
    matriz: str,
    amostragem: str,
) -> Recorte:
    """O rodapé do mapa: de que população os coeficientes saíram, por que método e o que sobrou.

    `n_indefinidas` conta as features que o núcleo descartou. Uma feature constante não está na
    matriz, e sem a contagem 'esta matriz tem 60 features' e 'tinha 63 e três não dão para
    medir' sairiam iguais.

    O número de amostras entra no texto do método, não no 'N de M features': a contagem do
    rodapé é de features, e a correlação é medida sobre amostras.
    """
    descartadas = max(0, inteiro_do_fio(n_indefinidas))
    indefinidas = frase_contada(
        descartadas,
        nenhuma="nenhuma feature ficou sem coeficiente definido",
        uma="1 feature ficou sem coeficiente definido e saiu da matriz",
        algumas="features ficaram sem coeficiente definido e saíram da matriz",
    )
    amostras = max(0, inteiro_do_fio(n_amostras))
    return Recorte(
        desenhadas=len(mapa.features),
        disponiveis=denominador_ou_recusa(n_features_totais, de="a correlação"),
        amostragem=amostragem,
        matriz=matriz,
        metodo=(
            f"{frase_do_coeficiente(mapa.metodo)}, sobre {milhar(amostras)} amostras de estudo — "
            f"QC e validação ficam de fora. A escala vai de -1 a +1 e é fixa, para duas execuções "
            f"serem comparáveis. Sem teste estatístico: um coeficiente não tem p-valor aqui, e "
            f"{milhar(len(mapa.features))} features dão "
            f"{milhar(len(mapa.features) * (len(mapa.features) - 1) // 2)} pares. {indefinidas}"
        ),
    )


def _caixa_do_eixo(eixo: Axes) -> tuple[float, float]:
    """A largura e a altura da caixa deste eixo, em pontos tipográficos.

    É medida com o renderizador, porque a barra de cor já tirou largura do eixo. A medição
    acontece antes do `tight_layout` de `aplica`, e aqui isso é exato: a barra de cor fixa a
    geometria horizontal e a reserva do rodapé fixa a vertical, então a caixa não se move.

    Sem renderizador (figura sem canvas) devolve `(0, 0)`, e nenhuma marca cabe.
    """
    figura = eixo.get_figure()
    obter = getattr(getattr(figura, "canvas", None), "get_renderer", None)
    if figura is None or not callable(obter):  # pragma: no cover - `nova_figura` liga um `Agg`
        return (0.0, 0.0)
    caixa = eixo.get_window_extent(obter())
    por_ponto = PONTOS_POR_POLEGADA / float(figura.dpi)
    return (float(caixa.width) * por_ponto, float(caixa.height) * por_ponto)


ROTULO_DO_EIXO_DO_MAPA: Final = "feature"

SEM_NOMES_NUM_EIXO: Final = "os nomes das features saem só num dos eixos: não cabem no outro"
SEM_NOMES_EM_EIXO_NENHUM: Final = (
    "os nomes das features não cabem em eixo nenhum; reduza a população para os ver"
)
"""As duas frases que declaram por que um eixo de marcas ficou sem nome.

São três estados: os nomes cabem nos dois eixos, só num, ou em nenhum. Com as 60 features do
teto os dois eixos perdem os nomes, e uma frase única que apontasse para o eixo oposto seria
falsa nesse caso.

As frases vão na segunda linha do título, não no rótulo do eixo. Um rótulo de eixo y é
desenhado rodado 90° e seria cortado pelo topo da figura, e `recorte.requebra_o_que_couber`
alcança o título e os textos marcados, não os rótulos de eixo.
"""


def _titulo_do_mapa(*, cabem_em_x: bool, cabem_em_y: bool) -> str:
    """O título, com a segunda linha a declarar a ausência de nomes quando ela existe."""
    if cabem_em_x and cabem_em_y:
        return TITULO_DO_MAPA_DE_CORRELACAO
    aviso = SEM_NOMES_NUM_EIXO if (cabem_em_x or cabem_em_y) else SEM_NOMES_EM_EIXO_NENHUM
    return "\n".join((TITULO_DO_MAPA_DE_CORRELACAO, aviso))


def figura_do_mapa_de_correlacao(*, mapa: MapaDeCorrelacao, recorte: Recorte) -> Figure:
    """A matriz de correlação entre features, com a escala fixa em ±1.

    Desenha a população inteira que recebe. Quantas features correlacionar é decisão de quem
    chama; acima do teto herdado de `statistics.clustering` a figura é recusada, e a recusa é
    de apresentação, não de cálculo.

    A barra de cor recebe a vírgula decimal e os rótulos de feature não. `recorte.aplica` troca
    o ponto pela vírgula só em escala com `ScalarFormatter`, e os rótulos de feature são postos
    por `set_*ticklabels`, que instala um `FuncFormatter`.
    """
    exige_abaixo_do_teto(
        len(mapa.features), TETO_DE_FEATURES_PARA_DESENHAR, de="o mapa de correlação"
    )

    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    imagem = eixo.imshow(
        mapa.valores,
        aspect="auto",
        interpolation="nearest",
        cmap=PALETA_DIVERGENTE,
        # Limites fixos, não derivados do dado, para duas execuções serem comparáveis.
        vmin=-1.0,
        vmax=1.0,
        label=ROTULO_DA_ESCALA_DA_CORRELACAO,
    )
    barra = figura.colorbar(imagem, ax=eixo)
    barra.set_label(ROTULO_DA_ESCALA_DA_CORRELACAO, fontsize=TAMANHO_DA_FONTE)
    # As marcas e o rótulo da barra vão à esquerda dela. À direita, o que sobressai da caixa do
    # eixo taparia os números da escala.
    barra.ax.yaxis.set_ticks_position("left")
    barra.ax.yaxis.set_label_position("left")

    posicoes = np.arange(len(mapa.features), dtype=float).tolist()
    largura, altura = _caixa_do_eixo(eixo)
    # Cada eixo decide sozinho se os nomes cabem. A caixa é mais larga que alta, então o eixo y
    # aperta primeiro, e uma decisão única apagaria os nomes do x quando só os do y não cabem.
    eixo.set_xticks(posicoes)
    cabem_em_x = marcas_que_cabem(largura, len(mapa.features))
    if cabem_em_x:
        eixo.set_xticklabels(list(mapa.features), rotation=90, fontsize=TAMANHO_DA_FONTE)
    else:
        eixo.set_xticklabels([])
    eixo.set_yticks(posicoes)
    cabem_em_y = marcas_que_cabem(altura, len(mapa.features))
    if cabem_em_y:
        eixo.set_yticklabels(list(mapa.features), fontsize=TAMANHO_DA_FONTE)
    else:
        eixo.set_yticklabels([])
    # A ausência dos nomes é declarada na segunda linha do título. Não pode ir no `Recorte`, que
    # é montado antes da figura: só a geometria, depois do `imshow` e da barra de cor, sabe se
    # os nomes cabem.
    eixo.set_xlabel(ROTULO_DO_EIXO_DO_MAPA)
    eixo.set_ylabel(ROTULO_DO_EIXO_DO_MAPA)
    eixo.set_title(
        _titulo_do_mapa(cabem_em_x=cabem_em_x, cabem_em_y=cabem_em_y),
        fontsize=TAMANHO_DA_FONTE,
    )

    aplica(figura, recorte)
    return figura


# ---------------------------------------------------------------------------
# 2. A busca por padrão
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuscaPorPadrao:
    """As features ordenadas pela aderência ao perfil, e o perfil contra o qual elas correram.

    `features` e `coeficiente` são paralelos; a conferência corre uma vez, em `busca_do_fio`.
    """

    features: tuple[str, ...]
    coeficiente: np.ndarray
    """Um por feature, na mesma ordem. Imutável."""

    referencia: str
    tipo_da_referencia: str
    perfil: tuple[float, ...]
    ordem_dos_grupos: tuple[str, ...]
    """A ordem assumida entre os grupos. A figura a escreve no rodapé."""

    metodo: str


def busca_do_fio(
    *,
    features: Any,
    coeficiente: Any,
    referencia: Any,
    tipo_da_referencia: Any,
    perfil: Any = (),
    ordem_dos_grupos: Any = (),
    metodo: Any,
) -> BuscaPorPadrao:
    """Monta uma `BuscaPorPadrao` a partir do que o fio ou o núcleo trazem, com as guardas."""
    nomes = tuple(str(f) for f in features) if isinstance(features, (list, tuple)) else ()
    if not nomes:
        raise MatrizVaziaError(
            "a busca por padrão chegou sem feature nenhuma: não há população para desenhar. Um "
            "construtor de `plots/` recebe a população inteira (KD2), e população vazia é defeito "
            "de quem chama — figura em branco seria indistinguível de figura quebrada"
        )
    valores = np.asarray(
        [
            float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else np.nan
            for c in (coeficiente if isinstance(coeficiente, (list, tuple, np.ndarray)) else ())
        ],
        dtype=float,
    )
    if valores.shape[0] != len(nomes):
        raise BuscaDesalinhadaError(
            f"chegaram {len(nomes)} nome(s) de feature e {valores.shape[0]} coeficiente(s). "
            "Os dois vetores indexam a MESMA feature em cada posição, e um vetor mais curto "
            "desenharia o "
            "coeficiente de uma feature ao lado do nome de outra"
        )
    grupos = (
        tuple(str(g) for g in ordem_dos_grupos)
        if isinstance(ordem_dos_grupos, (list, tuple))
        else ()
    )
    numeros = (
        tuple(float(v) for v in perfil if isinstance(v, (int, float)) and not isinstance(v, bool))
        if isinstance(perfil, (list, tuple))
        else ()
    )
    if grupos and len(numeros) != len(grupos):
        raise BuscaDesalinhadaError(
            f"chegaram {len(numeros)} número(s) de perfil para {len(grupos)} grupo(s). O perfil é "
            "por grupo e a ordem é o que lhe dá significado (OQ2): um vetor mais curto poria cada "
            "número ao lado do grupo errado, com a figura bem formada"
        )
    frase_do_coeficiente(str(metodo))
    frase_da_referencia(str(tipo_da_referencia), str(referencia))
    valores.flags.writeable = False
    return BuscaPorPadrao(
        features=nomes,
        coeficiente=valores,
        referencia=str(referencia),
        tipo_da_referencia=str(tipo_da_referencia),
        perfil=numeros,
        ordem_dos_grupos=grupos,
        metodo=str(metodo),
    )


def frase_da_ordem(busca: BuscaPorPadrao) -> str:
    """O que o rodapé diz sobre a ordem assumida entre os grupos.

    Com um perfil, a ordem torna o número interpretável; com uma feature, torna o perfil dela
    legível. A figura a escreve porque a ordenação é por ponto de código, e não a que uma pessoa
    esperaria: `ATLL` vem antes de `Asymptomatic` porque `T` vem antes de `s`.

    Sem grupos declarados a frase diz isso por extenso. Matriz sem fator clínico é caso
    legítimo, e o silêncio se leria como 'a ordem não importa'.
    """
    if not busca.ordem_dos_grupos:
        return "a matriz não declara grupo nenhum, então nenhuma ordem foi assumida"
    pares = ", ".join(
        f"{grupo} = {decimal(valor)}"
        for grupo, valor in zip(busca.ordem_dos_grupos, busca.perfil, strict=False)
    )
    return (
        f"a ordem assumida entre os grupos é {list(busca.ordem_dos_grupos)}, por ponto de código — "
        f"e o perfil nela é {pares}"
    )


def recorte_da_busca_por_padrao(
    *,
    busca: BuscaPorPadrao,
    n_desenhadas: int,
    n_features_totais: Any,
    n_amostras: Any,
    n_indefinidas: Any,
    matriz: str,
) -> Recorte:
    """O rodapé da busca: contra o quê, por que método, em que ordem de grupos e quantas ficaram.

    Não há argumento de amostragem. A busca corre sobre todas as features e o recorte é só de
    quantas se desenham; um campo livre permitiria declarar um sorteio que não aconteceu.
    """
    descartadas = max(0, inteiro_do_fio(n_indefinidas))
    indefinidas = frase_contada(
        descartadas,
        nenhuma="nenhuma feature ficou sem coeficiente definido",
        uma="1 feature ficou sem coeficiente definido e saiu da lista",
        algumas="features ficaram sem coeficiente definido e saíram da lista",
    )
    amostras = max(0, inteiro_do_fio(n_amostras))
    return Recorte(
        desenhadas=n_desenhadas,
        disponiveis=denominador_ou_recusa(n_features_totais, de="a busca por padrão"),
        amostragem=(
            f"as {milhar(n_desenhadas)} de maior coeficiente em módulo, entre as "
            f"{milhar(len(busca.features))} medidas"
        ),
        matriz=matriz,
        metodo=(
            f"{frase_do_coeficiente(busca.metodo)} "
            f"{frase_da_referencia(busca.tipo_da_referencia, busca.referencia)}, sobre "
            f"{milhar(amostras)} amostras de estudo — QC e validação ficam de fora. Sem teste "
            f"estatístico: o coeficiente não traz p-valor, e a ordem é a dele com SINAL. "
            f"{frase_da_ordem(busca)}. {indefinidas}"
        ),
    )


def _escolhidas(busca: BuscaPorPadrao, n_features: int) -> np.ndarray:
    """Os índices das `n_features` de maior coeficiente em módulo, na ordem do coeficiente com
    sinal.

    O desempate é pelo nome, para o conjunto desenhado não depender da ordem em que as features
    chegaram.
    """
    definidos = np.isfinite(busca.coeficiente)
    candidatos = [int(i) for i in np.flatnonzero(definidos)]
    por_modulo = sorted(
        candidatos, key=lambda i: (-abs(float(busca.coeficiente[i])), busca.features[i])
    )[: max(0, n_features)]
    return np.array(
        sorted(por_modulo, key=lambda i: (-float(busca.coeficiente[i]), busca.features[i])),
        dtype=int,
    )


def figura_da_busca_por_padrao(
    *, busca: BuscaPorPadrao, n_features: int, recorte: Recorte
) -> Figure:
    """As features que mais acompanham o perfil de referência e as que mais o contrariam.

    Desenha a população inteira que recebe; quantas desenhar é `n_features`, decisão de quem
    chama.

    São duas séries, uma por sinal. Quando um dos lados está vazio, a série dele não é
    desenhada, para a legenda não prometer segmentos que não existem.

    Sem nenhuma feature desenhável, a figura declara isso em texto em vez de sair vazia.
    """
    escolhidas = _escolhidas(busca, n_features)
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    eixo.set_xlabel(ROTULO_DA_ESCALA_DA_CORRELACAO)
    eixo.set_ylabel("feature")
    eixo.set_title(TITULO_DA_BUSCA, fontsize=TAMANHO_DA_FONTE)

    if escolhidas.size == 0:
        # O texto é marcado como requebrável, para `recorte.aplica` o partir pela largura real
        # do eixo.
        eixo.text(
            0.02,
            0.5,
            "nenhuma feature tem coeficiente definido contra este perfil — o resultado é vazio, "
            "e o vazio é um achado",
            transform=eixo.transAxes,
            va="center",
            fontsize=TAMANHO_DA_FONTE,
            gid=GID_REQUEBRAVEL,
        )
        eixo.set_xlim(-1.0, 1.0)
        eixo.set_yticks([])
        aplica(figura, recorte)
        return figura

    valores = np.asarray([float(busca.coeficiente[i]) for i in escolhidas.tolist()], dtype=float)
    posicoes = np.arange(escolhidas.size, dtype=float)
    positiva, negativa = cores_do_ciclo(2)

    maos: list[Artist] = []
    for mascara, rotulo, cor in (
        (valores >= 0.0, ROTULO_DAS_SEGUIDORAS, positiva),
        (valores < 0.0, ROTULO_DAS_CONTRARIAS, negativa),
    ):
        if not mascara.any():
            continue
        maos.append(
            eixo.hlines(
                posicoes[mascara],
                0.0,
                valores[mascara],
                colors=cor,
                linewidth=4.0,
                label=rotulo,
            )
        )

    eixo.set_yticks(posicoes.tolist())
    eixo.set_yticklabels(
        [busca.features[i] for i in escolhidas.tolist()], fontsize=TAMANHO_DA_FONTE
    )
    # As posições foram atribuídas em ordem decrescente e o eixo y cresce para cima; a inversão
    # põe o maior coeficiente no alto.
    eixo.invert_yaxis()
    # Os limites são os da escala, não os do dado: duas buscas são comparadas lado a lado.
    eixo.set_xlim(-1.0, 1.0)
    # A marca do zero é a grade e não um `axvline`: uma linha de eixo entraria em `ax.lines` e
    # os testes a leriam como uma série de dado.
    eixo.grid(visible=True, axis="x", linewidth=0.4, alpha=0.4)
    eixo.set_axisbelow(True)
    legenda_fora(eixo, maos)

    aplica(figura, recorte)
    return figura

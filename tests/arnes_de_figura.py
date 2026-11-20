"""O arnês das figuras: as quatro classes de asserção, e as plantas que as provam.

## Por que quatro classes, e por que as duas últimas são as que importam

As três mutilações que ocorrem a quem escreve um teste de figura são todas mutações de **dado**:
uma série a menos, os eixos trocados, o sinal invertido. As três se apanham lendo o dado do artista.

Mas um `set_ylim` que corta a série inteira não muda dado nenhum: `ax.lines[0].get_ydata()`
continua a devolver exactamente o array certo, e a tela fica **em branco**. As três plantas passam,
os rótulos estão lá, a legenda está lá, e a figura não mostra nada. Daí a classe (c), o estado da
vista. E daí a classe (d): o número que a figura **declara** sobre si — *"57 de 6.035 features"* —
não é dado desenhado, é uma frase, e uma frase errada é a única coisa numa figura que outra pessoa
pode desmentir.

| Classe | O que lê | Função |
|---|---|---|
| (a) dado do artista | `get_xdata`, `get_offsets`, `get_segments`, `get_array` | `series` |
| (b) rótulos e legenda | `get_xlabel`, `get_title`, `get_legend` | `rotulos`, `confere_rotulos` |
| (c) estado da vista | `get_xlim`, `get_yscale`, `get_alpha` | `vista`, `confere_vista` |
| (d) número declarado | os `Text` da figura | `numeros_declarados`, `confere_numeros` |

## A classe (a) lia DOIS artistas, e as figuras novas desenham quatro

Até esta unidade `series` lia exactamente `ax.lines` (por `get_xdata`/`get_ydata`) e
`ax.collections` (por `get_offsets`). **Medido em 13/09/2026, sobre figuras de propósito:**

| O artista | O que a classe (a) lia | O que está desenhado |
|---|---|---|
| `AxesImage`, de `imshow` | **nada** — `ax.images` não era percorrido | uma matriz inteira |
| `LineCollection`, de segmentos | **uma série com um ponto em `(0, 0)`** | 250 arestas |

No primeiro caso `confere_series(ax, {})` devolvia `[]` — asserção de ausência satisfeita por
**vacuidade**, sobre um mapa de calor com conteúdo. No segundo ela acusava a série a mais,
mas pelo rótulo: o **dado** lido era o par nulo que `Collection.get_offsets()` devolve quando não há
deslocamento nenhum, e trocar o conteúdo inteiro não mudava a leitura. É o instrumento a
escrever sobre o alvo — na forma mais barata que existe: `[[0., 0.]]` lê-se como *"um ponto"*.

As duas leituras ficavam **idênticas com o conteúdo trocado**, e é isso que a planta afirma
que deixou de ser verdade. Cinco das figuras novas — dendrograma, mapa de calor, rede de correlação
— são desenhadas exactamente com estes dois artistas.

**As classes (c) e (d) afirmam contra valores DERIVADOS e nunca contra literais.** `confere_vista`
não sabe qual devia ser o limite: sabe que os pontos desenhados têm de caber dentro dele, e o que
está desenhado ela lê da figura. `confere_numeros` recebe os números que o teste derivou da entrada,
de fonte independente do construtor.

## As plantas trazem a âncora dentro delas

*A minha planta cai sobre dado em que as duas coisas que ela separa diferem?* Trocar os eixos de uma
figura em que `x == y` não muda nada; inverter o sinal de uma série toda a zeros não muda nada;
cortar o eixo de uma série constante pode não a excluir. Cada `planta_*` afirma a condição de
separação **antes** de mutilar, e levanta `PlantaQueNaoSeparaError` se ela não valer — porque
planta vácua produz negativo verde, que é a verificação vazia mais convincente que existe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import Collection
    from matplotlib.figure import Figure

__all__ = [
    "PlantaQueNaoSeparaError",
    "Rotulos",
    "Serie",
    "Vista",
    "confere_conteudo",
    "confere_numeros",
    "confere_rotulos",
    "confere_series",
    "confere_vista",
    "milhar_do_teste",
    "numeros_declarados",
    "planta_conteudo_trocado",
    "planta_corte_de_eixo",
    "planta_eixos_trocados",
    "planta_serie_removida",
    "planta_sinal_invertido",
    "rotulos",
    "series",
    "texto_corrido",
    "vista",
]

# Tolerância de comparação numérica entre o que se pediu e o que o artista guardou. O `matplotlib`
# guarda o array que recebeu, então a diferença esperada é zero; a folga existe para o caso de o
# construtor converter de `float32`.
TOLERANCIA = 1e-9

# Um número em texto, na forma que o rodapé produz: `6.035`, `57`, `0,25`, `1e-6`.
NUMERO = re.compile(r"\d[\d.,]*(?:[eE][+-]?\d+)?")


class PlantaQueNaoSeparaError(AssertionError):
    """A planta caiu sobre dado em que as duas coisas que ela separa **não** diferem.

    É `AssertionError` e não `ValueError` de propósito: isto é a reprovação de um teste, não um erro
    de uso de biblioteca. Um negativo que planta e não separa passa, e passa por vacuidade.
    """


@dataclass(frozen=True)
class Serie:
    """Uma série desenhada — linha, dispersão, feixe de segmentos ou imagem.

    **`valores` é o quarto campo e só a imagem o preenche**, e ele existe porque um `AxesImage` não
    tem coordenada por ponto: tem uma **matriz** e um retângulo onde ela é desenhada. Pôr o conteúdo
    em `x`/`y` seria pôr duas coisas no mesmo valor de retorno, e deixá-lo
    de fora seria cegueira: com só o retângulo, trocar o conteúdo inteiro do
    mapa de calor não muda a leitura.
    """

    rotulo: str
    x: np.ndarray
    y: np.ndarray
    valores: np.ndarray | None = None
    """O conteúdo, quando o artista carrega um valor por célula em vez de um par de coordenadas."""


@dataclass(frozen=True)
class Rotulos:
    """Classe (b): o que a figura escreve à volta do dado."""

    x: str
    y: str
    titulo: str
    legenda: tuple[str, ...]


@dataclass(frozen=True)
class Vista:
    """Classe (c): o estado da vista, que nenhuma mutação de dado altera."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    xscale: str
    yscale: str
    visiveis: tuple[bool, ...]
    alphas: tuple[float, ...]


# ---------------------------------------------------------------------------
# (a) o dado do artista
# ---------------------------------------------------------------------------


def series(ax: Axes) -> tuple[Serie, ...]:
    """Todo artista de dado dos eixos, na ordem `lines`, `collections`, `images`.

    Linhas, dispersões, feixes de segmentos e imagens juntos, porque a pergunta *"o que está
    desenhado aqui?"* não distingue os quatro — e um construtor que troque `plot` por `scatter`, ou
    `plot` por `LineCollection`, não deve fazer o teste ficar mudo.

    **A ordem é contrato e não acaso:** `vista` percorre os mesmos três grupos na mesma ordem e
    `confere_vista` faz `zip(..., strict=True)` das duas leituras. Acrescentar um artista a uma e
    não à outra reprova de imediato, que é o desfecho certo.

    **`get_segments` ANTES de `get_offsets`, e a ordem é o conteúdo.** Uma `LineCollection` tem as
    duas: os segmentos são o que está desenhado, e os deslocamentos são a translação aplicada a
    eles — que numa coleção comum é o par nulo `[[0, 0]]`. Perguntar primeiro pelos deslocamentos é
    exactamente o que fazia 250 arestas lerem-se como *"um ponto na origem"*.
    """
    achadas: list[Serie] = []
    for linha in ax.lines:
        achadas.append(
            Serie(
                rotulo=str(linha.get_label()),
                x=np.asarray(linha.get_xdata(), dtype=float),
                y=np.asarray(linha.get_ydata(), dtype=float),
            )
        )
    for colecao in ax.collections:
        vertices = _vertices_dos_segmentos(colecao)
        if vertices is not None:
            achadas.append(
                Serie(rotulo=str(colecao.get_label()), x=vertices[:, 0], y=vertices[:, 1])
            )
            continue
        pontos = np.asarray(colecao.get_offsets(), dtype=float)
        if pontos.size == 0:
            achadas.append(Serie(str(colecao.get_label()), np.empty(0), np.empty(0)))
            continue
        achadas.append(Serie(rotulo=str(colecao.get_label()), x=pontos[:, 0], y=pontos[:, 1]))
    for imagem in ax.images:
        esquerda, direita, baixo, cima = (float(v) for v in imagem.get_extent())
        achadas.append(
            Serie(
                rotulo=str(imagem.get_label()),
                x=np.array([esquerda, direita], dtype=float),
                y=np.array([baixo, cima], dtype=float),
                valores=np.asarray(imagem.get_array(), dtype=float),
            )
        )
    return tuple(achadas)


def _vertices_dos_segmentos(colecao: Collection) -> np.ndarray | None:
    """Os vértices de todos os segmentos de uma `LineCollection`, ou `None` se não for uma.

    **Devolve `None` e não um array vazio.** *"Não é coleção de segmentos"* e *"é
    uma coleção de segmentos e está vazia"* são fatos diferentes com desfechos diferentes: o
    primeiro manda ler os deslocamentos, o segundo é uma série vazia legítima. Um array vazio para
    os dois faria uma `LineCollection` sem arestas cair no caminho dos deslocamentos, e voltar a ler
    o par nulo da origem — que é o defeito exacto que esta função existe para fechar.

    **A concatenação perde a estrutura de aresta de propósito.** A classe (a) afirma *que dado está
    desenhado*, não *como ele está agrupado*; a alternativa — comparar listas de segmentos — não
    entraria em `confere_series`, que compara vetores por eixo. Duas árvores de dendrograma com os
    mesmos vértices e arestas diferentes passariam nesta classe, e isso vai declarado: o que as
    separa é a classe (d), o número que a figura declara sobre si.
    """
    obtem = getattr(colecao, "get_segments", None)
    if obtem is None:
        return None
    segmentos = obtem()
    if not len(segmentos):
        return np.empty((0, 2), dtype=float)
    return np.concatenate([np.asarray(s, dtype=float) for s in segmentos], axis=0)


def confere_series(
    ax: Axes, esperadas: Mapping[str, tuple[Sequence[float], Sequence[float]]]
) -> list[str]:
    """Acusa série ausente, série a mais, e série cujo dado não é o que se pediu.

    **Devolve a lista em vez de afirmar**, que é o que torna o negativo honesto: a planta entra como
    **argumento** desta função, então o negativo percorre exactamente o caminho que o portão
    percorre. Negativo em processo não consegue ficar inerte.

    É esta função que apanha as três mutações de dado. **Eixos trocados** aparecem como `x` que não
    bate; **sinal invertido**, como `y` que não bate; **série removida**, como rótulo em falta.
    """
    achados: list[str] = []
    desenhadas = {s.rotulo: s for s in series(ax)}

    faltam = sorted(set(esperadas) - set(desenhadas))
    if faltam:
        achados.append(f"séries esperadas e NÃO desenhadas: {faltam}")
    sobram = sorted(set(desenhadas) - set(esperadas))
    if sobram:
        achados.append(f"séries desenhadas e não esperadas: {sobram}")

    for rotulo in sorted(set(esperadas) & set(desenhadas)):
        x_esperado, y_esperado = esperadas[rotulo]
        serie = desenhadas[rotulo]
        for eixo, obtido, esperado in (
            ("x", serie.x, np.asarray(x_esperado, dtype=float)),
            ("y", serie.y, np.asarray(y_esperado, dtype=float)),
        ):
            if obtido.shape != esperado.shape:
                achados.append(
                    f"série {rotulo!r}, eixo {eixo}: {obtido.shape} pontos desenhados contra "
                    f"{esperado.shape} esperados"
                )
            elif not np.allclose(obtido, esperado, atol=TOLERANCIA, equal_nan=True):
                pior = int(np.nanargmax(np.abs(obtido - esperado)))
                achados.append(
                    f"série {rotulo!r}, eixo {eixo}: divergem; a maior diferença é no índice "
                    f"{pior} — desenhado {obtido[pior]!r}, esperado {esperado[pior]!r}"
                )
    return achados


def confere_conteudo(ax: Axes, esperados: Mapping[str, Sequence[Sequence[float]]]) -> list[str]:
    """Acusa imagem cujo CONTEÚDO não é o que se pediu, e imagem que ninguém esperava.

    **Função própria e não um terceiro elemento em `confere_series`, e a razão é o que cada uma
    afirma.** `confere_series` compara **coordenadas**, e para um `AxesImage` as coordenadas são o
    retângulo em que a matriz é desenhada — uma afirmação legítima e independente: uma imagem posta
    no sítio errado é um defeito, e o conteúdo dela pode estar certo. Um valor de retorno que
    cobrisse as duas tornaria os dois fatos indistinguíveis em contagem.

    **É esta que fecha essa cegueira**: sem ela, trocar o conteúdo inteiro de um mapa de
    calor não muda asserção nenhuma da classe (a), porque o retângulo é o mesmo.

    **A comparação é sobre a matriz com a FORMA dela**, e não sobre os valores achatados: uma
    transposta tem os mesmos valores noutra forma, e transpor um mapa de calor troca as features
    pelas amostras — que é a mutação mais provável desta família.
    """
    achados: list[str] = []
    com_conteudo = {s.rotulo: s.valores for s in series(ax) if s.valores is not None}

    faltam = sorted(set(esperados) - set(com_conteudo))
    if faltam:
        achados.append(f"conteúdos esperados e NÃO desenhados: {faltam}")
    sobram = sorted(set(com_conteudo) - set(esperados))
    if sobram:
        achados.append(f"artistas com conteúdo e não esperados: {sobram}")

    for rotulo in sorted(set(esperados) & set(com_conteudo)):
        obtido = com_conteudo[rotulo]
        esperado = np.asarray(esperados[rotulo], dtype=float)
        assert obtido is not None  # a compreensão acima já o garante; é para o leitor
        if obtido.shape != esperado.shape:
            achados.append(
                f"conteúdo de {rotulo!r}: forma {obtido.shape} desenhada contra {esperado.shape} "
                "esperada"
            )
        elif not np.allclose(obtido, esperado, atol=TOLERANCIA, equal_nan=True):
            diferentes = int(np.count_nonzero(~np.isclose(obtido, esperado, atol=TOLERANCIA)))
            achados.append(
                f"conteúdo de {rotulo!r}: {diferentes} de {obtido.size} célula(s) divergem"
            )
    return achados


# ---------------------------------------------------------------------------
# (b) rótulos e legenda
# ---------------------------------------------------------------------------


def sem_a_matriz(texto: str) -> str:
    """O título sem a linha «Matriz: …» que `recorte.aplica` acrescenta.

    A classe (b) afirma o título que o construtor escreveu; a linha da matriz é afirmada à parte,
    contra o rodapé, por `confere_a_matriz_do_titulo` — que `confere_rotulos` chama. São duas
    afirmações, e misturá-las faria cada teste de figura repetir a frase do rodapé.
    """
    from refino.plots.recorte import PREFIXO_DA_MATRIZ_NO_TITULO

    corte = texto.rfind(PREFIXO_DA_MATRIZ_NO_TITULO)
    return texto if corte < 0 else texto[:corte].rstrip()


def confere_a_matriz_do_titulo(figura: Figure) -> list[str]:
    """Acusa a figura cujo título não diz a matriz que o rodapé diz.

    **O oráculo é o outro caminho**: a frase sai do rodapé que `aplica` desenhou, e não de
    um literal. Figura sem rodapé não tem o que conferir e devolve vazio — e quem garante que as
    figuras do registo TÊM rodapé é a classe (d).
    """
    from refino.plots.base import GID_DO_RODAPE
    from refino.plots.recorte import PREFIXO_DA_MATRIZ_NO_TITULO, SEPARADOR

    rodapes = [t for t in figura.texts if t.get_gid() == GID_DO_RODAPE]
    if not rodapes:
        return []
    corrido = texto_corrido(rodapes[0].get_text())
    achado = re.search(r"matriz: (.*?)" + re.escape(SEPARADOR) + "método:", corrido)
    if achado is None:
        return [f"o rodapé não tem o campo da matriz: {corrido!r}"]
    esperada = texto_corrido(PREFIXO_DA_MATRIZ_NO_TITULO + achado.group(1))
    titulos = [t.get_text() for t in figura.texts] + [e.get_title() for e in figura.axes]
    if not any(esperada in texto_corrido(t) for t in titulos):
        return [f"o título não diz a matriz do rodapé: {esperada!r}"]
    return []


def rotulos(ax: Axes) -> Rotulos:
    legenda = ax.get_legend()
    return Rotulos(
        x=ax.get_xlabel(),
        y=ax.get_ylabel(),
        titulo=sem_a_matriz(ax.get_title()),
        legenda=tuple(t.get_text() for t in legenda.get_texts()) if legenda else (),
    )


def texto_corrido(texto: str) -> str:
    """O texto com o espaço em branco COLAPSADO, e isto não é conveniência.

    Um rótulo que quebra em linhas continua a ser o mesmo rótulo: `recorte.aplica` requebra os
    títulos pela largura real do eixo, e a largura real depende da legenda, do rodapé e do número
    de eixos. Comparar com `\\n` dentro seria fixar a **quebra** como contrato quando o contrato é o
    **texto** — e a comparação passaria a reprovar sempre que uma figura mudasse de tamanho.

    É um falso negativo dentro de uma asserção: a
    frase existe, está inteira, e a comparação diz que não. Colapsa-se **dos dois lados**, que é a
    forma prescrita — nunca só a do meio.
    """
    return " ".join(texto.split())


def confere_rotulos(
    ax: Axes, *, x: str, y: str, titulo: str, legenda: Sequence[str] = ()
) -> list[str]:
    """Acusa eixo sem rótulo, rótulo trocado e legenda que não nomeia o que está desenhado.

    **A comparação é sobre o texto corrido, não sobre a quebra** — ver `texto_corrido`. O que esta
    classe afirma é *"o eixo diz o que devia dizer"*, e onde as linhas caem é geometria, que é
    assunto da classe (c) e da leitura da imagem montada.
    """
    achados: list[str] = []
    lidos = rotulos(ax)
    for nome, obtido, esperado in (
        ("rótulo de x", lidos.x, x),
        ("rótulo de y", lidos.y, y),
        # Dos dois lados, como o espaço em branco: um esperado lido de outra figura já traz a linha.
        ("título", lidos.titulo, sem_a_matriz(titulo)),
    ):
        if texto_corrido(obtido) != texto_corrido(esperado):
            achados.append(f"{nome}: {obtido!r} desenhado, {esperado!r} esperado")
    if tuple(legenda) != lidos.legenda:
        achados.append(f"legenda: {lidos.legenda} desenhada, {tuple(legenda)} esperada")
    figura = ax.get_figure()
    if figura is not None:
        achados += confere_a_matriz_do_titulo(figura)  # type: ignore[arg-type]
    return achados


# ---------------------------------------------------------------------------
# (c) o estado da vista
# ---------------------------------------------------------------------------


def vista(ax: Axes) -> Vista:
    """O estado da vista, com `visiveis` e `alphas` na MESMA ordem que `series` devolve.

    **`ax.images` entra aqui junto com a classe (a), e tem de entrar.** `confere_vista`
    faz `zip(series(ax), lida.visiveis, lida.alphas, strict=True)`; um grupo de artistas lido por
    uma das duas e não pela outra reprova de imediato — que é o desfecho certo, e é o que impede as
    duas leituras de se afastarem em silêncio.
    """
    artistas = [*ax.lines, *ax.collections, *ax.images]
    return Vista(
        xlim=tuple(float(v) for v in ax.get_xlim()),  # type: ignore[arg-type]
        ylim=tuple(float(v) for v in ax.get_ylim()),  # type: ignore[arg-type]
        xscale=ax.get_xscale(),
        yscale=ax.get_yscale(),
        visiveis=tuple(bool(a.get_visible()) for a in artistas),
        alphas=tuple(1.0 if a.get_alpha() is None else float(a.get_alpha()) for a in artistas),
    )


def confere_vista(ax: Axes, *, alpha_minimo: float = 0.05) -> list[str]:
    """Acusa a figura que está tecnicamente certa e visualmente vazia.

    Três defeitos, e nenhum deles altera dado nenhum:

    1. **limite que exclui a série** — `set_ylim` fora do intervalo dos pontos desenhados. O limite
       esperado não é literal nenhum: é *"tem de conter o que está desenhado"*, e o que está
       desenhado lê-se da figura.
    2. **escala logarítmica sobre dado não positivo** — o `matplotlib` deixa passar e recorta os
       pontos em silêncio.
    3. **artista invisível ou transparente** — série que existe no objeto e não na tela.
    """
    achados: list[str] = []
    lida = vista(ax)
    desenhadas = series(ax)

    for serie in desenhadas:
        for eixo, valores, (menor, maior), escala in (
            ("x", serie.x, sorted(lida.xlim), lida.xscale),
            ("y", serie.y, sorted(lida.ylim), lida.yscale),
        ):
            finitos = valores[np.isfinite(valores)]
            if finitos.size == 0:
                continue
            dentro = (finitos >= menor) & (finitos <= maior)
            if not dentro.any():
                achados.append(
                    f"série {serie.rotulo!r}: NENHUM dos {finitos.size} pontos cai dentro do "
                    f"limite de {eixo} [{menor}, {maior}] — o intervalo desenhado é "
                    f"[{finitos.min()}, {finitos.max()}]. A figura está em branco"
                )
            elif not dentro.all():
                achados.append(
                    f"série {serie.rotulo!r}: {int((~dentro).sum())} de {finitos.size} pontos "
                    f"caem fora do limite de {eixo} [{menor}, {maior}]"
                )
            if escala == "log" and (finitos <= 0).any():
                achados.append(
                    f"série {serie.rotulo!r}: escala de {eixo} é logarítmica e "
                    f"{int((finitos <= 0).sum())} pontos são <= 0 — o matplotlib recorta-os em "
                    "silêncio"
                )

    for artista, visivel, alpha in zip(desenhadas, lida.visiveis, lida.alphas, strict=True):
        if not visivel:
            achados.append(f"série {artista.rotulo!r} está desenhada e INVISÍVEL")
        elif alpha < alpha_minimo:
            achados.append(
                f"série {artista.rotulo!r} tem alpha {alpha}, abaixo de {alpha_minimo} — "
                "indistinguível de ausente"
            )
    return achados


# ---------------------------------------------------------------------------
# (d) o número que a figura declara
# ---------------------------------------------------------------------------


def numeros_declarados(figura: Figure) -> tuple[str, ...]:
    """Todo número que aparece em texto na figura: rodapé, título e legenda.

    Devolve **texto** e não `float` de propósito: `6.035` é seis mil e trinta e cinco no rodapé em
    português e seria 6,035 lido como `float`. Comparar a forma como ela foi escrita é o que o
    leitor faz, e é o que se quer conferir.
    """
    fontes: list[str] = [t.get_text() for t in figura.texts]
    for ax in figura.axes:
        fontes.append(ax.get_title())
        legenda = ax.get_legend()
        if legenda is not None:
            fontes.extend(t.get_text() for t in legenda.get_texts())
    achados: list[str] = []
    for texto in fontes:
        achados.extend(NUMERO.findall(texto))
    return tuple(achados)


def confere_numeros(figura: Figure, esperados: Sequence[str]) -> list[str]:
    """Acusa número que a figura declara e o teste não derivou, e o inverso.

    `esperados` vem do **teste**, derivado da entrada por caminho independente do construtor — é a
    metade que faz a classe (d) valer alguma coisa. Se o teste chamasse o mesmo `Recorte` que o
    construtor usou, os dois erravam juntos.
    """
    obtidos = numeros_declarados(figura)
    achados: list[str] = []
    faltam = [n for n in esperados if n not in obtidos]
    if faltam:
        achados.append(f"números que o teste derivou e a figura NÃO declara: {faltam}")
    sobram = [n for n in obtidos if n not in esperados]
    if sobram:
        achados.append(f"números que a figura declara e o teste não derivou: {sobram}")
    return achados


# ---------------------------------------------------------------------------
# As quatro plantas, cada uma com a âncora de separação
# ---------------------------------------------------------------------------


def planta_serie_removida(ax: Axes) -> str:
    """Apaga a última série desenhada. Devolve o rótulo do que foi apagado.

    **Âncora:** tem de haver ao menos duas séries. Apagar a única deixa eixos vazios, e eixos vazios
    reprovam por meia dúzia de razões — a atribuição da reprovação deixaria de ser a planta.
    """
    desenhadas = series(ax)
    if len(desenhadas) < 2:
        raise PlantaQueNaoSeparaError(
            f"a planta da série removida precisa de ao menos duas séries, e há {len(desenhadas)}. "
            "Com uma só, a reprovação não é atribuível a esta planta"
        )
    # A ordem é a inversa da de `series` — imagens, coleções, linhas —, porque *"a última série
    # desenhada"* é a última que aquela função devolve. Sem esta ordem, uma figura só de imagens
    # caía em `ax.lines[-1]` e levantava `IndexError`, que é uma reprovação não atribuível à planta.
    if ax.images:
        artista: Artist = ax.images[-1]
    elif ax.collections:
        artista = ax.collections[-1]
    else:
        artista = ax.lines[-1]
    rotulo = str(artista.get_label())
    artista.remove()
    return rotulo


def planta_eixos_trocados(ax: Axes) -> str:
    """Troca `x` por `y` na primeira série. Devolve o rótulo da série mutilada.

    **Âncora:** `x` e `y` têm de diferir. Numa série sobre a diagonal a troca é a identidade, e o
    teste passaria — com a planta aplicada.
    """
    desenhadas = series(ax)
    if not desenhadas:
        raise PlantaQueNaoSeparaError("não há série nenhuma para trocar os eixos")
    alvo = desenhadas[0]
    if alvo.x.shape == alvo.y.shape and np.allclose(alvo.x, alvo.y, atol=TOLERANCIA):
        raise PlantaQueNaoSeparaError(
            f"a série {alvo.rotulo!r} tem x == y: trocar os eixos é a identidade, e a planta "
            "não separa nada"
        )
    if ax.lines:
        linha = ax.lines[0]
        x, y = np.asarray(linha.get_xdata()), np.asarray(linha.get_ydata())
        linha.set_data(y, x)
    elif ax.collections:
        colecao = ax.collections[0]
        segmentos = _vertices_dos_segmentos(colecao)
        if segmentos is None:
            pontos = np.asarray(colecao.get_offsets(), dtype=float)
            colecao.set_offsets(pontos[:, ::-1])
        else:
            colecao.set_segments(  # type: ignore[attr-defined]
                [np.asarray(s, dtype=float)[:, ::-1] for s in colecao.get_segments()]  # type: ignore[attr-defined]
            )
    else:
        raise PlantaQueNaoSeparaError(
            "o primeiro artista destes eixos é uma imagem: trocar os eixos dela é transpor o "
            "conteúdo, que é outra mutação. Use `planta_conteudo_trocado`"
        )
    return alvo.rotulo


def planta_sinal_invertido(ax: Axes) -> str:
    """Multiplica o `y` da primeira série por -1. Devolve o rótulo da série mutilada.

    **Âncora:** algum `y` tem de ser diferente de zero. Inverter o sinal de uma série toda a zeros é
    a identidade — e é exactamente o caso de uma figura de resíduo bem comportado.
    """
    desenhadas = series(ax)
    if not desenhadas:
        raise PlantaQueNaoSeparaError("não há série nenhuma para inverter o sinal")
    alvo = desenhadas[0]
    finitos = alvo.y[np.isfinite(alvo.y)]
    if finitos.size == 0 or bool(np.allclose(finitos, 0.0, atol=TOLERANCIA)):
        raise PlantaQueNaoSeparaError(
            f"o `y` da série {alvo.rotulo!r} é todo zero: inverter o sinal é a identidade"
        )
    if ax.lines:
        linha = ax.lines[0]
        linha.set_ydata(-np.asarray(linha.get_ydata(), dtype=float))
    elif ax.collections:
        colecao = ax.collections[0]
        if _vertices_dos_segmentos(colecao) is None:
            pontos = np.asarray(colecao.get_offsets(), dtype=float)
            colecao.set_offsets(np.column_stack((pontos[:, 0], -pontos[:, 1])))
        else:
            invertidos = []
            for segmento in colecao.get_segments():  # type: ignore[attr-defined]
                pares = np.asarray(segmento, dtype=float).copy()
                pares[:, 1] = -pares[:, 1]
                invertidos.append(pares)
            colecao.set_segments(invertidos)  # type: ignore[attr-defined]
    else:
        raise PlantaQueNaoSeparaError(
            "o primeiro artista destes eixos é uma imagem: ela não tem `y` por ponto para inverter"
        )
    return alvo.rotulo


def planta_conteudo_trocado(ax: Axes) -> str:
    """Troca o CONTEÚDO da primeira imagem, deixando o retângulo intacto. Devolve o rótulo dela.

    **É a quinta direção do par causal, e a que as quatro anteriores não alcançam.** As três
    mutações de dado mexem em coordenadas e a do corte mexe na vista; nenhuma toca a **matriz** de
    um `AxesImage`. Sem `ax.images` esta planta não separava nada — `series(ax)` devolvia `()` para
    uma figura de `imshow`, com o conteúdo ou com outro qualquer.

    **Âncora:** o conteúdo plantado tem de diferir do desenhado. A mutação é uma reflexão nos
    dois eixos somada a um deslocamento, então ela só é a identidade num conteúdo que seja ao mesmo
    tempo simétrico e nulo — e nesse caso a planta recusa-se em vez de passar por vacuidade.
    """
    if not ax.images:
        raise PlantaQueNaoSeparaError("não há imagem nenhuma nestes eixos para trocar o conteúdo")
    imagem = ax.images[0]
    original = np.asarray(imagem.get_array(), dtype=float)
    trocado = original[::-1, ::-1] + 1.0
    if np.allclose(original, trocado, atol=TOLERANCIA, equal_nan=True):
        raise PlantaQueNaoSeparaError(
            "o conteúdo trocado é igual ao desenhado: a planta não separa nada"
        )
    imagem.set_data(trocado)
    return str(imagem.get_label())


def planta_corte_de_eixo(ax: Axes) -> tuple[float, float]:
    """Põe o `ylim` acima de todo o dado desenhado. Devolve o limite plantado.

    **Esta é a planta que as três de cima não alcançam.** Dado nenhum muda: `get_ydata` devolve o
    array certo, os rótulos ficam, a legenda fica, e a tela fica em branco.

    **Âncora:** o dado tem de ter alguma extensão em `y`. Numa série constante, um limite «acima de
    tudo» ainda pode calhar tocá-la, e a planta deixaria de separar.
    """
    desenhadas = series(ax)
    todos = (
        np.concatenate([s.y[np.isfinite(s.y)] for s in desenhadas]) if desenhadas else np.empty(0)
    )
    if todos.size == 0:
        raise PlantaQueNaoSeparaError("não há dado finito em `y` para cortar")
    extensao = float(todos.max() - todos.min())
    if extensao <= TOLERANCIA:
        raise PlantaQueNaoSeparaError(
            f"o `y` desenhado é constante em {todos.max()}: um limite «acima de tudo» não é "
            "distinguível de um limite que a contém"
        )
    limite = (float(todos.max()) + extensao, float(todos.max()) + 2.0 * extensao)
    ax.set_ylim(limite)
    return limite


# **Mudou-se para ca em 17/09/2026**, quando o arquivo de teste do painel da correcao se
# dividiu em tres: duas das partes precisam dela, e uma a importar da outra acoplaria dois assuntos
# que a divisao existe para separar. **A independencia e o conteudo dela e nao um detalhe**: viver
# aqui, no arnes, e o que a mantem longe do modulo sob teste.
def milhar_do_teste(n: int) -> str:
    """A separação de milhar do teste, DERIVADA aqui e não importada do módulo sob teste.

    Se o teste chamasse `recorte.milhar`, os dois errariam juntos — é a metade que faz a classe (d)
    valer alguma coisa.
    """
    return f"{n:,}".replace(",", ".")

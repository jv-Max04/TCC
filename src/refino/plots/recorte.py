"""Rodapé das figuras: onde ele vai e o que acontece quando não cabe.

Toda figura declara na própria imagem o recorte que aplicou (quantas features de quantas, qual
amostragem, qual matriz) e o teste ou método que usou. O mecanismo é um só para o pacote
inteiro, porque o comportamento de quebra é o que cada construtor deixaria de escrever, e num
canvas de tamanho fixo o texto que não cabe some sem aviso.

O rodapé é quebrado, nunca truncado: truncar apagaria o número que ele existe para publicar.
O texto é partido em linhas pela largura útil da figura, palavra longa demais é partida, e o
espaço vertical das linhas é reservado subindo o fundo dos eixos. Se nem quebrado ele couber
em `RESERVA_MAXIMA` da altura, `aplica` levanta `RecorteInvalidoError` em vez de desenhar por
cima dos eixos.

Reservar a faixa não basta para afastar os eixos dela. `subplots_adjust(bottom=…)` move o
fundo da caixa dos eixos, mas as marcas do eixo x e o `xlabel` são desenhados abaixo da caixa
e cairiam dentro da faixa reservada. Por isso `aplica` também chama `tight_layout(rect=…)`,
que conhece o tamanho real das marcas e do rótulo, com o `rect` a começar no topo da faixa.
O fundo dos eixos só sobe, nunca desce.

O módulo não define cor: o rodapé herda `text.color` dos `rcParams`.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from refino.plots.base import (
    GID_DO_RODAPE,
    GID_REQUEBRAVEL,
    RecusaDoConstrutorError,
    virgula_decimal,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.figure import Figure
    from matplotlib.text import Text

__all__ = [
    "COLUNAS_MINIMAS",
    "ENTRELINHA",
    "FATOR_DE_LARGURA",
    "MARGEM",
    "PASSADAS_DE_AJUSTE",
    "PASSADAS_DO_RODAPE",
    "PONTOS_POR_POLEGADA",
    "PREFIXO_DA_MATRIZ_NO_TITULO",
    "RESERVA_MAXIMA",
    "TAMANHO_DA_FONTE",
    "Recorte",
    "RecorteInvalidoError",
    "alvos_requebraveis",
    "aplica",
    "colunas_da_figura",
    "colunas_do_eixo",
    "decimal",
    "folga_do_rodape",
    "frase_contada",
    "largura_util_em_pontos",
    "legenda_fora",
    "linhas_do_rodape",
    "marcas_que_cabem",
    "milhar",
    "poe_a_matriz_no_titulo",
    "quebra",
    "reajusta",
    "requebra",
    "requebra_o_que_couber",
    "retira",
]

# Pontos tipográficos. 8 pt é o tamanho a que o rodapé se lê sem competir com os rótulos dos eixos.
TAMANHO_DA_FONTE: Final = 8.0

# Multiplicador de entrelinha do `matplotlib` para texto de várias linhas.
ENTRELINHA: Final = 1.35

# Fração da largura e da altura da figura deixada de margem à volta do rodapé.
MARGEM: Final = 0.012

# Largura média de um caractere, em múltiplos do tamanho da fonte; 0,55 é o valor para a DejaVu
# Sans, a fonte padrão do `matplotlib`. É estimativa: medir exige um renderizador. Ela serve de
# ponto de partida e de recuo para `linhas_do_rodape`, que mede a linha desenhada quando pode,
# porque a linha mais larga que sobra da quebra passa da média e a estimativa sozinha deixa o
# rodapé transbordar. Subir o valor não resolve: o glifo mais largo mede perto de 0,99 da
# fonte, e com esse fator o rodapé da figura padrão passaria de `RESERVA_MAXIMA`.
# `distribuicao.rotulos_cabem_na_altura` lê a mesma constante na direção vertical, então mudar
# o valor mudaria também aquela decisão.
FATOR_DE_LARGURA: Final = 0.55

# Piso de colunas. Uma figura minúscula não deve produzir quebra a cada palavra.
COLUNAS_MINIMAS: Final = 24

# Fração máxima da altura da figura que o rodapé pode ocupar antes de a recusa disparar.
RESERVA_MAXIMA: Final = 0.45

# Teto de passadas do ajuste entre texto e geometria em `aplica` e do encolhimento em
# `_colunas_para`. As figuras que não precisam de quebra assentam na primeira passada, e o pior
# caso conhecido, um motivo longo num eixo encolhido pela legenda de fora, na segunda.
PASSADAS_DE_AJUSTE: Final = 3

# Teto de passadas do requebrar medido do rodapé. É outro número porque governa o encolhimento
# de colunas contra uma largura fixa, em passos discretos de ao menos uma coluna. O pior caso
# medido foi de 7 passadas; doze dá folga. O laço termina de qualquer forma, porque cada
# passada tira ao menos uma coluna e o piso é `COLUNAS_MINIMAS`.
PASSADAS_DO_RODAPE: Final = 12

# Quanto da largura do eixo um texto requebrado pode ocupar. O texto marcado é posto a 2% da
# borda esquerda do eixo, e os outros 2% cobrem a diferença entre o glifo e a caixa que o
# `matplotlib` desenha à volta dele.
_FRACAO_UTIL_DO_EIXO: Final = 0.96

# Separador entre os campos do rodapé. Ponto médio e não vírgula, porque os campos levam
# vírgula ('Welch, com Benjamini-Hochberg').
SEPARADOR: Final = " · "

# Uma polegada tem 72 pontos tipográficos, a unidade em que o `matplotlib` mede tamanho de
# fonte e espessura de linha.
PONTOS_POR_POLEGADA: Final = 72.0


class RecorteInvalidoError(RecusaDoConstrutorError):
    """O recorte declarado não descreve nada que possa ser verdade.

    O rodapé é a única parte da figura que um leitor pode conferir contra outra fonte; é dele
    que sai '57 de 6.035'. Um denominador errado ali torna a figura desmentível, por isso a
    recusa vem no lugar de um rodapé com o número errado.
    """


def milhar(n: int) -> str:
    """`6035` vira `"6.035"`.

    É uma função porque a forma inline `f"{n:,} features".replace(",", ".")` troca também as
    vírgulas do texto à volta: literais adjacentes concatenam antes do acesso ao atributo.
    """
    return f"{n:,}".replace(",", ".")


def decimal(valor: float, casas: int = 4) -> str:
    """`0.7791437` vira `"0,7791"`. A vírgula é obrigatória.

    O rodapé escreve `5.942` para cinco mil novecentos e quarenta e dois. Ao lado disso, um
    `0.05` com ponto seria ambíguo na mesma frase.

    O formato é `:g` e não `:f`, para preservar a ordem de grandeza: um `p` de `2,37e-95`
    sairia como `0,0000` com `:.4f`, e 'muito pequeno' viraria 'zero'.

    É uma função pelo mesmo motivo de `milhar`: a forma inline com `.replace(".", ",")` troca
    também os pontos do texto à volta.
    """
    return f"{valor:.{casas}g}".replace(".", ",")


def frase_contada(quantas: int, *, nenhuma: str, uma: str, algumas: str) -> str:
    """`0` devolve `nenhuma`; `1` devolve `uma`; `12` devolve `"as 12 {algumas}"`.

    Várias figuras escrevem uma contagem numa frase, e o plural aritmético ('as 0 descartadas',
    'as 1 partes') lê-se como defeito de formatação e põe em dúvida o número ao lado. Esta
    função decide a ramificação e formata o milhar.

    `nenhuma` e `uma` são orações inteiras, de quem chama, porque o verbo muda com a frase
    ('ficou', 'recebeu', 'é', 'sai') e cada figura conta uma coisa diferente. `uma` é
    obrigatório, para toda frase nova decidir como concorda no singular.

    Não há guarda contra `quantas` negativo. As chamadas passam uma diferença de contagens, e
    diferença negativa significa 'desenhei mais do que existe', que `Recorte` já recusa com
    `RecorteInvalidoError`, a família que o painel apanha. Um `ValueError` aqui chegaria antes
    e iria para o `stderr` do slot do Qt. O negativo cai no ramo do plural.
    """
    if quantas == 0:
        return nenhuma
    if quantas == 1:
        return uma
    return f"as {milhar(quantas)} {algumas}"


@dataclass(frozen=True)
class Recorte:
    """O que a figura desenhou, de que população, com que amostragem, de que matriz e por que teste.

    `metodo` é obrigatório: o profissional compara esta tela com a de outra ferramenta, e um
    campo que se pode omitir fica omitido. Figura sem teste estatístico declara 'sem teste
    estatístico' por extenso.
    """

    desenhadas: int
    disponiveis: int
    amostragem: str
    matriz: str
    metodo: str

    def __post_init__(self) -> None:
        if self.disponiveis < 1:
            raise RecorteInvalidoError(
                f"`disponiveis` é {self.disponiveis}: a população de onde a figura recortou tem de "
                "ter ao menos uma feature. População vazia é `MatrizVaziaError`, e é outro fato"
            )
        if self.desenhadas < 0:
            raise RecorteInvalidoError(f"`desenhadas` é {self.desenhadas} e não pode ser negativo")
        if self.desenhadas > self.disponiveis:
            raise RecorteInvalidoError(
                f"o recorte declara {self.desenhadas} desenhadas de {self.disponiveis} "
                "disponíveis. Desenhar mais do que existe é sinal de que o numerador e o "
                "denominador vêm de populações diferentes"
            )
        for nome, valor in (
            ("amostragem", self.amostragem),
            ("matriz", self.matriz),
            ("metodo", self.metodo),
        ):
            if not valor.strip():
                raise RecorteInvalidoError(
                    f"`{nome}` está em branco. Campo em branco no rodapé lê-se como campo "
                    "irrelevante; se não há o que declarar, declare-o por extenso"
                )

    def texto(self) -> str:
        """A frase única do rodapé, com os cinco campos na ordem em que se lêem."""
        return SEPARADOR.join(
            (
                f"{milhar(self.desenhadas)} de {milhar(self.disponiveis)} features",
                f"amostragem: {self.amostragem}",
                f"matriz: {self.matriz}",
                f"método: {self.metodo}",
            )
        )


def largura_util_em_pontos(figura: Figure) -> float:
    """A largura desta figura em pontos tipográficos, já descontada a margem dos dois lados.

    O `float()` é necessário: `get_figwidth` devolve `np.float64`, e uma comparação que saia
    dele é `np.bool_`, que não é `True` sob `is`.
    """
    return float(figura.get_figwidth()) * PONTOS_POR_POLEGADA * (1.0 - 2.0 * MARGEM)


def marcas_que_cabem(comprimento_em_pontos: float, quantas: int) -> bool:
    """Se `quantas` marcas de texto cabem lado a lado num eixo deste comprimento, sem colidir.

    Sem este teste, os nomes de um eixo com muitas marcas saem uns por cima dos outros, como
    uma mancha preta. No mapa de correlação da figura padrão, o eixo y colide a partir de cerca
    de 20 features e o x a partir de cerca de 46, por isso a decisão é por eixo.

    O comprimento é argumento porque o eixo pode ser bem menor que a figura (barra de cor,
    rótulos), e a mesma conta serve ao eixo x e ao y. O limite por marca é
    `TAMANHO_DA_FONTE * ENTRELINHA`, a altura de uma linha de texto, que é o que um rótulo
    ocupa na direção em que não se estende.

    Sem marca nenhuma a resposta é `False`: marcas vazias reservariam espaço para texto que
    não existe.
    """
    if quantas < 1:
        return False
    return bool(comprimento_em_pontos / quantas >= TAMANHO_DA_FONTE * ENTRELINHA)


def colunas_da_figura(figura: Figure) -> int:
    """Quantos caracteres cabem numa linha do rodapé desta figura.

    Derivado da largura da figura e do tamanho da fonte, nunca de um literal: a mesma frase numa
    figura de 4 polegadas e noutra de 12 tem de quebrar em sítios diferentes.
    """
    por_caractere = TAMANHO_DA_FONTE * FATOR_DE_LARGURA
    return max(COLUNAS_MINIMAS, int(largura_util_em_pontos(figura) / por_caractere))


def linhas_do_rodape(figura: Figure, texto: str) -> tuple[str, ...]:
    """As linhas do rodapé desta figura: medidas quando há renderizador, estimadas quando não.

    `colunas_da_figura` divide a largura útil pela largura média de um caractere. O que
    transborda é a linha mais larga que sobra da quebra, e `textwrap` corta por contagem de
    caracteres, então essa linha leva os glifos que calharem. A diferença para a média muda a
    cada largura de figura, e nenhum fator único a corrige. Por isso a quebra parte da
    estimativa e encolhe as colunas até a linha mais larga, medida, caber.

    O renderizador já existe: `plots.base.nova_figura` liga um canvas `Agg` a toda figura. Sem
    ele (figura sem canvas) a função devolve a estimativa.
    """
    estimativa = colunas_da_figura(figura)
    linhas = quebra(texto, estimativa)
    obter = getattr(getattr(figura, "canvas", None), "get_renderer", None)
    if not callable(obter):  # pragma: no cover - `nova_figura` liga um `Agg` a toda figura
        return linhas
    disponivel = float(figura.get_figwidth()) * float(figura.dpi) * (1.0 - 2.0 * MARGEM)
    if disponivel <= 0.0:  # pragma: no cover - figura de largura nula não é desenhável
        return linhas

    desenhador = obter()
    # A sonda é um `Text` da figura, com a fonte e a entrelinha do rodapé, porque é assim que
    # ele é desenhado.
    sonda = figura.text(MARGEM, MARGEM, "", fontsize=TAMANHO_DA_FONTE, linespacing=ENTRELINHA)
    try:
        colunas = estimativa
        for _ in range(PASSADAS_DO_RODAPE):
            maior = max(_largura(sonda, linha, desenhador) for linha in linhas)
            if maior <= disponivel:
                break
            if colunas <= COLUNAS_MINIMAS:
                # O piso já foi alcançado e a linha ainda não cabe. Encolher mais seria quebrar a
                # cada palavra, que é o que `COLUNAS_MINIMAS` existe para impedir.
                break
            # Regra de três sobre o medido, com decremento garantido: sem o `min`, perto da
            # borda o quociente devolve a mesma coluna e o laço gira sem encolher.
            colunas = max(COLUNAS_MINIMAS, min(colunas - 1, int(colunas * disponivel / maior)))
            linhas = quebra(texto, colunas)
    finally:
        # No `finally`, para uma exceção não deixar a sonda desenhada na figura.
        sonda.remove()
    return linhas


def colunas_do_eixo(eixo: Axes, tamanho_da_fonte: float) -> int:
    """Quantos caracteres cabem na largura deste eixo, com esta fonte.

    O rodapé tem a figura inteira, e um título de eixo tem só o eixo; numa figura de dois
    lados isso é pouco mais de um terço da largura.

    Lê `get_position()`, a posição real. Chamada depois de `tight_layout`, traz a caixa que os
    rótulos e a legenda deixaram, que é a largura em que o texto será desenhado. Antes, traria
    a de `subplotpars`, bem maior quando a legenda fica fora da área de dado.

    O tamanho da fonte é argumento porque os títulos do pacote não têm todos o mesmo tamanho.
    """
    figura = eixo.get_figure()
    if figura is None:  # pragma: no cover - um `Axes` construído sempre pertence a uma figura
        return COLUNAS_MINIMAS
    # Via `bbox` e `dpi`, e não `get_figwidth()`: um `Axes` pode viver numa `SubFigure`, que
    # não tem esse método. Para uma `Figure` comum o valor é o mesmo.
    largura_util = eixo.get_position().width * figura.bbox.width * PONTOS_POR_POLEGADA / figura.dpi
    return max(COLUNAS_MINIMAS, int(largura_util / (tamanho_da_fonte * FATOR_DE_LARGURA)))


def requebra(texto: str, colunas: int) -> str:
    """Requebra um texto de várias linhas, cada linha por si, e devolve as linhas reunidas.

    Os títulos do pacote têm duas linhas com papéis diferentes, o nome da figura e a contagem
    da população. Refluí-las como parágrafo único juntaria o fim de uma ao começo da outra.

    É idempotente sobre texto que já cabe, o que permite correr a função sobre todos os
    títulos.
    """
    return "\n".join("\n".join(quebra(linha, colunas)) for linha in texto.split("\n"))


def alvos_requebraveis(figura: Figure) -> list[tuple[Axes, Text]]:
    """Os textos que `aplica` pode requebrar: o título de cada eixo e o que trouxer a marca.

    O título descreve o eixo, e requebrá-lo nunca é errado. Um `Text` dentro da área de dado
    pode ser qualquer coisa (a anotação de um clique, um rótulo sobre um ponto), então só é
    requebrado quando leva `GID_REQUEBRAVEL`.
    """
    achados: list[tuple[Axes, Text]] = []
    for eixo in figura.axes:
        achados.append((eixo, eixo.title))
        achados += [(eixo, t) for t in eixo.texts if t.get_gid() == GID_REQUEBRAVEL]
    return achados


def requebra_o_que_couber(figura: Figure, originais: Mapping[Text, str] | None = None) -> int:
    """Requebra cada alvo pela largura real do eixo dele. Devolve quantos mudaram.

    Corre depois do `tight_layout`: antes dele os eixos ainda têm a largura de origem, que não
    é a largura em que o texto será desenhado.

    `originais` traz o texto de antes de qualquer quebra. `requebra` quebra linha a linha e
    nunca volta a juntar, então sem ele uma segunda passada mais estreita partiria as linhas
    que a primeira deixou, alternando linhas compridas e curtas. Com ele, cada passada quebra
    a partir da frase inteira.

    A contagem devolvida permite a um teste verificar que o caminho correu.
    """
    tocados = 0
    colunas_de_eixos = _colunas_de_eixos(figura)
    for eixo, alvo in alvos_requebraveis(figura):
        atual = alvo.get_text()
        fonte = atual if originais is None else originais.get(alvo, atual)
        if not fonte.strip():
            continue
        novo = requebra(fonte, _colunas_para(eixo, alvo, fonte, colunas_de_eixos))
        if novo != atual:
            alvo.set_text(novo)
            tocados += 1
    return tocados


def _colunas_de_eixos(figura: Figure) -> int:
    """Quantas colunas de eixos esta figura tem: 1 para um gráfico, 2 para um antes e depois.

    Contadas pelos centros horizontais distintos, arredondados: dois eixos empilhados são uma
    coluna, e dois lado a lado são duas. É o denominador da largura que um título tem.
    """
    centros = {
        round(float(eixo.get_position().x0 + eixo.get_position().x1) / 2.0, 2)
        for eixo in figura.axes
    }
    return max(1, len(centros))


def _colunas_para(eixo: Axes, alvo: Text, texto: str, colunas_de_eixos: int) -> int:
    """As colunas deste texto neste eixo: medidas quando há renderizador, estimadas quando não.

    A estimativa por largura média de caractere erra para prosa com glifos largos, e a linha
    entraria na legenda. Como isto corre dentro de `aplica`, sobre uma figura que já tem
    renderizador, a largura do texto é medida.

    Sem renderizador (figura sem canvas) devolve a estimativa.
    """
    estimativa = colunas_do_eixo(eixo, float(alvo.get_fontsize()))
    figura = eixo.get_figure()
    obter = getattr(getattr(figura, "canvas", None), "get_renderer", None)
    if not callable(obter):  # pragma: no cover - todo canvas deste pacote é `Agg` ou `QtAgg`
        return estimativa
    desenhador = obter()
    disponivel = _disponivel(eixo, alvo, colunas_de_eixos, desenhador) * _FRACAO_UTIL_DO_EIXO
    if disponivel <= 0.0:  # pragma: no cover - eixo de largura nula não é desenhável
        return estimativa

    # Mede o texto já quebrado e não a largura média por caractere: `textwrap` corta por
    # contagem de caracteres, e a linha mais larga do resultado passa da média.
    guardado = alvo.get_text()
    try:
        # O ponto de partida sai da largura disponível, não de `colunas_do_eixo`. O laço só
        # encolhe, e com a estimativa do eixo um título nunca usaria a largura que tem: numa
        # figura de legenda fora o eixo é bem mais estreito que o espaço do título.
        colunas = _semente(alvo, texto, disponivel, desenhador)
        for _ in range(PASSADAS_DE_AJUSTE):
            # Mede o bloco montado, não o máximo das linhas soltas: um `Text` de várias linhas
            # é desenhado com alinhamento múltiplo, e a caixa dele pode ser mais larga.
            largura = _largura(alvo, requebra(texto, colunas), desenhador)
            if largura <= disponivel:
                break
            # Encolhe na proporção do excesso; uma passada quase sempre basta.
            colunas = max(COLUNAS_MINIMAS, int(colunas * disponivel / largura))
    finally:
        # No `finally`, para uma exceção não deixar a linha de sonda no lugar do título.
        alvo.set_text(guardado)
    return colunas


def _semente(alvo: Text, texto: str, disponivel: float, desenhador: RendererBase) -> int:
    """O palpite inicial de colunas, por regra de três sobre a linha mais comprida.

    Mede a linha inteira e escala pela razão entre o que cabe e o que ela mede. Se ela já
    cabe, a semente é o comprimento dela, e o laço de fora sai sem quebrar nada.
    """
    mais_longa = max(texto.split("\n"), key=len)
    largura = _largura(alvo, mais_longa, desenhador)
    if largura <= disponivel or largura <= 0.0:
        return max(COLUNAS_MINIMAS, len(mais_longa))
    return max(COLUNAS_MINIMAS, int(len(mais_longa) * disponivel / largura))


def _disponivel(eixo: Axes, alvo: Text, colunas_de_eixos: int, desenhador: RendererBase) -> float:
    """A largura em píxeis que este alvo pode ocupar.

    Um título é desenhado centrado sobre o eixo e pode passar por cima do espaço branco à
    volta dele; o que o limita é o eixo vizinho ou a borda da figura, então ele dispõe da
    coluna da figura. Um texto marcado vive dentro da área de dado e dispõe só do eixo, porque
    sair dele é entrar na legenda.

    Medir o título contra o eixo o quebraria demais numa figura de legenda fora: o título
    ficaria mais alto, o `tight_layout` devolveria largura ao eixo, e a passada seguinte
    desfaria a quebra, deslocando a geometria.
    """
    if alvo is eixo.title:
        dona = eixo.get_figure()
        largura_da_figura = 0.0 if dona is None else float(dona.bbox.width)
        return largura_da_figura / colunas_de_eixos
    return float(eixo.get_window_extent(desenhador).width)


def _largura(alvo: Text, linha: str, desenhador: RendererBase) -> float:
    """A largura em píxeis desta linha, desenhada com as propriedades deste artista."""
    alvo.set_text(linha)
    return float(alvo.get_window_extent(desenhador).width)


def quebra(texto: str, colunas: int) -> tuple[str, ...]:
    """Quebra em linhas de até `colunas` caracteres. Nunca devolve tupla vazia.

    `break_long_words=True` porque partir uma palavra longa é preferível a deixá-la sair pela
    borda. `break_on_hyphens=False` porque há rótulos com hífen dentro do identificador, e
    parti-los ali daria duas metades que se leem como dois rótulos.
    """
    linhas = textwrap.wrap(
        texto,
        width=colunas,
        break_long_words=True,
        break_on_hyphens=False,
    )
    return tuple(linhas) if linhas else ("",)


def legenda_fora(eixo: Axes, maos: list[Artist] | None = None) -> None:
    """Põe a legenda fora da área de dado, encostada à direita.

    Com `loc="best"` o `matplotlib` escolhe o canto de menor sobreposição, e num gráfico de
    scores em que a nuvem ocupa o meio e a elipse o resto, o melhor canto ainda tapa pontos.
    Fora do eixo a legenda não tapa nada, e o `tight_layout` que `aplica` chama a seguir
    encolhe os eixos para dar lugar a ela.

    Sem `maos`, a legenda colhe os artistas rotulados do eixo; com `maos`, usa exatamente esses.
    """
    if maos is None:
        eixo.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            fontsize=TAMANHO_DA_FONTE,
        )
        return
    eixo.legend(
        handles=maos,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=TAMANHO_DA_FONTE,
    )


# A matriz também vai no título, que é o que se lê primeiro; o rodapé a traz em letra pequena
# e por último. A linha usa o campo `matriz` do `Recorte`, então os dois não podem discordar.
PREFIXO_DA_MATRIZ_NO_TITULO: Final = "Matriz: "


def poe_a_matriz_no_titulo(figura: Figure, recorte: Recorte) -> Text | None:
    """Acrescenta 'Matriz: …' ao título da figura e devolve o artista que a recebeu.

    Usa o título geral, quando há um, e senão o do primeiro eixo. O do eixo é requebrado por
    `aplica` quando não cabe; o geral não é, por isso só recebe a linha quando já existe. É
    idempotente. Figura sem título e sem eixo devolve `None`.
    """
    linha = f"{PREFIXO_DA_MATRIZ_NO_TITULO}{recorte.matriz}"
    # `_suptitle` é privado no `matplotlib` 3.11, e é o único caminho até o artista: `get_suptitle`
    # devolve só o texto, e reescrevê-lo por `suptitle()` repõe as propriedades padrão.
    geral: Text | None = getattr(figura, "_suptitle", None)
    if geral is not None and geral.get_text():
        alvo = geral
    elif figura.axes:
        alvo = figura.axes[0].title
    else:
        return None
    texto = alvo.get_text()
    if linha not in texto:
        alvo.set_text(f"{texto}\n{linha}" if texto else linha)
    return alvo


def aplica(figura: Figure, recorte: Recorte) -> Text:
    """Põe o rodapé na figura, reserva o espaço dele, afasta os eixos e devolve o artista.

    Devolve o `Text` para que quem chama, e os testes, leiam o que ficou desenhado em vez de
    reconstruir a frase.

    Faz também o que vale para toda figura que a chama: troca o ponto decimal pela vírgula nas
    escalas, põe a matriz no título e requebra os títulos e textos marcados que não cabem. Os
    construtores a chamam uma vez, no fim, com o dado e os rótulos já postos.

    O `tight_layout` corre ao menos duas vezes quando há quebra. A primeira fixa a largura real
    dos eixos, contra a qual o título é quebrado; a seguinte dá ao título a altura das linhas
    novas. Numa passada só, o título de duas linhas teria a reserva vertical de uma e sairia
    pelo topo da figura.
    """
    # Vem antes de qualquer medição, para a figura medida ser a que será desenhada.
    virgula_decimal(figura)
    poe_a_matriz_no_titulo(figura, recorte)

    linhas = linhas_do_rodape(figura, recorte.texto())

    altura_em_pontos = len(linhas) * TAMANHO_DA_FONTE * ENTRELINHA
    altura_da_figura = figura.get_figheight() * PONTOS_POR_POLEGADA
    reservado = altura_em_pontos / altura_da_figura + 2.0 * MARGEM

    if reservado > RESERVA_MAXIMA:
        raise RecorteInvalidoError(
            f"o rodapé quebrou em {len(linhas)} linhas e precisaria de "
            f"{reservado:.0%} da altura da figura, acima do teto de {RESERVA_MAXIMA:.0%}. "
            "Ele não é truncado — encurte os campos do `Recorte` ou dê mais altura à figura"
        )

    # A reserva depende da altura da figura em polegadas neste instante. Quem embute a figura
    # num widget muda essa altura sem passar por aqui, e `reajusta` refaz a conta no tamanho
    # novo. Para isso ficam guardados o texto de antes da quebra e o fundo que o construtor
    # tinha antes de esta função o subir.
    figura._recorte_pristino = recorte.texto()  # type: ignore[attr-defined]
    figura._recorte_base = figura.subplotpars.bottom  # type: ignore[attr-defined]

    # Só sobe, nunca desce: um construtor que já tenha reservado espaço embaixo, por um rótulo
    # de eixo rodado, por exemplo, não o perde por chamar este auxiliar.
    figura.subplots_adjust(bottom=max(figura.subplotpars.bottom, reservado + MARGEM))

    # As marcas e o rótulo do eixo x sobem com a caixa: o `rect` começa no fundo fixado acima,
    # e `tight_layout` sabe quanto as marcas ocupam. A guarda de `figura.axes` existe porque
    # `tight_layout` levanta numa figura sem eixos, e rodapé em figura sem eixos é legítimo.
    if figura.axes:
        # A largura do eixo depende do texto e o texto depende da largura do eixo. Além disso,
        # quando um texto é largo demais o `tight_layout` desiste e deixa a geometria de
        # origem, então a primeira medição de largura sai a de antes do ajuste. O laço repete
        # até a largura parar de mudar, com teto. Cada passada quebra a partir do texto
        # original, não do que a anterior deixou, para a última ganhar em vez de acumular
        # quebras.
        originais = {alvo: alvo.get_text() for _eixo, alvo in alvos_requebraveis(figura)}
        # O fundo do `rect` é fixado antes do laço. `tight_layout` escreve em
        # `subplotpars.bottom`; lido dentro do laço, o fundo subiria a cada passada, até a
        # área de dado quase desaparecer sem aviso nenhum.
        fundo_reservado = figura.subplotpars.bottom
        for _ in range(PASSADAS_DE_AJUSTE):
            figura.tight_layout(rect=(0.0, fundo_reservado, 1.0, 1.0))
            if not requebra_o_que_couber(figura, originais):
                break

    # O `gid` separa este `Text` do título de figura. Quem embute a figura num widget refaz a
    # faixa a cada redimensionamento e procura o rodapé em `figura.texts`, que traz também o
    # `_suptitle`.
    return figura.text(
        MARGEM,
        MARGEM,
        "\n".join(linhas),
        ha="left",
        va="bottom",
        fontsize=TAMANHO_DA_FONTE,
        linespacing=ENTRELINHA,
        gid=GID_DO_RODAPE,
    )


def folga_do_rodape(figura: Figure) -> float | None:
    """A distância, em pontos, entre o topo do rodapé e o mais baixo dos eixos.

    Positiva é folga; zero ou negativa é colisão. Devolve `None` quando a figura não tem
    rodapé desta família ou não tem eixos, porque `0.0` diria que encostou.

    A caixa do eixo é a apertada (`get_tightbbox`), com os rótulos das marcas e do eixo: o
    rodapé que cobre o rótulo do eixo x também colide. O valor sai em pontos para não depender
    do DPI.

    Não pinta nada: `draw_without_rendering` faz o leiaute sem tocar o canvas.
    """
    rodape = next((t for t in figura.texts if t.get_gid() == GID_DO_RODAPE), None)
    eixos = [eixo for eixo in figura.axes if eixo.get_visible()]
    if rodape is None or not eixos:
        return None
    figura.draw_without_rendering()
    desenhador = figura.canvas.get_renderer()  # type: ignore[attr-defined]
    topo = float(rodape.get_window_extent(desenhador).y1)
    caixas = [eixo.get_tightbbox(desenhador) for eixo in eixos]
    fundo = min(float(caixa.y0) for caixa in caixas if caixa is not None)
    return (fundo - topo) * PONTOS_POR_POLEGADA / figura.dpi


def reajusta(figura: Figure, rodape: Text) -> bool:
    """Refaz a faixa do rodapé no tamanho atual da figura. Devolve se mexeu em alguma coisa.

    `aplica` calcula a faixa a partir da altura da figura em polegadas no instante da chamada.
    Quem embute a figura num widget muda essa altura depois. O texto é medido em pontos, que
    são absolutos, e a faixa é fração da altura; numa figura menor o texto ocupa
    proporcionalmente mais, até invadir o eixo.

    As duas grandezas que mudam com o tamanho são refeitas a partir do que `aplica` guardou:

    - a quebra, a partir do texto de antes de qualquer quebra, para não acumular quebras;
    - a fração reservada, na altura atual, contra o fundo que o construtor tinha antes de
      `aplica` o subir, para a faixa não crescer a cada redimensionamento.

    É uma função separada porque `aplica` acrescenta um `Text` a cada chamada. Esta recebe o
    artista que aquela devolveu e não cria nada.

    Não recusa. Em `aplica` a recusa acontece na construção, quando quem monta a frase pode
    encurtá-la; aqui o tamanho é do utilizador, e quando a faixa passaria de `RESERVA_MAXIMA`
    ela fica no teto. Também não desenha: quem chama redesenha.
    """
    pristino = getattr(figura, "_recorte_pristino", None)
    if pristino is None:
        # Figura que não passou por `aplica` não tem rodapé desta família, e reajustar a caixa dela
        # seria mexer em geometria de que esta função não sabe nada.
        return False

    linhas = linhas_do_rodape(figura, pristino)
    altura_em_pontos = len(linhas) * TAMANHO_DA_FONTE * ENTRELINHA
    altura_da_figura = figura.get_figheight() * PONTOS_POR_POLEGADA
    reservado = min(altura_em_pontos / altura_da_figura + 2.0 * MARGEM, RESERVA_MAXIMA)

    base = getattr(figura, "_recorte_base", 0.0)
    fundo = max(base, reservado + MARGEM)

    texto_novo = "\n".join(linhas)
    mexeu = texto_novo != rodape.get_text() or abs(figura.subplotpars.bottom - fundo) > 1e-9
    if not mexeu:
        return False

    rodape.set_text(texto_novo)
    figura.subplots_adjust(bottom=fundo)
    if figura.axes:
        # Uma passada, sem o laço de requebra. Requebrar o título e os textos marcados na
        # largura do widget refluiria a figura toda (título em mais linhas, área de dado
        # espremida, legenda por cima dos pontos), e isso é decisão do construtor. Com o
        # redimensionamento só muda a faixa do rodapé.
        figura.tight_layout(rect=(0.0, figura.subplotpars.bottom, 1.0, 1.0))
    return True


def retira(figura: Figure, rodape: Text) -> str | None:
    """Tira o rodapé da figura e devolve a faixa dele ao gráfico. Devolve o texto do rodapé.

    Devolve `None`, sem mudar nada, quando a figura não passou por `aplica`.

    Serve à exportação em tamanhos pequenos, como a coluna simples de revista (84 a 90 mm), em
    que o rodapé de 8 pt não cabe sem encostar no gráfico. A figura sai sem ele e o texto vai
    para um arquivo ao lado, que a janela escreve; aqui fica só a geometria.

    O texto devolvido é o de antes da quebra. O desenhado está partido na largura da figura, e
    juntar as linhas não o reconstrói, porque `quebra` parte palavras longas.

    O `tight_layout` corre sobre a figura inteira, com o retângulo a começar no zero. Só apagar
    o texto manteria o leiaute com a faixa reservada, que nesses tamanhos não se aplica e
    deixa o gráfico sair pelas bordas.

    O texto fica vazio, e não invisível. `reajusta` refaz o rodapé quando o texto desenhado
    difere do que ele quebraria, e um texto vazio difere sempre: para repor o rodapé basta
    chamar `reajusta`.
    """
    pristino = getattr(figura, "_recorte_pristino", None)
    if pristino is None:
        return None
    rodape.set_text("")
    if figura.axes:
        figura.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    return str(pristino)


# Fica aqui, ao lado de `decimal`, porque o violino e a composição por classe a usam.
def em_percentagem(fracao: float) -> str:
    """`0.1399` vira `"13,99%"`.

    Não é `percentagem_do_r`, que replica `signif(v, 4) * 100` de `tools.R:1554` (quatro
    algarismos da fração e só depois a percentagem) para dar a mesma cadeia que o PNG do
    laboratório. Esta usa `decimal`, a formatação de número do pacote.
    """
    return f"{decimal(fracao * 100.0)}%"

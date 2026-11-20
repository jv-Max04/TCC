"""A criação da figura, sem `pyplot`, e a recusa tipada para população vazia."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import Formatter, ScalarFormatter

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = [
    "ALTURA_PADRAO",
    "GID_DAS_FEATURES",
    "GID_DO_RODAPE",
    "GID_REQUEBRAVEL",
    "LARGURA_PADRAO",
    "NAO_DECLARADA",
    "PALETA_DIVERGENTE",
    "FeaturesDemaisError",
    "MatrizVaziaError",
    "RecusaDoConstrutorError",
    "cores_do_ciclo",
    "denominador_ou_recusa",
    "exige_abaixo_do_teto",
    "exige_populacao",
    "inteiro_do_fio",
    "nova_figura",
    "numero_do_fio",
    "objetos_do_fio",
    "texto_do_fio",
    "vetor_do_fio",
    "vetor_do_fio_ou_nucleo",
    "virgula_decimal",
]

# O texto usado quando o corpo JSON não diz de que matriz um eixo saiu. Não é um nome de
# matriz: escrever `ingerida` ou `processada` aqui faria a figura afirmar uma procedência que
# ninguém declarou.
NAO_DECLARADA: Final = "matriz não declarada"

PALETA_DIVERGENTE: Final = "RdBu_r"
"""A paleta da escala de uma matriz-imagem (mapa de calor, matriz de correlação).

É divergente porque as quantidades desenhadas são centradas em zero: o z-score por feature
e o coeficiente de correlação. Numa paleta sequencial o zero cairia num ponto qualquer da
rampa; na divergente ele é o tom neutro, e as duas direções do desvio se leem como direções.
Vermelho-azul, e não verde-vermelho, porque a segunda não se distingue na forma mais
frequente de daltonismo.

É escrita aqui, e não herdada dos `rcParams`, porque codifica o dado e não o tema:
`image.cmap` é um nome só, sem a noção de divergente, e herdá-lo faria a leitura da figura
mudar com a configuração.
"""

# Os dois textos de ausência abaixo são lidos pelo painel da correção e pelo do violino.

# O texto do eixo direito quando o corpo não declara o corretor. Não pode ser um nome do
# vocabulário: `none` é corretor legítimo, e usá-lo aqui faria a figura afirmar que a execução
# correu sem correção.
CORRETOR_NAO_DECLARADO: Final = "corretor não declarado"

# O texto dos dois rótulos que tornam um RSD comparável, `rsd_estimator` e `rsd_variant`,
# quando faltam. Não é `NAO_DECLARADA`, que fala da matriz e não da régua.
REGUA_AUSENTE: Final = "ausente do corpo"

# O vocabulário fechado de `gid`. É como este pacote declara, a quem consome a figura, a
# intenção de um artista: `recorte.aplica` decide por ele o que pode requebrar, e a janela
# decide de qual artista é o nome que o cursor devolve. Sem a marca, as duas camadas teriam de
# inferir por contagem de pontos ou por ordem de desenho. Um `gid` desconhecido é tratado como
# artista não marcado.

GID_DAS_FEATURES: Final = "refino:features"
"""Os pontos deste artista são indexados pelo vetor de nomes que o painel da janela recebe.

Os artistas marcados desenham uma população filtrada (só os pontos desenháveis), e o vetor
de nomes tem de ser filtrado do mesmo modo por quem o passa. Passar a população inteira a um
artista filtrado devolve o nome do vizinho, com a figura bem formada e o rótulo errado.

A marca também diz de qual artista vale a identificação: numa figura de cargas, um clique
na cruz da origem não pode devolver o nome de uma feature.
"""

GID_REQUEBRAVEL: Final = "refino:requebrável"
"""Este `Text` é prosa, e `recorte.aplica` pode requebrá-lo pela largura final do eixo.

Quem o desenha não sabe que largura o eixo vai ter: `tight_layout` só corre no fim, e uma
legenda fora da área de dado encolhe o eixo depois de o texto já estar posto.
"""

GID_DO_RODAPE: Final = "refino:rodapé"
"""Este `Text` é o rodapé do recorte, e `recorte.reajusta` pode refazer a faixa dele.

Quem embute a figura refaz a faixa a cada redimensionamento, percorrendo `figura.texts`. Essa
lista traz também o título de figura, e sem a marca o reajuste cairia sobre ele. A marca
evita depender de `figura._suptitle`, atributo privado do `matplotlib`.
"""


class _VirgulaDecimal(Formatter):
    """Envolve o formatador de um eixo e troca o ponto decimal pela vírgula.

    Envolve em vez de substituir porque o `ScalarFormatter` original recalcula a cada desenho
    o número de casas, o expoente comum e o deslocamento. Por isso `set_locs`, `set_axis` e
    `get_offset` são reencaminhados: o interno decide o que se escreve, e este só troca o
    separador.
    """

    def __init__(self, interno: Formatter) -> None:
        super().__init__()
        self._interno = interno

    def __call__(self, x: float, pos: int | None = None) -> str:
        return str(self._interno(x, pos)).replace(".", ",")

    def set_locs(self, locs: Any) -> None:
        self._interno.set_locs(locs)

    def set_axis(self, axis: Any) -> None:
        """Reencaminha o eixo ao formatador interno.

        O parâmetro é `Any` porque a união aceita por `TickHelper.set_axis` inclui tipos
        privados do `matplotlib`, e estreitar para `Axis` violaria a assinatura do supertipo.
        """
        super().set_axis(axis)
        self._interno.set_axis(axis)

    def get_offset(self) -> str:
        """O expoente comum e o deslocamento que o eixo imprime ao canto, com a mesma troca."""
        return str(self._interno.get_offset()).replace(".", ",")


def virgula_decimal(figura: Figure) -> int:
    """Põe a vírgula decimal nas marcas de toda escala numérica da figura. Devolve quantas trocou.

    O rodapé usa o ponto como separador de milhar (`6.035`), então um eixo com `0.005` usaria o
    mesmo símbolo com o sentido oposto na mesma imagem.

    Só troca `ScalarFormatter`. Os outros formatadores carregam texto que não é número
    escolhido pelo eixo: nomes de feature como `21.37_508.2876m/z` no VIP, nomes de amostra
    nas caixas. Trocar o ponto ali fabricaria um identificador que não existe.

    O sinal de menos das marcas negativas é U+2212 e não é afetado.
    """
    trocados = 0
    for eixo in figura.axes:
        for escala in (eixo.xaxis, eixo.yaxis):
            for maior in (True, False):
                atual = escala.get_major_formatter() if maior else escala.get_minor_formatter()
                if not isinstance(atual, ScalarFormatter):
                    continue
                envolvido = _VirgulaDecimal(atual)
                if maior:
                    escala.set_major_formatter(envolvido)
                else:
                    escala.set_minor_formatter(envolvido)
                trocados += 1
    return trocados


# Polegadas. Decidem a proporção e o tamanho inicial no `Agg`. O tamanho na tela é de quem
# embute a figura, que a redimensiona a cada troca.
LARGURA_PADRAO: Final = 7.0
ALTURA_PADRAO: Final = 4.5


class RecusaDoConstrutorError(ValueError):
    """A base de toda recusa deste pacote.

    As figuras são construídas dentro de um slot do Qt, onde uma exceção não apanhada vai para
    o `stderr` e o painel fica com a figura anterior. O painel apanha esta família e mostra o
    texto da recusa no lugar da figura.

    É uma base própria, e não `except ValueError`, para não absorver o `ValueError` de um
    defeito de programação: quem levanta uma subclasse declara que o problema é do dado. O
    painel escolhe o texto pelo subtipo, então 'não veio população' continua sendo outra frase
    que 'veio população desalinhada'.
    """


class MatrizVaziaError(RecusaDoConstrutorError):
    """Um construtor recebeu população vazia.

    Levanta em vez de devolver figura em branco, que seria indistinguível de figura quebrada.
    Um vazio legítimo do dado, como zero features aprovadas no FDR, é declarado em texto pela
    figura, e não por esta recusa.
    """


class FeaturesDemaisError(RecusaDoConstrutorError):
    """A figura receberia mais features do que se consegue desenhar a tempo.

    A construção corre na thread da interface, e um redesenho de segundos aparece como o
    aplicativo travado. Recusar é melhor do que desenhar as primeiras N em silêncio, que daria
    uma imagem bem formada lida como a matriz inteira. Amostrar é decisão do chamador, e por
    isso a mensagem traz a contagem e o teto.
    """


def exige_abaixo_do_teto(quantas: int, teto: int, *, de: str) -> int:
    """Devolve `quantas`, ou levanta `FeaturesDemaisError` quando ela passa de `teto`.

    `de` é o nome da figura com o artigo (`"o mapa de calor"`) e só entra na mensagem. O teto
    é parâmetro porque depende do que cada figura desenha: fica no módulo da figura, junto
    de onde foi medido.
    """
    if quantas > teto:
        raise FeaturesDemaisError(
            f"{de} receberia {quantas} features e o teto medido é {teto}: acima dele o redesenho "
            f"passa do orçamento de interação e a janela fica parada sem nada a acusar, porque a "
            f"figura é construída na thread da interface. Reduza a população antes de chamar — "
            f"por posto, por FDR, ou pela amostragem que o recorte declara —, e a figura passa a "
            f"dizer quantas de quantas desenhou."
        )
    return quantas


def exige_populacao(nome: str, valores: Sequence[float] | np.ndarray) -> np.ndarray:
    """Converte para `ndarray` e levanta `MatrizVaziaError` se não houver o que desenhar.

    Devolve o array para que o chamador use o resultado da conversão e não converta duas vezes.
    """
    array = np.asarray(valores, dtype=float)
    if array.size == 0:
        raise MatrizVaziaError(
            f"`{nome}` está vazio: não há população para desenhar. Um construtor de `plots/` "
            "recebe a população inteira (KD2), e população vazia é defeito de quem chama — "
            "figura em branco seria indistinguível de figura quebrada"
        )
    return array


def denominador_ou_recusa(n_features_totais: Any, *, de: str) -> int:
    """O denominador do rodapé de um recorte, ou a recusa por população vazia.

    `de` é o nome do método com o artigo (`"a PCA"`, `"o PLS-DA"`) e só entra na mensagem.
    Levanta `MatrizVaziaError`, que o painel mostra como texto no lugar da figura.
    """
    totais = inteiro_do_fio(n_features_totais)
    if totais < 1:
        raise MatrizVaziaError(
            f"a figura não declara de quantas features {de} saiu: `n_features_totais` é "
            f"{n_features_totais!r}. Sem execução não há população, e figura em branco seria "
            "indistinguível de figura quebrada"
        )
    return totais


def inteiro_do_fio(valor: Any) -> int:
    """Um inteiro vindo de JSON, ou `0` quando não é número finito.

    `None` é o que chega quando o campo não veio, e `0` cai na recusa por população vazia de
    quem monta o `Recorte`. `bool` é excluído antes de `int` porque `True` é um `int` em
    Python e viraria a população `1`.
    """
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return 0
    return 0 if not np.isfinite(valor) else int(valor)


def numero_do_fio(valor: Any) -> float:
    """Um `float` vindo de JSON, ou `nan`. `bool` é excluído: `True` não é uma medida.

    Devolve `nan` em vez de descartar para o valor indefinido continuar na posição dele: o
    vetor fica alinhado com os outros que descrevem as mesmas features, e o construtor conta
    e declara os indefinidos na figura.
    """
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return float("nan")
    return float(valor)


def vetor_do_fio(valores: Any) -> np.ndarray:
    """Um vetor de `float` vindo de JSON, com o que não é número convertido em `nan`.

    O que não é lista devolve vetor vazio, e o construtor levanta `MatrizVaziaError`.
    """
    if not isinstance(valores, list):
        return np.empty(0, dtype=float)
    return np.array([numero_do_fio(x) for x in valores], dtype=float)


def vetor_do_fio_ou_nucleo(valores: Any) -> np.ndarray:
    """Um vetor de `float`, venha ele de JSON como lista ou do núcleo como `ndarray`.

    Os dois formatos passam por `vetor_do_fio`, para o `ndarray` receber a mesma guarda que a
    lista.
    """
    if isinstance(valores, np.ndarray):
        return vetor_do_fio(valores.ravel().tolist())
    if isinstance(valores, (list, tuple)):
        return vetor_do_fio(list(valores))
    return np.empty(0, dtype=float)


def objetos_do_fio(valores: Any) -> list[dict[str, Any]]:
    """Só os elementos que são objeto, de uma lista que pode não ser lista.

    O que não é lista devolve lista vazia, e o construtor levanta `MatrizVaziaError` ou
    declara a ausência em texto quando ela é um fato legítimo do estudo.
    """
    if not isinstance(valores, list):
        return []
    return [bruto for bruto in valores if isinstance(bruto, dict)]


def texto_do_fio(valor: Any, recuo: str) -> str:
    """Uma `str` não vazia vinda de JSON, ou o recuo.

    Um campo ausente, vazio ou numérico não é nome de nada, e escrevê-lo no rodapé seria
    procedência inventada. O texto do recuo é do chamador: `NAO_DECLARADA` para a matriz, outra
    frase para outro campo.
    """
    return valor if isinstance(valor, str) and valor.strip() else recuo


def cores_do_ciclo(quantas: int) -> tuple[Any, ...]:
    """`quantas` cores tiradas do ciclo dos `rcParams`.

    O ciclo é lido no momento da construção, e assim um tema configurado antes chega aqui sem
    este pacote conhecer temas.

    O tipo é `Any` porque o ciclo pode trazer uma cadeia como `'#1f77b4'` ou uma tupla RGB,
    conforme o estilo vigente. Converter com `str` produziria um valor que o `matplotlib`
    recusa.

    Com o ciclo vazio todas as posições recebem `lines.color`. As séries deixam de se
    distinguir pela cor, e por isso toda legenda leva texto.
    """
    if quantas < 0:
        raise ValueError(f"`quantas` não pode ser negativo; recebi {quantas}")
    ciclo = tuple(matplotlib.rcParams["axes.prop_cycle"].by_key().get("color") or ())
    if not ciclo:  # pragma: no cover - `rcParams` de origem traz sempre dez cores
        return tuple(matplotlib.rcParams["lines.color"] for _ in range(quantas))
    return tuple(ciclo[i % len(ciclo)] for i in range(quantas))


def nova_figura(
    *,
    largura: float = LARGURA_PADRAO,
    altura: float = ALTURA_PADRAO,
) -> Figure:
    """Uma `Figure` com canvas `Agg` já ligado, sem opinião sobre cor.

    Não usa `pyplot`, que registra cada figura num gestor global só limpo por `close()`: num
    aplicativo com vários painéis isso seria um vazamento por painel aberto. Aqui a figura é
    um objeto comum, recolhido quando ninguém o referencia.

    O canvas `Agg` existe porque sem renderizador não se mede a caixa de um texto, e o rodapé
    do recorte depende dessa medida. Quem embute a figura substitui o canvas pelo do Qt.

    Não define `facecolor`, `edgecolor` nem cor de texto: a figura herda os `rcParams`, e quem
    pinta é a janela, sobre a figura já construída.
    """
    figura = Figure(figsize=(largura, altura))
    FigureCanvasAgg(figura)
    return figura

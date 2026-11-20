"""As três figuras do PLS-DA multiclasse: scores, cargas e VIP dos modelos um-contra-resto.

O PLS-DA multiclasse é feito de K modelos um-contra-resto com dois componentes, e o resultado
é uma coleção: `q2` e `permutation_pvalue` vêm como tuplas de K e nada os resume. A média
apagaria a amplitude entre modelos, e combinar K p-valores num só escolheria a pergunta por
quem lê. Por isso cada figura é uma grelha, com um modelo por painel e os K `Q²` visíveis ao
mesmo tempo. Os K modelos não cabem numa nuvem só: cada um ajusta a própria projeção contra o
alvo `0/1` dele, e os componentes de dois modelos não são o mesmo eixo.

Não há como desenhar scores sem `q2` e `permutation_pvalue`. `Validacao` é campo obrigatório
das três cargas de desenho, cada painel imprime o par do modelo dele no título, e o rodapé
repete os K pares com o nome da classe ao lado. Um par só no rodapé de uma grelha seria um
agregado que ninguém calculou.

VIP indefinido não é VIP zero, e a contagem é por modelo. O painel de um modelo sem VIP
calculável escreve o porquê no lugar dos pontos, e o rodapé declara a contagem de cada modelo
em separado.

As séries dos painéis de score são os grupos clínicos da coorte, com a paleta de
`series.series_por_grupo`, calculada uma vez sobre o vetor de grupos partilhado, para a mesma
classe ter a mesma cor em todos os painéis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from refino.plots.base import (
    ALTURA_PADRAO,
    LARGURA_PADRAO,
    MatrizVaziaError,
    cores_do_ciclo,
    denominador_ou_recusa,
    nova_figura,
    vetor_do_fio_ou_nucleo,
)

# A recusa é a de `plots.plsda`, importada: o defeito que ela nomeia é o mesmo nos dois
# módulos, e uma segunda classe daria ao painel duas frases para ele.
from refino.plots.plsda import (
    LIMIAR_DE_REFERENCIA_DO_VIP,
    MATRIZ_DO_MODELO,
    ROTULO_DA_REFERENCIA,
    PlsdaDesalinhadoError,
)
from refino.plots.recorte import (
    TAMANHO_DA_FONTE,
    Recorte,
    aplica,
    decimal,
    frase_contada,
    milhar,
)
from refino.plots.series import series_por_grupo
from refino.plots.validacao import Validacao, frase_curta_da_validacao

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from refino.plots.series import EstilosDosGrupos, SerieDeGrupo

__all__ = [
    "COLUNAS_DA_GRELHA",
    "FEATURES_DA_IMPORTANCIA",
    "FRACAO_DA_FAIXA_DA_LEGENDA",
    "MIN_MODELOS",
    "PRETRATAMENTO_MULTICLASSE",
    "ROTULO_DA_ORIGEM",
    "SEPARADOR_DA_VALIDACAO",
    "TITULO_DAS_CARGAS_MULTICLASSE",
    "TITULO_DA_IMPORTANCIA_MULTICLASSE",
    "TITULO_DOS_ESCORES_MULTICLASSE",
    "CargaDeModelo",
    "CargasDoModelo",
    "EscoresDoModelo",
    "ImportanciaDoModelo",
    "cargas_do_modelo",
    "escores_do_modelo",
    "figura_da_importancia_um_contra_resto",
    "figura_das_cargas_um_contra_resto",
    "figura_dos_escores_um_contra_resto",
    "frase_da_importancia_indisponivel",
    "frase_dos_pares",
    "frase_dos_vereditos",
    "importancia_do_modelo",
    "recorte_da_importancia_um_contra_resto",
    "recorte_das_cargas_um_contra_resto",
    "recorte_dos_escores_um_contra_resto",
    "rotulo_do_painel",
    "selecao_da_importancia",
    "titulo_do_painel",
]

TITULO_DOS_ESCORES_MULTICLASSE: Final = "Scores de PLS-DA um-contra-resto — um painel por classe"

TITULO_DAS_CARGAS_MULTICLASSE: Final = "Cargas de PLS-DA um-contra-resto — um painel por classe"

TITULO_DA_IMPORTANCIA_MULTICLASSE: Final = "VIP de PLS-DA um-contra-resto — um painel por classe"

# Menos de dois modelos não é multiclasse: um modelo só é o caminho binário, e uma grelha de um
# painel seria a figura binária com o título errado. Repete
# `statistics.plsda_multiclasse.MIN_GRUPOS`, que é a mesma política do lado do cálculo.
MIN_MODELOS: Final = 2

# Colunas da grelha. Com as quatro classes da coorte ela sai 2x2, e os painéis se comparam aos
# pares; uma linha de quatro daria painéis estreitos demais para ler o eixo x. Com K ímpar a
# última linha fica com um painel só, e nenhum painel vazio é criado para fechar a grelha.
COLUNAS_DA_GRELHA: Final = 2

# O separador que `frase_curta_da_validacao` usa entre o `Q²` e o `p`. `titulo_do_painel` o
# troca por uma quebra de linha; se aquela função mudar de separador, esta linha muda junto.
SEPARADOR_DA_VALIDACAO: Final = " · "

# Rótulo da cruz da origem nas cargas. Os dois braços o usam, um deles com o prefixo `_`, que
# o tira da legenda.
ROTULO_DA_ORIGEM: Final = "origem dos dois componentes"

# Altura da faixa da legenda, em fração de uma linha de painéis. A legenda é partilhada pelos K
# painéis e fica numa célula própria da grelha, com o eixo desligado.
FRACAO_DA_FAIXA_DA_LEGENDA: Final = 0.25

# Quantas features cada painel de VIP desenha por omissão. Os rótulos de marca vizinhos só se
# sobrepõem a partir de 20 features e com 16 a folga é de cerca de um píxel, então 12 deixa
# margem. É também o valor de `plsda.FEATURES_DO_VIP`, para as figuras binária e multiclasse
# terem a mesma profundidade quando comparadas lado a lado. O custo de redesenho cresce com o
# número de rótulos, mas a grelha de quatro painéis já passa de 100 ms com qualquer valor. É
# recorte de quem chama, não teto.
FEATURES_DA_IMPORTANCIA: Final = 12

# A procedência que as três figuras declaram. Difere de `plsda.PRETRATAMENTO`, que descreve o
# contraste entre dois grupos; a matriz é a mesma, por isso `MATRIZ_DO_MODELO` é importado.
PRETRATAMENTO_MULTICLASSE: Final = (
    "PLS-DA um-contra-resto sobre a matriz PROCESSADA — imputação, normalização, "
    "transformação e escalonamento da spec —, um modelo por classe, com orientação canônica de "
    "sinal"
)

# ---------------------------------------------------------------------------
# As três cargas de desenho, uma por modelo, cada uma com a `Validacao` obrigatória
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EscoresDoModelo:
    """Os scores de um modelo um-contra-resto, com o grupo positivo dele e a validação.

    `x`, `y` e `grupos` têm uma entrada por amostra de estudo, na ordem das linhas da matriz
    processada. O vetor `grupos` é o mesmo nos K modelos; o que muda é qual classe é positiva.

    `grupo` é a classe positiva deste modelo: `y = 1` são as amostras dela e `y = 0` todas as
    outras. Sem ele os K painéis seriam indistinguíveis.
    """

    grupo: str
    x: np.ndarray
    y: np.ndarray
    grupos: tuple[str, ...]
    validacao: Validacao

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def desenhaveis(self) -> np.ndarray:
        """Booleano por amostra: `True` onde as duas coordenadas são finitas."""
        return np.isfinite(self.x) & np.isfinite(self.y)


@dataclass(frozen=True)
class CargasDoModelo:
    """A carga de cada feature nos dois componentes de um modelo, com o nome de cada uma.

    Uma entrada por feature da matriz processada, alinhada a `features`. Nomes de outra
    população rotulariam cada ponto com o nome de outra feature, numa figura bem formada.
    """

    grupo: str
    x: np.ndarray
    y: np.ndarray
    features: tuple[str, ...]
    validacao: Validacao

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def desenhaveis(self) -> np.ndarray:
        """Booleano por feature: `True` onde as duas cargas são finitas."""
        return np.isfinite(self.x) & np.isfinite(self.y)


@dataclass(frozen=True)
class ImportanciaDoModelo:
    """O VIP por feature de um modelo, com zero e indefinido separados.

    `nan` é 'não foi possível calcular' e `0.0` é 'sem importância'. `statistics.plsda._vip`
    devolve `nan` para todas as features quando a soma de quadrados total do modelo é zero.
    `calculaveis` separa os dois casos, `n_indefinidas` conta, e a figura declara as
    indefinidas em vez de desenhar uma coluna de zeros.
    """

    grupo: str
    valores: np.ndarray
    features: tuple[str, ...]
    validacao: Validacao

    @property
    def n(self) -> int:
        return int(self.valores.size)

    @property
    def calculaveis(self) -> np.ndarray:
        """Booleano por feature: `True` onde o VIP é um número."""
        return np.isfinite(self.valores)

    @property
    def n_calculaveis(self) -> int:
        return int(self.calculaveis.sum())

    @property
    def n_indefinidas(self) -> int:
        return int(self.n - self.n_calculaveis)


# As três cargas partilham `grupo` e `validacao`, e é só isso que as frases da coleção leem.
CargaDeModelo = EscoresDoModelo | CargasDoModelo | ImportanciaDoModelo


# ---------------------------------------------------------------------------
# A fronteira: o que chega do fio não é tipado
# ---------------------------------------------------------------------------


def _textos(valores: Any) -> tuple[str, ...]:
    """Uma tupla de `str` do fio. O ilegível vira `""` em vez de ser descartado.

    A posição tem de sobreviver: nome deslocado identifica errado numa figura bem formada, e
    nome vazio se vê.
    """
    if not isinstance(valores, (list, tuple)):
        return ()
    return tuple("" if isinstance(v, bool) or not isinstance(v, str) else v for v in valores)


def _grupo_ou_recusa(valor: Any) -> str:
    """O nome da classe positiva deste modelo, ou a recusa. Não há recuo.

    O nome diz de que modelo o `Q²` do painel fala. Um recuo como `"(sem grupo)"` poria o
    título a nomear um contraste que ninguém consegue reproduzir.
    """
    if isinstance(valor, str) and valor.strip():
        return valor
    raise PlsdaDesalinhadoError(
        f"a classe positiva de um modelo um-contra-resto chegou como {valor!r}. É ela que diz de "
        "que modelo o Q² do painel fala, e um nome em branco faria o título descrever um "
        "contraste que ninguém consegue reproduzir"
    )


def escores_do_modelo(
    *,
    grupo: Any,
    comp1: Sequence[float] | np.ndarray | Any,
    comp2: Sequence[float] | np.ndarray | Any,
    grupos: Sequence[str] | Any,
    validacao: Validacao,
) -> EscoresDoModelo:
    """Monta os scores de um modelo a partir do fio, com as guardas de fronteira.

    `validacao` é argumento obrigatório, sem padrão: não há `EscoresDoModelo` sem o par.

    O vetor de grupos não tem domínio fechado, ao contrário do caminho binário. A coorte tem K
    classes e o modelo vê todas: a classe positiva é uma delas e as outras são o resto. A
    guarda exige que o `grupo` deste modelo esteja no vetor, senão o painel prometeria uma
    classe positiva sem amostra desenhada.
    """
    x = vetor_do_fio_ou_nucleo(comp1)
    y = vetor_do_fio_ou_nucleo(comp2)
    nome = _grupo_ou_recusa(grupo)
    if x.size == 0:
        raise MatrizVaziaError(
            f"o modelo {nome!r} não trouxe score nenhum: o PLS-DA multiclasse não correu para esta "
            "execução. Figura em branco seria indistinguível de figura quebrada"
        )
    if x.size != y.size:
        raise PlsdaDesalinhadoError(
            f"o modelo {nome!r} traz {x.size} score(s) do componente 1 para {y.size} do "
            "componente 2. Os dois vetores descrevem as mesmas amostras, e comprimentos "
            "diferentes significam que um é de outra população"
        )
    rotulos = _textos(grupos)
    if len(rotulos) != x.size:
        raise PlsdaDesalinhadoError(
            f"o modelo {nome!r} traz {len(rotulos)} rótulo(s) de grupo para {x.size} score(s). O "
            "grupo é o que separa as séries do painel, e um vetor desalinhado poria amostras de "
            "uma classe na cor de outra por acidente de índice"
        )
    if nome not in rotulos:
        raise PlsdaDesalinhadoError(
            f"a classe positiva do modelo é {nome!r} e o vetor de grupos traz "
            f"{sorted(set(rotulos))}. Um modelo um-contra-resto sem nenhuma amostra do lado "
            "positivo desenha um painel cujo título nomeia uma classe que não está lá"
        )
    return EscoresDoModelo(grupo=nome, x=x, y=y, grupos=rotulos, validacao=validacao)


def cargas_do_modelo(
    *,
    grupo: Any,
    comp1: Sequence[float] | np.ndarray | Any,
    comp2: Sequence[float] | np.ndarray | Any,
    features: Sequence[str] | Any,
    validacao: Validacao,
) -> CargasDoModelo:
    """As cargas de um modelo vindas do fio, alinhadas ao vetor de nomes das features."""
    x = vetor_do_fio_ou_nucleo(comp1)
    y = vetor_do_fio_ou_nucleo(comp2)
    nome = _grupo_ou_recusa(grupo)
    if x.size == 0:
        raise MatrizVaziaError(
            f"o modelo {nome!r} não trouxe carga nenhuma: o PLS-DA multiclasse não correu para "
            "esta execução. Figura em branco seria indistinguível de figura quebrada"
        )
    if x.size != y.size:
        raise PlsdaDesalinhadoError(
            f"o modelo {nome!r} traz {x.size} carga(s) do componente 1 para {y.size} do "
            "componente 2. Os dois vetores descrevem as mesmas features, e comprimentos "
            "diferentes significam que um é de outra população"
        )
    nomes = _textos(features)
    if len(nomes) != x.size:
        raise PlsdaDesalinhadoError(
            f"o modelo {nome!r} traz {len(nomes)} nome(s) de feature para {x.size} carga(s). Um "
            "vetor desalinhado devolveria o nome do vizinho — com a figura bem formada"
        )
    return CargasDoModelo(grupo=nome, x=x, y=y, features=nomes, validacao=validacao)


def importancia_do_modelo(
    *,
    grupo: Any,
    vip: Sequence[float] | np.ndarray | Any,
    features: Sequence[str] | Any,
    validacao: Validacao,
) -> ImportanciaDoModelo:
    """Monta o VIP de um modelo a partir do fio. `None` no fio vira `nan`, não zero.

    O FastAPI serializa `nan` como `null`, e `vetor_do_fio_ou_nucleo` faz a conversão de volta.
    Um recuo para `0.0` apagaria a distinção entre indefinido e sem importância.
    """
    valores = vetor_do_fio_ou_nucleo(vip)
    nome = _grupo_ou_recusa(grupo)
    if valores.size == 0:
        raise MatrizVaziaError(
            f"o modelo {nome!r} não trouxe VIP nenhum: o PLS-DA multiclasse não correu para esta "
            "execução. Figura em branco seria indistinguível de figura quebrada"
        )
    nomes = _textos(features)
    if len(nomes) != valores.size:
        raise PlsdaDesalinhadoError(
            f"o modelo {nome!r} traz {len(nomes)} nome(s) de feature para {valores.size} valor(es) "
            "de VIP. O nome é o rótulo da marca do eixo, e um vetor desalinhado atribuiria a "
            "importância de uma feature ao nome de outra"
        )
    return ImportanciaDoModelo(grupo=nome, valores=valores, features=nomes, validacao=validacao)


# ---------------------------------------------------------------------------
# A coleção: as três cargas descrevem os mesmos K modelos, e isso é conferido
# ---------------------------------------------------------------------------


def _exige_colecao[T: (EscoresDoModelo, CargasDoModelo, ImportanciaDoModelo)](
    modelos: Sequence[T], *, o_que: str
) -> tuple[T, ...]:
    """Devolve os K modelos, ou recusa. A unidade conferida é a coleção.

    - Ao menos `MIN_MODELOS`: uma grelha de um painel seria a figura binária com o título
      errado.
    - Nomes de classe distintos: dois painéis com o mesmo título dariam dois `Q²` à mesma
      classe.
    - Populações do mesmo tamanho: os K modelos saem da mesma matriz, então tamanhos diferentes
      indicam uma coleção montada a partir de duas execuções.
    """
    colecao = tuple(modelos)
    if len(colecao) < MIN_MODELOS:
        raise PlsdaDesalinhadoError(
            f"a coleção de {o_que} traz {len(colecao)} modelo(s) e o um-contra-resto exige ao "
            f"menos {MIN_MODELOS}. Uma grelha de um painel é a figura binária com o título errado, "
            "e com dois grupos quem responde a mesma pergunta com um modelo só é `plots.plsda`"
        )
    nomes = [m.grupo for m in colecao]
    if len(set(nomes)) != len(nomes):
        repetidos = sorted({n for n in nomes if nomes.count(n) > 1})
        raise PlsdaDesalinhadoError(
            f"a coleção de {o_que} traz a classe {repetidos} mais de uma vez: {nomes}. Cada painel "
            "é um modelo, e dois painéis com o mesmo título dariam dois Q² à mesma classe"
        )
    tamanhos = {m.grupo: _tamanho(m) for m in colecao}
    if len(set(tamanhos.values())) != 1:
        raise PlsdaDesalinhadoError(
            f"os modelos de {o_que} descrevem populações de tamanhos diferentes: {tamanhos}. Os K "
            "modelos saem da MESMA matriz por construção, então tamanhos diferentes significam "
            "que a coleção foi montada a partir de duas execuções"
        )
    return colecao


def _tamanho(modelo: CargaDeModelo) -> int:
    """A população que este modelo descreve: amostras nos escores, features nas outras duas."""
    return modelo.n


# ---------------------------------------------------------------------------
# Os rótulos e a seleção
# ---------------------------------------------------------------------------


def rotulo_do_painel(grupo: str) -> str:
    """`"HAMTSP"` vira `"HAMTSP contra o resto"`.

    'Contra o resto' é o que diz que o alvo é `0/1`, e não uma comparação entre duas classes
    nomeadas. Sem isso, uma grelha de quatro painéis se leria como quatro contrastes par a par.
    """
    return f"{grupo} contra o resto"


def titulo_do_painel(grupo: str, validacao: Validacao) -> str:
    """O título de um painel: a classe positiva e, abaixo, o par de validação em duas linhas.

    O par vai em cada painel porque os K `Q²` são uma coleção; um número só no alto da grelha
    seria atribuído ao painel que o leitor estivesse a olhar.

    São três linhas porque, numa grelha de duas colunas, `frase_curta_da_validacao` numa linha
    só é mais larga que metade da figura: os títulos de painéis vizinhos se tocariam e o da
    direita seria cortado na borda. A quebra troca o separador da frase por uma quebra de
    linha, para os dois números terem uma fonte só.
    """
    par = frase_curta_da_validacao(validacao).replace(SEPARADOR_DA_VALIDACAO, "\n")
    return f"{rotulo_do_painel(grupo)}\n{par}"


def selecao_da_importancia(importancia: ImportanciaDoModelo, quantas: int) -> np.ndarray:
    """Os índices das `quantas` features de maior VIP deste modelo, em ordem decrescente.

    O empate é resolvido pelo nome, como em `plsda.selecao_do_vip`, para a mesma execução não
    desenhar features diferentes em duas máquinas. Não há sorteio nem semente.

    As indefinidas ficam fora da seleção e dentro da contagem que o rodapé declara, por modelo.
    """
    if quantas < 0:
        raise ValueError(f"`quantas` não pode ser negativo; recebi {quantas}")
    calculaveis = np.nonzero(importancia.calculaveis)[0]
    ordem = sorted(
        calculaveis.tolist(),
        key=lambda i: (-float(importancia.valores[i]), importancia.features[i]),
    )
    return np.array(ordem[:quantas], dtype=int)


# ---------------------------------------------------------------------------
# As frases da coleção: o par de validação dos K modelos, sem agregar
# ---------------------------------------------------------------------------


def frase_dos_pares(modelos: Sequence[CargaDeModelo]) -> str:
    """Os K pares de validação, um por classe, para o rodapé.

    Sai `"Asymptomatic: Q² = -0,1201 · p = 0,3333 | ATLL: …"`. São K pares e não um, porque
    combinar K p-valores escolheria a pergunta ('alguma classe separa' ou 'todas separam') por
    quem lê.

    O regime sai uma vez. `cv_folds`, `n_permutacoes` e `n_componentes` são os mesmos nos K
    modelos, e repeti-los por classe alongaria o rodapé sem informar. Ele é lido da primeira
    validação.
    """
    primeira = modelos[0].validacao
    pares = " | ".join(
        f"{m.grupo}: Q² = {_numero(m.validacao.q2)} · p = {_numero(m.validacao.permutation_pvalue)}"
        for m in modelos
    )
    return (
        f"validação por classe — {pares}. "
        f"Regime comum aos {milhar(len(modelos))} modelos: validação cruzada de "
        f"{milhar(primeira.cv_folds)} dobras estratificadas pelo vetor de classe, "
        f"{milhar(primeira.n_permutacoes)} permutações (resolução do p = "
        f"{_resolucao(primeira.n_permutacoes)}), {milhar(primeira.n_componentes)} componentes"
    )


def _numero(valor: float) -> str:
    """`0,2371` ou `"não declarado"`. Nunca um número no lugar da ausência dele.

    Repete `validacao._numero_ou_ausente`, que é privada daquele módulo, com o mesmo texto,
    para as figuras declararem a ausência com as mesmas palavras.
    """
    return decimal(valor) if math.isfinite(valor) else "não declarado"


def _resolucao(n_permutacoes: int) -> str:
    """`1/(n+1)` por extenso: o menor `p` que o teste de permutação consegue produzir.

    Sem ela, `p = 0,000999` parece medida quando é o mínimo possível de mil sorteios.
    """
    if n_permutacoes < 1:
        return "não declarada — o número de permutações não veio"
    return decimal(1.0 / (n_permutacoes + 1.0))


def frase_dos_vereditos(modelos: Sequence[CargaDeModelo]) -> str:
    """Quantos dos K modelos caem em cada um dos três desfechos de `Q²`, e a amplitude entre eles.

    É uma contagem, não um veredito único sobre K modelos, que seria um agregado. Os números de
    cada classe estão em `frase_dos_pares`, logo antes.

    Zero é a definição, não um limiar escolhido: `Q² = 1 - PRESS/TSS` com `PRESS = TSS` é errar
    tanto quanto errar a média dos rótulos.

    A amplitude do `Q²` entre modelos é reportada e não classifica nada; não tem limiar. O
    verbo concorda com a contagem, para a frase não sair como '1 têm'.
    """
    valores = [m.validacao.q2 for m in modelos]
    sem_medida = sum(1 for v in valores if not math.isfinite(v))
    piores = sum(1 for v in valores if math.isfinite(v) and v <= 0.0)
    melhores = len(valores) - sem_medida - piores
    finitos = [v for v in valores if math.isfinite(v)]
    amplitude = (
        f"a amplitude do Q² entre os modelos medidos é {decimal(max(finitos) - min(finitos))}"
        if finitos
        else "nenhum modelo tem Q² medido, então não há amplitude a reportar"
    )
    return (
        f"dos {milhar(len(valores))} modelos, {_tem(piores)} Q² <= 0 — {_predizem(piores)} PIOR "
        f"que a média dos rótulos, e a separação desenhada é ajuste dentro da amostra e não "
        f"predição —, {_tem(melhores)} Q² > 0 e {_tem(sem_medida, negado=True)} Q² medido. "
        f"{amplitude}, e ela é reportada e não classifica nada"
    )


def _tem(quantos: int, *, negado: bool = False) -> str:
    """`1` vira `"1 tem"`; `3` vira `"3 têm"`. O verbo concorda com a contagem."""
    verbo = "tem" if quantos == 1 else "têm"
    return f"{milhar(quantos)} não {verbo}" if negado else f"{milhar(quantos)} {verbo}"


def _predizem(quantos: int) -> str:
    """`1` vira `"prediz"`; `3` vira `"predizem"`."""
    return "prediz" if quantos == 1 else "predizem"


# ---------------------------------------------------------------------------
# Os três recortes
# ---------------------------------------------------------------------------


def recorte_dos_escores_um_contra_resto(
    *, modelos: Sequence[EscoresDoModelo], n_features_totais: Any
) -> Recorte:
    """O rodapé da grelha de scores: o recorte, o método e os K pares de validação.

    O numerador são as features do modelo, não as amostras desenhadas, como em
    `plsda.recorte_dos_scores_do_plsda`: o campo diz de quantas features os modelos saíram, que
    se confere contra o `n_features_saida` da execução. A contagem de amostras fica na legenda.
    """
    colecao = _exige_colecao(modelos, o_que="scores")
    totais = denominador_ou_recusa(n_features_totais, de="o PLS-DA multiclasse")
    amostras = colecao[0].n
    return Recorte(
        desenhadas=totais,
        disponiveis=totais,
        amostragem=(
            f"nenhuma — as {milhar(totais)} features da matriz processada entram todas em cada um "
            f"dos {milhar(len(colecao))} modelos, e os {milhar(amostras)} pontos de cada painel "
            f"são as MESMAS amostras projetadas no espaço daquele modelo. Os espaços não são "
            f"comparáveis entre painéis: cada modelo ajusta a projeção dele contra o alvo 0/1 da "
            f"classe que o título nomeia"
        ),
        matriz=MATRIZ_DO_MODELO,
        metodo=(
            f"{PRETRATAMENTO_MULTICLASSE}. {frase_dos_pares(colecao)}. "
            f"{frase_dos_vereditos(colecao)}"
        ),
    )


def recorte_das_cargas_um_contra_resto(
    *, modelos: Sequence[CargasDoModelo], n_features_totais: Any
) -> Recorte:
    """O rodapé das cargas. O numerador são as features desenháveis em todos os modelos.

    As features com carga não finita em ao menos um dos K modelos são contadas, não
    descartadas, e a frase diz quantas são em cada modelo.
    """
    colecao = _exige_colecao(modelos, o_que="cargas")
    em_todos = np.all([m.desenhaveis for m in colecao], axis=0)
    desenhaveis = int(em_todos.sum())
    fora = colecao[0].n - desenhaveis
    por_modelo = {m.grupo: int(m.n - int(m.desenhaveis.sum())) for m in colecao}
    indefinidas = (
        "nenhuma tem carga não finita em modelo nenhum"
        if fora == 0
        else (
            f"as {milhar(fora)} sem carga finita em ao menos um dos modelos são contadas, não "
            f"descartadas — por modelo: {por_modelo}"
        )
    )
    return Recorte(
        desenhadas=desenhaveis,
        disponiveis=denominador_ou_recusa(n_features_totais, de="o PLS-DA multiclasse"),
        amostragem=(
            f"nenhuma — as {milhar(colecao[0].n)} features do modelo entram todas em cada um dos "
            f"{milhar(len(colecao))} painéis, e {indefinidas}"
        ),
        matriz=MATRIZ_DO_MODELO,
        metodo=(
            f"{PRETRATAMENTO_MULTICLASSE}. A carga é a projeção da feature no componente, e NÃO "
            f"uma medida de significância. {frase_dos_pares(colecao)}. "
            f"{frase_dos_vereditos(colecao)}"
        ),
    )


def recorte_da_importancia_um_contra_resto(
    *, modelos: Sequence[ImportanciaDoModelo], n_desenhadas_por_modelo: int
) -> Recorte:
    """O rodapé do VIP: quantas features distintas a grelha desenha e as indefinidas por modelo.

    O numerador é a união dos K conjuntos selecionados, não a soma. A soma contaria várias
    vezes a feature importante em mais de um modelo e poderia passar da população, o que
    `Recorte` recusa.

    A contagem de indefinidas é por modelo. Um total apagaria que um modelo não ajustou e os
    outros ajustaram.
    """
    colecao = _exige_colecao(modelos, o_que="importância")
    selecionadas: set[int] = set()
    for modelo in colecao:
        selecionadas.update(int(i) for i in selecao_da_importancia(modelo, n_desenhadas_por_modelo))
    por_modelo = {m.grupo: m.n_indefinidas for m in colecao}
    indefinidas = (
        "nenhuma feature ficou sem VIP calculável em modelo nenhum"
        if not any(por_modelo.values())
        else (
            f"features sem VIP calculável, por modelo, de {milhar(colecao[0].n)}: {por_modelo} — "
            "isso é «não foi possível calcular», e não «importância zero»"
        )
    )
    return Recorte(
        desenhadas=len(selecionadas),
        disponiveis=denominador_ou_recusa(colecao[0].n, de="o PLS-DA multiclasse"),
        amostragem=(
            f"{_frase_da_selecao(len(selecionadas), n_desenhadas_por_modelo, len(colecao))} — "
            f"escolha determinística, com empate resolvido pelo nome; nenhuma amostragem "
            f"aleatória. {indefinidas}"
        ),
        matriz=MATRIZ_DO_MODELO,
        metodo=(
            f"{PRETRATAMENTO_MULTICLASSE}. VIP_j² = p · Σ_a(w_ja² · SS_a) / Σ_a SS_a — a média de "
            f"VIP² sobre as features é 1 por construção, e a referência é essa média e não um "
            f"corte de decisão. {frase_dos_pares(colecao)}. {frase_dos_vereditos(colecao)}"
        ),
    )


def _frase_da_selecao(distintas: int, por_modelo: int, n_modelos: int) -> str:
    """`(21, 8, 4)` vira `"as 21 features DISTINTAS de maior VIP — até 8 por painel, em 4…"`.

    Usa `recorte.frase_contada`, para o zero e o um saírem como frases e não como 'as 0'.
    """
    return frase_contada(
        distintas,
        nenhuma="nenhuma feature foi desenhada, porque nenhum modelo tem VIP calculável",
        uma=(
            f"1 feature DISTINTA de maior VIP — até {milhar(por_modelo)} por painel, em "
            f"{milhar(n_modelos)} painéis; uma feature importante em vários modelos conta uma vez"
        ),
        algumas=(
            f"features DISTINTAS de maior VIP — até {milhar(por_modelo)} por painel, em "
            f"{milhar(n_modelos)} painéis; uma feature importante em vários modelos conta uma vez"
        ),
    )


# ---------------------------------------------------------------------------
# A grelha: um painel por modelo
# ---------------------------------------------------------------------------


def _grelha(quantos: int, titulo: str) -> tuple[Figure, tuple[Axes, ...], Axes]:
    """`quantos` painéis em `COLUNAS_DA_GRELHA` colunas, o título comum e a faixa da legenda.

    A legenda tem uma célula própria, que atravessa todas as colunas, com o eixo desligado.
    Fora do último painel ela faria o `tight_layout` encolher só aquele eixo, e um painel
    menor que os outros se lê como se tivesse menos dado.

    Os painéis nascem por `add_subplot`, e não por `subplots(linhas, colunas)`, que com K
    ímpar deixaria um eixo montado e vazio.

    A altura cresce com as linhas, cada uma com `ALTURA_PADRAO`, e a largura não. O tamanho
    real na janela é do widget; isto decide a proporção.
    """
    colunas = min(COLUNAS_DA_GRELHA, quantos)
    linhas = math.ceil(quantos / colunas)
    altura = ALTURA_PADRAO * (linhas + FRACAO_DA_FAIXA_DA_LEGENDA)
    figura = nova_figura(largura=LARGURA_PADRAO, altura=altura)
    grade = figura.add_gridspec(
        linhas + 1, colunas, height_ratios=[1.0] * linhas + [FRACAO_DA_FAIXA_DA_LEGENDA]
    )
    eixos = tuple(figura.add_subplot(grade[i // colunas, i % colunas]) for i in range(quantos))
    faixa = figura.add_subplot(grade[linhas, :])
    faixa.set_axis_off()
    figura.suptitle(titulo)
    return figura, eixos, faixa


def _legenda_na_faixa(faixa: Axes, eixo: Axes) -> None:
    """A legenda partilhada, horizontal, na faixa, colhida dos artistas de um painel.

    Basta um painel porque as séries são as mesmas em todos; colher de todos daria K entradas
    repetidas por série.
    """
    maos, rotulos = eixo.get_legend_handles_labels()
    if not maos:  # pragma: no cover - todo painel deste módulo rotula os artistas que desenha
        return
    faixa.legend(
        maos,
        rotulos,
        loc="center",
        ncols=min(len(maos), COLUNAS_DA_GRELHA + 1),
        frameon=False,
        fontsize=TAMANHO_DA_FONTE,
    )


def _expoente_ao_lado_do_painel(eixo: Axes) -> None:
    """Põe o expoente do eixo x à direita, do lado de fora do painel.

    Em `plsda.expoente_fora_do_rotulo` o expoente sobe para a linha das marcas, e isso basta
    numa figura de um eixo largo. Aqui os painéis têm metade da largura, o `matplotlib` põe uma
    marca no extremo do eixo, e o expoente alinhado à direita em `x = 1` cai em cima dela.

    Alargar a figura não resolve, porque a marca do extremo acompanha a borda do eixo. Dar
    margem ao eixo x também não: a colisão só muda de painel. O que resolve é a âncora: em
    `x = 1,02`, alinhado à esquerda, o expoente fica fora do eixo. Ele entra no `tightbbox`, e
    o `tight_layout` encolhe o painel para lhe dar lugar em vez de cortá-lo.
    """
    expoente = eixo.xaxis.get_offset_text()
    expoente.set_position((1.02, 1.0))
    expoente.set_horizontalalignment("left")
    expoente.set_verticalalignment("bottom")


def _rotulo_do_componente(indice: int) -> str:
    """`0` vira `"componente 1"`. Não leva ressalva de sinal, porque a orientação é canônica.

    É mais curto que o do caminho binário (`"componente 1 de PLS-DA"`) porque o eixo x de um
    painel tem metade da largura, e o nome do método já está no título da figura e no rodapé.
    """
    return f"componente {indice + 1}"


# ---------------------------------------------------------------------------
# 1. Os scores, com o par de validação em cada painel
# ---------------------------------------------------------------------------


def figura_dos_escores_um_contra_resto(
    *,
    modelos: Sequence[EscoresDoModelo],
    recorte: Recorte,
    estilos: EstilosDosGrupos | None = None,
) -> Figure:
    """Um painel por modelo: componente 1 contra componente 2, com as amostras coloridas por classe.

    Desenha a população inteira que recebe; não amostra.

    A paleta é calculada uma vez, do vetor de grupos do primeiro modelo, que é o mesmo nos K, e
    usada em todos os painéis.

    A legenda é uma só, na faixa abaixo da grelha, colhida do primeiro painel.

    Com `estilos`, cada classe leva a cor e o símbolo escolhidos, nos K painéis. Sem ele, o
    ciclo de cores e o círculo.
    """
    colecao = _exige_colecao(modelos, o_que="scores")
    figura, eixos, faixa = _grelha(len(colecao), TITULO_DOS_ESCORES_MULTICLASSE)
    paleta = series_por_grupo(colecao[0].grupos, estilos=estilos)

    desenhados = 0
    for eixo, modelo in zip(eixos, colecao, strict=True):
        desenhados += _desenha_os_escores(eixo, modelo, paleta)

    if desenhados == 0:
        raise MatrizVaziaError(
            f"há {colecao[0].n} amostra(s) e nenhuma com os dois componentes finitos em modelo "
            f"nenhum dos {len(colecao)}: não há ponto a desenhar. Uma grelha de eixos montados e "
            "vazios lê-se como figura quebrada"
        )

    _legenda_na_faixa(faixa, eixos[0])
    aplica(figura, recorte)
    return figura


def _desenha_os_escores(eixo: Axes, modelo: EscoresDoModelo, paleta: Sequence[SerieDeGrupo]) -> int:
    """Um painel de scores. Devolve quantos pontos desenhou.

    A classe positiva fica por cima, por `zorder`, porque a pergunta do painel é se ela se
    destaca do resto. Desenhá-la por último mudaria a ordem dos grupos em cada painel, e a
    legenda partilhada é colhida de um só.
    """
    desenhaveis = modelo.desenhaveis
    desenhados = 0
    for serie in paleta:
        mascara = serie.mascara & desenhaveis
        quantas = int(mascara.sum())
        if quantas == 0:
            continue
        desenhados += quantas
        eixo.plot(
            modelo.x[mascara],
            modelo.y[mascara],
            marker=serie.marcador,
            linestyle="none",
            color=serie.cor,
            label=serie.rotulo,
            zorder=3 if serie.grupo == modelo.grupo else 2,
        )
    eixo.set_xlabel(_rotulo_do_componente(0))
    eixo.set_ylabel(_rotulo_do_componente(1))
    eixo.set_title(titulo_do_painel(modelo.grupo, modelo.validacao), fontsize=TAMANHO_DA_FONTE)
    # Com scores de ordem `1e8`, que aparecem quando a spec tem `transformation` e `scaling`
    # em `none`, o expoente do eixo x colidiria com o rótulo ou com a última marca.
    _expoente_ao_lado_do_painel(eixo)
    return desenhados


# ---------------------------------------------------------------------------
# 2. As cargas
# ---------------------------------------------------------------------------


def figura_das_cargas_um_contra_resto(
    *, modelos: Sequence[CargasDoModelo], recorte: Recorte
) -> Figure:
    """Um painel por modelo: um ponto por feature, a carga no componente 1 contra a do 2.

    Desenha a população inteira que recebe.

    Nenhum artista leva `GID_DAS_FEATURES`. Essa marca diz que os pontos do artista são
    indexados pelo vetor de nomes que o painel recebe, que é um vetor para um artista. Aqui há
    K artistas de feature, e cada um desenha o subconjunto desenhável do modelo dele, que pode
    diferir. A identificação destas figuras usa o rótulo da série.

    As cruzes da origem são desenhadas em coordenadas de dado, não por `axhline`, que cria um
    `Line2D` de transformação mista; os testes de limites leem o dado do artista e acusariam
    pontos fora do eixo.
    """
    colecao = _exige_colecao(modelos, o_que="cargas")
    figura, eixos, faixa = _grelha(len(colecao), TITULO_DAS_CARGAS_MULTICLASSE)

    desenhados = 0
    cheio = eixos[0]
    for eixo, modelo in zip(eixos, colecao, strict=True):
        quantos = _desenha_as_cargas(eixo, modelo)
        if quantos and not desenhados:
            cheio = eixo
        desenhados += quantos

    if desenhados == 0:
        raise MatrizVaziaError(
            f"há {colecao[0].n} feature(s) e nenhuma com carga finita nos dois componentes em "
            f"modelo nenhum dos {len(colecao)}: não há ponto a desenhar. Uma grelha de eixos "
            "montados e vazios lê-se como figura quebrada"
        )

    _legenda_na_faixa(faixa, cheio)
    aplica(figura, recorte)
    return figura


def _desenha_as_cargas(eixo: Axes, modelo: CargasDoModelo) -> int:
    """Um painel de cargas. Devolve quantos pontos desenhou.

    Painel sem carga finita fica com um texto no lugar dos pontos e não levanta: um modelo
    degenerado ao lado de outros que ajustaram é fato do dado. A figura só é recusada quando
    nenhum painel tem o que desenhar.
    """
    eixo.set_xlabel(_rotulo_do_componente(0))
    eixo.set_ylabel(_rotulo_do_componente(1))
    eixo.set_title(titulo_do_painel(modelo.grupo, modelo.validacao), fontsize=TAMANHO_DA_FONTE)

    desenhaveis = modelo.desenhaveis
    quantas = int(desenhaveis.sum())
    if quantas == 0:
        _texto_no_lugar_dos_pontos(
            eixo,
            f"nenhuma das {milhar(modelo.n)} features tem carga finita\n"
            "nos dois componentes deste modelo.",
        )
        return 0

    px = modelo.x[desenhaveis]
    py = modelo.y[desenhaveis]
    da_nuvem, da_origem = cores_do_ciclo(2)
    # Os rótulos dos artistas são distintos porque os testes indexam as séries por rótulo.
    eixo.plot(
        [float(px.min()), float(px.max())],
        [0.0, 0.0],
        linewidth=0.8,
        color=da_origem,
        label=ROTULO_DA_ORIGEM,
    )
    # O segundo braço da cruz leva `_` no início do rótulo, o que o tira da legenda: a cruz
    # entra uma vez. O rótulo continua distinto do primeiro.
    eixo.plot(
        [0.0, 0.0],
        [float(py.min()), float(py.max())],
        linewidth=0.8,
        color=da_origem,
        label=f"_{ROTULO_DA_ORIGEM} (braço vertical)",
    )
    eixo.scatter(px, py, s=6, color=da_nuvem, label=f"{milhar(quantas)} features")
    _expoente_ao_lado_do_painel(eixo)
    return quantas


# ---------------------------------------------------------------------------
# 3. A importância, com zero e indefinido separados por modelo
# ---------------------------------------------------------------------------


def frase_da_importancia_indisponivel(importancia: ImportanciaDoModelo) -> str:
    """O que se escreve no lugar dos pontos quando este modelo não tem VIP calculável.

    A população não está vazia: são `n` features, todas com VIP indefinido porque a soma de
    quadrados explicada pelo modelo é zero. `MatrizVaziaError` diria outra coisa e, numa
    grelha, apagaria os painéis dos modelos que ajustaram.
    """
    return (
        f"nenhuma das {milhar(importancia.n)} features tem VIP calculável\n"
        "neste modelo: a soma de quadrados explicada por ele é zero,\n"
        "então a razão que define o VIP não existe.\n"
        "Isso é «não foi possível calcular» — e NÃO «importância zero»."
    )


def figura_da_importancia_um_contra_resto(
    *,
    modelos: Sequence[ImportanciaDoModelo],
    n_features: int = FEATURES_DA_IMPORTANCIA,
    recorte: Recorte,
) -> Figure:
    """Um painel por modelo: as `n_features` de maior VIP daquele modelo, com a referência em 1.

    Desenha a população inteira que recebe. Quantas desenhar é decisão de quem chama, e
    `selecao_da_importancia` é determinística, então o rodapé pode refazer a seleção.

    São pontos e não barras: `ax.bar` cria `Rectangle` em `ax.patches`, que os testes de figura
    e o cursor do painel não leem.

    Modelo sem VIP calculável diz isso no painel dele e não derruba a figura. A recusa só
    acontece quando nenhum dos K tem o que desenhar.
    """
    colecao = _exige_colecao(modelos, o_que="importância")
    figura, eixos, faixa = _grelha(len(colecao), TITULO_DA_IMPORTANCIA_MULTICLASSE)

    desenhados = 0
    cheio = eixos[0]
    for eixo, modelo in zip(eixos, colecao, strict=True):
        quantos = _desenha_a_importancia(eixo, modelo, n_features)
        if quantos and not desenhados:
            cheio = eixo
        desenhados += quantos

    if desenhados == 0:
        raise MatrizVaziaError(
            f"nenhum dos {len(colecao)} modelos tem VIP calculável, e a figura não tem ponto "
            "nenhum a desenhar em painel nenhum. Com ao menos um modelo calculável a grelha "
            "desenha-o e declara os outros por extenso, que é o fato oposto"
        )

    _legenda_na_faixa(faixa, cheio)
    aplica(figura, recorte)
    return figura


def _desenha_a_importancia(eixo: Axes, modelo: ImportanciaDoModelo, quantas: int) -> int:
    """Um painel de VIP. Devolve quantos pontos desenhou.

    A ordem é decrescente e o eixo y é invertido, para a maior ficar em cima.
    """
    eixo.set_xlabel("VIP")
    eixo.set_ylabel("feature")
    eixo.set_title(titulo_do_painel(modelo.grupo, modelo.validacao), fontsize=TAMANHO_DA_FONTE)

    indices = selecao_da_importancia(modelo, quantas)
    if indices.size == 0:
        _texto_no_lugar_dos_pontos(eixo, frase_da_importancia_indisponivel(modelo))
        # As marcas saem: sem ponto nenhum o `matplotlib` desenharia os limites de origem, e o
        # painel pareceria afirmar que o VIP deste modelo vai de zero a um.
        eixo.set_xticks([])
        eixo.set_yticks([])
        return 0

    alturas = np.arange(indices.size, dtype=float)
    valores = modelo.valores[indices]
    da_nuvem, da_referencia = cores_do_ciclo(2)
    eixo.plot(
        [LIMIAR_DE_REFERENCIA_DO_VIP, LIMIAR_DE_REFERENCIA_DO_VIP],
        [float(alturas.min()), float(alturas.max())],
        linestyle="--",
        linewidth=0.8,
        color=da_referencia,
        label=ROTULO_DA_REFERENCIA,
    )
    eixo.plot(
        valores,
        alturas,
        marker="o",
        linestyle="none",
        color=da_nuvem,
        label=f"{milhar(int(indices.size))} features de maior VIP",
    )
    eixo.set_yticks(alturas.tolist())
    eixo.set_yticklabels([modelo.features[int(i)] for i in indices], fontsize=TAMANHO_DA_FONTE)
    eixo.invert_yaxis()
    _expoente_ao_lado_do_painel(eixo)
    return int(indices.size)


def _texto_no_lugar_dos_pontos(eixo: Axes, texto: str) -> None:
    """A frase que ocupa um painel sem dado desenhável.

    Painel em branco ao lado de outros cheios se lê como figura que falhou, e não como modelo
    que não produziu o número.
    """
    eixo.text(
        0.5,
        0.5,
        texto,
        transform=eixo.transAxes,
        ha="center",
        va="center",
        fontsize=TAMANHO_DA_FONTE,
    )

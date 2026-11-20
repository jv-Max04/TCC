"""Violino de RSD por feature: a dispersão dos QCs contra a das amostras.

Duas distribuições lado a lado, o RSD por feature medido sobre os QCs e o medido sobre as
amostras, da mesma matriz. A matriz é a corrigida ou, antes do primeiro corretor, a ingerida;
o rodapé diz qual. A pergunta é se a dispersão técnica é menor que a biológica, e a resposta
se lê na posição e na extensão dos dois corpos. Cada contorno é escalado à própria densidade
máxima, então as larguras não se comparam entre lados.

O `n` de cada lado é de features. As contagens de injeção vão no rótulo do eixo x, nomeadas
como injeções, e servem para separar 'não há QC nenhum' de 'há QCs e nenhuma feature com RSD
definido'. As duas populações de features diferem e nenhuma é proxy da outra, por isso cada
lado escreve a sua. O título de cada lado conta o vetor desenhado e o rodapé traz o campo
`n_features_finitas_*` do servidor; se divergirem, a imagem mostra os dois.

Cada lado tem três desfechos: com dispersão, desenha contorno, caixa de quartis e mediana; com
features e um só valor distinto, desenha caixa e mediana e declara o contorno ausente; sem
feature com RSD definido, não desenha nada e declara a causa. Os dois lados vazios levantam
`MatrizVaziaError`.

O contorno é `Line2D` e não `ax.violinplot`: o corpo que `violinplot` devolve é uma coleção
sem `get_segments`, e os testes não conseguiriam ler dela o que foi desenhado. A densidade vem
de `distribuicao.curva_da_densidade` com margem zero, porque o RSD não é negativo.

Esta figura não ordena corretores. Os dois lados vêm da mesma matriz e o estimador é
`in_sample`: com um corretor que ajusta modelo, o lado dos QCs é medido nos pontos que ele
usou. O rodapé escreve o estimador por isso. As cores vêm do ciclo do tema
(`base.cores_do_ciclo`), porque aqui a cor só separa duas séries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from matplotlib.lines import Line2D

from refino.plots.base import (
    CORRETOR_NAO_DECLARADO,
    GID_REQUEBRAVEL,
    NAO_DECLARADA,
    REGUA_AUSENTE,
    MatrizVaziaError,
    cores_do_ciclo,
    denominador_ou_recusa,
    inteiro_do_fio,
    nova_figura,
    texto_do_fio,
    vetor_do_fio_ou_nucleo,
)
from refino.plots.distribuicao import (
    MINIMO_PARA_DENSIDADE,
    curva_da_densidade,
)
from refino.plots.recorte import (
    TAMANHO_DA_FONTE,
    Recorte,
    aplica,
    em_percentagem,
    legenda_fora,
    milhar,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "LARGURA_DA_CAIXA_DE_QUARTIS",
    "LARGURA_DO_VIOLINO",
    "MATRIZ_DA_LEITURA",
    "PAPEL_DA_DENSIDADE",
    "PAPEL_DA_MEDIANA",
    "PAPEL_DOS_QUARTIS",
    "ROTULO_DAS_AMOSTRAS",
    "ROTULO_DOS_QCS",
    "ROTULO_DO_EIXO_DO_RSD",
    "TITULO_DO_VIOLINO",
    "LadoDoRsd",
    "ParDoRsd",
    "figura_do_violino_do_rsd_por_feature",
    "frase_do_lado_ausente",
    "frase_do_lado_do_rsd",
    "frase_sem_densidade",
    "lado_do_rsd",
    "par_do_rsd",
    "recorte_do_violino_do_rsd_por_feature",
    "rotulo_do_rsd",
]

TITULO_DO_VIOLINO: Final = "Dispersão do RSD por feature — os QCs contra as amostras"

# Estágio da carga que vem do diagnóstico, antes de qualquer corretor. É a mesma cadeia que
# `api/rotas.py` publica em `ESTAGIO_DA_INGERIDA`, e um teste compara as duas.
MATRIZ_DA_LEITURA: Final = "ingerida"

# `metrics/rsd.py` devolve `sd/mean`, que é fração. O eixo vai em percentagem, que é a unidade
# em que o laboratório lê RSD e a da fila de barras da mesma aba, e o rótulo diz a unidade.
ROTULO_DO_EIXO_DO_RSD: Final = "RSD por feature (%)"

ROTULO_DOS_QCS: Final = "QCs"
ROTULO_DAS_AMOSTRAS: Final = "amostras"

# Papel de cada artista de dado. São três por lado no mesmo eixo, e os testes indexam as séries
# por rótulo: dois artistas com o mesmo texto colapsariam num só.
PAPEL_DA_DENSIDADE: Final = "densidade"
PAPEL_DOS_QUARTIS: Final = "quartis"
PAPEL_DA_MEDIANA: Final = "mediana"

# Meia largura do contorno, em unidades do eixo x. Os dois lados ficam em `0` e `1`, então `0,38`
# deixa folga entre eles e a caixa de cada um não toca a do vizinho.
LARGURA_DO_VIOLINO: Final = 0.38

# Meia largura da caixa de quartis. Bem menor que a do contorno, para a caixa se ler dentro do
# corpo sem esconder a forma dele.
LARGURA_DA_CAIXA_DE_QUARTIS: Final = 0.07

# O padrão de `curva_da_densidade` estende a grelha três larguras de banda para cada lado, como
# o `density()` do R. O RSD é `sd/mean` sobre valores não negativos: uma cauda abaixo do mínimo
# observado desenharia dispersão que ninguém mediu e, perto de zero, RSD negativo.
SEM_MARGEM: Final = 0.0

QUARTIS: Final = (0.25, 0.5, 0.75)

# Os limites do eixo x. Fixos porque as duas posições são fixas, e a folga é a da caixa da legenda.
LIMITES_DO_EIXO_DOS_LADOS: Final = (-0.6, 1.6)

# Quanto cada frase declarada desce, em fração da altura do eixo. São no máximo duas.
PASSO_DA_DECLARACAO: Final = 0.08


@dataclass(frozen=True)
class LadoDoRsd:
    """Um lado da comparação: o vetor de RSD por feature e o que o descreve.

    `valores` traz só os finitos; o filtro acontece em `lado_do_rsd`. Não há alinhamento por
    posição entre os dois lados: as duas distribuições correm sobre populações diferentes de
    features.
    """

    rotulo: str
    valores: np.ndarray
    n_nao_finitos: int
    """Quantos vieram do fio e não são número. O servidor promete mandar só os finitos, e este
    campo conta os que quebram a promessa, para a curva não descrever uma população menor que a
    declarada sem nada acusar."""

    n_features_finitas: int
    """O campo do corpo. Vai para o rodapé; o título leva o que de fato se desenhou."""

    n_injecoes: int
    """Quantas injeções entraram neste lado. O `n` da figura é de features; este número separa
    'não há QC nenhum' de 'há sete QCs e nenhuma feature com RSD definido'."""

    @property
    def n(self) -> int:
        """Quantas features com RSD definido este lado desenha."""
        return int(self.valores.size)

    @property
    def desenhavel(self) -> bool:
        return self.n > 0

    @property
    def tem_densidade(self) -> bool:
        """Se há dispersão para estimar. Com um só valor distinto, `gaussian_kde` é singular."""
        return int(np.unique(self.valores).size) >= MINIMO_PARA_DENSIDADE

    @property
    def percentagens(self) -> np.ndarray:
        """O vetor em percentagem. É o único ponto de conversão de fração para percentagem."""
        return self.valores * 100.0


def lado_do_rsd(
    *, rotulo: str, valores: Any, n_features_finitas: Any, n_injecoes: Any
) -> LadoDoRsd:
    """Monta um lado a partir do fio, com as guardas de fronteira.

    O que não é número vira `nan` em `vetor_do_fio_ou_nucleo` e sai do vetor aqui.
    `n_nao_finitos` guarda quantos saíram, para a figura não desenhar uma população e declarar
    outra.
    """
    todos = vetor_do_fio_ou_nucleo(valores)
    finitos = todos[np.isfinite(todos)]
    return LadoDoRsd(
        rotulo=rotulo,
        valores=finitos,
        n_nao_finitos=int(todos.size - finitos.size),
        n_features_finitas=inteiro_do_fio(n_features_finitas),
        n_injecoes=inteiro_do_fio(n_injecoes),
    )


@dataclass(frozen=True)
class ParDoRsd:
    """Os dois lados, mais os cinco rótulos sem os quais nenhum dos dois números é comparável.

    `matriz`, `corretor`, `variante`, `estimador` e `n_features` são do par e não de cada lado:
    os dois vetores saem da mesma matriz, sob a mesma variante e o mesmo estimador.
    """

    qcs: LadoDoRsd
    amostras: LadoDoRsd
    matriz: str
    corretor: str
    variante: str
    estimador: str
    n_features: int

    @property
    def lados(self) -> tuple[LadoDoRsd, LadoDoRsd]:
        """Os dois, na ordem em que são desenhados: os QCs à esquerda."""
        return (self.qcs, self.amostras)


def par_do_rsd(
    *,
    qcs: LadoDoRsd,
    amostras: LadoDoRsd,
    matriz: Any,
    corretor: Any,
    variante: Any,
    estimador: Any,
    n_features: Any,
) -> ParDoRsd:
    """Monta o par, com os rótulos passados pela guarda de fronteira.

    `corretor` e os dois rótulos de régua passam por `texto_do_fio` e não por uma guarda de
    vocabulário. O vocabulário é validado na fronteira, por `RsdPorFeature`, e uma segunda lista
    de valores válidos aqui poderia divergir da primeira.
    """
    return ParDoRsd(
        qcs=qcs,
        amostras=amostras,
        matriz=texto_do_fio(matriz, NAO_DECLARADA),
        corretor=texto_do_fio(corretor, CORRETOR_NAO_DECLARADO),
        variante=texto_do_fio(variante, REGUA_AUSENTE),
        estimador=texto_do_fio(estimador, REGUA_AUSENTE),
        n_features=inteiro_do_fio(n_features),
    )


def rotulo_do_rsd(lado: str, papel: str) -> str:
    """`("QCs", "densidade")` vira `"QCs — densidade"`. Único por artista de dado.

    Os testes indexam as séries por rótulo. A figura tem seis artistas no mesmo eixo (contorno,
    caixa e mediana de cada lado), e dois com o mesmo texto colapsariam num só.
    """
    return f"{lado} — {papel}"


def frase_do_lado_do_rsd(lado: LadoDoRsd) -> str:
    """O que vai no título sobre um lado: a contagem de features e a mediana.

    A contagem sai do vetor desenhado. A mediana fica no título porque é o número que responde
    à pergunta da figura; com um corretor que ajusta modelo, uma distribuição puxada para o
    zero sem número ao lado se lê como sucesso.
    """
    if not lado.desenhavel:
        return f"{lado.rotulo}: 0 features com RSD definido"
    mediana = em_percentagem(float(np.median(lado.valores)))
    corpo = f"{lado.rotulo}: {milhar(lado.n)} features, mediana {mediana}"
    if lado.n_nao_finitos:
        corpo += f" ({milhar(lado.n_nao_finitos)} não finito(s) fora da estimativa)"
    return corpo


def frase_do_lado_ausente(lado: LadoDoRsd) -> str:
    """Por que este lado não tem distribuição. São dois fatos distintos.

    'Não há injeção desta classe nesta matriz' é desenho do estudo; 'há injeções e nenhuma
    feature com RSD definido' é dado. Os dois dão `n == 0` e pedem ações diferentes de quem lê.
    """
    if lado.n_injecoes == 0:
        return (
            f"{lado.rotulo}: nenhuma injeção desta classe nesta matriz, logo não há distribuição a "
            "desenhar deste lado"
        )
    return (
        f"{lado.rotulo}: {milhar(lado.n_injecoes)} injeção(ões) e NENHUMA feature com RSD "
        "definido — a medição correu, e o resultado dela é vazio"
    )


def frase_sem_densidade(lado: LadoDoRsd) -> str:
    """O terceiro desfecho: há features e não há dispersão para estimar uma densidade.

    Uma reta afirmaria distribuição uniforme, e recusar a figura inteira apagaria o outro lado.
    Fica a caixa: a mediana e os quartis de um valor só não inventam nada.
    """
    return (
        f"{lado.rotulo}: {milhar(lado.n)} features e "
        f"{milhar(int(np.unique(lado.valores).size))} valor(es) distinto(s) — sem contorno de "
        "densidade; a caixa de quartis fica"
    )


def recorte_do_violino_do_rsd_por_feature(*, par: ParDoRsd) -> Recorte:
    """O rodapé: a população de cada lado, a procedência e a régua com a ressalva.

    `Recorte.texto()` escreve um único 'N de M features', e a figura tem duas populações. O par
    do topo é o do lado dos QCs e o rodapé diz isso, para o leitor não atribuir o número aos
    dois lados. As duas contagens vêm do corpo, não do vetor desenhado.
    """
    return Recorte(
        desenhadas=par.qcs.n_features_finitas,
        disponiveis=denominador_ou_recusa(par.n_features, de="o violino do RSD por feature"),
        amostragem=(
            f"nenhuma; o par acima é do lado «{par.qcs.rotulo}», e o «{par.amostras.rotulo}» traz "
            f"{milhar(par.amostras.n_features_finitas)} de {milhar(par.n_features)} — as duas "
            "populações DIFEREM, e nenhuma é proxy da outra"
        ),
        matriz=_matriz_do_violino(par),
        metodo=_metodo_do_violino(par),
    )


def _matriz_do_violino(par: ParDoRsd) -> str:
    """O campo `matriz` do rodapé: o estágio e o corretor que o produziu.

    Na matriz ingerida o corretor não é escrito e o rodapé diz 'antes de qualquer correção': o
    corpo do diagnóstico traz `corretor` nulo, e o recuo de `par_do_rsd` se leria como defeito.
    Nos outros estágios o recuo aparece, porque corpo de execução sem corretor é defeito e a
    imagem tem de mostrá-lo.
    """
    if par.matriz == MATRIZ_DA_LEITURA:
        return f"uma — «{par.matriz}», antes de qualquer correção"
    return f"uma — «{par.matriz}», do corretor «{par.corretor}»"


def _metodo_do_violino(par: ParDoRsd) -> str:
    """O campo `metodo`: como o contorno é estimado, o que não se compara e com que régua.

    A última frase não pode faltar. Os dois lados vêm da mesma matriz e o estimador é in-sample;
    ao lado da fila de barras, a imagem poderia ser lida como 'a correção funcionou'.
    """
    return (
        "contorno espelhado de densidade por núcleo gaussiano (`scipy.stats.gaussian_kde`, banda "
        "de Scott), avaliado só ENTRE o mínimo e o máximo observados — o RSD não é negativo. Caixa "
        "do 1.º ao 3.º quartil e mediana no traço, do MESMO vetor. Cada contorno é escalado à "
        "própria densidade máxima: as larguras NÃO se comparam entre lados, e sim a posição e a "
        f"extensão em RSD. Variante «{par.variante}», estimador «{par.estimador}»: os dois lados "
        "saem da MESMA matriz e esta figura NÃO ordena corretores — com um corretor que ajusta "
        "modelo, o lado dos QCs é medido nos pontos que ele usou. Sem teste estatístico"
    )


def figura_do_violino_do_rsd_por_feature(*, par: ParDoRsd, recorte: Recorte) -> Figure:
    """As duas distribuições de RSD por feature, lado a lado.

    Desenha a população inteira que recebe; não amostra.

    Usa um eixo só. As duas populações são RSD da mesma matriz, na mesma unidade, e o que se lê
    é se uma está acima da outra; dois eixos com limites independentes impediriam essa leitura.
    """
    figura = nova_figura()
    eixo = figura.add_subplot(111)

    declaracoes: list[str] = []
    provas: list[Artist] = []
    for posicao, (lado, cor) in enumerate(zip(par.lados, cores_do_ciclo(2), strict=True)):
        if not lado.desenhavel:
            declaracoes.append(frase_do_lado_ausente(lado))
            continue
        _desenha_o_violino(eixo, lado, posicao=posicao, cor=cor)
        if not lado.tem_densidade:
            declaracoes.append(frase_sem_densidade(lado))
        provas.append(Line2D([], [], color=cor, label=lado.rotulo))

    if not provas:
        raise MatrizVaziaError(
            f"nenhum dos dois lados tem feature com RSD definido: «{par.qcs.rotulo}» traz "
            f"{par.qcs.n} e «{par.amostras.rotulo}» traz {par.amostras.n}. Um eixo montado com "
            "duas frases e nenhum artista lê-se como figura quebrada, e o que é verdade é que "
            "não houve população nenhuma a desenhar"
        )

    for indice, frase in enumerate(declaracoes):
        declarada = eixo.text(
            0.02,
            0.98 - indice * PASSO_DA_DECLARACAO,
            frase,
            transform=eixo.transAxes,
            ha="left",
            va="top",
            fontsize=TAMANHO_DA_FONTE,
        )
        declarada.set_gid(GID_REQUEBRAVEL)

    eixo.set_xticks([posicao for posicao, _ in enumerate(par.lados)])
    eixo.set_xticklabels([lado.rotulo for lado in par.lados])
    eixo.set_xlim(*LIMITES_DO_EIXO_DOS_LADOS)
    eixo.set_ylabel(ROTULO_DO_EIXO_DO_RSD)
    # As contagens de injeção ficam no rótulo do eixo x, nunca no título. O rótulo nomeia a
    # unidade delas e repete que o `n` é de features.
    eixo.set_xlabel(
        f"{milhar(par.qcs.n_injecoes)} injeção(ões) de {par.qcs.rotulo} · "
        f"{milhar(par.amostras.n_injecoes)} de {par.amostras.rotulo} — o «n» acima é de FEATURES"
    )
    eixo.set_title(
        " · ".join(frase_do_lado_do_rsd(lado) for lado in par.lados), fontsize=TAMANHO_DA_FONTE
    )

    # A legenda leva uma entrada por lado desenhado. Lado ausente não ganha entrada.
    legenda_fora(eixo, provas)
    figura.suptitle(TITULO_DO_VIOLINO)
    aplica(figura, recorte)
    return figura


def _desenha_o_violino(eixo: Axes, lado: LadoDoRsd, *, posicao: int, cor: Any) -> None:
    """Um lado: o contorno espelhado, a caixa de quartis e a mediana, três artistas rotulados.

    Os três saem do mesmo vetor, para a caixa e o contorno descreverem a mesma população.

    Não há guarda contra densidade máxima zero: o núcleo gaussiano é estritamente positivo e a
    grelha cobre `[min, max]` do próprio dado.
    """
    valores = lado.percentagens
    if lado.tem_densidade:
        grelha, densidade = curva_da_densidade(valores, margem_em_bandas=SEM_MARGEM)
        largura = LARGURA_DO_VIOLINO * (densidade / float(densidade.max()))
        eixo.plot(
            np.concatenate([posicao + largura, (posicao - largura)[::-1], [posicao + largura[0]]]),
            np.concatenate([grelha, grelha[::-1], [grelha[0]]]),
            color=cor,
            label=rotulo_do_rsd(lado.rotulo, PAPEL_DA_DENSIDADE),
        )

    q1, mediana, q3 = (float(v) for v in np.quantile(valores, QUARTIS))
    meia = LARGURA_DA_CAIXA_DE_QUARTIS
    eixo.plot(
        [posicao - meia, posicao + meia, posicao + meia, posicao - meia, posicao - meia],
        [q1, q1, q3, q3, q1],
        color=cor,
        label=rotulo_do_rsd(lado.rotulo, PAPEL_DOS_QUARTIS),
    )
    eixo.plot(
        [posicao - meia, posicao + meia],
        [mediana, mediana],
        color=cor,
        linewidth=2.0,
        label=rotulo_do_rsd(lado.rotulo, PAPEL_DA_MEDIANA),
    )

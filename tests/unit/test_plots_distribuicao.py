"""As quatro figuras de antes-e-depois: o que desenham, o que declaram, e o que recusam.

## O que este módulo prova, e o que fica noutro sítio

Aqui está o **construtor puro**: arrays entram, `Figure` sai, e as quatro classes do arnês leem o
objeto. A **ligação** — que o corpo de `/execucoes` chega ao painel, que o painel escolhe o
construtor certo e que a lista que ele alimenta não diverge do registo — está em
`tests/test_desktop_figuras.py`. A **amostragem** é do servidor e está provada em
`tests/test_api_rotas.py`: a semente, o teto de 50 e a borda de menos de 50 features são
propriedade de `_resumo_das_distribuicoes`, não destes construtores.

## A âncora de separação desta unidade, e ela é o eixo do módulo

> **Uma planta sobre matriz já normalizada não separa nada.**

Antes e depois de uma normalização que não fez nada são o mesmo desenho, e toda asserção sobre
*"a figura mostra o efeito"* passaria por coincidência. Então o fixture desta unidade nasce com o
«antes» e o «depois» **medidamente diferentes** — `ANCORA_DA_SEPARACAO` afirma-o antes de qualquer
teste o usar —, e a mediana de cada lado é a que o deslocamento conhecido produz.

## Por que o número de artistas do `bxp` é enumerado e não contado por alto

`Axes.bxp` desenha **seis** `Line2D` por caixa — a caixa, a mediana, dois bigodes e duas pontas —,
e a figura rotula cada uma com o nome da feature. A classe (a) do arnês lê `ax.lines` inteiro,
então a expectativa tem de enumerar as seis; e é bom que tenha, porque é essa enumeração que fixa
**qual** número está em cada artista. Uma mediana desenhada no `q1` passa por qualquer asserção
mais frouxa.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pytest

from refino.api.esquemas import COEFICIENTE_DE_TUKEY
from refino.io.readers import load_matrix
from refino.plots.base import MatrizVaziaError, nova_figura, vetor_do_fio
from refino.plots.distribuicao import (
    ANTES,
    DEPOIS,
    ROTULO_DA_DENSIDADE,
    ROTULO_DAS_CAIXAS,
    SOBREVIVERAM,
    SUBSTANTIVO_DAS_AMOSTRAS,
    SUBSTANTIVO_DAS_FEATURES,
    TITULO_DA_DENSIDADE_POR_AMOSTRA,
    TITULO_DA_DENSIDADE_POR_FEATURE,
    TITULO_DAS_CAIXAS,
    TITULO_DAS_CAIXAS_POR_AMOSTRA,
    UNIDADE_POR_AMOSTRA,
    UNIDADE_POR_FEATURE,
    Caixa,
    CaixaDeAmostra,
    CorpoDaCaixa,
    DensidadeIndefinidaError,
    ParDesalinhadoError,
    caixas_do_fio,
    caixas_por_amostra_do_fio,
    figura_da_densidade_por_amostra,
    figura_da_densidade_por_feature,
    figura_das_caixas_por_amostra,
    figura_das_caixas_por_feature,
    frase_da_densidade,
    frase_das_caixas,
    recorte_da_densidade_por_amostra,
    recorte_da_densidade_por_feature,
    recorte_das_caixas,
    recorte_das_caixas_por_amostra,
    rotulo_do_artista,
    rotulos_cabem,
    rotulos_cabem_na_altura,
)
from refino.plots.recorte import Recorte, milhar
from tests.arnes_de_figura import (
    confere_numeros,
    confere_rotulos,
    confere_series,
    confere_vista,
    planta_corte_de_eixo,
    planta_eixos_trocados,
    planta_serie_removida,
    planta_sinal_invertido,
    sem_a_matriz,
    series,
    texto_corrido,
)
from tests.conftest import measurements_path, sample_list_path

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Sequence

    from matplotlib.figure import Figure

# As duas pontas da legenda — `N de M features, as que sobreviveram ao corte`.
# São literais aqui de propósito: é a fonte independente da classe (d) do arnês.
COMPARADAS: Final = 5801
ANTES_TOTAL: Final = 6035
AMOSTRAS: Final = 68
SEMENTE: Final = 4242
TETO: Final = 50

MATRIZ_ANTES: Final = "ingerida"
MATRIZ_DEPOIS: Final = "processada"


# ---------------------------------------------------------------------------
# O fixture, e a âncora de separação que o torna utilizável
# ---------------------------------------------------------------------------


def _cinco(centro: float, largura: float, fora: int | None) -> dict[str, Any]:
    """Os cinco números e a contagem, **num só sítio para as duas caixas**.

    Duas aritméticas iguais nos dois construtores de fixture divergiriam no dia em que uma delas
    mudasse a forma da caixa, e a irmã não teria motivo nenhum para ser reexecutada.
    """
    return {
        "bigode_inferior": centro - 2.0 * largura,
        "q1": centro - largura,
        "mediana": centro,
        "q3": centro + largura,
        "bigode_superior": centro + 2.0 * largura,
        "n_discrepantes": fora,
    }


def caixa(nome: str, centro: float, largura: float, fora: int | None = 0) -> Caixa:
    """Uma caixa de Tukey bem formada à volta de `centro`, de uma FEATURE."""
    return Caixa(feature=nome, **_cinco(centro, largura, fora))


def caixa_de_amostra(
    nome: str, centro: float, largura: float, fora: int | None = 0
) -> CaixaDeAmostra:
    """A mesma caixa, de uma AMOSTRA. O que muda é o campo do nome, e é só isso que muda."""
    return CaixaDeAmostra(amostra=nome, **_cinco(centro, largura, fora))


NOMES: Final = tuple(f"{2.0 + i / 10.0:.2f}_{810.0 + i:.4f}m/z" for i in range(6))

# **O deslocamento é CONHECIDO, e é ele que a figura tem de mostrar.** O «antes» está à volta de
# 5.000 com largura 1.000; o «depois» à volta de 0 com largura 1 — que é o que um `log10` seguido
# de `auto` faz. As medianas do «depois» NÃO são as do «antes», e a asserção abaixo é a que impede
# este fixture de ser uma matriz já normalizada disfarçada.
CAIXAS_ANTES: Final = tuple(
    caixa(nome, 5_000.0 + 250.0 * i, 1_000.0 + 100.0 * i) for i, nome in enumerate(NOMES)
)
CAIXAS_DEPOIS: Final = tuple(
    caixa(nome, 0.1 * i - 0.25, 1.0 + 0.05 * i) for i, nome in enumerate(NOMES)
)

MEDIAS_ANTES: Final = np.array([4_800.0, 5_100.0, 5_400.0, 6_000.0, 4_200.0, 5_900.0, 7_100.0])
MEDIAS_DEPOIS: Final = np.array([-0.9, -0.3, 0.1, 0.4, -1.2, 0.8, 1.4])

# ---------------------------------------------------------------------------
# O fixture do QUARTO construtor — uma caixa por AMOSTRA
# ---------------------------------------------------------------------------
#
# **Os nomes são de amostra e não de feature, e a distinção é o eixo desta figura.** `RT_mz` é
# nome de sinal; o que rotula uma caixa desta figura é a injeção. Reusar `NOMES` aqui faria a
# figura por amostra desenhar nomes de feature — o rótulo certo sobre a quantidade errada, e
# exactamente o defeito que a caixa ter um tipo próprio existe para impedir.
#
# **O COMPRIMENTO é do corpus real, e isso é requisito e não realismo decorativo.** Os nomes de
# amostra dos quatro arquivos têm 37 a 43 caracteres — contra 16 a 17 de um `RT_mz` —, e é esse
# comprimento que decide se as marcas do eixo cabem na altura. Um fixture com nomes curtos deixaria
# a figura por amostra a exercitar o ramo que a de feature já exercita, e o defeito que a leitura
# da imagem montada encontrou não teria onde cair.
NOMES_DE_AMOSTRA: Final = (
    "HTLV_C18_PLASMA_MSe_NEG_0_ID99_Asymptomatic",
    "HTLV_C18_PLASMA_MSe_NEG_2_ID41_ATLL_00",
    "HTLV_C18_PLASMA_MSe_NEG_3_ID52_HAMTSP0",
    "HTLV_C18_PLASMA_MSe_NEG_4_ID15_NoHTLV0",
    "HTLV_C18_PLASMA_MSe_NEG_5_QC_00000000",
)

# O «antes» à volta de 4.500 e o «depois» à volta de 0 — o mesmo deslocamento conhecido da irmã, e
# com larguras próprias para que as duas figuras não sejam a mesma imagem com outro título.
CAIXAS_DE_AMOSTRA_ANTES: Final = tuple(
    caixa_de_amostra(nome, 4_500.0 + 400.0 * i, 900.0 + 150.0 * i)
    for i, nome in enumerate(NOMES_DE_AMOSTRA)
)
CAIXAS_DE_AMOSTRA_DEPOIS: Final = tuple(
    caixa_de_amostra(nome, 0.15 * i - 0.3, 1.2 + 0.08 * i)
    for i, nome in enumerate(NOMES_DE_AMOSTRA)
)

AMOSTRAS_COMPARADAS: Final = 68


def test_ancora_o_ANTES_e_o_DEPOIS_DIFEREM_de_facto() -> None:  # noqa: N802
    """**A âncora de separação, e ela corre ANTES de tudo o que a usa.**

    Uma planta sobre matriz já normalizada não separa nada: as duas metades da figura sairiam
    iguais, e toda asserção sobre *"o desenho mostra o que a normalização fez"* passaria por
    coincidência. O que se afirma aqui é que as duas quantidades que o fixture separa **divergem**,
    sobre o próprio dado do teste, e a conta é barata — calcular as duas e
    afirmar que diferem, antes de plantar.
    """
    medianas_antes = np.array([c.mediana for c in CAIXAS_ANTES])
    medianas_depois = np.array([c.mediana for c in CAIXAS_DEPOIS])
    assert not np.allclose(medianas_antes, medianas_depois), (
        "o «antes» e o «depois» do fixture têm as mesmas medianas: nada neste módulo separa"
    )
    # E a ordem de grandeza também difere — é o que obriga os dois lados a terem eixos próprios.
    assert float(np.abs(medianas_antes).mean()) > 100.0 * float(np.abs(medianas_depois).mean())
    assert not np.allclose(MEDIAS_ANTES, MEDIAS_DEPOIS)


def test_ancora_o_ANTES_e_o_DEPOIS_POR_AMOSTRA_DIFEREM_de_facto() -> None:  # noqa: N802
    """**A mesma âncora de separação, para o eixo das amostras — e ela é obrigatória e não
    simetria.**

    O fixture da figura por amostra é outro conjunto de números, e sobre ele a planta pode não
    separar mesmo que separe no da irmã. Sem esta linha, toda asserção sobre *"a `Sample View`
    mostra o que a normalização fez às amostras"* passaria por coincidência num fixture em que os
    dois lados coincidissem.
    """
    antes = np.array([c.mediana for c in CAIXAS_DE_AMOSTRA_ANTES])
    depois = np.array([c.mediana for c in CAIXAS_DE_AMOSTRA_DEPOIS])
    assert not np.allclose(antes, depois), (
        "o «antes» e o «depois» por amostra têm as mesmas medianas: nada nesta figura separa"
    )
    assert float(np.abs(antes).mean()) > 100.0 * float(np.abs(depois).mean())


def test_ancora_as_caixas_por_AMOSTRA_nao_sao_as_caixas_por_FEATURE() -> None:  # noqa: N802
    """**A âncora da TRANSPOSTA, e é ela que impede este bloco de medir nada.**

    Esta figura é a transposta da irmã e não uma redução dela. Um teste que passasse igualmente nas
    duas — porque os dois fixtures trazem os mesmos números com outro rótulo — não distinguiria a
    figura certa da figura que desenha o eixo errado com o título certo. Então: os **nomes** são
    disjuntos e os **cinco números** de cada caixa divergem, afirmado sobre o próprio dado do
    teste, antes de qualquer construtor correr.
    """
    nomes_de_feature = {c.nome for c in CAIXAS_ANTES}
    nomes_de_amostra = {c.nome for c in CAIXAS_DE_AMOSTRA_ANTES}
    assert nomes_de_feature and nomes_de_amostra, "âncora: nenhum dos dois conjuntos está vazio"
    assert not (nomes_de_feature & nomes_de_amostra), (
        "os dois fixtures partilham nome: trocar um eixo pelo outro deixaria de ser detetável"
    )
    quintetos_de_feature = {
        (c.bigode_inferior, c.q1, c.mediana, c.q3, c.bigode_superior) for c in CAIXAS_ANTES
    }
    quintetos_de_amostra = {
        (c.bigode_inferior, c.q1, c.mediana, c.q3, c.bigode_superior)
        for c in CAIXAS_DE_AMOSTRA_ANTES
    }
    assert not (quintetos_de_feature & quintetos_de_amostra), (
        "há caixas idênticas nos dois eixos — uma asserção sobre «as caixas certas» passaria "
        "sobre o eixo errado"
    )


def rodape(figura: Figure) -> Any:
    """O `Text` do recorte, e **não** `figura.texts[0]`.

    **A primeira redação deste módulo lia `texts[0]`, e isso é o `suptitle`.** As três figuras
    desta unidade põem um título de figura antes de chamar `aplica`, então o rodapé é o **último**
    texto; a de `deriva.py` não tem `suptitle`, e lá `texts[0]` calha ser o rodapé. Ler pela
    posição faria este arnês medir o título e chamar-lhe rodapé — o veredito seria sobre o
    instrumento, e ele reprovou com *"os dois textos sobrepõem-se"* sobre uma figura sã.

    A leitura é por **propriedade** e não por posição: o rodapé é o único texto de nível de figura
    ancorado em baixo, e a asserção de unicidade é o que impede a busca de casar outra coisa.
    """
    achados = [t for t in figura.texts if t.get_va() == "bottom"]
    assert len(achados) == 1, f"{len(achados)} textos ancorados em baixo: {achados}"
    return achados[0]


def recorte_de_prova(*, amostrada: bool = True, n_caixas: int = TETO) -> Recorte:
    return recorte_das_caixas(
        n_caixas=n_caixas,
        n_comparadas=COMPARADAS,
        n_antes=ANTES_TOTAL,
        amostrada=amostrada,
        seed=SEMENTE,
        teto_de_caixas=TETO,
        coeficiente_de_tukey=COEFICIENTE_DE_TUKEY,
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )


def figura_das_caixas() -> Figure:
    return figura_das_caixas_por_feature(
        antes=CAIXAS_ANTES,
        depois=CAIXAS_DEPOIS,
        recorte=recorte_de_prova(n_caixas=len(NOMES)),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )


def recorte_de_prova_por_amostra(*, amostrada: bool = True, n_caixas: int = TETO) -> Recorte:
    return recorte_das_caixas_por_amostra(
        n_caixas=n_caixas,
        n_amostras_comparadas=AMOSTRAS_COMPARADAS,
        n_comparadas=COMPARADAS,
        n_antes=ANTES_TOTAL,
        amostrada=amostrada,
        seed=SEMENTE,
        teto_de_caixas=TETO,
        coeficiente_de_tukey=COEFICIENTE_DE_TUKEY,
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )


def figura_das_caixas_de_amostra() -> Figure:
    return figura_das_caixas_por_amostra(
        antes=CAIXAS_DE_AMOSTRA_ANTES,
        depois=CAIXAS_DE_AMOSTRA_DEPOIS,
        recorte=recorte_de_prova_por_amostra(n_caixas=len(NOMES_DE_AMOSTRA)),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )


def figura_da_densidade() -> Figure:
    return figura_da_densidade_por_feature(
        antes=MEDIAS_ANTES,
        depois=MEDIAS_DEPOIS,
        recorte=recorte_da_densidade_por_feature(
            n_comparadas=COMPARADAS,
            n_antes=ANTES_TOTAL,
            matriz_antes=MATRIZ_ANTES,
            matriz_depois=MATRIZ_DEPOIS,
        ),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )


def esperadas_do_lado(
    caixas: Sequence[CorpoDaCaixa],
) -> dict[str, tuple[list[float], list[float]]]:
    """O dado de CADA artista que `bxp` desenha, derivado das caixas de entrada.

    Seis por caixa. Os números vêm do `Caixa` que entrou; a geometria — o `x` de cada segmento —
    vem da convenção do `bxp`, que numera as posições de 1 em diante. É a classe (a) do arnês no
    grau mais fino que ela permite: a mediana desenhada no `q1` reprova aqui.
    """
    esperadas: dict[str, tuple[list[float], list[float]]] = {}
    for i, c in enumerate(caixas):
        posicao = float(i + 1)
        meia = 0.25  # metade da largura padrão de `bxp`, que é 0,5
        esperadas[f"{c.nome} · caixa"] = (
            [posicao - meia, posicao + meia, posicao + meia, posicao - meia, posicao - meia],
            [c.q1, c.q1, c.q3, c.q3, c.q1],
        )
        esperadas[f"{c.nome} · mediana"] = (
            [posicao - meia, posicao + meia],
            [c.mediana, c.mediana],
        )
        esperadas[f"{c.nome} · bigode inferior"] = (
            [posicao, posicao],
            [c.q1, c.bigode_inferior],
        )
        esperadas[f"{c.nome} · bigode superior"] = (
            [posicao, posicao],
            [c.q3, c.bigode_superior],
        )
        esperadas[f"{c.nome} · ponta inferior"] = (
            [posicao - meia / 2.0, posicao + meia / 2.0],
            [c.bigode_inferior, c.bigode_inferior],
        )
        esperadas[f"{c.nome} · ponta superior"] = (
            [posicao - meia / 2.0, posicao + meia / 2.0],
            [c.bigode_superior, c.bigode_superior],
        )
    return esperadas


# ---------------------------------------------------------------------------
# 1. Caminho feliz — as quatro classes do arnês, nos dois lados
# ---------------------------------------------------------------------------


def test_as_caixas_passam_nas_quatro_classes_nos_DOIS_lados() -> None:  # noqa: N802
    """A execução com a planta removida, e é ela que torna as quatro seguintes atribuíveis."""
    figura = figura_das_caixas()
    assert len(figura.axes) == 2, "a figura tem um eixo por lado"

    for eixo, caixas, lado, matriz in (
        (figura.axes[0], CAIXAS_ANTES, ANTES, MATRIZ_ANTES),
        (figura.axes[1], CAIXAS_DEPOIS, DEPOIS, MATRIZ_DEPOIS),
    ):
        esperadas = esperadas_do_lado(caixas)
        assert len(series(eixo)) == len(esperadas) == 6 * len(caixas), (
            f"âncora: {len(series(eixo))} artistas desenhados contra {len(esperadas)} esperados"
        )
        assert confere_series(eixo, esperadas) == []
        assert (
            confere_rotulos(
                eixo,
                x=f"{milhar(len(caixas))} features, uma caixa cada",
                y=ROTULO_DAS_CAIXAS,
                titulo=f"{lado} — {matriz}\n{frase_das_caixas(len(caixas), 0, 0, 0)}",
            )
            == []
        )
        assert confere_vista(eixo) == []

    assert sem_a_matriz(figura.get_suptitle()) == TITULO_DAS_CAIXAS
    assert confere_numeros(figura, numeros_das_caixas()) == []


def numeros_das_caixas() -> list[str]:
    """Todo número que a figura das caixas declara, derivado por caminho independente.

    As duas pontas da legenda saem das constantes deste módulo; as contagens saem do **fixture**,
    não do construtor. `1.5` é o coeficiente de Tukey, e ele entra aqui porque o rodapé o cita —
    e porque a única fonte dele é `esquemas.COEFICIENTE_DE_TUKEY`, que o painel injeta.
    """
    seis = str(len(NOMES))
    return [
        # o rodapé: `N de M features`, a semente, o teto e o coeficiente
        f"{COMPARADAS:,}".replace(",", "."),
        f"{ANTES_TOTAL:,}".replace(",", "."),
        seis,
        f"{COMPARADAS:,}".replace(",", "."),
        str(SEMENTE),
        str(TETO),
        "1.5",
        "1.",
        "3.",
        # o título de cada lado: a contagem de caixas, duas vezes
        seis,
        seis,
        # o rótulo do eixo x de cada lado, duas vezes
        seis,
        seis,
    ]


def test_a_densidade_passa_nas_quatro_classes_nos_DOIS_lados() -> None:  # noqa: N802
    """A curva desenhada é a que a estimativa produz, e o eixo declara a unidade certa."""
    figura = figura_da_densidade()
    assert len(figura.axes) == 2

    for eixo, valores, lado, matriz in (
        (figura.axes[0], MEDIAS_ANTES, ANTES, MATRIZ_ANTES),
        (figura.axes[1], MEDIAS_DEPOIS, DEPOIS, MATRIZ_DEPOIS),
    ):
        desenhadas = series(eixo)
        assert len(desenhadas) == 1, f"uma curva por lado, e há {len(desenhadas)}"
        curva = desenhadas[0]
        assert curva.rotulo == f"{lado} — {matriz}"
        # (a) o dado: a grelha cobre o intervalo dos valores, com a margem de `cut = 3`.
        assert float(curva.x.min()) < float(valores.min())
        assert float(curva.x.max()) > float(valores.max())
        assert bool((curva.y >= 0.0).all()), "densidade negativa não existe"
        # (c) a vista: os pontos desenhados cabem nos limites, e a escala é linear.
        assert confere_vista(eixo) == []
        assert (
            confere_rotulos(
                eixo,
                x=UNIDADE_POR_FEATURE,
                y=ROTULO_DA_DENSIDADE,
                titulo=f"{lado} — {matriz}\n{frase_da_densidade(valores.size, 0)}",
            )
            == []
        )

    assert sem_a_matriz(figura.get_suptitle()) == TITULO_DA_DENSIDADE_POR_FEATURE


def test_a_densidade_por_AMOSTRA_diz_amostra_em_todo_o_lado_onde_diz_feature() -> None:  # noqa: N802
    """**Numa legenda, o rótulo do eixo tem de nomear a quantidade que está no eixo.**

    As duas densidades desenham a mesma curva com o mesmo código; o que as separa é o **título**, a
    **unidade do eixo** e a cláusula de amostragem. Uma delas com o texto da outra produziria uma
    figura correta sobre a pergunta errada, e nada levantaria.
    """
    figura = figura_da_densidade_por_amostra(
        antes=MEDIAS_ANTES,
        depois=MEDIAS_DEPOIS,
        recorte=recorte_da_densidade_por_amostra(
            n_comparadas=COMPARADAS,
            n_antes=ANTES_TOTAL,
            n_amostras=AMOSTRAS,
            matriz_antes=MATRIZ_ANTES,
            matriz_depois=MATRIZ_DEPOIS,
        ),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    assert sem_a_matriz(figura.get_suptitle()) == TITULO_DA_DENSIDADE_POR_AMOSTRA
    for eixo in figura.axes:
        assert eixo.get_xlabel() == UNIDADE_POR_AMOSTRA
        assert eixo.get_xlabel() != UNIDADE_POR_FEATURE

    declarado = rodape(figura).get_text()
    assert "amostras" in declarado, declarado
    # E o denominador do rodapé continua a ser em FEATURES, porque é sobre elas que cada média foi
    # calculada. A pergunta — *qual é a população em que este número foi calculado?* —
    # tem duas respostas aqui, e as duas estão escritas.
    assert f"{milhar(COMPARADAS)} de {milhar(ANTES_TOTAL)} features" in declarado.replace("\n", " ")
    assert milhar(AMOSTRAS) in declarado


# ---------------------------------------------------------------------------
# 1b. A QUARTA figura — uma caixa por AMOSTRA, que é a transposta e não uma redução
# ---------------------------------------------------------------------------


def numeros_das_caixas_por_amostra() -> list[str]:
    """Todo número que a figura por amostra declara, derivado por caminho independente.

    O rodapé desta figura tem **duas** populações e as duas aparecem: o par `N de M features` é o
    que está dentro de cada caixa, e o `K de L amostras` é quantas caixas há. As contagens dos
    títulos saem do fixture, não do construtor.
    """
    cinco = str(len(NOMES_DE_AMOSTRA))
    return [
        # o rodapé: `N de M features`, o par de amostras, a semente, o teto e o coeficiente
        f"{COMPARADAS:,}".replace(",", "."),
        f"{ANTES_TOTAL:,}".replace(",", "."),
        cinco,
        str(AMOSTRAS_COMPARADAS),
        str(SEMENTE),
        str(TETO),
        f"{COMPARADAS:,}".replace(",", "."),
        "1.5",
        "1.",
        "3.",
        # o título de cada lado: a contagem de caixas, duas vezes
        cinco,
        cinco,
    ]


def test_as_caixas_por_AMOSTRA_passam_nas_quatro_classes_nos_DOIS_lados() -> None:  # noqa: N802
    """A execução com a planta removida, e é ela que torna as plantas seguintes atribuíveis."""
    figura = figura_das_caixas_de_amostra()
    assert len(figura.axes) == 2, "a figura tem um eixo por lado"

    for eixo, caixas, lado, matriz in (
        (figura.axes[0], CAIXAS_DE_AMOSTRA_ANTES, ANTES, MATRIZ_ANTES),
        (figura.axes[1], CAIXAS_DE_AMOSTRA_DEPOIS, DEPOIS, MATRIZ_DEPOIS),
    ):
        esperadas = esperadas_do_lado(caixas)
        assert len(series(eixo)) == len(esperadas) == 6 * len(caixas), (
            f"âncora: {len(series(eixo))} artistas desenhados contra {len(esperadas)} esperados"
        )
        assert confere_series(eixo, esperadas) == []
        assert (
            confere_rotulos(
                eixo,
                x=f"{milhar(len(caixas))} {SUBSTANTIVO_DAS_AMOSTRAS}, uma caixa cada",
                y=ROTULO_DAS_CAIXAS,
                titulo=f"{lado} — {matriz}\n{frase_das_caixas(len(caixas), 0, 0, 0)}",
            )
            == []
        )
        assert confere_vista(eixo) == []

    assert sem_a_matriz(figura.get_suptitle()) == TITULO_DAS_CAIXAS_POR_AMOSTRA
    assert confere_numeros(figura, numeros_das_caixas_por_amostra()) == []


def test_a_figura_por_AMOSTRA_diz_amostra_em_todo_o_lado_onde_a_irma_diz_feature() -> None:  # noqa: N802
    """**Na figura inteira, o substantivo tem de nomear a unidade que está no eixo.**

    As duas figuras de caixas desenham o mesmo objeto — uma caixa de Tukey por unidade — com o
    mesmo código, sobre eixos opostos da mesma matriz. O que as separa é o **título**, o
    **substantivo do eixo x** e a cláusula de amostragem. Uma delas com o texto da outra produziria
    uma figura correta sobre a pergunta errada, e nada levantaria.

    **E a metade que fecha a comparação é a linha do `desenhadas`:** o par `N de M features` do
    rodapé é o mesmo nas duas, e tem de ser — é a população *dentro* de cada caixa, e cada caixa
    por amostra é calculada sobre as features comuns. O que muda é a cláusula de amostragem.
    """
    por_amostra = figura_das_caixas_de_amostra()
    por_feature = figura_das_caixas()

    assert sem_a_matriz(por_amostra.get_suptitle()) == TITULO_DAS_CAIXAS_POR_AMOSTRA
    assert sem_a_matriz(por_amostra.get_suptitle()) != sem_a_matriz(por_feature.get_suptitle())
    for eixo in por_amostra.axes:
        assert eixo.get_xlabel().endswith(f"{SUBSTANTIVO_DAS_AMOSTRAS}, uma caixa cada")
        assert SUBSTANTIVO_DAS_FEATURES not in eixo.get_xlabel()

    declarado = " ".join(rodape(por_amostra).get_text().split())
    assert f"{milhar(AMOSTRAS_COMPARADAS)} amostras comuns" in declarado, declarado
    # O denominador do par continua a ser em FEATURES, porque é sobre elas que cada caixa foi
    # calculada — a pergunta pela população tem duas respostas aqui, e as duas estão escritas.
    assert f"{milhar(COMPARADAS)} de {milhar(ANTES_TOTAL)} features" in declarado, declarado
    assert f"cada caixa é sobre as {milhar(COMPARADAS)} features comuns" in declarado, declarado
    assert "por AMOSTRA" in declarado and "transposta" in declarado, declarado


def test_os_rotulos_do_eixo_da_figura_por_amostra_cabem_NA_ALTURA_ou_nao_saem() -> None:  # noqa: N802
    """**O achado da leitura da imagem montada, com os dois desfechos na mesma função.**

    Rodado a 90 graus, um rótulo ocupa a **altura** do seu comprimento. `rotulos_cabem` só olha a
    largura, e com cinco caixas ela dizia *"cabem"*: os nomes de amostra deste laboratório têm 37 a
    43 caracteres, o `tight_layout` não os acomodava, e as marcas saíam impressas **por cima do
    rodapé**. Nenhuma das quatro classes do arnês vê isso — colisão de dois textos é propriedade da
    imagem montada.

    **A planta cai sobre dado em que as duas coisas diferem**: os nomes longos são os reais
    do `neg_c18` e os curtos são o mesmo fixture com o nome encurtado, e as duas figuras têm o
    **mesmo** número de caixas — então o que muda o desfecho é o comprimento e não a contagem, que
    é a atribuição que este par existe para fixar.
    """
    figura = nova_figura()
    assert not rotulos_cabem_na_altura(figura, list(NOMES_DE_AMOSTRA)), (
        f"nomes de {max(len(n) for n in NOMES_DE_AMOSTRA)} caracteres foram dados por cabíveis"
    )
    assert rotulos_cabem_na_altura(figura, list(NOMES)), (
        f"nomes `RT_mz` de {max(len(n) for n in NOMES)} caracteres não cabem — o limiar deixou de "
        "distinguir os dois eixos, e nenhuma figura mostraria marca nenhuma"
    )
    # E a largura continua a dizer «cabem» para as cinco: sem esta linha, o desfecho de cima podia
    # vir do predicado antigo e a atribuição seria ao mecanismo errado.
    assert rotulos_cabem(figura, len(NOMES_DE_AMOSTRA)), "âncora: a LARGURA não é o que corta aqui"

    curtas = tuple(
        caixa_de_amostra(f"S{i}", c.mediana, c.q3 - c.mediana)
        for i, c in enumerate(CAIXAS_DE_AMOSTRA_ANTES)
    )
    assert len(curtas) == len(CAIXAS_DE_AMOSTRA_ANTES), "âncora: a planta muda o NOME e não a conta"
    com_nome_curto = figura_das_caixas_por_amostra(
        antes=curtas,
        depois=curtas,
        recorte=recorte_de_prova_por_amostra(n_caixas=len(curtas)),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    assert [t.get_text() for t in com_nome_curto.axes[0].get_xticklabels()] == [
        c.nome for c in curtas
    ], "com nome curto o eixo tem de mostrar as marcas"
    assert list(figura_das_caixas_de_amostra().axes[0].get_xticks()) == [], (
        "com os nomes reais de amostra as marcas ficaram no eixo, e elas imprimem por cima do "
        "rodapé na imagem montada"
    )


def test_os_artistas_da_figura_por_amostra_levam_o_nome_da_AMOSTRA() -> None:  # noqa: N802
    """R2 no eixo novo: a identificação por clique tem de dizer **qual amostra** é.

    **A planta é a que separa os dois eixos**, e ela cai sobre dado em que as duas coisas diferem:
    nenhum nome de amostra está no conjunto dos nomes de feature (a âncora acima o afirma), então
    um construtor que lesse o campo errado produziria rótulos vazios ou de feature.
    """
    figura = figura_das_caixas_de_amostra()
    rotulos = {s.rotulo.split(" · ")[0] for s in series(figura.axes[0])}
    assert rotulos == set(NOMES_DE_AMOSTRA), rotulos
    assert not (rotulos & set(NOMES)), "um artista leva nome de feature numa figura por amostra"


def test_par_causal_serie_removida_nas_caixas_por_amostra() -> None:
    figura = figura_das_caixas_de_amostra()
    eixo = figura.axes[0]
    esperadas = esperadas_do_lado(CAIXAS_DE_AMOSTRA_ANTES)
    assert confere_series(eixo, esperadas) == []
    rotulo = planta_serie_removida(eixo)
    achados = confere_series(eixo, esperadas)
    assert any(rotulo in a for a in achados), achados


def test_par_causal_eixos_trocados_nas_caixas_por_amostra() -> None:
    figura = figura_das_caixas_de_amostra()
    eixo = figura.axes[0]
    esperadas = esperadas_do_lado(CAIXAS_DE_AMOSTRA_ANTES)
    assert confere_series(eixo, esperadas) == []
    planta_eixos_trocados(eixo)
    assert confere_series(eixo, esperadas) != []


def test_par_causal_do_numero_declarado_na_figura_por_amostra() -> None:
    """A classe (d) no eixo novo: o rodapé é lido do ARTISTA, e um número trocado nele acusa."""
    figura = figura_das_caixas_de_amostra()
    assert confere_numeros(figura, numeros_das_caixas_por_amostra()) == []
    artista = rodape(figura)
    artista.set_text(artista.get_text().replace(milhar(AMOSTRAS_COMPARADAS), "999"))
    assert confere_numeros(figura, numeros_das_caixas_por_amostra()) != []


def test_o_par_de_AMOSTRAS_desalinhado_e_recusado_e_o_alinhado_nao() -> None:  # noqa: N802
    """A mesma guarda da irmã, no eixo novo — e as duas direções na mesma função.

    **A guarda vale aqui porque o servidor casa as amostras por NOME, como as features.** Sem ela,
    um chamador futuro que pareasse por posição desenharia o rótulo de uma amostra sobre a caixa de
    outra, e a imagem sairia bem formada.
    """
    trocado = (
        CAIXAS_DE_AMOSTRA_DEPOIS[1],
        CAIXAS_DE_AMOSTRA_DEPOIS[0],
        *CAIXAS_DE_AMOSTRA_DEPOIS[2:],
    )
    assert [c.nome for c in trocado] != [c.nome for c in CAIXAS_DE_AMOSTRA_ANTES], "âncora"
    with pytest.raises(ParDesalinhadoError, match="amostras diferentes"):
        figura_das_caixas_por_amostra(
            antes=CAIXAS_DE_AMOSTRA_ANTES,
            depois=trocado,
            recorte=recorte_de_prova_por_amostra(n_caixas=len(NOMES_DE_AMOSTRA)),
            matriz_antes=MATRIZ_ANTES,
            matriz_depois=MATRIZ_DEPOIS,
        )
    # A outra ponta: uma guarda que recusasse tudo passaria na metade de cima sozinha.
    assert len(figura_das_caixas_de_amostra().axes) == 2


def test_o_recorte_por_amostra_declara_QUE_NAO_SORTEOU_quando_cabe_tudo() -> None:  # noqa: N802
    """Os dois desfechos da cláusula de amostragem, no eixo novo.

    **Os dois eixos decidem independentemente**: um corpus de 40 amostras por 6.000
    features amostra num e não no outro, e uma bandeira só para os dois faria a figura declarar um
    recorte que ela não aplicou.
    """
    sorteou = recorte_de_prova_por_amostra(amostrada=True, n_caixas=TETO).amostragem
    nao_sorteou = recorte_de_prova_por_amostra(amostrada=False, n_caixas=40).amostragem
    assert "sorteadas com a semente" in sorteou and str(SEMENTE) in sorteou
    assert nao_sorteou.startswith("nenhuma —"), nao_sorteou
    assert "semente" not in nao_sorteou, nao_sorteou
    # E os dois declaram as DUAS populações: as amostras e as features de dentro de cada caixa.
    for frase in (sorteou, nao_sorteou):
        assert milhar(AMOSTRAS_COMPARADAS) in frase and milhar(COMPARADAS) in frase, frase
        assert SOBREVIVERAM in frase, frase


# ---------------------------------------------------------------------------
# 2. Os quatro pares causais do arnês, sobre as duas figuras
# ---------------------------------------------------------------------------


def test_par_causal_serie_removida_nas_caixas() -> None:
    figura = figura_das_caixas()
    eixo = figura.axes[0]
    esperadas = esperadas_do_lado(CAIXAS_ANTES)
    assert confere_series(eixo, esperadas) == []
    rotulo = planta_serie_removida(eixo)
    achados = confere_series(eixo, esperadas)
    assert any(rotulo in a for a in achados), achados


def test_par_causal_eixos_trocados_nas_caixas() -> None:
    figura = figura_das_caixas()
    eixo = figura.axes[0]
    esperadas = esperadas_do_lado(CAIXAS_ANTES)
    assert confere_series(eixo, esperadas) == []
    planta_eixos_trocados(eixo)
    assert confere_series(eixo, esperadas) != []


def test_par_causal_sinal_invertido_na_densidade() -> None:
    figura = figura_da_densidade()
    eixo = figura.axes[0]
    antes = series(eixo)[0]
    esperadas = {antes.rotulo: (antes.x, antes.y)}
    assert confere_series(eixo, esperadas) == []
    planta_sinal_invertido(eixo)
    assert confere_series(eixo, esperadas) != []


@pytest.mark.parametrize(
    "monta", [figura_das_caixas, figura_das_caixas_de_amostra, figura_da_densidade]
)
def test_par_causal_limite_que_corta_a_serie(monta: Any) -> None:
    """A planta que as três de dado não alcançam: a figura fica tecnicamente certa e em branco."""
    eixo = monta().axes[0]
    assert confere_vista(eixo) == []
    planta_corte_de_eixo(eixo)
    achados = confere_vista(eixo)
    assert any("A figura está em branco" in a for a in achados), achados


def test_par_causal_do_numero_declarado() -> None:
    """A classe (d): o rodapé é lido do ARTISTA, e um número trocado nele acusa."""
    figura = figura_das_caixas()
    assert confere_numeros(figura, numeros_das_caixas()) == []
    artista = rodape(figura)
    artista.set_text(artista.get_text().replace(milhar(COMPARADAS), "999"))
    assert confere_numeros(figura, numeros_das_caixas()) != []


# ---------------------------------------------------------------------------
# 3. O par desalinhado — o defeito que produz uma figura BEM FORMADA
# ---------------------------------------------------------------------------


def test_o_par_de_lados_com_features_DIFERENTES_e_recusado() -> None:  # noqa: N802
    """Mesma quantidade, ordem trocada: lado a lado seria o rótulo certo sobre outra feature."""
    trocado = (CAIXAS_DEPOIS[1], CAIXAS_DEPOIS[0], *CAIXAS_DEPOIS[2:])
    assert [c.feature for c in trocado] != [c.feature for c in CAIXAS_ANTES], "âncora da planta"
    with pytest.raises(ParDesalinhadoError, match="posições trazem features diferentes"):
        figura_das_caixas_por_feature(
            antes=CAIXAS_ANTES,
            depois=trocado,
            recorte=recorte_de_prova(n_caixas=len(NOMES)),
            matriz_antes=MATRIZ_ANTES,
            matriz_depois=MATRIZ_DEPOIS,
        )


def test_o_par_de_lados_com_COMPRIMENTOS_diferentes_e_recusado() -> None:  # noqa: N802
    """A outra direção da guarda, e ela tem mensagem própria.

    Comprimentos diferentes são outro defeito do lado do servidor.
    """
    with pytest.raises(ParDesalinhadoError, match="caixa\\(s\\) e o «depois»"):
        figura_das_caixas_por_feature(
            antes=CAIXAS_ANTES,
            depois=CAIXAS_DEPOIS[:-1],
            recorte=recorte_de_prova(n_caixas=len(NOMES)),
            matriz_antes=MATRIZ_ANTES,
            matriz_depois=MATRIZ_DEPOIS,
        )


def test_o_par_ALINHADO_nao_e_recusado() -> None:  # noqa: N802
    """A outra ponta: sem isto, uma guarda que recusasse tudo passaria nos dois testes acima."""
    assert len(figura_das_caixas().axes) == 2


# ---------------------------------------------------------------------------
# 4. As recusas tipadas — população vazia, e densidade sem dispersão
# ---------------------------------------------------------------------------


def test_lado_sem_caixa_desenhavel_e_MATRIZ_VAZIA() -> None:  # noqa: N802
    """`MatrizVaziaError` é a única que `PainelDeFigura` apanha para trocar a figura por texto."""
    nao_finitas = tuple(
        Caixa(c.feature, np.nan, np.nan, np.nan, np.nan, np.nan, None) for c in CAIXAS_DEPOIS
    )
    with pytest.raises(MatrizVaziaError, match="nenhuma com os cinco números finitos"):
        figura_das_caixas_por_feature(
            antes=CAIXAS_ANTES,
            depois=nao_finitas,
            recorte=recorte_de_prova(n_caixas=len(NOMES)),
            matriz_antes=MATRIZ_ANTES,
            matriz_depois=MATRIZ_DEPOIS,
        )


@pytest.mark.parametrize(
    ("valores", "trecho"),
    [
        ([], "não trouxe média nenhuma"),
        ([1.0], "não há dispersão"),
        ([2.5, 2.5, 2.5], "não há dispersão"),
        ([np.nan, np.nan], "não há dispersão"),
    ],
)
def test_a_densidade_sem_dispersao_RECUSA_em_vez_de_desenhar_uma_reta(  # noqa: N802
    valores: list[float], trecho: str
) -> None:
    """**Uma reta desenhada aqui afirmaria distribuição uniforme sobre o que não tem nenhuma.**

    E as duas recusas são classes **diferentes**: *"não veio população"* é defeito de
    quem chama e *"a população veio e é constante"* é fato do dado.
    """
    esperada = MatrizVaziaError if not valores else DensidadeIndefinidaError
    with pytest.raises(esperada, match=trecho):
        figura_da_densidade_por_feature(
            antes=np.asarray(valores, dtype=float),
            depois=MEDIAS_DEPOIS,
            recorte=recorte_da_densidade_por_feature(
                n_comparadas=COMPARADAS,
                n_antes=ANTES_TOTAL,
                matriz_antes=MATRIZ_ANTES,
                matriz_depois=MATRIZ_DEPOIS,
            ),
            matriz_antes=MATRIZ_ANTES,
            matriz_depois=MATRIZ_DEPOIS,
        )


def test_a_densidade_com_DOIS_valores_distintos_desenha() -> None:  # noqa: N802
    """A outra ponta da guarda, e ela fixa o mínimo: dois distintos já têm dispersão."""
    figura = figura_da_densidade_por_feature(
        antes=np.array([1.0, 2.0]),
        depois=MEDIAS_DEPOIS,
        recorte=recorte_da_densidade_por_feature(
            n_comparadas=COMPARADAS,
            n_antes=ANTES_TOTAL,
            matriz_antes=MATRIZ_ANTES,
            matriz_depois=MATRIZ_DEPOIS,
        ),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    assert len(series(figura.axes[0])) == 1


def test_o_nao_finito_da_densidade_e_CONTADO_e_nao_apagado() -> None:  # noqa: N802
    """R6 na superfície da curva: o que sai da estimativa aparece no título."""
    com_buraco = np.array([*MEDIAS_ANTES.tolist(), np.nan, np.inf])
    figura = figura_da_densidade_por_feature(
        antes=com_buraco,
        depois=MEDIAS_DEPOIS,
        recorte=recorte_da_densidade_por_feature(
            n_comparadas=COMPARADAS,
            n_antes=ANTES_TOTAL,
            matriz_antes=MATRIZ_ANTES,
            matriz_depois=MATRIZ_DEPOIS,
        ),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    titulo = figura.axes[0].get_title()
    assert "2 não finito(s)" in titulo, titulo
    assert f"{MEDIAS_ANTES.size} valor(es)" in titulo, titulo


# ---------------------------------------------------------------------------
# 5. A contagem de discrepantes — `None` e `0` NÃO colapsam
# ---------------------------------------------------------------------------


def test_a_frase_das_caixas_tem_os_QUATRO_desfechos() -> None:  # noqa: N802
    """Quatro contagens, quatro desfechos, e nenhum deles absorve outro.

    **`0` sai SEMPRE**, porque *"nenhum ponto fora dos bigodes"* é um fato: uma frase que o
    omitisse faria a ausência de aviso significar duas coisas.
    """
    assert frase_das_caixas(50, 0, 0, 0) == "50 caixas; nenhum ponto fora dos bigodes"
    assert "12 ponto(s) fora dos bigodes" in frase_das_caixas(50, 12, 0, 0)
    assert "3 sem valor em amostra nenhuma" in frase_das_caixas(50, 0, 3, 0)
    assert "4 sem caixa desenhável" in frase_das_caixas(50, 0, 0, 4)
    # As duas últimas coincidem no nosso servidor e continuam SEPARADAS na frase.
    das_duas = frase_das_caixas(46, 7, 4, 4)
    assert "sem valor em amostra nenhuma" in das_duas and "sem caixa desenhável" in das_duas
    # E uma caixa só tem singular — o plural errado numa figura é o que se lê primeiro.
    assert frase_das_caixas(1, 0, 0, 0).startswith("1 caixa;")


def test_o_None_de_n_discrepantes_NAO_vira_zero_na_figura() -> None:  # noqa: N802
    """**A distinção que o esquema pagou para carregar sobrevive ao desenho.**

    Uma coluna sem valor nenhum chega com `n_discrepantes = None` — *"não havia sobre o que
    decidir"* —, e é fato diferente de *"nenhum ponto ficou fora"*. Se o construtor a somasse como
    zero, as duas ficariam indistinguíveis na contagem que a figura publica.
    """
    com_nulo = (caixa(NOMES[0], 1.0, 0.5, None), *[caixa(n, 1.0, 0.5, 0) for n in NOMES[1:]])
    figura = figura_das_caixas_por_feature(
        antes=com_nulo,
        depois=com_nulo,
        recorte=recorte_de_prova(n_caixas=len(NOMES)),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    # **Corrido, e não como sai do artista.** `recorte.aplica`
    # requebra o título pela largura real do eixo, e num eixo de meia figura estas duas frases
    # atravessam a quebra: a asserção diria «não está lá» sobre um título que a contém inteira.
    titulo = texto_corrido(figura.axes[0].get_title())
    assert "1 sem valor em amostra nenhuma" in titulo, titulo
    assert "nenhum ponto fora dos bigodes" in titulo, titulo

    # A outra ponta: sem o `None`, a cláusula não aparece — senão ela seria decorativa.
    todas_com_zero = tuple(caixa(n, 1.0, 0.5, 0) for n in NOMES)
    outra = figura_das_caixas_por_feature(
        antes=todas_com_zero,
        depois=todas_com_zero,
        recorte=recorte_de_prova(n_caixas=len(NOMES)),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    assert "sem valor em amostra nenhuma" not in texto_corrido(outra.axes[0].get_title())


# ---------------------------------------------------------------------------
# 6. O rótulo de cada artista — R2, e o índice aos pares
# ---------------------------------------------------------------------------


def test_o_rotulo_do_artista_mapeia_o_par_de_bigodes_para_a_caixa_CERTA() -> None:  # noqa: N802
    """**O defeito que o `// 2` evita, e ele começa na SEGUNDA caixa.**

    `bxp` desenha dois bigodes e duas pontas por caixa, e um mapeamento por índice direto daria a
    feature errada a partir da segunda — com o rótulo a sair certo na primeira, que é onde alguém
    olharia para conferir.
    """
    assert rotulo_do_artista(CAIXAS_ANTES, "boxes", 0) == f"{NOMES[0]} · caixa"
    assert rotulo_do_artista(CAIXAS_ANTES, "medians", 2) == f"{NOMES[2]} · mediana"
    assert rotulo_do_artista(CAIXAS_ANTES, "whiskers", 0) == f"{NOMES[0]} · bigode inferior"
    assert rotulo_do_artista(CAIXAS_ANTES, "whiskers", 1) == f"{NOMES[0]} · bigode superior"
    # A planta: o índice 2 é o bigode inferior da SEGUNDA caixa, não da terceira.
    assert rotulo_do_artista(CAIXAS_ANTES, "whiskers", 2) == f"{NOMES[1]} · bigode inferior"
    assert rotulo_do_artista(CAIXAS_ANTES, "caps", 3) == f"{NOMES[1]} · ponta superior"


def test_todo_artista_das_caixas_leva_o_nome_de_UMA_feature() -> None:  # noqa: N802
    """R2 pela via do rótulo: `PainelDeFigura` cai nele quando não há vetor de nomes."""
    eixo = figura_das_caixas().axes[0]
    rotulos = {s.rotulo for s in series(eixo)}
    assert rotulos, "âncora: nenhum artista lido"
    sem_nome = sorted(r for r in rotulos if not any(n in r for n in NOMES))
    assert not sem_nome, f"artistas sem o nome da feature no rótulo: {sem_nome}"


# ---------------------------------------------------------------------------
# 7. Os rótulos do eixo cabem, ou não — DERIVADO, e não um limiar escolhido
# ---------------------------------------------------------------------------


def test_rotulos_cabem_tem_os_DOIS_desfechos_e_sai_da_LARGURA_da_figura() -> None:  # noqa: N802
    """**A alternativa ao limiar sem origem**, e ela tem de separar nos dois sentidos.

    O mesmo número de caixas dá respostas diferentes em figuras de larguras diferentes — que é a
    prova de que a conta é sobre a figura e não um literal disfarçado.
    """
    estreita, larga = nova_figura(largura=4.0), nova_figura(largura=20.0)
    assert rotulos_cabem(larga, 50) is True
    assert rotulos_cabem(estreita, 50) is False
    # E o ponto de viragem acompanha a largura: a estreita comporta menos caixas que a larga.
    cabem_na_estreita = max(n for n in range(1, 400) if rotulos_cabem(estreita, n))
    cabem_na_larga = max(n for n in range(1, 400) if rotulos_cabem(larga, n))
    assert cabem_na_larga > cabem_na_estreita, (cabem_na_larga, cabem_na_estreita)


def test_com_poucas_caixas_o_eixo_MOSTRA_os_nomes_e_com_muitas_NAO() -> None:  # noqa: N802
    """O par causal na figura montada: o mesmo construtor, dois desfechos de marca de eixo."""
    poucas = figura_das_caixas()
    assert [t.get_text() for t in poucas.axes[0].get_xticklabels()] == list(NOMES)

    muitas_caixas = tuple(
        caixa(f"{2.0 + i / 100.0:.2f}_{810.0 + i:.4f}m/z", 1.0 + i, 0.5) for i in range(TETO)
    )
    muitas = figura_das_caixas_por_feature(
        antes=muitas_caixas,
        depois=muitas_caixas,
        recorte=recorte_de_prova(),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    assert list(muitas.axes[0].get_xticks()) == [], (
        "com 50 caixas os nomes não cabem, e uma barra preta de texto afirma informação que não "
        "está legível — quem responde é a identificação por clique (R2)"
    )


# ---------------------------------------------------------------------------
# 8. O recorte — as duas pontas da legenda, a semente, e a guarda de fronteira
# ---------------------------------------------------------------------------


def test_o_recorte_das_caixas_declara_as_DUAS_pontas_e_a_semente() -> None:  # noqa: N802
    """A legenda diz *"N de M features, as que sobreviveram ao corte"*."""
    texto = recorte_de_prova().texto()
    assert f"{milhar(COMPARADAS)} de {milhar(ANTES_TOTAL)} features" in texto
    assert SOBREVIVERAM in texto
    assert f"semente {SEMENTE}" in texto
    assert f"teto de {TETO} caixas" in texto
    assert MATRIZ_ANTES in texto and MATRIZ_DEPOIS in texto
    assert "1,5" in texto or "1.5" in texto, texto


def test_amostrada_vem_do_FIO_e_nao_do_proxy_n_caixas_menor() -> None:  # noqa: N802
    """**O proxy `n_caixas < n_comparadas` mente na borda, e o esquema mede-o.**

    Com exactamente 50 features comparáveis não houve sorteio, e o proxy diria o mesmo que diz
    para 6.035. Este teste passa `amostrada=True` com `n_caixas == n_comparadas`: um construtor que
    derivasse do proxy escreveria *"nenhuma"* onde o servidor diz que sorteou.
    """
    sorteou = recorte_das_caixas(
        n_caixas=TETO,
        n_comparadas=TETO,
        n_antes=ANTES_TOTAL,
        amostrada=True,
        seed=SEMENTE,
        teto_de_caixas=TETO,
        coeficiente_de_tukey=COEFICIENTE_DE_TUKEY,
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    assert "sorteadas com a semente" in sorteou.amostragem, sorteou.amostragem

    nao_sorteou = recorte_de_prova(amostrada=False)
    assert "sorteadas" not in nao_sorteou.amostragem, nao_sorteou.amostragem
    assert nao_sorteou.amostragem.startswith("nenhuma"), nao_sorteou.amostragem


@pytest.mark.parametrize("n_antes", [None, 0, -3, "seis", float("nan"), True])
def test_sem_populacao_declarada_o_recorte_recusa_como_MATRIZ_VAZIA(n_antes: Any) -> None:  # noqa: N802
    """Guarda de fronteira, e a classe é a que o painel apanha."""
    for monta in (
        lambda: recorte_das_caixas(
            n_caixas=TETO,
            n_comparadas=COMPARADAS,
            n_antes=n_antes,
            amostrada=True,
            seed=SEMENTE,
            teto_de_caixas=TETO,
            coeficiente_de_tukey=COEFICIENTE_DE_TUKEY,
        ),
        lambda: recorte_da_densidade_por_feature(n_comparadas=COMPARADAS, n_antes=n_antes),
        lambda: recorte_da_densidade_por_amostra(
            n_comparadas=COMPARADAS, n_antes=n_antes, n_amostras=AMOSTRAS
        ),
    ):
        with pytest.raises(MatrizVaziaError, match="n_features_antes"):
            monta()


def test_o_coeficiente_de_tukey_e_INJETADO_e_aparece_no_rodape() -> None:  # noqa: N802
    """**O número que o servidor usa é o que a figura cita.**

    `plots/` não pode importar `refino.api` (regra 8 do README), e escrever `1,5` aqui criaria a
    segunda fonte de uma verdade só. Quem passa é o painel, com `esquemas.COEFICIENTE_DE_TUKEY`.
    Este teste prova que o parâmetro **chega ao texto** — um construtor que o ignorasse passaria por
    qualquer asserção que só olhasse o padrão.
    """
    outro = recorte_das_caixas(
        n_caixas=TETO,
        n_comparadas=COMPARADAS,
        n_antes=ANTES_TOTAL,
        amostrada=True,
        seed=SEMENTE,
        teto_de_caixas=TETO,
        coeficiente_de_tukey=3.0,
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    assert "3 x IQR" in outro.metodo, outro.metodo
    assert "1.5 x IQR" not in outro.metodo, outro.metodo


# ---------------------------------------------------------------------------
# 9. A fronteira do fio — nada disto passou pelo mypy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bruto", [None, {}, "caixas", 7, [1, 2, 3], [None]])
def test_caixas_do_fio_nao_levanta_com_corpo_ILEGIVEL(bruto: Any) -> None:  # noqa: N802
    """Excepção dentro de um *slot* do Qt é engolida para o `stderr` e o painel congela."""
    assert caixas_do_fio(bruto) == ()


def test_caixas_do_fio_converte_o_que_e_conversivel_e_PRESERVA_o_None() -> None:  # noqa: N802
    """O `None` de `n_discrepantes` sobrevive; o que não é número vira `nan` e não desaparece."""
    convertidas = caixas_do_fio(
        [
            {
                "feature": "a",
                "bigode_inferior": 1.0,
                "q1": 2.0,
                "mediana": 3.0,
                "q3": 4.0,
                "bigode_superior": 5.0,
                "n_discrepantes": None,
            },
            {
                "feature": "b",
                "bigode_inferior": None,
                "q1": 2.0,
                "mediana": 3.0,
                "q3": 4.0,
                "bigode_superior": 5.0,
                "n_discrepantes": 2,
            },
            {"feature": "c", "n_discrepantes": True},
        ]
    )
    assert len(convertidas) == 3
    assert convertidas[0].n_discrepantes is None and convertidas[0].desenhavel
    assert np.isnan(convertidas[1].bigode_inferior) and not convertidas[1].desenhavel
    assert convertidas[1].n_discrepantes == 2
    # `True` é um `int` em Python, e contar `true` como uma unidade seria um discrepante inventado.
    assert convertidas[2].n_discrepantes is None


@pytest.mark.parametrize("bruto", [None, {}, "caixas", 7, [1, 2, 3], [None]])
def test_caixas_por_amostra_do_fio_nao_levanta_com_corpo_ILEGIVEL(bruto: Any) -> None:  # noqa: N802
    """A mesma guarda de fronteira da irmã: nada disto passou pelo mypy."""
    assert caixas_por_amostra_do_fio(bruto) == ()


def test_o_conversor_por_amostra_le_a_chave_AMOSTRA_e_nao_a_chave_FEATURE() -> None:  # noqa: N802
    """**O par causal da chave: ler a chave errada devolve o desfecho de SUCESSO.**

    Uma carga com `feature` no lugar de `amostra` não levanta e não fica vazia — devolve o número
    certo de caixas, com os cinco números certos, **anónimas**. E caixa anónima não parece caixa
    errada: a identificação de R2 diria `· caixa`, e a guarda do par alinhado passaria por os dois
    lados serem igualmente vazios. Então as duas direções ficam na mesma função.

    Os seis números partilhados são lidos pelo **mesmo** `_campos_da_caixa` das duas, e é isso que
    esta função também afirma: o `None` sobrevive e o não-número vira `nan` aqui igual.
    """
    corpo = [
        {
            "amostra": "S_alpha",
            "bigode_inferior": 1.0,
            "q1": 2.0,
            "mediana": 3.0,
            "q3": 4.0,
            "bigode_superior": 5.0,
            "n_discrepantes": None,
        },
        {
            "amostra": "S_beta",
            "bigode_inferior": "nao é número",
            "q1": 2.0,
            "mediana": 3.0,
            "q3": 4.0,
            "bigode_superior": 5.0,
            "n_discrepantes": 3,
        },
    ]
    convertidas = caixas_por_amostra_do_fio(corpo)
    assert [c.amostra for c in convertidas] == ["S_alpha", "S_beta"]
    assert [c.nome for c in convertidas] == ["S_alpha", "S_beta"], "o `nome` é o campo `amostra`"
    assert convertidas[0].n_discrepantes is None and convertidas[0].desenhavel
    assert np.isnan(convertidas[1].bigode_inferior) and not convertidas[1].desenhavel
    assert convertidas[1].n_discrepantes == 3

    # A planta: a MESMA carga com a chave da irmã. Cai sobre dado em que as duas diferem — os
    # nomes são não vazios acima, e aqui têm de sair vazios.
    com_a_chave_errada = [
        {("feature" if chave == "amostra" else chave): valor for chave, valor in bruto.items()}
        for bruto in corpo
    ]
    assert all("amostra" not in bruto for bruto in com_a_chave_errada), (
        "âncora da planta: a chave `amostra` tem de ter SAÍDO da carga — a primeira escrita desta "
        "linha deixou-a lá, e o conversor achava-a na mesma"
    )
    anonimas = caixas_por_amostra_do_fio(com_a_chave_errada)
    assert len(anonimas) == 2, "âncora: a planta não muda quantas caixas chegam"
    assert [c.amostra for c in anonimas] == ["", ""], (
        "ler `feature` numa carga por amostra devolveu nome — a planta não separa"
    )


@pytest.mark.parametrize(
    ("bruto", "esperado"),
    [
        (None, []),
        ("nao", []),
        ([], []),
        ([1, 2.5], [1.0, 2.5]),
        ([1, "x", None, True], [1.0, np.nan, np.nan, np.nan]),
    ],
)
def test_vetor_do_fio_converte_e_marca_o_que_nao_e_numero(
    bruto: Any, esperado: list[float]
) -> None:
    lido = vetor_do_fio(bruto)
    assert lido.tolist() == pytest.approx(esperado, nan_ok=True)


# ---------------------------------------------------------------------------
# 10. O rodapé não é impresso por baixo do rótulo do eixo, nas TRÊS figuras
# ---------------------------------------------------------------------------


def faixas(figura: Figure) -> tuple[float, list[float]]:
    """`(topo do rodapé, fundo de CADA eixo com as marcas e o rótulo)`, em fração de figura."""
    renderizador = figura.canvas.get_renderer()  # type: ignore[attr-defined]
    para_figura = figura.transFigure.inverted()
    topo = float(rodape(figura).get_window_extent(renderizador).transformed(para_figura).y1)
    fundos = [
        float(eixo.get_tightbbox(renderizador).transformed(para_figura).y0) for eixo in figura.axes
    ]
    return topo, fundos


@pytest.mark.parametrize(
    "monta", [figura_das_caixas, figura_das_caixas_de_amostra, figura_da_densidade]
)
def test_o_rodape_NAO_colide_com_as_marcas_de_NENHUM_dos_dois_eixos(monta: Any) -> None:  # noqa: N802
    """**A leitura da imagem montada, mecanizada, e ela cobre os DOIS eixos.**

    `recorte.aplica` sobe `subplotpars.bottom`, que é comum à grelha inteira, e depois chama
    `tight_layout(rect=…)`. Com dois subplots a conta tem de valer para os dois, e um deles com
    rótulos de eixo rodados — as marcas do lado das caixas — é o caso apertado.

    **O par causal desta propriedade vive em `tests/unit/test_plots_deriva.py`**, onde a planta
    reconstrói o `aplica` antigo e afirma que os dois textos **de facto** colidem sem o
    afastamento. Aqui o que se afirma é o alvo — as figuras desta unidade —, e a âncora de
    não-vacuidade é a primeira linha: sem rodapé, a comparação não teria o que comparar.
    """
    figura = monta()
    figura.canvas.draw()
    topo, fundos = faixas(figura)
    assert 0.0 < topo < 1.0, f"âncora: o rodapé lido está fora da figura ({topo})"
    assert len(fundos) == 2, fundos
    for i, fundo in enumerate(fundos):
        assert fundo >= topo, (
            f"o eixo {i} começa em {fundo:.3f} e o rodapé vai até {topo:.3f}: os dois textos "
            "sobrepõem-se na imagem montada"
        )


# ---------------------------------------------------------------------------
# 11. Determinismo — regra 3 do README do lado do desenho
# ---------------------------------------------------------------------------


def test_duas_construcoes_da_MESMA_entrada_desenham_o_MESMO(  # noqa: N802
) -> None:
    """**O construtor não sorteia nada, e é essa a metade da regra 3 do README que lhe cabe.**

    O sorteio das 50 é do servidor, com a semente do corpo, e está provado em
    `tests/test_api_rotas.py`. O que se afirma aqui é que o desenho é função pura da entrada: um
    construtor que amostrasse por conta própria produziria duas figuras diferentes do mesmo corpo,
    e a semente que o rodapé declara deixaria de descrever o que está na tela.
    """
    uma, outra = figura_das_caixas(), figura_das_caixas()
    for eixo_a, eixo_b in zip(uma.axes, outra.axes, strict=True):
        a, b = series(eixo_a), series(eixo_b)
        assert [s.rotulo for s in a] == [s.rotulo for s in b]
        for sa, sb in zip(a, b, strict=True):
            assert np.array_equal(sa.x, sb.x) and np.array_equal(sa.y, sb.y)
    assert rodape(uma).get_text() == rodape(outra).get_text()


# ---------------------------------------------------------------------------
# 12. O corpus real — sem número fabricado (regra 1 do README)
# ---------------------------------------------------------------------------

MODO_REAL: Final = "neg_c18"


def test_o_neg_c18_real_atravessa_da_ingestao_ate_as_TRES_figuras() -> None:  # noqa: N802
    """**Ponta a ponta sobre o corpus real**, com as caixas calculadas por caminho independente.

    Os cinco números de cada caixa saem de `numpy` **aqui**, e não de `rotas._caixas`: é a fonte
    independente da classe (d) do arnês aplicada ao dado em vez de ao rótulo. As contagens que
    aparecem são **observação sobre este arquivo**, nunca invariante.
    """
    matriz = load_matrix(
        measurements_path(MODO_REAL),
        sample_list_path(MODO_REAL),
        MODO_REAL,
        orientation="features-in-rows",
    )
    assert matriz.n_features > TETO, "âncora: o corpus real tem de ter mais features que o teto"

    escolhidas = np.arange(TETO, dtype=np.int64)
    valores = matriz.values[:, escolhidas]
    q1, mediana, q3 = np.nanpercentile(valores, [25, 50, 75], axis=0)
    iqr = q3 - q1
    alto = np.nanmax(np.where(valores <= q3 + COEFICIENTE_DE_TUKEY * iqr, valores, np.nan), axis=0)
    baixo = np.nanmin(np.where(valores >= q1 - COEFICIENTE_DE_TUKEY * iqr, valores, np.nan), axis=0)
    reais = tuple(
        Caixa(
            feature=matriz.feature_names[coluna],
            bigode_inferior=float(min(baixo[i], q1[i])),
            q1=float(q1[i]),
            mediana=float(mediana[i]),
            q3=float(q3[i]),
            bigode_superior=float(max(alto[i], q3[i])),
            n_discrepantes=0,
        )
        for i, coluna in enumerate(escolhidas.tolist())
    )
    assert all(c.desenhavel for c in reais), "âncora: as 50 primeiras do neg_c18 têm caixa"

    figura = figura_das_caixas_por_feature(
        antes=reais,
        depois=reais,
        recorte=recorte_das_caixas(
            n_caixas=TETO,
            n_comparadas=matriz.n_features,
            n_antes=matriz.n_features,
            amostrada=True,
            seed=SEMENTE,
            teto_de_caixas=TETO,
            coeficiente_de_tukey=COEFICIENTE_DE_TUKEY,
            matriz_antes=MATRIZ_ANTES,
            matriz_depois=MATRIZ_DEPOIS,
        ),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    assert confere_series(figura.axes[0], esperadas_do_lado(reais)) == []
    assert confere_vista(figura.axes[0]) == []

    # E as duas densidades, sobre as MÉDIAS de cada eixo — que é a armadilha semântica da unidade.
    por_feature = np.nanmean(matriz.values, axis=0)
    por_amostra = np.nanmean(matriz.values, axis=1)
    assert por_feature.size == matriz.n_features and por_amostra.size == matriz.n_samples, (
        "âncora: a média por eixo tem de ter o tamanho do eixo — se as duas tivessem o mesmo "
        "tamanho, trocá-las não seria detetável"
    )
    densidade_f = figura_da_densidade_por_feature(
        antes=por_feature,
        depois=por_feature,
        recorte=recorte_da_densidade_por_feature(
            n_comparadas=matriz.n_features, n_antes=matriz.n_features
        ),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    densidade_a = figura_da_densidade_por_amostra(
        antes=por_amostra,
        depois=por_amostra,
        recorte=recorte_da_densidade_por_amostra(
            n_comparadas=matriz.n_features,
            n_antes=matriz.n_features,
            n_amostras=matriz.n_samples,
        ),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    # **A densidade é sobre as MÉDIAS e não sobre as células**, e é o título que o declara: a
    # contagem tem de ser a do eixo, nunca `n_features x n_samples`.
    celulas = matriz.n_features * matriz.n_samples
    for figura_da_curva, quantos in (
        (densidade_f, matriz.n_features),
        (densidade_a, matriz.n_samples),
    ):
        titulo = figura_da_curva.axes[0].get_title()
        assert f"{milhar(quantos)} valor(es)" in titulo, titulo
        assert milhar(celulas) not in titulo, titulo


def test_o_neg_c18_real_atravessa_ate_a_QUARTA_figura_e_ela_NAO_e_a_segunda() -> None:  # noqa: N802
    """**A âncora da transposta sobre o corpus real — e é ela que torna este bloco atribuível.**

    As duas figuras de caixas saem da **mesma matriz** pelos eixos opostos. Um teste que passasse
    igualmente nas duas não distinguiria a `Sample View` de uma segunda `Feature View` com outro
    título; então o que se afirma aqui, sobre o dado real e antes de qualquer comparação, é que os
    dois conjuntos de caixas **divergem**.

    Os cinco números de cada caixa saem de `numpy` **aqui** e não de `rotas._caixas_por_amostra`:
    é a fonte independente da classe (d) do arnês aplicada ao dado. As contagens são **observação
    sobre este arquivo**, nunca invariante.
    """
    matriz = load_matrix(
        measurements_path(MODO_REAL),
        sample_list_path(MODO_REAL),
        MODO_REAL,
        orientation="features-in-rows",
    )
    assert matriz.n_samples > 1, "âncora: sem amostras não há caixa por amostra"

    def cinco_numeros(valores: np.ndarray) -> np.ndarray:
        """As cinco estatísticas de cada COLUNA de `valores`, por caminho independente."""
        q1, mediana, q3 = np.nanpercentile(valores, [25, 50, 75], axis=0)
        iqr = q3 - q1
        alto = np.nanmax(
            np.where(valores <= q3 + COEFICIENTE_DE_TUKEY * iqr, valores, np.nan), axis=0
        )
        baixo = np.nanmin(
            np.where(valores >= q1 - COEFICIENTE_DE_TUKEY * iqr, valores, np.nan), axis=0
        )
        return np.vstack([np.minimum(baixo, q1), q1, mediana, q3, np.maximum(alto, q3)])

    linhas = np.arange(min(TETO, matriz.n_samples), dtype=np.int64)
    # **A transposta: cada amostra escolhida vira uma COLUNA.** É a mesma conta, sobre o outro eixo.
    por_amostra = cinco_numeros(matriz.values[linhas, :].T)
    por_feature = cinco_numeros(matriz.values[:, np.arange(linhas.size, dtype=np.int64)])

    # **A âncora da transposta**, e ela vem antes de tudo: os dois eixos da mesma matriz dão caixas
    # diferentes. Sem isto, uma figura que desenhasse o eixo errado passaria em tudo o que segue.
    assert por_amostra.shape == por_feature.shape, "âncora: as duas têm de ser comparáveis"
    assert not np.allclose(por_amostra, por_feature, equal_nan=True), (
        "as caixas por amostra e por feature do `neg_c18` são numericamente iguais — a asserção "
        "de que esta figura é a transposta e não a irmã não teria sobre o que cair"
    )

    reais = tuple(
        CaixaDeAmostra(
            amostra=matriz.sample_names[linha],
            bigode_inferior=float(por_amostra[0, i]),
            q1=float(por_amostra[1, i]),
            mediana=float(por_amostra[2, i]),
            q3=float(por_amostra[3, i]),
            bigode_superior=float(por_amostra[4, i]),
            n_discrepantes=0,
        )
        for i, linha in enumerate(linhas.tolist())
    )
    assert all(c.desenhavel for c in reais), "âncora: as amostras do `neg_c18` têm caixa"
    assert len({c.nome for c in reais}) == len(reais), "nomes de amostra repetidos"

    figura = figura_das_caixas_por_amostra(
        antes=reais,
        depois=reais,
        recorte=recorte_das_caixas_por_amostra(
            n_caixas=len(reais),
            n_amostras_comparadas=matriz.n_samples,
            n_comparadas=matriz.n_features,
            n_antes=matriz.n_features,
            amostrada=matriz.n_samples > TETO,
            seed=SEMENTE,
            teto_de_caixas=TETO,
            coeficiente_de_tukey=COEFICIENTE_DE_TUKEY,
            matriz_antes=MATRIZ_ANTES,
            matriz_depois=MATRIZ_DEPOIS,
        ),
        matriz_antes=MATRIZ_ANTES,
        matriz_depois=MATRIZ_DEPOIS,
    )
    assert confere_series(figura.axes[0], esperadas_do_lado(reais)) == []
    assert confere_vista(figura.axes[0]) == []
    assert sem_a_matriz(figura.get_suptitle()) == TITULO_DAS_CAIXAS_POR_AMOSTRA

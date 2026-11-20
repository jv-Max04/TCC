"""A série categórica por grupo clínico, que seis das oito famílias novas vão usar.

## O que este módulo prova, e o que ele deliberadamente não prova

O que se quer de `plots/series.py` é **uma** propriedade, e ela é fácil de enunciar e fácil de
perder: *a mesma coorte produz a mesma ordem e a mesma cor, chamada de qualquer sítio.* Sem ela,
seis módulos inventam seis ordens e o profissional vê a mesma coorte com cores diferentes entre abas
do mesmo painel — com as seis figuras **bem formadas**, que é o desfecho caro.

**O que NÃO se prova aqui é que a ordem escolhida seja a que um humano esperaria.** Ela é a
ordenação por ponto de código, e `ATLL` vem antes de `Asymptomatic`; isso é declarado na docstring
do módulo e tem teste próprio, para que ninguém o «conserte» supondo alfabeto de alguma língua.

**E não se prova nada sobre figura.** Este módulo não desenha: devolve ordem, cor, máscara e rótulo.
Quem os põe num eixo são as famílias que vêm a seguir, e é lá que as quatro classes de asserção do
arnês correm.

## O corpus é o deste estudo, e os nomes são literais de propósito

`Asymptomatic` 18, `HAMTSP` 14, `NoHTLV` 16 e `ATLL` 13 são os quatro grupos clínicos do dado real.
As contagens aqui são **observação sobre aquele corpus** e não invariante — o que
este módulo tem de suportar é N grupos arbitrários, e é isso que os testes de N variável afirmam.
"""

from __future__ import annotations

from typing import Final, get_args

import matplotlib
import matplotlib.colors
import matplotlib.markers
import numpy as np
import pytest

from refino.plots.base import cores_do_ciclo
from refino.plots.series import (
    CORES_DOS_GRUPOS,
    PARES_DE_ESTILO,
    SIMBOLOS_DOS_GRUPOS,
    CorDoGrupo,
    EstiloDoGrupo,
    EstiloInvalidoError,
    EstiloRepetidoError,
    EstilosDosGrupos,
    GrupoInvalidoError,
    SerieDeGrupo,
    SimboloDoGrupo,
    cores_dos_grupos,
    estilos_dos_grupos,
    grupos_do_fio,
    ordem_canonica,
    rotulo_do_grupo,
    series_por_grupo,
)

# Os quatro grupos clínicos deste estudo, com o `n` de cada um. **Observação sobre o corpus de
# 13/09/2026, não invariante**: outro estudo traz outros grupos, e em qualquer número.
COORTE: Final = {"Asymptomatic": 18, "HAMTSP": 14, "NoHTLV": 16, "ATLL": 13}


def vetor_da_coorte() -> list[str]:
    """O vetor por amostra, na ordem em que os grupos aparecem no dicionário acima."""
    return [nome for nome, quantas in COORTE.items() for _ in range(quantas)]


# ---------------------------------------------------------------------------
# 1. A propriedade inteira: a mesma coorte, a mesma ordem e a mesma cor
# ---------------------------------------------------------------------------


def test_a_mesma_coorte_da_a_mesma_ordem_vinda_de_DUAS_fontes_diferentes() -> None:  # noqa: N802
    """Par causal da ordem canônica: **os dois vetores diferem e as duas ordens não**.

    **A ponta de cima é a âncora:** se as duas entradas já viessem na mesma ordem, a asserção
    de igualdade passaria com a ordenação desligada — e é exactamente esse o teste vácuo que a
    ordenação existe para não ter. Aqui uma vem na ordem das linhas da matriz e a outra invertida,
    que é o que dois módulos de facto passam.
    """
    na_ordem_das_linhas = vetor_da_coorte()
    ao_contrario = list(reversed(na_ordem_das_linhas))
    assert na_ordem_das_linhas != ao_contrario, "âncora: as duas entradas têm de diferir"

    assert ordem_canonica(na_ordem_das_linhas) == ordem_canonica(ao_contrario)
    assert cores_dos_grupos(na_ordem_das_linhas) == cores_dos_grupos(ao_contrario)


def test_a_mesma_coorte_da_a_mesma_COR_chamada_de_dois_modulos() -> None:  # noqa: N802
    """A cor de um grupo é função da coorte, e de mais nada que o chamador possa variar.

    Um módulo que passe o vetor por amostra e outro que passe as chaves de um dicionário de
    contagens — que é a forma em que um inventário de grupos chega — têm de sair com a mesma paleta.
    """
    por_amostra = cores_dos_grupos(vetor_da_coorte())
    por_inventario = cores_dos_grupos(COORTE.keys())
    assert por_amostra == por_inventario

    ciclo = cores_do_ciclo(len(COORTE))
    assert list(por_amostra.values()) == list(ciclo), (
        "as cores não saem do ciclo dos `rcParams` na ordem canônica — e é o ciclo que o tema do "
        "sistema operativo configura"
    )


def test_a_ordem_canonica_e_a_ORDENACAO_e_nao_a_de_aparecimento() -> None:  # noqa: N802
    """`ATLL` antes de `Asymptomatic`, e isto está aqui para ninguém o «consertar».

    **Não é ordem alfabética de nenhuma língua**: é por ponto de código, e `T` (U+0054) vem antes de
    `s` (U+0073). O que se compra é determinismo independente de localidade; o que se perde é a
    ordem que um humano esperaria, e quem responde por essa é o rótulo, que nomeia o grupo.
    """
    assert ordem_canonica(vetor_da_coorte()) == ("ATLL", "Asymptomatic", "HAMTSP", "NoHTLV")
    assert ordem_canonica(["b", "a", "b"]) == ("a", "b")


def test_a_cor_acompanha_o_tema_porque_sai_do_ciclo_e_nao_de_um_literal() -> None:
    """As duas direções: a MESMA coorte, dois ciclos, duas paletas.

    Sem isto, um literal de cor neste módulo passaria em tudo o que está acima — a ordem continuaria
    certa e a paleta continuaria estável. O que ele quebraria é o tema, que é uma propriedade dos
    `rcParams` no momento da construção.
    """
    with matplotlib.rc_context(
        {"axes.prop_cycle": matplotlib.cycler(color=["#111111", "#222222"])}
    ):
        escura = cores_dos_grupos(["a", "b"])
    with matplotlib.rc_context(
        {"axes.prop_cycle": matplotlib.cycler(color=["#EEEEEE", "#DDDDDD"])}
    ):
        clara = cores_dos_grupos(["a", "b"])

    assert escura == {"a": "#111111", "b": "#222222"}
    assert clara == {"a": "#EEEEEE", "b": "#DDDDDD"}


def test_mais_grupos_do_que_cores_no_ciclo_reaproveita_em_vez_de_recusar() -> None:
    """O recuo declarado: a atribuição volta ao início do ciclo, e a legenda carrega o TEXTO.

    Recusar desenhar por causa de uma coorte grande trocaria informação por nada. O que se perde é a
    distinção pela cor, e é por isso que todo rótulo deste pacote nomeia a série.
    """
    with matplotlib.rc_context(
        {"axes.prop_cycle": matplotlib.cycler(color=["#111111", "#222222"])}
    ):
        paleta = cores_dos_grupos(["a", "b", "c", "d", "e"])
    assert len(paleta) == 5
    assert paleta["a"] == paleta["c"] == "#111111"
    assert paleta["b"] == paleta["d"] == "#222222"


# ---------------------------------------------------------------------------
# 2. Grupo ausente, grupo novo
# ---------------------------------------------------------------------------


def test_grupo_ausente_da_matriz_NAO_recebe_cor_nem_serie() -> None:  # noqa: N802
    """A ausência é a informação: uma entrada de legenda com `0` prometeria pontos que não existem.

    É a escolha que `pca.figura_dos_scores` já faz para a série `validate`, aqui como mecanismo em
    vez de como cuidado de cada construtor.
    """
    presentes = ["HAMTSP", "HAMTSP", "ATLL"]
    assert "NoHTLV" not in cores_dos_grupos(presentes)
    assert [s.grupo for s in series_por_grupo(presentes)] == ["ATLL", "HAMTSP"]
    assert all(s.n > 0 for s in series_por_grupo(presentes))


def test_grupo_novo_entra_na_ordem_canonica_de_forma_DETERMINISTICA() -> None:  # noqa: N802
    """Duas execuções com o grupo novo dão a mesma posição, e ela é a que a ordenação lhe dá.

    **E a consequência vai afirmada em vez de suposta:** o grupo novo desloca as cores dos que vêm
    depois dele. É o preço de a cor não depender da ordem das linhas, e a alternativa exigiria uma
    tabela de grupos conhecidos — a segunda fonte de verdade que este módulo existe para não ter.
    """
    antes = ordem_canonica(["ATLL", "NoHTLV"])
    depois = ordem_canonica(["ATLL", "NoHTLV", "HAMTSP"])
    assert antes == ("ATLL", "NoHTLV")
    assert depois == ("ATLL", "HAMTSP", "NoHTLV")
    assert ordem_canonica(["NoHTLV", "HAMTSP", "ATLL"]) == depois

    paleta_antes = cores_dos_grupos(antes)
    paleta_depois = cores_dos_grupos(depois)
    assert paleta_antes["ATLL"] == paleta_depois["ATLL"]
    assert paleta_antes["NoHTLV"] != paleta_depois["NoHTLV"], (
        "o grupo novo NÃO deslocou as cores dos seguintes — a consequência declarada na docstring "
        "do módulo deixou de ser verdade, e o texto passou a descrever outro comportamento"
    )


# ---------------------------------------------------------------------------
# 3. O rótulo, com o `n` e com o plural aritmético certo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quantas", "esperado"),
    [
        (0, "HAMTSP — 0 amostras"),
        (1, "HAMTSP — 1 amostra"),
        (14, "HAMTSP — 14 amostras"),
        (6035, "HAMTSP — 6.035 amostras"),
    ],
)
def test_o_rotulo_traz_o_n_e_o_plural_esta_certo_para_zero_um_e_muitos(
    quantas: int, esperado: str
) -> None:
    """Os três desfechos do plural, e o milhar em português.

    **Três e não dois**: `0` é o que se lê como defeito de formatação em `0 amostra`, `1` é o
    caso que o plural ingénuo erra, e o milhar é o que `recorte.milhar` existe para não errar.
    """
    assert rotulo_do_grupo("HAMTSP", quantas) == esperado


def test_o_rotulo_da_serie_conta_o_que_a_MASCARA_tem() -> None:  # noqa: N802
    """O `n` do rótulo sai da máscara, e não de um argumento que o chamador possa desalinhar.

    Duas fontes para a mesma contagem divergiriam no dia em que uma delas ganhasse um filtro, e o
    sintoma seria uma legenda a contar uma população que não está desenhada.
    """
    series = series_por_grupo(vetor_da_coorte())
    for serie in series:
        assert serie.rotulo == f"{serie.grupo} — {COORTE[serie.grupo]} amostras"
        assert serie.n == COORTE[serie.grupo]


# ---------------------------------------------------------------------------
# 4. A máscara indexa o vetor que o chamador passou
# ---------------------------------------------------------------------------


def test_as_mascaras_particionam_o_vetor_de_amostras_por_INTEIRO() -> None:  # noqa: N802
    """A partição é TOTAL e disjunta, e é isso que impede uma amostra de sumir entre as séries.

    **É o defeito** na outra ponta: um valor que caísse fora de todas as máscaras
    produziria uma amostra invisível à figura, com a legenda a contar menos do que a matriz tem — e
    sem nada levantar.
    """
    vetor = vetor_da_coorte()
    series = series_por_grupo(vetor)
    soma = np.zeros(len(vetor), dtype=int)
    for serie in series:
        assert serie.mascara.shape == (len(vetor),)
        soma += serie.mascara.astype(int)
    assert np.array_equal(soma, np.ones(len(vetor), dtype=int)), (
        f"alguma amostra caiu em zero ou em mais de uma série: {np.bincount(soma)}"
    )
    assert sum(s.n for s in series) == len(vetor) == sum(COORTE.values())


def test_a_mascara_seleciona_as_POSICOES_certas_do_vetor() -> None:  # noqa: N802
    """A máscara é posicional, e é com ela que o construtor fatia os vetores de coordenadas."""
    vetor = ["b", "a", "b", "c", "a"]
    por_grupo = {s.grupo: s for s in series_por_grupo(vetor)}
    assert por_grupo["a"].mascara.tolist() == [False, True, False, False, True]
    assert por_grupo["b"].mascara.tolist() == [True, False, True, False, False]
    assert por_grupo["c"].mascara.tolist() == [False, False, False, True, False]

    coordenadas = np.arange(5.0)
    assert coordenadas[por_grupo["a"].mascara].tolist() == [1.0, 4.0]


def test_a_posicao_e_o_indice_na_ordem_canonica() -> None:
    """`posicao` é o que decide a cor, e é o que se compara ao ler duas abas lado a lado."""
    series = series_por_grupo(vetor_da_coorte())
    assert [s.posicao for s in series] == [0, 1, 2, 3]
    assert [s.grupo for s in series] == list(ordem_canonica(vetor_da_coorte()))


def test_vetor_vazio_devolve_serie_nenhuma_em_vez_de_levantar() -> None:
    """Quem decide o que fazer com população vazia é o construtor, que levanta `MatrizVaziaError`.

    Levantar aqui obrigaria cada chamador a apanhar duas excepções para o mesmo fato.
    """
    assert series_por_grupo([]) == ()
    assert ordem_canonica([]) == ()
    assert cores_dos_grupos([]) == {}


# ---------------------------------------------------------------------------
# 5. A fronteira: o que chega do fio não passou pelo mypy
# ---------------------------------------------------------------------------


def test_os_grupos_do_fio_atravessam_a_guarda() -> None:
    """Lista de `str` passa, e a conversão preserva a ordem — que é o que a máscara indexa."""
    assert grupos_do_fio(["a", "b", "a"]) == ("a", "b", "a")
    assert grupos_do_fio(("a",)) == ("a",)


@pytest.mark.parametrize("plantado", [None, "", "   ", 12, True, 3.5, ["aninhado"]])
def test_um_nome_de_grupo_ILEGIVEL_recusa_em_vez_de_recuar(plantado: object) -> None:  # noqa: N802
    """As sete formas do valor que não é nome de grupo, uma por execução.

    **Recusa e não recuo**, pela razão de `pca._tipo_ou_recusa`: um nome em branco que virasse
    `"(sem grupo)"` poria a legenda a nomear uma população que ninguém declarou, e a figura sairia
    perfeitamente bem formada — que é o desfecho caro.
    """
    with pytest.raises(GrupoInvalidoError, match="chegou como"):
        grupos_do_fio(["Asymptomatic", plantado])


def test_a_guarda_do_fio_NAO_recusa_o_que_e_legitimo() -> None:  # noqa: N802
    """A outra ponta: sem planta, não acusa — senão uma guarda que recusasse tudo passaria acima."""
    assert grupos_do_fio(list(COORTE)) == tuple(COORTE)


def test_o_que_nao_e_lista_devolve_tupla_vazia_e_nao_levanta() -> None:
    """A distinção: *"não veio lista"* é população vazia, e quem a declara é o construtor."""
    assert grupos_do_fio(None) == ()
    assert grupos_do_fio("Asymptomatic") == ()
    assert grupos_do_fio({"a": 1}) == ()


def test_a_serie_e_IMUTAVEL() -> None:  # noqa: N802
    """`frozen=True`: o construtor que a recebe não a pode realinhar por acidente."""
    serie = series_por_grupo(["a"])[0]
    assert isinstance(serie, SerieDeGrupo)
    with pytest.raises(AttributeError):
        serie.grupo = "b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. A cor e o símbolo ESCOLHIDOS
# ---------------------------------------------------------------------------
#
# A cor e o símbolo de cada grupo experimental podem ser escolhidos. O vocabulário é fechado —
# oito cores de Okabe e Ito e oito símbolos —, e a escolha entra nos construtores como parâmetro;
# sem ele, a cor continua a do ciclo e o marcador o círculo, que é o que as secções acima afirmam
# e nenhuma delas precisou de mudar.

# As oito cores de Okabe e Ito, por valor. **O conjunto e não a ordem**: a ordem é decisão de
# leitura e tem teste próprio, que a deriva do contraste em vez de a copiar.
OKABE_ITO: Final = frozenset(
    {"#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"}
)

# O fundo das figuras — `desktop/janela/figuras.CORES_DA_FIGURA[0]`. Literal aqui porque
# importar a janela traria o Qt a um teste de `plots/`; o valor é o da trinca fixa, e é contra ele
# que a ordem da paleta foi medida.
FUNDO_DA_FIGURA: Final = "#ffffff"


def _contraste_wcag(cor: str, fundo: str) -> float:
    """A razão de contraste da WCAG 2 — o oráculo da ordem, escrito aqui e não importado."""

    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def luminancia(h: str) -> float:
        r, g, b = matplotlib.colors.to_rgb(h)
        return 0.2126 * linear(r) + 0.7152 * linear(g) + 0.0722 * linear(b)

    clara, escura = sorted((luminancia(cor), luminancia(fundo)), reverse=True)
    return (clara + 0.05) / (escura + 0.05)


def test_o_vocabulario_e_FECHADO_e_bijeta_com_os_dois_literais() -> None:  # noqa: N802
    """As oito cores de Okabe e Ito, oito marcadores cheios do `matplotlib`, e os `Literal` iguais.

    **A bijeção é nas duas direções**: o `mypy` confere que as chaves do dicionário são do
    `Literal`; este teste confere o contrário — um nome acrescentado ao `Literal` sem cor nenhuma
    passaria pela análise estática e chegaria ao `KeyError` na primeira figura.
    """
    assert set(CORES_DOS_GRUPOS.values()) == OKABE_ITO
    assert set(get_args(CorDoGrupo)) == set(CORES_DOS_GRUPOS)
    assert set(get_args(SimboloDoGrupo)) == set(SIMBOLOS_DOS_GRUPOS)

    marcadores = list(SIMBOLOS_DOS_GRUPOS.values())
    assert len(set(marcadores)) == len(marcadores) == 8, marcadores
    for marcador in marcadores:
        # **Cheio, e é a impressão em tons de cinza que o exige:** um marcador vazado de cor clara
        # desaparece no cinza, e é o símbolo que distingue os grupos ali.
        assert matplotlib.markers.MarkerStyle(marcador).is_filled(), marcador

    assert len(PARES_DE_ESTILO) == len(set(PARES_DE_ESTILO)) == 64
    assert PARES_DE_ESTILO[0] == (next(iter(CORES_DOS_GRUPOS)), "círculo"), (
        "o primeiro par não é a primeira cor com o círculo — e o círculo é o marcador de hoje"
    )


def test_a_ordem_da_paleta_e_o_CONTRASTE_contra_o_fundo_com_o_preto_por_ultimo() -> None:  # noqa: N802
    """A ordem é medida e não escolhida: contraste decrescente contra o branco.

    **O preto vai por último e é a única exceção**, porque é a tinta da própria figura — eixos,
    marcas, texto e rodapé saem a preto. As outras sete seguem o contraste, e é por isso que os dois
    primeiros grupos de qualquer coorte saem azul e vermelho, as duas acima de 3:1 com maior folga.
    """
    nomes = list(CORES_DOS_GRUPOS)
    assert nomes[-1] == "preto"
    contrastes = [_contraste_wcag(CORES_DOS_GRUPOS[n], FUNDO_DA_FIGURA) for n in nomes[:-1]]
    assert contrastes == sorted(contrastes, reverse=True), list(
        zip(nomes, contrastes, strict=False)
    )
    assert nomes[:2] == ["azul", "vermelho"]


@pytest.mark.parametrize(
    ("cor", "simbolo"),
    [
        ("magenta", "círculo"),
        ("#D55E00", "círculo"),
        ("Vermelho", "círculo"),
        (None, "círculo"),
        (["vermelho"], "círculo"),
        ("vermelho", "hexágono"),
        ("vermelho", "s"),
        ("vermelho", None),
        ("vermelho", ""),
    ],
)
def test_cor_fora_da_paleta_ou_simbolo_fora_do_conjunto_e_RECUSADO(  # noqa: N802
    cor: object, simbolo: object
) -> None:
    """As nove formas do valor fora do vocabulário, uma por execução.

    **O código hexadecimal e o código do marcador também são recusados**, e é de propósito: o
    vocabulário são os NOMES, e aceitar `#D55E00` ao lado de `vermelho` daria duas grafias para a
    mesma cor, que a recusa do par repetido compararia como diferentes.
    """
    with pytest.raises(EstiloInvalidoError, match="fora"):
        EstiloDoGrupo(cor=cor, simbolo=simbolo)  # type: ignore[arg-type]


def test_a_guarda_do_estilo_NAO_recusa_o_que_e_do_vocabulario() -> None:  # noqa: N802
    """A outra ponta: os 64 pares passam, e cada um diz a cor e o marcador que vai desenhar."""
    for cor, simbolo in PARES_DE_ESTILO:
        estilo = EstiloDoGrupo(cor=cor, simbolo=simbolo)
        assert estilo.hex == CORES_DOS_GRUPOS[cor]
        assert estilo.marcador == SIMBOLOS_DOS_GRUPOS[simbolo]
    vermelho_quadrado = EstiloDoGrupo(cor="vermelho", simbolo="quadrado")
    assert (vermelho_quadrado.hex, vermelho_quadrado.marcador) == ("#D55E00", "s")


def test_o_mesmo_par_em_dois_grupos_e_RECUSADO_e_meio_par_nao_e() -> None:  # noqa: N802
    """O par inteiro é o que distingue dois grupos; a cor sozinha ou o símbolo sozinho não.

    **O par causal é de uma variável por vez:** mesma cor com outro símbolo passa, mesmo símbolo com
    outra cor passa, e só a coincidência das duas é recusada. Uma guarda que recusasse por cor
    proibiria a escolha que serve à impressão em cinza — a mesma cor, símbolos diferentes.
    """
    controle = EstiloDoGrupo(cor="vermelho", simbolo="quadrado")
    with pytest.raises(EstiloRepetidoError, match="Controle"):
        EstilosDosGrupos(
            {"Controle": controle, "Caso": EstiloDoGrupo(cor="vermelho", simbolo="quadrado")}
        )
    EstilosDosGrupos({"Controle": controle, "Caso": EstiloDoGrupo(cor="vermelho", simbolo="cruz")})
    EstilosDosGrupos({"Controle": controle, "Caso": EstiloDoGrupo(cor="azul", simbolo="quadrado")})


@pytest.mark.parametrize("plantado", [None, "", "   ", 12])
def test_os_estilos_recusam_nome_de_grupo_ilegivel_e_valor_que_nao_e_estilo(
    plantado: object,
) -> None:
    """Nome de grupo ilegível recusa como no fio, e um valor que não é `EstiloDoGrupo` também."""
    with pytest.raises(GrupoInvalidoError):
        EstilosDosGrupos({plantado: EstiloDoGrupo(cor="azul", simbolo="círculo")})  # type: ignore[dict-item]
    with pytest.raises(EstiloInvalidoError):
        EstilosDosGrupos({"Controle": ("azul", "círculo")})  # type: ignore[dict-item]


def test_o_padrao_da_coorte_e_a_POSICAO_canonica_e_nao_a_ordem_de_chegada() -> None:  # noqa: N802
    """A mesma coorte em duas ordens de chegada dá os mesmos estilos, e o grupo `i` leva o par `i`.

    **É a propriedade de `cores_dos_grupos` trazida para o estilo**: a ordem canônica decide, e não
    a ordem das linhas. Com até oito grupos o símbolo é o círculo de hoje.
    """
    na_ordem_das_linhas = estilos_dos_grupos(vetor_da_coorte())
    invertida = estilos_dos_grupos(reversed(vetor_da_coorte()))
    assert na_ordem_das_linhas is not None and invertida is not None
    assert dict(na_ordem_das_linhas.por_grupo) == dict(invertida.por_grupo)

    for posicao, grupo in enumerate(ordem_canonica(COORTE)):
        assert na_ordem_das_linhas.por_grupo[grupo].par == PARES_DE_ESTILO[posicao], grupo
    assert {e.simbolo for e in na_ordem_das_linhas.por_grupo.values()} == {"círculo"}


def test_o_nono_grupo_muda_de_SIMBOLO_e_nenhum_par_se_repete() -> None:  # noqa: N802
    """Oito cores não chegam para nove grupos, e o símbolo é o que separa o nono do primeiro."""
    nomes = [f"g{i}" for i in range(9)]
    estilos = estilos_dos_grupos(nomes)
    assert estilos is not None
    primeiro, nono = estilos.por_grupo["g0"], estilos.por_grupo["g8"]
    assert primeiro.cor == nono.cor
    assert (primeiro.simbolo, nono.simbolo) == ("círculo", "quadrado")
    pares = [e.par for e in estilos.por_grupo.values()]
    assert len(set(pares)) == len(pares) == 9


def test_a_escolha_vale_e_SO_o_grupo_deslocado_muda() -> None:  # noqa: N802
    """Quem escolhe o par de outro grupo desloca SÓ esse, e para o primeiro par que ninguém usa.

    **A cascata é o defeito que isto evita.** Uma atribuição que reatribuísse a coorte inteira em
    torno da escolha mudaria a cor de grupos que a pessoa não tocou; aqui `c` e `d` ficam com o par
    da posição deles, e `b` — que perdeu o seu — recebe o que `a` deixou livre.
    """
    coorte = ["a", "b", "c", "d"]
    sem = estilos_dos_grupos(coorte)
    assert sem is not None
    com = estilos_dos_grupos(coorte, {"a": sem.por_grupo["b"]})
    assert com is not None

    assert com.por_grupo["a"] == sem.por_grupo["b"]
    assert com.por_grupo["b"] == sem.por_grupo["a"]
    assert (com.por_grupo["c"], com.por_grupo["d"]) == (sem.por_grupo["c"], sem.por_grupo["d"])


def test_a_escolha_de_um_grupo_FORA_da_coorte_nao_entra() -> None:  # noqa: N802
    """A escolha vale para a sessão e a coorte é a do estudo aberto: o ausente não é desenhado."""
    estilos = estilos_dos_grupos(["a", "b"], {"z": EstiloDoGrupo(cor="preto", simbolo="estrela")})
    assert estilos is not None
    assert set(estilos.por_grupo) == {"a", "b"}


def test_sem_grupo_ou_com_mais_grupos_que_pares_NAO_ha_estilo() -> None:  # noqa: N802
    """Os dois desfechos sem estilo, e o par causal na fronteira dos 64.

    **`None` para os dois, e o colapso é correto para a DECISÃO**: nos dois casos os
    construtores desenham como hoje. Quem precisa de os separar — a frase da tabela da janela —
    tem a coorte na mão e conta-a.
    """
    assert estilos_dos_grupos([]) is None
    cabem = [f"g{i:02d}" for i in range(len(PARES_DE_ESTILO))]
    assert estilos_dos_grupos(cabem) is not None
    assert estilos_dos_grupos([*cabem, "g99"]) is None


def test_o_grupo_que_os_estilos_NAO_tem_recebe_o_primeiro_par_livre() -> None:  # noqa: N802
    """O QC do K-means: desenhado na figura, ausente da coorte clínica — e sem par repetido.

    A composição por classe deixa o QC fora por desenho; o K-means e o SOM correm sobre a matriz
    inteira e desenham-no. Ele recebe o primeiro par que a coorte não usa, e os da coorte ficam.
    """
    estilos = estilos_dos_grupos(["HAMTSP", "ATLL"])
    assert estilos is not None
    da_figura = estilos.para(["QC", "ATLL", "HAMTSP", "QC"])

    assert list(da_figura) == ["ATLL", "HAMTSP", "QC"]
    assert da_figura["ATLL"] == estilos.por_grupo["ATLL"]
    assert da_figura["HAMTSP"] == estilos.por_grupo["HAMTSP"]
    usados = {e.par for e in estilos.por_grupo.values()}
    assert da_figura["QC"].par == next(p for p in PARES_DE_ESTILO if p not in usados)


def test_series_por_grupo_SEM_estilo_e_a_de_hoje_e_COM_estilo_e_a_escolhida() -> None:  # noqa: N802
    """As duas pontas do parâmetro, sobre o mesmo vetor: só a cor e o marcador mudam.

    **A metade sem estilo é a que mantém as famílias como estavam** — o ciclo na ordem canônica e
    o círculo —, e a máscara e o rótulo têm de sair iguais nas duas, porque o estilo não é dado.
    """
    vetor = vetor_da_coorte()
    hoje = series_por_grupo(vetor)
    assert [s.cor for s in hoje] == list(cores_do_ciclo(len(COORTE)))
    assert {s.marcador for s in hoje} == {"o"}

    estilos = estilos_dos_grupos(vetor, {"ATLL": EstiloDoGrupo(cor="vermelho", simbolo="quadrado")})
    com = {s.grupo: s for s in series_por_grupo(vetor, estilos=estilos)}
    assert (com["ATLL"].cor, com["ATLL"].marcador) == ("#D55E00", "s")
    for serie in hoje:
        assert np.array_equal(com[serie.grupo].mascara, serie.mascara)
        assert com[serie.grupo].rotulo == serie.rotulo
    # **Em hexadecimal dos dois lados**: o ciclo de origem traz tuplas RGB, e comparar `#D55E00` com
    # tuplas passaria sempre — a âncora seria vácua pela grafia e não pela cor.
    assert matplotlib.colors.to_hex(com["ATLL"].cor) not in {
        matplotlib.colors.to_hex(s.cor) for s in hoje
    }, "âncora: a cor escolhida coincide com uma do ciclo, e a planta não separaria nada"


def test_os_estilos_sao_IMUTAVEIS() -> None:  # noqa: N802
    """O mapa é congelado: um construtor que o recebe não reescreve a escolha da pessoa."""
    estilos = estilos_dos_grupos(["a"])
    assert estilos is not None
    with pytest.raises(TypeError):
        estilos.por_grupo["b"] = EstiloDoGrupo(cor="azul", simbolo="círculo")  # type: ignore[index]
    with pytest.raises(AttributeError):
        estilos.por_grupo = {}  # type: ignore[misc]

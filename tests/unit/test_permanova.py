"""PERMANOVA e o teste de dispersão.

## O que este arquivo prova, e por que os TRÊS pares são o centro dele

Afirma-se que a PERMANOVA e o teste de dispersão **medem coisas diferentes**, e que por isso o
`p` da primeira sozinho não separa *"os centroides diferem"* de *"a dispersão difere"*. Sem os três
cenários de par, essa afirmação seria prosa: um teste de dispersão que devolvesse o mesmo `p` da
PERMANOVA passaria em qualquer caminho feliz, e a decisão de os publicar juntos ficaria sem
evidência nenhuma.

**Cada cenário calcula as DUAS quantidades sobre o dado da planta e afirma que divergem, antes de
olhar o resultado da função sob teste**: a distância entre centroides e a dispersão média
por grupo. Sem essa âncora, um caso em que as duas coincidissem por acidente faria o par passar
sem separar nada.

## As duas âncoras de instrumento, e elas correm ANTES dos cenários

`_pseudo_f` e `_f_de_um_fator` são aritmética escrita à mão sobre uma matriz de distâncias e sobre
um vetor. As duas primeiras funções deste arquivo alimentam-nas com dado cuja resposta já se sabe —
a forma do centroide para a primeira, `scipy.stats.f_oneway` para a segunda. Ler um `p` antes de
saber que o instrumento mede o que diz seria a ordem invertida invertida.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
from scipy.stats import f_oneway

from refino.domain.cancellation import ExecutionCancelledError
from refino.statistics.permanova import (
    MIN_AMOSTRAS_POR_GRUPO,
    MIN_GRUPOS,
    PermanovaResult,
    _f_de_um_fator,
    _pseudo_f,
    _quadrados_das_distancias,
    permanova,
)

SEMENTE: Final = 20260917
"""A semente das plantas. **Uma constante e não um literal repetido:** os três cenários de par têm
de ser reproduzíveis juntos, e um número diferente por cenário tornaria a comparação entre eles
dependente de qual sorteio calhou."""

PERMUTACOES_DO_TESTE: Final = 999

# **Os dois limiares das plantas são RELATIVOS à dispersão dentro dos grupos, e não números
# escolhidos.** A pseudo-F compara separação de centroides contra dispersão interna, então é contra
# ela que «separado» e «junto» significam alguma coisa: um limiar absoluto seria limiar sem origem,
# e a primeira versão deste arquivo tinha um — `0,6` — que reprovou porque a distância
# entre dois centroides de 20 pontos de `N(0, I₂)` é uma Rayleigh de escala `0,32`, e `0,67` está a
# 1,7 desvios dela. O que muda não é o número: é ele passar a ter de onde vir.
SEPARADO: Final = 3.0
"""Quantas vezes a dispersão interna a quantidade separada tem de exceder, para a planta ter
aterrado."""

RELATIVO: Final = 0.25
"""Quanto duas quantidades podem diferir, em fração da maior, para contarem como iguais."""


def _codigos(tamanhos: tuple[int, ...]) -> np.ndarray:
    """`(3, 2)` → `[0, 0, 0, 1, 1]`. Os rótulos como índice, na ordem dos blocos."""
    return np.repeat(np.arange(len(tamanhos)), tamanhos)


def _rotulos(tamanhos: tuple[int, ...]) -> list[str]:
    """`(2, 1)` → `["g0", "g0", "g1"]`."""
    return [f"g{k}" for k in _codigos(tamanhos).tolist()]


def _distancia_entre_centroides(x: np.ndarray, codigos: np.ndarray) -> float:
    """A distância euclidiana entre os dois centroides — a quantidade que a PERMANOVA persegue."""
    centroides = np.stack([x[codigos == k].mean(axis=0) for k in np.unique(codigos)])
    return float(np.linalg.norm(centroides[0] - centroides[1]))


def _dispersoes(x: np.ndarray, codigos: np.ndarray) -> tuple[float, ...]:
    """A distância média ao próprio centroide, por grupo — a quantidade do teste de dispersão."""
    saida: list[float] = []
    for k in np.unique(codigos):
        bloco = x[codigos == k]
        saida.append(float(np.linalg.norm(bloco - bloco.mean(axis=0), axis=1).mean()))
    return tuple(saida)


# ---------------------------------------------------------------------------
# 1. As duas âncoras de instrumento, e elas vêm ANTES de qualquer `p`
# ---------------------------------------------------------------------------


def test_a_pseudo_f_pela_matriz_de_DISTANCIAS_bate_a_forma_do_CENTROIDE() -> None:  # noqa: N802
    """A âncora barata: alimentar o instrumento com algo cuja resposta já se sabe.

    **A PERMANOVA é definida sobre a matriz de distâncias**, e é assim que `_pseudo_f` a calcula —
    é o que autoriza a figura a escrever o nome do método. Sobre distância **euclidiana** a soma de
    quadrados dentro de cada grupo tem uma segunda forma, pela identidade de Huygens: a soma dos
    quadrados das distâncias ao centroide daquele grupo.

    As duas formas são a mesma quantidade e o código só implementa uma. Esta afirmação é o que
    impede um erro de fator dois — o `d.sum()` conta cada par duas vezes — de passar por todos os
    cenários de par, que só leem `p` e não `F`.
    """
    gerador = np.random.default_rng(SEMENTE)
    x = gerador.normal(size=(24, 3))
    codigos = _codigos((10, 14))

    pela_distancia = _pseudo_f(_quadrados_das_distancias(x), codigos, 2)

    total = float(((x - x.mean(axis=0)) ** 2).sum())
    dentro = 0.0
    for k in (0, 1):
        bloco = x[codigos == k]
        dentro += float(((bloco - bloco.mean(axis=0)) ** 2).sum())
    pelo_centroide = ((total - dentro) / 1) / (dentro / (24 - 2))

    assert pela_distancia == pytest.approx(pelo_centroide, rel=1e-10), (
        f"a forma da distância dá {pela_distancia} e a do centroide {pelo_centroide}. Sobre "
        "distância euclidiana são a MESMA quantidade — divergir aqui é erro de fator, e nenhum "
        "cenário de `p` o apanharia"
    )


def test_o_f_do_teste_de_dispersao_bate_o_f_oneway_do_SCIPY() -> None:  # noqa: N802
    """A segunda âncora: o `F` de uma via escrito à mão contra o do `scipy`.

    **Escrito à mão e não `f_oneway` no laço, e a razão é medida no custo**: a chamada do `scipy`
    valida entrada e monta objetos a cada uma das 999 permutações. O que se paga por escrevê-lo é
    exactamente o risco que esta função remove.
    """
    gerador = np.random.default_rng(SEMENTE)
    z = gerador.gamma(shape=2.0, size=30)
    codigos = _codigos((9, 11, 10))

    meu = _f_de_um_fator(z, codigos, 3)
    do_scipy = float(f_oneway(*(z[codigos == k] for k in (0, 1, 2))).statistic)

    assert meu == pytest.approx(do_scipy, rel=1e-10), (
        f"o meu `F` é {meu} e o do `scipy` {do_scipy}. O teste de dispersão compara distâncias ao "
        "centroide por ANOVA de uma via, e um `F` diferente do dela não é esse teste"
    )


# ---------------------------------------------------------------------------
# 2. OS TRÊS PARES — a evidência
# ---------------------------------------------------------------------------


def test_grupos_IDENTICOS_por_construcao_nao_separam_em_nenhum_dos_dois() -> None:  # noqa: N802
    """Caso nulo: a MESMA nuvem em dois grupos. Os dois `p` no máximo, e por construção.

    **«Idênticos por construção» e não «uma nuvem sorteada partida em duas», e a diferença foi
    medida.** A primeira versão deste teste partia 40 pontos de `N(0, I₂)` ao meio e afirmava `p >
    0,05`. Ela reprovou com **`p = 0,050`** — e não por defeito nenhum: sob o nulo o `p` é
    **uniforme**, então uma semente em vinte cai no ou abaixo do limiar, e esta caiu (a distância
    entre os centroides sorteados ficou a 1,7 desvios da Rayleigh que os descreve). Procurar uma
    semente que passasse seria escolher o número em vez do argumento.

    Com os dois grupos **iguais ponto a ponto**, os centroides coincidem exactamente: `SS_entre` é
    zero, `F` observado é zero, e **nenhuma** permutação pode dar menos que zero — os dois `p`
    saem em `1,0`, sem sorteio a decidir. É o que a asserção afirma.

    **E isto não é um teste que passaria com qualquer implementação:** um `p` normalizado por
    `acusadas / permutacoes` daria `0,999` aqui, e os dois cenários abaixo excluem uma
    implementação que devolvesse `1,0` sempre.
    """
    nuvem = np.random.default_rng(SEMENTE).normal(size=(20, 2))
    x = np.vstack([nuvem, nuvem])
    codigos = _codigos((20, 20))

    # **A planta aterrou?** As duas quantidades são EXACTAMENTE iguais aqui, e é isso que faz deste
    # o caso nulo verdadeiro em vez de um sorteio que calhou nulo.
    a, b = _dispersoes(x, codigos)
    assert _distancia_entre_centroides(x, codigos) == pytest.approx(0.0, abs=1e-12)
    assert a == pytest.approx(b, rel=1e-12), f"as dispersões diferem: {a} e {b}"

    saida = permanova(
        x, _rotulos((20, 20)), permutations=PERMUTACOES_DO_TESTE, seed=SEMENTE, cancelled=None
    )

    assert saida.statistic == pytest.approx(0.0, abs=1e-20), (
        f"pseudo-F = {saida.statistic} com os centroides no mesmo ponto: `SS_entre` não é zero"
    )
    assert saida.pvalue == 1.0, f"a PERMANOVA separou grupos idênticos: p = {saida.pvalue}"
    assert saida.dispersion_pvalue == 1.0, (
        f"o teste de dispersão separou grupos idênticos: p = {saida.dispersion_pvalue}"
    )


def test_centroides_DESLOCADOS_com_a_mesma_dispersao_separam_so_na_permanova() -> None:  # noqa: N802
    """**O par que prova que os dois testes medem coisas diferentes.**

    Duas nuvens de mesma forma, uma empurrada no eixo. A PERMANOVA tem de acusar; o teste de
    dispersão tem de **não** acusar, porque a distância de cada ponto ao centroide **do próprio
    grupo** não mudou — e é essa a quantidade que ele compara.

    Sem este cenário, um "teste de dispersão" que fosse a PERMANOVA com outro nome passaria em
    tudo o resto deste arquivo.
    """
    gerador = np.random.default_rng(SEMENTE)
    # `noqa: RUF005` — o `+ [5.0, 0.0]` é BROADCAST de numpy sobre o eixo dos componentes, e
    # não concatenação de lista. A sugestão do `ruff` achataria a nuvem num vetor de 42.
    x = np.vstack(
        [
            gerador.normal(size=(20, 2)),
            gerador.normal(size=(20, 2)) + [5.0, 0.0],  # noqa: RUF005
        ]
    )
    codigos = _codigos((20, 20))

    # **A planta aterrou, e as duas quantidades DIVERGEM** — a de cima muito, a de baixo pouco.
    a, b = _dispersoes(x, codigos)
    assert _distancia_entre_centroides(x, codigos) > SEPARADO * min(a, b), (
        "a planta do deslocamento não aterrou"
    )
    assert abs(a - b) < RELATIVO * max(a, b), f"a planta moveu também a dispersão: {a} e {b}"

    saida = permanova(
        x, _rotulos((20, 20)), permutations=PERMUTACOES_DO_TESTE, seed=SEMENTE, cancelled=None
    )

    assert saida.pvalue <= 0.01, (
        f"a PERMANOVA não viu 5 desvios de deslocamento: p = {saida.pvalue}"
    )
    assert saida.dispersion_pvalue > 0.05, (
        f"o teste de dispersão acusou um deslocamento de centroide: p = {saida.dispersion_pvalue}. "
        "Ele compara distância ao centroide do PRÓPRIO grupo, que o deslocamento não move"
    )


def test_dispersoes_DIFERENTES_com_o_mesmo_centroide_separam_so_na_dispersao() -> None:  # noqa: N802
    """**A outra metade do par, e sem ela a acusação seria afirmação sem evidência.**

    Duas nuvens concêntricas, uma muito mais larga. O teste de dispersão tem de acusar; a
    PERMANOVA não, porque os centroides coincidem — e é exactamente este o caso em que ler o `p`
    dela sozinho levaria a concluir *"os grupos não diferem"* sobre um dado em que eles diferem
    muito.

    **Os grupos têm o MESMO `n` de propósito.** A pseudo-F é conhecidamente sensível a
    heterogeneidade de dispersão quando os tamanhos são desbalanceados; com `n` igual ela é
    robusta, e é essa a configuração em que a separação entre os dois testes é atribuível.
    """
    gerador = np.random.default_rng(SEMENTE)
    x = np.vstack([gerador.normal(size=(25, 2)), gerador.normal(size=(25, 2)) * 8.0])
    codigos = _codigos((25, 25))

    # **A planta aterrou, e as duas quantidades DIVERGEM — agora ao contrário do cenário acima.**
    a, b = _dispersoes(x, codigos)
    assert b > SEPARADO * a, f"a planta da dispersão não aterrou: {a} e {b}"
    assert _distancia_entre_centroides(x, codigos) < min(a, b), (
        "a planta moveu também os centroides"
    )

    saida = permanova(
        x, _rotulos((25, 25)), permutations=PERMUTACOES_DO_TESTE, seed=SEMENTE, cancelled=None
    )

    assert saida.dispersion_pvalue <= 0.01, (
        f"o teste de dispersão não viu um fator 8 de dispersão: p = {saida.dispersion_pvalue}"
    )
    assert saida.pvalue > 0.05, (
        f"a PERMANOVA acusou separação entre nuvens concêntricas: p = {saida.pvalue}. É o caso que "
        "KTD9 nomeia, e o teste de dispersão é o que o desambigua"
    )


@pytest.mark.parametrize("deslocamento", [2.0, 5.0, 12.0, 40.0])
def test_transladar_um_grupo_MOVE_a_permanova_e_nao_move_a_dispersao(  # noqa: N802
    deslocamento: float,
) -> None:
    """**A propriedade EXACTA que separa as duas metades do par, e ela não tem limiar.**

    Transladar um grupo inteiro muda a distância entre centroides e não muda distância nenhuma ao
    centroide do **próprio** grupo — o centroide anda com o grupo. Então: a pseudo-F tem de se
    mover, e o `p` da dispersão tem de ficar **idêntico**, bit a bit, em todos os deslocamentos.

    **Esta é a asserção mais forte deste arquivo, porque é uma igualdade e não uma desigualdade.**
    Os três cenários de par afirmam *"um `p` alto e outro baixo"*, o que precisa de limiares; este
    afirma *"este número não se move"*, que não precisa de nenhum.

    ## Por que ele existe: uma planta que NENHUM dos outros dezassete testes apanhou

    Medido em 18/09/2026. A docstring do módulo afirmava que recalcular os centroides dentro do laço
    das permutações *"faria o teste responder à separação de centroides"*, e a planta que fez
    exactamente isso **passou nos dezassete testes**. A afirmação estava na docstring e em nenhum
    mecanismo: a verificação nunca existiu.

    Remedido, o efeito é **deriva e não inversão**: com os centroides recalculados o `p` da
    dispersão vai de 0,704 a **0,298** entre deslocamentos de 2 e 40 desvios, contra **0,701 fixo**
    nos quatro. A docstring foi corrigida para dizer a magnitude, e este teste é o mecanismo que
    faltava.

    ## A igualdade é exacta no `p` e APROXIMADA na estatística, e a distinção é medida

    A primeira versão afirmava igualdade exacta nos dois, e reprovou em três dos quatro
    deslocamentos: `0,17481858355208058` contra `0,17481858355208174`. A invariância é exacta na
    aritmética real e não em `float64` — a média de `x + 40` perde cerca de três dígitos, e a
    distância ao centroide herda a perda. O `p` **é** exactamente igual porque é uma razão de
    contagens, e as contagens não se movem com um erro de 1e-15. Então: igualdade exacta onde ela
    é verdadeira, e tolerância relativa onde o `float64` manda — nunca tolerância nos dois por
    comodidade.
    """
    gerador = np.random.default_rng(SEMENTE)
    base = gerador.normal(size=(20, 2))
    outro = gerador.normal(size=(20, 2))
    rotulos = _rotulos((20, 20))

    junto = permanova(
        np.vstack([base, outro]), rotulos, permutations=199, seed=SEMENTE, cancelled=None
    )
    afastado = permanova(
        np.vstack([base, outro + [deslocamento, 0.0]]),  # noqa: RUF005 - broadcast, não concatenação
        rotulos,
        permutations=199,
        seed=SEMENTE,
        cancelled=None,
    )

    # **A planta aterrou:** a translação move de facto a quantidade que a PERMANOVA persegue.
    assert afastado.statistic > 2.0 * junto.statistic, (
        f"a pseudo-F foi de {junto.statistic} para {afastado.statistic} com um deslocamento de "
        f"{deslocamento} desvios: a planta não separa, e a igualdade abaixo passaria por acidente"
    )

    # E a metade da dispersão não se move — exacta no `p`, aproximada na estatística. Ver a
    # docstring: exacta na aritmetica real; o `float64` perde tres digitos ao transladar por 40.
    assert afastado.dispersion_statistic == pytest.approx(junto.dispersion_statistic, rel=1e-12), (
        f"o `F` da dispersão foi de {junto.dispersion_statistic} para "
        f"{afastado.dispersion_statistic}: a distância ao centroide do próprio grupo não pode "
        "depender de onde o grupo está"
    )
    assert afastado.dispersion_pvalue == junto.dispersion_pvalue, (
        f"o `p` da dispersão foi de {junto.dispersion_pvalue} para "
        f"{afastado.dispersion_pvalue}. Os centroides estão a ser RECALCULADOS dentro do laço "
        "das permutações, e a distribuição nula desta metade passou a depender da separação que a "
        "OUTRA metade testa"
    )
    assert afastado.mean_distance_to_centroid == pytest.approx(
        junto.mean_distance_to_centroid, rel=1e-12
    )


# ---------------------------------------------------------------------------
# 3. Determinismo, granularidade e o ponto de parada
# ---------------------------------------------------------------------------


def test_a_mesma_semente_repete_o_par_e_outra_semente_MOVE_algum_dos_dois() -> None:  # noqa: N802
    """Regra 3 do README, com a planta que prova que a semente não é decorativa.

    **A segunda metade é a que separa.** Uma implementação que ignorasse `seed` — sorteando de um
    gerador global — passaria a primeira asserção sempre que as duas chamadas corressem na mesma
    ordem, e este arquivo não teria como o notar.
    """
    x = np.random.default_rng(SEMENTE).normal(size=(30, 2))
    rotulos = _rotulos((15, 15))

    um = permanova(x, rotulos, permutations=199, seed=7, cancelled=None)
    outro = permanova(x, rotulos, permutations=199, seed=7, cancelled=None)
    assert (um.pvalue, um.dispersion_pvalue) == (outro.pvalue, outro.dispersion_pvalue)
    assert (um.statistic, um.dispersion_statistic) == (outro.statistic, outro.dispersion_statistic)

    terceiro = permanova(x, rotulos, permutations=199, seed=99, cancelled=None)
    assert (terceiro.pvalue, terceiro.dispersion_pvalue) != (um.pvalue, um.dispersion_pvalue), (
        "duas sementes diferentes deram o MESMO par de p sobre a mesma entrada: o gerador não "
        "está a ser semeado pelo argumento"
    )
    # A estatística observada NÃO depende da semente — só o `p` depende.
    assert terceiro.statistic == um.statistic
    assert terceiro.dispersion_statistic == um.dispersion_statistic


@pytest.mark.parametrize("quantas", [9, 99, 999])
def test_o_p_e_sempre_um_multiplo_de_um_sobre_permutacoes_mais_um(quantas: int) -> None:
    """O granulado do `p` é `1/(N+1)`, e é isso que o número de permutações muda.

    **`(1 + acusadas) / (1 + permutacoes)` e não `acusadas / permutacoes`**: o segundo admite `p =
    0`, que afirmaria probabilidade nula a partir de uma amostra finita. O menor `p` obtenível com
    `N` sorteios é `1/(N+1)`, e a figura declara `N` por isso.
    """
    x = np.random.default_rng(SEMENTE).normal(size=(20, 2))
    saida = permanova(x, _rotulos((10, 10)), permutations=quantas, seed=SEMENTE, cancelled=None)

    for p in (saida.pvalue, saida.dispersion_pvalue):
        assert p >= 1.0 / (quantas + 1) - 1e-12, f"p = {p} é menor que o mínimo obtenível"
        assert p <= 1.0
        assert round(p * (quantas + 1)) == pytest.approx(p * (quantas + 1), abs=1e-9), (
            f"p = {p} não é múltiplo de 1/{quantas + 1}"
        )
    assert saida.permutations == quantas


def test_o_cancelamento_PARA_dentro_do_laco_das_permutacoes() -> None:  # noqa: N802
    """O ponto de parada está no laço do domínio, não só antes da chamada.

    **As duas direções, e o par causal é a resposta do verificador.** Com `False` constante a
    chamada completa; com `True` depois de algumas consultas ela levanta, e o `where` nomeia o
    ponto. Só a segunda prova que o laço consulta; só a primeira prova que ele não levanta sozinho.
    """
    x = np.random.default_rng(SEMENTE).normal(size=(20, 2))
    rotulos = _rotulos((10, 10))

    consultas = 0

    def nunca() -> bool:
        nonlocal consultas
        consultas += 1
        return False

    permanova(x, rotulos, permutations=50, seed=SEMENTE, cancelled=nunca)
    assert consultas >= 50, (
        f"o laço consultou o verificador {consultas} vezes para 50 permutações: o ponto de parada "
        "não está DENTRO do laço, e cancelar durante o cálculo não teria efeito"
    )

    restantes = 5

    def depois_de_cinco() -> bool:
        nonlocal restantes
        restantes -= 1
        return restantes < 0

    with pytest.raises(ExecutionCancelledError) as erro:
        permanova(x, rotulos, permutations=999, seed=SEMENTE, cancelled=depois_de_cinco)
    assert "permanova" in erro.value.where, erro.value.where


# ---------------------------------------------------------------------------
# 4. As recusas de população, e o que NÃO é descartado em silêncio
# ---------------------------------------------------------------------------


def test_recusa_populacao_com_menos_de_dois_grupos() -> None:
    x = np.random.default_rng(SEMENTE).normal(size=(10, 2))
    with pytest.raises(ValueError, match=str(MIN_GRUPOS)):
        permanova(x, ["a"] * 10, permutations=9, seed=SEMENTE, cancelled=None)


def test_recusa_grupo_com_uma_amostra_so() -> None:
    x = np.random.default_rng(SEMENTE).normal(size=(10, 2))
    with pytest.raises(ValueError, match=str(MIN_AMOSTRAS_POR_GRUPO)):
        permanova(x, ["a"] * 9 + ["b"], permutations=9, seed=SEMENTE, cancelled=None)


def test_recusa_rotulos_desalinhados_dos_scores() -> None:
    """Comprimentos diferentes significam que um dos dois é de outra população."""
    x = np.random.default_rng(SEMENTE).normal(size=(10, 2))
    with pytest.raises(ValueError, match="10"):
        permanova(x, _rotulos((4, 4)), permutations=9, seed=SEMENTE, cancelled=None)


def test_recusa_nuvem_degenerada_em_que_todos_os_pontos_COINCIDEM() -> None:  # noqa: N802
    """`0/0` não é «não há separação»: é ausência de medida, e as duas pedem desfecho diferente."""
    x = np.zeros((10, 2))
    with pytest.raises(ValueError, match="mesmo ponto"):
        permanova(x, _rotulos((5, 5)), permutations=9, seed=SEMENTE, cancelled=None)


def test_linhas_nao_finitas_sao_DESCARTADAS_e_contadas() -> None:  # noqa: N802
    """Nada sai em silêncio: a linha não desenhável não entra no cálculo e aparece na contagem.

    **O par causal é contra a mesma nuvem sem a linha estragada**: os dois resultados têm de
    coincidir em tudo menos na contagem. Sem ele, uma implementação que propagasse `nan` para a
    estatística passaria — `nan` não levanta.
    """
    gerador = np.random.default_rng(SEMENTE)
    limpo = gerador.normal(size=(20, 2))
    sujo = np.vstack([limpo, [np.nan, 0.0]])

    saida = permanova(
        sujo, [*_rotulos((10, 10)), "g0"], permutations=99, seed=SEMENTE, cancelled=None
    )
    referencia = permanova(limpo, _rotulos((10, 10)), permutations=99, seed=SEMENTE, cancelled=None)

    assert saida.n_dropped_non_finite == 1
    assert referencia.n_dropped_non_finite == 0
    assert saida.statistic == pytest.approx(referencia.statistic)
    assert saida.n_per_group == referencia.n_per_group


def test_a_populacao_do_resultado_e_a_que_ENTROU_no_calculo() -> None:  # noqa: N802
    """Os campos que a figura declara descrevem o que correu, e não o que foi passado.

    É o rótulo certo sobre a quantidade errada, na forma que morde: uma contagem tirada do argumento
    em vez do que sobrou do descarte poria o `n` errado no rodapé, com o número certo ao lado.
    """
    x = np.random.default_rng(SEMENTE).normal(size=(21, 3))
    saida = permanova(x, _rotulos((7, 6, 8)), permutations=99, seed=SEMENTE, cancelled=None)

    assert isinstance(saida, PermanovaResult)
    assert saida.groups == ("g0", "g1", "g2")
    assert saida.n_per_group == (7, 6, 8)
    assert saida.n_components == 3
    assert saida.seed == SEMENTE
    assert len(saida.mean_distance_to_centroid) == 3


def test_o_par_e_INDIVISIVEL_por_construcao_e_nao_por_convencao() -> None:  # noqa: N802
    """O par mecanizado, no molde do critério: o tipo é que impede publicar um sem o outro.

    **Introspecção e não prosa.** Os dois testes são um par indivisível; uma docstring
    a dizê-lo não impede a próxima sessão de acrescentar `apenas_a_permanova()` ao lado. O que
    impede é (a) o módulo exportar **uma** função de cálculo e (b) os quatro campos do par serem
    obrigatórios no resultado — sem padrão, então não há como construir metade dele.

    É o mesmo mecanismo que guarda a regra 6 do README no PLS-DA binário, e pelo mesmo motivo:
    afrouxá-lo seria demolir a mecanização de uma decisão.
    """
    from dataclasses import MISSING, fields

    from refino.statistics import permanova as modulo

    publicas = {
        nome
        for nome in modulo.__all__
        if callable(getattr(modulo, nome)) and not isinstance(getattr(modulo, nome), type)
    }
    assert publicas == {"permanova"}, (
        f"o módulo exporta {sorted(publicas)} funções. Uma segunda tornaria possível publicar a "
        "PERMANOVA sem o teste de dispersão, que é exactamente o que KTD9 proíbe"
    )

    do_par = {"statistic", "pvalue", "dispersion_statistic", "dispersion_pvalue"}
    com_padrao = {
        campo.name
        for campo in fields(PermanovaResult)
        if campo.name in do_par
        and (campo.default is not MISSING or campo.default_factory is not MISSING)
    }
    assert com_padrao == set(), (
        f"os campos {sorted(com_padrao)} têm padrão. Um campo do par com padrão deixa construir "
        "metade do resultado, e a metade que falta lê-se como «não deu nada»"
    )
    assert do_par <= {campo.name for campo in fields(PermanovaResult)}

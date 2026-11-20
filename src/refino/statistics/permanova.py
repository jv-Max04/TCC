"""PERMANOVA e teste de dispersão sobre os componentes da PCA.

Os dois testes saem de uma função só e de um resultado só. O `p` da PERMANOVA é confundido por
heterogeneidade de dispersão: sozinho não separa 'os centroides dos grupos diferem' de 'a
dispersão dentro dos grupos difere'. Duas nuvens concêntricas, uma bem mais larga que a outra,
dão `p` alto na PERMANOVA sobre grupos que diferem muito. Por isso `PermanovaResult` carrega os
campos dos dois testes sem valor padrão, e os dois partilham a mesma sequência de permutações,
num laço só.

A distância é euclidiana, porque a entrada são scores de PCA: contínuos, centrados, com sinal.
Bray-Curtis e as outras distâncias de ecologia, de onde a PERMANOVA vem, supõem contagens não
negativas. A soma de quadrados dentro de um grupo é a dos quadrados das distâncias par a par
dividida pelo `n` do grupo, a definição de Anderson (2001). Sobre distância euclidiana ela é
igual à soma dos quadrados das distâncias ao centroide, e `tests/unit/test_permanova.py` confere
a igualdade.

O teste de dispersão é o PERMDISP de Anderson (2006), na forma de
`vegan::permutest.betadisper`: as distâncias ao centroide são calculadas uma vez, a partir do
agrupamento observado, e o que se permuta são os rótulos. Cada distância é ao centroide do
próprio grupo, então transladar um grupo inteiro não move o `p` da dispersão e move a pseudo-F.
Recalcular os centroides a cada permutação faria a distribuição nula da dispersão depender do
deslocamento entre os grupos, que é a alternativa que a outra metade testa.

O módulo não calcula a PCA: recebe os scores, e quem chama decide de qual decomposição eles
saem. Também não filtra a população: quem sabe que QC não é grupo clínico é quem chama, que tem
a máscara.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from refino.domain.cancellation import CancelCheck, raise_if_cancelled
from refino.statistics.anova import MIN_GRUPOS
from refino.statistics.univariate import MIN_AMOSTRAS_POR_GRUPO

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Sequence

__all__ = [
    "MIN_AMOSTRAS_POR_GRUPO",
    "MIN_GRUPOS",
    "PERMUTACOES",
    "PONTO_DE_PARADA",
    "PermanovaResult",
    "PermanovaSemPopulacaoError",
    "permanova",
]


class PermanovaSemPopulacaoError(ValueError):
    """A matriz não tem grupos que sustentem o teste. Não é defeito de quem chamou.

    É uma classe à parte para quem chama distinguir um fato sobre o dado, como uma matriz com um
    grupo clínico só, de um defeito de programação, como rótulos desalinhados dos scores, que
    sobe como `ValueError` comum. Desce de `ValueError` para quem não quer a distinção continuar
    a apanhá-la.
    """


PERMUTACOES = 999
"""Quantos sorteios de rótulo o `p` usa por omissão.

É o número do MetaboAnalyst e o valor de `AnalysisSpec.permanova_permutations`. Quem chama da
fronteira passa o campo da especificação, e não este padrão. O menor `p` obtenível com `N`
sorteios é `1/(N+1)`, então 999 dá `0,001`.
"""

PONTO_DE_PARADA = "permanova: laco das permutacoes"
"""O nome do ponto em que o cancelamento é consultado, no laço das permutações."""


@dataclass(frozen=True)
class PermanovaResult:
    """Os dois testes do par e a população em que cada um correu.

    É um objeto só: dois tipos permitiriam devolver só a PERMANOVA, cujo `p` sozinho não decide
    nada.

    `statistic` tem nome neutro, e não `pseudo_f`, como em `AnovaResult`: quem escreve a legenda
    tem de dizer qual teste a produziu.
    """

    groups: tuple[str, ...]
    """Os rótulos em ordem canônica: `sorted`, e não a de aparição.

    A ordem decide a das contagens e a das dispersões, e a de aparição faria o mesmo grupo
    trocar de posição entre duas matrizes que diferem só na ordem das injeções."""

    n_per_group: tuple[int, ...]
    n_components: int
    permutations: int
    seed: int

    statistic: float
    """A pseudo-F de Anderson (2001) sobre a matriz de distâncias euclidianas."""

    pvalue: float
    dispersion_statistic: float
    """O `F` de uma via sobre as distâncias ao centroide do próprio grupo."""

    dispersion_pvalue: float

    mean_distance_to_centroid: tuple[float, ...]
    """A dispersão média de cada grupo, na ordem de `groups`.

    Um `p` baixo no teste de dispersão diz que as dispersões diferem e não diz qual grupo é o
    mais largo. Estes números dão a direção."""

    n_dropped_non_finite: int
    """Quantas linhas saíram por terem score não finito.

    Uma linha com `nan` num componente contamina a matriz de distâncias inteira: a estatística
    sairia `nan` e o `p` sairia o mínimo, porque `nan >= F` é `False`. Por isso a linha é
    descartada, e o descarte é contado."""


def permanova(
    scores: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    *,
    permutations: int = PERMUTACOES,
    seed: int,
    cancelled: CancelCheck | None = None,
) -> PermanovaResult:
    """Testa a separação entre grupos nos componentes, e a dispersão deles, por permutação.

    `scores` é `(n_amostras, n_componentes)` e `groups` traz um rótulo por linha, alinhado.
    Comprimentos diferentes são recusados.

    `seed` é obrigatório e sem padrão: toda operação estocástica recebe semente declarada.

    `cancelled` é consultado dentro do laço das permutações, que é onde o tempo é gasto.

    Levanta `ValueError` quando a população não sustenta o teste: menos de `MIN_GRUPOS` grupos,
    grupo com menos de `MIN_AMOSTRAS_POR_GRUPO` amostras, rótulos desalinhados, todos os pontos
    coincidentes, todos os grupos com duas amostras (a dispersão de um par não varia, por
    geometria), ou distâncias ao centroide sem variância dentro de nenhum grupo.
    """
    if permutations < 1:
        raise ValueError(
            f"`permutations` é {permutations} e o mínimo é 1: um número de sorteios não positivo "
            "não produz `p` nenhum, e o menor obtenível com N sorteios é 1/(N+1)."
        )

    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(
            f"`scores` tem forma {x.shape} e a PERMANOVA precisa de uma matriz "
            "(n_amostras, n_componentes) com ao menos uma de cada. Forma vazia chega aqui quando "
            "a PCA não correu."
        )

    rotulos_crus = [str(g) for g in groups]
    if len(rotulos_crus) != x.shape[0]:
        raise ValueError(
            f"{len(rotulos_crus)} rótulo(s) de grupo para {x.shape[0]} linha(s) de score. Os dois "
            "descrevem as mesmas amostras, e comprimentos diferentes significam que um deles é de "
            "outra população."
        )

    # O descarte vem antes da contagem de grupos: um grupo que só tenha linhas não finitas deixa
    # de existir, e a recusa por população fala da população que de fato entrou.
    desenhaveis = np.isfinite(x).all(axis=1)
    n_descartadas = int((~desenhaveis).sum())
    x = x[desenhaveis]
    rotulos = np.asarray(
        [r for r, ok in zip(rotulos_crus, desenhaveis.tolist(), strict=True) if ok]
    )

    nomes = tuple(sorted(set(rotulos.tolist())))
    if len(nomes) < MIN_GRUPOS:
        raise PermanovaSemPopulacaoError(
            f"a PERMANOVA precisa de ao menos {MIN_GRUPOS} grupos e esta população tem "
            f"{len(nomes)}: {list(nomes)}. Com um grupo não há partição a testar, e "
            f"{n_descartadas} linha(s) saíram por score não finito."
        )

    codigos = np.searchsorted(np.asarray(nomes), rotulos)
    n_por_grupo = tuple(int((codigos == k).sum()) for k in range(len(nomes)))
    for nome, quantas in zip(nomes, n_por_grupo, strict=True):
        if quantas < MIN_AMOSTRAS_POR_GRUPO:
            raise PermanovaSemPopulacaoError(
                f"grupo {nome!r} tem {quantas} amostra(s) utilizável(eis); a PERMANOVA exige ao "
                f"menos {MIN_AMOSTRAS_POR_GRUPO} por grupo — com uma observação a distância ao "
                "centroide do próprio grupo é zero por construção, e o teste de dispersão dela "
                "não mede nada."
            )
    if all(quantas == 2 for quantas in n_por_grupo):
        # As duas distâncias ao centroide de um par são iguais por geometria, então o teste de
        # dispersão não tem variância dentro dos grupos. A recusa é pela contagem, e não pelo
        # `F`: em ponto flutuante a soma dentro vira ruído de arredondamento, e o `F` sairia ora
        # como divisão por zero, ora finito e enorme.
        raise PermanovaSemPopulacaoError(
            "todos os grupos têm duas amostras "
            f"({list(zip(nomes, n_por_grupo, strict=True))}): as duas distâncias ao centroide de "
            "cada par são iguais por geometria, a dispersão não varia dentro de nenhum grupo, e o "
            "teste de dispersão, metade do par, não mede nada."
        )

    n_grupos = len(nomes)
    quadrados = _quadrados_das_distancias(x)
    if not quadrados.any():
        raise PermanovaSemPopulacaoError(
            f"as {x.shape[0]} amostras estão todas no mesmo ponto do espaço dos componentes: a "
            "soma de quadrados total é zero e a pseudo-F seria 0/0. Ausência de medida e ausência "
            "de separação são fatos diferentes, e um `p` aqui afirmaria o segundo."
        )

    observada = _pseudo_f(quadrados, codigos, n_grupos)
    distancias = _distancias_ao_centroide(x, codigos, n_grupos)
    dispersao_observada = _f_de_um_fator(distancias, codigos, n_grupos)
    if not np.isfinite(dispersao_observada):
        # Grupos de três ou mais em que cada um é um ponto só: as distâncias são todas zero e a
        # dispersão é 0/0. Cobre a degenerescência exata que a contagem não vê, e não o
        # arredondamento.
        raise PermanovaSemPopulacaoError(
            "as distâncias ao centroide não variam dentro de nenhum grupo "
            f"({list(zip(nomes, n_por_grupo, strict=True))}), e o teste de dispersão, metade do "
            "par, não mede nada."
        )

    acusaram = 0
    acusaram_dispersao = 0
    gerador = np.random.default_rng(seed)
    for _ in range(permutations):
        raise_if_cancelled(cancelled, PONTO_DE_PARADA)
        # Um sorteio alimenta os dois testes.
        sorteados = gerador.permutation(codigos)
        if _pseudo_f(quadrados, sorteados, n_grupos) >= observada:
            acusaram += 1
        if _f_de_um_fator(distancias, sorteados, n_grupos) >= dispersao_observada:
            acusaram_dispersao += 1

    return PermanovaResult(
        groups=nomes,
        n_per_group=n_por_grupo,
        n_components=int(x.shape[1]),
        permutations=permutations,
        seed=seed,
        statistic=observada,
        pvalue=_p(acusaram, permutations),
        dispersion_statistic=dispersao_observada,
        dispersion_pvalue=_p(acusaram_dispersao, permutations),
        mean_distance_to_centroid=tuple(
            float(distancias[codigos == k].mean()) for k in range(n_grupos)
        ),
        n_dropped_non_finite=n_descartadas,
    )


def _p(acusaram: int, permutations: int) -> float:
    """`(1 + acusadas) / (1 + sorteios)`.

    Sem o `1 +` o menor `p` seria zero, que afirma probabilidade nula a partir de uma amostra
    finita. Com ele o piso é `1/(N+1)`.
    """
    return (1.0 + acusaram) / (1.0 + permutations)


def _quadrados_das_distancias(x: np.ndarray) -> np.ndarray:
    """A matriz `(n, n)` dos quadrados das distâncias euclidianas par a par.

    Usa a identidade `|a-b|² = |a|² + |b|² - 2a·b`, e não o produto externo de diferenças, que
    materializaria `(n, n, k)`.

    O `maximum(., 0)` existe porque a identidade produz negativos da ordem do épsilon de máquina
    para pares quase coincidentes, e a raiz de um negativo apareceria adiante como `nan`.
    """
    quadrados_das_normas = (x**2).sum(axis=1)
    fora = quadrados_das_normas[:, None] + quadrados_das_normas[None, :] - 2.0 * (x @ x.T)
    return np.asarray(np.maximum(fora, 0.0), dtype=np.float64)


def _pseudo_f(quadrados: np.ndarray, codigos: np.ndarray, n_grupos: int) -> float:
    """A pseudo-F de Anderson (2001) sobre a matriz de quadrados de distâncias.

    `SS_total = Σ_{i<j} d²_ij / n` e `SS_dentro = Σ_g Σ_{i<j ∈ g} d²_ij / n_g`. A soma da matriz
    cheia conta cada par duas vezes e a diagonal zero nenhuma, e daí vem o `2 *` nos dois
    denominadores.

    Devolve `inf` quando a soma de quadrados dentro dos grupos é zero e a de entre é positiva:
    os grupos são internamente constantes e diferentes entre si, que é separação perfeita. Como
    `inf >= inf` é `True`, uma permutação igualmente perfeita continua a contar.
    """
    n = quadrados.shape[0]
    total = float(quadrados.sum()) / (2.0 * n)
    dentro = 0.0
    for k in range(n_grupos):
        do_grupo = codigos == k
        bloco = quadrados[np.ix_(do_grupo, do_grupo)]
        dentro += float(bloco.sum()) / (2.0 * int(do_grupo.sum()))
    entre = total - dentro
    # A divisão é entre `np.float64`, e não entre `float` do Python: o `np.errstate` só governa
    # a aritmética do numpy, e com `float` a divisão por zero levantaria `ZeroDivisionError` em
    # vez de dar `inf`.
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(entre / (n_grupos - 1)) / np.float64(dentro / (n - n_grupos)))


def _distancias_ao_centroide(x: np.ndarray, codigos: np.ndarray, n_grupos: int) -> np.ndarray:
    """A distância euclidiana de cada ponto ao centroide do próprio grupo.

    Calculada uma vez, do agrupamento observado, e nunca dentro do laço. Ver o cabeçalho do
    módulo.
    """
    centroides = np.stack([x[codigos == k].mean(axis=0) for k in range(n_grupos)])
    return np.asarray(np.linalg.norm(x - centroides[codigos], axis=1), dtype=np.float64)


def _f_de_um_fator(valores: np.ndarray, codigos: np.ndarray, n_grupos: int) -> float:
    """O `F` da ANOVA de uma via sobre um vetor, escrito à mão.

    Não usa `scipy.stats.f_oneway` porque corre uma vez por permutação, e a chamada do `scipy`
    valida a entrada e monta um objeto de resultado a cada uma.
    `test_o_f_do_teste_de_dispersao_bate_o_f_oneway_do_SCIPY` confere que é o mesmo `F`.

    A permutação preserva os tamanhos dos grupos, então os graus de liberdade são os mesmos em
    todas as passadas.
    """
    n = valores.size
    media = float(valores.mean())
    entre = 0.0
    dentro = 0.0
    for k in range(n_grupos):
        bloco = valores[codigos == k]
        media_do_grupo = float(bloco.mean())
        entre += bloco.size * (media_do_grupo - media) ** 2
        dentro += float(((bloco - media_do_grupo) ** 2).sum())
    # Entre `np.float64` pelo mesmo motivo de `_pseudo_f`: num sorteio, um grupo pode ficar com
    # valores iguais, e a divisão por zero dá `inf` ou `nan` em vez de derrubar o laço.
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.float64(entre / (n_grupos - 1)) / np.float64(dentro / (n - n_grupos)))

"""A trajetória do PC1 dos QCs ao longo da corrida.

Um QC é a mesma mistura injetada periodicamente, então toda variação medida nele é técnica.
Os scores de PC1 dos QCs, na ordem em que foram injetados, mostram se o equipamento derivou
durante a corrida: sem correção os QCs tendem a se deslocar, e depois de SERRF ou LOESS devem
ficar agrupados. O eixo x é a ordem de injeção do `Sample_List`, nunca a posição da coluna:
as colunas de `measurements` vêm agrupadas por classe, e desenhar contra elas daria pontos
igualmente espaçados sobre o eixo errado, sem erro nenhum. Por isso `ordens_de_injecao` é
obrigatório e sua ausência é recusa.

Com poucos QCs por corrida, a figura não ajusta tendência: as únicas séries são os QCs
medidos, o traço é tracejado porque entre dois QCs nada foi medido, a contagem por lote vai
no título e na legenda, e QC sem score finito é contado no título em vez de sumir. Cada lote
é uma série própria, e nenhum traço atravessa a fronteira entre lotes. O sinal do PC1 é
arbitrário: o módulo não o corrige, só o declara no rótulo do eixo e no rodapé.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from refino.plots.base import (
    MatrizVaziaError,
    RecusaDoConstrutorError,
    exige_populacao,
    inteiro_do_fio,
    nova_figura,
)
from refino.plots.recorte import Recorte, aplica

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Sequence

    from matplotlib.figure import Figure

__all__ = [
    "AMOSTRAGEM",
    "MATRIZ",
    "METODO",
    "MINIMO_PARA_LINHA",
    "ROTULO_DE_X",
    "ROTULO_DE_Y",
    "TITULO",
    "LotesDesalinhadosError",
    "SemOrdemDeInjecaoError",
    "figura_da_deriva_do_qc",
    "frase_da_incerteza",
    "recorte_da_deriva",
    "rotulo_do_lote",
]

# Dois pontos fazem um segmento. Abaixo disso não há o que ligar, e o título diz isso por
# extenso.
MINIMO_PARA_LINHA: Final = 2

TITULO: Final = "Trajetória do PC1 dos QCs ao longo da corrida"

ROTULO_DE_X: Final = "ordem de injeção"

# O sinal do PC1 é arbitrário e não se compara entre execuções. Sem a ressalva no eixo,
# 'subiu' seria lido como fato do dado.
ROTULO_DE_Y: Final = "score de PC1 (sinal arbitrário)"

# Os campos do `Recorte` que esta figura declara sempre igual. Ficam ao lado do código que os
# torna verdadeiros, e não no chamador, para mudarem junto com o método.
AMOSTRAGEM: Final = "todos os QCs da corrida, na ordem de injeção do Sample_List — sem amostragem"

MATRIZ: Final = "a matriz ingerida, antes de qualquer correção"

METODO: Final = (
    "PC1 da PCA do diagnóstico, sem teste estatístico. O traço liga QCs consecutivos do mesmo lote "
    "e NÃO é leitura entre eles; nenhuma tendência é ajustada. O sinal do componente é arbitrário "
    "e não é comparável entre execuções — leia o movimento, não a direção."
)


class SemOrdemDeInjecaoError(RecusaDoConstrutorError):
    """A ordem de injeção não veio, ou não descreve estes pontos.

    A posição da coluna não serve de substituto: as colunas de `measurements` são agrupadas
    por classe, e a figura sairia bem formada sobre o eixo errado.
    """


class LotesDesalinhadosError(RecusaDoConstrutorError):
    """Há rótulos de lote a mais, a menos, ou nenhum.

    Fica separada de `SemOrdemDeInjecaoError` porque o conserto é outro: a ordem vem do
    `Sample_List`, e o lote vem da sugestão de lotes do diagnóstico.
    """


def rotulo_do_lote(lote: str, n_qcs: int) -> str:
    """O rótulo de legenda de um lote, com a contagem de QCs: `"lote A — 7 QCs"`.

    A contagem é por lote porque é ela que decide o que se pode ler: o total esconde o lote
    que tem só dois QCs.
    """
    plural = "QC" if n_qcs == 1 else "QCs"
    return f"lote {lote} — {n_qcs} {plural}"


def frase_da_incerteza(n_desenhados: int, n_sem_score: int) -> str:
    """A frase do título sobre a população desenhada.

    Diz quando há um ponto só, quando não há ajuste de tendência e quantos QCs ficaram sem
    score finito. Um QC sem score não é desenhado, e sem a contagem a lacuna no traço se
    confundiria com uma deriva abrupta.
    """
    plural = "QC" if n_desenhados == 1 else "QCs"
    corpo = (
        f"{n_desenhados} {plural} — um ponto não é trajetória"
        if n_desenhados < MINIMO_PARA_LINHA
        else f"{n_desenhados} {plural}, sem ajuste de tendência"
    )
    if n_sem_score:
        return f"{corpo}; {n_sem_score} sem score finito"
    return corpo


def recorte_da_deriva(*, n_features_usadas: Any, n_features_totais: Any) -> Recorte:
    """O `Recorte` desta figura, a partir das contagens de features da PCA.

    O construtor recebe só os scores dos QCs. Quantas features a PCA reteve e quantas a matriz
    tinha vêm de quem executou o diagnóstico.

    Sem população levanta `MatrizVaziaError`, que é a recusa que o painel da janela apanha e
    mostra como texto no lugar da figura.
    """
    totais = inteiro_do_fio(n_features_totais)
    if totais < 1:
        raise MatrizVaziaError(
            "a trajetória não declara de quantas features a PCA saiu: `n_features_totais` é "
            f"{n_features_totais!r}. Sem execução de diagnóstico não há população, e figura em "
            "branco seria indistinguível de figura quebrada"
        )
    return Recorte(
        desenhadas=inteiro_do_fio(n_features_usadas),
        disponiveis=totais,
        amostragem=AMOSTRAGEM,
        matriz=MATRIZ,
        metodo=METODO,
    )


def figura_da_deriva_do_qc(
    *,
    scores: Sequence[float] | np.ndarray,
    ordens_de_injecao: Sequence[float] | np.ndarray,
    lotes: Sequence[str],
    recorte: Recorte,
) -> Figure:
    """Os QCs em ordem de injeção, um traço por lote, com a população declarada na imagem.

    Desenha todos os QCs, sem amostragem. `scores` tem um valor por QC, na ordem das linhas da
    matriz, que `RefinoMatrix` garante ser não decrescente em ordem de injeção.
    `ordens_de_injecao` e `lotes` são alinhados a ele, um por QC.
    """
    pontos = exige_populacao("scores", scores)
    _exige_vetor("scores", pontos)
    ordens = _ordens_ou_recusa(ordens_de_injecao, pontos.size)
    rotulos = _lotes_ou_recusa(lotes, pontos.size)

    finitos = np.isfinite(pontos)
    if not finitos.any():
        raise MatrizVaziaError(
            f"há {pontos.size} QC(s) e nenhum com score finito: não há ponto a desenhar. "
            "Uma figura com o eixo montado e nada dentro lê-se como figura quebrada"
        )

    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)

    for lote in _lotes_na_ordem(rotulos):
        do_lote = np.array([r == lote for r in rotulos], dtype=bool) & finitos
        n_no_lote = int(do_lote.sum())
        if n_no_lote == 0:
            # Lote sem nenhum QC de score finito: uma entrada de legenda sem ponto prometeria
            # um traço que não existe.
            continue
        eixo.plot(
            ordens[do_lote],
            pontos[do_lote],
            marker="o",
            # Tracejado porque entre dois QCs nada foi medido.
            linestyle="--" if n_no_lote >= MINIMO_PARA_LINHA else "none",
            label=rotulo_do_lote(lote, n_no_lote),
        )

    eixo.set_xlabel(ROTULO_DE_X)
    eixo.set_ylabel(ROTULO_DE_Y)
    eixo.set_title(f"{TITULO}\n{frase_da_incerteza(int(finitos.sum()), int((~finitos).sum()))}")
    eixo.legend()

    # `aplica` escreve o rodapé e afasta os eixos dele.
    aplica(figura, recorte)
    return figura


# ---------------------------------------------------------------------------
# As guardas de fronteira
# ---------------------------------------------------------------------------


def _exige_vetor(nome: str, valores: np.ndarray) -> None:
    if valores.ndim != 1:
        raise SemOrdemDeInjecaoError(
            f"`{nome}` tem {valores.ndim} dimensão(ões) e a figura precisa de um vetor. "
            "Um escalar chega aqui quando o campo veio `null` do fio"
        )


def _ordens_ou_recusa(valores: Sequence[float] | np.ndarray, n_pontos: int) -> np.ndarray:
    """A ordem de injeção, ou a recusa. Nunca a posição da coluna."""
    ordens = np.asarray(valores, dtype=float)
    _exige_vetor("ordens_de_injecao", ordens)
    if ordens.size == 0:
        raise SemOrdemDeInjecaoError(
            f"há {n_pontos} score(s) de QC e nenhuma ordem de injeção. A ordem temporal existe "
            "apenas no `Sample_List`, e a ordem das colunas do `measurements` está agrupada por "
            "classe, e não por injeção: usá-la desenharia os QCs igualmente espaçados sobre "
            "o eixo errado, sem nada levantar"
        )
    if ordens.size != n_pontos:
        raise SemOrdemDeInjecaoError(
            f"{ordens.size} ordem(ns) de injeção para {n_pontos} score(s) de QC. Os dois vetores "
            "descrevem os mesmos QCs, e comprimentos diferentes significam que um deles é de outra "
            "população"
        )
    if not np.isfinite(ordens).all():
        raise SemOrdemDeInjecaoError(
            f"{int((~np.isfinite(ordens)).sum())} de {ordens.size} ordens de injeção não são "
            "finitas. Um `null` no fio chega aqui como `nan`, e um ponto sem posição no eixo do "
            "tempo não é desenhável"
        )
    if bool(np.any(np.diff(ordens) < 0.0)):
        raise SemOrdemDeInjecaoError(
            f"as ordens de injeção não são não decrescentes: {ordens.tolist()}. "
            "`RefinoMatrix` ordena as linhas por `injection_order` na construção, então um vetor "
            "que desce não é a ordem de injeção destes QCs — é outra coisa com o mesmo nome. "
            "Empate é permitido: a ingestão desempata por nome e não proíbe ordens repetidas"
        )
    return ordens


def _lotes_ou_recusa(lotes: Sequence[str], n_pontos: int) -> tuple[str, ...]:
    """Um rótulo de lote por QC, ou a recusa."""
    rotulos = tuple(str(rotulo) for rotulo in lotes)
    if len(rotulos) != n_pontos:
        raise LotesDesalinhadosError(
            f"{len(rotulos)} rótulo(s) de lote para {n_pontos} score(s) de QC. O lote decide se o "
            "traço atravessa ou não a fronteira, e um vetor desalinhado ligaria QCs de lotes "
            "diferentes. Um único lote declara-se como um rótulo repetido, nunca como lista vazia"
        )
    return rotulos


def _lotes_na_ordem(rotulos: Sequence[str]) -> tuple[str, ...]:
    """Os lotes distintos, na ordem da primeira aparição, que é a ordem de injeção.

    Ordenar alfabeticamente poria `"10"` antes de `"2"`. A ordem de aparição é a temporal
    porque as linhas chegam ordenadas por `injection_order`.
    """
    vistos: dict[str, None] = {}
    for rotulo in rotulos:
        vistos.setdefault(rotulo, None)
    return tuple(vistos)

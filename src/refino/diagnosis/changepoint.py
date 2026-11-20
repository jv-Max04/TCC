"""Detecção de ponto de mudança na trajetória de QC.

O retorno é ordem de injeção, não índice na trajetória. Com sete QCs o índice vai de
0 a 6, enquanto as ordens de injeção são, por exemplo, 1, 12, 23, 34, 45, 56 e 68.
`propose_batches` consome a mesma lista para rotular amostras pela ordem de injeção.

Com tão poucos QCs um ponto de mudança é hipótese a confirmar. A função devolve os
cortes e não afirma que houve degrau; quem decide isso é a regra 2 da classificação,
que exige também nível sustentado.
"""

import numpy as np

from refino.diagnosis.trajectory import MIN_SEGMENT, rss_minimizing_split

__all__ = ["detect_changepoints"]


def detect_changepoints(
    trajectory: np.ndarray,
    injection_orders: np.ndarray,
    min_segment: int = MIN_SEGMENT,
    penalty: float | None = None,
) -> list[int]:
    """Pontos de mudança na trajetória, devolvidos em ordem de injeção.

    Segmentação binária sobre modelo constante por segmento: acha o corte que minimiza
    a soma dos quadrados residuais (RSS), aceita-o se a redução de RSS superar a
    penalidade e repete nos dois lados enquanto houver pontos para cortar.

    Com `penalty=None` a penalidade sai do próprio sinal: `PENALTY_NOISE_MULTIPLIER`
    vezes a variância do ruído estimada por `_sigma_ruido`. Um valor absoluto ficaria
    preso a uma escala de score, e a trajetória de PC1 não tem unidade fixa.

    Um corte devolvido significa que dois segmentos explicam o sinal melhor que um,
    além da penalidade. Série errática também tem corte que reduz RSS, então isso não
    basta para afirmar mudança de regime.

    Devolve lista vazia quando a trajetória não comporta `2 * min_segment` pontos:
    lote com poucos QCs é entrada legítima.
    """
    valores = np.asarray(trajectory, dtype=np.float64)
    ordens = np.asarray(injection_orders)
    if valores.size != ordens.size:
        raise ValueError(
            f"trajetória e ordens de injeção têm de ter o mesmo comprimento; recebi "
            f"{valores.size} e {ordens.size}. Desalinhamento aqui devolveria a ordem de "
            "injeção de outro QC, e o número resultante pareceria válido."
        )

    limiar = _limiar(valores, penalty)
    achados: list[int] = []
    _recursa(valores, 0, valores.size, min_segment, limiar, achados)
    return [int(ordens[indice]) for indice in sorted(achados)]


# Multiplicador da penalidade derivada, em unidades de variância de ruído. A faixa
# que funciona nos sinais construídos dos testes e no `neg_hilic` real vai de 2,5 a
# 200, e 20 fica perto do centro geométrico. Abaixo de 2,5 o detector acha um segundo
# corte no `neg_hilic`; acima de 200 perde o degrau real.
PENALTY_NOISE_MULTIPLIER = 20.0


def _sigma_ruido(valores: np.ndarray) -> float:
    """Desvio do ruído, estimado pela mediana das diferenças sucessivas.

    A mediana é robusta ao próprio degrau: ele contribui uma diferença grande entre
    várias pequenas. As alternativas falham em sentidos opostos. A variância do sinal
    inteiro é inflada pelo degrau e suprime um segundo corte. A variância por segmento
    encolhe junto com o trecho que deveria julgar e aceita corte dentro de trecho plano.

    Devolve zero em sinal sem ruído; a comparação estrita do chamador rejeita redução
    de RSS nula.
    """
    if valores.size < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(valores))) / np.sqrt(2.0))


def _limiar(valores: np.ndarray, penalty: float | None) -> float:
    """Penalidade em unidades de RSS: `PENALTY_NOISE_MULTIPLIER` vezes a variância do ruído.

    É conservadora de propósito: a saída é apresentada como sugestão a confirmar, e o
    erro preferível é não detectar. Ela não tenta separar degrau de série errática;
    isso fica com a regra 2 da classificação, pelo nível sustentado.
    """
    if penalty is not None:
        return penalty
    return float(PENALTY_NOISE_MULTIPLIER * _sigma_ruido(valores) ** 2)


def _recursa(
    valores: np.ndarray,
    inicio: int,
    fim: int,
    min_segment: int,
    limiar: float,
    achados: list[int],
) -> None:
    """Corta o trecho `[inicio, fim)` se valer a pena, e repete nos dois lados.

    O limiar é o mesmo em toda a recursão e vem do ruído da trajetória inteira.
    Recalculado por trecho, ele encolheria junto com o que deveria julgar e passaria a
    cortar ruído.
    """
    trecho = valores[inicio:fim]
    corte_local = rss_minimizing_split(trecho, min_segment=min_segment)
    if corte_local is None:
        return

    rss_um = float(((trecho - trecho.mean()) ** 2).sum())
    esquerda, direita = trecho[:corte_local], trecho[corte_local:]
    rss_dois = float(
        ((esquerda - esquerda.mean()) ** 2).sum() + ((direita - direita.mean()) ** 2).sum()
    )
    if rss_um - rss_dois <= limiar:
        return

    corte = inicio + corte_local
    achados.append(corte)
    _recursa(valores, inicio, corte, min_segment, limiar, achados)
    _recursa(valores, corte, fim, min_segment, limiar, achados)

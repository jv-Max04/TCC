"""Estatísticas da trajetória de QC que as regras 2 e 3 da classificação consomem.

A trajetória é o score de PC1 dos QCs em ordem de injeção. A regra 2 pergunta se o
nível desloca e permanece deslocado (`sustained_step_ratio`), e não quão grande é o
maior salto: série errática também tem salto dominante. A regra 3 mede deslocamento
líquido sobre bruto (`monotonicity_net_over_gross`), em valor e não em posto, porque
correlação de posto sobre poucos pontos quase empatados muda com o último bit.

Os limiares vêm das faixas que as duas estatísticas ocupam nos quatro arquivos de
referência, com cada score perturbado em ±1 unidade. Para a regra 2, 2,0 fica entre
1,615 (maior valor dos modos sem degrau) e 2,570 (menor valor do `neg_hilic`). Para
a regra 3, 0,487 fica entre 0,338 e 0,636. São observações sobre esses arquivos.
"""

import numpy as np

__all__ = [
    "MIN_SEGMENT",
    "jump_ratio",
    "monotonicity_net_over_gross",
    "rss_minimizing_split",
    "sustained_step_ratio",
    "total_movement",
]

# Dois pontos de cada lado do corte. Com sete QCs isso permite cortes em 2, 3, 4 e 5.
MIN_SEGMENT = 2


def rss_minimizing_split(trajectory: np.ndarray, *, min_segment: int = MIN_SEGMENT) -> int | None:
    """Índice do corte que minimiza a soma dos quadrados de dois segmentos constantes.

    Devolve `None` quando a trajetória não comporta `min_segment` pontos de cada lado.

    É o mesmo corte que a segmentação binária de `detect_changepoints` acha primeiro.
    Assim a regra 2 mede o nível sustentado na partição que foi detectada, e não em
    outra.
    """
    valores = np.asarray(trajectory, dtype=np.float64)
    n = valores.size
    if n < 2 * min_segment:
        return None

    melhor_indice: int | None = None
    melhor_rss = np.inf
    for corte in range(min_segment, n - min_segment + 1):
        esquerda, direita = valores[:corte], valores[corte:]
        rss = float(
            ((esquerda - esquerda.mean()) ** 2).sum() + ((direita - direita.mean()) ** 2).sum()
        )
        if rss < melhor_rss:
            melhor_rss, melhor_indice = rss, corte
    return melhor_indice


def sustained_step_ratio(trajectory: np.ndarray, *, split: int) -> float:
    """Quanto o nível desloca em relação ao quanto oscila dentro dos segmentos.

    `|média(depois) - média(antes)|` sobre a maior amplitude interna aos dois
    segmentos. É a estatística da regra 2.

    Devolve `inf` para degrau perfeito: dois patamares constantes têm amplitude
    interna zero. `inf` compara corretamente contra qualquer limiar; `nan` faria a
    regra falhar por semântica de ponto flutuante.

    Levanta quando `split` não deixa ao menos um ponto de cada lado.
    """
    valores = np.asarray(trajectory, dtype=np.float64)
    if not 1 <= split <= valores.size - 1:
        raise ValueError(
            f"`split` tem de deixar ao menos um ponto de cada lado; recebi {split} "
            f"para trajetória de {valores.size} pontos. Segmento vazio não tem média, e "
            "devolver `nan` aqui faria a regra 2 decidir por comparação com `nan`."
        )
    antes, depois = valores[:split], valores[split:]
    salto = abs(float(depois.mean()) - float(antes.mean()))
    amplitude = max(
        float(antes.max() - antes.min()),
        float(depois.max() - depois.min()),
    )
    if amplitude == 0.0:
        return np.inf if salto > 0.0 else 0.0
    return salto / amplitude


def monotonicity_net_over_gross(trajectory: np.ndarray) -> float:
    """Deslocamento líquido sobre deslocamento bruto: 1,0 em série monótona.

    `|v[último] - v[primeiro]| / Σ|Δv|`. É a estatística da regra 3. Usa valor e não
    posto, o que a livra da instabilidade de empates que a correlação de posto tem.

    Devolve `nan` quando não há movimento, e não zero: zero diria 'não monótono'
    (`erratic`), e ausência de movimento é outra classe (`unassessable`). Quem chama
    compara `total_movement` contra o piso da política antes de usar o número.
    """
    valores = np.asarray(trajectory, dtype=np.float64)
    if valores.size < 2:
        return np.nan
    bruto = float(np.abs(np.diff(valores)).sum())
    if bruto == 0.0:
        return np.nan
    return abs(float(valores[-1] - valores[0])) / bruto


def total_movement(trajectory: np.ndarray) -> float:
    """`Σ|Δv|`: o denominador da monotonicidade, exposto para o piso da política."""
    valores = np.asarray(trajectory, dtype=np.float64)
    if valores.size < 2:
        return 0.0
    return float(np.abs(np.diff(valores)).sum())


def jump_ratio(trajectory: np.ndarray) -> float:
    """Maior salto sobre a mediana dos demais saltos.

    Não decide nada na classificação: fica no vetor de características como insumo
    da política aprendida. A mediana exclui o maior salto para comparar o atípico
    contra o típico, sem que ele contamine a própria referência.

    Devolve `nan` com menos de três pontos: com um salto só não há 'os demais'.
    """
    valores = np.asarray(trajectory, dtype=np.float64)
    if valores.size < 3:
        return np.nan
    saltos = np.abs(np.diff(valores))
    maior = int(np.argmax(saltos))
    demais = np.delete(saltos, maior)
    mediana = float(np.median(demais))
    if mediana == 0.0:
        return np.inf if saltos[maior] > 0.0 else np.nan
    return float(saltos[maior]) / mediana

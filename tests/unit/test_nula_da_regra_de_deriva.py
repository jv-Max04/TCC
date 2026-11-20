"""A regra da deriva gradual sob passeio aleatório, nas sete posições de QC: cerca de 37% passam.

**É a procedência desse número.** Aqui ele sai da função de
produção (`monotonicity_net_over_gross`) e do limiar do aplicativo (`LIMIARES_INICIAIS`), e o teste
reprova se qualquer dos dois mudar — que é quando o número deixa de valer.

O que o número diz: com sete pontos, a razão entre o deslocamento líquido e o total passa de 0,487
em mais de um terço dos passeios aleatórios puros, e o percentil 95 dela fica perto de 0,94 — então
subir o limiar não separa deriva de passeio. É a estatística sobre sete pontos, e não defeito do
gerador nem do limiar.

**A planta é o ruído branco**, que não tem deriva nenhuma: nele a razão passa do limiar em poucos
por cento. Se a função devolvesse uma constante, ou o limiar fosse trivial, as duas populações
dariam o mesmo e a segunda asserção reprovaria.
"""

from __future__ import annotations

import numpy as np

from refino.diagnosis.policy import LIMIARES_INICIAIS
from refino.diagnosis.trajectory import monotonicity_net_over_gross

# As injeções 1, 12, 23, 34, 45, 56 e 68 das corridas reais, em índice de zero.
POSICOES_DE_QC = np.array([0, 11, 22, 33, 44, 55, 67])
REPLICAS = 20_000


def _razoes(trajetorias: np.ndarray) -> np.ndarray:
    return np.array([monotonicity_net_over_gross(t) for t in trajetorias])


def test_um_terco_dos_passeios_aleatorios_passa_o_limiar_da_deriva_gradual() -> None:
    gerador = np.random.default_rng(20261018)
    limiar = LIMIARES_INICIAIS["qc_monotonicity_min"]

    passeios = np.cumsum(gerador.standard_normal((REPLICAS, 68)), axis=1)[:, POSICOES_DE_QC]
    razoes = _razoes(passeios)
    assert np.isfinite(razoes).all()
    assert 0.35 < float(np.mean(razoes >= limiar)) < 0.39
    assert 0.92 < float(np.quantile(razoes, 0.95)) < 0.96

    ruido = gerador.standard_normal((REPLICAS, POSICOES_DE_QC.size))
    assert float(np.mean(_razoes(ruido) >= limiar)) < 0.10

"""As estatísticas de trajetória, contra as faixas MEDIDAS como oráculo.

**As faixas são o oráculo desta unidade e existiam antes da implementação.** Elas vêm
de perturbação exaustiva de ±1 unidade de score sobre a tabela 4.1 - 3^7 = 2.187
cantos -, que é a tolerância declarada para aquela tabela. Fixá-las aqui
é o que faz a escolha das duas estatísticas ser resultado em vez de preferência.

**São observação sobre os quatro arquivos do laboratório, não invariante**. Um
quinto modo recebe as mesmas funções e tem sua própria faixa medida.
"""

import itertools

import numpy as np
import pytest

from refino.diagnosis.trajectory import (
    jump_ratio,
    monotonicity_net_over_gross,
    rss_minimizing_split,
    sustained_step_ratio,
    total_movement,
)

# Tabela 4.1: scores de PC1 dos QCs relativos ao primeiro, arredondados para inteiro.
TRAJETORIAS = {
    "pos_hilic": (0, -10, 74, 96, -27, -37, -39),
    "pos_c18": (0, 45, -2, 44, 89, 46, 86),
    "neg_hilic": (0, -5, -4, -1, 7, 49, 70),
    "neg_c18": (0, 10, 10, 17, 15, 18, 23),
}

# Faixas medidas sob ±1, com os limiares das regras 2 e 3 entre os grupos.
FAIXA_SUSTENTACAO = {
    "pos_hilic": (0.674, 0.729),
    "pos_c18": (1.037, 1.187),
    "neg_c18": (0.854, 1.615),
    "neg_hilic": (2.570, 3.216),
}
FAIXA_MONOTONICIDADE = {
    "pos_hilic": (0.146, 0.165),
    "pos_c18": (0.309, 0.338),
    "neg_c18": (0.636, 1.000),
    "neg_hilic": (0.829, 0.923),
}
LIMIAR_SUSTENTACAO = 2.0
LIMIAR_MONOTONICIDADE = 0.487

TOLERANCIA = 1e-3  # as faixas foram reportadas com três casas


def _sustentacao(valores: tuple[int, ...]) -> float:
    corte = rss_minimizing_split(np.asarray(valores, dtype=np.float64))
    assert corte is not None
    return sustained_step_ratio(np.asarray(valores, dtype=np.float64), split=corte)


# ---------------------------------------------------------------------------
# Séries construídas: o comportamento fechado, antes dos dados reais
# ---------------------------------------------------------------------------


def test_degrau_puro_da_sustentacao_infinita() -> None:
    """Dois patamares constantes: amplitude interna zero, e o limite é `inf`.

    **`inf` e não `nan`, e a distinção é o ponto.** `inf` compara corretamente contra
    qualquer limiar; `nan` faria a regra 2 escorregar por semântica IEEE, que é o
    acidente aritmético que a regra 0 existe para impedir.
    """
    degrau = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    assert _sustentacao(tuple(int(v) for v in degrau)) == np.inf
    assert not np.isnan(_sustentacao(tuple(int(v) for v in degrau)))


def test_serie_constante_da_sustentacao_zero_e_nao_infinito() -> None:
    """Sem salto E sem oscilação: zero, porque nada deslocou."""
    assert _sustentacao((5, 5, 5, 5, 5, 5)) == 0.0


def test_rampa_pura_e_perfeitamente_monotonica() -> None:
    """Líquido igual a bruto: exactamente 1,0."""
    rampa = np.arange(7.0)
    assert monotonicity_net_over_gross(rampa) == pytest.approx(1.0)


def test_serie_que_volta_ao_inicio_tem_monotonicidade_zero() -> None:
    """Líquido zero, bruto positivo: nada de deriva, muito movimento."""
    ida_e_volta = np.array([0.0, 10.0, 20.0, 10.0, 0.0])
    assert monotonicity_net_over_gross(ida_e_volta) == pytest.approx(0.0)


def test_serie_constante_da_monotonicidade_e_nan_e_nao_zero() -> None:
    """**A distinção entre os dois casos, e é a razão de a função existir com este contrato.**

    Zero significaria "não monótono", que leva a `erratic`. Ausência de movimento
    leva a `unassessable`. Valor de retorno que cobre as duas torna as duas
    indistinguíveis em contagem, e as duas levam a ações opostas: `erratic` sugere
    método robusto, não avaliável sugere olhar o dado antes de corrigir.
    """
    constante = np.full(7, 42.0)
    resultado = monotonicity_net_over_gross(constante)
    assert np.isnan(resultado), "série constante tem de dar `nan`, nunca 0,0"
    assert total_movement(constante) == 0.0


def test_trajetoria_curta_nao_comporta_corte() -> None:
    """Menos de `2 * min_segment` pontos: `None`, e o chamador decide o que fazer."""
    assert rss_minimizing_split(np.array([1.0, 2.0, 3.0])) is None
    assert rss_minimizing_split(np.array([1.0, 2.0, 3.0, 4.0])) is not None


def test_corte_fora_do_intervalo_levanta() -> None:
    """Segmento vazio não tem média, e devolver `nan` deixaria a regra 2 decidir com `nan`."""
    valores = np.arange(7.0)
    for invalido in (0, 7, -1, 99):
        with pytest.raises(ValueError, match="ao menos um ponto de cada lado"):
            sustained_step_ratio(valores, split=invalido)


def test_o_corte_minimiza_rss_e_acha_o_degrau_construido() -> None:
    """Degrau em posição conhecida: o corte cai onde o nível muda."""
    degrau = np.array([1.0, 1.2, 0.9, 1.1, 8.0, 8.3, 7.9])
    assert rss_minimizing_split(degrau) == 4


# ---------------------------------------------------------------------------
# Os quatro arquivos reais, contra as faixas medidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modo", tuple(TRAJETORIAS))
def test_sustentacao_cai_na_faixa_medida_nos_quatro_arquivos(modo: str) -> None:
    """OBSERVAÇÃO sobre os quatro arquivos do laboratório, não invariante."""
    baixo, alto = FAIXA_SUSTENTACAO[modo]
    valor = _sustentacao(TRAJETORIAS[modo])
    assert baixo - TOLERANCIA <= valor <= alto + TOLERANCIA, (
        f"{modo}: sustentação {valor:.3f} fora da faixa medida [{baixo}, {alto}]. "
        "A faixa vem de perturbação exaustiva de ±1 sobre a tabela 4.1; sair dela é "
        "achado sobre a estatística, não defeito de código - medir antes de mover."
    )


@pytest.mark.parametrize("modo", tuple(TRAJETORIAS))
def test_monotonicidade_cai_na_faixa_medida_nos_quatro_arquivos(modo: str) -> None:
    """OBSERVAÇÃO sobre os quatro arquivos do laboratório, não invariante."""
    baixo, alto = FAIXA_MONOTONICIDADE[modo]
    valor = monotonicity_net_over_gross(np.asarray(TRAJETORIAS[modo], dtype=np.float64))
    assert baixo - TOLERANCIA <= valor <= alto + TOLERANCIA, (
        f"{modo}: monotonicidade {valor:.3f} fora da faixa medida [{baixo}, {alto}]"
    )


def test_a_sustentacao_separa_o_degrau_dos_outros_tres_com_a_folga_medida() -> None:
    """**O que faz a regra 2 ser regra: o `neg_hilic` de um lado, os três do outro.**

    A folga é 45,6% da largura do intervalo viável sobre o limiar, e o teste afirma a
    separação e não só os valores - porque separação é a propriedade que a regra usa.
    """
    degrau = _sustentacao(TRAJETORIAS["neg_hilic"])
    outros = {modo: _sustentacao(TRAJETORIAS[modo]) for modo in TRAJETORIAS if modo != "neg_hilic"}

    assert degrau > LIMIAR_SUSTENTACAO, f"o degrau real tem de casar a regra 2: {degrau:.3f}"
    for modo, valor in outros.items():
        assert valor < LIMIAR_SUSTENTACAO, (
            f"{modo} casaria a regra 2 com {valor:.3f} e sairia `step_change` sem ter degrau"
        )


def test_a_monotonicidade_separa_a_deriva_dos_erraticos_com_a_folga_medida() -> None:
    """O `neg_c18` acima do limiar, os dois erráticos abaixo.

    O `neg_hilic` é indiferente aqui: a regra 2 o intercepta antes, e degrau
    **também** é monótono - o que é propriedade da cascata e não frouxidão.
    """
    deriva = monotonicity_net_over_gross(np.asarray(TRAJETORIAS["neg_c18"], dtype=np.float64))
    assert deriva >= LIMIAR_MONOTONICIDADE, f"a deriva real tem de casar a regra 3: {deriva:.3f}"

    for modo in ("pos_hilic", "pos_c18"):
        valor = monotonicity_net_over_gross(np.asarray(TRAJETORIAS[modo], dtype=np.float64))
        assert valor < LIMIAR_MONOTONICIDADE, (
            f"{modo} casaria a regra 3 com {valor:.3f} e sairia `gradual_drift` sendo errático"
        )


@pytest.mark.parametrize("modo", tuple(TRAJETORIAS))
def test_as_duas_estatisticas_sobrevivem_a_perturbacao_de_uma_unidade(modo: str) -> None:
    """**O teste que prova que a separação não é do arredondamento.**

    Varre os 3^7 cantos de ±1 - a tolerância declarada - e afirma que o veredito de
    cada regra é o mesmo em todos. Sem ele, "as faixas separam" seria afirmação sobre
    o valor base, e é exactamente por não ter sido feito que o limiar de 0,70 da regra
    3 entrou na especificação sendo insatisfazível.
    """
    base = TRAJETORIAS[modo]
    espera_degrau = modo == "neg_hilic"
    espera_deriva = modo in ("neg_c18", "neg_hilic")

    for delta in itertools.product((-1, 0, 1), repeat=7):
        valores = np.asarray([b + d for b, d in zip(base, delta, strict=True)], dtype=np.float64)
        corte = rss_minimizing_split(valores)
        assert corte is not None
        assert (sustained_step_ratio(valores, split=corte) > LIMIAR_SUSTENTACAO) is espera_degrau
        assert (monotonicity_net_over_gross(valores) >= LIMIAR_MONOTONICIDADE) is espera_deriva, (
            f"{modo} sob {delta}: veredito da regra 3 mudou"
        )


# ---------------------------------------------------------------------------
# A razão de saltos, que NÃO decide nada
# ---------------------------------------------------------------------------


def test_a_razao_de_saltos_usa_a_mediana_dos_demais() -> None:
    """**A definição é única, e o `neg_c18` é o caso que a distingue.**

    Pelos **demais** dá 3,33; por **todos** dá 2,50. O antigo limiar da regra 2 era 3,
    entre as duas - então a discrepância decidia a classe. A regra 2 deixou de
    consumir esta quantidade, e é por isso que a definição pôde ser escolhida por
    mérito: ela compara o atípico contra o típico em vez de contra uma mediana que ele
    mesmo contamina.
    """
    valores = np.asarray(TRAJETORIAS["neg_c18"], dtype=np.float64)
    assert jump_ratio(valores) == pytest.approx(10.0 / 3.0, abs=1e-9)

    saltos = np.abs(np.diff(valores))
    por_todos = float(saltos.max()) / float(np.median(saltos))
    assert por_todos == pytest.approx(2.5, abs=1e-9)
    assert jump_ratio(valores) != pytest.approx(por_todos)


def test_a_razao_de_saltos_nao_e_definida_com_menos_de_tres_pontos() -> None:
    """Dois pontos dão um salto só, e "os demais" é conjunto vazio."""
    assert np.isnan(jump_ratio(np.array([1.0, 5.0])))
    assert not np.isnan(jump_ratio(np.array([1.0, 5.0, 6.0])))

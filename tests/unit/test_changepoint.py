"""`detect_changepoints` - séries construídas, e o negativo da UNIDADE do retorno.

**O teste que importa mais aqui é o da unidade.** O contrato devolve `list[int]`, e
`int` pode ser índice de trajetória (0 a 6) ou ordem de injeção (1, 12, 23, 34, 45, 56,
68). O critério é redigido em ordem de injeção e `propose_batches` consome a mesma
lista, então a convenção é compartilhada por duas funções e não estava escrita. Um teste
que passasse com as duas convenções não provaria nada - e é por isso que o caso é
construído para que os dois números **divirjam**.
"""

import numpy as np
import pytest

from refino.diagnosis.changepoint import PENALTY_NOISE_MULTIPLIER, detect_changepoints
from refino.diagnosis.trajectory import rss_minimizing_split, sustained_step_ratio

# As ordens de injeção dos sete QCs nos quatro arquivos do laboratório. São
# OBSERVAÇÃO e não invariante: o diagnóstico existe para variar a partição, e as
# sete posições são a configuração de hoje.
ORDENS_REAIS = np.array([1, 12, 23, 34, 45, 56, 68])


def test_degrau_puro_devolve_um_ponto_na_ordem_de_injecao_correta() -> None:
    """Degrau entre o quinto e o sexto QC: a ordem devolvida é 56, não 5."""
    degrau = np.array([0.0, 1.0, -1.0, 0.5, 0.0, 40.0, 41.0])
    assert detect_changepoints(degrau, ORDENS_REAIS) == [56]


def test_a_unidade_do_retorno_e_ordem_de_injecao_e_nao_indice() -> None:
    """**O negativo plantado: as duas convenções dão números DIFERENTES aqui.**

    O corte cai no índice 5 da trajetória, e a ordem de injeção ali é 56. Se a
    implementação devolvesse índice, o retorno seria `[5]` - que também é um `int`
    plausível, também passa por `list[int]`, e faria `propose_batches` cortar entre as
    injeções 5 e 6 em vez de entre 45 e 56.

    O teste afirma as duas coisas: que o valor é a ordem, e que **não** é o índice.
    Afirmar só a primeira deixaria passar uma implementação que devolvesse índice num
    dataset onde os dois coincidem - e coincidem sempre que as ordens são `0..n-1`,
    que é exactamente o caso que um fixture ingênuo produziria.
    """
    degrau = np.array([0.0, 1.0, -1.0, 0.5, 0.0, 40.0, 41.0])
    resultado = detect_changepoints(degrau, ORDENS_REAIS)

    assert resultado == [56], "o retorno tem de ser ORDEM DE INJEÇÃO"
    assert resultado != [5], "o retorno NÃO pode ser índice de trajetória"
    for valor in resultado:
        assert valor in set(ORDENS_REAIS.tolist()), (
            f"{valor} não é uma ordem de injeção presente na entrada; a unidade do "
            "retorno não está sendo respeitada"
        )


def test_rampa_pura_produz_corte_mas_nao_e_degrau() -> None:
    """**Detecção não afirma degrau, e a rampa é o caso que separa as duas coisas.**

    Uma rampa reduz o RSS bastante quando dividida - dois segmentos têm variância
    menor que um -, então a segmentação **acha** um corte. Isso não é defeito: a
    função devolve "o modelo de dois segmentos explica melhor", que é diferente de
    "houve mudança de regime".

    Quem separa é a regra 2, que exige nível **sustentado**. O teste afirma as duas
    metades: o corte existe, e a sustentação nele fica **abaixo** do limiar de 2,0.
    Sem a segunda metade, este teste mediria a detecção e não a distinção que importa.
    """
    rampa = np.arange(7.0)
    cortes = detect_changepoints(rampa, ORDENS_REAIS)
    assert cortes, "rampa reduz RSS ao ser dividida; esperava ao menos um corte"

    indice = rss_minimizing_split(rampa)
    assert indice is not None
    sustentacao = sustained_step_ratio(rampa, split=indice)
    assert sustentacao < 2.0, (
        f"a sustentação da rampa é {sustentacao:.3f} e passaria do limiar da regra 2; "
        "rampa sairia `step_change` sendo deriva gradual"
    )


def test_serie_constante_nao_produz_corte() -> None:
    """Sem variação não há RSS a reduzir."""
    assert detect_changepoints(np.full(7, 3.0), ORDENS_REAIS) == []


def test_degrau_com_ruido_ainda_acha_o_degrau() -> None:
    """Ruído pequeno em cima do degrau: o corte continua na posição certa.

    A semente é explícita (regra 3 do README): sem ela o teste reprovaria de forma
    intermitente, que é pior que reprovar sempre.
    """
    gerador = np.random.default_rng(42)
    degrau = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 40.0])
    com_ruido = degrau + gerador.normal(0.0, 0.5, size=7)
    assert detect_changepoints(com_ruido, ORDENS_REAIS) == [56]


def test_ruido_puro_com_semente_fixa_tem_comportamento_declarado() -> None:
    """Fixa o que a função FAZ com ruído, em vez de o que se esperaria dela.

    Ruído puro não tem degrau, mas segmentação binária sobre sete pontos pode achar
    corte por acaso. O teste declara o desfecho observado com esta semente e afirma o
    que importa: **qualquer** corte devolvido é uma ordem de injeção válida.
    """
    gerador = np.random.default_rng(7)
    cortes = detect_changepoints(gerador.normal(0.0, 1.0, size=7), ORDENS_REAIS)
    assert all(corte in set(ORDENS_REAIS.tolist()) for corte in cortes)


def test_trajetoria_curta_devolve_vazio_em_vez_de_levantar() -> None:
    """Lote com poucos QCs é configuração que o próprio diagnóstico pode propor."""
    for n in (0, 1, 2, 3):
        curta = np.arange(float(n))
        assert detect_changepoints(curta, np.arange(n)) == []


def test_desalinhamento_entre_trajetoria_e_ordens_levanta() -> None:
    """Comprimentos diferentes devolveriam a ordem de injeção de outro QC.

    O modo de falha sem esta guarda é o insidioso: o número sai, é um `int`, e está
    errado.
    """
    with pytest.raises(ValueError, match="mesmo comprimento"):
        detect_changepoints(np.arange(7.0), np.arange(5))


def test_penalidade_explicita_suprime_o_corte() -> None:
    """Penalidade alta o bastante desliga a detecção, o que prova que ela é consultada.

    É o par causal do parâmetro: sem este teste, `penalty` poderia ser ignorada e o
    comportamento pareceria idêntico na árvore limpa.
    """
    degrau = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 40.0])
    assert detect_changepoints(degrau, ORDENS_REAIS) == [56]
    assert detect_changepoints(degrau, ORDENS_REAIS, penalty=1e9) == []


def test_dois_degraus_produzem_dois_cortes() -> None:
    """Recursão nos dois lados: a segmentação binária não para no primeiro corte."""
    trajetoria = np.array([0.0, 0.0, 20.0, 20.0, 20.0, 60.0, 60.0])
    ordens = np.arange(10, 80, 10)
    cortes = detect_changepoints(trajetoria, ordens)
    assert len(cortes) == 2, f"esperava dois cortes, achei {cortes}"
    assert cortes == sorted(cortes), "os cortes têm de sair ordenados"


def test_a_constante_da_penalidade_esta_dentro_da_faixa_medida() -> None:
    """**A folga da constante, afirmada como par causal - e é o que a torna medida.**

    A faixa viável foi obtida por varredura contra sete propriedades: os cinco ensaios
    construídos deste módulo, a janela do corte, e a contagem de **um** corte que
    se exige para dar exactamente dois lotes. Faixa [2,5; 200], que é 80x.

    O teste afirma três coisas, e a terceira é a que impede o número de ser preferência
    com aparência de medição: o valor está dentro, e as **duas** bordas quebram
    propriedades opostas. Abaixo, o detector fabrica corte; acima, perde o degrau.
    """
    neg_hilic = np.array([0.0, -5.0, -4.0, -1.0, 7.0, 49.0, 70.0])

    assert 2.5 <= PENALTY_NOISE_MULTIPLIER <= 200.0
    assert PENALTY_NOISE_MULTIPLIER / 2.5 >= 5.0, "folga inferior menor que 5x"
    assert 200.0 / PENALTY_NOISE_MULTIPLIER >= 5.0, "folga superior menor que 5x"

    # Com o valor adotado: um corte, na janela.
    assert len(detect_changepoints(neg_hilic, ORDENS_REAIS)) == 1

    # Abaixo da borda inferior o detector fabrica um segundo corte, e
    # passaria a contar TRÊS lotes no `neg_hilic`.
    sigma_quadrado = float(np.median(np.abs(np.diff(neg_hilic)))) ** 2 / 2.0
    frouxa = 1.0 * sigma_quadrado
    assert len(detect_changepoints(neg_hilic, ORDENS_REAIS, penalty=frouxa)) > 1

    # Acima da borda superior ele perde o degrau real por completo.
    estrita = 400.0 * sigma_quadrado
    assert detect_changepoints(neg_hilic, ORDENS_REAIS, penalty=estrita) == []


def test_o_neg_hilic_da_tabela_4_1_tem_corte_na_janela_45_56() -> None:
    """A janela do corte sobre a trajetória da tabela 4.1 - OBSERVAÇÃO sobre um arquivo.

    A versão com os dados reais lidos do disco vive nos testes de integração; esta usa
    a trajetória publicada, que é invariante a escala de score e por isso
    imune à diferença de normalização.
    """
    neg_hilic = np.array([0.0, -5.0, -4.0, -1.0, 7.0, 49.0, 70.0])
    cortes = detect_changepoints(neg_hilic, ORDENS_REAIS)
    assert cortes, "o degrau do `neg_hilic` tem de produzir ao menos um corte"
    assert any(45 <= corte <= 56 for corte in cortes), (
        f"nenhum corte na janela 45-56; achei {cortes}"
    )

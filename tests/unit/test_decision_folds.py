"""A partição por modo — e o par causal, porque disjunção não é anti-vazamento.

**O item original pedia "nenhuma matriz de teste compartilha semente com matriz de treino", e essa
condição é satisfeita por uma partição que vaza tudo.** O gerador reamostra resíduos dos
quatro arquivos reais, então duas sementes distintas são duas reamostragens **da mesma população**.
Um teste que só verificasse disjunção de índices ficaria verde sobre o desenho errado.

Então o par é este: a partição por modo **não acusa**, e a partição por semente **acusa** — sobre o
mesmo corpus e pelo mesmo predicado. Se as duas dessem o mesmo desfecho, o predicado não estaria
lendo o modo de origem.
"""

import pytest

from refino.decision.folds import Fold, leakage_pair, mode_folds

pytestmark = pytest.mark.property

# Doze matrizes, tres por modo, na ordem em que o piloto as gera.
MODOS = ["pos_hilic"] * 3 + ["pos_c18"] * 3 + ["neg_hilic"] * 3 + ["neg_c18"] * 3


# ---------------------------------------------------------------------------
# O par causal
# ---------------------------------------------------------------------------


def test_a_particao_por_modo_nao_acusa_e_a_por_semente_acusa() -> None:
    """**O par causal, e é ele que distingue anti-vazamento de disjunção.**

    As duas metades importam. A honesta **sem defeito** é a premissa — sem ela o teste passaria
    por o predicado acusar tudo, que é o portão que sempre reprova. A vazada **com
    defeito** é a conclusão, e ela é o desenho que o item pedia.
    """
    par = leakage_pair(MODOS)

    honestos = [d for dobra in par.honest for d in dobra.defects_against(MODOS)]
    assert honestos == [], f"a partição por modo não podia ter defeito: {honestos}"

    vazados = [d for dobra in par.leaky for d in dobra.defects_against(MODOS)]
    assert vazados, (
        "a partição por SEMENTE não acusou defeito nenhum. Ou o predicado não lê o modo de origem, "
        "ou a partição vazada não vaza — e nos dois casos o [TO-9.4] estaria medindo disjunção"
    )
    assert any("derivam do modo" in d for d in vazados)


def test_a_particao_por_semente_satisfaz_a_condicao_antiga_e_vaza_de_todo_modo() -> None:
    """**A demonstração de que o critério original era vácuo, e ela é aritmética.**

    Treino e teste da partição vazada são disjuntos — logo nenhuma matriz de teste compartilha
    índice, e portanto semente, com matriz de treino. A condição antiga fica satisfeita em **todas**
    as dobras, e o vazamento de covariância está presente em **todas elas**.
    """
    par = leakage_pair(MODOS)
    for dobra in par.leaky:
        assert not (set(dobra.train) & set(dobra.test)), "a condição antiga exigia só isto"
        do_modo_validado = [i for i in dobra.train if MODOS[i] == dobra.held_out_mode]
        assert do_modo_validado, (
            f"dobra {dobra.held_out_mode}: a partição por semente devia ter matrizes do modo "
            "validado no treino — é o vazamento que ela existe para demonstrar"
        )


def test_as_duas_particoes_tem_o_mesmo_tamanho_de_teste_por_dobra() -> None:
    """Sem isso, a diferença mediria tamanho de amostra e não vazamento.

    Se a partição vazada tivesse teste maior, o desempenho dela seria melhor por ter mais dados,
    e a diferença entre as duas confundiria os dois efeitos.
    """
    par = leakage_pair(MODOS)
    assert len(par.honest) == len(par.leaky)
    for h, v in zip(par.honest, par.leaky, strict=True):
        assert h.held_out_mode == v.held_out_mode
        assert len(h.test) == len(v.test), (
            f"dobra {h.held_out_mode}: teste honesto tem {len(h.test)} e vazado tem {len(v.test)}"
        )


# ---------------------------------------------------------------------------
# As dobras
# ---------------------------------------------------------------------------


def test_uma_dobra_por_modo_com_teste_do_modo_e_treino_dos_outros_tres() -> None:
    dobras = mode_folds(MODOS)
    assert [d.held_out_mode for d in dobras] == ["pos_hilic", "pos_c18", "neg_hilic", "neg_c18"]
    for dobra in dobras:
        assert {MODOS[i] for i in dobra.test} == {dobra.held_out_mode}
        assert dobra.held_out_mode not in {MODOS[i] for i in dobra.train}
        assert len(dobra.train) == len(MODOS) - len(dobra.test)


def test_a_ordem_das_dobras_e_canonica_e_nao_a_de_chegada() -> None:
    """Duas execuções com o corpus embaralhado dão a mesma sequência de dobras.

    Ordenar pela ordem de chegada faria a dobra `k` de duas execuções ser um modo diferente, e
    qualquer comparação entre execuções compararia modos distintos sob o mesmo rótulo.
    """
    embaralhado = ["neg_c18", "pos_c18", "neg_hilic", "pos_hilic"] * 2
    assert [d.held_out_mode for d in mode_folds(embaralhado)] == [
        "pos_hilic",
        "pos_c18",
        "neg_hilic",
        "neg_c18",
    ]


def test_modo_ausente_nao_gera_dobra_e_um_modo_so_nao_gera_nenhuma() -> None:
    """`Não há dobra` e `a dobra não tem teste` são coisas diferentes."""
    tres = ["pos_hilic", "pos_hilic", "pos_c18", "neg_hilic"]
    assert [d.held_out_mode for d in mode_folds(tres)] == ["pos_hilic", "pos_c18", "neg_hilic"]
    # Com um modo so, o treino de qualquer dobra ficaria vazio: nao ha dobra a construir.
    assert mode_folds(["pos_hilic", "pos_hilic"]) == ()
    assert mode_folds([]) == ()


def test_modo_fora_do_vocabulario_levanta_na_fronteira() -> None:
    """`ImitatedMode` é fronteira de confiança: modo inválido não cai num `else` razoável."""
    with pytest.raises(ValueError, match="fora do vocabulário"):
        mode_folds(["pos_hilic", "pos_c18", "modo_inventado"])


def test_grupo_de_semente_com_tamanho_errado_levanta() -> None:
    with pytest.raises(ValueError, match="grupos para"):
        leakage_pair(MODOS, seed_groups=[0, 1])


# ---------------------------------------------------------------------------
# O predicado, com cada violação plantada
# ---------------------------------------------------------------------------


def test_o_predicado_acusa_cada_violacao_uma_por_vez() -> None:
    """Cada cláusula plantada em isolamento, para que uma delas não cubra a ausência de outra."""
    modos = ["pos_hilic", "pos_hilic", "pos_c18", "neg_hilic"]

    limpa = Fold(held_out_mode="pos_hilic", train=(2, 3), test=(0, 1))
    assert limpa.defects_against(modos) == [], "premissa: a dobra limpa não acusa"

    vaza = Fold(held_out_mode="pos_hilic", train=(1, 2, 3), test=(0,))
    assert any("derivam do modo" in d for d in vaza.defects_against(modos))

    teste_errado = Fold(held_out_mode="pos_hilic", train=(3,), test=(0, 2))
    assert any("NÃO derivam do modo" in d for d in teste_errado.defects_against(modos))

    sobreposta = Fold(held_out_mode="pos_hilic", train=(0, 2, 3), test=(0, 1))
    assert any("compartilham índices" in d for d in sobreposta.defects_against(modos))

    sem_treino = Fold(held_out_mode="pos_hilic", train=(), test=(0, 1))
    assert any("treino vazio" in d for d in sem_treino.defects_against(modos))

    sem_teste = Fold(held_out_mode="pos_hilic", train=(2, 3), test=())
    assert any("teste vazio" in d for d in sem_teste.defects_against(modos))

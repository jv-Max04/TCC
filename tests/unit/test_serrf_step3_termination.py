"""Terminação do passo 3 quando se pede mais preditores do que há features.

**A lacuna que este módulo paga, e ela foi achada pelo portão e não por alguém.** Na primeira
execução do mecanismo, o critério apareceu como `lacuna`: `tiny_drift` **não aparecia em
teste nenhum do SERRF**, e não havia caso com `n_correlated` maior que o número de features. O
`test_ramo_3_um_unico_preditor_retido_nao_degenera` usa `n_correlated = 1`, que é a condição
**oposta** — e a semelhança de nome é o que fazia a lacuna parecer coberta.

**O que o original faz aqui, e é a razão de o teto existir.** A linha do `tools.R` é
`while(length(sel_var) < num)`, e ela cresce `l` indefinidamente: com menos features que `num`, a
condição **nunca** é satisfeita. **O R trava.** Não é lentidão — é laço infinito, e é por isso que
`select_predictors` tem `ceiling`.

**E a terceira cláusula do critério cobria uma lacuna de IMPLEMENTAÇÃO, não só de teste.** O
O critério pede *"o número **efetivo** de preditores registrado em `provenance`"*, e a docstring de
`select_predictors` já dizia *"devolvemos o que houver, com o número efetivo registrado por quem
chama"*. **Quem chamava registrava o PEDIDO** — `n_correlated=10` — e nunca o efetivo. A distância
entre intenção e resultado era invisível no `provenance`, que é justamente onde alguém a procuraria.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from refino.correction.serrf import SerrfCorrector
from refino.io.readers import load_matrix
from tests.factories import render_pair
from tests.fixture_files import tiny_drift

# `tiny_drift` tem 5 features e o critério manda pedir 10 — o dobro do que existe.
N_FEATURES_DO_FIXTURE = 5
N_CORRELATED_IMPOSSIVEL = 10


@pytest.fixture
def matriz(tmp_path):  # type: ignore[no-untyped-def]
    """`tiny_drift` pelo caminho normal de ingestão, e não por construção direta.

    Ler do disco é o que garante que a matriz tem a mesma forma que a do usuário — máscara
    de QC, ordem de injeção e tipos vindos do `Sample_List`. Construir o objeto à mão
    testaria o corretor sobre algo que a ingestão nunca produz.
    """
    medidas, lista = render_pair(tiny_drift())
    # **`newline=""` NÃO é estilo: sem ele este teste não corre no Windows**. `render_*`
    # emite CRLF de propósito — *"como os arquivos reais"* —, e `write_text` sem esta opção
    # traduz cada `\n` para `\r\n` sobre um texto que já o tem, gravando `\r\r\n`. O leitor vê
    # linhas mangled e a ingestão morre com `IndexError` em `readers.py`. Os outros sete sítios
    # que escrevem CSV renderizado já a passavam; estes dois eram os únicos que não.
    (tmp_path / "m.csv").write_text(medidas, encoding="utf-8", newline="")
    (tmp_path / "s.csv").write_text(lista, encoding="utf-8", newline="")
    return load_matrix(tmp_path / "m.csv", tmp_path / "s.csv", "tiny_drift")


def _campo(provenance: str, nome: str) -> int:
    achado = re.search(rf"{nome}=(\d+)", provenance)
    assert achado is not None, f"`{nome}` não está no provenance: {provenance}"
    return int(achado.group(1))


def test_to_3_7_o_corretor_retorna_pedindo_mais_preditores_do_que_ha_features(matriz) -> None:  # type: ignore[no-untyped-def]
    """A primeira cláusula: ele **termina**.

    **Não há `timeout` neste teste, e a ausência é deliberada.** O laço do original não é lento, é
    **infinito**: se o teto sumir, este teste pendura para sempre e a suíte morre sem desfecho — que
    é ruidoso e inequívoco. Um `timeout` transformaria "laço infinito" em "teste lento que falhou",
    e as duas coisas mandam procurar em lugares diferentes.
    """
    assert matriz.n_features == N_FEATURES_DO_FIXTURE, (
        f"`tiny_drift` tem {matriz.n_features} features e o critério pressupõe "
        f"{N_FEATURES_DO_FIXTURE}; sem isso o teste não exercita `n_correlated > n_features`"
    )
    resultado = SerrfCorrector(n_correlated=N_CORRELATED_IMPOSSIVEL).fit_transform(matriz, seed=1)
    assert resultado.matrix.n_features == N_FEATURES_DO_FIXTURE
    assert np.isfinite(np.asarray(resultado.matrix.values)).all()


def test_to_3_7_o_provenance_traz_o_numero_efetivo_e_nao_so_o_pedido(matriz) -> None:  # type: ignore[no-untyped-def]
    """A terceira cláusula, e é a que estava sem implementação.

    Asserção de **presença** e não de ausência: afirma-se que o campo existe **e** que o
    valor é menor que o pedido. Conferir só a existência passaria com o efetivo igual ao pedido,
    é exatamente o estado anterior — o `provenance` trazia `n_correlated=10` e nada dizia que só 4
    foram retidos.
    """
    resultado = SerrfCorrector(n_correlated=N_CORRELATED_IMPOSSIVEL).fit_transform(matriz, seed=1)
    prov = resultado.matrix.provenance[-1]

    assert _campo(prov, "n_correlated") == N_CORRELATED_IMPOSSIVEL, "o pedido tem de continuar lá"
    efetivo_max = _campo(prov, "n_predictors_max")
    efetivo_min = _campo(prov, "n_predictors_min")

    assert efetivo_max < N_CORRELATED_IMPOSSIVEL, (
        f"o efetivo ({efetivo_max}) não é menor que o pedido ({N_CORRELATED_IMPOSSIVEL}); "
        "com 5 features é impossível reter 10, então este número não pode estar a medir o efetivo"
    )
    assert efetivo_max <= N_FEATURES_DO_FIXTURE - 1, (
        f"o efetivo ({efetivo_max}) excede o máximo possível: a própria feature sai do pool, "
        f"então o teto é {N_FEATURES_DO_FIXTURE - 1}"
    )
    assert 0 <= efetivo_min <= efetivo_max


def test_o_par_causal_pedido_alcancavel_retem_o_que_se_pediu(matriz) -> None:  # type: ignore[no-untyped-def]
    """**O negativo, e sem ele o teste acima é verde vazio.**

    Se `n_predictors_max` fosse sempre menor que o pedido — por defeito de contagem —, a asserção
    acima passaria sem exercitar nada. Aqui o pedido é **alcançável** e o efetivo tem de subir até
    ele: duas execuções idênticas exceto pelo `n_correlated`, com desfechos que **divergem**.
    """
    alcancavel = 2
    resultado = SerrfCorrector(n_correlated=alcancavel).fit_transform(matriz, seed=1)
    assert _campo(resultado.matrix.provenance[-1], "n_predictors_max") == alcancavel, (
        "com pedido alcançável o efetivo tem de igualar o pedido; se não iguala, o campo não está "
        "a medir retenção e o teste do caso impossível não prova nada"
    )


def test_intersecao_vazia_cai_na_politica_do_d57(matriz) -> None:  # type: ignore[no-untyped-def]
    """A segunda cláusula: interseção vazia devolve a feature **sem correção**, sem levantar.

    `n_correlated = 0` força o passo 3 a não reter nada. A política é devolver a feature
    intacta com a causa nomeada — e é o **mesmo** desfecho que o R produz por outro caminho, quando
    sobram zero colunas utilizáveis.
    """
    resultado = SerrfCorrector(n_correlated=0).fit_transform(matriz, seed=1)
    nomeadas = {f for features in resultado.uncorrected.values() for f in features}
    assert nomeadas, (
        "interseção vazia tem de NOMEAR as features não corrigidas; devolver tudo corrigido "
        "silenciosamente é o desfecho que a marcação existe para impedir"
    )
    assert "empty_predictor_intersection" in resultado.uncorrected
    assert np.isfinite(np.asarray(resultado.matrix.values)).all()

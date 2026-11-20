"""Os zeros do SERRF são unidirecionais, sobre os quatro modos reais.

**O que se afirma.** Onde a entrada era zero, a saída é zero; onde era ausente, a
saída é ausente. O sentido é o da especificação: o equipamento deixa a célula vazia quando
não detecta, e correção nenhuma transforma não-detecção em medida.

**O recorte é declarado e não silenciado.** SERRF completo no `pos_hilic` são 6.035
features a 500 árvores cada — três milhões de árvores, e o critério não é sobre
desempenho. Este módulo corre `FEATURES_PER_MODE` features por modo, e o número
aparece na saída do teste. Recorte de cobertura que não aparece lê-se como "cobri
tudo", e é o que a especificação quer evitar quando separa camadas.

A propriedade é **por célula**, então um subconjunto de features é amostra
legítima dela — o que um subconjunto **não** prova é a contagem agregada de zeros
por modo, e essa não é afirmada aqui.

**Marcado `slow` além de `golden`**: o CI de cada commit não vê, e é o noturno que
o roda.
"""

import numpy as np
import pytest

from refino.correction.serrf import SerrfCorrector
from tests.conftest import matrix_from_toserrf

pytestmark = [pytest.mark.golden, pytest.mark.slow]

MODES = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")

# Quantas features de cada tipo. O recorte é **determinístico** — as primeiras do
# arquivo que satisfazem cada condição — e não sorteado: sorteio faria o escopo
# mudar entre execuções, e um golden cujo escopo varia não é golden.
#
# **A seleção é por PRESENÇA DE ZERO, e não por posição, porque a posição falhou.**
# As primeiras 24 features do `pos_c18` não têm zero nenhum, e sobre elas
# `array_equal(out == 0, was_zero)` compara duas máscaras vazias e **passa**:
# verificação verdadeira e vazia, na sua forma mais convincente. Foi o
# guarda de vacuidade deste módulo que pegou, não a revisão.
#
# As features limpas entram junto de propósito: sem elas o recorte seria só o
# regime esparso, e `test_a_correcao_de_fato_mudou_alguma_coisa` mediria uma
# população que não representa o modo.
FEATURES_WITH_ZEROS = 16
FEATURES_WITHOUT_ZEROS = 8
FEATURES_PER_MODE = FEATURES_WITH_ZEROS + FEATURES_WITHOUT_ZEROS


def _subset(matrix: object) -> object:
    """Primeiras `FEATURES_WITH_ZEROS` com zero, mais `FEATURES_WITHOUT_ZEROS` sem.

    Levanta se o modo não tiver features suficientes de cada tipo, em vez de
    devolver recorte menor em silêncio: recorte que encolhe sozinho é o caminho
    para o portão vazio que este módulo acabou de sofrer.
    """
    from refino.domain.matrix import RefinoMatrix

    assert isinstance(matrix, RefinoMatrix)
    values = np.asarray(matrix.values, dtype=np.float64)
    has_zero = (values == 0.0).any(axis=0)

    with_zeros = [int(i) for i in np.flatnonzero(has_zero)[:FEATURES_WITH_ZEROS]]
    without_zeros = [int(i) for i in np.flatnonzero(~has_zero)[:FEATURES_WITHOUT_ZEROS]]
    if len(with_zeros) < FEATURES_WITH_ZEROS or len(without_zeros) < FEATURES_WITHOUT_ZEROS:
        raise AssertionError(
            f"{matrix.mode}: achei {len(with_zeros)} features com zero e "
            f"{len(without_zeros)} sem, e o recorte pede "
            f"{FEATURES_WITH_ZEROS} e {FEATURES_WITHOUT_ZEROS}"
        )

    columns = with_zeros + without_zeros
    return RefinoMatrix(
        values=values[:, columns],
        feature_names=tuple(matrix.feature_names[i] for i in columns),
        samples=matrix.samples,
        mode=matrix.mode,
    )


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_6_zeros_e_ausentes_sobrevivem_ao_serrf(mode: str) -> None:
    """Os dois conjuntos são **idênticos** entre entrada e saída.

    Duas asserções e não uma, porque zero e ausente são coisas diferentes na especificação:
    zero é não-detecção registrada, ausente é célula que o arquivo não trouxe.
    Colapsá-las faria um corretor que troca um pelo outro passar.
    """
    full = matrix_from_toserrf(mode)  # `skip` nomeado quando o dado falta
    matrix = _subset(full)
    values = np.asarray(matrix.values, dtype=np.float64)  # type: ignore[attr-defined]

    was_zero = values == 0.0
    was_missing = np.isnan(values)
    print(
        f"\n{mode}: {FEATURES_PER_MODE} de {len(full.feature_names)} features; "
        f"{int(was_zero.sum())} zeros, {int(was_missing.sum())} ausentes no recorte"
    )

    out = SerrfCorrector().fit_transform(matrix, seed=42).matrix.values  # type: ignore[arg-type]

    assert np.array_equal(out == 0.0, was_zero), (
        f"{mode}: o conjunto de zeros mudou. Zero que virou valor é sinal inventado; "
        "valor que virou zero é o `created` e tem de sobreviver marcado."
    )
    assert np.array_equal(np.isnan(out), was_missing), f"{mode}: o conjunto de ausentes mudou"


@pytest.mark.parametrize("mode", MODES)
def test_o_recorte_tem_zeros_para_afirmar(mode: str) -> None:
    """**O teste que impede que o anterior passe vacuamente.**

    Sobre um recorte sem nenhum zero, `array_equal(out == 0, was_zero)` compara
    duas máscaras vazias e passa — verificação verdadeira e vazia, que é a forma
    mais convincente de portão inerte. Aqui se afirma que
    há zero a preservar.
    """
    matrix = _subset(matrix_from_toserrf(mode))
    values = np.asarray(matrix.values, dtype=np.float64)  # type: ignore[attr-defined]
    n_zeros = int((values == 0.0).sum())
    assert n_zeros > 0, (
        f"{mode}: as primeiras {FEATURES_PER_MODE} features não têm zero nenhum, "
        "então o teste de preservação não afirma nada. Aumente FEATURES_PER_MODE "
        "ou escolha o recorte por presença de zero — e declare a mudança."
    )


@pytest.mark.parametrize("mode", MODES)
def test_a_correcao_de_fato_mudou_alguma_coisa(mode: str) -> None:
    """A outra metade da vacuidade: preservar zeros é trivial se nada mudou.

    Um corretor que devolvesse a entrada intacta passaria os dois testes acima com
    folga. Esta asserção exige que as posições **não** zero e **não** ausentes
    tenham mudado — ou seja, que houve correção a que os zeros sobreviveram.
    """
    matrix = _subset(matrix_from_toserrf(mode))
    values = np.asarray(matrix.values, dtype=np.float64)  # type: ignore[attr-defined]
    out = SerrfCorrector().fit_transform(matrix, seed=42).matrix.values  # type: ignore[arg-type]

    touchable = (values != 0.0) & ~np.isnan(values)
    assert touchable.any(), f"{mode}: recorte sem célula corrigível"
    assert not np.allclose(out[touchable], values[touchable]), (
        f"{mode}: a saída é igual à entrada nas posições corrigíveis. Os zeros "
        "estarem preservados não diz nada se o corretor não corrigiu."
    )


@pytest.mark.parametrize("mode", MODES)
def test_a_metade_dos_ausentes_e_vacua_no_dado_real_e_isso_fica_declarado(mode: str) -> None:
    """**Vacuidade declarada em vez de silenciosa**, e ela é estrutural.

    Medido nos quatro arquivos: **nenhuma** célula ausente. O equipamento escreve
    `0` quando não detecta, nunca deixa a célula vazia — 64.780 zeros no
    `pos_hilic` e zero ausentes, e o mesmo padrão nos outros três. Logo a asserção
    `array_equal(isnan(out), was_missing)` de `test_ca_3_6_...` compara **duas
    máscaras vazias** e passa por construção.

    Isso não é defeito do recorte e nenhum `FEATURES_PER_MODE` conserta. Quem
    exercita a metade dos ausentes é
    `tests/unit/test_serrf_corrector.py::test_ca_3_6_zeros_e_ausentes_sao_unidirecionais`,
    que **planta** um `NaN` — e é lá que a asserção tem conteúdo.

    Este teste existe para que a próxima pessoa não conclua, de "12 passed", que a
    preservação de ausentes foi verificada sobre dado real. Se um arquivo novo
    trouxer ausentes, ele reprova e avisa que a metade vacua deixou de ser vacua.
    """
    values = np.asarray(matrix_from_toserrf(mode).values, dtype=np.float64)
    n_missing = int(np.isnan(values).sum())
    assert n_missing == 0, (
        f"{mode}: apareceram {n_missing} células ausentes no dado real. A metade dos "
        "ausentes do [CA-3.6] deixou de ser vácua sobre este arquivo — agora ela "
        "pode e deve ser afirmada aqui, e não só no teste com NaN plantado."
    )

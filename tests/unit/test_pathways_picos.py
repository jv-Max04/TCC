"""Os nomes de feature lidos como picos — e a distinção que custa um próton.

**O teste que carrega este arquivo é `test_mesmo_numero_sufixo_diferente_da_mz_diferente`.**
Os outros verificam a leitura; esse verifica que a leitura SEPARA. Sem ele, uma
implementação que ignorasse o sufixo e devolvesse sempre o número cru passaria em
tudo o resto — que é a forma da verificação vácua.
"""

import pytest

from refino.pathways.picos import (
    MASSA_DO_PROTON,
    interpreta,
    interpreta_todos,
)
from tests.conftest import measurements_path, sample_list_path

# As contagens medidas nos quatro arquivos do laboratório em 18/09/2026.
# **São OBSERVAÇÃO sobre estes quatro arquivos, não invariante**: um quinto
# arquivo tem as proporções dele, e a proporção de massas neutras depende da lista de
# adutos que o operador escolheu antes de importar no Progenesis.
CONTAGENS_MEDIDAS = {
    "pos_hilic": (6035, 4648, 1387),
    "pos_c18": (2572, 2003, 569),
    "neg_hilic": (2039, 1891, 148),
    "neg_c18": (2184, 1927, 257),
}


def test_sufixo_mz_e_lido_como_ion() -> None:
    pico = interpreta("2.25_810.7739m/z")
    assert pico is not None
    assert pico.tipo == "ion"
    assert pico.rt == pytest.approx(2.25)
    assert pico.massa == pytest.approx(810.7739)


def test_sufixo_n_e_lido_como_massa_neutra() -> None:
    pico = interpreta("23.59_464.2346n")
    assert pico is not None
    assert pico.tipo == "neutral_mass"
    assert pico.rt == pytest.approx(23.59)
    assert pico.massa == pytest.approx(464.2346)


@pytest.mark.parametrize(
    "nome",
    [
        "2.25_810.7739",  # sem sufixo nenhum
        "810.7739m/z",  # sem tempo de retenção
        "2.25_810.7739mz",  # sem a barra
        "2.25_810.7739 n",  # espaço interno
        "nome inventado",
        "",
    ],
)
def test_forma_desconhecida_devolve_none_em_vez_de_levantar(nome: str) -> None:
    """Forma desconhecida é DADO e não defeito de programação.

    Levantar faria um quinto arquivo com uma terceira forma abortar a análise
    inteira, em vez de reportar quantos nomes não soube ler.
    """
    assert interpreta(nome) is None


def test_mesmo_numero_sufixo_diferente_da_mz_diferente() -> None:
    """O par causal do módulo: a distinção do sufixo TEM de mudar a conta.

    Duas entradas idênticas menos o sufixo. Se `como_mz` devolvesse o mesmo para as
    duas, a separação entre íon e massa neutra seria decorativa — e todo o resto
    deste arquivo passaria na mesma.
    """
    ion = interpreta("2.25_464.2346m/z")
    neutra = interpreta("2.25_464.2346n")
    assert ion is not None and neutra is not None

    assert ion.como_mz("positive") != neutra.como_mz("positive")
    assert neutra.como_mz("positive") - ion.como_mz("positive") == pytest.approx(MASSA_DO_PROTON)
    # E na direção oposta o próton é SUBTRAÍDO: um sinal trocado aqui deslocaria
    # toda feature neutra em 2 prótons no modo negativo, e a busca por ppm não
    # acharia nada — falha que se manifesta como "nenhuma via", não como erro.
    assert ion.como_mz("negative") - neutra.como_mz("negative") == pytest.approx(MASSA_DO_PROTON)


def test_ion_e_invariante_ao_modo() -> None:
    """Para um íon já medido, `como_mz` é identidade nos dois modos.

    A polaridade já está embutida no número que o equipamento reportou; aplicá-la
    outra vez seria contá-la duas vezes.
    """
    ion = interpreta("2.25_810.7739m/z")
    assert ion is not None
    assert ion.como_mz("positive") == ion.massa
    assert ion.como_mz("negative") == ion.massa


def test_interpreta_todos_separa_e_preserva_o_que_nao_leu() -> None:
    resumo = interpreta_todos(("1.0_100.0m/z", "2.0_200.0n", "lixo", "3.0_300.0m/z"))
    assert [p.nome for p in resumo.ions] == ["1.0_100.0m/z", "3.0_300.0m/z"]
    assert [p.nome for p in resumo.massas_neutras] == ["2.0_200.0n"]
    assert resumo.nao_interpretados == ("lixo",)
    assert resumo.total == 4


def test_total_do_resumo_nao_perde_ninguem() -> None:
    """A soma das três gavetas é a entrada inteira.

    É a asserção que impede a gaveta de não interpretados de ser esquecida: sem ela,
    uma implementação que descartasse silenciosamente o que não casou passaria nos
    dois testes acima.
    """
    nomes = (*(f"{i}.0_{100 + i}.0m/z" for i in range(5)), "a", "b")
    resumo = interpreta_todos(nomes)
    assert resumo.total == len(nomes)


@pytest.mark.parametrize("modo", sorted(CONTAGENS_MEDIDAS))
def test_contagens_sobre_os_arquivos_reais(modo: str) -> None:
    """As duas formas, contadas nos arquivos do laboratório.

    Sobre os arquivos REAIS e não sobre matriz de factory (regra 1 do README): uma factory
    que reproduzisse a proporção estaria a testar a factory.
    """
    from refino.io.readers import load_matrix

    matriz = load_matrix(measurements_path(modo), sample_list_path(modo), modo)
    resumo = interpreta_todos(matriz.feature_names)

    total, ions, neutras = CONTAGENS_MEDIDAS[modo]
    assert resumo.total == total
    assert len(resumo.ions) == ions
    assert len(resumo.massas_neutras) == neutras
    # A partição é limpa nestes quatro arquivos. A asserção é sobre eles, e é ela que
    # avisaria se um arquivo novo trouxesse uma terceira forma — em vez de a terceira
    # forma desaparecer da análise sem ninguém notar.
    assert resumo.nao_interpretados == ()

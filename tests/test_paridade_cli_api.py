"""A superfície de exaustividade da paridade CLI↔API.

**O que ela fecha, e por que não bastava consertar as três rotas.** `refino corrigir` e
`refino exportar` nasceram sem rota, e `POST /execucoes` calculava a matriz processada e a
descartava. Nada acusou. Consertadas as três, **um subcomando novo nasceria igual** — e é essa a
divergência que este portão impede, não a de hoje.

**A pergunta que ele responde:**

> **toda operação da CLI que escreve arquivo tem rota que produz o mesmo artefato?**

Derivado dos **dois** lados: os subcomandos saem de `build_parser()` por introspecção do
`argparse`; as rotas saem do manifesto de `docs/rotas-api.md`, que é o mesmo que
`tests/test_api_contrato.py` confere contra a aplicação. Nenhuma lista escrita à mão.

## A condição deste portão

O negativo que importa **não** é uma exceção mal declarada — é **remover uma rota existente e ver
o portão acusar por nome**. Um portão que só pegasse exceção malformada nasceria sem alcançar o
caso que o motivou:

> Quem planta escolhe o caso fácil, e o portão passa a provar a classe em vez do instante. Aqui
> o instante é *"alguém acrescentou subcomando e esqueceu a rota"*, e a planta que o reproduz é
> remover a rota, não corromper a lista de exceções.

Os dois negativos estão abaixo, e o primeiro é o que a condição exige.
"""

from __future__ import annotations

import argparse
import re

import pytest

from refino.cli import build_parser
from tests.conftest import REPO

MANIFESTO = REPO / "docs" / "rotas-api.md"

_LINHA = re.compile(r"^\|\s*`(?P<metodo>[A-Z]+)`\s*\|\s*`(?P<caminho>[^`]+)`\s*\|")

# **A rota que serve cada subcomando que escreve arquivo.** É o mapa que o portão confere, e não
# a fonte de nenhum dos dois lados: os subcomandos vêm do `argparse`, as rotas do manifesto, e
# esta tabela só diz qual casa com qual.
ROTA_DO_SUBCOMANDO: dict[str, str] = {
    "corrigir": "/matrizes/corrigir",
    "processar": "/matrizes/processar",
    "exportar": "/matrizes/exportar",
}

# **Subcomando que deliberadamente NÃO tem rota.** Vazia hoje, e entrada nela exige justificativa
# escrita aqui. Uma lista de isenção sem motivo é a mesma coisa que não ter portão, com a
# aparência de ter.
SEM_ROTA_POR_DECISAO: dict[str, str] = {}


def subcomandos_que_escrevem_arquivo() -> set[str]:
    """Subcomandos com `--output`, por introspecção do `argparse`.

    **Introspecção e não lista:** um subcomando novo aparece aqui no dia em que é escrito, que é
    a única forma de o portão alcançar a divergência que ainda não aconteceu.
    """
    parser = build_parser()
    achados: set[str] = set()
    for acao in parser._actions:
        if not isinstance(acao, argparse._SubParsersAction):
            continue
        for nome, sub in acao.choices.items():
            if any("--output" in (a.option_strings or ()) for a in sub._actions):
                achados.add(nome)
    return achados


def rotas_do_manifesto(texto: str | None = None) -> set[str]:
    """Os caminhos declarados no manifesto das rotas."""
    bruto = MANIFESTO.read_text(encoding="utf-8") if texto is None else texto
    return {m["caminho"] for linha in bruto.splitlines() if (m := _LINHA.match(linha))}


def violacoes(subcomandos: set[str], rotas: set[str]) -> list[str]:
    """Subcomandos que escrevem arquivo e cuja rota não está no manifesto.

    Função e não corpo de teste porque é o que torna os negativos honestos: a planta percorre o
    mesmo caminho que a árvore limpa.
    """
    fora: list[str] = []
    for nome in sorted(subcomandos):
        if nome in SEM_ROTA_POR_DECISAO:
            continue
        rota = ROTA_DO_SUBCOMANDO.get(nome)
        if rota is None:
            fora.append(f"{nome}: escreve arquivo e não tem rota mapeada")
        elif rota not in rotas:
            fora.append(f"{nome}: mapeado para `{rota}`, que o manifesto não declara")
    return fora


def test_ca_12_11_todo_subcomando_que_escreve_arquivo_tem_rota() -> None:
    """A árvore limpa: os três têm rota."""
    subcomandos = subcomandos_que_escrevem_arquivo()

    assert subcomandos, "nenhum subcomando escreve arquivo; a asserção abaixo seria vácua"
    assert violacoes(subcomandos, rotas_do_manifesto()) == []


def test_ca_12_11_a_enumeracao_acha_os_tres_que_a_unidade_conhece() -> None:
    """Não-vacuidade com nome: se a introspecção quebrar, o portão acima passa por vazio.

    Asserção de ausência é satisfeita por conjunto vazio, e a guarda vai na mesma função que dela
    depende, não num teste vizinho.
    """
    assert subcomandos_que_escrevem_arquivo() == {"corrigir", "processar", "exportar"}


@pytest.mark.parametrize(
    "rota_removida", ["/matrizes/corrigir", "/matrizes/processar", "/matrizes/exportar"]
)
def test_ca_12_11_o_portao_acusa_por_nome_quando_uma_rota_real_some(rota_removida: str) -> None:
    """**A condição deste portão, e é o negativo que importa.**

    Não é uma exceção mal declarada: é **remover uma rota que existe** e exigir que o portão
    acuse, **nomeando o subcomando órfão**. Um portão que só pegasse lista de isenção corrompida
    nasceria sem alcançar o caso que o motivou, e é por isso que este teste é parametrizado nas
    três em vez de plantado numa.
    """
    restantes = rotas_do_manifesto() - {rota_removida}

    achados = violacoes(subcomandos_que_escrevem_arquivo(), restantes)

    esperado = next(k for k, v in ROTA_DO_SUBCOMANDO.items() if v == rota_removida)
    assert len(achados) == 1, f"esperava uma violação e vieram {achados}"
    assert achados[0].startswith(f"{esperado}:"), (
        f"o portão acusou sem nomear o subcomando órfão: {achados[0]}"
    )
    assert rota_removida in achados[0], "a mensagem não diz QUAL rota falta"


def test_ca_12_11_com_a_rota_de_volta_o_portao_deixa_de_acusar() -> None:
    """O par causal: ver o negativo deixar de acusar não é ver o portão reprovar."""
    assert violacoes(subcomandos_que_escrevem_arquivo(), rotas_do_manifesto()) == []


def test_ca_12_11_subcomando_novo_sem_rota_reprova() -> None:
    """O caso que o portão existe para impedir: o quarto artefato, que ainda não existe.

    É o mesmo defeito que produziu esta unidade — `corrigir` e `exportar` nasceram sem rota — na
    forma em que ele voltaria.
    """
    achados = violacoes({*subcomandos_que_escrevem_arquivo(), "relatar"}, rotas_do_manifesto())

    assert achados == ["relatar: escreve arquivo e não tem rota mapeada"]


def test_ca_12_11_a_direcao_inversa_rota_de_artefato_sem_subcomando() -> None:
    """Rota órfã: toda rota mapeada corresponde a um subcomando que existe.

    Sem esta direção, apagar `refino exportar` deixaria `/matrizes/exportar` servindo um artefato
    que nenhum caminho de linha de comando produz — e a paridade estaria quebrada do outro lado,
    com o portão verde.
    """
    orfas = sorted(set(ROTA_DO_SUBCOMANDO) - subcomandos_que_escrevem_arquivo())

    assert orfas == [], f"rota de artefato sem subcomando equivalente: {orfas}"


def test_ca_12_11_a_lista_de_excecoes_esta_vazia_e_toda_entrada_tem_justificativa() -> None:
    """Isenção sem motivo é não ter portão, com a aparência de ter.

    Hoje a lista é **vazia**, e isso é afirmação e não acaso: se alguém acrescentar entrada, ela
    tem de trazer texto — e este teste é o que a obriga.
    """
    assert SEM_ROTA_POR_DECISAO == {}, (
        "a lista de exceções deixou de ser vazia; cada entrada precisa de justificativa escrita"
    )
    for nome, motivo in SEM_ROTA_POR_DECISAO.items():
        assert len(motivo) > 40, f"{nome}: justificativa curta demais para ser justificativa"


def test_ca_12_11_o_manifesto_declara_as_tres_rotas_de_artefato() -> None:
    """Âncora: se o manifesto não for lido, todos os testes acima passam por vazio."""
    rotas = rotas_do_manifesto()

    assert rotas, "o manifesto não foi lido"
    for rota in ROTA_DO_SUBCOMANDO.values():
        assert rota in rotas, f"o manifesto não declara `{rota}`"


def test_ca_12_11_o_leitor_do_manifesto_consegue_devolver_vazio() -> None:
    """O leitor não é imune: um manifesto sem tabela devolve conjunto vazio, e é o que a âncora
    acima existe para pegar. Sem este teste, `rotas_do_manifesto` poderia estar sempre a devolver
    o conjunto completo por defeito e ninguém saberia.
    """
    assert rotas_do_manifesto("texto sem tabela nenhuma") == set()

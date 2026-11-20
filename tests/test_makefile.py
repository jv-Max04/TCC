"""O Makefile declara os alvos do projeto, e cada alvo pode passar pelo que seleciona."""

import re

import pytest

from tests.conftest import REPO

MAKEFILE = REPO / "Makefile"

# Alvos que a documentação original do projeto prometia. Estão aqui, e não
# derivados dos documentos, porque este conjunto é o contrato: um deles
# desaparecer do Makefile é regressão mesmo que ninguém o cite no dia.
PROMISED = frozenset({"setup", "test", "test-golden", "test-all", "lint", "check"})

# Alvos cuja seleção é um SUBCONJUNTO da suíte e que por isso não alcançam o
# portão de cobertura de 85% por construção — sozinhos, os golden cobrem 68,11%.
# Sem `--no-cov` eles reprovam com todos os testes verdes.
#
# Não é afrouxar o portão: quem o mede é `make check`, sobre a seleção inteira. É
# não exigir de uma parte o número que só a soma pode dar. Cobertura medida sobre recorte
# arbitrário é número verde ou vermelho sem sinal nenhum.
SUBSET_TARGETS = frozenset({"test-golden"})
FULL_COVERAGE_TARGETS = frozenset({"test", "test-all", "check"})


def _declared_targets(text: str) -> set[str]:
    """Nomes de alvo declarados no Makefile, ignorando variáveis e receitas."""
    return {
        match.group(1)
        for line in text.splitlines()
        if (match := re.match(r"^([a-zA-Z][\w.-]*)\s*:(?!=)", line))
    }


@pytest.fixture(scope="module")
def declared() -> set[str]:
    assert MAKEFILE.is_file(), "o Makefile é referenciado pela documentação e precisa existir"
    return _declared_targets(MAKEFILE.read_text(encoding="utf-8"))


def test_makefile_declara_os_alvos_prometidos(declared: set[str]) -> None:
    faltando = PROMISED - declared
    assert not faltando, f"alvos prometidos ausentes do Makefile: {sorted(faltando)}"


def test_phony_cobre_todos_os_alvos_prometidos(declared: set[str]) -> None:
    """Alvo que não produz arquivo com o próprio nome tem de ser .PHONY.

    Sem isso, um diretório chamado `test` na raiz faz `make test` reportar
    'nothing to be done' — sucesso silencioso onde deveria haver execução.
    """
    text = MAKEFILE.read_text(encoding="utf-8")
    phony = set(re.findall(r"^\.PHONY:\s*(.*)$", text, re.MULTILINE))
    nomes = {nome for linha in phony for nome in linha.split()}
    assert nomes >= PROMISED, f"fora do .PHONY: {sorted(PROMISED - nomes)}"


def test_check_agrega_lint_e_testes() -> None:
    """`check` é o que se aponta como portão de fim de fase."""
    text = MAKEFILE.read_text(encoding="utf-8")
    corpo = re.search(r"^check:(.*?)(?=^\S|\Z)", text, re.MULTILINE | re.DOTALL)
    assert corpo, "o Makefile precisa de um alvo `check`"
    bloco = corpo.group(1)
    assert "lint" in bloco.splitlines()[0], "`check` precisa depender de `lint`"
    assert "pytest" in bloco or "PYTEST" in bloco, "`check` precisa rodar a suíte"


def test_alvos_usam_o_interpretador_do_ambiente() -> None:
    """Receita que chama `python` solto roda no interpretador do sistema.

    O projeto pina 3.12 em quatro lugares; um alvo que ignore o `.venv` reintroduz
    a deriva que o teste de paridade de versão existe para pegar.
    """
    text = MAKEFILE.read_text(encoding="utf-8")
    receitas = [linha for linha in text.splitlines() if linha.startswith("\t")]
    soltas = [
        linha
        for linha in receitas
        if re.search(r"(?<![.\w/$(])python\b", linha) and "python3.12 -m venv" not in linha
    ]
    assert not soltas, f"receitas chamando python fora do .venv: {soltas}"


@pytest.mark.parametrize("target", sorted(SUBSET_TARGETS))
def test_alvo_de_subconjunto_desliga_o_portao_de_cobertura(target: str) -> None:
    """Alvo que seleciona parte da suíte tem de passar `--no-cov`.

    **Defeito real, e ficou dois anos-fase invisível.** `make test-golden` está
    documentado desde o início e nunca pôde passar: 182 testes verdes e o portão
    de 85% reprovando sobre 68,11%. Ninguém rodou o alvo.

    A paridade pergunta se o alvo **existe**. Esta pergunta se ele **pode
    passar**, que é outra coisa e não era verificada.
    """
    recipe = _recipe_of(MAKEFILE.read_text(encoding="utf-8"), target)
    assert "--no-cov" in recipe, (
        f"`{target}` seleciona um subconjunto da suíte e não desliga o portão de "
        f"cobertura: ele reprova com todos os testes verdes. Receita: {recipe!r}"
    )


@pytest.mark.parametrize("target", sorted(FULL_COVERAGE_TARGETS))
def test_alvo_de_suite_inteira_mantem_o_portao(target: str) -> None:
    """E a direção oposta, porque portão desligado em todo lugar é portão ausente.

    Se `--no-cov` vazasse para `check`, o piso de 85% deixaria de existir sem que
    nada acusasse — e é justamente o modo de falha em que a métrica não
    piorou, ela parou de medir.
    """
    recipe = _recipe_of(MAKEFILE.read_text(encoding="utf-8"), target)
    # Afirmação de não-vacuidade: `"--no-cov" not in ""` é verdadeiro, então um
    # extrator quebrado deixaria este teste verde para sempre. O irmão acima não
    # tem o problema — lá a ausência do flag reprova —, e é por isso que só este
    # precisa da guarda.
    assert "$(PYTEST)" in recipe, f"receita de `{target}` não extraída: {recipe!r}"
    assert "--no-cov" not in recipe, (
        f"`{target}` roda a suíte inteira e não pode desligar a cobertura. Receita: {recipe!r}"
    )


def _recipe_of(text: str, target: str) -> str:
    """Linhas de receita de um alvo, concatenadas."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"{target}:"):
            body = []
            for following in lines[index + 1 :]:
                if following.startswith("\t"):
                    body.append(following.strip())
                elif following.strip() and not following.startswith(("#", " ")):
                    break
            return " ".join(body)
    raise AssertionError(f"alvo {target} não encontrado no Makefile")

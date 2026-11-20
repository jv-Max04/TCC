"""Confere que o que o README promete existe.

O README é o primeiro arquivo que um recém-chegado lê, e `pyproject.toml` o declara como
`readme`. Uma instrução que apodreceu manda quem chega por um caminho que não funciona, então
cada teste compara uma promessa do texto com a fonte dela: o Makefile, o `pyproject.toml`, a
linha de comando e a árvore de arquivos.

Fica de fora o que só se confere executando: que a suíte passa com as seleções documentadas,
que a janela abre e que os roteiros de congelamento correm no Windows.
"""

import argparse
import importlib.util
import re
import shlex

import pytest

from refino.cli import build_parser
from tests.conftest import REPO, pinned_python_minor, pyproject

README = REPO / "README.md"
MAKEFILE = REPO / "Makefile"
MAX_SETUP_COMMANDS = 5

SECOES = (
    "Instalação",
    "Dados de referência",
    "Uso",
    "Testes",
    "Estrutura do repositório",
)

# Raízes versionadas. `serrf/`, `data/reference/`, `out/`, `dist/` e `.venv/` ficam de fora
# porque não viajam no clone: exigir que existam reprovaria em toda máquina sem os dados.
RAIZES_VERSIONADAS = ("src", "tests", "scripts", "docs", "r_service", "data/fixtures")
CAMINHO = re.compile(r"(?<![\w/\\.:~-])((?:" + "|".join(RAIZES_VERSIONADAS) + r")[/\\][\w./\\-]*)")


def _texto() -> str:
    return README.read_text(encoding="utf-8")


def _secao(titulo: str) -> str:
    """O corpo de uma seção de nível dois, até a próxima do mesmo nível."""
    match = re.search(
        rf"^## {re.escape(titulo)}[ \t]*\n(.*?)(?=^## |\Z)", _texto(), re.DOTALL | re.MULTILINE
    )
    assert match, f"o README precisa da seção '## {titulo}'"
    return match.group(1)


def _blocos(texto: str) -> list[list[str]]:
    """Os comandos de cada bloco de código, com as linhas de continuação já unidas."""
    blocos = []
    for corpo in re.findall(r"```[a-z]*\n(.*?)```", texto, re.DOTALL):
        unido = re.sub(r"\\\n\s*", "", corpo)
        blocos.append(
            [
                linha.strip()
                for linha in unido.splitlines()
                if linha.strip() and not linha.strip().startswith("#")
            ]
        )
    return blocos


def _makefile() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def _alvos_do_makefile() -> set[str]:
    return set(re.findall(r"^([a-zA-Z][\w.-]*)\s*:(?!=)", _makefile(), re.MULTILINE))


def _receita(alvo: str) -> list[str]:
    """As linhas de receita de um alvo, com as variáveis do Makefile expandidas."""
    texto = _makefile()
    variaveis = dict(re.findall(r"^(\w+)\s*:?=\s*(.*)$", texto, re.MULTILINE))
    linhas = texto.splitlines()
    inicio = next(i for i, linha in enumerate(linhas) if linha.startswith(f"{alvo}:"))
    receita = []
    for linha in linhas[inicio + 1 :]:
        if not linha.startswith("\t"):
            break
        comando = linha.strip()
        # Duas passadas, porque uma variável pode citar outra: `PYTEST := $(PY) -m pytest`.
        for _ in range(2):
            for nome, valor in variaveis.items():
                comando = comando.replace(f"$({nome})", valor.strip())
        receita.append(comando)
    return receita


def _caminhos_ausentes(texto: str) -> tuple[set[str], list[str]]:
    """Os caminhos versionados que o texto cita, e os que não existem na árvore."""
    citados = {
        achado.replace("\\", "/").rstrip(".").rstrip("/") for achado in CAMINHO.findall(texto)
    }
    return citados, sorted(c for c in citados if not (REPO / c).exists())


def _subcomandos_da_cli() -> set[str]:
    acoes = [a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)]
    assert len(acoes) == 1, "a CLI deixou de ter um único grupo de subcomandos"
    return set(acoes[0].choices)


def _exemplos_da_cli() -> list[list[str]]:
    """Os argumentos de cada invocação de `refino` nos blocos de código, fora as de `--help`."""
    return [
        shlex.split(comando)[1:]
        for bloco in _blocos(_texto())
        for comando in bloco
        if comando.startswith(".venv/bin/refino ") and "--help" not in comando
    ]


def test_readme_exists() -> None:
    """`pyproject.toml` declara `readme = "README.md"`: sem o arquivo, o `pip install -e` falha."""
    assert pyproject()["project"]["readme"] == "README.md"
    assert README.is_file()


@pytest.mark.parametrize("titulo", SECOES)
def test_readme_has_the_expected_sections(titulo: str) -> None:
    """Uma seção que some ou muda de nome reprova aqui, e atualizar a lista é escolha de alguém."""
    assert re.search(rf"^## {re.escape(titulo)}[ \t]*$", _texto(), re.MULTILINE), (
        f"seção ausente: ## {titulo}"
    )


def test_readme_states_the_pinned_python_minor() -> None:
    """O README manda criar a venv com o mesmo minor que o `pyproject.toml` exige."""
    minor = pinned_python_minor()
    texto = _texto()
    assert f"Python {minor}" in texto, f"o README não menciona Python {minor}"
    assert f"python{minor} -m venv" in texto, f"o README não cria a venv com python{minor}"


def test_a_instalacao_cabe_em_cinco_comandos() -> None:
    """A instalação tem dois caminhos, o alvo `setup` e os comandos à mão, e cada um é curto.

    São exatamente dois blocos para a contagem não ficar ambígua: um terceiro bloco seria um
    passo de instalação que nenhum dos dois caminhos conta.
    """
    blocos = _blocos(_secao("Instalação"))
    assert len(blocos) == 2, f"a seção de instalação tem {len(blocos)} blocos de código, e não 2"
    assert blocos[0] == ["make setup"]
    for bloco in blocos:
        assert 1 <= len(bloco) <= MAX_SETUP_COMMANDS, (
            f"um caminho de instalação tem {len(bloco)} comandos, máximo {MAX_SETUP_COMMANDS}:\n"
            + "\n".join(f"  {c}" for c in bloco)
        )


def test_os_comandos_a_mao_sao_a_receita_do_setup() -> None:
    """O README diz que os comandos à mão são o que `make setup` faz, e isto confere a frase.

    Sem a comparação, o Makefile poderia ganhar um passo e quem instalasse à mão ficaria com
    um ambiente diferente do de quem usou o alvo.
    """
    receita = _receita("setup")
    assert receita, "a receita de `setup` não foi extraída do Makefile"
    assert _blocos(_secao("Instalação"))[1] == receita


def test_todo_alvo_de_make_citado_existe() -> None:
    citados = set(re.findall(r"\bmake ([a-z][\w-]*)", _texto()))
    assert {"setup", "lint"} <= citados, f"a extração dos alvos citados falhou: {sorted(citados)}"
    ausentes = citados - _alvos_do_makefile()
    assert not ausentes, f"o README cita alvos que o Makefile não declara: {sorted(ausentes)}"


def test_todo_caminho_versionado_citado_existe() -> None:
    """Os blocos de código e a prosa só citam arquivos e pastas que estão na árvore."""
    citados, ausentes = _caminhos_ausentes(_texto())
    assert {
        "scripts/fetch_reference_data.sh",
        "scripts/congela_windows.ps1",
        "tests/unit",
        "src/refino",
    } <= citados, f"a extração dos caminhos citados falhou: {sorted(citados)}"
    assert not ausentes, f"o README cita caminhos que não existem: {ausentes}"


def test_um_caminho_inexistente_e_acusado() -> None:
    """O par do teste acima: com um caminho inventado no texto, a conferência o devolve."""
    _, ausentes = _caminhos_ausentes(_texto() + "\nVeja `scripts/roteiro_que_nao_existe.sh`.\n")
    assert ausentes == ["scripts/roteiro_que_nao_existe.sh"]


def test_todo_modulo_executado_com_menos_m_existe() -> None:
    """`python -m <módulo>` só aparece com módulo que a instalação documentada traz."""
    modulos = set(re.findall(r"python[\d.]* -m ([\w.]+)", _texto()))
    assert {"pytest", "refino.desktop.arranque"} <= modulos, sorted(modulos)
    ausentes = sorted(m for m in modulos if importlib.util.find_spec(m) is None)
    assert not ausentes, f"o README manda executar módulos que não existem: {ausentes}"


def test_os_subcomandos_listados_sao_os_da_cli() -> None:
    """A tabela de subcomandos e a CLI têm os mesmos nomes, nas duas direções."""
    listados = set(re.findall(r"^\| `(\w+)` \|", _secao("Uso"), re.MULTILINE))
    assert listados == _subcomandos_da_cli()


def test_os_exemplos_da_cli_sao_aceitos_pelo_parser() -> None:
    """Cada exemplo de `refino` usa um subcomando e opções que a CLI aceita.

    O parser recusa opção desconhecida e opção obrigatória ausente com `SystemExit`. Os valores
    de vocabulário fechado, como o de `--method`, são validados depois e não por ele.
    """
    exemplos = _exemplos_da_cli()
    assert len(exemplos) >= 2, f"esperava ao menos dois exemplos da CLI, achei {len(exemplos)}"
    for argumentos in exemplos:
        assert build_parser().parse_args(argumentos).command == argumentos[0]


def test_um_exemplo_com_opcao_inventada_e_recusado(capsys: pytest.CaptureFixture[str]) -> None:
    """O par do teste acima: o mesmo exemplo com uma opção que não existe não passa."""
    argumentos = [*_exemplos_da_cli()[0], "--opcao-que-nao-existe", "1"]
    with pytest.raises(SystemExit):
        build_parser().parse_args(argumentos)
    assert "--opcao-que-nao-existe" in capsys.readouterr().err


def test_os_marcadores_citados_estao_declarados() -> None:
    """As seleções `-m "..."` e a tabela de marcadores só usam marcadores do `pyproject.toml`.

    O pytest não reclama de marcador desconhecido numa expressão de `-m`: ele seleciona como se
    nenhum teste o tivesse. Uma seleção documentada com um marcador que saiu do projeto rodaria
    outra coisa sem avisar.
    """
    declarados = {
        linha.split(":", 1)[0].strip()
        for linha in pyproject()["tool"]["pytest"]["ini_options"]["markers"]
    }
    expressoes = re.findall(r'-m "([^"]+)"', _texto())
    assert expressoes, "o README não tem nenhuma seleção por marcador"
    usados = {
        nome
        for expressao in expressoes
        for nome in re.findall(r"[A-Za-z_]\w*", expressao)
        if nome not in {"and", "or", "not"}
    }
    tabela = set(re.findall(r"^\| `(\w+)` \|", _secao("Testes"), re.MULTILINE))
    assert tabela, "a seção de testes perdeu a tabela de marcadores"
    assert not (usados | tabela) - declarados, (
        f"marcadores citados e não declarados: {sorted((usados | tabela) - declarados)}"
    )

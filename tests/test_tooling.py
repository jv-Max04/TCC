"""Verifica lint e tipos.

Cada portão é testado duas vezes: uma afirmando que a árvore limpa passa, e
outra plantando uma violação e afirmando que o portão reprova. A segunda é a
que importa. Um `mypy` apontado para o caminho errado, ou um override cujo
padrão de módulo não casa, produz "Success: no issues found" — verde
indistinguível de um estrito de verdade.
"""

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import REPO, pyproject, run_module

# Pacotes que a especificação nomeia para mypy estrito.
STRICT_PACKAGES = ["domain", "metrics", "correction"]

# Sonda de estrito: genérico nu (`list` em vez de `list[int]`).
#
# A escolha da sonda não é arbitrária. O candidato óbvio — função sem
# anotação — NÃO serve: o mypy 2.x passou a reprovar `no-untyped-def` por
# padrão, então ele acusa tanto sob estrito quanto sob a configuração base, e
# um teste construído sobre ele ficaria verde mesmo com o estrito desligado.
# Medido no mypy 2.3.0: `def f() -> list` dá 0 erro no padrão e 1 sob
# `strict` (`disallow_any_generics`). É a diferença que discrimina.
BARE_GENERIC = '''"""Sonda temporária do teste de tipos."""


def devolve_generico_nu() -> list:
    return []
'''


@pytest.fixture
def planted_untyped_module() -> Iterator[Path]:
    """Fábrica de sonda: escreve um módulo com genérico nu e limpa depois."""
    written: list[Path] = []

    def _plant(relative: str) -> Path:
        path = REPO / relative
        path.write_text(BARE_GENERIC, encoding="utf-8")
        written.append(path)
        return path

    yield _plant  # type: ignore[misc]

    for path in written:
        path.unlink(missing_ok=True)
        shutil.rmtree(path.parent / "__pycache__", ignore_errors=True)
    shutil.rmtree(REPO / ".mypy_cache", ignore_errors=True)


# ---------------------------------------------------------------------------
# ruff
# ---------------------------------------------------------------------------


def test_ruff_passes_on_clean_tree() -> None:
    """`ruff check` retorna zero."""
    result = run_module("ruff", "check", "src", "tests")
    assert result.returncode == 0, f"ruff reprovou:\n{result.stdout}\n{result.stderr}"


def test_ruff_format_is_clean() -> None:
    """A formatação está estável, para que o CI não brigue com o editor."""
    result = run_module("ruff", "format", "--check", "src", "tests")
    assert result.returncode == 0, f"formatação divergente:\n{result.stdout}"


def test_ruff_actually_flags_violations(planted_untyped_module) -> None:
    """ruff está ativo, não apenas configurado.

    A sonda tem import ausente e variável não usada — se ruff estiver
    apontado para o lugar errado, isto passa e o portão é decorativo.
    """
    probe = REPO / "src" / "refino" / "_lint_probe.py"
    probe.write_text(
        '"""Sonda de lint."""\n\nimport os\n\n\ndef f():\n    naoUsada = 1\n    return 2\n',
        encoding="utf-8",
    )
    try:
        result = run_module("ruff", "check", str(probe))
        assert result.returncode != 0, f"ruff não flagrou a sonda:\n{result.stdout}"
        assert "F401" in result.stdout or "unused" in result.stdout.lower()
    finally:
        probe.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# mypy
# ---------------------------------------------------------------------------


def test_mypy_passes_on_domain() -> None:
    """`mypy src/refino/domain` retorna zero."""
    result = run_module("mypy", "src/refino/domain")
    assert result.returncode == 0, f"mypy reprovou em domain:\n{result.stdout}"


@pytest.mark.parametrize("package", STRICT_PACKAGES)
def test_mypy_passes_on_each_strict_package(package: str) -> None:
    """Os três pacotes que a especificação põe sob estrito passam."""
    result = run_module("mypy", f"src/refino/{package}")
    assert result.returncode == 0, f"mypy reprovou em {package}:\n{result.stdout}"


def test_mypy_passes_on_whole_tree() -> None:
    """A árvore inteira passa sob a configuração vigente."""
    result = run_module("mypy", "src/refino")
    assert result.returncode == 0, f"mypy reprovou na árvore:\n{result.stdout}"


@pytest.mark.parametrize("package", STRICT_PACKAGES)
def test_strict_mode_is_active_in_named_packages(
    package: str,
    planted_untyped_module,
) -> None:
    """O estrito está ATIVO nos três pacotes, não só declarado.

    Genérico nu é erro sob `strict` (`disallow_any_generics`) e silêncio sem
    ele. Se este teste ficar verde, o override do pyproject deixou de casar
    com o módulo — que é como o estrito morre em silêncio numa refatoração.
    """
    probe = planted_untyped_module(f"src/refino/{package}/_type_probe.py")
    result = run_module("mypy", str(probe))
    assert result.returncode != 0, (
        f"mypy NÃO reprovou genérico nu em {package}/ — "
        f"o modo estrito não está ativo ali.\n{result.stdout}"
    )
    assert "type-arg" in result.stdout or "type parameters" in result.stdout


def test_strict_mode_is_scoped_and_not_global(planted_untyped_module) -> None:
    """O estrito é por pacote, como a especificação pede — não global.

    O MESMO arquivo que reprova em domain/ passa em api/. Sem esta metade,
    um `strict = true` global também passaria os testes acima, e a especificação seria
    cumprida por acidente e não por desenho.
    """
    probe = planted_untyped_module("src/refino/api/_type_probe.py")
    result = run_module("mypy", str(probe))
    assert result.returncode == 0, (
        "mypy reprovou genérico nu em api/, que NÃO está sob estrito. "
        f"O escopo por pacote da §7 não está sendo respeitado.\n{result.stdout}"
    )


def test_pyproject_declares_strict_for_the_three_named_packages() -> None:
    """Os três pacotes da especificação estão sob os flags de estrito.

    Procura os flags componentes e não a chave `strict`, porque o atalho não
    pode ser usado num override: no mypy 2.3.0 ele vaza para a configuração
    global e destrói o escopo por pacote. Ver o comentário no pyproject.
    """
    config = pyproject()
    strict_modules: set[str] = set()
    for override in config["tool"]["mypy"].get("overrides", []):
        if override.get("disallow_any_generics") and override.get("disallow_untyped_defs"):
            for module in override["module"]:
                strict_modules.add(module.removesuffix(".*"))

    for package in STRICT_PACKAGES:
        assert f"refino.{package}" in strict_modules, (
            f"refino.{package} não está sob os flags de estrito; "
            f"declarados: {sorted(strict_modules)}"
        )


def test_strict_shortcut_is_not_used_in_overrides() -> None:
    """Guarda de regressão contra o vazamento do atalho `strict`.

    Alguém que "simplifique" os onze flags de volta para `strict = true` faz
    a árvore inteira ficar estrita sem que nenhum outro teste acuse — o
    escopo por pacote da especificação morre em silêncio. Este teste é o alarme.
    """
    config = pyproject()
    offenders = [
        override["module"]
        for override in config["tool"]["mypy"].get("overrides", [])
        if "strict" in override
    ]
    assert not offenders, (
        "`strict` usado num [[tool.mypy.overrides]]: no mypy 2.3.0 isso vaza "
        "para a configuração global e a árvore toda fica estrita. Expanda nos "
        f"flags componentes. Overrides afetados: {offenders}"
    )


# ---------------------------------------------------------------------------
# O `ruff format` reescreve blocos de codigo dentro de `.md`
# ---------------------------------------------------------------------------

# Um trecho que ja corrompeu um documento do repositorio. **Argumento nomeado numa chamada**, que
# lido como programa e atribuicao de tupla.
_TRECHO_HOSTIL = "```python\nexposta=True,\n```\n"


def test_a_configuracao_do_ruff_exclui_markdown_nas_duas_metades() -> None:
    """As duas chaves, e **nenhuma das duas basta sozinha** — e por isso ambas sao afirmadas.

    `extend-exclude` tira os `.md` da **varredura**: medido, `ruff format --check .` passou de 295
    arquivos com um a reformatar para 261 com zero. Mas um arquivo passado **explicitamente** na
    linha de comando continuava a ser formatado — e caminho explicito e exactamente o que um editor
    com formatacao ao gravar passa. `force-exclude` e a metade que fecha esse caso.
    """
    ruff = pyproject()["tool"]["ruff"]

    assert "*.md" in ruff.get("extend-exclude", []), (
        "`extend-exclude` deixou de excluir `*.md`: qualquer `ruff format` de escopo largo volta a "
        "reescrever os blocos de codigo dos `.md`"
    )
    assert ruff.get("force-exclude") is True, (
        "`force-exclude` saiu: a exclusao deixa de valer para caminho passado explicitamente, que "
        "e o caso do editor com formatacao ao gravar"
    )


def test_a_arvore_inteira_nao_tem_arquivo_a_reformatar() -> None:
    """`ruff format --check .` sobre a raiz, que e o escopo que um editor usa.

    **Este teste mede o DESFECHO e o de cima mede a CAUSA.** Sem ele, alguem que trocasse `*.md`
    por um padrao que nao casa nada passaria na assercao de configuracao.
    """
    result = run_module("ruff", "format", "--check", ".")
    assert result.returncode == 0, (
        # **A mensagem NAO atribui a causa, e a primeira versao atribuia.** Ela dizia *"a exclusao de
        # `.md` pode ter deixado de valer"* — mas este alvo reprova por QUALQUER arquivo da arvore,
        # e culpar a exclusao mandaria a proxima sessao ao lugar errado. Quem nomeia o arquivo e o
        # `ruff`; o teste so mostra o que ele disse.
        "`ruff format --check .` reprovou. Se o arquivo listado abaixo for um `.md`, a exclusao de "
        "`*.md` deixou de valer; se for outro, e formatacao normal e nao tem relacao com este "
        f"portao:\n{result.stdout}\n{result.stderr}"
    )


def test_o_ruff_recusa_formatar_um_markdown_passado_explicitamente(tmp_path: Path) -> None:
    """A metade do `force-exclude`, com o trecho que causou o defeito.

    **Par causal:** o mesmo arquivo, o mesmo `ruff`, e a unica diferenca e a configuracao. Com a do
    repositorio ele e recusado; com a exclusao retirada ele e reescrito.

    A assercao da planta e o que impede este portao de ficar inerte: se um dia o `ruff` deixar de
    formatar `.md`, a primeira assercao passa por motivo nenhum, e so a segunda acusa.
    """
    alvo = tmp_path / "sonda.md"
    alvo.write_text(_TRECHO_HOSTIL, encoding="utf-8")

    com_a_config = run_module("ruff", "format", "--check", str(alvo))
    assert com_a_config.returncode == 0, (
        f"o `ruff` quis reformatar um `.md` com a configuracao do repositorio:\n"
        f"{com_a_config.stdout}"
    )

    sem_exclusao = tmp_path / "pyproject.toml"
    sem_exclusao.write_text(
        '[tool.ruff]\ntarget-version = "py312"\nline-length = 100\n', encoding="utf-8"
    )
    plantado = run_module(
        "ruff", "format", "--check", "--config", str(sem_exclusao), str(alvo), cwd=tmp_path
    )

    assert plantado.returncode != 0, (
        "**a planta nao separou**: sem a exclusao o `ruff` tambem deixou o arquivo em paz, entao a "
        "assercao de cima nao prova que a exclusao e o que o protege. Talvez o `ruff` tenha deixado "
        f"de formatar `.md`, e ai este portao inteiro deixou de medir algo:\n{plantado.stdout}\n"
        f"{plantado.stderr}"
    )

"""Paridade entre o override de estrito e o que o mypy instalado considera strict.

Por que este arquivo existe. O `pyproject.toml` não pode usar `strict = true`
num `[[tool.mypy.overrides]]`: medido no mypy 2.3.0, o atalho vaza para a
configuração global e o escopo por pacote que a especificação pede desaparece.
A saída foi expandir nos flags componentes — e expandir cria uma dívida:
a lista fica congelada no dia em que foi escrita, enquanto o conjunto que o
mypy chama de `strict` muda entre versões.

Uma versão nova que acrescente um flag ao `strict` tornaria os três pacotes
*menos* estritos do que a especificação pede, silenciosamente. Este teste é
o alarme: deriva a lista do mypy instalado e compara com a declarada.

Mesmo princípio dos portões mutados anteriores — uma
configuração só está verificada quando existe um caminho que a reprova.
"""

import re

import pytest

from tests.conftest import pyproject, run_module

# Flags do `--strict` que recebem argumento em vez de serem toggles booleanos.
# `--enable-error-code`/`--disable-error-code` aparecem perto do bloco do
# `--strict` no help e não são ativáveis por `flag = true` num override.
NON_BOOLEAN_STRICT_FLAGS = frozenset({"enable_error_code", "disable_error_code"})

# Flags que a configuração base do projeto liga para a árvore INTEIRA. Fazem
# parte do `strict` do mypy, então contam como satisfeitos — só não estão no
# override porque valem mais amplamente.
#
# ISENÇÃO DE ESCOPO, NÃO DE APLICAÇÃO. Ambos estão LIGADOS; valem para a árvore
# toda, o que é mais estrito do que a especificação exige. A isenção é de constar
# no override, não de vigorar.
#
#   warn_redundant_casts  Cast redundante é defeito em qualquer pacote. Escopar
#                         aos três da especificação deixaria os outros dez acumulá-los.
#   warn_unused_ignores   `# type: ignore` obsoleto silencia uma verificação que
#                         já não dispara, escondendo o PRÓXIMO erro real na
#                         linha. Restringir seria escolher onde tolerar cegueira.
#
# REVISÃO. O gatilho por evento já é mecanizado e é o que protege de fato:
# test_installed_mypy_matches_the_pinned_version reprova ao bumpar mypy sem
# re-travar, e test_override_covers_every_strict_flag_mypy_declares reprova se o
# conjunto `strict` mudar. Revisão por calendário: 2027-01-26 — deliberadamente
# NÃO implementada como teste temporizado, que quebraria o CI numa terça sem
# defeito nenhum e viraria incidente em vez de nota.
GLOBALLY_ENABLED = frozenset({"warn_redundant_casts", "warn_unused_ignores"})
GLOBALLY_ENABLED_REVIEW_DATE = "2027-01-26"

STRICT_PACKAGES = ("refino.domain", "refino.metrics", "refino.correction")


def installed_mypy_strict_flags() -> frozenset[str]:
    """Os flags que o mypy INSTALADO enumera sob `--strict`.

    Derivado de `mypy --help`, que lista o conjunto explicitamente. É a única
    fonte estável entre versões: `mypy.main` monta a lista dentro de uma
    função, sem expor um símbolo importável, e o pacote é distribuído
    compilado — `inspect.getsource` falha.
    """
    result = run_module("mypy", "--help")
    assert result.returncode == 0, f"mypy --help falhou:\n{result.stderr}"

    match = re.search(
        r"--strict\s+Strict mode; enables the following flags:(.*?)\n\s{2}--[a-z]",
        result.stdout,
        re.DOTALL,
    )
    assert match, "não encontrei a enumeração do --strict no help do mypy"

    # O help quebra nomes de flag no meio da palavra depois de um hífen
    # ("--disallow-subclassing-\nany"), então é preciso costurar antes de casar.
    body = re.sub(r"-\s+", "-", re.sub(r"\s+", " ", match.group(1)))
    flags = {flag[2:].replace("-", "_") for flag in re.findall(r"--[a-z0-9-]+", body)}
    return frozenset(flags - NON_BOOLEAN_STRICT_FLAGS)


def declared_strict_override() -> dict[str, object]:
    """O override que aplica estrito aos três pacotes da especificação."""
    for override in pyproject()["tool"]["mypy"].get("overrides", []):
        modules = {module.removesuffix(".*") for module in override["module"]}
        if set(STRICT_PACKAGES).issubset(modules):
            return dict(override)
    raise AssertionError(f"nenhum override cobre os três pacotes {STRICT_PACKAGES}")


def test_mypy_version_is_pinned_exactly() -> None:
    """A versão do mypy é exata, não uma faixa.

    Sem pino exato, o conjunto `strict` pode mudar numa resolução de
    dependências e a paridade abaixo passa a comparar contra um alvo móvel.
    """
    dev = pyproject()["project"]["optional-dependencies"]["dev"]
    entries = [entry for entry in dev if entry.lower().startswith("mypy")]
    assert len(entries) == 1, f"esperava um pino de mypy, achei {entries}"
    assert re.fullmatch(r"mypy==\d+\.\d+(\.\d+)?", entries[0]), (
        f"mypy deveria ser pinado exato com ==, veio {entries[0]!r}"
    )


def test_installed_mypy_matches_the_pinned_version() -> None:
    """O mypy instalado é o que o pyproject pina."""
    dev = pyproject()["project"]["optional-dependencies"]["dev"]
    pinned = next(entry for entry in dev if entry.lower().startswith("mypy")).split("==")[1]

    result = run_module("mypy", "--version")
    assert result.returncode == 0, result.stderr
    assert pinned in result.stdout, (
        f"pyproject pina {pinned}, instalado é {result.stdout.strip()!r}"
    )


def test_the_help_enumeration_is_parseable() -> None:
    """A derivação funciona — se o formato do help mudar, falha aqui.

    Separado dos testes de paridade para que uma mudança no formato do help do
    mypy produza uma falha diagnosticável, em vez de um conjunto vazio fazendo
    a paridade passar por vacuidade.
    """
    flags = installed_mypy_strict_flags()
    assert len(flags) >= 10, f"conjunto strict suspeito de raso: {sorted(flags)}"
    assert "disallow_untyped_defs" in flags
    assert "disallow_any_generics" in flags


def test_override_covers_every_strict_flag_mypy_declares() -> None:
    """Nenhum flag do `strict` do mypy fica de fora dos três pacotes da especificação.

    É a metade que pega a deriva de versão: se um mypy novo acrescentar um flag
    ao `strict`, ele aparece aqui como ausente e alguém decide se entra no
    override. Sem este teste, os pacotes ficariam menos estritos do que a
    especificação pede e nada acusaria.
    """
    expected = installed_mypy_strict_flags()
    override = declared_strict_override()
    enabled = {flag for flag, value in override.items() if value is True}

    missing = sorted(expected - enabled - GLOBALLY_ENABLED)
    assert not missing, (
        f"o mypy {_version()} considera strict flags que o override não liga: {missing}\n"
        "Acrescente-os ao [[tool.mypy.overrides]] dos três pacotes da §7, ou "
        "à configuração base se devem valer para a árvore toda — e atualize "
        "GLOBALLY_ENABLED neste arquivo se for o caso."
    )


def test_override_declares_nothing_that_is_not_strict() -> None:
    """O override não inventa estritez além do `strict` do mypy.

    Divergência nos dois sentidos importa. Um flag a mais faz os três pacotes
    serem mais estritos do que a especificação pede, o que é uma decisão legítima mas tem
    de ser deliberada — não resíduo de quem copiou uma lista de outro projeto.
    """
    expected = installed_mypy_strict_flags()
    override = declared_strict_override()
    boolean_flags = {flag for flag, value in override.items() if isinstance(value, bool)}

    extra = sorted(boolean_flags - expected)
    assert not extra, (
        f"o override liga flags que o mypy {_version()} NÃO conta como strict: {extra}\n"
        "Se for deliberado, documente no pyproject por que este projeto é mais "
        "estrito que o strict do mypy."
    )


def test_globally_enabled_flags_are_actually_enabled_globally() -> None:
    """O que a lista de exceções isenta está de fato ligado na base.

    Sem isto, acrescentar um nome a GLOBALLY_ENABLED silenciaria o teste de
    cobertura acima sem ligar nada — a lista de exceções viraria a saída de
    emergência que esvazia a verificação.
    """
    base = pyproject()["tool"]["mypy"]
    for flag in sorted(GLOBALLY_ENABLED):
        assert base.get(flag) is True, (
            f"{flag!r} está em GLOBALLY_ENABLED mas não está ligado em [tool.mypy]; "
            "a isenção não corresponde à configuração real"
        )


def test_strict_shortcut_still_absent_from_overrides() -> None:
    """O atalho continua fora — a razão de tudo isto existir."""
    offenders = [
        override["module"]
        for override in pyproject()["tool"]["mypy"].get("overrides", [])
        if "strict" in override
    ]
    assert not offenders, (
        "`strict` reapareceu num override; no mypy 2.3.0 vaza para a configuração "
        f"global e destrói o escopo por pacote. Overrides: {offenders}"
    )


@pytest.mark.parametrize("package", STRICT_PACKAGES)
def test_the_three_packages_are_in_the_override(package: str) -> None:
    """Os três pacotes que a especificação nomeia estão cobertos."""
    modules = {module.removesuffix(".*") for module in declared_strict_override()["module"]}  # type: ignore[attr-defined]
    assert package in modules


def _version() -> str:
    """Versão do mypy instalado, para mensagens de erro."""
    return run_module("mypy", "--version").stdout.strip() or "desconhecida"

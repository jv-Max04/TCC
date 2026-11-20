"""Os ganchos da guarda de `skip` declarado.

Separado de `tests/conftest.py` por um motivo: um
`conftest` só alcança o que está abaixo dele, então a guarda **não poderia ser
testada** com um arquivo temporário fora de `tests/`. Como plugin, o teste que
planta a violação a carrega com `-p tests.skip_guard` e afirma a reprovação. Guarda
cuja reprovação ninguém verificou é indistinguível de guarda ausente.

`conftest.py` registra estes ganchos por importação; a política — quais motivos são
declarados e por quê — mora em `tests/skip_policy.py`.
"""

import pytest

from tests.skip_policy import undeclared_reason

_observed_skips: list[tuple[str, str]] = []


def _skip_reason(report: pytest.TestReport | pytest.CollectReport) -> str:
    """Motivo do `skip`, das duas formas que o pytest o representa.

    `pytest.skip()` em runtime dá a tripla `(arquivo, linha, "Skipped: motivo")`;
    `skipif` na coleta dá o `reason` do marcador. Ler só uma das duas deixaria
    metade dos `skip` sem motivo, e `skip` sem motivo passa por qualquer padrão que
    não seja ancorado — foi o que motivou os quatro casos negativos do teste de
    classificação.
    """
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).removeprefix("Skipped: ")
    return str(longrepr) if longrepr else ""


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Coleta `skip`, e **não** `xfail`.

    O pytest reporta teste `xfail` com `report.skipped` verdadeiro, e sem a distinção
    o gancho pedia regime declarado para ele - o que é a pergunta errada: `xfail` não
    é `skip`. Forçá-lo no registro faz a direção estática da guarda reprovar, porque o
    sítio declarado não emite `skip` nenhum.

    A exclusão só é honesta com a compensação que veio junto:
    `non_strict_xfail_sites` exige `strict=True` em todo `xfail`. Estrito é o que
    torna `xfail` não silencioso - `XPASS` reprova -, e é por isso que ele não precisa
    do registro de `skip`, cuja razão de existir é justamente o silêncio.
    """
    if report.skipped and not hasattr(report, "wasxfail"):
        _observed_skips.append((report.nodeid, _skip_reason(report)))


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.skipped:  # pragma: no cover - módulo inteiro pulado; não ocorre hoje
        _observed_skips.append((report.nodeid, _skip_reason(report)))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Reprova a sessão quando algum `skip` não tem regime declarado.

    Reprovar num gancho e não num teste é deliberado: um teste só vê os `skip` que
    aconteceram antes dele, e a ordem de execução não é contrato. O gancho vê a
    sessão inteira.
    """
    if getattr(session.config, "workerinput", None) is not None:  # pragma: no cover
        return  # sob xdist, quem julga é o processo que agrega

    nao_declarados = [
        (nodeid, motivo) for nodeid, motivo in _observed_skips if undeclared_reason(motivo)
    ]
    if not nao_declarados:
        return

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:  # pragma: no branch - sempre presente fora de embutimento
        reporter.write_sep("=", "skip sem regime declarado (AR13)", red=True)
        for nodeid, motivo in nao_declarados:
            reporter.write_line(f"  {nodeid}\n    motivo: {motivo!r}")
        reporter.write_line(
            "  Declare o regime em tests/skip_policy.py, ou conserte o teste. "
            "Um skip que ninguém declarou se esconde entre os esperados."
        )
    session.exitstatus = pytest.ExitCode.TESTS_FAILED

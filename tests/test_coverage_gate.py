"""Prova que o portão de cobertura do critério está ativo.

Por que este arquivo existe. No começo `src/refino/` tinha zero statements, e
`coverage.py` reporta 0/0 como 100%. O build fica verde — mas *vacuamente*:
um `--cov-fail-under` completamente ausente produziria exatamente o mesmo
verde. O critério exige "falhando o build abaixo disso", que é uma
propriedade que só se demonstra observando uma reprovação de verdade.

Então o teste planta um módulo com statements não cobertos, roda `pytest` num
subprocesso e afirma que o portão reprova.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO, pyproject

# 6 statements, nenhum coberto. Bem acima do que 85% toleraria sobre a sonda.
PLANTED_SOURCE = '''"""Módulo não coberto, plantado para provar que o portão reprova."""


def uncovered(value: int) -> int:
    if value > 0:
        return value * 2
    if value == 0:
        return 0
    return -value
'''

# 2 statements, ambos cobertos pelo teste trivial. É a LINHA DE BASE, e sem ela o
# negativo não consegue atribuir a reprovação à planta.
COVERED_SOURCE = '''"""Módulo coberto: dá a linha de base de 100% da árvore-sonda."""


def double(value: int) -> int:
    return value * 2
'''

TRIVIAL_TEST = '''"""Teste que cobre `pkg.covered` e NÃO toca o módulo plantado."""

import sys

sys.path.insert(0, r"{root}")

from pkg.covered import double


def test_double() -> None:
    assert double(2) == 4
'''


def _run_pytest(*args: str) -> subprocess.CompletedProcess[str]:
    """Roda pytest num subprocesso, sem herdar addopts do processo atual."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _probe_tree(tmp_path: Path, *, with_plant: bool) -> tuple[Path, Path, Path]:
    """Árvore mínima: um pacote com um módulo COBERTO e um teste que o cobre.

    Com `with_plant`, acrescenta um módulo de 6 statements que nada exercita.
    Nada é escrito em `src/refino/` — a versão anterior deste teste plantava na
    árvore real, o que deixava lixo se o processo morresse entre o `write` e o
    `finally`.
    """
    package = tmp_path / "pkg"
    tests_dir = tmp_path / "t"
    package.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "covered.py").write_text(COVERED_SOURCE, encoding="utf-8")
    (tests_dir / "test_covered.py").write_text(TRIVIAL_TEST.format(root=tmp_path), encoding="utf-8")
    if with_plant:
        (package / "_gate_probe.py").write_text(PLANTED_SOURCE, encoding="utf-8")
    return package, tests_dir, tmp_path


def _run_gate(package: Path, tests_dir: Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Roda o portão sobre a árvore-sonda, fora do `pyproject.toml` do projeto.

    `cwd` é o diretório temporário de propósito: rodando de dentro do repositório,
    o pytest herdaria os `addopts` — inclusive um segundo `--cov` — e a medição
    deixaria de ser a que este teste declara.

    **E o ambiente vai sem o arquivo de cobertura de quem chama**, pelo mesmo motivo.
    Com `COVERAGE_FILE` herdado, a sonda — fora do `pyproject.toml`, portanto sem
    `branch = true` — gravava dados SEM ramos no arquivo do processo que a corria; no
    fim, esse processo não conseguia juntar os seus dados COM ramos a ele, e a
    cobertura da seleção inteira ficava órfã. Medido em 01/10/2026 na régua de quatro
    seleções de `docs/linha-de-base-windows.md`, que dá um `COVERAGE_FILE` a cada uma:
    o `.cov.s2` final continha os 3 arquivos da sonda e nenhum de `src/refino`.
    """
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            str(tests_dir),
            f"--cov={package}",
            "--cov-fail-under=85",
            "--no-header",
            "-q",
        ],
        cwd=cwd,
        env=_ambiente_sem_cobertura_herdada(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _ambiente_sem_cobertura_herdada() -> dict[str, str]:
    """O ambiente de quem chama, menos o que aponta a cobertura DELE (ver `_run_gate`)."""
    return {
        k: v
        for k, v in os.environ.items()
        if k != "COVERAGE_FILE" and not k.startswith("COV_CORE_")
    }


def test_a_sonda_nao_grava_no_arquivo_de_cobertura_de_quem_a_corre(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O arquivo de cobertura de quem corre a suíte fica intocado pela sonda do portão.

    O par causal está na mesma função: a sonda grava o PRÓPRIO arquivo, no `cwd` dela —
    então ela mediu, e não gravou no de fora porque não o herdou.
    """
    alheio = tmp_path / "de-quem-corre" / ".coverage.alheio"
    alheio.parent.mkdir()
    monkeypatch.setenv("COVERAGE_FILE", str(alheio))
    package, tests_dir, cwd = _probe_tree(tmp_path / "sonda", with_plant=False)
    r = _run_gate(package, tests_dir, cwd)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not alheio.exists(), "a sonda gravou no arquivo de cobertura de quem a corre"
    # A âncora vem DEPOIS só para a regressão reprovar pela frase certa — com o ambiente
    # herdado a sonda grava lá fora e não aqui. Na mesma função, ela ainda reprova o caso
    # vácuo, em que a sonda não mediu nada e o arquivo de fora também não existiria.
    assert (cwd / ".coverage").exists(), "âncora: a sonda não gravou cobertura nenhuma"


def test_gate_fails_when_coverage_below_threshold(tmp_path: Path) -> None:
    """Statements não cobertos reprovam o build, e **é a planta que
    causa a reprovação**.

    **A versão anterior deste teste era inerte, na sua forma mais
    irônica.** Ela rodava `pytest tests/_gate_probe --cov=src/refino
    --cov-fail-under=85`: um punhado de testes medido contra o pacote inteiro dá
    **0,00%** de cobertura e reprova o portão **com ou sem o módulo plantado**.
    Medido: sem a planta, `TOTAL 942 942 0%`, `FAIL Required test coverage of 85%
    not reached`. O teste provava "rodar um subconjunto reprova o portão", que é
    verdade e não é o que ele afirmava provar.

    A ironia fecha o círculo: é o exemplo fundador da regra do projeto — *o portão de cobertura
    passando vacuamente com zero statements* — e ele próprio passava vacuamente.

    **O desenho agora é causal:** duas execuções idênticas, diferindo **somente**
    pela presença do módulo não coberto. Sem ele o portão aprova (100%); com ele
    reprova (25%). Um teste que só afirmasse a segunda metade não distinguiria
    "a planta derruba a cobertura" de "esta invocação sempre reprova".
    """
    package, tests_dir, cwd = _probe_tree(tmp_path / "limpo", with_plant=False)
    baseline = _run_gate(package, tests_dir, cwd)
    assert baseline.returncode == 0, (
        "a árvore-sonda SEM planta já reprova o portão, então o negativo não "
        f"conseguiria atribuir a reprovação à planta.\n{baseline.stdout}"
    )
    assert "reached" in baseline.stdout, baseline.stdout

    package, tests_dir, cwd = _probe_tree(tmp_path / "plantado", with_plant=True)
    planted = _run_gate(package, tests_dir, cwd)
    assert planted.returncode != 0, (
        "O portão de cobertura NÃO reprovou com um módulo não coberto plantado. "
        f"O [CA-0.4] não está sendo cumprido.\nstdout:\n{planted.stdout}"
    )
    combined = planted.stdout + planted.stderr
    assert "Required test coverage of 85% not reached" in combined, (
        f"Saiu não-zero, mas sem a mensagem do portão — a causa pode ser outra.\n{combined}"
    )


def test_gate_threshold_is_configured_at_85() -> None:
    """O limiar declarado em pyproject.toml é o que o critério pede."""
    config = pyproject()
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--cov-fail-under=85" in addopts, f"limiar ausente ou diferente em addopts: {addopts!r}"
    assert "--cov=src/refino" in addopts, f"escopo de medição ausente em addopts: {addopts!r}"


@pytest.mark.parametrize(
    "marker",
    ["property", "golden", "integration", "synthetic", "slow"],
)
def test_section_8_1_markers_are_registered(marker: str) -> None:
    """Os marcadores da especificação resolvem sob --strict-markers."""
    config = pyproject()
    declared = config["tool"]["pytest"]["ini_options"]["markers"]
    names = {entry.split(":", 1)[0].strip() for entry in declared}
    assert marker in names, f"marcador {marker!r} não registrado; declarados: {sorted(names)}"


def test_unknown_marker_is_rejected() -> None:
    """--strict-markers está ativo: marcador desconhecido reprova a coleta.

    Sem isto, um `-m goldne` com erro de digitação selecionaria zero testes e
    reportaria sucesso — a falha silenciosa que --strict-markers previne.
    """
    probe = REPO / "tests" / "_marker_probe.py"
    probe.write_text(
        '"""Sonda temporária."""\n\nimport pytest\n\n\n'
        "@pytest.mark.marcador_que_nao_existe\n"
        "def test_probe() -> None:\n    assert True\n",
        encoding="utf-8",
    )
    try:
        result = _run_pytest(str(probe), "--no-cov", "--no-header")
        assert result.returncode != 0, "marcador desconhecido deveria reprovar sob --strict-markers"
        assert "marcador_que_nao_existe" in (result.stdout + result.stderr)
    finally:
        probe.unlink(missing_ok=True)

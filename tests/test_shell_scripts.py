"""Varredura dos scripts de shell contra a corrida de SIGPIPE.

**O defeito que motivou.** `scripts/check_compose.sh` fazia
`docker compose config --services | grep -qx "$service"` sob `set -o pipefail`.
O `grep -q` sai no primeiro casamento e fecha o cano; se o escritor ainda tinha o
que escrever, ele morre com SIGPIPE (status 141), e o `pipefail` transforma isso
em falha da pipeline **mesmo com o grep tendo casado**.

É corrida, então passa quase sempre. Passou em toda execução local, passou no job
dedicado do compose na mesma execução do CI, e reprovou no job do runner do R —
onde a máquina estava ocupada construindo imagem e rodando contêiner.

**Por que teste estático e não teste de execução.** Reproduzir a corrida exige
carga, e um teste que só reprova sob carga é um teste que reprova de forma
intermitente — pior que reprovar sempre, pelo argumento sobre
determinismo. O padrão textual, esse é decidível.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = sorted((REPO / "scripts").glob("*.sh"))

# Consumidores que encerram a leitura ANTES do fim da entrada. São eles que
# entregam SIGPIPE ao escritor. `grep` sem `-q`/`-l`/`-m` lê tudo e não entra.
_EARLY_EXIT_CONSUMERS = re.compile(
    r"\|\s*(?:grep\s+(?:-[a-zA-Z]*[qlm][a-zA-Z]*\s+)|head\b|"
    r"sed\s+-n\s+['\"][^'\"]*q['\"]|awk\s+['\"][^'\"]*exit)"
)


def _uses_pipefail(text: str) -> bool:
    return bool(re.search(r"set\s+-[a-zA-Z]*o\s+pipefail|set\s+-o\s+pipefail", text))


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_early_exit_consumer_in_a_pipeline_under_pipefail(script: Path) -> None:
    """Nenhum cano cujo leitor sai cedo, em script com `pipefail`.

    A correção não é remover `pipefail` — ele existe para que erro no meio de um
    cano não passe despercebido, que é o defeito oposto e mais comum. A correção
    é não ter o cano: capturar a saída numa variável e casar sobre ela, ou usar o
    `q` do próprio `sed` em vez de `| head -1`.
    """
    text = script.read_text(encoding="utf-8")
    if not _uses_pipefail(text):
        return

    offenders = [
        (number, line.strip())
        for number, line in enumerate(text.splitlines(), 1)
        if _EARLY_EXIT_CONSUMERS.search(line) and not line.lstrip().startswith("#")
    ]
    assert offenders == [], (
        f"{script.name} usa `pipefail` e tem cano cujo leitor sai cedo:\n"
        + "\n".join(f"  linha {n}: {line}" for n, line in offenders)
        + "\nO escritor morre com SIGPIPE e o pipefail reporta falha mesmo no caminho "
        "de sucesso. Capture a saída numa variável e case sobre ela."
    )


def test_the_sweep_would_catch_the_original_defect() -> None:
    """**O teste que prova que a varredura consegue reprovar**.

    Sem isto o padrão poderia estar errado — regex que não casa nada passa em
    tudo, e é indistinguível de árvore limpa. Aqui a linha exata que reprovou no
    CI é replantada e a varredura tem de reconhecê-la.
    """
    original = 'docker compose config --services | grep -qx "$service" \\'
    assert _EARLY_EXIT_CONSUMERS.search(original), (
        "a varredura não reconhece a linha que de fato causou a reprovação"
    )
    assert _uses_pipefail("set -euo pipefail")
    assert _uses_pipefail("set -o pipefail")

    # E o inverso: `grep` que lê a entrada inteira não é o padrão perigoso, e
    # marcá-lo faria a varredura reprovar uso legítimo até alguém desativá-la.
    assert not _EARLY_EXIT_CONSUMERS.search('cat x | grep "FROM" > out')
    assert not _EARLY_EXIT_CONSUMERS.search("sha256sum \"$INPUT\" | cut -d' ' -f1")

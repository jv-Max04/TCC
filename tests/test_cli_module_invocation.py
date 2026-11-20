"""`python -m refino.cli` despacha todo subcomando — defeito real.

**O defeito.** A guarda `if __name__ == "__main__": raise SystemExit(main())` estava no
**meio** de `src/refino/cli.py`, antes de `_main_processar` e `_main_exportar`. Sob
`python -m`, o corpo do módulo executa de cima para baixo: a guarda chamava `main()`
com o módulo pela metade, e `python -m refino.cli processar ...` levantava
`NameError: name '_main_processar' is not defined` — **desde o início**.

**Por que ninguém viu.** Três caminhos chamam
`main` e só um estava quebrado:

| Invocação | Estado | Por quê |
|---|---|---|
| `refino ...` (entrada instalada) | funcionava | importa o módulo **inteiro** antes de chamar |
| `main(argv)` nos testes | funcionava | idem — o import completa antes |
| `python -m refino.cli ...` | **NameError** | executa o corpo do módulo, e a guarda
disparava no meio |

A invocação quebrada é justamente a que os equivalentes
diretos do `make` usam. Nenhum teste a exercitava, então o defeito ficou **silencioso**:
não havia saída errada a notar, porque nada rodava por aquele caminho.

**Este é negativo EXTERNO**: constrói a invocação fora do processo. Foi visto
reprovar com a guarda de volta no meio do arquivo, e passar com ela no fim.
"""

import subprocess
import sys

import pytest

from tests.conftest import REPO

# Argumentos mínimos que alcançam o DESPACHO de cada subcomando. Apontam para
# arquivo inexistente de propósito: o que se afirma é que o despacho acontece, não
# que o processamento termine. Um `--help` não serviria — ele sai no argparse, antes
# do despacho, e é exatamente por isso que o defeito atravessou duas fases.
SUBCOMMANDS: dict[str, list[str]] = {
    "corrigir": [
        "--measurements",
        "/tmp/refino-inexistente.csv",
        "--sample-list",
        "/tmp/refino-inexistente.csv",
        "--mode",
        "pos_hilic",
        "--output",
        "/tmp/refino-saida.csv",
    ],
    "processar": [
        "--measurements",
        "/tmp/refino-inexistente.csv",
        "--sample-list",
        "/tmp/refino-inexistente.csv",
        "--mode",
        "pos_hilic",
        "--output",
        "/tmp/refino-saida.csv",
        "--imputation",
        "none",
        "--sample-normalization",
        "none",
        "--transformation",
        "none",
        "--scaling",
        "none",
        "--rsd-cutoff",
        "0.3",
    ],
    "exportar": [
        "--measurements",
        "/tmp/refino-inexistente.csv",
        "--sample-list",
        "/tmp/refino-inexistente.csv",
        "--mode",
        "pos_hilic",
        "--output",
        "/tmp/refino-saida.csv",
    ],
}


def _run(subcommand: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "refino.cli", subcommand, *SUBCOMMANDS[subcommand]],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=120,
    )


@pytest.mark.parametrize("subcommand", sorted(SUBCOMMANDS))
def test_o_subcomando_despacha_sem_nameerror(subcommand: str) -> None:
    """A asserção é sobre `NameError`, e o alvo é estreito de propósito.

    Não se afirma que o comando tenha sucesso — ele recebe arquivo inexistente e
    **deve** falhar. Afirma-se que ele falha **no lugar certo**: no processamento, e
    não no despacho por símbolo ainda não definido. É a distinção entre reprovar
    pelo motivo certo e reprovar por qualquer motivo.
    """
    result = _run(subcommand)
    combined = result.stdout + result.stderr

    assert "NameError" not in combined, (
        f"`python -m refino.cli {subcommand}` levantou NameError. A guarda "
        f"`__main__` provavelmente voltou para o meio de `cli.py`, antes das funções "
        f"de despacho. Saída:\n{combined[-800:]}"
    )
    assert "not defined" not in combined


@pytest.mark.parametrize("subcommand", sorted(SUBCOMMANDS))
def test_o_subcomando_falha_no_processamento_e_nao_no_despacho(subcommand: str) -> None:
    """A direção que prova que o teste acima não passa vacuamente.

    Se a invocação morresse antes de tocar o arquivo, "sem NameError" seria
    verdadeiro e não significaria nada. Aqui se afirma que a execução chegou até o
    ponto de **procurar o arquivo** — que é depois do despacho.
    """
    result = _run(subcommand)
    combined = result.stdout + result.stderr

    assert result.returncode != 0, "arquivo inexistente tem de falhar"
    assert "refino-inexistente.csv" in combined, (
        "a execução não chegou a mencionar o arquivo de entrada, então não é possível "
        "afirmar que ela passou do despacho"
    )


def test_a_guarda_main_e_a_ultima_coisa_do_modulo() -> None:
    """A causa-raiz, afirmada estruturalmente e não pelo sintoma.

    Os testes acima pegam o defeito **desta** forma; este impede a **classe**. Se
    alguém acrescentar uma função depois da guarda, os dois de cima só reprovam se o
    símbolo novo for usado no despacho — e este reprova de imediato.
    """
    lines = (REPO / "src" / "refino" / "cli.py").read_text(encoding="utf-8").splitlines()
    meaningful = [
        index
        for index, line in enumerate(lines)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    guard = next(
        index for index, line in enumerate(lines) if line.startswith('if __name__ == "__main__"')
    )

    after = [index for index in meaningful if index > guard and not lines[index].startswith(" ")]
    assert not after, (
        "há código de nível de módulo depois da guarda `__main__`: "
        f"{[lines[index] for index in after][:3]}. Sob `python -m`, a guarda dispara "
        "`main()` antes desse código existir."
    )

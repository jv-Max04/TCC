"""O `refino` que a suíte importa é o da árvore que ela está a testar.

## O defeito que este arquivo existe para acusar

O `refino` está instalado no venv como *editable* apontando para o `src` do **clone principal**, e
um worktree não tem venv próprio. Então, de dentro de um worktree, sem `PYTHONPATH`:

    $ python -c "import refino; print(refino.__file__)"
    C:\\rq\\src\\refino\\__init__.py                                     # o clone PRINCIPAL

**O desfecho normal desse defeito é VERDE.** Em 02/09/2026 ele levantou apenas porque o clone
principal estava atrás e faltava-lhe um símbolo, e o import explodiu. Com os dois em dia — que é o
caso comum, porque o worktree costuma sair de `main` — a suíte corre sobre a versão **antiga** de
cada arquivo editado e **passa**. Quem a corre lê *"221 passed"* como prova de uma mudança que ela
nunca viu.

É a família do instrumento que lê o referente errado, com um alcance próprio: não é a instância
que o instrumento lê **dentro** do clone, nem o clone estar atrás do **remoto**. Esta é sobre o
instrumento ler outra **árvore de trabalho na mesma máquina**.

## Por que ACUSAR e não CONSERTAR

O item 64 põe três saídas na mesa, e a segunda — um `conftest` que insere o `src` local no
`sys.path` — resolve sozinha, sem ninguém ter de lembrar de nada. Ela foi **recusada**, e a razão é
o alcance: mexer no `sys.path` muda **como o projeto inteiro resolve o próprio pacote**, inclusive
no clone principal onde o *editable* já resolve, e a divergência entre o que o teste importa e o
que o executável congelado embala é exactamente a classe do problema.

> **Este arquivo não muda resolução de pacote nenhuma. Ele compara e reprova.**

O custo declarado da escolha: continua a ser preciso passar `PYTHONPATH` para **rodar** de um
worktree. O que muda é que esquecer deixou de ser silencioso.

## Este teste NÃO entra na linha de base

Reprovação aqui **não** é dívida declarada: é o sinal a funcionar. Quem a vir num worktree passa a
`PYTHONPATH=<worktree>/src` e ela some. Acrescentá-la à linha de base seria desligar o mecanismo no
dia em que ele acusou pela primeira vez.
"""

from __future__ import annotations

import types
from pathlib import Path

import refino

RAIZ_DOS_TESTES = Path(__file__).resolve().parents[1]


def arvore_do_pacote(modulo: types.ModuleType) -> Path:
    """De onde este módulo foi de facto carregado — o `src` que o contém.

    **Recebe o módulo em vez de o importar**, e é isso que torna o par causal possível: o negativo
    passa um duplo cujo `__file__` aponta para outra árvore, e a asserção corre sobre ele sem que
    nada no processo tenha de ser remendado. Um teste que importasse `refino` aqui dentro só
    conseguiria exercitar a direção que já é verdadeira.
    """
    arquivo = getattr(modulo, "__file__", None)
    if arquivo is None:  # pragma: no cover - pacote sem `__file__` é namespace, e `refino` não é
        raise AssertionError(f"{modulo!r} não tem `__file__`: não dá para dizer de que árvore veio")
    return Path(arquivo).resolve().parents[1]


def test_o_refino_importado_e_o_da_arvore_destes_testes() -> None:
    """A direção verdadeira, e a mensagem nomeia as DUAS árvores.

    **A mensagem carrega os dois caminhos de propósito.** Quem vir esta reprovação está, por
    construção, num worktree — e o que ele precisa de saber não é *"falhou"*, é **qual árvore a
    suíte leu** e **qual ele está a editar**. Sem os dois, o gesto seguinte é procurar um defeito
    no código que a suíte nem sequer abriu.
    """
    daqui = (RAIZ_DOS_TESTES / "src").resolve()
    de_onde = arvore_do_pacote(refino)

    # **Âncora de não-vacuidade, e ela é a metade barata.** Se `src/` não existisse — nome
    # trocado, layout mudado —, `daqui` seria um caminho que não resolve para nada e a igualdade
    # poderia passar a comparar duas ausências.
    assert daqui.is_dir(), f"âncora: `{daqui}` não é diretório, e a comparação perde o referente"

    assert de_onde == daqui, (
        f"a suíte importou o `refino` de `{de_onde}`, e a árvore que se edita é `{daqui}`.\n"
        f"Num worktree isto acontece sem `PYTHONPATH`: o *editable* do venv aponta para o clone "
        f"principal. Rode com `PYTHONPATH={daqui}`.\n"
        f"NÃO acrescente esta reprovação à linha de base: ela é o sinal, não é dívida."
    )


def test_a_comparacao_acusa_quando_a_arvore_diverge(tmp_path: Path) -> None:
    """O par causal, em processo — a planta é argumento da função sob teste.

    **É a linha de cima da taxonomia de negativo plantado**, a que *não pode ficar inerte*: a
    entrada alimenta a asserção directamente, e não há subprocesso, arnês nem invocação externa
    entre a planta e o desfecho que possa deixar de a receber.

    **A planta cai sobre dado em que as duas coisas DIFEREM**, e a âncora abaixo mede-o em
    vez de o supor: `tmp_path` é de outra árvore que a do repositório, e a asserção confirma-o
    antes de comparar. Sem ela, um `tmp_path` que por acidente caísse dentro do repositório tornaria
    este teste vácuo — ele passaria a afirmar que dois caminhos iguais são iguais.
    """
    outra = tmp_path / "src" / "refino"
    outra.mkdir(parents=True)
    (outra / "__init__.py").write_text("", encoding="utf-8")

    daqui = (RAIZ_DOS_TESTES / "src").resolve()
    assert not str(outra).startswith(str(daqui)), (
        "âncora: a planta caiu DENTRO da árvore real, e o teste seria vácuo"
    )

    falso = types.ModuleType("refino")
    falso.__file__ = str(outra / "__init__.py")

    assert arvore_do_pacote(falso) != daqui, (
        "a comparação não separou duas árvores diferentes: ela não consegue acusar o defeito que "
        "este arquivo existe para acusar"
    )
    assert arvore_do_pacote(falso) == (tmp_path / "src").resolve()

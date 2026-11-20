"""Teste que deixa de existir sem mudar a contagem de `passed`.

**O defeito que motivou, e ele é meu.** Ao inserir um teste em
`test_declared_artifacts.py`, a edição substituiu a **linha de definição** do teste
vizinho em vez de inserir antes dela. As duas funções fundiram: as três asserções
do teste absorvido passaram a rodar **dentro** do outro, e a docstring dele virou
uma expressão de string solta no meio do corpo. A suíte continuou verde, as
asserções continuaram executando, e **um teste deixou de existir como teste
nomeado**. Nada na saída mudou.

É um defeito parecido, numa forma nova. Antes o teste estava sendo **pulado** em silêncio;
aqui ele foi
**absorvido** — e absorção é pior de achar, porque `skipped` ao menos aparece na
linha de resumo e um teste fundido não deixa vestígio nenhum no agregado.

**O que este módulo verifica, e por que é a impressão digital e não a contagem.**
Nenhuma função de teste pode conter expressão de **string solta** — `ast.Expr`
envolvendo `ast.Constant` de `str` — em posição diferente da primeira do corpo. Só
a primeira é docstring; qualquer outra é código morto, e é exatamente o rastro que
a fusão de duas funções deixa. String solta como statement não tem uso legítimo em
Python: ela é avaliada e descartada.

**Nenhuma regra do `ruff` pega isto, e foi medido.** `B018` (*useless expression*)
existe e está selecionado — o projeto usa `select = [..., "B", ...]` —, mas ele
**ignora** expressões de string de propósito, para não acusar docstring. Rodado
`--select ALL` sobre o caso plantado: nada além de `INP001`, `D100`, `CPY001` e
`S101`, nenhum deles relacionado. Por isso o mecanismo é próprio.

**A contagem declarada por arquivo foi avaliada e REJEITADA, e o motivo é medido.**
A proposta era declarar quantos testes cada arquivo tem e conferir contra o
coletado, como a paridade dos treze flags do `strict`. Ela pegaria as três
formas — fusão, remoção acidental e renomeação silenciosa —, e o custo é o problema:

| Quantidade | Valor |
|---|---|
| Arquivos de teste hoje | **56** |
| Toques em arquivo de teste no histórico | **120** |

Seriam 56 números mantidos à mão, com uma edição por toque. E a diferença
decisiva contra a analogia: **os treze flags têm fonte independente** — o
`mypy` instalado os declara, e a divergência significa que um dos dois mudou. Uma
contagem de testes **não tem fonte independente**: o número *é* o que escrevemos,
então a verificação é um retrato, não uma paridade. O modo de falha previsível é
atualizar o número sem olhar, o que é a pergunta respondida na direção
errada — a métrica pararia de medir sem que ninguém notasse, e com 56 sítios para
isso acontecer.

**ESCOPO NEGATIVO, porque a leitura natural sugeriria mais.** Este módulo pega
**fusão**. Não pega remoção acidental nem renomeação silenciosa: as duas mudam o
conjunto de nomes coletados sem deixar string órfã, e cobri-las exigiria o retrato
que o parágrafo acima rejeita. Fica declarado como descoberto em vez de suposto
coberto — remoção intencional é legítima (a regra exige que teste `scaffold` seja
**removido** por fase posterior), então um portão sobre desaparecimento precisaria
de um canal de intenção que não existe hoje.
"""

import ast
from pathlib import Path

from tests.conftest import REPO

TESTS = REPO / "tests"


def _sources() -> list[Path]:
    return sorted(TESTS.rglob("*.py"))


def orphan_string_statements(source: str) -> list[tuple[str, int]]:
    """Pares `(nome da função, linha)` de string solta fora da posição de docstring.

    Função e não corpo de teste pelo mesmo motivo: o negativo
    plantado chama **esta** função, o mesmo caminho que o portão percorre.
    """
    found: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for position, statement in enumerate(node.body):
            if position == 0:
                continue
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                found.append((node.name, statement.lineno))
    return found


# ---------------------------------------------------------------------------
# O portão
# ---------------------------------------------------------------------------


def test_nenhuma_funcao_de_teste_tem_string_orfa() -> None:
    """O portão. String solta no meio de uma função é rastro de fusão."""
    offenders = {
        str(path.relative_to(REPO)): orphan_string_statements(path.read_text(encoding="utf-8"))
        for path in _sources()
    }
    dirty = {path: hits for path, hits in offenders.items() if hits}
    assert not dirty, (
        f"strings soltas dentro de função de teste: {dirty}. "
        "Rastro de duas funções fundidas — um teste deixou de existir como teste "
        "nomeado, sem mudar a contagem de `passed`."
    )


def test_todo_arquivo_de_teste_coleta_ao_menos_um_teste() -> None:
    """Arquivo de teste sem nenhuma função `test_` é o caso extremo da fusão.

    Barato e disjunto do portão acima: fusão **dentro** de um arquivo deixa string
    órfã, fusão que esvazia um arquivo não deixa nada. `skip_policy.py` e
    `conftest.py` não são arquivos de teste e estão fora por nome.
    """
    empty = [
        str(path.relative_to(REPO))
        for path in _sources()
        if path.name.startswith("test_")
        and not any(
            isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    ]
    assert not empty, f"arquivos `test_*.py` que não coletam teste nenhum: {empty}"


# ---------------------------------------------------------------------------
# O teste que prova que o portão consegue reprovar
# ---------------------------------------------------------------------------


def test_o_portao_e_um_par_causal() -> None:
    """**As duas direções, e a planta é argumento da função sob teste.**

    Pelo critério de triagem este negativo é **em processo**: a fonte
    plantada alimenta a asserção diretamente, sem subprocesso nem arquivo
    intermediário, então não há caminho por onde ele fique inerte.

    A planta reproduz a fusão real, e não uma string solta qualquer: docstring
    legítima na primeira posição, asserções, e a docstring órfã do teste absorvido
    no meio — que é exatamente a forma que o commit defeituoso tinha.
    """
    fundido = (
        "def test_a() -> None:\n"
        '    """Docstring legítima."""\n'
        "    assert True\n"
        '    """Docstring órfã do teste absorvido."""\n'
        "    assert True\n"
    )
    limpo = (
        "def test_a() -> None:\n"
        '    """Docstring legítima."""\n'
        "    assert True\n"
        "\n"
        "def test_b() -> None:\n"
        '    """Docstring do segundo teste, no lugar certo."""\n'
        "    assert True\n"
    )

    assert orphan_string_statements(fundido) == [("test_a", 4)], (
        "violação plantada não foi detectada: o portão está inerte"
    )
    assert orphan_string_statements(limpo) == [], (
        "o controle acusou sem planta: o portão sempre reprova, o defeito oposto"
    )


def test_a_docstring_na_primeira_posicao_nunca_acusa() -> None:
    """A pré-condição do controle, isolada — senão o portão acusaria todo arquivo."""
    assert orphan_string_statements('def f() -> None:\n    """Só docstring."""\n') == []
    assert orphan_string_statements("def f() -> None:\n    return None\n") == []


def test_string_como_valor_nao_e_string_solta() -> None:
    """`x = "s"` e `f("s")` são uso legítimo; só `ast.Expr` de string acusa.

    Sem esta distinção o portão acusaria toda função que passa literal a uma
    chamada, que é quase todas — e um portão que sempre reprova é desligado.
    """
    assert orphan_string_statements('def f() -> None:\n    x = "s"\n    print(x)\n') == []
    assert orphan_string_statements('def f() -> None:\n    assert g("s")\n') == []
    assert orphan_string_statements('def f() -> None:\n    return "s"\n') == []

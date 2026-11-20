"""Nenhum módulo define a mesma função ou classe duas vezes no nível de topo — C23.

**A segunda definição vence em silêncio, e o `ruff` não a vê.** O F811 acusa a redefinição de um
nome NÃO usado; quando a primeira definição é chamada dentro do corpo de outra função — o caso de
todo auxiliar de teste —, o uso conta e a redefinição passa. Foi assim em `tests/test_api_rotas.py`:
`_matriz_de_grupos` de ruído, e mil e setecentas linhas abaixo outra `_matriz_de_grupos` em rampa,
e os testes escritos para a primeira correram sobre a segunda, que é a estrutura plantada que a
docstring da primeira diz que eles não podem ter (revisão de 24/10/2026, C23).

**O que fica de fora, e vai declarado:** definição dentro de `if`/`try` no topo — a forma legítima
de duas versões por plataforma — e `@overload`, que existe para repetir o nome.
"""

import ast
from collections import Counter
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
PASTAS = ("src", "tests", "scripts")


def _e_overload(no: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        (isinstance(d, ast.Name) and d.id == "overload")
        or (isinstance(d, ast.Attribute) and d.attr == "overload")
        for d in no.decorator_list
    )


def duplicadas(fonte: str) -> list[str]:
    """Os nomes definidos mais de uma vez como filhos DIRETOS do módulo."""
    nomes = Counter(
        no.name
        for no in ast.parse(fonte).body
        if isinstance(no, ast.ClassDef)
        or (isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _e_overload(no))
    )
    return sorted(nome for nome, vezes in nomes.items() if vezes > 1)


def _modulos() -> list[Path]:
    return sorted(
        caminho
        for pasta in PASTAS
        for caminho in (RAIZ / pasta).rglob("*.py")
        if "__pycache__" not in caminho.parts
    )


def test_ha_modulos_para_varrer() -> None:
    """Âncora de presença antes da asserção de ausência: lista vazia passaria sempre."""
    assert len(_modulos()) > 100, len(_modulos())


def test_nenhum_modulo_define_o_mesmo_nome_duas_vezes() -> None:
    achados = {
        str(caminho.relative_to(RAIZ)): nomes
        for caminho in _modulos()
        if (nomes := duplicadas(caminho.read_text(encoding="utf-8")))
    }
    assert not achados, (
        f"definições repetidas no nível de topo: {achados}. A segunda vence em silêncio, e quem "
        "chama a primeira roda a segunda — renomeie uma das duas pelo que ela é"
    )


@pytest.mark.parametrize(
    ("fonte", "esperado"),
    [
        pytest.param("def f(): ...\ndef g(): ...\ndef f(): ...\n", ["f"], id="funcao_repetida"),
        pytest.param("class A: ...\nclass A: ...\n", ["A"], id="classe_repetida"),
        pytest.param("def f(): ...\ndef g(): f()\n", [], id="sem_repeticao"),
        pytest.param(
            "import sys\nif sys.platform == 'win32':\n    def f(): ...\nelse:\n    def f(): ...\n",
            [],
            id="por_plataforma",
        ),
        pytest.param(
            "from typing import overload\n@overload\ndef f(x: int) -> int: ...\n"
            "@overload\ndef f(x: str) -> str: ...\ndef f(x): return x\n",
            [],
            id="overload",
        ),
    ],
)
def test_o_detector_acusa_a_repeticao_e_so_ela(fonte: str, esperado: list[str]) -> None:
    """O par causal: a planta repetida é acusada, e as três formas legítimas não."""
    assert duplicadas(fonte) == esperado

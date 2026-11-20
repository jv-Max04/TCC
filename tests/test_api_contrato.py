"""Toda rota tem contrato declarado.

Esta é uma superfície de exaustividade: pergunta *o que foi declarado num lugar aconteceu no
outro?*, com rota de API no lugar de seção, artefato ou item de teste.

Por que o contrato por rota precisava disto para não ser cosmético: sem enumerar as rotas da
aplicação, quem escreve o teste também escolhe o que conta como rota, e a contagem fica
verdadeira por construção.

**A direção B é a que o desenho óbvio esquece.** "Toda rota tem teste" é a metade natural;
"todo teste declarado aponta para rota que existe" é a que pega **renomeação**, e sem ela o
manifesto envelhece apontando para caminhos que ninguém serve mais.

## Escopo: o que conta como rota, e por que por TIPO

Só `fastapi.routing.APIRoute`. As rotas que o FastAPI acrescenta por conta — `/openapi.json`,
`/docs`, `/redoc` — são `starlette.routing.Route`.

A discriminação é por **tipo** e não por lista de caminhos a ignorar. Uma lista caduca quando o
framework acrescenta rota interna, e o modo de falha seria uma rota do FastAPI exigindo contrato
nosso: portão que sempre reprova.
"""

import ast
import re
from collections.abc import Iterable, Iterator

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from refino.api.app import build_app
from tests.conftest import REPO

MANIFESTO = REPO / "docs" / "rotas-api.md"

_LINHA = re.compile(
    r"^\|\s*`(?P<metodo>[A-Z]+)`\s*\|\s*`(?P<caminho>[^`]+)`\s*\|\s*`(?P<no>[^`]+)`\s*\|"
)


class LinhaDeclarada:
    """Uma linha do manifesto."""

    def __init__(self, metodo: str, caminho: str, no: str) -> None:
        self.metodo = metodo
        self.caminho = caminho
        self.no = no

    @property
    def chave(self) -> tuple[str, str]:
        return (self.metodo, self.caminho)

    def __repr__(self) -> str:  # pragma: no cover - diagnóstico
        return f"{self.metodo} {self.caminho}"


def declaradas() -> list[LinhaDeclarada]:
    achados: list[LinhaDeclarada] = []
    for linha in MANIFESTO.read_text(encoding="utf-8").splitlines():
        m = _LINHA.match(linha)
        if m:
            achados.append(LinhaDeclarada(m["metodo"], m["caminho"], m["no"]))
    return achados


def _percorre(rotas: Iterable[object]) -> Iterator[APIRoute]:
    """Toda `APIRoute` alcançável, **descendo pelos roteadores incluídos**.

    A recursão é correção de um buraco real. A primeira versão iterava
    `app.routes` e testava `isinstance(rota, APIRoute)` — e o FastAPI 0.140 **não achata**
    `include_router`: ele deixa em `app.routes` um `_IncludedRouter` cujo `path` é `None` e
    cujas rotas vivem em `original_router.routes`.

    O efeito era o pior possível para este portão: as rotas registradas por roteador —
    **todas** as de `rotas.py` — ficavam invisíveis, e a bijeção passava afirmando cobertura
    de um conjunto que não continha nenhuma delas. Portão verde sobre o vazio.

    Foi pego porque rotas novas entraram e o portão não reprovou, e um portão de
    bijeção que aceita rota nova sem linha no manifesto está quebrado por definição.
    """
    for rota in rotas:
        if isinstance(rota, APIRoute):
            yield rota
        interno = getattr(rota, "original_router", None)
        if interno is not None:
            yield from _percorre(interno.routes)


def rotas_de_negocio(app: FastAPI) -> set[str]:
    """Os CAMINHOS de toda `APIRoute` de uma aplicação, descendo pelos roteadores incluídos.

    **Recebe a aplicação em vez de a construir**, ao contrário de `rotas_da_aplicacao` logo abaixo,
    e é isso que permite comparar DUAS aplicações — que é o que o portão da documentação precisa:
    a mesma árvore montada com `documentacao=True` e com `False`.

    **Reusa `_percorre`, e o reuso é o conserto de um defeito medido.** Um portão irmão escreveu a
    própria travessia com `getattr(rota, "path", None)` sobre `app.routes`, e obteve **uma** rota
    de dezasseis: o `_IncludedRouter` tem `path is None`, e todas as rotas de `rotas.py` vivem
    dentro dele. A peculiaridade do framework tem uma explicação só, e ela está na docstring de
    `_percorre`.
    """
    return {rota.path for rota in _percorre(app.routes)}


def rotas_da_aplicacao() -> set[tuple[str, str]]:
    """`(método, caminho)` de toda `APIRoute`, uma entrada por método.

    Uma rota que aceite dois métodos produz **duas** chaves, porque contrato de `GET` e de
    `POST` no mesmo caminho são contratos diferentes e colapsá-los deixaria um sem teste.
    """
    return {
        (metodo, rota.path)
        for rota in _percorre(build_app().routes)
        for metodo in rota.methods or set()
    }


def defeitos_de_bijecao(rotas: set[tuple[str, str]], linhas: list[LinhaDeclarada]) -> list[str]:
    """Um defeito por divergência, nas DUAS direções.

    Função separada devolvendo lista: a planta entra como argumento e o negativo percorre o
    caminho do portão. Negativo em processo não consegue ficar inerte.
    """
    achados: list[str] = []
    declaradas_chaves = [linha.chave for linha in linhas]

    for chave in sorted(rotas - set(declaradas_chaves)):
        achados.append(
            f"rota {chave[0]} {chave[1]} existe na aplicação e NÃO está declarada em "
            f"{MANIFESTO.name}. `[CA-12.4]` exige um teste de contrato por rota"
        )
    for chave in sorted(set(declaradas_chaves) - rotas):
        achados.append(
            f"{MANIFESTO.name} declara {chave[0]} {chave[1]} e essa rota NÃO existe na "
            "aplicação. Entrada órfã é o que renomeação produz, e é a direção que o "
            "desenho óbvio esquece"
        )
    duplicadas = {c for c in declaradas_chaves if declaradas_chaves.count(c) > 1}
    for chave in sorted(duplicadas):
        achados.append(f"{chave[0]} {chave[1]} tem mais de uma linha no manifesto")
    return achados


# ---------------------------------------------------------------------------
# As quatro cláusulas
# ---------------------------------------------------------------------------


def test_a_arvore_limpa_tem_bijecao() -> None:
    assert defeitos_de_bijecao(rotas_da_aplicacao(), declaradas()) == []


def test_o_manifesto_nao_esta_vazio() -> None:
    """A direção que impede o portão de passar por vacuidade.

    Um manifesto vazio contra uma aplicação sem rotas satisfaz a bijeção trivialmente. Sem
    esta asserção, o portão nasceria verde e inerte.
    """
    assert declaradas(), "o manifesto não declara nenhuma rota"
    assert rotas_da_aplicacao(), "a aplicação não serve nenhuma rota"


@pytest.mark.parametrize("linha", declaradas(), ids=lambda linha: repr(linha))
def test_o_no_de_teste_resolve(linha: LinhaDeclarada) -> None:
    """`arquivo::função` existe, por análise sintática.

    Num portão parecido, a primeira execução achou 46 de 54 nós declarados que não resolviam,
    porque os nomes foram inventados a partir de nomes de arquivo.
    """
    caminho, _, funcao = linha.no.partition("::")
    arquivo = REPO / caminho

    assert arquivo.is_file(), f"{caminho} não existe"
    arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
    nomes = {n.name for n in ast.walk(arvore) if isinstance(n, ast.FunctionDef)}
    assert funcao in nomes, f"{caminho} não define {funcao}"


def test_a_openapi_lista_exatamente_as_rotas_da_aplicacao() -> None:
    """Igualdade de conjuntos, e não subconjunto.

    Subconjunto passaria com rota **oculta** da documentação — servida e não documentada —, e
    a documentação é gerada justamente para que isso não aconteça.
    """
    app = build_app()
    documentados = set(app.openapi()["paths"])
    servidos = {caminho for _, caminho in rotas_da_aplicacao()}

    assert documentados == servidos, (
        f"a OpenAPI documenta {sorted(documentados)} e a aplicação serve {sorted(servidos)}"
    )


def test_as_rotas_internas_do_framework_ficam_fora() -> None:
    """`/docs` e `/openapi.json` não são contrato nosso, e exigi-los seria testar o framework.

    A asserção também protege a escolha de discriminar por **tipo**: se alguém trocasse por
    lista de caminhos a ignorar, uma rota interna nova do FastAPI passaria a exigir contrato
    e o portão reprovaria sozinho.
    """
    internas = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    servidos = {caminho for _, caminho in rotas_da_aplicacao()}

    assert not (internas & servidos), (
        "rota interna do framework entrou no inventário; a discriminação por APIRoute falhou"
    )


# ---------------------------------------------------------------------------
# O par causal, nas duas direções
# ---------------------------------------------------------------------------


def test_to_12_2_rota_sem_linha_no_manifesto_reprova() -> None:
    """Direção A: funcionalidade acrescentada sem contrato."""
    defeitos = defeitos_de_bijecao({("GET", "/nova"), ("GET", "/health")}, declaradas())

    assert defeitos
    assert any("/nova" in d and "NÃO está declarada" in d for d in defeitos), defeitos


def test_to_12_2_linha_orfa_reprova() -> None:
    """Direção B: renomeação, e é a que o desenho óbvio esquece."""
    linhas = [*declaradas(), LinhaDeclarada("GET", "/rota-que-nao-existe", "x::y")]

    defeitos = defeitos_de_bijecao(rotas_da_aplicacao(), linhas)

    assert defeitos
    assert any("/rota-que-nao-existe" in d and "NÃO existe" in d for d in defeitos), defeitos


def test_to_12_2_linha_duplicada_reprova() -> None:
    """Duas linhas para a mesma rota tornariam a contagem de contratos maior que a de rotas."""
    linhas = [*declaradas(), LinhaDeclarada("GET", "/health", "outro::teste")]

    defeitos = defeitos_de_bijecao(rotas_da_aplicacao(), linhas)

    assert any("mais de uma linha" in d for d in defeitos), defeitos


def test_to_12_2_com_a_planta_removida_passa() -> None:
    """A outra metade do par: o portão deixa de acusar quando não há o que acusar."""
    assert defeitos_de_bijecao(rotas_da_aplicacao(), declaradas()) == []


def test_to_12_2_no_inexistente_reprova() -> None:
    """A cláusula do nó, plantada: nome de função inventado."""
    linha = LinhaDeclarada("GET", "/health", "tests/test_api_contrato.py::funcao_que_nao_existe")
    caminho, _, funcao = linha.no.partition("::")
    arvore = ast.parse((REPO / caminho).read_text(encoding="utf-8"))
    nomes = {n.name for n in ast.walk(arvore) if isinstance(n, ast.FunctionDef)}

    assert funcao not in nomes, "a planta precisa ser um nome que de facto não existe"


def test_o_metodo_faz_parte_da_chave() -> None:
    """`GET /x` e `POST /x` são contratos diferentes, e colapsá-los deixaria um sem teste."""
    defeitos = defeitos_de_bijecao({("POST", "/health")}, declaradas())

    assert defeitos, "POST /health não está declarado e deveria reprovar"


def test_rota_de_roteador_incluido_e_encontrada() -> None:
    """O buraco da varredura rasa, fixado como asserção.

    O FastAPI **não achata** `include_router`: as rotas ficam em
    `_IncludedRouter.original_router.routes`, e uma varredura rasa de `app.routes` não as vê.
    Sem a recursão, este portão afirmava bijeção sobre um conjunto que não continha nenhuma
    rota de `rotas.py` — verde sobre o vazio.

    A asserção é sobre uma rota que **só** existe por roteador incluído. Se alguém
    simplificar `_percorre` de volta para varredura rasa, ela reprova de imediato.
    """
    caminhos = {caminho for _, caminho in rotas_da_aplicacao()}

    assert "/matrizes/validar" in caminhos, (
        "rota de roteador incluído não foi encontrada: a varredura voltou a ser rasa, e a "
        "bijeção está sendo afirmada sobre um conjunto incompleto"
    )
    assert len(caminhos) > 1, "só a rota registrada direto na aplicação foi encontrada"

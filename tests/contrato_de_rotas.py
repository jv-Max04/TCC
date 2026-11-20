"""Auxiliares dos testes de contrato entre a janela e a API.

`contratos_das_rotas` lê o esquema OpenAPI e devolve o que cada rota declara receber e devolver.
`chamadas_diretas_ao_transporte` acha, num arquivo-fonte, as chamadas que falam com o transporte
HTTP sem passar por `envia`, que é o único caminho que o contrato confere.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContratoDeRota:
    """O que uma rota declara receber e devolver.

    `saida` vazio é estado legítimo e não ausência de contrato: as rotas de artefato devolvem
    `text/csv`, e nenhuma chave de JSON é lida delas.
    """

    entrada: frozenset[str]
    saida: frozenset[str]
    devolve_json: bool


def contratos_das_rotas(esquema: dict[str, Any]) -> dict[tuple[str, str], ContratoDeRota]:
    """`(método, caminho) -> ContratoDeRota`, lido do esquema OpenAPI.

    `entrada` une o corpo e os parâmetros, porque o cliente passa tudo em `campos` e o transporte
    decide onde cada um vai.
    """
    componentes = esquema.get("components", {}).get("schemas", {})

    def propriedades(ref: str) -> frozenset[str]:
        nome = ref.rsplit("/", 1)[-1]
        return frozenset(componentes.get(nome, {}).get("properties", {}))

    contratos: dict[tuple[str, str], ContratoDeRota] = {}
    for caminho, operacoes in esquema.get("paths", {}).items():
        for metodo, operacao in operacoes.items():
            entrada: set[str] = {p["name"] for p in operacao.get("parameters", [])}
            for spec in operacao.get("requestBody", {}).get("content", {}).values():
                alvo = spec.get("schema", {})
                if "$ref" in alvo:
                    entrada |= set(propriedades(alvo["$ref"]))
                else:
                    entrada |= set(alvo.get("properties", {}))

            # `devolve_json` sai do content-type e não da forma do esquema: uma rota com
            # `response_model=list[X]` ou uma união não tem `$ref` no topo e ficaria fora da
            # conferência sem nada acusar.
            saida: set[str] = set()
            devolve_json = False
            for codigo in ("200", "202"):
                spec = operacao.get("responses", {}).get(codigo, {})
                json_ = spec.get("content", {}).get("application/json")
                if json_ is None:
                    continue
                esquema = json_.get("schema") or {}
                # Esquema vazio é rota sem `response_model`: o FastAPI declara `application/json`
                # com `{}`, e as rotas de artefato, que devolvem CSV, caem aqui.
                if not esquema:
                    continue
                devolve_json = True
                saida |= _propriedades_do_esquema(esquema, componentes)

            contratos[(metodo.upper(), caminho)] = ContratoDeRota(
                entrada=frozenset(entrada), saida=frozenset(saida), devolve_json=devolve_json
            )
    return contratos


def _propriedades_do_esquema(alvo: dict[str, Any], componentes: dict[str, Any]) -> frozenset[str]:
    """As propriedades de um esquema de resposta, seguindo `$ref`, `anyOf`, `allOf` e `items`.

    Conjunto vazio é resposta legítima: o `/health` devolve um mapa de chave livre, sem
    propriedades declaradas.
    """
    if not alvo:
        return frozenset()
    if "$ref" in alvo:
        nome = str(alvo["$ref"]).rsplit("/", 1)[-1]
        return frozenset(componentes.get(nome, {}).get("properties", {}))
    if "properties" in alvo:
        return frozenset(alvo["properties"])
    for chave in ("anyOf", "oneOf", "allOf"):
        if chave in alvo:
            reunidas: set[str] = set()
            for ramo in alvo[chave]:
                reunidas |= _propriedades_do_esquema(ramo, componentes)
            return frozenset(reunidas)
    if "items" in alvo:
        return _propriedades_do_esquema(alvo["items"], componentes)
    return frozenset()


def chamadas_diretas_ao_transporte(fonte: str) -> list[str]:
    """As chamadas que falam com o transporte sem passar por `envia`, uma por linha.

    Dois critérios, nenhum dependente do nome da variável: um verbo de escrita (`.post`, `.put`,
    `.delete`, `.patch`, `.request`, que não são métodos de `dict`) em qualquer receptor, e a
    fábrica `constroi_transporte` referenciada fora de `_transporte`. `.get` fica de fora porque
    acusaria todo `dict.get`; um `.get` de transporte precisaria da fábrica para existir.
    """
    arvore = ast.parse(fonte)
    achados: list[str] = []

    for no in ast.walk(arvore):
        if (
            isinstance(no, ast.Call)
            and isinstance(no.func, ast.Attribute)
            and no.func.attr in ("post", "put", "delete", "patch", "request")
        ):
            achados.append(f"linha {no.lineno} — {ast.unparse(no)[:70]}")

    dentro_da_fabrica = {
        filho
        for funcao in ast.walk(arvore)
        if isinstance(funcao, ast.FunctionDef) and funcao.name == "_transporte"
        for filho in ast.walk(funcao)
    }
    for no in ast.walk(arvore):
        if (
            isinstance(no, ast.Name)
            and no.id == "constroi_transporte"
            and no not in dentro_da_fabrica
        ):
            achados.append(f"linha {no.lineno} — `constroi_transporte` fora de `_transporte`")

    return achados

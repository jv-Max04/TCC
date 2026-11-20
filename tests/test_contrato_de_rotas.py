"""Os auxiliares de contrato de rota acusam o que existem para acusar, e só isso."""

from __future__ import annotations

import pytest

from refino.api.app import build_app
from tests.contrato_de_rotas import (
    ContratoDeRota,
    _propriedades_do_esquema,
    chamadas_diretas_ao_transporte,
    contratos_das_rotas,
)


@pytest.fixture(scope="module")
def contratos() -> dict[tuple[str, str], ContratoDeRota]:
    return contratos_das_rotas(build_app().openapi())


def test_a_guarda_do_transporte_direto_acusa_quando_ha_uma() -> None:
    """Par causal da guarda: a mesma função acusa a planta e cala sem ela."""
    plantado = 'def f(transporte):\n    return transporte.post("/x", data={})\n'
    limpo = 'def f(transporte):\n    return envia(transporte, "POST", "/x")\n'

    assert chamadas_diretas_ao_transporte(plantado), (
        "a guarda não vê a chamada direta que ela existe para ver"
    )
    assert not chamadas_diretas_ao_transporte(limpo), (
        "com a planta REMOVIDA a guarda tem de deixar de acusar"
    )


def test_a_guarda_do_transporte_apanha_a_fabrica_e_os_verbos(
    contratos: dict[tuple[str, str], ContratoDeRota],
) -> None:
    """A guarda não depende do nome `transporte`: apanha a fábrica e o verbo em todo receptor."""
    assert contratos, "âncora: o contexto do teste está montado"

    # (a) verbo de escrita em receptor de nome qualquer.
    fabrica = "def f():\n    return constroi_transporte().post('/x', data={})\n"
    apelidado = "def f(cliente):\n    return cliente.post('/x', data={})\n"
    limpo = 'def f(transporte):\n    return envia(transporte, "POST", "/x")\n'

    assert chamadas_diretas_ao_transporte(fabrica), "a fábrica chamada direto passou"
    assert chamadas_diretas_ao_transporte(apelidado), "o transporte sob outro nome passou"
    assert not chamadas_diretas_ao_transporte(limpo), "com a planta removida tem de calar"

    # (b) a fábrica referenciada fora de `_transporte`, mesmo sem chamar verbo nenhum.
    vazado = "def outra():\n    guardado = constroi_transporte\n    return guardado\n"
    contido = "def _transporte(st, x):\n    return x or constroi_transporte()\n"

    assert chamadas_diretas_ao_transporte(vazado), "a fábrica vazou e a guarda calou"
    assert not chamadas_diretas_ao_transporte(contido), (
        "dentro de `_transporte` a fábrica é legítima — acusá-la seria portão que sempre reprova"
    )

    # (c) `.get` de dicionário não é acusado.
    dicionario = 'def f(dados):\n    return dados.get("chave")\n'
    assert not chamadas_diretas_ao_transporte(dicionario), (
        "`dict.get` foi acusado — a guarda vira portão que sempre reprova"
    )


def test_toda_rota_que_devolve_json_resolve_propriedades(
    contratos: dict[tuple[str, str], ContratoDeRota],
) -> None:
    """Toda rota com `application/json` resolve propriedades, salvo o `/health`, de chave livre."""
    com_json = {chave for chave, c in contratos.items() if c.devolve_json}
    assert com_json, "âncora: alguma rota devolve JSON"

    sem_propriedades = {chave for chave in com_json if not contratos[chave].saida}
    assert sem_propriedades == {("GET", "/health")}, (
        f"rota que devolve JSON e não resolve propriedade nenhuma: {sorted(sem_propriedades)}. "
        "A conferência de leitura fica cega sobre ela — declare-a ou ensine o resolvedor a seguir "
        "a forma"
    )


def test_o_resolvedor_de_esquema_segue_as_formas_que_nao_sao_ref() -> None:
    """`list[X]` e uniões, que o FastAPI gera sem `$ref` no topo, resolvem as propriedades de X."""
    componentes = {"X": {"properties": {"a": {}, "b": {}}}}

    direto = {"$ref": "#/components/schemas/X"}
    lista = {"type": "array", "items": {"$ref": "#/components/schemas/X"}}
    uniao = {"anyOf": [{"$ref": "#/components/schemas/X"}, {"type": "null"}]}
    livre = {"additionalProperties": {"type": "string"}, "type": "object"}

    assert _propriedades_do_esquema(direto, componentes) == {"a", "b"}
    assert _propriedades_do_esquema(lista, componentes) == {"a", "b"}, "`list[X]` ficava cega"
    assert _propriedades_do_esquema(uniao, componentes) == {"a", "b"}, "a união ficava cega"
    assert _propriedades_do_esquema(livre, componentes) == frozenset(), (
        "mapa de chave livre resolve vazio — e é quem chama que decide se isso é cegueira"
    )

"""O Swagger sai do executável congelado.

**O que este portão afirma, e o que ele NÃO afirma.** Ele afirma que as quatro rotas de
documentação deixam de responder quando `documentacao=False`, que continuam a responder quando
`True`, e que o interruptor por omissão consulta `sys.frozen`. Ele **não** afirma nada sobre
segurança de rede: o listener é `127.0.0.1` nos dois casos, e `/docs` nunca esteve exposto à rede
— esteve exposto a quem está na máquina. Ver a docstring de `build_app`.

**A lista das quatro é escrita à mão em `api/app.py` e CONFERIDA contra o próprio FastAPI** —
`docs_url`,
`redoc_url`, `openapi_url` e `swagger_ui_oauth2_redirect_url` são atributos da aplicação. Uma
lista pinada é o retrato de ontem, e a primeira versão do recorte desta
tarefa nomeava **duas** das quatro. Derivá-las é o que faz o portão crescer sozinho se a
biblioteca acrescentar uma quinta.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from refino.api.app import ROTAS_DE_DOCUMENTACAO, build_app, serve_documentacao
from tests.test_api_contrato import rotas_de_negocio


def _rotas_que_o_framework_declara(app: FastAPI) -> set[str]:
    """As rotas de documentação **segundo a própria aplicação**, e não segundo esta suíte."""
    return {
        caminho
        for caminho in (
            app.docs_url,
            app.redoc_url,
            app.openapi_url,
            app.swagger_ui_oauth2_redirect_url,
        )
        if caminho is not None
    }


def test_a_constante_bate_com_o_que_o_framework_declara() -> None:
    """A lista deste módulo é a do FastAPI, conferida e não copiada.

    Sem esta linha, `ROTAS_DE_DOCUMENTACAO` seria um retrato: o dia em que o FastAPI mudasse
    `/redoc` de sítio, o portão abaixo continuaria a afirmar `404` sobre um caminho que já não
    existia — sã sobre o alvo errado, com desfecho de sucesso.
    """
    declaradas = _rotas_que_o_framework_declara(build_app(documentacao=True))

    assert declaradas, "âncora: a aplicação com documentação ligada não declarou rota nenhuma"
    assert declaradas == set(ROTAS_DE_DOCUMENTACAO), (
        f"o FastAPI declara {sorted(declaradas)} e a constante diz {sorted(ROTAS_DE_DOCUMENTACAO)}"
    )


def test_com_documentacao_ligada_as_quatro_respondem() -> None:
    """**A direção que impede o teste seguinte de passar por vacuidade.**

    Sem ela, um `build_app` que devolvesse uma aplicação vazia faria os quatro `404` do teste de
    baixo passarem — e o portão certificaria o desligamento de rotas que nunca existiram.
    """
    cliente = TestClient(build_app(documentacao=True))

    codigos = {rota: cliente.get(rota).status_code for rota in ROTAS_DE_DOCUMENTACAO}

    assert codigos == dict.fromkeys(ROTAS_DE_DOCUMENTACAO, 200), (
        f"com a documentação LIGADA, nem tudo respondeu 200: {codigos}"
    )


def test_com_documentacao_desligada_as_quatro_somem() -> None:
    """As **quatro**, e não as duas que se escreve de memória.

    Medido contra o congelado antes de haver interruptor: as quatro serviam `200`, e
    `/docs/oauth2-redirect` é a que não aparece em nenhuma discussão do assunto.
    """
    cliente = TestClient(build_app(documentacao=False))

    codigos = {rota: cliente.get(rota).status_code for rota in ROTAS_DE_DOCUMENTACAO}

    assert codigos == dict.fromkeys(ROTAS_DE_DOCUMENTACAO, 404), (
        f"com a documentação DESLIGADA, alguma rota continuou a servir: {codigos}"
    )


def test_desligar_a_documentacao_nao_tira_rota_de_negocio() -> None:
    """**O custo em capacidade é zero, e isto é a medição e não a afirmação.**

    O par de contagens é o conteúdo: as rotas de negócio são as **mesmas** nas duas configurações.
    Um portão que só contasse a aplicação desligada não distinguiria *"desliguei a documentação"*
    de *"desliguei a aplicação"*.

    ## A travessia é `rotas_de_negocio`, e a primeira versão deste teste media UMA rota de dezasseis

    **O FastAPI não achata `include_router`**: ele deixa em `app.routes` um `_IncludedRouter` cujo
    `path` é `None` e cujas rotas vivem em `original_router.routes`. A primeira versão iterava
    `app.routes` com `getattr(r, "path", None)` e obtinha **`{"/health"}`** nos dois lados — a
    âncora `assert negocio_ligada` era satisfeita por essa única rota, e o portão certificava que
    *"as rotas de negócio são as mesmas"* tendo comparado uma.

    **Medido por planta, com a mutação conferida no arquivo antes de o desfecho ser lido:** uma
    aplicação desligada à qual falta `/execucoes` — quinze rotas contra dezasseis — deixava este
    arquivo com **`8 passed`**. Perder uma rota de negócio ao desligar a documentação é
    exactamente o que ele existe para impedir.

    > **O repositório já tinha pago este defeito e escrito a correção a 130 linhas daqui.**
    > `tests/test_api_contrato.py::_percorre` nasceu dele, e a docstring dela di-lo verbatim:
    > *"as rotas registradas por roteador — **todas** as de `rotas.py` — ficavam invisíveis, e a
    > bijeção passava afirmando cobertura de um conjunto que não continha nenhuma delas"*. Reusar
    > a travessia é o conserto; reescrevê-la seria a segunda fonte de verdade sobre a mesma
    > peculiaridade do framework.
    """
    ligada = build_app(documentacao=True)
    desligada = build_app(documentacao=False)

    negocio_ligada = rotas_de_negocio(ligada)
    negocio_desligada = rotas_de_negocio(desligada)

    assert len(negocio_ligada) > 1, (
        f"âncora: a travessia achou {len(negocio_ligada)} rota(s) de negócio. Este projeto tem "
        "mais de uma, e uma contagem de 1 significa que ela parou no `_IncludedRouter` — que foi "
        "o defeito da primeira versão deste teste"
    )
    assert negocio_desligada == negocio_ligada, (
        "desligar a documentação mexeu nas rotas de negócio: "
        f"sumiram {sorted(negocio_ligada - negocio_desligada)}, "
        f"apareceram {sorted(negocio_desligada - negocio_ligada)}"
    )
    assert TestClient(desligada).get("/health").status_code == 200


def test_o_metodo_openapi_continua_a_devolver_o_contrato_com_a_rota_desligada() -> None:
    """**Quem precisa do contrato em teste chama o MÉTODO; quem o pedia pela ROTA é que muda.**

    **Sete** sítios desta suíte já chamavam `app.openapi()` antes deste portão existir, e nenhum
    deles quebra — medido. É essa medição que torna o desligamento barato, e é ela que esta linha
    protege de caducar. (A docstring dizia **seis**: a varredura que a produziu não contou
    `tests/test_historico_na_janela.py`, escrito no mesmo dia.)
    """
    caminhos = set(build_app(documentacao=False).openapi()["paths"])

    assert caminhos, "o método `openapi()` devolveu documento sem caminho nenhum"
    assert "/multiversos" in caminhos, (
        f"o contrato perdeu as rotas de negócio ao desligar a documentação: {sorted(caminhos)}"
    )


@pytest.mark.parametrize(
    ("congelado", "esperado"),
    [(False, 200), (True, 404)],
    ids=["em codigo serve", "congelado nao serve"],
)
def test_o_padrao_consulta_o_processo(
    monkeypatch: pytest.MonkeyPatch, congelado: bool, esperado: int
) -> None:
    """`build_app()` **sem argumento** decide por `sys.frozen`, e as duas direções são medidas.

    É a costura entre `serve_documentacao` e `build_app`: as duas podiam estar certas em separado
    com o `None` a nunca chegar à função, que é um defeito de ligação.
    """
    import refino.api.app as modulo

    monkeypatch.setattr(modulo.sys, "frozen", congelado, raising=False)

    assert serve_documentacao() is (not congelado)
    assert TestClient(build_app()).get("/docs").status_code == esperado


def test_a_pagina_do_swagger_e_a_unica_coisa_que_puxa_de_fora() -> None:
    """**Desligar `/docs` deixa o binário sem dependência de CDN nenhuma.**

    Medido em 28/08/2026 contra o congelado: `/docs` puxa **3** URLs de 2 hospedeiros e `/redoc`
    outras **3** de 3 — e a união é de **cinco URLs distintas**, porque o favicon de
    `fastapi.tiangolo.com` repete-se nas duas páginas.

    O portão afirma o lado que ele consegue ver de dentro: o corpo servido com a documentação
    ligada nomeia hospedeiro externo, e com ela desligada não há corpo nenhum onde ele possa estar.
    """
    corpo_ligado = TestClient(build_app(documentacao=True)).get("/docs").text

    assert "https://" in corpo_ligado, (
        "âncora: a página do Swagger deixou de puxar recurso externo, e o resto deste teste "
        "passaria por vacuidade"
    )

    corpo_desligado = TestClient(build_app(documentacao=False)).get("/docs").text

    # **A asserção é sobre o HOSPEDEIRO e não sobre a diferença de texto.** A primeira versão
    # afirmava `corpo_desligado != corpo_ligado`, que já está implicado pelo `404` do teste de
    # cima — e a docstring prometia mais: *"não há corpo nenhum onde ele possa estar"*.
    for hospedeiro in ("cdn.jsdelivr.net", "fastapi.tiangolo.com", "fonts.googleapis.com"):
        assert hospedeiro not in corpo_desligado, (
            f"o corpo servido com a documentação desligada ainda nomeia {hospedeiro}"
        )
    assert corpo_desligado != corpo_ligado

"""A aplicação FastAPI e a fronteira de rede.

Esta é a camada que serve os dados, e os arquivos de `data/reference/` são clínicos: o nome de
cada amostra liga um pseudônimo estável a um diagnóstico, entre eles ATLL e HAM/TSP. Por isso o
endereço de escuta é fixo, e não configurável.

A API não tem autenticação: ela escuta só em loopback, dentro do processo da janela, e a
proteção é o binding. Autenticação mal feita sugeriria uma proteção que não existe.
"""

import sys
from typing import Any, Final

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from refino.api.rotas import router
from refino.correction.serrf import IntensidadeNegativaError
from refino.metrics.structure import EstruturaSemPopulacaoError

__all__ = [
    "LOOPBACK_HOST",
    "ROTAS_DE_DOCUMENTACAO",
    "build_app",
    "default_host",
    "serve_documentacao",
]

# As quatro rotas que o Swagger acrescenta. O portão da documentação desligada percorre todas, e
# `/docs/oauth2-redirect` é a que se esquece ao escrever de memória.
ROTAS_DE_DOCUMENTACAO: Final[tuple[str, ...]] = (
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
)

# O binding é a única fronteira de rede: a API escuta só em loopback.
LOOPBACK_HOST: Final = "127.0.0.1"


def default_host() -> str:
    """O endereço em que a API escuta: sempre loopback."""
    return LOOPBACK_HOST


def serve_documentacao() -> bool:
    """Se a documentação interativa serve: não, quando o processo é o executável congelado.

    É função e não constante porque `sys.frozen` é atributo do processo: um valor calculado no
    `import` fixaria a resposta antes de um teste poder trocá-la, e como função o teste a substitui
    e mede os dois lados. `getattr` porque o atributo só existe sob PyInstaller, o mesmo idioma que
    `desktop/arranque.py` usa.
    """
    return not getattr(sys, "frozen", False)


def build_app(*, documentacao: bool | None = None) -> FastAPI:
    """Monta a aplicação, com as rotas e os tradutores de erro.

    `documentacao=None` decide por `serve_documentacao()`; `True` e `False` servem ao teste. Sem a
    documentação, as quatro rotas de `ROTAS_DE_DOCUMENTACAO` respondem `404`, as rotas de negócio
    continuam servidas, e `app.openapi()` continua a devolver o contrato, que é como os testes o
    leem. Os três parâmetros vão a `None` mesmo que `openapi_url=None` baste no FastAPI atual: essa
    cascata é detalhe da biblioteca, e declarar os três mantém a intenção se ela mudar.

    O servidor só escuta em loopback, então `/docs` nunca fica exposto à rede. Desligar a
    documentação no congelado tira do executável entregue um índice clicável das rotas que recebem
    e servem dado clínico, e o deixa sem dependência de CDN: as páginas do Swagger e do ReDoc são
    as únicas que puxam recursos de fora. Nada em `src/` consome essas rotas.
    """
    documentacao = serve_documentacao() if documentacao is None else documentacao
    # `{}` no caminho ligado, e não os padrões escritos à mão: uma cópia deles aqui decidiria no dia
    # em que o FastAPI mudasse `/docs` de lugar, e ninguém a reveria porque parece inerte.
    # `dict[str, Any]` e não `dict[str, None]`: com `**` de um dicionário de `None`, o mypy tenta
    # casar `None` com o tipo de cada parâmetro de `FastAPI` não passado, e reprova em quatro.
    sem_documentacao: dict[str, Any] = (
        {} if documentacao else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    )
    app = FastAPI(
        title="refino",
        description=(
            "Refino quimiométrico de dados de metabolômica LC-MS. "
            "Servida em loopback: os dados são clínicos humanos identificáveis."
        ),
        **sem_documentacao,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Responde `ok` quando o servidor está pronto para receber pedidos.

        O arranque da janela consulta esta rota antes de abrir.
        """
        return {"status": "ok"}

    @app.exception_handler(EstruturaSemPopulacaoError)
    async def estrutura_sem_populacao(_: Request, erro: EstruturaSemPopulacaoError) -> JSONResponse:
        """`422` para a matriz que a medida estrutural não sustenta.

        Um tradutor só, no aplicativo, e não um `try` por rota: a recusa nasce em
        `metrics/structure.py` e chega por várias rotas, e uma cópia do `except` em cada uma seria
        a que falta na próxima. O texto começa por 'a matriz não sustenta', uma das
        `MARCAS_DE_CONTEUDO` de `interface/cliente.py`: sem ela, o cliente leria este `422` como
        campo
        faltando, e não como problema do arquivo.
        """
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": f"a matriz não sustenta a medida estrutural: {erro}"},
        )

    @app.exception_handler(IntensidadeNegativaError)
    async def intensidade_negativa(_: Request, erro: IntensidadeNegativaError) -> JSONResponse:
        """`422` para a matriz que o passo 1 do SERRF não sorteia.

        É fato do arquivo, intensidade abaixo de -0,1 ao lado de zeros, e não falha do servidor; a
        mensagem leva a mesma marca do tradutor de cima. O SERRF corre quase sempre no trabalho de
        fundo, traduzido por `_executa_trabalho`; este tradutor cobre o que corre no disparo, o
        Palpite medido de `auto=medida`, que `_delegacao_ou_422` mede antes de despachar
        `/execucoes` e `/matrizes/corrigir`.
        """
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": f"a matriz não sustenta o SERRF: {erro}"},
        )

    app.include_router(router)
    return app

"""O cliente HTTP da janela: traduz cada desfecho de uma chamada à API em causa e mensagem.

A causa diz quem tem de agir (`usuario`, `dado` ou `ambiente`), e não é o código HTTP: o `422`
cobre 'você errou o rótulo' (`usuario`) e 'o seu arquivo está malformado' (`dado`), que pedem
ações opostas, e o discriminador é mecânico (o `detail` da validação do FastAPI é lista de
dicionários; o das nossas guardas, string). O mapeamento é função pura de `(código, corpo)`, e o
`500` também é tratado, porque cliente que não o trata mostra tela em branco.

`envia` recebe o transporte em vez de construí-lo: em produção é um `httpx2.Client`, e no teste é
o `TestClient` do FastAPI, que fala com a app ASGI em processo, sem subir servidor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Final, Literal, Protocol, cast
from uuid import uuid4

__all__ = [
    "CAMPO_DA_CHAVE",
    "CAUSA_DE_FALHA",
    "DESFECHOS",
    "PREFIXO_DE_DIVERGENCIA",
    "SEM_FALHA",
    "Causa",
    "Resposta",
    "Transporte",
    "base_da_api",
    "causa_de",
    "chave_de_invalidacao",
    "constroi_transporte",
    "envia",
    "mensagem_de",
    "mensagem_por_chave",
]

# Vocabulário fechado de três valores: a resposta a 'quem tem de agir?', que é a pergunta que a
# pessoa faz diante de um erro.
Causa = Literal["usuario", "dado", "ambiente"]

# Os dois desfechos que não são falha ficam fora do vocabulário: classificar sucesso como causa
# faria a contagem de falhas por causa contar sucesso.
SEM_FALHA: Final[frozenset[int]] = frozenset({200, 202})

# O mapeamento. As chaves do `422` e do `503` são compostas porque o código sozinho não decide.
# Os três `503` são `ambiente`, mas pedem ações diferentes: `503:banco` vem da API, e
# `503:demora` e `503:rede` são fabricados aqui quando não há resposta. Com uma frase só, quem
# apenas esperou demais seria mandado tratar um banco que está saudável.
CAUSA_DE_FALHA: Final[dict[str, Causa]] = {
    "404": "usuario",
    "409": "dado",
    # `410` significa 'você cancelou': é do usuário, e não do dado, e a ação certa é nenhuma.
    # Classificá-lo como `dado` mandaria a pessoa conferir um arquivo perfeito.
    "410": "usuario",
    "422:forma": "usuario",
    "422:conteudo": "dado",
    "500": "ambiente",
    "503:banco": "ambiente",
    "503:demora": "ambiente",
    "503:rede": "ambiente",
}

# Todos os desfechos que a interface pode receber.
DESFECHOS: Final[tuple[int, ...]] = (200, 202, 404, 409, 410, 422, 500, 503)

_MENSAGEM: Final[dict[str, str]] = {
    "404": "Teste de robustez não encontrado. Confira o identificador que você colou.",
    "409": "Os filtros escolhidos removeram todas as features. Afrouxe o corte ou revise o dado.",
    # Não fala de arquivo nem manda conferir nada: o trabalho parou porque a pessoa mandou. O
    # fato acionável é que disparar de novo cria trabalho novo, porque a recusa de duplicação
    # não vale para trabalho cancelado.
    "410": (
        "Você cancelou este trabalho, então não há resultado a buscar. Nada está errado com o seu "
        "arquivo. Se ainda o quiser, dispare outra vez — será um trabalho novo, do início."
    ),
    "422:forma": "Faltou um campo, ou um valor não pertence ao vocabulário aceito.",
    "422:conteudo": "O arquivo enviado foi recusado pela validação. Os problemas estão abaixo.",
    "500": (
        "Defeito da ferramenta, não do seu arquivo. Reporte esta tela a quem mantém o projeto; "
        "não adianta conferir o dado."
    ),
    # A frase não manda operar infraestrutura: a janela grava o histórico em SQLite num arquivo
    # (`desktop/banco.py`), e quem a lê não tem serviço de banco a subir. Ela é neutra quanto à
    # causa concreta, que o servidor põe no `detail`, e diz o que continua a funcionar: só
    # executar o pipeline e ler o histórico de multiversos dependem do banco.
    "503:banco": (
        "O programa não conseguiu abrir o banco onde ele grava o histórico. É do ambiente e não "
        "do seu arquivo: o dado que você enviou está intacto. Validar, diagnosticar, recomendar e "
        "gerar os arquivos corrigidos NÃO usam o banco e continuam a funcionar; o que para é o "
        "que grava — executar o pipeline e ler o histórico de testes de robustez. Se repetir não "
        "resolver, reporte a quem instalou a ferramenta."
    ),
    # A operação pode continuar viva, e é isso que separa esta mensagem da seguinte: o cliente
    # desiste de uma requisição em 300 s, e o servidor segue. As rotas longas recusam duplicação
    # e devolvem o mesmo trabalho, então repetir é seguro e não há processo a conferir.
    "503:demora": (
        "A chamada passou do tempo que esta tela espera. A operação pode continuar em curso no "
        "servidor: repita o disparo — a ferramenta devolve o MESMO trabalho em vez de duplicar o "
        "cálculo, e o identificador é o que recupera o resultado."
    ),
    # Aqui nada está correndo: o transporte nunca completou.
    "503:rede": (
        "Não foi possível falar com a API. Isto é do ambiente, não do seu arquivo: confirme que o "
        "serviço está de pé e que o endereço está certo."
    ),
}


class Transporte(Protocol):
    """O mínimo que o cliente usa. Protocolo, e não classe, para o `TestClient` servir sem herdar.

    `httpx2.Client` e `fastapi.testclient.TestClient` imitam a interface do `requests` e satisfazem
    os dois métodos, e é isso que permite ao teste falar com a app ASGI em processo.
    """

    def post(self, url: str, **kwargs: Any) -> Any: ...

    def get(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class Resposta:
    """O que a janela recebe: nunca a resposta crua, e nunca uma exceção.

    `causa` é `None` só quando `ok` é verdadeiro, e um teste afirma a bicondicional: sem ela,
    `None` cobriria 'não falhou' e 'falhou e não sei classificar'.
    """

    codigo: int
    ok: bool
    causa: Causa | None
    mensagem: str
    corpo: Any
    # Os bytes do artefato, ou `None` quando não houve artefato. O botão de download existe se e
    # somente se há bytes, e `b""` para 'sem artefato' confundiria 'não produziu' com 'produziu
    # vazio'.
    conteudo: bytes | None = None


def base_da_api() -> str:
    """A URL da API. Loopback por padrão, para a API não ficar exposta na rede."""
    return os.environ.get("REFINO_API_BASE", "http://127.0.0.1:8000")


def constroi_transporte() -> Transporte:
    """O transporte de produção. Importa `httpx2` aqui, e não no topo do módulo.

    No topo, todo teste que tocasse este arquivo exigiria a biblioteca, e o que se quer afirmar é
    o mapeamento, que é aritmética sobre `(código, corpo)`.

    O `timeout` de 300 s vale por requisição. A espera da janela por um trabalho são muitas
    requisições curtas (o `202` e as releituras de `GET /trabalhos/{id}`), e a mais lenta do
    caminho leva menos de um segundo. Subir o valor por precaução trocaria a única coisa que ele
    mede, 'uma requisição travou', por silêncio: uma chamada presa esperaria horas em vez de
    reportar `503:demora`.
    """
    import httpx2

    # `cast` porque `httpx2.Client.post` tem parâmetros keyword-only tipados, e o `Transporte` os
    # declara como `**kwargs: Any`; o mypy recusa a compatibilidade estrutural. As duas chamadas
    # de `envia` funcionam contra `httpx2.Client` e contra `TestClient`, e há teste do segundo.
    return cast(Transporte, httpx2.Client(base_url=base_da_api(), timeout=300.0))


# As marcas que dizem 'o problema é o arquivo'. O `detail` não tem campo estruturado, então a
# classificação é por substring; o que impede a próxima divergência é o teste que dirige a rota
# real e afirma a classificação, em vez de reconstruir a frase aqui.
MARCAS_DE_CONTEUDO: Final[tuple[str, ...]] = (
    "recusado pela validação",  # `IngestionError`, a lista de problemas da ingestão
    "não pôde ser lido",  # `ValueError` do leitor: linha curta, coluna a mais (rotas.py)
    # `EstruturaSemPopulacaoError`: a matriz que a validação aceita e a medida estrutural não
    # sustenta, como a de um QC só.
    "a matriz não sustenta",
)


def _forma_do_422(corpo: Any) -> str:
    """Discrimina as duas causas do `422`, de forma mecânica.

    A validação do FastAPI devolve `detail` como lista de dicionários com `loc`; as nossas guardas
    devolvem string. String não basta para dizer `conteudo`, e por isso há marcas: a recusa da
    combinação proibida e a de rótulo fora do vocabulário também são strings nossas, e são erro de
    escolha, não de arquivo.

    Sem `detail` reconhecível, cai em `forma`: o `422` veio da fronteira, e não do conteúdo.
    """
    detalhe = corpo.get("detail") if isinstance(corpo, dict) else None
    if isinstance(detalhe, str) and any(marca in detalhe for marca in MARCAS_DE_CONTEUDO):
        return "422:conteudo"
    return "422:forma"


def chave_de(codigo: int, corpo: Any) -> str:
    """A chave do mapeamento para o que vem de resposta da API.

    Um `503` que a API devolve é sempre o do banco: `_repositorio()` é o único lugar que o emite.
    As chaves de demora e de rede não derivam de `(código, corpo)` porque não há resposta, e
    `envia` as seleciona direto (ver `mensagem_por_chave`).
    """
    if codigo == 422:
        return _forma_do_422(corpo)
    if codigo == 503:
        return "503:banco"
    return str(codigo)


def _chave_do_transporte(erro: BaseException) -> str:
    """Discrimina demora de rede pelo nome da classe da exceção, sem importar `httpx2`.

    A família de tempo esgotado do `httpx` tem `Timeout` no nome (`ReadTimeout`, `ConnectTimeout`,
    `PoolTimeout`), e o resto do transporte não. Comparar por nome mantém o cliente sem depender
    dos tipos do transporte, e o `TestClient` serve de duplo sem herdar nada.
    """
    return "503:demora" if "Timeout" in type(erro).__name__ else "503:rede"


def causa_de(codigo: int, corpo: Any = None) -> Causa | None:
    """De quem é a causa, ou `None` quando não houve falha."""
    if codigo in SEM_FALHA:
        return None
    return CAUSA_DE_FALHA.get(chave_de(codigo, corpo), "ambiente")


def mensagem_por_chave(chave: str) -> str:
    """A frase, endereçada pela chave, que é a forma de que os desfechos fabricados precisam.

    Nem toda chave deriva de `(código, corpo)`: as falhas de transporte não têm resposta HTTP, e
    `mensagem_de` não as alcança. Com busca só por código, as chaves de `503` dariam a mesma frase,
    e a alternativa seria fabricar um corpo sentinela para um caminho que não tem resposta.
    """
    return _MENSAGEM.get(
        chave, f"Desfecho {chave} não previsto pela interface. Reporte a quem mantém o projeto."
    )


def mensagem_de(codigo: int, corpo: Any = None) -> str:
    """A frase que a janela mostra, distinta por desfecho."""
    if codigo in SEM_FALHA:
        return "OK"
    return mensagem_por_chave(chave_de(codigo, corpo))


def envia(
    transporte: Transporte,
    metodo: str,
    rota: str,
    *,
    arquivos: dict[str, Any] | None = None,
    campos: dict[str, Any] | None = None,
) -> Resposta:
    """Faz a chamada e devolve `Resposta`. Não levanta, de propósito.

    Uma exceção que subisse até a tela viraria stack trace, e quem lê Python conclui que errou.
    Falha de rede vira `503`: o serviço não respondeu, e isso é `ambiente`.
    """
    try:
        if metodo == "GET":
            resposta = transporte.get(rota, params=campos or {})
        else:
            resposta = transporte.post(rota, files=arquivos or None, data=campos or {})
    except Exception as erro:
        # A chave é escolhida aqui, e não por `chave_de`, porque não há resposta de onde derivá-la.
        # O texto da exceção vai em `corpo["detail"]`, separado da frase, para não colar um trecho
        # de exceção Python na instrução.
        chave = _chave_do_transporte(erro)
        return Resposta(
            codigo=503,
            ok=False,
            causa=CAUSA_DE_FALHA[chave],
            mensagem=mensagem_por_chave(chave),
            corpo={"detail": f"{type(erro).__name__}: {erro}"},
        )

    # O corpo é lido pelo `content-type`: tentar `resposta.json()` e cair em `None` descartaria
    # em silêncio o artefato de uma rota que devolve `text/csv`.
    tipo = str(resposta.headers.get("content-type", ""))
    corpo: Any = None
    conteudo: bytes | None = None
    if "json" in tipo:
        try:
            corpo = resposta.json()
        except Exception:
            corpo = None
    elif "csv" in tipo:
        # Só carrega bytes quando a resposta é o artefato: o `content` de uma resposta de erro não
        # é vazio, e o botão de download apareceria sobre uma mensagem de erro.
        conteudo = resposta.content

    codigo = int(resposta.status_code)
    return Resposta(
        codigo=codigo,
        ok=codigo in SEM_FALHA,
        causa=causa_de(codigo, corpo),
        mensagem=mensagem_de(codigo, corpo),
        corpo=corpo,
        conteudo=conteudo if codigo in SEM_FALHA else None,
    )


# ---------------------------------------------------------------------------
# A chave de invalidação das figuras
# ---------------------------------------------------------------------------

CAMPO_DA_CHAVE: Final = "chave_de_invalidacao"

# O prefixo torna o marcador inconfundível com uma chave real: `_chave_do_trabalho` devolve um
# `sha256().hexdigest()`, 64 caracteres de `0-9a-f`, e `:` não é hexadecimal. Há teste que o
# afirma.
PREFIXO_DE_DIVERGENCIA: Final = "divergente:"


def chave_de_invalidacao(resposta: Resposta) -> str:
    """A identidade da execução que esta resposta descreve, ou um marcador de divergência.

    `PainelDeArtefatos.define_entrada` recebe uma `str` e retorna cedo na igualdade. Se duas
    ausências comparassem iguais (`None`, ou um marcador constante), o painel não descartaria
    nada, e a figura da execução anterior sobreviveria ao `202` e ao erro. Por isso o marcador é
    irrepetível: aqui o `uuid4` significa 'não sei o que isto é, e não posso afirmar que é o mesmo
    de antes'. No servidor, uma chave aleatória destruiria o mecanismo.

    Os desfechos são distintos no texto, para o painel escrever a frase de ausência certa:
    - `200` com a chave: a própria chave, 64 hex; é a execução corrente;
    - `200` sem o campo: `divergente:sem-campo:<nonce>`; nenhuma rota põe chave neste corpo;
    - `200` com campo ilegível: `divergente:campo-ilegivel:<nonce>`; veio, e não é uma chave;
    - outro código (o `202`, o erro): `divergente:codigo-<n>:<nonce>`; ainda não há resultado.
    O `202` cai no último, e não no segundo: um trabalho em curso é a mesma rota antes de ter a
    resposta.
    """
    # `!= 200`, e não `not in SEM_FALHA`: `SEM_FALHA` inclui o `202`, e um trabalho em curso, que
    # vai ter chave, sairia como 'nenhuma rota põe chave neste corpo'.
    if resposta.codigo != 200:
        return _marcador(f"codigo-{resposta.codigo}")
    corpo = resposta.corpo
    if not isinstance(corpo, dict) or CAMPO_DA_CHAVE not in corpo:
        return _marcador("sem-campo")
    valor = corpo[CAMPO_DA_CHAVE]
    if not isinstance(valor, str) or not valor:
        return _marcador("campo-ilegivel")
    return valor


def _marcador(motivo: str) -> str:
    """Um marcador de divergência irrepetível, com o motivo legível à frente do ruído."""
    return f"{PREFIXO_DE_DIVERGENCIA}{motivo}:{uuid4().hex}"

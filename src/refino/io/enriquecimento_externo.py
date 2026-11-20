"""O modo externo do enriquecimento: a ORA KEGG pelo servidor do MetaboAnalyst.

É o único caminho do programa que envia algo para fora da máquina. A ORA do MSEA sobre a
biblioteca KEGG só existe no servidor (`www.xialab.ca/api`): o pacote a chama por `.do.api.call`,
que grava o `mSet` inteiro com `saveRDS`, o envia e devolve `unserialize` da resposta. Este módulo
faz o mesmo pedido sem o R e sem o `unserialize`. A carga é montada por esquema fechado
(`monta_carga`), só com os nomes canônicos que o mapeamento local confirmou e um vocabulário fixo;
`confere_carga` confere os bytes que vão sair; a resposta é lida pelo leitor restrito
(`refino.io.rds_restrito`); e o transporte vai a um endereço constante, com o certificado
verificado, sem seguir redirecionamento, com teto de tempo e de bytes.

O servidor não aplica o fundo do ensaio: com o filtro ligado ele devolve o mesmo universo e os
mesmos valores. Por isso este modo não envia fundo, e o universo é a biblioteca KEGG inteira, o
que a tela informa.
"""

from __future__ import annotations

import _socket
import contextlib
import http.client
import math
import socket
import ssl
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final, Literal, Protocol

from refino.io.rds_restrito import (
    MAX_BYTES,
    Lista,
    Objeto,
    RdsRecusadoError,
    Vetor,
    escreve_rds,
    inventario,
    le_rds,
)

__all__ = [
    "BIBLIOTECA_EXTERNA",
    "DESTINO",
    "ESQUEMA_DA_CARGA",
    "TEMPO_LIMITE_S",
    "TEMPO_TOTAL_S",
    "CargaRecusadaError",
    "DesfechoDoEnvio",
    "RespostaDoServidor",
    "confere_carga",
    "envia_ao_servidor",
    "le_resposta",
    "monta_carga",
]

HOST: Final = "www.xialab.ca"
CAMINHO: Final = "/api/msetora"
DESTINO: Final = f"https://{HOST}{CAMINHO}"
BIBLIOTECA_EXTERNA: Final = "kegg_pathway"
# Os envios de teste levaram de 5,4 a 6,1 s, e 60 s é cerca de dez vezes o maior. O teto é o
# `timeout` do socket, que o `http.client` aplica a cada operação de socket (cada tentativa de
# conexão, o aperto de mão do TLS, cada leitura), e não ao envio inteiro. A resolução do nome
# (`getaddrinfo`) fica fora dele. Um servidor que mandasse um byte a cada poucos segundos não
# seria cortado por este teto.
TEMPO_LIMITE_S: Final = 60.0
# O teto do envio inteiro. A troca corre numa thread própria, e quem espera por ela consulta o
# cancelamento e este teto a cada `_INTERVALO_DA_ESPERA_S`, sem esperar o socket. 120 s é o dobro
# do teto por operação; quando ele vence, a conexão é derrubada e a thread termina sozinha, sem
# segurar o processo (é `daemon`). A resolução do nome continua fora do alcance, porque não há
# socket a derrubar antes dela, mas não prende quem espera.
TEMPO_TOTAL_S: Final = 120.0
_INTERVALO_DA_ESPERA_S: Final = 0.1
_EXCLUDE_NUM: Final = 2.0
_VOCABULARIO_FIXO: Final = frozenset({"conc", "msetora", BIBLIOTECA_EXTERNA})

# O cabeçalho que a carga tem de ter: `X\n`, o formato 2, a versão 4.4.3 que o escritor declara
# e a mínima 2.3.0. É literal, e não derivado das constantes do escritor, para que um escritor
# que mude o cabeçalho reprove aqui em vez de mudar os dois lados da comparação juntos.
_CABECALHO_DA_CARGA: Final = bytes.fromhex("580a000000020004040300020300")

# O esquema fechado: caminho e tipo de cada folha, na ordem em que `monta_carga` as escreve. O
# tamanho de `oraVec` é o número de nomes; o de cada outra folha é 1.
ESQUEMA_DA_CARGA: Final[tuple[tuple[str, str], ...]] = (
    ("raiz$mSet$dataSet$type", "texto"),
    ("raiz$mSet$analSet$type", "texto"),
    ("raiz$mSet$analSet$msetlibname", "texto"),
    ("raiz$mSet$api$filter", "logico"),
    ("raiz$mSet$api$libname", "texto"),
    ("raiz$mSet$api$excludeNum", "real"),
    ("raiz$mSet$api$oraVec", "texto"),
    ("raiz$libNm", "texto"),
    ("raiz$filter", "logico"),
    ("raiz$oraVec", "texto"),
    ("raiz$excludeNum", "real"),
)
_ORAVEC_DA_RAIZ: Final = "raiz$oraVec"
_VETORES_DE_NOMES: Final = frozenset({"raiz$mSet$api$oraVec", _ORAVEC_DA_RAIZ})

DesfechoDoEnvio = Literal[
    "ok", "sem_rede", "transporte_recusado", "servidor_recusou", "resposta_malformada", "cancelado"
]


class CargaRecusadaError(ValueError):
    """A carga montada não passa na conferência: esquema diferente, ou texto que não pode sair."""


def _texto(*valores: str) -> Vetor:
    return Vetor("texto", tuple(valores))


def _nomeada(**campos: Objeto) -> Lista:
    return Lista(tuple(campos.values()), {"names": _texto(*campos)})


def monta_carga(nomes: Sequence[str]) -> bytes:
    """Monta a carga do pedido, por esquema fechado, a partir dos nomes canônicos confirmados."""
    if not nomes:
        raise CargaRecusadaError("a carga precisa de ao menos um nome")
    if len(set(nomes)) != len(nomes):
        raise CargaRecusadaError("a carga não leva nome repetido")
    vetor = _texto(*nomes)
    falso = Vetor("logico", (False,))
    exclusao = Vetor("real", (_EXCLUDE_NUM,))
    carga = _nomeada(
        mSet=_nomeada(
            dataSet=_nomeada(type=_texto("conc")),
            analSet=_nomeada(type=_texto("msetora"), msetlibname=_texto(BIBLIOTECA_EXTERNA)),
            api=_nomeada(
                filter=falso,
                libname=_texto(BIBLIOTECA_EXTERNA),
                excludeNum=exclusao,
                oraVec=vetor,
            ),
        ),
        libNm=_texto(BIBLIOTECA_EXTERNA),
        filter=falso,
        oraVec=vetor,
        excludeNum=exclusao,
    )
    return escreve_rds(carga)


def _folhas(objeto: Objeto, caminho: str = "raiz") -> list[tuple[str, Vetor]]:
    if isinstance(objeto, Vetor):
        return [(caminho, objeto)]
    if isinstance(objeto, Lista):
        saida: list[tuple[str, Vetor]] = []
        for nome, item in zip(objeto.nomes(), objeto.itens, strict=True):
            saida.extend(_folhas(item, f"{caminho}${nome}"))
        return saida
    raise CargaRecusadaError(f"{caminho}: folha que não é vetor")


def confere_carga(carga: bytes, permitidos: Iterable[str]) -> None:
    """Confere os bytes que vão sair: esquema exato, texto só permitido ou fixo, e forma canônica.

    A forma canônica exige que os bytes sejam exatamente os que `monta_carga` escreve com os nomes
    lidos da própria carga. As conferências por folha leem o objeto, e os bytes levam mais do que
    as folhas: atributos de folha e da raiz, o nome de codificação do cabeçalho do formato 3, os
    bytes das versões do R no cabeçalho, o nome de arquivo do envelope `gzip`. Recusar cada um
    desses lugares seria uma lista que não fecha. Com o esquema fechado, dados os nomes só existe
    uma carga, e é essa que se exige.

    As conferências por folha vêm antes porque dizem qual regra a carga quebrou. A mensagem da
    forma canônica diz em que byte a carga diverge.
    """
    autorizados = frozenset(permitidos)
    try:
        objeto = le_rds(carga)
    except RdsRecusadoError as erro:
        raise CargaRecusadaError(f"a carga não é dado puro: {erro}") from erro
    if carga[: len(_CABECALHO_DA_CARGA)] != _CABECALHO_DA_CARGA:
        raise CargaRecusadaError("a carga não tem o cabeçalho do escritor deste programa")
    forma = tuple((c, t) for c, t, _ in inventario(objeto))
    if forma != ESQUEMA_DA_CARGA:
        raise CargaRecusadaError(f"a carga não tem o esquema fechado: {forma}")
    _confere_atributos(objeto, "raiz")
    nomes_lidos: list[str] = []
    for caminho, vetor in _folhas(objeto):
        if caminho in _VETORES_DE_NOMES:
            fora = [v for v in vetor.valores if v not in autorizados]
            if fora or not vetor.valores:
                raise CargaRecusadaError(f"{caminho}: texto que não é nome confirmado: {fora}")
            if caminho == _ORAVEC_DA_RAIZ:
                nomes_lidos = [v for v in vetor.valores if isinstance(v, str)]
        elif len(vetor.valores) != 1:
            raise CargaRecusadaError(f"{caminho}: {len(vetor.valores)} valores, esperava 1")
        elif vetor.tipo == "texto" and vetor.valores[0] not in _VOCABULARIO_FIXO:
            raise CargaRecusadaError(f"{caminho}: texto fora do vocabulário fixo")
        elif vetor.tipo == "logico" and vetor.valores[0] is not False:
            raise CargaRecusadaError(f"{caminho}: o filtro do fundo não vai ligado")
        elif vetor.tipo == "real" and vetor.valores[0] != _EXCLUDE_NUM:
            raise CargaRecusadaError(f"{caminho}: parâmetro fora do fixo")
    # Nome repetido não tem forma canônica: `monta_carga` o recusa, e a recusa dela vale aqui.
    canonica = monta_carga(nomes_lidos)
    if carga != canonica:
        byte = next(
            (i for i, (a, b) in enumerate(zip(carga, canonica, strict=False)) if a != b),
            min(len(carga), len(canonica)),
        )
        raise CargaRecusadaError(
            f"a carga não é a forma canônica do esquema fechado: difere a partir do byte {byte} "
            f"({len(carga)} bytes; a forma canônica com estes nomes tem {len(canonica)})"
        )


def _confere_atributos(objeto: Objeto, caminho: str) -> None:
    """Exige que toda lista leve só `names` e que nenhum vetor leve atributo.

    A forma canônica também cobre isso, mas muda junto com o escritor: se ele passar a gravar um
    atributo, só esta conferência o vê.
    """
    if isinstance(objeto, Vetor):
        if objeto.atributos:
            raise CargaRecusadaError(f"{caminho}: folha com atributo {sorted(objeto.atributos)}")
    elif isinstance(objeto, Lista):
        if set(objeto.atributos) != {"names"}:
            raise CargaRecusadaError(f"{caminho}: atributos {sorted(objeto.atributos)}")
        for nome, item in zip(objeto.nomes(), objeto.itens, strict=True):
            _confere_atributos(item, f"{caminho}${nome}")


@dataclass(frozen=True)
class RespostaDoServidor:
    """O desfecho do envio. `dados` tem a forma da resposta de `ora_local` do roteiro R."""

    desfecho: DesfechoDoEnvio
    motivo: str = ""
    dados: dict[str, Any] | None = None
    segundos: float = 0.0
    # O status HTTP quando o servidor respondeu com um diferente de 200: o painel distingue o 500,
    # que o servidor dá para uma lista curta, da recusa genérica.
    status: int | None = None


def _vetor(objeto: Objeto, tipo: str, onde: str) -> Vetor:
    if not isinstance(objeto, Vetor) or objeto.tipo != tipo:
        raise RdsRecusadoError(f"{onde}: esperava vetor {tipo}")
    return objeto


def le_resposta(resposta: bytes, nomes: Sequence[str]) -> dict[str, Any]:
    """Extrai da resposta a matriz da ORA, os acertos e o universo, na forma do R local.

    Levanta `RdsRecusadoError` se faltar algo ou se a forma não for a esperada, e só ela: cada
    índice e cada conversão tem a guarda antes, porque `envia_ao_servidor` não levanta por
    resposta. Os p não são conferidos aqui; quem confere é
    `pathways.enriquecimento.ora_da_resposta`, como no modo local.

    `na_lista` é a contagem de nomes distintos nos acertos, e não `len(nomes)`. O servidor conta só
    os nomes da lista que estão em alguma via KEGG, a mesma regra do modo local, em que o roteiro
    faz `intersect(ora_vec, universo)`. Um acerto que não foi enviado é forma errada: a contagem
    só vale se os acertos forem nomes da lista.
    """
    objeto = le_rds(resposta)
    if not isinstance(objeto, Lista):
        raise RdsRecusadoError("a resposta não é uma lista")
    anal = objeto.campo("analSet")
    dados_do_conjunto = objeto.campo("dataSet")
    if not (isinstance(anal, Lista) and isinstance(dados_do_conjunto, Lista)):
        raise RdsRecusadoError("a resposta não traz analSet e dataSet como listas")
    universo = _vetor(dados_do_conjunto.campo("uniq.count"), "inteiro", "uniq.count").valores
    matriz = _vetor(anal.campo("ora.mat"), "real", "ora.mat")
    acertos = anal.campo("ora.hits")
    dim = matriz.atributos.get("dim")
    dimnames = matriz.atributos.get("dimnames")
    if not (
        len(universo) == 1
        and isinstance(universo[0], int)
        and isinstance(dim, Vetor)
        and dim.tipo == "inteiro"
        and len(dim.valores) == 2
        and all(isinstance(x, int) and x >= 0 for x in dim.valores)
        and isinstance(dimnames, Lista)
        and len(dimnames.itens) == 2
        and isinstance(acertos, Lista)
    ):
        raise RdsRecusadoError("a matriz da ORA não tem a forma esperada")
    n, k = (int(x) for x in dim.valores if isinstance(x, int))
    linhas = _vetor(dimnames.itens[0], "texto", "linhas da ORA").valores
    colunas = _vetor(dimnames.itens[1], "texto", "colunas da ORA").valores
    esperadas = ("total", "expected", "hits", "Raw p", "Holm p", "FDR")
    if tuple(colunas) != esperadas or len(linhas) != n or k != len(esperadas):
        raise RdsRecusadoError(f"colunas da ORA fora da forma: {colunas}")
    if len(matriz.valores) != n * k:
        raise RdsRecusadoError(
            f"a matriz da ORA tem {len(matriz.valores)} valores, e `dim` diz {n} por {k}"
        )
    por_via = dict(zip(acertos.nomes(), acertos.itens, strict=True))
    enviados = {n.casefold() for n in nomes}
    nos_acertos: set[str] = set()
    vias = []
    for i, via in enumerate(linhas):
        valores = [matriz.valores[i + j * n] for j in range(k)]
        if not isinstance(via, str) or not all(
            isinstance(v, float) and math.isfinite(v) for v in valores
        ):
            raise RdsRecusadoError(f"linha {i + 1} da ORA fora da forma")
        compostos = por_via.get(via)
        textos = (
            []
            if compostos is None
            else list(_vetor(compostos, "texto", f"acertos de {via}").valores)
        )
        for c in textos:
            if not isinstance(c, str) or c.casefold() not in enviados:
                raise RdsRecusadoError(f"acertos de {via}: um acerto que não foi enviado")
            nos_acertos.add(c.casefold())
        vias.append(
            {
                "via": via,
                "total": valores[0],
                "esperado": valores[1],
                "acertos": valores[2],
                "p": valores[3],
                "holm": valores[4],
                "fdr": valores[5],
                "compostos": sorted(str(c) for c in textos),
            }
        )
    return {
        "status": "ok",
        "biblioteca": BIBLIOTECA_EXTERNA,
        "universo": {"tipo": "biblioteca", "tamanho": universo[0]},
        "na_lista": len(nos_acertos),
        "vias": vias,
    }


class _Resposta(Protocol):
    status: int
    reason: str

    def getheader(self, nome: str, /) -> str | None: ...

    def read(self, n: int = -1, /) -> bytes: ...


class _Conexao(Protocol):
    def request(self, metodo: str, url: str, body: bytes, headers: dict[str, str], /) -> None: ...

    def getresponse(self) -> _Resposta: ...

    def close(self) -> None: ...


class _ConexaoHttps(http.client.HTTPSConnection):
    """A `HTTPSConnection` do `http.client`, com o socket TLS criado antes do aperto de mão.

    `wrap_socket` com o aperto de mão no construtor desacopla o socket cru e só devolve o TLS
    quando o aperto termina: durante ele, `self.sock` é um socket já sem descritor, e `_derruba`
    não tem o que derrubar. Contra um servidor que aceita e não responde, a thread da troca
    ficaria presa no aperto. Com o aperto feito depois, por `do_handshake`, a verificação é a
    mesma, porque o contexto e o nome são os mesmos.
    """

    def __init__(self, host: str, tempo_limite: float) -> None:
        self._contexto = ssl.create_default_context()
        super().__init__(host, timeout=tempo_limite, context=self._contexto)

    def connect(self) -> None:
        http.client.HTTPConnection.connect(self)
        tls = self._contexto.wrap_socket(
            self.sock, server_hostname=self.host, do_handshake_on_connect=False
        )
        self.sock = tls
        tls.do_handshake()


def _conexao_padrao(host: str, tempo_limite: float) -> _Conexao:
    """HTTPS com o certificado verificado pelo contexto padrão.

    `http.client` não segue redirecionamento: um 3xx chega como status e é recusado em `_troca`.
    """
    return _ConexaoHttps(host, tempo_limite)


def _corpo_multipart(carga: bytes) -> tuple[bytes, dict[str, str]]:
    fronteira = uuid.uuid4().hex
    corpo = (
        (
            f"--{fronteira}\r\n"
            'Content-Disposition: form-data; name="rds"; filename="tosend.rds"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("ascii")
        + carga
        + f"\r\n--{fronteira}--\r\n".encode("ascii")
    )
    cabecalhos = {
        "Content-Type": f"multipart/form-data; boundary={fronteira}",
        "Content-Length": str(len(corpo)),
    }
    return corpo, cabecalhos


def _troca(
    conexao: Callable[[str, float], _Conexao],
    corpo: bytes,
    cabecalhos: dict[str, str],
    aberta: dict[str, _Conexao],
) -> bytes | RespostaDoServidor:
    """Faz a ida e a volta, na thread da troca. Devolve os bytes da resposta ou o desfecho da falha.

    A conexão aberta fica em `aberta`, para que quem espera a possa derrubar (`_derruba`).
    """
    c: _Conexao | None = None
    try:
        c = aberta["conexao"] = conexao(HOST, TEMPO_LIMITE_S)
        c.request("POST", CAMINHO, corpo, cabecalhos)
        r = c.getresponse()
        if r.status != 200:
            return RespostaDoServidor(
                "servidor_recusou", f"o servidor respondeu {r.status} {r.reason}", status=r.status
            )
        tipo = (r.getheader("Content-Type") or "").split(";")[0].strip()
        if tipo != "application/octet-stream":
            return RespostaDoServidor("resposta_malformada", f"tipo de conteúdo {tipo!r}")
        return r.read(MAX_BYTES + 1)
    except ssl.SSLError as erro:
        return RespostaDoServidor("transporte_recusado", f"TLS recusado: {erro}")
    except (OSError, http.client.HTTPException) as erro:
        return RespostaDoServidor("sem_rede", f"{type(erro).__name__}: {erro}")
    finally:
        if c is not None:
            c.close()


def _derruba(aberta: dict[str, _Conexao]) -> None:
    """Derruba o socket da troca em voo, para a thread dela sair da espera.

    São dois gestos porque cada sistema solta com um. No Windows, `shutdown` não solta um `recv`
    bloqueado noutra thread, e o `close` do objeto também não quando há um arquivo aberto sobre o
    socket, como na leitura da resposta; o que solta é o `close` da camada de baixo, que fecha o
    descritor na hora. No Linux é o contrário: `shutdown` acorda o `recv`, e fechar o descritor
    não. Os dois juntos soltam nos dois sistemas.

    Antes de o socket existir (resolução do nome e conexão TCP) não há o que derrubar: a espera
    volta mesmo assim, e a thread termina sozinha no teto por operação (`TEMPO_LIMITE_S`).
    """
    sock = getattr(aberta.get("conexao"), "sock", None)
    if sock is None:
        return
    # Um socket que a thread já fechou ou que o servidor derrubou: nada a derrubar.
    with contextlib.suppress(OSError):
        sock.shutdown(socket.SHUT_RDWR)
    if isinstance(sock, socket.socket):
        with contextlib.suppress(OSError):
            _socket.socket.close(sock)


def envia_ao_servidor(
    carga: bytes,
    nomes: Sequence[str],
    *,
    conexao: Callable[[str, float], _Conexao] = _conexao_padrao,
    cancelado: Callable[[], bool] = lambda: False,
    relogio: Callable[[], float] = time.monotonic,
) -> RespostaDoServidor:
    """Envia `carga` ao endereço constante e lê a resposta. Não levanta por rede nem por resposta.

    A carga já passou por `confere_carga`; este passo não a refaz nem a altera, e o que sai são
    os bytes conferidos. A troca corre numa thread própria e esta função espera por ela
    consultando o cancelamento (ver `TEMPO_TOTAL_S`). Os dois desfechos de cancelamento dizem se
    o pedido pode ter saído, porque a tela diz ao usuário se os nomes chegaram ao servidor.
    """
    inicio = relogio()
    if cancelado():
        return RespostaDoServidor("cancelado", "cancelado antes do envio: nada saiu da máquina")
    corpo, cabecalhos = _corpo_multipart(carga)
    aberta: dict[str, _Conexao] = {}
    fim: list[bytes | RespostaDoServidor | BaseException] = []
    pronto = threading.Event()

    def troca() -> None:
        try:
            fim.append(_troca(conexao, corpo, cabecalhos, aberta))
        except BaseException as erro:  # devolvido a quem espera, que o levanta lá
            fim.append(erro)
        finally:
            pronto.set()

    threading.Thread(target=troca, name="envio-ao-metaboanalyst", daemon=True).start()
    esperas = 0
    interrompido: RespostaDoServidor | None = None
    while not pronto.wait(_INTERVALO_DA_ESPERA_S):
        esperas += 1
        if cancelado():
            interrompido = RespostaDoServidor(
                "cancelado", "cancelado durante o envio: o pedido pode ter chegado ao servidor"
            )
        elif esperas * _INTERVALO_DA_ESPERA_S >= TEMPO_TOTAL_S:
            interrompido = RespostaDoServidor(
                "sem_rede", f"o servidor não terminou de responder em {TEMPO_TOTAL_S:g} s"
            )
        if interrompido is not None:
            _derruba(aberta)
            # Um instante para a thread fechar a conexão, sem voltar a esperar o servidor.
            pronto.wait(1.0)
            return replace(interrompido, segundos=relogio() - inicio)
    segundos = relogio() - inicio
    resultado = fim[0]
    if isinstance(resultado, BaseException):
        raise resultado
    # Os ramos de falha também medem a espera: quem esperou até o tempo esgotar esperou de
    # verdade, e `segundos` igual a zero diria que não.
    if isinstance(resultado, RespostaDoServidor):
        return replace(resultado, segundos=segundos)
    if cancelado():
        return RespostaDoServidor(
            "cancelado",
            "cancelado durante o envio: o pedido chegou ao servidor, e a resposta foi descartada",
            segundos=segundos,
        )
    if len(resultado) > MAX_BYTES:
        return RespostaDoServidor(
            "resposta_malformada", f"resposta acima de {MAX_BYTES} bytes", segundos=segundos
        )
    try:
        dados = le_resposta(resultado, nomes)
    except RdsRecusadoError as erro:
        return RespostaDoServidor("resposta_malformada", str(erro), segundos=segundos)
    return RespostaDoServidor("ok", dados=dados, segundos=segundos)

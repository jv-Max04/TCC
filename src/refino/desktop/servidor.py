"""O servidor da API, embutido no processo do aplicativo de mesa.

As rotas de trabalho já executam em `threading.Thread` e devolvem `202`, então a janela e a API
podem partilhar processo, pool de threads e memória. Um processo separado acrescentaria
supervisão, encerramento ordenado e um segundo lugar aonde `REFINO_DB_URL` teria de chegar.

O servidor escuta só em `HOST`, e `enderecos_de_escuta` confere isso lendo a fonte por AST. A
prontidão vem de `GET /health`, e não de `uvicorn.Server.started`, que não é API documentada. A
sonda usa `urllib.request` e não o `httpx2` do projeto: é um único `GET` ao loopback, e importar um
cliente HTTP aqui só somaria tempo ao arranque da janela.
"""

from __future__ import annotations

import ast
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import uvicorn

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Callable

    from fastapi import FastAPI

_ABRIDOR: Final = urllib.request.build_opener(urllib.request.ProxyHandler({}))
"""Abridor HTTP sem proxy, para a sonda de prontidão.

O abridor padrão de `urllib.request.urlopen` lê `http_proxy` do ambiente. Numa máquina com proxy
corporativo, o `GET` a `127.0.0.1` iria para o proxy, a sonda esgotaria o teto e a mensagem
culparia o servidor. Loopback não passa por proxy."""

HOST: Final = "127.0.0.1"
"""O único endereço em que este módulo escuta, literal e sem caminho para ser outro.

Os dados servidos são clínicos e identificáveis. Um parâmetro de host convidaria a passar
`0.0.0.0`; uma constante literal não deixa onde escrever outro valor."""

ROTA_DE_PRONTIDAO: Final = "/health"
"""A rota que `api/app.py` serve como sinal de prontidão."""

TETO_DE_ARRANQUE_S: Final = 15.0
"""Teto, em segundos, para a sonda de prontidão. É escolhido, e não derivado de medição.

Generoso de propósito: errar para mais custa uma espera que ninguém nota, e errar para menos
transforma máquina lenta em falha de arranque. Não decide aprovação de nada."""

_LOOPBACK: Final = frozenset({"127.0.0.1", "::1"})
"""Os únicos endereços aceitos como escuta não pública.

É uma lista de permissão, e não de proibição, para que um endereço novo, como um de rede local,
seja acusado em vez de passar. Acusar demais custa uma linha aqui; acusar de menos expõe dado
clínico na rede."""

CALCULADO: Final = "<calculado>"
"""Sentinela para endereço que o extrator vê mas não consegue resolver.

Um host como `os.environ.get("REFINO_HOST", HOST)` não é literal. Reportá-lo como sentinela faz o
portão acusar o que não consegue provar, em vez de omiti-lo. O extrator não avalia a expressão; só
registra que ela existe."""


class ServidorNaoSubiuError(RuntimeError):
    """O servidor não respondeu na rota de prontidão dentro do teto.

    É causa 'ambiente' no vocabulário de `interface/cliente.py`: quem opera não pode consertar. A
    mensagem
    nomeia a porta porque disputa de porta é o modo de falha mais comum.
    """


class PortaDisputadaError(ServidorNaoSubiuError):
    """O `bind` foi recusado: outro processo tomou a porta entre a escolha e a ligação.

    Herda de `ServidorNaoSubiuError` porque uma disputa de porta é um servidor que não subiu, e
    `arranque.main` traduz os dois da mesma forma. Existe à parte para `sobe` tentar outra porta só
    nesta causa: um teto de prontidão esgotado também levanta `ServidorNaoSubiuError`, e repetir
    nele gastaria o orçamento duas vezes.

    A assinatura de um `bind` recusado é o `uvicorn` chamar `sys.exit(STARTUP_FAILURE)`, que chega
    a `causa_da_morte()` como `SystemExit`.
    """


def enderecos_de_escuta(fonte: str) -> set[str]:
    """Todos os endereços em que a fonte dada pode fazer o servidor escutar, lidos por AST.

    Recebe a fonte e não o caminho, para o teste plantar a violação direto no argumento.

    Colhe três formas e resolve `HOST` quando ele aparece por nome:
    - a constante de módulo `HOST`;
    - o argumento nomeado `host`, como em `uvicorn.Config`;
    - o primeiro elemento da tupla passada a um `bind`.

    O que não consegue resolver vira `CALCULADO` e é reportado. Um `uvicorn.run(app, "0.0.0.0",
    porta)` posicional fica fora: este código não usa essa forma, e se passar a usar o ramo tem de
    ser acrescentado.
    """
    arvore = ast.parse(fonte)
    constante = _constante_de_host(arvore)
    achados: set[str] = set() if constante is None else {constante}

    for no in ast.walk(arvore):
        if not isinstance(no, ast.Call):
            continue
        for kw in no.keywords:
            if kw.arg == "host":
                achados.add(_resolve_endereco(kw.value, constante))
        alvo = no.func
        if isinstance(alvo, ast.Attribute) and alvo.attr == "bind" and no.args:
            primeiro = no.args[0]
            if isinstance(primeiro, ast.Tuple) and primeiro.elts:
                achados.add(_resolve_endereco(primeiro.elts[0], constante))
            else:
                # `bind(algo)` com `algo` não literal: endereço que não se pode ler.
                achados.add(CALCULADO)
    return achados


def _constante_de_host(arvore: ast.AST) -> str | None:
    """O valor literal da constante `HOST` na árvore dada, ou `None`."""
    for no in ast.walk(arvore):
        if not isinstance(no, ast.AnnAssign | ast.Assign):
            continue
        alvos = [no.target] if isinstance(no, ast.AnnAssign) else no.targets
        if (
            any(isinstance(a, ast.Name) and a.id == "HOST" for a in alvos)
            and isinstance(no.value, ast.Constant)
            and isinstance(no.value.value, str)
        ):
            return no.value.value
    return None


def _resolve_endereco(no: ast.expr, constante: str | None) -> str:
    """O endereço que `no` denota: o literal, o valor de `HOST`, ou `CALCULADO`."""
    if isinstance(no, ast.Constant) and isinstance(no.value, str):
        return no.value
    if isinstance(no, ast.Name) and no.id == "HOST" and constante is not None:
        return constante
    return CALCULADO


def escuta_fora_do_loopback(fonte: str) -> set[str]:
    """O que a fonte dada pode escutar fora do loopback. Vazio é o desfecho correto.

    Subtrai uma lista de permissão, em vez de intersectar com uma de proibição, para que um endereço
    novo seja acusado em vez de passar.
    """
    return enderecos_de_escuta(fonte) - _LOOPBACK


def porta_livre() -> int:
    """Uma porta que o sistema declara livre em `HOST`, agora.

    Entre este `close` e o `bind` do `uvicorn`, outro processo pode tomar a porta. Essa corrida não
    se fecha aqui; `sobe` a trata tentando outra porta quando o chamador não fixou nenhuma.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def _porta_esta_livre(porta: int) -> bool:
    """Se o `uvicorn` conseguiria ligar-se a `porta` em `HOST` neste instante.

    `SO_REUSEADDR` reproduz a opção que o `uvicorn` põe nos sockets dele. Sem ela, a sonda recusaria
    uma porta em `TIME_WAIT`, em que o `uvicorn` ligaria sem problema, e o arranque reportaria
    disputa de porta onde não há. No Linux, a opção não deixa ligar por cima de um socket em
    `LISTEN`, então uma porta com servidor de pé continua a dar `False`.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((HOST, porta))
        except OSError:
            return False
        return True


@dataclass
class ServidorEmbutido:
    """Um `uvicorn` a servir a API em loopback, numa thread do processo do aplicativo.

    Não é gestor de contexto: o arranque sobe o servidor, e ele vive enquanto a janela viver.
    `desliga` serve ao teste e ao encerramento ordenado, que a chamam explicitamente.
    """

    porta: int
    _servidor: uvicorn.Server
    _thread: threading.Thread = field(repr=False)
    _falha: list[BaseException] = field(default_factory=list, repr=False)
    """Caixa de uma posição, escrita pela thread de serviço e lida pela chamadora.

    Uma lista criada antes da thread é o objeto mais simples que as duas veem sem sincronização
    explícita, e a escrita é única e final."""

    @property
    def base(self) -> str:
        """A URL base, no formato que `REFINO_API_BASE` espera."""
        return f"http://{HOST}:{self.porta}"

    def causa_da_morte(self) -> BaseException | None:
        """A exceção que derrubou a thread de serviço, se houve uma.

        O `uvicorn` chama `sys.exit(STARTUP_FAILURE)` quando o `bind` falha, e um `SystemExit` que
        escapa de uma thread vira traceback cru no `stderr`. Retida aqui, ela vira causa nomeada
        dentro de `ServidorNaoSubiuError`.
        """
        return self._falha[0] if self._falha else None

    def esta_vivo(self) -> bool:
        """Se a thread que serve continua de pé.

        Separado da prontidão: o servidor pode responder a `/health` e morrer depois.
        """
        return self._thread.is_alive()

    def desliga(self, *, teto_s: float = 10.0) -> bool:
        """Sinaliza a saída, junta a thread e devolve se ela de fato parou.

        Idempotente: o `finally` de um teste e o encerramento da janela podem coincidir. Uma thread
        que ignore `should_exit` fica viva com a porta ligada, e o valor de retorno é o único jeito
        de quem chamou saber. Não levanta, porque é chamada de `finally`, e uma exceção ali
        esconderia a causa real.
        """
        self._servidor.should_exit = True
        self._thread.join(timeout=teto_s)
        return not self._thread.is_alive()


def _sonda_prontidao(
    base: str,
    *,
    teto_s: float,
    vivo: Callable[[], bool],
    causa: Callable[[], BaseException | None] = lambda: None,
) -> None:
    """Espera a rota de prontidão responder, ou levanta.

    Desiste cedo quando a thread morre: thread morta não vai responder, e esperar o teto inteiro só
    atrasaria a mensagem de erro. `causa` é opcional; quando dada, a mensagem de morte nomeia o que
    matou a thread.
    """
    limite = time.monotonic() + teto_s
    ultima: Exception | None = None
    while time.monotonic() < limite:
        if not vivo():
            morte = causa()
            porque = f" Causa: {morte!r}" if morte is not None else ""
            raise ServidorNaoSubiuError(
                f"a thread do servidor morreu antes de {base}{ROTA_DE_PRONTIDAO} responder.{porque}"
            )
        try:
            with _ABRIDOR.open(f"{base}{ROTA_DE_PRONTIDAO}", timeout=1.0) as resposta:
                codigo = int(resposta.status)
        except OSError as erro:
            # Um único `except` cobre os três casos em que a sonda continua: a porta ainda não
            # aceita conexão (`URLError`), a rota responde `>= 400` (`HTTPError`), ou a porta aceita
            # e não responde (`TimeoutError`, subclasse de `OSError` que não passa por `URLError`).
            # O terceiro é o estado de uma porta disputada, com um socket em `listen` sem leitor.
            ultima = erro
        else:
            if codigo == 200:
                return
            # Um `2xx` que não é `200` responde sem declarar prontidão; aceitá-lo abriria a janela
            # contra um servidor ainda a arrancar.
            ultima = RuntimeError(f"{ROTA_DE_PRONTIDAO} devolveu {codigo}")
        time.sleep(0.05)
    raise ServidorNaoSubiuError(
        f"{base}{ROTA_DE_PRONTIDAO} não respondeu em {teto_s:.0f} s. Última causa: {ultima}"
    )


def sobe(
    fabrica: Callable[[], FastAPI],
    *,
    porta: int | None = None,
    teto_s: float = TETO_DE_ARRANQUE_S,
) -> ServidorEmbutido:
    """Constrói a aplicação, sobe-a em loopback e só devolve quando `/health` responde.

    Recebe a fábrica e não a aplicação para a construção correr na thread chamadora: uma exceção
    nela sobe síncrona e com a pilha inteira, em vez de aparecer como teto de prontidão esgotado.

    `porta=None` escolhe uma porta livre e tenta outra vez se perder a corrida de `porta_livre`. Uma
    `porta` fixada não é trocada: quem a fixou quer aquela, e trocá-la esconderia a disputa.
    """
    app = fabrica()

    if porta is not None:
        if not _porta_esta_livre(porta):
            raise ServidorNaoSubiuError(
                f"a porta {porta} já está em uso em {HOST}. Outro programa está escutando nela; "
                f"escolha outra porta ou feche o que está lá"
            )
        return _serve(app, porta, teto_s=teto_s)

    escolhida = porta_livre()
    inicio = time.monotonic()
    try:
        return _serve(app, escolhida, teto_s=teto_s)
    except PortaDisputadaError:
        # A corrida entre `porta_livre` e o `bind` do uvicorn foi perdida. Uma segunda tentativa,
        # e só uma: um laço aqui esconderia uma máquina sem portas atrás de uma espera longa. O
        # orçamento é o que resta do teto, e não outro teto inteiro.
        resta = max(0.5, teto_s - (time.monotonic() - inicio))
        return _serve(app, porta_livre(), teto_s=resta)


def _serve(app: FastAPI, porta: int, *, teto_s: float) -> ServidorEmbutido:
    """Sobe o `uvicorn` na porta dada e espera a prontidão.

    Nenhuma exceção escapa da thread de serviço. `uvicorn.Server.run` chama
    `sys.exit(STARTUP_FAILURE)` quando o `bind` falha, e um `SystemExit` que sobe de uma thread não
    mata o processo: imprime traceback no `stderr`, e a thread morre sem avisar quem chamou. No
    `pytest`, isso vira `PytestUnhandledThreadExceptionWarning`, que com `filterwarnings = error`
    reprova a sessão pelo motivo errado.
    """
    config = uvicorn.Config(app, host=HOST, port=porta, log_level="warning")
    servidor = uvicorn.Server(config)
    falha: list[BaseException] = []

    def corre() -> None:
        """Corre o servidor e RETÉM a exceção, em vez de a deixar escapar da thread."""
        try:
            servidor.run()
        except BaseException as erro:
            # Largo de propósito: na fronteira de uma thread não há quem apanhe acima, e o que
            # não for retido aqui vira traceback solto. A exceção vai para a caixa e a mensagem a
            # nomeia.
            falha.append(erro)

    thread = threading.Thread(target=corre, name=f"refino-api-{porta}", daemon=True)
    thread.start()
    embutido = ServidorEmbutido(porta=porta, _servidor=servidor, _thread=thread, _falha=falha)
    try:
        _sonda_prontidao(
            embutido.base,
            teto_s=teto_s,
            vivo=embutido.esta_vivo,
            causa=embutido.causa_da_morte,
        )
    except ServidorNaoSubiuError as erro:
        parou = embutido.desliga()
        if isinstance(embutido.causa_da_morte(), SystemExit):
            # Só esta causa justifica outra porta: é a assinatura de um `bind` recusado.
            raise PortaDisputadaError(str(erro)) from erro
        if not parou:
            # A thread sobreviveu ao `desliga` e continua ligada à porta. Tentar outra porta
            # deixaria a primeira a servir sem que ninguém saiba o número, o que é pior que falhar.
            raise ServidorNaoSubiuError(
                f"{erro} E a thread anterior não parou: a porta {porta} continua ligada"
            ) from erro
        raise
    return embutido

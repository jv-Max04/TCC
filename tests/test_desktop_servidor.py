"""O servidor embutido do aplicativo de mesa.

## O que estes testes alcançam, e o recorte é a seção inteira

Eles exercitam **sockets de verdade** e um `uvicorn` de verdade, em loopback, no processo do
`pytest`. Não é duplo: a pergunta é sobre o socket, e um duplo de socket responderia sobre o
duplo.

**O que eles NÃO alcançam:** o executável congelado. Nada aqui prova que o `--onedir` arranca — isso
corre em Windows, e não entra em `make check`, porque o PyInstaller dentro do WSL produz binário
Linux.

**E não alcançam firewall.** Que `bind` em `127.0.0.1` não dispara aviso do Windows é medido na
primeira execução do binário, não aqui.
"""

from __future__ import annotations

import socket
import sys
import urllib.request
from pathlib import Path
from typing import Final

import pytest
from _pytest.outcomes import Skipped
from fastapi import FastAPI

from refino.api.app import build_app
from refino.desktop import servidor
from refino.desktop.servidor import (
    CALCULADO,
    HOST,
    ROTA_DE_PRONTIDAO,
    ServidorEmbutido,
    ServidorNaoSubiuError,
    enderecos_de_escuta,
    escuta_fora_do_loopback,
    porta_livre,
    sobe,
)
from tests.conftest import REPO

pytestmark = pytest.mark.integration

FONTE: Final[Path] = REPO / "src" / "refino" / "desktop" / "servidor.py"

porta_livre_real = porta_livre
"""Referência à função original, para restaurar depois de um espião a substituir."""


def _app_minima() -> FastAPI:
    """Uma aplicação com a rota de prontidão e nada mais.

    **Separada de `build_app` de propósito:** a maioria destes testes é sobre o socket, e amarrá-los
    à aplicação real faria uma mudança de rota reprovar um teste de rede. O teste que precisa da
    aplicação real usa-a, e é um só.
    """
    app = FastAPI()

    @app.get(ROTA_DE_PRONTIDAO)
    def prontidao() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _endereco_externo() -> str | None:
    """O IPv4 não-loopback desta máquina, ou `None`.

    **A primeira versão dependia de ROTA e o `skip` mentia por isso.** Ela usava só o truque do
    socket UDP — `connect(("8.8.8.8", 80))` sem enviar pacote — e numa máquina **sem rota de saída**
    aquele `connect` falha. O `None` resultante fazia o `skip` disparar com a mensagem *"máquina sem
    endereço IPv4 não-loopback"*, quando o que faltava era a **rota**. Marcador a absorver outra
    causa.

    Agora tenta **duas** vias: o resolvedor de nomes, que não precisa de rota, e depois o truque da
    rota. **E o conserto NÃO é independência de rota — isso foi medido e é falso nesta máquina.**
    `getaddrinfo(gethostname())` devolve `127.0.1.1` aqui, que é loopback; só o truque da rota dá
    endereço útil. Afirmar independência seria a mesma classe de defeito que este conserto endereça.

    **O que se consertou é a MENSAGEM do `skip`.** Ela nomeia a **disjunção** do que foi tentado —
    *"nem pelo resolvedor de nomes nem pela rota"* — em vez de afirmar uma causa que pode não ser a
    verdadeira. O marcador deixa de absorver causa diferente da nomeada porque a nomeada passou a
    ser *"nenhuma das duas vias respondeu"*, que é exactamente o que se sabe.
    """
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        infos = []
    for info in infos:
        endereco = str(info[4][0])
        if not endereco.startswith("127."):
            return endereco

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
        except OSError:
            return None
        endereco = str(sock.getsockname()[0])
    return None if endereco.startswith("127.") else endereco


# ═══════════════════════════════════════════════════════════ a porta


def test_porta_livre_devolve_porta_que_aceita_bind() -> None:
    """E duas chamadas seguidas devolvem portas diferentes.

    A segunda metade é o que distingue *"escolhe uma porta"* de *"devolve sempre a mesma"*, e o
    segundo passaria numa asserção que só conferisse que a porta é ligável.
    """
    primeira = porta_livre()
    segunda = porta_livre()

    assert primeira != segunda, (
        f"as duas chamadas devolveram a mesma porta {primeira} — o sistema não está a escolher, "
        "ou a função está a devolver constante"
    )
    for porta in (primeira, segunda):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((HOST, porta))  # levanta se não estiver livre


# ═══════════════════════════════════════════════════════════ o par causal do host


# **O literal, e NÃO `HOST` importado do módulo sob teste.** A primeira versão desta asserção era
# `== {HOST}`, e a revisão de código mediu que ela é **autorreferencial**: plantado
# `HOST = "192.168.1.50"`, o extrator devolvia `{"192.168.1.50"}` e `{HOST}` valia o mesmo — as duas
# asserções ficavam verdes sobre um servidor exposto à rede local.
#
# Uma asserção que compara o código com ele mesmo não é asserção de presença; é tautologia com
# aparência de presença. O literal é o que a torna uma medida.
LOOPBACK_LITERAL: Final = "127.0.0.1"


def test_o_modulo_escuta_exatamente_no_loopback() -> None:
    """**Asserção contra o LITERAL, e é isso que a torna não-tautológica.**

    `escuta_fora_do_loopback(...) == set()` sozinho passaria sobre uma fonte de que o extrator não
    casasse forma nenhuma. E `== {HOST}` — a primeira versão — passaria sobre **qualquer** valor de
    `HOST`, porque compara o módulo consigo mesmo. O que protege é a igualdade ao literal.
    """
    fonte = FONTE.read_text(encoding="utf-8")

    assert enderecos_de_escuta(fonte) == {LOOPBACK_LITERAL}, (
        f"os endereços de escuta deste módulo são {sorted(enderecos_de_escuta(fonte))}, e têm de "
        f"ser exatamente ['{LOOPBACK_LITERAL}'] — o `[CA-12.5]` existe porque os dados são clínicos"
    )
    assert escuta_fora_do_loopback(fonte) == set()
    assert HOST == LOOPBACK_LITERAL, (
        f"o módulo declara HOST = {HOST!r}, e este arquivo mede contra {LOOPBACK_LITERAL!r}: se "
        "mudarem o host de propósito, esta linha é a que obriga a decisão a passar por aqui"
    )


@pytest.mark.parametrize(
    ("antigo", "novo", "esperado"),
    [
        # **A âncora da linha seguinte não é zelo: o literal `HOST: Final = "127.0.0.1"` aparece
        # DUAS vezes na fonte** — a atribuição real, e um exemplo na tabela da docstring de
        # `enderecos_de_escuta`. Sem o começo da docstring no padrão, `replace(…, 1)` cairia na
        # DOCUMENTAÇÃO, o `HOST` real ficaria em loopback, e o portão não teria o que acusar. Quem
        # achou isto foi a asserção de aterragem abaixo, na primeira execução do par causal.
        pytest.param(
            f'HOST: Final = "{HOST}"\n"""O único',
            'HOST: Final = "0.0.0.0"\n"""O único',
            "0.0.0.0",
            id="constante",
        ),
        pytest.param("host=HOST, port=porta", 'host="0.0.0.0", port=porta', "0.0.0.0", id="kwarg"),
        # **Âncora outra vez, e pela terceira vez neste arquivo:** `sock.bind((HOST, 0))` passou a
        # aparecer DUAS vezes na fonte quando a tabela da docstring de `enderecos_de_escuta` ganhou
        # aquele exemplo. Quem achou foi a asserção de aterragem, na primeira execução depois do
        # conserto.
        pytest.param(
            "sock.bind((HOST, 0))\n        return int(sock.getsockname()[1])",
            'sock.bind(("", 0))\n        return int(sock.getsockname()[1])',
            "",
            id="bind_vazio",
        ),
        # ── as DUAS plantas que a revisão de código descobriu que passavam ──
        #
        # Um endereço de LAN: não estava na lista negra de cinco nomes, e a asserção da árvore limpa
        # comparava contra o `HOST` importado. Portão verde sobre servidor exposto à rede local.
        pytest.param(
            f'HOST: Final = "{LOOPBACK_LITERAL}"\n"""O único',
            'HOST: Final = "192.168.1.50"\n"""O único',
            "192.168.1.50",
            id="lan",
        ),
        # Um host vindo do ambiente: não é `ast.Constant`, então a versão anterior simplesmente NÃO
        # o colhia — e produzia exactamente o mesmo conjunto que o código correto.
        pytest.param(
            "host=HOST, port=porta",
            'host=os.environ.get("REFINO_HOST", HOST), port=porta',
            CALCULADO,
            id="calculado",
        ),
    ],
)
def test_o_portao_do_host_acusa_cada_forma_de_escuta_publica(
    antigo: str, novo: str, esperado: str
) -> None:
    """As três formas que o extrator colhe, plantadas uma a uma na fonte real.

    **Cada planta AFIRMA que aterrou antes de o desfecho ser lido.** Substituição de texto não
    verificada tratada como aplicada já fez arneses reportarem desfecho que não mediram — e aqui uma
    planta que não casasse produziria exatamente o desfecho que se quer ver: nenhum endereço
    público, portão verde.
    """
    fonte = FONTE.read_text(encoding="utf-8")
    assert fonte.count(antigo) == 1, (
        f"a planta NÃO PODE ATERRAR: {fonte.count(antigo)} ocorrências de {antigo!r} na fonte. "
        "O módulo mudou e este par causal deixou de medir o que a razão dele descreve"
    )
    plantada = fonte.replace(antigo, novo, 1)

    assert escuta_fora_do_loopback(plantada) == {esperado}, (
        f"o portão não acusou a escuta pública plantada como {novo!r} — ele não vê esta forma, e a "
        "asserção da árvore limpa é vácua para ela"
    )


# ═══════════════════════════════════════════════════════════ o servidor de verdade


def test_o_servidor_sobe_e_a_rota_de_prontidao_responde_pelo_loopback() -> None:
    """O caminho feliz, com a aplicação REAL — e é o único teste que a usa.

    Prova três coisas de uma vez: `build_app` serve `/health`, o `uvicorn` embutido a serve em
    loopback, e a URL que `sobe` devolve é a que responde. A terceira é a que importa para o
    arranque, porque é ela que vai para `REFINO_API_BASE`.
    """
    embutido = sobe(build_app)
    try:
        assert embutido.base == f"http://{HOST}:{embutido.porta}"
        with urllib.request.urlopen(f"{embutido.base}{ROTA_DE_PRONTIDAO}", timeout=5.0) as resposta:
            assert resposta.status == 200, resposta.read()
        assert embutido.esta_vivo()
    finally:
        embutido.desliga()


def test_o_servidor_nao_escuta_no_endereco_externo_da_maquina() -> None:
    """A prova de runtime do que o portão de AST afirma sobre a fonte.

    **A asserção é POSITIVA, e é por isso que não precisa de rede:** com o servidor ligado só a
    `127.0.0.1`, ligar um segundo socket a `(externo, mesma_porta)` **funciona**. Ligado a
    `0.0.0.0`, o mesmo `bind` daria `EADDRINUSE`. Nenhum pacote sai da máquina.

    **Não é determinística, e a ressalva vale a linha:** a porta foi escolhida como livre em
    `127.0.0.1`, e nada garante que outro processo não a tenha no endereço externo. É raro e é
    possível — se este teste ficar intermitente, esta é a causa, e não o mecanismo sob teste.
    """
    externo = _endereco_externo()
    if externo is None:
        pytest.skip(
            "sem IPv4 não-loopback: nem pelo resolvedor de nomes nem pela rota — não há o que "
            "medir aqui"
        )

    embutido = sobe(_app_minima)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((externo, embutido.porta))  # levanta se o servidor estivesse em 0.0.0.0
    finally:
        embutido.desliga()


def _aceita_conexao(porta: int) -> bool:
    """Se alguém está a servir em `(HOST, porta)` agora.

    **A pergunta é esta, e não «a porta aceita `bind`».** O primeiro rascunho deste par causal
    usava `bind`, e ele reprovou na primeira execução com `EADDRINUSE`: a sonda de prontidão abriu
    conexão, então a porta fica em `TIME_WAIT` depois do encerramento, e um `bind` nu — sem
    `SO_REUSEADDR`, que é o que o `uvicorn` põe — recusa uma porta que não tem servidor nenhum.

    A medição estava certa e a **pergunta** era outra: `bind` mede *"posso ocupar esta porta"*, e o
    que o teste precisa é *"há alguém a servir aqui"*.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((HOST, porta))
        except OSError:
            return False
        return True


def test_desliga_para_de_servir_e_e_idempotente() -> None:
    """**Par causal de duas execuções que diferem só pelo `desliga`.**

    A idempotência não é conveniência — o encerramento da janela e o `finally` de um teste podem
    coincidir, e um segundo `desliga` a levantar transformaria limpeza em falha.

    A âncora do desfecho **antes** é o que impede a vacuidade: sem ela, um `sobe` que nunca tivesse
    servido daria «não aceita conexão» depois, e o teste passaria sem medir nada.
    """
    embutido = sobe(_app_minima)
    porta = embutido.porta

    assert _aceita_conexao(porta), (
        f"âncora: com o servidor de pé, {HOST}:{porta} tem de aceitar conexão — sem isto o "
        "desfecho «não aceita» depois do `desliga` não separa nada"
    )

    embutido.desliga()
    embutido.desliga()  # idempotente: levanta se não for

    assert not embutido.esta_vivo(), "a thread do servidor continua viva depois de `desliga`"
    assert not _aceita_conexao(porta), (
        f"{HOST}:{porta} continua a aceitar conexão depois de `desliga` — o servidor não parou"
    )


# ═══════════════════════════════════════════════════════════ os quatro desfechos de arranque


def test_desfecho_porta_ocupada_levanta_nomeando_a_porta() -> None:
    """Com a porta fixada já ocupada, o erro nomeia a porta em vez de apontar outro lugar."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ocupante:
        ocupante.bind((HOST, 0))
        ocupante.listen(1)
        porta = int(ocupante.getsockname()[1])

        with pytest.raises(ServidorNaoSubiuError, match=str(porta)):
            sobe(_app_minima, porta=porta)


def test_desfecho_a_fabrica_levanta_e_a_excecao_original_sobe() -> None:
    """**A fábrica corre na thread chamadora, e por isso a exceção não vira teto de prontidão.**

    Se a construção corresse na thread de serviço, este caso e o de porta ocupada dariam a
    MESMA saída — `ServidorNaoSubiuError` — para causas completamente diferentes, e cada
    desfecho tem de produzir mensagem distinta. O teste afirma que sobe a
    exceção da fábrica, e **não** a nossa.
    """

    class DefeitoDaFabricaError(RuntimeError):
        pass

    def fabrica_que_levanta() -> FastAPI:
        raise DefeitoDaFabricaError("a aplicação não pôde ser construída")

    with pytest.raises(DefeitoDaFabricaError):
        sobe(fabrica_que_levanta)


def test_desfecho_a_thread_morre_e_a_sonda_desiste_cedo() -> None:
    """O quarto desfecho, e é o que mais facilmente ficaria inerte.

    Ele dispara por **ausência** — a thread não está viva — e não por presença de condição, então
    não há o que plantar nele. É o ramo a exercitar primeiro.

    A sonda é chamada direto, com `vivo` a devolver `False`: forçar uma thread real a morrer no
    instante certo seria um teste sobre o escalonador, que é a forma intermitente.
    """
    with pytest.raises(ServidorNaoSubiuError, match="morreu"):
        servidor._sonda_prontidao(
            f"http://{HOST}:1",
            teto_s=5.0,
            vivo=lambda: False,
        )


def test_a_sonda_esgota_o_teto_quando_ninguem_responde() -> None:
    """O desfecho de teto, distinto do de thread morta — e as mensagens dizem coisas diferentes.

    Sem este teste os dois ramos de `_sonda_prontidao` seriam indistinguíveis em contagem: os dois
    levantam `ServidorNaoSubiuError`, e só a mensagem separa *"morreu"* de *"não respondeu"*.
    """
    porta = porta_livre()
    with pytest.raises(ServidorNaoSubiuError, match="não respondeu"):
        servidor._sonda_prontidao(
            f"http://{HOST}:{porta}",
            teto_s=0.3,
            vivo=lambda: True,
        )


def test_uma_porta_fixada_e_livre_e_usada_como_pedida() -> None:
    """A outra direção do caso de porta ocupada, e ela é o ramo `True` de `_porta_esta_livre`.

    Sem este teste aquela função teria **um** desfecho exercitado de dois, e o `False` é o que
    levanta — ou seja, o caminho felizmente silencioso ficaria inerte.
    """
    pedida = porta_livre()
    embutido = sobe(_app_minima, porta=pedida)
    try:
        assert embutido.porta == pedida, (
            f"pedi a porta {pedida} e o servidor subiu na {embutido.porta} — porta fixada não pode "
            "ser reescolhida em silêncio, senão a disputa fica escondida"
        )
    finally:
        embutido.desliga()


def test_uma_rota_de_prontidao_que_responde_2xx_sem_ser_200_nao_conta_como_pronta() -> None:
    """O ramo do `else` que não é `200`, e ele é o único caminho para lá.

    `urlopen` **levanta** em `>= 400`, então um `503` chega ao ramo de `URLError` e não aqui. O que
    chega aqui é `2xx` que não é `200` — um `204`. A rota existe, responde, e **não** declara
    prontidão; tratar isso como pronta abriria a janela contra um servidor a arrancar.
    """

    def app_que_responde_204() -> FastAPI:
        app = FastAPI()

        @app.get(ROTA_DE_PRONTIDAO, status_code=204)
        def prontidao() -> None:
            return None

        return app

    with pytest.raises(ServidorNaoSubiuError, match="devolveu 204"):
        sobe(app_que_responde_204, teto_s=0.6)


def test_a_corrida_da_porta_livre_e_tentada_outra_vez(monkeypatch: pytest.MonkeyPatch) -> None:
    """A segunda tentativa quando a corrida entre `porta_livre` e o `bind` é perdida.

    **A corrida é real e não é reproduzível por espera:** entre o `close` de `porta_livre` e o
    `bind` do `uvicorn` cabe outro processo, e esperar por isso seria um teste sobre o escalonador.
    O que se substitui é **`porta_livre`**, para a primeira chamada devolver uma porta que já tem
    dono — que é exactamente o estado que a corrida produz.

    E o teste afirma a porta FINAL: sem isso, um `sobe` que ignorasse a segunda tentativa e subisse
    na primeira porta por acaso passaria.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ocupante:
        ocupante.bind((HOST, 0))
        ocupante.listen(1)
        ocupada = int(ocupante.getsockname()[1])
        livre = porta_livre()
        assert ocupada != livre, "âncora: as duas portas do par têm de diferir"

        entregas = iter([ocupada, livre])
        monkeypatch.setattr(servidor, "porta_livre", lambda: next(entregas))

        embutido = sobe(_app_minima, teto_s=1.0)
        try:
            assert embutido.porta == livre, (
                f"a segunda tentativa não aconteceu: subiu na {embutido.porta}, e a porta livre "
                f"entregue pela segunda chamada era a {livre}"
            )
        finally:
            embutido.desliga()


def test_a_causa_da_morte_da_thread_e_nomeada_na_mensagem() -> None:
    """O `SystemExit` do `uvicorn` chega ao usuário como causa, e não como traceback solto.

    **A asserção é sobre a MENSAGEM, e é por isso que ela vale.** Retê-la na thread já impediria o
    traceback; o que este teste protege é a metade seguinte — que a exceção retida **apareça**. Uma
    caixa que engolisse a exceção em silêncio passaria numa asserção que só olhasse o `stderr`.

    `_serve` é chamado direto e não `sobe`: `sobe` **captura** este erro e tenta outra porta, então
    pela porta da frente a mensagem nunca surge. O mecanismo age numa camada e o registro nomeia
    outra.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ocupante:
        ocupante.bind((HOST, 0))
        ocupante.listen(1)
        ocupada = int(ocupante.getsockname()[1])

        with pytest.raises(ServidorNaoSubiuError, match="Causa:") as capturado:
            servidor._serve(_app_minima(), ocupada, teto_s=3.0)

    assert "morreu" in str(capturado.value), str(capturado.value)
    assert "SystemExit" in str(capturado.value), (
        f"a causa retida não foi nomeada na mensagem: {capturado.value}"
    )


def test_um_bind_calculado_e_reportado_como_sentinela_e_nao_ignorado() -> None:
    """**Este teste INVERTEU depois da revisão, e a inversão é o conserto.**

    Ele afirmava `== set()`: um `bind` com o endereço numa variável **não era colhido**, e a
    docstring chamava isso de recorte declarado. A revisão mediu o custo do recorte —
    `host=os.environ.get("REFINO_HOST", HOST)` produzia o mesmo conjunto que o código correto — e
    omitir deixou de ser aceitável.

    Agora o extrator reporta `CALCULADO`, que **não** está na lista de permissão. O portão passa a
    acusar o que não consegue provar, em vez de o esconder.
    """
    cabeca = 'import socket\ns = socket.socket()\nendereco = "0.0.0.0"\n'

    literal = cabeca + 's.bind(("0.0.0.0", 0))\n'
    assert enderecos_de_escuta(literal) == {"0.0.0.0"}, (
        "âncora: com o endereço em TUPLA LITERAL o extrator tem de o colher"
    )

    calculado = cabeca + "s.bind(endereco)\n"
    assert enderecos_de_escuta(calculado) == {CALCULADO}, (
        "endereço calculado tem de virar sentinela; devolver conjunto vazio é o ponto cego que a "
        "revisão de código mediu"
    )
    assert escuta_fora_do_loopback(calculado) == {CALCULADO}, (
        "o sentinela tem de contar como escuta FORA do loopback — é o que faz o portão acusar"
    )


def test_o_tipo_devolvido_carrega_a_porta_e_a_base() -> None:
    """`ServidorEmbutido` é o contrato entre o servidor e o arranque.

    O arranque lê `base` para exportar `REFINO_API_BASE`. Um teste que só confirmasse que o servidor
    sobe não protegeria essa travessia.
    """
    embutido = sobe(_app_minima)
    try:
        assert isinstance(embutido, ServidorEmbutido)
        assert embutido.porta > 0
        assert embutido.base.startswith(f"http://{HOST}:")
        assert str(embutido.porta) in embutido.base
    finally:
        embutido.desliga()


def test_a_sonda_tenta_as_duas_vias_e_so_devolve_none_quando_as_duas_falham(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**Par causal, e o que ele afirma é a DISJUNÇÃO — não independência de rota.**

    Medido nesta máquina: `getaddrinfo(gethostname())` dá `127.0.1.1`, loopback, então aqui só o
    truque da rota responde. Um teste que afirmasse *"funciona sem rota"* passaria a mentir, que é a
    classe de defeito que este conserto endereça.

    O que se afirma: com pelo menos uma via a responder, a sonda devolve endereço; com **as duas**
    derrubadas, devolve `None` — e é só nesse caso que o `skip` dispara, com a mensagem que nomeia
    as duas.
    """
    assert _endereco_externo() is not None, (
        "âncora: nesta máquina alguma das duas vias responde, senão o par não separa nada"
    )

    def resolvedor_mudo(*_a: object, **_k: object) -> list[object]:
        raise OSError("resolvedor sem resposta")

    class SemRota(socket.socket):  # type: ignore[misc]
        def connect(self, *_a: object, **_k: object) -> None:
            raise OSError("nenhuma rota para o host")

    monkeypatch.setattr(socket, "getaddrinfo", resolvedor_mudo)
    monkeypatch.setattr(socket, "socket", SemRota)

    assert _endereco_externo() is None, (
        "com as duas vias derrubadas a sonda tem de devolver `None`; qualquer outra coisa "
        "significa que há uma terceira via que este teste não conhece"
    )


def test_o_regime_de_skip_dispara_com_a_mensagem_que_a_politica_declara(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**O `skip` exercitado pela suíte, e não só à mão.**

    Eu o exercitei com uma planta e li os três desfechos. A revisão apontou o que falta: *"eu vi"*
    não é *"a suíte vê"*. Numa máquina com endereço externo — todas as de hoje — o ramo nunca
    dispara por conta própria, e ramo que nunca dispara não verifica nada.

    Este teste força o ramo e afirma que a mensagem casa o regime declarado em
    `tests/skip_policy.py` — que é a metade que impede o `skip` de virar falha à espera da condição
    externa.
    """
    from tests.skip_policy import undeclared_reason

    # **`sys.modules[__name__]` e não o caminho por texto.** A primeira versão usava
    # `"tests.test_desktop_servidor._endereco_externo"` e a substituição **não aterrou** — o
    # `pytest` pode registrar o módulo com outro nome —, então o `skip` não disparou e o teste
    # reprovou com *"DID NOT RAISE"*. Substituir no objeto do módulo é imune ao nome.
    monkeypatch.setattr(sys.modules[__name__], "_endereco_externo", lambda: None)

    with pytest.raises(Skipped, match="nem pelo resolvedor de nomes nem pela rota") as capturado:
        test_o_servidor_nao_escuta_no_endereco_externo_da_maquina()

    razao = str(capturado.value)
    assert not undeclared_reason(razao), (
        f"a mensagem do `skip` não casa nenhum regime declarado: {razao!r}. Sem regime, ela é "
        "falha à espera da condição externa e passa em toda máquina onde ela não ocorre"
    )


def test_desliga_reporta_quando_a_thread_nao_para() -> None:
    """**Os dois desfechos de `desliga`, e o segundo é o que a revisão pediu.**

    A versão anterior devolvia `None`: uma thread que ignorasse `should_exit` ficava viva, **com a
    porta ligada**, e quem chamou não tinha como saber. No caminho de segunda tentativa isso
    deixaria o primeiro servidor a servir numa porta que ninguém mais conhece.

    O desfecho `False` é forçado por um duplo de thread que nunca morre — forçar uma thread real a
    ignorar o sinal seria um teste sobre o `uvicorn`, não sobre este código.
    """
    embutido = sobe(_app_minima)
    assert embutido.desliga() is True, "âncora: um servidor de verdade para, e `desliga` diz `True`"

    class ThreadTeimosa:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            return None

    teimoso = ServidorEmbutido(
        porta=1,
        _servidor=embutido._servidor,
        _thread=ThreadTeimosa(),  # type: ignore[arg-type]
    )
    assert teimoso.desliga() is False, (
        "`desliga` devolveu `True` sobre uma thread que continua viva — o desfecho de silêncio que "
        "a revisão apontou"
    )


def test_a_segunda_tentativa_dispara_so_na_disputa_de_porta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**Par causal do conserto: por CAUSA e não por tipo.**

    Antes, `sobe` capturava `ServidorNaoSubiuError` — que é também o que um teto esgotado levanta —,
    então uma falha que **reproduz** na porta nova gastava o orçamento duas vezes.

    | Causa | O que tem de acontecer |
    |---|---|
    | porta disputada (`SystemExit` do `bind`) | segunda tentativa, com o tempo **restante** |
    | prontidão que não chega (`204` para sempre) | **uma** tentativa, e levanta |

    O espião conta as chamadas de `_serve` e guarda os tetos, então as duas metades são medidas e
    não inferidas da duração.
    """
    chamadas: list[float] = []
    original = servidor._serve

    def espia(app: object, porta: int, *, teto_s: float) -> object:
        chamadas.append(teto_s)
        return original(app, porta, teto_s=teto_s)  # type: ignore[arg-type]

    monkeypatch.setattr(servidor, "_serve", espia)

    # ── metade 1: disputa de porta -> DUAS chamadas, a segunda com menos orçamento ──
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ocupante:
        ocupante.bind((HOST, 0))
        ocupante.listen(1)
        ocupada = int(ocupante.getsockname()[1])
        livre = porta_livre()
        entregas = iter([ocupada, livre])
        monkeypatch.setattr(servidor, "porta_livre", lambda: next(entregas))

        embutido = sobe(_app_minima, teto_s=3.0)
        try:
            assert len(chamadas) == 2, f"a segunda tentativa não aconteceu: {chamadas}"
            assert chamadas[1] < chamadas[0], (
                f"a segunda tentativa recebeu {chamadas[1]} e a primeira {chamadas[0]}: o "
                "orçamento não é o restante, e o teto duplica"
            )
        finally:
            embutido.desliga()

    # ── metade 2: prontidão que nunca chega -> UMA chamada ──
    chamadas.clear()
    monkeypatch.setattr(servidor, "porta_livre", porta_livre_real)

    def app_204() -> FastAPI:
        app = FastAPI()

        @app.get(ROTA_DE_PRONTIDAO, status_code=204)
        def prontidao() -> None:
            return None

        return app

    with pytest.raises(ServidorNaoSubiuError):
        sobe(app_204, teto_s=0.6)

    assert len(chamadas) == 1, (
        f"uma prontidão que nunca chega reproduz na porta nova, então NÃO devia haver segunda "
        f"tentativa: {chamadas}"
    )

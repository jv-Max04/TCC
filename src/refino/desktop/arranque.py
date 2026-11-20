"""Ponto de entrada do aplicativo de mesa.

A ordem do arranque é UTF-8, banco, servidor, `REFINO_API_BASE` e janela, e cada passo depende do
anterior. Com o banco antes do UTF-8, a mensagem de erro do banco sairia mutilada pela página de
código; com o servidor antes do banco, a primeira gravação da API falharia longe da causa; com a
janela antes da variável, o cliente apontaria para a porta padrão 8000 e não para o servidor
embutido.

Toda falha de arranque tem causa `ambiente`, no vocabulário de `interface/cliente.py`: quem clicou
não
escolheu porta, diretório nem permissão. O UTF-8 é garantido nas duas formas de execução, porque o
executável congelado ignora `PYTHONUTF8` (o build passa `X utf8=1`) e quem roda do código-fonte
numa console cp932 não tem essa opção.
"""

from __future__ import annotations

import contextlib
import hashlib
import multiprocessing
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final, TextIO

# Este import vem antes da primeira etapa da tela de arranque, e os outros de `refino` não: o
# módulo `banco` só traz biblioteca padrão e `persistence.config`, então a etapa `(1 de 4)` continua
# a legendar o import caro que vem depois. O cache de figuras usa `PASTA` debaixo do diretório por
# usuário mesmo no congelado, onde o banco fica ao lado do executável, porque a pasta do programa
# pode ser somente-leitura (ver `aponta_o_cache_de_figuras`).
from refino.desktop.banco import PASTA

# `garante_utf8` vive em `io/saida.py` porque a linha de comando também a usa e não pode importar
# este pacote; o nome continua importável daqui.
from refino.io.saida import garante_utf8


def _diz_na_tela(etapa: str) -> None:
    """Escreve a etapa na tela de arranque do carregador, se houver uma.

    A tela estática responde 'executou?', mas não 'ainda está andando?', que é a pergunta que faz
    alguém fechar a janela ou clicar de novo numa espera longa. O carregador em C desenha a tela em
    instantes; quase toda a espera é o Python importando `sklearn`, `scipy` e `numpy`, e é isso que
    as etapas legendam.

    `pyi_splash` só existe no congelado com tela. Na CLI, na suíte e num congelado sem `Splash(...)`
    o import falha, e o `except` largo é deliberado: um programa que abre sem legenda é melhor que
    um que não abre.
    """
    try:
        import pyi_splash  # type: ignore[import-untyped]

        pyi_splash.update_text(etapa)
    except Exception:  # pragma: no cover - fora do congelado o módulo não existe
        pass


VARIAVEL_DO_CACHE_DE_FIGURAS: Final = "MPLCONFIGDIR"
"""A variável que o `matplotlib` lê para saber onde guardar o cache de fontes."""

PASTA_DO_CACHE_DE_FIGURAS: Final = "cache-de-figuras"
"""A pasta, debaixo do diretório por usuário, onde os caches por caminho de instalação vivem."""


def aponta_o_cache_de_figuras(
    ambiente: MutableMapping[str, str] | None = None,
    *,
    executavel: str | None = None,
) -> tuple[Path | None, str]:
    """Aponta `MPLCONFIGDIR` para um cache persistente, derivado do caminho de instalação.

    Devolve `(diretório apontado, "")` ou `(None, motivo)` e não levanta: sem cache persistente a
    abertura só fica mais lenta, e o `matplotlib` recua sozinho para um diretório temporário.

    O gancho de runtime do PyInstaller (`pyi_rth_mplconfig.py`) aponta a variável para um diretório
    temporário apagado no `atexit`, e o cache de fontes é reconstruído a cada abertura. Um
    `--runtime-hook` nosso não corrige, porque os ganchos declarados correm antes dos implicados
    pelos pacotes e seriam sobrescritos; o script de entrada corre depois de todos. O `atexit` do
    gancho apaga o diretório que ele mesmo criou, sem reler a variável.

    O gancho existe porque o `fontlist-v*.json` guarda caminhos absolutos para dentro do congelado,
    que num `--onefile` mudam a cada execução. Num `--onedir` eles ficam estáveis enquanto a pasta
    não sair do lugar. A chave é o hash do caminho de instalação, e o cache fica fora da pasta do
    programa: uma cópia noutro lugar recebe cache próprio em vez de herdar caminhos da origem.

    Para reverter, além de tirar o `matplotlib` da composição, apague
    `<dados do usuário>/cache-de-figuras`: nada limpa os subdiretórios que se acumulam ali.
    """
    env = os.environ if ambiente is None else ambiente
    binario = (sys.executable if executavel is None else executavel).strip()
    if not binario:
        return None, "o programa nao sabe o caminho do proprio executavel"

    base = (env.get("LOCALAPPDATA") or "").strip() or (env.get("APPDATA") or "").strip()
    if not base:
        base = (env.get("XDG_CACHE_HOME") or "").strip()
    if not base:
        lar = (env.get("HOME") or "").strip()
        base = str(Path(lar) / ".cache") if lar else ""
    if not base:
        return None, "nem LOCALAPPDATA, nem APPDATA, nem XDG_CACHE_HOME, nem HOME estao definidos"

    # O hash é do diretório de instalação em minúsculas, porque o Windows não distingue
    # maiúsculas em caminho: `C:\Refino` e `c:\refino` dariam duas pastas de cache.
    instalacao = str(Path(binario).parent)
    marca = hashlib.sha256(instalacao.casefold().encode("utf-8")).hexdigest()[:16]
    destino = Path(base) / PASTA / PASTA_DO_CACHE_DE_FIGURAS / marca

    # A prova de escrita é um arquivo de verdade: no Windows, `os.access` com `W_OK` só olha o
    # atributo de somente-leitura e diz 'pode' sobre pasta que a ACL recusa.
    try:
        destino.mkdir(parents=True, exist_ok=True)
        prova = destino / ".prova-de-escrita"
        prova.write_text("", encoding="utf-8")
        prova.unlink()
    except OSError as erro:
        return None, f"{destino} nao serve para escrita ({type(erro).__name__})"

    env[VARIAVEL_DO_CACHE_DE_FIGURAS] = str(destino)
    return destino, ""


AVISO_DO_CACHE_DE_FIGURAS: str = ""
"""Por que o cache de figuras degradou; vazio quando não degradou.

É lido por `main`: neste ponto do arranque `sys.stderr` ainda pode ser `None` (`--windowed`, antes
de `garante_fluxos`), e uma queixa escrita aqui se perderia.
"""

# Só no congelado: lá o `loky` relançaria o próprio `refino.exe` para cada worker, com janela.
# `aponta_o_loky` corre antes de qualquer import do `joblib`, porque `spawn._python_exe` é lido na
# importação do backend.
if getattr(sys, "frozen", False):  # pragma: no cover - so no congelado
    from refino.desktop.paralelismo import aponta_o_loky

    aponta_o_loky()

    # O cache também só no congelado, porque só lá existe o gancho que isto desfaz; fora dele o
    # `matplotlib` usa `~/.matplotlib`. Só o aviso fica em global: o diretório quem lê é o
    # `matplotlib`, pela variável de ambiente.
    _, AVISO_DO_CACHE_DE_FIGURAS = aponta_o_cache_de_figuras()

# A primeira etapa vai aqui, no nível do módulo, porque o import seguinte é o mais caro do
# arranque: `refino.api.app` traz `sklearn`, `scipy` e `numpy`, e no primeiro arranque o antivírus
# varre cada arquivo deles. Dentro de `main` ela chegaria depois dessa espera.
_diz_na_tela("a carregar as bibliotecas cientificas... (1 de 4)")

# Os `noqa: E402` existem porque estes imports precisam vir depois da chamada acima, que legenda
# justamente o primeiro deles.
from refino.api.app import build_app  # noqa: E402
from refino.desktop.banco import (  # noqa: E402
    BancoDeMesaError,
    diretorio_de_dados,
    prepara,
    quem_le_o_diretorio,
)
from refino.desktop.servidor import (  # noqa: E402
    ServidorEmbutido,
    ServidorNaoSubiuError,
    sobe,
)
from refino.persistence.config import VARIAVEL_DA_URL  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Callable, Mapping, MutableMapping, Sequence
    from pathlib import Path

    from refino.desktop.banco import BancoDeMesa
    from refino.interface.cliente import Causa

VARIAVEL_DA_BASE: Final = "REFINO_API_BASE"
"""A variável que `interface/cliente.py::base_da_api` lê e que este módulo escreve.

O nome é repetido como literal para não acoplar o arranque ao cliente só para lê-lo.
"""

SAIU_BEM: Final = 0
SAIU_POR_AMBIENTE: Final = 1
"""Códigos de saída. São dois para que quem rode o executável de um `.bat` ou de um agendador
distinga sucesso de falha."""

# Arquivo onde a saída do aplicativo cai quando não há console. Fica onde
# `banco.diretorio_de_dados` põe o banco, para que o log siga o banco (no congelado, para a pasta
# do executável) sem que este módulo precise saber a regra.
NOME_DO_LOG: Final = "refino.log"

# Teto do log, que é aberto em modo de acréscimo; 1 MiB cabe muitos arranques e ainda abre num
# editor.
TETO_DO_LOG: Final = 1024 * 1024


def garante_fluxos(
    ambiente: Mapping[str, str] | None = None,
    *,
    plataforma: str | None = None,
    congelado: bool | None = None,
    executavel: str | None = None,
) -> tuple[str, ...]:
    """Liga `sys.stdout` e `sys.stderr` a um arquivo quando eles são `None` e devolve o que ligou.

    Um executável `--windowed` lançado sem handles de saída, que é o que o Explorer faz no duplo
    clique, corre com os dois fluxos em `None`. O formatador padrão do `uvicorn` chama
    `sys.stdout.isatty()`, e o arranque morria sem ter onde dizer por quê. Ligar os fluxos, em vez
    de só configurar o `uvicorn`, protege também da próxima biblioteca que toque em `sys.stdout`, e
    dá ao usuário um arquivo para ler.

    Nunca levanta. Se os fluxos existem, não toca em nada e devolve `()`; se o log não abre, usa
    `os.devnull`, que basta para o `isatty()`; se nem esse abre, deixa como está. O retorno permite
    ao teste afirmar mais que 'não levantou', como em `garante_utf8`, de quem a assinatura é irmã.
    """
    nomes = tuple(nome for nome in ("stdout", "stderr") if getattr(sys, nome, None) is None)
    if not nomes:
        return ()

    fluxo = (
        _abre_o_log(ambiente, plataforma=plataforma, congelado=congelado, executavel=executavel)
        or _abre_o_devnull()
    )
    if fluxo is None:  # pragma: no cover - nem o `devnull` abriu; não há terceiro recuo
        return ()

    for nome in nomes:
        setattr(sys, nome, fluxo)
    return nomes


def _abre_o_log(
    ambiente: Mapping[str, str] | None = None,
    *,
    plataforma: str | None = None,
    congelado: bool | None = None,
    executavel: str | None = None,
) -> TextIO | None:
    r"""O arquivo de log, ou `None` se nenhum lugar aceitar escrita.

    Acrescenta em vez de truncar: quem clica duas vezes porque a primeira pareceu não fazer nada
    quer ver as duas tentativas. Passado o teto, o arquivo recomeça.

    No congelado o log tem um segundo lugar, e o banco não. A pasta do executável pode não aceitar
    gravação (zip aberto de rede, `C:\Program Files`, mídia somente-leitura). Para o banco isso é
    fatal de propósito, porque um segundo lugar faria execuções gravarem em bancos diferentes sem
    ninguém notar. O log guarda por que o programa não abriu, e cair direto em `NUL` deixaria o
    usuário sem diagnóstico justamente quando precisa dele. A ordem é a pasta do executável, o
    diretório por usuário e só então `NUL`; a primeira linha diz onde o log caiu.
    """
    tentativas: list[Path] = []
    with contextlib.suppress(BancoDeMesaError, ValueError):
        tentativas.append(
            diretorio_de_dados(
                ambiente, plataforma=plataforma, congelado=congelado, executavel=executavel
            )
        )
    if congelado if congelado is not None else getattr(sys, "frozen", False):
        # o recuo só existe no congelado: fora dele o primeiro lugar já é o diretório por usuário
        with contextlib.suppress(BancoDeMesaError, ValueError):
            por_usuario = diretorio_de_dados(ambiente, plataforma=plataforma, congelado=False)
            if por_usuario not in tentativas:
                tentativas.append(por_usuario)

    for i, base in enumerate(tentativas):
        try:
            alvo = base / NOME_DO_LOG
            alvo.parent.mkdir(parents=True, exist_ok=True)
            if alvo.exists() and alvo.stat().st_size > TETO_DO_LOG:
                alvo.unlink()
            fluxo = alvo.open("a", encoding="utf-8", errors="backslashreplace", buffering=1)
        except (OSError, ValueError):
            continue
        if i > 0:
            # A primeira linha diz onde ele caiu; sem ela, quem procurar o log na pasta do
            # programa concluiria que ele não existe.
            fluxo.write(
                f"[refino] a pasta do programa nao aceitou gravacao; este diario esta em {alvo}\n"
            )
        return fluxo
    return None


def _abre_o_devnull() -> TextIO | None:
    """O recuo: não salva o diagnóstico, mas deixa o arranque seguir."""
    try:
        return open(os.devnull, "w", encoding="utf-8")
    except OSError:  # pragma: no cover - `os.devnull` indisponível não é caso previsto
        return None


def _relata(mensagem: str, *, causa: Causa) -> None:
    """Escreve a falha onde quem rodou o executável a vê.

    Vai para `stderr` e não levanta: no ponto em que é chamada, levantar produziria um traceback no
    caminho do usuário.
    """
    fluxo = getattr(sys, "stderr", None)
    if fluxo is None:  # `--windowed` sem console: não há onde escrever
        return
    print(f"refino: {mensagem} (causa: {causa})", file=fluxo)


def _relata_quem_le(diretorio: Path, *, plataforma: str | None = None) -> None:
    """Escreve no `refino.log` quem lê o que este programa grava. Não levanta e não recusa.

    O log, e não a aba de limites, porque ele volta com a pasta devolvida, é escrito antes de a
    janela abrir e sobrevive a ela não abrir. A aba deriva o estado de cada seção da composição da
    janela e não tem onde receber uma medição de arranque; ela diz que a proteção depende de onde a
    pasta está e aponta para o log.

    Os três desfechos saem, inclusive o `restrito`: uma verificação que só fala quando há problema
    não se distingue de uma que não correu. A linha com os SIDs sai sempre que houver algum, porque
    o veredito é heurística e a lista é a evidência.
    """
    leitura = quem_le_o_diretorio(diretorio, plataforma=plataforma)
    fluxo = getattr(sys, "stderr", None)
    if fluxo is None:  # pragma: no cover - `--windowed` sem console e sem log; ver `garante_fluxos`
        return
    print(f"refino: {leitura.para_quem_opera()}", file=fluxo)
    if leitura.herdam_leitura:
        print(
            f"refino: [técnico] leem por herança: {', '.join(leitura.herdam_leitura)}",
            file=fluxo,
        )


def abre_a_janela(base: str) -> int:
    """Sobe a `QApplication`, mostra a janela e devolve o código de saída do laço de eventos.

    O import de Qt fica dentro da função porque o grupo `desktop` é opcional: no topo do módulo, o
    arranque deixaria de ser importável sem PySide6, e a CLI arrastaria o Qt sem ter janela.

    `constroi_transporte` monta o cliente a partir de `REFINO_API_BASE` no ambiente, e não do
    argumento. A conferência abaixo recusa a divergência entre os dois, em que a janela falaria com
    outra coisa que o servidor recém-subido e toda rota responderia `503`. Em `main` as duas pontas
    saem da mesma expressão, então a guarda só acusa quando `abre_a_janela` é chamada de fora de
    `main` sem a variável exportada.
    """
    from refino.interface.cliente import base_da_api

    lido = base_da_api()
    if lido != base:
        _relata(
            f"a janela apontaria para {lido} e o servidor está em {base}: "
            f"{VARIAVEL_DA_BASE} não chegou a este processo",
            causa="ambiente",
        )
        return SAIU_POR_AMBIENTE

    return _corre_a_janela()


def _fecha_a_tela_de_arranque() -> None:
    """Fecha a tela que o carregador do PyInstaller desenhou, se houver uma.

    `pyi_splash` só existe no congelado com tela, e o `except` é largo porque uma falha ao fechar a
    tela não pode derrubar o arranque da janela. É chamada logo antes de importar o PySide6: fechar
    depois deixaria a tela por cima da janela real por um instante.
    """
    try:
        import pyi_splash

        pyi_splash.close()
    except Exception:  # pragma: no cover - fora do congelado o módulo não existe
        pass


def _corre_a_janela() -> int:
    """A parte do arranque que precisa de Qt.

    Fica separada para a guarda de `abre_a_janela` ser testável nos dois sentidos: sem ela o caminho
    de sucesso chamaria `app.exec()` e o teste bloquearia; com ela o arnês troca esta função por um
    espião e afirma que a guarda deixou passar.
    """
    # A quarta etapa é escrita antes do fecho da tela: o import do PySide6 é o último pedaço grande
    # do arranque, e depois do fecho não há tela onde escrever.
    _diz_na_tela("a abrir a janela... (4 de 4)")
    _fecha_a_tela_de_arranque()

    # `pandas`, `sklearn` e `matplotlib` antes do PySide6. O gancho de import do PySide6
    # (`shibokensupport.feature`) inspeciona todo módulo importado depois dele, e o importador
    # preguiçoso do `six`, registrado pelo `dateutil`, não suporta a inspeção: a janela não abria a
    # partir do código-fonte. O `noqa: I001` protege a ordem contra o organizador de imports.
    import matplotlib.axes  # noqa: I001
    import matplotlib.figure  # noqa: F401

    import refino.cli  # noqa: F401

    from PySide6.QtWidgets import QApplication

    from refino.desktop.janela.principal import JanelaPrincipal
    from refino.desktop.janela.tema import veste_a_aplicacao, veste_a_janela
    from refino.interface.cliente import constroi_transporte

    # `instance()` antes de construir: duas `QApplication` no mesmo processo abortam o Qt, e um
    # arnês como o `qtbot` já tem uma de pé.
    app = QApplication.instance() or QApplication([])
    # O tema antes da janela: aplicada depois, a folha de estilo obrigaria o Qt a repolir a árvore
    # inteira de widgets.
    veste_a_aplicacao(app)  # type: ignore[arg-type]
    janela = JanelaPrincipal(constroi_transporte())
    veste_a_janela(janela)
    janela.show()
    codigo = int(app.exec())
    # A piscina é drenada depois do laço porque `main` desliga o servidor logo que esta função
    # volta; uma chamada em voo (um SERRF leva minutos) falaria com um `uvicorn` descendo, e a
    # exceção numa thread nativa ninguém lê. A drenagem descarta a fila antes de esperar, porque
    # `waitForDone` também espera o que está enfileirado. A mensagem do estouro diz que o processo
    # vai bloquear porque ele bloqueia: o destrutor de `QThreadPool` chama `waitForDone()` sem teto,
    # e o cliente tem 300 s de leitura.
    if not janela.drena_a_piscina():
        _relata(
            "havia chamada em curso quando a janela fechou e o teto de espera passou. O processo "
            "só termina quando ela voltar: o descarte da piscina espera sem teto, e o cliente tem "
            "300 s de leitura",
            causa="ambiente",
        )
    return codigo


def arranca(
    ambiente: Mapping[str, str] | None = None,
    *,
    plataforma: str | None = None,
    congelado: bool | None = None,
    executavel: str | None = None,
    migrar: bool = True,
) -> tuple[ServidorEmbutido, BancoDeMesa]:
    """Prepara banco e servidor, exporta as duas variáveis de ambiente e devolve os dois objetos.

    Fica separada de `main`, cujo único trabalho é traduzir exceção em código de saída, para que um
    teste possa ter o servidor de pé sem passar por um `int`. Devolve também o `BancoDeMesa`, para
    que quem chama possa afirmar contra qual arquivo a API está falando.
    """
    _diz_na_tela("a preparar o banco de dados... (2 de 4)")
    banco = prepara(
        ambiente,
        plataforma=plataforma,
        congelado=congelado,
        executavel=executavel,
        migrar=migrar,
    )

    # A exportação é necessária: as rotas resolvem o banco a cada pedido por `url_do_banco()`, que
    # lê `os.environ`, e `banco.migra` só põe `REFINO_DB_URL` durante o `alembic upgrade`. Sem ela,
    # o banco seria criado e migrado e todo pedido que grava responderia `503`.
    os.environ[VARIAVEL_DA_URL] = banco.url
    _diz_na_tela("a subir o servidor local... (3 de 4)")
    servidor = sobe(build_app)
    os.environ[VARIAVEL_DA_BASE] = servidor.base
    return servidor, banco


def main(
    argv: Sequence[str] | None = None,
    *,
    ambiente: Mapping[str, str] | None = None,
    plataforma: str | None = None,
    congelado: bool | None = None,
    executavel: str | None = None,
    abre_janela: Callable[[str], int] | None = None,
) -> int:
    """Ponto de entrada: traduz falha em código de saída e mensagem com causa.

    `argv` é aceito e ignorado. `[project.scripts]` chama `main()` sem argumentos, e declarar o
    parâmetro evita mudar a assinatura no dia em que houver bandeira.
    """
    _ = argv
    # `freeze_support` vem antes de tudo: num filho relançado pelo `multiprocessing` ele executa a
    # tarefa e termina o processo, em vez de deixar o aplicativo abrir outra janela; no pai e fora
    # do congelado devolve na hora. O alcance é parcial, porque o corpo do módulo (a sonda de
    # `aponta_o_loky` e o import de `refino.api.app`) já correu antes daqui. Hoje nenhum caminho usa
    # `multiprocessing` para gerar processo, e o `loky` lança o `pythonw.exe` embarcado.
    multiprocessing.freeze_support()

    # `garante_fluxos` antes de `garante_utf8`, porque é ele que cria os fluxos que o outro
    # reconfigura.
    garante_fluxos(ambiente, plataforma=plataforma, congelado=congelado, executavel=executavel)
    garante_utf8()

    # A queixa do cache de figuras sai aqui, onde `sys.stderr` já existe e cai no `refino.log`. Ela
    # não impede o programa de abrir: sem o cache persistente a abertura só fica mais lenta.
    if AVISO_DO_CACHE_DE_FIGURAS:
        _relata(
            "o cache de fontes das figuras nao ficou persistente e sera reconstruido a cada "
            f"abertura: {AVISO_DO_CACHE_DE_FIGURAS}",
            causa="ambiente",
        )

    abre = abre_a_janela if abre_janela is None else abre_janela
    try:
        # `_relata_quem_le` precisa do diretório em que o banco de fato caiu; derivá-lo de novo de
        # `diretorio_de_dados` criaria uma segunda fonte para a mesma resposta.
        servidor, banco = arranca(
            ambiente, plataforma=plataforma, congelado=congelado, executavel=executavel
        )
    except (BancoDeMesaError, ServidorNaoSubiuError) as erro:
        _relata(str(erro), causa="ambiente")
        return SAIU_POR_AMBIENTE
    except Exception as erro:
        # Rede de segurança: num `--onedir` sem console, qualquer tipo que a tupla acima não nomeie
        # sairia como traceback que ninguém lê. A mensagem leva o tipo e não o texto do erro,
        # porque mensagens do SQLAlchemy trazem a URL do banco, e a de Postgres traz a senha.
        _relata(
            f"falha não prevista no arranque: {type(erro).__name__}. O detalhe fica fora da "
            "mensagem de propósito — mensagens de banco carregam a URL, e ela carrega a senha",
            causa="ambiente",
        )
        return SAIU_POR_AMBIENTE

    # A medição de quem lê vem depois de `arranca`, porque é `prepara` que cria o diretório; antes
    # dela, `GetNamedSecurityInfoW` falharia e todo primeiro arranque daria 'não medido'.
    try:
        # Fica dentro da rede para que uma falha aqui tenha causa declarada e passe pelo `finally`
        # que desliga o servidor. O `except` próprio impede que ela saia como 'a janela não abriu':
        # a medição é diagnóstico e não pode impedir a janela de abrir.
        try:
            _relata_quem_le(banco.caminho.parent, plataforma=plataforma)
        except Exception as erro:
            _relata(
                f"não foi possível registar quem lê a pasta do banco: {type(erro).__name__}",
                causa="ambiente",
            )
        return abre(servidor.base)
    except Exception as erro:
        # A mesma rede, agora sobre a janela: uma exceção em `JanelaPrincipal.__init__` sairia de
        # `main` sem causa declarada. Ela cobre `abre` inteiro, porque o laço de eventos do Qt já
        # engole as exceções dos slots, e a causa é `ambiente`, como nas outras falhas de arranque.
        # Com `sys.stderr` em `None`, a mensagem cai no `refino.log` que `garante_fluxos` instalou,
        # e leva só o tipo, pelo mesmo motivo do `except` de cima.
        _relata(
            f"a janela não abriu: {type(erro).__name__}. O servidor foi desligado. O detalhe fica "
            "fora da mensagem de propósito — mensagens de banco carregam a URL, e ela a senha",
            causa="ambiente",
        )
        return SAIU_POR_AMBIENTE
    finally:
        # `finally` e não `else`: a janela pode fechar por exceção, e um servidor que sobrevive à
        # janela deixa a porta presa até alguém matar o processo.
        servidor.desliga()


if __name__ == "__main__":  # pragma: no cover - o ponto de entrada, exercitado por `main`
    raise SystemExit(main())

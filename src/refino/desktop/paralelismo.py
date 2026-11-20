"""Regime de paralelismo do executável congelado.

O `loky` cria cada worker relançando `sys.executable`, que no congelado é o próprio `refino.exe`:
cada worker reabriria o aplicativo, com janela. Desligar os processos com
`JOBLIB_MULTIPROCESSING=0` põe tudo em série, e threads não ajudam porque o trabalho por feature
fica preso ao GIL. Fazer o `refino.exe` executar o `-c` que o `loky` passa o tornaria um
interpretador de código arbitrário. A saída é apontar o `loky` para o `pythonw.exe` embarcado em
`_internal`, em dois sítios: `spawn._python_exe`, lido pelo `resource_tracker`, e
`popen_loky_win32.get_command_line`, lido pelo worker. O ramo `sys.frozen` deste último devolve
uma linha sem `-c` e sem `parent_pid`, e o interpretador sai; por isso o remendo devolve, do lado
do pai, a forma não congelada da linha.

Se não há interpretador na pasta, se o módulo vendorizado do `joblib` mudou de forma, se o caminho
resolvido sai da pasta ou se o interpretador não arranca, o regime cai para série: ser lento é
melhor que abrir uma janela por worker. Fora do congelado (CLI, suíte) nada é remendado e o regime
continua paralelo, porque lá o `loky` funciona.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

__all__ = [
    "NOME_DO_INTERPRETADOR",
    "PARALELO",
    "SERIE",
    "TETO_DA_SONDA_S",
    "aponta_o_loky",
    "n_jobs_efetivo",
    "regime_do_processo",
    "resolve_o_interpretador",
]

# `pythonw.exe` e não `python.exe`: o `loky` cria o processo sem flags e o `refino.exe` não tem
# console, então um filho de subsistema console ganharia um console novo, uma janela por worker.
NOME_DO_INTERPRETADOR: Final = "pythonw.exe"

# O vocabulário do `n_jobs` do `joblib`: `-1` são todos os núcleos, `1` é série.
PARALELO: Final = -1
SERIE: Final = 1

# Quanto se espera pela sonda do arranque. É generoso porque o primeiro arranque de um
# interpretador em `_internal` paga o cache do sistema de arquivos, e um teto curto poria em série
# uma máquina lenta que estava bem.
TETO_DA_SONDA_S: Final = 30


# A capacidade do processo, resolvida uma vez no arranque: há interpretador embarcado e o
# remendo pegou. A escolha de cada execução vem do pedido, e o `n_jobs` efetivo só é `PARALELO`
# quando as duas concordam. O padrão é `PARALELO` porque fora do congelado ninguém chama
# `aponta_o_loky`, e lá o `loky` já funciona.
_REGIME: int = PARALELO


def regime_do_processo() -> int:
    """O `n_jobs` que a capacidade deste processo permite; `PARALELO` até prova em contrário."""
    return _REGIME


def n_jobs_efetivo(paralelo: bool) -> int:
    """O `n_jobs` que sai da escolha do pedido combinada com a capacidade do processo.

    Quando a rede caiu para série, `aponta_o_loky` não desliga o `loky`: um chamador que pedisse
    `n_jobs=-1` faria o `loky` relançar o `refino.exe`. Por isso o pedido nunca vence a capacidade.
    Desmarcar a caixa é sempre atendido; marcá-la é permissão, não ordem.
    """
    if not paralelo:
        return SERIE
    return _REGIME


def resolve_o_interpretador(executavel: str | None = None) -> Path | None:
    """O interpretador embarcado, ou `None`.

    A pasta vem de `sys.executable` e de mais lugar nenhum. O `PATH` não é consultado e nada é
    resolvido contra o diretório de trabalho: um `pythonw.exe` plantado onde o programa foi aberto
    seria executado sem sintoma, e um interpretador que importasse outro `refino` daria resultado
    errado sem levantar.

    São três condições: o candidato existe, há um `._pth` ao lado dele, e o caminho resolvido fica
    debaixo da pasta do congelado, o que recusa link simbólico e junção que apontem para fora.
    """
    bruto = executavel if executavel is not None else sys.executable
    if not bruto:
        return None
    raiz = Path(bruto).resolve().parent
    interno = raiz / "_internal"
    candidato = interno / NOME_DO_INTERPRETADOR
    if not candidato.is_file():
        return None
    # O `._pth` é o que isola o worker. Sem ele o interpretador procura a stdlib fora de
    # `_internal` (registro do Windows ou prefixo de construção); na máquina que constrói isso
    # acha a instalação original e mascara a falha, e o worker importa `numpy` e `refino` de fora
    # da pasta entregue. Glob e não nome exato: `python312._pth` acopla à versão, e
    # `pythonw._pth` também vale.
    if not any(interno.glob("python*._pth")):
        return None
    # `is_relative_to` sobre os dois já resolvidos: uma junção ou link simbólico em `_internal`
    # que aponte para fora da pasta é resolvida aqui e recusada.
    if not candidato.resolve().is_relative_to(raiz):
        return None
    return candidato


def _o_interpretador_arranca(caminho: str) -> bool:
    """Diz se o interpretador embarcado corre um programa trivial; existir não basta.

    Devolve `False` em qualquer desfecho que não seja código de saída `0`: executável inválido,
    versão incompatível, `._pth` ausente, DLL em falta ou arranque que passe do teto. Nunca levanta,
    porque uma exceção aqui derrubaria o arranque do aplicativo, o que é pior que correr em série.
    """
    import subprocess

    # `CREATE_NO_WINDOW` só existe no Windows; fora dele a sonda não corre, porque o congelado
    # também não existe.
    sem_janela = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        corrida = subprocess.run(
            [caminho, "-c", "import sys; sys.exit(0)"],
            capture_output=True,
            timeout=TETO_DA_SONDA_S,
            creationflags=sem_janela,
        )
    except Exception:
        return False
    return corrida.returncode == 0


def aponta_o_loky(
    executavel: str | None = None,
    *,
    congelado: bool | None = None,
    sonda: Callable[[str], bool] = _o_interpretador_arranca,
) -> int:
    """Aponta os dois sítios do `loky` para o interpretador embarcado; devolve o `n_jobs` do regime.

    Fora do congelado devolve `PARALELO` sem escrever nada, porque lá o `loky` já funciona. É
    idempotente: a segunda chamada vê o remendo posto e não empilha outro.

    `sonda` é a costura de teste. O padrão lança o candidato de fato; um teste que monte uma pasta
    com a forma da entrega e um `pythonw.exe` falso passa uma sonda própria e mede só os remendos.
    """
    global _REGIME

    esta_congelado = getattr(sys, "frozen", False) if congelado is None else congelado
    if not esta_congelado:
        _REGIME = PARALELO
        return PARALELO

    interpretador = resolve_o_interpretador(executavel)
    if interpretador is None:
        _REGIME = SERIE
        return SERIE

    try:
        from joblib.externals.loky.backend import popen_loky_win32, spawn
    except Exception:
        # O módulo vendorizado mudou de forma: cai para série em vez de aplicar um remendo que
        # já não encaixa.
        _REGIME = SERIE
        return SERIE

    caminho = str(interpretador)

    # As guardas acima só provam que o arquivo existe. Um `pythonw.exe` de outra versão, com o
    # `._pth` errado ou corrompido, passaria por elas, e os workers morreriam com
    # `TerminatedWorkerError`; a sonda devolve o modo de falha a 'ser lento'. Ela corre com
    # `CREATE_NO_WINDOW` para não abrir o console que este bloco existe para evitar.
    if not sonda(caminho):
        _REGIME = SERIE
        return SERIE

    # A leitura do atributo também cai na rede. Se o símbolo for renomeado sem a árvore mudar, o
    # `import` passa e `get_command_line` levantaria `AttributeError` no import de `arranque.py`,
    # antes de haver log. E vem antes de atribuir `spawn._python_exe`: atribuição nunca levanta, e
    # deixaria o remendo aplicado pela metade.
    anterior = getattr(popen_loky_win32, "get_command_line", None)
    if anterior is None:
        _REGIME = SERIE
        return SERIE

    spawn._python_exe = caminho

    if getattr(anterior, "_refino_remendado", False):
        _REGIME = PARALELO
        return PARALELO

    def linha_de_comando(pipe_handle: Any, parent_pid: Any, **kwds: Any) -> list[str]:
        """A forma não congelada da linha de comando, que leva `-c` e `parent_pid`.

        O ramo `sys.frozen` do `loky` devolve `[sys.executable, "--multiprocessing-fork",
        pipe_handle]`, e o interpretador embarcado sairia com uma opção que não conhece antes de
        `main()` correr.
        """
        from multiprocessing import util

        # O filho repõe o `stdio` antes de tudo. Um `pythonw.exe` filho de um pai sem console não
        # herda handle nenhum, então `sys.stdin`, `sys.stdout` e `sys.stderr` nascem `None`, e o
        # `_bootstrap` do `multiprocessing` chama `sys.stdin.close()`: o worker morreria sem
        # traceback. Fora do congelado o pai é um console e o problema não aparece.
        # O programa não tem quebra de linha porque o `loky` cita cada argumento ao montar a linha
        # do `CreateProcess`, e uma quebra dentro dele é frágil; daí `x or devnull` e não um laço.
        prog = (
            "import os,sys;"
            "_d=lambda m:open(os.devnull,m);"
            "sys.stdin=sys.stdin or _d('r');"
            "sys.stdout=sys.stdout or _d('w');"
            "sys.stderr=sys.stderr or _d('w');"
            "from joblib.externals.loky.backend.popen_loky_win32 import main;"
            f"main(pipe_handle={pipe_handle}, parent_pid={parent_pid})"
        )
        return [
            caminho,
            *util._args_from_interpreter_flags(),  # type: ignore[attr-defined]
            "-c",
            prog,
            "--multiprocessing-fork",
        ]

    linha_de_comando._refino_remendado = True  # type: ignore[attr-defined]
    popen_loky_win32.get_command_line = linha_de_comando
    _REGIME = PARALELO
    return PARALELO

"""O lançador do R embarcado para as vias: a fronteira de processo do enriquecimento.

Roda `r_service/vias_entrypoint.R` na árvore R portátil, com JSON em stdin e JSON em stdout, e
devolve um resultado tipado que não levanta por indisponibilidade: motor ausente, motor que
falhou, tempo esgotado, resposta malformada, cancelamento e recusa do roteiro são desfechos.
Recebe a raiz do R, o roteiro e a pasta das bibliotecas já resolvidos como `Path`; aqui não se
consulta o `PATH` nem o diretório de trabalho.

O filho roda isolado:

1. `bin\\x64\\Rscript.exe` direto. O `bin\\Rscript.exe` relança por `cmd.exe`, que reinterpreta
   o caminho (um `&` ou `%` na pasta muda o comando) e sobrevive ao `kill` do filho.
2. Job Object com "matar ao fechar". O cancelamento termina o Job inteiro, netos incluídos, e se
   o `refino.exe` morrer de forma abrupta o sistema leva o R junto. O `Rscript` x64 não lança
   filho no arranque, então nenhum neto nasce antes de o processo entrar no Job.
3. Ambiente limpo. Sai toda variável `R_*`, `NO_PROXY` e `ALL_PROXY`; as quatro que o R lê para
   achar biblioteca e configuração apontam para um caminho absoluto e inexistente no temporário
   da corrida; os proxies apontam para um endereço morto, para que um cliente que tente sair
   falhe em vez de sair.
4. `--vanilla`, sem janela, um temporário novo por corrida e UTF-8 nos dois lados. A entrada vai
   por `stdin`, que é um arquivo e não um tubo, para que o cancelamento valha durante o arranque.
5. O roteiro devolve o `.libPaths()` que o R viu, e um caminho fora da árvore é "motor falhou".

O `status` e o desfecho de recusa que voltam do R são validados contra vocabulário fechado
(`ViasRecusa`, em `domain/labels.py`); um valor de fora é resposta malformada.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

from refino.domain.labels import InvalidLabelError, ViasRecusa, validate_label

__all__ = [
    "PROXY_MORTO",
    "TEMPO_LIMITE_PADRAO_S",
    "DesfechoDoLancador",
    "RespostaDoR",
    "ambiente_isolado",
    "corre_roteiro_de_vias",
    "rscript_da_arvore",
]

DesfechoDoLancador = Literal[
    "ok",
    "motor_ausente",
    "motor_falhou",
    "tempo_esgotado",
    "resposta_malformada",
    "cancelado",
    "recusado",
]

PROXY_MORTO: Final = "http://127.0.0.1:9"
# O teto vale para as chamadas depois da primeira de cada processo. 120 s são cerca de 25 vezes a
# maior chamada medida a quente (4,78 s, a ORA com um fundo de 2.000 nomes) e 17 vezes a primeira
# chamada numa cópia recém-feita da pasta (6,91 s). A primeira chamada de cada processo vai sem
# teto (`None`), só com o cancelamento: é ela que paga antivírus e cache frio do sistema, que não
# foram medidos.
TEMPO_LIMITE_PADRAO_S: Final = 120.0
_INTERVALO_S: Final = 0.2
_RESUMO_DO_STDERR: Final = 1500
_VARIAVEIS_DO_R: Final = ("R_LIBS_USER", "R_LIBS_SITE", "R_ENVIRON_USER", "R_PROFILE_USER")


@dataclass(frozen=True)
class RespostaDoR:
    """O que uma chamada ao roteiro devolveu.

    `dados` só existe em `ok`, e `recusa` só em `recusado`.
    """

    desfecho: DesfechoDoLancador
    motivo: str
    dados: Mapping[str, object] | None = None
    recusa: ViasRecusa | None = None
    stderr: str = ""
    segundos: float = 0.0


def rscript_da_arvore(r_home: Path) -> Path:
    return r_home / "bin" / "x64" / "Rscript.exe"


def ambiente_isolado(
    raiz_temp: Path, *, proxy_morto: bool = True, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Monta o ambiente do filho, conforme o item 3 da docstring do módulo."""
    origem = os.environ if base is None else base
    env = {
        k: v
        for k, v in origem.items()
        if not k.upper().startswith("R_")
        and k.upper() not in {"NO_PROXY", "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"}
    }
    inexistente = raiz_temp / "nao-existe"
    for k in _VARIAVEIS_DO_R:
        env[k] = str(inexistente)
    # O `Rtmp*` da sessão nasce na pasta da corrida, que o lançador apaga: cancelar mata o R antes
    # de ele apagar o seu, e no `%TEMP%` comum ele sobraria.
    for k in ("TMPDIR", "TMP", "TEMP"):
        env[k] = str(raiz_temp)
    if proxy_morto:
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env[k] = PROXY_MORTO
    return env


class _Job:
    """Um Job Object do Windows com "matar ao fechar". Fora do Windows, encerra só o processo."""

    def __init__(self) -> None:
        self._handle: int | None = None
        # Declarado antes do teste de plataforma, e com `Any`: fora do Windows o resto deste
        # construtor é inalcançável para o `mypy`, e sem esta linha o atributo ficaria sem tipo
        # nos três métodos abaixo.
        self._k32: Any = None
        if sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
        k32.SetInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        )
        k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)

        class _Basica(ctypes.Structure):
            _fields_ = (
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            )

        class _Io(ctypes.Structure):
            _fields_ = tuple((n, ctypes.c_uint64) for n in ("R", "W", "O", "RT", "WT", "OT"))

        class _Estendida(ctypes.Structure):
            _fields_ = (
                ("BasicLimitInformation", _Basica),
                ("IoInfo", _Io),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            )

        handle = k32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW falhou")
        info = _Estendida()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not k32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            k32.CloseHandle(handle)
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject falhou")
        self._k32 = k32
        self._handle = handle

    def adota(self, processo: subprocess.Popen[bytes]) -> None:
        # O teste de plataforma é redundante em tempo de execução, porque fora do Windows `_handle`
        # fica `None`. Existe para o `mypy`: `ctypes.get_last_error` só existe no Windows.
        if self._handle is None or sys.platform != "win32":
            return
        pid_handle = int(cast(int, processo._handle))  # type: ignore[attr-defined, unused-ignore]
        if not self._k32.AssignProcessToJobObject(self._handle, pid_handle):
            import ctypes

            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject falhou")

    def termina(self, processo: subprocess.Popen[bytes]) -> None:
        if self._handle is not None:
            self._k32.TerminateJobObject(self._handle, 1)
        else:
            processo.kill()

    def fecha(self) -> None:
        if self._handle is not None:
            self._k32.CloseHandle(self._handle)
            self._handle = None


def _resumo(texto: str) -> str:
    return texto[-_RESUMO_DO_STDERR:].strip()


def _dentro(caminho: str, raiz: Path) -> bool:
    try:
        return Path(caminho).resolve().is_relative_to(raiz.resolve())
    except (OSError, ValueError):
        return False


def _interpreta(
    stdout: str, stderr: str, codigo: int, r_home: Path, segundos: float
) -> RespostaDoR:
    def resposta(
        desfecho: DesfechoDoLancador,
        motivo: str,
        *,
        dados: Mapping[str, object] | None = None,
        recusa: ViasRecusa | None = None,
    ) -> RespostaDoR:
        return RespostaDoR(
            desfecho, motivo, dados=dados, recusa=recusa, stderr=_resumo(stderr), segundos=segundos
        )

    linhas = [ln for ln in stdout.splitlines() if ln.strip()]
    if not linhas:
        if codigo != 0:
            return resposta("motor_falhou", f"o R saiu com {codigo} sem responder")
        return resposta("resposta_malformada", "o R não escreveu resposta")
    try:
        lido = json.loads(linhas[-1])
    except json.JSONDecodeError:
        return resposta("resposta_malformada", "a última linha do R não é JSON")
    if not isinstance(lido, dict):
        return resposta("resposta_malformada", "a resposta do R não é objeto JSON")
    status = lido.get("status")
    if status not in ("ok", "erro"):
        return resposta("resposta_malformada", f"status desconhecido: {status!r}")
    ambiente = lido.get("ambiente")
    libs = ambiente.get("libPaths") if isinstance(ambiente, dict) else None
    lista_libs = [libs] if isinstance(libs, str) else libs
    if (
        not isinstance(lista_libs, list)
        or not lista_libs
        or not all(isinstance(x, str) and _dentro(x, r_home) for x in lista_libs)
    ):
        return resposta(
            "motor_falhou", f"isolamento: o R usou bibliotecas fora da árvore embarcada: {libs!r}"
        )
    if status == "erro":
        desfecho = lido.get("desfecho")
        try:
            validate_label("ViasRecusa", desfecho)
        except InvalidLabelError:
            return resposta("resposta_malformada", f"desfecho de recusa desconhecido: {desfecho!r}")
        motivo = str(lido.get("motivo", ""))
        if desfecho == "motor_falhou":
            return resposta("motor_falhou", motivo)
        return resposta("recusado", motivo, recusa=cast(ViasRecusa, desfecho))
    if codigo != 0:
        return resposta("motor_falhou", f"o R respondeu ok e saiu com {codigo}")
    return resposta("ok", "", dados=lido)


def corre_roteiro_de_vias(
    pedido: Mapping[str, object],
    *,
    r_home: Path,
    roteiro: Path,
    tempo_limite_s: float | None = TEMPO_LIMITE_PADRAO_S,
    cancelado: Callable[[], bool] = lambda: False,
    comando: Sequence[str] | None = None,
) -> RespostaDoR:
    """Faz uma chamada ao roteiro. Não levanta por indisponibilidade; devolve o desfecho.

    `tempo_limite_s=None` é a chamada sem teto, só com o cancelamento. É a primeira de cada
    processo, que paga o cache frio e o antivírus.

    `comando` substitui o `Rscript` da árvore e existe para o duplo de processo dos testes: o resto
    da receita (ambiente, Job, temporário, leitura da resposta) é o mesmo.
    """
    rscript = rscript_da_arvore(r_home)
    if comando is None and not rscript.is_file():
        return RespostaDoR("motor_ausente", f"o R embarcado não está em {rscript}")
    if not roteiro.is_file():
        return RespostaDoR("motor_ausente", f"o roteiro das vias não está em {roteiro}")
    args = (
        [*comando, str(roteiro)]
        if comando is not None
        else [str(rscript), "--vanilla", str(roteiro)]
    )
    entrada = json.dumps(dict(pedido), ensure_ascii=False).encode("utf-8")
    raiz = Path(tempfile.mkdtemp(prefix="refino-vias-"))
    job: _Job | None = None
    inicio = time.monotonic()
    try:
        job = _Job()
        env = ambiente_isolado(raiz)
        # A entrada vai de arquivo, como a saída e o erro, e não por tubo. Escrever num tubo antes
        # do laço abaixo bloqueia até o R arrancar e ler, e um pedido maior que o buffer do tubo
        # (um fundo de algumas centenas de nomes) deixaria o cancelamento e o teto sem efeito
        # durante o arranque. Com arquivo não há escrita que espere o filho.
        entrada_txt = raiz / "stdin.json"
        entrada_txt.write_bytes(entrada)
        saida, erro = raiz / "stdout.txt", raiz / "stderr.txt"
        with (
            entrada_txt.open("rb") as f_in,
            saida.open("wb") as f_out,
            erro.open("wb") as f_err,
        ):
            processo = subprocess.Popen(
                args,
                stdin=f_in,
                stdout=f_out,
                stderr=f_err,
                cwd=raiz,
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                job.adota(processo)
            except OSError:
                processo.kill()
                processo.wait()
                raise
            while True:
                try:
                    codigo = processo.wait(timeout=_INTERVALO_S)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if cancelado():
                    job.termina(processo)
                    processo.wait()
                    return RespostaDoR(
                        "cancelado", "a análise foi cancelada", segundos=time.monotonic() - inicio
                    )
                if tempo_limite_s is not None and time.monotonic() - inicio > tempo_limite_s:
                    job.termina(processo)
                    processo.wait()
                    return RespostaDoR(
                        "tempo_esgotado",
                        f"o R passou de {tempo_limite_s:.0f} s",
                        segundos=time.monotonic() - inicio,
                    )
        segundos = time.monotonic() - inicio
        stdout = saida.read_bytes().decode("utf-8", errors="replace")
        stderr = erro.read_bytes().decode("utf-8", errors="replace")
        return _interpreta(stdout, stderr, codigo, r_home, segundos)
    except OSError as e:
        return RespostaDoR(
            "motor_falhou", f"o R não pôde ser lançado: {e}", segundos=time.monotonic() - inicio
        )
    finally:
        if job is not None:
            job.fecha()
        _apaga(raiz)


def _apaga(raiz: Path) -> None:
    # O processo acabou de morrer e o Windows pode segurar um arquivo por instantes.
    for _ in range(20):
        with contextlib.suppress(OSError):
            shutil.rmtree(raiz)
        if not raiz.exists():
            return
        time.sleep(0.1)

"""Banco SQLite do aplicativo de mesa.

Reaproveita `persistence/config.py` (`url_do_banco()`, em que `REFINO_DB_URL` vence, e
`prepara_sqlite()`): este módulo só escolhe o caminho e chama o que já existe. Fora do congelado o
banco fica no diretório de dados do usuário, e no Windows a ACL herdada de `%LOCALAPPDATA%` já o
restringe à conta do usuário, ao `SYSTEM` e aos administradores (lá o `chmod` só alterna
somente-leitura). No congelado ele fica ao lado do executável, para que a pasta entregue contenha
tudo e baste zipá-la para devolver. Os custos aceitos: a pasta pode ser somente-leitura, e então o
arranque falha com mensagem, sem recuo para outro lugar; duas pessoas que usem a mesma pasta
partilham o registro; e o arquivo herda a ACL de onde a pasta foi extraída, o que
`quem_le_o_diretorio` mede e o arranque escreve no `refino.log`, sem recusar.

O `alembic` é configurado em memória, com `script_location` apontando para o pacote: o
`alembic.ini` usa um caminho relativo ao repositório, que não existe num `--onedir`. O `env.py`
aceita um `Config()` sem arquivo, porque só chama `fileConfig` quando há arquivo e tira a URL de
`url_do_banco()`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from refino.persistence.config import (
    VARIAVEL_DA_URL,
    ConfiguracaoDeBancoError,
    caminho_do_sqlite,
    prepara_sqlite,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Callable, Mapping

NOME_DO_ARQUIVO: Final = "refino.db"
"""O arquivo do banco. O nome não é configurável: seria mais um lugar onde duas execuções olham
bancos diferentes sem ninguém notar."""

PASTA: Final = "refino"
"""A pasta dentro do diretório de dados do usuário."""


class BancoDeMesaError(RuntimeError):
    """O banco do aplicativo de mesa não pôde ser preparado.

    Tem causa `ambiente` no vocabulário de `interface/cliente.py`: quem opera não conserta permissão
    de
    diretório nem disco cheio, e a mensagem nomeia o caminho porque é a única coisa acionável.
    """


@dataclass(frozen=True)
class BancoDeMesa:
    """O que o arranque precisa saber depois de preparar o banco."""

    url: str
    caminho: Path
    ja_existia: bool
    """Se o arquivo já existia antes desta preparação; separa o primeiro arranque dos seguintes."""

    interrompidas: tuple[str, ...] = ()
    """As execuções de multiverso que este arranque marcou como `interrompido`.

    Os identificadores, e não a contagem, para que a janela possa abrir cada uma no histórico. Tupla
    vazia é o caso normal; a falha ao marcar sai por `stderr` em `_fecha_orfas`, porque é
    diagnóstico de defeito e não estado do trabalho do usuário."""


def esta_congelado(congelado: bool | None = None) -> bool:
    """Diz se o programa corre de dentro de um executável do PyInstaller.

    É uma função só para que `diretorio_de_dados` (onde o registro cai) e `prepara` (que saída a
    mensagem de erro oferece) nunca discordem. O parâmetro deixa o ramo do congelado testável sem
    congelar nada, já que `sys.frozen` só existe lá.
    """
    return bool(getattr(sys, "frozen", False)) if congelado is None else congelado


def diretorio_de_dados(
    ambiente: Mapping[str, str] | None = None,
    *,
    plataforma: str | None = None,
    congelado: bool | None = None,
    executavel: str | None = None,
) -> Path:
    """O diretório onde o banco e o log vivem: ao lado do executável, quando congelado.

    É a única ramificação por plataforma do pacote `desktop`. Os quatro argumentos entram por
    injeção para que todos os ramos sejam testáveis sem congelar e em qualquer sistema.

    - congelado, em qualquer sistema: a pasta que contém o executável;
    - `win32`: `%LOCALAPPDATA%`, ou `%APPDATA%` se aquele faltar, mais `PASTA`;
    - outro: `$XDG_DATA_HOME`, ou `~/.local/share`, mais `PASTA`.

    O congelado vence a plataforma porque 'onde o usuário acha o arquivo' tem a mesma resposta nos
    dois sistemas. Ele não acrescenta `PASTA`, para que o `refino.db` apareça na primeira tela da
    pasta, junto do `LEIA-ME.txt` e do `VERSAO.txt`.
    """
    env = os.environ if ambiente is None else ambiente
    alvo = sys.platform if plataforma is None else plataforma

    if esta_congelado(congelado):
        binario = (sys.executable if executavel is None else executavel).strip()
        if not binario:
            raise BancoDeMesaError(
                "o programa diz estar congelado e não sabe o caminho do próprio executável: não há "
                "pasta ao lado da qual pôr o registro. Extraia outra vez a pasta do programa a "
                "partir do arquivo compactado original, e abra-o de lá"
            )
        return Path(binario).parent

    if alvo == "win32":
        base = (env.get("LOCALAPPDATA") or "").strip() or (env.get("APPDATA") or "").strip()
        if not base:
            raise BancoDeMesaError(
                "nem LOCALAPPDATA nem APPDATA estão definidos: não há diretório por usuário onde "
                "pôr o banco. Defina "
                f"{VARIAVEL_DA_URL} — por exemplo `sqlite:///C:/Users/<voce>/refino.db`"
            )
        return Path(base) / PASTA

    xdg = (env.get("XDG_DATA_HOME") or "").strip()
    if xdg:
        return Path(xdg) / PASTA
    lar = (env.get("HOME") or "").strip()
    if not lar:
        raise BancoDeMesaError(
            "HOME não está definido e não há XDG_DATA_HOME: não há diretório por usuário onde pôr "
            f"o banco. Defina {VARIAVEL_DA_URL}"
        )
    return Path(lar) / ".local" / "share" / PASTA


def url_de_mesa(
    ambiente: Mapping[str, str] | None = None,
    *,
    plataforma: str | None = None,
    congelado: bool | None = None,
    executavel: str | None = None,
) -> str:
    """A URL de SQLite do aplicativo de mesa, com caminho absoluto.

    Três barras e o caminho absoluto, como o SQLAlchemy exige: um caminho relativo faria o banco
    seguir o `cwd`, que num executável é onde o usuário clicou.
    """
    caminho = (
        diretorio_de_dados(
            ambiente, plataforma=plataforma, congelado=congelado, executavel=executavel
        )
        / NOME_DO_ARQUIVO
    )
    return f"sqlite:///{caminho.as_posix()}"


# ---------------------------------------------------------------------------
# Quem lê o que este programa grava
# ---------------------------------------------------------------------------
#
# Ao lado do executável, a proteção do banco depende de onde a pasta foi extraída, e uma proteção
# que varia sem ser reportada equivale a nenhuma. Esta seção a mede.

SIDS_DE_MAIS_DE_UMA_CONTA: Final[Mapping[str, str]] = {
    "S-1-1-0": "qualquer pessoa que use este computador (Todos)",
    "S-1-2-0": "qualquer pessoa sentada a este computador",
    "S-1-5-3": "as tarefas agendadas desta máquina",
    "S-1-5-4": "qualquer conta que abra sessão nesta máquina",
    "S-1-5-6": "os serviços desta máquina",
    "S-1-5-7": "acesso sem conta nenhuma (anônimo)",
    "S-1-5-11": "qualquer conta autenticada desta máquina",
    "S-1-5-113": "qualquer conta local desta máquina",
    "S-1-5-114": "qualquer conta local do grupo de administradores",
    "S-1-5-32-545": "o grupo Usuários — todas as contas comuns desta máquina",
    "S-1-5-32-546": "o grupo Convidados",
    "S-1-5-32-547": "o grupo Usuários Avançados",
}
"""SIDs do Windows que significam mais de uma conta, com o que dizer a quem opera.

SIDs e não nomes, porque `icacls` e `Get-Acl` imprimem nomes traduzidos (`BUILTIN\\Usuários`,
`Todos`), e um verificador que comparasse literais em inglês diria 'restrito' em toda máquina
localizada. A lista vem da tabela de SIDs conhecidos da Microsoft. Um SID amplo fora dela, como um
grupo local criado à mão ou um grupo de domínio, não vira `amplo`; por isso `herdam_leitura`
carrega todos os SIDs achados, e o `refino.log` mostra a lista inteira."""

_ACE_QUE_PERMITE: Final = 0
"""`ACCESS_ALLOWED_ACE_TYPE`. As entradas de negação (tipo 1) não contam: tiram acesso, e a
pergunta é quem tem."""

_HERDA_PARA_ARQUIVOS: Final = 0x01
"""`OBJECT_INHERIT_ACE`. Sem este bit a entrada vale para o diretório e não desce para os arquivos
criados nele."""

_MASCARAS_DE_LEITURA: Final = 0x00000001 | 0x10000000 | 0x80000000
"""`FILE_READ_DATA`, `GENERIC_ALL` e `GENERIC_READ`.

Os três, porque uma entrada genérica, como a de `S-1-5-11` na raiz `C:\\` (máscara `0xe0010000`),
concede leitura sem ter o bit `0x1` posto."""


class LeituraDeAclError(OSError):
    """A lista de controle de acesso do diretório não pôde ser lida.

    Erro e não tupla vazia: vazia significaria ao mesmo tempo 'ninguém mais lê' e 'não consegui
    medir', e a primeira leitura é a que tranquiliza.
    """


def _sids_com_leitura_herdavel(diretorio: Path) -> tuple[str, ...]:
    """Os SIDs que herdam leitura sobre os arquivos criados dentro de `diretorio`. Só Windows.

    Lê a DACL do diretório com `GetNamedSecurityInfoW` e devolve os SIDs das entradas que permitem,
    descem para arquivos e concedem leitura. O referente é o diretório porque é ele que decide o que
    herdam os arquivos novos: o `refino.db`, os auxiliares do SQLite e o `refino.log`.

    Usa `ctypes` e não `icacls` nem `Get-Acl`: estes imprimem nomes traduzidos e, lançados de um
    congelado `--windowed`, abrem um console que pisca. O `if` de plataforma protege o import de
    `ctypes.wintypes`, que levanta fora do Windows; a política fica em `quem_le_o_diretorio`.
    """
    if sys.platform != "win32":  # pragma: no cover - `ctypes.wintypes` nem importa fora do Windows
        raise LeituraDeAclError(f"a leitura de ACL é do Windows, e este sistema é {sys.platform}")

    import ctypes
    from ctypes import wintypes

    class _Acl(ctypes.Structure):
        _fields_ = (
            ("AclRevision", ctypes.c_ubyte),
            ("Sbz1", ctypes.c_ubyte),
            ("AclSize", ctypes.c_ushort),
            ("AceCount", ctypes.c_ushort),
            ("Sbz2", ctypes.c_ushort),
        )

    class _CabecalhoDaAce(ctypes.Structure):
        _fields_ = (
            ("AceType", ctypes.c_ubyte),
            ("AceFlags", ctypes.c_ubyte),
            ("AceSize", ctypes.c_ushort),
        )

    class _AceAteAMascara(ctypes.Structure):
        # O SID começa logo depois destes dois campos, e o `sizeof` desta estrutura dá o
        # deslocamento dele (8 bytes) sem número escrito à mão.
        _fields_ = (("Header", _CabecalhoDaAce), ("Mask", wintypes.DWORD))

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.GetAce.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL

    ponteiro_da_dacl = ctypes.POINTER(_Acl)()
    descritor = ctypes.c_void_p()
    se_file_object = 1
    dacl_security_information = 0x00000004

    codigo = advapi32.GetNamedSecurityInfoW(
        ctypes.c_wchar_p(str(diretorio)),
        ctypes.c_int(se_file_object),
        wintypes.DWORD(dacl_security_information),
        None,
        None,
        ctypes.byref(ponteiro_da_dacl),
        None,
        ctypes.byref(descritor),
    )
    if codigo != 0:
        raise LeituraDeAclError(f"GetNamedSecurityInfoW devolveu {codigo} para {diretorio}")

    achados: list[str] = []
    try:
        if not ponteiro_da_dacl:  # pragma: no cover - exige um objeto sem DACL; ver abaixo
            # DACL nula quer dizer acesso total a todos, e não 'nenhuma entrada': devolver vazio
            # diria 'restrito' sobre o caso mais amplo. Sem cobertura porque `icacls` não escreve
            # DACL nula e não há como plantá-la.
            return ("S-1-1-0",)
        for indice in range(ponteiro_da_dacl.contents.AceCount):
            ace = ctypes.c_void_p()
            if not advapi32.GetAce(
                ponteiro_da_dacl, wintypes.DWORD(indice), ctypes.byref(ace)
            ):  # pragma: no cover - só falha com índice fora do `AceCount` que acabámos de ler
                continue
            campos = ctypes.cast(ace, ctypes.POINTER(_AceAteAMascara)).contents
            if campos.Header.AceType != _ACE_QUE_PERMITE:
                continue
            if not campos.Header.AceFlags & _HERDA_PARA_ARQUIVOS:
                continue
            if not campos.Mask & _MASCARAS_DE_LEITURA:
                continue
            texto = wintypes.LPWSTR()
            sid = ctypes.c_void_p(ctypes.addressof(campos) + ctypes.sizeof(_AceAteAMascara))
            if not advapi32.ConvertSidToStringSidW(
                sid, ctypes.byref(texto)
            ):  # pragma: no cover - só falha com SID malformado dentro de uma ACE válida
                continue
            if texto.value is not None:  # pragma: no cover - `texto` vem preenchido de um sucesso
                achados.append(texto.value)
            kernel32.LocalFree(texto)
    finally:
        if descritor:
            kernel32.LocalFree(descritor)
    return tuple(sorted(set(achados)))


Veredito = Literal["restrito", "amplo", "nao_medido"]
"""Os três desfechos. 'Não consegui medir' e 'medi e ninguém mais lê' pedem ações opostas a quem
lê o log, e dois valores colapsariam as duas no valor tranquilizador."""


@dataclass(frozen=True)
class LeituraDoDiretorio:
    """Quem herda leitura sobre o que este programa grava, e o que dizer a quem opera."""

    veredito: Veredito
    diretorio: Path
    herdam_leitura: tuple[str, ...]
    """Todos os SIDs achados, não só os amplos (ver `SIDS_DE_MAIS_DE_UMA_CONTA`); é a evidência
    ao lado do veredito."""

    amplos: tuple[str, ...]
    detalhe: str = ""
    """Por que não foi medido. Vazio nos outros dois desfechos."""

    def para_quem_opera(self) -> str:
        """Uma frase em português, sem jargão, para o `refino.log`.

        Os três desfechos têm frase, inclusive `restrito`: uma verificação que só fala quando há
        problema não se distingue de uma que não correu.
        """
        if self.veredito == "amplo":
            # Dois pontos e não 'por': as descrições são sintagmas nominais, e 'lido por as
            # tarefas' é o que sairia de concatená-las depois da preposição.
            quem = ", ".join(SIDS_DE_MAIS_DE_UMA_CONTA.get(sid, sid) for sid in self.amplos)
            return (
                f"ATENÇÃO: o que este programa grava em {self.diretorio} não fica só com você. "
                f"Também leem: {quem}. O registro guarda o apelido e o diagnóstico de cada "
                "amostra. Se isso for um problema, feche o programa, mova a pasta inteira para a "
                "sua Área de Trabalho e abra-o de lá — os arquivos vão junto"
            )
        if self.veredito == "restrito":
            # A frase não afirma 'só a sua conta, os administradores e o Windows leem':
            # `SIDS_DE_MAIS_DE_UMA_CONTA` é uma lista negativa, e um grupo fora dela também
            # classifica como `restrito`. Ela diz o que foi medido e quantas entradas há, para quem
            # lê conferir na própria pasta.
            entradas = len(self.herdam_leitura)
            return (
                f"o que este programa grava fica em {self.diretorio}. A permissão dessa pasta tem "
                f"{entradas} entrada(s), e nenhuma delas é de um grupo que dê acesso às outras "
                "contas da máquina — normalmente são a sua conta, os administradores e o próprio "
                "Windows"
            )
        return (
            f"não foi possível medir quem lê {self.diretorio} ({self.detalhe}). O registro fica "
            "ali de todo modo"
        )


def quem_le_o_diretorio(
    diretorio: Path,
    *,
    plataforma: str | None = None,
    leitor: Callable[[Path], tuple[str, ...]] | None = None,
) -> LeituraDoDiretorio:
    """Mede quem herda leitura sobre `diretorio`. Avisa, e nunca recusa nem levanta.

    Recusar deixaria quem recebeu a pasta com um problema que ele não sabe nomear; o desfecho é uma
    frase no `refino.log`, que volta com a pasta. `plataforma` e `leitor` entram por injeção para
    que os três desfechos, inclusive `amplo`, sejam testáveis em qualquer sistema.
    """
    sistema = sys.platform if plataforma is None else plataforma
    if sistema != "win32":
        return LeituraDoDiretorio(
            veredito="nao_medido",
            diretorio=diretorio,
            herdam_leitura=(),
            amplos=(),
            detalhe=(
                f"a leitura de permissões implementada aqui é do Windows, e o sistema é {sistema}"
            ),
        )

    le = _sids_com_leitura_herdavel if leitor is None else leitor
    try:
        sids = le(diretorio)
    except (OSError, ValueError) as erro:
        return LeituraDoDiretorio(
            veredito="nao_medido",
            diretorio=diretorio,
            herdam_leitura=(),
            amplos=(),
            detalhe=f"{type(erro).__name__}: {erro}",
        )

    amplos = tuple(sid for sid in sids if sid in SIDS_DE_MAIS_DE_UMA_CONTA)
    return LeituraDoDiretorio(
        veredito="amplo" if amplos else "restrito",
        diretorio=diretorio,
        herdam_leitura=sids,
        amplos=amplos,
    )


def _diretorio_das_migracoes() -> Path:
    """O diretório do `alembic`, resolvido a partir do pacote pai e não do repositório.

    `migrations/` não tem `__init__.py` e importa como pacote de namespace, sem `__file__`; por
    isso a derivação parte de `refino.persistence`, que é pacote regular. Vale do código-fonte e de
    dentro de um `--onedir`, onde o PyInstaller reescreve `__file__` para a árvore extraída.
    """
    from refino import persistence

    if persistence.__file__ is None:  # pragma: no cover - `persistence` tem `__init__.py`
        raise BancoDeMesaError(
            "não foi possível localizar as migrações: `refino.persistence` não tem `__file__`. "
            "Isto só acontece se o pacote virar namespace, o que seria mudança de layout"
        )
    return Path(persistence.__file__).parent / "migrations"


def migra(url: str) -> None:
    """Sobe o banco até `head`, configurando o `alembic` em memória.

    A URL vai pelo ambiente e não por `sqlalchemy.url`, porque o `env.py` a tira de
    `url_do_banco()`, a mesma função da aplicação; uma segunda fonte para o mesmo valor poderia
    divergir.
    """
    from alembic import command
    from alembic.config import Config

    config = Config()  # sem arquivo: `env.py` guarda o `fileConfig` atrás de um `is not None`
    config.set_main_option("script_location", str(_diretorio_das_migracoes()))

    anterior = os.environ.get(VARIAVEL_DA_URL)
    os.environ[VARIAVEL_DA_URL] = url
    try:
        command.upgrade(config, "head")
    finally:
        if anterior is None:
            os.environ.pop(VARIAVEL_DA_URL, None)
        else:
            os.environ[VARIAVEL_DA_URL] = anterior


def prepara(
    ambiente: Mapping[str, str] | None = None,
    *,
    plataforma: str | None = None,
    congelado: bool | None = None,
    executavel: str | None = None,
    migrar: bool = True,
) -> BancoDeMesa:
    """Resolve o caminho, cria o arquivo com a permissão que a plataforma dá, e migra.

    `migrar=False` existe para o teste que só quer o caminho e a permissão. O arranque sempre migra:
    sem tabelas, o erro só apareceria na primeira gravação, longe da causa.
    """
    # `REFINO_DB_URL` vence, como promete `config.url_do_banco`, e a mensagem de erro do ramo não
    # congelado manda defini-la. Só SQLite é aceito: `BancoDeMesa.caminho` é `Path`, e o arranque
    # mede a ACL dele e mostra o nome do arquivo; uma URL de outro banco sai por erro nomeado.
    declarada = ((os.environ if ambiente is None else ambiente).get(VARIAVEL_DA_URL) or "").strip()
    if declarada and not declarada.startswith("sqlite"):
        raise BancoDeMesaError(
            f"{VARIAVEL_DA_URL} está definida como `{declarada.split('://')[0]}://…` e o caminho "
            f"de mesa é de SQLite. Aponte-a para um arquivo — `sqlite:///C:/Users/<você>/"
            f"refino.db` — ou apague a variável para usar o lugar padrão"
        )

    url = declarada or url_de_mesa(
        ambiente, plataforma=plataforma, congelado=congelado, executavel=executavel
    )
    caminho = caminho_do_sqlite(url) or (
        diretorio_de_dados(
            ambiente, plataforma=plataforma, congelado=congelado, executavel=executavel
        )
        / NOME_DO_ARQUIVO
    )
    ja_existia = caminho.is_file()

    try:
        prepara_sqlite(url, congelado=congelado)
    except (OSError, ConfiguracaoDeBancoError) as erro:
        # `ConfiguracaoDeBancoError` é `ValueError`, não `OSError`, e `prepara_sqlite` a levanta
        # para banco dentro da árvore do repositório; sem ela aqui, sairia como traceback cru.
        # A saída sugerida muda com o congelado porque muda quem lê: lá a ação é mover a pasta, e
        # fora dele quem lê desenvolve e resolve pela configuração.
        saida = (
            "Mova a pasta inteira do programa para a Área de Trabalho e abra-o de lá: esta pasta "
            "não aceita gravação"
            if esta_congelado(congelado)
            else f"Verifique a permissão do diretório, ou defina {VARIAVEL_DA_URL} para outro lugar"
        )
        raise BancoDeMesaError(f"não foi possível preparar {caminho}: {erro}. {saida}") from erro

    if migrar:
        try:
            migra(url)
        except Exception as erro:
            # Largo de propósito: o `alembic` pode levantar `OperationalError` (banco trancado por
            # outra instância), `CommandError` (revisão desconhecida) ou `DatabaseError` (arquivo
            # que não é SQLite), e nenhum é `OSError`. A mensagem nomeia o caminho e o tipo, não o
            # texto do erro: mensagens do SQLAlchemy carregam a URL, e a de Postgres traz a senha.
            raise BancoDeMesaError(
                f"o banco em {caminho} não pôde ser migrado ({type(erro).__name__}). Se o arquivo "
                f"está corrompido, apague-o e abra o aplicativo outra vez; se há outra instância "
                f"aberta, feche-a. O detalhe fica fora desta mensagem de propósito: mensagens de "
                f"banco carregam a URL, e ela carrega a senha"
            ) from erro

    return BancoDeMesa(
        url=url,
        caminho=caminho,
        ja_existia=ja_existia,
        # As órfãs só são fechadas quando há migração: sem ela a tabela pode nem existir.
        interrompidas=() if not migrar else _fecha_orfas(url),
    )


def _fecha_orfas(url: str) -> tuple[str, ...]:
    """Marca como `interrompido` toda execução de multiverso que ficou aberta. Nunca levanta.

    O registro de progresso vive em memória, por processo. Um processo novo não tem execução em
    curso, então toda linha `executando` que ele encontra é de um processo que morreu no meio de uma
    grade. Fechá-las no arranque é o que permite gravar o progresso sem deixar um 'executando'
    eterno.

    Uma falha aqui não impede o programa de abrir, porque é limpeza de histórico. Ela sai por
    `stderr`, e os identificadores marcados voltam a quem chama, para a janela mostrá-los.
    """
    from sqlalchemy import create_engine

    from refino.persistence.repositories import MultiversoRepository

    motor = None
    try:
        motor = create_engine(url)
        return MultiversoRepository(motor).fecha_orfas()
    except Exception as erro:
        print(
            f"[refino] as execucoes de multiverso deixadas em aberto nao puderam ser marcadas "
            f"como interrompidas ({type(erro).__name__}). O aplicativo abre na mesma; o historico "
            f"pode mostrar uma execucao antiga como se ainda estivesse a correr",
            file=sys.stderr,
        )
        return ()
    finally:
        if motor is not None:
            motor.dispose()

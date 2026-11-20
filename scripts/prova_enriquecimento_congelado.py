"""Carga da prova do enriquecimento no congelado — corre DENTRO da pasta, pelo interpretador dela.

Chamada por `scripts/prova_enriquecimento_congelado.ps1` com o `pythonw.exe` de `_internal`, ela
importa o `refino` empacotado e chama o lançador de produção sobre a pasta `R` e a pasta
`vias-dados` da própria pasta congelada. Grava o resultado num JSON, porque `pythonw` não tem
console. O que ela afirma — e cada item é o que só o congelado mede:

1. **o motor resolve a partir do executável**, não do `PATH`;
2. **o roteiro da pasta é o do repositório** (sha256 igual ao que o congelamento copiou);
3. **mapear e ORA local calculam** com as bibliotecas fixadas, e o p de cada via confere com o
   hipergeométrico das contagens;
4. **com o diretório de DLL do PyInstaller herdado** (`SetDllDirectoryW(_internal)`, que é o que o
   bootloader faz no processo do `refino.exe`), as DLLs que o R carrega vêm todas da pasta `R` —
   nenhuma de `_internal`;
   **O item 4 afirma menos do que diz.** O oráculo é
   `getLoadedDLLs()`, que lista só as DLLs que o R registra por `dyn.load`/`library.dynam`
   (caminho absoluto dentro de `library/`). Uma dependência que o Windows resolve pelo nome, onde
   o diretório herdado atuaria, fica fora da lista. A herança de `SetDllDirectoryW(_internal)`
   pelo `Rscript` filho é inferida do retorno da chamada no processo pai, e não medida no filho.
   O desfecho «fora» ainda não foi visto, e o limite fica aberto.
5. **cancelar encerra o R e não sobra diretório temporário da corrida** — nem a pasta do lançador
   (`refino-vias-*`) nem o `Rtmp*` da sessão do R, os dois procurados no `%TEMP%`;
6. **o modo externo, SEM REDE** — em `prova_externa`, numa chave própria do JSON e num `try`
   próprio, para que uma falha do R não esconda este item nem o contrário:
   a. `monta_carga` e a G2 (`confere_carga`) aprovam a carga dos nomes públicos desta prova;
   b. a G2 RECUSA cada carga plantada — um nome de amostra entre os nomes, e o vocabulário fixo
      trocado nos bytes já montados —, e a prova confere que a planta aterrou e que o motivo é o
      dela;
   c. uma resposta SINTÉTICA, montada com `escreve_rds` e vias inventadas, atravessa
      `envia_ao_servidor` por uma conexão FALSA injetada, é lida pelo leitor restrito, e
      `ora_da_resposta` confere cada p contra a hipergeométrica;
   d. a resposta com função (os bytes de `data/fixtures/rds/com_funcao.ser`, embutidos aqui porque
      `data/` não viaja) é recusada pelo leitor, com o tipo recusado no motivo.

   **O que só o congelado mede neste item:** que `refino.io.enriquecimento_externo` e
   `refino.io.rds_restrito` viajaram e funcionam no interpretador embarcado. Ele lê a cópia solta
   que o `.spec` deriva de `analise.pure`, a mesma lista de que sai o PYZ do `refino.exe`. O `ssl` e
   o `http.client` não são novidade deste item: `refino.io.dados_vias`, que o item 1 importa, já os
   importa.

   **Nada fala com a rede, e isso é MEDIDO e não só prometido:** durante o item, `sem_rede` troca a
   resolução de nome (`socket.getaddrinfo`) e a conexão (`socket.socket.connect` e `connect_ex`),
   que são os dois passos por onde o `http.client` sai, por funções que anotam a tentativa e
   levantam. O JSON leva as tentativas, e a prova reprova se houver alguma.

   **Escopo negativo:** o item não constrói o contexto TLS, não verifica certificado e não envia.
   O envio bem-sucedido não é conferido no congelado. O envio real a partir da pasta congelada foi
   feito à parte, uma vez, sem a janela; esta prova continua sem enviar.

**O item 5 já olhou só `refino-vias-*`.** Ele deu OK em
`7f89f0f` quando, pelo que a mensagem de `16ad9ca` registra, cancelar deixava um `Rtmp*` no
`%TEMP%`: o instrumento afirmava algo sobre a corrida e só via o prefixo do lançador.
Desde `16ad9ca` o `Rtmp*` nasce dentro de `refino-vias-*`, e a cobertura dele passara a ser
indireta. A diferença de conjuntos é agora uma função pura, `sobras`, com o desfecho de reprovação
visto em processo por `tests/test_prova_enriquecimento_congelado.py` — o congelado não é preciso
para isso, e o contrato desta carga (os dois argumentos e as chaves do JSON) não mudou.

**O que o padrão `Rtmp*` aceita como ruído:** outro R da máquina que arranque entre as duas
fotografias — antes e depois da chamada cancelada — cria um `Rtmp*` no mesmo `%TEMP%`, e a prova o
acusa. O desfecho do ruído é uma reprovação a mais e não uma aprovação a mais, que é a direção
barata.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import socket
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, NoReturn

if TYPE_CHECKING:
    from refino.io.enriquecimento_externo import RespostaDoServidor

# Os dois rastros que uma corrida pode deixar no `%TEMP%`: a pasta que o lançador cria por corrida,
# e o `Rtmp*` da sessão do R quando o lançador não o desvia para dentro dela.
PADROES_DE_TEMPORARIO: Final = ("refino-vias-*", "Rtmp*")

# Os nomes públicos da prova: compostos do metabolismo central, sem nada do estudo. Servem ao R
# (mapear e ORA local) e ao modo externo (a carga que a G2 tem de aprovar).
NOMES_DA_PROVA: Final = (
    "Glycine",
    "L-Alanine",
    "Citric acid",
    "Pyruvic acid",
    "Succinic acid",
    "Fumaric acid",
    "L-Glutamic acid",
    "L-Glutamine",
    "Oxoglutaric acid",
    "Urea",
    "L-Ornithine",
)

# ---------------------------------------------------------------------------
# O modo externo, sem rede (item 6)
# ---------------------------------------------------------------------------

# Um nome com a forma de rótulo de amostra: é o que a G2 existe para não deixar sair.
AMOSTRA_PLANTADA: Final = "QC_PLANTADO_07"

# A resposta sintética. As vias e o universo são INVENTADOS: a resposta real do servidor traz dados
# derivados do KEGG, cuja licença não permite redistribuir, e nenhuma é versionada. O universo não é
# o 1.519 medido em 01/10/2026, para que nada aqui se leia como dado do servidor.
UNIVERSO_SINTETICO: Final = 1000
VIAS_INVENTADAS: Final = (  # (via, total na via, acertos)
    ("Via inventada A da prova", 20, 4),
    ("Via inventada B da prova", 60, 2),
    ("Via inventada C da prova", 9, 1),
)
COLUNAS_DA_ORA: Final = ("total", "expected", "hits", "Raw p", "Holm p", "FDR")
# Os acertos de cada via são os primeiros nomes da carga, então os nomes em alguma via são os
# primeiros `max(acertos)`: é a lista que o servidor usa nas contas (medido em 02/10/2026).
NA_LISTA_SINTETICA: Final = max(a for _, _, a in VIAS_INVENTADAS)

# Os 181 bytes de `data/fixtures/rds/com_funcao.ser` — `serialize(list(ok = 1, f = function(x)
# x + 1), NULL)`, gerado pelo R da árvore em 01/10/2026 (`gera.R` ao lado da fixture). Embutidos
# porque `data/` não viaja na pasta congelada; o teste desta carga confere que continuam iguais aos
# da fixture.
COM_FUNCAO: Final = bytes.fromhex(
    "580a000000030004040300030500000000055554462d3800000213000000020000"
    "000e000000013ff000000000000000000403000000fd0000040200000001000400"
    "090000000178000000fb000000fe000000060000000100040009000000012b0000"
    "0002000001ff000000020000000e000000013ff0000000000000000000fe000004"
    "020000000100040009000000056e616d6573000000100000000200040009000000"
    "026f6b000400090000000166000000fe"
)


def _bloqueia(rotulo: str, tentativas: list[str]) -> Callable[..., NoReturn]:
    def bloqueado(*args: object, **_: object) -> NoReturn:
        # `connect(self, endereco)`; `getaddrinfo(host, porta, ...)`.
        alvo = args[1] if rotulo.startswith("connect") else args[:2]
        tentativas.append(f"{rotulo} {alvo!r}")
        raise OSError(f"a prova bloqueia a rede: {rotulo} {alvo!r}")

    return bloqueado


_AUSENTE: Final = object()


@contextlib.contextmanager
def sem_rede(tentativas: list[str]) -> Iterator[None]:
    """Durante o bloco, resolver nome e conectar socket falham, e cada tentativa fica anotada.

    São os dois passos por onde o `http.client` sai — `socket.create_connection` resolve por
    `getaddrinfo` e conecta por `socket.socket.connect` —, e trocá-los pelo bloqueio antes da
    syscall faz a promessa *«nada fala com a rede»* virar contagem no JSON. A restauração devolve o
    estado exacto: `connect` não está no dicionário de `socket.socket` (é herdado de
    `_socket.socket`), e deixar lá o descritor herdado seria uma mudança a mais.
    """
    alvos = ((socket, "getaddrinfo"), (socket.socket, "connect"), (socket.socket, "connect_ex"))
    antes = [(dono, nome, vars(dono).get(nome, _AUSENTE)) for dono, nome in alvos]
    for dono, nome in alvos:
        setattr(dono, nome, _bloqueia(nome, tentativas))
    try:
        yield
    finally:
        for dono, nome, valor in antes:
            if valor is _AUSENTE:
                delattr(dono, nome)
            else:
                setattr(dono, nome, valor)


def resposta_sintetica(
    nomes: Sequence[str],
    *,
    universo_declarado: int | None = None,
    vias_declaradas: Sequence[str] | None = None,
) -> bytes:
    """A resposta do servidor na forma que `le_resposta` lê, com as vias inventadas.

    Cada p é o hipergeométrico das contagens sobre `UNIVERSO_SINTETICO`, e é por isso que
    `ora_da_resposta` a aceita. Os dois argumentos nomeados existem para as plantas do teste, e cada
    um vira um campo diferente do registro:

    - `universo_declarado`: declarar um universo e calcular com outro é o defeito que a conferência
      dos p existe para pegar (a docstring de `pathways.enriquecimento.p_da_via_confere`) — vira
      `ora_confere`;
    - `vias_declaradas`: os nomes das vias, na ordem de `VIAS_INVENTADAS`, trocados com as contagens
      intactas. Os p continuam coerentes, a ORA confere, e só a comparação dos nomes acusa — vira
      `vias_conferem`, que a planta do universo não alcança porque desvia para o `except`.

    Holm e FDR não são conferidos por `ora_da_resposta` e vão como Bonferroni, que é cota superior
    dos dois. A lista das contas é a dos nomes que estão em ALGUMA via — `NA_LISTA_SINTETICA` —, e
    não a dos enviados: é a regra do servidor, medida em 02/10/2026, e a que `le_resposta` devolve.
    """
    from refino.io.rds_restrito import Lista, Vetor, escreve_rds
    from refino.pathways.enriquecimento import p_hipergeometrico

    n = len(set(nomes[:NA_LISTA_SINTETICA]))
    linhas = []
    for _via, total, acertos in VIAS_INVENTADAS:
        p = p_hipergeometrico(acertos, total, n, UNIVERSO_SINTETICO)
        corrigido = min(1.0, p * len(VIAS_INVENTADAS))
        linhas.append(
            (float(total), total * n / UNIVERSO_SINTETICO, float(acertos), p, corrigido, corrigido)
        )
    if vias_declaradas is None:
        vias = tuple(v for v, _, _ in VIAS_INVENTADAS)
    else:
        vias = tuple(vias_declaradas)
    # O R guarda a matriz por coluna.
    valores = tuple(linha[j] for j in range(len(COLUNAS_DA_ORA)) for linha in linhas)
    matriz = Vetor(
        "real",
        valores,
        {
            "dim": Vetor("inteiro", (len(linhas), len(COLUNAS_DA_ORA))),
            "dimnames": Lista((Vetor("texto", vias), Vetor("texto", COLUNAS_DA_ORA))),
        },
    )
    acertos_por_via = Lista(
        tuple(Vetor("texto", tuple(nomes[:a])) for _, _, a in VIAS_INVENTADAS),
        {"names": Vetor("texto", vias)},
    )
    universo = UNIVERSO_SINTETICO if universo_declarado is None else universo_declarado
    raiz = Lista(
        (
            Lista((matriz, acertos_por_via), {"names": Vetor("texto", ("ora.mat", "ora.hits"))}),
            Lista((Vetor("inteiro", (universo,)),), {"names": Vetor("texto", ("uniq.count",))}),
        ),
        {"names": Vetor("texto", ("analSet", "dataSet"))},
    )
    return escreve_rds(raiz)


class _RespostaFalsa:
    """O que a conexão falsa devolve: 200, o tipo que o servidor declara, e os bytes dados."""

    status = 200
    reason = "OK"

    def __init__(self, corpo: bytes) -> None:
        self._corpo = corpo

    def getheader(self, nome: str, /) -> str | None:
        return "application/octet-stream" if nome.lower() == "content-type" else None

    def read(self, n: int = -1, /) -> bytes:
        return self._corpo if n < 0 else self._corpo[:n]


class ConexaoFalsa:
    """O transporte injetado em `envia_ao_servidor`: anota o pedido e devolve `resposta`.

    Nenhum socket é aberto. É o ponto de injeção que `envia_ao_servidor` declara (`conexao=`), e é
    o que deixa a prova atravessar o caminho de produção — corpo multipart, status, tipo, teto e
    leitor — sem sair da máquina.
    """

    def __init__(self, resposta: bytes) -> None:
        self.resposta = resposta
        self.visto: dict[str, object] = {}

    def __call__(self, host: str, tempo_limite: float) -> ConexaoFalsa:
        self.visto["host"] = host
        return self

    def request(self, metodo: str, url: str, body: bytes, headers: dict[str, str], /) -> None:
        self.visto.update(metodo=metodo, caminho=url, corpo=body)

    def getresponse(self) -> _RespostaFalsa:
        return _RespostaFalsa(self.resposta)

    def close(self) -> None:
        self.visto["fechada"] = True


def envia_sem_rede(
    carga: bytes, nomes: Sequence[str], resposta: bytes
) -> tuple[RespostaDoServidor, dict[str, object]]:
    """`envia_ao_servidor` pela conexão falsa: o desfecho, e o registro do que teria saído."""
    from refino.io import enriquecimento_externo as ext

    conexao = ConexaoFalsa(resposta)
    r = ext.envia_ao_servidor(carga, nomes, conexao=conexao)
    corpo = conexao.visto.get("corpo")
    registro: dict[str, object] = {
        "desfecho": r.desfecho,
        "motivo": r.motivo,
        "destino": f"https://{conexao.visto.get('host')}{conexao.visto.get('caminho')}",
        # Os bytes conferidos pela G2 vão exactamente uma vez no corpo — nem alterados, nem outros.
        "leva_a_carga": isinstance(corpo, bytes) and corpo.count(carga) == 1,
    }
    registro["destino_constante"] = registro["destino"] == ext.DESTINO
    return r, registro


def g2(carga: bytes, permitidos: Sequence[str]) -> tuple[bool, str]:
    """A G2 sobre `carga`, como desfecho: aprovada, e o motivo da recusa quando não."""
    from refino.io import enriquecimento_externo as ext

    try:
        ext.confere_carga(carga, permitidos)
    except ext.CargaRecusadaError as erro:
        return False, str(erro)
    return True, ""


def prova_externa(resultado: dict[str, object]) -> None:
    """Preenche `resultado` com o item 6, sob `sem_rede`. Ver a docstring do módulo."""
    from refino.io import enriquecimento_externo as ext
    from refino.io import rds_restrito
    from refino.pathways import enriquecimento as en

    tentativas: list[str] = []
    rede: dict[str, object] = {"tentativas": tentativas}
    resultado["rede"] = rede
    original = socket.getaddrinfo
    with sem_rede(tentativas):
        # A âncora da ausência abaixo: a guarda está de facto posta durante o item.
        rede["guarda_posta"] = socket.getaddrinfo is not original and "connect" in vars(
            socket.socket
        )
        nomes = list(NOMES_DA_PROVA)

        # a. a carga dos nomes da prova passa na G2
        carga = ext.monta_carga(nomes)
        aprovada, motivo = g2(carga, nomes)
        folhas = rds_restrito.inventario(rds_restrito.le_rds(carga))
        resultado["carga"] = {
            "aprovada": aprovada,
            "motivo": motivo,
            "bytes": len(carga),
            # lidos de volta dos bytes, e não da lista que se passou a `monta_carga`
            "nomes": sum(n for caminho, _, n in folhas if caminho == "raiz$oraVec"),
            "nomes_esperados": len(nomes),
        }

        # b. cada planta é recusada, e pelo motivo dela. A segunda troca um texto do vocabulário
        # fixo por outro do MESMO tamanho, para que a serialização continue legível e a recusa
        # seja da G2 e não do leitor.
        com_amostra = ext.monta_carga([*nomes, AMOSTRA_PLANTADA])
        trocada = carga.replace(b"msetora", b"msetorX")
        plantas = (
            ("nome de amostra entre os nomes", com_amostra, AMOSTRA_PLANTADA),
            ("vocabulario fixo trocado nos bytes ja montados", trocada, "vocabulário fixo"),
        )
        registros = []
        for rotulo, plantada, trecho in plantas:
            passou, recusa = g2(plantada, nomes)
            registros.append(
                {
                    "planta": rotulo,
                    "aterrou": plantada != carga,
                    "recusada": not passou,
                    "motivo": recusa,
                    "pelo_motivo_certo": trecho in recusa,
                }
            )
        resultado["g2_plantas"] = registros

        # c. a resposta sintética atravessa o envio pela conexão falsa, e cada p confere
        r, registro = envia_sem_rede(carga, nomes, resposta_sintetica(nomes))
        # As chaves existem em todo desfecho: o `.ps1` as lê sem ramo, e o teste confere o contrato.
        registro.update(
            ora_confere=False,
            motivo_da_ora="",
            vias=[],
            vias_conferem=False,
            universo=None,
            na_lista=None,
        )
        resultado["resposta"] = registro
        if r.desfecho == "ok":
            try:
                ora = en.ora_da_resposta(r.dados)
            except en.RespostaInvalidaError as erro:
                registro.update(ora_confere=False, motivo_da_ora=str(erro))
            else:
                lidas = sorted(v.via for v in ora.vias)
                registro.update(
                    ora_confere=True,
                    vias=lidas,
                    vias_conferem=lidas == sorted(v for v, _, _ in VIAS_INVENTADAS),
                    universo=ora.tamanho_do_universo,
                    na_lista=ora.na_lista,
                )

        # d. a resposta com função é recusada pelo leitor, e o motivo nomeia o tipo
        _, com_funcao = envia_sem_rede(carga, nomes, COM_FUNCAO)
        com_funcao["pelo_motivo_certo"] = "CLOSXP" in str(com_funcao["motivo"])
        resultado["com_funcao"] = com_funcao


def temporarios(pasta: Path) -> set[str]:
    """Os nomes que casam algum dos padrões, no primeiro nível de `pasta`."""
    return {p.name for padrao in PADROES_DE_TEMPORARIO for p in pasta.glob(padrao)}


def sobras(antes: set[str], depois: set[str]) -> list[str]:
    """O que nasceu entre as duas fotografias e continuou lá — a acusação do item 5."""
    return sorted(depois - antes)


def prova(pasta: Path, resultado: dict[str, object]) -> None:
    """Preenche `resultado` item a item: o que já foi medido fica, se um passo seguinte levantar."""
    from refino.desktop.vias_r import resolve_motor
    from refino.io import dados_vias
    from refino.io.enriquecimento_r import corre_roteiro_de_vias
    from refino.pathways import enriquecimento as en

    motor = resolve_motor(executavel=str(pasta / "refino.exe"))
    resultado["r_home"] = str(motor.r_home)
    resultado["r_home_dentro"] = motor.r_home.resolve().is_relative_to(pasta.resolve())
    conferencia = dados_vias.confere_pasta(motor.bibliotecas_fixadas, dados_vias.le_manifesto())
    resultado["bibliotecas"] = conferencia.estado

    # 4. o diretório de DLL que o bootloader do PyInstaller põe no processo do refino.exe
    resultado["set_dll_directory"] = bool(
        ctypes.windll.kernel32.SetDllDirectoryW(str(pasta / "_internal"))  # type: ignore[attr-defined]
    )

    lista = list(NOMES_DA_PROVA)
    t = time.monotonic()
    m = corre_roteiro_de_vias(
        {
            "action": "mapear",
            "dados_dir": str(motor.bibliotecas_fixadas),
            "lista": lista,
            "fundo": lista,
        },
        r_home=motor.r_home,
        roteiro=motor.roteiro,
    )
    resultado["mapear"] = {
        "desfecho": m.desfecho,
        "motivo": m.motivo,
        "segundos": round(time.monotonic() - t, 2),
    }
    if m.desfecho == "ok":
        mapa_lista, _ = en.mapeamento_da_resposta(m.dados)
        canon = [x.casado for x in mapa_lista if x.hmdb and x.casado]
        t = time.monotonic()
        o = corre_roteiro_de_vias(
            {"action": "ora_local", "dados_dir": str(motor.bibliotecas_fixadas), "lista": canon},
            r_home=motor.r_home,
            roteiro=motor.roteiro,
        )
        resultado["ora"] = {
            "desfecho": o.desfecho,
            "motivo": o.motivo,
            "segundos": round(time.monotonic() - t, 2),
        }
        if o.desfecho == "ok":
            ora = en.ora_da_resposta(o.dados)  # levanta se algum p não confere
            resultado["ora"]["vias"] = len(ora.vias)  # type: ignore[index]

    # As DLLs que o R carregou, pelo próprio R: `getLoadedDLLs()` dá o caminho de cada uma.
    sonda = Path(tempfile.mkdtemp()) / "dlls.R"
    sonda.write_text(
        "suppressPackageStartupMessages(library(MetaboAnalystR))\n"
        "d <- vapply(getLoadedDLLs(), function(x) x[['path']], '')\n"
        "cat(jsonlite::toJSON(list(status='ok', dlls=unname(d), "
        "ambiente=list(libPaths=normalizePath(.libPaths(), '/'))), auto_unbox=TRUE), '\\n')\n",
        encoding="utf-8",
    )
    d = corre_roteiro_de_vias({}, r_home=motor.r_home, roteiro=sonda)
    dlls = [str(x) for x in (d.dados or {}).get("dlls", [])]
    resultado["dlls"] = {
        "desfecho": d.desfecho,
        "n": len(dlls),
        "fora_do_R": [
            x
            for x in dlls
            if "/" in x and not Path(x).resolve().is_relative_to(motor.r_home.resolve())
        ],
    }

    # 5. cancelar: um pedido que o R leva segundos a terminar, cancelado logo depois de começar
    temp = Path(tempfile.gettempdir())
    antes = temporarios(temp)
    inicio = time.monotonic()
    c = corre_roteiro_de_vias(
        {"action": "ora_local", "dados_dir": str(motor.bibliotecas_fixadas), "lista": lista},
        r_home=motor.r_home,
        roteiro=motor.roteiro,
        cancelado=lambda: time.monotonic() - inicio > 1.0,
    )
    resultado["cancelar"] = {
        "desfecho": c.desfecho,
        "sobrou_temporario": sobras(antes, temporarios(temp)),
    }


def main(argv: list[str]) -> None:
    pasta, saida = Path(argv[1]), Path(argv[2])
    resultado: dict[str, object] = {}
    try:
        prova(pasta, resultado)
    except BaseException as erro:
        resultado["erro"] = f"{type(erro).__name__}: {erro}"
    # Depois do R e num `try` próprio: o item 6 não depende do R, e correr depois deixa os itens 1 a
    # 5 no processo em que foram medidos até aqui — sem a troca de `socket` de `sem_rede`, que um
    # defeito na restauração deixaria posta para o lançador.
    externo: dict[str, object] = {}
    resultado["externo"] = externo
    try:
        prova_externa(externo)
    except BaseException as erro:
        externo["erro"] = f"{type(erro).__name__}: {erro}"
    saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv)

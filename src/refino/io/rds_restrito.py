"""Leitor restrito do formato de serialização do R, sem `unserialize`.

O modo externo do enriquecimento conversa com o servidor do MetaboAnalyst por objetos R
serializados. Um objeto assim pode trazer função, ambiente, linguagem e referência a pacote, e
desserializá-lo com o R é rodar o que veio de fora. Este módulo lê o formato XDR à mão e aceita só
dado puro: vetor lógico, inteiro, real e de texto, lista, `NULL`, e os atributos deles (nomes,
`dim`, `dimnames`, classe). Qualquer outro tipo é recusado com o caminho em que apareceu, e não
pulado, porque pular exigiria percorrer a estrutura dele.

O mesmo leitor confere os bytes da carga que vai sair e lê a resposta. Há tetos de profundidade,
de itens, de tamanho de vetor e de texto, de valores decodificados e de tamanho descomprimido; um
conteúdo malformado ou além deles levanta `RdsRecusadoError`, e nunca devolve objeto parcial. O
formato está em `src/main/serialize.c` do R; os códigos de tipo abaixo são os de `Rinternals.h`.
"""

from __future__ import annotations

import gzip
import io
import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Final, Literal

__all__ = [
    "Lista",
    "RdsRecusadoError",
    "Simbolo",
    "Vetor",
    "escreve_rds",
    "inventario",
    "le_rds",
]

TipoDoVetor = Literal["logico", "inteiro", "real", "texto"]

_NILSXP: Final = 0
_SYMSXP: Final = 1
_LISTSXP: Final = 2
_CHARSXP: Final = 9
_LGLSXP: Final = 10
_INTSXP: Final = 13
_REALSXP: Final = 14
_STRSXP: Final = 16
_VECSXP: Final = 19
_ALTREP_SXP: Final = 238
_REFSXP: Final = 255
_NILVALUE_SXP: Final = 254

# Os que se reconhecem só para dizer o que foi recusado; nenhum é lido.
_NOMES_RECUSADOS: Final = {
    3: "função (CLOSXP)",
    4: "ambiente (ENVSXP)",
    5: "promessa (PROMSXP)",
    6: "linguagem (LANGSXP)",
    7: "função interna (SPECIALSXP)",
    8: "função interna (BUILTINSXP)",
    17: "dots (DOTSXP)",
    20: "expressão (EXPRSXP)",
    21: "código compilado (BCODESXP)",
    22: "ponteiro externo (EXTPTRSXP)",
    23: "referência fraca (WEAKREFSXP)",
    24: "bytes (RAWSXP)",
    25: "objeto S4 (S4SXP)",
    240: "lista de atributos de linguagem (ATTRLISTSXP)",
    241: "linguagem com atributos (ATTRLANGSXP)",
    242: "ambiente base (BASEENV_SXP)",
    243: "ambiente vazio (EMPTYENV_SXP)",
    244: "código compilado (BCREPDEF)",
    245: "referência genérica (GENERICREFSXP)",
    246: "referência de classe (CLASSREFSXP)",
    247: "objeto persistente (PERSISTSXP)",
    248: "pacote (PACKAGESXP)",
    249: "namespace (NAMESPACESXP)",
    250: "namespace base (BASENAMESPACE_SXP)",
    251: "argumento ausente (MISSINGARG_SXP)",
    252: "valor não ligado (UNBOUNDVALUE_SXP)",
    253: "ambiente global (GLOBALENV_SXP)",
}

_NA_INTEIRO: Final = -(2**31)
_UTF8: Final = 1 << 3
_LATIN1: Final = 1 << 2
_BYTES: Final = 1 << 1

# Tetos. Uma resposta do servidor tem cerca de 14 kB; os tetos ficam três ordens acima, e o que
# os passa é recusado em vez de lido devagar.
MAX_BYTES: Final = 20_000_000
MAX_PROFUNDIDADE: Final = 64
MAX_ITENS: Final = 2_000_000
MAX_COMPRIMENTO: Final = 10_000_000
MAX_TEXTO: Final = 1_000_000
# O total de valores decodificados, somado sobre todos os vetores. Os tetos acima contam itens e
# bytes, e uma sequência compacta (`compact_intseq`) é um item de poucos bytes que vira até
# `MAX_COMPRIMENTO` valores. O teto é o que `MAX_BYTES` deixaria escrever por extenso no tipo mais
# estreito (4 bytes por valor), então a forma compacta não rende mais que a extensa.
MAX_VALORES: Final = MAX_BYTES // 4
_INT32_MIN: Final = -(2**31) + 1  # o menor inteiro do R; -2^31 é o NA
_INT32_MAX: Final = 2**31 - 1

# As classes ALTREP da base cujo estado é dado puro e cuja expansão se faz aqui sem o R.
_ALTREP_COMPACTOS: Final = frozenset({"compact_intseq", "compact_realseq"})
_ALTREP_EMBRULHOS: Final = frozenset(
    {"wrap_integer", "wrap_real", "wrap_logical", "wrap_string", "wrap_list"}
)


class RdsRecusadoError(ValueError):
    """O conteúdo tem um tipo que este leitor não aceita, está malformado, ou passa de um teto."""


@dataclass(frozen=True)
class Simbolo:
    nome: str


@dataclass(frozen=True)
class Vetor:
    tipo: TipoDoVetor
    valores: tuple[bool | int | float | str | None, ...]
    atributos: dict[str, Objeto] = field(default_factory=dict)


@dataclass(frozen=True)
class Lista:
    itens: tuple[Objeto, ...]
    atributos: dict[str, Objeto] = field(default_factory=dict)

    def nomes(self) -> tuple[str | None, ...]:
        n = self.atributos.get("names")
        if isinstance(n, Vetor) and n.tipo == "texto" and len(n.valores) == len(self.itens):
            return tuple(v if isinstance(v, str) else None for v in n.valores)
        return tuple(None for _ in self.itens)

    def campo(self, nome: str) -> Objeto:
        """O item de nome `nome`; levanta se ele não existe ou se o nome se repete."""
        achados = [i for i, n in enumerate(self.nomes()) if n == nome]
        if len(achados) != 1:
            raise RdsRecusadoError(f"campo «{nome}»: {len(achados)} ocorrências, esperava 1")
        return self.itens[achados[0]]


Objeto = Vetor | Lista | Simbolo | None


class _Leitor:
    def __init__(self, dados: bytes) -> None:
        self._f = io.BytesIO(dados)
        self._tamanho = len(dados)
        self._referencias: list[Simbolo] = []
        self._itens = 0
        self._valores = 0

    # ------------------------------------------------------------------ primitivas
    def _bytes(self, n: int) -> bytes:
        b = self._f.read(n)
        if len(b) != n:
            raise RdsRecusadoError(f"o conteúdo acabou: pedidos {n} bytes, havia {len(b)}")
        return b

    def _int(self) -> int:
        return int(struct.unpack(">i", self._bytes(4))[0])

    def _comprimento(self, caminho: str) -> int:
        n = self._int()
        if n == -1:
            alto, baixo = self._int(), self._int()
            n = (alto << 32) + baixo
        if n < 0 or n > MAX_COMPRIMENTO:
            raise RdsRecusadoError(f"{caminho}: comprimento {n} fora do teto de {MAX_COMPRIMENTO}")
        return n

    def _conta_valores(self, n: int, caminho: str) -> None:
        """Cobra `n` valores do orçamento antes de construí-los (ver `MAX_VALORES`)."""
        self._valores += n
        if self._valores > MAX_VALORES:
            raise RdsRecusadoError(f"{caminho}: mais de {MAX_VALORES} valores no conteúdo")

    def fim(self) -> None:
        sobra = self._tamanho - self._f.tell()
        if sobra:
            raise RdsRecusadoError(f"{sobra} bytes depois do objeto: o conteúdo não é um objeto só")

    # ------------------------------------------------------------------ cabeçalho
    def cabecalho(self) -> None:
        if self._bytes(2) != b"X\n":
            raise RdsRecusadoError("não é serialização XDR do R (cabeçalho diferente de «X\\n»)")
        versao = self._int()
        self._int()  # a versão do R que escreveu: informativa
        self._int()  # a versão mínima que lê: informativa
        if versao == 3:
            n = self._int()
            if not 0 < n <= 64:
                raise RdsRecusadoError(f"nome de codificação com {n} bytes")
            self._bytes(n)
        elif versao != 2:
            raise RdsRecusadoError(f"versão de serialização {versao} não aceita (só 2 e 3)")

    # ------------------------------------------------------------------ itens
    def item(self, caminho: str, profundidade: int) -> Objeto:
        if profundidade > MAX_PROFUNDIDADE:
            raise RdsRecusadoError(f"{caminho}: profundidade acima de {MAX_PROFUNDIDADE}")
        self._itens += 1
        if self._itens > MAX_ITENS:
            raise RdsRecusadoError(f"mais de {MAX_ITENS} itens")
        flags = self._int()
        tipo = flags & 0xFF
        tem_atributo = bool(flags & (1 << 9))
        if tipo in (_NILVALUE_SXP, _NILSXP):
            return None
        if tipo == _REFSXP:
            indice = flags >> 8 or self._int()
            if not 0 < indice <= len(self._referencias):
                raise RdsRecusadoError(f"{caminho}: referência {indice} a algo que não é símbolo")
            return self._referencias[indice - 1]
        if tipo == _SYMSXP:
            nome = self._charsxp(caminho)
            if nome is None:
                raise RdsRecusadoError(f"{caminho}: símbolo sem nome")
            s = Simbolo(nome)
            self._referencias.append(s)
            return s
        if tipo == _LISTSXP:
            return self._pairlist(flags, caminho, profundidade)
        if tipo in (_LGLSXP, _INTSXP, _REALSXP, _STRSXP):
            v = self._vetor(tipo, caminho)
            atributos = self._atributos(caminho, profundidade) if tem_atributo else {}
            return Vetor(v[0], v[1], atributos)
        if tipo == _VECSXP:
            n = self._comprimento(caminho)
            self._conta_valores(n, caminho)
            itens = tuple(self.item(f"{caminho}[[{i + 1}]]", profundidade + 1) for i in range(n))
            atributos = self._atributos(caminho, profundidade) if tem_atributo else {}
            return Lista(itens, atributos)
        if tipo == _ALTREP_SXP:
            return self._altrep(caminho, profundidade)
        nome = _NOMES_RECUSADOS.get(tipo, f"tipo {tipo}")
        raise RdsRecusadoError(f"{caminho}: {nome} recusado — o leitor aceita só dado puro")

    def _charsxp(self, caminho: str) -> str | None:
        flags = self._int()
        if flags & 0xFF != _CHARSXP:
            raise RdsRecusadoError(f"{caminho}: esperava texto (CHARSXP), veio tipo {flags & 0xFF}")
        n = self._int()
        if n == -1:
            return None
        if n < 0 or n > MAX_TEXTO:
            raise RdsRecusadoError(f"{caminho}: texto de {n} bytes fora do teto de {MAX_TEXTO}")
        bruto = self._bytes(n)
        niveis = flags >> 12
        if niveis & _LATIN1:
            return bruto.decode("latin-1")
        if niveis & _BYTES:
            raise RdsRecusadoError(f"{caminho}: texto marcado como bytes, sem codificação")
        try:
            return bruto.decode("utf-8")
        except UnicodeDecodeError as erro:
            raise RdsRecusadoError(f"{caminho}: texto que não é UTF-8 ({erro})") from erro

    def _vetor(
        self, tipo: int, caminho: str
    ) -> tuple[TipoDoVetor, tuple[bool | int | float | str | None, ...]]:
        n = self._comprimento(caminho)
        self._conta_valores(n, caminho)
        if tipo == _STRSXP:
            return "texto", tuple(self._charsxp(f"{caminho}[{i + 1}]") for i in range(n))
        if tipo == _REALSXP:
            return "real", tuple(float(x) for x in struct.unpack(f">{n}d", self._bytes(8 * n)))
        brutos = struct.unpack(f">{n}i", self._bytes(4 * n))
        if tipo == _INTSXP:
            return "inteiro", tuple(None if x == _NA_INTEIRO else int(x) for x in brutos)
        logicos: list[bool | None] = []
        for x in brutos:
            if x == _NA_INTEIRO:
                logicos.append(None)
            elif x in (0, 1):
                logicos.append(bool(x))
            else:
                raise RdsRecusadoError(f"{caminho}: lógico com valor {x}")
        return "logico", tuple(logicos)

    def _pairlist(self, flags: int, caminho: str, profundidade: int) -> Lista:
        """Uma pairlist vira `Lista` com nomes. Ela aparece como lista de atributos."""
        itens: list[Objeto] = []
        nomes: list[str | None] = []
        atual = flags
        while True:
            if atual & (1 << 9):
                self.item(f"{caminho}@atributos", profundidade + 1)
            nome: str | None = None
            if atual & (1 << 10):
                rotulo = self.item(f"{caminho}@rótulo", profundidade + 1)
                if not isinstance(rotulo, Simbolo):
                    raise RdsRecusadoError(f"{caminho}: rótulo de pairlist que não é símbolo")
                nome = rotulo.nome
            itens.append(self.item(f"{caminho}${nome or len(itens) + 1}", profundidade + 1))
            nomes.append(nome)
            proximo = self._int()
            tipo = proximo & 0xFF
            if tipo == _NILVALUE_SXP:
                break
            if tipo != _LISTSXP:
                raise RdsRecusadoError(f"{caminho}: pairlist que não termina em NULL (tipo {tipo})")
            self._itens += 1
            atual = proximo
        rotulos = Vetor("texto", tuple(nomes)) if any(n is not None for n in nomes) else None
        return Lista(tuple(itens), {"names": rotulos} if rotulos else {})

    def _atributos(self, caminho: str, profundidade: int) -> dict[str, Objeto]:
        lista = self.item(f"{caminho}@atributos", profundidade + 1)
        if not isinstance(lista, Lista):
            raise RdsRecusadoError(f"{caminho}: atributos que não são pairlist")
        saida: dict[str, Objeto] = {}
        for nome, valor in zip(lista.nomes(), lista.itens, strict=True):
            if nome is None:
                raise RdsRecusadoError(f"{caminho}: atributo sem nome")
            saida[nome] = valor
        return saida

    def _altrep(self, caminho: str, profundidade: int) -> Objeto:
        info = self.item(f"{caminho}@altrep", profundidade + 1)
        if not isinstance(info, Lista) or len(info.itens) < 2:
            raise RdsRecusadoError(f"{caminho}: ALTREP sem a descrição da classe")
        classe, pacote = info.itens[0], info.itens[1]
        if not (isinstance(classe, Simbolo) and isinstance(pacote, Simbolo)):
            raise RdsRecusadoError(f"{caminho}: ALTREP com classe que não é símbolo")
        if pacote.nome != "base" or classe.nome not in _ALTREP_COMPACTOS | _ALTREP_EMBRULHOS:
            raise RdsRecusadoError(
                f"{caminho}: ALTREP {pacote.nome}::{classe.nome} recusado — só as classes "
                "compactas e os embrulhos da base"
            )
        if classe.nome in _ALTREP_COMPACTOS:
            estado = self.item(f"{caminho}@estado", profundidade + 1)
        else:
            estado = self._estado_do_embrulho(f"{caminho}@estado", profundidade + 1)
        atributos_obj = self.item(f"{caminho}@atributos", profundidade + 1)
        atributos: dict[str, Objeto] = {}
        if isinstance(atributos_obj, Lista):
            for nome, valor in zip(atributos_obj.nomes(), atributos_obj.itens, strict=True):
                if nome is not None:
                    atributos[nome] = valor
        if classe.nome in _ALTREP_COMPACTOS:
            return self._compacto(classe.nome, estado, atributos, caminho)
        if isinstance(estado, Vetor):
            return Vetor(estado.tipo, estado.valores, {**estado.atributos, **atributos})
        if isinstance(estado, Lista):
            return Lista(estado.itens, {**estado.atributos, **atributos})
        raise RdsRecusadoError(f"{caminho}: embrulho ALTREP de algo que não é vetor")

    def _estado_do_embrulho(self, caminho: str, profundidade: int) -> Objeto:
        """Lê o estado de um `wrap_*` da base, que é um par pontilhado. Devolve o vetor embrulhado.

        `wrapper_Serialized_state` em `src/main/altclasses.c` devolve `CONS(x, meta)`: a cabeça é o
        vetor embrulhado e a cauda é o vetor inteiro de metadados (`srt`, `no_na`), sem o `NULL`
        que fecha uma pairlist. Lido como pairlist, todo embrulho que o R grava seria recusado.
        """
        flags = self._int()
        if flags != _LISTSXP:  # sem atributo, sem rótulo, sem bit de objeto: só o par
            raise RdsRecusadoError(f"{caminho}: embrulho ALTREP sem o par (vetor, metadados)")
        self._itens += 1
        dentro = self.item(f"{caminho}$vetor", profundidade + 1)
        meta = self.item(f"{caminho}$metadados", profundidade + 1)
        if not (isinstance(meta, Vetor) and meta.tipo == "inteiro" and len(meta.valores) == 2):
            raise RdsRecusadoError(f"{caminho}: metadados do embrulho fora da forma")
        if not isinstance(dentro, Vetor | Lista):
            raise RdsRecusadoError(f"{caminho}: embrulho ALTREP de algo que não é vetor")
        return dentro

    def _compacto(
        self, classe: str, estado: Objeto, atributos: dict[str, Objeto], caminho: str
    ) -> Vetor:
        if not (isinstance(estado, Vetor) and estado.tipo == "real" and len(estado.valores) == 3):
            raise RdsRecusadoError(f"{caminho}: sequência compacta com estado fora da forma")
        n_bruto, inicio, passo = (float(x) for x in estado.valores if isinstance(x, float))
        if not (
            n_bruto.is_integer()
            and 0 <= n_bruto <= MAX_COMPRIMENTO
            and passo in (1.0, -1.0)
            and math.isfinite(inicio)
        ):
            raise RdsRecusadoError(
                f"{caminho}: sequência compacta fora da forma ({estado.valores})"
            )
        n = int(n_bruto)
        self._conta_valores(n, caminho)
        if classe == "compact_intseq":
            fim = inicio + passo * (n - 1) if n else inicio
            if not inicio.is_integer():
                raise RdsRecusadoError(f"{caminho}: sequência inteira com início {inicio}")
            if min(inicio, fim) < _INT32_MIN or max(inicio, fim) > _INT32_MAX:
                raise RdsRecusadoError(f"{caminho}: sequência inteira além do inteiro do R")
            return Vetor(
                "inteiro", tuple(int(inicio) + int(passo) * i for i in range(n)), atributos
            )
        return Vetor("real", tuple(inicio + passo * i for i in range(n)), atributos)


def le_rds(dados: bytes) -> Objeto:
    """Lê o objeto em `dados`, serialização XDR do R crua ou em `gzip`, só se for dado puro."""
    if len(dados) > MAX_BYTES:
        raise RdsRecusadoError(f"{len(dados)} bytes acima do teto de {MAX_BYTES}")
    if dados[:2] == b"\x1f\x8b":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(dados)) as g:
                dados = g.read(MAX_BYTES + 1)
        # `zlib.error` é o fluxo `deflate` corrompido, que não é `OSError`.
        except (OSError, EOFError, zlib.error) as erro:
            raise RdsRecusadoError(f"gzip ilegível: {erro}") from erro
        if len(dados) > MAX_BYTES:
            raise RdsRecusadoError(f"descomprimido passa do teto de {MAX_BYTES} bytes")
    leitor = _Leitor(dados)
    leitor.cabecalho()
    objeto = leitor.item("raiz", 0)
    leitor.fim()
    _recusa_simbolo_solto(objeto, "raiz")
    return objeto


def _recusa_simbolo_solto(objeto: Objeto, caminho: str) -> None:
    """Recusa símbolo que aparece como valor.

    Símbolo só é aceito como nome (rótulo de pairlist ou classe ALTREP), e esses já viraram texto.
    Um símbolo que sobra como valor é o nome de algo no R, e não dado; a resposta do servidor não
    traz nenhum.
    """
    if isinstance(objeto, Simbolo):
        raise RdsRecusadoError(f"{caminho}: símbolo como valor recusado — o leitor aceita só dado")
    if isinstance(objeto, Lista):
        for i, (nome, item) in enumerate(zip(objeto.nomes(), objeto.itens, strict=True)):
            _recusa_simbolo_solto(item, f"{caminho}${nome if nome else f'[[{i + 1}]]'}")
    if isinstance(objeto, (Lista, Vetor)):
        for nome_do_atributo, valor in objeto.atributos.items():
            _recusa_simbolo_solto(valor, f"{caminho}@{nome_do_atributo}")


_VERSAO_DO_R: Final = (4 << 16) | (4 << 8) | 3  # quem escreve se declara 4.4.3, a árvore embarcada
_VERSAO_MINIMA: Final = (2 << 16) | (3 << 8) | 0  # o formato 2 é lido desde o R 2.3.0
_ASCII: Final = 1 << 6


def escreve_rds(objeto: Objeto) -> bytes:
    """Devolve a serialização XDR (formato 2) de `objeto`, sem compressão.

    Só escreve o que `le_rds` aceita. A carga do modo externo é escrita por aqui, por esquema
    fechado e a partir de valores já conferidos, e não pelo objeto que o pacote montaria.
    """
    saida = io.BytesIO()
    saida.write(b"X\n")
    saida.write(struct.pack(">iii", 2, _VERSAO_DO_R, _VERSAO_MINIMA))
    _escreve_item(saida, objeto)
    return saida.getvalue()


def _escreve_item(f: io.BytesIO, objeto: Objeto) -> None:
    if objeto is None:
        f.write(struct.pack(">i", _NILVALUE_SXP))
        return
    if isinstance(objeto, Simbolo):
        raise RdsRecusadoError("símbolo solto não é dado: o escritor só os usa como nome")
    tem_atributo = bool(objeto.atributos)
    # O bit de objeto acompanha o atributo `class`: sem ele o R lê a classe e não a trata como
    # objeto, e `identical` com o original dá FALSE.
    e_objeto = "class" in objeto.atributos
    if isinstance(objeto, Lista):
        f.write(
            struct.pack(">ii", _VECSXP | (e_objeto << 8) | (tem_atributo << 9), len(objeto.itens))
        )
        for item in objeto.itens:
            _escreve_item(f, item)
    else:
        tipo = {"logico": _LGLSXP, "inteiro": _INTSXP, "real": _REALSXP, "texto": _STRSXP}[
            objeto.tipo
        ]
        f.write(
            struct.pack(">ii", tipo | (e_objeto << 8) | (tem_atributo << 9), len(objeto.valores))
        )
        if objeto.tipo == "texto":
            for v in objeto.valores:
                _escreve_texto(f, v)
        elif objeto.tipo == "real":
            f.write(struct.pack(f">{len(objeto.valores)}d", *(float(v) for v in objeto.valores)))  # type: ignore[arg-type]
        else:
            f.write(
                struct.pack(
                    f">{len(objeto.valores)}i",
                    *(_NA_INTEIRO if v is None else int(v) for v in objeto.valores),
                )
            )
    if tem_atributo:
        for nome, valor in objeto.atributos.items():
            f.write(struct.pack(">i", _LISTSXP | (1 << 10)))
            f.write(struct.pack(">i", _SYMSXP))
            _escreve_texto(f, nome)
            _escreve_item(f, valor)
        f.write(struct.pack(">i", _NILVALUE_SXP))


def _escreve_texto(f: io.BytesIO, texto: object) -> None:
    if texto is None:
        f.write(struct.pack(">ii", _CHARSXP, -1))
        return
    if not isinstance(texto, str):
        raise RdsRecusadoError(f"vetor de texto com {type(texto).__name__}")
    bruto = texto.encode("utf-8")
    nivel = _ASCII if texto.isascii() else _UTF8
    f.write(struct.pack(">ii", _CHARSXP | (nivel << 12), len(bruto)))
    f.write(bruto)


def inventario(objeto: Objeto, caminho: str = "raiz") -> list[tuple[str, str, int]]:
    """Lista cada folha com caminho, tipo e tamanho, para a guarda do envio e para os testes."""
    if objeto is None:
        return [(caminho, "nulo", 0)]
    if isinstance(objeto, Simbolo):
        return [(caminho, "símbolo", 1)]
    if isinstance(objeto, Vetor):
        return [(caminho, objeto.tipo, len(objeto.valores))]
    saida: list[tuple[str, str, int]] = []
    if not objeto.itens:
        saida.append((caminho, "lista", 0))
    for i, (nome, item) in enumerate(zip(objeto.nomes(), objeto.itens, strict=True)):
        saida.extend(inventario(item, f"{caminho}${nome if nome else f'[[{i + 1}]]'}"))
    return saida

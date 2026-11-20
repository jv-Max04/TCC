"""As bibliotecas de dados do enriquecimento de vias: confere a pasta e baixa com confirmação.

O MetaboAnalystR lê, no modo local, arquivos `.qs` que o servidor do MetaboAnalyst serve sem
versão e troca sem aviso. Na pasta do laboratório as cópias viajam fixadas, e `confere_pasta` só
aprova arquivo com o tamanho e o sha256 do manifesto. A release pública sai sem elas:
`baixa_bibliotecas` só baixa com confirmação, registra hash e data ao lado dos arquivos, e a
cópia cujo hash difere do manifesto fica marcada como não validada. O manifesto fica em
`src/refino/pathways/dados_vias.json`, dentro do pacote, para resolver igual no clone e no
executável congelado.

O arquivo baixado vai direto para o desserializador do R, e um objeto R serializado pode carregar
código. Sem hash fixado a conferir antes, o canal é a única barreira: só `https` e só o host do
manifesto, certificado verificado, nenhum redirecionamento para outro esquema ou host, teto de
bytes conferido durante a leitura, total lido igual ao `Content-Length`, e escrita atômica por
temporários na própria pasta. Aceitar o download é aceitar conteúdo de terceiro que o R vai
carregar, e a tela diz isso na confirmação.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import ssl
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, Literal, Protocol

from refino import pathways

__all__ = [
    "MANIFESTO_EMBALADO",
    "REGISTRO_DE_DOWNLOAD",
    "Biblioteca",
    "BibliotecaBaixada",
    "ConferenciaDaPasta",
    "ManifestoInvalidoError",
    "ResultadoDoDownload",
    "baixa_bibliotecas",
    "confere_baixadas",
    "confere_pasta",
    "le_data_do_manifesto",
    "le_manifesto",
]

MANIFESTO_EMBALADO: Final[Path] = Path(pathways.__file__).resolve().parent / "dados_vias.json"
REGISTRO_DE_DOWNLOAD: Final = "baixadas.json"
# O teto de bytes de cada arquivo é este múltiplo do tamanho do manifesto. Dá folga para o
# servidor ter trocado o arquivo por um maior e barra uma resposta que não é a biblioteca.
FATOR_DO_TETO: Final = 4
TEMPO_LIMITE_S: Final = 120.0
_BLOCO: Final = 1 << 16
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_ARQUIVO: Final = re.compile(r"^[A-Za-z0-9_.-]+\.qs$")
_INTEIRO: Final = re.compile(r"[0-9]+")
_DATA: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

EstadoDaPasta = Literal["ok", "ausente", "divergente"]
DesfechoDoDownload = Literal["baixado", "recusado", "sem_rede", "cancelado", "transporte_recusado"]


class ManifestoInvalidoError(ValueError):
    """O manifesto empacotado não tem a forma que esta fronteira confia."""


class TransporteRecusadoError(Exception):
    """Um controle de transporte recusou a conexão: esquema, host, redirecionamento ou tamanho."""


@dataclass(frozen=True)
class Biblioteca:
    arquivo: str
    caminho_remoto: str
    tamanho: int
    sha256: str
    url: str


@dataclass(frozen=True)
class ConferenciaDaPasta:
    estado: EstadoDaPasta
    ausentes: tuple[str, ...]
    divergentes: tuple[str, ...]


@dataclass(frozen=True)
class BibliotecaBaixada:
    arquivo: str
    sha256: str
    tamanho: int
    baixada_em: str
    validada: bool


@dataclass(frozen=True)
class ResultadoDoDownload:
    desfecho: DesfechoDoDownload
    motivo: str
    baixadas: tuple[BibliotecaBaixada, ...]


class _Cabecalhos(Protocol):
    def get(self, nome: str, /) -> str | None: ...


class Resposta(Protocol):
    @property
    def headers(self) -> _Cabecalhos: ...

    def read(self, n: int = -1, /) -> bytes: ...

    def close(self) -> None: ...


Abridor = Callable[[str, float], Resposta]


def le_manifesto(caminho: Path | None = None) -> tuple[Biblioteca, ...]:
    """Lê e valida o manifesto, que é conferido campo a campo mesmo sendo empacotado."""
    alvo = MANIFESTO_EMBALADO if caminho is None else caminho
    try:
        dados = json.loads(alvo.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erro:
        raise ManifestoInvalidoError(
            f"o manifesto das bibliotecas em {alvo} não pôde ser lido: {erro}"
        ) from erro
    origem = dados.get("origem") if isinstance(dados, dict) else None
    if not isinstance(origem, str) or urllib.parse.urlparse(origem).scheme != "https":
        raise ManifestoInvalidoError(f"a origem do manifesto não é https: {origem!r}")
    entradas = dados.get("bibliotecas")
    if not isinstance(entradas, list) or not entradas:
        raise ManifestoInvalidoError("o manifesto não lista bibliotecas")
    saida: list[Biblioteca] = []
    for e in entradas:
        if not isinstance(e, dict):
            raise ManifestoInvalidoError(f"entrada do manifesto não é objeto: {e!r}")
        arquivo, remoto, tamanho, sha = (
            e.get("arquivo"),
            e.get("caminho_remoto"),
            e.get("tamanho"),
            e.get("sha256"),
        )
        if not isinstance(arquivo, str) or not _ARQUIVO.match(arquivo):
            raise ManifestoInvalidoError(f"nome de arquivo inválido no manifesto: {arquivo!r}")
        if not isinstance(remoto, str) or remoto.startswith("/") or ".." in remoto.split("/"):
            raise ManifestoInvalidoError(f"caminho remoto inválido no manifesto: {remoto!r}")
        if not isinstance(tamanho, int) or isinstance(tamanho, bool) or tamanho <= 0:
            raise ManifestoInvalidoError(f"tamanho inválido para {arquivo}: {tamanho!r}")
        if not isinstance(sha, str) or not _SHA256.match(sha):
            raise ManifestoInvalidoError(f"sha256 inválido para {arquivo}: {sha!r}")
        saida.append(
            Biblioteca(arquivo, remoto, tamanho, sha, urllib.parse.urljoin(origem, remoto))
        )
    return tuple(saida)


def le_data_do_manifesto(caminho: Path | None = None) -> str:
    """A data em que os hashes do manifesto foram medidos, `AAAA-MM-DD`.

    Serve para dizer, junto do resultado, de que versão das bibliotecas fixadas ele veio. Uma
    data fora da forma levanta, em vez de virar ausência de data numa exportação que existe
    para dizer a procedência.
    """
    alvo = MANIFESTO_EMBALADO if caminho is None else caminho
    try:
        dados = json.loads(alvo.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erro:
        raise ManifestoInvalidoError(
            f"o manifesto das bibliotecas em {alvo} não pôde ser lido: {erro}"
        ) from erro
    data = dados.get("medido_em") if isinstance(dados, dict) else None
    if not isinstance(data, str) or not _DATA.fullmatch(data):
        raise ManifestoInvalidoError(f"a data do manifesto não é AAAA-MM-DD: {data!r}")
    try:
        date.fromisoformat(data)
    except ValueError as erro:
        raise ManifestoInvalidoError(f"a data do manifesto não existe: {data!r}") from erro
    return data


def _sha256(caminho: Path) -> str:
    with caminho.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def confere_pasta(pasta: Path, bibliotecas: Sequence[Biblioteca]) -> ConferenciaDaPasta:
    """Confere as cópias fixadas da pasta do laboratório contra o tamanho e o hash do manifesto.

    Divergente vence ausente, porque pede outra ação: arquivo ausente pede instalação; arquivo
    presente e diferente diz que a pasta foi alterada, e esse sinal não pode ficar escondido
    atrás de um ausente.
    """
    if not bibliotecas:
        raise ManifestoInvalidoError("conferência sem bibliotecas: passaria por vacuidade")
    ausentes: list[str] = []
    divergentes: list[str] = []
    for b in bibliotecas:
        alvo = pasta / b.arquivo
        if not alvo.is_file():
            ausentes.append(b.arquivo)
        elif alvo.stat().st_size != b.tamanho or _sha256(alvo) != b.sha256:
            divergentes.append(b.arquivo)
    estado: EstadoDaPasta = "divergente" if divergentes else ("ausente" if ausentes else "ok")
    return ConferenciaDaPasta(estado, tuple(ausentes), tuple(divergentes))


class _RedirecionamentoRestrito(urllib.request.HTTPRedirectHandler):
    """Segue redirecionamento só para o mesmo esquema `https` e o mesmo host."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        antes = urllib.parse.urlparse(req.full_url)
        depois = urllib.parse.urlparse(urllib.parse.urljoin(req.full_url, newurl))
        if depois.scheme != "https" or depois.hostname != antes.hostname:
            raise TransporteRecusadoError(
                f"redirecionamento recusado de {antes.hostname} para {depois.scheme}://{depois.hostname}"
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


def _abre_com_seguranca(url: str, tempo_limite: float) -> Resposta:
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        _RedirecionamentoRestrito(),
    )
    resposta: Resposta = opener.open(url, timeout=tempo_limite)
    return resposta


def _tamanho_declarado(resposta: Resposta, arquivo: str) -> int | None:
    """O `Content-Length` da resposta, ou `None` quando o servidor não o mandou.

    É o que permite detectar corpo cortado: sem ele, um fim de conexão antes da hora é
    indistinguível de fim de arquivo. O cabeçalho vem da rede, então um valor que não é inteiro
    não negativo recusa o canal, em vez de passar por não declarado e desligar a conferência.
    """
    valor = resposta.headers.get("Content-Length")
    if valor is None:
        return None
    if not _INTEIRO.fullmatch(valor.strip()):
        raise TransporteRecusadoError(f"{arquivo}: Content-Length inválido: {valor!r}")
    return int(valor.strip())


def _confere_url(url: str, host: str) -> None:
    partes = urllib.parse.urlparse(url)
    if partes.scheme != "https" or partes.hostname != host:
        raise TransporteRecusadoError(f"URL fora do canal permitido: {url}")


def _apaga(caminhos: Sequence[Path]) -> None:
    for c in caminhos:
        c.unlink(missing_ok=True)


def baixa_bibliotecas(
    pasta: Path,
    bibliotecas: Sequence[Biblioteca],
    *,
    confirmado: bool,
    cancelado: Callable[[], bool] = lambda: False,
    abre: Abridor | None = None,
    agora: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ResultadoDoDownload:
    """Baixa as bibliotecas para `pasta`, tudo ou nada. Nunca levanta por rede: devolve o desfecho.

    Sem `confirmado`, não toca na pasta nem na rede. A confirmação é pedida pela janela, e esta
    função garante que ela não pode ser pulada.
    """
    if not confirmado:
        return ResultadoDoDownload("recusado", "o download não foi confirmado", ())
    if not bibliotecas:
        raise ManifestoInvalidoError("download sem bibliotecas")
    abridor = _abre_com_seguranca if abre is None else abre
    host = urllib.parse.urlparse(bibliotecas[0].url).hostname or ""
    pasta.mkdir(parents=True, exist_ok=True)
    temporarios: list[Path] = []
    prontos: list[tuple[Path, Path, BibliotecaBaixada]] = []
    try:
        for b in bibliotecas:
            _confere_url(b.url, host)
            teto = FATOR_DO_TETO * b.tamanho
            # A conexão abre antes do temporário: se ela falhar, não fica descritor aberto num
            # arquivo que o Windows então se recusaria a apagar.
            resposta = abridor(b.url, TEMPO_LIMITE_S)
            try:
                # Lido antes do temporário pelo mesmo motivo de descritor que a abertura acima.
                declarado = _tamanho_declarado(resposta, b.arquivo)
                descritor, nome = tempfile.mkstemp(
                    dir=pasta, prefix=f".{b.arquivo}.", suffix=".parcial"
                )
            except BaseException:
                resposta.close()
                raise
            tmp = Path(nome)
            temporarios.append(tmp)
            h = hashlib.sha256()
            total = 0
            try:
                with os.fdopen(descritor, "wb") as saida:
                    while True:
                        if cancelado():
                            return ResultadoDoDownload("cancelado", "o download foi cancelado", ())
                        bloco = resposta.read(_BLOCO)
                        if not bloco:
                            break
                        total += len(bloco)
                        if total > teto:
                            raise TransporteRecusadoError(
                                f"{b.arquivo}: a resposta passou do teto de {teto} bytes"
                            )
                        h.update(bloco)
                        saida.write(bloco)
            finally:
                resposta.close()
            if declarado is not None and total != declarado:
                # OSError e não recusa do canal: corpo cortado é conexão que caiu, e a ação na tela
                # é a de falta de rede, tentar de novo.
                raise OSError(
                    f"{b.arquivo}: a resposta trouxe {total} de {declarado} bytes declarados"
                )
            sha = h.hexdigest()
            baixada = BibliotecaBaixada(
                b.arquivo, sha, total, agora().isoformat(timespec="seconds"), sha == b.sha256
            )
            prontos.append((tmp, pasta / b.arquivo, baixada))
        for tmp, destino, _ in prontos:
            os.replace(tmp, destino)
        registro = [asdict(b) for _, _, b in prontos]
        _escreve_atomico(
            pasta / REGISTRO_DE_DOWNLOAD, json.dumps(registro, ensure_ascii=False, indent=1)
        )
        return ResultadoDoDownload("baixado", "", tuple(b for _, _, b in prontos))
    except TransporteRecusadoError as erro:
        return ResultadoDoDownload("transporte_recusado", str(erro), ())
    except urllib.error.HTTPError as erro:
        return ResultadoDoDownload(
            "transporte_recusado", f"o servidor respondeu {erro.code} para {erro.url}", ()
        )
    except urllib.error.URLError as erro:
        # Certificado que não verifica chega embrulhado em URLError, e não é falta de rede: é o
        # controle de transporte recusando, e a tela tem de dizer isso.
        if isinstance(erro.reason, ssl.SSLError):
            return ResultadoDoDownload(
                "transporte_recusado", f"o certificado não foi aceito: {erro.reason}", ()
            )
        return ResultadoDoDownload("sem_rede", f"o download falhou: {erro.reason}", ())
    except (OSError, TimeoutError, http.client.HTTPException) as erro:
        # `HTTPException` não é `OSError`: o corpo `chunked` cortado levanta `IncompleteRead`, e a
        # linha de estado ilegível levanta `BadStatusLine`. Sem ela, os dois escapariam desta
        # função, que não levanta por rede.
        return ResultadoDoDownload("sem_rede", f"o download falhou: {erro}", ())
    finally:
        _apaga(temporarios)


def _escreve_atomico(destino: Path, texto: str) -> None:
    descritor, nome = tempfile.mkstemp(
        dir=destino.parent, prefix=f".{destino.name}.", suffix=".parcial"
    )
    try:
        with os.fdopen(descritor, "w", encoding="utf-8") as f:
            f.write(texto)
        os.replace(nome, destino)
    except BaseException:
        _apaga([Path(nome)])
        raise


def confere_baixadas(
    pasta: Path, bibliotecas: Sequence[Biblioteca]
) -> tuple[BibliotecaBaixada, ...] | None:
    """A cópia baixada antes, se ela ainda é a que o registro descreve; senão `None`.

    `None` cobre os casos em que nunca se baixou, o registro sumiu ou um arquivo mudou depois.
    Os três pedem a mesma ação, oferecer o download de novo. Um campo do registro fora da forma
    também dá `None`.

    `validada` sai do manifesto de `bibliotecas`, e não do registro. O campo gravado é o veredito
    contra o manifesto da versão que baixou; a pasta sobrevive à atualização, e o manifesto pode
    ter mudado desde então.
    """
    try:
        dados = json.loads((pasta / REGISTRO_DE_DOWNLOAD).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(dados, list):
        return None
    registradas: dict[str, tuple[str, int, str]] = {}
    for e in dados:
        try:
            registradas[str(e["arquivo"])] = (
                str(e["sha256"]),
                int(e["tamanho"]),
                str(e["baixada_em"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    saida: list[BibliotecaBaixada] = []
    for bib in bibliotecas:
        registrada = registradas.get(bib.arquivo)
        alvo = pasta / bib.arquivo
        if registrada is None or not alvo.is_file():
            return None
        sha, tamanho, baixada_em = registrada
        if alvo.stat().st_size != tamanho or _sha256(alvo) != sha:
            return None
        saida.append(BibliotecaBaixada(bib.arquivo, sha, tamanho, baixada_em, sha == bib.sha256))
    return tuple(saida)

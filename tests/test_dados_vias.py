"""As bibliotecas de dados das vias: o manifesto, a conferência da pasta e o download.

**Os bytes aqui são sintéticos, e isso não fere a regra 1 do README.** O que se testa é o
MECANISMO — conferir hash, recusar canal, escrever tudo-ou-nada —, e ele é o mesmo para qualquer
conteúdo. A biblioteca real não entra num teste de unidade: ela é de terceiro, tem 12 MB, e o que
a valida é `scripts/busca_dados_vias.py` contra o manifesto, na montagem. O manifesto real,
esse, é lido aqui.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import ssl
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from refino.io import dados_vias
from refino.io.dados_vias import Biblioteca, TransporteRecusadoError, _RedirecionamentoRestrito

HOST = "https://www.metaboanalyst.ca/resources/libs/"


def _bib(nome: str, conteudo: bytes, *, url: str | None = None) -> Biblioteca:
    return Biblioteca(
        nome, nome, len(conteudo), hashlib.sha256(conteudo).hexdigest(), url or HOST + nome
    )


class _Resposta:
    def __init__(self, dados: bytes, *, quebra_depois: int | None = None) -> None:
        self._io = io.BytesIO(dados)
        self._quebra = quebra_depois
        self._lido = 0
        self.fechada = False
        # Declara o tamanho que de fato serve, como um servidor honesto: a quebra simulada pela
        # `quebra_depois` levanta, e não chega a ser corpo curto.
        self.headers = {"Content-Length": str(len(dados))}

    def read(self, n: int = -1, /) -> bytes:
        if self._quebra is not None and self._lido >= self._quebra:
            raise urllib.error.URLError(ConnectionResetError("a conexão caiu"))
        bloco = self._io.read(n if self._quebra is None else min(n, self._quebra - self._lido or 1))
        self._lido += len(bloco)
        return bloco

    def close(self) -> None:
        self.fechada = True


def _abridor(por_url: dict[str, bytes], **kw: int) -> Callable[[str, float], _Resposta]:
    chamadas: list[str] = []

    def abre(url: str, _t: float) -> _Resposta:
        chamadas.append(url)
        return _Resposta(por_url[url], **kw)

    abre.chamadas = chamadas  # type: ignore[attr-defined]
    return abre


def _relogio() -> datetime:
    return datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


class _Soquete:
    """Só `makefile`: o `http.client` REAL lê a resposta destes bytes, sem rede nenhuma.

    É o que dá dente aos testes de corpo cortado: o comportamento sob teste — fim de arquivo sem
    erro quando a conexão fecha antes do `Content-Length`, e `IncompleteRead` no `chunked` — é do
    `http.client`, e um duplo escrito à mão afirmaria só o que eu acho que ele faz.
    """

    def __init__(self, bruto: bytes) -> None:
        self._arquivo = io.BytesIO(bruto)

    def makefile(self, *_a: object, **_k: object) -> io.BytesIO:
        return self._arquivo


def _resposta_http(cabecalhos: str, corpo: bytes) -> http.client.HTTPResponse:
    bruto = b"HTTP/1.1 200 OK\r\n" + cabecalhos.encode("ascii") + b"\r\n" + corpo
    resposta = http.client.HTTPResponse(_Soquete(bruto))  # type: ignore[arg-type]
    resposta.begin()
    return resposta


def _serve(cabecalhos: str, corpo: bytes) -> Callable[[str, float], http.client.HTTPResponse]:
    return lambda _url, _t: _resposta_http(cabecalhos, corpo)


def test_o_manifesto_embalado_tem_as_cinco_bibliotecas_com_hash_e_tamanho() -> None:
    bibs = dados_vias.le_manifesto()
    assert len(bibs) == 5, (
        "âncora: o manifesto tem de listar as cinco antes de qualquer asserção de ausência"
    )
    assert {b.arquivo for b in bibs} == {
        "compound_db.qs",
        "syn_nms.qs",
        "master_compound_db.qs",
        "msets_smpdb_pathway.qs",
        "smpdb_hsa.qs",
    }
    for b in bibs:
        assert len(b.sha256) == 64 and b.tamanho > 0
        assert b.url.startswith(HOST), b.url


@pytest.mark.parametrize(
    ("campo", "valor", "trecho"),
    [
        ("sha256", "abc", "sha256 inválido"),
        ("tamanho", 0, "tamanho inválido"),
        ("arquivo", "../fora.qs", "nome de arquivo inválido"),
        ("caminho_remoto", "../../etc/passwd", "caminho remoto inválido"),
    ],
)
def test_o_manifesto_e_fronteira_de_confianca(
    tmp_path: Path, campo: str, valor: object, trecho: str
) -> None:
    dados = json.loads(dados_vias.MANIFESTO_EMBALADO.read_text(encoding="utf-8"))
    dados["bibliotecas"][0][campo] = valor
    alvo = tmp_path / "m.json"
    alvo.write_text(json.dumps(dados), encoding="utf-8")
    with pytest.raises(dados_vias.ManifestoInvalidoError, match=trecho):
        dados_vias.le_manifesto(alvo)


def test_a_data_do_manifesto_embalado_e_a_da_medicao() -> None:
    """A âncora dos casos abaixo: a data viaja para o CSV como a versão das bibliotecas (R13)."""
    assert dados_vias.le_data_do_manifesto() == "2026-10-01"


@pytest.mark.parametrize(
    ("valor", "trecho"),
    [
        pytest.param(None, "não é AAAA-MM-DD", id="ausente"),
        pytest.param("01/10/2026", "não é AAAA-MM-DD", id="outra_forma"),
        pytest.param(20261001, "não é AAAA-MM-DD", id="numero"),
        pytest.param("2026-02-30", "não existe", id="data_impossivel"),
    ],
)
def test_a_data_do_manifesto_e_fronteira_de_confianca(
    tmp_path: Path, valor: object, trecho: str
) -> None:
    dados = json.loads(dados_vias.MANIFESTO_EMBALADO.read_text(encoding="utf-8"))
    if valor is None:
        del dados["medido_em"]
    else:
        dados["medido_em"] = valor
    alvo = tmp_path / "m.json"
    alvo.write_text(json.dumps(dados), encoding="utf-8")
    with pytest.raises(dados_vias.ManifestoInvalidoError, match=trecho):
        dados_vias.le_data_do_manifesto(alvo)


def test_origem_fora_de_https_e_recusada(tmp_path: Path) -> None:
    dados = json.loads(dados_vias.MANIFESTO_EMBALADO.read_text(encoding="utf-8"))
    dados["origem"] = "http://www.metaboanalyst.ca/resources/libs/"
    alvo = tmp_path / "m.json"
    alvo.write_text(json.dumps(dados), encoding="utf-8")
    with pytest.raises(dados_vias.ManifestoInvalidoError, match="https"):
        dados_vias.le_manifesto(alvo)


def test_um_byte_trocado_da_divergente_e_a_original_passa(tmp_path: Path) -> None:
    a, b = b"conteudo-a" * 100, b"conteudo-b" * 50
    bibs = [_bib("a.qs", a), _bib("b.qs", b)]
    (tmp_path / "a.qs").write_bytes(a)
    (tmp_path / "b.qs").write_bytes(b)
    assert dados_vias.confere_pasta(tmp_path, bibs).estado == "ok"
    trocado = bytearray(a)
    trocado[7] ^= 0xFF
    (tmp_path / "a.qs").write_bytes(bytes(trocado))
    r = dados_vias.confere_pasta(tmp_path, bibs)
    assert (r.estado, r.divergentes, r.ausentes) == ("divergente", ("a.qs",), ())


def test_ausente_e_divergente_sao_desfechos_distintos_e_divergente_vence(tmp_path: Path) -> None:
    a, b = b"x" * 10, b"y" * 10
    bibs = [_bib("a.qs", a), _bib("b.qs", b)]
    (tmp_path / "a.qs").write_bytes(a)
    assert dados_vias.confere_pasta(tmp_path, bibs).estado == "ausente"
    (tmp_path / "a.qs").write_bytes(b"z" * 10)
    r = dados_vias.confere_pasta(tmp_path, bibs)
    assert (r.estado, r.ausentes, r.divergentes) == ("divergente", ("b.qs",), ("a.qs",))
    with pytest.raises(dados_vias.ManifestoInvalidoError):
        dados_vias.confere_pasta(tmp_path, [])


def test_sem_confirmacao_nada_e_baixado_nem_aberto(tmp_path: Path) -> None:
    bibs = [_bib("a.qs", b"abc")]
    abre = _abridor({bibs[0].url: b"abc"})
    pasta = tmp_path / "dados"
    r = dados_vias.baixa_bibliotecas(pasta, bibs, confirmado=False, abre=abre)
    assert r.desfecho == "recusado"
    assert abre.chamadas == []  # type: ignore[attr-defined]
    assert not pasta.exists()


def test_com_confirmacao_grava_hash_data_e_marca_validada(tmp_path: Path) -> None:
    a, b = b"abc" * 1000, b"def" * 10
    bibs = [_bib("a.qs", a), _bib("b.qs", b)]
    servido = {bibs[0].url: a, bibs[1].url: b"trocado-pelo-servidor"}
    r = dados_vias.baixa_bibliotecas(
        tmp_path, bibs, confirmado=True, abre=_abridor(servido), agora=_relogio
    )
    assert r.desfecho == "baixado", r.motivo
    por = {x.arquivo: x for x in r.baixadas}
    assert por["a.qs"].validada and not por["b.qs"].validada
    assert por["b.qs"].sha256 == hashlib.sha256(b"trocado-pelo-servidor").hexdigest()
    assert por["a.qs"].baixada_em == "2026-10-01T12:00:00+00:00"
    assert (tmp_path / "b.qs").read_bytes() == b"trocado-pelo-servidor"
    assert dados_vias.confere_baixadas(tmp_path, bibs) == r.baixadas
    assert not list(tmp_path.glob("*.parcial"))


def test_falha_de_rede_no_meio_deixa_a_pasta_como_estava(tmp_path: Path) -> None:
    a, b = b"a" * 5000, b"b" * 5000
    bibs = [_bib("a.qs", a), _bib("b.qs", b)]
    (tmp_path / "a.qs").write_bytes(b"versao-antiga")
    antes = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    r = dados_vias.baixa_bibliotecas(
        tmp_path,
        bibs,
        confirmado=True,
        abre=_abridor({bibs[0].url: a, bibs[1].url: b}, quebra_depois=3000),
    )
    assert r.desfecho == "sem_rede", r.motivo
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == antes
    # par causal: sem a quebra, o mesmo download troca a pasta
    r2 = dados_vias.baixa_bibliotecas(
        tmp_path, bibs, confirmado=True, abre=_abridor({bibs[0].url: a, bibs[1].url: b})
    )
    assert r2.desfecho == "baixado" and (tmp_path / "a.qs").read_bytes() == a


def test_cancelar_nao_deixa_arquivo_parcial(tmp_path: Path) -> None:
    a = b"a" * 200_000
    bibs = [_bib("a.qs", a)]
    chamadas = {"n": 0}

    def cancela_no_segundo_bloco() -> bool:
        chamadas["n"] += 1
        return chamadas["n"] > 1

    r = dados_vias.baixa_bibliotecas(
        tmp_path,
        bibs,
        confirmado=True,
        cancelado=cancela_no_segundo_bloco,
        abre=_abridor({bibs[0].url: a}),
    )
    assert r.desfecho == "cancelado"
    assert list(tmp_path.iterdir()) == []


def test_url_http_e_recusada_sem_abrir_a_rede(tmp_path: Path) -> None:
    bibs = [
        _bib("a.qs", b"abc"),
        _bib("b.qs", b"def", url="http://www.metaboanalyst.ca/resources/libs/b.qs"),
    ]
    abre = _abridor({bibs[0].url: b"abc"})
    r = dados_vias.baixa_bibliotecas(tmp_path, bibs, confirmado=True, abre=abre)
    assert r.desfecho == "transporte_recusado", r.motivo
    assert abre.chamadas == [bibs[0].url]  # type: ignore[attr-defined]
    assert list(tmp_path.iterdir()) == []


def test_resposta_acima_do_teto_e_descartada_sem_deixar_arquivo(tmp_path: Path) -> None:
    bibs = [_bib("a.qs", b"abc")]
    teto = dados_vias.FATOR_DO_TETO * 3
    r = dados_vias.baixa_bibliotecas(
        tmp_path, bibs, confirmado=True, abre=_abridor({bibs[0].url: b"z" * (teto + 1)})
    )
    assert r.desfecho == "transporte_recusado" and "teto" in r.motivo
    assert list(tmp_path.iterdir()) == []
    # par causal: no teto exato, passa
    r2 = dados_vias.baixa_bibliotecas(
        tmp_path, bibs, confirmado=True, abre=_abridor({bibs[0].url: b"z" * teto})
    )
    assert r2.desfecho == "baixado"


@pytest.mark.parametrize(
    ("destino", "recusa"),
    [
        ("http://www.metaboanalyst.ca/resources/libs/a.qs", True),
        ("https://outro-host.example/a.qs", True),
        ("https://www.metaboanalyst.ca/outro/a.qs", False),
    ],
)
def test_redirecionamento_so_para_o_mesmo_host_em_https(destino: str, recusa: bool) -> None:
    req = urllib.request.Request(HOST + "a.qs")
    manipulador = _RedirecionamentoRestrito()
    if recusa:
        with pytest.raises(TransporteRecusadoError):
            manipulador.redirect_request(req, None, 302, "Found", {}, destino)
    else:
        novo = manipulador.redirect_request(req, None, 302, "Found", {}, destino)
        assert novo is not None and novo.full_url == destino


def test_registro_adulterado_pede_o_download_de_novo(tmp_path: Path) -> None:
    a = b"abc" * 10
    bibs = [_bib("a.qs", a)]
    r = dados_vias.baixa_bibliotecas(
        tmp_path, bibs, confirmado=True, abre=_abridor({bibs[0].url: a})
    )
    assert dados_vias.confere_baixadas(tmp_path, bibs) == r.baixadas
    (tmp_path / "a.qs").write_bytes(b"mexido")
    assert dados_vias.confere_baixadas(tmp_path, bibs) is None
    (tmp_path / dados_vias.REGISTRO_DE_DOWNLOAD).write_text('[{"arquivo": 1}]', encoding="utf-8")
    assert dados_vias.confere_baixadas(tmp_path, bibs) is None


def test_corpo_mais_curto_que_o_content_length_nao_entra_na_pasta(tmp_path: Path) -> None:
    a = b"a" * 1000
    bibs = [_bib("a.qs", a)]
    r = dados_vias.baixa_bibliotecas(
        tmp_path, bibs, confirmado=True, abre=_serve("Content-Length: 1000\r\n", a[:500])
    )
    assert r.desfecho == "sem_rede" and "500 de 1000" in r.motivo, r.motivo
    assert list(tmp_path.iterdir()) == []
    assert dados_vias.confere_baixadas(tmp_path, bibs) is None
    # par causal: o mesmo cabeçalho com o corpo inteiro entra, e entra validado
    r2 = dados_vias.baixa_bibliotecas(
        tmp_path, bibs, confirmado=True, abre=_serve("Content-Length: 1000\r\n", a)
    )
    assert r2.desfecho == "baixado" and r2.baixadas[0].validada, r2.motivo


def test_corpo_chunked_cortado_vira_sem_rede_e_nao_escapa(tmp_path: Path) -> None:
    a = b"a" * 1000
    bibs = [_bib("a.qs", a)]
    cortado = b"3e8\r\n" + a[:500]
    r = dados_vias.baixa_bibliotecas(
        tmp_path, bibs, confirmado=True, abre=_serve("Transfer-Encoding: chunked\r\n", cortado)
    )
    assert r.desfecho == "sem_rede" and "IncompleteRead" in r.motivo, r.motivo
    assert list(tmp_path.iterdir()) == []
    # par causal: o mesmo `chunked`, terminado, entra
    inteiro = b"3e8\r\n" + a + b"\r\n0\r\n\r\n"
    r2 = dados_vias.baixa_bibliotecas(
        tmp_path, bibs, confirmado=True, abre=_serve("Transfer-Encoding: chunked\r\n", inteiro)
    )
    assert r2.desfecho == "baixado" and r2.baixadas[0].validada, r2.motivo


def test_content_length_que_nao_e_inteiro_recusa_o_canal(tmp_path: Path) -> None:
    a = b"a" * 1000
    bibs = [_bib("a.qs", a)]
    r = dados_vias.baixa_bibliotecas(
        tmp_path, bibs, confirmado=True, abre=_serve("Content-Length: mil\r\n", a)
    )
    assert r.desfecho == "transporte_recusado" and "Content-Length" in r.motivo, r.motivo
    assert list(tmp_path.iterdir()) == []
    # par causal: sem o cabeçalho, o `http.client` lê até fechar e o download entra
    r2 = dados_vias.baixa_bibliotecas(tmp_path, bibs, confirmado=True, abre=_serve("", a))
    assert r2.desfecho == "baixado", r2.motivo


def test_a_marca_de_validada_segue_o_manifesto_da_versao_que_corre(tmp_path: Path) -> None:
    a, trocado = b"abc" * 10, b"o-servidor-trocou"
    antigo = [_bib("a.qs", a)]
    r = dados_vias.baixa_bibliotecas(
        tmp_path / "1", antigo, confirmado=True, abre=_abridor({antigo[0].url: a})
    )
    assert r.baixadas[0].validada, "âncora: sob o manifesto que baixou, a cópia é validada"
    achadas = dados_vias.confere_baixadas(tmp_path / "1", [_bib("a.qs", b"refixado")])
    assert achadas is not None and not achadas[0].validada
    assert achadas[0].baixada_em == r.baixadas[0].baixada_em
    # a direção oposta: baixada não validada, e o manifesto novo fixa exactamente o que veio
    r2 = dados_vias.baixa_bibliotecas(
        tmp_path / "2", antigo, confirmado=True, abre=_abridor({antigo[0].url: trocado})
    )
    assert not r2.baixadas[0].validada, "âncora: sob o manifesto que baixou, a cópia não valida"
    achadas2 = dados_vias.confere_baixadas(tmp_path / "2", [_bib("a.qs", trocado)])
    assert achadas2 is not None and achadas2[0].validada


def test_o_abridor_padrao_monta_tls_verificado_e_o_redirecionamento_restrito(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_abre_com_seguranca` é a única peça que liga os controles de transporte ao `urllib`.

    Todo download dos outros testes injeta `abre`, então tirar o manipulador ou trocar o contexto
    TLS aqui não reprovaria nenhum deles. A cadeia é montada pelo `build_opener` real e
    inspecionada; só o `open` é substituído, e por isso nada sai para a rede.
    """
    montar_de_verdade = urllib.request.build_opener
    montados: list[tuple[urllib.request.OpenerDirector, list[tuple[str, float]]]] = []

    class _Abridor:
        def __init__(self, cadeia: urllib.request.OpenerDirector) -> None:
            self.abertas: list[tuple[str, float]] = []
            montados.append((cadeia, self.abertas))

        def open(self, url: str, timeout: float) -> str:
            self.abertas.append((url, timeout))
            return "resposta"

    def monta(*manipuladores: urllib.request.BaseHandler) -> _Abridor:
        return _Abridor(montar_de_verdade(*manipuladores))

    monkeypatch.setattr(urllib.request, "build_opener", monta)
    assert dados_vias._abre_com_seguranca(HOST + "a.qs", 7.0) == "resposta"
    ((cadeia, abertas),) = montados
    assert abertas == [(HOST + "a.qs", 7.0)]
    redirecionadores = [
        type(m) for m in cadeia.handlers if isinstance(m, urllib.request.HTTPRedirectHandler)
    ]
    assert redirecionadores == [_RedirecionamentoRestrito]
    (https,) = [m for m in cadeia.handlers if isinstance(m, urllib.request.HTTPSHandler)]
    contexto: ssl.SSLContext = https._context  # type: ignore[attr-defined]
    assert contexto.verify_mode == ssl.CERT_REQUIRED and contexto.check_hostname


@pytest.mark.parametrize(
    ("erro", "desfecho", "trecho"),
    [
        (
            urllib.error.URLError(ssl.SSLCertVerificationError("certificado vencido")),
            "transporte_recusado",
            "certificado",
        ),
        # par causal do de cima: o mesmo `URLError`, com outra razão, é falta de rede
        (urllib.error.URLError(ConnectionResetError("a conexão caiu")), "sem_rede", "caiu"),
        (
            urllib.error.HTTPError(
                HOST + "a.qs", 404, "Not Found", http.client.HTTPMessage(), None
            ),
            "transporte_recusado",
            "404",
        ),
        (TimeoutError("demorou demais"), "sem_rede", "demorou"),
        (http.client.RemoteDisconnected("fechou sem resposta"), "sem_rede", "fechou"),
        (http.client.BadStatusLine("lixo"), "sem_rede", "lixo"),
    ],
    ids=["certificado", "conexao-caiu", "http-404", "timeout", "desconectado", "status-ilegivel"],
)
def test_cada_erro_do_transporte_tem_o_seu_desfecho(
    tmp_path: Path, erro: BaseException, desfecho: str, trecho: str
) -> None:
    bibs = [_bib("a.qs", b"abc")]

    def abre(_url: str, _t: float) -> _Resposta:
        raise erro

    r = dados_vias.baixa_bibliotecas(tmp_path, bibs, confirmado=True, abre=abre)
    assert (r.desfecho, trecho in r.motivo) == (desfecho, True), r.motivo
    assert list(tmp_path.iterdir()) == []


def _spec_leva_o_manifesto(texto: str) -> bool:
    return '"pathways" / "dados_vias.json"' in texto and '"refino/pathways"' in texto


def test_o_congelado_leva_o_manifesto_no_datas() -> None:
    """JSON não é módulo: sem a entrada no `datas=`, o executável não acharia o manifesto."""
    spec = (Path(__file__).resolve().parent.parent / "refino-desktop.spec").read_text(
        encoding="utf-8"
    )
    assert _spec_leva_o_manifesto(spec)
    # par causal: sem a linha, o predicado acusa
    assert not _spec_leva_o_manifesto(
        spec.replace('"pathways" / "dados_vias.json"', '"pathways" / "outro.json"')
    )

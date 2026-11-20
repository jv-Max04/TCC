"""`scripts/busca_dados_vias.py`: os dois desfechos de reprovação que só ele tem (P2#4).

O script é portão chamado por subprocesso em `scripts/congela_windows.ps1` e em
`scripts/prepara_entrega.ps1`, e até aqui só o desfecho ok tinha sido visto. Os dois ramos de
reprovação são dele e não da fronteira `refino.io.dados_vias`, que `tests/test_dados_vias.py` já
cobre:

- o **`sobram`** do `--so-confere`, que reprova arquivo que o manifesto não lista — e que olhava só
  o primeiro nível da pasta, enquanto o congelamento a copia com `Copy-Item -Recurse`;
- as **`trocadas`** do download, que apagam as baixadas quando o hash não é o do manifesto, para a
  pasta do laboratório não levar uma biblioteca que ninguém validou.

**Os bytes são sintéticos, e isso não fere a regra 1 do README**, pelo mesmo motivo de
`tests/test_dados_vias.py`: o que se testa é o mecanismo — comparar hash, listar o que sobra,
apagar o que não valida —, e ele é o mesmo para qualquer conteúdo. O manifesto também é sintético
e passa pela validação REAL de `le_manifesto`, que só recebe outro caminho.

**Sem rede.** O script não aceita abridor nem manifesto por argumento, e mudar isso mudaria o
contrato dele; então o teste troca `dados_vias._abre_com_seguranca`, que é o único ponto por onde o
download chega à rede quando o chamador não passa `abre` — e o script não passa.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from refino.io import dados_vias

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "busca_dados_vias.py"
ORIGEM = "https://www.metaboanalyst.ca/resources/libs/"

# Duas bibliotecas, e não uma: com uma só, «apagou as baixadas» e «apagou a trocada» seriam a
# mesma afirmação, e o caso (c) troca só UMA das duas.
CONTEUDO: dict[str, bytes] = {"alfa.qs": b"alfa" * 300, "beta.qs": b"beta" * 500}


def _modulo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("busca_dados_vias", SCRIPT)
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["busca_dados_vias"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


busca = _modulo()


class _Resposta:
    def __init__(self, dados: bytes) -> None:
        self._io = io.BytesIO(dados)
        self.headers = {"Content-Length": str(len(dados))}

    def read(self, n: int = -1, /) -> bytes:
        return self._io.read(n)

    def close(self) -> None:
        pass


@pytest.fixture
def manifesto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Um manifesto sintético no lugar do empacotado, lido pela validação real."""
    alvo = tmp_path / "dados_vias.json"
    alvo.write_text(
        json.dumps(
            {
                "origem": ORIGEM,
                "medido_em": "2026-10-01",
                "bibliotecas": [
                    {
                        "arquivo": nome,
                        "caminho_remoto": nome,
                        "tamanho": len(dados),
                        "sha256": hashlib.sha256(dados).hexdigest(),
                    }
                    for nome, dados in CONTEUDO.items()
                ],
            }
        ),
        encoding="utf-8",
    )
    original = dados_vias.le_manifesto
    monkeypatch.setattr(dados_vias, "le_manifesto", lambda caminho=None: original(alvo))
    return alvo


def _abridor_falso(monkeypatch: pytest.MonkeyPatch, por_arquivo: dict[str, bytes]) -> list[str]:
    """Serve `por_arquivo` no lugar da rede; devolve a lista das URLs pedidas."""
    pedidas: list[str] = []

    def abre(url: str, _tempo: float) -> _Resposta:
        pedidas.append(url)
        return _Resposta(por_arquivo[url.removeprefix(ORIGEM)])

    monkeypatch.setattr(dados_vias, "_abre_com_seguranca", abre)
    return pedidas


def _roda(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *args: str
) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", ["busca_dados_vias.py", *args])
    codigo = busca.main()
    return codigo, capsys.readouterr().out


def _pasta_que_confere(destino: Path) -> Path:
    destino.mkdir()
    for nome, dados in CONTEUDO.items():
        (destino / nome).write_bytes(dados)
    return destino


def _sobram(saida: str) -> str:
    assert "sobram " in saida, saida
    return saida.split("sobram ", 1)[1].strip()


# --- (a) a pasta que confere ------------------------------------------------------------------


@pytest.mark.usefixtures("manifesto")
def test_so_confere_aprova_a_pasta_que_confere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destino = _pasta_que_confere(tmp_path / "vias-dados")
    codigo, saida = _roda(monkeypatch, capsys, "--destino", str(destino), "--so-confere")
    assert codigo == 0, saida
    assert "conferência: ok" in saida
    assert _sobram(saida) == "nenhum"


# --- (b) arquivo a mais, no primeiro nível e dentro de subpasta --------------------------------


@pytest.fixture(params=["leia-me.txt", "extra/leia-me.txt", "copia/alfa.qs"])
def a_mais(request: pytest.FixtureRequest) -> str:
    """O arquivo plantado, pelo caminho relativo.

    `copia/alfa.qs` tem o NOME de uma biblioteca do manifesto: a comparação por nome a esconderia,
    e a cópia recursiva a levaria ao laboratório sem conferência.
    """
    valor: str = request.param
    return valor


@pytest.mark.usefixtures("manifesto")
def test_so_confere_reprova_arquivo_que_o_manifesto_nao_lista(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    a_mais: str,
) -> None:
    destino = _pasta_que_confere(tmp_path / "vias-dados")
    plantado = destino / a_mais
    plantado.parent.mkdir(parents=True, exist_ok=True)
    plantado.write_bytes(CONTEUDO["alfa.qs"] if a_mais.endswith(".qs") else b"nao listado")
    # A planta aterrou, e as bibliotecas continuam a conferir: o único motivo para reprovar é ela.
    assert plantado.is_file()
    assert dados_vias.confere_pasta(destino, dados_vias.le_manifesto()).estado == "ok"

    codigo, saida = _roda(monkeypatch, capsys, "--destino", str(destino), "--so-confere")
    assert codigo == 1, saida
    assert "conferência: ok" in saida
    assert _sobram(saida) == repr([a_mais])


# --- (c) o download que traz bytes com outro hash ---------------------------------------------


@pytest.mark.usefixtures("manifesto")
def test_download_com_hash_do_manifesto_aprova_e_deixa_a_pasta_conferida(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """O par do caso seguinte, com o mesmo arnês: só os bytes servidos diferem.

    Sem ele, o «sai 1» do caso trocado poderia vir de o arnês nunca conseguir baixar.
    """
    pedidas = _abridor_falso(monkeypatch, dict(CONTEUDO))
    destino = tmp_path / "vias-dados"
    codigo, saida = _roda(monkeypatch, capsys, "--destino", str(destino))
    assert codigo == 0, saida
    assert sorted(pedidas) == sorted(ORIGEM + nome for nome in CONTEUDO)
    assert dados_vias.confere_pasta(destino, dados_vias.le_manifesto()).estado == "ok"
    # Na pasta do laboratório vale o manifesto, e o registro da release pública sai.
    assert not (destino / dados_vias.REGISTRO_DE_DOWNLOAD).exists()


@pytest.mark.usefixtures("manifesto")
def test_download_com_hash_trocado_reprova_e_nao_deixa_biblioteca_nao_validada(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    servido = dict(CONTEUDO)
    servido["beta.qs"] = b"outra" * 400  # outro hash, e cabe no teto de 4x o tamanho
    assert (
        hashlib.sha256(servido["beta.qs"]).digest() != hashlib.sha256(CONTEUDO["beta.qs"]).digest()
    )
    pedidas = _abridor_falso(monkeypatch, servido)
    destino = tmp_path / "vias-dados"

    # Espia o download para ancorar a ausência do fim: as duas bibliotecas CHEGARAM à pasta, e a
    # pasta vazia depois é porque o script as apagou — não porque nada foi baixado.
    original = dados_vias.baixa_bibliotecas
    logo_depois: list[tuple[str, list[str]]] = []

    def espia(*args: object, **kwargs: object) -> dados_vias.ResultadoDoDownload:
        r: dados_vias.ResultadoDoDownload = original(*args, **kwargs)  # type: ignore[arg-type]
        logo_depois.append((r.desfecho, sorted(p.name for p in destino.iterdir())))
        return r

    monkeypatch.setattr(dados_vias, "baixa_bibliotecas", espia)

    codigo, saida = _roda(monkeypatch, capsys, "--destino", str(destino))

    assert codigo == 1, saida
    assert "o servidor trocou 1 biblioteca(s) desde o manifesto: beta.qs" in saida
    assert sorted(pedidas) == sorted(ORIGEM + nome for nome in CONTEUDO)
    assert logo_depois == [("baixado", sorted([*CONTEUDO, dados_vias.REGISTRO_DE_DOWNLOAD]))]
    # Tudo-ou-nada: sai a trocada e sai a que validou, porque vieram do mesmo download.
    restantes = sorted(p.relative_to(destino).as_posix() for p in destino.rglob("*"))
    assert restantes == [], restantes

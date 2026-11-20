"""O conferidor do «Palpite» no congelado acusa o binário com o modelo errado.

`scripts/confere_palpite_no_congelado.py` decide se a versão a publicar leva a política retreinada
no corpus de 7 QCs. Os testes são todos EM PROCESSO: a pasta, a resposta gravada, o commit e o
estado de `src/` são argumentos da função sob teste, então nenhum negativo pode ficar inerte por um
intermediário que não recebe a planta.

A âncora de poder vem antes das plantas: o modelo da `windows-2026.10.15.1`, lido do histórico em
`a488add`, dá pela mesma rota uma resposta DIFERENTE da de `src/` no `neg_c18`. Sem
isso, plantar a resposta velha seria plantar a mesma coisa, e o DIVERGE que se espera dela seria
sorte. A âncora não afirma QUAL campo difere: que nessa matriz os dois modelos proponham o mesmo
corretor é medição sobre estes dois artefatos (está na docstring do script), e não propriedade a
congelar num teste — um retreino seguinte pode mudá-la.

Os três desfechos são vistos: APROVA com a pasta igual à árvore; DIVERGE com a resposta
do modelo anterior, com o artefato trocado ou ausente e com uma resposta de erro; NAO_MEDIU sem a
resposta gravada, com a resposta de um congelamento anterior, com a pasta de outro commit e com
`src/` alterado. E o `git status` que alimenta a
última guarda é exercitado num repositório de verdade, nas duas direções.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.conftest import measurements_path, sample_list_path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "confere_palpite_no_congelado.py"
MODO = "neg_c18"
# O commit de que a `windows-2026.10.15.1` foi congelada: o modelo que o binário publicado
# antes do retreino leva.
COMMIT_DA_VERSAO_ANTERIOR = "a488add"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _modulo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("confere_palpite_no_congelado", SCRIPT)
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["confere_palpite_no_congelado"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


conferidor = _modulo()


def _limpo() -> str:
    return ""


@pytest.fixture(scope="module")
def entrada() -> Path:
    sample_list_path(MODO)
    return measurements_path(MODO).parent


@pytest.fixture(scope="module")
def resposta_de_src(entrada: Path) -> dict[str, Any]:
    resposta: dict[str, Any] = conferidor.resposta_local(entrada)
    return resposta


@pytest.fixture(scope="module")
def resposta_do_modelo_anterior(
    entrada: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    """A mesma rota, servindo o artefato que a versão anterior leva — o «binário velho»."""
    import refino.api.rotas as rotas
    from refino.io.politica import carrega_politica_aprendida

    lido = subprocess.run(
        ["git", "show", f"{COMMIT_DA_VERSAO_ANTERIOR}:src/refino/decision/politica_aprendida.json"],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if lido.returncode != 0:
        pytest.skip(
            f"clone raso: o commit {COMMIT_DA_VERSAO_ANTERIOR}, de que a versão anterior foi "
            "congelada, não está no histórico deste clone"
        )
    artefato = tmp_path_factory.mktemp("anterior") / "politica_aprendida.json"
    artefato.write_bytes(lido.stdout)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            rotas, "carrega_politica_aprendida", lambda: carrega_politica_aprendida(artefato)
        )
        resposta: dict[str, Any] = conferidor.resposta_local(entrada)
    return resposta


def _pasta(tmp_path: Path, commit: str = COMMIT, artefato: bytes | None = None) -> Path:
    """Uma pasta com o que o conferidor lê de um congelamento: o carimbo e o artefato."""
    pasta = tmp_path / "refino"
    destino = pasta / conferidor.ARTEFATO_NA_PASTA
    destino.parent.mkdir(parents=True)
    destino.write_bytes(conferidor.ARTEFATO_NO_REPO.read_bytes() if artefato is None else artefato)
    # O segundo artefato vai sempre íntegro: os testes que o plantam trocam-no depois.
    (pasta / conferidor.ARTEFATO_PELA_VERDADE_NA_PASTA).write_bytes(
        conferidor.ARTEFATO_PELA_VERDADE_NO_REPO.read_bytes()
    )
    # As linhas de cima do `VERSAO.txt` que o cenário 1b de `congela_windows.ps1` escreve.
    (pasta / "VERSAO.txt").write_text(
        "refino - aplicativo nativo de Windows\n\n"
        f"commit    : {commit}\n"
        "arvore    : arvore limpa\n",
        encoding="ascii",
    )
    return pasta


def _gravada(tmp_path: Path, corpo: Any) -> Path:
    caminho = tmp_path / "refino-u10" / "fluxo-recomendar.json"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(corpo, ensure_ascii=False), encoding="utf-8")
    return caminho


def test_o_modelo_anterior_responde_outra_coisa_no_neg_c18(
    resposta_de_src: dict[str, Any], resposta_do_modelo_anterior: dict[str, Any]
) -> None:
    """A âncora: a planta da resposta velha cai sobre dado em que as duas respostas diferem."""
    assert resposta_de_src["correction_proposta"], resposta_de_src
    assert conferidor.campos_divergentes(resposta_do_modelo_anterior, resposta_de_src)


def test_a_pasta_igual_a_src_aprova(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    relato = conferidor.confere(
        _pasta(tmp_path),
        _gravada(tmp_path, resposta_de_src),
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
    )
    assert relato["veredito"] == "APROVA", relato
    assert conferidor.codigo_de_saida(relato) == conferidor.APROVA
    # Não-vacuidade: a comparação correu, sobre uma resposta com corpo.
    assert relato["campos_divergentes"] == []
    assert relato["resposta_do_binario"]["correction_proposta"]
    assert relato["sha256_na_pasta"] == relato["sha256_no_repo"]
    assert relato["sha256_pela_verdade_na_pasta"] == relato["sha256_pela_verdade_no_repo"]


def test_a_resposta_do_modelo_anterior_diverge(
    tmp_path: Path, entrada: Path, resposta_do_modelo_anterior: dict[str, Any]
) -> None:
    relato = conferidor.confere(
        _pasta(tmp_path),
        _gravada(tmp_path, resposta_do_modelo_anterior),
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
    )
    assert relato["veredito"] == "DIVERGE", relato
    assert conferidor.codigo_de_saida(relato) == conferidor.DIVERGE
    assert relato["campos_divergentes"], relato


def test_o_artefato_trocado_diverge_mesmo_com_a_resposta_igual(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    """Um byte a mais no artefato da pasta: a resposta não muda, e o sha256 acusa sozinho."""
    pasta = _pasta(tmp_path, artefato=conferidor.ARTEFATO_NO_REPO.read_bytes() + b"\n")
    relato = conferidor.confere(
        pasta,
        _gravada(tmp_path, resposta_de_src),
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
    )
    assert relato["veredito"] == "DIVERGE", relato
    assert relato["campos_divergentes"] == []
    assert any("não é o de src/" in linha for linha in relato["divergencias"]), relato


def test_o_artefato_pela_verdade_trocado_ou_ausente_diverge(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    """O segundo artefato: trocado acusa pelo sha256, ausente acusa pela falta.

    **A resposta gravada é a de `src/` nos dois casos**, então o que reprova é só o item 2 — e é
    isso que prova que a guarda nova age sozinha, e não pela comparação da resposta.
    """
    trocada = tmp_path / "trocada"
    pasta = _pasta(trocada)
    alvo = pasta / conferidor.ARTEFATO_PELA_VERDADE_NA_PASTA
    alvo.write_bytes(alvo.read_bytes() + b"\n")
    relato = conferidor.confere(
        pasta,
        _gravada(trocada, resposta_de_src),
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
    )
    assert relato["veredito"] == "DIVERGE", relato
    assert relato["campos_divergentes"] == []
    assert relato["divergencias"] == ["o politica_pela_verdade.json da pasta não é o de src/"]

    ausente = tmp_path / "ausente"
    pasta = _pasta(ausente)
    (pasta / conferidor.ARTEFATO_PELA_VERDADE_NA_PASTA).unlink()
    relato = conferidor.confere(
        pasta,
        _gravada(ausente, resposta_de_src),
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
    )
    assert relato["veredito"] == "DIVERGE", relato
    assert any("politica_pela_verdade.json" in linha for linha in relato["divergencias"]), relato


def test_o_artefato_ausente_diverge(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    pasta = _pasta(tmp_path)
    (pasta / conferidor.ARTEFATO_NA_PASTA).unlink()
    relato = conferidor.confere(
        pasta,
        _gravada(tmp_path, resposta_de_src),
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
    )
    assert relato["veredito"] == "DIVERGE", relato
    assert any("não leva o artefato" in linha for linha in relato["divergencias"]), relato


def test_a_resposta_de_erro_diverge_e_nao_aprova_pela_intersecao_vazia(
    tmp_path: Path, entrada: Path
) -> None:
    """O corpo de um `503` não tem nenhum campo da resposta boa: comparar só os comuns aprovaria."""
    erro = {"detail": "artefato da política aprendida ausente"}
    relato = conferidor.confere(
        _pasta(tmp_path),
        _gravada(tmp_path, erro),
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
    )
    assert relato["veredito"] == "DIVERGE", relato
    assert {"detail", "correction_proposta"} <= set(relato["campos_divergentes"])


def test_sem_a_resposta_gravada_nao_mediu(tmp_path: Path, entrada: Path) -> None:
    relato = conferidor.confere(
        _pasta(tmp_path),
        tmp_path / "refino-u10" / "fluxo-recomendar.json",
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
    )
    assert relato["veredito"] == "NAO_MEDIU", relato
    assert conferidor.codigo_de_saida(relato) == conferidor.NAO_MEDIU


def test_a_resposta_de_um_congelamento_anterior_nao_mediu(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    """A pasta de saídas do congelamento nunca é limpa: uma resposta mais velha que a pasta é
    de outro congelamento, e aprová-la seria aprovar o que este não mediu."""
    pasta = _pasta(tmp_path)
    gravada = _gravada(tmp_path, resposta_de_src)
    carimbo = (pasta / "VERSAO.txt").stat().st_mtime
    os.utime(gravada, (carimbo - 3600, carimbo - 3600))
    relato = conferidor.confere(pasta, gravada, entrada, commit_esperado=COMMIT, alteracoes=_limpo)
    assert relato["veredito"] == "NAO_MEDIU", relato
    assert any("anterior a este congelamento" in linha for linha in relato["nao_medido"])

    os.utime(gravada, (carimbo + 1, carimbo + 1))
    relato = conferidor.confere(pasta, gravada, entrada, commit_esperado=COMMIT, alteracoes=_limpo)
    assert relato["veredito"] == "APROVA", relato


def test_a_pasta_de_outro_commit_nao_mediu_e_nao_compara(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    relato = conferidor.confere(
        _pasta(tmp_path, commit="f" * 40),
        _gravada(tmp_path, resposta_de_src),
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
    )
    assert relato["veredito"] == "NAO_MEDIU", relato
    assert "sha256_na_pasta" not in relato and "campos_divergentes" not in relato


def test_src_alterado_nao_mediu(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    relato = conferidor.confere(
        _pasta(tmp_path),
        _gravada(tmp_path, resposta_de_src),
        entrada,
        commit_esperado=COMMIT,
        alteracoes=lambda: " M src/refino/decision/features.py",
    )
    assert relato["veredito"] == "NAO_MEDIU", relato
    assert any("src/" in linha for linha in relato["nao_medido"]), relato


def test_o_estado_de_src_le_o_git_nas_duas_direcoes(tmp_path: Path) -> None:
    """A guarda de `src/` num repositório de verdade: limpo, alterado, e alterado fora dele."""

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "LEIA.md").write_text("um\n", encoding="utf-8")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "inicial")
    assert conferidor.alteracoes_em_src(tmp_path) == ""

    (tmp_path / "LEIA.md").write_text("dois\n", encoding="utf-8")
    assert conferidor.alteracoes_em_src(tmp_path) == ""

    (tmp_path / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert "src/a.py" in conferidor.alteracoes_em_src(tmp_path)


def test_main_grava_o_relatorio_e_devolve_o_codigo(
    tmp_path: Path,
    entrada: Path,
    resposta_de_src: dict[str, Any],
    resposta_do_modelo_anterior: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A linha de comando, nos dois desfechos que decidem a publicação.

    `main` pede sempre a quarta conferência, então o Palpite medido vai gravado e igual ao de
    `src/` — pelo duplo `_local`, definido abaixo — nos dois desfechos: o que este teste separa
    continua a ser só a resposta da árvore.
    """
    monkeypatch.setattr(conferidor, "alteracoes_em_src", _limpo)
    corpo_medido = _corpo_do_palpite()
    monkeypatch.setattr(conferidor, "palpite_medido_local", _local(corpo_medido))
    pasta = _pasta(tmp_path)
    relatorio = tmp_path / "verdicto.json"
    medido = _palpite_gravado(tmp_path, corpo_medido)
    comuns = [
        "--pasta",
        str(pasta),
        "--entrada",
        str(entrada),
        "--commit",
        COMMIT,
        "--palpite-medido",
        str(medido),
    ]

    boa = _gravada(tmp_path / "boa", resposta_de_src)
    assert conferidor.main([*comuns, "--resposta", str(boa), "--relatorio", str(relatorio)]) == 0
    assert json.loads(relatorio.read_text(encoding="utf-8"))["veredito"] == "APROVA"

    velha = _gravada(tmp_path / "velha", resposta_do_modelo_anterior)
    assert conferidor.main([*comuns, "--resposta", str(velha), "--relatorio", str(relatorio)]) == 1
    assert json.loads(relatorio.read_text(encoding="utf-8"))["veredito"] == "DIVERGE"


# ---------------------------------------------------------------------------------------------
# A quarta conferência — o Palpite MEDIDO, que a janela usa
# ---------------------------------------------------------------------------------------------
#
# **A medição daqui custa minutos**, então nos testes em processo quem a produz é um duplo, e o
# corpo dele tem a forma da rota com números escolhidos — o precedente é o `_Medidor` de
# `tests/test_api_palpite.py`. A conferência compara dois corpos por igualdade, e o que ela decide
# não depende dos valores. **O caminho real corre no teste `slow` do fim**, contra os vetores da
# população gravados: o outro caminho como oráculo.


def _corpo_do_palpite(**troca: Any) -> dict[str, Any]:
    from refino.decision.palpite import decide_o_palpite

    medido = decide_o_palpite({"none": 0.31, "loess": 0.2, "serrf": 0.186}, n_features_medidas=200)
    corpo: dict[str, Any] = {
        "segunda_opiniao": None,
        "correction_proposta": medido.correction,
        "escores": dict(medido.escores),
        "margem": medido.margem,
        "segundo": medido.segundo,
        "confianca": medido.confianca,
        "abstem": medido.abstem,
        "avaliavel": medido.avaliavel,
        "rationale": medido.rationale,
        "n_features_medidas": medido.n_features_medidas,
        "limiar_de_empate": medido.limiar,
        "variante": "outlier_removed",
        "n_lotes": 1,
        "qcs_por_lote": [7],
        "disponivel_para_execucao": True,
    }
    corpo.update(troca)
    return corpo


def _palpite_gravado(tmp_path: Path, corpo: Any) -> Path:
    caminho = tmp_path / "verifica-app" / "palpite-medido.json"
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(corpo, ensure_ascii=False), encoding="utf-8")
    return caminho


def _local(corpo: dict[str, Any]) -> Any:
    chamadas: list[tuple[Path, str]] = []

    def palpite_local(entrada: Path, modo: str) -> dict[str, Any]:
        chamadas.append((entrada, modo))
        return dict(corpo)

    palpite_local.chamadas = chamadas  # type: ignore[attr-defined]
    return palpite_local


def _confere_com_palpite(
    tmp_path: Path,
    entrada: Path,
    resposta_de_src: dict[str, Any],
    gravado: Path | None,
    local: Any,
) -> dict[str, Any]:
    pasta = _pasta(tmp_path)
    resposta = _gravada(tmp_path, resposta_de_src)
    if gravado is not None and gravado.is_file():
        # Depois do carimbo da pasta, como na ordem do roteiro: congelar, e só então a prova.
        os.utime(gravado)
    relato: dict[str, Any] = conferidor.confere(
        pasta,
        resposta,
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
        palpite_gravado=gravado,
        palpite_local=local,
    )
    return relato


def test_o_palpite_medido_igual_ao_de_src_aprova(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    corpo = _corpo_do_palpite()
    local = _local(corpo)
    relato = _confere_com_palpite(
        tmp_path, entrada, resposta_de_src, _palpite_gravado(tmp_path, corpo), local
    )
    assert relato["veredito"] == "APROVA", relato
    # Não-vacuidade: a comparação correu, uma vez, sobre o modo do roteiro, com corpo dos dois
    # lados.
    assert local.chamadas == [(entrada, MODO)]
    assert relato["campos_divergentes_do_palpite_medido"] == []
    assert relato["palpite_medido_do_binario"]["correction_proposta"] == "serrf"
    assert relato["segundos_do_palpite_de_src"] >= 0


def test_o_palpite_medido_diferente_diverge_ate_no_ultimo_bit(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    """A igualdade é EXACTA: um escore diferente em 1e-12 já é outro binário (semente 42)."""
    corpo = _corpo_do_palpite()
    escores = dict(corpo["escores"], loess=corpo["escores"]["loess"] + 1e-12)
    assert escores != corpo["escores"]
    relato = _confere_com_palpite(
        tmp_path,
        entrada,
        resposta_de_src,
        _palpite_gravado(tmp_path, dict(corpo, escores=escores)),
        _local(corpo),
    )
    assert relato["veredito"] == "DIVERGE", relato
    assert relato["campos_divergentes_do_palpite_medido"] == ["escores"]

    outro = tmp_path / "outro"
    relato = _confere_com_palpite(
        outro,
        entrada,
        resposta_de_src,
        _palpite_gravado(outro, dict(corpo, correction_proposta="loess")),
        _local(corpo),
    )
    assert relato["veredito"] == "DIVERGE", relato
    assert "correction_proposta" in relato["campos_divergentes_do_palpite_medido"]


def test_o_erro_do_binario_no_palpite_medido_diverge(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    """O corpo de um `422` não tem campo nenhum do Palpite: a interseção vazia não aprova."""
    erro = {"detail": "com 1 lote(s), os QCs por lote são [2]"}
    relato = _confere_com_palpite(
        tmp_path,
        entrada,
        resposta_de_src,
        _palpite_gravado(tmp_path, erro),
        _local(_corpo_do_palpite()),
    )
    assert relato["veredito"] == "DIVERGE", relato
    assert {"detail", "correction_proposta"} <= set(relato["campos_divergentes_do_palpite_medido"])


def test_sem_o_palpite_medido_gravado_nao_mediu_e_nao_mede_aqui(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    local = _local(_corpo_do_palpite())
    relato = _confere_com_palpite(
        tmp_path, entrada, resposta_de_src, tmp_path / "verifica-app" / "palpite-medido.json", local
    )
    assert relato["veredito"] == "NAO_MEDIU", relato
    assert any("verifica_o_app.ps1" in linha for linha in relato["nao_medido"]), relato
    assert local.chamadas == []


def test_o_palpite_medido_de_antes_do_congelamento_nao_mediu(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    corpo = _corpo_do_palpite()
    pasta = _pasta(tmp_path)
    gravado = _palpite_gravado(tmp_path, corpo)
    carimbo = (pasta / "VERSAO.txt").stat().st_mtime
    comuns: dict[str, Any] = {
        "commit_esperado": COMMIT,
        "alteracoes": _limpo,
        "palpite_gravado": gravado,
        "palpite_local": _local(corpo),
    }
    os.utime(gravado, (carimbo - 3600, carimbo - 3600))
    resposta = _gravada(tmp_path, resposta_de_src)
    relato = conferidor.confere(pasta, resposta, entrada, **comuns)
    assert relato["veredito"] == "NAO_MEDIU", relato
    assert any("Palpite medido gravado é anterior" in linha for linha in relato["nao_medido"])

    os.utime(gravado, (carimbo + 1, carimbo + 1))
    relato = conferidor.confere(pasta, resposta, entrada, **comuns)
    assert relato["veredito"] == "APROVA", relato


def test_a_quarta_conferencia_nao_pedida_fica_escrita_no_relato(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    """`None` não é aprovação em silêncio: o relato diz que ela não foi pedida."""
    local = _local(_corpo_do_palpite())
    relato = _confere_com_palpite(tmp_path, entrada, resposta_de_src, None, local)
    assert relato["veredito"] == "APROVA", relato
    assert relato["palpite_medido"] == "não pedido"
    assert local.chamadas == []


def test_main_pede_a_quarta_conferencia_sempre(
    tmp_path: Path,
    entrada: Path,
    resposta_de_src: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A linha de comando pede a quarta conferência mesmo sem a opção — e sem o corpo, não mede."""
    corpo = _corpo_do_palpite()
    monkeypatch.setattr(conferidor, "alteracoes_em_src", _limpo)
    monkeypatch.setattr(conferidor, "palpite_medido_local", _local(corpo))
    pasta = _pasta(tmp_path)
    relatorio = tmp_path / "verdicto.json"
    comuns = [
        "--pasta",
        str(pasta),
        "--entrada",
        str(entrada),
        "--commit",
        COMMIT,
        "--resposta",
        str(_gravada(tmp_path, resposta_de_src)),
        "--relatorio",
        str(relatorio),
    ]
    # Sem a opção, o padrão é lido na CHAMADA de `main`: aponta-se para um caminho que não existe,
    # e a conferência acontece — e não mede —, em vez de ficar de fora em silêncio.
    padrao = tmp_path / "nao-existe.json"
    monkeypatch.setattr(conferidor, "PALPITE_MEDIDO_GRAVADO", padrao)
    assert conferidor.main(comuns) == 2
    lido = json.loads(relatorio.read_text(encoding="utf-8"))
    assert lido["veredito"] == "NAO_MEDIU"
    assert any(str(padrao) in linha for linha in lido["nao_medido"]), lido
    gravado = _palpite_gravado(tmp_path, corpo)
    assert conferidor.main([*comuns, "--palpite-medido", str(gravado)]) == 0
    assert json.loads(relatorio.read_text(encoding="utf-8"))["veredito"] == "APROVA"
    ausente = tmp_path / "ausente.json"
    assert conferidor.main([*comuns, "--palpite-medido", str(ausente)]) == 2
    assert json.loads(relatorio.read_text(encoding="utf-8"))["veredito"] == "NAO_MEDIU"


@pytest.fixture(scope="module")
def palpite_local_e_gravado(entrada: Path) -> tuple[dict[str, Any], dict[str, float]]:
    """A rota local no `neg_c18` e as medianas dos vetores gravados nas mesmas 200 features.

    Uma medição para os dois testes de baixo: a rota custa perto de um minuto com 12 núcleos.

    Os vetores são os da F2, gravados no Windows com o SERRF e o LOESS em uso.
    """
    import numpy as np

    from refino.decision.palpite import N_FEATURES_DO_PALPITE
    from refino.io.readers import load_matrix
    from refino.metrics.crossval import sampled_feature_names

    corpo = conferidor.palpite_medido_local(entrada)
    matriz = load_matrix(
        entrada / f"measurements_{MODO}.csv", entrada / f"Sample_List_{MODO}.csv", mode=MODO
    )
    vetores = REPO / f"data/fixtures/medicoes/jh/f2_vetores/unidades/real-{MODO}.npz"
    with np.load(vetores) as f2:
        gravado = {c: f2[c] for c in ("none", "loess", "serrf")}
    nomes = set(sampled_feature_names(matriz, N_FEATURES_DO_PALPITE, 42) or ())
    idx = [i for i, n in enumerate(matriz.feature_names) if n in nomes]
    assert len(idx) == 200
    return corpo, {c: float(np.nanmedian(v[idx])) for c, v in gravado.items()}


@pytest.mark.slow
def test_o_palpite_medido_local_e_o_da_populacao_restrita_as_200(
    palpite_local_e_gravado: tuple[dict[str, Any], dict[str, float]],
) -> None:
    """O caminho real da quarta conferência, contra o outro caminho como oráculo.

    A medição das 200 sorteadas dá o mesmo RSD por feature que a da população inteira do
    `neg_c18`, então os escores da rota são as medianas do gravado nas mesmas 200 — igualdade
    exacta.

    Este confere o instrumento em qualquer sistema; o de baixo, os dois corretores e a proposta.
    Uma reprovação diz qual dos dois quebrou.
    """
    corpo, esperado = palpite_local_e_gravado
    assert corpo["escores"]["none"] == esperado["none"]
    assert corpo["n_features_medidas"] == 200


@pytest.mark.slow
@pytest.mark.skipif(
    sys.platform != "win32",
    reason="só no Windows: os vetores da F2 foram gravados no Windows, e fora dele o LOESS "
    "fica a 1 ULP da mediana gravada",
)
def test_os_escores_do_serrf_e_do_loess_da_rota_local_sao_os_dos_vetores_gravados(
    palpite_local_e_gravado: tuple[dict[str, Any], dict[str, float]],
) -> None:
    """A parte do teste de cima que confere os dois corretores e a proposta.

    O teste corre só no Windows, onde os vetores foram gravados. Lá a igualdade bit a bit foi vista
    no `mede_o_palpite`, e não nesta rota. Num contêiner Linux, com o mesmo código, a rota dá no
    LOESS 0.16451113535487225 contra os 0.16451113535487222 gravados, 1 ULP, e o SERRF igual.
    """
    corpo, esperado = palpite_local_e_gravado
    assert {c: corpo["escores"][c] for c in ("loess", "serrf")} == {
        c: esperado[c] for c in ("loess", "serrf")
    }
    assert corpo["correction_proposta"] == min(esperado, key=lambda c: esperado[c])


# ---------------------------------------------------------------------------------------------
# Medição que não aconteceu é 2, e não 1
# ---------------------------------------------------------------------------------------------
#
# Até a revisão, um corpo gravado truncado ou a rota daqui a falhar LEVANTAVAM: o processo saía 1,
# que é o código de DIVERGE, e o relatório nem era escrito. Os testes abaixo plantam cada uma das
# quatro falhas e afirmam NAO_MEDIU, com a frase de cada uma.


def _truncado(caminho: Path) -> Path:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text('{"correction_proposta": "ser', encoding="utf-8")
    return caminho


def _falha_aqui(_entrada: Path, _modo: str = MODO) -> dict[str, Any]:
    raise RuntimeError("a rota local devolveu 422: com 1 lote(s), os QCs por lote são [2]")


def test_o_palpite_medido_truncado_nao_mediu_e_nao_mede_aqui(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    local = _local(_corpo_do_palpite())
    gravado = _truncado(tmp_path / "verifica-app" / "palpite-medido.json")
    relato = _confere_com_palpite(tmp_path, entrada, resposta_de_src, gravado, local)
    assert relato["veredito"] == "NAO_MEDIU", relato
    assert any("Palpite medido gravado não é JSON legível" in x for x in relato["nao_medido"])
    assert local.chamadas == []


def test_a_medicao_daqui_que_falha_nao_mediu(
    tmp_path: Path, entrada: Path, resposta_de_src: dict[str, Any]
) -> None:
    corpo = _corpo_do_palpite()
    relato = _confere_com_palpite(
        tmp_path, entrada, resposta_de_src, _palpite_gravado(tmp_path, corpo), _falha_aqui
    )
    assert relato["veredito"] == "NAO_MEDIU", relato
    assert any("Palpite medido não foi medido aqui" in x for x in relato["nao_medido"]), relato
    assert "campos_divergentes_do_palpite_medido" not in relato


def test_a_resposta_da_arvore_truncada_ou_sem_rota_aqui_nao_mediu(
    tmp_path: Path,
    entrada: Path,
    resposta_de_src: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mesma classe na terceira conferência, que já a tinha."""
    truncada = tmp_path / "truncada"
    relato = conferidor.confere(
        _pasta(truncada),
        _truncado(truncada / "refino-u10" / "fluxo-recomendar.json"),
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
    )
    assert relato["veredito"] == "NAO_MEDIU", relato
    assert any("resposta gravada da árvore não é JSON legível" in x for x in relato["nao_medido"])

    sem_rota = tmp_path / "sem-rota"
    monkeypatch.setattr(conferidor, "resposta_local", _falha_aqui)
    relato = conferidor.confere(
        _pasta(sem_rota),
        _gravada(sem_rota, resposta_de_src),
        entrada,
        commit_esperado=COMMIT,
        alteracoes=_limpo,
    )
    assert relato["veredito"] == "NAO_MEDIU", relato
    assert any("rota da árvore não respondeu aqui" in x for x in relato["nao_medido"]), relato


def test_main_com_o_corpo_truncado_sai_2_e_escreve_o_relatorio(
    tmp_path: Path,
    entrada: Path,
    resposta_de_src: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O desfecho que a revisão mediu: saía 1, sem relatório. Agora é 2, com ele."""
    monkeypatch.setattr(conferidor, "alteracoes_em_src", _limpo)
    monkeypatch.setattr(conferidor, "palpite_medido_local", _local(_corpo_do_palpite()))
    pasta = _pasta(tmp_path)
    relatorio = tmp_path / "verdicto.json"
    codigo = conferidor.main(
        [
            "--pasta",
            str(pasta),
            "--entrada",
            str(entrada),
            "--commit",
            COMMIT,
            "--resposta",
            str(_gravada(tmp_path, resposta_de_src)),
            "--palpite-medido",
            str(_truncado(tmp_path / "verifica-app" / "palpite-medido.json")),
            "--relatorio",
            str(relatorio),
        ]
    )
    assert codigo == conferidor.NAO_MEDIU
    assert json.loads(relatorio.read_text(encoding="utf-8"))["veredito"] == "NAO_MEDIU"

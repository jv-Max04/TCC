"""O lançador do R das vias, com duplos de processo — sem R instalado.

O duplo é um script Python no lugar do `Rscript`, pelo argumento `comando` do lançador: o resto da
receita — ambiente, Job Object, temporário, leitura e validação da resposta — é o mesmo da produção.
É o molde de `tests/test_metaboanalystr_degradation.py`, e o motivo é o mesmo: a fronteira de
processo é o que se testa, e um mock em processo não a atravessaria.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

from refino.io import enriquecimento_r as er

# Só os dois testes do NETO são do Windows, e o `skipif` está neles e não no módulo. Eles contam
# processo vivo por `OpenProcess` e dependem de o Job Object levar o neto junto; o resto do arquivo
# é a fronteira de processo e a de confiança — `_interpreta`, `ambiente_isolado`, teto e
# cancelamento — e não usa nada do Windows. O marcador de módulo que esteve aqui pulava os 21 casos
# fora do Windows com uma razão que só valia para dois, e uma regressão nos outros feita e
# conferida no Linux sairia como «skipped». Medido no WSL Ubuntu, sobre os 27 casos que o arquivo
# tinha então: 25 passam e 2 pulam; com a conferência do isolamento desligada por planta, o
# arquivo antigo dava 21 skipped e saída 0, e este dá 2 failed.


def _r_home(tmp_path: Path) -> Path:
    r_home = tmp_path / "R"
    (r_home / "library").mkdir(parents=True, exist_ok=True)
    return r_home


def _roteiro(tmp_path: Path, nome: str = "vias_entrypoint.R") -> Path:
    p = tmp_path / nome
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# roteiro de mentira; o duplo nao o le\n", encoding="utf-8")
    return p


def _duplo(tmp_path: Path, corpo: str, *, antes_de_ler: str = "") -> list[str]:
    """Um `Rscript` de mentira. `LIB` e o `.libPaths()` que ele declara, dentro da árvore.

    `antes_de_ler` corre antes de o duplo ler o `stdin`: é o arranque do R, que lê o pedido só
    depois de carregar o `jsonlite` (`le_pedido` em `r_service/vias_entrypoint.R`).
    """
    script = tmp_path / f"duplo_{abs(hash((antes_de_ler, corpo)))}.py"
    script.write_text(
        "import json, os, sys, time, subprocess\n"
        f"LIB = {str(tmp_path / 'R' / 'library')!r}\n"
        + textwrap.dedent(antes_de_ler)
        + "PEDIDO = json.loads(sys.stdin.buffer.read().decode('utf-8') or '{}')\n"
        "def responde(d):\n"
        "    d.setdefault('ambiente', {'libPaths': [LIB]})\n"
        # Bytes UTF-8, como o roteiro R escreve: a página de código da consola não entra no meio.
        "    sys.stdout.buffer.write((json.dumps(d, ensure_ascii=False) + '\\n').encode('utf-8'))\n"
        "    sys.stdout.flush()\n" + textwrap.dedent(corpo),
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


# O prefixo do temporário de cada corrida, copiado de `tempfile.mkdtemp(prefix=...)` no lançador. A
# cópia é à mão, e por isso as ausências sobre `_temporarios()` levam ao lado a âncora de que o
# temporário REAL da corrida é desta população: sem ela, um lançador que mudasse o prefixo ou a
# pasta deixava a ausência vácua. Com o prefixo trocado e a limpeza anulada, os dois testes
# passavam com dois diretórios vazados.
_PREFIXO_DO_TEMPORARIO = "refino-vias-"


def _temporarios() -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob(f"{_PREFIXO_DO_TEMPORARIO}*")}


def _grava_cwd(arquivo: Path) -> str:
    """Linhas do duplo que gravam o diretório de trabalho dele, que é o temporário da corrida: o
    lançador corre o filho com `cwd=raiz`. Grava num nome provisório e renomeia, para que quem vê o
    arquivo existir o leia inteiro."""
    parcial = f"{arquivo}.parcial"
    return (
        f"open({parcial!r}, 'w', encoding='utf-8').write(os.getcwd())\n"
        f"os.replace({parcial!r}, {str(arquivo)!r})\n"
    )


def _vivo(pid: int) -> bool:
    import ctypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        codigo = ctypes.c_ulong()
        k32.GetExitCodeProcess(h, ctypes.byref(codigo))
        return codigo.value == 259  # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def test_resposta_ok_chega_integra(tmp_path: Path) -> None:
    cmd = _duplo(
        tmp_path, "responde({'status': 'ok', 'eco': PEDIDO['lista'], 'acento': 'Ácido β'})\n"
    )
    r = er.corre_roteiro_de_vias(
        {"lista": ["Glicina", "ç"]},
        r_home=_r_home(tmp_path),
        roteiro=_roteiro(tmp_path),
        comando=cmd,
    )
    assert r.desfecho == "ok", r
    assert (
        r.dados is not None
        and r.dados["eco"] == ["Glicina", "ç"]
        and r.dados["acento"] == "Ácido β"
    )


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="só no Windows: o neto morre porque o Job Object o leva, e `_vivo` usa OpenProcess",
)
def test_cancelar_encerra_o_filho_e_o_neto_e_nao_sobra_temporario(tmp_path: Path) -> None:
    pid_do_neto = tmp_path / "neto.pid"
    cwd_do_filho = tmp_path / "filho.cwd"
    corpo = f"""
    neto = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
    open({str(pid_do_neto)!r}, 'w').write(str(neto.pid))
    time.sleep(120)
    """
    antes = _temporarios()
    inicio = time.monotonic()

    def cancela() -> bool:
        return pid_do_neto.exists() or time.monotonic() - inicio > 60

    r = er.corre_roteiro_de_vias(
        {},
        r_home=_r_home(tmp_path),
        roteiro=_roteiro(tmp_path),
        comando=_duplo(tmp_path, corpo, antes_de_ler=_grava_cwd(cwd_do_filho)),
        cancelado=cancela,
    )
    assert r.desfecho == "cancelado", r
    neto = int(pid_do_neto.read_text())
    for _ in range(50):
        if not _vivo(neto):
            break
        time.sleep(0.1)
    assert not _vivo(neto), "o neto sobreviveu ao cancelamento"
    # Âncora da ausência abaixo: o temporário em que o filho correu é da população que
    # `_temporarios()` olha — o prefixo e a pasta.
    cwd = Path(cwd_do_filho.read_text(encoding="utf-8"))
    assert cwd.name.startswith(_PREFIXO_DO_TEMPORARIO), f"âncora: {cwd} fora do prefixo"
    assert os.path.normcase(cwd.parent) == os.path.normcase(tempfile.gettempdir()), (
        f"âncora: {cwd} fora de {tempfile.gettempdir()}"
    )
    assert _temporarios() - antes == set(), "sobrou diretório temporário da corrida"


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="só no Windows: o par causal do Job Object conta o neto por OpenProcess e o mata "
    "com taskkill",
)
def test_par_causal_sem_o_job_o_neto_sobrevive_ao_kill_do_filho(tmp_path: Path) -> None:
    """Sem isto, o teste acima passaria também com um lançador que só matasse o filho."""
    pid_do_neto = tmp_path / "neto.pid"
    filho = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import subprocess, sys, time\n"
            "n = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            f"open({str(pid_do_neto)!r}, 'w').write(str(n.pid))\n"
            "time.sleep(120)",
        ],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        for _ in range(100):
            if pid_do_neto.exists() and pid_do_neto.read_text():
                break
            time.sleep(0.1)
        neto = int(pid_do_neto.read_text())
        filho.kill()
        filho.wait()
        time.sleep(0.5)
        assert _vivo(neto), "a planta não aterrou: o neto morreu sem Job"
    finally:
        subprocess.run(
            ["taskkill", "/F", "/PID", pid_do_neto.read_text()], capture_output=True, check=False
        )


def test_sem_a_arvore_e_motor_ausente_e_o_path_nao_e_consultado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    falso = tmp_path / "no_path"
    falso.mkdir()
    (falso / "Rscript.exe").write_bytes(b"MZ")
    monkeypatch.setenv("PATH", f"{falso}{os.pathsep}{os.environ.get('PATH', '')}")
    sem_r = tmp_path / "sem-R"
    r = er.corre_roteiro_de_vias({}, r_home=sem_r, roteiro=_roteiro(tmp_path))
    assert r.desfecho == "motor_ausente"
    assert str(er.rscript_da_arvore(sem_r)) in r.motivo


def test_codigo_de_saida_diferente_de_zero_e_motor_falhou_com_stderr(tmp_path: Path) -> None:
    cmd = _duplo(
        tmp_path,
        "sys.stderr.write('Error in library(MetaboAnalystR): pacote corrompido\\n'); sys.exit(3)\n",
    )
    r = er.corre_roteiro_de_vias(
        {}, r_home=_r_home(tmp_path), roteiro=_roteiro(tmp_path), comando=cmd
    )
    assert r.desfecho == "motor_falhou", r
    assert "pacote corrompido" in r.stderr


def test_duplo_que_nao_termina_e_tempo_esgotado_e_nada_sobra(tmp_path: Path) -> None:
    cwd_do_filho = tmp_path / "filho.cwd"
    antes = _temporarios()

    def espera_o_arranque() -> bool:
        # Não cancela: segura o laço até o duplo gravar o diretório de trabalho, para que a âncora
        # abaixo não dependa de o filho arrancar antes do teto de 1 s.
        limite = time.monotonic() + 30
        while not cwd_do_filho.exists() and time.monotonic() < limite:
            time.sleep(0.05)
        return False

    r = er.corre_roteiro_de_vias(
        {},
        r_home=_r_home(tmp_path),
        roteiro=_roteiro(tmp_path),
        comando=_duplo(tmp_path, "time.sleep(120)\n", antes_de_ler=_grava_cwd(cwd_do_filho)),
        tempo_limite_s=1.0,
        cancelado=espera_o_arranque,
    )
    assert r.desfecho == "tempo_esgotado", r
    # Âncora da ausência abaixo, a mesma do teste do cancelamento.
    cwd = Path(cwd_do_filho.read_text(encoding="utf-8"))
    assert cwd.name.startswith(_PREFIXO_DO_TEMPORARIO), f"âncora: {cwd} fora do prefixo"
    assert os.path.normcase(cwd.parent) == os.path.normcase(tempfile.gettempdir()), (
        f"âncora: {cwd} fora de {tempfile.gettempdir()}"
    )
    assert _temporarios() - antes == set(), "sobrou diretório temporário da corrida"


# Um pedido do tamanho de um fundo real (`pedido_mapear` leva o fundo inteiro), e bem acima do
# buffer de um tubo anônimo: é nesse tamanho que uma escrita por tubo esperava o filho ler.
_PEDIDO_GRANDE = {"fundo": [f"Ácido {i:05d} " + "x" * 90 for i in range(2000)]}
_ARRANQUE_S = 6.0


@pytest.mark.parametrize(
    ("cancela", "teto", "esperado"),
    [(True, None, "cancelado"), (False, 0.5, "tempo_esgotado")],
    ids=["cancelamento", "teto"],
)
def test_cancelamento_e_teto_valem_durante_o_arranque_com_pedido_grande(
    tmp_path: Path, cancela: bool, teto: float | None, esperado: str
) -> None:
    """O R lê o pedido só depois de arrancar. Com o `stdin` por tubo, um pedido maior que o buffer
    prendia o lançador na escrita até o filho ler, e o cancelamento e o teto — que vivem no laço
    DEPOIS da escrita — ficavam mudos durante o arranque inteiro (revisão de 01/10/2026).
    """
    tamanho = len(json.dumps(_PEDIDO_GRANDE, ensure_ascii=False).encode("utf-8"))
    # A planta tem de cair onde tubo e arquivo diferem: muito acima do buffer de um tubo.
    assert tamanho > 100_000, tamanho
    r = er.corre_roteiro_de_vias(
        _PEDIDO_GRANDE,
        r_home=_r_home(tmp_path),
        roteiro=_roteiro(tmp_path),
        comando=_duplo(
            tmp_path, "responde({'status': 'ok'})\n", antes_de_ler=f"time.sleep({_ARRANQUE_S})\n"
        ),
        cancelado=lambda: cancela,
        tempo_limite_s=teto,
    )
    assert r.desfecho == esperado, r
    assert r.segundos < _ARRANQUE_S / 2, r


def test_pedido_grande_chega_inteiro_pelo_stdin(tmp_path: Path) -> None:
    """O `stdin` de arquivo entrega o pedido inteiro, com acento, a um filho que lê depois de
    arrancar — o contrato de `le_pedido` no roteiro, que lê até o fim do `stdin`."""
    r = er.corre_roteiro_de_vias(
        _PEDIDO_GRANDE,
        r_home=_r_home(tmp_path),
        roteiro=_roteiro(tmp_path),
        comando=_duplo(
            tmp_path,
            "F = PEDIDO['fundo']\nresponde({'status': 'ok', 'n': len(F), 'ultimo': F[-1]})\n",
            antes_de_ler="time.sleep(0.5)\n",
        ),
    )
    assert r.desfecho == "ok", r
    assert r.dados is not None
    assert (r.dados["n"], r.dados["ultimo"]) == (2000, _PEDIDO_GRANDE["fundo"][-1])


@pytest.mark.parametrize(
    "corpo",
    [
        "print('Loading required package: qs2')\n",
        "responde({'status': 'talvez'})\n",
        "responde({'status': 'erro', 'desfecho': 'desfecho_inventado', 'motivo': 'x'})\n",
        "print(json.dumps([1, 2, 3]))\n",
    ],
    ids=["nao-json", "status-desconhecido", "desfecho-desconhecido", "nao-objeto"],
)
def test_resposta_fora_do_contrato_e_malformada(tmp_path: Path, corpo: str) -> None:
    r = er.corre_roteiro_de_vias(
        {}, r_home=_r_home(tmp_path), roteiro=_roteiro(tmp_path), comando=_duplo(tmp_path, corpo)
    )
    assert r.desfecho == "resposta_malformada", r


def test_recusa_valida_chega_tipada(tmp_path: Path) -> None:
    cmd = _duplo(
        tmp_path,
        "responde({'status': 'erro', 'desfecho': 'fundo_recusado', 'motivo': 'mais da metade'})\n"
        "sys.exit(1)\n",
    )
    r = er.corre_roteiro_de_vias(
        {}, r_home=_r_home(tmp_path), roteiro=_roteiro(tmp_path), comando=cmd
    )
    assert (r.desfecho, r.recusa, r.motivo) == ("recusado", "fundo_recusado", "mais da metade")


def test_libpaths_fora_da_arvore_e_motor_falhou_por_isolamento(tmp_path: Path) -> None:
    cmd = _duplo(
        tmp_path,
        "FORA = 'C:/Users/fulano/AppData/Local/R/win-library/4.4'\n"
        "responde({'status': 'ok', 'ambiente': {'libPaths': [LIB, FORA]}})\n",
    )
    r = er.corre_roteiro_de_vias(
        {}, r_home=_r_home(tmp_path), roteiro=_roteiro(tmp_path), comando=cmd
    )
    assert r.desfecho == "motor_falhou" and "isolamento" in r.motivo, r
    # par causal: só com a da árvore, passa
    ok = er.corre_roteiro_de_vias(
        {},
        r_home=_r_home(tmp_path),
        roteiro=_roteiro(tmp_path),
        comando=_duplo(tmp_path, "responde({'status': 'ok'})\n"),
    )
    assert ok.desfecho == "ok"


def test_libpaths_em_texto_como_o_roteiro_real_emite(tmp_path: Path) -> None:
    """A forma de PRODUÇÃO: o roteiro serializa com `auto_unbox = TRUE`, e um `.libPaths()` de um
    caminho só chega como TEXTO, não como lista. Os outros testes alimentam lista, e um lançador
    que só aceitasse lista passaria por eles e
    declararia «isolamento» em toda chamada real.

    Par causal no mesmo teste: o mesmo texto, fora da árvore, continua a ser recusado — o ramo do
    texto não é um atalho que aceita qualquer caminho.
    """
    dentro = er.corre_roteiro_de_vias(
        {},
        r_home=_r_home(tmp_path),
        roteiro=_roteiro(tmp_path),
        comando=_duplo(tmp_path, "responde({'status': 'ok', 'ambiente': {'libPaths': LIB}})\n"),
    )
    assert dentro.desfecho == "ok", dentro
    fora = er.corre_roteiro_de_vias(
        {},
        r_home=_r_home(tmp_path),
        roteiro=_roteiro(tmp_path),
        comando=_duplo(
            tmp_path,
            "responde({'status': 'ok', 'ambiente': "
            "{'libPaths': 'C:/Users/fulano/AppData/Local/R/win-library/4.4'}})\n",
        ),
    )
    assert fora.desfecho == "motor_falhou" and "isolamento" in fora.motivo, fora


def test_o_ambiente_do_filho_e_o_da_receita(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k, v in {
        "R_HOME": "C:/outro/R",
        "R_LIBS": "C:/libs",
        "NO_PROXY": "*",
        "ALL_PROXY": "socks5://x",
    }.items():
        monkeypatch.setenv(k, v)
    cmd = _duplo(
        tmp_path, "responde({'status': 'ok', 'env': dict(os.environ), 'cwd': os.getcwd()})\n"
    )
    r = er.corre_roteiro_de_vias(
        {}, r_home=_r_home(tmp_path), roteiro=_roteiro(tmp_path), comando=cmd
    )
    assert r.desfecho == "ok", r
    assert r.dados is not None
    env = {k.upper(): v for k, v in r.dados["env"].items()}  # type: ignore[union-attr]
    assert not {"R_HOME", "R_LIBS", "NO_PROXY", "ALL_PROXY"} & set(env)
    alvos = {env[k] for k in ("R_LIBS_USER", "R_LIBS_SITE", "R_ENVIRON_USER", "R_PROFILE_USER")}
    assert len(alvos) == 1
    (alvo,) = alvos
    assert Path(alvo).is_absolute() and not Path(alvo).exists()
    assert Path(alvo).parent == Path(str(r.dados["cwd"]))
    assert env["HTTPS_PROXY"] == er.PROXY_MORTO


def test_caminho_com_e_comercial_e_porcento_passa_intacto(tmp_path: Path) -> None:
    roteiro = _roteiro(tmp_path / "pasta a&b %PATH%", "vias_entrypoint.R")
    cmd = _duplo(tmp_path, "responde({'status': 'ok', 'argv': sys.argv[-1]})\n")
    r = er.corre_roteiro_de_vias({}, r_home=_r_home(tmp_path), roteiro=roteiro, comando=cmd)
    assert r.desfecho == "ok", r
    assert r.dados is not None and r.dados["argv"] == str(roteiro)


def test_o_ambiente_isolado_e_a_fonte_unica_da_montagem() -> None:
    """A montagem (`scripts/r_portatil.py`) e o lançador usam a mesma receita."""
    texto = (Path(__file__).resolve().parent.parent / "scripts" / "r_portatil.py").read_text(
        encoding="utf-8"
    )
    assert "from refino.io.enriquecimento_r import" in texto
    assert "def ambiente_isolado(" not in texto
    env = er.ambiente_isolado(Path("C:/t"), base={"R_X": "1", "PATH": "p", "no_proxy": "x"})
    assert env["PATH"] == "p" and "R_X" not in env and "no_proxy" not in env
    assert env["R_LIBS_USER"] == str(Path("C:/t") / "nao-existe")


def test_o_temporario_da_sessao_do_r_nasce_na_pasta_da_corrida() -> None:
    """Cancelar mata o R antes de ele apagar o `Rtmp*`; na pasta da corrida, o lançador o apaga."""
    env = er.ambiente_isolado(Path("C:/t"), base={"TEMP": "C:/comum", "TMP": "C:/comum"})
    assert {env[k] for k in ("TMPDIR", "TMP", "TEMP")} == {str(Path("C:/t"))}


def test_sem_teto_a_chamada_lenta_termina_e_com_teto_ela_esgota(tmp_path: Path) -> None:
    """Par causal do teto opcional: o mesmo duplo de 1,5 s, com e sem `tempo_limite_s`.

    `None` é a primeira chamada de cada processo (resolução 12): ela só para pelo cancelamento.
    """
    cmd = _duplo(tmp_path, "time.sleep(1.5)\nresponde({'status': 'ok'})\n")
    comum = {"r_home": _r_home(tmp_path), "roteiro": _roteiro(tmp_path), "comando": cmd}
    sem_teto = er.corre_roteiro_de_vias({}, tempo_limite_s=None, **comum)  # type: ignore[arg-type]
    assert sem_teto.desfecho == "ok", sem_teto
    com_teto = er.corre_roteiro_de_vias({}, tempo_limite_s=0.5, **comum)  # type: ignore[arg-type]
    assert com_teto.desfecho == "tempo_esgotado", com_teto


# ---------------------------------------------------------------------------
# Os ramos de classificação que nenhum duplo alcançava
# ---------------------------------------------------------------------------
#
# Medido com `--cov-branch` antes destes testes: seis ramos do lançador sem execução. Cada teste
# afirma o desfecho E o motivo, porque o desfecho sozinho não diz qual ramo o produziu — quatro
# ramos de `_interpreta` e o `except OSError` de `corre_roteiro_de_vias` devolvem `motor_falhou`.


def test_saida_zero_sem_resposta_e_malformada(tmp_path: Path) -> None:
    """O `else` de `codigo != 0` com o `stdout` vazio: o R saiu bem e não respondeu. O irmão com
    saída diferente de zero é `test_codigo_de_saida_diferente_de_zero_e_motor_falhou_com_stderr`.
    """
    r = er.corre_roteiro_de_vias(
        {}, r_home=_r_home(tmp_path), roteiro=_roteiro(tmp_path), comando=_duplo(tmp_path, "pass\n")
    )
    assert (r.desfecho, r.motivo) == ("resposta_malformada", "o R não escreveu resposta"), r


def test_motor_falhou_declarado_pelo_r_chega_como_motor_falhou_e_nao_como_recusa(
    tmp_path: Path,
) -> None:
    """É o destino das chamadas `recusar("motor_falhou", …)` de `r_service/vias_entrypoint.R`
    (nove em 01/10/2026): um R que arrancou e não conseguiu calcular não é recusa do pedido."""
    cmd = _duplo(
        tmp_path,
        "responde({'status': 'erro', 'desfecho': 'motor_falhou', "
        "'motivo': 'library(MetaboAnalystR) falhou: x'})\n"
        "sys.exit(1)\n",
    )
    r = er.corre_roteiro_de_vias(
        {}, r_home=_r_home(tmp_path), roteiro=_roteiro(tmp_path), comando=cmd
    )
    assert (r.desfecho, r.motivo, r.recusa) == (
        "motor_falhou",
        "library(MetaboAnalystR) falhou: x",
        None,
    ), r


def test_status_ok_com_saida_diferente_de_zero_e_motor_falhou(tmp_path: Path) -> None:
    cmd = _duplo(tmp_path, "responde({'status': 'ok', 'n': 1})\nsys.exit(2)\n")
    r = er.corre_roteiro_de_vias(
        {}, r_home=_r_home(tmp_path), roteiro=_roteiro(tmp_path), comando=cmd
    )
    assert (r.desfecho, r.motivo, r.dados) == (
        "motor_falhou",
        "o R respondeu ok e saiu com 2",
        None,
    ), r


def test_roteiro_ausente_e_motor_ausente_mesmo_com_o_comando_presente(tmp_path: Path) -> None:
    """O duplo responderia `ok` se fosse lançado: sem a guarda do roteiro, o desfecho muda."""
    roteiro = tmp_path / "nao-ha" / "vias_entrypoint.R"
    r = er.corre_roteiro_de_vias(
        {},
        r_home=_r_home(tmp_path),
        roteiro=roteiro,
        comando=_duplo(tmp_path, "responde({'status': 'ok'})\n"),
    )
    assert (r.desfecho, r.motivo) == (
        "motor_ausente",
        f"o roteiro das vias não está em {roteiro}",
    ), r


def test_executavel_que_nao_existe_e_motor_falhou_e_nao_levanta(tmp_path: Path) -> None:
    """O `OSError` do `Popen` é o que sustenta «nunca levanta por indisponibilidade». O texto do
    erro vem do sistema e da língua dele, então o motivo é conferido pelo prefixo do lançador."""
    r = er.corre_roteiro_de_vias(
        {},
        r_home=_r_home(tmp_path),
        roteiro=_roteiro(tmp_path),
        comando=[str(tmp_path / "nao-ha" / "Rscript.exe")],
    )
    assert r.desfecho == "motor_falhou", r
    assert r.motivo.startswith("o R não pôde ser lançado: "), r


def test_job_que_recusa_o_filho_o_mata_e_e_motor_falhou(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O ramo de `AssignProcessToJobObject` falhar: o filho já nasceu, fora do Job, e o «matar ao
    fechar» não o alcançaria — o lançador o mata antes de devolver.

    Não achei forma limpa de provocar a falha real do Windows, então o duplo é do Job e não do
    processo: uma subclasse cujo `adota` levanta como o de produção levanta. `termina` e `fecha`
    continuam os de produção, e o filho é um processo de verdade.
    """
    nascidos: list[subprocess.Popen[bytes]] = []

    class _JobQueRecusa(er._Job):
        def adota(self, processo: subprocess.Popen[bytes]) -> None:
            nascidos.append(processo)
            raise OSError(5, "AssignProcessToJobObject falhou")

    monkeypatch.setattr(er, "_Job", _JobQueRecusa)
    try:
        r = er.corre_roteiro_de_vias(
            {},
            r_home=_r_home(tmp_path),
            roteiro=_roteiro(tmp_path),
            comando=_duplo(tmp_path, "time.sleep(120)\n"),
            tempo_limite_s=60.0,
        )
        assert r.desfecho == "motor_falhou", r
        assert r.motivo.startswith("o R não pôde ser lançado: "), r
        assert "AssignProcessToJobObject falhou" in r.motivo, r
        assert len(nascidos) == 1, "âncora: o filho nasceu e o Job o recusou"
        assert nascidos[0].poll() is not None, "o filho que o Job recusou ficou vivo"
    finally:
        for p in nascidos:
            if p.poll() is None:
                p.kill()
                p.wait()

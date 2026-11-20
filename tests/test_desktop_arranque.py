"""O arranque do aplicativo de mesa.

## O que estes testes alcançam

O ponto de entrada de verdade: banco de verdade em diretório temporário, servidor de verdade em
loopback, e a variável `REFINO_API_BASE` de verdade. O único duplo é a **janela**.

## O que eles NÃO alcançam, e a qualificação é a seção inteira

**A janela.** Nenhuma asserção aqui é sobre Qt. O que se afirma é que a **base chega** ao passo
seguinte, que é a travessia que o arranque existe para montar.

**O executável congelado.** `garante_utf8` é exercitado em subprocesso com a página de código
forçada, o que mede o **arranque**; que o binário `--onedir` respeite `--python-option X utf8=1` foi
medido à parte, e não é testado aqui.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from refino.api.app import build_app
from refino.desktop import arranque
from refino.desktop.arranque import (
    SAIU_BEM,
    SAIU_POR_AMBIENTE,
    VARIAVEL_DA_BASE,
    arranca,
    garante_utf8,
    main,
)
from refino.desktop.banco import (
    BancoDeMesaError,
    diretorio_de_dados,
)
from tests.permissao_capability import (
    DETALHE_DO_SEM_ESCRITA,
    permissao_de_diretorio_respeitada,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.integration

# Prosa acentuada que reprova sob `cp932`, que é a página de código medida nesta máquina Windows.
_ACENTOS = "Refino quimiométrico de metabolômica — imputação, normalização"


@pytest.fixture(autouse=True)
def ambiente_do_processo_limpo() -> Iterator[None]:
    """Restaura **as duas** variáveis que `arranca` escreve, em **todo** teste deste módulo.

    ## Por que `autouse`, e não duas fixtures pedidas por nome

    A primeira versão eram duas fixtures declaradas por `usefixtures`, e **vazaram**. Um conserto
    fez `arranca` escrever `REFINO_DB_URL`; dois testes que chamam `main` — logo `arranca` —
    tinham só a fixture da base. A variável sobreviveu à sessão e um teste que grava no banco
    passou a apontar para um SQLite temporário que já não existia.

    **Medido por par causal:** esse teste passa sozinho e reprova com este módulo antes dele.
    Sob a aleatorização do `make check`, a ordem decide — que é a forma intermitente, a pior.

    > **Pedir por nome é o conserto de LEMBRAR; `autouse` é o de MECANISMO.** Um terceiro teste que
    > chame `main` amanhã não pode vazar, e não é preciso que quem o escreva saiba disto.

    ## E por que isto passou por mim

    **Modificar um mecanismo invalida a evidência dele.** Eu acrescentei uma
    escrita em `os.environ` e não auditei quem a exercita — a evidência das duas fixtures era de
    antes de existir a segunda variável, e ela não se herda.
    """
    import os

    from refino.persistence.config import VARIAVEL_DA_URL

    nomes = (VARIAVEL_DA_BASE, VARIAVEL_DA_URL)
    antes = {nome: os.environ.get(nome) for nome in nomes}
    try:
        yield
    finally:
        for nome, valor in antes.items():
            if valor is None:
                os.environ.pop(nome, None)
            else:
                os.environ[nome] = valor


def _ambiente(raiz: Path) -> dict[str, str]:
    return {"XDG_DATA_HOME": str(raiz)}


# ═══════════════════════════════════════════════════════════ o UTF-8


def test_garante_utf8_devolve_o_que_reconfigurou() -> None:
    """**Asserção de PRESENÇA, e é por isso que o valor de retorno existe.**

    Uma asserção de *"não levantou"* é satisfeita por uma função vazia. O que separa
    *"reconfigurou"* de *"não fez nada"* é o nome do fluxo estar na lista devolvida.
    """
    feitos = garante_utf8()

    assert "stdout" in feitos or "stderr" in feitos, (
        f"nenhum fluxo foi reconfigurado: {feitos}. Sob `pytest` os fluxos são capturados, mas "
        "ainda expõem `reconfigure` — se isto falhar, a função deixou de fazer o que o nome diz"
    )


def test_garante_utf8_tolera_fluxo_ausente(monkeypatch: pytest.MonkeyPatch) -> None:
    """O ramo do `--windowed`, que dispara por AUSÊNCIA e por isso ficaria inerte.

    Um executável PyInstaller sem console tem `sys.stdout is None`. Uma reconfiguração que
    levantasse ali transformaria a forma sem terminal em falha de arranque — e é a forma que o
    laboratório vai usar.
    """
    monkeypatch.setattr(sys, "stdout", None)

    feitos = garante_utf8(["stdout"])

    assert feitos == (), f"não devia ter reconfigurado nada: {feitos}"


def test_garante_utf8_tolera_fluxo_que_recusa_reconfiguracao(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O terceiro desfecho: o fluxo existe, tem `reconfigure`, e ele levanta.

    Acontece com saída redirecionada para certos canos. **Não é falha de arranque**, e a asserção é
    que a função devolve lista vazia em vez de propagar.
    """

    class FluxoTeimoso:
        def reconfigure(self, **_: object) -> None:
            raise ValueError("este cano não aceita reconfiguração")

    monkeypatch.setattr(sys, "stdout", FluxoTeimoso())

    assert garante_utf8(["stdout"]) == ()


def test_a_prosa_acentuada_sobrevive_a_pagina_de_codigo_hostil(tmp_path: Path) -> None:
    """**Par causal em SUBPROCESSO: duas execuções idênticas que diferem só pela chamada.**

    Sob `PYTHONIOENCODING=cp932` — a página de código medida nesta máquina — imprimir prosa
    acentuada reprova com `UnicodeEncodeError`. Com `garante_utf8()` antes, sai.

    É negativo **externo**, então a metade que importa é ver o teste **deixar de acusar** quando a
    planta sai: por isso as duas execuções estão aqui, e não em testes separados.
    """
    corpo = textwrap.dedent(f"""
        import sys
        {{chamada}}
        print({_ACENTOS!r})
    """)

    sem = tmp_path / "sem.py"
    sem.write_text(corpo.format(chamada=""), encoding="utf-8")
    com = tmp_path / "com.py"
    com.write_text(
        corpo.format(chamada="from refino.desktop.arranque import garante_utf8; garante_utf8()"),
        encoding="utf-8",
    )

    ambiente = {"PYTHONIOENCODING": "cp932", "PYTHONPATH": "src"}
    from tests.conftest import REPO

    def roda(script: Path) -> subprocess.CompletedProcess[str]:
        import os

        env = {**os.environ, **ambiente}
        return subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=env,
        )

    resultado_sem = roda(sem)
    resultado_com = roda(com)

    assert resultado_sem.returncode != 0, (
        "**a planta não separou**: sob `cp932` a impressão acentuada passou SEM `garante_utf8`, "
        "então este par não mede nada. Talvez o `PYTHONIOENCODING` não esteja a valer.\n"
        f"stdout={resultado_sem.stdout!r} stderr={resultado_sem.stderr!r}"
    )
    assert "UnicodeEncodeError" in resultado_sem.stderr, (
        f"reprovou por outro motivo que não a página de código:\n{resultado_sem.stderr}"
    )
    assert resultado_com.returncode == 0, (
        f"com `garante_utf8` ainda reprovou:\n{resultado_com.stderr}"
    )
    assert "quimiométrico" in resultado_com.stdout, (
        f"a prosa saiu, mas sem os acentos: {resultado_com.stdout!r}"
    )


# ═══════════════════════════════════════════════════════════ a ordem, e o que ela protege


def test_a_falha_do_banco_nao_chega_a_subir_o_servidor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A ordem é o conteúdo, e este é o teste que a afirma.**

    Um espião no lugar de `sobe` conta as invocações. Se o banco falhar e o contador subir, a ordem
    inverteu — e o sintoma em produção seria a janela a abrir contra uma API cuja primeira gravação
    falha, longe da causa.
    """
    if not permissao_de_diretorio_respeitada():
        pytest.skip(f"permissão POSIX ignorada: {DETALHE_DO_SEM_ESCRITA}")

    chamadas: list[object] = []
    monkeypatch.setattr(arranque, "sobe", lambda *a, **k: chamadas.append((a, k)))

    trancado = tmp_path / "trancado"
    trancado.mkdir(mode=0o500)
    try:
        with pytest.raises(BancoDeMesaError):
            arranca(_ambiente(trancado), plataforma="linux")
    finally:
        trancado.chmod(0o700)

    assert chamadas == [], (
        f"o servidor foi iniciado {len(chamadas)} vez(es) apesar de o banco ter falhado — a ordem "
        "do arranque inverteu"
    )


def test_arranca_exporta_as_duas_variaveis_e_a_api_alcanca_o_banco(tmp_path: Path) -> None:
    """**O teste que não existia, e a ausência dele foi o que deixou passar um P0.**

    A versão anterior parava em `base_da_api()`: afirmava a travessia do **servidor** e não a do
    **banco**. Quatro revisores independentes acharam o mesmo defeito — `arranca` criava e migrava
    o SQLite por usuário e o processo ficava sem `REFINO_DB_URL`, porque `migra` a restaura no
    `finally`. Toda rota que grava respondia `503`.

    **As duas travessias, no mesmo teste, e a segunda é a que faltava:**

    | Variável | Quem a lê | O que a asserção prova |
    |---|---|---|
    | `REFINO_API_BASE` | `interface/cliente.py::base_da_api` | a janela fala com este servidor |
    | `REFINO_DB_URL` | `_repositorio`, **por pedido** | a API grava **neste** arquivo |

    E a última asserção é a que separa *"a variável está posta"* de *"a API consegue usá-la"*:
    `_repositorio()` é exactamente a função que respondia `503`, e construí-la é o desfecho que
    importa.
    """
    import os

    from refino.persistence.config import VARIAVEL_DA_URL, url_do_banco

    servidor, banco = arranca(_ambiente(tmp_path), plataforma="linux")
    try:
        # ── a travessia do servidor ──
        assert servidor.base.startswith("http://127.0.0.1:"), servidor.base
        assert os.environ[VARIAVEL_DA_BASE] == servidor.base, (
            "a base devolvida e a exportada divergem — o cliente falaria com outra porta"
        )
        assert servidor.esta_vivo()

        from refino.interface.cliente import base_da_api

        assert base_da_api() == servidor.base, (
            "`base_da_api()` não vê a base do arranque: a travessia entre a U1 e o cliente quebrou"
        )

        # ── a travessia do banco, que é a que faltava ──
        assert os.environ.get(VARIAVEL_DA_URL) == banco.url, (
            "`REFINO_DB_URL` não aponta para o banco que o arranque preparou. É o P0: `migra` a "
            "restaura no `finally`, e sem esta exportação toda rota que grava responde 503"
        )
        assert url_do_banco() == banco.url, (
            "`url_do_banco()` — a função que `_repositorio` chama a cada pedido — não devolve "
            f"{banco.url}"
        )

        from refino.api.rotas import _repositorio

        repo = _repositorio()
        assert repo is not None, (
            "`_repositorio()` não construiu: é exactamente a chamada que respondia 503 ao "
            "laboratório, a mandar subir um `docker compose` que o produto não tem"
        )
    finally:
        servidor.desliga()


# ═══════════════════════════════════════════════════════════ main, e os seus desfechos


def test_main_devolve_zero_e_entrega_a_base_a_janela(tmp_path: Path) -> None:
    """O desfecho de sucesso, com a janela substituída por um espião.

    O espião recebe a base — é a única coisa que o contrato entre o arranque e a janela promete.
    """
    recebidas: list[str] = []

    def janela(base: str) -> int:
        recebidas.append(base)
        return SAIU_BEM

    codigo = main(ambiente=_ambiente(tmp_path), plataforma="linux", abre_janela=janela)

    assert codigo == SAIU_BEM
    assert len(recebidas) == 1, f"a janela foi chamada {len(recebidas)} vezes"
    assert recebidas[0].startswith("http://127.0.0.1:"), recebidas[0]


def test_main_desliga_o_servidor_mesmo_quando_a_janela_levanta(tmp_path: Path) -> None:
    """O `finally`, e ele dispara por caminho de exceção — o ramo que ficaria inerte.

    Um servidor que sobrevive à janela deixa a porta presa até alguém matar o processo. Num
    executável de mesa isso é o usuário a descobrir que a segunda abertura não funciona.

    > **O DESFECHO mudou em 31/08/2026 e a PROPRIEDADE não.** Este teste envolvia a
    > chamada num `pytest.raises(RuntimeError)`, porque a exceção da janela **atravessava** `main`
    > — e era esse o defeito: num `--onedir --windowed` ela matava o processo sem mensagem. Agora
    > `main` a traduz em `SAIU_POR_AMBIENTE` com causa declarada.
    >
    > O que este teste guarda continua a ser o `finally`: **o servidor desliga**, e desliga pelo
    > caminho de exceção. Trocar `pytest.raises` pela asserção do código de saída preserva o ramo
    > que ele exercita; tirá-lo deixaria o `finally` sem quem o percorresse por exceção.
    """
    vistos: list[object] = []
    original = arranque.sobe

    def espia(*a: object, **k: object) -> object:
        embutido = original(*a, **k)  # type: ignore[arg-type]
        vistos.append(embutido)
        return embutido

    arranque.sobe = espia  # type: ignore[assignment]
    try:
        codigo = main(
            ambiente=_ambiente(tmp_path),
            plataforma="linux",
            abre_janela=lambda _: (_ for _ in ()).throw(RuntimeError("a janela explodiu")),
        )
    finally:
        arranque.sobe = original  # type: ignore[assignment]

    assert codigo == SAIU_POR_AMBIENTE, (
        f"a exceção da janela deu código {codigo}; ela devia virar desfecho com causa declarada"
    )

    assert len(vistos) == 1, "o servidor não foi iniciado, e o teste não mede nada"
    assert not vistos[0].esta_vivo(), (  # type: ignore[attr-defined]
        "o servidor continua vivo depois de a janela levantar — a porta fica presa"
    )


def test_main_traduz_falha_de_banco_em_codigo_de_saida_e_causa(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """O desfecho de falha: código não-zero, e a causa NOMEADA na mensagem.

    Cada desfecho declara de quem é a causa. Aqui a causa é `ambiente`
    — quem clicou não escolheu permissão de diretório —, e a asserção é que a palavra aparece.
    """
    if not permissao_de_diretorio_respeitada():
        pytest.skip(f"permissão POSIX ignorada: {DETALHE_DO_SEM_ESCRITA}")

    trancado = tmp_path / "trancado"
    trancado.mkdir(mode=0o500)
    try:
        codigo = main(ambiente=_ambiente(trancado), plataforma="linux", abre_janela=lambda _: 99)
    finally:
        trancado.chmod(0o700)

    assert codigo == SAIU_POR_AMBIENTE, (
        f"o código de saída foi {codigo}; a janela não devia nem ter sido chamada — ela "
        "devolveria 99"
    )
    capturado = capsys.readouterr()
    assert "ambiente" in capturado.err, (
        f"a causa não foi declarada na saída de erro: {capturado.err!r}"
    )
    assert str(trancado) in capturado.err, (
        f"a mensagem não nomeia o caminho, e sem ele não é acionável: {capturado.err!r}"
    )


def test_relata_nao_levanta_quando_nao_ha_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """O ramo do `--windowed` na saída de erro, e ele dispara por AUSÊNCIA.

    Num executável sem console não há onde escrever. Levantar ali trocaria uma mensagem perdida por
    um traceback — que é o desfecho que este módulo existe para não produzir.
    """
    monkeypatch.setattr(sys, "stderr", None)

    arranque._relata("qualquer coisa", causa="ambiente")  # não levanta, e é isso que se afirma


def test_a_janela_recusa_abrir_quando_a_base_do_cliente_nao_e_a_do_servidor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A guarda de costura entre o arranque e a janela, na direção que ACUSA.

    `constroi_transporte` monta o cliente com `base_da_api()`, que lê o ambiente — não o
    argumento. Se a exportação não chegou ao processo, a janela abriria contra a porta padrão
    8000, e o sintoma seria `503` em toda rota.

    O espião é o par causal: aqui ele **não** pode ser chamado.
    """
    chamado: list[int] = []
    monkeypatch.setattr(arranque, "_corre_a_janela", lambda: chamado.append(1) or 0)
    monkeypatch.setenv(arranque.VARIAVEL_DA_BASE, "http://127.0.0.1:8000")

    codigo = arranque.abre_a_janela("http://127.0.0.1:54321")

    assert codigo == SAIU_POR_AMBIENTE
    assert chamado == [], "a janela abriu apesar de a base divergir"
    erro = capsys.readouterr().err
    assert "127.0.0.1:8000" in erro and "127.0.0.1:54321" in erro, (
        f"a mensagem não nomeia as DUAS bases, e sem as duas não se sabe qual está errada: {erro!r}"
    )
    assert "ambiente" in erro


def test_a_janela_abre_quando_as_duas_bases_coincidem(monkeypatch: pytest.MonkeyPatch) -> None:
    """A outra direção da mesma guarda, e é a que ficaria inerte.

    **Ver o portão reprovar não é vê-lo deixar de acusar.** Sem este teste, uma guarda que
    recusasse *sempre* — um `!=` trocado por `==`, um `strip` a mais — passaria pelo teste acima e o
    aplicativo nunca abriria janela nenhuma, com mensagem de causa `ambiente` perfeitamente formada.
    """
    chamado: list[str] = []
    monkeypatch.setattr(arranque, "_corre_a_janela", lambda: chamado.append("abriu") or 7)
    monkeypatch.setenv(arranque.VARIAVEL_DA_BASE, "http://127.0.0.1:54321")

    codigo = arranque.abre_a_janela("http://127.0.0.1:54321")

    assert chamado == ["abriu"], "a guarda recusou uma base que coincide"
    assert codigo == 7, "o código de saída do laço de eventos não foi repassado"


def test_main_usa_a_janela_de_verdade_por_padrao(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A ligação, afirmada em vez de suposta.

    O andaime `_janela_por_construir` foi **removido**, em vez de afrouxado. Este teste é o que
    impede a remoção de deixar `main` a chamar outra coisa: sem ele, `abre_a_janela` poderia existir
    e nunca ser usada.
    """
    vistas: list[str] = []
    monkeypatch.setattr(arranque, "abre_a_janela", lambda base: vistas.append(base) or 0)

    codigo = arranque.main(ambiente=_ambiente(tmp_path), plataforma="linux")

    assert codigo == SAIU_BEM
    assert len(vistas) == 1, f"`main` não chamou `abre_a_janela` exactamente uma vez: {vistas}"
    assert vistas[0].startswith("http://127.0.0.1:"), (
        f"a base passada à janela não é loopback: {vistas[0]!r}"
    )


def test_garante_utf8_tolera_fluxo_que_levanta_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """A outra metade do par `(OSError, ValueError)`, que estava inerte.

    A revisão notou que só o `ValueError` era exercitado. Um `except` com dois tipos e um exercitado
    é meio mecanismo: o ramo não testado pode ter deixado de ser alcançável sem nada acusar.
    """

    class FluxoQueRecusa:
        def reconfigure(self, **_: object) -> None:
            raise OSError("descritor inválido")

    monkeypatch.setattr(sys, "stdout", FluxoQueRecusa())

    assert garante_utf8(["stdout"]) == ()


def test_uma_migracao_que_falha_vira_causa_de_ambiente_e_nao_traceback(tmp_path: Path) -> None:
    """**A falha de migração traduzida, e a mensagem que NÃO pode carregar a URL.**

    Três revisores independentes acharam que `migra` ficava fora do `try` de `prepara`: alembic a
    levantar — banco corrompido, revisão desconhecida, arquivo trancado — escapava da tupla de dois
    tipos de `main` e saía como traceback. Num `--onedir` sem console isso é queda muda.

    **A planta é um arquivo com bytes que não são SQLite**, e a asserção tem três metades: o tipo
    traduzido, a causa declarada, e a URL **ausente** da mensagem — porque mensagens de banco
    carregam a URL, e a URL pode carregar a senha.
    """
    raiz = tmp_path / "dados"
    (raiz / "refino").mkdir(parents=True)
    alvo = raiz / "refino" / "refino.db"
    alvo.write_bytes(b"isto nao e um banco SQLite, e o alembic vai reclamar")

    with pytest.raises(BancoDeMesaError, match="não pôde ser migrado") as capturado:
        arranca(_ambiente(raiz), plataforma="linux")

    mensagem = str(capturado.value)
    assert str(alvo) in mensagem, f"a mensagem não nomeia o caminho: {mensagem}"
    assert "sqlite:///" not in mensagem, (
        f"a mensagem carrega a URL do banco, e a de Postgres carregaria a senha: {mensagem}"
    )

    codigo = main(ambiente=_ambiente(raiz), plataforma="linux", abre_janela=lambda _: 99)
    assert codigo == SAIU_POR_AMBIENTE, (
        f"main devolveu {codigo}: a falha de migração não foi traduzida em código de saída"
    )


def test_uma_falha_nao_prevista_no_arranque_ainda_declara_a_causa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rede de segurança de `main`, e ela dispara por AUSÊNCIA de tipo conhecido.

    **O ramo que a revisão pediu**, e ele é o `else` da tradução: qualquer exceção que esta camada
    não saiba nomear tinha de sair como traceback. A asserção é que a causa aparece e que a
    mensagem **não** carrega o texto do erro.
    """

    class DefeitoInesperadoError(RuntimeError):
        pass

    # **Token que SÓ pode vir da planta.** A primeira versão procurava a palavra `"senha"`, e ela
    # aparece na **minha própria** mensagem explicativa — *"e ela carrega a senha"*. A agulha casava
    # o texto do conserto em vez do segredo plantado, e reprovou por isso. É a terceira agulha desta
    # sessão a medir a coisa errada.
    credencial = "postgresql://u:XYZZY-SEGREDO-PLANTADO@host/base"

    def arranca_que_explode(*_a: object, **_k: object) -> object:
        raise DefeitoInesperadoError(credencial)

    monkeypatch.setattr(arranque, "arranca", arranca_que_explode)

    codigo = main(ambiente=_ambiente(tmp_path), plataforma="linux", abre_janela=lambda _: 99)

    assert codigo == SAIU_POR_AMBIENTE
    erro = capsys.readouterr().err
    assert "ambiente" in erro, f"a causa não foi declarada: {erro!r}"
    assert "DefeitoInesperadoError" in erro, f"o tipo não foi nomeado: {erro!r}"
    assert "XYZZY-SEGREDO-PLANTADO" not in erro, (
        f"o texto do erro entrou na mensagem, e ele pode carregar credencial: {erro!r}"
    )


# ---------------------------------------------------------------------------
# OS FLUXOS SEM CONSOLA: o aplicativo nao abria por duplo clique
# ---------------------------------------------------------------------------
#
# **Medido em 23/08/2026 sobre o binario congelado**, quatro caminhos de lancamento:
#
# | Como se lanca | Desfecho |
# |---|---|
# | com `stdout` **ou** `stderr` redirecionado, ate para `NUL` | janela em 3 s |
# | sem handle nenhum — duplo clique, `explorer.exe`, `cmd /c start` | **morre em 2 s, codigo 1** |
#
# Nao era o diretorio de trabalho: com redirecao funciona de qualquer pasta, sem redirecao falha de
# qualquer pasta. A causa, reproduzida na venv do alvo:
#
#     uvicorn/logging.py:42  ->  self.use_colors = sys.stdout.isatty()
#     AttributeError: 'NoneType' object has no attribute 'isatty'
#     ValueError: Unable to configure formatter 'default'
#
# **E nenhum portao pegou porque TODOS os cenarios de `congela_windows.ps1` lancam com
# `-RedirectStandardOutput`/`-RedirectStandardError`** — o arnes media o binario na unica condicao
# em que o defeito nao pode aparecer.


class _SemFluxos:
    """Poe `sys.stdout` e `sys.stderr` em `None` e restaura ao sair. **Restaura no `finally`.**

    Sem a restauracao, um teste que reprove no meio deixa a sessao inteira sem saida — e o `pytest`
    passa a nao conseguir escrever o proprio relatorio. Medido: e um modo de falha real deste
    padrao, e por isso o contexto existe em vez de duas atribuicoes soltas.
    """

    def __enter__(self) -> None:
        self._guardados = (sys.stdout, sys.stderr)
        sys.stdout = None  # type: ignore[assignment]
        sys.stderr = None  # type: ignore[assignment]

    def __exit__(self, *_: object) -> None:
        sys.stdout, sys.stderr = self._guardados


def test_u15_o_defeito_real_e_o_conserto_dele(tmp_path: Path) -> None:
    """**O par causal contra o DEFEITO, e nao contra o mecanismo.**

    Um `garante_fluxos` que ligasse os fluxos a um objeto sem `isatty` passaria em todo teste de
    mecanismo — o fluxo deixaria de ser `None`, o retorno estaria certo — e falharia aqui. Este
    teste afirma o que o laboratorio precisa: **o `uvicorn.Config` constroi**.

    A primeira metade e a ancora: sem ela, um `uvicorn` que um dia deixasse de chamar `isatty()`
    tornaria a segunda metade vacua sem nada acusar.
    """
    import uvicorn

    ambiente = {"LOCALAPPDATA": str(tmp_path), "APPDATA": str(tmp_path)}

    # ── ancora: SEM o conserto, a construcao levanta ──
    with _SemFluxos():
        try:
            uvicorn.Config(build_app, factory=True, host="127.0.0.1", port=0)
            levantou = None
        except Exception as erro:
            levantou = f"{type(erro).__name__}: {erro}"
    assert levantou is not None, (
        "ancora: `uvicorn.Config` NAO levantou com os fluxos em `None`. O defeito que esta unidade "
        "conserta deixou de existir, e a metade de baixo passou a ser vacua"
    )
    assert "formatter" in levantou or "isatty" in levantou, (
        f"levantou por outro motivo que o medido: {levantou}"
    )

    # ── COM o conserto, constroi ──
    with _SemFluxos():
        ligados = arranque.garante_fluxos(ambiente, plataforma="win32")
        try:
            uvicorn.Config(build_app, factory=True, host="127.0.0.1", port=0)
            falhou = None
        except Exception as erro:
            falhou = f"{type(erro).__name__}: {erro}"
    assert ligados == ("stdout", "stderr"), f"ligou {ligados}"
    assert falhou is None, f"com o conserto ainda levantou: {falhou}"


def test_u15_nao_toca_nos_fluxos_quando_eles_existem(tmp_path: Path) -> None:
    """A direcao do «deixa de acusar»: com console, a funcao nao faz nada e devolve vazio.

    **Sem esta, um `garante_fluxos` que substituisse os fluxos SEMPRE passaria no teste de cima** —
    e o `pytest` perderia a propria saida em toda sessao.
    """
    antes = (sys.stdout, sys.stderr)
    ligados = arranque.garante_fluxos(
        {"LOCALAPPDATA": str(tmp_path), "APPDATA": str(tmp_path)}, plataforma="win32"
    )
    assert ligados == ()
    assert (sys.stdout, sys.stderr) == antes, "trocou fluxos que ja existiam"


def test_u15_o_log_fica_ao_lado_do_banco_e_recebe_o_que_se_escreve(tmp_path: Path) -> None:
    """O arquivo e o mesmo diretorio por usuario onde o banco vive, e ele de facto RECEBE.

    **A asserção e sobre o conteudo e nao sobre a existencia**, porque um arquivo criado e nunca
    escrito e exactamente o que um `devnull` disfarcado produziria.
    """
    ambiente = {"LOCALAPPDATA": str(tmp_path), "APPDATA": str(tmp_path)}
    esperado = diretorio_de_dados(ambiente, plataforma="win32") / arranque.NOME_DO_LOG

    with _SemFluxos():
        ligados = arranque.garante_fluxos(ambiente, plataforma="win32")
        print("marca-da-sonda", file=sys.stdout)
        sys.stdout.flush()

    assert ligados == ("stdout", "stderr")
    assert esperado.is_file(), f"o log nao foi criado em {esperado}"
    assert "marca-da-sonda" in esperado.read_text(encoding="utf-8"), (
        f"o log existe e nao recebeu o que se escreveu: {esperado.read_text(encoding='utf-8')!r}"
    )


def test_u15_o_log_recomeca_quando_passa_do_teto(tmp_path: Path) -> None:
    """Acrescenta, e recomeca passado o teto. **As duas direcoes, na mesma funcao.**"""
    ambiente = {"LOCALAPPDATA": str(tmp_path), "APPDATA": str(tmp_path)}
    alvo = diretorio_de_dados(ambiente, plataforma="win32") / arranque.NOME_DO_LOG
    alvo.parent.mkdir(parents=True, exist_ok=True)

    # ── abaixo do teto: ACRESCENTA ──
    alvo.write_text("linha-velha\n", encoding="utf-8")
    with _SemFluxos():
        arranque.garante_fluxos(ambiente, plataforma="win32")
        print("linha-nova", file=sys.stdout)
        sys.stdout.flush()
    conteudo = alvo.read_text(encoding="utf-8")
    assert "linha-velha" in conteudo and "linha-nova" in conteudo, conteudo

    # ── acima do teto: RECOMECA ──
    alvo.write_text("x" * (arranque.TETO_DO_LOG + 1), encoding="utf-8")
    assert alvo.stat().st_size > arranque.TETO_DO_LOG, "ancora: o arquivo nao passou o teto"
    with _SemFluxos():
        arranque.garante_fluxos(ambiente, plataforma="win32")
        print("depois-do-teto", file=sys.stdout)
        sys.stdout.flush()
    conteudo = alvo.read_text(encoding="utf-8")
    assert "depois-do-teto" in conteudo, conteudo
    assert "x" * 100 not in conteudo, f"passou do teto e NAO recomecou: {alvo.stat().st_size} bytes"


def test_u15_recua_para_o_devnull_quando_o_diretorio_nao_serve() -> None:
    """Sem `LOCALAPPDATA` nem `APPDATA`, `diretorio_de_dados` levanta — e o arranque nao pode
    morrer.

    **`os.devnull` nao conserta o diagnostico e conserta o arranque**, que e o que bloqueia hoje: o
    `uvicorn` so precisa de um objeto com `isatty`.
    """
    import uvicorn

    with _SemFluxos():
        ligados = arranque.garante_fluxos({}, plataforma="win32")
        try:
            uvicorn.Config(build_app, factory=True, host="127.0.0.1", port=0)
            falhou = None
        except Exception as erro:
            falhou = f"{type(erro).__name__}: {erro}"
        tem_isatty = hasattr(sys.stdout, "isatty")

    assert ligados == ("stdout", "stderr"), f"nao recuou: {ligados}"
    assert tem_isatty, "o recuo ligou os fluxos a algo sem `isatty`"
    assert falhou is None, f"com o recuo ainda levantou: {falhou}"


def test_u15_main_chama_garante_fluxos_antes_do_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ORDEM e o conteudo: `garante_utf8` reconfigura o que `garante_fluxos` faz existir.

    Invertidos, `garante_utf8` nao acha fluxo nenhum e o arranque morre em silencio tres chamadas
    depois. **Um teste que so afirmasse «os dois foram chamados» nao separaria as duas ordens.**
    """
    ordem: list[str] = []
    monkeypatch.setattr(arranque, "garante_fluxos", lambda *a, **k: ordem.append("fluxos") or ())
    monkeypatch.setattr(arranque, "garante_utf8", lambda *a, **k: ordem.append("utf8") or ())
    monkeypatch.setattr(
        arranque, "arranca", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("para aqui"))
    )

    codigo = arranque.main([])

    assert codigo == arranque.SAIU_POR_AMBIENTE
    assert ordem == ["fluxos", "utf8"], f"a ordem foi {ordem}"


# ═══════════════════════════════ o congelado: o diario tecnico ao lado do executavel


def _pasta_recebida(tmp_path: Path) -> tuple[Path, str]:
    """A pasta que o usuário recebeu, e o caminho do executavel dentro dela."""
    recebida = tmp_path / "pasta-que-o-usuario-recebeu"
    recebida.mkdir(parents=True, exist_ok=True)
    return recebida, str(recebida / "refino.exe")


def test_o_log_do_congelado_fica_ao_lado_do_executavel(tmp_path: Path) -> None:
    """**Par causal**: o mesmo ambiente, e so `congelado` a mudar.

    O log NAO resolve o caminho por conta propria — ele pergunta a `banco.diretorio_de_dados`, que e
    a fonte unica. Este teste e o que prova que a fonte unica se pagou: nada em `arranque.py` sabe
    que a regra mudou, e o diario tecnico seguiu o banco para dentro da pasta do executavel.

    **Sem a metade de baixo**, um `_abre_o_log` que ignorasse `congelado` e escrevesse sempre
    ao lado do executavel passaria — e quem desenvolve perderia o log no diretorio por usuario.
    """
    recebida, exe = _pasta_recebida(tmp_path)
    ambiente = {"LOCALAPPDATA": str(tmp_path / "AppData"), "APPDATA": str(tmp_path / "AppData")}
    por_usuario = tmp_path / "AppData" / "refino" / arranque.NOME_DO_LOG
    ao_lado = recebida / arranque.NOME_DO_LOG

    # ── congelado: cai ao lado do executavel ──
    with _SemFluxos():
        ligados = arranque.garante_fluxos(
            ambiente, plataforma="win32", congelado=True, executavel=exe
        )
        print("marca-do-congelado", file=sys.stdout)
        sys.stdout.flush()

    assert ligados == ("stdout", "stderr"), ligados
    assert ao_lado.is_file(), f"o log do congelado nao foi criado em {ao_lado}"
    assert "marca-do-congelado" in ao_lado.read_text(encoding="utf-8")
    assert not por_usuario.exists(), (
        f"o log tambem foi escrito em {por_usuario}: ha dois diarios, e o usuario devolve um so"
    )

    # ── do clone: continua no diretorio por usuario ──
    with _SemFluxos():
        arranque.garante_fluxos(ambiente, plataforma="win32", congelado=False, executavel=exe)
        print("marca-do-clone", file=sys.stdout)
        sys.stdout.flush()

    assert por_usuario.is_file(), f"o log do clone nao foi criado em {por_usuario}"
    assert "marca-do-clone" in por_usuario.read_text(encoding="utf-8")
    assert "marca-do-clone" not in ao_lado.read_text(encoding="utf-8"), (
        "o ramo do clone escreveu no log do congelado: os dois ramos apontam para o mesmo arquivo"
    )


def test_main_no_congelado_poe_banco_e_diagnostico_na_pasta_recebida(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A travessia inteira, e ela e o que a mudanca de produto promete.

    O que o usuario devolve e uma pasta. Entao a asserção nao e *"o caminho esta certo"* — e que,
    depois de `main` correr, **o banco esta la** e **o diagnostico sobre quem le aquela pasta esta
    escrito**, nomeando-a. Um `main` que resolvesse o banco por um caminho e relatasse outro
    passaria em duas asserções separadas e reprova nesta.
    """
    recebida, exe = _pasta_recebida(tmp_path)

    codigo = main(
        ambiente={"XDG_DATA_HOME": str(tmp_path / "por-usuario")},
        plataforma="linux",
        congelado=True,
        executavel=exe,
        abre_janela=lambda _: 99,
    )

    assert codigo == 99, f"a janela nao foi chamada: {codigo}"
    assert (recebida / "refino.db").is_file(), (
        f"o banco nao ficou na pasta recebida: {sorted(p.name for p in recebida.iterdir())}"
    )
    assert not (tmp_path / "por-usuario").exists(), (
        "o diretorio por usuario tambem foi criado — ha dois registros"
    )
    erro = capsys.readouterr().err
    assert str(recebida) in erro, f"o arranque nao disse onde ficam os dados: {erro!r}"


def test_o_aviso_de_leitura_ampla_sai_com_a_acao_e_o_restrito_nao_alarma(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """**Planta os dois vereditos e afirma que a saida DIFERE.**

    A planta e necessaria porque o veredito depende da ACL da maquina em que a suite corre, e essa
    nao se escolhe. O que se planta e a leitura; o que se mede e o relato.

    **E a planta cai sobre dado em que as duas coisas diferem:** a leitura ampla e a restrita
    partilham `S-1-5-18` e diferem por `S-1-1-0`. Um relato que ignorasse `veredito` e imprimisse
    sempre a mesma frase reprova na terceira assercao.

    A linha tecnica sai nos **dois** casos, e e ela que carrega a evidencia completa: o veredito e
    heuristica sobre uma lista de SIDs conhecidos, e sem a lista crua ao lado um SID amplo fora da
    tabela sairia como *"restrito"* sem deixar rastro.
    """
    from refino.desktop.banco import LeituraDoDiretorio

    def planta(veredito: str, amplos: tuple[str, ...], todos: tuple[str, ...]):
        def leitura(diretorio: Path, **_: object) -> LeituraDoDiretorio:
            return LeituraDoDiretorio(
                veredito=veredito,  # type: ignore[arg-type]
                diretorio=diretorio,
                herdam_leitura=todos,
                amplos=amplos,
            )

        return leitura

    monkeypatch.setattr(
        arranque,
        "quem_le_o_diretorio",
        planta("amplo", ("S-1-1-0",), ("S-1-1-0", "S-1-5-18")),
    )
    arranque._relata_quem_le(tmp_path, plataforma="win32")
    amplo = capsys.readouterr().err

    monkeypatch.setattr(arranque, "quem_le_o_diretorio", planta("restrito", (), ("S-1-5-18",)))
    arranque._relata_quem_le(tmp_path, plataforma="win32")
    restrito = capsys.readouterr().err

    assert "ATENCAO" in amplo or "ATENÇÃO" in amplo, amplo
    assert "mova a pasta" in amplo, f"o aviso nao diz o que fazer: {amplo!r}"
    assert str(tmp_path) in amplo
    assert amplo != restrito, (
        "os dois vereditos produziram a mesma saida: o relato ignora o veredito"
    )
    assert "ATENÇÃO" not in restrito, f"o desfecho tranquilo esta a alarmar: {restrito!r}"

    for saida, sids in ((amplo, ("S-1-1-0", "S-1-5-18")), (restrito, ("S-1-5-18",))):
        for sid in sids:
            assert sid in saida, f"a linha tecnica nao lista {sid}: {saida!r}"


def test_o_relato_de_quem_le_nao_levanta_sem_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """O ramo do `--windowed` sem console, e ele dispara por AUSENCIA.

    E o mesmo desenho de `_relata`: no ponto em que esta funcao corre, levantar produziria um
    traceback ao lado de uma janela que nao abriu. **A ancora e a primeira metade** — sem ela, uma
    funcao que nunca escrevesse nada passaria nesta assercao.
    """
    import io

    fluxo = io.StringIO()
    monkeypatch.setattr(sys, "stderr", fluxo)
    arranque._relata_quem_le(Path("/nao/existe"), plataforma="linux")
    assert fluxo.getvalue(), "ancora: com `stderr` a funcao tem de escrever alguma coisa"

    monkeypatch.setattr(sys, "stderr", None)
    arranque._relata_quem_le(Path("/nao/existe"), plataforma="linux")


# ═══════════════════════════════════════════════════ a janela que não abre


def test_item57_excecao_na_janela_vira_desfecho_com_causa_em_vez_de_matar_calado(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """**Exceção dentro da janela produz desfecho DECLARADO, e não morte silenciosa.**

    Até 31/08/2026 o `try` em volta de `abre` tinha só `finally`. Qualquer exceção levantada em
    `JanelaPrincipal.__init__` atravessava `main`, o `raise SystemExit(main())` nunca corria, e num
    pacote `--onedir --windowed` **nada aparecia**: o processo morria sem dizer por quê, ao lado de
    uma janela que não abriu.

    Cada desfecho declara a causa, e este era o mais visível de todos.

    **A causa é `ambiente` e não um valor novo**, porque o cabeçalho do módulo já o declara: quem
    clicou não escolheu porta, diretório nem permissão.
    """

    def levanta(_base: str) -> int:
        raise RuntimeError("segredo: postgresql://refino:senha@db/refino")

    codigo = main(ambiente=_ambiente(tmp_path), plataforma="linux", abre_janela=levanta)

    assert codigo == SAIU_POR_AMBIENTE, f"a exceção da janela deu código {codigo}"
    err = capsys.readouterr().err
    assert "a janela não abriu" in err, f"o desfecho não foi relatado: {err!r}"
    assert "RuntimeError" in err, f"o TIPO não entrou na mensagem, e é o diagnóstico: {err!r}"
    assert "ambiente" in err, f"a causa não foi declarada: {err!r}"


def test_item57_a_mensagem_nao_carrega_o_texto_da_excecao(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """**O texto da exceção FICA DE FORA, e isso é requisito e não estilo.**

    Mensagens de SQLAlchemy trazem a URL do banco, e a URL pode carregar a senha em linha — o que
    `persistence/config.py` já declara. O `refino.log` volta com a pasta quando o usuário a
    devolve, então o que entra nele é o que se aceita publicar.

    É o mesmo argumento que o `except` de `arranca` já aplicava; este teste é a outra metade — o de
    cima afirma que o **tipo** entra, e este que o **texto** não.
    """

    def levanta(_base: str) -> int:
        raise RuntimeError("postgresql://refino:SENHA_SECRETA@db/refino")

    main(ambiente=_ambiente(tmp_path), plataforma="linux", abre_janela=levanta)

    err = capsys.readouterr().err
    assert "SENHA_SECRETA" not in err, f"a senha vazou para o relato: {err!r}"
    assert "postgresql://" not in err, f"a URL do banco vazou para o relato: {err!r}"


def test_item57_a_janela_que_abre_bem_continua_a_devolver_o_codigo_dela(tmp_path: Path) -> None:
    """**O par causal**: sem isto, um `except` que engolisse tudo passaria no teste de cima.

    O mecanismo que sempre acusa é tão inútil quanto o que nunca acusa, e aqui há um valor
    concreto para afirmar: o código de saída da janela.
    """
    codigo = main(ambiente=_ambiente(tmp_path), plataforma="linux", abre_janela=lambda _: 99)

    assert codigo == 99, (
        f"a janela devolveu 99 e `main` devolveu {codigo}; o `except` novo está a engolir o "
        "caminho feliz"
    )


# ═══════════════════════════════════════════════ o cache de fontes das figuras


def test_o_cache_de_figuras_aponta_a_variavel_para_um_diretorio_que_existe(
    tmp_path: Path,
) -> None:
    """**Asserção de PRESENÇA**: o que separa *"apontou"* de *"não fez nada"* é a variável escrita.

    Uma asserção de *"não levantou"* seria satisfeita por uma função vazia — é a mesma razão pela
    qual `garante_utf8` devolve o que reconfigurou.
    """
    ambiente = {"LOCALAPPDATA": str(tmp_path)}
    executavel = str(tmp_path / "instalado" / "refino.exe")

    destino, aviso = arranque.aponta_o_cache_de_figuras(ambiente, executavel=executavel)

    assert aviso == "", f"degradou sem motivo: {aviso}"
    assert destino is not None
    assert destino.is_dir(), f"{destino} não foi criado"
    assert ambiente[arranque.VARIAVEL_DO_CACHE_DE_FIGURAS] == str(destino), (
        "a função devolveu um caminho e não o escreveu no ambiente — o `matplotlib` lê a variável, "
        "não o valor de retorno"
    )
    assert arranque.PASTA_DO_CACHE_DE_FIGURAS in destino.parts, (
        f"o cache não caiu debaixo da pasta declarada: {destino}"
    )


def test_o_cache_de_figuras_e_estavel_para_o_mesmo_caminho_de_instalacao(
    tmp_path: Path,
) -> None:
    """A metade que faz a segunda abertura não reconstruir o cache: duas chamadas, um diretório."""
    ambiente = {"LOCALAPPDATA": str(tmp_path)}
    executavel = str(tmp_path / "instalado" / "refino.exe")

    primeiro, _ = arranque.aponta_o_cache_de_figuras(dict(ambiente), executavel=executavel)
    segundo, _ = arranque.aponta_o_cache_de_figuras(dict(ambiente), executavel=executavel)

    assert primeiro == segundo, (
        "duas aberturas da MESMA pasta receberam caches diferentes — o `matplotlib` reconstruiria "
        f"a lista de fontes em cada uma: {primeiro} != {segundo}"
    )


def test_o_cache_de_figuras_difere_entre_caminhos_de_instalacao(tmp_path: Path) -> None:
    """**O par causal do anterior**, e sem ele *"estável"* seria satisfeito por um caminho fixo.

    É a metade que R12b cobra: duas cópias da pasta em caminhos diferentes, abertas
    alternadamente, desenham as duas. O `fontlist-v*.json` guarda caminhos **absolutos** para
    dentro do congelado, então uma cópia que herdasse o cache da outra apontaria para a pasta de
    origem — válido enquanto ela existisse, e quebrado no dia em que alguém a apagasse.
    """
    ambiente = {"LOCALAPPDATA": str(tmp_path)}

    aqui, _ = arranque.aponta_o_cache_de_figuras(
        dict(ambiente), executavel=str(tmp_path / "copia-a" / "refino.exe")
    )
    ali, _ = arranque.aponta_o_cache_de_figuras(
        dict(ambiente), executavel=str(tmp_path / "copia-b" / "refino.exe")
    )

    assert aqui != ali, (
        "duas instalações em caminhos diferentes partilharam o cache; a segunda leria caminhos "
        f"absolutos da primeira: {aqui}"
    )


def test_o_cache_de_figuras_ignora_a_caixa_do_caminho(tmp_path: Path) -> None:
    """O Windows não distingue maiúsculas em caminho, e sem dobrar a caixa seriam dois caches."""
    ambiente = {"LOCALAPPDATA": str(tmp_path)}

    minusculo, _ = arranque.aponta_o_cache_de_figuras(
        dict(ambiente), executavel=str(tmp_path / "refino") + "/refino.exe"
    )
    maiusculo, _ = arranque.aponta_o_cache_de_figuras(
        dict(ambiente), executavel=str(tmp_path / "REFINO") + "/refino.exe"
    )

    assert minusculo == maiusculo, (
        f"a mesma pasta em caixas diferentes deu dois caches: {minusculo} != {maiusculo}"
    )


def test_o_cache_de_figuras_degrada_quando_o_diretorio_nao_pode_ser_criado(
    tmp_path: Path,
) -> None:
    """**A planta cai sobre dado em que as duas coisas diferem**: a base é um ARQUIVO, não pasta.

    Nenhum duplo e nenhum `monkeypatch`: o `mkdir` levanta de verdade, porque não se cria pasta
    debaixo de arquivo. O desfecho exigido é degradar — motivo devolvido, variável **não** escrita
    — e não levantar.
    """
    obstaculo = tmp_path / "isto-e-um-arquivo"
    obstaculo.write_text("", encoding="utf-8")
    ambiente = {"LOCALAPPDATA": str(obstaculo)}

    destino, aviso = arranque.aponta_o_cache_de_figuras(
        ambiente, executavel=str(tmp_path / "instalado" / "refino.exe")
    )

    assert destino is None, f"devolveu {destino} sobre uma base que não pode existir"
    assert aviso, "degradou em silêncio: sem motivo, `main` não tem o que registrar"
    assert arranque.VARIAVEL_DO_CACHE_DE_FIGURAS not in ambiente, (
        "escreveu a variável apontando para um diretório que não existe — o `matplotlib` "
        "levantaria no primeiro desenho em vez de recuar para o temporário"
    )


def test_o_cache_de_figuras_degrada_quando_nao_ha_diretorio_por_usuario() -> None:
    """O segundo caminho de degradação, e ele não é o mesmo: aqui não há base nenhuma a tentar."""
    destino, aviso = arranque.aponta_o_cache_de_figuras({}, executavel="C:/refino/refino.exe")

    assert destino is None
    assert "LOCALAPPDATA" in aviso, f"o motivo não diz o que falta: {aviso!r}"


def test_o_cache_de_figuras_degrada_quando_nao_ha_executavel(tmp_path: Path) -> None:
    """O terceiro, e o mais silencioso: sem caminho de instalação não há de que derivar a chave."""
    destino, aviso = arranque.aponta_o_cache_de_figuras(
        {"LOCALAPPDATA": str(tmp_path)}, executavel="   "
    )

    assert destino is None
    assert "executavel" in aviso, f"o motivo não nomeia o que falta: {aviso!r}"


def test_main_registra_o_aviso_do_cache_de_figuras_quando_ele_degradou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """**Degrada e REGISTRA.** Sem esta metade, o desfecho seria silêncio.

    O aviso nasce a nível de módulo, antes de `garante_fluxos`; escrevê-lo ali iria para lado
    nenhum num `--windowed`, onde `sys.stderr` é `None`. Este teste afirma que ele sai no ponto em
    que há para onde escrever.
    """
    monkeypatch.setattr(arranque, "AVISO_DO_CACHE_DE_FIGURAS", "C:/nada nao serve (OSError)")

    codigo = main(ambiente=_ambiente(tmp_path), plataforma="linux", abre_janela=lambda _: 0)

    assert codigo == SAIU_BEM
    err = capsys.readouterr().err
    assert "cache de fontes das figuras" in err, f"o aviso não foi registrado: {err!r}"
    assert "C:/nada nao serve" in err, f"o motivo não chegou ao registro: {err!r}"


def test_main_nao_registra_nada_quando_o_cache_de_figuras_ficou_bem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """**O par causal do anterior**: um relato incondicional passaria no teste de cima."""
    monkeypatch.setattr(arranque, "AVISO_DO_CACHE_DE_FIGURAS", "")

    main(ambiente=_ambiente(tmp_path), plataforma="linux", abre_janela=lambda _: 0)

    err = capsys.readouterr().err
    assert "cache de fontes das figuras" not in err, (
        f"registrou a queixa sem haver queixa — o relato é incondicional: {err!r}"
    )

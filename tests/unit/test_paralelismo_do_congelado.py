"""Os DOIS remendos do `loky`, e a rede que cai para série.

## O que estes testes fecham

O congelado corria em **um núcleo** porque o `loky` relançava o `refino.exe` e cada worker abria
janela. O remendo aponta os dois sítios que geram processo para um interpretador embarcado.

**A tentativa de 07/09/2026 remendou só metade e falhou**, e é essa metade que o teste da linha de
comando existe para não deixar passar: escrever `spawn._python_exe` satisfaz o `resource_tracker` e
deixa o **worker** a receber uma linha sem `-c`.

## Por que os testes forçam `congelado=True` em vez de mexer em `sys.frozen`

`sys.frozen` é global do interpretador e escrevê-lo faz o `joblib` — e tudo o que o importa — mudar
de comportamento no meio da suíte. O parâmetro existe para que a decisão seja testável sem esse
efeito colateral, e é o mesmo idioma que `arranque.py` já usa para `congelado` e `plataforma`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from refino.desktop import paralelismo
from refino.desktop.paralelismo import (
    NOME_DO_INTERPRETADOR,
    PARALELO,
    SERIE,
    aponta_o_loky,
    resolve_o_interpretador,
)


@pytest.fixture(autouse=True)
def regime_restaurado() -> Iterator[None]:
    """`_REGIME` volta ao que era depois de cada teste — ele é GLOBAL DO PROCESSO.

    **Isto não é higiene: é a barreira que impede um vazamento medível.** `aponta_o_loky` escreve
    em `_REGIME` como efeito, e `desktop/janela/pipeline.py` lê-o **na importação do módulo**, em
    `EM_REGIME_SERIAL`. Um teste daqui que deixasse o regime em `SERIE` faria a janela nascer, na
    mesma sessão de `pytest`, a prometer o tempo do regime errado — e o desfecho seria uma
    reprovação em `test_desktop_janela.py`, com a causa a dois arquivos de distância.

    **Hoje o vazamento não morde por ORDEM e não por desenho:** o último teste deste arquivo a
    chamar `aponta_o_loky` deixa `PARALELO`. Ordem não é garantia — basta acrescentar um teste no
    fim, ou correr com `-p randomly`.
    """
    anterior = paralelismo._REGIME
    yield
    paralelismo._REGIME = anterior


# **A sonda de mentira, e ela vai NOMEADA em cada chamada que a usa.**
#
# A `pasta_congelada` tem a FORMA da entrega e um `pythonw.exe` que e um arquivo de texto. Desde
# 08/09/2026 `aponta_o_loky` SONDA o candidato antes de remendar, e um arquivo de texto nao arranca
# — sem esta costura, todos os testes abaixo passariam a medir «o falso nao arranca», que e a
# pergunta de `test_a_sonda_reprova_um_arquivo_que_nao_e_executavel`, e nao a que eles fazem.
#
# **Passar a sonda EXPLICITAMENTE e o que torna a suposicao legivel na chamada.** Um padrao global
# que a desligasse na suite esconderia a mesma coisa.
def _ARRANCA(_caminho: str) -> bool:  # noqa: N802
    """A sonda que diz sempre que sim — estes testes medem os REMENDOS, nao o arranque."""
    return True


@pytest.fixture
def pasta_congelada(tmp_path: Path) -> Path:
    """Uma pasta com a forma da entrega: `refino.exe`, `_internal/pythonw.exe` e o `._pth`.

    **O `._pth` entrou em 08/09/2026 e não é enfeite: ele é o quarto gatilho de queda.** Sem ele,
    `resolve_o_interpretador` devolve `None` e todo teste que espere `PARALELO` reprova — que foi
    exactamente o que aconteceu quando o gatilho entrou. A fixture tem de ter a forma da entrega
    REAL, e a entrega leva os dois arquivos.
    """
    exe = tmp_path / "refino.exe"
    exe.write_bytes(b"nao e um executavel, e nunca e corrido")
    interno = tmp_path / "_internal"
    interno.mkdir()
    (interno / NOME_DO_INTERPRETADOR).write_bytes(b"nem este")
    (interno / "python312._pth").write_text(".\n# import site\n", encoding="ascii")
    return tmp_path


def test_o_interpretador_e_resolvido_a_partir_do_executavel(pasta_congelada: Path) -> None:
    """A metade positiva: com o arquivo no lugar, o caminho sai absoluto e dentro da pasta."""
    achado = resolve_o_interpretador(str(pasta_congelada / "refino.exe"))

    assert achado is not None, "o interpretador está na pasta e não foi achado"
    assert achado.is_absolute()
    assert achado == pasta_congelada / "_internal" / NOME_DO_INTERPRETADOR


def test_sem_o_interpretador_a_resolucao_devolve_nada(pasta_congelada: Path) -> None:
    """A metade negativa, e ela é o primeiro gatilho da rede."""
    (pasta_congelada / "_internal" / NOME_DO_INTERPRETADOR).unlink()

    assert resolve_o_interpretador(str(pasta_congelada / "refino.exe")) is None


def test_um_interpretador_no_diretorio_de_trabalho_nao_e_resolvido(
    pasta_congelada: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A planta que separa «resolveu pela pasta» de «resolveu pelo ambiente».**

    Sem ela os dois desfechos são iguais: um `pythonw.exe` plantado onde o programa foi aberto seria
    executado sem sintoma, e um interpretador errado que importasse outro `refino` produziria
    resultado errado **sem levantar**.

    A planta cai sobre dado em que as duas coisas diferem: o arquivo existe no diretório de
    trabalho e **não** existe ao lado do executável.
    """
    (pasta_congelada / "_internal" / NOME_DO_INTERPRETADOR).unlink()
    outro = tmp_path / "outro_lugar"
    outro.mkdir()
    (outro / NOME_DO_INTERPRETADOR).write_bytes(b"o intruso")
    monkeypatch.chdir(outro)

    assert resolve_o_interpretador(str(pasta_congelada / "refino.exe")) is None, (
        "resolveu um interpretador que está no diretório de trabalho e não na pasta do congelado"
    )


def test_fora_do_congelado_nao_escreve_nada_e_deixa_paralelo(pasta_congelada: Path) -> None:
    """A CLI já paraleliza, e o remendo não lhe diz respeito.

    Devolver `SERIE` aqui a poria em série, 6,7x mais lenta, por causa de um conserto que existe
    para o caso oposto. A âncora é o valor de `spawn._python_exe` antes e depois.
    """
    from joblib.externals.loky.backend import spawn

    antes = spawn._python_exe

    assert aponta_o_loky(str(pasta_congelada / "refino.exe"), congelado=False) == PARALELO
    assert spawn._python_exe == antes, "escreveu no `loky` fora do congelado"


def test_no_congelado_com_interpretador_aponta_os_DOIS_sitios(  # noqa: N802
    pasta_congelada: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A asserção que a tentativa de 07/09/2026 não tinha, e que a fez falhar.**

    Escrever `spawn._python_exe` satisfaz o `resource_tracker` e **deixa o worker de fora**: o ramo
    `sys.frozen` de `get_command_line` devolve `[sys.executable, "--multiprocessing-fork",
    pipe_handle]`, sem `-c` e sem `parent_pid`. O interpretador recebe uma opção que não conhece e
    sai; `main()` nunca corre.

    Por isso este teste afirma a **linha de comando** e não só o valor do símbolo.

    ## A FORMA DA LINHA NÃO SEPARA OS DOIS SÍTIOS, e a revisão de 08/09/2026 mediu-o

    As quatro asserções abaixo — o interpretador na posição zero, o `-c`, o `parent_pid` e o
    `pipe_handle` — descrevem a linha que o **ramo não congelado** do `get_command_line`
    vendorizado já produz. E aquele ramo obtém o executável de `spawn.get_executable()`, que lê o
    `_python_exe` que o primeiro remendo acabou de escrever.

    > **Com o segundo sítio deixado original, as quatro passavam.** O teste afirmava dois remendos
    > e media um.

    O que separa as duas versões é a **identidade da função** e o programa de reposição do `stdio`,
    que só a nossa escreve. As duas asserções abaixo vêm primeiro por isso: sem elas, o resto é
    âncora de forma sobre a função errada.
    """
    from joblib.externals.loky.backend import popen_loky_win32, spawn

    monkeypatch.setattr(spawn, "_python_exe", "NAO_TOCADO", raising=False)
    monkeypatch.setattr(
        popen_loky_win32, "get_command_line", popen_loky_win32.get_command_line, raising=False
    )
    original = popen_loky_win32.get_command_line
    esperado = str(pasta_congelada / "_internal" / NOME_DO_INTERPRETADOR)

    assert (
        aponta_o_loky(str(pasta_congelada / "refino.exe"), congelado=True, sonda=_ARRANCA)
        == PARALELO
    )

    # **O SEGUNDO sítio foi de facto substituído.** É esta linha que distingue «os dois remendos»
    # de «só o primeiro»; sem ela o teste é satisfeito pela função vendorizada.
    assert popen_loky_win32.get_command_line is not original, (
        "`popen_loky_win32.get_command_line` continua a ser a função vendorizada: só o primeiro "
        "sítio foi remendado, e o ramo `sys.frozen` dela devolve a linha SEM `-c` e SEM "
        "`parent_pid` — o interpretador recebe uma opção que não conhece e sai (AR192)"
    )
    assert getattr(popen_loky_win32.get_command_line, "_refino_remendado", False), (
        "a função foi trocada por outra que não é a nossa — a marca de idempotência não está lá"
    )

    # (1) o sítio que o `resource_tracker` lê
    assert spawn._python_exe == esperado

    # (2) o sítio que o WORKER lê — e é este que faltava
    linha = popen_loky_win32.get_command_line(pipe_handle=1234, parent_pid=99)
    assert linha[0] == esperado, f"a linha do worker não nomeia o interpretador: {linha}"
    assert "-c" in linha, f"a linha do worker não leva `-c`: {linha}"
    assert any("parent_pid=99" in str(parte) for parte in linha), (
        f"a linha do worker não leva o `parent_pid` — foi ele que matou 07/09: {linha}"
    )
    assert any("pipe_handle=1234" in str(parte) for parte in linha), linha


def test_sem_interpretador_no_congelado_cai_para_serie_sem_tocar_no_loky(
    pasta_congelada: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rede, e o modo de falha aceitável é ser LENTO — nunca doze janelas."""
    from joblib.externals.loky.backend import popen_loky_win32, spawn

    (pasta_congelada / "_internal" / NOME_DO_INTERPRETADOR).unlink()
    monkeypatch.setattr(spawn, "_python_exe", "NAO_TOCADO", raising=False)
    original = popen_loky_win32.get_command_line

    assert (
        aponta_o_loky(str(pasta_congelada / "refino.exe"), congelado=True, sonda=_ARRANCA) == SERIE
    )
    assert spawn._python_exe == "NAO_TOCADO"
    assert popen_loky_win32.get_command_line is original


def test_o_modulo_vendorizado_a_levantar_tambem_cai_para_serie(
    pasta_congelada: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O terceiro gatilho: uma versão do `joblib` com outra árvore.

    **É a rede que torna isto adoptável.** Sem ela, um `joblib` novo faria o remendo encaixar a
    meio — e o desfecho de um remendo a meio, medido em 07/09/2026, são doze caixas de erro.
    """
    import builtins

    real = builtins.__import__

    def recusa(nome: str, *args: Any, **kwargs: Any) -> Any:
        if "popen_loky_win32" in nome or "loky.backend" in nome:
            raise ImportError("planta: o módulo vendorizado mudou de forma")
        return real(nome, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recusa)

    assert (
        aponta_o_loky(str(pasta_congelada / "refino.exe"), congelado=True, sonda=_ARRANCA) == SERIE
    )


def test_chamar_duas_vezes_nao_empilha_o_remendo(
    pasta_congelada: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Idempotência, e ela não é zelo: sem a guarda, cada chamada envolve a anterior.

    O arranque chama uma vez; um teste ou uma reentrada chamariam mais, e a pilha cresceria em
    silêncio até estourar a recursão dentro de um `CreateProcess`.
    """
    from joblib.externals.loky.backend import popen_loky_win32, spawn

    monkeypatch.setattr(spawn, "_python_exe", "NAO_TOCADO", raising=False)
    monkeypatch.setattr(
        popen_loky_win32, "get_command_line", popen_loky_win32.get_command_line, raising=False
    )

    aponta_o_loky(str(pasta_congelada / "refino.exe"), congelado=True, sonda=_ARRANCA)
    primeira = popen_loky_win32.get_command_line
    aponta_o_loky(str(pasta_congelada / "refino.exe"), congelado=True, sonda=_ARRANCA)

    assert popen_loky_win32.get_command_line is primeira, "a segunda chamada envolveu a primeira"


def test_o_nome_e_pythonw_e_a_razao_e_a_janela() -> None:
    """**`python.exe` reintroduziria o defeito que este módulo existe para fechar.**

    Os dois sítios do `loky` chamam `CreateProcess` com `dwCreationFlags = 0`, e o `refino.exe` é
    `console=False` — não há console a herdar. Um filho de subsistema console ganha um console
    novo, e são doze. Este teste é o que impede a troca de voltar por descuido.
    """
    assert NOME_DO_INTERPRETADOR == "pythonw.exe", (
        "o interpretador embarcado voltou a ser de console — cada worker passa a abrir uma janela "
        "preta, e o `MainWindowHandle` do portão nem a vê, porque ela pertence ao `conhost.exe`"
    )


def test_sem_executavel_nenhum_a_resolucao_nao_levanta(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sys.executable` vazio acontece em ambientes embutidos, e não pode derrubar o arranque."""
    monkeypatch.setattr(sys, "executable", "")

    assert resolve_o_interpretador() is None


def test_a_linha_do_worker_repoe_o_stdio_antes_de_chamar_o_main(
    pasta_congelada: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**O conserto que o congelado exigiu, e a falha dele é SILENCIOSA — daí o teste.**

    Medido em 08/09/2026: o worker nascia — a contagem de `pythonw` ia de 0 a 1 — e a correção
    terminava em `TerminatedWorkerError` em 4,6 s, **sem traceback em lado nenhum**. Um
    `pythonw.exe` filho de um pai `--windowed` não herda handle nenhum, então `sys.stdin`,
    `sys.stdout` e `sys.stderr` nascem `None`; o `_bootstrap` do `multiprocessing` faz
    `sys.stdin.close()` no princípio, e `AttributeError` num processo sem saída não deixa rasto.

    **A âncora é a ordem**, e não só a presença: repor o `stdio` depois do `import main` não
    serviria de nada, porque é o `main()` que desce até ao `_bootstrap`.
    """
    from joblib.externals.loky.backend import popen_loky_win32, spawn

    # **O remendo tem de estar APLICADO para a linha existir**, e a primeira versão deste teste
    # esqueceu-o: leu a `get_command_line` original do `loky` e reprovou por motivo errado.
    monkeypatch.setattr(spawn, "_python_exe", "NAO_TOCADO", raising=False)
    monkeypatch.setattr(
        popen_loky_win32, "get_command_line", popen_loky_win32.get_command_line, raising=False
    )
    assert (
        aponta_o_loky(str(pasta_congelada / "refino.exe"), congelado=True, sonda=_ARRANCA)
        == PARALELO
    )

    gerada = popen_loky_win32.get_command_line(pipe_handle=1, parent_pid=2)
    programa = next((str(p) for p in gerada if "popen_loky_win32" in str(p)), "")

    assert "sys.stdin=sys.stdin or" in programa, (
        f"a linha do worker não repõe o `stdio`: {programa[:200]}"
    )
    assert programa.index("sys.stderr=") < programa.index("import main"), (
        "o `stdio` é reposto DEPOIS do import do `main`, e aí já não serve"
    )
    assert "\n" not in programa, (
        "o programa do worker tem quebra de linha, e o `loky` junta a linha com aspas — "
        "uma quebra dentro do argumento citado é frágil na fronteira do `CreateProcess`"
    )


def test_a_sonda_aprova_um_interpretador_de_verdade() -> None:
    """A metade positiva da sonda, contra o interpretador que está a correr esta suíte.

    **Sem esta, a sonda podia estar sempre a devolver `False`** — e o desfecho seria o congelado a
    correr em série para sempre, com a rede a parecer sã. O modo de falha é «acusa sempre», e o
    sintoma é lentidão silenciosa.
    """
    from refino.desktop.paralelismo import _o_interpretador_arranca

    assert _o_interpretador_arranca(sys.executable), (
        f"a sonda recusou {sys.executable!r}, que é o interpretador que está a correr este teste. "
        "Ela reprova tudo, e o congelado cai para série mesmo com o interpretador embarcado bom"
    )


def test_a_sonda_reprova_um_arquivo_que_nao_e_executavel(pasta_congelada: Path) -> None:
    """A metade negativa: existir não é arrancar.

    **É o buraco que o achado #5 da revisão de 08/09/2026 nomeou.** As três guardas de
    `resolve_o_interpretador` são de existência; um `pythonw.exe` de 100 KB de zeros passa as três,
    o remendo é aplicado, e os workers morrem em `TerminatedWorkerError` — falha dura onde o modo
    declarado é «ser lento». O portão do congelamento plantou exactamente esse arquivo.
    """
    from refino.desktop.paralelismo import _o_interpretador_arranca

    falso = pasta_congelada / "_internal" / NOME_DO_INTERPRETADOR
    assert falso.is_file(), "âncora: a planta existe no disco"
    assert not _o_interpretador_arranca(str(falso)), (
        "a sonda aprovou um arquivo que não é executável — ela não distingue «existe» de «arranca»"
    )


def test_o_interpretador_que_nao_arranca_cai_para_serie(
    pasta_congelada: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O quarto gatilho de queda, com a sonda REAL: presente e inutilizável vira SÉRIE.

    **Par causal com os testes acima**: as mesmas condições, a mesma pasta, e a única diferença é
    a sonda ser a de produção em vez do duplo. Ali dá `PARALELO`; aqui dá `SERIE` — e o `loky` fica
    por tocar, que é o que impede as doze janelas.
    """
    from joblib.externals.loky.backend import popen_loky_win32, spawn

    # Pelo `monkeypatch`, como nos testes acima, e nao por atribuicao: `_python_exe` e global do
    # processo, e escrito a seco ele ficava "NAO_TOCADO" ate o fim da sessao, e todo worker novo do
    # `loky` depois deste arquivo morria com `WinError 2`.
    monkeypatch.setattr(spawn, "_python_exe", "NAO_TOCADO", raising=False)
    original = popen_loky_win32.get_command_line

    assert aponta_o_loky(str(pasta_congelada / "refino.exe"), congelado=True) == SERIE, (
        "um interpretador que não arranca deu regime PARALELO: a rede não o apanha, e a correção "
        "vai falhar com `TerminatedWorkerError` em vez de correr devagar"
    )
    assert spawn._python_exe == "NAO_TOCADO"
    assert popen_loky_win32.get_command_line is original


def test_sem_o_pth_a_resolucao_devolve_nada(
    pasta_congelada: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""O quarto gatilho: o interpretador está lá, o `._pth` não, e o regime cai para SÉRIE.

    **Este é o gatilho que MASCARA, e é por isso que ele existe apesar da sonda.** Sem `._pth`, o
    interpretador não acha a stdlib pelo marco `Lib\os.py` — que não existe em `_internal` — e cai
    no registro do Windows ou no prefixo de construção. **Na máquina que constrói, isso resolve
    para a instalação original do `uv`**: o worker arranca, a sonda aprova, e ele importa `numpy`,
    `sklearn` e `refino` de fora da pasta entregue. O resultado seria produzido por código que não
    viajou.

    Na máquina de quem recebe o executável não há instalação para onde cair, e ele morre. **A
    sonda não distingue os dois casos; a presença do `._pth` distingue.**
    """
    from joblib.externals.loky.backend import popen_loky_win32, spawn

    pth = pasta_congelada / "_internal" / "python312._pth"
    assert pth.is_file(), "âncora: a pasta nasce COM o `._pth`, e é a sua remoção que se mede"
    pth.unlink()

    assert resolve_o_interpretador(str(pasta_congelada / "refino.exe")) is None, (
        "o interpretador foi resolvido sem `._pth` ao lado: na máquina que constrói ele importaria "
        "da instalação do `uv` e o defeito ficaria invisível até chegar a quem recebe o executável"
    )

    monkeypatch.setattr(spawn, "_python_exe", "NAO_TOCADO", raising=False)
    original = popen_loky_win32.get_command_line
    assert aponta_o_loky(str(pasta_congelada / "refino.exe"), congelado=True) == SERIE
    assert spawn._python_exe == "NAO_TOCADO"
    assert popen_loky_win32.get_command_line is original


def test_a_aresta_de_api_para_desktop_e_barata_por_MEDICAO(  # noqa: N802
) -> None:
    """As duas premissas que autorizam `api` a importar de `desktop` são AFIRMADAS, não supostas.

    **`src/refino/api/rotas.py` importa `refino.desktop.paralelismo`, e essa é a primeira aresta
    nessa direção no projeto inteiro.** A direção declarada é a oposta: a janela alcança o núcleo
    pela API. O comentário no import argumenta que as duas são *fronteiras*, ambas fora do núcleo,
    e apoia-se em dois factos:

    1. `refino/desktop/__init__.py` é **só docstring**, sem import nenhum;
    2. `desktop/paralelismo.py` importa **só biblioteca padrão**.

    **Sem este portão os dois eram afirmação de comentário.** Um `import` de `PySide6` em qualquer
    um deles faria a CLI arrastar o Qt — que é grupo opcional, e cuja ausência é
    condição normal em quem só usa a API.
    """
    import ast
    import sys as _sys

    from tests.conftest import REPO

    pacote = REPO / "src" / "refino" / "desktop"

    init = ast.parse((pacote / "__init__.py").read_text(encoding="utf-8"))
    importa = [n for n in ast.walk(init) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not importa, (
        f"`refino/desktop/__init__.py` passou a importar {len(importa)} módulo(s). Toda importação "
        "de `refino.desktop.paralelismo` — inclusive a de `api/rotas.py` — paga isto, e a tela web "
        "e a CLI não devem pagar o Qt"
    )

    arvore = ast.parse((pacote / "paralelismo.py").read_text(encoding="utf-8"))
    # Só os imports de topo: os de dentro de função são deliberados e documentados (o `joblib`
    # vendorizado, o `subprocess` da sonda, o `multiprocessing.util` da linha de comando).
    de_topo: list[str] = []
    for no in arvore.body:
        if isinstance(no, ast.Import):
            de_topo += [a.name.split(".")[0] for a in no.names]
        elif isinstance(no, ast.ImportFrom) and no.module and no.level == 0:
            de_topo.append(no.module.split(".")[0])

    assert de_topo, "âncora: o módulo não declarou import nenhum no topo — o detector falhou"
    fora = sorted(set(de_topo) - _sys.stdlib_module_names)
    assert not fora, (
        f"`desktop/paralelismo.py` passou a importar {fora} no topo, e isso não é biblioteca "
        "padrão. `api/rotas.py` importa este módulo, e o argumento que autoriza essa aresta é "
        "precisamente ele ser barato"
    )

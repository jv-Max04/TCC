"""O banco do caminho de mesa.

## O que estes testes alcançam

Bancos SQLite de verdade, em diretórios temporários de verdade, com o `alembic` a correr de verdade.
Os **dois** ramos de plataforma são exercitados aqui no Linux, porque `diretorio_de_dados` recebe
ambiente e plataforma por injeção — não há `skip` de Windows nesta unidade, e isso é desenho e não
sorte.

## O que eles NÃO alcançam, e a qualificação é a seção inteira

**A ACL do Windows.** Foi medido que um arquivo sob `%LOCALAPPDATA%` herda `SYSTEM`,
`Administrators` e a conta do usuário, e **nem `Everyone` nem `BUILTIN\\Users`** — mas essa medição
foi feita com `Get-Acl` e `icacls` **em Windows**, e a suíte corre no WSL. O que se afirma aqui é o
**caminho escolhido**; que aquele caminho tenha aquela ACL é medição feita à parte, não teste.

**O modo 0600 é afirmado só em POSIX**, que é onde `chmod` significa algo. Não há `skip`: o teste do
modo é POSIX e o teste do caminho de Windows é outro, e cada um mede o que a sua plataforma tem.
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

from refino.desktop import banco
from refino.desktop.banco import (
    NOME_DO_ARQUIVO,
    PASTA,
    SIDS_DE_MAIS_DE_UMA_CONTA,
    BancoDeMesa,
    BancoDeMesaError,
    LeituraDeAclError,
    diretorio_de_dados,
    esta_congelado,
    migra,
    prepara,
    quem_le_o_diretorio,
    url_de_mesa,
)
from refino.persistence.schema import METADATA
from tests.permissao_capability import (
    DETALHE_DO_MODO,
    DETALHE_DO_SEM_ESCRITA,
    modo_de_arquivo_respeitado,
    permissao_de_diretorio_respeitada,
)

pytestmark = pytest.mark.integration


def _tabelas(caminho: Path) -> set[str]:
    """Os nomes de tabela que o arquivo SQLite contém."""
    con = sqlite3.connect(caminho)
    try:
        return {
            linha[0] for linha in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        con.close()


# ═══════════════════════════════════════════════════ o caminho, nos dois ramos


def test_o_ramo_de_windows_poe_o_banco_sob_localappdata() -> None:
    """E `%APPDATA%` é o recuo, quando `%LOCALAPPDATA%` falta.

    **Os dois valores são afirmados, e não só o primeiro:** um recuo que nunca foi exercitado é um
    ramo `else`, que dispara por ausência de condição.
    """
    local = diretorio_de_dados(
        {"LOCALAPPDATA": r"C:\Users\alguem\AppData\Local"}, plataforma="win32"
    )
    assert local.name == "refino"
    assert "Local" in str(local), f"o caminho não caiu sob LOCALAPPDATA: {local}"

    recuo = diretorio_de_dados({"APPDATA": r"C:\Users\alguem\AppData\Roaming"}, plataforma="win32")
    assert recuo.name == "refino"
    assert "Roaming" in str(recuo), f"o recuo não caiu sob APPDATA: {recuo}"


def test_sem_localappdata_nem_appdata_a_mensagem_nomeia_a_saida() -> None:
    """O terceiro desfecho do ramo de Windows, e a mensagem tem de ser acionável.

    Quem opera não pode consertar variável de ambiente ausente sem saber qual pôr, então a mensagem
    nomeia `REFINO_DB_URL` e dá um exemplo.
    """
    with pytest.raises(BancoDeMesaError, match="REFINO_DB_URL"):
        diretorio_de_dados({}, plataforma="win32")


def test_o_ramo_de_posix_respeita_xdg_e_recua_para_local_share() -> None:
    """As duas formas de POSIX, no mesmo teste — a variável e o padrão."""
    com_xdg = diretorio_de_dados({"XDG_DATA_HOME": "/dados"}, plataforma="linux")
    assert com_xdg == Path("/dados/refino")

    sem_xdg = diretorio_de_dados({"HOME": "/lar/alguem"}, plataforma="linux")
    assert sem_xdg == Path("/lar/alguem/.local/share/refino")


def test_sem_home_nem_xdg_a_mensagem_nomeia_a_saida() -> None:
    """O terceiro desfecho do ramo de POSIX."""
    with pytest.raises(BancoDeMesaError, match="REFINO_DB_URL"):
        diretorio_de_dados({}, plataforma="linux")


def test_a_url_e_absoluta_nas_duas_plataformas() -> None:
    """**As quatro barras do POSIX não são erro de digitação**, e a asserção existe para isso.

    `sqlite:///caminho` é RELATIVO ao `cwd`; `sqlite:////caminho` é absoluto. Num executável de mesa
    o `cwd` é onde o laboratório clicou, então a diferença de uma barra é a diferença entre um banco
    e vários. A forma do Windows tem três barras mais a letra da unidade, e é igualmente absoluta.

    A prova de que a forma POSIX funciona não é esta asserção: é `migra` a correr contra ela, mais
    abaixo.
    """
    posix = url_de_mesa({"HOME": "/lar/alguem"}, plataforma="linux")
    assert posix == "sqlite:////lar/alguem/.local/share/refino/refino.db"

    windows = url_de_mesa({"LOCALAPPDATA": "C:/Users/alguem/AppData/Local"}, plataforma="win32")
    assert windows.startswith("sqlite:///C:/"), windows
    assert windows.endswith(f"/refino/{NOME_DO_ARQUIVO}"), windows


# ═══════════════════════════════════════════════════ as migrações


def test_o_diretorio_das_migracoes_e_resolvido_pelo_pacote_e_nao_pelo_repositorio() -> None:
    """**Asserção de PRESENÇA sobre o que aquele diretório contém**, e ela é o que pega mudança de
    layout.

    O primeiro rascunho resolvia por `migrations.__file__`, e aquele valor é `None`: o diretório
    **não tem `__init__.py`**, então importa como pacote de **namespace**. O erro chegava como
    `TypeError` sobre `__fspath__`, longe da causa. A derivação certa parte de `refino.persistence`,
    que é pacote regular.
    """
    diretorio = banco._diretorio_das_migracoes()

    assert diretorio.is_dir(), f"{diretorio} não é diretório"
    assert (diretorio / "env.py").is_file(), (
        f"{diretorio} não contém `env.py` — o `script_location` aponta para o lugar errado"
    )
    versoes = diretorio / "versions"
    assert versoes.is_dir(), f"{diretorio} não contém `versions/`"
    assert list(versoes.glob("*.py")), (
        f"{versoes} está vazio — `alembic upgrade head` não teria o que aplicar, e a asserção das "
        "tabelas abaixo passaria a medir outra coisa"
    )


def test_a_migracao_programatica_cria_todas_as_tabelas_do_metadata(tmp_path: Path) -> None:
    """**A lista esperada sai do `METADATA` e não de um número escrito à mão.**

    Uma asserção de contagem — *"nove tabelas"* — passaria a medir a coisa errada no dia em que uma
    tabela nascesse, e o defeito seria uma migração faltante a parecer um número desatualizado.
    """
    caminho = tmp_path / NOME_DO_ARQUIVO
    migra(f"sqlite:///{caminho.as_posix()}")

    assert caminho.is_file(), "o arquivo do banco não foi criado"
    achadas = _tabelas(caminho)
    esperadas = set(METADATA.tables)

    assert esperadas, "âncora: o `METADATA` não declara tabela nenhuma, e a asserção seria vácua"
    assert esperadas <= achadas, f"faltam tabelas: {sorted(esperadas - achadas)}"
    assert "alembic_version" in achadas, (
        "o `alembic` não gravou a versão — a segunda migração recomeçaria do zero"
    )


# ═══════════════════════════════════════════════════ prepara, ponta a ponta


def _ambiente(raiz: Path) -> dict[str, str]:
    """Um ambiente POSIX que põe o banco dentro de `raiz`."""
    return {"XDG_DATA_HOME": str(raiz)}


def test_o_primeiro_arranque_cria_e_migra_e_o_segundo_nao_recomeca(tmp_path: Path) -> None:
    """**Par causal dos dois arranques, e o dado escrito entre eles é a prova.**

    `ja_existia` sozinho seria fraco: ele diz o que a função observou, não o que aconteceu com o
    banco. A asserção que separa *"não remigrou"* de *"remigrou e recriou vazio"* é a linha inserida
    entre os dois arranques continuar lá.
    """
    ambiente = _ambiente(tmp_path)

    primeiro = prepara(ambiente, plataforma="linux")
    assert isinstance(primeiro, BancoDeMesa)
    assert primeiro.ja_existia is False, "âncora: no primeiro arranque o arquivo não existia"
    assert primeiro.caminho.is_file()
    assert set(METADATA.tables) <= _tabelas(primeiro.caminho)

    con = sqlite3.connect(primeiro.caminho)
    # `criado_em` é `NOT NULL` sem padrão no esquema, e `nome` é `nome` e não `name` — conferido em
    # `METADATA.tables["studies"]` antes de escrever esta linha, em vez de adivinhado.
    con.execute(
        "INSERT INTO studies (nome, criado_em) VALUES (?, ?)",
        ("sonda-do-segundo-arranque", "2026-08-22T00:00:00"),
    )
    con.commit()
    con.close()

    segundo = prepara(ambiente, plataforma="linux")
    assert segundo.ja_existia is True, "o segundo arranque não viu o arquivo do primeiro"
    assert segundo.caminho == primeiro.caminho

    con = sqlite3.connect(segundo.caminho)
    try:
        sobreviventes = list(con.execute("SELECT nome FROM studies"))
    finally:
        con.close()
    assert sobreviventes == [("sonda-do-segundo-arranque",)], (
        "o dado do primeiro arranque não sobreviveu ao segundo — a migração recomeçou"
    )


def test_o_diretorio_pai_e_criado_quando_falta(tmp_path: Path) -> None:
    """Dois níveis de profundidade, e não um.

    Um `mkdir` sem `parents=True` passaria num teste de um nível só — e o caminho real tem dois,
    porque o diretório de dados do usuário pode não existir numa conta nova.
    """
    fundo = tmp_path / "que" / "nao" / "existe"
    assert not fundo.exists(), "âncora: o diretório não pode existir antes"

    resultado = prepara(_ambiente(fundo), plataforma="linux")

    assert resultado.caminho.is_file()
    assert resultado.caminho.parent.is_dir()


def test_o_arquivo_nasce_em_0600(tmp_path: Path) -> None:
    """O modo, afirmado sem condicional de plataforma — e a ausência do `skipif` é deliberada.

    **Um `skipif(sys.platform == "win32")` aqui seria ramo INERTE**: a suíte corre no WSL, então
    aquele ramo nunca dispararia na máquina de quem desenvolve, e um `skip` que nunca dispara não
    mede nada.

    Sem ele, correr a suíte em Python nativo de Windows faz este teste **reprovar** — e isso é
    informação e não defeito: a mensagem diz que a garantia de `chmod` não existe lá, e a medição
    com `Get-Acl` e `icacls` registra o que existe em vez dela. Este teste mede POSIX;
    aquela medição mede Windows; nenhum dos dois finge medir o outro.

    > **A GUARDA que existe NÃO é a que os dois parágrafos acima recusam.**
    > A recusa é de `skipif(sys.platform == "win32")` — uma pergunta sobre o NOME do sistema —, e o
    > argumento dela continua inteiro: em WSL aquele ramo seria inerte. Esta pergunta pela
    > **capacidade**, e a diferença é medível: `permissao_de_diretorio_respeitada()` e
    > `modo_de_arquivo_respeitado()` devolvem `True` em WSL, e o teste corre inteiro lá. O ramo não
    > é inerte — ele dispara exactamente onde a medição não pode acontecer.
    >
    > **E a metade «informação e não defeito» é preservada e não descartada:** a informação passou
    > da mensagem de reprovação para a mensagem do `skip`, que diz a mesma coisa e é conferida por
    > dois portões (o de tempo de execução e o estático de `skip_policy.py`). O que mudou é que ela
    > deixou de viver numa linha permanentemente vermelha — e linha permanentemente vermelha é o que
    > a equipa aprende a ignorar.
    """
    if not modo_de_arquivo_respeitado():
        pytest.skip(f"permissão POSIX ignorada: {DETALHE_DO_MODO}")

    resultado = prepara(_ambiente(tmp_path), plataforma="linux")

    modo = stat.S_IMODE(resultado.caminho.stat().st_mode)
    assert modo == 0o600, (
        f"o banco nasceu em {modo:o} e não em 600 — dado clínico com permissão larga"
    )


def test_diretorio_sem_permissao_de_escrita_produz_causa_de_ambiente(tmp_path: Path) -> None:
    """O desfecho de falha, com a mensagem a nomear o caminho.

    **Par causal:** o mesmo diretório funciona antes de a permissão sair. Sem essa âncora, um erro
    por qualquer outra razão passaria por «sem permissão».
    """
    if not permissao_de_diretorio_respeitada():
        pytest.skip(f"permissão POSIX ignorada: {DETALHE_DO_SEM_ESCRITA}")

    raiz = tmp_path / "dados"
    raiz.mkdir()

    antes = prepara(_ambiente(raiz), plataforma="linux")
    assert antes.caminho.is_file(), "âncora: com permissão, o banco nasce"

    outro = tmp_path / "trancado"
    outro.mkdir(mode=0o500)
    try:
        # **`re.escape` e nao o caminho cru.** `match=` e REGEX, e um caminho do Windows traz
        # `\U` de `\Users`, que o `re` recusa com *"incomplete escape"*. Enquanto ele esteve cru,
        # este teste reprovava pela compilacao do padrao — nunca chegou a medir se a excecao foi
        # levantada, e a linha de base classificava a reprovacao como «o Windows nao tem bits
        # POSIX», que e outra causa. O veredito acusava a plataforma e o defeito era do arnes.
        with pytest.raises(BancoDeMesaError, match=re.escape(str(outro))):
            prepara(_ambiente(outro), plataforma="linux")
    finally:
        outro.chmod(0o700)


def test_migrar_falso_prepara_o_arquivo_e_nao_cria_tabela(tmp_path: Path) -> None:
    """O parâmetro existe para o teste, e o teste afirma o que ele faz.

    Um parâmetro de conveniência que ninguém exercita é parâmetro que muda de sentido em silêncio na
    primeira refatoração.
    """
    # **A âncora vem PRIMEIRO, e a revisão apontou a ausência dela.** `_tabelas(...) == set()`
    # passa sobre um arquivo que não existe, sobre um caminho errado, e sobre uma consulta que
    # falhou — todos dão conjunto vazio. O que separa *"não migrou"* de *"`_tabelas` não vê nada"*
    # é provar, no mesmo teste, que ela **vê** um esquema quando há um.
    com_migracao = prepara(_ambiente(tmp_path / "com"), plataforma="linux")
    assert _tabelas(com_migracao.caminho) != set(), (
        "âncora: `_tabelas` tem de detetar um esquema povoado. Não detetou, então a asserção "
        "seguinte não mede nada"
    )

    resultado = prepara(_ambiente(tmp_path / "sem"), plataforma="linux", migrar=False)

    assert resultado.caminho.is_file(), "o arquivo tem de existir, com permissão fixada"
    assert _tabelas(resultado.caminho) == set(), (
        f"com `migrar=False` não devia haver tabela nenhuma: {sorted(_tabelas(resultado.caminho))}"
    )


def test_migra_restaura_uma_url_que_ja_estava_no_ambiente(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`migra` escreve `REFINO_DB_URL` para o `env.py` a ler, e tem de a devolver como estava.

    **O ramo de restauração dispara só quando a variável JÁ existia**, e é o caso do desenvolvedor
    que corre a ferramenta com outro banco configurado. Sem ele exercitado, uma migração de mesa
    deixaria a sessão a apontar para o SQLite temporário — e o sintoma seria a gravação seguinte a
    ir para o banco errado, sem erro nenhum.
    """
    from refino.persistence.config import VARIAVEL_DA_URL

    anterior = "postgresql+psycopg2://alguem:senha@servidor:5432/refino"
    monkeypatch.setenv(VARIAVEL_DA_URL, anterior)

    caminho = tmp_path / NOME_DO_ARQUIVO
    migra(f"sqlite:///{caminho.as_posix()}")

    import os

    assert os.environ[VARIAVEL_DA_URL] == anterior, (
        "`migra` não restaurou a URL anterior: a sessão passou a apontar para o banco de mesa"
    )
    assert caminho.is_file(), "âncora: a migração aconteceu de facto"


def test_sem_o_file_do_pacote_a_mensagem_diz_o_que_mudou(monkeypatch: pytest.MonkeyPatch) -> None:
    """O ramo que estava sob `pragma: no cover`, e a revisão mostrou que é testável.

    Tenta-se o teste antes do `pragma`. Eu pus o `pragma` com a justificativa de que
    `refino.persistence` tem `__init__.py` — verdade, e irrelevante: o ramo é alcançável por
    substituição, e um `pragma` sobre código alcançável esconde comportamento em vez de declarar
    inalcançabilidade.
    """
    from refino import persistence

    monkeypatch.setattr(persistence, "__file__", None)

    with pytest.raises(BancoDeMesaError, match="não tem `__file__`"):
        banco._diretorio_das_migracoes()


# ═══════════════════════════════════════ o congelado: o registro ao lado do executável


def _executavel_falso(pasta: Path) -> str:
    """Um caminho de executável dentro de `pasta`, com a pasta criada.

    **O arquivo não é criado, e a ausência dele é o ponto.** `diretorio_de_dados` responde
    *"a pasta que contém o executável"*, e um teste que precisasse de um `.exe` de verdade estaria a
    medir o sistema de arquivos em vez da regra.
    """
    pasta.mkdir(parents=True, exist_ok=True)
    return str(pasta / "refino.exe")


def test_esta_congelado_responde_pelos_dois_lados_e_pelo_sys() -> None:
    """As **três** entradas da função, e a terceira é a que produção usa.

    Sem a última asserção, `esta_congelado` poderia ignorar `sys.frozen` por completo e os dois
    primeiros casos continuariam verdes — a injeção teria substituído o mecanismo em vez de o
    tornar exercitável.
    """
    assert esta_congelado(True) is True
    assert esta_congelado(False) is False
    assert esta_congelado(None) is bool(getattr(sys, "frozen", False))
    assert esta_congelado() is False, (
        "âncora: a suíte não corre dentro de um congelado, então o padrão tem de ser `False`. "
        "Se isto reprovar, todos os testes abaixo estão a medir o ramo errado"
    )


def test_o_congelado_poe_o_registro_ao_lado_do_executavel_e_o_nao_congelado_nao(
    tmp_path: Path,
) -> None:
    """**Par causal com o MESMO ambiente**, e é a metade de baixo que impede o falso positivo.

    Uma implementação que devolvesse sempre a pasta do executável passaria numa asserção só. O que
    separa *"o ramo novo funciona"* de *"o ramo antigo morreu"* é as duas chamadas diferirem apenas
    em `congelado` e responderem coisas diferentes.
    """
    recebida = tmp_path / "pasta-que-o-usuario-recebeu"
    exe = _executavel_falso(recebida)
    ambiente = {"LOCALAPPDATA": str(tmp_path / "AppData"), "APPDATA": str(tmp_path / "AppData")}

    congelado = diretorio_de_dados(ambiente, plataforma="win32", congelado=True, executavel=exe)
    do_clone = diretorio_de_dados(ambiente, plataforma="win32", congelado=False, executavel=exe)

    assert congelado == recebida, (
        f"o congelado não caiu ao lado do executável: {congelado}. **E a pasta chama-se "
        f"«{recebida.name}» de propósito**: se o ramo acrescentasse `PASTA` — {PASTA!r} — o "
        "caminho teria um nível a mais, e uma pasta chamada `refino` esconderia isso"
    )
    assert do_clone == tmp_path / "AppData" / PASTA, do_clone
    assert do_clone != congelado, "os dois ramos responderam o mesmo: o par causal é vácuo"


def test_o_ramo_do_congelado_nao_depende_da_plataforma(tmp_path: Path) -> None:
    """`win32` e `linux` respondem o mesmo quando congelado, e é decisão declarada.

    *"Onde o usuário acha o arquivo?"* não muda com o sistema. **Sem esta asserção, um `if
    plataforma == "win32"` dentro do ramo do congelado passaria despercebido** — e a suíte corre
    também no WSL, onde ele seria o ramo tomado.
    """
    recebida = tmp_path / "recebida"
    exe = _executavel_falso(recebida)

    for plataforma in ("win32", "linux"):
        achado = diretorio_de_dados({}, plataforma=plataforma, congelado=True, executavel=exe)
        assert achado == recebida, f"em {plataforma} o congelado caiu em {achado}"


def test_o_congelado_sem_caminho_de_executavel_nomeia_a_saida() -> None:
    """O **terceiro** desfecho do ramo novo, e ele dispara por ausência.

    `sys.executable` é vazio em alguns embutidos. A mensagem tem de dar uma ação que o usuário
    consiga executar — e ela **não** cita `REFINO_DB_URL`, porque no caminho de mesa aquela variável
    não vence (ver `test_refino_db_url_nao_vence_no_caminho_de_mesa`).
    """
    with pytest.raises(BancoDeMesaError, match="congelado") as erro:
        diretorio_de_dados({}, plataforma="win32", congelado=True, executavel="   ")

    assert "compactado" in str(erro.value), f"a mensagem não diz o que fazer: {erro.value}"


def test_o_congelado_migra_o_banco_para_dentro_da_pasta_recebida(tmp_path: Path) -> None:
    """Ponta a ponta: o arquivo nasce, migra, e está **dentro** da pasta do executável.

    A asserção que importa não é o caminho — é `_tabelas`: um `prepara` que resolvesse o caminho
    novo e migrasse o antigo deixaria as duas coisas verdadeiras em arquivos diferentes.
    """
    recebida = tmp_path / "refino"
    exe = _executavel_falso(recebida)
    ambiente = {"LOCALAPPDATA": str(tmp_path / "AppData"), "APPDATA": str(tmp_path / "AppData")}

    resultado = prepara(ambiente, plataforma="win32", congelado=True, executavel=exe)

    assert resultado.caminho == recebida / NOME_DO_ARQUIVO, resultado.caminho
    assert resultado.caminho.is_file()
    assert set(METADATA.tables) <= _tabelas(resultado.caminho), (
        "o arquivo ao lado do executável não tem o esquema: a migração foi noutro lugar"
    )
    fora = tmp_path / "AppData" / PASTA / NOME_DO_ARQUIVO
    assert not fora.exists(), (
        f"o banco por usuário também foi criado, em {fora} — há dois registros, e a mudança existe "
        "para haver um"
    )


def test_a_permissao_e_fixada_antes_da_migracao_tambem_no_congelado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ORDEM `prepara_sqlite` → `migra`, afirmada por espião e não por leitura do diff.

    `prepara_sqlite` cria o arquivo e fixa o modo **antes de qualquer escrita**, e a própria
    docstring dela diz que a ordem é requisito: os auxiliares do SQLite — `-wal`, `-journal`, `-shm`
    — herdam o modo do principal, então migrar primeiro reabriria a janela em que eles nascem com
    permissão larga.

    **A mudança de lugar não podia inverter isso**, e a forma barata de o afirmar é registar a ordem
    das duas chamadas. Um teste que só olhasse o modo final não distinguiria *"fixado antes"* de
    *"fixado depois"*.
    """
    recebida = tmp_path / "recebida"
    exe = _executavel_falso(recebida)
    ordem: list[str] = []

    verdadeiro_prepara_sqlite = banco.prepara_sqlite
    verdadeiro_migra = banco.migra

    def espia_prepara_sqlite(url: str, *, congelado: bool | None = None) -> Path | None:
        # O `congelado` entrou na assinatura em 30/08/2026 e o duplo o REPASSA em vez de o
        # descartar: duplo que engole argumento nomeado esconde defeito, e aqui ele
        # silenciaria justamente o sinal que decide se a guarda de lugar corre.
        ordem.append("prepara_sqlite")
        return verdadeiro_prepara_sqlite(url, congelado=congelado)

    def espia_migra(url: str) -> None:
        ordem.append("migra")
        verdadeiro_migra(url)

    monkeypatch.setattr(banco, "prepara_sqlite", espia_prepara_sqlite)
    monkeypatch.setattr(banco, "migra", espia_migra)

    prepara({}, plataforma="win32", congelado=True, executavel=exe)

    assert ordem == ["prepara_sqlite", "migra"], (
        f"a ordem foi {ordem}; os auxiliares do SQLite herdam o modo do arquivo principal, e "
        "migrar antes de o fixar é a janela que `prepara_sqlite` existe para fechar"
    )


# ═══════════════════════════════════════ a guarda de lugar, nas TRÊS direções


def test_a_guarda_de_lugar_SALTA_no_congelado_e_continua_a_recusar_fora_dele(  # noqa: N802
    tmp_path: Path,
) -> None:
    """As três direções da guarda de lugar, e a (a) inverteu.

    ## O que a revisão adversarial mediu, e por que o alargamento não era o conserto

    A primeira versão desta unidade pôs `dist` em `DIRETORIOS_IGNORADOS`, porque medira
    `ConfiguracaoDeBancoError` para `sqlite:///<raiz>/dist/refino/refino.db`. **A medição foi feita
    na instância errada** — a partir do clone, onde `RAIZ_DO_REPOSITORIO` é de facto a raiz.

    No CONGELADO, que é o caso real, ela não é. `config.py` viaja no PYZ (medido:
    `_internal/refino/persistence/` contém apenas `migrations/`), então `__file__` ancora em
    `<exe>/_internal/refino/persistence/config.py` e `parents[3]` reancora na **pasta do
    executável**. O banco ao lado do executável fica com caminho relativo `refino.db`, cujo
    primeiro trecho não é `out` nem `dist` — e a guarda recusava.

    > **O executável não abriria, em pasta nenhuma, na máquina do usuário.** Nenhum teste
    > apanhava isso porque nenhum resolve `__file__` no mundo congelado.

    O conserto é a guarda saltar quando `sys.frozen`: ali não há repositório, não há `git add -A`,
    e ela não tem referente. Com o salto, `dist` deixou de ser necessário e saiu.
    """
    from refino.persistence.config import (
        RAIZ_DO_REPOSITORIO,
        ConfiguracaoDeBancoError,
        prepara_sqlite,
    )

    # ── (a) congelado, DENTRO da árvore -> ACEITA, porque a guarda salta ──
    #
    # É o cenário de fumo do congelamento, que corre da árvore. Antes do salto ele recusava, e o
    # executável não abria para quem o constrói.
    dentro = RAIZ_DO_REPOSITORIO / "dist" / "sonda-da-guarda-do-congelado"
    try:
        resultado = prepara(
            {},
            plataforma="win32",
            congelado=True,
            executavel=str(dentro / "refino.exe"),
            migrar=False,
        )
        assert resultado.caminho == dentro / NOME_DO_ARQUIVO, resultado.caminho
        assert resultado.caminho.is_file()
    finally:
        shutil.rmtree(dentro, ignore_errors=True)

    # ── (b) NÃO congelado, dentro da árvore e fora de `out/` -> RECUSA ──
    #
    # **É esta a direção que a guarda de lugar protege**, e ela tem de continuar de pé: um
    # `git add -A` na máquina de quem desenvolve levaria dado de paciente para o histórico. A âncora
    # contra vacuidade é a alínea (a) logo acima: o mesmo caminho, com `congelado=True`, PASSA —
    # então a recusa aqui separa os dois mundos e não é propriedade do caminho.
    proibido = RAIZ_DO_REPOSITORIO / "src" / "refino.db"
    with pytest.raises(ConfiguracaoDeBancoError) as erro:
        prepara_sqlite(f"sqlite:///{proibido.as_posix()}")
    assert "repositório" in str(erro.value), f"não é a guarda de lugar: {erro.value}"
    assert not proibido.exists(), "a guarda recusou DEPOIS de tocar em disco"

    # ── (c) congelado, FORA da árvore — a máquina do usuário -> ACEITA ──
    recebida = tmp_path / "refino"
    resultado = prepara(
        {}, plataforma="win32", congelado=True, executavel=_executavel_falso(recebida), migrar=False
    )
    assert resultado.caminho == recebida / NOME_DO_ARQUIVO, resultado.caminho
    assert resultado.caminho.is_file()


def test_rodar_do_clone_nao_congelado_vai_para_o_diretorio_por_usuario_e_nao_cai_na_guarda(
    tmp_path: Path,
) -> None:
    """A segunda metade da alínea (d) da tarefa, e ela é sobre quem DESENVOLVE.

    Correr o aplicativo do clone — `python -m refino.desktop.arranque`, não congelado — tem de
    continuar a pôr o registro no diretório por usuário, que fica **fora** da árvore. Se a mudança
    tivesse feito o não-congelado seguir `sys.executable`, o caminho seria o do interpretador —
    `.venv/Scripts/` dentro do repositório — e a guarda de lugar recusaria o arranque de quem
    desenvolve.

    **A âncora é a segunda asserção:** sem ela, um caminho por usuário que por acaso caísse dentro
    da árvore passaria por «fora dela».
    """
    from refino.persistence.config import RAIZ_DO_REPOSITORIO

    ambiente = {"XDG_DATA_HOME": str(tmp_path / "dados"), "HOME": str(tmp_path)}
    resultado = prepara(ambiente, plataforma="linux", migrar=False)

    assert resultado.caminho == tmp_path / "dados" / PASTA / NOME_DO_ARQUIVO, resultado.caminho
    with pytest.raises(ValueError):
        resultado.caminho.resolve().relative_to(RAIZ_DO_REPOSITORIO)


def test_refino_db_url_nao_vence_no_caminho_de_mesa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**ACHADO, e o teste afirma o que foi MEDIDO e não o que se esperava.**

    A tarefa pedia para confirmar que *"`REFINO_DB_URL` continua a vencer; quem define a variável
    manda"*. Medido em 30/08/2026, com a variável posta antes de chamar: **ela não vence no caminho
    de mesa**, e não é regressão desta mudança — `url_de_mesa` nunca a leu, e
    `arranque.arranca` **sobrescreve** `os.environ[REFINO_DB_URL]` com a URL de mesa logo a seguir
    a `prepara`.

    A sobrescrita é deliberada e está registrada: o comentário de `arranca` diz que sem ela *"o
    `alembic` migrava o SQLite **e as linhas de paciente iam para o outro banco**"* — o P0 que
    quatro revisores independentes acharam.

    **As duas camadas divergem, e a segunda asserção é a âncora que o prova:** em
    `persistence.config.url_do_banco`, que é o que a CLI e a API usam, a variável **vence**. Este
    teste trava a divergência para que ela seja vista, e não para a defender.
    """
    pretendida = f"sqlite:///{(tmp_path / 'escolhida-pelo-operador.db').as_posix()}"
    monkeypatch.setenv("REFINO_DB_URL", pretendida)

    recebida = tmp_path / "recebida"
    de_mesa = url_de_mesa(
        {}, plataforma="win32", congelado=True, executavel=_executavel_falso(recebida)
    )
    assert de_mesa != pretendida, (
        "`url_de_mesa` passou a respeitar `REFINO_DB_URL`. Isso pode ser a correção certa, e é "
        "mudança de comportamento: precisa de decisão registrada, não de um teste a mudar sozinho"
    )
    assert de_mesa == f"sqlite:///{(recebida / NOME_DO_ARQUIVO).as_posix()}", de_mesa

    from refino.persistence.config import url_do_banco

    assert url_do_banco() == pretendida, (
        "âncora: na camada que a CLI e a API usam, `REFINO_DB_URL` VENCE. Se esta asserção "
        "reprovar, a divergência acima deixou de ser entre duas camadas e passou a ser um defeito"
    )


# ═══════════════════════════════════════ quem lê o diretório: os TRÊS desfechos


def test_a_leitura_classifica_pelo_identificador_e_nao_pelo_nome(tmp_path: Path) -> None:
    """Par causal sobre a classificação, com o leitor injetado.

    As duas listas diferem por **um** SID — `S-1-5-32-545`, o grupo Usuários —, e é ele que a
    medição de 30/08/2026 achou na raiz `C:\\` e não achou em `Downloads`. Um classificador que
    respondesse sempre a mesma coisa reprova aqui.

    **`herdam_leitura` carrega TUDO nos dois desfechos**, e a asserção existe porque o veredito é
    heurística sobre uma lista de SIDs conhecidos: sem a evidência completa ao lado, um SID amplo
    fora da lista sairia como *"restrito"* sem deixar rastro nenhum.
    """
    do_downloads = ("S-1-5-18", "S-1-5-21-1-2-3-1003", "S-1-5-32-544")
    da_raiz = (*do_downloads, "S-1-5-32-545")

    restrito = quem_le_o_diretorio(tmp_path, plataforma="win32", leitor=lambda _: do_downloads)
    amplo = quem_le_o_diretorio(tmp_path, plataforma="win32", leitor=lambda _: da_raiz)

    assert restrito.veredito == "restrito", restrito
    assert restrito.amplos == ()
    assert restrito.herdam_leitura == do_downloads

    assert amplo.veredito == "amplo", amplo
    assert amplo.amplos == ("S-1-5-32-545",), amplo.amplos
    assert amplo.herdam_leitura == da_raiz, (
        "a evidência foi filtrada: quem lê o log deixaria de ver os SIDs que a lista não classifica"
    )
    assert "S-1-5-32-545" in SIDS_DE_MAIS_DE_UMA_CONTA, (
        "âncora: o SID que separa as duas listas tem de estar na tabela; senão o par é vácuo"
    )


def test_os_dois_caminhos_de_nao_medido_sao_distintos_de_restrito(tmp_path: Path) -> None:
    """O **terceiro** desfecho, pelas suas **duas** entradas — e ele não é o segundo.

    *"Não consegui medir"* e *"medi e ninguém mais lê"* pedem ações opostas a quem lê o log, e um
    vocabulário de dois valores colapsaria as duas no valor que tranquiliza. As duas entradas são a
    plataforma sem ACL e o leitor que levanta.
    """

    def leitor_que_levanta(_: Path) -> tuple[str, ...]:
        raise LeituraDeAclError("a sonda plantou esta falha")

    fora_do_windows = quem_le_o_diretorio(tmp_path, plataforma="linux")
    assert fora_do_windows.veredito == "nao_medido", fora_do_windows
    assert "linux" in fora_do_windows.detalhe, fora_do_windows.detalhe

    leitura_falhou = quem_le_o_diretorio(tmp_path, plataforma="win32", leitor=leitor_que_levanta)
    assert leitura_falhou.veredito == "nao_medido", leitura_falhou
    assert "plantou" in leitura_falhou.detalhe, (
        f"o detalhe não carrega a causa: {leitura_falhou.detalhe!r}"
    )
    assert leitura_falhou.herdam_leitura == () and leitura_falhou.amplos == ()


def test_refino_db_url_declarada_VENCE_no_caminho_de_mesa_e_a_de_outro_motor_e_recusada(  # noqa: N802
    tmp_path: Path,
) -> None:
    """A saída que a mensagem de erro nomeia tem de existir — e até 30/08/2026 não existia.

    ## O que o defeito era

    `config.url_do_banco` promete *"`REFINO_DB_URL` vence"*, e o cabeçalho de `desktop/banco.py`
    repete a promessa. `prepara` chamava `url_de_mesa` incondicionalmente. **A mensagem do ramo não
    congelado manda definir a variável**, e segui-la não mudava nada: mesmo erro, mesmo caminho,
    sem sinal de ter sido ignorada — o desfecho é indistinguível de a variável estar errada.

    ## A planta cai sobre dado em que as duas coisas diferem

    O caminho declarado é **fora** do lugar padrão, então *"venceu"* e *"não venceu"* produzem
    caminhos diferentes. Apontá-la para o próprio lugar de mesa passaria com as duas
    implementações.

    ## As três direções, e a terceira é a que impede a correção de trocar um silêncio por outro

    Uma URL de outro motor não pode cair num `else` razoável: `BancoDeMesa.caminho` é `Path` e o
    arranque mede a ACL dele. Ela sai por erro **nomeado**.
    """
    from refino.persistence.config import VARIAVEL_DA_URL

    escolhido = tmp_path / "escolhido-por-quem-opera" / "refino.db"
    escolhido.parent.mkdir(parents=True)

    # ── (a) declarada e de SQLite -> VENCE ──
    resultado = prepara(
        {VARIAVEL_DA_URL: f"sqlite:///{escolhido.as_posix()}"},
        plataforma="win32",
        congelado=False,
        migrar=False,
    )
    assert resultado.caminho == escolhido, resultado.caminho
    assert escolhido.is_file(), "venceu no nome e não em disco"

    # ── (b) NÃO declarada -> o lugar de mesa, e é OUTRO ──
    #
    # A âncora contra vacuidade da alínea (a): sem ela, um `prepara` que ignorasse a variável
    # poderia coincidir com o caminho declarado e (a) passaria por acaso.
    padrao = prepara(
        {"LOCALAPPDATA": str(tmp_path / "appdata")},
        plataforma="win32",
        congelado=False,
        migrar=False,
    )
    assert padrao.caminho != escolhido, "as duas direções não separam nada"

    # ── (c) declarada e de outro motor -> recusa NOMEADA ──
    with pytest.raises(BancoDeMesaError) as erro:
        prepara(
            {VARIAVEL_DA_URL: "postgresql+psycopg://u:senha@host/refino"},
            plataforma="win32",
            congelado=False,
            migrar=False,
        )
    assert VARIAVEL_DA_URL in str(erro.value), erro.value
    assert "SQLite" in str(erro.value), erro.value
    # E a senha não viaja na mensagem — `config.py` já declara que a URL pode carregá-la.
    assert "senha" not in str(erro.value), f"a mensagem carrega credencial: {erro.value}"


def test_a_frase_do_restrito_nao_afirma_quem_le_alem_do_que_foi_medido() -> None:
    """O par causal na prosa: SID DESCONHECIDO continua `restrito`, e a frase não mente.

    ## A planta cai sobre dado em que as duas coisas diferem

    O duplo devolve os três SIDs esperados **mais um que a lista negativa não conhece** — a forma
    de um grupo criado pelo setor de informática de uma instituição. As duas quantidades que este
    teste separa **divergem sobre esse dado**: o veredito continua `restrito`, e o número de
    leitores medidos passa de três para quatro. Sobre os três esperados sozinhos, a frase antiga e
    a nova seriam ambas defensáveis, e o teste seria vácuo.

    ## Por que a asserção é de AUSÊNCIA **e** de presença

    *"a frase não diz «e mais ninguém»"* passa por vacuidade se a frase vier vazia. A âncora de
    não-vacuidade é a contagem: ela tem de aparecer no texto, e tem de ser **4**.
    """
    tres_e_um_desconhecido = (
        "S-1-5-18",
        "S-1-5-32-544",
        "S-1-5-21-4259856341-1257019974-363056593-1003",
        "S-1-5-21-4259856341-1257019974-363056593-9999",
    )
    leitura = quem_le_o_diretorio(
        Path("C:/qualquer"), plataforma="win32", leitor=lambda _: tres_e_um_desconhecido
    )

    # A classificação NÃO muda — é isso que torna a frase o único lugar onde o defeito podia estar.
    assert leitura.veredito == "restrito", leitura
    assert leitura.amplos == (), leitura.amplos

    frase = leitura.para_quem_opera()

    # Não-vacuidade, e ela é a contagem: a frase fala de QUATRO, que é o que se mediu.
    assert "4 entrada(s)" in frase, frase

    # E a ausência: nenhuma forma de exaustividade sobre quem lê.
    for exaustiva in ("só a sua conta", "e mais ninguém", "apenas a sua conta"):
        assert exaustiva not in frase, f"exaustividade não medida: {exaustiva!r} em {frase!r}"


def test_a_frase_para_quem_opera_difere_nos_tres_desfechos(tmp_path: Path) -> None:
    """A frase que vai para o `refino.log`, nos três casos — e as três têm de ser **diferentes**.

    Um `para_quem_opera` que devolvesse a mesma coisa sempre passaria em qualquer asserção de
    substring tomada isoladamente. O que separa é as três serem distintas **e** a do `amplo` trazer
    a ação: fechar, mover, abrir de lá.
    """
    amplo = quem_le_o_diretorio(
        tmp_path, plataforma="win32", leitor=lambda _: ("S-1-1-0",)
    ).para_quem_opera()
    restrito = quem_le_o_diretorio(
        tmp_path, plataforma="win32", leitor=lambda _: ("S-1-5-18",)
    ).para_quem_opera()
    nao_medido = quem_le_o_diretorio(tmp_path, plataforma="linux").para_quem_opera()

    assert len({amplo, restrito, nao_medido}) == 3, "duas das três frases são iguais"
    assert "ATENÇÃO" in amplo and "mova a pasta" in amplo, amplo
    assert "ATENÇÃO" not in restrito, f"o desfecho tranquilo está a alarmar: {restrito}"
    for frase in (amplo, restrito, nao_medido):
        assert str(tmp_path) in frase, f"a frase não nomeia o diretório: {frase}"


def test_a_leitura_real_mede_o_que_esta_plataforma_tem() -> None:
    """O leitor **de verdade**, contra as âncoras de 30/08/2026 — e sem `skipif`.

    **A ausência do `skipif` é deliberada, e é a mesma escolha de `test_o_arquivo_nasce_em_0600`.**
    Um `skip` fora do Windows nunca dispararia na máquina de quem desenvolveu isto, e ramo que nunca
    dispara é ramo inerte. Aqui cada plataforma afirma o que ela tem: no Windows, a medição; fora
    dele, que a função **recusa** medir em vez de devolver vazio.

    **A população das âncoras é «os diretórios desta máquina, 30/08/2026»**, e não «o Windows».
    Uma divergência aqui é achado sobre a máquina em que a suíte corre — uma raiz `C:` endurecida,
    um perfil com ACL própria — e não defeito do leitor. O que seria defeito do leitor é a linha de
    baixo: uma lista **vazia** sobre um diretório que existe.
    """
    if sys.platform != "win32":
        with pytest.raises(LeituraDeAclError, match="Windows"):
            banco._sids_com_leitura_herdavel(Path.home())
        assert quem_le_o_diretorio(Path.home()).veredito == "nao_medido"
        return

    import os

    pessoal = Path(os.environ["LOCALAPPDATA"])
    raiz = Path(os.environ.get("SYSTEMDRIVE", "C:") + "\\")

    sids = banco._sids_com_leitura_herdavel(pessoal)
    assert sids, (
        f"o leitor devolveu lista VAZIA para {pessoal}, que existe. Vazio aqui produziria o "
        "veredito «restrito» sobre uma medição que não aconteceu"
    )
    assert all(s.startswith("S-1-") for s in sids), sids

    assert quem_le_o_diretorio(pessoal).veredito == "restrito", (
        f"%LOCALAPPDATA% saiu amplo: {quem_le_o_diretorio(pessoal)}"
    )
    da_raiz = quem_le_o_diretorio(raiz)
    assert da_raiz.veredito == "amplo", (
        f"a raiz {raiz} saiu restrita: {da_raiz}. É a âncora do lado AMPLO — sem ela, um leitor "
        "que nunca classificasse nada como amplo passaria na asserção de cima"
    )


def test_a_leitura_real_ignora_negacao_e_ignora_escrita_sem_leitura(tmp_path: Path) -> None:
    """**As duas plantas que mudariam o veredito se os filtros estivessem errados.**

    O leitor conta uma entrada quando ela (a) PERMITE, (b) desce para arquivos, e (c) concede
    leitura. As duas primeiras condições são fáceis de escrever ao contrário, e nenhuma reprovaria
    numa asserção sobre `%LOCALAPPDATA%`:

    | Planta | O que o filtro certo faz | O que um filtro errado diria |
    |---|---|---|
    | NEGAR leitura a `S-1-1-0` | ignora: negação tira acesso | *"Todos leem"* — alarme falso |
    | dar só ESCRITA a `S-1-5-32-545` | ignora: escrever não é ler | *"o grupo Usuários lê"* |

    **E a âncora é a terceira parte:** o mesmo `S-1-5-32-545`, com leitura de verdade, **tem** de
    sair `amplo`. Sem ela, um leitor que nunca achasse nada passaria nas duas primeiras.

    Medido em 30/08/2026 com `icacls` sobre diretórios temporários desta máquina. Fora do Windows o
    teste afirma a **recusa** de medir, que é o desfecho correto lá.
    """
    if sys.platform != "win32":
        with pytest.raises(LeituraDeAclError, match="Windows"):
            banco._sids_com_leitura_herdavel(tmp_path)
        return

    import subprocess

    def icacls(diretorio: Path, *argumentos: str) -> None:
        feito = subprocess.run(
            ["icacls", str(diretorio), *argumentos], capture_output=True, text=True, check=False
        )
        assert feito.returncode == 0, (
            f"âncora: `icacls {' '.join(argumentos)}` falhou ({feito.returncode}); sem a planta "
            f"aterrada a asserção abaixo não mede nada: {feito.stdout}{feito.stderr}"
        )

    # ── (a) negação de leitura a «Todos», e escrita sem leitura ao grupo Usuários ──
    plantado = tmp_path / "com-plantas"
    plantado.mkdir()
    icacls(plantado, "/grant", "*S-1-5-32-545:(OI)(CI)(W)")
    icacls(plantado, "/deny", "*S-1-1-0:(OI)(CI)(RD)")

    leitura = quem_le_o_diretorio(plantado)
    # **A não-vacuidade vem PRIMEIRO, e na mesma função.** As duas asserções seguintes são de
    # AUSÊNCIA, e ausência é satisfeita por conjunto vazio — que é o que uma leitura falhada
    # devolveria. Sem esta linha, elas passariam sobre uma medição que não aconteceu.
    assert leitura.herdam_leitura, (
        f"o leitor não achou entrada nenhuma em {plantado}: as asserções de ausência abaixo "
        "passariam por vacuidade"
    )
    assert "S-1-1-0" not in leitura.herdam_leitura, (
        f"uma entrada de NEGAÇÃO foi contada como quem lê: {leitura.herdam_leitura}"
    )
    assert "S-1-5-32-545" not in leitura.herdam_leitura, (
        f"uma entrada de ESCRITA sem leitura foi contada como quem lê: {leitura.herdam_leitura}"
    )
    assert leitura.veredito == "restrito", leitura

    # ── (b) âncora: o MESMO SID, com leitura de verdade, tem de sair amplo ──
    ancora = tmp_path / "com-leitura-de-verdade"
    ancora.mkdir()
    icacls(ancora, "/grant", "*S-1-5-32-545:(OI)(CI)(R)")

    com_leitura = quem_le_o_diretorio(ancora)
    assert com_leitura.veredito == "amplo", (
        f"âncora: com leitura concedida de facto o veredito tem de virar: {com_leitura}"
    )
    assert com_leitura.amplos == ("S-1-5-32-545",), com_leitura.amplos


def test_o_leitor_real_recusa_um_diretorio_que_nao_existe(tmp_path: Path) -> None:
    """O caminho de erro que produz `nao_medido`, com a API de verdade e não com um duplo.

    `test_os_dois_caminhos_de_nao_medido_sao_distintos_de_restrito` planta a exceção; este mostra
    que ela **acontece** — e é a única forma de saber que o tipo que `quem_le_o_diretorio` apanha é
    o tipo que o leitor levanta.
    """
    ausente = tmp_path / "esta-pasta-nunca-existiu"
    assert not ausente.exists(), "âncora: a pasta não pode existir"

    if sys.platform != "win32":
        with pytest.raises(LeituraDeAclError, match="Windows"):
            banco._sids_com_leitura_herdavel(ausente)
    else:
        with pytest.raises(LeituraDeAclError, match="GetNamedSecurityInfoW"):
            banco._sids_com_leitura_herdavel(ausente)

    assert quem_le_o_diretorio(ausente).veredito == "nao_medido", (
        "o erro do leitor não virou `nao_medido` — ele escapou, ou virou `restrito`"
    )

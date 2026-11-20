"""A persistência sobre SQLite: a configuração, o arquivo, a migração e a gravação.

O banco do produto é um arquivo SQLite, declarado em `REFINO_DB_URL`. Estes testes conferem
que a configuração aceita e recusa o que deve, que o arquivo nasce com permissão restrita, que
a migração corre nas duas direções e que a gravação e a leitura fecham.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import pytest
from sqlalchemy import create_engine, inspect, text

from refino.domain.diagnosis import DefectProfile
from refino.domain.spec import REQUIRED_METRIC_FIELDS
from refino.persistence.config import (
    VARIAVEL_DA_URL,
    ConfiguracaoDeBancoError,
    caminho_do_sqlite,
    prepara_sqlite,
    url_do_banco,
)
from refino.persistence.repositories import SEM_SEMENTE, RunRepository, SobrescritaExercida
from refino.persistence.schema import METADATA
from tests.conftest import REPO
from tests.permissao_capability import DETALHE_DO_MODO, modo_de_arquivo_respeitado
from tests.reclamacao import reclama

# As nove tabelas da especificação. Derivado de `METADATA` e não redigitado: tabela nova entra
# sozinha, e uma contagem escrita à mão divergiria no dia em que a décima nascer.
TABELAS_DA_SPEC: Final[frozenset[str]] = frozenset(METADATA.tables)

PERFIL = DefectProfile(
    defect_type="erratic",
    confidence=0.61,
    suggested_batches=(0, 0, 1),
    changepoints=(45,),
    qc_rsd_median=0.3,
    qc_rsd_p75=0.4,
    zero_fraction=0.158,
    missing_fraction=0.0,
    pc1_injection_correlation=0.29,
    qc_dispersion_ratio=0.93,
    n_features=6035,
    n_samples=68,
    n_qcs=7,
)


def _alembic(*args: str, url: str) -> subprocess.CompletedProcess[str]:
    """Roda o Alembic por SUBPROCESSO com a URL injetada, como o usuário roda.

    Subprocesso e não a API em processo pelo motivo de `tests/integration/test_migrations.py`:
    em processo partilha o `sys.modules` do pytest e testa um caminho que ninguém executa.

    **O ambiente é COPIADO e a variável entra só na cópia.** Exportá-la no processo do pytest
    vazaria para os testes seguintes, e a ordem de execução passaria a decidir o resultado.
    """
    ambiente = dict(os.environ)
    ambiente[VARIAVEL_DA_URL] = url
    return subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=ambiente,
    )


@pytest.fixture
def url_sqlite(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'refino.db'}"


@contextlib.contextmanager
def monkeypatch_ambiente(nome: str, valor: str) -> Iterator[None]:
    """Define uma variável e a **restaura**, incluindo quando o corpo levanta.

    Gerenciador de contexto e não `monkeypatch` porque o corpo é um `pytest.raises`: com a
    fixture, a restauração acontece no fim do teste, e um teste que reprovasse a meio deixaria
    `REFINO_DB_URL` exportada para os seguintes — a ordem de execução passaria a decidir o
    resultado, que é a forma mais cara de portão intermitente.
    """
    anterior = os.environ.get(nome)
    os.environ[nome] = valor
    try:
        yield
    finally:
        if anterior is None:
            os.environ.pop(nome, None)
        else:
            os.environ[nome] = anterior


# ---------------------------------------------------------------------------
# A configuração — o que ela aceita, o que recusa, e o que NÃO diz
# ---------------------------------------------------------------------------


URLS_DE_SQLITE: Final[tuple[str, ...]] = (
    "sqlite:///out/refino.db",
    "sqlite:////var/lib/refino/refino.db",
    "sqlite:///:memory:",
    "sqlite+pysqlite:///out/refino.db",
)


@pytest.mark.parametrize("url", URLS_DE_SQLITE)
def test_a_url_de_sqlite_declarada_sai_como_foi_escrita(url: str) -> None:
    """A URL de SQLite em `REFINO_DB_URL` é devolvida sem alteração."""
    assert url_do_banco({VARIAVEL_DA_URL: url}) == url
    assert url_do_banco({VARIAVEL_DA_URL: f"  {url}  "}) == url, "espaço nas pontas é aparado"


@pytest.mark.parametrize("ambiente", [{}, {VARIAVEL_DA_URL: ""}, {VARIAVEL_DA_URL: "   "}])
def test_sem_url_declarada_levanta_nomeando_a_variavel(ambiente: dict[str, str]) -> None:
    """Sem `REFINO_DB_URL`, ou com ela em branco, a recusa nomeia a variável e dá o exemplo."""
    with pytest.raises(ConfiguracaoDeBancoError) as erro:
        url_do_banco(ambiente)

    texto = str(erro.value)
    assert VARIAVEL_DA_URL in texto
    assert "sqlite:///" in texto, "a mensagem não mostra a forma da URL que resolve"


VARIAVEIS_DO_POSTGRES: Final[dict[str, str]] = {
    "REFINO_DB_HOST": "10.0.0.5",
    "REFINO_DB_PORT": "5432",
    "REFINO_DB_NAME": "refino",
    "REFINO_DB_USER": "refino",
    "REFINO_DB_PASSWORD": "SenhaQueNaoPodeVazar123",
}


def test_as_variaveis_do_postgres_nao_montam_url_nenhuma() -> None:
    """Sem `REFINO_DB_URL`, as variáveis `REFINO_DB_*` não formam URL: a recusa é a mesma."""
    with pytest.raises(ConfiguracaoDeBancoError) as sem_nada:
        url_do_banco({})
    with pytest.raises(ConfiguracaoDeBancoError) as com_host:
        url_do_banco(dict(VARIAVEIS_DO_POSTGRES))

    assert str(com_host.value) == str(sem_nada.value), "as variáveis antigas mudaram a recusa"
    texto = str(com_host.value)
    assert VARIAVEL_DA_URL in texto, "âncora: a mensagem nomeia a configuração"
    for proibido in ("postgresql", "10.0.0.5", "SenhaQueNaoPodeVazar123", "docker", "compose"):
        assert proibido not in texto, f"a recusa cita {proibido!r}"


def test_as_variaveis_do_postgres_nao_mudam_a_url_declarada() -> None:
    """Par causal: com e sem as variáveis `REFINO_DB_*`, a URL declarada sai igual."""
    declarada = {VARIAVEL_DA_URL: "sqlite:///out/refino.db"}

    assert url_do_banco(declarada) == "sqlite:///out/refino.db"
    assert url_do_banco({**VARIAVEIS_DO_POSTGRES, **declarada}) == "sqlite:///out/refino.db"


def test_a_mensagem_de_erro_nao_ecoa_a_url_porque_ela_carrega_a_senha() -> None:
    """A exceção chega à TELA por `ui/cliente.py`, e uma URL de Postgres traz a senha em linha.

    Par causal com uma URL válida: o mesmo caminho, e a diferença é só o dialeto ser recusado.
    """
    senha = "SenhaQueNaoPodeVazar123"
    invalida = f"mysql+pymysql://refino:{senha}@10.0.0.5/refino"

    with pytest.raises(ConfiguracaoDeBancoError) as erro:
        url_do_banco({VARIAVEL_DA_URL: invalida})

    texto = str(erro.value)
    assert VARIAVEL_DA_URL in texto, "âncora: a mensagem nomeia a configuração"
    assert "mysql" in texto, "âncora: ela diz qual dialeto foi recusado"
    assert senha not in texto, "a senha vazou para o texto da exceção, que chega à tela"
    assert invalida not in texto


def test_url_sem_esquema_e_recusada() -> None:
    """`/caminho/para/banco.db` é o engano natural de quem espera um caminho e não uma URL."""
    with pytest.raises(ConfiguracaoDeBancoError, match=VARIAVEL_DA_URL):
        url_do_banco({VARIAVEL_DA_URL: "/home/eu/refino.db"})


@pytest.mark.parametrize(
    ("url", "esperado"),
    [
        ("sqlite:///out/refino.db", "out/refino.db"),
        ("sqlite:////var/lib/refino.db", "/var/lib/refino.db"),
        ("postgresql+psycopg2://u:p@h/db", None),
        ("sqlite://", None),
        # **A quinta forma fecha o ramo `Path(resto) if resto`** As quatro
        # acima entram todas pelo ramo da barra ou pelo `None`, então a segunda metade da última
        # linha de `caminho_do_sqlite` nunca corria. E `coverage` não conta expressão condicional
        # como ramo, logo os 91,53% não a cobriam: só um caso a alcança.
        ("sqlite://relativo.db", "relativo.db"),
        # Prefixo de driver: o dialeto é `sqlite` na mesma, e o caminho tem de sair igual.
        ("sqlite+pysqlite:///out/refino.db", "out/refino.db"),
    ],
)
def test_o_caminho_do_arquivo_sai_da_url(url: str, esperado: str | None) -> None:
    """As quatro formas, e a terceira barra é onde se erra.

    `None` significa *"não há arquivo"* e não *"não sei"*: para Postgres não há o que preparar,
    e colapsar os dois num caminho vazio faria a preparação criar um arquivo chamado `.` no
    diretório de trabalho.
    """
    achado = caminho_do_sqlite(url)

    assert achado == (None if esperado is None else Path(esperado))


# ---------------------------------------------------------------------------
# O arquivo — permissão e ignore, porque o conteúdo é DADO CLÍNICO
# ---------------------------------------------------------------------------


def test_o_arquivo_nasce_com_modo_0600(tmp_path: Path) -> None:
    """As tabelas guardam `samples.name` e `samples.grupo` — pseudónimo mais diagnóstico.

    O `sqlite3` criaria o arquivo com o `umask` do processo, que em máquina de trabalho dá
    `0644`. `prepara_sqlite` cria e fixa o modo **antes** de o motor abrir, o que fecha a janela
    em que o dado clínico existe legível por qualquer conta local.
    """
    if not modo_de_arquivo_respeitado():
        pytest.skip(f"permissão POSIX ignorada: {DETALHE_DO_MODO}")

    alvo = tmp_path / "sub" / "refino.db"
    caminho = prepara_sqlite(f"sqlite:///{alvo}")

    assert caminho == alvo, "âncora: a função devolve o caminho que preparou"
    assert alvo.is_file(), "o arquivo não foi criado, então o modo não foi fixado em nada"
    assert alvo.stat().st_mode & 0o777 == 0o600, oct(alvo.stat().st_mode & 0o777)


def test_preparar_so_toca_o_disco_quando_ha_arquivo(tmp_path: Path) -> None:
    """Par causal **na mesma função**: a URL de SQLite cria, as outras duas não.

    A âncora positiva fica aqui e não num teste vizinho porque `assert depois == antes` é
    satisfeito por uma implementação que nunca faz nada — e o vizinho pode ser removido sem que
    nada ligue os dois.

    A terceira forma é `:memory:`, que **é** SQLite e mesmo assim não tem arquivo: sem o ramo,
    um arquivo literalmente chamado `:memory:` nasceria no diretório de trabalho e ficaria lá.
    """
    antes = set(tmp_path.iterdir())

    criado = prepara_sqlite(f"sqlite:///{tmp_path / 'existe.db'}")
    assert criado is not None and criado.is_file(), "âncora: com SQLite ele cria de facto"

    assert prepara_sqlite("postgresql+psycopg2://u:p@h/db") is None
    assert prepara_sqlite("sqlite:///:memory:") is None, "`:memory:` não tem arquivo a preparar"

    assert set(tmp_path.iterdir()) - antes == {criado}, (
        "alguma das duas formas sem arquivo tocou o disco"
    )


def test_o_caminho_do_banco_em_arquivo_e_ignorado_pelo_git() -> None:
    """Um `git add -A` não pode comprometer dado de paciente no histórico.

    **Âncora de não-vacuidade primeiro**: o conjunto de candidatos não é vazio. Sem ela, uma
    lista vazia faria `all(...)` passar afirmando que o git ignora tudo — a asserção de ausência
    satisfeita por vacuidade.
    """
    candidatos = ["out/refino.db", "refino.db", "dados/estudo.sqlite", "x.sqlite3"]

    assert candidatos, "sem candidatos, a asserção abaixo é vácua"

    nao_ignorados = [
        alvo
        for alvo in candidatos
        if subprocess.run(["git", "check-ignore", "-q", alvo], cwd=REPO, check=False).returncode
        != 0
    ]

    assert not nao_ignorados, (
        f"o git versionaria estes caminhos de banco: {nao_ignorados}. As tabelas guardam o par "
        "pseudónimo/diagnóstico, e o repositório é aberto por plano."
    )


def test_o_negativo_do_ignore_acusa_um_caminho_que_o_git_versiona() -> None:
    """O par causal do anterior — ele próprio é um mecanismo e podia ficar inerte.

    Se `git check-ignore` devolvesse zero para tudo, o teste acima passaria sem verificar nada.
    """
    versionado = subprocess.run(
        ["git", "check-ignore", "-q", "README.md"], cwd=REPO, check=False
    ).returncode

    assert versionado != 0, "`git check-ignore` diz que o README é ignorado — o arnês está cego"


# ---------------------------------------------------------------------------
# O esquema e a migração, sobre SQLite de verdade
# ---------------------------------------------------------------------------


def test_o_esquema_sobe_em_sqlite_com_as_nove_tabelas(url_sqlite: str) -> None:
    """`METADATA.create_all()` não precisa de alteração nenhuma nos modelos."""
    prepara_sqlite(url_sqlite)
    motor = create_engine(url_sqlite)
    METADATA.create_all(motor)

    achadas = set(inspect(motor).get_table_names())

    assert len(TABELAS_DA_SPEC) == 9, (
        f"a §6.2 declara nove tabelas; METADATA tem {len(TABELAS_DA_SPEC)}"
    )
    assert achadas >= TABELAS_DA_SPEC, sorted(TABELAS_DA_SPEC - achadas)


def _revisao_que_introduz(coluna: str) -> tuple[str | None, str | None]:
    """`(revisão que acrescenta a coluna, a `down_revision` dela)`, lidas dos arquivos.

    **Derivado e não pinado**: um literal aqui seria o retrato de hoje, e é
    exactamente assim que o `-1` desta suíte envelheceu — ele apontava para a revisão certa
    enquanto ela foi a `head`, e continuou a apontar para «a última» depois de não o ser.
    """
    versoes = REPO / "src" / "refino" / "persistence" / "migrations" / "versions"
    achadas = []
    for arquivo in sorted(versoes.glob("*.py")):
        texto = arquivo.read_text(encoding="utf-8")
        if f'"{coluna}"' not in texto:
            continue
        rev = re.search(r'^revision: str = "([^"]+)"', texto, re.M)
        abaixo = re.search(r'^down_revision: str \| None = "([^"]+)"', texto, re.M)
        if rev and abaixo and "add_column" in texto:
            achadas.append((rev.group(1), abaixo.group(1)))
    # **Uma e só uma**: `count != 1` em vez de `in`. Duas revisões a
    # acrescentar a mesma coluna significaria que o esquema a perdeu pelo caminho, e o alvo da
    # descida deixaria de ser único.
    return achadas[0] if len(achadas) == 1 else (None, None)


_REVISAO_DAS_COLUNAS, _REVISAO_ABAIXO_DAS_COLUNAS = _revisao_que_introduz("campos_sobrescritos")


def test_alembic_sobe_e_desce_em_sqlite(url_sqlite: str) -> None:
    """O ciclo completo, e a DESCIDA entra porque o critério nasceu de uma que nunca correu.

    A revisão `2d12e7fb4974` foi reescrita em `batch_alter_table` nesta unidade. Antes disso,
    medido em 16/08/2026, o `upgrade` reprovava em SQLite com `NotImplementedError: No support
    for ALTER of constraints in SQLite dialect` — e a reprovação era em
    `op.create_check_constraint`, não no `alter_column` que o plano previa.
    """
    subida = _alembic("upgrade", "head", url=url_sqlite)
    assert subida.returncode == 0, subida.stderr[-2000:]

    motor = create_engine(url_sqlite)
    colunas = {c["name"] for c in inspect(motor).get_columns("decisions")}
    assert {"apresentacao", "campos_sobrescritos"} <= colunas, sorted(colunas)

    # **O alvo da descida é DERIVADO e não `-1`, e o `-1` era o defeito.** Quando este teste foi
    # escrito, `2d12e7fb4974` era a `head`, e `-1` apontava para ela. Duas revisões entraram depois
    # — `a3f7c1b90e42` e `c8b1e64f27a9` —, e `-1` passou a desfazer a ÚLTIMA, que não mexe nestas
    # colunas. A asserção seguinte reprovava com *"a descida não removeu as colunas"*, e a
    # `docs/linha-de-base-windows.md` registava a causa como *"`alembic` em SQLite"*: o veredito
    # acusava o dialeto, e o defeito era do alvo. **É o referente aplicado ao alvo de
    # um teste** — mudança feita, referente parado —, e o SQLite desta máquina é 3.53.1, que suporta
    # `DROP COLUMN` desde a 3.35.
    #
    # **Derivar em vez de pinar**: a revisão sob teste é a que introduz as colunas, e o alvo é
    # a `down_revision` dela. Uma revisão nova no topo não mexe mais neste número.
    assert _REVISAO_DAS_COLUNAS is not None, (
        "não se achou a revisão que introduz `apresentacao`/`campos_sobrescritos` — sem ela o "
        "alvo da descida seria adivinhado, que é o defeito que este comentário registra"
    )
    descida = _alembic("downgrade", _REVISAO_ABAIXO_DAS_COLUNAS, url=url_sqlite)
    assert descida.returncode == 0, descida.stderr[-2000:]

    motor.dispose()
    motor = create_engine(url_sqlite)
    depois = {c["name"] for c in inspect(motor).get_columns("decisions")}
    assert not ({"apresentacao", "campos_sobrescritos"} & depois), (
        "a descida não removeu as colunas: o par de colunas fica meio aplicado, e o `upgrade` "
        "seguinte é no-op sobre estado herdado"
    )


def test_o_alembic_em_sqlite_chega_a_revisao_de_head(url_sqlite: str) -> None:
    """O `version_num` gravado é o `head` que o próprio Alembic declara.

    Comparar contra o `head` declarado, e não contra um literal escrito aqui, é o que impede
    este teste de caducar na próxima revisão — e é a mesma forma de derivar em vez de redigitar
    que `TABELAS_DA_SPEC` usa.
    """
    cabecas = _alembic("heads", url=url_sqlite)
    assert cabecas.returncode == 0, cabecas.stderr[-2000:]
    declarada = cabecas.stdout.split()[0]
    assert declarada, "o Alembic não declarou head nenhum; o arnês não leu o que pensa que leu"

    assert _alembic("upgrade", "head", url=url_sqlite).returncode == 0

    motor = create_engine(url_sqlite)
    with motor.connect() as conexao:
        gravada = conexao.execute(text("SELECT version_num FROM alembic_version")).scalar_one()

    assert gravada == declarada, f"gravado {gravada!r}, declarado {declarada!r}"


# ---------------------------------------------------------------------------
# A gravação — o que a unidade existe para permitir
# ---------------------------------------------------------------------------


def _grava(repo: RunRepository, **kw: Any) -> Any:
    base: dict[str, Any] = {
        "estudo": "ATLL/HAM-TSP",
        "mode": "pos_hilic",
        "checksum_entrada": "a" * 64,
        "n_samples": 68,
        "n_features": 6035,
        "amostras": [{"name": "QC001", "injection_order": 1, "sample_type": "qc", "grupo": "QC"}],
        "perfil": PERFIL,
        "spec_json": json.dumps({"correction": "serrf", "n_batches": 1}),
        "seed": 42,
        "rationale": "Recomendado `serrf` por política constante.",
        "origem": "modelo",
        "metrics_json": '{"qc_rsd_median": 0.3}',
        "apresentacao": "exibida",
    }
    base.update(kw)
    return repo.grava_execucao(**base)


def test_grava_e_reconstroi_a_concordancia_em_sqlite(url_sqlite: str) -> None:
    """R23 fechado: gravar e ler o par funciona sem contêiner nenhum de pé.

    **A asserção passa por `classifica_concordancia`, que LÊ `decisions`**, e não pelo retorno
    de `grava_execucao`. É a forma de arnês que o teste do par já usa: um valor que
    a função acabou de compor provaria consistência consigo mesma e nada sobre o banco.
    """
    assert _alembic("upgrade", "head", url=url_sqlite).returncode == 0
    repo = RunRepository(create_engine(url_sqlite))

    gravada = _grava(repo)

    estado = repo.classifica_concordancia(gravada.matrix_id, gravada.diagnosis_id)
    assert estado == "nao_sobrescreveu", estado


def test_a_escolha_da_politica_e_gravada_como_par_ao_lado_da_sobrescrita(url_sqlite: str) -> None:
    """A escolha automática ocupa a linha A do par, e a do humano a linha B.

    ## O que isto verifica, e por que precisa do banco

    O critério pede que a sobrescrita seja persistida **com o par**. A pergunta que esta unidade
    abre é a linha A: quando quem propôs foi a **política** — e não a heurística —, o que fica
    gravado é o argumento dela, com as características numéricas que levaram ao corretor.

    A tabela `decisions` **não tem unicidade em `(matrix_id, diagnosis_id)`**, e é isso que faz o
    par caber sem migração: as duas linhas partilham os dois identificadores. Ler isso do banco em
    vez de do retorno de `grava_execucao` é a forma de arnês que os testes vizinhos já
    usam — um valor que a função acabou de compor provaria consistência consigo mesma.

    ## O par causal, e ele cai onde as duas linhas DIFEREM

    As duas trazem o mesmo `matrix_id` e o mesmo `diagnosis_id`; o que as separa é `origem` e
    `rationale`. Afirmar só a contagem seria satisfeito por duas linhas idênticas, que é o defeito
    que faria a concordância medida não medir nada.
    """
    assert _alembic("upgrade", "head", url=url_sqlite).returncode == 0
    motor = create_engine(url_sqlite)
    repo = RunRepository(motor)

    justificativa_da_politica = (
        "Recomendado `serrf` com confiança de 0.97. **Por quê:** "
        "`pc1_injection_correlation` = 0.2415 ≤ 0.4744."
    )
    gravada = _grava(
        repo,
        rationale="Escolha do pesquisador. Eixos sobrescritos: ['correction'].",
        origem="manual",
        sobrescrita=SobrescritaExercida(
            spec_proposta_json=json.dumps({"correction": "loess", "n_batches": 1}),
            rationale_proposta=justificativa_da_politica,
            origem_proposta="modelo",
            campos=("correction",),
        ),
    )

    with motor.connect() as c:
        linhas = [
            dict(r._mapping)
            for r in c.execute(
                text(
                    "SELECT matrix_id, diagnosis_id, origem, rationale, campos_sobrescritos "
                    "FROM decisions ORDER BY id"
                )
            )
        ]

    assert len(linhas) == 2, (
        f"a política propôs e o humano sobrescreveu, e há {len(linhas)} linha(s)"
    )

    proposta = next(linha for linha in linhas if linha["origem"] != "manual")
    escolhida = next(linha for linha in linhas if linha["origem"] == "manual")

    assert proposta["rationale"] == justificativa_da_politica, (
        "a linha da máquina não carrega o argumento da política: sem ele o par regista QUE houve "
        "discordância e não COM O QUÊ se discordou"
    )
    assert escolhida["rationale"] != proposta["rationale"], (
        "as duas linhas do par têm a mesma justificativa, e então elas não separam nada"
    )
    assert proposta["origem"] == "modelo"

    # **A propriedade que faz o par caber sem migração**, e ela é afirmada e não pressuposta.
    assert proposta["matrix_id"] == escolhida["matrix_id"]
    assert proposta["diagnosis_id"] == escolhida["diagnosis_id"]

    assert repo.classifica_concordancia(gravada.matrix_id, gravada.diagnosis_id) == (
        "sobrescreveu_discordando"
    )


def test_sem_sobrescrita_a_delegacao_grava_uma_linha_com_a_justificativa_da_politica(
    url_sqlite: str,
) -> None:
    """**A outra metade do par causal: delegar e NÃO sobrescrever grava uma linha só.**

    Sem esta direção, uma implementação que gravasse o par sempre passaria no teste acima e faria
    *"concordância"* contar como discordância todo caso em que o humano aceitou a proposta.

    E o que a linha única carrega é a justificativa da **política** — não a frase sobre o gesto do
    humano. É o critério na camada de persistência: a decisão automática fica registada com o
    argumento que a produziu.
    """
    assert _alembic("upgrade", "head", url=url_sqlite).returncode == 0
    motor = create_engine(url_sqlite)
    repo = RunRepository(motor)

    justificativa_da_politica = "Recomendado `serrf` por política constante."
    gravada = _grava(repo, rationale=justificativa_da_politica, origem="modelo")

    with motor.connect() as c:
        linhas = [
            dict(r._mapping)
            for r in c.execute(text("SELECT origem, rationale FROM decisions ORDER BY id"))
        ]

    assert len(linhas) == 1, f"ninguém sobrescreveu e o banco tem {len(linhas)} linha(s)"
    assert linhas[0]["origem"] == "modelo"
    assert linhas[0]["rationale"] == justificativa_da_politica
    assert repo.classifica_concordancia(gravada.matrix_id, gravada.diagnosis_id) == (
        "nao_sobrescreveu"
    )


# ---------------------------------------------------------------------------
# O `Verification` da unidade, pela ROTA que o botão de facto chama
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_a_rota_de_execucao_grava_em_sqlite_sem_conteiner_nenhum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R23 fechado onde ele é afirmado: *"o botão grava e devolve o estado de concordância"*.

    **Os testes acima param na camada do repositório, e essa distância era a lacuna.** Eles
    provam que `RunRepository` fala com SQLite; o que a unidade promete é que a **rota** o faz —
    e entre as duas está `_repositorio`, que é justamente a função cuja guarda moveu. Um teste
    que parasse no repositório deixaria o `503` por falta de `REFINO_DB_HOST` intacto e verde,
    porque nunca passaria por ali.

    `TestClient` fala com a aplicação ASGI **em processo**: não abre porta, não sobe `uvicorn` e
    não precisa de contêiner — que é exactamente a condição que a unidade existe para criar.
    """
    if not modo_de_arquivo_respeitado():
        pytest.skip(f"permissão POSIX ignorada: {DETALHE_DO_MODO}")

    from fastapi.testclient import TestClient

    from refino.api.app import build_app
    from tests.conftest import measurements_path, sample_list_path

    modo = "pos_hilic"
    banco = tmp_path / "refino.db"
    url = f"sqlite:///{banco}"

    monkeypatch.setenv(VARIAVEL_DA_URL, url)

    assert _alembic("upgrade", "head", url=url).returncode == 0

    arquivos = {
        "measurements": (f"m_{modo}.csv", measurements_path(modo).read_bytes(), "text/csv"),
        "sample_list": (f"s_{modo}.csv", sample_list_path(modo).read_bytes(), "text/csv"),
    }
    campos = {
        "mode": modo,
        "estudo": "ATLL/HAM-TSP",
        "correction": "none",
        "n_batches": 1,
        "rsd_variant": "plain",
        "imputation": "none",
        "sample_normalization": "none",
        "transformation": "none",
        "scaling": "none",
        "apresentacao": "exibida",
    }

    # **`reclama` e nao `post`, e a razao.** A rota devolve `202` com identificador e a
    # gravacao acontece na thread de trabalho; ler o banco depois do disparo correria contra ela. O
    # auxiliar afirma o `202` do disparo e devolve a resposta da RECLAMACAO, que e onde o
    # `estado_concordancia` lido de `decisions` chega.
    resposta = reclama(TestClient(build_app()), "/execucoes", files=arquivos, data=campos)

    assert resposta.status_code == 200, resposta.text[:600]
    corpo = resposta.json()
    assert corpo["estado_concordancia"] == "nao_sobrescreveu", corpo
    assert corpo["matrix_id"] > 0 and corpo["run_id"] > 0, corpo

    assert banco.is_file(), "a rota respondeu 200 e não deixou banco em disco"
    assert banco.stat().st_mode & 0o777 == 0o600, oct(banco.stat().st_mode & 0o777)

    # **Os dois campos, no ÚNICO caminho ponta a ponta de `/execucoes` desta árvore.**
    #
    # Sem estas linhas as duas funções que compõem a resposta ficariam verificadas **em processo** —
    # `tests/test_api_execucoes.py` as chama direto — e a **rota** não: a chamada delas dentro de
    # `trabalho()` seria código de produção que nenhum teste alcança, e a suíte diria o contrário.
    # A superfície em causa é a rota e não o helper.
    #
    # `correction=none` mantém o custo onde já estava: `NoneCorrector` devolve a matriz inalterada
    # em valor, então as duas medianas são **iguais** por construção — propriedade determinística,
    # sem tolerância a escolher, e a única correção que a permite afirmar sem medir nada novo.
    assert set(corpo["qualidade"]) >= set(REQUIRED_METRIC_FIELDS), (
        f"o relatório inteiro não atravessou a rota: {sorted(corpo['qualidade'])}"
    )
    assert set(corpo["qualidade"]["computed"]) == set(REQUIRED_METRIC_FIELDS), corpo["qualidade"]
    par = corpo["rsd_antes_e_depois"]
    assert par["correcao"] == "none" and par["rsd_variant"] == "plain", par
    assert par["rsd_estimator"] == "in_sample", par
    assert par["estimador_alternativo"] == "leave_one_qc_out", par
    assert par["n_features_comparadas"] > 0, "âncora: a igualdade abaixo seria entre dois `null`"
    assert par["mediana_antes"] == par["mediana_depois"], (
        f"`none` não altera a matriz em valor, e as duas medianas divergiram: {par}"
    )


class _CorretorQueApagaADispersaoDosQCs:
    """O `none` real, mais uma alteração **O(1)** que zera o RSD dos QCs.

    **Existe porque o teste acima NÃO PODE falsificar a origem do «antes», e isso foi
    medido.** Com
    `correction=none` a matriz corrigida **é** a crua, então trocar o primeiro argumento de
    `_rsd_antes_e_depois` por `corrigida.matrix` deixa o teste acima **verde** — plantado e
    conferido: *"1 passed"*. O par antes/depois passaria a comparar a matriz corrigida
    consigo mesma, a tela mostraria melhora nula para sempre, e nada acusaria.

    **Duplo e não `loess`, e a escolha é medida:** `loess` no `neg_c18` custa **11,18 s** (2.184
    features), e pagar isso no portão de commit por uma propriedade de ligação de um argumento é o
    alvo que a equipe aprende a ignorar. Zerar as linhas de QC custa uma atribuição.

    **QC constante e não QC escalado**, e a diferença é aritmética: RSD é `sd/média`, logo
    **invariante a fator constante** — multiplicar as linhas de QC por 2 não mudaria número
    nenhum, e o par causal seria vácuo. Com os QCs todos iguais o `sd` é zero e o RSD é
    **0,0** para toda feature — divergência garantida por construção e não por sorteio.
    """

    def __init__(self) -> None:
        from refino.correction.none import NoneCorrector

        self._real = NoneCorrector()

    def __getattr__(self, nome: str) -> Any:
        """Tudo o que não é a barreira vai ao real.

        **Entrou em 07/09/2026, com o item 6.** `POST /execucoes` passou a GUARDAR a correção
        para os três botões de download, e guardar exige a chave — que sai de
        `Corrector.identidade()`. Sem a delegação o duplo morre com `AttributeError` **dentro da
        thread de trabalho**, a rota responde `500`, e o teste reprova pelo motivo errado: a
        mensagem fala do duplo e não do par antes/depois que se está a medir. É o mesmo defeito
        nesta superfície, e o irmão de `test_desktop_qualidade.py` levou o mesmo conserto.
        """
        return getattr(self._real, nome)

    def identidade(self) -> str:
        """A identidade é PRÓPRIA, e não a do real — item 6 tornou-a a CHAVE DO CACHE.

        **Par causal medido em 07/09/2026**, com este duplo e três testes de
        `test_ui_consumo.py` no mesmo processo:

        | Quem corre antes | `test_o_par_causal_com_valor_o_corte_e_aplicado` |
        |---|---|
        | ninguém | **passa** |
        | este teste | **reprova** — *"corte não aplicado: 2188 de 2188"* |

        A cadeia: `POST /execucoes` passou a **guardar** a correção (`_corrige_guardando`), a chave
        sai de `Corrector.identidade()`, e delegar o método ao real fazia a matriz **plantada** —
        com o RSD dos QCs zerado — ser guardada debaixo da chave do `none` **honesto**. Os
        `/matrizes/*` de qualquer teste seguinte reusavam-na, e o corte por RSD deixava de cortar.

        **Delegar tudo é errado exactamente aqui:** um duplo que mente sobre os VALORES não pode
        dizer-se o mesmo corretor, porque a identidade é o que separa uma correção da outra na
        memória partilhada. O `__getattr__` continua a valer para o resto da barreira.
        """
        return "duplo_que_apaga_a_dispersao_dos_qcs()"

    def fit_transform(self, matriz: Any, **kwargs: Any) -> Any:
        from dataclasses import replace as _replace

        from refino.correction.none import NoneCorrector

        real = NoneCorrector().fit_transform(matriz, **kwargs)
        valores = real.matrix.values.copy()
        valores[matriz.qc_mask] = 1.0
        return _replace(real, matrix=real.matrix.with_values(valores, "planta do teste da U4"))


@pytest.mark.integration
def test_o_par_antes_e_depois_mede_a_matriz_crua_e_nao_a_corrigida(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A direção que o caminho `none` não consegue exercitar — R9, e ela foi achada por planta.**

    O que se afirma é que o *"antes"* sai da matriz **como o usuário a enviou** e o *"depois"* da
    matriz **corrigida**. Com um corretor que apaga a dispersão dos QCs, a mediana de depois é
    **0,0** por construção e a de antes tem de ser **maior que zero**: se as duas coincidirem, o
    argumento errado está ligado.

    **Ordem das asserções é conteúdo:** a âncora de que há população comparável vem
    primeiro, senão a comparação seria entre dois `null` — que é o que a serialização produz
    para `nan` — e `None != None` é falso, então a asserção passaria por vacuidade.
    """
    from fastapi.testclient import TestClient

    from refino.api import rotas
    from refino.api.app import build_app
    from tests.conftest import measurements_path, sample_list_path

    modo = "neg_c18"
    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    # **`setitem` e não `setattr`:** `rotas` importa o dicionário de `refino.cli`, e a rota faz
    # `AVAILABLE_METHODS[spec.correction]()` — trocar o nome no módulo não alcançaria a chamada.
    monkeypatch.setitem(rotas.AVAILABLE_METHODS, "none", _CorretorQueApagaADispersaoDosQCs)

    assert _alembic("upgrade", "head", url=url).returncode == 0

    arquivos = {
        "measurements": (f"m_{modo}.csv", measurements_path(modo).read_bytes(), "text/csv"),
        "sample_list": (f"s_{modo}.csv", sample_list_path(modo).read_bytes(), "text/csv"),
    }
    campos = {
        "mode": modo,
        # **`estudo` PRÓPRIO, e não é decoração.** Ele entra na chave do trabalho (R32), e o
        # registro é de processo: um estudo repetido faria este disparo ser a **reclamação** do
        # resultado de outro teste, com o corretor real — o teste suporia ordem de execução.
        "estudo": "u4-antes-e-depois",
        "correction": "none",
        "n_batches": 1,
        "rsd_variant": "plain",
        "imputation": "none",
        "sample_normalization": "none",
        "transformation": "none",
        "scaling": "none",
        "apresentacao": "exibida",
    }

    resposta = reclama(TestClient(build_app()), "/execucoes", files=arquivos, data=campos)

    assert resposta.status_code == 200, resposta.text[:600]
    par = resposta.json()["rsd_antes_e_depois"]
    assert par["n_features_comparadas"] > 0, "âncora: sem população comparável nada abaixo mede"
    assert par["mediana_depois"] == 0.0, (
        f"o duplo apaga a dispersão dos QCs, então o RSD depois é zero por construção: {par}"
    )
    assert par["mediana_antes"] > 0.0, (
        "a mediana ANTES saiu zero: o «antes» mede a matriz CORRIGIDA e não a que o "
        f"usuário enviou, e o painel mostraria melhora nula para sempre — {par}"
    )


# ---------------------------------------------------------------------------
# Os consertos da revisão de 16/08/2026
# ---------------------------------------------------------------------------


def test_o_recreate_em_batch_preserva_as_decisoes_que_ja_existiam(url_sqlite: str) -> None:
    """**O achado #1 da revisão, e o de três revisores independentes.**

    Em SQLite o modo batch derruba `decisions`, recria por reflexão e copia as linhas. Um
    ciclo sobre banco vazio não exercita a cópia, e é nela que uma linha pode se perder.

    O teste afirma três coisas: a linha sobrevive, o valor preenchido é `sem_apresentacao`, e o
    `server_default` saiu.

    E afirma uma quarta: que o CHECK irmão, `origem_no_vocabulario`, da revisão inicial,
    sobrevive ao recreate. Ele depende de reflexão, e perdê-lo seria silencioso: nada falha até
    alguém gravar `origem` inválida.
    """
    import sqlite3

    assert _alembic("upgrade", "72944a1130f3", url=url_sqlite).returncode == 0

    banco = str(caminho_do_sqlite(url_sqlite))
    with sqlite3.connect(banco) as con:
        con.execute("INSERT INTO studies (nome, criado_em) VALUES ('antigo', '2026-01-01')")
        con.execute(
            "INSERT INTO matrices (study_id, mode, n_samples, n_features, checksum_entrada) "
            "SELECT id, 'pos_hilic', 68, 6035, 'ant' FROM studies"
        )
        con.execute(
            "INSERT INTO decisions (matrix_id, spec_json, rationale, origem, criado_em) "
            "SELECT id, '{\"correction\": \"none\"}', 'razão antiga', 'modelo', '2026-01-01' "
            "FROM matrices"
        )

    subida = _alembic("upgrade", "head", url=url_sqlite)
    assert subida.returncode == 0, subida.stderr[-2000:]

    with sqlite3.connect(banco) as con:
        linhas = list(
            con.execute("SELECT rationale, apresentacao, campos_sobrescritos FROM decisions")
        )

    assert len(linhas) == 1, "o recreate PERDEU a decisão que já existia"
    assert linhas[0][0] == "razão antiga", "o conteúdo da linha antiga mudou na cópia"
    assert linhas[0][1] == "sem_apresentacao", (
        "decisão gravada antes de existir interface não foi 'não exibida' — não havia onde "
        "exibir. `sem_apresentacao` é o único valor honesto para ela"
    )
    assert linhas[0][2] == "[]"

    # O `server_default` saiu: omitir a coluna agora REPROVA em vez de receber valor em silêncio.
    with sqlite3.connect(banco) as con, pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO decisions (matrix_id, spec_json, rationale, origem, criado_em) "
            "SELECT id, '{}', 'sem apresentacao declarada', 'modelo', '2026-01-01' FROM matrices"
        )

    # O CHECK IRMÃO sobreviveu ao recreate — par causal: `origem` inválida reprova, válida passa.
    with sqlite3.connect(banco) as con, pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO decisions "
            "(matrix_id, spec_json, rationale, origem, criado_em, apresentacao, "
            "campos_sobrescritos) "
            "SELECT id, '{}', 'r', 'lixo', '2026-01-01', 'exibida', '[]' FROM matrices"
        )
    with sqlite3.connect(banco) as con:
        con.execute(
            "INSERT INTO decisions "
            "(matrix_id, spec_json, rationale, origem, criado_em, apresentacao, "
            "campos_sobrescritos) "
            "SELECT id, '{}', 'r', 'manual', '2026-01-01', 'exibida', '[]' FROM matrices"
        )


def test_banco_dentro_da_arvore_e_fora_de_out_e_recusado(tmp_path: Path) -> None:
    """**O achado #3 — o único com exposição de dado de paciente.**

    Os três padrões por extensão do `.gitignore` não casam os arquivos que o SQLite escreve **ao
    lado** do banco: `-wal`, `-journal`, `-shm`. Um revisor conduziu um banco real e achou
    `PAC-0043` e `ATLL` como bytes crus dentro do `-wal`, com `git check-ignore` a reportar os
    três como **versionáveis**.

    **Extensão é o proxy errado e nenhuma lista de globs o conserta:** `sqlite:///dados/refino`
    não tem sufixo. A guarda é por **lugar** — dentro da árvore só `out/`, que o `.gitignore`
    ignora por diretório e portanto cobre os side files sem os enumerar.

    Par causal em três direções, porque a guarda tem três desfechos e não dois.
    """
    from refino.persistence.config import RAIZ_DO_REPOSITORIO

    # (a) dentro da árvore e fora de `out/` -> recusa, nomeando a configuração e não o valor.
    with pytest.raises(ConfiguracaoDeBancoError) as erro:
        prepara_sqlite(f"sqlite:///{RAIZ_DO_REPOSITORIO / 'dados' / 'refino'}")
    assert VARIAVEL_DA_URL in str(erro.value)
    assert "wal" in str(erro.value), "a mensagem não diz por que a extensão não basta"

    # (b) dentro de `out/` -> aceite. Âncora: sem esta, (a) passaria com uma guarda que recusa tudo.
    dentro = RAIZ_DO_REPOSITORIO / "out" / "teste-da-guarda.db"
    try:
        assert prepara_sqlite(f"sqlite:///{dentro}") == dentro
    finally:
        dentro.unlink(missing_ok=True)

    # (c) fora da árvore -> aceite. `git add` não o alcança, e o lugar é decisão do operador.
    fora = tmp_path / "refino.db"
    assert prepara_sqlite(f"sqlite:///{fora}") == fora


def test_o_arquivo_ja_existente_com_permissao_larga_e_corrigido(tmp_path: Path) -> None:
    """A janela que `prepara_sqlite` existe para fechar, no caso que a docstring dela nomeia.

    O teste que existia só cobria arquivo **inexistente**. Um banco criado antes — por uma versão
    anterior, ou por `sqlite3` à mão — nasceria com o `umask` do processo e ficaria assim.
    """
    if not modo_de_arquivo_respeitado():
        pytest.skip(f"permissão POSIX ignorada: {DETALHE_DO_MODO}")

    alvo = tmp_path / "antigo.db"
    alvo.touch()
    alvo.chmod(0o644)
    assert alvo.stat().st_mode & 0o777 == 0o644, "âncora: o arquivo começa legível por todos"

    prepara_sqlite(f"sqlite:///{alvo}")

    assert alvo.stat().st_mode & 0o777 == 0o600, oct(alvo.stat().st_mode & 0o777)


def test_falha_de_disco_e_url_recusada_pelo_sqlalchemy_dao_503_e_nao_500() -> None:
    """**O achado #2 — corroborado por dois revisores independentes.**

    `_repositorio` protegia só `url_do_banco`. As duas linhas seguintes falham por motivos de
    **ambiente**: `prepara_sqlite` faz `mkdir`/`touch`/`chmod` num caminho do operador, e
    `create_engine` levanta quando o dialeto passa o vocabulário e o **driver** não existe —
    `_recusa_se_invalida` descarta o sufixo `+driver`, então `postgresql+asyncpg://` atravessa.

    Sem as duas cláusulas os dois davam **500**, e `ui/cliente.py` traduz 500 como *"defeito da
    ferramenta, não do seu arquivo; não adianta conferir o dado"* — mandando o profissional
    ignorar exactamente o que ele consegue consertar.
    """
    from fastapi import HTTPException

    from refino.api.rotas import _repositorio

    with (
        monkeypatch_ambiente(VARIAVEL_DA_URL, "postgresql+asyncpg://u:SenhaSecreta@h/db"),
        pytest.raises(HTTPException) as erro,
    ):
        _repositorio()
    assert erro.value.status_code == 503, erro.value.status_code
    # A âncora de presença vem ANTES da de ausência: `detail` vazio faria a segunda passar
    # afirmando que a senha não vazou, quando o que não há é mensagem nenhuma.
    assert VARIAVEL_DA_URL in str(erro.value.detail)
    assert "SenhaSecreta" not in str(erro.value.detail), "a senha vazou para o detail"


def test_os_tres_ramos_de_erro_do_repositorio_dao_503(tmp_path: Path) -> None:
    """**Os TRÊS desfechos de erro de `_repositorio`, e dois nunca tinham disparado.**

    O conserto do achado #2 acrescentou três cláusulas `except`, e a primeira versão deste arquivo
    exercitava **uma**. Mecanismo que classifica em N desfechos tem N direções, e as que ficam
    inertes são sempre as que precisam de condição rara para acontecer — aqui, disco e caminho.

    A quarta direção é a âncora: com configuração boa, `_repositorio` **não** levanta.
    """
    from fastapi import HTTPException

    from refino.api.rotas import _repositorio
    from refino.persistence.config import RAIZ_DO_REPOSITORIO

    # (a) configuração ausente -> `url_do_banco` levanta.
    with (
        monkeypatch_ambiente(VARIAVEL_DA_URL, ""),
        pytest.raises(HTTPException) as sem_config,
    ):
        _repositorio()
    assert sem_config.value.status_code == 503

    # (b) lugar recusado -> `prepara_sqlite` levanta `ConfiguracaoDeBancoError`, e a rota traduz.
    dentro = RAIZ_DO_REPOSITORIO / "dados" / "proibido.db"
    with (
        monkeypatch_ambiente(VARIAVEL_DA_URL, f"sqlite:///{dentro}"),
        pytest.raises(HTTPException) as lugar,
    ):
        _repositorio()
    assert lugar.value.status_code == 503
    assert "wal" in str(lugar.value.detail), "não é a mensagem da guarda de lugar"

    # (c) falha de DISCO -> `mkdir` sobre um caminho cujo pai é ARQUIVO levanta `OSError`.
    #     É o ramo que a revisão nomeou e que nenhum teste alcançava.
    obstaculo = tmp_path / "isto-e-um-arquivo"
    obstaculo.write_text("nao sou diretorio", encoding="utf-8")
    with (
        monkeypatch_ambiente(VARIAVEL_DA_URL, f"sqlite:///{obstaculo / 'sub' / 'refino.db'}"),
        pytest.raises(HTTPException) as disco,
    ):
        _repositorio()
    assert disco.value.status_code == 503, disco.value.status_code
    assert "disco" in str(disco.value.detail) or "permissão" in str(disco.value.detail)

    # (d) ÂNCORA: com tudo bom, não levanta. Sem ela, um `_repositorio` que levantasse SEMPRE
    #     satisfaria as três acima — o portão que sempre reprova.
    bom = tmp_path / "refino.db"
    with monkeypatch_ambiente(VARIAVEL_DA_URL, f"sqlite:///{bom}"):
        assert _repositorio() is not None
    assert bom.is_file()


def test_o_alembic_da_suite_de_integracao_fixa_a_url_no_subprocesso(url_sqlite: str) -> None:
    """O subprocesso do alembic recebe a URL do motor do teste, e não a herdada do processo.

    O duplo intercepta `subprocess.run` e lê o ambiente que ele de facto recebeu.
    """
    import subprocess as _subprocess

    from tests.integration import test_migrations

    capturado: dict[str, Any] = {}

    def duplo(*args: Any, **kwargs: Any) -> Any:
        capturado.update(kwargs)
        return _subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    motor = create_engine(url_sqlite)
    original = test_migrations.subprocess.run
    test_migrations.subprocess.run = duplo  # type: ignore[assignment]
    try:
        with monkeypatch_ambiente(VARIAVEL_DA_URL, "sqlite:///out/NAO-DEVE-VAZAR.db"):
            test_migrations._alembic(motor, "--version")
    finally:
        test_migrations.subprocess.run = original  # type: ignore[assignment]
        motor.dispose()

    ambiente = dict(capturado.get("env") or {})
    assert ambiente, "âncora: o duplo recebeu um ambiente"
    assert ambiente[VARIAVEL_DA_URL] == url_sqlite, (
        "o subprocesso do alembic herdou a URL do processo e migraria o arquivo errado"
    )


def _repo_em_sqlite(url: str) -> RunRepository:
    """Um repositório sobre SQLite vazio, com o esquema criado a partir do `METADATA`.

    **`create_all` e não `alembic upgrade`, e a razão é que estes testes têm de CORRER.** O
    `alembic` por linha de comando não roda neste clone Windows — sai 1, e é uma das reprovações
    de plataforma que a suíte já tem. Um teste que dependesse dele ficaria inerte justamente na
    máquina onde o aplicativo de mesa vive, que é o referente destes consertos.

    A migração continua conferida por quem a mede — `test_alembic_sobe_e_desce_em_sqlite` —, e o
    que estes testes precisam é do esquema e não do caminho que o produz. `METADATA` é a mesma
    fonte de onde a migração inicial foi gerada.
    """
    motor = create_engine(url)
    METADATA.create_all(motor)
    return RunRepository(motor)


# ---------------------------------------------------------------------------
# A MATRIZ É REAPROVEITADA, e não reinserida — o 500 medido em 26/08/2026
# ---------------------------------------------------------------------------


def test_mesma_entrada_no_mesmo_estudo_reaproveita_a_matriz(url_sqlite: str) -> None:
    """Duas execuções sobre a mesma entrada dão UMA matriz e DUAS runs.

    **O par causal do defeito.** `uq_matrices_origem` declara que
    `(study_id, mode, checksum_entrada)` é única, e o código inseria sempre — então a segunda
    execução sobre a mesma entrada violava a regra que o próprio esquema declara. Medido contra os
    arquivos reais do `neg_c18`: trocar o método de correção mantendo o estudo dava **HTTP 500**
    com `UNIQUE constraint failed` cru na tela, e o gatilho é a coisa mais natural que se faz —
    comparar dois métodos sobre os mesmos dados.

    A repetição do MESMO método passava por acidente: a rota devolve o trabalho já existente e
    nunca chega a gravar de novo. Este teste não passa por esse acidente porque chama o
    repositório direto.
    """
    repo = _repo_em_sqlite(url_sqlite)

    a = _grava(repo, spec_json='{"correction": "none"}')
    b = _grava(repo, spec_json='{"correction": "loess"}')

    assert a.matrix_id == b.matrix_id, (
        "a mesma entrada no mesmo estudo produziu duas matrizes — a constraint diz que é uma"
    )
    with repo._engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM matrices")).scalar_one() == 1
        assert c.execute(text("SELECT count(*) FROM runs")).scalar_one() == 2
        assert c.execute(text("SELECT count(*) FROM studies")).scalar_one() == 1


def test_as_amostras_nao_sao_reinseridas_sobre_matriz_reaproveitada(url_sqlite: str) -> None:
    """`uq_samples_nome` é `(matrix_id, name)`, e reinserir trocaria um 500 por outro.

    **A asserção é de contagem e não de ausência de erro**: «não levantou» seria satisfeito por uma
    implementação que inserisse amostras duplicadas numa tabela sem a constraint.
    """
    repo = _repo_em_sqlite(url_sqlite)

    amostras = [
        {"name": "A1", "injection_order": 1, "sample_type": "qc", "grupo": "QC"},
        {"name": "A2", "injection_order": 2, "sample_type": "sample", "grupo": "ATLL"},
    ]
    _grava(repo, amostras=amostras)
    _grava(repo, amostras=amostras, spec_json='{"correction": "serrf"}')

    with repo._engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM samples")).scalar_one() == 2, (
            "as amostras foram reinseridas sobre a matriz reaproveitada"
        )


def test_entrada_diferente_no_mesmo_estudo_continua_a_dar_matriz_nova(url_sqlite: str) -> None:
    """A direção oposta, e sem ela o conserto acima seria indistinguível de «reusa sempre».

    Uma implementação que devolvesse a primeira matriz para qualquer entrada passaria nos dois
    testes acima e estaria completamente errada.
    """
    repo = _repo_em_sqlite(url_sqlite)

    a = _grava(repo)
    b = _grava(repo, checksum_entrada="f" * 64)
    assert a.matrix_id != b.matrix_id, "entradas diferentes partilharam a mesma matriz"
    with repo._engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM matrices")).scalar_one() == 2


def test_a_mesma_entrada_com_forma_diferente_levanta_em_vez_de_reaproveitar(
    url_sqlite: str,
) -> None:
    """Os mesmos bytes lidos com formas diferentes é inconsistência, e a guarda pode disparar.

    Só acontece se o leitor mudar entre duas execuções — cenário real entre versões. Reaproveitar
    em silêncio ligaria as amostras da leitura antiga a um `run` da leitura nova, e nada acusaria.
    """
    repo = _repo_em_sqlite(url_sqlite)

    _grava(repo, n_features=100)
    with pytest.raises(ValueError, match="formas diferentes"):
        _grava(repo, n_features=101)


# ---------------------------------------------------------------------------
# A DURAÇÃO GRAVADA deixa de ser zero por construção
# ---------------------------------------------------------------------------


def test_a_duracao_gravada_e_a_que_o_chamador_mediu(url_sqlite: str) -> None:
    """`iniciado_em` vem de fora, e a duração deixa de ser zero.

    Medido no banco de mesa real antes do conserto: `iniciado_em` e `terminado_em` idênticos ao
    microssegundo em **todas** as linhas — a duração de toda execução do projeto era zero.
    """
    repo = _repo_em_sqlite(url_sqlite)

    comecou = datetime.now(UTC) - timedelta(seconds=90)
    _grava(repo, iniciado_em=comecou)

    with repo._engine.connect() as c:
        inicio, fim = c.execute(text("SELECT iniciado_em, terminado_em FROM runs")).one()
    inicio_dt = inicio if isinstance(inicio, datetime) else datetime.fromisoformat(str(inicio))
    fim_dt = fim if isinstance(fim, datetime) else datetime.fromisoformat(str(fim))
    duracao = (fim_dt - inicio_dt).total_seconds()
    assert duracao > 60, f"a duração gravada foi {duracao:.1f}s, e o chamador declarou 90s"


def test_sem_iniciado_em_o_comportamento_antigo_fica(url_sqlite: str) -> None:
    """A direção oposta: quem não cronometrou não ganha um início inventado.

    Há chamadores que gravam um desfecho sem terem medido nada, e dar-lhes um início falso seria
    pior que declarar zero — seria uma duração que ninguém mediu, com aparência de medida.
    """
    repo = _repo_em_sqlite(url_sqlite)

    _grava(repo)
    with repo._engine.connect() as c:
        inicio, fim = c.execute(text("SELECT iniciado_em, terminado_em FROM runs")).one()
    assert inicio == fim, "sem `iniciado_em` a duração deve continuar a ser zero, e declarada"


# ---------------------------------------------------------------------------
# AS TRÊS ROTAS DE ARTEFATO GRAVAM QUANTO CUSTARAM — 30/08/2026
# ---------------------------------------------------------------------------
#
# **O vão, medido antes do conserto:** `corrigir`, `processar` e `exportar` tinham **zero** chamadas
# ao repositório, e só `POST /execucoes` gravava. A execução mais cara que o profissional faz é
# justamente a que não deixava rasto — `serrf` em `/matrizes/corrigir` custa **1.334,3 s** no
# `pos_hilic` e **451 s** no `neg_c18`, medidos dentro do executável congelado.
#
# **A população destes testes é o `neg_c18`**, que tem **68 amostras e 2.184 features** — medido
# sobre os arquivos de referência em 30/08/2026, e é observação sobre aqueles arquivos e não
# invariante. É o SEGUNDO menor dos quatro modos — 2.184 features contra as
# 2.039 do `neg_hilic` —, e o custo é a razão de ser ele.

DIMENSOES_DO_NEG_C18: Final = (68, 2184)


def _como_datetime(valor: Any) -> datetime:
    """O que a coluna devolveu, como `datetime`.

    SQLite não tem tipo de data: o SQLAlchemy grava texto ISO e devolve `datetime` pelo tipo
    declarado, mas um driver ou dialeto que devolvesse a string crua faria a comparação de ordem
    ser **lexicográfica** — que acerta por acidente no formato ISO e mente na primeira mudança de
    formato. Converter aqui é o que mantém a asserção sobre instantes.
    """
    return valor if isinstance(valor, datetime) else datetime.fromisoformat(str(valor))


def _grava_artefato(repo: RunRepository, **kw: Any) -> Any:
    base: dict[str, Any] = {
        "operacao": "corrigir",
        "estudo": "sem_identificacao",
        "mode": "neg_c18",
        "checksum_entrada": "c" * 64,
        "n_samples": 68,
        "n_features": 2184,
        "amostras": [
            {
                "name": "QC001",
                "injection_order": 1,
                "sample_type": "qc",
                "grupo": "QC",
                "batch": "A",
            },
        ],
        "parametros": {"method": "none", "seed": 42},
        "iniciado_em": datetime.now(UTC) - timedelta(seconds=7),
        "status": "ok",
    }
    base.update(kw)
    return repo.grava_artefato(**base)


def test_o_vocabulario_de_artefato_e_fechado_nos_dois_eixos(url_sqlite: str) -> None:
    """As duas guardas de fronteira, e a âncora que impede o teste de ser vácuo.

    **`runs.status` NÃO tem `CHECK` no esquema** — ao contrário de `decisions.origem` e de
    `multiverse_runs.estado`, onde o banco é a segunda metade da guarda. Aqui o validador é a
    guarda **inteira**: sem ele, um desfecho mal soletrado entraria em silêncio e a contagem por
    desfecho ganharia uma classe que ninguém declarou.

    **A âncora vem primeiro**: um `grava_artefato` que levantasse sempre passaria nas duas
    metades negativas e o teste diria que a guarda funciona.
    """
    repo = _repo_em_sqlite(url_sqlite)

    assert _grava_artefato(repo).run_id > 0, "âncora: a chamada válida tem de gravar"

    with pytest.raises(ValueError, match="operacao"):
        _grava_artefato(repo, operacao="baixar")
    with pytest.raises(ValueError, match="status"):
        _grava_artefato(repo, status="concluido")


def test_dois_artefatos_sobre_a_mesma_entrada_dao_uma_matriz_e_duas_runs(url_sqlite: str) -> None:
    """Baixar duas vezes a mesma matriz é `run` novo, não `matrix` nova.

    É a mesma propriedade que `test_mesma_entrada_no_mesmo_estudo_reaproveita_a_matriz` afirma para
    o pipeline, e ela precisa de ser afirmada **outra vez** aqui porque o caminho é outro: as rotas
    de artefato gravam todas sob o mesmo estudo `sem_identificacao`, então elas colidem em
    `uq_matrices_origem` com muito mais frequência que `/execucoes`. Sem o reaproveitamento, o
    segundo download dos mesmos arquivos levantaria `UNIQUE constraint failed` — e como a gravação
    é melhor esforço, a violação apareceria como uma **ausência silenciosa** de linha em vez de um
    erro.
    """
    repo = _repo_em_sqlite(url_sqlite)

    a = _grava_artefato(repo, operacao="corrigir")
    b = _grava_artefato(repo, operacao="exportar", parametros={"max_features": 5000})

    assert a.matrix_id == b.matrix_id, "a mesma entrada produziu duas matrizes"
    assert a.run_id != b.run_id, "dois downloads produziram uma `run` só"
    with repo._engine.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM matrices")).scalar_one() == 1
        assert c.execute(text("SELECT COUNT(*) FROM runs")).scalar_one() == 2


def test_a_operacao_sem_semente_e_a_com_semente_gravam_valores_diferentes(
    url_sqlite: str,
) -> None:
    """`runs.seed` é `NOT NULL` e `exportar` não tem semente — o par causal da sentinela.

    **A planta cai sobre dado em que as duas coisas diferem.** Um teste que só afirmasse
    *"`exportar` grava `SEM_SEMENTE`"* passaria numa implementação que ignorasse `parametros` e
    escrevesse a sentinela **sempre** — e a semente de `corrigir`, que é o que reproduz o
    artefato, desapareceria sem nada acusar. A asserção é que as duas linhas **divergem**.

    **A âncora contra vacuidade:** afirmar que a chave `seed` NÃO está no `spec_json` de
    `exportar` é satisfeito por um `spec_json` vazio, que é o que uma serialização quebrada
    produz. Por isso `parametros` é afirmado não-vazio **antes**, na mesma função.
    """
    repo = _repo_em_sqlite(url_sqlite)

    _grava_artefato(repo, operacao="exportar", parametros={"max_features": 5000})
    _grava_artefato(repo, operacao="corrigir", parametros={"method": "loess", "seed": 7})

    with repo._engine.connect() as c:
        linhas = c.execute(text("SELECT seed, spec_json FROM runs ORDER BY id")).all()

    (semente_sem, json_sem), (semente_com, json_com) = linhas
    parametros_sem = json.loads(json_sem)["parametros"]
    assert parametros_sem, "âncora: o `spec_json` de `exportar` não tem parâmetro nenhum"
    assert "seed" not in parametros_sem, (
        f"`exportar` não sorteia nada e o `spec_json` declarou uma semente: {parametros_sem}"
    )
    assert json.loads(json_com)["parametros"]["seed"] == 7

    assert semente_sem == SEM_SEMENTE, semente_sem
    assert semente_com == 7, semente_com
    assert semente_sem != semente_com, (
        "a coluna `seed` ficou igual nas duas: `parametros` não alcança a gravação, e a semente "
        "que reproduz o artefato de `corrigir` não está no banco"
    )


# As etapas que cada rota de artefato declara. **Escritas aqui e conferidas contra `rotas.py`**
# pelo teste abaixo: uma cópia que ninguém confere seria a segunda fonte que diverge no dia em que
# uma etapa mudar de nome — o defeito exacto que o item 45 nomeia ao recusar passar a tupla duas
# vezes.
ETAPAS_ESPERADAS = {
    "corrigir": ("correcao", "escrita do artefato"),
    "processar": ("correcao", "pipeline", "escrita do artefato"),
    "exportar": ("correcao", "exportacao"),
}


def test_as_etapas_esperadas_batem_com_as_que_as_rotas_declaram() -> None:
    """A cópia acima é conferida contra o código, e não contra a memória de quem a escreveu.

    **Sem isto o teste das durações mediria a si próprio**: ele afirma que os nomes gravados são os
    de `ETAPAS_ESPERADAS`, e se essa tupla fosse a única fonte a asserção seria uma tautologia com
    um passo pelo disco no meio. O oráculo é o texto de `rotas.py`, que é onde a rota de facto as
    declara.
    """
    fonte = (Path(__file__).resolve().parents[1] / "src" / "refino" / "api" / "rotas.py").read_text(
        encoding="utf-8"
    )
    for rota, etapas in ETAPAS_ESPERADAS.items():
        literal = "(" + ", ".join(f'"{e}"' for e in etapas) + ")"
        assert fonte.count(literal) >= 1, (
            f"a rota {rota} não declara {literal} em `rotas.py`. Ou os nomes mudaram e esta cópia "
            "ficou para trás, ou a forma da chamada mudou e este oráculo cegou"
        )


@pytest.mark.parametrize(
    ("rota", "extra", "semente_esperada"),
    [
        # **Os valores são ESQUISITOS de propósito, e não é estilo.** `_chave_do_trabalho` deriva a
        # identidade do trabalho da operação mais estes campos, e `TRABALHOS` é global do processo:
        # com os padrões, um destes disparos seria a **reclamação** do resultado que outro arquivo
        # de teste já produziu — nenhuma thread nasceria, nada seria gravado, e o teste dependeria
        # da ordem de execução da suíte. É a mesma armadilha que o teste do par antes-e-depois
        # resolve com um `estudo` próprio.
        ("corrigir", {"method": "none", "seed": 20260830}, 20260830),
        ("processar", {"method": "none", "seed": 20260830}, 20260830),
        # **O `exportar` DEIXOU DE SER `SEM_SEMENTE` em 30/08/2026**, e a mudança é de referente e
        # não de coluna. O comentário da rota dizia *"sem `seed`, e a ausência é o dado:
        # `write_metaboanalyst` não sorteia nada"* — era verdade enquanto a rota não corria
        # corretor. Ela passou a correr, o `batch_loess` sorteia no passo 4 e o `serrf` em três
        # passos, então a operação tem semente e o banco tem de a guardar: sem ela o artefato
        # exportado não é reproduzível a partir do registro.
        ("exportar", {"max_features": 4321, "seed": 20260830}, 20260830),
    ],
)
def test_as_tres_rotas_de_artefato_gravam_o_tempo_pela_rota(
    rota: str,
    extra: dict[str, Any],
    semente_esperada: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O que o usuário devolve tem de conter o tempo — pela ROTA que o botão de facto chama.

    **Os testes de repositório acima param uma camada antes, e essa distância era o defeito.** Eles
    provam que `grava_artefato` escreve; o que faltava é que as **rotas** o chamem — e entre as
    duas está o envoltório, que é código de produção que nenhum teste de repositório alcança. A
    superfície em causa é a rota, não o método.

    **A duração é afirmada ESTRITAMENTE maior que zero, e é aqui que ela pode sê-lo.** Um registro
    que gravasse o mesmo instante nas duas colunas satisfaz *"gravou o tempo"* e não separa nada —
    é o defeito exacto que `runs` teve até 26/08/2026, com `iniciado_em` e `terminado_em` idênticos
    ao microssegundo em todas as linhas. Aqui as duas divergem porque há trabalho real no meio.

    **A gravação corre ANTES de o registro em memória mudar de estado**, porque o envoltório está
    dentro de `trabalho()` e `_executa_trabalho` só chama `conclui` depois. É isso que torna esta
    leitura determinística: quando `reclama` devolve `200`, a linha já está em disco.
    """
    from fastapi.testclient import TestClient

    from refino.api.app import build_app
    from tests.conftest import measurements_path, sample_list_path

    modo = "neg_c18"
    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    METADATA.create_all(create_engine(url))

    arquivos = {
        "measurements": (f"m_{modo}.csv", measurements_path(modo).read_bytes(), "text/csv"),
        "sample_list": (f"s_{modo}.csv", sample_list_path(modo).read_bytes(), "text/csv"),
    }
    resposta = reclama(
        TestClient(build_app()), f"/matrizes/{rota}", files=arquivos, data={"mode": modo, **extra}
    )

    assert resposta.status_code == 200, resposta.text[:400]
    assert resposta.content, "âncora: sem artefato produzido não há trabalho cujo tempo medir"

    with create_engine(url).connect() as c:
        linhas = c.execute(
            text(
                "SELECT r.spec_json, r.status, r.seed, r.iniciado_em, r.terminado_em, "
                "m.mode, m.n_samples, m.n_features, r.etapas_json "
                "FROM runs r JOIN matrices m ON m.id = r.matrix_id"
            )
        ).all()

    assert len(linhas) == 1, f"a rota {rota} deixou {len(linhas)} linha(s) em `runs`"
    (spec_json, estado, semente, inicio, fim, mode_gravado, n_samples, n_features, etapas_json) = (
        linhas[0]
    )

    spec = json.loads(spec_json)
    assert spec["operacao"] == rota, spec
    assert spec["parametros"], "âncora: o `spec_json` gravado não tem parâmetro nenhum"
    assert estado == "ok", estado
    assert semente == semente_esperada, (semente, semente_esperada)

    # A forma da matriz vem de `matrices` e **não** é repetida em `runs`: é ela que responde
    # *"este tempo é sobre quantas features?"*, sem a qual o número não compara hardware nenhum.
    assert (mode_gravado, n_samples, n_features) == (modo, *DIMENSOES_DO_NEG_C18), (
        f"a forma gravada não é a do {modo}: {(mode_gravado, n_samples, n_features)}"
    )

    duracao = (_como_datetime(fim) - _como_datetime(inicio)).total_seconds()
    assert duracao > 0, (
        f"a duração gravada para a rota {rota} foi {duracao}s: as duas colunas receberam o mesmo "
        "instante, e um par assim satisfaz «gravou o tempo» sem medir nada"
    )

    # ---- A DURAÇÃO POR ETAPA — item 45 -------------------------------------------------
    #
    # **A soma das etapas tem de ficar DENTRO da duração total**, e é a asserção que separa
    # «gravou um mapa» de «gravou o mapa certo». Um cronómetro que reiniciasse a cada aviso, ou que
    # medisse do início em vez de da etapa anterior, produziria um mapa plausível cuja soma
    # ultrapassaria o total — e nenhuma asserção de presença o veria.
    assert etapas_json is not None, (
        f"a rota {rota} não gravou `etapas_json`: a linha diz quanto demorou e nunca em quê, que é "
        "exactamente o que o item 45 existe para consertar"
    )
    etapas = json.loads(etapas_json)
    assert etapas, "âncora: o mapa de etapas está VAZIO, e um mapa vazio satisfaz «não é None»"
    assert set(etapas) == set(ETAPAS_ESPERADAS[rota]), (
        f"as etapas gravadas para {rota} são {sorted(etapas)} e a rota declara "
        f"{sorted(ETAPAS_ESPERADAS[rota])}. Os nomes têm de sair de UMA fonte"
    )
    assert all(v >= 0 for v in etapas.values()), etapas
    assert sum(etapas.values()) <= duracao + 1e-6, (
        f"as etapas somam {sum(etapas.values()):.4f}s e a execução inteira durou {duracao:.4f}s. "
        "O relógio de etapa está a medir desde o início e não desde a etapa anterior"
    )
    # **A CORREÇÃO É A MAIS BARATA AQUI, e é esta a asserção que apanha desalinhamento.** As três
    # rotas correm com `method=none`, que é a identidade: a correção não faz trabalho nenhum, e o
    # custo está todo na escrita. Se o relógio estivesse deslocado de uma posição — o aviso `k` a
    # fechar a etapa `k+1` —, a correção herdaria o tempo da escrita e passaria a ser a mais cara.
    # Medido: `{'correcao': 0.0, 'exportacao': 0.127}`.
    #
    # **A primeira versão desta asserção exigia o CONTRÁRIO** — que a correção fosse a mais cara —,
    # por transcrever para aqui o *"o corretor é 99,7% do custo"* do item 45. É verdade sobre o
    # `serrf` e é falso sobre o `none`, e a população deste teste é o `none`: um número transcrito
    # sem a população dele. A execução desfez.
    assert etapas["correcao"] == min(etapas.values()), (
        f"com `method=none` a correção não é a etapa mais barata de {rota}: {etapas}. O corretor "
        "não faz nada aqui, então ou o relógio está deslocado de uma etapa, ou os nomes e as "
        "durações foram emparelhados por outra ordem"
    )
    assert max(etapas.values()) > 0, (
        f"todas as etapas de {rota} custaram zero: {etapas}. O relógio não está a andar, e um mapa "
        "de zeros satisfaz «gravou as etapas» sem medir nada"
    )


def test_os_tres_desfechos_do_envoltorio_divergem_e_a_excecao_continua_a_subir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**O par causal dos três desfechos** — concluir, cancelar, falhar, no mesmo envoltório.

    **`cancelado` e `falhou` têm de DIVERGIR, e não é detalhe de rótulo.** Se
    `ExecutionCancelledError` fosse apanhada pelo `except Exception` largo, o disco diria *"a
    ferramenta quebrou"* sobre um botão que o profissional carregou de propósito — a mesma
    distinção que se preservou em memória, aqui em disco. Exercitar um desfecho só deixaria os
    outros dois inertes: são **três** direções, e a pergunta é *"eu vi todas?"*.

    **A segunda metade é a exceção CONTINUAR A SUBIR.** Um envoltório que gravasse e engolisse
    faria um trabalho falhado chegar a `_executa_trabalho` como sucesso: a rota devolveria `200`
    com resultado nenhum, e o disco diria `falhou` ao lado. Sem estes dois `pytest.raises`, essa
    regressão passaria com os três rótulos certos no banco.

    **Pelo envoltório e não por HTTP**, e a razão é poder de teste e não custo: cancelar pela rota
    exigiria apanhar um trabalho vivo no meio, e um teste que dependa de ganhar essa corrida
    reprova por relógio em vez de por defeito.
    """
    import numpy as np

    from refino.api.rotas import _com_registro_de_tempo
    from refino.domain.cancellation import ExecutionCancelledError
    from refino.domain.matrix import RefinoMatrix, SampleMetadata

    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    METADATA.create_all(create_engine(url))

    matriz = RefinoMatrix(
        values=np.arange(12.0).reshape(4, 3) + 1.0,
        feature_names=("1.0_100m/z", "1.0_101m/z", "1.0_102m/z"),
        samples=tuple(
            SampleMetadata(
                name=f"S{i}",
                sample_type="qc" if i % 2 == 0 else "sample",
                injection_order=i + 1,
                batch="A",
                group="QC" if i % 2 == 0 else "G1",
            )
            for i in range(4)
        ),
        mode="neg_c18",
    )

    def envolve(trabalho: Any) -> Any:
        return _com_registro_de_tempo(
            "corrigir",
            mode="neg_c18",
            matriz=matriz,
            checksum="d" * 64,
            parametros={"method": "none", "seed": 3},
            etapas=("correcao", "escrita do artefato"),
            trabalho=trabalho,
        )

    def marca() -> None:
        return None

    def nunca() -> bool:
        return False

    def conclui(m: Any, c: Any) -> object:
        return "pronto"

    def cancela(m: Any, c: Any) -> object:
        raise ExecutionCancelledError("planta: cancelado no meio")

    def falha(m: Any, c: Any) -> object:
        raise ValueError("planta: o trabalho quebrou")

    assert envolve(conclui)(marca, nunca) == "pronto", (
        "o envoltório trocou o resultado do trabalho: quem reclama receberia outra coisa"
    )
    with pytest.raises(ExecutionCancelledError):
        envolve(cancela)(marca, nunca)
    with pytest.raises(ValueError, match="quebrou"):
        envolve(falha)(marca, nunca)

    with create_engine(url).connect() as c:
        estados = [linha[0] for linha in c.execute(text("SELECT status FROM runs ORDER BY id"))]

    assert estados == ["ok", "cancelado", "falhou"], estados
    assert len(set(estados)) == 3, (
        f"os três desfechos colapsaram em {len(set(estados))} rótulo(s): {estados}. Com "
        "`cancelado` a cair em `falhou`, o disco acusa a ferramenta por uma parada pedida"
    )

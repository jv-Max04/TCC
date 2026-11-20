"""A migração sobe e desce em banco POPULADO.

**"Sem perda" numa migração que CRIA tabelas não é o que a frase sugere, e a distinção vai aqui
porque ela decide o que o teste afirma.** O `downgrade` da migração inicial **derruba** as nove
tabelas: dado dentro delas some por definição, e exigir que sobreviva seria exigir o
impossível de um `DROP TABLE`. O que "sem perda" significa **para esta migração** é
**fidelidade estrutural do ciclo**: o que o `up` cria, o `down` remove — sem sobra —, e o `up`
seguinte recria o mesmo esquema.

**A leitura de preservação de dado vale para as migrações FUTURAS** — as que alteram tabela
existente —, e é lá que a contagem de linhas antes e depois é a asserção certa. Declarar isso
agora impede que alguém leia este teste como prova de algo que ele não mede: é a distância de
um defeito num teste em vez de num texto.

**E a asserção NÃO é ausência de exceção**, que é o que o critério proíbe: o ciclo é
conferido por **inventário de objetos** — tabelas, e para cada uma as colunas —, comparado
antes e depois. Um `downgrade` que deixasse tabela órfã, sequência ou constraint para trás
passaria no teste de exceção e reprova neste.
"""

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.pool import NullPool

from refino.persistence.config import VARIAVEL_DA_URL
from refino.persistence.schema import TABELAS

REPO = Path(__file__).resolve().parent.parent.parent

pytestmark = pytest.mark.integration


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """Motor sobre um arquivo SQLite novo, num diretório temporário.

    `NullPool` como a aplicação usa: cada conexão fecha ao voltar, e o alembic do subprocesso
    não encontra o arquivo preso.
    """
    motor = create_engine(f"sqlite:///{tmp_path / 'refino.db'}", poolclass=NullPool)
    yield motor
    motor.dispose()


def _alembic(motor: Engine, *args: str) -> subprocess.CompletedProcess[str]:
    """Roda o Alembic por subprocesso, como o usuário roda, sobre o banco de `motor`.

    A URL entra numa cópia do ambiente. Assim o subprocesso migra o banco que o teste lê, e
    não o que estiver em `REFINO_DB_URL` no processo do pytest.
    """
    ambiente = dict(os.environ)
    ambiente[VARIAVEL_DA_URL] = motor.url.render_as_string(hide_password=False)
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


def _inventario(motor: Engine) -> dict[str, list[str]]:
    """`tabela -> colunas e RESTRIÇÕES ordenadas`, sem a tabela de controle do próprio Alembic.

    **As restrições entraram em 28/08/2026, e a docstring deste módulo já as prometia.** Ela diz,
    verbatim, que *"um `downgrade` que deixasse tabela órfã, sequência ou constraint para trás
    passaria no teste de exceção e reprova neste"* — e a versão anterior desta função lia só
    `get_columns`. A promessa era mais larga que o mecanismo:
    afirmação de cobertura maior que a superfície coberta.

    O prefixo distingue as três classes no mesmo dicionário, para que um `check` desaparecido não
    possa ser compensado por uma coluna nova com o mesmo nome.
    """
    insp = inspect(motor)
    return {
        t: sorted(
            [f"col:{c['name']}" for c in insp.get_columns(t)]
            + [f"uq:{u['name']}" for u in insp.get_unique_constraints(t)]
            + [f"ck:{c['name']}" for c in insp.get_check_constraints(t)]
        )
        for t in insp.get_table_names()
        if t != "alembic_version"
    }


def _popula(motor: Engine) -> dict[str, int]:
    """Uma linha em cada uma das nove, respeitando as chaves estrangeiras. Devolve as contagens."""
    with motor.begin() as c:
        c.execute(
            text("INSERT INTO studies (nome, criado_em) VALUES ('estudo', CURRENT_TIMESTAMP)")
        )
        c.execute(
            text(
                "INSERT INTO matrices (study_id, mode, n_samples, n_features, checksum_entrada) "
                "SELECT id, 'pos_hilic', 68, 6035, 'abc' FROM studies"
            )
        )
        c.execute(
            text(
                "INSERT INTO samples (matrix_id, name, injection_order, sample_type) "
                "SELECT id, 'QC001', 1, 'qc' FROM matrices"
            )
        )
        c.execute(
            text(
                "INSERT INTO runs (matrix_id, spec_json, seed, iniciado_em, status) "
                "SELECT id, '{}', 42, CURRENT_TIMESTAMP, 'ok' FROM matrices"
            )
        )
        c.execute(
            text(
                "INSERT INTO diagnoses (matrix_id, defect_type, confidence, profile_json, "
                "criado_em) SELECT id, 'erratic', 0.5, '{}', CURRENT_TIMESTAMP FROM matrices"
            )
        )
        c.execute(
            text(
                # `apresentacao` e `campos_sobrescritos` sao EXPLICITOS aqui, e o `server_default`
                # foi retirado pela migracao `2d12e7fb4974` de proposito: omiti-los tem de
                # reprovar, e ha teste proprio afirmando isso. Quem grava decisao declara se a
                # justificativa foi exibida.
                "INSERT INTO decisions (matrix_id, diagnosis_id, spec_json, rationale, origem, "
                "criado_em, apresentacao, campos_sobrescritos) "
                "SELECT m.id, d.id, '{}', 'porque sim', 'modelo', CURRENT_TIMESTAMP, "
                "'sem_apresentacao', '[]' FROM matrices m, diagnoses d"
            )
        )
        c.execute(
            text("INSERT INTO quality_reports (run_id, metrics_json) SELECT id, '{}' FROM runs")
        )
        c.execute(
            text(
                # `identificador`, `estado` e `n_feitas` sao EXPLICITOS, e a migracao `a3f7c1b90e42`
                # nao lhes deu `server_default` de proposito: omiti-los tem de reprovar. Um padrao
                # em `estado` faria um INSERT distraido receber um rotulo plausivel em silencio.
                "INSERT INTO multiverse_runs "
                "(matrix_id, identificador, estado, n_feitas, n_specs, iniciado_em, terminado_em) "
                "SELECT id, 'sonda-da-migracao', 'concluido', 96, 96, CURRENT_TIMESTAMP, "
                "CURRENT_TIMESTAMP FROM matrices"
            )
        )
        c.execute(
            text(
                "INSERT INTO feature_stability "
                "(multiverse_run_id, feature_name, frac_significant, n_cells) "
                "SELECT id, '2.25_810.7739m/z', 0.0, 96 FROM multiverse_runs"
            )
        )
    return _contagens(motor)


def _contagens(motor: Engine) -> dict[str, int]:
    with motor.connect() as c:
        return {t: c.execute(text(f"SELECT count(*) FROM {t}")).scalar_one() for t in TABELAS}


def test_to_10_5_o_ciclo_sobe_popula_desce_e_sobe_com_inventario_conferido(engine: Engine) -> None:
    """**A asserção é o INVENTÁRIO, não a ausência de exceção**.

    `downgrade` que deixe tabela órfã, coluna a mais ou coluna a menos passa num teste de exceção e
    reprova neste.
    """
    assert _alembic(engine, "upgrade", "head").returncode == 0
    antes = _inventario(engine)
    assert set(antes) == set(TABELAS), (
        f"o `up` não criou as nove: {sorted(set(TABELAS) - set(antes))}"
    )

    contagens = _popula(engine)
    assert all(n == 1 for n in contagens.values()), f"população incompleta: {contagens}"

    assert _alembic(engine, "downgrade", "base").returncode == 0
    assert _inventario(engine) == {}, (
        "o `down` deixou objeto para trás. Numa migração que CRIA tabelas, 'sem perda' é "
        "fidelidade estrutural do ciclo — o que o `up` cria, o `down` remove"
    )

    assert _alembic(engine, "upgrade", "head").returncode == 0
    assert _inventario(engine) == antes, "o segundo `up` não recriou o mesmo esquema"
    assert all(n == 0 for n in _contagens(engine).values()), (
        "tabela recriada com linha dentro — o `down` não derrubou de verdade"
    )
    _alembic(engine, "downgrade", "base")


def test_o_check_de_origem_recusa_valor_fora_do_vocabulario(engine: Engine) -> None:
    """`decisions.origem` é fronteira, e o `CHECK` é a metade que vive no banco."""
    from sqlalchemy.exc import IntegrityError

    assert _alembic(engine, "upgrade", "head").returncode == 0
    _popula(engine)
    with pytest.raises(IntegrityError), engine.begin() as c:
        c.execute(
            text(
                "INSERT INTO decisions (matrix_id, spec_json, rationale, origem, criado_em, "
                "apresentacao, campos_sobrescritos) "
                "SELECT id, '{}', 'x', 'heuristico', CURRENT_TIMESTAMP, 'sem_apresentacao', '[]' "
                "FROM matrices"
            )
        )
    _alembic(engine, "downgrade", "base")


def test_o_par_causal_origem_valida_e_aceita(engine: Engine) -> None:
    """**Visto passar sem a planta**: as três do vocabulário entram."""
    assert _alembic(engine, "upgrade", "head").returncode == 0
    _popula(engine)
    with engine.begin() as c:
        for origem in ("heuristica", "modelo", "manual"):
            c.execute(
                text(
                    "INSERT INTO decisions (matrix_id, spec_json, rationale, origem, criado_em, "
                    "apresentacao, campos_sobrescritos) "
                    f"SELECT id, '{{}}', 'x', '{origem}', CURRENT_TIMESTAMP, "
                    "'sem_apresentacao', '[]' FROM matrices"
                )
            )
    assert _contagens(engine)["decisions"] == 4
    _alembic(engine, "downgrade", "base")

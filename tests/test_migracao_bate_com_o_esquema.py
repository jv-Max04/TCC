"""A migração e o `METADATA` declaram o MESMO banco — o portão que faltava, 28/08/2026.

## O que ele responde, e por que nenhum outro respondia

`tests/integration/test_migrations.py` compara o banco **contra ele mesmo**: sobe, popula, desce,
sobe, e afirma que o inventário do fim é o do começo. Isso é fidelidade do **ciclo**, e é o que o
O critério pede. Não é a mesma pergunta que esta:

> **o banco que a migração constrói é o banco que `schema.py` declara?**

As duas fontes existem em paralelo por decisão — a revisão anterior escreve que *"migracao descreve
o estado do banco NAQUELE ponto da historia, e importar do codigo faria uma revisao antiga mudar de
significado quando o vocabulario evoluir"* — sem acento, que e como ela esta escrita. A duplicação é
deliberada e correta; o que faltava era o portão que confere que a
**última** revisão e o `METADATA` de hoje convergem.

## A instância que o motivou, e ela é medida

Escrevendo `a3f7c1b90e42` eu passei o nome da restrição única **nu** — `identificador` —, por
analogia com a revisão anterior, que só tem `CheckConstraint`. Medido depois: o banco ficou com
`identificador` e o `METADATA` produz `uq_multiverse_runs_identificador`. A regra é a
`naming_convention`, e ela trata os dois tipos de forma diferente:

| Tipo | Padrão | O nome que a migração passa |
|---|---|---|
| `ck` | `ck_%(table_name)s_%(constraint_name)s` | **nu** — há onde encaixar |
| `uq` | `uq_%(table_name)s_%(column_0_name)s` | **já expandido** — não há |

**As duas pontas divergiram, o ciclo do critério passava** — ele compara a migração consigo
mesma, e o nome errado é o mesmo nas duas medições — **e nada acusava.**

## Por que SQLite e não Postgres

Este portão corre em `make check`, sem serviço nenhum em pé — que é o que faz dele um portão de
commit em vez de um teste que ninguém vê reprovar. O que ele compara é **estrutura
declarada**, e essa é a mesma nos dois dialetos; o que difere entre eles é como o `ALTER` chega
lá, e disso trata o teste de integração.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, Inspector, create_engine, inspect

from refino.desktop.banco import migra
from refino.persistence.schema import METADATA, TABELAS


def _forma(insp: Inspector, tabela: str) -> dict[str, object]:
    """A forma de uma tabela: colunas com nulabilidade, e as restrições por nome.

    **Nulabilidade e `server_default` entram; TIPO não**, e a exclusão vai declarada. `VARCHAR(64)`
    do SQLAlchemy e o que o SQLite reflete não são a mesma string em todos os casos, e um portão
    que comparasse tipos reprovaria por dialeto em vez de por divergência — o portão sempre
    vermelho que a equipe aprende a ignorar. Nulabilidade, padrão de servidor e nome de restrição
    são reflexos fiéis nos dois dialetos.

    **O que continua FORA, e vai nomeado em vez de subentendido:** tipo, chaves estrangeiras,
    índices não-únicos e ordem de coluna. A pergunta do cabeçalho — *"o banco que a migração
    constrói é o banco que `schema.py` declara?"* — é respondida **nestes quatro eixos**, e não em
    todos.
    """
    return {
        "colunas": {c["name"]: c["nullable"] for c in insp.get_columns(tabela)},
        # **`server_default` ENTRA, e a ausência dele era um buraco medido.**
        #
        # A migração `a3f7c1b90e42` declara, em prosa, que `identificador`, `estado` e `n_feitas`
        # **não** têm padrão de servidor — *"um padrão em `estado` faria um `INSERT` que o omitisse
        # receber um rótulo plausível em silêncio"*. Medido em 28/08/2026: dar um
        # `server_default="PLANTA"` a `identificador` na migração deixava este portão **verde**.
        #
        # A propriedade análoga para `decisions.apresentacao` tem portão
        # (`pytest.raises(IntegrityError)`); esta não tinha nenhum, e a prosa afirmava-a como se
        # tivesse.
        "padroes": {
            c["name"]: (None if c["default"] is None else str(c["default"]))
            for c in insp.get_columns(tabela)
        },
        "unicas": sorted(
            (u["name"], tuple(u["column_names"])) for u in insp.get_unique_constraints(tabela)
        ),
        "checagens": sorted(c["name"] for c in insp.get_check_constraints(tabela)),
    }


@pytest.fixture
def por_migracao(tmp_path: Path) -> Iterator[Engine]:
    """O banco como a **migração** o constrói: `alembic upgrade head` até a revisão de hoje."""
    url = f"sqlite:///{(tmp_path / 'migrado.db').as_posix()}"
    migra(url)
    motor = create_engine(url)
    yield motor
    motor.dispose()


@pytest.fixture
def por_metadata(tmp_path: Path) -> Iterator[Engine]:
    """O banco como o **`schema.py`** o declara: `METADATA.create_all`."""
    url = f"sqlite:///{(tmp_path / 'declarado.db').as_posix()}"
    motor = create_engine(url)
    METADATA.create_all(motor)
    yield motor
    motor.dispose()


def test_as_duas_fontes_declaram_as_mesmas_tabelas(
    por_migracao: Engine, por_metadata: Engine
) -> None:
    """A âncora, e ela vem primeiro: sem tabela nenhuma as comparações abaixo são vácuas."""
    migradas = {t for t in inspect(por_migracao).get_table_names() if t != "alembic_version"}
    declaradas = set(inspect(por_metadata).get_table_names())

    assert migradas, "âncora: a migração não criou tabela nenhuma"
    assert migradas == set(TABELAS), (
        f"a migração criou {sorted(migradas)} e `TABELAS` diz {sorted(TABELAS)}"
    )
    assert migradas == declaradas, (
        f"só na migração: {sorted(migradas - declaradas)}; "
        f"só no METADATA: {sorted(declaradas - migradas)}"
    )


@pytest.mark.parametrize("tabela", TABELAS)
def test_cada_tabela_tem_a_mesma_forma_nas_duas_fontes(
    tabela: str, por_migracao: Engine, por_metadata: Engine
) -> None:
    """Coluna a coluna e restrição a restrição, **por tabela**.

    Parametrizado e não num laço: com o laço, a primeira divergência esconde as outras oito, e a
    mensagem de reprovação diria *"o banco diverge"* em vez de *"`multiverse_runs` diverge"*.
    """
    migrada = _forma(inspect(por_migracao), tabela)
    declarada = _forma(inspect(por_metadata), tabela)

    assert migrada["colunas"], f"âncora: `{tabela}` veio sem coluna nenhuma da migração"
    assert migrada == declarada, (
        f"`{tabela}` tem formas diferentes nas duas fontes.\n"
        f"  migração: {migrada}\n"
        f"  METADATA: {declarada}"
    )

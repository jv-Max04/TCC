"""o par de concordancia: apresentacao e campos_sobrescritos em decisions

Gerada por `alembic revision --autogenerate` e ajustada à mão em três pontos que a
autogeração não alcança:

- a `CheckConstraint` de `apresentacao`, que o alembic não autogera. Sem ela a guarda existiria
  só em Python, e uma escrita que não passasse por `validate_apresentacao` entraria no banco;
- o `server_default` na criação das colunas, porque `nullable=False` sem valor de partida
  reprova em `decisions` já populada. `sem_apresentacao` é o valor correto para as linhas
  antigas: elas nasceram sem superfície de apresentação;
- a retirada do `server_default` depois do preenchimento. Mantido, ele daria `sem_apresentacao`
  em silêncio a um `INSERT` que omitisse a coluna. Sem ele, quem grava decisão declara a
  apresentação.

Esta revisão altera uma tabela que pode ter linhas, então a descida e a subida têm de
preservar o dado, e não só a estrutura.

Revision ID: 2d12e7fb4974
Revises: 72944a1130f3
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "2d12e7fb4974"
down_revision: str | None = "72944a1130f3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Literais e não importados de `schema.py`: a migração descreve o banco naquele ponto da
# história, e importar do código faria uma revisão antiga mudar de significado quando o
# vocabulário evoluir.
APRESENTACAO_INICIAL = "sem_apresentacao"
CAMPOS_INICIAIS = "[]"


def upgrade() -> None:
    op.add_column(
        "decisions",
        sa.Column(
            "apresentacao",
            sa.String(length=32),
            nullable=False,
            server_default=APRESENTACAO_INICIAL,
        ),
    )
    op.add_column(
        "decisions",
        sa.Column("campos_sobrescritos", sa.Text(), nullable=False, server_default=CAMPOS_INICIAIS),
    )
    # As três operações seguintes vão em batch: o SQLite não tem
    # `ALTER TABLE ... ADD CONSTRAINT` nem `ALTER COLUMN`, e o modo batch do alembic cria a
    # tabela nova com a forma final, copia as linhas e troca.
    with op.batch_alter_table("decisions") as batch_op:
        batch_op.create_check_constraint(
            "apresentacao_no_vocabulario",
            "apresentacao IN ('exibida', 'nao_exibida', 'sem_apresentacao')",
        )
        # As linhas antigas já estão preenchidas pelo `server_default`; o que se retira é o
        # padrão para os INSERTs futuros.
        batch_op.alter_column("apresentacao", server_default=None)
        batch_op.alter_column("campos_sobrescritos", server_default=None)


def downgrade() -> None:
    # O nome da restrição vai sem prefixo: a `naming_convention` de `METADATA` o expande nas
    # duas pontas, e passar `ck_decisions_apresentacao_no_vocabulario` aqui procuraria um nome
    # com o prefixo aplicado duas vezes, que não existe.
    #
    # A descida também vai em batch: `DROP CONSTRAINT` não existe em SQLite, e `DROP COLUMN` só
    # a partir do 3.35.
    with op.batch_alter_table("decisions") as batch_op:
        batch_op.drop_constraint("apresentacao_no_vocabulario", type_="check")
        batch_op.drop_column("campos_sobrescritos")
        batch_op.drop_column("apresentacao")

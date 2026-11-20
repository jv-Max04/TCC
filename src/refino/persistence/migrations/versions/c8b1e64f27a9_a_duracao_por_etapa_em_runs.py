"""a duracao por etapa em `runs`

`runs` já tem `iniciado_em` e `terminado_em`, que dão a duração da execução inteira. A coluna
nova diz onde o tempo foi gasto, etapa por etapa. É coluna própria e não um campo do
`spec_json` porque o `spec_json` guarda o que foi pedido, `operacao` e `parametros`, e a
duração é o que aconteceu.

Anulável e sem `server_default`, porque a ausência é um estado real: `POST /execucoes` não
cronometra etapas, e as linhas anteriores a esta revisão também não têm. Um padrão de `'{}'`
diria que a execução correu e nenhuma etapa custou nada, sobre linhas em que ninguém mediu.

Batch nas duas pontas: o SQLite só tem `DROP COLUMN` a partir do 3.35, e o `downgrade`
precisa dele.

Revision ID: c8b1e64f27a9
Revises: a3f7c1b90e42
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c8b1e64f27a9"
down_revision: str | None = "a3f7c1b90e42"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("runs") as batch_op:
        batch_op.add_column(sa.Column("etapas_json", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("runs") as batch_op:
        batch_op.drop_column("etapas_json")

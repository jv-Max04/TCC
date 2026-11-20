"""o multiverso repousa em disco: estado terminal e conclusao

Escrita à mão e não por `--autogenerate`, porque a `CheckConstraint`, as duas renomeações de
coluna e o modo batch estão fora do que ele alcança.

Quando esta revisão foi escrita, nada no código gravava em `multiverse_runs` nem em
`feature_stability`, e isso decide a forma dela:

- as colunas `nullable=False` entram sem `server_default`. Um padrão em `estado` daria um
  rótulo plausível, em silêncio, a um `INSERT` que o omitisse. Sem ele, quem grava execução de
  multiverso declara o estado, e se a tabela tiver linhas a migração reprova em vez de
  preencher com valor inventado;
- renomear duas colunas não perde dado. `median_pvalue` guardaria o que `FeatureStability`
  chama `median_qvalue`, um q-valor, já corrigido por FDR, e um nome que diz p onde há q leva
  a ler a coluna errado.

Batch nas duas pontas: o SQLite não tem `ALTER TABLE ... ADD CONSTRAINT`, não renomeia coluna
antes do 3.25, e só tem `DROP COLUMN` a partir do 3.35.

Revision ID: a3f7c1b90e42
Revises: 2d12e7fb4974
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a3f7c1b90e42"
down_revision: str | None = "2d12e7fb4974"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Literal e não importado de `schema.py`: a migração descreve o banco naquele ponto da
# história, e importar do código faria uma revisão antiga mudar de significado quando o
# vocabulário evoluir.
ESTADOS = "('executando', 'concluido', 'falhou', 'cancelado', 'interrompido')"


def upgrade() -> None:
    with op.batch_alter_table("multiverse_runs") as batch_op:
        batch_op.add_column(sa.Column("identificador", sa.String(length=64), nullable=False))
        batch_op.add_column(sa.Column("estado", sa.String(length=16), nullable=False))
        batch_op.add_column(sa.Column("n_feitas", sa.Integer(), nullable=False))
        batch_op.add_column(sa.Column("celula_que_falhou", sa.String(length=500), nullable=True))
        batch_op.add_column(sa.Column("erro", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("group_a", sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column("group_b", sa.String(length=100), nullable=True))
        # As duas restrições pedem o nome em formas diferentes, e o que decide é se o padrão da
        # `naming_convention` de `METADATA` para aquele tipo contém `%(constraint_name)s`. O de
        # `ck` contém, então o nome vai sem prefixo e a convenção o expande. O de `uq` é montado
        # com o nome da coluna, então o nome vai já expandido; passado sem prefixo, a restrição
        # ficaria no banco com um nome diferente do que `METADATA` produz.
        batch_op.create_unique_constraint("uq_multiverse_runs_identificador", ["identificador"])
        batch_op.create_check_constraint("estado_no_vocabulario", f"estado IN {ESTADOS}")

    with op.batch_alter_table("feature_stability") as batch_op:
        batch_op.alter_column("median_pvalue", new_column_name="median_qvalue")
        batch_op.alter_column("rank_median", new_column_name="median_rank")
        batch_op.add_column(sa.Column("rank_iqr", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("sign_consistency", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("n_cells", sa.Integer(), nullable=False))


def downgrade() -> None:
    with op.batch_alter_table("feature_stability") as batch_op:
        batch_op.drop_column("n_cells")
        batch_op.drop_column("sign_consistency")
        batch_op.drop_column("rank_iqr")
        batch_op.alter_column("median_rank", new_column_name="rank_median")
        batch_op.alter_column("median_qvalue", new_column_name="median_pvalue")

    with op.batch_alter_table("multiverse_runs") as batch_op:
        batch_op.drop_constraint("estado_no_vocabulario", type_="check")
        batch_op.drop_constraint("uq_multiverse_runs_identificador", type_="unique")
        for coluna in (
            "group_b",
            "group_a",
            "erro",
            "celula_que_falhou",
            "n_feitas",
            "estado",
            "identificador",
        ):
            batch_op.drop_column(coluna)

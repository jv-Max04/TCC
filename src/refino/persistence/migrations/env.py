"""Ambiente do Alembic.

A URL vem de `REFINO_DB_URL`, por `persistence.config`, e nunca do `alembic.ini`.

`compare_type=True` e `compare_server_default=True` porque migração gerada sem eles ignora mudança
de tipo e de padrão, e migração que não vê a mudança tem descida que não a desfaz.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from refino.persistence.config import prepara_sqlite, url_do_banco
from refino.persistence.schema import METADATA

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = METADATA


def _url() -> str:
    """A URL do banco, de `persistence.config`: a mesma que a aplicação usa."""
    return url_do_banco()


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _url()
    prepara_sqlite(url)
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = url
    engine = engine_from_config(cfg, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with engine.connect() as conn:
        context.configure(
            connection=conn,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            # `render_as_batch` é opção de autogeração: as migrações novas saem em
            # `batch_alter_table`, que o SQLite exige para alterar restrição e coluna. Ele não
            # muda como as já escritas executam.
            render_as_batch=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():  # pragma: no cover - caminho de geração, não de teste
    run_migrations_offline()
else:
    run_migrations_online()

"""Alembic migrations, runnable programmatically (API startup) or via the CLI."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from evalforge.config import get_settings

HERE = Path(__file__).parent


def alembic_config() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(HERE))
    cfg.set_main_option("sqlalchemy.url", get_settings().resolved_database_url)
    return cfg


def upgrade_to_head() -> None:
    command.upgrade(alembic_config(), "head")

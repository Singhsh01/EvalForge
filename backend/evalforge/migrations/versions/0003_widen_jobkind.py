"""widen jobs.kind for the pipeline / eval_suite job kinds on non-PostgreSQL databases

On PostgreSQL ``jobkind`` is a native enum and 0002 added the values with ALTER TYPE. On
SQLite the enum is a VARCHAR sized to the longest value at creation time (8, "evaluate"),
which is too short for "eval_suite". SQLite does not enforce the length, so nothing broke,
but the schema must match the ORM.

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

OLD = sa.Enum("train", "evaluate", name="jobkind")
NEW = sa.Enum("train", "evaluate", "pipeline", "eval_suite", name="jobkind")


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        return
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.alter_column("kind", existing_type=OLD, type_=NEW, existing_nullable=False)


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        return
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.alter_column("kind", existing_type=NEW, type_=OLD, existing_nullable=False)

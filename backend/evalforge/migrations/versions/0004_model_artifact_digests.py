"""model artifact digests (audit SEC-05)

Records the SHA-256 of each serialized model so the platform can detect a tampered or
swapped artifact before loading it.

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("training_runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("model_sha256", sa.String(length=64), nullable=True))
    with op.batch_alter_table("model_versions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("artifact_sha256", sa.String(length=64), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("model_versions", schema=None) as batch_op:
        batch_op.drop_column("artifact_sha256")
    with op.batch_alter_table("training_runs", schema=None) as batch_op:
        batch_op.drop_column("model_sha256")

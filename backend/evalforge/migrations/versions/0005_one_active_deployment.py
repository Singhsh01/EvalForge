"""at most one active stable/canary deployment per model and environment (audit RES-03)

Concurrent deploys could both commit an active stable deployment. A partial unique index
makes the database reject the second one. Existing duplicates (from that race) are retired
first, keeping the newest, so the index can be created on any database.

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            "SELECT id, model_name, environment, role FROM deployments "
            "WHERE status = 'active' ORDER BY created_at DESC"
        )
    ).fetchall()
    seen: set[tuple[str, str, str]] = set()
    for dep_id, name, env, role in rows:
        key = (name, env, role)
        if key in seen:
            bind.execute(
                sa.text(
                    "UPDATE deployments SET status = 'superseded', "
                    "note = note || ' [retired by migration 0005: duplicate active]' "
                    "WHERE id = :id"
                ),
                {"id": dep_id},
            )
        seen.add(key)
    op.create_index(
        "uq_deployments_one_active",
        "deployments",
        ["model_name", "environment", "role"],
        unique=True,
        sqlite_where=sa.text("status = 'active'"),
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    op.drop_index("uq_deployments_one_active", table_name="deployments")

"""Remember the accounts the operator vouched for, so a resumed run keeps them.

A discovery run that was still in flight when the process stopped is picked
up again on the next start. The resumed run used to lose the list of accounts
the operator vouched for, so the bot detection ran on them again and the
accounts the operator had already confirmed could be filtered out of the
results.

The list is stored on the run row, so it survives the restart.

Revision ID: c4e1a7b3d952
Revises: b2c9f4e17a03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "c4e1a7b3d952"
down_revision: str | None = "b2c9f4e17a03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "discovery_runs",
        sa.Column(
            "known_human_accounts",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("discovery_runs", "known_human_accounts")

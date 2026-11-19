"""Drop the approval queue's APPROVED state from the dedup lock.

The approval queue is gone: there is no background dispatcher any more, so a
record can never sit in ``APPROVED`` waiting to be drained. Two things follow
from that:

* the partial unique index that enforces the dedup lock no longer needs
  ``APPROVED`` in its predicate, and
* any record still left in ``APPROVED`` from before this change would be
  stranded in a state the code no longer handles, so it is moved back to
  ``PENDING_APPROVAL`` where it shows up as the one pending draft for that
  recipient.

Revision ID: b2c9f4e17a03
Revises: 771d0697afa5
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "b2c9f4e17a03"
down_revision: str | None = "771d0697afa5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: The statuses that block any further outreach to the same recipient.
_NEW_PREDICATE = "status IN ('PENDING_APPROVAL', 'SENT', 'OPTED_OUT')"
_OLD_PREDICATE = "status IN ('PENDING_APPROVAL', 'APPROVED', 'SENT', 'OPTED_OUT')"


def upgrade() -> None:
    # A record that was waiting in the queue becomes a pending draft again.
    op.execute(
        sa.text(
            "UPDATE outreach_records SET status = 'PENDING_APPROVAL' "
            "WHERE status = 'APPROVED'"
        )
    )

    # The dedup lock is rewritten with the new predicate.
    op.drop_index(
        "uq_outreach_active_x_user_id",
        table_name="outreach_records",
        postgresql_where=_OLD_PREDICATE,
    )
    op.create_index(
        "uq_outreach_active_x_user_id",
        "outreach_records",
        ["x_user_id"],
        unique=True,
        postgresql_where=_NEW_PREDICATE,
    )


def downgrade() -> None:
    op.drop_index(
        "uq_outreach_active_x_user_id",
        table_name="outreach_records",
        postgresql_where=_NEW_PREDICATE,
    )
    op.create_index(
        "uq_outreach_active_x_user_id",
        "outreach_records",
        ["x_user_id"],
        unique=True,
        postgresql_where=_OLD_PREDICATE,
    )

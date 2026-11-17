"""Tests for the development guardrail helpers in ``scripts/``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from seed_test_contacts import read_handles  # noqa: E402


def test_read_handles_skips_blanks_and_comments(tmp_path: Path) -> None:
    listing = tmp_path / "contacts.txt"
    listing.write_text(
        "# Accounts we control\n"
        "\n"
        "1234567890\n"
        "@another_account\n"
        "  spaced_handle  \n",
        encoding="utf-8",
    )

    assert read_handles(listing) == ["1234567890", "another_account", "spaced_handle"]


def test_read_handles_ignores_comment_only_files(tmp_path: Path) -> None:
    listing = tmp_path / "contacts.txt"
    listing.write_text("# nothing here\n\n", encoding="utf-8")

    assert read_handles(listing) == []


@pytest.mark.asyncio
async def test_seed_suppresses_everything_outside_the_opt_in_list(
    db_session,
) -> None:
    """Discovered accounts that are not opted in must be suppressed."""
    from seed_test_contacts import seed

    from app.models import DiscoveredAccount

    db_session.add(
        DiscoveredAccount(
            x_user_id="u1",
            username="opted_in",
            display_name="Opted In",
            followers_count=10,
            following_count=5,
            tweets_count=1,
            bot_score=0.0,
            bot_verdict="human",
            bot_reasons=[],
            key_info={},
            segment_tags=[],
        )
    )
    db_session.add(
        DiscoveredAccount(
            x_user_id="u2",
            username="stranger",
            display_name="Stranger",
            followers_count=10,
            following_count=5,
            tweets_count=1,
            bot_score=0.0,
            bot_verdict="human",
            bot_reasons=[],
            key_info={},
            segment_tags=[],
        )
    )
    await db_session.flush()

    from app.services import dedup

    await seed(["opted_in"], apply=True, session=db_session)

    assert await dedup.is_suppressed(db_session, "u2") is True
    assert await dedup.is_suppressed(db_session, "u1") is False

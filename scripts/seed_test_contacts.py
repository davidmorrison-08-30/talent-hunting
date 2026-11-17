"""Guardrail helper for development sends.

Real sends during development must only reach accounts you control or that
explicitly opted in. This script makes that safe by default:

* every account in the **opt-in list** is the only thing you may contact;
* every account in the **discovery results** that is *not* on the opt-in list
  is added to the permanent suppression list, so it can never be drafted to;
* accounts already contacted are left untouched (the dedup ledger wins).

Usage::

    uv run python scripts/seed_test_contacts.py --list opted-in.txt
    uv run python scripts/seed_test_contacts.py --list opted-in.txt --apply

Without ``--apply`` the script only reports what it would change.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import dispose_engine, get_sessionmaker
from app.models import DiscoveredAccount
from app.services import suppression as suppression_service


def read_handles(path: Path) -> list[str]:
    """Read one X user id or @handle per line, ignoring blanks and comments."""
    handles: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        handles.append(line.lstrip("@"))
    return handles


async def seed(
    opt_in: list[str], *, apply: bool, session: AsyncSession | None = None
) -> int:
    """Suppress every discovered account that is not explicitly opted in."""
    allowed = {handle.lower() for handle in opt_in}

    if session is not None:
        return await _seed_in_session(session, allowed, apply=apply)

    async with get_sessionmaker()() as own_session:
        return await _seed_in_session(own_session, allowed, apply=apply)


async def _seed_in_session(
    session: AsyncSession, allowed: set[str], *, apply: bool
) -> int:
    """Do the suppression work inside an open session."""
    result = await session.execute(select(DiscoveredAccount))
    accounts = list(result.scalars().all())

    to_suppress = [
        account
        for account in accounts
        if (account.username or "").lower() not in allowed
        and account.x_user_id not in allowed
    ]

    for account in to_suppress:
        if apply:
            await suppression_service.suppress(
                session,
                x_user_id=account.x_user_id,
                handle=account.username,
                reason="Not on the development opt-in list",
            )
        print(f"suppress  {account.username} ({account.x_user_id})")

    if apply:
        await session.commit()

    print(
        f"\n{len(to_suppress)} of {len(accounts)} discovered accounts are "
        f"not on the opt-in list "
        f"({'suppressed' if apply else 'would be suppressed'})."
    )
    print(f"{len(allowed)} account(s) remain contactable.")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        type=Path,
        default=Path("test-contacts.txt"),
        help="File with one opted-in X user id or @handle per line.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the suppressions. Without it, only report.",
    )
    args = parser.parse_args(argv)

    if not args.list.exists():
        print(
            f"No opt-in list found at {args.list}. Create it with one X user id "
            "or @handle per line, then re-run.",
            file=sys.stderr,
        )
        return 1

    opt_in = read_handles(args.list)
    if not opt_in:
        print(f"{args.list} contains no handles.", file=sys.stderr)
        return 1

    async def _run() -> int:
        try:
            return await seed(opt_in, apply=args.apply)
        finally:
            await dispose_engine()

    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())

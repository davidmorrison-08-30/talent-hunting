"""Stage 2 of discovery: turn a target segment into a search query plan.

The two X search endpoints accept very different syntax, so the plan keeps them
separate and every query is sanitized before it reaches the API:

* ``/2/users/search`` accepts plain text only, max 50 characters.
* tweet search accepts the full operator language, max 512 characters
  (self-serve) or 1024 (full archive).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.llm.bedrock import BedrockLLM, get_llm
from app.llm.prompts import (
    ANTI_FABRICATION,
    QUERY_PLAN_SCHEMA,
    QUERY_PLAN_SYSTEM,
    QUERY_PLAN_USER,
)

log = structlog.get_logger(__name__)

#: X caps a user-search query at 50 characters.
USER_QUERY_MAX_LENGTH = 50
#: Self-serve tweet search caps a query at 512 characters.
POST_QUERY_MAX_LENGTH = 512

#: Characters X's user search accepts: letters, digits, spaces, underscores, apostrophes.
_USER_QUERY_ALLOWED = re.compile(r"[^A-Za-z0-9 _']")

#: Search-operator words that must not leak into a plain-text user query.
_USER_QUERY_STOPWORDS = frozenset(
    {
        "and",
        "or",
        "not",
        "is",
        "retweet",
        "replies",
        "reply",
        "filter",
        "links",
        "lang",
        "from",
        "to",
        "min",
        "max",
        "faves",
        "retweets",
        "since",
        "until",
        "null",
    }
)


@dataclass
class QueryPlan:
    """The set of queries to run against X."""

    user_queries: list[str] = field(default_factory=list)
    post_queries: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_payload(self) -> list[dict[str, str]]:
        """A flat, log-friendly description of the queries used."""
        return [
            {"endpoint": "/2/users/search", "query": q} for q in self.user_queries
        ] + [{"endpoint": "/2/tweets/search", "query": q} for q in self.post_queries]


def sanitize_user_query(query: str) -> str | None:
    """Reduce a query to what ``/2/users/search`` accepts.

    Strips operator punctuation and operator keywords, since ``/2/users/search``
    matches plain text only and would otherwise index words like "retweet".

    Returns ``None`` when nothing usable remains.
    """
    cleaned = _USER_QUERY_ALLOWED.sub(" ", query or "")
    words = [w for w in re.split(r"\s+", cleaned) if w]
    words = [w for w in words if w.lower() not in _USER_QUERY_STOPWORDS]
    cleaned = " ".join(words).strip()
    if not cleaned:
        return None
    if len(cleaned) > USER_QUERY_MAX_LENGTH:
        cleaned = cleaned[:USER_QUERY_MAX_LENGTH].rstrip()
    return cleaned or None


#: X web-search operator names that the API rejects with HTTP 400, mapped to
#: their valid API equivalents.
#:
#: ``min_faves:`` and ``min_retweets:`` are X *web search* syntax. The API
#: spells them ``min_likes:`` and ``min_reposts:``.
_POST_QUERY_OPERATOR_ALIASES: dict[str, str] = {
    "min_faves:": "min_likes:",
    "min_retweets:": "min_reposts:",
}

#: The API rejects a bare ``AND`` with
#: ``Ambiguous use of and as a keyword. Use a space to logically join two
#: clauses, or "and" to find occurrences of and in text``. A plain space is the
#: API's AND. ``OR`` is a valid API operator, but a lowercase ``or`` is not,
#: so only the uppercase spelling is treated as an operator.
#:
#: The keyword is only replaced when it stands alone as a word, so a quoted
#: phrase such as ``"cats and dogs"`` is left untouched.
#: A tweet-search query must contain at least one positive, non-stopword
#: clause. X rejects a query that is only negations with
#: ``Rules must contain a non-negation term``.
_POST_QUERY_NEGATION_PREFIX = "-"


def _rewrite_boolean_keywords(query: str) -> str:
    """Replace API-rejected boolean keywords with X's API equivalents.

    ``"AI researcher" AND blog`` becomes ``"AI researcher" blog`` (a space is
    the API's AND) and ``NOT crypto`` becomes ``-crypto``. Occurrences inside
    double quotes are preserved, because a quoted phrase is matched literally.
    """

    def _replace_segment(segment: str) -> str:
        # ``AND`` is the API's implicit conjunction, so it becomes a space.
        segment = re.sub(
            r"(?<![A-Za-z0-9_])AND(?![A-Za-z0-9_])", " ", segment
        )
        # ``NOT`` must hug the term it negates: ``-crypto``, never ``- crypto``.
        segment = re.sub(
            r"(?<![A-Za-z0-9_-])NOT\s+(?=[^\s-])", "-", segment
        )
        return segment

    # Split on double-quoted spans and only rewrite the unquoted parts.
    parts = re.split(r'("[^"]*")', query)
    return "".join(
        part if part.startswith('"') else _replace_segment(part) for part in parts
    )


def _has_positive_clause(query: str) -> bool:
    """Whether a tweet-search query contains a positive (non-negated) term."""
    for token in re.split(r"[\s()]+", query):
        token = token.strip().strip('"')
        if not token:
            continue
        if token.startswith(_POST_QUERY_NEGATION_PREFIX):
            continue
        # An operator such as ``lang:en`` is a term for this purpose; X
        # accepts it as long as it is not the only clause.
        return True
    return False


def sanitize_post_query(query: str) -> str | None:
    """Normalise a tweet-search query so X accepts it.

    Rewrites the X web-search operator spellings that the API rejects
    (``min_faves:`` -> ``min_likes:``) and the boolean keywords the API
    rejects (``AND`` -> a space, ``NOT`` -> ``-``), then trims the query to
    the API's length limit.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return None

    for invalid, valid in _POST_QUERY_OPERATOR_ALIASES.items():
        cleaned = cleaned.replace(invalid, valid)
        # The negated form (-min_faves:10) is invalid in the same way.
        cleaned = cleaned.replace(f"-{invalid}", f"-{valid}")

    cleaned = _rewrite_boolean_keywords(cleaned)
    # Collapse the whitespace left behind by a removed ``AND`` keyword.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not _has_positive_clause(cleaned):
        log.warning("post_query_dropped_no_positive_clause", query=cleaned)
        return None

    if len(cleaned) > POST_QUERY_MAX_LENGTH:
        cleaned = cleaned[:POST_QUERY_MAX_LENGTH].rstrip()
    return cleaned or None


async def build_query_plan(
    keywords: list[str],
    exclude_keywords: list[str] | None = None,
    llm: BedrockLLM | None = None,
) -> QueryPlan:
    """Ask the LLM for a query plan, then sanitize every query."""
    llm = llm or get_llm()

    if not keywords:
        return QueryPlan(rationale="No keywords were provided.")

    keyword_text = "\n".join(f"- {k}" for k in keywords)
    if exclude_keywords:
        keyword_text += "\n\nExclude these:\n" + "\n".join(
            f"- {k}" for k in exclude_keywords
        )

    try:
        result = await llm.converse_json(
            system=QUERY_PLAN_SYSTEM,
            user=QUERY_PLAN_USER.format(
                keywords=keyword_text,
                anti_fabrication=ANTI_FABRICATION,
            ),
            schema=QUERY_PLAN_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("query_plan_failed", error=str(exc))
        return _fallback_plan(keywords, exclude_keywords)

    user_queries = _dedupe(
        [
            q
            for q in (
                sanitize_user_query(str(x)) for x in result.get("user_queries") or []
            )
            if q
        ]
    )
    post_queries = _dedupe(
        [
            q
            for q in (
                sanitize_post_query(str(x)) for x in result.get("post_queries") or []
            )
            if q
        ]
    )

    # Guarantee at least one usable query on each side.
    if not user_queries:
        user_queries = _fallback_plan(keywords, exclude_keywords).user_queries
    if not post_queries:
        post_queries = _fallback_plan(keywords, exclude_keywords).post_queries

    return QueryPlan(
        user_queries=user_queries,
        post_queries=post_queries,
        rationale=str(result.get("rationale") or ""),
    )


def _fallback_plan(
    keywords: list[str], exclude_keywords: list[str] | None
) -> QueryPlan:
    """Build a usable plan without the LLM, straight from the keywords."""
    user_queries: list[str] = []
    post_queries: list[str] = []

    for keyword in keywords[:8]:
        cleaned = sanitize_user_query(keyword)
        if cleaned:
            user_queries.append(cleaned)

    if keywords:
        phrases = " OR ".join(f'"{k}"' for k in keywords[:6])
        query = f"({phrases}) -is:retweet"
        if exclude_keywords:
            exclusions = " ".join(f'-"{k}"' for k in exclude_keywords[:3])
            query = f"{query} {exclusions}"
        sanitized = sanitize_post_query(query)
        if sanitized:
            post_queries.append(sanitized)

    return QueryPlan(
        user_queries=_dedupe(user_queries),
        post_queries=_dedupe(post_queries),
        rationale="Generated from the raw keyword list without LLM assistance.",
    )


def _dedupe(items: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def parse_query_response(data: dict[str, Any]) -> tuple[list[str], str | None]:
    """Extract the candidate ids and pagination token from a search response."""
    users = data.get("data") or []
    ids = [str(u.get("id")) for u in users if isinstance(u, dict) and u.get("id")]
    next_token = (data.get("meta") or {}).get("next_token")
    return ids, next_token

# X Talent-Hunting APIs

A FastAPI service that **finds real (non-bot) accounts on X** from a target segment or a set of seed
accounts, and **chats with them under human approval** — evidence-grounded drafts, a mandatory review
queue, paced dispatch, a hard dedup ledger and a permanent opt-out list.

Everything is built around the compliance rules in [`build-docs/`](build-docs/): no message is ever sent
without a human approving it, no recipient is ever contacted twice, no suppressed account is ever
contacted at all, and no burst of messages ever leaves the service.

---

## Table of contents

1. [Architecture](#architecture)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Connecting an X account](#connecting-an-x-account)
5. [AWS Bedrock](#aws-bedrock)
6. [API reference](#api-reference)
7. [How the discovery agent works](#how-the-discovery-agent-works)
8. [How the outreach agent works](#how-the-outreach-agent-works)
9. [Compliance controls](#compliance-controls)
10. [X API access tier & costs](#x-api-access-tier--costs)
11. [Development guardrails](#development-guardrails)
12. [Testing](#testing)
13. [Project layout](#project-layout)

---

## Architecture

```
                         ┌──────────────────────────────────────────┐
   operator ──HTTP──▶    │  FastAPI app  (app/api/*)                │
                         │                                          │
                         │  /api/v1/discovery/search   (ENDPOINT I) │
                         │      │                                   │
                         │      ▼                                   │
                         │  app/agents/segment  → target segment    │
                         │  app/agents/queries  → query plan        │
                         │  app/agents/search   → search + hydrate  │
                         │  app/agents/botdetect → hybrid scoring   │
                         │  app/agents/profile  → key information   │
                         │                                           │
                         │  /api/v1/outreach/drafts   (ENDPOINT II) │
                         │      │                                    │
                         │      ▼                                    │
                         │  app/agents/drafts   → draft / revise    │
                         │  app/services/dedup    → hard dedup lock  │
                         │  app/services/suppression → opt-out list  │
                         │  app/services/pacing   → cap + delays     │
                         │  app/services/dispatcher → drip queue     │
                         │  app/services/conversations → DM events   │
                         └───────────┬───────────────────┬──────────┘
                                     │                   │
                       ┌─────────────▼──────┐   ┌────────▼─────────────────┐
                       │  AWS Bedrock       │   │  X API v2 (OAuth 2.0)  │
                       │  Converse + tool   │   │  users / tweets / DMs  │
                       │  use → JSON schema │   │  rate limits, probes   │
                       └────────────────────┘   └────────────────────────┘
                                     │                   │
                       ┌─────────────▼───────────────────▼──────────┐
                       │  PostgreSQL 16 (asyncpg + SQLAlchemy 2)   │
                       │  audit trail, dedup ledger, budget, cache │
                       └────────────────────────────────────────────┘
```

Two **primary endpoints** carry the product; a small set of supporting compliance endpoints makes the
required approval, dedup and opt-out behaviour reachable. Every request/response is a Pydantic model,
so the full contract lives at `/docs`.

---

## Requirements

| Component | Version | Notes |
|---|---|---|
| Python | 3.12+ | The project pins `3.12` in `.python-version`. |
| [`uv`](https://docs.astral.sh/uv/) | latest | Python env + dependency manager. |
| PostgreSQL | 16 | Shipped in `docker-compose.yml`. |
| Docker | any recent | Only used for PostgreSQL. |
| AWS credentials | — | Bedrock access, plus the model id. |
| X developer app | — | OAuth 2.0 consumer key/secret. |

---

## Setup

### 1. Install `uv` and create the environment

```bash
# Install uv (Linux/macOS) — installs to ~/.local/bin/uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# No curl? Either alternative works:
#   pipx install uv
#   python3 -m pip install --user uv

uv --version
```

Then, from the repo root:

```bash
uv python install 3.12     # provision the pinned interpreter
uv venv --python 3.12      # creates .venv/ (gitignored)
uv sync                    # installs exactly what uv.lock pins
```

`uv sync` is the single command that installs every dependency from the committed `uv.lock`. Use
`uv sync --frozen` in CI to guarantee the lock file is not modified. Every command below is run
through `uv run` so it always uses the project environment.

### 2. Start PostgreSQL

```bash
docker compose up -d postgres
```

The default compose service listens on `localhost:5432` with `postgres`/`postgres` and creates the
`talent_hunting` database. For the test suite, create a second database:

```bash
docker compose exec postgres createdb -U postgres talent_hunting_test
```

### 3. Configure the environment

```bash
cp .env.example .env
$EDITOR .env
```

The important keys:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Async SQLAlchemy URL for PostgreSQL. |
| `X_CLIENT_ID` / `X_CLIENT_SECRET` | OAuth 2.0 credentials of your X developer app. |
| `X_REDIRECT_URI` | Must match a redirect URI registered on the X app. |
| `AWS_BEDROCK_MODEL_ID` | e.g. `anthropic.claude-3-5-sonnet-20241022-v2:0`. |
| `AWS_REGION` | Bedrock region (`us-east-1` by default). |
| `DM_DAILY_CAP` | Application-level daily DM ceiling (default `25`). |
| `DM_MIN_DELAY_SECONDS` / `DM_MAX_DELAY_SECONDS` | Randomized pre-send delay window (default `30`–`90`). |
| `DM_SIMILARITY_THRESHOLD` | Ratio above which a draft counts as copypasta (default `0.85`). |
| `DISCOVERY_SPEND_CEILING_USD` | Per-run pay-per-use guardrail (default `5.0`). |

### 4. Apply the database migrations

```bash
uv run alembic upgrade head
```

### 5. Run the service

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Then open <http://localhost:8000/docs> for the interactive OpenAPI UI.

Migrations also run automatically on startup unless `RUN_MIGRATIONS_ON_STARTUP=false`.

---

## Connecting an X account

The service uses the **OAuth 2.0 PKCE user-context** flow (no bot account, no password scraping).

1. Register a redirect URI on your X developer app that matches `X_REDIRECT_URI`
   (default `http://localhost:8000/auth/x/callback`).
2. Request the scopes `tweet.read users.read dm.read dm.write offline.access`.
3. Start the flow:

   ```bash
   curl http://localhost:8000/auth/x/start
   ```

   The response contains an `authorization_url`. Open it in a browser and authorize the **human**
   operator account.

4. X redirects to `/auth/x/callback?code=…&state=…`. The service exchanges the code, stores the
   access/refresh token pair and shows a confirmation page.
5. Check the connection:

   ```bash
   curl http://localhost:8000/auth/x/status
   ```

Access tokens live about two hours; refresh tokens about six months and are **single-use**. The client
transparently refreshes on `401` or on expiry (with a 60-second skew) and overwrites the stored
refresh token atomically each time.

> **X Premium (Verified) strongly recommended.** Cold DMs to accounts that do not accept DMs from
> non-followers are silently dropped. A verified sending account dramatically improves deliverability.

---

## AWS Bedrock

The LLM layer wraps the Bedrock **Converse** API and forces **structured output** by declaring a JSON
Schema as a single tool (`toolConfig`). The model must call that tool, so the reply is always valid
JSON of the expected shape — no regex parsing, no free-text parsing.

Prompts live in `app/llm/prompts.py` as versioned constants and carry explicit anti-fabrication rules:

> Use only the evidence provided. If the evidence is insufficient, say so instead of inventing details.

Concretely, the model must never fabricate a post, a quote, a URL, a job title or an interest. Every
draft carries the citations it relied on so a reviewer can verify them.

Credentials are resolved from the standard AWS chain (environment, shared config, or
`AWS_DEFAULT_PROFILE`).

---

## API reference

Full interactive documentation: <http://localhost:8000/docs>.

### Health & capabilities

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe. |
| `GET` | `/api/v1/health` | Health with database / Bedrock / OAuth / token-validity / rate-budget checks. |
| `GET` | `/api/v1/x/capabilities` | Live probe of what the connected account can actually call. |

### OAuth

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/auth/x/start` | Begin the PKCE flow; returns the authorization URL. |
| `GET` | `/auth/x/callback` | Complete the PKCE flow and store the token. |
| `GET` | `/auth/x/status` | Whether a usable token is stored. |
| `GET` | `/auth/x/disconnect` | Delete the stored token. |

### ENDPOINT I — discovery

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/discovery/search` | Find accounts matching a segment and/or seeds. |
| `GET` | `/api/v1/discovery/runs` | List recent discovery runs. |
| `GET` | `/api/v1/discovery/runs/{run_id}` | One run with its statistics. |
| `GET` | `/api/v1/discovery/accounts` | Accounts found so far (filter by verdict, followers, tag). || `GET` | `/api/v1/discovery/accounts/{x_user_id}` | One account with its key information. |

Request body:

```json
{
  "target_segment": ["robotics engineer", "embodied AI"],
  "seed_accounts": ["https://x.com/someone"],
  "known_human_accounts": ["https://x.com/a-friend"],
  "max_accounts": 2,
  "max_candidates": 5,
  "max_posts_per_account": 11,
  "include_bots": false,
  "bot_score_threshold": 0.6
}
```

At least one of `target_segment` / `seed_accounts` is required; both may be supplied and are merged.
Response includes `run_id`, the derived `target_segment`, `queries_used`, the surviving `accounts`
(with `bot` assessment and `key_information`), `stats` (candidates, bots filtered, API calls per
endpoint, estimated cost), live `capabilities` and any `warnings`.

### ENDPOINT II — outreach

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/outreach/drafts` | Draft a grounded DM and park it in `PENDING_APPROVAL`. |
| `GET` | `/api/v1/outreach/drafts` | List outreach records (filter by status / recipient). |
| `GET` | `/api/v1/outreach/drafts/{record_id}` | One record with its evidence snapshot. |
| `POST` | `/api/v1/outreach/drafts/{record_id}/approve` | Approve and send (paced). |
| `POST` | `/api/v1/outreach/drafts/{record_id}/deny` | Reject with feedback and get a revision. |
| `POST` | `/api/v1/outreach/poll-events` | Poll for inbound replies; drafts responses for approval. |
| `POST` | `/api/v1/outreach/dispatch/resume` | Restart the drip dispatcher and drain anything left queued. |
| `GET` | `/api/v1/outreach/queue` | Everything awaiting a human decision. |
| `GET` | `/api/v1/outreach/budget` | Today's dispatch and read budget. |
| `GET` | `/api/v1/outreach/conversations/{x_user_id}` | The cached DM history with one recipient. |

Draft request:

```json
{ "x_user_id": "…", "goal": "Invite to a robotics meetup in SEA", "tone": "concise, warm" }
```

The response carries the draft text, the citations it relied on, the full evidence snapshot and the
DM conversation history so the reviewer can judge the personalization.

`approve` accepts `?dispatch=blocking` (default — sends inside the request, honouring the randomized
30–90 s delay) or `?dispatch=queued` (hands the record to the in-process drip dispatcher, which
sends sequentially with the same pacing and survives restarts because the queue lives in Postgres).

### Compliance

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/suppressions` | Permanently suppress an account ("Do Not Contact"). Cancels pending drafts and removes the account from discovery results. |
| `GET` | `/api/v1/suppressions` | List suppressed accounts. |
| `GET` | `/api/v1/outreach/records` | The outreach ledger (filter by recipient / status). |
| `GET` | `/api/v1/outreach/records/{record_id}` | One ledger entry. |

`GET /api/v1/discovery/accounts` hides suppressed accounts by default; pass
`?include_suppressed=true` to audit them.

### Errors

Every error uses one envelope:

```json
{
  "error": {
    "code": "duplicate_outreach",
    "message": "An outreach record already exists for this account",
    "details": { "x_user_id": "…" }
  }
}
```

| Status | Codes |
|---|---|
| `404` | `not_found` |
| `409` | `duplicate_outreach`, `suppressed`, `daily_cap_reached`, `conflict` |
| `422` | `validation_error` |
| `429` | `rate_limited` |
| `502` | `upstream_error`, `llm_error` |
| `503` | `configuration_error` |

A `403` from X on a DM send is **not** an error response: the record is marked `BLOCKED` and returned
with `status: "BLOCKED"` so a partial batch is never lost.

---

## How the discovery agent works

`POST /api/v1/discovery/search` runs this pipeline:

1. **Seed analysis** — resolve each seed handle, pull its posts newest → oldest, and extract bio and
   post text.
2. **Seed screening** — the seed accounts are checked against the suppression list and the bot verdict
   before anything else happens, because their verdict decides how many slots the search may still fill.
   A seed that is filtered out frees its slot instead of consuming one.
3. **Segment derivation** — the LLM turns that content into a target segment: keyword list plus
   optional `exclude_keywords`. If the caller also supplied `target_segment`, the two are merged.
4. **Query formation** — the LLM emits two kinds of query:
   - plain-text queries for `GET /2/users/search` (sanitized to ≤50 characters, operator words such as
     `is`, `retweet` and `lang` stripped, because that endpoint matches plain text only);
   - operator-rich queries for `GET /2/tweets/search/recent` — or `/all` when the capability probe
     confirmed full-archive access. Post queries are normalised before use: X web-search operators
     (`min_faves:`, `min_retweets:`) are rewritten to their API spellings (`min_likes:`,
     `min_reposts:`), and a query with no positive clause is dropped, because X rejects both with
     HTTP 400.
5. **Search and hydration** — candidates are collected from both searches, hydrated in batches of 100
   via `GET /2/users` (the by-IDs endpoint; `GET /2/users/by` is the by-usernames endpoint), and their
   recent posts fetched for evidence and bot signals.
6. **Hybrid bot detection** — deterministic heuristics (see below) plus an LLM verdict, combined into
   one score. `bot_score >= bot_score_threshold` excludes the account. Every account keeps its score
   and reasons for audit, and `include_bots=true` returns them instead of dropping them.
7. **Key-information extraction** — the LLM produces `bio`, `occupation`, `titles`,
   `specializations`, `interests`, `milestones`, `future_plans` and `evidence[]`, grounded in the real
   fetched content only.
8. **Persistence** — accounts are upserted into `discovered_accounts` with a `source_run_id`, and
   `discovery_runs.stats` records candidates seen, bots filtered, API calls per endpoint and estimated
   cost.

### The seed accounts come first

The caller named the seed accounts explicitly, so they are the most relevant accounts a run can
return, and they are always placed at the front of the results:

- the seeds are returned in the order the caller supplied them;
- the number of **other** accounts is `max_accounts` minus the number of seeds that survived the
  filters;
- a seed that is suppressed or flagged as a bot is not returned and does not use up a slot — the
  search gets that slot instead;
- when the seeds fill `max_accounts` on their own, **no X search is performed at all**: the query
  plan, the capability probe and the search endpoints are all skipped, so the run costs nothing
  beyond resolving the seeds themselves.

The run statistics report this as `seeds_supplied`, `seeds_resolved`, `seeds_included` and
`seeds_dropped`. A seed is only dropped when the caller supplied more seeds than `max_accounts`
allows; the first ones in the caller's order are kept.

### Known humans skip bot detection

`known_human_accounts` names accounts the operator personally vouches for. A matching account — a
seed or one the search found — **skips bot detection entirely**: it is recorded as `human` with a
score of `0.0` and is never filtered out, even when the heuristics would have flagged it.

The entries accept the same handle forms as the seed accounts (`https://x.com/someone`, `@someone`
and `someone` all refer to the same account), and the match is case-insensitive. The list is a
bypass, not a candidate source: it never adds an account to the results on its own, it only exempts
an account that the run would otherwise have assessed.

The suppression list still wins. An account that asked not to be contacted is excluded even when
the operator also named it as a known human, because the opt-out is permanent.

The run statistics report how many accounts took the bypass as `humans_bypassed`, and the accounts
themselves carry the `known_human` tag.

### Bot detection heuristics

Taken directly from `build-docs/bot-detection.md`:

- extreme following ≫ followers ratio;
- empty, single-issue or link-only bio;
- blank/default avatar or a generated `Firstname12345`-style handle;
- account created after 2023-10 combined with high posting volume;
- near-24/7 posting cadence (no human sleep gap across the fetched window);
- repeated or templated post text;
- high follower count with near-zero engagement on recent posts;
- no `Automated by`-style label despite machine-like behaviour.

The heuristics produce weighted signals; the LLM returns `{is_bot, confidence, reasons[]}`; the two are
combined into a single score with the reasons preserved.

---

## How the outreach agent works

1. **Draft** (`POST /api/v1/outreach/drafts`) — pre-flight checks run first (suppression list, existing
   live record, daily cap). The service then loads the stored key information, fetches the recipient's
   latest public posts as fresh evidence, and pulls the DM history (cached locally). The LLM writes a
   message that **must** reference at least one concrete, verified detail and **must not** contain bare
   links, generic flattery or templated phrasing. It returns the citations it used. The draft is stored
   as `PENDING_APPROVAL` with its evidence snapshot. **Nothing is sent.**
2. **Review** — a human reads the draft against the evidence.
   - **Approve** → `POST …/approve` re-checks every guard immediately before sending, applies the
     randomized delay, sends, and records the result.
   - **Deny** → `POST …/deny` with feedback regenerates the message. The previous version stays in the
     history, so the whole review trail is auditable, and the new version returns to
     `PENDING_APPROVAL`.
3. **Reply handling** — `POST /api/v1/outreach/poll-events` polls `GET /2/dm_events`, caches new events
   and creates a reply draft for each inbound message. The reply still waits for approval.

---

## Compliance controls

| Rule from `build-docs/x-rules-and-limits.md` | How it is enforced |
|---|---|
| Mandatory human approval queue | Every draft starts in `PENDING_APPROVAL`; the only path to `SENT` is an explicit approve call. |
| Evidence-backed personalization | The prompt requires concrete cited details; the response carries the citations and the evidence snapshot for review. |
| Hard deduplication lock | A partial unique index on `outreach_records(x_user_id)` where the status is live, plus a service-level check before drafting *and* again before sending. |
| Paced dispatch (drip queue) | Randomized 30–90 s delay per send; `dispatch=queued` drains sequentially, never in a burst. |
| Opt-out / suppression list | `POST /api/v1/suppressions` blocks all future outreach permanently, cancels any pending draft, and removes the account from discovery results. |
| No copypasta | Every send is compared against recently sent bodies; a near-duplicate is rejected back to `PENDING_APPROVAL` with a note. |
| No link-only spam | Explicitly forbidden by the prompt; the draft must carry real context. |
| Daily cap | Application-level ceiling (default 25/day) checked at draft time and again before sending. |

Additional guardrails beyond the minimum:

- **TOCTOU protection** — dedup and suppression are re-checked inside the send path, so an opt-out
  that arrives between drafting and approval still blocks the message.
- **Capability detection** — the service never assumes a subscription tier; it probes the live account.
- **Spend ceiling** — a discovery run refuses to exceed `DISCOVERY_SPEND_CEILING_USD`.
- **Rate-limit awareness** — per-endpoint sliding windows mirror X's published ceilings, and the
  `x-rate-limit-remaining` / `x-rate-limit-reset` headers are honoured so a window pause skips the
  wait instead of sleeping blindly.

---

## X API access tier & costs

X retired the Basic and Pro tiers in early 2026 and moved to **pay-per-use** and **Enterprise**
access. Which endpoints an account can actually call therefore depends on the live entitlements of the
connected account, not on a plan name. The service therefore **probes at runtime** and exposes the
result at `GET /api/v1/x/capabilities`:

| Capability | How it is detected | Fallback |
|---|---|---|
| Full-archive tweet search (`/2/tweets/search/all`) | Probed once, cached for an hour | Falls back to `/2/tweets/search/recent` (last 7 days) |
| Recent tweet search | Probed once, cached for an hour | — |
| User-context authentication | `GET /2/users/me` | — |
| DM read / write | Inferred from the granted OAuth scopes | — |

### Published rate limits

| Endpoint | Per user | Per app |
|---|---|---|
| `GET /2/users/search` | 900 / 15 min | 300 / 15 min |
| `GET /2/tweets/search/recent` | 300 / 15 min | 450 / 15 min |
| `GET /2/users` (by IDs) | 900 / 15 min | 300 / 15 min |
| `GET /2/users/by` (by usernames) | 900 / 15 min | 300 / 15 min |
| `GET /2/users/:id/tweets` | 900 / 15 min | — |
| `POST /2/dm_conversations/…/messages` | 15 / 15 min | 1,440 / 24 h |
| `GET /2/dm_events` | 15 / 15 min | — |

DM sending is additionally bound by the **500 DMs/day account ceiling** (a verified account may DM any
user); unverified accounts face a much lower, unpublished limit. This service treats 500/day as a hard
ceiling and enforces a far more conservative application-level cap on top of it.

### Expected costs

| Item | Approximate cost |
|---|---|
| X pay-per-use reads | Priced per post and per user record. Defaults used for estimation: **$0.005/post read**, **$0.010/user read**. |
| `GET /2/tweets/counts/recent` | **Not used** — it is billed per request and only would size a query. |
| AWS Bedrock | Per model invocation. Every LLM call is one structured tool-use request. |

The estimated spend of each discovery run is returned in `stats.estimated_cost_usd` and accumulated in
the daily budget (`GET /api/v1/outreach/budget`). A run refuses to exceed
`DISCOVERY_SPEND_CEILING_USD` (default `$5.00`). The numbers above are configuration defaults, not
vendor quotes — check your X developer portal for the current pay-per-use pricing of your account.

---

## Development guardrails

> **Real sends during development must only target accounts you control or that explicitly opted in.**

This is not optional — it is the rule the compliance document sets, and violating it risks the
sending account and the developer app.

**Before running any real approval:**

1. **List your test recipients.** Keep a short, explicit list of the X accounts you own or that opted
   in to being contacted. Write it down and keep it next to the service:

   ```text
   # test-contacts.txt — accounts that opted in (one X user id or @handle per line)
   1234567890
   @my_other_account
   ```

2. **Restrict discovery to your own segment** while testing, so only opted-in accounts surface:

   ```bash
   curl -X POST http://localhost:8000/api/v1/discovery/search \
     -H 'content-type: application/json' \
     -d '{"target_segment": ["my name", "my company"], "max_accounts": 5}'
   ```

3. **Suppress everyone else immediately.** Once a non-test account appears, suppress it before it can
   be drafted to:

   ```bash
   curl -X POST http://localhost:8000/api/v1/suppressions \
     -H 'content-type: application/json' \
     -d '{"x_user_id": "…", "handle": "…", "reason": "not a test contact"}'
   ```

   Or let the helper do it for every discovered account at once. Keep the opt-in list in a file and
   run the script after each discovery round:

   ```bash
   uv run python scripts/seed_test_contacts.py --list test-contacts.txt          # dry run
   uv run python scripts/seed_test_contacts.py --list test-contacts.txt --apply  # write it
   ```

   The script suppresses every account in `discovered_accounts` that is **not** on the opt-in list, so
   a stray discovery result can never become a DM. Accounts already contacted are left alone — the
   dedup ledger wins.

4. **Use `include_bots=false` and a low `max_accounts`** — a smaller blast radius is a safer blast
   radius.

5. **Prefer `dispatch=queued` for the first sends** so you can watch the drip dispatcher work and stop
   it between messages.

6. **Never disable the pacing delay or the daily cap** to "make a test faster". Set
   `DM_MIN_DELAY_SECONDS` / `DM_MAX_DELAY_SECONDS` low *in your local `.env` only* if you must, and
   keep `DM_DAILY_CAP` realistic.

7. **After a restart, resume the queue explicitly** if you had queued sends in flight:

   ```bash
   curl -X POST http://localhost:8000/api/v1/outreach/dispatch/resume
   ```

### Other guardrails

- **No scraping.** The service uses only the official X API v2. It does not use headless browsers,
  browser extensions, unofficial private endpoints or session cookies.
- **No credential in code.** Secrets are read from the environment only; `.env` is git-ignored.
- **Audit trail.** Every draft version, feedback, citation, send, block and error is persisted in
  `outreach_records`, and every discovery run in `discovery_runs`.
- **Rate-limit politeness.** `GET /2/dm_events` is limited to 15 calls per 15 minutes, so replies are
  polled on a conservative schedule (`DM_POLL_INTERVAL_SECONDS`, default 240 s) and every event is
  cached locally. Polling and sending do not share a budget.

---

## Testing

```bash
uv run pytest              # the whole suite
uv run pytest -q tests/test_outreach.py
```

The suite is fully offline:

- the X API is mocked with **`respx`** (`httpx` transport-level mocking);
- Bedrock is replaced by a deterministic fake that returns the same schema-conformant payload;
- the database is a real PostgreSQL schema that is dropped and recreated per test.

Coverage highlights:

| File | What it covers |
|---|---|
| `tests/test_discovery.py` | Segment derivation, query formation and sanitization, each bot heuristic, the hybrid verdict, key-information extraction, the full search loop. |
| `tests/test_outreach.py` | Dedup lock, suppression, daily cap, pacing window, drafting, denial → revision, approval and send, `403` → `BLOCKED`, copypasta rejection. |
| `tests/test_api.py` | Every route: validation, status codes, response shapes, the error envelope, OAuth start/status/disconnect. |
| `tests/test_services.py` | Rate limiting (sliding window, server headers, daily ceiling), PKCE helpers, token refresh, DM event polling, the drip dispatcher, budget rollover. |

Quality gates:

```bash
uv run ruff check app tests
uv run mypy app
```

---

## Project layout

```
app/
  main.py              FastAPI app, exception handlers, router registration
  config.py            Pydantic Settings (env-driven)
  db.py                Async engine, sessionmaker, FastAPI dependency
  models.py            SQLAlchemy 2.0 ORM models
  schemas.py           Pydantic request/response models
  errors.py            Domain exceptions and their HTTP mapping
  x/
    auth.py            OAuth 2.0 PKCE flow, token persistence, refresh
    client.py          Async X API v2 client with retries and error mapping
    ratelimit.py       Per-endpoint sliding-window budgets
    capabilities.py    Runtime entitlement probe
    errors.py          X-specific errors (rate limited, blocked, forbidden)
  llm/
    bedrock.py         Bedrock Converse wrapper with structured tool use
    prompts.py         Versioned prompts, JSON schemas, anti-fabrication rules
  agents/
    segment.py         Seed analysis and target-segment derivation
    queries.py         Query plan and query sanitization
    botdetect.py       Heuristics + LLM bot judgment + combination
    profile.py         Key-information extraction
    search.py          The discovery orchestrator
    drafts.py          Drafting, revision, approval and send
  services/
    dedup.py           The hard dedup ledger
    suppression.py     The permanent opt-out list
    pacing.py          Daily cap and randomized inter-send delay
    dispatcher.py      The in-process drip dispatcher
    conversations.py   DM conversation and event state
  api/
    health.py          Liveness, detailed health, capability probe
    auth_routes.py     OAuth routes
    discovery.py       ENDPOINT I
    outreach.py        ENDPOINT II and the approval endpoints
    compliance.py      Suppression list and the outreach ledger
alembic/               Migrations
build-docs/            The source requirements this service implements
scripts/               Development guardrail helpers
tests/                 The offline test suite
docker-compose.yml     PostgreSQL 16
```

---

## License

Provided as-is for the take-home assignment.

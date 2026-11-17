# X (Twitter) Rules, API Limits, and Compliance Guide

This document outlines the X platform rules, API access limits, Direct Message (DM) caps, and compliance requirements referenced in the **GenAI Fund – FDE Take-Home Assignment**, enriched with X's official platform and API specifications.

---

## 1. Summary of Requirements from the Assignment PDF

The assignment establishes strict guardrails around how the prototype interacts with X:

1. **Official Access & Authorization:**
   * Must connect to GenAI Fund’s own authorized X account and API keys (`OAuth 2.0` user context).
   * Do not hardcode secrets or share credentials.
   * Explicitly state required X API tier, plan, and expected operating costs.
2. **Strict Adherence to X Terms & Rate Limits:**
   * Follow X’s Terms of Service, Developer Agreement, Automation Rules, and API rate limits.
   * **No prohibited scraping** (e.g., unauthorized web scraping, headless browser automation, or bypassing rate limits/paywalls).
3. **Human-in-the-Loop & Anti-Spam:**
   * The tool must draft personalized DMs for **human review and approval** before sending—never fully automated unsolicited blasts.
   * Personalization must be grounded in **real evidence** (e.g., recent posts, bio, projects) rather than fabricated familiarity.
   * **Idempotency / Deduplication:** The system must strictly track outreach state and **never double-message** anyone.
4. **Testing Guardrails:**
   * Real DM sending during development/testing must only target **accounts you control** or **opted-in test participants**—never unsolicited outreach to strangers.
5. **Graceful Failure & Transparency:**
   * Surface API limitations, blocked actions (e.g., closed DMs), and rate-limit states clearly in the UI.

---

## 2. X Direct Message (DM) Limits & Reachability Rules

### Account-Level Daily DM Caps
Account-level limits apply across **all channels combined** (web, mobile app, and official API). Reaching these limits results in temporary lockouts (minutes to hours) or account restrictions.

| Account / Access Type | Daily DM Limit (24 Hours) | Notes |
| :--- | :--- | :--- |
| **Standard / Unverified Account** | `< 500` (Unpublished lower cap) | Heavily restricted; cold DM requests to non-followers are frequently silently dropped or filtered. |
| **Verified Account (X Premium / Business)** | `500` DMs / day | Official Help Center ceiling across all devices and API calls. (Includes replies in existing threads, not just new conversations). |
| **X API Endpoint Ceiling (Per User)** | `1,440` requests / 24 hrs (`15` / 15 min) | Technical API endpoint ceiling (`POST /2/dm_conversations/...`), **bound first by the 500/day account cap**. |
| **X API Endpoint Ceiling (Per App)** | `1,440` – `15,000` / 24 hrs | Shared across all users of the developer application depending on API tier. |

> **Best Practice for Prototype Safety:** Treat `500/day` as a hard technical ceiling, not a target. For warm, human-approved outreach, enforce a conservative application-level daily cap (e.g., **20–50 DMs/day**) with paced intervals.

### Recipient Reachability Gates
Before attempting to send a DM, a recipient must be reachable under X's privacy settings:
1. **Mutual or One-Way Follow:** The target user follows the sending account (bypasses message requests and lands directly in the primary inbox).
2. **Open DMs (`Everyone`):** The target user has enabled "Allow message requests from everyone."
3. **Verified-Only DMs:** The target user restricts message requests to **Verified accounts** (requires the sending account to have X Premium / Verified Organization status).
4. **Closed DMs:** If the recipient only accepts DMs from accounts they follow and does not follow GenAI Fund, the API will return an error (`403 Forbidden` / cannot message user). The prototype must detect and gracefully surface this as a **Blocked Action**.

---

## 3. X API v2 Rate Limits & Pricing Considerations

X discontinued its legacy Free tier for general read/search workflows and enforces strict per-window (15-minute) and 24-hour rate limits, alongside pay-per-usage or tiered monthly access.

### Relevant X API v2 Endpoints & Rate Limits

| Action / Feature | Endpoint | Method | Per-User Limit (15 min) | Per-App Limit (15 min) | Notes & Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Search Recent Posts** | `/2/tweets/search/recent` | `GET` | `300` | `450` | Searches last 7 days of public posts (useful for finding AI/robotics builders). |
| **User Search** | `/2/users/search` | `GET` | `900` | `300` | Search users by keyword/bio. |
| **User Lookup (Batch, by IDs)** | `/2/users` | `GET` | `900` | `300` | Up to 100 user IDs per request (`ids` parameter). Used for discovery hydration. Check profile bio, metrics, and metadata. |
| **User Lookup (Batch, by usernames)** | `/2/users/by` | `GET` | `900` | `300` | Up to 100 usernames per request (`usernames` parameter). Rejects the `ids` parameter. |
| **User Timeline** | `/2/users/:id/tweets` | `GET` | `900` | `10,000` | Fetch recent tweets to gather real evidence for DM personalization. |
| **Followers / Following** | `/2/users/:id/followers` | `GET` | `300` | `300` | Up to 1,000 users per page; useful for seed-account expansion. |
| **Send Direct Message** | `/2/dm_conversations/with/:participant_id/messages` | `POST` | `15` (+ `1,440`/24h) | `1,440`/24h | Requires `dm.write` OAuth 2.0 scope. Subject to 500/day account cap. |
| **Read DM Events** | `/2/dm_events` | `GET` | `15` | `n/a` | **Very tight limit** (`1` req/min average). Requires `dm.read` scope. Must cache/poll carefully to track replies. |
| **Follow / Like / Repost** | `/2/users/:id/following` (etc.) | `POST` | `50` | `n/a` | Restricted to **Enterprise** access for automated write engagement. |

### Critical Technical Bottleneck: Reading DMs (`15 / 15 min`)
* Because `GET /2/dm_events` is capped at **15 requests per 15 minutes per user**, polling aggressively for replies will immediately trigger `HTTP 429 Too Many Requests`.
* **Architectural Requirement:** Poll DM events on a conservative schedule (e.g., once every **2–5 minutes**), store conversation states locally in a database, and respect `x-rate-limit-remaining` and `x-rate-limit-reset` response headers.

### Parameter Constraints (`max_results` minimums)
X rejects a `max_results` below an endpoint's documented minimum with `HTTP 400 Invalid Request`. The client must clamp to these bounds rather than sending a smaller value:

| Endpoint | Minimum | Maximum |
| :--- | :--- | :--- |
| `/2/tweets/search/recent` | `10` | `100` |
| `/2/tweets/search/all` | `10` | `500` |
| `/2/users/:id/tweets` | `5` | `100` |
| `/2/users/search` | `1` | `1,000` |
| `/2/dm_events` | `1` | `100` |

* **Architectural Requirement:** The capability probe must request at least `10` results, otherwise a probe of the recent-search tier is indistinguishable from a genuine entitlement failure.

### Query Syntax Rules (tweet search)
X rejects a malformed tweet-search query with `HTTP 400 Invalid Request`. The query planner must normalise every generated query:

| Rule | X response | Handling |
| :--- | :--- | :--- |
| X web-search operator names | `Reference to invalid operator 'min_faves'` | Rewrite `min_faves:` → `min_likes:` and `min_retweets:` → `min_reposts:` |
| A conjunction-required operator alone (`lang:en`, `is:reply`, `has:media`) | `is/has/lang cannot be used as a standalone operator` | Always pair with at least one plain term |
| No positive clause (only negations, e.g. `-is:retweet`) | `Rules must contain a non-negation term` | Drop the query |
| Only stopwords (`a`, `the`, `is`) | `Rules must contain at least one positive, non-stopword clause` | Use a substantive term |

* **Architectural Requirement:** The capability probe query must be a single substantive term (for example `AI`) — a bare `lang:en` or `a` measures the query syntax rather than the account's tier.

### Endpoint Split: User Lookup
X separates batch user lookup into two distinct endpoints, and each accepts only its own parameter:

| Purpose | Endpoint | Required parameter | Accepts `ids`? |
| :--- | :--- | :--- | :--- |
| Look up by user IDs | `/2/users` | `ids` (comma-separated, ≤100) | yes |
| Look up by usernames | `/2/users/by` | `usernames` (comma-separated, ≤100) | **no** |
| Look up one username | `/2/users/by/username/{username}` | path parameter | n/a |

* **Architectural Requirement:** Discovery hydration must use `GET /2/users`. Posting `ids` to `GET /2/users/by` returns `The query parameter [ids] is not one of [usernames,user.fields]`.

---

## 4. X Automation Rules & Anti-Spam Policies

Under X’s **Automation Rules** and **Platform Manipulation and Spam Policy**, automated or semi-automated outreach is strictly policed based on behavioral signals:

### Prohibited Behaviors (Will Trigger Suspension or API Revocation)
* **Bulk Unsolicited DMs:** Sending unsolicited messages at scale to users who have not opted in or interacted with you.
* **Duplicative Content ("Copypasta"):** Sending identical or near-identical message templates across multiple recipients.
* **Unauthorized Scraping / Session Hijacking:** Using headless browsers (Puppeteer/Playwright), browser extensions, or unofficial private endpoints using session cookies/passwords to bypass API costs or limits.
* **Ignoring Opt-Outs:** Continuing to message a user after they decline, ignore repeatedly, or ask not to be contacted.
* **Link-Only Spam:** Sending bare URLs without meaningful, personalized context.

### Required Compliance Controls in the Prototype
1. **Mandatory Human Approval Queue:** Every drafted DM must remain in a `PENDING_APPROVAL` state until an authorized user explicitly reviews, edits (if needed), and clicks **Approve & Send**.
2. **Evidence-Backed Personalization:** Prompts generating DM drafts must inject verifiable context (e.g., specific recent posts, GitHub/project links mentioned in bio, or shared ecosystem ties in Southeast Asia) so every message is genuinely unique.
3. **Hard Deduplication Lock:** Maintain a persistent database record keyed by `recipient_x_user_id` (and handle). Before queuing or sending, verify that no prior `SENT`, `QUEUED`, or `OPTED_OUT` record exists.
4. **Paced Dispatching (Drip Queue):** Even when a user approves 20 messages at once, dispatch them sequentially with randomized human-like delays (e.g., 30–90 seconds apart) rather than bursting 15 requests in 10 seconds.
5. **Opt-Out / Suppression List:** Provide a one-click "Do Not Contact / Suppress" flag in the UI so any user who opts out is permanently blocked from future discovery or outreach.
6. **X Premium Requirement Note:** Clearly document in setup instructions that the connected GenAI Fund X account should have **X Premium (Verified)** enabled so outreach DMs are not silently filtered out by recipients' message-request settings.
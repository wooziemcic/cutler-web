"""Substack Live (RapidAPI) integration — FAST MODE.

FAST MODE contract (matches your requirements):
- Exactly 1 search call per ticker via GET /search/post.
- Strict lookback enforced; if list-stage timestamp is missing, we allow ONE best candidate through to /reader/post to obtain an authoritative timestamp (still max 1 /reader/post call).
- At most 1 post returned per ticker.
- At most 1 /reader/post call per ticker (only for the best candidate).
- Noise control: keep only if at least one substantive paragraph mentions the ticker in-context.

This module is designed to be a drop-in replacement for your existing substack_excerpts.py:
- Keeps requests dependency only.
- Keeps return schema keys: post_id, title, author, published_at, url, excerpt, body.

Environment variables (optional tuning):
- SUBSTACK_RAPIDAPI_BASE_URL (default https://substack-live.p.rapidapi.com)
- SUBSTACK_RAPIDAPI_HOST (recommended: substack-live.p.rapidapi.com)
- SUBSTACK_RAPIDAPI_KEY (or RAPIDAPI_KEY fallback)
- SUBSTACK_HTTP_TIMEOUT (default 10)
- SUBSTACK_HTTP_RETRIES (default 0)
- SUBSTACK_HTTP_BACKOFF_BASE (default 0.5)
- SUBSTACK_DEBUG ("1" to raise detailed errors)
- SUBSTACK_MIN_PARAGRAPH_CHARS (default 200)
- SUBSTACK_MIN_PARAGRAPH_WORDS (default 30)
- SUBSTACK_MAX_PARAGRAPHS_PER_ITEM (default 2)
- SUBSTACK_FINANCE_KEYWORDS (override comma-separated list)
- SUBSTACK_AMBIGUOUS_TICKERS (override comma-separated list)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_BASE_URL = os.getenv("SUBSTACK_RAPIDAPI_BASE_URL", "https://substack-live.p.rapidapi.com")
DEFAULT_TIMEOUT_S = float(os.getenv("SUBSTACK_HTTP_TIMEOUT", "10"))
DEFAULT_RETRIES = int(os.getenv("SUBSTACK_HTTP_RETRIES", "0"))
DEFAULT_BACKOFF_BASE = float(os.getenv("SUBSTACK_HTTP_BACKOFF_BASE", "0.5"))
SUBSTACK_DEBUG = os.getenv("SUBSTACK_DEBUG", "0").strip() == "1"

MIN_PARA_CHARS = int(os.getenv("SUBSTACK_MIN_PARAGRAPH_CHARS", "200"))
MIN_PARA_WORDS = int(os.getenv("SUBSTACK_MIN_PARAGRAPH_WORDS", "30"))
MAX_PARAS_PER_ITEM = int(os.getenv("SUBSTACK_MAX_PARAGRAPHS_PER_ITEM", "2"))

DEFAULT_FINANCE_KEYWORDS = [
    "earnings",
    "eps",
    "guidance",
    "valuation",
    "multiple",
    "margin",
    "revenue",
    "gross margin",
    "operating margin",
    "cash flow",
    "free cash flow",
    "fcf",
    "capex",
    "buy",
    "sell",
    "hold",
    "rating",
    "upgrade",
    "downgrade",
    "target price",
    "price target",
    "thesis",
    "position",
    "portfolio",
    "long",
    "short",
    "shares",
    "options",
    "calls",
    "puts",
    "10-q",
    "10k",
    "10-k",
    "macro",
    "fed",
    "inflation",
    "rates",
    "stock",
    "equity",
    "ticker",
    "nasdaq",
    "nyse",
    "market cap",
]

FINANCE_KEYWORDS = [
    k.strip().lower() for k in os.getenv("SUBSTACK_FINANCE_KEYWORDS", ",".join(DEFAULT_FINANCE_KEYWORDS)).split(",") if k.strip()
]

DEFAULT_AMBIGUOUS = [
    # short/common tickers that collide with normal words or acronyms
    "T", "F", "AI", "AGI", "SIRI", "PINE", "LEG", "ON", "IT", "CAR", "ALL"
]
AMBIGUOUS_TICKERS = {t.strip().upper() for t in os.getenv("SUBSTACK_AMBIGUOUS_TICKERS", ",".join(DEFAULT_AMBIGUOUS)).split(",") if t.strip()}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None

        if re.fullmatch(r"\d{10,13}", s):
            try:
                ts = float(s)
                if len(s) >= 13:
                    ts /= 1000.0
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                pass

        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    return None


def _get_key() -> str:
    key = os.getenv("SUBSTACK_RAPIDAPI_KEY") or os.getenv("RAPIDAPI_KEY") or ""
    if not key:
        raise RuntimeError("SUBSTACK_RAPIDAPI_KEY env var is not set (or RAPIDAPI_KEY fallback is missing).")
    return key


def _headers() -> Dict[str, str]:
    h = {
        "x-rapidapi-key": _get_key(),
        "accept": "application/json",
    }
    host = os.getenv("SUBSTACK_RAPIDAPI_HOST", "").strip()
    if host:
        h["x-rapidapi-host"] = host
    return h


def _request_json(path: str, params: Dict[str, Any]) -> Any:
    url = DEFAULT_BASE_URL.rstrip("/") + "/" + path.lstrip("/")
    last_err: Optional[Exception] = None

    for attempt in range(DEFAULT_RETRIES + 1):
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=DEFAULT_TIMEOUT_S)

            if resp.status_code in (429, 503, 504):
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")

            return resp.json()

        except Exception as e:
            last_err = e
            if attempt >= DEFAULT_RETRIES:
                break
            sleep_s = (DEFAULT_BACKOFF_BASE * (2 ** attempt)) + random.uniform(0, 0.2)
            time.sleep(sleep_s)

    raise RuntimeError(f"Substack API request failed for {url} params={params}: {last_err}")


def _extract_posts_from_search(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if not isinstance(payload, dict):
        return []

    # common keys
    for k in ("results", "items", "posts"):
        v = payload.get(k)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]

    # nested containers
    for k in ("data", "response", "result"):
        v = payload.get(k)
        if isinstance(v, dict):
            out = _extract_posts_from_search(v)
            if out:
                return out
        elif isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]

    # last resort: BFS for a list-of-dicts that looks like posts
    def looks_like_posts(lst: List[Any]) -> bool:
        if not lst:
            return False
        sample = next((x for x in lst if isinstance(x, dict)), None)
        if not sample:
            return False
        return any(k in sample for k in ("post_id", "postId", "id"))

    q: List[Any] = [payload]
    for _ in range(60):
        if not q:
            break
        node = q.pop(0)
        if isinstance(node, dict):
            for vv in node.values():
                if isinstance(vv, list) and looks_like_posts(vv):
                    return [x for x in vv if isinstance(x, dict)]
                if isinstance(vv, (dict, list)):
                    q.append(vv)
        elif isinstance(node, list):
            for vv in node:
                if isinstance(vv, (dict, list)):
                    q.append(vv)

    return []


def _candidate_post_id(row: Dict[str, Any]) -> str:
    for k in ("post_id", "postId", "id"):
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _candidate_title(row: Dict[str, Any]) -> str:
    for k in ("title", "headline", "name", "subject"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _candidate_url(row: Dict[str, Any]) -> str:
    for k in ("url", "canonical_url", "link", "permalink"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _candidate_snippet(row: Dict[str, Any]) -> str:
    for k in ("snippet", "summary", "description", "subtitle", "excerpt"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _candidate_published_at(row: Dict[str, Any]) -> Optional[datetime]:
    for k in ("published_at", "publishedAt", "published", "created_at", "createdAt", "date"):
        if k in row:
            dt = _parse_datetime(row.get(k))
            if dt:
                return dt
    return None


def _strip_html(text: str) -> str:
    if not text:
        return ""
    if "<" not in text:
        return re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_paragraphs(text: str) -> List[str]:
    # text is already whitespace-normalized, so split on sentence-ish boundaries is risky.
    # Prefer splitting on newlines if any survived; otherwise, chunk by " . " patterns.
    if "\n" in text:
        parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        return parts
    # fallback: rough chunking
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s{2,}", text) if p.strip()]
    return parts


def _finance_hits(text: str) -> int:
    t = (text or "").lower()
    hits = 0
    for kw in FINANCE_KEYWORDS:
        if kw and kw in t:
            hits += 1
    return hits


def _ticker_patterns(ticker: str) -> Tuple[re.Pattern, re.Pattern, re.Pattern]:
    # strong: $TSLA
    p_dollar = re.compile(rf"\${re.escape(ticker)}\b")
    # exchange: NASDAQ:TSLA / NYSE:TSLA / TSX:TSLA
    p_exch = re.compile(rf"\b(?:NASDAQ|NYSE|AMEX|TSX|TSXV|LSE|FWB|HKEX|ASX|NSE|BSE)\s*:\s*{re.escape(ticker)}\b", re.IGNORECASE)
    # bare word: TSLA
    p_word = re.compile(rf"\b{re.escape(ticker)}\b")
    return p_dollar, p_exch, p_word


def _list_stage_is_promising(ticker: str, title: str, snippet: str) -> bool:
    """Gate BEFORE /reader/post to minimize detail calls."""
    p_dollar, p_exch, p_word = _ticker_patterns(ticker)
    # Include a bit more list-stage context (author/publication fields may carry the ticker symbols).
    text = f"{title} {snippet}".strip()

    strong = bool(p_dollar.search(text) or p_exch.search(text))
    word = bool(p_word.search(text))

    if ticker in AMBIGUOUS_TICKERS:
        # For ambiguous tickers, require strong signal AND finance context.
        return strong and _finance_hits(text) > 0

    # For normal tickers, keep FAST MODE recall high:
    # - If the ticker appears as a word in the list-stage text, allow it through.
    # - Noise is still controlled later by the substantive paragraph gate on the full body.
    return strong or word


def search_posts_fast(keyword: str, *, page: int = 0) -> List[Dict[str, Any]]:
    """FAST MODE: exactly one call to /search/post."""
    payload = _request_json("/search/post", params={"query": keyword, "page": page})
    return _extract_posts_from_search(payload)


def fetch_post_details(post_id: str) -> Dict[str, Any]:
    post_id = (post_id or "").strip()
    if not post_id:
        return {}
    # per RapidAPI, the canonical param is postId
    try:
        payload = _request_json("/reader/post", params={"postId": post_id})
        return payload if isinstance(payload, dict) else {"data": payload}
    except Exception:
        if SUBSTACK_DEBUG:
            raise
        return {}


def _extract_body_from_post_details(payload: Any) -> Tuple[str, str, str, str, str]:
    """Return (body, author, title, published_raw, url)."""
    if not isinstance(payload, dict):
        return "", "", "", "", ""

    base = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    post = base.get("post") if isinstance(base.get("post"), dict) else None
    node = post or base

    body = (
        node.get("body_html")
        or node.get("bodyHtml")
        or node.get("html")
        or node.get("content_html")
        or node.get("contentHtml")
        or node.get("raw_body")
        or node.get("rawBody")
        or node.get("body")
        or node.get("text")
        or node.get("content")
        or node.get("description")
        or ""
    )

    author = (
        node.get("author")
        or node.get("author_name")
        or node.get("authorName")
        or node.get("byline")
        or ""
    )

    if isinstance(author, dict):
        author = author.get("name") or author.get("handle") or ""

    if not author:
        pub = node.get("publication") if isinstance(node.get("publication"), dict) else None
        if pub:
            author = pub.get("name") or ""
        user = node.get("user") if isinstance(node.get("user"), dict) else None
        if user and not author:
            author = user.get("name") or user.get("handle") or ""

    title = node.get("title") or node.get("headline") or node.get("subject") or ""
    published_raw = (
        node.get("published_at")
        or node.get("publishedAt")
        or node.get("published")
        or node.get("created_at")
        or node.get("createdAt")
        or node.get("date")
        or ""
    )

    url = node.get("url") or node.get("canonical_url") or node.get("link") or ""

    return str(body or ""), str(author or ""), str(title or ""), str(published_raw or ""), str(url or "")


def _select_substantive_paragraphs(ticker: str, body_clean: str) -> List[str]:
    p_dollar, p_exch, p_word = _ticker_patterns(ticker)
    paras = _split_paragraphs(body_clean)

    chosen: List[str] = []
    for p in paras:
        if len(p) < MIN_PARA_CHARS:
            continue
        if len(p.split()) < MIN_PARA_WORDS:
            continue

        has_strong = bool(p_dollar.search(p) or p_exch.search(p))
        has_word = bool(p_word.search(p))
        fin = _finance_hits(p)

        if ticker in AMBIGUOUS_TICKERS:
            ok = has_strong and fin > 0
        else:
            ok = has_strong or (has_word and fin > 0)

        if not ok:
            continue

        chosen.append(p)
        if len(chosen) >= MAX_PARAS_PER_ITEM:
            break

    return chosen


def fetch_posts_for_ticker(ticker: str, *, lookback_days: int = 1, max_posts: int = 1) -> List[Dict[str, Any]]:
    """FAST MODE: returns [] or [one_item]."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return []

    lookback_days = max(1, int(lookback_days))
    max_posts = 1  # enforced in fast mode

    cutoff = _now_utc() - timedelta(days=lookback_days)

    # Single search call (FAST MODE): one query string only.
    # Keep it tight but allow modest recall in one call.
    # IMPORTANT: still only 1 call to /search/post for the ticker.
    query = ticker
    try:
        rows = search_posts_fast(query, page=0)
    except Exception:
        if SUBSTACK_DEBUG:
            raise
        return []

    if not rows:
        return []

    # Pick the best candidate with minimal risk of extra calls:
    # - Prefer list-stage items that pass cutoff and look promising.
    # - If list-stage has no timestamp (common), allow ONE fallback: pick the best promising row
    #   and enforce cutoff after the single /reader/post call.
    best: Optional[Dict[str, Any]] = None
    best_dt: Optional[datetime] = None
    best_untimed: Optional[Dict[str, Any]] = None

    for row in rows:
        pid = _candidate_post_id(row)
        if not pid:
            continue

        title = _candidate_title(row)
        snippet = _candidate_snippet(row)
        # Some list results carry useful context in author/publication fields; include lightly for gating.
        author_hint = row.get("author") or row.get("author_name") or row.get("authorName") or ""
        pub_hint = ""
        pub = row.get("publication")
        if isinstance(pub, dict):
            pub_hint = pub.get("name") or pub.get("subdomain") or ""
        snippet_gate = f"{snippet} {author_hint} {pub_hint}".strip()

        if not _list_stage_is_promising(ticker, title, snippet_gate):
            continue

        dt = _candidate_published_at(row)
        if not dt:
            # allow ONE fallback candidate without list-stage timestamp
            if best_untimed is None:
                best_untimed = row
            continue

        if dt < cutoff:
            continue

        if best is None or (best_dt is not None and dt > best_dt) or (best_dt is None):
            best = row
            best_dt = dt

    if not best:
        best = best_untimed
        best_dt = None

    if not best:
        return []

    # Exactly one reader call
    pid = _candidate_post_id(best)
    details = fetch_post_details(pid)
    body_raw, author, title_d, published_raw, url_d = _extract_body_from_post_details(details)

    body_clean = _strip_html(body_raw)
    if not body_clean:
        return []

    # Confirm/parse published date from details if present; still enforce cutoff.
    dt2 = _parse_datetime(published_raw) or best_dt
    if not dt2 or dt2 < cutoff:
        return []

    title = title_d.strip() if title_d.strip() else _candidate_title(best)
    url = url_d.strip() if url_d.strip() else _candidate_url(best)

    chosen_paras = _select_substantive_paragraphs(ticker, body_clean)
    if not chosen_paras:
        return []

    # Keep excerpt/body compact and relevant (selected paragraphs only)
    excerpt = "\n\n".join(chosen_paras)

    return [
        {
            "post_id": pid,
            "title": title,
            "author": author,
            "published_at": published_raw or (dt2.isoformat() if dt2 else ""),
            "url": url,
            "excerpt": excerpt,
            "body": excerpt,
        }
    ]


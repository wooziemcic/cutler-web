"""
Substack Live (RapidAPI) integration.

Design goals:
- Cost control: strict lookback + per-ticker caps.
- Safety: timeouts, light retries/backoff, and graceful failures.
- Avoid expensive loops: filter at list-stage; only fetch /reader/post for candidates that pass filters.

This module intentionally exposes a small surface area so final.py can remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import os
import time
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_BASE_URL = os.getenv("SUBSTACK_RAPIDAPI_BASE_URL", "https://substack-live.p.rapidapi.com")
DEFAULT_TIMEOUT_S = float(os.getenv("SUBSTACK_HTTP_TIMEOUT", "15"))
DEFAULT_RETRIES = int(os.getenv("SUBSTACK_HTTP_RETRIES", "2"))
DEFAULT_BACKOFF_BASE = float(os.getenv("SUBSTACK_HTTP_BACKOFF_BASE", "0.6"))


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> Optional[datetime]:
    """
    Best-effort parsing for common timestamp formats:
    - ISO 8601 strings (with or without 'Z')
    - epoch seconds / ms (int/float or numeric string)
    """
    if value is None:
        return None

    # epoch
    if isinstance(value, (int, float)):
        ts = float(value)
        # heuristic: ms if very large
        if ts > 1e12:
            ts = ts / 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            return None

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None

        # numeric epoch in string
        if re.fullmatch(r"\d{10,13}", s):
            try:
                ts = float(s)
                if len(s) >= 13:
                    ts = ts / 1000.0
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                pass

        # ISO
        # Handle trailing Z
        if s.endswith("Z"):
            s2 = s[:-1] + "+00:00"
        else:
            s2 = s
        # Some APIs return "2026-01-13T10:01:02.123Z"
        try:
            return datetime.fromisoformat(s2)
        except Exception:
            pass
        # Fallback: just date
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            return None

    return None


def _get_key() -> str:
    key = os.getenv("SUBSTACK_RAPIDAPI_KEY") or os.getenv("RAPIDAPI_KEY") or ""
    if not key:
        raise RuntimeError("SUBSTACK_RAPIDAPI_KEY env var is not set (or RAPIDAPI_KEY fallback is missing).")
    return key


def _headers() -> Dict[str, str]:
    key = _get_key()
    # RapidAPI typically accepts just x-rapidapi-key; host header can be optional.
    host = os.getenv("SUBSTACK_RAPIDAPI_HOST", "").strip()
    h = {
        "x-rapidapi-key": key,
        "accept": "application/json",
    }
    if host:
        h["x-rapidapi-host"] = host
    return h


def _request_json(path: str, params: Dict[str, Any]) -> Any:
    """
    GET request with light retries/backoff.
    """
    url = DEFAULT_BASE_URL.rstrip("/") + "/" + path.lstrip("/")
    last_err: Optional[Exception] = None

    for attempt in range(DEFAULT_RETRIES + 1):
        try:
            resp = requests.get(url, headers=_headers(), params=params, timeout=DEFAULT_TIMEOUT_S)
            # If rate-limited, backoff and retry
            if resp.status_code in (429, 503, 504):
                raise RuntimeError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt >= DEFAULT_RETRIES:
                break
            # jittered exponential-ish backoff
            sleep_s = (DEFAULT_BACKOFF_BASE * (2 ** attempt)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)

    raise RuntimeError(f"Substack API request failed: {last_err}")


def _extract_posts_from_search(payload: Any) -> List[Dict[str, Any]]:
    """
    RapidAPI payloads vary. We normalize to a list of dicts representing posts/comments.
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        # Common shapes:
        #   {"data": {"results": [ ... ]}}
        #   {"data": [ ... ]}
        #   {"results": [ ... ]}
        #   {"items": [ ... ]}
        for key in ("results", "items", "posts"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

        vdata = payload.get("data")
        if isinstance(vdata, list):
            return [x for x in vdata if isinstance(x, dict)]
        if isinstance(vdata, dict):
            out = _extract_posts_from_search(vdata)
            if out:
                return out

        # sometimes nested
        if "response" in payload and isinstance(payload["response"], dict):
            return _extract_posts_from_search(payload["response"])
    return []


def _candidate_published_at(row: Dict[str, Any]) -> Optional[datetime]:
    for k in ("published_at", "publishedAt", "published", "date", "created_at", "createdAt", "post_date"):
        if k in row:
            dt = _parse_datetime(row.get(k))
            if dt:
                return dt
    return None


def _candidate_post_id(row: Dict[str, Any]) -> str:
    for k in ("post_id", "postId", "id", "postid"):
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _candidate_url(row: Dict[str, Any]) -> str:
    for k in ("url", "canonical_url", "link", "permalink"):
        v = row.get(k)
        if v and isinstance(v, str):
            return v.strip()
    return ""


def _candidate_title(row: Dict[str, Any]) -> str:
    for k in ("title", "headline", "name"):
        v = row.get(k)
        if v and isinstance(v, str):
            return v.strip()
    return ""


def _extract_body_from_post_details(payload: Any) -> Tuple[str, str, str, str]:
    """
    Returns (body, author, title, published_at_string)
    """
    if payload is None:
        return "", "", "", ""

    if isinstance(payload, dict):
        base = payload.get("data") if isinstance(payload.get("data"), dict) else payload

        # body candidates (rich HTML vs plain)
        body = (
            base.get("body")
            or base.get("text")
            or base.get("content")
            or base.get("subtitle")
            or base.get("description")
            or ""
        )

        # author
        author = (
            base.get("author")
            or base.get("author_name")
            or base.get("authorName")
            or (base.get("publication") or {}).get("name") if isinstance(base.get("publication"), dict) else ""
        )

        title = base.get("title") or base.get("headline") or ""
        published_raw = (
            base.get("published_at")
            or base.get("published")
            or base.get("date")
            or base.get("created_at")
            or ""
        )

        return str(body or ""), str(author or ""), str(title or ""), str(published_raw or "")

    return "", "", "", ""


def _strip_html(text: str) -> str:
    # very lightweight HTML stripping (keep it dependency-free)
    if "<" not in text:
        return text.strip()
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def search_posts(keyword: str, *, page: int = 0, limit: int = 20) -> List[Dict[str, Any]]:
    """Search posts for a keyword.

    The RapidAPI Substack Live API has used both `/search/post` (singular) and
    `/search/posts` (plural) over time. In production we prefer `/search/post`
    because it is the endpoint that currently returns results in the
    `{"data": {"results": [...]}}` shape.

    We keep a tiny fallback to `/search/posts` purely for resiliency.
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    # Prefer the working singular endpoint with the most common params.
    for params in (
        {"query": keyword, "page": page},
        {"query": keyword, "page": page, "limit": limit},
        {"keyword": keyword, "page": page},
        {"q": keyword, "page": page},
    ):
        try:
            payload = _request_json("/search/post", params=params)
            rows = _extract_posts_from_search(payload)
            if rows:
                return rows
        except Exception:
            continue

    # Fallback: plural endpoint (older behavior)
    for params in (
        {"query": keyword, "page": max(1, int(page) or 1), "limit": limit},
        {"keyword": keyword, "page": max(1, int(page) or 1), "limit": limit},
    ):
        try:
            payload = _request_json("/search/posts", params=params)
            rows = _extract_posts_from_search(payload)
            if rows:
                return rows
        except Exception:
            continue

    return []


_FINANCE_TERMS = [
    # High-signal finance / investing language
    "stock",
    "shares",
    "earnings",
    "eps",
    "revenue",
    "guidance",
    "valuation",
    "multiple",
    "p/e",
    "pe ",
    "ebitda",
    "margin",
    "dividend",
    "buy",
    "sell",
    "rating",
    "upgrade",
    "downgrade",
    "target price",
    "price target",
    "position",
    "portfolio",
    "10-q",
    "10k",
    "10-k",
    "sec",
    "macro",
    "ticker",
]


def _iter_paragraphs(text: str) -> List[str]:
    """Split cleaned text into paragraphs.

    We keep this dependency-free: first try splitting on newlines; if the text
    is already flattened, fall back to splitting on sentence clusters.
    """
    if not text:
        return []

    # Prefer newline paragraphing when available
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p and p.strip()]
    if len(parts) >= 2:
        return parts

    # If HTML stripping collapsed whitespace, re-split on " ." clusters.
    # This is intentionally conservative.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s{2,}", text) if p and p.strip()]
    return parts


def _paragraph_is_substantive(p: str, *, min_words: int, min_chars: int) -> bool:
    if not p:
        return False
    if len(p) < min_chars:
        return False
    words = re.findall(r"[A-Za-z0-9$:_-]+", p)
    return len(words) >= min_words


def _paragraph_mentions_ticker(p: str, ticker: str) -> bool:
    t = ticker.upper()
    # Strong patterns first
    if re.search(rf"\${re.escape(t)}\b", p, flags=re.IGNORECASE):
        return True
    if re.search(rf"\b(?:NYSE|NASDAQ|AMEX|OTC|TSX|LSE|ASX)\s*:\s*{re.escape(t)}\b", p, flags=re.IGNORECASE):
        return True
    # Weak pattern: bare word match
    return bool(re.search(rf"\b{re.escape(t)}\b", p, flags=re.IGNORECASE))


def _paragraph_has_finance_context(p: str) -> bool:
    pl = p.lower()
    return any(term in pl for term in _FINANCE_TERMS)


def _select_qualifying_paragraphs(text: str, ticker: str) -> List[str]:
    """Return only paragraphs that are (a) substantive and (b) ticker-relevant.

    Goal: avoid noise where a company is only mentioned briefly or where the
    ticker is an ambiguous token (e.g., AGI meaning "artificial general intelligence").
    We require finance context in the same paragraph.
    """
    min_words = int(os.getenv("SUBSTACK_MIN_PARAGRAPH_WORDS", "30"))
    min_chars = int(os.getenv("SUBSTACK_MIN_PARAGRAPH_CHARS", "180"))
    max_paras = int(os.getenv("SUBSTACK_MAX_PARAGRAPHS_PER_ITEM", "2"))

    paras = _iter_paragraphs(text)
    out: List[str] = []
    for p in paras:
        if not _paragraph_is_substantive(p, min_words=min_words, min_chars=min_chars):
            continue
        if not _paragraph_mentions_ticker(p, ticker):
            continue
        if not _paragraph_has_finance_context(p):
            continue
        out.append(p.strip())
        if len(out) >= max_paras:
            break
    return out


def fetch_post_details(post_id: str) -> Dict[str, Any]:
    """
    Calls /reader/post to fetch full post data.
    """
    post_id = (post_id or "").strip()
    if not post_id:
        return {}
    # similarly resilient param naming
    for params in ({"postId": post_id}, {"post_id": post_id}, {"id": post_id}):
        try:
            payload = _request_json("/reader/post", params=params)
            if payload:
                return payload if isinstance(payload, dict) else {"data": payload}
        except Exception:
            continue
    return {}


def fetch_posts_for_ticker(ticker: str, *, lookback_days: int = 1, max_posts: int = 1) -> List[Dict[str, Any]]:
    """
    High-level helper used by final.py.

    Strategy:
      1) List stage: /search/posts for keyword=ticker (page 1 only).
      2) Filter to lookback window using list-stage timestamps where possible.
      3) Dedupe by post_id.
      4) Fetch full details via /reader/post only for top N candidates.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return []

    lookback_days = max(1, int(lookback_days))
    max_posts = max(1, int(max_posts))

    cutoff = _now_utc() - timedelta(days=lookback_days)

    # Fast-mode defaults: one search call (page=0) and a small pool.
    # IMPORTANT: Substack search is materially better for tickers when using the "$TICKER" pattern.
    # This keeps recall high without adding extra API calls.
    keyword = f"${ticker}"
    candidates = search_posts(keyword, page=0, limit=max(20, max_posts * 5))

    # Optional (disabled by default): allow a second search call using the bare ticker if desired.
    # Keep this OFF to preserve the "1 list-stage call per ticker" cost guarantee.
    if not candidates and os.getenv("SUBSTACK_ALLOW_BARE_TICKER_FALLBACK", "0") == "1":
        candidates = search_posts(ticker, page=0, limit=max(20, max_posts * 5))

    if not candidates:
        return []

    # Filter + dedupe at list-stage
    seen: set[str] = set()
    filtered: List[Dict[str, Any]] = []

    for row in candidates:
        pid = _candidate_post_id(row)
        if not pid:
            continue
        if pid in seen:
            continue
        seen.add(pid)

        dt = _candidate_published_at(row)
        if dt and dt < cutoff:
            continue

        filtered.append(row)

        if len(filtered) >= max_posts * 3:
            # keep a small pool, then we will fetch details for first N
            break

    if not filtered:
        return []

    results: List[Dict[str, Any]] = []
    for row in filtered[:max_posts]:
        pid = _candidate_post_id(row)
        title = _candidate_title(row)
        url = _candidate_url(row)
        published_dt = _candidate_published_at(row)
        published_raw = row.get("published_at") or row.get("published") or row.get("date") or ""

        details = fetch_post_details(pid)
        body_raw, author, title_d, pub_d = _extract_body_from_post_details(details)

        body_clean = _strip_html(body_raw) if body_raw else ""
        if title_d and not title:
            title = title_d
        if pub_d and not published_raw:
            published_raw = pub_d

        # If details provided a better URL, prefer it
        if isinstance(details, dict):
            base = details.get("data") if isinstance(details.get("data"), dict) else details
            u2 = base.get("url") or base.get("canonical_url") or base.get("link")
            if isinstance(u2, str) and u2.strip():
                url = u2.strip()

        # Final cutoff check (if only details has timestamp)
        dt2 = _parse_datetime(published_raw) or published_dt
        if dt2 and dt2 < cutoff:
            continue

        # Only include if there is a substantive, finance-context paragraph mention.
        qualifying = _select_qualifying_paragraphs(body_clean, ticker)
        if not qualifying:
            continue

        selected_text = "\n\n".join(qualifying)
        excerpt = selected_text[:1600] if selected_text else ""

        results.append(
            {
                "post_id": pid,
                "title": title,
                "author": author,
                "published_at": published_raw or (dt2.isoformat() if dt2 else ""),
                "url": url,
                "excerpt": excerpt,
                "body": selected_text,
            }
        )

    return results

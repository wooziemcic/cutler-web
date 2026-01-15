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

# Strictness / noise controls (override via env / Streamlit Secrets)
SUBSTACK_MIN_PARAGRAPH_CHARS = int(os.getenv("SUBSTACK_MIN_PARAGRAPH_CHARS", "220"))
SUBSTACK_MIN_PARAGRAPH_WORDS = int(os.getenv("SUBSTACK_MIN_PARAGRAPH_WORDS", "35"))
SUBSTACK_MAX_PARAGRAPHS_PER_ITEM = int(os.getenv("SUBSTACK_MAX_PARAGRAPHS_PER_ITEM", "2"))
SUBSTACK_MAX_CANDIDATES_MULTIPLIER = int(os.getenv("SUBSTACK_MAX_CANDIDATES_MULTIPLIER", "8"))

# Tickers that are common words/acronyms; require strong mention patterns ($TICKER or EXCHANGE:TICKER)
_AMBIGUOUS_TICKERS_DEFAULT = {"AGI", "T", "F", "SIRI", "PINE", "LEG"}
_AMBIGUOUS_TICKERS = {
    t.strip().upper()
    for t in os.getenv("SUBSTACK_AMBIGUOUS_TICKERS", ",".join(sorted(_AMBIGUOUS_TICKERS_DEFAULT))).split(",")
    if t.strip()
}

_FINANCE_KEYWORDS = {
    "earnings", "eps", "revenue", "guidance", "margin", "margins", "valuation", "multiple", "dcf",
    "upgrade", "downgrade", "rating", "buy", "sell", "hold", "target", "price target",
    "bull", "bear", "thesis", "position", "positions", "portfolio", "allocation",
    "10-q", "10k", "10-k", "8-k", "sec", "filing", "dividend", "yield", "cash flow",
    "free cash flow", "fcf", "balance sheet", "debt", "leverage", "capex",
    "macro", "inflation", "rates", "fed", "recession", "spread", "credit",
    "shares", "stock", "stocks", "equity", "equities", "market cap", "market",
    "quarter", "q1", "q2", "q3", "q4", "invest", "investing", "trade", "trading",
    "options", "calls", "puts", "long", "short",
}


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
        for key in ("data", "results", "items", "posts"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
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


def _norm_text(s: Any) -> str:
    if not s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return s.strip()


def _build_ticker_query(ticker: str) -> str:
    """Tightened query to reduce false positives."""
    t = ticker.strip().upper()
    # Keep it simple: many providers accept quoted phrases and OR in the query string.
    # If the backend ignores OR, it still works as a broad query.
    return f"${t} OR \"{t} stock\" OR \"NASDAQ:{t}\" OR \"NYSE:{t}\" OR \"({t})\""


def _has_finance_terms(text: str) -> bool:
    s = text.lower()
    return any(k in s for k in _FINANCE_KEYWORDS)


def _mention_strength(ticker: str, text: str) -> int:
    """
    3 = $TICKER
    2 = EXCHANGE:TICKER
    1 = bare word-boundary ticker
    0 = no match
    """
    t = ticker.upper()
    if not text:
        return 0
    if f"${t}" in text:
        return 3
    if re.search(rf"\b(?:NYSE|NASDAQ|AMEX|OTC|TSX|TSXV|LSE|NYSEARCA|CBOE):\s*{re.escape(t)}\b", text, flags=re.I):
        return 2
    if re.search(rf"\b{re.escape(t)}\b", text):
        return 1
    return 0


def _split_paragraphs(body: str) -> List[str]:
    """Split into paragraph-ish chunks from stripped text."""
    if not body:
        return []
    # Prefer newline separation if available; otherwise create soft breaks.
    raw = body.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in raw.split("\n") if p.strip()]
    if len(parts) <= 1:
        # Fallback: split by sentence groups to approximate paragraphs.
        parts = re.split(r"(?<=[.!?])\s{2,}", raw)
        parts = [p.strip() for p in parts if p.strip()]
    return parts


def _substantive_paragraphs_for_ticker(ticker: str, body_clean: str) -> List[str]:
    """Return substantive, finance-relevant paragraphs that mention the ticker."""
    t = ticker.upper()
    paras = _split_paragraphs(body_clean)
    out: List[str] = []
    for p in paras:
        if len(p) < SUBSTACK_MIN_PARAGRAPH_CHARS:
            continue
        if len(p.split()) < SUBSTACK_MIN_PARAGRAPH_WORDS:
            continue
        ms = _mention_strength(t, p)
        if ms <= 0:
            continue

        finance_ok = _has_finance_terms(p)

        # For ambiguous tickers (AGI, T, F, SIRI...), require strong patterns.
        if t in _AMBIGUOUS_TICKERS and ms < 2:
            continue

        # For non-ambiguous tickers, allow bare mention only if finance terms exist.
        if ms == 1 and not finance_ok:
            continue

        # Basic anti-false-positive: AGI often appears as "Artificial General Intelligence".
        if t == "AGI" and re.search(r"artificial\s+general\s+intelligence", p, flags=re.I):
            continue

        out.append(p)
        if len(out) >= SUBSTACK_MAX_PARAGRAPHS_PER_ITEM:
            break
    return out


def search_posts(keyword: str, *, page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Primary list-stage search.
    1) Try /search/post (singular) as per RapidAPI UI.
    2) If empty, fall back to /search/top.
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    def _try(endpoint: str) -> List[Dict[str, Any]]:
        for params in (
            {"query": keyword, "page": page},
            {"query": keyword, "page": max(page - 1, 0)},  # some backends are 0-based
            {"q": keyword, "page": page},
            {"keyword": keyword, "page": page},
        ):
            try:
                payload = _request_json(endpoint, params=params)
                rows = _extract_posts_from_search(payload)
                if rows:
                    return rows
            except Exception:
                continue
        return []

    rows = _try("/search/post")
    if rows:
        return rows
    return _try("/search/top")


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


def fetch_posts_for_ticker(ticker: str, *, lookback_days: int = 1, max_posts: int = 3) -> List[Dict[str, Any]]:
    """
    High-level helper used by final.py.

    Strategy:
      1) List stage: /search/post using a tightened query (reduces false positives).
      2) Rank candidates by (recency + finance keywords + strong $TICKER/exchange:ticker patterns).
      3) Only fetch /reader/post for top candidates until max_posts are found.
      4) Keep an item only if at least one substantive, finance-relevant paragraph mentions the ticker.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return []

    lookback_days = max(1, int(lookback_days))
    max_posts = max(1, int(max_posts))

    cutoff = _now_utc() - timedelta(days=lookback_days)

    query = _build_ticker_query(ticker)
    candidates = search_posts(query, page=1, limit=max(35, max_posts * 6))
    if not candidates:
        return []

    # Filter + dedupe + score at list-stage
    seen: set[str] = set()
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for row in candidates:
        pid = _candidate_post_id(row)
        if not pid or pid in seen:
            continue
        seen.add(pid)

        title = _candidate_title(row)
        url = _candidate_url(row)
        pub_dt = _candidate_published_at(row)

        # If list-stage has a timestamp and it's older than cutoff, drop early.
        if pub_dt and pub_dt < cutoff:
            continue

        blob = f"{title} {url}"
        ms = _mention_strength(ticker, blob)
        fin = 1 if _has_finance_terms(blob) else 0

        # Heuristic score: finance terms + strong mention patterns + recency
        recency_bonus = 0.0
        if pub_dt:
            age_hours = max(0.0, (_now_utc() - pub_dt).total_seconds() / 3600.0)
            recency_bonus = max(0.0, 48.0 - age_hours)  # newest wins

        score = (ms * 10.0) + (fin * 6.0) + recency_bonus
        scored.append((score, row))

        # Keep a bounded pool to limit work
        if len(scored) >= max_posts * SUBSTACK_MAX_CANDIDATES_MULTIPLIER:
            break

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)

    results: List[Dict[str, Any]] = []
    # Fetch details only until we have max_posts that pass the paragraph gate
    for _, row in scored:
        if len(results) >= max_posts:
            break

        pid = _candidate_post_id(row)
        title = _candidate_title(row)
        url = _candidate_url(row)
        published_dt = _candidate_published_at(row)
        published_raw = row.get("published_at") or row.get("published") or row.get("date") or row.get("created_at") or ""

        # Light pre-filter: if nothing looks finance-related AND mention is weak, skip expensive details.
        blob = f"{title} {url}"
        ms = _mention_strength(ticker, blob)
        fin = _has_finance_terms(blob)
        if ms == 1 and not fin:
            continue
        if ticker in _AMBIGUOUS_TICKERS and ms < 2:
            continue

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

        dt2 = _parse_datetime(published_raw) or published_dt
        if not dt2:
            # Strict lookback: skip if we can't validate recency.
            continue
        if dt2 < cutoff:
            continue

        # Paragraph gate: only keep if we have substantive, finance-relevant ticker paragraphs.
        paras = _substantive_paragraphs_for_ticker(ticker, body_clean)
        if not paras:
            continue

        excerpt = "\n\n".join(paras)

        results.append(
            {
                "post_id": pid,
                "title": title,
                "author": author,
                "published_at": published_raw or dt2.isoformat(),
                "url": url,
                "excerpt": excerpt,
                "body": excerpt,  # keep body consistent with excerpt to reduce noise
            }
        )

    return results

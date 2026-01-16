"""
Substack Live (RapidAPI) integration.

Design goals:
- Cost control: strict lookback + per-ticker caps.
- Safety: timeouts, light retries/backoff, and graceful failures.
- Avoid expensive loops: filter at list-stage; only fetch /reader/post for candidates that pass filters.

This module intentionally exposes a small surface area so final.py can remain stable.

PATCH NOTES (speed + noise control):
- Uses tightened query variants for ticker search (e.g., "$TSLA", "TSLA stock", "NASDAQ:TSLA").
- Finance heuristic filter BEFORE /reader/post to avoid expensive detail calls for irrelevant candidates.
- Candidate ranking (recency + finance keyword hits + $TICKER/exchange pattern hits).
- Early-stop once enough qualifying posts are found.
- Optional threaded /reader/post fetching (small pool) to reduce wall-clock time.

No external deps added.
"""

from __future__ import annotations

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

# Optional concurrency for /reader/post calls
DETAIL_WORKERS = int(os.getenv("SUBSTACK_DETAIL_WORKERS", "4"))
DETAIL_WORKERS = max(1, min(DETAIL_WORKERS, 8))

# Opt-in debugging. Set SUBSTACK_DEBUG="1" in Streamlit Secrets to surface request errors.
SUBSTACK_DEBUG = os.getenv("SUBSTACK_DEBUG", "0").strip() == "1"

# Keep these conservative by default; tune via secrets if needed.
MIN_PARAGRAPH_CHARS = int(os.getenv("SUBSTACK_MIN_PARAGRAPH_CHARS", "220"))
MIN_PARAGRAPH_WORDS = int(os.getenv("SUBSTACK_MIN_PARAGRAPH_WORDS", "35"))
MAX_PARAGRAPHS_PER_ITEM = int(os.getenv("SUBSTACK_MAX_PARAGRAPHS_PER_ITEM", "2"))

# Ambiguous tickers (1-3 chars) that frequently collide with common words/abbreviations.
# Require strong patterns ($TICKER or EXCHANGE:TICKER) for these.
AMBIGUOUS_TICKERS = {
    t.strip().upper()
    for t in (os.getenv("SUBSTACK_AMBIGUOUS_TICKERS", "AGI,T,F,SIRI,PINE,LEG").split(","))
    if t.strip()
}

# Finance context keywords used at list-stage and paragraph-stage
FINANCE_KEYWORDS = [
    "stock", "shares", "equity", "earnings", "eps", "revenue", "guidance", "outlook",
    "valuation", "multiple", "p/e", "pe", "ev/ebitda", "margin", "free cash", "fcf",
    "buy", "sell", "rating", "upgrade", "downgrade", "target price", "price target",
    "position", "portfolio", "holding", "thesis", "catalyst", "10-q", "10k", "10-k",
    "sec", "filing", "quarter", "q1", "q2", "q3", "q4", "macro", "rates", "fed",
    "dividend", "buyback", "acquisition", "merger", "deal",
    "nasdaq", "nyse", "amex", "tsx", "lse",
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Best-effort parsing for ISO 8601 strings and epoch seconds/ms."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        ts = float(value)
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

        if re.fullmatch(r"\d{10,13}", s):
            try:
                ts = float(s)
                if len(s) >= 13:
                    ts = ts / 1000.0
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                pass

        s2 = (s[:-1] + "+00:00") if s.endswith("Z") else s
        try:
            dt = datetime.fromisoformat(s2)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

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
    host = os.getenv("SUBSTACK_RAPIDAPI_HOST", "").strip()
    h = {
        "x-rapidapi-key": key,
        "accept": "application/json",
    }
    if host:
        h["x-rapidapi-host"] = host
    return h


def _request_json(path: str, params: Dict[str, Any]) -> Any:
    """GET request with light retries/backoff."""
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
            sleep_s = (DEFAULT_BACKOFF_BASE * (2 ** attempt)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)

    raise RuntimeError(f"Substack API request failed for {url} params={params}: {last_err}")


def _extract_posts_from_search(payload: Any) -> List[Dict[str, Any]]:
    """Robustly pull a list of post-like dicts from varying RapidAPI payload shapes."""
    if payload is None:
        return []

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if not isinstance(payload, dict):
        return []

    for key in ("results", "items", "posts", "data"):
        v = payload.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]

    for key in ("data", "response", "result"):
        v = payload.get(key)
        if isinstance(v, dict):
            out = _extract_posts_from_search(v)
            if out:
                return out

    for key in ("results", "items", "posts", "comments"):
        v = payload.get(key)
        if isinstance(v, dict):
            out = _extract_posts_from_search(v)
            if out:
                return out

    def looks_like_posts(lst: List[Any]) -> bool:
        if not lst:
            return False
        sample = next((x for x in lst if isinstance(x, dict)), None)
        if not sample:
            return False
        return any(k in sample for k in ("post_id", "postId", "id", "postid"))

    queue: List[Any] = [payload]
    for _ in range(60):
        if not queue:
            break
        node = queue.pop(0)

        if isinstance(node, dict):
            for vv in node.values():
                if isinstance(vv, list) and looks_like_posts(vv):
                    return [x for x in vv if isinstance(x, dict)]
                if isinstance(vv, (dict, list)):
                    queue.append(vv)
        elif isinstance(node, list):
            for vv in node:
                if isinstance(vv, (dict, list)):
                    queue.append(vv)

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


def _strip_html(text: str) -> str:
    # very lightweight HTML stripping (keep it dependency-free)
    if not text:
        return ""
    if "<" not in text:
        return text.strip()
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_body_from_post_details(payload: Any) -> Tuple[str, str, str, str]:
    """Returns (body, author, title, published_at_string)."""
    if payload is None or not isinstance(payload, dict):
        return "", "", "", ""

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
        or node.get("subtitle")
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

    return str(body or ""), str(author or ""), str(title or ""), str(published_raw or "")


def _build_query_variants(ticker: str) -> List[str]:
    """Tight search queries to reduce noise while keeping recall."""
    t = ticker.upper().strip()
    if not t:
        return []
    variants = [
        f"${t}",
        f"{t} stock",
        f"NASDAQ:{t}",
        f"NYSE:{t}",
        f"TSX:{t}",
        f"LSE:{t}",
        t,
    ]
    # de-dupe while preserving order
    seen = set()
    out = []
    for q in variants:
        q2 = q.strip()
        if q2 and q2 not in seen:
            seen.add(q2)
            out.append(q2)
    return out


def _text_for_row(row: Dict[str, Any]) -> str:
    parts = []
    for k in ("title", "headline", "name", "description", "subtitle", "summary", "snippet", "excerpt"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    pub = row.get("publication")
    if isinstance(pub, dict):
        n = pub.get("name")
        if isinstance(n, str) and n.strip():
            parts.append(n.strip())
    return "\n".join(parts)


def _finance_signal_score(ticker: str, text: str) -> int:
    """Cheap list-stage scoring to decide whether to fetch /reader/post."""
    if not text:
        return 0
    t = ticker.upper()
    s = text.lower()

    score = 0

    # strong ticker patterns
    if f"${t}" in text:
        score += 4
    if re.search(rf"\b(?:NASDAQ|NYSE|TSX|LSE)\s*:\s*{re.escape(t)}\b", text, flags=re.I):
        score += 4

    # bare ticker mention
    if re.search(rf"\b{re.escape(t)}\b", text):
        score += 1

    # finance context hits
    hits = 0
    for kw in FINANCE_KEYWORDS:
        if kw in s:
            hits += 1
    score += min(hits, 5)

    # for ambiguous tickers, require strong patterns (we'll enforce later too)
    if t in AMBIGUOUS_TICKERS and score < 4:
        return 0

    return score


def _extract_substantive_paragraphs_for_ticker(*, ticker: str, body_text: str) -> List[str]:
    """Return up to MAX_PARAGRAPHS_PER_ITEM substantive paragraphs mentioning ticker + finance context."""
    if not body_text:
        return []

    t = ticker.upper()

    # Split on blank lines first; fallback to sentence chunking if needed
    paras = [p.strip() for p in re.split(r"\n\s*\n+", body_text) if p.strip()]
    if len(paras) <= 1:
        # fallback: chunk into ~longer segments
        paras = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z$])", body_text) if p.strip()]

    out: List[str] = []
    for p in paras:
        if len(p) < MIN_PARAGRAPH_CHARS:
            continue
        if len(p.split()) < MIN_PARAGRAPH_WORDS:
            continue

        # Require ticker mention
        strong = (f"${t}" in p) or bool(re.search(rf"\b(?:NASDAQ|NYSE|TSX|LSE)\s*:\s*{re.escape(t)}\b", p, flags=re.I))
        bare = bool(re.search(rf"\b{re.escape(t)}\b", p))

        if t in AMBIGUOUS_TICKERS and not strong:
            continue
        if not (strong or bare):
            continue

        # Require finance context IN THE PARAGRAPH
        pl = p.lower()
        if not any(kw in pl for kw in FINANCE_KEYWORDS):
            # allow strong $TICKER/exchange pattern to pass even if finance words missing
            if not strong:
                continue

        out.append(p)
        if len(out) >= MAX_PARAGRAPHS_PER_ITEM:
            break

    return out


def search_posts(keyword: str, *, page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
    """Calls /search/post (RapidAPI); falls back to /search/top if needed."""
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    endpoints = ["/search/post", "/search/top"]
    for endpoint in endpoints:
        for params in (
            {"query": keyword, "page": page, "limit": limit},
            {"keyword": keyword, "page": page, "limit": limit},
            {"q": keyword, "page": page, "limit": limit},
            {"query": keyword, "page": page},
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


def fetch_post_details(post_id: str) -> Dict[str, Any]:
    """Calls /reader/post to fetch full post data."""
    post_id = (post_id or "").strip()
    if not post_id:
        return {}
    for params in ({"postId": post_id}, {"post_id": post_id}, {"id": post_id}):
        try:
            payload = _request_json("/reader/post", params=params)
            if payload:
                return payload if isinstance(payload, dict) else {"data": payload}
        except Exception:
            continue
    return {}


def fetch_posts_for_ticker(ticker: str, *, lookback_days: int = 1, max_posts: int = 2) -> List[Dict[str, Any]]:
    """High-level helper used by final.py.

    Strategy (optimized):
      1) List stage: search with tightened query variants.
      2) Early filter + score at list-stage (recency + finance signal) and keep a small pool.
      3) Fetch /reader/post only for top-scoring candidates.
      4) Extract substantive ticker paragraphs; drop drive-by mentions.

    NOTE: If no parsable timestamp exists at list-stage AND details-stage, the item is skipped.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return []

    lookback_days = max(1, int(lookback_days))
    max_posts = max(1, int(max_posts))

    cutoff = _now_utc() - timedelta(days=lookback_days)

    # ---- list-stage: gather candidates from a few query variants ----
    seen: set[str] = set()
    pool: List[Dict[str, Any]] = []

    # Hard cap to prevent runaway scanning
    max_pool = max(12, max_posts * 6)

    for q in _build_query_variants(ticker):
        rows = search_posts(q, page=1, limit=25)
        if not rows:
            continue

        for row in rows:
            pid = _candidate_post_id(row)
            if not pid or pid in seen:
                continue
            seen.add(pid)

            dt = _candidate_published_at(row)
            if dt is None:
                # strict cost rule: if we can't date it cheaply, only keep if very strong finance signal
                if _finance_signal_score(ticker, _text_for_row(row)) < 6:
                    continue
            else:
                if dt < cutoff:
                    continue

            txt = _text_for_row(row)
            sig = _finance_signal_score(ticker, txt)
            if sig <= 0:
                continue

            row["__sig"] = sig
            row["__dt"] = dt.isoformat() if dt else ""
            pool.append(row)

            if len(pool) >= max_pool:
                break

        if len(pool) >= max_pool:
            break

        # early stop if we already have enough strong candidates
        strong = sum(1 for r in pool if int(r.get("__sig") or 0) >= 7)
        if strong >= max_posts:
            break

    if not pool:
        return []

    # ---- sort pool: recency + signal ----
    def sort_key(r: Dict[str, Any]):
        dt = _candidate_published_at(r) or _parse_datetime(r.get("__dt"))
        ts = dt.timestamp() if dt else 0.0
        sig = int(r.get("__sig") or 0)
        return (-ts, -sig)

    pool.sort(key=sort_key)

    # Limit detail fetches aggressively
    top = pool[: max(4, max_posts * 2)]

    # ---- details-stage: optionally parallel fetch ----
    results: List[Dict[str, Any]] = []

    def build_item(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pid = _candidate_post_id(row)
        if not pid:
            return None

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

        # prefer url from details if present
        if isinstance(details, dict):
            base = details.get("data") if isinstance(details.get("data"), dict) else details
            u2 = base.get("url") or base.get("canonical_url") or base.get("link")
            if isinstance(u2, str) and u2.strip():
                url = u2.strip()

        dt2 = _parse_datetime(published_raw) or published_dt
        if dt2 is None:
            return None  # strict
        if dt2 < cutoff:
            return None

        # Paragraph gating for substantive mention
        paras = _extract_substantive_paragraphs_for_ticker(ticker=ticker, body_text=body_clean)
        if not paras:
            return None

        excerpt = "\n\n".join(paras)
        return {
            "post_id": pid,
            "title": title,
            "author": author,
            "published_at": dt2.isoformat(),
            "url": url,
            "excerpt": excerpt,
            "body": body_clean,
        }

    # small parallelism for details
    if DETAIL_WORKERS > 1 and len(top) > 1:
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as ex:
                futs = [ex.submit(build_item, r) for r in top]
                for fut in as_completed(futs):
                    it = fut.result()
                    if it:
                        results.append(it)
                        if len(results) >= max_posts:
                            break
        except Exception:
            # fall back to sequential on any thread/runtime weirdness
            results = []

    if not results:
        for row in top:
            it = build_item(row)
            if it:
                results.append(it)
                if len(results) >= max_posts:
                    break

    # Final sort by published time desc
    results.sort(key=lambda x: x.get("published_at") or "", reverse=True)
    return results

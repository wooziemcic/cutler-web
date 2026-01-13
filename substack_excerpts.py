"""
Substack Live (RapidAPI) integration.

Design goals:
- Cost control: strict lookback + per-ticker caps.
- Safety: timeouts, light retries/backoff, and graceful failures.
- Avoid expensive loops: filter at list-stage; only fetch /reader/post for candidates that pass filters.

This module intentionally exposes a small surface area so final.py can remain stable.
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

# Opt-in debugging. Set SUBSTACK_DEBUG="1" in Streamlit Secrets to surface request errors.
SUBSTACK_DEBUG = os.getenv("SUBSTACK_DEBUG", "0").strip() == "1"

# -----------------------------
# Search + cost-control helpers
# -----------------------------

# Finance heuristic terms for list-stage filtering (cheap, big cost win).
# If the list-stage text has *no* finance signal, we avoid /reader/post calls.
FINANCE_TERMS: List[str] = [
    "earnings", "guidance", "eps", "revenue", "sales", "margin", "gross margin", "operating margin",
    "valuation", "multiple", "p/e", "pe", "ev/ebitda", "ebitda", "cash flow", "free cash flow", "fcf",
    "balance sheet", "debt", "leverage", "liquidity", "downgrade", "upgrade", "rating", "buy", "sell",
    "hold", "overweight", "underweight", "price target", "pt", "DCF", "discounted cash flow",
    "10-q", "10k", "10-k", "8-k", "form 4", "sec filing", "quarter", "q1", "q2", "q3", "q4",
    "portfolio", "position", "allocation", "risk", "macro", "inflation", "rates", "fed", "yield",
    "dividend", "buyback", "repurchase", "accretion", "dilution", "guidance", "outlook",
    "bear", "bull", "thesis", "catalyst",
]

_FINANCE_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in FINANCE_TERMS) + r")\b", re.IGNORECASE)

def build_search_queries(ticker: str) -> List[str]:
    """Tighten list-stage search queries to reduce false positives."""
    t = (ticker or "").strip().upper()
    if not t:
        return []
    # Order matters: most precise first.
    return [
        f"${t}",
        f"NASDAQ:{t}",
        f"{t} stock",
        f"{t} shares",
        f"{t} earnings",
        f"{t} ({t})",
    ]

def _candidate_snippet(row: Dict[str, Any]) -> str:
    for k in ("snippet", "summary", "description", "subtitle", "subTitle", "dek", "excerpt", "text"):
        v = row.get(k)
        if v and isinstance(v, str):
            return v.strip()
    # Sometimes nested
    for k in ("publication", "author", "user"):
        v = row.get(k)
        if isinstance(v, dict):
            for kk in ("name", "handle", "subdomain", "title"):
                vv = v.get(kk)
                if vv and isinstance(vv, str):
                    return vv.strip()
    return ""

def _list_stage_text(row: Dict[str, Any], *, title: str, url: str) -> str:
    parts: List[str] = []
    if title:
        parts.append(title)
    snip = _candidate_snippet(row)
    if snip:
        parts.append(snip)
    if url:
        parts.append(url)
    # common author/publication fields
    for k in ("author", "author_name", "authorName", "publication", "pub", "publisher"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, dict):
            for kk in ("name", "handle", "subdomain"):
                vv = v.get(kk)
                if isinstance(vv, str) and vv.strip():
                    parts.append(vv.strip())
    return " ".join(parts)

def _finance_hits(text: str) -> int:
    if not text:
        return 0
    return len(_FINANCE_RE.findall(text))

def _ticker_signal_score(text: str, ticker: str) -> int:
    if not text or not ticker:
        return 0
    t = ticker.upper()
    txt_u = text.upper()
    score = 0
    if f"${t}" in text:
        score += 3
    if f"NASDAQ:{t}" in txt_u:
        score += 2
    # keep this small to avoid generic ticker matches dominating
    if t in txt_u:
        score += 1
    return score




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

        if s.endswith("Z"):
            s2 = s[:-1] + "+00:00"
        else:
            s2 = s

        try:
            return datetime.fromisoformat(s2)
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
    """
    GET request with light retries/backoff.
    """
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
    """
    RapidAPI payloads vary and are often nested like:
      { "data": { "items": [...] } } or { "data": { "posts": [...] } }
      { "response": { "data": { ... } } }

    This extractor:
    - checks common top-level keys,
    - recursively descends into dict containers (data/response/result),
    - and as a last resort searches a few levels deep for the first list of dicts
      that looks like "posts" (contains post_id/postId/id).
    """
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


def _extract_body_from_post_details(payload: Any) -> Tuple[str, str, str, str]:
    """
    Returns (body, author, title, published_at_string)
    """
    if payload is None:
        return "", "", "", ""

    if not isinstance(payload, dict):
        return "", "", "", ""

    base = payload.get("data") if isinstance(payload.get("data"), dict) else payload

    # Substack/RapidAPI often nests the post under "post"
    post = base.get("post") if isinstance(base.get("post"), dict) else None
    node = post or base

    # body candidates (HTML first, then text-ish)
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

    # author candidates
    author = (
        node.get("author")
        or node.get("author_name")
        or node.get("authorName")
        or node.get("byline")
        or ""
    )

    # Sometimes author is a nested object
    if isinstance(author, dict):
        author = author.get("name") or author.get("handle") or ""

    # If still blank, try publication / user objects
    if not author:
        pub = node.get("publication") if isinstance(node.get("publication"), dict) else None
        if pub:
            author = pub.get("name") or ""
        user = node.get("user") if isinstance(node.get("user"), dict) else None
        if user and not author:
            author = user.get("name") or user.get("handle") or ""

    title = (
        node.get("title")
        or node.get("headline")
        or node.get("subject")
        or ""
    )

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


def _strip_html(text: str) -> str:
    if "<" not in text:
        return text.strip()
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _try_search_endpoint(endpoint_path: str, keyword: str, page: int, limit: int) -> List[Dict[str, Any]]:
    """
    Internal helper to try a search endpoint using the correct parameter names.
    Per RapidAPI UI, /search/post requires:
      - query (required)
      - page (optional, often 0-based)
    """
    last_err: Optional[Exception] = None

    # The RapidAPI console shows: query (required), page optional (example uses page=0)
    # We keep a small set of variants for resilience, but keep "query" first.
    for params in (
        {"query": keyword, "page": page},          # primary (matches UI)
        {"query": keyword, "page": max(page - 1, 0)},  # in case provider uses 0-based paging
        {"q": keyword, "page": page},              # fallback
        {"keyword": keyword, "page": page},        # fallback
    ):
        try:
            payload = _request_json(endpoint_path, params=params)
            rows = _extract_posts_from_search(payload)
            if rows:
                return rows
        except Exception as e:
            last_err = e
            continue

    if SUBSTACK_DEBUG and last_err:
        raise RuntimeError(f"{endpoint_path} failed for keyword='{keyword}': {last_err}")

    return []


def search_posts(keyword: str, *, page: int = 1, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Primary list-stage search.
    1) Try /search/post (singular)  <-- correct per RapidAPI
    2) If empty, fall back to /search/top
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    # IMPORTANT: endpoint is singular: /search/post
    rows = _try_search_endpoint("/search/post", keyword, page, limit)
    if rows:
        return rows

    # Fallback list-stage endpoint
    return _try_search_endpoint("/search/top", keyword, page, limit)


def fetch_post_details(post_id: str) -> Dict[str, Any]:
    """
    Calls /reader/post to fetch full post data.
    RapidAPI UI shows the parameter is: postId (required).
    """
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


def fetch_posts_for_ticker(ticker: str, *, lookback_days: int = 1, max_posts: int = 3) -> List[Dict[str, Any]]:
    """
    High-level helper used by final.py.

    Strategy:
      1) List stage: /search/post for query=ticker (page 1 only).
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

    candidates: List[Dict[str, Any]] = []

    # Tightened multi-query search (reduces false positives)
    for q in build_search_queries(ticker):
        rows = search_posts(q, page=1, limit=max(25, max_posts * 5))
        if rows:
            candidates.extend(rows)
        # light early stop if we already have plenty of candidates
        if len(candidates) >= max(50, max_posts * 12):
            break
    if not candidates:
        return []

    seen: set[str] = set()
    scored: List[Tuple[Dict[str, Any], Optional[datetime], int, int]] = []

    for row in candidates:
        pid = _candidate_post_id(row)
        if not pid:
            continue
        if pid in seen:
            continue
        seen.add(pid)

        dt = _candidate_published_at(row)
        # Strict lookback: if we can't determine recency at list-stage, skip to control cost.
        if not dt:
            continue
        if dt < cutoff:
            continue

        title = _candidate_title(row)
        url = _candidate_url(row)
        text = _list_stage_text(row, title=title, url=url)

        fh = _finance_hits(text)
        ts = _ticker_signal_score(text, ticker)

        # Finance heuristic BEFORE /reader/post (big cost win)
        # Only keep candidates that show finance signal or an explicit $TICKER tag.
        if fh <= 0 and ts < 3:
            continue

        scored.append((row, dt, fh, ts))

        # Light early stop: we only need a bit more than max_posts to rank
        if len(scored) >= max(60, max_posts * 20):
            break

    if not scored:
        return []

    # Hard cap + sorting: (a) recency (b) finance keywords (c) $TICKER presence
    def _sort_key(x: Tuple[Dict[str, Any], Optional[datetime], int, int]) -> Tuple[float, int, int]:
        _row, _dt, _fh, _ts = x
        ts_epoch = (_dt.timestamp() if _dt else 0.0)
        return (ts_epoch, _fh, _ts)

    scored.sort(key=_sort_key, reverse=True)
    filtered: List[Dict[str, Any]] = [row for (row, _dt, _fh, _ts) in scored[: max_posts * 3]]

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

        if isinstance(details, dict):
            base = details.get("data") if isinstance(details.get("data"), dict) else details
            u2 = base.get("url") or base.get("canonical_url") or base.get("link")
            if isinstance(u2, str) and u2.strip():
                url = u2.strip()

        dt2 = _parse_datetime(published_raw) or published_dt
        if dt2 and dt2 < cutoff:
            continue

        excerpt = body_clean[:1600] if body_clean else ""

        results.append(
            {
                "post_id": pid,
                "title": title,
                "author": author,
                "published_at": published_raw or (dt2.isoformat() if dt2 else ""),
                "url": url,
                "excerpt": excerpt,
                "body": body_clean,
            }
        )

    return results

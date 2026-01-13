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
        if s.endswith("Z"):
            s2 = s[:-1] + "+00:00"
        else:
            s2 = s

        try:
            return datetime.fromisoformat(s2)
        except Exception:
            pass

        # Fallback: date only
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

            # transient
            if resp.status_code in (429, 503, 504):
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")

            # surface errors (helps debugging)
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

    # 1) direct list keys
    for key in ("results", "items", "posts", "data"):
        v = payload.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]

    # 2) nested containers (common in RapidAPI)
    for key in ("data", "response", "result"):
        v = payload.get(key)
        if isinstance(v, dict):
            out = _extract_posts_from_search(v)
            if out:
                return out

    # 3) nested list keys inside a dict payload
    for key in ("results", "items", "posts", "comments"):
        v = payload.get(key)
        if isinstance(v, dict):
            out = _extract_posts_from_search(v)
            if out:
                return out

    # 4) last-resort: look for any list-of-dicts that includes a post identifier
    def looks_like_posts(lst: List[Any]) -> bool:
        if not lst:
            return False
        sample = next((x for x in lst if isinstance(x, dict)), None)
        if not sample:
            return False
        return any(k in sample for k in ("post_id", "postId", "id", "postid"))

    # scan a few levels deep without being expensive
    queue: List[Any] = [payload]
    for _ in range(60):  # hard cap to prevent runaway
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

    if isinstance(payload, dict):
        base = payload.get("data") if isinstance(payload.get("data"), dict) else payload

        body = (
            base.get("body")
            or base.get("text")
            or base.get("content")
            or base.get("subtitle")
            or base.get("description")
            or ""
        )

        author = (
            base.get("author")
            or base.get("author_name")
            or base.get("authorName")
            or ((base.get("publication") or {}).get("name") if isinstance(base.get("publication"), dict) else "")
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
    if "<" not in text:
        return text.strip()
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _try_search_endpoint(endpoint_path: str, keyword: str, page: int, limit: int) -> List[Dict[str, Any]]:
    """
    Internal helper to try a search endpoint with a few param name variants.
    """
    last_err: Optional[Exception] = None

    for params in (
        {"keyword": keyword, "page": page, "limit": limit},
        {"query": keyword, "page": page, "limit": limit},
        {"q": keyword, "page": page, "limit": limit},
        {"keyword": keyword, "page": page},
        {"query": keyword, "page": page},
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
    1) Try /search/posts
    2) If empty, fall back to /search/top (some providers expose results here instead)
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    rows = _try_search_endpoint("/search/posts", keyword, page, limit)
    if rows:
        return rows

    # Fallback (still list-stage; doesn't fetch full bodies)
    return _try_search_endpoint("/search/top", keyword, page, limit)


def fetch_post_details(post_id: str) -> Dict[str, Any]:
    """
    Calls /reader/post to fetch full post data.
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


def fetch_posts_for_ticker(ticker: str, *, lookback_days: int = 2, max_posts: int = 3) -> List[Dict[str, Any]]:
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

    candidates = search_posts(ticker, page=1, limit=max(25, max_posts * 5))
    if not candidates:
        return []

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

        # Prefer details URL if present
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

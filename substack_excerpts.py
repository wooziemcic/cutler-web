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


class SubstackHTTPError(RuntimeError):
    """HTTP error with status code attached (used for targeted fallbacks)."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = int(status_code)


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
            if 400 <= int(resp.status_code) < 600:
                # Preserve status code so callers can do a *single* targeted fallback (e.g., $TICKER -> TICKER on 500).
                raise SubstackHTTPError(int(resp.status_code), f"HTTP {resp.status_code}")
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt >= DEFAULT_RETRIES:
                break
            # jittered exponential-ish backoff
            sleep_s = (DEFAULT_BACKOFF_BASE * (2 ** attempt)) + random.uniform(0, 0.25)
            time.sleep(sleep_s)

    raise RuntimeError(f"Substack API request failed: {last_err}")


def _get_company_context(ticker: str) -> Tuple[str, List[str]]:
    """Best-effort company name/aliases from tickers.py (if available).

    This stays optional so the module remains portable.
    """
    t = (ticker or "").strip().upper()
    if not t:
        return "", []
    try:
        from tickers import tickers as _T  # type: ignore

        row = (_T or {}).get(t) if isinstance(_T, dict) else None
        if isinstance(row, dict):
            name = str(row.get("name") or row.get("company") or row.get("long_name") or "").strip()
            aliases = row.get("aliases") or row.get("alias") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            aliases = [str(a).strip() for a in (aliases or []) if str(a).strip()]
            return name, aliases
    except Exception:
        pass
    return "", []


def _is_ambiguous_ticker(ticker: str) -> bool:
    """Tickers that are short/word-like produce noisy matches on Substack."""
    t = (ticker or "").strip().upper()
    if not t:
        return True
    if len(t) <= 2:
        return True
    # Known common collisions in your universe (can expand later without changing callers)
    if t in {"AI", "AGI", "F", "T", "IT", "ON", "OR", "CAT", "C", "A"}:
        return True
    return False


def _normalize_text(s: str) -> str:
    s = (s or "").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _split_paragraphs(text: str) -> List[str]:
    """Split into paragraphs with a conservative heuristic."""
    t = _normalize_text(text)
    if not t:
        return []
    # Primary: blank-line paragraphs
    parts = [p.strip() for p in re.split(r"\n\s*\n", t) if p and p.strip()]
    if len(parts) >= 2:
        return parts
    # Fallback: sentence-ish chunks (rare when body was single-line)
    parts = [p.strip() for p in re.split(r"(?<=[\.!\?])\s{2,}", t) if p and p.strip()]
    return parts

# Sentence splitter used as a fallback when Substack bodies arrive as single-line or mega-paragraph text.
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\(])')


def _split_sentences(text: str) -> List[str]:
    """Split text into sentence-like units (best-effort).

    We keep this lightweight and dependency-free. It is intentionally conservative:
    it won't be perfect, but it dramatically improves signal when a 'paragraph'
    contains multiple tickers (e.g., roundup posts).
    """
    t = (text or '').strip()
    if not t:
        return []
    t = re.sub(r'\s+', ' ', t).strip()
    parts = _SENT_SPLIT.split(t)
    return [p.strip() for p in parts if p and p.strip()]


def extract_ticker_paragraphs(
    *,
    body_text: str,
    ticker: str,
    company_name: str = "",
    aliases: Optional[List[str]] = None,
    min_chars: int = 80,
) -> List[str]:
    """Return ONLY paragraphs that credibly mention the ticker (and/or company context).

    - For ambiguous tickers, require strong finance-oriented patterns ($TICKER, NYSE:TICKER, etc.),
      OR company-name/alias co-occurrence.
    - For non-ambiguous tickers, accept word-boundary ticker mentions, but still prefer strong patterns.
    """
    t = (ticker or "").strip().upper()
    if not t:
        return []

    body = _normalize_text(body_text)
    if not body:
        return []

    name = (company_name or "").strip()
    alias_list = [a.strip() for a in (aliases or []) if a and a.strip()]
    if not name and not alias_list:
        n2, a2 = _get_company_context(t)
        name = name or n2
        alias_list = alias_list or a2

    # Patterns
    strong_ticker = re.compile(
        rf"(?:\${re.escape(t)}\b|\b(?:NYSE|NASDAQ|AMEX)\s*:\s*{re.escape(t)}\b|\b{re.escape(t)}\s*\([A-Z]{2,6}:\s*{re.escape(t)}\)\b)",
        re.IGNORECASE,
    )
    weak_ticker = re.compile(rf"\b{re.escape(t)}\b")

    name_pat = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE) if name else None
    alias_pats = [re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE) for a in alias_list if len(a) >= 3]

    # Common false-positive suppressors (cheap heuristics)
    suppress_phrases: List[re.Pattern] = []
    # Example: AGI frequently equals "Artificial General Intelligence"
    if t == "AGI":
        suppress_phrases.append(re.compile(r"\bartificial\s+general\s+intelligence\b", re.IGNORECASE))
    # Example: CTO often equals "Chief Technology Officer"
    if t == "CTO":
        suppress_phrases.append(re.compile(r"\bchief\s+technology\s+officer\b", re.IGNORECASE))
    # Example: AI often equals "Artificial Intelligence"
    if t == "AI":
        suppress_phrases.append(re.compile(r"\bartificial\s+intelligence\b", re.IGNORECASE))

    paras = _split_paragraphs(body)
    kept: List[str] = []
    ambiguous = _is_ambiguous_ticker(t)

    for p in paras:
        if len(p) < int(min_chars):
            continue

        # If it's clearly the generic meaning (and not finance-anchored), drop it.
        suppressed = False
        for sp in suppress_phrases:
            if sp.search(p) and (not strong_ticker.search(p)):
                suppressed = True
                break
        if suppressed:
            continue

        has_strong = bool(strong_ticker.search(p))
        has_weak = bool(weak_ticker.search(p))
        has_name = bool(name_pat.search(p)) if name_pat else False
        has_alias = any(ap.search(p) for ap in alias_pats) if alias_pats else False

        if ambiguous:
            # Tighten: ambiguous tickers should not pass on company name alone.
            ok = has_strong or (has_weak and (has_name or has_alias))
        else:
            ok = has_strong or has_weak or has_name or has_alias

        if not ok:
            continue

        # If a paragraph is huge, keep only the sentences that mention the ticker
        # (plus 1 sentence of context on each side).
        if len(p) > 900:
            sents = _split_sentences(p)
            keep_idx: List[int] = []
            for i, s in enumerate(sents):
                s_has_strong = bool(strong_ticker.search(s))
                s_has_weak = bool(weak_ticker.search(s))
                s_has_name = bool(name_pat.search(s)) if name_pat else False
                s_has_alias = any(ap.search(s) for ap in alias_pats) if alias_pats else False

                if ambiguous:
                    s_ok = s_has_strong or (s_has_weak and (s_has_name or s_has_alias))
                else:
                    s_ok = s_has_strong or s_has_weak

                if s_ok:
                    keep_idx.append(i)

            if not keep_idx:
                continue

            idx_set = set()
            for i in keep_idx:
                idx_set.add(i)
                if i - 1 >= 0:
                    idx_set.add(i - 1)
                if i + 1 < len(sents):
                    idx_set.add(i + 1)

            clipped = " ".join(sents[i] for i in sorted(idx_set)).strip()
            if len(clipped) >= int(min_chars):
                kept.append(clipped)
        else:
            kept.append(p)

    return kept


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
        # 1) {"data": {"results": [ ... ]}}
        # 2) {"results": [ ... ]}
        # 3) {"data": [ ... ]}

        for key in ("results", "items", "posts"):
            v = payload.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

        v = payload.get("data")
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if isinstance(v, dict):
            # recurse into nested data payload
            return _extract_posts_from_search(v)
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

    def _find_first_text(obj: Any, keys: Tuple[str, ...]) -> str:
        """Depth-first search for the first non-empty string value for any of the given keys."""
        if obj is None:
            return ""
        if isinstance(obj, dict):
            # direct hit
            for k in keys:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            # recurse
            for v in obj.values():
                out = _find_first_text(v, keys)
                if out:
                    return out
        elif isinstance(obj, list):
            for v in obj:
                out = _find_first_text(v, keys)
                if out:
                    return out
        return ""

    if isinstance(payload, dict):
        # Some endpoints return {"data": {...}}, others return {...}. Keep both.
        base = payload.get("data") if isinstance(payload.get("data"), (dict, list)) else payload

        # body candidates (prefer HTML). Many /reader/post responses nest this under "post" or similar.
        body = _find_first_text(
            base,
            (
                "body_html",
                "bodyHtml",
                "body",
                "body_text",
                "bodyText",
                "text",
                "content",
                "content_html",
                "contentHtml",
                "description",
                "subtitle",
                "summary",
            ),
        )

        # author candidates (often nested in publishedBylines[0].name)
        author = _find_first_text(
            base,
            (
                "author",
                "author_name",
                "authorName",
                "name",
                "handle",
            ),
        )

        # title
        title = _find_first_text(base, ("title", "headline", "name"))

        # published timestamp
        published_raw = _find_first_text(
            base,
            (
                "published_at",
                "publishedAt",
                "published",
                "date",
                "created_at",
                "createdAt",
                "post_date",
            ),
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
    """Search Substack posts.

    IMPORTANT: Substack Live uses the **singular** endpoint `/search/post`.
    The plural `/search/posts` is not available for this API and will 404.

    We keep this as a list-stage call and avoid variants/fallbacks to preserve
    the "1 search call per ticker" cost rule.
    """
    keyword = (keyword or "").strip()
    if not keyword:
        return []

    # Normalize page: API accepts 0-based pages; callers may pass 1.
    try:
        p = int(page)
    except Exception:
        p = 0
    if p >= 1:
        p = p - 1

    # Use the param name that the API actually expects: `query`.
    params = {"query": keyword, "page": p}
    payload = _request_json("/search/post", params=params)
    return _extract_posts_from_search(payload)


def fetch_post_details(post_id: str) -> Dict[str, Any]:
    """
    Calls /reader/post to fetch full post data.
    """
    post_id = (post_id or "").strip()
    if not post_id:
        return {}
    # similarly resilient param naming; some endpoints are picky about numeric IDs
    post_id_int: Optional[int] = None
    try:
        post_id_int = int(str(post_id))
    except Exception:
        post_id_int = None

    param_variants = [
        {"postId": post_id_int if post_id_int is not None else post_id},
        {"post_id": post_id_int if post_id_int is not None else post_id},
        {"id": post_id_int if post_id_int is not None else post_id},
    ]

    for params in param_variants:
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

    # Substack search is far more consistent for tickers when using the cashtag form ($TICKER).
    # Cost rule: 1 search call per ticker ideally; allow EXACTLY ONE fallback call when the $TICKER
    # query triggers an upstream 500 for that symbol.
    keyword = f"${ticker}"
    try:
        candidates = search_posts(keyword, page=1, limit=max(25, max_posts * 5))
    except SubstackHTTPError as e:
        if int(getattr(e, "status_code", 0)) == 500:
            candidates = search_posts(ticker, page=1, limit=max(25, max_posts * 5))
        else:
            raise
    if not candidates:
        return []

    # Filter + dedupe at list-stage
    seen: set[str] = set()
    filtered: List[Dict[str, Any]] = []

    company_name, aliases = _get_company_context(ticker)
    ambiguous = _is_ambiguous_ticker(ticker)
    strong_list_pat = re.compile(
        rf"(?:\${re.escape(ticker)}\b|\b(?:NYSE|NASDAQ|AMEX)\s*:\s*{re.escape(ticker)}\b)",
        re.IGNORECASE,
    )
    name_pat = re.compile(rf"\b{re.escape(company_name)}\b", re.IGNORECASE) if company_name else None
    alias_pats = [re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE) for a in (aliases or []) if len(str(a)) >= 3]

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

        # For ambiguous tickers, require a list-stage anchor in title/description to avoid
        # burning /reader/post calls on obvious acronym collisions.
        if ambiguous:
            title = _candidate_title(row)
            desc = ""
            for k in ("description", "summary", "subtitle", "excerpt", "text"):
                v = row.get(k)
                if isinstance(v, str) and v.strip():
                    desc = v.strip()
                    break
            blob = f"{title}\n{desc}".strip()
            has_anchor = bool(strong_list_pat.search(blob))
            if not has_anchor and name_pat and name_pat.search(blob):
                has_anchor = True
            if not has_anchor and alias_pats and any(ap.search(blob) for ap in alias_pats):
                has_anchor = True
            if not has_anchor:
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

        # If /reader/post yielded an empty body (common), fall back to list-stage description
        # WITHOUT making extra calls.
        if not body_clean:
            list_desc = ""
            for k in ("description", "summary", "subtitle", "excerpt", "text"):
                v = row.get(k)
                if isinstance(v, str) and v.strip():
                    list_desc = v.strip()
                    break
            body_clean = _strip_html(list_desc) if list_desc else ""

        # Paragraph-only extraction (quality gate). If no qualifying paragraphs remain, drop the post.
        hit_paras = extract_ticker_paragraphs(
            body_text=body_clean,
            ticker=ticker,
            company_name=company_name,
            aliases=aliases,
        )
        if not hit_paras:
            continue

        # Provide a short excerpt for UI expanders (first few hit paragraphs only).
        excerpt = "\n\n".join(hit_paras[:3]).strip()
        if len(excerpt) > 1800:
            excerpt = excerpt[:1800].rstrip() + " …"

        results.append(
            {
                "post_id": pid,
                "title": title,
                "author": author,
                "published_at": published_raw or (dt2.isoformat() if dt2 else ""),
                "url": url,
                "excerpt": excerpt,
                "body": body_clean,
                "hit_paragraphs": hit_paras,
            }
        )

    return results

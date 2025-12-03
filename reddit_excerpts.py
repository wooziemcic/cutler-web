"""
reddit_excerpts.py

Lightweight Reddit snapshot module for the Cutler platform.

- Uses the reddit34 RapidAPI (socialminer / reddit34) for per-subreddit tops.
- Tries Reddit's own /search.json endpoint to build an Extras bucket that
  matches the browser search for `$TICKER` with Relevance + Past week.
- If Reddit blocks /search.json (403), falls back to reddit34 getSearchPosts.

This module is used both as:
  - a CLI tester:  python reddit_excerpts.py AMZN
  - a library:     imported by final.py (draw_reddit_pulse_section)
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import requests
from requests import HTTPError

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAPIDAPI_HOST = "reddit34.p.rapidapi.com"
TOP_POSTS_ENDPOINT = f"https://{RAPIDAPI_HOST}/getTopPostsBySubreddit"
SEARCH_POSTS_ENDPOINT = f"https://{RAPIDAPI_HOST}/getSearchPosts"

# Core finance subs we care about
FINANCE_SUBREDDITS: List[str] = [
    "stocks",
    "investing",
    "wallstreetbets",
    "StockMarket",
    "SecurityAnalysis",
    "finance",
    "dividends",
    "ValueInvesting",
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class RedditPost:
    subreddit: str
    title: str
    permalink: str
    score: int
    num_comments: int
    selftext: str
    body: str
    created_utc: Optional[float] = None
    raw: Optional[dict] = None  # original JSON for debugging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str:
    """
    Fetch RapidAPI key from environment.

    Supports multiple variable names so you can reuse existing keys from .env.
    """
    key = (
        os.environ.get("RAPIDAPI_KEY")
        or os.environ.get("REDDIT34_API_KEY")
        or os.environ.get("REDDIT34_KEY")
        or os.environ.get("SA_RAPIDAPI_KEY")
        or os.environ.get("REDDAPI_KEY")
    )
    if not key:
        raise RuntimeError(
            "No RapidAPI key found. Please set RAPIDAPI_KEY "
            "(or REDDIT34_API_KEY / REDDIT34_KEY / SA_RAPIDAPI_KEY)."
        )
    return key


def _make_headers() -> Dict[str, str]:
    key = _get_api_key()
    return {
        "x-rapidapi-key": key,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }


def _http_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Small wrapper around requests.get with basic error handling.
    """
    headers = _make_headers()
    logger.debug("[reddit34] GET %s params=%s", url, params)
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    logger.debug("[reddit34] Status %s", resp.status_code)
    resp.raise_for_status()
    data = resp.json()
    return data


def _extract_wrapped_posts(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    reddit34 responses tend to look like:
      {"success": true, "data": {"cursor": "...", "posts": [ { "kind": "t3",
                                                                "data": {...}}, ...]}}

    But we keep this helper defensive to survive minor shape changes.
    Returns a list of "post wrapper" dicts (usually with keys kind + data).
    """
    if not isinstance(payload, dict):
        return []

    data = payload.get("data")
    if isinstance(data, dict) and "posts" in data:
        posts = data.get("posts") or []
        if isinstance(posts, list):
            return posts

    # Fallbacks (just in case)
    if "posts" in payload and isinstance(payload["posts"], list):
        return payload["posts"]

    if isinstance(data, list):
        return data

    return []


def _to_reddit_post(wrapper: Dict[str, Any]) -> RedditPost:
    """
    Convert a reddit34 or reddit.com "post wrapper" into our RedditPost dataclass.

    Works with:
      - {"kind": "t3", "data": {...}}
      - {"subreddit": "...", "title": "...", ...} (we treat this as data directly)
    """
    if not isinstance(wrapper, dict):
        wrapper = {}

    data = wrapper.get("data") or wrapper

    subreddit = data.get("subreddit") or ""
    title = data.get("title") or ""
    permalink = data.get("permalink") or ""
    score = int(data.get("score") or 0)
    num_comments = int(data.get("num_comments") or 0)

    selftext = data.get("selftext") or ""
    body = data.get("body") or ""  # some APIs use "body" instead of "selftext"
    created_utc = data.get("created_utc")

    return RedditPost(
        subreddit=subreddit,
        title=title,
        permalink=permalink,
        score=score,
        num_comments=num_comments,
        selftext=selftext,
        body=body,
        created_utc=created_utc,
        raw=data,
    )


def _matches_ticker(text: str, ticker: str) -> bool:
    """
    Heuristic for finance-sub filtering:

      - Match '$TICKER' anywhere, case-insensitive
      - Match 'TICKER' as a standalone token (after stripping punctuation)
      - Do NOT match 'TICKER' as a pure substring inside other words/URLs.

    This keeps the finance-sub view focused. Extras may rely more directly on
    search, but this is still a good filter to avoid junk.
    """
    if not text:
        return False

    t = ticker.upper()
    text_up = text.upper()

    # Strong signal: explicit stock-style '$AMZN'
    if f"${t}" in text_up:
        return True

    # Token-level check for bare 'AMZN'
    tokens = [
        tok.strip(" .,!?:;()[]{}\"'")
        for tok in text_up.split()
    ]
    return t in tokens


# ---------------------------------------------------------------------------
# reddit34 calls for finance subs
# ---------------------------------------------------------------------------


def _fetch_top_posts_by_subreddit(
    subreddit: str,
    time_window: str = "week",
) -> List[RedditPost]:
    """
    Use reddit34 getTopPostsBySubreddit endpoint.
    time_window can be: hour, day, week, month, year, all
    """
    params = {
        "subreddit": subreddit,
        "time": time_window or "week",
    }
    payload = _http_get(TOP_POSTS_ENDPOINT, params)
    wrappers = _extract_wrapped_posts(payload)
    posts = [_to_reddit_post(w) for w in wrappers]
    logger.debug(
        "[reddit34] /r/%s top %s -> %d posts",
        subreddit,
        time_window,
        len(posts),
    )
    return posts


# ---------------------------------------------------------------------------
# Extras: primary via reddit.com search.json, fallback via reddit34
# ---------------------------------------------------------------------------


def _search_reddit_web(
    query: str,
    *,
    time_window: str = "week",
    limit: int = 5,
) -> List[RedditPost]:
    """
    Use Reddit's public /search.json endpoint to mimic browser search:

      q=$TICKER, sort=relevance, t=week, type=link, limit=5

    This gives you the same kind of results as the UI screenshot for `$AMZN`
    with "Relevance" + "Past week".
    """
    url = "https://www.reddit.com/search.json"

    params = {
        "q": query,
        "sort": "relevance",
        "t": time_window,   # hour / day / week / month / year / all
        "type": "link",
        "limit": limit,
    }

    headers = {
        # Reddit requires a User-Agent for API/JSON calls
        "User-Agent": "CutlerRedditPulse/1.0 (by u/cutler_internal_tool)",
        "Accept": "application/json",
    }

    logger.debug("[reddit_web] GET %s params=%s", url, params)
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    logger.debug("[reddit_web] Status %s", resp.status_code)
    resp.raise_for_status()
    data = resp.json()

    children = data.get("data", {}).get("children", []) or []
    posts = [_to_reddit_post(child) for child in children]

    logger.info(
        "[reddit_web] search '%s' (t=%s, limit=%d) -> %d posts",
        query,
        time_window,
        limit,
        len(posts),
    )
    return posts


def _search_cross_subreddits_rapidapi(
    query: str,
    *,
    time_window: str = "week",
    max_results: int = 5,
) -> List[RedditPost]:
    """
    Fallback for Extras: use reddit34 getSearchPosts when reddit.com search
    is blocked or not available.

    reddit34 requires:
      - time only used with sort='top'
    So we use:
      query="$TICKER", sort="top", time="week"
    """
    params = {
        "query": query,
        "sort": "top",                   # required when using 'time'
        "time": time_window or "week",   # hour / day / week / month / year / all
    }
    payload = _http_get(SEARCH_POSTS_ENDPOINT, params)

    if isinstance(payload, dict) and not payload.get("success", True):
        logger.warning(
            "[reddit34] fallback search unsuccessful for query=%r time=%r; data=%r",
            query,
            time_window,
            payload.get("data"),
        )
        return []

    wrappers = _extract_wrapped_posts(payload)
    posts = [_to_reddit_post(w) for w in wrappers]
    posts = posts[:max_results]

    logger.info(
        "[reddit34] fallback search '%s' (%s) -> %d posts (capped to %d)",
        query,
        time_window,
        len(wrappers),
        len(posts),
    )
    return posts


def _build_extras_bucket(
    ticker: str,
    current_by_subreddit: Optional[Dict[str, List[RedditPost]]] = None,
    max_total: int = 5,
    time_window: str = "week",
) -> List[RedditPost]:
    """
    Build a cross-subreddit 'extras' bucket for a ticker.

    Preferred:
      - Reddit web search `/search.json`:
          q=$TICKER, sort=relevance, t=week, type=link, limit=...
    Fallback (if 403 or other HTTP error):
      - reddit34 getSearchPosts:
          query="$TICKER", sort="top", time=week

    Both paths:
      - De-duplicate vs existing finance-sub posts via permalink
      - Keep order from the underlying source, cap to max_total
    """
    query = f"${ticker.upper()}"

    # 1) Try Reddit web search (ideal parity with browser UI)
    search_posts: List[RedditPost] = []
    try:
        search_posts = _search_reddit_web(
            query=query,
            time_window=time_window,
            limit=max_total * 2,  # grab extra to allow dedupe
        )
    except HTTPError as http_exc:
        status = getattr(http_exc.response, "status_code", None)
        logger.warning(
            "[reddit_extras] reddit web search HTTPError (%s) for %s: %s",
            status,
            query,
            http_exc,
        )
    except Exception as exc:
        logger.warning(
            "[reddit_extras] reddit web search error for %s: %s",
            query,
            exc,
        )

    # 2) Fallback via reddit34 if web search failed or returned nothing
    if not search_posts:
        logger.info(
            "[reddit_extras] falling back to reddit34 search for %s",
            query,
        )
        search_posts = _search_cross_subreddits_rapidapi(
            query=query,
            time_window=time_window,
            max_results=max_total * 2,
        )

    # 3) Build de-duplication set
    existing_permalinks = set()
    if current_by_subreddit:
        for posts in current_by_subreddit.values():
            for p in posts:
                if p.permalink:
                    existing_permalinks.add(p.permalink)

    # 4) Take posts in original order (web or fallback), skipping duplicates
    extras: List[RedditPost] = []
    for p in search_posts:
        if len(extras) >= max_total:
            break
        if p.permalink and p.permalink in existing_permalinks:
            continue
        extras.append(p)

    logger.info(
        "[reddit_extras] extras for %s -> %d posts",
        ticker,
        len(extras),
    )
    return extras


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_posts_for_ticker(
    ticker: str,
    *,
    subreddits: Optional[List[str]] = None,
    max_per_sub: int = 5,
    time_window: str = "week",
) -> Dict[str, List[RedditPost]]:
    """
    Main entry point used by final.py.

    For a given ticker:
      - Fetch top posts from each finance subreddit (weekly) via reddit34.
      - Filter posts that mention the ticker (title/body) using _matches_ticker.
      - Cap to max_per_sub per subreddit.
      - Build an "__extras__" bucket primarily from Reddit web search; if that
        fails or is blocked, fall back to reddit34 getSearchPosts.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return {}

    subs = subreddits or FINANCE_SUBREDDITS
    result: Dict[str, List[RedditPost]] = {}

    logger.info(
        "[reddit_pulse] Fetching posts for ticker %s across %d finance subs",
        ticker,
        len(subs),
    )

    # Finance subs via reddit34
    for sub in subs:
        try:
            raw_posts = _fetch_top_posts_by_subreddit(sub, time_window=time_window)
        except Exception as exc:
            logger.warning("Error fetching /r/%s: %s", sub, exc)
            continue

        matched: List[RedditPost] = []
        for p in raw_posts:
            text = f"{p.title}\n{p.selftext}\n{p.body}"
            if _matches_ticker(text, ticker):
                matched.append(p)
                if len(matched) >= max_per_sub:
                    break

        if matched:
            result[sub] = matched
            logger.info(
                "[reddit_pulse] /r/%s -> %d posts mentioning %s",
                sub,
                len(matched),
                ticker,
            )

    # Extras via reddit web search + fallback
    try:
        extras = _build_extras_bucket(
            ticker=ticker,
            current_by_subreddit=result,
            max_total=5,
            time_window=time_window,
        )
        if extras:
            result["__extras__"] = extras
    except Exception as exc:
        logger.warning("Error building extras bucket for %s: %s", ticker, exc)

    # Return only subs with at least one post
    cleaned = {k: v for k, v in result.items() if v}
    logger.info(
        "[reddit_pulse] DONE %s -> %d buckets (including extras if present)",
        ticker,
        len(cleaned),
    )
    return cleaned


# ---------------------------------------------------------------------------
# CLI tester
# ---------------------------------------------------------------------------


def _cli_main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("Usage: python reddit_excerpts.py TICKER", file=sys.stderr)
        return 1

    ticker = argv[1].upper()
    print(f"Testing Reddit fetch for ticker: {ticker}")

    try:
        posts_by_sub = fetch_posts_for_ticker(ticker)
    except Exception as exc:
        print(f"[ERROR] Failed to fetch posts: {exc}", file=sys.stderr)
        return 2

    if not posts_by_sub:
        print("[MAIN] No posts matched this ticker.")
        return 0

    for sub, posts in posts_by_sub.items():
        print("=" * 80)
        if sub == "__extras__":
            print(f"EXTRAS – {len(posts)} posts")
        else:
            print(f"r/{sub} – {len(posts)} posts")
        print("=" * 80)

        for p in posts:
            print(f"{sub} – Score: {p.score} | Comments: {p.num_comments}")
            print(p.title)
            body = p.selftext or p.body or ""
            if body:
                snippet = body[:400].replace("\n", " ")
                if len(body) > 400:
                    snippet += " [...]"
                print(snippet)
            print("-" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main(sys.argv))

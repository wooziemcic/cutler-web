"""
sa_analysis_api.py

Use RapidAPI's Seeking Alpha endpoints to fetch ANALYSIS articles
for a given symbol, then optionally build an AI digest with OpenAI.

Depends on:
  - SA_RAPIDAPI_KEY in your .env
  - OPENAI_API_KEY in your .env
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import requests
from dotenv import load_dotenv
import openai

load_dotenv()

# -------------------------------------------------------------------
#  Config  (UPDATED: lazy key checks to avoid import-time crash)
# -------------------------------------------------------------------

def _get_rapidapi_key() -> str:
    key = os.getenv("SA_RAPIDAPI_KEY")
    if not key:
        raise RuntimeError(
            "SA_RAPIDAPI_KEY is not set. Please add it to your .env or environment."
        )
    return key

def _get_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please add it to your .env or environment."
        )
    return key

# Set OpenAI key if present; don't crash app on import if missing.
try:
    openai.api_key = _get_openai_key()
except RuntimeError:
    openai.api_key = None

BASE_HOST = "seeking-alpha.p.rapidapi.com"
BASE_URL = f"https://{BASE_HOST}"

LIST_PATH = "/analysis/v2/list"
DETAILS_PATH = "/analysis/v2/get-details"
AUTHOR_DETAILS_PATH = "/authors/get-details"

def _headers() -> dict:
    """Build RapidAPI headers only when needed."""
    return {
        "x-rapidapi-key": _get_rapidapi_key(),
        "x-rapidapi-host": BASE_HOST,
    }

log = logging.getLogger("sa_analysis_api")

# Simple in-module cache for author lookups (optional).
_AUTHOR_CACHE: Dict[str, Dict[str, Any]] = {}


# -------------------------------------------------------------------
#  Data model
# -------------------------------------------------------------------

@dataclass
class AnalysisArticle:
    id: str
    symbol: str
    title: str
    published: str
    url: str
    primary_tickers: List[str] = field(default_factory=list)
    body_html: Optional[str] = None  # filled after get-details

    # NEW: author fields (best-effort)
    author_name: Optional[str] = None
    author_slug: Optional[str] = None


# -------------------------------------------------------------------
#  Low-level helpers
# -------------------------------------------------------------------

def _request(path: str, params: dict) -> dict:
    url = BASE_URL + path
    resp = requests.get(url, headers=_headers(), params=params, timeout=25)
    if resp.status_code != 200:
        raise RuntimeError(f"RapidAPI error {resp.status_code}: {resp.text[:500]}")
    try:
        return resp.json()
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from {url}: {e}")


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """
    Try a sequence of keys on a dict-like object. Returns first non-empty.
    """
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            return v
    return default


# -------------------------------------------------------------------
#  Public API – list & details
# -------------------------------------------------------------------

def fetch_analysis_list(symbol: str, size: int = 10) -> List[AnalysisArticle]:
    """
    Get list of recent analysis articles for a single ticker symbol.

    symbol: 'TSLA', 'AAPL', etc.
    size:   how many articles to pull from page 1 (max 40 allowed by API)
    """
    params = {
        "id": symbol.lower(),  # symbol slug
        "size": str(size),
        "number": "1",         # first page
    }

    payload = _request(LIST_PATH, params)

    data = payload.get("data")
    if not isinstance(data, list):
        log.debug("Unexpected analysis list payload: %r", payload)
        return []

    articles: List[AnalysisArticle] = []
    for item in data:
        try:
            # SeekingAlpha RapidAPI shapes:
            # item: { id, attributes:{title,publishOn,authorName,authorSlug...}, links:{self}, relationships:{...} }
            art_id = str(item.get("id", "")).strip()
            if not art_id:
                continue

            attrs = item.get("attributes", {}) if isinstance(item.get("attributes"), dict) else {}

            title = _safe_get(attrs, "title", default="") or ""
            published = _safe_get(attrs, "publishOn", "publishedAt", "published_at", default="") or ""

            link_self = _safe_get(item.get("links", {}), "self", default="") or ""
            url = f"https://seekingalpha.com{link_self}" if link_self else ""

            # NEW: author (best-effort)
            author_name = _safe_get(attrs, "authorName", "author", "author_name", default=None)
            author_slug = _safe_get(attrs, "authorSlug", "author_slug", default=None)

            # relationships -> sentiments -> primaryTickers ids (left as-is; you can extend later)
            primary: List[str] = []
            rel = item.get("relationships", {}) if isinstance(item.get("relationships"), dict) else {}
            sentiments = rel.get("sentiments", {}) if isinstance(rel.get("sentiments"), dict) else {}
            sdata = sentiments.get("data") or []
            # Keeping existing placeholder behavior; do not break anything.
            # If you later want primary tickers, you can enrich via included[].
            _ = sdata

            articles.append(
                AnalysisArticle(
                    id=art_id,
                    symbol=symbol.upper(),
                    title=title,
                    published=published,
                    url=url,
                    primary_tickers=primary,
                    author_name=author_name,
                    author_slug=author_slug,
                )
            )
        except Exception as e:
            log.warning("Failed to parse analysis item: %r (%s)", item, e)
            continue

    return articles


def fetch_analysis_details(article_id: str) -> dict:
    """
    Fetch full Seeking Alpha article details (HTML body, title, summary, images).
    Returns dict or {} on failure.

    Output keys:
      - title
      - summary_html
      - body_html
      - images
      - url
      - author_name (best-effort)
      - author_slug (best-effort)
    """
    params = {"id": str(article_id)}

    try:
        payload = _request(DETAILS_PATH, params)
        if not isinstance(payload, dict):
            return {}

        data = payload.get("data") or {}
        if not isinstance(data, dict):
            return {}

        attributes = data.get("attributes") or {}
        if not isinstance(attributes, dict):
            attributes = {}

        title = _safe_get(attributes, "title", default="") or ""
        summary_html = _safe_get(attributes, "summary", "summary_html", default="") or ""
        body_html = _safe_get(attributes, "content", "body_html", default="") or ""
        images = attributes.get("images", [])
        if not isinstance(images, list):
            images = []

        # NEW: author (details sometimes include it; if not present, return None)
        author_name = _safe_get(attributes, "authorName", "author", "author_name", default=None)
        author_slug = _safe_get(attributes, "authorSlug", "author_slug", default=None)

        # Prefer canonical SA article URL if present in links; else fallback
        links = data.get("links") or {}
        link_self = _safe_get(links, "self", default="") or ""
        url = f"https://seekingalpha.com{link_self}" if link_self else f"https://seekingalpha.com/article/{article_id}"

        return {
            "title": title,
            "summary_html": summary_html,
            "body_html": body_html,
            "images": images,
            "url": url,
            "author_name": author_name,
            "author_slug": author_slug,
        }

    except Exception as e:
        log.warning("fetch_analysis_details ERROR for id=%s: %s", article_id, e)
        return {}


def fetch_article_details(article_id: str) -> Optional[str]:
    """
    Fetch full HTML body of a specific analysis article.
    Returns HTML string or None on failure.
    """
    details = fetch_analysis_details(article_id)
    body = details.get("body_html") if isinstance(details, dict) else None
    if not body:
        return None
    return body


def fetch_author_details(author_slug: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Fetch author details via RapidAPI endpoint:
      /authors/get-details?slug=...

    Returns dict (may include name/bio/etc) or {} on failure.
    """
    slug = (author_slug or "").strip()
    if not slug:
        return {}

    if use_cache and slug in _AUTHOR_CACHE:
        return _AUTHOR_CACHE[slug]

    try:
        payload = _request(AUTHOR_DETAILS_PATH, {"slug": slug})
        if not isinstance(payload, dict):
            return {}

        # Keep raw payload; callers can pick what they want.
        result = payload

        if use_cache:
            _AUTHOR_CACHE[slug] = result

        return result
    except Exception as e:
        log.warning("fetch_author_details ERROR slug=%s: %s", slug, e)
        return {}


# -------------------------------------------------------------------
#  AI summarisation
# -------------------------------------------------------------------

def build_sa_analysis_digest(
    symbol: str,
    articles: List[AnalysisArticle],
    model: str = "gpt-4o-mini",
    max_articles: int = 4,
) -> str:
    """
    Use OpenAI to build a concise digest of the most recent analysis pieces.

    Returns a markdown text string.
    """
    if not articles:
        return f"No recent Seeking Alpha analysis articles found for {symbol}."

    # sort by published (descending) just in case
    arts_sorted = sorted(articles, key=lambda a: a.published, reverse=True)
    selected = arts_sorted[:max_articles]

    # Ensure each selected article has body_html filled in
    for art in selected:
        if art.body_html is None:
            try:
                art.body_html = fetch_article_details(art.id)
            except Exception as e:
                log.warning("Failed to fetch body for %s: %s", art.id, e)
                art.body_html = ""

        # If details has author fields, backfill onto article (best-effort)
        if (art.author_name is None) or (art.author_slug is None):
            details = fetch_analysis_details(art.id)
            if details:
                art.author_name = art.author_name or details.get("author_name")
                art.author_slug = art.author_slug or details.get("author_slug")

    # Build prompt context
    context_chunks = []
    for art in selected:
        body = art.body_html or ""
        # keep prompt manageable – truncate
        if len(body) > 4000:
            body = body[:4000] + " [...]"

        author_line = ""
        if art.author_name:
            author_line = f"Author: {art.author_name}\n"

        context_chunks.append(
            f"### Article\n"
            f"Title: {art.title}\n"
            f"{author_line}"
            f"Published: {art.published}\n"
            f"URL: {art.url}\n"
            f"Body (HTML):\n{body}\n"
        )

    prompt = "\n\n".join(context_chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an equity analyst writing a concise research brief for "
                "portfolio managers. You will be given several Seeking Alpha "
                "analysis articles about one stock. Extract the key points."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Stock symbol: {symbol}\n\n"
                f"Below are recent analysis articles (titles, dates, URLs and HTML bodies). "
                f"Synthesize them into:\n"
                f"1) 4–7 bullet points covering: thesis, valuation, earnings, balance sheet, "
                f"key catalysts, and major risks.\n"
                f"2) One short concluding line with an overall stance: Bullish, Neutral, "
                f"or Bearish, plus 1–2 key reasons.\n\n"
                f"Articles:\n{prompt}"
            ),
        },
    ]

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.35,
            max_tokens=700,
        )
        digest = resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error("OpenAI error while building SA analysis digest: %s", e)
        digest = (
            "Error while generating AI digest. "
            "Please check OpenAI logs or try again later."
        )

    return digest


def analyse_symbol_with_digest(
    symbol: str,
    list_size: int = 10,
    model: str = "gpt-4o-mini",
) -> Tuple[List[AnalysisArticle], str]:
    """
    Convenience wrapper used by Streamlit:
      - fetch analysis list
      - fetch bodies for a handful of most recent
      - build AI digest
    Returns (articles, digest_text)
    """
    articles = fetch_analysis_list(symbol, size=list_size)

    # Pre-fill body_html for a few of the top ones to avoid repeat calls later
    for art in articles[:4]:
        try:
            details = fetch_analysis_details(art.id)
            art.body_html = details.get("body_html") if details else ""
            # Backfill author if details had it
            if details:
                art.author_name = art.author_name or details.get("author_name")
                art.author_slug = art.author_slug or details.get("author_slug")
        except Exception as e:
            log.warning("Failed to fetch article %s: %s", art.id, e)
            art.body_html = ""

    digest = build_sa_analysis_digest(symbol, articles, model=model)
    return articles, digest


# -------------------------------------------------------------------
#  Simple CLI test
# -------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "TSLA"
    print(f"Fetching analysis list + digest for {sym}...")
    arts, digest_text = analyse_symbol_with_digest(sym)

    print(f"Got {len(arts)} articles")
    for a in arts:
        author = f" | author={a.author_name}" if a.author_name else ""
        print(f"- {a.published} | {a.title} | id={a.id} | {a.url}{author}")

    print("\n=== AI DIGEST ===\n")
    print(digest_text)

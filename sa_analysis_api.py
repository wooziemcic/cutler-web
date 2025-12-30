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
from typing import List, Optional, Tuple

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
AUTHORS_DETAILS_PATH = "/authors/get-details"

def _headers() -> dict:
    """Build RapidAPI headers only when needed."""
    return {
        "x-rapidapi-key": _get_rapidapi_key(),
        "x-rapidapi-host": BASE_HOST,
    }

log = logging.getLogger("sa_analysis_api")


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

    # Optional author fields (may be blank depending on API payload / plan)
    author_name: str = ""
    author_slug: str = ""

    # Filled after get-details
    body_html: Optional[str] = None


# -------------------------------------------------------------------
#  Payload parsing helpers
# -------------------------------------------------------------------

def _author_map_from_included(payload: dict) -> dict:
    """Return mapping: author_id -> {name, slug}. Defensive to missing shapes."""
    out: dict = {}
    included = payload.get("included")
    if not isinstance(included, list):
        return out
    for it in included:
        if not isinstance(it, dict):
            continue
        if it.get("type") not in ("author", "authors"):
            continue
        aid = it.get("id")
        attrs = it.get("attributes") or {}
        if aid:
            out[str(aid)] = {
                "name": (attrs.get("name") or attrs.get("displayName") or "").strip(),
                "slug": (attrs.get("slug") or "").strip(),
            }
    return out


def _extract_author_from_item(item: dict, author_map: dict) -> tuple[str, str]:
    """Return (author_name, author_slug) for a list row, if present."""
    rel = item.get("relationships") or {}
    auth = rel.get("author") or {}
    data = auth.get("data") or {}
    aid = data.get("id")
    if aid and str(aid) in author_map:
        m = author_map[str(aid)]
        return m.get("name", "") or "", m.get("slug", "") or ""
    return "", ""


def _extract_author_from_details(payload: dict) -> tuple[str, str]:
    """Return (author_name, author_slug) from a get-details payload, if present."""
    author_map = _author_map_from_included(payload)
    data = payload.get("data")
    main = None
    if isinstance(data, list) and data:
        main = data[0]
    elif isinstance(data, dict):
        main = data
    if not isinstance(main, dict):
        return "", ""
    return _extract_author_from_item(main, author_map)


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

    author_map = _author_map_from_included(payload)

    data = payload.get("data")
    if not isinstance(data, list):
        log.debug("Unexpected analysis list payload: %r", payload)
        return []

    articles: List[AnalysisArticle] = []
    for item in data:
        try:
            art_id = str(item["id"])
            attrs = item.get("attributes", {})
            title = attrs.get("title") or ""
            published = attrs.get("publishOn") or ""
            link_self = item.get("links", {}).get("self") or ""
            url = f"https://seekingalpha.com{link_self}" if link_self else ""

            # relationships -> sentiments -> primaryTickers ids
            primary = []
            rel = item.get("relationships", {})
            sentiments = rel.get("sentiments", {})
            sdata = sentiments.get("data") or []
            for s in sdata:
                if s.get("type") == "sentiment":
                    # inside each sentiment, there is another data-> primaryTickers,
                    # but for our purposes we just collect their ids if present
                    # (you can extend this later if you need mapping to sym)
                    pass

            articles.append(
                AnalysisArticle(
                    id=art_id,
                    symbol=symbol.upper(),
                    title=title,
                    published=published,
                    url=url,
                    primary_tickers=primary,
                    author_name=_extract_author_from_item(item, author_map)[0],
                    author_slug=_extract_author_from_item(item, author_map)[1],
                )
            )
        except Exception as e:
            log.warning("Failed to parse analysis item: %r (%s)", item, e)
            continue

    return articles


def fetch_article_details(article_id: str) -> Optional[str]:
    """Fetch full HTML body of a specific analysis article.

    RapidAPI shapes vary:
      - payload["data"] may be a dict or a list with a single dict
      - body may appear under 'content', 'bodyHtml', 'body_html', 'body', or 'html'
    Returns HTML (or sometimes plain text) string, or None if missing.
    """
    params = {"id": str(article_id)}
    payload = _request(DETAILS_PATH, params)

    data = payload.get("data")
    main = None
    if isinstance(data, list) and data:
        main = data[0]
    elif isinstance(data, dict):
        main = data

    if not isinstance(main, dict):
        log.debug("No usable data in get-details payload: %r", payload)
        return None

    attrs = main.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    body = (
        attrs.get("content")
        or attrs.get("bodyHtml")
        or attrs.get("body_html")
        or attrs.get("body")
        or attrs.get("html")
        or attrs.get("text")
        or ""
    )

    body = str(body or "").strip()
    return body or None


def fetch_author_details(author_slug: str) -> dict:
    """Optional: fetch author details by slug.

    Endpoint: /authors/get-details?slug=...
    Returns dict (possibly empty). Caller must be defensive.
    """
    if not author_slug:
        return {}
    try:
        payload = _request(AUTHORS_DETAILS_PATH, {"slug": str(author_slug)})
        # Some variants return {"data":{...}}; others {"data":[...]}
        return payload or {}
    except Exception:
        return {}


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

    # Build prompt context
    context_chunks = []
    for art in selected:
        body = art.body_html or ""
        # keep prompt manageable – truncate
        if len(body) > 4000:
            body = body[:4000] + " [...]"

        context_chunks.append(
            f"### Article\n"
            f"Title: {art.title}\n"
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
            art.body_html = fetch_article_details(art.id)
        except Exception as e:
            log.warning("Failed to fetch article %s: %s", art.id, e)
            art.body_html = ""

    digest = build_sa_analysis_digest(symbol, articles, model=model)
    return articles, digest


def fetch_analysis_details(article_id: str) -> dict:
    """
    Fetch full Seeking Alpha article details (HTML body, title, summary, images).
    Returns dict or {} on failure.
    """
    url = "https://seeking-alpha.p.rapidapi.com/analysis/v2/get-details"
    headers = _headers()
    params = {"id": str(article_id)}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, dict):
            return {}

        # Extract content
        attributes = data.get("data", {}).get("attributes", {})

        return {
            "title": attributes.get("title", ""),
            "summary_html": attributes.get("summary", ""),
            "body_html": attributes.get("content", ""),
            "images": attributes.get("images", []),
            "url": f"https://seekingalpha.com/article/{article_id}"
        }

    except Exception as e:
        print("fetch_analysis_details ERROR:", e)
        return {}


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
        print(f"- {a.published} | {a.title} | id={a.id} | {a.url}")

    print("\n=== AI DIGEST ===\n")
    print(digest_text)

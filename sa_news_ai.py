import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence
from urllib.parse import urlparse

import openai
import requests
from bs4 import BeautifulSoup

# Configure OpenAI using the classic client style (same as ai_insights.py)
openai.api_key = os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------


@dataclass
class SANewsItem:
    symbol: str
    headline: str
    url: str
    date: str  # ISO date: YYYY-MM-DD
    source: str = "Seeking Alpha"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _parse_sa_date(text: str) -> Optional[str]:
    """Convert SA-style date strings into YYYY-MM-DD."""
    if not text:
        return None
    t = text.strip().replace("·", "").strip()

    for fmt in ("%b. %d, %Y", "%b %d, %Y"):
        try:
            dt = datetime.strptime(t, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def _browser_headers() -> Dict[str, str]:
    """
    Browser-like headers to avoid looking like a bot client.
    This is *not* an anti-bot bypass – just polite scraping.
    """
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                  "image/avif,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
    }


def _is_useful_news_url(url: str) -> bool:
    """
    Filter out noise like SA homepage, terms, help pages, search, etc.
    Keep:
      - Any non-SA news URL.
      - For seekingalpha.com, only /news/ or /article/ paths.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    host = (parsed.netloc or "").lower()
    path = parsed.path or "/"

    # Filter out SA non-article pages
    if "about.seekingalpha.com" in host:
        return False
    if "help.seekingalpha.com" in host:
        return False

    if "seekingalpha.com" in host:
        # Drop root/home, basic search, generic hash-only links
        if path == "/" or "/basic-search" in path:
            return False
        # Only keep news/article content
        if "/news/" not in path and "/article/" not in path:
            return False

    # If we reached here, URL looks like a real article/news
    return True


# ---------------------------------------------------------------------
# Seeking Alpha HTML fetch + parse
# ---------------------------------------------------------------------


def _fetch_sa_symbol_news_once(symbol: str) -> Optional[str]:
    """Fetch raw HTML for the Seeking Alpha symbol news page once."""
    url = f"https://seekingalpha.com/symbol/{symbol}/news"
    try:
        resp = requests.get(url, headers=_browser_headers(), timeout=20)
    except Exception:
        return None

    if resp.status_code != 200:
        return None

    return resp.text


def _scrape_sa_news_from_html(symbol: str, html: str, lookback_days: int) -> List[SANewsItem]:
    cutoff = datetime.utcnow().date() - timedelta(days=lookback_days)

    soup = BeautifulSoup(html, "html.parser")

    # SA rotates class names; search broadly and then filter.
    cards = soup.find_all(["li", "article", "div"], class_=re.compile("news|article|list|item|sc-"))

    items: List[SANewsItem] = []
    for card in cards:
        a = card.find("a", href=True)
        if not a:
            continue
        headline = a.get_text(strip=True)
        if not headline:
            continue

        href = a["href"]
        if not href.startswith("http"):
            href = "https://seekingalpha.com" + href

        # Drop nav links / terms / etc.
        if not _is_useful_news_url(href):
            continue

        # find date
        date_el = card.find("time") or card.find("span")
        raw_date = date_el.get_text(strip=True) if date_el else ""
        parsed = _parse_sa_date(raw_date)
        if parsed:
            news_date = datetime.strptime(parsed, "%Y-%m-%d").date()
        else:
            # if we cannot parse date, assume "today" so it passes the cutoff
            news_date = datetime.utcnow().date()

        if news_date < cutoff:
            continue

        items.append(
            SANewsItem(
                symbol=symbol.upper(),
                headline=headline,
                url=href,
                date=parsed or news_date.isoformat(),
                source="Seeking Alpha",
            )
        )

    return items


# ---------------------------------------------------------------------
# Google News fallback (RSS) – also using _is_useful_news_url
# ---------------------------------------------------------------------


def _fetch_google_news(symbol: str, lookback_days: int = 14, max_items: int = 30) -> List[SANewsItem]:
    """
    Fallback: use Google News RSS when SA HTML is unavailable or empty.
    """
    query = f"{symbol} stock"
    url = (
        "https://news.google.com/rss/search"
        f"?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        resp = requests.get(url, headers=_browser_headers(), timeout=20)
    except Exception:
        return []

    if resp.status_code != 200:
        return []

    soup = BeautifulSoup(resp.text, "xml")
    items_xml = soup.find_all("item")[:max_items]

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    out: List[SANewsItem] = []

    for it in items_xml:
        title_el = it.find("title")
        link_el = it.find("link")
        pub_el = it.find("pubDate")

        if not title_el or not link_el:
            continue

        headline = title_el.get_text(strip=True)
        url_item = link_el.get_text(strip=True)

        # Filter out junk like SA homepage, terms, help, etc.
        if not _is_useful_news_url(url_item):
            continue

        # parse pubDate, fallback to today on failure
        raw_date = pub_el.get_text(strip=True) if pub_el else ""
        try:
            dt = datetime.strptime(raw_date, "%a, %d %b %Y %H:%M:%S %Z")
        except Exception:
            dt = datetime.utcnow()

        if dt < cutoff:
            continue

        out.append(
            SANewsItem(
                symbol=symbol.upper(),
                headline=headline,
                url=url_item,
                date=dt.strftime("%Y-%m-%d"),
                source="Google News",
            )
        )

    return out


# ---------------------------------------------------------------------
# Public API: single- and multi-ticker news fetch
# ---------------------------------------------------------------------


def fetch_symbol_news(
    symbol: str,
    lookback_days: int = 14,
    max_retries: int = 3,
    backoff_seconds: int = 5,
    use_google_fallback: bool = True,
) -> List[SANewsItem]:
    """
    High-level entry used by Streamlit.

    - Try SA symbol news page a few times with backoff.
    - If SA is not reachable or yields zero headlines, optionally fall back to
      Google News RSS.
    - Never raises on network issues; returns [] instead.
    """
    symbol = symbol.upper()
    html: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        html = _fetch_sa_symbol_news_once(symbol)
        if html:
            break
        time.sleep(backoff_seconds * attempt)

    items: List[SANewsItem] = []
    if html:
        try:
            items = _scrape_sa_news_from_html(symbol, html, lookback_days)
        except Exception:
            items = []

    if not items and use_google_fallback:
        items = _fetch_google_news(symbol, lookback_days)

    return items


def fetch_news_for_tickers(
    symbols: Sequence[str],
    lookback_days: int = 14,
    max_retries: int = 3,
    backoff_seconds: int = 5,
    use_google_fallback: bool = True,
) -> Dict[str, List[SANewsItem]]:
    """
    Convenience for multi-ticker ingestion.

    Returns a dict mapping ticker -> list of SANewsItem.
    Does not raise on network issues; each list may be empty.
    """
    out: Dict[str, List[SANewsItem]] = {}
    for sym in symbols:
        out[sym.upper()] = fetch_symbol_news(
            sym,
            lookback_days=lookback_days,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            use_google_fallback=use_google_fallback,
        )
    return out


# ---------------------------------------------------------------------
# AI digest builder
# ---------------------------------------------------------------------


NEWS_DIGEST_PROMPT = """You are an institutional-grade financial analyst at a buy-side firm.

You are given a list of recent headlines for a single stock. Your job:

1. Summarise, in 3–6 sentences, what has been happening with the company over this window.
2. Focus on material developments: earnings, guidance, products, AI strategy, regulation,
   management changes, capital allocation, large contracts, or macro impacts.
3. Do not invent events that are not implied by the headlines.
4. Keep the tone analytical and concise. This will be pasted directly into internal research notes.
"""


def build_news_digest(symbol: str, items: List[SANewsItem], model: str = "gpt-4o-mini") -> str:
    """
    Summarise the news items using OpenAI's ChatCompletion API.

    If there are no items, we return a short message instead of calling OpenAI.
    """
    if not items:
        return "No recent public headlines available for this ticker in the selected window."

    headlines_block = "\n".join(f"- {it.headline} ({it.date}, {it.source})" for it in items)

    user_prompt = f"""Ticker: {symbol.upper()}

Recent headlines:
{headlines_block}

{NEWS_DIGEST_PROMPT}
"""

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001
        return f"[AI digest unavailable: {exc}]\n\nHeadlines:\n{headlines_block}"

    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return "Model returned an unexpected response format."

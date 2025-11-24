"""
Seeking Alpha symbol news → Cutler-style excerpts + manifest.

For each ticker (e.g. TSLA), this module:
- Opens https://seekingalpha.com/symbol/{TICKER}/news in a real browser
  (Selenium + headless Chrome).
- Scrapes the public news list (headlines + URLs, and dates where possible).
- Does NOT log in or attempt to bypass any article paywall.
- Writes one excerpts_clean.json per ticker under:
      BSD/Excerpts/<Quarter>/SeekingAlpha/<TICKER>/excerpts_clean.json
- Writes a manifest JSON under:
      BSD/Manifests/SeekingAlpha_<Quarter>.json

Streamlit app (sa_app.py) calls:
    from seekingalpha_excerpts import SAConfig, run_seekingalpha_excerpts
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

from datetime import datetime, timedelta


# ---------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
BSD_ROOT = HERE / "BSD"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("seekingalpha_excerpts")


# ---------------------------------------------------------------------
# Config dataclass (must match sa_app.py usage)
# ---------------------------------------------------------------------


@dataclass
class SAConfig:
    quarter: str
    tickers: List[str]
    days: int = 14
    min_words: int = 0

    # Kept for compatibility with older version; not used here
    rss_urls: List[str] | None = None
    build_pdf: bool = False

    base_dir: Path = BSD_ROOT


# ---------------------------------------------------------------------
# Selenium helpers
# ---------------------------------------------------------------------


def _new_driver() -> webdriver.Chrome:
    """Create a headless Chrome driver suitable for scraping the news page."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1400,900")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def _fetch_sa_news_page_selenium(
    ticker: str,
    max_refreshes: int = 4,
) -> Optional[str]:
    """
    Use Selenium to load https://seekingalpha.com/symbol/{ticker}/news
    and return the fully rendered page_source, or None on error.

    To mimic what worked manually (incognito + hard refresh), we:
    - load the page
    - wait for at least one '/news/...' link
    - if we don't see it in time, we refresh and try again, up to max_refreshes.
    """
    symbol = ticker.upper().strip()
    url = f"https://seekingalpha.com/symbol/{symbol}/news"

    driver = None
    try:
        driver = _new_driver()
        driver.get(url)

        for attempt in range(1, max_refreshes + 1):
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, 'a[href^="/news/"]')
                    )
                )
                links = driver.find_elements(By.CSS_SELECTOR, 'a[href^="/news/"]')
                if links:
                    log.info(
                        "Loaded SA news page for %s on attempt %d (found %d links).",
                        symbol,
                        attempt,
                        len(links),
                    )
                    return driver.page_source
                else:
                    log.warning(
                        "Attempt %d for %s: no links found after wait, refreshing.",
                        attempt,
                        symbol,
                    )
            except TimeoutException:
                log.warning(
                    "Attempt %d for %s: timeout waiting for news links, refreshing.",
                    attempt,
                    symbol,
                )

            if attempt < max_refreshes:
                driver.refresh()
            else:
                log.error(
                    "Failed to load usable news page for %s after %d attempts.",
                    symbol,
                    max_refreshes,
                )

        return None

    except Exception as exc:
        log.exception("Error loading SA news page for %s: %s", symbol, exc)
        return None
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------


def _normalise_date(raw: Optional[str]) -> str:
    """
    Normalize various date formats from Seeking Alpha symbol pages
    into YYYY-MM-DD.

    Handles inputs like:
      - "2025-11-07"
      - "Nov. 07, 2025"
      - "Nov 07, 2025"
      - "Fri, Nov. 07"
      - "Nov. 07"
      - "Today" / "Yesterday"
    """
    now = datetime.utcnow()

    if not raw:
        return now.date().isoformat()

    # Normalize whitespace
    raw = " ".join(str(raw).split())
    lower = raw.lower()

    # Relative dates
    if "today" in lower:
        return now.date().isoformat()
    if "yesterday" in lower:
        return (now.date() - timedelta(days=1)).isoformat()

    # Strip weekday prefix like "Fri, Nov. 07"
    # If there is a comma and the part before it is a single token -> likely weekday
    raw_no_wd = raw
    if "," in raw:
        first, rest = raw.split(",", 1)
        if len(first.split()) == 1:  # "Fri"
            raw_no_wd = rest.strip()
        else:
            raw_no_wd = raw

    # Check whether a 4-digit year is already present
    year = None
    for tok in raw_no_wd.split():
        if tok.isdigit() and len(tok) == 4:
            year = int(tok)
            break

    if year is None:
        year = now.year
        candidate = f"{raw_no_wd} {year}"
    else:
        candidate = raw_no_wd

    # Try several common formats
    fmts = [
        "%b %d %Y",       # Nov 07 2025
        "%b. %d %Y",      # Nov. 07 2025
        "%B %d %Y",       # November 07 2025
        "%b %d, %Y",      # Nov 07, 2025
        "%b. %d, %Y",     # Nov. 07, 2025
        "%B %d, %Y",      # November 07, 2025
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ]

    for fmt in fmts:
        try:
            dt = datetime.strptime(candidate, fmt)
            return dt.date().isoformat()
        except ValueError:
            continue

    # Fallback if everything fails
    return now.date().isoformat()



def _parse_sa_news(html: str, ticker: str, max_items: int = 40) -> List[Dict[str, Any]]:
    """
    Extract a list of news items from the rendered HTML.

    We look for anchor tags with href starting "/news/..." and treat the
    visible text as the headline. For each anchor, we inspect its parent
    container for small bits of text that look like dates.
    """
    soup = BeautifulSoup(html, "html.parser")
    base_url = "https://seekingalpha.com"

    items: List[Dict[str, Any]] = []

    # Month name snippets we use to detect likely date strings
    month_fragments = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ]

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/news/"):
            continue

        title = a.get_text(strip=True)
        # Filter out tiny or non-headline texts
        if not title or len(title) < 15:
            continue

        # Try to find a "card" / container for this link
        card = (
            a.find_parent("article")
            or a.find_parent("div")
            or a.parent
        )

        date_str = None

        # 1) First preference: any <time> tag in the card
        if card is not None:
            for time_tag in card.find_all("time"):
                txt = time_tag.get("datetime") or time_tag.get_text(strip=True)
                if txt:
                    date_str = txt
                    break

        # 2) If no <time> tag, scan small bits of text in the card for month names
        if card is not None and not date_str:
            for tag in card.find_all(["span", "div", "p", "small"], limit=30):
                txt = tag.get_text(" ", strip=True)
                if not txt:
                    continue

                lower = txt.lower()
                if any(m in lower for m in month_fragments) or "today" in lower or "yesterday" in lower:
                    date_str = txt
                    break

        published = _normalise_date(date_str)
        full_url = urljoin(base_url, href)

        item = {
            "headline": title,
            "url": full_url,
            "date": published,
            "source": "Seeking Alpha",
            "ticker": ticker.upper(),
        }
        items.append(item)

        if len(items) >= max_items:
            break

    return items


# ---------------------------------------------------------------------
# JSON writing helpers
# ---------------------------------------------------------------------


def _write_excerpts_json(
    base_dir: Path, quarter: str, ticker: str, news_items: List[Dict[str, Any]]
) -> Path:
    """
    Create:
        BSD/Excerpts/<Quarter>/SeekingAlpha/<TICKER>/excerpts_clean.json

    JSON structure:

    {
      "ticker": "TSLA",
      "source": "Seeking Alpha symbol news",
      "items": [
        {
          "headline": "...",
          "url": "...",
          "date": "YYYY-MM-DD",
          "source": "Seeking Alpha"
        },
        ...
      ]
    }
    """
    quarter_folder = quarter.strip()  # keep "2025 Q3" exactly

    out_dir = (
        base_dir
        / "Excerpts"
        / quarter_folder
        / "SeekingAlpha"
        / ticker.upper().strip()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "excerpts_clean.json"

    payload = {
        "ticker": ticker.upper().strip(),
        "source": "Seeking Alpha symbol news",
        "items": news_items,
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _write_manifest(
    base_dir: Path, quarter: str, run_ts: str, manifest_items: List[Dict[str, Any]]
) -> Path:
    """
    Create one manifest file:

        BSD/Manifests/SeekingAlpha_<Quarter>.json
    """
    quarter_folder = quarter.strip()
    manifests_dir = base_dir / "Manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    quarter_slug = quarter_folder.replace(" ", "_")
    manifest_path = manifests_dir / f"SeekingAlpha_{quarter_slug}.json"

    manifest = {
        "quarter": quarter,
        "created_at": run_ts,
        "items": manifest_items,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------
# Public entrypoint used by sa_app.py
# ---------------------------------------------------------------------


def run_seekingalpha_excerpts(cfg: SAConfig) -> List[Dict[str, Any]]:
    """
    Orchestrates Selenium fetch + parse + JSON write for each ticker.
    Returns manifest-style list of items for Streamlit.
    """
    if not cfg.tickers:
        log.warning("No tickers provided to SAConfig; nothing to do.")
        return []

    run_ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    manifest_items: List[Dict[str, Any]] = []

    for ticker in cfg.tickers:
        symbol = ticker.upper().strip()
        log.info("Fetching Seeking Alpha news for %s …", symbol)

        html = _fetch_sa_news_page_selenium(symbol)
        if not html:
            log.warning("Skipping %s – could not fetch symbol news page.", symbol)
            continue

        news_items = _parse_sa_news(html, symbol)
        if not news_items:
            log.warning(
                "No news items parsed for %s – HTML structure may have changed.",
                symbol,
            )
            continue

        # Write excerpts_clean.json
        excerpts_path = _write_excerpts_json(
            cfg.base_dir, cfg.quarter, symbol, news_items
        )

        first_date = news_items[0].get("date") or run_ts[:10]
        manifest_item = {
            "source": "Seeking Alpha symbol news",
            "id": f"{symbol}_{first_date}",
            "url": f"https://seekingalpha.com/symbol/{symbol}/news",
            "title": f"{symbol} news snapshot ({cfg.days}d)",
            "published": first_date,
            "tickers": [symbol],
            "excerpts_json": str(excerpts_path),
            "forced_ticker": symbol,
            "feed_url": f"https://seekingalpha.com/symbol/{symbol}/news",
        }
        manifest_items.append(manifest_item)

    if manifest_items:
        _write_manifest(cfg.base_dir, cfg.quarter, run_ts, manifest_items)

    return manifest_items


# ---------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch Seeking Alpha symbol news and write Cutler-style excerpts."
    )
    parser.add_argument("--quarter", required=True, help='e.g. "2025 Q3"')
    parser.add_argument(
        "--tickers", nargs="+", required=True, help="Tickers like TSLA NVDA AAPL"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Lookback window (only used in manifest title).",
    )
    args = parser.parse_args()

    cfg = SAConfig(
        quarter=args.quarter,
        tickers=args.tickers,
        days=args.days,
    )
    out_items = run_seekingalpha_excerpts(cfg)
    print(json.dumps(out_items, indent=2))

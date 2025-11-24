# rss_lookup.py
import requests
from bs4 import BeautifulSoup
import re

COMMON_RSS_PATTERNS = [
    r"rss",
    r".xml",
    r"feed",
    r"podcast",
]

HOSTING_KEYWORDS = [
    "buzzsprout",
    "simplecast",
    "libsyn",
    "megaphone",
    "podbean",
    "anchor.fm",
    "captivate.fm",
    "spreaker",
]

def discover_rss_from_website(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(r.text, "html.parser")

    # Method 1: <link rel="alternate" type="application/rss+xml">
    for link in soup.find_all("link"):
        if link.get("type") == "application/rss+xml" and link.get("href"):
            return link["href"]

    # Method 2: search for anchors containing rss/xml/feed
    for a in soup.find_all("a"):
        href = a.get("href", "").lower()
        if any(pat in href for pat in COMMON_RSS_PATTERNS):
            return href

    # Method 3: guess from hosting provider URLs
    text = soup.get_text(" ", strip=True).lower()
    if any(h in text for h in HOSTING_KEYWORDS):
        for a in soup.find_all("a"):
            href = a.get("href", "").lower()
            if "xml" in href or "rss" in href:
                return href

    return None

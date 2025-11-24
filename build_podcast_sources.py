import csv
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from rss_lookup import discover_rss_from_website


ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


def search_itunes_podcast(podcast_name: str) -> Optional[dict]:
    """
    Call the iTunes Search API to find a podcast by name.
    Returns the best-matching result dict or None.
    """
    params = {
        "term": podcast_name,
        "media": "podcast",
        "limit": 5,
    }
    try:
        resp = requests.get(ITUNES_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] iTunes search failed for '{podcast_name}': {e}")
        return None

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return None

    # Try exact match on collectionName first, then fallback to "contains"
    podcast_name_l = podcast_name.lower().strip()

    exact = None
    partial = None
    for r in results:
        coll = (r.get("collectionName") or "").lower().strip()
        if coll == podcast_name_l:
            exact = r
            break
        if podcast_name_l in coll and partial is None:
            partial = r

    return exact or partial or results[0]


def get_publisher_website_from_apple_page(apple_url: str) -> Optional[str]:
    """
    Visit the Apple Podcasts show page and try to extract the
    publisher 'Website' link (not the Apple page itself).
    """
    if not apple_url:
        return None

    try:
        resp = requests.get(
            apple_url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (CutlerResearchBot)"},
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Could not fetch Apple page {apple_url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Apple typically has an "Website" anchor somewhere on the page
    for a in soup.find_all("a"):
        text = (a.get_text() or "").strip().lower()
        if "website" in text:
            href = a.get("href")
            if href and not href.startswith("mailto:"):
                return href

    return None


def process_podcast(name: str):
    """
    For one podcast name:
      - search iTunes
      - get apple_url, feedUrl (rss_url)
      - scrape publisher website from apple page
      - optionally refine rss_url via discover_rss_from_website()
    """
    result = search_itunes_podcast(name)
    if not result:
        print(f"[WARN] No iTunes result found for '{name}'")
        return {
            "podcast_name": name,
            "apple_url": None,
            "website_url": None,
            "rss_url": None,
            "rss_from_website": None,
        }

    apple_url = result.get("collectionViewUrl")
    rss_url = result.get("feedUrl")  # direct RSS from Apple (often all you need)

    website_url = get_publisher_website_from_apple_page(apple_url)
    rss_from_website = None
    if website_url:
        try:
            rss_from_website = discover_rss_from_website(website_url)
        except Exception as e:
            print(f"[WARN] discover_rss_from_website failed for {website_url}: {e}")

    print(
        f"[OK] {name} | apple_url={bool(apple_url)} "
        f"website_url={bool(website_url)} rss_url={bool(rss_url)}"
    )

    return {
        "podcast_name": name,
        "apple_url": apple_url,
        "website_url": website_url,
        "rss_url": rss_url,
        "rss_from_website": rss_from_website,
    }


def main():
    in_path = Path("podcast_list.csv")
    out_path = Path("podcast_sources.csv")

    rows = []
    with in_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["podcast_name"].strip()
            if not name:
                continue
            rows.append(name)

    print(f"Found {len(rows)} podcasts in {in_path}")

    results = []
    for i, name in enumerate(rows, start=1):
        print(f"\n[{i}/{len(rows)}] Processing '{name}'")
        rec = process_podcast(name)
        results.append(rec)
        # small delay to be polite to Apple
        time.sleep(0.4)

    fieldnames = [
        "podcast_name",
        "apple_url",
        "website_url",
        "rss_url",
        "rss_from_website",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nWrote resolved podcast sources to {out_path}")


if __name__ == "__main__":
    main()

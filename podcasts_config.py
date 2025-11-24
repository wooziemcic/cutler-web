# podcasts_config.py
"""
Podcast universe configuration for the Cutler Research Platform.
...
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# --- PATH FIX (PyInstaller-safe) ---
HERE = Path(__file__).resolve().parent

def _resolve_sources_csv() -> Path:
    """
    Resolve podcast_sources.csv robustly:
    1) same folder as this file (works in _internal)
    2) current working directory (dev mode)
    3) PyInstaller _MEIPASS (extra safety)
    """
    candidates = [
        HERE / "podcast_sources.csv",
        Path.cwd() / "podcast_sources.csv",
    ]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "podcast_sources.csv")

    for c in candidates:
        if c.exists():
            return c

    # fall back to the first candidate for a helpful error message
    return candidates[0]

SOURCES_CSV_DEFAULT = _resolve_sources_csv()
# --- END PATH FIX ---


@dataclass
class Podcast:
    id: str
    name: str
    rss: str
    website_url: Optional[str] = None
    apple_url: Optional[str] = None
    priority: str = "core"  # "core" | "secondary" | "experimental"


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "podcast"


def load_podcasts(
    csv_path: Path = SOURCES_CSV_DEFAULT,
    *,
    include_without_rss: bool = False,
) -> List[Podcast]:
    """
    Load Podcast definitions from podcast_sources.csv.

    Priority rules for RSS:
    - prefer rss_url if present
    - else rss_from_website
    - else skip (unless include_without_rss=True)
    """
    podcasts: List[Podcast] = []

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("podcast_name") or "").strip()
            if not name:
                continue

            rss_url = (row.get("rss_url") or "").strip()
            rss_from_web = (row.get("rss_from_website") or "").strip()
            rss = rss_url or rss_from_web

            if not rss and not include_without_rss:
                continue

            website_url = (row.get("website_url") or "").strip() or None
            apple_url = (row.get("apple_url") or "").strip() or None
            podcast_id = _slugify(name)

            podcasts.append(
                Podcast(
                    id=podcast_id,
                    name=name,
                    rss=rss,
                    website_url=website_url,
                    apple_url=apple_url,
                    priority="core",
                )
            )

    return podcasts


def get_podcast_by_id(pid: str, podcasts: List[Podcast]) -> Optional[Podcast]:
    for p in podcasts:
        if p.id == pid:
            return p
    return None


PODCASTS: List[Podcast] = load_podcasts()

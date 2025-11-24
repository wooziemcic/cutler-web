# podcast_ingest.py

from __future__ import annotations

import os
import json
import re
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from dotenv import load_dotenv
import feedparser
import requests
from bs4 import BeautifulSoup
import openai

from podcasts_config import PODCASTS, Podcast

# ----- Env / OpenAI setup -----
HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Whisper chunking
WHISPER_CHUNK_BYTES = 20 * 1024 * 1024  # 20 MB per chunk, safely below 25 MB API limit

# External transcript APIs
LISTENNOTES_API_KEY = os.getenv("LISTENNOTES_API_KEY")
PODCHASER_API_TOKEN = os.getenv("PODCHASER_API_TOKEN")  # bearer JWT

LISTENNOTES_BASE = "https://listen-api.listennotes.com/api/v2"
PODCHASER_GQL = "https://api.podchaser.com/graphql"

LISTENNOTES_MONTHLY_CAP = int(os.getenv("LISTENNOTES_MONTHLY_CAP", "300"))


# ------------------------------
# Dataclasses
# ------------------------------

@dataclass
class EpisodeRecord:
    podcast_id: str
    podcast_name: str
    episode_id: str          # unique per feed (entry.id or link)
    title: str
    published: str           # ISO 8601 string
    audio_url: Optional[str]
    page_url: Optional[str]
    transcript_text: Optional[str]
    transcript_source: str   # "rss_content" | "rss_summary" | "html_page" | "podchaser" | "listennotes" | "whisper" | "none"


# ------------------------------
# Utility functions
# ------------------------------

def _sanitize_for_filename(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "episode"


def _parse_published_date(entry) -> Optional[datetime]:
    dt = None
    if getattr(entry, "published", None):
        try:
            dt = parsedate_to_datetime(entry.published)
        except Exception:
            dt = None

    if dt is None and getattr(entry, "updated", None):
        try:
            dt = parsedate_to_datetime(entry.updated)
        except Exception:
            dt = None

    if dt is None:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _entry_audio_url(entry) -> Optional[str]:
    enclosures = getattr(entry, "enclosures", []) or []
    for enc in enclosures:
        url = enc.get("href") or enc.get("url")
        if url:
            return url

    links = getattr(entry, "links", []) or []
    for ln in links:
        if ln.get("type", "").startswith("audio/") and ln.get("href"):
            return ln["href"]

    return None


def _looks_like_transcript(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower()
    if (
        "cloudflare ray id" in lowered
        or "performance & security by cloudflare" in lowered
        or "error 5" in lowered
        or ("<html" in lowered and "cloudflare" in lowered)
    ):
        return False

    plain = re.sub(r"<[^>]+>", " ", text)
    plain = re.sub(r"\s+", " ", plain).strip()
    if len(plain) < 800:
        return False

    sentences = re.split(r"[.!?]", plain)
    long_sentences = [s for s in sentences if len(s.split()) >= 6]
    return len(long_sentences) >= 10


def _extract_text_from_rss_entry(entry) -> Tuple[Optional[str], str]:
    contents = getattr(entry, "content", None)
    if contents:
        best = max((c.get("value", "") for c in contents), key=len, default="")
        if _looks_like_transcript(best):
            return best.strip(), "rss_content"

    summary = getattr(entry, "summary", "") or ""
    if _looks_like_transcript(summary):
        return summary.strip(), "rss_summary"

    desc = getattr(entry, "description", "") or ""
    if _looks_like_transcript(desc):
        return desc.strip(), "rss_summary"

    return None, "none"


def _fetch_html_transcript(page_url: str, timeout: int = 10) -> Optional[str]:
    if not page_url:
        return None

    try:
        resp = requests.get(page_url, timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return None

    lower = resp.text.lower()
    if (
        "cloudflare ray id" in lower
        or "performance & security by cloudflare" in lower
        or "error 5" in lower
    ):
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    for selector in ("article", "main", "div.entry-content", "div.post-content"):
        node = soup.select_one(selector)
        if node:
            text = node.get_text(" ", strip=True)
            if _looks_like_transcript(text):
                return text

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    big_blob = "\n".join(paragraphs)
    if _looks_like_transcript(big_blob):
        return big_blob

    return None


# ------------------------------
# API usage accounting
# ------------------------------

def _usage_path(output_root: Path) -> Path:
    return output_root / "_api_usage.json"


def _load_usage(output_root: Path) -> Dict[str, Any]:
    path = _usage_path(output_root)
    if not path.exists():
        return {"listen_notes_calls": 0, "listen_notes_cap": LISTENNOTES_MONTHLY_CAP}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"listen_notes_calls": 0, "listen_notes_cap": LISTENNOTES_MONTHLY_CAP}

def _transcribe_with_deepgram(audio_url: str, timeout: int = 60) -> Optional[str]:
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    if not DEEPGRAM_API_KEY or not audio_url:
        return None

    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    params = {"punctuate": "true", "diarize": "false"}

    try:
        resp = requests.post(
            "https://api.deepgram.com/v1/listen",
            headers=headers,
            params=params,
            json={"url": audio_url},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        # parse transcript (keep defensive)
        return (
            data.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
                .strip()
        ) or None
    except Exception as e:
        print(f"    [ERROR] Deepgram failed: {e}")
        return None

def _save_usage(output_root: Path, usage: Dict[str, Any]) -> None:
    path = _usage_path(output_root)
    path.write_text(json.dumps(usage, indent=2), encoding="utf-8")


def _listen_notes_budget_left(output_root: Path) -> bool:
    usage = _load_usage(output_root)
    return usage.get("listen_notes_calls", 0) < usage.get("listen_notes_cap", LISTENNOTES_MONTHLY_CAP)


def _bump_listen_notes_usage(output_root: Path) -> None:
    usage = _load_usage(output_root)
    usage["listen_notes_calls"] = int(usage.get("listen_notes_calls", 0)) + 1
    usage["listen_notes_cap"] = int(usage.get("listen_notes_cap", LISTENNOTES_MONTHLY_CAP))
    _save_usage(output_root, usage)


# ------------------------------
# Podchaser transcript fetch
# ------------------------------

def _fetch_podchaser_transcript(
    podcast_name: str,
    episode_title: str,
    page_url: Optional[str] = None,
    timeout: int = 15,
) -> Optional[str]:
    if not PODCHASER_API_TOKEN:
        return None

    # Search episodes by title + podcast name
    query = """
    query SearchEpisodes($q: String!) {
      searchEpisodes(q: $q, first: 3) {
        edges {
          node {
            id
            title
            podcast { title }
            transcripts {
              edges {
                node {
                  text
                  url
                }
              }
            }
          }
        }
      }
    }
    """

    q_str = f"{episode_title} {podcast_name}".strip()
    headers = {
        "Authorization": f"Bearer {PODCHASER_API_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(
            PODCHASER_GQL,
            headers=headers,
            json={"query": query, "variables": {"q": q_str}},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    [WARN] Podchaser search failed: {e}")
        return None

    edges = (((data or {}).get("data") or {}).get("searchEpisodes") or {}).get("edges") or []
    for edge in edges:
        node = (edge or {}).get("node") or {}
        transcripts = ((node.get("transcripts") or {}).get("edges")) or []
        if not transcripts:
            continue

        # Prefer direct text if present
        for tedge in transcripts:
            tnode = (tedge or {}).get("node") or {}
            ttext = (tnode.get("text") or "").strip()
            turl = (tnode.get("url") or "").strip()

            if ttext and _looks_like_transcript(ttext):
                return ttext

            if turl:
                try:
                    t_resp = requests.get(turl, timeout=timeout)
                    if t_resp.ok:
                        raw = t_resp.text.strip()
                        if _looks_like_transcript(raw):
                            return raw
                except Exception:
                    pass

    return None


# ------------------------------
# Listen Notes transcript fetch
# ------------------------------

def _fetch_listennotes_transcript(
    output_root: Path,
    podcast_name: str,
    episode_title: str,
    timeout: int = 15,
) -> Optional[str]:
    if not LISTENNOTES_API_KEY:
        return None
    if not _listen_notes_budget_left(output_root):
        return None

    headers = {"X-ListenAPI-Key": LISTENNOTES_API_KEY}

    # 1) Search for the episode
    search_params = {
        "q": f"{episode_title} {podcast_name}",
        "type": "episode",
        "only_in": "title,description",
        "sort_by_date": 1,
        "offset": 0,
        "len_min": 5,
    }

    try:
        _bump_listen_notes_usage(output_root)
        s_resp = requests.get(
            f"{LISTENNOTES_BASE}/search",
            headers=headers,
            params=search_params,
            timeout=timeout,
        )
        s_resp.raise_for_status()
        s_data = s_resp.json()
    except Exception as e:
        print(f"    [WARN] Listen Notes search failed: {e}")
        return None

    results = s_data.get("results") or []
    if not results:
        return None

    episode_id = results[0].get("id")
    if not episode_id:
        return None

    # 2) Fetch episode details with transcript
    try:
        if not _listen_notes_budget_left(output_root):
            return None

        _bump_listen_notes_usage(output_root)
        e_resp = requests.get(
            f"{LISTENNOTES_BASE}/episodes/{episode_id}",
            headers=headers,
            params={"show_transcript": 1},
            timeout=timeout,
        )
        e_resp.raise_for_status()
        e_data = e_resp.json()
    except Exception as e:
        print(f"    [WARN] Listen Notes episode fetch failed: {e}")
        return None

    transcript = (e_data.get("transcript") or "").strip()
    if transcript and _looks_like_transcript(transcript):
        return transcript

    return None


# ------------------------------
# Whisper helpers (unchanged)
# ------------------------------

def _download_audio_to_tmp(
    audio_url: str,
    tmp_dir: Path,
    max_bytes: int = 20 * 1024 * 1024,
) -> Optional[Path]:
    if not audio_url:
        return None

    tmp_dir.mkdir(parents=True, exist_ok=True)
    last_part = audio_url.split("/")[-1] or "audio"
    sanitized = _sanitize_for_filename(last_part)

    valid_exts = [".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"]
    ext = None
    lower_name = sanitized.lower()
    for e in valid_exts:
        if lower_name.endswith(e):
            ext = e
            break
    if ext is None:
        ext = ".mp3"

    out_path = tmp_dir / f"{sanitized}{ext}"

    try:
        with requests.get(audio_url, stream=True, timeout=20) as r:
            r.raise_for_status()
            with out_path.open("wb") as f:
                size = 0
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > max_bytes:
                        break
                    f.write(chunk)
    except Exception:
        return None

    return out_path


def _transcribe_with_whisper(
    audio_url: str,
    tmp_dir: Path,
) -> Optional[str]:
    tmp_dir.mkdir(parents=True, exist_ok=True)

    audio_path = _download_audio_to_tmp(audio_url, tmp_dir)
    if not audio_path:
        return None

    try:
        with audio_path.open("rb") as f:
            print(f"    [INFO] Calling Whisper on {audio_path.name}.")
            resp = openai.Audio.transcribe(
                model="whisper-1",
                file=f,
            )
        text = (resp.get("text") or "").strip()
        if not text:
            return None
        return text
    except Exception as e:
        print(f"    [ERROR] Whisper transcription failed: {e}")
        return None


# ------------------------------
# Per-feed ingestion
# ------------------------------

def _ingest_podcast_feed(
    podcast: Podcast,
    since: datetime,
    until: datetime,
    enable_whisper: bool,
    whisper_tmp_dir: Path,
    output_root: Path,
) -> List[EpisodeRecord]:
    print(f"Fetching feed for {podcast.name} ...")
    try:
        feed = feedparser.parse(podcast.rss or "")
    except Exception as e:
        print(f"  [ERROR] Failed to parse feed: {e}")
        return []

    if feed.bozo:
        print(f"  [WARN] feedparser bozo=True: {feed.bozo_exception}")
    entries = getattr(feed, "entries", []) or []
    if not entries:
        print("  -> 0 episodes (0 with transcripts)")
        return []

    records: List[EpisodeRecord] = []
    episodes_with_transcripts = 0

    for entry in entries:
        pub_dt = _parse_published_date(entry)
        if not pub_dt:
            continue
        if not (since <= pub_dt <= until):
            continue

        title = getattr(entry, "title", "") or "(untitled episode)"
        page_url = getattr(entry, "link", None)
        audio_url = _entry_audio_url(entry)

        # Layer A1: RSS-level transcript
        transcript_text, source = _extract_text_from_rss_entry(entry)

        # Layer A2: HTML transcript on episode page
        if not transcript_text and page_url:
            html_text = _fetch_html_transcript(page_url)
            if html_text:
                transcript_text = html_text
                source = "html_page"

        # Layer B1: Podchaser
        if not transcript_text:
            pc_text = _fetch_podchaser_transcript(
                podcast_name=podcast.name,
                episode_title=title,
                page_url=page_url,
            )
            if pc_text:
                transcript_text = pc_text
                source = "podchaser"

        # Layer B2: Listen Notes (budgeted)
        if not transcript_text:
            ln_text = _fetch_listennotes_transcript(
                output_root=output_root,
                podcast_name=podcast.name,
                episode_title=title,
            )
            if ln_text:
                transcript_text = ln_text
                source = "listennotes"

        # Layer C: External STT fallback
        if not transcript_text and audio_url:
            if enable_whisper:
                w_text = _transcribe_with_whisper(audio_url, whisper_tmp_dir)
                if w_text:
                    transcript_text, source = w_text, "whisper"
            else:
                dg_text = _transcribe_with_deepgram(audio_url)
                if dg_text:
                    transcript_text, source = dg_text, "deepgram"

        if transcript_text:
            episodes_with_transcripts += 1

        ep_id = getattr(entry, "id", None) or getattr(entry, "guid", None) or page_url or audio_url or ""
        if not ep_id:
            ep_id = f"{podcast.id}-{pub_dt.isoformat()}"

        record = EpisodeRecord(
            podcast_id=podcast.id,
            podcast_name=podcast.name,
            episode_id=ep_id,
            title=title,
            published=pub_dt.isoformat(),
            audio_url=audio_url,
            page_url=page_url,
            transcript_text=transcript_text,
            transcript_source=source,
        )
        records.append(record)

    print(f"  -> {len(records)} episodes ({episodes_with_transcripts} with transcripts)")
    return records


# ------------------------------
# High-level orchestrator
# ------------------------------

def ingest_podcasts(
    output_root: Path,
    since: datetime,
    until: datetime,
    podcast_ids: Optional[List[str]] = None,
    max_episodes_per_podcast: int = 20,
    enable_whisper: bool = False,
) -> List[EpisodeRecord]:
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: List[EpisodeRecord] = []
    whisper_tmp_dir = output_root / "_audio_tmp"

    all_podcasts: List[Podcast] = PODCASTS
    if podcast_ids:
        requested = set(podcast_ids)
        all_podcasts = [p for p in all_podcasts if p.id in requested]

    for podcast in all_podcasts:
        records = _ingest_podcast_feed(
            podcast=podcast,
            since=since,
            until=until,
            enable_whisper=enable_whisper,
            whisper_tmp_dir=whisper_tmp_dir,
            output_root=output_root,
        )

        if max_episodes_per_podcast and len(records) > max_episodes_per_podcast:
            records = sorted(records, key=lambda r: r.published, reverse=True)[:max_episodes_per_podcast]

        pod_dir = output_root / podcast.id
        pod_dir.mkdir(parents=True, exist_ok=True)

        for rec in records:
            ep_slug = _sanitize_for_filename(rec.episode_id or rec.title)
            ep_dir = pod_dir / ep_slug
            ep_dir.mkdir(parents=True, exist_ok=True)

            meta = asdict(rec).copy()
            # Backward-compat shim for downstream scripts expecting episode_url
            if "episode_url" not in meta:
                meta["episode_url"] = meta.get("page_url")
            meta_path = ep_dir / "metadata.json"
            with meta_path.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            if rec.transcript_text:
                transcript_path = ep_dir / "transcript.txt"
                transcript_path.write_text(rec.transcript_text, encoding="utf-8")

        all_records.extend(records)

    return all_records


# ------------------------------
# CLI
# ------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest podcast episodes + transcripts from RSS/HTML/Podchaser/ListenNotes, then Whisper."
    )
    parser.add_argument("--out", required=True, help="Output root directory for podcast data.")
    parser.add_argument("--days", type=int, default=7, help="How many days back to fetch episodes.")
    parser.add_argument("--podcasts", nargs="*", help="Optional list of podcast IDs to restrict ingestion.")
    parser.add_argument("--max-per-podcast", type=int, default=20, help="Max episodes per podcast.")
    parser.add_argument("--whisper", action="store_true", help="Enable Whisper fallback when no transcript found.")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    since = now - timedelta(days=args.days)

    records = ingest_podcasts(
        output_root=Path(args.out),
        since=since,
        until=now,
        podcast_ids=args.podcasts,
        max_episodes_per_podcast=args.max_per_podcast,
        enable_whisper=args.whisper,
    )

    print(
        f"\nDone. Total episodes: {len(records)} "
        f"(with transcripts: {sum(1 for r in records if r.transcript_text)})"
    )

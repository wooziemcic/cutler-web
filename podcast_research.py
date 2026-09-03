"""Orchestration and the persistent podcast research store.

Ties the pieces together in cheapest-first order:

    candidates (metadata)      <- podcast_discovery
      -> skip anything already processed
      -> acquire transcript    <- podcast_ingest's existing layered sources
      -> verify relevance      <- podcast_evidence.classify_relevance
      -> extract verbatim      <- podcast_evidence.extract_excerpts
      -> persist

Transcripts are written to a store that survives the run. The interactive
pipeline wipes its working directory on every run, so anything kept only there
is gone as soon as the PDF is built; research evidence needs to outlive that.

Layout:

    <root>/_research/
        index.json                       one row per processed episode
        <episode_id>/metadata.json       ticker(s), podcast, dates, relevance
        <episode_id>/transcript.txt      full transcript, as fetched
        <episode_id>/excerpts.json       verbatim excerpts + topics
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from podcast_discovery import Candidate, DiscoveryState
from podcast_entities import TickerProfile, build_profiles
from podcast_evidence import (
    Excerpt,
    Relevance,
    classify_relevance,
    extract_excerpts,
    verify_verbatim,
)

RESEARCH_DIRNAME = "_research"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ResearchStore:
    """Durable home for transcripts, excerpts and processing status."""

    def __init__(self, root: Path):
        self.root = Path(root) / RESEARCH_DIRNAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def _load_index(self) -> Dict[str, Any]:
        try:
            if self.index_path.exists():
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_index(self, data: Dict[str, Any]) -> None:
        try:
            tmp = self.index_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.index_path)  # atomic: never leave a half index
        except Exception:
            pass

    def episode_dir(self, episode_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(episode_id))[:120] or "episode"
        d = self.root / safe
        d.mkdir(parents=True, exist_ok=True)
        return d

    def has_transcript(self, episode_id: str) -> bool:
        return (self.episode_dir(episode_id) / "transcript.txt").exists()

    def read_transcript(self, episode_id: str) -> str:
        p = self.episode_dir(episode_id) / "transcript.txt"
        try:
            return p.read_text(encoding="utf-8") if p.exists() else ""
        except Exception:
            return ""

    def save(
        self,
        *,
        episode_id: str,
        metadata: Dict[str, Any],
        transcript: str = "",
        excerpts: Optional[Sequence[Excerpt]] = None,
    ) -> Path:
        d = self.episode_dir(episode_id)
        if transcript:
            (d / "transcript.txt").write_text(transcript, encoding="utf-8")
        if excerpts is not None:
            (d / "excerpts.json").write_text(
                json.dumps([e.to_dict() for e in excerpts], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        metadata = dict(metadata)
        metadata["stored_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        metadata["has_transcript"] = bool(transcript) or self.has_transcript(episode_id)
        (d / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        idx = self._load_index()
        idx[str(episode_id)] = {
            k: metadata.get(k)
            for k in (
                "episode_id", "tickers", "podcast_name", "title", "published",
                "transcript_source", "relevance", "relevance_reason",
                "excerpt_count", "stored_at", "has_transcript",
            )
        }
        self._save_index(idx)
        return d

    def all_records(self) -> List[Dict[str, Any]]:
        return list(self._load_index().values())



# ---------------------------------------------------------------------------
# Free transcript sources, tried before anything that costs money
# ---------------------------------------------------------------------------

def _strip_captions(text: str) -> str:
    """Turn a VTT/SRT caption file into plain prose.

    Drops cue numbers, timing lines and WEBVTT headers, keeps any speaker
    prefix the file carries, and collapses the result.
    """
    import re as _re

    lines = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s or s.upper().startswith("WEBVTT") or s.isdigit():
            continue
        if "-->" in s:  # timing cue
            continue
        lines.append(s)
    out = " ".join(lines)
    return _re.sub(r"\s+", " ", out).strip()


def extract_html_transcript(html: str, min_words: int = 400) -> str:
    """Pull a speaker-labelled transcript out of an episode web page.

    Publishers who post a transcript render it as alternating speaker turns.
    Finding those turns is more reliable than guessing a container class, and
    it preserves the attribution the page already gives rather than inventing
    one.

    The hard part is not finding candidate turns but telling them from page
    chrome: a navigation block like "Core Values" or "Personal Finance"
    followed by a sentence looks identical to a speaker turn in isolation. The
    discriminator is repetition - a real conversation has two or three names
    alternating many times, while chrome labels appear once each.

    Returns "Speaker: text" lines for parse_transcript(), or "" if no
    transcript is present.
    """
    import re as _re

    if not html:
        return ""

    body = _re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=_re.S | _re.I)
    body = _re.sub(r"<(br|/p|/div|/h[1-6]|/li)[^>]*>", "\n", body, flags=_re.I)
    body = _re.sub(r"<[^>]+>", "\n", body)
    try:
        import html as _htmllib

        body = _htmllib.unescape(body)
    except Exception:
        pass
    lines = [l.strip() for l in body.splitlines()]
    lines = [l for l in lines if l]

    name_re = _re.compile(r"^([A-Z][A-Za-z.'-]{1,20}(?:\s+[A-Z][A-Za-z.'-]{1,20}){0,3})\s*:?\s*$")
    inline_re = _re.compile(r"^([A-Z][A-Za-z.'-]{1,20}(?:\s+[A-Z][A-Za-z.'-]{1,20}){0,3})\s*:\s+(\S.*)$")

    turns: list = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m_inline = inline_re.match(line)
        if m_inline and len(m_inline.group(2).split()) >= 4:
            turns.append((m_inline.group(1), m_inline.group(2), i))
            i += 1
            continue
        m_name = name_re.match(line)
        if m_name and i + 1 < len(lines):
            nxt = lines[i + 1]
            if len(nxt.split()) >= 6 and not name_re.match(nxt):
                turns.append((m_name.group(1), nxt, i))
                i += 2
                continue
        i += 1

    if len(turns) < 6:
        return ""

    # Telling a transcript from page chrome comes down to repetition: a real
    # conversation has a couple of names taking turns over and over, while
    # navigation labels ("Research", "Investment Management") appear a handful
    # of times at most.
    #
    # Names are written inconsistently on the same page - this one carries
    # "Brad Jacobs", "BRAD JACOBS" and a bare "Jacobs" - so speakers are keyed
    # on surname, otherwise one voice looks like three minor ones.
    def _key(name: str) -> str:
        parts = name.lower().replace(".", " ").split()
        return parts[-1] if parts else ""

    counts: dict = {}
    for sp, _t, _i in turns:
        k = _key(sp)
        if k:
            counts[k] = counts.get(k, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    if len(ranked) < 2 or ranked[1][1] < 3:
        return ""  # no second recurring voice: not an interview

    # An interview is two voices, sometimes with a narrator, so the top three
    # surnames are the conversation and everything else is furniture. Anyone
    # far below the second speaker is a passing quote, not a participant.
    floor = max(3, ranked[1][1] // 3)
    dominant = {k for k, c in ranked[:3] if c >= floor}
    if len(dominant) < 2:
        return ""

    # Publishers often print the same transcript twice on one page - a video
    # version and a text version - which would otherwise duplicate every
    # excerpt. Keep the first occurrence of each distinct line.
    seen: set = set()
    kept = []
    for sp, txt, idx in sorted(turns, key=lambda t: t[2]):
        if _key(sp) not in dominant:
            continue
        fingerprint = (_key(sp), " ".join(txt.lower().split())[:160])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        kept.append((sp, txt, idx))

    if len(kept) < 6:
        return ""

    # One page can print the same person as "Brad Jacobs", "BRAD JACOBS" and a
    # bare "Jacobs". Settle on the fullest, properly-cased variant so the report
    # attributes every turn to one name instead of three.
    canonical: dict = {}
    for sp, _t, _i in kept:
        k = _key(sp)
        cand = sp.title() if sp.isupper() else sp
        best = canonical.get(k)
        if best is None or len(cand.split()) > len(best.split()):
            canonical[k] = cand

    out = "\n".join(f"{canonical.get(_key(sp), sp)}: {txt}" for sp, txt, _ in kept)
    return out if len(out.split()) >= min_words else ""


def fetch_feed_transcript(feed_url: str, *, guid: str = "", title: str = "", timeout: int = 20) -> str:
    """Look for a transcript published in the podcast's own RSS feed.

    Two places carry one for free: the <podcast:transcript> tag from the
    Podcasting 2.0 namespace, and occasionally a full transcript inline in the
    entry content. Costs nothing but one feed fetch, so it runs before any paid
    speech-to-text.
    """
    if not feed_url:
        return ""
    try:
        import feedparser
        import requests

        parsed = feedparser.parse(feed_url)
    except Exception:
        return ""

    def _matches(entry) -> bool:
        if guid and str(entry.get("id") or entry.get("guid") or "") == guid:
            return True
        if title:
            a = "".join(ch for ch in str(entry.get("title") or "").lower() if ch.isalnum())
            b = "".join(ch for ch in title.lower() if ch.isalnum())
            return bool(a) and (a[:60] == b[:60])
        return False

    entry = next((e for e in getattr(parsed, "entries", []) or [] if _matches(e)), None)
    if entry is None:
        return ""

    # 1. <podcast:transcript url="..." type="..."/>
    urls: list = []
    for key in ("podcast_transcript", "transcript"):
        val = entry.get(key)
        if isinstance(val, dict) and val.get("url"):
            urls.append(str(val["url"]))
        elif isinstance(val, list):
            urls.extend(str(v.get("url")) for v in val if isinstance(v, dict) and v.get("url"))
    for link in entry.get("links", []) or []:
        if "transcript" in str(link.get("rel", "")).lower() and link.get("href"):
            urls.append(str(link["href"]))

    import requests as _rq

    for url in urls:
        try:
            resp = _rq.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 CutlerResearch"})
            resp.raise_for_status()
            body = resp.text or ""
        except Exception:
            continue
        if url.lower().endswith((".vtt", ".srt")) or "-->" in body[:2000]:
            body = _strip_captions(body)
        elif url.lower().endswith(".json"):
            try:
                import json as _json

                data = _json.loads(body)
                segs = data.get("segments") if isinstance(data, dict) else None
                if isinstance(segs, list):
                    body = " ".join(str(s.get("body") or s.get("text") or "") for s in segs)
            except Exception:
                pass
        body = body.strip()
        if body:
            return body

    # 2. A full transcript sometimes sits inline in the entry content.
    try:
        import podcast_ingest as ing

        text, _src = ing._extract_text_from_rss_entry(entry)
        return text or ""
    except Exception:
        return ""

def fetch_page_transcript(url: str, timeout: int = 30) -> str:
    """Fetch a web page and pull a speaker-labelled transcript out of it.

    Many publishers post the full transcript on their own site even when the
    RSS feed carries nothing but a blurb - Morgan Stanley's Hard Lessons is one,
    and its feed has no episode link at all. Given the URL this recovers the
    real conversation for free, which is otherwise a paid transcription.
    """
    if not url:
        return ""
    try:
        import requests as _rq

        resp = _rq.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                )
            },
        )
        resp.raise_for_status()
    except Exception:
        return ""
    return extract_html_transcript(resp.text)


# ---------------------------------------------------------------------------
# Transcript acquisition for a discovered episode
# ---------------------------------------------------------------------------

def acquire_transcript(
    candidate: Candidate,
    *,
    output_root: Path,
    allow_stt: bool = True,
) -> tuple[str, str]:
    """Get a transcript for one candidate, cheapest source first.

    Reuses podcast_ingest's existing layers rather than duplicating them:
    Listen Notes (we already hold the episode id) -> episode page HTML ->
    Podchaser -> speech-to-text. Returns (text, source); ("", "none") when no
    transcript could be obtained.
    """
    try:
        import podcast_ingest as ing
    except Exception as exc:  # pragma: no cover
        print(f"    [WARN] podcast_ingest unavailable: {exc}")
        return "", "none"

    # 1. A transcript published in the podcast's own feed. Free.
    if getattr(candidate, "feed_url", ""):
        try:
            text = fetch_feed_transcript(
                candidate.feed_url, guid=getattr(candidate, "guid", ""), title=candidate.title
            )
            if text and ing._looks_like_transcript(text):
                return text, "rss_transcript"
        except Exception:
            pass

    # 2. Transcript published on the episode page. Free.
    if candidate.page_url:
        try:
            text = fetch_page_transcript(candidate.page_url)
            if text:
                return text, "html_page"
        except Exception:
            pass
        try:
            text = ing._fetch_html_transcript(candidate.page_url)
            if text and ing._looks_like_transcript(text):
                return text, "html_page"
        except Exception:
            pass

    # 3. Podchaser, then Listen Notes. Both are quota/token gated, so they sit
    #    after the genuinely free sources but before paid transcription.
    try:
        text = ing._fetch_podchaser_transcript(
            podcast_name=candidate.podcast_name,
            episode_title=candidate.title,
            page_url=candidate.page_url or None,
        )
        if text and ing._looks_like_transcript(text):
            return text, "podchaser"
    except Exception:
        pass

    if candidate.episode_id.startswith("listennotes:") or not candidate.episode_id.startswith(("apple:", "rss:")):
        try:
            text = _listennotes_transcript_by_id(candidate.episode_id, output_root)
            if text and ing._looks_like_transcript(text):
                return text, "listennotes"
        except Exception:
            pass

    # 4. Speech-to-text. This is the only step that costs money, so it runs
    #    only after every free source has been tried and only when explicitly
    #    allowed by the caller.
    if allow_stt and candidate.audio_url:
        try:
            tmp = Path(output_root) / "_whisper_tmp"
            text = ing._transcribe_with_whisper(candidate.audio_url, tmp)
            if text:
                return text, "whisper"
        except Exception:
            pass
        try:
            text = ing._transcribe_with_deepgram(candidate.audio_url)
            if text:
                return text, "deepgram"
        except Exception:
            pass

    return "", "none"


def _listennotes_transcript_by_id(episode_id: str, output_root: Path) -> str:
    """Fetch a transcript straight from the episode id, respecting the budget."""
    import os

    import requests

    import podcast_ingest as ing

    key = (os.getenv("LISTENNOTES_API_KEY") or "").strip()
    if not key or not episode_id:
        return ""
    if not ing._listen_notes_budget_left(Path(output_root)):
        return ""
    try:
        ing._bump_listen_notes_usage(Path(output_root))
        resp = requests.get(
            f"{ing.LISTENNOTES_BASE}/episodes/{episode_id}",
            headers={"X-ListenAPI-Key": key},
            params={"show_transcript": 1},
            timeout=20,
        )
        resp.raise_for_status()
        return str(resp.json().get("transcript") or "").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# The processing pipeline
# ---------------------------------------------------------------------------

@dataclass
class ProcessedEpisode:
    """One episode after transcript verification."""

    candidate: Candidate
    relevance: Relevance
    excerpts: List[Excerpt] = field(default_factory=list)
    transcript_source: str = "none"
    transcript_words: int = 0

    @property
    def reportable(self) -> bool:
        return self.relevance.reportable and bool(self.excerpts)


def process_candidates(
    candidates: Sequence[Candidate],
    *,
    output_root: Path,
    profiles: Optional[Dict[str, TickerProfile]] = None,
    state: Optional[DiscoveryState] = None,
    store: Optional[ResearchStore] = None,
    allow_stt: bool = True,
    max_episodes: int = 25,
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> List[ProcessedEpisode]:
    """Transcribe, verify and excerpt each candidate.

    Failures are per-episode: a dead audio URL or an API timeout skips that
    episode and the run continues.
    """
    store = store or ResearchStore(Path(output_root))
    profiles = profiles or build_profiles({c.ticker for c in candidates})
    out: List[ProcessedEpisode] = []

    # Best candidates first so a capped run spends its budget well.
    ordered = sorted(candidates, key=lambda c: -c.score)[:max_episodes]

    for cand in ordered:
        prof = profiles.get(cand.ticker)
        names = [n for n in ([prof.company] if prof and prof.company else []) + (prof.aliases if prof else []) if n]
        people = [p.name for p in (prof.people if prof else [])]
        ambiguous = bool(prof and prof.ambiguous)

        def note(msg: str) -> None:
            if on_progress:
                on_progress(cand.uid, msg)

        try:
            # Reuse a transcript we already paid for. Keyed on dedupe_key, not
            # episode_id: Apple and RSS can each surface the same episode
            # under a different id, and which one wins the cross-backend
            # merge can vary run to run, so storage has to be addressed by
            # the identity that's stable across backends - otherwise a
            # transcript fetched (or manually supplied) under one backend's
            # id is invisible the next time the other backend wins.
            transcript = store.read_transcript(cand.dedupe_key)
            source = "cached" if transcript else ""
            if not transcript:
                note("fetching transcript")
                transcript, source = acquire_transcript(
                    cand, output_root=Path(output_root), allow_stt=allow_stt
                )

            if not transcript:
                note("no transcript available")
                if state:
                    state.mark(cand.dedupe_key, "no_transcript", title=cand.title)
                store.save(
                    episode_id=cand.dedupe_key,
                    metadata={
                        "episode_id": cand.episode_id,
                        "tickers": [cand.ticker],
                        "podcast_name": cand.podcast_name,
                        "title": cand.title,
                        "published": cand.published,
                        "transcript_source": "none",
                        "relevance": "Unknown",
                        "relevance_reason": "no transcript could be obtained",
                        "excerpt_count": 0,
                        "discovery_score": cand.score,
                        "matched_on": cand.matched,
                        "source": cand.source,
                    },
                )
                # Still surfaced (not skipped) so the manual-URL override in
                # the UI can offer it - a dead audio link or an unreadable
                # feed is exactly the case a publisher's own transcript page
                # rescues.
                out.append(
                    ProcessedEpisode(
                        candidate=cand,
                        relevance=Relevance(
                            "Unknown", 0.0, "no transcript could be obtained"
                        ),
                        transcript_source="none",
                        transcript_words=0,
                    )
                )
                continue

            note("verifying relevance from transcript")
            rel = classify_relevance(
                transcript,
                ticker=cand.ticker,
                company_names=names or [cand.ticker],
                people=people,
                title=cand.title,
                ambiguous_ticker=ambiguous,
            )

            excerpts: List[Excerpt] = []
            if rel.reportable:
                note(f"{rel.label} - extracting excerpts")
                excerpts = extract_excerpts(
                    transcript,
                    ticker=cand.ticker,
                    company_names=names or [cand.ticker],
                    people=people,
                    ambiguous_ticker=ambiguous,
                )
                # Never let anything but transcript text reach the report.
                excerpts = [e for e in excerpts if verify_verbatim(e, transcript)]
            else:
                note(f"{rel.label} - not reportable")

            store.save(
                episode_id=cand.dedupe_key,
                metadata={
                    "episode_id": cand.episode_id,
                    "tickers": [cand.ticker],
                    "podcast_name": cand.podcast_name,
                    "title": cand.title,
                    "published": cand.published,
                    "page_url": cand.page_url,
                    "listennotes_url": cand.listennotes_url,
                    "transcript_source": source,
                    "relevance": rel.label,
                    "relevance_reason": rel.reason,
                    "relevance_confidence": rel.confidence,
                    "people_present": rel.people_present,
                    "excerpt_count": len(excerpts),
                    "discovery_score": cand.score,
                    "matched_on": cand.matched,
                    "source": cand.source,
                },
                transcript=transcript,
                excerpts=excerpts,
            )
            if state:
                state.mark(
                    cand.dedupe_key,
                    "analyzed" if rel.reportable else "irrelevant",
                    relevance=rel.label,
                    title=cand.title,
                    excerpts=len(excerpts),
                )

            out.append(
                ProcessedEpisode(
                    candidate=cand,
                    relevance=rel,
                    excerpts=excerpts,
                    transcript_source=source,
                    transcript_words=len(transcript.split()),
                )
            )
        except Exception as exc:
            # One bad episode must not end the run.
            note(f"failed: {type(exc).__name__}: {exc}")
            if state:
                state.mark(cand.dedupe_key, "error", error=str(exc)[:200], title=cand.title)
            continue

    if state:
        state.save()
    return out


def to_excerpt_records(processed: Sequence[ProcessedEpisode]) -> Dict[str, List[dict]]:
    """Shape results like podcast_excerpts.py output so the existing PDF and
    insights code can consume discovery results unchanged."""
    out: Dict[str, List[dict]] = {}
    episodes: Dict[str, dict] = {}

    for pe in processed:
        if not pe.reportable:
            continue
        c = pe.candidate
        uid = f"discovery__{c.episode_id}"
        episodes[uid] = {
            "episode_id": c.episode_id,
            "podcast_id": "discovery",
            "title": c.title,
            "published": c.published,
            "episode_url": c.page_url or c.listennotes_url,
            "transcript_source": pe.transcript_source,
        }
        rows = out.setdefault(c.ticker, [])
        for ex in pe.excerpts:
            rows.append(
                {
                    "ticker": c.ticker,
                    "company_names": [],
                    "snippet": ex.text,
                    "podcast_id": "discovery",
                    "podcast_name": c.podcast_name,
                    "episode_id": c.episode_id,
                    "title": c.title,
                    "published": c.published,
                    "episode_url": c.page_url or c.listennotes_url,
                    "transcript_source": pe.transcript_source,
                    "evidence_confidence": "high",
                    "relevance_score": int(min(100, pe.relevance.confidence * 100)),
                    "relevance_reason": pe.relevance.reason,
                    # discovery extras the evidence-first PDF renders
                    "topics": ex.topics,
                    "topic_label": ex.topic_label,
                    "time_range": ex.time_range,
                    "speakers": ex.speakers,
                    "relevance_label": pe.relevance.label,
                    "words": ex.words,
                }
            )
    out["_episodes"] = episodes  # type: ignore[assignment]
    return out

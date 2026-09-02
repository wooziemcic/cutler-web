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

    # 1. Listen Notes episode endpoint - we already have a stable id, so this
    #    is one call and needs no search.
    try:
        text = _listennotes_transcript_by_id(candidate.episode_id, output_root)
        if text and ing._looks_like_transcript(text):
            return text, "listennotes"
    except Exception:
        pass

    # 2. Transcript published on the episode page.
    if candidate.page_url:
        try:
            text = ing._fetch_html_transcript(candidate.page_url)
            if text and ing._looks_like_transcript(text):
                return text, "html_page"
        except Exception:
            pass

    # 3. Podchaser.
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

    # 4. Speech-to-text - the expensive option, so it is last.
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
            # Reuse a transcript we already paid for.
            transcript = store.read_transcript(cand.episode_id)
            source = "cached" if transcript else ""
            if not transcript:
                note("fetching transcript")
                transcript, source = acquire_transcript(
                    cand, output_root=Path(output_root), allow_stt=allow_stt
                )

            if not transcript:
                note("no transcript available")
                if state:
                    state.mark(cand.uid, "no_transcript", title=cand.title)
                store.save(
                    episode_id=cand.episode_id,
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
                episode_id=cand.episode_id,
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
                    cand.uid,
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
                state.mark(cand.uid, "error", error=str(exc)[:200], title=cand.title)
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

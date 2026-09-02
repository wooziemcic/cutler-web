"""Ticker-driven podcast discovery.

The existing pipeline answers "what did our monitored feeds say about our
companies?". This module answers the wider question: "what podcast anywhere
discussed one of our companies, or a person who runs one, in the last week?"

It is deliberately the cheap end of the funnel. It only ever looks at episode
metadata (title, description, podcast name) and produces scored *candidates*.
Nothing here downloads audio, buys a transcript or calls an LLM - those
happen later and only for candidates that survive.

    entities -> Listen Notes search -> dedupe -> score -> candidates
                                                             |
                              (transcript + verification happen downstream)

Processed-state is persisted by stable episode id so a rerun does not
rediscover, re-transcribe or re-analyse the same episode.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

from podcast_entities import Entity, TickerProfile, build_profiles

LISTENNOTES_BASE = "https://listen-api.listennotes.com/api/v2"
DEFAULT_LOOKBACK_DAYS = 7

# Score a candidate must reach on metadata alone to be worth a transcript.
MIN_CANDIDATE_SCORE = 3.0


@dataclass
class Candidate:
    """One episode that might be relevant to one ticker."""

    episode_id: str  # stable Listen Notes id
    ticker: str
    podcast_name: str
    title: str
    description: str
    published: str  # ISO date
    audio_url: str = ""
    page_url: str = ""
    listennotes_url: str = ""
    score: float = 0.0
    matched: List[str] = field(default_factory=list)  # human-readable reasons
    source: str = "discovery"  # "discovery" | "monitored"

    @property
    def uid(self) -> str:
        return f"{self.ticker}::{self.episode_id}"

    def why(self) -> str:
        return "; ".join(self.matched) or "no explicit match"


# ---------------------------------------------------------------------------
# Processed-state
# ---------------------------------------------------------------------------

class DiscoveryState:
    """Remembers which (ticker, episode) pairs have already been handled.

    Keeps discovery idempotent: reruns skip episodes already transcribed or
    already judged irrelevant, so Listen Notes quota and transcription minutes
    are spent once.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: Dict[str, Any] = {"episodes": {}, "searches": {}}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.setdefault("episodes", {})
                    self.data.setdefault("searches", {})
                    self.data.update(
                        {
                            "episodes": loaded.get("episodes") or {},
                            "searches": loaded.get("searches") or {},
                        }
                    )
        except Exception:
            # A corrupt state file must not stop a run; worst case we redo work.
            self.data = {"episodes": {}, "searches": {}}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            tmp.replace(self.path)  # atomic; a killed run cannot truncate state
        except Exception:
            pass

    def status(self, uid: str) -> str:
        rec = (self.data.get("episodes") or {}).get(uid) or {}
        return str(rec.get("status") or "")

    def is_done(self, uid: str) -> bool:
        return self.status(uid) in {"analyzed", "irrelevant", "no_transcript"}

    def mark(self, uid: str, status: str, **extra: Any) -> None:
        eps = self.data.setdefault("episodes", {})
        rec = eps.setdefault(uid, {})
        rec["status"] = status
        rec["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rec.update(extra)

    def seen_search(self, key: str, ttl_hours: int = 20) -> bool:
        """True when this exact query ran recently enough to skip."""
        rec = (self.data.get("searches") or {}).get(key)
        if not rec:
            return False
        try:
            when = datetime.fromisoformat(str(rec.get("at")))
            return datetime.now(timezone.utc) - when < timedelta(hours=ttl_hours)
        except Exception:
            return False

    def mark_search(self, key: str) -> None:
        self.data.setdefault("searches", {})[key] = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")
        }


# ---------------------------------------------------------------------------
# Listen Notes search
# ---------------------------------------------------------------------------

def _api_key() -> str:
    return (os.getenv("LISTENNOTES_API_KEY") or "").strip()


def search_episodes(
    query: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    page_size: int = 10,
    timeout: int = 20,
    api_key: Optional[str] = None,
    diagnostics: Optional[List[str]] = None,
) -> List[dict]:
    """One Listen Notes episode search restricted to the lookback window.

    Returns raw result dicts. Network and quota failures return [] rather than
    raising - discovery is best-effort and must never abort a run - but they
    are recorded in `diagnostics` so the caller can tell "nothing was
    published" apart from "the API refused us".
    """
    def _note(msg: str) -> None:
        print(f"    [WARN] {msg}")
        if diagnostics is not None and msg not in diagnostics:
            diagnostics.append(msg)

    key = (api_key or _api_key()).strip()
    if not key:
        _note("LISTENNOTES_API_KEY is not set - discovery cannot search.")
        return []
    if not query.strip():
        return []

    since_ms = int(
        (datetime.now(timezone.utc) - timedelta(days=max(1, lookback_days))).timestamp()
        * 1000
    )
    params = {
        "q": query,
        "type": "episode",
        "only_in": "title,description",
        "language": "English",
        "safe_mode": 0,
        "published_after": since_ms,
        "sort_by_date": 1,
        "page_size": max(1, min(10, page_size)),
    }
    try:
        resp = requests.get(
            f"{LISTENNOTES_BASE}/search",
            headers={"X-ListenAPI-Key": key},
            params=params,
            timeout=timeout,
        )
        if resp.status_code == 429:
            _note(
                "Listen Notes returned 429 - the API key is rate limited or its "
                "monthly quota is used up. No episodes can be discovered until it resets."
            )
            return []
        if resp.status_code in (401, 403):
            _note(f"Listen Notes rejected the API key (HTTP {resp.status_code}).")
            return []
        resp.raise_for_status()
        return list(resp.json().get("results") or [])
    except Exception as exc:
        _note(f"Listen Notes search failed for {query!r}: {type(exc).__name__}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _word_boundary_hit(needle: str, haystack: str) -> bool:
    """Whole-token match so 'ON' does not fire inside 'monetary'."""
    if not needle or not haystack:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack, re.IGNORECASE) is not None


def score_candidate(
    profile: TickerProfile,
    title: str,
    description: str,
    podcast_name: str = "",
) -> tuple[float, List[str]]:
    """Score episode metadata against a ticker's entities.

    Weights come from the entity kind: a named executive outranks a company
    name, which outranks a bare symbol. Title matches count double because a
    company named in the title is usually the subject, not an aside.
    """
    title_l = title or ""
    desc_l = description or ""
    score = 0.0
    reasons: List[str] = []

    for ent in profile.entities():
        if ent.weight <= 0:  # ambiguous symbol: never scores on its own
            continue
        in_title = _word_boundary_hit(ent.name, title_l)
        in_desc = _word_boundary_hit(ent.name, desc_l)
        if not (in_title or in_desc):
            continue
        gain = ent.weight * (2.0 if in_title else 1.0)
        score += gain
        where = "title" if in_title else "description"
        label = f"{ent.kind}:{ent.name}"
        if ent.role:
            label += f" ({ent.role})"
        reasons.append(f"{label} in {where}")

    # A guest interview is the highest-value format, so reward the shape of an
    # episode that names a person we track in its title.
    if any(r.startswith("person:") and "in title" in r for r in reasons):
        score += 2.0
        reasons.append("likely executive interview")

    return score, reasons


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _result_to_candidate(res: dict, ticker: str, profile: TickerProfile) -> Optional[Candidate]:
    ep_id = str(res.get("id") or "").strip()
    if not ep_id:
        return None
    title = re.sub(r"<[^>]+>", " ", str(res.get("title_original") or res.get("title_highlighted") or ""))
    desc = re.sub(r"<[^>]+>", " ", str(res.get("description_original") or res.get("description_highlighted") or ""))
    title = re.sub(r"\s+", " ", title).strip()
    desc = re.sub(r"\s+", " ", desc).strip()

    pub_ms = res.get("pub_date_ms")
    try:
        published = datetime.fromtimestamp(int(pub_ms) / 1000, tz=timezone.utc).isoformat()
    except Exception:
        published = ""

    score, reasons = score_candidate(profile, title, desc, str(res.get("podcast_title_original") or ""))
    return Candidate(
        episode_id=ep_id,
        ticker=ticker,
        podcast_name=re.sub(r"<[^>]+>", " ", str(res.get("podcast_title_original") or "")).strip(),
        title=title,
        description=desc,
        published=published,
        audio_url=str(res.get("audio") or ""),
        page_url=str(res.get("link") or ""),
        listennotes_url=str(res.get("listennotes_url") or ""),
        score=score,
        matched=reasons,
    )


def discover_for_tickers(
    tickers: Sequence[str],
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    state: Optional[DiscoveryState] = None,
    max_terms_per_ticker: int = 3,
    max_calls: int = 60,
    min_score: float = MIN_CANDIDATE_SCORE,
    profiles: Optional[Dict[str, TickerProfile]] = None,
    skip_processed: bool = True,
    on_progress=None,
    diagnostics: Optional[List[str]] = None,
    budget_root: Optional[Path] = None,
) -> List[Candidate]:
    """Find recent episodes plausibly about the given tickers.

    Cheapest-first: metadata search, then dedupe, then score. Only candidates
    at or above `min_score` are returned, and `max_calls` caps total Listen
    Notes usage for the whole run.
    """
    profiles = profiles or build_profiles(tickers)
    by_uid: Dict[str, Candidate] = {}
    calls = 0
    cached_skips = 0

    # Share podcast_ingest's monthly Listen Notes counter so discovery and
    # transcript fetching draw down one budget rather than two.
    def _budget_ok() -> bool:
        if budget_root is None:
            return True
        try:
            import podcast_ingest as ing

            return bool(ing._listen_notes_budget_left(Path(budget_root)))
        except Exception:
            return True

    def _budget_spend() -> None:
        if budget_root is None:
            return
        try:
            import podcast_ingest as ing

            ing._bump_listen_notes_usage(Path(budget_root))
        except Exception:
            pass

    for sym in [str(t).strip().upper() for t in tickers if str(t).strip()]:
        profile = profiles.get(sym)
        if not profile:
            continue
        terms = profile.search_terms()[:max_terms_per_ticker]
        if not terms:
            # Ambiguous symbol with no company or people on file: searching it
            # would only return noise, so skip it rather than spend a call.
            if on_progress:
                on_progress(sym, "skipped (no distinctive search terms)", 0)
            continue

        found = 0
        for term in terms:
            if calls >= max_calls:
                break
            skey = f"{sym}|{term}|{lookback_days}"
            if state and state.seen_search(skey):
                cached_skips += 1
                continue
            if not _budget_ok():
                msg = "Listen Notes monthly budget reached (LISTENNOTES_MONTHLY_CAP); stopping discovery."
                if diagnostics is not None and msg not in diagnostics:
                    diagnostics.append(msg)
                break
            calls += 1
            _budget_spend()
            results = search_episodes(
                term, lookback_days=lookback_days, diagnostics=diagnostics
            )
            if state:
                state.mark_search(skey)
            time.sleep(0.25)  # be polite to the API

            for res in results:
                cand = _result_to_candidate(res, sym, profile)
                if not cand or cand.score < min_score:
                    continue
                if skip_processed and state and state.is_done(cand.uid):
                    continue
                prev = by_uid.get(cand.uid)
                if prev is None or cand.score > prev.score:
                    by_uid[cand.uid] = cand
                    found += 1

        if on_progress:
            on_progress(sym, f"{found} candidate(s) from {len(terms)} term(s)", found)

    if diagnostics is not None and calls >= max_calls:
        diagnostics.append(
            f"Stopped after {calls} searches (max_calls={max_calls}); some tickers were not searched."
        )
    # Without this, a fully cached rerun returns zero with no explanation -
    # indistinguishable from "nothing was published".
    if diagnostics is not None and cached_skips and not calls:
        diagnostics.append(
            f"All {cached_skips} search(es) were skipped because the same queries ran "
            "in the last 20 hours. Tick 'Re-check processed' to force a fresh search."
        )

    out = sorted(by_uid.values(), key=lambda c: (-c.score, c.published), reverse=False)
    return sorted(out, key=lambda c: -c.score)


def merge_with_monitored(
    discovered: Sequence[Candidate],
    monitored: Sequence[Candidate],
) -> List[Candidate]:
    """Combine discovery hits with the monitored-feed episodes, deduped.

    The same episode can arrive from both paths. Keep one record per
    (ticker, episode), preferring the monitored copy because that path already
    has the feed's own audio and page URLs.
    """
    merged: Dict[str, Candidate] = {}
    for cand in list(discovered) + list(monitored):
        key = cand.uid
        prev = merged.get(key)
        if prev is None:
            merged[key] = cand
            continue
        # Monitored wins on provenance; keep the better score and both reasons.
        winner = prev if prev.source == "monitored" else cand
        loser = cand if winner is prev else prev
        winner.score = max(prev.score, cand.score)
        for reason in loser.matched:
            if reason not in winner.matched:
                winner.matched.append(reason)
        if loser.source == "monitored":
            winner.source = "monitored"
        merged[key] = winner
    return sorted(merged.values(), key=lambda c: -c.score)


def candidates_to_json(cands: Iterable[Candidate]) -> List[dict]:
    return [asdict(c) for c in cands]


if __name__ == "__main__":  # manual smoke test
    import sys

    syms = sys.argv[1:] or ["QXO"]
    days = int(os.getenv("LOOKBACK_DAYS", "7"))
    print(f"Discovering for {syms} over {days} days...")
    for prof in build_profiles(syms).values():
        print(f"  {prof.describe()}")
    cands = discover_for_tickers(syms, lookback_days=days)
    print(f"\n{len(cands)} candidate(s):")
    for c in cands[:15]:
        print(f"  [{c.score:5.1f}] {c.published[:10]} {c.podcast_name} - {c.title[:70]}")
        print(f"          why: {c.why()}")

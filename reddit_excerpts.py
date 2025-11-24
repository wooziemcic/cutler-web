# reddit_excerpts.py
# ------------------------------------------------------------
# Cutler Capital — Reddit excerption (add-on lane)
#
# Requirements:
#   pip install praw python-dotenv
#   Set env:
#     REDDIT_CLIENT_ID=...
#     REDDIT_CLIENT_SECRET=...
#     REDDIT_USER_AGENT="CutlerIntelligence/1.0 by <your-handle>"
#
# Usage examples:
#   python reddit_excerpts.py --quarter "2025 Q3" --subs investing stocks ValueInvesting \
#       --days 7 --min-upvotes 25 --limit-per-sub 300 --build-pdf
#
# Output:
#   BSD/Excerpts/<Quarter>/Reddit/r_<sub>__<postid>/excerpts_clean.json
#   (optional) Excerpted_<post>.pdf in the same folder when --build-pdf is used
#
from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# --- Local imports (project root) ---
import importlib.util

HERE = Path(__file__).resolve().parent
def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if not spec or not spec.loader:
        raise ImportError(f"Cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

tickers_mod = _import("tickers", HERE / "tickers.py")
TICKERS_MAP: Dict[str, List[str]] = getattr(tickers_mod, "tickers", {})
excerpt_mod = _import("excerpt_check", HERE / "excerpt_check.py")
make_pdf = _import("make_pdf", HERE / "make_pdf.py")

# We will reuse narrative_score from excerpt_check.py to keep quality rules consistent
narrative_score = getattr(excerpt_mod, "narrative_score")

# --- Third-party (PRAW) ---
try:
    import praw  # Reddit official API wrapper
except Exception as e:
    raise SystemExit(
        "Missing dependency: praw\nInstall with: pip install praw\n"
        f"ImportError: {e}"
    )

# ---------- Config / constants ----------

CASHTAG_RE = re.compile(r'(?<![A-Za-z0-9_])\$[A-Z]{1,5}(?![A-Za-z])')
MULTISPACE_RE = re.compile(r'\s+')
URL_RE = re.compile(r'https?://\S+')
EMOJI_OR_SYMBOL_RE = re.compile(r'[\u2600-\u27BF\U0001F300-\U0001FAFF]+')

DEFAULT_SUBS = ["investing", "stocks", "ValueInvesting"]
DEFAULT_MIN_UPVOTES = 25
DEFAULT_MIN_WORDS = 80
DEFAULT_LIMIT_PER_SUB = 300  # max submissions pulled from 'new' per subreddit
DEFAULT_DAYS = 7

BASE = HERE / "BSD"
EX_DIR = BASE / "Excerpts"

# ---------- Utilities ----------

def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "file"

def _utc_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _split_paragraphs(text: str) -> List[str]:
    # Prefer double newlines as paragraph breaks; fall back to sentence-ish chunks
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs
    # Backup: split on sentence boundaries if no blank lines
    parts = re.split(r'(?<=[.!?])\s+', raw)
    out: List[str] = []
    cur: List[str] = []
    n = 0
    for s in parts:
        w = len(s.split())
        if n + w > 90 and cur:
            out.append(" ".join(cur).strip())
            cur = [s]
            n = w
        else:
            cur.append(s)
            n += w
    if cur:
        out.append(" ".join(cur).strip())
    return [p for p in out if p]

def _word_count(s: str) -> int:
    return len((s or "").split())

def _compile_alias_regexes(tmap: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    out: Dict[str, re.Pattern] = {}
    for tkr, aliases in tmap.items():
        alts: List[str] = []
        for kw in aliases or []:
            kw_esc = re.escape(kw)
            if re.match(r'^[A-Za-z0-9.&\-]+$', kw):
                alts.append(rf'(?<!\w){kw_esc}(?!\w)')
            else:
                alts.append(kw_esc)
        if not alts:
            # allow bare ticker as a weak signal (not preferred vs cashtag)
            alts.append(rf'(?<!\w){re.escape(tkr)}(?!\w)')
        out[tkr] = re.compile(r'(' + '|'.join(alts) + r')', re.IGNORECASE)
    return out

ALIAS_RX = _compile_alias_regexes(TICKERS_MAP)

def _detect_tickers(text: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Returns (tickers_found, hits_by_ticker[ticker] = [matched_strings...])
    Detection uses cashtags + alias patterns. We do not rely on plain $TICKER only.
    """
    hits: Dict[str, List[str]] = {}
    # Cashtags
    for m in CASHTAG_RE.finditer(text or ""):
        sym = m.group(0)[1:]  # strip $
        if sym in TICKERS_MAP or sym in ALIAS_RX:
            hits.setdefault(sym, []).append(m.group(0))

    # Aliases
    for tkr, rx in ALIAS_RX.items():
        for m in rx.finditer(text or ""):
            hits.setdefault(tkr, []).append(m.group(0))

    # Normalize keys to canonical tickers where possible
    tickers = sorted(set(hits.keys()))
    return tickers, hits

def _quality_ok(
    text: str,
    min_words: int = DEFAULT_MIN_WORDS,
    max_url_ratio: float = 0.25,
    max_emoji_ratio: float = 0.05,
) -> bool:
    words = _word_count(text)
    if words < min_words:
        return False
    s = text or ""
    url_chars = sum(len(m.group(0)) for m in URL_RE.finditer(s))
    emoji_chars = len(EMOJI_OR_SYMBOL_RE.findall(s))
    L = max(len(s), 1)
    if (url_chars / L) > max_url_ratio:
        return False
    if (emoji_chars / L) > max_emoji_ratio:
        return False
    return True

# ---------- Core extraction ----------

@dataclass
class RedditConfig:
    quarter: str
    subreddits: List[str]
    days: int = DEFAULT_DAYS
    min_upvotes: int = DEFAULT_MIN_UPVOTES
    min_words: int = DEFAULT_MIN_WORDS
    limit_per_sub: int = DEFAULT_LIMIT_PER_SUB
    flair_allow: Optional[List[str]] = None  # e.g., ["DD", "Analysis", "News"]
    build_pdf: bool = False

def _init_reddit_client() -> praw.Reddit:
    client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
    user_agent = os.getenv("REDDIT_USER_AGENT", "").strip()
    if not (client_id and client_secret and user_agent):
        raise SystemExit(
            "Reddit API credentials missing.\n"
            "Set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT in your environment (or a .env)."
        )
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        ratelimit_seconds=2,
    )

def _target_dir_for_post(quarter: str, subreddit: str, post_id: str, title: str) -> Path:
    root = EX_DIR / quarter / "Reddit" / f"r_{_safe(subreddit)}__{_safe(post_id)}"
    root.mkdir(parents=True, exist_ok=True)
    # Drop a light metadata file for quick inspection
    meta = root / "post_meta.json"
    if not meta.exists():
        with open(meta, "w", encoding="utf-8") as f:
            json.dump({"created_at": _now_utc_iso(), "title": title}, f, indent=2)
    # Ensure tickers copy for make_pdf import (optional)
    tcopy = root / "tickers.py"
    if not tcopy.exists():
        try:
            (HERE / "tickers.py").exists() and Path(HERE / "tickers.py").replace(tcopy)
        except Exception:
            pass
    return root

def _write_excerpts_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _build_pdf_for_post(out_dir: Path, post_title: str, letter_date_iso: str) -> Optional[Path]:
    src_json = out_dir / "excerpts_clean.json"
    if not src_json.exists():
        return None
    # Name the PDF with the post title
    out_pdf = out_dir / f"Excerpted_{_safe(post_title)[:80]}.pdf"
    try:
        make_pdf.build_pdf(
            excerpts_json_path=str(src_json),
            output_pdf_path=str(out_pdf),
            report_title=f"Reddit Excerpts – {post_title[:90]}",
            source_pdf_name=f"{post_title[:120]}.reddit",
            format_style="legacy",
            letter_date=letter_date_iso.split("T")[0] if letter_date_iso else None,
        )
        return out_pdf if out_pdf.exists() else None
    except Exception:
        return None

def _paragraph_items_from_post_body(
    body: str,
    created_utc_iso: str,
    subreddit: str,
    url: str,
    author: str,
    ups: int,
    flair_text: Optional[str],
    min_words: int,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for para in _split_paragraphs(body or ""):
        if not _quality_ok(para, min_words=min_words):
            continue
        keep, score, reasons = narrative_score(para)
        if not keep:
            # Reddit text is noisier; allow borderline paragraphs with score==1 if long enough
            if not (score == 1 and _word_count(para) >= max(120, min_words + 20)):
                continue
        tickers, hits_by = _detect_tickers(para)
        if not tickers:
            continue
        # Build one record per TICKER (same paragraph may map to multiple)
        for tkr in sorted(set(tickers)):
            hits = hits_by.get(tkr, [])
            items.append(
                {
                    "ticker": tkr,
                    "text": para.strip(),
                    "pages": [],  # web source has no pages
                    "hits": hits,
                    "narrative_score": score,
                    "narrative_reasons": reasons,
                    "source_name": "Reddit",
                    "source_meta": {
                        "url": url,
                        "author": author or "",
                        "subreddit_or_section": f"r/{subreddit}",
                        "post_datetime": created_utc_iso,
                        "upvotes": int(ups or 0),
                        "flair": flair_text or "",
                    },
                }
            )
    return items

def run_reddit_excerpts(cfg: RedditConfig) -> List[Dict[str, Any]]:
    reddit = _init_reddit_client()
    cutoff = datetime.now(timezone.utc) - timedelta(days=cfg.days)
    results: List[Dict[str, Any]] = []

    for sub in cfg.subreddits:
        sr = reddit.subreddit(sub)
        pulled = 0
        for submission in sr.new(limit=cfg.limit_per_sub):
            pulled += 1
            created_ts = float(getattr(submission, "created_utc", time.time()))
            created_dt = datetime.fromtimestamp(created_ts, tz=timezone.utc)
            if created_dt < cutoff:
                # new() returns in reverse-chron order; once we hit cutoff we can break
                break

            ups = int(getattr(submission, "score", 0))
            if ups < cfg.min_upvotes:
                continue

            flair_text = getattr(submission, "link_flair_text", None)
            if cfg.flair_allow:
                if not flair_text or flair_text not in cfg.flair_allow:
                    continue

            title = (submission.title or "").strip()
            selftext = (submission.selftext or "").strip()
            url = f"https://www.reddit.com{submission.permalink}" if getattr(submission, "permalink", "") else (submission.url or "")

            # Basic fast pre-filter: only proceed if title or body hints at tickers
            title_tickers, _ = _detect_tickers(title)
            body_tickers, _ = _detect_tickers(selftext)
            if not (title_tickers or body_tickers):
                # Allow some alias-only cases that may not use cashtags—try a weaker heuristic on title
                weak_alias_hit = any(rx.search(title) for rx in ALIAS_RX.values())
                if not weak_alias_hit:
                    continue

            # Paragraphize & score from body only (narrative lives there)
            items = _paragraph_items_from_post_body(
                body=selftext,
                created_utc_iso=_utc_from_timestamp(created_ts),
                subreddit=sub,
                url=url,
                author=str(getattr(submission, "author", "") or ""),
                ups=ups,
                flair_text=flair_text,
                min_words=cfg.min_words,
            )
            if not items:
                continue

            # Group into the flat dict per ticker that make_pdf accepts
            by_ticker: Dict[str, List[Dict[str, Any]]] = {}
            for it in items:
                by_ticker.setdefault(it["ticker"], []).append(
                    {
                        "text": it["text"],
                        "pages": it["pages"],
                        "hits": it["hits"],
                        "narrative_score": it["narrative_score"],
                        "narrative_reasons": it["narrative_reasons"],
                        # keep metadata on each item to preserve provenance
                        "source_name": it["source_name"],
                        "source_meta": it["source_meta"],
                    }
                )

            # Write per-post excerpts_clean.json
            out_dir = _target_dir_for_post(cfg.quarter, sub, submission.id, title)
            out_json = out_dir / "excerpts_clean.json"
            _write_excerpts_json(out_json, by_ticker)

            if cfg.build_pdf:
                _build_pdf_for_post(out_dir, post_title=title or f"r/{sub} {submission.id}", letter_date_iso=_utc_from_timestamp(created_ts))

            results.append(
                {
                    "source": "Reddit",
                    "subreddit": sub,
                    "id": submission.id,
                    "url": url,
                    "title": title,
                    "created_utc": _utc_from_timestamp(created_ts),
                    "excerpts_json": str(out_json),
                    "tickers": sorted(list(by_ticker.keys())),
                    "upvotes": ups,
                    "flair": flair_text or "",
                }
            )

    return results

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Cutler Reddit excerption (add-on lane)")
    ap.add_argument("--quarter", required=True, help='Quarter label, e.g., "2025 Q3"')
    ap.add_argument("--subs", nargs="*", default=DEFAULT_SUBS, help="Subreddits to scan")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Lookback window in days")
    ap.add_argument("--min-upvotes", type=int, default=DEFAULT_MIN_UPVOTES, help="Minimum upvotes filter")
    ap.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS, help="Minimum words per kept paragraph")
    ap.add_argument("--limit-per-sub", type=int, default=DEFAULT_LIMIT_PER_SUB, help="Max posts to pull from 'new' per subreddit")
    ap.add_argument("--flair-allow", nargs="*", default=None, help='Optional flair allowlist, e.g., --flair-allow DD Analysis News')
    ap.add_argument("--build-pdf", action="store_true", help="Also build a per-post PDF using make_pdf.py")
    args = ap.parse_args()

    cfg = RedditConfig(
        quarter=args.quarter,
        subreddits=args.subs,
        days=args.days,
        min_upvotes=args.min_upvotes,
        min_words=args.min_words,
        limit_per_sub=args.limit_per_sub,
        flair_allow=args.flair_allow,
        build_pdf=args.build_pdf,
    )

    results = run_reddit_excerpts(cfg)
    # Emit a tiny run summary (you can later write a source manifest if desired)
    summary = {
        "quarter": cfg.quarter,
        "created_at": _now_utc_iso(),
        "items": results,
    }
    out_manifest = EX_DIR / cfg.quarter / "Reddit" / f"manifest_reddit_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Reddit run complete. Items: {len(results)}")
    print(f"Wrote manifest: {out_manifest}")

if __name__ == "__main__":
    main()

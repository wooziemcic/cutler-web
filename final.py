"""
Cutler Capital — Hedge Fund Letter Scraper
------------------------------------------
Internal Cutler Capital tool to scrape, excerpt, and compile hedge-fund letters
by fund family and quarter. Uses an external hedge-fund letter database as the
data source; all branding in the UI is Cutler-only.
"""
from __future__ import annotations

# Windows Playwright policy fix
import asyncio, platform
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass
import shutil
import re
import os
import sys
import shutil
import traceback
import json
import hashlib
import openai
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import re as _re
import html as html_lib
import streamlit as st
import requests
try:
    import sa_analysis_api as sa_api
    from sa_analysis_api import AnalysisArticle
except BaseException:
    sa_api = None
    AnalysisArticle = None
from fund_families_biglist import BIG_LIST_RAW
from podcasts_config import PODCASTS
from tickers import tickers
import math
import subprocess
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# pypdf compat
try:
    from pypdf import PdfMerger
except Exception:
    from pypdf import PdfWriter, PdfReader
    class PdfMerger:
        def __init__(self): self._w = PdfWriter()
        def append(self, p: str):
            r = PdfReader(p)
            for pg in r.pages: self._w.add_page(pg)
        def write(self, out: str):
            with open(out, 'wb') as f: self._w.write(f)
        def close(self): pass
from pypdf import PdfReader as _PdfReader, PdfWriter as _PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import sa_news_ai as sa_news
import seekingalpha_excerpts as sa_scraper
import sys

def ensure_playwright_chromium_installed(show_messages: bool = True) -> bool:
    """
    Ensure Chromium exists in the default Playwright cache directory.
    Streamlit Cloud sometimes wipes this on container rebuilds.
    Returns True if Chromium is installed and ready, False if installation failed.
    """

    # The default Python-playwright installation path:
    chromium_dir = Path.home() / ".cache/ms-playwright"
    found = list(chromium_dir.glob("chromium-*/*/chrome-linux/chrome"))

    if found:
        return True  # Chromium already present

    # Not found — try installing now
    if show_messages:
        st.warning("Chromium not found. Installing Playwright browser… (this may take ~20–40 seconds)")

    try:
        # Run: python -m playwright install chromium
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            if show_messages:
                st.success("Playwright Chromium installed successfully.")
            return True
        else:
            if show_messages:
                st.error("Playwright Chromium installation failed.")
                st.code(result.stderr)
            return False

    except Exception as e:
        if show_messages:
            st.error("Unexpected error during Chromium installation.")
            st.code(repr(e))
        return False


# ---------------------- compatibility helpers (Batch 8) ----------------------
# These wrappers are intentionally tiny and only exist to support the Batch 8
# code path without altering any existing Fund Families / other tab logic.

def _is_probable_ticker(s: str) -> bool:
    """
    Conservative check for equity tickers.
    Allows 1–6 uppercase letters (e.g., AAPL, MSFT, ABBV).
    """
    if not s:
        return False
    s = s.strip()
    if not s.isupper():
        return False
    if not (1 <= len(s) <= 6):
        return False
    return s.isalpha()


def _ensure_chromium_ready() -> bool:
    """Backwards-compatible alias used by the Batch 8 'Latest' runner."""
    return ensure_playwright_chromium_installed()


def _downloads_dir() -> Path:
    """Return a stable download directory for Batch 8 runs."""
    d = DL_DIR / "_latest"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _already_completed(brand: str, quarter: str) -> bool:
    """Check marker file for Batch 8 completion."""
    try:
        return _brand_progress_path(BATCH8_NAME, quarter, brand).exists()
    except Exception:
        return False


def _mark_completed(brand: str, quarter: str) -> None:
    """Create marker file for Batch 8 completion."""
    try:
        p = _brand_progress_path(BATCH8_NAME, quarter, brand)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("done", encoding="utf-8")
    except Exception:
        pass


def _excerpt_pdf(
    pdf_path: Path,
    *,
    brand: str,
    quarter: str,
    use_first_word: bool,
    source_pdf_name: Optional[str] = None,
    letter_date: Optional[str] = None,
    source_url: Optional[str] = None,
) -> Optional[Path]:
    """Batch 8 helper: run excerption + build excerpt PDF, returning the excerpt output dir.

    This is a very thin wrapper around the existing Fund Families pipeline
    (run_excerpt_and_build) to avoid any behavior changes elsewhere.
    """
    try:
        # Keep the same folder convention used by the standard batch runner.
        out_dir = EX_DIR / quarter / _safe(brand) / _safe(pdf_path.stem)
        built = run_excerpt_and_build(
            pdf_path,
            out_dir,
            source_pdf_name=source_pdf_name or pdf_path.name,
            letter_date=letter_date,
            source_url=source_url,
        )
        return out_dir if built else None
    except Exception:
        return None

def _brand_progress_path(batch_name: str, quarter: str, brand: str) -> Path:
    """
    Return a small marker file path that indicates we have fully processed
    this fund family (brand) for a given batch + quarter in this container.
    Used to resume long runs without re-doing earlier brands.
    """
    return MAN_DIR / "_progress" / _safe(batch_name) / quarter / f"{_safe(brand)}.done"


# local imports
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

substack_excerpts = _import("substack_excerpts", HERE / "substack_excerpts.py")

excerpt_check = _import("excerpt_check", HERE / "excerpt_check.py")
make_pdf = _import("make_pdf", HERE / "make_pdf.py")

# Seeking Alpha news + AI digest
try:
    sa_news_ai = _import("sa_news_ai", HERE / "sa_news_ai.py")
except Exception:
    sa_news_ai = None  # SA integration is optional

# Ticker dictionary (we reuse for the SA dropdown)
try:
    tickers_mod = _import("tickers", HERE / "tickers.py")
except Exception:
    tickers_mod = None


# paths (still stored under BSD/ on disk; UI is Cutler-branded only)
BASE = HERE / "BSD"
DL_DIR = BASE / "Downloads"
EX_DIR = BASE / "Excerpts"
CP_DIR = BASE / "Compiled"
MAN_DIR = BASE / "Manifests"   # run manifests (Document Checker + incremental)
DELTA_DIR = BASE / "Delta"     # delta PDFs + JSONs (Document Checker)
for d in (DL_DIR, EX_DIR, CP_DIR, MAN_DIR, DELTA_DIR):
    d.mkdir(parents=True, exist_ok=True)

ai_insights = _import("ai_insights", HERE / "ai_insights.py")

def _clean_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\[.*?\]", "", s)        # drop bracketed notes
    s = re.sub(r"\s+", " ", s)           # collapse spaces
    return s.strip("-•· ")

def _parse_big_list(raw: str) -> List[str]:
    seen = set(); out: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        name = _clean_name(line)
        key = name.lower()
        if key and key not in seen:
            seen.add(key); out.append(name)
    return out

def _chunk_round_robin(items: List[str], k: int) -> List[List[str]]:
    buckets = [[] for _ in range(k)]
    for i, it in enumerate(items): buckets[i % k].append(it)
    return buckets

ALL_FUND_NAMES = _parse_big_list(BIG_LIST_RAW)
BATCH_COUNT = 7
_batches = _chunk_round_robin(ALL_FUND_NAMES, BATCH_COUNT)
RUNNABLE_BATCHES: Dict[str, List[str]] = {f"Batch {i+1}": b for i, b in enumerate(_batches)}
BATCH8_NAME = "Batch 8 — Latest"  # dynamic weekly/latest mode


# External data source URL (kept internal; not shown in UI)
BSD_URL = "https://www.buysidedigest.com/hedge-fund-database/"
FILTERS = {
    "fund": "#md-fund-letter-table-fund-search",
    "quarter": "#md-fund-letter-table-select",
    "search_btn": "input.md-search-btn",
}
TABLE_ROW = "table tbody tr"
COLMAP = {"quarter": 1, "letter_date": 2, "fund_name": 3}

@dataclass
class Hit:
    quarter: str
    letter_date: str
    fund_name: str
    fund_href: str

_DEF_WORD_RE = re.compile(r"^[A-Za-z0-9'&.-]+")

def _first_word(name: str) -> str:
    m = _DEF_WORD_RE.search(name)
    return m.group(0) if m else (name.split()[0] if name.split() else name)


def _clear_session_keys(*, exact=None, prefixes=None) -> None:
    """
    Delete selected st.session_state keys without wiping unrelated tab state.
    - exact: list of exact keys to remove
    - prefixes: list of prefixes; any key starting with one of these is removed
    """
    import streamlit as st

    exact = list(exact or [])
    prefixes = list(prefixes or [])
    for k in list(st.session_state.keys()):
        if k in exact or any(k.startswith(p) for p in prefixes):
            try:
                del st.session_state[k]
            except Exception:
                pass


# ---------------------- Run All orchestration (persistent) ----------------------

_RUN_ALL_DIR = MAN_DIR / "_run_all"
_RUN_ALL_STATE_PATH = _RUN_ALL_DIR / "run_all_state.json"

def _load_run_all_state() -> dict:
    try:
        if _RUN_ALL_STATE_PATH.exists():
            return json.loads(_RUN_ALL_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _save_run_all_state(state: dict) -> None:
    try:
        _RUN_ALL_DIR.mkdir(parents=True, exist_ok=True)
        _RUN_ALL_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass

def _clear_run_all_state() -> None:
    try:
        if _RUN_ALL_STATE_PATH.exists():
            _RUN_ALL_STATE_PATH.unlink()
    except Exception:
        pass

def _make_batches(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i+size] for i in range(0, len(items), size)]

def _sa_article_row_basic(a) -> dict:
    """Normalize a Seeking Alpha list row (dict or object) into a simple dict.

    This helper is intentionally defensive because the RapidAPI payload shape can vary
    (list endpoint vs cached/serialized dicts). It is used by both the single-ticker
    Seeking Alpha tab and Run All. Keep it lightweight.
    """
    if isinstance(a, dict):
        return {
            "id": str(a.get("id") or a.get("article_id") or ""),
            "title": a.get("title") or "",
            "url": a.get("url") or a.get("link") or "",
            # publish timestamps can show up as published_at / published / publishOn / date
            "published_at": a.get("published_at") or a.get("published") or a.get("publishOn") or a.get("date") or "",
            # Prefer display author name; fall back to slug/alt keys when name isn't present
            "author": (
                a.get("author")
                or a.get("author_name")
                or a.get("authorName")
                or a.get("author_slug")
                or a.get("authorSlug")
                or ""
            ),
        }

    # object-like (dataclass or similar)
    return {
        "id": str(getattr(a, "id", "") or ""),
        "title": str(getattr(a, "title", "") or ""),
        "url": str(getattr(a, "url", "") or ""),
        "published_at": str(
            getattr(a, "published_at", "")
            or getattr(a, "published", "")
            or getattr(a, "publishOn", "")
            or getattr(a, "date", "")
            or ""
        ),
        "author": str(
            getattr(a, "author", "")
            or getattr(a, "author_name", "")
            or getattr(a, "authorName", "")
            or getattr(a, "author_slug", "")
            or getattr(a, "authorSlug", "")
            or ""
        ),
    }

def _extract_sa_details_fields(details: dict) -> tuple[str, str, str, str]:
    # returns body_raw, author, title, published_at
    base = details or {}
    body = (
        base.get("body_clean")
        or base.get("body_html")
        or base.get("content")
        or base.get("body")
        or base.get("html")
        or base.get("text")
        or ""
    )
    author = base.get("author") or base.get("author_name") or base.get("authorName") or ""
    title = base.get("title") or ""
    published = base.get("published_at") or base.get("published") or base.get("date") or ""
    return str(body or ""), str(author or ""), str(title or ""), str(published or "")

def _build_sa_compiled_pdf_for_universe(*, universe: list[str], max_articles: int, model: str) -> Path:
    # Builds one compiled PDF across the full universe in 10-ticker batches (to reduce API bursts).
    import tempfile
    import json as _json
    from zoneinfo import ZoneInfo

    cache = st.session_state.get("sa_cache", {})
    combined: dict[str, list[dict]] = {}

    # Reuse export caps if present
    max_paras_per_ticker = int(st.session_state.get("sa_pdf_max_paras_per_ticker", 5000))
    max_paras_per_article = int(st.session_state.get("sa_pdf_max_paras_per_article", 500))

    batches = _make_batches(list(universe), 10)
    prog = st.progress(0.0)
    status = st.empty()

    total = max(1, len(universe))
    processed = 0

    for b in batches:
        for sym in b:
            status.info(f"Run All: Seeking Alpha — fetching {sym} ({processed+1}/{total}) …")
            try:
                raw_list = sa_api.fetch_analysis_list(sym, size=max_articles) or []
                # Enrich with author credibility signals (followers / rating / etc.)
                try:
                    if hasattr(sa_api, "enrich_articles_with_author_metrics"):
                        raw_list = sa_api.enrich_articles_with_author_metrics(raw_list)
                except Exception:
                    pass
                _sa_metrics_by_id = {}
                try:
                    for _art in raw_list or []:
                        _aid = str(getattr(_art, "id", "") or "")
                        if not _aid:
                            continue
                        _sa_metrics_by_id[_aid] = {
                            "followers": getattr(_art, "author_followers", None),
                            "rating": getattr(_art, "author_rating", None),
                            "articles": getattr(_art, "author_articles_count", None),
                        }
                except Exception:
                    _sa_metrics_by_id = {}
                # Enrich with author credibility signals (followers / rating / etc.)
                try:
                    if hasattr(sa_api, "enrich_articles_with_author_metrics"):
                        raw_list = sa_api.enrich_articles_with_author_metrics(raw_list)
                except Exception:
                    pass
                _sa_metrics_by_id = {}
                try:
                    for _art in raw_list or []:
                        _aid = str(getattr(_art, "id", "") or "")
                        if not _aid:
                            continue
                        _sa_metrics_by_id[_aid] = {
                            "followers": getattr(_art, "author_followers", None),
                            "rating": getattr(_art, "author_rating", None),
                            "articles": getattr(_art, "author_articles_count", None),
                        }
                except Exception:
                    _sa_metrics_by_id = {}
                rows = [_sa_article_row_basic(x) for x in raw_list]
                rows = [r for r in rows if r.get("id")]

                for r in rows:
                    aid = r.get("id") or ""
                    if not aid:
                        continue
                    details = {}
                    try:
                        details = sa_api.fetch_analysis_details(aid) or {}
                    except Exception:
                        details = {}
                    body_raw, author_d, title_d, pub_d = _extract_sa_details_fields(details)

                    if "<" in (body_raw or "") and ">" in (body_raw or ""):
                        r["body_clean"] = clean_sa_html(body_raw)
                    else:
                        r["body_clean"] = (body_raw or "").strip()

                    if author_d and not r.get("author"):
                        r["author"] = author_d
                    if title_d and not r.get("title"):
                        r["title"] = title_d
                    if pub_d and not r.get("published_at"):
                        r["published_at"] = pub_d
                    if isinstance(details, dict) and details.get("url") and not r.get("url"):
                        r["url"] = details.get("url")

                items: list[dict] = []
                for a in rows:
                    body = (a.get("body_clean") or "").strip()
                    if not body:
                        continue

                    title = (a.get("title") or "").strip()
                    author = (a.get("author") or "").strip()
                    url = (a.get("url") or "").strip()
                    published = (a.get("published_at") or "").strip()

                    header_parts = []
                    if title:
                        header_parts.append(title)
                    if author:
                        header_parts.append(f"— {author}")
                    header = " ".join(header_parts).strip()

                    meta_lines = []
                    if published:
                        meta_lines.append(f"Date: {published[:10]}")
                    if url:
                        meta_lines.append(f"Source: {url}")
                    # Credibility line (best-effort)
                    try:
                        _m = _sa_metrics_by_id.get(str(a.get("id") or ""), {})
                    except Exception:
                        _m = {}
                    cred_parts = []
                    af = _m.get("followers")
                    ar = _m.get("rating")
                    ac = _m.get("articles")
                    try:
                        if isinstance(af, (int, float)):
                            cred_parts.append(f"{int(af):,} followers")
                    except Exception:
                        pass
                    try:
                        if isinstance(ar, (int, float)):
                            cred_parts.append(f"{float(ar):.1f} rating")
                    except Exception:
                        pass
                    try:
                        if isinstance(ac, (int, float)):
                            cred_parts.append(f"{int(ac):,} articles")
                    except Exception:
                        pass
                    if cred_parts:
                        meta_lines.append("Credibility: " + " | ".join(cred_parts))
                    if meta_lines:
                        header = (header + "\n" if header else "") + "\n".join(meta_lines)

                    if header:
                        items.append({"text": header, "pages": [], "is_header": True})

                    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
                    kept = 0
                    for p in paras:
                        if kept >= max_paras_per_article:
                            break
                        if len(p) < 60:
                            continue
                        items.append({"text": p, "pages": []})
                        kept += 1

                if items:
                    trimmed: list[dict] = []
                    body_count = 0
                    for it in items:
                        txt = (it.get("text") or "")
                        if it.get("is_header"):
                            trimmed.append(it)
                            continue
                        if body_count >= max_paras_per_ticker:
                            break
                        trimmed.append(it)
                        body_count += 1
                    combined[sym] = trimmed
            except Exception:
                pass

            processed += 1
            prog.progress(processed / total)

    if not combined:
        raise RuntimeError("No Seeking Alpha articles were available to export for the current universe/config.")

    now_et = datetime.now(ZoneInfo("America/New_York"))
    pdf_name = f"{now_et.strftime('%m.%d.%y')} Seeking Alpha ALL.pdf"
    out_path = CP_DIR / pdf_name

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        excerpts_path = td_path / "sa_excerpts.json"
        excerpts_path.write_text(_json.dumps(combined, indent=2), encoding="utf-8")
        out_pdf = td_path / "sa_compiled.pdf"
        status.info("Run All: Seeking Alpha — rendering compiled PDF…")
        make_pdf.build_pdf(
            excerpts_json_path=str(excerpts_path),
            output_pdf_path=str(out_pdf),
            report_title="Seeking Alpha Analysis",
            source_pdf_name=pdf_name,
            format_style="compact",
            ai_score=True,
            ai_model="heuristic",
            include_index=True,
            index_label="Index — Hit Tickers",
        )
        out_path.write_bytes(out_pdf.read_bytes())

    prog.empty()
    status.empty()
    return out_path




def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "file"


def _now_et() -> datetime:
    """Return current time in America/New_York (handles EST/EDT)."""
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now()


def _build_text_pdf(
    *,
    output_path: Path,
    title: str,
    subtitle: str | None = None,
    sections: list[tuple[str, str]] | None = None,
) -> Path:
    """Build a simple, clean, text-first PDF (used for SA/Podcast downloads)."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = getSampleStyleSheet()
    Title = ParagraphStyle(
        "DLTitle",
        parent=base["Title"],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4b2142"),
        spaceAfter=6,
    )
    Sub = ParagraphStyle(
        "DLSub",
        parent=base["Normal"],
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#6b4f7a"),
        spaceAfter=12,
    )
    H = ParagraphStyle(
        "DLH",
        parent=base["Heading2"],
        fontSize=12,
        leading=15,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#111827"),
        spaceBefore=10,
        spaceAfter=6,
    )
    Body = ParagraphStyle(
        "DLBody",
        parent=base["BodyText"],
        fontSize=10,
        leading=13,
        alignment=TA_LEFT,
        spaceAfter=8,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.75 * 72,
        rightMargin=0.75 * 72,
        topMargin=0.75 * 72,
        bottomMargin=0.75 * 72,
        title=title,
    )

    story: list = []
    story.append(Paragraph(title, Title))
    if subtitle:
        story.append(Paragraph(subtitle, Sub))
    else:
        story.append(Spacer(1, 8))

    if sections:
        for i, (h, b) in enumerate(sections):
            if i and i % 6 == 0:
                story.append(PageBreak())
            if h:
                story.append(Paragraph(h, H))
            if b:
                # Convert newlines to <br/> for ReportLab Paragraph
                safe_b = b.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                safe_b = safe_b.replace("\n", "<br/>")
                story.append(Paragraph(safe_b, Body))
    doc.build(story)
    return output_path




def _build_podcast_all_pdf(
    *,
    excerpts_path: Path,
    insights_path: Path,
    output_path: Path,
    days_back: int | None = None,
    group_label: str | None = None,
) -> Path:
    """
    Build a single 'Podcast ALL' PDF from existing podcast pipeline outputs.

    This is intentionally lightweight and reuses the existing text-PDF renderer
    used in the Podcasts tab (_build_text_pdf). It does NOT change the podcast
    pipeline itself (ingest/excerpts/insights).
    """
    # Load excerpts (for quick mention counts)
    excerpts: dict = {}
    try:
        if excerpts_path and Path(excerpts_path).exists():
            excerpts = json.loads(Path(excerpts_path).read_text(encoding="utf-8"))
    except Exception:
        excerpts = {}

    # Load insights (company stance summaries)
    insights: list[dict] = []
    try:
        if insights_path and Path(insights_path).exists():
            data = json.loads(Path(insights_path).read_text(encoding="utf-8"))
            if isinstance(data, list):
                insights = [d for d in data if isinstance(d, dict)]
    except Exception:
        insights = []

    now_et = _now_et()

    # Mention counts from excerpts (exclude _episodes)
    mention_counts: list[tuple[str, int]] = []
    if isinstance(excerpts, dict):
        for k, v in excerpts.items():
            if not isinstance(k, str) or k.startswith("_"):
                continue
            if isinstance(v, list):
                mention_counts.append((k, len(v)))
    mention_counts.sort(key=lambda x: x[1], reverse=True)

    # Keep it skimmable: top N companies
    top_symbols = [sym for sym, cnt in mention_counts if cnt > 0][:25]
    if not top_symbols:
        # Fallback: use insights tickers (excluding not_mentioned)
        top_symbols = [
            d.get("ticker") for d in insights
            if isinstance(d.get("ticker"), str) and d.get("stance") != "not_mentioned"
        ][:25]

    # Build sections
    sections: list[tuple[str, str]] = []

    # Overview
    window_line = f"Lookback: last {days_back} days" if days_back else "Lookback: recent window"
    if group_label:
        window_line += f"\nPodcasts: {group_label}"
    sections.append(("Podcast Intelligence — Overview", window_line))

    # Summary table-ish block
    if mention_counts:
        top_lines = []
        for sym, cnt in mention_counts[:20]:
            top_lines.append(f"{sym}: {cnt} mention(s)")
        sections.append(("Most-mentioned tickers (top 20)", "\n".join(top_lines)))

    # Detailed stance summaries
    insights_by_ticker = {d.get("ticker"): d for d in insights if isinstance(d.get("ticker"), str)}

    for sym in top_symbols:
        d = insights_by_ticker.get(sym, {})
        stance = d.get("stance", "unknown")
        conf = d.get("stance_confidence", 0.0)
        overall = d.get("overall_summary", "") or ""
        # Supporting points and risks (keep short)
        sp = d.get("supporting_points") or []
        rk = d.get("risks_or_headwinds") or []

        body_parts: list[str] = []
        body_parts.append(f"Stance: {stance} (confidence: {conf})")
        if overall:
            body_parts.append("")
            body_parts.append(overall.strip())

        if isinstance(sp, list) and sp:
            body_parts.append("")
            body_parts.append("Supporting points:")
            for x in sp[:5]:
                if isinstance(x, str) and x.strip():
                    body_parts.append(f"- {x.strip()}")

        if isinstance(rk, list) and rk:
            body_parts.append("")
            body_parts.append("Risks / headwinds:")
            for x in rk[:5]:
                if isinstance(x, str) and x.strip():
                    body_parts.append(f"- {x.strip()}")

        sections.append((f"{sym} — Podcast stance", "\n".join(body_parts).strip() or "No summary available."))

    subtitle = f"Generated {now_et:%Y-%m-%d %I:%M %p ET} • {window_line.replace(chr(10),' • ')}"
    return _build_text_pdf(
        output_path=output_path,
        title="Cutler Capital — Podcast Intelligence (ALL)",
        subtitle=subtitle,
        sections=sections,
    )

def _set_quarter(page, wanted: str) -> bool:
    """
    Try to select the requested quarter in the site's <select>.
    Returns True if it exists, False otherwise.
    """
    sel = page.locator(FILTERS["quarter"]).first
    try:
        sel.select_option(value=wanted)
    except Exception:
        try:
            sel.select_option(label=wanted)
        except Exception:
            return False
    page.wait_for_timeout(250)
    return True

def _search_by_fund(page, keyword: str, retries: int = 2) -> None:
    """
    Type a fund keyword into the BSD fund search box and trigger the search.

    More robust than the original version:
    - Waits explicitly for the search input to be visible.
    - Uses longer timeouts.
    - Retries a couple of times on timeout (with reload) before giving up.
    """
    for attempt in range(retries + 1):
        try:
            # Wait for the search input to actually be there
            inp = page.locator(FILTERS["fund"]).first
            inp.wait_for(state="visible", timeout=20000)

            # Clear and type the keyword
            inp.fill("")
            inp.type(keyword, delay=10)

            # Click search
            page.locator(FILTERS["search_btn"]).first.click(force=True)

            # Wait for either network idle or at least one row to show
            try:
                page.wait_for_load_state("networkidle", timeout=15000)
            except PlaywrightTimeoutError:
                page.locator(TABLE_ROW).first.wait_for(
                    state="visible",
                    timeout=15000,
                )

            # If we got here without exceptions, search was successful
            return

        except PlaywrightTimeoutError:
            # If we still have retries left, reload and try again
            if attempt < retries:
                page.reload()
                page.wait_for_load_state("domcontentloaded", timeout=20000)
                continue
            # Out of retries: re-raise so caller logs the error for this fund only
            raise


def _parse_letter_date_to_date(s: str) -> Optional[datetime]:
    """Parse BSD 'letter_date' cell into a datetime (date-only semantics).

    BSD commonly uses formats like:
      - MM/DD/YYYY
      - MM/DD/YY
      - YYYY-MM-DD
    Returns None if parsing fails.
    """
    if not s:
        return None
    s = (s or "").strip()
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _today_et_date() -> datetime:
    """Return current datetime in America/New_York."""
    return datetime.now(ZoneInfo("America/New_York"))


def _normalize_fund_name(name: str) -> str:
    n = (name or "").lower()
    n = re.sub(r"[^a-z0-9\s]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    # remove very common noise tokens
    noise = {"fund", "strategy", "portfolio", "trust", "class", "institutional", "investor", "shares", "share", "l.p", "lp"}
    toks = [t for t in n.split() if t and t not in noise]
    return " ".join(toks).strip()


def _build_fund_to_batch_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build lookups: exact_name -> Batch X, normalized_name -> Batch X."""
    exact: Dict[str, str] = {}
    norm: Dict[str, str] = {}
    for bname, names in RUNNABLE_BATCHES.items():
        for nm in names:
            if not nm:
                continue
            exact[nm.strip()] = bname
            nn = _normalize_fund_name(nm)
            if nn and nn not in norm:
                norm[nn] = bname
    return exact, norm

def _parse_rows(page, quarter: str) -> List[Hit]:
    rows = page.locator(TABLE_ROW)
    hits: List[Hit] = []
    for i in range(rows.count()):
        row = rows.nth(i)
        try:
            q = row.locator("td").nth(COLMAP["quarter"]-1).inner_text().strip()
            if q != quarter:
                continue
            letter_date = row.locator("td").nth(COLMAP["letter_date"]-1).inner_text().strip()
            fund_cell = row.locator("td").nth(COLMAP["fund_name"]-1)
            link = fund_cell.locator("a").first
            fund_name = (link.inner_text() or '').strip()
            fund_href = link.get_attribute("href") or ""
            if fund_href:
                hits.append(Hit(q, letter_date, fund_name, fund_href))
        except Exception:
            continue
    return hits

def _download_quarter_pdf_from_fund(page, quarter: str, dest_dir: Path) -> List[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    pdfs: List[Path] = []
    try:
        page.locator("text=Quarterly Letters").first.wait_for(state="visible", timeout=8000)
    except Exception:
        pass
    anchors = page.locator("a").all()
    candidates = []
    for a in anchors:
        try:
            text = (a.inner_text() or '').strip()
            title = a.get_attribute("title") or ""
            href = a.get_attribute("href") or ""
            if not href:
                continue
            if (text == quarter or quarter in title) and ("letters/file" in href or href.lower().endswith('.pdf')):
                candidates.append((a, href))
        except Exception:
            continue
    for a, href in candidates:
        try:
            with page.expect_download(timeout=8000) as dl_info:
                a.click(force=True)
            dl = dl_info.value
            fname = _safe(Path(dl.suggested_filename or Path(href).name or f"{quarter}.pdf").name)
            path = dest_dir / fname
            dl.save_as(str(path))
            pdfs.append(path)
            continue
        except Exception:
            pass
        try:
            r = requests.get(href, timeout=20)
            if r.status_code == 200 and r.content:
                fname = _safe(Path(href).name or f"{quarter}.pdf")
                path = dest_dir / fname
                with open(path, 'wb') as f:
                    f.write(r.content)
                pdfs.append(path)
        except Exception:
            continue
    return pdfs

# excerption + build

def run_excerpt_and_build(
    pdf_path: Path,
    out_dir: Path,
    source_pdf_name: Optional[str] = None,
    letter_date: Optional[str] = None,
    source_url: Optional[str] = None,
) -> Optional[Path]:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        tp = out_dir / "tickers.py"
        if not tp.exists():
            # place a copy so make_pdf can import user tickers
            (HERE / "tickers.py").exists() and shutil.copy2(HERE / "tickers.py", tp)

        excerpt_check.excerpt_pdf_for_tickers(str(pdf_path), debug=False)

        src_json = pdf_path.parent / "excerpts_clean.json"
        if not src_json.exists():
            return None

        dst_json = out_dir / "excerpts_clean.json"
        if src_json != dst_json:
            shutil.copy2(src_json, dst_json)

        out_pdf = out_dir / f"Excerpted_{_safe(pdf_path.stem)}.pdf"

        # Build compiled excerpt PDF (Fund Families style)
        make_pdf.build_pdf(
            excerpts_json_path=str(dst_json),
            output_pdf_path=str(out_pdf),
            report_title=f"Cutler Capital Excerpts – {pdf_path.stem}",
            source_pdf_name=source_pdf_name or pdf_path.name,
            format_style="legacy",
            letter_date=letter_date,
            source_url=source_url,
            ai_score=bool(st.session_state.get("ai_score_enabled", False)),
            ai_model=str(st.session_state.get("ai_score_model", "gpt-4o-mini") or "gpt-4o-mini"),
        )

        return out_pdf if out_pdf.exists() else None

    except Exception:
        traceback.print_exc()
        return None


# stamping + compile

def _overlay_single_page(w: float, h: float, left: str, mid: str, right: str) -> BytesIO:
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(w, h))
    c.setFont("Helvetica", 8.5)
    c.setFillColor(colors.HexColor("#4b2142"))  # Cutler purple
    L = R = 0.75 * 72
    T = 0.75 * 72
    if left:
        c.drawString(L, h - T + 0.35 * 72, left)
    if mid:
        text = (mid[:95] + '…') if len(mid) > 96 else mid
        c.drawCentredString(w / 2.0, h - T + 0.35 * 72, text)
    if right:
        c.drawRightString(w - R, h - T + 0.35 * 72, right)
    c.save()
    buf.seek(0)
    return buf

def _stamp_pdf(src: Path, left: str, mid: str, right: str) -> Path:
    try:
        r = _PdfReader(str(src))
    except Exception:
        return src
    w = _PdfWriter()
    for pg in r.pages:
        W = float(pg.mediabox.width)
        H = float(pg.mediabox.height)
        ov = _PdfReader(_overlay_single_page(W, H, left, mid, right)).pages[0]
        try:
            pg.merge_page(ov)
        except Exception:
            pass
        w.add_page(pg)
    tmp = src.with_suffix('.stamped.tmp.pdf')
    with open(tmp, 'wb') as f:
        w.write(f)
    return tmp

def _build_compiled_filename(batch: str, *, incremental: bool = False, dt: Optional[datetime] = None) -> str:
    """Return a human-friendly compiled PDF name like '12.08.25 Batch 1 Excerpt.pdf'.

    - Uses America/New_York time (EST/EDT) so file names match your local day.
    - Keeps the quarter inside the PDF body/header; the file name is what interns archive.
    """
    if dt is None:
        dt = _now_et()
    # Desired format: 12.8.25 Batch 1 Excerpt.pdf (no leading zeros on month/day)
    date_str = f"{dt.month}.{dt.day}.{dt.strftime('%y')}"
    suffix = "Incremental Excerpt" if incremental else "Excerpt"
    return f"{date_str} {batch} {suffix}.pdf"

# ---------- Batch-level ticker index (Fund Families compiled PDFs) ----------
def _collect_hit_tickers_from_excerpts_json(path: Path) -> List[str]:
    """Return tickers (keys) that have at least one kept excerpt."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    out: List[str] = []
    for tkr, items in data.items():
        try:
            if items and isinstance(items, list):
                out.append(str(tkr).strip())
        except Exception:
            continue
    return [t for t in out if t]


def _batch_hit_tickers_in_order(collected_pdfs: List[Path]) -> List[str]:
    """
    Collect hit tickers across a batch (first-appearance order) by reading the
    sibling excerpts_clean.json next to each excerpted PDF.
    """
    seen = set()
    ordered: List[str] = []
    for pdf_path in collected_pdfs:
        j = pdf_path.parent / "excerpts_clean.json"
        if not j.exists():
            # fallback for older layouts
            j = pdf_path.parent / "excerpts.json"
        if not j.exists():
            continue
        for t in _collect_hit_tickers_from_excerpts_json(j):
            if t not in seen:
                seen.add(t)
                ordered.append(t)
    return ordered


def _build_batch_index_pdf(*, tickers_in_doc: List[str], out_pdf: Path, label: str) -> Optional[Path]:
    """Render a 1+ page index PDF listing all hit tickers in the compiled batch."""
    if not tickers_in_doc:
        return None
    try:
        # Local import to keep global imports minimal in Streamlit
        from reportlab.lib.pagesizes import LETTER
        from reportlab.platypus import SimpleDocTemplate

        name_map = {}
        try:
            # Reuse make_pdf's ticker display-name loader (reads tickers.py if present)
            name_map = make_pdf._load_ticker_display_names(Path(".").resolve())  # type: ignore[attr-defined]
        except Exception:
            name_map = {}

        doc = SimpleDocTemplate(
            str(out_pdf),
            pagesize=LETTER,
            leftMargin=0.75 * 72,
            rightMargin=0.75 * 72,
            topMargin=0.75 * 72,
            bottomMargin=0.75 * 72,
        )
        story: List[Any] = []
        # Reuse make_pdf's index renderer for identical look-and-feel.
        make_pdf._render_index_page(  # type: ignore[attr-defined]
            story,
            tickers_in_doc=tickers_in_doc,
            name_map=name_map,
            label=label,
        )
        doc.build(story)
        return out_pdf if out_pdf.exists() else None
    except Exception:
        traceback.print_exc()
        return None

def compile_merged(batch: str, quarter: str, collected: List[Path], *, incremental: bool = False) -> Optional[Path]:
    if not collected:
        return None

    # File name now matches your convention:
    #   12.8.25 Batch 1 Excerpt.pdf
    #   12.8.25 Batch 1 Incremental Excerpt.pdf
    out_name = _build_compiled_filename(batch, incremental=incremental)
    out = CP_DIR / out_name

    m = PdfMerger()
    added = 0

    # --- NEW: prepend a batch-level ticker index page (Fund Families) ---
    try:
        tickers_in_doc = _batch_hit_tickers_in_order(collected)
        if tickers_in_doc:
            idx_pdf = CP_DIR / f"_{_safe(batch.replace('—', '-').replace('  ', ' ').strip())}_{quarter}_Index.pdf"
            built = _build_batch_index_pdf(
                tickers_in_doc=tickers_in_doc,
                out_pdf=idx_pdf,
                label="Index — Hit Tickers",
            )
            if built and built.exists():
                stamped_idx = _stamp_pdf(
                    built,
                    left=batch,
                    mid="Index — Hit Tickers",
                    right=f"Run {_now_et():%Y-%m-%d %H:%M} ET",
                )
                m.append(str(stamped_idx))
                added += 1
    except Exception:
        # Index is best-effort; compilation should still succeed without it.
        pass

    for p in collected:
        try:
            title = p.stem.replace("_", " ").replace("-", " ")
            stamped = _stamp_pdf(
                p,
                left=batch,
                mid=title,
                right=f"Run {_now_et():%Y-%m-%d %H:%M} ET",
            )
            m.append(str(stamped))
            added += 1
        except Exception:
            continue

    if not added:
        m.close()
        return None

    try:
        m.write(str(out))
    finally:
        m.close()

    return out

# -------------------------------------------------------------------
# Seeking Alpha Analysis API helpers (RapidAPI)
# -------------------------------------------------------------------

SA_ANALYSIS_BASE = "https://seeking-alpha.p.rapidapi.com"


def _get_sa_rapidapi_key() -> str:
    key = os.getenv("SA_RAPIDAPI_KEY")
    if not key:
        raise RuntimeError(
            "SA_RAPIDAPI_KEY env var is not set – add it to your .env for Seeking Alpha Analysis."
        )
    return key

# sa_analysis_api.py

import os
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

SA_RAPIDAPI_KEY = os.environ.get("SA_RAPIDAPI_KEY")
if not SA_RAPIDAPI_KEY:
    raise SystemExit("SA_RAPIDAPI_KEY env var is not set")

BASE_URL = "https://seeking-alpha.p.rapidapi.com"

HEADERS = {
    "x-rapidapi-key": SA_RAPIDAPI_KEY,
    "x-rapidapi-host": "seeking-alpha.p.rapidapi.com",
}

@dataclass
class AnalysisArticle:
    id: str
    title: str
    published: str
    url: str
    author: str = ""
    author_slug: str = ""


def _call_sa(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = BASE_URL + endpoint
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()
    

def fetch_analysis_list(symbol: str, size: int = 5) -> List[AnalysisArticle]:
    """
    GET /analysis/v2/list?id={symbol}&size={size}&number=1
    """
    payload = _call_sa("/analysis/v2/list", {"id": symbol.lower(), "size": size, "number": 1})

    items = payload.get("data", [])
    out: List[AnalysisArticle] = []

    for item in items:
        attrs = item.get("attributes", {})
        art_id = str(item.get("id"))
        title = attrs.get("title", "")
        published = attrs.get("publishOn", "")
        link = "https://seekingalpha.com" + item.get("links", {}).get("self", "")

        # Author fields are inconsistently present; try attributes first, then relationships.
        author = (
            attrs.get("authorName")
            or attrs.get("author")
            or attrs.get("author_name")
            or ""
        )
        author_slug = (
            attrs.get("authorSlug")
            or attrs.get("author_slug")
            or ""
        )
        if not author_slug:
            try:
                rel_author = (item.get("relationships") or {}).get("author") or {}
                rel_data = rel_author.get("data") or {}
                if isinstance(rel_data, dict):
                    author_slug = str(rel_data.get("id") or "") or author_slug
            except Exception:
                pass

        out.append(AnalysisArticle(
            id=art_id,
            title=title,
            published=published,
            url=link,
            author=author,
            author_slug=author_slug,
        ))

    return out


def fetch_analysis_details(article_id: str) -> Dict[str, Any]:
    """
    GET /analysis/v2/get-details?id={article_id}
    Returns a dict with title, body_html, summary_html and image_url.
    """
    payload = _call_sa("/analysis/v2/get-details", {"id": article_id})

    try:
        main = payload["data"][0]
        attrs = main.get("attributes", {})
    except Exception:
        return {}

    # Different fields exist depending on endpoint version; cover both
    body_html = (
        attrs.get("bodyHtml")    # camelCase
        or attrs.get("body_html")
        or ""
    )
    summary_html = (
        attrs.get("summaryHtml")
        or attrs.get("summary_html")
        or ""
    )
    image_url = attrs.get("gettyImageUrl") or ""

    author = (
        attrs.get("authorName")
        or attrs.get("author")
        or attrs.get("author_name")
        or ""
    )
    author_slug = (
        attrs.get("authorSlug")
        or attrs.get("author_slug")
        or ""
    )

    return {
        "title": attrs.get("title", ""),
        "body_html": body_html,
        "summary_html": summary_html,
        "image_url": image_url,
        "author": author,
        "author_slug": author_slug,
    }

def fetch_sa_analysis_list(symbol: str, size: int = 10) -> list[dict]:
    """
    Call /analysis/v2/list for a single symbol and return a simplified list
    of articles: id, title, published, primary_tickers, url.
    """
    api_key = _get_sa_rapidapi_key()
    url = f"{SA_ANALYSIS_BASE}/analysis/v2/list"

    params = {
        "id": symbol.lower(),  # API expects lowercase id like 'tsla', 'aapl'
        "size": str(size),
        "number": "1",         # first "page"
    }
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "seeking-alpha.p.rapidapi.com",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("data", [])
    items: list[dict] = []

    for row in data:
        if not isinstance(row, dict):
            continue
        art_id = row.get("id")
        attrs = row.get("attributes", {}) or {}
        rel = row.get("relationships", {}) or {}

        publish_on = attrs.get("publishOn")
        title = attrs.get("title", "").strip()

        # primary tickers come through as tag ids – we just keep them for debugging;
        # you already know which symbol you asked for.
        pt_data = ((rel.get("primaryTickers") or {}).get("data")) or []
        primary_ids = [
            t.get("id") for t in pt_data
            if isinstance(t, dict) and t.get("id")
        ]

        article_url = f"https://seekingalpha.com/article/{art_id}" if art_id else ""

        items.append(
            {
                "id": art_id,
                "title": title,
                "published": publish_on,
                "primary_tickers": primary_ids,
                "url": article_url,
            }
        )

    return items


def fetch_sa_analysis_body(article_id: str) -> str:
    """
    Call /analysis/v2/get-details for a single article and return the raw HTML body.
    """
    api_key = _get_sa_rapidapi_key()
    url = f"{SA_ANALYSIS_BASE}/analysis/v2/get-details"

    params = {"id": str(article_id)}
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "seeking-alpha.p.rapidapi.com",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("data") or []
    if not data:
        return ""

    first = data[0]
    attrs = first.get("attributes", {}) or {}

    # In practice this field is usually called "content"
    body_html = attrs.get("content") or ""
    return body_html

def clean_sa_html(html: str, max_len: Optional[int] = None) -> str:
    """
    Convert Seeking Alpha article HTML into clean, readable plain text.

    - Turns block tags (p/div/br/li/h1–h6) into paragraph breaks.
    - Strips all other tags.
    - Normalises whitespace but *keeps* paragraph breaks.
    - Inserts spaces between digits and letters to fix things like
      '13Bofcash' -> '13B of cash'.
    - If max_len is given, truncates on a word boundary.
    """
    import re
    import html as html_lib

    if not html:
        return ""

    # Decode HTML entities (&amp;, &nbsp;, etc.)
    text = html_lib.unescape(html)

    # 1) Turn common block / line-break tags into newlines
    text = re.sub(
        r"(?i)<\s*(br\s*/?|/p|p|/div|div|li|/li|h[1-6]|/h[1-6])[^>]*>",
        "\n",
        text,
    )

    # 2) Remove any remaining tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3) Normalise whitespace but keep newlines-as-paragraphs
    text = text.replace("\r", "")
    lines = []
    for ln in text.split("\n"):
        # collapse spaces/tabs on each line
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if ln:
            lines.append(ln)
    # rebuild paragraphs with a blank line between them
    text = "\n\n".join(lines)

    # 4) Fix digit/letter run-ons: 15.7Billion -> 15.7 Billion, EPS1.86 -> EPS 1.86
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)

    # 5) Optional truncation for model input
    if max_len and len(text) > max_len:
        cut = text[:max_len]
        cut = cut.rsplit(" ", 1)[0]  # don’t cut in the middle of a word
        text = cut + " ..."

    return text

def build_sa_analysis_digest(
    symbol: str,
    articles,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Build a short bullet-point digest from a list of AnalysisArticle objects.

    `articles` is the list returned by sa_analysis_api.fetch_analysis_list.
    We use only lightweight metadata (date, title, url).
    """
    if not articles:
        return f"No recent Seeking Alpha analysis articles found for {symbol}."

    # Build a compact text summary of the articles we have.
    lines = []
    for art in articles:
        try:
            date_str = art.published.split("T", 1)[0] if art.published else ""
            title = art.title or ""
            url = art.url or ""
            lines.append(f"- {date_str} — {title} ({url})")
        except Exception:
            continue

    context_block = "\n".join(lines)

    system_msg = (
        "You are helping a fundamental portfolio manager at a small buy-side shop.\n"
        "You will receive a list of recent Seeking Alpha ANALYSIS articles for one ticker.\n"
        "Write a concise bullet-point digest (4–7 bullets max) capturing:\n"
        "- Overall stance (bullish / bearish / mixed) across authors\n"
        "- Key fundamental drivers mentioned (earnings, pipeline, margins, cash flow, etc.)\n"
        "- Any repeated risks or points of disagreement\n"
        "- Any notable technical or sentiment comments\n"
        "Keep language plain, professional, and focused on what a PM should know."
    )

    user_msg = (
        f"TICKER: {symbol}\n\n"
        "Recent Seeking Alpha analysis articles:\n"
        f"{context_block}\n\n"
        "Now write the digest."
    )

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        digest = resp["choices"][0]["message"]["content"].strip()
        return digest
    except Exception as e:
        return f"Error while calling OpenAI: {e}"

def clean_html_to_text(html: str) -> str:
    if not html:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Convert multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_sa_html_to_markdown(raw_html: str) -> str:
    """
    Convert Seeking Alpha article HTML into clean, readable text for Streamlit.

    - Converts block tags to paragraph breaks.
    - Strips all other tags.
    - Normalises whitespace but keeps paragraph breaks.
    - Fixes digit/letter run-ons like '13Bofcash' -> '13B of cash'.
    """
    import re
    import html as html_lib

    if not raw_html:
        return ""

    # Decode entities (&amp;, &nbsp;, etc.)
    text = html_lib.unescape(raw_html)

    # Turn common block / break tags into newlines
    text = re.sub(
        r"(?i)<\s*(br\s*/?|/p|p|/div|div|/li|li|h[1-6]|/h[1-6])[^>]*>",
        "\n",
        text,
    )

    # Remove remaining tags, leaving a space so words don't glue together
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalise whitespace but keep paragraph structure
    text = text.replace("\r", "")
    lines = []
    for ln in text.split("\n"):
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if ln:
            lines.append(ln)
    text = "\n\n".join(lines)

    # Fix digit/letter run-ons
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)

    return text.strip()


def draw_seeking_alpha_news_section() -> None:
    """
    Seeking Alpha – Analysis articles by ticker (RapidAPI).

    - Supports Batch mode (10 tickers per batch) and Manual mode (pick up to 10 tickers)
    - Caches fetched results in-session to avoid repeated API calls
    - Robust to RapidAPI returning article rows as objects/dicts/IDs
    """
    import streamlit as st
    import pandas as pd
    from pathlib import Path
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import sa_analysis_api as sa_api

    # ---------------- Session cache ----------------
    if "sa_cache" not in st.session_state:
        # cache_key -> {"articles": list[dict], "digest_text": str|None}
        st.session_state["sa_cache"] = {}
    if "sa_pdf_bytes" not in st.session_state:
        st.session_state["sa_pdf_bytes"] = None
    if "sa_pdf_name" not in st.session_state:
        st.session_state["sa_pdf_name"] = None

    def _is_probable_ticker(s: str) -> bool:
        s = (s or "").strip().upper()
        return s.isalnum() and 1 <= len(s) <= 6

    def _pretty_company_name(v) -> str:
        if not v:
            return ""
        if isinstance(v, dict):
            return (v.get("name") or v.get("company") or "").strip()
        if isinstance(v, str):
            return v.strip()
        return ""

    def _make_batches(items: list[str], batch_size: int) -> list[list[str]]:
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    def _sa_article_id(a) -> str:
        if a is None:
            return ""
        if isinstance(a, (str, int)):
            return str(a)
        if isinstance(a, dict):
            return str(a.get("id") or a.get("article_id") or a.get("articleId") or a.get("uid") or "")
        for attr in ("id", "article_id", "articleId", "uid"):
            if hasattr(a, attr):
                v = getattr(a, attr)
                if v:
                    return str(v)
        return ""
    
    def _sa_article_as_dict(a) -> dict:
        """
        Normalize an SA article row (dict or AnalysisArticle object) into a dict
        that downstream UI + PDF code can use consistently.
        """
        if a is None:
            return {}

        # Already dict
        if isinstance(a, dict):
            d = dict(a)  # shallow copy
            # ensure standard keys exist
            if "id" not in d:
                d["id"] = _sa_article_id(a)
            d.setdefault("title", d.get("headline") or d.get("name") or "")
            d.setdefault("url", d.get("link") or d.get("permalink") or "")
            d.setdefault("published_at", d.get("published") or d.get("publishOn") or d.get("date") or "")
            d.setdefault("author", d.get("author_name") or d.get("author") or d.get("authorName") or "")
            d.setdefault("author_slug", d.get("author_slug") or d.get("authorSlug") or "")
            return d

        # Object / dataclass (patched sa_analysis_api.AnalysisArticle)
        aid = _sa_article_id(a)
        if not aid:
            return {}

        return {
            "id": aid,
            "title": getattr(a, "title", "") or "",
            "url": getattr(a, "url", "") or "",
            "published_at": getattr(a, "published", "") or getattr(a, "published_at", "") or "",
            # author fields from patched sa_analysis_api.py
            "author": getattr(a, "author_name", "") or getattr(a, "author", "") or "",
            "author_slug": getattr(a, "author_slug", "") or "",
        }

    def _sa_article_row(a) -> dict:
        """Normalize a list row into a dict with stable keys."""
        try:
            if a is None:
                return {}
            if isinstance(a, dict):
                return {
                    "id": str(a.get("id") or a.get("article_id") or a.get("articleId") or a.get("uid") or ""),
                    "title": a.get("title") or "",
                    "url": a.get("url") or a.get("link") or "",
                    "published_at": a.get("published_at") or a.get("published") or a.get("date") or "",
                    "author": a.get("author") or a.get("author_name") or a.get("authorName") or "",
                    "author_slug": a.get("author_slug") or a.get("authorSlug") or "",
                }
            if isinstance(a, (str, int)):
                return {"id": str(a)}
            # object / dataclass-like
            aid = _sa_article_id(a)
            return {
                "id": aid,
                "title": getattr(a, "title", "") or "",
                "url": getattr(a, "url", "") or getattr(a, "link", "") or "",
                "published_at": getattr(a, "published_at", "") or getattr(a, "published", "") or "",
                "author": getattr(a, "author", "") or getattr(a, "author_name", "") or getattr(a, "authorName", "") or "",
                "author_slug": getattr(a, "author_slug", "") or getattr(a, "authorSlug", "") or "",
            }
        except Exception:
            aid = _sa_article_id(a)
            return {"id": aid} if aid else {}

    def _extract_from_details(details: dict) -> tuple[str, str, str]:
        """Return (body_html_or_text, author, title) from a details payload (defensive)."""
        if not isinstance(details, dict):
            return "", "", ""
        # Some APIs return JSON:API shapes: {"data":{"attributes":{...}}}
        base = details
        if isinstance(details.get("data"), dict):
            attrs = details["data"].get("attributes")
            if isinstance(attrs, dict):
                base = {**details, **attrs}

        body = (
            base.get("body_clean")
            or base.get("body_html")
            or base.get("content")
            or base.get("body")
            or base.get("html")
            or base.get("text")
            or ""
        )
        author = base.get("author") or base.get("author_name") or base.get("authorName") or ""
        title = base.get("title") or ""
        return str(body or ""), str(author or ""), str(title or "")

    def _cache_key(sym: str, max_articles: int, model: str, include_digest: bool) -> str:
        return f"sa|{sym.upper()}|{max_articles}|{model}|{int(include_digest)}"

    # ---------------- Universe (from tickers.py if available) ----------------
    CUTLER_TICKERS = {}
    universe: list[str] = []
    try:
        from tickers import tickers as CUTLER_TICKERS  # type: ignore
        if isinstance(CUTLER_TICKERS, dict):
            universe = [t for t in CUTLER_TICKERS.keys() if _is_probable_ticker(t)]
    except Exception:
        CUTLER_TICKERS = {}

    # fallback: allow manual input universe if tickers.py missing
    if not universe:
        universe = sorted(list({t for t in ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "ABBV"]}))

    st.markdown("### Seeking Alpha")
    st.caption("Pull recent Seeking Alpha Analysis articles by ticker. Use Batch mode to run 10 tickers at a time.")

    # ---------------- Mode: batch vs manual ----------------
    batch_mode = st.toggle("Batch mode (10 tickers per batch)", value=True, key="sa_batch_mode_toggle")

    selected_tickers: list[str] = []
    if batch_mode:
        batches = _make_batches(universe, 10)
        batch_options = [f"Batch {i+1} ({len(b)} tickers)" for i, b in enumerate(batches)]
        batch_idx = st.selectbox(
            "Seeking Alpha batch",
            options=list(range(len(batch_options))),
            format_func=lambda i: batch_options[i],
            key="sa_batch_select_idx",
        )
        selected_tickers = batches[int(batch_idx)]
        st.caption("Tickers in this batch: " + ", ".join(selected_tickers))
    else:
        selected_tickers = st.multiselect(
            "Select up to 10 tickers for Seeking Alpha",
            options=universe,
            default=["ABBV"] if "ABBV" in universe else [],
            max_selections=10,
            key="sa_manual_tickers_multiselect",
        )

    if not selected_tickers:
        st.info("Select at least one ticker to continue.")
        return

    # ---------------- Navigation within chosen tickers ----------------
    nav_key = "sa_nav_idx_batch" if batch_mode else "sa_nav_idx_manual"
    if nav_key not in st.session_state:
        st.session_state[nav_key] = 0
    st.session_state[nav_key] = max(0, min(int(st.session_state[nav_key]), len(selected_tickers) - 1))
    ticker = selected_tickers[int(st.session_state[nav_key])]

    nav_l, nav_mid, nav_r = st.columns([1, 6, 1])
    with nav_l:
        if st.button("Previous", key=f"sa_prev_{nav_key}", disabled=st.session_state[nav_key] == 0, use_container_width=True):
            st.session_state[nav_key] -= 1
            st.rerun()
    with nav_r:
        if st.button("Next", key=f"sa_next_{nav_key}", disabled=st.session_state[nav_key] >= len(selected_tickers) - 1, use_container_width=True):
            st.session_state[nav_key] += 1
            st.rerun()

    ticker_name = _pretty_company_name(CUTLER_TICKERS.get(ticker))
    st.markdown(f"**Currently viewing:** `{ticker}`" + (f" — {ticker_name}" if ticker_name else ""))

    # ---------------- Controls ----------------
    max_articles = st.slider(
        "Number of recent analysis articles to use",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        key="sa_max_articles_slider",
    )

    model = st.selectbox(
        "OpenAI model for digest",
        options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
        index=0,
        key="sa_digest_model_select",
    )

    include_ai_digest = st.checkbox("Include AI digest (optional)", value=False, key="sa_include_digest")

    fetch_clicked = st.button(
        "Fetch / refresh Seeking Alpha analysis",
        key=f"sa_fetch_refresh_{'batch' if batch_mode else 'manual'}_{max_articles}_{model}_{int(include_ai_digest)}",
        use_container_width=True,
    )

    def _fetch_for_symbol(sym: str) -> tuple[list[dict], str | None]:
        # list items
        raw_list = sa_api.fetch_analysis_list(sym, size=max_articles) or []
        rows = [_sa_article_as_dict(a) for a in raw_list]
        rows = [r for r in rows if r.get("id")]
        if not rows:
            return [], None

        # details for bodies / richer metadata
        for r in rows:
            try:
                details = sa_api.fetch_analysis_details(r["id"]) or {}
                body_raw, author_d, title_d = _extract_from_details(details)

                # Body: if it looks like HTML, clean it; else keep as-is
                if "<" in body_raw and ">" in body_raw:
                    r["body_clean"] = clean_sa_html(body_raw)
                else:
                    r["body_clean"] = (body_raw or "").strip()

                # author/title backfill
                if not r.get("author"):
                    r["author"] = details.get("author_name") or details.get("author") or details.get("authorName") or ""
                if title_d and not r.get("title"):
                    r["title"] = title_d
                # url/published backfill if present
                if isinstance(details, dict):
                    r["url"] = r.get("url") or details.get("url") or details.get("link") or r.get("url") or ""
                    r["published_at"] = r.get("published_at") or details.get("published_at") or details.get("published") or r.get("published_at") or ""
            except Exception:
                # partial data is ok
                if "body_clean" not in r:
                    r["body_clean"] = ""
                continue

        digest_text = None
        if include_ai_digest:
            try:
                digest_text = sa_api.build_sa_analysis_digest(sym, raw_list, model=model, max_articles=min(max_articles, 6))
            except Exception as e:
                st.warning(f"Digest failed for {sym}: {e}")

        return rows, digest_text

    # ---------------- Fetch + cache ----------------
    cache = st.session_state["sa_cache"]
    if fetch_clicked:
        # Batch: fetch all tickers in the selected batch so Next works without re-running each symbol
        tickers_to_fetch = selected_tickers if batch_mode else [ticker]
        with st.spinner("Fetching Seeking Alpha analysis..."):
            for sym in tickers_to_fetch:
                ck = _cache_key(sym, max_articles, model, include_ai_digest)
                try:
                    articles_sym, digest_sym = _fetch_for_symbol(sym)
                    cache[ck] = {"articles": articles_sym, "digest_text": digest_sym}
                except Exception as e:
                    cache[ck] = {"articles": [], "digest_text": None, "error": str(e)}
        # Reset any prior PDF when data refreshes
        st.session_state["sa_pdf_bytes"] = None
        st.session_state["sa_pdf_name"] = None
        st.success("Seeking Alpha fetch complete.")
        st.rerun()

    # If the current ticker has no cached results for this config, fetch just this ticker (lazy) in batch mode
    ck_current = _cache_key(ticker, max_articles, model, include_ai_digest)
    if ck_current not in cache and batch_mode:
        with st.spinner(f"Fetching {ticker} (not cached for this batch/config)..."):
            try:
                articles_sym, digest_sym = _fetch_for_symbol(ticker)
                cache[ck_current] = {"articles": articles_sym, "digest_text": digest_sym}
            except Exception as e:
                cache[ck_current] = {"articles": [], "digest_text": None, "error": str(e)}

    # ---------------- Render ----------------
    payload = cache.get(ck_current) or {}
    if payload.get("error"):
        st.error(f"Seeking Alpha API error for {ticker}: {payload['error']}")

    articles = payload.get("articles") or []
    digest_text = payload.get("digest_text")

    if not articles:
        st.info(f"No usable analysis articles returned for {ticker} with the current settings.")
        return

    if digest_text:
        st.markdown("#### AI Analysis Digest")
        st.write(digest_text)

    # Table
    rows = []
    for a in articles:
        rows.append(
            {
                "date": a.get("published_at") or a.get("date") or "",
                "title": a.get("title") or "",
                "author": a.get("author") or a.get("author_name") or "",
                "url": a.get("url") or a.get("link") or "",
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Bodies (highlight must-read paragraphs)
    st.markdown("#### Full article bodies (cleaned)")
    # Must-read highlighting (heuristic, tuned to reduce false positives)
    MUST_PHRASES = [
        "strong buy", "strong sell", "buy rating", "sell rating", "hold rating",
        "initiated", "downgrade", "upgrade", "price target", "target price",
        "raised guidance", "cut guidance",
    ]
    SCORE_KEYWORDS = [
        "valuation", "p/e", "pe ratio", "multiple", "eps", "earnings", "revenue", "guidance",
        "dividend", "free cash flow", "fcf", "margin", "risk", "catalyst",
        "upside", "downside", "bear case", "bull case",
        "pipeline", "patent", "competition", "regulatory",
    ]

    def _must_read_para(p: str) -> bool:
        pl = (p or "").lower().strip()
        if not pl:
            return False

        # Always highlight if the paragraph explicitly references the ticker.
        if re.search(rf"\b{re.escape(ticker.lower())}\b", pl):
            return True

        # Hard triggers
        if any(ph in pl for ph in MUST_PHRASES):
            return True

        # Soft scoring: require at least 2 "finance-relevant" terms to avoid highlighting generic prose.
        score = sum(1 for k in SCORE_KEYWORDS if k in pl)
        return score >= 2

    for i, a in enumerate(articles, start=1):
        title = a.get("title") or f"Article {i}"
        url = a.get("url") or a.get("link") or ""
        body = (a.get("body_clean") or a.get("body") or "").strip()
        # Normalize/choose author field defensively
        author = (a.get("author") or a.get("author_name") or a.get("authorName") or "").strip()

        exp_label = f"{i}. {title}" + (f" — {author}" if author else "")
        with st.expander(exp_label, expanded=False):
            if url:
                st.markdown(f"Source: {url}")
            if body:
                paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
                for p in paras:
                    if _must_read_para(p):
                        st.markdown(
                            f"""<div style="background:#d9f7e5;padding:8px;border-radius:8px;margin:6px 0;">{p}</div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""<div style="padding:6px 2px;margin:2px 0;">{p}</div>""",
                            unsafe_allow_html=True,
                        )
            else:
                st.caption("No body returned for this article.")

    # ---------------- Export PDF (persistent) ----------------
    def _build_sa_pdf_bytes(sym: str, articles: list[dict], digest_text: str | None) -> bytes:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from xml.sax.saxutils import escape
        import io

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=LETTER,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )
        styles = getSampleStyleSheet()
        base = styles["BodyText"]
        base.leading = 12

        must_style = ParagraphStyle(
            "MustRead",
            parent=base,
            backColor=colors.HexColor("#d9f7e5"),
            borderPadding=6,
            spaceAfter=6,
        )

        story = []
        story.append(Paragraph("Cutler Capital – Seeking Alpha Export", styles["Title"]))
        ts = datetime.now(ZoneInfo("America/New_York")).strftime("%m.%d.%y %I:%M %p ET")
        story.append(Paragraph(f"Generated {escape(ts)} • Ticker: {escape(sym)}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        if digest_text:
            story.append(Paragraph("AI Digest", styles["Heading2"]))
            story.append(Paragraph(escape(digest_text).replace("\n", "<br/>"), base))
            story.append(Spacer(1, 0.15 * inch))

        story.append(Paragraph("Articles", styles["Heading2"]))
        for idx, a in enumerate(articles, start=1):
            title = a.get("title") or f"Article {idx}"
            author = a.get("author") or a.get("author_name") or ""
            url = a.get("url") or a.get("link") or ""
            body = (a.get("body_clean") or a.get("body") or "").strip()

            story.append(Paragraph(f"{idx}. {escape(title)}" + (f" — {escape(author)}" if author else ""), styles["Heading3"]))
            if url:
                story.append(Paragraph(f"Source: {escape(url)}", styles["Normal"]))
                story.append(Spacer(1, 0.08 * inch))

            if body:
                paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
                for p in paras:
                    p_esc = escape(p)
                    story.append(Paragraph(p_esc, must_style if _must_read_para(p) else base))
            else:
                story.append(Paragraph("No body returned for this article.", base))

            story.append(Spacer(1, 0.15 * inch))

        doc.build(story)
        return buf.getvalue()

    # Build once, persist bytes + filename
    with st.expander("PDF build settings (performance)", expanded=False):
        st.caption("Optional limits. Set high values to include full article bodies; lower values can speed up PDF builds on Streamlit Cloud.")
        st.number_input(
            "Max body paragraphs per ticker (export)",
            min_value=5,
            max_value=5000,
            value=int(st.session_state.get("sa_pdf_max_paras_per_ticker", 5000)),
            step=5,
            key="sa_pdf_max_paras_per_ticker",
        )
        st.number_input(
            "Max body paragraphs per article (export)",
            min_value=2,
            max_value=500,
            value=int(st.session_state.get("sa_pdf_max_paras_per_article", 500)),
            step=1,
            key="sa_pdf_max_paras_per_article",
        )

    if st.button("Build downloadable Seeking Alpha PDF", key=f"sa_build_pdf_{ck_current}", use_container_width=True):
        with st.spinner("Building PDF (Fund Families style)..."):
            try:
                import tempfile
                import json as _json

                tickers_for_pdf = list(selected_tickers) if selected_tickers else [ticker]
                combined: dict[str, list[dict]] = {}

                # Performance guardrails: scoring every paragraph across 10 tickers can be slow.
                # We cap paragraphs exported per ticker to keep build time reasonable on Streamlit Cloud.
                max_paras_per_ticker = int(st.session_state.get("sa_pdf_max_paras_per_ticker", 5000))
                max_paras_per_article = int(st.session_state.get("sa_pdf_max_paras_per_article", 500))

                progress = st.progress(0.0)
                status = st.empty()


                # Ensure we have cached results for each ticker, even if user didn't click through Next/Prev
                for i_sym, sym in enumerate(tickers_for_pdf):
                    status.info(f"Preparing {sym} ({i_sym+1}/{len(tickers_for_pdf)}) for PDF…")
                    progress.progress((i_sym)/max(1,len(tickers_for_pdf)))
                    ck_sym = _cache_key(sym, max_articles, model, include_ai_digest)
                    if ck_sym not in cache:
                        try:
                            raw_list = sa_api.fetch_analysis_list(sym, size=max_articles)
                            # Enrich with author credibility signals (followers / rating / etc.)
                            try:
                                if hasattr(sa_api, "enrich_articles_with_author_metrics"):
                                    raw_list = sa_api.enrich_articles_with_author_metrics(raw_list or [])
                            except Exception:
                                pass
                            _sa_metrics_by_id = {}
                            try:
                                for _art in raw_list or []:
                                    _aid = str(getattr(_art, "id", "") or "")
                                    if not _aid:
                                        continue
                                    _sa_metrics_by_id[_aid] = {
                                        "followers": getattr(_art, "author_followers", None),
                                        "rating": getattr(_art, "author_rating", None),
                                        "articles": getattr(_art, "author_articles_count", None),
                                    }
                            except Exception:
                                _sa_metrics_by_id = {}
                            rows = [_sa_article_row(x) for x in (raw_list or [])]
                            rows = [r for r in rows if r.get("id")]

                            # Fetch article details to get bodies/authors/titles/urls
                            for r in rows:
                                aid = r.get("id") or ""
                                if not aid:
                                    continue
                                details = {}
                                try:
                                    if hasattr(sa_api, "fetch_analysis_details"):
                                        details = sa_api.fetch_analysis_details(aid) or {}
                                except Exception:
                                    details = {}

                                body_raw, author_d, title_d = _extract_from_details(details)

                                # pick up author/slug if available in details
                                if isinstance(details, dict):
                                    if details.get("author") and not r.get("author"):
                                        r["author"] = details.get("author")
                                    if details.get("author_slug") and not r.get("author_slug"):
                                        r["author_slug"] = details.get("author_slug")

                                if "<" in (body_raw or "") and ">" in (body_raw or ""):
                                    r["body_clean"] = clean_sa_html(body_raw)
                                else:
                                    r["body_clean"] = (body_raw or "").strip()

                                if author_d and not r.get("author"):
                                    r["author"] = author_d
                                if title_d and not r.get("title"):
                                    r["title"] = title_d
                                if isinstance(details, dict) and details.get("url") and not r.get("url"):
                                    r["url"] = details.get("url")

                        except Exception as e:
                            rows = []
                            cache[ck_sym] = {"articles": [], "digest_text": None, "error": str(e)}
                        else:
                            cache[ck_sym] = {"articles": rows, "digest_text": None}

                    payload_sym = cache.get(ck_sym) or {}
                    arts_sym = payload_sym.get("articles") or []

                    items: list[dict] = []
                    for a in arts_sym:
                        body = (a.get("body_clean") or "").strip()
                        if not body:
                            continue

                        title = (a.get("title") or "").strip()
                        author = (a.get("author") or a.get("author_name") or "").strip()
                        url = (a.get("url") or a.get("link") or "").strip()

                        header_bits = []
                        if title:
                            header_bits.append(title)
                        if author:
                            header_bits.append(f"— {author}")
                        header = " ".join(header_bits).strip()
                        if url:
                            header = f"{header}\nSource: {url}" if header else f"Source: {url}"

                        # Credibility line (best-effort)
                        try:
                            _m = _sa_metrics_by_id.get(str(a.get("id") or ""), {})
                        except Exception:
                            _m = {}
                        cred_parts = []
                        af = _m.get("followers")
                        ar = _m.get("rating")
                        ac = _m.get("articles")
                        try:
                            if isinstance(af, (int, float)):
                                cred_parts.append(f"{int(af):,} followers")
                        except Exception:
                            pass
                        try:
                            if isinstance(ar, (int, float)):
                                cred_parts.append(f"{float(ar):.1f} rating")
                        except Exception:
                            pass
                        try:
                            if isinstance(ac, (int, float)):
                                cred_parts.append(f"{int(ac):,} articles")
                        except Exception:
                            pass
                        if cred_parts:
                            header = f"{header}\nCredibility: " + " | ".join(cred_parts) if header else "Credibility: " + " | ".join(cred_parts)

                        # Credibility line (best-effort)
                        try:
                            _m = _sa_metrics_by_id.get(str(a.get("id") or ""), {})
                        except Exception:
                            _m = {}
                        cred_parts = []
                        af = _m.get("followers")
                        ar = _m.get("rating")
                        ac = _m.get("articles")
                        try:
                            if isinstance(af, (int, float)):
                                cred_parts.append(f"{int(af):,} followers")
                        except Exception:
                            pass
                        try:
                            if isinstance(ar, (int, float)):
                                cred_parts.append(f"{float(ar):.1f} rating")
                        except Exception:
                            pass
                        try:
                            if isinstance(ac, (int, float)):
                                cred_parts.append(f"{int(ac):,} articles")
                        except Exception:
                            pass
                        if cred_parts:
                            header = f"{header}\nCredibility: " + " | ".join(cred_parts) if header else "Credibility: " + " | ".join(cred_parts)

                        if header:
                            items.append({"text": header, "pages": [], "is_header": True})

                        # Split into paragraphs; filter tiny fragments
                        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
                        # Limit paragraphs per article to avoid huge OpenAI scoring cost.
                        kept_in_article = 0
                        for p in paras:
                            if kept_in_article >= max_paras_per_article:
                                break
                            if len(p) < 60:
                                continue
                            items.append({"text": p, "pages": []})
                            kept_in_article += 1

                    if items:
                        # Keep only up to max_paras_per_ticker body paragraphs per ticker (headers excluded).
                        trimmed: list[dict] = []
                        body_count = 0
                        for it in items:
                            txt = (it.get("text") or "")
                            # treat short lines / headers as non-body
                            if txt.startswith("Source:") or ("\nSource:" in txt) or len(txt) < 120:
                                trimmed.append(it)
                                continue
                            if body_count >= max_paras_per_ticker:
                                break
                            trimmed.append(it)
                            body_count += 1
                        combined[sym] = trimmed

                if not combined:
                    raise RuntimeError("No article bodies available to export for the selected tickers.")

                # Build one compiled PDF (same visual language as Fund Families)
                pdf_name = f"{datetime.now(ZoneInfo('America/New_York')).strftime('%m.%d.%y')} Seeking Alpha {'_'.join(tickers_for_pdf)}.pdf"

                from pathlib import Path
                with tempfile.TemporaryDirectory() as td:
                    td_path = Path(td)
                    excerpts_path = td_path / "sa_excerpts.json"
                    excerpts_path.write_text(_json.dumps(combined, indent=2), encoding="utf-8")

                    out_pdf = td_path / "sa_compiled.pdf"
                    progress.progress(0.92)
                    status.info("Rendering compiled PDF (Fund Families style)…")
                    make_pdf.build_pdf(
                        excerpts_json_path=str(excerpts_path),
                        output_pdf_path=str(out_pdf),
                        report_title="Seeking Alpha Analysis",
                        source_pdf_name=pdf_name,
                        format_style="compact",
                        ai_score=True,
                        ai_model="heuristic",
                        include_index=True,
                        index_label="Index — Hit Tickers",
                    )
                    st.session_state["sa_pdf_bytes"] = out_pdf.read_bytes()
                    st.session_state["sa_pdf_name"] = pdf_name

                progress.progress(1.0)
                status.empty()
                st.success("Seeking Alpha PDF built successfully.")
            except Exception as e:
                progress.empty()
                status.empty()
                st.session_state["sa_pdf_bytes"] = None
                st.session_state["sa_pdf_name"] = None
                st.error(f"Seeking Alpha PDF build failed: {e}")

    if st.session_state.get("sa_pdf_bytes") and st.session_state.get("sa_pdf_name"):
        st.download_button(
            "Download Seeking Alpha PDF",
            data=st.session_state["sa_pdf_bytes"],
            file_name=st.session_state["sa_pdf_name"],
            mime="application/pdf",
            key=f"sa_download_pdf_{ck_current}",
            use_container_width=True,
        )


def _get_available_tickers_for_substack() -> list[str]:
    """
    Return available tickers from tickers.py for the Substack segment.
    Mirrors the prior segment behavior: use the same Cutler universe without hardcoding.
    """
    try:
        from tickers import tickers  # { "AAPL": {...}, ... }
        if isinstance(tickers, dict):
            return sorted(tickers.keys())
    except Exception:
        pass
    return []


def draw_substack_section():
    """
    Substack (RapidAPI) segment:
      - ticker-driven search
      - strict lookback
      - per-ticker caps to control cost
      - compiled PDF export (skimmable, investment-grade) via make_pdf.py
    """
    st.markdown("### Substack — recent posts (RapidAPI)")
    st.caption(
        "We query Substack for recent, ticker-relevant posts and export a skimmable compiled PDF. "
        "Cost controls: strict lookback + per-ticker caps + we only fetch full post bodies for candidates "
        "that pass the list-stage filters."
    )

    # Universe
    available_tickers = _get_available_tickers_for_substack()
    default_index = available_tickers.index("AMZN") if "AMZN" in available_tickers else 0

    # Inputs
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_ticker = st.selectbox(
            "Ticker symbol",
            options=available_tickers,
            index=default_index,
            key="substack_ticker_select",
        )
    with col2:
        lookback_days = st.selectbox(
            "Lookback window (days)",
            options=[2, 7],
            index=0,
            key="substack_lookback_days",
        )
    with col3:
        max_posts = st.number_input(
            "Max posts per ticker",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("substack_max_posts", 3) or 3),
            step=1,
            key="substack_max_posts",
        )

    # Simple cache (avoid rerun API hits)
    cache_key = "substack_cache"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}

    cached = st.session_state[cache_key].get(selected_ticker)

    # Fetch button — only call API here
    run_clicked = st.button(
        f"Fetch Substack posts for {selected_ticker}",
        use_container_width=True,
        key="substack_fetch_button",
    )

    if run_clicked:
        with st.spinner("Querying Substack…"):
            try:
                cached = substack_excerpts.fetch_posts_for_ticker(
                    selected_ticker,
                    lookback_days=int(lookback_days),
                    max_posts=int(max_posts),
                )
            except Exception as e:
                cached = []
                st.error(f"Substack fetch failed: {e}")
        st.session_state[cache_key][selected_ticker] = cached

    if not cached:
        st.info(
            "No recent Substack posts found for this ticker in the selected lookback window. "
            "Try 7 days or a higher cap if you want broader coverage."
        )
    else:
        st.markdown(f"Showing **{len(cached)}** post(s) for **{selected_ticker}**:")
        for post in cached:
            title = (post.get("title") or "(no title)").strip()
            author = (post.get("author") or "").strip()
            published = (post.get("published_at") or "").strip()
            url = (post.get("url") or "").strip()
            excerpt = (post.get("excerpt") or post.get("body") or "").strip()

            meta = []
            if author:
                meta.append(author)
            if published:
                meta.append(published[:19])
            meta_s = " · ".join(meta)
            header = f"{title}" + (f" — {meta_s}" if meta_s else "")

            with st.expander(header, expanded=False):
                if url:
                    st.markdown(f"[Open on Substack]({url})")
                if excerpt:
                    max_chars = 4500
                    if len(excerpt) > max_chars:
                        st.write(excerpt[:max_chars] + " …")
                        st.caption("Truncated — open on Substack to read the full post.")
                    else:
                        st.write(excerpt)
                else:
                    st.caption("No body text available for this item.")

    st.markdown("---")

    # Compiled PDF export (universe-wide, like SA)
    st.markdown("#### Export compiled Substack PDF")
    st.caption(
        "Build one compiled PDF across your full ticker universe. "
        "This will re-run the Substack search for each ticker using the lookback and caps above."
    )

    if "substack_pdf_bytes" not in st.session_state:
        st.session_state["substack_pdf_bytes"] = None
        st.session_state["substack_pdf_name"] = None

    export_clicked = export_clicked = st.button(
        "Build Substack compiled PDF (ALL tickers)",
        use_container_width=True,
        key="substack_export_all_button",
    )

    # Resumable ALL-tickers Substack build (prevents long runs from resetting)
    run_key = "substack_all_run_state"
    state = st.session_state.get(run_key)

    # Show a Resume button if a run is in progress but not actively running (e.g., script reran)
    resume_clicked = False
    if state and (not state.get("done")) and (not state.get("running")):
        resume_clicked = st.button(
            "Resume Substack build",
            use_container_width=True,
            key="substack_export_all_resume_button",
        )

    # Allow a stop/reset while running
    stop_clicked = False
    if state and state.get("running"):
        stop_clicked = st.button(
            "Stop / Reset Substack build",
            use_container_width=True,
            key="substack_export_all_stop_button",
        )

    if stop_clicked:
        st.session_state.pop(run_key, None)
        st.session_state.pop("substack_pdf_bytes", None)
        st.success("Stopped Substack build.")
        _substack_rerun()

    if export_clicked or resume_clicked:
        universe = available_tickers or ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "ABBV"]

        # Start fresh if no state, state is done, or parameters changed
        if (not state) or state.get("done") or state.get("universe") != list(universe) or int(state.get("lookback_days", -1)) != int(lookback_days) or int(state.get("max_posts", -1)) != int(max_posts):
            state = _substack_all_run_init(
                universe=universe,
                lookback_days=int(lookback_days),
                max_posts=int(max_posts),
            )
            st.session_state[run_key] = state
        else:
            state["running"] = True

        _substack_rerun()

    # If running, do small work per rerun and checkpoint progress
    if state and state.get("running") and (not state.get("done")):
        prog = st.progress(min(1.0, float(state.get("idx", 0)) / max(1, float(state.get("total", 1)))))
        status = st.empty()
        status.info(f"Substack ALL — {state.get('idx', 0)}/{state.get('total', 0)} tickers processed…")

        if state.get("error"):
            st.warning(state.get("error"))

        with st.spinner("Continuing Substack build…"):
            _substack_all_run_step(state=state, batch_size=int(st.session_state.get("substack_all_batch_size", 3)), time_budget_s=float(st.session_state.get("substack_all_time_budget_s", 12.0)))

        st.session_state[run_key] = state

        # Keep the run moving until done
        if not state.get("done"):
            _substack_rerun()

    # If done, show download button (existing behavior relies on substack_pdf_bytes)

    if st.session_state.get("substack_pdf_bytes") and st.session_state.get("substack_pdf_name"):
        st.download_button(
            "Download Substack compiled PDF",
            data=st.session_state["substack_pdf_bytes"],
            file_name=st.session_state["substack_pdf_name"],
            mime="application/pdf",
            use_container_width=True,
            key="substack_download_button",
        )


def _build_substack_compiled_pdf_for_universe(*, universe: list[str], lookback_days: int, max_posts: int) -> Path:
    """Build one compiled Substack PDF across the given ticker universe."""
    import tempfile
    import json as _json
    from zoneinfo import ZoneInfo

    combined: dict[str, list[dict]] = {}

    # Keep these caps aligned with SA defaults (can be overridden via session_state if needed)
    max_paras_per_ticker = int(st.session_state.get("substack_pdf_max_paras_per_ticker", 2000))
    max_paras_per_post = int(st.session_state.get("substack_pdf_max_paras_per_post", 250))

    prog = st.progress(0.0)
    status = st.empty()

    total = max(1, len(universe))
    processed = 0
    global_seen_post_ids: set[str] = set()

    for sym in universe:
        status.info(f"Substack — fetching {sym} ({processed+1}/{total}) …")
        try:
            posts = substack_excerpts.fetch_posts_for_ticker(
                sym,
                lookback_days=int(lookback_days),
                max_posts=int(max_posts),
            ) or []

            items: list[dict] = []
            for p in posts:
                pid = str(p.get("post_id") or "").strip()
                if pid:
                    if pid in global_seen_post_ids:
                        continue
                    global_seen_post_ids.add(pid)

                title = (p.get("title") or "").strip()
                author = (p.get("author") or "").strip()
                url = (p.get("url") or "").strip()
                published = (p.get("published_at") or "").strip()
                body = (p.get("body") or p.get("excerpt") or "").strip()

                if not body and not title:
                    continue

                header_parts = []
                if title:
                    header_parts.append(title)
                if author:
                    header_parts.append(f"— {author}")
                header = " ".join(header_parts).strip()

                meta_lines = []
                if published:
                    meta_lines.append(f"Date: {published[:19]}")
                if url:
                    meta_lines.append(f"Source: {url}")
                if meta_lines:
                    header = (header + "\n" if header else "") + "\n".join(meta_lines)

                header_added = False
                if header:
                    items.append({"text": header, "pages": [], "is_header": True})
                header_added = True
                # Paragraph-only extraction: include ONLY paragraphs that credibly mention the ticker.
                # Always run a second-pass filter/rerank at render time to avoid low-signal upstream
                # hit_paragraphs (event calendars, tag blocks, leaderboards, etc.).
                hit_paras = p.get("hit_paragraphs")
                if not isinstance(hit_paras, list):
                    hit_paras = []

                # Candidate text for extraction: prefer full body, but also include any upstream
                # hit snippets (some feeds truncate body fields).
                candidate_parts = []
                if body:
                    candidate_parts.append(body)
                if hit_paras:
                    candidate_parts.append("\n\n".join([str(x) for x in hit_paras if str(x).strip()]))

                candidate_text = "\n\n".join([x for x in candidate_parts if x]).strip()

                paras = []
                if candidate_text:
                    try:
                        paras = substack_excerpts.extract_ticker_paragraphs(
                            body_text=candidate_text,
                            ticker=str(sym),
                        ) or []
                    except Exception:
                        paras = hit_paras or []
                else:
                    paras = hit_paras or []

                # Final safety filters (best-effort) for common Substack noise patterns
                import re as _re
                cleaned_paras = []
                for _p in paras:
                    _t = str(_p).strip()
                    if not _t:
                        continue
                    _low = _t.lower()

                    if (
                        ("asset-types:" in _low)
                        or ("entropy:" in _low)
                        or ("staleness:" in _low)
                        or ("uncertainty:" in _low)
                        or ("sentiment:" in _low)
                    ):
                        continue

                    if (
                        ("subscribe" in _low)
                        or ("share" in _low)
                        or ("read original article" in _low)
                        or ("full analysis" in _low and len(_t) < 250)
                    ):
                        continue

                    if ("this week" in _low and "events" in _low) or (_t.count("■") + _t.count("❤") >= 4):
                        continue
                    if len(_re.findall(r"(?:mon|tue|wed|thu|fri|sat|sun)", _low)) >= 3 and "feb" in _low:
                        continue

                    if _t.count("$") >= 3 and (
                        ("relative volume" in _low) or ("52-week" in _low) or ("peak gain" in _low)
                    ):
                        continue

                    cleaned_paras.append(_t)

                paras = cleaned_paras
                # Skip empty posts (no qualifying paragraphs after filters)
                if not paras:
                    if header_added and items and items[-1].get('is_header'):
                        items.pop()
                    continue
                kept = 0
                for para in paras:
                    if kept >= max_paras_per_post:
                        break
                    if len(para) < 60:
                        continue
                    items.append({"text": para, "pages": []})
                    kept += 1
                if kept == 0:
                    if header_added and items and items[-1].get('is_header'):
                        items.pop()
                    continue

            if items:
                trimmed: list[dict] = []
                body_count = 0
                for it in items:
                    if it.get('is_header'):
                        trimmed.append(it)
                        continue
                    if body_count >= max_paras_per_ticker:
                        break
                    trimmed.append(it)
                    body_count += 1
                # Only include tickers that contributed at least one body paragraph
                if body_count > 0:
                    combined[sym] = trimmed
        except Exception:
            pass

        processed += 1
        prog.progress(processed / total)

    prog.empty()
    status.empty()
    if not combined:
        combined = {"—": [{"text": "No qualifying Substack excerpts found for the selected lookback window and filters.\nTry a longer lookback or higher cap if you want broader coverage.", "pages": []}]}

    now_et = datetime.now(ZoneInfo("America/New_York"))
    pdf_name = f"{now_et.strftime('%m.%d.%y')} Substack ALL.pdf"
    out_path = CP_DIR / pdf_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        excerpts_path = td_path / "substack_excerpts.json"
        excerpts_path.write_text(_json.dumps(combined, indent=2), encoding="utf-8")
        out_pdf = td_path / "substack_compiled.pdf"

        make_pdf.build_pdf(
            excerpts_json_path=str(excerpts_path),
            output_pdf_path=str(out_pdf),
            report_title="Substack Research",
            source_pdf_name=pdf_name,
            format_style="compact",
            ai_score=True,
            ai_model="heuristic",
            include_index=True,
            index_label="Index — Hit Tickers",
        )
        out_path.write_bytes(out_pdf.read_bytes())

    return out_path

def _substack_rerun():
    """Compatibility wrapper for Streamlit rerun."""
    try:
        st.rerun()
    except Exception:
        # older Streamlit
        st.experimental_rerun()


def _render_substack_compiled_pdf_from_combined(*, combined: dict[str, list[dict]], pdf_name: str) -> Path:
    """Render compiled Substack PDF using make_pdf, returning the final output path."""
    import tempfile
    import json as _json
    from zoneinfo import ZoneInfo

    out_path = Path.cwd() / pdf_name

    # If filters are very strict, it's possible to have zero qualifying hits.
    # We still generate a small PDF so the UI shows a download button (and Run All can proceed).
    if not combined or not any((v or []) for v in combined.values()):
        combined = {"—": [{"text": "No qualifying Substack excerpts found for the selected lookback window and filters.\nTry a longer lookback or higher cap if you want broader coverage.", "pages": []}]}

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        excerpts_path = td_path / "substack_excerpts.json"
        excerpts_path.write_text(_json.dumps(combined, indent=2), encoding="utf-8")
        out_pdf = td_path / "substack_compiled.pdf"

        make_pdf.build_pdf(
            excerpts_json_path=str(excerpts_path),
            output_pdf_path=str(out_pdf),
            report_title="Substack Research",
            source_pdf_name=pdf_name,
            format_style="compact",
            ai_score=True,
            ai_model="heuristic",
            include_index=True,
            index_label="Index — Hit Tickers",
        )
        out_path.write_bytes(out_pdf.read_bytes())

    return out_path


def _substack_all_run_init(*, universe: list[str], lookback_days: int, max_posts: int) -> dict:
    """Initialize resumable ALL-tickers Substack build state."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    now_local = datetime.now(ZoneInfo("America/New_York"))
    pdf_name = f"{now_local.strftime('%m.%d.%y')} Substack ALL.pdf"

    state = {
        "running": True,
        "done": False,
        "error": "",
        "universe": list(universe),
        "lookback_days": int(lookback_days),
        "max_posts": int(max_posts),
        "idx": 0,
        "total": len(universe),
        "combined": {},          # ticker -> list[dict] paragraphs
        "seen_post_ids": set(),  # global dedupe across tickers
        "pdf_name": pdf_name,
        "started_at": now_local.isoformat(),
    }
    return state


def _substack_all_run_step(*, state: dict, batch_size: int = 3, time_budget_s: float = 12.0) -> None:
    """
    Process a small batch of tickers and checkpoint progress into state.
    This keeps the Streamlit script responsive and resumable.
    """
    import time as _time
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if state.get("done"):
        state["running"] = False
        return

    universe: list[str] = state["universe"]
    total = max(1, int(state.get("total") or len(universe)))
    idx = int(state.get("idx") or 0)

    # Keep these caps aligned with SA defaults (can be overridden via session_state if needed)
    max_paras_per_ticker = int(st.session_state.get("substack_pdf_max_paras_per_ticker", 2000))
    max_paras_per_post = int(st.session_state.get("substack_pdf_max_paras_per_post", 250))

    combined: dict[str, list[dict]] = state.get("combined") or {}
    seen_post_ids: set = state.get("seen_post_ids") or set()

    start_t = _time.time()
    processed_this_step = 0

    while idx < len(universe) and processed_this_step < batch_size:
        if (_time.time() - start_t) > float(time_budget_s):
            break

        sym = (universe[idx] or "").strip().upper()
        idx += 1

        if not sym:
            continue

        try:
            posts = substack_excerpts.fetch_posts_for_ticker(
                sym,
                lookback_days=int(state["lookback_days"]),
                max_posts=int(state["max_posts"]),
            ) or []
        except Exception as e:
            # Keep going; store error for visibility but do not crash the run.
            state["error"] = f"{sym}: {e}"
            posts = []

        items: list[dict] = []
        for post in posts:
            pid = str(post.get("post_id") or "").strip()
            if not pid or pid in seen_post_ids:
                continue
            seen_post_ids.add(pid)

            title = (post.get("title") or "").strip()
            author = (post.get("author") or "").strip()
            published = (post.get("published_at") or "").strip()
            url = (post.get("url") or "").strip()
            body = (post.get("body") or post.get("excerpt") or "").strip()

            header = title or "(Untitled)"
            meta_lines = []
            if author:
                meta_lines.append(f"Author: {author}")
            if published:
                meta_lines.append(f"Date: {published[:19]}")
            if url:
                meta_lines.append(f"Source: {url}")
            if meta_lines:
                header = header + "\n" + "\n".join(meta_lines)

            items.append({"text": header, "pages": [], "is_header": True})

            # Paragraph-only extraction: include ONLY paragraphs that credibly mention the ticker.
            hit_paras = post.get("hit_paragraphs")
            if not isinstance(hit_paras, list):
                hit_paras = None

            if hit_paras is None and body:
                try:
                    hit_paras = substack_excerpts.extract_ticker_paragraphs(
                        body_text=body,
                        ticker=str(sym),
                    )
                except Exception:
                    hit_paras = None

            paras = [str(x).strip() for x in (hit_paras or []) if str(x).strip()] if hit_paras else []

            kept = 0
            for para in paras:
                if kept >= max_paras_per_post:
                    break
                if len(para) < 60:
                    continue
                items.append({"text": para, "pages": []})
                kept += 1

        if items:
            trimmed: list[dict] = []
            body_count = 0
            for it in items:
                if it.get("is_header"):
                    trimmed.append(it)
                    continue
                if body_count >= max_paras_per_ticker:
                    break
                trimmed.append(it)
                body_count += 1

            combined[sym] = trimmed

        processed_this_step += 1

    # checkpoint back
    state["idx"] = idx
    state["combined"] = combined
    state["seen_post_ids"] = seen_post_ids

    if idx >= len(universe):
        # render final PDF
        try:
            out_path = _render_substack_compiled_pdf_from_combined(
                combined=combined,
                pdf_name=state["pdf_name"],
            )
            st.session_state["substack_pdf_bytes"] = Path(out_path).read_bytes()
            st.session_state["substack_pdf_name"] = state.get("pdf_name")
            state["done"] = True
            state["running"] = False
        except Exception as e:
            state["error"] = f"PDF render failed: {e}"
            state["done"] = True
            state["running"] = False


def build_podcast_groups(n_groups: int = 9):
    """
    Split PODCASTS into up to n_groups buckets.
    Each bucket is shown as a label -> list of podcast_ids.
    Assumes each Podcast has .id and .name.
    """
    if not PODCASTS:
        return {}

    size = math.ceil(len(PODCASTS) / n_groups)
    groups = {}
    for i in range(n_groups):
        start = i * size
        chunk = PODCASTS[start:start + size]
        if not chunk:
            break

        # Nice label like: "Set 1: Bloomberg Surveillance, Odd Lots, Money For The Rest..."
        preview_names = [p.name for p in chunk[:4]]
        label = f"Set {i+1}: " + ", ".join(preview_names)
        if len(chunk) > 4:
            label += " ..."

        groups[label] = [p.id for p in chunk]

    return groups

def run_podcast_pipeline_from_ui(
    days_back: int,
    podcast_ids,
    podcasts_root: Path,
    excerpts_path: Path,
    insights_path: Path,
    model_name: str = "gpt-4o-mini",
):
    """
    Orchestrates:
      1) podcast_ingest.py  -> transcripts under podcasts_root
      2) podcast_excerpts.py -> excerpts JSON (including _episodes)
      3) podcast_insights.py -> insights JSON (list of dicts)

    This mirrors the CLI runs you've already tested, but wired into Streamlit.
    """
    base_dir = Path(__file__).parent

    # Make sure output dir exists, BUT clear old run first
    if podcasts_root.exists():
        import shutil
        shutil.rmtree(podcasts_root)
    podcasts_root.mkdir(parents=True, exist_ok=True)

    # -------- 1) INGEST: download + Whisper --------
    ingest_cmd = [
        sys.executable,
        str(base_dir / "podcast_ingest.py"),
        "--out", str(podcasts_root),
        "--days", str(days_back),
    ]

    # limit to chosen podcast IDs (batch) if provided
    if podcast_ids:
        ingest_cmd += ["--podcasts"] + list(podcast_ids)

    # always use Whisper in this UI flow
    ingest_cmd.append("--whisper")

    ingest_proc = subprocess.run(
        ingest_cmd,
        text=True,
        capture_output=True,
    )
    if ingest_proc.returncode != 0:
        raise RuntimeError(f"podcast_ingest.py failed:\n{ingest_proc.stderr}")

    # -------- 2) EXCERPTS: company-specific snippets + _episodes --------
    excerpts_cmd = [
        sys.executable,
        str(base_dir / "podcast_excerpts.py"),
        "--root", str(podcasts_root),
        "--out", str(excerpts_path),
        "--window", "2",  # sentence window around mentions
        # no --tickers: script will use full Cutler universe from tickers.py
    ]

    excerpts_proc = subprocess.run(
        excerpts_cmd,
        text=True,
        capture_output=True,
    )
    if excerpts_proc.returncode != 0:
        raise RuntimeError(f"podcast_excerpts.py failed:\n{excerpts_proc.stderr}")

    # -------- 3) INSIGHTS: call GPT on excerpts + episodes --------
    insights_cmd = [
        sys.executable,
        str(base_dir / "podcast_insights.py"),
        "--in", str(excerpts_path),
        "--out", str(insights_path),
        "--model", model_name,
    ]
    insights_proc = subprocess.run(
        insights_cmd,
        text=True,
        capture_output=True,
    )
    if insights_proc.returncode != 0:
        raise RuntimeError(f"podcast_insights.py failed:\n{insights_proc.stderr}")

    return {
        "ingest_stdout": ingest_proc.stdout,
        "excerpts_stdout": excerpts_proc.stdout,
        "insights_stdout": insights_proc.stdout,
    }

# -------------------------------------------------------------------
#  Podcast Run-All helpers (merge group outputs + avoid long blocking runs)
# -------------------------------------------------------------------

def _load_json_safe(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def _save_json_safe(path: Path, payload) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Best-effort persistence; do not fail the run.
        return


def _merge_podcast_excerpts_dict(dest: dict, src: dict) -> dict:
    """Merge excerpt dicts of the form {ticker: [snips...], '_episodes': [...]}"""
    if not isinstance(dest, dict):
        dest = {}
    if not isinstance(src, dict):
        return dest

    # merge episodes
    eps_dest = dest.get("_episodes") if isinstance(dest.get("_episodes"), list) else []
    eps_src = src.get("_episodes") if isinstance(src.get("_episodes"), list) else []
    seen_ep = set()
    merged_eps = []
    for e in (eps_dest + eps_src):
        if not isinstance(e, dict):
            continue
        # prefer stable identifiers if present
        key = (
            str(e.get("episode_id") or e.get("id") or "")
            + "|"
            + str(e.get("podcast_id") or "")
            + "|"
            + str(e.get("url") or e.get("link") or "")
        ).strip("|")
        if not key or key in seen_ep:
            continue
        seen_ep.add(key)
        merged_eps.append(e)
    if merged_eps:
        dest["_episodes"] = merged_eps

    # merge tickers
    for k, v in src.items():
        if k == "_episodes":
            continue
        if not isinstance(v, list):
            continue
        cur = dest.get(k)
        if not isinstance(cur, list):
            cur = []
        # simple dedupe by normalized string form
        seen = set()
        merged = []
        for item in (cur + v):
            s = None
            if isinstance(item, str):
                s = item.strip()
            elif isinstance(item, dict):
                s = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            merged.append(item)
        if merged:
            dest[k] = merged
    return dest


def _merge_podcast_insights_list(dest, src):
    """Merge insights lists (list[dict]) via stable JSON string dedupe."""
    if not isinstance(dest, list):
        dest = []
    if not isinstance(src, list):
        return dest
    seen = set()
    out = []
    for item in (dest + src):
        if not isinstance(item, dict):
            continue
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _podcast_run_all_group_ids(n_groups: int = 9) -> list[list[str]]:
    """Return podcast_id groups similar to the Podcast tab grouping."""
    try:
        groups = build_podcast_groups(n_groups=n_groups)
        return [list(v or []) for _, v in groups.items()]
    except Exception:
        return []



def draw_podcast_intelligence_section():
    st.subheader("Podcast Intelligence – Company mentions across finance podcasts")

    # -------------------------
    # 0) Session cache init
    # -------------------------
    if "podcast_cache" not in st.session_state:
        # cache_key -> {"excerpts_path": str, "insights_path": str, "logs": {...}}
        st.session_state["podcast_cache"] = {}
    if "podcast_last_cache_key" not in st.session_state:
        st.session_state["podcast_last_cache_key"] = None

    # --- 1) Choose podcast group (9 buckets) ---
    podcast_groups = build_podcast_groups(n_groups=9)
    group_labels = list(podcast_groups.keys())
    if not group_labels:
        st.info("No podcasts configured yet. Check podcasts_config.PODCASTS.")
        return

    selected_group_label = st.selectbox("Podcasts", group_labels)
    selected_podcast_ids = podcast_groups.get(selected_group_label, [])

    # Show all podcasts in this batch as "bubbles"
    if selected_podcast_ids:
        st.markdown("**Podcasts in this batch:**")
        bubble_html = ""
        for pod_id in selected_podcast_ids:
            bubble_html += (
                "<span style='display:inline-block; margin:2px 6px 4px 0; "
                "padding:4px 10px; border-radius:999px; "
                "border:1px solid #999; font-size:0.85rem;'>"
                f"{pod_id}</span>"
            )
        st.markdown(bubble_html, unsafe_allow_html=True)

    # --- 2) Date range -> converted to days_back ---
    today = datetime.now(timezone.utc).date()
    default_start = today - timedelta(days=2)

    date_input_value = st.date_input(
        "Episode date range",
        value=(default_start, today),
        format="YYYY/MM/DD",
    )

    if not isinstance(date_input_value, (list, tuple)) or len(date_input_value) != 2:
        st.error("Please select a valid start and end date.")
        return

    date_from, date_to = date_input_value

    if date_from > date_to:
        st.error("Start date must be on or before end date.")
        return

    days_back = max(1, (date_to - date_from).days + 1)
    st.caption(
        f"{days_back} day lookback window "
        f"({date_from.isoformat()} – {date_to.isoformat()})."
    )

    # -------------------------
    # 2.5) Previous-run picker
    # -------------------------
    podcast_cache = st.session_state["podcast_cache"]

    # list only keys for this group
    prev_keys_for_group = [
        k for k in podcast_cache.keys()
        if k.startswith(selected_group_label + "|")
    ]

    chosen_prev_key = None
    if prev_keys_for_group:
        chosen_prev_key = st.selectbox(
            "Load a previous run (this session)",
            options=["(none)"] + prev_keys_for_group,
            index=0,
            help="Pick a prior batch/date run to view without re-running.",
        )
        if chosen_prev_key == "(none)":
            chosen_prev_key = None

    # --- Paths for pipeline outputs ---
    base_dir = Path(__file__).parent
    podcasts_root = base_dir / "data" / "podcasts_ui"

    # IMPORTANT: per-run JSON so old runs don't get overwritten
    safe_group = re.sub(r"[^A-Za-z0-9_-]+", "_", selected_group_label)[:60]
    safe_from = date_from.isoformat()
    safe_to = date_to.isoformat()

    excerpts_path = base_dir / "data" / f"podcast_excerpts_ui_{safe_group}_{safe_from}_{safe_to}.json"
    insights_path = base_dir / "data" / f"podcast_insights_ui_{safe_group}_{safe_from}_{safe_to}.json"

    cache_key = f"{selected_group_label}|{safe_from}|{safe_to}"

    # --- 3) Run pipeline button ---
    run_clicked = st.button("Run fresh podcast analysis")

    if run_clicked:
        try:
            with st.spinner("Downloading podcasts, transcribing, and analyzing..."):
                logs = run_podcast_pipeline_from_ui(
                    days_back=days_back,
                    podcast_ids=selected_podcast_ids,
                    podcasts_root=podcasts_root,
                    excerpts_path=excerpts_path,
                    insights_path=insights_path,
                    model_name="gpt-4o-mini",
                )
            st.success("Podcast analysis updated.")

            # store in cache
            podcast_cache[cache_key] = {
                "excerpts_path": str(excerpts_path),
                "insights_path": str(insights_path),
                "logs": logs,
            }
            st.session_state["podcast_last_cache_key"] = cache_key

            with st.expander("Show pipeline logs"):
                st.text("=== podcast_ingest.py ===")
                st.code((logs.get("ingest_stdout") or "")[-2000:])
                st.text("=== podcast_excerpts.py ===")
                st.code((logs.get("excerpts_stdout") or "")[-2000:])
                st.text("=== podcast_insights.py ===")
                st.code((logs.get("insights_stdout") or "")[-2000:])

        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            return

    # -------------------------
    # 3.5) Decide what to show
    # -------------------------
    active_key = None
    if run_clicked:
        active_key = cache_key
    elif chosen_prev_key:
        active_key = chosen_prev_key
    else:
        # Nothing selected and not run in this rerun -> show clean slate
        st.info("Run the analysis first, or load a previous run to see results.")
        return

    active_entry = podcast_cache.get(active_key)
    if not active_entry:
        st.info("Selected run is not available in cache. Please re-run.")
        return

    excerpts_path = Path(active_entry["excerpts_path"])
    insights_path = Path(active_entry["insights_path"])

    if not insights_path.exists() or not excerpts_path.exists():
        st.info("Run the analysis first to see podcast intelligence.")
        return

    # --- 4) Display results from the chosen JSON ---

    # Load excerpts
    try:
        with open(excerpts_path, "r", encoding="utf-8") as f:
            excerpts_data = json.load(f)
    except Exception as e:
        st.error(f"Could not read excerpts JSON: {e}")
        return

    # Load insights
    try:
        with open(insights_path, "r", encoding="utf-8") as f:
            raw_insights = json.load(f)
    except Exception as e:
        st.error(f"Could not read podcast insights JSON: {e}")
        return

    if isinstance(raw_insights, dict):
        company_insights_list = raw_insights.get("company_insights", []) or []
        episode_summaries_list = raw_insights.get("__episode_summaries", []) or []
    else:
        company_insights_list = raw_insights or []
        episode_summaries_list = []
        for ins in company_insights_list:
            for ep in ins.get("episode_summaries", []) or []:
                episode_summaries_list.append(ep)

    ticker_insights: Dict[str, dict] = {}
    for ins in company_insights_list:
        t = ins.get("ticker")
        if t:
            ticker_insights[t] = ins

    all_episode_records: Dict[str, dict] = {}
    for ep in episode_summaries_list:
        eid = ep.get("episode_uid") or (
            f"{ep.get('podcast_id','')}|"
            f"{ep.get('title','')}|"
            f"{ep.get('published','')}"
        )
        all_episode_records[eid] = {
            "episode_uid": eid,
            **ep,
        }

    tickers_with_mentions = set()
    for t, ins in ticker_insights.items():
        if ins.get("has_mentions"):
            tickers_with_mentions.add(t)
        elif len(excerpts_data.get(t, [])) > 0:
            tickers_with_mentions.add(t)

    has_any_mentions = bool(tickers_with_mentions)

    # MODE 1: Company-centric
    if has_any_mentions:
        def _format_company_label(sym: str) -> str:
            names = tickers.get(sym, [])
            primary = names[0] if names else sym
            return f"{primary} ({sym})"

        candidate_tickers = sorted(tickers_with_mentions)
        if candidate_tickers:
            labels = [_format_company_label(t) for t in candidate_tickers]
            label_to_ticker = dict(zip(labels, candidate_tickers))

            selected_label = st.selectbox(
                "Company to inspect",
                options=labels,
                index=0,
                key=f"pod_company_{active_key}",
            )
            selected_ticker = label_to_ticker[selected_label]

            snippets = excerpts_data.get(selected_ticker, []) or []
            insight_for_ticker = ticker_insights.get(selected_ticker, {})

            filtered_snippets = []
            for s in snippets:
                try:
                    pub_dt = datetime.fromisoformat(s.get("published", ""))
                    pub_date = pub_dt.date()
                except Exception:
                    continue

                if not (date_from <= pub_date <= date_to):
                    continue
                if selected_podcast_ids and s.get("podcast_id") not in selected_podcast_ids:
                    continue
                filtered_snippets.append(s)

            st.markdown(
                f"{len(filtered_snippets)} snippet(s) found for **{selected_label}** "
                f"between {date_from} and {date_to}."
            )

            st.markdown("### AI Podcast Stance")

            stance_label = insight_for_ticker.get("stance", "Unknown / Mixed")
            stance_summary = insight_for_ticker.get(
                "overall_summary",
                "No stance summary is available yet for this company.",
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown("**Stance**")
                st.markdown(stance_label)
            with col2:
                st.markdown("**Summary across recent podcast mentions**")
                st.write(stance_summary)


            # --- Download: Podcast report PDF for current selection ---
            if st.button("Build downloadable Podcast PDF", key="pod_build_pdf"):
                now_et = _now_et()
                tkr = (insight_for_ticker.get("ticker") or selected_label or "Podcast").strip()
                safe_tkr = _safe(tkr)
                out_name = f"{now_et:%m.%d.%y} Podcast {safe_tkr}.pdf"
                out_path = (BASE / "Podcasts" / out_name)

                # Build sections: stance summary + up to 25 evidence snippets
                sections: list[tuple[str, str]] = []
                sections.append((f"{tkr} – AI Podcast Stance", f"{stance_label}\n\n{stance_summary}"))

                # Snippet evidence
                ev_lines: list[str] = []
                for i, sn in enumerate(filtered_snippets[:25], 1):
                    pod_name = sn.get("podcast_name") or sn.get("podcast") or sn.get("podcast_id") or ""
                    ep_title = sn.get("episode_title") or sn.get("episode") or sn.get("title") or ""
                    ep_date = sn.get("published_date") or sn.get("date") or sn.get("published") or ""
                    txt = sn.get("text") or sn.get("snippet") or ""
                    header = f"{pod_name} — {ep_title}".strip(" —")
                    line = f"{i}. {header} ({ep_date}){txt}"
                    ev_lines.append(line.strip())
                sections.append(("Episode snippets (evidence)", "\n\n".join(ev_lines) if ev_lines else "No snippets available in this window."))

                subtitle = f"Generated {now_et:%Y-%m-%d %I:%M %p %Z} • Podcasts: {selected_group_label} • Window: {date_from} → {date_to}"
                try:
                    pdf_path = _build_text_pdf(
                        output_path=out_path,
                        title="Cutler Capital – Podcast Intelligence",
                        subtitle=subtitle,
                        sections=sections,
                    )
                    st.session_state["pod_export_pdf_path"] = str(pdf_path)
                    st.success("Podcast PDF is ready.")
                except Exception as e:
                    st.error(f"Could not build Podcast PDF: {e}")

            pod_pdf_path = st.session_state.get("pod_export_pdf_path")
            if pod_pdf_path and Path(pod_pdf_path).exists():
                try:
                    with open(pod_pdf_path, "rb") as f:
                        st.download_button(
                            "Download Podcast PDF",
                            data=f.read(),
                            file_name=Path(pod_pdf_path).name,
                            mime="application/pdf",
                            key="pod_download_pdf",
                        )
                except Exception:
                    st.warning("PDF is built but could not be opened for download.")

            st.markdown("### Episode snippets (evidence)")

            if not filtered_snippets:
                st.info(
                    "No podcast snippets for this company within the selected podcasts "
                    "and date window, even though the model detected mentions in the batch."
                )
                return

            for s in filtered_snippets:
                title = s.get("title", "Untitled episode")
                pod_name = s.get("podcast_name", s.get("podcast_id", ""))
                pub = s.get("published", "")
                header = f"{pod_name} – {title} ({pub[:10]})"

                with st.expander(header):
                    st.write(s.get("snippet", ""))
                    meta_cols = st.columns(3)
                    meta_cols[0].markdown(f"**Podcast**: {pod_name}")
                    meta_cols[1].markdown(f"**Published**: {pub}")
                    if s.get("episode_url"):
                        meta_cols[2].markdown(f"[Open episode]({s['episode_url']})")
            return

    # MODE 2: Episode-centric (no mentions)
    episode_list = list(all_episode_records.values())
    if not episode_list:
        st.warning(
            "No company mentions were detected and no episode summaries are "
            "available yet for this batch."
        )
        return

    st.markdown("### Episode summaries (no company-specific mentions detected in this batch)")

    filtered_eps: List[dict] = []
    for ep in episode_list:
        pub_raw = ep.get("published") or ""
        try:
            pub_dt = datetime.fromisoformat(pub_raw)
            pub_date = pub_dt.date()
        except Exception:
            pub_date = None

        if pub_date and not (date_from <= pub_date <= date_to):
            continue
        if selected_podcast_ids and ep.get("podcast_id") not in selected_podcast_ids:
            continue

        filtered_eps.append(ep)

    if not filtered_eps:
        st.info(
            "Episodes were summarised, but none fall within the selected podcasts "
            "and date window."
        )
        return

    filtered_eps.sort(key=lambda ep: ep.get("published") or "", reverse=True)

    for ep in filtered_eps:
        pod_id = ep.get("podcast_id", "Unknown podcast")
        title = ep.get("title", "Untitled episode")
        pub = (ep.get("published") or "")[:10]
        header = f"{pod_id} – {title} ({pub})"

        with st.expander(header):
            summary = ep.get("summary") or ep.get(
                "overall_summary",
                "No summary is available for this episode yet.",
            )
            st.write(summary)

            meta_cols = st.columns(3)
            meta_cols[0].markdown(f"**Podcast**: {pod_id}")
            meta_cols[1].markdown(f"**Published**: {ep.get('published', '')}")
            if ep.get("episode_url"):
                meta_cols[2].markdown(f"[Open episode]({ep['episode_url']})")

# ---------- Manifest + Delta helpers ----------

def _write_manifest(
    batch: str,
    quarter: str,
    compiled: Optional[Path],
    items: List[Dict[str, Any]],
    table_rows: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Path]:
    """
    Store a small JSON manifest for a compiled (or incremental) run so the
    Document Checker and incremental updater can compare runs later.
    """
    try:
        qdir = MAN_DIR / quarter
        qdir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        payload: Dict[str, Any] = {
            "batch": batch,
            "quarter": quarter,
            "compiled_pdf": str(compiled) if compiled else "",
            "created_at": now.isoformat(timespec="seconds"),
            "items": items,
        }
        if table_rows is not None:
            payload["table_rows"] = table_rows

        fname = qdir / f"manifest_{batch.replace(' ', '')}_{now:%Y%m%d_%H%M%S}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return fname
    except Exception:
        traceback.print_exc()
        return None

def _load_manifests(batch: str, quarter: str) -> List[Dict[str, Any]]:
    """
    Load all manifests for a given batch + quarter, newest first.
    """
    qdir = MAN_DIR / quarter
    if not qdir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in qdir.glob("manifest_*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("batch") != batch or data.get("quarter") != quarter:
                continue
            data["_path"] = str(p)
            out.append(data)
        except Exception:
            continue
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out

def _normalize_para_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())

def _collect_keys_from_manifest(manifest: Dict[str, Any]) -> set:
    """
    Build a set of (fund, source_pdf, ticker, text_hash) keys for all narrative
    paragraphs in a manifest. Used to detect whether a paragraph is "new".
    """
    keys = set()
    for meta in manifest.get("items", []):
        ej = meta.get("excerpts_json")
        if not ej:
            continue
        p = Path(ej)
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for ticker, lst in data.items():
            if not isinstance(lst, list):
                continue
            for item in lst:
                txt_norm = _normalize_para_text(item.get("text", ""))
                if not txt_norm:
                    continue
                h = hashlib.sha1(txt_norm.encode("utf-8")).hexdigest()
                key = (
                    meta.get("fund_family", ""),
                    meta.get("source_pdf_name", ""),
                    str(ticker),
                    h,
                )
                keys.add(key)
    return keys

def build_delta_pdf(old_manifest: Dict[str, Any], new_manifest: Dict[str, Any]) -> Optional[Path]:
    """
    Compare two manifests (older vs newer) and build a PDF containing ONLY
    paragraphs that are new in the newer manifest.

    Returns the PDF path, or None if no new paragraphs were found.
    """
    old_keys = _collect_keys_from_manifest(old_manifest)
    aggregated: Dict[str, List[Dict[str, Any]]] = {}

    for meta in new_manifest.get("items", []):
        ej = meta.get("excerpts_json")
        if not ej:
            continue
        p = Path(ej)
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        for ticker, lst in data.items():
            if not isinstance(lst, list):
                continue
            for item in lst:
                txt = item.get("text", "")
                norm = _normalize_para_text(txt)
                if not norm:
                    continue
                h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
                key = (
                    meta.get("fund_family", ""),
                    meta.get("source_pdf_name", ""),
                    str(ticker),
                    h,
                )
                if key in old_keys:
                    continue  # already existed in the older run

                pages = item.get("pages") or []
                decorated = f"[{meta.get('fund_family', 'Unknown')} – {meta.get('source_pdf_name', '')}] {txt}"
                aggregated.setdefault(str(ticker), []).append(
                    {"text": decorated, "pages": pages}
                )

    if not aggregated:
        return None

    DELTA_DIR.mkdir(parents=True, exist_ok=True)
    batch = new_manifest.get("batch", "Batch")
    quarter = new_manifest.get("quarter", "Quarter")
    old_id = (old_manifest.get("created_at", "old")
              .replace(":", "").replace("-", "").replace("T", "_"))
    new_id = (new_manifest.get("created_at", "new")
              .replace(":", "").replace("-", "").replace("T", "_"))

    json_path = DELTA_DIR / f"delta_{batch.replace(' ', '')}_{quarter}_{old_id}_to_{new_id}.json"
    pdf_path = DELTA_DIR / f"Delta_Cutler_{batch.replace(' ', '')}_{quarter}_{old_id}_to_{new_id}.pdf"

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)

        make_pdf.build_pdf(
            excerpts_json_path=str(json_path),
            output_pdf_path=str(pdf_path),
            report_title="New ticker commentary vs prior run",
            source_pdf_name=f"{batch.replace(' ', '')}_{quarter}_Delta",
            format_style="legacy",
            letter_date=None,
        )
    except Exception:
        traceback.print_exc()
        return None

    return pdf_path if pdf_path.exists() else None

# ---------- Quarter helpers ----------

@st.cache_data(show_spinner=False)
def get_available_quarters() -> List[str]:
    """
    Read the available quarters from the site's <select> element.
    Skips 'all' and 'latest_two'. Returns values like:
      ['2025 Q4', '2025 Q3', '2025 Q2', ...]
    Cached so we don't hit the site on every rerun.
    """
    vals: List[str] = []
    try:
        # Ensure Chromium exists (Streamlit Cloud containers can be wiped)
        ensure_playwright_chromium_installed(show_messages=False)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            ctx = browser.new_context()
            page = ctx.new_page()
            page.set_default_timeout(30000)
            page.goto(BSD_URL)

            sel = page.locator(FILTERS["quarter"]).first
            options = sel.locator("option").all()

            for opt in options:
                val = (opt.get_attribute("value") or "").strip()
                if not val:
                    continue
                if val in ("all", "latest_two"):
                    continue
                vals.append(val)

            browser.close()
    except Exception as e:
        print("WARN: Failed to auto-detect quarters; using fallback list.", e)

    # De-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append(v)

    if not out:
        # Dynamic fallback: include current quarter and a few previous quarters
        now = datetime.now(ZoneInfo("America/New_York"))
        y = now.year
        m = now.month
        if 1 <= m <= 3:
            cur_q = 1
        elif 4 <= m <= 6:
            cur_q = 2
        elif 7 <= m <= 9:
            cur_q = 3
        else:
            cur_q = 4

        out = []
        yy, qq = y, cur_q
        for _ in range(8):
            out.append(f"{yy} Q{qq}")
            qq -= 1
            if qq == 0:
                qq = 4
                yy -= 1

    return out

def _parse_quarter_label(label: str) -> Optional[Tuple[int, int]]:
    """
    Parse 'YYYY QN' into (YYYY, N). Returns None if it doesn't match.
    """
    m = re.match(r"^(\d{4})\s+Q([1-4])$", label.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))

def _last_completed_us_quarter(today: Optional[datetime] = None) -> Tuple[int, int]:
    """
    Given today's date, return (year, quarter) for the **last completed US quarter**.
    US quarters:
      Q1: Jan–Mar
      Q2: Apr–Jun
      Q3: Jul–Sep
      Q4: Oct–Dec
    """
    if today is None:
        today = datetime.now()
    y = today.year
    m = today.month

    if 1 <= m <= 3:
        return y - 1, 4
    elif 4 <= m <= 6:
        return y, 1
    elif 7 <= m <= 9:
        return y, 2
    else:
        return y, 3

def choose_default_quarter(available: List[str]) -> Optional[str]:
    """
    Choose default quarter for UI:
    1) Prefer the **current US quarter** if present (e.g., 2025 Q4 once Q4 starts).
    2) Else fall back to the **last completed US quarter** if present.
    3) Else choose the most recent available.
    """
    if not available:
        return None

    parsed: List[Tuple[str, int, int]] = []
    for lab in available:
        pq = _parse_quarter_label(lab)
        if pq:
            parsed.append((lab, pq[0], pq[1]))

    if not parsed:
        return available[0]

    # Sort by (year DESC, quarter DESC) so index 0 is newest
    parsed.sort(key=lambda x: (x[1], x[2]), reverse=True)

    now = datetime.now(ZoneInfo("America/New_York"))
    m = now.month
    if 1 <= m <= 3:
        cur = (now.year, 1)
    elif 4 <= m <= 6:
        cur = (now.year, 2)
    elif 7 <= m <= 9:
        cur = (now.year, 3)
    else:
        cur = (now.year, 4)

    for lab, year, q in parsed:
        if (year, q) == cur:
            return lab

    target_year, target_q = _last_completed_us_quarter(now)

    for lab, year, q in parsed:
        if year < target_year or (year == target_year and q <= target_q):
            return lab

    return parsed[0][0]

    return parsed[0][0]

# ---------- run one batch (full run, with manifest + table rows) ----------

def run_batch(batch_name: str, quarters: List[str], use_first_word: bool, subset: Optional[List[str]] = None):
    st.markdown(f"### Running {batch_name}")

    # --------- SESSION MEMORY: reuse results if this batch+quarters already ran ---------
    cache_all = st.session_state.get("batch_cache", {})
    cache_entry = cache_all.get(batch_name)

    if cache_entry and cache_entry.get("quarters") == quarters:
        st.info("Using cached results for this batch in this session (no new scraping).")

        for q in quarters:
            qdata = cache_entry["by_quarter"].get(q, {})
            compiled_str = qdata.get("compiled") or ""
            compiled = Path(compiled_str) if compiled_str else None
            manifest_items = qdata.get("manifest_items") or []

            # 1) Compiled excerpt PDF download (if still present)
            if compiled and compiled.exists():
                st.success(f"[Cached] Compiled PDF for {q}: {compiled}")

                try:
                    with open(compiled, "rb") as f:
                        st.download_button(
                            label=f"Download {batch_name} {q} excerpt PDF",
                            data=f.read(),
                            file_name=compiled.name,  # e.g. Batch1_2025-12-04_Excerpt.pdf
                            mime="application/pdf",
                            key=f"download_cached_{batch_name.replace(' ', '')}_{q}".replace('/', '_'),
                        )
                except Exception:
                    st.warning("Cached compiled PDF exists but could not be opened for download.")
            else:
                st.info(
                    f"[Cached] No compiled excerpt PDF found for **{q}**. "
                    "It may have been deleted from disk."
                )

            # 2) Full-letter downloads (original PDFs) from cached manifest_items
            if manifest_items:
                unique_letters: Dict[str, Dict[str, Any]] = {}
                for item in manifest_items:
                    pdf_path = item.get("downloaded_pdf") or ""
                    if not pdf_path:
                        continue
                    if pdf_path in unique_letters:
                        continue
                    unique_letters[pdf_path] = item

                if unique_letters:
                    with st.expander(f"[Cached] View / download full letters for {q}"):
                        for idx, item in enumerate(unique_letters.values(), start=1):
                            pdf_path = Path(item["downloaded_pdf"])
                            label_bits = [
                                item.get("fund_family") or "",
                                item.get("fund_name") or "",
                                item.get("letter_date") or "",
                            ]
                            label_text = " – ".join([b for b in label_bits if b])

                            if pdf_path.exists():
                                try:
                                    with open(pdf_path, "rb") as f:
                                        st.download_button(
                                            label=f"Download full letter #{idx}: {label_text or pdf_path.name}",
                                            data=f.read(),
                                            file_name=pdf_path.name,
                                            mime="application/pdf",
                                            key=f"download_full_cached_{q}_{idx}",
                                        )
                                except Exception:
                                    st.warning(f"Could not open full letter: {pdf_path}")
                            else:
                                st.warning(f"Full letter file not found on disk: {pdf_path}")

        return  # IMPORTANT: skip scraping if we used cache
    # ---------------------------------------------------------------------- END CACHE ----

    # Ensure Chromium exists in this container (auto-install if needed)
    if not ensure_playwright_chromium_installed():
        st.error("Cannot proceed — Chromium is not available.")
        return

    brands = RUNNABLE_BATCHES.get(batch_name, [])
    if subset:
        brands = [b for b in brands if b in subset]
    if not brands:
        st.info("No runnable fund families in this batch (after filter).")
        return
    tokens = [(b, _first_word(b) if use_first_word else b) for b in brands]

    from playwright.sync_api import Error as PlaywrightError

    # We'll fill this and store it into st.session_state at the end
    batch_cache_entry: Dict[str, Any] = {
        "quarters": quarters,
        "by_quarter": {},
    }

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--no-sandbox"],  # important for Streamlit/other PaaS
            )
            ctx = browser.new_context(accept_downloads=True)
            page = ctx.new_page()
            page.set_default_timeout(30000)
            page.goto(BSD_URL)

            for q in quarters:
                st.write(f"Searching quarter {q} across {len(tokens)} fund families…")

                if not _set_quarter(page, q):
                    st.warning(
                        f"Quarter **{q}** is not available on the data source at the moment. "
                        "It may not have any letters yet."
                    )
                    continue

                outs: List[Path] = []
                # NEW: seed outs with any existing excerpt PDFs on disk for this batch+quarter
                existing_outs: List[Path] = []
                existing_root = EX_DIR / q
                if existing_root.exists():
                    for brand in RUNNABLE_BATCHES.get(batch_name, []):
                        base = existing_root / _safe(brand)
                        if not base.exists():
                            continue
                        for pdf in base.rglob("*.pdf"):
                            if pdf.is_file():
                                existing_outs.append(pdf)

                existing_outs_set = {p.resolve() for p in existing_outs}
                outs.extend(existing_outs)

                manifest_items: List[Dict[str, Any]] = []
                table_rows: List[Dict[str, Any]] = []  # snapshot of table rows

                for i, (brand, token) in enumerate(tokens, 1):
                    progress_path = _brand_progress_path(batch_name, q, brand)
                    if progress_path.exists():
                        st.info(f"[{q}] Skipping {brand} (already completed in this container).")
                        continue

                    st.write(f"[{q}] {i}/{len(tokens)} — {brand} (search: {token})")

                    try:
                        _search_by_fund(page, token)
                        hits = _parse_rows(page, q)

                        if not hits:
                            # Brand was searched, but there were no letters for this quarter.
                            # We still want to mark it as completed so we don’t search again.
                            pass
                        else:
                            seen = set()
                            for h in hits:
                                # record table row
                                table_rows.append(
                                    {
                                        "fund_family": brand,
                                        "search_token": token,
                                        "quarter": h.quarter,
                                        "letter_date": h.letter_date,
                                        "fund_name": h.fund_name,
                                        "fund_href": h.fund_href,
                                    }
                                )

                                if h.fund_href in seen:
                                    continue
                                seen.add(h.fund_href)
                                page.goto(h.fund_href)
                                page.wait_for_load_state("domcontentloaded")

                                dest = DL_DIR / q / _safe(brand)
                                pdfs = _download_quarter_pdf_from_fund(page, q, dest)
                                for pdf in pdfs:
                                    out_dir = EX_DIR / q / _safe(brand) / _safe(pdf.stem)
                                    built = run_excerpt_and_build(
                                        pdf,
                                        out_dir,
                                        source_pdf_name=pdf.name,
                                        letter_date=h.letter_date or None,
                                        source_url=h.fund_href,
                                    )

                                    manifest_items.append(
                                        {
                                            "fund_family": brand,
                                            "search_token": token,
                                            "letter_date": h.letter_date or "",
                                            "downloaded_pdf": str(pdf),
                                            "source_pdf_name": pdf.name,
                                            "excerpt_dir": str(out_dir),
                                            "excerpts_json": str(out_dir / "excerpts_clean.json"),
                                            "excerpt_pdf": str(built) if built else "",
                                            "fund_name": h.fund_name,
                                            "fund_href": h.fund_href,
                                        }
                                    )

                                    if built and built.resolve() not in existing_outs_set:
                                        outs.append(built)

                                page.go_back()
                                page.wait_for_load_state("domcontentloaded")

                    except Exception as e:
                        st.error(f"Error on fund family {brand}: {e}")
                        # do NOT mark this brand as done – we want to retry on next run
                        continue

                    # If we reach here, the search for this brand+quarter finished without error,
                    # regardless of whether there were hits or downloads. Mark it as completed.
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    progress_path.write_text(datetime.now().isoformat())

                # 1) Compiled excerpt PDF (BatchN_Date_Excerpt.pdf)
                compiled = compile_merged(batch_name, q, outs, incremental=False)
                if compiled:
                    st.success(f"Compiled PDF for {q}: {compiled}")

                    # Offer direct download so interns always get BatchN_Date_Excerpt.pdf
                    try:
                        with open(compiled, "rb") as f:
                            st.download_button(
                                label=f"Download {batch_name} {q} excerpt PDF",
                                data=f.read(),
                                file_name=compiled.name,  # e.g. Batch1_2025-12-04_Excerpt.pdf
                                mime="application/pdf",
                                key=f"download_{batch_name.replace(' ', '')}_{q}".replace('/', '_'),
                            )
                    except Exception:
                        st.warning("Compiled PDF created but could not be opened for download. Check server logs.")
                else:
                    st.info(
                        f"No excerpt PDFs produced for **{q}**. "
                        "The selected fund families may not yet have letters or ticker mentions for this quarter."
                    )

                # 2) Full-letter downloads (original PDFs)
                if manifest_items:
                    unique_letters: Dict[str, Dict[str, Any]] = {}
                    for item in manifest_items:
                        pdf_path = item.get("downloaded_pdf") or ""
                        if not pdf_path:
                            continue
                        if pdf_path in unique_letters:
                            continue
                        unique_letters[pdf_path] = item

                    if unique_letters:
                        with st.expander(f"View / download full letters for {q}"):
                            for idx, item in enumerate(unique_letters.values(), start=1):
                                pdf_path = Path(item["downloaded_pdf"])
                                label_bits = [
                                    item.get("fund_family") or "",
                                    item.get("fund_name") or "",
                                    item.get("letter_date") or "",
                                ]
                                label_text = " – ".join([b for b in label_bits if b])

                                if pdf_path.exists():
                                    try:
                                        with open(pdf_path, "rb") as f:
                                            st.download_button(
                                                label=f"Download full letter #{idx}: {label_text or pdf_path.name}",
                                                data=f.read(),
                                                file_name=pdf_path.name,
                                                mime="application/pdf",
                                                key=f"download_full_{q}_{idx}",
                                            )
                                    except Exception:
                                        st.warning(f"Could not open full letter: {pdf_path}")
                                else:
                                    st.warning(f"Full letter file not found on disk: {pdf_path}")

                # write manifest regardless (so we capture table_rows snapshot)
                _write_manifest(batch_name, q, compiled, manifest_items, table_rows=table_rows)

                # Save this quarter's results into cache entry
                batch_cache_entry["by_quarter"][q] = {
                    "compiled": str(compiled) if compiled else "",
                    "manifest_items": manifest_items,
                    "table_rows": table_rows,
                }

            browser.close()

    except PlaywrightError as e:
        st.error("Playwright could not start Chromium in this environment.")
        st.code(repr(e))
        return
    except Exception as e:
        st.error("Unexpected error starting Playwright browser.")
        st.code(repr(e))
        return

    # --------- AFTER SUCCESS: persist cache into session_state ---------
    cache_all = st.session_state.get("batch_cache", {})
    cache_all[batch_name] = batch_cache_entry
    st.session_state["batch_cache"] = cache_all

# ---------- NEW: incremental per-batch updater ----------


def run_batch8_latest(quarter_options: List[str], lookback_days: int, use_first_word: bool, *, ensure_compiled_index: bool = False):
    """
    Batch 8 — Latest:
    - Ignore quarter selection; scan the BSD database table for anything published within the last N days.
    - For each hit, download the correct quarter PDF from the fund page (quarter comes from the row).
    - If fund name matches a known batch (1–7), label as '<FundName>_BatchX' for traceability.
    """
    st.markdown(f"### Running {BATCH8_NAME} (last {lookback_days} days)")

    cache_all = st.session_state.get("batch_cache", {})
    cache_key = f"{BATCH8_NAME}|{lookback_days}d"
    cache_entry = cache_all.get(cache_key)

    def _render_latest_results(by_quarter: Dict[str, Dict[str, Any]]) -> None:
        if not by_quarter:
            st.info("No letters found within the selected lookback window.")
            return

        # Newest quarter first
        def _q_sort_key(q: str) -> Tuple[int, int]:
            pq = _parse_quarter_label(q) or (0, 0)
            return pq[0], pq[1]

        for q in sorted(by_quarter.keys(), key=_q_sort_key, reverse=True):
            qdata = by_quarter.get(q) or {}
            compiled_str = qdata.get("compiled") or ""
            compiled = Path(compiled_str) if compiled_str else None
            manifest_items = qdata.get("manifest_items") or []

            if compiled and compiled.exists():
                st.success(f"[{q}] Compiled excerpt PDF ready.")
                try:
                    with open(compiled, "rb") as f:
                        st.download_button(
                            label=f"Download compiled excerpts ({q})",
                            data=f.read(),
                            file_name=compiled.name,
                            mime="application/pdf",
                            key=f"download_compiled_latest_{q}",
                            use_container_width=True,
                        )
                except Exception:
                    st.warning(f"Could not open compiled PDF: {compiled}")

            if manifest_items:
                with st.expander(f"[{q}] Download full letters ({len(manifest_items)})", expanded=False):
                    for idx, row in enumerate(manifest_items):
                        pdf_path = Path(row.get("downloaded_pdf") or "")
                        label_text = row.get("fund_family") or row.get("fund_name") or ""
                        if pdf_path.exists():
                            try:
                                with open(pdf_path, "rb") as f:
                                    st.download_button(
                                        label=f"Download full letter #{idx+1}: {label_text or pdf_path.name}",
                                        data=f.read(),
                                        file_name=pdf_path.name,
                                        mime="application/pdf",
                                        key=f"download_full_latest_{q}_{idx}",
                                    )
                            except Exception:
                                st.warning(f"Could not open full letter: {pdf_path}")
                        else:
                            st.warning(f"Full letter file not found on disk: {pdf_path}")

    # cache reuse (session-only)
    if cache_entry and cache_entry.get("lookback_days") == lookback_days:
        # In Run All, we want to ensure the compiled PDF includes the ticker index page.
        # If results are cached, we can rebuild the compiled PDF from the already-produced excerpt PDFs
        # without re-scraping any websites.
        if ensure_compiled_index:
            try:
                by_q = cache_entry.get("by_quarter") or {}
                for q, qd in (by_q or {}).items():
                    manifest_items = (qd or {}).get("manifest_items") or []
                    excerpt_pdfs = []
                    for row in manifest_items:
                        ep = row.get("excerpt_pdf") or ""
                        if ep:
                            p = Path(ep)
                            if p.exists():
                                excerpt_pdfs.append(p)
                    if excerpt_pdfs:
                        compiled = compile_merged(BATCH8_NAME, q, excerpt_pdfs, incremental=False)
                        if compiled:
                            (qd or {})["compiled"] = str(compiled)
                cache_entry["by_quarter"] = by_q
                cache_all = st.session_state.get("batch_cache", {}) or {}
                cache_all[cache_key] = cache_entry
                st.session_state["batch_cache"] = cache_all
            except Exception:
                pass

        st.info("Using cached results for this latest run in this session (no new scraping).")
        _render_latest_results(cache_entry.get("by_quarter") or {})
        return

    # ---------------------- scrape ----------------------
    _ensure_chromium_ready()

    exact_lookup, norm_lookup = _build_fund_to_batch_lookup()

    today = _today_et_date().date()
    start_date = today - timedelta(days=max(1, lookback_days) - 1)

    def _label_with_batch(fund_name: str) -> str:
        fn = (fund_name or "").strip()
        if not fn:
            return fn
        b = exact_lookup.get(fn)
        if not b:
            b = norm_lookup.get(_normalize_fund_name(fn))
        if b:
            return f"{fn}_{b.replace(' ', '')}"  # FundName_Batch3
        return fn

    # Collect hits from the database table within the window, grouped by row quarter.
    hits_by_quarter: Dict[str, List[Hit]] = {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(accept_downloads=True)
        page = ctx.new_page()
        page.set_default_timeout(30000)
        page.goto(BSD_URL)

        # IMPORTANT: for Batch 8 we do NOT iterate quarters; we want the table sorted by most recent.
        # Use "Last Two Quarters" view (latest_two) to ensure recency and reduce pagination.
        try:
            sel = page.locator(FILTERS["quarter"]).first
            # Set to latest_two if present; else leave current selection.
            sel.select_option("latest_two")
        except Exception:
            pass

        try:
            # clear fund filter and search
            page.locator(FILTERS["fund"]).first.fill("")
            page.locator(FILTERS["search_btn"]).first.click()
            page.wait_for_timeout(900)
        except Exception:
            pass

        st.write(f"Scanning latest table for items in window {start_date.isoformat()} → {today.isoformat()} …")

        rows = page.locator(TABLE_ROW)

        # If the table is sorted newest-first (as BSD indicates), we can early-stop once we hit older dates.
        for i in range(rows.count()):
            row = rows.nth(i)
            try:
                letter_date_str = row.locator("td").nth(COLMAP["letter_date"]-1).inner_text().strip()
                dt = _parse_letter_date_to_date(letter_date_str)
                if dt is None:
                    continue

                d = dt.date()
                if d > today:
                    continue
                if d < start_date:
                    break  # older than window; stop scanning

                # quarter comes from the row itself
                q_row = row.locator("td").nth(COLMAP["quarter"]-1).inner_text().strip()
                q_row = q_row or "UNKNOWN"

                fund_cell = row.locator("td").nth(COLMAP["fund_name"]-1)
                link = fund_cell.locator("a").first
                fund_name = (link.inner_text() or "").strip()
                fund_href = link.get_attribute("href") or ""
                if fund_href and fund_href.startswith("/"):
                    fund_href = "https://www.buysidedigest.com" + fund_href

                h = Hit(
                    quarter=q_row,
                    letter_date=letter_date_str,
                    fund_name=fund_name,
                    fund_href=fund_href,
                )
                hits_by_quarter.setdefault(q_row, []).append(h)
            except Exception:
                continue

        try:
            ctx.close()
        finally:
            browser.close()

    if not hits_by_quarter:
        cache_all[cache_key] = {
            "quarters": ["LATEST"],
            "lookback_days": lookback_days,
            "by_quarter": {},
        }
        st.session_state["batch_cache"] = cache_all
        st.info("No letters found within the selected lookback window.")
        return

    # ---------------------- process hits (download + excerpt) ----------------------
    by_quarter: Dict[str, Dict[str, Any]] = {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(accept_downloads=True)
        page = ctx.new_page()
        page.set_default_timeout(30000)

        for q, hits in hits_by_quarter.items():
            table_rows: List[Dict[str, Any]] = []
            manifest_items: List[Dict[str, Any]] = []
            excerpt_pdfs: List[Path] = []

            for h in hits:
                brand = _label_with_batch(h.fund_name or "")
                st.write(f"[{q}] Latest — {brand} ({h.letter_date})")

                if _already_completed(brand, q):
                    st.info(f"[{q}] Skipping {brand} (already completed in this container).")
                    continue

                table_rows.append(
                    {
                        "fund_family": brand,
                        "search_token": "",
                        "quarter": q,
                        "letter_date": h.letter_date,
                        "fund_name": h.fund_name,
                        "fund_href": h.fund_href,
                    }
                )

                try:
                    page.goto(h.fund_href)
                    pdfs = _download_quarter_pdf_from_fund(page, q, _downloads_dir())
                    if not pdfs:
                        _mark_completed(brand, q)
                        continue

                    seen = set()
                    for pdf in pdfs:
                        if pdf.name in seen:
                            continue
                        seen.add(pdf.name)

                        out_dir = EX_DIR / q / _safe(brand) / _safe(pdf.stem)
                        built = run_excerpt_and_build(
                            pdf,
                            out_dir,
                            source_pdf_name=pdf.name,
                            letter_date=h.letter_date or None,
                            source_url=h.fund_href,
                        )

                        manifest_items.append(
                            {
                                "fund_family": brand,
                                "search_token": "",
                                "letter_date": h.letter_date or "",
                                "downloaded_pdf": str(pdf),
                                "source_pdf_name": pdf.name,
                                "excerpt_dir": str(out_dir),
                                "excerpts_json": str(out_dir / "excerpts_clean.json"),
                                "excerpt_pdf": str(built) if built else "",
                                "fund_name": h.fund_name,
                                "fund_href": h.fund_href,
                            }
                        )

                        if built:
                            excerpt_pdfs.append(Path(built))

                    _mark_completed(brand, q)
                except Exception as e:
                    st.warning(f"[{q}] Failed to process {brand}: {e}")
                    continue

            compiled = compile_merged(BATCH8_NAME, q, excerpt_pdfs, incremental=False)
            by_quarter[q] = {
                "compiled": str(compiled) if compiled else "",
                "manifest_items": manifest_items,
                "table_rows": table_rows,
            }

        try:
            ctx.close()
        finally:
            browser.close()

    # Save to session cache
    cache_all[cache_key] = {
        "quarters": ["LATEST"],
        "lookback_days": lookback_days,
        "by_quarter": by_quarter,
    }
    st.session_state["batch_cache"] = cache_all

    st.success("Batch 8 latest run complete.")
    _render_latest_results(by_quarter)




def _row_key(row: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """
    Build a stable key for a table row so we can compare snapshots.
    """
    return (
        row.get("fund_family", ""),
        row.get("fund_name", ""),
        row.get("quarter", ""),
        row.get("letter_date", ""),
        row.get("fund_href", ""),
    )

def run_incremental_update(batch_name: str, quarter: str, use_first_word: bool):
    """
    Fast per-batch incremental mode:
      - Reads the latest manifest for (batch, quarter) to get previous table_rows.
      - Scans the BSD table now (no downloads yet).
      - If table_rows unchanged => nothing to do.
      - If some rows are new/changed => downloads and processes only those.
    """
    st.markdown(f"### Incremental update – {batch_name} / {quarter}")

    manifests = _load_manifests(batch_name, quarter)
    if not manifests:
        st.info(
            "No manifest history found yet for this batch and quarter. "
            "Run a full batch once under 'Run scope' before using incremental mode."
        )
        return

    latest = manifests[0]
    prev_rows = latest.get("table_rows")
    if not prev_rows:
        st.info(
            "Latest manifest for this batch and quarter does not contain table-level "
            "snapshot data (likely created before the incremental feature). "
            "Run a full batch once so the next manifest includes table_rows, "
            "then use incremental mode."
        )
        return

    prev_key_set = { _row_key(r) for r in prev_rows }

    brands = RUNNABLE_BATCHES.get(batch_name, [])
    if not brands:
        st.info("No runnable fund families in this batch.")
        return

    tokens = [(b, _first_word(b) if use_first_word else b) for b in brands]

    current_rows: List[Dict[str, Any]] = []
    row_by_key: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(accept_downloads=True)
        page = ctx.new_page()
        page.set_default_timeout(30000)
        page.goto(BSD_URL)

        st.write(f"Scanning BSD table for {batch_name} / {quarter} (no downloads yet)…")

        if not _set_quarter(page, quarter):
            st.warning(
                f"Quarter **{quarter}** is not available on the data source at the moment. "
                "It may not have any letters yet."
            )
            browser.close()
            return

        # scan table rows for all brands
        for i, (brand, token) in enumerate(tokens, 1):
            st.write(f"[{quarter}] {i}/{len(tokens)} — {brand} (search: {token})")
            try:
                _search_by_fund(page, token)
                hits = _parse_rows(page, quarter)
                for h in hits:
                    row = {
                        "fund_family": brand,
                        "search_token": token,
                        "quarter": h.quarter,
                        "letter_date": h.letter_date,
                        "fund_name": h.fund_name,
                        "fund_href": h.fund_href,
                    }
                    key = _row_key(row)
                    current_rows.append(row)
                    row_by_key[key] = row
            except Exception as e:
                st.error(f"Error scanning fund family {brand}: {e}")
                continue

        # Compare snapshots
        current_key_set = set(row_by_key.keys())
        new_keys = current_key_set - prev_key_set

        if not new_keys:
            st.success(
                "No new or changed letters detected for this batch and quarter "
                "compared to the latest manifest. Nothing to download today."
            )
            # Still write a snapshot-only manifest so next comparison is up to date
            _write_manifest(batch_name, quarter, compiled=None, items=[], table_rows=current_rows)
            browser.close()
            return

        st.write(f"Found {len(new_keys)} new or changed table rows. Downloading only those…")

        outs: List[Path] = []
        manifest_items: List[Dict[str, Any]] = []

        processed_hrefs: set = set()

        for key in new_keys:
            row = row_by_key[key]
            href = row.get("fund_href") or ""
            brand = row.get("fund_family") or ""
            token = row.get("search_token") or ""
            letter_date = row.get("letter_date") or ""
            fund_name = row.get("fund_name") or ""

            if not href or not brand:
                continue
            if href in processed_hrefs:
                continue
            processed_hrefs.add(href)

            try:
                st.write(f"Downloading new/updated letter for {brand} – {fund_name} ({letter_date})")
                page.goto(href)
                page.wait_for_load_state("domcontentloaded")

                dest = DL_DIR / quarter / _safe(brand)
                pdfs = _download_quarter_pdf_from_fund(page, quarter, dest)
                for pdf in pdfs:
                    out_dir = EX_DIR / quarter / _safe(brand) / _safe(pdf.stem)
                    built = run_excerpt_and_build(
                        pdf,
                        out_dir,
                        source_pdf_name=pdf.name,
                        letter_date=letter_date or None,
                        source_url=href,
                    )

                    manifest_items.append(
                        {
                            "fund_family": brand,
                            "search_token": token,
                            "letter_date": letter_date,
                            "downloaded_pdf": str(pdf),
                            "source_pdf_name": pdf.name,
                            "excerpt_dir": str(out_dir),
                            "excerpts_json": str(out_dir / "excerpts_clean.json"),
                            "excerpt_pdf": str(built) if built else "",
                            "fund_name": fund_name,
                            "fund_href": href,
                        }
                    )

                    if built:
                        outs.append(built)
            except Exception as e:
                st.error(f"Error downloading for {brand}: {e}")
                continue

        compiled = None
        if outs:
            compiled = compile_merged(batch_name, quarter, outs, incremental=True)
            if compiled:
                st.success(f"Incremental compiled PDF created: {compiled}")

                # Direct download: BatchN_Date_Incremental_Excerpt.pdf
                try:
                    with open(compiled, "rb") as f:
                        st.download_button(
                            label="Download incremental excerpt PDF",
                            data=f.read(),
                            file_name=compiled.name,  # e.g. Batch1_2025-12-04_Incremental_Excerpt.pdf
                            mime="application/pdf",
                            key=f"download_inc_{batch_name.replace(' ', '')}_{quarter}".replace('/', '_'),
                        )
                except Exception:
                    st.warning("Incremental PDF created but could not be opened for download. Check server logs.")
            else:
                st.info(
                    "New letters were found, but no excerpt PDFs were produced. "
                    "They may not contain any tracked tickers."
                )

        # Write manifest capturing current snapshot + any new items we processed
        _write_manifest(batch_name, quarter, compiled, manifest_items, table_rows=current_rows)

        browser.close()

# ---------- UI ----------

def main():
    st.set_page_config(page_title="Cutler Capital Scraper", layout="wide")

    # Global styling: Cutler purple theme and modernized controls
    st.markdown(
        """
        <style>
        /* Center all images (logo) */
        .stImage img {
            display: block;
            margin-left: calc(100% - 20px);  /* pushes it ~20px to the right */
            transform: translateX(-50%);
        }
        /* Overall background and font tweaks */
        .stApp {
            background: radial-gradient(circle at top left, #f5f0fb 0, #ffffff 40%, #f7f3fb 100%);
        }
        .block-container {
            padding-top: 4rem;
            max-width: 1100px;
        }
        .app-title {
            text-align: center;
            color: #4b2142;
            font-size: 1.9rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .app-subtitle {
            text-align: center;
            color: #6b4f7a;
            font-size: 0.95rem;
            margin-top: 0.1rem;
            margin-bottom: 1.4rem;
        }

        /* Sidebar */
        [data-testid="stSidebar"] > div {
            background: #fbf8ff;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
            color: #4b2142;
        }
        
        header[data-testid="stHeader"] {
            background: radial-gradient(circle at top left, #f5f0fb 0, #ffffff 40%, #f7f3fb 100%) !important;
            box-shadow: none !important;
            border-bottom: none !important;
        }
        [data-testid="stToolbar"] {
            background: transparent !important;
        }
        header[data-testid="stHeader"] * {
            color: #4b2142 !important;
        }

        /* Buttons: long, pill-shaped, purple */
        .stButton>button {
            width: 100%;
            border-radius: 999px;
            background: #4b2142;
            color: #ffffff;
            border: 1px solid #4b2142;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .stButton>button:hover {
            background: #612a58;
            border-color: #612a58;
        }

        /* Radio group as pill toggle */
        div[role="radiogroup"] {
            display: flex;
            flex-wrap: nowrap;
            gap: 0.4rem;
        }
        div[role="radiogroup"] > label {
            flex: 1 1 0;
            justify-content: center;
            border-radius: 999px !important;
            padding: 0.35rem 0.95rem !important;
            border: 1px solid #d7c4f3 !important;
            background: #f7f3fb !important;
            color: #4b2142 !important;
            font-weight: 500 !important;
            white-space: nowrap;
        }
        div[role="radiogroup"] > label:hover {
            border-color: #4b2142 !important;
        }
        div[role="radiogroup"] > label[data-baseweb="radio"] > div:first-child > div[aria-checked="true"] + div {
            background: #4b2142 !important;
        }

        /* Card-style containers */
        .cc-card {
            background: #ffffffdd;
            border-radius: 20px;
            padding: 1.3rem 1.4rem;
            border: 1px solid rgba(75,33,66,0.08);
            box-shadow: 0 10px 30px rgba(75,33,66,0.04);
            margin-bottom: 1.1rem;
        }

        /* Fund chips */
        .fund-chip{
            display:inline-block;
            margin:6px 6px 0 0;
            padding:6px 12px;
            border-radius:14px;
            background:#f5effc;
            color:#4b2142;
            font-size:12px;
            font-weight:600;
            border:1px solid rgba(75,33,66,0.35);
            white-space:nowrap;
        }

        /* Gauge (needle) */
        .gauge-wrapper {
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .gauge {
            width: 220px;
            height: 120px;
            margin: 0.2rem auto 0.1rem;
            position: relative;
        }
        .gauge-body {
            width: 100%;
            height: 100%;
            border-radius: 220px 220px 0 0;
            background: #f5effc;
            border: 1px solid rgba(75,33,66,0.25);
            position: relative;
            overflow: hidden;
        }
        .gauge-needle {
            position: absolute;
            width: 2px;
            height: 85%;
            top: 15%;
            left: 50%;
            background: #4b2142;
            transform-origin: bottom center;
            transition: transform 0.25s ease-out;
        }
        .gauge-cover {
            width: 68%;
            height: 68%;
            background: #ffffff;
            border-radius: 50%;
            position: absolute;
            bottom: -10%;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            color: #4b2142;
        }

        /* --- Website-style FULL-WIDTH tabs --- */

        /* Tabs container spacing */
        div[data-testid="stTabs"] {
            width: 100%;
            margin-top: 0.75rem;
            margin-bottom: 1.25rem;
        }

        /* The tabs row */
        div[data-testid="stTabs"] [role="tablist"] {
            width: 100%;
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            padding: 0.35rem 0.45rem;
            border-bottom: 1px solid rgba(75,33,66,0.12);
        }

        /* Each tab button (equal width) */
        div[data-testid="stTabs"] [role="tab"] {
            flex: 1 1 0;
            width: 100%;
            text-align: center;
            justify-content: center;

            background: transparent;
            border: 1px solid rgba(75,33,66,0.18);
            border-bottom: 0;
            border-radius: 14px 14px 0 0;
            padding: 0.65rem 0.95rem;

            color: #4b2142;
            font-weight: 650;
            font-size: 0.95rem;
            transition: all 120ms ease-in-out;
        }

        /* Hover state */
        div[data-testid="stTabs"] [role="tab"]:hover {
            background: rgba(75,33,66,0.06);
            border-color: rgba(75,33,66,0.28);
        }

        /* Selected tab */
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            background: #ffffff;
            border-color: rgba(75,33,66,0.35);
            box-shadow: 0 10px 20px rgba(75,33,66,0.05);
            position: relative;
        }

        /* Selected underline accent */
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {
            content: "";
            position: absolute;
            left: 12%;
            right: 12%;
            bottom: -2px;
            height: 3px;
            border-radius: 999px;
            background: #4b2142;
        }

        /* Remove Streamlit's default focus outline and replace with subtle ring */
        div[data-testid="stTabs"] [role="tab"]:focus-visible {
            outline: none;
            box-shadow: 0 0 0 3px rgba(75,33,66,0.18);
        }

        /* Content panel spacing */
        div[data-testid="stTabs"] [data-testid="stTabContent"] {
            padding-top: 0.75rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header: centered logo and text
        # Header: logo + title in a centered column
    logo_path = HERE / "cutler.png"
    col_left, col_center, col_right = st.columns([1, 2, 1])

    with col_center:
        if logo_path.exists():
            st.image(str(logo_path), width=260)

        st.markdown("<div class='app-title'>Cutler Capital Letter Scraper</div>", unsafe_allow_html=True)

    # Sidebar: run settings
    st.sidebar.header("Run settings")

    quarter_options = get_available_quarters()
    default_q = choose_default_quarter(quarter_options)

    # Auto-update the default quarter based on current date (ET),
    # without overriding a user's manual selection.
    if "quarters" not in st.session_state:
        st.session_state["quarters"] = ([default_q] if default_q else quarter_options[:1])
        st.session_state["auto_default_quarter"] = (st.session_state["quarters"][0] if st.session_state["quarters"] else None)
    else:
        prev_auto = st.session_state.get("auto_default_quarter")
        if prev_auto and st.session_state.get("quarters") == [prev_auto] and default_q and default_q != prev_auto:
            st.session_state["quarters"] = [default_q]
            st.session_state["auto_default_quarter"] = default_q

    quarters = st.sidebar.multiselect(
        "Quarters",
        quarter_options,
        default=st.session_state.get("quarters") or ([default_q] if default_q else quarter_options[:1]),
        key="quarters",
    )

    use_first_word = st.sidebar.checkbox(
        "Use first word for search (recommended)",
        value=True,
    )

    # Optional: AI relevance scoring inside excerpt PDFs (adds 1–5 rating + highlight per paragraph)
    ai_score_enabled = st.sidebar.checkbox(
        "AI relevance scoring (1–5 highlights)",
        value=False,
        help="Uses OpenAI to rate how directly a paragraph discusses the company. "
             "Adds a rating tag and background highlight for faster skimming.",
        key="ai_score_enabled",
    )
    ai_score_model = st.sidebar.text_input(
        "AI model (for relevance scoring)",
        value="gpt-4o-mini",
        help="Used only if AI relevance scoring is enabled.",
        key="ai_score_model",
    )

    batch_names = list(RUNNABLE_BATCHES.keys())

    # --- Tabs (website-style nav) ---
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    
    # ---------------------- RUN ALL (orchestrator) ----------------------
    st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
    st.markdown("### Run All (Fund Families Latest + Seeking Alpha All + Substack + Podcasts All)")

    ra_state = _load_run_all_state()
    ra_cfg = ra_state.get("config") or {}
    ra_mf_days = st.selectbox(
        "Fund Families Latest lookback (days)",
        options=[7, 14, 30],
        index=[7, 14, 30].index(int(ra_cfg.get("mf_lookback_days", 7))) if int(ra_cfg.get("mf_lookback_days", 7)) in [7, 14, 30] else 0,
        key="run_all_mf_days",
    )
    ra_sa_max = st.number_input(
        "Seeking Alpha max articles per ticker",
        min_value=1,
        max_value=20,
        value=int(ra_cfg.get("sa_max_articles", 5)),
        step=1,
        key="run_all_sa_max_articles",
    )
    ra_sa_model = st.selectbox(
        "Seeking Alpha model (digest/export)",
        options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
        index=0,
        key="run_all_sa_model",
    )

    ra_substack_days = st.selectbox(
        "Substack lookback (days)",
        options=[2, 7],
        index=[2, 7].index(int(ra_cfg.get("substack_lookback_days", 2))) if int(ra_cfg.get("substack_lookback_days", 2)) in [2, 7] else 0,
        key="run_all_substack_days",
    )
    ra_substack_max = st.number_input(
        "Substack max posts per ticker",
        min_value=1,
        max_value=10,
        value=int(ra_cfg.get("substack_max_posts", 3)),
        step=1,
        key="run_all_substack_max_posts",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        start_all = st.button("Run All", use_container_width=True, key="run_all_start")
    with c2:
        resume_all = st.button(
            "Resume",
            use_container_width=True,
            key="run_all_resume",
            disabled=not bool(ra_state and ra_state.get("status") == "running"),
        )
    with c3:
        clear_all = st.button("Clear Run All state", use_container_width=True, key="run_all_clear")

    if clear_all:
        _clear_run_all_state()
        st.rerun()

    if start_all:
        ra_state = {
            "status": "running",
            "current_step": "fund_families",
            "completed": [],
            "outputs": {},
            "config": {
                "mf_lookback_days": int(ra_mf_days),
                "sa_max_articles": int(ra_sa_max),
                "sa_model": str(ra_sa_model),
                            "substack_lookback_days": int(ra_substack_days),
                "substack_max_posts": int(ra_substack_max),
},
            "started_at": _now_et().isoformat(),
        }
        _save_run_all_state(ra_state)
        st.rerun()

    if resume_all:
        st.rerun()

    # Existing outputs (persisted)
    outs = ra_state.get("outputs") or {}
    mf_paths = (outs.get("fund_families") or {}).get("paths") or []
    if mf_paths:
        st.markdown("**Fund Families outputs:**")
        for pinfo in mf_paths:
            try:
                fp = Path(pinfo.get("path") or "")
                if fp.exists():
                    st.download_button(
                        f"Download {fp.name}",
                        data=fp.read_bytes(),
                        file_name=fp.name,
                        mime="application/pdf",
                        key=f"ra_dl_mf_{fp.name}",
                        use_container_width=True,
                    )
            except Exception:
                pass

    sa_path = (outs.get("seeking_alpha") or {}).get("path") or ""
    if sa_path:
        try:
            fp = Path(sa_path)
            if fp.exists():
                st.download_button(
                    f"Download {fp.name}",
                    data=fp.read_bytes(),
                    file_name=fp.name,
                    mime="application/pdf",
                    key=f"ra_dl_sa_{fp.name}",
                    use_container_width=True,
                )
        except Exception:
            pass


    sub_path = (outs.get("substack") or {}).get("path") or ""
    if sub_path:
        try:
            fp = Path(sub_path)
            if fp.exists():
                st.download_button(
                    f"Download {fp.name}",
                    data=fp.read_bytes(),
                    file_name=fp.name,
                    mime="application/pdf",
                    key=f"ra_dl_sub_{fp.name}",
                    use_container_width=True,
                )
        except Exception:
            pass

    pod_path = (outs.get("podcasts") or {}).get("path") or ""
    if pod_path:
        try:
            fp = Path(pod_path)
            if fp.exists():
                st.download_button(
                    f"Download {fp.name}",
                    data=fp.read_bytes(),
                    file_name=fp.name,
                    mime="application/pdf",
                    key=f"ra_dl_pod_{fp.name}",
                    use_container_width=True,
                )
        except Exception:
            pass

    # Execute next step if running
    if ra_state.get("status") == "running":
        step = ra_state.get("current_step")
        cfg = ra_state.get("config") or {}

        # Auto-skip completed steps (prevents download_button reruns from restarting work)
        # If an output path is already persisted and the file exists on disk, mark the step complete and advance.
        _advance_guard = 0
        while _advance_guard < 6 and ra_state.get("status") == "running":
            _advance_guard += 1
            step = ra_state.get("current_step")
            outs = ra_state.get("outputs") or {}
            completed = ra_state.get("completed") or []
            if not isinstance(completed, list):
                completed = []

            def _mark_done(_step: str, _next: str):
                if _step not in completed:
                    completed.append(_step)
                ra_state["completed"] = completed
                ra_state["current_step"] = _next
                _save_run_all_state(ra_state)

            if step == "fund_families":
                mf_paths = (outs.get("fund_families") or {}).get("paths") or []
                ok = False
                try:
                    for pinfo in mf_paths:
                        fp = Path((pinfo or {}).get("path") or "")
                        if fp and fp.exists():
                            ok = True
                            break
                except Exception:
                    ok = False
                if ok:
                    _mark_done("fund_families", "seeking_alpha")
                    continue

            if step == "seeking_alpha":
                sa_path = (outs.get("seeking_alpha") or {}).get("path") or ""
                if sa_path and Path(sa_path).exists():
                    _mark_done("seeking_alpha", "substack")
                    continue

            if step == "substack":
                sub_path = (outs.get("substack") or {}).get("path") or ""
                if sub_path and Path(sub_path).exists():
                    _mark_done("substack", "podcasts")
                    continue

            if step == "podcasts":
                pod_path = (outs.get("podcasts") or {}).get("path") or ""
                if pod_path and Path(pod_path).exists():
                    # If podcasts already exist, finalize Run All state.
                    if "podcasts" not in completed:
                        completed.append("podcasts")
                    ra_state["completed"] = completed
                    ra_state["current_step"] = "done"
                    ra_state["status"] = "complete"
                    ra_state["completed_at"] = _now_et().isoformat()
                    _save_run_all_state(ra_state)
                    break

            break

        try:
            if step == "fund_families":
                days = int(cfg.get("mf_lookback_days", 7))
                # NOTE: Do not wrap Fund Families in st.status(expanded=True) because Fund Families uses expanders internally.
                st.info(f"Run All: Fund Families — Batch 8 Latest (last {days} days)")
                quarter_options = get_available_quarters()
                run_batch8_latest(quarter_options, days, use_first_word, ensure_compiled_index=True)
                cache_all = st.session_state.get("batch_cache", {}) or {}
                cache_key = f"{BATCH8_NAME}|{days}d"
                by_q = (cache_all.get(cache_key) or {}).get("by_quarter") or {}
                paths = []
                for q, qd in by_q.items():
                    c = (qd or {}).get("compiled") or ""
                    if c:
                        paths.append({"quarter": q, "path": c})
                ra_state.setdefault("outputs", {}).setdefault("fund_families", {})["paths"] = paths
                ra_state.setdefault("completed", []).append("fund_families")
                ra_state["current_step"] = "seeking_alpha"
                _save_run_all_state(ra_state)
                st.rerun()

            if step == "seeking_alpha":
                max_articles = int(cfg.get("sa_max_articles", 5))
                model_name = str(cfg.get("sa_model", "gpt-4o-mini"))
                with st.status("Run All: Seeking Alpha — building compiled PDF for ALL tickers", expanded=True):
                    # Build full universe from tickers.py if available; else fall back to current universe in SA section.
                    universe = []
                    try:
                        from tickers import tickers as _T  # type: ignore
                        if isinstance(_T, dict):
                            universe = [t for t in _T.keys() if _is_probable_ticker(t)]
                    except Exception:
                        universe = []
                    if not universe:
                        universe = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "ABBV"]
                    out_pdf = _build_sa_compiled_pdf_for_universe(universe=universe, max_articles=max_articles, model=model_name)
                    ra_state.setdefault("outputs", {}).setdefault("seeking_alpha", {})["path"] = str(out_pdf)
                ra_state.setdefault("completed", []).append("seeking_alpha")
                ra_state["current_step"] = "substack"
                _save_run_all_state(ra_state)
                st.rerun()


            if step == "substack":
                days_back = int(cfg.get("substack_lookback_days", 2))
                max_posts = int(cfg.get("substack_max_posts", 3))
                with st.status(f"Run All: Substack — building compiled PDF (last {days_back} days)", expanded=True):
                    # Use the same ticker universe logic as Seeking Alpha (Cutler tickers.py)
                    try:
                        from tickers import tickers as _T  # type: ignore
                        universe = []
                        if isinstance(_T, dict):
                            universe = [t for t in _T.keys() if _is_probable_ticker(t)]
                    except Exception:
                        universe = []
                    if not universe:
                        universe = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "ABBV"]

                    out_pdf = _build_substack_compiled_pdf_for_universe(
                        universe=universe,
                        lookback_days=days_back,
                        max_posts=max_posts,
                    )
                    ra_state.setdefault("outputs", {}).setdefault("substack", {})["path"] = str(out_pdf)

                ra_state.setdefault("completed", []).append("substack")
                ra_state["current_step"] = "podcasts"
                _save_run_all_state(ra_state)
                st.rerun()

            if step == "podcasts":
                days_back = int(cfg.get("podcast_lookback_days", cfg.get("mf_lookback_days", 2)))
                model_name = str(cfg.get("sa_model", "gpt-4o-mini"))

                # Run podcasts in small groups to avoid long blocking runs in Streamlit.
                run_dir = BASE / "Podcasts" / "_run_all"
                run_dir.mkdir(parents=True, exist_ok=True)

                pr = ra_state.get("podcasts_runall") or {}
                if not isinstance(pr, dict):
                    pr = {}
                group_index = int(pr.get("group_index", 0))

                # Prefer the same grouping logic used in the Podcast tab (9 buckets).
                groups = _podcast_run_all_group_ids(n_groups=9)
                if not groups:
                    # Fallback: single group from podcasts_config
                    try:
                        from podcasts_config import PODCASTS as _PODCASTS  # type: ignore
                        fallback_ids = []
                        for p in (_PODCASTS or []):
                            pid = getattr(p, "podcast_id", None) or getattr(p, "id", None) or getattr(p, "pod_id", None)
                            if pid:
                                fallback_ids.append(str(pid))
                        groups = [fallback_ids] if fallback_ids else []
                    except Exception:
                        groups = []

                total_groups = len(groups)

                # Persistent checkpointing for Run All podcasts (survives reloads/timeouts)
                ckpt_path = (BASE / "Podcasts" / "runall_podcasts_state.json")
                run_sig = {
                    "date": str(_now_et().date()),
                    "days_back": days_back,
                    "model_name": model_name,
                    "total_groups": total_groups,
                }
                ckpt = _load_json_safe(ckpt_path, {})
                if not isinstance(ckpt, dict):
                    ckpt = {}
                # If the signature changed (new day/params), reset checkpoint.
                if ckpt.get("sig") != run_sig:
                    ckpt = {"sig": run_sig, "completed_groups": []}
                    _save_json_safe(ckpt_path, ckpt)

                completed_groups = ckpt.get("completed_groups") or []
                if not isinstance(completed_groups, list):
                    completed_groups = []
                completed_set = {int(x) for x in completed_groups if str(x).isdigit()}
                # Resume from first incomplete group, regardless of session_state value.
                for gi in range(total_groups):
                    if gi not in completed_set:
                        group_index = gi
                        break
                else:
                    group_index = total_groups

                # Progress bar for Run All podcasts (group-level)
                _completed_groups_n = len(completed_set)
                _prog_den = max(1, int(total_groups))
                _prog_num = min(_prog_den, int(_completed_groups_n))
                st.progress(float(_prog_num) / float(_prog_den))
                st.caption(f"Run All: Podcasts progress — {_prog_num}/{_prog_den} groups complete")



                if not groups or total_groups == 0:
                    st.warning("No podcast IDs found; skipping podcasts.")
                    out_pdf = None
                else:
                    # Process one group per rerun for stability
                    if group_index < total_groups:
                        with st.status(
                            f"Run All: Podcasts — processing group {group_index + 1}/{total_groups} (last {days_back} days)",
                            expanded=True,
                        ):
                            podcast_ids = groups[group_index] or []
                            if not podcast_ids:
                                st.info("Empty podcast group; skipping.")
                            else:
                                group_dir = run_dir / f"g{group_index + 1:02d}"
                                podcasts_root = group_dir / "transcripts"
                                excerpts_path = group_dir / "podcast_excerpts.json"
                                insights_path = group_dir / "podcast_insights.json"
                                group_dir.mkdir(parents=True, exist_ok=True)

                                _ = run_podcast_pipeline_from_ui(
                                    days_back=days_back,
                                    podcast_ids=podcast_ids,
                                    podcasts_root=podcasts_root,
                                    excerpts_path=excerpts_path,
                                    insights_path=insights_path,
                                    model_name=model_name,
                                )

                        # Mark this group completed and persist progress
                        try:
                            ckpt = _load_json_safe(ckpt_path, {})
                            if not isinstance(ckpt, dict):
                                ckpt = {"sig": run_sig, "completed_groups": []}
                            cg = ckpt.get("completed_groups") or []
                            if not isinstance(cg, list):
                                cg = []
                            if group_index not in cg:
                                cg.append(group_index)
                            ckpt["sig"] = run_sig
                            ckpt["completed_groups"] = cg
                            _save_json_safe(ckpt_path, ckpt)
                        except Exception:
                            pass

                        pr["group_index"] = group_index + 1
                        pr["total_groups"] = total_groups
                        ra_state["podcasts_runall"] = pr
                        _save_run_all_state(ra_state)
                        st.rerun()

                    # All groups done -> merge and build final PDF once
                    with st.status(
                        f"Run All: Podcasts — merging groups and building compiled PDF (last {days_back} days)",
                        expanded=True,
                    ):
                        merged_excerpts: dict = {}
                        merged_insights = []

                        for gi in range(total_groups):
                            group_dir = run_dir / f"g{gi + 1:02d}"
                            ep = group_dir / "podcast_excerpts.json"
                            ip = group_dir / "podcast_insights.json"
                            merged_excerpts = _merge_podcast_excerpts_dict(
                                merged_excerpts, _load_json_safe(ep, {})
                            )
                            merged_insights = _merge_podcast_insights_list(
                                merged_insights, _load_json_safe(ip, [])
                            )

                        excerpts_path = run_dir / "podcast_excerpts.json"
                        insights_path = run_dir / "podcast_insights.json"
                        excerpts_path.write_text(json.dumps(merged_excerpts, ensure_ascii=False, indent=2), encoding="utf-8")
                        insights_path.write_text(json.dumps(merged_insights, ensure_ascii=False, indent=2), encoding="utf-8")

                        now_et = _now_et()
                        out_name = f"{now_et:%m.%d.%y} Podcast ALL.pdf"
                        out_path = (BASE / "Podcasts" / out_name)
                        out_path.parent.mkdir(parents=True, exist_ok=True)

                        out_pdf = _build_podcast_all_pdf(
                            excerpts_path=excerpts_path,
                            insights_path=insights_path,
                            output_path=out_path,
                            days_back=days_back,
                        )
                        # Cleanup checkpoint on successful completion
                        try:
                            if ckpt_path.exists():
                                ckpt_path.unlink()
                        except Exception:
                            pass


                if out_pdf:
                    ra_state.setdefault("outputs", {}).setdefault("podcasts", {})["path"] = str(out_pdf)

                ra_state.setdefault("completed", []).append("podcasts")
                ra_state["current_step"] = "done"
                ra_state["status"] = "complete"
                ra_state["completed_at"] = _now_et().isoformat()
                _save_run_all_state(ra_state)
                st.rerun()
        except Exception as e:
            ra_state["status"] = "error"
            ra_state["error"] = str(e)
            _save_run_all_state(ra_state)
            st.error(f"Run All failed at step '{step}': {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    tab_mf, tab_sa, tab_substack, tab_podcast = st.tabs(
        ["Fund Families", "Seeking Alpha", "Substack", "Podcast"]
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    with tab_mf:
        if st.button("Clean current tab cache", key="clean_mf_cache", use_container_width=True):
            _clear_session_keys(
                exact=["batch_cache"],
                prefixes=["mf_","fund_","funds_","batch_"],
            )
            st.rerun()

        # Main controls in a card – full run
        with st.container():
            st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
            st.markdown("#### Run scope", unsafe_allow_html=True)
            st.write("Choose whether you want a full run across all batches or a targeted test.")

            run_mode = st.radio(
                "Run mode",
                ["Run all 7 batches", "Run a specific batch", "Run Batch 8 — Latest"],
                index=1,
            )

            if run_mode == "Run all 7 batches":
                st.info(
                    "Runs every fund family in all 7 batches for the selected quarter(s). "
                    "Use the specific batch mode below if you are just testing a few names."
                )
                if st.button("Run all 7 batches", use_container_width=True):
                    for bn in batch_names:
                        run_batch(bn, quarters, use_first_word, subset=None)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.stop()
            elif run_mode == "Run Batch 8 — Latest":
                lookback_days = st.selectbox(
                    "Lookback window (days)",
                    options=[7, 14, 30],
                    index=0,
                )
                st.info(
                    "Batch 8 scans the BSD database for anything published within the lookback window. "
                    "If a fund matches one of Batch 1–7 names, it is labeled with that batch for traceability."
                )
                if st.button("Run Batch 8 — Latest", use_container_width=True):
                    run_batch8_latest(quarter_options, int(lookback_days), use_first_word)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                selected_batch = st.selectbox("Choose a batch to run", batch_names)
                if selected_batch:
                    names_in_batch = RUNNABLE_BATCHES[selected_batch]
                    st.write(f"{selected_batch} contains **{len(names_in_batch)}** fund families.")

                    with st.expander("Preview fund families in this batch"):
                        chips = "".join(
                            f"<span class='fund-chip'>{name}</span>"
                            for name in names_in_batch
                        )
                        st.markdown(chips, unsafe_allow_html=True)

                    selected_funds = st.multiselect(
                        "Optionally target specific fund families "
                        "(leave empty to run the entire batch):",
                        options=names_in_batch,
                    )
                    subset = selected_funds or None

                    # Button only decides whether we *run* scraping.
                    run_clicked = st.button(f"Run {selected_batch}", use_container_width=True)

                    if run_clicked:
                        # First time or explicit re-run: run_batch may scrape,
                        # build excerpts, compile, and update session cache.
                        run_batch(selected_batch, quarters, use_first_word, subset=subset)
                    else:
                        # No click this rerun (e.g. user just hit a download button),
                        # but if we have cached results for this batch+quarters,
                        # re-render them without scraping.
                        cache_all = st.session_state.get("batch_cache", {})
                        cache_entry = cache_all.get(selected_batch)
                        if cache_entry and cache_entry.get("quarters") == quarters:
                            run_batch(selected_batch, quarters, use_first_word, subset=subset)


            st.markdown("</div>", unsafe_allow_html=True)

        # Incremental per-batch updater
        st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
        st.markdown("#### Incremental update (per batch)", unsafe_allow_html=True)
        st.write(
            "Use this when interns run the tool daily. It compares the current BSD table "
            "to the latest stored manifest for a batch and quarter, and only downloads "
            "letters that are new or changed."
        )

        inc_quarter = st.selectbox(
            "Quarter for incremental check",
            options=quarter_options,
            index=quarter_options.index(default_q) if default_q in quarter_options else 0,
            key="inc_quarter",
        )
        inc_batch = st.selectbox(
            "Batch for incremental update",
            options=batch_names,
            index=0,
            key="inc_batch",
        )

        if st.button("Check for updates and download new letters", key="inc_btn"):
            run_incremental_update(inc_batch, inc_quarter, use_first_word)

        st.markdown("</div>", unsafe_allow_html=True)

        # Document Checker
        st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
        st.markdown("#### Document Checker", unsafe_allow_html=True)
        st.write(
            "Compare two compiled runs for a given batch and quarter, and generate a PDF "
            "containing only **new** ticker-related paragraphs found in the newer run."
        )

        checker_quarter = st.selectbox(
            "Quarter to inspect",
            options=quarter_options,
            index=quarter_options.index(default_q) if default_q in quarter_options else 0,
            key="checker_quarter",
        )
        checker_batch = st.selectbox(
            "Batch",
            options=batch_names,
            index=0,
            key="checker_batch",
        )

        manifests = _load_manifests(checker_batch, checker_quarter)
        if not manifests:
            st.info(
                "No history found yet for this batch and quarter. "
                "Run the scraper at least twice to compare documents."
            )
        elif len(manifests) == 1:
            only = manifests[0]
            st.info(
                "Only one compiled run is stored so far for this batch and quarter "
                f"(created {only.get('created_at', '')}). Run the scraper again to "
                "create a second run for comparison."
            )
        else:
            labels = [
                f"{i+1}. {m.get('created_at', '')} – {Path(m.get('compiled_pdf', '')).name or '[no compiled PDF]'}"
                for i, m in enumerate(manifests)
            ]
            idx_new = 0
            idx_old = 1 if len(manifests) > 1 else 0

            new_idx = st.selectbox(
                "Newer run",
                options=list(range(len(manifests))),
                format_func=lambda i: labels[i],
                index=idx_new,
                key="checker_new",
            )
            old_idx = st.selectbox(
                "Older run to compare against",
                options=list(range(len(manifests))),
                format_func=lambda i: labels[i],
                index=idx_old,
                key="checker_old",
            )

            if new_idx == old_idx:
                st.warning("Please select two different runs to compare.")
            else:
                if st.button("Generate 'New Since' PDF", key="checker_btn"):
                    delta_pdf = build_delta_pdf(
                        old_manifest=manifests[old_idx],
                        new_manifest=manifests[new_idx],
                    )
                    if delta_pdf:
                        st.success(f"Delta PDF created: {delta_pdf}")
                        try:
                            with open(delta_pdf, "rb") as f:
                                st.download_button(
                                    "Download delta PDF",
                                    data=f,
                                    file_name=delta_pdf.name,
                                    mime="application/pdf",
                                    key="checker_download",
                                )
                        except Exception:
                            pass
                    else:
                        st.info(
                            "No new ticker-related commentary found between these two runs. "
                            "Everything appears to be the same."
                        )

        st.markdown("</div>", unsafe_allow_html=True)

            # ---------- AI Insights: Buy / Hold / Sell ----------
        st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
        st.markdown("#### AI Insights – Buy / Hold / Sell by company", unsafe_allow_html=True)
        st.write(
            "Use OpenAI to classify each ticker in a compiled run as **buy**, **hold**, "
            "**sell**, or **unclear**, with reasoning grounded in the excerpted letters."
        )

        ai_quarter = st.selectbox(
            "Quarter for AI analysis",
            options=quarter_options,
            index=quarter_options.index(default_q) if default_q in quarter_options else 0,
            key="ai_quarter",
        )
        ai_batch = st.selectbox(
            "Batch for AI analysis",
            options=batch_names,
            index=0,
            key="ai_batch",
        )

        ai_manifests = _load_manifests(ai_batch, ai_quarter)
        if not ai_manifests:
            st.info(
                "No manifests found yet for this batch and quarter. "
                "Run this batch at least once (full or incremental) before using AI insights."
            )
        else:
            labels = [
                f"{i+1}. {m.get('created_at', '')} – "
                f"{Path(m.get('compiled_pdf', '')).name or '[no compiled PDF]'}"
                for i, m in enumerate(ai_manifests)
            ]
            ai_manifest_idx = st.selectbox(
                "Which run should the AI analyse?",
                options=list(range(len(ai_manifests))),
                format_func=lambda i: labels[i],
                index=0,
                key="ai_manifest_idx",
            )

            ai_model = st.text_input(
                "OpenAI model name",
                value="gpt-4o-mini",
                help="Any chat-compatible model, e.g. gpt-4o or gpt-4o-mini.",
            )
            ai_use_web = st.checkbox(
                "Allow OpenAI to use web search",
                value=True,
                help="For now this mainly controls how much external context the model "
                     "is encouraged to bring into the `web_check` field.",
            )

            results: List[Dict[str, Any]] = []
            if st.button("Run AI analysis for this run", key="ai_run_btn"):
                manifest = ai_manifests[ai_manifest_idx]
                with st.spinner("Calling OpenAI for ticker-level stances…"):
                    try:
                        results = ai_insights.generate_ticker_stances(
                            manifest=manifest,
                            batch=ai_batch,
                            quarter=ai_quarter,
                            model=ai_model,
                            use_web=ai_use_web,
                        )
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
                        results = []

                st.session_state["ai_results"] = results

            # If we already have results in session, reuse them so we can interact with dropdown
            if "ai_results" in st.session_state and not results:
                results = st.session_state["ai_results"]

            if results:
                # Compact summary table
                summary_rows = []
                for r in results:
                    summary_rows.append(
                        {
                            "Ticker": r.get("ticker"),
                            "Company": ", ".join(r.get("company_names") or []),
                            "Stance": r.get("stance"),
                            "Confidence": round(float(r.get("confidence", 0.0)), 2),
                        }
                    )
                st.write("**Summary by ticker**")
                st.dataframe(summary_rows, use_container_width=True)

                # Detailed view: dropdown + gauge + reasoning
                ticker_options = [row["Ticker"] for row in summary_rows]
                if ticker_options:
                    focus_ticker = st.selectbox(
                        "Detailed view – choose a ticker",
                        options=ticker_options,
                        key="ai_focus_ticker",
                    )
                    detail = next((r for r in results if r.get("ticker") == focus_ticker), None)

                    if detail:
                        stance = (detail.get("stance") or "").lower()
                        conf = float(detail.get("confidence") or 0.0)

                        # Map stance + confidence to a 0–1 position for the gauge
                        if stance == "buy":
                            pos = 0.5 + 0.5 * conf
                        elif stance == "sell":
                            pos = 0.5 - 0.5 * conf
                        elif stance == "hold":
                            pos = 0.5
                        else:  # unclear
                            pos = 0.5
                        pos = max(0.0, min(1.0, pos))
                        angle = -90 + 180 * pos  # -90 (sell) .. 0 (hold) .. +90 (buy)

                        gauge_html = f"""
                        <div class="gauge-wrapper">
                          <div class="gauge">
                            <div class="gauge-body">
                              <div class="gauge-needle" style="transform: rotate({angle:.1f}deg);"></div>
                              <div class="gauge-cover">{stance.upper() if stance else "UNCLEAR"}</div>
                            </div>
                          </div>
                          <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#6b4f7a;margin-top:0.15rem;">
                            <span>Sell</span><span>Hold</span><span>Buy</span>
                          </div>
                        </div>
                        """
                        st.markdown(gauge_html, unsafe_allow_html=True)

                        company_label = ", ".join(detail.get("company_names") or [])
                        st.markdown(f"**Reasoning for {focus_ticker} ({company_label})**")
                        st.write(detail.get("primary_reasoning", ""))

                        st.markdown("**Evidence from commentaries**")
                        for ev in detail.get("commentary_evidence") or []:
                            st.markdown(f"- {ev}")

                        st.markdown("**Web check**")
                        st.write(detail.get("web_check_summary") or "No additional web context used.")

                        funds = detail.get("fund_families") or []
                        if funds:
                            chips = "".join(
                                f"<span class='fund-chip'>{f}</span>" for f in funds
                            )
                            st.markdown("**Fund sources used in this decision:**", unsafe_allow_html=True)
                            st.markdown(chips, unsafe_allow_html=True)
            else:
                st.info(
                    "Run the AI analysis above to see ticker stances, then select a ticker "
                    "for a detailed gauge view."
                )

        st.markdown("</div>", unsafe_allow_html=True)


    with tab_sa:
        if st.button("Clean current tab cache", key="clean_sa_cache", use_container_width=True):
            _clear_session_keys(
                exact=["sa_cache","sa_pdf_bytes","sa_pdf_name","sa_nav_idx","sa_batch_select_idx","sa_ticker_select","sa_manual_tickers"],
                prefixes=["sa_"],
            )
            st.rerun()

        # ---------- Seeking Alpha news + AI digest ----------
        draw_seeking_alpha_news_section()

        # ---------- Navigation: move to next ticker in the selected list ----------
        selected_tickers = st.session_state.get("sa_selected_tickers_prev", [])
        if selected_tickers and len(selected_tickers) > 1:
            col_prev, col_spacer, col_next = st.columns([1, 3, 1])
            with col_next:
                if st.button("Next ticker ▶", key="sa_next_ticker"):
                    current_idx = st.session_state.get("sa_current_index", 0)
                    next_idx = (current_idx + 1) % len(selected_tickers)
                    st.session_state["sa_current_index"] = next_idx
                    st.rerun()
            with col_prev:
                if st.button("◀ Previous", key="sa_prev_ticker"):
                    current_idx = st.session_state.get("sa_current_index", 0)
                    prev_idx = (current_idx - 1) % len(selected_tickers)
                    st.session_state["sa_current_index"] = prev_idx
                    st.rerun()
    

    with tab_substack:
        if st.button("Clean current tab cache", key="clean_substack_cache", use_container_width=True):
            _clear_session_keys(
                exact=["substack_cache"],
                prefixes=["substack_"],
            )
            st.rerun()

        # ---------- Substack (research feed) ----------
        draw_substack_section()


    with tab_podcast:
        if st.button("Clean current tab cache", key="clean_podcast_cache", use_container_width=True):
            _clear_session_keys(
                exact=["podcast_cache","podcast_last_cache_key","pod_export_pdf_path","podcast_pdf_bytes","podcast_pdf_name"],
                prefixes=["pod_","podcast_"],
            )
            st.rerun()

        # ---------- Podcast intelligence (ticker mentions across podcasts) ----------
        draw_podcast_intelligence_section()


    # Output path
    st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
    st.write("Output folder (local reference):")
    st.code(r"V:\CCM-AI\2025")
    st.caption(
        "This path is on your local machine. "
        "Copy and paste it into Windows Explorer to open."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Streamlit needs main() to run on import.
main()
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
from pypdf.generic import NameObject, ArrayObject

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

def ensure_playwright_chromium_installed() -> bool:
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
            st.success("Playwright Chromium installed successfully.")
            return True
        else:
            st.error("Playwright Chromium installation failed.")
            st.code(result.stderr)
            return False

    except Exception as e:
        st.error("Unexpected error during Chromium installation.")
        st.code(repr(e))
        return False

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

reddit_excerpts = _import("reddit_excerpts", HERE / "reddit_excerpts.py")

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

# External data source URL (kept internal; not shown in UI)
BSD_URL = "https://www.buysidedigest.com/hedge-fund-database/"
FILTERS = {
    "fund": "#md-fund-letter-table-fund-search",
    "quarter": "#md-fund-letter-table-select",
    "search_btn": "input.md-search-btn",
}
TABLE_ROW = "table tbody tr"
COLMAP = {"quarter": 1, "letter_date": 2, "fund_name": 3}

def _merge_page_preserve_annots(base_page, overlay_page):
    """
    Merge overlay onto base page but preserve existing link annotations.
    Without this, ReportLab hyperlink annotations often disappear in compiled PDFs.
    """
    annots = base_page.get("/Annots")
    base_page.merge_page(overlay_page)

    if annots:
        # Ensure annots remains an ArrayObject in the final page dict
        try:
            base_page[NameObject("/Annots")] = annots if isinstance(annots, ArrayObject) else ArrayObject(annots)
        except Exception:
            # fallback: try setting raw
            base_page[NameObject("/Annots")] = annots

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

def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "file"

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
            (HERE / "tickers.py").exists() and shutil.copy2(HERE / "tickers.py", tp)

        excerpt_check.excerpt_pdf_for_tickers(str(pdf_path), debug=False)

        src_json = pdf_path.parent / "excerpts_clean.json"
        if not src_json.exists():
            return None
        dst_json = out_dir / "excerpts_clean.json"
        if src_json != dst_json:
            shutil.copy2(src_json, dst_json)

        out_pdf = out_dir / f"Excerpted_{_safe(pdf_path.stem)}.pdf"
        make_pdf.build_pdf(
            excerpts_json_path=str(dst_json),
            output_pdf_path=str(out_pdf),
            report_title=f"Cutler Capital Excerpts – {pdf_path.stem}",
            source_pdf_name=source_pdf_name or pdf_path.name,
            format_style="legacy",
            letter_date=letter_date,
            source_url=source_url,  
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

def _stamp_pdf(in_pdf: Path, out_pdf: Path, stamp_pdf: Path) -> Path:
    r = PdfReader(str(in_pdf))
    s = PdfReader(str(stamp_pdf))
    w = PdfWriter()

    stamp_page = s.pages[0]

    for page in r.pages:
        _merge_page_preserve_annots(page, stamp_page)  # <-- use helper
        w.add_page(page)

    with open(out_pdf, "wb") as f:
        w.write(f)

    return out_pdf


def _build_compiled_filename(batch: str, *, incremental: bool = False, dt: Optional[datetime] = None) -> str:
    """Return a human-friendly compiled PDF name like Batch1_2025-12-04_Excerpt.pdf.

    We keep the actual quarter inside the PDF body/header; the file name is what
    interns will see and archive in their 2025/Dec folders.
    """
    if dt is None:
        dt = datetime.now()
    batch_token = batch.replace(" ", "")  # "Batch 1" -> "Batch1"
    date_str = dt.strftime("%Y-%m-%d")
    suffix = "Incremental_Excerpt" if incremental else "Excerpt"
    return f"{batch_token}_{date_str}_{suffix}.pdf"


def compile_merged(batch: str, quarter: str, collected: List[Path], *, incremental: bool = False) -> Optional[Path]:
    if not collected:
        return None

    # File name now matches your convention:
    #   BatchN_YYYY-MM-DD_Excerpt.pdf
    #   BatchN_YYYY-MM-DD_Incremental_Excerpt.pdf
    out_name = _build_compiled_filename(batch, incremental=incremental)
    out = CP_DIR / out_name

    m = PdfMerger()
    added = 0
    for p in collected:
        try:
            title = p.stem.replace('_', ' ').replace('-', ' ')
            stamped = _stamp_pdf(
                p,
                left=batch,
                mid=title,
                right=f"Run {datetime.now():%Y-%m-%d %H:%M}",
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
    # you already have these fields in your version, keep them if present
    # summary_html: Optional[str] = None
    # body_html: Optional[str] = None


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

        out.append(AnalysisArticle(
            id=art_id,
            title=title,
            published=published,
            url=link,
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

    return {
        "title": attrs.get("title", ""),
        "body_html": body_html,
        "summary_html": summary_html,
        "image_url": image_url,
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
    Seeking Alpha – Analysis digest by ticker.

    Uses:
      - sa_analysis_api.fetch_analysis_list
      - sa_analysis_api.fetch_analysis_details
      - sa_analysis_api.build_sa_analysis_digest (which calls OpenAI)
    """
    import pandas as pd
    import streamlit as st

    # Simple in-session cache for SA results: {cache_key: {"articles": ..., "digest_text": ...}}
    if "sa_cache" not in st.session_state:
        st.session_state["sa_cache"] = {}

    st.markdown("### Seeking Alpha – Analysis digest by ticker")

    # ---------- Ticker selection: up to 10, show one at a time ----------
    try:
        from tickers import tickers as CUTLER_TICKERS
        universe = sorted(list(CUTLER_TICKERS.keys()))
    except Exception:
        universe = []

    default_universe = ["AMZN", "AAPL", "MSFT", "GOOGL", "META", "TSLA", "JPM", "V", "MA", "BRK.B"]

    # Multiselect behaves like a dropdown that can hold up to 10 names
    selected_tickers = st.multiselect(
        "Select up to 10 tickers for Seeking Alpha",
        options=universe or default_universe,
        default=(universe[:3] if universe else default_universe[:3]),
        help="Choose a small set of names you want to quickly flip through.",
        max_selections=10 if hasattr(st.multiselect, "__call__") else None,  # Streamlit ignores unknown args, so safe
    )

    # Enforce max 10 tickers manually in case Streamlit version does not support max_selections
    if len(selected_tickers) > 10:
        st.warning("Please select at most 10 tickers. Only the first 10 will be used.")
        selected_tickers = selected_tickers[:10]

    # Track navigation index in session_state
    if "sa_selected_tickers_prev" not in st.session_state:
        st.session_state["sa_selected_tickers_prev"] = []
    if "sa_current_index" not in st.session_state:
        st.session_state["sa_current_index"] = 0

    # If the selection changed since last run, reset index to 0
    if st.session_state["sa_selected_tickers_prev"] != selected_tickers:
        st.session_state["sa_current_index"] = 0
    st.session_state["sa_selected_tickers_prev"] = selected_tickers

    if selected_tickers:
        idx = st.session_state["sa_current_index"]
        # Safety: clamp index if list shrank
        if idx >= len(selected_tickers):
            idx = 0
            st.session_state["sa_current_index"] = 0
        ticker = selected_tickers[idx]
        st.markdown(f"**Currently viewing:** `{ticker}`")
    else:
        # Fallback: single-ticker dropdown if nothing selected
        ticker = st.selectbox(
            "Ticker",
            universe or default_universe,
            index=0,
        )
        st.info("No watchlist selected above. Using single-ticker mode.")

    # Control how many recent articles to use
    max_articles = st.slider(
        "Number of recent analysis articles to use",
        min_value=3,
        max_value=10,
        value=5,
        help="How many recent Seeking Alpha *analysis* articles to use.",
    )

    model = st.selectbox(
        "OpenAI model for digest",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    sa_cache = st.session_state["sa_cache"]
    cache_key = f"{ticker}|{max_articles}|{model}"

    # Button now means "fetch / refresh"; cached results will show even if you don't click again
    fetch_clicked = st.button("Fetch / refresh Seeking Alpha analysis & build AI digest")

    articles = None
    digest_text = None

    # Case 1: we have cached data for this combo and user did NOT click refresh
    if cache_key in sa_cache and not fetch_clicked:
        cached = sa_cache[cache_key]
        articles = cached.get("articles")
        digest_text = cached.get("digest_text")
        if articles:
            st.info(f"Showing cached Seeking Alpha analysis for `{ticker}` (model: {model}).")
        else:
            st.info("Cached entry exists but no articles stored; please refresh.")
    # Case 2: user explicitly clicked button → fetch fresh data and overwrite cache
    elif fetch_clicked:
        if not ticker:
            st.warning("Please choose a ticker first.")
            return

        # ---------- 1) Fetch list of analysis articles ----------
        try:
            with st.spinner(f"Pulling Seeking Alpha analysis for {ticker} via RapidAPI..."):
                articles = sa_api.fetch_analysis_list(ticker, size=max_articles)
        except Exception as e:
            st.error(f"Error while fetching Seeking Alpha analysis: {e}")
            return

        if not articles:
            st.info(f"No Seeking Alpha analysis articles returned for {ticker}.")
            # store empty so we don't keep trying silently
            sa_cache[cache_key] = {"articles": [], "digest_text": None}
            return

        # ---------- 3) AI digest (delegates to sa_analysis_api) ----------
        st.markdown("#### AI Analysis Digest")
        try:
            with st.spinner("Asking OpenAI for a short analysis digest..."):
                # IMPORTANT: use the implementation from sa_analysis_api.py
                digest_text = sa_api.build_sa_analysis_digest(
                    symbol=ticker,
                    articles=articles,
                    model=model,
                )
        except Exception as e:
            st.error(f"Error while calling OpenAI: {e}")
            digest_text = None

        # store in cache
        sa_cache[cache_key] = {
            "articles": articles,
            "digest_text": digest_text,
        }

    # Case 3: no cache and user hasn't clicked yet → nothing to show
    else:
        st.info("Click the button above to fetch Seeking Alpha analysis for this ticker.")
        return

    # At this point, we expect to have `articles` (possibly from cache) and maybe `digest_text`
    if not articles:
        st.info("No articles available for this ticker / configuration.")
        return

    # ---------- 2) Show table of articles ----------
    rows = []
    for art in articles:
        date_str = art.published.split("T", 1)[0] if getattr(art, "published", "") else ""
        rows.append(
            {
                "Date": date_str,
                "Title": getattr(art, "title", ""),
                "Source": "Seeking Alpha (Analysis)",
                "URL": getattr(art, "url", ""),
            }
        )

    df = pd.DataFrame(rows)
    st.markdown("#### Recent Seeking Alpha analysis articles")
    st.dataframe(df, use_container_width=True)

    # ---------- 3) Render AI digest (using cached or fresh text) ----------
    st.markdown("#### AI Analysis Digest")
    if digest_text:
        st.markdown(digest_text)
    else:
        st.info("No AI digest available for this ticker. Try refreshing if needed.")

    # -----------------------------------------------------------
    # 4) Pull and display full article bodies (cleaned)
    # -----------------------------------------------------------
    st.markdown("#### Article bodies (full, cleaned)")

    # Helper: normalise any HTML-ish field to a single string
    def _normalize_html(part) -> str:
        if part is None:
            return ""
        if isinstance(part, list):
            return "\n".join(str(x) for x in part if x is not None)
        return str(part)

    # Only fetch bodies for a few articles to avoid hammering the API
    articles_for_bodies = articles[:5]

    for art in articles_for_bodies:
        art_id = art.id
        title = art.title or "Untitled article"

        with st.expander(title):
            st.caption("Full article (cleaned)")

            try:
                # This returns a *flat* dict:
                # {title, summary_html, body_html, images, url}
                details = sa_api.fetch_analysis_details(str(art_id))
            except Exception as e:
                st.write(f"Could not fetch article body: {e}")
                continue

            if not isinstance(details, dict) or not details:
                st.write("No article body text returned.")
                continue

            # In case you ever switch back to raw API JSON, support both shapes:
            if "data" in details:
                # raw RapidAPI payload
                data = details.get("data") or {}
                attrs = data.get("attributes") or {}
                summary_html = attrs.get("summary_html") or attrs.get("summary") or ""
                body_html = (
                    attrs.get("body_html")
                    or attrs.get("content")
                    or attrs.get("body")
                    or ""
                )
                images = attrs.get("images") or []
            else:
                # current helper output from sa_analysis_api.fetch_analysis_details
                summary_html = details.get("summary_html") or details.get("summary") or ""
                body_html = (
                    details.get("body_html")
                    or details.get("content")
                    or details.get("body")
                    or ""
                )
                images = details.get("images") or []

            # -------- Image (if present) --------
            image_url = None
            if isinstance(images, list):
                for img in images:
                    if not isinstance(img, dict):
                        continue
                    image_url = (
                        img.get("url")
                        or img.get("imageUrl")
                        or img.get("src")
                    )
                    if image_url:
                        break

            if not image_url:
                # Fallback if API ever sends a direct field
                image_url = details.get("gettyImageUrl") or details.get("imageUrl")

            if image_url:
                try:
                    st.image(image_url, use_column_width=True)
                except Exception:
                    # If Streamlit can't load it, just skip the image
                    pass

            # -------- Clean and render text --------
            combined_html = (
                _normalize_html(summary_html)
                + "\n\n"
                + _normalize_html(body_html)
            )

            if not combined_html.strip():
                st.write("No article body text returned.")
                continue

            try:
                cleaned_text = clean_sa_html_to_markdown(combined_html)
            except NameError:
                # Very simple fallback if helper is missing
                import re as _re
                tmp = _re.sub(r"<(br|p|div|li)[^>]*>", "\n", combined_html, flags=_re.I)
                tmp = _re.sub(r"<[^>]+>", "", tmp)
                tmp = tmp.replace("\xa0", " ")
                cleaned_text = _re.sub(r"\n{3,}", "\n\n", tmp).strip()

            st.write(cleaned_text)


# --- Reddit snapshot section (uses reddit34 via reddit_excerpts) ---

def _get_available_tickers_for_reddit():
    """
    Build the Reddit ticker universe from your tickers.py file.
    """
    try:
        if isinstance(tickers, dict):
            return sorted(str(sym).upper() for sym in tickers.keys())
        elif isinstance(tickers, (list, tuple, set)):
            return sorted(str(sym).upper() for sym in tickers)
    except Exception:
        pass

    # Fallback so the UI doesn’t break if something changes
    return ["AAPL", "AMZN", "MSFT", "GOOG"]


def draw_reddit_pulse_section():
    st.markdown("### Reddit pulse – weekly sentiment snapshot")
    st.caption(
        "Lightweight Reddit snapshot: we pull a small number of high-signal posts from "
        "the last week in key finance subs, filtered by ticker, to keep API usage under control."
    )

    # 1) Ticker selection from your real universe
    available_tickers = _get_available_tickers_for_reddit()
    default_index = available_tickers.index("AMZN") if "AMZN" in available_tickers else 0

    selected_ticker = st.selectbox(
        "Ticker symbol",
        options=available_tickers,
        index=default_index,
        key="reddit_ticker_select",
    )

    # 2) Simple cache so we don't hit the API on every rerun
    cache_key = "reddit_pulse_cache"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = {}

    posts_by_subreddit = st.session_state[cache_key].get(selected_ticker)

    # 3) Fetch button – ONLY here do we call the API
    run_clicked = st.button(
        f"Fetch Reddit posts for {selected_ticker}",
        use_container_width=True,
        key="reddit_fetch_button",
    )

    if run_clicked:
        with st.spinner("Querying Reddit…"):
            posts_by_subreddit = reddit_excerpts.fetch_posts_for_ticker(
                selected_ticker,
                time_window="week",  # weekly snapshot
                max_per_sub=5,       # cost control
            )
        # store result in cache
        st.session_state[cache_key][selected_ticker] = posts_by_subreddit

    # If we have no data yet (button never clicked, or API returned nothing)
    if not posts_by_subreddit:
        st.info(
            "No relevant Reddit posts found in the last week for this ticker, "
            "or the API did not return any results yet."
        )
        return

    # 4) Subreddit + extras dropdown (only subs that actually have posts)
    extras_key = "__extras__"
    subreddit_labels = []
    label_to_sub = {}

    # Core finance subs first
    core_items = [
        (k, v) for k, v in posts_by_subreddit.items()
        if k != extras_key
    ]
    core_items = sorted(core_items, key=lambda kv: kv[0].lower())

    for sub, posts in core_items:
        if not posts:
            continue
        label = f"r/{sub} ({len(posts)} posts)"
        subreddit_labels.append(label)
        label_to_sub[label] = sub

    # Extras bucket if present
    extras_posts = posts_by_subreddit.get(extras_key)
    if extras_posts:
        extras_label = (
            f"Extras – top cross-subreddit posts mentioning "
            f"${selected_ticker} ({len(extras_posts)} posts)"
        )
        subreddit_labels.append(extras_label)
        label_to_sub[extras_label] = extras_key

    if not subreddit_labels:
        st.info(
            "We fetched data from Reddit, but none of the top posts mentioned this ticker "
            "explicitly in the last week."
        )
        return

    chosen_label = st.selectbox(
        "Choose a subreddit or extras view",
        options=subreddit_labels,
        key="reddit_subreddit_select",
    )
    chosen_sub = label_to_sub[chosen_label]

    if chosen_sub == extras_key:
        st.markdown(
            f"Showing posts from **all finance subreddits** mentioning `${selected_ticker}` "
            f"(top {len(posts_by_subreddit.get(chosen_sub, []))} this week):"
        )
    else:
        st.markdown(f"Showing posts for **r/{chosen_sub}**:")

    # 5) Render posts for the chosen key
    for post in posts_by_subreddit.get(chosen_sub, []):
        title = post.title or "(no title)"
        permalink = post.permalink or ""
        score = post.score or 0
        num_comments = post.num_comments or 0

        body = (
            getattr(post, "selftext", None)
            or getattr(post, "body", None)
            or ""
        ).strip()

        header = f"{title} – Score {score} · {num_comments} comments"

        with st.expander(header, expanded=False):
            if permalink:
                url = f"https://www.reddit.com{permalink}"
                st.markdown(f"[Open on Reddit]({url})")

            if body:
                max_chars = 4000
                if len(body) > max_chars:
                    st.write(body[:max_chars] + " …")
                    st.caption(
                        "Truncated for length – open on Reddit to read the full post."
                    )
                else:
                    st.write(body)
            else:
                st.caption(
                    "This is a link post with no text body on Reddit. "
                    "Open on Reddit to read the article and full discussion."
                )

            st.markdown("---")

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
      ['2025 Q3', '2025 Q2', '2025 Q1', '2024 Q4', ...]
    Cached so we don't hit the site on every rerun.
    """
    vals: List[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
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
        # conservative fallback if the site scrape fails
        out = [
            "2025 Q3",
            "2025 Q2",
            "2025 Q1",
            "2024 Q4",
            "2024 Q3",
            "2024 Q2",
            "2024 Q1",
        ]
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
    Given the list of available quarters from the site, choose the default
    as the **last completed US quarter**, if present. If not present,
    choose the most recent available.
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

    target_year, target_q = _last_completed_us_quarter()

    for lab, year, q in parsed:
        if year < target_year or (year == target_year and q <= target_q):
            return lab

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
        st.markdown(
            "<div class='app-subtitle'>Scrape, excerpt, and compile fund letters by fund family and quarter.</div>",
            unsafe_allow_html=True,
        )


    # Sidebar: run settings
    st.sidebar.header("Run settings")

    quarter_options = get_available_quarters()
    default_q = choose_default_quarter(quarter_options)

    quarters = st.sidebar.multiselect(
        "Quarters",
        quarter_options,
        default=[default_q] if default_q else quarter_options[:1],
    )

    use_first_word = st.sidebar.checkbox(
        "Use first word for search (recommended)",
        value=True,
    )

    batch_names = list(RUNNABLE_BATCHES.keys())

    # --- Tabs (website-style nav) ---
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    tab_mf, tab_sa, tab_reddit, tab_podcast = st.tabs(
        ["Mutual Fund", "Seeking Alpha", "Reddit", "Podcast"]
    )

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    with tab_mf:
        # Main controls in a card – full run
        with st.container():
            st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
            st.markdown("#### Run scope", unsafe_allow_html=True)
            st.write("Choose whether you want a full run across all batches or a targeted test.")

            run_mode = st.radio(
                "Run mode",
                ["Run all 7 batches", "Run a specific batch"],
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
    

    with tab_reddit:
        # ---------- Reddit pulse (retail sentiment) ----------
        draw_reddit_pulse_section()


    with tab_podcast:
        # ---------- Podcast intelligence (ticker mentions across podcasts) ----------
        draw_podcast_intelligence_section()


    # Output path
    st.markdown("<div class='cc-card'>", unsafe_allow_html=True)
    st.write("**Output root folder (on this machine):**")
    st.code(str(BASE))
    st.markdown("</div>", unsafe_allow_html=True)

# Streamlit needs main() to run on import.
main()


# make_pdf.py - Cutler-branded excerpt PDF builder
from __future__ import annotations

import json
import os
import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

# ---------- Styles ----------

_base = getSampleStyleSheet()

CoverMain = ParagraphStyle(
    "CoverMain",
    parent=_base["Title"],
    fontSize=22,
    leading=28,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#4b2142"),  # Cutler purple
)

CoverSub = ParagraphStyle(
    "CoverSub",
    parent=_base["Normal"],
    fontSize=11,
    leading=14,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#6b4f7a"),
)

CoverDocTitle = ParagraphStyle(
    "CoverDocTitle",
    parent=_base["Title"],
    fontSize=18,
    leading=22,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#111827"),
)

MetaX = ParagraphStyle(
    "MetaX",
    parent=_base["Normal"],
    fontSize=9,
    leading=12,
    alignment=TA_LEFT,
    textColor=colors.HexColor("#4b5563"),
)

LeftHeader = ParagraphStyle(
    "LeftHeader",
    parent=_base["Heading2"],
    fontSize=11,
    leading=13,
    alignment=TA_CENTER,
    textColor=colors.white,
)

LeftTicker = ParagraphStyle(
    "LeftTicker",
    parent=_base["Normal"],
    fontSize=9,
    leading=11,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#E5E7EB"),
)

PagesLegacy = ParagraphStyle(
    "PagesLegacy",
    parent=_base["Normal"],
    fontSize=9,
    leading=12,
    textColor=colors.HexColor("#4b5563"),
    spaceAfter=2,
)

BodyLegacy = ParagraphStyle(
    "BodyLegacy",
    parent=_base["BodyText"],
    fontSize=10,
    leading=13,
    spaceAfter=8,
    splitLongWords=1,
    wordWrap="CJK",
)

# ---------- Utilities ----------

def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "file"


def _load_ticker_display_names(here: Path) -> Dict[str, str]:
    """
    Import tickers.py and return {ticker: preferred_display_name}.
    """
    mapping: Dict[str, str] = {}
    tp = here / "tickers.py"
    if not tp.exists():
        return mapping

    import importlib.util

    spec = importlib.util.spec_from_file_location("tickers", str(tp))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    sys.modules["tickers"] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    tickers = getattr(mod, "tickers", {})
    for tkr, names in (tickers or {}).items():
        mapping[tkr] = (names[0] if isinstance(names, list) and names else str(tkr))
    return mapping


def _read_excerpts(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_pages(raw) -> List[int]:
    pages = raw or []
    out: Set[int] = set()
    for p in pages:
        try:
            out.add(int(p))
        except Exception:
            continue
    return sorted(out)


@dataclass
class RawItem:
    ticker: str
    text: str
    pages: List[int]
    order_hint: Tuple[int, int]  # (min_page_or_big, seq)


def _flatten_raw_items(data: Dict[str, Any]) -> List[RawItem]:
    """
    Accept both shapes:
      1) {"companies":[{"ticker","name","items":[...]}]}
      2) {"TICKER":[{"text","pages"}, ...], ...}
    and return a flat list of RawItem.
    """
    items: List[RawItem] = []
    seq = 0

    if "companies" in data and isinstance(data["companies"], list):
        for comp_idx, c in enumerate(data["companies"]):
            tkr = str(c.get("ticker") or c.get("name") or "").strip() or "—"
            for it_idx, it in enumerate(c.get("items") or []):
                txt = (it.get("text") or "").strip()
                if not txt:
                    continue
                pages = _normalize_pages(it.get("pages"))
                order = (min(pages) if pages else 9999, seq)
                items.append(RawItem(ticker=tkr, text=txt, pages=pages, order_hint=order))
                seq += 1
        return items

    # flat mapping shape (what excerpt_check.py writes today)
    for tkr, lst in data.items():
        if not isinstance(lst, list):
            continue
        tkr_s = str(tkr).strip() or "—"
        for it_idx, it in enumerate(lst):
            txt = (it.get("text") or "").strip()
            if not txt:
                continue
            pages = _normalize_pages(it.get("pages"))
            order = (min(pages) if pages else 9999, seq)
            items.append(RawItem(ticker=tkr_s, text=txt, pages=pages, order_hint=order))
            seq += 1

    return items


@dataclass
class ParagraphAgg:
    text: str
    pages: Set[int]
    tickers: Set[str]
    order_hint: Tuple[int, int]


@dataclass
class Group:
    tickers: Tuple[str, ...]
    items: List[Dict[str, Any]]  # each: {"text": str, "pages": List[int], "order_hint": (..,..)}
    pages: List[int]
    first_order: Tuple[int, int]


def _aggregate_paragraphs(raw_items: List[RawItem]) -> Dict[str, ParagraphAgg]:
    """
    Aggregate duplicates by exact text, unioning tickers and pages.
    If the same paragraph appears under AMZN and TSM, it will become
    one ParagraphAgg with tickers={AMZN, TSM}.
    """
    agg: Dict[str, ParagraphAgg] = {}
    for it in raw_items:
        txt = it.text.strip()
        if not txt:
            continue
        if txt not in agg:
            agg[txt] = ParagraphAgg(
                text=txt,
                pages=set(it.pages),
                tickers={it.ticker},
                order_hint=it.order_hint,
            )
        else:
            a = agg[txt]
            a.pages.update(it.pages)
            a.tickers.add(it.ticker)
            if it.order_hint < a.order_hint:
                a.order_hint = it.order_hint
    return agg


def _groups_from_agg(agg: Dict[str, ParagraphAgg]) -> List[Group]:
    """
    Turn paragraph aggregations into groups keyed by ticker-set.
    Each Group can contain multiple paragraphs that share the same ticker set.
    """
    by_tickerset: Dict[frozenset, List[Dict[str, Any]]] = {}
    for para in agg.values():
        key = frozenset(para.tickers)
        item = {
            "text": para.text,
            "pages": sorted(para.pages),
            "order_hint": para.order_hint,
        }
        by_tickerset.setdefault(key, []).append(item)

    groups: List[Group] = []
    for tickerset, items in by_tickerset.items():
        items.sort(key=lambda x: x["order_hint"])
        pages_union = sorted({p for it in items for p in it["pages"]})
        first_order = min(it["order_hint"] for it in items)
        groups.append(
            Group(
                tickers=tuple(sorted(tickerset)),
                items=items,
                pages=pages_union,
                first_order=first_order,
            )
        )

    groups.sort(key=lambda g: g.first_order)
    return groups


def _pages_text(pages: List[int]) -> str:
    return ", ".join(str(p) for p in pages) if pages else "—"


def _chunk_text(txt: str, max_words: int = 140) -> List[str]:
    """
    Split long text into reasonably sized paragraphs.
    """
    txt = txt.strip()
    if not txt:
        return []
    parts = re.split(r"(?<=[.!?])\s+", txt)
    out: List[str] = []
    cur: List[str] = []
    n = 0
    for s in parts:
        w = len(s.split())
        if n + w > max_words and cur:
            out.append(" ".join(cur))
            cur = [s]
            n = w
        else:
            cur.append(s)
            n += w
    if cur:
        out.append(" ".join(cur))
    return out


def _humanize_source_name(source_pdf_name: str) -> str:
    """
    Turn a raw filename like 'Q325-abrdn-Emerging-Markets-Fund-Commentary-1.pdf'
    into 'ABRDN Emerging Markets Fund Commentary'.
    """
    stem = Path(source_pdf_name).stem
    tokens = re.split(r"[_\-]+", stem)
    cleaned: List[str] = []
    for tok in tokens:
        if not tok:
            continue
        # drop obvious quarter / version prefixes
        if re.match(r"^[1-4]Q\d{2,4}$", tok, re.IGNORECASE):
            continue
        if re.match(r"^Q\d{2,4}$", tok, re.IGNORECASE):
            continue
        if tok.lower() in {"commentary", "quarterly", "report", "fund"}:
            cleaned.append(tok.capitalize())
        else:
            cleaned.append(tok)

    if not cleaned:
        cleaned = [stem.replace("_", " ").replace("-", " ")]

    def smart_title(word: str) -> str:
        if word.isupper():
            return word
        if len(word) <= 3:
            return word.upper()
        return word.capitalize()

    return " ".join(smart_title(w) for w in cleaned)


# ---------- AI relevance scoring (optional) ----------

# Rating scale:
# 5 = directly about the company (thesis, actions, catalysts, performance, risks, outlook)
# 4 = meaningful discussion but less direct/central
# 3 = moderate relevance (some context, light commentary)
# 2 = weak relevance (passing mention, minor context)
# 1 = essentially irrelevant (lists/holdings/indices/tickers with no substance)

_RATING_COLORS = {
    5: colors.HexColor("#DFF3E3"),  # green-ish
    4: colors.HexColor("#EAF7ED"),
    3: colors.HexColor("#FFF7DB"),  # light yellow
    2: colors.HexColor("#FFE7D6"),  # light orange
    1: colors.HexColor("#FDE2E2"),  # light red
}

def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()

def _openai_score_paragraph(
    *,
    company_label: str,
    tickers: str,
    paragraph: str,
    model: str,
    cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Return dict: {rating:int, rationale:str}. Uses cache keyed by content hash."""
    key = _hash_key(model, company_label, tickers, paragraph)
    if key in cache:
        return cache[key]

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_APIKEY")
    if not api_key:
        out = {"rating": 3, "rationale": "No OPENAI_API_KEY set; defaulted to 3."}
        cache[key] = out
        return out

    try:
        # OpenAI Python SDK v1.x
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=api_key)

        sys_prompt = (
            "You are a strict equity-research analyst helping a buy-side firm. "
            "Score how relevant a mutual-fund commentary paragraph is to the target company. "
            "Output ONLY valid JSON."
        )

        user_prompt = f"""Target company: {company_label}
Tickers: {tickers}

Paragraph:
{paragraph}

Return JSON with:
- rating: integer 1-5 (5 = directly about the company with meaningful discussion; 1 = basically irrelevant / only a list mention)
- rationale: 1 short sentence explaining why.
Rules:
- If the paragraph discusses position sizing, buy/sell, thesis, catalysts, risks, fundamentals, management, macro impact on the company: rating 4-5.
- If it's a passing mention, sector list, index/holdings list, or name-drop without analysis: rating 1-2.
- Be conservative (prefer lower ratings unless clearly substantive).
"""

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        content = (resp.choices[0].message.content or "").strip()
        # best-effort JSON parse (model should return JSON)
        parsed = json.loads(content)
        rating = int(parsed.get("rating", 3))
        rating = 1 if rating < 1 else 5 if rating > 5 else rating
        rationale = str(parsed.get("rationale", "")).strip()

        out = {"rating": rating, "rationale": rationale}
        cache[key] = out
        return out
    except Exception as e:
        out = {"rating": 3, "rationale": f"AI scoring failed; defaulted to 3. ({e})"}
        cache[key] = out
        return out

# ---------- Rendering ----------

def _render_group_block_table(
    story: List[Any],
    group: Group,
    name_map: Dict[str, str],
) -> None:
    """
    Legacy renderer: two-column table.
    Left: combined company name(s) + tickers.
    Right: Pages + paragraphs.
    """
    display_names = [name_map.get(t, t) for t in group.tickers]
    left_title = ", ".join(display_names)
    left_tickers = ", ".join(group.tickers)

    left_html = f"{left_title}<br/><font size='9' color='#D1D5DB'>({left_tickers})</font>"
    header_pages_text = _pages_text(group.pages)

    rows: List[List[Any]] = []
    rows.append(
        [
            Paragraph(left_html, LeftHeader),
            Paragraph(f"Pages: {header_pages_text}", PagesLegacy),
        ]
    )

    # Body rows
    for it in group.items:
        txt = (it.get("text") or "").strip()
        if not txt:
            continue
        ipages = it.get("pages") or []
        if ipages and set(ipages) != set(group.pages):
            rows.append(["", Paragraph(f"Pages: {', '.join(map(str, ipages))}", PagesLegacy)])
        for chunk in _chunk_text(txt, max_words=140):
            rows.append(["", Paragraph(chunk, BodyLegacy)])

    table = Table(
        rows,
        colWidths=[2.3 * 72, None],  # 2.3" left, rest right
        repeatRows=1,
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#4b2142")),
                ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#e5e7eb")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#f3f4f6")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        ),
    )

    story.append(table)
    story.append(Spacer(1, 10))


def _render_group_block_compact(
    story: List[Any],
    group: Group,
    name_map: Dict[str, str],
    *,
    ai_score: bool,
    ai_model: str,
    ai_cache: Dict[str, Dict[str, Any]],
) -> None:
    """
    Compact renderer (default): full-width blocks with a single header line,
    then numbered paragraphs below. This significantly reduces page count
    versus nested tables.
    """

    # Styles created lazily so we don't mutate global stylesheet at import time
    header_style = ParagraphStyle(
        "GroupHeaderCompact",
        parent=_base["Heading3"],
        fontSize=10.5,
        leading=13,
        textColor=colors.white,
        backColor=colors.HexColor("#4b2142"),
        leftIndent=0,
        rightIndent=0,
        spaceBefore=6,
        spaceAfter=6,
        borderPadding=(6, 8, 6, 8),
    )

    meta_style = ParagraphStyle(
        "GroupMetaCompact",
        parent=_base["Normal"],
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#6b4f7a"),
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "BodyCompact",
        parent=_base["BodyText"],
        fontSize=9.5,
        leading=12.5,
        spaceAfter=5,
        splitLongWords=1,
        wordWrap="CJK",
    )

    num_style = ParagraphStyle(
        "BodyCompactNumber",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-14,
    )

    # Per-rating highlight styles (only used when ai_score=True)
    rating_styles: Dict[int, ParagraphStyle] = {}
    if ai_score:
        for r in (1, 2, 3, 4, 5):
            rating_styles[r] = ParagraphStyle(
                f"NumStyleR{r}",
                parent=num_style,
                backColor=_RATING_COLORS.get(r),
                borderPadding=(4, 6, 4, 6),
                spaceAfter=6,
            )

    # Group header/meta
    display_names = [name_map.get(t, t) for t in group.tickers]
    title = ", ".join(display_names)
    tickers = ", ".join(group.tickers)
    pages = _pages_text(group.pages)

    story.append(Paragraph(escape(title), header_style))
    story.append(
        Paragraph(
            f"Tickers: {escape(tickers)} &nbsp;&nbsp;|&nbsp;&nbsp; Pages: {pages}",
            meta_style,
        )
    )

    # Numbered paragraphs
    n = 1
    company_label = f"{title} ({tickers})"
    for it in group.items:
        txt = (it.get("text") or "").strip()
        if not txt:
            continue

        for chunk in _chunk_text(txt, max_words=140):
            safe_chunk = escape(chunk)

            rating: Optional[int] = None
            rationale = ""
            if ai_score:
                scored = _openai_score_paragraph(
                    company_label=company_label,
                    tickers=tickers,
                    paragraph=chunk,
                    model=ai_model,
                    cache=ai_cache,
                )
                try:
                    rating = int(scored.get("rating", 3) or 3)
                except Exception:
                    rating = 3
                rating = 1 if rating < 1 else 5 if rating > 5 else rating
                rationale = str(scored.get("rationale", "") or "").strip()

            if rating is not None:
                style = rating_styles.get(rating, num_style)
                prefix = f"[{rating}] "
                story.append(Paragraph(f"{n}. {prefix}{safe_chunk}", style))
                if rationale:
                    story.append(
                        Paragraph(
                            f"<font size='8' color='#6b4f7a'>Why: {escape(rationale)}</font>",
                            meta_style,
                        )
                    )
            else:
                story.append(Paragraph(f"{n}. {safe_chunk}", num_style))

            n += 1

    story.append(Spacer(1, 8))



def _render_group_block(
    story: List[Any],
    group: Group,
    name_map: Dict[str, str],
    *,
    format_style: str = "legacy",
    ai_score: bool = False,
    ai_model: str = "gpt-4o-mini",
    ai_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Dispatch between legacy table view and compact view."""
    if (format_style or "").lower() in {"table", "grid", "legacy_table"}:
        _render_group_block_table(story, group, name_map)
        return

    _render_group_block_compact(
        story,
        group,
        name_map,
        ai_score=ai_score,
        ai_model=ai_model,
        ai_cache=ai_cache or {},
    )



# ---------- Main builder ----------


def build_pdf(
    excerpts_json_path: str = "excerpts_clean.json",
    output_pdf_path: str = "excerpts_report.pdf",
    report_title: str = "Excerpts",
    source_pdf_name: Optional[str] = None,
    format_style: str = "legacy",
    letter_date: Optional[str] = None,
    source_url: Optional[str] = None,
    *,
    ai_score: bool = False,
    ai_model: str = "gpt-4o-mini",
) -> Optional[str]:

    here = Path(".").resolve()
    data = _read_excerpts(Path(excerpts_json_path))
    raw_items = _flatten_raw_items(data)
    if not raw_items:
        print("SKIP: No narrative excerpts found; not writing a PDF.")
        return None

    # Aggregate and group paragraphs (de-dup across tickers)
    agg = _aggregate_paragraphs(raw_items)
    groups = _groups_from_agg(agg)
    if not groups:
        print("SKIP: No narrative excerpts after de-dup; not writing a PDF.")
        return None

    # Map tickers to nicer display names if available
    name_map = _load_ticker_display_names(here)

    # Determine source name
    if not source_pdf_name:
        # best-effort: pick most recent PDF in working dir
        pdfs = sorted(here.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        source_pdf_name = pdfs[0].name if pdfs else "Unknown.pdf"

    commentary_title = _humanize_source_name(source_pdf_name)

    # Document
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=LETTER,
        leftMargin=0.75 * 72,
        rightMargin=0.75 * 72,
        topMargin=0.75 * 72,
        bottomMargin=0.75 * 72,
    )

    story: List[Any] = []
    ai_cache: Dict[str, Dict[str, Any]] = {}
    now = datetime.now(ZoneInfo("America/New_York"))

    # --- Cover / header ---
    # (Compact by default to reduce page count. Use format_style='table' for the old look.)
    story.append(Spacer(1, 0.25 * 72))
    story.append(Paragraph("Cutler Capital Excerption", CoverMain))
    story.append(Spacer(1, 0.1 * 72))
    story.append(Paragraph(f"for {now:%B %d, %Y}", CoverSub))
    story.append(Spacer(1, 0.2 * 72))
    safe_title = escape(commentary_title)
    if source_url:
        safe_url = escape(str(source_url))
        story.append(Paragraph(f'<a href="{safe_url}">{safe_title}</a>', CoverDocTitle))
    else:
        story.append(Paragraph(safe_title, CoverDocTitle))
    story.append(Spacer(1, 0.15 * 72))
    story.append(Paragraph(f"Run: {now:%Y-%m-%d %H:%M:%S}", MetaX))
    story.append(Paragraph(f"Source: {source_pdf_name}", MetaX))
    if source_url:
        safe_url = escape(str(source_url))
        story.append(Paragraph(f'Source link: <a href="{safe_url}">{safe_url}</a>', MetaX))
    if letter_date:
        story.append(Paragraph(f"Letter Date: {letter_date}", MetaX))
    story.append(Spacer(1, 0.25 * 72))

    # --- Excerpt groups ---
    for grp in groups:
        _render_group_block(story, grp, name_map, format_style=format_style, ai_score=ai_score, ai_model=ai_model, ai_cache=ai_cache)

    doc.build(story)
    print(f"PDF created: {output_pdf_path}")
    return output_pdf_path


if __name__ == "__main__":
    build_pdf(
        excerpts_json_path="excerpts_clean.json",
        output_pdf_path="excerpts_report.pdf",
        report_title="Excerpts",
        source_pdf_name=None,
        format_style="legacy",
        letter_date=None,
    )

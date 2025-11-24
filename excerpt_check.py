# excerpt_check.py (narrative-filtered)
# Requires: pip install pymupdf
# Usage:    python excerpt_check.py ".\baron-funds-quarterly-report-6.30.2025.pdf"

import json
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter

try:
    import fitz  # PyMuPDF
except ImportError as e:
    raise SystemExit(
        "PyMuPDF (fitz) is required.\nInstall with:  pip install pymupdf\n"
        f"ImportError: {e}"
    )

# --------------------------
# Normalization + paragraph end heuristics
# --------------------------

SENTENCE_END_RE = re.compile(r'[.!?…]["”’\)\]]*\s*$')

def normalize_block_text(text: str) -> str:
    t = text.replace('\r\n', '\n').replace('\r', '\n')
    # Fix soft wraps: "exam-\nple" -> "example"
    t = re.sub(r'-\n(?=\w)', '', t)
    # Temporarily mark blank-line breaks
    t = t.replace('\n\n', '¶¶')
    # Collapse remaining newlines to spaces
    t = re.sub(r'\n+', ' ', t)
    # Restore blank-line paragraph hints
    t = t.replace('¶¶', '\n\n')
    # Squash spaces
    t = re.sub(r'[ \t]+', ' ', t).strip()
    return t

def looks_like_paragraph_ending(text: str) -> bool:
    parts = [p.strip() for p in text.split('\n') if p.strip()]
    if not parts:
        return True
    last = parts[-1]
    return bool(SENTENCE_END_RE.search(last))

# --------------------------
# Narrative vs. Table filter
# --------------------------

HARD_EXCLUDE_PHRASES = [
    "top detractors from performance",
    "top contributors to performance",
    "top contributors from performance",
    "performance attribution",
    "year acquired",
    "market cap when acquired",
    "quarter end market cap",
    "total return (%)",
    "contribution to return (%)",
    "retail shares:", "institutional shares:", "r6 shares:",
]

def is_hard_exclude(text: str) -> bool:
    t = text.lower()
    return any(ph in t for ph in HARD_EXCLUDE_PHRASES)

TABLE_HEADER_PHRASES = [
    "portfolio of investments",
    "percent of net assets",
    "shares cost value",
    "shares cost  value",
    "common stocks",
    "warrants",
    "preferred",
    "total investments",
    "repurchase agreement",
    "adrs", "adr",
    "portfolio structure",
    "top 10 holdings",
    "top five holdings",
    "portfolio holdings",
    "net assets",
    "market value",
    "principal amount",
    "mortgage reits",
    "data center reits",
    "reit", "reits",
    "msci", "russell", "s&p 500",
]

TABLE_COLUMNY_HINTS = [
    r'\b[A-Z][a-zA-Z&.\- ]+\s+\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',  # Name + number
    r'\bTotal [A-Za-z].*?\d',                                   # "Total France 31,454,781"
    r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b',                        # 1,234,567.89
]

CURRENCY_RE = re.compile(r'[$€£¥]|USD|EUR|GBP|JPY', re.I)
PERCENT_RE = re.compile(r'\b\d+(?:\.\d+)?\s?%')
NUMBER_RE = re.compile(r'\b\d+(?:[.,]\d+)?\b')
MANY_TABS_DOTS_RE = re.compile(r' {2,}|\t|·{2,}|\.{2,}')

# Common English stopwords as a weak proxy for prose
STOPWORDS = set("""
the and of to in that we for with on as is are was will would can our not this
be have it by from at which into over more their about or you they than if when
""".split())

def count_matches(patterns, text):
    return sum(len(re.findall(p, text)) for p in patterns)

def narrative_score(text: str):
    # basic stats
    chars = len(text)
    words = re.findall(r"[A-Za-z']+", text)
    n_words = len(words)
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)

    # densities
    digit_ratio = (digits / max(chars, 1))
    letter_ratio = (letters / max(chars, 1))
    money_count = len(CURRENCY_RE.findall(text))
    pct_count = len(PERCENT_RE.findall(text))
    number_count = len(NUMBER_RE.findall(text))
    tabdot_count = len(MANY_TABS_DOTS_RE.findall(text))
    header_hits = sum(1 for ph in TABLE_HEADER_PHRASES if ph in text.lower())
    col_hint_hits = count_matches(TABLE_COLUMNY_HINTS, text)

    # stopwords presence
    stop_count = sum(1 for w in words if w.lower() in STOPWORDS)

    # sentences
    has_sentence_end = bool(re.search(r'[.!?…]["”’\)\]]?\s', text))

    # ---- Early hard drops for attribution/table-y blocks ----
    if is_hard_exclude(text):
        return False, -99, "hard_exclude_phrase"

    # extremely numbery + multiple percents is usually attribution/holdings list
    if number_count >= 40 and pct_count >= 2:
        return False, -98, f"numbers={number_count}, percents={pct_count}"

    # ---- Scoring ----
    score = 0
    if has_sentence_end:             score += 2
    if letter_ratio >= 0.65:         score += 2
    if stop_count >= max(3, n_words * 0.05):  score += 2
    if n_words >= 25:                score += 1

    # penalties
    score -= int(digit_ratio > 0.18) * 2
    score -= min(money_count, 4)
    score -= min(pct_count // 2, 4)
    score -= min(number_count // 15, 4)
    score -= min(tabdot_count // 2, 4)
    score -= min(header_hits, 3) * 2
    score -= min(col_hint_hits, 3) * 2

    # decision
    reasons = []
    if digit_ratio > 0.18: reasons.append(f"digit_ratio={digit_ratio:.2f}")
    if money_count:        reasons.append(f"currency={money_count}")
    if pct_count:          reasons.append(f"percents={pct_count}")
    if number_count > 25:  reasons.append(f"numbers={number_count}")
    if tabdot_count:       reasons.append(f"tab/dots={tabdot_count}")
    if header_hits:        reasons.append(f"hdrs={header_hits}")
    if col_hint_hits:      reasons.append(f"colhints={col_hint_hits}")
    if not has_sentence_end: reasons.append("no_sentence_end")
    if letter_ratio < 0.65: reasons.append(f"letter_ratio={letter_ratio:.2f}")
    if stop_count < max(3, n_words * 0.05):
        reasons.append(f"low_stopwords={stop_count}/{n_words}")

    keep = score >= 2
    return keep, score, ", ".join(reasons)


# --------------------------
# Regex compilation per ticker
# --------------------------

def compile_keyword_regexes(tickers_dict: dict) -> dict:
    rx_by_ticker = {}
    for ticker, keywords in tickers_dict.items():
        alts = []
        for kw in keywords:
            kw_esc = re.escape(kw)
            if re.match(r'^[A-Za-z0-9.&\-]+$', kw):
                alts.append(rf'(?<!\w){kw_esc}(?!\w)')
            else:
                alts.append(kw_esc)
        pattern = re.compile(r'(' + '|'.join(alts) + r')', re.IGNORECASE)
        rx_by_ticker[ticker] = pattern
    return rx_by_ticker

def load_tickers():
    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    try:
        from tickers import tickers  # type: ignore
    except Exception as e:
        raise SystemExit(f"Failed to import tickers from tickers.py: {e}")
    if not isinstance(tickers, dict):
        raise SystemExit("tickers.py must define a dict named `tickers`.")
    return tickers

# --------------------------
# PDF block reading & stitching
# --------------------------

def read_pdf_blocks(pdf_path: Path):
    doc = fitz.open(pdf_path)
    blocks = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        for b in page.get_text("blocks"):
            if len(b) < 5:
                continue
            text = normalize_block_text(b[4])
            if not text:
                continue
            blocks.append({
                'page': pno,
                'text': text,
                'bbox': tuple(b[:4]),
            })
    doc.close()
    return blocks

def expand_to_full_paragraph(blocks, idx):
    n = len(blocks)
    start = idx
    while start > 0:
        prev_text = blocks[start - 1]['text']
        if looks_like_paragraph_ending(prev_text):
            break
        start -= 1
    end = idx
    while end < n - 1:
        cur_text = blocks[end]['text']
        if looks_like_paragraph_ending(cur_text):
            break
        end += 1
    if end < n and not looks_like_paragraph_ending(blocks[end]['text']) and end < n - 1:
        end += 1
    parts, pages = [], set()
    for j in range(start, end + 1):
        t = blocks[j]['text'].strip()
        if t:
            parts.append(t)
            pages.add(blocks[j]['page'])
    stitched = ' '.join(parts)
    stitched = re.sub(r'[ \t]+', ' ', stitched).strip()
    return stitched, sorted(pages)

# --------------------------
# Main excerption with narrative filtering
# --------------------------

def excerpt_pdf_for_tickers(pdf_path: str, debug=False):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    tickers_dict = load_tickers()
    rx_by_ticker = compile_keyword_regexes(tickers_dict)
    blocks = read_pdf_blocks(pdf_path)

    results_all = defaultdict(list)     # raw (pre-filter) by ticker
    results_clean = defaultdict(list)   # narrative-only by ticker
    summary_hits = []

    # Pre-scan matches by block, then stitch and filter stitched paragraphs
    for i, blk in enumerate(blocks):
        text = blk['text']
        for ticker, rx in rx_by_ticker.items():
            if rx.search(text):
                para, pages = expand_to_full_paragraph(blocks, i)
                hits = [m.group(0) for m in rx.finditer(para)]
                payload = {'text': para, 'pages': pages, 'hits': hits}
                # De-dupe within ticker by text
                if all(payload['text'] != existing['text'] for existing in results_all[ticker]):
                    results_all[ticker].append(payload)

    # Filter narrative paragraphs
    dropped_reasons_counter = Counter()
    for ticker, paras in results_all.items():
        for item in paras:
            keep, score, reasons = narrative_score(item['text'])
            item_with_meta = dict(item)
            item_with_meta['narrative_score'] = score
            item_with_meta['narrative_reasons'] = reasons
            if keep:
                results_clean[ticker].append(item_with_meta)
                summary_hits.append((ticker, item['pages'], len(item['hits']), score))
            else:
                if debug:
                    print(f"[DROP] {ticker} pgs {','.join(str(p+1) for p in item['pages'])}: {reasons}")
                for r in (reasons or "unspecified").split(', '):
                    if r:
                        dropped_reasons_counter[r] += 1

    # Write outputs
    out_md_raw = pdf_path.with_name("excerpts.md")
    out_json_raw = pdf_path.with_name("excerpts.json")
    out_md = pdf_path.with_name("excerpts_clean.md")
    out_json = pdf_path.with_name("excerpts_clean.json")

    # Raw (for reference)
    with open(out_md_raw, "w", encoding="utf-8") as f:
        f.write(f"# Excerpts for: {pdf_path.name}\n\n")
        for ticker in sorted(results_all.keys()):
            f.write(f"## {ticker}\n\n")
            if not results_all[ticker]:
                f.write("_No matches found._\n\n")
                continue
            for k, item in enumerate(results_all[ticker], 1):
                pages_h = ", ".join(str(p + 1) for p in item['pages'])
                f.write(f"**{k}. Pages:** {pages_h}\n\n")
                f.write(item['text'] + "\n\n")

    with open(out_json_raw, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)

    # Clean (narrative only)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Narrative Excerpts for: {pdf_path.name}\n\n")
        for ticker in sorted(results_clean.keys()):
            f.write(f"## {ticker}\n\n")
            if not results_clean[ticker]:
                f.write("_No narrative matches kept._\n\n")
                continue
            for k, item in enumerate(results_clean[ticker], 1):
                pages_h = ", ".join(str(p + 1) for p in item['pages'])
                f.write(f"**{k}. Pages:** {pages_h}  |  **score:** {item['narrative_score']}\n\n")
                f.write(item['text'] + "\n\n")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False)

    # Summary print
    total_tickers = len([t for t in results_clean if results_clean[t]])
    total_excerpts = sum(len(v) for v in results_clean.values())
    print(f"Cleaned. Tickers with narrative matches: {total_tickers}, narrative excerpts: {total_excerpts}")
    print(f"Wrote: {out_md.name} and {out_json.name}")
    # Top hits by score then hit count
    top = sorted(summary_hits, key=lambda x: (-x[3], -x[2], x[0]))[:10]
    if top:
        print("\nTop narrative hits (ticker, pages, hits, score):")
        for t, pgs, cnt, sc in top:
            pgs_h = [str(x + 1) for x in pgs]
            print(f"  {t:6s}  pages {','.join(pgs_h):<10s}  hits {cnt:<2d}  score {sc}")
    # Drop reasons (if many tables were removed)
    if dropped_reasons_counter:
        print("\nMost common drop reasons:")
        for reason, cnt in dropped_reasons_counter.most_common(10):
            print(f"  {reason:24s} x{cnt}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python excerpt_check.py \"/path/to/your.pdf\"")
        sys.exit(1)
    # Set debug=True to print each dropped paragraph + reason
    excerpt_pdf_for_tickers(sys.argv[1], debug=True)

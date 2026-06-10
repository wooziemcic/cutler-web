from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


CATEGORIES = ("Banks", "Companies", "Credit", "Green Street", "Research")
SUPPORTED_TEXT_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".txt", ".md", ".csv"}
JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
JUNK_PARTS = {"__macosx", ".git", ".svn", "__pycache__"}

BROKER_PATTERNS = [
    ("Morgan Stanley", r"\b(?:MORGAN\s+STANLEY|MS)\b"),
    ("Gimme Credit", r"\bGIMME\s+CREDIT\b"),
    ("Green Street", r"\bGREEN\s+STREET\b"),
    ("Morningstar", r"\bMORNINGSTAR\b"),
    ("JPM", r"\b(?:JPM|J\.?P\.?\s*MORGAN)\b"),
    ("GS", r"\b(?:GS|GOLDMAN\s+SACHS)\b"),
    ("Barclays", r"\bBARCLAYS\b"),
    ("RBC", r"\bRBC\b"),
    ("Jefferies", r"\bJEFFERIES\b"),
    ("Hovde", r"\bHOVDE\b"),
    ("Brean", r"\bBREAN\b"),
    ("Evercore", r"\bEVERCORE\b"),
    ("Mizuho", r"\bMIZUHO\b"),
    ("Guggenheim", r"\bGUGGENHEIM\b"),
    ("WF", r"\b(?:WF|WELLS\s+FARGO)\b"),
    ("Stifel", r"\bSTIFEL\b"),
    ("Deutsche", r"\bDEUTSCHE\b"),
    ("Cantor", r"\bCANTOR\b"),
    ("Citizens", r"\bCITIZENS\b"),
    ("BMO", r"\bBMO\b"),
    ("UBS", r"\bUBS\b"),
    ("Citi", r"\b(?:CITI|CITIGROUP)\b"),
    ("BofA", r"\b(?:BOFA|BANK\s+OF\s+AMERICA|BAML)\b"),
    ("Truist", r"\bTRUIST\b"),
    ("Piper", r"\bPIPER\b"),
    ("Wedbush", r"\bWEDBUSH\b"),
]

DOCUMENT_TYPES = [
    ("earnings presentation", (r"\bearnings?\s+presentation\b", r"\binvestor\s+presentation\b")),
    ("earnings supplement", (r"\bearnings?\s+supplement\b",)),
    ("earnings release", (r"\bearnings?\s+release\b", r"\bresults?\s+release\b")),
    ("prepared remarks", (r"\bprepared\s+remarks?\b",)),
    ("transcript", (r"\btranscript\b", r"\bearnings?\s+call\b")),
    ("10-Q/10Q", (r"\b10[\s_-]?q\b",)),
    ("8-K/8K", (r"\b8[\s_-]?k\b",)),
    ("current report", (r"\bcurrent\s+report\b",)),
    ("credit report", (r"\bcredit\s+(?:report|update|research)\b",)),
    ("analyst report", (r"\banalyst\s+report\b", r"\bequity\s+research\b")),
    ("sector report", (r"\bsector\s+(?:report|update|outlook)\b",)),
    ("industry report", (r"\bindustry\s+(?:report|update|outlook)\b",)),
    ("company overview", (r"\bcompany\s+overview\b", r"\bcompany\s+profile\b")),
    ("data supplement", (r"\bdata\s+supplement\b",)),
    ("MD&A", (r"\bmd\s*&\s*a\b", r"\bmanagement.?s discussion\b")),
    ("press release", (r"\bpress\s+release\b",)),
    ("dividend announcement", (r"\bdividend\b",)),
    ("insider trading", (r"\binsider\s+(?:trading|transaction|buy|sale)\b",)),
    ("fear and greed", (r"\bfear\s+(?:and|&)\s+greed\b",)),
    ("recommendation change", (r"\b(?:recommendation|upgrade|downgrade|initiated)\b",)),
    ("rating change", (r"\brating\s+(?:change|upgrade|downgrade)\b",)),
    ("guidance", (r"\bguidance\b",)),
    ("estimate revision", (r"\bestimate\s+(?:revision|change|increase|decrease)\b",)),
]

FINANCE_KEYWORDS = {
    "earnings", "revenue", "margin", "guidance", "estimate", "credit", "rating",
    "dividend", "liquidity", "leverage", "capital", "cash flow", "valuation",
    "outlook", "quarter", "results", "transcript", "10-q", "8-k", "research",
    "recommendation", "sector", "upgrade", "downgrade",
}

TICKER_STOPWORDS = {
    "PDF", "XLSX", "XLS", "CSV", "TXT", "MD", "Q", "FY", "FQ", "USD", "US",
    "USA", "UK", "EU", "SEC", "CEO", "CFO", "EPS", "EBITDA", "NAV", "REIT",
    "JPM", "GS", "MS", "RBC", "WF", "BMO", "UBS", "CITI", "BOFA", "BAML", "MD&A",
    "BANKS", "COMPANIES", "CREDIT", "RESEARCH", "ROOT", "GREEN", "STREET",
}

DOCUMENT_TYPE_SCORES = {
    "earnings release": 25,
    "earnings presentation": 24,
    "earnings supplement": 23,
    "transcript": 23,
    "10-Q/10Q": 24,
    "8-K/8K": 20,
    "current report": 18,
    "credit report": 22,
    "analyst report": 21,
    "sector report": 22,
    "Green Street report": 24,
    "industry report": 18,
    "prepared remarks": 22,
    "guidance": 24,
    "estimate revision": 23,
    "recommendation change": 26,
    "rating change": 21,
    "press release": 14,
    "data supplement": 15,
    "MD&A": 20,
    "company overview": 12,
    "dividend announcement": 13,
    "insider trading": 12,
    "fear and greed": 8,
    "other": 4,
}

INVENTORY_COLUMNS = [
    "category", "ticker", "company_or_identifier", "source_or_broker", "document_type",
    "file_name", "file_extension", "file_size_mb", "relative_path", "modified_date",
    "extraction_status", "extracted_path",
]
RELEVANCE_COLUMNS = INVENTORY_COLUMNS + ["relevance_score", "priority_level", "reason_for_score"]


def create_session_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="cutler_daily_research_"))


def remove_session_dir(path: str | Path | None) -> None:
    if not path:
        return
    try:
        target = Path(path).resolve()
        if target.exists() and target.name.startswith("cutler_daily_research_"):
            shutil.rmtree(target)
    except Exception:
        pass


def _is_junk(parts: Iterable[str], name: str) -> bool:
    lowered = {p.lower() for p in parts}
    return (
        name.lower() in JUNK_NAMES
        or name.startswith("._")
        or bool(lowered & JUNK_PARTS)
    )


def _safe_member_path(root: Path, member_name: str) -> Optional[Path]:
    normalized = member_name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        return None
    target = (root / Path(*pure.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target


def safe_extract_zip(
    zip_path: str | Path,
    extract_root: str | Path,
    *,
    max_total_uncompressed: int = 2 * 1024 * 1024 * 1024,
    max_member_size: int = 500 * 1024 * 1024,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    root = Path(extract_root)
    root.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    extracted_total = 0

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            rel = info.filename.replace("\\", "/")
            pure = PurePosixPath(rel)
            base = pure.name
            status = "extracted"
            target = _safe_member_path(root, rel)

            if target is None:
                status = "skipped_unsafe_path"
            elif _is_junk(pure.parts, base):
                status = "skipped_junk"
            elif info.file_size > max_member_size:
                status = "skipped_too_large"
            elif extracted_total + info.file_size > max_total_uncompressed:
                status = "skipped_total_size_limit"

            if status == "extracted":
                try:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    extracted_total += info.file_size
                except Exception as exc:
                    status = f"failed: {type(exc).__name__}"
                    warnings.append(f"{rel}: {exc}")

            records.append(
                {
                    "relative_path": rel,
                    "file_name": base,
                    "file_size": int(info.file_size),
                    "modified_date": _zip_modified_date(info),
                    "extraction_status": status,
                    "extracted_path": str(target) if target and status == "extracted" else "",
                }
            )
    return records, warnings


def _zip_modified_date(info: zipfile.ZipInfo) -> str:
    try:
        return datetime(*info.date_time).isoformat(timespec="seconds")
    except Exception:
        return ""


def detect_category(relative_path: str) -> str:
    parts = [p.lower() for p in PurePosixPath(relative_path.replace("\\", "/")).parts[:-1]]
    for category in CATEGORIES:
        label = category.lower()
        if label in parts or (category == "Green Street" and {"green", "street"} <= set(parts)):
            return category
    joined = " ".join(parts)
    for category in CATEGORIES:
        if category.lower() in joined:
            return category
    return "Root"


def detect_source(text: str) -> str:
    normalized = _normalize_metadata_text(text)
    for label, pattern in BROKER_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return label
    return ""


def _normalize_metadata_text(text: str) -> str:
    normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    normalized = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", normalized)
    normalized = re.sub(r"[_\-.]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def detect_document_type(text: str, *, category: str = "Root", source: str = "") -> str:
    normalized = _normalize_metadata_text(text)
    for label, patterns in DOCUMENT_TYPES:
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
            return label
    if category == "Credit" and source:
        return "credit report"
    if category == "Green Street":
        return "Green Street report" if source == "Green Street" else "sector report"
    if source and category in {"Research", "Banks", "Companies", "Root"}:
        return "analyst report"
    return "other"


def detect_ticker(
    file_name: str,
    known_tickers: Optional[set[str]] = None,
    *,
    source: str = "",
) -> str:
    stem = Path(file_name).stem
    tokens = re.findall(r"(?<![A-Za-z0-9])[A-Z]{1,5}(?![A-Za-z0-9])", stem)
    candidates = [t for t in tokens if t not in TICKER_STOPWORDS and not re.fullmatch(r"Q[1-4]", t)]
    if known_tickers:
        known = [t for t in candidates if t in known_tickers]
        if known:
            return known[0]
    if not candidates:
        return ""
    if source:
        # A recognized broker following a leading all-caps token is a strong,
        # conservative filename convention even when the ticker universe is stale.
        first_token = re.match(r"^\s*([A-Z]{1,5})(?=\s|[_\-.])", stem)
        if first_token and first_token.group(1) in candidates:
            return first_token.group(1)
    if known_tickers:
        return ""
    first = candidates[0]
    return first if stem.upper().count(first) >= 2 else ""


def company_or_identifier(file_name: str, ticker: str, source: str, document_type: str) -> str:
    stem = Path(file_name).stem
    cleaned = re.sub(r"[_\-.]+", " ", stem)
    for token in (ticker, source, document_type):
        if token:
            cleaned = re.sub(re.escape(token), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:20\d{2}|19\d{2}|Q[1-4]|FY\d{2,4})\b", " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()[:160]


def build_inventory(records: List[Dict[str, Any]], known_tickers: Optional[set[str]] = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in records:
        rel = str(record.get("relative_path") or "")
        name = str(record.get("file_name") or Path(rel).name)
        category = detect_category(rel)
        source = detect_source(f"{rel} {name}")
        doc_type = detect_document_type(name, category=category, source=source)
        ticker = detect_ticker(name, known_tickers=known_tickers, source=source)
        rows.append(
            {
                "category": category,
                "ticker": ticker,
                "company_or_identifier": company_or_identifier(name, ticker, source, doc_type),
                "source_or_broker": source,
                "document_type": doc_type,
                "file_name": name,
                "file_extension": Path(name).suffix.lower(),
                "file_size_mb": round(float(record.get("file_size") or 0) / (1024 * 1024), 3),
                "relative_path": rel,
                "modified_date": record.get("modified_date") or "",
                "extraction_status": record.get("extraction_status") or "",
                "extracted_path": record.get("extracted_path") or "",
            }
        )
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


def score_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame(columns=RELEVANCE_COLUMNS)
    rows: List[Dict[str, Any]] = []
    ticker_counts = Counter(str(x) for x in inventory.get("ticker", []) if str(x).strip())

    for _, row in inventory.iterrows():
        category = str(row.get("category") or "Root")
        ticker = str(row.get("ticker") or "")
        source = str(row.get("source_or_broker") or "")
        doc_type = str(row.get("document_type") or "other")
        name = str(row.get("file_name") or "")
        status = str(row.get("extraction_status") or "")
        searchable = re.sub(r"[_\-.]+", " ", f"{name} {row.get('relative_path') or ''}").lower()
        reasons: List[str] = []
        score = 0

        category_points = {
            "Companies": 18, "Banks": 16, "Credit": 16, "Green Street": 18,
            "Research": 12, "Root": 5,
        }.get(category, 5)
        score += category_points
        reasons.append(f"{category} folder +{category_points}")

        type_points = DOCUMENT_TYPE_SCORES.get(doc_type, 4)
        score += type_points
        reasons.append(f"{doc_type} +{type_points}")

        if source:
            score += 10
            reasons.append(f"recognized source {source} +10")
        if ticker:
            score += 14
            reasons.append(f"recognized ticker {ticker} +14")
            if ticker_counts[ticker] > 1:
                boost = min(10, (ticker_counts[ticker] - 1) * 2)
                score += boost
                reasons.append(f"ticker repeated across files +{boost}")

        keyword_hits = sorted({kw for kw in FINANCE_KEYWORDS if kw in searchable})
        if keyword_hits:
            boost = min(15, len(keyword_hits) * 3)
            score += boost
            reasons.append(f"finance keywords ({', '.join(keyword_hits[:4])}) +{boost}")

        if status != "extracted":
            score = max(0, score - 40)
            reasons.append("not extracted -40")
        if str(row.get("file_extension") or "") not in SUPPORTED_TEXT_EXTENSIONS:
            score = max(0, score - 8)
            reasons.append("unsupported text type -8")

        score = min(100, int(score))
        priority = "High" if score >= 60 else "Medium" if score >= 35 else "Low"
        enriched = row.to_dict()
        enriched.update(
            {
                "relevance_score": score,
                "priority_level": priority,
                "reason_for_score": "; ".join(reasons),
            }
        )
        rows.append(enriched)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["relevance_score", "ticker", "file_name"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    return result


def select_files_for_text(relevance: pd.DataFrame, max_files: int) -> pd.DataFrame:
    if relevance.empty:
        return relevance.copy()
    eligible = relevance[
        (relevance["extraction_status"] == "extracted")
        & (relevance["file_extension"].isin(SUPPORTED_TEXT_EXTENSIONS))
        & (relevance["priority_level"] == "High")
    ]
    return eligible.head(int(max_files)).copy()


def extract_selected_text(
    selected: pd.DataFrame,
    *,
    max_pdf_pages: int,
    max_chars_per_file: int,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for _, row in selected.iterrows():
        path = Path(str(row.get("extracted_path") or ""))
        text = ""
        status = "not_scanned"
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                text = _extract_pdf(path, max_pdf_pages, max_chars_per_file)
            elif suffix == ".xlsx":
                text = _extract_xlsx_preview(path, max_chars_per_file)
            elif suffix == ".xls":
                text = "Legacy XLS file inventoried; lightweight preview is not available."
            elif suffix in {".txt", ".md", ".csv"}:
                text = path.read_text(encoding="utf-8", errors="replace")[:max_chars_per_file]
            status = "scanned" if text.strip() else "scanned_no_text"
        except Exception as exc:
            status = f"scan_failed: {type(exc).__name__}"
            text = ""
        output.append(
            {
                "relative_path": row.get("relative_path") or "",
                "file_name": row.get("file_name") or "",
                "ticker": row.get("ticker") or "",
                "category": row.get("category") or "",
                "source_or_broker": row.get("source_or_broker") or "",
                "document_type": row.get("document_type") or "",
                "relevance_score": int(row.get("relevance_score") or 0),
                "text_extraction_status": status,
                "extracted_text": text[:max_chars_per_file],
            }
        )
    return output


def _extract_pdf(path: Path, max_pages: int, max_chars: int) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages[:max_pages]:
        parts.append(page.extract_text() or "")
        if sum(len(p) for p in parts) >= max_chars:
            break
    return "\n".join(parts)[:max_chars]


def _extract_xlsx_preview(path: Path, max_chars: int) -> str:
    try:
        from openpyxl import load_workbook
    except Exception:
        return "XLSX file inventoried; openpyxl is unavailable for preview."
    workbook = load_workbook(path, read_only=True, data_only=True)
    lines = ["Sheets: " + ", ".join(workbook.sheetnames)]
    for sheet in workbook.worksheets[:3]:
        lines.append(f"\n[{sheet.title}]")
        for row in sheet.iter_rows(min_row=1, max_row=8, values_only=True):
            values = [str(v) for v in row[:8] if v is not None]
            if values:
                lines.append(" | ".join(values))
            if len("\n".join(lines)) >= max_chars:
                break
    workbook.close()
    return "\n".join(lines)[:max_chars]


def build_ticker_summary(relevance: pd.DataFrame) -> pd.DataFrame:
    if relevance.empty:
        return pd.DataFrame(columns=["ticker", "file_count", "high_priority_files", "max_relevance_score", "categories", "sources"])
    known = relevance[relevance["ticker"].astype(str).str.strip() != ""]
    rows = []
    for ticker, group in known.groupby("ticker"):
        rows.append(
            {
                "ticker": ticker,
                "file_count": len(group),
                "high_priority_files": int((group["priority_level"] == "High").sum()),
                "max_relevance_score": int(group["relevance_score"].max()),
                "categories": ", ".join(sorted({str(x) for x in group["category"] if str(x)})),
                "sources": ", ".join(sorted({str(x) for x in group["source_or_broker"] if str(x)})),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_relevance_score", "file_count"], ascending=False) if rows else pd.DataFrame()


def build_category_summary(relevance: pd.DataFrame) -> pd.DataFrame:
    if relevance.empty:
        return pd.DataFrame(columns=["category", "file_count", "high_priority_files", "average_relevance_score", "top_document_types"])
    rows = []
    for category, group in relevance.groupby("category"):
        top_types = group["document_type"].value_counts().head(3).index.tolist()
        rows.append(
            {
                "category": category,
                "file_count": len(group),
                "high_priority_files": int((group["priority_level"] == "High").sum()),
                "average_relevance_score": round(float(group["relevance_score"].mean()), 1),
                "top_document_types": ", ".join(top_types),
            }
        )
    return pd.DataFrame(rows).sort_values(["high_priority_files", "file_count"], ascending=False)


def build_deterministic_brief(
    relevance: pd.DataFrame,
    selected_text: List[Dict[str, Any]],
    *,
    source_name: str,
) -> str:
    ticker_summary = build_ticker_summary(relevance)
    category_summary = build_category_summary(relevance)
    selected_by_path = {str(x.get("relative_path") or ""): x for x in selected_text}
    high = relevance[relevance["priority_level"] == "High"].head(15) if not relevance.empty else relevance
    skipped = relevance[~relevance["relative_path"].isin(selected_by_path.keys())] if not relevance.empty else relevance
    lines = [
        "# Daily Research Brief",
        "",
        f"Source archive: `{source_name}`",
        f"Files inventoried: {len(relevance)}",
        f"Files lightly scanned: {len(selected_text)}",
        f"High-priority files: {len(relevance[relevance['priority_level'] == 'High']) if not relevance.empty else 0}",
        "",
        "## Top Tickers / Entities",
    ]
    if ticker_summary.empty:
        lines.append("No ticker could be identified conservatively from the filenames.")
    else:
        for _, row in ticker_summary.head(12).iterrows():
            lines.append(
                f"- **{row['ticker']}**: {row['file_count']} file(s), "
                f"{row['high_priority_files']} high priority; categories: {row['categories'] or 'Unspecified'}."
            )

    lines.extend(["", "## Top High-Priority Documents"])
    if high.empty:
        lines.append("No files reached the deterministic high-priority threshold.")
    else:
        for _, row in high.iterrows():
            lines.append(
                f"- Score {int(row['relevance_score'])}: `{row['relative_path']}` "
                f"({row['document_type']}; {row['source_or_broker'] or 'source unknown'}; "
                f"{row['ticker'] or 'ticker uncertain'})."
            )

    lines.extend(["", "## Category Summary"])
    for _, row in category_summary.iterrows():
        lines.append(
            f"- **{row['category']}**: {row['file_count']} file(s), "
            f"{row['high_priority_files']} high priority; common types: {row['top_document_types'] or 'other'}."
        )

    lines.extend(["", "## Possible Signals From Metadata and Limited Text"])
    if not selected_text:
        lines.append("Not enough extracted text to verify details beyond filename/folder metadata.")
    else:
        for item in selected_text[:12]:
            text = re.sub(r"\s+", " ", str(item.get("extracted_text") or "")).strip()
            if len(text) < 80:
                signal = "Not enough extracted text to verify details beyond filename/folder metadata."
            else:
                signal = text[:300] + ("..." if len(text) > 300 else "")
            lines.append(f"- `{item['relative_path']}`: {signal}")

    lines.extend(
        [
            "",
            "## Recommended Follow-Up for Geoff / Mitko",
            "- Review the highest-scoring documents first, especially repeated tickers across multiple sources.",
            "- Open source files before acting on any possible signal; this brief uses only filenames, metadata, and limited first-page text.",
            "- Validate any ratings, estimates, financial figures, or conclusions directly in the source documents.",
            "",
            "## Skipped or Lightly Scanned Files",
        ]
    )
    if skipped.empty:
        lines.append("All inventoried supported files were included in the lightweight scan.")
    else:
        for _, row in skipped.head(30).iterrows():
            lines.append(f"- `{row['relative_path']}`: inventory/scoring only.")
        if len(skipped) > 30:
            lines.append(f"- ...and {len(skipped) - 30} additional inventory-only file(s).")

    lines.extend(["", "## Source References"])
    for item in selected_text:
        lines.append(f"- `{item['relative_path']}` ({item['text_extraction_status']})")
    return "\n".join(lines)


def build_llm_evidence_payload(
    relevance: pd.DataFrame,
    selected_text: List[Dict[str, Any]],
    *,
    max_total_chars: int = 22000,
) -> str:
    records = []
    used = 0
    for item in selected_text:
        snippet = re.sub(r"\s+", " ", str(item.get("extracted_text") or "")).strip()[:1200]
        record = {
            "relative_path": item.get("relative_path"),
            "ticker": item.get("ticker"),
            "category": item.get("category"),
            "source_or_broker": item.get("source_or_broker"),
            "document_type": item.get("document_type"),
            "relevance_score": item.get("relevance_score"),
            "limited_text": snippet or "Not enough extracted text to verify details beyond filename/folder metadata.",
        }
        encoded = json.dumps(record, ensure_ascii=False)
        if used + len(encoded) > max_total_chars:
            break
        records.append(record)
        used += len(encoded)
    metadata = relevance[
        ["relative_path", "ticker", "category", "source_or_broker", "document_type", "relevance_score", "priority_level"]
    ].head(30).to_dict(orient="records") if not relevance.empty else []
    return json.dumps({"selected_sources": records, "top_metadata": metadata}, ensure_ascii=False)

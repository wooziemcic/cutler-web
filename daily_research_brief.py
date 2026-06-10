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
    ("Morgan Stanley", r"\bMORGAN\s+STANLEY\b"),
    ("Gimme Credit", r"\bGIMME\s+CREDIT\b"),
    ("Green Street", r"\bGREEN\s+STREET\b"),
    ("Morningstar", r"\bMORNINGSTAR\b"),
    ("JPMorgan", r"\b(?:JPM|J\.?\s*P\.?\s*MORGAN|JP\s*MORGAN|JPMORGAN)\b"),
    ("Goldman Sachs", r"\b(?:GS|GOLDMAN(?:\s+SACHS)?)\b"),
    ("Wells Fargo", r"\b(?:WF|WELLS\s+FARGO)\b"),
    ("Bank of America", r"\b(?:BOFA|BOF\s+A|BANK\s+OF\s+AMERICA|BAML)\b"),
    ("Barclays", r"\bBARCLAYS\b"),
    ("RBC", r"\bRBC\b"),
    ("Jefferies", r"\bJEFFERIES\b"),
    ("Hovde", r"\bHOVDE\b"),
    ("Brean", r"\bBREAN\b"),
    ("Evercore", r"\bEVERCORE\b"),
    ("Mizuho", r"\bMIZUHO\b"),
    ("Guggenheim", r"\bGUGGENHEIM\b"),
    ("Stifel", r"\bSTIFEL\b"),
    ("Deutsche", r"\bDEUTSCHE\b"),
    ("Cantor", r"\bCANTOR\b"),
    ("Citizens", r"\bCITIZENS\b"),
    ("BMO", r"\bBMO\b"),
    ("UBS", r"\bUBS\b"),
    ("Citi", r"\b(?:CITI|CITIGROUP)\b"),
    ("Truist", r"\bTRUIST\b"),
    ("Piper", r"\bPIPER\b"),
    ("Wedbush", r"\bWEDBUSH\b"),
    ("Needham", r"\bNEEDHAM\b"),
    ("Oppenheimer", r"\bOPPENHEIMER\b"),
    ("Raymond James", r"\bRAYMOND\s+JAMES\b"),
    ("TD Cowen", r"\b(?:TD\s+COWEN|COWEN)\b"),
    ("KeyBanc", r"\bKEY\s*BANC\b"),
    ("Wolfe", r"\bWOLFE\b"),
    ("Baird", r"\bBAIRD\b"),
    ("Stephens", r"\bSTEPHENS\b"),
]

TOP_TIER_BROKERS = {
    "JPMorgan", "Goldman Sachs", "Wells Fargo", "Bank of America", "Morgan Stanley",
    "Barclays", "Citi", "UBS", "Deutsche", "RBC",
}
SPECIALTY_SOURCES = {
    "Gimme Credit", "Green Street", "Morningstar", "Hovde", "Brean", "Needham",
    "Oppenheimer", "Raymond James", "TD Cowen", "KeyBanc", "Wolfe", "Baird", "Stephens",
}

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
    ("rating change", (r"\brating\s+(?:change|upgrade|downgrade)\b",)),
    ("recommendation change", (r"\b(?:recommendation|upgrade|downgrade|initiated)\b",)),
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
    "HOVDE", "BREAN", "WOLFE", "BAIRD", "PIPER",
    "BANKS", "COMPANIES", "CREDIT", "RESEARCH", "ROOT", "GREEN", "STREET",
}

DOCUMENT_TYPE_SCORES = {
    "earnings release": 28,
    "earnings presentation": 27,
    "earnings supplement": 24,
    "transcript": 28,
    "10-Q/10Q": 28,
    "8-K/8K": 25,
    "current report": 18,
    "credit report": 30,
    "analyst report": 27,
    "sector report": 26,
    "Green Street report": 28,
    "industry report": 18,
    "prepared remarks": 22,
    "guidance": 28,
    "estimate revision": 27,
    "recommendation change": 29,
    "rating change": 28,
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
    "category", "ticker", "all_detected_tickers", "company_or_identifier", "source_or_broker", "document_type",
    "file_name", "file_extension", "file_size_mb", "relative_path", "modified_date",
    "extraction_status", "extracted_path",
]
RELEVANCE_COLUMNS = INVENTORY_COLUMNS + [
    "relevance_score", "priority_level", "reason_for_score", "score_breakdown",
    "investment_rationale",
]


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
    if category == "Credit" and (source or re.search(r"\bcredit\b", normalized, flags=re.IGNORECASE)):
        return "credit report"
    if category == "Green Street" or source == "Green Street":
        return "Green Street report" if source == "Green Street" else "sector report"
    if source and category in {"Research", "Banks", "Companies", "Root"}:
        return "analyst report"
    return "other"


def detect_ticker(
    file_name: str,
    known_tickers: Optional[set[str]] = None,
    *,
    source: str = "",
    category: str = "Root",
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
    if source or category in {"Banks", "Companies", "Credit", "Research"}:
        # A recognized broker following a leading all-caps token is a strong,
        # conservative research-filename convention even when the ticker universe is stale.
        first_token = re.match(r"^\s*([A-Z]{1,5})(?=\s|[_\-.])", stem)
        if first_token and first_token.group(1) in candidates:
            return first_token.group(1)
    if known_tickers:
        return ""
    first = candidates[0]
    return first if stem.upper().count(first) >= 2 else ""


def detect_all_tickers(
    file_name: str,
    known_tickers: Optional[set[str]] = None,
    *,
    source: str = "",
    category: str = "Root",
) -> List[str]:
    stem = Path(file_name).stem
    tokens = re.findall(r"(?<![A-Za-z0-9])[A-Z]{1,5}(?![A-Za-z0-9])", stem)
    candidates = [t for t in tokens if t not in TICKER_STOPWORDS and not re.fullmatch(r"Q[1-4]", t)]
    detected: List[str] = []
    for token in candidates:
        if token in detected:
            continue
        if known_tickers and token in known_tickers:
            detected.append(token)
        elif source and token == candidates[0]:
            detected.append(token)
        elif category == "Credit" and len(candidates) > 1:
            detected.append(token)
    return detected


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
        ticker = detect_ticker(name, known_tickers=known_tickers, source=source, category=category)
        all_tickers = detect_all_tickers(
            name,
            known_tickers=known_tickers,
            source=source,
            category=category,
        )
        if ticker and ticker not in all_tickers:
            all_tickers.insert(0, ticker)
        rows.append(
            {
                "category": category,
                "ticker": ticker,
                "all_detected_tickers": ", ".join(all_tickers),
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
    ticker_categories: Dict[str, set[str]] = {}
    ticker_sources: Dict[str, set[str]] = {}
    for _, source_row in inventory.iterrows():
        tickers_in_row = [
            t.strip() for t in str(source_row.get("all_detected_tickers") or source_row.get("ticker") or "").split(",")
            if t.strip()
        ]
        for detected in tickers_in_row:
            ticker_categories.setdefault(detected, set()).add(str(source_row.get("category") or "Root"))
            source = str(source_row.get("source_or_broker") or "")
            if source:
                ticker_sources.setdefault(detected, set()).add(source)

    for _, row in inventory.iterrows():
        category = str(row.get("category") or "Root")
        ticker = str(row.get("ticker") or "")
        source = str(row.get("source_or_broker") or "")
        doc_type = str(row.get("document_type") or "other")
        name = str(row.get("file_name") or "")
        status = str(row.get("extraction_status") or "")
        searchable = re.sub(r"[_\-.]+", " ", f"{name} {row.get('relative_path') or ''}").lower()
        breakdown: List[str] = []
        score = 0

        category_points = {
            "Credit": 20, "Companies": 18, "Banks": 18, "Research": 16,
            "Green Street": 22, "Root": 8,
        }.get(category, 8)
        score += category_points
        breakdown.append(f"{category} folder +{category_points}")

        type_points = DOCUMENT_TYPE_SCORES.get(doc_type, 4)
        score += type_points
        breakdown.append(f"{doc_type} +{type_points}")

        if source:
            score += 8
            breakdown.append(f"recognized source {source} +8")
            if source in TOP_TIER_BROKERS:
                score += 5
                breakdown.append("top-tier broker +5")
            elif source in SPECIALTY_SOURCES:
                score += 5
                breakdown.append("specialty source +5")
        if ticker:
            score += 12
            breakdown.append(f"recognized ticker {ticker} +12")
            if ticker_counts[ticker] > 1:
                boost = min(10, (ticker_counts[ticker] - 1) * 2)
                score += boost
                breakdown.append(f"repeated ticker coverage +{boost}")
            if len(ticker_categories.get(ticker, set())) > 1:
                score += 6
                breakdown.append("cross-category ticker coverage +6")
            if len(ticker_sources.get(ticker, set())) > 1:
                boost = min(10, len(ticker_sources[ticker]) * 2)
                score += boost
                breakdown.append(f"multiple broker/source coverage +{boost}")

        keyword_hits = sorted({kw for kw in FINANCE_KEYWORDS if kw in searchable})
        if keyword_hits:
            boost = min(12, len(keyword_hits) * 3)
            score += boost
            breakdown.append(f"finance keywords ({', '.join(keyword_hits[:4])}) +{boost}")

        if status != "extracted":
            score = max(0, score - 40)
            breakdown.append("not extracted -40")
        if str(row.get("file_extension") or "") not in SUPPORTED_TEXT_EXTENSIONS:
            score = max(0, score - 8)
            breakdown.append("unsupported text type -8")

        score = min(100, int(score))
        priority = "High" if score >= 70 else "Medium" if score >= 45 else "Low"
        rationale_parts = [f"{doc_type} in {category}"]
        if ticker:
            rationale_parts.append(f"covers {ticker}")
        if source:
            rationale_parts.append(f"from {source}")
        if ticker and len(ticker_sources.get(ticker, set())) > 1:
            rationale_parts.append("with multiple-source coverage")
        if ticker and len(ticker_categories.get(ticker, set())) > 1:
            rationale_parts.append("appearing across categories")
        rationale = "; ".join(rationale_parts) + "."
        enriched = row.to_dict()
        enriched.update(
            {
                "relevance_score": score,
                "priority_level": priority,
                "reason_for_score": rationale,
                "score_breakdown": "; ".join(breakdown),
                "investment_rationale": rationale,
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
        return pd.DataFrame()
    known = _expand_rows_by_detected_ticker(relevance)
    rows = []
    for ticker, group in known.groupby("ticker"):
        sources = sorted({str(x) for x in group["source_or_broker"] if str(x)})
        categories = sorted({str(x) for x in group["category"] if str(x)})
        doc_types = group["document_type"].astype(str)
        broker_reports = group[group["source_or_broker"].astype(str).str.strip() != ""]
        attention = _attention_reason(group, ticker=ticker)
        rows.append(
            {
                "ticker": ticker,
                "file_count": len(group),
                "high_priority_files": int((group["priority_level"] == "High").sum()),
                "max_relevance_score": int(group["relevance_score"].max()),
                "broker_report_count": len(broker_reports),
                "credit_report_count": int((doc_types == "credit report").sum()),
                "earnings_file_count": int(doc_types.str.startswith("earnings").sum()),
                "filing_file_count": int(doc_types.isin(["10-Q/10Q", "8-K/8K", "current report", "MD&A"]).sum()),
                "transcript_count": int((doc_types == "transcript").sum()),
                "has_multiple_brokers": len(sources) > 1,
                "categories": ", ".join(categories),
                "sources": ", ".join(sources),
                "attention_reason": attention,
            }
        )
    return pd.DataFrame(rows).sort_values(["max_relevance_score", "file_count"], ascending=False) if rows else pd.DataFrame()


def build_broker_coverage_summary(relevance: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker", "broker_report_count", "brokers_or_sources", "analyst_report_files",
        "credit_report_files", "categories_present", "attention_reason",
    ]
    if relevance.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    known = _expand_rows_by_detected_ticker(relevance)
    for ticker, group in known.groupby("ticker"):
        broker_group = group[group["source_or_broker"].astype(str).str.strip() != ""]
        if broker_group.empty:
            continue
        report_group = broker_group
        rows.append(
            {
                "ticker": ticker,
                "broker_report_count": len(report_group),
                "brokers_or_sources": ", ".join(sorted(set(report_group["source_or_broker"].astype(str)))),
                "analyst_report_files": int((report_group["document_type"] == "analyst report").sum()),
                "credit_report_files": int((report_group["document_type"] == "credit report").sum()),
                "categories_present": ", ".join(sorted(set(report_group["category"].astype(str)))),
                "attention_reason": _attention_reason(group, ticker=ticker),
            }
        )
    return (
        pd.DataFrame(rows, columns=columns).sort_values(
            ["broker_report_count", "ticker"], ascending=[False, True]
        )
        if rows else pd.DataFrame(columns=columns)
    )


def _attention_reason(group: pd.DataFrame, *, ticker: str) -> str:
    reasons = []
    sources = {str(x) for x in group["source_or_broker"] if str(x)}
    categories = {str(x) for x in group["category"] if str(x)}
    doc_types = group["document_type"].astype(str)
    if len(sources) > 1:
        reasons.append(f"{len(sources)} brokers/sources")
    if len(categories) > 1:
        reasons.append(f"coverage across {len(categories)} categories")
    if (doc_types == "credit report").any():
        reasons.append("credit research present")
    if doc_types.str.startswith("earnings").any() or (doc_types == "transcript").any():
        reasons.append("earnings-related material present")
    if doc_types.isin(["recommendation change", "rating change", "guidance", "estimate revision"]).any():
        reasons.append("change-oriented document present")
    if not reasons:
        reasons.append(f"{len(group)} relevant file(s) for {ticker}")
    return "; ".join(reasons)


def _expand_rows_by_detected_ticker(relevance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in relevance.iterrows():
        tickers_in_row = [
            t.strip() for t in str(row.get("all_detected_tickers") or row.get("ticker") or "").split(",")
            if t.strip()
        ]
        for detected in tickers_in_row:
            expanded = row.to_dict()
            expanded["ticker"] = detected
            rows.append(expanded)
    return pd.DataFrame(rows, columns=relevance.columns) if rows else relevance.iloc[0:0].copy()


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
    broker_summary = build_broker_coverage_summary(relevance)
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
                f"- **{row['ticker']}**: {row['attention_reason']}; "
                f"{row['file_count']} file(s), {row['high_priority_files']} high priority, "
                f"max score {row['max_relevance_score']}."
            )

    lines.extend(["", "## Top High-Priority Documents"])
    if high.empty:
        lines.append("No files reached the deterministic high-priority threshold.")
    else:
        for _, row in high.iterrows():
            lines.append(
                f"- Score {int(row['relevance_score'])}: `{row['relative_path']}` "
                f"({row['document_type']}; {row['source_or_broker'] or 'source unknown'}; "
                f"{row['ticker'] or 'ticker uncertain'}). Rationale: {row['investment_rationale']}"
            )

    lines.extend(["", "## Broker Coverage Highlights"])
    if broker_summary.empty:
        lines.append("No normalized broker/source coverage was identified.")
    else:
        for _, row in broker_summary.head(12).iterrows():
            lines.append(
                f"- **{row['ticker']}**: {row['broker_report_count']} broker/source report(s) from "
                f"{row['brokers_or_sources']}; {row['attention_reason']}."
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
            "- Prioritize tickers with multiple brokers, cross-category coverage, or both credit and equity research.",
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
    reference_paths = set()
    for _, row in high.iterrows():
        path = str(row.get("relative_path") or "")
        if path and path not in reference_paths:
            lines.append(f"- `{path}` (high-priority metadata source)")
            reference_paths.add(path)
    for item in selected_text:
        path = str(item.get("relative_path") or "")
        if path and path not in reference_paths:
            lines.append(f"- `{path}` ({item['text_extraction_status']})")
            reference_paths.add(path)
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
        [
            "relative_path", "ticker", "all_detected_tickers", "category", "source_or_broker",
            "document_type", "relevance_score", "priority_level", "investment_rationale",
        ]
    ].head(30).to_dict(orient="records") if not relevance.empty else []
    broker_coverage = build_broker_coverage_summary(relevance).head(20).to_dict(orient="records")
    ticker_summary = build_ticker_summary(relevance).head(20).to_dict(orient="records")
    return json.dumps(
        {
            "selected_sources": records,
            "top_metadata": metadata,
            "broker_coverage": broker_coverage,
            "ticker_summary": ticker_summary,
        },
        ensure_ascii=False,
    )

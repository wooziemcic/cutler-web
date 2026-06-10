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

INSUFFICIENT_TEXT_NOTICE = (
    "No verified financial claims were generated because extracted text was insufficient. "
    "The observations below are based on filenames, folders, document types, and "
    "broker/source coverage only."
)
SENSITIVE_FINANCE_PATTERNS = [
    r"\bbeat(?:s|en|ing)?\b", r"\bmiss(?:es|ed|ing)?\b", r"\beps\b", r"\brevenue\b",
    r"\bprice target\b", r"\brating (?:was |has been )?(?:raised|lowered|upgraded|downgraded)\b",
    r"\brating (?:changed|moved|cut|increased|decreased)\b",
    r"\bguidance (?:was |has been )?(?:raised|lowered|increased|decreased|cut)\b",
    r"\bunderwriting performance\b", r"\bcredit (?:deterioration|improvement)\b",
    r"\bcredit (?:improved|deteriorated|weakened|strengthened)\b",
    r"\bmargins?\b", r"\bcapex\b", r"\bliquidity\b", r"\bvaluation\b",
    r"\b(?:market weight|overweight|underweight)\b",
    r"\boperational (?:strength|weakness|performance)\b", r"\bperformance was (?:robust|strong|weak)\b",
    r"\bstrong earnings\b", r"\bweak earnings\b",
]

BROKER_COMPARISON_CATEGORIES = {"Research", "Banks", "Credit", "Companies", "Green Street"}
BROKER_COMPARISON_DOCUMENT_TYPES = {
    "analyst report", "credit report", "Green Street report", "sector report",
    "industry report", "recommendation change", "rating change", "guidance",
    "estimate revision",
}
COMPARISON_REVIEW_TERMS = {
    "guidance": r"\bguidance\b",
    "rating": r"\brating\b",
    "estimate": r"\bestimates?\b",
    "margin": r"\bmargins?\b",
    "capex": r"\bcapex\b|\bcapital expenditures?\b",
    "liquidity": r"\bliquidity\b",
    "valuation": r"\bvaluation\b",
    "credit": r"\bcredit\b",
}
DISCLOSURE_PHRASES = [
    "intended for institutional investors", "independence and disclosure standards",
    "finra", "conflicts of interest", "seeks to do business", "analyst certification",
    "important disclosures", "investors should consider this report", "all rights reserved",
    "use this report only", "not subject to", "discretionary basis",
    "covered in its research reports",
]
INVESTMENT_EVIDENCE_PATTERNS = {
    "Rating / PT": (
        r"\brating\b", r"\bprice target\b", r"\boverweight\b", r"\bunderweight\b",
        r"\bmarket weight\b", r"\bneutral\b", r"\bbuy\b", r"\bsell\b", r"\boutperform\b",
    ),
    "Earnings / Results": (
        r"\beps\b", r"\brevenue\b", r"\bsales\b", r"\bmargin\b", r"\boperating income\b",
    ),
    "Guidance / Estimates": (r"\bguidance\b", r"\bestimates?\b", r"\brevision\b"),
    "Credit / Liquidity": (r"\bliquidity\b", r"\bleverage\b", r"\bdebt\b", r"\bcredit\b"),
    "Valuation / Thesis": (r"\bvaluation\b", r"\bthesis\b", r"\bupside\b", r"\bdownside\b"),
    "Risk / Outlook": (r"\brisk\b", r"\boutlook\b"),
    "Other": (r"\bcapex\b", r"\baws\b", r"\bcloud\b", r"\bretail\b", r"\badvertising\b"),
}


def create_session_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="cutler_daily_research_"))


def parse_daily_folder_date_from_name(source_name: str) -> Optional[datetime]:
    """Parse a daily research date from an archive or source-folder name."""
    name = PurePosixPath(str(source_name or "").replace("\\", "/")).name.strip()
    name = re.sub(r"\.(?:zip|pdf|xlsx?|csv|txt|md)$", "", name, flags=re.IGNORECASE)
    for match in re.finditer(r"(?<!\d)(\d{1,2})[._-](\d{1,2})[._-](\d{2,4})(?!\d)", name):
        month, day, year = (int(value) for value in match.groups())
        if year < 100:
            year += 2000
        try:
            return datetime(year, month, day)
        except ValueError:
            continue
    return None


def daily_source_title_suffix(source_name: str, relative_paths: Optional[Iterable[str]] = None) -> str:
    candidates = [str(source_name or "")]
    for relative_path in relative_paths if relative_paths is not None else []:
        parts = PurePosixPath(str(relative_path).replace("\\", "/")).parts
        if parts:
            candidates.append(parts[0])
    for candidate in candidates:
        parsed = parse_daily_folder_date_from_name(candidate)
        if parsed:
            return parsed.strftime("%B %d, %Y").replace(" 0", " ")
    stem = Path(str(source_name or "daily_research")).stem.strip() or "daily_research"
    return f"Source Folder: {stem}"


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


def select_broker_comparison_files(
    relevance: pd.DataFrame,
    ticker: str,
    *,
    max_files: int,
) -> pd.DataFrame:
    """Select same-day broker/source reports for one ticker, ordered by relevance."""
    if relevance.empty or not ticker:
        return relevance.iloc[0:0].copy()
    expanded = _expand_rows_by_detected_ticker(relevance)
    ticker_rows = expanded[expanded["ticker"].astype(str).str.upper() == ticker.upper()].copy()
    if ticker_rows.empty:
        return ticker_rows
    broker_rows = ticker_rows[
        ticker_rows["source_or_broker"].astype(str).str.strip().ne("")
        & ticker_rows["category"].astype(str).isin(BROKER_COMPARISON_CATEGORIES)
    ].copy()
    preferred = broker_rows[
        broker_rows["document_type"].astype(str).isin(BROKER_COMPARISON_DOCUMENT_TYPES)
    ].copy()
    selected = preferred if not preferred.empty else broker_rows[
        ~broker_rows["document_type"].astype(str).isin(
            ["10-Q/10Q", "8-K/8K", "current report", "earnings release",
             "earnings presentation", "earnings supplement", "transcript", "prepared remarks"]
        )
    ].copy()
    if selected.empty:
        return selected
    selected = selected.drop_duplicates(subset=["relative_path"]).sort_values(
        ["relevance_score", "source_or_broker", "file_name"],
        ascending=[False, True, True],
    )
    return selected.head(int(max_files)).reset_index(drop=True)


def prepare_broker_comparison_text(
    selected: pd.DataFrame,
    existing_text: List[Dict[str, Any]],
    *,
    max_pdf_pages: int,
    max_chars_per_file: int,
    reuse_existing: bool,
) -> List[Dict[str, Any]]:
    """Reuse prior snippets when requested and scan only missing comparison files."""
    if selected.empty:
        return []
    existing_by_path = {
        str(item.get("relative_path") or ""): item
        for item in existing_text
        if str(item.get("relative_path") or "")
    }
    output: List[Dict[str, Any]] = []
    missing_rows = []
    for _, row in selected.iterrows():
        path = str(row.get("relative_path") or "")
        prior = existing_by_path.get(path)
        if reuse_existing and prior:
            reused = dict(prior)
            reused["extracted_text"] = str(reused.get("extracted_text") or "")[:max_chars_per_file]
            output.append(reused)
        else:
            missing_rows.append(row.to_dict())
    if missing_rows:
        missing = pd.DataFrame(missing_rows, columns=selected.columns)
        output.extend(
            extract_selected_text(
                missing,
                max_pdf_pages=max_pdf_pages,
                max_chars_per_file=max_chars_per_file,
            )
        )
    order = {str(row.get("relative_path") or ""): i for i, row in selected.iterrows()}
    return sorted(output, key=lambda item: order.get(str(item.get("relative_path") or ""), 10**9))


def _investment_evidence_type(text: str) -> str:
    scored = []
    for evidence_type, patterns in INVESTMENT_EVIDENCE_PATTERNS.items():
        hits = sum(bool(re.search(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)
        if hits:
            scored.append((hits, evidence_type))
    return max(scored)[1] if scored else "Other"


def best_investment_useful_snippet(item: Dict[str, Any], *, max_chars: int = 500) -> Dict[str, Any]:
    """Choose an investment-useful extracted passage while avoiding disclosure-heavy text."""
    status = str(item.get("text_extraction_status") or "metadata-only")
    text = _normalized_extracted_text(item)
    result = {
        "evidence_type": "Other",
        "snippet": "",
        "quality": "unavailable",
        "extraction_status": status,
    }
    if status != "scanned" or not text:
        return result

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\s{2,}", text)
        if sentence.strip()
    ]
    if not sentences:
        sentences = [text]
    candidates = []

    def add_candidate(passage: str) -> None:
        passage = passage.strip()[:max_chars]
        if not passage:
            return
        useful_hits = sum(
            bool(re.search(pattern, passage, flags=re.IGNORECASE))
            for patterns in INVESTMENT_EVIDENCE_PATTERNS.values()
            for pattern in patterns
        )
        disclosure_hits = sum(phrase in passage.lower() for phrase in DISCLOSURE_PHRASES)
        score = (useful_hits * 6) - (disclosure_hits * 15)
        if useful_hits:
            score += min(len(passage), max_chars) / max_chars
        candidates.append((score, useful_hits, disclosure_hits, passage))

    for index in range(len(sentences)):
        for window_size in (1, 2, 3):
            add_candidate(" ".join(sentences[index:index + window_size]))
    for start in range(0, len(text), max(200, max_chars // 2)):
        add_candidate(text[start:start + max_chars])

    best = max(candidates, default=None, key=lambda value: value[0])
    if best and best[1] > 0 and best[0] > 0:
        passage = best[3]
        return {
            "evidence_type": _investment_evidence_type(passage),
            "snippet": passage,
            "quality": "investment_useful",
            "extraction_status": status,
        }
    if any(phrase in text.lower() for phrase in DISCLOSURE_PHRASES):
        result["quality"] = "disclosure_only"
    else:
        result["quality"] = "no_useful_snippet"
    return result


def _markdown_table_text(value: Any) -> str:
    return str(value or "").replace("|", r"\|").replace("\n", " ").strip()


def build_deterministic_broker_comparison(
    ticker: str,
    files: pd.DataFrame,
    snippets: List[Dict[str, Any]],
    *,
    report_date: str,
    mode: str,
) -> str:
    """Build a conservative same-day comparison from metadata and direct excerpts."""
    snippets_by_path = {str(item.get("relative_path") or ""): item for item in snippets}
    verified = verified_extracted_text_items(snippets)
    sources = sorted({str(x) for x in files["source_or_broker"] if str(x)}) if not files.empty else []
    cited_paths = ", ".join(f"`{path}`" for path in files["relative_path"].astype(str)) if not files.empty else ""
    lines = [
        f"# Broker Consensus Report - {ticker} - {report_date}",
        "",
        f"Comparison mode: {mode}",
        "",
        "## Files Compared",
        "",
        "| Broker/source | File name | Document type | Extraction status | Relevance score |",
        "|---|---|---|---|---:|",
    ]
    for _, row in files.iterrows():
        path = str(row.get("relative_path") or "")
        item = snippets_by_path.get(path, {})
        status = str(item.get("text_extraction_status") or "metadata-only")
        lines.append(
            f"| {row.get('source_or_broker') or 'Unknown'} | `{row.get('file_name') or path}` | "
            f"{row.get('document_type') or 'other'} | {status} | {int(row.get('relevance_score') or 0)} |"
        )

    lines.extend(
        [
            "",
            "## Key Extracted Evidence",
            "",
            "| Broker/source | File name | Evidence type | Extracted evidence | Extraction status |",
            "|---|---|---|---|---|",
        ]
    )
    for _, row in files.iterrows():
        path = str(row.get("relative_path") or "")
        item = snippets_by_path.get(path, {})
        evidence = best_investment_useful_snippet(item)
        snippet = evidence["snippet"] or "No investment-useful snippet found in limited extraction."
        lines.append(
            f"| {_markdown_table_text(row.get('source_or_broker') or 'Unknown')} | "
            f"`{_markdown_table_text(row.get('file_name') or path)}` | "
            f"{evidence['evidence_type']} | \"{_markdown_table_text(snippet)}\" | "
            f"{_markdown_table_text(evidence['extraction_status'])} |"
        )

    lines.extend(["", "## Executive Takeaway"])
    if files.empty:
        lines.append("No qualifying same-day broker/source reports were found for this ticker.")
    elif not verified:
        lines.append(
            f"{len(files)} qualifying report(s) from {len(sources)} broker/source(s) were identified for "
            f"**{ticker}**. The comparison is based mostly on metadata because extracted text was insufficient. "
            f"No broker-view or financial conclusions were generated. Sources: {cited_paths}"
        )
    else:
        lines.append(
            f"{len(files)} qualifying report(s) from {len(sources)} broker/source(s) were compared for "
            f"**{ticker}**. Limited extracted snippets are quoted below; review the source PDFs before "
            f"drawing financial conclusions. Sources: {cited_paths}"
        )

    lines.extend(["", "## Broker-by-Broker Summary"])
    if files.empty:
        lines.append("No broker/source reports available.")
    for broker, group in files.groupby("source_or_broker", sort=True):
        lines.append(f"### {broker}")
        for _, row in group.iterrows():
            path = str(row.get("relative_path") or "")
            item = snippets_by_path.get(path, {})
            status = str(item.get("text_extraction_status") or "metadata-only")
            evidence = best_investment_useful_snippet(item)
            lines.append(
                f"- `{path}`: detected as {row.get('document_type') or 'other'} in "
                f"{row.get('category') or 'Unknown'}; extraction status: {status}."
            )
            if evidence["snippet"]:
                lines.append(
                    f"  Best investment-useful excerpt from `{path}`: "
                    f"\"{evidence['snippet']}\""
                )
            elif evidence["quality"] == "disclosure_only":
                lines.append(
                    f"  Limited extraction mostly captured disclosure/disclaimer text; review PDF directly. "
                    f"Source: `{path}`"
                )
            else:
                lines.append(
                    f"  No investment-useful snippet found in limited extraction. Source: `{path}`"
                )

    lines.extend(["", "## Consensus Themes"])
    if len(sources) >= 2:
        cited = ", ".join(f"`{path}`" for path in files["relative_path"].astype(str).head(12))
        lines.append(
            f"- Multiple brokers/sources covered **{ticker}** in the uploaded research set: "
            f"{', '.join(sources)}. Sources: {cited}"
        )
    doc_counts = files["document_type"].astype(str).value_counts() if not files.empty else pd.Series(dtype=int)
    for document_type, count in doc_counts.items():
        if count < 2:
            continue
        paths = files.loc[files["document_type"].astype(str) == document_type, "relative_path"].astype(str)
        lines.append(
            f"- {count} files were detected as **{document_type}**. Sources: "
            + ", ".join(f"`{path}`" for path in paths)
        )
    verified_term_sources: Dict[str, List[str]] = {}
    for term, pattern in COMPARISON_REVIEW_TERMS.items():
        matched = [
            str(item.get("relative_path") or "")
            for item in verified
            if re.search(
                pattern,
                str(best_investment_useful_snippet(item).get("snippet") or ""),
                flags=re.IGNORECASE,
            )
        ]
        if len(set(matched)) >= 2:
            verified_term_sources[term] = sorted(set(matched))
            lines.append(
                f"- The term **{term}** appears in limited extracted text from at least two sources: "
                + ", ".join(f"`{path}`" for path in sorted(set(matched)))
                + ". This records term overlap only, not agreement or direction."
            )
    if len(sources) < 2 and not any(count >= 2 for count in doc_counts) and not verified_term_sources:
        lines.append("No consensus theme could be verified beyond the available file metadata.")

    lines.extend(["", "## Divergences / Differences"])
    lines.append("Not enough extracted text to verify broker-level differences.")

    lines.extend(
        [
            "",
            "## Items to Verify Manually",
            "- Rating or price target changes in the source PDFs.",
            "- EPS or revenue beats/misses in the source PDFs.",
            "- Guidance changes and estimate revisions.",
            "- Credit concerns, liquidity commentary, and valuation assumptions.",
            "",
            "## Source References",
        ]
    )
    for _, row in files.iterrows():
        path = str(row.get("relative_path") or "")
        status = str(snippets_by_path.get(path, {}).get("text_extraction_status") or "metadata-only")
        lines.append(f"- `{path}` ({row.get('source_or_broker') or 'source unknown'}; {status})")
    return "\n".join(lines)


def build_broker_comparison_evidence_payload(
    ticker: str,
    files: pd.DataFrame,
    snippets: List[Dict[str, Any]],
    *,
    max_total_chars: int = 30000,
) -> str:
    snippets_by_path = {str(item.get("relative_path") or ""): item for item in snippets}
    records = []
    used = 0
    for _, row in files.iterrows():
        path = str(row.get("relative_path") or "")
        item = snippets_by_path.get(path, {})
        text = _normalized_extracted_text(item)
        status = str(item.get("text_extraction_status") or "metadata-only")
        verified = status == "scanned" and len(text) >= 160
        evidence = best_investment_useful_snippet(item, max_chars=900)
        record = {
            "relative_path": path,
            "file_name": row.get("file_name"),
            "ticker": ticker,
            "category": row.get("category"),
            "source_or_broker": row.get("source_or_broker"),
            "document_type": row.get("document_type"),
            "relevance_score": int(row.get("relevance_score") or 0),
            "investment_rationale": row.get("investment_rationale"),
            "text_extraction_status": status,
            "verified_text_available": verified,
            "evidence_type": evidence["evidence_type"],
            "evidence_quality": evidence["quality"],
            "limited_text": evidence["snippet"] if verified and evidence["snippet"] else INSUFFICIENT_TEXT_NOTICE,
        }
        encoded = json.dumps(record, ensure_ascii=False)
        if used + len(encoded) > max_total_chars:
            break
        records.append(record)
        used += len(encoded)
    return json.dumps(
        {
            "ticker": ticker,
            "grounding_rule": (
                "Metadata supports file/source/category/document-type observations only. Financial or broker-view "
                "claims require explicit verified limited text and a same-line source filename citation."
            ),
            "files": records,
        },
        ensure_ascii=False,
    )


def validate_broker_comparison_grounding(text: str, snippets: List[Dict[str, Any]]) -> bool:
    required = [
        "## Files Compared", "## Key Extracted Evidence", "## Executive Takeaway",
        "## Broker-by-Broker Summary", "## Consensus Themes", "## Divergences / Differences",
        "## Items to Verify Manually", "## Source References",
    ]
    if not all(section in text for section in required):
        return False
    key_section = text.split("## Key Extracted Evidence", 1)[1].split("## Executive Takeaway", 1)[0]
    verified = verified_extracted_text_items(snippets)
    for line in key_section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "broker/source" in stripped.lower() or bool(re.fullmatch(r"[\s|:-]+", stripped)):
            continue
        cited = [
            item for item in snippets
            if str(item.get("relative_path") or "") in line or str(item.get("file_name") or "") in line
        ]
        if not cited:
            return False
        quotes = [part.strip() for part in re.findall(r'"([^"]+)"', line) if part.strip()]
        for quote in quotes:
            if quote == "No investment-useful snippet found in limited extraction.":
                continue
            if not any(quote.lower() in _normalized_extracted_text(item).lower() for item in cited):
                return False
    before_evidence, after_evidence = text.split("## Key Extracted Evidence", 1)
    after_evidence = after_evidence.split("## Executive Takeaway", 1)[1]
    claim_text = before_evidence + "## Executive Takeaway" + after_evidence
    claim_text = claim_text.split("## Items to Verify Manually", 1)[0]
    if not validate_llm_brief_grounding(claim_text, snippets):
        return False
    source_names = {
        value
        for item in snippets
        for value in [str(item.get("relative_path") or ""), str(item.get("file_name") or "")]
        if value
    }
    for line in claim_text.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or bool(re.fullmatch(r"[\s|:-]+", stripped))
            or (stripped.startswith("|") and "broker" in stripped.lower() and "file" in stripped.lower())
            or stripped.startswith("Comparison mode:")
            or stripped == "Not enough extracted text to verify broker-level differences."
        ):
            continue
        if not any(source in stripped for source in source_names):
            return False
    divergence = text.split("## Divergences / Differences", 1)[1].split("## Items to Verify Manually", 1)[0]
    if "Not enough extracted text to verify broker-level differences." not in divergence:
        cited_sources = {
            str(item.get("relative_path") or "")
            for item in verified
            if str(item.get("relative_path") or "") in divergence
        }
        matched_quotes = {
            quote
            for quote in re.findall(r'"([^"]+)"', divergence)
            if len(quote.strip()) >= 20
            and any(quote.lower() in _normalized_extracted_text(item).lower() for item in verified)
        }
        if len(cited_sources) < 2 or len(matched_quotes) < 2:
            return False
    return True


def select_ticker_memo_files(
    relevance: pd.DataFrame,
    ticker: str,
    *,
    max_files: int,
) -> pd.DataFrame:
    """Select a capped, diversified set of same-day documents for one ticker."""
    if relevance.empty or not ticker:
        return relevance.iloc[0:0].copy()
    expanded = _expand_rows_by_detected_ticker(relevance)
    rows = expanded[
        (expanded["ticker"].astype(str).str.upper() == ticker.upper())
    ].drop_duplicates(subset=["relative_path"]).copy()
    if rows.empty:
        return rows

    rows["_memo_type_rank"] = rows["document_type"].astype(str).map(
        {
            "credit report": 0, "earnings release": 0, "earnings presentation": 0,
            "transcript": 0, "10-Q/10Q": 0, "8-K/8K": 0, "analyst report": 1,
            "earnings supplement": 1, "prepared remarks": 1, "guidance": 1,
            "estimate revision": 1, "rating change": 1, "recommendation change": 1,
        }
    ).fillna(2)
    rows = rows.sort_values(
        ["_memo_type_rank", "relevance_score", "source_or_broker", "file_name"],
        ascending=[True, False, True, True],
    )
    selected_indices = []
    for _, group in rows.groupby("document_type", sort=False):
        selected_indices.append(group.index[0])
        if len(selected_indices) >= int(max_files):
            break
    for index in rows.index:
        if len(selected_indices) >= int(max_files):
            break
        if index not in selected_indices:
            selected_indices.append(index)
    return rows.loc[selected_indices].drop(columns=["_memo_type_rank"]).reset_index(drop=True)


def build_ticker_memo_evidence_rows(
    files: pd.DataFrame,
    snippets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    snippets_by_path = {str(item.get("relative_path") or ""): item for item in snippets}
    rows = []
    for _, row in files.iterrows():
        path = str(row.get("relative_path") or "")
        item = snippets_by_path.get(path, {})
        evidence = best_investment_useful_snippet(item)
        rows.append(
            {
                "source_or_broker": row.get("source_or_broker") or "Company / filing source",
                "file_name": row.get("file_name") or path,
                "document_type": row.get("document_type") or "other",
                "category": row.get("category") or "Unknown",
                "evidence_type": evidence["evidence_type"],
                "extracted_evidence": evidence["snippet"],
                "evidence_quality": evidence["quality"],
                "extraction_status": evidence["extraction_status"],
                "relative_path": path,
            }
        )
    return rows


def build_deterministic_ticker_memo(
    ticker: str,
    files: pd.DataFrame,
    snippets: List[Dict[str, Any]],
    *,
    report_date: str,
    mode: str,
) -> str:
    """Build a source-grounded ticker memo without inferring financial conclusions."""
    evidence_rows = build_ticker_memo_evidence_rows(files, snippets)
    paths = files["relative_path"].astype(str).tolist() if not files.empty else []
    cited_paths = ", ".join(f"`{path}`" for path in paths)
    useful = [row for row in evidence_rows if row["extracted_evidence"]]
    lines = [
        f"# Ticker Investment Memo - {ticker} - {report_date}",
        "",
        f"Memo mode: {mode}",
        "",
        "## Files Reviewed",
        "",
        "| Source/broker | File name | Document type | Category | Extraction status | Relevance score |",
        "|---|---|---|---|---|---:|",
    ]
    snippets_by_path = {str(item.get("relative_path") or ""): item for item in snippets}
    for _, row in files.iterrows():
        path = str(row.get("relative_path") or "")
        status = str(snippets_by_path.get(path, {}).get("text_extraction_status") or "metadata-only")
        lines.append(
            f"| {_markdown_table_text(row.get('source_or_broker') or 'Company / filing source')} | "
            f"`{_markdown_table_text(row.get('file_name') or path)}` | "
            f"{_markdown_table_text(row.get('document_type') or 'other')} | "
            f"{_markdown_table_text(row.get('category') or 'Unknown')} | {status} | "
            f"{int(row.get('relevance_score') or 0)} |"
        )

    lines.extend(["", "## Executive Summary"])
    if files.empty:
        lines.append("No qualifying same-day files were identified for this ticker.")
    else:
        lines.append(
            f"{len(files)} same-day file(s) were reviewed for **{ticker}** across "
            f"{files['document_type'].nunique()} detected document type(s). "
            f"Limited extracted text was available; review source PDFs before making investment conclusions. "
            f"Sources: {cited_paths}"
        )

    lines.extend(["", "## Document Coverage Overview"])
    if files.empty:
        lines.append("No document coverage available.")
    else:
        for document_type, group in files.groupby("document_type", sort=True):
            group_paths = ", ".join(f"`{path}`" for path in group["relative_path"].astype(str))
            lines.append(f"- **{document_type}**: {len(group)} file(s). Sources: {group_paths}")

    lines.extend(
        [
            "",
            "## Key Extracted Evidence",
            "",
            "| Source/broker | File name | Evidence type | Extracted evidence | Extraction status |",
            "|---|---|---|---|---|",
        ]
    )
    for row in evidence_rows:
        snippet = row["extracted_evidence"] or "No investment-useful snippet found in limited extraction."
        lines.append(
            f"| {_markdown_table_text(row['source_or_broker'])} | `{_markdown_table_text(row['file_name'])}` | "
            f"{row['evidence_type']} | \"{_markdown_table_text(snippet)}\" | "
            f"{_markdown_table_text(row['extraction_status'])} |"
        )

    def add_evidence_section(title: str, selected_rows: List[Dict[str, Any]], empty_text: str) -> None:
        lines.extend(["", f"## {title}"])
        if not selected_rows:
            lines.append(empty_text)
            return
        for row in selected_rows:
            lines.append(
                f"- Direct limited-text excerpt from `{row['relative_path']}`: "
                f"\"{row['extracted_evidence']}\""
            )

    broker_rows = [row for row in useful if row["source_or_broker"] != "Company / filing source"]
    credit_rows = [
        row for row in useful
        if row["document_type"] == "credit report" or row["evidence_type"] == "Credit / Liquidity"
    ]
    earnings_rows = [
        row for row in useful
        if row["document_type"] in {
            "earnings release", "earnings presentation", "earnings supplement", "transcript",
            "prepared remarks", "10-Q/10Q", "8-K/8K", "MD&A",
        } or row["evidence_type"] in {"Earnings / Results", "Guidance / Estimates"}
    ]
    add_evidence_section(
        "Broker / Source Views",
        broker_rows,
        "No investment-useful broker/source excerpt was found in the limited extraction.",
    )
    add_evidence_section(
        "Credit / Balance Sheet Notes",
        credit_rows,
        "No verified credit, liquidity, leverage, or debt note was found in the limited extraction.",
    )
    add_evidence_section(
        "Earnings / Operating Notes",
        earnings_rows,
        "No verified earnings or operating note was found in the limited extraction.",
    )

    lines.extend(
        [
            "",
            "## Potential Bullish Points",
            "No bullish investment conclusion was generated automatically. Review the direct extracted evidence "
            f"and source PDFs before assessing upside. Sources reviewed: {cited_paths}",
            "",
            "## Potential Bearish Points / Risks",
            "No bearish investment conclusion was generated automatically. Review the direct extracted evidence "
            f"and source PDFs before assessing downside or risk. Sources reviewed: {cited_paths}",
            "",
            "## Open Questions for Geoff/Mitko",
            "- Do the source PDFs contain explicit rating or price-target changes?",
            "- Do company materials verify any EPS, revenue, margin, guidance, or estimate change?",
            "- Do credit documents identify liquidity, leverage, debt, or covenant concerns?",
            "- Which extracted evidence warrants full-document review?",
            "",
            "## Recommended Next Steps",
            "- Open the highest-relevance source documents and verify all financial figures and conclusions.",
            "- Compare company-provided materials with broker and credit-source commentary.",
            "- Record any verified rating, estimate, guidance, valuation, or credit changes manually.",
            "",
            "## Source References",
        ]
    )
    for row in evidence_rows:
        lines.append(
            f"- `{row['relative_path']}` ({row['document_type']}; {row['source_or_broker']}; "
            f"{row['extraction_status']})"
        )
    return "\n".join(lines)


def build_ticker_memo_evidence_payload(
    ticker: str,
    files: pd.DataFrame,
    snippets: List[Dict[str, Any]],
    *,
    max_total_chars: int = 40000,
) -> str:
    records = []
    used = 0
    for row in build_ticker_memo_evidence_rows(files, snippets):
        record = dict(row)
        record["verified_text_available"] = bool(row["extracted_evidence"])
        record["limited_text"] = record.pop("extracted_evidence") or INSUFFICIENT_TEXT_NOTICE
        encoded = json.dumps(record, ensure_ascii=False)
        if used + len(encoded) > max_total_chars:
            break
        records.append(record)
        used += len(encoded)
    return json.dumps(
        {
            "ticker": ticker,
            "grounding_rule": (
                "Metadata supports file/source/category/document-type observations only. Every financial or "
                "investment claim must be a direct quote from limited_text and cite relative_path."
            ),
            "files": records,
        },
        ensure_ascii=False,
    )


def validate_ticker_memo_grounding(text: str, snippets: List[Dict[str, Any]]) -> bool:
    required = [
        "## Files Reviewed", "## Executive Summary", "## Document Coverage Overview",
        "## Key Extracted Evidence", "## Broker / Source Views", "## Credit / Balance Sheet Notes",
        "## Earnings / Operating Notes", "## Potential Bullish Points",
        "## Potential Bearish Points / Risks", "## Open Questions for Geoff/Mitko",
        "## Recommended Next Steps", "## Source References",
    ]
    if not all(section in text for section in required):
        return False
    key_section = text.split("## Key Extracted Evidence", 1)[1].split("## Broker / Source Views", 1)[0]
    for line in key_section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "source/broker" in stripped.lower() or bool(re.fullmatch(r"[\s|:-]+", stripped)):
            continue
        cited = [
            item for item in snippets
            if str(item.get("relative_path") or "") in line or str(item.get("file_name") or "") in line
        ]
        if not cited:
            return False
        for quote in [part.strip() for part in re.findall(r'"([^"]+)"', line) if part.strip()]:
            if quote == "No investment-useful snippet found in limited extraction.":
                continue
            if not any(quote.lower() in _normalized_extracted_text(item).lower() for item in cited):
                return False
    source_names = {
        value
        for item in snippets
        for value in [str(item.get("relative_path") or ""), str(item.get("file_name") or "")]
        if value
    }
    claim_text = text.split("## Open Questions for Geoff/Mitko", 1)[0]
    if not validate_llm_brief_grounding(claim_text, snippets):
        return False
    for line in claim_text.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or bool(re.fullmatch(r"[\s|:-]+", stripped))
            or (stripped.startswith("|") and "file" in stripped.lower())
            or stripped.startswith("Memo mode:")
            or stripped == "Limited extracted text was available; review source PDFs before making investment conclusions."
        ):
            continue
        if not any(source in stripped for source in source_names):
            return False
        quotes = [part.strip() for part in re.findall(r'"([^"]+)"', stripped) if part.strip()]
        for quote in quotes:
            if quote == "No investment-useful snippet found in limited extraction.":
                continue
            cited = [
                item for item in snippets
                if str(item.get("relative_path") or "") in stripped or str(item.get("file_name") or "") in stripped
            ]
            if not any(quote.lower() in _normalized_extracted_text(item).lower() for item in cited):
                return False
    return True


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
    verified_text = verified_extracted_text_items(selected_text)
    lines = [
        f"# Daily Research Brief - {daily_source_title_suffix(source_name, relevance.get('relative_path', []))}",
        "",
        f"Source archive: `{source_name}`",
        f"Files inventoried: {len(relevance)}",
        f"Files lightly scanned: {len(selected_text)}",
        f"High-priority files: {len(relevance[relevance['priority_level'] == 'High']) if not relevance.empty else 0}",
        "",
        "## Metadata-Based Observations",
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

    lines.extend(["", "### Top High-Priority Documents"])
    if high.empty:
        lines.append("No files reached the deterministic high-priority threshold.")
    else:
        for _, row in high.iterrows():
            lines.append(
                f"- Score {int(row['relevance_score'])}: `{row['relative_path']}` "
                f"({row['document_type']}; {row['source_or_broker'] or 'source unknown'}; "
                f"{row['ticker'] or 'ticker uncertain'}). Rationale: {row['investment_rationale']}"
            )

    lines.extend(["", "### Broker Coverage Highlights"])
    if broker_summary.empty:
        lines.append("No normalized broker/source coverage was identified.")
    else:
        for _, row in broker_summary.head(12).iterrows():
            lines.append(
                f"- **{row['ticker']}**: {row['broker_report_count']} broker/source report(s) from "
                f"{row['brokers_or_sources']}; {row['attention_reason']}."
            )

    lines.extend(["", "### Category Summary"])
    for _, row in category_summary.iterrows():
        lines.append(
            f"- **{row['category']}**: {row['file_count']} file(s), "
            f"{row['high_priority_files']} high priority; common types: {row['top_document_types'] or 'other'}."
        )

    lines.extend(["", "## Extracted-Text Signals"])
    if not verified_text:
        lines.append(INSUFFICIENT_TEXT_NOTICE)
        lines.extend(["", "### Potential Areas to Review"])
        lines.append(
            "Review the high-priority source files listed above for actual results, estimates, "
            "ratings, guidance, and other financial conclusions."
        )
    else:
        lines.append(
            "The following are direct excerpts from successfully extracted source text. "
            "They are not independent conclusions."
        )
        for item in verified_text[:12]:
            text = _normalized_extracted_text(item)[:300]
            excerpt = text + ("..." if len(_normalized_extracted_text(item)) > 300 else "")
            lines.append(f"- `{item['relative_path']}`: extracted-text excerpt: \"{excerpt}\"")

    lines.extend(
        [
            "",
            "## Recommended Follow-Up",
            "For Geoff / Mitko:",
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


def _normalized_extracted_text(item: Dict[str, Any]) -> str:
    return re.sub(r"\s+", " ", str(item.get("extracted_text") or "")).strip()


def verified_extracted_text_items(selected_text: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    verified = []
    for item in selected_text:
        status = str(item.get("text_extraction_status") or "")
        text = _normalized_extracted_text(item)
        if status != "scanned" or len(text) < 160:
            continue
        verified.append(item)
    return verified


def validate_llm_brief_grounding(text: str, selected_text: List[Dict[str, Any]]) -> bool:
    """Require sensitive financial claims to be direct quotes from a cited, verified source."""
    verified = verified_extracted_text_items(selected_text)
    if not verified:
        return not any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SENSITIVE_FINANCE_PATTERNS)

    for line in text.splitlines():
        matched_patterns = [
            pattern for pattern in SENSITIVE_FINANCE_PATTERNS
            if re.search(pattern, line, flags=re.IGNORECASE)
        ]
        if not matched_patterns:
            continue
        cited = [
            item for item in verified
            if str(item.get("relative_path") or "") in line or str(item.get("file_name") or "") in line
        ]
        if not cited:
            return False
        quoted_parts = [part.strip() for part in re.findall(r'"([^"]+)"', line) if part.strip()]
        if not quoted_parts:
            return False
        if not any(
            any(
                len(quote) >= 20
                and quote.lower() in _normalized_extracted_text(item).lower()
                and all(re.search(pattern, quote, flags=re.IGNORECASE) for pattern in matched_patterns)
                for quote in quoted_parts
            )
            for item in cited
        ):
            return False
    return True


def build_llm_evidence_payload(
    relevance: pd.DataFrame,
    selected_text: List[Dict[str, Any]],
    *,
    max_total_chars: int = 22000,
) -> str:
    records = []
    used = 0
    for item in selected_text:
        snippet = _normalized_extracted_text(item)[:1200]
        status = str(item.get("text_extraction_status") or "")
        verified = status == "scanned" and len(snippet) >= 160
        record = {
            "relative_path": item.get("relative_path"),
            "ticker": item.get("ticker"),
            "category": item.get("category"),
            "source_or_broker": item.get("source_or_broker"),
            "document_type": item.get("document_type"),
            "relevance_score": item.get("relevance_score"),
            "text_extraction_status": status,
            "verified_text_available": verified,
            "limited_text": snippet if verified else INSUFFICIENT_TEXT_NOTICE,
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
            "grounding_rule": (
                "Metadata may establish file availability, ticker/source/category coverage, and document type only. "
                "Financial claims require verified_text_available=true and must cite that source filename."
            ),
            "selected_sources": records,
            "top_metadata": metadata,
            "broker_coverage": broker_coverage,
            "ticker_summary": ticker_summary,
        },
        ensure_ascii=False,
    )

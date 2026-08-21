from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests


GOOGLE_SHEET_ID = "14LWRzd5QeAOQKc4VlwU84gQBWGIuKX68tikxyzwN1_8"
GOOGLE_SHEET_GID = "940406158"
GOOGLE_SHEET_SHARED_URL = (
    f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/edit?usp=sharing"
)
GOOGLE_SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export"
    f"?format=csv&gid={GOOGLE_SHEET_GID}"
)
GOOGLE_SHEET_GVIZ_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/gviz/tq"
    f"?tqx=out:csv&gid={GOOGLE_SHEET_GID}"
)
GOOGLE_SHEET_PUBLISHED_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/pub"
    f"?gid={GOOGLE_SHEET_GID}&single=true&output=csv"
)
GOOGLE_SHEET_CSV_PATTERNS = [
    ("export", GOOGLE_SHEET_CSV_URL),
    ("gviz", GOOGLE_SHEET_GVIZ_CSV_URL),
    ("published", GOOGLE_SHEET_PUBLISHED_CSV_URL),
]
DEFAULT_TTL_SECONDS = 30 * 60

_CACHE: dict[str, Any] = {
    "loaded_at_monotonic": 0.0,
    "tickers": None,
    "status": None,
}


@dataclass
class TickerLoadStatus:
    source: str
    ok: bool
    count: int
    loaded_at: str
    message: str = ""
    url_pattern: str = ""


def _load_local_tickers() -> Dict[str, List[str]]:
    try:
        from tickers import tickers as local_tickers  # type: ignore
    except Exception:
        return {}
    if not isinstance(local_tickers, dict):
        return {}
    return _normalize_ticker_mapping(local_tickers)


def _normalize_ticker_mapping(raw: dict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for ticker, names in (raw or {}).items():
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            continue
        company_names: List[str] = []
        if isinstance(names, list):
            company_names = [str(x).strip() for x in names if str(x).strip()]
        elif names:
            company_names = [str(names).strip()]
        if symbol not in out:
            out[symbol] = []
        for name in company_names:
            if name and name not in out[symbol]:
                out[symbol].append(name)
    return dict(sorted(out.items()))


def _find_column(headers: list[str], candidates: set[str], allow_substring: bool = True) -> str | None:
    normalized = {h: str(h or "").strip().lower().replace("_", " ") for h in headers}
    for original, cleaned in normalized.items():
        if cleaned in candidates:
            return original
    if not allow_substring:
        return None
    for original, cleaned in normalized.items():
        if any(c in cleaned for c in candidates):
            return original
    return None


def _split_ticker_cell(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in re_split_ticker_delimiters(raw)]
    out: list[str] = []
    for part in parts:
        symbol = part.replace("$", "").strip().upper()
        if len(symbol.split()) > 1:
            # Section dividers such as "CREDIT ONLY" are labels, not tickers.
            # Real symbols never contain an internal space; dual listings like
            # "IAUX / IAU" have already been split on the delimiter above.
            continue
        if not symbol:
            continue
        if symbol not in out:
            out.append(symbol)
    return out


def re_split_ticker_delimiters(value: str) -> list[str]:
    import re

    return re.split(r"\s*(?:/|,|;|\bor\b)\s*", value, flags=re.IGNORECASE)


def _looks_like_ticker_cell(value: str) -> bool:
    symbols = _split_ticker_cell(value)
    return bool(symbols) and all(1 <= len(sym) <= 12 and any(ch.isalpha() for ch in sym) for sym in symbols)


TICKER_HEADER_NAMES = {"ticker", "tickers", "symbol", "ticker symbol", "stock ticker"}
COMPANY_HEADER_NAMES = {
    "company",
    "company name",
    "name",
    "issuer",
    "security name",
    "security",
    "entity",
    "entity name",
}


def _find_header_row(rows: list[list[str]]) -> tuple[int | None, int | None, int | None]:
    """
    Locate the header row and the ticker/company column positions.

    Returns (header_idx, ticker_col_idx, company_col_idx). company_col_idx is
    None when the sheet carries no company-name column at all.
    """
    for idx, row in enumerate(rows[:10]):
        ticker_col = _find_column(row, TICKER_HEADER_NAMES)
        if not ticker_col:
            continue
        ticker_col_idx = row.index(ticker_col)
        # Exact matching only: a coverage column such as "Goldman Sachs" must
        # never be mistaken for a company-name column.
        company_col = _find_column(row, COMPANY_HEADER_NAMES, allow_substring=False)
        company_col_idx = row.index(company_col) if company_col in row else None
        return idx, ticker_col_idx, company_col_idx
    return None, None, None


def _rows_to_ticker_map(
    rows: list[list[str]],
    *,
    ticker_col_idx: int = 0,
    company_col_idx: int | None = None,
    start: int = 0,
) -> Dict[str, List[str]]:
    """Build {TICKER: [display_name]} from data rows, skipping banners/headers."""
    out: Dict[str, List[str]] = {}
    for row in rows[start:]:
        ticker_value = row[ticker_col_idx] if ticker_col_idx < len(row) else ""
        company = (
            row[company_col_idx].strip()
            if company_col_idx is not None and company_col_idx < len(row)
            else ""
        )
        for ticker in _split_ticker_cell(ticker_value):
            out.setdefault(ticker, [])
            display_name = company or ticker
            if display_name and display_name not in out[ticker]:
                out[ticker].append(display_name)
    return dict(sorted(out.items()))


def _parse_sheet_csv(text: str) -> Dict[str, List[str]]:
    rows = list(csv.reader(io.StringIO(text)))
    rows = [[str(cell or "").strip() for cell in row] for row in rows if any(str(cell or "").strip() for cell in row)]
    if not rows:
        raise ValueError("Google Sheet CSV export returned no rows.")

    # Find the real header row first. Google Sheets tabs often carry banner or
    # note rows above the column names, and those rows must never be read as data.
    header_idx, ticker_col_idx, company_col_idx = _find_header_row(rows)
    if header_idx is not None and ticker_col_idx is not None:
        return _rows_to_ticker_map(
            rows,
            ticker_col_idx=ticker_col_idx,
            company_col_idx=company_col_idx,
            start=header_idx + 1,
        )

    # Headerless ticker tab: Column A ticker, optional Column B company name.
    if _looks_like_ticker_cell(rows[0][0] if rows[0] else ""):
        return _rows_to_ticker_map(rows, ticker_col_idx=0, company_col_idx=1, start=0)

    # Last-resort: first column is ticker-like across most rows, so treat it as a ticker list.
    if sum(1 for row in rows if row and _looks_like_ticker_cell(row[0])) >= max(1, len(rows) // 2):
        return _rows_to_ticker_map(rows, ticker_col_idx=0, company_col_idx=1, start=0)

    raise ValueError("No ticker-like column found in Google Sheet Tickers tab.")


def _apply_local_company_names(mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Fill in company names for tickers the sheet only knows by symbol.

    The Tickers tab has no company column - column A is the symbol and
    columns B+ are broker coverage flags that echo the symbol back - so the
    parser has no name to attach and falls back to the symbol itself.

    That fallback is not only cosmetic. These names are compiled into the
    keyword regexes used to find mentions in fund letters, Substack posts and
    podcast transcripts, so a symbol-only name makes the scraper search for
    "AMZN" instead of "Amazon" and miss the prose mentions that make up most
    real hits. Prefer a real name whenever tickers.py knows one.

    A symbol with no known name keeps the symbol as its keyword: an empty
    keyword list would compile to a regex that matches every paragraph.
    """
    local = _load_local_tickers()
    out: Dict[str, List[str]] = {}
    for symbol, names in (mapping or {}).items():
        # A sheet value equal to the symbol means "no name", not a real name.
        from_sheet = [
            str(n).strip()
            for n in (names or [])
            if str(n).strip() and str(n).strip().upper() != symbol
        ]
        if from_sheet:
            out[symbol] = from_sheet
            continue
        out[symbol] = list(local.get(symbol) or []) or [symbol]
    return out


def _http_error_summary(resp: requests.Response) -> str:
    reason = (resp.reason or "").strip()
    if reason:
        return f"{resp.status_code} {reason}"
    return f"HTTP {resp.status_code}"


def load_tickers_from_google_sheet(timeout: int = 12) -> tuple[Dict[str, List[str]], str]:
    # Keep CSV export as the actual read path; GOOGLE_SHEET_SHARED_URL is the
    # human-facing reference URL for the same workbook.
    errors: list[str] = []
    headers = {"User-Agent": "Mozilla/5.0 CutlerPlatformBuild"}
    for pattern_name, url in GOOGLE_SHEET_CSV_PATTERNS:
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            if resp.status_code >= 400:
                errors.append(f"{pattern_name}: {_http_error_summary(resp)}")
                continue
            text = resp.text or ""
            if not text.strip():
                errors.append(f"{pattern_name}: empty CSV response")
                continue
            parsed = _parse_sheet_csv(text)
            if not parsed:
                errors.append(f"{pattern_name}: CSV parsed but no tickers found")
                continue
            return _apply_local_company_names(parsed), pattern_name
        except Exception as exc:
            errors.append(f"{pattern_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors) if errors else "Google Sheet CSV fetch failed.")


def load_live_tickers(
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    force_refresh: bool = False,
) -> Tuple[Dict[str, List[str]], TickerLoadStatus]:
    now = time.monotonic()
    cached_tickers = _CACHE.get("tickers")
    cached_status = _CACHE.get("status")
    if (
        not force_refresh
        and isinstance(cached_tickers, dict)
        and isinstance(cached_status, TickerLoadStatus)
        and now - float(_CACHE.get("loaded_at_monotonic") or 0.0) < ttl_seconds
    ):
        return cached_tickers, cached_status

    loaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        live, url_pattern = load_tickers_from_google_sheet()
        if not live:
            raise ValueError("Google Sheet did not contain any valid tickers.")
        status = TickerLoadStatus(
            source="google_sheet",
            ok=True,
            count=len(live),
            loaded_at=loaded_at,
            message=f"Tickers loaded from Google Sheet: {len(live)}",
            url_pattern=url_pattern,
        )
        _CACHE.update({"loaded_at_monotonic": now, "tickers": live, "status": status})
        return live, status
    except Exception as exc:
        fallback = _load_local_tickers()
        status = TickerLoadStatus(
            source="local_fallback",
            ok=False,
            count=len(fallback),
            loaded_at=loaded_at,
            message=f"Using fallback local ticker list: {len(fallback)} — Google Sheet returned {exc}",
            url_pattern="local_fallback",
        )
        _CACHE.update({"loaded_at_monotonic": now, "tickers": fallback, "status": status})
        return fallback, status


def get_ticker_universe(*, ttl_seconds: int = DEFAULT_TTL_SECONDS, force_refresh: bool = False) -> Dict[str, List[str]]:
    tickers, _status = load_live_tickers(ttl_seconds=ttl_seconds, force_refresh=force_refresh)
    return tickers


def get_ticker_load_status() -> TickerLoadStatus:
    status = _CACHE.get("status")
    if isinstance(status, TickerLoadStatus):
        return status
    tickers, status = load_live_tickers()
    return status

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
GOOGLE_SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export"
    f"?format=csv&gid={GOOGLE_SHEET_GID}"
)
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


def _find_column(headers: list[str], candidates: set[str]) -> str | None:
    normalized = {h: str(h or "").strip().lower().replace("_", " ") for h in headers}
    for original, cleaned in normalized.items():
        if cleaned in candidates:
            return original
    for original, cleaned in normalized.items():
        if any(c in cleaned for c in candidates):
            return original
    return None


def _parse_sheet_csv(text: str) -> Dict[str, List[str]]:
    reader = csv.DictReader(io.StringIO(text))
    headers = reader.fieldnames or []
    ticker_col = _find_column(headers, {"ticker", "symbol", "ticker symbol", "stock ticker"})
    if not ticker_col:
        raise ValueError("No ticker-like column found in Google Sheet Tickers tab.")
    company_col = _find_column(headers, {"company", "company name", "name", "issuer", "security name"})

    out: Dict[str, List[str]] = {}
    for row in reader:
        ticker = str(row.get(ticker_col) or "").strip().upper()
        if not ticker:
            continue
        ticker = ticker.replace("$", "").strip()
        if not ticker:
            continue
        company = str(row.get(company_col) or "").strip() if company_col else ""
        out.setdefault(ticker, [])
        if company and company not in out[ticker]:
            out[ticker].append(company)
    return dict(sorted(out.items()))


def load_tickers_from_google_sheet(timeout: int = 12) -> Dict[str, List[str]]:
    resp = requests.get(GOOGLE_SHEET_CSV_URL, timeout=timeout)
    resp.raise_for_status()
    text = resp.text or ""
    if not text.strip():
        raise ValueError("Google Sheet CSV export returned empty content.")
    return _parse_sheet_csv(text)


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
        live = load_tickers_from_google_sheet()
        if not live:
            raise ValueError("Google Sheet did not contain any valid tickers.")
        status = TickerLoadStatus(
            source="google_sheet",
            ok=True,
            count=len(live),
            loaded_at=loaded_at,
            message="Tickers loaded from Google Sheet",
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
            message=f"Using fallback local ticker list ({type(exc).__name__}: {exc})",
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

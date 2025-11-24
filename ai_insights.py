"""
AI Insights for Cutler Capital Scraper

Given a manifest (one compiled run: batch + quarter), this module:
- Loads all excerpts_clean.json files referenced in the manifest.
- Aggregates paragraphs by ticker across all funds in that run.
- Calls the OpenAI Responses API (optionally with web_search) to classify
  each ticker as buy / hold / sell / unclear, with reasoning.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import openai
import importlib.util

HERE = Path(__file__).resolve().parent

# Load .env file automatically and set API key
load_dotenv(dotenv_path=HERE / ".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# Import tickers mapping so we can show company names
import importlib.util

def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if not spec or not spec.loader:
        raise ImportError(f"Cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod

tickers_mod = _import("tickers", HERE / "tickers.py")
TICKERS_MAP: Dict[str, List[str]] = getattr(tickers_mod, "tickers", {})

SYSTEM_PROMPT = """
You are an equity research assistant for Cutler Capital.

You are given:
- Ticker symbols and company names
- Narrative excerpts from professional fund commentaries
- Basic metadata (fund family, source PDF, letter date, pages)

Your job is to infer the *effective stance* for each ticker for the quarter:
- "buy": adding / initiating / clearly positive with intent to own more or continue owning
- "hold": generally positive or neutral, continuing to own but not clearly adding or exiting
- "sell": reducing / trimming / exiting / clearly negative or loss of conviction
- "unclear": commentary does not provide enough evidence to decide, or is very mixed

STRICT RULES:
- Base your stance primarily on the fund commentary excerpts.
- Do NOT guess. Err on the side of "unclear" when evidence is weak or contradictory.
- If multiple funds have different views (e.g. some buy, some sell), describe that and
  choose "unclear" or the stance that best reflects the balance of evidence, but explain.
- All reasoning must be traceable back to the excerpts plus obvious financial logic.

WEB_CHECK FIELD:
- Always populate `web_check_summary` with at least one short sentence.
- If you have no additional context beyond the commentary, set it to:
  "No additional web context used."
"""

def _build_ticker_context_from_manifest(
    manifest: Dict[str, Any],
    max_chars_per_ticker: int = 4000,
) -> Dict[str, Dict[str, Any]]:
    """
    From a manifest dict, aggregate paragraphs per ticker.

    Returns:
        {
          "ABNB": {
             "company_names": [...],
             "fund_families": [...],
             "snippets": [
                 "Source: Baron Funds – Q3 2025 Growth Fund (letter date: Sep 30, 2025; pages: [5])\nThe shares of Airbnb...",
                 ...
             ]
          },
          ...
        }
    """
    by_ticker: Dict[str, Dict[str, Any]] = {}

    for item in manifest.get("items", []):
        ej_path = item.get("excerpts_json")
        if not ej_path:
            continue
        p = Path(ej_path)
        if not p.exists():
            continue

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        fund_family = item.get("fund_family", "Unknown fund")
        source_pdf = item.get("source_pdf_name", "")
        letter_date = item.get("letter_date", "")

        for ticker, entries in data.items():
            if not isinstance(entries, list):
                continue

            record = by_ticker.setdefault(
                ticker,
                {
                    "company_names": TICKERS_MAP.get(ticker, []),
                    "snippets": [],
                    "fund_families": set(),  # we'll convert to list later
                },
            )
            record["fund_families"].add(fund_family)

            for entry in entries:
                text = (entry.get("text") or "").strip()
                if not text:
                    continue
                pages = entry.get("pages") or []
                meta = f"Source: {fund_family} – {source_pdf} (letter date: {letter_date}; pages: {pages})"
                snippet = f"{meta}\n{text}\n"
                record["snippets"].append(snippet)

    # Truncate and normalise types
    for ticker, rec in by_ticker.items():
        # snippets truncation
        snippets = rec["snippets"]
        combined: List[str] = []
        current_len = 0
        for s in snippets:
            if current_len + len(s) > max_chars_per_ticker:
                break
            combined.append(s)
            current_len += len(s)
        rec["snippets"] = combined

        # fund_families: set -> sorted list
        funds = rec.get("fund_families", set())
        if isinstance(funds, set):
            rec["fund_families"] = sorted(funds)

    return by_ticker


def _call_openai_for_ticker(
    ticker: str,
    company_names: List[str],
    quarter: str,
    batch: str,
    snippet_block: str,
    model: str = "gpt-4o-mini",
    use_web: bool = True,  # kept for signature compatibility; ignored for now
) -> Dict[str, Any]:
    """
    Call OpenAI Chat Completions API for a single ticker.
    Returns a dict with stance / confidence / reasoning.
    """
    companies_str = ", ".join(company_names) if company_names else "N/A"

    user_prompt = f"""
You are analysing one stock.

Ticker: {ticker}
Company names / aliases: {companies_str}
Quarter: {quarter}
Batch label (internal only): {batch}

Below are excerpts from fund commentaries that mention this ticker.
Use ONLY these excerpts to decide whether the commentary implies BUY, HOLD, SELL,
or UNCLEAR for this quarter.

Excerpts:
\"\"\" 
{snippet_block}
\"\"\" 

Now, respond ONLY as JSON with this exact schema:

{{
  "ticker": "{ticker}",
  "stance": "buy" | "hold" | "sell" | "unclear",
  "confidence": float between 0.0 and 1.0,
  "primary_reasoning": "short explanation (2-4 sentences) grounded in the excerpts",
  "commentary_evidence": [
    "1-3 short supporting quotes or paraphrases from the excerpts"
  ],
  "web_check_summary": "optional 1-3 sentence summary of any important current info; empty string if not needed"
}}
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # IMPORTANT: use ChatCompletion for compatibility (no `openai.chat` namespace)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )

    # Handle both dict-like and object-like responses
    try:
        choice = response.choices[0]
        msg = choice["message"] if isinstance(choice, dict) else getattr(choice, "message", choice)
        if isinstance(msg, dict):
            text = msg.get("content", "")
        else:
            text = getattr(msg, "content", str(msg))
    except Exception:
        text = str(response)

    try:
        data = json.loads(text)
    except Exception:
        data = {
            "ticker": ticker,
            "stance": "unclear",
            "confidence": 0.0,
            "primary_reasoning": "Model returned non-JSON output; treating as unclear.",
            "commentary_evidence": [],
            "web_check_summary": "",
            "raw_output": text,
        }
    return data

def generate_ticker_stances(
    manifest: Dict[str, Any],
    batch: str,
    quarter: str,
    model: str = "gpt-4o-mini",
    use_web: bool = True,
) -> List[Dict[str, Any]]:
    """
    High-level entry point:
    - manifest: one manifest dict (for a specific batch+quarter run)
    - Returns a list of per-ticker results dicts.
    """
    contexts = _build_ticker_context_from_manifest(manifest)
    results: List[Dict[str, Any]] = []

    if not contexts:
        return results

    for idx, (ticker, ctx) in enumerate(sorted(contexts.items()), 1):
        snippets = ctx.get("snippets") or []
        if not snippets:
            continue
        snippet_block = "\n\n---\n\n".join(snippets)
        company_names = ctx.get("company_names") or []
        try:
            res = _call_openai_for_ticker(
                ticker=ticker,
                company_names=company_names,
                quarter=quarter,
                batch=batch,
                snippet_block=snippet_block,
                model=model,
                use_web=use_web,
            )
            # Ensure ticker field is set correctly
            res.setdefault("ticker", ticker)
            res.setdefault("company_names", company_names)
            res.setdefault("fund_families", ctx.get("fund_families", []))
            results.append(res)
        except Exception as e:
            results.append(
                {
                    "ticker": ticker,
                    "company_names": company_names,
                    "stance": "unclear",
                    "confidence": 0.0,
                    "primary_reasoning": f"OpenAI call failed: {e}",
                    "commentary_evidence": [],
                    "web_check_summary": "",
                }
            )

    return results

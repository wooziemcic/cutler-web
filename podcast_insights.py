from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import re

import openai

# Try to pull full Cutler universe for nicer company labels (optional)
try:
    from tickers import tickers as CUTLER_TICKERS
except Exception:
    CUTLER_TICKERS = {}

# ------------------------------
# OpenAI setup
# ------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY


@dataclass
class PodcastSnippet:
    ticker: str
    podcast_id: str
    episode_id: str
    podcast_name: str
    title: str
    published: str
    snippet: str
    episode_url: str | None = None
    company_names: List[str] | None = None


@dataclass
class EpisodeRecord:
    episode_uid: str
    episode_id: str
    podcast_id: str
    title: str
    published: str
    episode_url: str | None
    transcript: str


# ---------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------

COMPANY_SYSTEM_PROMPT = """You are a research assistant helping a buy-side
investment firm understand how specific companies are discussed across
finance podcasts.

Given short excerpts where a particular stock or company is mentioned,
infer the qualitative stance of the discussion (bullish, bearish,
neutral/hold, or unclear) and summarise the reasoning in a structured
JSON format.

You are NOT giving investment advice. You are only summarizing tone and
qualitative commentary from the excerpts.

Definitions:
- "buy": clearly optimistic tone, expecting upside or strongly positive
  vs peers.
- "sell": clearly negative tone, highlighting major risks, overvaluation,
  or downside vs peers.
- "hold": mixed, balanced, or only mildly directional; basically
  monitor/neutral.
- "unclear": the company is mentioned but not enough signal to infer a
  stance (passing reference, lists of names, etc.).

You must output a single JSON object with keys:
- ticker (string)
- stance ("buy" | "sell" | "hold" | "unclear")
- stance_confidence (float 0.0–1.0)
- overall_summary (string, 3–7 sentences)
- supporting_points (list of strings)
- risks_or_headwinds (list of strings)
- episodes (list of episode objects):
    [{
       "podcast_name": ...,
       "episode_title": ...,
       "published": ...,
       "episode_url": ...,
       "episode_view": "2–4 sentences on how this episode contributes to the stance"
     }]

Rules:
- Use ONLY the info in the excerpts.
- Do NOT fabricate numbers, events, or fundamentals that are not
  clearly implied.
- If different episodes disagree, you may choose "hold" and explain both
  sides.
- If the company is only mentioned briefly, you may choose "unclear",
  but still provide a useful narrative summary of what IS being
  discussed (macro themes, sector comments, etc.).
- Respond with VALID JSON only. No extra commentary.
"""


EPISODE_SYSTEM_PROMPT = """You are a senior research assistant for a buy-side
investment firm. Your job is to summarise full finance podcast episodes
for busy portfolio managers.

You will be given the transcript of ONE episode. Produce a concise,
medium-length paragraph that captures:
- core topic and thesis of the episode
- key macro or sector themes
- any notable risks, tensions, or debates
- overall tone (constructive, cautious, speculative, etc.)

You are not making stock recommendations, only summarizing what is
discussed.

Output MUST be JSON with a single key:
{ "summary": "..." }

Do NOT include any commentary outside JSON.
"""


def build_company_user_message(
    ticker: str,
    company_names: List[str],
    snippets: List[PodcastSnippet],
) -> str:
    company_label = ", ".join(company_names) if company_names else ticker
    lines: List[str] = []

    lines.append(
        f"You are analyzing recent podcast excerpts that mention the company "
        f"{company_label} (ticker: {ticker})."
    )
    lines.append("Each snippet includes podcast metadata and the surrounding text where the company appears.")
    lines.append("")
    lines.append("EXCERPTS:")

    for i, sn in enumerate(snippets, start=1):
        lines.append(f"[Snippet {i}]")
        lines.append(f"Podcast: {sn.podcast_name}")
        lines.append(f"Episode: {sn.title}")
        lines.append(f"Published: {sn.published}")
        if sn.episode_url:
            lines.append(f"URL: {sn.episode_url}")
        lines.append("Text:")
        lines.append(sn.snippet.strip())
        lines.append("")

    lines.append("")
    lines.append("Now respond with a single JSON object with the keys described in the system prompt.")
    return "\n".join(lines)


def build_episode_user_message(ep: EpisodeRecord, max_chars: int = 6000) -> str:
    """
    Build a prompt for summarising a single episode. Transcript is truncated
    to keep tokens under control.
    """
    transcript = ep.transcript.strip()
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars]

    lines: List[str] = []
    lines.append(f"Podcast: {ep.podcast_id}")
    lines.append(f"Episode title: {ep.title}")
    lines.append(f"Published: {ep.published}")
    if ep.episode_url:
        lines.append(f"URL: {ep.episode_url}")
    lines.append("")
    lines.append("TRANSCRIPT:")
    lines.append(transcript)
    lines.append("")
    lines.append('Respond with JSON of the form: { "summary": "..." }')
    return "\n".join(lines)


# ---------------------------------------------------------------------
# OpenAI wrappers
# ---------------------------------------------------------------------

def call_chat_completion(model: str, system_prompt: str, user_message: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY (or OPENAI_KEY) is not set in environment.")
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    return resp["choices"][0]["message"]["content"]


def generate_company_insight(
    ticker: str,
    company_names: List[str],
    snippets: List[PodcastSnippet],
    model: str,
) -> Dict[str, Any]:
    """
    Normal path: company *is* mentioned in excerpts.
    """
    if not snippets:
        return {
            "ticker": ticker,
            "company_names": company_names,
            "stance": "unclear",
            "stance_confidence": 0.0,
            "overall_summary": "No valid excerpts were available for this ticker.",
            "supporting_points": [],
            "risks_or_headwinds": [],
            "episodes": [],
            "episode_summaries": [],
            "raw_model_output": "",
            "has_mentions": False,
        }

    user_message = build_company_user_message(ticker, company_names, snippets)

    try:
        content = call_chat_completion(model, COMPANY_SYSTEM_PROMPT, user_message)
    except Exception as e:
        return {
            "ticker": ticker,
            "company_names": company_names,
            "stance": "unclear",
            "stance_confidence": 0.0,
            "overall_summary": f"OpenAI call failed: {e}",
            "supporting_points": [],
            "risks_or_headwinds": [],
            "episodes": [],
            "episode_summaries": [],
            "raw_model_output": "",
            "has_mentions": True,
        }

    try:
        parsed = json.loads(content)
    except Exception:
        # Fallback: treat the content as a free-form summary
        return {
            "ticker": ticker,
            "company_names": company_names,
            "stance": "unclear",
            "stance_confidence": 0.0,
            "overall_summary": content.strip(),
            "supporting_points": [],
            "risks_or_headwinds": [],
            "episodes": [],
            "episode_summaries": [],
            "raw_model_output": content,
            "has_mentions": True,
        }

    return {
        "ticker": parsed.get("ticker", ticker),
        "company_names": company_names,
        "stance": parsed.get("stance", "unclear"),
        "stance_confidence": float(parsed.get("stance_confidence", 0.0) or 0.0),
        "overall_summary": (parsed.get("overall_summary") or "").strip(),
        "supporting_points": parsed.get("supporting_points") or [],
        "risks_or_headwinds": parsed.get("risks_or_headwinds") or [],
        "episodes": parsed.get("episodes") or [],
        "episode_summaries": [],
        "raw_model_output": content,
        "has_mentions": True,
    }


def generate_episode_summary(ep: EpisodeRecord, model: str) -> str:
    """
    Generate a medium-length paragraph summary for one episode.
    Used in fallback mode when a ticker has *no* snippets.
    """
    user_message = build_episode_user_message(ep)
    try:
        content = call_chat_completion(model, EPISODE_SYSTEM_PROMPT, user_message)
    except Exception as e:
        return f"Failed to summarise this episode due to an API error: {e}"

    try:
        parsed = json.loads(content)
        summary = parsed.get("summary", "").strip()
        if summary:
            return summary
    except Exception:
        pass

    return content.strip()


# ---------------------------------------------------------------------
# Loading excerpts + episodes
# ---------------------------------------------------------------------

def load_excerpts_and_episodes(path: Path) -> Tuple[Dict[str, List[PodcastSnippet]], Dict[str, EpisodeRecord]]:
    """
    Load the JSON from podcast_excerpts.py and split into:
    - excerpts_by_ticker: { ticker: [PodcastSnippet, ...], ... }
    - episodes: { episode_uid: EpisodeRecord, ... }
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    episodes_raw = raw.pop("_episodes", {})
    episodes: Dict[str, EpisodeRecord] = {}
    for uid, obj in episodes_raw.items():
        episodes[uid] = EpisodeRecord(
            episode_uid=uid,
            episode_id=obj.get("episode_id", ""),
            podcast_id=obj.get("podcast_id", ""),
            title=obj.get("title", ""),
            published=obj.get("published", ""),
            episode_url=obj.get("episode_url"),
            transcript=obj.get("transcript", ""),
        )

    excerpts_by_ticker: Dict[str, List[PodcastSnippet]] = {}
    for ticker, items in raw.items():
        snippets: List[PodcastSnippet] = []
        for s in items:
            snippets.append(
                PodcastSnippet(
                    ticker=ticker,
                    podcast_id=s.get("podcast_id", ""),
                    episode_id=s.get("episode_id", ""),
                    podcast_name=s.get("podcast_name", s.get("podcast_id", "")),
                    title=s.get("title", ""),
                    published=s.get("published", ""),
                    snippet=s.get("snippet", ""),
                    episode_url=s.get("episode_url"),
                    company_names=s.get("company_names") or [],
                )
            )
        excerpts_by_ticker[ticker] = snippets

    return excerpts_by_ticker, episodes


# --- Heuristic for "strong" company mentions -----------------------------

# Generic finance/industry words – if a company name is *only* these,
# we treat it as too generic to use as an explicit alias.
_GENERIC_TOKENS = {
    "financial", "institution", "institutions", "group", "business",
    "bank", "banks", "capital", "services", "service", "company",
    "corp", "corporation", "inc", "inc.", "limited", "ltd", "holdings",
    "trust", "national", "international", "global", "life", "met",
    "engineering", "first"
}


def _normalize_for_match(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_name_variants(company_names):
    variants = []
    for raw in company_names or []:
        norm = _normalize_for_match(raw)
        if not norm:
            continue
        tokens = norm.split()
        if all(t in _GENERIC_TOKENS for t in tokens):
            continue
        variants.append(" ".join(tokens))
    # dedupe
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _snippet_has_specific_alias(snippet: str, company_names) -> bool:
    variants = _build_name_variants(company_names)
    if not variants:
        return False
    text = f" {_normalize_for_match(snippet)} "
    return any(f" {v} " in text for v in variants)


def has_strong_company_match(
    snippets: list["PodcastSnippet"],
    ticker: str,
    company_names: list[str],
) -> bool:
    """
    Return True only if at least one snippet clearly contains the company
    name (not just generic words or fuzzy noise).

    We deliberately do *not* rely on the ticker symbol itself here to avoid
    false positives for tickers like T, J, etc.
    """
    if not snippets:
        return False

    variants = _build_name_variants(company_names)
    if not variants:
        # No specific name tokens to look for – treat as "no strong signal"
        return False

    # Check each snippet's text
    for sn in snippets:
        text = _normalize_for_match(sn.snippet or "")
        if not text:
            continue
        padded = f" {text} "
        for v in variants:
            if f" {v} " in padded:
                return True

    return False


# ---------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------

def build_insights(
    excerpts_by_ticker: Dict[str, List[PodcastSnippet]],
    episodes: Dict[str, EpisodeRecord],
    model: str,
) -> List[Dict[str, Any]]:
    """
    Build insights for each ticker.

    Behaviour:

    - First, we decide which tickers have *strong* mentions using
      has_strong_company_match (explicit company name in text).
    - If AT LEAST ONE ticker has strong mentions:
        * Those tickers get full AI stance analysis (Company Mode).
        * Other tickers are marked has_mentions=False and get no episode
          summaries attached.
    - If NO ticker has strong mentions:
        * We do NOT produce any company-level stances.
        * Instead, we generate episode-level summaries once and attach the
          SAME episode_summaries list to every ticker
          (final.py merges them and goes into Episode Mode).
    """
    episode_summaries_cache: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []

    tickers_sorted = sorted(excerpts_by_ticker.keys())
    print(f"[INFO] Building podcast insights for {len(tickers_sorted)} tickers using model {model}")

    # ---------- 1) Pre-compute company_names per ticker ----------
    per_ticker_company_names: Dict[str, List[str]] = {}
    for ticker in tickers_sorted:
        snippets = excerpts_by_ticker[ticker]
        company_names: List[str] = []

        # union of company_names embedded in snippets
        for sn in snippets:
            if sn.company_names:
                for name in sn.company_names:
                    if name not in company_names:
                        company_names.append(name)

        # if still empty, pull from master tickers mapping
        if not company_names and ticker in CUTLER_TICKERS:
            names_from_master = CUTLER_TICKERS.get(ticker, [])
            company_names = list(names_from_master)

        per_ticker_company_names[ticker] = company_names

    # ---------- 2) Decide “strong mention” vs none ----------
    strong_flags: Dict[str, bool] = {}
    for ticker in tickers_sorted:
        snippets = excerpts_by_ticker[ticker]
        company_names = per_ticker_company_names.get(ticker, [])
        strong = has_strong_company_match(snippets, ticker, company_names)
        strong_flags[ticker] = strong
        print(f"[INFO] Strong-mention check for {ticker}: {strong} (snippets={len(snippets)})")

    any_strong_mentions = any(strong_flags.values())
    print(f"[INFO] Any strong mentions? {any_strong_mentions}")

    # ---------- 3) If NO strong mentions, build episode summaries once ----------
    global_episode_summaries: List[Dict[str, Any]] = []
    if not any_strong_mentions and episodes:
        print("[INFO] No strong company mentions detected. Generating episode summaries for Episode Mode.")
        for uid, ep in episodes.items():
            if uid in episode_summaries_cache:
                summary = episode_summaries_cache[uid]
            else:
                summary = generate_episode_summary(ep, model=model)
                episode_summaries_cache[uid] = summary

            global_episode_summaries.append(
                {
                    "episode_id": ep.episode_id or uid,
                    "podcast_id": ep.podcast_id,
                    "podcast_name": ep.podcast_id,  # we don't have a nicer name here
                    "episode_title": ep.title,
                    "published": ep.published,
                    "episode_url": ep.episode_url,
                    "summary": summary,
                }
            )

    # ---------- 4) Build per-ticker insight objects ----------
    for idx, ticker in enumerate(tickers_sorted, start=1):
        snippets = excerpts_by_ticker[ticker]
        company_names = per_ticker_company_names.get(ticker, [])
        print(
            f"[INFO] [{idx}/{len(tickers_sorted)}] Processing {ticker} "
            f"with {len(snippets)} snippets (strong={strong_flags[ticker]})..."
        )

        if strong_flags[ticker]:
            # Normal company-specific stance path
            insight = generate_company_insight(
                ticker=ticker,
                company_names=company_names,
                snippets=snippets,
                model=model,
            )
            insight["has_mentions"] = True

        else:
            # No strong mention for this ticker
            if not any_strong_mentions and global_episode_summaries:
                # Global Episode Mode: attach shared episode summaries
                episode_summaries = global_episode_summaries
            else:
                # There ARE other tickers with strong mentions, so we do not
                # attach episode summaries to this ticker.
                episode_summaries = []

            insight = {
                "ticker": ticker,
                "company_names": company_names,
                "stance": "not_mentioned",
                "stance_confidence": 0.0,
                "overall_summary": (
                    "This company was not explicitly and clearly mentioned in "
                    "any of the selected podcast transcripts for this batch."
                    if not any_strong_mentions
                    else "This company does not have clear, explicit mentions "
                         "in the excerpts used for company-level analysis."
                ),
                "supporting_points": [],
                "risks_or_headwinds": [],
                "episodes": [],
                "episode_summaries": episode_summaries,
                "raw_model_output": "",
                "has_mentions": False,
            }

        results.append(insight)

    return results


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate podcast AI insights from excerpts JSON."
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input JSON from podcast_excerpts.py",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output JSON path for insights",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default="gpt-4o-mini",
        help="OpenAI ChatCompletion model name",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    model = args.model

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # Load data from podcast_excerpts.py
    excerpts_by_ticker, episodes = load_excerpts_and_episodes(in_path)

    # Build list of per-ticker insights (what final.py expects)
    insights = build_insights(excerpts_by_ticker, episodes, model=model)

    # Write list -> JSON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(insights, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote podcast insights for {len(insights)} tickers to {out_path}")


if __name__ == "__main__":
    main()

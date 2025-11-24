#!/usr/bin/env python3
"""
podcast_excerpts.py

Extracts:
- Ticker-specific snippets from podcast transcripts
- AND full-episode transcripts + metadata for fallback summarization

Output JSON structure:

{
    "AAPL": [
        { "snippet": "...", "title": "...", "podcast_id": "...", ... }
    ],
    "MSFT": [...],
    "_episodes": {
        "<podcast>__<ep_id>": {
            "episode_id": "...",
            "podcast_id": "...",
            "title": "...",
            "published": "...",
            "episode_url": "...",
            "transcript": "..."
        }
    }
}
"""

import argparse
import json
import re
from pathlib import Path

from tickers import tickers  # full universe


# -------------------------------
# UTILS
# -------------------------------

def load_transcript(path: Path) -> str:
    """Load transcript text from a .txt file (UTF-8)."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def normalize_text(t: str) -> str:
    """
    Clean up transcript text for matching:
    - strip out raw URLs (http/https)
    - strip simple HTML tags
    - normalize whitespace
    """
    if not t:
        return ""

    # drop URLs like https://www.amazon.com/...
    t = re.sub(r"https?://\S+", " ", t)

    # strip basic HTML tags like <p>, <a href="...">, <strong>, etc.
    t = re.sub(r"<[^>]+>", " ", t)

    # collapse whitespace
    return " ".join(t.split())


# -------------------------------
# COMPANY MATCHING
# -------------------------------

# Very generic tokens that cause lots of false positives
_GENERIC_TOKENS = {
    "financial", "institution", "institutions", "group", "business",
    "bank", "banks", "capital", "services", "service", "company",
    "corp", "corporation", "inc", "inc.", "limited", "ltd", "holdings",
    "trust", "national", "international", "global", "life",
    "engineering", "first", "partners", "systems", "technologies"
}

# Common surname collisions seen in podcasts
_COMMON_SURNAME_TOKENS = {
    "johnson", "kim", "morgan", "dwayne", "kardashian"
}


def _clean_name(name: str) -> str:
    """Lowercase, remove punctuation, collapse spaces."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _acronym_from_name(name: str) -> str:
    """
    Build acronym from multi-word names, e.g.
    "International Business Machines" -> "ibm"
    Only returns acronyms length >= 3.
    """
    tokens = [t for t in _clean_name(name).split() if t]
    if len(tokens) < 2:
        return ""
    acronym = "".join(t[0] for t in tokens if t[0].isalpha())
    if len(acronym) >= 3:
        return acronym
    return ""


def build_company_regex(company_names, ticker=None):
    """
    Create a case-insensitive regex that matches:
    - Exact company name(s)
    - Cleaned variants (punctuation-stripped)
    - Acronyms/abbreviations derived from multi-word names
    - Optional ticker symbol (only if length >= 3 to avoid T/J noise)
    - A safe "core" word (first word) only when it is specific
    """
    parts = []

    # 1) Add raw + cleaned company names + acronyms
    for cname in company_names or []:
        if not cname:
            continue

        raw = cname.strip()
        if raw:
            s = re.escape(raw)
            s = s.replace(r"\ ", r"\s+")
            parts.append(s)

        cleaned = _clean_name(raw)
        if cleaned and cleaned != raw.lower():
            s2 = re.escape(cleaned)
            s2 = s2.replace(r"\ ", r"\s+")
            parts.append(s2)

        acr = _acronym_from_name(raw)
        if acr:
            parts.append(re.escape(acr))

    # 2) Add ticker ONLY if it is not a short/common ticker
    if ticker:
        tkr = ticker.strip()
        if len(tkr) >= 3:
            parts.append(re.escape(tkr))

    # 3) Add a safe "core" (first word) only if it's not generic/surname
    for cname in company_names or []:
        if not cname:
            continue
        core = _clean_name(cname).split()[0] if _clean_name(cname) else ""
        if (
            core
            and len(core) >= 4
            and core not in _GENERIC_TOKENS
            and core not in _COMMON_SURNAME_TOKENS
        ):
            parts.append(re.escape(core))

    # Deduplicate
    deduped = []
    seen = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    if not deduped:
        # Avoid invalid regex edge-case
        return r"^\b$"

    return r"\b(" + "|".join(deduped) + r")\b"


def extract_snippets(transcript: str, company_names, ticker=None, window=2):
    """
    Extract contextual sentence snippets around company mentions.
    """
    transcript = normalize_text(transcript)
    if not transcript:
        return []

    sent_pat = r'[^\.!?]+[\.!?]'
    sentences = re.findall(sent_pat, transcript)
    if not sentences:
        sentences = [transcript]

    comp_regex = re.compile(build_company_regex(company_names, ticker=ticker), re.IGNORECASE)

    hits = []
    for i, sent in enumerate(sentences):
        if comp_regex.search(sent):
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            block = " ".join(sentences[start:end]).strip()
            hits.append(block)

    return hits


# ----------------------------------------------------------
# NEW STRONG-ALIAS FILTER (added to reduce false positives)
# ----------------------------------------------------------

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
        # Skip names that are entirely generic words
        if all(t in _GENERIC_TOKENS for t in tokens):
            continue
        variants.append(" ".join(tokens))
    # Deduplicate
    return list(dict.fromkeys(variants))


def _snippet_has_specific_alias(snippet: str, company_names) -> bool:
    variants = _build_name_variants(company_names)
    if not variants:
        return False
    text = f" {_normalize_for_match(snippet)} "
    return any(f" {v} " in text for v in variants)


# -------------------------------
# MAIN PIPELINE
# -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/podcasts", help="Root folder of podcast transcripts")
    parser.add_argument("--out", type=str, default="data/podcast_excerpts_ui.json", help="Output JSON path")
    parser.add_argument("--tickers", nargs="*", help="Optional list of tickers to limit runs")
    parser.add_argument("--window", type=int, default=2, help="Sentence window around company mentions")
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)

    if args.tickers:
        universe = args.tickers
        print(f"[INFO] Using explicit tickers: {universe}")
    else:
        universe = list(tickers.keys())
        print("[INFO] No tickers provided; using full Cutler universe from tickers.py")

    # Output structure
    output_data = {t: [] for t in universe}
    output_data["_episodes"] = {}

    # Walk directories for episodes
    for pod_dir in root.iterdir():
        if not pod_dir.is_dir():
            continue

        podcast_id = pod_dir.name

        for ep_dir in pod_dir.iterdir():
            if not ep_dir.is_dir():
                continue

            meta_file = ep_dir / "metadata.json"
            txt_file = ep_dir / "transcript.txt"
            if not meta_file.exists() or not txt_file.exists():
                continue

            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            ep_id = meta.get("episode_id")
            title = meta.get("title", "")
            published = meta.get("published", "")
            episode_url = meta.get("episode_url", "")

            transcript = load_transcript(txt_file)
            if not transcript.strip():
                continue

            # Save full episode transcript for fallback summarization
            episode_uid = f"{podcast_id}__{ep_id}"
            output_data["_episodes"][episode_uid] = {
                "episode_id": ep_id,
                "podcast_id": podcast_id,
                "title": title,
                "published": published,
                "episode_url": episode_url,
                "transcript": transcript,
            }

            # Extract snippets per ticker
            for t in universe:
                names = tickers.get(t, [])
                if not names:
                    continue

                snippets = extract_snippets(transcript, names, ticker=t, window=args.window)
                if not snippets:
                    continue

                # Store each match with metadata
                for sn in snippets:

                    # ----------------------------------------------------
                    # NEW STRONG-ALIAS FILTER (exactly where needed)
                    # ----------------------------------------------------
                    if not _snippet_has_specific_alias(sn, names):
                        continue

                    output_data[t].append({
                        "ticker": t,
                        "company_names": names,
                        "snippet": sn,
                        "podcast_id": podcast_id,
                        "episode_id": ep_id,
                        "title": title,
                        "published": published,
                        "episode_url": episode_url,
                    })

    non_empty = sum(len(v) for k, v in output_data.items() if k != "_episodes")
    print(f"[INFO] Extracted {non_empty} total snippets")

    out_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"[OK] Saved episode data + snippets to {out_path}")


if __name__ == "__main__":
    main()

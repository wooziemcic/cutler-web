"""Evidence-first podcast report sections.

Builds the (heading, body) section list consumed by final._build_text_pdf, so
the report keeps the existing Cutler PDF styling rather than introducing a
second reporting system.

The ordering is deliberate. Verbatim transcript is the product; the analyst
should get most of the value from this document with the AI sections ignored
entirely. So each episode renders as:

    header (ticker, podcast, episode, date, guest, relevance)
    VERBATIM PODCAST EXCERPTS   <- the bulk of the page
    Why it matters / Key takeaways  <- short, optional, clearly secondary

AI never writes an excerpt. It only ever comments on text already selected
from the transcript, and every quotation is checked against the transcript
before it is rendered.
"""

from __future__ import annotations

import textwrap
from typing import Callable, Dict, List, Optional, Sequence

from podcast_evidence import Excerpt
from podcast_research import ProcessedEpisode

# Kept small on purpose: this is commentary, not the evidence.
_INSIGHT_SYSTEM = (
    "You are an equity research assistant. You are given verbatim excerpts from "
    "a podcast. Explain briefly why they matter to an investor. "
    "Never invent quotations and never restate the excerpt as if it were your "
    "own analysis. If the excerpts do not support a point, do not make it."
)


def _fmt_speakers(ex: Excerpt) -> str:
    if not ex.speakers:
        return ""
    return " / ".join(ex.speakers)


def _excerpt_block(ex: Excerpt, number: int) -> str:
    """Render one verbatim excerpt with whatever provenance we actually have."""
    head_bits = [f"[{number}] {ex.topic_label}"]
    if ex.time_range:
        head_bits.append(ex.time_range)
    header = "  -  ".join(head_bits)

    lines = [header]
    speakers = _fmt_speakers(ex)
    if speakers:
        lines.append(f"Speakers: {speakers}")
    lines.append("")

    # When the transcript carried speaker labels, keep the turn structure so a
    # question and its answer stay visibly distinct.
    labelled = [t for t in ex.turns if t.speaker]
    if labelled and len(labelled) >= 2:
        for turn in ex.turns:
            if not turn.text.strip():
                continue
            prefix = f"{turn.speaker}: " if turn.speaker else ""
            stamp = f"[{turn.timestamp}] " if turn.timestamp else ""
            lines.append(textwrap.fill(f"{stamp}{prefix}{turn.text.strip()}", width=100))
            lines.append("")
    else:
        lines.append(textwrap.fill(ex.text.strip(), width=100))
        lines.append("")

    lines.append(f"({ex.words} words verbatim from transcript)")
    return "\n".join(lines)


def build_ai_commentary(
    excerpts: Sequence[Excerpt],
    *,
    ticker: str,
    episode_title: str,
    model: str = "gpt-4o-mini",
    chat_fn: Optional[Callable] = None,
) -> tuple[str, str, str]:
    """Return (why_it_matters, key_takeaways, error).

    error is "" on success and otherwise a short reason the section could
    not be produced - the caller shows this instead of silently omitting
    the section, since "AI interpretation" checked on with nothing rendered
    and no explanation looks like a missing feature, not a skipped one.

    Deliberately fed only the already-selected excerpts. The model cannot
    reach the full transcript, so it cannot surface a "quote" nobody selected.
    """
    if not excerpts:
        return "", "", "no excerpts to comment on"
    if chat_fn is None:
        try:
            from openai_legacy import chat_completion_text as chat_fn  # type: ignore
        except Exception as exc:
            return "", "", f"openai_legacy unavailable: {type(exc).__name__}: {exc}"

    body = "\n\n".join(f"[{i}] {e.topic_label}\n{e.text}" for i, e in enumerate(excerpts, 1))
    prompt = (
        f"Ticker: {ticker}\nEpisode: {episode_title}\n\n"
        f"Verbatim excerpts:\n{body}\n\n"
        "Reply in exactly this form:\n"
        "WHY: <one or two sentences on why this matters to an investor>\n"
        "TAKEAWAYS:\n- <short point>\n- <short point>\n- <short point>"
    )
    try:
        text, err = chat_fn(
            messages=[
                {"role": "system", "content": _INSIGHT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.1,
            max_tokens=400,
        )
        if err:
            return "", "", err
        if not text:
            return "", "", "empty_response"
    except Exception as exc:
        return "", "", f"{type(exc).__name__}: {exc}"

    why, takeaways = "", ""
    if "TAKEAWAYS:" in text:
        head, _, tail = text.partition("TAKEAWAYS:")
        why = head.replace("WHY:", "").strip()
        takeaways = tail.strip()
    else:
        why = text.replace("WHY:", "").strip()
    return why, takeaways, ""


def build_sections(
    processed: Sequence[ProcessedEpisode],
    *,
    company_names: Optional[Dict[str, str]] = None,
    include_ai: bool = True,
    model: str = "gpt-4o-mini",
    chat_fn: Optional[Callable] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> List[tuple[str, str]]:
    """Group reportable episodes by ticker and render evidence-first sections."""
    company_names = company_names or {}
    by_ticker: Dict[str, List[ProcessedEpisode]] = {}
    for pe in processed:
        if pe.reportable:
            by_ticker.setdefault(pe.candidate.ticker, []).append(pe)

    sections: List[tuple[str, str]] = []
    if not by_ticker:
        sections.append(
            (
                "No qualifying episodes",
                "No episode in this window contained a materially relevant discussion.\n"
                "Passing mentions and irrelevant episodes are deliberately excluded.",
            )
        )
        return sections

    total_excerpts = sum(len(pe.excerpts) for eps in by_ticker.values() for pe in eps)
    sections.append(
        (
            "Summary",
            f"Tickers with qualifying discussion: {len(by_ticker)}\n"
            f"Episodes: {sum(len(v) for v in by_ticker.values())}\n"
            f"Verbatim excerpts: {total_excerpts}\n\n"
            "Every excerpt below is transcript text, reproduced word for word. "
            "AI commentary is secondary and clearly separated.",
        )
    )

    for ticker in sorted(by_ticker):
        episodes = sorted(by_ticker[ticker], key=lambda p: p.candidate.published, reverse=True)
        label = company_names.get(ticker) or ticker
        heading_ticker = ticker if label == ticker else f"{ticker} - {label}"

        for pe in episodes:
            c = pe.candidate
            if on_progress:
                on_progress(f"{ticker}: {c.title[:60]}")

            meta = [
                f"Podcast:   {c.podcast_name or 'Unknown'}",
                f"Episode:   {c.title}",
                f"Date:      {(c.published or '')[:10]}",
                f"Relevance: {pe.relevance.label} ({pe.relevance.reason})",
                f"Transcript source: {pe.transcript_source}",
            ]
            if pe.relevance.people_present:
                meta.insert(3, f"Guest:     {', '.join(pe.relevance.people_present)}")
            if c.matched:
                meta.append(f"Matched on: {c.why()}")

            body = ["\n".join(meta), "", "=" * 72, "VERBATIM PODCAST EXCERPTS", "=" * 72, ""]
            for i, ex in enumerate(pe.excerpts, 1):
                body.append(_excerpt_block(ex, i))
                body.append("")

            if include_ai:
                why, takeaways, ai_err = build_ai_commentary(
                    pe.excerpts,
                    ticker=ticker,
                    episode_title=c.title,
                    model=model,
                    chat_fn=chat_fn,
                )
                body.append("-" * 72)
                body.append("AI INTERPRETATION (secondary - the excerpts above are the evidence)")
                body.append("-" * 72)
                if why or takeaways:
                    if why:
                        body.append(f"Why it matters: {why}")
                    if takeaways:
                        body.append("")
                        body.append(f"Key takeaways:\n{takeaways}")
                else:
                    body.append(f"Not generated: {ai_err or 'unknown error'}")

            sections.append((f"{heading_ticker}  -  {c.podcast_name}", "\n".join(body)))

    return sections

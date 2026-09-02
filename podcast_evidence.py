"""Transcript verification and verbatim excerpt extraction.

This is the heart of the podcast layer. An analyst should be able to read this
output instead of listening to the episode, so two rules hold throughout:

  1. Quotations are sliced out of the transcript and never rewritten. No LLM
     touches the text of an excerpt. AI may later say *why* a passage matters,
     but the passage itself is always the speaker's own words.
  2. Nothing is invented. Speaker names and timestamps are emitted only when
     the transcript actually carries them.

Two jobs live here:

  classify_relevance()  - decide, from the transcript rather than the title or
                          description, whether an episode is genuinely about
                          the company.
  extract_excerpts()    - pull the meaningful stretches of discussion, sized to
                          the discussion itself rather than a fixed character
                          window, and label them by topic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Relevance classes, strongest first
# ---------------------------------------------------------------------------

EXEC_INTERVIEW = "Executive Interview"
DIRECT_DISCUSSION = "Direct Company Discussion"
MATERIAL_ANALYSIS = "Material Company Analysis"
MATERIAL_INDUSTRY = "Material Industry Discussion"
PASSING_MENTION = "Passing Mention"
IRRELEVANT = "Irrelevant"

# Only these reach the research PDF.
REPORTABLE = {EXEC_INTERVIEW, DIRECT_DISCUSSION, MATERIAL_ANALYSIS, MATERIAL_INDUSTRY}

# ---------------------------------------------------------------------------
# Topic taxonomy - deterministic keyword matching, so labelling never rewrites
# or hallucinates the quote it labels.
# ---------------------------------------------------------------------------

TOPIC_KEYWORDS: Dict[str, Sequence[str]] = {
    "M&A / Acquisitions": (
        "acquisition", "acquire", "acquired", "acquiring", "merger", "m&a",
        "takeover", "roll-up", "rollup", "roll up", "consolidat", "bolt-on",
        "tuck-in", "deal closed", "bid for", "poison pill", "divest",
    ),
    "Capital Allocation": (
        "capital allocation", "allocate capital", "buyback", "repurchase",
        "dividend", "balance sheet", "leverage", "debt", "equity raise",
        "dilution", "return on capital", "roic", "free cash flow",
    ),
    "Margins & Profitability": (
        "margin", "ebitda", "operating income", "profitability", "cost synerg",
        "gross profit", "opex", "unit economics", "pricing power",
    ),
    "Revenue & Growth": (
        "revenue", "top line", "topline", "organic growth", "same store",
        "grew", "growth rate", "cagr", "run rate", "bookings",
    ),
    "Guidance & Targets": (
        "guidance", "outlook", "target", "forecast", "we expect", "by 2030",
        "long-term target", "five year", "ten year", "billion in revenue",
    ),
    "Demand & End Markets": (
        "demand", "end market", "backlog", "order", "volume", "customer",
        "housing", "construction", "repair and remodel",
    ),
    "Competition & Moat": (
        "competit", "moat", "market share", "barrier to entry", "rival",
        "differentiat", "fragmented", "scale advantage",
    ),
    "Industry Conditions": (
        "industry", "sector", "cycle", "commoditi", "supply chain",
        "consolidation", "distribution", "channel",
    ),
    "Risks": (
        "risk", "headwind", "concern", "downside", "bear case", "what could go wrong",
        "threat", "vulnerab", "recession",
    ),
    "Management & Philosophy": (
        "philosophy", "track record", "culture", "incentive", "management team",
        "how i think about", "my approach", "leadership", "hire",
    ),
    "Valuation": (
        "valuation", "multiple", "intrinsic value", "fair value", "price target",
        "trading at", "discount", "expensive", "cheap",
    ),
    "Technology & Operations": (
        "technology", "software", "ai ", "automation", "digital", "platform",
        "logistics", "efficienc", "operational",
    ),
}

# Words suggesting substantive investor discussion rather than chit-chat.
_SUBSTANCE_HINTS = tuple(
    kw for kws in TOPIC_KEYWORDS.values() for kw in kws
)

_AD_MARKERS = (
    "this episode is brought to you by", "sponsored by", "promo code",
    "use code", "visit our sponsor", "start your free trial",
)

# ---------------------------------------------------------------------------
# Transcript structure
# ---------------------------------------------------------------------------

# "Brad Jacobs:" / "BRAD JACOBS:" / "Interviewer:" at the start of a line.
_SPEAKER_RE = re.compile(
    r"^\s*(?P<speaker>[A-Z][A-Za-z.\-']{1,24}(?:\s+[A-Z][A-Za-z.\-']{1,24}){0,3})\s*:\s*(?P<text>\S.*)$"
)
# "[00:12:14]" / "(12:14)" / "00:12:14" at a line start.
_TIMESTAMP_RE = re.compile(
    r"[\[\(]?\s*(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\s*[\]\)]?"
)


@dataclass
class Turn:
    """One contiguous chunk of transcript, ideally one speaker's turn."""

    text: str
    speaker: str = ""       # "" when the transcript does not say
    timestamp: str = ""     # "" when the transcript does not say
    index: int = 0

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Excerpt:
    """A verbatim stretch of transcript, plus what it is about."""

    text: str                                   # never paraphrased
    turns: List[Turn] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    speakers: List[str] = field(default_factory=list)
    mention_count: int = 0
    score: float = 0.0

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def time_range(self) -> str:
        if self.start_time and self.end_time and self.start_time != self.end_time:
            return f"{self.start_time} - {self.end_time}"
        return self.start_time or ""

    @property
    def topic_label(self) -> str:
        return ", ".join(self.topics) if self.topics else "General Discussion"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["time_range"] = self.time_range
        d["topic_label"] = self.topic_label
        d["words"] = self.words
        return d


def parse_transcript(transcript: str) -> List[Turn]:
    """Split a transcript into turns, keeping speakers/timestamps if present.

    Handles three shapes seen from the transcript sources: speaker-labelled
    lines, timestamped lines, and one undifferentiated blob (what Whisper
    returns). For the blob case the text is chunked on sentence boundaries so
    downstream code has something to work with, with no speaker attributed -
    guessing a speaker would be inventing evidence.
    """
    text = (transcript or "").strip()
    if not text:
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    turns: List[Turn] = []
    structured = False

    for ln in lines:
        ts = ""
        body = ln
        m_ts = _TIMESTAMP_RE.match(body)
        if m_ts and m_ts.start() == 0:
            ts = m_ts.group("ts")
            body = body[m_ts.end():].strip(" -–—")
            structured = True

        speaker = ""
        m_sp = _SPEAKER_RE.match(body)
        if m_sp:
            speaker = m_sp.group("speaker").strip()
            body = m_sp.group("text").strip()
            structured = True

        if not body:
            continue
        if turns and not speaker and not ts and not structured:
            turns[-1].text += " " + body
        else:
            turns.append(Turn(text=body, speaker=speaker, timestamp=ts, index=len(turns)))

    if structured:
        for i, t in enumerate(turns):
            t.index = i
        return turns

    # Unstructured: chunk into paragraph-sized groups of sentences so excerpts
    # can still be sized to the discussion.
    sentences = re.findall(r"[^.!?]+[.!?]", text) or [text]
    chunks: List[Turn] = []
    buf: List[str] = []
    for sent in sentences:
        buf.append(sent.strip())
        if sum(len(s.split()) for s in buf) >= 60:  # ~25 seconds of speech
            chunks.append(Turn(text=" ".join(buf), index=len(chunks)))
            buf = []
    if buf:
        chunks.append(Turn(text=" ".join(buf), index=len(chunks)))
    return chunks


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def build_mention_regex(names: Sequence[str], ticker: str = "", include_ticker: bool = True) -> re.Pattern:
    """Whole-word regex over company names, aliases and (optionally) the ticker."""
    parts: List[str] = []
    for n in names:
        n = str(n or "").strip()
        if n:
            parts.append(re.escape(n))
    if include_ticker and ticker:
        parts.append(re.escape(ticker))
    if not parts:
        return re.compile(r"(?!x)x")  # matches nothing
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


def detect_topics(text: str, max_topics: int = 3) -> List[str]:
    """Label a passage from its own words. Deterministic - no model involved."""
    low = (text or "").lower()
    scored: List[Tuple[int, str]] = []
    for topic, kws in TOPIC_KEYWORDS.items():
        hits = sum(low.count(kw) for kw in kws)
        if hits:
            scored.append((hits, topic))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, t in scored[:max_topics]]


def _is_ad(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _AD_MARKERS)


# ---------------------------------------------------------------------------
# Relevance classification, from the transcript
# ---------------------------------------------------------------------------

@dataclass
class Relevance:
    label: str
    confidence: float
    reason: str
    mention_count: int = 0
    people_present: List[str] = field(default_factory=list)

    @property
    def reportable(self) -> bool:
        return self.label in REPORTABLE


def classify_relevance(
    transcript: str,
    *,
    ticker: str,
    company_names: Sequence[str],
    people: Sequence[str] = (),
    title: str = "",
    ambiguous_ticker: bool = False,
) -> Relevance:
    """Decide relevance from what was actually said.

    Metadata only ever nominates a candidate. An episode is not relevant just
    because an executive is in the title, the ticker appears once, or the
    company shows up in the description - the transcript has to carry a real
    discussion.
    """
    text = (transcript or "").strip()
    if not text:
        return Relevance(IRRELEVANT, 0.0, "no transcript")

    words = max(1, len(text.split()))
    # For an ambiguous symbol the bare ticker is not evidence of anything.
    rx = build_mention_regex(company_names, ticker, include_ticker=not ambiguous_ticker)
    mentions = len(rx.findall(text))

    person_rx = [(p, re.compile(rf"(?<!\w){re.escape(p)}(?!\w)", re.IGNORECASE)) for p in people if p]
    present = [p for p, r in person_rx if r.search(text)]

    if mentions == 0 and not present:
        return Relevance(IRRELEVANT, 0.85, "company never mentioned in transcript", 0, [])

    # Mentions per 1,000 words: distinguishes a sustained discussion from a
    # single name-drop in a long episode.
    density = mentions / (words / 1000.0)
    substance = sum(1 for kw in _SUBSTANCE_HINTS if kw in text.lower())

    # An executive we track is speaking and the company comes up repeatedly.
    if present and mentions >= 3:
        return Relevance(
            EXEC_INTERVIEW,
            0.9,
            f"{', '.join(present)} present with {mentions} company mentions",
            mentions,
            present,
        )

    if mentions >= 20 or (density >= 3.0 and mentions >= 8):
        return Relevance(
            DIRECT_DISCUSSION,
            0.85,
            f"{mentions} mentions ({density:.1f}/1k words) - sustained discussion",
            mentions,
            present,
        )

    if mentions >= 6 and substance >= 8:
        return Relevance(
            MATERIAL_ANALYSIS,
            0.7,
            f"{mentions} mentions alongside substantive investment discussion",
            mentions,
            present,
        )

    if mentions >= 3 and substance >= 12:
        return Relevance(
            MATERIAL_INDUSTRY,
            0.55,
            f"{mentions} mentions within a broader industry discussion",
            mentions,
            present,
        )

    return Relevance(
        PASSING_MENTION,
        0.8,
        f"only {mentions} mention(s) in {words} words - not a real discussion",
        mentions,
        present,
    )


# ---------------------------------------------------------------------------
# Verbatim excerpt extraction
# ---------------------------------------------------------------------------

def extract_excerpts(
    transcript: str,
    *,
    ticker: str,
    company_names: Sequence[str],
    people: Sequence[str] = (),
    ambiguous_ticker: bool = False,
    min_words: int = 60,
    max_words: int = 900,
    max_excerpts: int = 6,
    context_turns: int = 1,
) -> List[Excerpt]:
    """Pull the meaningful stretches of discussion, verbatim.

    Not "find the ticker and take N characters". Mentions are grouped into
    runs, each run is grown outward across whole turns until the discussion
    stops, and the result is whatever length that discussion actually was -
    a short exchange or several minutes of answer.

    When the transcript has speaker labels, a preceding question turn is kept
    so an answer is not stranded without the question that prompted it.
    """
    turns = parse_transcript(transcript)
    if not turns:
        return []

    rx = build_mention_regex(company_names, ticker, include_ticker=not ambiguous_ticker)
    person_rx = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(p) for p in people if p) + r")(?!\w)",
        re.IGNORECASE,
    ) if any(people) else None

    hit_idx = [i for i, t in enumerate(turns) if rx.search(t.text)]
    if not hit_idx:
        return []

    # Group hits that sit close together: one continuous discussion, not one
    # excerpt per sentence.
    groups: List[List[int]] = [[hit_idx[0], hit_idx[0]]]
    for i in hit_idx[1:]:
        if i - groups[-1][1] <= context_turns + 1:
            groups[-1][1] = i
        else:
            groups.append([i, i])

    excerpts: List[Excerpt] = []
    prev_end = -1
    for first, last in groups:
        start = max(0, first - context_turns, prev_end + 1)
        end = min(len(turns) - 1, last + context_turns)
        if start > end:
            continue

        # Grow backwards onto a question so an answer keeps its prompt.
        if start > 0 and turns[start].speaker:
            prior = turns[start - 1]
            if prior.speaker and prior.speaker != turns[start].speaker and prior.words <= 120:
                if "?" in prior.text or prior.words <= 60:
                    start = max(prev_end + 1, start - 1)

        # Grow forwards while the same speaker keeps talking - a long answer
        # should not be truncated mid-thought.
        while (
            end + 1 < len(turns)
            and turns[end].speaker
            and turns[end + 1].speaker == turns[end].speaker
            and sum(t.words for t in turns[start : end + 2]) <= max_words
        ):
            end += 1

        # Trailing context is useful when it continues the same thought, but a
        # fresh question that never mentions the company starts a new topic -
        # keeping it drags unrelated material into the evidence.
        while end > last:
            tail = turns[end]
            if rx.search(tail.text):
                break
            # A different speaker with nothing to say about the company is the
            # interviewer moving on, so the discussion has ended here.
            if tail.speaker and tail.speaker != turns[last].speaker:
                end -= 1
                continue
            break

        block = turns[start : end + 1]
        text = " ".join(t.text.strip() for t in block).strip()
        text = re.sub(r"\s+", " ", text)
        if not text or _is_ad(text):
            continue

        # Trim over-long passages back to whole turns around the mentions.
        while len(text.split()) > max_words and end > last:
            end -= 1
            block = turns[start : end + 1]
            text = re.sub(r"\s+", " ", " ".join(t.text.strip() for t in block)).strip()
        while len(text.split()) > max_words and start < first:
            start += 1
            block = turns[start : end + 1]
            text = re.sub(r"\s+", " ", " ".join(t.text.strip() for t in block)).strip()

        mention_count = len(rx.findall(text))
        if len(text.split()) < min_words and mention_count < 2:
            continue

        speakers = []
        for t in block:
            if t.speaker and t.speaker not in speakers:
                speakers.append(t.speaker)
        times = [t.timestamp for t in block if t.timestamp]

        score = mention_count * 2.0 + len(detect_topics(text)) * 1.5
        if person_rx and person_rx.search(text):
            score += 4.0  # the executive speaking about the company

        excerpts.append(
            Excerpt(
                text=text,
                turns=list(block),
                topics=detect_topics(text),
                start_time=times[0] if times else "",
                end_time=times[-1] if times else "",
                speakers=speakers,
                mention_count=mention_count,
                score=score,
            )
        )
        prev_end = end

    excerpts.sort(key=lambda e: -e.score)
    return excerpts[:max_excerpts]


def verify_verbatim(excerpt: "Excerpt | str", transcript: str) -> bool:
    """Confirm an excerpt really is transcript text, not something rewritten.

    A guard against any future change that lets a model near the quotations.
    Speaker labels are lifted out of the line into Turn.speaker, so the joined
    excerpt is not always a literal substring of the raw transcript. Each turn
    is therefore checked on its own; that is the text actually quoted.
    """
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip().lower()

    hay = norm(transcript)
    if isinstance(excerpt, Excerpt) and excerpt.turns:
        return all(norm(t.text) in hay for t in excerpt.turns if t.text.strip())
    text = excerpt.text if isinstance(excerpt, Excerpt) else str(excerpt)
    return norm(text) in hay


if __name__ == "__main__":  # manual check
    demo = (
        "Interviewer: Brad, tell us about the acquisition strategy at QXO.\n"
        "Brad Jacobs: We think about capital allocation first. QXO is buying "
        "distribution assets where we can take out cost and cross-sell. The "
        "TopBuild deal takes us to eighteen billion in pro forma revenue and "
        "the margin opportunity is substantial because procurement scale is a "
        "real weapon against smaller rivals.\n"
        "Interviewer: And the risks?\n"
        "Brad Jacobs: The industry is commoditised, so competition is fierce "
        "and we have to defend margins every day at QXO.\n"
        "Interviewer: Let us talk about the weather now.\n"
        "Guest: It has been mild.\n"
    )
    rel = classify_relevance(demo, ticker="QXO", company_names=["QXO"], people=["Brad Jacobs"])
    print("relevance:", rel.label, rel.confidence, "-", rel.reason)
    for i, ex in enumerate(extract_excerpts(demo, ticker="QXO", company_names=["QXO"], people=["Brad Jacobs"], min_words=10), 1):
        print(f"\n[{i}] {ex.topic_label} | speakers={ex.speakers} | {ex.words} words")
        print("   ", ex.text[:200])
        print("    verbatim:", verify_verbatim(ex, demo))

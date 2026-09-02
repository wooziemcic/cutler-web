"""Ticker -> searchable entities (company, aliases, executives).

Podcast discovery needs more than a ticker symbol. An episode titled
"Brad Jacobs: Question Deeply, Act Quickly" is highly relevant to QXO and
never mentions the ticker, so the search side has to know that Brad Jacobs is
QXO's chairman and CEO.

Sources, in order:
  1. podcast_entities.csv  - hand-maintained aliases and people. Deliberately a
     flat CSV so it can be edited by hand now and swapped for a Google Sheet
     later without touching this module (see load_entity_overrides).
  2. live_tickers          - the live Cutler universe supplies company names.
  3. the ticker itself     - always available as a last resort.

Missing override data is normal and not an error: a ticker with no CSV row
still gets its company name from the sheet and remains fully searchable.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
ENTITIES_CSV = HERE / "podcast_entities.csv"

# Tickers that are also ordinary English words. Searching podcast metadata for
# "CAT" or "ON" returns overwhelming noise, so for these the bare symbol is
# never used as a search term and never counts as evidence on its own - the
# company and people names carry the query instead.
_AMBIGUOUS_WORDS = {
    "a", "all", "an", "and", "any", "are", "arm", "as", "at", "be", "best",
    "big", "box", "boy", "by", "can", "car", "cat", "ceo", "co", "cost", "cow",
    "cup", "cut", "dad", "day", "dd", "do", "dog", "eat", "edit", "end", "ever",
    "eye", "fast", "fix", "flow", "for", "form", "fun", "gap", "get", "go",
    "good", "gps", "grow", "has", "hat", "he", "him", "his", "hit", "hope",
    "hot", "how", "hug", "ice", "in", "info", "is", "it", "job", "joy", "key",
    "kid", "know", "lab", "land", "law", "lead", "life", "like", "lion", "live",
    "load", "log", "look", "love", "low", "man", "map", "mass", "max", "me",
    "mind", "mode", "mom", "moon", "most", "move", "nav", "net", "new", "next",
    "nice", "no", "now", "of", "off", "oil", "ok", "old", "on", "one", "open",
    "or", "our", "out", "own", "pay", "peak", "pet", "plan", "play", "plus",
    "post", "power", "pro", "pure", "push", "rate", "read", "real", "red",
    "rise", "road", "rock", "run", "safe", "see", "self", "sell", "she", "ship",
    "shop", "show", "site", "so", "soft", "sold", "some", "star", "stay",
    "step", "stop", "sun", "take", "talk", "team", "tech", "tell", "ten",
    "the", "them", "then", "they", "this", "time", "tip", "to", "top", "tour",
    "town", "true", "try", "two", "up", "us", "use", "very", "view", "vote",
    "walk", "wall", "want", "war", "was", "wave", "way", "we", "well", "west",
    "what", "when", "who", "why", "will", "win", "wise", "with", "work",
    "world", "yes", "you", "your",
}

# Suffixes stripped when building a loose company alias, so "QXO, Inc." also
# matches a plain "QXO" and "Becton Dickinson and Company" matches the short form.
_CORP_SUFFIX_RE = re.compile(
    r"[,\s]+(inc|inc\.|incorporated|corp|corp\.|corporation|co|co\.|company|"
    r"plc|ltd|ltd\.|limited|llc|lp|l\.p\.|nv|n\.v\.|sa|s\.a\.|ag|holdings?|"
    r"group|the)\b\.?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Entity:
    """One searchable name attached to a ticker."""

    name: str
    kind: str  # "ticker" | "company" | "alias" | "person"
    role: str = ""  # e.g. "Chairman & CEO"; blank for non-people
    weight: float = 1.0  # how much a metadata match on this name counts

    @property
    def is_person(self) -> bool:
        return self.kind == "person"


@dataclass
class TickerProfile:
    """Everything the discovery layer knows about one ticker."""

    ticker: str
    company: str = ""
    aliases: List[str] = field(default_factory=list)
    people: List[Entity] = field(default_factory=list)
    ambiguous: bool = False

    def entities(self) -> List[Entity]:
        """Every searchable name, strongest signal first."""
        out: List[Entity] = []
        for person in self.people:
            out.append(person)
        if self.company:
            out.append(Entity(self.company, "company", weight=3.0))
        for alias in self.aliases:
            out.append(Entity(alias, "alias", weight=2.5))
        # A distinctive symbol is a genuine signal; an ambiguous one is not, so
        # it is kept for display but weighted to zero.
        out.append(
            Entity(self.ticker, "ticker", weight=0.0 if self.ambiguous else 1.5)
        )
        return out

    def search_terms(self) -> List[str]:
        """Names worth spending a metadata search call on, best first.

        Ambiguous symbols are excluded: querying "ON" or "IT" burns quota and
        returns noise. Their company and people names still drive discovery.
        """
        terms: List[str] = []
        seen: set = set()
        for ent in self.entities():
            if ent.weight <= 0:
                continue
            key = ent.name.strip().lower()
            if not key or key in seen or len(key) < 3:
                continue
            seen.add(key)
            terms.append(ent.name.strip())
        return terms

    def describe(self) -> str:
        """One line for the UI so the analyst sees what is being searched."""
        bits = [self.ticker]
        if self.company and self.company.upper() != self.ticker:
            bits.append(self.company)
        bits.extend(self.aliases)
        for p in self.people:
            bits.append(f"{p.name}{f' ({p.role})' if p.role else ''}")
        line = " · ".join(bits)
        return line + ("  [ambiguous symbol]" if self.ambiguous else "")


def is_ambiguous_ticker(ticker: str) -> bool:
    """True when the bare symbol is too common to search on."""
    sym = (ticker or "").strip()
    if not sym:
        return True
    if len(sym) <= 2:  # A, ON, IT, GE... too short to disambiguate in prose
        return True
    return sym.lower() in _AMBIGUOUS_WORDS


def strip_corp_suffix(name: str) -> str:
    """'QXO, Inc.' -> 'QXO'. Returns the input when nothing is stripped."""
    prev = (name or "").strip()
    for _ in range(3):  # "Holdings Inc." style stacked suffixes
        nxt = _CORP_SUFFIX_RE.sub("", prev).strip(" ,.")
        if nxt == prev:
            break
        prev = nxt
    return prev or (name or "").strip()


def load_entity_overrides(csv_path: Path = ENTITIES_CSV) -> Dict[str, dict]:
    """Read podcast_entities.csv into {TICKER: {aliases, people}}.

    Columns: ticker, company, aliases, people
      aliases - pipe separated, e.g. "QXO Inc|Beacon Roofing"
      people  - pipe separated "Name:Role", e.g. "Brad Jacobs:Chairman & CEO"

    A missing or malformed file is not fatal; discovery falls back to the
    company names in the live ticker sheet.
    """
    out: Dict[str, dict] = {}
    try:
        if not csv_path.exists():
            return out
        with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                sym = str(row.get("ticker") or "").strip().upper()
                if not sym:
                    continue
                aliases = [
                    a.strip()
                    for a in str(row.get("aliases") or "").split("|")
                    if a.strip()
                ]
                people: List[Entity] = []
                for chunk in str(row.get("people") or "").split("|"):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    name, _, role = chunk.partition(":")
                    name = name.strip()
                    if not name:
                        continue
                    # A named executive is the strongest discovery signal there
                    # is: an interview rarely puts the ticker in the title.
                    people.append(
                        Entity(name, "person", role=role.strip(), weight=4.0)
                    )
                out[sym] = {
                    "company": str(row.get("company") or "").strip(),
                    "aliases": aliases,
                    "people": people,
                }
    except Exception:
        return out
    return out


def build_profile(
    ticker: str,
    universe_names: Optional[Iterable[str]] = None,
    overrides: Optional[Dict[str, dict]] = None,
) -> TickerProfile:
    """Assemble one ticker's profile from the sheet plus any CSV overrides."""
    sym = (ticker or "").strip().upper()
    ov = (overrides or {}).get(sym, {})

    names = [str(n).strip() for n in (universe_names or []) if str(n).strip()]
    # live_tickers echoes the symbol back when the sheet has no company column.
    names = [n for n in names if n.upper() != sym]

    company = ov.get("company") or (names[0] if names else "")

    aliases: List[str] = []
    seen = {company.lower()} if company else set()
    for cand in list(ov.get("aliases") or []) + names[1:]:
        low = cand.lower()
        if low and low not in seen:
            seen.add(low)
            aliases.append(cand)
    # "QXO, Inc." and "QXO" should both match.
    if company:
        short = strip_corp_suffix(company)
        if short and short.lower() not in seen and short.upper() != sym:
            aliases.append(short)

    return TickerProfile(
        ticker=sym,
        company=company,
        aliases=aliases,
        people=list(ov.get("people") or []),
        ambiguous=is_ambiguous_ticker(sym),
    )


def build_profiles(
    tickers: Optional[Iterable[str]] = None,
    csv_path: Path = ENTITIES_CSV,
) -> Dict[str, TickerProfile]:
    """Profiles for the given tickers, or the whole live universe."""
    overrides = load_entity_overrides(csv_path)

    universe: Dict[str, List[str]] = {}
    try:
        from live_tickers import get_ticker_universe  # type: ignore

        universe = get_ticker_universe() or {}
    except Exception:
        universe = {}

    if tickers:
        wanted = [str(t).strip().upper() for t in tickers if str(t).strip()]
    else:
        wanted = sorted(set(universe) | set(overrides))

    return {
        sym: build_profile(sym, universe.get(sym), overrides) for sym in wanted
    }


if __name__ == "__main__":  # quick manual check
    import sys

    syms = sys.argv[1:] or ["QXO", "CAT", "ON", "IT", "AMZN"]
    for sym, prof in build_profiles(syms).items():
        print(f"{sym}: {prof.describe()}")
        print(f"    search terms: {prof.search_terms()}")

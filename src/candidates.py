"""Speaker-candidate enumeration.

Attribution is framed here as ranking, not generation: for each quotation the
system picks a speaker out of a candidate list. That makes the candidate list a
hard ceiling -- if the gold speaker is not in it, no ranker can ever be right --
so this module's real output is not the candidates but the *oracle recall* they
imply, reported per quote type.

Candidates come from character mentions found in the narration around a quote.
Mentions are located by matching the novel's own alias gazetteer rather than by
running NER. That is a deliberate choice, and it is the generous one: PDNC's
aliases are gold, so a spaCy pipeline could only lose mentions relative to
this. Any ceiling reported here is therefore an upper bound on what an
end-to-end system with automatic NER would reach, which is the honest direction
for a ceiling to err in.

The gold gazetteer is nonetheless incomplete, and ``derive_forms`` repairs it
by rule (first/last name, kept only when unambiguous). That is worth 5.3 points
of ceiling on train -- 86.1% to 91.4% -- for 0.4 extra candidates per quote,
and it matters most on exactly the quotes the project is about: explicit rises
92.3 -> 96.1, implicit 85.2 -> 91.6.

What remains out of reach:

* First-person narrators. A narrator speaks constantly and is addressed rather
  than named, so his quotes rely on some other character saying his name
  nearby. Derived first names help more than expected here -- The Sun Also
  Rises goes from 46% to 85% once "Jake" is matchable -- but a gap survives
  (86.6% first-person vs 92.5% third at the configured window), and it is
  reported separately rather than hidden in the pooled number.

* Implicit quotes in long exchanges. Once two characters are established, an
  author can alternate for a page without naming either. Widening the window is
  the only lever the candidate stage has, and it is a real trade:
  ``CANDIDATE_WINDOW_SWEEP`` prices it at roughly +2pt recall per +1.3
  candidates per quote, which the ranker then has to sort out.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

from config import (
    CANDIDATE_CHARS_AFTER,
    CANDIDATE_CHARS_BEFORE,
    HONORIFICS,
    SPEECH_VERBS,
)
from pdnc import Character, Novel, Quote

logger = logging.getLogger(__name__)

_SPEECH_VERB_RE = re.compile(
    r"\b(?:" + "|".join(sorted(SPEECH_VERBS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- mentions


@dataclass(frozen=True)
class Mention:
    """One surface form in the narration that names a character."""

    character: Character
    surface: str
    start: int                 # absolute char offset into the novel text
    end: int
    zone: str                  # "before" | "inner" | "after"
    distance: int              # chars from the nearest edge of the quote
    near_speech_verb: bool     # a speech verb within 40 chars


@dataclass
class Candidate:
    """One character the ranker may choose, with the evidence for it."""

    character: Character
    mentions: list[Mention]
    is_gold: bool = False

    @property
    def nearest(self) -> Mention:
        return min(self.mentions, key=lambda m: m.distance)

    @property
    def n_mentions(self) -> int:
        return len(self.mentions)

    @property
    def has_tagged_mention(self) -> bool:
        """A mention sitting next to a speech verb -- the explicit-tag cue."""
        return any(m.near_speech_verb for m in self.mentions)


_WS = re.compile(r"\s+")
_STOP_TOKENS = {"the", "a", "an", "of", "in", "and", "who", "with"}


def derive_forms(
    characters: list[Character], honorifics: set[str]
) -> dict[str, Character]:
    """Invent the short names PDNC's alias lists leave out.

    PDNC's gazetteer is gold but not exhaustive. "Bill Gorton" in The Sun Also
    Rises carries exactly one alias -- his full name -- while Hemingway writes
    "Bill said" every time, so 156 of his explicitly tagged quotations have no
    matching mention anywhere in the window. That is a gazetteer gap, not a
    modelling problem, and no ranker can recover from it.

    The fix is a uniform rule, applied identically to every novel: from any
    two- or three-token personal name, propose its first token and its last
    token as additional surface forms. "Bill Gorton" yields "Bill" and
    "Gorton"; "Mike Campbell" yields "Mike".

    A proposed form is kept only if it is unambiguous within the novel -- it
    must not already name another character, nor be proposed for two of them.
    Mansfield Park has Tom, Edmund, Lady and Sir Thomas Bertram, so "Bertram"
    is proposed four times, dropped, and Tom's 13 misses stand. That is the
    intended behaviour: the rule buys back real coverage without ever guessing
    between two people, and where it declines to guess the ceiling honestly
    stays down.
    """
    gold = {a for c in characters for a in c.aliases}
    proposals: dict[str, set[int]] = {}
    owner: dict[int, Character] = {c.cid: c for c in characters}

    for c in characters:
        for alias in c.aliases:
            tokens = alias.split()
            if not 2 <= len(tokens) <= 3:
                # One token is already as short as it gets; four or more is a
                # description ("The Gentleman In The White Waistcoat"), whose
                # fragments are not names.
                continue
            core = [t for t in tokens if t.strip(".").lower() not in honorifics]
            if not core or not all(t[:1].isupper() for t in core):
                continue
            for token in {core[0], core[-1]}:
                bare = token.strip(".,;:!?")
                if len(bare) < 3 or bare.lower() in _STOP_TOKENS:
                    continue
                proposals.setdefault(bare, set()).add(c.cid)

    derived: dict[str, Character] = {}
    for form, cids in proposals.items():
        if form in gold:            # PDNC already assigned this form
            continue
        if len(cids) != 1:          # ambiguous between characters -> refuse
            continue
        derived[form] = owner[next(iter(cids))]
    return derived


def build_gazetteer(novel: Novel, derive: bool = True) -> dict[str, Character]:
    """Surface form -> character, gold aliases plus derived short names."""
    forms = dict(novel.aliases)
    if derive:
        for form, character in derive_forms(novel.characters, HONORIFICS).items():
            forms.setdefault(form, character)
    return forms


def _flex(alias: str) -> str:
    """Escape an alias so it still matches across a line break.

    PDNC ships the novels hard-wrapped at ~70 columns, so a two-word name
    sometimes lands astride a newline: "Mr.\\nWoodhouse", "said\\nMr. Bertram".
    Matching a literal space misses those. The effect is small on its own
    (+0.2pp explicit ceiling, ~14 quotations in train) but it is free, and
    leaving it in place would attribute a typographic accident to the model.
    Every run of whitespace becomes ``\\s+``.
    """
    return r"\s+".join(re.escape(part) for part in _WS.split(alias.strip()) if part)


def _alias_pattern(forms: dict[str, Character]) -> re.Pattern:
    """One alternation regex over every surface form, longest first.

    Longest-first is load-bearing. Python's ``|`` is first-match-wins, not
    longest-match-wins, so with "Bennet" listed before "Mr. Bennet" every
    mention of the father would be recorded as the bare surname -- which in
    Pride and Prejudice belongs to a different character record. Sorting by
    descending length makes the longer form win wherever both apply.
    """
    ordered = sorted(forms, key=len, reverse=True)
    # \b fails against aliases ending in punctuation ("Mrs." / "M."), so the
    # right edge uses a lookahead for a non-word character instead.
    return re.compile(r"\b(?:" + "|".join(_flex(a) for a in ordered) + r")(?!\w)")


# Cache one gazetteer and compiled pattern per novel: rebuilding a 300-way
# alternation for each of 37k quotes dominates runtime otherwise.
_GAZETTEERS: dict[tuple[str, bool], tuple[dict[str, Character], re.Pattern]] = {}


def _gazetteer_for(
    novel: Novel, derive: bool = True
) -> tuple[dict[str, Character], re.Pattern]:
    key = (novel.name, derive)
    if key not in _GAZETTEERS:
        forms = build_gazetteer(novel, derive)
        _GAZETTEERS[key] = (forms, _alias_pattern(forms))
    return _GAZETTEERS[key]


def find_mentions(
    novel: Novel,
    quote: Quote,
    chars_before: int = CANDIDATE_CHARS_BEFORE,
    chars_after: int = CANDIDATE_CHARS_AFTER,
    derive: bool = True,
) -> list[Mention]:
    """Every character mention in the narration window around one quote.

    Text *inside* the quotation is excluded. A speaker naming someone else mid
    speech ("I told you, Elizabeth, that...") is an addressee cue, not a
    speaker cue, and including it would put the wrong character at distance
    zero on exactly the quotes that are hardest anyway.
    """
    forms, pattern = _gazetteer_for(novel, derive)
    text = novel.text
    q_start, q_end = quote.start, quote.end

    # Narration zones: before the quote, between its spans, and after it.
    zones: list[tuple[int, int, str]] = [
        (max(0, q_start - chars_before), q_start, "before"),
        (q_end, min(len(text), q_end + chars_after), "after"),
    ]
    for (_, a_end), (b_start, _) in zip(quote.spans, quote.spans[1:]):
        zones.append((a_end, b_start, "inner"))

    mentions: list[Mention] = []
    for lo, hi, zone in zones:
        if hi <= lo:
            continue
        window = text[lo:hi]
        for m in pattern.finditer(window):
            # The match may carry the newline the pattern was built to span
            # ("Mr.\nWoodhouse"), so collapse whitespace before looking it up.
            surface = _WS.sub(" ", m.group())
            character = forms.get(surface)
            if character is None:            # alias cased differently; skip
                continue
            start, end = lo + m.start(), lo + m.end()
            distance = 0 if q_start <= start <= q_end else min(
                abs(q_start - end), abs(start - q_end)
            )
            neighbourhood = text[max(0, start - 40) : end + 40]
            mentions.append(
                Mention(
                    character=character,
                    surface=m.group(),
                    start=start,
                    end=end,
                    zone=zone,
                    distance=distance,
                    near_speech_verb=bool(_SPEECH_VERB_RE.search(neighbourhood)),
                )
            )

    mentions.sort(key=lambda m: m.distance)
    return mentions


def enumerate_candidates(
    novel: Novel,
    quote: Quote,
    chars_before: int = CANDIDATE_CHARS_BEFORE,
    chars_after: int = CANDIDATE_CHARS_AFTER,
    mark_gold: bool = True,
    derive: bool = True,
) -> list[Candidate]:
    """Group the window's mentions into one candidate per character."""
    grouped: dict[int, list[Mention]] = {}
    for m in find_mentions(novel, quote, chars_before, chars_after, derive):
        grouped.setdefault(m.character.cid, []).append(m)

    candidates = [
        Candidate(
            character=novel.character(cid),
            mentions=ms,
            is_gold=mark_gold and cid == quote.speaker_id,
        )
        for cid, ms in grouped.items()
    ]
    candidates.sort(key=lambda c: c.nearest.distance)
    return candidates


# ---------------------------------------------------------------- ceiling


def oracle_recall(
    novels: list[Novel],
    chars_before: int = CANDIDATE_CHARS_BEFORE,
    chars_after: int = CANDIDATE_CHARS_AFTER,
    derive: bool = True,
) -> dict:
    """Fraction of quotes whose gold speaker appears among the candidates.

    Broken out by quote type and by narrative person, because those are the two
    axes on which the ceiling actually moves.
    """
    hit = Counter()
    total = Counter()
    n_cands = Counter()

    for novel in novels:
        key_person = f"person{novel.person}"
        for q in novel.quotes:
            cands = enumerate_candidates(
                novel, q, chars_before, chars_after, derive=derive
            )
            found = any(c.is_gold for c in cands)
            for key in ("all", q.quote_type, key_person, novel.name):
                total[key] += 1
                hit[key] += found
                n_cands[key] += len(cands)

    return {
        "hit": hit,
        "total": total,
        "mean_candidates": {
            k: n_cands[k] / max(total[k], 1) for k in total
        },
        "recall": {k: hit[k] / max(total[k], 1) for k in total},
    }


if __name__ == "__main__":
    import sys

    from config import CANDIDATE_WINDOW_SWEEP
    from pdnc import load_corpus, load_split

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    novels = load_corpus() if split == "all" else load_split(split)
    print(f"candidate-set ceiling on {split} ({len(novels)} novels)\n")

    # --- window sweep: what does a wider net buy, and what does it cost?
    hdr = (
        f"{'window':>13s} {'all':>7s} {'explicit':>9s} {'anaphoric':>10s} "
        f"{'implicit':>9s} {'1st-pers':>9s} {'3rd-pers':>9s} {'cands/q':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for before, after in CANDIDATE_WINDOW_SWEEP:
        res = oracle_recall(novels, before, after)
        r, m = res["recall"], res["mean_candidates"]
        print(
            f"{f'-{before}/+{after}':>13s} "
            f"{100*r['all']:>6.1f}% {100*r['explicit']:>8.1f}% "
            f"{100*r['anaphoric']:>9.1f}% {100*r['implicit']:>8.1f}% "
            f"{100*r.get('person1', 0):>8.1f}% {100*r.get('person3', 0):>8.1f}% "
            f"{m['all']:>8.1f}"
        )

    # --- what the derived-name rule is worth, at the configured window
    print("\ngazetteer ablation (at the configured window):")
    for derive, label in ((False, "PDNC aliases only"), (True, "+ derived names")):
        res = oracle_recall(novels, derive=derive)
        r, m = res["recall"], res["mean_candidates"]
        print(
            f"  {label:20s} all={100*r['all']:>5.1f}%  "
            f"expl={100*r['explicit']:>5.1f}%  anap={100*r['anaphoric']:>5.1f}%  "
            f"impl={100*r['implicit']:>5.1f}%  ({m['all']:.1f} cands/q)"
        )

    # --- per novel at the configured window
    res = oracle_recall(novels)
    print(f"\nper-novel recall at -{CANDIDATE_CHARS_BEFORE}/+{CANDIDATE_CHARS_AFTER}:")
    for novel in sorted(novels, key=lambda n: res["recall"][n.name]):
        print(
            f"  {novel.name:32s} person={novel.person} "
            f"{100*res['recall'][novel.name]:>5.1f}%  "
            f"({res['mean_candidates'][novel.name]:.1f} cands/q)"
        )

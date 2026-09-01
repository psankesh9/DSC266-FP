"""Quotation extraction and quote-type classification.

Two jobs:

1. Find every span of quoted speech in a novel.
2. Label HOW its speaker is signalled -- explicit, anaphoric, or implicit.

Step 2 is the analytical spine of the project. Explicit quotes carry their own
answer in a speech tag and are nearly free to attribute; anaphoric and implicit
quotes are the real task. Reporting a single pooled accuracy hides that gap
entirely, which is exactly the failure this project is about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field

from config import (
    CONTEXT_CHARS_AFTER,
    CONTEXT_CHARS_BEFORE,
    MAX_QUOTE_CHARS,
    MIN_QUOTE_CHARS,
    SPEECH_VERBS,
    THIRD_PERSON_PRONOUNS,
)

_VERBS = "|".join(sorted(SPEECH_VERBS, key=len, reverse=True))
_NAME = r"(?:[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,2})"

# "<quote>," said Elizabeth.   /   "<quote>," Elizabeth said.
TAG_AFTER = re.compile(rf"^[,.;!?]?[\"']?\s*(?:{_VERBS})\s+(?P<name>{_NAME})\b", re.I)
TAG_AFTER_INV = re.compile(rf"^[,.;!?]?[\"']?\s*(?P<name>{_NAME})\s+(?:{_VERBS})\b")

# Elizabeth said, "<quote>"   /   said Elizabeth, "<quote>"
TAG_BEFORE = re.compile(rf"(?P<name>{_NAME})\s+(?:{_VERBS})[,:]?\s*$")
TAG_BEFORE_INV = re.compile(rf"(?:{_VERBS})\s+(?P<name>{_NAME})[,:]?\s*$", re.I)

# Pronoun-tagged: "<quote>," she said.  /  she said, "<quote>"
PRON = "|".join(sorted(THIRD_PERSON_PRONOUNS))
PRON_AFTER = re.compile(rf"^[,.;!?]?[\"']?\s*(?:(?:{_VERBS})\s+(?P<p1>{PRON})|(?P<p2>{PRON})\s+(?:{_VERBS}))\b", re.I)
PRON_BEFORE = re.compile(rf"\b(?P<p>{PRON})\s+(?:{_VERBS})[,:]?\s*$", re.I)


@dataclass
class Quote:
    """One span of quoted speech with the context needed to attribute it."""
    book: str
    qid: int
    start: int                  # char offset of text inside the quote marks
    end: int
    text: str
    quote_type: str = "implicit"
    tag_name: str | None = None      # speaker surface form, if explicitly tagged
    tag_pronoun: str | None = None   # pronoun, if anaphorically tagged
    para_index: int = 0
    context_before: str = ""
    context_after: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """Split into paragraphs, keeping each one's absolute character offset."""
    out, pos = [], 0
    for chunk in text.split("\n\n"):
        if chunk.strip():
            out.append((pos, chunk))
        pos += len(chunk) + 2
    return out


def _spans_in_paragraph(para: str) -> list[tuple[int, int]]:
    """Pair up quotation marks within one paragraph.

    Marks are paired strictly left to right. An odd final mark is dropped: in
    19th-century typesetting a paragraph of continued speech opens with a mark
    and never closes it, so an unpaired opener is normal rather than an error.
    """
    marks = [m.start() for m in re.finditer(r'"', para)]
    spans = []
    for i in range(0, len(marks) - 1, 2):
        a, b = marks[i], marks[i + 1]
        if b - a > 1:
            spans.append((a + 1, b))
    return spans


def classify(quote_text: str, before: str, after: str) -> tuple[str, str | None, str | None]:
    """Decide how this quote's speaker is signalled.

    Returns (quote_type, tag_name, tag_pronoun).

    Order matters: a named tag beats a pronoun tag, and a tag on either side
    of the quote beats no tag at all.
    """
    # Named speech tag immediately after the closing mark.
    for rx in (TAG_AFTER, TAG_AFTER_INV):
        m = rx.match(after)
        if m:
            return "explicit", m.group("name"), None

    # Named speech tag immediately before the opening mark.
    for rx in (TAG_BEFORE, TAG_BEFORE_INV):
        m = rx.search(before)
        if m:
            return "explicit", m.group("name"), None

    # Pronoun tag on either side -> the speaker is referred to, but by pronoun,
    # so resolving it needs coreference rather than string matching.
    m = PRON_AFTER.match(after)
    if m:
        return "anaphoric", None, (m.group("p1") or m.group("p2")).lower()
    m = PRON_BEFORE.search(before)
    if m:
        return "anaphoric", None, m.group("p").lower()

    # No tag at all: the speaker must be inferred from conversational structure.
    return "implicit", None, None


def extract(text: str, book: str) -> list[Quote]:
    """Extract every usable quote from one novel."""
    quotes: list[Quote] = []
    qid = 0

    for para_idx, (para_off, para) in enumerate(_paragraphs(text)):
        for a, b in _spans_in_paragraph(para):
            qtext = para[a:b].strip()
            if not (MIN_QUOTE_CHARS <= len(qtext) <= MAX_QUOTE_CHARS):
                continue

            abs_start = para_off + a
            abs_end = para_off + b

            before = text[max(0, abs_start - CONTEXT_CHARS_BEFORE) : abs_start - 1]
            after = text[abs_end + 1 : abs_end + 1 + CONTEXT_CHARS_AFTER]

            qtype, name, pron = classify(qtext, before, after)

            quotes.append(
                Quote(
                    book=book,
                    qid=qid,
                    start=abs_start,
                    end=abs_end,
                    text=qtext,
                    quote_type=qtype,
                    tag_name=name,
                    tag_pronoun=pron,
                    para_index=para_idx,
                    context_before=before,
                    context_after=after,
                )
            )
            qid += 1

    return quotes


if __name__ == "__main__":
    from collections import Counter
    from config import CLEAN_DIR, GUTENBERG_CORPUS

    print(f"{'book':22s} {'quotes':>7s}  {'explicit':>9s} {'anaphoric':>10s} {'implicit':>9s}")
    print("-" * 64)
    grand = Counter()
    for _, name, _ in GUTENBERG_CORPUS:
        text = (CLEAN_DIR / f"{name}.txt").read_text(encoding="utf-8")
        qs = extract(text, name)
        c = Counter(q.quote_type for q in qs)
        grand.update(c)
        grand["total"] += len(qs)
        n = max(len(qs), 1)
        print(
            f"{name:22s} {len(qs):>7,}  "
            f"{c['explicit']:>6,} {100*c['explicit']/n:>4.0f}% "
            f"{c['anaphoric']:>6,} {100*c['anaphoric']/n:>3.0f}% "
            f"{c['implicit']:>6,} {100*c['implicit']/n:>3.0f}%"
        )
    t = grand["total"]
    print("-" * 64)
    print(
        f"{'TOTAL':22s} {t:>7,}  "
        f"{grand['explicit']:>6,} {100*grand['explicit']/t:>4.0f}% "
        f"{grand['anaphoric']:>6,} {100*grand['anaphoric']/t:>3.0f}% "
        f"{grand['implicit']:>6,} {100*grand['implicit']/t:>3.0f}%"
    )

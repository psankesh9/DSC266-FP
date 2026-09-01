"""Loader for the Project Dialogism Novel Corpus (Vishnubhotla et al., 2022).

PDNC is the labelled backbone of this project: 28 novels, 37,131 quotations,
each carrying a gold speaker, addressee list, and quote type. This module turns
the on-disk CSVs into typed records with the narration context that attribution
actually needs.

Five properties of the release are handled here rather than left to bite later:

1. ``quoteByteSpans`` holds CHARACTER offsets, not byte offsets. Decoding the
   text as bytes and slicing by these numbers matches 0 of 267 spot-checked
   quotations; slicing the decoded string matches all of them.

2. The ``speaker`` field is not always a ``Main Name``. 2,053 quotations (5.5%)
   name the speaker by an alias -- "Lucy" for "Lucy Honeychurch". Resolving
   through the alias gazetteer takes coverage from 94.5% to 100% and, checked
   across all 28 novels, no alias is ambiguous between two characters.

3. 29% of quotations are split into several spans by an interjected speech tag
   (``"My dear Mr. Bennet," said his lady, "have you heard..."``). The gap
   between spans is narration, and it is exactly where the tag lives, so it is
   preserved separately instead of being spliced out or swallowed into the
   quote text.

4. 28 rows have a blank ``quoteType``. They are dropped, not guessed: the
   per-type breakdown is the headline analysis and is not worth contaminating
   for 0.08% more data.

5. Gender is 'U' when unknown and 'X' when un-annotated. Both mean "no usable
   gender", and collapsing them keeps downstream agreement features honest --
   a third of characters have no gender at all.
"""

from __future__ import annotations

import ast
import logging
import pickle
from dataclasses import dataclass

import pandas as pd

from config import (
    BUILD_DIR,
    CONTEXT_CHARS_AFTER,
    CONTEXT_CHARS_BEFORE,
    DROP_UNTYPED,
    PDNC_CORPUS,
    PDNC_NOVELS,
    QUOTE_TYPES,
)

logger = logging.getLogger(__name__)

_CACHE = BUILD_DIR / "pdnc_corpus.pkl"
_CACHE_VERSION = 1


# ---------------------------------------------------------------- records


@dataclass(frozen=True)
class Character:
    """One annotated character in one novel."""

    cid: int
    name: str                      # PDNC "Main Name"
    aliases: frozenset[str]        # includes the main name
    gender: str                    # "M", "F", or "" when unknown/un-annotated
    category: str                  # major / intermediate / minor

    @property
    def has_gender(self) -> bool:
        return self.gender in ("M", "F")


@dataclass
class Quote:
    """One gold-annotated quotation plus the narration around it."""

    book: str
    qid: str
    spans: list[tuple[int, int]]   # character offsets into the novel text
    text: str                      # spoken words only, spans joined
    speaker: str                   # resolved to the character's main name
    speaker_id: int
    quote_type: str                # explicit / anaphoric / implicit
    addressees: list[str]
    referring_expression: str      # e.g. "said his lady to him"; may be ""

    left: str = ""                 # narration before the quote opens
    inner: str = ""                # narration interjected between spans
    right: str = ""                # narration after the quote closes

    @property
    def start(self) -> int:
        return self.spans[0][0]

    @property
    def end(self) -> int:
        return self.spans[-1][1]

    @property
    def is_split(self) -> bool:
        """True when a speech tag interrupts the quotation."""
        return len(self.spans) > 1

    @property
    def narration(self) -> str:
        """All narration a speaker could be named in, nearest material last.

        Ordering matters for the baseline: the nearest preceding mention is the
        strongest single cue in the literature, so ``left`` ends adjacent to
        the quote and is searched from its right edge backwards.
        """
        return f"{self.left}\n{self.inner}\n{self.right}"


@dataclass
class Novel:
    """One PDNC novel: its text, cast, and quotations."""

    name: str
    author: str
    person: int                    # narrative person, 1 or 3
    genre: str
    split: str
    text: str
    characters: list[Character]
    quotes: list[Quote]

    def __post_init__(self) -> None:
        self._by_id = {c.cid: c for c in self.characters}
        # Surface form -> character. Checked across all 28 novels: no alias is
        # shared by two characters, so this mapping loses nothing. Consumers
        # that scan text for these forms must still try the LONGEST first, or
        # "Bennet" will fire inside "Mr. Bennet" and pick the wrong person --
        # see candidates.py, which sorts before matching.
        self._by_alias: dict[str, Character] = {}
        for c in self.characters:
            for a in c.aliases:
                self._by_alias.setdefault(a, c)

    def character(self, cid: int) -> Character:
        return self._by_id[cid]

    def by_alias(self, alias: str) -> Character | None:
        return self._by_alias.get(alias)

    @property
    def aliases(self) -> dict[str, Character]:
        """Every surface form in this novel that names a character."""
        return self._by_alias

    def __repr__(self) -> str:  # keeps debugging output readable
        return (
            f"<Novel {self.name} [{self.split}] "
            f"{len(self.quotes)} quotes, {len(self.characters)} characters>"
        )


# ---------------------------------------------------------------- parsing


def _parse(value) -> list:
    """Parse a cell holding a Python list or set literal, unevaluated.

    PDNC writes aliases as ``{'a', 'b'}`` in some rows and ``['a']`` in others
    (822 vs 406 across the corpus) because they were serialised from different
    Python types. Both are valid literals, so one parser covers them.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        logger.warning("unparseable literal: %.60s", value)
        return []
    if isinstance(parsed, (set, frozenset, tuple)):
        return list(parsed)
    return parsed if isinstance(parsed, list) else [parsed]


def _strings(value) -> list[str]:
    """Parse a cell holding a flat list/set of names."""
    return [str(x).strip() for x in _parse(value) if str(x).strip()]


def _spans(value) -> list[tuple[int, int]]:
    """Parse ``quoteByteSpans`` into (start, end) character offsets.

    Despite the column name these index the decoded string, not its bytes:
    slicing ``novel_text.txt`` as UTF-8 bytes by these numbers reproduces 0 of
    267 spot-checked quotations, slicing the decoded text reproduces all 267.
    The novels contain enough non-ASCII punctuation (em dashes, curly quotes)
    that the distinction shifts offsets by thousands of positions by end of
    book, so this is not a cosmetic detail.
    """
    out = []
    for pair in _parse(value):
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            out.append((int(pair[0]), int(pair[1])))
        else:
            logger.warning("malformed span %r", pair)
    return out


def _load_characters(folder: str) -> tuple[list[Character], dict[str, Character]]:
    """Read character_info.csv and build the alias gazetteer."""
    df = pd.read_csv(PDNC_NOVELS / folder / "character_info.csv")
    chars: list[Character] = []
    index: dict[str, Character] = {}

    for _, row in df.iterrows():
        name = str(row["Main Name"]).strip()
        aliases = set(_strings(row["Aliases"]))
        aliases.add(name)
        gender = str(row["Gender"]).strip()
        c = Character(
            cid=int(row["Character ID"]),
            name=name,
            aliases=frozenset(aliases),
            # 'U' (unknown) and 'X' (un-annotated) both mean "no gender".
            gender=gender if gender in ("M", "F") else "",
            category=str(row["Category"]).strip(),
        )
        chars.append(c)
        for a in aliases:
            index.setdefault(a, c)

    return chars, index


def _contexts(text: str, spans: list[tuple[int, int]]) -> tuple[str, str, str]:
    """Slice the narration before, inside, and after a quotation."""
    start, end = spans[0][0], spans[-1][1]
    left = text[max(0, start - CONTEXT_CHARS_BEFORE) : start]
    right = text[end : end + CONTEXT_CHARS_AFTER]
    inner = " ".join(
        text[a[1] : b[0]] for a, b in zip(spans, spans[1:])
    )
    return left, inner, right


def _load_quotes(
    folder: str, text: str, index: dict[str, Character]
) -> tuple[list[Quote], int, int]:
    """Read quotation_info.csv. Returns (quotes, n_untyped, n_unresolved)."""
    df = pd.read_csv(PDNC_NOVELS / folder / "quotation_info.csv")
    quotes: list[Quote] = []
    untyped = unresolved = 0

    for _, row in df.iterrows():
        qtype = str(row["quoteType"]).strip().lower()
        if qtype not in QUOTE_TYPES:
            untyped += 1
            if DROP_UNTYPED:
                continue

        # A handful of speaker cells carry trailing whitespace
        # ("Frederick Tilney " in Northanger Abbey). Without the strip these
        # are the only 5 quotations in the corpus that fail to resolve.
        speaker_raw = str(row["speaker"]).strip()
        character = index.get(speaker_raw)
        if character is None:
            unresolved += 1
            logger.warning("%s: unresolved speaker %r", folder, speaker_raw)
            continue

        spans = _spans(row["quoteByteSpans"])
        if not spans:
            continue

        left, inner, right = _contexts(text, spans)
        quotes.append(
            Quote(
                book=folder,
                qid=str(row["quoteID"]),
                spans=spans,
                text=" ".join(text[a:b] for a, b in spans),
                speaker=character.name,
                speaker_id=character.cid,
                quote_type=qtype,
                addressees=_strings(row["addressees"]),
                referring_expression=(
                    str(row["referringExpression"]).strip()
                    if pd.notna(row["referringExpression"])
                    else ""
                ),
                left=left,
                inner=inner,
                right=right,
            )
        )

    quotes.sort(key=lambda q: q.start)
    return quotes, untyped, unresolved


def load_novel(folder: str, author: str, person: int, genre: str, split: str) -> Novel:
    """Load one novel end to end."""
    text = (PDNC_NOVELS / folder / "novel_text.txt").read_text(encoding="utf-8")
    characters, index = _load_characters(folder)
    quotes, untyped, unresolved = _load_quotes(folder, text, index)

    if untyped or unresolved:
        logger.info(
            "%s: dropped %d untyped, %d unresolved-speaker quotations",
            folder, untyped, unresolved,
        )

    return Novel(
        name=folder,
        author=author,
        person=person,
        genre=genre,
        split=split,
        text=text,
        characters=characters,
        quotes=quotes,
    )


# ---------------------------------------------------------------- corpus API


def load_corpus(rebuild: bool = False) -> list[Novel]:
    """Load all 28 novels, caching the parsed result.

    Parsing is a few seconds, dominated by ``literal_eval`` over 37k span and
    alias cells. Cheap enough to redo, but every downstream script loads the
    corpus, so the cache keeps iteration fast.
    """
    if _CACHE.exists() and not rebuild:
        try:
            blob = pickle.loads(_CACHE.read_bytes())
            if blob.get("version") == _CACHE_VERSION:
                return blob["novels"]
        except Exception as exc:  # noqa: BLE001 - a stale cache must not be fatal
            logger.warning("ignoring unreadable cache (%s); rebuilding", exc)

    novels = [
        load_novel(folder, author, person, genre, split)
        for folder, author, person, genre, split in PDNC_CORPUS
    ]
    _CACHE.write_bytes(
        pickle.dumps({"version": _CACHE_VERSION, "novels": novels})
    )
    return novels


def load_split(split: str, rebuild: bool = False) -> list[Novel]:
    """Load only the novels in one split."""
    if split not in ("train", "dev", "test"):
        raise ValueError(f"unknown split {split!r}")
    return [n for n in load_corpus(rebuild) if n.split == split]


def all_quotes(novels: list[Novel]) -> list[Quote]:
    return [q for n in novels for q in n.quotes]


# ---------------------------------------------------------------- report

if __name__ == "__main__":
    import collections
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    novels = load_corpus(rebuild="--rebuild" in sys.argv)

    hdr = f"{'novel':30s} {'split':6s} {'q':>5s} {'expl':>11s} {'anap':>11s} {'impl':>11s} {'cast':>5s} {'split-q':>7s}"
    print(hdr)
    print("-" * len(hdr))

    per_split = collections.defaultdict(collections.Counter)
    for n in novels:
        c = collections.Counter(q.quote_type for q in n.quotes)
        d = max(len(n.quotes), 1)
        per_split[n.split].update(c)
        per_split[n.split]["total"] += len(n.quotes)
        per_split[n.split]["books"] += 1
        nsplit = sum(q.is_split for q in n.quotes)
        print(
            f"{n.name:30s} {n.split:6s} {len(n.quotes):>5,} "
            f"{c['explicit']:>5,}{100*c['explicit']/d:>5.0f}% "
            f"{c['anaphoric']:>5,}{100*c['anaphoric']/d:>5.0f}% "
            f"{c['implicit']:>5,}{100*c['implicit']/d:>5.0f}% "
            f"{len(n.characters):>5,} {100*nsplit/d:>6.0f}%"
        )

    print("-" * len(hdr))
    for split in ("train", "dev", "test"):
        s = per_split[split]
        d = max(s["total"], 1)
        print(
            f"{split.upper():30s} {'':6s} {s['total']:>5,} "
            f"{s['explicit']:>5,}{100*s['explicit']/d:>5.0f}% "
            f"{s['anaphoric']:>5,}{100*s['anaphoric']/d:>5.0f}% "
            f"{s['implicit']:>5,}{100*s['implicit']/d:>5.0f}% "
            f"{s['books']:>4} bk"
        )

    total = sum(len(n.quotes) for n in novels)
    print(f"\n{total:,} quotations across {len(novels)} novels")

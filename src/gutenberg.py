"""Download Project Gutenberg novels and strip their boilerplate.

Gutenberg wraps every book in a licence header and footer. Left in place the
header alone contributes hundreds of spurious capitalised tokens ("Project",
"Gutenberg", "Literary Archive Foundation") that the character detector would
happily promote to characters, so removing it is a correctness issue rather
than cosmetic tidying.
"""

from __future__ import annotations

import logging
import re
import time

import requests

from config import CLEAN_DIR, GUTENBERG_CORPUS, RAW_DIR

logger = logging.getLogger(__name__)

MIRRORS = [
    "https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
    "https://www.gutenberg.org/files/{gid}/{gid}.txt",
    "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
]

START_PATTERNS = [
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
]
END_PATTERNS = [
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    r"\*\*\*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
]


def download(gid: int, name: str, force: bool = False) -> str:
    """Fetch one book, caching the raw bytes so reruns cost nothing."""
    raw_path = RAW_DIR / f"{name}.txt"
    if raw_path.exists() and not force:
        return raw_path.read_text(encoding="utf-8", errors="replace")

    last_err = None
    for template in MIRRORS:
        url = template.format(gid=gid)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 10_000:
                # Decode explicitly. requests' charset sniffing guesses latin-1
                # for these files and silently turns em-dashes into U+FFFD,
                # which then splits words in the middle during tokenisation.
                try:
                    text = resp.content.decode("utf-8")
                except UnicodeDecodeError:
                    text = resp.content.decode("latin-1")
                raw_path.write_text(text, encoding="utf-8")
                logger.info("downloaded %s (%d chars) from %s", name, len(text), url)
                time.sleep(0.5)          # be polite to the mirror
                return text
            last_err = f"HTTP {resp.status_code}"
        except Exception as exc:         # noqa: BLE001 - report and try next mirror
            last_err = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(f"could not download {name} (gid={gid}): {last_err}")


def strip_boilerplate(text: str) -> str:
    """Cut everything outside the START/END markers."""
    start = 0
    for pat in START_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            start = m.end()
            break

    end = len(text)
    for pat in END_PATTERNS:
        m = re.search(pat, text[start:], re.IGNORECASE | re.DOTALL)
        if m:
            end = start + m.start()
            break

    return text[start:end]


def normalise(text: str) -> str:
    """Canonicalise whitespace and quote marks.

    Curly and straight marks are unified so the quote extractor only has to
    reason about one pair. Apostrophes inside words ("don't", "Darcy's") are
    protected first, otherwise the right single quote would be misread as a
    closing quotation mark and shred the dialogue.
    """
    # Protect intra-word apostrophes before touching single quotes at all.
    text = re.sub(r"(?<=\w)['’](?=\w)", "", text)

    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("„", '"').replace("‟", '"')

    text = text.replace("", "'")          # restore apostrophes

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Underscore emphasis is a Gutenberg convention, not part of the prose.
    text = re.sub(r"_([^_\n]{1,80})_", r"\1", text)
    return text.strip()


# A heading is a SHORT standalone line naming a division and its number.
# Requiring the numeral matters: without it the bare words "part" and "book"
# match wrapped prose lines ("book, he was regardless of time...").
HEADING_RE = re.compile(
    r"^[ \t]*(chapter|part|book|adventure|letter|canto|act)\s+"
    r"([ivxlcdm]+|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"[.\s:—-]*.{0,40}$",
    re.IGNORECASE,
)

# Dialogue density required to believe a heading opens real narrative.
_DENSITY_WINDOW = 15_000
_DENSITY_MIN_QUOTES = 60

# Longest plausible gap between consecutive chapter headings.
_MAX_CHAPTER_CHARS = 60_000
# Shortest span of text that can plausibly be a chapter rather than a TOC line.
_MIN_CHAPTER_CHARS = 1_000


def _drop_contents_block(text: str) -> str:
    """Remove a 'Contents' listing whose entries are bare roman numerals.

    Short-story collections (Sherlock Holmes) and novellas (Heart of Darkness)
    index by numeral alone, so the CHAPTER-keyword heading rule never fires.
    The listing ends where the first real prose paragraph begins.
    """
    m = re.search(r"(?im)^[ \t]*contents[ \t]*$", text[:6000])
    if not m:
        return text
    for para in re.finditer(r"\n\n", text[m.end():]):
        start = m.end() + para.end()
        # Measure to the next BLANK line, not the next newline: Gutenberg hard
        # wraps prose at ~70 characters, so a per-line length test can never
        # distinguish a paragraph from a one-line contents entry.
        end = text.find("\n\n", start)
        end = len(text) if end == -1 else end
        if end - start > 200:               # a real paragraph, not an entry
            return text[start:]
    return text


def _find_headings(text: str) -> list[int]:
    """Character offsets of every plausible division heading."""
    out = []
    pos = 0
    for line in text.split("\n"):
        if len(line.strip()) <= 60 and HEADING_RE.match(line):
            out.append(pos)
        pos += len(line) + 1
    return out


def _drop_toc(text: str, headings: list[int]) -> tuple[str, list[int]]:
    """Delete a table of contents: a dense run of headings with no prose.

    In a TOC the headings sit within a couple of hundred characters of each
    other. In the body they are thousands apart.
    """
    if len(headings) < 4:
        return text, headings

    # A table-of-contents entry is followed immediately by the next entry; a
    # real chapter heading is followed by a chapter's worth of prose. Testing
    # for prose directly avoids tuning a character-gap threshold, which fails
    # whenever the body's first heading happens to sit close behind the TOC.
    bounds = headings[1:] + [len(text)]
    for i, (start, nxt) in enumerate(zip(headings, bounds)):
        if nxt - start >= _MIN_CHAPTER_CHARS:
            if i == 0:
                return text, headings          # no TOC present
            return text[start:], [h - start for h in headings[i:]]
    return text, headings


def strip_front_matter(text: str) -> str:
    """Drop tables of contents, prefaces, and critical introductions.

    Gutenberg editions frequently prepend an editor's essay. Left in place it
    contributes thousands of capitalised names with no dialogue, which skews
    both the character gazetteer and the quote-type distribution. Narrative is
    distinguished from criticism by dialogue density.
    """
    text = _drop_contents_block(text)
    headings = _find_headings(text)
    text, headings = _drop_toc(text, headings)

    for i, start in enumerate(headings):
        window = text[start : start + _DENSITY_WINDOW]
        if window.count('"') < _DENSITY_MIN_QUOTES:
            continue
        # Walk back over earlier headings spaced like ordinary chapters. The
        # opening chapter of a novel is often quiet enough to miss the density
        # bar on its own, and dropping it would silently discard real data;
        # the gap to a preface is much larger than a chapter's length.
        j = i
        while j > 0 and headings[j] - headings[j - 1] < _MAX_CHAPTER_CHARS:
            j -= 1
        return text[headings[j] :]

    # Sparse-dialogue books (Heart of Darkness, War of the Worlds) never clear
    # the density bar; fall back to the first heading if it is early enough.
    if headings and headings[0] < 0.10 * len(text):
        return text[headings[0] :]
    return text


def prepare(gid: int, name: str, force: bool = False) -> str:
    """Download, clean, and cache one novel. Returns the clean text."""
    clean_path = CLEAN_DIR / f"{name}.txt"
    if clean_path.exists() and not force:
        return clean_path.read_text(encoding="utf-8")

    raw = download(gid, name, force=force)
    # normalise() must run before strip_front_matter(): the latter anchors on
    # line-start/line-end and counts '"' for dialogue density, and both break
    # against raw CRLF endings and un-normalised curly quotation marks.
    text = strip_front_matter(normalise(strip_boilerplate(raw)))
    clean_path.write_text(text, encoding="utf-8")
    return text


def prepare_all(force: bool = False) -> dict[str, str]:
    """Prepare every novel in the unlabelled out-of-domain corpus."""
    out: dict[str, str] = {}
    for gid, name, author in GUTENBERG_CORPUS:
        try:
            text = prepare(gid, name, force=force)
            out[name] = text
            print(f"  {name:22s} {author:9s} {len(text):>9,} chars")
        except Exception as exc:  # noqa: BLE001 - one bad mirror must not stop the run
            print(f"  {name:22s} FAILED: {exc}")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("Preparing out-of-domain corpus from Project Gutenberg\n")
    books = prepare_all()
    print(f"\n{len(books)}/{len(GUTENBERG_CORPUS)} novels ready.")

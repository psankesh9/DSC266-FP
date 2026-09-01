"""Extrinsic evaluation: how much of an audiobook a system mis-voices.

Accuracy counts quotations. A listener hears *seconds*. The two are not the
same measurement, because quotations are not the same length: a page-long
monologue and a one-word retort each count once toward accuracy, but the
monologue occupies a hundred times more audio. This module re-scores the same
predictions by spoken duration and asks whether the intrinsic ranking survives.

The headline number is **mis-voiced seconds per minute of audio**: of every
minute a listener hears, how many seconds arrive in the wrong character's
voice. The denominator is the whole book, narration included, because that is
what is actually played. A second figure divides by dialogue only, for readers
who want the number comparable to accuracy.

No TTS engine is installed on the development machine, so duration is derived
analytically from ``SPEAKING_RATE_WPM`` rather than by synthesising and timing
real audio. This metric is therefore a length-weighted error rate expressed in
seconds, and is described as such -- it is not a listening study, and it does
not model pauses, prosody, or a voice actor's pacing.

Two modelling decisions worth stating plainly:

* A quote the system declined to attribute (``pred is None``, which happens
  when the candidate set comes up empty) counts as mis-voiced. A real pipeline
  has to render those seconds somehow, and the narrator's voice on a line of
  dialogue is exactly the error this metric exists to count.
* Every character is assumed to have a distinct voice. A production cast pools
  minor characters into shared voices, which would make some confusions
  inaudible, so the numbers here are an upper bound on audible damage.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field

from config import (
    CONFIDENCE,
    N_BOOTSTRAP,
    QUOTE_TYPES,
    RESULTS_DIR,
    SEED,
    SPEAKING_RATE_WPM,
)
from evaluate import Prediction
from pdnc import load_split

logger = logging.getLogger(__name__)


def seconds(n_words: float) -> float:
    """Spoken duration of ``n_words`` words at the configured speaking rate."""
    return 60.0 * n_words / SPEAKING_RATE_WPM


@dataclass
class AudioReport:
    """Duration-weighted error of one system on one split."""

    name: str
    # Pooled over novels. Words, not seconds, are the stored unit: seconds are
    # a linear function of words, so keeping words lets the bootstrap resample
    # exact integer counts and convert once at the end.
    misvoiced_words: int = 0
    dialogue_words: int = 0
    total_words: int = 0
    per_novel: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    # Share of all mis-voiced audio attributable to each quote type.
    misvoiced_share: dict[str, float] = field(default_factory=dict)
    # Duration-weighted error rate within each quote type.
    rate_by_type: dict[str, float] = field(default_factory=dict)
    ci: tuple[float, float] = (float("nan"), float("nan"))
    # Error rate counting quotes rather than seconds, i.e. 1 - accuracy. Kept
    # here so the two weightings can be compared without reopening the report.
    quote_error_rate: float = 0.0

    @property
    def sec_per_audio_minute(self) -> float:
        return 60.0 * self.misvoiced_words / max(self.total_words, 1)

    @property
    def sec_per_dialogue_minute(self) -> float:
        return 60.0 * self.misvoiced_words / max(self.dialogue_words, 1)

    @property
    def duration_error_rate(self) -> float:
        """Share of *dialogue duration* voiced by the wrong character."""
        return self.misvoiced_words / max(self.dialogue_words, 1)

    @property
    def length_bias(self) -> float:
        """Duration-weighted error rate over quote-weighted error rate.

        Above 1.0 the system errs on longer-than-average quotes, so accuracy
        understates the audible damage; below 1.0 its mistakes are concentrated
        in short lines and accuracy overstates it. This ratio is the only thing
        the extrinsic metric can say that accuracy cannot.
        """
        return self.duration_error_rate / max(self.quote_error_rate, 1e-9)

    @property
    def misvoiced_hours(self) -> float:
        return seconds(self.misvoiced_words) / 3600.0

    def __str__(self) -> str:
        return (
            f"{self.name}\n"
            f"  {self.sec_per_audio_minute:5.2f} s mis-voiced per audio minute "
            f"[{self.ci[0]:.2f}, {self.ci[1]:.2f}]\n"
            f"  {self.sec_per_dialogue_minute:5.2f} s per dialogue minute\n"
            f"  {100*self.duration_error_rate:5.1f}% of dialogue duration "
            f"vs {100*self.quote_error_rate:.1f}% of quotes "
            f"(length bias {self.length_bias:.2f})"
        )


def _word_count(text: str) -> int:
    return len(text.split())


def _quote_words(novels) -> tuple[dict[tuple[str, str], int], dict[str, tuple[int, int]]]:
    """Word counts per quotation, and (dialogue, total) words per novel.

    Quote text is the spoken words alone -- quotation marks and the surrounding
    narration are excluded by the corpus loader -- so this counts exactly what
    a character voice would utter.
    """
    per_quote: dict[tuple[str, str], int] = {}
    per_novel: dict[str, tuple[int, int]] = {}
    for novel in novels:
        spoken = 0
        for q in novel.quotes:
            n = _word_count(q.text)
            per_quote[(novel.name, q.qid)] = n
            spoken += n
        per_novel[novel.name] = (spoken, _word_count(novel.text))
    return per_quote, per_novel


def _bootstrap_ratio(
    per_novel: dict[str, tuple[int, int]],
    scale: float = 60.0,
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile bootstrap of a ratio of sums, resampling whole novels.

    Same unit of resampling as the accuracy intervals -- novels, not quotes --
    for the reason given in ``evaluate._bootstrap_ci``: errors within a book
    are heavily correlated, and here they are correlated twice over, since one
    misattributed conversation supplies both many errors and many seconds.
    """
    books = [b for b, (_, d) in per_novel.items() if d > 0]
    if len(books) < 2:
        return (float("nan"), float("nan"))

    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        picked = [books[rng.randrange(len(books))] for _ in books]
        num = sum(per_novel[b][0] for b in picked)
        den = sum(per_novel[b][1] for b in picked)
        if den:
            vals.append(scale * num / den)

    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    tail = (1.0 - confidence) / 2.0
    return (vals[int(tail * (len(vals) - 1))],
            vals[int((1.0 - tail) * (len(vals) - 1))])


def measure(name: str, predictions: list[Prediction], novels) -> AudioReport:
    """Re-score one system's predictions by spoken duration."""
    per_quote, novel_words = _quote_words(novels)

    seen = {(p.book, p.qid) for p in predictions}
    missing = set(per_quote) - seen
    if missing:
        # Every system must be charged for the same audio, so a quote absent
        # from a system's output cannot simply vanish from its denominator.
        logger.warning("%s: %d quotes have no prediction; counted as mis-voiced",
                       name, len(missing))

    report = AudioReport(name=name)
    by_type_bad: dict[str, int] = {t: 0 for t in QUOTE_TYPES}
    by_type_all: dict[str, int] = {t: 0 for t in QUOTE_TYPES}
    per_novel_bad: dict[str, int] = {n.name: 0 for n in novels}

    for p in predictions:
        words = per_quote.get((p.book, p.qid))
        if words is None:                       # prediction for a filtered quote
            continue
        by_type_all[p.quote_type] = by_type_all.get(p.quote_type, 0) + words
        if not p.correct:
            report.misvoiced_words += words
            per_novel_bad[p.book] = per_novel_bad.get(p.book, 0) + words
            by_type_bad[p.quote_type] = by_type_bad.get(p.quote_type, 0) + words

    for key in missing:
        report.misvoiced_words += per_quote[key]
        per_novel_bad[key[0]] = per_novel_bad.get(key[0], 0) + per_quote[key]

    report.dialogue_words = sum(d for d, _ in novel_words.values())
    report.total_words = sum(t for _, t in novel_words.values())
    report.per_novel = {
        b: (per_novel_bad.get(b, 0), novel_words[b][0], novel_words[b][1])
        for b in novel_words
    }
    report.ci = _bootstrap_ratio(
        {b: (bad, novel_words[b][1]) for b, bad in per_novel_bad.items()}
    )

    total_bad = max(report.misvoiced_words, 1)
    report.misvoiced_share = {t: by_type_bad[t] / total_bad for t in QUOTE_TYPES}
    report.rate_by_type = {
        t: by_type_bad[t] / max(by_type_all[t], 1) for t in QUOTE_TYPES
    }

    scored = [p for p in predictions if (p.book, p.qid) in per_quote]
    n = len(scored) + len(missing)
    report.quote_error_rate = (
        sum(not p.correct for p in scored) + len(missing)
    ) / max(n, 1)
    return report


def paired_seconds_delta(
    a: AudioReport,
    b: AudioReport,
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int = SEED,
) -> dict:
    """Bootstrap the difference in mis-voiced seconds per minute, A minus B.

    Paired on novels for the reason set out in ``evaluate.paired_delta``: book
    difficulty swamps system difference, so each system's own interval is far
    wider than the gap between them.
    """
    books = [k for k in a.per_novel if k in b.per_novel]
    if len(books) < 2:
        return {"delta": float("nan"), "ci": (float("nan"), float("nan")),
                "p": float("nan")}

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        picked = [books[rng.randrange(len(books))] for _ in books]
        total = sum(a.per_novel[k][2] for k in picked)
        if not total:
            continue
        da = 60.0 * sum(a.per_novel[k][0] for k in picked) / total
        db = 60.0 * sum(b.per_novel[k][0] for k in picked) / total
        deltas.append(da - db)

    if not deltas:
        return {"delta": float("nan"), "ci": (float("nan"), float("nan")),
                "p": float("nan")}
    observed = a.sec_per_audio_minute - b.sec_per_audio_minute
    # One-sided: how often did A fail to reduce mis-voiced audio?
    worse = sum(d >= 0 for d in deltas) / len(deltas)
    deltas.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        "delta": observed,
        "ci": (deltas[int(tail * (len(deltas) - 1))],
               deltas[int((1.0 - tail) * (len(deltas) - 1))]),
        "p": worse,
    }


def load_reports(split: str) -> dict[str, list[Prediction]]:
    """Recover every saved system's predictions for one split.

    Reads the JSON written by ``evaluate.save`` rather than recomputing, so the
    audio metric scores exactly the predictions the accuracy table reported and
    needs neither a GPU nor a rerun of the baselines.
    """
    out: dict[str, list[Prediction]] = {}
    for path in sorted(RESULTS_DIR.glob(f"*_{split}.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(blob, dict) or "predictions" not in blob:
            continue                            # comparison_*.json, figure data
        name = blob.get("name") or path.stem
        if name.startswith("ranker_"):
            # Tagged training artefacts (``ranker_e8``, ``ranker_smoke``) are
            # one run each, and several may be lying around. The canonical
            # ranker predictions are the ones compare.py writes for the chosen
            # checkpoint; scoring both would put the same system in the table
            # twice under different names.
            logger.info("skipping training artefact %s", path.name)
            continue
        out[name] = [Prediction(**row) for row in blob["predictions"]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mis-voiced audio per minute, from saved predictions")
    parser.add_argument("split", nargs="?", default="dev")
    parser.add_argument("--reference", default="nearest-mention",
                        help="system every delta is measured against")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    novels = load_split(args.split)
    systems = load_reports(args.split)
    if not systems:
        raise SystemExit(f"no saved predictions for split '{args.split}' in "
                         f"{RESULTS_DIR}; run evaluate.py first")

    reports = {n: measure(n, preds, novels) for n, preds in systems.items()}
    order = sorted(reports, key=lambda n: -reports[n].sec_per_audio_minute)

    _, novel_words = _quote_words(novels)
    total = sum(t for _, t in novel_words.values())
    dialogue = sum(d for d, _ in novel_words.values())
    print(f"extrinsic audio metric on {args.split}: {len(novels)} novels, "
          f"{seconds(total)/3600:.1f} h of audio at {SPEAKING_RATE_WPM:.0f} wpm, "
          f"{100*dialogue/total:.0f}% of it dialogue\n")

    hdr = (f"{'system':22s} {'s/audio min':>12s} {'95% CI':>16s} "
           f"{'s/dialogue min':>15s} {'h mis-voiced':>13s} {'length bias':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for name in order:
        r = reports[name]
        print(f"{name:22s} {r.sec_per_audio_minute:>11.2f}s "
              f"{f'[{r.ci[0]:.2f}, {r.ci[1]:.2f}]':>16s} "
              f"{r.sec_per_dialogue_minute:>14.2f}s "
              f"{r.misvoiced_hours:>12.2f}h {r.length_bias:>12.2f}")

    if args.reference in reports:
        ref = reports[args.reference]
        print(f"\npaired bootstrap vs {args.reference} "
              f"(negative = less mis-voiced audio):")
        for name in order:
            if name == args.reference:
                continue
            d = paired_seconds_delta(reports[name], ref)
            lo, hi = d["ci"]
            mark = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {name:22s} {d['delta']:+6.2f}s "
                  f"[{lo:+5.2f}, {hi:+5.2f}]{mark}  p={d['p']:.3f}")
        print("\n* interval excludes zero")

    print("\nshare of mis-voiced audio by quote type "
          "(and duration-weighted error rate within type):")
    thdr = f"{'system':22s} " + " ".join(f"{t:>22s}" for t in QUOTE_TYPES)
    print(thdr)
    print("-" * len(thdr))
    for name in order:
        r = reports[name]
        cells = [f"{100*r.misvoiced_share[t]:>9.1f}% ({100*r.rate_by_type[t]:>5.1f}%)"
                 for t in QUOTE_TYPES]
        print(f"{name:22s} " + " ".join(cells))

    out = RESULTS_DIR / f"extrinsic_{args.split}.json"
    out.write_text(json.dumps({
        "split": args.split,
        "speaking_rate_wpm": SPEAKING_RATE_WPM,
        "n_novels": len(novels),
        "total_words": total,
        "dialogue_words": dialogue,
        "audio_hours": seconds(total) / 3600.0,
        "systems": {
            n: {
                "sec_per_audio_minute": r.sec_per_audio_minute,
                "sec_per_dialogue_minute": r.sec_per_dialogue_minute,
                "ci": list(r.ci),
                "misvoiced_words": r.misvoiced_words,
                "misvoiced_hours": r.misvoiced_hours,
                "duration_error_rate": r.duration_error_rate,
                "quote_error_rate": r.quote_error_rate,
                "length_bias": r.length_bias,
                "misvoiced_share_by_type": r.misvoiced_share,
                "rate_by_type": r.rate_by_type,
                "per_novel": {b: list(v) for b, v in r.per_novel.items()},
            }
            for n, r in reports.items()
        },
        "paired_vs_reference": {
            n: {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in paired_seconds_delta(r, reports[args.reference]).items()}
            for n, r in reports.items()
            if args.reference in reports and n != args.reference
        },
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

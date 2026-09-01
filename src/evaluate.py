"""Evaluation harness and the heuristic baselines every model must beat.

The single number this project refuses to report on its own is pooled accuracy.
A system that solves explicit quotes and guesses everywhere else scores well on
a corpus that is 30% explicit while being useless for the task that motivates
the work, so every result here is broken out by quote type.

Confidence intervals bootstrap over NOVELS, not quotations. Quotes inside one
book are heavily dependent -- a single mishandled conversation contributes
dozens of correlated errors, and the cast, narrator, and style are shared
throughout -- so resampling quotations would report an interval several times
narrower than the true uncertainty over books. The question this project asks
is "does it work on a novel it has not seen", and the novel is the unit.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from candidates import Candidate, enumerate_candidates
from config import (
    CONFIDENCE,
    CONVERSATION_GAP_CHARS,
    N_BOOTSTRAP,
    QUOTE_TYPES,
    RESULTS_DIR,
    SEED,
)
from pdnc import Novel, Quote

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- results


@dataclass
class Prediction:
    """One system output, kept alongside enough context to do error analysis."""

    book: str
    qid: str
    quote_type: str
    gold: str
    pred: str | None
    n_candidates: int
    gold_in_candidates: bool
    # Gold addressees, carried so error analysis can ask the question that
    # matters most here: when the system is wrong, did it name the person being
    # spoken TO? That is the characteristic failure of proximity-based
    # attribution, since an addressee is usually named right beside the quote.
    addressees: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.pred is not None and self.pred == self.gold

    @property
    def named_addressee(self) -> bool:
        return (
            not self.correct
            and self.pred is not None
            and self.pred in self.addressees
        )


@dataclass
class Report:
    """Accuracy of one system, sliced the ways that matter."""

    name: str
    predictions: list[Prediction]
    accuracy: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    per_novel: dict[str, float] = field(default_factory=dict)
    coverage: float = 0.0

    def __str__(self) -> str:
        lines = [f"{self.name}"]
        for key in ("all", *QUOTE_TYPES):
            if key not in self.accuracy:
                continue
            lo, hi = self.ci.get(key, (float("nan"), float("nan")))
            lines.append(
                f"  {key:<10s} {100*self.accuracy[key]:5.1f}%  "
                f"[{100*lo:4.1f}, {100*hi:4.1f}]  n={self.counts[key]:,}"
            )
        lines.append(f"  ceiling    {100*self.coverage:5.1f}%  (gold in candidate set)")
        return "\n".join(lines)


def _bootstrap_ci(
    per_novel_hits: dict[str, tuple[int, int]],
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int = SEED,
) -> tuple[float, float]:
    """Percentile bootstrap over novels.

    Each resample draws whole novels with replacement and pools their quotes,
    so a book that happens to be easy is either fully in or fully out. Novels
    with no quotes of the slice being measured are excluded first, otherwise
    resamples that draw only such books produce 0/0.
    """
    books = [b for b, (_, n) in per_novel_hits.items() if n > 0]
    if len(books) < 2:
        return (float("nan"), float("nan"))

    rng = random.Random(seed)
    accs = []
    for _ in range(n_boot):
        picked = [books[rng.randrange(len(books))] for _ in books]
        hits = sum(per_novel_hits[b][0] for b in picked)
        total = sum(per_novel_hits[b][1] for b in picked)
        if total:
            accs.append(hits / total)

    if not accs:
        return (float("nan"), float("nan"))
    accs.sort()
    tail = (1.0 - confidence) / 2.0
    lo = accs[int(tail * (len(accs) - 1))]
    hi = accs[int((1.0 - tail) * (len(accs) - 1))]
    return (lo, hi)


def paired_delta(
    a: list[Prediction],
    b: list[Prediction],
    key: str = "all",
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int = SEED,
) -> dict:
    """Bootstrap the DIFFERENCE between two systems on the same novels.

    Comparing two independently-bootstrapped intervals is the wrong test and
    badly underpowered here. Book difficulty dominates both systems' scores --
    dev accuracy ranges 34% to 78% across six novels -- so each system's own
    interval is enormous, and the two overlap even when one system is better on
    every single book. Measured that way the ranker's +23 points over the
    baseline looks insignificant, which is an artefact of the test, not a fact
    about the systems.

    Resampling novels *once* per iteration and scoring both systems on that
    same resample cancels the shared book effect, leaving the interval to
    describe the quantity actually in question: how much better is A than B.

    ``p`` is the one-sided bootstrap proportion of resamples in which A failed
    to beat B -- a direct answer to "could this gap be noise?", not a p-value
    from a parametric test whose assumptions nobody checked.
    """
    index_b = {(p.book, p.qid): p for p in b}
    pairs = [(pa, index_b[(pa.book, pa.qid)]) for pa in a
             if (pa.book, pa.qid) in index_b]
    if key != "all":
        pairs = [(x, y) for x, y in pairs if x.quote_type == key]
    if not pairs:
        return {"delta": float("nan"), "ci": (float("nan"), float("nan")),
                "p": float("nan"), "n": 0}

    per_novel: dict[str, list[int]] = {}
    for x, y in pairs:
        cell = per_novel.setdefault(x.book, [0, 0, 0])
        cell[0] += x.correct
        cell[1] += y.correct
        cell[2] += 1

    books = list(per_novel)
    observed = (
        sum(c[0] for c in per_novel.values()) / sum(c[2] for c in per_novel.values())
        - sum(c[1] for c in per_novel.values()) / sum(c[2] for c in per_novel.values())
    )

    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        picked = [books[rng.randrange(len(books))] for _ in books]
        ha = sum(per_novel[bk][0] for bk in picked)
        hb = sum(per_novel[bk][1] for bk in picked)
        n = sum(per_novel[bk][2] for bk in picked)
        if n:
            deltas.append(ha / n - hb / n)

    deltas.sort()
    tail = (1.0 - confidence) / 2.0
    lo = deltas[int(tail * (len(deltas) - 1))]
    hi = deltas[int((1.0 - tail) * (len(deltas) - 1))]
    p = sum(d <= 0 for d in deltas) / len(deltas)
    return {"delta": observed, "ci": (lo, hi), "p": p, "n": len(pairs)}


def error_breakdown(predictions: list[Prediction]) -> dict:
    """Where the errors go, split by whether they were winnable at all.

    An error with the gold speaker absent from the candidate set is a candidate
    failure, not a ranking failure, and lumping the two together would credit
    the ranker for mistakes it had no chance to avoid.
    """
    total = len(predictions) or 1
    wrong = [p for p in predictions if not p.correct]
    unreachable = [p for p in wrong if not p.gold_in_candidates]
    rankable = [p for p in wrong if p.gold_in_candidates]
    addressee = [p for p in rankable if p.named_addressee]

    return {
        "n": len(predictions),
        "correct": sum(p.correct for p in predictions) / total,
        "wrong": len(wrong) / total,
        "wrong_unreachable": len(unreachable) / total,
        "wrong_rankable": len(rankable) / total,
        "wrong_named_addressee": len(addressee) / total,
        # Of the errors the ranker could in principle have avoided, what share
        # picked the addressee instead of the speaker?
        "addressee_share_of_rankable": len(addressee) / max(len(rankable), 1),
    }


def score(name: str, predictions: list[Prediction]) -> Report:
    """Turn raw predictions into the sliced, interval-bearing report."""
    hits: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    counts = Counter()
    correct = Counter()

    for p in predictions:
        for key in ("all", p.quote_type):
            counts[key] += 1
            correct[key] += p.correct
            cell = hits[key][p.book]
            cell[0] += p.correct
            cell[1] += 1

    report = Report(name=name, predictions=predictions)
    for key in ("all", *QUOTE_TYPES):
        if not counts[key]:
            continue
        report.accuracy[key] = correct[key] / counts[key]
        report.counts[key] = counts[key]
        report.ci[key] = _bootstrap_ci(
            {b: (c[0], c[1]) for b, c in hits[key].items()}
        )

    for book, cell in hits["all"].items():
        report.per_novel[book] = cell[0] / max(cell[1], 1)

    report.coverage = sum(p.gold_in_candidates for p in predictions) / max(
        len(predictions), 1
    )
    return report


def save(report: Report, filename: str | None = None) -> None:
    """Write accuracy and every prediction, so error analysis needs no rerun."""
    path = RESULTS_DIR / (filename or f"{report.name.replace(' ', '_')}.json")
    path.write_text(
        json.dumps(
            {
                "name": report.name,
                "accuracy": report.accuracy,
                "counts": report.counts,
                "ci": {k: list(v) for k, v in report.ci.items()},
                "per_novel": report.per_novel,
                "coverage": report.coverage,
                "predictions": [vars(p) for p in report.predictions],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", path)


def load_report(filename: str, name: str | None = None) -> Report:
    """Re-score a system from its saved per-quote predictions.

    Intervals and slices are recomputed from the predictions rather than read
    back from the file, so a report loaded here is identical to one produced
    in-process. This is what lets the comparison table include Model 2 without
    holding an 8B decoder and the ranker on the same 12GB card: each model
    writes its predictions once, and the table is assembled from those.
    """
    blob = json.loads((RESULTS_DIR / filename).read_text(encoding="utf-8"))
    predictions = [Prediction(**p) for p in blob["predictions"]]
    return score(name or blob["name"], predictions)


# ---------------------------------------------------------------- baselines


# Reseeded by run_baseline so the chance floor is reproducible. A predictor
# reaching for the global `random` would shift by half a point between runs
# depending on what else had drawn from it.
_RNG = random.Random(SEED)


def predict_random(candidates: list[Candidate], _q=None, _h=None) -> str | None:
    """Uniform choice among candidates -- the chance floor.

    Without this the reader cannot tell whether 27% on implicit quotes is a
    weak result or essentially no result. With ~5 candidates per quote, chance
    is around 20%, so the nearest-mention baseline's implicit accuracy turns
    out to sit only a few points above guessing. That framing is the point of
    the project and it needs a measured floor, not an assumed one.
    """
    if not candidates:
        return None
    return _RNG.choice(candidates).character.name


def predict_nearest(candidates: list[Candidate], _q=None, _h=None) -> str | None:
    """Nearest mention wins.

    The standard trivial baseline: whoever is named closest to the quotation is
    the speaker. It is strong on explicit quotes, where the tag sits inches
    away, and it is close to a coin flip on implicit ones, where the nearest
    name is often the *addressee* or a third party being discussed.
    """
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.nearest.distance).character.name


def predict_alternation(candidates: list[Candidate], quote: Quote,
                        history: list[tuple]) -> str | None:
    """In a two-party exchange, the speaker is whoever spoke two turns ago.

    This is the obvious heuristic for untagged dialogue and the one a reader
    will ask about first, so leaving it out would flatter the model. It is
    evaluated honestly: the history it consults is this baseline's OWN previous
    predictions, never the gold speakers, so errors propagate exactly as they
    would at inference time.

    A conversation is broken by more than ``CONVERSATION_GAP_CHARS`` of
    narration; across a scene break "two turns ago" means nothing. Where the
    rule cannot fire it falls back to nearest mention, so the comparison
    isolates the alternation signal rather than testing coverage.
    """
    if not candidates:
        return None

    turns = []
    last_end = quote.start
    for prev_quote, prev_pred in reversed(history):
        if last_end - prev_quote.end > CONVERSATION_GAP_CHARS:
            break
        turns.append(prev_pred)
        last_end = prev_quote.start
        if len(turns) >= 2:
            break

    if len(turns) >= 2 and turns[1] is not None:
        # turns[0] is the previous speaker, turns[1] the one before -- the
        # alternation partner. Only trust it if that name is still a candidate.
        names = {c.character.name for c in candidates}
        if turns[1] in names and turns[1] != turns[0]:
            return turns[1]

    return predict_nearest(candidates)


def predict_nearest_with_verb(candidates: list[Candidate], _q=None,
                              _h=None) -> str | None:
    """Nearest mention that sits next to a speech verb, else nearest mention.

    One rule smarter than the baseline above, and the cheapest approximation of
    Muzny et al.'s (2017) first sieve: prefer a name that is actually part of a
    speech tag. This separates "the model learned to read speech tags" from
    "the model learned anything else", which is the comparison the per-type
    breakdown exists to make.
    """
    if not candidates:
        return None
    tagged = [c for c in candidates if c.has_tagged_mention]
    pool = tagged or candidates
    return min(pool, key=lambda c: c.nearest.distance).character.name


def predict_most_frequent(candidates: list[Candidate], _q=None,
                          _h=None) -> str | None:
    """Whoever is named most often in the window.

    A deliberately different kind of wrong: it ignores position entirely. Where
    this beats nearest-mention, proximity is misleading rather than absent.
    """
    if not candidates:
        return None
    return max(
        candidates, key=lambda c: (c.n_mentions, -c.nearest.distance)
    ).character.name


def _candidates_for(novels: list[Novel]) -> dict:
    """Enumerate candidates once and share them across every baseline.

    Candidate enumeration runs a large alternation regex over ~3,300 characters
    per quote. Recomputing it per baseline multiplied the cost of the whole
    comparison by the number of systems for no reason.
    """
    return {
        (n.name, q.qid): enumerate_candidates(n, q)
        for n in novels for q in n.quotes
    }


def run_baseline(novels: list[Novel], predictor, name: str,
                 cache: dict | None = None, seed: int = SEED) -> Report:
    """Apply one predictor to every quote, in reading order within each novel.

    Order matters: stateful predictors see only what precedes the quote they
    are attributing, and only their own predictions -- never the gold labels.
    """
    cache = cache if cache is not None else _candidates_for(novels)
    _RNG.seed(seed)
    predictions: list[Prediction] = []

    for novel in novels:
        history: list[tuple] = []
        for q in novel.quotes:                  # already sorted by position
            cands = cache[(novel.name, q.qid)]
            pred = predictor(cands, q, history)
            history.append((q, pred))
            predictions.append(
                Prediction(
                    book=novel.name,
                    qid=q.qid,
                    quote_type=q.quote_type,
                    gold=q.speaker,
                    pred=pred,
                    n_candidates=len(cands),
                    gold_in_candidates=any(c.is_gold for c in cands),
                    addressees=list(q.addressees),
                )
            )
    return score(name, predictions)


BASELINES = {
    "random-candidate": predict_random,
    "most-frequent": predict_most_frequent,
    "nearest-mention": predict_nearest,
    "nearest+speech-verb": predict_nearest_with_verb,
    "alternation": predict_alternation,
}


if __name__ == "__main__":
    import sys

    from pdnc import load_split

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    novels = load_split(split)
    print(f"baselines on {split} ({len(novels)} novels, "
          f"{sum(len(n.quotes) for n in novels):,} quotes)\n")

    cache = _candidates_for(novels)
    reports = {}
    for name, fn in BASELINES.items():
        report = run_baseline(novels, fn, name, cache=cache)
        reports[name] = report
        save(report, f"baseline_{name}_{split}.json")

    # --- accuracy table
    hdr = f"{'baseline':22s} {'all':>8s} {'explicit':>10s} {'anaphoric':>10s} {'implicit':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in reports.items():
        print(
            f"{name:22s} {100*r.accuracy['all']:>7.1f}% "
            f"{100*r.accuracy['explicit']:>9.1f}% "
            f"{100*r.accuracy['anaphoric']:>9.1f}% "
            f"{100*r.accuracy['implicit']:>8.1f}%"
        )
    print("-" * len(hdr))

    # Report a winner PER TYPE. Ranking baselines by pooled accuracy would
    # commit the exact error this project exists to point out: on this corpus
    # most-frequent beats nearest-mention on both hard types while losing
    # heavily overall, so a single "best baseline" hides the real picture.
    print("\nstrongest baseline by quote type:")
    for key in ("all", *QUOTE_TYPES):
        best = max(reports.values(), key=lambda r: r.accuracy.get(key, 0))
        floor = reports["random-candidate"].accuracy.get(key, 0)
        print(
            f"  {key:<10s} {best.name:22s} {100*best.accuracy[key]:5.1f}%  "
            f"(chance {100*floor:4.1f}%, +{100*(best.accuracy[key]-floor):4.1f} over chance)"
        )

    # --- paired comparison against the strongest overall heuristic
    ref = reports["nearest-mention"]
    print(f"\npaired bootstrap vs {ref.name} (resampling novels, 95% CI):")
    for name, r in reports.items():
        if name == ref.name:
            continue
        d = paired_delta(r.predictions, ref.predictions, "all")
        lo, hi = d["ci"]
        flag = "" if lo > 0 or hi < 0 else "   (interval spans 0)"
        print(
            f"  {name:22s} {100*d['delta']:+6.1f} pts  "
            f"[{100*lo:+5.1f}, {100*hi:+5.1f}]  p={d['p']:.3f}{flag}"
        )

    # --- where the errors go
    print("\nerror breakdown (nearest-mention):")
    e = error_breakdown(ref.predictions)
    print(f"  correct                     {100*e['correct']:5.1f}%")
    print(f"  wrong, gold not a candidate {100*e['wrong_unreachable']:5.1f}%")
    print(f"  wrong, gold was available   {100*e['wrong_rankable']:5.1f}%")
    print(f"    of which named addressee  {100*e['addressee_share_of_rankable']:5.1f}% "
          f"of avoidable errors")

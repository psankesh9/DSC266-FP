"""Head-to-head comparison of every system on one split.

This is the script that produces the report's results table. It exists
separately from ``evaluate.py`` because it needs a trained checkpoint, and the
baselines must stay runnable without one.

Every comparison is paired: the same resampled novels score both systems, so
the interval describes the difference rather than the difference of two
intervals. On six or seven held-out novels that distinction decides whether the
headline result is reportable at all.
"""

from __future__ import annotations

import argparse
import json
import logging

import torch

from config import BATCH_SIZE, MODEL_DIR, QUOTE_TYPES, RESULTS_DIR
from evaluate import (
    BASELINES,
    _candidates_for,
    error_breakdown,
    load_report,
    paired_delta,
    run_baseline,
    save,
    score,
)
from pdnc import load_split

logger = logging.getLogger(__name__)


def load_ranker(path):
    """Restore a trained ranker, its tokenizer, and the window it was fit under.

    The window is read back from the checkpoint rather than taken from
    ``config``: a model trained on a clamped context must be evaluated on the
    same clamped context, or the ablation silently measures a train/test
    mismatch instead of the effect of context. Older checkpoints predate the
    field and are assumed to be the pre-swap RoBERTa regime.
    """
    from transformers import AutoTokenizer

    from device import get_device
    from ranker import SpanRanker, Window

    blob = torch.load(path, map_location="cpu", weights_only=False)
    encoder_name = blob.get("encoder", "roberta-base")
    window = Window(**blob["window"]) if "window" in blob else Window(900, 400, 512)
    # Same reasoning as the window: a model fit with the hand features zeroed
    # must be evaluated with them zeroed. Feeding real features to a head that
    # only ever saw zeros scores the ablation against a distribution it was
    # never trained on, which is a bug rather than an ablation.
    use_features = blob.get("use_features", True)
    model = SpanRanker(encoder_name, use_features=use_features)
    model.load_state_dict(blob["state_dict"])
    model.to(get_device()).eval()
    return model, AutoTokenizer.from_pretrained(encoder_name), window


def main() -> None:
    parser = argparse.ArgumentParser(description="compare all systems")
    parser.add_argument("split", nargs="?", default="dev")
    parser.add_argument("--checkpoint", default="ranker_e8.pt")
    parser.add_argument("--reference", default="nearest-mention",
                        help="system every delta is measured against")
    parser.add_argument("--include", nargs="*", default=[],
                        help="extra systems to fold into the table, as "
                             "results filenames written by evaluate.save() "
                             "(e.g. generator-zeroshot_dev.json). Use this for "
                             "Model 2 rather than loading an 8B decoder here.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    novels = load_split(args.split)
    n_quotes = sum(len(n.quotes) for n in novels)
    print(f"comparison on {args.split}: {len(novels)} novels, {n_quotes:,} quotes\n")

    cache = _candidates_for(novels)
    reports = {
        name: run_baseline(novels, fn, name, cache=cache)
        for name, fn in BASELINES.items()
    }

    ckpt = MODEL_DIR / args.checkpoint
    if ckpt.exists():
        from device import get_device
        from ranker import make_loader, predict, to_predictions

        model, tokenizer, window = load_ranker(ckpt)
        print(f"ranker : {ckpt.name}, context -{window.chars_before}/"
              f"+{window.chars_after} chars, {window.max_length} tokens"
              + ("" if model.use_features else ", features ZEROED") + "\n")
        loader = make_loader(novels, tokenizer, BATCH_SIZE, window=window)
        rows = predict(model, loader, get_device())
        reports["ranker"] = score("ranker", to_predictions(rows))
        # Persist the per-quote records too: extrinsic.py re-scores exactly
        # these predictions by duration, and must not have to reload the GPU
        # model to do it.
        save(reports["ranker"], f"ranker_{args.split}.json")
    else:
        print(f"(no checkpoint at {ckpt}; baselines only)\n")

    for filename in args.include:
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"(skipping {filename}: not found)")
            continue
        extra = load_report(filename)
        # Guard against silently comparing systems scored on different splits:
        # every system in one table must have attributed the same quotations.
        n_ref = len(reports[args.reference].predictions)
        if len(extra.predictions) != n_ref:
            print(f"(skipping {filename}: {len(extra.predictions):,} predictions "
                  f"but this split has {n_ref:,})")
            continue
        reports[extra.name] = extra

    # ---------------------------------------------------------- accuracy
    hdr = (f"{'system':22s} {'all':>8s} {'explicit':>10s} {'anaphoric':>10s} "
           f"{'implicit':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in reports.items():
        print(
            f"{name:22s} {100*r.accuracy['all']:>7.1f}% "
            f"{100*r.accuracy['explicit']:>9.1f}% "
            f"{100*r.accuracy['anaphoric']:>9.1f}% "
            f"{100*r.accuracy['implicit']:>8.1f}%"
        )
    ceiling = reports[args.reference].coverage
    print("-" * len(hdr))
    print(f"{'ceiling':22s} {100*ceiling:>7.1f}%   (gold speaker in candidate set)")

    # ------------------------------------------------- paired comparisons
    ref = reports[args.reference]
    print(f"\npaired bootstrap vs {args.reference}, by quote type "
          f"(95% CI over novels):")
    sub = f"{'system':22s} " + " ".join(f"{k:>22s}" for k in ("all", *QUOTE_TYPES))
    print(sub)
    print("-" * len(sub))
    for name, r in reports.items():
        if name == args.reference:
            continue
        cells = []
        for key in ("all", *QUOTE_TYPES):
            d = paired_delta(r.predictions, ref.predictions, key)
            lo, hi = d["ci"]
            mark = "*" if (lo > 0 or hi < 0) else " "
            cells.append(f"{100*d['delta']:+6.1f}[{100*lo:+5.1f},{100*hi:+5.1f}]{mark}")
        print(f"{name:22s} " + " ".join(cells))
    print("\n* interval excludes zero")

    # ------------------------------------------------------ error profile
    print("\nerror profile (share of all quotes):")
    ehdr = (f"{'system':22s} {'correct':>9s} {'unreachable':>12s} "
            f"{'avoidable':>10s} {'addressee':>11s}")
    print(ehdr)
    print("-" * len(ehdr))
    for name, r in reports.items():
        e = error_breakdown(r.predictions)
        print(
            f"{name:22s} {100*e['correct']:>8.1f}% "
            f"{100*e['wrong_unreachable']:>11.1f}% "
            f"{100*e['wrong_rankable']:>9.1f}% "
            f"{100*e['addressee_share_of_rankable']:>10.1f}%"
        )
    print("\naddressee column: share of AVOIDABLE errors that named the "
          "person spoken to")

    out = RESULTS_DIR / f"comparison_{args.split}.json"
    out.write_text(json.dumps({
        "split": args.split,
        "n_novels": len(novels),
        "ceiling": ceiling,
        "accuracy": {n: r.accuracy for n, r in reports.items()},
        "ci": {n: {k: list(v) for k, v in r.ci.items()} for n, r in reports.items()},
        "per_novel": {n: r.per_novel for n, r in reports.items()},
        "paired_vs_reference": {
            n: {k: paired_delta(r.predictions, ref.predictions, k)
                for k in ("all", *QUOTE_TYPES)}
            for n, r in reports.items() if n != args.reference
        },
        "errors": {n: error_breakdown(r.predictions) for n, r in reports.items()},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

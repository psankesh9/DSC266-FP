"""Paired bootstrap for the ablation contrasts themselves.

compare.py measures every system against nearest-mention. The ablation claims
are differences between two rankers, so they need their own paired intervals.

Explicit quotes get their own column because several of the claims in the
report are specifically about *where* a gain lands: the window and backbone
arguments both predict a large effect on anaphoric/implicit quotes and
approximately none on the quotes whose speaker is named right beside them.
A pooled number cannot distinguish "helps everywhere a little" from that.
"""
from evaluate import load_report, paired_delta
from config import RESULTS_DIR

SLICES = ("all", "explicit", "anaphoric", "implicit")

CONTRASTS = [
    ("ranker_dev.json",        "ranker_clamp_dev.json",  "window: -2500/+800 @1152 vs clamped 512"),
    ("ranker_clamp_dev.json",  "ranker_legacy_dev.json", "backbone: ModernBERT vs RoBERTa (window matched)"),
    ("ranker_dev.json",        "ranker_nofeat_dev.json", "hand features: on vs zeroed"),
    ("ranker_dev.json",        "ranker_large_dev.json",  "capacity: base vs large (CONFOUNDED)"),
    # The row the confounded one cannot license. Same 8 epochs, same 2e-5, and
    # batch 1 x grad-accum 8 reproduces base's effective batch of 8, so the
    # encoder size is the only thing that differs.
    ("ranker_dev.json",        "ranker_large_clean_dev.json", "capacity: base vs large (batch MATCHED)"),
    ("generator-lora_dev.json","ranker_dev.json",        "Qwen3-8B LoRA vs ModernBERT ranker"),
    ("generator-lora_dev.json","generator-zeroshot_dev.json", "LoRA vs zero-shot (same 8B model)"),
]

WIDTH = 52
header = f"{'contrast':{WIDTH}s} " + " ".join(f"{s:>20s}" for s in SLICES)
print(header)
print("-" * len(header))

missing = []
for a, b, label in CONTRASTS:
    absent = [f for f in (a, b) if not (RESULTS_DIR / f).exists()]
    if absent:
        # A contrast whose run has not landed yet must not silently vanish from
        # the table -- that is how a placeholder becomes a forgotten claim.
        missing.append((label, absent))
        print(f"{label:{WIDTH}s} " + f"{'-- not yet run --':>20s}")
        continue
    ra, rb = load_report(a), load_report(b)
    cells = []
    for key in SLICES:
        d = paired_delta(ra.predictions, rb.predictions, key)
        lo, hi = d["ci"]
        mark = "*" if (lo > 0 or hi < 0) else " "
        cells.append(f"{100*d['delta']:+6.1f}[{100*lo:+5.1f},{100*hi:+5.1f}]{mark}")
    print(f"{label:{WIDTH}s} " + " ".join(cells))

print("\n* interval excludes zero (paired over the 6 dev novels)")
for label, absent in missing:
    print(f"  pending: {label} -- missing {', '.join(absent)}")

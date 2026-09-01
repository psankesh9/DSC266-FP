"""Report figures for the ceiling and baseline analyses.

Every figure here answers one question, and the questions are the ones the
report's argument actually turns on:

  fig1  How much of the task is left once speech tags are read?
  fig2  What does widening the candidate window buy, and what does it cost?
  fig3  Why are the confidence intervals bootstrapped over novels?
  fig4  What is the derived-name gazetteer rule worth?

Output is PNG at 300 dpi for drafting plus PDF vector for the final document.
Figures are light-mode only: they are going into a printed report, not a page
with a theme toggle, so a dark variant would be dead weight.

Palette and mark conventions follow the project's data-viz standard. The three
quote types take categorical slots 1-3, which clear the colourblind separation
gate on all pairs (worst deutan dE 9.2, normal-vision 24.0). Slot 3 (aqua) sits
below 3:1 against the surface, so every mark that uses it is directly labelled
rather than relying on the fill alone to be read.
"""

from __future__ import annotations

import json
import re
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from candidates import oracle_recall
from config import CANDIDATE_WINDOW_SWEEP, PLOTS_DIR, QUOTE_TYPES, RESULTS_DIR
from evaluate import BASELINES, run_baseline
from pdnc import load_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- style

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
BLUE_LIGHT = "#9ec5f4"
BLUE_PALE = "#cde2fb"

TYPE_COLOR = {"explicit": BLUE, "anaphoric": ORANGE, "implicit": AQUA}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "600",
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
})


def _despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
        if side in keep:
            ax.spines[side].set_linewidth(0.8)


def _pretty(folder: str) -> str:
    """PDNC folder names are CamelCase; axis labels should not be."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", folder)


def _save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = PLOTS_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  wrote {stem}.png / .pdf")


# ---------------------------------------------------------------- data


def gather(split: str = "dev", rebuild: bool = False) -> dict:
    """Compute (or reload) every number the figures need."""
    cache = RESULTS_DIR / f"figure_data_{split}.json"
    if cache.exists() and not rebuild:
        return json.loads(cache.read_text(encoding="utf-8"))

    novels = load_split(split)
    data: dict = {"split": split, "n_novels": len(novels)}

    print("  baselines...")
    data["baselines"] = {}
    for name, fn in BASELINES.items():
        r = run_baseline(novels, fn, name)
        data["baselines"][name] = {
            "accuracy": r.accuracy,
            "ci": {k: list(v) for k, v in r.ci.items()},
            "counts": r.counts,
            "per_novel": r.per_novel,
            "coverage": r.coverage,
        }

    print("  window sweep...")
    data["sweep"] = []
    for before, after in CANDIDATE_WINDOW_SWEEP:
        res = oracle_recall(novels, before, after)
        data["sweep"].append({
            "before": before,
            "after": after,
            "recall": dict(res["recall"]),
            "mean_candidates": dict(res["mean_candidates"]),
        })

    print("  gazetteer ablation...")
    data["ablation"] = {}
    for derive, label in ((False, "PDNC aliases only"), (True, "+ derived names")):
        res = oracle_recall(novels, derive=derive)
        data["ablation"][label] = {
            "recall": dict(res["recall"]),
            "mean_candidates": res["mean_candidates"]["all"],
        }

    # Per-novel ceiling, for the novel-variation figure.
    res = oracle_recall(novels)
    data["per_novel_ceiling"] = {n.name: res["recall"][n.name] for n in novels}
    data["novel_person"] = {n.name: n.person for n in novels}

    cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


# ---------------------------------------------------------------- figures


def fig1_headroom(data: dict) -> None:
    """Baseline accuracy against the candidate ceiling, per quote type.

    A dumbbell rather than paired bars: the quantity the report argues about is
    the *gap*, and a dumbbell draws the gap as the mark itself instead of
    leaving the reader to subtract two bar heights.
    """
    b = data["baselines"]["nearest-mention"]
    ceilings = data["ablation"]["+ derived names"]["recall"]

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    rows = list(QUOTE_TYPES)[::-1]          # explicit at top
    ys = range(len(rows))

    for y, qtype in zip(ys, rows):
        acc = b["accuracy"][qtype]
        ceil = ceilings[qtype]
        lo, hi = b["ci"][qtype]

        # headroom band: what modelling has left to win
        ax.plot([acc, ceil], [y, y], color=BLUE_PALE, linewidth=9,
                solid_capstyle="round", zorder=1)
        # confidence interval on the achieved number
        ax.plot([lo, hi], [y, y], color=BLUE, linewidth=1.6,
                solid_capstyle="butt", zorder=3, alpha=0.55)
        ax.scatter([ceil], [y], s=64, facecolor=SURFACE, edgecolor=BLUE_LIGHT,
                   linewidth=2.0, zorder=4)
        ax.scatter([acc], [y], s=70, color=BLUE, zorder=5,
                   edgecolor=SURFACE, linewidth=1.5)

        ax.annotate(f"{100*acc:.0f}%", (acc, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9, color=INK,
                    fontweight="600")
        ax.annotate(f"{100*ceil:.0f}%", (ceil, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9, color=INK_2)
        ax.annotate(f"+{100*(ceil-acc):.0f} pts", ((acc + ceil) / 2, y),
                    textcoords="offset points", xytext=(0, -15), ha="center",
                    fontsize=8, color=MUTED, style="italic")

    ax.set_yticks(list(ys))
    ax.set_yticklabels([q.capitalize() for q in rows], fontsize=9.5, color=INK)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.65, len(rows) - 0.35)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("speaker accuracy")
    ax.grid(axis="y", visible=False)
    _despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)

    ax.set_title(
        "Reading speech tags solves the easy third of the task and little else",
        loc="left", pad=30,
    )
    ax.text(
        0.0, 1.045,
        f"Nearest-mention baseline vs candidate-set ceiling · PDNC {data['split']} "
        f"split, {data['n_novels']} held-out novels",
        fontsize=8.5, color=MUTED, ha="left", transform=ax.transAxes,
    )
    # Legend below the plot: the rows span most of the width, and every
    # in-axes corner collided with either a mark or its direct label.
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=8,
                   markeredgecolor=SURFACE, label="nearest-mention baseline"),
            Line2D([], [], marker="o", color=BLUE_LIGHT, linestyle="",
                   markersize=8, markerfacecolor=SURFACE, markeredgewidth=2,
                   label="candidate-set ceiling"),
            Line2D([], [], color=BLUE, linewidth=1.6, alpha=0.55,
                   label="95% CI (novel bootstrap)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=3,
        handletextpad=0.6, columnspacing=1.6,
    )
    _save(fig, "fig1_headroom_by_quote_type")


def fig2_window_sweep(data: dict) -> None:
    """Ceiling against window width, with the cost panel underneath.

    Two measures on one x-axis, deliberately NOT on two y-scales: recall and
    candidates-per-quote share the window setting but nothing else, so they get
    stacked panels instead of a dual axis.
    """
    sweep = data["sweep"]
    labels = [f"−{s['before']}\n+{s['after']}" for s in sweep]
    xs = list(range(len(sweep)))

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.0, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.18},
    )

    for qtype in QUOTE_TYPES:
        ys = [s["recall"][qtype] for s in sweep]
        ax.plot(xs, ys, color=TYPE_COLOR[qtype], linewidth=2.0,
                marker="o", markersize=5.5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=3)
        ax.annotate(
            f"{qtype}  {100*ys[-1]:.0f}%", (xs[-1], ys[-1]),
            textcoords="offset points", xytext=(8, 0), va="center",
            fontsize=8.5, color=TYPE_COLOR[qtype], fontweight="600",
        )

    ax.set_ylim(0.5, 1.0)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["50%", "60%", "70%", "80%", "90%", "100%"])
    ax.set_ylabel("gold speaker in candidate set")
    ax.set_xlim(-0.3, len(sweep) - 0.3)
    _despine(ax, keep=("bottom",))
    ax.set_title(
        "A wider net keeps paying on implicit quotes, and keeps costing candidates",
        loc="left", pad=22,
    )
    fig.text(
        0.0, 1.02,
        f"Candidate-set ceiling by narration window · PDNC {data['split']} split",
        fontsize=8.5, color=MUTED, ha="left", transform=ax.transAxes,
    )

    cands = [s["mean_candidates"]["all"] for s in sweep]
    ax2.bar(xs, cands, width=0.45, color=BLUE_LIGHT, zorder=3)
    for x, c in zip(xs, cands):
        ax2.annotate(f"{c:.1f}", (x, c), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=8, color=INK_2)
    ax2.set_ylabel("candidates\nper quote", fontsize=8.5)
    ax2.set_ylim(0, max(cands) * 1.35)
    ax2.set_yticks([])
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_xlabel("narration window before / after the quotation (characters)")
    ax2.grid(visible=False)
    _despine(ax2, keep=("bottom",))

    _save(fig, "fig2_candidate_window_sweep")


def fig3_novel_variation(data: dict) -> None:
    """Per-novel accuracy and ceiling -- the spread the CIs have to respect."""
    b = data["baselines"]["nearest-mention"]
    per_novel = b["per_novel"]
    ceilings = data["per_novel_ceiling"]
    person = data["novel_person"]

    order = sorted(per_novel, key=lambda n: per_novel[n])
    ys = range(len(order))

    fig, ax = plt.subplots(figsize=(7.0, 0.42 * len(order) + 1.9))
    for y, book in zip(ys, order):
        acc, ceil = per_novel[book], ceilings[book]
        ax.plot([acc, ceil], [y, y], color=BLUE_PALE, linewidth=7,
                solid_capstyle="round", zorder=1)
        ax.scatter([ceil], [y], s=44, facecolor=SURFACE, edgecolor=BLUE_LIGHT,
                   linewidth=1.8, zorder=3)
        ax.scatter([acc], [y], s=48, color=BLUE, zorder=4,
                   edgecolor=SURFACE, linewidth=1.2)
        ax.annotate(f"{100*acc:.0f}%", (acc, y), textcoords="offset points",
                    xytext=(-9, 0), ha="right", va="center", fontsize=8,
                    color=INK)

    pooled = b["accuracy"]["all"]
    ax.axvline(pooled, color=ORANGE, linewidth=1.4, linestyle=(0, (4, 3)),
               zorder=2)
    ax.annotate(f"pooled {100*pooled:.0f}%", (pooled, len(order) - 0.42),
                textcoords="offset points", xytext=(6, 0), fontsize=8.5,
                color=ORANGE, fontweight="600", va="bottom")

    ax.set_yticks(list(ys))
    ax.set_yticklabels(
        [f"{_pretty(b_)} · {'3rd' if person[b_] == 3 else '1st'}-pers"
         for b_ in order],
        fontsize=8.5, color=INK,
    )
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(order) - 0.2)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("speaker accuracy (nearest-mention baseline)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    _despine(ax, keep=("bottom",))
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=7,
                   markeredgecolor=SURFACE, label="nearest-mention baseline"),
            Line2D([], [], marker="o", color=BLUE_LIGHT, linestyle="",
                   markersize=7, markerfacecolor=SURFACE, markeredgewidth=1.8,
                   label="candidate-set ceiling"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
        handletextpad=0.6, columnspacing=2.0,
    )
    ax.set_title(
        "Books differ more than systems do — which is why the CIs resample novels",
        loc="left", pad=26,
    )
    ax.text(
        0.0, 1.02,
        f"Per-novel accuracy and ceiling · PDNC {data['split']} split",
        fontsize=8.5, color=MUTED, ha="left", transform=ax.transAxes,
    )
    _save(fig, "fig3_per_novel_variation")


def fig4_gazetteer_ablation(data: dict) -> None:
    """What the derived-name rule recovers, per quote type."""
    before = data["ablation"]["PDNC aliases only"]["recall"]
    after = data["ablation"]["+ derived names"]["recall"]

    keys = ["all", *QUOTE_TYPES]
    xs = range(len(keys))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, (label, series, color) in enumerate(
        (("PDNC aliases only", before, BLUE_LIGHT), ("+ derived names", after, BLUE))
    ):
        offs = [x + (i - 0.5) * (width + 0.03) for x in xs]
        vals = [series[k] for k in keys]
        ax.bar(offs, vals, width=width, color=color, label=label, zorder=3)
        for x, v in zip(offs, vals):
            ax.annotate(f"{100*v:.0f}", (x, v), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8, color=INK_2)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([k.capitalize() for k in keys], fontsize=9, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("gold speaker in candidate set")
    ax.grid(axis="x", visible=False)
    _despine(ax, keep=("bottom",))
    # Bars run to the baseline, so every in-axes corner is occupied.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              handletextpad=0.6, columnspacing=2.0)
    ax.set_title(
        "PDNC's gold alias lists are incomplete; one rule buys the gap back",
        loc="left", pad=26,
    )
    ax.text(
        0.0, 1.03,
        f"Candidate-set ceiling with and without derived short names · "
        f"PDNC {data['split']} split",
        fontsize=8.5, color=MUTED, ha="left", transform=ax.transAxes,
    )
    _save(fig, "fig4_gazetteer_ablation")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    rebuild = "--rebuild" in sys.argv

    print(f"gathering figure data ({split})...")
    data = gather(split, rebuild=rebuild)

    print("rendering figures...")
    fig1_headroom(data)
    fig2_window_sweep(data)
    fig3_novel_variation(data)
    fig4_gazetteer_ablation(data)
    print(f"\nfigures in {PLOTS_DIR}")

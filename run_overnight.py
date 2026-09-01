"""Unattended driver for the Week 2 + Week 3 model runs.

Launch this when you are done using the machine for the night:

    python run_overnight.py

It runs, in order: the ModernBERT ranker, the comparison table, Qwen3-8B
zero-shot, Qwen3-8B LoRA, and finally a combined table with all three systems.
Roughly eight hours end to end. Every step writes its own log under
outputs/results/ and its own results JSON, so a step that dies does not take
the earlier results with it -- and the driver keeps going, because the
zero-shot decoder run does not depend on the ranker having succeeded.

NOTE ON THE GPU: these runs take most of the card's memory. That is fine
overnight and miserable if you are trying to use the desktop at the same time.
Nothing here touches the TEST split; the test novels are scored once, in
Week 4, deliberately.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
LOGS = ROOT / "outputs" / "results"
LOGS.mkdir(parents=True, exist_ok=True)

# (label, log filename, argv). Ordered so the cheapest results land first: if
# something goes wrong at 3am, the Week 2 deliverable is already on disk.
STEPS: list[tuple[str, str, list[str]]] = [
    (
        "Model 1: ModernBERT ranker, 8 epochs",
        "train_ranker_mb.log",
        ["ranker.py", "--epochs", "8", "--batch-size", "8", "--tag", "_mb"],
    ),
    (
        "Model 1 vs baselines on dev (Week 2 deliverable)",
        "compare_dev_mb.log",
        ["compare.py", "dev", "--checkpoint", "ranker_mb.pt"],
    ),
    (
        "Model 2: Qwen3-8B zero-shot on dev",
        "generator_zeroshot_dev.log",
        ["generator.py", "dev", "--zero-shot"],
    ),
    (
        "Model 2: Qwen3-8B LoRA (8k quotes) + dev",
        "generator_lora_dev.log",
        ["generator.py", "dev", "--epochs", "1", "--limit-quotes", "8000"],
    ),
    (
        "Combined table: baselines + ranker + both decoders",
        "compare_dev_all.log",
        ["compare.py", "dev", "--checkpoint", "ranker_mb.pt", "--include",
         "generator-zeroshot_dev.json", "generator-lora_dev.json"],
    ),
]


def run(label: str, log_name: str, argv: list[str]) -> tuple[bool, float]:
    """Run one step, streaming both streams to its own log file."""
    log_path = LOGS / log_name
    started = time.time()
    print(f"\n{'=' * 70}")
    print(f"[{datetime.now():%H:%M:%S}] START  {label}")
    print(f"          log: {log_path}")
    sys.stdout.flush()

    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        # stderr is merged into stdout: the training loops log through the
        # logging module, which writes to stderr, and splitting the two would
        # interleave a run's losses and its progress across two files.
        proc = subprocess.run(
            [sys.executable, *argv], cwd=SRC, stdout=log,
            stderr=subprocess.STDOUT,
        )

    mins = (time.time() - started) / 60
    ok = proc.returncode == 0
    print(f"[{datetime.now():%H:%M:%S}] {'DONE ' if ok else 'FAILED'} {label} "
          f"({mins:.0f} min, exit {proc.returncode})")
    if not ok:
        print(f"          see {log_path} for the traceback")
    sys.stdout.flush()
    return ok, mins


def main() -> None:
    print(f"overnight run started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"working directory: {SRC}")
    results = []
    for label, log_name, argv in STEPS:
        ok, mins = run(label, log_name, argv)
        results.append((label, ok, mins))

    print(f"\n{'=' * 70}")
    print(f"overnight run finished {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    for label, ok, mins in results:
        print(f"  {'ok    ' if ok else 'FAILED'} {mins:6.0f} min  {label}")
    failed = [label for label, ok, _ in results if not ok]
    total = sum(m for _, _, m in results)
    print(f"\n  total {total / 60:.1f} h, {len(failed)} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

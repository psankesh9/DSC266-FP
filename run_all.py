"""Unattended, resumable driver for every remaining model run.

Start it and walk away:

    python run_all.py

If it is interrupted -- a crash, a reboot, a closed laptop lid -- run the same
command again. Nothing that already finished is repeated.

Two layers of recovery, because they fail differently:

* Between steps, this driver keeps a state file. A step that finished is never
  re-run, so relaunching costs only the steps that had not completed.
* Inside a step, ``ranker.py`` and ``generator.py`` checkpoint their own
  training state. A relaunched step resumes mid-epoch rather than from the
  start. This is what the first overnight attempt lacked: it died 143 minutes
  into a LoRA epoch with an access violation and left nothing behind.

Failures do not stop the queue. A step that fails is retried, and the ordering
below puts the cheap, load-bearing results first and the optional capacity
ablation last, so an interruption at hour nine still leaves the report's
required numbers on disk.

Nothing in the test phase influences anything trained before it. The test
novels are scored once here, at the end, with the models already selected on
dev -- which is the single scoring run the proposal commits to.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RESULTS = ROOT / "outputs" / "results"
MODELS = ROOT / "outputs" / "models"
RESULTS.mkdir(parents=True, exist_ok=True)

STATE_PATH = RESULTS / "run_all_state.json"
PROGRESS_PATH = RESULTS / "run_all_progress.txt"

# A step that fails this fast did not fail from bad luck -- it failed on
# configuration, most often by not fitting in VRAM. Retrying it unchanged would
# waste attempts, so the driver moves to the next fallback command instead.
# A step that fails *late* is worth retrying as-is, because its own checkpoints
# mean the retry starts near where the crash happened.
FAST_FAIL_MINUTES = 8.0


@dataclass
class Step:
    """One subprocess, with its fallbacks and its resume contract."""

    key: str                      # stable id; the state file is keyed on it
    label: str
    log: str
    argv: list[str]
    # Tried in order once the primary command fails fast. Used for the runs
    # whose only plausible failure is memory, where a smaller batch is a real
    # answer and a retry is not.
    fallbacks: list[list[str]] = field(default_factory=list)
    # Results file whose existence proves the step finished. A backstop for a
    # lost state file only, so it must name a file that does not exist yet.
    produces: str | None = None
    # Keys that must have completed first. Guards against a step silently
    # consuming a stale artefact from an earlier, different run.
    requires: list[str] = field(default_factory=list)
    max_attempts: int = 4
    note: str = ""
    # Deferred steps are queued and documented but never run by a plain
    # `python run_all.py`. They need the GPU to themselves and are not on the
    # critical path, so they wait for `--only <key>` or `--include-deferred`.
    deferred: bool = False


STEPS: list[Step] = [
    # ---------------------------------------------------- ranker ablations
    Step(
        key="ranker-clamp",
        label="Ablation: ModernBERT with context clamped to 512",
        log="train_ranker_clamp.log",
        argv=["ranker.py", "--clamp", "--epochs", "8", "--tag", "_clamp"],
        produces="ranker_clamp_dev.json",
        note="the ablation the advisor's context argument turns on",
    ),
    Step(
        key="ranker-legacy",
        label="Ablation: RoBERTa at the pre-swap window",
        log="train_ranker_legacy.log",
        argv=["ranker.py", "--encoder", "roberta-base", "--clamp",
              "--epochs", "8", "--tag", "_legacy"],
        produces="ranker_legacy_dev.json",
        note="with the clamp run, separates backbone from window",
    ),
    Step(
        key="ranker-nofeat",
        label="Ablation: hand features zeroed",
        log="train_ranker_nofeat.log",
        argv=["ranker.py", "--no-features", "--epochs", "8", "--tag", "_nofeat"],
        produces="ranker_nofeat_dev.json",
    ),
    # ---------------------------------------------------------- decoder
    Step(
        key="generator-lora-dev",
        label="Model 2: Qwen3-8B LoRA (8k quotes) + dev",
        log="generator_lora_dev.log",
        argv=["generator.py", "dev", "--epochs", "1", "--limit-quotes", "8000"],
        produces="generator-lora_dev.json",
        # The step that crashed last time. Its own checkpoints make each retry
        # start near the crash, so more attempts are worth having here.
        max_attempts=6,
    ),
    Step(
        key="generator-small-dev",
        label="Ablation: Qwen3-1.7B zero-shot on dev",
        log="generator_zeroshot_small_dev.log",
        argv=["generator.py", "dev", "--zero-shot",
              "--model", "Qwen/Qwen3-1.7B", "--tag", "-1.7b"],
        produces="generator-zeroshot-1.7b_dev.json",
        note="the documented size fallback, reported as a comparison",
    ),
    # ------------------------------------------------------- dev table
    Step(
        key="compare-dev",
        label="Dev table: baselines + ranker + ablations + every decoder",
        log="compare_dev_all.log",
        # The four ranker ablations are folded in here rather than left in
        # their own json files, because the ablation claims are comparative:
        # "the wide window is worth +3 points" is a statement about a
        # difference, and only this table computes the paired bootstrap
        # interval over novels that says whether the difference is separable
        # from novel-to-novel variation. Ablations are dev-only by design --
        # the test split is scored once, with the models dev already chose.
        argv=["compare.py", "dev", "--checkpoint", "ranker_mb.pt", "--include",
              "ranker_clamp_dev.json", "ranker_legacy_dev.json",
              "ranker_nofeat_dev.json", "ranker_large_dev.json",
              "generator-zeroshot_dev.json", "generator-lora_dev.json",
              "generator-zeroshot-1.7b_dev.json"],
    ),
    # ------------------------------------------- test: scored exactly once
    Step(
        key="compare-test",
        label="TEST: baselines + ModernBERT ranker",
        log="compare_test_ranker.log",
        argv=["compare.py", "test", "--checkpoint", "ranker_mb.pt"],
    ),
    Step(
        key="generator-zeroshot-test",
        label="TEST: Qwen3-8B zero-shot",
        log="generator_zeroshot_test.log",
        argv=["generator.py", "test", "--zero-shot"],
        produces="generator-zeroshot_test.json",
    ),
    Step(
        key="generator-lora-test",
        label="TEST: Qwen3-8B LoRA",
        log="generator_lora_test.log",
        argv=["generator.py", "test", "--adapter", "generator-lora.adapter"],
        produces="generator-lora_test.json",
        # Without this the step would happily score whatever adapter happens to
        # be on disk -- including a smoke-test one -- and label it the result.
        requires=["generator-lora-dev"],
    ),
    Step(
        key="compare-test-all",
        label="TEST table: every system",
        log="compare_test_all.log",
        argv=["compare.py", "test", "--checkpoint", "ranker_mb.pt", "--include",
              "generator-zeroshot_test.json", "generator-lora_test.json"],
    ),
    # ------------------------------------------------ extrinsic + figures
    Step(
        key="extrinsic-dev",
        label="Extrinsic audiobook metric on dev",
        log="extrinsic_dev.log",
        argv=["extrinsic.py", "dev"],
    ),
    Step(
        key="extrinsic-test",
        label="Extrinsic audiobook metric on test",
        log="extrinsic_test.log",
        argv=["extrinsic.py", "test"],
    ),
    Step(
        key="figures",
        label="Rebuild report figures",
        log="figures.log",
        argv=["figures.py", "dev", "--rebuild"],
    ),
    # ------------------------------------------------------------ optional
    Step(
        key="ranker-large",
        label="Ablation: ModernBERT-large (capacity)",
        log="train_ranker_large.log",
        argv=["ranker.py", "--encoder", "answerdotai/ModernBERT-large",
              "--epochs", "8", "--batch-size", "4",
              "--memory-fraction", "0.92", "--tag", "_large"],
        # 395M parameters at 1,152 tokens on a 12GB card is genuinely marginal.
        # If batch 4 will not fit, a smaller batch is the honest answer, and the
        # run is reported with the batch size it actually used.
        fallbacks=[
            ["ranker.py", "--encoder", "answerdotai/ModernBERT-large",
             "--epochs", "8", "--batch-size", "2",
             "--memory-fraction", "0.92", "--tag", "_large"],
            ["ranker.py", "--encoder", "answerdotai/ModernBERT-large",
             "--epochs", "6", "--batch-size", "1",
             "--memory-fraction", "0.92", "--tag", "_large"],
        ],
        produces="ranker_large_dev.json",
        note="last on purpose: the most expensive and the least load-bearing",
    ),
    # ------------------------------------------------- deferred: do later
    Step(
        key="ranker-large-clean",
        label="PRIORITY-LATER: ModernBERT-large, effective batch matched to base",
        log="train_ranker_large_clean.log",
        # Identical to the base runs in every respect except the encoder:
        # batch 1 x 8 accumulation reproduces base's batch of 8, same 8 epochs,
        # same 2e-5. The first large run could only fit batch 1 and so differed
        # from base in capacity AND effective batch AND epochs at once, which
        # is not an ablation anyone can draw a conclusion from. This one is.
        argv=["ranker.py", "--encoder", "answerdotai/ModernBERT-large",
              "--epochs", "8", "--batch-size", "1", "--grad-accum", "8",
              "--lr", "2e-5", "--memory-fraction", "0.92",
              "--tag", "_large_clean"],
        # If even batch 1 stops fitting, recompute activations rather than
        # give up the matched effective batch -- the matching is the point.
        fallbacks=[
            ["ranker.py", "--encoder", "answerdotai/ModernBERT-large",
             "--epochs", "8", "--batch-size", "1", "--grad-accum", "8",
             "--lr", "2e-5", "--grad-checkpoint", "--memory-fraction", "0.92",
             "--tag", "_large_clean"],
        ],
        produces="ranker_large_clean_dev.json",
        deferred=True,
        note="~7h, needs the GPU alone; run after the test split is scored",
    ),
]


# ------------------------------------------------------------------- state


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("state file unreadable; starting a fresh one")
    return {"steps": {}}


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_PATH)


def write_progress(state: dict) -> None:
    """A plain-text summary, rewritten after every step.

    The point is that someone who was not watching can open one file and see
    what happened, without reading a 15-hour log.
    """
    lines = [f"run_all progress - updated {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    for step in STEPS:
        rec = state["steps"].get(step.key, {})
        status = rec.get("status", "pending")
        mins = rec.get("minutes")
        when = rec.get("finished", "")
        detail = f"{mins:6.0f} min" if mins else " " * 10
        lines.append(f"  {status:8s} {detail}  {step.label}")
        if status == "failed":
            lines.append(f"           after {rec.get('attempts', 0)} attempts; "
                         f"see outputs/results/{step.log}")
        elif status == "skipped":
            lines.append(f"           {rec.get('reason', '')}")
        if when:
            lines[-1] = lines[-1]
    done = sum(1 for s in STEPS
               if state["steps"].get(s.key, {}).get("status") == "done")
    lines += ["", f"  {done}/{len(STEPS)} steps complete"]
    PROGRESS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ running


def run_once(step: Step, argv: list[str]) -> tuple[int, float]:
    """Run one command, streaming both streams into the step's own log."""
    log_path = RESULTS / step.log
    started = time.time()
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n{'=' * 70}\n")
        log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  "
                  f"{' '.join(argv)}\n")
        log.write(f"{'=' * 70}\n")
        log.flush()
        # stderr is merged: the training loops log through the logging module,
        # which writes to stderr, and splitting the streams would scatter one
        # run's losses and its progress across two files.
        proc = subprocess.run([sys.executable, *argv], cwd=SRC,
                              stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode, (time.time() - started) / 60


def already_done(step: Step, state: dict) -> bool:
    rec = state["steps"].get(step.key, {})
    if rec.get("status") == "done":
        return True
    # Backstop for a lost or hand-edited state file.
    if step.produces and (RESULTS / step.produces).exists():
        print(f"  (state file says pending, but {step.produces} exists "
              f"- treating as done)")
        state["steps"][step.key] = {"status": "done", "attempts": 0,
                                    "note": "inferred from output file"}
        return True
    return False


def execute(step: Step, state: dict) -> str:
    """Run one step to completion, failure, or skip. Returns the status."""
    missing = [k for k in step.requires
               if state["steps"].get(k, {}).get("status") != "done"]
    if missing:
        reason = f"requires {', '.join(missing)}, which did not complete"
        print(f"[{datetime.now():%H:%M:%S}] SKIP   {step.label}")
        print(f"          {reason}")
        state["steps"][step.key] = {"status": "skipped", "reason": reason}
        return "skipped"

    variants = [step.argv, *step.fallbacks]
    attempts = 0
    vi = 0
    while attempts < step.max_attempts and vi < len(variants):
        argv = variants[vi]
        attempts += 1
        print(f"\n{'=' * 70}")
        print(f"[{datetime.now():%H:%M:%S}] START  {step.label}"
              f"  (attempt {attempts})")
        if step.note and attempts == 1:
            print(f"          {step.note}")
        print(f"          {' '.join(argv)}")
        print(f"          log: outputs/results/{step.log}")
        sys.stdout.flush()

        code, mins = run_once(step, argv)
        # A step that wrote its results file and then died on the way out has
        # already done its work. The 1.7B run on 2026-08-25 printed its summary,
        # wrote its json, then exited 0xC0000005 during teardown, and this
        # driver threw the whole thing away and started again. Trust the
        # artefact over the exit code.
        produced = bool(step.produces and (RESULTS / step.produces).exists())
        if code == 0 or produced:
            if code != 0:
                print(f"          exit {code}, but {step.produces} was "
                      f"written; counting the step as done")
            print(f"[{datetime.now():%H:%M:%S}] DONE   {step.label} "
                  f"({mins:.0f} min)")
            record = {
                "status": "done", "attempts": attempts, "minutes": round(mins, 1),
                "argv": argv, "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
            }
            if code != 0:
                record["exit_after_write"] = code
            state["steps"][step.key] = record
            return "done"

        print(f"[{datetime.now():%H:%M:%S}] FAIL   {step.label} "
              f"({mins:.0f} min, exit {code})")
        state["steps"][step.key] = {
            "status": "running", "attempts": attempts, "last_exit": code,
            "last_minutes": round(mins, 1),
        }
        save_state(state)

        if mins < FAST_FAIL_MINUTES and vi + 1 < len(variants):
            print("          failed fast; trying the next fallback command")
            vi += 1
        elif mins < FAST_FAIL_MINUTES and not step.fallbacks:
            # Nothing to fall back to and nothing was accomplished. One more
            # try in case it was transient, then give up rather than spin.
            if attempts >= 2:
                break
        else:
            print("          retrying; the step resumes from its own checkpoint")
        sys.stdout.flush()

    print(f"[{datetime.now():%H:%M:%S}] GIVING UP on {step.label} "
          f"after {attempts} attempts")
    state["steps"][step.key] = {"status": "failed", "attempts": attempts}
    return "failed"


# -------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run every remaining model, resumably")
    parser.add_argument("--list", action="store_true",
                        help="show the queue and each step's status, run nothing")
    parser.add_argument("--only", nargs="*", default=[],
                        help="run just these step keys")
    parser.add_argument("--redo", nargs="*", default=[],
                        help="clear these step keys' state so they run again")
    parser.add_argument("--redo-all", action="store_true",
                        help="clear every step's state")
    parser.add_argument("--include-deferred", action="store_true",
                        help="also run the deferred steps, which a plain run "
                             "holds back because they want the GPU alone")
    args = parser.parse_args()

    state = load_state()
    if args.redo_all:
        state["steps"] = {}
    for key in args.redo:
        state["steps"].pop(key, None)
    if args.redo or args.redo_all:
        save_state(state)

    if args.list:
        print(f"{'key':26s} {'status':10s} label")
        for step in STEPS:
            rec = state["steps"].get(step.key, {})
            status = rec.get("status", "pending")
            if step.deferred and status != "done":
                status = "DEFERRED"
            print(f"{step.key:26s} {status:10s} {step.label}")
        return

    if args.only:
        queue = [s for s in STEPS if s.key in args.only]
    else:
        queue = [s for s in STEPS
                 if not s.deferred or args.include_deferred]
        held = [s for s in STEPS if s.deferred and not args.include_deferred]
        for s in held:
            print(f"deferred (not queued): {s.key} - {s.note}")

    print(f"run_all started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"working directory: {SRC}")
    print(f"state: {STATE_PATH}")
    print(f"progress summary: {PROGRESS_PATH}")
    print(f"{len(queue)} steps queued\n")
    state.setdefault("started", f"{datetime.now():%Y-%m-%d %H:%M:%S}")
    write_progress(state)

    began = time.time()
    for step in queue:
        if already_done(step, state):
            print(f"[{datetime.now():%H:%M:%S}] SKIP   {step.label} "
                  f"(already done)")
            save_state(state)
            write_progress(state)
            continue
        execute(step, state)
        save_state(state)
        write_progress(state)

    # ------------------------------------------------------------ summary
    print(f"\n{'=' * 70}")
    print(f"run_all finished {datetime.now():%Y-%m-%d %H:%M:%S} "
          f"({(time.time() - began) / 3600:.1f} h)\n")
    tally: dict[str, int] = {}
    for step in STEPS:
        rec = state["steps"].get(step.key, {})
        status = rec.get("status", "pending")
        tally[status] = tally.get(status, 0) + 1
        mins = rec.get("minutes")
        print(f"  {status:8s} {f'{mins:.0f} min' if mins else '':>10s}  "
              f"{step.label}")
    print("\n  " + ", ".join(f"{n} {s}" for s, n in sorted(tally.items())))

    failed = [s.key for s in STEPS
              if state["steps"].get(s.key, {}).get("status") in
              ("failed", "running")]
    if failed:
        print(f"\n  unfinished: {', '.join(failed)}")
        print("  re-run `python run_all.py` to retry only those.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

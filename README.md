# Who Said That? Speaker Attribution for Untagged Dialogue in Novels

DSC266 final project. Pratheek Sankeshi, solo.

Quotation attribution on the Project Dialogism Novel Corpus, evaluated on
author-disjoint splits and broken out by how the speaker is signalled
(explicit / anaphoric / implicit). A ModernBERT span ranker and a constrained
Qwen3-8B decoder are compared against five baselines and against the ceiling
that candidate enumeration imposes.

## The submission

**`paper.pdf`** is the report (6 pages, ACL-style two columns). It is built
from `paper.md` by `src/build_paper.py`.

`report.pdf` is an extended 16-page version of the same work, kept for
reference. It carries the full window sweep, a fourth figure, and longer
methods and limitations sections. **It is not the submission**; where the two
disagree in emphasis, `paper.pdf` is current.

## Headline result

Test split, 7 novels by 7 unseen authors, 8,443 quotations, scored once:

| System | Accuracy | Share of reachable headroom |
|---|---|---|
| nearest-mention baseline | 49.5% | 55.0% |
| Qwen3-8B zero-shot | 77.2% | 85.6% |
| ModernBERT ranker | 82.3% | 91.3% |
| Qwen3-8B LoRA | 87.7% | 97.2% |
| candidate ceiling | 90.1% | 100% |

## Layout

    paper.md / paper.html / paper.pdf     the submission
    report.md / report.html / report.pdf  extended version
    proposal.md                           the approved proposal
    src/                                  all model and evaluation code
    outputs/plots/                        the four generated figures
    outputs/results/                      training logs and summary JSONs

### `src/`

| File | Role |
|---|---|
| `pdnc.py`, `quotes.py` | corpus loading, author-disjoint splits, quote types |
| `candidates.py` | candidate enumeration and the ceiling measurement |
| `ranker.py` | Model 1: the ModernBERT span ranker |
| `generator.py` | Model 2: constrained Qwen3-8B scoring, zero-shot and LoRA |
| `evaluate.py`, `compare.py` | scoring and the novel-level bootstrap |
| `ablation_contrasts.py` | paired ranker-vs-ranker contrasts for the ablations |
| `extrinsic.py` | the audiobook mis-voiced-seconds metric |
| `figures.py`, `build_paper.py`, `build_report.py` | figures and rendering |
| `run_all.py` | the resumable experiment driver |
| `config.py`, `device.py`, `gutenberg.py` | configuration, GPU setup, out-of-domain demo |

Backbones come from HuggingFace. Candidate enumeration, the ranking head, the
constrained scoring loop, the evaluation harness and the bootstrap are mine.

## Not in this repository

- **Model checkpoints (6.5 GB).** Nine ModernBERT and RoBERTa rankers, 478 MB
  to 1.5 GB each, plus a 167 MB Qwen3-8B LoRA adapter. Available on request.
- **PDNC (119 MB).** Public, and better obtained from source than
  redistributed: github.com/Priya22/project-dialogism-novel-corpus
- **Per-quotation prediction dumps (45 MB).** The summary JSONs and logs in
  `outputs/results/` are tracked and carry every number the paper cites.

## Reproducing

`run_all.py` drives the whole experiment queue. It is resumable and records
per-step state, so a partial run picks up where it stopped.

    python run_all.py                      # the 14 load-bearing steps
    python run_all.py --include-deferred   # plus the matched-capacity run
    python run_all.py --only <step-key>    # a single step

Roughly 3.3 hours on a 12 GB card for the main queue, plus about 5 hours for
the deferred ModernBERT-large run. All 15 steps are recorded complete in
`outputs/results/run_all_progress.txt`.

    python src/build_paper.py     # paper.md  -> paper.html
    python src/build_report.py    # report.md -> report.html

Both are printed to PDF with headless Chrome:

    chrome --headless --disable-gpu --no-pdf-header-footer \
           --print-to-pdf=paper.pdf paper.html

## Two notes for a reader

1. **Dev and test accuracy are not directly comparable.** The candidate ceiling
   is 86.1% on dev and 90.1% on test, so test is a genuinely easier candidate
   problem. Section 3.2 explains why; cross-split comparisons use share of
   reachable headroom instead.
2. **Ablation contrasts are reported as `a minus b`**, so a positive number
   favours the first-named condition in every row of Table 4.

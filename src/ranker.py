"""Model 1: a span-based bilinear ranker over candidate speakers.

The formulation is coreference-style rather than classification-style. For one
quotation the encoder runs **once** over the surrounding narration; the quote
and each candidate mention are then pooled out of that single contextual
sequence and scored against each other. A cross-encoder that re-reads the
context once per candidate would cost 5x the compute for the same information,
because the context does not change between candidates.

Three decisions worth stating outright:

* ``Character.category`` (major / intermediate / minor) is **never** used as a
  feature. PDNC derives it from how much dialogue a character speaks, so it is
  a function of the answer; a model given it would score well by reading the
  label. This is the same circularity that sank an earlier version of this
  project, and it is excluded deliberately rather than overlooked.

* Candidates come from a wider window than the encoder reads. Those falling
  outside the encoded text get a learned vector instead of a pooled span, so
  the wide window's higher ceiling stays reachable.

* The loss is a softmax over candidates, not an independent binary decision per
  candidate. Exactly one candidate is correct, and normalising across them lets
  the model spend its capacity on ordering rather than on calibration.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from candidates import Candidate, enumerate_candidates
from config import (
    BATCH_SIZE,
    CLAMP_CHARS_AFTER,
    CLAMP_CHARS_BEFORE,
    CLAMP_MAX_LENGTH,
    ENCODER_MODEL,
    FEATURE_DROPOUT,
    HEAD_LEARNING_RATE,
    LEARNING_RATE,
    MAX_CANDIDATES,
    MAX_GRAD_NORM,
    MAX_LENGTH,
    MODEL_CHARS_AFTER,
    MODEL_CHARS_BEFORE,
    NUM_EPOCHS,
    QUOTE_HEAD_CHARS,
    QUOTE_TAIL_CHARS,
    SEED,
    SPAN_DIM,
    THIRD_PERSON_PRONOUNS,
    USE_AMP,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    PRONOUN_GENDER,
)
from device import get_device
from pdnc import Novel, Quote

logger = logging.getLogger(__name__)

N_FEATURES = 11


@dataclass(frozen=True)
class Window:
    """How much text the encoder reads, and its token budget.

    Carried explicitly rather than read from ``config`` at each call site
    because the context ablation changes it, and a model trained clamped must
    be *evaluated* clamped. Storing it in the checkpoint is what makes that
    guarantee hold across processes.
    """

    chars_before: int = MODEL_CHARS_BEFORE
    chars_after: int = MODEL_CHARS_AFTER
    max_length: int = MAX_LENGTH

    def as_dict(self) -> dict:
        return {"chars_before": self.chars_before,
                "chars_after": self.chars_after,
                "max_length": self.max_length}


FULL_WINDOW = Window()
# The pre-swap regime: what RoBERTa could physically read.
CLAMPED_WINDOW = Window(CLAMP_CHARS_BEFORE, CLAMP_CHARS_AFTER, CLAMP_MAX_LENGTH)


def _atomic_save(obj, path) -> None:
    """Write a checkpoint that a crash mid-write cannot corrupt.

    The resume files are large and are overwritten every epoch, so a power
    cut or a driver fault during ``torch.save`` would otherwise leave a
    truncated file that fails to load -- turning a recoverable interruption
    into a run that must start from scratch. Writing to a sibling and then
    renaming makes the swap atomic on NTFS.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- encoding


@dataclass
class Example:
    """One quotation, encoded and ready for the model."""

    book: str
    qid: str
    quote_type: str
    gold: str
    input_ids: list[int]
    # Token ranges of the SPOKEN sub-spans only. A split quotation's interjected
    # speech tag sits between them and is deliberately excluded: pooling it into
    # the quote vector would put "thought Newland Archer" inside the
    # representation the candidates are scored against, so every split quote
    # would carry its own answer. The tag stays in the passage as context, where
    # the candidate's own span can pick it up.
    quote_ranges: list[tuple[int, int]]
    cand_spans: list[tuple[int, int] | None]
    cand_feats: list[list[float]]
    cand_names: list[str]
    gold_index: int                        # -1 when the gold is not a candidate
    addressees: list[str]                  # gold, for addressee error analysis


def _truncate_quote(text: str) -> str:
    """Keep the head and tail of a long speech, drop the middle."""
    if len(text) <= QUOTE_HEAD_CHARS + QUOTE_TAIL_CHARS:
        return text
    return f"{text[:QUOTE_HEAD_CHARS]} ... {text[-QUOTE_TAIL_CHARS:]}"


def _tag_pronoun_gender(quote: Quote) -> str:
    """Gender implied by a pronoun in the speech tag, if there is one.

    Only the narration immediately around the quotation counts. A pronoun
    inside the quoted speech belongs to whoever is being talked about, not to
    the speaker.
    """
    window = f"{quote.inner} {quote.right[:60]}".lower()
    for token in window.replace(",", " ").replace(".", " ").split():
        if token in THIRD_PERSON_PRONOUNS:
            return PRONOUN_GENDER[token]
    return ""


def _features(cand: Candidate, quote: Quote, all_cands: list[Candidate],
              in_window: bool) -> list[float]:
    """Hand features for one candidate. Text-derivable only -- see module docs."""
    nearest = cand.nearest
    total_mentions = sum(c.n_mentions for c in all_cands) or 1
    min_distance = min(c.nearest.distance for c in all_cands)
    tag_gender = _tag_pronoun_gender(quote)

    return [
        math.log1p(nearest.distance) / 8.0,
        1.0 if nearest.zone == "before" else 0.0,
        1.0 if nearest.zone == "inner" else 0.0,
        1.0 if nearest.zone == "after" else 0.0,
        1.0 if nearest.near_speech_verb else 0.0,
        1.0 if cand.has_tagged_mention else 0.0,
        1.0 if nearest.distance == min_distance else 0.0,
        math.log1p(cand.n_mentions) / 3.0,
        cand.n_mentions / total_mentions,
        # Gender agreement between the tag pronoun and the candidate. Uses
        # PDNC's annotated gender, so it is corpus metadata rather than a
        # purely textual signal; ablated in the results table.
        1.0 if (tag_gender and cand.character.gender == tag_gender) else
        (-1.0 if (tag_gender and cand.character.gender) else 0.0),
        1.0 if in_window else 0.0,
    ]


class QuoteDataset(Dataset):
    """Pre-encodes every quotation once.

    Encoding is not free -- it runs the alias regex over a 3,300-character
    window and then tokenises -- and it is deterministic, so doing it lazily
    would repeat the identical work on every epoch. Encoding the 23,818 train
    quotes up front costs about a minute and roughly 200 MB.
    """

    def __init__(self, novels: list[Novel], tokenizer, label: str = "",
                 window: Window = FULL_WINDOW):
        self.examples: list[Example] = []
        self.window = window
        for novel in novels:
            for quote in novel.quotes:
                self.examples.append(encode(novel, quote, tokenizer, window))
        if label:
            covered = sum(e.gold_index >= 0 for e in self.examples)
            # How often a candidate had to fall back to the learned absent
            # vector. Under the full window this should be near zero; a
            # non-trivial rate means mentions are being lost to tokenisation
            # rather than to the window, which is a bug, not a design limit.
            spans = [s for e in self.examples for s in e.cand_spans]
            absent = sum(s is None for s in spans)
            logger.info(
                "%s: %d quotes encoded, gold in candidates for %.1f%%, "
                "%.1f%% of candidates out of window",
                label, len(self.examples),
                100 * covered / max(len(self.examples), 1),
                100 * absent / max(len(spans), 1),
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def encode(novel: Novel, quote: Quote, tokenizer,
           window: Window = FULL_WINDOW) -> Example:
    """Build the encoder input and locate every span inside it."""
    text = novel.text
    lo = max(0, quote.start - window.chars_before)
    hi = min(len(text), quote.end + window.chars_after)

    before = text[lo:quote.start]
    middle = _truncate_quote(text[quote.start:quote.end])
    after = text[quote.end:hi]
    passage = f"{before}{middle}{after}"

    # Character ranges of the spoken sub-spans within the passage. Skipped when
    # the middle was truncated, since the head-and-tail rewrite invalidates the
    # interior offsets; the whole middle is then treated as the quote.
    base = len(before)
    truncated = middle != text[quote.start:quote.end]
    if truncated:
        spoken = [(base, base + len(middle))]
    else:
        spoken = [(base + (a - quote.start), base + (b - quote.start))
                  for a, b in quote.spans]

    enc = tokenizer(
        passage,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,
    )
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # Overlong passages are trimmed from the FRONT. The quotation and the
    # narration after it must survive -- that is where speech tags live -- and
    # the oldest context is the least informative thing to give up.
    if len(ids) > window.max_length:
        drop = len(ids) - window.max_length
        ids = [ids[0]] + ids[1 + drop:]
        offsets = [offsets[0]] + offsets[1 + drop:]

    def span_of(c_start: int, c_end: int) -> tuple[int, int] | None:
        """Token span covering a character range, or None if trimmed away."""
        toks = [
            i for i, (a, b) in enumerate(offsets)
            if i != 0 and b > a and a < c_end and b > c_start
        ]
        return (toks[0], toks[-1] + 1) if toks else None

    quote_ranges = [r for r in (span_of(a, b) for a, b in spoken) if r]
    if not quote_ranges:
        quote_ranges = [(0, 1)]

    cands = enumerate_candidates(novel, quote)[:MAX_CANDIDATES]
    cand_spans: list[tuple[int, int] | None] = []
    cand_feats: list[list[float]] = []
    cand_names: list[str] = []
    gold_index = -1

    for i, cand in enumerate(cands):
        # Pick the mention closest to the quote that survived encoding.
        span = None
        for m in sorted(cand.mentions, key=lambda m: m.distance):
            if lo <= m.start and m.end <= hi:
                span = span_of(m.start - lo, m.end - lo)
                if span is not None:
                    break
        cand_spans.append(span)
        cand_feats.append(_features(cand, quote, cands, span is not None))
        cand_names.append(cand.character.name)
        if cand.is_gold:
            gold_index = i

    return Example(
        book=novel.name,
        qid=quote.qid,
        quote_type=quote.quote_type,
        gold=quote.speaker,
        input_ids=ids,
        quote_ranges=quote_ranges,
        cand_spans=cand_spans,
        cand_feats=cand_feats,
        cand_names=cand_names,
        gold_index=gold_index,
        addressees=list(quote.addressees),
    )


def collate(batch: list[Example], pad_id: int) -> dict:
    """Pad to the batch maximum; mask padded tokens and absent candidates."""
    bsz = len(batch)
    max_len = max(len(e.input_ids) for e in batch)
    max_cands = max(max(len(e.cand_spans), 1) for e in batch)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((bsz, max_len), dtype=torch.long)
    quote_mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    cand_spans = torch.zeros((bsz, max_cands, 2), dtype=torch.long)
    cand_has_span = torch.zeros((bsz, max_cands), dtype=torch.bool)
    cand_mask = torch.zeros((bsz, max_cands), dtype=torch.bool)
    feats = torch.zeros((bsz, max_cands, N_FEATURES), dtype=torch.float)
    labels = torch.full((bsz,), -100, dtype=torch.long)

    for i, e in enumerate(batch):
        n = len(e.input_ids)
        input_ids[i, :n] = torch.tensor(e.input_ids)
        attention[i, :n] = 1
        for a, b in e.quote_ranges:
            quote_mask[i, a:b] = True
        for j, span in enumerate(e.cand_spans):
            cand_mask[i, j] = True
            feats[i, j] = torch.tensor(e.cand_feats[j])
            if span is not None:
                cand_spans[i, j] = torch.tensor(span)
                cand_has_span[i, j] = True
        if e.gold_index >= 0:
            labels[i] = e.gold_index

    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "quote_mask": quote_mask,
        "cand_spans": cand_spans,
        "cand_has_span": cand_has_span,
        "cand_mask": cand_mask,
        "features": feats,
        "labels": labels,
        "meta": batch,
    }


# ---------------------------------------------------------------- model


def pool_spans(hidden: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
    """Mean-pool token vectors over [start, end) spans.

    ``hidden`` is (B, L, H); ``spans`` is (..., 2) indexing into L. Returns one
    vector per span. Empty or degenerate spans pool to the start token rather
    than to zeros, which keeps the head from seeing a magnitude discontinuity
    between "no span" and "a real but short span".
    """
    b, length, hdim = hidden.shape
    flat = spans.reshape(b, -1, 2)
    n = flat.shape[1]

    positions = torch.arange(length, device=hidden.device).view(1, 1, length)
    starts = flat[..., 0:1]
    ends = flat[..., 1:2].clamp(min=1)
    mask = (positions >= starts) & (positions < ends)          # (B, n, L)
    mask = mask | ((mask.sum(-1, keepdim=True) == 0) & (positions == starts))

    weights = mask.float()
    weights = weights / weights.sum(-1, keepdim=True).clamp(min=1.0)
    pooled = torch.bmm(weights, hidden)                        # (B, n, H)
    return pooled.view(*spans.shape[:-1], hdim)


class SpanRanker(nn.Module):
    """Scores each candidate mention against the quotation."""

    def __init__(self, encoder_name: str = ENCODER_MODEL,
                 n_features: int = N_FEATURES, span_dim: int = SPAN_DIM,
                 use_features: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.use_features = use_features

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        self.project = nn.Sequential(
            nn.Linear(hidden, span_dim), nn.GELU(), nn.LayerNorm(span_dim)
        )
        # Stands in for a candidate whose mentions all fell outside the encoded
        # text. Learned, so the model can place it wherever the data says an
        # unseen-but-nearby character belongs.
        self.absent = nn.Parameter(torch.zeros(span_dim))
        nn.init.normal_(self.absent, std=0.02)

        self.bilinear = nn.Bilinear(span_dim, span_dim, 1)
        self.feature_drop = nn.Dropout(FEATURE_DROPOUT)
        self.mlp = nn.Sequential(
            nn.Linear(4 * span_dim + n_features, span_dim), nn.GELU(),
            nn.Dropout(FEATURE_DROPOUT), nn.Linear(span_dim, 1),
        )

    def forward(self, input_ids, attention_mask, quote_mask, cand_spans,
                cand_has_span, cand_mask, features, **_) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        # Masked mean over the spoken sub-spans, which need not be contiguous.
        w = quote_mask.to(hidden.dtype)
        w = w / w.sum(-1, keepdim=True).clamp(min=1.0)
        q = self.project(torch.bmm(w.unsqueeze(1), hidden).squeeze(1))  # (B, D)
        c = self.project(pool_spans(hidden, cand_spans))           # (B, C, D)
        c = torch.where(
            cand_has_span.unsqueeze(-1), c,
            self.absent.to(c.dtype).view(1, 1, -1).expand_as(c),
        )

        q_exp = q.unsqueeze(1).expand_as(c)
        interaction = self.bilinear(q_exp.contiguous(), c.contiguous())
        # Feature ablation zeroes the hand features rather than removing the
        # slice. The head keeps its shape, its parameter count, and its
        # initialisation, so the contrast measures what the features knew and
        # not how much smaller the model got.
        feats = (self.feature_drop(features) if self.use_features
                 else torch.zeros_like(features))
        joint = torch.cat(
            [q_exp, c, q_exp * c, (q_exp - c).abs(), feats],
            dim=-1,
        )
        scores = (self.mlp(joint) + interaction).squeeze(-1)       # (B, C)
        return scores.masked_fill(~cand_mask, torch.finfo(scores.dtype).min)


# ---------------------------------------------------------------- training


def make_loader(novels, tokenizer, batch_size: int = BATCH_SIZE,
                shuffle: bool = False, label: str = "",
                window: Window = FULL_WINDOW) -> DataLoader:
    pad = tokenizer.pad_token_id
    return DataLoader(
        QuoteDataset(novels, tokenizer, label=label, window=window),
        batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda b: collate(b, pad), num_workers=0,
    )


@torch.no_grad()
def predict(model, loader, device) -> list[tuple]:
    """Returns one tuple per quote, matching ``to_predictions`` below."""
    model.eval()
    out = []
    for batch in loader:
        tensors = {
            k: v.to(device) for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=USE_AMP and device.type == "cuda"):
            scores = model(**tensors)
        best = scores.float().argmax(dim=-1).cpu().tolist()
        for e, b in zip(batch["meta"], best):
            pred = e.cand_names[b] if e.cand_names else None
            out.append((e.book, e.qid, e.quote_type, e.gold, pred,
                        len(e.cand_names), e.gold_index >= 0, e.addressees))
    return out


def train_model(train_novels, dev_novels, epochs: int = NUM_EPOCHS,
                batch_size: int = BATCH_SIZE, encoder_name: str = ENCODER_MODEL,
                seed: int = SEED, ckpt_path=None,
                window: Window = FULL_WINDOW, use_features: bool = True,
                resume_path=None, grad_accum: int = 1,
                lr: float = LEARNING_RATE, head_lr: float = HEAD_LEARNING_RATE,
                grad_checkpoint: bool = False):
    """Fine-tune the ranker, selecting the checkpoint by dev accuracy.

    When ``ckpt_path`` is given the best weights are written there as soon as
    they improve, so a run killed part-way still leaves its best epoch on disk.

    ``resume_path`` makes the run restartable at epoch granularity. Weights,
    optimiser moments, the LR schedule, and both RNG streams are written after
    every epoch, so a process killed at epoch 6 of 8 resumes at epoch 6 rather
    than repeating two hours of work. Restoring the RNG matters as much as the
    weights: the shuffled epoch order and the dropout masks are drawn from it,
    and a resumed run that reseeds would train on a different data order than
    the one its optimiser state was accumulated under.
    """
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(seed)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    model = SpanRanker(encoder_name, use_features=use_features).to(device)
    if grad_checkpoint:
        # Recompute activations in the backward pass instead of holding them.
        # Costs roughly a third of the step time and buys enough memory that
        # ModernBERT-large can train at a real batch size rather than at 1.
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("gradient checkpointing enabled on the encoder")

    train_loader = make_loader(train_novels, tokenizer, batch_size,
                               shuffle=True, label="train", window=window)
    dev_loader = make_loader(dev_novels, tokenizer, batch_size, label="dev",
                             window=window)

    # The pretrained encoder and the randomly initialised head do not want the
    # same step size; the head is allowed to move an order of magnitude faster.
    head_params = [p for n, p in model.named_parameters()
                   if not n.startswith("encoder.")]
    enc_params = [p for n, p in model.named_parameters()
                  if n.startswith("encoder.")]
    optimiser = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    # The schedule counts optimiser steps, not micro-batches. Under gradient
    # accumulation the two differ by grad_accum, and a schedule built on
    # micro-batches would decay the LR to zero grad_accum times too early.
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimiser, int(WARMUP_RATIO * total_steps), total_steps
    )

    best_acc, best_state = -1.0, None
    start_epoch = 0
    if resume_path is not None and resume_path.exists():
        # Loaded to CPU, not to ``device``: the RNG state is a ByteTensor that
        # torch.set_rng_state refuses to accept once it has been moved to CUDA,
        # and both load_state_dict calls below place their own tensors anyway.
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        optimiser.load_state_dict(blob["optimiser"])
        scheduler.load_state_dict(blob["scheduler"])
        torch.set_rng_state(blob["cpu_rng"].cpu().to(torch.uint8))
        if blob.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(blob["cuda_rng"])
        random.setstate(blob["py_rng"])
        np.random.set_state(blob["np_rng"])
        start_epoch = blob["epochs_done"]
        best_acc = blob["best_acc"]
        # The best weights live in the epoch checkpoint, not here: duplicating
        # a 600 MB state dict into every resume file buys nothing when the
        # partial checkpoint already holds exactly it.
        if ckpt_path is not None and ckpt_path.exists():
            best_state = torch.load(ckpt_path, map_location="cpu",
                                    weights_only=False)["state_dict"]
        logger.info("resumed from %s at epoch %d (best dev %.4f so far)",
                    resume_path.name, start_epoch + 1, best_acc)

    for epoch in range(start_epoch, epochs):
        model.train()
        running, seen = 0.0, 0
        pending = 0            # micro-batches accumulated since the last step
        for step, batch in enumerate(train_loader, 1):
            tensors = {
                k: v.to(device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            labels = tensors.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=USE_AMP and device.type == "cuda"):
                scores = model(**tensors)
            # Quotes whose gold speaker is not among the candidates carry
            # label -100 and are ignored: there is no correct answer to teach,
            # and forcing one would train the model toward an arbitrary
            # candidate. They still count against accuracy at evaluation.
            loss = F.cross_entropy(scores.float(), labels, ignore_index=-100)
            if torch.isnan(loss):
                continue
            # Scaled so the accumulated gradient is the mean over the whole
            # effective batch rather than its sum; without this, grad_accum
            # would silently multiply the step size by grad_accum.
            (loss / grad_accum).backward()
            pending += 1
            if pending == grad_accum:
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
                pending = 0

            running += loss.item()
            seen += 1
            if step % 200 == 0:
                logger.info("epoch %d step %d/%d loss %.4f",
                            epoch + 1, step, len(train_loader), running / seen)
                running, seen = 0.0, 0

        # A trailing partial group would otherwise be dropped, and its
        # gradients would leak into the first step of the next epoch.
        if pending:
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimiser.step()
            scheduler.step()
            optimiser.zero_grad(set_to_none=True)
            pending = 0

        rows = predict(model, dev_loader, device)
        acc = sum(r[3] == r[4] for r in rows) / max(len(rows), 1)
        logger.info("epoch %d dev accuracy %.4f", epoch + 1, acc)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            if ckpt_path is not None:
                _atomic_save({"state_dict": best_state, "encoder": encoder_name,
                              "window": window.as_dict(),
                              "use_features": use_features,
                              "batch_size": batch_size,
                              "grad_accum": grad_accum,
                              "effective_batch": batch_size * grad_accum,
                              "lr": lr,
                              "epoch": epoch + 1, "dev_accuracy": acc},
                             ckpt_path)
                logger.info("epoch %d checkpointed to %s", epoch + 1, ckpt_path)

        if resume_path is not None:
            _atomic_save({
                "epochs_done": epoch + 1,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None),
                "py_rng": random.getstate(),
                "np_rng": np.random.get_state(),
            }, resume_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer, best_acc


# ---------------------------------------------------------------- entry point


def to_predictions(rows: list[tuple]) -> list:
    """Adapt raw model output to the shared evaluation record."""
    from evaluate import Prediction

    return [
        Prediction(book=b, qid=q, quote_type=t, gold=g, pred=p,
                   n_candidates=n, gold_in_candidates=c, addressees=a)
        for b, q, t, g, p, n, c, a in rows
    ]


if __name__ == "__main__":
    import argparse

    from config import MODEL_DIR
    from device import describe
    from evaluate import save, score
    from pdnc import load_split

    parser = argparse.ArgumentParser(description="train the span ranker")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="micro-batches per optimiser step; batch_size * "
                             "grad_accum is the effective batch, so a card "
                             "that only fits 1 can still train at the batch "
                             "the other runs used")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="encoder learning rate; large encoders usually "
                             "want half to a third of the base rate")
    parser.add_argument("--head-lr", type=float, default=HEAD_LEARNING_RATE)
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="recompute activations in the backward pass: "
                             "slower per step, fits a far larger batch")
    parser.add_argument("--encoder", default=ENCODER_MODEL)
    parser.add_argument("--clamp", action="store_true",
                        help="context ablation: read only 512 tokens of a "
                             "-900/+400 window, the pre-ModernBERT regime")
    parser.add_argument("--no-features", action="store_true",
                        help="feature ablation: zero the hand features so the "
                             "head scores candidates from the encoder alone")
    parser.add_argument("--memory-fraction", type=float, default=None,
                        help="override the VRAM cap; the larger encoder does "
                             "not fit under the desktop-friendly default")
    parser.add_argument("--limit-train", type=int, default=0,
                        help="use only the first N training novels (smoke test)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any resume state and start from epoch 1")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="", help="suffix for output filenames")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    window = CLAMPED_WINDOW if args.clamp else FULL_WINDOW

    # Claim the card before anything else touches CUDA: device.py applies the
    # cap on the first call and will not revise it afterwards.
    get_device(memory_fraction=args.memory_fraction)

    train_novels = load_split("train")
    if args.limit_train:
        train_novels = train_novels[: args.limit_train]
    dev_novels = load_split("dev")

    print(f"device : {get_device()} — {describe()}")
    print(f"encoder: {args.encoder}")
    print(f"context: -{window.chars_before}/+{window.chars_after} chars, "
          f"{window.max_length} tokens" + ("  [CLAMPED]" if args.clamp else ""))
    print(f"train  : {len(train_novels)} novels, "
          f"{sum(len(n.quotes) for n in train_novels):,} quotes")
    print(f"dev    : {len(dev_novels)} novels, "
          f"{sum(len(n.quotes) for n in dev_novels):,} quotes\n")

    name = f"ranker{args.tag}"
    # Written every time dev accuracy improves. Kept separate from the final
    # checkpoint so an interrupted run cannot clobber a good previous one.
    partial = MODEL_DIR / f"{name}.partial.pt"
    # Full training state, rewritten every epoch, deleted on success. Its
    # presence is what makes an unattended relaunch pick up where it stopped.
    resume = MODEL_DIR / f"{name}.resume.pt"
    if args.fresh:
        resume.unlink(missing_ok=True)

    print(f"batch  : {args.batch_size} x {args.grad_accum} accum = "
          f"{args.batch_size * args.grad_accum} effective, lr {args.lr:g}"
          + ("  [grad-checkpoint]" if args.grad_checkpoint else ""))
    print(f"features: {'zeroed [ABLATED]' if args.no_features else 'on'}")
    if resume.exists():
        print(f"resume : {resume.name} present — continuing that run")
    print()

    model, tokenizer, best = train_model(
        train_novels, dev_novels, epochs=args.epochs,
        batch_size=args.batch_size, encoder_name=args.encoder, seed=args.seed,
        ckpt_path=partial, window=window,
        use_features=not args.no_features, resume_path=resume,
        grad_accum=args.grad_accum, lr=args.lr, head_lr=args.head_lr,
        grad_checkpoint=args.grad_checkpoint,
    )

    dev_loader = make_loader(dev_novels, tokenizer, args.batch_size,
                             window=window)
    report = score(name, to_predictions(predict(model, dev_loader, get_device())))
    print()
    print(report)
    save(report, f"{name}_dev.json")

    ckpt = MODEL_DIR / f"{name}.pt"
    _atomic_save({"state_dict": model.state_dict(), "encoder": args.encoder,
                  "window": window.as_dict(),
                  "use_features": not args.no_features,
                  "batch_size": args.batch_size,
                  "grad_accum": args.grad_accum,
                  "effective_batch": args.batch_size * args.grad_accum,
                  "lr": args.lr,
                  "epochs": args.epochs, "dev_accuracy": best}, ckpt)
    partial.unlink(missing_ok=True)
    # Only now: while this file exists the run is considered unfinished.
    resume.unlink(missing_ok=True)
    print(f"\nsaved {ckpt} (best dev accuracy {best:.4f})")

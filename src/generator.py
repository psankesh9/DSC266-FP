"""Model 2: an instruction-tuned decoder constrained to the candidate list.

Qwen3-8B is asked, in plain language, who spoke a quotation, given the
narration around it and an enumerated list of characters. The interesting
design question is what "constrained to the candidate list" should mean
mechanically.

Free generation plus string matching is the obvious reading and the wrong one.
A zero-shot model produces "the old man", "Mr. B", "Elizabeth Bennet" when the
gazetteer says "Elizabeth", or a refusal -- and every one of those has to be
mapped back onto a candidate by some fuzzy rule that is itself a model,
unmeasured and tuned by whoever wrote it. Accuracy would then partly measure
the matcher.

So the model never generates. Each candidate name is *scored* as a decoder
continuation of the same prompt, and the highest-likelihood name wins. This is
constrained decoding done exactly: the output is always a candidate, the
zero-shot number is well defined, nothing is parsed, and the result is a
ranking over the identical candidate sets Model 1 ranks -- so the two models
are compared on equal terms, under the same ceiling, in the same table.

The comparison this supports is deliberately unflattering to the generator.
Model 1 encodes the passage once and pools spans out of it; Model 2 re-reads
the same passage and must locate the speaker through a text interface. If the
instruction-tuned prior is worth anything here, it should show up zero-shot,
where Model 1 has nothing at all.

Two practical notes. The prompt is prefilled once per quotation and its KV
cache is reused across that quotation's candidates -- the prompt does not
change between candidates, and re-prefilling a 1,100-token passage seven times
would multiply the expensive half of the model by seven for no new
information. And 8B parameters in bf16 are 16GB against this machine's 12.8GB,
so the model is loaded 4-bit NF4 and adapted with LoRA. That is a hardware
constraint rather than a finding, and the report states it.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass

# Must precede the first CUDA allocation. Training alternates a ~1,100-token
# prefill with tiny per-candidate forwards, which leaves the caching allocator
# holding many differently-sized blocks; without expandable segments the run
# OOMs with over 2GB reserved-but-unallocated on a 12GB card.
#
# Windows has no implementation of expandable segments: setting it there emits
# a warning at the first allocation and changes nothing, so it is only asked
# for where it exists. The fragmentation it would have solved is handled on
# Windows by emptying the cache periodically during training instead.
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from candidates import enumerate_candidates
from config import (
    GENERATOR_BATCH_SIZE,
    GENERATOR_COMPUTE_DTYPE,
    GENERATOR_ENABLE_THINKING,
    GENERATOR_EPOCHS,
    GENERATOR_GRAD_ACCUM,
    GENERATOR_LEARNING_RATE,
    GENERATOR_LENGTH_NORMALISE,
    GENERATOR_LOAD_4BIT,
    GENERATOR_MAX_LENGTH,
    GENERATOR_MODEL,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_CANDIDATES,
    MAX_GRAD_NORM,
    MODEL_CHARS_AFTER,
    MODEL_CHARS_BEFORE,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from device import get_device
from pdnc import Novel, Quote
from ranker import _truncate_quote, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- loading


def load_backbone(model_name: str = GENERATOR_MODEL, four_bit: bool = GENERATOR_LOAD_4BIT,
                  for_training: bool = False):
    """Load the decoder, quantised, and its tokenizer.

    ``for_training`` attaches LoRA adapters and turns on gradient
    checkpointing. Inference does neither: checkpointing would recompute
    activations we are not going to backprop through, and the KV cache it
    disables is exactly the thing that makes candidate scoring affordable.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, GENERATOR_COMPUTE_DTYPE)
    kwargs = {"dtype": dtype}
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": 0}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not four_bit:
        model = model.to(get_device())

    if for_training:
        from peft import LoraConfig, get_peft_model

        # Deliberately NOT peft's prepare_model_for_kbit_training. That helper
        # upcasts every non-quantised parameter to fp32, and bitsandbytes
        # leaves the output head unquantised -- Qwen3-8B's lm_head is
        # 151,936 x 4,096 = 622M parameters, so the upcast alone costs 2.5GB
        # and OOMs this card before the first optimiser step. The two things
        # the helper does that are actually needed are done directly, and the
        # head stays in bf16.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        # Frozen 4-bit embeddings produce no grad, which would leave the
        # checkpointed blocks with nothing to recompute against.
        model.enable_input_require_grads()
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=list(LORA_TARGET_MODULES),
            bias="none", task_type="CAUSAL_LM",
        ))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("LoRA: %.1fM trainable of %.1fM (%.3f%%)",
                    trainable / 1e6, total / 1e6, 100 * trainable / total)
    return model, tokenizer


# ---------------------------------------------------------------- prompting


@dataclass
class GenExample:
    """One quotation as a prompt plus the candidate names to score against it."""

    book: str
    qid: str
    quote_type: str
    gold: str
    prompt: str
    cand_names: list[str]
    gold_index: int                 # -1 when the gold is not a candidate
    addressees: list[str]


def build_prompt(novel: Novel, quote: Quote, tokenizer) -> tuple[str, list[str], int]:
    """Render one quotation as a chat turn, with its candidate list.

    The passage is the same narration window Model 1 encodes, so neither model
    gets to see context the other cannot. The quotation is repeated after the
    passage rather than only marked inside it: in a split quotation the two
    halves are far apart, and a model reading a long window otherwise has to
    guess which of several quoted lines the question is about.

    The chat template is applied with the generation prompt appended, so the
    string ends exactly where the assistant's answer would begin -- which is
    the position the candidate names are scored at. Qwen3's reasoning block is
    disabled: the model never generates here, and a ``<think>`` span would sit
    between the prompt and the name being scored.
    """
    text = novel.text
    lo = max(0, quote.start - MODEL_CHARS_BEFORE)
    hi = min(len(text), quote.end + MODEL_CHARS_AFTER)
    passage = (
        f"{text[lo:quote.start]}"
        f"{_truncate_quote(text[quote.start:quote.end])}"
        f"{text[quote.end:hi]}"
    ).replace("\n", " ").strip()

    cands = enumerate_candidates(novel, quote)[:MAX_CANDIDATES]
    names = [c.character.name for c in cands]
    gold_index = next((i for i, c in enumerate(cands) if c.is_gold), -1)

    spoken = " ".join(_truncate_quote(quote.text).split())
    options = "; ".join(names) if names else "unknown"
    content = (
        "Read the passage from a novel and decide which character speaks the "
        "quoted line.\n\n"
        f"Passage: {passage}\n\n"
        f'Quoted line: "{spoken}"\n\n'
        f"Characters: {options}\n\n"
        "Reply with the name of the character who speaks the quoted line, "
        "exactly as it appears in the list above."
    )
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=GENERATOR_ENABLE_THINKING,
    )
    return prompt, names, gold_index


class GenDataset(Dataset):
    """Prompts for every quotation in a set of novels."""

    def __init__(self, novels: list[Novel], tokenizer, label: str = ""):
        self.examples: list[GenExample] = []
        for novel in novels:
            for quote in novel.quotes:
                prompt, names, gold_index = build_prompt(novel, quote, tokenizer)
                if not names:
                    # No candidate at all: unanswerable by construction. Kept
                    # so the denominator matches every other system's.
                    names, gold_index = [], -1
                self.examples.append(GenExample(
                    book=novel.name, qid=quote.qid, quote_type=quote.quote_type,
                    gold=quote.speaker, prompt=prompt, cand_names=names,
                    gold_index=gold_index, addressees=list(quote.addressees),
                ))
        if label:
            covered = sum(e.gold_index >= 0 for e in self.examples)
            logger.info("%s: %d prompts built, gold in candidates for %.1f%%",
                        label, len(self.examples),
                        100 * covered / max(len(self.examples), 1))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> GenExample:
        return self.examples[idx]


def _collate(batch: list[GenExample]) -> list[GenExample]:
    """Tokenisation happens on the GPU side, so the loader just groups."""
    return batch


def make_loader(novels, tokenizer, batch_size: int = GENERATOR_BATCH_SIZE,
                shuffle: bool = False, label: str = "") -> DataLoader:
    return DataLoader(GenDataset(novels, tokenizer, label=label),
                      batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_collate)


# ---------------------------------------------------------------- scoring


def _name_ids(tokenizer, name: str, cache: dict) -> list[int]:
    """Token ids for a candidate name, memoised across a whole novel."""
    ids = cache.get(name)
    if ids is None:
        ids = tokenizer(name, add_special_tokens=False).input_ids or [
            tokenizer.eos_token_id]
        cache[name] = ids
    return ids


@torch.no_grad()
def score_example(model, tokenizer, example: GenExample, device,
                  name_cache: dict, max_length: int = GENERATOR_MAX_LENGTH
                  ) -> tuple[list[float], list[float]]:
    """Log-likelihood of every candidate name for one quotation.

    Returns ``(mean_logprob, sum_logprob)``, one entry per candidate. The
    prompt is prefilled once; each candidate is then scored against that cache
    and the cache is cropped back to the prompt length, so the passage is read
    exactly once no matter how many candidates it has.
    """
    if not example.cand_names:
        return [], []

    ids = tokenizer(example.prompt, return_tensors="pt", truncation=True,
                    max_length=max_length).input_ids.to(device)
    # ``logits_to_keep=1`` is not an optimisation detail: Qwen3's vocabulary is
    # 151,936, so materialising logits for all ~1,100 prompt positions would
    # allocate 668 MB per forward pass in fp32 and OOM a 12 GB card before the
    # first candidate is scored. Only the last position is ever read.
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    prompt_len = ids.shape[1]
    # Distribution over the first answer token, read off the prompt's last
    # position. Every candidate's first token is scored from this one vector.
    first = F.log_softmax(out.logits[:, -1].float(), dim=-1)

    means, sums = [], []
    for name in example.cand_names:
        nid = _name_ids(tokenizer, name, name_cache)
        total = first[0, nid[0]].item()
        if len(nid) > 1:
            step = torch.tensor([nid[:-1]], device=device)
            cont = model(input_ids=step, past_key_values=cache, use_cache=True)
            lp = F.log_softmax(cont.logits.float(), dim=-1)
            total += sum(lp[0, i, nid[i + 1]].item() for i in range(len(nid) - 1))
            # Drop this candidate's tokens so the next one starts from the
            # prompt again rather than from "prompt + previous candidate".
            cache.crop(prompt_len)
        sums.append(total)
        means.append(total / len(nid))
    return means, sums


def predict(model, tokenizer, loader, device,
            length_normalise: bool = GENERATOR_LENGTH_NORMALISE,
            resume_path=None, save_every: int = 500
            ) -> tuple[list[tuple], dict[str, float]]:
    """Attribute every quotation, under both scoring rules.

    The secondary accuracy is not decoration: names differ in length ("Emma"
    vs "Mrs Fitzwilliam Darcy") and an unnormalised sequence log-probability
    systematically prefers the short one. Whether length normalisation changes
    the answer is the difference between a model that has learned the task and
    one ranking names by how short they are.
    """
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    rows: list[tuple] = []
    hits = {"normalised": 0, "unnormalised": 0, "n": 0}
    name_cache: dict = {}

    # Scoring the test split is over an hour of GPU work with no gradient to
    # lose, which makes it the one place where a crash used to cost the most
    # for the least reason. The loader is unshuffled, so the rows already on
    # disk correspond one-for-one to the first examples of the dataset and the
    # run resumes by fast-forwarding past them.
    if resume_path is not None and resume_path.exists():
        blob = json.loads(resume_path.read_text(encoding="utf-8"))
        rows = [tuple(r) for r in blob["rows"]]
        hits = blob["hits"]
        logger.info("resumed scoring from %s at %d/%d", resume_path.name,
                    len(rows), len(loader.dataset))
    skip, visited, scored = len(rows), 0, 0

    for batch in loader:
        for example in batch:
            visited += 1
            if visited <= skip:
                continue
            try:
                means, sums = score_example(model, tokenizer, example, device,
                                            name_cache)
            except torch.cuda.OutOfMemoryError:
                # One pathological passage must not end an unattended run. The
                # quote is recorded as unanswered, which counts against
                # accuracy exactly as a wrong answer would.
                logger.warning("OOM scoring %s/%s; recorded as no prediction",
                               example.book, example.qid)
                torch.cuda.empty_cache()
                means, sums = [], []
            scored += 1
            if not example.cand_names or not means:
                pred = None
            else:
                chosen = means if length_normalise else sums
                pred = example.cand_names[max(range(len(chosen)),
                                              key=chosen.__getitem__)]
                hits["n"] += 1
                if example.cand_names[
                        max(range(len(means)), key=means.__getitem__)
                ] == example.gold:
                    hits["normalised"] += 1
                if example.cand_names[
                        max(range(len(sums)), key=sums.__getitem__)
                ] == example.gold:
                    hits["unnormalised"] += 1
            rows.append((
                example.book, example.qid, example.quote_type, example.gold,
                pred, len(example.cand_names), example.gold_index >= 0,
                example.addressees,
            ))
        if scored and len(rows) % 500 < len(batch):
            logger.info("scored %d/%d", len(rows), len(loader.dataset))
        if (resume_path is not None and scored
                and len(rows) % save_every < len(batch)):
            _atomic_json(resume_path, {"rows": rows, "hits": hits})

    if resume_path is not None:
        _atomic_json(resume_path, {"rows": rows, "hits": hits})
    return rows, hits


def to_predictions(rows: list[tuple]) -> list:
    """Adapt raw output to the shared evaluation record."""
    from evaluate import Prediction

    return [
        Prediction(book=b, qid=q, quote_type=t, gold=g, pred=p,
                   n_candidates=n, gold_in_candidates=c, addressees=a)
        for b, q, t, g, p, n, c, a in rows
    ]


# ---------------------------------------------------------------- resume


def _atomic_json(path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def _save_train_state(model, optimiser, scheduler, resume_dir, epoch,
                      examples_done, best_acc, best_rows) -> None:
    """Persist everything needed to restart this run where it stopped.

    Only the adapter is written, not the model: the 4-bit base is frozen and
    byte-identical on every restart, so the whole cost of a checkpoint is the
    175MB of LoRA weights plus a small optimiser state. Cheap enough to do
    every few hundred examples, which is the point -- the first attempt at this
    run died 143 minutes in and left nothing behind.
    """
    resume_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(resume_dir / "adapter"))
    tmp = resume_dir / "trainer.pt.tmp"
    torch.save({
        "epoch": epoch,
        "examples_done": examples_done,
        "best_acc": best_acc,
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        "py_rng": random.getstate(),
    }, tmp)
    os.replace(tmp, resume_dir / "trainer.pt")
    if best_rows is not None:
        _atomic_json(resume_dir / "best_rows.json", best_rows)


def _load_train_state(model, optimiser, scheduler, resume_dir):
    """Restore a previous run, or return None if there is nothing to restore."""
    state_path = resume_dir / "trainer.pt"
    adapter_path = resume_dir / "adapter"
    if not (state_path.exists() and adapter_path.exists()):
        return None

    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(
        model, load_file(str(adapter_path / "adapter_model.safetensors")))
    blob = torch.load(state_path, map_location="cpu", weights_only=False)
    # The adapter weights are the part that must survive; optimiser moments are
    # a convenience. If bitsandbytes cannot rehydrate its 8-bit state across
    # processes, warming Adam up again over a few hundred examples costs far
    # less than refusing to resume at all, so this degrades instead of raising.
    try:
        optimiser.load_state_dict(blob["optimiser"])
        scheduler.load_state_dict(blob["scheduler"])
    except (ValueError, KeyError, RuntimeError, TypeError) as exc:
        logger.warning("could not restore optimiser state (%s); resuming with "
                       "fresh moments at the saved position", exc)
    torch.set_rng_state(blob["cpu_rng"])
    if blob.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    random.setstate(blob["py_rng"])
    rows_path = resume_dir / "best_rows.json"
    best_rows = None
    if rows_path.exists():
        best_rows = [tuple(r) for r in
                     json.loads(rows_path.read_text(encoding="utf-8"))]
    return blob["epoch"], blob["examples_done"], blob["best_acc"], best_rows


# ---------------------------------------------------------------- training


def train_model(model, tokenizer, train_examples, dev_loader, device,
                epochs: int = GENERATOR_EPOCHS,
                grad_accum: int = GENERATOR_GRAD_ACCUM,
                seed: int = SEED, ckpt_dir=None, resume_dir=None,
                save_every: int = 400):
    """LoRA-tune the decoder to continue the prompt with the gold speaker.

    Training is ordinary teacher-forced cross-entropy on the name tokens only:
    the prompt is masked out with -100, so no gradient is spent on modelling
    the novel's prose. The constraint to the candidate list is applied at
    inference, not here. No end-of-turn token is appended, because scoring at
    inference does not include one either -- training on a token the metric
    never sees would optimise a slightly different objective than the one
    reported.

    Quotes whose gold speaker is not among the candidates are dropped from
    training, matching Model 1: there is no reachable right answer to teach,
    and supervising toward one the model cannot select at test time only adds
    noise. They still count against accuracy at evaluation.
    """
    from transformers import get_linear_schedule_with_warmup

    set_seed(seed)
    usable = [e for e in train_examples if e.gold_index >= 0]
    logger.info("training on %d of %d quotes (gold reachable)",
                len(usable), len(train_examples))

    params = [p for p in model.parameters() if p.requires_grad]
    # 8-bit Adam, the standard QLoRA pairing: two fp32 moments over 44M adapter
    # parameters would be 350MB of state on a card that has already spent 5.5GB
    # on weights, and 8-bit quarters that.
    #
    # NOT the *paged* variant, which is what the QLoRA recipe nominally calls
    # for. Paged optimisers hold their state in CUDA unified memory so it can
    # spill to host RAM under pressure, and unified-memory oversubscription is
    # not supported by the Windows WDDM driver model. The first overnight
    # attempt died at 0xC0000005 -- an access violation inside the driver, with
    # no Python traceback -- 143 minutes into training, which is the signature
    # of exactly that. Paging buys nothing here in any case: 44M parameters of
    # 8-bit state is ~90MB, and it never needed to spill.
    optimiser = None
    if not os.environ.get("GENERATOR_TORCH_ADAMW"):
        try:
            from bitsandbytes.optim import AdamW8bit

            optimiser = AdamW8bit(params, lr=GENERATOR_LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
            logger.info("optimiser: bitsandbytes AdamW8bit (not paged)")
        except (ImportError, RuntimeError) as exc:
            logger.warning("bitsandbytes AdamW8bit unavailable (%s); "
                           "falling back to torch AdamW", exc)
    if optimiser is None:
        optimiser = torch.optim.AdamW(params, lr=GENERATOR_LEARNING_RATE,
                                      weight_decay=WEIGHT_DECAY)
        logger.info("optimiser: torch AdamW (fp32 moments)")
    steps_per_epoch = max(1, len(usable) // grad_accum)
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimiser, int(WARMUP_RATIO * total_steps), total_steps)

    best_acc, best_rows = -1.0, None
    order = list(range(len(usable)))

    start_epoch, start_i = 0, 0
    if resume_dir is not None:
        restored = _load_train_state(model, optimiser, scheduler, resume_dir)
        if restored is not None:
            start_epoch, start_i, best_acc, best_rows = restored
            logger.info("resumed at epoch %d, example %d/%d (best dev %.4f)",
                        start_epoch + 1, start_i, len(usable), best_acc)

    for epoch in range(start_epoch, epochs):
        model.train()
        model.config.use_cache = False      # incompatible with checkpointing
        random.Random(seed + epoch).shuffle(order)
        running, seen = 0.0, 0
        # Replaying the shuffle is what makes the skip below correct: the order
        # is a pure function of (seed, epoch), so a resumed process walks the
        # identical sequence and fast-forwards past the examples the previous
        # process already trained on rather than training on them twice.
        skip = start_i if epoch == start_epoch else 0
        if skip:
            logger.info("epoch %d: skipping %d already-trained examples",
                        epoch + 1, skip)

        for i, idx in enumerate(order, 1):
            if i <= skip:
                continue
            example = usable[idx]
            prompt_ids = tokenizer(example.prompt, truncation=True,
                                   max_length=GENERATOR_MAX_LENGTH).input_ids
            name_ids = tokenizer(example.gold, add_special_tokens=False).input_ids
            if not name_ids:
                continue
            ids = torch.tensor([prompt_ids + name_ids], device=device)
            n = len(name_ids)
            # Loss is computed here rather than by passing ``labels``, for the
            # same reason scoring passes ``logits_to_keep``: letting the model
            # emit logits over all ~1,100 positions costs 668 MB of fp32 that
            # is then almost entirely masked to -100 and thrown away. Keeping
            # n+1 positions gives exactly the ones that predict the name --
            # position P-1 predicts the first name token, and the final
            # position predicts nothing we supervise, so it is dropped.
            out = model(input_ids=ids, logits_to_keep=n + 1)
            logits = out.logits[:, :-1].float()
            target = torch.tensor([name_ids], device=device)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1)
            ) / grad_accum
            if torch.isnan(loss):
                continue
            loss.backward()
            running += loss.item() * grad_accum
            seen += 1

            if i % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad(set_to_none=True)
            if i % (grad_accum * 50) == 0:
                logger.info("epoch %d %d/%d loss %.4f", epoch + 1, i,
                            len(usable), running / max(seen, 1))
                running, seen = 0.0, 0
            # Only at an accumulation boundary, where the gradient buffers are
            # empty: checkpointing mid-accumulation would resume having dropped
            # a partial gradient the optimiser state does not account for.
            if (resume_dir is not None and i % save_every == 0
                    and i % grad_accum == 0):
                _save_train_state(model, optimiser, scheduler, resume_dir,
                                  epoch, i, best_acc, best_rows)
                logger.info("epoch %d checkpointed at example %d/%d",
                            epoch + 1, i, len(usable))
            # Windows has no expandable segments, so the alternation between
            # long prefills and tiny per-candidate forwards fragments the
            # caching allocator instead. Releasing it periodically is what
            # keeps a multi-hour run from reserving its way into an OOM.
            if sys.platform == "win32" and i % 1000 == 0:
                torch.cuda.empty_cache()

        optimiser.zero_grad(set_to_none=True)
        # Recorded before the dev pass, which is itself an hour of GPU work: a
        # crash inside scoring then resumes at the evaluation instead of
        # replaying the entire epoch of training that preceded it.
        if resume_dir is not None:
            _save_train_state(model, optimiser, scheduler, resume_dir,
                              epoch, len(order), best_acc, best_rows)
        dev_resume = (resume_dir / f"dev_epoch{epoch}.json"
                      if resume_dir is not None else None)
        rows, _ = predict(model, tokenizer, dev_loader, device,
                          resume_path=dev_resume)
        acc = sum(r[3] == r[4] for r in rows) / max(len(rows), 1)
        logger.info("epoch %d dev accuracy %.4f", epoch + 1, acc)
        if acc > best_acc:
            best_acc, best_rows = acc, rows
            if ckpt_dir is not None:
                # Only the adapter is saved: the 4-bit base is unchanged and
                # re-downloading it is cheaper than storing 5.5GB per epoch.
                model.save_pretrained(str(ckpt_dir))
                logger.info("epoch %d adapter saved to %s", epoch + 1, ckpt_dir)
        if resume_dir is not None:
            _save_train_state(model, optimiser, scheduler, resume_dir,
                              epoch + 1, 0, best_acc, best_rows)
            if dev_resume is not None:
                dev_resume.unlink(missing_ok=True)
    # The best epoch's dev predictions are handed back so the caller can skip
    # re-scoring dev after training. That is not a micro-optimisation: a dev
    # pass over 4,842 quotes costs about an hour on this card, and the rows are
    # already exact. They also come from the *best* epoch, whereas the model
    # left in memory is the last one -- so reusing them keeps the reported
    # accuracy consistent with the adapter that was actually saved.
    return best_acc, best_rows


# ---------------------------------------------------------------- entry point


def main() -> None:
    import argparse

    from config import MODEL_DIR, RESULTS_DIR
    from device import describe
    from evaluate import save, score
    from pdnc import load_split

    parser = argparse.ArgumentParser(description="train/evaluate the generator")
    parser.add_argument("split", nargs="?", default="dev")
    parser.add_argument("--zero-shot", action="store_true",
                        help="evaluate the instruction-tuned model untrained")
    parser.add_argument("--adapter", default="",
                        help="load a trained LoRA adapter and evaluate it")
    parser.add_argument("--epochs", type=int, default=GENERATOR_EPOCHS)
    parser.add_argument("--model", default=GENERATOR_MODEL)
    parser.add_argument("--no-4bit", action="store_true",
                        help="load in bf16 (needs ~2x the VRAM)")
    parser.add_argument("--limit-train", type=int, default=0,
                        help="use only the first N training novels (smoke test)")
    parser.add_argument("--limit-quotes", type=int, default=0,
                        help="cap training quotes; reported when set")
    parser.add_argument("--limit-eval", type=int, default=0,
                        help="score only the first N quotes (smoke test)")
    parser.add_argument("--limit-dev", type=int, default=0,
                        help="cap the per-epoch dev evaluation during training "
                             "(smoke test; makes checkpoint selection partial)")
    parser.add_argument("--fresh", action="store_true",
                        help="discard any resume state and start over")
    parser.add_argument("--save-every", type=int, default=400,
                        help="examples between training checkpoints; must be "
                             "a multiple of the gradient accumulation step")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    # Claim the card before anything else touches CUDA: the cap is applied on
    # the first call to get_device() and is not revisable afterwards.
    from device import LARGE_MODEL_FRACTION

    device = get_device(memory_fraction=LARGE_MODEL_FRACTION)
    four_bit = GENERATOR_LOAD_4BIT and not args.no_4bit
    training = not (args.zero_shot or args.adapter)

    novels = load_split(args.split)
    print(f"device : {device} - {describe()}")
    print(f"model  : {args.model}" + ("  [4-bit NF4]" if four_bit else "  [bf16]"))
    print(f"context: -{MODEL_CHARS_BEFORE}/+{MODEL_CHARS_AFTER} chars")
    print(f"{args.split:7s}: {len(novels)} novels, "
          f"{sum(len(n.quotes) for n in novels):,} quotes\n")

    model, tokenizer = load_backbone(args.model, four_bit=four_bit,
                                     for_training=training)
    reuse_rows = None

    if args.adapter:
        from peft import PeftModel

        name = f"generator-lora{args.tag}"
        model = PeftModel.from_pretrained(model, str(MODEL_DIR / args.adapter))
    elif args.zero_shot:
        name = f"generator-zeroshot{args.tag}"
    else:
        name = f"generator-lora{args.tag}"
        train_novels = load_split("train")
        if args.limit_train:
            train_novels = train_novels[: args.limit_train]
        train_examples = GenDataset(train_novels, tokenizer, label="train").examples
        if args.limit_quotes:
            random.Random(args.seed).shuffle(train_examples)
            train_examples = train_examples[: args.limit_quotes]
            print(f"train  : capped to {len(train_examples):,} quotes\n")
        dev_loader = make_loader(load_split("dev"), tokenizer, label="dev")
        if args.limit_dev:
            dev_loader.dataset.examples = \
                dev_loader.dataset.examples[: args.limit_dev]
            print(f"dev    : capped to {args.limit_dev:,} quotes for "
                  f"in-training checkpoint selection\n")
        resume_dir = MODEL_DIR / f"{name}.resume"
        if args.fresh and resume_dir.exists():
            shutil.rmtree(resume_dir, ignore_errors=True)
        if resume_dir.exists():
            print(f"resume : {resume_dir.name} present - continuing that run")
        best, trained_rows = train_model(
            model, tokenizer, train_examples, dev_loader, device,
            epochs=args.epochs, seed=args.seed,
            ckpt_dir=MODEL_DIR / f"{name}.adapter",
            resume_dir=resume_dir, save_every=args.save_every)
        print(f"\nbest dev accuracy {best:.4f}")
        # Training already scored the full dev split at the best epoch; scoring
        # it a second time would cost another hour for identical numbers.
        if (args.split == "dev" and not args.limit_dev and not args.limit_eval
                and trained_rows is not None):
            reuse_rows = trained_rows

    if reuse_rows is not None:
        rows, hits = reuse_rows, {"n": 0}
        print("(reusing the dev predictions scored during training)")
    else:
        loader = make_loader(novels, tokenizer, label=args.split)
        if args.limit_eval:
            loader.dataset.examples = loader.dataset.examples[: args.limit_eval]
        # Partial scores land beside the results they will become. A smoke test
        # gets no resume file: its truncated row list would be picked up by the
        # next full run and silently accepted as the first N scores.
        eval_resume = (None if args.limit_eval
                       else RESULTS_DIR / f"{name}_{args.split}.partial.json")
        if eval_resume is not None and args.fresh:
            eval_resume.unlink(missing_ok=True)
        rows, hits = predict(model, tokenizer, loader, device,
                             resume_path=eval_resume)
    report = score(name, to_predictions(rows))
    print()
    print(report)

    if hits["n"]:
        print(f"\n  scoring rule: length-normalised "
              f"{100*hits['normalised']/hits['n']:.1f}% vs unnormalised "
              f"{100*hits['unnormalised']/hits['n']:.1f}% "
              f"(of {hits['n']:,} quotes with candidates)")
    if not args.limit_eval:
        save(report, f"{name}_{args.split}.json")
        # Everything below is scaffolding for an interrupted run; the results
        # file above is the durable artefact, so the scaffolding goes only once
        # that has been written.
        (RESULTS_DIR / f"{name}_{args.split}.partial.json").unlink(missing_ok=True)
        if training:
            shutil.rmtree(MODEL_DIR / f"{name}.resume", ignore_errors=True)


if __name__ == "__main__":
    main()

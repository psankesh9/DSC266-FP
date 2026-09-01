# Who Said That? Speaker Attribution for Untagged Dialogue in Novels

DSC266 Final Project. Pratheek Sankeshi, solo.

## 1. Objective

Given a novel and a quotation in it, name the character who speaks it. The interesting part of the problem is not the average case but what it is made of, so every number in this report is broken out by how the speaker is signalled:

- explicit: the speech tag names the speaker ("Run," John said);
- anaphoric: the tag refers to the speaker without naming them (she whispered), so attribution requires resolving the pronoun;
- implicit: there is no speech tag at all, and the speaker must be inferred from who is present, who spoke last, and who is being addressed.

Pooled accuracy hides this structure, and hides it in a specific direction: a system can look strong while solving only the third of quotations that a regular expression already solves. My primary metric is therefore per-type accuracy on novels by authors absent from training, with confidence intervals bootstrapped over novels rather than quotations.

The concrete claims I set out to test were: (1) that framing attribution as ranking over enumerated candidates is sound, and that the enumeration ceiling this imposes is measurable and worth reporting as a result in its own right; (2) that the gains of a modern encoder come specifically from reading a window wide enough to contain the candidates it is asked to rank, not from the backbone alone; and (3) that a constrained decoder LLM, given the same context, beats a purpose-built ranker, and if so, whether it does that zero-shot or only after fine-tuning.

All three resolved cleanly, and one of them resolved against the prior I started with.

## 2. Background

Automatic attribution begins with Elson and McKeown (2010), who framed it as classification over syntactic patterns around the quote. Muzny et al. (2017) established the strongest non-neural approach: a two-stage sieve, a cascade of hand-written rules applied highest-precision-first with each assignment final. Sieves work well where the surface offers a pattern to match and degrade sharply where it does not, which is exactly the explicit/implicit divide this report is organised around.

Vishnubhotla et al. (2022) introduced the Project Dialogism Novel Corpus (PDNC), the first corpus large enough to measure that divide properly, with gold speaker, addressee, quote type, and a per-novel character gazetteer. Their follow-up (Vishnubhotla et al., 2023) reports the finding this project starts from: errors concentrate in implicit quotations and in novels the model has not seen.

Two recent results are stronger than anything here. Michel et al. (2024) add fictional-character embeddings; Michel et al. (2025) evaluate LLaMa-3 on PDNC directly. Both report higher accuracy than I do. Neither is comparable to my numbers, because both evaluate with novels split at the book level while PDNC contains six Austen novels and four Forster novels. A model can then learn one author's naming conventions in training and be rewarded for them at test time. I hold out whole authors, which is a harder setting and a lower number; I use those papers to anchor where the ceiling is, not as a target to beat.

## 3. Data

PDNC, current release: 28 novels, 37,131 quotations. PDNC leaves `quoteType` blank on 28 rows (0.08%); they are dropped rather than guessed, since a heuristic backfill would pollute the per-type breakdown that is the whole point of the report. That leaves 37,103 quotations.

The split is by author, not by novel. Austen contributes six books and Forster four, so a novel-level split would let a model lean on an author's conventions rather than learn attribution. The seven test novels are by seven authors who appear nowhere in training, and the split is additionally balanced on narrative person, because a first-person narrator speaks without ever being named and so changes the task rather than merely making it harder.

| Split | Novels | Authors | Quotations | Explicit | Anaphoric | Implicit | 1st-person (by quote) |
|---|---|---|---|---|---|---|---|
| train | 15 | 7 | 23,818 | — | — | — | 19% |
| dev | 6 | 6 | 4,842 | 1,722 | 1,750 | 1,370 | 40% |
| test | 7 | 7 | 8,443 | 2,498 | 2,390 | 3,555 | 22% |

Dev runs hotter on first-person than either train or test (40% vs 19%/22%). This is a deliberate accepted imbalance, since dev only selects checkpoints, but it has a consequence that matters for reading the results, discussed in §4.

The test split was scored exactly once, after every model and ablation was frozen on dev, as the proposal committed to.

## 4. Task formulation and the candidate ceiling

Attribution is framed as ranking: for each quotation, enumerate candidate speakers from a window of surrounding narration, then score each candidate. This is the standard formulation, and it imposes a hard ceiling: if the gold speaker is never enumerated, no amount of modelling recovers it. That ceiling is measurable, and measuring it turned out to be one of the more useful things in the project.

Sweeping the enumeration window on dev:

| Window (chars) | Mean candidates | All | Explicit | Anaphoric | Implicit |
|---|---|---|---|---|---|
| −500 / +200 | 2.52 | 70.2% | 95.1% | 61.0% | 50.8% |
| −900 / +400 | 3.38 | 78.0% | 95.4% | 72.0% | 63.7% |
| −1500 / +600 | 4.15 | 82.5% | 95.5% | 78.2% | 71.8% |
| **−2500 / +800** | **5.03** | **86.1%** | **95.8%** | **83.0%** | **77.8%** |
| −4000 / +1500 | 6.28 | 88.4% | 96.1% | 86.7% | 81.1% |

Explicit recall is flat at ~95-96% across a factor of eight in window width: the speaker of an explicitly tagged quote is next to the quote by definition. Every point of ceiling the window buys is bought on anaphoric and implicit quotes, which gain 26 and 30 points respectively over the same range. I settled on −2500/+800 as the operating point: the next step up buys 2.3 points of ceiling for 25% more candidates to score, and, more to the point, Model 1 must read the whole window for it to help (§5.2). Reading 4,000 characters of context was not affordable on this hardware.

Candidate enumeration uses PDNC's gazetteer plus names derived from the text. The derived names matter more than I expected: aliases alone reach a 78.3% ceiling at 4.25 candidates per quote, and adding derived names reaches 86.1% at 5.03, so that is +7.8 points of ceiling for 0.8 more candidates.

The ceiling differs between splits, and this is the single most important caveat in the report. It is 86.1% on dev and 90.1% on test, so the unreachable floor is 13.9% on dev against 9.9% on test. Test is a genuinely easier candidate problem than dev. Dev and test accuracies must therefore never be compared directly. Where a cross-split comparison is needed, I report share of reachable headroom (accuracy ÷ ceiling) instead.

The cause is visible per-novel. Dev ceilings range from 66.9% (*The Sign of the Four*) and 67.1% (*The Gambler*) to 99.1% (*Alice's Adventures in Wonderland*), and the two low outliers are both first-person: Watson and Alexei Ivanovich narrate, speak constantly, and are never named in the narration around their own dialogue. Dev's 40% first-person share is what drags its ceiling down. This is a limitation of the candidate formulation itself, not of any model in this report, and it is the largest single thing I would fix next (§11).

## 5. Systems

### 5.1 Baselines

Five, all operating over the same enumerated candidate set so the ceiling applies to them equally:

- random-candidate: chance within the candidate set, which reads as how much enumeration alone has narrowed the problem.
- most-frequent: the most-mentioned character in the window.
- nearest-mention: the nearest preceding character mention. This is the reference system for every paired interval.
- nearest+speech-verb: nearest mention, preferring one adjacent to a speech verb.
- alternation: assume two-party alternation and assign the speaker of two quotes back.

### 5.2 Model 1: the ModernBERT ranker

ModernBERT-base (149M) encodes the passage; a head scores each candidate span. The quote representation is a masked mean over the spoken sub-spans (which do not have to be contiguous, since split quotations are common); candidate representations are pooled over their mention spans. Both are projected to 256 dimensions and combined by a bilinear interaction plus an MLP over `[q, c, q*c, |q−c|, features]`. A candidate whose mentions all fell outside the encoded text gets a learned `absent` vector rather than a zero, so the model can place "named, but not in what I read" wherever the data says it belongs.

Eleven hand features supplement the encoder: log distance to the nearest mention, its zone (before/inside/after the quote), whether it sits beside a speech verb, whether the candidate has any tagged mention, whether it is the nearest candidate, mention counts raw and normalised, gender agreement between a tag pronoun and PDNC's annotated character gender, and whether the candidate fell inside the encoded window.

The decisive design choice is the window, not the backbone. The encoder reads −2500/+800 characters at 1,152 tokens, equal to the candidate enumeration window. Under the original RoBERTa configuration the encoder read −900/+400 at 512 tokens while candidates were still drawn from −2500/+800, which meant 38.2% of candidates fell outside the text the encoder saw and were scored from hand features alone. Matching the windows brings that to 0.2%. The 1,152-token budget is measured, not rounded: the longest dev passage is 1,037 ModernBERT tokens (p99 = 1,008), and attention is quadratic, so headroom past the measured maximum buys nothing.

The window is carried in a `Window` dataclass threaded through encoding, loading and training, and stored in the checkpoint. A model trained clamped has to be evaluated clamped, or the context ablation measures a train/test mismatch instead of a context effect.

### 5.3 Model 2: Qwen3-8B, constrained

Qwen3-8B, 4-bit NF4 with double quantisation (~5.5 GB), LoRA on all attention and MLP projections (r=16, α=32, dropout 0.05; 43.6M trainable of 4,761.5M, 0.92%). Quantisation here is a memory constraint rather than a design preference, and I report it as such.

The model never generates. Each candidate name is scored as a continuation of the prompt and the highest-scoring name wins, so the output is always a member of the candidate set and nothing is string-matched out of free text. Scores are length-normalised: names differ in length ("Emma" vs "Mrs Fitzwilliam Darcy") and an unnormalised sequence log-probability systematically prefers the short one, which matters far more zero-shot than after tuning. Qwen3's `<think>` block is disabled, since a reasoning span would sit between the prompt and the token positions being scored. The prompt is prefilled once per quotation and its KV cache reused across that quote's candidates.

The passage given to Qwen3 is the same −2500/+800 window ModernBERT reads, so neither model sees context the other cannot. The quotation is repeated after the passage as well as being marked inside it: in a split quotation the two halves are far apart, and a model reading a long window otherwise has to guess which quoted line the question is about.

## 6. Experimental setup

Single RTX 5070, 12.8 GB. Python 3.14, torch 2.13+cu130, transformers 5.13, peft 0.20, bitsandbytes 0.50.1.

Model 1: batch 8, lr 2e-5 (head 1e-4), 8 epochs, weight decay 0.01, 10% warmup, bf16 autocast, at most 12 candidates per quote, checkpoint selected by dev accuracy. Model 2: 1 epoch over 8,000 training quotations (7,337 with the gold speaker reachable), batch 1 × grad-accum 8, lr 1e-4, PagedAdamW8bit.

Three hardware constraints shaped the implementation rather than just inconveniencing it, and I report them because they bound what the comparison can claim. First, `logits_to_keep` must be passed on every Qwen3 forward: the vocabulary is 151,936, so full-sequence logits cost 668 MB per pass. Second, peft's `prepare_model_for_kbit_training` cannot be used, because it upcasts the unquantised 622M-parameter `lm_head` to fp32 at a cost of 2.5 GB; gradient checkpointing and input-grad enabling are called directly instead. Third, ModernBERT-large does not fit above batch 1 at 1,152 tokens (§8).

Intervals. All confidence intervals are 95% bootstrap over novels, not quotations. Books vary far more than systems do (per-novel test accuracy for the ranker spans 69.0% to 92.1%), and resampling quotations would report intervals several times too narrow. Contrasts between two systems use a paired bootstrap over the same novel resamples. `compare.py` scores every system against nearest-mention; the ablation contrasts are ranker-vs-ranker and so needed their own paired intervals, computed by `ablation_contrasts.py`.

## 7. Results

### Test (7 held-out novels, 8,443 quotations, scored once)

| System | All | Explicit | Anaphoric | Implicit | ÷ ceiling |
|---|---|---|---|---|---|
| random-candidate | 22.2% | 23.0% | 24.2% | 20.3% | 24.6% |
| alternation | 36.2% | 44.3% | 32.3% | 33.2% | 40.2% |
| most-frequent | 42.5% | 54.0% | 36.3% | 38.5% | 47.1% |
| nearest-mention | 49.5% | 89.6% | 32.4% | 32.9% | 55.0% |
| nearest+speech-verb | 50.3% | 87.8% | 33.2% | 35.6% | 55.9% |
| Qwen3-8B zero-shot | 77.2% | 84.3% | 77.2% | 72.2% | 85.6% |
| ModernBERT ranker | 82.3% | 95.1% | 79.2% | 75.4% | 91.3% |
| **Qwen3-8B LoRA** | **87.7%** | 94.0% | **85.7%** | **84.6%** | **97.2%** |
| *candidate ceiling* | *90.1%* | — | — | — | *100%* |

Paired against nearest-mention, bootstrapped over novels:

| System | All | Explicit | Anaphoric | Implicit |
|---|---|---|---|---|
| ModernBERT ranker | +32.8 [+26.3, +37.0]\* | +5.5 [+3.6, +7.5]\* | +46.9 [+33.2, +55.5]\* | +42.5 [+37.3, +45.9]\* |
| Qwen3-8B zero-shot | +27.6 [+17.3, +34.8]\* | −5.3 [−14.4, −0.8]\* | +44.8 [+26.1, +56.0]\* | +39.3 [+30.8, +46.2]\* |
| Qwen3-8B LoRA | +38.1 [+30.6, +43.6]\* | +4.4 [+0.6, +6.4]\* | +53.3 [+41.0, +60.5]\* | +51.6 [+45.7, +56.6]\* |

\* interval excludes zero.

The ordering is the same on dev (LoRA 82.7%, ranker 79.0%, zero-shot 74.1%, nearest-mention 48.9%, ceiling 86.1%), and the headroom column confirms the ordering is not an artefact of the easier test candidate problem: LoRA takes 96.1% of reachable headroom on dev and 97.2% on test.

Two things are worth reading off the per-type columns rather than the pooled one. The baseline is not weak, it is narrow. Nearest-mention gets 89.6% of explicit quotes, which is close to all a shallow method can be expected to get, and 32.4%/32.9% on the other two types, barely above the 22.2% you get by guessing within the candidate set. Its pooled 49.5% is almost entirely the explicit third. And the models' gains land exactly where the framing predicted: +46.9 and +42.5 points for the ranker on anaphoric and implicit, against +5.5 on explicit.

## 8. Ablations

All contrasts paired over the six dev novels.

| Contrast | All | Anaphoric | Implicit |
|---|---|---|---|
| Window −2500/+800 @1152 vs clamped 512 | +3.3 [+1.2, +4.9]\* | +4.9 [+2.4, +6.9]\* | +4.7 [+0.6, +6.4]\* |
| ModernBERT vs RoBERTa (window matched) | +2.1 [+0.8, +3.0]\* | +4.0 [+2.3, +5.0]\* | +2.2 [+0.3, +3.1]\* |
| Hand features on vs zeroed | +0.5 [−0.0, +1.0] | +1.8 [+0.7, +2.5]\* | −0.6 [−1.9, +0.0] |
| Capacity: base vs large (**confounded**) | +0.1 [−0.7, +1.0] | +1.4 [−0.9, +3.0] | −1.3 [−3.5, +2.0] |
| Capacity: base vs large (**batch matched**) | −1.9 [−3.1, −0.2]\* | −1.9 [−3.6, −0.1]\* | −4.1 [−6.8, +0.2] |
| Qwen3-8B LoRA vs ModernBERT ranker | +3.8 [+1.0, +5.9]\* | +4.1 [+2.2, +5.2]\* | +8.2 [+4.0, +12.0]\* |
| LoRA vs zero-shot (same 8B model) | +8.6 [+5.4, +11.1]\* | +8.2 [+4.8, +10.2]\* | +9.4 [+6.2, +12.4]\* |

Window (+3.3). This is the project's central mechanistic claim, and it holds. Crucially it holds where the mechanism predicts: +4.9 anaphoric and +4.7 implicit against +0.5 [+0.1, +0.9] on explicit. The explicit effect is separable from zero but roughly ten times smaller. Widening the window helps precisely the quotes whose speaker was named several turns back, and barely moves quotes whose speaker is named beside them. Separating this from the backbone required running RoBERTa at its own native window as a third condition, since the naive comparison confounds the two.

Backbone (+2.1). ModernBERT beats RoBERTa by 2.1 points with the window held equal, so the total encoder gain of roughly 5.4 points splits into about 3.3 from reading more and 2.1 from the better-pretrained encoder. Neither number alone would have supported the claim the proposal made. The backbone gain is even more sharply concentrated than the window gain: +4.0 anaphoric and +2.2 implicit against +0.0 [−0.2, +0.1] on explicit. Whatever ModernBERT's larger pretraining corpus bought, it bought nothing at all on the quotes a regular expression already solves.

Hand features (+0.5, not significant). The interval touches zero, and the effect is negative on implicit quotes (−0.6). The honest reading is that the eleven features are near-redundant given an encoder that can now see the whole window, because the encoder learns distance and recency from the text itself. They buy a real +1.8 on anaphoric, which is where gender agreement has something to say that the text does not make easy. I do not claim the feature set helps overall. The ablation zeroes the features rather than removing the slice, so the head keeps its shape, parameter count and initialisation; the contrast measures what the features knew, not how much smaller the model got.

Capacity (+1.9 for the larger encoder, once the recipe is matched). The first large run is uninterpretable, and is kept in the table above to show why. ModernBERT-large does not fit above batch 1 at 1,152 tokens on 12.8 GB. Batch 4 and batch 2 both OOM within two minutes, batch 4 dying with 9.71 GB allocated while trying to add 26 MB, so it trained at batch 1 for 6 epochs against base's batch 8 for 8 epochs with the learning rate held at 2e-5. Batch size and epoch count are confounded with capacity there, and its +0.1 supports no conclusion about model size.

Gradient accumulation removes the confound. Re-running large at batch 1 × 8 accumulation reproduces base's effective batch of 8, at the same 8 epochs, the same 2e-5 encoder rate and 1e-4 head rate, leaving the encoder, 395M parameters against 149M, as the only thing that differs. The result reverses the null. Large reaches 80.83% dev against base's 78.98%, and the paired contrast is separable from novel-to-novel variation: −1.9 [−3.1, −0.2] in the table's base-minus-large direction, which is to say the larger encoder wins by 1.9 points. Its avoidable error is 5.2% against base's 7.1%. As with every other gain in this project, the effect is concentrated off the easy quotes: −1.9 [−3.6, −0.1] anaphoric and −4.1 [−6.8, +0.2] implicit, against −0.1 [−0.2, +0.1] on explicit.

Two things keep the claim modest. It is dev-selected and dev-reported, like every ablation in this section. And it is expensive: 1.9 points is a little over half what widening the window bought (+3.3), from a model 2.6× the size that trained at 37 minutes per epoch against base's 15, so five hours against two. It is also a floor rather than an estimate, because the two runs were not equally converged at their shared budget: base had flattened by epoch 6 (78.85, 78.95, 78.98 across its last three epochs) while large took its best at epoch 8 and was still improving when the epochs ran out.

Both findings stand, and they are different findings. The OOM is a genuine result about this hardware: at this context length, 395M parameters is untrainable on a 12 GB card at any batch above 1, and only gradient accumulation makes a matched comparison possible at all. And capacity does help, by roughly half what the window bought, for about two and a half times the compute.

Model size, downward. Qwen3-1.7B zero-shot collapses to 35.2% on dev, below every baseline except random, and 13.8 points below nearest-mention. Constrained scoring does not rescue a model that cannot follow the task, so the 8B result is not a generic "LLMs are good at this."

## 9. Error analysis

Errors split into unreachable (gold speaker never enumerated, which is the ceiling) and avoidable (gold was in the candidate set and the system picked something else). Only the second is a modelling failure, and it is the one to report.

| System (test) | Correct | Unreachable | Avoidable | of which named the addressee |
|---|---|---|---|---|
| nearest-mention | 49.5% | 9.9% | 40.6% | 59.0% |
| Qwen3-8B zero-shot | 77.2% | 9.9% | 13.0% | 70.0% |
| ModernBERT ranker | 82.3% | 9.9% | 7.8% | 58.9% |
| Qwen3-8B LoRA | 87.7% | 9.9% | **2.5%** | **40.0%** |

Finding 1: zero-shot Qwen3-8B is worse than the trivial baseline on explicit quotes. 84.3% against nearest-mention's 89.6%, a paired −5.3 [−14.4, −0.8], separable from zero. An 8B instruction-tuned model, given a passage that literally contains "…," said Hercule Poirot, picks someone else more often than a rule that takes the nearest preceding name. It over-reasons about who ought to be speaking given the scene. Fine-tuning flips the sign to +4.4 [+0.6, +6.4]. A substantial part of what LoRA teaches on this task is not new knowledge but restraint: stop overthinking quotes whose speaker is named right there. This is also why the pooled number understates the zero-shot model's real weakness: it is genuinely good at the hard types (77.2% anaphoric, within two points of the ranker) while being bad at the easy one.

Finding 2: addressee confusion is the dominant residual error mode, and fine-tuning attacks it directly. Of avoidable errors, the share that named the person being spoken to rather than the speaker: zero-shot 70.0%, ranker 58.9%, LoRA 40.0%. In dialogue both parties are named nearby and the local cues are near-symmetric; telling them apart requires tracking conversational structure, not proximity. LoRA's total avoidable error is 2.5% against a 9.9% unreachable floor, so it solves about 97% of what candidate generation makes solvable, and of the 12.3 points of error remaining between LoRA and a perfect score, four-fifths is enumeration failure rather than ranking failure. Further gains on this task should come from candidate enumeration, not from a better ranker.

Per-novel variation is larger than the gaps between systems, and it tracks the ceiling. Test ranker accuracy runs from 69.0% (*The Invisible Man*) to 92.1% (*Anne of Green Gables*); LoRA runs 68.9% to 96.4%. But *The Invisible Man*, the hardest book for every system and the only one where LoRA does not beat the ranker, is not hard to rank. Its candidate ceiling is 71.5%, the lowest in the split, and both systems reach about 96.5% of that reachable headroom. It is hard because Wells names people sparsely in narration: it averages 2.36 candidates per quotation against 4.76 across the test split, so enumeration frequently never proposes the speaker at all. The same pattern explains second-hardest *The Mysterious Affair at Styles* (ceiling 82.5%, first-person Hastings narrating). Raw per-novel accuracy is therefore mostly a measure of how enumerable a novel's characters are, which is a further reason to report the ceiling alongside every number. This variance is also why intervals are bootstrapped over novels; a quotation-level bootstrap would have reported roughly a third of the width and called differences significant that are not.

## 10. Extrinsic evaluation: an audiobook error metric

Attribution errors are not equally costly. Misattributing a two-word interjection and misattributing a long speech count as one error each under accuracy, but in a multi-voice audiobook the second is far worse. I therefore weight each error by the duration of the quotation, at 150 words per minute, and report seconds of mis-voiced audio per minute of audiobook.

This is worked out from a speaking rate, not by synthesising audio and not by a listening study. It is a length-weighted error rate expressed in seconds, and I claim nothing more for it.

Test (61.1 h of audio, 39% of it dialogue), paired against nearest-mention:

| System | s / audio min | s / dialogue min | h mis-voiced | Length bias | vs nearest-mention |
|---|---|---|---|---|---|
| nearest-mention | 11.52s | 29.35s | 11.73h | 0.97 | — |
| ModernBERT ranker | 4.57s | 11.64s | 4.65h | 1.10 | −6.95s [−9.10, −5.55]\* |
| Qwen3-8B zero-shot | 4.55s | 11.58s | 4.63h | 0.85 | −6.98s [−9.83, −4.86]\* |
| Qwen3-8B LoRA | **2.71s** | **6.90s** | **2.76h** | 0.93 | **−8.82s [−11.70, −6.68]\*** |

Every system improves separably on the baseline, and LoRA turns 11.7 hours of mis-voiced audio into 2.8, separably better than the ranker on this metric too, at a paired −1.86s [−2.79, −1.11]. But the ranker's five-point accuracy advantage over zero-shot does not survive the re-weighting. The two are indistinguishable in seconds (paired +0.03s [−1.78, +1.44]), and the gap is inseparable on dev as well (−0.69s [−1.83, +0.21]) even though the point estimates order correctly there.

The length-bias column is why, and it is the one thing this metric can say that accuracy cannot. It is the duration-weighted error rate divided by the quote-weighted one: above 1.0 a system errs on longer-than-average quotations, below 1.0 on shorter ones. The ranker sits at 1.10 and zero-shot at 0.85. The ranker is wrong less often, 17.7% of quotations against 22.8%, but it is wrong on longer speeches, and the two effects cancel almost exactly. To a listener the two systems would be about equally annoying. This does not overturn the intrinsic result, which accuracy measures correctly; it says the intrinsic gap between those two systems buys less than it appears to in a setting where errors cost time. Only LoRA improves on both metrics at once. It is both the most accurate and mildly short-biased (0.93), which is the sense in which the extrinsic metric backs it up while declining to separate the other two.

One caveat needs stating carefully. Better systems put a larger share of their remaining mis-voiced audio into implicit quotes: 66.7% for the ranker and 65.5% for LoRA, against 62.5% for nearest-mention. This is not a regression: the within-type error rate on implicit quotes falls from 68.8% (baseline) to 29.1% (ranker) to 16.9% (LoRA). The share rises because the easy types have been solved and implicit quotes are what remains. Stated carelessly it reads exactly backwards.

## 11. Limitations

The candidate ceiling is the binding constraint, and first-person narration is why. 9.9% of test quotations cannot be answered under this formulation. The failure is systematic, not random: a first-person narrator speaks constantly and is never named in the surrounding narration, so novels like *The Sign of the Four* and *The Gambler* have ceilings near 67%. Adding an explicit narrator candidate to every quotation is the obvious fix and the first thing I would do next. I did not do it here because it changes the candidate set that every number in this report is built on.

Fine-tuning budget is unequal. Model 1 trains on all 23,818 training quotations for 8 epochs; Model 2 on 8,000 for 1 epoch, because an 8B model at 4-bit on this card is roughly an order of magnitude slower per example. LoRA wins anyway, so the direction of the result is safe, but the size of the LoRA-vs-ranker gap is a lower bound, not an estimate.

Single seed. Every model is trained once. The intervals in this report capture variation across novels, which dominates, but not variation across initialisations. The +0.5 feature contrast in particular is well within the range a seed could move, and the matched capacity contrast, separable but with a lower bound of 0.2 points, is not far outside it.

The gender-agreement feature uses PDNC's annotated character gender, which is corpus metadata rather than a signal derivable from raw text; a system run on an un-annotated novel would not have it. This is part of why the feature ablation was run at all, and conveniently, that ablation shows the feature set is close to redundant, so the dependency is not load-bearing.

Quantisation is uncontrolled. Qwen3-8B is evaluated only at 4-bit NF4. I cannot separate what the model would do in bf16 from what it does quantised, because bf16 8B weights do not fit in 12.8 GB.

## 12. Conclusion

Framing quotation attribution as constrained ranking works, and both models beat the nearest-mention baseline by wide, novel-bootstrapped margins on author-disjoint test novels: +32.8 for the ModernBERT ranker and +38.1 for LoRA-tuned Qwen3-8B, against a candidate ceiling of 90.1%.

The three claims resolved as follows. The enumeration ceiling is real, measurable, and now the binding constraint: LoRA leaves only 2.5% avoidable error against a 9.9% unreachable floor, so the next gain on this task comes from candidate generation, not from ranking. The window matters, separably from the backbone: +3.3 for reading the full candidate window and +2.1 for the better encoder, with the window's gain concentrated on anaphoric and implicit quotes exactly as the mechanism predicts. And the decoder LLM beats the ranker only after fine-tuning. Zero-shot it is not merely worse overall but worse than a trivial rule on explicit quotes, which was the result I did not expect and which turned out to be the most informative one in the project: most of what LoRA teaches an 8B model here is when not to reason.

## References

1. Vishnubhotla, K., Hammond, A., and Hirst, G. (2022). The Project Dialogism Novel Corpus: A Dataset for Quotation Attribution in Literary Texts. *LREC 2022*, 5838-5848.
2. Vishnubhotla, K., Rudzicz, F., Hirst, G., and Hammond, A. (2023). Improving Automatic Quotation Attribution in Literary Novels. *ACL 2023 (Short Papers)*, 737-746.
3. Muzny, G., Fang, M., Chang, A., and Jurafsky, D. (2017). A Two-stage Sieve Approach for Quote Attribution. *EACL 2017*, 460-470.
4. Elson, D. K., and McKeown, K. R. (2010). Automatic Attribution of Quoted Speech in Literary Narrative. *Proceedings of AAAI 2010*.
5. Michel, G., Epure, E. V., Hennequin, R., and Cerisara, C. (2024). Improving Quotation Attribution with Fictional Character Embeddings. *Findings of EMNLP 2024*, 12723-12735.
6. Michel, G., Epure, E. V., Hennequin, R., and Cerisara, C. (2025). Evaluating LLMs for Quotation Attribution in Literary Texts: A Case Study of LLaMa3. *NAACL 2025 (Short Papers)*, 742-755.
7. Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., and Poli, I. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. *arXiv:2412.13663*.
8. Qwen Team (2025). Qwen3 Technical Report. *arXiv:2505.09388*.

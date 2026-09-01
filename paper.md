# Who Said That? Speaker Attribution for Untagged Dialogue in Novels

Pratheek Sankeshi. DSC266 Final Project, 2 September 2026.

## Abstract

Quotation attribution asks which character speaks each quotation in a novel. Pooled accuracy hides the structure of the task: roughly a third of quotations carry an explicit speech tag that a simple rule already solves, while anaphoric and implicit quotations need coreference and some tracking of conversational structure. We frame attribution as ranking over candidates enumerated from a narration window, measure the ceiling that enumeration imposes, and compare a ModernBERT span ranker against a constrained Qwen3-8B decoder on author-disjoint splits of PDNC. On seven held-out novels, LoRA-tuned Qwen3-8B reaches 87.7% and the ranker 82.3%, against 49.5% for a nearest-mention baseline and a 90.1% enumeration ceiling. Ablations separate the encoder's gain into +3.3 points from reading the full candidate window and +2.1 from the backbone itself. Zero-shot, the 8B model is worse than the trivial baseline on explicit quotations (-5.3), so a lot of what fine-tuning teaches is restraint. The remaining error is dominated by addressee confusion and by enumeration failure rather than ranking failure.

## 1. Introduction

Given a novel and a quotation in it, name the character who speaks it. Attribution underpins character-level analysis of fiction and multi-voice audiobook narration, where a single misattributed line breaks the listener's sense of a scene.

The interesting part of the problem is not the average case but what it is made of, so every number in this paper is broken out by how the speaker is signalled: explicit, where the speech tag names the speaker ("Run," John said); anaphoric, where the tag refers to the speaker without naming them (she whispered), so attribution requires resolving the pronoun; and implicit, where there is no speech tag at all and the speaker must be inferred from who is present, who spoke last, and who is being addressed.

Pooled accuracy hides this structure, and hides it in a specific direction: a system can look strong while solving only the third of quotations that a regular expression already solves. So our main metric is per-type accuracy on novels by authors absent from training, with confidence intervals bootstrapped over novels rather than quotations.

We wanted to test three claims. First, that framing attribution as ranking over enumerated candidates is sound, and that the enumeration ceiling this imposes is measurable and worth reporting as a result in its own right. Second, that the gains of a modern encoder come specifically from reading a window wide enough to contain the candidates it is asked to rank, not from the backbone alone. Third, that a constrained decoder LLM, given the same context, beats a purpose-built ranker, and if so, whether it does that zero-shot or only after fine-tuning. All three came out cleanly, and the third went against what we expected.

Our main contributions are the enumeration ceiling, measured as a result in its own right and shown to bind harder than ranking quality does, an ablation separating context width from backbone quality, and the zero-shot failure on the easiest quote type.

## 2. Background

Automatic attribution begins with Elson and McKeown (2010), who framed it as classification over syntactic patterns around the quote. Muzny et al. (2017) set the strongest non-neural approach: a two-stage sieve, a cascade of hand-written rules applied highest-precision-first with each assignment final. Sieves work well where the surface offers a pattern to match and degrade sharply where it does not, which is exactly the explicit/implicit divide this paper is built around.

Vishnubhotla et al. (2022) introduced the Project Dialogism Novel Corpus (PDNC), the first corpus large enough to measure that divide properly, with gold speaker, addressee, quote type, and a per-novel character gazetteer. Their follow-up (Vishnubhotla et al., 2023) reports the finding this project starts from: errors concentrate in implicit quotations and in novels the model has not seen.

Two recent results are stronger than anything here. Michel et al. (2024) add fictional-character embeddings; Michel et al. (2025) evaluate LLaMa-3 on PDNC directly. Both report higher accuracy than we do. Neither is really comparable to our numbers, though, because both evaluate with novels split at the book level while PDNC contains six Austen novels and four Forster novels. A model can then learn one author's naming conventions in training and be rewarded for them at test time. We hold out whole authors, which is a harder setting and a lower number; we use those papers to anchor where the ceiling is, not as a target to beat.

## 3. Methods

### 3.1 Data and splits

PDNC, current release: 28 novels, 37,131 quotations. PDNC leaves quoteType blank on 28 rows (0.08%); they are dropped rather than guessed, since a heuristic backfill would muddy the per-type breakdown that is the whole point of the paper. That leaves 37,103 quotations.

The split is by author, not by novel. Austen contributes six books and Forster four, so a novel-level split would let a model lean on an author's conventions rather than learn attribution. The seven test novels are by seven authors who appear nowhere in training, and the split is additionally balanced on narrative person, because a first-person narrator speaks without ever being named and so changes the task rather than merely making it harder.

| Split | Novels | Authors | Quotations | Explicit | Anaphoric | Implicit | 1st-person |
|---|---|---|---|---|---|---|---|
| train | 15 | 7 | 23,818 | - | - | - | 19% |
| dev | 6 | 6 | 4,842 | 1,722 | 1,750 | 1,370 | 40% |
| test | 7 | 7 | 8,443 | 2,498 | 2,390 | 3,555 | 22% |

Table 1: Split composition. Quote-type counts are given for the evaluation splits.

Dev runs hotter on first-person than either train or test (40% against 19% and 22%). This is a deliberate trade-off, since dev only selects checkpoints, but it has a consequence discussed in 3.2. The test split was scored exactly once, after every model and ablation was frozen on dev.

### 3.2 Task formulation and the candidate ceiling

Attribution is framed as ranking: for each quotation, enumerate candidate speakers from a window of surrounding narration, then score each candidate. This is the standard formulation, and it imposes a hard ceiling: if the gold speaker is never enumerated, no amount of modelling recovers it. That ceiling is measurable, and measuring it turned out to be one of the more useful things in the project.

Sweeping the enumeration window on dev (Figure 1), explicit recall is flat at roughly 95 to 96% across a factor of eight in window width, because the speaker of an explicitly tagged quote is next to the quote by definition. Every point of ceiling the window buys is bought on anaphoric and implicit quotations, which gain 26 and 30 points respectively over the same range. We settled on -2500/+800 characters as the operating point, reaching an 86.1% dev ceiling at 5.03 candidates per quotation; the next step up buys 2.3 points of ceiling for 25% more candidates to score, and Model 1 must read the whole window for it to help.

Candidate enumeration uses PDNC's gazetteer plus names derived from the text. The derived names matter more than expected: aliases alone reach a 78.3% ceiling at 4.25 candidates per quotation, and adding derived names reaches 86.1% at 5.03, so that is +7.8 points of ceiling for 0.8 more candidates.

The ceiling differs between splits, and this is the single most important caveat in the paper. It is 86.1% on dev and 90.1% on test, so the unreachable floor is 13.9% on dev against 9.9% on test. Test is a genuinely easier candidate problem than dev, and dev and test accuracies must therefore never be compared directly. Where a cross-split comparison is needed we report share of reachable headroom (accuracy divided by ceiling) instead. The cause is visible per-novel: dev ceilings range from 66.9% (The Sign of the Four) and 67.1% (The Gambler) to 99.1% (Alice's Adventures in Wonderland), and the two low outliers are both first-person, where the narrator speaks constantly and is never named in the surrounding narration.

### 3.3 Systems

Five baselines operate over the same enumerated candidate set, so the ceiling applies to them equally: random-candidate (chance within the candidate set), most-frequent (the most-mentioned character in the window), nearest-mention (the nearest preceding character mention, and the reference system for every paired interval), nearest+speech-verb, and alternation (assume two-party alternation and assign the speaker of two quotations back).

Model 1 is a ModernBERT-base (149M) span ranker. The encoder runs once over the surrounding narration; the quote and each candidate mention are pooled out of that single contextual sequence and scored against each other by a bilinear interaction plus an MLP over 256-dimensional projections. The quote representation is a masked mean over the spoken sub-spans, which do not have to be contiguous since split quotations are common. Eleven hand features supplement the encoder, covering distance and zone relative to the quote, speech-verb adjacency, mention counts, gender agreement between a tag pronoun and PDNC's annotated character gender, and whether the candidate fell inside the encoded window. A previous-speaker feature was planned but not added: the alternation baseline isolates that signal on its own, and the feature ablation in 4.2 suggests it would be redundant given an encoder that reads the whole window and can see the preceding turns directly. A candidate whose mentions all fell outside the encoded text gets a learned absent vector rather than a zero. PDNC's character category label is never used, since it is derived from how much dialogue a character speaks and is therefore a function of the answer.

The design choice that matters most is the window, not the backbone. The encoder reads -2500/+800 characters at 1,152 tokens, equal to the candidate enumeration window. Under the original RoBERTa configuration the encoder read -900/+400 at 512 tokens while candidates were still drawn from -2500/+800, which meant 38.2% of candidates fell outside the text the encoder saw and were scored from hand features alone. Matching the windows brings that to 0.2%. The window is stored in the checkpoint, because a model trained clamped has to be evaluated clamped or the context ablation measures a train/test mismatch instead of a context effect.

Model 2 is Qwen3-8B at 4-bit NF4, LoRA on all attention and MLP projections (r=16, alpha=32, 43.6M trainable of 4,761.5M); quantisation is a memory constraint rather than a design preference. The model never generates: each candidate name is scored as a continuation of the prompt and the highest-scoring name wins, so the output is always a member of the candidate set. Scores are length-normalised, since an unnormalised sequence log-probability systematically prefers the shorter name. Qwen3's thinking block is disabled, because a reasoning span would sit between the prompt and the token positions being scored. Model 2 reads the same window as Model 1, so neither sees context the other cannot.

### 3.4 Experimental setup

All runs use a single RTX 5070 with 12.8 GB. Model 1 trains at batch 8, lr 2e-5 (head 1e-4), 8 epochs, with the checkpoint selected by dev accuracy. Model 2 trains for 1 epoch over 8,000 quotations at batch 1 with gradient accumulation 8, lr 1e-4. Memory limits what the comparison can claim: full-sequence logits over Qwen3's 151,936-token vocabulary cost 668 MB per forward pass and must be suppressed, and ModernBERT-large does not fit above batch 1 at 1,152 tokens, which shapes the capacity ablation in 4.2.

All confidence intervals are 95% bootstrap over novels, not quotations. Books vary far more than systems do (per-novel test accuracy for the ranker spans 69.0% to 92.1%), and resampling quotations would give intervals several times too narrow. Contrasts between two systems use a paired bootstrap over the same novel resamples.

## 4. Results and discussion

### 4.1 Main results

| System | All | Expl. | Anap. | Impl. | / ceil. |
|---|---|---|---|---|---|
| random-candidate | 22.2% | 23.0% | 24.2% | 20.3% | 24.6% |
| alternation | 36.2% | 44.3% | 32.3% | 33.2% | 40.2% |
| most-frequent | 42.5% | 54.0% | 36.3% | 38.5% | 47.1% |
| nearest-mention | 49.5% | 89.6% | 32.4% | 32.9% | 55.0% |
| nearest+speech-verb | 50.3% | 87.8% | 33.2% | 35.6% | 55.9% |
| Qwen3-8B zero-shot | 77.2% | 84.3% | 77.2% | 72.2% | 85.6% |
| ModernBERT ranker | 82.3% | 95.1% | 79.2% | 75.4% | 91.3% |
| Qwen3-8B LoRA | 87.7% | 94.0% | 85.7% | 84.6% | 97.2% |
| candidate ceiling | 90.1% | - | - | - | 100% |

Table 2: Test accuracy, 7 held-out novels, 8,443 quotations, scored once.

| System | All | Explicit | Anaphoric | Implicit |
|---|---|---|---|---|
| ModernBERT ranker | +32.8 [+26.3, +37.0]* | +5.5 [+3.6, +7.5]* | +46.9 [+33.2, +55.5]* | +42.5 [+37.3, +45.9]* |
| Qwen3-8B zero-shot | +27.6 [+17.3, +34.8]* | -5.3 [-14.4, -0.8]* | +44.8 [+26.1, +56.0]* | +39.3 [+30.8, +46.2]* |
| Qwen3-8B LoRA | +38.1 [+30.6, +43.6]* | +4.4 [+0.6, +6.4]* | +53.3 [+41.0, +60.5]* | +51.6 [+45.7, +56.6]* |

Table 3: Paired against nearest-mention, bootstrapped over novels. An asterisk marks an interval excluding zero.

The ordering is the same on dev (LoRA 82.7%, ranker 79.0%, zero-shot 74.1%, nearest-mention 48.9%, ceiling 86.1%), and the headroom column confirms it is not an artefact of the easier test candidate problem: LoRA takes 96.1% of reachable headroom on dev and 97.2% on test.

Two things are worth reading off the per-type columns rather than the pooled one. The baseline is not weak, it is narrow: nearest-mention gets 89.6% of explicit quotations, close to all a shallow method can be expected to get, and 32.4% and 32.9% on the other two types, barely above the 22.2% obtained by guessing within the candidate set. Its pooled 49.5% is almost entirely the explicit third. And the models' gains land exactly where the framing predicted, at +46.9 and +42.5 points for the ranker on anaphoric and implicit against +5.5 on explicit.

### 4.2 Ablations

| Contrast | All | Anaphoric | Implicit |
|---|---|---|---|
| Window -2500/+800 vs clamped 512 | +3.3 [+1.2, +4.9]* | +4.9 [+2.4, +6.9]* | +4.7 [+0.6, +6.4]* |
| ModernBERT vs RoBERTa (window matched) | +2.1 [+0.8, +3.0]* | +4.0 [+2.3, +5.0]* | +2.2 [+0.3, +3.1]* |
| Hand features on vs zeroed | +0.5 [-0.0, +1.0] | +1.8 [+0.7, +2.5]* | -0.6 [-1.9, +0.0] |
| Capacity: large vs base (confounded) | -0.1 [-1.0, +0.7] | -1.4 [-3.0, +0.9] | +1.3 [-2.0, +3.5] |
| Capacity: large vs base (batch matched) | +1.9 [+0.2, +3.1]* | +1.9 [+0.1, +3.6]* | +4.1 [-0.2, +6.8] |
| Qwen3-8B LoRA vs ModernBERT ranker | +3.8 [+1.0, +5.9]* | +4.1 [+2.2, +5.2]* | +8.2 [+4.0, +12.0]* |
| LoRA vs zero-shot (same 8B model) | +8.6 [+5.4, +11.1]* | +8.2 [+4.8, +10.2]* | +9.4 [+6.2, +12.4]* |

Table 4: Ablation contrasts, paired over the six dev novels. Positive favours the first-named condition.

Window. This is the central claim about mechanism, and it holds where the mechanism predicts: +4.9 anaphoric and +4.7 implicit against +0.5 [+0.1, +0.9] on explicit. The explicit effect is separable from zero but roughly ten times smaller. Widening the window helps precisely the quotations whose speaker was named several turns back, and barely moves those whose speaker is named beside them. To separate this from the backbone we had to run RoBERTa at its own native window as a third condition, since the naive comparison confounds the two.

Backbone. ModernBERT beats RoBERTa by 2.1 points with the window held equal, so the total encoder gain of roughly 5.4 points splits into about 3.3 from reading more and 2.1 from the better-pretrained encoder. Neither number alone would have supported the claim. The backbone gain is even more sharply concentrated than the window gain: +4.0 anaphoric and +2.2 implicit against +0.0 [-0.2, +0.1] on explicit. Whatever ModernBERT's larger pretraining corpus bought, it bought nothing on the quotations a regular expression already solves.

Hand features. The interval touches zero and the effect is negative on implicit quotations, so we do not claim the feature set helps overall. The eleven features are near-redundant given an encoder that can see the whole window, because the encoder learns distance and recency from the text itself. They buy a real +1.8 on anaphoric, which is where gender agreement has something to say that the text does not make easy.

Capacity. ModernBERT-large does not fit above batch 1 at 1,152 tokens on 12.8 GB, so a first run trained at batch 1 for 6 epochs against base's batch 8 for 8 epochs. Batch size and epoch count are confounded with capacity there, and that run's -0.1 supports no conclusion about model size; it is kept in the table to show why. Gradient accumulation removes the confound: re-running large at batch 1 with accumulation 8 reproduces base's effective batch at the same 8 epochs and learning rates, leaving the encoder (395M parameters against 149M) as the only difference. This reverses the null. Large reaches 80.83% dev against base's 78.98%, a paired +1.9 [+0.2, +3.1], with avoidable error of 5.2% against base's 7.1%. Two things keep the claim modest. It is dev-selected and dev-reported, and it is expensive: 1.9 points is a little over half what widening the window bought, from a model 2.6 times the size at 37 minutes per epoch against base's 15. It is also a floor rather than an estimate, because base had flattened by epoch 6 while large took its best at epoch 8 and was still improving when the budget ran out.

Model size, downward. Qwen3-1.7B zero-shot collapses to 35.2% on dev, below every baseline except random and 13.8 points below nearest-mention. Constrained scoring does not rescue a model that cannot follow the task, so the 8B result is not a generic claim that LLMs are good at this.

### 4.3 Error analysis

Errors split into unreachable (the gold speaker was never enumerated, which is the ceiling) and avoidable (the gold speaker was in the candidate set and the system picked someone else). Only the second is really a modelling failure.

| System (test) | Correct | Unreach. | Avoid. | Addressee |
|---|---|---|---|---|
| nearest-mention | 49.5% | 9.9% | 40.6% | 59.0% |
| Qwen3-8B zero-shot | 77.2% | 9.9% | 13.0% | 70.0% |
| ModernBERT ranker | 82.3% | 9.9% | 7.8% | 58.9% |
| Qwen3-8B LoRA | 87.7% | 9.9% | 2.5% | 40.0% |

Table 5: Error decomposition. The last column is the share of avoidable errors that named the addressee rather than the speaker.

Zero-shot Qwen3-8B is worse than the trivial baseline on explicit quotations: 84.3% against nearest-mention's 89.6%, a paired -5.3 [-14.4, -0.8], separable from zero. An 8B instruction-tuned model, given a passage that literally contains a named speech tag, picks someone else more often than a rule that takes the nearest preceding name. It over-reasons about who ought to be speaking given the scene. Fine-tuning flips the sign to +4.4 [+0.6, +6.4]. A lot of what LoRA teaches here is not new knowledge but restraint. This is also why the pooled number understates the zero-shot model's weakness: it is genuinely good at the hard types, within two points of the ranker on anaphoric, while being bad at the easy one.

Addressee confusion is the dominant residual error mode, and fine-tuning attacks it directly. The share of avoidable errors that named the person being spoken to rather than the speaker falls from 70.0% zero-shot to 58.9% for the ranker to 40.0% for LoRA. In dialogue both parties are named nearby and the local cues look almost the same either way, so telling them apart means tracking conversational structure rather than proximity. LoRA's total avoidable error is 2.5% against a 9.9% unreachable floor, so it solves about 97% of what candidate generation makes solvable, and of the 12.3 points of error between LoRA and a perfect score, four-fifths is enumeration failure rather than ranking failure. Further gains should come from candidate enumeration, not from a better ranker.

Per-novel variation is larger than the gaps between systems, and it tracks the ceiling (Figure 3). Test ranker accuracy runs from 69.0% (The Invisible Man) to 92.1% (Anne of Green Gables). But The Invisible Man, the hardest book for every system and the only one where LoRA does not beat the ranker, is not hard to rank. Its candidate ceiling is 71.5%, the lowest in the split, and both systems reach about 96.5% of that reachable headroom. It is hard because Wells names people sparsely in narration: it averages 2.36 candidates per quotation against 4.76 across the test split, so enumeration frequently never proposes the speaker at all. Raw per-novel accuracy is therefore mostly a measure of how enumerable a novel's characters are.

### 4.4 Extrinsic evaluation

Attribution errors are not equally costly: a misattributed two-word interjection and a misattributed long speech count as one error each under accuracy, but in a multi-voice audiobook the second is far worse. We weight each error by the duration of the quotation at 150 words per minute and report seconds of mis-voiced audio per minute of audiobook. This is worked out from a speaking rate, not by synthesising audio and not by a listening study; it is a length-weighted error rate expressed in seconds and we claim nothing more for it.

| System | s / audio min | h mis-voiced | Length bias |
|---|---|---|---|
| nearest-mention | 11.52s | 11.73h | 0.97 |
| ModernBERT ranker | 4.57s | 4.65h | 1.10 |
| Qwen3-8B zero-shot | 4.55s | 4.63h | 0.85 |
| Qwen3-8B LoRA | 2.71s | 2.76h | 0.93 |

Table 6: Extrinsic audiobook metric on test (61.1 h of audio, 39% dialogue).

Every system improves separably on the baseline, and LoRA turns 11.7 hours of mis-voiced audio into 2.8, separably better than the ranker at a paired -1.86s [-2.79, -1.11]. But the ranker's five-point accuracy advantage over zero-shot does not survive the re-weighting: the two are indistinguishable in seconds (paired +0.03s [-1.78, +1.44]), and the gap is inseparable on dev as well.

The length-bias column is why, and it is the one thing this metric can say that accuracy cannot. It is the duration-weighted error rate divided by the quotation-weighted one: above 1.0 a system errs on longer-than-average quotations. The ranker sits at 1.10 and zero-shot at 0.85, so the ranker is wrong less often, 17.7% of quotations against 22.8%, but wrong on longer speeches, and the two effects cancel almost exactly. Only LoRA improves on both metrics at once. One caveat needs stating carefully: better systems put a larger share of their remaining mis-voiced audio into implicit quotations (65.5% for LoRA against 62.5% for nearest-mention), but this is not a regression, since the within-type error rate there falls from 68.8% to 16.9%.

## 5. Conclusion

Framing quotation attribution as constrained ranking works, and both models beat the nearest-mention baseline by wide, novel-bootstrapped margins on author-disjoint test novels: +32.8 for the ModernBERT ranker and +38.1 for LoRA-tuned Qwen3-8B, against a candidate ceiling of 90.1%.

The three claims resolved as follows. The enumeration ceiling is real, measurable, and now the binding constraint: LoRA leaves only 2.5% avoidable error against a 9.9% unreachable floor, so the next gain on this task comes from candidate generation rather than ranking. The window matters separably from the backbone, at +3.3 for reading the full candidate window and +2.1 for the better encoder, with the window's gain concentrated on anaphoric and implicit quotations exactly as the mechanism predicts. And the decoder LLM beats the ranker only after fine-tuning. Zero-shot it is not merely worse overall but worse than a trivial rule on explicit quotations, which we did not expect and which turned out to be the most informative result in the project: much of what LoRA teaches an 8B model here is when not to reason.

Four limitations shape these results. The candidate ceiling is the binding constraint and first-person narration is why, since a narrator speaks constantly and is never named in the surrounding narration; adding an explicit narrator candidate is the obvious next step, and we left it out here because it changes the candidate set every number is built on. The fine-tuning budget is unequal, since Model 1 trains on 23,818 quotations for 8 epochs against Model 2's 8,000 for 1 epoch, so the size of the LoRA-vs-ranker gap is a lower bound rather than an estimate. Every model is trained once, so the intervals capture variation across novels, which dominates, but not across initialisations. And the gender-agreement feature uses PDNC's annotated character gender, which is corpus metadata rather than a signal derivable from raw text, though the feature ablation shows the dependency is not load-bearing.

## References

1. Vishnubhotla, K., Hammond, A., and Hirst, G. (2022). The Project Dialogism Novel Corpus: A Dataset for Quotation Attribution in Literary Texts. LREC 2022, 5838-5848.
2. Vishnubhotla, K., Rudzicz, F., Hirst, G., and Hammond, A. (2023). Improving Automatic Quotation Attribution in Literary Novels. ACL 2023 (Short Papers), 737-746.
3. Muzny, G., Fang, M., Chang, A., and Jurafsky, D. (2017). A Two-stage Sieve Approach for Quote Attribution. EACL 2017, 460-470.
4. Elson, D. K., and McKeown, K. R. (2010). Automatic Attribution of Quoted Speech in Literary Narrative. Proceedings of AAAI 2010.
5. Michel, G., Epure, E. V., Hennequin, R., and Cerisara, C. (2024). Improving Quotation Attribution with Fictional Character Embeddings. Findings of EMNLP 2024, 12723-12735.
6. Michel, G., Epure, E. V., Hennequin, R., and Cerisara, C. (2025). Evaluating LLMs for Quotation Attribution in Literary Texts: A Case Study of LLaMa3. NAACL 2025 (Short Papers), 742-755.
7. Warner, B., Chaffin, A., Clavie, B., Weller, O., Hallstrom, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., and Poli, I. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. arXiv:2412.13663.
8. Qwen Team (2025). Qwen3 Technical Report. arXiv:2505.09388.

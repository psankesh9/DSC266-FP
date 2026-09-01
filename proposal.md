# Who Said That? Speaker Attribution for Untagged Dialogue in Novels

**DSC266 Final Project Proposal — Pratheek Sankeshi, solo**

**What I plan to do.** Assign every quotation in a novel to its speaker, reporting accuracy separately by how the speaker is signaled: explicit, where the speech tag names the speaker (*"Run," John said*); anaphoric, where it refers to them without naming (*she whispered*), requiring coreference; and implicit, where there is no speech tag at all.

**Why it is important and challenging.** Attribution underpins character-level analysis of fiction and multi-voice audiobook narration, where one misattributed line breaks the listener's sense of the scene. Explicit quotes fall to shallow heuristics, but anaphoric and implicit quotes, which are most of fictional dialogue, demand tracking of conversational alternation, addressee, and long-range coreference. Muzny et al. (2017) show that sieve heuristics degrade sharply there: these are cascades of hand-written rules applied highest-precision-first, with each assignment final, so the precise rules match only on surface patterns beside the quote, and untagged quotes offer none. Vishnubhotla et al. (2023) find errors concentrate in implicit quotes and unseen novels. Pooled accuracy hides this, so per-type accuracy on held-out novels is my primary metric.

**Dataset.** The Project Dialogism Novel Corpus (Vishnubhotla et al., 2022; current release 28 novels, 37,131 quotations), annotated with gold speaker, addressee, quote type, and a character gazetteer. I split by author rather than by novel: Jane Austen alone contributes six books, so a novel-level split would let a model ride one author's naming conventions. I balance splits on narrative person, since a first-person narrator speaks without ever being named. The held-out test split is seven novels by seven authors absent from training, scored once.

**Algorithms.** I frame attribution as ranking over candidate speakers drawn from a narration window, and measure the ceiling it imposes. Baseline: nearest preceding mention. Model: ModernBERT (Warner et al., 2024) encodes quote and context; a bilinear head scores each candidate span using positional, gender-agreement, and recent-speaker features. Architecturally it is BERT, but trained on a far larger corpus and with an 8K context, so a whole conversational exchange fits in one forward pass and a candidate thousands of tokens from the quote — the regime where implicit attribution actually fails — survives instead of being truncated at 512. I ablate exactly that by clamping the window back to 512. I compare an instruction-tuned decoder LLM (Qwen3-8B, 32K context) given the same wide context and constrained to the candidate list, zero-shot and LoRA-tuned, since there the prompt does most of the work. Backbones come from HuggingFace; candidate enumeration, the ranking head, and the evaluation harness are mine. Confidence intervals bootstrap over novels, not quotations, since books vary more than systems do. Michel et al. (2024, 2025) report stronger PDNC results using character-style embeddings and LLaMa-3; my author-disjoint split makes those figures non-comparable, so they anchor the ceiling rather than set a target.

---

## Timeline

Four weeks, ending **Wednesday 2 September 2026**. The test split is scored exactly once, in Week 4.

| Week | Dates | Work | Deliverable |
|---|---|---|---|
| 1 | Aug 6–12 | PDNC loader; author-disjoint splits balanced on narrative person; quote-type labels. Candidate enumeration plus a window sweep to fix the recall/candidate-count tradeoff. Nearest-mention and most-frequent-speaker baselines. Novel-level bootstrap harness. | Baseline accuracy by quote type, and the candidate-set ceiling that bounds every later model. |
| 2 | Aug 13–19 | ModernBERT ranker: span pooling, bilinear scoring head, feature set, training loop. Tune on dev only. | Dev accuracy by quote type, with intervals, against the baseline. |
| 3 | Aug 20–26 | Constrained Qwen3-8B generator, zero-shot and LoRA-tuned. Ablations: context clamped to 512 tokens, ModernBERT-base for large, features removed. Error analysis on anaphoric/implicit failures. | Model comparison table; ablation table; a characterization of what still fails. |
| 4 | Aug 27–Sep 2 | Single scoring run on the seven held-out test novels. Figures, extrinsic audiobook error metric, write-up. | Final report. |

**Risks and contingencies.** The decoder LLM is the most likely thing to slip, since an 8B model at long context strains a single GPU; I run it 4-bit quantized with LoRA, and if that still does not fit, I fall back to a 3–4B instruct model and report the size honestly rather than dropping the comparison. If it slips anyway, the ranker ablations in Week 3 stand alone as the analytical contribution. If the candidate-set ceiling measured in Week 1 proves too low to support the ranking formulation, I widen the enumeration window and report the cost in candidates per quote as a result in its own right. The extrinsic audiobook metric is computed analytically from a words-per-minute speaking rate, not by synthesizing audio, so it is a length-weighted error rate expressed in seconds. I will describe it as such and not claim a listening study.

---

## References

1. Vishnubhotla, K., Hammond, A., and Hirst, G. (2022). The Project Dialogism Novel Corpus: A Dataset for Quotation Attribution in Literary Texts. *LREC 2022*, 5838–5848.
2. Vishnubhotla, K., Rudzicz, F., Hirst, G., and Hammond, A. (2023). Improving Automatic Quotation Attribution in Literary Novels. *ACL 2023 (Short Papers)*, 737–746.
3. Muzny, G., Fang, M., Chang, A., and Jurafsky, D. (2017). A Two-stage Sieve Approach for Quote Attribution. *EACL 2017*, 460–470.
4. Elson, D. K., and McKeown, K. R. (2010). Automatic Attribution of Quoted Speech in Literary Narrative. *Proceedings of AAAI 2010*.
5. Michel, G., Epure, E. V., Hennequin, R., and Cerisara, C. (2024). Improving Quotation Attribution with Fictional Character Embeddings. *Findings of EMNLP 2024*, 12723–12735.
6. Michel, G., Epure, E. V., Hennequin, R., and Cerisara, C. (2025). Evaluating LLMs for Quotation Attribution in Literary Texts: A Case Study of LLaMa3. *NAACL 2025 (Short Papers)*, 742–755.
7. Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G., Howard, J., and Poli, I. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. *arXiv:2412.13663*.
8. Qwen Team (2025). Qwen3 Technical Report. *arXiv:2505.09388*.

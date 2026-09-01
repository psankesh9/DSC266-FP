# Who Said That? Quotation Attribution as the Bottleneck for Multi-Voice Audiobook Generation

**DSC266 Final Project Proposal (revised) — [Your Name], solo**

## What I plan to do

I will build and evaluate models that assign every quotation in a novel to its speaker, reporting accuracy separately for quotes whose speaker is named in a speech tag and quotes whose speaker is not. I then measure how attribution errors propagate into a multi-voice audiobook rendered from the same text.

## Why it is important and challenging

Millions of web novels and public-domain works have no audiobook, and single-voice TTS makes dialogue-heavy fiction hard to follow; attribution is the bottleneck for multi-voice narration. Explicitly tagged quotes (*"Run," John said*) fall to shallow heuristics, but anaphoric (*she whispered*) and untagged quotes are most of fictional dialogue and demand tracking of conversational alternation, addressee, and long-range coreference. Muzny et al. (2017) show heuristics collapse there; Vishnubhotla et al. (2023) find errors concentrate in implicit quotes and in unseen novels.

My own baselines confirm this sharply. A nearest-preceding-mention rule reaches **87.4%** on explicit quotes but **28.6%** on anaphoric and **26.6%** on implicit ones (dev split, 4,842 quotes). Pooled accuracy is 48.9%, a number that flatters a system solving only the easy third of the problem. That gap is the project.

## Data

**Primary: the Project Dialogism Novel Corpus** (Vishnubhotla et al., 2022) — 28 novels, 37,131 quotations annotated with speaker, addressee, quote type, and a per-novel character gazetteer. I split **by author**, not merely by novel: Austen contributes six books and Forster four, so a novel-level split would let a model ride an author's naming conventions rather than learn attribution. Splits are also balanced on narrative person, since a first-person narrator speaks without ever being named. This yields train 15 novels / 23,818 quotes, dev 6 / 4,842, test 7 / 8,443, with disjoint author sets.

The held-out test split — seven novels by seven authors absent from training, spanning literary, crime, children's, and science fiction — *is* the generalization test. **LitBank has been dropped** (its entity and coreference supervision would require a separate pipeline that the ranking formulation below does not need). **RiQuA is a stretch goal**, not a commitment: an additional out-of-domain test set if time allows after the models are trained.

## Algorithms

Attribution is framed as **ranking over candidate speakers**, so the candidate set is a hard ceiling and I measure it first. Candidates are characters mentioned in a narration window around the quote, matched against the novel's gazetteer. At the configured window the gold speaker is present for 86.1% of dev quotes (95.8% explicit, 83.0% anaphoric, 77.8% implicit) at 5.0 candidates per quote.

- **Baseline:** nearest preceding mention. A second rule prefers a mention adjacent to a speech verb — a cheap stand-in for the first sieve of Muzny et al. (2017), *not* a reimplementation of their system.
- **Model 1:** encode quote and context with RoBERTa and score each candidate with a bilinear head, adding positional, gender-agreement, and recent-speaker features.
- **Model 2:** an instruction-tuned generator (Flan-T5) emitting a speaker constrained to the candidate list, evaluated zero-shot and fine-tuned.

Encoders come from HuggingFace; candidate enumeration, the ranking head, the features, and the evaluation harness are mine.

## Objective

Success is speaker accuracy on held-out novels, reported overall **and by quote type**, against the nearest-mention baseline. Confidence intervals bootstrap over **novels, not quotations** — books vary far more than systems do (dev novels range 34% to 78% around a 49% pooled mean), so a quote-level interval would be dishonestly narrow. The target is a large gain on anaphoric and implicit quotes; a gain confined to explicit quotes is a null result and will be reported as one.

An extrinsic metric — mis-voiced seconds per minute of audio — tests whether intrinsic gains survive downstream. **This is computed analytically** from a words-per-minute speaking rate rather than by synthesizing and timing real audio, since no TTS engine is installed on the development machine. It is therefore a length-weighted error rate expressed in seconds, and I will describe it as such rather than claiming a listening study.

## Secondary contribution

PDNC's gold alias lists are incomplete in a way that silently caps any system built on them: Hemingway's "Bill Gorton" carries only his full name as an alias while the text always reads "Bill said." A uniform rule proposing first and last names — kept only when unambiguous within the novel — raises the dev ceiling from 78.3% to 86.1% for 0.8 extra candidates per quote. I report this as a measured corpus limitation with the fix ablated, since it bounds the published numbers of anyone using PDNC's gazetteer directly.

---

## Changes from the original proposal

Distant supervision over Project Gutenberg was abandoned: its silver labels would have come from the same speech-tag patterns the evaluation then scored. Gutenberg novels are retained only for the unlabelled audiobook demo. LitBank is dropped and RiQuA demoted to a stretch goal (above). The baseline matches mentions against PDNC's gold gazetteer rather than spaCy NER — deliberately the *generous* choice, since it makes every ceiling reported here an upper bound on what an end-to-end system with automatic NER could reach.

---

## References

1. Vishnubhotla, K., Hammond, A., Hirst, G., and Sims, M. (2022). The Project Dialogism Novel Corpus: A Dataset for Quotation Attribution in Literary Texts. *LREC 2022*.
2. Vishnubhotla, K., Rudzicz, F., Hirst, G., and Hammond, A. (2023). Improving Automatic Quotation Attribution in Literary Novels. *ACL 2023 (Short Papers)*.
3. Muzny, G., Fang, M., Chang, A., and Jurafsky, D. (2017). A Two-stage Sieve Approach for Quote Attribution. *EACL 2017*.
4. Elson, D. K., and McKeown, K. R. (2010). Automatic Attribution of Quoted Speech in Literary Narrative. *AAAI 2010*.
5. Papay, S., and Padó, S. (2020). RiQuA: A Corpus of Rich Quotation Annotation for English Literary Text. *LREC 2020*.
6. Kim, J., Kong, J., and Son, J. (2021). Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech. *ICML 2021*.

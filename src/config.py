"""Central configuration: paths, corpus definition, and model/training constants.

The primary corpus is PDNC (Vishnubhotla et al., 2022), which ships gold
speaker labels. A second, unlabelled Gutenberg corpus is kept for the
out-of-domain demo and the extrinsic audiobook metric only.
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PDNC_DIR = DATA_DIR / "pdnc"          # gold-annotated primary corpus
PDNC_NOVELS = PDNC_DIR / "novels"
RAW_DIR = DATA_DIR / "raw"            # untouched Gutenberg downloads
CLEAN_DIR = DATA_DIR / "clean"        # boilerplate stripped
BUILD_DIR = DATA_DIR / "build"        # extracted quotes / candidates / splits

OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = OUTPUT_DIR / "models"

for _d in (RAW_DIR, CLEAN_DIR, BUILD_DIR, RESULTS_DIR, PLOTS_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- corpus
# PDNC: 28 novels, 37,131 quotations, each with a gold speaker, addressee, and
# quote type. Splitting is BY NOVEL and in fact BY AUTHOR -- see below.
#
# (folder, author_code, narrative_person, genre, split)
PDNC_CORPUS = [
    ("Emma",                         "AUST", 3, "literary", "train"),
    ("MansfieldPark",                "AUST", 3, "literary", "train"),
    ("NorthangerAbbey",              "AUST", 3, "literary", "train"),
    ("Persuasion",                   "AUST", 3, "literary", "train"),
    ("PrideAndPrejudice",            "AUST", 3, "literary", "train"),
    ("SenseAndSensibility",          "AUST", 3, "literary", "train"),
    ("APassageToIndia",              "FORS", 3, "literary", "train"),
    ("ARoomWithAView",               "FORS", 3, "literary", "train"),
    ("HowardsEnd",                   "FORS", 3, "literary", "train"),
    ("WhereAngelsFearToTread",       "FORS", 1, "literary", "train"),
    ("HardTimes",                    "DICK", 3, "literary", "train"),
    ("OliverTwist",                  "DICK", 3, "literary", "train"),
    ("AHandfulOfDust",               "WAUG", 3, "literary", "train"),
    ("TheSunAlsoRises",              "HEMI", 1, "literary", "train"),
    ("WinnieThePooh",                "MILN", 1, "children", "train"),

    ("TheAgeOfInnocence",            "WHAR", 3, "literary", "dev"),
    ("TheManWhoWasThursday",         "CHES", 3, "literary", "dev"),
    ("TheGambler",                   "DOST", 1, "literary", "dev"),
    ("TheSignOfTheFour",             "DOYL", 1, "crime",    "dev"),
    ("DaisyMiller",                  "JAME", 1, "literary", "dev"),
    ("AlicesAdventuresInWonderland", "CARR", 3, "children", "dev"),

    ("NightAndDay",                  "WOOL", 3, "literary", "test"),
    ("TheMysteriousAffairAtStyles",  "CHRI", 1, "crime",    "test"),
    ("AnneOfGreenGables",            "MONT", 3, "children", "test"),
    ("ThePictureOfDorianGray",       "WILD", 3, "literary", "test"),
    ("TheInvisibleMan",              "WELL", 3, "scifi",    "test"),
    ("TheSportOfTheGods",            "DUNB", 3, "literary", "test"),
    ("TheAwakening",                 "CHOP", 3, "literary", "test"),
]
# Why author-disjoint and not merely novel-disjoint: Austen contributes six
# novels and Forster four. A novel-level split would put Emma in train and
# Persuasion in test, and a model could then ride Austen's naming conventions
# ("Miss ---", free indirect style) rather than learning attribution. Holding
# out whole authors is the setting Vishnubhotla et al. (2023) show is hard.
#
# The split is also balanced on narrative person, because a first-person
# narrator changes the task: the narrator speaks without ever being named, so
# "I said" quotes have no candidate mention in the window at all. Train is 19%
# first-person by quote count and test 22%, so held-out accuracy is not just
# measuring a shift in narrative stance. Dev runs hotter (40%) but dev only
# picks checkpoints.
#
# Resulting sizes, in quotations: train 23,838 (64%), dev 4,846 (13%),
# test 8,447 (23%).

SPLITS = ("train", "dev", "test")

# ---------------------------------------------------------------- OOD corpus
# Unlabelled Gutenberg novels. These carry NO gold speakers, so they are never
# used to fit or to score attribution models. Two uses only:
#   1. the extrinsic audiobook demo, where errors are counted by hand;
#   2. a qualitative out-of-domain check on authors PDNC does not cover.
GUTENBERG_CORPUS = [
    # (gutenberg_id, short_name, author)
    (1400, "GreatExpectations",   "Dickens"),
    (768,  "WutheringHeights",    "Bronte"),
    (1260, "JaneEyre",            "Bronte"),
    (1661, "SherlockHolmes",      "Doyle"),
    (145,  "Middlemarch",         "Eliot"),
    (36,   "WarOfTheWorlds",      "Wells"),
    (76,   "HuckleberryFinn",     "Twain"),
]

# ---------------------------------------------------------------- quotes
# Characters that open/close quoted speech. Gutenberg texts mix straight and
# curly marks, sometimes within a single book, so both are handled.
QUOTE_PAIRS = [("“", "”"), ('"', '"'), ("‘", "’")]

MIN_QUOTE_CHARS = 8      # drop '"Yes."'-style fragments too short to attribute
MAX_QUOTE_CHARS = 1200

# Window of narration searched for a speaker, in characters either side.
CONTEXT_CHARS_BEFORE = 900
CONTEXT_CHARS_AFTER = 400

# ---------------------------------------------------------------- candidates
# Wider window used only to ENUMERATE candidate speakers. It is deliberately
# larger than the context a model reads: in an alternating exchange the last
# time a speaker was named can be many turns back, and a candidate the ranker
# never sees is an error no amount of modelling can undo. The sweep in
# candidates.py reports what each setting costs in candidates per quote.
CANDIDATE_CHARS_BEFORE = 2500
CANDIDATE_CHARS_AFTER = 800
CANDIDATE_WINDOW_SWEEP = [(500, 200), (900, 400), (1500, 600), (2500, 800), (4000, 1500)]

# ---------------------------------------------------------------- characters
MIN_CHARACTER_MENTIONS = 5   # below this a name is noise, not a character

HONORIFICS = {
    "mr", "mrs", "miss", "ms", "dr", "doctor", "sir", "lady", "lord",
    "captain", "colonel", "major", "professor", "rev", "reverend",
    "aunt", "uncle", "madame", "mademoiselle", "monsieur", "master",
}

# Speech verbs, used by the heuristic quote-type classifier in quotes.py and by
# the baseline's speech-tag rule. Kept deliberately tight: on the baseline,
# precision matters far more than recall.
SPEECH_VERBS = {
    "said", "says", "say", "replied", "asked", "answered", "cried",
    "exclaimed", "shouted", "whispered", "murmured", "muttered",
    "continued", "added", "returned", "responded", "remarked",
    "observed", "declared", "inquired", "interrupted", "began",
    "repeated", "protested", "sighed", "laughed", "called", "gasped",
    "stammered", "retorted", "urged", "insisted", "concluded",
}

PRONOUN_GENDER = {
    "he": "M", "him": "M", "his": "M", "himself": "M",
    "she": "F", "her": "F", "hers": "F", "herself": "F",
}
THIRD_PERSON_PRONOUNS = set(PRONOUN_GENDER)

# ---------------------------------------------------------------- labels
# How a quote's speaker is signalled in the text. This is the primary axis
# of analysis: the whole point of the project is that accuracy on EXPLICIT
# quotes is nearly free while ANAPHORIC/IMPLICIT quotes are the real task.
# PDNC capitalises these; the loader lowercases so the two corpora agree.
QUOTE_TYPES = ("explicit", "anaphoric", "implicit")

# PDNC leaves quoteType blank on 28 of 37,131 rows. They are dropped rather
# than guessed: 0.08% of the data is not worth a heuristic that would then
# contaminate the per-type breakdown this project reports.
DROP_UNTYPED = True

# ---------------------------------------------------------------- models
# ModernBERT (Warner et al., 2024) rather than RoBERTa. Architecturally it is
# still a bidirectional encoder, so the ranking head below is unchanged; what
# differs is a far larger and more recent pretraining corpus and an 8,192-token
# context. The context is the reason for the swap: at 512 tokens the encoder
# could only read a third of the window candidates are drawn from, so a
# candidate named early in an exchange was enumerated but never actually seen.
ENCODER_MODEL = "answerdotai/ModernBERT-base"
# Capacity ablation upward: same recipe, 395M params instead of 149M, to
# separate "the ranking formulation works" from "the bigger encoder did it".
ENCODER_MODEL_LARGE = "answerdotai/ModernBERT-large"
# The pre-swap backbone, kept runnable so the report can attribute the gain to
# the backbone and the window separately rather than to their sum.
ENCODER_MODEL_LEGACY = "roberta-base"

# Nothing is truncated at this length: at the -2500/+800 window the longest dev
# passage is 1,037 ModernBERT tokens (p99 = 1,008). Set from the measured
# distribution rather than rounded up to a power of two, since attention cost
# is quadratic and the headroom buys nothing.
MAX_LENGTH = 1152
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 1e-4    # the randomly-initialised head can move faster
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
MAX_CANDIDATES = 12          # candidates scored per quote
USE_AMP = True               # bf16 autocast; Blackwell has native bf16
SPAN_DIM = 256               # width of the pooled span representation
FEATURE_DROPOUT = 0.1

# Text the encoder actually reads, in characters. Now set EQUAL to the
# candidate enumeration window above, which is the whole point of the
# ModernBERT swap: every candidate the enumerator proposes is now inside the
# text the encoder reads, so every candidate gets a real pooled span. Under
# RoBERTa the encoder read -900/+400 while candidates came from -2500/+800, so
# the candidates that mattered most -- named many turns back, in exactly the
# alternating exchanges where implicit attribution fails -- fell outside the
# encoded text and were scored from hand features alone.
#
# The learned "out-of-window" vector in ranker.py is retained but should now be
# near-dead; how often it still fires is reported, since a non-zero rate means
# a mention was lost to tokenisation rather than to the window.
MODEL_CHARS_BEFORE = CANDIDATE_CHARS_BEFORE
MODEL_CHARS_AFTER = CANDIDATE_CHARS_AFTER

# The context ablation: the pre-swap regime, reproduced exactly. Clamping the
# character window WITHOUT clamping MAX_LENGTH would test nothing, since the
# short window already fits in 512 tokens; clamping MAX_LENGTH without the
# character window would test front-truncation instead of context. Both move
# together, so the ablation isolates one variable -- how much text the encoder
# sees -- while backbone, head, features, and candidate set are held fixed.
CLAMP_CHARS_BEFORE = 900
CLAMP_CHARS_AFTER = 400
CLAMP_MAX_LENGTH = 512
# Long speeches waste the token budget: the median quote is 65 characters but
# the 99th percentile is 937. Speaker cues sit at the edges of a quotation, not
# in the middle of it, so overlong quotes are kept head-and-tail.
QUOTE_HEAD_CHARS = 220
QUOTE_TAIL_CHARS = 110

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ---------------------------------------------------------------- evaluation
N_BOOTSTRAP = 1000
CONFIDENCE = 0.95
# Bootstrap resamples NOVELS, not quotations. Quotes within a book are far from
# independent -- one badly modelled conversation supplies dozens of correlated
# errors -- so a quote-level interval would be dishonestly narrow.
BOOTSTRAP_UNIT = "novel"

# Two quotations belong to the same conversation if less than this much
# narration separates them. Used by the alternation baseline, which is only
# meaningful within a continuous exchange -- across a scene break the "speaker
# two turns ago" is a different person in a different room.
CONVERSATION_GAP_CHARS = 1200

# Extrinsic audio metric. No TTS engine is installed on this machine, so the
# mis-voiced-seconds metric is computed analytically from a words-per-minute
# speaking rate rather than by synthesising and timing real audio.
SPEAKING_RATE_WPM = 150.0

# ------------------------------------------------- model 2: decoder generator
# A recent instruction-tuned decoder rather than Flan-T5. The prompt does the
# work here, so pretraining recency and context length matter more than the
# encoder-decoder architecture: Qwen3-8B reads the same -2500/+800 window
# ModernBERT does, well inside its 32K native context, and gets it as text.
GENERATOR_MODEL = "Qwen/Qwen3-8B"
# Size ablation, and the documented fallback if 8B will not fit: the comparison
# is reported with the parameter count stated rather than dropped.
GENERATOR_MODEL_SMALL = "Qwen/Qwen3-1.7B"

# 8B weights are 16GB in bf16 and this machine has 12.8GB of VRAM, so the model
# is loaded 4-bit NF4 with double quantisation (~5.5GB) and adapted with LoRA.
# Quantisation is a memory constraint, not a design choice, and is reported.
GENERATOR_LOAD_4BIT = True
GENERATOR_COMPUTE_DTYPE = "bfloat16"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Attention and MLP projections both. Attention-only LoRA is the cheaper
# convention, but the task is closer to retrieval-within-context than to style
# transfer, and the MLP is where the model would store "a speech tag two turns
# back binds this quote".
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj")

GENERATOR_MAX_LENGTH = 2048   # prompt cap; the window is ~1,050 Qwen tokens
# One prompt is prefilled at a time and its KV cache is reused across that
# quote's candidates, so the batch axis is grad accumulation, not throughput.
GENERATOR_BATCH_SIZE = 1
GENERATOR_GRAD_ACCUM = 8
GENERATOR_LEARNING_RATE = 1e-4   # LoRA adapters want a far larger step
GENERATOR_EPOCHS = 1
# Qwen3 emits a <think> block by default. It is disabled: the model never
# generates here, it only scores candidate names as continuations, and a
# reasoning trace would sit between the prompt and the name being scored.
GENERATOR_ENABLE_THINKING = False
# Candidate names are scored as decoder continuations and then compared. Names
# differ in length ("Emma" vs "Mrs Fitzwilliam Darcy"), and an unnormalised
# sequence log-probability systematically prefers the short one, which matters
# far more zero-shot than after fine-tuning. Both are computed; this selects
# which one the reported accuracy uses.
GENERATOR_LENGTH_NORMALISE = True

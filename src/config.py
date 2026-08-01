import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / 'data'
ENGLISH_DIR = DATA_DIR / 'english'
HINDI_DIR = DATA_DIR / 'hindi'
PROCESSED_DIR = DATA_DIR / 'processed'

EN_CLEAN_DIR = ENGLISH_DIR / 'clean'
EN_ORIGINAL_DIR = ENGLISH_DIR / 'original'
HI_CLEAN_DIR = HINDI_DIR / 'clean'
HI_ORIGINAL_DIR = HINDI_DIR / 'original'
HI_PREPROCESSED_DIR = HINDI_DIR / 'preprocessed'
EN_PREPROCESSED_DIR = ENGLISH_DIR / 'preprocessed'

DOC_IDS = list(range(1, 31))

CORRUPTED_DOC_IDS = [6, 14, 22, 25, 26]

# Frozen document-level split (seed 42 in output_format.py). Do not change lightly.
TRAIN_DOC_IDS = [
    2,
    3,
    5,
    6,
    7,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    25,
    26,
    27,
    28,
    29,
    30,
]
DEV_DOC_IDS = [8, 9, 24]
TEST_DOC_IDS = [1, 4, 21]

# Resolve pdftotext from PATH so configs work off-Homebrew machines; None if absent.
PDFTOTEXT_CMD = shutil.which('pdftotext')

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F

# Track C production SentencePiece (joint full Unigram 41k; see DESIGN_DECISIONS §18)
SPM_V2_PRIMARY = (
    PROJECT_ROOT
    / 'data'
    / 'models'
    / 'tokenizers'
    / 'sentencepiece_legal_v2_joint_full_41000.model'
)

# Assignment aligned pairs (after LaBSE)
ALIGNED_DIR = DATA_DIR / 'aligned'
ALIGNED_ALL = ALIGNED_DIR / 'all.jsonl'

# Dual eval policies (see docs/EXPERIMENTS.md, split_external_eval.py)
EXTERNAL_PARALLEL_DIR = DATA_DIR / 'external' / 'parallel'
STAGE_A_ALL = EXTERNAL_PARALLEL_DIR / 'stage_a_en_hi.jsonl'
STAGE_A_TRAIN = EXTERNAL_PARALLEL_DIR / 'stage_a_train.jsonl'
EXTERNAL_EVAL_DIR = EXTERNAL_PARALLEL_DIR / 'eval'

# Artifacts
ANALYSIS_DIR = DATA_DIR / 'analysis'
MODELS_DIR = DATA_DIR / 'models'
TOKENIZER_DIR = MODELS_DIR / 'tokenizers'
RUNS_DIR = DATA_DIR / 'runs'

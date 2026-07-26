from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ENGLISH_DIR = DATA_DIR / "english"
HINDI_DIR = DATA_DIR / "hindi"
PROCESSED_DIR = DATA_DIR / "processed"

EN_CLEAN_DIR = ENGLISH_DIR / "clean"
EN_ORIGINAL_DIR = ENGLISH_DIR / "original"
HI_CLEAN_DIR = HINDI_DIR / "clean"
HI_ORIGINAL_DIR = HINDI_DIR / "original"
HI_PREPROCESSED_DIR = HINDI_DIR / "preprocessed"
EN_PREPROCESSED_DIR = ENGLISH_DIR / "preprocessed"

DOC_IDS = list(range(1, 31))

CORRUPTED_DOC_IDS = [6, 14, 22, 25, 26]

PDFTOTEXT_CMD = "/opt/homebrew/bin/pdftotext"

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F

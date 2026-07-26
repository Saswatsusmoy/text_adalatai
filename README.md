# Adalat AI -- Legal EN->HI Translation Assignment

Prototype text-translation system for **Indian court judgments** (English -> Hindi), with focus on:

1. **Token efficiency** for Indic scripts under LLMs / NMT tokenizers
2. **Light domain adaptation** for high-fidelity legal translation

## Status

| Phase | State |
|-------|-------|
| Preprocessing (OCR, join, segment, align, split) | Done -- 1,458 EN-HI pairs |
| Tokenizer analysis + domain SentencePiece | Done -- SP 16k/32k/41k trained |
| MT training / evaluation modules | Not started |

Corpus: 30 parallel Supreme Court judgments. Working artifacts live under `data/`; living project history is `CHANGELOG.md` and `DESIGN_DECISIONS.md`.

## Repository layout

```
data/                 # Judgments, intermediate pipeline output, models
src/preprocessing/    # PDF OCR, line join, segmentation, LaBSE align, splits
src/tokenizer/        # Corpus prep, SentencePiece train, benchmark, deep dive
src/utils/            # Shared validation helpers
configs/              # Pipeline config (paths + alignment thresholds)
tests/                # pytest suite (run with PYTHONPATH=.)
Makefile              # install, preprocess, tokenizer-*, test targets
run_pipeline.py       # Step orchestrator
scripts/reproduce_all.sh
```

Private plans and research notes live in **`.local/`** (never commit; listed in `.git/info/exclude`).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
# or: pip install -r requirements.txt && python3 -m spacy download en_core_web_sm

make test
make preprocess          # full cleaning -> train/dev/test JSONL
make tokenizer-bench     # needs trained models under data/models/tokenizers/
```

Common targets: `make reextract`, `join`, `segment`, `align`, `output`, `tokenizer-train-all`, `tokenizer-bench`, `test`.

## Pipeline outputs

```
data/hindi/preprocessed/     # Tesseract OCR text
data/english/preprocessed/   # Line-joined English
data/{english,hindi}/segmented/
data/aligned/all.jsonl       # 1,458 filtered pairs
data/processed/{train,dev,test}.jsonl   # doc-level 80/10/10 (gitignored)
data/models/tokenizers/      # SentencePiece models (gitignored)
data/analysis/               # Tokenizer benchmark JSON
```

Alignment filters (code + `configs/preprocessing.yaml`): min LaBSE similarity 0.5, EN:HI char ratio 0.3-3.0, EN Jaccard near-dedup 0.85.

## License / data

Assignment corpus provided for evaluation purposes. External public corpora (if used) are documented with sources and licenses in CHANGELOG / DESIGN_DECISIONS.

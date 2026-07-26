# Adalat AI -- Legal EN->HI Translation Assignment

Prototype text-translation system for **Indian court judgments** (English -> Hindi), with focus on:

1. **Token efficiency** for Indic scripts under LLMs / NMT tokenizers
2. **Light domain adaptation** for high-fidelity legal translation

## Status

| Phase | State |
|-------|-------|
| Preprocessing (OCR, join, segment, align, split) | Done -- 1,458 EN-HI pairs |
| Tokenizer analysis + domain SPM v1/v2 | Done -- **Track C freeze: joint_full 41k** |
| External Stage A legal bitext (MILPaC + Anuvaad) | Done -- ~993k filtered pairs |
| Dual-track MT (defaults + custom vocab) | Plan adopted; training not started |

Corpus: 30 parallel Supreme Court judgments. Working history: `CHANGELOG.md`, `DESIGN_DECISIONS.md`. **Full experiment tables and freezes:** [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Repository layout

```
data/                 # Judgments, intermediate pipeline output, models
docs/EXPERIMENTS.md   # Research findings, benches, freezes
src/preprocessing/    # PDF OCR, join, segment, LaBSE, splits, external Stage A ingest
src/tokenizer/        # Corpus prep, SentencePiece train, benchmark, deep dive
src/utils/            # Shared validation helpers
configs/              # Pipeline config (assignment + external Stage A)
tests/                # pytest suite (run with PYTHONPATH=.)
Makefile              # install, preprocess, external-*, tokenizer-*, test
run_pipeline.py       # Step orchestrator (preprocess | external | all)
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
make preprocess          # assignment: cleaning -> train/dev/test JSONL
make external-ingest     # Stage A: MILPaC + Anuvaad -> data/external/parallel/
# make external-download # fetch raw external files if missing
make tokenizer-bench     # needs trained models under data/models/tokenizers/
# python run_pipeline.py --steps external
# python run_pipeline.py --steps all
```

Common targets: `make reextract`, `join`, `segment`, `align`, `output`, `external-download`, `external-ingest`, `tokenizer-train-all`, `tokenizer-c0`, `tokenizer-spm-v2-full-joint`, `tokenizer-bench`, `test`.

**Track C production SPM:** `data/models/tokenizers/sentencepiece_legal_v2_joint_full_41000.model` (`src.config.SPM_V2_PRIMARY`).

## Pipeline outputs

```
data/hindi/preprocessed/     # Tesseract OCR text
data/english/preprocessed/   # Line-joined English
data/{english,hindi}/segmented/
data/aligned/all.jsonl       # 1,458 filtered pairs
data/processed/{train,dev,test}.jsonl   # doc-level 80/10/10 (gitignored)
data/models/tokenizers/      # SentencePiece models (gitignored)
data/analysis/               # Tokenizer benchmark JSON
data/external/raw/           # MILPaC xlsx + Anuvaad zips (gitignored)
data/external/parallel/      # Filtered Stage A JSONL (gitignored)
```

**Assignment** filters (LaBSE path): min similarity 0.5, EN:HI char ratio 0.3-3.0, EN Jaccard near-dedup 0.85.

**External Stage A** filters: same char ratio + min length + exact pair dedup; no LaBSE re-score (see DESIGN_DECISIONS §17). Combined file: `data/external/parallel/stage_a_en_hi.jsonl` (~993k pairs).

## Documentation map

| Doc | Role |
|-----|------|
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | All research findings, benches, freezes, reproduce commands |
| [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) | Why major choices were made |
| [`CHANGELOG.md`](CHANGELOG.md) | What changed over time |
| [`AGENTS.md`](AGENTS.md) | Agent/coding rules for this repo |
| `.local/` | Private plans (not committed) |

## License / data

Assignment corpus provided for evaluation purposes. External Stage A: MILPaC (CC BY-NC-SA 4.0), Anuvaad legal EN-HI (CC BY 4.0). Details in CHANGELOG / DESIGN_DECISIONS / EXPERIMENTS.

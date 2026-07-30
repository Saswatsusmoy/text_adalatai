# Adalat AI -- Legal EN->HI Translation Assignment

Prototype text-translation system for **Indian court judgments** (English -> Hindi), with focus on:

1. **Token efficiency** for Indic scripts under LLMs / NMT tokenizers
2. **Light domain adaptation** for high-fidelity legal translation

## Start here (submission)

| Deliverable | Path |
|-------------|------|
| **Assignment report** | [`REPORT.md`](REPORT.md) |
| Interactive process story | [`story/index.html`](story/index.html) |
| Full experiment tables | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| Design rationale | [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) |

**Production system:** NLLB-200 distilled 600M + LoRA Stage **A2**  
**Adapters:** `data/runs/nllb600_A_A2_h200_A2_ddp2_20260726T212958Z/checkpoints/best_primary`  
**Scores (BLEU / chrF++):** I_test 21.86 / 49.66; MILPaC 34.90 / 56.46; Anuvaad 45.80 / 64.83  
(vs zero-shot 18.85 / 44.74; 34.28 / 55.22; 39.39 / 60.08)

## Status

| Phase | State |
|-------|-------|
| Preprocessing (OCR, join, segment, align, split) | Done -- 1,458 EN-HI pairs |
| Tokenizer analysis + domain SPM v1/v2 | Done -- **Track C freeze: joint_full 41k** |
| External Stage A legal bitext (MILPaC + Anuvaad) | Done -- ~993k filtered pairs |
| Dual-track MT (D = NLLB LoRA; C = vocab extend) | Done -- **production = Track D A2** |
| Dual-policy eval (I + E) + hyp dumps | Done -- `data/analysis/` |
| Assignment report | Done -- `REPORT.md` |

Corpus: 30 parallel Supreme Court judgments. Working history: `CHANGELOG.md`, `DESIGN_DECISIONS.md`.

## Repository layout

```
REPORT.md             # Grader-facing write-up (start here)
story/                # Interactive process log (open index.html)
docs/EXPERIMENTS.md   # Research findings, benches, freezes
src/preprocessing/    # PDF OCR, join, segment, LaBSE, splits, Stage A ingest
src/tokenizer/        # Corpus prep, SentencePiece train, benchmark, deep dive
src/training/         # NLLB LoRA, subsample, vocab extend, CUDA/DDP
src/evaluation/       # BLEU/chrF++, suite loaders, zero-shot + adapter decode
src/utils/            # Validation + hardware profile
configs/              # Preprocess + training YAMLs
tests/                # pytest (PYTHONPATH=.)
Makefile              # install, preprocess, tokenizer-*, train-*, test
run_pipeline.py       # Step orchestrator
scripts/reproduce_all.sh
data/analysis/        # Metrics JSON + hyp JSONL (lightweight)
data/runs/            # Local PEFT runs (gitignored; large)
data/models/          # SPM + extended bases (gitignored)
data/external/        # Stage A bitext (gitignored)
```

Private plans live in **`.local/`** (never commit).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
# or: pip install -r requirements.txt && python3 -m spacy download en_core_web_sm

make test
make preprocess          # assignment: cleaning -> train/dev/test JSONL
make external-ingest     # Stage A pool: MILPaC + Anuvaad -> parallel/
make external-eval-split # Policy E held-out + stage_a_train (required before MT)
# make external-download # fetch raw external files if missing
make tokenizer-bench     # needs trained models under data/models/tokenizers/
make zero-shot-nllb      # baseline decode + metrics (downloads NLLB from HF)
# make train-nllb-smoke  # short LoRA smoke on local GPU/MPS
```

Common targets: `make reextract`, `join`, `segment`, `align`, `output`, `external-download`,
`external-ingest`, `tokenizer-train-all`, `tokenizer-c0`, `tokenizer-spm-v2-full-joint`,
`tokenizer-bench`, `train-nllb-A1`, `train-nllb-A1-h200`, `test`.

**Track C production SPM:** `data/models/tokenizers/sentencepiece_legal_v2_joint_full_41000.model`
(`src.config.SPM_V2_PRIMARY`).

### Score production adapters (if present locally)

```bash
PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb \
  --adapters data/runs/nllb600_A_A2_h200_A2_ddp2_20260726T212958Z/checkpoints/best_primary \
  --tag A2_best
```

Base weights: HuggingFace `facebook/nllb-200-distilled-600M`. Adapters are small PEFT folders;
full Stage A bitext and multi-GB run trees are not in git (rebuild via Makefile / attach your run).

## Pipeline outputs

```
data/hindi/preprocessed/     # Tesseract OCR text
data/english/preprocessed/   # Line-joined English
data/{english,hindi}/segmented/
data/aligned/all.jsonl       # 1,458 filtered pairs
data/processed/{train,dev,test}.jsonl   # doc-level 80/10/10 (gitignored)
data/models/tokenizers/      # SentencePiece models (gitignored)
data/analysis/               # Tokenizer + MT metrics / hyps
data/external/raw/           # MILPaC xlsx + Anuvaad zips (gitignored)
data/external/parallel/      # Filtered Stage A JSONL (gitignored)
data/runs/                   # Training runs + PEFT checkpoints (gitignored)
```

**Assignment** filters (LaBSE path): min similarity 0.5, EN:HI char ratio 0.3-3.0, EN Jaccard near-dedup 0.85.

**External Stage A** filters: same char ratio + min length + exact pair dedup; no LaBSE re-score
(see DESIGN_DECISIONS §17). Pool: `stage_a_en_hi.jsonl` (~993k).
**MT train file:** `stage_a_train.jsonl` (after `make external-eval-split`).

**Eval policies** (score every system on both):

| Policy | Test sets |
|--------|-----------|
| **I** internal | `data/processed/test.jsonl` (190 pairs, docs 1/4/21) |
| **E** external | `eval/milpac_test.jsonl` (117) + `eval/anuvaad_test.jsonl` (3k) |

## Documentation map

| Doc | Role |
|-----|------|
| [`REPORT.md`](REPORT.md) | **Submission write-up** (tokenizer, train, metrics, qualitative, reflection) |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | All research findings, benches, freezes, reproduce commands |
| [`docs/HARDWARE_MLX.md`](docs/HARDWARE_MLX.md) | Local M4 16GB profile + MLX vs MPS strategy |
| [`docs/TRAINING_STRATEGY.md`](docs/TRAINING_STRATEGY.md) | Stage A/B LoRA plan, monitoring, success bars |
| [`docs/NLLB_ARCHITECTURE.md`](docs/NLLB_ARCHITECTURE.md) | NLLB module map + targeted LoRA profiles |
| `configs/training.yaml` | Declarative train defaults (snapshotted per run) |
| [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) | Why major choices were made |
| [`CHANGELOG.md`](CHANGELOG.md) | What changed over time |
| [`story/index.html`](story/index.html) | Interactive process story |

## License / data

Assignment corpus provided for evaluation purposes. External Stage A: MILPaC (CC BY-NC-SA 4.0),
Anuvaad legal EN-HI (CC BY 4.0). Details in CHANGELOG / DESIGN_DECISIONS / EXPERIMENTS.

# Adalat AI -- Legal EN->HI Translation Assignment

Prototype text-translation system for **Indian court judgments** (English -> Hindi):

1. **Token efficiency** for Indic scripts under LLMs / NMT tokenizers
2. **Light domain adaptation** for high-fidelity legal translation

## Start here

| Audience | Open first |
|----------|------------|
| Graders | [`REPORT.md`](REPORT.md) |
| Interview walkthrough | [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) |
| Full experiment tables | [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) |
| Interactive story | [`story/index.html`](story/index.html) |
| Design rationale | [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) |

**Production:** NLLB-200 distilled 600M + LoRA Stage **A2**  
**Adapters:** `data/runs/nllb600_A_A2_h200_A2_ddp2_20260726T212958Z/checkpoints/best_primary`  
**Scores (BLEU / chrF++):** I 21.86 / 49.66 · MILPaC 34.90 / 56.46 · Anuvaad 45.80 / 64.83  
(vs zero-shot 18.85 / 44.74 · 34.28 / 55.22 · 39.39 / 60.08)

## Four phases (clean separation)

| Phase | Package | Responsibility | Make / pipeline |
|-------|---------|----------------|-----------------|
| **1 Data** | `src/preprocessing/` | OCR, join, segment, LaBSE, splits; Stage A ingest + E holdout | `make preprocess` · `make external-eval-split` |
| **2 Tokenizer** | `src/tokenizer/` | Survey, domain SPM v1/v2, benches | `make tokenizer-bench` · `make tokenizer-c0` |
| **3 Train** | `src/training/` | Track D NLLB LoRA; Track C Marian / vocab-extend | `make train-nllb-smoke` · H200 targets |
| **4 Eval** | `src/evaluation/` | Dual policy I+E, BLEU/chrF++, decode | `make zero-shot-nllb` |

Shared: `src/config.py` (paths + frozen IDs), `src/utils/` (jsonl, validation, hardware).  
Configs: `configs/preprocessing.yaml`, `configs/training*.yaml` (one YAML per curriculum).  
Orchestrator: `python run_pipeline.py --list`

```text
data/ PDFs
  -> [1 preprocess]  aligned + train/dev/test + Stage A
  -> [2 tokenizer]   domain SPM freeze (optional Track C)
  -> [3 train]       LoRA adapters under data/runs/
  -> [4 eval]        data/analysis/*_report.json + hyps
```

## Status

| Phase | State |
|-------|-------|
| Preprocessing | Done -- 1,458 EN-HI pairs, doc-level 80/10/10 |
| Tokenizer analysis + domain SPM | Done -- **joint_full 41k freeze** |
| External Stage A (MILPaC + Anuvaad) | Done -- ~993k pairs + dual-policy split |
| Track D NLLB LoRA | Done -- **production = A2** |
| Track C vocab path | Done -- C1c negative vs A2 (documented) |
| Dual-policy eval + report | Done -- `REPORT.md` + `data/analysis/` |

## Repository layout

```
REPORT.md                 # assignment write-up
docs/WALKTHROUGH.md       # interview tour (phase by phase)
docs/EXPERIMENTS.md       # freezes, tables, reproduce commands
src/preprocessing/        # phase 1
src/tokenizer/            # phase 2
src/training/             # phase 3
src/evaluation/           # phase 4
src/utils/                # shared jsonl + validation + hardware
configs/                  # preprocessing + training YAMLs
tests/                    # mirrors src packages
Makefile                  # phase targets
run_pipeline.py           # step orchestrator
scripts/reproduce_all.sh  # end-to-end data + tokenizer + tests
data/analysis/            # metrics + hyps (lightweight)
data/runs/                # PEFT runs (gitignored)
data/models/              # SPM + extended bases (gitignored)
data/external/            # Stage A bitext (gitignored)
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
# make install-dev   # ruff + pre-commit

make test
make lint

# Phase 1
make preprocess
make external-ingest
make external-eval-split

# Phase 2
make tokenizer-bench     # needs SPM under data/models/tokenizers/

# Phase 3 / 4 smokes
make zero-shot-nllb      # or: make train-nllb-smoke
# python run_pipeline.py --steps all
```

**Track C SPM freeze:** `src.config.SPM_V2_PRIMARY`  
(`data/models/tokenizers/sentencepiece_legal_v2_joint_full_41000.model`)

### Score production adapters (if present)

```bash
PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb \
  --adapters data/runs/nllb600_A_A2_h200_A2_ddp2_20260726T212958Z/checkpoints/best_primary \
  --tag A2_best
```

Base weights: HuggingFace `facebook/nllb-200-distilled-600M`.

## Eval policies

| Policy | Test sets |
|--------|-----------|
| **I** internal | `data/processed/test.jsonl` (docs 1/4/21, 190 pairs) |
| **E** external | MILPaC test (117) + Anuvaad test (3k); never in Stage A train |

## Lint / format

```bash
make install-dev
make lint
make format
make check          # format-check + test
pre-commit install  # optional
```

Config: `pyproject.toml` (ruff line-length 100, single quotes).

## Documentation map

| Doc | Role |
|------|------|
| [`REPORT.md`](REPORT.md) | Submission write-up |
| [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) | Interview phase tour |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Freezes, benches, reproduce |
| [`docs/TRAINING_STRATEGY.md`](docs/TRAINING_STRATEGY.md) | Stage A/B LoRA plan |
| [`docs/NLLB_ARCHITECTURE.md`](docs/NLLB_ARCHITECTURE.md) | LoRA module map |
| [`docs/HARDWARE_MLX.md`](docs/HARDWARE_MLX.md) | Local M4 vs H200 |
| [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md) | Why choices were made |
| [`CHANGELOG.md`](CHANGELOG.md) | What changed |

## License / data

Assignment corpus for evaluation. External Stage A: MILPaC (CC BY-NC-SA 4.0),
Anuvaad legal EN-HI (CC BY 4.0). Details in CHANGELOG / DESIGN_DECISIONS / EXPERIMENTS.

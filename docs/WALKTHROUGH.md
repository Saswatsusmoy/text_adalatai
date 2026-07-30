# Interview walkthrough -- Adalat AI

Use this doc to walk the codebase phase by phase. Full numbers live in
`REPORT.md` and `docs/EXPERIMENTS.md`. Interactive story: `story/index.html`.

**Rule:** every experiment that ran still has code + config + (where small) metrics
under `data/analysis/`. Nothing was deleted for this modularization.

---

## 60-second map

| Phase | Package | What it does | Reproduce |
|-------|---------|--------------|-----------|
| 1 Data | `src/preprocessing/` | PDFs -> aligned pairs -> splits; Stage A external | `make preprocess` / `make external-eval-split` |
| 2 Tokenizer | `src/tokenizer/` | Survey + domain SPM v1/v2 | `make tokenizer-bench` / `make tokenizer-c0` |
| 3 Train | `src/training/` | NLLB LoRA (D), Marian/C1c (C) | `make train-nllb-smoke` / H200 targets |
| 4 Eval | `src/evaluation/` | Dual policy I+E, BLEU/chrF++, hyps | `make zero-shot-nllb` |
| Config | `configs/` | Preprocess + all train YAMLs | snapshotted per run |
| Report | `REPORT.md` | Grader write-up | open file |

Orchestrator: `python run_pipeline.py --list` then `--steps preprocess|external|tokenizer|all`.

---

## Phase 1 -- Preprocessing (`src/preprocessing/`)

### Assignment corpus (30 judgments)

```text
PDF (EN/HI)
  -> reextract_pdfs.py     # Tesseract OCR for Hindi (not PDF text layer)
  -> join_lines.py         # EN hard wraps + proper nouns
  -> segment_sentences.py  # spaCy EN, danda HI
  -> align_sentences.py    # LaBSE mutual-best + filters
  -> output_format.py      # doc-level 80/10/10, seed 42
```

| Output | Path |
|--------|------|
| Aligned pairs | `data/aligned/all.jsonl` (~1458) |
| Frozen splits | `data/processed/{train,dev,test}.jsonl` |
| Doc IDs | `src/config.py` `TRAIN_DOC_IDS` / `DEV_DOC_IDS` / `TEST_DOC_IDS` |

**Interview point:** document-level split prevents judgment leakage into test.

### External Stage A (optional scale)

```text
ingest_external_parallel.py  # MILPaC + Anuvaad -> stage_a_en_hi.jsonl
split_external_eval.py       # Policy E holdout + stage_a_train.jsonl
eval_sets.py (validation)    # no train/eval pair leak
```

| Output | Role |
|--------|------|
| `stage_a_train.jsonl` | MT Stage A train only (~988k) |
| `eval/milpac_*`, `eval/anuvaad_*` | Policy E never trained on |

**Interview point:** dual policies catch Stage B overfit on assignment alone.

---

## Phase 2 -- Tokenizer (`src/tokenizer/`)

| Module | Role |
|--------|------|
| `benchmark.py` | Cross-model packing table (chars/token, HI/EN, totals) |
| `deep_dive.py` | Byte-fallback / Devanagari merge diagnostics |
| `prepare_corpus.py` + `train.py` | Domain SP v1 (Prarabdha mono HI 16/32/41k) |
| `prepare_spm_corpus.py` | Stage A + assignment train only (firewall) |
| `train_v2.py` / `train_full_joint.py` | Joint Unigram; full train on 16GB via dedupe |

**Production freeze (Track C0):** `src.config.SPM_V2_PRIMARY`  
`data/models/tokenizers/sentencepiece_legal_v2_joint_full_41000.model`

**Interview points:**
1. Byte-level BPE is weak on Hindi; Unigram SP / multilingual Dev pieces are strong.
2. HI-only SP packs HI better but fragments EN -- joint required for MT.
3. 41k frozen over 64k for emb size vs packing.

---

## Phase 3 -- Training (`src/training/`)

### Track D (shipped production)

```text
configs/training.yaml (+ training_h200*.yaml)
  -> subsample.py           # smoke / A1 / A2 / Bp mixes
  -> train_nllb_lora.py     # PEFT LoRA on NLLB-600M
  -> nllb_data.py           # encode + collate
  -> cuda_backend.py        # Hopper bf16 / fused AdamW / SDPA
  -> dist_utils.py          # torchrun DDP
```

| Curriculum | Data | Config |
|------------|------|--------|
| smoke | 2k Stage A | `training.yaml` |
| A1 | 50-80k quality mix | `training.yaml` / `training_h200.yaml` |
| A2 | 150k, resume A1 | `training_h200_A2.yaml` |
| B | assignment train only | `training_h200_B.yaml` (ablation) |
| Bp | assignment + A2 replay | `training_h200_Bp.yaml` |
| A2 DoRA | same as A2, DoRA | `training_h200_A2_dora.yaml` |

**Production checkpoint:**  
`data/runs/nllb600_A_A2_h200_A2_ddp2_*/checkpoints/best_primary`  
Base: `facebook/nllb-200-distilled-600M`

**Interview point:** Stage B raises I BLEU but fails E anti-forget; ship A2.

### Track C (vocab experiments; not production)

| Code | Role |
|------|------|
| `spm_tokenizer.py` | HF-style wrapper on SPM_V2_PRIMARY |
| `legal_mt_model.py` / `legal_mt_data.py` / `train_legal_mt.py` | C1 Marian from-scratch path |
| `vocab_extend_nllb.py` | C1c v1 bulk extend (failed ablation) |
| `vocab_extend_nllb_v2.py` | C1c v2 careful extend (below ZS on A1 budget) |
| `configs/training_c1*.yaml`, `training_c1c*.yaml` | C1 / C1c train configs |

---

## Phase 4 -- Evaluation (`src/evaluation/`)

| Module | Role |
|--------|------|
| `eval_sets.py` | Policy I + E suite paths + leak validation |
| `metrics_mt.py` | sacrebleu BLEU + chrF++ |
| `zero_shot_nllb.py` | Decode base or PEFT adapters; write hyps + report; `--mbr` flag routes through `mbr_decode.py` |
| `mbr_decode.py` | Sample N + argmax mean pairwise sentence-chrF++ (MBR decode ablation; DESIGN §31) |
| `eval_legal_mt.py` | Track C1 Marian checkpoints |

Score production adapters:

```bash
PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb \
  --adapters data/runs/.../checkpoints/best_primary \
  --tag A2_best
```

MBR decode ablation (`make eval-mbr-a2`; -2.5 chrF++ vs beam4 at N=8 top_p=0.9 T=1.0, EXPERIMENTS §5.4):

```bash
PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb \
  --adapters data/runs/.../checkpoints/best_primary \
  --mbr --mbr-samples 8 --mbr-utility chrfpp
```

Metrics JSON + hyps: `data/analysis/*_report.json`, `*_hyps.jsonl`.

---

## Shared utilities

| Path | Role |
|------|------|
| `src/config.py` | Paths, frozen doc IDs, SPM freeze |
| `src/utils/jsonl.py` | Shared load/write/append JSONL |
| `src/utils/validation.py` | Devanagari heuristics |
| `src/utils/profile_hardware.py` | M4 / CUDA profile |
| `src/training/common.py` | seed, device, autocast, loss.item |

---

## Suggested 15-minute interview order

1. **Problem** (1 min) -- legal EN->HI; token efficiency + domain fidelity. Open `REPORT.md` goals.
2. **Data** (3 min) -- open `src/preprocessing/__init__.py` docstring, show pipeline, frozen test docs 1/4/21.
3. **Tokenizer** (3 min) -- `docs/EXPERIMENTS.md` §4 table; domain SP vs byte-BPE.
4. **Train** (4 min) -- Track D LoRA A1->A2; dual-policy selection; why not B / not C1c.
5. **Eval** (2 min) -- score table in `REPORT.md`; one qualitative example (revisional / writ petition).
6. **Code map** (2 min) -- four packages + `make` targets; `run_pipeline.py --list`.

---

## Makefile quick index

```text
# Phase 1
make preprocess
make external-ingest external-eval-split

# Phase 2
make tokenizer-bench
make tokenizer-c0
make tokenizer-spm-v2-full-joint

# Phase 3 (local smoke / full remote)
make train-nllb-smoke
make train-nllb-A1-h200 train-nllb-Bp-h200   # H200

# Phase 4
make zero-shot-nllb
make eval-c1

# Quality
make lint test
```

Full experiment freezes and reproduce commands: `docs/EXPERIMENTS.md`.

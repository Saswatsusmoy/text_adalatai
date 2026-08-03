# `data/analysis/` — artifact inventory

Every metric report, hypothesis dump, and bench summary in this directory
maps to a specific experiment or freeze. This file lists them all and
marks each as **current** (source of truth for that claim), **superseded**
(kept for reproducibility of an earlier analysis; a newer file now leads),
or **reference** (helper input, not a result).

Nothing here is stale in the sense of "wrong" — everything reflects the
data at the time it was written. `superseded` just means a later, more
comprehensive artifact now leads the corresponding table in the docs.

## MT system reports and hypothesis dumps

Each system has one `*_report.json` (BLEU + chrF++ per suite) and up to
three `*_{I_test,E_milpac_test,E_anuvaad_test}_hyps.jsonl` files (raw
hypotheses vs references, one line per pair).

| System | Report | Hyps | Status |
|--------|--------|------|--------|
| Zero-shot NLLB (H200 canonical) | `zero_shot_nllb_report_h200.json` | `zero_shot_nllb_{I,E_milpac,E_anuvaad}_test_hyps.jsonl` | current |
| Zero-shot NLLB (earlier local) | `zero_shot_nllb_report.json` | (same hyps) | reference (earlier local baseline; H200 report leads) |
| Track D A1 LoRA | `nllb600_A1_h200_best_report.json` | `nllb600_A1_h200_best_*` | current |
| **Track D A2 LoRA (production)** | `nllb600_A2_h200_best_report.json` | `nllb600_A2_h200_best_*` | **current (shipped)** |
| Track D A2 DoRA (method ablation) | `nllb600_A2_dora_h200_best_report.json` | `nllb600_A2_dora_h200_best_*` | current |
| Track D Stage B (failed anti-forget) | `nllb600_B_h200_best_report.json` | `nllb600_B_h200_best_*` | current |
| Track D Stage B' replay | `nllb600_Bp_h200_best_report.json` + `nllb600_Bp_h200_run_summary.json` | `nllb600_Bp_h200_best_*` | current |
| Track C1c v1 bulk extend (catastrophic) | `nllb600_c1c_v1_h200_best_report.json` | `nllb600_c1c_v1_h200_best_*` | current |
| Track C1c v2 careful extend | `nllb600_c1c_v2_h200_best_report.json` | `nllb600_c1c_v2_h200_best_*` | current |
| A2 MPS beam4 control (device/precision confound test) | `nllb600_A2_mps_beam4_best_report.json` | `nllb600_A2_mps_beam4_best_{I_test,E_milpac_test}_*` | current (E_anuvaad skipped on MPS -- ~4h budget) |
| A2 MPS MBR N=8 ablation | `nllb600_A2_mps_mbr8_best_report.json` | `nllb600_A2_mps_mbr8_best_{I_test,E_milpac_test}_*` | current (E_anuvaad skipped on MPS) |

## Combined / cross-system summaries

| File | Content | Status |
|------|---------|--------|
| `final_dual_policy_report.json` | Combined dual-policy decision dump (all Track D + C1c systems on I + E) | current |
| `comet22_summary.json` | COMET-22 (Unbabel/wmt22-comet-da) system-level scores for all shipped hyps | current (Phase 4: written under `schema: "v2"`, cache keyed on hyp-file SHA256 prefix + model; pre-v2 cache ignored on next run) |
| `compare_zero_shot_vs_A1_h200.json` | Pairwise ZS vs A1 diff dump | reference (A1 is a stepping stone; A2 is production, see `nllb600_A1_h200_best_report.json`) |

## Tokenizer analyses

The 35-config full matrix (`tokenizer_matrix.json`) is the current source
of truth for tokenizer packing on this corpus. Earlier bench JSONs remain
in the tree for reproducibility of the analyses that referenced them, but
downstream docs (REPORT §1.2, EXPERIMENTS §4, story tokenizers page) all
lead with the matrix.

| File | Content | Status |
|------|---------|--------|
| `tokenizer_matrix.json` | **35-config Phase 1 + Phase 2 matrix** on held-out 322 pairs | **current** |
| `tokenizer_matrix_manifest.json` | Training manifest per matrix config (elapsed_s, model size) | current |
| `tokenizer_metrics.json` | Early cross-family + v1 Prarabdha bench on assignment bitext (1,458 pairs) | superseded by `tokenizer_matrix.json` for domain SPMs; still the source of the Gemma / GPT-4o comparison numbers in REPORT §1.1 |
| `tokenizer_metrics_v2.json` | Pre-matrix v2 joint / hi bench (subset of what the matrix now covers) | superseded by `tokenizer_matrix.json` |
| `tokenizer_benchmark_c0.json` | Early Track C0 freeze bench | superseded by `tokenizer_matrix.json` |
| `tokenizer_vocab_size_ablation.json` | Early 41k / 48k / 64k Unigram-only ablation | superseded by `tokenizer_matrix.json` (which covers the same vocab ladder for both Unigram AND BPE, plus secondary axes) |
| `tokenizer_report.json` | Older top-level bench dump | superseded by `tokenizer_matrix.json` |

## Environment / logs

| File | Content | Status |
|------|---------|--------|
| `hardware_profile.json` | Local M4 16GB CPU + MPS profile | reference (used by `docs/HARDWARE_MLX.md`) |
| `train_nllb_A1.log` | Training log from an early A1 attempt | reference (kept for provenance; canonical A1 run is `nllb600_A_A1_h200_ddp2_20260726T204933Z`) |

## What lives outside this directory

- LoRA / DoRA adapters (~44 MB each): `data/runs/<run_id>/checkpoints/best_primary/` — gitignored
- Tokenizer .model / .vocab pairs: `data/models/tokenizers/` — 6 v1 files tracked in git; v2 + matrix models (~30 files, ~45 MB) present locally, not tracked
- Aligned assignment corpus: `data/aligned/all.jsonl` (tracked); split JSONLs in `data/processed/`
- Stage A external bitext: `data/external/parallel/` — gitignored (~200 MB)
- SPM training corpora: `data/external/spm_corpus_legal_v2_*.txt` — gitignored (~1.3 GB total)

## How to regenerate anything

- `make preprocess` — assignment corpus (OCR, align, split)
- `make external-ingest && make external-eval-split` — Stage A + Policy E carve
- `make tokenizer-c0` — v2 Unigram sample-mode
- `make tokenizer-spm-v2-full-joint` — v2 Unigram full-joint 41k (shipped freeze)
- `make tokenizer-matrix-{phase1,phase2,bench}` — full 35-config matrix
- `make train-nllb-A1-h200 && make train-nllb-A2-*` — production adapter curriculum
- `make comet-score` — regenerates `comet22_summary.json` from any `*_hyps.jsonl`

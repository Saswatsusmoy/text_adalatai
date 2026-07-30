# Changelog

## [Unreleased]

### Added

- **Multi-page assignment docs walkthrough (`story/`):**
  - Docs-style light site with separate pages (assignment, pipeline, tokenizers, Stage A,
    dual eval, Track D/C, scoreboard, qualitative, production, failures, reflection,
    interview, artifacts, glossary, reproduce).
  - Shared shell via `js/site.js` (sidebar TOC + prev/next). Open `story/index.html`.

### Fixed

- **requirements.txt:** remove invalid `python>=3.10` pip line; loosen pins so clean
  installs work (spaCy on 3.12; model via `spacy download` not a frozen wheel URL).

### Added

- **Modular phase layout for interview / graders (no experiment loss):**
  - Package maps in `src/{preprocessing,tokenizer,training,evaluation,utils}/__init__.py`.
  - Shared `src/utils/jsonl.py`; eval/subsample/split/ingest/output use it.
  - `docs/WALKTHROUGH.md` -- 15-min tour + phase/module tables.
  - `run_pipeline.py` -- external_eval_split, eval_sets, zero_shot_smoke,
    train_nllb_smoke; groups external/eval_smoke/train_smoke; `--list`.
  - `scripts/reproduce_all.sh` -- dual-policy split + optional zero-shot smoke + lint.
  - README phase table; AGENTS phase status; DESIGN §16/§30.

- **Ruff lint/format tooling:**
  - `pyproject.toml` -- ruff lint + format (line-length 100, single quotes, py310,
    first-party `src`), pytest config.
  - `requirements-dev.txt`, `.pre-commit-config.yaml` (ruff + ruff-format).
  - Makefile: `install-dev`, `lint`, `format`, `format-check`, `check`.
  - Applied format/autofix across `src/`, `tests/`, `run_pipeline.py`; fixed remaining
    E741/F841/E722/B007/B011 issues so `make lint` is clean.

### Changed

- **Training/eval LOC + efficiency pass (no feature loss):**
  - Shared helpers in `src/training/common.py` (seed, jsonl, autocast, batch H2D, loss.item).
  - DDP: `broadcast_object` / `all_reduce_max` helpers; denser `train_nllb_lora` / `train_legal_mt`.
  - Collate pads into preallocated tensors; NLLB dataset sets src/tgt lang once.
  - Stripped filler docs/comments; train+eval packages ~3954 -> ~3300 LOC.

### Added

- **Submission package:**
  - `REPORT.md` -- assignment-facing write-up (tokenizer, data, NLLB LoRA, BLEU/chrF++ tables,
    qualitative ZS vs A2 panel, reflection, codebase map).
  - README rewritten as submission entrypoint (report link, production A2 scores, train/eval
    layout, adapter score command).
  - Lightweight MT metrics + I/E reports under `data/analysis/` (scores JSON + hyp JSONL for
    qualitative review). Full multi-GB `data/runs/` and Stage A pools remain gitignored.

- **MBR decode ablation on A2** (DESIGN_DECISIONS §31, `docs/EXPERIMENTS.md` §5.4):
  - `src/evaluation/mbr_decode.py`: `mbr_pick`, `sample_candidates`, `translate_batch_mbr` (nucleus sample N + argmax mean pairwise sentence-chrF++ utility; Eikema & Aziz 2020, Freitag et al. 2022).
  - `src/evaluation/zero_shot_nllb.py`: `--mbr --mbr-samples N --mbr-temperature T --mbr-top-p P --mbr-utility {chrf,chrfpp}`; auto-appends `_mbr{N}` to tag so hyp files don't clobber beam4 runs.
  - Make: `eval-mbr-a2`, `eval-mbr-zs`, `eval-mbr-smoke`. Tests: `tests/evaluation/test_mbr_decode.py`.
  - **Real runs (A2 adapters, MPS fp16):** I_test (n=190) MBR N=8 top_p=0.9 T=1.0 = 18.16/47.13; MPS beam4 control = 21.85/49.68; H200 beam4 shipped = 21.86/49.66. E_milpac_test (n=117) MBR = 31.39/54.07; MPS beam4 = 34.71/56.55; H200 beam4 = 34.90/56.46. Device/precision delta <=0.18 BLEU / 0.09 chrF++ (noise). Decode-only delta MBR-beam4: I -3.68/-2.56, E_milpac -3.33/-2.48. E_anuvaad skipped on MPS (~4h). **Beam4 stays shipped.** Follow-ups not run: lower T / epsilon-sampling, N=32-128, COMET utility.
  - Reports: `data/analysis/nllb600_A2_mps_{beam4,mbr8}_best_report.json` + hyps under same tag prefix.

- **DoRA ablation on A2 data** (DESIGN_DECISIONS §28, `docs/EXPERIMENTS.md` §5.1):
  - `build_lora_config` supports `peft.use_dora` / `peft.method: dora` (PEFT weight-decomposed LoRA).
  - Config `configs/training_h200_A2_dora.yaml`: decoder_attn r=16, A2 150k from base, LR 1e-4, 3000 steps DDP2.
  - Make: `train-nllb-A2-dora-h200`. Tests: `tests/training/test_lora_dora.py`.
  - **Full test scores (H200 bf16 batch=32 beam=4, same protocol as A2 LoRA):** I_test 21.80/49.18; E_milpac_test 35.23/56.43; E_anuvaad_test 45.42/64.43. Delta vs A2 LoRA: I -0.05/-0.47, E_milpac +0.33/-0.03, E_anuvaad -0.38/-0.40 (all <=0.5 chrF++, inside noise). Decode elapsed: DoRA 554.8s vs A2 LoRA 269.8s (DoRA magnitude scaling adds per-forward cost). **A2 LoRA stays shipped**; DoRA is a valid method ablation, not a promotion. Report: `data/analysis/nllb600_A2_dora_h200_best_report.json` + hyps.

- **Stage B' anti-forget replay trained + scored on H200** (DESIGN_DECISIONS §27):
  - `build_stage_b_replay_mix` in `src/training/subsample.py` -- all assignment train + domain replay from A2 subsample (default 90/10 by count, seed 42, exact pair dedup); writes `data/external/parallel/subsamples/stage_b_Bp_*` + manifest.
  - `configs/training_h200_Bp.yaml` -- resume A2 best, LR 2e-5, max 500 steps, stage_b_replay enabled, slightly higher E weight in selection.
  - `train_nllb_lora` -- Stage B honors `data.train_jsonl` or builds Bp mix when `curriculum=Bp` / `stage_b_replay.enabled`; resume adapters from `resume.adapters` or `--resume-adapters`; config-driven stage/curriculum defaults.
  - Make: `stage-b-replay-mix`, `train-nllb-Bp-h200`.
  - Tests: `tests/training/test_subsample.py` (90/10 counts, roles, seed determinism, assignment-dup exclusion).
  - Docs: `docs/TRAINING_STRATEGY.md` §2.2 updated (pure B = failed ablation; B' = recommended specialize path).
  - **Run:** `nllb600_B_Bp_h200_Bp_ddp2_20260727T011740Z` (best step 100, ~278s DDP2). Full test: I 22.22/49.41; MILPaC 33.82/54.84; Anuvaad 43.46/62.51.
  - **vs A2:** E_milpac chrF++ drop 1.62 (≤2.0 pass); I chrF++ −0.25; Anuvaad −2.32. **vs pure B:** large E recovery. **Production remains A2.** Report: `data/analysis/nllb600_Bp_h200_best_report.json`; dual-policy table updated.

- **Technical process log site** (`story/`): dense HTML log (tables, deltas, run IDs) covering OCR bake-off, pipeline, Stage A, dual eval, full SPM v1/v2 benches, dual-track MT (ZS/A1/A2/B/C1c), failures, 26 decisions, artifacts. Open `story/index.html`.

- **Track C1c NLLB vocab-extend experiments + dual-policy close** (DESIGN_DECISIONS §26, `docs/EXPERIMENTS.md` §5.2):
  - **v1 bulk (ablation):** `vocab_extend_nllb.py` +~8k raw SPM pieces; full `embed_tokens` via `modules_to_save`. Model `data/models/nllb600_c1c_sp_ext`. Run `nllb600_A_A1_c1c_h200_ddp2_20260726T225043Z`. Full test: I 6.38/24.86, MILPaC 10.66/28.63, Anuvaad 15.65/34.35.
  - **v2 careful (primary C recipe):** `vocab_extend_nllb_v2.py` -- surface forms only if NLLB fragments; reject protected-substring / probe regression (e.g. `न्यायालय` stays single); +1500 -> vocab 257669; mean-init from base encode; LoRA + grad mask `new_embed_start`. Config `training_c1c_v2_h200.yaml` (DDP2, global batch 32). Scoreable run `…c1c_v2…234856Z` (best step 1000, dev primary 53.6). Full test: I 17.79/43.86, MILPaC 28.20/49.78, Anuvaad 37.64/58.46.
  - **Decision:** production dual-policy remains **Track D A2** (`…A2…212958Z/best_primary`). C1c v2 loses to zero-shot and A2 on all three suites; v1 is a failed ablation. C1c does not replace Track D on this budget.
  - Reports: `data/analysis/nllb600_c1c_v{1,2}_h200_best_report.json`; `final_dual_policy_report.json` updated with C1c scores + decision text.

### Fixed

- **C1c v2 emb rows not in PEFT adapter:** with `modules_to_save=null`, grad-mask-trained emb rows were live-only and lost at save (adapter ~12MB LoRA-only). `_save_peft` now writes `new_embed_rows.pt`; `apply_new_embed_rows` on resume/eval (`zero_shot_nllb` + train). First DDP v2 run discarded; retrain before scoring.

### Added (earlier)

- **Track D NLLB LoRA -- Hopper H200 / dual-GPU production path** (DESIGN_DECISIONS §23-25):
  - `src/training/cuda_backend.py` -- bf16/fp16 dtype resolve; TF32 + float32 matmul precision high; cudnn benchmark; Flash/mem-efficient SDPA (math SDP off); fused AdamW; pin_memory + prefetch loaders; optional `torch.compile` helper (single-GPU only); Linux/macOS RSS + CUDA mem reporting.
  - `src/training/dist_utils.py` -- `torchrun`/NCCL DDP init (`device_id`), rank/world helpers, unwrap DDP for save+generate, barriers, cleanup.
  - `configs/training_h200.yaml` (+ `_A2` / `_B`) -- bf16, gradient_checkpointing off, `global_batch_size: 32` (16/device x 2), pad_to_fixed + pad_to_multiple_of 8, **`torch_compile: false`**, gen `eval_batch_size` 32, frozen `train_jsonl` for A1/A2.
  - `train_nllb_lora.py` -- CUDA autocast, non_blocking H2D, DDP wrap (`broadcast_buffers=False`), DistributedSampler, rank-0 eval/save, stop-flag all_reduce; **forces compile off when world_size > 1** (NCCL/PEFT hang); `data.train_jsonl` freeze; batched gen eval; `backend_info.json`.
  - `nllb_data.collate_nllb` -- pad-to-multiple-of-8 and fixed pad (labels -100 / attn mask 0; loss unchanged).
  - Make: `train-nllb-A1-h200` -> `torchrun --standalone --nproc_per_node=2`.
  - Tests: `tests/training/test_cuda_backend.py`, `test_nllb_data.py`.
  - Remote: `/data/adalat_ai`, HF cache `/data/hf-cache`, venv torch 2.13+cu126; runs `nllb600_A_{A1,A2}_h200_*`, `nllb600_B_full_h200_*`.

- **`docs/EXPERIMENTS.md`:** Consolidated research log (assignment pipeline, Stage A data, cross-model tokenizer survey, SPM v1/v2, joint vs HI-only, full-joint 16GB path, vocab ablation 41/48/64k, Track C freeze joint_full_41000, dual-track plan, artifact index, reproduce commands). README and DESIGN_DECISIONS link to it.

- **NLLB architecture analysis for targeted LoRA:** `docs/NLLB_ARCHITECTURE.md` -- M2M100 12+12, d=1024, cross-attn as MT hinge; LoRA profiles `decoder_attn` (default Stage A, ~0.51% params), `cross_attn`, `decoder_full`, `last4_decoder`, `attn_all`. `train_nllb_lora.build_lora_config` path-filters modules (verified: decoder_attn has no encoder self-attn). `configs/training.yaml` peft.profile=`decoder_attn`.

- **Track D NLLB LoRA training (MPS):** `src/training/subsample.py` (smoke/A1/A2 curriculum), `nllb_data.py`, `train_nllb_lora.py` (PEFT LoRA, AdamW+cosine, train/eval JSONL logs, best_primary checkpoints under `data/runs/`). Smoke run verified: 20 steps on 2k pairs, ~0.76% trainable. Make: `train-nllb-smoke`, `train-nllb-A1`, `stage-a-subsample-*`. Deps: peft, pyyaml.

- **Training strategy (pre-implementation):** `docs/TRAINING_STRATEGY.md` + `configs/training.yaml` -- Stage A curriculum (smoke/A1/A2), LoRA defaults for NLLB-600M on MPS, dual-policy selection metrics, early-stop/anti-forgetting rules, run layout under `data/runs/`, success bars vs zero-shot baselines. DESIGN_DECISIONS §21.

- **Track D zero-shot NLLB-600M (MPS):** `src/evaluation/zero_shot_nllb.py` + `metrics_mt.py`. EN->HI on Policy I_test / E_milpac_test / E_anuvaad_test. Final: I BLEU 18.78 chrF++ 44.62; MILPaC 34.14 / 55.12; Anuvaad 39.44 / 60.08. Hyps + `data/analysis/zero_shot_nllb_report.json`. Make: `zero-shot-nllb`.

- **Dual eval policies (I + E):** `split_external_eval.py` carves held-out MILPaC (10% dev/test) and Anuvaad (1k dev / 3k test) from Stage A pool; writes `stage_a_train.jsonl` (~988k) + `data/external/parallel/eval/*`. `src/evaluation/eval_sets.py` loads suites and validates no train/eval pair leak. Make: `external-eval-split`. Stage A MT must use `stage_a_train`, not full pool.

- **Local hardware + MLX policy:** `src/utils/profile_hardware.py` profiles Apple Silicon (chip, unified memory, MLX GPU smoke, PyTorch MPS). Writes `data/analysis/hardware_profile.json`. Docs: `docs/HARDWARE_MLX.md` -- M4 16GB local-only; MLX for small LLM LoRA; PyTorch MPS for NLLB/InLegalTrans enc-dec. `make profile-hardware`. requirements: `mlx`, `mlx-lm`, `torch`.

- **Joint full vocab ablation 48k + 64k:** Trained `sentencepiece_legal_v2_joint_full_{48000,64000}` on same deduped joint corpus (profile=full). Held-out/test/all benches vs 41k in `data/analysis/tokenizer_vocab_size_ablation.json`. **Track C production freeze: `sentencepiece_legal_v2_joint_full_41000`** (generalization / emb size over max packing; 64k ablation only).

- **Full-joint SPM on 16GB (dedupe path):** `dedupe_text_file` + `train_full_joint.py` tries Unigram profiles `full` -> `full_tight` -> `full_sample_15` in a child process (OOM-safe). Dedupe joint corpus (~2% exact dups) + max 4096 chars enabled full Unigram train on **all remaining lines** (`input_sentence_size=0`, seed 250k). Winner: `sentencepiece_legal_v2_joint_full_41000` (does not overwrite sample `joint_41000`). Makefile: `tokenizer-spm-v2-full-joint`. No byte-level BPE.

- **Track C0 legal SentencePiece v2:** `prepare_spm_corpus.py` builds SPM train text from Stage A + assignment train only (hard-excludes dev/test docs 8,9,24 and 1,4,21). Corpora: joint ~1.99M lines / 291M chars; hi ~994k lines / 139M chars. `train_v2.py` trains `sentencepiece_legal_v2_{hi,joint}_{32k,41k}` without overwriting v1. Joint train samples 1M sentences (RAM). Held-out bench (322 pairs): **joint 41k wins for MT** (HI c/t 4.34, HI/EN 0.724, total 11,004 vs v1 41k 3.95 / 0.739 / 11,965). HI-only packs HI better but fragments EN. `benchmark.py --eval held_out` -> `data/analysis/tokenizer_metrics_v2.json`. Makefile: `tokenizer-c0`. Tests: `test_prepare_spm_corpus.py`.

- **External legal EN-HI ingest (Gate 9 T0)** (`src/preprocessing/ingest_external_parallel.py`): Downloads/processes MILPaC (Law-AI) and Anuvaad legal EN-HI (judiciary, HC/SUVAS, law commission, names dict, augmented, legal terms). Already-aligned pairs are mapped to project JSONL (`en_text`, `hi_text`, `source`, `doc_id`) and filtered with the same char-length ratio (0.3-3.0) and min-length rules as post-alignment QC; exact pair dedup. Outputs under `data/external/parallel/` including `stage_a_en_hi.jsonl` + `ingest_report.json`. Makefile: `make external-download`, `make external-ingest`. Tests: `tests/preprocessing/test_ingest_external_parallel.py`.

### Changed

- **Zero-shot NLLB on H200:** Re-ran full I_test / E_milpac_test / E_anuvaad_test with `--device cuda --batch-size 32` (bf16 + SDPA). ~159s wall for 3307 pairs. BLEU/chrF++: I 18.85/44.74; MILPaC 34.28/55.22; Anuvaad 39.39/60.08 (matches prior MPS within noise). Report + hyps under `data/analysis/`; copy `zero_shot_nllb_report_h200.json`.

- **A1 LoRA test eval + compare:** `zero_shot_nllb` supports `--adapters` / `--tag`. Scored H200 `best_primary` on same three test suites. Compare: `data/analysis/compare_zero_shot_vs_A1_h200.json`. Deltas vs zero-shot: I_test BLEU +2.82 / chrF++ +4.42; MILPaC +0.38 / +0.76; Anuvaad +5.78 / +4.25.

- **Track D plan complete (T4+T5 on H200)** (DESIGN_DECISIONS §25):
  - **A2:** `configs/training_h200_A2.yaml`, subsample `stage_a_A2_n150000.jsonl` (150k), resume A1 `best_primary`, LR 5e-5, 3000 steps DDP. Run `nllb600_A_A2_h200_A2_ddp2_20260726T212958Z`. Test: I 21.86/49.66; MILPaC 34.90/56.46; Anuvaad 45.80/64.83.
  - **Stage B:** `configs/training_h200_B.yaml`, assignment `train.jsonl` (1136), resume A2 best, LR 3e-5, 800 steps. Run `nllb600_B_full_h200_B_ddp2_20260726T214744Z`. Test: I 23.10/48.89; MILPaC 30.92/51.22; Anuvaad 40.44/59.60.
  - **Final table:** `data/analysis/final_dual_policy_report.json` (+ per-system reports/hyps under `data/analysis/`).
  - **Recommend A2 best** for dual-policy use: B raises I_test BLEU but fails Stage B anti-forget (E_milpac chrF++ drop 5.24 > 2.0; Anuvaad also regresses).

- **H200 DDP stability:** Disable `torch.compile` under DDP (Dynamo cannot trace NCCL; first loss_eval barrier aborted rank1). Config `torch_compile: false`; code forces off when `world_size>1`. Compile only on single-GPU path. Eval uses unwrapped module. `init_process_group(..., device_id=...)`. DDP `broadcast_buffers=False`. Workers 4.

- **Track D train loop is multi-backend:** `train_nllb_lora` supports MPS (local default via `configs/training.yaml`) and CUDA/Hopper DDP (`configs/training_h200.yaml` + torchrun). Same Stage A curriculum, dual-policy eval, and LoRA profiles on both paths. DESIGN_DECISIONS §19 updated: local M4 remains default; optional remote H200 when VRAM is free (never kill co-resident vLLM without owner OK).

- **External Stage A wired through docs/orchestrators:** `configs/preprocessing.yaml` documents `external_ingest` + paths/licenses/filters. `run_pipeline.py` steps/groups: `external_download`, `external_ingest`, groups `external` / `external_full`; `all` includes `external_ingest`. `scripts/reproduce_all.sh` runs Stage A ingest (with `--download` unless `--skip-downloads`). README quick-start and license notes updated.

### Fixed

- **Docs/orchestrator drift:** Updated README and AGENTS.md to match finished preprocess + tokenizer phases. Rewrote `configs/preprocessing.yaml` for live steps, skipped steps, and alignment thresholds (min sim 0.5, char ratio 0.3-3.0). Fixed `run_pipeline.py` module paths (`src.tokenizer.benchmark`), removed calls to missing `src.evaluation` / `src.training`, and flattened group expansion. Makefile: `all` now uses `tokenizer-train-all`, dropped broken `train`/`eval` targets, alias `tokenizer-train`. `scripts/reproduce_all.sh` no longer calls missing metrics module. DESIGN_DECISIONS renumbered (1-16), tokenizer file names corrected (`benchmark.py` not `analysis.py` / `reproduce_benchmarks.py`), staging path documented as `preprocessed/`. `.gitignore` adds `data/models/`.

### Added

- **Project scaffolding**: Created `src/`, `tests/`, `configs/` directory structure with Python package init files.
- **Configuration module** (`src/config.py`): Centralized all paths (data dirs, PDF tool path), doc ID lists, and Unicode range constants.
- **Validation utilities** (`src/utils/validation.py`): Devanagari character counting, ratio computation, and Hindi-likelihood heuristics.
- **PDF re-extraction script** (`src/preprocessing/reextract_pdfs.py`): Re-extracts Hindi text from corrupted PDFs using `pdftotext`, validates Devanagari content, compares old vs new, scans all 30 PDFs for quality issues, and can apply fixes to `clean/`.
- **Test suite** (`tests/preprocessing/test_reextract_pdfs.py`): 16 tests covering extraction, validation, comparison, apply, and full PDF scan.
- **Pipeline configuration** (`configs/preprocessing.yaml`): Declarative step listing with I/O paths and validation thresholds.
- **CHANGELOG.md**: This file  --  records all changes to the project.

### Changed

- **Extraction backend switched from `pdftotext` to Tesseract OCR**: After evaluating 5 alternatives (pdftotext, PyMuPDF, pdfminer.six, pdfplumber, Tesseract), the script now uses `tesseract -l hin --psm 6` by default. The `--backend pdftotext` flag is retained as a fallback. See DESIGN_DECISIONS.md for the full evaluation.

### Added

- **Intelligent line joining** (`src/preprocessing/join_lines.py`): Joins hard-wrapped lines in 17 English docs using heuristic + legal proper noun list. Reduces non-blank lines from 2,819 to 1,176 (58% reduction). Zero false positives. Output in `data/english/preprocessed/`.
- **Test suite** (`tests/preprocessing/test_join_lines.py`): 24 tests covering join logic, edge cases, document processing, and regression.
- **Sentence segmentation** (`src/preprocessing/segment_sentences.py`): Two-tool approach -- spaCy `en_core_web_sm` model (dependency parser) for English with pre-tokenization protection for Hindi honorifics (`Smt.`, `Shri.`), and danda (।) split for Hindi. Produces 2,495 English and 7,427 Hindi sentences across 30 docs. Auto-detects language. New dependency: `python3 -m spacy download en_core_web_sm`.
- **Test suite** (`tests/preprocessing/test_segment_sentences.py`): 22 tests covering English abbreviation handling, Hindi danda split, auto-detection, and full run.
- **Sentence alignment + quality filters** (`src/preprocessing/align_sentences.py`): LaBSE-based bilingual alignment using greedy bidirectional matching. Produces 1,458 EN-HI sentence pairs across 30 docs (avg 49/doc). Applies length ratio (0.3-3.0), similarity (>0.5), and near-dedup filters. New dependency: `sentence-transformers/LaBSE` (~1.8GB model). Output in `data/aligned/all.jsonl`. BGE-M3 (2024 SoTA) was also evaluated end-to-end but produces fewer pairs (1,347 vs 1,458). LaBSE retained for more training data.
- **Test suite** (`tests/preprocessing/test_align_sentences.py`): 18 tests covering loading, filtering, dedup, output format, and quality checks.

- **Final output format** (`src/preprocessing/output_format.py`): Splits 1,458 aligned pairs into train (1,136), dev (132), and test (190) at document level. Generates `metadata.json` and `alignment_report.json`. Output in `data/processed/`.
- **Proper noun discovery script** (`src/preprocessing/discover_proper_nouns.py`): Data-driven discovery of legal proper nouns for line joining. Scans all 30 English clean files for words appearing at continuation line starts. Derives 34 verified proper nouns with zero guessing.
- **Tokenizer analysis framework** (`src/tokenizer/benchmark.py`, `deep_dive.py`): Full corpus benchmark of 17 tokenizers across 14 model families (2024-2026): Custom SP 41K, Gemma 4, GPT-4o, Phi-4-mini, NLLB-200, Mistral Small 4, Qwen3/3.5/3.6, MiniMax M3, DeepSeek V3/V4 Pro, GLM 5.2, Phi-4, OLMo 3. Measures chars/token, HI/EN ratio, byte fallback detection. Finds that SentencePiece and multilingual BPE handle Hindi well, while byte-level BPE (all Llama-family models) cost 1.1-2.7x more tokens for Hindi regardless of vocabulary size. See DESIGN_DECISIONS.md for full comparison.
- **Custom SentencePiece tokenizer** (`data/models/tokenizers/`): Trained 3 SentencePiece models (16K/32K/41K vocab) on 14M characters of Indian legal Hindi text from Prarabdha/indian-legal-supervised-fine-tuning-data (`src/tokenizer/prepare_corpus.py` outputs to `data/external/legal_hindi_corpus.txt`, gitignored). The 41K model achieves 16,840 Devanagari tokens, 3.84 HI chars/tok, and 0.743 HI/EN ratio -- beating Gemma 4 on Hindi efficiency despite 6x smaller vocabulary. Training fully reproducible via `make tokenizer-train-all`.
- **Tokenizer benchmarks** (`src/tokenizer/benchmark.py`): Benchmarks accessible tokenizers and saves results to `data/analysis/tokenizer_metrics.json` (`--full` for the full model set).
- **Reproducible pipeline**: `Makefile` with targets (`make preprocess`, `make tokenizer-train-all`, `make tokenizer-bench`, `make test`). `run_pipeline.py` Python orchestrator. `scripts/reproduce_all.sh` bash reproduction script. `requirements.txt` with pinned dependencies.

### Fixed

- **Re-extracted corrupted Hindi PDFs (Docs 6, 14, 22, 25, 26)**: The 5 corrupted Hindi clean files contained 0 Devanagari characters  --  every glyph replaced with `?`. Re-extracted via Tesseract OCR. Recovered **42,536 Devanagari characters** across all 5 documents. Also re-extracted all 25 non-corrupted PDFs for consistency, producing full document text (including headers/parties) vs the original clean files which only had numbered paragraphs. Output in `data/hindi/preprocessed/` (renamed from `re_extracted/`). Applied to `clean/` on 2025-07-25.

### Discovered

- **Doc 17 PDF text layer**: `pdftotext` produces transliterated output (`Ekkuuh;` instead of `माननीय`). Tesseract OCR handles it correctly  --  not an issue.
- **PDF vs clean file mismatch**: The original clean files contain only numbered body paragraphs, while PDFs contain full judgments (headers, citations, parties, body). The 25 non-corrupted clean files are a subset of the full document. Our Tesseract extraction captures the complete document.

### Project structure

- **Phase-based organization**: `src/preprocessing/` (Phase 1), `src/tokenizer/` (Phase 2) -- only completed phases exist. No future-phase scaffolding.
- **Everything scripted, nothing interactive**: All analyses moved from ad-hoc commands to reproducible scripts with `if __name__ == "__main__"` entry points.
- **108 tests**: Covering preprocessing, tokenizer, and pipeline orchestration. Each phase has its own test directory.

### Skipped (not needed for this corpus)

- **Strip UTF-8 BOM (original plan Step 2)**: 25/30 Hindi clean files have BOM, but the working directory `data/hindi/preprocessed/` has 0 BOM files. Pipeline operates on preprocessed/ so this step is unnecessary. See DESIGN_DECISIONS.md for evidence.
- **Fix OCR Roman numerals (original plan Step 5)**: Zero instances of `li.`/`lili.` OCR artifacts found in any English clean file or raw PDF extraction. All `L.` instances are legitimate legal abbreviations. See DESIGN_DECISIONS.md for evidence.
- **Normalize line endings (original plan Step 3)**: `data/hindi/preprocessed/` is already 100% LF (30/30 files). The 25 CRLF files are legacy `clean/` files that don't reach the pipeline. See DESIGN_DECISIONS.md for evidence.
- **Paragraph segmentation (original plan Step 6)**: Already satisfied by Steps 1 and 4. Both English joined (782 paras) and Hindi OCR (951 paras) already have clear paragraph structure via blank lines. No additional processing needed. See DESIGN_DECISIONS.md for evidence.

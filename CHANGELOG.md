# Changelog

## [Unreleased]

### Added

- **Evaluation harness Phase 4 fixes** (DESIGN §41): harness code + tests only,
  no re-decode / re-score of real data, no edits to historical score tables.
  - **COMET cache invalidation** (`comet_score.py`): cache key is now
    (tag, suite, hyp-file SHA256 prefix, model_id); `comet22_summary.json` is
    rewritten under `schema: "v2"` and pre-v2 caches are ignored, so a
    regenerated hyp file under an existing tag is re-scored instead of
    reporting the stale score.
  - **Bootstrap CIs** (`metrics_mt.py`): BLEU/chrF++ now report mean + 95% CI
    via sacreBLEU's built-in bootstrap (`n_bootstrap`, `SACREBLEU_SEED`-seeded);
    a per-suite `confidence` block is emitted. `paired_ci` /
    `compare_score_pairs` give a paired-bootstrap difference CI (delta, CI,
    `significant`) so "DoRA vs A2" / "B' vs A2" verdicts carry intervals;
    wired as `--compare-tags A,B` in score-only mode.
  - **Report fingerprints** (`fingerprint.py`, `zero_shot_nllb.py`): reports
    record model id, adapter path (or `base`), beam, max_in/max_new, tokenizer
    vocab size, decode device/dtype, seed, and per-suite hyp-file SHA256 prefix.
    Resume refuses to append rows when the recorded fingerprint differs
    (`--force-resume` overrides); `--score-only` honors `--max-pairs`.
  - **Seeded MBR** (`mbr_decode.py`): `set_seed` on the MBR sampling path
    (default 12345, `--seed`), recorded in every report.
  - **Length ratio + TER** (`metrics_mt.py`): `len_ratio` (sys_len/ref_len) and
    sacreBLEU TER per suite so verbosity is visible (I_test hyps ~1.377x ref
    length with BLEU `bp: 1.0`; chrF++ has no length penalty).
  - **Ref-cleaned column** (`ref_cleaner.py`): `clean_ref` strips stray danda
    before punctuation, `#.`/`॥.` and leading bare-digit markers from
    references only; `ref_cleaned` BLEU/chrF++ is additive and labeled.
  - **Entity panel** (`entity_panel.py`): legal-entity recall/precision/F1 per
    suite reusing the tokenizer bench probe lists, case citations and dates,
    matched across scripts.

### Fixed

- **Story reflects the Phase 3 training-harness fixes** (DESIGN §40):
  - `track-d.html`: new "Selection harness" section -- before/after table (raw
    weighted mean vs z-scored + caps, NaN deadlock, resume safety, MPS scaler,
    batch parity, pool hashes, run registry), exact formula, verified cap
    rejection callout; ops notes extended for MPS/scaler + selection logging.
  - `failures.html`: new entries (caps never enforced, NaN DDP deadlock, MPS
    fp16 no scaling, silent embed-row resume) + learned lesson on selection.
  - `dual-eval.html`: anti-forget bar now described as trainer-enforced.
  - `interview.html`: Q&A on how checkpoint selection works now (z-score + caps).
  - `artifacts.html`: `selection.py` module + `runs.json` registry.
  - `index.html`: Track D card tagline mentions z-scored anti-forget selection.

- **Training harness Phase 3 fixes** (DESIGN §40):
  - **Anti-forget caps + z-scored Stage B selection now enforced** in code
    (`src/training/selection.py`). Previously the docs promised caps
    (`stage_b_max_drop_I_dev: 2.0`, `stage_b_max_drop_E_milpac_dev: 3.0`) and
    z-scoring but the loop used a raw weighted mean of chrF++ and picked argmax
    -- the exact gap that let the pure-B run drop E_milpac chrF++ 5.24 with no
    guardrail.
  - Exact formula (also in DESIGN §40):
    `z_i = (s_i - mean_i)/std_i` when `std_i > 0`, else `z_i = s_i - b_i`
    (delta); `primary = sum_i(w_i * z_i)/sum_i(w_i)`. mean/std are population
    stats over all `gen_eval` rows of the baseline run's `eval_log.jsonl`;
    `b_i` is the chrF++ of the baseline run's best-primary row. No baseline
    available -> raw weighted mean + WARN.
  - Baseline resolution: `eval.selection.baseline` explicit path, else the
    resumed checkpoint's run dir `metrics/eval_log.jsonl` (weights taken from
    that run's `config.snapshot.yaml` so the best row matches how it was
    picked). JSON report files with a `suites` list also work.
  - Cap enforcement: candidate is rejected as `best_primary` when any capped
    suite's drop `b_i - s_i` exceeds its cap; counts toward
    `bad_evals`/patience. `cap_ok`/`cap_violations`/`z` are logged per gen eval.
  - Gating (no silent Stage A change): stage A stays raw unless
    `eval.selection.baseline` is set; stage B defaults to zscore+caps;
    `eval.selection.zscore: true|false` forces either way.
  - Tests: `tests/training/test_selection.py` (hand-computed z-scores, cap
    rejection, baseline-from-resume, gating).
  - **NaN DDP deadlock fixed**: `sync_nan_stop()` computes the local NaN/Inf
    flag on every rank and all ranks reach `all_reduce_max` together, so a rank
    0 NaN no longer breaks while rank 1 keeps training into a hung collective.
    Unit tests incl. a simulated 2-rank max-reduce.
  - **Resume safety**: `apply_new_embed_rows(..., required=True)` raises when
    `new_embed_rows.pt` is missing and `peft.new_embed_start` is configured
    (was a silent `False`); the new-embed grad mask is reinstalled on the
    resume path (shared `_install_new_embed_grad_mask`). Tests cover
    missing-file raise, shape mismatch, and mask zeroing of old rows.
  - **MPS fp16 GradScaler**: MPS now trains with fp32 master weights + mps
    autocast (fp16 compute) + `torch.amp.GradScaler('mps')`. Verified on torch
    2.13: `GradScaler('mps')` constructs but raises "unscale FP16 gradients" on
    fp16 master weights, so master weights must be fp32. Gated by
    `build_grad_scaler`; CUDA bf16 path unchanged. `autocast_ctx` now supports
    mps. Tests incl. a gated MPS scaler training loop.
  - **Global-batch parity check** (`global_batch_parity`): warns when
    `batch_size * world * grad_accum_steps != train.global_batch_size`
    (`strict_global_batch: true` raises). Added `global_batch_size: 16` to the
    default `configs/training.yaml` so local (16) vs H200 (32) is explicit.
  - **Train-pool hash verify at launch** (`verify_pool_hashes`): recomputes
    `file_sha256` prefix for every `*_sha256_prefix` in the data manifest
    (`source_pool`, `assignment`, `replay_pool`) and warns on drift
    (`data.strict_source_pool_hash: true` raises). `resolve_train_path` now
    merges the frozen build's sibling `*_manifest.json` hash keys into the
    `train_jsonl` override manifest, so the frozen A1/A2/Bp files are actually
    checked against the current pool at launch (previously the override path
    carried no hash and nothing was verified).
  - **Run registry** (`register_run`): writes/updates
    `{output_root}/runs.json` mapping `run_id -> {run_dir, stage, curriculum,
    config_snapshot, data_manifest, backend_info, train_log, eval_log,
    run_summary, best_primary, resume_adapters, start_ts}` so "A2 best" is
    findable without grep archaeology.

- **Tokenizer tests for all previously-untested modules** (AGENTS.md coverage):
  - `test_matrix_configs.py`: dataclass determinism, name/prefix, opts_tag,
    phase1 (20 configs) / phase2 (5 toggles) presets, `_with` override,
    sus=False default, seed-informational.
  - `test_bench_matrix.py`: probe single/multi-piece, UNK rate (incl. div-by-zero),
    model discovery glob, held-out doc filtering.
  - `test_benchmark.py`: doc-id normalization, Devanagari counting, entropy
    (uniform/single/empty), Dev-pieces on a tiny SPM.
  - `test_train_matrix.py`: top-name ranking by HI c/t + v2_joint filter,
    manifest resume (cached skip, corrupt-manifest fallback).
  - `test_train_v2.py`: model prefix, corpus ensure/prepare/raise, grid skip/train.
  - `test_train_full_joint.py`: unigram/bpe attempt ladders.
  - `test_deep_dive.py`: BPE merge-priority records + Devanagari detection,
    theoretical-bounds keys.
  - Tiny SPM fixtures train on synthetic legal text (tmp_path only).

- **Tokenizer freeze now matrix-consistent (split_by_unicode_script=False)** (DESIGN §39):
  - `train()` never passed `split_by_unicode_script`, so the shipped freeze
    `joint_full_41000` inherited SPM's default True while all 35 matrix models
    use False -- the matrix did not actually confirm the freeze, and the
    "+7% 41k->64k" claim conflated vocab size with the script-split axis
    (true vocab-only gain ~1.9%).
  - `train()` now takes and passes `split_by_unicode_script` (default False);
    `train_full_joint` threads it; `--split-by-unicode-script` CLI flag.
  - Retrained the freeze on 2x H200 (profile=full, same dedup corpus, sus=False);
    proto verified; re-benchmarks identical to matrix 41k (HI c/t 4.715 on the
    post-alignment held-out set).
  - Dropped the invalid `seed` SPM arg (TrainerSpec has no seed field --
    `TokenizerConfig.seed` is informational only).

- **Alignment quality gates: 0.6 floor + margin + junk filters** (DESIGN §38):
  - `MIN_SIMILARITY` 0.5 -> 0.6; `SIM_MARGIN = 0.01` (mutual-best winner beats
    runner-up on both sides, kills exact-tie duplicates, tuned so genuine
    near-tie sentences survive -- 0.02 lost 15 complete sentences, 0.01 loses 4).
  - Junk filters: number-only pairs; EN fragments ending in a bare preposition,
    length-gated to <= 60 chars (protects legitimate long legal sentences).
  - Dead code removed: `SKIP_PENALTY`, `pair_type`, `matched_hi`.
  - Regenerated: 1,422 -> **1,300** pairs, avg sim 0.779 -> **0.796**,
    pairs < 0.6: 99 -> **0**, HI-without-danda 10.9% -> **8.4%**;
    splits 1,110/128/184 -> **1,010/122/168**.
  - Tests: margin keep/drop, number-only, dangling-fragment, long-sentence-keep.

- **Story data-pipeline page made comprehensive** (DESIGN §37):
  - Added inputs table (PDFs / EN clean / legacy clean), EN/HI body-vs-full-doc asymmetry rationale.
  - Deepened every step: doc-6 text-layer regression story + backup, OCR invariant gate,
    data-derived EN proper-noun method, Hindi join rule rationale (vocabulary vs length,
    verified fixed point), over-segmentation design, alignment method (mutual-best vs DP,
    LaBSE vs BGE-M3, current similarity distribution, residual QC noise), frozen-split
    rationale, and merge-safe partial re-align.
  - Corrected segmented danda-less fraction 40.3% -> **34.7%** (1,118/3,221) in
    DESIGN_DECISIONS, CHANGELOG, and the story table -- the tracked files verify 34.7%,
    the earlier 40.3% figure was a mis-print from an intermediate run.

- **Hindi OCR hard-wraps now joined (danda-aware) before segmentation** (DESIGN §37):
  - New `src/preprocessing/join_hindi_lines.py` (mirror of `join_lines.py`): joins a line to
    the next when it does not end in a danda (।) and the join does not cross a blank line, a
    numbered item / bullet / list marker, or a standalone header (case headers, judge names,
    section labels like `बनाम`/`निर्णय`; header detection uses a vocabulary/pattern set on
    both the previous AND next line so a long wrap never absorbs a following header). Dates
    (DD.MM.YYYY) are exempt from the numbered-item guard so date-start continuations still
    join. Writes `preprocessed/` in place; idempotent (verified fixed-point on all 30 docs).
  - Wired as `join_hi` into `make preprocess` and `run_pipeline.py` (`reextract -> join ->
    join_hi -> segment -> align -> output`).
  - Effect on real corpus: preprocessed HI lines 5,117 -> 1,749; segmented HI sentences
    7,418 -> 3,221 with danda-less fraction 60.7% -> 34.7% (remaining are headers /
    citations / colon intros); aligned pairs 1,458 -> 1,422 with HI-without-danda fraction
    55.3% -> 10.9%; avg LaBSE similarity 0.70 -> 0.779; train/dev/test 1,136/132/190 ->
    1,110/128/184.
  - Tests: `tests/preprocessing/test_join_hindi_lines.py` (tmp_path/monkeypatch only;
    synthetic + real doc-6 OCR snippet, next-header absorption guard, real-corpus
    idempotency over all 30 preprocessed files).

- **OCR invariant now enforced in code, not a manual backup dir** (DESIGN §36):
  - `PDFTOTEXT_CMD` resolved via `shutil.which('pdftotext')` (was hardcoded
    `/opt/homebrew/bin/pdftotext`); `extract_with_pdftotext` returns `None` when absent.
  - `verify_ocr_quality()` checks CORRUPTED docs for Devanagari floors +
    text-layer ligature markers; `run()` flags any write that fails the check;
    `--verify-ocr` CLI mode exits 1 on issues; `make verify-ocr` gate added.
  - Regression test uses the shared `MIN_OCR_DEV` / `TEXTLAYER_MARKERS` constants
    instead of duplicating them; negative test proves degraded files are caught.

- **Preprocessing tests were data-mutating** (DESIGN §35):
  - `test_reextract_pdfs.py` wrote OCR/text-layer output into the real
    `data/hindi/preprocessed/` and `clean/` (e.g. `test_pdftotext_backend_works`
    overwrote doc 6 with degraded text-layer text), and `test_segment_sentences.py`
    rewrote all `segmented/*.txt`. Running the suite destroyed the §34 OCR fix.
  - All reextract/segment/output tests now redirect writes to `tmp_path`
    (`monkeypatch` on `HI_PREPROCESSED_DIR`/`HI_CLEAN_DIR`/`OUTPUT_DIRS`/`OUTPUT_DIR`).
  - Verified: full `pytest tests/` passes and `preprocessed/6.txt` +
    `segmented/6.txt` are byte-identical before/after.

- **Doc 6 (and CORRUPTED set) Hindi was text-layer, not Tesseract** (DESIGN §34):
  - `data/hindi/preprocessed/6.txt` was byte-identical to degraded `clean/6.txt` (4,657 Dev chars; mid-word splits `भार ीय` / `सिसविवल`). Fresh Tesseract: **6,027** Dev chars; ligatures restored.
  - Re-OCR docs 6, 14, 22, 25, 26; text-layer backup in `data/hindi/preprocessed/_backup_textlayer_20260802/`. Docs 14/22/25/26 were already OCR-equivalent (idempotent rewrite).
  - Re-segmented HI for those docs; re-aligned with merge (1458 pairs total unchanged; doc 6 still 36 pairs, HI refs now OCR).
  - Rebuilt `data/processed/{train,dev,test}.jsonl` on frozen Policy-I doc IDs.
  - Rebuilt Stage B' mix `stage_b_Bp_a1136_r126_f0.9.jsonl` so assignment rows match repaired HI.
  - `align_sentences.run` merges when `--doc-ids` is a subset (avoids wiping other docs).
  - `output_format.run` uses frozen `TRAIN/DEV/TEST_DOC_IDS` when the full 30-doc set is present.
  - `segment_sentences` honors `--lang`.
  - Tests: `tests/preprocessing/test_corrupted_docs_ocr.py`.
  - **Not** re-trained A2 or re-scored COMET; prior metrics used the old doc-6 HI refs.

### Added

- **Full tokenizer matrix (35 configs)** (DESIGN §33, EXPERIMENTS §4.5):
  - `src/tokenizer/matrix_configs.py` -- `TokenizerConfig` dataclass with 14 fields (model_type, vocab_size, corpus_key, character_coverage, byte_fallback, split_digits, split_by_unicode_script, normalization, max_sentence_length, user_defined_symbols, seed_sentencepiece_size, input_sentence_size, num_threads, seed). Deterministic name/prefix. Presets: `phase1_configs()` (Cartesian, 20 configs) and `phase2_configs(top)` (secondary-axis toggles, 5 per base).
  - `src/tokenizer/train_matrix.py` -- ProcessPoolExecutor runner (subprocess-per-config so OOM in one doesn't kill others); resumable via `data/analysis/tokenizer_matrix_manifest.json`; ranks phase-1 winners by HI c/t filtered to `v2_joint` (v2_hi excluded as MT-unusable).
  - `src/tokenizer/bench_matrix.py` -- auto-discovers all `sentencepiece_legal_v2_*.model` in `data/models/tokenizers/`, encodes assignment held-out (322 pairs), reports HI/EN c/t, HI/EN ratio, total tokens, Devanagari vocab pieces, legal-term single-piece probe (15 HI + 12 EN terms), and UNK rate. Writes `data/analysis/tokenizer_matrix.json`.
  - Make: `tokenizer-matrix-phase1`, `tokenizer-matrix-phase2`, `tokenizer-matrix-bench` (`MATRIX_PARALLEL=6` default).
  - **Trained on H200 (48 cores, parallel-6):** Phase 1 = 20 configs (5 BPE joint 20-28s each, 5 Unigram joint 320-368s each, 10 v2_hi variants). Phase 2 = 15 configs on top-3 joint bases. Total wall time ~30 min for the whole matrix.
  - **Phase 1 winners (joint corpus, MT-usable):** BPE 64k and Unigram 64k tied at HI c/t **4.695** / total **10,027-10,040**. Ladder: 16k 4.30, 32k 4.55, 41k 4.61, 48k 4.64, 64k 4.70.
  - **Phase 2 axis effects (avg over 3 bases):** `byte_fallback=True` neutral (free robustness); `character_coverage=0.9995` -0.03 c/t + 0.83% UNK (reject); `split_digits=True` **-0.70 c/t catastrophic** (splits case numbers/dates/sections); `split_by_unicode_script=True` -0.24 (kills mixed-script pieces); `user_defined_symbols` 22 legal terms -0.25 + legal-HI probe rate 1.00->0.33 (UDS interferes with merge lattice).
  - **Decision:** `SPM_V2_PRIMARY` stays Unigram 41k. Track D shipped uses NLLB native tokens (no SPM change affects shipped output); existing Track C configs reference 41k freeze; +7% packing gain doesn't justify churn for a track that lost dual-policy. **Recommendation for future Track C rebuild:** `bpe_64k_bf` or `unigram_64k_bf` (byte_fallback for OOV robustness, 100% legal probe hit-rate).

- **BPE vs Unigram at v2 41K -- packing ablation** (DESIGN §32, EXPERIMENTS §4.3):
  - `src/tokenizer/train_full_joint.py --model-type {unigram,bpe}` -- trainer now supports BPE with a distinct output prefix (`_bpe` infix) so the two live side by side.
  - Model: `data/models/tokenizers/sentencepiece_legal_v2_joint_full_bpe_41000.model` (same deduped v2 joint corpus, same profile `full`, same char coverage, same special-token IDs; only `model_type` differs).
  - Held-out (322 pairs): BPE 41K = HI c/t 4.40 / total 10,898 vs Unigram 41K shipped 4.37 / 10,978. Delta +0.7% packing at the same parameter budget. BPE lands between Unigram 48K and 64K on packing.
  - **Decision:** `SPM_V2_PRIMARY` stays Unigram 41K. Delta too small to churn the freeze, and no MT-quality run was done on BPE (packing != translation quality). BPE model kept as ablation artifact for a future C1a / C1c-style run.

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

- **COMET-22 scoring (Unbabel/wmt22-comet-da)** for all shipped hyps (`docs/EXPERIMENTS.md` §5.1, `REPORT.md` §4):
  - `src/evaluation/comet_score.py`: scans `data/analysis/*_hyps.jsonl`, scores each with reference-based COMET-22 on GPU, writes `data/analysis/comet22_summary.json` (cache-safe, resumable). Make: `comet-score`.
  - **Scores (H200 batch=32):** ZS 0.7074 / 0.8022 / 0.7853; A1 0.7140 / 0.7996 / 0.7931; A2 0.7142 / 0.8012 / **0.7944**; A2 DoRA 0.7113 / 0.7980 / 0.7927; B 0.7095 / 0.7888 / 0.7780; B' **0.7165** / 0.7971 / 0.7881; C1c v2 0.6631 / 0.7502 / 0.7529; C1c v1 0.4971 / 0.5319 / 0.5441 (I_test / E_milpac / E_anuvaad).
  - COMET reinforces BLEU/chrF++ verdicts. Nuance: B' edges A2 on I_test COMET (+0.002) but A2 still wins dual policy (E_anuvaad +0.006). Adapters slightly regress E_milpac COMET vs zero-shot (0.8022 -> 0.8012 for A2) -- MILPaC domain shift.

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

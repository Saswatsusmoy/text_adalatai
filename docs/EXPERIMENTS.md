# Experiments and research findings

Running log of experiments for Adalat AI (legal EN->HI). Decisions and short
rationales also live in `DESIGN_DECISIONS.md`; this file holds tables, freezes,
and how to reproduce.

**Assignment report (start here for submission):** [`REPORT.md`](../REPORT.md)

**Interview tour (phase packages):** [`WALKTHROUGH.md`](WALKTHROUGH.md)

**Related:** `CHANGELOG.md`, `DESIGN_DECISIONS.md`, `.local/dual_track_experiment_plan.md` (private).

---

## 1. Project goals (two pillars)

1. **Token efficiency** for Indic (Devanagari) under modern tokenizers.
2. **Domain-adapted translation** EN->HI for Indian court judgments.

**Dual-track plan (adopted):**

| Track | Name | Idea |
|-------|------|------|
| **D** | Defaults | InLegalTrans / NLLB (native tokenizer); Stage A then B LoRA |
| **C** | Custom vocab | Domain SentencePiece + adapt/train model to that vocab |

Both tracks share the same frozen assignment splits and Stage A external data.

---

## 2. Assignment corpus and preprocessing

### 2.1 Data

| Item | Value |
|------|------:|
| Parallel judgments | 30 EN-HI Supreme Court of India docs |
| After LaBSE + filters | **1,458** sentence pairs |
| Train / dev / test pairs | 1,136 / 132 / 190 |
| Split level | **Document** (seed 42) |

Source file is `data/HC Judgments _ ML Assignment Text.xlsx` -- the "HC" in the filename is a misnomer. 21/30 PDFs open with `IN THE SUPREME COURT OF INDIA`; the remaining 9 use the equivalent `[YYYY] N S.C.R.` / `INSC` neutral-citation header. All 30 are SC judgments reviewing High Court orders (mostly Allahabad HC), which likely explains the source filename.

### 2.2 Frozen document IDs (do not change)

```text
train: 2,3,5,6,7,10,11,12,13,14,15,16,17,18,19,20,22,23,25,26,27,28,29,30
dev:   8, 9, 24
test:  1, 4, 21
```

Also in `src/config.py` as `TRAIN_DOC_IDS` / `DEV_DOC_IDS` / `TEST_DOC_IDS`.

### 2.3 Pipeline steps (live)

| Step | Module | Notes |
|------|--------|-------|
| Re-extract Hindi PDFs | `reextract_pdfs.py` | Tesseract OCR default (not text-layer) |
| Join EN hard wraps | `join_lines.py` | Heuristic + data-driven proper nouns |
| Segment | `segment_sentences.py` | spaCy EN; danda HI |
| Align | `align_sentences.py` | LaBSE mutual-best (not DP) |
| Split | `output_format.py` | train/dev/test JSONL |

**Skipped (evidence-based):** BOM strip, CRLF normalize, OCR roman `li.` fix, separate paragraph step. See DESIGN_DECISIONS §2-5.

### 2.4 Alignment filters (assignment)

| Filter | Threshold |
|--------|-----------|
| Min LaBSE similarity | >= 0.5 |
| EN:HI char ratio | 0.3 - 3.0 |
| Min text length | > 3 chars |
| EN near-dedup | Jaccard > 0.85 drops weaker |

Outputs: `data/aligned/all.jsonl`, `data/processed/{train,dev,test}.jsonl`.

---

## 3. External Stage A legal bitext (Gate 9)

### 3.1 Sources

| Source | License | Role |
|--------|---------|------|
| MILPaC (Law-AI) | CC BY-NC-SA 4.0 | Clean legal EN-HI |
| Anuvaad legal EN-HI | CC BY 4.0 | Scale (judiciary, HC/SUVAS, law commission, terms, etc.) |

**Not Stage A bitext:** Prarabdha SFT dataset (mostly EN Q&A; used only for SPM mono text v1).

### 3.2 Processing

Already sentence-aligned upstream. We do **not** re-run PDF/OCR/LaBSE.

- Normalize to project JSONL (`en_text`, `hi_text`, `source`, `doc_id`)
- Length filters (same char ratio / min length as assignment)
- Exact pair dedup

| Artifact | Approx size |
|----------|-------------|
| Combined Stage A | **~992,565** pairs |
| Path | `data/external/parallel/stage_a_en_hi.jsonl` |

Code: `src/preprocessing/ingest_external_parallel.py`  
Make: `make external-ingest`

### 3.3 Role split (dual eval policies)

| Data | Use |
|------|-----|
| **`stage_a_train.jsonl`** | Stage A MT train only (~988k) |
| Assignment train | Stage B only |
| **Policy I** assignment test/dev | Internal eval (frozen docs) |
| **Policy E** external held-out | External legal eval (never MT train) |

#### Policy I -- internal assignment

| Split | Path | Pairs (approx) |
|-------|------|---------------:|
| test | `data/processed/test.jsonl` | 190 |
| dev | `data/processed/dev.jsonl` | 132 |
| train | `data/processed/train.jsonl` | 1,136 |

Docs: test **1,4,21**; dev **8,9,24**; train the rest (seed 42).

#### Policy E -- external held-out

Carved from `stage_a_en_hi.jsonl` **before** Stage A FT (`make external-eval-split`):

| File | Content | Count (seed 42) |
|------|---------|----------------:|
| `eval/milpac_test.jsonl` | 10% MILPaC | 117 |
| `eval/milpac_dev.jsonl` | 10% MILPaC | 117 |
| `eval/anuvaad_test.jsonl` | 3,000 Anuvaad | 3,000 |
| `eval/anuvaad_dev.jsonl` | 1,000 Anuvaad | 1,000 |
| `eval/external_test.jsonl` | milpac+anuvaad test | 3,117 |
| `stage_a_train.jsonl` | remainder | 988,331 |

Manifest: `data/external/parallel/eval/eval_manifest.json`  
Validate: `PYTHONPATH=. python3 -m src.evaluation.eval_sets`  
Code: `split_external_eval.py`, `src/evaluation/eval_sets.py`

**Every MT system reports at least:** `I_test`, `E_milpac_test`, `E_anuvaad_test` (or combined `E_external_test`).

**SPM note:** v2 was fit on full `stage_a_en_hi` before this carve. MT must still train only on `stage_a_train`. Optional later: rebuild SPM excluding E lines for strictest science.

---

## 4. Tokenizer research (pillar 1)

### 4.1 Cross-model survey (early experiment)

**Setup:** Encode all 1,458 assignment pairs; measure HI chars/token, HI/EN ratio, total tokens, Devanagari vocab count.

**Headline families:**

| Family | Hindi behavior |
|--------|----------------|
| SentencePiece (char-level Unigram) | Strong; Dev first-class |
| Multilingual BPE with Dev pieces (o200k, NLLB) | Good |
| **Byte-level BPE** (many Llama-line models) | **Weak**: often 0 Dev tokens; UTF-8 byte fallback; 1.1-2.7x HI cost |

**Important distinction:** "BPE is bad" was **not** proven for all BPE. The failure mode is **byte-level / Dev-blind BPE**. SentencePiece `model_type=bpe` is still Unicode-aware and is a different animal. Domain work uses **Unigram SP**.

Detailed ranking table: DESIGN_DECISIONS §15. Scripts: `benchmark.py`, `deep_dive.py`. Older dump: `data/analysis/tokenizer_metrics.json`.

### 4.2 Custom SPM v1 (Prarabdha mono HI)

| Item | Detail |
|------|--------|
| Data | ~14M chars HI from Prarabdha (Devanagari filter on context/response) |
| Type | Unigram, character_coverage=1.0 |
| Sizes | 16k / 32k / 41k (41k ~ max for that char budget) |
| Paths | `data/models/tokenizers/sentencepiece_{16000,32000,41000}.model` |

On full 1,458 pairs (historical table): domain SP 41k beat Gemma-class packing on this legal set despite tiny pretrain vs web-scale models. Legal HI terms often 1 piece (न्यायालय, अपीलार्थी, ...).

**Reproduce:** `make tokenizer-train-all` (rebuilds from Prarabdha download).

### 4.3 Custom SPM v2 (Stage A + assignment train) -- Track C0

#### Firewall

SPM train text may use:

- All Stage A `en_text` / `hi_text` lines
- Assignment **train** pairs only

Must **not** use assignment **dev** or **test** docs (hard fail in code).

#### Corpora

| Mode | Lines (approx) | Chars (approx) | Path |
|------|---------------:|---------------:|------|
| joint | 1,987,402 | 291M | `spm_corpus_legal_v2_joint.txt` |
| hi | 993,701 | 139M | `spm_corpus_legal_v2_hi.txt` |
| joint deduped | 1,947,527 | 287M | `spm_corpus_legal_v2_joint_dedup_c4096.txt` |

Code: `prepare_spm_corpus.py`, `dedupe_text_file`.

#### Models trained (v1 never overwritten)

| Prefix | Notes |
|--------|--------|
| `sentencepiece_legal_v2_hi_{32,41}k` | Full HI corpus |
| `sentencepiece_legal_v2_joint_{32,41}k` | Early joint; 1M sentence sample (RAM) |
| `sentencepiece_legal_v2_joint_full_{41,48,64}k` | Deduped joint; **all unique lines**; Unigram profile=full |

Full joint on 16GB: exact dedupe (~2% dups) + max line 4096 + `seed_sentencepiece_size=250000` + `train_extremely_large_corpus`. No byte-level BPE.

#### HI-only vs joint (held-out dev+test, 322 pairs)

| Model | HI c/t | HI/EN | Total tok |
|-------|-------:|------:|----------:|
| v2 hi 41k | 4.46 | 0.43 | 14,880 |
| v2 joint_full 41k | 4.37 | 0.72 | 10,978 |
| v1 Prarabdha 41k | 3.95 | 0.74 | 11,965 |

**Finding:** HI-only packs Hindi best but **fragments English** (bad for MT). **Joint** is required for translation-oriented vocab.

#### Vocab-size ablation (joint_full Unigram, same corpus)

| Size | Held-out HI c/t | Held-out total | Test total |
|-----:|----------------:|---------------:|-----------:|
| 41k | 4.37 | 10,978 | 6,211 |
| 48k | 4.38 | 10,937 | 6,194 |
| 64k | **4.42** | **10,819** | **6,125** |

Gains 41k->64k are real but small (~1.4% tokens). Larger V can allocate more pieces to frequent legal collocations and leaves a longer tail of rare IDs for MT embeddings.

#### BPE vs Unigram ablation at 41k (same v2 joint corpus, same profile)

Follow-up to close a gap called out in DESIGN §15: v1 was Unigram only, and no BPE 41k on the v2 joint corpus had been trained. Same dedup+truncate corpus, same profile `full`, same `character_coverage=1.0`, same special-token IDs -- only `model_type` differs.

| Model | Vocab | HI c/t | HI/EN | Total tok (held-out) | Dev pieces |
|-------|------:|-------:|------:|---------------------:|-----------:|
| v2 joint_full 41k (Unigram, shipped) | 41000 | 4.37 | 0.720 | 10,978 | 16,217 |
| **v2 joint_full 41k BPE (new)** | 41000 | **4.40** | 0.721 | **10,898** | 16,371 |
| v2 joint_full 48k Unigram | 48000 | 4.38 | 0.721 | 10,937 | 18,524 |
| v2 joint_full 64k Unigram | 64000 | **4.42** | 0.722 | 10,819 | 23,742 |

**Finding:** BPE 41k marginally beats Unigram 41k on packing: +0.03 HI c/t (+0.7%) and -80 total tokens (-0.7%) on 322 held-out pairs. BPE 41k lands between Unigram 48k and 64k on packing at the 41k parameter budget, with slightly more Devanagari pieces in the vocab.

**Nuance:** SentencePiece BPE is Unicode-aware (`character_coverage=1.0`, no byte fallback) -- it is **not** the byte-level BPE that §4.1 flagged as Devanagari-blind. This BPE variant is a valid vocab strategy for legal HI.

**Caveat -- packing only:** no MT training run was done with the BPE model. Track D shipped keeps NLLB native tokens (Track C1c already showed vocab surgery on a pretrained NLLB is not free even when packing improves). A from-scratch legal MT (C1a-style) or a full C1c-style extension of NLLB with this BPE 41k are the natural next steps -- not run.

Model: `data/models/tokenizers/sentencepiece_legal_v2_joint_full_bpe_41000.model` (1.08 MB vs Unigram 1.12 MB). Trainer: `src/tokenizer/train_full_joint.py --model-type bpe`.

#### Production freeze (Track C)

```text
data/models/tokenizers/sentencepiece_legal_v2_joint_full_41000.model
```

Constant: `src/config.py` -> `SPM_V2_PRIMARY`

**Why 41k (not packing-max 64k):**

- Limit over-specialization on frequent subwords / collocations
- Smaller embedding matrix for Track C1 (~56% fewer rows than 64k)
- Still large gain vs v1 (~8% fewer tokens on frozen test)
- 48k/64k kept as **ablations only**

JSON dumps:

- `data/analysis/tokenizer_metrics_v2.json`
- `data/analysis/tokenizer_benchmark_c0.json`
- `data/analysis/tokenizer_vocab_size_ablation.json`

#### Glossary (n_tokens; 1 is best)

Typical pattern for joint_full 41k: single pieces for न्यायालय, अपीलार्थी, Section, impugned; Writ Petition often 2; S.L.P. still multi-piece. HI-only fragments some English legal strings (Section, impugned).

### 4.4 Reproduce tokenizer work

```bash
# v1 Prarabdha SP
make tokenizer-train-all

# v2 corpora + grid (sample joint + hi)
make tokenizer-c0

# full joint Unigram (dedupe path)
make tokenizer-spm-v2-full-joint
# extra sizes:
PYTHONPATH=. python3 src/tokenizer/train_full_joint.py --vocab-size 48000 --max-chars 4096
PYTHONPATH=. python3 src/tokenizer/train.py \
  --input data/external/spm_corpus_legal_v2_joint_dedup_c4096.txt \
  --vocab-size 64000 \
  --model-prefix sentencepiece_legal_v2_joint_full_64000 \
  --profile full

# held-out bench (default)
make tokenizer-spm-v2-bench
# PYTHONPATH=. python3 src/tokenizer/benchmark.py --eval held_out|test|all
```

Eval default for v2 benches: **held_out** = assignment dev+test only (SPM never trained on those docs).

### 4.5 Full tokenizer matrix (DESIGN §33)

35 configs trained on H200 (48-core, parallel-6). Phase 1 = Cartesian
`{unigram, bpe} x {16k, 32k, 41k, 48k, 64k} x {v2_joint, v2_hi}` = 20 configs.
Phase 2 = 5 secondary-axis toggles applied to the top 3 Phase-1 joint configs
(bpe_64k, unigram_64k, bpe_48k) = 15 additional configs.

Code: `src/tokenizer/matrix_configs.py` (dataclass + presets), `train_matrix.py`
(subprocess-per-config, resumable manifest), `bench_matrix.py` (auto-discover +
legal-term probe + UNK rate). Make: `tokenizer-matrix-{phase1,phase2,bench}`.

Reports:
- `data/analysis/tokenizer_matrix.json` -- full bench (35 rows on 322 held-out pairs)
- `data/analysis/tokenizer_matrix_manifest.json` -- training manifest with elapsed_s + model size

**Phase 1 -- main matrix** (held-out 322 pairs; joint corpus is MT-usable, v2_hi fragments EN):

| Config | Vocab | HI c/t | Total tok | Legal HI probe | Legal EN probe |
|--------|------:|-------:|----------:|---------------:|---------------:|
| v2_joint unigram 16k | 16000 | 4.295 | 11,084 | 1.00 | 1.00 |
| v2_joint bpe 16k | 16000 | 4.303 | 11,099 | 1.00 | 1.00 |
| v2_joint unigram 32k | 32000 | 4.564 | 10,403 | 1.00 | 1.00 |
| v2_joint bpe 32k | 32000 | 4.527 | 10,448 | 1.00 | 1.00 |
| v2_joint unigram 41k (shipped) | 41000 | 4.609 | 10,261 | 1.00 | 1.00 |
| v2_joint bpe 41k | 41000 | 4.604 | 10,253 | 1.00 | 1.00 |
| v2_joint unigram 48k | 48000 | 4.634 | 10,198 | 1.00 | 1.00 |
| v2_joint bpe 48k | 48000 | 4.638 | 10,166 | 1.00 | 1.00 |
| **v2_joint unigram 64k** | 64000 | **4.695** | 10,040 | 1.00 | 1.00 |
| **v2_joint bpe 64k** | 64000 | **4.695** | **10,027** | 1.00 | 1.00 |

v2_hi mirror (unusable for MT: EN probe 0-50%): HI c/t 4.525-4.818 across the ladder.

BPE vs Unigram is not consistent -- Unigram wins 32k (4.564 > 4.527), BPE wins
48k (4.638 > 4.634), tied at 64k. The two families are effectively equivalent on
this corpus/vocab budget.

**Phase 2 -- secondary-axis ablation** (top-3 joint bases: `bpe_64k`, `unigram_64k`, `bpe_48k`):

| Axis | Change vs baseline (avg over 3 bases) | Decision |
|------|--------------------------------------:|----------|
| `byte_fallback=True` | HI c/t 0.00 (unchanged); still 0.00% UNK | **Adopt** -- free robustness against unseen chars |
| `character_coverage=0.9995` | HI c/t -0.029; UNK 0.83% | **Reject** -- introduces UNK for negligible gain |
| `split_digits=True` | HI c/t **-0.700 (catastrophic)** | **Reject** -- splits case numbers, dates, section numbers into digit tokens |
| `split_by_unicode_script=True` | HI c/t -0.237 | **Reject** -- forces EN/HI script boundary splits, kills mixed-script pieces |
| `user_defined_symbols` (22 legal HI+EN) | HI c/t -0.246; legal-HI probe rate 1.00 -> 0.33 | **Reject** -- UDS entries take vocab slots that would have gone to composite pieces; the forced-single-piece guarantee interferes with the merge lattice |

**Frozen ranking** (joint corpus, MT-usable, decode-time packing):

1. `bpe_64k_bf` / `unigram_64k_bf` -- HI c/t 4.695, 10,027-10,040 total tokens, `byte_fallback` for robustness. Tied best.
2. `bpe_64k` / `unigram_64k` -- same packing, no byte-fallback.
3. `bpe_48k` -- HI c/t 4.638, 10,166 total; 16k fewer vocab rows than 64k.

**Decision:** `SPM_V2_PRIMARY` **stays at** `sentencepiece_legal_v2_joint_full_41000.model`.
- Track D shipped uses NLLB native tokens, not v2 SPM; changing the SPM freeze changes no shipped output.
- Existing Track C artifacts and configs reference the 41k freeze.
- +7% packing gain (4.695 vs 4.37 old) doesn't justify churning downstream for a track that already lost dual-policy.

**Recommendation for any future Track C rebuild** (from-scratch Marian, fresh C1c-style extend, etc.): use `sentencepiece_legal_v2_v2_joint_bpe_64000_bf.model` or the unigram equivalent. Model files present under `data/models/tokenizers/`.

**Naming note:** matrix models have a redundant `v2_v2_joint` prefix due to the `TokenizerConfig.name()` template (`legal_v2_{corpus_key}_{...}` with `corpus_key='v2_joint'`). Cosmetic; not fixed since renaming means retraining 35 models.

---

## 5. Dual-track MT (Track D + Track C1c) -- results

Hardware: local M4 default; optional remote 2xH200 (`ssh.jnan.ai`) when free.
See [`docs/HARDWARE_MLX.md`](HARDWARE_MLX.md), [`docs/TRAINING_STRATEGY.md`](TRAINING_STRATEGY.md),
DESIGN_DECISIONS §23-26.

**Decode protocol (all full tests):** beam 4, max_new 256, max_in 256, batch 32, bf16 on H200.

### 5.1 Track D (stock NLLB-600M + LoRA)

| Phase | Run (remote) | Data | Resume | Steps / LR |
|-------|--------------|------|--------|------------|
| Zero-shot | base HF | -- | -- | -- |
| A1 | `nllb600_A_A1_h200_ddp2_*` | A1 80k | base | 3000 / 1e-4 |
| A2 | `nllb600_A_A2_h200_A2_ddp2_*` | A2 150k | A1 best | 3000 / 5e-5 |
| B | `nllb600_B_full_h200_B_ddp2_*` | assignment 1136 | A2 best | 800 / 3e-5 |
| A2 DoRA (ablation) | `nllb600_A_A2_h200_A2_dora_ddp2_*` | A2 150k | base | 3000 / 1e-4 |

Configs: `configs/training_h200.yaml`, `_A2`, `_B`, `_A2_dora`.

| System | I_test BLEU/chrF++ | E_milpac | E_anuvaad |
|--------|-------------------:|---------:|----------:|
| Zero-shot | 18.85 / 44.74 | 34.28 / 55.22 | 39.39 / 60.08 |
| A1 best | 21.67 / 49.16 | 34.66 / 55.98 | 45.17 / 64.33 |
| **A2 best (production)** | **21.86 / 49.66** | **34.90 / 56.46** | **45.80 / 64.83** |
| A2 DoRA best | 21.80 / 49.18 | 35.23 / 56.43 | 45.42 / 64.43 |
| B best | 23.10 / 48.89 | 30.92 / 51.22 | 40.44 / 59.60 |

**Decision (DESIGN §25):** production dual-policy = **A2 `best_primary`**, not Stage B.
B boosts I BLEU but fails E anti-forget (MILPaC chrF++ drop >5).

**DoRA vs A2 LoRA (DESIGN §28):** decode-time delta on all three suites is inside +/-0.5 chrF++
(I_test -0.47, E_milpac -0.03, E_anuvaad -0.40 chrF++). DoRA neither improves nor regresses at
this budget with same data + module surface + step count. A2 LoRA stays shipped; DoRA is a
valid method ablation, not a promotion. Decode elapsed on H200 batch=32 bf16 beam4:
DoRA 554.8s vs A2 LoRA 269.8s (DoRA adds magnitude scaling per forward, ~2x decode cost).

Reports: `data/analysis/nllb600_{A1,A2,B}_h200_best_report.json`,
`data/analysis/nllb600_A2_dora_h200_best_report.json`,
`data/analysis/final_dual_policy_report.json`, `zero_shot_nllb_report(_h200).json`.

**COMET-22 (Unbabel/wmt22-comet-da), same shipped hyps:**

| System | I_test | E_milpac | E_anuvaad |
|--------|-------:|---------:|----------:|
| Zero-shot | 0.7074 | 0.8022 | 0.7853 |
| A1 | 0.7140 | 0.7996 | 0.7931 |
| **A2 (production)** | 0.7142 | 0.8012 | **0.7944** |
| A2 DoRA | 0.7113 | 0.7980 | 0.7927 |
| B | 0.7095 | 0.7888 | 0.7780 |
| B' | **0.7165** | 0.7971 | 0.7881 |
| C1c v2 careful A1 | 0.6631 | 0.7502 | 0.7529 |
| C1c v1 bulk A1 | 0.4971 | 0.5319 | 0.5441 |

COMET reinforces the BLEU/chrF++ ordering with one nuance: **B' edges A2 on I_test COMET (+0.002)** while A2 still leads dual-policy (E_anuvaad +0.006 vs B'). Zero-shot has the highest E_milpac COMET; adapters shift slightly toward SC-style legal Hindi. DoRA is within noise of A2 on all three suites. Scorer: `src/evaluation/comet_score.py`; summary: `data/analysis/comet22_summary.json`.

### 5.2 Track C1c (NLLB vocab-extend + LoRA A1)

C0 freeze still `joint_full_41000` for pure custom-vocab story. C1c instead **extends
NLLB vocab** with legal pieces (keeps NLLB priors) rather than training from-scratch (C1a
Marian scaffold stopped as weak quality path). DESIGN §26.

#### C1c v1 -- bulk extend (ablation)

| Item | Value |
|------|--------|
| Script | `src/training/vocab_extend_nllb.py` |
| Model | `data/models/nllb600_c1c_sp_ext` (+~8k raw SPM piece strings) |
| Train | full `embed_tokens` via PEFT `modules_to_save` + decoder_attn LoRA |
| Config | `configs/training_c1c_h200.yaml` |
| Run | `nllb600_A_A1_c1c_h200_ddp2_20260726T225043Z` (3000 steps, DDP2) |
| Dev best | primary chrF++ ~28.7 @ step 500 (collapsed later) |

**Failure mode:** bulk `add_tokens` can split good NLLB singles (e.g. probe regressions) and
destabilize the embedding table.

#### C1c v2 -- careful extend (primary C experiment)

| Item | Value |
|------|--------|
| Script | `src/training/vocab_extend_nllb_v2.py` |
| Model | `data/models/nllb600_c1c_sp_ext_v2` (+1500 verified surfaces, vocab 257669) |
| Rules | surface only; add only if base fragments; reject protected-substring / probe regression |
| Init | mean of base encode pieces for new rows |
| Train | LoRA + **grad mask** only new emb rows (`peft.new_embed_start=256204`) |
| Config | `configs/training_c1c_v2_h200.yaml` (global batch 32, DDP2) |
| Run (scoreable) | `nllb600_A_A1_c1c_v2_h200_ddp2_20260726T234856Z` |
| Dev best | primary chrF++ **53.61** @ step 1000 |

**Checkpoint fix:** first DDP run lost trained emb rows (PEFT LoRA-only adapter, no
`modules_to_save`). `_save_peft` now writes `new_embed_rows.pt`; eval/resume apply via
`apply_new_embed_rows`. Retrain required before full test.

#### Full test comparison (same protocol as Track D)

| System | I_test | E_milpac | E_anuvaad |
|--------|-------:|---------:|----------:|
| Zero-shot | 18.85 / 44.74 | 34.28 / 55.22 | 39.39 / 60.08 |
| **D A2 (production)** | **21.86 / 49.66** | **34.90 / 56.46** | **45.80 / 64.83** |
| C1c v2 careful A1 | 17.79 / 43.86 | 28.20 / 49.78 | 37.64 / 58.46 |
| C1c v1 bulk A1 | 6.38 / 24.86 | 10.66 / 28.63 | 15.65 / 34.35 |

Deltas C1c v2 vs zero-shot: I -1.06 / -0.89; MILPaC -6.08 / -5.44; Anuvaad -1.75 / -1.63 (BLEU/chrF++).
Deltas C1c v2 vs A2: I -4.07 / -5.80; MILPaC -6.70 / -6.68; Anuvaad -8.16 / -6.37.

Reports: `data/analysis/nllb600_c1c_v{1,2}_h200_best_report.json` (hyps same tag prefix).

**Decision (DESIGN §26 after test):** production dual-policy stays **Track D A2**.
C1c v2 is a negative result vs zero-shot on this A1 budget; careful extend is documented
methodology, not a quality win. C1c v1 is a failed bulk-extend ablation. Optional later
research only (C1c-A2 resume, longer emb warm-up) -- not production.

### 5.3 Dual-track scoreboard (closed for production pick)

```text
System                  I_test BLEU/chrF++   E_milpac        E_anuvaad
Zero-shot NLLB          18.85 / 44.74        34.28 / 55.22   39.39 / 60.08
D A1 LoRA               21.67 / 49.16        34.66 / 55.98   45.17 / 64.33
D A2 LoRA (prod)        21.86 / 49.66        34.90 / 56.46   45.80 / 64.83
D B LoRA                23.10 / 48.89        30.92 / 51.22   40.44 / 59.60
C1c v2 careful A1       17.79 / 43.86        28.20 / 49.78   37.64 / 58.46
C1c v1 bulk A1           6.38 / 24.86        10.66 / 28.63   15.65 / 34.35
```

### 5.4 Decode ablation -- MBR vs beam4 (A2 adapters)

Inference-only ablation (no retrain). Same A2 adapters, same base weights, same
sacreBLEU signatures. MBR = sample N candidates, output argmax mean pairwise
sentence-chrF++ utility (Eikema & Aziz 2020; Freitag et al. 2022).

**Wiring:** `src/evaluation/mbr_decode.py`; CLI `--mbr --mbr-samples N --mbr-temperature T --mbr-top-p P --mbr-utility {chrf,chrfpp}` on `src.evaluation.zero_shot_nllb`. Sampling: `do_sample=True, num_beams=1, top_p=0.9, temperature=1.0, num_return_sequences=N` in one `generate` call. Cost: N x beam1 decode + N^2 sentence-chrF per source (chrF cost negligible vs generation).

| System | I_test | E_milpac_test |
|--------|-------:|--------------:|
| A2 beam4 (H200 bf16, shipped) | 21.86 / 49.66 | 34.90 / 56.46 |
| A2 beam4 (MPS fp16, control) | 21.85 / 49.68 | 34.71 / 56.55 |
| A2 MBR N=8 top_p=0.9 T=1.0 (MPS fp16) | 18.16 / 47.13 | 31.39 / 54.07 |

Device/precision delta (MPS beam4 - H200 beam4): I_test -0.01 / +0.02, E_milpac -0.18 / +0.09 (inside noise).

Decode-only delta (MPS MBR - MPS beam4): I_test **-3.68 / -2.56**, E_milpac **-3.33 / -2.48**.

E_anuvaad_test not run: ~4h on MPS at this rate.

**Decision:** do not promote MBR at these settings. Beam4 stays shipped. Follow-ups not run:
lower temperature (T=0.3-0.5) or epsilon-sampling (Freitag 2022), higher N (32-128), COMET utility.
Negative result stands only at the tried configuration.

Reports: `data/analysis/nllb600_A2_mps_mbr8_best_report.json`, `data/analysis/nllb600_A2_mps_beam4_best_report.json`. Hyps under same tag prefixes. DESIGN_DECISIONS §31.

Legal error / entity panel: not automated (COMET-22 is run over all shipped hyps; see §5.1 COMET table above).

---

## 6. Key conclusions (evidence so far)

1. Assignment pipeline yields a small but clean **doc-level** parallel set (1,458 pairs).
2. External **legal EN-HI** Stage A (~993k) is required for serious domain FT; Prarabdha is not bitext.
3. **Byte-level BPE** is a poor fit for Hindi; domain **SentencePiece Unigram** is strong.
4. Domain SP on legal text beats general SPMs on **this** legal bench (v1 and v2).
5. For MT, train SPM on **joint EN+HI**, not HI-only.
6. Full joint train is feasible on 16GB with **dedupe + memory profile**.
7. Larger vocab (64k) packs slightly better; **41k frozen** for pure Track C emb size.
8. **Track D NLLB LoRA works:** A2 is best dual-policy checkpoint; Stage B without E replay overfits I and forgets E.
9. **Bulk NLLB vocab-extend (C1c v1) fails hard** (breaks singles / emb; test far below zero-shot).
10. **Careful vocab-extend (C1c v2) is still below zero-shot and A2** on full I+E after A1-only train;
    production stays D A2. Custom-vocab surgery did not beat stock NLLB priors on this recipe.
11. Always persist trained emb rows when using grad-mask / non-`modules_to_save` emb training
    (`new_embed_rows.pt`).
12. **MBR N=8 (top_p=0.9, T=1.0, chrF++ utility)** loses to beam4 by ~2.5 chrF++ on both A2 test suites;
    decode-only delta after MPS/H200 control. Beam4 stays shipped. Larger N / lower T / epsilon-sampling /
    COMET utility not yet tried (§5.4).
13. **DoRA** on the same A2 150k data with the same LoRA module surface matches A2 LoRA within +/-0.5 chrF++
    on every suite and costs ~2x decode. Documented method ablation, not a promotion (§5.1, DESIGN §28).
14. **COMET-22 (Unbabel/wmt22-comet-da)** reinforces the BLEU/chrF++ ordering. Nuance: B' edges A2 on I_test
    COMET (+0.002); A2 still leads dual-policy (E_anuvaad +0.006). Adapters slightly regress E_milpac COMET
    vs zero-shot -- MILPaC is close to NLLB's base distribution (§5.1, REPORT §4).

---

## 7. Artifact index

| Path | Contents |
|------|----------|
| `data/aligned/all.jsonl` | Assignment aligned pairs |
| `data/processed/{train,dev,test}.jsonl` | Frozen splits |
| `data/external/parallel/stage_a_en_hi.jsonl` | Stage A bitext |
| `data/external/parallel/stage_a_train.jsonl` | Stage A train after E holdout |
| `data/external/parallel/eval/*` | E milpac/anuvaad dev+test |
| `data/external/spm_corpus_legal_v2_*.txt` | SPM train text |
| `data/models/tokenizers/sentencepiece_*.model` | v1 + v2 SPMs |
| `data/models/tokenizers/sentencepiece_legal_v2_joint_full_41000.model` | **Track C0 freeze** |
| `data/models/nllb600_c1c_sp_ext/` | C1c v1 extended NLLB |
| `data/models/nllb600_c1c_sp_ext_v2/` | C1c v2 careful extended NLLB |
| `data/analysis/tokenizer_*.json` | Bench dumps |
| `data/analysis/zero_shot_nllb_report*.json` | Zero-shot full test |
| `data/analysis/nllb600_{A1,A2,B}_h200_best_report.json` | Track D full test |
| `data/analysis/nllb600_c1c_v{1,2}_h200_best_report.json` | Track C1c full test |
| `data/analysis/nllb600_A2_dora_h200_best_report.json` | Track D A2 DoRA ablation full test |
| `data/analysis/nllb600_A2_mps_{beam4,mbr8}_best_report.json` | A2 decode ablation (MPS beam4 control + MBR N=8) |
| `data/analysis/comet22_summary.json` | COMET-22 per-system per-suite summary (all shipped hyps) |
| `data/analysis/final_dual_policy_report.json` | Combined dual-policy decision dump |
| `data/runs/nllb600_A_A2_* /checkpoints/best_primary` | **Production adapters (D A2)** |
| `src/config.py` | `SPM_V2_PRIMARY`, split IDs |

---

## 8. Local hardware + MLX (training policy)

**Policy:** local-only (no cloud GPU required). Profile before choosing model sizes.

```bash
make profile-hardware
# -> data/analysis/hardware_profile.json
```

Full write-up: [`docs/HARDWARE_MLX.md`](HARDWARE_MLX.md).

| Item | Profiled value |
|------|----------------|
| Chip | Apple M4 (10 cores) |
| Memory | 16 GB unified |
| MLX | GPU device; mlx + mlx-lm OK |
| PyTorch | MPS OK |

| Path | Backend |
|------|---------|
| NLLB / InLegalTrans (enc-dec) | **PyTorch MPS** |
| Small LLM LoRA (1B-3B 4-bit) | **MLX / mlx-lm** |
| Custom SPM freeze | `SPM_V2_PRIMARY` (joint_full 41k); needs training path that can load it |

16GB rules: batch 1, seq ~256, prefer 4-bit for LLMs, subsample Stage A before full 992k.

---

## 9. Open work

- [x] Profile local M4 16GB + MLX/MPS smoke  
- [x] Dual eval policies I + E (split + validate)  
- [x] Track D zero-shot NLLB-600M on MPS (I_test + E_milpac + E_anuvaad)  
  - I_test: BLEU 18.78 / chrF++ 44.62 (n=190)  
  - E_milpac_test: BLEU 34.14 / chrF++ 55.12 (n=117)  
  - E_anuvaad_test: BLEU 39.44 / chrF++ 60.08 (n=3000)  
  - Report: `data/analysis/zero_shot_nllb_report.json`  
- [x] Training strategy + `configs/training.yaml` (`docs/TRAINING_STRATEGY.md`)  
- [x] Implement train loop T1-T2 (subsample, LoRA, logging, checkpoints); smoke 20 steps OK  
- [x] Stage A LoRA full run (A1 then A2) + gen eval / final I+E on H200  
  - A1 best test: I 21.67/49.16; MILPaC 34.66/55.98; Anuvaad 45.17/64.33  
  - A2 best test: I 21.86/49.66; MILPaC 34.90/56.46; Anuvaad 45.80/64.83  
- [x] Stage B LoRA + final I/E table  
  - B best test: I 23.10/48.89; MILPaC 30.92/51.22; Anuvaad 40.44/59.60  
  - Recommend **A2 best** (B fails E anti-forget); report `data/analysis/final_dual_policy_report.json`  
- [x] Track C1c NLLB vocab-extend + LoRA A1 (v1 bulk + v2 careful) + full I+E test  
  - v2 careful: I 17.79/43.86; MILPaC 28.20/49.78; Anuvaad 37.64/58.46 (below zero-shot)  
  - v1 bulk: I 6.38/24.86; MILPaC 10.66/28.63; Anuvaad 15.65/34.35 (failed ablation)  
  - **Production remains D A2**; C1c does not replace Track D (DESIGN §26, EXPERIMENTS §5.2)  
- [ ] Optional mlx-lm 1B-3B 4-bit LoRA smoke  
- [ ] Optional C1c-A2 / longer emb warm-up (research only; not production)  
- [ ] Assignment report write-up  

Last updated: 2026-07-27 (Track D + C1c full dual-policy tests closed; prod = D A2).

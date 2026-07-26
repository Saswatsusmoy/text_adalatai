# Experiments and research findings

Running log of experiments for Adalat AI (legal EN->HI). Decisions and short
rationales also live in `DESIGN_DECISIONS.md`; this file holds tables, freezes,
and how to reproduce.

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
| Parallel judgments | 30 EN-HI Supreme Court docs |
| After LaBSE + filters | **1,458** sentence pairs |
| Train / dev / test pairs | 1,136 / 132 / 190 |
| Split level | **Document** (seed 42) |

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

### 3.3 Role split

| Data | Use |
|------|-----|
| Stage A JSONL | Domain FT/LoRA only (Track D and C) |
| Assignment train | Stage B only |
| Assignment dev/test | Eval only; never SPM fit, never MT train |

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

---

## 5. Dual-track experiment plan (MT next)

Full private write-up: `.local/dual_track_experiment_plan.md`.

### Track D (defaults)

1. Zero-shot InLegalTrans and/or NLLB on assignment **test**
2. Stage A LoRA on `stage_a_en_hi.jsonl`
3. Stage B LoRA on assignment `train.jsonl`
4. Metrics: BLEU, chrF++, COMET, legal error panel, token cost

### Track C (custom vocab)

1. **C0 done:** freeze `joint_full_41000`
2. **C1 next:** adapt or train a model to that vocab (emb resize / vocab-extend / from-scratch)
3. Same Stage A/B data and eval as Track D

### Comparison table (to fill after MT)

```text
System                     | BLEU | chrF++ | COMET | HI tok/doc | legal panel
D zero-shot ...            |      |        |       |            |
D A+B LoRA ...             |      |        |       |            |
C A+B custom-vocab ...     |      |        |       |            |
```

---

## 6. Key conclusions (evidence so far)

1. Assignment pipeline yields a small but clean **doc-level** parallel set (1,458 pairs).
2. External **legal EN-HI** Stage A (~993k) is required for serious domain FT; Prarabdha is not bitext.
3. **Byte-level BPE** is a poor fit for Hindi; domain **SentencePiece Unigram** is strong.
4. Domain SP on legal text beats general SPMs on **this** legal bench (v1 and v2).
5. For MT, train SPM on **joint EN+HI**, not HI-only.
6. Full joint train is feasible on 16GB with **dedupe + memory profile**.
7. Larger vocab (64k) packs slightly better; **41k frozen** to reduce subword-overfit risk and emb size.
8. MT training and dual-track **quality** metrics are **not started** yet.

---

## 7. Artifact index

| Path | Contents |
|------|----------|
| `data/aligned/all.jsonl` | Assignment aligned pairs |
| `data/processed/{train,dev,test}.jsonl` | Frozen splits |
| `data/external/parallel/stage_a_en_hi.jsonl` | Stage A bitext |
| `data/external/spm_corpus_legal_v2_*.txt` | SPM train text |
| `data/models/tokenizers/sentencepiece_*.model` | v1 + v2 SPMs |
| `data/models/tokenizers/sentencepiece_legal_v2_joint_full_41000.model` | **Track C freeze** |
| `data/analysis/tokenizer_*.json` | Bench dumps |
| `src/config.py` | `SPM_V2_PRIMARY`, split IDs |

---

## 8. Open work

- [ ] Track D zero-shot baselines  
- [ ] Track D Stage A + B LoRA  
- [ ] Track C1 adapt model to `SPM_V2_PRIMARY`  
- [ ] Unified quality + token-cost table  
- [ ] Assignment report write-up  

Last updated: 2026-07-26 (C0 complete; 41k joint_full freeze).

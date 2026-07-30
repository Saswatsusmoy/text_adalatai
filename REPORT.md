# Adalat AI -- Assignment Report

English to Hindi translation for Indian court judgments, with two goals:

1. Improve **token efficiency** for Devanagari under modern tokenizers.
2. **Light domain adaptation** of a small MT model for legal EN->HI.

**Production system:** NLLB-200 distilled 600M + LoRA Stage A2  
**Checkpoint:** `data/runs/nllb600_A_A2_h200_A2_ddp2_20260726T212958Z/checkpoints/best_primary`  
**Interactive walkthrough (interview):** [`story/index.html`](story/index.html)  
**Code tour:** [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md)  
**Full tables and freezes:** [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)

---

## 1. Tokenizer analysis and integration strategy

### 1.1 What we compared

On the full assignment bitext (1,458 EN-HI pairs after alignment), we measured:

- Hindi **chars per token** (higher = better packing)
- **HI/EN token ratio** (closer to 1 = less Indic overhead vs English)
- Total token count (proxy for **context length and inference cost**)
- Presence of Devanagari pieces in the vocab (vs pure byte fallback)

Families covered (via HuggingFace / vendor tokenizers + custom SPMs):

| Family | Examples | Hindi behavior on legal text |
|--------|----------|------------------------------|
| Char-aware SentencePiece Unigram | domain SPM, Gemma-class SP | Strong; Devanagari first-class |
| Multilingual BPE with Dev pieces | NLLB, OpenAI o200k | Good packing |
| **Byte-level BPE** | many Llama-line models | **Weak**: often 0 Dev vocab entries; UTF-8 byte split; ~1.1-2.7x HI cost |

Important distinction: the failure mode is **byte-level / Devanagari-blind BPE**, not "all BPE". SentencePiece BPE is still Unicode-aware. Domain work uses **Unigram SentencePiece**.

Code: `src/tokenizer/benchmark.py`, `src/tokenizer/deep_dive.py`. Dumps: `data/analysis/tokenizer_*.json`.

### 1.2 Domain SentencePiece

**v1 (mono legal HI):** Unigram on ~14M chars from Prarabdha Indian legal SFT data. Sizes 16k / 32k / 41k.

On the assignment corpus, **custom SP 41k beat large general models** on packing despite far less pretraining data:

| Model | Vocab | HI c/t | HI/EN | Total tok |
|-------|------:|-------:|------:|----------:|
| Domain SP 41k | 41k | **3.84** | **0.743** | **53,124** |
| Domain SP 32k | 32k | 3.70 | 0.751 | 54,789 |
| Domain SP 16k | 16k | 3.33 | 0.772 | 59,833 |
| Gemma 4 (ref SP) | 262k | 3.42 | 0.800 | 57,095 |
| GPT-4o o200k (ref) | 200k | 2.97 | 0.949 | 60,034 |

**Legal terms as single pieces** under domain SP (examples): न्यायालय, अपीलार्थी, अनुच्छेद, अधिकारिता.

**v2 (Track C0, production SPM freeze):** joint EN+HI from Stage A legal bitext + assignment **train only** (dev/test docs hard-excluded). Full Unigram train on 16GB via exact-line dedupe + optional 4096-char cap.

| Model | HI c/t (held-out) | HI/EN | Total tok |
|-------|------------------:|------:|----------:|
| v2 HI-only 41k | 4.46 | 0.43 | 14,880 |
| **v2 joint_full 41k** | 4.37 | **0.72** | **10,978** |
| v1 Prarabdha 41k | 3.95 | 0.74 | 11,965 |

HI-only packs Hindi best but **fragments English** (bad for MT). **Joint EN+HI** is required for a translation-oriented vocab.

Vocab ablation (same joint corpus): 64k packs ~1.4% better than 41k. Freeze is **41k** for embedding size and lower over-specialization:

```text
data/models/tokenizers/sentencepiece_legal_v2_joint_full_41000.model
```

### 1.3 Integration strategy (two tracks)

| Track | Idea | Outcome |
|-------|------|---------|
| **D (production)** | Keep NLLB native tokenizer; adapt **model** with LoRA | **Shipped** (A2) |
| **C** | Domain SPM + vocab surgery / from-scratch | C0 freeze done; C1c careful extend **below** zero-shot on this budget |

**Why production stays on NLLB tokenizer:** NLLB already has strong multilingual Dev packing. Domain SPM wins on raw token count, but grafting a new vocab onto a 600M pretrained model without long emb warm-up lost quality (Section 4). Token savings without quality is not useful for legal fidelity.

**Practical token-cost lesson for LLMs:** prefer char-aware or multilingual tokenizers over byte-level BPE for Hindi legal text; domain Unigram SP is a strong low-cost option if you control the full stack.

---

## 2. Dataset preparation

### 2.1 Assignment corpus

| Item | Value |
|------|------:|
| Parallel judgments | 30 (Supreme Court of India)[^court] |
| After LaBSE + filters | **1,458** pairs |
| Train / dev / test | **1,136 / 132 / 190** |
| Split | **Document-level**, seed 42 |

[^court]: The assignment package ships as `HC Judgments _ ML Assignment Text.xlsx`, but every PDF is a Supreme Court judgment (21/30 declare `IN THE SUPREME COURT OF INDIA` in the header; the remaining 9 use the equivalent `[YYYY] N S.C.R.` / `INSC` neutral citation). The cases are SC appeals reviewing High Court orders (mostly Allahabad HC), which likely explains the source-file naming.

Frozen doc IDs:

```text
train: 2,3,5,6,7,10-20,22,23,25-30
dev:   8, 9, 24
test:  1, 4, 21
```

### 2.2 Preprocessing

| Step | Module | Notes |
|------|--------|-------|
| Hindi re-extract | `reextract_pdfs.py` | Tesseract OCR (text-layer ligatures too damaged) |
| English line join | `join_lines.py` | Hard wraps + data-driven proper nouns |
| Segment | `segment_sentences.py` | spaCy EN; danda for HI |
| Align | `align_sentences.py` | LaBSE mutual-best |
| Split | `output_format.py` | train/dev/test JSONL |

Filters: LaBSE sim >= 0.5, EN:HI char ratio 0.3-3.0, min length, EN Jaccard near-dedup 0.85.

Reproduce: `make preprocess` (or `python run_pipeline.py --steps preprocess`).

### 2.3 Optional external legal bitext (Stage A)

For serious domain FT, assignment-only data is too small. We ingested public legal EN-HI:

| Source | License | Role |
|--------|---------|------|
| MILPaC (Law-AI) | CC BY-NC-SA 4.0 | Clean legal pairs |
| Anuvaad legal EN-HI | CC BY 4.0 | Scale (judiciary, HC/SUVAS, law commission, terms, ...) |

Pool: ~993k filtered pairs -> `stage_a_en_hi.jsonl`. After held-out carve for Policy E: **`stage_a_train.jsonl`** (~988k) for MT train only.

### 2.4 Dual evaluation policies

Every system is scored on both:

| Policy | Sets | Purpose |
|--------|------|---------|
| **I** (internal) | assignment test (190) | In-domain judgments |
| **E** (external) | MILPaC test (117) + Anuvaad test (3k) | Broader legal; anti-overfit |

No train/eval pair leak check: `src/evaluation/eval_sets.py`.

---

## 3. Model selection and training

### 3.1 Choice

**facebook/nllb-200-distilled-600M** (enc-dec NMT, EN->HI via `eng_Latn` / `hin_Deva`).

Why not a small LLM (TinyLLaMA / Mistral) as the main path:

- Assignment is **sentence-level legal translation**, where a dedicated NMT model is the right prior.
- LoRA on NLLB is single-GPU / dual-GPU friendly (~0.5% trainable params on the default profile).
- Local M4 16GB path (MPS) and remote Hopper H200 path share the same curriculum and metrics.

### 3.2 Efficient adaptation

| Setting | Value |
|---------|--------|
| Method | PEFT **LoRA** (`decoder_attn` profile: decoder self+cross attn) |
| Trainable | ~0.5-0.8% of 600M |
| Stage A1 | 50k Stage A subsample, higher LR |
| Stage A2 | 150k subsample, resume A1, lower LR, 3000 steps |
| Stage B | assignment train only (ablation) |
| Stage B' | assignment + 10% A2 replay (anti-forget) |
| Selection | dual-policy chrF++ with E anti-forget constraint |

Configs: `configs/training.yaml`, `configs/training_h200*.yaml`.  
Trainer: `src/training/train_nllb_lora.py`.  
Eval / hyps: `src/evaluation/zero_shot_nllb.py`, `src/evaluation/metrics_mt.py` (SacreBLEU BLEU + chrF++).

### 3.3 Track C (vocab extend, negative result)

Careful NLLB vocab extend (+1500 legal surfaces) + A1 LoRA (**C1c v2**) stayed **below zero-shot** on full I+E. Bulk extend (**C1c v1**) collapsed quality. Production remains Track D A2. See EXPERIMENTS §5.2.

---

## 4. Quantitative evaluation

Decode protocol (all rows): beam=4, max_new=256, max_in=256, H200 CUDA bf16 batch=32.

Scores are **BLEU / chrF++**.

| System | I_test | E_milpac | E_anuvaad |
|--------|-------:|---------:|----------:|
| Zero-shot NLLB | 18.85 / 44.74 | 34.28 / 55.22 | 39.39 / 60.08 |
| D A1 LoRA | 21.67 / 49.16 | 34.66 / 55.98 | 45.17 / 64.33 |
| **D A2 LoRA (production)** | **21.86 / 49.66** | **34.90 / 56.46** | **45.80 / 64.83** |
| D A2 DoRA (weight-decomposed) | 21.80 / 49.18 | 35.23 / 56.43 | 45.42 / 64.43 |
| D B LoRA (assignment-only specialize) | 23.10 / 48.89 | 30.92 / 51.22 | 40.44 / 59.60 |
| D B' (replay) | 22.22 / 49.41 | 33.82 / 54.84 | 43.46 / 62.51 |
| C1c v2 careful extend A1 | 17.79 / 43.86 | 28.20 / 49.78 | 37.64 / 58.46 |
| C1c v1 bulk extend A1 | 6.38 / 24.86 | 10.66 / 28.63 | 15.65 / 34.35 |

**Delta DoRA vs A2 LoRA (same data, same protocol on H200):** I_test -0.05 / -0.47, E_milpac +0.33 / -0.03, E_anuvaad -0.38 / -0.40 (BLEU / chrF++). All differences <= 0.5 chrF++ -- DoRA neither helps nor hurts on this budget. A2 LoRA stays shipped.

**Deltas A2 vs zero-shot:**

| Suite | BLEU | chrF++ |
|-------|-----:|-------:|
| I_test | +3.01 | +4.92 |
| E_milpac | +0.62 | +1.24 |
| E_anuvaad | +6.41 | +4.75 |

**Why not Stage B:** raises I BLEU but **forgets E** (MILPaC chrF++ drop ~5.2 > 2.0 anti-forget limit). B' recovers much of E but does not beat A2 on dual policy. **Ship A2.**

COMET: not run (optional). Machine reports: `data/analysis/final_dual_policy_report.json` and per-system `*_best_report.json`.

### Decode ablation -- MBR vs beam4 (A2 adapters)

Inference-only test on the shipped A2 checkpoint (no retrain). MBR = sample N candidates, output argmax mean pairwise sentence-chrF++ utility (Eikema & Aziz 2020; Freitag et al. 2022). Wiring: `src/evaluation/mbr_decode.py`, `--mbr --mbr-samples N ...` on `zero_shot_nllb`.

| System | I_test | E_milpac_test |
|--------|-------:|--------------:|
| A2 beam4 (H200 bf16, shipped) | 21.86 / 49.66 | 34.90 / 56.46 |
| A2 beam4 (MPS fp16, control) | 21.85 / 49.68 | 34.71 / 56.55 |
| A2 MBR N=8 top_p=0.9 T=1.0 (MPS fp16) | **18.16 / 47.13** | **31.39 / 54.07** |

Device/precision delta MPS beam4 - H200 beam4: <=0.18 BLEU / 0.09 chrF++ (noise). Decode-only delta MBR - MPS beam4: I_test **-3.68 / -2.56**, E_milpac **-3.33 / -2.48**. E_anuvaad_test not run (~4h on MPS at this rate).

**Verdict:** at N=8 with top_p=0.9 T=1.0 chrF++ utility, MBR does not help. Beam4 stays shipped. Follow-ups not run: lower temperature (T=0.3-0.5) or epsilon-sampling (Freitag 2022), larger N (32-128), COMET utility. Negative result stands only at the tried configuration. Details: DESIGN_DECISIONS §31, `docs/EXPERIMENTS.md` §5.4.

### Token usage (before/after, packing)

Tokenizer path (assignment pairs, domain SP 41k vs general baselines): domain SP cuts total tokens vs Gemma/GPT-class tokenizers on this legal set (Section 1.2 table: ~53k vs ~57-60k). That is a direct reduction in sequence length / cost for any model that *uses* that tokenizer.

MT path (production): stays on **NLLB native tokens** so that pretrained multilingual priors remain intact. Quality gains in the table above are from **LoRA domain adaptation**, not from shrinking the NLLB vocab. Track C tried to combine both; quality did not support shipping it on this budget.

---

## 5. Qualitative evaluation

Examples from **Policy I test** (docs 1, 4, 21). Side-by-side: English source, human reference (OCR pipeline; some refs are truncated or noisy), **zero-shot NLLB**, **A2 LoRA**.

### 5.1 Legal terms + constitutional phrasing

**EN:** Assailing the order of the revisional court, the appellant filed a petition under Article 227 of the Constitution of India, invoking the supervisory jurisdiction of the High Court of Judicature at Allahabad.

| | Hindi |
|--|-------|
| Ref | पुनरीक्षण अदालत के आदेश को चुनौती देते हुए, अपीलकर्ता ने भारत के संविधान के अनुच्छेद 227 के तहत ... |
| Zero-shot | **पुनरावलोकन** अदालत के आदेश पर बल देते हुए, अपीलकर्ता ने ... अनुच्छेद 227 के तहत याचिका दायर की। |
| **A2** | **पुनरीक्षण** न्यायालय के आदेश पर बल देते हुए, अपीलकर्ता ने भारत के संविधान के अनुच्छेद 227 के तहत एक याचिका दायर की, ... पर्यवेक्षी अधिकार क्षेत्र ... |

A2 recovers the legal term **पुनरीक्षण** (revisional) vs zero-shot **पुनरावलोकन** (review), closer to the reference and standard court usage.

**EN:** We also direct, in exercise of our jurisdiction under Article 142 of the Constitution of India, that his salary should be paid over for the period for which he works until a regular candidate is appointed.

| | Hindi |
|--|-------|
| Zero-shot | ... अनुच्छेद 142 ... **अधिकार क्षेत्र** ... |
| **A2** | ... अनुच्छेद 142 ... **अधिकारिता** ... |

A2 prefers the more formal legal register **अधिकारिता** for jurisdiction in this constitutional framing.

### 5.2 Writ petition / party terms

**EN:** 7 The writ petition filed by the appellant was dismissed by a learned Single Judge of the High Court on 9 October 2013.

| | Hindi |
|--|-------|
| Zero-shot | ... दायर **याचिका** ... |
| **A2** | ... दायर **रिट याचिका** ... |

A2 keeps the compound **रिट याचिका** (writ petition) rather than bare "petition".

**EN:** dismissing the Writ Petition filed by the Appellant.

| | Hindi |
|--|-------|
| Zero-shot | **याचिकाकर्ता** द्वारा दायर लिखित याचिका ... |
| **A2** | **अपीलकर्ता** द्वारा दायर लिखित याचिका ... |

Party role: A2 uses **अपीलकर्ता** (appellant), matching the English and the reference; zero-shot often collapses to याचिकाकर्ता (petitioner).

### 5.3 Archaic / formal legal English

**EN:** It was also observed by the revisional court that although it was averred by the appellant that he was put in dark by the counsel earlier engaged by him, there is no reference to his name.

| | Hindi |
|--|-------|
| Zero-shot | **पुनरावलोकन** न्यायालय ... **appellant** (English leak) ... **वकील** ... |
| **A2** | **पुनरीक्षण** न्यायालय ... **अपीलकर्ता** ... **अधिवक्ता** ... |

A2 avoids English leak of "appellant", uses revisional terminology and **अधिवक्ता** (advocate) rather than colloquial **वकील**.

### 5.4 Long compound sentence (procedural)

**EN:** What prevented the appellant from filing the application under Order IX, Rule 7 that year itself has not been satisfactorily explained at all, as the first application was only filed in the year 2017.

Both systems preserve **Order IX, Rule 7** structure. A2 uses **अपीलकर्ता** and **दायर** more consistently with judgment style; zero-shot uses **याचिकाकर्ता** / **दाखिल**. Fidelity is better on role terms and register than on full clause structure (both remain free translations of a long relative construction).

### 5.5 Residual issues (honest)

- **Impugned order** often stays as bare "आदेश" rather than **आक्षेपित आदेश** in both systems.
- References themselves can be OCR-truncated or wrong (e.g. year noise in some HI refs); automatic scores understate ceiling when refs are incomplete.
- Numbers, party lists, and very long multi-clause sentences still need a legal QA pass for production use.

Full hyp dumps: `data/analysis/zero_shot_nllb_I_test_hyps.jsonl`, `data/analysis/nllb600_A2_h200_best_I_test_hyps.jsonl`.

---

## 6. Reflection

### 6.1 What worked

1. **Document-level splits** -- no judgment leakage into test.
2. **OCR over PDF text layer** for Hindi -- ligature damage is worse than OCR noise.
3. **Dual-policy eval (I + E)** -- Stage B looked good on assignment test alone and would have shipped incorrectly without E.
4. **LoRA on NLLB decoder attention** -- cheap, stable, real gains on Anuvaad and internal legal test.
5. **Curriculum A1 -> A2** -- small Stage A subsets first, then larger resume, better than jumping straight to full data on limited steps.
6. **Domain SentencePiece science** -- clear, reproducible evidence that byte-level BPE is a poor fit for Hindi legal text; joint Unigram SP is the right custom-vocab prior.
7. **Documenting failures** -- bulk vocab extend, pure Stage B, C1a from-scratch stop -- as valuable as the winning path.

### 6.2 What did not

1. **Custom vocab + short LoRA budget (C1c)** did not beat stock NLLB priors. Embedding surgery needs more warm-up and data than this assignment budget.
2. **Stage B assignment-only FT** overfits I and forgets external legal.
3. **COMET / legal error panel** not automated (optional; would strengthen human-aligned scoring).
4. Reference HI from OCR is imperfect -- metrics mix model error and ref noise.

### 6.3 How we would improve

| Direction | Why |
|-----------|-----|
| Longer emb warm-up or C1c -> A2 resume | Give vocab-extend a fair shot |
| B' with stronger E replay / multi-task | Specialize without forget |
| COMET or CometKiwi + glossary panel | Better than BLEU alone for legal adequacy |
| Constrained decode for party names / sections | Reduce number and entity slips |
| Human legal review on 50-100 sentences | Ground automatic gains |
| More languages | Same Stage A pattern (legal bitext + dual holdout + LoRA on NLLB/IndicNMT) for other Indic pairs |
| Longer documents | Hierarchical or sliding-window translate + consistency pass for full judgments |

### 6.4 Feasibility note

The core prototype (preprocess, tokenizer benches, LoRA smoke, dual-policy metrics) runs on a **single consumer GPU or Apple Silicon 16GB**. Full A2 / dual-GPU H200 runs were used to finish the dual-track scoreboard quickly; configs and code for the laptop path remain in `configs/training.yaml` and the Makefile smoke targets.

---

## 7. Codebase map

| Path | Role |
|------|------|
| `src/preprocessing/` | OCR, join, segment, align, splits, Stage A ingest |
| `src/tokenizer/` | SPM train, benchmark, deep dive |
| `src/training/` | NLLB LoRA, subsample, vocab extend, CUDA/DDP helpers |
| `src/evaluation/` | BLEU/chrF++, suite loaders, zero-shot / adapter decode |
| `configs/` | Preprocess + training YAMLs |
| `tests/` | pytest (preprocess, tokenizer, train, eval) |
| `Makefile` | One-command targets |
| `docs/EXPERIMENTS.md` | Full freeze tables |
| `story/` | Interactive process log |
| `data/analysis/` | Metrics JSON + hyp JSONL |

```bash
# Tests
make test

# Assignment preprocess
make preprocess

# Tokenizer benches (after SPM train)
make tokenizer-bench

# Zero-shot baseline (needs HF NLLB weights)
make zero-shot-nllb

# LoRA smoke (local)
make train-nllb-smoke
```

Production adapters are small PEFT folders (~tens of MB). Base weights load from HuggingFace `facebook/nllb-200-distilled-600M`. Large Stage A bitext and full run trees are gitignored; rebuild with Makefile targets or attach adapters at eval time:

```bash
PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb \
  --adapters data/runs/.../checkpoints/best_primary \
  --tag A2_best
```

---

## 8. Key takeaways

1. **Token inefficiency for Hindi legal text is real** under byte-level BPE; domain Unigram SentencePiece and multilingual char-aware tokenizers fix most of the overhead.
2. For **high-fidelity translation**, adapting **NLLB with LoRA on legal Stage A bitext** beat zero-shot on internal and external legal tests without multi-hour full fine-tunes.
3. **Domain vocab surgery is not free** -- it can destroy quality faster than it saves tokens if embeddings are not trained carefully.
4. **Always hold out external legal data** when specializing on a tiny assignment set; otherwise you ship an overfit model.
5. Production recommendation: **Track D A2 LoRA adapters on stock NLLB-600M**, with dual-policy monitoring as a permanent gate.

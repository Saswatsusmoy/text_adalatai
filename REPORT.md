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

### 1.1 Cross-family survey (motivation)

Measured on the raw assignment bitext (1,458 EN-HI pairs, pre Hindi-line-join): Hindi chars per token, HI/EN token ratio, total tokens, presence of Devanagari pieces in the vocab.

| Family | Examples | Hindi behavior on legal text |
|--------|----------|------------------------------|
| Char-aware SentencePiece Unigram | domain SPM, Gemma-class SP | Strong; Devanagari first-class |
| Multilingual BPE with Dev pieces | NLLB, OpenAI o200k | Good packing |
| **Byte-level BPE** | many Llama-line models | **Weak**: often 0 Dev vocab entries; UTF-8 byte split; ~1.1-2.7x HI cost |

**Distinction that matters:** the failure mode is **byte-level / Devanagari-blind BPE**, not "all BPE". SentencePiece `model_type=bpe` is still Unicode-aware -- see matrix results in §1.2 where it ties Unigram at the top end.

**Domain SPM vs large general tokenizers** (v1 Unigram on ~14M chars of legal HI, raw assignment bitext):

| Model | Vocab | HI c/t | HI/EN | Total tok |
|-------|------:|-------:|------:|----------:|
| Domain SP 41k | 41k | **3.84** | **0.743** | **53,124** |
| Domain SP 32k | 32k | 3.70 | 0.751 | 54,789 |
| Domain SP 16k | 16k | 3.33 | 0.772 | 59,833 |
| Gemma 4 (ref SP) | 262k | 3.42 | 0.800 | 57,095 |
| GPT-4o o200k (ref) | 200k | 2.97 | 0.949 | 60,034 |

A 41k domain SPM beats Gemma-262k and GPT-4o-200k on packing despite ~1000x less pretraining data. Legal terms encode as single pieces: न्यायालय, अपीलार्थी, अनुच्छेद, अधिकारिता. This established the direction: build a proper domain SPM on the full Stage A + assignment train corpus and sweep the design space.

Code: `src/tokenizer/benchmark.py`, `src/tokenizer/deep_dive.py`. Dumps: `data/analysis/tokenizer_*.json`.

### 1.2 Full 35-config tokenizer matrix (DESIGN §33)

Rather than picking one freeze by hand, ran the full Cartesian for primary axes plus a one-at-a-time ablation for secondary axes. All models trained on the **v2 legal corpus** = Stage A (MILPaC + Anuvaad, ~993k pairs) + assignment **train only** (dev/test docs hard-excluded in code). 35 tokenizers, H200 48-core parallel-6, ~30 min total wall clock.

**Method.**

*Phase 1 -- main matrix (20 configs, all defaults):* `{unigram, bpe} × {16k, 32k, 41k, 48k, 64k} × {v2_joint, v2_hi}`.

*Phase 2 -- secondary-axis ablation (15 configs; 5 single-axis toggles on top-3 Phase-1 joint bases):* `byte_fallback`, `character_coverage=0.9995`, `split_digits`, `split_by_unicode_script`, `user_defined_symbols` = 22 legal EN+HI protected terms.

Code: `src/tokenizer/matrix_configs.py`, `train_matrix.py`, `bench_matrix.py`. Make: `tokenizer-matrix-{phase1,phase2,bench}`.

**Phase 1 result (joint corpus, MT-usable)** -- held-out 322 pairs (assignment dev + test, never in SPM train):

| Vocab | Unigram HI c/t | BPE HI c/t | Best total tok |
|------:|---------------:|-----------:|---------------:|
| 16k | 4.295 | 4.303 | 11,084 |
| 32k | 4.564 | 4.527 | 10,403 |
| 41k | 4.609 | 4.604 | 10,253 |
| 48k | 4.634 | 4.638 | 10,166 |
| **64k** | **4.695** | **4.695** | **10,027** |

- Packing scales with vocab through 64k (4.30 → 4.70). Diminishing returns above 41k (~1.4% each doubling).
- **BPE vs Unigram is a wash**: Unigram wins 32k, BPE wins 48k, tied at 64k. Model type is not a lever here.
- Legal single-piece probe rate: **100% HI** (15/15 terms) and **100% EN** (12/12) for all 10 joint models. UNK rate 0.00%.
- v2_hi mirror packs slightly better on HI (up to 4.818) but EN legal-probe rate 0-50% → unusable for MT.

**Phase 2 axis effects (avg delta over 3 joint bases):**

| Axis | Δ HI c/t | Notes | Decision |
|------|---------:|-------|----------|
| `byte_fallback=True` | 0.000 | Packing unchanged; UNK 0.00%; adds OOV robustness | **Adopt** |
| `character_coverage=0.9995` | −0.029 | Introduces 0.83% UNK for negligible gain | Reject |
| `split_digits=True` | **−0.700** | Case numbers, dates, section numbers split to digit tokens | Reject (catastrophic) |
| `split_by_unicode_script=True` | −0.237 | Forces EN/HI boundary splits; kills mixed-script pieces | Reject |
| `user_defined_symbols` (22 legal) | −0.246 | Also drops legal-HI probe 1.00 → 0.33 (UDS interferes with merge lattice) | Reject |

**Ranking:**

1. `bpe_64k_bf` / `unigram_64k_bf` -- HI c/t **4.695**, total **10,027-10,040**, `byte_fallback` for robustness. Tied best.
2. `bpe_64k` / `unigram_64k` baseline -- same packing, no byte-fallback.
3. `bpe_48k` -- HI c/t 4.638, 10,166 total; 16k fewer vocab rows than 64k.

**Freeze decision.** `SPM_V2_PRIMARY` stays `sentencepiece_legal_v2_joint_full_41000.model` (Unigram 41k). Track D shipped uses NLLB native tokens -- changing v2 SPM affects no shipped output. +7% packing (4.37 → 4.695) doesn't justify churn for a track that lost dual-policy in §26. **Recommendation for any future Track C rebuild:** `bpe_64k_bf` or `unigram_64k_bf`.

Artifacts: `data/analysis/tokenizer_matrix.json` (35-model bench), `data/analysis/tokenizer_matrix_manifest.json` (training manifest), `data/models/tokenizers/sentencepiece_legal_v2_*.{model,vocab}`.

### 1.3 Integration strategy (two tracks)

| Track | Idea | Outcome |
|-------|------|---------|
| **D (production)** | Keep NLLB native tokenizer; adapt **model** with LoRA | **Shipped** (A2) |
| **C** | Domain SPM + vocab surgery / from-scratch | C0 freeze done; C1c careful extend **below** zero-shot on this budget |

**Why production stays on NLLB tokenizer:** NLLB already has strong multilingual Devanagari packing. Domain SPM wins on raw token count, but grafting a new vocab onto a 600M pretrained model without long emb warm-up lost quality (Section 4). Token savings without quality is not useful for legal fidelity.

**Practical token-cost lesson for LLMs:** prefer char-aware or multilingual tokenizers over byte-level BPE for Hindi legal text; domain SentencePiece (Unigram OR BPE -- they tie) with `byte_fallback` is a strong low-cost option if you control the full stack.

---

## 2. Dataset preparation

### 2.1 Assignment corpus

| Item | Value |
|------|------:|
| Parallel judgments | 30 (Supreme Court of India)[^court] |
| After LaBSE + filters | **1,422** pairs[^join] |
| Train / dev / test | **1,110 / 128 / 184** |
| Split | **Document-level**, seed 42 |

[^join]: The alignment produced 1,458 pairs before the Hindi line-join step. Joining danda-aware OCR hard-wraps yields longer, complete units, so LaBSE mutual-best pairing shifts slightly (1,458 -> 1,422). Scores in Section 4 were measured on the pre-join corpus; no retrain was run after the data-quality fix.

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

### COMET-22 (Unbabel/wmt22-comet-da, reference-based)

Neural metric on the same shipped hyps. Scored on H200 (`src/evaluation/comet_score.py`, `make comet-score`).

| System | I_test | E_milpac | E_anuvaad |
|--------|-------:|---------:|----------:|
| Zero-shot NLLB | 0.7074 | **0.8022** | 0.7853 |
| D A1 LoRA | 0.7140 | 0.7996 | 0.7931 |
| **D A2 LoRA (production)** | 0.7142 | 0.8012 | **0.7944** |
| D A2 DoRA | 0.7113 | 0.7980 | 0.7927 |
| D B LoRA | 0.7095 | 0.7888 | 0.7780 |
| D B' (replay) | **0.7165** | 0.7971 | 0.7881 |
| C1c v2 careful A1 | 0.6631 | 0.7502 | 0.7529 |
| C1c v1 bulk A1 | 0.4971 | 0.5319 | 0.5441 |

Observations:

- **Adapters help I_test COMET** (A2 +0.007, B' +0.009 vs zero-shot). Small absolute values but consistent direction.
- **Adapters slightly hurt E_milpac COMET** (all adapted systems score below zero-shot). MILPaC is already close to base's training distribution -- LoRA moves the model toward SC-style legal Hindi and away from MILPaC's mixed sub-domains.
- **B' edges A2 on I_test COMET** (+0.002); A2 still leads dual-policy overall (E_anuvaad +0.006 vs B'). Same story as chrF++.
- **DoRA is inside noise vs A2 on all three** (-0.003 max). Consistent with the BLEU/chrF++ verdict.
- **C1c v1 catastrophic**, C1c v2 clearly below zero-shot. Consistent.

Machine reports: `data/analysis/comet22_summary.json`, per-system `*_best_report.json` (BLEU/chrF++), `final_dual_policy_report.json`.

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
3. **DoRA on same A2 data** matched A2 LoRA within +/-0.5 chrF++ on every suite and cost ~2x decode -- valid method ablation, not a win.
4. **MBR N=8 (top_p=0.9, T=1.0, chrF++ utility)** lost to beam4 by ~2.5 chrF++ on both suites tested; adds no quality at this configuration.
5. **Legal error / entity panel** not automated (COMET-22 is now run and reinforces the ordering; reference-free QE like CometKiwi and a glossary panel remain gaps).
6. Reference HI from OCR is imperfect -- metrics mix model error and ref noise.

### 6.3 How we would improve (ranked, not run)

Concrete levers to push A2 beyond its current dual-policy numbers. Each entry names the change, an expected delta on I_test BLEU, wall-clock on H200, and the paper/technique it draws from. Deltas are literature estimates on comparable MT benchmarks, not measured on this corpus.

**Tier 1 -- biggest defensible gains**

| Lever | Change | Expected delta | H200 cost | Basis |
|-------|--------|---------------:|----------:|-------|
| Scale base to NLLB-1.3B distilled | Swap `nllb-200-distilled-600M` -> `-1.3B`; same LoRA decoder_attn r=16; rerun A1 -> A2 curriculum | +2-3 BLEU | ~4-6h | Classic scaling; single-lever change |
| Back-translation on Anuvaad HI-only | Reverse-decode monolingual legal HI via NLLB HI -> EN; LaBSE >= 0.6 filter; append synthetic pairs to Stage A; retrain A2 | +0.5-1.5 BLEU | ~4h | Sennrich et al. 2016 (canonical MT augmentation) |
| CPO on top of A2 (ALMA-R) | (ref, A2 hyp, ZS hyp) triplets from existing hyp files; Contrastive Preference Optimization loss for 500-1000 steps | +1-2 BLEU | ~5-8h | Xu et al. 2024; current MT preference-optimization SOTA; no new data required |

**Tier 2 -- solid, narrower**

| Lever | Change | Expected delta | H200 cost | Basis |
|-------|--------|---------------:|----------:|-------|
| Sequence-level KD from NLLB-3.3B | NLLB-3.3B decodes Stage A EN -> HI; A2-600M student trains on teacher outputs | +1-2 BLEU | ~6-8h | Kim & Rush 2016; teacher-bounded, low risk |
| Rejection sampling + COMET-QE filter | A2 samples N=8 per source; xCOMET-QE keeps top 25%; SFT one more A2 pass | +0.3-0.8 BLEU | ~4-6h | Meta Llama-3 recipe; self-distillation |
| Larger LoRA module surface | Add encoder q,k,v,out and both-sides FFN `fc1,fc2`; retrain A2 same steps | +0.3-0.8 BLEU | ~4h | More capacity at same-ish param budget; overfitting risk on 1,136 train pairs |

**Tier 3 -- polish (stacks on Tier 1)**

| Lever | Change | Expected delta | H200 cost | Basis |
|-------|--------|---------------:|----------:|-------|
| Constrained decode for entities + sections | Glossary from Stage A (Article X -> अनुच्छेद X, Section Y -> धारा Y, party names, S.L.P. numbers); `PrefixConstrainedLogitsProcessor` at inference | +0.2-0.5 BLEU | ~4-6h | Higher qualitative signal than the number suggests; fixes concrete errors in §5 |
| MBR redo (Freitag 2022) | N=64, epsilon-sampling e=0.02 (not top_p=0.9), COMET-utility (not chrF utility) | +0-1 chrF++ | ~4-6h | Rescue attempt for the N=8 chrF-utility MBR that lost in DESIGN §31; this is what the paper actually recommends |
| Larger Stage A subsample (A3=400k) | Bump A2's 150k to 400k of the 988k Stage A at same steps | +0.3-0.8 BLEU | ~4h | Simple data scale; diminishing returns past ~50% |
| CometKiwi (reference-free) + glossary panel | Add xCOMET-QE / CometKiwi as a fourth metric; extract per-suite legal-entity accuracy on the glossary | metric only | ~2h | Beyond reference-based COMET-22 already scored; QE is what production uses when refs don't exist |

**Recommended combos:**

- **Best single lever**: Tier 1 #1 (NLLB-1.3B) -- biggest gain, one config change, lowest risk.
- **Best two-lever combo (~1 day H200)**: Tier 1 #1 + #2 (1.3B + back-translation). Independent axes, additive gains. Estimated **+3-5 BLEU on I_test**.
- **Modern-technique story**: Tier 1 #1 + #3 (1.3B + CPO). Frontier flavour; harder to defend if drilled on CPO length bias / KL / reward hacking.
- **Assignment-fidelity story**: Tier 1 #1 + Tier 3 constrained decode + qualitative rerun on the §5 examples.

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

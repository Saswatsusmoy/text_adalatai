# Training and monitoring strategy (Track D first)

Advanced plan for **efficient and accurate** fine-tuning on **local Apple M4 16GB**.
Implements dual-track intent: defaults (NLLB/InLegalTrans on MPS) first; custom SPM (Track C) later.

**Related:** `docs/HARDWARE_MLX.md`, `docs/EXPERIMENTS.md`, `configs/training.yaml`,
zero-shot baselines in `data/analysis/zero_shot_nllb_report.json`.

This document is the **source of truth for how we train**. Code must match it.

---

## 0. Goals and non-goals

### Goals

1. Improve over **zero-shot NLLB** on Policy **I** and **E** without leaking test data.
2. Fit **16GB unified memory** (stable runs, resume-safe).
3. Detect **overfitting**, **catastrophic forgetting**, and **eval-set gaming**.
4. Keep every run **reproducible** (seed, data hash, config snapshot, checkpoint).

### Non-goals (this phase)

- Full FT of 1B+ models
- Cloud multi-GPU
- Track C vocab surgery (separate after D works)
- Optimizing only I_test while ignoring E (or the reverse)

---

## 1. Frozen baselines (do not move the goalposts)

Zero-shot NLLB-600M (`facebook/nllb-200-distilled-600M`), MPS, beam 4, max 256:

| Suite | n | BLEU | chrF++ |
|-------|--:|-----:|-------:|
| I_test | 190 | **18.78** | **44.62** |
| E_milpac_test | 117 | **34.14** | **55.12** |
| E_anuvaad_test | 3000 | **39.44** | **60.08** |

Any Stage A/B claim must beat these **on the same suites** with the same decode settings unless an ablation is labeled.

---

## 2. Data contracts (absolute)

| Role | Path | Allowed use |
|------|------|-------------|
| Stage A train | `data/external/parallel/stage_a_train.jsonl` | Domain FT only |
| Stage B train | `data/processed/train.jsonl` | Assignment specialize only |
| I_dev | `data/processed/dev.jsonl` | Early stop / LR / rank selection |
| I_test | `data/processed/test.jsonl` | **Final** internal score only |
| E_milpac_dev / E_anuvaad_dev | `data/external/parallel/eval/*_dev.jsonl` | Domain early stop / sanity |
| E_*_test | `eval/*_test.jsonl` | **Final** external score only |

**Forbidden:** train on any `*_test.jsonl`; train on full `stage_a_en_hi.jsonl` after the E carve;
tune hyperparameters on I_test or E_*_test.

### 2.1 Stage A curriculum (efficiency + accuracy)

Do **not** dump all ~988k pairs on day one.

| Phase | Data mix | Approx size | Why |
|-------|----------|-------------|-----|
| **A0 smoke** | Random 2k from stage_a_train | 2k | Pipeline + OOM check |
| **A1 quality core** | All MILPaC train + HC/SUVAS + law_commission subsample | ~50–80k | Clean legal register |
| **A2 scale** | A1 + judiciary sample | +50–100k (cap ~150k total first) | Volume without 988k cost |
| **A3 full (optional)** | Remaining stage_a_train | up to ~988k | Only if A2 still underfits E |

Sampling: **seed 42**, stratified by `source` field, written to
`data/external/parallel/stage_a_train_subsample_{tag}.jsonl` with a manifest
(hash, counts per source, seed).

### 2.2 Stage B

- Pure assignment train (~1136 pairs) was the first B run; it **failed** dual-policy (E collapse). Kept as ablation only.
- **Stage B' (recommended if specializing):** 90% assignment + 10% Stage A replay from A2 subsample.
  - Builder: `python -m src.training.subsample --curriculum Bp --config configs/training_h200_Bp.yaml`
  - Config: `configs/training_h200_Bp.yaml` (resume A2 adapters, LR 2e-5, ~500 steps).
  - Make: `make stage-b-replay-mix` then `make train-nllb-Bp-h200`.
- Low LR, few steps, strong early stop on **I_dev** + hard caps on **E_milpac_dev**.
- Resume continues A2 LoRA surface (decoder_attn r=16); do not expect a new lower-rank profile without merging.

---

## 3. Model and PEFT (Track D default)

### 3.1 Primary model

| Field | Value |
|-------|--------|
| Base | `facebook/nllb-200-distilled-600M` |
| Direction | eng_Latn -> hin_Deva |
| Backend | PyTorch **MPS** (fp16 weights when possible) |
| Full FT | **No** on 16GB |
| Method | **LoRA** (PEFT) |

### 3.2 LoRA targets (NLLB / M2M100)

See **`docs/NLLB_ARCHITECTURE.md`** (inspected structure). Default profile:

```text
peft.profile: decoder_attn
# decoder self_attn + encoder_attn (cross) q/k/v/out only
# ~3.15M trainable (0.51%); encoder frozen
```

Other profiles: `cross_attn`, `decoder_full` (+ decoder FFN), `last4_decoder`, `attn_all`.

| Hyperparam | Stage A start | Stage B start | Notes |
|------------|---------------|---------------|--------|
| rank `r` | 16 | 8–16 | Lower B = less overfit on 1k pairs |
| `lora_alpha` | 32 | 16–32 | alpha ≈ 2r common |
| `lora_dropout` | 0.05–0.1 | 0.1 | Higher on B |
| LR | 1e-4 | 2e-5–5e-5 | B much smaller |
| weight decay | 0.01 | 0.01–0.05 | |
| warmup | 3–5% steps | 5–10% | |
| scheduler | cosine | cosine | |
| epochs / max steps | by tokens not epochs | 3–8 epochs max | See monitoring |
| max_source_length | 256 | 256 | 512 only if memory allows |
| max_target_length | 256 | 256 | |
| train batch | 1 | 1 | |
| grad accum | 8–16 | 4–8 | effective batch 8–16 |
| grad clip | 1.0 | 1.0 | |
| label smoothing | 0.1 | 0.1 | seq2seq default |
| seed | 42 | 42 | |

### 3.3 Precision / memory

- MPS: fp16 or bf16 if stable; fallback fp32 on numerical issues.
- Gradient checkpointing: **on** if available for this model.
- Clear MPS cache between eval generations.
- Close browsers / other heavy apps during train.

### 3.4 Decode for fair compare

Match zero-shot when scoring:

```text
num_beams = 4
max_new_tokens = 256
forced_bos = hin_Deva
```

Ablations (greedy, beam 1) allowed only if labeled.

---

## 4. Training loop design

### 4.1 Step unit

Use **optimizer steps** (after grad accum), not raw batches, for:

- logging interval
- eval interval
- checkpoint interval
- LR schedule

### 4.2 Ordered pipeline

```text
1. Validate data contracts (eval_sets.validate_policies)
2. Build/load Stage A subsample manifest (A0/A1/A2)
3. Init base + LoRA; log trainable param count + % 
4. Smoke: 20 train steps + 1 micro-eval (loss only)
5. Stage A main loop with monitoring (Section 5)
6. Select best Stage A ckpt by primary metric (Section 5.3)
7. Final Stage A score on I_test + E_*_test (once)
8. Stage B from best A (or from base as ablation)
9. Select best B on I_dev (secondary: E_milpac_dev must not collapse)
10. Final B score on I_test + E_*_test (once)
11. Write run_summary.json + promote best adapters
```

### 4.3 Checkpoint layout

```text
data/runs/{run_id}/
  config.snapshot.yaml      # frozen copy of training.yaml + CLI overrides
  data_manifest.json        # subsample paths, counts, sha256
  metrics/
    train_log.jsonl         # one line per log step
    eval_log.jsonl          # one line per eval
  checkpoints/
    step_{k}/               # adapter weights + optimizer optional
    best_primary/           # best by primary selection rule
    last/
  hyps/
    {suite}_step_{k}.jsonl
  run_summary.json
```

`run_id` = `{model_short}_{stage}_{timestamp}_{git_sha_short}`

### 4.4 Resume

- Save `global_step`, RNG states if feasible, best metric, config hash.
- Resume from `last/` or explicit `--resume step_k`.
- Never silently change data manifest mid-run.

---

## 5. Monitoring (accurate fine-tuning)

### 5.1 Always-on signals (every N train steps)

| Signal | Interval (start) | Action if bad |
|--------|------------------|---------------|
| train loss | 10–20 steps | NaN/Inf -> stop; spike -> lower LR |
| grad norm | 10–20 steps | clip; if always huge, LR down |
| tokens/sec, step time | 20 steps | throughput regression note |
| process RSS / unified mem estimate | 50 steps | if >14GB warn; reduce length/accum |
| GPU/MPS OOM | on failure | auto-retry once with smaller max_len or accum |

Log to `train_log.jsonl` (append-only).

### 5.2 Evaluation cadence

| Eval | When | Metrics | Cost control |
|------|------|---------|--------------|
| **Fast loss eval** | every 100–200 steps | NLL on I_dev + E_milpac_dev (no generate) | cheap |
| **Gen micro-eval** | every 400–800 steps | BLEU/chrF++ on **capped** sets: I_dev full; E_milpac_dev full; E_anuvaad_dev **max 200** | medium |
| **Gen full-eval** | end of stage or on new best only | I_test + E_milpac_test + E_anuvaad_test | expensive; **rare** |

Rule: **I_test and E_*_test at most a few times per stage** (best + final), not every epoch.

### 5.3 Primary metric for model selection

Multi-objective, not BLEU-only:

**Stage A (domain):**

```text
primary = 0.5 * z(E_milpac_dev chrF++) + 0.3 * z(E_anuvaad_dev_sample chrF++) + 0.2 * z(I_dev chrF++)
```

Use **chrF++** as lead metric for HI; still log BLEU.  
`z` = standardize vs running history or vs zero-shot baseline delta.

**Stage B (assignment):**

```text
primary = 0.7 * I_dev chrF++ + 0.3 * E_milpac_dev chrF++
```

Hard constraints (reject checkpoint even if primary high):

1. I_dev chrF++ must not fall > **2.0** absolute below Stage A best (Stage B).
2. E_milpac_dev chrF++ must not fall > **3.0** below Stage A best (Stage B anti-forgetting).
3. No NaN loss in last 50 steps.

### 5.4 Overfitting detectors

| Detector | Definition | Response |
|----------|------------|----------|
| Train/eval gap | train loss down, I_dev/E_dev gen flat or down for 3 evals | early stop; raise dropout; reduce r |
| Length bias | hyp/ref length ratio outside 0.7–1.4 on dev | check length penalty; inspect hyps |
| Copy-source | high EN overlap in HI hyp (latin ratio spike) | sample hyps; lower LR |
| E collapse | E metrics drop while I rises (Stage B) | enable replay mix; stop B earlier |
| Memorization | near-dupe train strings score perfectly, hard legal terms fail | qualitative panel |

### 5.5 Qualitative monitor (cheap, high value)

Every gen micro-eval, dump **12 fixed challenge items** (hand-picked once):

- numbers / dates  
- section citations  
- party roles (appellant/respondent)  
- long compounds  
- Latinisms (*impugned*)  

Store side-by-side en / ref / hyp in `metrics/qualitative_{step}.md`.  
Human skim beats chasing 0.2 BLEU.

### 5.6 Dashboard (lightweight)

No heavy MLOps required. Minimum:

1. `train_log.jsonl` + `eval_log.jsonl`
2. Optional: simple plot script later (loss vs step, chrF++ vs step)
3. Console summary every eval:

```text
step=... loss=... gn=... | I_dev chrF++=... dZ=... | E_milpac chrF++=... dZ=... | best=... mem=
```

`dZ` = delta vs zero-shot on same suite (if defined).

---

## 6. Early stopping and patience

| Stage | Patience | Based on |
|-------|----------|----------|
| A | 4–6 gen micro-evals without primary improve | Stage A primary |
| B | 3–4 gen micro-evals | Stage B primary + hard constraints |

On stop: restore `best_primary` adapters; run final test suites once.

---

## 7. Efficiency tactics (16GB)

1. **Subsample Stage A** (Section 2.1) before full data.  
2. **Grad checkpointing** + batch 1 + accum 8–16.  
3. **Cap eval generation** (Anuvaad dev sample 200 during train).  
4. **Cache tokenized datasets** on disk (`data/cache/tokenized/...`) with config hash key.  
5. **Skip full E_anuvaad_test** until stage end (3000 gens ≈ hours).  
6. **Single beam ablation** only for smoke; final = beam 4.  
7. Prefer **more steps on A1/A2 quality data** over one epoch on 988k junk-heavy lines.

---

## 8. Accuracy tactics

1. **Curriculum:** clean legal (MILPaC/HC) before noisy judiciary bulk.  
2. **Label smoothing** + moderate dropout.  
3. **Conservative Stage B LR** (preserve Stage A).  
4. **Dual-policy selection** (Section 5.3) so I and E both matter.  
5. **Decode parity** with zero-shot.  
6. **Seed 42** everywhere; log library versions.  
7. After best ckpt: short **error analysis** on 20 I_test misses (not just aggregate).

---

## 9. Success criteria (Stage A done)

Promote Stage A only if **all** hold on final tests (same decode):

| Suite | Minimum bar |
|-------|-------------|
| I_test chrF++ | >= zero-shot + **1.0** (target +2.0) |
| E_milpac_test chrF++ | >= zero-shot + **1.0** |
| E_anuvaad_test chrF++ | >= zero-shot (no regression > 0.5) |
| No constraint violation | Section 5.3 |

Stage B success:

| Suite | Minimum bar |
|-------|-------------|
| I_test chrF++ | best among {zero-shot, Stage A, Stage B} and >= Stage A |
| E_milpac_test | not worse than Stage A by > 2.0 chrF++ |
| Qualitative panel | no systematic party/number breakage vs Stage A |

---

## 10. Risk register

| Risk | Mitigation |
|------|------------|
| MPS OOM | batch 1, len 256, checkpointing, empty_cache, smaller accum |
| Slow E_anuvaad eval | sample during train; full only at end |
| Overfit assignment in B | low LR, high dropout, E constraint, few epochs |
| Anuvaad noise dominates A | curriculum A1 before A2; source caps |
| Non-reproducible | freeze config snapshot + data manifest + seed |
| False win on I_test | dual E suites + rare full test |

---

## 11. Implementation phases (code)

| Phase | Deliverable |
|-------|-------------|
| **T0** | This doc + `configs/training.yaml` | Done |
| **T1** | `src/training/subsample.py` curriculum manifests | Done |
| **T2** | `src/training/train_nllb_lora.py` LoRA + logs + ckpts | Done (smoke OK) |
| **T3** | Gen eval inside train loop (I_dev / E_milpac / E_anu sample) | Done (use without `--skip-gen-eval`) |
| **T4** | Stage A run (A1 -> A2) | Done on H200 (A2 best recommended for E) |
| **T5** | Stage B + final dual-policy report | Done; pure B boosts I BLEU, regresses E (use A2) |
| **T5b** | Stage B' 90/10 assignment+A2 replay | Done H200; E anti-forget pass vs A2; dual-policy still A2 (DESIGN §27) |
| **T6** | DoRA on A2 data (from base) | Config + code ready; H200 run (DESIGN §28) |
| **C1c** | NLLB vocab-extend v1 bulk + v2 careful + full I+E | Done; both lose to D A2 / v2 loses zero-shot; prod stays A2 (DESIGN §26, EXPERIMENTS §5.2) |

Smoke: `make train-nllb-smoke` (or `--curriculum smoke --max-steps 30 --skip-gen-eval`).

---

## 12. Config entrypoint

Declarative defaults: **`configs/training.yaml`**.  
CLI overrides allowed but must be snapshotted into `data/runs/{run_id}/config.snapshot.yaml`.

---

## 13. Summary

Train **LoRA NLLB** on **curated Stage A subsets**, monitor **loss + dual-policy chrF++ + memory + qualitative probes**, select checkpoints with **multi-objective rules and anti-forgetting constraints**, evaluate **full I/E tests rarely**, then **light Stage B** on assignment train. That is the efficient and accurate path on M4 16GB.

**Status:** T0-T5 and C1c closed. Production dual-policy = Track D A2 (not B, not C1c). See DESIGN §25-26 and `docs/EXPERIMENTS.md` §5.

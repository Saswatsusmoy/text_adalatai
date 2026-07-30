# NLLB-600M architecture analysis for targeted LoRA

Model: `facebook/nllb-200-distilled-600M`  
HF class: `M2M100ForConditionalGeneration` (`model_type: m2m_100`)  
Inspected from live config/weights (2026-07-27).

Purpose: choose **which modules to adapt** so fine-tuning is **efficient on M4 16GB**, **targeted at EN->HI legal MT**, and less wasteful than adapting everything.

---

## 1. High-level stack

```text
input EN tokens
    |
    v
shared embedding (vocab 256206 x d=1024)  ~262M params (tied with lm_head)
    |
    +---> Encoder x12  (self-attn + FFN)     ~202M
    |
    +---> Decoder x12  (self-attn + cross-attn + FFN)  ~202M
    |
    v
lm_head (tied to shared embedding)
    |
    v
HI token logits (hin_Deva forced BOS)
```

| Spec | Value |
|------|------:|
| Encoder layers | 12 |
| Decoder layers | 12 |
| `d_model` | 1024 |
| Attention heads | 16 |
| FFN dim | 4096 |
| Activation | ReLU |
| Max positions | 1024 |
| Vocab | 256,206 (many languages) |
| Total params (reported) | ~615M (embedding shared; lm_head tied) |

**Distilled 600M** means smaller/faster than full NLLB-1.3B/3.3B, still full multilingual enc-dec.

---

## 2. Block anatomy (each layer)

### Encoder layer `i`

```text
self_attn: q_proj, k_proj, v_proj, out_proj   # 1024x1024 each
fc1: 1024 -> 4096
fc2: 4096 -> 1024
(+ LayerNorms; not LoRA'd typically)
```

### Decoder layer `i`

```text
self_attn:   q,k,v,out     # causal LM-side
encoder_attn: q,k,v,out    # CROSS-ATTENTION to encoder states  << critical for MT
fc1 / fc2                  # FFN
```

**Cross-attention (`encoder_attn`)** is where the decoder **reads English encodings while writing Hindi**. For translation domain adaptation this is usually the highest-value block family.

Linear inventory:

| Suffix | Count | Why |
|--------|------:|-----|
| q/k/v/out_proj | 36 each | 12 enc self + 12 dec self + 12 dec cross |
| fc1 / fc2 | 24 each | 12 enc + 12 dec FFN |

---

## 3. What should move for legal EN->HI?

| Subsystem | Role in EN->HI | FT priority |
|-----------|----------------|-------------|
| **Decoder cross-attn** | Align HI generation to EN source | **Highest** |
| **Decoder self-attn** | HI fluency / legal phrasing | High |
| **Decoder FFN** | Lexical/register transforms | Medium-high |
| **Encoder self-attn + FFN** | EN legal understanding | Medium (base NLLB already strong EN) |
| **Shared embedding / lm_head** | Token surface forms | **Avoid** on 16GB (huge, overfit-prone) |

Implication: **decoder-heavy LoRA** is more targeted than encoder+decoder attention for MT domain shift when the source language is already well modeled.

---

## 4. LoRA parameter budgets (r=16)

Approx trainable params if only those Linear modules get LoRA A/B:

| Strategy | Trainable | % of ~615M | Fit / role |
|----------|----------:|-----------:|------------|
| **attn all qkvo** (current default) | 4.72M | 0.77% | Balanced; smoke-verified |
| attn q,v only | 2.36M | 0.38% | Classic PEFT thrift |
| **decoder-only qkvo** (self+cross) | 3.15M | 0.51% | **Recommended Stage A** |
| cross-attn only qkvo | 1.57M | 0.26% | Most MT-targeted, may underfit |
| encoder-only qkvo | 1.57M | 0.26% | Weak for generation-side legal HI |
| attn all + all FFN | 8.65M | 1.41% | Max capacity; more mem/overfit risk |
| decoder attn + decoder FFN | 5.11M | 0.83% | Strong Stage A if underfit |
| last-4 decoder layers (attn+ffn) | 1.70M | 0.28% | Efficient; top layers often task-specific |

**16GB note:** All of these are small vs the frozen base. Bottleneck is **activations** (seq length, beam eval), not LoRA size. Prefer **targeted modules** for accuracy/regularization, not only memory.

---

## 5. Recommended targeting policy

### Stage A (domain legal bitext) -- default

```text
profile: decoder_attn
modules: all decoder self_attn + encoder_attn  q,k,v,out
freeze:  encoder*, embeddings, lm_head
r=16, alpha=32, dropout=0.05
```

**Why:** Domain shift for judgment EN->HI is largely **alignment + HI realization**. Encoder already encodes English well; adapting encoder early can waste capacity or overfit Anuvaad noise.

### Stage A if underfit (after A1 metrics stall)

```text
profile: decoder_attn_ffn
modules: decoder qkvo + decoder fc1/fc2
```

### Stage B (assignment, ~1k pairs) -- lower capacity

```text
profile: cross_attn  OR  decoder_attn with r=8
modules: encoder_attn qkvo only  (or full decoder attn at r=8)
dropout: 0.1
lr: 2e-5 .. 5e-5
```

**Why:** Tiny Stage B set overfits if whole stack is open. Cross-attn only nudges alignment to assignment style while keeping domain HI from Stage A.

### Ablations worth running once

| Ablation | Question |
|----------|----------|
| attn_all vs decoder_attn | Does encoder LoRA help E_milpac? |
| cross_only vs decoder_attn | Is self-attn needed for legal HI? |
| + decoder FFN | Lexical gains vs overfit |
| last-4 decoder layers | Enough for Stage B? |

---

## 6. What not to LoRA (unless experiment)

| Target | Reason |
|--------|--------|
| `model.shared` / embed | 256k x 1024; huge; needs different recipe |
| `lm_head` (tied) | Same mass; Stage B memorizes tokens |
| LayerNorm / biases only | Weak for domain MT |
| Full FT all layers | Unfit for 16GB; overkill for 150k pairs |

Custom SPM (Track C) would force **embedding surgery** -- separate from this NLLB default-tokenizer path.

---

## 7. Depth specialization (optional advanced)

Literature and practice often find **upper decoder layers** more task-specific.

**Efficient variant:**

```text
LoRA on decoder layers 8..11 only (last 4):
  self_attn + encoder_attn (+ optional fc1/fc2)
```

Use if full decoder_attn overfits Stage B or memory during gen eval is tight.

Implementation: PEFT `layers_to_transform` / module name filters (see `train_nllb_lora` profiles).

---

## 8. Interaction with our data and metrics

| Signal | Architectural reading |
|--------|----------------------|
| Zero-shot **I_test** weak (BLEU ~19) vs **E** stronger (~34–39) | Model general MT OK; **assignment legal style / domain gap** -- Stage A should help E and A; Stage B needed for I |
| Anuvaad noise | Prefer **not** max encoder FT on full judiciary early; curriculum A1 first |
| Dual eval | Decoder/cross focus should lift E_milpac (clean legal) without only memorizing assignment |

---

## 9. Efficiency checklist (architecture-aware)

1. LoRA **decoder-first**, not full model.  
2. Keep **max length 256** -- activation cost dominates.  
3. Gen eval: batch 1, empty MPS cache (already).  
4. Do not touch 256k embedding for Track D.  
5. Stage B: **reduce r** or **cross-attn-only**.  
6. Measure trainable% every run (`print_trainable_parameters`).

---

## 10. Config profiles (names used in training.yaml)

| Profile name | Modules | Intended stage |
|--------------|---------|----------------|
| `attn_all` | enc+dec self + cross qkvo | Baseline (current smoke) |
| `decoder_attn` | dec self + cross qkvo | **Stage A default** |
| `cross_attn` | dec encoder_attn qkvo only | Stage B / thrifty |
| `decoder_full` | dec attn + dec FFN | Stage A capacity boost |
| `last4_decoder` | layers 8-11 dec attn (+ffn opt) | Efficient / Stage B |

---

## 11. Bottom line

NLLB-600M is a **12+12 M2M100 enc-dec** with **shared 256k embeddings** and **decoder cross-attention** as the MT hinge.

For optimized legal FT on your machine:

- **Stage A:** LoRA **decoder attention (self + cross)**, r=16  
- **Stage B:** LoRA **cross-attn only or r=8 decoder attn**  
- **Avoid** embedding/lm_head FT  
- **Ablate** encoder LoRA only if decoder-only underperforms on E_milpac  

This is more targeted than “LoRA every q_proj in the whole net,” and better matched to translation than encoder-heavy updates.

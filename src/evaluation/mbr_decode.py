"""MBR decoding: sample N candidates, pick argmax mean pairwise chrF utility.

Eikema & Aziz 2020; Freitag et al. 2022.

Phase 4: the sampling path is seeded via `set_seed` so N-sample runs are
reproducible; the seed is recorded in the eval report.
"""

from __future__ import annotations

import torch
from sacrebleu.metrics import CHRF

from src.training.common import is_cuda, set_seed


DEFAULT_UTILITY = 'chrfpp'
DEFAULT_SAMPLES = 8
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_SEED = 12345

_CHRF_METRICS = {
    'chrf': CHRF(word_order=0),
    'chrfpp': CHRF(word_order=2),
}


def _sentence_scorer(utility: str):
    if utility not in _CHRF_METRICS:
        raise ValueError(f'unknown utility {utility!r}; use chrf or chrfpp')
    metric = _CHRF_METRICS[utility]

    def score(hyp: str, ref: str) -> float:
        return metric.sentence_score(hyp, [ref]).score

    return score


def mbr_pick(candidates: list[str], utility: str = DEFAULT_UTILITY) -> str:
    """Return the candidate with the highest mean pairwise chrF vs peers."""
    if not candidates:
        raise ValueError('empty candidate set')
    if len(candidates) == 1:
        return candidates[0]
    score = _sentence_scorer(utility)
    n = len(candidates)
    utilities = [0.0] * n
    for i in range(n):
        acc = 0.0
        for j in range(n):
            if i == j:
                continue
            acc += score(candidates[i], candidates[j])
        utilities[i] = acc / (n - 1)
    return candidates[max(range(n), key=lambda k: utilities[k])]


def sample_candidates(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    forced_bos_token_id: int,
    max_input_length: int = 256,
    max_new_tokens: int = 256,
    n_samples: int = DEFAULT_SAMPLES,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> list[list[str]]:
    """Nucleus-sample n_samples per input; return one list per input."""
    if not texts:
        return []
    enc = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    enc = {k: v.to(device, non_blocking=is_cuda(device)) for k, v in enc.items()}
    gen_kw = dict(
        forced_bos_token_id=forced_bos_token_id,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=n_samples,
    )
    with torch.no_grad():
        if is_cuda(device):
            dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type='cuda', dtype=dt):
                out = model.generate(**enc, **gen_kw)
        else:
            out = model.generate(**enc, **gen_kw)
    flat = tokenizer.batch_decode(out, skip_special_tokens=True)
    del enc, out
    if device == 'mps' and hasattr(torch, 'mps'):
        torch.mps.empty_cache()
    expected = len(texts) * n_samples
    if len(flat) != expected:
        raise RuntimeError(f'expected {expected} sampled hyps, got {len(flat)}')
    return [flat[i * n_samples : (i + 1) * n_samples] for i in range(len(texts))]


def translate_batch_mbr(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    forced_bos_token_id: int,
    max_input_length: int = 256,
    max_new_tokens: int = 256,
    n_samples: int = DEFAULT_SAMPLES,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    utility: str = DEFAULT_UTILITY,
    seed: int | None = None,
) -> list[str]:
    """Sample + MBR pick for a batch of source strings.

    When `seed` is given, the sampler RNG is reset first so a direct call is
    reproducible; callers that batch many calls should seed once and pass
    `seed=None` so the RNG advances across batches."""
    if seed is not None:
        set_seed(seed)
    batches = sample_candidates(
        texts,
        tokenizer,
        model,
        device,
        forced_bos_token_id=forced_bos_token_id,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
    )
    return [mbr_pick(cands, utility=utility) for cands in batches]

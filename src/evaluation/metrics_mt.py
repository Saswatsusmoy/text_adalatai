"""BLEU + chrF++ (sacrebleu) for dual-policy scoring, plus Phase 4 additions.

Primary BLEU/chrF++ columns (score/signature/precisions/bp/sys_len/ref_len)
are unchanged. Phase 4 adds only additive fields:

- per-metric 'ci' from sacrebleu's built-in bootstrap (`corpus_score(...,
  n_bootstrap=N)`, seeded via the SACREBLEU_SEED env var);
- 'ter' (sacrebleu TER, lower is better) and 'len_ratio' (sys_len / ref_len);
- 'ref_cleaned' (BLEU/chrF++ on references with OCR artifacts stripped;
  hypotheses and sources are untouched -- see ref_cleaner);
- 'entities' (legal-entity recall/precision/F1 -- see entity_panel);
- a 'confidence' summary block (score + mean + 95% CI) per metric;
- paired_ci() / compare_score_pairs(): paired-bootstrap difference CI so
  "DoRA vs A2" / "B' vs A2" verdicts carry confidence intervals.
"""

from __future__ import annotations

import os

import numpy as np
from sacrebleu.metrics import BLEU, CHRF, TER

from src.evaluation.entity_panel import entity_panel
from src.evaluation.ref_cleaner import clean_ref


DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_SEED = 12345
_METRIC_FACTORIES = {
    'bleu': BLEU,
    'chrf': lambda: CHRF(word_order=0),
    'chrfpp': lambda: CHRF(word_order=2),
}


def _sacrebleu_seed() -> str:
    return os.environ.get('SACREBLEU_SEED', str(DEFAULT_SEED))


def _ci_block(r, n_bootstrap: int) -> dict:
    mean, half = float(r._mean), float(r._ci)
    return {
        'mean': round(mean, 4),
        'ci_low': round(mean - half, 4),
        'ci_high': round(mean + half, 4),
        'ci': round(half, 4),
        'n_bootstrap': n_bootstrap,
        'seed': _sacrebleu_seed(),
    }


def corpus_bleu(hyps: list[str], refs: list[str], n_bootstrap: int = 0) -> dict:
    if not hyps:
        return {'score': 0.0, 'signature': '', 'counts': []}
    m = BLEU()
    r = m.corpus_score(hyps, [refs], n_bootstrap=n_bootstrap)
    out = {
        'score': round(r.score, 4),
        'signature': str(m.get_signature()),
        'precisions': [round(p, 4) for p in r.precisions],
        'bp': round(r.bp, 4),
        'sys_len': r.sys_len,
        'ref_len': r.ref_len,
    }
    if n_bootstrap > 1:
        out['ci'] = _ci_block(r, n_bootstrap)
    return out


def corpus_chrf(
    hyps: list[str],
    refs: list[str],
    word_order: int = 2,
    n_bootstrap: int = 0,
) -> dict:
    if not hyps:
        return {'score': 0.0, 'signature': '', 'word_order': word_order}
    m = CHRF(word_order=word_order)
    r = m.corpus_score(hyps, [refs], n_bootstrap=n_bootstrap)
    out = {
        'score': round(r.score, 4),
        'signature': str(m.get_signature()),
        'word_order': word_order,
    }
    if n_bootstrap > 1:
        out['ci'] = _ci_block(r, n_bootstrap)
    return out


def corpus_ter(hyps: list[str], refs: list[str]) -> dict:
    if not hyps:
        return {'score': 0.0, 'signature': ''}
    m = TER()
    r = m.corpus_score(hyps, [refs])
    return {
        'score': round(r.score, 4),
        'signature': str(m.get_signature()),
    }


def _len_ratio(bleu: dict) -> float | None:
    ref_len = bleu.get('ref_len') or 0
    if not ref_len:
        return None
    return round((bleu.get('sys_len') or 0) / ref_len, 4)


def _ref_cleaned_block(hyps: list[str], refs: list[str]) -> dict:
    cleaned = [clean_ref(r) for r in refs]
    return {
        'n': len(hyps),
        'bleu': corpus_bleu(hyps, cleaned),
        'chrfpp': corpus_chrf(hyps, cleaned, word_order=2),
    }


def _confidence_block(metric: dict) -> dict:
    block = {'score': metric['score']}
    if 'ci' in metric:
        block.update(metric['ci'])
    return block


def score_pairs(
    hyps: list[str],
    refs: list[str],
    n_bootstrap: int = 0,
    seed: int | None = None,
) -> dict:
    assert len(hyps) == len(refs), 'hyps/refs length mismatch'
    if seed is not None:
        os.environ['SACREBLEU_SEED'] = str(seed)
    bleu = corpus_bleu(hyps, refs, n_bootstrap=n_bootstrap)
    chrfpp = corpus_chrf(hyps, refs, word_order=2, n_bootstrap=n_bootstrap)
    out = {
        'n': len(hyps),
        'bleu': bleu,
        'chrfpp': chrfpp,
        'ter': corpus_ter(hyps, refs),
        'len_ratio': _len_ratio(bleu),
        'ref_cleaned': _ref_cleaned_block(hyps, refs),
        'entities': entity_panel(hyps, refs),
    }
    if n_bootstrap > 1:
        out['confidence'] = {
            'bleu': _confidence_block(bleu),
            'chrfpp': _confidence_block(chrfpp),
        }
    return out


def paired_ci(
    hyps_a: list[str],
    hyps_b: list[str],
    refs: list[str],
    metric_name: str = 'chrfpp',
    n_resamples: int = 1000,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Paired-bootstrap 95% CI for the corpus metric delta between two systems.

    Resamples per-segment stats from sacrebleu's own extractor (the same
    machinery its built-in bootstrap uses) and recomputes the corpus score on
    each resample, so the CI is over the corpus-level metric, not a mean of
    sentence scores. `significant` is False when the CI spans zero.
    """
    if metric_name not in _METRIC_FACTORIES:
        raise ValueError(f'unknown metric {metric_name!r}; use {sorted(_METRIC_FACTORIES)}')
    if not (len(hyps_a) == len(hyps_b) == len(refs)):
        raise ValueError('paired CI needs equal-length hyps_a/hyps_b/refs')
    empty = {
        'n': 0,
        'delta': 0.0,
        'mean': 0.0,
        'ci_low': 0.0,
        'ci_high': 0.0,
        'significant': False,
        'n_resamples': n_resamples,
        'seed': seed,
    }
    if not hyps_a:
        return empty
    metric = _METRIC_FACTORIES[metric_name]()
    stats_a = np.array(metric._extract_corpus_statistics(hyps_a, [refs]), dtype='float64')
    stats_b = np.array(metric._extract_corpus_statistics(hyps_b, [refs]), dtype='float64')
    full_a = metric._compute_score_from_stats(stats_a.sum(0)).score
    full_b = metric._compute_score_from_stats(stats_b.sum(0)).score
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = rng.integers(0, len(stats_a), size=len(stats_a))
        score_a = metric._compute_score_from_stats(stats_a[idx].sum(0)).score
        score_b = metric._compute_score_from_stats(stats_b[idx].sum(0)).score
        deltas[k] = score_a - score_b
    low = float(np.percentile(deltas, 2.5))
    high = float(np.percentile(deltas, 97.5))
    return {
        'delta': round(float(full_a - full_b), 4),
        'mean': round(float(deltas.mean()), 4),
        'ci_low': round(low, 4),
        'ci_high': round(high, 4),
        'significant': not (low <= 0.0 <= high),
        'n_resamples': n_resamples,
        'seed': seed,
    }


def compare_score_pairs(
    hyps_a: list[str],
    hyps_b: list[str],
    refs: list[str],
    n_resamples: int = 1000,
    seed: int = DEFAULT_SEED,
) -> dict:
    return {
        'bleu': paired_ci(hyps_a, hyps_b, refs, 'bleu', n_resamples=n_resamples, seed=seed),
        'chrfpp': paired_ci(hyps_a, hyps_b, refs, 'chrfpp', n_resamples=n_resamples, seed=seed),
    }

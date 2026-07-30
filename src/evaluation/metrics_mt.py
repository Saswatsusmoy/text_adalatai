"""
MT metrics for dual-policy scoring (BLEU + chrF++).

Uses sacrebleu. Corpus-level scores over hypothesis/reference lists.
"""

from sacrebleu.metrics import BLEU, CHRF


def corpus_bleu(hyps: list[str], refs: list[str]) -> dict:
    if not hyps:
        return {'score': 0.0, 'signature': '', 'counts': []}
    metric = BLEU()
    result = metric.corpus_score(hyps, [refs])
    return {
        'score': round(result.score, 4),
        'signature': str(metric.get_signature()),
        'precisions': [round(p, 4) for p in result.precisions],
        'bp': round(result.bp, 4),
        'sys_len': result.sys_len,
        'ref_len': result.ref_len,
    }


def corpus_chrf(hyps: list[str], refs: list[str], word_order: int = 2) -> dict:
    """chrF++ when word_order=2 (default)."""
    if not hyps:
        return {'score': 0.0, 'signature': '', 'word_order': word_order}
    metric = CHRF(word_order=word_order)
    result = metric.corpus_score(hyps, [refs])
    return {
        'score': round(result.score, 4),
        'signature': str(metric.get_signature()),
        'word_order': word_order,
    }


def score_pairs(hyps: list[str], refs: list[str]) -> dict:
    assert len(hyps) == len(refs), 'hyps/refs length mismatch'
    return {
        'n': len(hyps),
        'bleu': corpus_bleu(hyps, refs),
        'chrfpp': corpus_chrf(hyps, refs, word_order=2),
    }

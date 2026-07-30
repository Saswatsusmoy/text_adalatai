"""BLEU + chrF++ (sacrebleu) for dual-policy scoring."""

from sacrebleu.metrics import BLEU, CHRF


def corpus_bleu(hyps: list[str], refs: list[str]) -> dict:
    if not hyps:
        return {'score': 0.0, 'signature': '', 'counts': []}
    m = BLEU()
    r = m.corpus_score(hyps, [refs])
    return {
        'score': round(r.score, 4),
        'signature': str(m.get_signature()),
        'precisions': [round(p, 4) for p in r.precisions],
        'bp': round(r.bp, 4),
        'sys_len': r.sys_len,
        'ref_len': r.ref_len,
    }


def corpus_chrf(hyps: list[str], refs: list[str], word_order: int = 2) -> dict:
    if not hyps:
        return {'score': 0.0, 'signature': '', 'word_order': word_order}
    m = CHRF(word_order=word_order)
    r = m.corpus_score(hyps, [refs])
    return {
        'score': round(r.score, 4),
        'signature': str(m.get_signature()),
        'word_order': word_order,
    }


def score_pairs(hyps: list[str], refs: list[str]) -> dict:
    assert len(hyps) == len(refs), 'hyps/refs length mismatch'
    return {
        'n': len(hyps),
        'bleu': corpus_bleu(hyps, refs),
        'chrfpp': corpus_chrf(hyps, refs, word_order=2),
    }

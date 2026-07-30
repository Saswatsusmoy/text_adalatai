"""Unit tests for BLEU/chrF++ helpers (no GPU/model required)."""

from src.evaluation.metrics_mt import corpus_bleu, corpus_chrf, score_pairs


class TestMetricsMt:
    def test_identical_scores_high(self):
        refs = ['यह एक परीक्षण वाक्य है।', 'न्यायालय ने अपील खारिज की।']
        hyps = list(refs)
        s = score_pairs(hyps, refs)
        assert s['n'] == 2
        assert s['bleu']['score'] == 100.0
        assert s['chrfpp']['score'] > 99.0

    def test_empty_safe(self):
        assert corpus_bleu([], [])['score'] == 0.0
        assert corpus_chrf([], [])['score'] == 0.0

    def test_mismatch_length_raises(self):
        try:
            score_pairs(['a'], ['a', 'b'])
            raise AssertionError('expected assert')
        except AssertionError:
            pass

"""Unit tests for BLEU/chrF++ helpers (no GPU/model required)."""

from src.evaluation.metrics_mt import (
    compare_score_pairs,
    corpus_bleu,
    corpus_chrf,
    paired_ci,
    score_pairs,
)


REFS = [
    'the appellant filed a writ petition before the high court',
    'the respondent contested the impugned order of the tribunal',
    'the court dismissed the appeal with costs',
    'the petitioner sought review of the earlier judgment',
    'the constitution bench heard the matter on wednesday',
]


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


class TestBootstrapCi:
    def test_confidence_block_present_with_bounds(self):
        s = score_pairs(REFS, REFS, n_bootstrap=30, seed=7)
        conf = s['confidence']
        for metric in ('bleu', 'chrfpp'):
            assert conf[metric]['score'] == 100.0
            assert conf[metric]['mean'] >= 99.9
            assert conf[metric]['ci_low'] <= conf[metric]['ci_high']
            assert conf[metric]['n_bootstrap'] == 30
            assert conf[metric]['seed'] == '7'
        assert 'ci' in s['bleu']
        assert 'ci' in s['chrfpp']

    def test_reproducible_with_fixed_seed(self):
        a = score_pairs(REFS, REFS, n_bootstrap=30, seed=7)
        b = score_pairs(REFS, REFS, n_bootstrap=30, seed=7)
        assert a['confidence'] == b['confidence']
        assert a['bleu']['ci'] == b['bleu']['ci']

    def test_no_confidence_without_bootstrap(self):
        s = score_pairs(REFS, REFS, n_bootstrap=0)
        assert 'confidence' not in s
        assert 'ci' not in s['bleu']

    def test_empty_hyps_bootstrap_safe(self):
        s = score_pairs([], [], n_bootstrap=30)
        assert s['n'] == 0
        assert s['confidence']['bleu']['score'] == 0.0


class TestTerAndLenRatio:
    def test_ter_zero_and_len_ratio_one_when_identical(self):
        s = score_pairs(REFS, REFS)
        assert s['ter']['score'] == 0.0
        assert s['len_ratio'] == 1.0

    def test_len_ratio_over_one_for_verbose_hyps(self):
        verbose = [r + ' extra words here' for r in REFS]
        s = score_pairs(verbose, REFS)
        assert s['len_ratio'] > 1.0

    def test_ter_positive_for_mismatch(self):
        bad = ['unrelated text that shares no tokens']
        s = score_pairs(bad, REFS[:1])
        assert s['ter']['score'] > 50.0


class TestRefCleaned:
    def test_ref_cleaned_improves_noisy_refs(self):
        refs = ['1. यह धारा 227 का मामला #. है', '2. न्यायालय ने , आदेश दिया']
        hyps = ['यह धारा 227 का मामला है', 'न्यायालय ने, आदेश दिया']
        s = score_pairs(hyps, refs)
        assert s['ref_cleaned']['n'] == 2
        assert s['ref_cleaned']['bleu']['score'] > s['bleu']['score']
        assert s['ref_cleaned']['chrfpp']['score'] >= s['chrfpp']['score']
        assert s['bleu']['score'] < 100.0


class TestPairedCi:
    def test_significant_when_systems_differ(self):
        bad = [
            'completely unrelated words here and there',
            'another totally different sentence now',
            'nothing in common with the references at all',
            'this is not a legal translation whatsoever',
            'yet another random string of english tokens',
        ]
        d = compare_score_pairs(REFS, bad, REFS, n_resamples=200, seed=7)
        assert d['bleu']['delta'] > 80.0
        assert d['bleu']['significant'] is True
        assert d['chrfpp']['significant'] is True
        assert d['chrfpp']['ci_low'] > 0.0

    def test_not_significant_when_same_system(self):
        d = compare_score_pairs(REFS, REFS, REFS, n_resamples=200, seed=7)
        assert d['bleu']['delta'] == 0.0
        assert d['chrfpp']['delta'] == 0.0
        assert d['bleu']['significant'] is False
        assert d['chrfpp']['significant'] is False

    def test_reproducible_with_fixed_seed(self):
        bad = ['completely unrelated words here and there'] * 5
        a = compare_score_pairs(REFS, bad, REFS, n_resamples=100, seed=3)
        b = compare_score_pairs(REFS, bad, REFS, n_resamples=100, seed=3)
        assert a == b

    def test_ci_bounds_ordered(self):
        bad = ['completely unrelated words here and there'] * 5
        d = paired_ci(REFS, bad, REFS, 'chrfpp', n_resamples=100, seed=1)
        assert d['ci_low'] <= d['ci_high']
        assert d['delta'] > 0.0

    def test_length_mismatch_raises(self):
        try:
            paired_ci(['a'], ['a', 'b'], ['a', 'b'])
            raise AssertionError('expected ValueError')
        except ValueError:
            pass

    def test_unknown_metric_raises(self):
        try:
            paired_ci(REFS, REFS, REFS, 'ter', n_resamples=10)
            raise AssertionError('expected ValueError')
        except ValueError:
            pass

"""Hermetic tests for score-only helpers (no model load, tmp files only)."""

import json

from src.evaluation.zero_shot_nllb import compare_hyp_files, score_hyp_file


ROWS = [
    {
        'en_text': 'the appellant filed a writ petition before the high court',
        'hi_text': 'अपीलकर्ता ने उच्च न्यायालय के समक्ष रिट याचिका दायर की',
        'hyp_hi': 'अपीलकर्ता ने उच्च न्यायालय के समक्ष रिट याचिका दायर की',
    },
    {
        'en_text': 'the respondent contested the impugned order',
        'hi_text': 'प्रतिवादी ने विवादित आदेश का विरोध किया',
        'hyp_hi': 'प्रतिवादी ने विवादित आदेश का विरोध किया',
    },
    {
        'en_text': 'the court dismissed the appeal with costs',
        'hi_text': 'न्यायालय ने अपील खर्चे सहित खारिज कर दी',
        'hyp_hi': 'न्यायालय ने अपील खर्चे सहित खारिज कर दी',
    },
]


def _write(tmp_path, name, rows):
    p = tmp_path / name
    with open(p, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return p


class TestScoreHypFile:
    def test_respects_max_pairs(self, tmp_path):
        path = _write(tmp_path, 'a_I_test_hyps.jsonl', ROWS)
        full = score_hyp_file('I_test', path=path, max_pairs=None)
        sliced = score_hyp_file('I_test', path=path, max_pairs=2)
        assert full['n'] == 3
        assert sliced['n'] == 2
        assert sliced['bleu']['score'] == 100.0

    def test_emits_phase4_fields(self, tmp_path):
        path = _write(tmp_path, 'a_I_test_hyps.jsonl', ROWS)
        out = score_hyp_file('I_test', path=path)
        assert out['ter']['score'] == 0.0
        assert out['len_ratio'] == 1.0
        assert out['ref_cleaned']['n'] == 3
        assert out['entities']['n'] == 3
        assert len(out['hyp_file_sha256']) == 16
        assert 'confidence' in out

    def test_missing_file(self, tmp_path):
        out = score_hyp_file('I_test', path=tmp_path / 'nope.jsonl')
        assert out['n'] == 0
        assert out['error']


class TestCompareHypFiles:
    def test_aligns_by_source_and_scores_delta(self, tmp_path):
        good = _write(tmp_path, 'good_I_test_hyps.jsonl', ROWS)
        bad_rows = [dict(r, hyp_hi='बिल्कुल असंबंधित वाक्य') for r in reversed(ROWS)]
        bad = _write(tmp_path, 'bad_I_test_hyps.jsonl', bad_rows)
        out = compare_hyp_files('I_test', good, bad, n_resamples=100, seed=7)
        assert out['n'] == 3
        assert out['chrfpp']['delta'] > 0.0
        assert out['chrfpp']['significant'] is True
        assert 'bleu' in out

    def test_skips_rows_missing_from_other(self, tmp_path):
        good = _write(tmp_path, 'good_I_test_hyps.jsonl', ROWS)
        bad = _write(tmp_path, 'bad_I_test_hyps.jsonl', ROWS[:2])
        out = compare_hyp_files('I_test', good, bad, n_resamples=50, seed=1)
        assert out['n'] == 2

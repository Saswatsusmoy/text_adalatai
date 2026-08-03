"""Tests for the tokenizer benchmark helpers (pure functions)."""

import pytest
import sentencepiece as spm

from src.tokenizer import benchmark as bm


class TestNormDocId:
    def test_int_passthrough(self):
        assert bm._norm_doc_id(5) == 5

    def test_numeric_str(self):
        assert bm._norm_doc_id('21') == 21

    def test_non_numeric_returns_none(self):
        assert bm._norm_doc_id('milpac:1') is None
        assert bm._norm_doc_id(None) is None


class TestCountDevanagari:
    def test_count(self):
        # न्यायालय = 6 letters + 2 matras = 8 Devanagari codepoints
        assert bm.count_devanagari('न्यायालय') == 8
        assert bm.count_devanagari('abc Section') == 0
        assert bm.count_devanagari('') == 0


class TestEntropy:
    def test_uniform(self):
        assert bm._entropy([1, 1, 1, 1]) == pytest.approx(2.0)

    def test_single_symbol_zero(self):
        assert bm._entropy([10]) == 0.0

    def test_empty(self):
        assert bm._entropy([]) == 0.0

    def test_skips_zero_counts(self):
        assert bm._entropy([0, 4]) == 0.0


@pytest.fixture(scope='module')
def dev_spm(tmp_path_factory):
    root = tmp_path_factory.mktemp('devspm')
    corpus = root / 'c.txt'
    lines = [
        'न्यायालय अपीलकर्ता अनुच्छेद धारा नियम आदेश निर्णय अधिकारिता प्रतिवादी याचिका रिट संविधान भारत।',
        'the appellant filed a writ petition in the high court under article 227.',
        'Section 331 of the Act has been the subject of a series of proceedings.',
    ]
    corpus.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    prefix = root / 'm'
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(prefix), vocab_size=80)
    return spm.SentencePieceProcessor(model_file=str(prefix) + '.model')


class TestCountDevPieces:
    def test_counts_devanagari_pieces(self, dev_spm):
        n = bm.count_dev_pieces(dev_spm)
        assert n >= 1  # न्यायालय / अनुच्छेद pieces must contain Devanagari
        total = dev_spm.GetPieceSize()
        assert n <= total

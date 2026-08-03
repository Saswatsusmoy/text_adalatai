"""Tests for the deep-dive analysis helpers."""

import pytest
import sentencepiece as spm

from src.tokenizer import deep_dive as dd


@pytest.fixture(scope='module')
def dev_spm(tmp_path_factory):
    root = tmp_path_factory.mktemp('ddspm')
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


class TestBpeMergePriorities:
    def test_returns_token_records(self, dev_spm):
        tokens = dd.analyze_bpe_merge_priorities(dev_spm, 'न्यायालय Section')
        assert len(tokens) >= 1
        for t in tokens:
            assert 'id' in t
            assert 'text' in t
            assert 'length' in t
            assert 'is_byte_range' in t
            assert 'is_devanagari' in t

    def test_detects_devanagari_tokens(self, dev_spm):
        tokens = dd.analyze_bpe_merge_priorities(dev_spm, 'न्यायालय')
        assert any(t['is_devanagari'] for t in tokens)

    def test_empty_text(self, dev_spm):
        assert dd.analyze_bpe_merge_priorities(dev_spm, '') == []


class TestTheoreticalBounds:
    def test_returns_expected_keys(self):
        bounds = dd.compute_theoretical_bounds()
        assert isinstance(bounds, dict)
        # keys that feed the report narrative
        assert 'cl100k_actual_tokens_per_devanagari_char' in bounds
        assert 'nllb_actual_tokens_per_devanagari_char' in bounds

"""Tests for the matrix bench harness (probe, UNK rate, held-out loader)."""

import json

import pytest
import sentencepiece as spm

from src.tokenizer import bench_matrix as bm
from src.tokenizer.benchmark import HELD_OUT_DOCS


@pytest.fixture(scope='module')
def tiny_spm(tmp_path_factory):
    """Train a tiny SentencePiece model on synthetic legal text."""
    root = tmp_path_factory.mktemp('spm')
    corpus = root / 'corpus.txt'
    lines = [
        'न्यायालय अपीलकर्ता अनुच्छेद धारा नियम आदेश निर्णय अधिकारिता प्रतिवादी याचिका रिट संविधान भारत।',
        'the appellant filed a writ petition in the high court under article 227.',
        'Section 331 of the Act has been the subject of a series of proceedings.',
        'The respondent impugned the order of the revisional court before the appellate court.',
    ]
    corpus.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    prefix = root / 'tiny'
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(prefix),
        vocab_size=80,
        character_coverage=1.0,
        model_type='unigram',
    )
    return spm.SentencePieceProcessor(model_file=str(prefix) + '.model')


class TestProbe:
    def test_single_piece_hit(self, tiny_spm):
        # 'the' appears verbatim and stays one piece on the tiny model
        result = bm._probe_single_piece(tiny_spm, ('the',))
        assert result['hits'] == 1
        assert result['total'] == 1
        assert result['rate'] == 1.0
        assert result['per_term']['the'] == 1

    def test_probe_rate_zero_for_unknown(self, tiny_spm):
        # A long invented word should not be a single piece
        result = bm._probe_single_piece(tiny_spm, ('xxyzzqwxzz',))
        assert result['hits'] == 0
        assert result['rate'] == 0.0

    def test_probe_rate_fraction(self, tiny_spm):
        result = bm._probe_single_piece(tiny_spm, ('the', 'zznotawordzz'))
        assert result['hits'] == 1
        assert result['total'] == 2
        assert result['rate'] == 0.5

    def test_per_term_counts_multi_piece(self, tiny_spm):
        # 'न्यायालय' fragments on the tiny model -> per_term shows the true count
        result = bm._probe_single_piece(tiny_spm, ('न्यायालय',))
        assert result['per_term']['न्यायालय'] > 1
        assert result['hits'] == 0


class TestUnkRate:
    def test_zero_unks(self, tiny_spm):
        ids = tiny_spm.encode(['न्यायालय', 'Section'])
        assert bm._unk_rate(tiny_spm, [ids]) == 0.0

    def test_all_unks(self, tiny_spm):
        # Force unknown tokens via a bytes string with no coverage
        unk_id = tiny_spm.unk_id()
        ids = [[unk_id, unk_id]]
        assert bm._unk_rate(tiny_spm, ids) == 1.0

    def test_empty_input_no_divzero(self, tiny_spm):
        assert bm._unk_rate(tiny_spm, []) == 0.0


class TestDiscoverModels:
    def test_returns_sorted_paths(self, tmp_path, monkeypatch):
        d = tmp_path / 'models'
        d.mkdir()
        (d / 'sentencepiece_legal_v2_joint_bpe_41000.model').touch()
        (d / 'sentencepiece_legal_v2_joint_unigram_16000.model').touch()
        (d / 'other.model').touch()  # must be ignored
        monkeypatch.setattr(bm, 'MODEL_DIR', d)
        found = bm.discover_matrix_models()
        assert len(found) == 2
        assert all('legal_v2' in f.name for f in found)

    def test_no_models_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bm, 'MODEL_DIR', tmp_path)
        assert bm.discover_matrix_models() == []


class TestLoadHeldOut:
    def test_only_held_out_docs(self, tmp_path, monkeypatch):
        f = tmp_path / 'all.jsonl'
        rows = [
            {'doc_id': 1, 'en_text': 'a', 'hi_text': 'अ'},
            {'doc_id': 8, 'en_text': 'b', 'hi_text': 'ब'},
            {'doc_id': 21, 'en_text': 'c', 'hi_text': 'स'},
            {'doc_id': 2, 'en_text': 'd', 'hi_text': 'द'},
        ]
        f.write_text(
            '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8'
        )
        monkeypatch.setattr(bm, 'ALIGNED_PATH', f)
        loaded = bm.load_held_out()
        docs = {r['doc_id'] for r in loaded}
        assert docs == set(HELD_OUT_DOCS) & {1, 2, 8, 21}
        assert 2 not in docs  # doc 2 is train, excluded

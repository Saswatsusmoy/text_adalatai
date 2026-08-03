"""Tests for the full-joint SPM trainer helpers."""

from src.tokenizer import train_full_joint as tfj


class TestAttemptsFor:
    def test_unigram_ladder(self):
        attempts = tfj.attempts_for('unigram')
        assert attempts[0][0] == 'full'
        assert attempts[0][1] == 'sentencepiece_legal_v2_joint_full_{vocab}'
        assert len(attempts) == 3

    def test_bpe_has_bpe_infix(self):
        attempts = tfj.attempts_for('bpe')
        assert attempts[0][0] == 'full'
        assert 'bpe' in attempts[0][1]

    def test_default_is_unigram(self):
        assert tfj.attempts_for('unigram') == tfj.attempts_for('unigram')
        # bpe ladder distinct from unigram ladder
        assert [a[1] for a in tfj.attempts_for('bpe')] != [
            a[1] for a in tfj.attempts_for('unigram')
        ]

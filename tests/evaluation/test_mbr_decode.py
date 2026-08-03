"""Tests for the MBR pick objective (pure function; no model load)."""

import pytest
import torch

from src.evaluation.mbr_decode import mbr_pick, translate_batch_mbr


def test_mbr_pick_returns_consensus_candidate():
    consensus = 'the appellant filed a writ petition'
    outlier = 'unrelated sentence with different tokens entirely'
    assert mbr_pick([consensus, consensus, consensus, outlier]) == consensus


def test_mbr_pick_single_candidate_is_identity():
    only = 'only one hypothesis'
    assert mbr_pick([only]) == only


def test_mbr_pick_empty_raises():
    with pytest.raises(ValueError):
        mbr_pick([])


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        n = len(texts)
        return {'input_ids': torch.full((n, 3), 7, dtype=torch.long)}

    def batch_decode(self, ids, skip_special_tokens=True):
        return [' '.join(f'hyp_{int(i)}' for i in row) for row in ids]


class _FakeModel:
    def generate(self, **kwargs):
        n = kwargs['input_ids'].shape[0]
        n_samples = kwargs['num_return_sequences']
        return torch.randint(0, 10000, (n * n_samples, 3))


def _fake_args():
    return {
        'texts': ['the appellant filed a writ petition'],
        'tokenizer': _FakeTokenizer(),
        'model': _FakeModel(),
        'device': 'cpu',
        'forced_bos_token_id': 3,
    }


class TestSeedReproducibility:
    def test_same_seed_reproduces_samples(self):
        first = translate_batch_mbr(n_samples=8, seed=12345, **_fake_args())
        second = translate_batch_mbr(n_samples=8, seed=12345, **_fake_args())
        assert first == second

    def test_different_seed_draws_different_samples(self):
        first = translate_batch_mbr(n_samples=8, seed=12345, **_fake_args())
        third = translate_batch_mbr(n_samples=8, seed=99999, **_fake_args())
        assert first != third

    def test_direct_call_seeds_for_reproducibility(self):
        a = translate_batch_mbr(n_samples=8, seed=7, **_fake_args())
        b = translate_batch_mbr(n_samples=8, seed=7, **_fake_args())
        assert a == b

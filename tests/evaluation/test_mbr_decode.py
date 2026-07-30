"""Unit + integration tests for MBR decoding (no NLLB weights loaded)."""

from unittest.mock import MagicMock

import pytest

from src.evaluation.mbr_decode import mbr_pick, translate_batch_mbr


def test_mbr_pick_selects_consensus_over_outlier():
    consensus = 'the appellant filed a writ petition'
    candidates = [
        consensus,
        consensus,
        consensus,
        'completely unrelated legal jargon banana',
        'xyzzy plugh frobnicate',
    ]
    assert mbr_pick(candidates, utility='chrfpp') == consensus


def test_mbr_pick_single_candidate_returns_it():
    only = 'लो अकेला अनुवाद'
    assert mbr_pick([only]) == only


def test_mbr_pick_empty_raises():
    with pytest.raises(ValueError):
        mbr_pick([])


def test_mbr_pick_unknown_utility_raises():
    with pytest.raises(ValueError):
        mbr_pick(['a', 'b'], utility='bleu')


def test_mbr_pick_two_identical_returns_first_stably():
    assert mbr_pick(['same text here', 'same text here']) == 'same text here'


def test_mbr_pick_chrf_word_order_zero_also_works():
    consensus = 'न्यायालय ने अपीलकर्ता की याचिका खारिज कर दी'
    candidates = [consensus, consensus, 'completely different sentence xyz']
    assert mbr_pick(candidates, utility='chrf') == consensus


def test_mbr_pick_devanagari_consensus():
    consensus = 'भारत के संविधान के अनुच्छेद 227 के तहत'
    candidates = [
        consensus,
        consensus,
        'भारत के संविधान अनुच्छेद 227 के अधीन',
        'completely unrelated english text',
    ]
    picked = mbr_pick(candidates, utility='chrfpp')
    assert picked in {consensus, 'भारत के संविधान अनुच्छेद 227 के अधीन'}
    assert picked != 'completely unrelated english text'


class _FakeTokenizer:
    """Records generate output and rehydrates deterministic candidates."""

    pad_token_id = 0

    def __init__(self, per_input_samples):
        self._per_input_samples = per_input_samples

    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        import torch

        return {'input_ids': torch.zeros((len(texts), 1), dtype=torch.long)}

    def batch_decode(self, tokens, skip_special_tokens):
        flat = []
        for samples in self._per_input_samples:
            flat.extend(samples)
        assert len(tokens) == len(flat), 'fake generate returned wrong shape'
        return flat


def _fake_model(per_input_samples):
    import torch

    total = sum(len(s) for s in per_input_samples)
    fake_out = torch.zeros((total, 1), dtype=torch.long)
    model = MagicMock()
    model.generate.return_value = fake_out
    return model


def test_translate_batch_mbr_picks_per_input_consensus():
    """E2E through sample_candidates + mbr_pick with a mocked model."""
    per_input = [
        [
            'the court dismissed the appeal',
            'the court dismissed the appeal',
            'the court dismissed the appeal',
            'unrelated noise sentence one',
        ],
        [
            'भारत के संविधान के अनुच्छेद 227',
            'भारत के संविधान के अनुच्छेद 227',
            'completely different english line',
            'completely different english line',
        ],
    ]
    tok = _FakeTokenizer(per_input)
    model = _fake_model(per_input)
    picks = translate_batch_mbr(
        ['src A', 'src B'],
        tok,
        model,
        device='cpu',
        forced_bos_token_id=1,
        n_samples=4,
        utility='chrfpp',
    )
    assert picks[0] == 'the court dismissed the appeal'
    assert picks[1] in {'भारत के संविधान के अनुच्छेद 227', 'completely different english line'}
    call_kwargs = model.generate.call_args.kwargs
    assert call_kwargs['do_sample'] is True
    assert call_kwargs['num_beams'] == 1
    assert call_kwargs['num_return_sequences'] == 4
    assert call_kwargs['forced_bos_token_id'] == 1


def test_translate_batch_mbr_empty_texts_short_circuits():
    tok = _FakeTokenizer([])
    model = MagicMock()
    picks = translate_batch_mbr([], tok, model, device='cpu', forced_bos_token_id=1, n_samples=4)
    assert picks == []
    model.generate.assert_not_called()


def test_translate_batch_mbr_size_mismatch_raises():
    """If generate returns wrong shape, we fail loudly not silently."""
    import torch

    tok = MagicMock()
    tok.__call__ = MagicMock(return_value={'input_ids': torch.zeros((1, 1), dtype=torch.long)})
    tok.batch_decode = MagicMock(return_value=['only one sample'])
    model = MagicMock()
    model.generate.return_value = torch.zeros((1, 1), dtype=torch.long)
    with pytest.raises(RuntimeError, match='expected'):
        translate_batch_mbr(['src A'], tok, model, device='cpu', forced_bos_token_id=1, n_samples=4)

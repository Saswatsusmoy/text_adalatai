"""Tests for NLLB collate / Tensor Core padding."""

from src.training.nllb_data import collate_nllb


def test_collate_pad_to_multiple_of_8():
    feats = [
        {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1], 'labels': [4, 5]},
        {'input_ids': [1, 2], 'attention_mask': [1, 1], 'labels': [4]},
    ]
    batch = collate_nllb(feats, pad_token_id=0, pad_to_multiple_of=8)
    assert batch['input_ids'].shape[1] == 8
    assert batch['labels'].shape[1] == 8
    assert batch['attention_mask'][0].tolist()[:3] == [1, 1, 1]
    assert batch['attention_mask'][0].tolist()[3:] == [0] * 5
    assert batch['labels'][1].tolist() == [4] + [-100] * 7


def test_collate_pad_to_fixed_masks_loss():
    feats = [
        {'input_ids': [1, 2], 'attention_mask': [1, 1], 'labels': [9]},
    ]
    batch = collate_nllb(
        feats,
        pad_token_id=0,
        pad_to_fixed=(4, 4),
    )
    assert batch['input_ids'].shape == (1, 4)
    assert batch['labels'].shape == (1, 4)
    assert batch['labels'][0].tolist() == [9, -100, -100, -100]
    assert batch['attention_mask'][0].sum().item() == 2

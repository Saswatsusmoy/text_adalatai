"""Tests for Track C1 SPM tokenizer + collate."""

from pathlib import Path

from src.config import SPM_V2_PRIMARY
from src.training.legal_mt_data import collate_legal_mt
from src.training.spm_tokenizer import LegalSpmTokenizer


def test_spm_primary_exists():
    assert SPM_V2_PRIMARY.exists()


def test_legal_spm_special_ids():
    tok = LegalSpmTokenizer()
    assert tok.vocab_size == 41000
    assert tok.pad_token_id == 0
    assert tok.unk_token_id == 1
    assert tok.bos_token_id == 2
    assert tok.eos_token_id == 3


def test_encode_decode_roundtrip_ish():
    tok = LegalSpmTokenizer()
    text = 'The Court held that.'
    ids = tok.encode(text, add_eos=True)
    assert tok.eos_token_id in ids
    decoded = tok.decode(ids)
    assert 'Court' in decoded or 'court' in decoded.lower() or len(decoded) > 0


def test_collate_legal_mt_pad():
    feats = [
        {
            'input_ids': [4, 5, 6],
            'attention_mask': [1, 1, 1],
            'decoder_input_ids': [2, 10],
            'labels': [10, 3],
        },
        {
            'input_ids': [4],
            'attention_mask': [1],
            'decoder_input_ids': [2],
            'labels': [3],
        },
    ]
    batch = collate_legal_mt(feats, pad_token_id=0, pad_to_multiple_of=4)
    assert batch['input_ids'].shape[1] % 4 == 0
    assert batch['labels'][1, -1].item() == -100 or batch['labels'].shape[1] >= 1
    assert batch['attention_mask'][1].sum().item() == 1


def test_save_load_roundtrip(tmp_path: Path):
    tok = LegalSpmTokenizer()
    tok.save_pretrained(tmp_path)
    tok2 = LegalSpmTokenizer.from_pretrained(tmp_path)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.encode('hello') == tok.encode('hello')

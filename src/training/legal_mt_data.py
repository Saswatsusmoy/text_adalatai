"""Bitext dataset for Track C1 (LegalSpmTokenizer)."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.evaluation.eval_sets import load_jsonl
from src.training.nllb_data import _ceil_mult
from src.training.spm_tokenizer import LegalSpmTokenizer


class LegalMtJsonlDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: LegalSpmTokenizer,
        max_source_length: int = 256,
        max_target_length: int = 256,
        max_pairs: int | None = None,
    ):
        self.rows = load_jsonl(Path(path))
        if max_pairs is not None:
            self.rows = self.rows[:max_pairs]
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        src = self.tokenizer.encode(
            row['en_text'],
            add_bos=False,
            add_eos=False,
            max_length=self.max_source_length,
        )
        tgt = self.tokenizer.encode(
            row['hi_text'],
            add_bos=True,
            add_eos=True,
            max_length=self.max_target_length,
        )
        return {
            'input_ids': src,
            'attention_mask': [1] * len(src),
            'decoder_input_ids': tgt[:-1],
            'labels': tgt[1:],
        }


def collate_legal_mt(
    features: list[dict],
    pad_token_id: int,
    label_pad: int = -100,
    pad_to_multiple_of: int = 1,
    pad_to_fixed: tuple[int, int] | None = None,
) -> dict:
    if pad_to_fixed is not None:
        max_src, max_tgt = pad_to_fixed
    else:
        max_src = _ceil_mult(max(len(f['input_ids']) for f in features), pad_to_multiple_of)
        max_tgt = _ceil_mult(
            max(len(f['decoder_input_ids']) for f in features),
            pad_to_multiple_of,
        )

    b = len(features)
    input_ids = torch.full((b, max_src), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((b, max_src), dtype=torch.long)
    decoder_input_ids = torch.full((b, max_tgt), pad_token_id, dtype=torch.long)
    labels = torch.full((b, max_tgt), label_pad, dtype=torch.long)
    for i, f in enumerate(features):
        s = f['input_ids'][:max_src]
        d = f['decoder_input_ids'][:max_tgt]
        lab = f['labels'][:max_tgt]
        ns, nd = len(s), len(d)
        input_ids[i, :ns] = torch.as_tensor(s, dtype=torch.long)
        attention_mask[i, :ns] = 1
        decoder_input_ids[i, :nd] = torch.as_tensor(d, dtype=torch.long)
        labels[i, : len(lab)] = torch.as_tensor(lab, dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'decoder_input_ids': decoder_input_ids,
        'labels': labels,
    }

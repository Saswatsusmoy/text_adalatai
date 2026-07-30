"""Bitext dataset using LegalSpmTokenizer (Track C1)."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.evaluation.eval_sets import load_jsonl
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
        # encoder: plain tokens; decoder labels: bos + tokens + eos (teacher forcing)
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
        # decoder input = tgt[:-1]; labels = tgt[1:] (standard)
        decoder_input_ids = tgt[:-1]
        labels = tgt[1:]
        return {
            'input_ids': src,
            'attention_mask': [1] * len(src),
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
        }


def collate_legal_mt(
    features: list[dict],
    pad_token_id: int,
    label_pad: int = -100,
    pad_to_multiple_of: int = 1,
    pad_to_fixed: tuple[int, int] | None = None,
) -> dict:
    def ceil_mult(n: int, m: int) -> int:
        if m <= 1:
            return n
        return ((n + m - 1) // m) * m

    if pad_to_fixed is not None:
        max_src, max_tgt = pad_to_fixed
    else:
        max_src = max(len(f['input_ids']) for f in features)
        max_tgt = max(len(f['decoder_input_ids']) for f in features)
        max_src = ceil_mult(max_src, pad_to_multiple_of)
        max_tgt = ceil_mult(max_tgt, pad_to_multiple_of)

    input_ids, attention_mask, decoder_input_ids, labels = [], [], [], []
    for f in features:
        s = f['input_ids'][:max_src]
        am = f['attention_mask'][:max_src]
        d = f['decoder_input_ids'][:max_tgt]
        lab = f['labels'][:max_tgt]
        ps = max_src - len(s)
        input_ids.append(s + [pad_token_id] * ps)
        attention_mask.append(am + [0] * ps)
        pd = max_tgt - len(d)
        decoder_input_ids.append(d + [pad_token_id] * pd)
        labels.append(lab + [label_pad] * pd)
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }

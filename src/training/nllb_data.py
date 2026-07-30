"""NLLB EN-HI bitext dataset + collate."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.evaluation.eval_sets import load_jsonl


def _ceil_mult(n: int, mult: int) -> int:
    return n if mult <= 1 else ((n + mult - 1) // mult) * mult


class NllbJsonlDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        src_lang: str = 'eng_Latn',
        tgt_lang: str = 'hin_Deva',
        max_source_length: int = 256,
        max_target_length: int = 256,
        max_pairs: int | None = None,
    ):
        self.rows = load_jsonl(Path(path))
        if max_pairs is not None:
            self.rows = self.rows[:max_pairs]
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        if hasattr(tokenizer, 'src_lang'):
            tokenizer.src_lang = src_lang
        if hasattr(tokenizer, 'tgt_lang'):
            tokenizer.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        mi = self.tokenizer(
            row['en_text'],
            text_target=row['hi_text'],
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
        )
        if 'labels' not in mi:
            with self.tokenizer.as_target_tokenizer():
                lab = self.tokenizer(
                    row['hi_text'],
                    max_length=self.max_target_length,
                    truncation=True,
                    padding=False,
                )
            mi['labels'] = lab['input_ids']
        if len(mi['labels']) > self.max_target_length:
            mi['labels'] = mi['labels'][: self.max_target_length]
        return {k: mi[k] for k in ('input_ids', 'attention_mask', 'labels')}


def collate_nllb(
    features: list[dict],
    pad_token_id: int,
    label_pad: int = -100,
    pad_to_multiple_of: int = 1,
    pad_to_fixed: tuple[int, int] | None = None,
) -> dict:
    if pad_to_fixed is not None:
        max_len, max_lab = pad_to_fixed
    else:
        max_len = _ceil_mult(max(len(f['input_ids']) for f in features), pad_to_multiple_of)
        max_lab = _ceil_mult(max(len(f['labels']) for f in features), pad_to_multiple_of)

    b = len(features)
    input_ids = torch.full((b, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((b, max_len), dtype=torch.long)
    labels = torch.full((b, max_lab), label_pad, dtype=torch.long)
    for i, f in enumerate(features):
        ids = f['input_ids'][:max_len]
        am = f['attention_mask'][:max_len]
        lab = f['labels'][:max_lab]
        n, m = len(ids), len(lab)
        input_ids[i, :n] = torch.as_tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = torch.as_tensor(am, dtype=torch.long)
        labels[i, :m] = torch.as_tensor(lab, dtype=torch.long)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

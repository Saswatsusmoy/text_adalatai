"""Tokenize EN-HI pairs for NLLB seq2seq training."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.evaluation.eval_sets import load_jsonl


def _ceil_mult(n: int, mult: int) -> int:
    if mult <= 1:
        return n
    return ((n + mult - 1) // mult) * mult


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

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        src = row['en_text']
        tgt = row['hi_text']
        if hasattr(self.tokenizer, 'src_lang'):
            self.tokenizer.src_lang = self.src_lang
        model_inputs = self.tokenizer(
            src,
            text_target=tgt,
            max_length=self.max_source_length,
            truncation=True,
            padding=False,
        )
        if 'labels' not in model_inputs:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    tgt,
                    max_length=self.max_target_length,
                    truncation=True,
                    padding=False,
                )
            model_inputs['labels'] = labels['input_ids']
        if len(model_inputs['labels']) > self.max_target_length:
            model_inputs['labels'] = model_inputs['labels'][: self.max_target_length]
        return {k: model_inputs[k] for k in ('input_ids', 'attention_mask', 'labels')}


def collate_nllb(
    features: list[dict],
    pad_token_id: int,
    label_pad: int = -100,
    pad_to_multiple_of: int = 1,
    pad_to_fixed: tuple[int, int] | None = None,
) -> dict:
    """
    Pad batch for NLLB.

    pad_to_multiple_of: round lengths up (8 keeps Hopper Tensor Cores happy).
    pad_to_fixed: (src_len, tgt_len) for static shapes (stable torch.compile / CUDA graphs).
    Pad positions on labels use label_pad (-100) so loss is unchanged.
    """
    if pad_to_fixed is not None:
        max_len, max_lab = pad_to_fixed
    else:
        max_len = max(len(f['input_ids']) for f in features)
        max_lab = max(len(f['labels']) for f in features)
        max_len = _ceil_mult(max_len, pad_to_multiple_of)
        max_lab = _ceil_mult(max_lab, pad_to_multiple_of)

    input_ids = []
    attention_mask = []
    labels = []
    for f in features:
        ids = f['input_ids'][:max_len]
        am = f['attention_mask'][:max_len]
        lab = f['labels'][:max_lab]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append(am + [0] * pad_len)
        lp = max_lab - len(lab)
        labels.append(lab + [label_pad] * lp)
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long),
    }

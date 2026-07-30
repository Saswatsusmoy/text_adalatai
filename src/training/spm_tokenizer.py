"""HF-friendly wrapper around Track C SPM_V2_PRIMARY (joint Unigram 41k)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import sentencepiece as spm
import torch

from src.config import SPM_V2_PRIMARY


class LegalSpmTokenizer:
    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path) if model_path else SPM_V2_PRIMARY
        if not path.exists():
            raise FileNotFoundError(path)
        self.model_path = path
        self.sp = spm.SentencePieceProcessor(model_file=str(path))
        self.pad_token_id = int(self.sp.pad_id()) if self.sp.pad_id() >= 0 else 0
        self.unk_token_id = int(self.sp.unk_id())
        self.bos_token_id = int(self.sp.bos_id()) if self.sp.bos_id() >= 0 else 2
        self.eos_token_id = int(self.sp.eos_id()) if self.sp.eos_id() >= 0 else 3
        self.vocab_size = int(self.sp.get_piece_size())
        self.pad_token = self.sp.id_to_piece(self.pad_token_id)
        self.unk_token = self.sp.id_to_piece(self.unk_token_id)
        self.bos_token = self.sp.id_to_piece(self.bos_token_id)
        self.eos_token = self.sp.id_to_piece(self.eos_token_id)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
        truncation: bool = True,
    ) -> list[int]:
        ids = list(self.sp.encode(text, out_type=int))
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        if max_length is not None and truncation and len(ids) > max_length:
            if add_eos and max_length >= 1:
                ids = ids[: max_length - 1] + [self.eos_token_id]
            else:
                ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
            ids = [i for i in ids if i not in special]
        return self.sp.decode(ids)

    def batch_decode(
        self,
        sequences: list[list[int]] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in sequences]

    def save_pretrained(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        dest = path / 'spiece.model'
        if not dest.exists() or dest.resolve() != self.model_path.resolve():
            shutil.copy2(self.model_path, dest)
        meta = {
            'model_file': 'spiece.model',
            'vocab_size': self.vocab_size,
            'pad_token_id': self.pad_token_id,
            'unk_token_id': self.unk_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'source': str(self.model_path),
        }
        (path / 'tokenizer_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    @classmethod
    def from_pretrained(cls, path: str | Path) -> LegalSpmTokenizer:
        path = Path(path)
        model = path / 'spiece.model'
        return cls(model if model.exists() else path)

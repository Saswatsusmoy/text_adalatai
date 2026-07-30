"""Shared train/eval helpers (seed, I/O, device, autocast)."""

from __future__ import annotations

import gc
import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_json(path: Path, obj, indent: int = 2):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding='utf-8')


def count_nonempty_lines(path: Path | str) -> int:
    n = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def is_cuda(device: str) -> bool:
    return str(device).startswith('cuda')


def move_batch(batch: dict, device: str) -> dict:
    nb = is_cuda(device)
    return {k: v.to(device, non_blocking=nb) for k, v in batch.items()}


def autocast_ctx(device: str, dtype: torch.dtype):
    if is_cuda(device) and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type='cuda', dtype=dtype)
    return nullcontext()


def empty_device_cache(device: str):
    gc.collect()
    if device == 'mps' and hasattr(torch, 'mps'):
        torch.mps.empty_cache()
    elif is_cuda(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()


def loss_value(loss: torch.Tensor) -> float:
    return float(loss.detach().item())

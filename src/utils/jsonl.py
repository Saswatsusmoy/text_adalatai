"""Shared JSONL load/write used by preprocess, tokenizer, train, and eval."""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path | str, rows: list[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def append_jsonl(path: Path | str, row: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

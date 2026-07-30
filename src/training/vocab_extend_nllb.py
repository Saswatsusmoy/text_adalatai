"""
Track C1c: extend NLLB tokenizer+embeddings with legal SPM pieces that NLLB fragments.

Select top-K SPM pieces by frequency on Stage A sample that are not exact NLLB
tokens; add them; resize embeddings; init new rows as mean of NLLB subword embeds.

Usage:
  PYTHONPATH=. python -m src.training.vocab_extend_nllb \\
      --top-k 8000 --out data/models/nllb600_c1c_sp_ext
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import SPM_V2_PRIMARY


def load_spm(path: Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor(model_file=str(path))
    return sp


def piece_in_tokenizer(tok, piece: str) -> bool:
    tid = tok.convert_tokens_to_ids(piece)
    if tid is None or tid == tok.unk_token_id:
        return False
    return tok.convert_ids_to_tokens(tid) == piece


def collect_candidate_freq(
    sp: spm.SentencePieceProcessor,
    tok,
    jsonl_path: Path,
    max_lines: int = 20000,
) -> Counter:
    specials = {sp.id_to_piece(i) for i in range(min(4, sp.get_piece_size()))}
    freq: Counter = Counter()
    n = 0
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            if n >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for text in (row.get('en_text') or '', row.get('hi_text') or ''):
                if not text:
                    continue
                for pid in sp.encode(text, out_type=int):
                    piece = sp.id_to_piece(pid)
                    if piece in specials:
                        continue
                    if not piece_in_tokenizer(tok, piece):
                        freq[piece] += 1
            n += 1
    return freq


def select_tokens(freq: Counter, top_k: int, min_len: int = 2) -> list[str]:
    # skip pure quotes / single-char junk often from OCR
    skip_chars = set('“”‘’"\'`')
    chosen = []
    for piece, _c in freq.most_common():
        core = piece.replace('▁', '')
        if len(core) < min_len and piece.startswith('▁'):
            continue
        if core and all(ch in skip_chars for ch in core):
            continue
        if piece in skip_chars:
            continue
        chosen.append(piece)
        if len(chosen) >= top_k:
            break
    return chosen


@torch.no_grad()
def init_new_embeddings(model, tok, new_tokens: list[str], old_len: int):
    """Mean-init each new token from NLLB subword embeddings of its surface form."""
    emb = model.get_input_embeddings()
    weight = emb.weight
    device = weight.device
    dtype = weight.dtype
    for piece in new_tokens:
        tid = tok.convert_tokens_to_ids(piece)
        if tid is None or tid < old_len:
            continue
        # surface for subwording: strip SPM underline
        surface = piece[1:] if piece.startswith('▁') else piece
        if not surface:
            surface = piece
        # encode with original behavior: temporarily the new token exists; use
        # tokenize of surface which may still hit new token if exact match.
        # Prefer encoding without using the added token: break into chars via
        # existing pieces only by using tok.backend if needed.
        # Simple approach: encode surface; if only self, use unk neighbors.
        ids = tok.encode(surface, add_special_tokens=False)
        ids = [i for i in ids if i != tid and i < old_len]
        if not ids:
            # random small noise around mean of all embeds (avoid zeros)
            weight[tid] = weight[:old_len].mean(dim=0)
            continue
        vec = weight[ids].mean(dim=0)
        weight[tid] = vec.to(dtype=dtype)
    # if tied output embeddings, HF resize keeps them tied
    model.tie_weights()


def build_extended(
    base_model_id: str,
    spm_path: Path,
    bitext_path: Path,
    out_dir: Path,
    top_k: int = 8000,
    max_lines: int = 20000,
    torch_dtype: str = 'float32',
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sp = load_spm(spm_path)
    tok = AutoTokenizer.from_pretrained(base_model_id)
    old_len = len(tok)

    if verbose:
        print(f'base={base_model_id} old_vocab={old_len}')
        print(f'spm={spm_path} spm_size={sp.get_piece_size()}')
        print(f'scanning {bitext_path} max_lines={max_lines}')

    freq = collect_candidate_freq(sp, tok, bitext_path, max_lines=max_lines)
    candidates = select_tokens(freq, top_k=top_k)
    if verbose:
        print(f'candidates_missing={len(freq)} selected={len(candidates)}')
        print('top10:', [(t, freq[t]) for t in candidates[:10]])

    n_added = tok.add_tokens(candidates)
    new_len = len(tok)
    if verbose:
        print(f'add_tokens returned={n_added} new_vocab={new_len}')

    dtype = torch.float32
    if torch_dtype in ('float16', 'fp16'):
        dtype = torch.float16
    elif torch_dtype in ('bfloat16', 'bf16'):
        dtype = torch.bfloat16

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, dtype=dtype)
    model.resize_token_embeddings(new_len)
    init_new_embeddings(model, tok, candidates, old_len=old_len)

    tok.save_pretrained(out_dir)
    model.save_pretrained(out_dir)

    # also copy SPM freeze for reference
    import shutil

    shutil.copy2(spm_path, out_dir / 'source_spm.model')

    manifest = {
        'base_model_id': base_model_id,
        'spm_path': str(spm_path),
        'bitext_path': str(bitext_path),
        'max_lines': max_lines,
        'top_k_requested': top_k,
        'old_vocab': old_len,
        'new_vocab': new_len,
        'n_added': n_added,
        'selected_tokens': candidates,
        'token_freq': {t: freq[t] for t in candidates},
        'output': str(out_dir),
    }
    (out_dir / 'c1c_vocab_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    # lighter index without full freq dump for quick inspect
    slim = {k: v for k, v in manifest.items() if k not in ('selected_tokens', 'token_freq')}
    slim['n_selected'] = len(candidates)
    slim['sample_tokens'] = candidates[:50]
    (out_dir / 'c1c_summary.json').write_text(
        json.dumps(slim, indent=2, ensure_ascii=False), encoding='utf-8',
    )
    if verbose:
        print(f'wrote {out_dir} (+{n_added} tokens)')
    return manifest


def main():
    parser = argparse.ArgumentParser(description='C1c: extend NLLB vocab with legal SPM pieces')
    parser.add_argument('--base', default='facebook/nllb-200-distilled-600M')
    parser.add_argument('--spm', default=str(SPM_V2_PRIMARY))
    parser.add_argument(
        '--bitext',
        default='data/external/parallel/subsamples/stage_a_A1_n80000.jsonl',
    )
    parser.add_argument('--out', default='data/models/nllb600_c1c_sp_ext')
    parser.add_argument('--top-k', type=int, default=8000)
    parser.add_argument('--max-lines', type=int, default=20000)
    parser.add_argument('--dtype', default='float32', help='float32|float16|bfloat16 for save')
    args = parser.parse_args()
    build_extended(
        base_model_id=args.base,
        spm_path=Path(args.spm),
        bitext_path=Path(args.bitext),
        out_dir=Path(args.out),
        top_k=args.top_k,
        max_lines=args.max_lines,
        torch_dtype=args.dtype,
    )


if __name__ == '__main__':
    main()

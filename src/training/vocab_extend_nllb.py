"""C1c v1: bulk NLLB vocab extend from legal SPM pieces (ablation)."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import SPM_V2_PRIMARY


def load_spm(path: Path) -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=str(path))


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
                    if piece not in specials and not piece_in_tokenizer(tok, piece):
                        freq[piece] += 1
            n += 1
    return freq


def select_tokens(freq: Counter, top_k: int, min_len: int = 2) -> list[str]:
    skip_chars = set('“”‘’"\'`')
    chosen = []
    for piece, _c in freq.most_common():
        core = piece.replace('▁', '')
        if len(core) < min_len and piece.startswith('▁'):
            continue
        if (core and all(ch in skip_chars for ch in core)) or piece in skip_chars:
            continue
        chosen.append(piece)
        if len(chosen) >= top_k:
            break
    return chosen


@torch.no_grad()
def init_new_embeddings(model, tok, new_tokens: list[str], old_len: int):
    weight = model.get_input_embeddings().weight
    dtype = weight.dtype
    for piece in new_tokens:
        tid = tok.convert_tokens_to_ids(piece)
        if tid is None or tid < old_len:
            continue
        surface = piece[1:] if piece.startswith('▁') else piece or piece
        ids = [i for i in tok.encode(surface, add_special_tokens=False) if i != tid and i < old_len]
        weight[tid] = (
            weight[ids].mean(dim=0).to(dtype=dtype) if ids else weight[:old_len].mean(dim=0)
        )
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
    sp, tok = load_spm(spm_path), AutoTokenizer.from_pretrained(base_model_id)
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

    dtype = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }.get(torch_dtype, torch.float32)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, dtype=dtype)
    model.resize_token_embeddings(new_len)
    init_new_embeddings(model, tok, candidates, old_len=old_len)
    tok.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
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
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    slim = {k: v for k, v in manifest.items() if k not in ('selected_tokens', 'token_freq')}
    slim['n_selected'] = len(candidates)
    slim['sample_tokens'] = candidates[:50]
    (out_dir / 'c1c_summary.json').write_text(
        json.dumps(slim, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    if verbose:
        print(f'wrote {out_dir} (+{n_added} tokens)')
    return manifest


def main():
    p = argparse.ArgumentParser(description='C1c: extend NLLB vocab with legal SPM pieces')
    p.add_argument('--base', default='facebook/nllb-200-distilled-600M')
    p.add_argument('--spm', default=str(SPM_V2_PRIMARY))
    p.add_argument(
        '--bitext',
        default='data/external/parallel/subsamples/stage_a_A1_n80000.jsonl',
    )
    p.add_argument('--out', default='data/models/nllb600_c1c_sp_ext')
    p.add_argument('--top-k', type=int, default=8000)
    p.add_argument('--max-lines', type=int, default=20000)
    p.add_argument('--dtype', default='float32')
    a = p.parse_args()
    build_extended(
        base_model_id=a.base,
        spm_path=Path(a.spm),
        bitext_path=Path(a.bitext),
        out_dir=Path(a.out),
        top_k=a.top_k,
        max_lines=a.max_lines,
        torch_dtype=a.dtype,
    )


if __name__ == '__main__':
    main()

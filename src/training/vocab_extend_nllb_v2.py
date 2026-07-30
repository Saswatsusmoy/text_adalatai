"""C1c v2: careful NLLB vocab extend (surface filter + no regression)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import SPM_V2_PRIMARY


REGRESSION_PROBES = [
    'न्यायालय',
    'Court',
    'India',
    'section',
    'अदालत',
    'विधि',
    'संविधान',
    'High',
    'Supreme',
    'judgment',
    'order',
    'appeal',
]


def pieces(tok, surface: str) -> list[str]:
    return list(tok.tokenize(surface)) if surface else []


def already_single_token(tok, surface: str) -> bool:
    p = pieces(tok, surface)
    if len(p) != 1:
        return False
    t = p[0]
    core = t[1:] if t.startswith('▁') else t
    return core == surface or t == surface


def is_sub_of_protected(surface: str, protected: list[str]) -> bool:
    return any(surface != p and surface in p for p in protected)


def collect_fragment_freq(
    sp: spm.SentencePieceProcessor,
    tok_base,
    jsonl_path: Path,
    max_lines: int = 30000,
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
                    surface = piece[1:] if piece.startswith('▁') else piece
                    if len(surface) < 4:
                        continue
                    if re.fullmatch(r'[\W\d_]+', surface, flags=re.UNICODE):
                        continue
                    if is_sub_of_protected(surface, REGRESSION_PROBES):
                        continue
                    if already_single_token(tok_base, surface):
                        continue
                    if len(pieces(tok_base, surface)) < 2:
                        continue
                    freq[surface] += 1
            n += 1
    return freq


def uses_token(tok, surface: str, tid: int) -> bool:
    for probe in (surface, f'The {surface} is', f'{surface} ने', f' {surface} '):
        if tid in tok.encode(probe, add_special_tokens=False):
            return True
    return False


def no_regression(tok, tok_base) -> tuple[bool, str | None]:
    for p in REGRESSION_PROBES:
        if len(pieces(tok, p)) > len(pieces(tok_base, p)):
            return False, p
    return True, None


@torch.no_grad()
def init_new_rows_from_base(model, tok_ext, tok_base, surfaces: list[str], old_len: int):
    emb = model.get_input_embeddings().weight
    dtype = emb.dtype
    for surface in surfaces:
        tid = tok_ext.convert_tokens_to_ids(surface)
        if tid is None or tid < old_len:
            continue
        ids = [i for i in tok_base.encode(surface, add_special_tokens=False) if i < old_len]
        emb[tid] = emb[ids].mean(dim=0).to(dtype=dtype) if ids else emb[:old_len].mean(dim=0)
    model.tie_weights()


def filter_accepted(tok, tok_base, candidates: list[str]) -> list[str]:
    ok = []
    for surface in candidates:
        tid = tok.convert_tokens_to_ids(surface)
        if not uses_token(tok, surface, tid):
            continue
        if len(pieces(tok, surface)) >= len(pieces(tok_base, surface)):
            continue
        ok.append(surface)
    return ok


def build_extended_v2(
    base_model_id: str,
    spm_path: Path,
    bitext_path: Path,
    out_dir: Path,
    top_k: int = 1500,
    max_lines: int = 30000,
    torch_dtype: str = 'float32',
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=str(spm_path))
    tok_base = AutoTokenizer.from_pretrained(base_model_id)
    old_len = len(tok_base)
    if verbose:
        print(f'base={base_model_id} old_vocab={old_len}')
        print(f'scan {bitext_path} max_lines={max_lines}')

    freq = collect_fragment_freq(sp, tok_base, bitext_path, max_lines=max_lines)
    ranked = [s for s, _ in freq.most_common(top_k)]
    if verbose:
        print(f'fragment_candidates={len(freq)} taking_top={len(ranked)}')

    tok = AutoTokenizer.from_pretrained(base_model_id)
    tok.add_tokens(ranked)
    accepted = filter_accepted(tok, tok_base, ranked)
    if verbose:
        print(f'after filter_accepted={len(accepted)}')

    def rebuild(surfaces: list[str]):
        t = AutoTokenizer.from_pretrained(base_model_id)
        if surfaces:
            t.add_tokens(surfaces)
        return t

    surfaces = accepted
    while surfaces:
        tok = rebuild(surfaces)
        ok, bad = no_regression(tok, tok_base)
        if ok:
            surfaces = filter_accepted(tok, tok_base, surfaces)
            tok = rebuild(surfaces)
            ok2, bad2 = no_regression(tok, tok_base)
            if ok2:
                break
            surfaces = surfaces[: -max(1, len(surfaces) // 10)]
            if verbose:
                print(f'regression after refilter on {bad2}; shrink to {len(surfaces)}')
            continue
        cut = max(1, len(surfaces) // 5)
        if verbose:
            print(
                f'regression on {bad!r}; drop last {cut} ({len(surfaces)} -> {len(surfaces) - cut})'
            )
        surfaces = surfaces[:-cut]
    else:
        surfaces, tok = [], rebuild([])

    accepted, new_len = surfaces, len(tok)
    if verbose:
        print(f'final accepted={len(accepted)} new_vocab={new_len}')
        for s in accepted[:10]:
            print(f'  {s!r}: {pieces(tok_base, s)} -> {pieces(tok, s)}')
        for p in REGRESSION_PROBES[:6]:
            print(f'  probe {p!r}: {pieces(tok_base, p)} -> {pieces(tok, p)}')

    dtype = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }.get(torch_dtype, torch.float32)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, dtype=dtype)
    model.resize_token_embeddings(new_len)
    init_new_rows_from_base(model, tok, tok_base, accepted, old_len=old_len)
    tok.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    shutil.copy2(spm_path, out_dir / 'source_spm.model')

    manifest = {
        'version': 2,
        'method': 'fragment_surface_filter_no_regression',
        'base_model_id': base_model_id,
        'spm_path': str(spm_path),
        'bitext_path': str(bitext_path),
        'old_vocab': old_len,
        'new_vocab': new_len,
        'n_accepted': len(accepted),
        'accepted_tokens': accepted,
        'token_freq': {t: freq[t] for t in accepted},
        'regression_probes': REGRESSION_PROBES,
        'output': str(out_dir),
        'train_hint': {
            'new_embed_start': old_len,
            'freeze_old_embeddings_via_grad_mask': True,
        },
    }
    (out_dir / 'c1c_v2_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    (out_dir / 'c1c_v2_summary.json').write_text(
        json.dumps(
            {
                'version': 2,
                'old_vocab': old_len,
                'new_vocab': new_len,
                'n_accepted': len(accepted),
                'sample_tokens': accepted[:40],
                'output': str(out_dir),
                'train_hint': manifest['train_hint'],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )
    if verbose:
        print(f'wrote {out_dir}')
    return manifest


def main():
    p = argparse.ArgumentParser(description='C1c v2 careful NLLB vocab extend')
    p.add_argument('--base', default='facebook/nllb-200-distilled-600M')
    p.add_argument('--spm', default=str(SPM_V2_PRIMARY))
    p.add_argument(
        '--bitext',
        default='data/external/parallel/subsamples/stage_a_A1_n80000.jsonl',
    )
    p.add_argument('--out', default='data/models/nllb600_c1c_sp_ext_v2')
    p.add_argument('--top-k', type=int, default=1500)
    p.add_argument('--max-lines', type=int, default=30000)
    p.add_argument('--dtype', default='float32')
    a = p.parse_args()
    build_extended_v2(
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

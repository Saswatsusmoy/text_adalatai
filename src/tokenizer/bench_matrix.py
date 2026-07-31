"""Bench harness for the tokenizer matrix.

Auto-discovers all `sentencepiece_legal_v2_*` models in `data/models/tokenizers/`,
scores each on assignment held-out (322 pairs), and writes a single JSON summary
with a legal-term single-piece probe and UNK-rate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sentencepiece as spm

from src.tokenizer.benchmark import HELD_OUT_DOCS, _norm_doc_id, count_dev_pieces


ALIGNED_PATH = Path('data/aligned/all.jsonl')
MODEL_DIR = Path('data/models/tokenizers')
OUT_PATH = Path('data/analysis/tokenizer_matrix.json')

# Probe list: legal terms that a good legal tokenizer should keep as a single piece
LEGAL_PROBES_HI: tuple[str, ...] = (
    'न्यायालय',
    'अनुच्छेद',
    'अपीलकर्ता',
    'प्रतिवादी',
    'याचिका',
    'रिट',
    'धारा',
    'आदेश',
    'नियम',
    'अधिवक्ता',
    'पुनरीक्षण',
    'अधिकारिता',
    'निर्णय',
    'भारत',
    'संविधान',
)
LEGAL_PROBES_EN: tuple[str, ...] = (
    'Section',
    'Article',
    'Order',
    'Rule',
    'Writ',
    'Petition',
    'Appellant',
    'Respondent',
    'Judgment',
    'impugned',
    'appellant',
    'respondent',
)


def load_held_out() -> list[dict]:
    pairs = []
    with open(ALIGNED_PATH, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if _norm_doc_id(r.get('doc_id')) in HELD_OUT_DOCS:
                pairs.append(r)
    return pairs


def _unk_rate(sp: spm.SentencePieceProcessor, ids_list: list[list[int]]) -> float:
    unk = sp.unk_id()
    total = 0
    unks = 0
    for ids in ids_list:
        total += len(ids)
        unks += sum(1 for i in ids if i == unk)
    return round(unks / max(total, 1), 6)


def _probe_single_piece(sp: spm.SentencePieceProcessor, terms: tuple[str, ...]) -> dict:
    """Return per-term piece counts and single-piece hit rate."""
    per_term = {}
    hits = 0
    for t in terms:
        ids = sp.encode(t)
        per_term[t] = len(ids)
        if len(ids) == 1:
            hits += 1
    return {
        'hits': hits,
        'total': len(terms),
        'rate': round(hits / max(len(terms), 1), 4),
        'per_term': per_term,
    }


def _bench_one(mp: Path, corpus: list[dict]) -> dict:
    sp = spm.SentencePieceProcessor(model_file=str(mp))
    en_chars = sum(len(p['en_text']) for p in corpus)
    hi_chars = sum(len(p['hi_text']) for p in corpus)

    t0 = time.time()
    en_ids = [sp.encode(p['en_text']) for p in corpus]
    hi_ids = [sp.encode(p['hi_text']) for p in corpus]
    elapsed = round(time.time() - t0, 3)

    en_flat = sum(len(x) for x in en_ids)
    hi_flat = sum(len(x) for x in hi_ids)

    return {
        'name': mp.stem.replace('sentencepiece_', ''),
        'model_path': str(mp),
        'model_size_bytes': mp.stat().st_size,
        'vocab_size': sp.GetPieceSize(),
        'dev_pieces': count_dev_pieces(sp),
        'en_chars_per_tok': round(en_chars / max(en_flat, 1), 3),
        'hi_chars_per_tok': round(hi_chars / max(hi_flat, 1), 3),
        'hi_en_ratio': round(hi_flat / max(en_flat, 1), 4),
        'total_tokens': en_flat + hi_flat,
        'en_tokens': en_flat,
        'hi_tokens': hi_flat,
        'encode_s_held_out': elapsed,
        'unk_rate_held_out': _unk_rate(sp, en_ids + hi_ids),
        'legal_probe_hi': _probe_single_piece(sp, LEGAL_PROBES_HI),
        'legal_probe_en': _probe_single_piece(sp, LEGAL_PROBES_EN),
    }


def discover_matrix_models() -> list[Path]:
    return sorted(MODEL_DIR.glob('sentencepiece_legal_v2_*.model'))


def run(out_path: Path = OUT_PATH, verbose: bool = True) -> dict:
    corpus = load_held_out()
    models = discover_matrix_models()
    if verbose:
        print(f'Held-out pairs: {len(corpus)}; models discovered: {len(models)}')

    results = []
    for mp in models:
        r = _bench_one(mp, corpus)
        results.append(r)
        if verbose:
            print(
                f'  {r["name"]:60}  vocab={r["vocab_size"]:>6}  '
                f'HI c/t={r["hi_chars_per_tok"]:.3f}  '
                f'total={r["total_tokens"]:>6}  '
                f'legal-HI={r["legal_probe_hi"]["rate"]:.2f}  '
                f'legal-EN={r["legal_probe_en"]["rate"]:.2f}  '
                f'unk={r["unk_rate_held_out"]:.4f}'
            )

    payload = {
        'eval_split': 'held_out',
        'n_pairs': len(corpus),
        'legal_probes_hi': list(LEGAL_PROBES_HI),
        'legal_probes_en': list(LEGAL_PROBES_EN),
        'results': results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    if verbose:
        print(f'\nSaved: {out_path}')
    return payload


def main():
    p = argparse.ArgumentParser(description='Bench the tokenizer matrix on assignment held-out')
    p.add_argument('--out', type=Path, default=OUT_PATH)
    a = p.parse_args()
    run(out_path=a.out)


if __name__ == '__main__':
    main()

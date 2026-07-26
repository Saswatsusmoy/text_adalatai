"""
Benchmark tokenizers on EN-HI legal pairs.

Default: custom SP v1 + legal v2 on held-out assignment dev+test
  (pairs whose doc_id is in DEV_DOC_IDS | TEST_DOC_IDS).

Usage:
    PYTHONPATH=. python3 src/tokenizer/benchmark.py
    PYTHONPATH=. python3 src/tokenizer/benchmark.py --eval all
    PYTHONPATH=. python3 src/tokenizer/benchmark.py --full
"""

import json
import math
import time
from pathlib import Path

import sentencepiece as spm

from src.config import DEV_DOC_IDS, TEST_DOC_IDS

ALIGNED_PATH = Path('data/aligned/all.jsonl')
PROCESSED_DIR = Path('data/processed')
MODEL_DIR = Path('data/models/tokenizers')
ANALYSIS_DIR = Path('data/analysis')

HELD_OUT_DOCS = set(DEV_DOC_IDS) | set(TEST_DOC_IDS)


def _norm_doc_id(raw) -> int | None:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return None


def load_corpus(eval_split: str = 'held_out') -> list[dict]:
    """eval_split: held_out (dev+test), all, train, dev, test."""
    if eval_split in ('train', 'dev', 'test'):
        path = PROCESSED_DIR / f'{eval_split}.jsonl'
        if path.exists():
            pairs = []
            with open(path, encoding='utf-8') as f:
                for line in f:
                    pairs.append(json.loads(line))
            return pairs

    aligned = []
    with open(ALIGNED_PATH, encoding='utf-8') as f:
        for line in f:
            aligned.append(json.loads(line))

    if eval_split == 'all':
        return aligned
    if eval_split == 'held_out':
        return [p for p in aligned if _norm_doc_id(p.get('doc_id')) in HELD_OUT_DOCS]
    if eval_split == 'train':
        from src.config import TRAIN_DOC_IDS
        train = set(TRAIN_DOC_IDS)
        return [p for p in aligned if _norm_doc_id(p.get('doc_id')) in train]
    raise ValueError(f'unknown eval_split {eval_split}')


def count_devanagari(text: str) -> int:
    return sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)


def count_dev_pieces(sp: spm.SentencePieceProcessor) -> int:
    return sum(
        1
        for i in range(sp.GetPieceSize())
        if any(0x0900 <= ord(c) <= 0x097F for c in sp.IdToPiece(i))
    )


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def benchmark_tokenizer(encode_fn, decode_fn, corpus, label='', vocab_size=0, dev_count=0):
    en_chars = sum(len(p['en_text']) for p in corpus)
    hi_chars = sum(len(p['hi_text']) for p in corpus)

    t0 = time.time()
    en_ids = [encode_fn(p['en_text']) for p in corpus]
    hi_ids = [encode_fn(p['hi_text']) for p in corpus]
    elapsed = time.time() - t0

    en_flat = [i for ids in en_ids for i in ids]
    hi_flat = [i for ids in hi_ids for i in ids]

    en_tokens = [decode_fn([i]) for i in en_flat[:10000]]
    hi_tokens = [decode_fn([i]) for i in hi_flat[:10000]]

    en_lens = [len(t) for t in en_tokens]
    hi_lens = [len(t) for t in hi_tokens]

    return {
        'name': label,
        'vocab': vocab_size,
        'dev_tokens': dev_count,
        'en_chars_per_tok': round(en_chars / max(len(en_flat), 1), 2),
        'hi_chars_per_tok': round(hi_chars / max(len(hi_flat), 1), 2),
        'hi_en_ratio': round(len(hi_flat) / max(len(en_flat), 1), 3),
        'total_tokens': len(en_flat) + len(hi_flat),
        'time_s': round(elapsed, 1),
        'en_subword_regularity': round(
            _entropy([en_lens.count(i) for i in range(1, 20) if en_lens.count(i) > 0]), 2
        )
        if en_lens
        else 0,
        'hi_subword_regularity': round(
            _entropy([hi_lens.count(i) for i in range(1, 20) if hi_lens.count(i) > 0]), 2
        )
        if hi_lens
        else 0,
    }


def print_table(results: list[dict]):
    print(
        f"{'Tokenizer':<36} {'Vocab':<7} {'Dev':<6} {'HI c/t':<8} "
        f"{'HI/EN':<8} {'Total':<8}"
    )
    print('-' * 80)
    for r in sorted(results, key=lambda x: -x.get('hi_chars_per_tok', 0)):
        print(
            f"{r['name']:<36} {r['vocab']:<7} {str(r.get('dev_tokens', '-')):<6} "
            f"{r['hi_chars_per_tok']:<8.2f} {r['hi_en_ratio']:<8.3f} {r['total_tokens']:<8,}"
        )


def _bench_sp_file(mp: Path, corpus: list[dict], label: str) -> dict | None:
    if not mp.exists():
        return None
    sp = spm.SentencePieceProcessor(model_file=str(mp))
    dev = count_dev_pieces(sp)
    return benchmark_tokenizer(
        lambda t, sp=sp: sp.encode(t),
        lambda ids, sp=sp: sp.decode(ids),
        corpus,
        label=label,
        vocab_size=sp.GetPieceSize(),
        dev_count=dev,
    )


def discover_custom_models() -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    # v1 Prarabdha baseline
    for vs in [16000, 32000, 41000]:
        mp = MODEL_DIR / f'sentencepiece_{vs}.model'
        if mp.exists():
            found.append((mp, f'v1 SP {vs} (Prarabdha)'))
    # v2 legal
    if MODEL_DIR.exists():
        for mp in sorted(MODEL_DIR.glob('sentencepiece_legal_v2_*.model')):
            label = f"v2 {mp.stem.replace('sentencepiece_legal_v2_', 'SP ')}"
            found.append((mp, label))
    return found


def run(full: bool = False, eval_split: str = 'held_out') -> list[dict]:
    corpus = load_corpus(eval_split)
    if not corpus:
        raise FileNotFoundError(f'no pairs for eval_split={eval_split}')

    print(f'Eval split: {eval_split} ({len(corpus)} pairs)')
    results: list[dict] = []

    for mp, label in discover_custom_models():
        r = _bench_sp_file(mp, corpus, label)
        if r:
            results.append(r)

    if not full:
        print_table(results)
        out = ANALYSIS_DIR / 'tokenizer_metrics_v2.json'
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            'eval_split': eval_split,
            'n_pairs': len(corpus),
            'results': results,
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'\nSaved: {out}')
        return results

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer as HFTok

    models = [
        ('facebook/nllb-200-distilled-600M', 'NLLB-200'),
        ('Qwen/Qwen3-8B', 'Qwen3'),
        ('microsoft/Phi-4-mini-instruct', 'Phi-4-mini'),
    ]

    for repo_id, label in models:
        try:
            path = hf_hub_download(
                repo_id=repo_id, filename='tokenizer.json', cache_dir='/tmp/tokenizers',
            )
            tok = HFTok.from_file(path)
            vocab = tok.get_vocab_size()
            dev = sum(
                1 for k in tok.get_vocab() if any(0x0900 <= ord(c) <= 0x097F for c in k)
            )
            r = benchmark_tokenizer(
                lambda t, tok=tok: tok.encode(t).ids,
                lambda ids, tok=tok: tok.decode(ids),
                corpus,
                label=label,
                vocab_size=vocab,
                dev_count=dev,
            )
            results.append(r)
        except Exception as e:
            print(f'  [SKIP] {label}: {e}')

    try:
        import tiktoken

        o200k = tiktoken.get_encoding('o200k_base')
        r = benchmark_tokenizer(
            lambda t, tok=o200k: tok.encode(t),
            lambda ids, tok=o200k: tok.decode(ids),
            corpus,
            label='GPT-4o (o200k)',
            vocab_size=o200k.n_vocab,
            dev_count=0,
        )
        results.append(r)
    except Exception:
        pass

    print_table(results)
    out = ANALYSIS_DIR / 'tokenizer_metrics_v2.json'
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {'eval_split': eval_split, 'n_pairs': len(corpus), 'results': results}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSaved: {out}')
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark tokenizers on EN-HI corpus')
    parser.add_argument(
        '--full', action='store_true', help='Also bench a few HF / tiktoken models',
    )
    parser.add_argument(
        '--eval',
        default='held_out',
        choices=['held_out', 'all', 'train', 'dev', 'test'],
        help='Which pairs to encode (default held_out = assignment dev+test)',
    )
    args = parser.parse_args()
    run(full=args.full, eval_split=args.eval)

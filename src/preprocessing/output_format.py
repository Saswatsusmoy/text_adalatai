"""
Final output format: train/dev/test splits for parallel corpus.

Reads aligned pairs from data/aligned/all.jsonl, splits at document level
(80/10/10), and writes to data/processed/ as JSONL.

Also generates metadata.json and alignment_report.json.
"""

import json
import random
from pathlib import Path

from src.config import DEV_DOC_IDS, TEST_DOC_IDS, TRAIN_DOC_IDS


ALIGNED_DIR = Path('data/aligned')
OUTPUT_DIR = Path('data/processed')

TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42


def load_aligned() -> list[dict]:
    path = ALIGNED_DIR / 'all.jsonl'
    if not path.exists():
        return []
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def split_docs(doc_ids: list[int]) -> dict[str, list[int]]:
    n = len(doc_ids)
    n_train = int(n * TRAIN_RATIO)
    n_dev = int(n * DEV_RATIO)
    train = doc_ids[:n_train]
    dev = doc_ids[n_train : n_train + n_dev]
    test = doc_ids[n_train + n_dev :]

    return {'train': sorted(train), 'dev': sorted(dev), 'test': sorted(test)}


def frozen_splits(doc_ids: list[int] | None = None) -> dict[str, list[int]]:
    """Document-level split frozen in src.config (seed-42 historical shuffle)."""
    present = set(doc_ids) if doc_ids is not None else None
    splits = {
        'train': list(TRAIN_DOC_IDS),
        'dev': list(DEV_DOC_IDS),
        'test': list(TEST_DOC_IDS),
    }
    if present is None:
        return splits
    return {name: [d for d in ids if d in present] for name, ids in splits.items()}


def build_metadata(pairs: list[dict], splits: dict[str, list[int]]) -> dict:
    doc_pairs = {}
    for p in pairs:
        doc_id = p['doc_id']
        doc_pairs.setdefault(doc_id, []).append(p)

    split_name_for_doc = {}
    for split_name, doc_ids in splits.items():
        for doc_id in doc_ids:
            split_name_for_doc[doc_id] = split_name

    doc_entries = []
    for doc_id in sorted(doc_pairs.keys()):
        doc_p = doc_pairs[doc_id]
        avg_sim = sum(p['similarity'] for p in doc_p) / len(doc_p)
        doc_entries.append(
            {
                'doc_id': doc_id,
                'split': split_name_for_doc.get(doc_id, 'unknown'),
                'num_pairs': len(doc_p),
                'avg_similarity': round(avg_sim, 4),
                'en_chars': sum(len(p['en_text']) for p in doc_p),
                'hi_chars': sum(len(p['hi_text']) for p in doc_p),
            }
        )

    return {
        'corpus': {
            'total_pairs': len(pairs),
            'total_docs': len(doc_pairs),
            'languages': ['en', 'hi'],
            'domain': 'legal',
            'source': 'Indian Supreme Court judgments (adalat_ai)',
        },
        'splits': {name: {'doc_ids': ids, 'num_docs': len(ids)} for name, ids in splits.items()},
        'documents': doc_entries,
    }


def build_report(pairs: list[dict]) -> dict:
    sims = [p['similarity'] for p in pairs]
    en_lens = [len(p['en_text']) for p in pairs]
    hi_lens = [len(p['hi_text']) for p in pairs]

    doc_ids = set(p['doc_id'] for p in pairs)

    return {
        'pipeline': {
            'steps_completed': [
                '1. PDF re-extraction (Tesseract OCR)',
                '2. Line joining (English)',
                '3. Sentence segmentation (spaCy + danda split)',
                '4. LaBSE alignment + quality filters',
            ],
            'steps_skipped': [
                'UTF-8 BOM (not in preprocessed data)',
                'OCR Roman numerals (not present in corpus)',
                'CRLF normalization (preprocessed is LF-only)',
                'Paragraph segmentation (already correct)',
            ],
        },
        'alignment': {
            'model': 'sentence-transformers/LaBSE',
            'method': 'bidirectional greedy matching',
            'filters': {
                'min_similarity': 0.5,
                'char_ratio_range': [0.3, 3.0],
                'min_text_length': 3,
                'dedup_jaccard_threshold': 0.85,
            },
        },
        'statistics': {
            'total_pairs': len(pairs),
            'total_docs': len(doc_ids),
            'avg_similarity': round(sum(sims) / len(sims), 4) if sims else 0,
            'similarity_std': round(
                (sum((s - sum(sims) / len(sims)) ** 2 for s in sims) / len(sims)) ** 0.5, 4
            )
            if sims
            else 0,
            'avg_en_chars': round(sum(en_lens) / len(en_lens), 1) if en_lens else 0,
            'avg_hi_chars': round(sum(hi_lens) / len(hi_lens), 1) if hi_lens else 0,
            'total_en_chars': sum(en_lens),
            'total_hi_chars': sum(hi_lens),
        },
        'pair_type_distribution': {
            '1-1 (mutual best)': len(pairs),
        },
    }


def write_jsonl(pairs: list[dict], path: Path):
    from src.utils.jsonl import write_jsonl as _write

    _write(path, pairs)


def run(verbose: bool = True) -> dict:
    pairs = load_aligned()
    if not pairs:
        if verbose:
            print('No aligned pairs found. Run align_sentences.py first.')
        return {'status': 'no_data'}

    all_doc_ids = sorted(set(p['doc_id'] for p in pairs))
    # Prefer frozen Policy-I doc IDs so re-align/rebuild never reshuffles assignment splits.
    frozen = frozen_splits(all_doc_ids)
    covered = set(frozen['train']) | set(frozen['dev']) | set(frozen['test'])
    if covered == set(all_doc_ids):
        splits = frozen
    else:
        random.seed(RANDOM_SEED)
        shuffled = list(all_doc_ids)
        random.shuffle(shuffled)
        splits = split_docs(shuffled)

    split_name_for_doc = {}
    for split_name, doc_ids in splits.items():
        for doc_id in doc_ids:
            split_name_for_doc[doc_id] = split_name

    split_pairs = {}
    for name in ['train', 'dev', 'test']:
        split_pairs[name] = []

    for p in pairs:
        split_name = split_name_for_doc.get(p['doc_id'], 'unknown')
        split_pairs[split_name].append(p)

    for name in ['train', 'dev', 'test']:
        out_path = OUTPUT_DIR / f'{name}.jsonl'
        write_jsonl(split_pairs[name], out_path)
        if verbose:
            print(f'  {name}: {len(split_pairs[name]):4d} pairs -> {out_path}')

    metadata = build_metadata(pairs, splits)
    meta_path = OUTPUT_DIR / 'metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f'  metadata: -> {meta_path}')

    report = build_report(pairs)
    report_path = OUTPUT_DIR / 'alignment_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f'  report: -> {report_path}')

    return {
        'status': 'ok',
        'train': len(split_pairs['train']),
        'dev': len(split_pairs['dev']),
        'test': len(split_pairs['test']),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate train/dev/test splits and metadata',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output',
    )
    args = parser.parse_args()
    run(verbose=not args.quiet)


if __name__ == '__main__':
    main()

"""
Carve dual eval policies from Stage A external bitext.

Policy I (internal): assignment data/processed/{train,dev,test}.jsonl (unchanged).
Policy E (external held-out): MILPaC + Anuvaad slices never used for Stage A MT train.

Writes:
  data/external/parallel/stage_a_train.jsonl     # Stage A MT only
  data/external/parallel/eval/milpac_{dev,test}.jsonl
  data/external/parallel/eval/anuvaad_{dev,test}.jsonl
  data/external/parallel/eval/eval_manifest.json

Does not modify assignment splits. Seed frozen at 42.
"""

import json
import random
from pathlib import Path

from src.config import EXTERNAL_EVAL_DIR, EXTERNAL_PARALLEL_DIR, STAGE_A_ALL, STAGE_A_TRAIN
from src.utils.jsonl import load_jsonl, write_jsonl


PARALLEL_DIR = EXTERNAL_PARALLEL_DIR
EVAL_DIR = EXTERNAL_EVAL_DIR

SEED = 42
# MILPaC: quality legal; keep most for train, hold out stable fractions
MILPAC_DEV_FRAC = 0.10
MILPAC_TEST_FRAC = 0.10
# Anuvaad: large/noisy; fixed-size held-out for stable automatic metrics
ANUVAAD_DEV_N = 1000
ANUVAAD_TEST_N = 3000


def pair_key(p: dict) -> tuple[str, str]:
    return (p.get('en_text', '').strip(), p.get('hi_text', '').strip())


def is_milpac(source: str) -> bool:
    return (source or '').startswith('milpac')


def is_anuvaad(source: str) -> bool:
    return (source or '').startswith('anuvaad')


def split_fraction(
    rows: list[dict],
    dev_frac: float,
    test_frac: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not rows:
        return [], [], []
    order = list(rows)
    rng.shuffle(order)
    n = len(order)
    n_test = max(1, int(round(n * test_frac))) if n >= 10 else max(0, min(1, n // 5))
    n_dev = max(1, int(round(n * dev_frac))) if n >= 10 else max(0, min(1, n // 5))
    if n_test + n_dev >= n:
        n_test = max(1, n // 10) if n >= 10 else 0
        n_dev = max(1, n // 10) if n >= 10 else 0
        if n_test + n_dev >= n:
            n_test = min(1, n)
            n_dev = 0 if n < 3 else 1
            if n_test + n_dev >= n:
                return order, [], []
    test = order[:n_test]
    dev = order[n_test : n_test + n_dev]
    train = order[n_test + n_dev :]
    return train, dev, test


def split_fixed_n(
    rows: list[dict],
    dev_n: int,
    test_n: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not rows:
        return [], [], []
    order = list(rows)
    rng.shuffle(order)
    n = len(order)
    test_n = min(test_n, max(0, n // 2))
    dev_n = min(dev_n, max(0, (n - test_n) // 5))
    test = order[:test_n]
    dev = order[test_n : test_n + dev_n]
    train = order[test_n + dev_n :]
    return train, dev, test


def assert_no_overlap(train: list[dict], *held: list[dict]):
    train_keys = {pair_key(p) for p in train}
    for group in held:
        for p in group:
            k = pair_key(p)
            if k in train_keys and k != ('', ''):
                raise ValueError(f'eval pair leaked into train: {k[0][:60]!r}')


def run(
    stage_a_path: Path = STAGE_A_ALL,
    seed: int = SEED,
    anuvaad_dev_n: int = ANUVAAD_DEV_N,
    anuvaad_test_n: int = ANUVAAD_TEST_N,
    milpac_dev_frac: float = MILPAC_DEV_FRAC,
    milpac_test_frac: float = MILPAC_TEST_FRAC,
    verbose: bool = True,
) -> dict:
    all_pairs = load_jsonl(stage_a_path)
    if not all_pairs:
        raise FileNotFoundError(f'no pairs at {stage_a_path}; run external ingest first')

    rng = random.Random(seed)
    milpac = [p for p in all_pairs if is_milpac(p.get('source', ''))]
    anuvaad = [p for p in all_pairs if is_anuvaad(p.get('source', ''))]
    other = [
        p
        for p in all_pairs
        if not is_milpac(p.get('source', '')) and not is_anuvaad(p.get('source', ''))
    ]

    m_train, m_dev, m_test = split_fraction(
        milpac,
        milpac_dev_frac,
        milpac_test_frac,
        rng,
    )
    a_train, a_dev, a_test = split_fixed_n(
        anuvaad,
        anuvaad_dev_n,
        anuvaad_test_n,
        rng,
    )

    train = m_train + a_train + other
    rng.shuffle(train)

    assert_no_overlap(train, m_dev, m_test, a_dev, a_test)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(STAGE_A_TRAIN, train)
    write_jsonl(EVAL_DIR / 'milpac_dev.jsonl', m_dev)
    write_jsonl(EVAL_DIR / 'milpac_test.jsonl', m_test)
    write_jsonl(EVAL_DIR / 'anuvaad_dev.jsonl', a_dev)
    write_jsonl(EVAL_DIR / 'anuvaad_test.jsonl', a_test)

    # Combined Policy E convenience files
    e_dev = m_dev + a_dev
    e_test = m_test + a_test
    write_jsonl(EVAL_DIR / 'external_dev.jsonl', e_dev)
    write_jsonl(EVAL_DIR / 'external_test.jsonl', e_test)

    manifest = {
        'seed': seed,
        'source_pool': str(stage_a_path),
        'policy_I': {
            'name': 'internal_assignment',
            'train': 'data/processed/train.jsonl',
            'dev': 'data/processed/dev.jsonl',
            'test': 'data/processed/test.jsonl',
            'note': 'Document-level split; frozen doc IDs in src.config',
        },
        'policy_E': {
            'name': 'external_held_out',
            'stage_a_train': str(STAGE_A_TRAIN),
            'milpac_dev': str(EVAL_DIR / 'milpac_dev.jsonl'),
            'milpac_test': str(EVAL_DIR / 'milpac_test.jsonl'),
            'anuvaad_dev': str(EVAL_DIR / 'anuvaad_dev.jsonl'),
            'anuvaad_test': str(EVAL_DIR / 'anuvaad_test.jsonl'),
            'external_dev': str(EVAL_DIR / 'external_dev.jsonl'),
            'external_test': str(EVAL_DIR / 'external_test.jsonl'),
            'milpac_dev_frac': milpac_dev_frac,
            'milpac_test_frac': milpac_test_frac,
            'anuvaad_dev_n': anuvaad_dev_n,
            'anuvaad_test_n': anuvaad_test_n,
        },
        'counts': {
            'pool_total': len(all_pairs),
            'milpac_total': len(milpac),
            'anuvaad_total': len(anuvaad),
            'other_total': len(other),
            'stage_a_train': len(train),
            'milpac_dev': len(m_dev),
            'milpac_test': len(m_test),
            'anuvaad_dev': len(a_dev),
            'anuvaad_test': len(a_test),
            'external_dev': len(e_dev),
            'external_test': len(e_test),
        },
        'spm_note': (
            'SPM v2 may have been fit on full stage_a_en_hi before this carve. '
            'MT must train only on stage_a_train. Optional strict SPM rebuild excludes E-eval lines.'
        ),
    }
    man_path = EVAL_DIR / 'eval_manifest.json'
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    if verbose:
        print('Dual eval split complete')
        for k, v in manifest['counts'].items():
            print(f'  {k}: {v:,}')
        print(f'  stage_a_train -> {STAGE_A_TRAIN}')
        print(f'  manifest -> {man_path}')
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Split external Stage A into train + Policy E eval'
    )
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--anuvaad-test-n', type=int, default=ANUVAAD_TEST_N)
    parser.add_argument('--anuvaad-dev-n', type=int, default=ANUVAAD_DEV_N)
    args = parser.parse_args()
    run(
        seed=args.seed,
        anuvaad_test_n=args.anuvaad_test_n,
        anuvaad_dev_n=args.anuvaad_dev_n,
    )


if __name__ == '__main__':
    main()

"""
Build Stage A curriculum subsamples and Stage B anti-forget replay mixes.

Tags:
  smoke -- small random sample
  A1    -- quality-weighted sources, capped judiciary
  A2    -- larger mix up to A2_max_n
  full  -- copy pointer / all lines (no subsample file rewrite of full 988k unless forced)
  Bp    -- Stage B: all assignment train + domain replay (default 90/10)
"""

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.training.config import deep_get, load_training_config

OUT_DIR = Path('data/external/parallel/subsamples')
STAGE_B_OUT_DIR = Path('data/external/parallel/subsamples')


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def file_sha256(path: Path, max_bytes: int = 8_000_000) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1_000_000, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _take_up_to(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if n <= 0 or not rows:
        return []
    if n >= len(rows):
        return list(rows)
    return rng.sample(rows, n)


def build_subsample(
    curriculum: str = 'A1',
    config_path: str | Path = 'configs/training.yaml',
    verbose: bool = True,
) -> dict:
    cfg = load_training_config(config_path)
    src_path = Path(deep_get(cfg, 'data', 'stage_a_train'))
    if not src_path.exists():
        raise FileNotFoundError(
            f'{src_path} missing; run make external-eval-split first'
        )

    sub = deep_get(cfg, 'data', 'subsample', default={}) or {}
    seed = int(sub.get('seed', deep_get(cfg, 'run', 'seed', default=42)))
    rng = random.Random(seed)

    all_rows = load_jsonl(src_path)
    by_src: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_src[r.get('source') or 'unknown'].append(r)

    prefer = list(sub.get('prefer_sources') or [])
    judiciary_cap = int(sub.get('judiciary_cap_A1', 40000))

    if curriculum == 'smoke':
        n = int(sub.get('smoke_n', 2000))
        chosen = _take_up_to(all_rows, n, rng)
        tag = f'smoke_n{len(chosen)}'
    elif curriculum == 'full':
        chosen = list(all_rows)
        tag = f'full_n{len(chosen)}'
    elif curriculum in ('A1', 'A2'):
        max_n = int(sub.get('A1_max_n' if curriculum == 'A1' else 'A2_max_n', 80000))
        chosen = []
        # Take all milpac + prefer non-judiciary first
        for src in prefer:
            if src == 'anuvaad_judiciary':
                continue
            chosen.extend(by_src.get(src, []))
        # judiciary capped for A1, larger for A2
        jud = by_src.get('anuvaad_judiciary', [])
        jud_n = judiciary_cap if curriculum == 'A1' else min(
            len(jud), max(0, max_n - len(chosen))
        )
        if curriculum == 'A2':
            jud_n = min(len(jud), max(judiciary_cap, max_n // 2))
        chosen.extend(_take_up_to(jud, jud_n, rng))
        # fill remainder from leftover sources
        if len(chosen) < max_n:
            leftover = []
            chosen_ids = {id(x) for x in chosen}
            for src, rows in by_src.items():
                for r in rows:
                    if id(r) not in chosen_ids:
                        leftover.append(r)
            need = max_n - len(chosen)
            chosen.extend(_take_up_to(leftover, need, rng))
        if len(chosen) > max_n:
            chosen = _take_up_to(chosen, max_n, rng)
        rng.shuffle(chosen)
        tag = f'{curriculum}_n{len(chosen)}'
    else:
        raise ValueError(f'unknown curriculum {curriculum}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'stage_a_{tag}.jsonl'
    write_jsonl(out_path, chosen)

    src_counts = Counter(r.get('source') or 'unknown' for r in chosen)
    manifest = {
        'curriculum': curriculum,
        'tag': tag,
        'seed': seed,
        'source_pool': str(src_path),
        'source_pool_sha256_prefix': file_sha256(src_path),
        'output': str(out_path),
        'n': len(chosen),
        'source_counts': dict(src_counts),
    }
    man_path = OUT_DIR / f'stage_a_{tag}_manifest.json'
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    if verbose:
        print(f'Subsample {curriculum}: {len(chosen):,} pairs -> {out_path}')
        for s, c in src_counts.most_common():
            print(f'  {c:7d} {s}')
        print(f'Manifest: {man_path}')
    return manifest


def _pair_key(row: dict) -> tuple[str, str]:
    return (row.get('en_text') or '', row.get('hi_text') or '')


def build_stage_b_replay_mix(
    config_path: str | Path = 'configs/training.yaml',
    verbose: bool = True,
) -> dict:
    """
    Stage B' mix: all assignment train pairs + domain replay from Stage A subsample.

    Default 90/10 by count: keep every assignment row, sample
    n_replay = round(n_assign * (1 - assignment_frac) / assignment_frac)
    from the replay pool (excluding exact EN-HI pairs already in assignment).
    """
    cfg = load_training_config(config_path)
    assign_path = Path(deep_get(cfg, 'data', 'stage_b_train'))
    if not assign_path.exists():
        raise FileNotFoundError(f'{assign_path} missing; run preprocess splits first')

    replay_cfg = deep_get(cfg, 'data', 'stage_b_replay', default={}) or {}
    pool_raw = replay_cfg.get('pool') or deep_get(cfg, 'data', 'stage_a_train')
    replay_path = Path(pool_raw)
    if not replay_path.exists():
        a2_default = Path('data/external/parallel/subsamples/stage_a_A2_n150000.jsonl')
        if a2_default.exists():
            replay_path = a2_default
        else:
            raise FileNotFoundError(
                f'replay pool missing: {replay_path}; build A2 subsample first'
            )

    seed = int(
        replay_cfg.get('seed')
        or deep_get(cfg, 'data', 'subsample', 'seed', default=None)
        or deep_get(cfg, 'run', 'seed', default=42)
    )
    assignment_frac = float(replay_cfg.get('assignment_frac', 0.9))
    if not 0.5 <= assignment_frac < 1.0:
        raise ValueError(f'assignment_frac must be in [0.5, 1.0), got {assignment_frac}')

    rng = random.Random(seed)
    assign_rows = load_jsonl(assign_path)
    if not assign_rows:
        raise ValueError(f'empty assignment train: {assign_path}')

    assign_keys = {_pair_key(r) for r in assign_rows}
    pool_rows = [
        r for r in load_jsonl(replay_path)
        if _pair_key(r) not in assign_keys and (r.get('en_text') and r.get('hi_text'))
    ]
    n_assign = len(assign_rows)
    n_replay = int(round(n_assign * (1.0 - assignment_frac) / assignment_frac))
    n_replay = max(0, min(n_replay, len(pool_rows)))
    replay_rows = _take_up_to(pool_rows, n_replay, rng)

    mixed: list[dict] = []
    for r in assign_rows:
        row = dict(r)
        row['mix_role'] = 'assignment'
        mixed.append(row)
    for r in replay_rows:
        row = dict(r)
        row['mix_role'] = 'replay'
        mixed.append(row)
    rng.shuffle(mixed)

    STAGE_B_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f'Bp_a{n_assign}_r{len(replay_rows)}_f{assignment_frac:g}'
    out_path = STAGE_B_OUT_DIR / f'stage_b_{tag}.jsonl'
    write_jsonl(out_path, mixed)

    role_counts = Counter(r['mix_role'] for r in mixed)
    src_counts = Counter(r.get('source') or 'unknown' for r in mixed)
    actual_assign_frac = role_counts['assignment'] / len(mixed) if mixed else 0.0
    manifest = {
        'curriculum': 'Bp',
        'tag': tag,
        'seed': seed,
        'assignment_frac_target': assignment_frac,
        'assignment_frac_actual': round(actual_assign_frac, 4),
        'assignment_path': str(assign_path),
        'assignment_sha256_prefix': file_sha256(assign_path),
        'replay_pool': str(replay_path),
        'replay_pool_sha256_prefix': file_sha256(replay_path),
        'output': str(out_path),
        'n': len(mixed),
        'n_assignment': n_assign,
        'n_replay': len(replay_rows),
        'role_counts': dict(role_counts),
        'source_counts': dict(src_counts),
    }
    man_path = STAGE_B_OUT_DIR / f'stage_b_{tag}_manifest.json'
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    if verbose:
        print(
            f'Stage B replay mix: {len(mixed):,} pairs '
            f'(assign={n_assign}, replay={len(replay_rows)}, '
            f'frac={actual_assign_frac:.3f}) -> {out_path}'
        )
        print(f'Manifest: {man_path}')
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Build Stage A curriculum subsample or Stage B replay mix',
    )
    parser.add_argument(
        '--curriculum',
        default='A1',
        choices=['smoke', 'A1', 'A2', 'full', 'Bp'],
    )
    parser.add_argument('--config', default='configs/training.yaml')
    args = parser.parse_args()
    if args.curriculum == 'Bp':
        build_stage_b_replay_mix(config_path=args.config)
    else:
        build_subsample(curriculum=args.curriculum, config_path=args.config)


if __name__ == '__main__':
    main()

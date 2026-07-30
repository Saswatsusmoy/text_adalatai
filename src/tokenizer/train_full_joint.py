"""
Train joint legal SPM trying full (or fuller) coverage within ~16GB RAM.

Strategy (Unigram only — no byte-level BPE):
  1. Ensure base joint corpus exists
  2. Exact-line dedupe (+ optional truncate) to shrink SA peak
  3. Try profiles in order: full -> full_tight -> full_sample_15
  4. Write new model prefix (does not overwrite sample joint_41000)

Usage:
  PYTHONPATH=. python3 src/tokenizer/train_full_joint.py
  PYTHONPATH=. python3 src/tokenizer/train_full_joint.py --vocab-size 41000
"""

import json
import subprocess
import sys
from pathlib import Path

from src.tokenizer.prepare_spm_corpus import dedupe_text_file, run as prepare_corpus
from src.tokenizer.train import MODEL_DIR, TRAIN_PROFILES


CORPUS_DIR = Path('data/external')
BASE_JOINT = CORPUS_DIR / 'spm_corpus_legal_v2_joint.txt'
DEFAULT_VOCAB = 41000

# Profiles to try for "as full as possible" on 16GB (Unigram)
FULL_ATTEMPTS = [
    ('full', 'sentencepiece_legal_v2_joint_full_{vocab}'),
    ('full_tight', 'sentencepiece_legal_v2_joint_fulltight_{vocab}'),
    ('full_sample_15', 'sentencepiece_legal_v2_joint_fullsamp15_{vocab}'),
]


def ensure_base_joint(verbose: bool = True) -> Path:
    if BASE_JOINT.exists() and BASE_JOINT.stat().st_size > 0:
        if verbose:
            print(f'Using base joint corpus: {BASE_JOINT}')
        return BASE_JOINT
    prepare_corpus(mode='joint', verbose=verbose)
    return BASE_JOINT


def build_deduped(
    max_chars: int | None = 4096,
    verbose: bool = True,
) -> tuple[Path, dict]:
    ensure_base_joint(verbose=verbose)
    suffix = 'dedup' if max_chars is None else f'dedup_c{max_chars}'
    out = CORPUS_DIR / f'spm_corpus_legal_v2_joint_{suffix}.txt'
    if out.exists() and out.stat().st_size > 0:
        if verbose:
            print(f'Using existing deduped corpus: {out}')
        report = out.with_name(out.stem + '_dedupe_report.json')
        stats = json.loads(report.read_text()) if report.exists() else {}
        return out, stats
    stats = dedupe_text_file(BASE_JOINT, out, max_chars=max_chars, verbose=verbose)
    return out, stats


def _train_subprocess(
    corpus: Path,
    vocab_size: int,
    model_prefix: str,
    profile: str,
) -> int:
    """Run train in a child process so OOM kill does not kill the parent."""
    code = f"""
from src.tokenizer.train import train
train(
    {str(corpus)!r},
    {vocab_size},
    model_prefix={model_prefix!r},
    profile={profile!r},
    model_type='unigram',
)
"""
    env = {**dict(**__import__('os').environ), 'PYTHONPATH': '.'}
    proc = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
    )
    return proc.returncode


def run(
    vocab_size: int = DEFAULT_VOCAB,
    max_chars: int | None = 4096,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    corpus, dstats = build_deduped(max_chars=max_chars, verbose=verbose)
    report: dict = {
        'corpus': str(corpus),
        'dedupe': dstats,
        'vocab_size': vocab_size,
        'attempts': [],
        'winner': None,
    }

    for profile, prefix_tmpl in FULL_ATTEMPTS:
        prefix = prefix_tmpl.format(vocab=vocab_size)
        out = MODEL_DIR / f'{prefix}.model'
        if out.exists() and out.stat().st_size > 0 and not force:
            if verbose:
                print(f'Skip existing {out}')
            report['attempts'].append(
                {'profile': profile, 'prefix': prefix, 'status': 'exists', 'path': str(out)}
            )
            report['winner'] = {
                'profile': profile,
                'prefix': prefix,
                'path': str(out),
                'status': 'exists',
            }
            break

        if verbose:
            print(f'\n=== Attempt profile={profile} prefix={prefix} ===')
            print(f'    corpus={corpus} opts={TRAIN_PROFILES[profile]}')

        rc = _train_subprocess(corpus, vocab_size, prefix, profile)
        attempt = {
            'profile': profile,
            'prefix': prefix,
            'returncode': rc,
            'path': str(out),
        }
        if rc == 0 and out.exists():
            attempt['status'] = 'ok'
            report['attempts'].append(attempt)
            report['winner'] = {
                'profile': profile,
                'prefix': prefix,
                'path': str(out),
                'status': 'trained',
            }
            if verbose:
                print(f'SUCCESS: {out}')
            break
        attempt['status'] = 'failed_or_oom'
        report['attempts'].append(attempt)
        if verbose:
            print(f'FAILED profile={profile} rc={rc} (OOM or error); trying next')
        # clean partial artifacts
        for ext in ('.model', '.vocab'):
            p = MODEL_DIR / f'{prefix}{ext}'
            if p.exists() and p.stat().st_size == 0:
                p.unlink()

    out_report = CORPUS_DIR / f'spm_full_joint_{vocab_size}_report.json'
    out_report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if verbose:
        print(f'\nReport: {out_report}')
        if report['winner']:
            print(f'Winner: {report["winner"]}')
        else:
            print('No full-joint model trained; keep sample joint_41000')
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Full-as-possible joint Unigram SPM on 16GB')
    parser.add_argument('--vocab-size', type=int, default=DEFAULT_VOCAB)
    parser.add_argument(
        '--max-chars',
        type=int,
        default=4096,
        help='Truncate lines to this many chars when deduping (0 = no truncate)',
    )
    parser.add_argument('--force', action='store_true', help='Retrain even if model exists')
    args = parser.parse_args()
    max_chars = None if args.max_chars == 0 else args.max_chars
    run(vocab_size=args.vocab_size, max_chars=max_chars, force=args.force)


if __name__ == '__main__':
    main()

"""Parallel matrix trainer for tokenizer configs.

- Trains each config as a subprocess so OOM in one doesn't kill others.
- Skips configs whose model file already exists (idempotent).
- Writes per-config train_report.json + a top-level manifest.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.tokenizer.matrix_configs import (
    MODEL_DIR,
    TokenizerConfig,
    phase1_configs,
    phase2_configs,
)


MANIFEST_PATH = Path('data/analysis/tokenizer_matrix_manifest.json')


def _train_one(cfg: TokenizerConfig) -> dict:
    out_model = cfg.model_path()
    result: dict = {
        'name': cfg.name(),
        'model_type': cfg.model_type,
        'vocab_size': cfg.vocab_size,
        'corpus_key': cfg.corpus_key,
        'options': {
            'character_coverage': cfg.character_coverage,
            'byte_fallback': cfg.byte_fallback,
            'split_digits': cfg.split_digits,
            'split_by_unicode_script': cfg.split_by_unicode_script,
            'normalization': cfg.normalization,
            'user_defined_symbols': list(cfg.user_defined_symbols),
        },
        'path': str(out_model),
    }
    if out_model.exists() and out_model.stat().st_size > 0:
        result['status'] = 'skip_exists'
        result['elapsed_s'] = 0.0
        return result

    corpus = cfg.corpus_path()
    if not corpus.exists():
        result['status'] = 'error_missing_corpus'
        return result

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    uds_list = list(cfg.user_defined_symbols)
    code = f"""
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input={str(corpus)!r},
    model_prefix={str(MODEL_DIR / cfg.model_prefix())!r},
    vocab_size={cfg.vocab_size},
    character_coverage={cfg.character_coverage},
    model_type={cfg.model_type!r},
    normalization_rule_name={cfg.normalization!r},
    byte_fallback={cfg.byte_fallback},
    split_digits={cfg.split_digits},
    split_by_unicode_script={cfg.split_by_unicode_script},
    user_defined_symbols={uds_list!r},
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    train_extremely_large_corpus=True,
    input_sentence_size={cfg.input_sentence_size},
    seed_sentencepiece_size={cfg.seed_sentencepiece_size},
    max_sentence_length={cfg.max_sentence_length},
    num_threads={cfg.num_threads},
    shuffle_input_sentence=True,
)
"""
    env = {**os.environ, 'PYTHONPATH': '.'}
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, '-c', code],
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = round(time.time() - t0, 1)
    if proc.returncode == 0 and out_model.exists():
        result['status'] = 'trained'
        result['elapsed_s'] = elapsed
        result['model_size_bytes'] = out_model.stat().st_size
    else:
        result['status'] = f'error_rc{proc.returncode}'
        result['elapsed_s'] = elapsed
        result['stderr_tail'] = proc.stderr.splitlines()[-30:] if proc.stderr else []
    return result


def run(configs: list[TokenizerConfig], parallel: int = 4, verbose: bool = True) -> list[dict]:
    manifest: list[dict] = []
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8')).get('runs', [])
        except json.JSONDecodeError:
            manifest = []
    manifest_by_name = {r['name']: r for r in manifest}

    todo = [c for c in configs if manifest_by_name.get(c.name(), {}).get('status') != 'trained']
    if verbose:
        print(f'Matrix: {len(configs)} configs total; {len(todo)} to train (skipping cached).')
    if not todo:
        return manifest

    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(_train_one, c) for c in todo]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            manifest_by_name[r['name']] = r
            if verbose:
                print(f'  [{r["status"]:14}] {r["name"]}  ({r.get("elapsed_s", 0)}s)')
            MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST_PATH.write_text(
                json.dumps(
                    {'runs': list(manifest_by_name.values())},
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
    return list(manifest_by_name.values())


def _load_top_names(top_from: Path | None, top_k: int, filter_substr: str = 'v2_joint') -> list[str]:
    """Rank by HI c/t; filter to MT-usable corpora (v2_joint by default; v2_hi fragments EN)."""
    if not top_from or not top_from.exists():
        return []
    data = json.loads(top_from.read_text(encoding='utf-8'))
    results = data.get('results', [])
    if filter_substr:
        results = [r for r in results if filter_substr in r['name']]
    ranked = sorted(results, key=lambda r: r.get('hi_chars_per_tok', 0.0), reverse=True)
    return [r['name'] for r in ranked[:top_k]]


def main():
    p = argparse.ArgumentParser(description='Tokenizer matrix trainer')
    p.add_argument('--phase', choices=['1', '2'], required=True)
    p.add_argument('--parallel', type=int, default=4, help='Concurrent training jobs')
    p.add_argument(
        '--top-from',
        type=Path,
        default=Path('data/analysis/tokenizer_matrix.json'),
        help='Phase-1 bench JSON to pick top-N base configs for Phase 2',
    )
    p.add_argument('--top-k', type=int, default=3)
    a = p.parse_args()

    if a.phase == '1':
        cfgs = phase1_configs()
    else:
        p1_all = {c.name(): c for c in phase1_configs()}
        top_names = _load_top_names(a.top_from, a.top_k)
        top_cfgs = [p1_all[n] for n in top_names if n in p1_all]
        if not top_cfgs:
            raise SystemExit('phase 2: no top-N configs from bench JSON; run phase 1 + bench first')
        cfgs = phase2_configs(top_cfgs)

    run(cfgs, parallel=a.parallel)


if __name__ == '__main__':
    main()

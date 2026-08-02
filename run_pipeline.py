"""
Orchestrate Adalat AI phases that exist as runnable modules.

Phases (see docs/WALKTHROUGH.md):
  preprocess  -- assignment PDF -> train/dev/test
  external    -- Stage A legal bitext + dual-policy split
  tokenizer   -- benches (models must already exist)
  eval_smoke  -- zero-shot NLLB smoke on dual policies
  train_smoke -- short NLLB LoRA smoke (needs Stage A subsample data)

Usage:
  python run_pipeline.py --steps preprocess
  python run_pipeline.py --steps external
  python run_pipeline.py --steps tokenizer
  python run_pipeline.py --list
  python run_pipeline.py --steps all
"""

import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable
ROOT = Path(__file__).parent

# Each step is one real module. Groups only expand to step names (flat).
STEPS = {
    # Phase 1 -- assignment preprocessing
    'reextract': {
        'module': 'src.preprocessing.reextract_pdfs',
        'args': ['--all'],
        'desc': 'Re-extract Hindi PDFs with Tesseract OCR',
    },
    'reextract_compare': {
        'module': 'src.preprocessing.reextract_pdfs',
        'args': ['--compare-all'],
        'desc': 'Compare re-extracted vs originals',
    },
    'join': {
        'module': 'src.preprocessing.join_lines',
        'args': [],
        'desc': 'Join hard-wrapped English lines',
    },
    'join_hi': {
        'module': 'src.preprocessing.join_hindi_lines',
        'args': [],
        'desc': 'Join hard-wrapped Hindi OCR lines (danda-aware)',
    },
    'segment': {
        'module': 'src.preprocessing.segment_sentences',
        'args': [],
        'desc': 'Segment sentences (EN + HI)',
    },
    'align': {
        'module': 'src.preprocessing.align_sentences',
        'args': [],
        'desc': 'LaBSE alignment + quality filters',
    },
    'output': {
        'module': 'src.preprocessing.output_format',
        'args': [],
        'desc': 'Train/dev/test splits + metadata',
    },
    # Phase 1b -- external Stage A
    'external_download': {
        'module': 'src.preprocessing.ingest_external_parallel',
        'args': ['--download'],
        'desc': 'Download MILPaC + Anuvaad legal EN-HI raw files',
    },
    'external_ingest': {
        'module': 'src.preprocessing.ingest_external_parallel',
        'args': [],
        'desc': 'Ingest external legal EN-HI to Stage A JSONL',
    },
    'external_eval_split': {
        'module': 'src.preprocessing.split_external_eval',
        'args': [],
        'desc': 'Carve Policy E held-out + stage_a_train',
    },
    'eval_sets': {
        'module': 'src.evaluation.eval_sets',
        'args': [],
        'desc': 'Validate dual-policy suites (I + E)',
    },
    # Phase 2 -- tokenizer
    'tokenizer_bench': {
        'module': 'src.tokenizer.benchmark',
        'args': [],
        'desc': 'Benchmark tokenizers on aligned corpus',
    },
    'tokenizer_deep_dive': {
        'module': 'src.tokenizer.deep_dive',
        'args': [],
        'desc': 'Byte-fallback and Devanagari merge analysis',
    },
    # Phase 3/4 smokes (full H200 curricula stay on Makefile)
    'zero_shot_smoke': {
        'module': 'src.evaluation.zero_shot_nllb',
        'args': ['--max-pairs', '20', '--suites', 'I_test,E_milpac_test'],
        'desc': 'Zero-shot NLLB smoke (20 pairs per suite)',
    },
    'train_nllb_smoke': {
        'module': 'src.training.train_nllb_lora',
        'args': ['--curriculum', 'smoke', '--max-steps', '20', '--skip-gen-eval'],
        'desc': 'NLLB LoRA smoke (20 steps; needs Stage A data)',
    },
}

GROUPS = {
    'preprocess': ['reextract', 'join', 'join_hi', 'segment', 'align', 'output'],
    'external': ['external_ingest', 'external_eval_split', 'eval_sets'],
    'external_full': [
        'external_download',
        'external_ingest',
        'external_eval_split',
        'eval_sets',
    ],
    'tokenizer': ['tokenizer_bench'],
    'eval_smoke': ['zero_shot_smoke'],
    'train_smoke': ['train_nllb_smoke'],
    # Data path only (no multi-hour train). Matches assignment reproducibility.
    'all': [
        'reextract',
        'join',
        'join_hi',
        'segment',
        'align',
        'output',
        'external_ingest',
        'external_eval_split',
        'eval_sets',
        'tokenizer_bench',
    ],
}


def expand_steps(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        if name in GROUPS:
            out.extend(GROUPS[name])
        elif name in STEPS:
            out.append(name)
        else:
            print(f'Unknown step/group: {name}')
            print(f'Steps: {", ".join(STEPS)}')
            print(f'Groups: {", ".join(GROUPS)}')
            sys.exit(1)
    return out


def list_pipeline():
    print('Steps:')
    for name, step in STEPS.items():
        print(f'  {name:<22} {step["desc"]}')
    print('\nGroups:')
    for name, members in GROUPS.items():
        print(f'  {name:<22} {", ".join(members)}')


def run_step(name: str):
    step = STEPS[name]
    cmd = [PYTHON, '-m', step['module']] + step['args']
    print(f'\n{"=" * 60}')
    print(f'Step: {name} -- {step["desc"]}')
    print(f'Command: {" ".join(cmd)}')
    print(f'{"=" * 60}\n')
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f'FAILED: {name}')
        sys.exit(1)
    print(f'OK: {name}')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run Adalat AI pipeline steps')
    parser.add_argument(
        '--steps',
        default='preprocess',
        help='Comma-separated steps or groups (see --list)',
    )
    parser.add_argument('--list', action='store_true', help='List steps and groups')
    args = parser.parse_args()

    if args.list:
        list_pipeline()
        return

    requested = [s.strip() for s in args.steps.split(',') if s.strip()]
    for step in expand_steps(requested):
        run_step(step)

    print('\nAll steps completed successfully.')


if __name__ == '__main__':
    main()

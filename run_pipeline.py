"""
End-to-end orchestration of the Adalat AI pipeline steps that exist today.

Usage:
    python run_pipeline.py --steps all
    python run_pipeline.py --steps preprocess
    python run_pipeline.py --steps external
    python run_pipeline.py --steps align,output
    python run_pipeline.py --steps tokenizer_bench
"""

import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable
ROOT = Path(__file__).parent

STEPS = {
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
}

# Groups expand to step names only (flat). Nested group names are expanded once.
GROUPS = {
    'preprocess': ['reextract', 'join', 'segment', 'align', 'output'],
    'external': ['external_ingest'],
    'external_full': ['external_download', 'external_ingest'],
    'tokenizer': ['tokenizer_bench'],
    # assignment preprocess + external Stage A + tokenizer bench
    'all': [
        'reextract',
        'join',
        'segment',
        'align',
        'output',
        'external_ingest',
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
        help=(
            'Comma-separated steps or group: preprocess, external, external_full, tokenizer, all'
        ),
    )
    args = parser.parse_args()

    requested = [s.strip() for s in args.steps.split(',') if s.strip()]
    steps_to_run = expand_steps(requested)

    for step in steps_to_run:
        run_step(step)

    print('\nAll steps completed successfully.')


if __name__ == '__main__':
    main()

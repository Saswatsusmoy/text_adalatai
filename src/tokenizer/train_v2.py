"""
Track C0: train legal SentencePiece v2 grid.

Does not overwrite v1 models (sentencepiece_{16,32,41}k.*).
Writes sentencepiece_legal_v2_{hi|joint}_{vocab}.model
"""

from pathlib import Path

from src.tokenizer.prepare_spm_corpus import run as prepare_corpus
from src.tokenizer.train import train

CORPUS_DIR = Path('data/external')

# (mode, vocab_size) — joint 41k is primary; hi ablations + joint 32k
DEFAULT_GRID = [
    ('hi', 32000),
    ('hi', 41000),
    ('joint', 32000),
    ('joint', 41000),
]


def model_prefix(mode: str, vocab_size: int) -> str:
    return f'sentencepiece_legal_v2_{mode}_{vocab_size}'


def ensure_corpus(mode: str, prarabdha_frac: float = 0.0, verbose: bool = True) -> Path:
    path = CORPUS_DIR / f'spm_corpus_legal_v2_{mode}.txt'
    if path.exists() and path.stat().st_size > 0:
        if verbose:
            print(f'Using existing corpus: {path}')
        return path
    prepare_corpus(mode=mode, include_prarabdha_frac=prarabdha_frac, verbose=verbose)
    if not path.exists():
        raise FileNotFoundError(f'corpus missing after prepare: {path}')
    return path


def run_grid(
    grid: list[tuple[str, int]] | None = None,
    prarabdha_frac: float = 0.0,
    verbose: bool = True,
) -> list[Path]:
    grid = grid or DEFAULT_GRID
    modes_needed = sorted({m for m, _ in grid})
    for mode in modes_needed:
        ensure_corpus(mode, prarabdha_frac=prarabdha_frac, verbose=verbose)

    trained: list[Path] = []
    from src.tokenizer.train import MODEL_DIR

    for mode, vocab in grid:
        corpus = CORPUS_DIR / f'spm_corpus_legal_v2_{mode}.txt'
        prefix = model_prefix(mode, vocab)
        out = MODEL_DIR / f'{prefix}.model'
        if out.exists() and out.stat().st_size > 0:
            if verbose:
                print(f'\n=== Skip existing {out} ===')
            trained.append(out)
            continue
        if verbose:
            print(f'\n=== Train {prefix} ===')
        path = train(corpus, vocab, model_prefix=prefix)
        trained.append(path)
    return trained


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train legal SPM v2 grid (Track C0)')
    parser.add_argument(
        '--prarabdha-frac',
        type=float,
        default=0.0,
        help='Optional Prarabdha mix when building missing corpora',
    )
    parser.add_argument(
        '--only',
        default='',
        help='Optional single run as mode:vocab (e.g. joint:41000)',
    )
    args = parser.parse_args()

    grid = DEFAULT_GRID
    if args.only:
        mode, vs = args.only.split(':')
        grid = [(mode, int(vs))]

    paths = run_grid(grid=grid, prarabdha_frac=args.prarabdha_frac, verbose=True)
    print('\nTrained:')
    for p in paths:
        print(f'  {p}')


if __name__ == '__main__':
    main()

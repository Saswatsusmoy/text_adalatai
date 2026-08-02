"""
Train SentencePiece tokenizer on a text corpus.

Saves .model and .vocab files to data/models/tokenizers/.
"""

from pathlib import Path

import sentencepiece as spm


MODEL_DIR = Path('data/models/tokenizers')

# Memory profiles for 16GB-class machines. 'sample' is the safe default for huge joint dumps.
TRAIN_PROFILES: dict[str, dict] = {
    'sample': {
        'train_extremely_large_corpus': True,
        'input_sentence_size': 1_000_000,
        'seed_sentencepiece_size': 500_000,
        'max_sentence_length': 8192,
        'num_threads': 4,
    },
    # After exact-line dedupe: try all lines, smaller SA seed
    'full': {
        'train_extremely_large_corpus': True,
        'input_sentence_size': 0,
        'seed_sentencepiece_size': 250_000,
        'max_sentence_length': 4096,
        'num_threads': 4,
    },
    # Tighter peak if 'full' OOMs
    'full_tight': {
        'train_extremely_large_corpus': True,
        'input_sentence_size': 0,
        'seed_sentencepiece_size': 150_000,
        'max_sentence_length': 2048,
        'num_threads': 2,
    },
    # Cover more than 1M if full still fails
    'full_sample_15': {
        'train_extremely_large_corpus': True,
        'input_sentence_size': 1_500_000,
        'seed_sentencepiece_size': 250_000,
        'max_sentence_length': 4096,
        'num_threads': 4,
    },
}


def train(
    text_path: str | Path,
    vocab_size: int,
    model_prefix: str | None = None,
    profile: str = 'sample',
    model_type: str = 'unigram',
    split_by_unicode_script: bool = False,
) -> Path:
    if model_prefix is None:
        model_prefix = f'sentencepiece_{vocab_size}'
    if profile not in TRAIN_PROFILES:
        raise ValueError(f'unknown profile {profile}; choose from {list(TRAIN_PROFILES)}')
    if model_type not in ('unigram', 'bpe'):
        raise ValueError('model_type must be unigram or bpe')

    model_path = MODEL_DIR / model_prefix
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    opts = dict(TRAIN_PROFILES[profile])
    # Matrix family (and this project's preference) keeps mixed-script pieces
    # intact; SPM's default is True, so it must be set explicitly to stay
    # consistent with the 35-config matrix (DESIGN 33). Note: SPM's TrainerSpec
    # has NO seed field -- trainer randomness is not user-controllable.
    opts['split_by_unicode_script'] = split_by_unicode_script

    spm.SentencePieceTrainer.train(
        input=str(text_path),
        model_prefix=str(model_path),
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        shuffle_input_sentence=True,
        **opts,
    )

    model_file = model_path.with_suffix('.model')
    print(f'Trained: {model_file} (vocab={vocab_size} profile={profile} type={model_type})')
    return model_file


def load(model_path: str | Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def encode_text(sp: spm.SentencePieceProcessor, texts: list[str]) -> list[list[int]]:
    return [sp.encode(t) for t in texts]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train SentencePiece tokenizer')
    parser.add_argument('--input', required=True, help='Path to training text file')
    parser.add_argument('--vocab-size', type=int, default=16000, help='Vocabulary size')
    parser.add_argument(
        '--model-prefix',
        default=None,
        help='Output name under data/models/tokenizers/ (default sentencepiece_{vocab})',
    )
    parser.add_argument(
        '--profile',
        default='sample',
        choices=sorted(TRAIN_PROFILES),
        help='Memory/coverage profile (sample | full | full_tight | full_sample_15)',
    )
    parser.add_argument(
        '--model-type',
        default='unigram',
        choices=['unigram', 'bpe'],
        help='SentencePiece model type (unigram preferred for Indic)',
    )
    parser.add_argument(
        '--split-by-unicode-script',
        action='store_true',
        help='Force EN/HI script boundary splits (default: False, keeps mixed-script pieces)',
    )
    args = parser.parse_args()
    train(
        args.input,
        args.vocab_size,
        model_prefix=args.model_prefix,
        profile=args.profile,
        model_type=args.model_type,
        split_by_unicode_script=args.split_by_unicode_script,
    )

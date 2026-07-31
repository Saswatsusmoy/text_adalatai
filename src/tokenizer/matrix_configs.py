"""Config dataclass + preset builders for the tokenizer matrix experiment.

Phase 1 (main matrix): {unigram, bpe} x {16k, 32k, 41k, 48k, 64k} x {v2_joint, v2_hi}.
Phase 2 (secondary-axis ablation): apply one toggle at a time to top-N Phase 1 configs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


CORPUS_ROOT = Path('data/external')
MODEL_DIR = Path('data/models/tokenizers')

# Corpus keys -> file paths (use deduped joint by default for speed + memory)
CORPUS_PATHS: dict[str, Path] = {
    'v2_joint': CORPUS_ROOT / 'spm_corpus_legal_v2_joint_dedup_c4096.txt',
    'v2_hi': CORPUS_ROOT / 'spm_corpus_legal_v2_hi.txt',
}

# Legal-domain protected symbols for Phase 2 UDS ablation.
# EN + HI; kept short so unigram doesn't overfit them.
LEGAL_PROTECTED_SYMBOLS: tuple[str, ...] = (
    'Section',
    'Article',
    'Order',
    'Rule',
    'Writ',
    'Petition',
    'Appellant',
    'Respondent',
    'SLP',
    'WP',
    'S.C.R.',
    'INSC',
    'अनुच्छेद',
    'धारा',
    'आदेश',
    'नियम',
    'रिट',
    'याचिका',
    'अपीलकर्ता',
    'प्रतिवादी',
    'न्यायालय',
    'अधिवक्ता',
)


@dataclass(frozen=True)
class TokenizerConfig:
    model_type: str  # 'unigram' | 'bpe'
    vocab_size: int
    corpus_key: str  # key into CORPUS_PATHS
    character_coverage: float = 1.0
    byte_fallback: bool = False
    split_digits: bool = False
    split_by_unicode_script: bool = False
    normalization: str = 'nmt_nfkc'
    max_sentence_length: int = 4096
    user_defined_symbols: tuple[str, ...] = field(default_factory=tuple)
    seed_sentencepiece_size: int = 250_000
    input_sentence_size: int = 0  # 0 = use all lines
    num_threads: int = 8
    seed: int = 42

    def opts_tag(self) -> str:
        parts = []
        if self.byte_fallback:
            parts.append('bf')
        if self.split_digits:
            parts.append('sd')
        if self.split_by_unicode_script:
            parts.append('sus')
        if self.character_coverage != 1.0:
            parts.append(f'cc{self.character_coverage}')
        if self.normalization != 'nmt_nfkc':
            parts.append(f'norm-{self.normalization}')
        if self.user_defined_symbols:
            parts.append(f'uds{len(self.user_defined_symbols)}')
        return '_'.join(parts)

    def name(self) -> str:
        base = f'legal_v2_{self.corpus_key}_{self.model_type}_{self.vocab_size}'
        tag = self.opts_tag()
        return f'{base}_{tag}' if tag else base

    def model_prefix(self) -> str:
        return f'sentencepiece_{self.name()}'

    def model_path(self) -> Path:
        return MODEL_DIR / f'{self.model_prefix()}.model'

    def corpus_path(self) -> Path:
        return CORPUS_PATHS[self.corpus_key]


VOCAB_LADDER: tuple[int, ...] = (16000, 32000, 41000, 48000, 64000)
MODEL_TYPES: tuple[str, ...] = ('unigram', 'bpe')
CORPUS_KEYS: tuple[str, ...] = ('v2_joint', 'v2_hi')


def phase1_configs() -> list[TokenizerConfig]:
    """20 configs = 2 model_types x 5 vocabs x 2 corpora (all defaults)."""
    out: list[TokenizerConfig] = []
    for corpus_key in CORPUS_KEYS:
        for mt in MODEL_TYPES:
            for vs in VOCAB_LADDER:
                out.append(TokenizerConfig(model_type=mt, vocab_size=vs, corpus_key=corpus_key))
    return out


def phase2_configs(top_configs: list[TokenizerConfig]) -> list[TokenizerConfig]:
    """For each top Phase 1 config, emit 5 single-axis toggles."""
    out: list[TokenizerConfig] = []
    for base in top_configs:
        # byte_fallback = True
        out.append(_with(base, byte_fallback=True))
        # split_digits = True
        out.append(_with(base, split_digits=True))
        # split_by_unicode_script = True
        out.append(_with(base, split_by_unicode_script=True))
        # character_coverage = 0.9995
        out.append(_with(base, character_coverage=0.9995))
        # user_defined_symbols = legal protected list
        out.append(_with(base, user_defined_symbols=LEGAL_PROTECTED_SYMBOLS))
    return out


def _with(base: TokenizerConfig, **overrides) -> TokenizerConfig:
    fields_ = {
        'model_type': base.model_type,
        'vocab_size': base.vocab_size,
        'corpus_key': base.corpus_key,
        'character_coverage': base.character_coverage,
        'byte_fallback': base.byte_fallback,
        'split_digits': base.split_digits,
        'split_by_unicode_script': base.split_by_unicode_script,
        'normalization': base.normalization,
        'max_sentence_length': base.max_sentence_length,
        'user_defined_symbols': base.user_defined_symbols,
        'seed_sentencepiece_size': base.seed_sentencepiece_size,
        'input_sentence_size': base.input_sentence_size,
        'num_threads': base.num_threads,
        'seed': base.seed,
    }
    fields_.update(overrides)
    return TokenizerConfig(**fields_)

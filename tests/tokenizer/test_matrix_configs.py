"""Tests for matrix config dataclass and preset builders."""

from src.tokenizer.matrix_configs import (
    CORPUS_PATHS,
    LEGAL_PROTECTED_SYMBOLS,
    MODEL_DIR,
    TokenizerConfig,
    _with,
    phase1_configs,
    phase2_configs,
)


class TestTokenizerConfig:
    def test_default_split_by_unicode_script_false(self):
        c = TokenizerConfig(model_type='unigram', vocab_size=41000, corpus_key='v2_joint')
        assert c.split_by_unicode_script is False

    def test_name_is_deterministic(self):
        c = TokenizerConfig(model_type='bpe', vocab_size=64000, corpus_key='v2_joint')
        assert c.name() == c.name()
        assert '64000' in c.name()
        assert 'v2_joint' in c.name()
        assert 'bpe' in c.name()

    def test_opts_tag_empty_on_defaults(self):
        c = TokenizerConfig(model_type='unigram', vocab_size=41000, corpus_key='v2_joint')
        assert c.opts_tag() == ''

    def test_opts_tag_encodes_toggles(self):
        c = TokenizerConfig(
            model_type='bpe',
            vocab_size=64000,
            corpus_key='v2_joint',
            byte_fallback=True,
            split_digits=True,
        )
        tag = c.opts_tag()
        assert 'bf' in tag
        assert 'sd' in tag

    def test_model_prefix_uses_sentencepiece_prefix(self):
        c = TokenizerConfig(model_type='unigram', vocab_size=41000, corpus_key='v2_joint')
        assert c.model_prefix().startswith('sentencepiece_')
        assert c.model_path().suffix == '.model'

    def test_corpus_path_resolves(self):
        c = TokenizerConfig(model_type='unigram', vocab_size=41000, corpus_key='v2_joint')
        assert c.corpus_path() == CORPUS_PATHS['v2_joint']

    def test_seed_is_informational_only(self):
        # SPM TrainerSpec has no seed field; the field exists but must not be passed to SPM.
        c = TokenizerConfig(model_type='unigram', vocab_size=41000, corpus_key='v2_joint')
        assert c.seed == 42


class TestPresets:
    def test_phase1_is_20_configs(self):
        configs = phase1_configs()
        assert len(configs) == 20

    def test_phase1_covers_grid(self):
        configs = phase1_configs()
        keys = {(c.model_type, c.vocab_size, c.corpus_key) for c in configs}
        assert ('unigram', 41000, 'v2_joint') in keys
        assert ('bpe', 64000, 'v2_joint') in keys
        assert ('unigram', 16000, 'v2_hi') in keys

    def test_phase1_all_default_sus_false(self):
        assert all(c.split_by_unicode_script is False for c in phase1_configs())

    def test_phase2_emits_5_toggles_per_base(self):
        base = [TokenizerConfig(model_type='bpe', vocab_size=64000, corpus_key='v2_joint')]
        configs = phase2_configs(base)
        assert len(configs) == 5
        flags = [c.opts_tag() for c in configs]
        assert 'bf' in flags
        assert 'sd' in flags
        assert 'sus' in flags
        assert 'cc0.9995' in flags
        assert 'uds22' in flags

    def test_phase2_preserves_base_fields(self):
        base = [TokenizerConfig(model_type='bpe', vocab_size=64000, corpus_key='v2_joint')]
        toggled = phase2_configs(base)[0]
        assert toggled.model_type == 'bpe'
        assert toggled.vocab_size == 64000
        assert toggled.corpus_key == 'v2_joint'

    def test_with_overrides_single_field(self):
        base = TokenizerConfig(model_type='bpe', vocab_size=64000, corpus_key='v2_joint')
        out = _with(base, byte_fallback=True)
        assert out.byte_fallback is True
        assert out.vocab_size == 64000
        assert out.split_digits is False


class TestConstants:
    def test_legal_protected_symbols_nonempty(self):
        assert len(LEGAL_PROTECTED_SYMBOLS) >= 20
        assert 'Section' in LEGAL_PROTECTED_SYMBOLS
        assert 'न्यायालय' in LEGAL_PROTECTED_SYMBOLS

    def test_corpus_paths_point_to_external(self):
        for key, path in CORPUS_PATHS.items():
            assert 'data/external' in str(path)
            assert key in ('v2_joint', 'v2_hi')

    def test_model_dir_is_tokenizers(self):
        assert str(MODEL_DIR).endswith('tokenizers')

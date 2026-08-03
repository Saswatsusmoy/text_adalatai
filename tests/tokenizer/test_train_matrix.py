"""Tests for the tokenizer matrix trainer (manifest resume + top-name ranking)."""

import json

from src.tokenizer import train_matrix as tm
from src.tokenizer.matrix_configs import TokenizerConfig


class TestLoadTopNames:
    def test_ranks_by_hi_chars_per_tok(self, tmp_path):
        f = tmp_path / 'bench.json'
        f.write_text(
            json.dumps(
                {
                    'results': [
                        {'name': 'a_v2_joint_16000', 'hi_chars_per_tok': 4.2},
                        {'name': 'b_v2_joint_64000', 'hi_chars_per_tok': 4.7},
                        {'name': 'c_v2_joint_41000', 'hi_chars_per_tok': 4.5},
                    ]
                }
            ),
            encoding='utf-8',
        )
        names = tm._load_top_names(f, top_k=2)
        assert names == ['b_v2_joint_64000', 'c_v2_joint_41000']

    def test_filters_to_joint(self, tmp_path):
        f = tmp_path / 'bench.json'
        f.write_text(
            json.dumps(
                {
                    'results': [
                        {'name': 'a_v2_joint_64000', 'hi_chars_per_tok': 4.7},
                        {'name': 'b_v2_hi_64000', 'hi_chars_per_tok': 4.9},
                    ]
                }
            ),
            encoding='utf-8',
        )
        names = tm._load_top_names(f, top_k=5)
        assert names == ['a_v2_joint_64000']  # v2_hi excluded (EN-fragmenting)

    def test_missing_file_returns_empty(self, tmp_path):
        assert tm._load_top_names(tmp_path / 'nope.json', top_k=3) == []

    def test_empty_results(self, tmp_path):
        f = tmp_path / 'bench.json'
        f.write_text(json.dumps({'results': []}), encoding='utf-8')
        assert tm._load_top_names(f, top_k=3) == []


class TestRunManifestResume:
    def test_skips_cached_trained(self, tmp_path, monkeypatch):
        manifest = tmp_path / 'manifest.json'
        manifest.write_text(
            json.dumps(
                {
                    'runs': [
                        {
                            'name': 'legal_v2_v2_joint_unigram_41000',
                            'status': 'trained',
                            'elapsed_s': 1.0,
                        }
                    ]
                }
            ),
            encoding='utf-8',
        )
        monkeypatch.setattr(tm, 'MANIFEST_PATH', manifest)

        configs = [TokenizerConfig(model_type='unigram', vocab_size=41000, corpus_key='v2_joint')]
        # Cached 'trained' entry -> run() returns it without spawning any training.
        result = tm.run(configs, parallel=1, verbose=False)
        assert result[0]['status'] == 'trained'

    def test_bad_manifest_falls_back_to_pool(self, tmp_path, monkeypatch):
        # A corrupt manifest must not crash; run() falls back and attempts the
        # config. The outcome depends on whether the real model/corpus exists:
        # skip_exists (model file present) or error_missing_corpus (absent) --
        # either proves the stale cache was NOT honored and training was tried.
        manifest = tmp_path / 'manifest.json'
        manifest.write_text('not json{{', encoding='utf-8')
        monkeypatch.setattr(tm, 'MANIFEST_PATH', manifest)
        configs = [TokenizerConfig(model_type='unigram', vocab_size=41000, corpus_key='v2_joint')]
        result = tm.run(configs, parallel=1, verbose=False)
        assert len(result) == 1
        assert result[0]['status'] in ('skip_exists', 'error_missing_corpus')

"""Tests for the Track C0 v2 SPM grid trainer."""


import pytest

from src.tokenizer import train_v2 as tv


class TestModelPrefix:
    def test_format(self):
        assert tv.model_prefix('joint', 41000) == 'sentencepiece_legal_v2_joint_41000'
        assert tv.model_prefix('hi', 32000) == 'sentencepiece_legal_v2_hi_32000'


class TestEnsureCorpus:
    def test_uses_existing(self, tmp_path, monkeypatch):
        cdir = tmp_path / 'external'
        cdir.mkdir()
        (cdir / 'spm_corpus_legal_v2_joint.txt').write_text('x', encoding='utf-8')
        monkeypatch.setattr(tv, 'CORPUS_DIR', cdir)
        path = tv.ensure_corpus('joint', verbose=False)
        assert path.exists()

    def test_prepare_when_missing(self, tmp_path, monkeypatch):
        cdir = tmp_path / 'external'
        cdir.mkdir()
        monkeypatch.setattr(tv, 'CORPUS_DIR', cdir)

        def fake_prepare(mode, include_prarabdha_frac=0.0, verbose=True):
            (cdir / f'spm_corpus_legal_v2_{mode}.txt').write_text('x', encoding='utf-8')

        monkeypatch.setattr(tv, 'prepare_corpus', fake_prepare)
        path = tv.ensure_corpus('joint', verbose=False)
        assert path.exists()

    def test_raises_when_prepare_fails(self, tmp_path, monkeypatch):
        cdir = tmp_path / 'external'
        cdir.mkdir()
        monkeypatch.setattr(tv, 'CORPUS_DIR', cdir)
        monkeypatch.setattr(tv, 'prepare_corpus', lambda *a, **k: None)
        with pytest.raises(FileNotFoundError):
            tv.ensure_corpus('joint', verbose=False)


class TestRunGrid:
    def test_skips_existing_models(self, tmp_path, monkeypatch):
        from src.tokenizer import train as train_mod

        cdir = tmp_path / 'external'
        cdir.mkdir()
        (cdir / 'spm_corpus_legal_v2_joint.txt').write_text('x', encoding='utf-8')
        mdir = tmp_path / 'models'
        mdir.mkdir()
        (mdir / 'sentencepiece_legal_v2_joint_41000.model').write_text('x', encoding='utf-8')
        monkeypatch.setattr(tv, 'CORPUS_DIR', cdir)
        monkeypatch.setattr(train_mod, 'MODEL_DIR', mdir)

        trained = []
        monkeypatch.setattr(
            tv,
            'train',
            lambda corpus, vocab, model_prefix=None: (
                trained.append((str(corpus), vocab, model_prefix))
                or (mdir / f'{model_prefix}.model')
            ),
        )

        paths = tv.run_grid(grid=[('joint', 41000)], verbose=False)
        assert len(paths) == 1
        assert trained == []  # existing model skipped, train() not called

    def test_trains_missing_models(self, tmp_path, monkeypatch):
        from src.tokenizer import train as train_mod

        cdir = tmp_path / 'external'
        cdir.mkdir()
        (cdir / 'spm_corpus_legal_v2_joint.txt').write_text('x', encoding='utf-8')
        mdir = tmp_path / 'models'
        mdir.mkdir()
        monkeypatch.setattr(tv, 'CORPUS_DIR', cdir)
        monkeypatch.setattr(train_mod, 'MODEL_DIR', mdir)

        calls = []
        monkeypatch.setattr(
            tv,
            'train',
            lambda corpus, vocab, model_prefix=None: (
                calls.append((vocab, model_prefix)) or (mdir / f'{model_prefix}.model')
            ),
        )

        paths = tv.run_grid(grid=[('joint', 41000)], verbose=False)
        assert len(paths) == 1
        assert calls == [(41000, 'sentencepiece_legal_v2_joint_41000')]

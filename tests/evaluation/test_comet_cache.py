"""Unit tests for COMET cache invalidation (no comet package / model needed).

The full `run` flow is exercised with a fake `comet` module and a fake model
that just counts `predict` calls; the package is only imported inside `run`.
"""

import json
import sys
import types
from pathlib import Path

from src.evaluation import comet_score


class _FakeResult:
    system_score = 0.5


class _FakeModel:
    def __init__(self):
        self.predict_calls = 0

    def predict(self, data, batch_size=None, gpus=None):
        self.predict_calls += 1
        return _FakeResult()


def _install_fake_comet(monkeypatch, model):
    fake = types.ModuleType('comet')
    fake.download_model = lambda mid: 'ckpt'
    fake.load_from_checkpoint = lambda ckpt: model
    monkeypatch.setitem(sys.modules, 'comet', fake)


def _write_hyp(analysis_dir, tag='zero_shot_nllb', suite='I_test', body=None):
    path = Path(analysis_dir) / f'{tag}_{suite}_hyps.jsonl'
    body = body or '{"en_text": "src", "hi_text": "ref", "hyp_hi": "hyp"}'
    path.write_text(body, encoding='utf-8')
    return path


class TestShouldRescore:
    def test_missing_entry_rescores(self):
        assert comet_score.should_rescore(None, 'abc', 'm') is True
        assert comet_score.should_rescore({'score': None}, 'abc', 'm') is True

    def test_matching_entry_skips(self):
        entry = {'score': 0.5, 'fingerprint': 'abc', 'model_id': 'm'}
        assert comet_score.should_rescore(entry, 'abc', 'm') is False

    def test_fingerprint_change_rescores(self):
        entry = {'score': 0.5, 'fingerprint': 'abc', 'model_id': 'm'}
        assert comet_score.should_rescore(entry, 'abd', 'm') is True

    def test_model_change_rescores(self):
        entry = {'score': 0.5, 'fingerprint': 'abc', 'model_id': 'm'}
        assert comet_score.should_rescore(entry, 'abc', 'other') is True


class TestCacheSchema:
    def test_old_schema_ignored(self, tmp_path):
        p = tmp_path / 'comet22_summary.json'
        p.write_text(json.dumps({'model': 'm', 'systems': {'t': {'I_test': {'score': 0.5}}}}))
        assert comet_score.load_summary(p) == {}

    def test_v2_schema_loaded(self, tmp_path):
        p = tmp_path / 'comet22_summary.json'
        systems = {'t': {'I_test': {'score': 0.5, 'fingerprint': 'abc', 'model_id': 'm'}}}
        p.write_text(json.dumps({'schema': 'v2', 'model': 'm', 'systems': systems}))
        assert comet_score.load_summary(p) == systems


class TestRunCaching:
    def test_rescore_on_hyp_change(self, tmp_path, monkeypatch):
        model = _FakeModel()
        _install_fake_comet(monkeypatch, model)
        analysis = tmp_path / 'analysis'
        analysis.mkdir()
        path = _write_hyp(analysis)
        summary = tmp_path / 'comet22_summary.json'

        comet_score.run(model_id='m', analysis_dir=analysis, summary_path=summary, gpus=0)
        assert model.predict_calls == 1
        payload = json.loads(summary.read_text(encoding='utf-8'))
        assert payload['schema'] == 'v2'
        assert payload['systems']['zero_shot_nllb']['I_test']['fingerprint']

        comet_score.run(model_id='m', analysis_dir=analysis, summary_path=summary, gpus=0)
        assert model.predict_calls == 1

        path.write_text('{"en_text": "s2", "hi_text": "r2", "hyp_hi": "h2"}\n', encoding='utf-8')
        comet_score.run(model_id='m', analysis_dir=analysis, summary_path=summary, gpus=0)
        assert model.predict_calls == 2

    def test_model_change_rescores(self, tmp_path, monkeypatch):
        model = _FakeModel()
        _install_fake_comet(monkeypatch, model)
        analysis = tmp_path / 'analysis'
        analysis.mkdir()
        _write_hyp(analysis)
        summary = tmp_path / 'comet22_summary.json'

        comet_score.run(model_id='m', analysis_dir=analysis, summary_path=summary, gpus=0)
        comet_score.run(model_id='m2', analysis_dir=analysis, summary_path=summary, gpus=0)
        assert model.predict_calls == 2

    def test_old_schema_rescores_everything(self, tmp_path, monkeypatch):
        model = _FakeModel()
        _install_fake_comet(monkeypatch, model)
        analysis = tmp_path / 'analysis'
        analysis.mkdir()
        _write_hyp(analysis)
        summary = tmp_path / 'comet22_summary.json'
        summary.write_text(
            json.dumps({'model': 'm', 'systems': {'zero_shot_nllb': {'I_test': {'score': 0.5}}}})
        )
        comet_score.run(model_id='m', analysis_dir=analysis, summary_path=summary, gpus=0)
        assert model.predict_calls == 1


class TestParseHypPath:
    def test_parses_tag_and_suite(self):
        assert comet_score.parse_hyp_path(Path('data/analysis/zs_I_test_hyps.jsonl')) == (
            'zs',
            'I_test',
        )
        assert comet_score.parse_hyp_path(Path('tag_E_milpac_test_hyps.jsonl')) == (
            'tag',
            'E_milpac_test',
        )

    def test_non_matching_ignored(self):
        assert comet_score.parse_hyp_path(Path('data/analysis/foo.txt')) is None

"""Unit tests for run/hyp-file fingerprints and the resume gate."""

import json

import pytest

from src.evaluation.fingerprint import (
    FingerprintMismatchError,
    build_config_fingerprint,
    check_resume_fingerprint,
    file_sha256_prefix,
    sha256_prefix,
)


PENDING = build_config_fingerprint(
    model_id='facebook/nllb-200-distilled-600M',
    adapters='data/runs/foo/checkpoints/best_primary',
    max_input_length=256,
    max_new_tokens=256,
    num_beams=4,
    vocab_size=256102,
    device='cuda',
    dtype='bfloat16',
    seed=12345,
    mbr=None,
)


def _write_report(tmp_path, payload):
    p = tmp_path / 'report.json'
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    return p


class TestSha:
    def test_deterministic_and_content_sensitive(self):
        assert sha256_prefix(b'abc') == sha256_prefix(b'abc')
        assert sha256_prefix(b'abc') != sha256_prefix(b'abd')
        assert len(sha256_prefix(b'abc')) == 16

    def test_file_prefix_changes_on_rewrite(self, tmp_path):
        p = tmp_path / 'hyps.jsonl'
        p.write_text('one\n', encoding='utf-8')
        first = file_sha256_prefix(p)
        p.write_text('two\n', encoding='utf-8')
        assert file_sha256_prefix(p) != first

    def test_missing_file_empty(self, tmp_path):
        assert file_sha256_prefix(tmp_path / 'nope.jsonl') == ''


class TestConfigFingerprint:
    def test_adapters_none_becomes_base(self):
        fp = build_config_fingerprint('m', None, 1, 2, 3, 4, 'cpu', 'fp32', 5, None)
        assert fp['adapters'] == 'base'
        assert fp['model_id'] == 'm'

    def test_full_fields(self):
        assert PENDING['vocab_size'] == 256102
        assert PENDING['decode_device'] == 'cuda'
        assert PENDING['dtype'] == 'bfloat16'


class TestResumeGate:
    def test_no_hyp_file_is_fine(self, tmp_path):
        check_resume_fingerprint(
            tmp_path / 'missing.jsonl', tmp_path / 'report.json', PENDING, tag='t', suite='I_test'
        )

    def test_hyp_without_report_raises(self, tmp_path):
        hyp = tmp_path / 't_I_test_hyps.jsonl'
        hyp.write_text('{}\n', encoding='utf-8')
        with pytest.raises(FingerprintMismatchError):
            check_resume_fingerprint(
                hyp, tmp_path / 'report.json', PENDING, tag='t', suite='I_test'
            )

    def test_report_without_fingerprint_raises(self, tmp_path):
        hyp = tmp_path / 't_I_test_hyps.jsonl'
        hyp.write_text('{}\n', encoding='utf-8')
        report = _write_report(tmp_path, {'suites': []})
        with pytest.raises(FingerprintMismatchError):
            check_resume_fingerprint(hyp, report, PENDING, tag='t', suite='I_test')

    def test_mismatch_raises(self, tmp_path):
        hyp = tmp_path / 't_I_test_hyps.jsonl'
        hyp.write_text('{}\n', encoding='utf-8')
        other = dict(PENDING, num_beams=8)
        report = _write_report(tmp_path, {'fingerprint': other})
        with pytest.raises(FingerprintMismatchError):
            check_resume_fingerprint(hyp, report, PENDING, tag='t', suite='I_test')

    def test_matching_fingerprint_passes(self, tmp_path):
        hyp = tmp_path / 't_I_test_hyps.jsonl'
        hyp.write_text('{}\n', encoding='utf-8')
        report = _write_report(tmp_path, {'fingerprint': PENDING})
        check_resume_fingerprint(hyp, report, PENDING, tag='t', suite='I_test')

    def test_force_resume_overrides(self, tmp_path):
        hyp = tmp_path / 't_I_test_hyps.jsonl'
        hyp.write_text('{}\n', encoding='utf-8')
        report = _write_report(tmp_path, {'fingerprint': dict(PENDING, num_beams=8)})
        check_resume_fingerprint(hyp, report, PENDING, force_resume=True, tag='t', suite='I_test')

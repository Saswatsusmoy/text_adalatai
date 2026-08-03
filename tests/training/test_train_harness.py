"""Harness safety tests: NaN collective, resume emb rows, parity, hashes, registry."""

import json
from pathlib import Path

import pytest
import torch

from src.training.subsample import file_sha256
from src.training.train_nllb_lora import (
    _install_new_embed_grad_mask,
    apply_new_embed_rows,
    global_batch_parity,
    register_run,
    sync_nan_stop,
    verify_pool_hashes,
)


class _EmbedMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(10, 4))
        self._linear = torch.nn.Linear(4, 4)

    def get_input_embeddings(self):
        return self

    def forward(self, x):
        return self._linear(x) + self.weight.sum()


def test_sync_nan_stop_detects_nan_and_inf():
    stop = torch.zeros(1)
    assert sync_nan_stop(torch.tensor(float('nan')), True, stop) is True
    assert stop.item() == 1
    stop2 = torch.zeros(1)
    assert sync_nan_stop(torch.tensor(float('inf')), True, stop2) is True
    assert stop2.item() == 1


def test_sync_nan_stop_finite_passes_and_honours_flag():
    stop = torch.zeros(1)
    assert sync_nan_stop(torch.tensor(1.0), True, stop) is False
    assert stop.item() == 0
    assert sync_nan_stop(torch.tensor(float('nan')), False, torch.zeros(1)) is False


def test_sync_nan_stop_two_rank_max_reduce(monkeypatch):
    import src.training.train_nllb_lora as mod

    seen = []

    def fake_all_reduce(flag):
        seen.append(flag)
        if len(seen) == 2:
            for f in seen:
                f.copy_(torch.tensor([1.0]))

    monkeypatch.setattr(mod, 'all_reduce_max', fake_all_reduce)
    stop0, stop1 = torch.zeros(1), torch.zeros(1)
    rank0 = mod.sync_nan_stop(torch.tensor(float('nan')), True, stop0)
    rank1 = mod.sync_nan_stop(torch.tensor(1.0), True, stop1)
    assert rank0 is True
    assert rank1 is True
    assert stop0.item() == 1
    assert stop1.item() == 1
    assert len(seen) == 2


def test_apply_new_embed_rows_missing_raises_when_required(tmp_path: Path):
    m = _EmbedMod()
    with pytest.raises(FileNotFoundError):
        apply_new_embed_rows(m, tmp_path / 'ckpt', required=True)


def test_apply_new_embed_rows_missing_returns_false_when_not_required(tmp_path: Path):
    assert apply_new_embed_rows(_EmbedMod(), tmp_path / 'ckpt') is False


def test_apply_new_embed_rows_shape_mismatch_raises(tmp_path: Path):
    from src.training.train_nllb_lora import NEW_EMBED_ROWS_NAME

    ckpt = tmp_path / 'ckpt'
    ckpt.mkdir()
    torch.save(
        {'new_embed_start': 2, 'rows': torch.zeros(5, 8), 'vocab_size': 10, 'hidden': 8},
        ckpt / NEW_EMBED_ROWS_NAME,
    )
    with pytest.raises(RuntimeError):
        apply_new_embed_rows(_EmbedMod(), ckpt)


def test_install_new_embed_grad_mask_zeroes_old_rows():
    torch.manual_seed(0)
    m = _EmbedMod()
    assert _install_new_embed_grad_mask(m, 6) is True
    assert m.weight.requires_grad is True
    loss = (m(torch.randn(3, 4)) + 1.0).mean()
    loss.backward()
    assert torch.all(m.weight.grad[:6] == 0)
    assert torch.any(m.weight.grad[6:] != 0)


def test_global_batch_parity_math():
    assert global_batch_parity(1, 1, 16, 16) == (16, True)
    assert global_batch_parity(16, 2, 1, 32) == (32, True)
    assert global_batch_parity(1, 1, 16, 32) == (16, False)
    assert global_batch_parity(16, 2, 1, None) == (32, True)


def test_verify_pool_hashes_detects_drift(tmp_path: Path):
    pool = tmp_path / 'stage_a_train.jsonl'
    pool.write_text('a\nb\n', encoding='utf-8')
    manifest = {
        'source_pool': str(pool),
        'source_pool_sha256_prefix': file_sha256(pool),
        'output': str(tmp_path / 'out.jsonl'),
    }
    checks = verify_pool_hashes({}, manifest)
    assert len(checks) == 1
    assert checks[0]['ok'] is True
    pool.write_text('a\nc\n', encoding='utf-8')
    checks = verify_pool_hashes({}, manifest)
    assert checks[0]['ok'] is False


def test_verify_pool_hashes_checks_replay_pools_too(tmp_path: Path):
    assign = tmp_path / 'assign.jsonl'
    replay = tmp_path / 'replay.jsonl'
    assign.write_text('x\n', encoding='utf-8')
    replay.write_text('y\n', encoding='utf-8')
    manifest = {
        'assignment_path': str(assign),
        'assignment_sha256_prefix': file_sha256(assign),
        'replay_pool': str(replay),
        'replay_pool_sha256_prefix': file_sha256(replay),
    }
    checks = verify_pool_hashes({}, manifest)
    assert len(checks) == 2
    assert all(c['ok'] for c in checks)


def test_resolve_train_path_merges_sibling_manifest_hashes(tmp_path: Path):
    from src.training.train_nllb_lora import resolve_train_path

    pool = tmp_path / 'stage_a_train.jsonl'
    pool.write_text('line1\nline2\n', encoding='utf-8')
    frozen = tmp_path / 'stage_a_A1_n80000.jsonl'
    frozen.write_text('a\nb\n', encoding='utf-8')
    manifest = {
        'curriculum': 'A1',
        'tag': 'A1_n80000',
        'seed': 42,
        'source_pool': str(pool),
        'source_pool_sha256_prefix': file_sha256(pool),
        'output': str(frozen),
        'n': 2,
    }
    (tmp_path / 'stage_a_A1_n80000_manifest.json').write_text(
        json.dumps(manifest), encoding='utf-8'
    )
    cfg = {'data': {'train_jsonl': str(frozen), 'stage_a_train': str(pool)}}
    path, man = resolve_train_path(cfg, 'A', 'A1')
    assert path == frozen
    assert man['source_pool_sha256_prefix'] == manifest['source_pool_sha256_prefix']
    checks = verify_pool_hashes(cfg, man)
    assert len(checks) == 1
    assert checks[0]['ok'] is True
    pool.write_text('line1\nchanged\n', encoding='utf-8')
    checks = verify_pool_hashes(cfg, man)
    assert checks[0]['ok'] is False


def test_register_run_writes_and_merges(tmp_path: Path):
    root = tmp_path / 'runs'
    p1 = register_run(root, 'run_a', {'best_primary': 'x'})
    assert p1 == root / 'runs.json'
    register_run(root, 'run_b', {'best_primary': 'y'})
    db = json.loads(root.joinpath('runs.json').read_text(encoding='utf-8'))
    assert set(db) == {'run_a', 'run_b'}
    assert db['run_a']['best_primary'] == 'x'

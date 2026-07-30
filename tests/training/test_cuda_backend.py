"""Unit tests for CUDA/Hopper helpers (run on CPU/MPS without GPU)."""

from pathlib import Path

import torch

from src.training.cuda_backend import (
    build_optimizer,
    configure_torch_backend,
    dataloader_kwargs,
    resolve_dtype,
    rss_gb,
)
from src.training.train_nllb_lora import _save_peft


def test_resolve_dtype_float32_on_cpu():
    assert resolve_dtype('bfloat16', 'cpu') == torch.float32
    assert resolve_dtype('float16', 'cpu') == torch.float32
    assert resolve_dtype('float32', 'cpu') == torch.float32


def test_resolve_dtype_names():
    d = resolve_dtype('fp16', 'mps')
    assert d in (torch.float16, torch.float32)


def test_configure_backend_cpu_is_noop():
    info = configure_torch_backend('cpu', {'hardware': {'allow_tf32': True}})
    assert info['device'] == 'cpu'
    assert info['tf32'] is False


def test_dist_utils_single_process():
    from src.training.dist_utils import get_rank, get_world_size, is_main, unwrap_model

    assert get_rank() == 0
    assert get_world_size() == 1
    assert is_main() is True

    class Inner:
        pass

    class Outer:
        def __init__(self):
            self.module = Inner()

    assert isinstance(unwrap_model(Outer()), Inner)


def test_dataloader_kwargs_cpu_no_workers():
    kw = dataloader_kwargs('cpu', {'train': {'num_workers': 4}})
    assert kw['num_workers'] == 0
    assert kw['pin_memory'] is False


def test_dataloader_kwargs_cuda_shape():
    kw = dataloader_kwargs(
        'cuda:0',
        {'train': {'num_workers': 2, 'pin_memory': True, 'prefetch_factor': 3}},
    )
    assert kw['num_workers'] == 2
    assert kw['pin_memory'] is True
    assert kw['prefetch_factor'] == 3
    assert kw['persistent_workers'] is True


def test_build_optimizer_cpu():
    p = [torch.nn.Parameter(torch.zeros(4))]
    opt = build_optimizer(p, 1e-4, (0.9, 0.999), 0.01, 'cpu')
    assert opt is not None
    assert len(opt.param_groups) == 1


def test_rss_gb_returns_float_or_none():
    v = rss_gb()
    assert v is None or isinstance(v, float)


def test_save_peft_unwrap_compile_marker(tmp_path: Path):
    class FakeTok:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / 'tok').write_text('ok', encoding='utf-8')

    class FakePeft:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / 'adapter').write_text('ok', encoding='utf-8')

    class Wrapped:
        def __init__(self):
            self._orig_mod = FakePeft()

    out = tmp_path / 'ckpt'
    _save_peft(Wrapped(), out, FakeTok())
    assert (out / 'adapter').exists()
    assert (out / 'tok').exists()


def test_save_peft_new_embed_rows(tmp_path: Path):
    import torch

    from src.training.train_nllb_lora import (
        NEW_EMBED_ROWS_NAME,
        _save_peft,
        apply_new_embed_rows,
    )

    class FakeTok:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    class Emb(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(20, dtype=torch.float32).view(10, 2))

    class FakePeft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._emb = Emb()

        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / 'adapter').write_text('ok', encoding='utf-8')

        def get_input_embeddings(self):
            return self._emb

    model = FakePeft()
    out = tmp_path / 'ckpt'
    _save_peft(model, out, FakeTok(), new_embed_start=7)
    f = out / NEW_EMBED_ROWS_NAME
    assert f.is_file()
    payload = torch.load(f, map_location='cpu', weights_only=True)
    assert payload['new_embed_start'] == 7
    assert payload['rows'].shape == (3, 2)

    # mutate then restore
    model._emb.weight.data.zero_()
    assert apply_new_embed_rows(model, out)
    assert torch.allclose(model._emb.weight[7:], payload['rows'])

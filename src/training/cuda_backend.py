"""CUDA / Hopper helpers for NLLB LoRA training."""

from __future__ import annotations

import os
import sys


def resolve_dtype(name: str | None, device: str):
    import torch

    name = (name or 'float32').lower()
    if name in ('bf16', 'bfloat16'):
        if device.startswith('cuda') and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if device == 'mps' else torch.float32
    if name in ('fp16', 'float16'):
        if device.startswith('cuda') or device == 'mps':
            return torch.float16
        return torch.float32
    return torch.float32


def pick_best_cuda_device() -> str | None:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return None
    best_i, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            free = 0
        if free > best_free:
            best_free, best_i = free, i
    return f'cuda:{best_i}'


def enable_flash_sdp(prefer_flash: bool = True) -> dict:
    import torch

    info = {'flash': False, 'mem_efficient': False, 'math': True}
    try:
        if prefer_flash and hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            return {'flash': True, 'mem_efficient': True, 'math': False}
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            info['mem_efficient'] = True
    except Exception:
        pass
    return info


def configure_torch_backend(device: str, cfg: dict | None = None) -> dict:
    import torch

    cfg = cfg or {}
    hw = (cfg.get('hardware') or {}) if isinstance(cfg, dict) else {}
    info = {
        'device': device,
        'tf32': False,
        'cudnn_benchmark': False,
        'matmul_precision': None,
        'sdpa': False,
        'sdp_backends': None,
        'cuda_name': None,
        'cuda_cc': None,
        'free_gb': None,
    }
    if not device.startswith('cuda') or not torch.cuda.is_available():
        return info

    idx = int(device.split(':')[1]) if ':' in device else torch.cuda.current_device()
    torch.cuda.set_device(idx)
    props = torch.cuda.get_device_properties(idx)
    info['cuda_name'] = props.name
    info['cuda_cc'] = f'{props.major}.{props.minor}'
    try:
        free, total = torch.cuda.mem_get_info(idx)
        info['free_gb'] = round(free / (1024**3), 2)
        info['total_gb'] = round(total / (1024**3), 2)
    except Exception:
        pass

    if bool(hw.get('allow_tf32', True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
            info['matmul_precision'] = 'high'
        except Exception:
            pass
        info['tf32'] = True

    if bool(hw.get('cudnn_benchmark', True)):
        torch.backends.cudnn.benchmark = True
        info['cudnn_benchmark'] = True

    for attr, key in (
        ('allow_fp16_reduced_precision_reduction', 'allow_fp16_reduced_precision_reduction'),
        ('allow_bf16_reduced_precision_reduction', 'allow_bf16_reduced_precision_reduction'),
    ):
        if bool(hw.get(key, True)):
            try:
                setattr(torch.backends.cuda.matmul, attr, True)
            except Exception:
                pass

    info['sdp_backends'] = enable_flash_sdp(prefer_flash=bool(hw.get('prefer_flash_sdp', True)))
    info['sdpa'] = True
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.environ.setdefault('NCCL_P2P_DISABLE', '0')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    return info


def gpu_mem_gb(device: str) -> float | None:
    import torch

    if not device.startswith('cuda') or not torch.cuda.is_available():
        return None
    idx = int(device.split(':')[1]) if ':' in device else torch.cuda.current_device()
    try:
        return round(torch.cuda.memory_allocated(idx) / (1024**3), 3)
    except Exception:
        return None


def rss_gb() -> float | None:
    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(raw / (1024**3 if sys.platform == 'darwin' else 1024**2), 3)
    except Exception:
        return None


def maybe_compile(model, device: str, enabled: bool, mode: str = 'default'):
    if not enabled or not device.startswith('cuda'):
        return model, False
    import torch

    try:
        return torch.compile(model, mode=mode, fullgraph=False, dynamic=False), True
    except TypeError:
        try:
            return torch.compile(model, mode=mode, fullgraph=False), True
        except Exception as e:
            print(f'torch.compile disabled ({type(e).__name__}: {e})')
            return model, False
    except Exception as e:
        print(f'torch.compile disabled ({type(e).__name__}: {e})')
        return model, False


def dataloader_kwargs(device: str, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    train, hw = cfg.get('train') or {}, cfg.get('hardware') or {}
    n_workers = int(train.get('num_workers', hw.get('num_workers', 0)))
    cuda = device.startswith('cuda')
    kw = {
        'num_workers': n_workers if cuda else 0,
        'pin_memory': bool(train.get('pin_memory', cuda)) and cuda,
    }
    if kw['num_workers'] > 0:
        kw['persistent_workers'] = True
        kw['prefetch_factor'] = int(train.get('prefetch_factor', 2))
    return kw


def build_optimizer(params, lr: float, betas: tuple, weight_decay: float, device: str):
    import torch

    kwargs = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    if device.startswith('cuda'):
        try:
            return torch.optim.AdamW(params, **kwargs, fused=True)
        except (TypeError, RuntimeError, ValueError):
            pass
    return torch.optim.AdamW(params, **kwargs)

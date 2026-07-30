"""DistributedDataParallel helpers (torchrun / NCCL)."""

from __future__ import annotations

import os


def dist_is_available() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not dist_is_available():
        return 0
    import torch.distributed as dist

    return dist.get_rank()


def get_world_size() -> int:
    if not dist_is_available():
        return 1
    import torch.distributed as dist

    return dist.get_world_size()


def is_main() -> bool:
    return get_rank() == 0


def setup_distributed() -> dict:
    """
    Init process group if launched under torchrun (LOCAL_RANK set).
    Returns {enabled, rank, local_rank, world_size, device}.
    """
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank < 0 or not torch.cuda.is_available():
        return {
            'enabled': False,
            'rank': 0,
            'local_rank': 0,
            'world_size': 1,
            'device': None,
        }

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # device_id mutes NCCL barrier device warnings on torch 2.x
        try:
            dist.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{local_rank}'))
        except TypeError:
            dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    return {
        'enabled': True,
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world,
        'device': f'cuda:{local_rank}',
    }


def cleanup_distributed():
    if not dist_is_available():
        return
    import torch.distributed as dist

    dist.barrier()
    dist.destroy_process_group()


def unwrap_model(model):
    """Strip DDP / compile wrappers for save and generate."""
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def barrier():
    if dist_is_available():
        import torch.distributed as dist

        dist.barrier()

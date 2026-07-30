"""DDP helpers (torchrun / NCCL)."""

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
        try:
            dist.init_process_group(
                backend='nccl',
                device_id=torch.device(f'cuda:{local_rank}'),
            )
        except TypeError:
            dist.init_process_group(backend='nccl')
    return {
        'enabled': True,
        'rank': dist.get_rank(),
        'local_rank': local_rank,
        'world_size': dist.get_world_size(),
        'device': f'cuda:{local_rank}',
    }


def cleanup_distributed():
    if not dist_is_available():
        return
    import torch.distributed as dist

    dist.barrier()
    dist.destroy_process_group()


def unwrap_model(model):
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


def all_reduce_max(flag):
    if not dist_is_available():
        return
    import torch.distributed as dist

    dist.all_reduce(flag, op=dist.ReduceOp.MAX)


def broadcast_object(obj, src: int = 0):
    import torch.distributed as dist

    box = [obj] if get_rank() == src else [None]
    dist.broadcast_object_list(box, src=src)
    return box[0]

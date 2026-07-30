"""Load training YAML into dicts."""

from pathlib import Path

import yaml


DEFAULT_CONFIG = Path('configs/training.yaml')


def load_training_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('training config must be a mapping')
    for key in ('model', 'data', 'optim', 'train', 'eval'):
        if key not in cfg:
            raise ValueError(f'training config missing section: {key}')
    return cfg


def deep_get(cfg: dict, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

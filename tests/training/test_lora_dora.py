"""DoRA / LoRA config wiring for NLLB PEFT."""

import torch.nn as nn
from peft import LoraConfig

from src.training.train_nllb_lora import build_lora_config


class _TinyDec(nn.Module):
    """Minimal tree so decoder_attn path filters find Linear modules."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.decoder = nn.Module()
        self.model.decoder.layers = nn.ModuleList()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.q_proj = nn.Linear(8, 8, bias=False)
        layer.self_attn.k_proj = nn.Linear(8, 8, bias=False)
        layer.self_attn.v_proj = nn.Linear(8, 8, bias=False)
        layer.self_attn.out_proj = nn.Linear(8, 8, bias=False)
        layer.encoder_attn = nn.Module()
        layer.encoder_attn.q_proj = nn.Linear(8, 8, bias=False)
        layer.encoder_attn.k_proj = nn.Linear(8, 8, bias=False)
        layer.encoder_attn.v_proj = nn.Linear(8, 8, bias=False)
        layer.encoder_attn.out_proj = nn.Linear(8, 8, bias=False)
        self.model.decoder.layers.append(layer)


def test_build_lora_config_use_dora_flag():
    model = _TinyDec()
    cfg = {
        'peft': {
            'profile': 'decoder_attn',
            'r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.0,
            'use_dora': True,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            'bias': 'none',
        },
    }
    lcfg = build_lora_config(cfg, model)
    assert isinstance(lcfg, LoraConfig)
    assert lcfg.use_dora is True
    assert lcfg.r == 4


def test_build_lora_config_method_dora_enables_flag():
    model = _TinyDec()
    cfg = {
        'peft': {
            'method': 'dora',
            'profile': 'decoder_attn',
            'r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.0,
            'use_dora': False,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            'bias': 'none',
        },
    }
    lcfg = build_lora_config(cfg, model)
    assert lcfg.use_dora is True


def test_build_lora_config_default_no_dora():
    model = _TinyDec()
    cfg = {
        'peft': {
            'profile': 'decoder_attn',
            'r': 4,
            'lora_alpha': 8,
            'lora_dropout': 0.0,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            'bias': 'none',
        },
    }
    lcfg = build_lora_config(cfg, model)
    assert lcfg.use_dora is False

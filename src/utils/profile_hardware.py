"""
Profile local machine for Adalat AI training/inference decisions.

Emphasizes Apple Silicon + MLX + PyTorch MPS. Writes JSON report under
data/analysis/ so Track D/C can pick model sizes without cloud assumptions.

Usage:
  PYTHONPATH=. python3 src/utils/profile_hardware.py
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT_PATH = Path('data/analysis/hardware_profile.json')


def _sysctl(key: str) -> str | None:
    try:
        r = subprocess.run(
            ['sysctl', '-n', key],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except FileNotFoundError:
        return None
    return None


def _sp_hardware() -> dict:
    try:
        r = subprocess.run(
            ['system_profiler', 'SPHardwareDataType', '-json'],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            return {}
        data = json.loads(r.stdout)
        items = data.get('SPHardwareDataType') or []
        return items[0] if items else {}
    except Exception:
        return {}


def _bytes_to_gb(n: int | float | None) -> float | None:
    if n is None:
        return None
    return round(float(n) / (1024 ** 3), 2)


def profile_mlx() -> dict:
    info: dict = {'installed': False}
    try:
        import mlx.core as mx

        info['installed'] = True
        info['version'] = getattr(mx, '__version__', None)
        try:
            info['default_device'] = str(mx.default_device())
        except Exception as e:
            info['default_device_error'] = str(e)
        # Smoke matmul on default device (GPU on Apple Silicon)
        a = mx.random.normal((512, 512))
        b = mx.random.normal((512, 512))
        c = a @ b
        mx.eval(c)
        info['matmul_smoke'] = True
        info['matmul_sample'] = float(c[0, 0])
    except Exception as e:
        info['error'] = str(e)
    try:
        import mlx_lm  # noqa: F401
        from importlib.metadata import version

        info['mlx_lm_installed'] = True
        try:
            info['mlx_lm_version'] = version('mlx-lm')
        except Exception:
            info['mlx_lm_version'] = None
    except Exception as e:
        info['mlx_lm_installed'] = False
        info['mlx_lm_error'] = str(e)
    return info


def profile_torch() -> dict:
    info: dict = {'installed': False}
    try:
        import torch

        info['installed'] = True
        info['version'] = torch.__version__
        info['mps_built'] = bool(torch.backends.mps.is_built())
        info['mps_available'] = bool(torch.backends.mps.is_available())
        info['cuda_available'] = bool(torch.cuda.is_available())
        if info['mps_available']:
            x = torch.randn(512, 512, device='mps')
            y = x @ x
            info['mps_matmul_smoke'] = True
            info['mps_matmul_shape'] = list(y.shape)
    except Exception as e:
        info['error'] = str(e)
    return info


def recommend(profile: dict) -> dict:
    """Map hardware facts to concrete Adalat track guidance."""
    mem_gb = profile.get('memory_gb') or 0
    chip = (profile.get('chip') or '').lower()
    apple = 'apple' in chip or profile.get('machine') == 'arm64'
    mlx_ok = profile.get('mlx', {}).get('installed') and not profile.get('mlx', {}).get('error')
    mps_ok = profile.get('torch', {}).get('mps_available')

    rec: dict = {
        'training_location': 'local_only',
        'backend_primary': None,
        'backend_secondary': None,
        'track_d_models': [],
        'track_c_models': [],
        'max_batch_hint': None,
        'quantization': None,
        'notes': [],
    }

    if not apple:
        rec['notes'].append('Non-Apple host: MLX not available; use CUDA/CPU instead.')
        rec['backend_primary'] = 'pytorch_cuda_or_cpu'
        return rec

    rec['backend_primary'] = 'mlx' if mlx_ok else 'pytorch_mps'
    rec['backend_secondary'] = 'pytorch_mps' if mps_ok else 'cpu'

    if mem_gb <= 18:
        # 16GB unified memory class (this machine)
        rec['unified_memory_gb'] = mem_gb
        rec['quantization'] = '4bit_preferred_for_7b_class'
        rec['max_batch_hint'] = 1
        rec['track_d_models'] = [
            {
                'name': 'NLLB-200-distilled-600M',
                'path': 'facebook/nllb-200-distilled-600M',
                'backend': 'pytorch_mps',
                'role': 'enc-dec MT baseline / Stage A-B LoRA if PEFT fits',
                'fit_16gb': 'comfortable_inference; light_ft_possible',
            },
            {
                'name': 'InLegalTrans-En2Indic-1B',
                'path': 'law-ai/InLegalTrans-En2Indic-1B',
                'backend': 'pytorch_mps',
                'role': 'legal MT baseline (HF seq2seq); not mlx-lm native',
                'fit_16gb': 'inference_ok; full_ft_risky; prefer_lora_or_freeze_encoder',
            },
        ]
        rec['track_c_models'] = [
            {
                'name': 'small_instruct_1b_to_3b_4bit',
                'examples': [
                    'mlx-community/Llama-3.2-1B-Instruct-4bit',
                    'mlx-community/Qwen2.5-1.5B-Instruct-4bit',
                    'mlx-community/gemma-2-2b-it-4bit',
                ],
                'backend': 'mlx_lm_lora',
                'role': 'instruction MT LoRA; custom SPM needs C1 emb work (harder on mlx-lm)',
                'fit_16gb': 'lora_ok_with_short_seq_and_batch_1',
            },
        ]
        rec['notes'].extend(
            [
                '16GB unified: treat RAM+weights+activations as one pool; close browsers during train.',
                'MLX-LM is strongest for decoder-only LLM LoRA, not a drop-in for NLLB/InLegalTrans enc-dec.',
                'Track D seq2seq: use PyTorch MPS (or CPU fallback). Track D/C LLM path: use MLX-LM LoRA 4-bit 1B-3B.',
                'Avoid 7B+ full FT on 16GB; 7B 4-bit LoRA may work at batch 1 short context only.',
                'Custom SPM (joint_full_41000) pairs best with a model you control (small enc-dec or emb-resize path); mlx-lm uses the base LLM tokenizer unless you implement vocab surgery.',
            ]
        )
    elif mem_gb <= 36:
        rec['quantization'] = '4bit_or_8bit'
        rec['max_batch_hint'] = 2
        rec['notes'].append('32GB class: 3B-7B LoRA more comfortable; still prefer LoRA over full FT.')
    else:
        rec['quantization'] = 'fp16_or_4bit'
        rec['max_batch_hint'] = 4
        rec['notes'].append('>=48GB: larger LoRA and longer sequences feasible.')

    rec['sequence_length_hint'] = 256 if mem_gb <= 18 else 512
    rec['stage_a_subsample_hint'] = (
        'Start with MILPaC + HC/SUVAS + 50k-100k judiciary lines before full 992k on 16GB.'
        if mem_gb <= 18
        else 'Full Stage A feasible with care.'
    )
    return rec


def run(verbose: bool = True) -> dict:
    hw = _sp_hardware()
    mem_bytes = _sysctl('hw.memsize')
    mem_gb = _bytes_to_gb(int(mem_bytes)) if mem_bytes and mem_bytes.isdigit() else None
    if mem_gb is None and hw.get('physical_memory'):
        # e.g. "16 GB"
        raw = str(hw.get('physical_memory', '')).split()[0]
        try:
            mem_gb = float(raw)
        except ValueError:
            pass

    profile = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python': sys.version.split()[0],
        'chip': hw.get('chip_type') or _sysctl('machdep.cpu.brand_string'),
        'model_name': hw.get('machine_name') or hw.get('model_name'),
        'model_identifier': hw.get('machine_model') or hw.get('model_identifier'),
        'cpu_cores_reported': hw.get('number_processors') or hw.get('cpu_type'),
        'physical_cpu': _sysctl('hw.physicalcpu'),
        'logical_cpu': _sysctl('hw.logicalcpu'),
        'memory_bytes': int(mem_bytes) if mem_bytes and mem_bytes.isdigit() else None,
        'memory_gb': mem_gb,
        'mlx': profile_mlx(),
        'torch': profile_torch(),
    }
    # system_profiler field names vary by macOS version
    if not profile['chip'] and hw:
        profile['chip'] = hw.get('chip_type') or hw.get('cpu_type')
    if hw.get('physical_memory') and not profile['memory_gb']:
        try:
            profile['memory_gb'] = float(str(hw['physical_memory']).split()[0])
        except ValueError:
            pass

    profile['recommendations'] = recommend(profile)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(profile, indent=2), encoding='utf-8')

    if verbose:
        print('=== Hardware profile ===')
        print(f"chip:     {profile.get('chip')}")
        print(f"model:    {profile.get('model_name')} ({profile.get('model_identifier')})")
        print(f"memory:   {profile.get('memory_gb')} GB unified")
        print(f"cpus:     physical={profile.get('physical_cpu')} logical={profile.get('logical_cpu')}")
        print(f"mlx:      {profile['mlx']}")
        print(f"torch:    {profile['torch']}")
        print('--- recommendations ---')
        rec = profile['recommendations']
        for k, v in rec.items():
            if k == 'notes':
                print('notes:')
                for n in v:
                    print(f'  - {n}')
            else:
                print(f'{k}: {v}')
        print(f'\nWrote {OUT_PATH}')
    return profile


def main():
    run(verbose=True)


if __name__ == '__main__':
    main()

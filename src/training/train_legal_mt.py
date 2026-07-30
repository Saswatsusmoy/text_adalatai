"""
Track C1: train small Marian enc-dec with SPM_V2_PRIMARY (joint legal 41k).

Usage:
  PYTHONPATH=. python -m src.training.train_legal_mt --curriculum smoke --max-steps 50
  PYTHONPATH=. python -m src.training.train_legal_mt --config configs/training_c1.yaml --curriculum A1
  torchrun --standalone --nproc_per_node=2 -m src.training.train_legal_mt \\
      --config configs/training_c1_h200.yaml --curriculum A1 --device cuda
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from src.evaluation.eval_sets import load_jsonl
from src.evaluation.metrics_mt import score_pairs
from src.training.config import deep_get, load_training_config
from src.training.cuda_backend import (
    build_optimizer,
    configure_torch_backend,
    dataloader_kwargs,
    gpu_mem_gb,
    pick_best_cuda_device,
    rss_gb,
)
from src.training.dist_utils import (
    barrier,
    cleanup_distributed,
    is_main,
    setup_distributed,
    unwrap_model,
)
from src.training.legal_mt_data import LegalMtJsonlDataset, collate_legal_mt
from src.training.legal_mt_model import build_legal_mt_model
from src.training.spm_tokenizer import LegalSpmTokenizer
from src.training.subsample import build_subsample
from src.evaluation.zero_shot_nllb import pick_device


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def resolve_train_path(cfg: dict, stage: str, curriculum: str) -> tuple[Path, dict | None]:
    if stage == 'B':
        path = Path(deep_get(cfg, 'data', 'stage_b_train'))
        if not path.exists():
            raise FileNotFoundError(path)
        return path, None
    override = deep_get(cfg, 'data', 'train_jsonl', default=None)
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(path)
        n = sum(1 for line in open(path, encoding='utf-8') if line.strip())
        return path, {
            'curriculum': curriculum,
            'output': str(path),
            'n': n,
            'prebuilt': True,
        }
    if curriculum == 'full':
        path = Path(deep_get(cfg, 'data', 'stage_a_train'))
        return path, {'curriculum': 'full', 'output': str(path), 'n': 'all'}
    man = build_subsample(curriculum=curriculum, verbose=is_main())
    return Path(man['output']), man


@torch.no_grad()
def eval_loss(model, loader, device, max_batches=50, dtype=torch.float32) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {
            k: v.to(device, non_blocking=str(device).startswith('cuda'))
            for k, v in batch.items()
        }
        if str(device).startswith('cuda') and dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type='cuda', dtype=dtype):
                out = model(**batch)
        else:
            out = model(**batch)
        total += float(out.loss.detach().float().cpu())
        n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def eval_generate(
    model,
    tokenizer: LegalSpmTokenizer,
    pairs: list[dict],
    device: str,
    max_pairs: int | None,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
    gen_batch_size: int = 8,
) -> dict:
    raw = unwrap_model(model)
    raw.eval()
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    hyps, refs = [], []
    pad_id = tokenizer.pad_token_id
    for i in range(0, len(pairs), gen_batch_size):
        chunk = pairs[i: i + gen_batch_size]
        enc_ids = [
            tokenizer.encode(p['en_text'], max_length=max_input_length)
            for p in chunk
        ]
        max_len = max(len(x) for x in enc_ids)
        input_ids = torch.tensor(
            [x + [pad_id] * (max_len - len(x)) for x in enc_ids],
            dtype=torch.long,
            device=device,
        )
        attention_mask = (input_ids != pad_id).long()
        out = raw.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            decoder_start_token_id=tokenizer.bos_token_id,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        hyps.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        refs.extend(p['hi_text'] for p in chunk)
    model.train()
    return score_pairs(hyps, refs)


def run(
    config_path: str = 'configs/training_c1.yaml',
    stage: str = 'A',
    curriculum: str = 'smoke',
    max_steps: int | None = None,
    resume_from: str | None = None,
    device: str | None = None,
    skip_gen_eval: bool = False,
    verbose: bool = True,
) -> dict:
    cfg = load_training_config(config_path)
    dist_info = setup_distributed()
    rank = dist_info['rank']
    world = dist_info['world_size']
    main = is_main()
    verbose = verbose and main

    seed = int(deep_get(cfg, 'run', 'seed', default=42))
    set_seed(seed, rank=rank)

    if dist_info['enabled']:
        device = dist_info['device']
    elif device is None:
        device = pick_device(prefer_mps=True)
    elif device == 'cuda':
        device = pick_best_cuda_device() or 'cuda:0'

    backend_info = configure_torch_backend(device, cfg)
    backend_info['world_size'] = world
    backend_info['rank'] = rank
    backend_info['ddp'] = dist_info['enabled']

    train_path, data_manifest = resolve_train_path(cfg, stage, curriculum)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    tag = deep_get(cfg, 'run', 'tag', default='c1') or 'c1'
    if dist_info['enabled']:
        run_id_local = f"legal_mt_{stage}_{curriculum}_{tag}_ddp{world}_{ts}"
    else:
        run_id_local = f"legal_mt_{stage}_{curriculum}_{tag}_{ts}"
    if dist_info['enabled']:
        import torch.distributed as dist

        obj = [run_id_local] if main else [None]
        dist.broadcast_object_list(obj, src=0)
        run_id = obj[0]
    else:
        run_id = run_id_local

    run_dir = Path(deep_get(cfg, 'run', 'output_root', default='data/runs')) / run_id
    ckpt_dir = run_dir / 'checkpoints'
    metrics_dir = run_dir / 'metrics'
    if main:
        for d in (run_dir, ckpt_dir, metrics_dir, run_dir / 'hyps'):
            d.mkdir(parents=True, exist_ok=True)
        (run_dir / 'config.snapshot.yaml').write_text(
            Path(config_path).read_text(encoding='utf-8'), encoding='utf-8',
        )
        if data_manifest:
            (run_dir / 'data_manifest.json').write_text(
                json.dumps(data_manifest, indent=2, ensure_ascii=False), encoding='utf-8',
            )
        (run_dir / 'backend_info.json').write_text(
            json.dumps(backend_info, indent=2), encoding='utf-8',
        )
    barrier()

    spm_path = deep_get(cfg, 'model', 'spm_path', default=None)
    tokenizer = LegalSpmTokenizer(spm_path)
    if resume_from:
        model = build_legal_mt_model(tokenizer, deep_get(cfg, 'model', 'arch', default={}) or {})
        # load weights
        state_path = Path(resume_from)
        if (state_path / 'model.safetensors').exists() or (state_path / 'pytorch_model.bin').exists():
            from transformers import MarianMTModel

            model = MarianMTModel.from_pretrained(state_path)
        else:
            raise FileNotFoundError(f'no model weights in {state_path}')
    else:
        model = build_legal_mt_model(tokenizer, deep_get(cfg, 'model', 'arch', default={}) or {})

    if str(device).startswith('cuda') and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        model = model.to(device=device, dtype=dtype)
    elif str(device) in ('cuda', 'mps') or str(device).startswith('cuda'):
        dtype = torch.float16
        model = model.to(device=device, dtype=dtype)
    else:
        dtype = torch.float32
        model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f'run_id={run_id}')
        print(f'device={device} stage={stage} curriculum={curriculum} world={world}')
        print(f'spm={tokenizer.model_path} vocab={tokenizer.vocab_size}')
        print(f'train_data={train_path}')
        print(f'params={n_params:,} dtype={dtype}')

    if dist_info['enabled']:
        local_rank = dist_info['local_rank']
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    max_src = int(deep_get(cfg, 'data', 'max_source_length', default=256))
    max_tgt = int(deep_get(cfg, 'data', 'max_target_length', default=256))
    train_ds = LegalMtJsonlDataset(
        train_path, tokenizer,
        max_source_length=max_src, max_target_length=max_tgt,
    )
    if len(train_ds) == 0:
        raise RuntimeError('empty training set')

    pad_id = tokenizer.pad_token_id
    pad_fixed = bool(deep_get(cfg, 'train', 'pad_to_fixed', default=False))
    pad_mult = int(deep_get(cfg, 'train', 'pad_to_multiple_of', default=1))
    fixed = (max_src, max_tgt) if pad_fixed else None
    collate = partial(
        collate_legal_mt,
        pad_token_id=pad_id,
        pad_to_multiple_of=pad_mult,
        pad_to_fixed=fixed,
    )

    batch_size = int(deep_get(cfg, 'train', 'batch_size', default=8))
    target_global = deep_get(cfg, 'train', 'global_batch_size', default=None)
    if target_global is not None and world > 1:
        target_global = int(target_global)
        if target_global % world == 0:
            batch_size = target_global // world

    dl_kw = dataloader_kwargs(device, cfg)
    sampler = None
    if dist_info['enabled']:
        sampler = DistributedSampler(
            train_ds, num_replicas=world, rank=rank, shuffle=True, seed=seed,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate,
        **dl_kw,
    )

    def make_loader(path, max_pairs=None):
        p = Path(path)
        if not p.exists():
            return None
        ds = LegalMtJsonlDataset(
            p, tokenizer, max_source_length=max_src, max_target_length=max_tgt,
            max_pairs=max_pairs,
        )
        if len(ds) == 0:
            return None
        eval_bs = int(deep_get(cfg, 'decode', 'eval_batch_size', default=8))
        return DataLoader(ds, batch_size=eval_bs, shuffle=False, collate_fn=collate)

    i_dev_loader = make_loader(deep_get(cfg, 'eval', 'policy_I_dev')) if main else None
    e_milpac_dev_loader = (
        make_loader(deep_get(cfg, 'eval', 'policy_E_milpac_dev')) if main else None
    )

    lr = float(
        deep_get(cfg, 'optim', 'lr_stage_b' if stage == 'B' else 'lr', default=3e-4)
    )
    weight_decay = float(deep_get(cfg, 'optim', 'weight_decay', default=0.01))
    betas = (
        float(deep_get(cfg, 'optim', 'adam_beta1', default=0.9)),
        float(deep_get(cfg, 'optim', 'adam_beta2', default=0.999)),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(params, lr, betas, weight_decay, device)

    accum = int(deep_get(cfg, 'train', 'grad_accum_steps', default=1))
    default_max = int(
        deep_get(
            cfg, 'train',
            'max_steps_stage_b' if stage == 'B' else 'max_steps_stage_a',
            default=5000,
        )
    )
    max_steps = int(max_steps if max_steps is not None else default_max)
    warmup_ratio = float(deep_get(cfg, 'optim', 'warmup_ratio', default=0.05))
    warmup_steps = max(1, int(max_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
    )
    max_grad_norm = float(deep_get(cfg, 'optim', 'max_grad_norm', default=1.0))
    log_every = int(deep_get(cfg, 'train', 'log_every_steps', default=20))
    eval_loss_every = int(deep_get(cfg, 'train', 'eval_loss_every_steps', default=150))
    eval_gen_every = int(deep_get(cfg, 'train', 'eval_gen_every_steps', default=500))
    save_every = int(deep_get(cfg, 'train', 'save_every_steps', default=500))
    patience = int(deep_get(cfg, 'train', 'early_stop_patience_evals', default=5))
    stop_on_nan = bool(deep_get(cfg, 'monitoring', 'stop_on_nan', default=True))

    num_beams = int(deep_get(cfg, 'decode', 'num_beams', default=4))
    max_new_tokens = int(deep_get(cfg, 'decode', 'max_new_tokens', default=256))
    gen_bs = int(deep_get(cfg, 'decode', 'eval_batch_size', default=8))
    anu_cap = int(deep_get(cfg, 'eval', 'anuvaad_dev_max_pairs', default=200))

    i_dev_pairs = load_jsonl(Path(deep_get(cfg, 'eval', 'policy_I_dev'))) if main else []
    e_milpac_dev_pairs = (
        load_jsonl(Path(deep_get(cfg, 'eval', 'policy_E_milpac_dev'))) if main else []
    )
    e_anu_dev_pairs = (
        load_jsonl(Path(deep_get(cfg, 'eval', 'policy_E_anuvaad_dev'))) if main else []
    )

    train_log = metrics_dir / 'train_log.jsonl'
    eval_log = metrics_dir / 'eval_log.jsonl'

    model.train()
    global_step = 0
    micro = 0
    running_loss = 0.0
    best_primary = -1e9
    best_step = -1
    bad_evals = 0
    t0 = time.time()
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    data_iter = iter(train_loader)
    stop_flag = torch.zeros(1, device=device if str(device).startswith('cuda') else 'cpu')

    if verbose:
        print(
            f'train pairs={len(train_ds)} max_steps={max_steps} '
            f'batch={batch_size} world={world} accum={accum} lr={lr}'
        )

    while global_step < max_steps:
        if dist_info['enabled'] and stop_flag.item() > 0:
            break
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {
            k: v.to(device, non_blocking=str(device).startswith('cuda'))
            for k, v in batch.items()
        }
        if str(device).startswith('cuda') and dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type='cuda', dtype=dtype):
                out = model(**batch)
                loss = out.loss / accum
        else:
            out = model(**batch)
            loss = out.loss / accum

        if stop_on_nan and (torch.isnan(loss) or torch.isinf(loss)):
            if main:
                print(f'NaN/Inf at step {global_step}; stop')
            stop_flag.fill_(1)
            if dist_info['enabled']:
                import torch.distributed as dist

                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            break

        loss.backward()
        running_loss += float(loss.detach().float().cpu())
        micro += 1
        del batch, out

        if micro % accum != 0:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if main and (global_step % log_every == 0 or global_step == 1):
            denom = log_every if global_step > 1 else 1
            row = {
                'step': global_step,
                'loss': round(running_loss / denom, 6),
                'lr': scheduler.get_last_lr()[0],
                'grad_norm': float(grad_norm),
                'rss_gb': rss_gb(),
                'gpu_mem_gb': gpu_mem_gb(device),
                'elapsed_s': round(time.time() - t0, 1),
            }
            append_jsonl(train_log, row)
            if verbose:
                print(
                    f"step={global_step} loss={row['loss']:.4f} "
                    f"lr={row['lr']:.2e} gn={row['grad_norm']:.3f}"
                )
            running_loss = 0.0

        do_loss = global_step % eval_loss_every == 0
        do_gen = (not skip_gen_eval) and global_step % eval_gen_every == 0
        do_save = global_step % save_every == 0
        if do_loss or do_gen or do_save:
            barrier()

        if main and do_loss:
            losses = {'step': global_step, 'type': 'loss_eval'}
            if i_dev_loader is not None:
                losses['I_dev_loss'] = round(
                    eval_loss(model, i_dev_loader, device, dtype=dtype), 4,
                )
            if e_milpac_dev_loader is not None:
                losses['E_milpac_dev_loss'] = round(
                    eval_loss(model, e_milpac_dev_loader, device, dtype=dtype), 4,
                )
            append_jsonl(eval_log, losses)
            if verbose:
                print(f'  loss_eval {losses}')

        if main and do_gen:
            gen = {'step': global_step, 'type': 'gen_eval'}
            if i_dev_pairs:
                gen['I_dev'] = eval_generate(
                    model, tokenizer, i_dev_pairs, device, None,
                    max_src, max_new_tokens, num_beams, gen_bs,
                )
            if e_milpac_dev_pairs:
                gen['E_milpac_dev'] = eval_generate(
                    model, tokenizer, e_milpac_dev_pairs, device, None,
                    max_src, max_new_tokens, num_beams, gen_bs,
                )
            if e_anu_dev_pairs:
                gen['E_anuvaad_dev_sample'] = eval_generate(
                    model, tokenizer, e_anu_dev_pairs, device, anu_cap,
                    max_src, max_new_tokens, num_beams, gen_bs,
                )
            weights = deep_get(cfg, 'eval', 'selection', 'stage_a_weights', default={}) or {}
            if stage == 'B':
                weights = deep_get(cfg, 'eval', 'selection', 'stage_b_weights', default={}) or {}
            primary, wsum = 0.0, 0.0
            for key, w in weights.items():
                block = gen.get(key)
                if block and 'chrfpp' in block:
                    primary += float(w) * float(block['chrfpp']['score'])
                    wsum += float(w)
            if wsum > 0:
                primary /= wsum
            gen['primary_chrfpp'] = round(primary, 4)
            append_jsonl(eval_log, gen)
            if verbose:
                print(f"  gen_eval primary_chrfpp={gen['primary_chrfpp']}")
                for k in ('I_dev', 'E_milpac_dev', 'E_anuvaad_dev_sample'):
                    if k in gen and 'chrfpp' in gen[k]:
                        print(
                            f"    {k}: BLEU={gen[k]['bleu']['score']:.2f} "
                            f"chrF++={gen[k]['chrfpp']['score']:.2f}"
                        )
            if primary > best_primary:
                best_primary = primary
                best_step = global_step
                bad_evals = 0
                best_dir = ckpt_dir / 'best_primary'
                unwrap_model(model).save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                if verbose:
                    print(f'  new best primary={best_primary:.4f} -> {best_dir}')
            else:
                bad_evals += 1
                if bad_evals >= patience:
                    if verbose:
                        print(f'Early stop at {global_step}')
                    stop_flag.fill_(1)

        if main and do_save:
            step_dir = ckpt_dir / f'step_{global_step}'
            unwrap_model(model).save_pretrained(step_dir)
            tokenizer.save_pretrained(step_dir)

        if do_loss or do_gen or do_save:
            if dist_info['enabled']:
                import torch.distributed as dist

                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            barrier()

    barrier()
    if main:
        last_dir = ckpt_dir / 'last'
        unwrap_model(model).save_pretrained(last_dir)
        tokenizer.save_pretrained(last_dir)
        summary = {
            'run_id': run_id,
            'track': 'C1',
            'stage': stage,
            'curriculum': curriculum,
            'device': device,
            'world_size': world,
            'spm': str(tokenizer.model_path),
            'vocab_size': tokenizer.vocab_size,
            'params': n_params,
            'dtype': str(dtype).replace('torch.', ''),
            'train_path': str(train_path),
            'steps': global_step,
            'best_primary_chrfpp': best_primary if best_primary > -1e8 else None,
            'best_step': best_step if best_step >= 0 else None,
            'elapsed_s': round(time.time() - t0, 1),
            'checkpoints': {
                'last': str(last_dir),
                'best_primary': (
                    str(ckpt_dir / 'best_primary') if best_step >= 0 else None
                ),
            },
        }
        (run_dir / 'run_summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8',
        )
        if verbose:
            print(f'Done. summary={run_dir / "run_summary.json"}')
    else:
        summary = {'run_id': run_id, 'rank': rank, 'steps': global_step}

    cleanup_distributed()
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Track C1: train legal MT with custom SPM')
    parser.add_argument('--config', default='configs/training_c1.yaml')
    parser.add_argument('--stage', default='A', choices=['A', 'B'])
    parser.add_argument(
        '--curriculum', default='smoke', choices=['smoke', 'A1', 'A2', 'full'],
    )
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--resume-from', default=None, help='HF model dir to continue')
    parser.add_argument('--device', default=None)
    parser.add_argument('--skip-gen-eval', action='store_true')
    args = parser.parse_args()
    run(
        config_path=args.config,
        stage=args.stage,
        curriculum=args.curriculum if args.stage == 'A' else 'full',
        max_steps=args.max_steps,
        resume_from=args.resume_from,
        device=args.device,
        skip_gen_eval=args.skip_gen_eval,
    )


if __name__ == '__main__':
    main()

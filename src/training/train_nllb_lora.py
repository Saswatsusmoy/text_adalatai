"""Track D: LoRA fine-tune NLLB-600M legal EN->HI (MPS or CUDA DDP)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.evaluation.eval_sets import load_jsonl
from src.evaluation.metrics_mt import score_pairs
from src.evaluation.zero_shot_nllb import pick_device, translate_batch
from src.training.common import (
    append_jsonl,
    autocast_ctx,
    count_nonempty_lines,
    empty_device_cache,
    is_cuda,
    loss_value,
    move_batch,
    set_seed,
    write_json,
)
from src.training.config import deep_get, load_training_config
from src.training.cuda_backend import (
    build_grad_scaler,
    build_optimizer,
    configure_torch_backend,
    dataloader_kwargs,
    gpu_mem_gb,
    maybe_compile,
    pick_best_cuda_device,
    resolve_compute_dtype,
    resolve_dtype,
    rss_gb,
)
from src.training.dist_utils import (
    all_reduce_max,
    barrier,
    broadcast_object,
    cleanup_distributed,
    is_main,
    setup_distributed,
    unwrap_model,
)
from src.training.nllb_data import NllbJsonlDataset, collate_nllb
from src.training.selection import (
    evaluate_selection,
    load_baseline,
    parse_caps,
    selection_mode,
    weighted_chrfpp_primary,
)
from src.training.subsample import build_stage_b_replay_mix, build_subsample, file_sha256


NEW_EMBED_ROWS_NAME = 'new_embed_rows.pt'


def discover_lora_targets(model, preferred: list[str]) -> list[str]:
    names = {n.split('.')[-1] for n, _ in model.named_modules()}
    found = [t for t in preferred if t in names]
    if found:
        return found
    hits = {
        n.split('.')[-1] for n, _ in model.named_modules() if any(n.endswith(t) for t in preferred)
    }
    return sorted(hits) if hits else preferred


def _path_suffix_targets(model, suffixes: list[str], path_pred) -> list[str]:
    full = [
        name
        for name, mod in model.named_modules()
        if (type(mod).__name__ == 'Linear' or hasattr(mod, 'weight'))
        and any(name.endswith(s) for s in suffixes)
        and path_pred(name)
    ]
    if full:
        return full
    return sorted(
        {
            name.split('.')[-1]
            for name, _ in model.named_modules()
            if any(name.endswith(s) for s in suffixes) and path_pred(name)
        }
    )


def build_lora_config(cfg: dict, model) -> LoraConfig:
    profile = deep_get(cfg, 'peft', 'profile', default='decoder_attn')
    r = int(deep_get(cfg, 'peft', 'r', default=16))
    alpha = int(deep_get(cfg, 'peft', 'lora_alpha', default=32))
    dropout = float(deep_get(cfg, 'peft', 'lora_dropout', default=0.05))
    preferred = list(
        deep_get(cfg, 'peft', 'target_modules')
        or [
            'q_proj',
            'k_proj',
            'v_proj',
            'out_proj',
        ]
    )
    ffn = ['fc1', 'fc2']

    if profile == 'attn_all':
        target = discover_lora_targets(model, preferred)
    elif profile == 'decoder_attn':
        target = _path_suffix_targets(
            model,
            preferred,
            lambda n: '.decoder.layers.' in n and ('.self_attn.' in n or '.encoder_attn.' in n),
        )
    elif profile == 'cross_attn':
        target = _path_suffix_targets(model, preferred, lambda n: '.encoder_attn.' in n)
    elif profile == 'decoder_full':
        target = _path_suffix_targets(
            model,
            preferred + ffn,
            lambda n: (
                '.decoder.layers.' in n
                and (
                    '.self_attn.' in n
                    or '.encoder_attn.' in n
                    or n.endswith('.fc1')
                    or n.endswith('.fc2')
                )
            ),
        )
    elif profile == 'last4_decoder':

        def pred(n: str) -> bool:
            if '.decoder.layers.' not in n:
                return False
            try:
                idx = int(n.split('.decoder.layers.')[1].split('.')[0])
            except (IndexError, ValueError):
                return False
            return idx >= 8 and (
                '.self_attn.' in n
                or '.encoder_attn.' in n
                or n.endswith('.fc1')
                or n.endswith('.fc2')
            )

        target = _path_suffix_targets(model, preferred + ffn, pred)
    else:
        raise ValueError(
            f'unknown peft.profile={profile!r}; '
            f'use attn_all|decoder_attn|cross_attn|decoder_full|last4_decoder'
        )

    if not target:
        target = discover_lora_targets(model, preferred)

    modules_to_save = deep_get(cfg, 'peft', 'modules_to_save', default=None)
    if modules_to_save is not None:
        modules_to_save = list(modules_to_save)

    use_dora = bool(deep_get(cfg, 'peft', 'use_dora', default=False))
    method = str(deep_get(cfg, 'peft', 'method', default='lora') or 'lora').lower()
    if method in ('dora', 'dora_lora'):
        use_dora = True

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target,
        bias=deep_get(cfg, 'peft', 'bias', default='none'),
        task_type=TaskType.SEQ_2_SEQ_LM,
        layers_to_transform=None,
        modules_to_save=modules_to_save,
        use_dora=use_dora,
    )


def _save_peft(model, path: Path, tokenizer, new_embed_start: int | None = None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    raw = unwrap_model(model)
    if hasattr(raw, 'save_pretrained'):
        raw.save_pretrained(path)
    else:
        torch.save(raw.state_dict(), path / 'pytorch_model.bin')
    tokenizer.save_pretrained(path)
    if new_embed_start is None:
        return
    emb = raw.get_input_embeddings() if hasattr(raw, 'get_input_embeddings') else None
    if emb is None or not hasattr(emb, 'weight'):
        return
    start = int(new_embed_start)
    w = emb.weight.detach().float().cpu()
    torch.save(
        {
            'new_embed_start': start,
            'rows': w[start:].contiguous(),
            'vocab_size': int(w.shape[0]),
            'hidden': int(w.shape[1]),
        },
        path / NEW_EMBED_ROWS_NAME,
    )


def apply_new_embed_rows(model, path: Path | str, required: bool = False) -> bool:
    path = Path(path)
    f = path / NEW_EMBED_ROWS_NAME if path.is_dir() else path
    if not f.is_file():
        if required:
            raise FileNotFoundError(
                f'{NEW_EMBED_ROWS_NAME} missing in {path}; resume of a vocab-extended '
                f'model needs the saved emb rows'
            )
        return False
    payload = torch.load(f, map_location='cpu', weights_only=True)
    start, rows = int(payload['new_embed_start']), payload['rows']
    emb = unwrap_model(model).get_input_embeddings()
    if emb is None or not hasattr(emb, 'weight'):
        return False
    w, n = emb.weight, rows.shape[0]
    if start + n > w.shape[0] or rows.shape[1] != w.shape[1]:
        raise RuntimeError(
            f'new_embed_rows shape mismatch: start={start} rows={tuple(rows.shape)} '
            f'emb={tuple(w.shape)}'
        )
    with torch.no_grad():
        w[start : start + n].copy_(rows.to(device=w.device, dtype=w.dtype))
    return True


def _install_new_embed_grad_mask(model, new_embed_start: int) -> bool:
    emb = model.get_input_embeddings()
    if emb is None or not hasattr(emb, 'weight'):
        return False
    emb.weight.requires_grad = True

    def _mask_old_emb_grad(grad, start=new_embed_start):
        grad = grad.clone()
        grad[:start].zero_()
        return grad

    emb.weight.register_hook(_mask_old_emb_grad)
    return True


def resolve_train_path(
    cfg: dict,
    stage: str,
    curriculum: str,
    config_path: str | Path = 'configs/training.yaml',
) -> tuple[Path, dict | None]:
    override = deep_get(cfg, 'data', 'train_jsonl', default=None)
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(path)
        man = {
            'curriculum': curriculum,
            'output': str(path),
            'n': count_nonempty_lines(path),
            'prebuilt': True,
        }
        sibling = path.with_name(f'{path.stem}_manifest.json')
        if sibling.is_file():
            try:
                with open(sibling, encoding='utf-8') as f:
                    saved = json.load(f)
            except (OSError, ValueError):
                saved = {}
            for key in (
                'source_pool',
                'source_pool_sha256_prefix',
                'assignment_path',
                'assignment_sha256_prefix',
                'replay_pool',
                'replay_pool_sha256_prefix',
            ):
                if key in saved:
                    man[key] = saved[key]
        return path, man
    if stage == 'B':
        replay_cfg = deep_get(cfg, 'data', 'stage_b_replay', default={}) or {}
        if curriculum == 'Bp' or replay_cfg.get('enabled'):
            man = build_stage_b_replay_mix(config_path=config_path, verbose=is_main())
            return Path(man['output']), man
        path = Path(deep_get(cfg, 'data', 'stage_b_train'))
        if not path.exists():
            raise FileNotFoundError(path)
        return path, {
            'curriculum': 'B',
            'output': str(path),
            'n': count_nonempty_lines(path),
            'prebuilt': True,
            'replay': False,
        }
    if curriculum == 'full':
        path = Path(deep_get(cfg, 'data', 'stage_a_train'))
        return path, {'curriculum': 'full', 'output': str(path), 'n': 'all'}
    man = build_subsample(
        curriculum=curriculum,
        config_path=config_path,
        verbose=is_main(),
    )
    return Path(man['output']), man


def build_model(cfg: dict, device: str, resume_adapters: str | None = None):
    model_id = deep_get(cfg, 'model', 'id')
    src_lang = deep_get(cfg, 'model', 'src_lang', default='eng_Latn')
    tgt_lang = deep_get(cfg, 'model', 'tgt_lang', default='hin_Deva')
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if hasattr(tokenizer, 'src_lang'):
        tokenizer.src_lang = src_lang
    if hasattr(tokenizer, 'tgt_lang'):
        tokenizer.tgt_lang = tgt_lang

    dtype = resolve_dtype(deep_get(cfg, 'model', 'torch_dtype', default='float32'), device)
    if device == 'mps' and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32  # GradScaler needs fp32 master weights; autocast does fp16 compute
    load_kw = {'dtype': dtype} if dtype != torch.float32 else {}

    if is_cuda(device):
        try:
            base = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                attn_implementation='sdpa',
                **load_kw,
            )
        except (TypeError, ValueError, OSError):
            base = AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_kw)
    else:
        base = AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_kw)

    ls = deep_get(cfg, 'optim', 'label_smoothing', default=None)
    if ls is not None and hasattr(base, 'config'):
        try:
            base.config.label_smoothing_factor = float(ls)
        except Exception:
            pass

    if deep_get(cfg, 'model', 'gradient_checkpointing', default=True):
        base.gradient_checkpointing_enable()
    if hasattr(base, 'config'):
        base.config.use_cache = False

    if resume_adapters:
        model = PeftModel.from_pretrained(base, resume_adapters, is_trainable=True)
        new_emb_start = deep_get(cfg, 'peft', 'new_embed_start', default=None)
        if new_emb_start is not None:
            new_emb_start = int(new_emb_start)
        applied = apply_new_embed_rows(model, resume_adapters, required=new_emb_start is not None)
        if applied and is_main():
            print(f'loaded new_embed_rows from {resume_adapters}')
        if new_emb_start is not None and is_main():
            if _install_new_embed_grad_mask(model, new_emb_start):
                print(
                    f'new_embed_grad_mask reinstalled on resume: '
                    f'train emb rows [{new_emb_start}:{model.get_input_embeddings().weight.shape[0]}]'
                )
        model.to(device)
        return tokenizer, model, device, dtype

    lora_cfg = build_lora_config(cfg, base)
    profile = deep_get(cfg, 'peft', 'profile', default='decoder_attn')
    if is_main():
        dora = ' DoRA' if getattr(lora_cfg, 'use_dora', False) else ''
        print(
            f'LoRA{dora} profile={profile} '
            f'target_modules={len(lora_cfg.target_modules or [])} '
            f'r={lora_cfg.r} alpha={lora_cfg.lora_alpha}'
        )
    model = get_peft_model(base, lora_cfg)

    new_emb_start = deep_get(cfg, 'peft', 'new_embed_start', default=None)
    if new_emb_start is not None:
        new_emb_start = int(new_emb_start)
        if _install_new_embed_grad_mask(model, new_emb_start) and is_main():
            print(
                f'new_embed_grad_mask: train emb rows '
                f'[{new_emb_start}:{model.get_input_embeddings().weight.shape[0]}]'
            )

    model.to(device)
    return tokenizer, model, device, dtype


@torch.no_grad()
def eval_loss(model, loader, device, max_batches=50, dtype=torch.float32) -> float:
    raw = unwrap_model(model)
    raw.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = move_batch(batch, device)
        with autocast_ctx(device, dtype):
            out = raw(**batch)
        total += loss_value(out.loss)
        n += 1
        del batch, out
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def eval_generate(
    model,
    tokenizer,
    pairs,
    device,
    max_pairs,
    max_input_length,
    max_new_tokens,
    num_beams,
    gen_batch_size=1,
) -> dict:
    raw = unwrap_model(model)
    raw.eval()
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    texts = [p['en_text'] for p in pairs]
    hyps = []
    bs = max(1, gen_batch_size)
    for i in range(0, len(texts), bs):
        hyps.extend(
            translate_batch(
                texts[i : i + bs],
                tokenizer,
                raw,
                device,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
        )
    model.train()
    return score_pairs(hyps, [p['hi_text'] for p in pairs])


def _primary_chrf(gen: dict, weights: dict) -> float:
    return weighted_chrfpp_primary(gen, weights)


def sync_nan_stop(loss, stop_on_nan: bool, stop_flag: torch.Tensor) -> bool:
    local = torch.zeros(1, device=stop_flag.device)
    if stop_on_nan and (torch.isnan(loss).any() or torch.isinf(loss).any()):
        local.fill_(1)
    all_reduce_max(local)
    stopped = local.item() > 0
    if stopped:
        stop_flag.fill_(1)
    return stopped


def global_batch_parity(
    batch_size: int, world: int, accum: int, target: int | None
) -> tuple[int, bool]:
    actual = batch_size * world * accum
    if target is None:
        return actual, True
    return actual, actual == int(target)


def _manifest_path_key(manifest: dict, base: str) -> str | None:
    for cand in (base, f'{base}_path', f'{base}_pool'):
        if manifest.get(cand):
            return cand
    return None


def verify_pool_hashes(cfg: dict, data_manifest: dict | None) -> list[dict]:
    checks = []
    if not data_manifest:
        return checks
    for key in sorted(data_manifest):
        if not key.endswith('_sha256_prefix'):
            continue
        base = key[: -len('_sha256_prefix')]
        path_key = _manifest_path_key(data_manifest, base)
        p_raw = data_manifest.get(path_key) if path_key else None
        if not p_raw:
            p_raw = deep_get(cfg, 'data', 'stage_a_train')
        if not p_raw:
            continue
        p = Path(p_raw)
        if not p.exists():
            continue
        current = file_sha256(p)
        checks.append(
            {
                'key': key,
                'path': str(p),
                'manifest_prefix': data_manifest[key],
                'current_prefix': current,
                'ok': current == data_manifest[key],
            }
        )
    return checks


def register_run(output_root: Path, run_id: str, entry: dict) -> Path:
    output_root = Path(output_root)
    reg = output_root / 'runs.json'
    db = {}
    if reg.exists():
        try:
            db = json.loads(reg.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            db = {}
    db[run_id] = entry
    output_root.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps(db, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return reg


def run(
    config_path: str = 'configs/training.yaml',
    stage: str = 'A',
    curriculum: str = 'smoke',
    max_steps: int | None = None,
    resume_adapters: str | None = None,
    device: str | None = None,
    skip_gen_eval: bool = False,
    verbose: bool = True,
) -> dict:
    cfg = load_training_config(config_path)
    dist_info = setup_distributed()
    rank, world = dist_info['rank'], dist_info['world_size']
    main = is_main()
    verbose = verbose and main
    set_seed(int(deep_get(cfg, 'run', 'seed', default=42)), rank=rank)

    if dist_info['enabled']:
        device = dist_info['device']
    elif device is None:
        device = (
            pick_best_cuda_device() or pick_device()
            if deep_get(cfg, 'hardware', 'pick_most_free_gpu', default=False)
            else pick_device()
        )
    elif device == 'cuda':
        device = pick_best_cuda_device() or 'cuda:0'

    backend_info = configure_torch_backend(device, cfg)
    backend_info.update(world_size=world, rank=rank, ddp=dist_info['enabled'])
    if resume_adapters is None:
        resume_adapters = deep_get(cfg, 'resume', 'adapters', default=None)

    train_path, data_manifest = resolve_train_path(
        cfg,
        stage,
        curriculum,
        config_path=config_path,
    )
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    tag = deep_get(cfg, 'run', 'tag', default='') or ''
    tag_part = f'_{tag}' if tag else ''
    if dist_info['enabled']:
        tag_part = f'{tag_part}_ddp{world}' if tag_part else f'_ddp{world}'
    run_id = f'nllb600_{stage}_{curriculum}{tag_part}_{ts}'
    if dist_info['enabled']:
        run_id = broadcast_object(run_id)

    run_dir = Path(deep_get(cfg, 'run', 'output_root', default='data/runs')) / run_id
    ckpt_dir, metrics_dir = run_dir / 'checkpoints', run_dir / 'metrics'
    if main:
        for d in (run_dir, ckpt_dir, metrics_dir, run_dir / 'hyps'):
            d.mkdir(parents=True, exist_ok=True)
        (run_dir / 'config.snapshot.yaml').write_text(
            Path(config_path).read_text(encoding='utf-8'),
            encoding='utf-8',
        )
        if data_manifest:
            write_json(run_dir / 'data_manifest.json', data_manifest)
        write_json(run_dir / 'backend_info.json', backend_info)
        if resume_adapters:
            (run_dir / 'resume_adapters.txt').write_text(
                str(resume_adapters) + '\n',
                encoding='utf-8',
            )
        register_run(
            deep_get(cfg, 'run', 'output_root', default='data/runs'),
            run_id,
            {
                'run_dir': str(run_dir),
                'stage': stage,
                'curriculum': curriculum,
                'start_ts': ts,
                'config_snapshot': str(run_dir / 'config.snapshot.yaml'),
                'data_manifest': str(run_dir / 'data_manifest.json'),
                'backend_info': str(run_dir / 'backend_info.json'),
                'train_log': str(metrics_dir / 'train_log.jsonl'),
                'eval_log': str(metrics_dir / 'eval_log.jsonl'),
                'run_summary': str(run_dir / 'run_summary.json'),
                'best_primary': str(ckpt_dir / 'best_primary'),
                'resume_adapters': resume_adapters,
            },
        )
    barrier()

    if verbose:
        print(f'run_id={run_id}')
        print(f'device={device} stage={stage} curriculum={curriculum} world={world}')
        print(f'train_data={train_path}')
        if resume_adapters:
            print(f'resume_adapters={resume_adapters}')
        if backend_info.get('cuda_name'):
            print(
                f'cuda={backend_info["cuda_name"]} cc={backend_info["cuda_cc"]} '
                f'free_gb={backend_info.get("free_gb")} tf32={backend_info["tf32"]} '
                f'sdpa={backend_info["sdpa"]} sdp={backend_info.get("sdp_backends")}'
            )

    hash_checks = verify_pool_hashes(cfg, data_manifest)
    strict_hash = bool(deep_get(cfg, 'data', 'strict_source_pool_hash', default=False))
    for check in hash_checks:
        if check['ok']:
            continue
        msg = (
            f'{check["key"]} drift: manifest {check["manifest_prefix"]} != '
            f'current {check["current_prefix"]} for {check["path"]}'
        )
        if strict_hash:
            raise RuntimeError(msg)
        if verbose:
            print(f'WARN {msg}')

    tokenizer, model, device, dtype = build_model(
        cfg,
        device,
        resume_adapters=resume_adapters,
    )
    requested_dtype = deep_get(cfg, 'model', 'torch_dtype', default='float32')
    compute_dtype = resolve_compute_dtype(requested_dtype, device, dtype)
    scaler = build_grad_scaler(device, requested_dtype)

    # torch.compile + DDP hangs on NLLB/PEFT (Dynamo/NCCL)
    compile_on = bool(deep_get(cfg, 'train', 'torch_compile', default=False))
    if dist_info['enabled'] and compile_on:
        if main:
            print('torch.compile disabled under DDP (NCCL/PEFT stability)')
        compile_on = False
    model, compiled = maybe_compile(
        model,
        device,
        compile_on,
        mode=deep_get(cfg, 'train', 'torch_compile_mode', default='default'),
    )

    if dist_info['enabled']:
        lr = dist_info['local_rank']
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[lr],
            output_device=lr,
            find_unused_parameters=bool(
                deep_get(cfg, 'train', 'ddp_find_unused_parameters', default=True)
            ),
            gradient_as_bucket_view=True,
            static_graph=bool(deep_get(cfg, 'train', 'ddp_static_graph', default=False)),
            broadcast_buffers=False,
        )

    if verbose:
        print(
            f'dtype={dtype} compute_dtype={compute_dtype} '
            f'torch_compile={compiled} ddp={dist_info["enabled"]}'
        )
        if scaler is not None:
            print('grad_scaler=mps fp16')
        raw = unwrap_model(model)
        if hasattr(raw, 'print_trainable_parameters'):
            raw.print_trainable_parameters()

    max_src = int(deep_get(cfg, 'data', 'max_source_length', default=256))
    max_tgt = int(deep_get(cfg, 'data', 'max_target_length', default=256))
    src_lang = deep_get(cfg, 'model', 'src_lang', default='eng_Latn')
    tgt_lang = deep_get(cfg, 'model', 'tgt_lang', default='hin_Deva')
    train_ds = NllbJsonlDataset(
        train_path,
        tokenizer,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_source_length=max_src,
        max_target_length=max_tgt,
    )
    if len(train_ds) == 0:
        raise RuntimeError('empty training set')

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_mult = int(deep_get(cfg, 'train', 'pad_to_multiple_of', default=1))
    pad_fixed = bool(deep_get(cfg, 'train', 'pad_to_fixed', default=False))
    collate = partial(
        collate_nllb,
        pad_token_id=pad_id,
        pad_to_multiple_of=pad_mult,
        pad_to_fixed=(max_src, max_tgt) if pad_fixed else None,
    )

    batch_size = int(deep_get(cfg, 'train', 'batch_size', default=1))
    target_global = deep_get(cfg, 'train', 'global_batch_size', default=None)
    if target_global is not None:
        target_global = int(target_global)
        if batch_size * world != target_global and world > 1 and target_global % world == 0:
            batch_size = target_global // world
            if verbose:
                print(
                    f'global_batch_size={target_global} -> '
                    f'per_device_batch={batch_size} (world={world})'
                )

    dl_kw = dataloader_kwargs(device, cfg)
    sampler = None
    if dist_info['enabled']:
        sampler = DistributedSampler(
            train_ds,
            num_replicas=world,
            rank=rank,
            shuffle=True,
            seed=int(deep_get(cfg, 'run', 'seed', default=42)),
            drop_last=False,
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate,
        drop_last=False,
        **dl_kw,
    )

    def make_loader(path, max_pairs=None):
        p = Path(path)
        if not p.exists():
            return None
        ds = NllbJsonlDataset(
            p,
            tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_source_length=max_src,
            max_target_length=max_tgt,
            max_pairs=max_pairs,
        )
        if len(ds) == 0:
            return None
        eval_bs = int(deep_get(cfg, 'decode', 'eval_batch_size', default=1))
        if not is_cuda(device):
            eval_bs = 1
        return DataLoader(
            ds,
            batch_size=eval_bs,
            shuffle=False,
            collate_fn=collate,
            num_workers=min(2, dl_kw.get('num_workers', 0)),
            pin_memory=dl_kw.get('pin_memory', False),
        )

    i_dev_loader = make_loader(deep_get(cfg, 'eval', 'policy_I_dev')) if main else None
    e_milpac_dev_loader = (
        make_loader(deep_get(cfg, 'eval', 'policy_E_milpac_dev')) if main else None
    )

    lr = float(
        deep_get(
            cfg,
            'optim',
            'lr_stage_b' if stage == 'B' else 'lr',
            default=1e-4,
        )
    )
    weight_decay = float(deep_get(cfg, 'optim', 'weight_decay', default=0.01))
    betas = (
        float(deep_get(cfg, 'optim', 'adam_beta1', default=0.9)),
        float(deep_get(cfg, 'optim', 'adam_beta2', default=0.999)),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(params, lr, betas, weight_decay, device)

    accum = int(deep_get(cfg, 'train', 'grad_accum_steps', default=16))
    global_batch, gb_ok = global_batch_parity(
        batch_size,
        world,
        accum,
        deep_get(cfg, 'train', 'global_batch_size', default=None),
    )
    if not gb_ok:
        msg = (
            f'global batch mismatch: {batch_size}*{world}*{accum}={global_batch} '
            f'!= global_batch_size={deep_get(cfg, "train", "global_batch_size")}'
        )
        if bool(deep_get(cfg, 'train', 'strict_global_batch', default=False)):
            raise AssertionError(msg)
        if verbose:
            print(f'WARN {msg}')
    default_max = int(
        deep_get(
            cfg,
            'train',
            'max_steps_stage_b' if stage == 'B' else 'max_steps_stage_a',
            default=3000,
        )
    )
    max_steps = int(max_steps if max_steps is not None else default_max)
    warmup_ratio = float(deep_get(cfg, 'optim', 'warmup_ratio', default=0.05))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(max_steps * warmup_ratio)),
        num_training_steps=max_steps,
    )
    max_grad_norm = float(deep_get(cfg, 'optim', 'max_grad_norm', default=1.0))
    log_every = int(deep_get(cfg, 'train', 'log_every_steps', default=20))
    eval_loss_every = int(deep_get(cfg, 'train', 'eval_loss_every_steps', default=150))
    eval_gen_every = int(deep_get(cfg, 'train', 'eval_gen_every_steps', default=500))
    save_every = int(deep_get(cfg, 'train', 'save_every_steps', default=500))
    patience = int(deep_get(cfg, 'train', 'early_stop_patience_evals', default=5))
    stop_on_nan = bool(deep_get(cfg, 'monitoring', 'stop_on_nan', default=True))
    mem_warn = float(deep_get(cfg, 'monitoring', 'memory_warn_gb', default=14.0))
    new_embed_start = deep_get(cfg, 'peft', 'new_embed_start', default=None)
    if new_embed_start is not None:
        new_embed_start = int(new_embed_start)

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

    train_log, eval_log = metrics_dir / 'train_log.jsonl', metrics_dir / 'eval_log.jsonl'
    model.train()
    global_step = micro = 0
    running_loss = 0.0
    best_primary, best_step, bad_evals = -1e9, -1, 0
    t0, epoch = time.time(), 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    data_iter = iter(train_loader)
    stop_flag = torch.zeros(1, device=device if is_cuda(device) else 'cpu')

    sel_mode = selection_mode(cfg, stage)
    selection_caps = parse_caps(deep_get(cfg, 'eval', 'selection', default={}) or {})
    baseline = None
    if sel_mode == 'zscore':
        baseline = load_baseline(cfg, stage, resume_adapters)
        if baseline is None and verbose:
            print(
                'WARN z-score selection requested but no baseline eval log found; '
                'using raw weighted primary'
            )

    if verbose:
        print(
            f'train pairs={len(train_ds)} max_steps={max_steps} '
            f'per_device_batch={batch_size} world={world} accum={accum} '
            f'global_batch={global_batch} lr={lr} pad_fixed={pad_fixed} '
            f'pad_mult={pad_mult}'
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

        batch = move_batch(batch, device)
        with autocast_ctx(device, compute_dtype):
            out = model(**batch)
            loss = out.loss / accum
        if sync_nan_stop(loss, stop_on_nan, stop_flag):
            if main:
                append_jsonl(
                    train_log,
                    {
                        'step': global_step,
                        'event': 'nan_loss',
                        'loss': loss_value(loss),
                    },
                )
                print(f'NaN/Inf loss at step {global_step}; stopping')
            break
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        running_loss += loss_value(loss)
        micro += 1
        del batch, out
        if micro % accum != 0:
            continue

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if main and (global_step % log_every == 0 or global_step == 1):
            mem, gmem = rss_gb(), gpu_mem_gb(device)
            denom = log_every if global_step > 1 else 1
            row = {
                'step': global_step,
                'loss': round(running_loss / denom, 6),
                'lr': scheduler.get_last_lr()[0],
                'grad_norm': float(grad_norm) if grad_norm is not None else None,
                'rss_gb': mem,
                'gpu_mem_gb': gmem,
                'elapsed_s': round(time.time() - t0, 1),
                'global_batch': global_batch,
                'world_size': world,
            }
            append_jsonl(train_log, row)
            if verbose:
                gtxt = f' gpu={gmem}' if gmem is not None else ''
                print(
                    f'step={global_step} loss={row["loss"]:.4f} '
                    f'lr={row["lr"]:.2e} gn={row["grad_norm"]:.3f} mem={mem}{gtxt}'
                )
            if mem is not None and mem > mem_warn and verbose:
                print(f'  WARN memory {mem} GB > {mem_warn}')
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
                    eval_loss(model, i_dev_loader, device, dtype=dtype),
                    4,
                )
            if e_milpac_dev_loader is not None:
                losses['E_milpac_dev_loss'] = round(
                    eval_loss(model, e_milpac_dev_loader, device, dtype=dtype),
                    4,
                )
            append_jsonl(eval_log, losses)
            if verbose:
                print(f'  loss_eval {losses}')

        if main and do_gen:
            gen = {'step': global_step, 'type': 'gen_eval'}
            if i_dev_pairs:
                gen['I_dev'] = eval_generate(
                    model,
                    tokenizer,
                    i_dev_pairs,
                    device,
                    None,
                    max_src,
                    max_new_tokens,
                    num_beams,
                    gen_batch_size=gen_bs,
                )
            if e_milpac_dev_pairs:
                gen['E_milpac_dev'] = eval_generate(
                    model,
                    tokenizer,
                    e_milpac_dev_pairs,
                    device,
                    None,
                    max_src,
                    max_new_tokens,
                    num_beams,
                    gen_batch_size=gen_bs,
                )
            if e_anu_dev_pairs:
                gen['E_anuvaad_dev_sample'] = eval_generate(
                    model,
                    tokenizer,
                    e_anu_dev_pairs,
                    device,
                    anu_cap,
                    max_src,
                    max_new_tokens,
                    num_beams,
                    gen_batch_size=gen_bs,
                )
            wkey = 'stage_b_weights' if stage == 'B' else 'stage_a_weights'
            weights = deep_get(cfg, 'eval', 'selection', wkey, default={}) or {}
            sel = evaluate_selection(
                gen,
                weights,
                baseline if sel_mode == 'zscore' else None,
                selection_caps if sel_mode == 'zscore' else {},
            )
            primary = sel['primary']
            gen['primary_chrfpp'] = round(primary, 4)
            gen['selection'] = {
                'mode': sel['mode'],
                'cap_ok': sel['cap_ok'],
                'cap_violations': sel['cap_violations'],
                'z': sel['z'],
            }
            append_jsonl(eval_log, gen)
            if verbose:
                print(
                    f'  gen_eval primary_chrfpp={gen["primary_chrfpp"]} '
                    f'mode={sel["mode"]} cap_ok={sel["cap_ok"]}'
                )
                if sel['cap_violations']:
                    print(f'    cap violations: {sel["cap_violations"]}')
                for k in ('I_dev', 'E_milpac_dev', 'E_anuvaad_dev_sample'):
                    if k in gen and 'chrfpp' in gen[k]:
                        print(
                            f'    {k}: BLEU={gen[k]["bleu"]["score"]:.2f} '
                            f'chrF++={gen[k]["chrfpp"]["score"]:.2f}'
                        )
            if sel['cap_ok'] and primary > best_primary:
                best_primary, best_step, bad_evals = primary, global_step, 0
                best_dir = ckpt_dir / 'best_primary'
                _save_peft(model, best_dir, tokenizer, new_embed_start=new_embed_start)
                if verbose:
                    print(f'  new best primary={best_primary:.4f} -> {best_dir}')
            else:
                bad_evals += 1
                if bad_evals >= patience:
                    if verbose:
                        print(f'Early stop at step {global_step} (patience={patience})')
                    stop_flag.fill_(1)

        if main and do_save:
            _save_peft(
                model,
                ckpt_dir / f'step_{global_step}',
                tokenizer,
                new_embed_start=new_embed_start,
            )

        if do_loss or do_gen or do_save:
            all_reduce_max(stop_flag)
            barrier()

        if device == 'mps' and global_step % 50 == 0:
            empty_device_cache(device)

    barrier()
    if main:
        last_dir = ckpt_dir / 'last'
        _save_peft(model, last_dir, tokenizer, new_embed_start=new_embed_start)
        summary = {
            'run_id': run_id,
            'stage': stage,
            'curriculum': curriculum,
            'device': device,
            'world_size': world,
            'dtype': str(dtype).replace('torch.', ''),
            'compute_dtype': str(compute_dtype).replace('torch.', ''),
            'grad_scaler': scaler is not None,
            'selection_mode': sel_mode,
            'baseline': baseline,
            'torch_compile': compiled,
            'batch_size_per_device': batch_size,
            'global_batch': global_batch,
            'grad_accum_steps': accum,
            'train_path': str(train_path),
            'steps': global_step,
            'best_primary_chrfpp': best_primary if best_primary > -1e8 else None,
            'best_step': best_step if best_step >= 0 else None,
            'elapsed_s': round(time.time() - t0, 1),
            'backend': backend_info,
            'checkpoints': {
                'last': str(last_dir),
                'best_primary': (str(ckpt_dir / 'best_primary') if best_step >= 0 else None),
            },
            'rss_gb': rss_gb(),
            'gpu_mem_gb': gpu_mem_gb(device),
        }
        write_json(run_dir / 'run_summary.json', summary)
        if verbose:
            print(f'Done. summary={run_dir / "run_summary.json"}')
    else:
        summary = {'run_id': run_id, 'rank': rank, 'steps': global_step}

    cleanup_distributed()
    return summary


def main():
    import argparse

    p = argparse.ArgumentParser(description='LoRA fine-tune NLLB for legal EN->HI')
    p.add_argument('--config', default='configs/training.yaml')
    p.add_argument('--stage', default=None, choices=['A', 'B'])
    p.add_argument(
        '--curriculum',
        default=None,
        choices=['smoke', 'A1', 'A2', 'full', 'Bp'],
    )
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--resume-adapters', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--skip-gen-eval', action='store_true')
    a = p.parse_args()

    cfg = load_training_config(a.config)
    stage = str(a.stage or deep_get(cfg, 'run', 'stage', default='A')).upper()
    if stage not in ('A', 'B'):
        raise SystemExit(f'invalid stage {stage!r}; use A or B')
    curriculum = a.curriculum or deep_get(cfg, 'data', 'curriculum', default=None)
    if curriculum is None:
        curriculum = (
            'Bp'
            if stage == 'B'
            and deep_get(
                cfg,
                'data',
                'stage_b_replay',
                'enabled',
                default=False,
            )
            else ('smoke' if stage == 'A' else 'full')
        )
    if stage == 'B' and curriculum in ('smoke', 'A1', 'A2'):
        curriculum = 'full'

    run(
        config_path=a.config,
        stage=stage,
        curriculum=curriculum,
        max_steps=a.max_steps,
        resume_adapters=a.resume_adapters,
        device=a.device,
        skip_gen_eval=a.skip_gen_eval,
    )


if __name__ == '__main__':
    main()

"""NLLB EN->HI decode + dual-policy scores (zero-shot or PEFT adapters)."""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.evaluation.eval_sets import load_jsonl, scoring_suites
from src.evaluation.metrics_mt import score_pairs
from src.training.common import empty_device_cache, is_cuda


DEFAULT_MODEL = 'facebook/nllb-200-distilled-600M'
SRC_LANG = 'eng_Latn'
TGT_LANG = 'hin_Deva'
OUT_DIR = Path('data/analysis')
DEFAULT_SUITES = ['I_test', 'E_milpac_test', 'E_anuvaad_test']


def pick_device(prefer_mps: bool = True) -> str:
    if prefer_mps and torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_model(
    model_id: str = DEFAULT_MODEL,
    device: str | None = None,
    adapters: str | Path | None = None,
):
    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(adapters if adapters else model_id)
    if hasattr(tokenizer, 'src_lang'):
        tokenizer.src_lang = SRC_LANG

    if is_cuda(device) and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device in ('mps', 'cuda') or is_cuda(device):
        dtype = torch.float16
    else:
        dtype = torch.float32

    load_kw = {'dtype': dtype}
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

    if adapters:
        from peft import PeftModel

        from src.training.train_nllb_lora import apply_new_embed_rows

        model = PeftModel.from_pretrained(base, str(adapters))
        if apply_new_embed_rows(model, adapters):
            print(f'Applied new_embed_rows from {adapters}')
    else:
        model = base

    model.to(device)
    model.eval()
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.max_length = None
    return tokenizer, model, device


def _forced_bos_id(tokenizer) -> int:
    if hasattr(tokenizer, 'lang_code_to_id') and TGT_LANG in tokenizer.lang_code_to_id:
        return tokenizer.lang_code_to_id[TGT_LANG]
    tid = tokenizer.convert_tokens_to_ids(TGT_LANG)
    if tid is None or tid == tokenizer.unk_token_id:
        raise RuntimeError(f'cannot resolve target lang id for {TGT_LANG}')
    return tid


def translate_batch(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    max_input_length: int = 256,
    max_new_tokens: int = 256,
    num_beams: int = 4,
) -> list[str]:
    if not texts:
        return []
    enc = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )
    enc = {k: v.to(device, non_blocking=is_cuda(device)) for k, v in enc.items()}
    gen_kw = dict(
        forced_bos_token_id=_forced_bos_id(tokenizer),
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    with torch.no_grad():
        if is_cuda(device):
            dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.autocast(device_type='cuda', dtype=dt):
                out = model.generate(**enc, **gen_kw)
        else:
            out = model.generate(**enc, **gen_kw)
    hyps = tokenizer.batch_decode(out, skip_special_tokens=True)
    del enc, out
    if device == 'mps' and hasattr(torch, 'mps'):
        torch.mps.empty_cache()
    return hyps


def hyp_path_for(suite_name: str, tag: str = 'zero_shot_nllb') -> Path:
    return OUT_DIR / f'{tag}_{suite_name}_hyps.jsonl'


def load_existing_hyps(path: Path) -> list[dict]:
    return load_jsonl(path)


def translate_pairs(
    pairs: list[dict],
    tokenizer,
    model,
    device: str,
    suite_name: str,
    batch_size: int = 1,
    max_input_length: int = 256,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    resume: bool = True,
    verbose: bool = True,
    tag: str = 'zero_shot_nllb',
) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = hyp_path_for(suite_name, tag=tag)
    results: list[dict] = []
    done_keys: set[tuple[str, str]] = set()

    if resume and path.exists():
        results = load_existing_hyps(path)
        done_keys = {(r.get('en_text', ''), r.get('hi_text', '')) for r in results}
        if verbose and results:
            print(f'  resume: {len(results)} hyps already in {path}')

    pending = [p for p in pairs if (p.get('en_text', ''), p.get('hi_text', '')) not in done_keys]
    if not pending:
        return results

    t0 = time.time()
    with open(path, 'a' if results else 'w', encoding='utf-8') as f:
        n_pending = len(pending)
        for start in range(0, n_pending, batch_size):
            chunk = pending[start : start + batch_size]
            hyps = translate_batch(
                [p['en_text'] for p in chunk],
                tokenizer,
                model,
                device,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            for p, hyp in zip(chunk, hyps):
                row = {
                    'en_text': p['en_text'],
                    'hi_text': p['hi_text'],
                    'hyp_hi': hyp,
                    'doc_id': p.get('doc_id'),
                    'source': p.get('source'),
                }
                results.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            bi = start // batch_size
            if bi % 10 == 0:
                empty_device_cache(device)
            if verbose and (bi % 25 == 0 or start + batch_size >= n_pending):
                done = min(start + batch_size, n_pending)
                rate = done / max(time.time() - t0, 1e-6)
                print(
                    f'  translated +{done}/{n_pending} this run '
                    f'(total hyps {len(results)}, {rate:.2f} pairs/s)'
                )
    return results


def score_hyp_file(
    suite_name: str,
    path: Path | None = None,
    tag: str = 'zero_shot_nllb',
) -> dict:
    path = path or hyp_path_for(suite_name, tag=tag)
    rows = load_existing_hyps(path)
    if not rows:
        return {
            'suite': suite_name,
            'n': 0,
            'error': f'missing_hyps:{path}',
            'hypotheses': str(path),
        }
    scores = score_pairs([r['hyp_hi'] for r in rows], [r['hi_text'] for r in rows])
    return {
        'suite': suite_name,
        'path': str(path),
        'n': scores['n'],
        'bleu': scores['bleu'],
        'chrfpp': scores['chrfpp'],
        'hypotheses': str(path),
    }


def evaluate_suite(
    suite_name: str,
    path: Path,
    tokenizer,
    model,
    device: str,
    max_pairs: int | None = None,
    batch_size: int = 1,
    max_input_length: int = 256,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    resume: bool = True,
    verbose: bool = True,
    tag: str = 'zero_shot_nllb',
) -> dict:
    pairs = load_jsonl(path)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if verbose:
        print(f'\n=== Suite {suite_name} ({len(pairs)} pairs) path={path} ===')
    if not pairs:
        return {
            'suite': suite_name,
            'path': str(path),
            'n': 0,
            'error': 'empty_or_missing',
        }

    rows = translate_pairs(
        pairs,
        tokenizer,
        model,
        device,
        suite_name=suite_name,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        resume=resume,
        verbose=verbose,
        tag=tag,
    )
    key_to_hyp = {(r['en_text'], r['hi_text']): r['hyp_hi'] for r in rows}
    hyps, refs, missing = [], [], 0
    for p in pairs:
        k = (p['en_text'], p['hi_text'])
        if k not in key_to_hyp:
            missing += 1
            continue
        hyps.append(key_to_hyp[k])
        refs.append(p['hi_text'])
    scores = score_pairs(hyps, refs)
    out = {
        'suite': suite_name,
        'path': str(path),
        'n': scores['n'],
        'missing_hyps': missing,
        'bleu': scores['bleu'],
        'chrfpp': scores['chrfpp'],
        'hypotheses': str(hyp_path_for(suite_name, tag=tag)),
    }
    if verbose:
        print(
            f'  BLEU={scores["bleu"]["score"]:.2f}  '
            f'chrF++={scores["chrfpp"]["score"]:.2f}  n={scores["n"]}'
        )
    return out


def write_report(report: dict, verbose: bool = True) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = report.get('tag') or 'zero_shot_nllb'
    out_path = OUT_DIR / f'{tag}_report.json'
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if verbose:
        print(f'\nReport: {out_path}\nSummary:')
        for r in report.get('suites', []):
            if 'bleu' in r:
                print(
                    f'  {r["suite"]:<18} n={r["n"]:<5} '
                    f'BLEU={r["bleu"]["score"]:.2f}  chrF++={r["chrfpp"]["score"]:.2f}'
                )
    return out_path


def run(
    model_id: str = DEFAULT_MODEL,
    suites: list[str] | None = None,
    max_pairs: int | None = None,
    batch_size: int = 1,
    max_input_length: int = 256,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    device: str | None = None,
    score_only: bool = False,
    resume: bool = True,
    verbose: bool = True,
    adapters: str | None = None,
    tag: str | None = None,
) -> dict:
    suites = suites or list(DEFAULT_SUITES)
    available = scoring_suites()
    unknown = [s for s in suites if s not in available]
    if unknown:
        raise ValueError(f'unknown suites {unknown}; choose from {list(available)}')
    if tag is None:
        tag = 'nllb_lora' if adapters else 'zero_shot_nllb'

    if score_only:
        report = {
            'model_id': model_id,
            'adapters': adapters,
            'tag': tag,
            'mode': 'score_only',
            'suites': [score_hyp_file(n, tag=tag) for n in suites],
        }
        write_report(report, verbose=verbose)
        return report

    device = device or pick_device()
    if verbose:
        print(f'Model: {model_id}')
        if adapters:
            print(f'Adapters: {adapters}')
        print(f'Tag: {tag}\nDevice: {device}\nSuites: {suites}')
        if max_pairs:
            print(f'max_pairs per suite: {max_pairs}')

    t_load = time.time()
    tokenizer, model, device = load_model(model_id, device=device, adapters=adapters)
    if verbose:
        print(f'Loaded in {time.time() - t_load:.1f}s')

    results, t0 = [], time.time()
    for name in suites:
        results.append(
            evaluate_suite(
                name,
                available[name],
                tokenizer,
                model,
                device,
                max_pairs=max_pairs,
                batch_size=batch_size,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                resume=resume,
                verbose=verbose,
                tag=tag,
            )
        )
        empty_device_cache(device)

    report = {
        'model_id': model_id,
        'adapters': adapters,
        'tag': tag,
        'device': device,
        'src_lang': SRC_LANG,
        'tgt_lang': TGT_LANG,
        'max_input_length': max_input_length,
        'max_new_tokens': max_new_tokens,
        'num_beams': num_beams,
        'batch_size': batch_size,
        'max_pairs': max_pairs,
        'elapsed_s': round(time.time() - t0, 1),
        'suites': results,
    }
    write_report(report, verbose=verbose)
    return report


def main():
    import argparse

    p = argparse.ArgumentParser(description='NLLB EN->HI on dual eval policies')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--suites', default=','.join(DEFAULT_SUITES))
    p.add_argument('--max-pairs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--max-input-length', type=int, default=256)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--num-beams', type=int, default=4)
    p.add_argument('--device', default=None)
    p.add_argument('--adapters', default=None)
    p.add_argument('--tag', default=None)
    p.add_argument('--score-only', action='store_true')
    p.add_argument('--no-resume', action='store_true')
    a = p.parse_args()
    run(
        model_id=a.model,
        suites=[s.strip() for s in a.suites.split(',') if s.strip()],
        max_pairs=a.max_pairs,
        batch_size=a.batch_size,
        max_input_length=a.max_input_length,
        max_new_tokens=a.max_new_tokens,
        num_beams=a.num_beams,
        device=a.device,
        score_only=a.score_only,
        resume=not a.no_resume,
        adapters=a.adapters,
        tag=a.tag,
    )


if __name__ == '__main__':
    main()

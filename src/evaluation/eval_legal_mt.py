"""Eval Track C1 Marian+SPM checkpoint on dual-policy test suites."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import MarianMTModel

from src.evaluation.eval_sets import load_jsonl, scoring_suites
from src.evaluation.metrics_mt import score_pairs
from src.evaluation.zero_shot_nllb import pick_device
from src.training.spm_tokenizer import LegalSpmTokenizer


@torch.no_grad()
def translate_pairs(
    pairs,
    tokenizer,
    model,
    device,
    max_input_length=256,
    max_new_tokens=256,
    num_beams=4,
    batch_size=16,
) -> list[str]:
    hyps, pad_id = [], tokenizer.pad_token_id
    model.eval()
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        enc = [tokenizer.encode(p['en_text'], max_length=max_input_length) for p in chunk]
        max_len = max(len(x) for x in enc) if enc else 1
        input_ids = torch.tensor(
            [x + [pad_id] * (max_len - len(x)) for x in enc],
            dtype=torch.long,
            device=device,
        )
        out = model.generate(
            input_ids=input_ids,
            attention_mask=(input_ids != pad_id).long(),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            decoder_start_token_id=tokenizer.bos_token_id,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        hyps.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return hyps


def run(
    checkpoint: str,
    suites: list[str] | None = None,
    device: str | None = None,
    batch_size: int = 16,
    num_beams: int = 4,
    max_input_length: int = 256,
    max_new_tokens: int = 256,
    tag: str = 'legal_mt',
    verbose: bool = True,
) -> dict:
    suites = suites or ['I_test', 'E_milpac_test', 'E_anuvaad_test']
    available = scoring_suites()
    device = device or pick_device()
    ckpt = Path(checkpoint)
    tokenizer = LegalSpmTokenizer.from_pretrained(ckpt)
    model = MarianMTModel.from_pretrained(ckpt)
    model.to(device)
    model.eval()

    out_dir = Path('data/analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    results, t0 = [], time.time()
    for name in suites:
        path = available[name]
        pairs = load_jsonl(path)
        if verbose:
            print(f'=== {name} n={len(pairs)} ===')
        hyps = translate_pairs(
            pairs,
            tokenizer,
            model,
            device,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            batch_size=batch_size,
        )
        scores = score_pairs(hyps, [p['hi_text'] for p in pairs])
        hyp_path = out_dir / f'{tag}_{name}_hyps.jsonl'
        with open(hyp_path, 'w', encoding='utf-8') as f:
            for p, h in zip(pairs, hyps):
                f.write(
                    json.dumps(
                        {
                            'en_text': p['en_text'],
                            'hi_text': p['hi_text'],
                            'hyp_hi': h,
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )
        results.append(
            {
                'suite': name,
                'path': str(path),
                'n': scores['n'],
                'bleu': scores['bleu'],
                'chrfpp': scores['chrfpp'],
                'hypotheses': str(hyp_path),
            }
        )
        if verbose:
            print(f'  BLEU={scores["bleu"]["score"]:.2f} chrF++={scores["chrfpp"]["score"]:.2f}')

    report = {
        'track': 'C1',
        'tag': tag,
        'checkpoint': str(ckpt),
        'device': device,
        'num_beams': num_beams,
        'batch_size': batch_size,
        'elapsed_s': round(time.time() - t0, 1),
        'suites': results,
    }
    report_path = out_dir / f'{tag}_report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if verbose:
        print(f'Report: {report_path}')
    return report


def main():
    import argparse

    p = argparse.ArgumentParser(description='Eval Track C1 legal MT checkpoint')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--suites', default='I_test,E_milpac_test,E_anuvaad_test')
    p.add_argument('--device', default=None)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-beams', type=int, default=4)
    p.add_argument('--tag', default='legal_mt')
    a = p.parse_args()
    run(
        checkpoint=a.checkpoint,
        suites=[s.strip() for s in a.suites.split(',') if s.strip()],
        device=a.device,
        batch_size=a.batch_size,
        num_beams=a.num_beams,
        tag=a.tag,
    )


if __name__ == '__main__':
    main()

"""COMET-22 (Unbabel/wmt22-comet-da) scoring for all *_hyps.jsonl files.

Reference-based COMET. Each hyp file needs `en_text`, `hi_text`, `hyp_hi`.
Writes one summary JSON with { tag: { suite: {score, n, fingerprint} } }.

Phase 4: the cache is keyed on (tag, suite, hyp-file SHA256 prefix, model_id),
so regenerating a hyp file under an existing tag (decode fix, resume that
changed rows, different COMET model) re-scores instead of silently reporting
the stale score. The summary schema is bumped to 'v2'; pre-v2 cache files are
ignored entirely.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from src.evaluation.fingerprint import file_sha256_prefix
from src.utils.jsonl import load_jsonl


DEFAULT_MODEL = 'Unbabel/wmt22-comet-da'
DEFAULT_ANALYSIS = Path('data/analysis')
DEFAULT_SUMMARY = DEFAULT_ANALYSIS / 'comet22_summary.json'
SUMMARY_SCHEMA = 'v2'

_SUITE_ORDER = (
    'I_test',
    'I_dev',
    'E_milpac_test',
    'E_milpac_dev',
    'E_anuvaad_test',
    'E_anuvaad_dev_sample',
)
_SUITE_PAT = re.compile(r'_(' + '|'.join(re.escape(s) for s in _SUITE_ORDER) + r')_hyps\.jsonl$')


def parse_hyp_path(path: Path) -> tuple[str, str] | None:
    m = _SUITE_PAT.search(path.name)
    if not m:
        return None
    suite = m.group(1)
    tag = path.name[: -(len(suite) + len('_hyps.jsonl') + 1)]
    return tag, suite


def should_rescore(entry: dict | None, fingerprint: str, model_id: str) -> bool:
    if not entry or entry.get('score') is None:
        return True
    if entry.get('fingerprint') != fingerprint:
        return True
    if entry.get('model_id') != model_id:
        return True
    return False


def load_summary(summary_path: Path) -> dict[str, dict[str, dict]]:
    if not summary_path.exists():
        return {}
    payload = json.loads(summary_path.read_text(encoding='utf-8'))
    if payload.get('schema') != SUMMARY_SCHEMA:
        print('  old comet cache schema found; ignoring cache and rewriting as v2')
        return {}
    return payload.get('systems', {})


def save_summary(summary_path: Path, model_id: str, systems: dict[str, dict[str, dict]]):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'schema': SUMMARY_SCHEMA, 'model': model_id, 'systems': systems}
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def score_file(model, path: Path, gpus: int, batch_size: int) -> dict:
    rows = load_jsonl(path)
    data = [
        {'src': r['en_text'], 'mt': r['hyp_hi'], 'ref': r['hi_text']}
        for r in rows
        if r.get('en_text') and r.get('hyp_hi') and r.get('hi_text')
    ]
    if not data:
        return {'n': 0, 'score': None, 'error': 'no valid rows'}
    result = model.predict(data, batch_size=batch_size, gpus=gpus)
    return {'n': len(data), 'score': round(float(result.system_score), 4)}


def run(
    model_id: str = DEFAULT_MODEL,
    analysis_dir: Path = DEFAULT_ANALYSIS,
    summary_path: Path = DEFAULT_SUMMARY,
    gpus: int = 1,
    batch_size: int = 32,
    only_tag: str | None = None,
    only_suite: str | None = None,
) -> dict:
    from comet import download_model, load_from_checkpoint

    ckpt = download_model(model_id)
    model = load_from_checkpoint(ckpt)

    paths: list[Path] = sorted(analysis_dir.glob('*_hyps.jsonl'))
    summary = load_summary(Path(summary_path))

    for path in paths:
        parsed = parse_hyp_path(path)
        if not parsed:
            continue
        tag, suite = parsed
        if only_tag and tag != only_tag:
            continue
        if only_suite and suite != only_suite:
            continue
        fingerprint = file_sha256_prefix(path)
        entry = summary.get(tag, {}).get(suite)
        if not should_rescore(entry, fingerprint, model_id):
            print(f'  skip (cached): {tag} / {suite}')
            continue
        t0 = time.time()
        r = score_file(model, path, gpus=gpus, batch_size=batch_size)
        r['elapsed_s'] = round(time.time() - t0, 1)
        r['path'] = str(path)
        r['fingerprint'] = fingerprint
        r['model_id'] = model_id
        summary.setdefault(tag, {})[suite] = r
        print(
            f'  {tag} / {suite}: '
            f'{"score=" + format(r["score"], ".4f") if r.get("score") is not None else r.get("error")}  '
            f'n={r["n"]}  {r["elapsed_s"]}s'
        )
        save_summary(Path(summary_path), model_id, summary)

    return {'model': model_id, 'systems': summary}


def main():
    p = argparse.ArgumentParser(description='COMET-22 scoring over data/analysis/*_hyps.jsonl')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--analysis-dir', type=Path, default=DEFAULT_ANALYSIS)
    p.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY)
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--only-tag', default=None)
    p.add_argument('--only-suite', default=None)
    a = p.parse_args()
    run(
        model_id=a.model,
        analysis_dir=a.analysis_dir,
        summary_path=a.summary,
        gpus=a.gpus,
        batch_size=a.batch_size,
        only_tag=a.only_tag,
        only_suite=a.only_suite,
    )


if __name__ == '__main__':
    main()

"""
Load dual evaluation policies for scoring MT systems.

Policy I -- internal assignment (frozen doc-level split).
Policy E -- external held-out (MILPaC + Anuvaad carve from Stage A).

Usage:
  PYTHONPATH=. python3 -m src.evaluation.eval_sets
  PYTHONPATH=. python3 -m src.evaluation.eval_sets --validate
"""

import json
from pathlib import Path

from src.config import DEV_DOC_IDS, TEST_DOC_IDS, TRAIN_DOC_IDS

PROCESSED = Path('data/processed')
EVAL_DIR = Path('data/external/parallel/eval')
STAGE_A_TRAIN = Path('data/external/parallel/stage_a_train.jsonl')
MANIFEST = EVAL_DIR / 'eval_manifest.json'


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pair_key(p: dict) -> tuple[str, str]:
    return (p.get('en_text', '').strip(), p.get('hi_text', '').strip())


def policy_i() -> dict[str, list[dict]]:
    return {
        'train': load_jsonl(PROCESSED / 'train.jsonl'),
        'dev': load_jsonl(PROCESSED / 'dev.jsonl'),
        'test': load_jsonl(PROCESSED / 'test.jsonl'),
    }


def policy_e() -> dict[str, list[dict]]:
    return {
        'stage_a_train': load_jsonl(STAGE_A_TRAIN),
        'milpac_dev': load_jsonl(EVAL_DIR / 'milpac_dev.jsonl'),
        'milpac_test': load_jsonl(EVAL_DIR / 'milpac_test.jsonl'),
        'anuvaad_dev': load_jsonl(EVAL_DIR / 'anuvaad_dev.jsonl'),
        'anuvaad_test': load_jsonl(EVAL_DIR / 'anuvaad_test.jsonl'),
        'external_dev': load_jsonl(EVAL_DIR / 'external_dev.jsonl'),
        'external_test': load_jsonl(EVAL_DIR / 'external_test.jsonl'),
    }


def scoring_suites() -> dict[str, Path]:
    """Named suites every model should report on."""
    return {
        'I_test': PROCESSED / 'test.jsonl',
        'I_dev': PROCESSED / 'dev.jsonl',
        'E_milpac_test': EVAL_DIR / 'milpac_test.jsonl',
        'E_anuvaad_test': EVAL_DIR / 'anuvaad_test.jsonl',
        'E_external_test': EVAL_DIR / 'external_test.jsonl',
    }


def validate_policies(verbose: bool = True) -> dict:
    """Check Policy I doc IDs and no train/eval pair overlap for Policy E."""
    report: dict = {'ok': True, 'errors': [], 'counts': {}}

    pi = policy_i()
    pe = policy_e()

    for name, rows in {**{f'I_{k}': v for k, v in pi.items()}, **{f'E_{k}': v for k, v in pe.items()}}.items():
        report['counts'][name] = len(rows)

    # Policy I doc IDs
    def docs(rows):
        out = set()
        for p in rows:
            d = p.get('doc_id')
            if isinstance(d, str) and d.isdigit():
                d = int(d)
            if isinstance(d, int):
                out.add(d)
        return out

    train_docs, dev_docs, test_docs = docs(pi['train']), docs(pi['dev']), docs(pi['test'])
    if train_docs and train_docs != set(TRAIN_DOC_IDS):
        # allow subset if files partial
        if not train_docs.issubset(set(TRAIN_DOC_IDS)):
            report['ok'] = False
            report['errors'].append(f'I_train unexpected docs: {sorted(train_docs - set(TRAIN_DOC_IDS))}')
    if dev_docs - set(DEV_DOC_IDS):
        report['ok'] = False
        report['errors'].append(f'I_dev unexpected docs: {sorted(dev_docs - set(DEV_DOC_IDS))}')
    if test_docs - set(TEST_DOC_IDS):
        report['ok'] = False
        report['errors'].append(f'I_test unexpected docs: {sorted(test_docs - set(TEST_DOC_IDS))}')
    if train_docs & test_docs:
        report['ok'] = False
        report['errors'].append('I_train and I_test share doc_ids')
    if train_docs & dev_docs:
        report['ok'] = False
        report['errors'].append('I_train and I_dev share doc_ids')

    # Policy E overlap
    train_keys = {pair_key(p) for p in pe['stage_a_train']}
    for label in ('milpac_test', 'milpac_dev', 'anuvaad_test', 'anuvaad_dev'):
        leaked = sum(1 for p in pe[label] if pair_key(p) in train_keys)
        if leaked:
            report['ok'] = False
            report['errors'].append(f'E_{label}: {leaked} pairs also in stage_a_train')

    if not STAGE_A_TRAIN.exists():
        report['ok'] = False
        report['errors'].append('stage_a_train.jsonl missing; run split_external_eval')
    if not MANIFEST.exists():
        report['ok'] = False
        report['errors'].append('eval_manifest.json missing; run split_external_eval')

    if verbose:
        print('Policy validation:', 'OK' if report['ok'] else 'FAILED')
        for k, v in sorted(report['counts'].items()):
            print(f'  {k}: {v:,}')
        for e in report['errors']:
            print(f'  ERROR: {e}')
        print('Scoring suites:')
        for name, path in scoring_suites().items():
            n = len(load_jsonl(path)) if path.exists() else 0
            print(f'  {name}: {path} ({n} pairs)')
    return report


def run(validate: bool = True, verbose: bool = True) -> dict:
    if validate:
        return validate_policies(verbose=verbose)
    suites = {k: str(v) for k, v in scoring_suites().items()}
    if verbose:
        print(json.dumps(suites, indent=2))
    return {'suites': suites}


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Dual eval policy loaders / validation')
    parser.add_argument('--validate', action='store_true', default=True)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    if args.list_only:
        run(validate=False, verbose=True)
    else:
        rep = validate_policies(verbose=True)
        if not rep['ok']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Tests for dual eval Policy E carve and Policy I/E validation."""

import json
from pathlib import Path

from src.evaluation.eval_sets import pair_key, validate_policies
from src.preprocessing.split_external_eval import run as split_run


def _write_pairs(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


class TestSplitExternalEval:
    def test_no_leak_and_counts(self, tmp_path: Path, monkeypatch):
        import src.preprocessing.split_external_eval as mod

        pool = tmp_path / 'stage_a_en_hi.jsonl'
        rows = []
        for i in range(100):
            rows.append(
                {
                    'en_text': f'Milpac English sentence number {i} about courts.',
                    'hi_text': f'मिल्पैक हिंदी वाक्य संख्या {i} न्यायालय।',
                    'source': 'milpac_Acts',
                    'doc_id': f'm:{i}',
                }
            )
        for i in range(500):
            rows.append(
                {
                    'en_text': f'Anuvaad English judgment line {i} for appeal.',
                    'hi_text': f'अनुवाद हिंदी निर्णय पंक्ति {i} अपील।',
                    'source': 'anuvaad_judiciary',
                    'doc_id': f'a:{i}',
                }
            )
        _write_pairs(pool, rows)

        train_path = tmp_path / 'stage_a_train.jsonl'
        eval_dir = tmp_path / 'eval'
        monkeypatch.setattr(mod, 'STAGE_A_ALL', pool)
        monkeypatch.setattr(mod, 'STAGE_A_TRAIN', train_path)
        monkeypatch.setattr(mod, 'EVAL_DIR', eval_dir)
        monkeypatch.setattr(mod, 'PARALLEL_DIR', tmp_path)

        man = split_run(
            stage_a_path=pool,
            seed=42,
            anuvaad_dev_n=20,
            anuvaad_test_n=50,
            milpac_dev_frac=0.1,
            milpac_test_frac=0.1,
            verbose=False,
        )

        assert man['counts']['milpac_test'] >= 1
        assert man['counts']['anuvaad_test'] == 50
        assert man['counts']['stage_a_train'] + man['counts']['milpac_dev'] + man[
            'counts'
        ]['milpac_test'] + man['counts']['anuvaad_dev'] + man['counts']['anuvaad_test'] == 600

        train = [
            json.loads(l)
            for l in train_path.read_text(encoding='utf-8').splitlines()
            if l.strip()
        ]
        test_a = [
            json.loads(l)
            for l in (eval_dir / 'anuvaad_test.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()
        ]
        train_keys = {pair_key(p) for p in train}
        for p in test_a:
            assert pair_key(p) not in train_keys


class TestEvalSetsSmoke:
    def test_validate_if_artifacts_exist(self):
        # Live data: only assert structure if split already run
        if not Path('data/external/parallel/stage_a_train.jsonl').exists():
            return
        rep = validate_policies(verbose=False)
        assert 'counts' in rep
        if rep['ok']:
            assert rep['counts'].get('E_stage_a_train', 0) > 0

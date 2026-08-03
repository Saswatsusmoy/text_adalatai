"""Tests for checkpoint selection: z-score primary + anti-forget caps."""

import json
from pathlib import Path

from src.training.selection import (
    baseline_from_eval_log,
    evaluate_selection,
    load_baseline,
    parse_caps,
    selection_mode,
    weighted_chrfpp_primary,
)


def _gen(i: float, e: float) -> dict:
    return {
        'I_dev': {'chrfpp': {'score': i}, 'bleu': {'score': 0.0}},
        'E_milpac_dev': {'chrfpp': {'score': e}, 'bleu': {'score': 0.0}},
    }


WEIGHTS = {'I_dev': 0.7, 'E_milpac_dev': 0.3}
CAPS = {'I_dev': 2.0, 'E_milpac_dev': 3.0}


def test_weighted_primary_raw():
    gen = _gen(50.0, 60.0)
    assert weighted_chrfpp_primary(gen, WEIGHTS) == 0.7 * 50.0 + 0.3 * 60.0


def test_parse_caps_from_stage_b_keys_and_dict():
    sel = {
        'stage_b_max_drop_I_dev': 2.0,
        'stage_b_max_drop_E_milpac_dev': '3.0',
        'caps': {'X': 1.5},
    }
    caps = parse_caps(sel)
    assert caps['I_dev'] == 2.0
    assert caps['E_milpac_dev'] == 3.0
    assert caps['X'] == 1.5


def _baseline():
    return {
        'path': 'x',
        'step': 200,
        'primary': 53.2,
        'values': {'I_dev': 52.0, 'E_milpac_dev': 56.0},
        'mean': {'I_dev': 51.0, 'E_milpac_dev': 56.0},
        'std': {'I_dev': 0.81649658, 'E_milpac_dev': 0.81649658},
        'n_evals': 3,
    }


def test_zscore_primary_hand_computed():
    res = evaluate_selection(_gen(51.0, 57.0), WEIGHTS, _baseline(), CAPS)
    z_e = (57.0 - 56.0) / 0.81649658
    assert res['mode'] == 'zscore'
    assert res['primary'] == round(0.3 * z_e, 6)
    assert res['z']['I_dev'] == 0.0
    assert res['cap_ok'] is True


def test_cap_rejects_checkpoint_with_e_drop():
    res = evaluate_selection(_gen(52.5, 49.0), WEIGHTS, _baseline(), CAPS)
    assert res['cap_ok'] is False
    assert res['cap_violations'] == {'E_milpac_dev': 7.0}


def test_cap_rejects_even_when_primary_high():
    res = evaluate_selection(_gen(60.0, 40.0), WEIGHTS, _baseline(), CAPS)
    assert res['cap_ok'] is False
    assert res['cap_violations']['E_milpac_dev'] > 0


def test_no_baseline_falls_back_to_raw_weighted():
    res = evaluate_selection(_gen(50.0, 60.0), WEIGHTS, None, CAPS)
    assert res['mode'] == 'raw'
    assert res['primary'] == 0.7 * 50.0 + 0.3 * 60.0
    assert res['cap_ok'] is True


def test_zscore_delta_fallback_when_std_zero():
    baseline = {
        'path': 'x',
        'step': None,
        'primary': None,
        'values': {'I_dev': 52.0, 'E_milpac_dev': 56.0},
        'mean': {'I_dev': 52.0, 'E_milpac_dev': 56.0},
        'std': {'I_dev': 0.0, 'E_milpac_dev': 0.0},
        'n_evals': 1,
    }
    res = evaluate_selection(_gen(54.0, 55.0), WEIGHTS, baseline, {})
    assert res['mode'] == 'zscore'
    assert res['primary'] == round(0.7 * 2.0 + 0.3 * -1.0, 6)


def _write_eval_log(path: Path):
    rows = [
        {
            'step': 100,
            'type': 'gen_eval',
            'I_dev': {'chrfpp': {'score': 50.0}, 'bleu': {}},
            'E_milpac_dev': {'chrfpp': {'score': 55.0}, 'bleu': {}},
        },
        {
            'step': 200,
            'type': 'gen_eval',
            'I_dev': {'chrfpp': {'score': 52.0}, 'bleu': {}},
            'E_milpac_dev': {'chrfpp': {'score': 56.0}, 'bleu': {}},
        },
        {
            'step': 300,
            'type': 'gen_eval',
            'I_dev': {'chrfpp': {'score': 51.0}, 'bleu': {}},
            'E_milpac_dev': {'chrfpp': {'score': 57.0}, 'bleu': {}},
        },
    ]
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')


def test_baseline_from_eval_log_picks_best_row_and_stats(tmp_path: Path):
    p = tmp_path / 'eval_log.jsonl'
    _write_eval_log(p)
    base = baseline_from_eval_log(p, WEIGHTS)
    assert base is not None
    assert base['step'] == 200
    assert base['values'] == {'I_dev': 52.0, 'E_milpac_dev': 56.0}
    assert base['mean'] == {'I_dev': 51.0, 'E_milpac_dev': 56.0}
    assert base['n_evals'] == 3


def test_load_baseline_from_resume_adapters_run_dir(tmp_path: Path):
    run_dir = tmp_path / 'runs' / 'nllb600_A_A2'
    metrics = run_dir / 'metrics'
    metrics.mkdir(parents=True)
    _write_eval_log(metrics / 'eval_log.jsonl')
    (run_dir / 'config.snapshot.yaml').write_text(
        'run:\n  stage: A\nmodel: {id: x}\ndata: {}\noptim: {}\ntrain: {}\n'
        'eval:\n  selection:\n    stage_a_weights:\n      I_dev: 0.7\n      E_milpac_dev: 0.3\n',
        encoding='utf-8',
    )
    adapters = run_dir / 'checkpoints' / 'best_primary'
    adapters.mkdir(parents=True)
    cfg = {'eval': {'selection': {'stage_a_weights': WEIGHTS, 'stage_b_weights': WEIGHTS}}}
    base = load_baseline(cfg, 'B', adapters)
    assert base is not None
    assert base['step'] == 200


def test_load_baseline_explicit_config(tmp_path: Path):
    p = tmp_path / 'eval_log.jsonl'
    _write_eval_log(p)
    cfg = {
        'eval': {
            'selection': {
                'baseline': str(p),
                'stage_a_weights': WEIGHTS,
                'stage_b_weights': WEIGHTS,
            }
        }
    }
    base = load_baseline(cfg, 'A', None)
    assert base is not None
    assert base['values']['I_dev'] == 52.0


def test_load_baseline_none_without_resume_or_explicit(tmp_path: Path):
    cfg = {'eval': {'selection': {'stage_a_weights': WEIGHTS, 'stage_b_weights': WEIGHTS}}}
    assert load_baseline(cfg, 'B', None) is None


def test_selection_mode_gating():
    base_cfg = {'eval': {'selection': {'stage_a_weights': WEIGHTS, 'stage_b_weights': WEIGHTS}}}
    assert selection_mode(base_cfg, 'A') == 'raw'
    assert selection_mode(base_cfg, 'B') == 'zscore'
    cfg = {'eval': {'selection': {'baseline': 'x.json'}}}
    assert selection_mode(cfg, 'A') == 'zscore'
    off = {'eval': {'selection': {'zscore': False}}}
    assert selection_mode(off, 'B') == 'raw'

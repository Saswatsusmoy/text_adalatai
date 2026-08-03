"""Checkpoint selection: weighted primary, z-scoring, anti-forget caps.

Stage A keeps the raw weighted mean of chrF++ scores (unchanged behavior).
Stage B (or an explicit eval.selection.baseline) switches to z-scored primary
plus hard anti-forgetting caps against the resumed checkpoint's eval history.

Z-score formula (documented in docs/TRAINING_STRATEGY.md and DESIGN_DECISIONS):

    z_i = (s_i - mean_i) / std_i            if std_i > 0
    z_i = s_i - b_i                          else (single-row baseline, delta)

    primary = sum_i(w_i * z_i) / sum_i(w_i)

mean_i/std_i are the population statistics of suite i over all gen_eval rows
of the baseline eval log; b_i is the chrF++ at the baseline run's best row.
Caps reject a candidate when b_i - s_i > cap_i for any capped suite.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.training.config import deep_get, load_training_config


def weighted_chrfpp_primary(gen: dict, weights: dict) -> float:
    primary, wsum = 0.0, 0.0
    for key, w in weights.items():
        block = gen.get(key)
        if block and 'chrfpp' in block:
            primary += float(w) * float(block['chrfpp']['score'])
            wsum += float(w)
    return primary / wsum if wsum > 0 else 0.0


def parse_caps(selection_cfg: dict) -> dict:
    caps = dict(selection_cfg.get('caps') or {})
    for k, v in selection_cfg.items():
        if k.startswith('stage_b_max_drop_'):
            caps[k[len('stage_b_max_drop_') :]] = float(v)
    return caps


def _chrfpp_block(block) -> float | None:
    if not isinstance(block, dict) or 'chrfpp' not in block:
        return None
    return float(block['chrfpp']['score'])


def _suite_values(gen: dict, suites) -> dict:
    out = {}
    for s in suites:
        sc = _chrfpp_block(gen.get(s))
        if sc is not None:
            out[s] = sc
    return out


def _pop_stats(samples: list[float]) -> tuple[float, float]:
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    mean = sum(samples) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in samples) / n
    return mean, var**0.5


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def baseline_from_eval_log(path: Path | str, weights: dict) -> dict | None:
    scored = []
    for r in _read_jsonl(Path(path)):
        if r.get('type') != 'gen_eval':
            continue
        vals = _suite_values(r, weights)
        if not vals:
            continue
        scored.append((weighted_chrfpp_primary(r, weights), vals, r.get('step')))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0])
    primary, values, step = scored[-1]
    hist = {s: [vals[s] for _, vals, _ in scored if s in vals] for s in values}
    mean = {s: _pop_stats(v)[0] for s, v in hist.items()}
    std = {s: _pop_stats(v)[1] for s, v in hist.items()}
    return {
        'path': str(path),
        'step': step,
        'primary': primary,
        'values': values,
        'mean': mean,
        'std': std,
        'n_evals': len(scored),
    }


def _report_from_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get('suites'), list):
        return None
    values = {}
    for s in data['suites']:
        name = s.get('suite') if isinstance(s, dict) else None
        sc = _chrfpp_block(s) if isinstance(s, dict) else None
        if name and sc is not None:
            values[name] = sc
    if not values:
        return None
    return {
        'path': str(path),
        'step': None,
        'primary': None,
        'values': values,
        'mean': dict(values),
        'std': dict.fromkeys(values, 0.0),
        'n_evals': 1,
    }


def _run_dir_from_adapters(adapters: str | Path) -> Path:
    p = Path(adapters)
    if (p / 'metrics' / 'eval_log.jsonl').is_file():
        return p
    if len(p.parents) >= 2:
        return p.parents[1]
    return p


def _eval_log_from_adapters(adapters: str | Path) -> Path | None:
    p = Path(adapters)
    if (p / 'metrics' / 'eval_log.jsonl').is_file():
        return p / 'metrics' / 'eval_log.jsonl'
    cand = _run_dir_from_adapters(adapters) / 'metrics' / 'eval_log.jsonl'
    return cand if cand.is_file() else None


def _snapshot_weights(run_dir: Path) -> dict:
    snap = run_dir / 'config.snapshot.yaml'
    if not snap.is_file():
        return {}
    try:
        snap_cfg = load_training_config(snap)
    except Exception:
        return {}
    sstage = str(deep_get(snap_cfg, 'run', 'stage', default='A')).upper()
    wkey = 'stage_b_weights' if sstage == 'B' else 'stage_a_weights'
    w = deep_get(snap_cfg, 'eval', 'selection', wkey, default={}) or {}
    return dict(w)


def _current_weights(cfg: dict, stage: str) -> dict:
    wkey = 'stage_b_weights' if stage == 'B' else 'stage_a_weights'
    w = deep_get(cfg, 'eval', 'selection', wkey, default={}) or {}
    return dict(w)


def load_baseline(cfg: dict, stage: str, resume_adapters: str | Path | None) -> dict | None:
    sel = deep_get(cfg, 'eval', 'selection', default={}) or {}
    explicit = sel.get('baseline')
    if explicit:
        p = Path(explicit)
        if p.is_dir():
            ep = _eval_log_from_adapters(p)
            if ep is not None:
                return baseline_from_eval_log(ep, _current_weights(cfg, stage))
            return _report_from_json(p)
        if p.suffix == '.json':
            return _report_from_json(p)
        return baseline_from_eval_log(p, _current_weights(cfg, stage))
    if not resume_adapters:
        return None
    ep = _eval_log_from_adapters(resume_adapters)
    if ep is None:
        return None
    run_dir = _run_dir_from_adapters(resume_adapters)
    weights = _snapshot_weights(run_dir) or _current_weights(cfg, stage)
    return baseline_from_eval_log(ep, weights)


def selection_mode(cfg: dict, stage: str) -> str:
    sel = deep_get(cfg, 'eval', 'selection', default={}) or {}
    z = sel.get('zscore')
    if z is not None:
        return 'zscore' if bool(z) else 'raw'
    if sel.get('baseline') is not None:
        return 'zscore'
    return 'zscore' if stage == 'B' else 'raw'


def evaluate_selection(gen: dict, weights: dict, baseline: dict | None, caps: dict) -> dict:
    values = _suite_values(gen, weights)
    mode, z = 'raw', {}
    if baseline is not None:
        wsum, psum = 0.0, 0.0
        for suite, w in weights.items():
            if suite not in values:
                continue
            mean = (baseline.get('mean') or {}).get(suite, values[suite])
            std = (baseline.get('std') or {}).get(suite, 0.0)
            if std and std > 0:
                zz = (values[suite] - mean) / std
            else:
                base_val = (baseline.get('values') or {}).get(suite, mean)
                zz = values[suite] - base_val
            z[suite] = zz
            wsum += float(w)
            psum += float(w) * zz
        primary = psum / wsum if wsum > 0 else 0.0
        mode = 'zscore'
    else:
        primary = weighted_chrfpp_primary(gen, weights)
    violations = {}
    if caps and baseline is not None:
        bvals = baseline.get('values') or {}
        for suite, cap in caps.items():
            if suite in values and suite in bvals and bvals[suite] - values[suite] > float(cap):
                violations[suite] = round(bvals[suite] - values[suite], 4)
    return {
        'primary': round(primary, 6),
        'mode': mode,
        'cap_ok': not violations,
        'cap_violations': violations,
        'z': {k: round(v, 6) for k, v in z.items()},
    }

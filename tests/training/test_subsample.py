"""Tests for Stage A curriculum subsample and Stage B replay mix."""

import json
from pathlib import Path

from src.training.subsample import build_stage_b_replay_mix, build_subsample, write_jsonl


def _minimal_cfg(
    tmp_path: Path,
    *,
    stage_a: Path,
    stage_b: Path,
    replay_pool: Path | None = None,
    assignment_frac: float = 0.9,
) -> Path:
    pool = stage_a
    cfg = tmp_path / 'training.yaml'
    replay_block = ''
    if replay_pool is not None:
        replay_block = f"""
  stage_b_replay:
    enabled: true
    pool: {replay_pool}
    assignment_frac: {assignment_frac}
    seed: 42
"""
    cfg.write_text(
        f"""
run: {{seed: 42, output_root: {tmp_path / 'runs'}}}
model: {{id: x, src_lang: eng_Latn, tgt_lang: hin_Deva}}
peft: {{r: 8, lora_alpha: 16, lora_dropout: 0.1, target_modules: [q_proj], bias: none, task_type: SEQ_2_SEQ_LM}}
data:
  stage_a_train: {pool}
  stage_b_train: {stage_b}
{replay_block}
  subsample:
    seed: 42
    smoke_n: 20
    A1_max_n: 30
    A2_max_n: 40
    prefer_sources: [milpac_Acts, anuvaad_judiciary]
    judiciary_cap_A1: 15
  max_source_length: 64
  max_target_length: 64
optim: {{lr: 1.0e-4}}
train: {{batch_size: 1, grad_accum_steps: 1, max_steps_stage_a: 10}}
eval:
  policy_I_dev: {pool}
  policy_E_milpac_dev: {pool}
  policy_E_anuvaad_dev: {pool}
""",
        encoding='utf-8',
    )
    return cfg


class TestSubsample:
    def test_smoke_builds(self, tmp_path: Path, monkeypatch):
        import src.training.subsample as mod

        pool = tmp_path / 'stage_a_train.jsonl'
        rows = []
        for i in range(50):
            rows.append(
                {
                    'en_text': f'English court sentence {i}.',
                    'hi_text': f'हिंदी न्यायालय वाक्य {i}।',
                    'source': 'milpac_Acts' if i < 10 else 'anuvaad_judiciary',
                }
            )
        write_jsonl(pool, rows)
        cfg = _minimal_cfg(tmp_path, stage_a=pool, stage_b=pool)
        monkeypatch.setattr(mod, 'OUT_DIR', tmp_path / 'subsamples')
        man = build_subsample(curriculum='smoke', config_path=cfg, verbose=False)
        assert man['n'] == 20
        assert Path(man['output']).exists()


class TestStageBReplayMix:
    def test_ninety_ten_counts_and_roles(self, tmp_path: Path, monkeypatch):
        import src.training.subsample as mod

        assign = tmp_path / 'assign.jsonl'
        replay = tmp_path / 'replay.jsonl'
        assign_rows = [
            {
                'en_text': f'Assignment English {i}.',
                'hi_text': f'असाइनमेंट हिंदी {i}।',
                'source': 'preprocessed',
                'doc_id': i,
            }
            for i in range(90)
        ]
        # Include one duplicate of assignment to prove exact-pair exclusion
        replay_rows = [
            {
                'en_text': f'Replay English {i}.',
                'hi_text': f'रीप्ले हिंदी {i}।',
                'source': 'milpac_Acts' if i % 2 == 0 else 'anuvaad_judiciary',
                'doc_id': f'r{i}',
            }
            for i in range(200)
        ]
        replay_rows.append(dict(assign_rows[0]))
        write_jsonl(assign, assign_rows)
        write_jsonl(replay, replay_rows)

        cfg = _minimal_cfg(
            tmp_path,
            stage_a=replay,
            stage_b=assign,
            replay_pool=replay,
            assignment_frac=0.9,
        )
        out_dir = tmp_path / 'subsamples'
        monkeypatch.setattr(mod, 'STAGE_B_OUT_DIR', out_dir)
        man = build_stage_b_replay_mix(config_path=cfg, verbose=False)

        # n_replay = round(90 * 0.1 / 0.9) = 10
        assert man['n_assignment'] == 90
        assert man['n_replay'] == 10
        assert man['n'] == 100
        assert man['curriculum'] == 'Bp'
        assert 0.89 <= man['assignment_frac_actual'] <= 0.91

        out = Path(man['output'])
        assert out.exists()
        mixed = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines() if line]
        roles = [r['mix_role'] for r in mixed]
        assert roles.count('assignment') == 90
        assert roles.count('replay') == 10
        # No assignment EN text should appear as replay (dup excluded)
        assign_en = {r['en_text'] for r in assign_rows}
        for r in mixed:
            if r['mix_role'] == 'replay':
                assert r['en_text'] not in assign_en

        man_path = out_dir / f'stage_b_{man["tag"]}_manifest.json'
        assert man_path.exists()

    def test_deterministic_seed(self, tmp_path: Path, monkeypatch):
        import src.training.subsample as mod

        assign = tmp_path / 'assign.jsonl'
        replay = tmp_path / 'replay.jsonl'
        write_jsonl(
            assign,
            [
                {
                    'en_text': f'A{i}',
                    'hi_text': f'अ{i}',
                    'source': 'preprocessed',
                }
                for i in range(18)
            ],
        )
        write_jsonl(
            replay,
            [
                {
                    'en_text': f'R{i}',
                    'hi_text': f'र{i}',
                    'source': 'milpac_Acts',
                }
                for i in range(100)
            ],
        )
        cfg = _minimal_cfg(
            tmp_path,
            stage_a=replay,
            stage_b=assign,
            replay_pool=replay,
        )
        out_dir = tmp_path / 'subsamples'
        monkeypatch.setattr(mod, 'STAGE_B_OUT_DIR', out_dir)
        m1 = build_stage_b_replay_mix(config_path=cfg, verbose=False)
        # Rebuild overwrites same path; compare replay set by re-running load
        rows1 = [
            json.loads(line)
            for line in Path(m1['output']).read_text(encoding='utf-8').splitlines()
            if line
        ]
        keys1 = sorted((r['en_text'], r['mix_role']) for r in rows1 if r['mix_role'] == 'replay')
        m2 = build_stage_b_replay_mix(config_path=cfg, verbose=False)
        rows2 = [
            json.loads(line)
            for line in Path(m2['output']).read_text(encoding='utf-8').splitlines()
            if line
        ]
        keys2 = sorted((r['en_text'], r['mix_role']) for r in rows2 if r['mix_role'] == 'replay')
        assert keys1 == keys2
        assert m1['n_replay'] == 2  # round(18 * 0.1/0.9) = 2

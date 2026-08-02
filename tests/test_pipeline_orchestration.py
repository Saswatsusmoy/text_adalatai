"""Smoke tests: pipeline modules import; run_pipeline registry matches packages."""

import importlib

import run_pipeline


class TestPipelineImports:
    modules = [
        'src.preprocessing.reextract_pdfs',
        'src.preprocessing.join_lines',
        'src.preprocessing.segment_sentences',
        'src.preprocessing.align_sentences',
        'src.preprocessing.output_format',
        'src.preprocessing.ingest_external_parallel',
        'src.preprocessing.split_external_eval',
        'src.tokenizer.train',
        'src.tokenizer.benchmark',
        'src.evaluation.eval_sets',
        'src.evaluation.metrics_mt',
        'src.training.subsample',
        'src.utils.jsonl',
    ]

    def test_all_modules_importable(self):
        for mod_name in self.modules:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f'Failed to import {mod_name}'

    def test_preprocessing_has_run(self):
        for mod_name in self.modules[:7]:
            mod = importlib.import_module(mod_name)
            assert hasattr(mod, 'run'), f'{mod_name} missing run()'


class TestRunPipelineRegistry:
    def test_all_registered_modules_exist(self):
        for name, step in run_pipeline.STEPS.items():
            mod = importlib.import_module(step['module'])
            assert mod is not None, f'step {name} module missing: {step["module"]}'

    def test_groups_expand_to_known_steps_only(self):
        for group, _members in run_pipeline.GROUPS.items():
            expanded = run_pipeline.expand_steps([group])
            assert expanded, f'empty expansion for {group}'
            for step in expanded:
                assert step in run_pipeline.STEPS, f'{group} yields unknown step {step}'

    def test_preprocess_group_order(self):
        assert run_pipeline.expand_steps(['preprocess']) == [
            'reextract',
            'join',
            'join_hi',
            'segment',
            'align',
            'output',
        ]

    def test_external_group(self):
        assert run_pipeline.expand_steps(['external']) == [
            'external_ingest',
            'external_eval_split',
            'eval_sets',
        ]
        assert run_pipeline.expand_steps(['external_full']) == [
            'external_download',
            'external_ingest',
            'external_eval_split',
            'eval_sets',
        ]
        expanded_all = run_pipeline.expand_steps(['all'])
        assert 'external_ingest' in expanded_all
        assert 'external_eval_split' in expanded_all
        assert 'eval_sets' in expanded_all

    def test_phase_groups_cover_train_eval_smoke(self):
        assert 'zero_shot_smoke' in run_pipeline.expand_steps(['eval_smoke'])
        assert 'train_nllb_smoke' in run_pipeline.expand_steps(['train_smoke'])

"""Smoke tests: each pipeline module can be imported and has a run() function."""

import importlib


class TestPipelineImports:
    modules = [
        "src.preprocessing.reextract_pdfs",
        "src.preprocessing.join_lines",
        "src.preprocessing.segment_sentences",
        "src.preprocessing.align_sentences",
        "src.preprocessing.output_format",
        "src.tokenizer.train",
        "src.tokenizer.benchmark",
        # "src.evaluation.metrics",    # Phase 4 (not started)
        # "src.training.train",         # Phase 3 (not started)
    ]

    def test_all_modules_importable(self):
        for mod_name in self.modules:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Failed to import {mod_name}"

    def test_preprocessing_has_run(self):
        for mod_name in self.modules[:5]:
            mod = importlib.import_module(mod_name)
            assert hasattr(mod, "run"), f"{mod_name} missing run()"

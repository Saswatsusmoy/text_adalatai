"""
End-to-end orchestration of the Adalat AI preprocessing pipeline.

Usage:
    python run_pipeline.py --steps all
    python run_pipeline.py --steps preprocess
    python run_pipeline.py --steps align,output
"""

import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
ROOT = Path(__file__).parent

STEPS = {
    "reextract": {
        "module": "src.preprocessing.reextract_pdfs",
        "args": ["--all"],
        "desc": "Re-extract Hindi PDFs with Tesseract OCR",
    },
    "reextract_compare": {
        "module": "src.preprocessing.reextract_pdfs",
        "args": ["--compare-all"],
        "desc": "Compare re-extracted vs originals",
    },
    "join": {
        "module": "src.preprocessing.join_lines",
        "args": [],
        "desc": "Join hard-wrapped English lines",
    },
    "segment": {
        "module": "src.preprocessing.segment_sentences",
        "args": [],
        "desc": "Segment sentences (EN + HI)",
    },
    "align": {
        "module": "src.preprocessing.align_sentences",
        "args": [],
        "desc": "LaBSE alignment + quality filters",
    },
    "output": {
        "module": "src.preprocessing.output_format",
        "args": [],
        "desc": "Train/dev/test splits + metadata",
    },
    "tokenizer_bench": {
        "module": "src.analysis.tokenizer_analysis",
        "args": [],
        "desc": "Benchmark all tokenizers",
    },
    "eval": {
        "module": "src.evaluation.metrics",
        "args": ["--jsonl", "data/aligned/all.jsonl"],
        "desc": "Evaluate aligned pairs",
    },
}

GROUPS = {
    "preprocess": ["reextract", "join", "segment", "align", "output"],
    "analysis": ["tokenizer_bench", "eval"],
    "all": ["preprocess", "analysis"],
}


def run_step(name: str):
    step = STEPS[name]
    cmd = [PYTHON, "-m", step["module"]] + step["args"]
    print(f"\n{'='*60}")
    print(f"Step: {name} -- {step['desc']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"FAILED: {name}")
        sys.exit(1)
    print(f"OK: {name}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Adalat AI pipeline steps")
    parser.add_argument("--steps", default="all", help="Comma-separated steps or group name")
    args = parser.parse_args()

    steps_to_run = []
    for s in args.steps.split(","):
        s = s.strip()
        if s in GROUPS:
            steps_to_run.extend(GROUPS[s])
        elif s in STEPS:
            steps_to_run.append(s)
        else:
            print(f"Unknown step/group: {s}")
            sys.exit(1)

    for step in steps_to_run:
        run_step(step)

    print(f"\nAll steps completed successfully.")


if __name__ == "__main__":
    main()

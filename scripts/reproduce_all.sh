#!/bin/bash
# Reproduce data + tokenizer path (assignment-friendly wall time).
# Full H200 train curricula stay on Makefile train-nllb-*-h200 targets.
#
# Usage:
#   bash scripts/reproduce_all.sh [--skip-downloads]
#
set -e
cd "$(dirname "$0")/.."

SKIP_DOWNLOADS=false
if [ "$1" = "--skip-downloads" ]; then
    SKIP_DOWNLOADS=true
fi

PYTHON="${PYTHON:-python3}"
export PYTHONPATH=.

echo "============================================"
echo "Adalat AI - Full Reproduction"
echo "============================================"
echo "Date: $(date)"
echo "Python: $($PYTHON --version)"
echo ""

echo "--- Phase 0: environment ---"
$PYTHON -c "
import sys
required = [
    'sentencepiece', 'tokenizers', 'spacy', 'sentence_transformers',
    'datasets', 'tiktoken', 'openpyxl', 'pandas', 'yaml', 'sacrebleu',
]
missing = []
for m in required:
    try:
        __import__(m)
    except ImportError:
        missing.append(m)
if missing:
    print('MISSING: ' + ', '.join(missing))
    print('Run: pip install -r requirements.txt')
    sys.exit(1)
print('All core dependencies OK')
"
$PYTHON -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model OK')" 2>/dev/null || \
    { echo "Downloading spaCy model..."; $PYTHON -m spacy download en_core_web_sm; }

echo ""
echo "--- Phase 1: assignment preprocessing ---"
$PYTHON -m src.preprocessing.reextract_pdfs --all 2>&1 | tail -3
$PYTHON -m src.preprocessing.reextract_pdfs --compare-all 2>&1 | tail -3
$PYTHON -m src.preprocessing.join_lines 2>&1 | tail -3
$PYTHON -m src.preprocessing.segment_sentences 2>&1 | tail -3
$PYTHON -m src.preprocessing.align_sentences 2>&1 | tail -3
$PYTHON -m src.preprocessing.output_format 2>&1 | tail -3
echo "Assignment preprocessing complete."

echo ""
echo "--- Phase 1b: external Stage A + dual-policy split ---"
if [ "$SKIP_DOWNLOADS" = false ]; then
    $PYTHON -m src.preprocessing.ingest_external_parallel --download 2>&1 | tail -20
else
    if [ ! -d data/external/raw/milpac ] && [ ! -d data/external/raw/anuvaad ]; then
        echo "WARNING: no data/external/raw/; Stage A ingest may be empty"
    fi
    $PYTHON -m src.preprocessing.ingest_external_parallel 2>&1 | tail -20
fi
if [ -f data/external/parallel/stage_a_en_hi.jsonl ]; then
    $PYTHON -m src.preprocessing.split_external_eval 2>&1 | tail -10
    $PYTHON -m src.evaluation.eval_sets 2>&1 | tail -15
else
    echo "SKIP dual-policy split (no stage_a_en_hi.jsonl)"
fi

echo ""
echo "--- Phase 2: tokenizer ---"
if [ "$SKIP_DOWNLOADS" = false ]; then
    $PYTHON -m src.tokenizer.prepare_corpus 2>&1 | tail -3
    for vs in 16000 32000 41000; do
        echo "Training SP $vs..."
        $PYTHON -m src.tokenizer.train \
            --input data/external/legal_hindi_corpus.txt \
            --vocab-size $vs 2>&1 | tail -1
    done
else
    echo "SKIP tokenizer training (--skip-downloads)"
fi
$PYTHON -m src.tokenizer.benchmark 2>&1 | tail -20
$PYTHON -m src.tokenizer.deep_dive 2>&1 | tail -5
$PYTHON -m src.preprocessing.discover_proper_nouns 2>&1 | tail -5

echo ""
echo "--- Phase 4 smoke (optional zero-shot; skip if no torch/HF) ---"
if $PYTHON -c "import torch, transformers" 2>/dev/null; then
    $PYTHON -m src.evaluation.zero_shot_nllb \
        --max-pairs 5 --suites I_test --no-resume 2>&1 | tail -15 || \
        echo "WARNING: zero-shot smoke failed (network / weights)"
else
    echo "SKIP zero-shot smoke (torch/transformers missing)"
fi

echo ""
echo "--- Tests + lint ---"
$PYTHON -m pytest tests/ -q -k "not scan_all" 2>&1 | tail -20
if $PYTHON -c "import ruff" 2>/dev/null; then
    $PYTHON -m ruff check src tests run_pipeline.py 2>&1 | tail -5
else
    echo "SKIP ruff (pip install -r requirements-dev.txt)"
fi

echo ""
echo "============================================"
echo "Reproduction complete."
echo "Interview tour: docs/WALKTHROUGH.md"
echo "Scores / freezes: REPORT.md, docs/EXPERIMENTS.md"
echo "Full H200 train: make train-nllb-A1-h200 (etc.)"
echo "============================================"

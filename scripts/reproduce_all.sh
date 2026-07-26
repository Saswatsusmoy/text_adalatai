#!/bin/bash
# ============================================================
# Adalat AI - Full Reproduction Script
# ============================================================
# Run this to reproduce the pipeline steps that exist today.
#
# Usage:
#   bash scripts/reproduce_all.sh [--skip-downloads]
#
# Steps:
#   1. Verify environment
#   2. Assignment preprocessing (PDF -> aligned JSONL -> splits)
#   3. External Stage A legal EN-HI ingest (MILPaC + Anuvaad)
#   4. Train custom tokenizers (unless --skip-downloads)
#   5. Run tokenizer benchmarks
#   6. Run tokenizer deep dive + proper-noun discovery
#   7. Run tests
# ============================================================

set -e
cd "$(dirname "$0")/.."

SKIP_DOWNLOADS=false
if [ "$1" = "--skip-downloads" ]; then
    SKIP_DOWNLOADS=true
fi

echo "============================================"
echo "Adalat AI - Full Reproduction"
echo "============================================"
echo "Date: $(date)"
echo "Python: $(python3 --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Step 0: Verify environment
echo "--- Step 0: Verify environment ---"
python3 -c "
import sys
required = [
    'sentencepiece', 'tokenizers', 'spacy', 'sentence_transformers',
    'datasets', 'tiktoken', 'openpyxl', 'pandas',
]
missing = []
for m in required:
    try:
        __import__(m)
    except ImportError:
        missing.append(m)
if missing:
    print(f'MISSING: {\", \".join(missing)}')
    print('Run: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('All dependencies OK')
"

# Check spaCy model
python3 -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy model OK')" 2>/dev/null || \
    { echo "Downloading spaCy model..."; python3 -m spacy download en_core_web_sm; }

echo ""

# Step 1: Assignment preprocessing pipeline
echo "--- Step 1: Assignment preprocessing pipeline ---"

echo "[1a] Re-extract Hindi PDFs (if source PDFs exist)"
PYTHONPATH=. python3 src/preprocessing/reextract_pdfs.py --all 2>&1 | tail -3
PYTHONPATH=. python3 src/preprocessing/reextract_pdfs.py --compare-all 2>&1 | tail -3

echo "[1b] Join English lines"
PYTHONPATH=. python3 src/preprocessing/join_lines.py 2>&1 | tail -3

echo "[1c] Sentence segmentation"
PYTHONPATH=. python3 src/preprocessing/segment_sentences.py 2>&1 | tail -3

echo "[1d] Alignment + quality filters"
PYTHONPATH=. python3 src/preprocessing/align_sentences.py 2>&1 | tail -3

echo "[1e] Output format"
PYTHONPATH=. python3 src/preprocessing/output_format.py 2>&1 | tail -3

echo "Assignment preprocessing complete."
echo ""

# Step 2: External Stage A legal parallel
echo "--- Step 2: External Stage A (MILPaC + Anuvaad) ---"

if [ "$SKIP_DOWNLOADS" = false ]; then
    echo "[2a] Download raw external corpora if missing, then ingest"
    PYTHONPATH=. python3 src/preprocessing/ingest_external_parallel.py --download 2>&1 | tail -20
else
    echo "[2a] Ingest only (--skip-downloads; uses data/external/raw if present)"
    if [ ! -d data/external/raw/milpac ] && [ ! -d data/external/raw/anuvaad ]; then
        echo "WARNING: no data/external/raw/; Stage A ingest will be empty"
    fi
    PYTHONPATH=. python3 src/preprocessing/ingest_external_parallel.py 2>&1 | tail -20
fi

echo ""

# Step 3: Train custom tokenizers
echo "--- Step 3: Train custom tokenizers ---"

if [ "$SKIP_DOWNLOADS" = false ]; then
    echo "[3a] Download mono legal HI corpus and train tokenizers"
    PYTHONPATH=. python3 src/tokenizer/prepare_corpus.py 2>&1 | tail -3

    for vs in 16000 32000 41000; do
        echo "Training SP $vs..."
        PYTHONPATH=. python3 src/tokenizer/train.py \
            --input data/external/legal_hindi_corpus.txt \
            --vocab-size $vs 2>&1 | tail -1
    done
else
    echo "[SKIP] Tokenizer training (--skip-downloads)"
    if [ ! -f data/models/tokenizers/sentencepiece_16000.model ]; then
        echo "WARNING: no SP models under data/models/tokenizers/; tokenizer-bench will be limited"
    fi
fi

echo ""

# Step 4: Tokenizer benchmarks
echo "--- Step 4: Tokenizer benchmarks ---"
PYTHONPATH=. python3 src/tokenizer/benchmark.py 2>&1
echo ""

# Step 5: Analysis helpers (no MT evaluation module yet)
echo "--- Step 5: Analysis helpers ---"
echo "[5a] Tokenizer deep dive"
PYTHONPATH=. python3 src/tokenizer/deep_dive.py 2>&1 | tail -5

echo "[5b] Proper noun discovery"
PYTHONPATH=. python3 src/preprocessing/discover_proper_nouns.py 2>&1 | tail -5

echo ""

# Step 6: Tests
echo "--- Step 6: Tests ---"
PYTHONPATH=. python3 -m pytest tests/ -v -k "not scan_all" 2>&1 | tail -20

echo ""
echo "============================================"
echo "Reproduction complete!"
echo "============================================"

.PHONY: install preprocess align tokenize test clean

# --- Setup ---

install:
	pip install -r requirements.txt
	python3 -m spacy download en_core_web_sm

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python3 -m spacy download en_core_web_sm

# --- Preprocessing pipeline ---

# Step 1: Re-extract corrupted Hindi PDFs
reextract:
	PYTHONPATH=. python3 src/preprocessing/reextract_pdfs.py --all
	PYTHONPATH=. python3 src/preprocessing/reextract_pdfs.py --compare-all

# Step 4: Join hard-wrapped English lines
join:
	PYTHONPATH=. python3 src/preprocessing/join_lines.py

# Step 7: Sentence segmentation
segment:
	PYTHONPATH=. python3 src/preprocessing/segment_sentences.py

# Step 8: Alignment + quality filters
align:
	PYTHONPATH=. python3 src/preprocessing/align_sentences.py

# Step 10: Output train/dev/test splits
output:
	PYTHONPATH=. python3 src/preprocessing/output_format.py

# Full preprocessing pipeline
preprocess: reextract join segment align output
	@echo "Preprocessing complete"

# --- Tokenizer ---

# Download and prepare Hindi legal corpus for tokenizer training
tokenizer-prepare:
	PYTHONPATH=. python3 src/tokenizer/prepare_corpus.py

# Train custom SentencePiece tokenizers at multiple vocab sizes
tokenizer-train-16k: tokenizer-prepare
	PYTHONPATH=. python3 src/tokenizer/train.py \
		--input data/external/legal_hindi_corpus.txt \
		--vocab-size 16000

tokenizer-train-32k:
	PYTHONPATH=. python3 src/tokenizer/train.py \
		--input data/external/legal_hindi_corpus.txt \
		--vocab-size 32000

tokenizer-train-41k:
	PYTHONPATH=. python3 src/tokenizer/train.py \
		--input data/external/legal_hindi_corpus.txt \
		--vocab-size 41000

# Train all tokenizer variants
tokenizer-train-all: tokenizer-train-16k tokenizer-train-32k tokenizer-train-41k

# Benchmark all tokenizers on the aligned corpus
tokenizer-bench:
	PYTHONPATH=. python3 src/tokenizer/benchmark.py

# --- Training (scaffold) ---

train:
	PYTHONPATH=. python3 src/training/train.py

# --- Evaluation ---

eval:
	PYTHONPATH=. python3 src/evaluation/metrics.py --jsonl data/aligned/all.jsonl

# --- Tests ---

test:
	PYTHONPATH=. python3 -m pytest tests/ -v -k "not scan_all"

test-all:
	PYTHONPATH=. python3 -m pytest tests/ -v

# --- Cleanup ---

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-data:
	rm -rf data/aligned data/processed data/english/segmented data/hindi/segmented

# --- Reproducibility ---

# Full end-to-end: install -> preprocess -> tokenizer -> test
all: install preprocess tokenizer-train tokenizer-bench test
	@echo "Full pipeline complete"

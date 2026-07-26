.PHONY: install venv reextract join segment align output preprocess \
	tokenizer-prepare tokenizer-train-16k tokenizer-train-32k tokenizer-train-41k \
	tokenizer-train-all tokenizer-train tokenizer-bench test test-all clean clean-data all

# --- Setup ---

install:
	pip install -r requirements.txt
	python3 -m spacy download en_core_web_sm

venv:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python3 -m spacy download en_core_web_sm

# --- Preprocessing pipeline ---

reextract:
	PYTHONPATH=. python3 src/preprocessing/reextract_pdfs.py --all
	PYTHONPATH=. python3 src/preprocessing/reextract_pdfs.py --compare-all

join:
	PYTHONPATH=. python3 src/preprocessing/join_lines.py

segment:
	PYTHONPATH=. python3 src/preprocessing/segment_sentences.py

align:
	PYTHONPATH=. python3 src/preprocessing/align_sentences.py

output:
	PYTHONPATH=. python3 src/preprocessing/output_format.py

preprocess: reextract join segment align output
	@echo "Preprocessing complete"

# --- Tokenizer ---

tokenizer-prepare:
	PYTHONPATH=. python3 src/tokenizer/prepare_corpus.py

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

tokenizer-train-all: tokenizer-train-16k tokenizer-train-32k tokenizer-train-41k

# Alias used by reproduce docs / older notes
tokenizer-train: tokenizer-train-all

tokenizer-bench:
	PYTHONPATH=. python3 src/tokenizer/benchmark.py

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

# install -> preprocess -> custom SP train -> bench -> test
# Tokenizer train needs network for external corpus unless already under data/external/
all: install preprocess tokenizer-train-all tokenizer-bench test
	@echo "Full pipeline complete"

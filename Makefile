.PHONY: install venv reextract join segment align output preprocess \
	external-download external-ingest \
	tokenizer-prepare tokenizer-train-16k tokenizer-train-32k tokenizer-train-41k \
	tokenizer-train-all tokenizer-train tokenizer-bench \
	tokenizer-spm-v2-corpus tokenizer-spm-v2-train tokenizer-spm-v2-bench tokenizer-c0 \
	test test-all clean clean-data all

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

# --- External legal parallel (Stage A) ---

external-download:
	PYTHONPATH=. python3 src/preprocessing/ingest_external_parallel.py --download

external-ingest:
	PYTHONPATH=. python3 src/preprocessing/ingest_external_parallel.py

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
	PYTHONPATH=. python3 src/tokenizer/benchmark.py --eval held_out

# Track C0: legal SPM v2 (Stage A + assignment train; never dev/test)
tokenizer-spm-v2-corpus:
	PYTHONPATH=. python3 src/tokenizer/prepare_spm_corpus.py --mode joint
	PYTHONPATH=. python3 src/tokenizer/prepare_spm_corpus.py --mode hi

tokenizer-spm-v2-train: tokenizer-spm-v2-corpus
	PYTHONPATH=. python3 src/tokenizer/train_v2.py

tokenizer-spm-v2-bench:
	PYTHONPATH=. python3 src/tokenizer/benchmark.py --eval held_out

tokenizer-c0: tokenizer-spm-v2-train tokenizer-spm-v2-bench
	@echo "Track C0 complete (SPM v2 train + held-out bench)"

# Full-as-possible joint Unigram (dedupe + memory profiles; keeps sample joint_41k)
tokenizer-spm-v2-full-joint:
	PYTHONPATH=. python3 src/tokenizer/train_full_joint.py --vocab-size 41000 --max-chars 4096
	PYTHONPATH=. python3 src/tokenizer/benchmark.py --eval held_out

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

.PHONY: install install-dev venv reextract verify-ocr join segment align output preprocess \
	external-download external-ingest external-eval-split \
	tokenizer-prepare tokenizer-train-16k tokenizer-train-32k tokenizer-train-41k \
	tokenizer-train-all tokenizer-train tokenizer-bench \
	tokenizer-spm-v2-corpus tokenizer-spm-v2-train tokenizer-spm-v2-bench tokenizer-c0 \
	tokenizer-spm-v2-full-joint tokenizer-spm-v2-full-joint-bpe \
	tokenizer-matrix-phase1 tokenizer-matrix-phase2 tokenizer-matrix-bench profile-hardware \
	zero-shot-nllb zero-shot-nllb-smoke \
	eval-mbr-smoke eval-mbr-a2 eval-mbr-zs \
	comet-score \
	stage-a-subsample-smoke stage-a-subsample-A1 stage-b-replay-mix \
	train-nllb-smoke train-nllb-A1 train-nllb-A1-h200 train-nllb-Bp-h200 \
	train-nllb-A2-dora-h200 \
	train-c1-smoke train-c1-A1-h200 eval-c1 \
	c1c-vocab-extend train-c1c-A1-h200 \
	lint format format-check check \
	test test-all clean clean-data all

# --- Setup ---

install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

install-dev:
	pip install -r requirements-dev.txt
	python -m spacy download en_core_web_sm

venv:
	# Prefer 3.12 for spaCy wheels when available
	( command -v python3.12 >/dev/null && python3.12 -m venv .venv ) || python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m spacy download en_core_web_sm

# --- Phase 1: preprocessing (assignment corpus) ---

reextract:
	PYTHONPATH=. python3 src/preprocessing/reextract_pdfs.py --all
	PYTHONPATH=. python3 src/preprocessing/reextract_pdfs.py --compare-all

# Gate: preprocessed Hindi must be Tesseract OCR (fails on text-layer regression)
verify-ocr:
	PYTHONPATH=. python3 src/preprocessing/reextract_pdfs.py --verify-ocr

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

# --- Phase 1b: external Stage A + dual-policy split ---

external-download:
	PYTHONPATH=. python3 src/preprocessing/ingest_external_parallel.py --download

external-ingest:
	PYTHONPATH=. python3 src/preprocessing/ingest_external_parallel.py

# Policy E held-out + stage_a_train (after external-ingest)
external-eval-split:
	PYTHONPATH=. python3 src/preprocessing/split_external_eval.py
	PYTHONPATH=. python3 -m src.evaluation.eval_sets

# --- Phase 2: tokenizer ---

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

# BPE ablation (DESIGN §32): same v2 joint corpus, model_type=bpe, 41k
tokenizer-spm-v2-full-joint-bpe:
	PYTHONPATH=. python3 src/tokenizer/train_full_joint.py --vocab-size 41000 --max-chars 4096 --model-type bpe
	PYTHONPATH=. python3 src/tokenizer/benchmark.py --eval held_out

# --- Tokenizer matrix (DESIGN §33): {unigram,bpe} x {16,32,41,48,64}k x {joint,hi} + secondary axes ---

MATRIX_PARALLEL ?= 6

tokenizer-matrix-phase1:
	PYTHONPATH=. python3 -m src.tokenizer.train_matrix --phase 1 --parallel $(MATRIX_PARALLEL)

tokenizer-matrix-phase2:
	PYTHONPATH=. python3 -m src.tokenizer.train_matrix --phase 2 --parallel $(MATRIX_PARALLEL)

tokenizer-matrix-bench:
	PYTHONPATH=. python3 -m src.tokenizer.bench_matrix

# --- Hardware ---

profile-hardware:
	PYTHONPATH=. python3 src/utils/profile_hardware.py

# --- Phase 4: evaluation ---

zero-shot-nllb:
	PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb

zero-shot-nllb-smoke:
	PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb --max-pairs 5 --suites I_test,E_milpac_test

# --- MBR decode (sample N + argmax pairwise chrF++) ---

# Wiring smoke: 2 pairs, 4 samples, MPS/local. No adapters required.
MBR_SAMPLES ?= 8
MBR_TEMPERATURE ?= 1.0
MBR_TOP_P ?= 0.9
MBR_UTILITY ?= chrfpp
A2_ADAPTERS ?= data/runs/nllb600_A_A2_h200_A2_ddp2_20260726T212958Z/checkpoints/best_primary

eval-mbr-smoke:
	PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb \
		--max-pairs 2 --suites I_dev --max-new-tokens 64 \
		--mbr --mbr-samples 4 --mbr-utility $(MBR_UTILITY) \
		--tag zs_mbr_smoke

eval-mbr-zs:
	PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb \
		--mbr --mbr-samples $(MBR_SAMPLES) \
		--mbr-temperature $(MBR_TEMPERATURE) --mbr-top-p $(MBR_TOP_P) \
		--mbr-utility $(MBR_UTILITY)

eval-mbr-a2:
	PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb \
		--adapters $(A2_ADAPTERS) \
		--mbr --mbr-samples $(MBR_SAMPLES) \
		--mbr-temperature $(MBR_TEMPERATURE) --mbr-top-p $(MBR_TOP_P) \
		--mbr-utility $(MBR_UTILITY)

# --- COMET-22 scoring over all data/analysis/*_hyps.jsonl (cache-safe / resumable) ---

COMET_MODEL ?= Unbabel/wmt22-comet-da
COMET_BATCH ?= 32
COMET_GPUS ?= 1

comet-score:
	PYTHONPATH=. python3 -m src.evaluation.comet_score \
		--model $(COMET_MODEL) --batch-size $(COMET_BATCH) --gpus $(COMET_GPUS)

# --- Phase 3: training (Track D default; Track C via train-c1* / c1c*) ---

stage-a-subsample-smoke:
	PYTHONPATH=. python3 -m src.training.subsample --curriculum smoke

stage-a-subsample-A1:
	PYTHONPATH=. python3 -m src.training.subsample --curriculum A1

train-nllb-smoke:
	PYTHONPATH=. python3 -m src.training.train_nllb_lora --curriculum smoke --max-steps 30 --skip-gen-eval

train-nllb-A1:
	PYTHONPATH=. python3 -m src.training.train_nllb_lora --curriculum A1

# Hopper 2x H200 DDP (bf16 / Flash SDPA / fixed pad / global_batch 32 parity)
train-nllb-A1-h200:
	PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
		-m src.training.train_nllb_lora \
		--config configs/training_h200.yaml --curriculum A1 --device cuda

# Stage B' 90/10 assignment + A2 replay (anti-forget); resume A2 best
stage-b-replay-mix:
	PYTHONPATH=. python3 -m src.training.subsample \
		--curriculum Bp --config configs/training_h200_Bp.yaml

train-nllb-Bp-h200: stage-b-replay-mix
	PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
		-m src.training.train_nllb_lora \
		--config configs/training_h200_Bp.yaml --stage B --curriculum Bp --device cuda

# DoRA ablation: A2 data, decoder_attn, from base (not resume LoRA)
train-nllb-A2-dora-h200:
	PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
		-m src.training.train_nllb_lora \
		--config configs/training_h200_A2_dora.yaml --stage A --curriculum A2 --device cuda

# Track C1: Marian + SPM_V2_PRIMARY
train-c1-smoke:
	PYTHONPATH=. python3 -m src.training.train_legal_mt \
		--config configs/training_c1.yaml --curriculum smoke --max-steps 50 --skip-gen-eval

train-c1-A1-h200:
	PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
		-m src.training.train_legal_mt \
		--config configs/training_c1_h200.yaml --curriculum A1 --device cuda

eval-c1:
	PYTHONPATH=. python3 -m src.evaluation.eval_legal_mt \
		--checkpoint $(CKPT) --device cuda --batch-size 32 --tag $(TAG)

# Track C1c: NLLB vocab-extend from legal SPM + LoRA
c1c-vocab-extend:
	PYTHONPATH=. python3 -m src.training.vocab_extend_nllb \
		--top-k 8000 --out data/models/nllb600_c1c_sp_ext

train-c1c-A1-h200:
	PYTHONPATH=. torchrun --standalone --nproc_per_node=2 \
		-m src.training.train_nllb_lora \
		--config configs/training_c1c_h200.yaml --curriculum A1 --device cuda

# --- Lint / format (ruff; see pyproject.toml) ---

PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
RUFF = $(PYTHON) -m ruff
PYTEST = $(PYTHON) -m pytest

lint:
	$(RUFF) check src tests run_pipeline.py

format:
	$(RUFF) format src tests run_pipeline.py
	$(RUFF) check --fix src tests run_pipeline.py

format-check:
	$(RUFF) format --check src tests run_pipeline.py
	$(RUFF) check src tests run_pipeline.py

check: format-check test

# --- Tests ---

test:
	PYTHONPATH=. $(PYTEST) tests/ -v -k "not scan_all"

test-all:
	PYTHONPATH=. $(PYTEST) tests/ -v

# --- Cleanup ---

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-data:
	rm -rf data/aligned data/processed data/english/segmented data/hindi/segmented

# --- Reproducibility ---

# install -> preprocess -> custom SP train -> bench -> test
# Tokenizer train needs network for external corpus unless already under data/external/
all: install preprocess tokenizer-train-all tokenizer-bench test
	@echo "Full pipeline complete"

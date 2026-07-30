.PHONY: install venv reextract join segment align output preprocess \
	external-download external-ingest external-eval-split \
	tokenizer-prepare tokenizer-train-16k tokenizer-train-32k tokenizer-train-41k \
	tokenizer-train-all tokenizer-train tokenizer-bench \
	tokenizer-spm-v2-corpus tokenizer-spm-v2-train tokenizer-spm-v2-bench tokenizer-c0 \
	tokenizer-spm-v2-full-joint profile-hardware \
	zero-shot-nllb zero-shot-nllb-smoke \
	stage-a-subsample-smoke stage-a-subsample-A1 stage-b-replay-mix \
	train-nllb-smoke train-nllb-A1 train-nllb-A1-h200 train-nllb-Bp-h200 \
	train-nllb-A2-dora-h200 \
	train-c1-smoke train-c1-A1-h200 eval-c1 \
	c1c-vocab-extend train-c1c-A1-h200 \
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

# Policy E held-out + stage_a_train (after external-ingest)
external-eval-split:
	PYTHONPATH=. python3 src/preprocessing/split_external_eval.py
	PYTHONPATH=. python3 -m src.evaluation.eval_sets

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

# --- Local hardware / MLX ---

profile-hardware:
	PYTHONPATH=. python3 src/utils/profile_hardware.py

# Track D zero-shot NLLB (MPS/CPU) on Policy I + E
zero-shot-nllb:
	PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb

zero-shot-nllb-smoke:
	PYTHONPATH=. python3 -m src.evaluation.zero_shot_nllb --max-pairs 5 --suites I_test,E_milpac_test

# Stage A NLLB LoRA (MPS)
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

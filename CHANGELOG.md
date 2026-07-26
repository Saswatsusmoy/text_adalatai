# Changelog

## [Unreleased]

### Added

- **`docs/EXPERIMENTS.md`:** Consolidated research log (assignment pipeline, Stage A data, cross-model tokenizer survey, SPM v1/v2, joint vs HI-only, full-joint 16GB path, vocab ablation 41/48/64k, Track C freeze joint_full_41000, dual-track plan, artifact index, reproduce commands). README and DESIGN_DECISIONS link to it.

- **Joint full vocab ablation 48k + 64k:** Trained `sentencepiece_legal_v2_joint_full_{48000,64000}` on same deduped joint corpus (profile=full). Held-out/test/all benches vs 41k in `data/analysis/tokenizer_vocab_size_ablation.json`. **Track C production freeze: `sentencepiece_legal_v2_joint_full_41000`** (generalization / emb size over max packing; 64k ablation only).

- **Full-joint SPM on 16GB (dedupe path):** `dedupe_text_file` + `train_full_joint.py` tries Unigram profiles `full` -> `full_tight` -> `full_sample_15` in a child process (OOM-safe). Dedupe joint corpus (~2% exact dups) + max 4096 chars enabled full Unigram train on **all remaining lines** (`input_sentence_size=0`, seed 250k). Winner: `sentencepiece_legal_v2_joint_full_41000` (does not overwrite sample `joint_41000`). Makefile: `tokenizer-spm-v2-full-joint`. No byte-level BPE.

- **Track C0 legal SentencePiece v2:** `prepare_spm_corpus.py` builds SPM train text from Stage A + assignment train only (hard-excludes dev/test docs 8,9,24 and 1,4,21). Corpora: joint ~1.99M lines / 291M chars; hi ~994k lines / 139M chars. `train_v2.py` trains `sentencepiece_legal_v2_{hi,joint}_{32k,41k}` without overwriting v1. Joint train samples 1M sentences (RAM). Held-out bench (322 pairs): **joint 41k wins for MT** (HI c/t 4.34, HI/EN 0.724, total 11,004 vs v1 41k 3.95 / 0.739 / 11,965). HI-only packs HI better but fragments EN. `benchmark.py --eval held_out` -> `data/analysis/tokenizer_metrics_v2.json`. Makefile: `tokenizer-c0`. Tests: `test_prepare_spm_corpus.py`.

- **External legal EN-HI ingest (Gate 9 T0)** (`src/preprocessing/ingest_external_parallel.py`): Downloads/processes MILPaC (Law-AI) and Anuvaad legal EN-HI (judiciary, HC/SUVAS, law commission, names dict, augmented, legal terms). Already-aligned pairs are mapped to project JSONL (`en_text`, `hi_text`, `source`, `doc_id`) and filtered with the same char-length ratio (0.3-3.0) and min-length rules as post-alignment QC; exact pair dedup. Outputs under `data/external/parallel/` including `stage_a_en_hi.jsonl` + `ingest_report.json`. Makefile: `make external-download`, `make external-ingest`. Tests: `tests/preprocessing/test_ingest_external_parallel.py`.

### Changed

- **External Stage A wired through docs/orchestrators:** `configs/preprocessing.yaml` documents `external_ingest` + paths/licenses/filters. `run_pipeline.py` steps/groups: `external_download`, `external_ingest`, groups `external` / `external_full`; `all` includes `external_ingest`. `scripts/reproduce_all.sh` runs Stage A ingest (with `--download` unless `--skip-downloads`). README quick-start and license notes updated.

### Fixed

- **Docs/orchestrator drift:** Updated README and AGENTS.md to match finished preprocess + tokenizer phases. Rewrote `configs/preprocessing.yaml` for live steps, skipped steps, and alignment thresholds (min sim 0.5, char ratio 0.3-3.0). Fixed `run_pipeline.py` module paths (`src.tokenizer.benchmark`), removed calls to missing `src.evaluation` / `src.training`, and flattened group expansion. Makefile: `all` now uses `tokenizer-train-all`, dropped broken `train`/`eval` targets, alias `tokenizer-train`. `scripts/reproduce_all.sh` no longer calls missing metrics module. DESIGN_DECISIONS renumbered (1-16), tokenizer file names corrected (`benchmark.py` not `analysis.py` / `reproduce_benchmarks.py`), staging path documented as `preprocessed/`. `.gitignore` adds `data/models/`.

### Added

- **Project scaffolding**: Created `src/`, `tests/`, `configs/` directory structure with Python package init files.
- **Configuration module** (`src/config.py`): Centralized all paths (data dirs, PDF tool path), doc ID lists, and Unicode range constants.
- **Validation utilities** (`src/utils/validation.py`): Devanagari character counting, ratio computation, and Hindi-likelihood heuristics.
- **PDF re-extraction script** (`src/preprocessing/reextract_pdfs.py`): Re-extracts Hindi text from corrupted PDFs using `pdftotext`, validates Devanagari content, compares old vs new, scans all 30 PDFs for quality issues, and can apply fixes to `clean/`.
- **Test suite** (`tests/preprocessing/test_reextract_pdfs.py`): 16 tests covering extraction, validation, comparison, apply, and full PDF scan.
- **Pipeline configuration** (`configs/preprocessing.yaml`): Declarative step listing with I/O paths and validation thresholds.
- **CHANGELOG.md**: This file  --  records all changes to the project.

### Changed

- **Extraction backend switched from `pdftotext` to Tesseract OCR**: After evaluating 5 alternatives (pdftotext, PyMuPDF, pdfminer.six, pdfplumber, Tesseract), the script now uses `tesseract -l hin --psm 6` by default. The `--backend pdftotext` flag is retained as a fallback. See DESIGN_DECISIONS.md for the full evaluation.

### Added

- **Intelligent line joining** (`src/preprocessing/join_lines.py`): Joins hard-wrapped lines in 17 English docs using heuristic + legal proper noun list. Reduces non-blank lines from 2,819 to 1,176 (58% reduction). Zero false positives. Output in `data/english/preprocessed/`.
- **Test suite** (`tests/preprocessing/test_join_lines.py`): 24 tests covering join logic, edge cases, document processing, and regression.
- **Sentence segmentation** (`src/preprocessing/segment_sentences.py`): Two-tool approach -- spaCy `en_core_web_sm` model (dependency parser) for English with pre-tokenization protection for Hindi honorifics (`Smt.`, `Shri.`), and danda (।) split for Hindi. Produces 2,495 English and 7,427 Hindi sentences across 30 docs. Auto-detects language. New dependency: `python3 -m spacy download en_core_web_sm`.
- **Test suite** (`tests/preprocessing/test_segment_sentences.py`): 22 tests covering English abbreviation handling, Hindi danda split, auto-detection, and full run.
- **Sentence alignment + quality filters** (`src/preprocessing/align_sentences.py`): LaBSE-based bilingual alignment using greedy bidirectional matching. Produces 1,458 EN-HI sentence pairs across 30 docs (avg 49/doc). Applies length ratio (0.3-3.0), similarity (>0.5), and near-dedup filters. New dependency: `sentence-transformers/LaBSE` (~1.8GB model). Output in `data/aligned/all.jsonl`. BGE-M3 (2024 SoTA) was also evaluated end-to-end but produces fewer pairs (1,347 vs 1,458). LaBSE retained for more training data.
- **Test suite** (`tests/preprocessing/test_align_sentences.py`): 18 tests covering loading, filtering, dedup, output format, and quality checks.

- **Final output format** (`src/preprocessing/output_format.py`): Splits 1,458 aligned pairs into train (1,136), dev (132), and test (190) at document level. Generates `metadata.json` and `alignment_report.json`. Output in `data/processed/`.
- **Proper noun discovery script** (`src/preprocessing/discover_proper_nouns.py`): Data-driven discovery of legal proper nouns for line joining. Scans all 30 English clean files for words appearing at continuation line starts. Derives 34 verified proper nouns with zero guessing.
- **Tokenizer analysis framework** (`src/tokenizer/benchmark.py`, `deep_dive.py`): Full corpus benchmark of 17 tokenizers across 14 model families (2024-2026): Custom SP 41K, Gemma 4, GPT-4o, Phi-4-mini, NLLB-200, Mistral Small 4, Qwen3/3.5/3.6, MiniMax M3, DeepSeek V3/V4 Pro, GLM 5.2, Phi-4, OLMo 3. Measures chars/token, HI/EN ratio, byte fallback detection. Finds that SentencePiece and multilingual BPE handle Hindi well, while byte-level BPE (all Llama-family models) cost 1.1-2.7x more tokens for Hindi regardless of vocabulary size. See DESIGN_DECISIONS.md for full comparison.
- **Custom SentencePiece tokenizer** (`data/models/tokenizers/`): Trained 3 SentencePiece models (16K/32K/41K vocab) on 14M characters of Indian legal Hindi text from Prarabdha/indian-legal-supervised-fine-tuning-data (`src/tokenizer/prepare_corpus.py` outputs to `data/external/legal_hindi_corpus.txt`, gitignored). The 41K model achieves 16,840 Devanagari tokens, 3.84 HI chars/tok, and 0.743 HI/EN ratio -- beating Gemma 4 on Hindi efficiency despite 6x smaller vocabulary. Training fully reproducible via `make tokenizer-train-all`.
- **Tokenizer benchmarks** (`src/tokenizer/benchmark.py`): Benchmarks accessible tokenizers and saves results to `data/analysis/tokenizer_metrics.json` (`--full` for the full model set).
- **Reproducible pipeline**: `Makefile` with targets (`make preprocess`, `make tokenizer-train-all`, `make tokenizer-bench`, `make test`). `run_pipeline.py` Python orchestrator. `scripts/reproduce_all.sh` bash reproduction script. `requirements.txt` with pinned dependencies.

### Fixed

- **Re-extracted corrupted Hindi PDFs (Docs 6, 14, 22, 25, 26)**: The 5 corrupted Hindi clean files contained 0 Devanagari characters  --  every glyph replaced with `?`. Re-extracted via Tesseract OCR. Recovered **42,536 Devanagari characters** across all 5 documents. Also re-extracted all 25 non-corrupted PDFs for consistency, producing full document text (including headers/parties) vs the original clean files which only had numbered paragraphs. Output in `data/hindi/preprocessed/` (renamed from `re_extracted/`). Applied to `clean/` on 2025-07-25.

### Discovered

- **Doc 17 PDF text layer**: `pdftotext` produces transliterated output (`Ekkuuh;` instead of `माननीय`). Tesseract OCR handles it correctly  --  not an issue.
- **PDF vs clean file mismatch**: The original clean files contain only numbered body paragraphs, while PDFs contain full judgments (headers, citations, parties, body). The 25 non-corrupted clean files are a subset of the full document. Our Tesseract extraction captures the complete document.

### Project structure

- **Phase-based organization**: `src/preprocessing/` (Phase 1), `src/tokenizer/` (Phase 2) -- only completed phases exist. No future-phase scaffolding.
- **Everything scripted, nothing interactive**: All analyses moved from ad-hoc commands to reproducible scripts with `if __name__ == "__main__"` entry points.
- **108 tests**: Covering preprocessing, tokenizer, and pipeline orchestration. Each phase has its own test directory.

### Skipped (not needed for this corpus)

- **Strip UTF-8 BOM (original plan Step 2)**: 25/30 Hindi clean files have BOM, but the working directory `data/hindi/preprocessed/` has 0 BOM files. Pipeline operates on preprocessed/ so this step is unnecessary. See DESIGN_DECISIONS.md for evidence.
- **Fix OCR Roman numerals (original plan Step 5)**: Zero instances of `li.`/`lili.` OCR artifacts found in any English clean file or raw PDF extraction. All `L.` instances are legitimate legal abbreviations. See DESIGN_DECISIONS.md for evidence.
- **Normalize line endings (original plan Step 3)**: `data/hindi/preprocessed/` is already 100% LF (30/30 files). The 25 CRLF files are legacy `clean/` files that don't reach the pipeline. See DESIGN_DECISIONS.md for evidence.
- **Paragraph segmentation (original plan Step 6)**: Already satisfied by Steps 1 and 4. Both English joined (782 paras) and Hindi OCR (951 paras) already have clear paragraph structure via blank lines. No additional processing needed. See DESIGN_DECISIONS.md for evidence.

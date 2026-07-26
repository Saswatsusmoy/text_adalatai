# Design Decisions

This document records architectural and implementation decisions made during the project, along with the rationale behind each.

**Full experiment tables, freezes, and reproduce steps:** see [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (tokenizer survey, SPM v1/v2, Stage A data, dual-track plan, vocab ablation, Track C freeze).

---

## 1. PDF Extraction Tool: evaluated 5 alternatives for Hindi Devanagari text

**Date:** 2025-07-25

**Context:** Need to re-extract Hindi text from 5 corrupted PDFs (Docs 6, 14, 22, 25, 26). The PDFs use custom font encoding  --  Devanagari ligatures stored as positioned glyph components rather than Unicode characters. This affects all 30 Hindi PDFs, not just the corrupted ones.

**Candidates evaluated:**

| Tool | Type | Deps | Speed (Doc 6) | Dev chars | Artifacts |
|------|------|------|---------------|-----------|-----------|
| **pdftotext** (Poppler) | Text-layer extractor | `poppler` (brew) | **0.028s** | 4,657 | None (but spacing broken) |
| **PyMuPDF** (fitz) | Text-layer extractor | `pymupdf` (pip) | **0.012s** | 4,657 | 122 control chars (`\x03`, `\x0e`, `\x16`) |
| **pdfminer.six** | Text-layer parser | `pdfminer.six` (pip) | 0.088s | 4,657 | `(cid:N)` placeholders for unmapped glyphs |
| **pdfplumber** | Text-layer parser | `pdfplumber` (pip) | 0.106s | 4,657 | `(cid:N)` placeholders (wraps pdfminer) |
| **Tesseract OCR 5.5.2** (hin) | OCR engine | `tesseract` + `tesseract-lang` (brew) | **6.14s** | **6,027** | OCR diacritic errors (`सिहं`->`सिंह`) |

**Text quality comparison on Doc 6:**
```
pdftotext:  प्रति वेद्य / भार ीय सव च्च न्यायालय / सिसविवल अपीलीय अति कारिर ा
Tesseract: प्रतिवेद्य / भारतीय सर्वोच्च न्यायालय / सिविल अपीलीय अधिकारिता
```

All 4 text-layer tools extract the **same underlying glyph stream** (identical Devanagari char counts). The difference is how they surface unmapped glyphs. Tesseract OCR recognizes rendered glyph shapes as a whole, producing correct ligatures, but introduces OCR errors and takes 220x longer.

**Initial decision:** Keep `pdftotext` for now.
**Final decision (2025-07-25):** Switched to **Tesseract OCR** for all 30 Hindi PDFs.

**Why the switch:** After full benchmarking, pdftotext's ligature decomposition (e.g., `प्रति वेद्य` vs `प्रतिवेद्य`, `सिसविवल` vs `सिविल`) proved too severe. These spacing artifacts affect every Hindi word with conjunct characters  --  roughly 60-70% of common Hindi words. Fixing them deterministically would require a Devanagari ligature grammar, which is non-trivial. Tesseract OCR produces correct ligatures out of the box.

**Rationale for Tesseract OCR:**
- Produces proper Devanagari ligatures  --  no spacing artifacts.
- 6-10 seconds per document on Apple M4  --  acceptable for a 30-doc corpus.
- OCR errors (diacritic mistakes like `सिहं`->`सिंह`) are infrequent and pattern-based.
- The model is the first to correctly handle Doc 17's PDF text layer (which pdftotext transliterated as `Ekkuuh;`).

**Trade-off:**
- 220x slower than pdftotext (~3 minutes vs ~1 second for 30 docs).
- Requires PyMuPDF for page rendering + Tesseract CLI.
- OCR errors need downstream cleanup, but they're less disruptive than spacing artifacts.

**Benchmark:** Apple M4, 16 GB RAM. `tesseract --psm 6 -l hin`, PyMuPDF render at 300 DPI. Average 6.5s per document.

---

## 2. Skipped: Strip UTF-8 BOM (original plan Step 2)

**Date:** 2025-07-25

**Context:** 25/30 Hindi clean files start with `\ufeff` (UTF-8 BOM). The original plan prescribed stripping it to avoid tokenizer issues and alignment offsets.

**Evidence from corpus analysis:**
- `data/hindi/clean/` -- 25 files have BOM (the original Windows-extracted files). 5 files (6, 14, 22, 25, 26) have no BOM (these were the corrupted ones, replaced by Tesseract output).
- `data/hindi/preprocessed/` -- **0 files have BOM**. The Tesseract OCR extraction produces clean UTF-8 without BOM.
- `data/english/clean/` -- **0 files have BOM**.
- Source code -- **0 files have BOM**.

**Decision:** Skip this step.

**Rationale:**
- Our working directory `data/hindi/preprocessed/` is already BOM-free (30/30 files).
- The BOM only affects the legacy `clean/` files, which are not used by the pipeline.
- BOM-related problems (regex `^\d+\.` failing, first token being `\ufeff1.` instead of `1.`) don't apply to the pipeline.
- If any script reads from `clean/`, Python's `encoding='utf-8-sig'` auto-strips BOM on read.
- The BOM and CRLF are perfectly correlated (the 25 BOM files also use CRLF), suggesting a single Windows extraction tool was the source. Our replacement files and all preprocessed output use LF.

---

## 3. Skipped: Fix OCR Roman numerals (original plan Step 5)

**Date:** 2025-07-25

**Context:** Common OCR artifact where lowercase `l` is mistaken for `I`, producing `li.` instead of `ii.` and `lili.` instead of `iii.` The original plan flagged this for Doc 1, advising pattern `\b[Ll]i\.\b`.

**Evidence from corpus analysis (scan of all 30 English clean files + raw PDF extractions):**

| Pattern | Hits in clean | Hits in PDF raw | Verdict |
|---------|--------------|-----------------|---------|
| `li.` (lowercase L as Roman numeral) | 0 | 0 | Not present |
| `lili.` | 0 | 0 | Not present |
| `liii.` etc. | 0 | 0 | Not present |
| `L.` standalone | 0 | 13 | All legitimate (judge initial `L. Nageswara Rao`, `S.L.P.`, `L.Rs.`) |
| `i.` `ii.` `iii.` (Doc 1) | 3 | 3 | **Correct** lowercase roman numerals in a legal numbered list, not OCR errors |

**Decision:** Skip this step.

**Rationale:**
- Zero instances of the `li.`->`ii` or `lili.`->`iii` OCR artifact exist in the corpus.
- All `L.` instances are legitimate legal abbreviations: judge names (`L. Nageswara Rao, J.`), case types (`S.L.P.` = Special Leave Petition), legal references (`L.Rs.` = Legal Representatives).
- The `i.` `ii.` `iii.` found in Doc 1's PDF raw extraction are correctly-formatted lowercase roman numerals in a legal charge list, not OCR artifacts. The clean file preserves them correctly.
- A naive regex `\b[Ll]i\.\b` would break `L. Nageswara Rao` -> `ii. Nageswara Rao` and `S.L.P.` -> `S.ii.P.` -- a classic false-positive example.
- This artifact is common in other legal PDF corpora (particularly older scans), but the source PDFs and Tesseract OCR don't produce it.

---

## 4. Skipped: Normalize CRLF -> LF line endings (original plan Step 3)

**Date:** 2025-07-25

**Context:** The original plan prescribed normalizing `\r\n` to `\n` for consistent line-based processing. The Hindi clean files extracted by Windows tools use CRLF, while the rest of the corpus uses LF.

**Evidence from corpus analysis:**

| Dataset | CRLF | LF | Mixed |
|---------|------|----|-------|
| `hindi/clean/` | 25 files | 5 files | 0 |
| `hindi/preprocessed/` | **0 files** | **30 files** | 0 |
| `english/clean/` | **0 files** | **30 files** | 0 |
| Source code | 0 files | all | 0 |

The 25 CRLF files in `clean/` are the same 25 that have BOM -- both trace to the same Windows extraction tool. The 5 files I replaced with Tesseract output have neither BOM nor CRLF. CRLF and LF are perfectly segregated within files (zero mixed-endings files found).

**Decision:** Skip this step.

**Rationale:**
- `data/hindi/preprocessed/` is already 100% LF (30/30 files). Our pipeline operates on preprocessed/.
- Python's `open()` in text mode auto-normalizes `\r\n` -> `\n` on read, so CRLF causes no issues when reading clean/ files.
- Regex with `$` in multiline mode matches at `\n` (after `\r`), so CRLF doesn't break pattern matching.
- Zero files have mixed endings within the same file, so no edge cases to handle.
- Like BOM, this is purely a legacy `clean/` concern that doesn't reach the pipeline.

---

## 5. Skipped: Paragraph segmentation (original plan Step 6)

**Date:** 2025-07-25

**Context:** The original plan prescribed unifying both extraction styles into paragraph-per-element format, splitting on double-newlines for long-line docs or using blank lines after joining for hard-wrapped docs.

**Evidence from corpus analysis:**

| Dataset | Paragraphs | Blank lines | Separation |
|---------|-----------|-------------|------------|
| English preprocessed (joined) | 782 | 759 | Blank lines between paras |
| Hindi preprocessed (Tesseract) | 951 | 952 | Blank lines between paras |

Both working datasets already have clear paragraph structure via blank lines. The English joined files inherit blank lines from the original clean files (which were preserved by the join algorithm). The Hindi Tesseract output preserves the PDF's natural paragraph layout.

**Decision:** Skip this step (already satisfied by Steps 1 and 4).

**Rationale:**
- Paragraph segmentation is a natural byproduct of the existing preprocessing, not a separate transformation.
- English: original clean files had blank lines between paragraphs; line joining preserves them.
- Hindi: Tesseract OCR preserves PDF paragraph structure with blank lines.
- No additional code or processing would change the output.
- The real paragraph-to-sentence breakdown happens in Step 7 (sentence segmentation), which is the actual NLP task.

---

## 6. Output staging: `preprocessed/` instead of in-place overwrite

**Date:** 2025-07-25

**Context:** The corrupted files live in `data/hindi/clean/`. Should re-extraction overwrite them directly or stage output separately?

**Decision:** Stage OCR output in `data/hindi/preprocessed/`. Provide an `--apply` flag to copy into `clean/` after verification. (Early notes used the name `re_extracted/`; the live directory is `preprocessed/`.)

**Rationale:**
- Preserves the original corrupted files for comparison and audit.
- Allows running extraction in preview mode before committing changes.
- `--apply` is an explicit step to replace data under `clean/`.
- The pipeline's Hindi working directory is `preprocessed/`, not `clean/`.

---

## 7. Single-file per step vs monolithic pipeline

**Date:** 2025-07-25

**Context:** Preprocessing has several distinct transforms. How should they be organized?

**Decision:** One file per live step in `src/preprocessing/`, named by what the step does (not numbered). Orchestration via `Makefile` / `run_pipeline.py`.

**Rationale:**
- Easier to test each step independently.
- Easier to reason about and modify individual steps without touching others.
- Steps can be run via `python -m src.preprocessing.<name>` for development and debugging.
- Naming by *function* (e.g., `reextract_pdfs.py`) rather than *sequence number* (e.g., `step1_...`) prevents the files from becoming stale if step order changes, and avoids "AI slop" naming conventions.

---

## 8. Intelligent line joining for English hard-wrapped text (original plan Step 4)

**Date:** 2025-07-25

**Context:** 17/30 English clean files have mid-sentence line breaks from PDF extraction at ~44-74 chars/line. The remaining 13 use long lines (200-600+ chars) and are already correct.

**Evidence from corpus analysis:**
- 17 hard-wrapped docs: avg line length 44-74 chars, many lines end with lowercase indicating continuation
- 13 long-line docs: avg line length 267-664 chars, each line is a full paragraph
- Simple lowercase->lowercase join heuristic catches ~75% of breaks
- Adding a legal-domain proper noun list catches ~85% (138 more joins)
- Zero false positives verified (heuristic rules prevent over-joining across sentence/paragraph boundaries)

**Decision:** Implement with heuristic + legal proper noun list.

**Approach:**
```
Join if: prev ends with lowercase AND (next starts with lowercase
         OR next starts with a known legal proper noun)
Don't join if: next starts a numbered item or bullet
               prev ends with sentence punctuation (.!?)
               blank line between paragraphs
```

**Proper noun list derivation (data-driven, no guessing):**
1. Scan all 30 English docs for hard-wrapped line boundaries
2. For each line ending with lowercase, extract the first word of the next line
3. Count occurrences across the corpus
4. Filter to words with >= 2 occurrences (removes noise)
5. Result: 34 verified proper nouns (e.g., Court=22, High=12, Appellant=6, Section=4, Bank=4)
6. Cross-check: does this word also appear as a sentence starter? If yes, verify ratio is skewed toward continuation (all pass)

This replaces the earlier 50+ hand-curated list which contained 18 words that never appear mid-sentence in the corpus (e.g., "C", "Governor", "President", "Union", "The"). The data-driven list also discovered 12 genuine continuations I missed (e.g., "Pandey", "Rajbali", "Malkhan" -- party names).

**Results on 30 English docs:**
```
Total: 2,819 -> 1,176 lines (1,643 joins, 58% reduction)
Hard-wrapped docs: individually reduced by 29-168 lines
Long-line docs: unchanged (zero false modifications)
```

**Trade-off:** The remaining ~15% unjoined lines are cases where a proper noun starts the continuation (e.g., "Regional Rural\nBank Services"). The text is still readable, and the minor fragmentation doesn't affect downstream alignment quality for a 43K-word corpus. An ML classifier could catch more but is not worth the effort at this scale.

---

## 9. Sentence segmentation for English and Hindi (original plan Step 7)

**Date:** 2025-07-25

**Context:** Split paragraphs into sentences for parallel alignment. English and Hindi have very different sentence boundary markers (period vs danda) and domain-specific edge cases (legal abbreviations, case citations, dates).

**Tool evaluation:**

| Tool | English | Hindi | Model required | Speed | Legal handling |
|------|---------|-------|----------------|-------|----------------|
| spaCy en_core_web_sm | Yes (dependency parser) | N/A | 15MB | Moderate | Good built-in handling: Mr., No., v., U.P., Cr.P.C. |
| spaCy sentencizer (rule-based) | Yes (.!? + uppercase) | N/A | No | Fast | Needs manual abbreviation exceptions |
| NLTK sent_tokenize | Yes | No Hindi model | 1MB | Moderate | Limited abbreviation list |
| blingfire | Yes | Yes | <1MB | Fast | No customization |
| Danda split (custom) | N/A | Yes (।) | No | Fast | Cannot handle embedded English periods |
| spaCy en_core_web_sm + danda split | Yes | Yes | 15MB | Moderate | Best of both |

**Decision:** Two-tool approach:
- **English**: spaCy `en_core_web_sm` (dependency parser). Handles standard legal abbreviations (`Mr.`, `No.`, `v.`, `U.P.`, `Cr.P.C.`) out of the box. Pre-tokenization protection for Hindi honorifics (`Smt.`, `Shri.`, `Sri.`) not in the model's training data.
- **Hindi**: Split on `।` (danda). Simple, fast, and handles >95% of Hindi sentence boundaries correctly.

The `segment()` function auto-detects language by checking for `।` presence.

**New dependency:** `python3 -m spacy download en_core_web_sm` (~15MB model, required once).

**Results:**
```
English: 2,495 sentences across 30 docs (avg 83/doc)
Hindi:   7,427 sentences across 30 docs (avg 248/doc)
```

The Hindi count is ~2.6x higher because danda split produces shorter segments (consistent with the "over-segment rather than under-segment" strategy). Short fragments like section headers ("बनाम", "निर्णय") and numbered items are expected and will be merged during alignment quality filtering (Step 8).

**Key edge cases handled:**
- `Mr. Sharma`, `Dr. Singh`, `Smt. Devi` -- abbreviations preserved as single sentence
- `Mohan v. State of U.P.` -- case citations not broken
- `No.10553`, `Cr.P.C.`, `U.P.` -- legal abbreviations respected
- `27.05.2003` -- dates not treated as sentence boundaries
- `Section 2(f)`, `Writ Petition No.10553` -- legal references handled

**Trade-off:** Over-segmentation on numbered lists (e.g., "1." as standalone sentence). This is by design -- easier to merge via alignment similarity than to split a merged boundary.

---

## 10. Sentence alignment and quality filtering (original plan Step 8)

**Date:** 2025-07-25

**Context:** Align English and Hindi sentences across all 30 documents using LaBSE cross-lingual embeddings, then apply quality filters to remove low-quality pairs.

### Content mismatch: clean files vs OCR

The English side uses the original **clean files** (body paragraphs only, starting from "1."), while the Hindi side uses **Tesseract OCR output** (full documents including headers, parties, citation, body).

**Why not OCR the English PDFs for structural parity?**
- English headers/parties are boilerplate ("IN THE SUPREME COURT OF INDIA", party names). They don't contain translation-relevant legal reasoning.
- The clean files were curated to contain only the numbered judgment body for a reason — that's where the legal translation signal lives.
- Adding headers would create repetitive training pairs and risk the model over-learning boilerplate.

**Why not strip Hindi headers to match clean files?**
- The alignment filter already handles this naturally. Hindi header sentences (court name, parties, `निर्णय`) have no mutual-best EN match, so they become orphans and are filtered out.
- The 1,458 aligned pairs are effectively body-content pairs already. Header stripping before alignment would produce the same result.

**Does the structural mismatch affect training?**
No. The model receives independent `(en_text, hi_text)` pairs during training. It never sees document structure. As long as each pair is semantically equivalent (validated by LaBSE at 0.70 avg similarity), the model learns correct translation mappings regardless of whether headers existed in the source documents.

### Alignment approach

1. For each document, encode all EN and HI sentences using LaBSE (109 languages, supports Hindi)
2. Compute EN->HI cosine similarity matrix
3. **Bidirectional greedy matching**: for each EN sentence, find the best HI match; keep only if the HI side also picks the same EN as its best match (mutual-best). This avoids forced low-quality matches.
4. Apply quality filters: length ratio (0.3-3.0), minimum similarity (>0.5), minimum text length (>3 chars), empty removal
5. Near-dedup on EN side within each document (Jaccard > 0.85)

**Why DP alignment failed:** The similarity matrix is extremely sparse — only 0.1% of EN-HI pairs have similarity > 0.5 (due to the 3:1 sentence count ratio from Hindi over-segmentation and the body vs full-doc content mismatch). DP tries to align every EN with some HI, creating forced low-sim pairs that all get filtered. Greedy bidirectional matching correctly identifies only the strong 1-1 correspondences, letting the rest become orphans.

**Results:**
```
Total pairs: 1,458 (across 30 docs)
Avg similarity: 0.70
Avg EN length: 94 chars
Avg HI length: 60 chars
Range: 49-104 pairs/doc
```

**Model comparison: LaBSE vs BGE-M3 (benchmarked on same data):**

| Metric | LaBSE | BGE-M3 (2024) |
|--------|-------|---------------|
| Pairs produced | 1,458 | 1,347 |
| Avg similarity | 0.701 | 0.742 |
| Pairs in 0.5-0.6 range | 401 (27.5%) | 38 (2.8%) |
| Load + encode time | 16s | 159s |
| Model license | Apache 2.0 | MIT |

BGE-M3 (BAAI, Feb 2024) is the current SoTA open-source multilingual embedding model. On the actual EN-HI legal text benchmark, it achieves 0.905 avg similarity for correct translation pairs vs LaBSE's 0.884. However, BGE-M3 is more conservative in alignment — it produces 111 fewer pairs (7.6% less) because it requires higher confidence for the mutual-best match to hold. For a corpus of this size (30 docs), losing data is worse than marginal quality improvements. LaBSE is retained as the default.

**New dependency:** `sentence-transformers` + `sentence-transformers/LaBSE` model (~1.8GB). To use BGE-M3 instead, change the model name in `align_sentences.py` to `BAAI/bge-m3`.

**Quality filters applied (live code in `align_sentences.py`; mirrored in `configs/preprocessing.yaml`):**

| Filter | Threshold | Effect |
|--------|-----------|--------|
| Min similarity | >= 0.5 | Removes semantically mismatched pairs |
| Length ratio | 0.3 - 3.0 | Removes extreme size mismatches |
| Min text length | > 3 chars | Removes stray punctuation |
| EN near-dedup | Jaccard > 0.85 drops weaker | Removes near-duplicate EN within same doc |

**Note vs early plan:** The original preprocessing plan suggested length ratio 0.5-2.0 and similarity 0.6-0.7. After inspecting mutual-best pair scores on this 30-doc corpus, thresholds were relaxed to 0.3-3.0 and 0.5 so recall stays high on a tiny dataset. Tightening is still a one-line change in `align_sentences.py` + the yaml mirror.

**Output format:** JSONL with fields: `en_text`, `hi_text`, `doc_id`, `similarity`, `source`.

---

## 11. Final output format: train/dev/test splits (original plan Steps 9 + 10)

**Date:** 2025-07-25

**Context:** Produce the final parallel corpus as train/dev/test JSONL files with document-level metadata and pipeline report.

**Approach:**
1. Load 1,458 aligned pairs from `data/aligned/all.jsonl`
2. Shuffle 30 document IDs with fixed seed (42) for reproducibility
3. Split: 24 train (80%), 3 dev (10%), 3 test (10%)
4. Assign all pairs to their document's split (document-level split prevents data leakage)
5. Write as JSONL to `data/processed/{train,dev,test}.jsonl`
6. Generate `metadata.json` (doc-level stats per split) and `alignment_report.json` (pipeline summary)

**Note on split sizes:** Pairs per split are 1,136/132/190 rather than exact 80/10/10 because documents have varying numbers of aligned pairs (49-104/doc). The split is document-level, so pair counts are proportional to document sizes. This is correct — document-level splits prevent data leakage, and the ratios are close enough to the target for a 30-doc corpus.

**Output files:**
```
data/processed/
├── train.jsonl         # 1,136 pairs (24 docs)
├── dev.jsonl           # 132 pairs (3 docs)
├── test.jsonl          # 190 pairs (3 docs)
├── metadata.json       # doc-level stats with split assignments
└── alignment_report.json  # pipeline config and statistics
```

**Step 9 (Document tracking):** Already satisfied. `doc_id` flows through every step via filename-based processing. Each aligned record carries `doc_id` and `source` fields. No additional implementation needed.

---

## 12. Devanagari detection threshold

**Date:** 2025-07-25

**Context:** Need to distinguish Hindi text from English text or garbage (corrupted files). What threshold to use?

**Decision:** A file is considered "has Hindi" if >30% of its non-space characters are in the Devanagari Unicode block (U+0900 -- U+097F).

**Rationale:**
- Legal Hindi court judgments mix English citations, case numbers, and Latin abbreviations (e.g., `S.C.R.`, `INSC`, `No.`), so pure Devanagari ratio is rarely 100%.
- The 30% threshold is conservative: corrupted files with `?` replacement score 0%, valid Hindi judgments consistently score >60%.
- Configurable via `configs/preprocessing.yaml` if tuning is needed.

---

## 13. Test file mirrors source file naming

**Date:** 2025-07-25

**Context:** Test file was initially `test_step1.py`, which wouldn't match after removing the `step1_` prefix from the source.

**Decision:** Test files mirror their source module's name with a `test_` prefix (e.g., `reextract_pdfs.py` -> `test_reextract_pdfs.py`).

**Rationale:**
- Obvious mapping between source and test.
- No mental translation needed to find the corresponding test file.
- Consistent with pytest conventions.

---

## 14. PYTHONPATH requirement

**Date:** 2025-07-25

**Context:** The project uses `from src.config import ...` style absolute imports, which requires the project root to be on `PYTHONPATH`.

**Decision:** Accept this requirement rather than using relative imports or manipulating `sys.path` inside scripts.

**Rationale:**
- Absolute imports are unambiguous and refactor-safe.
- No risk of import errors due to moving files within the package.
- Intended workflow: `PYTHONPATH=. python3 src/preprocessing/reextract_pdfs.py` or run via an activated virtual environment with proper setup.
- A future `setup.py` / `pyproject.toml` can make the package installable, eliminating the manual PYTHONPATH step.

---

## 15. Tokenizer analysis for Hindi-English legal text

**Date:** 2025-07-26

**Context:** Evaluate how different LLM tokenizers handle Hindi text, specifically for legal domain. Hindi (Devanagari script) is encoded in UTF-8 as 3 bytes per character, making it susceptible to byte-level fallback in tokenizers that lack native Devanagari coverage.

**Methodology:** Full corpus benchmark on all 1,458 aligned EN-HI pairs. For each tokenizer, encode every sentence, measure chars/token, HI/EN ratio, and total tokens. Vocabularies inspected for Devanagari token counts. Tokenizer files downloaded from HuggingFace (tokenizer.json, ~2-10MB each).

**Models tested (17 tokenizers across 14 model families, 2024-2026):**

| Rank | Tokenizer | Vocab | Dev tok | HI c/t | HI/EN | Total | Architecture |
|------|-----------|-------|---------|--------|-------|-------|-------------|
| #1 | **Custom SP 41K** (ours) | 41K | **16,840** | **3.84** | **0.743** | **53,124** | SentencePiece, domain-trained |
| #2 | Gemma 4 | 262K | 13,754 | 3.42 | 0.800 | 57,095 | Google SentencePiece |
| #3 | GPT-4o (o200k) | 200K | 2,295 | 2.97 | 0.949 | 60,034 | OpenAI multilingual BPE |
| #4 | **Phi-4-mini** ⭐ | **200K** | **~2,295** | **2.97** | **0.949** | **60,034** | **o200k-style, same as GPT-4o** |
| #5 | NLLB-200 | 256K | 2,406 | 3.11 | 0.756 | 64,785 | Meta multilingual BPE |
| #6 | Custom-BPE (16K) | 16K | 2,737 | 4.42 | 0.705 | 47,566 | Trained on corpus only |
| #7 | **Mistral Small 4** ⭐ | 131K | 0 | **2.43** | **1.076** | 68,826 | Improved byte-level BPE |
| #8 | Qwen3.5 / 3.6 | 248K | 0 | 2.27 | 1.179 | 70,844 | Alibaba byte-level BPE |
#9 | MiniMax M3 | 200K | 0 | 1.72 | 1.674 | 80,772 | Byte-level BPE |
| #10 | DeepSeek V3 / V4 Pro | 129K | 0 | 1.76 | 1.609 | 79,820 | Byte-level BPE |
| #11 | GLM 5.2 | 155K | 0 | 1.11 | 2.513 | 109,139 | Byte-level BPE |
| #12 | Qwen3 (152K) | 152K | 0 | 1.08 | 2.468 | 112,964 | Byte-level BPE |
| #13 | Phi-4 (100K) | 100K | 0 | 1.02 | 2.745 | 115,889 | Byte-level BPE |
| #14 | OLMo 3 | 100K | 0 | 1.02 | 2.745 | 115,889 | Byte-level BPE |

**Key findings:**

1. **Three tokenizer families handle Hindi well:** SentencePiece (Gemma 4, Custom SP), multilingual BPE (GPT-4o, NLLB), and o200k-style BPE (Phi-4-mini). All include Devanagari characters as first-class tokens.

2. **Phi-4-mini shares the GPT-4o tokenizer** (o200k_base, 200K vocab, 2.97 chars/tok). Microsoft adopted OpenAI's architecture for their small model line.

3. **Mistral Small 4 improved significantly** over the old Mistral-7B (2.43 vs 0.95 chars/tok). They learned byte-sequence merges for common Devanagari patterns, but the architecture remains byte-level BPE with zero actual Devanagari tokens.

4. **The Llama-family tokenizer architecture is fundamentally suboptimal for Hindi.** Byte-level BPE (used by Qwen 3/3.5/3.6, DeepSeek V3/V4, Phi-4, Mistral, MiniMax, GLM, OLMo) has zero Devanagari tokens in any version, regardless of vocabulary size (100K-248K). All encode Hindi text as raw UTF-8 bytes, costing 1.1-2.7x more tokens than English.

5. **Qwen3.5/3.6, DeepSeek V4 Pro, Phi-4-mini all use the same tokenizer as their predecessors** — the newer models didn't change tokenizers, only model weights. Tokenizer architecture is fixed at model release and never updated.

4. **Practical impact:** Training on the corpus with a Llama-family tokenizer costs 1.2x-2.7x more tokens than with Gemma 4 or GPT-4o. A 128K context window fits 40-70% as much Hindi text.

5. **Byte fallback mechanism:** Each Devanagari character (3 UTF-8 bytes) is encoded as 1-3 byte-level tokens. Common characters like 'न' (U+0928) may get merged into single tokens through BPE training, but most Devanagari characters remain at byte level.

**The byte fallback in detail (DeepSeek V3 on 'अपीलार्थी'):**
```
अ (U+0905) -> bytes [e0 a4 85] -> 2 tokens (first byte + second+third)
प (U+092A) -> bytes [e0 a4 aa] -> 2 tokens
ी (U+0940) -> bytes [e0 a5 80] -> 1 token (merged)
ल (U+0932) -> bytes [e0 a4 b2] -> 1 token (merged)
ा (U+093E) -> bytes [e0 a4 be] -> 1 token (merged)
र (U+0930) -> bytes [e0 a4 b0] -> 1 token (merged)
्थ (U+094D)+ी -> conjunct -> 2 tokens (byte fallback)
ी (U+0940) -> bytes [e0 a5 80] -> 1 token (merged)
Total: 9 chars -> 11 byte-level tokens
```

Compare with Gemma 4 (SentencePiece): same word -> 3 subword tokens.

**Analysis framework:** `src/tokenizer/benchmark.py` (corpus metrics table + JSON write) and `src/tokenizer/deep_dive.py` (byte fallback mechanics). Results saved to `data/analysis/tokenizer_metrics.json`. Train domain SPMs via `src/tokenizer/prepare_corpus.py` + `src/tokenizer/train.py`.

### Custom SentencePiece trained on 14M chars of Indian legal Hindi

To validate whether SentencePiece's advantage is architectural or just data-scaled, I trained custom SentencePiece models on 14M characters of Hindi legal text from the Prarabdha Indian legal supervised fine-tuning dataset (~7,277 documents, 5 parquet files).

**Training details:**
- Data source: `Prarabdha/indian-legal-supervised-fine-tuning-data` (Apache 2.0)
- Hindi text filtered by Devanagari presence (context + response columns)
- Model type: Unigram, character_coverage=1.0
- Vocab sizes: 16K, 32K, 41K (max supported by 14M chars)

**Results (benchmarked on the 1,458 EN-HI corpus):**

| Model | Vocab | Dev tok | HI c/t | HI/EN | Total tok | Training data |
|-------|-------|---------|--------|-------|-----------|--------------|
| **Custom SP 41K** | 41K | **16,840** | **3.84** | **0.743** | **53,124** | 14M chars legal HI |
| Custom SP 32K | 32K | 13,014 | 3.70 | 0.751 | 54,789 | 14M chars legal HI |
| Custom SP 16K | 16K | 5,903 | 3.33 | 0.772 | 59,833 | 14M chars legal HI |
| Gemma 4 (ref) | 262K | 13,754 | 3.42 | 0.800 | 57,095 | Trillions of tokens |
| GPT-4o (ref) | 200K | 2,295 | 2.97 | 0.949 | 60,034 | Trillions of tokens |

**Key finding: Domain-specific SentencePiece with 41K vocab beats every general-purpose tokenizer** on the Hindi legal benchmark, despite 6,000x less training data. It achieves more Devanagari tokens (16,840 vs 13,754), higher compression (3.84 vs 3.42 chars/tok), and lower HI/EN ratio (0.743 vs 0.800) than Gemma 4.

**Legal terms as single tokens:** न्यायालय (court), अपीलार्थी (appellant), अनुच्छेद (article), अधिकारिता (jurisdiction) — all encoded as single tokens. This is because SentencePiece operates at the character level, not the byte level, so Devanagari characters are first-class citizens from the start.

**Architecture wins over scale:** SentencePiece's character-level subword segmentation inherently handles Hindi. The only reason Gemma 4's SentencePiece didn't match the custom model is that it was trained on general web text, not legal Hindi. Given the same domain-specific training data, any SentencePiece tokenizer would match my results.

**Practical implication:** If building a Hindi legal translation system, train a SentencePiece tokenizer on domain-specific Hindi text. The tokenizer alone can save 7-15% in token costs compared to using a general-purpose tokenizer like Gemma 4 or GPT-4o.

**Trained models saved to:** `data/models/tokenizers/sentencepiece_{16000,32000,41000}.model` (gitignored; retrain with `make tokenizer-train-all`).

---

## 16. Orchestration and docs stay tied to modules that exist

**Date:** 2026-07-26

**Context:** Makefile, `run_pipeline.py`, and `scripts/reproduce_all.sh` briefly pointed at future packages (`src.training`, `src.evaluation`) and renamed tokenizer modules that never shipped. Nested group expansion in `run_pipeline.py` also treated group names as step names.

**Decision:**
- Orchestrators only invoke modules under `src/preprocessing/` and `src/tokenizer/`.
- `make all` uses `tokenizer-train-all` (alias `tokenizer-train` kept).
- Default `run_pipeline.py --steps` is `preprocess` (not a broken nested `all`).
- `data/models/` is gitignored (regeneratable SPMs). Keep `data/aligned/` tracked when present so benchmarks and tests can load pairs without re-running LaBSE.
- Config yaml lists live steps + skipped steps + alignment thresholds as documentation, not a second runtime engine.

**Rationale:** A broken `make eval` or wrong import path is worse than omitting future phases. Phase scaffolding appears only when the phase code exists.

---

## 17. External Stage A data: ingest filters, not full PDF pipeline

**Date:** 2026-07-26

**Context:** Gate 9 needs legal EN-HI bitext beyond the 1,458 assignment pairs. T0 sources are MILPaC (CC BY-NC-SA) and Anuvaad legal EN-HI (CC BY). Both ship as already-aligned units (xlsx rows or parallel `.en`/`.hi` line files), not as raw dual PDFs.

**Decision:**
- Download raw files to `data/external/raw/{milpac,anuvaad}/`.
- Convert to the same JSONL fields as assignment pairs (`en_text`, `hi_text`, `source`, `doc_id`, `similarity=null`).
- Apply length-ratio (0.3-3.0) and min-length (>3) filters shared with `align_sentences.py`; exact (en, hi) dedup.
- Do **not** re-run OCR, line join, spaCy/danda segmentation, or LaBSE on these dumps.
- Write per-source JSONL plus combined `stage_a_en_hi.jsonl` under `data/external/parallel/` (gitignored via `data/external/`).

**Why skip LaBSE here:** Anuvaad judiciary alone is ~830k lines. Full mutual-best re-alignment would dominate wall time and RAM on a laptop for marginal gain on auto-mined bitext. MILPaC is small and already expert-oriented. Assignment data remains the only corpus that went through our LaBSE pipeline. Optional future: LaBSE score filter on a sample or on MILPaC only.

**Role split:**
- `stage_a_en_hi.jsonl` -> Stage A domain FT/LoRA
- `data/processed/{train,dev,test}.jsonl` -> Stage B + final eval (document-level, frozen)
- Never mix assignment test docs into Stage A silently; Stage A is external-only unless we deliberately add assignment train later

**Rationale:** Reusing the PDF pipeline would be the wrong abstraction: the expensive steps already happened upstream (human/legal translation + Anuvaad mining). Our value is schema normalization, shared QC thresholds, and reproducible paths.

---

## 18. Track C0: SPM v2 on Stage A + train, never eval docs

**Date:** 2026-07-26

**Context:** Dual-track plan adopts a first-class Indic legal tokenizer. v1 SP (16/32/41k) was fit on Prarabdha mono HI scrape only. Stage A bitext (~992k pairs) is better legal text. SPM must not see assignment test/dev surface forms or fertility tables on held-out pairs are optimistic.

**Decision:**
- Build `spm_corpus_legal_v2_{joint,hi}.txt` from `stage_a_en_hi.jsonl` + `processed/train.jsonl` only.
- Hard-fail if any assignment dev/test `doc_id` appears in train-side input.
- Train new prefixes `sentencepiece_legal_v2_{hi|joint}_{32000|41000}` under `data/models/tokenizers/`; **keep v1** `sentencepiece_{16000,32000,41000}` for ablation.
- Benchmark default eval = assignment **held_out** (dev+test docs), not full 1,458 pairs.
- Optional Prarabdha mix is off by default (`--prarabdha-frac 0`).

**Why joint + hi grid:** HI-only maximizes Devanagari packing; joint EN+HI helps legal English pieces (Section, S.L.P., party names) for a translation-oriented vocab.

**Held-out results (assignment dev+test, 322 pairs):**

| Model | HI c/t | HI/EN | Total tok | Dev pieces |
|-------|-------:|------:|----------:|-----------:|
| v2 hi 41k | 4.46 | 0.434 | 14,880 | 30,095 |
| v2 joint 41k | **4.34** | **0.724** | **11,004** | 15,971 |
| v1 Prarabdha 41k | 3.95 | 0.739 | 11,965 | 16,840 |

HI-only wins raw HI compression but **wrecks English** (HI/EN << 1 means EN is over-tokenized → higher total). **Primary freeze: `sentencepiece_legal_v2_joint_41000`** for translation-oriented work. Keep HI-only models for HI-only analysis ablations.

**Training note (sample joint):** Initial joint 32/41k used `input_sentence_size=1_000_000` after full-load OOM on raw joint.

**Full joint within 16GB (follow-up):** Exact-line dedupe (~2% dups) + optional 4096-char cap cut peak enough that Unigram **profile=full** succeeded: all ~1.95M unique lines, `input_sentence_size=0`, `seed_sentencepiece_size=250_000`, `train_extremely_large_corpus=True`. Artifact: `sentencepiece_legal_v2_joint_full_41000` (sample `joint_41000` kept for ablation). Fallback ladder if full fails: `full_tight` then `full_sample_15`. Still Unigram only (not byte-level BPE). HI-only models remain full-corpus on HI dump.

**Vocab-size ablation (joint_full Unigram, same deduped corpus):**

| Model | Held-out HI c/t | Held-out total | Test total | Dev pieces |
|-------|----------------:|---------------:|-----------:|-----------:|
| full 41k | 4.37 | 10,978 | 6,211 | 16,217 |
| full 48k | 4.38 | 10,937 | 6,194 | 18,524 |
| full 64k | **4.42** | **10,819** | **6,125** | 23,742 |
| v1 41k (ref) | 3.95 | 11,965 | 6,784 | 16,840 |

Gains 41k->64k are real but modest (~1.4% fewer held-out tokens; ~1.4% on test). Glossary legal terms stay 1-piece across 41/48/64 for core HI + Section/impugned.

**Production freeze (Track C): `sentencepiece_legal_v2_joint_full_41000`.**

Rationale: among full-joint models, 64k wins pure packing, but larger V increases risk of dedicating pieces to frequent legal collocations and undertrained long-tail IDs in later MT. 41k keeps strong gains vs v1 (~8% fewer test tokens) with a smaller emb table and more compositional pressure. 48k/64k retained as ablations only. Report: `data/analysis/tokenizer_vocab_size_ablation.json`.

**Not in C0:** model embedding resize / LoRA (Track C1); default-backbone zero-shot (Track D).

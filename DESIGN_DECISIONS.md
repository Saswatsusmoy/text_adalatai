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

**Correction (2026-08-02, §34):** Doc 6's `preprocessed/` file later regressed to text-layer output (byte-identical to degraded `clean/6.txt`). Re-OCR restored 6,027 Dev chars. Docs 14/22/25/26 re-OCR was idempotent. See §34.

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

**Date:** 2026-07-26 (updated 2026-07-31)

**Context:** Makefile / `run_pipeline.py` must not call missing packages. Train and eval modules now exist as first-class phases.

**Decision:**
- Four phase packages: `preprocessing`, `tokenizer`, `training`, `evaluation` (see `docs/WALKTHROUGH.md`).
- `run_pipeline.py` registers only real modules; groups expand flatly. Default `--steps preprocess`. Group `all` = data path (preprocess + external + dual-policy + tokenizer bench), not multi-hour H200 train.
- Smoke train/eval steps (`train_nllb_smoke`, `zero_shot_smoke`) are optional groups; full curricula stay on Makefile H200 targets.
- `data/models/` and `data/runs/` gitignored (regeneratable / large). Keep `data/aligned/` trackable when present.
- Config yaml documents live steps; runtime is the Python modules + Makefile.

**Rationale:** Clear phase boundaries for interview and graders; no phantom targets.

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

Gains 41k->64k are real but modest (~1.4% fewer held-out tokens; ~1.4% on test). Glossary legal terms stay 1-piece across 41/48/64 for core HI + Section/impugned. *(Historical numbers, measured on the pre-§37 corpus with the old sus=True freeze; the freeze was retrained sus=False in §39.)*

**Production freeze (Track C): `sentencepiece_legal_v2_joint_full_41000`.**

Rationale: among full-joint models, 64k wins pure packing, but larger V increases risk of dedicating pieces to frequent legal collocations and undertrained long-tail IDs in later MT. 41k keeps strong gains vs v1 (~8% fewer test tokens) with a smaller emb table and more compositional pressure. 48k/64k retained as ablations only. Report: `data/analysis/tokenizer_vocab_size_ablation.json`.

**Not in C0:** model embedding resize / LoRA (Track C1); default-backbone zero-shot (Track D).

> **Supersedes note (§33):** the vocab-size ablation above (41k / 48k / 64k, Unigram only) and the individual freeze rationale are subsumed by the full 35-config tokenizer matrix in §33 (Cartesian `{unigram, bpe} x {16k..64k} x {joint, hi}` + 5 secondary-axis ablations on top-3 bases). The §33 matrix confirms the 41k freeze rationale on the joint corpus, adds BPE data (`bpe_41k` HI c/t 4.604 vs Unigram 4.609 at 41k), and identifies `bpe_64k_bf` / `unigram_64k_bf` as the recommendation for any future Track C rebuild. **`SPM_V2_PRIMARY` unchanged.**
>
> **Config-consistency revision (§39):** the freeze model file was originally trained with SPM's default `split_by_unicode_script=True`, while every matrix model uses `False` -- so the matrix did not actually confirm the freeze, and the "+7% packing 41k -> 64k" claim conflated vocab size with the script-split axis (true vocab-only gain ~1.9%). The freeze was retrained with `split_by_unicode_script=False` (same dedup corpus, profile `full`, seed/threads aside), verified from the model proto, and now benchmarks identical to the matrix 41k (HI c/t 4.715 on the post-alignment held-out set). See §39.

---

## 19. Default local training: Apple M4 16GB + MLX + MPS (optional remote H200)

**Date:** 2026-07-26 (updated 2026-07-27)

**Context:** Default plan is local Apple Silicon. A remote 2xH200 box later became available for optional faster Stage A when VRAM is free.

**Profile (scripted local):** MacBook Air, Apple M4, 10 cores, **16GB unified memory**. MLX default device GPU; mlx-lm import OK; PyTorch MPS matmul OK. Report: `data/analysis/hardware_profile.json` via `make profile-hardware`.

**Decision:**
- **Default:** no cloud/remote GPU **requirement**. Local path remains first-class and documented.
- **MLX / mlx-lm:** primary path for **decoder-only** LLM LoRA (prefer 1B-3B **4-bit**, batch 1, seq ~256).
- **PyTorch MPS:** primary path for **encoder-decoder** MT (NLLB-600M, InLegalTrans ~1B) zero-shot and careful PEFT on the laptop.
- **Optional remote:** jnan 2xH200 when free (or owner-approved carve-out). Same Track D recipe via `configs/training_h200.yaml` + DDP (see §23-24). Never kill root vLLM without explicit OK.
- Do not claim mlx-lm runs NLLB/InLegalTrans natively; they remain HF seq2seq.
- Custom SPM freeze still requires a training path that can load `SPM_V2_PRIMARY` (not automatic with stock mlx-lm + base LLM tokenizer).
- Stage A: subsample before full ~992k on 16GB; remote can use the same frozen A1 80k file.

**Rationale:** 16GB unified is tight for 7B+ full FT. MLX uses unified memory efficiently for LLM LoRA. MPS reuses the existing HF ecosystem for legal enc-dec baselines. Remote H200 is an accelerator for the **same** experiment contract, not a second research track.

**Docs:** `docs/HARDWARE_MLX.md`, DESIGN_DECISIONS §23-24.

---

## 20. Dual eval policies: internal vs external held-out

**Date:** 2026-07-26

**Context:** Assignment test is only ~190 pairs / 3 docs. External Stage A adds ~993k legal pairs. Users need broader metrics without training on the test set.

**Decision:**
- **Policy I (internal):** frozen assignment `test.jsonl` / `dev.jsonl` (document-level).
- **Policy E (external held-out):** before Stage A MT, carve MILPaC 10%/10% dev/test and Anuvaad 1k/3k dev/test (seed 42); remainder -> `stage_a_train.jsonl`.
- Every system scores on **I_test** and **E** suites (`E_milpac_test`, `E_anuvaad_test` or `E_external_test`).
- Never train MT on Policy E files. Never use full `stage_a_en_hi.jsonl` as both train and test.

**Rationale:** Policy I answers assignment fidelity; Policy E answers broader legal generalization. Reporting both detects overfitting to the 30-doc slice. MILPaC held-out is small but clean; Anuvaad held-out stabilizes automatic metrics.

**Code:** `src/preprocessing/split_external_eval.py`, `src/evaluation/eval_sets.py`. Make: `external-eval-split`.

---

## 21. Advanced training + monitoring strategy (before Stage A code)

**Date:** 2026-07-27

**Context:** Zero-shot NLLB baselines exist on I+E. Local M4 16GB only. Need efficient, accurate FT without leaking tests or overfitting 1k assignment pairs.

**Decision:** Adopt `docs/TRAINING_STRATEGY.md` + `configs/training.yaml` as the train contract before implementing the loop:

- Track D first: LoRA on NLLB-600M (MPS), not full FT.
- Curriculum Stage A: smoke 2k -> quality core (~50-80k) -> optional scale (~150k) before full ~988k.
- Stage A train file: `stage_a_train.jsonl` only; Stage B: assignment train only.
- Monitor: train loss, grad norm, memory; dual-policy chrF++ on dev (Anuvaad capped during train); rare full I/E tests.
- Checkpoint selection: weighted multi-suite primary + anti-forgetting hard constraints on Stage B.
- Decode parity with zero-shot (beam 4, max 256) for reported numbers.
- Run artifacts under `data/runs/{run_id}/` with config snapshot and data manifest.

**Rationale:** Separating strategy from code freezes success metrics and hardware limits so implementation does not improvise LR/data/eval. Dual-policy selection prevents "win I_test, lose legal domain." Curriculum avoids spending days on noisy full Anuvaad before clean legal signal is learned.

**Status (2026-07-27):** LoRA train loop implemented (`src/training/train_nllb_lora.py`). Local MPS default: `configs/training.yaml`. Remote Hopper DDP: `configs/training_h200.yaml` (§23-24).

---

## 22. NLLB LoRA targets decoder attention first

**Date:** 2026-07-27

**Context:** Need efficient, accurate FT of NLLB-600M on 16GB. Blind "all q_proj" LoRA is simple but not MT-targeted.

**Finding:** Model is M2M100 12-enc + 12-dec, d_model 1024, FFN 4096, shared 256k embeddings. Decoder **cross-attn** (`encoder_attn`) is the MT alignment hinge; decoder self-attn drives HI fluency; encoder already strong for EN.

**Decision:** Default `peft.profile: decoder_attn` (~3.15M params). Stage B prefer `cross_attn` or lower r. Avoid embedding/lm_head LoRA on Track D. Document profiles in `docs/NLLB_ARCHITECTURE.md`.

**Rationale:** Domain EN->HI is mostly alignment + HI realization. Freezing encoder reduces overfit to Anuvaad noise and cuts trainable surface without saving much peak activation memory (still dominated by seq length).

---

## 23. Optional remote H200 for Stage A1 (Hopper CUDA path)

**Date:** 2026-07-27

**Context:** Local M4 MPS A1 is correct but slow (hours for 3000 steps at batch 1 / accum 16). A jnan host (`ssh.jnan.ai`, hostname reports 2xH100-class VM) exposes **2x NVIDIA H200** (~141 GB each, SM90, driver 580 / CUDA 13.0 toolkit 12.6). Production vLLM (DeepSeek + Chandra) often fills both GPUs; training is allowed only when free or after owner-approved stop/carve-out.

**Hardware fact check:**
- Trust `nvidia-smi` (H200), not the hostname string.
- Host RAM ~440 GiB; `/data` writable (~300G free when probed).
- System python 3.12 has no torch; train env is a project venv with `torch` cu126 wheels.

**Ops policy:**
- Prefer free VRAM or explicit downtime. Do **not** kill root vLLM without owner OK.
- When co-resident: use free slice only; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; pick most-free GPU if single-process.
- When services stopped: full dual-GPU DDP (§24).
- Secrets (SSH password) never committed; remote notes stay private (e.g. `.local/`, gitignored).

**Decision (CUDA recipe, shared with §24):**
- Config: `configs/training_h200.yaml`.
- Helpers: `src/training/cuda_backend.py`.
- **bf16** weights/activations (Hopper-native; more stable FT than fp16).
- **TF32** for residual fp32 matmuls (`allow_tf32`, `set_float32_matmul_precision('high')`).
- **Flash / mem-efficient SDPA**; math SDP disabled when possible; `attn_implementation='sdpa'` on load.
- **Fused AdamW** on CUDA.
- **gradient_checkpointing off** when VRAM allows (faster; NLLB-600M LoRA is small).
- Data: frozen A1 subsample via `data.train_jsonl` (`stage_a_A1_n80000.jsonl`, seed 42) so remote need not ship full ~988k pool.
- Same LoRA `decoder_attn` r=16, alpha=32, lr=1e-4, 3000 Stage A steps, beam-4 gen eval parity with zero-shot.
- Run artifacts: `data/runs/nllb600_A_A1_h200_*/` with `config.snapshot.yaml`, `backend_info.json`, metrics JSONL, PEFT checkpoints.

**Rationale:** Hopper SM90 is built for bf16/TF32 and Flash attention. Optional remote keeps the **same experiment contract** as MPS (data, LoRA surface, steps, eval policy) while cutting wall-clock. Co-residence rules protect production inference.

**Reproduce (remote, after env exists under `/data/adalat_ai`):**
```bash
cd /data/adalat_ai && source .venv/bin/activate
export HF_HOME=/data/hf-cache PYTHONPATH=/data/adalat_ai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# dual GPU (preferred when both free):
torchrun --standalone --nproc_per_node=2 -m src.training.train_nllb_lora \
  --config configs/training_h200.yaml --curriculum A1 --device cuda
# or: make train-nllb-A1-h200
```

---

## 24. Dual-H200 DDP without changing the optimization problem

**Date:** 2026-07-27

**Context:** First H200 A1 used one GPU only (GPU1 idle). Variable-length batches caused `torch.compile` recompile storms (labels width mismatch). Goal: use **both** H200s and Tensor Cores hard, without shifting selection metrics vs the single-GPU / MPS recipe.

**Accuracy-preserving contract (frozen):**

| Knob | Value | Notes |
|------|------:|-------|
| Global batch | **32** | 16 per device x world 2 x accum 1 |
| LR | 1e-4 | **No** linear LR scaling with world size |
| Steps | 3000 | Optimizer steps after accum |
| Seed | 42 | DistributedSampler seed = run seed |
| Data | A1 80k freeze | `train_jsonl` |
| LoRA | decoder_attn r=16 | ~3.15M trainable (~0.51%) |
| Decode | beam 4, max 256 | Same as zero-shot / MPS |

**Speed / kernel decisions (do not change the loss definition):**
- Launch: `torchrun --standalone --nproc_per_node=2` (NCCL, single-node; `NCCL_IB_DISABLE=1` on non-IB VMs).
- DDP: wrap after LoRA; `find_unused_parameters=true` (PEFT-safe); rank 0 only logs, loss eval, gen eval, checkpoint save; barriers around eval/save; early-stop broadcast via `all_reduce` MAX on a stop flag.
- **Fixed pad** to `max_source_length` / `max_target_length` (256): `attention_mask=0` on pad tokens; labels padded with **-100** so CE ignores pads. Loss matches dynamic pad; shapes static for Flash SDPA + compile.
- **pad_to_multiple_of 8** when not using fixed pad (Tensor Core friendly dims).
- **No `torch.compile` under DDP.** First A1 DDP run hung after step-150 loss_eval; rank1 died with `NCCL communicator was aborted` while inductor/Dynamo tried to trace DDP collectives. Single-GPU may still compile. DDP path: eager + Flash SDPA + bf16 is enough for 600M LoRA.
- Batched generation for in-train gen eval (`decode.eval_batch_size`); unwrap DDP before `generate` and loss eval.
- Avoid periodic `empty_cache` on CUDA (allocator thrash); keep for MPS only.
- Pin memory, multi workers, prefetch for host->device pipeline.
- `init_process_group(backend=nccl, device_id=...)`; DDP `broadcast_buffers=False`.

**Deliberately not done (would change the problem or bloat stack):**
- Raising global batch to 64+ without a new experiment ID / LR study.
- DeepSpeed / FSDP (unnecessary for 600M + LoRA).
- Embedding / lm_head LoRA.
- Multi-node, or IB-specific NCCL tuning beyond single-node defaults.
- Expecting 100% sustained SM util on 600M LoRA -- model is tiny vs H200; dual-GPU helps wall-clock and eval, not "fill 700W forever."
- torch.compile + DDP for this model (unstable; see hang above).

**Code map:**
- `src/training/dist_utils.py` -- process group, unwrap, barriers.
- `src/training/cuda_backend.py` -- Hopper knobs.
- `src/training/nllb_data.py` -- collate pad options.
- `src/training/train_nllb_lora.py` -- DDP train loop.
- `configs/training_h200.yaml` -- production defaults.
- Make target: `train-nllb-A1-h200`.
- Tests: `tests/training/test_cuda_backend.py`, `test_nllb_data.py`.

**Observed (smoke of production path, 2026-07-26/27 UTC):**
- Both GPUs allocated (~14-15 GiB each during A1 LoRA); Flash SDPA + TF32 reported in `backend_info.json`.
- After compile warmup, order of ~10+ optimizer steps/s at global batch 32 (gen eval still dominates wall-clock when enabled).
- Run id pattern: `nllb600_A_A1_h200_ddp2_{timestamp}`.

**Rationale:** Speed must come from data-parallel throughput and better kernels, **not** from a different effective batch or learning-rate schedule. That keeps local MPS, single-GPU CUDA, and 2xH200 DDP runs comparable on weighted chrF++ selection metrics and on final I/E test scores.

---

## 25. Track D curriculum complete: ship A2, not Stage B

**Date:** 2026-07-27

**Context:** Original plan T4 (A1 -> A2) then T5 (Stage B + final I+E). All executed on free 2xH200 with DDP recipe from §24. Full **test** suites scored for zero-shot, A1 best, A2 best, B best (same beam 4 / max 256 / batch 32).

**Runs (remote `/data/adalat_ai`):**

| Phase | Run id (prefix) | Train data | Resume | Steps / LR |
|-------|-----------------|------------|--------|------------|
| A1 | `nllb600_A_A1_h200_ddp2_*` | A1 80k | base | 3000 / 1e-4 |
| A2 | `nllb600_A_A2_h200_A2_ddp2_*` | A2 150k | A1 best | 3000 / 5e-5 |
| B | `nllb600_B_full_h200_B_ddp2_*` | assignment 1136 | A2 best | 800 / 3e-5 |

Configs: `configs/training_h200.yaml`, `training_h200_A2.yaml`, `training_h200_B.yaml`.

**Final test scores (BLEU / chrF++):**

| System | I_test | E_milpac_test | E_anuvaad_test |
|--------|-------:|--------------:|---------------:|
| Zero-shot | 18.85 / 44.74 | 34.28 / 55.22 | 39.39 / 60.08 |
| A1 best | 21.67 / 49.16 | 34.66 / 55.98 | 45.17 / 64.33 |
| **A2 best** | **21.86 / 49.66** | **34.90 / 56.46** | **45.80 / 64.83** |
| B best | 23.10 / 48.89 | 30.92 / 51.22 | 40.44 / 59.60 |

**Stage A success bars** (`docs/TRAINING_STRATEGY.md` §9) on **A2 vs zero-shot:** all pass (I +4.9 chrF++, MILPaC +1.2, Anuvaad +4.8).

**Stage B constraints vs A2:** fail.
- I_test chrF++ not >= A2 (−0.77).
- E_milpac chrF++ drop **5.24** (limit 2.0).
- Anuvaad also drops hard (−5.2 chrF++).
- Only clear B win: I_test **BLEU** (+1.2 vs A2).

**Decision:**
1. **Production / dual-policy checkpoint = A2 `best_primary`**, not Stage B.
2. Path: `data/runs/nllb600_A_A2_h200_A2_ddp2_20260726T212958Z/checkpoints/best_primary` (remote same under `/data/adalat_ai`).
3. Keep Stage B run as an ablation: assignment overfit without replay mix.
4. If a later B is needed: fewer steps, stronger E weight / early stop on E_milpac_dev, or 90/10 assignment+A2 replay per strategy §2.2.

**Artifacts:**
- `data/analysis/final_dual_policy_report.json`
- `data/analysis/nllb600_{A1,A2,B}_h200_best_report.json` + matching `*_hyps.jsonl`
- `data/analysis/zero_shot_nllb_report.json` (H200 re-decode)

**Rationale:** The training contract optimizes dual policy, not I_test alone. A2 is the best joint point on I+E. Stage B without anti-forget data mix trades domain legal MT for assignment BLEU and violates the hard E drop rule.

---

## 26. Track C1: prefer C1c (NLLB vocab-extend), not from-scratch

**Date:** 2026-07-27

**Context:** C0 froze `joint_full_41000`. Dual-track plan orders C1b/C1c then C1a. An initial C1a-small Marian (~66M from scratch) was scaffolded and started on H200, then **stopped**: with only ~80k bitext, from-scratch MT is a weak quality path vs NLLB priors.

**Decision (primary C1 = C1c; two experiment variants):**

**v1 (ablation, finished):** bulk `add_tokens` of 8k raw SPM piece strings + full emb train (`modules_to_save: embed_tokens`). Can break good NLLB singles and destabilize emb (gen primary ~25–28). Artifact `nllb600_c1c_sp_ext`.

**v2 (primary, careful extend):**
1. Surface forms only (strip SPM `▁`).
2. Add only if base NLLB **fragments** (>=2 pieces) and not already single-token.
3. Reject surfaces that are substrings of protected probes; after add, **require** use + fewer pieces; **reject if regression** on probes (e.g. `न्यायालय` must stay `▁न्यायालय`).
4. Rebuild clean tokenizer with survivors only (~1500 tokens, vocab 257669).
5. Mean-init new rows from **base** tokenizer encode only.
6. Train: LoRA `decoder_attn` + **grad mask** on emb so only rows `[old_vocab:)` get gradients (not full emb).
7. Same A1 80k / 3000 steps / beam-4 decode as Track D for fair compare.

**Artifacts:**
- v1: `vocab_extend_nllb.py`, `training_c1c_h200.yaml`
- v2: `vocab_extend_nllb_v2.py`, `training_c1c_v2_h200.yaml`, `data/models/nllb600_c1c_sp_ext_v2/`
- Emb checkpoint: `_save_peft` writes `new_embed_rows.pt` when `new_embed_start` set; eval loads via `apply_new_embed_rows` (PEFT LoRA-only adapter does not store those rows).
- C1a Marian scaffold kept optional.

**Full test scores (H200, beam 4 / max 256 / batch 32):**

| System | I_test | E_milpac_test | E_anuvaad_test |
|--------|-------:|--------------:|---------------:|
| Zero-shot | 18.85 / 44.74 | 34.28 / 55.22 | 39.39 / 60.08 |
| **D A2** | **21.86 / 49.66** | **34.90 / 56.46** | **45.80 / 64.83** |
| C1c v2 careful A1 | 17.79 / 43.86 | 28.20 / 49.78 | 37.64 / 58.46 |
| C1c v1 bulk A1 | 6.38 / 24.86 | 10.66 / 28.63 | 15.65 / 34.35 |

v2 run: `nllb600_A_A1_c1c_v2_h200_ddp2_20260726T234856Z` (retrain after emb-save fix; best step 1000, dev primary ~53.6).

**Decision (after test):** production dual-policy remains **Track D A2**. C1c v2 loses to zero-shot on all three suites (not only A2). C1c v1 is a negative ablation (bad extend). Track C does not replace Track D on this budget/recipe.

**Rationale:** Careful extend was the right *methodology* experiment, but +1500 legal tokens + A1 LoRA did not beat stock NLLB priors under dual-policy scoring. Possible causes: emb mean-init still undertrained, A1-only (no A2 scale), or vocab change harms shared NLLB geometry more than legal atoms help. Further C1c (A2 resume / more emb steps) is optional research, not production.

---

## 27. Stage B' anti-forget replay mix (next train after A2)

**Date:** 2026-07-27

**Context:** Pure Stage B (assignment-only from A2) raised I_test BLEU but failed dual-policy: E_milpac chrF++ drop 5.24 (>2.0 limit). Research and TRAINING_STRATEGY §2.2 prescribe assignment+domain replay, not a new PEFT method.

**Decision:** Implement and run **Stage B'** before DoRA / other PEFT upgrades.

| Item | Choice |
|------|--------|
| Mix | All assignment train (~1136) + replay so assignment ≈ **90%** of pairs |
| Replay pool | Frozen A2 subsample `stage_a_A2_n150000.jsonl` (exact EN-HI pairs in assignment excluded) |
| Seed | 42 |
| Resume | A2 `best_primary` adapters (continues decoder_attn r=16; profile not re-inited) |
| LR / steps | 2e-5 / max 500 steps, gen eval every 100, patience 4 |
| Selection | stage_b weights I 0.6 + E_milpac 0.4; hard E drop caps unchanged |
| Config | `configs/training_h200_Bp.yaml` |
| Builder | `src/training.subsample.build_stage_b_replay_mix` (`--curriculum Bp`) |
| Make | `make stage-b-replay-mix`, `make train-nllb-Bp-h200` |

**Promote B' over A2 only if** final E_milpac_test drop ≤ 2.0 chrF++ vs A2 **and** I improves or holds on dual-policy. If E dies again, keep A2 production and move to DoRA-on-A2 (not more pure-B variants).

**Rationale:** Catastrophic forgetting on ~1k pairs is a data-mix problem first. Replay is the minimal controlled experiment; PEFT renames would confound the ablation.

**Status (2026-07-27 H200 run complete):**

| Item | Value |
|------|--------|
| Run | `nllb600_B_Bp_h200_Bp_ddp2_20260727T011740Z` |
| Wall | ~278s train (500 steps DDP2); best_primary **step 100** |
| Full test (beam 4) | I 22.22/49.41; MILPaC 33.82/54.84; Anuvaad 43.46/62.51 |

| System | I_test | E_milpac | E_anuvaad |
|--------|-------:|---------:|----------:|
| A2 | 21.86 / 49.66 | 34.90 / 56.46 | 45.80 / 64.83 |
| Pure B | 23.10 / 48.89 | 30.92 / 51.22 | 40.44 / 59.60 |
| **B'** | **22.22 / 49.41** | **33.82 / 54.84** | **43.46 / 62.51** |

- E_milpac vs A2: chrF++ drop **1.62** (≤ 2.0 hard bar) -- anti-forget **pass** vs pure B's 5.24 fail.
- I chrF++ vs A2: **−0.25** (BLEU +0.36); does not clearly improve assignment under dual-policy.
- Anuvaad vs A2: **−2.32** chrF++.
- vs pure B: large E recovery (MILPaC +3.62, Anuvaad +2.91 chrF++).

**Decision after test:** production stays **A2**. B' is a successful anti-forget *method* ablation (prefer over pure B) but not a dual-policy winner. Optional next: DoRA on A2; more Stage B variants not required.

---

## 28. DoRA ablation on A2 data (next PEFT after B')

**Date:** 2026-07-27

**Context:** Research ladder: after B' (anti-forget OK, dual-policy still A2), next PEFT upgrade is **DoRA** (weight-decomposed LoRA) with a single knob vs production A2 LoRA.

**Decision:**

| Item | Choice |
|------|--------|
| Method | PEFT `LoraConfig(use_dora=True)` on `decoder_attn` r=16 α=32 |
| Data | Frozen A2 subsample `stage_a_A2_n150000.jsonl` (same as Track D A2) |
| Init | **From base NLLB** -- cannot resume LoRA adapters as DoRA |
| LR / steps | 1e-4 / 3000 (from-base schedule; A2 LoRA used 5e-5 only because it resumed A1) |
| Config | `configs/training_h200_A2_dora.yaml` |
| Make | `make train-nllb-A2-dora-h200` |
| Code | `build_lora_config` honors `peft.use_dora` or `peft.method: dora` |

**Compare:** full I/E tests vs A2 LoRA under same decode. Promote only if dual-policy improves. If flat or worse, keep A2 LoRA production.

**Rationale:** DoRA is the highest-EV PEFT upgrade after LoRA saturates; keeps adapter economics. Fair comparison is same data + module surface; full A1→A2 DoRA curriculum is optional follow-up if this under-trains relative to A2's two-stage exposure.

**Result (2026-07-30, H200 bf16 batch=32 beam=4, same protocol as A2 LoRA):**

| Suite | A2 LoRA (shipped) | A2 DoRA | Delta |
|-------|------------------:|--------:|------:|
| I_test        | 21.86 / 49.66 | 21.80 / 49.18 | -0.05 / -0.47 |
| E_milpac_test | 34.90 / 56.46 | 35.23 / 56.43 | +0.33 / -0.03 |
| E_anuvaad_test | 45.80 / 64.83 | 45.42 / 64.43 | -0.38 / -0.40 |

All BLEU/chrF++ deltas are inside +/-0.5 chrF++. DoRA neither improves nor regresses dual-policy quality at this budget on this data with the same module surface. Decode elapsed: DoRA 554.8s vs A2 LoRA 269.8s -- DoRA adds magnitude scaling per forward, roughly doubling decode cost at inference for equal-quality output.

**Decision:** production stays **A2 LoRA `best_primary`**. DoRA is a completed method ablation with a documented equal-quality / higher-latency outcome, not a promotion candidate. Followups not run (optional): DoRA with an A1 -> A2 curriculum (this run went from-base since PEFT LoRA adapters cannot resume as DoRA); DoRA at higher rank; DoRA on encoder + decoder attention. Report: `data/analysis/nllb600_A2_dora_h200_best_report.json` + hyps under same tag prefix.

---

## 29. Submission package: single report + code, not multi-GB runs

**Date:** 2026-07-30

**Context:** Assignment asks for a report and modular code. The repo had complete science (`docs/EXPERIMENTS.md`, dual-policy scores, hyps) but no single grader-facing write-up, and most MT code was still uncommitted.

**Decision:**

| Item | Choice |
|------|--------|
| Report | `REPORT.md` (tokenizer, data, train, BLEU/chrF++, qualitative ZS vs A2, reflection) |
| Deep dive | Keep `docs/EXPERIMENTS.md` + `story/` as optional depth |
| Git | Commit source, configs, tests, analysis JSON/hyps; **not** `data/runs/`, `data/models/`, `data/external/` |
| Production | Document A2 adapters path; base from HF; adapters local or attached by submitter |

**Rationale:** Graders need a clear path through the work. Multi-GB Stage A and run trees are regeneratable and are already gitignored; shipping them would dominate the archive without improving review.

---

## 30. Modular phase layout for interview walkthrough

**Date:** 2026-07-31

**Context:** Scoring wants clean modular code with preprocess / tokenizer / train / eval clearly separated and reproducible. Interview needs a fixed tour path without losing experiments.

**Decision:**
- Keep four packages; put a short phase map in each package `__init__.py` (no logic).
- Shared JSONL I/O in `src/utils/jsonl.py`; call sites re-export or wrap so tests keep working.
- Central paths stay in `src/config.py` (aligned, Stage A, analysis, runs).
- Interview script: `docs/WALKTHROUGH.md` (15-min order + module table). Do not rename experiment modules or drop Track C/C1c/B/DoRA configs.
- Orchestrator and `reproduce_all.sh` cover data + tokenizer + dual-policy validation; full train remains Makefile.

**Rationale:** Separation without a rewrite. Every experiment remains findable by the same module name; only wiring and docs clarify the boundaries.

---

## 31. MBR decoding

**Date:** 2026-07-31

**Method:** Replace beam4 with MBR: sample N candidates per source, output the one with highest mean pairwise sentence-chrF utility vs peers. Eikema & Aziz 2020; Freitag et al. 2022 (chrF utility).

**Wiring:**

| Item | Value |
|------|-------|
| Module | `src/evaluation/mbr_decode.py` |
| CLI | `src.evaluation.zero_shot_nllb --mbr --mbr-samples N --mbr-temperature T --mbr-top-p P --mbr-utility {chrf,chrfpp}` |
| Sampling | `do_sample=True, num_beams=1, top_p=0.9, temperature=1.0, num_return_sequences=N` (single `generate` call per input batch) |
| Utility | sentence chrF++ (word_order=2); chrF (word_order=0) also selectable |
| Default N | 8 |
| Tag | auto-append `_mbr{N}` when `--tag` omitted so hyp files don't clobber beam4 runs |
| Make | `eval-mbr-a2`, `eval-mbr-zs`, `eval-mbr-smoke` |

**Cost:** decode = N× beam1 sampling (single generate, N return sequences); MBR pick = N² sentence-chrF calls per source, negligible vs generation.

**Measured (A2 adapters):**

| Suite | System | BLEU | chrF++ | n |
|-------|--------|-----:|-------:|--:|
| I_test | H200 bf16 beam4 (shipped) | 21.86 | 49.66 | 190 |
| I_test | MPS fp16 beam4 (control) | 21.85 | 49.68 | 190 |
| I_test | MPS fp16 MBR N=8, top_p=0.9, T=1.0 | 18.16 | 47.13 | 190 |
| E_milpac_test | H200 bf16 beam4 (shipped) | 34.90 | 56.46 | 117 |
| E_milpac_test | MPS fp16 beam4 (control) | 34.71 | 56.55 | 117 |
| E_milpac_test | MPS fp16 MBR N=8, top_p=0.9, T=1.0 | 31.39 | 54.07 | 117 |

Device / precision delta (MPS beam4 − H200 beam4): I_test −0.01/+0.02, E_milpac −0.18/+0.09 — inside noise.

Decode-only delta (MPS MBR − MPS beam4): I_test **−3.68 BLEU / −2.56 chrF++**, E_milpac **−3.33 BLEU / −2.48 chrF++**.

E_anuvaad_test (n=3000) not run: MPS budget ~4h at this rate.

Reports: `data/analysis/nllb600_A2_mps_mbr8_best_report.json`, `data/analysis/nllb600_A2_mps_beam4_best_report.json`. Hyps under same tag prefix.

**Decision:** do not promote MBR at these settings. Beam4 remains the shipped decode. Follow-ups worth trying before rejecting MBR fully:

1. Lower temperature (T=0.3–0.5) or epsilon-sampling (ε=0.02, Freitag 2022) — the current top_p=0.9 T=1.0 produces high-diversity candidates that hurt consensus.
2. Higher N (32–128) — literature uses N ≥ 32; N=8 is a low-end MBR configuration.
3. COMET utility instead of chrF — reference-free QE-guided MBR is where recent papers report the actual gains.

None of these were run in this pass — negative result stands at the tried configuration.

---

## 32. BPE vs Unigram at v2 41K (packing ablation)

**Date:** 2026-07-31

**Context:** §15 flagged SentencePiece BPE as Unicode-aware (distinct from byte-level BPE). §17-§18 shipped Unigram 41K joint. But no BPE 41K on the v2 joint corpus had been trained -- a gap. The v1 note about "BPE at 16K looked promising on raw packing" was never scaled.

**What was run:** train SentencePiece BPE 41K on the exact same deduped v2 joint corpus (`spm_corpus_legal_v2_joint_dedup_c4096.txt`), profile `full`, `character_coverage=1.0`, same pad/unk/bos/eos IDs. Only `model_type` changes. Bench on the same 322 held-out (assignment dev+test) pairs.

**Result** (held-out, 322 pairs):

| Model | Vocab | HI c/t | HI/EN | Total tok | Dev pieces |
|-------|------:|-------:|------:|----------:|-----------:|
| v2 joint 41K Unigram (shipped) | 41000 | 4.37 | 0.720 | 10,978 | 16,217 |
| **v2 joint 41K BPE** | 41000 | **4.40** | 0.721 | **10,898** | 16,371 |
| v2 joint 48K Unigram | 48000 | 4.38 | 0.721 | 10,937 | 18,524 |
| v2 joint 64K Unigram | 64000 | 4.42 | 0.722 | 10,819 | 23,742 |

Delta: BPE 41K vs Unigram 41K = +0.03 HI c/t (+0.7%), -80 total tokens (-0.7%), +154 Devanagari pieces. BPE 41K packing sits between Unigram 48K and 64K at the 41K parameter budget.

**Decision:**

Do not switch the freeze. `SPM_V2_PRIMARY` stays at Unigram 41K:

1. Delta is small (0.7% packing). Not worth churning the shipped constant, existing configs, and any downstream that assumes the model file.
2. Track D shipped uses NLLB native tokens, not v2 SPM. The v2 SPM is Track C material and Track C1c already showed vocab surgery on pretrained NLLB is not free.
3. BPE vs Unigram is not a shipping decision without a translation-quality run, which was not performed.

**Kept as an ablation artifact:** `data/models/tokenizers/sentencepiece_legal_v2_joint_full_bpe_41000.model`. Available for a future C1a from-scratch build or a fresh C1c-style NLLB extension.

**Followups not run:**

1. MT train on BPE 41K (C1a from-scratch or C1c extend) -- required to answer "is BPE better for legal MT" beyond packing.
2. BPE at 48K / 64K to see if it dominates Unigram at those sizes too.
3. Subword-regularization comparison (Unigram has native sampling; BPE-dropout is separate).

---

## 33. Full tokenizer matrix (35 configs)

**Date:** 2026-07-31

**Context:** §32 closed the 41K BPE-vs-Unigram gap but only at one vocab size and only with default options. To answer "what tokenizer configuration is actually best for legal EN-HI packing?" a systematic sweep was needed.

**Method:** Two-phase sweep on H200 (48 CPU cores, parallel-6).

Phase 1 -- main matrix (20 configs, all defaults):
- `model_type` in {unigram, bpe}
- `vocab_size` in {16k, 32k, 41k, 48k, 64k}
- `corpus_key` in {v2_joint, v2_hi}

Phase 2 -- secondary-axis ablation (15 configs; top-3 Phase-1 joint bases x 5 single-axis toggles):
- `byte_fallback=True`
- `character_coverage=0.9995`
- `split_digits=True`
- `split_by_unicode_script=True`
- `user_defined_symbols` = 22 legal EN+HI protected terms

Code: `src/tokenizer/matrix_configs.py`, `train_matrix.py` (subprocess-per-config, resumable manifest via `data/analysis/tokenizer_matrix_manifest.json`), `bench_matrix.py` (auto-discovery + legal-term probe + UNK rate).
Make: `tokenizer-matrix-phase1`, `tokenizer-matrix-phase2`, `tokenizer-matrix-bench`.

**Result (joint corpus; v2_hi excluded as MT-unusable -- EN legal-probe rate 0-50%):**

Phase 1 best packing (held-out HI c/t / total tokens on 322 pairs):

| Vocab | Unigram HI c/t | BPE HI c/t | Best total tok |
|------:|---------------:|-----------:|---------------:|
| 16k | 4.295 | 4.303 | 11,084 |
| 32k | 4.564 | 4.527 | 10,403 |
| 41k | 4.609 | 4.604 | 10,253 |
| 48k | 4.634 | 4.638 | 10,166 |
| **64k** | **4.695** | **4.695** | **10,027** |

Phase 2 axis effects (average delta across the 3 base configs):

| Axis | Delta HI c/t | Notes |
|------|-------------:|-------|
| `byte_fallback=True` | 0.000 | Same packing; UNK stays 0.00%; free robustness |
| `character_coverage=0.9995` | -0.029 | Introduces 0.83% UNK; not worth 0.03 c/t loss |
| `split_digits=True` | **-0.700** | Catastrophic; splits case numbers, dates, section numbers |
| `split_by_unicode_script=True` | -0.237 | Kills mixed-script (EN-in-HI-sentence) subwords |
| `user_defined_symbols` (22 legal) | -0.246 | Also drops legal-HI probe rate 1.00 -> 0.33 (UDS interferes with merge lattice) |

**Decision:** `SPM_V2_PRIMARY` stays `sentencepiece_legal_v2_joint_full_41000.model` (Unigram 41K).

Reasons:
1. Track D shipped uses NLLB native tokens, so any v2 SPM change affects no shipped output.
2. Track C artifacts and configs reference the current 41k freeze.
3. The packing gain going 41k -> 64k is only ~1.9% once the `split_by_unicode_script` axis is held constant (4.609 -> 4.695 HI c/t at sus=False; the earlier "4.37 -> 4.695 = +7%" conflated vocab size with the sus True->False change). That does not justify churn for a track that already lost dual-policy in §26.

> **Revision (§39):** the freeze model file was retrained with `split_by_unicode_script=False` so the matrix now genuinely applies to it. The +7% figure above was a conflation; the true vocab-only gain is ~1.9%. `SPM_V2_PRIMARY` filename unchanged.

**Recommendation for future Track C rebuild (any C1a from-scratch or fresh C1c-style extend):** use `sentencepiece_legal_v2_v2_joint_bpe_64000_bf.model` or the unigram equivalent -- HI c/t 4.695, byte_fallback for OOV robustness, 100% legal HI + EN probe hit-rate, 0.00% UNK.

**Followups not run:**
1. MT-quality run (Track C1c-style vocab extend) with the winner config -- packing is not translation quality.
2. Rank sensitivity: same matrix at rank 8 / 16 / 32 LoRA on the extended model.
3. Subword-regularization on the winner during MT training (Unigram sampling alpha or BPE-dropout p).
4. Corpus expansion: repeat matrix with WikiMatrix + FLORES lines mixed into v2_joint.

**Artifact map:**
- 35 model+vocab pairs: `data/models/tokenizers/sentencepiece_legal_v2_v2_*.model` (joint variants pulled local; hi variants stayed on H200 -- MT-unusable)
- Full bench: `data/analysis/tokenizer_matrix.json`
- Training manifest: `data/analysis/tokenizer_matrix_manifest.json`

---

## 34. Re-OCR docs 6/14/22/25/26 after text-layer regression (doc 6)

**Date:** 2026-08-02

**Context:** Independent verification found `data/hindi/preprocessed/6.txt` byte-identical to degraded text-layer `clean/6.txt` (4,657 Dev chars; mid-word ligature splits such as `भार ीय`, `सिसविवल`). DESIGN §1 claimed Tesseract for all Hindi and for the CORRUPTED set, but doc 6's working corpus (segmented + all 36 train pairs) was text-layer junk. Docs 14/22/25/26 already matched Tesseract-quality text (re-OCR was idempotent).

**Actions:**
1. Tesseract re-OCR (`--doc-ids 6 14 22 25 26`): doc 6 -> **6,027** Dev chars; ligatures fixed (`भारतीय सर्वोच्च न्यायालय`, `सिविल`).
2. Text-layer backup under `data/hindi/preprocessed/_backup_textlayer_20260802/`.
3. Re-segment HI only for those docs; re-align with **merge** so other 25 docs' pairs are kept.
4. Rebuild `data/processed/{train,dev,test}.jsonl` via frozen Policy-I doc IDs (`src.config`).
5. Rebuild Stage B' replay mix (`stage_b_Bp_a1136_r126_f0.9.jsonl`) from the new train.jsonl.

**Code:**
- `align_sentences.run(..., merge=)` -- partial `--doc-ids` merges into `all.jsonl` instead of wiping other docs.
- `output_format.run` prefers `TRAIN_DOC_IDS` / `DEV_DOC_IDS` / `TEST_DOC_IDS` over reshuffle when the 30-doc set is present.
- `segment_sentences` honors `--lang`.

**Pair counts unchanged** (1458 total; doc 6 still 36 pairs) but HI refs for doc 6 are OCR body text. EN side still body-only clean files (unchanged content-mismatch design).

**Not re-run:** Track D A2 / B / B' / DoRA training or COMET rescoring. Shipped A2 scores were measured against the previous HI refs (doc 6 text-layer). Retrain/rescore is a separate decision if dual-policy numbers must reflect the repaired refs.

**Regression tests:** `tests/preprocessing/test_corrupted_docs_ocr.py` (OCR floors + no text-layer markers on doc 6; align merge unit test).

---

## 35. Preprocessing tests were data-mutating

**Date:** 2026-08-02

**Context:** Re-running the test suite after §34 reverted the doc-6 OCR fix. Root cause: the preprocessing tests wrote into the real data tree, not tmp dirs.

**Evidence:**
- `test_reextract_pdfs.py::test_pdftotext_backend_works` calls `reextract_single(6, backend='pdftotext')`, which saves straight into `data/hindi/preprocessed/6.txt` (`reextract_pdfs.py` `out_path = HI_PREPROCESSED_DIR / f'{doc_id}.txt'`). The pdftotext text-layer output (4,657 Dev chars) overwrote the §34 OCR fix (6,027).
- `test_reextract_*` and `test_apply_copies_files` wrote into real `preprocessed/`/`clean/`.
- `test_segment_sentences.py::test_run_all` rewrote all 60 `segmented/*.txt`.
- `test_output_format.py::test_output_files_exist` regenerated `data/processed/` (gitignored, so harmless, but still real-data writes).

**Decision:** all reextract/segment/output-format tests that write now monkeypatch their output dirs (`HI_PREPROCESSED_DIR`, `HI_CLEAN_DIR`, `OUTPUT_DIRS`, `OUTPUT_DIR`) to `tmp_path`. Reads of committed artifacts stay on real paths.

**Verification:** full `pytest tests/` (157 tests) green; `preprocessed/6.txt` and `segmented/6.txt` byte-identical before/after the run; doc-6 OCR fix intact.

---

## 36. OCR invariant: preprocessed must be Tesseract, enforced in code

**Date:** 2026-08-02

**Context:** §34 relied on a manual backup dir (`_backup_textlayer_20260802/`) and a single hardcoded Homebrew path for `pdftotext`. Both were fragile: nothing prevented a silent re-write of text-layer output into `preprocessed/`, and the tool path broke off-Homebrew machines.

**Changes:**
- `src/config.py`: `PDFTOTEXT_CMD = shutil.which('pdftotext')` (None if absent); `extract_with_pdftotext` returns `None` in that case.
- `src/preprocessing/reextract_pdfs.py`:
  - `TEXTLAYER_MARKERS` (mid-word ligature splits proven to be text-layer glyph streams) and `MIN_OCR_DEV` (per-doc Devanagari floors; doc 6 text-layer was 4,657 vs OCR 6,027).
  - `verify_ocr_quality(doc_ids)` returns issues for any file failing the floors or containing markers.
  - `run()` verifies its own output and moves failing docs into `failed`; `--verify-ocr` CLI exits 1 on issues.
- `Makefile`: `verify-ocr` gate.
- `tests/preprocessing/test_corrupted_docs_ocr.py`: imports the shared constants instead of redefining them; added a negative test proving degraded text-layer input is flagged.

**Why floors + markers instead of trusting the file:** Devanagari char count alone cannot distinguish OCR from text-layer for docs 14/22/25/26 (identical counts in both). The marker set (which only text-layer output produces, e.g. `सिसविवल`, `भार ीय`) catches the distinctive degradation on doc 6; floors cover the generic "too few Dev chars" case for all five.

**Verification:** `make verify-ocr` exits 0 on the real corpus; negative tests confirm a degraded `6.txt` is flagged by both the floor and marker paths, and `run(backend='pdftotext')` reports doc 6 as failed. Full suite (160 tests) green; preprocessed/segmented files byte-identical after the run.

---

## 37. Hindi line joining: danda-aware mirror of the English hard-wrap step

**Date:** 2026-08-02

**Context:** §8 joined hard-wrapped **English** lines (`join_lines.py`), but the Hindi side had no equivalent. Tesseract OCR hard-wraps Hindi mid-sentence at PDF line boundaries (median 71 chars/line), so a logical sentence spans 2+ OCR lines and only the last carries a danda (।). Because `segment()` routes every danda-less line to the spaCy EN path, wrapped fragments survived segmentation as truncated danda-less "sentences" and entered alignment: **806/1,458 (55.3%) of aligned HI texts lacked a danda** -- the largest single root cause of alignment defects.

**Evidence (scan of all 30 preprocessed files):**
- 4,322/5,117 non-empty lines do not end in a sentence terminator; 3,315 contain no danda at all.
- Non-terminator line lengths are bimodal: lines <= 40 chars are almost always case headers / judge names / section labels (बनाम, निर्णय, court names, citation list items); lines > 40 chars are almost always genuine mid-sentence wraps (median 71).
- A naive "join every non-danda-ending line" merges the top-of-document header block and interleaved `उद्घोषणा`/`अस्वीकरण` blocks into single lines, so a short-line guard is required.

**Rules (`should_join`, documented as WHY comments):**
1. Never join across a blank line (paragraph boundary).
2. Never join if the **next** line starts a numbered item / bullet / list marker -- Arabic and Devanagari digits, short roman numerals, `(क)` / `(॥)` markers, `-`/`•`. **Dates (DD.MM.YYYY, incl. Devanagari digits) are exempt**: a date-start line is usually a mid-sentence continuation (54 such cases were falsely blocked before the exemption).
3. Never join if the **current** line ends with a sentence terminator (danda `।`/`॥`, tolerating trailing punctuation/quotes such as `।"`).
4. Never join if either line is a **standalone header** (case header, judge name, section label): short (<= 40 chars), danda-less, and matching a header vocabulary/pattern (`बनाम`, `निर्णय`, `उद्घोषणा`, `अस्वीकरण`, `प्रतिवेद्य`, `हेडनोट`, `कोरम`, `प्रस्तुतियाँ`, `नई दिल्ली`, `न्यायमूर्ति...`, `(न्यायमूर्ति...`, `X:` / `X;`). The **next** line is checked too: a long wrap line must not absorb a following header. Rare short continuations (`के साथ`, `प्रथम तल`) start with a postposition and still join.
5. Otherwise join (mid-sentence OCR hard wrap). Idempotent: joining an already-joined file is a no-op.

**Module:** `src/preprocessing/join_hindi_lines.py` (`should_join` / `join_lines` / `process_doc` / `run`, mirroring §8). Writes `data/hindi/preprocessed/` in place; reads the same dir, so the OCR invariant (§36) is unaffected (Devanagari counts unchanged). Wired as `join_hi` into `make preprocess` and `run_pipeline.py` (`reextract -> join -> join_hi -> segment -> align -> output`).

**Results on the real corpus (full re-run of join -> segment -> align -> output):**

| Metric | Before | After |
|--------|-------:|------:|
| Preprocessed HI non-empty lines | 5,117 | 1,749 |
| Segmented HI sentences | 7,418 | 3,221 |
| Segmented HI danda-less | 4,501 (60.7%) | 1,118 (34.7%) |
| Aligned pairs | 1,458 | 1,422 |
| Aligned HI without danda | 806 (55.3%) | 155 (10.9%) |
| Avg LaBSE similarity | 0.70 | 0.779 |
| Train / dev / test pairs | 1,136 / 132 / 190 | 1,110 / 128 / 184 |

**Remaining danda-less aligned texts (155) are largely legitimate:** standalone headers (`बनाम`, `निर्णय`, court names) that happen to align, colon-terminated definition intros (`... इस प्रकार है:-`), and OCR-garbled citations.

**Known limitations (conservative by design):**
- Blank-line-separated wraps are not joined (e.g. doc 1's charge list splits each item across a blank line); ~11 such fragments remain in doc 1.
- Headers longer than 40 chars (e.g. `[एसएलपी (क्रि.)संख्या 2354 वर्ष 2023 से उत्पन्न]`) merge into the adjacent line; harmless since headers are alignment orphans.
- OCR noise inside otherwise-joined sentences (dates like `46.07.2044`, stray `।"` / `!` tokens) is not repaired here.

**Tests:** `tests/preprocessing/test_join_hindi_lines.py` -- tmp_path/monkeypatch only (never touches real data, per §35); synthetic wraps, real doc-6 OCR snippet, headers preserved, real-corpus idempotency (all 30 preprocessed files joined twice must be a no-op), next-header absorption guard. Pipeline-order test updated for the `join_hi` step; output-format pair-count test updated to the regenerated 1,422.

**Pair-count change flagged:** the total is **1,422, not 1,458** -- joined HI units are longer and complete, so LaBSE mutual-best matching pairs slightly differently per doc. Do not silently claim the old 1,458.

**Revision (2026-08-02):** the first version of this step was non-idempotent: `should_join` only guarded the *previous* line, so a long wrap absorbed a following header (`निर्णय`), and re-running the join on that output absorbed the next sentence too (15/30 docs changed on pass 2). The committed data was also not reproducible from the committed code (22/30 docs differed). Fix: header detection now checks both lines and uses a vocabulary/pattern set rather than length alone; regenerated the full chain from raw OCR. Verified fixed-point (0/30 docs change on re-join) and segment(join(preprocessed)) == segmented (30/30).

---

## 38. Alignment quality gates: similarity floor 0.6 + margin + junk filters

**Date:** 2026-08-02

**Context:** §37 (Hindi line-join) fixed the 55% danda-less-fragment problem but the alignment threshold stayed at `MIN_SIMILARITY = 0.5`, a deliberately loose floor set for recall on a tiny corpus. It admitted a tail of weak pairs: on the post-join data, 99 pairs (7%) sat in the 0.5-0.6 band (OCR-garbled dates, partial overlaps, header fragments), plus number-only and truncated-fragment pairs.

**Changes (`src/preprocessing/align_sentences.py`):**
1. `MIN_SIMILARITY` 0.5 -> **0.6**.
2. `SIM_MARGIN = 0.01`: mutual-best winner must beat the runner-up by >= 0.01 on BOTH sides. Kills exact/near-exact ties (duplicate boilerplate like `There will be no order as to costs.`), set low enough that genuine legal sentences with near-equal alternatives survive. Measured: 0.02 dropped 15 complete sentences; 0.01 drops only 4.
3. Junk-pair filters in `quality_filter`:
   - number-only pairs (both sides pure digits): no translation signal.
   - EN fragments ending in a bare preposition/conjunction (`of`, `and`, `the`, ...), length-gated to <= 60 chars. Data check: all true truncations were < 60 chars; legal English legitimately ends long sentences (100+) in `of`/`and`/`the`, so the gate protects those.
4. Dead code removed: `SKIP_PENALTY` (DP cost, unused), `pair_type` (always "1-1"), `matched_hi` (unused set).

**Effect (full re-align):**

| Metric | Before (§37) | After |
|--------|-------------:|------:|
| Aligned pairs | 1,422 | **1,300** |
| Avg LaBSE similarity | 0.779 | **0.796** |
| Pairs < 0.6 | 99 | **0** |
| HI without danda | 155 (10.9%) | **109 (8.4%)** |
| Train / dev / test | 1,110 / 128 / 184 | **1,010 / 122 / 168** |

**Why the margin is the right tool:** the 122 removed pairs are almost all weak (99 sub-0.6, 3 number-only, 11 short-dangling fragments, plus exact-tie duplicates). Only 4 complete sentences (sim 0.63-0.83) are lost -- genuine near-tie ambiguities where two HI sentences are near-equivalent matches and the alignment choice is arbitrary. That is a defensible trade for a corpus whose every remaining pair is >= 0.6.

**Verification:** `make verify-ocr` exits 0; join fixed-point 0/30; full suite 185 tests green; `make lint` clean. New tests: margin keep/drop, number-only, short-dangling reject, long-sentence-with-preposition keep.

---

## 39. Tokenizer freeze retrained: split_by_unicode_script False (matrix-consistent)

**Date:** 2026-08-02

**Context:** Independent proto inspection found the shipped freeze `sentencepiece_legal_v2_joint_full_41000.model` was trained with SPM's **default** `split_by_unicode_script=True`, while all 35 matrix models use `False` (`TokenizerConfig` default). `train.py` never passed the option, so it silently inherited True. Consequence: the §33 matrix did NOT actually confirm the freeze, and the "+7% packing 41k -> 64k (4.37 -> 4.695)" claim conflated two axes (vocab size AND the script-split axis). The true vocab-only gain at sus=False is ~1.9% (4.609 -> 4.695).

**Also found:** SentencePiece `TrainerSpec` has **no `seed` field** -- passing `seed=42` raised `RuntimeError: NOT_FOUND`. The `TokenizerConfig.seed` field is informational only; SPM trainer randomness is not user-controllable. The matrix's "reproducible/seed 42" claim was unenforceable by construction.

**Changes:**
1. `train()` now accepts `split_by_unicode_script` (default `False`, matrix family) and passes it explicitly; CLI flag `--split-by-unicode-script`.
2. `train_full_joint` threads the flag through; `train_matrix` no longer passes the invalid `seed`.
3. Retrained the freeze on the 2x H200 box (`profile=full`, same dedup corpus, sus=False): proto verified `split_by_unicode_script=False`, vocab 41000, Unigram.
4. Re-benchmarked on the post-alignment held-out set: **identical to matrix 41k** (HI c/t 4.715, total 17,521) -- the freeze is now genuinely matrix-confirmed.

**Note on bench numbers:** the held-out set shrank from 322 (pre-§37) to 290 pairs (122 dev + 168 test after alignment tightening in §38), so absolute c/t figures differ from the §33 tables; the freeze-vs-matrix equivalence is what matters and holds exactly.

**Verification:** `make lint` clean; tokenizer tests pass; model proto confirms the setting.

---

## 40. Training harness Phase 3 fixes: caps + z-score, NaN deadlock, resume, MPS scaler, parity, hash, registry

**Date:** 2026-08-03

**Context:** Independent deep-read of `src/training/train_nllb_lora.py` against `docs/TRAINING_STRATEGY.md` §5.3 and `configs/training.yaml` found the code did not enforce what the docs promise. The docs (and DESIGN §21, §25) promise "weighted multi-suite primary + anti-forgetting hard constraints on Stage B" and z-standardization (`0.5*z(E_milpac)+0.3*z(E_anuvaad)+0.2*z(I_dev)` for Stage A); the code computed a **raw weighted mean** of chrF++ and picked argmax. The pure-B run's E_milpac chrF++ drop of **5.24 vs the 2.0 cap** was only caught post-hoc. Several other latent harness hazards existed (NaN DDP deadlock, silent embed-row resume, MPS fp16 without scaling, false MPS/H200 batch-parity, unverified train pool, no run registry). This is harness code + tests + docs only: no checkpoint, frozen subsample, or data file changed.

**Decisions:**

1. **Z-scored primary + caps live in a new pure module `src/training/selection.py`.** Exact formula (matches TRAINING_STRATEGY intent, now literally implemented):
   `z_i = (s_i - mean_i) / std_i` when `std_i > 0`, else `z_i = s_i - b_i` (single-row baseline degenerates to a delta), and `primary = sum_i(w_i * z_i) / sum_i(w_i)`. `mean_i`/`std_i` are **population** statistics over all `gen_eval` rows of the baseline eval log (the "from the baseline" option); `b_i` is the chrF++ of the baseline run's best-primary row. Choice: baseline-history stats over a fixed std or a live moving window, because a single baseline value carries no scale and the baseline log already exists on every resume.
2. **Caps are hard constraints, not soft**: `stage_b_max_drop_<suite>` (or `selection.caps`) keys; candidate is rejected as `best_primary` if `b_i - s_i > cap_i` for any capped suite, and it counts toward `bad_evals`/patience. `cap_ok`/`cap_violations`/`z` are appended to each gen eval row for archaeology.
3. **Gating protects Stage A**: `selection_mode()` returns `raw` for Stage A unless `eval.selection.baseline` is configured (or `zscore` forced); Stage B defaults to `zscore`. Behavior for existing Stage A runs is unchanged (same weighted mean).
4. **Baseline source**: `eval.selection.baseline` explicit path wins; otherwise the resumed checkpoint's run dir `metrics/eval_log.jsonl`, weighted by **that run's** `config.snapshot.yaml` selection weights so the argmax row matches how the resumed best was picked. Fall back to raw weighted mean with a WARN when unavailable.
5. **NaN DDP deadlock (P0-2)**: replaced the `if main: ... all_reduce_max` pattern with `sync_nan_stop()`; every rank computes its local NaN/Inf flag and all ranks hit the MAX collective before breaking, so no rank 1 can outrun rank 0 into a hung NCCL op.
6. **Resume safety (P0-3)**: `apply_new_embed_rows(required=...)` raises when `new_embed_rows.pt` is absent while `peft.new_embed_start` is configured (previously silent `False` meant the extended rows silently reverted to init); the new-embed grad mask is reinstalled on resume via the shared `_install_new_embed_grad_mask`. Eval callers (`zero_shot_nllb`) keep the non-required default.
7. **MPS fp16 safety (P0-4)**: MPS now runs fp32 master weights + `torch.autocast('mps', fp16)` compute + `torch.amp.GradScaler('mps')`. Verified on torch 2.13 that `GradScaler('mps')` **constructs but raises `ValueError: unscale FP16 gradients` on fp16 master weights** -- so fp32 masters are required, not optional. `build_grad_scaler()` gates this to MPS + a low-precision request; CUDA bf16 is unchanged (bf16 needs no scaling). Impact on local MPS runs: ~2x base-weights memory for correct numerics. The fp32-master override lives in `train_nllb_lora.build_model` only, so `train_legal_mt.py` (deprecated Track C1, stopped) is untouched except the harmless `autocast_ctx` mps branch.
8. **Batch parity (P1-5)**: `global_batch_parity()` warns when `batch_size * world * accum` disagrees with `train.global_batch_size` (`strict_global_batch: true` raises). Added `global_batch_size: 16` to `training.yaml` so the local 16 vs H200 32 gap is explicit and catchable instead of an implicit parity claim.
9. **Train-pool hash verify (P1-6)**: at launch, recompute the `file_sha256` prefix for every `*_sha256_prefix` in the data manifest and warn on drift (`data.strict_source_pool_hash: true` raises). Covers `source_pool`, `assignment`, and `replay_pool`. `resolve_train_path` merges the frozen build's sibling `*_manifest.json` hash keys into the `train_jsonl` override manifest, so the frozen A1/A2/Bp files are actually verified against the current pool (the override path previously carried no hash and drifted silently).
10. **Run registry (P1-7)**: `register_run()` maintains `{output_root}/runs.json` mapping `run_id -> {run_dir, stage, curriculum, config_snapshot, data_manifest, backend_info, train_log, eval_log, run_summary, best_primary, resume_adapters, start_ts}`, so finding "A2 best" is a JSON lookup, not grep archaeology.

**What was deliberately scoped out:** re-training anything, touching the frozen A1/A2 subsamples, editing historical score tables, adding COMET, and implementing the moving-window z-score variant (baseline-history chosen). The `train_legal_mt.py` (Track C1) selection loop still uses the raw weighted mean via `_primary_chrf` (kept as a thin wrapper for that import).

**Verification:** `make lint` clean; new tests `test_selection.py` + `test_train_harness.py` + MPS-scaler additions in `test_cuda_backend.py`; full suite green.

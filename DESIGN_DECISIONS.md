# Design Decisions

This document records architectural and implementation decisions made during the project, along with the rationale behind each.

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

## 2. Skipped: Strip UTF-8 BOM (original plan)

**Date:** 2025-07-25

**Context:** 25/30 Hindi clean files start with `\ufeff` (UTF-8 BOM). The original plan prescribed stripping it to avoid tokenizer issues and alignment offsets.

**Evidence from corpus analysis:**
- `data/hindi/clean/` -- 25 files have BOM (the original Windows-extracted files). 5 files (6, 14, 22, 25, 26) have no BOM (these were the corrupted ones, replaced by Tesseract output).
- `data/hindi/preprocessed/` -- **0 files have BOM**. Our Tesseract OCR extraction produces clean UTF-8 without BOM.
- `data/english/clean/` -- **0 files have BOM**.
- Source code -- **0 files have BOM**.

**Decision:** Skip this step.

**Rationale:**
- Our working directory `data/hindi/preprocessed/` is already BOM-free (30/30 files).
- The BOM only affects the legacy `clean/` files, which are not used by the pipeline.
- BOM-related problems (regex `^\d+\.` failing, first token being `\ufeff1.` instead of `1.`) don't apply to our pipeline.
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
- Zero instances of the `li.`->`ii` or `lili.`->`iii` OCR artifact exist in our corpus.
- All `L.` instances are legitimate legal abbreviations: judge names (`L. Nageswara Rao, J.`), case types (`S.L.P.` = Special Leave Petition), legal references (`L.Rs.` = Legal Representatives).
- The `i.` `ii.` `iii.` found in Doc 1's PDF raw extraction are correctly-formatted lowercase roman numerals in a legal charge list, not OCR artifacts. The clean file preserves them correctly.
- A naive regex `\b[Ll]i\.\b` would break `L. Nageswara Rao` -> `ii. Nageswara Rao` and `S.L.P.` -> `S.ii.P.` -- a classic false-positive example.
- This artifact is common in other legal PDF corpora (particularly older scans), but our source PDFs and Tesseract OCR don't produce it.

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

The 25 CRLF files in `clean/` are the same 25 that have BOM -- both trace to the same Windows extraction tool. The 5 files we replaced with Tesseract output have neither BOM nor CRLF. CRLF and LF are perfectly segregated within files (zero mixed-endings files found).

**Decision:** Skip this step.

**Rationale:**
- `data/hindi/preprocessed/` is already 100% LF (30/30 files). Our pipeline operates on preprocessed/.
- Python's `open()` in text mode auto-normalizes `\r\n` -> `\n` on read, so CRLF causes no issues when reading clean/ files.
- Regex with `$` in multiline mode matches at `\n` (after `\r`), so CRLF doesn't break pattern matching.
- Zero files have mixed endings within the same file, so no edge cases to handle.
- Like BOM, this is purely a legacy `clean/` concern that doesn't reach our pipeline.

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

## 2. Output Staging: `re_extracted/` directory instead of in-place overwrite

**Date:** 2025-07-25

**Context:** The corrupted files live in `data/hindi/clean/`. Should Step 1 overwrite them directly or stage output separately?

**Decision:** Stage output in `data/hindi/re_extracted/`. Provide an `--apply` flag to copy into `clean/` after verification.

**Rationale:**
- Preserves the original corrupted files for comparison and audit.
- Allows running the pipeline in "preview mode" before committing changes.
- `--apply` gives a deliberate, explicit step to replace corrupted data.
- Follows the principle of separating concerns: extraction (Step 1) and deployment (a downstream action).

---

## 3. Single-file per step vs monolithic pipeline

**Date:** 2025-07-25

**Context:** The preprocessing pipeline has 10 distinct steps. How should they be organized?

**Decision:** One file per step in `src/preprocessing/`, named by what the step does (not numbered).

**Rationale:**
- Easier to test each step independently.
- Easier to reason about and modify individual steps without touching others.
- Steps can be run via `python -m src.preprocessing.<name>` for development and debugging.
- Naming by *function* (e.g., `reextract_pdfs.py`) rather than *sequence number* (e.g., `step1_...`) prevents the files from becoming stale if step order changes, and avoids "AI slop" naming conventions.
- A future `run_pipeline.py` orchestrator can sequence them by importing and calling each module's `run()` function.

---

## 5. Intelligent line joining for English hard-wrapped text (original plan Step 4)

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

This replaces the earlier 50+ hand-curated list which contained 18 words that never appear mid-sentence in our corpus (e.g., "C", "Governor", "President", "Union", "The"). The data-driven list also discovered 12 genuine continuations we missed (e.g., "Pandey", "Rajbali", "Malkhan" -- party names).

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

BGE-M3 (BAAI, Feb 2024) is the current SoTA open-source multilingual embedding model. On our actual EN-HI legal text benchmark, it achieves 0.905 avg similarity for correct translation pairs vs LaBSE's 0.884. However, BGE-M3 is more conservative in alignment — it produces 111 fewer pairs (7.6% less) because it requires higher confidence for the mutual-best match to hold. For a corpus of this size (30 docs), losing data is worse than marginal quality improvements. LaBSE is retained as the default.

**New dependency:** `sentence-transformers` + `sentence-transformers/LaBSE` model (~1.8GB). To use BGE-M3 instead, change the model name in `align_sentences.py` to `BAAI/bge-m3`.

**Quality filters applied:**

| Filter | Threshold | Effect |
|--------|-----------|--------|
| Min similarity | > 0.5 | Removes semantically mismatched pairs |
| Length ratio | 0.3 - 3.0 | Removes extreme size mismatches |
| Min text length | > 3 chars | Removes stray punctuation |
| EN near-dedup | Jaccard < 0.85 | Removes near-duplicate EN within same doc |

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

## 6. Devanagari detection threshold

**Date:** 2025-07-25

**Context:** Need to distinguish Hindi text from English text or garbage (corrupted files). What threshold to use?

**Decision:** A file is considered "has Hindi" if >30% of its non-space characters are in the Devanagari Unicode block (U+0900 -- U+097F).

**Rationale:**
- Legal Hindi court judgments mix English citations, case numbers, and Latin abbreviations (e.g., `S.C.R.`, `INSC`, `No.`), so pure Devanagari ratio is rarely 100%.
- The 30% threshold is conservative: corrupted files with `?` replacement score 0%, valid Hindi judgments consistently score >60%.
- Configurable via `configs/preprocessing.yaml` if tuning is needed.

---

## 7. Test file mirrors source file naming

**Date:** 2025-07-25

**Context:** Test file was initially `test_step1.py`, which wouldn't match after removing the `step1_` prefix from the source.

**Decision:** Test files mirror their source module's name with a `test_` prefix (e.g., `reextract_pdfs.py` -> `test_reextract_pdfs.py`).

**Rationale:**
- Obvious mapping between source and test.
- No mental translation needed to find the corresponding test file.
- Consistent with pytest conventions.

---

## 8. PYTHONPATH requirement

**Date:** 2025-07-25

**Context:** The project uses `from src.config import ...` style absolute imports, which requires the project root to be on `PYTHONPATH`.

**Decision:** Accept this requirement rather than using relative imports or manipulating `sys.path` inside scripts.

**Rationale:**
- Absolute imports are unambiguous and refactor-safe.
- No risk of import errors due to moving files within the package.
- Intended workflow: `PYTHONPATH=. python3 src/preprocessing/reextract_pdfs.py` or run via an activated virtual environment with proper setup.
- A future `setup.py` / `pyproject.toml` can make the package installable, eliminating the manual PYTHONPATH step.

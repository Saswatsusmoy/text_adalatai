import json
from pathlib import Path

from src.preprocessing.align_sentences import (
    load_sentences, process_doc, run, quality_filter, dedup,
    MIN_SIMILARITY, MIN_CHAR_RATIO, MAX_CHAR_RATIO,
)


class TestLoadSentences:
    def test_load_en_hi(self):
        en, hi = load_sentences(1)
        assert len(en) > 0
        assert len(hi) > 0

    def test_content_not_empty(self):
        en, hi = load_sentences(1)
        assert all(s.strip() for s in en)
        assert all(s.strip() for s in hi)


class TestQualityFilter:
    def test_empty_en_rejected(self):
        assert not quality_filter({"en_text": "", "hi_text": "test", "similarity": 0.9})

    def test_empty_hi_rejected(self):
        assert not quality_filter({"en_text": "test", "hi_text": "", "similarity": 0.9})

    def test_low_sim_rejected(self):
        assert not quality_filter({"en_text": "test", "hi_text": "test", "similarity": MIN_SIMILARITY - 0.1})

    def test_good_pair_accepted(self):
        assert quality_filter({"en_text": "Hello world", "hi_text": "नमस्ते दुनिया", "similarity": 0.8})

    def test_bad_ratio_rejected(self):
        too_long = "x" * 1000
        short = "y"
        assert not quality_filter({"en_text": too_long, "hi_text": short, "similarity": 0.9})
        assert not quality_filter({"en_text": short, "hi_text": too_long, "similarity": 0.9})


class TestDedup:
    def test_identical_keeps_best(self):
        en = ["The court dismissed the appeal.", "The court dismissed the appeal."]
        hi = ["अदालत ने अपील खारिज कर दी।", "कोर्ट ने अपील खारिज की।"]
        sims = [0.9, 0.7]
        en_r, hi_r, sims_r = dedup(en, hi, sims)
        assert len(en_r) == 1

    def test_no_duplicates(self):
        en = ["First sentence.", "Second sentence."]
        hi = ["पहला वाक्य।", "दूसरा वाक्य।"]
        sims = [0.9, 0.85]
        en_r, hi_r, sims_r = dedup(en, hi, sims)
        assert len(en_r) == 2

    def test_empty_input(self):
        en_r, hi_r, sims_r = dedup([], [], [])
        assert en_r == []


class TestProcessDoc:
    def test_process_doc_1(self):
        pairs = process_doc(1, verbose=False)
        assert len(pairs) > 0
        for p in pairs:
            assert "en_text" in p
            assert "hi_text" in p
            assert "doc_id" in p
            assert p["similarity"] >= MIN_SIMILARITY

    def test_all_pairs_have_content(self):
        pairs = process_doc(1, verbose=False)
        for p in pairs:
            assert len(p["en_text"].strip()) > 0
            assert len(p["hi_text"].strip()) > 0

    def test_devanagari_in_hi(self):
        from src.utils.validation import has_devanagari
        pairs = process_doc(1, verbose=False)
        for p in pairs:
            assert has_devanagari(p["hi_text"]), f"HI missing Devanagari: {p['hi_text'][:50]}"


class TestRun:
    def test_output_file_exists(self):
        assert Path("data/aligned/all.jsonl").exists()

    def test_output_format(self):
        with open("data/aligned/all.jsonl") as f:
            lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            p = json.loads(line)
            assert "en_text" in p
            assert "hi_text" in p
            assert "doc_id" in p
            assert "similarity" in p
            assert "source" in p

    def test_total_pairs(self):
        with open("data/aligned/all.jsonl") as f:
            pairs = [json.loads(line) for line in f]
        assert len(pairs) >= 1000, f"Expected >=1000 pairs, got {len(pairs)}"
        assert len(pairs) <= 3000, f"Expected <=3000 pairs, got {len(pairs)}"

    def test_avg_similarity(self):
        with open("data/aligned/all.jsonl") as f:
            pairs = [json.loads(line) for line in f]
        avg_sim = sum(p["similarity"] for p in pairs) / len(pairs)
        assert 0.6 <= avg_sim <= 0.8, f"Avg sim {avg_sim:.4f} outside expected range"

    def test_near_dedup_effective(self):
        with open("data/aligned/all.jsonl") as f:
            pairs = [json.loads(line) for line in f]
        en_texts = [p["en_text"] for p in pairs]
        # Within same doc, no duplicates expected
        from collections import defaultdict
        by_doc = defaultdict(list)
        for p in pairs:
            by_doc[p["doc_id"]].append(p["en_text"])
        for doc_id, texts in by_doc.items():
            dupes = len(texts) - len(set(texts))
            assert dupes == 0, f"Doc {doc_id}: {dupes} duplicate EN texts"

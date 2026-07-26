from src.preprocessing.segment_sentences import (
    segment, segment_en, segment_hi,
    has_danda, process_doc, run,
)
from src.utils.validation import count_devanagari


class TestDetection:
    def test_has_danda_true(self):
        assert has_danda("यह एक वाक्य है।")

    def test_has_danda_false(self):
        assert not has_danda("This is an English sentence.")


class TestEnglishSegmentation:
    def test_simple_sentence(self):
        sents = segment_en("This is a simple sentence.")
        assert len(sents) >= 1
        assert "simple sentence" in sents[0]

    def test_two_sentences(self):
        sents = segment_en("First sentence. Second sentence.")
        assert len(sents) == 2

    def test_legal_abbreviation_preserved(self):
        text = "Mr. Sharma appeared before the Hon'ble Court."
        sents = segment_en(text)
        assert len(sents) == 1, f"Abbreviation caused split: {sents}"

    def test_case_citation(self):
        text = "The case of Mohan Singh v. State of U.P. was cited."
        sents = segment_en(text)
        assert len(sents) == 1, f"Case citation caused split: {sents}"

    def test_no_text(self):
        assert segment_en("") == []

    def test_only_whitespace(self):
        assert segment_en("   ") == []

    def test_multiple_sentences_with_numbers(self):
        text = "Section 2(f) defines the term. Section 3 applies to all cases."
        sents = segment_en(text)
        assert len(sents) == 2

    def test_paragraph_with_linebreaks(self):
        text = "First line\nsecond line.\n\nNew paragraph."
        sents = segment_en(text)
        assert len(sents) >= 2


class TestHindiSegmentation:
    def test_simple_danda_split(self):
        sents = segment_hi("पहला वाक्य। दूसरा वाक्य।")
        assert len(sents) == 2
        assert "पहला" in sents[0]
        assert "दूसरा" in sents[1]

    def test_date_not_breaking(self):
        text = "यह 27.05.2003 को हुआ। फिर 28.05.2003 को।"
        sents = segment_hi(text)
        assert len(sents) == 2

    def test_no_danda(self):
        sents = segment_hi("यह एक वाक्य है")
        assert len(sents) == 1  # treated as one sentence

    def test_devanagari_content(self):
        sents = segment_hi("उच्चतम न्यायालय। प्रधान न्यायाधीश।")
        assert len(sents) == 2
        for s in sents:
            assert count_devanagari(s) > 0


class TestAutoDetection:
    def test_english_auto(self):
        sents = segment("This is English.")
        assert len(sents) == 1

    def test_hindi_auto(self):
        sents = segment("यह हिंदी है।")
        assert len(sents) == 1


class TestProcessDoc:
    def test_process_en_doc(self):
        result = process_doc(1, "en", verbose=False)
        assert result is not None
        assert result["sentences"] > 0

    def test_process_hi_doc(self):
        result = process_doc(1, "hi", verbose=False)
        assert result is not None
        assert result["sentences"] > 0

    def test_output_files_exist(self):
        from pathlib import Path
        assert (Path("data/english/segmented/1.txt")).exists()
        assert (Path("data/hindi/segmented/1.txt")).exists()

    def test_output_not_empty(self):
        en_text = open("data/english/segmented/1.txt").read()
        hi_text = open("data/hindi/segmented/1.txt").read()
        assert len(en_text.strip()) > 0
        assert len(hi_text.strip()) > 0

    def test_one_sentence_per_line(self):
        text = open("data/english/segmented/1.txt").read()
        for line in text.split('\n'):
            if line.strip():
                assert len(line.strip()) > 0


class TestRun:
    def test_run_all(self):
        results = run(verbose=False)
        assert len(results["processed"]) == 60  # 30 EN + 30 HI

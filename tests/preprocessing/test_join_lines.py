from pathlib import Path
from src.preprocessing.join_lines import join_lines, should_join, process_doc, run, PROPER_NOUNS
from src.config import EN_CLEAN_DIR, EN_PREPROCESSED_DIR, DOC_IDS


class TestShouldJoin:
    def test_lowercase_to_lowercase(self):
        assert should_join("working as an officer of Regional Rural", "bank services")

    def test_lowercase_to_proper_noun(self):
        assert should_join("committed by the", "High Court")

    def test_not_join_after_period(self):
        assert not should_join("end of sentence.", "Next sentence starts")

    def test_not_join_after_question(self):
        assert not should_join("is this correct?", "No it is not")

    def test_not_join_numbered_item(self):
        assert not should_join("previous text", "2. Next item")

    def test_not_join_bullet(self):
        assert not should_join("previous text", "- bullet point")

    def test_lowercase_to_unknown_uppercase(self):
        assert not should_join("random lowercase", "Zebras are animals")

    def test_empty_lines(self):
        assert not should_join("", "some text")
        assert not should_join("some text", "")

    def test_digit_to_lowercase(self):
        assert should_join("under Section 38", "kha of the Act")

    def test_proper_noun_list_comprehensive(self):
        expected_commons = {"Court", "High", "Bank", "Appellant", "Section", "Act", "State"}
        for word in expected_commons:
            assert word in PROPER_NOUNS, f"{word} should be in proper nouns list"


class TestJoinLines:
    def test_simple_join(self):
        text = "first line\nsecond line"
        assert join_lines(text) == "first line second line"

    def test_no_join_across_blank(self):
        text = "first line\n\nsecond line"
        assert join_lines(text) == "first line\n\nsecond line"

    def test_no_join_sentence_boundary(self):
        text = "End of sentence.\nNext paragraph."
        assert join_lines(text) == "End of sentence.\nNext paragraph."

    def test_mixed_scenario(self):
        text = "This is a long\nwrapped line here\n\nNew paragraph.\nNext line here"
        result = join_lines(text)
        assert "long wrapped line here" in result
        assert "\n\n" in result

    def test_proper_noun_join(self):
        text = "submitted before the\nHigh Court of Judicature"
        result = join_lines(text)
        assert "before the High Court of Judicature" in result

    def test_numbered_items_preserved(self):
        text = "Some text.\n\n1. First item\ncontinues\n\n2. Second item\nalso continues"
        result = join_lines(text)
        assert "1. First item continues" in result
        assert "2. Second item also continues" in result
        assert result.count("\n\n") == 2

    def test_trailing_newline(self):
        text = "line one\nline two\n"
        assert join_lines(text) == "line one line two\n"


class TestProcessDoc:
    def test_process_single_doc(self):
        result = process_doc(1, verbose=False)
        assert result is not None
        assert result["before"] > result["after"]

    def test_output_file_created(self):
        assert (EN_PREPROCESSED_DIR / "1.txt").exists()

    def test_output_has_devanagari(self):
        from src.utils.validation import count_devanagari
        text = (EN_PREPROCESSED_DIR / "1.txt").read_text(encoding="utf-8")
        # English files should have 0 Devanagari characters
        assert count_devanagari(text) == 0

    def test_long_line_docs_unchanged(self):
        for doc_id in [3, 4, 5, 9, 13, 15, 16, 18, 19, 20, 22, 23, 24]:
            clean_path = EN_CLEAN_DIR / f"{doc_id}.txt"
            prep_path = EN_PREPROCESSED_DIR / f"{doc_id}.txt"
            clean_text = clean_path.read_text(encoding="utf-8")
            prep_text = prep_path.read_text(encoding="utf-8")
            assert clean_text == prep_text, f"Doc {doc_id} should be unchanged"

    def test_hard_wrapped_docs_reduced(self):
        for doc_id in [1, 2, 6, 7, 8, 10, 11, 12, 14, 17, 21, 25, 26, 27, 28, 29, 30]:
            clean_path = EN_CLEAN_DIR / f"{doc_id}.txt"
            prep_path = EN_PREPROCESSED_DIR / f"{doc_id}.txt"
            clean_lines = len([l for l in clean_path.read_text(encoding="utf-8").split('\n') if l.strip()])
            prep_lines = len([l for l in prep_path.read_text(encoding="utf-8").split('\n') if l.strip()])
            assert prep_lines < clean_lines, f"Doc {doc_id}: joined file should have fewer lines"


class TestRun:
    def test_run_all(self):
        results = run(verbose=False)
        assert len(results["processed"]) == 30

    def test_run_subset(self):
        results = run(doc_ids=[1, 2], verbose=False)
        assert len(results["processed"]) == 2

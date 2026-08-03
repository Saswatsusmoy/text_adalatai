"""Unit tests for the reference-only OCR artifact cleaner."""

from src.evaluation.ref_cleaner import clean_ref


class TestCleanRef:
    def test_danda_before_punctuation_removed(self):
        assert clean_ref('वाक्य। .') == 'वाक्य.'
        assert clean_ref('वाक्य।,अगला') == 'वाक्य,अगला'
        assert clean_ref('वाक्य॥।;') == 'वाक्य;'

    def test_terminal_danda_kept(self):
        assert clean_ref('यह एक वाक्य।') == 'यह एक वाक्य।'

    def test_hash_markers_removed(self):
        assert clean_ref('यह #. एक # वाक्य') == 'यह एक वाक्य'

    def test_leading_bare_digits_removed(self):
        assert clean_ref('1. यह वाक्य') == 'यह वाक्य'
        assert clean_ref('12। यह वाक्य') == 'यह वाक्य'
        assert clean_ref('3 यह वाक्य') == 'यह वाक्य'

    def test_inline_section_numbers_kept(self):
        assert clean_ref('धारा 227 के अनुसार निर्णय।') == 'धारा 227 के अनुसार निर्णय।'

    def test_clean_ref_never_strips_mid_string_digits(self):
        assert clean_ref('कुल 227 अभिलेख') == 'कुल 227 अभिलेख'

    def test_empty_safe(self):
        assert clean_ref('') == ''

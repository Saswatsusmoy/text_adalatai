"""Tests for Hindi hard-wrap joining (mirror of test_join_lines.py, tmp_path-only)."""

from src.preprocessing import join_hindi_lines as jh
from src.preprocessing.join_hindi_lines import join_lines, process_doc, run, should_join


# Real Tesseract OCR snippet from data/hindi/preprocessed/6.txt: case header
# block (must stay standalone) + a hard-wrapped body paragraph (must join).
REAL_OCR_DOC6 = (
    'प्रतिवेद्य\n'
    'भारतीय सर्वोच्च न्यायालय\n'
    'सिविल अपीलीय अधिकारिता\n'
    'सिविल अपील संख्या 6945 वर्ष 2024\n'
    'उ.प्र. राज्य\n'
    '... अपीलार्थी(गण)\n'
    'बनाम\n'
    'चुन्नी लाल एवं अन्य\n'
    '... प्रत्यर्थी(गण)\n'
    'निर्णय\n'
    'न्यायमूर्ति एम.आर. शाह\n'
    '. रिट याचिका सं. 448 वर्ष 4996 (एस/बी) में इलाहाबाद उच्च\n'
    'न्यायालय(लखनऊ बेंच) द्वारा पारित 46.07.2044 दिनांकित आशक्षेपित निर्णय और\n'
    'आदेश से व्यथित और असंतुष्ट महसूस करते हुए, उ.प्र. राज्य ने वर्तमान अपील की\n'
    'है।\n'
    '\n'
    '2. उ.प्र. लोक सेवा आयोग द्वारा डिप्टी कलेक्टर के 35 पदों के लिए चयन\n'
    'प्रक्रिया शुरू की गई थी। वर्ष 985 में एक संयुक्त राज्य सेवा परीक्षा आयोजित की\n'
    'गई थी।'
)

# Long-enough Devanagari lines are required for a hard wrap to be joinable.
LONG_LINE = 'भारत संघ की ओर से विद्वान अपर सॉलिसिटर जनरल सुश्री ऐश्वर्या भाटी और'


class TestShouldJoin:
    def test_join_long_wrap(self):
        assert should_join(LONG_LINE, 'प्रतिवादी की ओर से उपस्थित हुए')

    def test_not_join_after_danda(self):
        assert not should_join('वर्तमान अपील सफल होती है।', 'उपरोक्त कारणों से')

    def test_not_join_danda_with_trailing_quote(self):
        assert not should_join('अनुवादित निर्णय मान्य होगा।"', 'अगली पंक्ति')

    def test_not_join_numbered_item(self):
        assert not should_join(LONG_LINE, '2. अगला आइटम')

    def test_not_join_devanagari_numbered_item(self):
        assert not should_join(LONG_LINE, '२. अगला आइटम')

    def test_not_join_roman_numbered_item(self):
        assert not should_join(LONG_LINE, 'i. पहला आइटम')

    def test_not_join_list_marker(self):
        assert not should_join(LONG_LINE, '(क) पहला उप-खंड')

    def test_not_join_bullet(self):
        assert not should_join(LONG_LINE, '- बुलेट पॉइंट')

    def test_not_join_short_header(self):
        assert not should_join('बनाम', 'चुन्नी लाल एवं अन्य')

    def test_not_join_judge_name(self):
        assert not should_join('न्यायमूर्ति एम.आर. शाह', 'नई दिल्ली')

    def test_join_date_start_continuation(self):
        assert should_join(LONG_LINE, '09.03.2046 को उनका निधन हो गया')

    def test_join_short_fragment_with_danda(self):
        assert should_join('है। इसलिए, उच्च न्यायालय', 'से कहा गया था')

    def test_empty_lines(self):
        assert not should_join('', 'some text')
        assert not should_join('some text', '')


class TestJoinLines:
    def test_simple_wrap_join(self):
        text = LONG_LINE + '\nप्रतिवादी की ओर से उपस्थित हुए।'
        assert join_lines(text) == LONG_LINE + ' प्रतिवादी की ओर से उपस्थित हुए।'

    def test_no_join_across_blank(self):
        text = 'पहली पंक्ति जो काफी लंबी है और लिखी गई है\n\nदूसरी पंक्ति जो अलग है'
        assert join_lines(text) == text

    def test_headers_preserved(self):
        result = join_lines('बनाम\nचुन्नी लाल एवं अन्य\nनिर्णय')
        assert result.split('\n') == ['बनाम', 'चुन्नी लाल एवं अन्य', 'निर्णय']

    def test_real_ocr_doc6(self):
        result = join_lines(REAL_OCR_DOC6)
        lines = [line for line in result.split('\n') if line.strip()]
        headers = ['प्रतिवेद्य', 'बनाम', 'निर्णय', 'न्यायमूर्ति एम.आर. शाह']
        for header in headers:
            assert header in lines, f'header {header!r} should stay standalone'
        body = [line for line in lines if 'रिट याचिका' in line]
        assert len(body) == 1
        assert body[0].endswith('है।')
        assert 'एस/बी) में इलाहाबाद उच्च न्यायालय(लखनऊ बेंच) द्वारा' in body[0]

    def test_numbered_paragraph_joined(self):
        result = join_lines(REAL_OCR_DOC6)
        lines = [line for line in result.split('\n') if line.strip()]
        para = [line for line in lines if line.startswith('2. उ.प्र. लोक सेवा')]
        assert len(para) == 1
        assert para[0].endswith('गई थी।')
        assert 'चयन प्रक्रिया शुरू की गई थी।' in para[0]

    def test_idempotent(self):
        once = join_lines(REAL_OCR_DOC6)
        assert join_lines(once) == once

    def test_trailing_newline(self):
        text = LONG_LINE + '\nप्रतिवादी उपस्थित हुए।\n'
        assert join_lines(text) == LONG_LINE + ' प्रतिवादी उपस्थित हुए।\n'


class TestProcessDoc:
    def test_joins_and_preserves_headers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(jh, 'HI_PREPROCESSED_DIR', tmp_path)
        (tmp_path / '6.txt').write_text(REAL_OCR_DOC6, encoding='utf-8')
        result = process_doc(6, verbose=False)
        assert result['before'] == 18
        assert result['after'] < result['before']
        out = (tmp_path / '6.txt').read_text(encoding='utf-8')
        assert 'बनाम\n' in out
        assert 'चयन प्रक्रिया शुरू की गई थी।' in out

    def test_idempotent_across_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(jh, 'HI_PREPROCESSED_DIR', tmp_path)
        (tmp_path / '6.txt').write_text(REAL_OCR_DOC6, encoding='utf-8')
        process_doc(6, verbose=False)
        first = (tmp_path / '6.txt').read_text(encoding='utf-8')
        process_doc(6, verbose=False)
        assert (tmp_path / '6.txt').read_text(encoding='utf-8') == first

    def test_missing_doc_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(jh, 'HI_PREPROCESSED_DIR', tmp_path)
        assert process_doc(99, verbose=False) is None

    def test_devanagari_preserved(self, tmp_path, monkeypatch):
        from src.utils.validation import count_devanagari

        monkeypatch.setattr(jh, 'HI_PREPROCESSED_DIR', tmp_path)
        (tmp_path / '6.txt').write_text(REAL_OCR_DOC6, encoding='utf-8')
        before = count_devanagari(REAL_OCR_DOC6)
        process_doc(6, verbose=False)
        after = count_devanagari((tmp_path / '6.txt').read_text(encoding='utf-8'))
        assert after == before


class TestRun:
    def test_run_subset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(jh, 'HI_PREPROCESSED_DIR', tmp_path)
        (tmp_path / '6.txt').write_text(REAL_OCR_DOC6, encoding='utf-8')
        (tmp_path / '14.txt').write_text('न्यायमूर्ति पंकज मित्तल\n', encoding='utf-8')
        results = run(doc_ids=[6, 14], verbose=False)
        assert len(results['processed']) == 2
        assert (tmp_path / '6.txt').read_text(encoding='utf-8') != REAL_OCR_DOC6

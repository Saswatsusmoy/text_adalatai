from pathlib import Path

from src.config import CORRUPTED_DOC_IDS, HI_CLEAN_DIR, HI_ORIGINAL_DIR, HI_PREPROCESSED_DIR
from src.preprocessing.reextract_pdfs import (
    BACKENDS,
    compare_all_with_originals,
    compare_with_originals,
    extract_with_pdftotext,
    extract_with_tesseract,
    reextract_single,
    run,
    scan_all_hindi_pdfs,
    verify_file,
)
from src.utils.validation import count_devanagari, devanagari_ratio, has_devanagari


class TestExtraction:
    def test_tesseract_available(self):
        result = extract_with_tesseract(HI_ORIGINAL_DIR / '6.pdf')
        assert result is not None
        assert count_devanagari(result) > 0

    def test_pdftotext_available(self):
        assert Path('/opt/homebrew/bin/pdftotext').exists()

    def test_corrupted_pdfs_exist(self):
        for doc_id in CORRUPTED_DOC_IDS:
            assert (HI_ORIGINAL_DIR / f'{doc_id}.pdf').exists(), f'Doc {doc_id} PDF missing'

    def test_reextract_single_returns_devanagari(self):
        result = reextract_single(6, backend='tesseract', verbose=False)
        assert result is not None
        assert result['devanagari_chars'] > 0

    def test_reextract_all_corrupted(self):
        results = run(backend='tesseract', verbose=False)
        assert len(results['re_extracted']) == 5
        assert len(results['failed']) == 0
        for r in results['re_extracted']:
            assert r['devanagari_chars'] > 0, f'Doc {r["doc_id"]}: no Devanagari'

    def test_output_files_created(self):
        for doc_id in CORRUPTED_DOC_IDS:
            assert (HI_PREPROCESSED_DIR / f'{doc_id}.txt').exists(), f'Doc {doc_id} output missing'

    def test_compare_shows_applied(self):
        comparison = compare_with_originals(verbose=False)
        for doc_id, diff in comparison.items():
            assert diff['new_devanagari'] > 0, (
                f'Doc {doc_id}: new file has {diff["new_devanagari"]} Devanagari chars (expected >0)'
            )
            assert diff['old_devanagari'] > 0, (
                f'Doc {doc_id}: clean file should have Devanagari after apply'
            )

    def test_compare_all(self):
        results = compare_all_with_originals(verbose=False)
        assert len(results['diffs']) > 0

    def test_scan_all_pdfs(self):
        result = scan_all_hindi_pdfs(backend='pdftotext', verbose=False)
        assert result['scanned'] == 30
        corruption_issues = [
            i for i in result['issues'] if i.get('issue', '').startswith('corrupted')
        ]
        assert len(corruption_issues) == 0, (
            f'Expected 0 corruption issues after fix, got {len(corruption_issues)}'
        )


class TestValidation:
    def test_count_devanagari_empty(self):
        assert count_devanagari('') == 0

    def test_count_devanagari_ascii_only(self):
        assert count_devanagari('Hello World!') == 0

    def test_count_devanagari_hindi(self):
        assert count_devanagari('प्रतिवेद्य') > 0

    def test_has_devanagari_true(self):
        assert has_devanagari('नमस्ते')

    def test_has_devanagari_false(self):
        assert not has_devanagari('Hello')

    def test_devanagari_ratio_full(self):
        assert devanagari_ratio('प्रति') == 1.0

    def test_devanagari_ratio_mixed(self):
        r = devanagari_ratio('Doc 1: प्रतिवेद्य')
        assert 0.0 < r < 1.0

    def test_verify_file(self):
        filepath = HI_PREPROCESSED_DIR / '6.txt'
        info = verify_file(filepath, 'test')
        assert info['has_devanagari'] is True
        assert info['total_chars'] > 0


class TestApply:
    def test_apply_copies_files(self):
        from src.preprocessing.reextract_pdfs import apply

        originals = {}
        for doc_id in CORRUPTED_DOC_IDS:
            f = HI_CLEAN_DIR / f'{doc_id}.txt'
            if f.exists():
                originals[doc_id] = f.read_text(encoding='utf-8', errors='replace')

        apply(verbose=False)

        for doc_id in CORRUPTED_DOC_IDS:
            clean_file = HI_CLEAN_DIR / f'{doc_id}.txt'
            reextracted_file = HI_PREPROCESSED_DIR / f'{doc_id}.txt'
            clean_text = clean_file.read_text(encoding='utf-8')
            reextracted_text = reextracted_file.read_text(encoding='utf-8')
            assert has_devanagari(clean_text), (
                f'Doc {doc_id}: clean file still has no Devanagari after apply'
            )
            assert clean_text == reextracted_text, f'Doc {doc_id}: content mismatch after apply'

        for doc_id, content in originals.items():
            (HI_CLEAN_DIR / f'{doc_id}.txt').write_text(content, encoding='utf-8')


class TestBackends:
    def test_backends_dict(self):
        assert 'tesseract' in BACKENDS
        assert 'pdftotext' in BACKENDS

    def test_pdftotext_backend_works(self):
        result = reextract_single(6, backend='pdftotext', verbose=False)
        assert result is not None
        assert result['devanagari_chars'] > 0

    def test_tesseract_vs_pdftotext_quality(self):
        tess = extract_with_tesseract(HI_ORIGINAL_DIR / '6.pdf')
        pdf = extract_with_pdftotext(HI_ORIGINAL_DIR / '6.pdf')
        assert tess is not None and pdf is not None
        tess_dev = count_devanagari(tess)
        pdf_dev = count_devanagari(pdf)
        assert tess_dev > pdf_dev, 'Tesseract should extract more Devanagari chars than pdftotext'

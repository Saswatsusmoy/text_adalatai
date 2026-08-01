"""Regression: CORRUPTED_DOC_IDS must be real Tesseract OCR, not text-layer junk."""

from src.config import CORRUPTED_DOC_IDS, HI_PREPROCESSED_DIR
from src.preprocessing.reextract_pdfs import (
    MIN_OCR_DEV,
    TEXTLAYER_MARKERS,
    verify_ocr_quality,
)
from src.utils.validation import count_devanagari


class TestCorruptedDocsAreOcr:
    def test_preprocessed_exists(self):
        for doc_id in CORRUPTED_DOC_IDS:
            path = HI_PREPROCESSED_DIR / f'{doc_id}.txt'
            assert path.exists(), f'missing {path}'

    def test_verify_ocr_quality_clean(self):
        qc = verify_ocr_quality(verbose=False)
        assert qc['issues'] == [], f'OCR issues: {qc["issues"]}'

    def test_devanagari_above_textlayer_baseline(self):
        for doc_id in CORRUPTED_DOC_IDS:
            text = (HI_PREPROCESSED_DIR / f'{doc_id}.txt').read_text(encoding='utf-8')
            dev = count_devanagari(text)
            assert dev >= MIN_OCR_DEV[doc_id], (
                f'doc {doc_id}: Dev chars {dev} < OCR floor {MIN_OCR_DEV[doc_id]}'
            )

    def test_no_textlayer_ligature_markers_doc6(self):
        text = (HI_PREPROCESSED_DIR / '6.txt').read_text(encoding='utf-8')
        for marker in TEXTLAYER_MARKERS:
            assert marker not in text, f'doc 6 still has text-layer artifact {marker!r}'

    def test_doc6_has_correct_ligatures(self):
        text = (HI_PREPROCESSED_DIR / '6.txt').read_text(encoding='utf-8')
        assert 'भारतीय' in text
        assert 'सर्वोच्च' in text
        assert 'सिविल' in text

    def test_verify_ocr_flags_degraded(self, tmp_path, monkeypatch):
        monkeypatch.setattr('src.preprocessing.reextract_pdfs.HI_PREPROCESSED_DIR', tmp_path)
        (tmp_path / '6.txt').write_text('प्रति वेद्य\nसिसविवल अपील\n', encoding='utf-8')
        qc = verify_ocr_quality(verbose=False)
        assert any(i['doc_id'] == 6 and 'text-layer' in i['issue'] for i in qc['issues'])


class TestAlignMerge:
    def test_merge_keeps_other_docs(self, tmp_path, monkeypatch):
        import json

        from src.preprocessing import align_sentences as al

        out = tmp_path / 'all.jsonl'
        existing = [
            {
                'en_text': 'a',
                'hi_text': 'अ',
                'doc_id': 1,
                'similarity': 0.9,
                'source': 'preprocessed',
            },
            {
                'en_text': 'b',
                'hi_text': 'ब',
                'doc_id': 6,
                'similarity': 0.6,
                'source': 'preprocessed',
            },
            {
                'en_text': 'c',
                'hi_text': 'स',
                'doc_id': 2,
                'similarity': 0.8,
                'source': 'preprocessed',
            },
        ]
        out.write_text(
            '\n'.join(json.dumps(p, ensure_ascii=False) for p in existing) + '\n',
            encoding='utf-8',
        )
        monkeypatch.setattr(al, 'OUTPUT_DIR', tmp_path)

        def fake_process(doc_id, verbose=False):
            return [
                {
                    'en_text': f'new{doc_id}',
                    'hi_text': f'न{doc_id}',
                    'doc_id': doc_id,
                    'similarity': 0.95,
                    'source': 'preprocessed',
                }
            ]

        monkeypatch.setattr(al, 'process_doc', fake_process)
        result = al.run(doc_ids=[6], verbose=False, merge=True)
        assert result['merged'] is True
        pairs = al._load_all_pairs(out)
        by_doc = {p['doc_id']: p for p in pairs}
        assert set(by_doc) == {1, 2, 6}
        assert by_doc[6]['en_text'] == 'new6'
        assert by_doc[1]['en_text'] == 'a'

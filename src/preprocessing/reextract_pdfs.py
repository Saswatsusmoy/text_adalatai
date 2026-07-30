"""
Re-extract Hindi text from PDFs using Tesseract OCR (default) or pdftotext.

5 Hindi clean files (Docs 6, 14, 22, 25, 26) were corrupted - every Devanagari
character replaced with '?'. This script re-extracts text from the original PDFs,
with Tesseract OCR giving better Devanagari ligature handling than text-layer extractors.

Output: data/hindi/preprocessed/{doc_id}.txt
"""

import subprocess
import tempfile
from pathlib import Path

import fitz

from src.config import (
    CORRUPTED_DOC_IDS,
    DOC_IDS,
    HI_CLEAN_DIR,
    HI_ORIGINAL_DIR,
    HI_PREPROCESSED_DIR,
    PDFTOTEXT_CMD,
)
from src.utils.validation import count_devanagari


def extract_with_pdftotext(pdf_path: Path) -> str | None:
    if not pdf_path.exists():
        return None
    result = subprocess.run(
        [PDFTOTEXT_CMD, str(pdf_path), '-'],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def extract_with_tesseract(pdf_path: Path, dpi: int = 300) -> str | None:
    if not pdf_path.exists():
        return None
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return None

    pages_text = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            pix.save(tmp.name)
            tmp_path = tmp.name
        result = subprocess.run(
            ['tesseract', tmp_path, 'stdout', '-l', 'hin', '--psm', '6'],
            capture_output=True,
            text=True,
            timeout=120,
        )
        Path(tmp_path).unlink(missing_ok=True)
        if result.returncode == 0:
            pages_text.append(result.stdout)

    doc.close()
    if not pages_text:
        return None
    return '\n'.join(pages_text)


def save_text(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding='utf-8')


def verify_file(filepath: Path, label: str) -> dict:
    text = filepath.read_text(encoding='utf-8', errors='replace')
    dev_count = count_devanagari(text)
    return {
        'file': str(filepath),
        'label': label,
        'exists': filepath.exists(),
        'size_bytes': filepath.stat().st_size,
        'total_chars': len(text),
        'devanagari_chars': dev_count,
        'has_devanagari': dev_count > 0,
    }


BACKENDS = {
    'pdftotext': extract_with_pdftotext,
    'tesseract': extract_with_tesseract,
}


def reextract_single(doc_id: int, backend: str = 'tesseract', verbose: bool = True) -> dict | None:
    pdf_path = HI_ORIGINAL_DIR / f'{doc_id}.pdf'
    out_path = HI_PREPROCESSED_DIR / f'{doc_id}.txt'

    extract_fn = BACKENDS.get(backend)
    if extract_fn is None:
        if verbose:
            print(f"  [FAIL] Doc {doc_id}: unknown backend '{backend}'")
        return None

    text = extract_fn(pdf_path)
    if text is None:
        if verbose:
            print(f'  [FAIL] Doc {doc_id}: {backend} failed')
        return None

    save_text(text, out_path)
    dev_count = count_devanagari(text)
    if verbose:
        status = 'OK' if dev_count > 0 else 'WARN (no Devanagari)'
        print(
            f'  [ {status:20s}] Doc {doc_id:2d}: {dev_count:5d} Devanagari chars, {len(text):6d} total chars'
        )
    return {
        'doc_id': doc_id,
        'devanagari_chars': dev_count,
        'total_chars': len(text),
        'output': str(out_path),
    }


def run(doc_ids: list[int] | None = None, backend: str = 'tesseract', verbose: bool = True) -> dict:
    if doc_ids is None:
        doc_ids = CORRUPTED_DOC_IDS

    results = {'re_extracted': [], 'failed': []}

    if verbose:
        print(f'Re-extracting {len(doc_ids)} Hindi PDFs via {backend}...')
        print()

    for doc_id in doc_ids:
        result = reextract_single(doc_id, backend=backend, verbose=verbose)
        if result:
            results['re_extracted'].append(result)
        else:
            results['failed'].append(doc_id)

    return results


def compare_with_originals(verbose: bool = True) -> dict:
    comparison = {}
    for doc_id in CORRUPTED_DOC_IDS:
        old_file = HI_CLEAN_DIR / f'{doc_id}.txt'
        new_file = HI_PREPROCESSED_DIR / f'{doc_id}.txt'

        old_stats = verify_file(old_file, 'old (corrupted)') if old_file.exists() else None
        new_stats = verify_file(new_file, 'new (re-extracted)') if new_file.exists() else None

        if old_stats and new_stats:
            diff = {
                'doc_id': doc_id,
                'old_size': old_stats['size_bytes'],
                'new_size': new_stats['size_bytes'],
                'old_devanagari': old_stats['devanagari_chars'],
                'new_devanagari': new_stats['devanagari_chars'],
                'recovered_chars': new_stats['devanagari_chars'] - old_stats['devanagari_chars'],
            }
            comparison[doc_id] = diff
            if verbose:
                print(
                    f'  Doc {doc_id:2d}: old={old_stats["size_bytes"]:>6}B dev={old_stats["devanagari_chars"]:>4d}  '
                    f'new={new_stats["size_bytes"]:>6}B dev={new_stats["devanagari_chars"]:>4d}  '
                    f'recovered={diff["recovered_chars"]:>5d} Devanagari chars'
                )
        elif old_stats and not new_stats:
            if verbose:
                print(f'  Doc {doc_id:2d}: old exists but re-extracted file missing')
        elif not old_stats and new_stats:
            if verbose:
                print(f'  Doc {doc_id:2d}: no original corrupted file, only re-extracted file')
    return comparison


def compare_all_with_originals(verbose: bool = True) -> dict:
    results = {'diffs': [], 'issues': []}
    for doc_id in DOC_IDS:
        clean_file = HI_CLEAN_DIR / f'{doc_id}.txt'
        new_file = HI_PREPROCESSED_DIR / f'{doc_id}.txt'
        if not new_file.exists():
            continue

        new_text = new_file.read_text(encoding='utf-8')
        new_dev = count_devanagari(new_text)

        if clean_file.exists():
            old_text = clean_file.read_text(encoding='utf-8', errors='replace')
            old_dev = count_devanagari(old_text)
            diff = new_dev - old_dev
            results['diffs'].append(
                {'doc_id': doc_id, 'old_dev': old_dev, 'new_dev': new_dev, 'diff': diff}
            )
            if verbose:
                label = (
                    'RECOVERED'
                    if old_dev == 0 and new_dev > 0
                    else 'MATCH'
                    if abs(diff) < 500
                    else 'DIFF'
                )
                print(
                    f'  [{label:10s}] Doc {doc_id:2d}: clean={old_dev:5d} dev -> OCR={new_dev:5d} dev (diff={diff:+5d})'
                )
        else:
            results['issues'].append({'doc_id': doc_id, 'issue': 'no clean file'})
            if verbose:
                print(f'  [NO CLEAN ] Doc {doc_id:2d}: no clean file to compare')
    return results


def scan_all_hindi_pdfs(backend: str = 'tesseract', verbose: bool = True) -> dict:
    issues = []
    extract_fn = BACKENDS.get(backend, extract_with_tesseract)
    for doc_id in DOC_IDS:
        pdf_path = HI_ORIGINAL_DIR / f'{doc_id}.pdf'
        txt_path = HI_CLEAN_DIR / f'{doc_id}.txt'

        if not pdf_path.exists():
            continue

        text = extract_fn(pdf_path)
        if text is None:
            issues.append({'doc_id': doc_id, 'issue': f'{backend} failed'})
            if verbose:
                print(f'  [EXTRACT FAIL] Doc {doc_id}')
            continue

        dev_count = count_devanagari(text)
        if dev_count == 0:
            issues.append(
                {
                    'doc_id': doc_id,
                    'issue': 'no Devanagari in extraction',
                    'text_preview': text[:100],
                }
            )
            if verbose:
                print(f'  [NO DEV] Doc {doc_id}: extracted text has 0 Devanagari chars')
        elif txt_path.exists():
            old_text = txt_path.read_text(encoding='utf-8', errors='replace')
            old_dev = count_devanagari(old_text)
            if old_dev == 0 and dev_count > 0:
                issues.append({'doc_id': doc_id, 'issue': 'corrupted clean file, PDF has content'})
                if verbose:
                    print(
                        f'  [CORRUPTED] Doc {doc_id}: clean file has {old_dev} Devanagari, PDF has {dev_count}'
                    )
        else:
            if verbose:
                print(f'  [OK] Doc {doc_id}: {dev_count} Devanagari chars, clean file exists')
    return {'scanned': len(DOC_IDS), 'issues': issues}


def apply(doc_ids: list[int] | None = None, verbose: bool = True) -> dict:
    if doc_ids is None:
        doc_ids = CORRUPTED_DOC_IDS
    copied = []
    missing = []
    for doc_id in doc_ids:
        src = HI_PREPROCESSED_DIR / f'{doc_id}.txt'
        dst = HI_CLEAN_DIR / f'{doc_id}.txt'
        if not src.exists():
            missing.append(doc_id)
            if verbose:
                print(f'  [SKIP] Doc {doc_id}: re-extracted file not found at {src}')
            continue
        dst.write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
        copied.append(doc_id)
        if verbose:
            print(f'  [APPLIED] Doc {doc_id}: {src} -> {dst}')
    return {'copied': copied, 'missing': missing}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Re-extract Hindi text from PDFs using OCR or text-layer extraction',
    )
    parser.add_argument(
        '--doc-ids',
        type=int,
        nargs='+',
        default=None,
        help='Document IDs to process (default: corrupted docs 6 14 22 25 26, or all with --all)',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all 30 Hindi PDFs',
    )
    parser.add_argument(
        '--backend',
        choices=list(BACKENDS.keys()),
        default='tesseract',
        help='Extraction backend (default: tesseract)',
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare re-extracted files with originals',
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare all re-extracted files with clean originals',
    )
    parser.add_argument(
        '--scan',
        action='store_true',
        help='Scan all 30 Hindi PDFs for extraction quality',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Copy re-extracted files to clean/ directory',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output',
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if args.scan:
        if verbose:
            print(f'Scanning all 30 Hindi PDFs via {args.backend}...\n')
        scan_all_hindi_pdfs(backend=args.backend, verbose=verbose)
        return

    if args.apply:
        doc_ids = args.doc_ids if args.doc_ids else CORRUPTED_DOC_IDS
        if verbose:
            print(f'Applying re-extracted files to clean/ (docs {doc_ids})...\n')
        apply(doc_ids=doc_ids, verbose=verbose)
        return

    doc_ids = args.doc_ids
    if args.all:
        doc_ids = DOC_IDS

    run(doc_ids, backend=args.backend, verbose=verbose)

    if args.compare:
        print()
        print('Comparing re-extracted vs corrupted originals...\n')
        compare_with_originals(verbose=verbose)

    if args.compare_all:
        print()
        print('Comparing all re-extracted vs clean originals...\n')
        compare_all_with_originals(verbose=verbose)

    print()
    reextracted = list(HI_PREPROCESSED_DIR.glob('*.txt'))
    print(f'Done. {len(reextracted)} file(s) in {HI_PREPROCESSED_DIR}')


if __name__ == '__main__':
    main()

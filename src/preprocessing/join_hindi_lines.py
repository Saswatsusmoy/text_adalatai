"""
Join hard-wrapped lines in Hindi OCR text.

Tesseract OCR hard-wraps Hindi mid-sentence at PDF line boundaries, so a
logical sentence often spans 2+ OCR lines and only the last carries a danda
(।). Unjoined, segment_sentences.py routes every danda-less line to the
English spaCy path and the resulting fragments survive into alignment as
truncated sentences (55% of aligned HI texts lacked a danda before this step).

This mirrors join_lines.py for English PDF wraps, adapted to the danda as the
sentence boundary.

Output: data/hindi/preprocessed/{doc_id}.txt (joined in place)
"""

import re

from src.config import DOC_IDS, HI_PREPROCESSED_DIR


# Next line starts a numbered item / bullet / list marker: a new unit that must
# not absorb the tail of the previous sentence. Covers Arabic and Devanagari
# digits, short roman numerals, Hindi list markers (क)/(॥), and bullets.
# Dates (DD.MM.YYYY) also start with digits+dots but are mid-sentence
# continuations, so they are excluded from the numbered-item rule.
_NUMBERED_ITEM_RE = re.compile(
    r'^\d{1,3}\s*[\.\)]'
    r'|[०-९]{1,3}\s*[\.\)]'
    r'|^[ivxlcdmIVXLCDM]{1,4}\s*[\.\)]'
    r'|^\([क-घ]\)'
    r'|^\([॥।]\)'
    r'|^[•▪◦\-*]'
)
_DATE_RE = re.compile(r'^\d{1,2}\.\d{1,2}\.\d{2,4}|^[०-९]{1,2}\.[०-९]{1,2}\.[०-९]{2,4}')

# Trailing punctuation that may follow a danda (e.g. closing quotes `।"`) is
# stripped so such lines still count as sentence ends.
_TRAILING_PUNCT_RE = re.compile(r'[\s.,;:!?"\'\)\]]+$')

# Danda-less lines shorter than this are case headers / judge names / section
# labels (बनाम, निर्णय, court names) and stay standalone. Genuine mid-sentence
# OCR wraps are almost always longer (median ~71 chars across all 30 docs).
MAX_HEADER_LEN = 40

# Short danda-less lines that are standalone headers, not sentence continuations.
# Derived by scanning all 30 preprocessed files: the frequent short lines are case
# headers, court names, and section labels; the rare short continuations (`के साथ`,
# `प्रथम तल`) start with a postposition and must still join.
_HEADER_WORDS = {
    'बनाम',
    'निर्णय',
    'उद्घोषणा',
    'उद्घोष्णा',
    'उद्घघोषणा',
    'अस्वीकरण',
    'प्रतिवेद्य',
    'अप्रतिवेद्य',
    'हेडनोट',
    'हेडनोट्स',
    'आदेश',
    'कोरम',
    'प्रस्तुतियाँ',
    'नई दिल्ली',
}


def _ends_sentence(line: str) -> bool:
    stripped = _TRAILING_PUNCT_RE.sub('', line.rstrip())
    return stripped.endswith(('।', '॥'))


def _is_header_like(line: str) -> bool:
    if len(line) > MAX_HEADER_LEN:
        return False
    if '।' in line or '॥' in line:
        return False
    stripped = line.strip().rstrip('.,;:')
    if stripped in _HEADER_WORDS:
        return True
    if stripped.startswith('न्यायमूर्ति'):
        return True
    if stripped.startswith('(न्यायमूर्ति'):
        return True
    if stripped.endswith((':', ';')):
        return True
    return False


def should_join(prev_line: str, next_line: str) -> bool:
    prev = prev_line.rstrip()
    nxt = next_line.strip()

    if not prev or not nxt:
        return False

    if _NUMBERED_ITEM_RE.match(nxt) and not _DATE_RE.match(nxt):
        return False

    if _ends_sentence(prev):
        return False

    if _is_header_like(prev) or _is_header_like(nxt):
        return False

    return True


def join_lines(text: str) -> str:
    lines = text.split('\n')
    result = []
    buffer = []

    for line in lines:
        stripped = line.rstrip()

        if not stripped:
            if buffer:
                result.append(' '.join(buffer))
                buffer = []
            result.append('')
            continue

        if buffer and should_join(buffer[-1], line):
            buffer.append(stripped)
        else:
            if buffer:
                result.append(' '.join(buffer))
            buffer = [stripped]

    if buffer:
        result.append(' '.join(buffer))

    return '\n'.join(result)


def process_doc(doc_id: int, verbose: bool = False) -> dict | None:
    path = HI_PREPROCESSED_DIR / f'{doc_id}.txt'

    if not path.exists():
        return None

    text = path.read_text(encoding='utf-8')
    joined = join_lines(text)
    path.write_text(joined, encoding='utf-8')

    orig_lines = len([line for line in text.split('\n') if line.strip()])
    new_lines = len([line for line in joined.split('\n') if line.strip()])

    if verbose:
        print(
            f'  Doc {doc_id:2d}: {orig_lines:4d} -> {new_lines:4d} lines (Δ={orig_lines - new_lines:4d})'
        )

    return {'doc_id': doc_id, 'before': orig_lines, 'after': new_lines}


def run(doc_ids: list[int] | None = None, verbose: bool = True) -> dict:
    if doc_ids is None:
        doc_ids = DOC_IDS

    results = []
    for doc_id in doc_ids:
        r = process_doc(doc_id, verbose=verbose)
        if r:
            results.append(r)

    if verbose:
        total_before = sum(r['before'] for r in results)
        total_after = sum(r['after'] for r in results)
        print(f'\nTotal: {total_before} -> {total_after} lines (Δ={total_before - total_after})')

    return {'processed': results}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Join hard-wrapped lines in Hindi OCR text',
    )
    parser.add_argument(
        '--doc-ids',
        type=int,
        nargs='+',
        default=None,
        help='Document IDs to process (default: all 30)',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output',
    )
    args = parser.parse_args()

    run(args.doc_ids, verbose=not args.quiet)


if __name__ == '__main__':
    main()

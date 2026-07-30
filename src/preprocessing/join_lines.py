"""
Join hard-wrapped lines in English legal text.

17/30 English docs have mid-sentence line breaks from PDF extraction
(avg 44-74 chars/line). Joins lines that belong to the same sentence
based on casing heuristics and a legal-domain proper noun list.

Output: data/english/preprocessed/{doc_id}.txt
"""

import re

from src.config import DOC_IDS, EN_CLEAN_DIR, EN_PREPROCESSED_DIR


# Data-derived proper nouns found by scanning all 30 English docs for words
# that start a continuation line after a lowercase-ending line (hard wrap).
# Only includes words with >= 2 occurrences in the corpus. See DESIGN_DECISIONS.md.
PROPER_NOUNS = {
    'Act',
    'Allahabad',
    'Appellant',
    'Appellate',
    'Authority',
    'Bank',
    'Civil',
    'Commission',
    'Consolidation',
    'Constitution',
    'Court',
    'Development',
    'Directors',
    'Disciplinary',
    'District',
    'Government',
    'High',
    'Land',
    'Malkhan',
    'No.2',
    'Officer',
    'Pandey',
    'Part',
    'Petition',
    'Rajbali',
    'Reference',
    'Respondent',
    'Respondents',
    'Revenue',
    'Rules',
    'Section',
    'Sections',
    'Service',
    'State',
}


def should_join(prev_line: str, next_line: str) -> bool:
    prev = prev_line.rstrip()
    nxt = next_line.strip()

    if not prev or not nxt:
        return False

    # Never join across numbered items or bullet points
    if re.match(r'^\d+[\.\)]', nxt) or re.match(r'^[\-\*•·]', nxt):
        return False

    # Never join if prev ends with sentence-ending punctuation
    if prev[-1] in '.!?':
        return False

    # Join if prev ends with lowercase AND next starts with lowercase
    if prev[-1].islower() and nxt[0].islower():
        return True

    # Join if prev ends with lowercase AND next starts with a known proper noun
    if prev[-1].islower():
        first_word = nxt.split()[0].rstrip('.,;:') if nxt.split() else ''
        if first_word in PROPER_NOUNS:
            return True

    # Join if prev ends with a digit (e.g. "Regulation No.38") and next starts with lowercase
    if prev[-1].isdigit() and nxt[0].islower():
        return True

    return False


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
    src = EN_CLEAN_DIR / f'{doc_id}.txt'
    dst = EN_PREPROCESSED_DIR / f'{doc_id}.txt'

    if not src.exists():
        return None

    text = src.read_text(encoding='utf-8')
    joined = join_lines(text)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(joined, encoding='utf-8')

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
        description='Join hard-wrapped lines in English legal text',
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

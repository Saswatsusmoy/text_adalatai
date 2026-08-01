"""
Segment paragraphs into sentences for English and Hindi.

English: spaCy en_core_web_sm (dependency parser handles standard
         abbreviations: Mr., No., v., U.P., Cr.P.C., etc.)
         with pre-tokenization protection for Hindi honorifics (Smt., Shri.)
Hindi:   Split on । (danda)

Output: data/{lang}/segmented/{doc_id}.txt (one sentence per line)
"""

from pathlib import Path

import spacy

from src.config import (
    DOC_IDS,
    EN_PREPROCESSED_DIR,
    HI_PREPROCESSED_DIR,
)


OUTPUT_DIRS = {
    'en': Path('data/english/segmented'),
    'hi': Path('data/hindi/segmented'),
}

# Lazy-loaded spaCy English model (dependency parser handles standard
# abbreviations like Mr., No., v., U.P., etc. out of the box)
_en_nlp = None


def _get_en_nlp():
    global _en_nlp
    if _en_nlp is None:
        _en_nlp = spacy.load('en_core_web_sm')
    return _en_nlp


def has_danda(text: str) -> bool:
    return '।' in text


# Abbreviations the spaCy model might not handle natively.
# Protected by replacing period with a sentinel before tokenization.
_PROTECTED_ABBREVS = {
    'Smt.',
    'Shri.',
    'Sri.',
}


def _protect_abbrevs(text: str) -> str:
    for abbr in _PROTECTED_ABBREVS:
        text = text.replace(abbr, abbr.replace('.', '<DOT>'))
    return text


def _restore_abbrevs(text: str) -> str:
    return text.replace('<DOT>', '.')


def segment_en(text: str) -> list[str]:
    protected = _protect_abbrevs(text)
    nlp = _get_en_nlp()
    doc = nlp(protected)

    sents = []
    for sent in doc.sents:
        chunk = sent.text.strip()
        if not chunk:
            continue
        chunk = _restore_abbrevs(chunk)
        sents.append(chunk)

    return sents


def segment_hi(text: str) -> list[str]:
    raw = text.split('।')
    sentences = []
    for chunk in raw:
        chunk = chunk.strip()
        if not chunk:
            continue
        sentences.append(chunk + '।')
    return sentences


def segment(text: str) -> list[str]:
    if has_danda(text):
        return segment_hi(text)
    return segment_en(text)


def process_doc(doc_id: int, lang: str, verbose: bool = False) -> dict | None:
    if lang == 'en':
        src_dir = EN_PREPROCESSED_DIR
    else:
        src_dir = HI_PREPROCESSED_DIR

    out_dir = OUTPUT_DIRS[lang]
    src = src_dir / f'{doc_id}.txt'
    dst = out_dir / f'{doc_id}.txt'

    if not src.exists():
        return None

    text = src.read_text(encoding='utf-8')
    paragraphs = text.split('\n')

    all_sentences = []
    for para in paragraphs:
        if not para.strip():
            all_sentences.append('')
            continue
        sents = segment(para.strip())
        all_sentences.extend(sents)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text('\n'.join(all_sentences), encoding='utf-8')

    sent_count = len([s for s in all_sentences if s.strip()])

    if verbose:
        print(f'  [{lang.upper()}] Doc {doc_id:2d}: {sent_count:4d} sentences -> {dst}')

    return {'doc_id': doc_id, 'lang': lang, 'sentences': sent_count}


def run(
    doc_ids: list[int] | None = None,
    verbose: bool = True,
    langs: list[str] | None = None,
) -> dict:
    if doc_ids is None:
        doc_ids = DOC_IDS
    if langs is None:
        langs = ['en', 'hi']

    results = []
    for lang in langs:
        for doc_id in doc_ids:
            r = process_doc(doc_id, lang, verbose=verbose)
            if r:
                results.append(r)

    if verbose:
        en_count = sum(r['sentences'] for r in results if r['lang'] == 'en')
        hi_count = sum(r['sentences'] for r in results if r['lang'] == 'hi')
        print(f'\nTotal: EN={en_count} sentences, HI={hi_count} sentences')

    return {'processed': results}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Segment paragraphs into sentences (EN + HI)',
    )
    parser.add_argument(
        '--doc-ids',
        type=int,
        nargs='+',
        default=None,
        help='Document IDs to process (default: all 30)',
    )
    parser.add_argument(
        '--lang',
        choices=['en', 'hi', 'both'],
        default='both',
        help='Language to process (default: both)',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output',
    )
    args = parser.parse_args()

    langs = ['en', 'hi'] if args.lang == 'both' else [args.lang]
    run(args.doc_ids, verbose=not args.quiet, langs=langs)


if __name__ == '__main__':
    main()

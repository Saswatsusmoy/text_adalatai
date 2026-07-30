"""
Ingest T0 external legal EN-HI parallel corpora into project JSONL format.

Sources:
  - MILPaC (Law-AI): high-quality legal EN->Indic xlsx
  - Anuvaad legal EN-HI: judiciary / HC-SUVAS / law commission / terms / etc.

These corpora are already sentence-aligned. We do NOT re-run PDF OCR, join,
segment, or LaBSE. We apply the same length / emptiness filters used after
alignment in align_sentences.py, plus exact pair dedup.

Raw downloads: data/external/raw/
Outputs:       data/external/parallel/*.jsonl
"""

import json
import zipfile
from pathlib import Path

import pandas as pd

from src.preprocessing.align_sentences import MAX_CHAR_RATIO, MIN_CHAR_RATIO


RAW_DIR = Path('data/external/raw')
MILPAC_DIR = RAW_DIR / 'milpac'
ANUVAAD_DIR = RAW_DIR / 'anuvaad'
OUT_DIR = Path('data/external/parallel')

MIN_TEXT_LEN = 3

MILPAC_URLS = {
    'MILPaC_IP_dataset.xlsx': (
        'https://raw.githubusercontent.com/Law-AI/MILPaC/main/Data/MILPaC/MILPaC_IP_dataset.xlsx'
    ),
    'MILPaC_CCI_FAQ_dataset.xlsx': (
        'https://raw.githubusercontent.com/Law-AI/MILPaC/main/Data/MILPaC/'
        'MILPaC_CCI_FAQ_dataset.xlsx'
    ),
    'MILPaC_Acts_dataset.xlsx': (
        'https://raw.githubusercontent.com/Law-AI/MILPaC/main/Data/MILPaC/MILPaC_Acts_dataset.xlsx'
    ),
}

ANUVAAD_URLS = {
    'ik-2021-v1-en-hi.zip': (
        'https://anuvaad-parallel-corpus.s3-us-west-2.amazonaws.com/ik-2021-v1-en-hi.zip'
    ),
    'internal-judicial-2021-v1-en-hi.zip': (
        'https://anuvaad-parallel-corpus.s3-us-west-2.amazonaws.com/'
        'internal-judicial-2021-v1-en-hi.zip'
    ),
    'legal-terms-2021-v1-en-hi.zip': (
        'https://anuvaad-parallel-corpus.s3-us-west-2.amazonaws.com/legal-terms-2021-v1-en-hi.zip'
    ),
}

# (zip_name, en_member_suffix, hi_member_suffix, source_tag)
ANUVAAD_MEMBERS = [
    (
        'ik-2021-v1-en-hi.zip',
        'indian_judiciary/ij.en',
        'indian_judiciary/ij.hi',
        'anuvaad_judiciary',
    ),
    (
        'ik-2021-v1-en-hi.zip',
        'law_commission/lc.en',
        'law_commission/lc.hi',
        'anuvaad_law_commission',
    ),
    (
        'ik-2021-v1-en-hi.zip',
        'names_dictionary/nd.en',
        'names_dictionary/nd.hi',
        'anuvaad_names_dict',
    ),
    (
        'ik-2021-v1-en-hi.zip',
        'augmented_corpus/ac.en',
        'augmented_corpus/ac.hi',
        'anuvaad_augmented',
    ),
    (
        'internal-judicial-2021-v1-en-hi.zip',
        'en-hi/ij-train.en',
        'en-hi/ij-train.hi',
        'anuvaad_hc_suvas',
    ),
    (
        'legal-terms-2021-v1-en-hi.zip',
        'en-hi/legal-terms.en',
        'en-hi/legal-terms.hi',
        'anuvaad_legal_terms',
    ),
]


def passes_length_filter(en_text: str, hi_text: str) -> bool:
    en = en_text.strip()
    hi = hi_text.strip()
    if not en or not hi:
        return False
    if len(en) < MIN_TEXT_LEN or len(hi) < MIN_TEXT_LEN:
        return False
    ratio = len(en) / max(len(hi), 1)
    if ratio < MIN_CHAR_RATIO or ratio > MAX_CHAR_RATIO:
        return False
    return True


def make_pair(en_text: str, hi_text: str, source: str, doc_id: str = '') -> dict:
    return {
        'en_text': en_text.strip(),
        'hi_text': hi_text.strip(),
        'doc_id': doc_id,
        'source': source,
        'similarity': None,
    }


def filter_and_dedup(pairs: list[dict]) -> tuple[list[dict], dict]:
    kept = []
    seen: set[tuple[str, str]] = set()
    stats = {
        'input': len(pairs),
        'dropped_length': 0,
        'dropped_dup': 0,
        'kept': 0,
    }
    for p in pairs:
        if not passes_length_filter(p['en_text'], p['hi_text']):
            stats['dropped_length'] += 1
            continue
        key = (p['en_text'], p['hi_text'])
        if key in seen:
            stats['dropped_dup'] += 1
            continue
        seen.add(key)
        kept.append(p)
    stats['kept'] = len(kept)
    return kept, stats


def download_file(url: str, dest: Path, verbose: bool = True) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if verbose:
            print(f'  skip (exists): {dest}')
        return dest
    if verbose:
        print(f'  downloading {url} -> {dest}')
    urllib.request.urlretrieve(url, dest)
    return dest


def download_all(verbose: bool = True) -> dict:
    results = {'milpac': [], 'anuvaad': []}
    for name, url in MILPAC_URLS.items():
        results['milpac'].append(str(download_file(url, MILPAC_DIR / name, verbose)))
    for name, url in ANUVAAD_URLS.items():
        results['anuvaad'].append(str(download_file(url, ANUVAAD_DIR / name, verbose)))
    return results


def load_milpac_en_hi(milpac_dir: Path = MILPAC_DIR) -> list[dict]:
    pairs = []
    for path in sorted(milpac_dir.glob('*.xlsx')):
        df = pd.read_excel(path)
        for col in ('src_lang', 'tgt_lang', 'src', 'tgt'):
            if col not in df.columns:
                raise ValueError(f'{path.name} missing column {col}')
        for _, row in df.iterrows():
            src_lang = str(row['src_lang']).strip().upper()
            tgt_lang = str(row['tgt_lang']).strip().upper()
            src = row['src']
            tgt = row['tgt']
            if not isinstance(src, str) or not isinstance(tgt, str):
                continue
            if src_lang == 'EN' and tgt_lang == 'HI':
                en, hi = src, tgt
            elif src_lang == 'HI' and tgt_lang == 'EN':
                en, hi = tgt, src
            else:
                continue
            dataset = str(row.get('dataset', path.stem))
            unit_id = str(row.get('id', ''))
            pairs.append(
                make_pair(
                    en,
                    hi,
                    source=f'milpac_{dataset}',
                    doc_id=f'{dataset}:{unit_id}',
                )
            )
    return pairs


def _find_zip_member(z: zipfile.ZipFile, suffix: str) -> str | None:
    for name in z.namelist():
        if name.startswith('__MACOSX'):
            continue
        if name.endswith(suffix) or name.endswith('/' + suffix):
            return name
        # allow suffix match on path end
        if name.replace('\\', '/').endswith(suffix):
            return name
    return None


def load_parallel_from_zip(
    zip_path: Path,
    en_suffix: str,
    hi_suffix: str,
    source: str,
) -> list[dict]:
    pairs = []
    with zipfile.ZipFile(zip_path) as z:
        en_name = _find_zip_member(z, en_suffix)
        hi_name = _find_zip_member(z, hi_suffix)
        if en_name is None or hi_name is None:
            raise FileNotFoundError(
                f'{zip_path.name}: missing members {en_suffix} / {hi_suffix} '
                f'(found en={en_name}, hi={hi_name})'
            )
        with z.open(en_name) as en_f, z.open(hi_name) as hi_f:
            for i, (en_line, hi_line) in enumerate(zip(en_f, hi_f)):
                en = en_line.decode('utf-8', errors='replace').rstrip('\n\r')
                hi = hi_line.decode('utf-8', errors='replace').rstrip('\n\r')
                pairs.append(make_pair(en, hi, source=source, doc_id=f'{source}:{i}'))
    return pairs


def load_anuvaad_all(anuvaad_dir: Path = ANUVAAD_DIR) -> dict[str, list[dict]]:
    by_source: dict[str, list[dict]] = {}
    for zip_name, en_suf, hi_suf, source in ANUVAAD_MEMBERS:
        zip_path = anuvaad_dir / zip_name
        if not zip_path.exists():
            continue
        by_source[source] = load_parallel_from_zip(zip_path, en_suf, hi_suf, source)
    return by_source


def write_jsonl(pairs: list[dict], path: Path):
    from src.utils.jsonl import write_jsonl as _write

    _write(path, pairs)


def run(download: bool = False, verbose: bool = True) -> dict:
    if download:
        if verbose:
            print('Downloading T0 corpora...')
        download_all(verbose=verbose)

    report: dict = {'sources': {}, 'totals': {}}
    all_kept: list[dict] = []

    # MILPaC
    if MILPAC_DIR.exists() and any(MILPAC_DIR.glob('*.xlsx')):
        raw = load_milpac_en_hi(MILPAC_DIR)
        kept, stats = filter_and_dedup(raw)
        out = OUT_DIR / 'milpac_en_hi.jsonl'
        write_jsonl(kept, out)
        report['sources']['milpac'] = {**stats, 'output': str(out)}
        all_kept.extend(kept)
        if verbose:
            print(f'MILPaC EN-HI: {stats}')
    elif verbose:
        print('MILPaC raw not found; skip (run with --download)')

    # Anuvaad subsets
    anuvaad = load_anuvaad_all(ANUVAAD_DIR)
    for source, raw in anuvaad.items():
        kept, stats = filter_and_dedup(raw)
        out = OUT_DIR / f'{source}.jsonl'
        write_jsonl(kept, out)
        report['sources'][source] = {**stats, 'output': str(out)}
        all_kept.extend(kept)
        if verbose:
            print(f'{source}: {stats}')

    # Combined Stage A file (exact-dedup again across sources)
    combined, cstats = filter_and_dedup(all_kept)
    # filter_and_dedup on already-filtered still drops cross-source dups
    comb_out = OUT_DIR / 'stage_a_en_hi.jsonl'
    write_jsonl(combined, comb_out)
    report['totals'] = {
        **cstats,
        'output': str(comb_out),
        'note': (
            'Already-aligned external pairs; length filter matches '
            f'align_sentences ({MIN_CHAR_RATIO}-{MAX_CHAR_RATIO}); '
            'no LaBSE re-score (cost); use for Stage A train only'
        ),
    }
    report_path = OUT_DIR / 'ingest_report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    if verbose:
        print(f'Combined Stage A: {cstats}')
        print(f'Report: {report_path}')

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Ingest T0 external legal EN-HI corpora')
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download raw MILPaC + Anuvaad files if missing',
    )
    args = parser.parse_args()
    run(download=args.download, verbose=True)


if __name__ == '__main__':
    main()

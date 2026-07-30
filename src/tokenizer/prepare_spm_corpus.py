"""
Build SentencePiece training text for Track C0 (legal SPM v2).

Sources (allowed):
  - Stage A external pairs: data/external/parallel/stage_a_en_hi.jsonl
  - Assignment train only: data/processed/train.jsonl
  - Optional: Prarabdha mono HI (fraction of lines)

Forbidden:
  - assignment dev/test pairs (docs 8,9,24 and 1,4,21)

Modes:
  - joint: EN and HI lines (default; bilingual legal vocab)
  - hi: Hindi only
  - en: English only

Output: data/external/spm_corpus_legal_v2_{mode}.txt + sidecars JSON report
"""

import json
import random
from pathlib import Path

from src.config import DEV_DOC_IDS, TEST_DOC_IDS, TRAIN_DOC_IDS


STAGE_A_PATH = Path('data/external/parallel/stage_a_en_hi.jsonl')
TRAIN_PATH = Path('data/processed/train.jsonl')
ALIGNED_PATH = Path('data/aligned/all.jsonl')
PRARABDHA_PATH = Path('data/external/legal_hindi_corpus.txt')
OUT_DIR = Path('data/external')

FORBIDDEN_DOCS = set(DEV_DOC_IDS) | set(TEST_DOC_IDS)
TRAIN_DOCS = set(TRAIN_DOC_IDS)


def _norm_doc_id(raw) -> int | None:
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return None


def _assert_no_forbidden(doc_id: int | None, source: str):
    if doc_id is not None and doc_id in FORBIDDEN_DOCS:
        raise ValueError(f'{source}: forbidden doc_id {doc_id} in SPM corpus')


def load_jsonl_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def lines_from_pairs(
    pairs: list[dict],
    mode: str,
    require_train_docs: bool = False,
    source_label: str = '',
) -> list[str]:
    out: list[str] = []
    for p in pairs:
        doc_id = _norm_doc_id(p.get('doc_id'))
        if require_train_docs:
            if doc_id is not None and doc_id in FORBIDDEN_DOCS:
                _assert_no_forbidden(doc_id, source_label)
            if doc_id is None or doc_id not in TRAIN_DOCS:
                continue
        else:
            # Stage A: doc_id is string like Acts:... or anuvaad_*:i — never assignment int
            if doc_id is not None and doc_id in FORBIDDEN_DOCS:
                raise ValueError(f'{source_label}: assignment eval doc leaked into Stage A path')

        en = (p.get('en_text') or '').strip()
        hi = (p.get('hi_text') or '').strip()
        if mode in ('joint', 'en') and en:
            out.append(en)
        if mode in ('joint', 'hi') and hi:
            out.append(hi)
    return out


def lines_from_prarabdha(path: Path, max_lines: int | None, seed: int) -> list[str]:
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text(encoding='utf-8').splitlines() if ln.strip()]
    if max_lines is not None and max_lines < len(lines):
        rng = random.Random(seed)
        lines = rng.sample(lines, max_lines)
    return lines


def build_corpus(
    mode: str = 'joint',
    include_prarabdha_frac: float = 0.0,
    seed: int = 42,
    stage_a_path: Path = STAGE_A_PATH,
    train_path: Path = TRAIN_PATH,
    aligned_path: Path = ALIGNED_PATH,
    prarabdha_path: Path = PRARABDHA_PATH,
) -> tuple[list[str], dict]:
    if mode not in ('joint', 'hi', 'en'):
        raise ValueError(f'unknown mode {mode}')

    stats: dict = {
        'mode': mode,
        'sources': {},
        'forbidden_docs': sorted(FORBIDDEN_DOCS),
        'train_docs': sorted(TRAIN_DOCS),
    }

    stage_a = load_jsonl_lines(stage_a_path)
    stage_lines = lines_from_pairs(stage_a, mode, require_train_docs=False, source_label='stage_a')
    stats['sources']['stage_a'] = {
        'pairs': len(stage_a),
        'lines': len(stage_lines),
        'path': str(stage_a_path),
    }

    train_pairs = load_jsonl_lines(train_path)
    if not train_pairs and aligned_path.exists():
        # Fallback: filter aligned by TRAIN_DOC_IDS if processed/ missing
        all_pairs = load_jsonl_lines(aligned_path)
        train_pairs = [p for p in all_pairs if _norm_doc_id(p.get('doc_id')) in TRAIN_DOCS]
        stats['sources']['assignment_train_fallback'] = (
            'aligned/all.jsonl filtered by TRAIN_DOC_IDS'
        )

    # Hard reject any train_pairs with forbidden docs
    for p in train_pairs:
        did = _norm_doc_id(p.get('doc_id'))
        _assert_no_forbidden(did, 'assignment_train')
        if did is not None and did not in TRAIN_DOCS:
            raise ValueError(f'assignment_train: doc_id {did} not in TRAIN_DOC_IDS')

    train_lines = lines_from_pairs(
        train_pairs,
        mode,
        require_train_docs=True,
        source_label='assignment_train',
    )
    stats['sources']['assignment_train'] = {
        'pairs': len(train_pairs),
        'lines': len(train_lines),
        'path': str(train_path if train_path.exists() else aligned_path),
    }

    lines = stage_lines + train_lines

    prarabdha_lines: list[str] = []
    if include_prarabdha_frac > 0:
        # Cap Prarabdha contribution as fraction of current line count
        max_extra = int(len(lines) * include_prarabdha_frac)
        prarabdha_lines = lines_from_prarabdha(prarabdha_path, max_extra, seed)
        lines = lines + prarabdha_lines
    stats['sources']['prarabdha_optional'] = {
        'lines': len(prarabdha_lines),
        'frac_requested': include_prarabdha_frac,
        'path': str(prarabdha_path),
    }

    # Drop empties; keep order stable for reproducibility
    lines = [ln for ln in lines if ln]
    chars = sum(len(ln) for ln in lines)
    dev = sum(1 for ln in lines for c in ln if 0x0900 <= ord(c) <= 0x097F)
    stats['totals'] = {
        'lines': len(lines),
        'chars': chars,
        'devanagari_chars': dev,
        'devanagari_char_frac': round(dev / max(chars, 1), 4),
    }
    return lines, stats


def write_corpus(lines: list[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for ln in lines:
            f.write(ln.replace('\n', ' ').replace('\r', ' ') + '\n')


def dedupe_text_file(
    in_path: Path,
    out_path: Path,
    max_chars: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Stream exact-line dedupe (hash set) to cut RAM vs storing full strings.
    Optional max_chars truncates each line for SPM-only training.
    """
    import hashlib

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[bytes] = set()
    kept = 0
    skipped_dup = 0
    truncated = 0
    chars_out = 0

    with open(in_path, encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for raw in fin:
            line = raw.rstrip('\n\r')
            if not line:
                continue
            if max_chars is not None and len(line) > max_chars:
                line = line[:max_chars]
                truncated += 1
            digest = hashlib.sha1(line.encode('utf-8')).digest()
            if digest in seen:
                skipped_dup += 1
                continue
            seen.add(digest)
            fout.write(line + '\n')
            kept += 1
            chars_out += len(line)

    stats = {
        'input': str(in_path),
        'output': str(out_path),
        'lines_kept': kept,
        'lines_skipped_dup': skipped_dup,
        'lines_truncated': truncated,
        'chars_out': chars_out,
        'max_chars': max_chars,
        'dedupe_rate': round(skipped_dup / max(kept + skipped_dup, 1), 4),
    }
    report = out_path.with_name(out_path.stem + '_dedupe_report.json')
    report.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    if verbose:
        print(f'Dedupe {in_path.name} -> {out_path.name}')
        print(
            f'  kept={kept:,} dups={skipped_dup:,} '
            f'trunc={truncated:,} chars={chars_out:,} '
            f'dedupe_rate={stats["dedupe_rate"]}'
        )
        print(f'  report={report}')
    return stats


def run(
    mode: str = 'joint',
    include_prarabdha_frac: float = 0.0,
    seed: int = 42,
    verbose: bool = True,
    dedupe: bool = False,
    max_chars: int | None = None,
) -> dict:
    lines, stats = build_corpus(
        mode=mode,
        include_prarabdha_frac=include_prarabdha_frac,
        seed=seed,
    )
    out_path = OUT_DIR / f'spm_corpus_legal_v2_{mode}.txt'
    write_corpus(lines, out_path)
    report_path = OUT_DIR / f'spm_corpus_legal_v2_{mode}_report.json'
    stats['output'] = str(out_path)
    report_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    if verbose:
        print(f'Wrote {out_path}')
        print(f'  lines={stats["totals"]["lines"]:,} chars={stats["totals"]["chars"]:,}')
        print(f'  devanagari_frac={stats["totals"]["devanagari_char_frac"]}')
        print(f'  report={report_path}')

    if dedupe or max_chars is not None:
        suffix = 'dedup'
        if max_chars is not None:
            suffix = f'dedup_c{max_chars}'
        dedup_path = OUT_DIR / f'spm_corpus_legal_v2_{mode}_{suffix}.txt'
        dstats = dedupe_text_file(out_path, dedup_path, max_chars=max_chars, verbose=verbose)
        stats['dedupe'] = dstats
        stats['output_deduped'] = str(dedup_path)
    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build legal SPM v2 training corpus (Track C0)')
    parser.add_argument('--mode', choices=['joint', 'hi', 'en'], default='joint')
    parser.add_argument(
        '--prarabdha-frac',
        type=float,
        default=0.0,
        help='Optional max fraction of lines from old Prarabdha mono (0 disables)',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--dedupe',
        action='store_true',
        help='Also write exact-line-deduped corpus (spm_corpus_legal_v2_{mode}_dedup.txt)',
    )
    parser.add_argument(
        '--max-chars',
        type=int,
        default=None,
        help='Optional per-line char cap when writing deduped corpus (SPM-only)',
    )
    args = parser.parse_args()
    run(
        mode=args.mode,
        include_prarabdha_frac=args.prarabdha_frac,
        seed=args.seed,
        dedupe=args.dedupe or args.max_chars is not None,
        max_chars=args.max_chars,
    )


if __name__ == '__main__':
    main()

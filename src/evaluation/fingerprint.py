"""Run / hyp-file fingerprints for eval reproducibility.

Phase 4: every eval report records a config fingerprint (model id, adapter
path, beam, max lengths, tokenizer vocab size, decode device/dtype, seed,
MBR config) plus a SHA256 prefix of each hyp file. Resume refuses to mix
rows decoded under a different config; `--force-resume` overrides. Pure
helpers live here so the COMET cache and the resume gate share them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SHA_PREFIX_LEN = 16


class FingerprintMismatchError(RuntimeError):
    pass


def sha256_prefix(data: bytes, length: int = SHA_PREFIX_LEN) -> str:
    return hashlib.sha256(data).hexdigest()[:length]


def file_sha256_prefix(path: Path, length: int = SHA_PREFIX_LEN) -> str:
    path = Path(path)
    if not path.exists():
        return ''
    with open(path, 'rb') as f:
        return sha256_prefix(f.read(), length=length)


def build_config_fingerprint(
    model_id: str,
    adapters: str | None,
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
    vocab_size: int | None,
    device: str,
    dtype: str,
    seed: int,
    mbr: dict | None,
) -> dict[str, Any]:
    return {
        'model_id': model_id,
        'adapters': adapters or 'base',
        'max_input_length': max_input_length,
        'max_new_tokens': max_new_tokens,
        'num_beams': num_beams,
        'vocab_size': vocab_size,
        'decode_device': device,
        'dtype': dtype,
        'seed': seed,
        'mbr': mbr,
    }


def check_resume_fingerprint(
    hyp_path: Path,
    report_path: Path,
    pending: dict[str, Any],
    force_resume: bool = False,
    tag: str = '',
    suite: str = '',
) -> None:
    """Raise FingerprintMismatchError when resuming a hyp file whose recorded
    config does not match the pending run. A hyp file with no recorded
    fingerprint (old-scheme report) is treated as a mismatch: the exact risk
    this gate exists for is mixing stale rows into a new-config run."""
    if force_resume:
        return
    if not Path(hyp_path).exists():
        return
    if not Path(report_path).exists():
        raise FingerprintMismatchError(
            f'{tag}/{suite}: hyp file exists but no report fingerprint; '
            'use --force-resume to override'
        )
    try:
        report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    except ValueError as e:
        raise FingerprintMismatchError(
            f'{tag}/{suite}: unreadable report {report_path}; use --force-resume to override'
        ) from e
    recorded = report.get('fingerprint')
    if recorded is None:
        raise FingerprintMismatchError(
            f'{tag}/{suite}: report has no fingerprint; use --force-resume to override'
        )
    if recorded != pending:
        raise FingerprintMismatchError(
            f'{tag}/{suite}: fingerprint mismatch, recorded {recorded} vs pending {pending}; '
            'use --force-resume to override'
        )

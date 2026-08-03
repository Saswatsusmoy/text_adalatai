"""Strip known OCR / list-marker noise from reference text only.

Phase 4: references carry artifacts that cap every corpus score -- stray
danda before punctuation, list-marker fragments like `#.` / `.` and bare
leading digits. `clean_ref` removes exactly these artifact families from the
reference side so a clearly-labeled `ref_cleaned` BLEU/chrF++ can be emitted
next to the primary columns. It never touches hypotheses or sources, and it
never substitutes words (typo normalization like `क्षेत्रिय` vs
`क्षेत्रीय` is deliberately out of scope).
"""

from __future__ import annotations

import re


DANDA_BEFORE_PUNCT = re.compile(r'[।॥]+\s*(?=[,.;:!?])')
HASH_MARKER = re.compile(r'#\.?')
LEADING_DIGIT_MARKER = re.compile(r'^\s*\d{1,2}(?:[.\u0964\u0965]\s*|\s)')
RUNNING_SPACE = re.compile(r' {2,}')


def clean_ref(text: str) -> str:
    if not text:
        return text
    out = DANDA_BEFORE_PUNCT.sub('', text)
    out = HASH_MARKER.sub('', out)
    out = LEADING_DIGIT_MARKER.sub('', out)
    out = RUNNING_SPACE.sub(' ', out)
    return out

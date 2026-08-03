"""Per-suite legal-entity match between hypotheses and references.

Phase 4: entity recall / precision / F1 between the translation (`hyp_hi`)
and the reference (`hi_text`) over a small legal glossary, case-citation and
date patterns. Probes reuse `LEGAL_PROBES_HI` / `LEGAL_PROBES_EN` from the
tokenizer bench; entities are canonicalized so cross-script pairs match:
`Article 227` == `अनुच्छेद 227`, `Section 227` == `धारा 227`,
`12 May 2024` == `12 मई 2024`, `S.C.R. 482` == `SCR 482`.

This is the only metric that directly answers "did it translate the law
correctly", so it is deliberately recall-oriented and approximate: entity
keys are coarse, and a wrong Article/Section mapping (e.g. translating
`Article 227` as `धारा 227`) deliberately scores a miss.
"""

from __future__ import annotations

import re
from datetime import datetime

from src.tokenizer.bench_matrix import LEGAL_PROBES_EN, LEGAL_PROBES_HI


_CANONICAL = {
    'article': 'article',
    'section': 'section',
    'order': 'order',
    'rule': 'rule',
    'writ': 'writ',
    'petition': 'petition',
    'appellant': 'appellant',
    'respondent': 'respondent',
    'judgment': 'judgment',
    'impugned': 'impugned',
}
_EN2ID = {p.lower(): _CANONICAL[p.lower()] for p in LEGAL_PROBES_EN if p.lower() in _CANONICAL}
_HI2ID = {
    'न्यायालय': 'court',
    'अनुच्छेद': 'article',
    'अपीलकर्ता': 'appellant',
    'प्रतिवादी': 'respondent',
    'याचिका': 'petition',
    'रिट': 'writ',
    'धारा': 'section',
    'आदेश': 'order',
    'नियम': 'rule',
    'अधिवक्ता': 'advocate',
    'पुनरीक्षण': 'revision',
    'अधिकारिता': 'jurisdiction',
    'निर्णय': 'judgment',
    'भारत': 'india',
    'संविधान': 'constitution',
}

_CASE_CITE = re.compile(r'\b(?:S\.C\.R\.?|INSC|AIR|SCC|SCR)\s*\d*\b')
_ARTICLE_NUM = re.compile(r'\b(?:Article|Art\.)\s*(\d+)\b', re.IGNORECASE)
_SECTION_NUM = re.compile(r'\b(?:Section|Sec\.)\s*(\d+)\b', re.IGNORECASE)
_ARTICLE_NUM_HI = re.compile(r'अनुच्छेद\s*(\d+)')
_SECTION_NUM_HI = re.compile(r'धारा\s*(\d+)')

_DATE_NUM = re.compile(r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b')
_MONTH_MAP = {
    'january': 1,
    'जनवरी': 1,
    'february': 2,
    'फरवरी': 2,
    'फ़रवरी': 2,
    'march': 3,
    'मार्च': 3,
    'april': 4,
    'अप्रैल': 4,
    'may': 5,
    'मई': 5,
    'june': 6,
    'जून': 6,
    'july': 7,
    'जुलाई': 7,
    'august': 8,
    'अगस्त': 8,
    'september': 9,
    'सितंबर': 9,
    'सितम्बर': 9,
    'october': 10,
    'अक्टूबर': 10,
    'november': 11,
    'नवंबर': 11,
    'नवम्बर': 11,
    'december': 12,
    'दिसंबर': 12,
    'दिसम्बर': 12,
}
_MONTH_ALT = '|'.join(re.escape(k) for k in _MONTH_MAP)
_DATE_WORD = re.compile(r'\b(\d{1,2})\s+(' + _MONTH_ALT + r')\s+(\d{4})\b', re.IGNORECASE)

_DATE_FMTS = ('%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%Y-%m-%d')


def _normalize_date(token: str) -> str:
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(token, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return token.lower()


def _normalize_case_cite(match: str) -> str:
    collapsed = re.sub(r'\s+', ' ', match.strip())
    return collapsed.replace('.', '').upper()


def extract_entities(text: str) -> set[tuple]:
    if not text:
        return set()
    low = text.lower()
    out: set[tuple] = set()
    for probe in LEGAL_PROBES_EN:
        ident = _EN2ID.get(probe.lower())
        if ident and re.search(r'\b' + re.escape(probe.lower()) + r's?\b', low):
            out.add(('probe', ident))
    for probe in LEGAL_PROBES_HI:
        ident = _HI2ID.get(probe)
        if ident and probe in text:
            out.add(('probe', ident))
    for m in _CASE_CITE.finditer(text):
        out.add(('cite', _normalize_case_cite(m.group(0))))
    for m in _ARTICLE_NUM.finditer(text):
        out.add(('cite', 'article', int(m.group(1))))
    for m in _SECTION_NUM.finditer(text):
        out.add(('cite', 'section', int(m.group(1))))
    for m in _ARTICLE_NUM_HI.finditer(text):
        out.add(('cite', 'article', int(m.group(1))))
    for m in _SECTION_NUM_HI.finditer(text):
        out.add(('cite', 'section', int(m.group(1))))
    for m in _DATE_NUM.finditer(text):
        out.add(('date', _normalize_date(m.group(0))))
    for m in _DATE_WORD.finditer(text):
        month = _MONTH_MAP[m.group(2).lower()]
        out.add(('date', f'{int(m.group(3)):04d}-{month:02d}-{int(m.group(1)):02d}'))
    return out


def entity_metrics(ref_entities: set[tuple], hyp_entities: set[tuple]) -> dict:
    ref_n, hyp_n = len(ref_entities), len(hyp_entities)
    matched = len(ref_entities & hyp_entities)
    recall = matched / ref_n if ref_n else 0.0
    precision = matched / hyp_n if hyp_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'n_ref_entities': ref_n,
        'n_hyp_entities': hyp_n,
        'matched': matched,
        'recall': round(recall, 4),
        'precision': round(precision, 4),
        'f1': round(f1, 4),
    }


def entity_panel(hyps: list[str], refs: list[str]) -> dict:
    if not hyps:
        return {'n': 0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
    total_ref = total_hyp = total_matched = 0
    for hyp, ref in zip(hyps, refs):
        m = entity_metrics(extract_entities(ref), extract_entities(hyp))
        total_ref += m['n_ref_entities']
        total_hyp += m['n_hyp_entities']
        total_matched += m['matched']
    recall = total_matched / total_ref if total_ref else 0.0
    precision = total_matched / total_hyp if total_hyp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'n': len(hyps),
        'recall': round(recall, 4),
        'precision': round(precision, 4),
        'f1': round(f1, 4),
        'n_ref_entities': total_ref,
        'n_hyp_entities': total_hyp,
        'matched': total_matched,
    }

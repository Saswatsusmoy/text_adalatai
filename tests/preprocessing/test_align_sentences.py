import json
from pathlib import Path

import torch

from src.preprocessing.align_sentences import (
    MIN_SIMILARITY,
    SIM_MARGIN,
    align_sentences,
    dedup,
    load_sentences,
    process_doc,
    quality_filter,
)


class TestLoadSentences:
    def test_load_en_hi(self):
        en, hi = load_sentences(1)
        assert len(en) > 0
        assert len(hi) > 0

    def test_content_not_empty(self):
        en, hi = load_sentences(1)
        assert all(s.strip() for s in en)
        assert all(s.strip() for s in hi)


class TestQualityFilter:
    def test_empty_en_rejected(self):
        assert not quality_filter({'en_text': '', 'hi_text': 'test', 'similarity': 0.9})

    def test_empty_hi_rejected(self):
        assert not quality_filter({'en_text': 'test', 'hi_text': '', 'similarity': 0.9})

    def test_low_sim_rejected(self):
        assert not quality_filter(
            {'en_text': 'test', 'hi_text': 'test', 'similarity': MIN_SIMILARITY - 0.1}
        )

    def test_good_pair_accepted(self):
        assert quality_filter(
            {'en_text': 'Hello world', 'hi_text': 'नमस्ते दुनिया', 'similarity': 0.8}
        )

    def test_bad_ratio_rejected(self):
        too_long = 'x' * 1000
        short = 'y'
        assert not quality_filter({'en_text': too_long, 'hi_text': short, 'similarity': 0.9})
        assert not quality_filter({'en_text': short, 'hi_text': too_long, 'similarity': 0.9})

    def test_number_only_rejected(self):
        assert not quality_filter({'en_text': '22.', 'hi_text': '22.', 'similarity': 0.9})
        assert not quality_filter({'en_text': '3', 'hi_text': '३', 'similarity': 0.9})

    def test_number_only_kept_when_mixed(self):
        # Number-only check requires BOTH sides to be digits
        assert quality_filter(
            {
                'en_text': 'The appellant cited section 22 of the Act.',
                'hi_text': 'अपीलकर्ता ने अधिनियम की धारा 22 का हवाला दिया।',
                'similarity': 0.9,
            }
        )

    def test_dangling_preposition_rejected(self):
        assert not quality_filter(
            {
                'en_text': 'A reply was submitted by the',
                'hi_text': 'जवाब प्रस्तुत किया।',
                'similarity': 0.8,
            }
        )

    def test_short_dangling_fragment_rejected(self):
        assert not quality_filter(
            {'en_text': 'The statement made by', 'hi_text': 'उद्घोषणा', 'similarity': 0.51}
        )

    def test_long_sentence_ending_in_preposition_kept(self):
        # Legal English legitimately ends long sentences in `of`/`and`/`the`;
        # the dangling filter is length-gated to short fragments only.
        long_en = 'The Committee further recommended that the penalty of dismissal be reviewed and the order of'
        assert len(long_en) > 60
        assert quality_filter(
            {
                'en_text': long_en,
                'hi_text': 'समिति ने आगे सिफारिश की कि बर्खास्तगी के दंड की समीक्षा की जाए और आदेश',
                'similarity': 0.73,
            }
        )


class TestDedup:
    def test_identical_keeps_best(self):
        en = ['The court dismissed the appeal.', 'The court dismissed the appeal.']
        hi = ['अदालत ने अपील खारिज कर दी।', 'कोर्ट ने अपील खारिज की।']
        sims = [0.9, 0.7]
        en_r, hi_r, sims_r = dedup(en, hi, sims)
        assert len(en_r) == 1

    def test_no_duplicates(self):
        en = ['First sentence.', 'Second sentence.']
        hi = ['पहला वाक्य।', 'दूसरा वाक्य।']
        sims = [0.9, 0.85]
        en_r, hi_r, sims_r = dedup(en, hi, sims)
        assert len(en_r) == 2

    def test_empty_input(self):
        en_r, hi_r, sims_r = dedup([], [], [])
        assert en_r == []


class TestMargin:
    def test_margin_keeps_clear_winner(self):
        # en0 uniquely matches hi0 (huge margin); en1 uniquely matches hi1.
        en_emb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        hi_emb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        pairs = align_sentences(en_emb, hi_emb, ['a', 'b'], ['x', 'y'])
        assert len(pairs) == 2
        assert all(p['similarity'] >= MIN_SIMILARITY for p in pairs)

    def test_margin_drops_exact_tie(self):
        # en0 is equally similar to hi0 and hi1 (margin 0) -> dropped as near-tie.
        en_emb = torch.tensor([[1.0, 0.0, 0.0]])
        hi_emb = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pairs = align_sentences(en_emb, hi_emb, ['a'], ['x', 'y'])
        assert pairs == []

    def test_sim_margin_positive(self):
        assert SIM_MARGIN > 0
        assert SIM_MARGIN <= 0.02


class TestProcessDoc:
    def test_process_doc_1(self):
        pairs = process_doc(1, verbose=False)
        assert len(pairs) > 0
        for p in pairs:
            assert 'en_text' in p
            assert 'hi_text' in p
            assert 'doc_id' in p
            assert p['similarity'] >= MIN_SIMILARITY

    def test_all_pairs_have_content(self):
        pairs = process_doc(1, verbose=False)
        for p in pairs:
            assert len(p['en_text'].strip()) > 0
            assert len(p['hi_text'].strip()) > 0

    def test_devanagari_in_hi(self):
        from src.utils.validation import has_devanagari

        pairs = process_doc(1, verbose=False)
        for p in pairs:
            assert has_devanagari(p['hi_text']), f'HI missing Devanagari: {p["hi_text"][:50]}'


class TestRun:
    def test_output_file_exists(self):
        assert Path('data/aligned/all.jsonl').exists()

    def test_output_format(self):
        with open('data/aligned/all.jsonl') as f:
            lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            p = json.loads(line)
            assert 'en_text' in p
            assert 'hi_text' in p
            assert 'doc_id' in p
            assert 'similarity' in p
            assert 'source' in p

    def test_total_pairs(self):
        with open('data/aligned/all.jsonl') as f:
            pairs = [json.loads(line) for line in f]
        assert len(pairs) >= 1000, f'Expected >=1000 pairs, got {len(pairs)}'
        assert len(pairs) <= 3000, f'Expected <=3000 pairs, got {len(pairs)}'

    def test_avg_similarity(self):
        with open('data/aligned/all.jsonl') as f:
            pairs = [json.loads(line) for line in f]
        avg_sim = sum(p['similarity'] for p in pairs) / len(pairs)
        assert 0.6 <= avg_sim <= 0.8, f'Avg sim {avg_sim:.4f} outside expected range'

    def test_near_dedup_effective(self):
        with open('data/aligned/all.jsonl') as f:
            pairs = [json.loads(line) for line in f]
        # Within same doc, no duplicates expected
        from collections import defaultdict

        by_doc = defaultdict(list)
        for p in pairs:
            by_doc[p['doc_id']].append(p['en_text'])
        for doc_id, texts in by_doc.items():
            dupes = len(texts) - len(set(texts))
            assert dupes == 0, f'Doc {doc_id}: {dupes} duplicate EN texts'

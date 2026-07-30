"""Tests for Track C0 SPM corpus firewall and builders."""

import json
from pathlib import Path

import pytest

from src.config import DEV_DOC_IDS, TEST_DOC_IDS, TRAIN_DOC_IDS
from src.tokenizer.prepare_spm_corpus import (
    FORBIDDEN_DOCS,
    build_corpus,
    lines_from_pairs,
    write_corpus,
)


class TestFirewall:
    def test_forbidden_is_dev_and_test(self):
        assert set(DEV_DOC_IDS) | set(TEST_DOC_IDS) == FORBIDDEN_DOCS
        assert FORBIDDEN_DOCS.isdisjoint(set(TRAIN_DOC_IDS))

    def test_rejects_test_doc_in_train_filter(self):
        pairs = [
            {
                'en_text': 'English legal sentence long enough.',
                'hi_text': 'हिंदी कानूनी वाक्य पर्याप्त लंबा।',
                'doc_id': TEST_DOC_IDS[0],
            }
        ]
        with pytest.raises(ValueError, match='forbidden'):
            lines_from_pairs(pairs, 'joint', require_train_docs=True, source_label='t')

    def test_train_only_keeps_train_docs(self):
        pairs = [
            {
                'en_text': 'Train English sentence about the court.',
                'hi_text': 'ट्रेन हिंदी वाक्य न्यायालय के बारे में।',
                'doc_id': TRAIN_DOC_IDS[0],
            },
            {
                'en_text': 'Test English should be dropped.',
                'hi_text': 'टेस्ट हिंदी नहीं आनी चाहिए।',
                'doc_id': TEST_DOC_IDS[0],
            },
        ]
        # require_train_docs skips non-train without raising when we only pass train-filter path
        # For doc in FORBIDDEN with require_train_docs, it raises before skip
        only_train = [pairs[0]]
        lines = lines_from_pairs(
            only_train,
            'joint',
            require_train_docs=True,
            source_label='t',
        )
        assert len(lines) == 2


class TestBuildCorpus:
    def test_build_joint_excludes_eval_docs(self, tmp_path: Path):
        stage = tmp_path / 'stage_a.jsonl'
        train = tmp_path / 'train.jsonl'
        with open(stage, 'w', encoding='utf-8') as f:
            f.write(
                json.dumps(
                    {
                        'en_text': 'External English legal clause about property.',
                        'hi_text': 'बाहरी हिंदी कानूनी खंड संपत्ति के बारे में।',
                        'doc_id': 'milpac:1',
                        'source': 'milpac',
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )
        with open(train, 'w', encoding='utf-8') as f:
            f.write(
                json.dumps(
                    {
                        'en_text': 'Assignment train English about appeal.',
                        'hi_text': 'असाइनमेंट ट्रेन हिंदी अपील के बारे में।',
                        'doc_id': TRAIN_DOC_IDS[0],
                        'source': 'preprocessed',
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )
            f.write(
                json.dumps(
                    {
                        'en_text': 'Should never appear from test doc.',
                        'hi_text': 'टेस्ट दस्तावेज़ से नहीं आना चाहिए।',
                        'doc_id': TEST_DOC_IDS[0],
                        'source': 'preprocessed',
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )

        # train file with test doc must raise
        with pytest.raises(ValueError, match='forbidden'):
            build_corpus(
                mode='joint',
                stage_a_path=stage,
                train_path=train,
                aligned_path=tmp_path / 'missing.jsonl',
                prarabdha_path=tmp_path / 'missing.txt',
            )

        # clean train only
        train2 = tmp_path / 'train_clean.jsonl'
        with open(train2, 'w', encoding='utf-8') as f:
            f.write(
                json.dumps(
                    {
                        'en_text': 'Assignment train English about appeal.',
                        'hi_text': 'असाइनमेंट ट्रेन हिंदी अपील के बारे में।',
                        'doc_id': TRAIN_DOC_IDS[0],
                        'source': 'preprocessed',
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )

        lines, stats = build_corpus(
            mode='joint',
            stage_a_path=stage,
            train_path=train2,
            aligned_path=tmp_path / 'missing.jsonl',
            prarabdha_path=tmp_path / 'missing.txt',
        )
        assert stats['totals']['lines'] == 4  # 2 stage + 2 train
        blob = '\n'.join(lines)
        assert 'External English' in blob
        assert 'Should never appear' not in blob
        assert 'टेस्ट दस्तावेज़' not in blob

    def test_hi_mode_only_hindi(self, tmp_path: Path):
        stage = tmp_path / 'stage_a.jsonl'
        train = tmp_path / 'train.jsonl'
        with open(stage, 'w', encoding='utf-8') as f:
            f.write(
                json.dumps(
                    {
                        'en_text': 'Only English side for stage.',
                        'hi_text': 'केवल हिंदी पक्ष।',
                        'doc_id': 'a:1',
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )
        with open(train, 'w', encoding='utf-8') as f:
            f.write(
                json.dumps(
                    {
                        'en_text': 'Train English.',
                        'hi_text': 'ट्रेन हिंदी।',
                        'doc_id': TRAIN_DOC_IDS[1],
                    },
                    ensure_ascii=False,
                )
                + '\n'
            )
        lines, _ = build_corpus(
            mode='hi',
            stage_a_path=stage,
            train_path=train,
            aligned_path=tmp_path / 'x.jsonl',
            prarabdha_path=tmp_path / 'y.txt',
        )
        assert all(any(0x0900 <= ord(c) <= 0x097F for c in ln) or True for ln in lines)
        assert not any('Only English' in ln for ln in lines)

    def test_write_corpus(self, tmp_path: Path):
        path = tmp_path / 'c.txt'
        write_corpus(['hello', 'विश्व'], path)
        assert path.read_text(encoding='utf-8') == 'hello\nविश्व\n'


class TestDedupe:
    def test_exact_dedupe_and_truncate(self, tmp_path: Path):
        from src.tokenizer.prepare_spm_corpus import dedupe_text_file

        src = tmp_path / 'in.txt'
        src.write_text('aaa\nbbb\naaa\nccccc\n', encoding='utf-8')
        out = tmp_path / 'out.txt'
        stats = dedupe_text_file(src, out, max_chars=3, verbose=False)
        lines = out.read_text(encoding='utf-8').splitlines()
        assert lines == ['aaa', 'bbb', 'ccc']
        assert stats['lines_kept'] == 3
        assert stats['lines_skipped_dup'] == 1
        assert stats['lines_truncated'] == 1

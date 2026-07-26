"""Tests for external parallel corpus ingest + length filters."""

import json
import zipfile
from pathlib import Path

import pandas as pd

from src.preprocessing.ingest_external_parallel import (
    filter_and_dedup,
    load_milpac_en_hi,
    load_parallel_from_zip,
    make_pair,
    passes_length_filter,
    run,
    write_jsonl,
)


class TestLengthFilter:
    def test_accepts_balanced_pair(self):
        assert passes_length_filter('This is an English legal sentence.', 'यह एक कानूनी वाक्य है।')

    def test_rejects_empty(self):
        assert not passes_length_filter('', 'हिंदी')
        assert not passes_length_filter('English', '  ')

    def test_rejects_too_short(self):
        assert not passes_length_filter('ab', 'cd')

    def test_rejects_extreme_ratio(self):
        en = 'x' * 100
        hi = 'य'
        assert not passes_length_filter(en, hi)


class TestFilterDedup:
    def test_drops_dups_and_bad_length(self):
        pairs = [
            make_pair('Hello court order about land.', 'अदालत का भूमि आदेश।', 't'),
            make_pair('Hello court order about land.', 'अदालत का भूमि आदेश।', 't'),
            make_pair('ab', 'cd', 't'),
        ]
        kept, stats = filter_and_dedup(pairs)
        assert stats['kept'] == 1
        assert stats['dropped_dup'] == 1
        assert stats['dropped_length'] == 1
        assert len(kept) == 1


class TestMilpacLoad:
    def test_loads_en_hi_only(self, tmp_path: Path):
        df = pd.DataFrame(
            [
                {
                    'dataset': 'IP',
                    'id': 'Q1',
                    'src_lang': 'EN',
                    'src': 'What is intellectual property?',
                    'tgt_lang': 'HI',
                    'tgt': 'बौद्धिक संपदा क्या है?',
                },
                {
                    'dataset': 'IP',
                    'id': 'Q1',
                    'src_lang': 'EN',
                    'src': 'What is intellectual property?',
                    'tgt_lang': 'BN',
                    'tgt': 'মেধা সম্পত্তি কী?',
                },
            ]
        )
        path = tmp_path / 'toy.xlsx'
        df.to_excel(path, index=False)
        pairs = load_milpac_en_hi(tmp_path)
        assert len(pairs) == 1
        assert pairs[0]['en_text'].startswith('What is')
        assert pairs[0]['source'] == 'milpac_IP'


class TestAnuvaadZip:
    def test_loads_parallel_lines(self, tmp_path: Path):
        zpath = tmp_path / 'toy.zip'
        with zipfile.ZipFile(zpath, 'w') as z:
            z.writestr('en-hi/train.en', 'First English sentence here.\nSecond English line.\n')
            z.writestr('en-hi/train.hi', 'पहला हिंदी वाक्य यहाँ।\nदूसरी हिंदी पंक्ति।\n')
        pairs = load_parallel_from_zip(zpath, 'en-hi/train.en', 'en-hi/train.hi', 'toy')
        assert len(pairs) == 2
        assert pairs[0]['source'] == 'toy'
        assert 'English' in pairs[0]['en_text']


class TestWriteAndRunSmoke:
    def test_write_jsonl_roundtrip(self, tmp_path: Path):
        pairs = [make_pair('English legal text about appeal.', 'अपील के बारे में कानूनी पाठ।', 'x')]
        out = tmp_path / 'out.jsonl'
        write_jsonl(pairs, out)
        loaded = [json.loads(line) for line in out.read_text(encoding='utf-8').splitlines()]
        assert loaded[0]['en_text'].startswith('English')

    def test_run_with_toy_raw(self, tmp_path: Path, monkeypatch):
        import src.preprocessing.ingest_external_parallel as mod

        milpac = tmp_path / 'raw' / 'milpac'
        anuvaad = tmp_path / 'raw' / 'anuvaad'
        out = tmp_path / 'parallel'
        milpac.mkdir(parents=True)
        anuvaad.mkdir(parents=True)

        df = pd.DataFrame(
            [
                {
                    'dataset': 'Acts',
                    'id': '1',
                    'src_lang': 'EN',
                    'src': 'Short title and commencement of the Act.',
                    'tgt_lang': 'HI',
                    'tgt': 'अधिनियम का संक्षिप्त नाम और प्रारंभ।',
                }
            ]
        )
        df.to_excel(milpac / 'toy.xlsx', index=False)

        zpath = anuvaad / 'internal-judicial-2021-v1-en-hi.zip'
        with zipfile.ZipFile(zpath, 'w') as z:
            z.writestr(
                'en-hi/ij-train.en',
                'The High Court dismissed the petition for want of merit.\n',
            )
            z.writestr(
                'en-hi/ij-train.hi',
                'उच्च न्यायालय ने याचिका गुणदोष के अभाव में खारिज की।\n',
            )

        monkeypatch.setattr(mod, 'MILPAC_DIR', milpac)
        monkeypatch.setattr(mod, 'ANUVAAD_DIR', anuvaad)
        monkeypatch.setattr(mod, 'OUT_DIR', out)

        report = run(download=False, verbose=False)
        assert report['sources']['milpac']['kept'] == 1
        assert report['sources']['anuvaad_hc_suvas']['kept'] == 1
        assert (out / 'stage_a_en_hi.jsonl').exists()
        assert report['totals']['kept'] == 2

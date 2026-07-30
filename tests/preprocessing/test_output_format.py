import json
from pathlib import Path

from src.preprocessing.output_format import (
    build_metadata,
    build_report,
    load_aligned,
    run,
    split_docs,
)


class TestLoadAligned:
    def test_loads_pairs(self):
        pairs = load_aligned()
        assert len(pairs) > 0
        assert 'en_text' in pairs[0]

    def test_doc_ids_present(self):
        pairs = load_aligned()
        for p in pairs:
            assert 'doc_id' in p


class TestSplitDocs:
    def test_split_30_docs(self):
        doc_ids = list(range(1, 31))
        splits = split_docs(doc_ids)
        assert len(splits['train']) == 24  # 80%
        assert len(splits['dev']) == 3  # 10%
        assert len(splits['test']) == 3  # 10%

    def test_no_overlap(self):
        doc_ids = list(range(1, 31))
        splits = split_docs(doc_ids)
        assert set(splits['train']) & set(splits['dev']) == set()
        assert set(splits['train']) & set(splits['test']) == set()
        assert set(splits['dev']) & set(splits['test']) == set()

    def test_all_docs_covered(self):
        doc_ids = list(range(1, 31))
        splits = split_docs(doc_ids)
        all_split = set(splits['train']) | set(splits['dev']) | set(splits['test'])
        assert all_split == set(doc_ids)


class TestMetadata:
    def test_metadata_structure(self):
        pairs = load_aligned()
        doc_ids = sorted(set(p['doc_id'] for p in pairs))
        splits = split_docs(doc_ids)
        metadata = build_metadata(pairs, splits)
        assert 'corpus' in metadata
        assert 'splits' in metadata
        assert 'documents' in metadata
        assert metadata['corpus']['total_pairs'] == len(pairs)

    def test_all_docs_in_metadata(self):
        pairs = load_aligned()
        doc_ids = sorted(set(p['doc_id'] for p in pairs))
        splits = split_docs(doc_ids)
        metadata = build_metadata(pairs, splits)
        assert len(metadata['documents']) == len(doc_ids)


class TestReport:
    def test_report_structure(self):
        pairs = load_aligned()
        report = build_report(pairs)
        assert 'pipeline' in report
        assert 'alignment' in report
        assert 'statistics' in report


class TestRun:
    def test_output_files_exist(self):
        run(verbose=False)
        assert (Path('data/processed/train.jsonl')).exists()
        assert (Path('data/processed/dev.jsonl')).exists()
        assert (Path('data/processed/test.jsonl')).exists()
        assert (Path('data/processed/metadata.json')).exists()
        assert (Path('data/processed/alignment_report.json')).exists()

    def test_total_pairs_preserved(self):
        with open('data/processed/train.jsonl') as f:
            train = len(f.readlines())
        with open('data/processed/dev.jsonl') as f:
            dev = len(f.readlines())
        with open('data/processed/test.jsonl') as f:
            test = len(f.readlines())
        assert train + dev + test == 1458

    def test_splits_are_disjoint(self):
        train_docs = set()
        with open('data/processed/train.jsonl') as f:
            for line in f:
                train_docs.add(json.loads(line)['doc_id'])
        dev_docs = set()
        with open('data/processed/dev.jsonl') as f:
            for line in f:
                dev_docs.add(json.loads(line)['doc_id'])
        test_docs = set()
        with open('data/processed/test.jsonl') as f:
            for line in f:
                test_docs.add(json.loads(line)['doc_id'])
        assert train_docs & dev_docs == set()
        assert train_docs & test_docs == set()
        assert dev_docs & test_docs == set()

    def test_each_split_has_pairs(self):
        for name in ['train', 'dev', 'test']:
            with open(f'data/processed/{name}.jsonl') as f:
                pairs = [json.loads(line) for line in f]
            assert len(pairs) > 0, f'{name} is empty'

    def test_jsonl_format(self):
        with open('data/processed/train.jsonl') as f:
            line = f.readline()
            p = json.loads(line)
            assert 'en_text' in p
            assert 'hi_text' in p
            assert 'doc_id' in p
            assert 'similarity' in p
            assert 'source' in p

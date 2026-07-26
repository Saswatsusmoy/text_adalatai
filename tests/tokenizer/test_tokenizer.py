from pathlib import Path

from src.tokenizer.benchmark import load_corpus
from src.tokenizer.train import load


class TestLoadTokenizer:
    def test_load_16k(self):
        path = Path('data/models/tokenizers/sentencepiece_16000.model')
        if path.exists():
            sp = load(path)
            assert sp.GetPieceSize() == 16000

    def test_load_32k(self):
        path = Path('data/models/tokenizers/sentencepiece_32000.model')
        if path.exists():
            sp = load(path)
            assert sp.GetPieceSize() == 32000

    def test_load_41k(self):
        path = Path('data/models/tokenizers/sentencepiece_41000.model')
        if path.exists():
            sp = load(path)
            assert sp.GetPieceSize() == 41000

    def test_has_devanagari_tokens(self):
        for vs in [16000, 32000, 41000]:
            path = Path(f'data/models/tokenizers/sentencepiece_{vs}.model')
            if not path.exists():
                continue
            sp = load(path)
            has_dev = any(
                0x0900 <= ord(c) <= 0x097F
                for i in range(sp.GetPieceSize())
                for c in sp.IdToPiece(i)
            )
            assert has_dev, f'Model {vs} has no Devanagari tokens'

    def test_legal_term_single_token(self):
        path = Path('data/models/tokenizers/sentencepiece_41000.model')
        if not path.exists():
            return
        sp = load(path)
        for word in ['न्यायालय', 'अपीलार्थी']:
            assert len(sp.encode(word)) == 1, f"'{word}' should be 1 token"


class TestLoadV2IfPresent:
    def test_v2_models_load(self):
        paths = list(Path('data/models/tokenizers').glob('sentencepiece_legal_v2_*.model'))
        for path in paths:
            sp = load(path)
            assert sp.GetPieceSize() > 1000


class TestHeldOutCorpus:
    def test_held_out_only_eval_docs(self):
        from src.config import DEV_DOC_IDS, TEST_DOC_IDS

        if not Path('data/aligned/all.jsonl').exists():
            return
        pairs = load_corpus('held_out')
        if not pairs:
            return
        allowed = set(DEV_DOC_IDS) | set(TEST_DOC_IDS)
        for p in pairs:
            did = p.get('doc_id')
            if isinstance(did, str) and did.isdigit():
                did = int(did)
            assert did in allowed

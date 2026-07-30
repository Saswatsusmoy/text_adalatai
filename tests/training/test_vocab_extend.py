"""Unit tests for C1c vocab selection helpers (no full model download)."""

from collections import Counter

from src.training.vocab_extend_nllb import select_tokens


def test_select_tokens_top_k_and_filters():
    freq = Counter({
        '▁petitioner': 100,
        '▁accused': 90,
        '▁': 50,  # too short after strip
        '“': 40,  # junk quote
        '▁याचिकाकर्ता': 80,
        'ab': 5,
    })
    chosen = select_tokens(freq, top_k=3, min_len=2)
    assert '▁petitioner' in chosen
    assert '▁accused' in chosen
    assert '▁याचिकाकर्ता' in chosen
    assert '“' not in chosen
    assert len(chosen) == 3

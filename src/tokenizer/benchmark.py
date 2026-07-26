"""
Single benchmark script for all tokenizers.

Measures chars/token, HI/EN ratio, byte fallback, subword regularity,
entropy, and merge depth across all accessible models.

Usage:
    PYTHONPATH=. python3 src/tokenizer/benchmark.py          # quick: custom SP only
    PYTHONPATH=. python3 src/tokenizer/benchmark.py --full   # all 17+ tokenizers
"""

import json, math, time
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tokenizers import Tokenizer as HFTokenizer


def load_corpus():
    aligned = []
    with open("data/aligned/all.jsonl") as f:
        for line in f:
            aligned.append(json.loads(line))
    return aligned


def count_devanagari(text: str) -> int:
    return sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def benchmark_tokenizer(encode_fn, decode_fn, corpus, label="", vocab_size=0, dev_count=0):
    en_chars = sum(len(p["en_text"]) for p in corpus)
    hi_chars = sum(len(p["hi_text"]) for p in corpus)

    t0 = time.time()
    en_ids = [encode_fn(p["en_text"]) for p in corpus]
    hi_ids = [encode_fn(p["hi_text"]) for p in corpus]
    elapsed = time.time() - t0

    en_flat = [i for ids in en_ids for i in ids]
    hi_flat = [i for ids in hi_ids for i in ids]

    # Token-level analysis
    en_tokens = [decode_fn([i]) for i in en_flat[:10000]]
    hi_tokens = [decode_fn([i]) for i in hi_flat[:10000]]

    en_lens = [len(t) for t in en_tokens]
    hi_lens = [len(t) for t in hi_tokens]

    return {
        "name": label,
        "vocab": vocab_size,
        "dev_tokens": dev_count,
        "en_chars_per_tok": round(en_chars / max(len(en_flat), 1), 2),
        "hi_chars_per_tok": round(hi_chars / max(len(hi_flat), 1), 2),
        "hi_en_ratio": round(len(hi_flat) / max(len(en_flat), 1), 3),
        "total_tokens": len(en_flat) + len(hi_flat),
        "time_s": round(elapsed, 1),
        "en_subword_regularity": round(_entropy([en_lens.count(i) for i in range(1, 20) if en_lens.count(i) > 0]), 2) if en_lens else 0,
        "hi_subword_regularity": round(_entropy([hi_lens.count(i) for i in range(1, 20) if hi_lens.count(i) > 0]), 2) if hi_lens else 0,
    }


def print_table(results: list[dict]):
    print(f"{'Tokenizer':<28} {'Vocab':<7} {'Dev':<5} {'HI c/t':<8} {'HI/EN':<8} {'Total':<8} {'Regul.':<7}")
    print("-" * 73)
    for r in sorted(results, key=lambda x: (x.get("hi_en_ratio", 999) if x.get("hi_en_ratio", 999) > 0 else 1/x.get("hi_en_ratio", 1))):
        reg = f"{r.get('hi_subword_regularity', '-'):<7.2f}" if isinstance(r.get('hi_subword_regularity'), float) else "-"
        print(f"{r['name']:<28} {r['vocab']:<7} {str(r.get('dev_tokens', '-')):<5} {r['hi_chars_per_tok']:<8.2f} {r['hi_en_ratio']:<8.3f} {r['total_tokens']:<8,} {reg}")


def run(full: bool = False):
    corpus = load_corpus()
    results = []

    # --- Custom SentencePiece models (always benchmarked) ---
    for vs in [16000, 32000, 41000]:
        mp = Path(f"data/models/tokenizers/sentencepiece_{vs}.model")
        if not mp.exists():
            continue
        sp = spm.SentencePieceProcessor(model_file=str(mp))
        dev = sum(1 for i in range(sp.GetPieceSize())
                  if any(0x0900 <= ord(c) <= 0x097F for c in sp.IdToPiece(i)))
        r = benchmark_tokenizer(
            lambda t, sp=sp: sp.encode(t),
            lambda ids, sp=sp: sp.decode(ids),
            corpus, label=f"Custom SP {vs}", vocab_size=sp.GetPieceSize(), dev_count=dev)
        results.append(r)

    if not full:
        print_table(results)
        return results

    # --- Full benchmark: download and test all accessible models ---
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer as HFTok

    models = [
        ("sentence-transformers/LaBSE", "LaBSE (for alignment)"),
        ("facebook/nllb-200-distilled-600M", "NLLB-200"),
        ("Qwen/Qwen3-8B", "Qwen3"),
        ("Qwen/Qwen3.6-27B", "Qwen3.6"),
        ("deepseek-ai/DeepSeek-V3", "DeepSeek V3"),
        ("deepseek-ai/DeepSeek-V4-Pro", "DeepSeek V4 Pro"),
        ("zai-org/GLM-5.2", "GLM 5.2"),
        ("mistralai/Mistral-Small-4-119B-2603", "Mistral Small 4"),
        ("microsoft/Phi-4-mini-instruct", "Phi-4-mini"),
        ("MiniMaxAI/MiniMax-M3", "MiniMax M3"),
        ("allenai/Olmo-3-32B-Think", "OLMo 3"),
        ("pankajpandey-dev/qwen3.5-9b-hindi-instruct", "Qwen3.5"),
    ]

    for repo_id, label in models:
        try:
            path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", cache_dir="/tmp/tokenizers")
            tok = HFTok.from_file(path)
            vocab = tok.get_vocab_size()
            dev = sum(1 for k in tok.get_vocab() if any(0x0900 <= ord(c) <= 0x097F for c in k))
            r = benchmark_tokenizer(
                lambda t, tok=tok: tok.encode(t).ids,
                lambda ids, tok=tok: tok.decode(ids),
                corpus, label=label, vocab_size=vocab, dev_count=dev)
            results.append(r)
        except Exception as e:
            print(f"  [SKIP] {label}: {e}")

    # Also try tiktoken (GPT-4o, no download needed)
    try:
        import tiktoken
        o200k = tiktoken.get_encoding("o200k_base")
        r = benchmark_tokenizer(
            lambda t, tok=o200k: tok.encode(t),
            lambda ids, tok=o200k: tok.decode(ids),
            corpus, label="GPT-4o (o200k)", vocab_size=o200k.n_vocab, dev_count=0)
        # Estimate Dev tokens from sample
        dev_sample = sum(1 for i in range(min(50000, o200k.n_vocab))
                         if any(0x0900 <= ord(c) <= 0x097F for c in o200k.decode([i])))
        r["dev_tokens"] = dev_sample
        results.append(r)
    except Exception:
        pass

    print_table(results)

    # Save
    out = Path("data/analysis/tokenizer_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark tokenizers on EN-HI corpus")
    parser.add_argument("--full", action="store_true", help="Benchmark all accessible models (not just custom SP)")
    args = parser.parse_args()
    run(full=args.full)

"""
Align English and Hindi sentences within each document using LaBSE embeddings,
then apply quality filters. Produces sentence-aligned parallel data as JSONL.

Uses dynamic programming for alignment (1-1, 1-many, many-1, many-many),
then filters by length ratio and LaBSE similarity threshold.

Output: data/aligned/all.jsonl (all pairs with doc_id and scores)
"""

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

from src.config import DOC_IDS

EN_SEGMENTED_DIR = Path("data/english/segmented")
HI_SEGMENTED_DIR = Path("data/hindi/segmented")
OUTPUT_DIR = Path("data/aligned")

# Quality filter thresholds
MIN_SIMILARITY = 0.5
MIN_CHAR_RATIO = 0.3
MAX_CHAR_RATIO = 3.0
SKIP_PENALTY = 0.5  # cost to skip a sentence (vs aligning)

# Near-dedup on EN side
DEDUP_JACCARD_THRESHOLD = 0.85

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/LaBSE")
    return _model


def load_sentences(doc_id: int) -> tuple[list[str], list[str]]:
    en_path = EN_SEGMENTED_DIR / f"{doc_id}.txt"
    hi_path = HI_SEGMENTED_DIR / f"{doc_id}.txt"

    en_sents = [l.strip() for l in en_path.read_text(encoding="utf-8").split('\n') if l.strip()]
    hi_sents = [l.strip() for l in hi_path.read_text(encoding="utf-8").split('\n') if l.strip()]

    return en_sents, hi_sents


def compute_embeddings(sentences: list[str]) -> torch.Tensor:
    model = _get_model()
    return model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)


def align_sentences(en_emb: torch.Tensor, hi_emb: torch.Tensor,
                    en_sents: list[str], hi_sents: list[str]) -> list[dict]:
    """Greedy bidirectional alignment with quality filtering."""
    sim = util.cos_sim(en_emb, hi_emb)

    # Find best HI match for each EN (best_en_to_hi)
    best_en_to_hi = sim.max(dim=1).values
    best_en_to_hi_idx = sim.argmax(dim=1)

    # Find best EN match for each HI (best_hi_to_en)
    best_hi_to_en_idx = sim.argmax(dim=0)

    pairs = []
    matched_hi = set()

    for en_idx in range(len(en_sents)):
        hi_idx = best_en_to_hi_idx[en_idx].item()
        similarity = best_en_to_hi[en_idx].item()

        # Check if this is a mutual best (bidirectional)
        is_mutual = (best_hi_to_en_idx[hi_idx].item() == en_idx)

        # Only keep if similarity meets threshold
        if similarity >= MIN_SIMILARITY and is_mutual:
            pairs.append({
                "en_idx": en_idx, "hi_idx": hi_idx,
                "similarity": similarity,
                "pair_type": "1-1" if is_mutual else "1-1",
            })
            matched_hi.add(hi_idx)

    return pairs


def build_aligned_text(pair: dict, en_sents: list[str], hi_sents: list[str]) -> dict:
    """Extract text for an alignment pair."""
    def _text(sents, idx):
        if idx is None:
            return ""
        if isinstance(idx, list):
            return " ".join(sents[i] for i in idx)
        return sents[idx]

    en_text = _text(en_sents, pair["en_idx"])
    hi_text = _text(hi_sents, pair["hi_idx"])

    return {
        "en_text": en_text,
        "hi_text": hi_text,
        "en_idx": pair["en_idx"],
        "hi_idx": pair["hi_idx"],
        "similarity": pair["similarity"],
        "pair_type": pair["pair_type"],
    }


def quality_filter(pair: dict) -> bool:
    en_text = pair["en_text"]
    hi_text = pair["hi_text"]

    # Remove empty or near-empty (single punctuation, whitespace-only)
    if not en_text.strip() or not hi_text.strip():
        return False
    if len(en_text.strip()) < 3 or len(hi_text.strip()) < 3:
        return False

    # Length ratio filter
    en_len = len(en_text)
    hi_len = len(hi_text)
    ratio = en_len / max(hi_len, 1)
    if ratio < MIN_CHAR_RATIO or ratio > MAX_CHAR_RATIO:
        return False

    # Similarity filter
    if pair["similarity"] < MIN_SIMILARITY:
        return False

    return True


def dedup(en_texts: list[str], hi_texts: list[str], sims: list[float]) -> tuple[list[str], list[str], list[float]]:
    """Remove near-duplicate EN sentences (keep highest similarity)."""
    if not en_texts:
        return en_texts, hi_texts, sims

    keep = [True] * len(en_texts)
    for i in range(len(en_texts)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(en_texts)):
            if not keep[j]:
                continue
            # Jaccard similarity on word sets
            words_i = set(en_texts[i].lower().split())
            words_j = set(en_texts[j].lower().split())
            union = words_i | words_j
            if not union:
                continue
            jaccard = len(words_i & words_j) / len(union)
            if jaccard > DEDUP_JACCARD_THRESHOLD:
                # Keep the one with higher similarity
                if sims[i] >= sims[j]:
                    keep[j] = False
                else:
                    keep[i] = False

    return (
        [t for t, k in zip(en_texts, keep) if k],
        [t for t, k in zip(hi_texts, keep) if k],
        [s for s, k in zip(sims, keep) if k],
    )


def process_doc(doc_id: int, verbose: bool = False) -> list[dict]:
    en_sents, hi_sents = load_sentences(doc_id)

    if not en_sents or not hi_sents:
        return []

    if verbose:
        print(f"  Doc {doc_id:2d}: EN={len(en_sents)} HI={len(hi_sents)}", end="")

    en_emb = compute_embeddings(en_sents)
    hi_emb = compute_embeddings(hi_sents)

    raw_pairs = align_sentences(en_emb, hi_emb, en_sents, hi_sents)

    # Build text for each pair
    text_pairs = [build_aligned_text(p, en_sents, hi_sents) for p in raw_pairs]

    # Count before filtering
    before = len(text_pairs)

    # Quality filters
    text_pairs = [p for p in text_pairs if quality_filter(p)]
    after_filter = len(text_pairs)

    # Near-dedup
    en_texts = [p["en_text"] for p in text_pairs]
    hi_texts = [p["hi_text"] for p in text_pairs]
    sims = [p["similarity"] for p in text_pairs]
    en_texts, hi_texts, sims = dedup(en_texts, hi_texts, sims)
    after_dedup = len(en_texts)

    # Rebuild with doc_id
    results = []
    for en_text, hi_text, sim in zip(en_texts, hi_texts, sims):
        results.append({
            "en_text": en_text,
            "hi_text": hi_text,
            "doc_id": doc_id,
            "similarity": round(sim, 4),
            "source": "preprocessed",
        })

    if verbose:
        print(f" -> {len(results)} pairs (filtered: {before}->{after_filter}->{after_dedup})")

    return results


def run(doc_ids: list[int] | None = None, verbose: bool = True) -> dict:
    if doc_ids is None:
        doc_ids = DOC_IDS

    all_pairs = []
    for doc_id in doc_ids:
        pairs = process_doc(doc_id, verbose=verbose)
        all_pairs.extend(pairs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "all.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    if verbose:
        print(f"\nTotal: {len(all_pairs)} aligned pairs -> {output_path}")

        # Stats
        sims = [p["similarity"] for p in all_pairs]
        en_lens = [len(p["en_text"]) for p in all_pairs]
        hi_lens = [len(p["hi_text"]) for p in all_pairs]
        print(f"  Avg similarity: {sum(sims)/len(sims):.3f}")
        print(f"  Avg EN length: {sum(en_lens)/len(en_lens):.1f} chars")
        print(f"  Avg HI length: {sum(hi_lens)/len(hi_lens):.1f} chars")

    return {"aligned": len(all_pairs), "output": str(output_path)}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Align EN-HI sentences using LaBSE and quality filters",
    )
    parser.add_argument(
        "--doc-ids", type=int, nargs="+", default=None,
        help="Document IDs to process (default: all 30)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress detailed output",
    )
    args = parser.parse_args()

    run(args.doc_ids, verbose=not args.quiet)


if __name__ == "__main__":
    main()

"""
Download Indian legal corpus and extract Hindi text for tokenizer training.

Source: Prarabdha/indian-legal-supervised-fine-tuning-data (Apache 2.0)
Output: data/external/legal_hindi_corpus.txt (one text per line, gitignored)
"""

from pathlib import Path

from datasets import load_dataset

DEFAULT_OUTPUT = Path("data/external/legal_hindi_corpus.txt")


def prepare(output_path: str | Path = DEFAULT_OUTPUT, max_files: int = 5, verbose: bool = True):
    ds = load_dataset(
        "Prarabdha/indian-legal-supervised-fine-tuning-data",
        split="train",
        cache_dir="/tmp/hf_datasets",
        streaming=True,
    )

    all_texts = []
    count = 0

    for row in ds:
        if max_files and count >= max_files * 183497:
            break

        for col in ["context", "response"]:
            text = row.get(col, "")
            if text and isinstance(text, str) and any(0x0900 <= ord(c) <= 0x097F for c in text):
                all_texts.append(text)

        count += 1
        if verbose and count % 100000 == 0:
            print(f"  Processed {count} rows, {len(all_texts):,} Hindi texts")

    out_path = Path(output_path)
    out_path.write_text("\n".join(all_texts), encoding="utf-8")

    if verbose:
        print(f"\nSaved: {out_path}")
        print(f"  Texts: {len(all_texts):,}")
        print(f"  Characters: {sum(len(t) for t in all_texts):,}")

    return str(out_path)


if __name__ == "__main__":
    prepare()

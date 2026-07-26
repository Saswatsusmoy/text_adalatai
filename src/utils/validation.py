from src.config import DEVANAGARI_START, DEVANAGARI_END


def count_devanagari(text: str) -> int:
    return sum(1 for c in text if DEVANAGARI_START <= ord(c) <= DEVANAGARI_END)


def has_devanagari(text: str, min_chars: int = 1) -> bool:
    return count_devanagari(text) >= min_chars


def devanagari_ratio(text: str) -> float:
    if not text:
        return 0.0
    dev_count = count_devanagari(text)
    non_space = len(text) - text.count(" ")
    return dev_count / non_space if non_space > 0 else 0.0


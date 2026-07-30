"""
Data-driven discovery of proper nouns for line joining.

Scans all 30 English clean files for words that appear at the start
of continuation lines (after a lowercase-ending line). These words
should be joined rather than treated as new sentences.

Output: Prints the discovered PROPER_NOUNS set for join_lines.py
"""

from collections import Counter
from pathlib import Path


def discover(output_file: str | None = None):
    continuation_starts = Counter()
    sentence_starts = Counter()

    for doc_id in range(1, 31):
        path = Path(f'data/english/clean/{doc_id}.txt')
        if not path.exists():
            continue

        text = path.read_text(encoding='utf-8')
        lines = text.split('\n')

        for i in range(len(lines) - 1):
            curr = lines[i].rstrip()
            nxt = lines[i + 1].strip()
            if not curr or not nxt:
                continue

            # Continuation candidate: line ends with lowercase
            if curr[-1].islower():
                first_word = nxt.split()[0].rstrip('.,;:(') if nxt.split() else ''
                if first_word and first_word[0].isupper():
                    continuation_starts[first_word] += 1

            # Sentence start: after .!? or blank line
            if curr[-1] in '.!?' or (i > 0 and not curr):
                first_word = nxt.split()[0].rstrip('.,;:(') if nxt.split() else ''
                if first_word and first_word[0].isupper():
                    sentence_starts[first_word] += 1

    stopwords = {
        'The',
        'In',
        'It',
        'We',
        'This',
        'On',
        'He',
        'She',
        'They',
        'That',
        'A',
        'An',
        'But',
        'As',
        'At',
        'By',
        'For',
        'From',
        'Of',
        'To',
        'With',
    }

    print(f'{"Word":<25} {"As continuation":<18} {"As sentence start":<18} {"Ratio":<8}')
    print('-' * 75)

    proper_nouns = set()
    for word, count in continuation_starts.most_common(100):
        if word in stopwords or count < 2:
            continue
        s_start = sentence_starts.get(word, 0)
        ratio = count / (count + s_start) * 100 if (count + s_start) > 0 else 0
        if ratio > 50:
            proper_nouns.add(word)
            print(f'{word:<25} {count:<18} {s_start:<18} {ratio:<7.0f}%')

    print(f'\nTotal data-derived proper nouns: {len(proper_nouns)}')
    print('\nPROPER_NOUNS for src/preprocessing/join_lines.py:')
    print('PROPER_NOUNS = {')
    for word in sorted(proper_nouns):
        print(f'    "{word}",')
    print('}')

    if output_file:
        out_path = Path(output_file)
        out_path.write_text('\n'.join(f'"{w}"' for w in sorted(proper_nouns)))
        print(f'\nSaved to {output_file}')

    return proper_nouns


if __name__ == '__main__':
    discover()

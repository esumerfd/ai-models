"""
EDA augmentation for Q&A pairs — answer spans only.

Reads gen/qa_pairs.txt, applies WordNet synonym replacement at 10% and 20% ratios
to the <|ai|>...<|end|> answer span of each pair. Questions and special tokens are
never modified.

Output: gen/qa_pairs_augmented.txt containing original + 10% + 20% augmented pairs (~3x).

Lesson from T2b: synonym contamination in UI copy (button labels, field names) produces
nonsense. Q&A answer text is domain prose — safer to augment, but domain nouns (STACK,
Takeoff, Estimate, Acumatica) are still skipped.

Usage:
    python phase_1_training/augment_qa_eda.py
    python phase_1_training/augment_qa_eda.py --input gen/qa_pairs.txt --output gen/qa_pairs_augmented.txt
"""

import argparse
import os
import random
import re
import sys

import nltk
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

INPUT_FILE = "gen/qa_pairs.txt"
OUTPUT_FILE = "gen/qa_pairs_augmented.txt"
DEFAULT_RATIOS = [0.1, 0.2]
RANDOM_SEED = 42

# Domain nouns and UI terms to never replace
_SKIP_RE = re.compile(
    r"^([A-Z]{2,}|STACK|Takeoffs?|Estimates?|Acumatica|Zendesk|QuickBooks|Sage|Procore|\d[\d.]*)$"
)
_WORD_RE = re.compile(r"[A-Za-z]+")

# Marks the answer span boundary in each formatted pair
_AI_SEP = "<|ai|>"
_END_SEP = "<|end|>"


def _synonyms(word: str) -> list[str]:
    syns = set()
    for syn in wordnet.synsets(word.lower()):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower() and " " not in candidate:
                syns.add(candidate)
    return list(syns)


def _augment_span(text: str, ratio: float, rng: random.Random) -> str:
    words = text.split()
    result = []
    for word in words:
        token = _WORD_RE.match(word)
        if token and not _SKIP_RE.match(word) and rng.random() < ratio:
            syns = _synonyms(word)
            if syns:
                replacement = rng.choice(syns)
                if word[0].isupper():
                    replacement = replacement.capitalize()
                word = replacement
        result.append(word)
    return " ".join(result)


def _augment_pair(pair: str, ratio: float, rng: random.Random) -> str:
    """Augment only the answer span of a formatted pair."""
    if _AI_SEP not in pair or _END_SEP not in pair:
        return pair
    pre, rest = pair.split(_AI_SEP, 1)
    answer, post = rest.split(_END_SEP, 1)
    augmented_answer = _augment_span(answer, ratio, rng)
    return f"{pre}{_AI_SEP}{augmented_answer}{_END_SEP}{post}"


def augment_pairs(pairs: list[str], ratio: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [_augment_pair(p, ratio, rng) for p in pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}. Run 'make synthesize' first.")

    with open(args.input, encoding="utf-8") as f:
        raw = f.read()

    pairs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    print(f"Loaded {len(pairs)} Q&A pairs")

    all_pairs = list(pairs)
    for i, ratio in enumerate(args.ratios):
        augmented = augment_pairs(pairs, ratio, seed=RANDOM_SEED + i)
        all_pairs.extend(augmented)
        print(f"Augmented at {ratio:.0%}: {len(augmented)} pairs")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_pairs) + "\n")

    print(f"\nTotal pairs: {len(all_pairs)} ({len(all_pairs) / len(pairs):.1f}x)")
    print(f"Written to {args.output}")

    # Show a sample diff
    print("\n--- Sample answer-span augmentation (ratio=0.1) ---")
    sample_augmented = augment_pairs(pairs, 0.1, RANDOM_SEED)
    shown = 0
    for orig, aug in zip(pairs, sample_augmented):
        if orig != aug:
            orig_ans = orig.split(_AI_SEP, 1)[1].split(_END_SEP)[0].strip()[:200]
            aug_ans = aug.split(_AI_SEP, 1)[1].split(_END_SEP)[0].strip()[:200]
            print(f"  orig: {orig_ans}")
            print(f"  aug:  {aug_ans}")
            print()
            shown += 1
            if shown >= 3:
                break


if __name__ == "__main__":
    main()

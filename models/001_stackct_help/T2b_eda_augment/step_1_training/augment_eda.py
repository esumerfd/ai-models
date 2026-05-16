"""
EDA (Easy Data Augmentation) augmentation script.

Applies synonym replacement to raw article text at two swap ratios (10% and 20%).
Produces gen/training_augmented.txt containing the original corpus plus both
augmented versions — tripling the effective training signal.

Strategy:
  - Tokenize each sentence into words.
  - For each word, with probability p, replace it with a random WordNet synonym
    that is not the same as the original word.
  - Skip stopwords, numbers, and domain-specific terms (all-caps, version strings).
  - Preserve sentence boundaries and article separators (---).

Usage:
    python step_1_training/augment_eda.py
    python step_1_training/augment_eda.py --input gen/training.txt --output gen/training_augmented.txt --ratios 0.1 0.2
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

INPUT_FILE = "gen/training.txt"
OUTPUT_FILE = "gen/training_augmented.txt"
DEFAULT_RATIOS = [0.1, 0.2]
RANDOM_SEED = 42

# Words to never replace: stopwords, domain brand names, all-caps acronyms
_SKIP_RE = re.compile(r"^([A-Z]{2,}|\d[\d.]*|STACK|Takeoff|Estimate|Zendesk)$")
_WORD_RE = re.compile(r"[A-Za-z]+")


def _synonyms(word: str) -> list[str]:
    syns = set()
    for syn in wordnet.synsets(word.lower()):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower() and " " not in candidate:
                syns.add(candidate)
    return list(syns)


def _augment_sentence(sentence: str, ratio: float, rng: random.Random) -> str:
    words = sentence.split()
    result = []
    for word in words:
        token = _WORD_RE.match(word)
        if token and not _SKIP_RE.match(word) and rng.random() < ratio:
            syns = _synonyms(word)
            if syns:
                replacement = rng.choice(syns)
                # Preserve original capitalisation
                if word[0].isupper():
                    replacement = replacement.capitalize()
                word = replacement
        result.append(word)
    return " ".join(result)


def augment(text: str, ratio: float, seed: int) -> str:
    rng = random.Random(seed)
    lines = text.splitlines()
    output = []
    for line in lines:
        if line.strip() in ("", "---") or line.startswith("#"):
            output.append(line)
        else:
            output.append(_augment_sentence(line, ratio, rng))
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--ratios", nargs="+", type=float, default=DEFAULT_RATIOS)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}. Run 'make retrieve' first.")

    with open(args.input, encoding="utf-8") as f:
        original = f.read()

    sections = [original]
    for i, ratio in enumerate(args.ratios):
        augmented = augment(original, ratio, seed=RANDOM_SEED + i)
        sections.append(augmented)
        print(f"Augmented at {ratio:.0%}: {len(augmented):,} chars")

    combined = "\n\n---\n\n".join(sections)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(combined)

    orig_chars = len(original)
    aug_chars = len(combined)
    print(f"\nOriginal:  {orig_chars:,} chars")
    print(f"Augmented: {aug_chars:,} chars  ({aug_chars / orig_chars:.1f}x)")
    print(f"Written to {args.output}")

    # Show a sample diff
    print("\n--- Sample (ratio=0.1) ---")
    orig_lines = original.splitlines()
    aug_lines = augment(original, 0.1, RANDOM_SEED).splitlines()
    shown = 0
    for o, a in zip(orig_lines, aug_lines):
        if o != a:
            print(f"  orig: {o[:120]}")
            print(f"  aug:  {a[:120]}")
            print()
            shown += 1
            if shown >= 3:
                break


if __name__ == "__main__":
    main()

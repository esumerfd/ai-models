"""
Q&A synthesis script: converts raw STACK support articles into question-answer pairs.

Strategy (rule-based, no external model required):
  - Split training.txt on article boundaries (--- separators).
  - For each article, extract the title (first # heading) and body sentences.
  - Generate Q&A pairs using pattern templates keyed on section headings and
    "How to", "What is", "Steps" patterns common in support documentation.
  - Format each pair as:
      <|start|><|system|>...<|user|>{question}<|ai|>{answer}<|end|>
  - Write all pairs to gen/qa_pairs.txt, one per line separated by a blank line.

Usage:
    python phase_1_training/synthesize_qa.py
    python phase_1_training/synthesize_qa.py --input gen/training.txt --output gen/qa_pairs.txt
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

INPUT_FILE = "gen/training.txt"
OUTPUT_FILE = "gen/qa_pairs.txt"

SYSTEM_PROMPT = "You are a helpful assistant. Answer questions based only on your training data."

# Minimum answer length (chars) to keep a pair
MIN_ANSWER_LEN = 80
# Maximum answer length (chars) — trim to keep context window manageable
MAX_ANSWER_LEN = 600


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Remove markdown links, keep link text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove bare URLs
    text = re.sub(r"https?://\S+", "", text)
    return text.strip()


def _split_articles(raw: str) -> list[str]:
    """Split on --- article separators used by retrieve_data.py."""
    parts = re.split(r"\n---\n", raw)
    return [p.strip() for p in parts if p.strip()]


def _extract_title(article: str) -> str | None:
    m = re.search(r"^#+\s+(.+)$", article, re.MULTILINE)
    return _clean(m.group(1)) if m else None


def _extract_sections(article: str) -> list[tuple[str, str]]:
    """Return list of (heading, body) for each ## section."""
    sections = re.split(r"\n#{2,3}\s+", article)
    result = []
    for sec in sections[1:]:
        lines = sec.strip().splitlines()
        if not lines:
            continue
        heading = _clean(lines[0])
        body = _clean(" ".join(lines[1:]))
        if body:
            result.append((heading, body))
    return result


def _body_sentences(article: str) -> list[str]:
    """Return non-heading sentences from the article body."""
    # Strip headings
    body = re.sub(r"^#+\s+.+$", "", article, flags=re.MULTILINE)
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", body)
    return [_clean(s) for s in sentences if len(_clean(s)) > 40]


def _format_pair(question: str, answer: str) -> str:
    answer = answer[:MAX_ANSWER_LEN].strip()
    return (
        f"<|start|><|system|>{SYSTEM_PROMPT}"
        f"<|user|>{question}"
        f"<|ai|>{answer}<|end|>"
    )


def synthesize(raw: str) -> list[str]:
    pairs: list[str] = []
    articles = _split_articles(raw)

    for article in articles:
        title = _extract_title(article)

        # 1. Title → "What is X?" / "How do I use X?"
        if title:
            body_sents = _body_sentences(article)
            if body_sents:
                answer = " ".join(body_sents[:3])
                if len(answer) >= MIN_ANSWER_LEN:
                    pairs.append(_format_pair(f"What is {title}?", answer))

                # "How do I" variant for procedural articles
                if re.search(r"\b(step|click|select|navigate|open|go to)\b", answer, re.I):
                    pairs.append(_format_pair(f"How do I use {title}?", answer))

        # 2. Section headings → "How do I {heading}?"
        for heading, body in _extract_sections(article):
            if len(body) < MIN_ANSWER_LEN:
                continue
            # Procedural headings make natural questions
            if re.search(r"\b(how|step|creat|add|edit|delet|updat|set|use|view|export|import)\b",
                         heading, re.I):
                pairs.append(_format_pair(f"How do I {heading.lower()}?", body))
            else:
                pairs.append(_format_pair(f"What is {heading}?", body))

        # 3. Explicit "Steps" or numbered-list sections → "How do I {title}?"
        steps_match = re.findall(r"(?:\d+\.\s+)(.+)", article)
        if steps_match and title and len(steps_match) >= 2:
            answer = " ".join(_clean(s) for s in steps_match[:5])
            if len(answer) >= MIN_ANSWER_LEN:
                pairs.append(_format_pair(f"What are the steps to {title.lower()}?", answer))

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}. Run 'make retrieve' first.")

    with open(args.input, encoding="utf-8") as f:
        raw = f.read()

    pairs = synthesize(raw)
    print(f"Generated {len(pairs):,} Q&A pairs from {len(_split_articles(raw)):,} articles")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n\n".join(pairs) + "\n")

    print(f"Written to {args.output}")

    # Print a few samples
    print("\n--- Sample pairs ---")
    for pair in pairs[:3]:
        print(pair[:300])
        print()


if __name__ == "__main__":
    main()

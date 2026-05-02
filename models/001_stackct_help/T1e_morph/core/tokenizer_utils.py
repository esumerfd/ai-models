"""
Tokenizer training and helpers using Morfessor (unsupervised morphological segmentation).

Pipeline:
  1. Train a Morfessor BaselineModel on the corpus word list.
  2. Build a vocabulary: special tokens + <unk> + WORD_SEP + all morphemes seen in corpus.
  3. Encode: split text on special tokens first, then word-split and segment each word.
     A WORD_SEP sentinel is inserted between words so decode can reconstruct spacing.
  4. Decode: map IDs back to morphemes, join (WORD_SEP becomes a space).
"""

import re
import pickle

import morfessor

from core.config import TOKENIZER_FILE

SPECIAL_TOKENS = {
    "<|start|>": 0,
    "<|end|>": 1,
    "<|system|>": 2,
    "<|user|>": 3,
    "<|ai|>": 4,
}
STOP_IDS = set(SPECIAL_TOKENS.values())

# Sentinel inserted between words so decode can reconstruct spacing.
_WORD_SEP = "▁"

# Splits text into word tokens (alnum/apostrophe runs) and punctuation chunks.
_WORD_RE = re.compile(r"[A-Za-z0-9']+|[^\s]")

# Ordered list of special token strings for encode splitting.
_SPECIAL_LIST = list(SPECIAL_TOKENS.keys())


class MorfessorTokenizer:
    """Trained Morfessor model plus an ID<->morpheme mapping."""

    def __init__(self, model, vocab: dict[str, int]):
        self._model = model
        self._vocab = vocab
        self._inv = {v: k for k, v in vocab.items()}
        self._unk_id = vocab.get("<unk>", 1)

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def vocab_size(self) -> int:
        return len(self._vocab)

    def _segment(self, word: str) -> list[str]:
        try:
            morphemes, _ = self._model.viterbi_segment(word)
            return morphemes
        except Exception:
            return [word]

    def _encode_span(self, text: str) -> list[int]:
        """Encode a plain-text span (no special tokens) into IDs."""
        ids: list[int] = []
        words = _WORD_RE.findall(text)
        for i, word in enumerate(words):
            if i > 0:
                ids.append(self._vocab.get(_WORD_SEP, self._unk_id))
            for m in self._segment(word):
                ids.append(self._vocab.get(m, self._unk_id))
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text, handling special tokens as atomic units."""
        # Split on special tokens in order, preserving their positions.
        pattern = "(" + "|".join(re.escape(s) for s in _SPECIAL_LIST) + ")"
        parts = re.split(pattern, text)
        ids: list[int] = []
        for part in parts:
            if part in SPECIAL_TOKENS:
                ids.append(SPECIAL_TOKENS[part])
            elif part:
                ids.extend(self._encode_span(part))
        return ids

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        for id_ in ids:
            tok = self._inv.get(id_, "<unk>")
            if tok == _WORD_SEP:
                parts.append(" ")
            else:
                parts.append(tok)
        return "".join(parts)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump((self._model, self._vocab), f)

    @classmethod
    def load(cls, path: str) -> "MorfessorTokenizer":
        with open(path, "rb") as f:
            model, vocab = pickle.load(f)
        return cls(model, vocab)


def train_tokenizer(text: str) -> "MorfessorTokenizer":
    """Train Morfessor on the corpus and build a morpheme vocabulary."""
    freq: dict[str, int] = {}
    for word in _WORD_RE.findall(text):
        freq[word] = freq.get(word, 0) + 1

    model = morfessor.BaselineModel()
    model.load_data([(count, word) for word, count in freq.items()])
    model.train_batch()

    morpheme_set: set[str] = set()
    for word in freq:
        try:
            morphemes, _ = model.viterbi_segment(word)
            morpheme_set.update(morphemes)
        except Exception:
            morpheme_set.add(word)

    vocab: dict[str, int] = dict(SPECIAL_TOKENS)
    next_id = len(SPECIAL_TOKENS)
    vocab["<unk>"] = next_id; next_id += 1
    vocab[_WORD_SEP] = next_id; next_id += 1
    for m in sorted(morpheme_set):
        if m not in vocab:
            vocab[m] = next_id
            next_id += 1

    tokenizer = MorfessorTokenizer(model, vocab)
    tokenizer.save(TOKENIZER_FILE)
    print(f"Morfessor tokenizer trained. Vocabulary size: {tokenizer.vocab_size()}")
    return tokenizer


def load_tokenizer() -> MorfessorTokenizer:
    return MorfessorTokenizer.load(TOKENIZER_FILE)


def encode(tokenizer: MorfessorTokenizer, s: str) -> list[int]:
    return tokenizer.encode(s)


def decode(tokenizer: MorfessorTokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)

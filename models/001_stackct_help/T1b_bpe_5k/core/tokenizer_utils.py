"""
Tokenizer training and helpers using HuggingFace tokenizers (BPE).
"""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from core.config import VOCAB_SIZE, TOKENIZER_FILE

# Special tokens and their fixed IDs
SPECIAL_TOKENS = {
    "<|start|>": 0,
    "<|end|>": 1,
    "<|system|>": 2,
    "<|user|>": 3,
    "<|ai|>": 4,
}
STOP_IDS = set(SPECIAL_TOKENS.values())  # generation stops on any of these


def train_tokenizer(text: str) -> Tokenizer:
    """Train a BPE tokenizer on the provided text and save it."""
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()

    # Reserve 5 slots for special tokens so vocab_size stays consistent
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE - len(SPECIAL_TOKENS),
        special_tokens=list(SPECIAL_TOKENS.keys()),
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    tokenizer.save(TOKENIZER_FILE)
    print(f"Tokenizer trained. Vocabulary size: {len(tokenizer.get_vocab())}")
    return tokenizer


def load_tokenizer() -> Tokenizer:
    """Load a previously saved tokenizer."""
    return Tokenizer.from_file(TOKENIZER_FILE)


def encode(tokenizer: Tokenizer, s: str) -> list[int]:
    return tokenizer.encode(s).ids


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)

from tokenizers import Tokenizer
from core.config import TOKENIZER_FILE


def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(TOKENIZER_FILE)


def encode(tokenizer: Tokenizer, s: str) -> list[int]:
    return tokenizer.encode(s).ids


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids)

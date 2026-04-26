"""
Tokenizer training and helpers using SentencePiece (Unigram model).
"""

import io
import sentencepiece as spm

from core.config import VOCAB_SIZE, TOKENIZER_FILE

# Special tokens — SentencePiece reserves 0=<pad>, 1=<unk>, 2=<s> by default.
# We add four more as user-defined symbols so their IDs are stable.
SPECIAL_TOKENS = {
    "<|start|>": 3,
    "<|end|>": 4,
    "<|system|>": 5,
    "<|user|>": 6,
    "<|ai|>": 7,
}
STOP_IDS = set(SPECIAL_TOKENS.values())


def _sentences(text: str):
    """Yield sentences by splitting on newlines then on '. ' for long lines."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= 2000:
            yield line
        else:
            for sentence in line.split(". "):
                sentence = sentence.strip()
                if sentence:
                    yield sentence


def train_tokenizer(text: str) -> spm.SentencePieceProcessor:
    """Train a SentencePiece Unigram tokenizer on the provided text and save it."""
    model_io = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=_sentences(text),
        model_writer=model_io,
        vocab_size=VOCAB_SIZE,
        model_type="unigram",
        user_defined_symbols=list(SPECIAL_TOKENS.keys()),
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=-1,  # disable default EOS so we control generation stop
        character_coverage=0.9995,
        max_sentence_length=4096,
    )
    model_bytes = model_io.getvalue()
    with open(TOKENIZER_FILE, "wb") as f:
        f.write(model_bytes)

    sp = spm.SentencePieceProcessor()
    sp.load_from_serialized_proto(model_bytes)
    print(f"Tokenizer trained. Vocabulary size: {sp.get_piece_size()}")
    return sp


def load_tokenizer() -> spm.SentencePieceProcessor:
    """Load a previously saved SentencePiece model."""
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_FILE)
    return sp


def encode(tokenizer: spm.SentencePieceProcessor, s: str) -> list[int]:
    return tokenizer.encode(s)


def decode(tokenizer: spm.SentencePieceProcessor, ids: list[int]) -> str:
    return tokenizer.decode(ids)

"""
Interactive inference for the Seq2Seq transformer.

Encodes the question, then decodes the answer token-by-token (greedy).

Usage:
    python step_2_generation/generate.py
    python step_2_generation/generate.py --checkpoint gen/model_step_5000.pth
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import (
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
    FFN_DIM, DROPOUT, MAX_SRC_LEN, MAX_TGT_LEN, CHECKPOINT_DIR, TOKENIZER_FILE,
)
from core.tokenizer_utils import load_tokenizer, encode, decode
from core.model import Seq2SeqTransformer, device

PAD_ID = 0
END_TOKEN = "<|end|>"


def respond(model, tokenizer, question: str, max_new_tokens: int = 128) -> str:
    model.eval()
    vocab = tokenizer.get_vocab()
    end_id = vocab.get(END_TOKEN, None)

    # Encode question
    src_ids = encode(tokenizer, question)[:MAX_SRC_LEN]
    src_ids += [PAD_ID] * (MAX_SRC_LEN - len(src_ids))
    src = torch.tensor([src_ids], dtype=torch.long, device=device)

    enc_out = model.encode(src)

    # Greedy decode
    tgt_ids = [PAD_ID]  # BOS
    for _ in range(max_new_tokens):
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=device)
        T = tgt.shape[1]
        tgt_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        dec_out = model.decode(tgt, enc_out, tgt_mask)
        next_id = model.output(dec_out[:, -1, :]).argmax(dim=-1).item()
        if end_id is not None and next_id == end_id:
            break
        tgt_ids.append(next_id)

    return decode(tokenizer, tgt_ids[1:])  # strip BOS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "model_final.pth"))
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.get_vocab())

    model = Seq2SeqTransformer(
        vocab_size, EMBEDDING_DIM, NUM_HEADS,
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
        FFN_DIM, MAX_SRC_LEN, MAX_TGT_LEN, DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded model from {args.checkpoint}  ({model.param_count():,} params)")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("USER: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        print(f"AI: {respond(model, tokenizer, query)}\n")


if __name__ == "__main__":
    main()

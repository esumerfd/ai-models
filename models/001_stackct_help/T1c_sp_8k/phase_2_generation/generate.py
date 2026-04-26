"""
Interactive inference script for the trained Small Language Model.

Usage:
    python phase_2_generation/generate.py
    python phase_2_generation/generate.py --checkpoint gen/model_step_1000.pth
"""

import argparse
import os
import sys
import torch

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import CONTEXT_LENGTH, CHECKPOINT_DIR, TOKENIZER_FILE
from core.tokenizer_utils import load_tokenizer, encode, decode, STOP_IDS
from core.model import SmallLanguageModel, device

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer questions based only on your training data."
)


def build_prompt(tokenizer, system: str, query: str) -> list[int]:
    prompt = f"<|start|><|system|>{system}<|user|>{query}<|ai|>"
    return encode(tokenizer, prompt)


def respond(model, tokenizer, query: str, max_new_tokens: int = 200) -> str:
    model.eval()
    input_ids = build_prompt(tokenizer, SYSTEM_PROMPT, query)
    generated_ids = []

    for _ in range(max_new_tokens):
        context = torch.tensor([input_ids], dtype=torch.long, device=device)
        output_ids = model.generate(context, max_new_tokens=1)[0].tolist()
        next_id = output_ids[-1]

        if next_id in STOP_IDS:
            break

        input_ids.append(next_id)
        generated_ids.append(next_id)

    return decode(tokenizer, generated_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(CHECKPOINT_DIR, "model_final.pth"),
        help="Path to model checkpoint (.pth)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(TOKENIZER_FILE):
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_FILE}. Run train.py first."
        )

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_piece_size()

    model = SmallLanguageModel(vocab_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded model from {args.checkpoint}  ({model.param_count():,} params)")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("USER: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        answer = respond(model, tokenizer, query)
        print(f"AI: {answer}\n")


if __name__ == "__main__":
    main()

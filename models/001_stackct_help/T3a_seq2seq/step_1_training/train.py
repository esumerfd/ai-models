"""
Training script for the Seq2Seq transformer.

Reads gen/qa_pairs.txt, parses question/answer spans, and trains the
encoder-decoder with teacher forcing.

Usage:
    python step_1_training/train.py
"""

import os
import sys
import re
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import (
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
    FFN_DIM, DROPOUT, MAX_SRC_LEN, MAX_TGT_LEN,
    BATCH_SIZE, LEARNING_RATE, STEPS, EVAL_INTERVAL, SAVE_INTERVAL, TRAIN_SPLIT,
    DATA_FILE, CHECKPOINT_DIR, TOKENIZER_FILE,
)
from core.tokenizer_utils import load_tokenizer, encode
from core.model import Seq2SeqTransformer, device

_USER_RE = re.compile(r"<\|user\|>(.*?)<\|ai\|>", re.DOTALL)
_AI_RE = re.compile(r"<\|ai\|>(.*?)<\|end\|>", re.DOTALL)

PAD_ID = 0  # BPE vocab: ID 0 is <|start|> but used here as pad (unused in generation)


def parse_pairs(path: str) -> list[tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    pairs = []
    for block in raw.split("\n\n"):
        block = block.strip()
        q_m = _USER_RE.search(block)
        a_m = _AI_RE.search(block)
        if q_m and a_m:
            pairs.append((q_m.group(1).strip(), a_m.group(1).strip()))
    return pairs


def pad(ids: list[int], max_len: int) -> list[int]:
    return (ids + [PAD_ID] * max_len)[:max_len]


class QADataset(Dataset):
    def __init__(self, pairs, tokenizer, max_src, max_tgt):
        self.data = []
        for q, a in pairs:
            src = pad(encode(tokenizer, q), max_src)
            # Decoder input: shift right (start with token 0 as BOS)
            tgt_ids = encode(tokenizer, a)
            tgt_in = pad([PAD_ID] + tgt_ids, max_tgt)
            tgt_out = pad(tgt_ids + [PAD_ID], max_tgt)
            self.data.append((src, tgt_in, tgt_out))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt_in, tgt_out = self.data[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_in, dtype=torch.long),
            torch.tensor(tgt_out, dtype=torch.long),
        )


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Training data not found: {DATA_FILE}. Run make synthesize first.")

    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.get_vocab())
    print(f"Tokenizer vocab size: {vocab_size}")

    pairs = parse_pairs(DATA_FILE)
    print(f"Loaded {len(pairs)} Q&A pairs")

    split = int(len(pairs) * TRAIN_SPLIT)
    train_ds = QADataset(pairs[:split], tokenizer, MAX_SRC_LEN, MAX_TGT_LEN)
    val_ds = QADataset(pairs[split:], tokenizer, MAX_SRC_LEN, MAX_TGT_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    model = Seq2SeqTransformer(
        vocab_size, EMBEDDING_DIM, NUM_HEADS,
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
        FFN_DIM, MAX_SRC_LEN, MAX_TGT_LEN, DROPOUT,
    ).to(device)
    print(f"Model parameters: {model.param_count():,}  Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=STEPS, pct_start=0.05
    )

    train_iter = iter(train_loader)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.train()
    for step in range(STEPS):
        try:
            src, tgt_in, tgt_out = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, tgt_in, tgt_out = next(train_iter)

        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

        logits = model(src, tgt_in)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            tgt_out.view(-1),
            ignore_index=PAD_ID,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % EVAL_INTERVAL == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Step {step:5d}  Loss: {loss.item():.4f}  LR: {lr:.2e}")

        if step % SAVE_INTERVAL == 0:
            path = os.path.join(CHECKPOINT_DIR, f"model_step_{step}.pth")
            torch.save(model.state_dict(), path)
            print(f"           -> Saved checkpoint: {path}")

    final_path = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()

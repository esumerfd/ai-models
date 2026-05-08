"""
Phase 1: MLM pretraining of BERT-style encoder on raw STACK article corpus.
Randomly masks 15% of tokens and trains the model to predict the originals.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from core.config import (
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, FFN_DIM,
    MAX_SEQ_LEN, DROPOUT, BATCH_SIZE, LEARNING_RATE, STEPS, EVAL_INTERVAL,
    SAVE_INTERVAL, MLM_MASK_PROB, CLS_ID, PAD_ID, MASK_ID,
    DATA_FILE, CHECKPOINT_DIR, TOKENIZER_FILE,
)
from core.model import BertEncoder, device
from core.tokenizer_utils import load_tokenizer, encode


def parse_articles(path: str) -> list[str]:
    articles, current = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("# ") and current:
                articles.append(" ".join(current))
                current = []
            current.append(line.strip())
    if current:
        articles.append(" ".join(current))
    return [a for a in articles if len(a) > 50]


def make_mlm_example(token_ids: list[int]) -> tuple[list[int], list[int]]:
    """Apply MLM masking; returns (masked_ids, labels) where label=-100 for unmasked."""
    masked = list(token_ids)
    labels = [-100] * len(token_ids)
    for i, tok in enumerate(token_ids):
        if tok in (CLS_ID, PAD_ID):
            continue
        if random.random() < MLM_MASK_PROB:
            labels[i] = tok
            r = random.random()
            if r < 0.8:
                masked[i] = MASK_ID
            elif r < 0.9:
                masked[i] = random.randint(5, VOCAB_SIZE - 1)
            # else keep original
    return masked, labels


def make_batch(articles: list[str], tokenizer, batch_size: int):
    samples = random.choices(articles, k=batch_size)
    input_ids_list, label_list = [], []
    for text in samples:
        ids = [CLS_ID] + encode(tokenizer, text)
        ids = ids[:MAX_SEQ_LEN]
        masked, labels = make_mlm_example(ids)
        # Pad to MAX_SEQ_LEN
        pad_len = MAX_SEQ_LEN - len(masked)
        masked += [PAD_ID] * pad_len
        labels += [-100] * pad_len
        input_ids_list.append(masked)
        label_list.append(labels)
    return (
        torch.tensor(input_ids_list, dtype=torch.long, device=device),
        torch.tensor(label_list, dtype=torch.long, device=device),
    )


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Seed artifacts from T2a (tokenizer + corpus)
    if not os.path.exists(TOKENIZER_FILE):
        src = os.path.join("..", "..", "T2a_qa_synth", "gen", "tokenizer.json")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, TOKENIZER_FILE)
            print(f"Copied tokenizer from T2a: {TOKENIZER_FILE}")
        else:
            print(f"ERROR: tokenizer not found at {src}")
            sys.exit(1)

    if not os.path.exists(DATA_FILE):
        src = os.path.join("..", "..", "T2a_qa_synth", "gen", "training.txt")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, DATA_FILE)
            print(f"Copied corpus from T2a: {DATA_FILE}")
        else:
            print(f"ERROR: training corpus not found at {src}")
            sys.exit(1)

    tokenizer = load_tokenizer()
    articles = parse_articles(DATA_FILE)
    print(f"Loaded {len(articles)} articles from corpus")

    model = BertEncoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_ENCODER_LAYERS,
        ffn_dim=FFN_DIM,
        max_seq_len=MAX_SEQ_LEN,
        pad_id=PAD_ID,
        dropout=DROPOUT,
    ).to(device)
    print(f"Model parameters: {model.param_count():,}")
    print(f"Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    losses = []
    for step in range(1, STEPS + 1):
        input_ids, labels = make_batch(articles, tokenizer, BATCH_SIZE)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE),
            labels.view(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % EVAL_INTERVAL == 0:
            avg = sum(losses[-EVAL_INTERVAL:]) / EVAL_INTERVAL
            print(f"step {step:>6} | loss {avg:.4f}")

        if step % SAVE_INTERVAL == 0:
            path = os.path.join(CHECKPOINT_DIR, f"model_step_{step}.pth")
            torch.save(model.state_dict(), path)
            print(f"  → checkpoint saved: {path}")

    final_path = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Final model: {final_path}")


if __name__ == "__main__":
    main()

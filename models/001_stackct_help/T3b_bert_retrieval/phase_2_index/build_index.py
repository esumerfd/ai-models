"""
Phase 2: Encode all STACK articles into embeddings and save a retrieval index.
Loads the MLM-pretrained encoder, mean-pools each article, L2-normalizes,
and writes gen/article_index.npz: {embeddings, texts}.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from core.config import (
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, FFN_DIM,
    MAX_SEQ_LEN, CLS_ID, PAD_ID, CHECKPOINT_DIR, TOKENIZER_FILE,
    DATA_FILE, INDEX_FILE,
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


def tokenize_article(tokenizer, text: str) -> torch.Tensor:
    ids = [CLS_ID] + encode(tokenizer, text)
    ids = ids[:MAX_SEQ_LEN]
    pad_len = MAX_SEQ_LEN - len(ids)
    ids += [PAD_ID] * pad_len
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def main():
    checkpoint = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    if not os.path.exists(checkpoint):
        print(f"ERROR: no trained model at {checkpoint}. Run phase_1_pretrain first.")
        sys.exit(1)

    tokenizer = load_tokenizer()
    articles = parse_articles(DATA_FILE)
    print(f"Indexing {len(articles)} articles…")

    model = BertEncoder(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_ENCODER_LAYERS,
        ffn_dim=FFN_DIM,
        max_seq_len=MAX_SEQ_LEN,
        pad_id=PAD_ID,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i, text in enumerate(articles):
            input_ids = tokenize_article(tokenizer, text)
            emb = model.embed(input_ids).squeeze(0).cpu().numpy()
            norm = np.linalg.norm(emb)
            embeddings.append(emb / norm if norm > 0 else emb)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(articles)} articles encoded")

    embeddings_arr = np.stack(embeddings)
    np.savez(INDEX_FILE, embeddings=embeddings_arr, texts=np.array(articles))
    print(f"\nIndex saved: {INDEX_FILE}")
    print(f"Shape: {embeddings_arr.shape}")


if __name__ == "__main__":
    main()

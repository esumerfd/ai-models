"""
Phase 3: Retrieval REPL — encode a query question and return the top-K most
similar STACK articles by cosine similarity against the prebuilt index.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from core.config import (
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, FFN_DIM,
    MAX_SEQ_LEN, CLS_ID, PAD_ID, CHECKPOINT_DIR, TOKENIZER_FILE,
    INDEX_FILE, TOP_K,
)
from core.model import BertEncoder, device
from core.tokenizer_utils import load_tokenizer, encode


def tokenize_query(tokenizer, text: str) -> torch.Tensor:
    ids = [CLS_ID] + encode(tokenizer, text)
    ids = ids[:MAX_SEQ_LEN]
    pad_len = MAX_SEQ_LEN - len(ids)
    ids += [PAD_ID] * pad_len
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def retrieve(query_emb: np.ndarray, embeddings: np.ndarray, texts: np.ndarray, k: int):
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb = query_emb / norm
    scores = embeddings @ query_emb
    top_k = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), texts[i]) for i in top_k]


def main():
    checkpoint = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    if not os.path.exists(checkpoint):
        print(f"ERROR: no trained model at {checkpoint}. Run step_1_pretrain first.")
        sys.exit(1)
    if not os.path.exists(INDEX_FILE):
        print(f"ERROR: no index at {INDEX_FILE}. Run step_2_index first.")
        sys.exit(1)

    tokenizer = load_tokenizer()

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

    data = np.load(INDEX_FILE, allow_pickle=True)
    embeddings = data["embeddings"]
    texts = data["texts"]
    print(f"Loaded index: {len(texts)} articles")
    print(f"Device: {device}")
    print("\nBERT Retrieval — type a question, Enter to search, Ctrl-C to quit\n")

    with torch.no_grad():
        while True:
            try:
                query = input("USER: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not query:
                continue
            input_ids = tokenize_query(tokenizer, query)
            query_emb = model.embed(input_ids).squeeze(0).cpu().numpy()
            results = retrieve(query_emb, embeddings, texts, TOP_K)
            print()
            for rank, (score, text) in enumerate(results, 1):
                preview = text[:300].replace("\n", " ")
                print(f"[{rank}] score={score:.4f}")
                print(f"    {preview}…")
                print()


if __name__ == "__main__":
    main()

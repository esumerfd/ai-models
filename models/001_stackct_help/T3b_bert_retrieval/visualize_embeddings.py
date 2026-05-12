"""
Visualize article and query embeddings via PCA (256D → 2D).
Shows the embedding collapse — all vectors cluster in a narrow region.

Usage:
    python visualize_embeddings.py
Outputs:
    gen/embedding_viz.png
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from core.config import (
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, FFN_DIM,
    MAX_SEQ_LEN, CLS_ID, PAD_ID, CHECKPOINT_DIR, INDEX_FILE, TOKENIZER_FILE,
)
from core.model import BertEncoder, device
from core.tokenizer_utils import load_tokenizer, encode

TEST_QUERIES = [
    "How do I create a takeoff?",
    "What is a change order?",
    "How do I add a subcontractor?",
    "What is STACK?",
    "How do I export a bid?",
    "How do I delete a project?",
    "What is the Audit Trail?",
]


def tokenize(tokenizer, text):
    ids = [CLS_ID] + encode(tokenizer, text)
    ids = ids[:MAX_SEQ_LEN]
    ids += [PAD_ID] * (MAX_SEQ_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def main():
    data = np.load(INDEX_FILE, allow_pickle=True)
    article_embs = data["embeddings"]   # (201, 256), already L2-normalised
    texts = data["texts"]

    tokenizer = load_tokenizer()
    model = BertEncoder(
        vocab_size=VOCAB_SIZE, embed_dim=EMBEDDING_DIM, num_heads=NUM_HEADS,
        num_layers=NUM_ENCODER_LAYERS, ffn_dim=FFN_DIM,
        max_seq_len=MAX_SEQ_LEN, pad_id=PAD_ID,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, "model_final.pth"), map_location=device
    ))
    model.eval()

    query_embs = []
    with torch.no_grad():
        for q in TEST_QUERIES:
            ids = tokenize(tokenizer, q)
            emb = model.embed(ids).squeeze(0).cpu().numpy()
            norm = np.linalg.norm(emb)
            query_embs.append(emb / norm if norm > 0 else emb)
    query_embs = np.array(query_embs)

    all_embs = np.vstack([article_embs, query_embs])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_embs)
    article_2d = all_2d[:len(article_embs)]
    query_2d = all_2d[len(article_embs):]
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("T3b BERT Retrieval — Embedding Space", fontsize=13, fontweight="bold")

    # Left: PCA scatter
    ax = axes[0]
    ax.scatter(article_2d[:, 0], article_2d[:, 1],
               s=20, alpha=0.5, color="steelblue", label=f"Articles (n={len(article_embs)})")
    ax.scatter(query_2d[:, 0], query_2d[:, 1],
               s=80, marker="*", color="crimson", zorder=5, label="Test queries (n=7)")
    for i, q in enumerate(TEST_QUERIES):
        ax.annotate(q[:25], query_2d[i], fontsize=6, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({var[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} variance)")
    ax.set_title("PCA projection (256D → 2D)\nCollapse visible as tight cluster")
    ax.legend(fontsize=8)

    # Right: histogram of pairwise cosine similarities between articles
    ax2 = axes[1]
    # Sample up to 5000 random pairs
    n = len(article_embs)
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, n, size=5000)
    idx_b = rng.integers(0, n, size=5000)
    mask = idx_a != idx_b
    sims = np.sum(article_embs[idx_a[mask]] * article_embs[idx_b[mask]], axis=1)

    ax2.hist(sims, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(sims.mean(), color="crimson", linestyle="--",
                label=f"Mean = {sims.mean():.3f}")
    ax2.set_xlabel("Cosine similarity")
    ax2.set_ylabel("Count")
    ax2.set_title("Pairwise cosine similarities between articles\nNarrow band = collapsed embeddings")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(CHECKPOINT_DIR, "embedding_viz.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    print(f"PCA variance explained: PC1={var[0]:.1%}  PC2={var[1]:.1%}")
    print(f"Pairwise cosine sim — mean={sims.mean():.3f}  std={sims.std():.4f}  "
          f"min={sims.min():.3f}  max={sims.max():.3f}")


if __name__ == "__main__":
    main()

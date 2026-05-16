"""
Phase 3: Embedding space visualization for the seq2seq encoder-decoder transformer.

Uses the encoder half of the trained model — mean-pools encoder output over token
positions to produce one vector per article, then projects to 2D with PCA.

Usage:
    python step_3_visualize/visualize.py    (run from experiment root)
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import (
    VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
    FFN_DIM, MAX_SRC_LEN, MAX_TGT_LEN, CHECKPOINT_DIR, TOKENIZER_FILE,
)
from core.model import Seq2SeqTransformer, device
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

EXPLANATION = """
SEQ2SEQ (ENCODER-DECODER) TRANSFORMER — EMBEDDING SPACE
========================================================

What this model is: a T5-style encoder-decoder trained with teacher forcing on 225
Q&A pairs. The encoder reads questions bidirectionally; the decoder generates answers
attending to encoder output via cross-attention.

How embeddings are extracted:
  Only the encoder half is used. Each article is tokenized, fed through the encoder
  blocks (bidirectional self-attention, no causal mask), and the output hidden states
  are mean-pooled into one vector per article. This is the same technique used in
  T3b for comparison.

Why this encoder might differ from T3b (BERT MLM):
  The encoder's training signal came indirectly through decoder cross-attention: it
  had to produce representations that the decoder could condition on. This may impose
  slightly more structure than MLM pretraining on its own. However, the training set
  was only 225 Q&A pairs — far fewer than the 201-article corpus used for T3b MLM.
  Cross-attention alignment on 225 pairs did not converge (see T3a results.md), so
  the encoder likely learned little useful structure.

What the charts show:
  LEFT — PCA scatter: Query stars should cluster near relevant articles in a working
  retrieval system. Collapse = all points form one undifferentiated blob.

  RIGHT — Cosine similarity histogram: Compare std to the T3b histogram (std=0.037).
  A wider spread here would indicate cross-attention training provided more
  discriminative signal than MLM pretraining on the full corpus.

Expected result: similar or worse collapse than T3b, because the encoder training
signal was too weak (225 pairs, non-converged cross-attention) to impose structure.
"""


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


def embed_texts(model, tokenizer, texts: list[str]) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            ids = encode(tokenizer, text)[:MAX_SRC_LEN]
            if len(ids) < 4:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            enc_out = model.encode(x)       # (1, T, embed_dim)
            emb = enc_out.mean(dim=1).squeeze(0).cpu().numpy()
            norm = np.linalg.norm(emb)
            embeddings.append(emb / norm if norm > 0 else emb)
    return np.array(embeddings)


def main():
    checkpoint = os.path.join(CHECKPOINT_DIR, "model_final.pth")
    if not os.path.exists(checkpoint):
        print(f"ERROR: no trained model at {checkpoint}. Run 'make train' first.")
        sys.exit(1)

    corpus_path = os.path.join(CHECKPOINT_DIR, "training.txt")
    if not os.path.exists(corpus_path):
        fallback = os.path.join("..", "T2a_qa_synth", "gen", "training.txt")
        if os.path.exists(fallback):
            import shutil
            shutil.copy(fallback, corpus_path)
            print(f"Copied corpus from T2a: {corpus_path}")
        else:
            print(f"ERROR: no training.txt found at {corpus_path}")
            sys.exit(1)

    print(EXPLANATION)

    tokenizer = load_tokenizer()
    articles = parse_articles(corpus_path)
    print(f"Loaded {len(articles)} articles")

    state = torch.load(checkpoint, map_location=device)
    vocab_size = state["embedding.weight"].shape[0]  # infer from checkpoint
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        ffn_dim=FFN_DIM,
        max_src_len=MAX_SRC_LEN,
        max_tgt_len=MAX_TGT_LEN,
    ).to(device)
    model.load_state_dict(state)
    print(f"Model loaded ({model.param_count():,} params) — device: {device}")

    article_embs = embed_texts(model, tokenizer, articles)
    query_embs = embed_texts(model, tokenizer, TEST_QUERIES)

    all_embs = np.vstack([article_embs, query_embs])
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_embs)
    article_2d = all_2d[:len(article_embs)]
    query_2d = all_2d[len(article_embs):]
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("T3a_seq2seq — Encoder Embedding Space", fontsize=13, fontweight="bold")

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
    ax.set_title("PCA projection (256D → 2D)\nEncoder output, mean-pooled per article")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    n = len(article_embs)
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, n, size=min(5000, n * n))
    idx_b = rng.integers(0, n, size=min(5000, n * n))
    mask = idx_a != idx_b
    sims = np.sum(article_embs[idx_a[mask]] * article_embs[idx_b[mask]], axis=1)
    ax2.hist(sims, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(sims.mean(), color="crimson", linestyle="--",
                label=f"Mean = {sims.mean():.3f}")
    ax2.set_xlabel("Cosine similarity")
    ax2.set_ylabel("Count")
    ax2.set_title("Pairwise article cosine similarities\nCompare std to T3b (std=0.037)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(CHECKPOINT_DIR, "embedding_viz.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    print(f"PCA variance explained: PC1={var[0]:.1%}  PC2={var[1]:.1%}")
    print(f"Pairwise cosine sim — mean={sims.mean():.3f}  std={sims.std():.4f}  "
          f"min={sims.min():.3f}  max={sims.max():.3f}")


if __name__ == "__main__":
    main()

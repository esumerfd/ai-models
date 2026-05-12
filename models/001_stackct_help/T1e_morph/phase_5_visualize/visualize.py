"""
Phase 5: Embedding space visualization for causal (decoder-only) transformer.

Produces gen/embedding_viz.png — a PCA scatter of article and query embeddings,
plus a pairwise cosine similarity histogram.

Usage:
    python phase_5_visualize/visualize.py    (run from experiment root)
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

from core.config import EMBEDDING_DIM, CONTEXT_LENGTH, NUM_HEADS, NUM_LAYERS, CHECKPOINT_DIR
from core.model import SmallLanguageModel, device
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
CAUSAL (DECODER-ONLY) TRANSFORMER — EMBEDDING SPACE
====================================================

What this model is: a GPT-style language model trained to predict the next token.
Attention is causal — each position only sees its left context. The model was NOT
designed to produce document embeddings; these are extracted post-hoc.

How embeddings are extracted:
  Each article is tokenized and fed through the trained transformer blocks.
  Hidden states at all token positions are mean-pooled into one vector.
  Because causal attention gives different context to early vs. late tokens,
  mean-pooling mixes position representations with unequal information content.

What the charts show:
  LEFT — PCA scatter (256D → 2D): Red stars are test queries; blue dots are articles.
  In a retrieval-grade model, each query star would sit near its relevant article.
  Collapse appears as all points forming one tight blob with no structure.

  RIGHT — Cosine similarity histogram: Pairwise similarities between all articles.
  Narrow band (std < 0.05) = collapsed embeddings — no usable ranking signal.
  Wide spread (std > 0.10) = discriminative embeddings — retrieval could work.

Expected result: tight clustering and narrow similarity band, since no retrieval
objective was used during training. Compare to T3b (BERT MLM) to see whether
generative or masked training collapses more.
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
            ids = encode(tokenizer, text)[:CONTEXT_LENGTH]
            if len(ids) < 4:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            _, T = x.shape
            tok_emb = model.tok_embedding(x)
            pos_emb = model.pos_embedding(torch.arange(T, device=device))
            h = model.blocks(tok_emb + pos_emb)   # (1, T, embed_dim)
            emb = h.mean(dim=1).squeeze(0).cpu().numpy()
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
        # Fall back to sibling T2a corpus
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

    # Infer vocab size from checkpoint in case config has VOCAB_SIZE = None
    state = torch.load(checkpoint, map_location="cpu")
    vocab_size = state["tok_embedding.weight"].shape[0]

    model = SmallLanguageModel(vocab_size).to(device)
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
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(__file__)))
    fig.suptitle(f"{exp_name} — Causal Decoder Embedding Space", fontsize=13, fontweight="bold")

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
    ax.set_title("PCA projection (256D → 2D)\nCollapse = tight single cluster")
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
    ax2.set_title("Pairwise article cosine similarities\nNarrow band = no ranking signal")
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

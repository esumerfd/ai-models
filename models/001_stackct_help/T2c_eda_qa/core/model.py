"""
Small Language Model — transformer architecture following the tutorial:
docs/building-small-language-model.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from core.config import (
    EMBEDDING_DIM, CONTEXT_LENGTH, NUM_HEADS, NUM_LAYERS,
    VOCAB_SIZE, DROPOUT
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    """Single causal self-attention head."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIM, head_size)
        self.query = nn.Linear(EMBEDDING_DIM, head_size)
        self.value = nn.Linear(EMBEDDING_DIM, head_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores and apply causal mask
        wei = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(device)
        wei = wei.masked_fill(mask, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Apply attention to values
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multiple attention heads running in parallel."""

    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, EMBEDDING_DIM)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network (expand 4x, ReLU, contract)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer layer: attention + feedforward with residuals and LayerNorm."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        head_size = hidden_dim // num_heads
        self.attention = MultiHeadAttention(head_size, num_heads)
        self.ff = FeedForward(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SmallLanguageModel(nn.Module):
    """Full SLM: embeddings → transformer blocks → token logits."""

    def __init__(self, vocab_size):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.pos_embedding = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_DIM)
        self.blocks = nn.Sequential(
            *[Block(EMBEDDING_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)]
        )
        self.ln_head = nn.Linear(EMBEDDING_DIM, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        tok_emb = self.tok_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        logits = self.ln_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop context to the maximum context length
            idx_cond = idx[:, -CONTEXT_LENGTH:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

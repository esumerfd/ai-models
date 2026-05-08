"""
BERT-style bidirectional encoder built from scratch with PyTorch.

Architecture:
  - Bidirectional self-attention (no causal mask) across all layers
  - MLM head: linear projection to vocab for pretraining
  - Pooling: mean of encoder output over non-padding positions
  - ~5M parameters at EMBEDDING_DIM=256, 6 layers, VOCAB_SIZE=10000
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if pad_mask is not None:
            # pad_mask: (B, T) — True where padding
            scores = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, pad_mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class BertEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 ffn_dim, max_seq_len, pad_id, dropout=0.1):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.mlm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.mlm_head.weight = self.token_emb.weight  # weight tying

    def encode(self, input_ids):
        B, T = input_ids.shape
        pad_mask = (input_ids == self.pad_id)  # (B, T)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.norm(x), pad_mask

    def forward(self, input_ids):
        hidden, _ = self.encode(input_ids)
        return self.mlm_head(hidden)

    def embed(self, input_ids):
        """Mean-pool encoder output over non-padding positions → (B, embed_dim)."""
        hidden, pad_mask = self.encode(input_ids)
        mask = (~pad_mask).unsqueeze(-1).float()  # (B, T, 1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

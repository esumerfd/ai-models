"""
T5-style encoder-decoder transformer built from scratch with PyTorch.

Architecture:
  - Shared token embedding between encoder, decoder, and output projection
  - Encoder: bidirectional self-attention (no causal mask)
  - Decoder: causal self-attention + cross-attention to encoder output
  - ~9M parameters at EMBEDDING_DIM=256, 3+3 layers, VOCAB_SIZE=10000
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

    def forward(self, query, key, value, mask=None):
        B, T_q, C = query.shape
        T_k = key.shape[1]
        q = self.q(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T_q, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
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

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ffn_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads,
                 num_encoder_layers, num_decoder_layers,
                 ffn_dim, max_src_len, max_tgt_len, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.src_pos = nn.Embedding(max_src_len, embed_dim)
        self.tgt_pos = nn.Embedding(max_tgt_len, embed_dim)
        self.encoder = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.norm_enc = nn.LayerNorm(embed_dim)
        self.norm_dec = nn.LayerNorm(embed_dim)
        # Share embedding weights with output projection
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.drop = nn.Dropout(dropout)

    def encode(self, src, src_mask=None):
        B, T = src.shape
        pos = torch.arange(T, device=src.device).unsqueeze(0)
        x = self.drop(self.embedding(src) + self.src_pos(pos))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.norm_enc(x)

    def decode(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        B, T = tgt.shape
        pos = torch.arange(T, device=tgt.device).unsqueeze(0)
        x = self.drop(self.embedding(tgt) + self.tgt_pos(pos))
        for layer in self.decoder:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.norm_dec(x)

    def forward(self, src, tgt, src_mask=None):
        T = tgt.shape[1]
        # Causal mask for decoder self-attention
        tgt_mask = torch.tril(torch.ones(T, T, device=tgt.device)).unsqueeze(0).unsqueeze(0)
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, tgt_mask, src_mask)
        logits = self.output(dec_out)
        return logits

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

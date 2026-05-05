"""
Convert the trained SmallLanguageModel PyTorch checkpoint to GGUF format.

The GGUF file can be loaded by llama.cpp and Ollama. We target the "gpt2"
architecture, which is the closest standard match to our transformer design.

Key weight transformations performed:
  - Per-head Q, K, V weights (shape [head_size, embd]) are concatenated
    into a single QKV matrix (shape [3 * embd, embd]) as expected by llama.cpp
  - All weights are written as float32

Architectural differences vs standard GPT-2 that affect llama.cpp inference:
  - We use ReLU in the FFN; GPT-2 uses GELU
  - We have no final LayerNorm (ln_f) before the output projection
  These mean the GGUF file runs correctly through llama.cpp's loader but
  inference results may differ slightly from our native generate.py.

Usage:
    python phase_3_conversion/convert_to_gguf.py
    python phase_3_conversion/convert_to_gguf.py --checkpoint gen/model_step_5000.pth
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import (
    CHECKPOINT_DIR, TOKENIZER_FILE,
    EMBEDDING_DIM, CONTEXT_LENGTH, NUM_HEADS, NUM_LAYERS,
)
from core.tokenizer_utils import load_tokenizer, SPECIAL_TOKENS
from core.model import SmallLanguageModel

try:
    from gguf import GGUFWriter
except ImportError:
    raise ImportError("Run: pip install gguf")


def t(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a contiguous float32 numpy array."""
    return tensor.detach().float().numpy()


def convert(checkpoint_path: str, output_path: str) -> None:
    print(f"Loading tokenizer from {TOKENIZER_FILE}")
    tokenizer = load_tokenizer()
    vocab_size = len(tokenizer.get_vocab())

    print(f"Loading model from {checkpoint_path}")
    model = SmallLanguageModel(vocab_size)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    state = model.state_dict()

    # -- Extract tokenizer vocab and merges from tokenizer.json ------------------
    with open(TOKENIZER_FILE, "r") as f:
        tokenizer_json = json.load(f)

    # Build token list sorted by ID
    vocab = tokenizer_json["model"]["vocab"]
    tokens = [""] * len(vocab)
    for token, idx in vocab.items():
        tokens[idx] = token

    # Token types: 3 = special, 1 = normal
    special_ids = set(SPECIAL_TOKENS.values())
    token_types = [3 if i in special_ids else 1 for i in range(len(tokens))]

    # BPE merges as "left right" strings — tokenizer.json stores them as
    # [["h","e"], ...] so join each pair into a single "h e" string
    raw_merges = tokenizer_json["model"].get("merges", [])
    merges = [m if isinstance(m, str) else " ".join(m) for m in raw_merges]

    print(f"Writing GGUF to {output_path}")
    writer = GGUFWriter(output_path, "gpt2")

    # -- Metadata ----------------------------------------------------------------
    writer.add_name("stackslm")
    writer.add_description("STACK support small language model")
    writer.add_context_length(CONTEXT_LENGTH)
    writer.add_embedding_length(EMBEDDING_DIM)
    writer.add_feed_forward_length(EMBEDDING_DIM * 4)
    writer.add_block_count(NUM_LAYERS)
    writer.add_head_count(NUM_HEADS)
    writer.add_layer_norm_eps(1e-5)
    writer.add_file_type(0)  # 0 = all F32

    # -- Tokenizer ---------------------------------------------------------------
    writer.add_tokenizer_model("gpt2")  # ByteLevel BPE — matches our pre-tokenizer
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    if merges:
        writer.add_token_merges(merges)
    writer.add_bos_token_id(SPECIAL_TOKENS["<|start|>"])
    writer.add_eos_token_id(SPECIAL_TOKENS["<|end|>"])

    # -- Embeddings --------------------------------------------------------------
    writer.add_tensor("token_embd.weight",    t(state["tok_embedding.weight"]))
    writer.add_tensor("position_embd.weight", t(state["pos_embedding.weight"]))

    # -- Transformer blocks ------------------------------------------------------
    for i in range(NUM_LAYERS):
        prefix = f"blocks.{i}"

        # Layer norms
        writer.add_tensor(f"blk.{i}.attn_norm.weight", t(state[f"{prefix}.ln1.weight"]))
        writer.add_tensor(f"blk.{i}.attn_norm.bias",   t(state[f"{prefix}.ln1.bias"]))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight",  t(state[f"{prefix}.ln2.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_norm.bias",    t(state[f"{prefix}.ln2.bias"]))

        # Concatenate per-head Q, K, V weights → single QKV matrix [3*embd, embd]
        q_w = torch.cat([state[f"{prefix}.attention.heads.{h}.query.weight"] for h in range(NUM_HEADS)])
        k_w = torch.cat([state[f"{prefix}.attention.heads.{h}.key.weight"]   for h in range(NUM_HEADS)])
        v_w = torch.cat([state[f"{prefix}.attention.heads.{h}.value.weight"] for h in range(NUM_HEADS)])
        q_b = torch.cat([state[f"{prefix}.attention.heads.{h}.query.bias"]   for h in range(NUM_HEADS)])
        k_b = torch.cat([state[f"{prefix}.attention.heads.{h}.key.bias"]     for h in range(NUM_HEADS)])
        v_b = torch.cat([state[f"{prefix}.attention.heads.{h}.value.bias"]   for h in range(NUM_HEADS)])

        writer.add_tensor(f"blk.{i}.attn_qkv.weight", t(torch.cat([q_w, k_w, v_w])))
        writer.add_tensor(f"blk.{i}.attn_qkv.bias",   t(torch.cat([q_b, k_b, v_b])))

        # Attention output projection
        writer.add_tensor(f"blk.{i}.attn_output.weight", t(state[f"{prefix}.attention.proj.weight"]))
        writer.add_tensor(f"blk.{i}.attn_output.bias",   t(state[f"{prefix}.attention.proj.bias"]))

        # Feed-forward (expand then contract)
        writer.add_tensor(f"blk.{i}.ffn_up.weight",   t(state[f"{prefix}.ff.net.0.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_up.bias",     t(state[f"{prefix}.ff.net.0.bias"]))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", t(state[f"{prefix}.ff.net.2.weight"]))
        writer.add_tensor(f"blk.{i}.ffn_down.bias",   t(state[f"{prefix}.ff.net.2.bias"]))

    # -- Output norm (identity — our model has no final LayerNorm, but llama.cpp
    #    requires this tensor for the gpt2 architecture) --------------------------
    writer.add_tensor("output_norm.weight", np.ones(EMBEDDING_DIM,  dtype=np.float32))
    writer.add_tensor("output_norm.bias",   np.zeros(EMBEDDING_DIM, dtype=np.float32))

    # -- Output projection -------------------------------------------------------
    # Note: llama.cpp GPT-2 does not use an output bias — omit it
    writer.add_tensor("output.weight", t(state["ln_head.weight"]))

    # -- Write to disk -----------------------------------------------------------
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Done. {output_path} ({size_mb:.1f} MB)")
    print()
    print("To load in Ollama:")
    print(f"  echo 'FROM ./{output_path}' > Modelfile")
    print(f"  ollama create stack-slm -f Modelfile")
    print(f"  ollama run stack-slm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(CHECKPOINT_DIR, "model_final.pth"),
        help="Path to .pth checkpoint (default: gen/model_final.pth)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(CHECKPOINT_DIR, "model.gguf"),
        help="Output GGUF file path (default: gen/model.gguf)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    convert(args.checkpoint, args.output)


if __name__ == "__main__":
    main()

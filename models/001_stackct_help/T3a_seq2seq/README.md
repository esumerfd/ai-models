# 001 — Seq2Seq (Encoder-Decoder) Transformer

A T5-style encoder-decoder transformer built from scratch with PyTorch, trained on the
[STACK](https://www.stackct.com) construction estimating platform support knowledge base.

---

## Repo Structure

```
T3a_seq2seq/                # This experiment — encoder-decoder architecture
├── core/
│   ├── config.py           # Hyperparameters
│   ├── model.py            # Encoder, decoder, cross-attention, Seq2SeqTransformer
│   └── tokenizer_utils.py  # Byte-level BPE 10K (shared from T2a)
│
├── phase_1_training/
│   └── train.py            # Parses Q&A pairs, trains with teacher forcing
│
├── phase_2_generation/
│   └── generate.py         # Greedy decode REPL: encode question → decode answer
│
├── gen/                    # Generated artifacts (gitignored)
│   ├── qa_pairs.txt        # Copied from T2a (225 pairs, best data-prep result)
│   ├── tokenizer.json      # Copied from T2a (BPE 10K)
│   └── model_final.pth     # Trained checkpoint
└── Makefile
```

---

## Experiment Design

All prior experiments (T1a–T2c) used a causal (GPT-style) decoder-only transformer.
This experiment switches to an encoder-decoder architecture — the same approach used by
T5 and BART — where the encoder reads the full question bidirectionally and the decoder
generates the answer attending to the encoded question.

**Hypothesis:** Bidirectional encoding of the question gives the model a richer
representation to condition on, producing more accurate and grounded answers than
causal continuation.

**Architecture differences from baseline:**
- Encoder: 3 transformer blocks, bidirectional self-attention (no causal mask)
- Decoder: 3 transformer blocks, causal self-attention + **cross-attention to encoder**
- Shared embedding between encoder, decoder, and output projection
- Separate positional embeddings for source (question) and target (answer)
- Inference: encode question once, then greedy-decode answer token-by-token

**Training:**
- Input: `qa_pairs.txt` from T2a — 225 pairs parsed into (question, answer) tuples
- Teacher forcing: decoder receives ground-truth answer tokens shifted right
- Loss: cross-entropy on answer tokens only (padding ignored)

---

## Model Architecture

| Hyperparameter | Value |
|---|---|
| Embedding dim | 256 |
| Attention heads | 4 |
| Encoder layers | 3 |
| Decoder layers | 3 |
| FFN dim | 512 |
| Max question length | 64 tokens |
| Max answer length | 128 tokens |
| Parameters | ~5.1M |
| Vocabulary | 10,000 (byte-level BPE, from T1d/T2a) |

---

## Getting Started

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

make train      # trains on gen/qa_pairs.txt
make generate   # interactive REPL
```

---

## Experiment Conclusions

See `results.md` for full analysis and comparison to T2a.

Worse than the causal baseline (0/5 on-topic vs T2a's 3/5). The encoder-decoder needs cross-attention alignment to converge — 225 training pairs is ~100× too small. The decoder ignores encoder output and collapses to repetition loops under greedy decoding. The architecture hypothesis is sound but requires substantially more data to test fairly.

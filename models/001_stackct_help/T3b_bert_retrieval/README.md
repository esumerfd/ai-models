# 001 — BERT Retrieval Encoder

A BERT-style bidirectional encoder built from scratch with PyTorch, trained on the
[STACK](https://www.stackct.com) construction estimating platform support knowledge base
using Masked Language Modeling (MLM). Given a question, retrieves the most relevant
help article rather than generating a free-form answer.

---

## Repo Structure

```
T3b_bert_retrieval/         # This experiment — BERT encoder + retrieval
├── core/
│   ├── config.py           # Hyperparameters and special token IDs
│   ├── model.py            # BertEncoder: bidirectional self-attention, MLM head, mean-pool
│   └── tokenizer_utils.py  # Byte-level BPE 10K (shared from T1d/T2a)
│
├── step_1_pretrain/
│   └── pretrain.py         # MLM training on article corpus
│
├── step_2_index/
│   └── build_index.py      # Encode articles → L2-normalised embedding index
│
├── step_3_retrieve/
│   └── retrieve.py         # Query REPL: encode question → cosine similarity → top-K articles
│
├── gen/                    # Generated artifacts (gitignored)
│   ├── training.txt        # Copied from T2a (raw article corpus)
│   ├── tokenizer.json      # Copied from T2a (BPE 10K)
│   ├── model_final.pth     # Trained encoder checkpoint
│   └── article_index.npz  # Article embeddings + texts
└── Makefile
```

---

## Experiment Design

All prior generative experiments (T1a–T3a) produced free-form text answers. This
experiment switches to a retrieval approach: given a question, find and return the
most relevant help article from the corpus rather than generating new text.

**Hypothesis:** For a domain with a fixed, known knowledge base (STACK help articles),
retrieval is more reliable than generation at small data scales. A model that returns
the correct article is factually accurate by construction; a model that generates
an answer can hallucinate even when the answer is in the training data.

**Why BERT for retrieval:**
- Bidirectional self-attention sees full context in both directions — better semantic
  representations than causal models for embedding use cases.
- MLM pretraining on the raw article corpus (1,403 lines) trains on far more signal
  than the 225 Q&A pairs available for generative experiments.
- At inference, both query and article are encoded into the same embedding space;
  cosine similarity identifies the closest article.

**Key difference from T3a seq2seq:**
- No decoder — this is an encoder-only architecture
- No Q&A pairs needed for pretraining — trains on the raw article corpus with MLM
- Inference: nearest-neighbour search, not autoregressive decoding
- Evaluation: does the top-1 retrieved article contain the correct answer?

---

## Model Architecture

| Hyperparameter | Value |
|---|---|
| Embedding dim | 256 |
| Attention heads | 4 |
| Encoder layers | 6 |
| FFN dim | 512 |
| Max sequence length | 256 tokens |
| Parameters | ~5M |
| Vocabulary | 10,000 (byte-level BPE, from T1d/T2a) |
| Pooling | Mean of non-padding encoder outputs |

**Special token reuse** (from shared BPE 10K tokenizer):
- ID 0 (`<|start|>`) → [CLS] — prepended to every sequence
- ID 1 (`<|end|>`) → [PAD] — padding token
- ID 2 (`<|system|>`) → [MASK] — MLM mask token

---

## Getting Started

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

make train      # MLM-pretrain on article corpus (copies tokenizer + corpus from T2a)
make index      # encode all articles → gen/article_index.npz
make retrieve   # interactive retrieval REPL
```

---

## Experiment Conclusions

See `results.md` for full analysis and comparison to generative baselines.

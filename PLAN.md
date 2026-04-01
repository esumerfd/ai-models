# Experiment Plan — STACK Support Language Models

Compare different training techniques by applying each to the same problem:
build a small language model from STACK support articles that can answer
domain-specific questions, deployable to a Raspberry Pi AI HAT+ 2 (Hailo-10H,
10 TOPS).

All experiments share the same data pipeline (`phase_1_training/retrieve_data.py`),
the same evaluation criteria, and the same hardware constraint.

---

## Baseline

**`models/001_stackct_help`** — Causal Language Model (GPT-style)

Decoder-only transformer trained with next-token prediction. Already implemented.
Serves as the reference point for all comparisons.

---

## Experiments

### 002 — Masked Language Modeling (BERT-style)

**Technique:** Randomly mask 15% of input tokens and train the model to predict
them from bidirectional context using an encoder-only transformer.

**What it tests:** Whether bidirectional context produces better representations
of STACK support content than left-to-right prediction.

**Key differences from baseline:**
- Encoder-only architecture (no causal mask)
- MLM pre-training objective with [MASK] tokens
- Requires a task-specific head for generation (e.g. iterative refinement or a
  separate small decoder)

**Evaluation challenge:** MLM doesn't generate text natively. Test with
fill-in-the-blank and extractive QA rather than open-ended generation.

---

### 003 — Encoder-Decoder (T5-style)

**Technique:** Separate encoder reads the input, decoder generates the output.
Train with a span corruption objective — randomly replace text spans with
sentinel tokens, then reconstruct the original spans.

**What it tests:** Whether an explicit encode-then-decode architecture handles
the question-answering pattern more naturally than a single decoder.

**Key differences from baseline:**
- Two-part architecture (encoder + decoder), roughly doubling parameter count
  at the same layer size
- Span corruption pre-training, then fine-tune on QA pairs
- Cross-attention between encoder and decoder

**Consideration:** May need to reduce layer count to stay within the Pi's
memory budget.

---

### 004 — Replaced Token Detection (ELECTRA-style)

**Technique:** Train a small generator to propose plausible replacements for
masked tokens. Train a discriminator to classify every token as original or
replaced.

**What it tests:** Sample efficiency — ELECTRA learns from every token in the
sequence, not just the 15% that are masked. With a small dataset like STACK
support articles, this could matter.

**Key differences from baseline:**
- Two-model setup during training (generator + discriminator)
- Only the discriminator is kept for inference
- Discriminator can be smaller than an equivalent MLM model for the same
  performance

**Consideration:** The generator can be very small (1/4 the discriminator size).
Total training compute is higher but the final model may be more compact.

---

### 005 — LSTM Language Model

**Technique:** Replace the transformer with a stacked LSTM. Train with the same
next-token prediction objective as the baseline.

**What it tests:** Whether attention is necessary for this domain and dataset
size, or whether a simpler recurrent architecture is sufficient.

**Key differences from baseline:**
- No attention mechanism — relies on hidden state for context
- Sequential processing (no parallelism across sequence length during training)
- Typically fewer parameters for the same hidden dimension

**Consideration:** LSTMs struggle with long-range dependencies but may perform
fine given the 128-token context window and short support articles.

---

### 006 — State Space Model (Mamba-style)

**Technique:** Replace transformer blocks with structured state space layers
that model sequences as continuous-time linear systems, discretized for
efficient computation.

**What it tests:** Whether SSMs can match transformer quality at this scale
while offering better inference efficiency — relevant for the constrained
Pi hardware.

**Key differences from baseline:**
- Linear-time sequence processing (vs quadratic attention)
- No explicit attention mechanism
- Potentially faster inference on sequential generation

**Consideration:** SSM libraries (mamba-ssm) currently require CUDA. May need
a pure-PyTorch implementation for CPU/Hailo deployment.

---

### 007 — Prefix Language Model

**Technique:** Use the same decoder-only transformer but change the attention
mask: a prefix portion (the question) attends bidirectionally, while the
suffix (the answer) attends causally.

**What it tests:** Whether giving the model bidirectional understanding of the
question improves answer quality — a minimal change from the baseline.

**Key differences from baseline:**
- Modified attention mask only — same architecture, same parameter count
- The prefix/suffix boundary must be marked in the training data
- Training data needs explicit question/answer formatting

**Consideration:** This is the smallest delta from the baseline and could be
run as a variant within `001` rather than a separate model.

---

### 008 — Denoising Autoencoder (BART-style)

**Technique:** Corrupt input text using multiple noise functions (token masking,
token deletion, sentence shuffling, span masking) and train an encoder-decoder
to reconstruct the original.

**What it tests:** Whether learning to reconstruct from diverse corruption
produces more robust representations than single-objective training.

**Key differences from baseline:**
- Encoder-decoder architecture (similar to 003)
- Multiple corruption strategies applied simultaneously
- Pre-training is reconstruction; fine-tuning on QA

**Consideration:** Similar architecture cost to 003 but with a more complex
pre-training setup. The diversity of corruption may help with a small dataset.

---

## Evaluation

Each experiment will be evaluated on:

| Metric | Method |
|---|---|
| **Training loss** | Final train/val loss curves |
| **Perplexity** | On held-out 10% validation set |
| **Generation quality** | Manual review of responses to a fixed set of 20 STACK support questions |
| **Model size** | Parameter count and GGUF file size |
| **Inference speed** | Tokens/second on CPU (Mac) and on Pi hardware |
| **Memory usage** | Peak RSS during inference |

## Experiment Order

Suggested order based on implementation complexity and learning value:

1. **005 — LSTM** — Simplest alternative, isolates the value of attention
2. **007 — Prefix LM** — Minimal change from baseline, quick to implement
3. **004 — ELECTRA** — Tests sample efficiency with small data
4. **002 — MLM (BERT)** — Introduces bidirectional encoder architecture
5. **003 — Encoder-Decoder (T5)** — Full seq2seq architecture
6. **008 — Denoising (BART)** — Builds on 003 with richer pre-training
7. **006 — SSM (Mamba)** — Most novel, depends on library availability

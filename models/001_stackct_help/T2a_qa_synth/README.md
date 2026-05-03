# 001 — GPT-Style Causal Transformer

A GPT-style causal transformer built from scratch with PyTorch, trained on the
[STACK](https://www.stackct.com) construction estimating platform support knowledge base.
Deployed to a Raspberry Pi cluster running the
[Raspberry Pi AI HAT+ 2](https://www.raspberrypi.com/products/ai-hat-plus-2/) (Hailo-10H, 10 TOPS).

Based on the tutorial:
**[Building a Small Language Model from Scratch](https://medium.com/@rajasami408/building-a-small-language-model-from-scratch-a-practical-guide-to-domain-specific-ai-59539131437f)**
by Abdul Sami.

---

## Repo Structure

```
T2a_qa_synth/               # This experiment — Q&A synthesis data prep
├── core/                   # Model architecture, config, and tokenizer
│   ├── config.py           # All hyperparameters
│   ├── model.py            # Transformer: Head, MultiHeadAttention, Block, SmallLanguageModel
│   └── tokenizer_utils.py  # Byte-level BPE tokenizer (carried forward from T1d)
│
├── phase_1_training/       # Data collection, synthesis, and model training
│   ├── retrieve_data.py    # Retrieves STACK support articles via Zendesk API
│   ├── synthesize_qa.py    # Converts raw articles into Q&A training pairs
│   └── train.py            # Training loop with scheduler and gradient clipping
│
├── phase_2_generation/     # Interactive inference
│   └── generate.py         # Chat REPL using the trained model
│
├── phase_3_conversion/     # Export for deployment
│   ├── convert_to_gguf.py  # Converts .pth checkpoint to GGUF format
│   └── modelfile-ollama    # Ollama Modelfile with system prompt
│
├── gen/                    # Generated artifacts (gitignored)
│   ├── training.txt        # Raw articles from Zendesk API
│   ├── qa_pairs.txt        # Synthesized Q&A corpus (output of synthesize_qa.py)
│   └── model_final.pth     # Trained model checkpoint
└── Makefile                # All workflow commands
```

---

## Experiment Design

All prior tokenization experiments (T1a–T1e) used raw STACK support articles as training
data. The model learned to continue article-style prose, not to answer questions. This
experiment keeps the architecture and tokenizer fixed (byte-level BPE 10K from T1d) and
changes only the training data format.

**Hypothesis:** Training directly on `<question, answer>` formatted pairs will produce a
model that generates answer-shaped responses instead of continuation-shaped responses.

**Q&A synthesis strategy (rule-based, no external model):**
- Split corpus on article boundaries
- Extract titles and section headings as question seeds
- Apply templates: `"What is {title}?"`, `"How do I {heading}?"`, `"What are the steps to {title}?"`
- Filter pairs by minimum answer length (80 chars) and trim to 600 chars max
- Format using the same special-token prompt structure used at inference time

---

## Model Architecture

| Hyperparameter | Value |
|---|---|
| Embedding dim | 256 |
| Context length | 128 tokens |
| Attention heads | 8 |
| Transformer layers | 6 |
| Parameters | ~9.9M |
| Vocabulary size | 10,000 (byte-level BPE, from T1d) |
| Tokenizer | Byte-level BPE — carried forward as best from tokenization phase |

---

## Getting Started

### Prerequisites

- Python 3.12
- [Ollama](https://ollama.com) (for deployment)

### 1. Create the virtual environment

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. Retrieve training data

```bash
make retrieve
```

Fetches all articles from the STACK support site via the Zendesk Help Center API
and writes them to `gen/training.txt`.

### 3. Synthesize Q&A pairs

```bash
make synthesize
```

Runs `synthesize_qa.py` against `gen/training.txt` and writes `gen/qa_pairs.txt`.
Prints a count of generated pairs and a few samples so you can inspect quality before training.

### 4. Train the model

```bash
make train
```

Trains on `gen/qa_pairs.txt` for 10,000 steps. Checkpoints are saved to `gen/` every 1,000 steps.

### 5. Chat with the model directly

```bash
make generate
```

Interactive REPL — type a question, get a response. Type `quit` to exit.

### 6. Convert to GGUF for Ollama

```bash
make convert
```

Exports the trained model to `gen/model.gguf` in GGUF format compatible with llama.cpp.

### 7. Load and run in Ollama

```bash
make ollama-load
make ollama-run
```

---

## Experiment Conclusions

*To be written after training completes — see `results.md`.*

---

## Makefile Reference

| Command | Description |
|---|---|
| `make retrieve` | Fetch training data from STACK support site |
| `make synthesize` | Generate Q&A pairs from raw articles |
| `make train` | Clean all generated files and retrain from scratch |
| `make generate` | Interactive chat via native PyTorch inference |
| `make convert` | Convert checkpoint to GGUF |
| `make ollama-load` | Register model with Ollama |
| `make ollama-run` | Chat via Ollama |
| `make clean-gguf` | Remove GGUF and deregister from Ollama |
| `make clean-all` | Remove all generated files including checkpoints |

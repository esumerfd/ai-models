# First Small Language Model

A mono-repo for building, training, and deploying a small domain-specific language model (SLM)
from scratch using a variety of techques as I learn then.

All content is suspect as its created knows nothing.

# 001_first_model

How to use PyTorch. The model is trained on the
[STACK](https://www.stackct.com) construction estimating platform support knowledge base and
deployed to a Raspberry Pi cluster running the
[Raspberry Pi AI HAT+ 2](https://www.raspberrypi.com/products/ai-hat-plus-2/) (Hailo-10H, 10 TOPS).

Based on the tutorial:
**[Building a Small Language Model from Scratch](https://medium.com/@rajasami408/building-a-small-language-model-from-scratch-a-practical-guide-to-domain-specific-ai-59539131437f)**
by Abdul Sami.

---

## Repo Structure

```
first-small/
├── core/                   # Shared model architecture, config, and tokenizer
│   ├── config.py           # All hyperparameters
│   ├── model.py            # Transformer: Head, MultiHeadAttention, Block, SmallLanguageModel
│   └── tokenizer_utils.py  # BPE tokenizer training and encode/decode helpers
│
├── phase_1_training/       # Data collection and model training
│   ├── retrieve_data.py    # Retrieves STACK support articles via Zendesk API
│   └── train.py            # Training loop with scheduler and gradient clipping
│
├── phase_2_generation/     # Interactive inference
│   └── generate.py         # Chat REPL using the trained model
│
├── phase_3_conversion/     # Export for deployment
│   ├── convert_to_gguf.py  # Converts .pth checkpoint to GGUF format
│   └── modelfile-ollama    # Ollama Modelfile with system prompt
│
├── gen/                    # Generated model checkpoints and GGUF (gitignored)
├── docs/                   # Reference documentation
└── Makefile                # All workflow commands
```

---

## Model Architecture

A GPT-style causal transformer built from scratch:

| Hyperparameter | Value |
|---|---|
| Embedding dim | 256 |
| Context length | 128 tokens |
| Attention heads | 8 |
| Transformer layers | 6 |
| Parameters | ~5.8M |
| Vocabulary size | ~2,000 tokens |

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

### 3. Train the model

```bash
make train
```

Trains for 10,000 steps (~35 passes through the data). Checkpoints are saved to `gen/`
every 1,000 steps.

### 4. Chat with the model directly

```bash
make generate
```

Interactive REPL — type a question, get a response. Type `quit` to exit.

### 5. Convert to GGUF for Ollama

```bash
make convert
```

Exports the trained model to `gen/model.gguf` in GGUF format compatible with llama.cpp.

### 6. Load and run in Ollama

```bash
make ollama-load
make ollama-run
```

---

## Makefile Reference

| Command | Description |
|---|---|
| `make retrieve` | Fetch training data from STACK support site |
| `make train` | Clean all generated files and retrain from scratch |
| `make generate` | Interactive chat via native PyTorch inference |
| `make convert` | Convert checkpoint to GGUF |
| `make ollama-load` | Register model with Ollama |
| `make ollama-run` | Chat via Ollama |
| `make clean-gguf` | Remove GGUF and deregister from Ollama |
| `make clean-all` | Remove all generated files including checkpoints |

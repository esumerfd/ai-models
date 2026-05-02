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
T1e_morph/                  # This experiment — Morfessor unsupervised morphological tokenizer
├── core/                   # Model architecture, config, and tokenizer
│   ├── config.py           # All hyperparameters
│   ├── model.py            # Transformer: Head, MultiHeadAttention, Block, SmallLanguageModel
│   └── tokenizer_utils.py  # Morfessor tokenizer training and encode/decode helpers
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
| Parameters | ~5.5M (vocab-size dependent) |
| Vocabulary size | Corpus-derived (~1–2K morphemes) |

The vocabulary size is not fixed — it is determined by the number of unique morphemes
Morfessor learns from the training corpus. The embedding layer size adjusts accordingly.

---

## Tokenizer Design

Unlike BPE (T1a–T1d), this experiment uses **Morfessor** — an unsupervised morphological
segmentation algorithm. Morfessor starts with individual characters and greedily merges
them using a minimum description length (MDL) objective, learning morpheme boundaries
directly from word-frequency statistics rather than character-pair merge rules.

Key differences from BPE:
- Vocabulary is smaller (morpheme inventory, not subword merges)
- Word-initial position is tracked via a `▁` separator inserted between words
- No `Ġ` byte-level prefix — morphemes are plain Unicode strings
- Unknown morphemes fall back to `<unk>` (no byte-level guarantee)

The tokenizer is saved as `gen/tokenizer.pkl` (pickle of the Morfessor model + vocab dict).

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

Trains the Morfessor tokenizer on the corpus, then trains the transformer for 10,000
steps. Checkpoints are saved to `gen/` every 1,000 steps.

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

## Experiment Conclusions

See `results.md` for full tokenizer analysis, generation samples, and cross-experiment comparison.

Morfessor produces the cleanest output of all five tokenizer experiments — no `Ġ` byte-level artifacts, clean morphological splits (`preconstruction` → `pre` + `construction`), and deterministic word spacing. Vocab size is corpus-derived at ~5,300 morphemes. Generation coherence is unchanged: the model recites training fragments regardless of tokenizer. The root constraint is training data format, not tokenization strategy.

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

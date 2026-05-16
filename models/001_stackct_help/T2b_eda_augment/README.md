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
T2b_eda_augment/            # This experiment — EDA synonym augmentation on raw articles
├── core/                   # Model architecture, config, and tokenizer
│   ├── config.py           # All hyperparameters
│   ├── model.py            # Transformer: Head, MultiHeadAttention, Block, SmallLanguageModel
│   └── tokenizer_utils.py  # Byte-level BPE tokenizer (carried forward from T1d)
│
├── step_1_training/       # Data collection, augmentation, and model training
│   ├── retrieve_data.py    # Retrieves STACK support articles via Zendesk API
│   ├── augment_eda.py      # Applies EDA synonym replacement at 10% and 20% ratios
│   └── train.py            # Training loop with scheduler and gradient clipping
│
├── step_2_generation/     # Interactive inference
│   └── generate.py         # Chat REPL using the trained model
│
├── step_3_conversion/     # Export for deployment
│   ├── convert_to_gguf.py  # Converts .pth checkpoint to GGUF format
│   └── modelfile-ollama    # Ollama Modelfile with system prompt
│
├── gen/                    # Generated artifacts (gitignored)
│   ├── training.txt        # Raw articles from Zendesk API
│   ├── training_augmented.txt  # Original + 10% + 20% augmented corpus (~3x size)
│   └── model_final.pth     # Trained model checkpoint
└── Makefile                # All workflow commands
```

---

## Experiment Design

T1a–T1e trained on raw articles (~1.1M chars). This experiment keeps the same raw-article
format and tokenizer (byte-level BPE 10K from T1d) but expands the effective training
corpus using Easy Data Augmentation (EDA) synonym replacement.

**Hypothesis:** On a small corpus, synonym replacement reduces overfitting by exposing
the model to lexical variation while preserving sentence structure and meaning.

**Augmentation strategy:**
- For each word, with probability p, replace with a random WordNet synonym
- Ratios tested: 10% and 20% token swap
- Skip: stopwords, all-caps acronyms, numbers, domain brand names (STACK, Takeoff, etc.)
- Output: original corpus + 10% augmented + 20% augmented concatenated (~3× size)

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

### 3. Augment the corpus

```bash
make augment
```

Runs `augment_eda.py` on `gen/training.txt` and writes `gen/training_augmented.txt`
(original + 10% swap + 20% swap, ~3× the original size). Prints a sample diff so you
can verify synonym quality before training.

### 4. Train the model

```bash
make train
```

Trains on `gen/training_augmented.txt` for 10,000 steps.

### 5. Chat with the model directly

```bash
make generate
```

### 6. Convert to GGUF for Ollama

```bash
make convert && make ollama-load && make ollama-run
```

---

## Experiment Conclusions

See `results.md` for full analysis and comparison to T1d control.

EDA augmentation on raw articles made things **worse**. Synonym contamination leaked into generation output — the model samples augmented variants at inference time, producing outputs like "pauperization additional assistance" and "Create Exploiter New". No coherence improvement over T1d. The Q&A format finding from T2a remains the only effective intervention so far.

---

## Makefile Reference

| Command | Description |
|---|---|
| `make retrieve` | Fetch training data from STACK support site |
| `make augment` | Apply EDA synonym augmentation (produces training_augmented.txt) |
| `make train` | Clean all generated files and retrain from scratch |
| `make generate` | Interactive chat via native PyTorch inference |
| `make convert` | Convert checkpoint to GGUF |
| `make ollama-load` | Register model with Ollama |
| `make ollama-run` | Chat via Ollama |
| `make clean-gguf` | Remove GGUF and deregister from Ollama |
| `make clean-all` | Remove all generated files including checkpoints |

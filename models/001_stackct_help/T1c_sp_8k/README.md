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
T1c_sp_8k/                  # This experiment — SentencePiece Unigram tokenizer, 8K vocab
├── core/                   # Model architecture, config, and tokenizer
│   ├── config.py           # All hyperparameters
│   ├── model.py            # Transformer: Head, MultiHeadAttention, Block, SmallLanguageModel
│   └── tokenizer_utils.py  # SentencePiece tokenizer training and encode/decode helpers
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
| Parameters | ~5.8M |
| Vocabulary size | ~3,000 tokens |

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

## Experiment Conclusions

### Comparison to T1a/T1b (BPE baseline)

This experiment replaces ByteLevel BPE with SentencePiece Unigram tokenization at 3K vocab (corpus size limited the target 8K). Unlike BPE which merges character pairs bottom-up, Unigram starts with a large candidate set and prunes by likelihood — producing more linguistically motivated splits on domain terminology.

**Corpus-size constraint:**

The STACK support corpus (~1.1M chars, ~498 long-line articles) is too small to support an 8K Unigram vocabulary. After splitting articles into sentence-level chunks (yielding ~10,700 training sentences), the maximum achievable vocab was 3,116. Vocab was capped at 3,000.

**Tokenization artifacts:**

Unigram introduces a new class of output artifacts not seen in BPE experiments:

- *Suffix fragments at generation start* — e.g. `mly` appearing as the first output token. Unigram vocabularies contain pure-suffix pieces (no space prefix); when the model's first generated token is one of these, it renders as a bare fragment. This happens because the `<|ai|>` special token boundary doesn't guarantee the model lands at a word-initial position.
- *Boundary-less concatenation* — e.g. `beend` from tokens `been` + `d`. In SentencePiece, word-initial tokens carry a `▁` prefix; mid-word tokens don't. When two mid-word tokens are decoded adjacently, they concatenate directly with no space, producing spurious strings.

**Generation quality:**

No improvement over BPE baselines. The model recites training document fragments regardless of tokenizer algorithm. The artifacts above are a new failure mode but the root bottleneck remains model capacity and training data format.

**Conclusion:** Tokenizer algorithm (BPE vs Unigram) does not meaningfully affect generation coherence at this scale. Unigram introduces additional output artifacts due to its suffix-piece vocabulary structure. The limiting factor is not tokenization strategy.

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

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
T2c_eda_qa/                 # This experiment — EDA augmentation on Q&A answer spans
├── core/                   # Model architecture, config, and tokenizer
│   ├── config.py           # All hyperparameters
│   ├── model.py            # Transformer: Head, MultiHeadAttention, Block, SmallLanguageModel
│   └── tokenizer_utils.py  # Byte-level BPE tokenizer (carried forward from T1d)
│
├── step_1_training/       # Data collection, synthesis, augmentation, and training
│   ├── retrieve_data.py    # Retrieves STACK support articles via Zendesk API
│   ├── synthesize_qa.py    # Converts raw articles into Q&A pairs (from T2a)
│   ├── augment_qa_eda.py   # Applies EDA to answer spans only — never questions
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
│   ├── qa_pairs.txt        # Synthesized Q&A pairs (from synthesize_qa.py)
│   ├── qa_pairs_augmented.txt  # Original + 10% + 20% augmented pairs (~3x)
│   └── model_final.pth     # Trained model checkpoint
└── Makefile                # All workflow commands
```

---

## Experiment Design

Combines the two data-prep findings so far:
- **T2a** showed Q&A format is the only intervention that improved coherence
- **T2b** showed EDA on raw articles introduces synonym contamination in UI copy

This experiment applies EDA augmentation only to the `<|ai|>...<|end|>` answer spans
of the synthesized Q&A pairs. Questions are never modified. Domain nouns (STACK, Takeoff,
Acumatica, etc.) are skipped even in answer spans.

**Hypothesis:** Augmenting answer text in domain prose context is safer than augmenting
UI copy, and the 3× corpus expansion may improve the model's on-topic rate beyond T2a's
3/5 prompts.

**Pipeline:**
1. `synthesize_qa.py` → 225 Q&A pairs (`qa_pairs.txt`)
2. `augment_qa_eda.py` → 675 pairs at 10% and 20% swap (`qa_pairs_augmented.txt`)
3. `train.py` → trains on augmented corpus

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

### 3. Synthesize Q&A pairs

```bash
make synthesize
```

### 4. Augment answer spans

```bash
make augment
```

Applies EDA at 10% and 20% swap ratios to answer spans only. Prints sample diffs
so you can verify synonym quality before training.

### 5. Train the model

```bash
make train
```

Trains on `gen/qa_pairs_augmented.txt` for 10,000 steps.

### 6. Chat with the model directly

```bash
make generate
```

---

## Experiment Conclusions

See `results.md` for full analysis and comparison to T2a control.

EDA augmentation on answer spans is **worse than T2a** (1/5 on-topic vs 3/5). Synonym contamination still leaks into generation — WordNet resolves domain words to wrong senses (`List` → `Inclination`, `assembly` → `burlesque`). EDA is ruled out entirely. T2a's 225 plain Q&A pairs remain the best result; next step is increasing pair volume through better synthesis, not lexical augmentation.

---

## Makefile Reference

| Command | Description |
|---|---|
| `make retrieve` | Fetch training data from STACK support site |
| `make synthesize` | Generate Q&A pairs from raw articles |
| `make augment` | Apply EDA to answer spans (produces qa_pairs_augmented.txt) |
| `make train` | Clean all generated files and retrain from scratch |
| `make generate` | Interactive chat via native PyTorch inference |
| `make convert` | Convert checkpoint to GGUF |
| `make ollama-load` | Register model with Ollama |
| `make ollama-run` | Chat via Ollama |
| `make clean-gguf` | Remove GGUF and deregister from Ollama |
| `make clean-all` | Remove all generated files including checkpoints |

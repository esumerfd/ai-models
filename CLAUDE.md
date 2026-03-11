# AI Models — Exploration Repo

## Goal

This repository is a learning ground for building and exploring language models from scratch.
Each model in `models/` represents a self-contained experiment using a different technique,
architecture, dataset, or deployment target.

The aim is to understand the underlying mechanics of how language models work — not to use
pre-built solutions, but to build, train, and deploy each model by hand.

## Structure

Models are organized as numbered directories under `models/`:

```
models/
├── 001_stackct_help/   # GPT-style transformer trained on STACK support data
└── ...                 # Future experiments
```

Each model directory is self-contained with its own:
- Training pipeline
- Generation/inference code
- Dependencies (`requirements.txt`)
- Documentation

## Techniques to Explore

- Causal transformer (GPT-style) from scratch — **001**
- Different tokenization strategies (BPE, WordPiece, character-level)
- Varying model scales and hyperparameter tuning
- Alternate architectures (RNN, LSTM, sparse attention)
- Quantization and export (GGUF, ONNX)
- Domain-specific fine-tuning

## Deployment Target

Models are intended to run on a Raspberry Pi cluster:
`/Users/esumerfd/GoogleDrive/edward/Personal/projects/ai/pi-cluster`

Hardware: **Raspberry Pi AI HAT+ 2** — Hailo-10H chip (10 TOPS)

## Conventions

- Python 3.12 with a per-model `.venv`
- Each model has a `Makefile` as the primary workflow interface
- Generated artifacts (`gen/`) are gitignored

# AI Models

![AI Models Banner](docs/images/banner.jpeg)

A collection of language model experiments built from scratch — each one exploring a different
technique, architecture, or dataset.

The goal is hands-on understanding: no pre-built pipelines, just raw implementation, training,
and deployment.

> All content is suspect — this repo is created by someone who knows nothing... yet.

---

## Models

### [001 — StackCT Help](models/001_stackct_help/)

GPT-style causal transformer built from scratch with PyTorch, trained on the
[STACK](https://www.stackct.com) construction estimating platform support knowledge base.
Implements BPE tokenization, multi-head self-attention, and causal language modelling
with a ~5.8M parameter model targeting Raspberry Pi deployment via GGUF/Ollama.

→ [Full details and setup instructions](models/001_stackct_help/README.md)

---

## Deployment Target

All models are built to run on a Raspberry Pi cluster using the
**Raspberry Pi AI HAT+ 2** (Hailo-10H, 10 TOPS).

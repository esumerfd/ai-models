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

**Tokenization**
- [T1a — BPE 2K](models/001_stackct_help/T1a_bpe_2k/README.md)
- [T1b — BPE 5K](models/001_stackct_help/T1b_bpe_5k/README.md)
- [T1c — SentencePiece 8K](models/001_stackct_help/T1c_sp_8k/README.md)
- [T1d — Byte-level BPE 10K](models/001_stackct_help/T1d_bbpe_10k/README.md)
- [T1e — Morfessor (morphological)](models/001_stackct_help/T1e_morph/README.md)

**Data Preparation**
- [T2a — Synthesized Q&A pairs](models/001_stackct_help/T2a_qa_synth/README.md)
- [T2b — EDA augmentation](models/001_stackct_help/T2b_eda_augment/README.md)
- [T2c — EDA + Q&A](models/001_stackct_help/T2c_eda_qa/README.md)

**Architecture**
- [T3a — Seq2Seq (encoder-decoder)](models/001_stackct_help/T3a_seq2seq/README.md)
- [T3b — BERT retrieval](models/001_stackct_help/T3b_bert_retrieval/README.md)

---

## Deployment Target

All models are built to run on a Raspberry Pi cluster using the
**Raspberry Pi AI HAT+ 2** (Hailo-10H, 10 TOPS).

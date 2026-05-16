# Experiment Plan — STACK Support Language Models

Build and compare small language models trained on [STACK](https://www.stackct.com)
construction estimating platform support articles. Goal: answer domain-specific questions,
deployable to a Raspberry Pi AI HAT+ 2 (Hailo-10H, 10 TOPS).

All experiments share the same evaluation criteria and hardware constraint.
The goal is learning, not production deployment.

---

## Status Legend

| Symbol | Meaning |
|---|---|
| ✅ complete | Experiment run, results captured in `results.md` |
| 🔄 in progress | Currently being built or trained |
| ⏳ pending | Planned, not yet started |
| ⏸ deferred | Out of scope for now |

---

## Experiment Status

### Baseline

| ID | Description | Status | Finding |
|---|---|---|---|
| 001 | Causal LM (GPT-style), BPE 2K tokenizer, plain-text corpus | ✅ complete | Reference point for all comparisons. 3/5 on-topic responses. |

---

### Tokenization Variants
*Same causal architecture and corpus — varies only the tokenizer.*

| ID | Tokenizer | Vocab | Status | Finding |
|---|---|---|---|---|
| T1a | Byte-level BPE | 2K | ✅ complete | Baseline. High UNK rate on domain terms. |
| T1b | Byte-level BPE | 5K | ✅ complete | Improved coverage, similar quality. |
| T1c | SentencePiece Unigram | 8K (corpus-limited to ~3K) | ✅ complete | Corpus too small for full 8K vocab. |
| T1d | Byte-level BPE | 10K | ✅ complete | **Best tokenizer.** Zero UNK, best coverage. Carried forward. |
| T1e | Morfessor (morphological) | corpus-derived | ✅ complete | Linguistically motivated but no quality improvement over BPE. |

---

### Data Preparation Variants
*Same architecture (causal, BBPE 10K) — varies how training data is prepared.*

| ID | Technique | Status | Finding |
|---|---|---|---|
| T2a | Synthesized Q&A pairs (Claude-generated) | ✅ complete | **Best generative result.** 3/5 on-topic. Training data format matches inference task. |
| T2b | EDA synonym augmentation (3× corpus expansion) | ✅ complete | Worse than T2a. Synonym contamination degrades domain vocabulary. |
| T2c | EDA augmentation applied to Q&A pairs | ✅ complete | Worse than T2a. EDA ruled out — augmentation degrades quality at this scale. |
| DR-A | Markdown-preserved (HTML → Markdown instead of stripping) | ⏳ pending | — |
| DR-C | Chunked sections (split articles at heading boundaries) | ⏳ pending | — |
| DR-D | Markdown + front matter (title/category/product metadata) | ⏳ pending | — |

**DR-B** (Structured QA pairs) is covered by T2a. **DR-E** (JSON Lines) is deferred — token budget wasted on syntax.

---

### Architecture Variants
*Same corpus (T2a Q&A pairs) — varies the model architecture.*

| ID | Architecture | Params | Status | Finding |
|---|---|---|---|---|
| T3a | T5-style encoder-decoder (seq2seq) | 5.1M | ✅ complete | 0/5 on-topic. Cross-attention requires ~100× more data to converge than available. |
| T3b | BERT-style encoder + cosine retrieval | 5.8M | ✅ complete | 0/7 on-topic. MLM-only pretraining produces collapsed embeddings — no retrieval signal without contrastive training. |

---

### Fine-Tuning
*Load a small pretrained open-source model and adapt it to the STACK domain.*

| ID | Technique | Base Model | Status | Finding |
|---|---|---|---|---|
| T4a | LoRA (Low-Rank Adaptation) | TBD (Qwen-0.5B or similar) | ⏳ pending | — |
| T4b | DPO (Direct Preference Optimization) | TBD | ⏳ pending | Requires LoRA baseline first. |

---

### Additional Training Techniques (from original plan)
*Architectures defined in `plan.md` — status reflects current prioritization.*

| ID | Technique | Status | Notes |
|---|---|---|---|
| 004 | ELECTRA (replaced token detection) | ⏸ deferred | Interesting for sample efficiency, but two-model training overhead not justified yet. |
| 005 | LSTM language model | ⏸ deferred | Useful architecture baseline — deferred in favour of transformer variants. |
| 006 | SSM / Mamba | ⏸ deferred | Requires CUDA; pure-PyTorch implementation needed for Pi deployment. |
| 007 | Prefix language model | ⏸ deferred | Minimal change from baseline — could be run as a T1x variant. |
| 008 | Denoising autoencoder (BART-style) | ⏸ deferred | Builds on seq2seq (T3a); revisit if more data available. |

---

## Combination Matrix

The data preparation and fine-tuning dimensions interact. This table tracks
which combinations are worth running and their priority.

| Input Format | Training Approach | Experiment | Priority |
|---|---|---|---|
| Plain text (T2a Q&A) | From-scratch causal SLM | T2a | ✅ done — reference |
| Plain text | LoRA fine-tuning | T4a | ⏳ high — next in sequence |
| Markdown (DR-A) | From-scratch causal SLM | DR-A baseline | ⏳ medium — cheap format validation |
| Markdown (DR-A) | LoRA fine-tuning | DR-A + LoRA | ⏳ medium — best-format + fine-tuning |
| Chunked (DR-C) | From-scratch causal SLM | DR-C baseline | ⏳ low — run if DR-A shows improvement |
| Markdown + metadata (DR-D) | From-scratch causal SLM | DR-D baseline | ⏳ low — metadata routing test |
| Plain text | DPO | T4b | ⏳ planned — after LoRA baseline |

**Sequencing logic:** Run DR-A from-scratch first (cheap, reuses existing pipeline).
If DR-A improves over T2a, carry Markdown format into LoRA. If neutral, go straight
to T4a (plain + LoRA). DPO follows after a LoRA baseline exists.

---

## Results Summary

| Experiment | On-topic rate | Failure mode |
|---|---|---|
| T1d / T2a (best from-scratch) | **3 / 5** | Hallucination — answers shaped but partially wrong |
| T3a seq2seq | 0 / 5 | Repetition loops — cross-attention did not converge |
| T3b BERT retrieval | 0 / 7 | Collapsed embeddings — no retrieval signal from MLM alone |

The causal decoder-only model (T2a) remains the best result across all from-scratch
experiments. The hypothesis that more complex architectures (seq2seq, retrieval) would
outperform it failed at this data scale (~225 Q&A pairs, ~1.1M character corpus).
Fine-tuning a pretrained model (T4a LoRA) is the most promising next step.

---

## Evaluation Criteria

Each experiment is evaluated on:

| Metric | Method |
|---|---|
| On-topic rate | Manual review: does the response correctly address the question? (n=5–7 standard questions) |
| Training loss | Final train/val loss |
| Response shape | Does output stop cleanly, loop, or hallucinate? |
| Model size | Parameter count |

Quantitative metrics (perplexity, BLEU) are not used — domain is too narrow and
responses too short for these metrics to be meaningful.

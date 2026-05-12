# T3b_bert_retrieval Results

**Experiment:** BERT-style bidirectional encoder, MLM pretraining, cosine similarity retrieval
**Date:** 2026-05-09
**Model:** BertEncoder, 5.8M params, trained 10,000 steps on 201 articles via MLM
**Tokenizer:** Byte-level BPE 10K (T1d/T2a)
**Control:** T2a_qa_synth — causal decoder-only, 225 Q&A pairs, same corpus

---

## Architecture Summary

| Component | Detail |
|---|---|
| Encoder | 6 layers, bidirectional self-attention |
| Pretraining | MLM, 15% token masking |
| Pooling | Mean of non-padding encoder outputs |
| Inference | Encode query → cosine similarity → top-3 articles |
| Parameters | 5.8M |

---

## Retrieval Results

Test questions are identical to those used in T2a and T3a evaluations.

---

**USER:** How do I create a takeoff?

**Top-1:** # Using the STACK Roofing Catalog (score=0.867) — **irrelevant**
**Top-2:** # STACK Keyboard Shortcuts (score=0.860) — irrelevant
**Top-3:** # STACK Quick Start Assemblies - Division 9 (score=0.846) — irrelevant

---

**USER:** What is a change order?

**Top-1:** # Using the STACK Roofing Catalog (score=0.870) — **irrelevant**
**Top-2:** # Locate a Measurement (score=0.859) — irrelevant
**Top-3:** # Your Personal Data (score=0.858) — irrelevant

---

**USER:** How do I add a subcontractor?

**Top-1:** # Using the STACK Roofing Catalog (score=0.872) — **irrelevant**
**Top-2:** # STACK Keyboard Shortcuts (score=0.872) — irrelevant
**Top-3:** # Locate a Measurement (score=0.866) — irrelevant

---

**USER:** What is STACK?

**Top-1:** # Using the STACK Roofing Catalog (score=0.874) — **irrelevant**
**Top-2:** # Export Your STACK Estimate to QuickBooks Desktop (score=0.861) — irrelevant
**Top-3:** # Your Personal Data (score=0.848) — irrelevant

---

**USER:** How do I export a bid?

**Top-1:** # Using the STACK Roofing Catalog (score=0.873) — **irrelevant**
**Top-2:** # STACK Keyboard Shortcuts (score=0.867) — irrelevant
**Top-3:** # Locate a Measurement (score=0.840) — irrelevant

---

**USER:** How do I delete a project?

**Top-1:** # Using the STACK Roofing Catalog (score=0.863) — **irrelevant**
**Top-2:** # STACK Keyboard Shortcuts (score=0.861) — irrelevant
**Top-3:** # Locate a Measurement (score=0.851) — irrelevant

---

**USER:** What is the Audit Trail?

**Top-1:** # Using the STACK Roofing Catalog (score=0.880) — **irrelevant**
**Top-2:** # Locate a Measurement (score=0.871) — irrelevant
**Top-3:** # Issue Multiple Assignee Matrix (score=0.864) — irrelevant

---

## Failure Mode Analysis

**Collapsed embeddings (anisotropy):** Every query returns "Using the STACK Roofing Catalog" as top-1, regardless of question content. Score spread across all 201 articles is extremely narrow (0.84–0.88). This is the classic embedding collapse failure: all article vectors cluster in a small region of embedding space, making cosine similarity essentially random with respect to semantic content.

**MLM ≠ retrieval:** MLM pretraining trains the model to predict masked tokens at each position — a token-level objective. Mean-pooling token representations from an MLM model produces vectors that reflect statistical token distributions, not sentence-level semantics. Without contrastive training (e.g. SimCSE, DPR) on (query, relevant-article) pairs, the encoder has no objective that aligns query and article embeddings into a shared semantic space.

**Score compression:** All cosine similarities cluster between 0.84 and 0.88. A working retrieval system would show a clear gap between the relevant article (score ~0.9+) and irrelevant ones (score ~0.5–0.7). The narrow band confirms the embeddings carry no discriminative signal.

**On-topic rate: 0 / 7** — no query returned its relevant article at any rank in the top-3.

---

## Comparison to Generative Baselines

| Dimension | T2a causal (control) | T3a seq2seq | T3b BERT retrieval (this) |
|---|---|---|---|
| Architecture | Decoder-only, causal | Encoder-decoder, cross-attention | Encoder-only, bidirectional |
| Approach | Generation | Generation | Retrieval |
| Parameters | 7.0M | 5.1M | 5.8M |
| Training signal | 225 Q&A pairs | 225 Q&A pairs | 201 articles (MLM) |
| On-topic rate | **3 / 5** | 0 / 5 | 0 / 7 |
| Failure mode | Hallucination | Repetition loops | Collapsed embeddings |
| Overall | **Best so far** | Worse than T2a | **Worst so far** |

---

## Root Cause

Retrieval with a from-scratch MLM encoder requires contrastive fine-tuning to work. The three requirements are:

1. **Pretrained representations** — MLM provides this, but weakly at our data scale
2. **Shared embedding space** — queries and articles must be aligned so similar meanings produce similar vectors. MLM does not provide this; it only trains token-level prediction.
3. **Contrastive signal** — the model needs (query, positive article, negative articles) triplets to learn to push relevant articles closer to their queries. We have no such labelled pairs.

Production retrieval systems (DPR, ColBERT, E5) are either fine-tuned on tens of thousands of (query, passage) pairs, or distilled from a much larger pre-trained language model. Neither path is available at our data scale from scratch.

---

## Conclusion

BERT retrieval from scratch is not viable at this data scale without contrastive training data. The MLM-only pretrained encoder produces degenerate embeddings that cannot discriminate between articles. The causal decoder-only model (T2a) remains the best result across all architecture experiments.

**Retrieval is the right long-term direction** — returning a factually correct article is strictly better than generating a hallucinated answer — but it requires either:
- A large pretrained encoder (e.g. sentence-transformers) fine-tuned on STACK Q&A pairs, or
- Synthetic (query, article) pairs generated from the corpus to enable contrastive training

Both require capabilities beyond the from-scratch training scope of this project.

**Next: fine-tuning phase (T4a LoRA)** — apply parameter-efficient fine-tuning to a pretrained base model rather than training from scratch.

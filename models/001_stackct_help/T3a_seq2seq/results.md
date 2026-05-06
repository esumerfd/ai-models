# T3a_seq2seq Results

**Experiment:** T5-style encoder-decoder transformer, trained from scratch
**Date:** 2026-05-05
**Model:** Seq2Seq transformer, 5.1M params, trained 10,000 steps on 225 Q&A pairs
**Tokenizer:** Byte-level BPE 10K (T1d/T2a)
**Control:** T2a_qa_synth — causal decoder-only, same corpus, same tokenizer

---

## Architecture Summary

| Component | Detail |
|---|---|
| Encoder | 3 layers, bidirectional self-attention |
| Decoder | 3 layers, causal self-attention + cross-attention to encoder |
| Embedding | Shared across encoder, decoder, output projection |
| Inference | Encode question once → greedy-decode answer |
| Parameters | 5.1M |

---

## Generation Samples

---

**USER:** How do I create a takeoff?

**AI (decoded):** STACK General you can help and manage your browser. It is usually of all the Estimate Worksheet: that does to create more, you can help consuming consuming and access to create and integrity them. login login takeoff from Completed Completed Completed the Estimate Worksheet c c c c c c you can use the connection/count from STACK...

---

**USER:** What is a change order?

**AI (decoded):** We know prospective projects are coming to create their environment environment to within a project. Click to create. Click on the original to create, various project. Click, and permissions, you COLLABORATOR various various Connect Reports, please...

---

**USER:** How do I add a subcontractor?

**AI (decoded):** STACK's integration a project is designed to easily on your plan. The App from STACK, and no in order to generate a takeoff. This reducing of the need to the connection, and t store quantities to the connection or delete? so it is included of the connection is interrupted...

---

**USER:** What is STACK?

**AI (decoded):** Key Information The STACK Community is an online space where STACK | Build & Operate. STACK on the STACK Field App from real- plans to) and subscription on- Inches you create your uploaded, and calculate the STACK Field App...

---

**USER:** How do I export a bid?

**AI (decoded):** The can table a project of to the STACK on your STACK Field App, and access to the takeoffs to the You can export that take a sales convert and time...

---

**USER:** How do I delete a project?

**AI (decoded):** The Daily Reports are part of the project plans and Labor your STACK | Build & Operate can access your organization... attending attending attending or data from...

---

**USER:** What is the Audit Trail?

**AI (decoded):** Key Information The dent who in STACK Field App is where you can be accessed in the produce... count count count count count plant plant plant plant plant plant plant...

---

## Failure Mode Analysis

**Repetition loops:** Greedy decoding with an undertrained model collapses into repetition — `c c c c c c`, `attending attending attending`, `count count count count count`, `launch launch launch`. This is the dominant failure mode: the decoder runs out of learned signal and samples the highest-probability token repeatedly.

**Partial signal:** Two responses open with `Key Information` — the training pairs frequently begin answers with this phrase. The model learned the opening pattern but cannot sustain coherent content beyond the first few tokens.

**On-topic rate:** 0/5 — no response correctly answers its question. One (`What is STACK?`) starts promisingly but immediately degrades.

---

## Comparison to T2a (Causal Baseline)

| Dimension | T2a causal (control) | T3a seq2seq (this) |
|---|---|---|
| Architecture | Decoder-only, causal | Encoder-decoder, cross-attention |
| Parameters | 7.0M | 5.1M |
| Training pairs | 225 | 225 |
| On-topic rate | **3 / 5** | 0 / 5 |
| Response shape | Answer-shaped, stops cleanly | Repetition loops, fills budget |
| Coherence | Best seen so far | Worst seen across all experiments |
| Overall | Better | **Worse than T2a** |

---

## Root Cause

Encoder-decoder architectures require significantly more data than decoder-only models trained on the same task. With only 225 pairs:

- The causal model can leverage the full token sequence (question + answer concatenated) as a single language modelling objective — every token in every pair contributes to the gradient.
- The seq2seq model must learn cross-attention alignment between encoded questions and decoded answers. With 225 examples, the cross-attention heads do not converge — the decoder ignores the encoder output and degrades to unconditional token prediction, which collapses to repetition under greedy decoding.

A T5-style model typically requires tens of thousands of training examples at minimum. 225 pairs is approximately two orders of magnitude too small.

---

## Conclusion

Seq2seq architecture from scratch is not viable at this data scale. The causal decoder-only model (T2a) is strictly better on 225 Q&A pairs because it has a simpler inductive bias that converges with less data.

The architecture hypothesis — that bidirectional encoding improves answer quality — is plausible but cannot be tested without substantially more training data. This is not a failure of seq2seq as an approach; it is a data-volume failure.

**Next architecture experiment (T3b BERT retrieval) is a retrieval approach and does not require Q&A pairs — it trains on MLM over the raw article corpus, which is much better matched to our data volume.**

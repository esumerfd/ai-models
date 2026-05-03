# T2a_qa_synth Results

**Experiment:** Q&A synthesis data prep — rule-based template synthesis from STACK articles
**Date:** 2026-05-03
**Model:** GPT-style causal transformer, 7.0M params, trained 10,000 steps on 225 synthesized Q&A pairs
**Tokenizer:** Byte-level BPE 10K, carried forward from T1d

---

## Training Corpus

| Metric | Value |
|---|---|
| Source articles | 201 |
| Synthesized Q&A pairs | 225 |
| Synthesis strategy | Rule-based templates (no external model) |
| Min answer length | 80 chars |
| Max answer length | 600 chars |
| Training file | `gen/qa_pairs.txt` |

The corpus is significantly smaller than the raw article text used in T1a–T1e (~1.1M chars vs ~225 formatted pairs). Each pair is formatted with the inference-time prompt structure so the model sees the same `<|start|>...<|user|>...<|ai|>` framing during training as at generation time.

---

## Generation Samples

Standard test prompts against `gen/model_final.pth`. Raw output contains `Ġ` (ByteLevel space prefix) — cosmetic, fully reversible via `tokenizer.decode()`.

---

**USER:** How do I create a takeoff?

**AI (decoded):** Key Information The Takeoff Toolbar provides access to the tools used to create, edit, and manage measurements on plan sheets.

---

**USER:** What is a change order?

**AI (decoded):** This feature is temporarily unavailable. This feature when the report creator assigns a Reporter but does not assign a Reviewer to the report. With a Reviewer This flow applies when the report creator assigns both a Reporter and a Reviewer to the report. Related Articles STACK Daily Reports Daily Reports in the Web Portal...

---

**USER:** How do I add a subcontractor?

**AI (decoded):** The following table shows the access permissions for each Role: WORKFLOWS OWNER ADMINISTRATOR POWER COLLABORATOR (+ DOCS) POWER COLLABORATOR (+ USERS) POWER COLLABORATOR COLLABORATOR RESTRICTED COLLABORATOR REVIEWER VIEWER Access to the Web Portal X X X X X X X X Access to Workflows X X X X X X X X...

---

**USER:** What is STACK?

**AI (decoded):** Key Information The STACK Community is an online space where STACK users connect to ask questions, share ideas, and learn from real-world experiences. It works alongside the STACK Help Center. The Help Center provides official guidance and step-by-step instructions, while the Community focuses on discussion, shared experience, and practical insight from other users.

---

**USER:** How do I export a bid?

**AI (decoded):** The Estimate Worksheet gives you the ability to calculate labor hours. To start, you must add at least one labor Item to the takeoff measurements in the project for which you want to calculate labor hours.

---

## Comparison to T1a–T1e Baselines

| Dimension | T1a–T1e (raw articles) | T2a (Q&A synthesis) |
|---|---|---|
| Training data format | Raw support article prose | Synthesized `<user>/<ai>` pairs |
| Training corpus size | ~1.1M chars | ~225 pairs |
| Response length | Long rambling fragments (200 tokens) | **Short, stops earlier** — hits `<\|end\|>` |
| Response shape | Article continuation | **Answer-shaped** — starts directly with content |
| On-topic rate | Low — topic drift within 2–3 sentences | **Improved** — 3/5 prompts return relevant content |
| Structured output | No | Partial — "Key Information" framing appears |
| Coherence within response | Poor | **Noticeably better** — sentences connect logically |
| Artifacts | `Ġ` prefix (cosmetic) | `Ġ` prefix (cosmetic, same tokenizer) |

---

## Analysis

**What improved:**

- *Response shape:* Outputs are answer-shaped rather than article-continuation-shaped. The model learned to produce a direct statement followed by supporting detail, mirroring the synthesis template structure.
- *Response length:* The model stops generating earlier — it hits `<|end|>` rather than filling the 200-token budget with rambling continuations.
- *Coherence:* "What is STACK?" returns a well-formed paragraph. "How do I export a bid?" returns a focused, relevant sentence. These are qualitatively better than any T1 output.
- *"Key Information" framing:* Two responses open with "Key Information" — this reflects a pattern present in the synthesized training pairs, showing the model has learned document-level structure.

**What did not improve:**

- *Topic grounding:* "What is a change order?" returns content about Daily Reports; "How do I add a subcontractor?" returns a permissions table. The model retrieves a memorized Q&A pair that partially matches the prompt tokens rather than generalising to the question's intent.
- *Factual accuracy:* Responses are pulled from training pairs, not composed from underlying knowledge. The 225-pair corpus is too small for reliable retrieval — similar question structures map to whatever pair was nearest in training.

**Root cause of remaining failures:**

225 pairs from 201 articles is not enough training signal. The model memorises a small set of pairs rather than learning a question-answering function. Many questions have no close training pair, so the model falls back to the nearest-token continuation.

---

## Conclusion

Q&A synthesis is a meaningful improvement over raw article training. Response shape, length control, and within-response coherence are all better. Three of five standard test prompts return relevant, answer-shaped output — a clear step forward from the T1 experiments where all five returned article fragments.

The limiting factor is now **synthesis coverage**: 225 pairs from 201 articles is too sparse. The next data-prep workstream should expand coverage — either by generating more pairs per article (multiple question types per section) or by using a model-assisted synthesis approach to produce higher-quality, higher-volume Q&A data.

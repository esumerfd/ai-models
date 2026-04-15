# Training Data Input Formats for Tokenizer Optimisation

**Date:** 2026-04-13
**Context:** wk-models / `001_stackct_help` and future experiments
**Source data:** STACK Help Center articles via Zendesk Help Center API

---

## The Problem: Plain Text Loses Structure

The Zendesk API returns rich HTML. The current `retrieve_data.py` strips it to plain text. All structural signal is discarded.

### What Is Lost Today

| HTML Element | Current Output | Lost Signal |
|-------------|---------------|-------------|
| `<h2>`, `<h3>` | stripped | Section hierarchy — only `<h1>` (title) is kept |
| `<p>` | blank line | Paragraph boundaries are implicit, not marked |
| `<ul>`, `<ol>`, `<li>` | space-separated text | List structure |
| `<strong>`, `<em>` | stripped | Emphasis — often marks key terms in support docs |
| `<code>`, `<pre>` | stripped | Commands and code — high-signal for support content |
| `<table>` | stripped entirely | Structured data: feature comparisons, settings, options |
| `<a href>` | link text only | Cross-references between articles |
| Article metadata | none | Category, tags, last updated date |
| `<div class="note">`, `<div class="warning">` | stripped | Callouts — high-value content for Q&A |

### Why This Matters for Tokenization

On a small corpus (~1.1M characters), statistical subword methods (BPE, Unigram) cannot infer structure from frequency alone. They need to see structural markers to learn them. A plain text dump trains the tokenizer and model to treat all text as undifferentiated prose — the model learns to continue help articles, not to answer questions.

---

## Input Formats

### Format 1 — Markdown

Convert HTML → Markdown using `html2text` or `markdownify`. Minimal code change to `retrieve_data.py`, maximum structural preservation.

**Example output:**

```markdown
# How to Reset Your Password

## Steps

1. Click **Account Settings** in the top navigation.
2. Select **Security**.
3. Click **Reset Password** and follow the prompts.

> **Note:** You must have access to your registered email address.

## Related Articles

- [Two-Factor Authentication](...)
- [Account Lockout Policy](...)
```

**Tokenizer benefit:**
- BPE/Unigram see `##`, `**`, `>`, `-`, `` ` `` as recurring structural markers
- The model learns section boundaries, emphasis, and list structure as first-class patterns
- No vocabulary extension required — Markdown syntax becomes high-frequency subwords naturally

**Implementation:**
```python
import html2text
converter = html2text.HTML2Text()
converter.ignore_links = False
text = converter.handle(article['body'])
output = f"# {article['title']}\n\n{text}"
```

---

### Format 2 — Structured Markdown with Boundary Tokens

Extend Markdown with explicit special tokens that make structural boundaries unambiguous. Best suited to small corpora where frequency-based subword learning is insufficient.

**Example output:**

```
<|article|>
<|meta|> category: Account Management | tags: password, security | updated: 2024-01
<|title|> How to Reset Your Password
<|section|> Steps
<|step|> Click **Account Settings** in the top navigation.
<|step|> Select **Security**.
<|step|> Click **Reset Password** and follow the prompts.
<|note|> You must have access to your registered email address.
<|section|> Related Articles
<|ref|> Two-Factor Authentication
<|ref|> Account Lockout Policy
<|end-article|>
```

**Tokenizer benefit:**
- Every structural boundary is an explicit, learnable token added to the vocabulary
- `<|section|>`, `<|note|>`, `<|step|>`, `<|warning|>`, `<|meta|>` become first-class vocabulary entries
- The model learns structural transitions explicitly, not statistically
- Especially effective on small corpora (~1.1M chars) where statistical inference is weak

**Suggested special tokens:**
| Token | Marks |
|-------|-------|
| `<|article|>` / `<|end-article|>` | Article boundaries |
| `<|meta|>` | Article metadata (category, tags, date) |
| `<|title|>` | Article title |
| `<|section|>` | Section heading (h2/h3) |
| `<|step|>` | Numbered step in a procedure |
| `<|note|>` | Informational callout |
| `<|warning|>` | Warning callout |
| `<|code|>` | Inline code or command |
| `<|ref|>` | Link/cross-reference to another article |

---

### Format 3 — Q&A Instruction Format

Synthesise question-answer pairs directly from article structure. Section headings become question candidates; section body becomes the answer. The training format exactly matches the inference format — no prompt-engineering gap.

**Example output:**

```
<|start|>
<|system|> You are a STACK support assistant. Answer based only on your training data.
<|user|> How do I reset my password?
<|context|> From "Account Management > Security": Click Account Settings, select Security, click Reset Password.
<|ai|> To reset your password: 1. Go to Account Settings. 2. Select Security. 3. Click Reset Password and follow the email prompt.
<|end|>
```

**Tokenizer benefit:**
- Training and inference token sequences are identical — special tokens (`<|user|>`, `<|ai|>`) are seen during training, not injected only at inference
- The model learns to produce answers in the expected format, not just continue prose
- Q&A structure provides natural boundaries for context window packing

**Question synthesis strategies:**
1. Section heading → "How do I [heading]?" / "What is [heading]?"
2. Callout/note → "What should I know about [topic]?"
3. Numbered steps → "What are the steps to [title]?"
4. Metadata tags → Generate questions per tag/category cluster

---

### Format 4 — JSONL Instruction Format

Required for LoRA fine-tuning (experiment 4a) and DPO alignment (experiment 4b). Standard Alpaca/ShareGPT format.

**Example output (Alpaca):**
```json
{"instruction": "How do I reset my password?", "input": "", "output": "Go to Account Settings → Security → Reset Password. You will receive a reset email."}
{"instruction": "What happens if I enter my password wrong too many times?", "input": "", "output": "After 5 failed attempts, your account locks for 30 minutes. Contact support if you need immediate access."}
```

**Example output (DPO preference pairs):**
```json
{"prompt": "How do I reset my password?", "chosen": "Go to Account Settings → Security → Reset Password.", "rejected": "You can try clicking around in the settings area to find a password option."}
```

**Tokenizer benefit:** None directly — JSONL is a delivery format for fine-tuning frameworks (transformers, trl), not for from-scratch tokenizer training. Produce alongside Format 1 or 2.

---

## Recommended Changes to `retrieve_data.py`

Generate two output files per retrieval run:

| File | Format | Used For |
|------|--------|---------|
| `gen/training_markdown.txt` | Format 1 — Markdown | Tokenizer training (phases 1–2) |
| `gen/training_structured.txt` | Format 2 — Boundary tokens | Tokenizer training variant for small-corpus experiments |
| `gen/training_qa.jsonl` | Format 3 or 4 — Q&A / JSONL | Fine-tuning phases (3–4) |

**Minimal implementation path:**

```python
# Step 1 — Markdown (pip install html2text)
import html2text
h = html2text.HTML2Text()
h.ignore_links = False
markdown_body = h.handle(article['body'])

# Step 2 — Structured tokens (custom parser on top of Step 1)
structured = add_boundary_tokens(article, markdown_body)

# Step 3 — Q&A synthesis (section heading → question heuristic)
qa_pairs = synthesise_qa_pairs(article['title'], markdown_body)
```

---

## Impact on Experiment Sequence

| Experiment | Plain text ok? | Recommended format |
|-----------|---------------|-------------------|
| 1b–1e tokenizer variants | ✅ acceptable | Format 1 (Markdown) improves all |
| 2a Q&A synthesis (D5) | ❌ | **Format 3 — this IS the synthesis step** |
| 2b EDA augmentation (D1) | ✅ | Any format |
| 3a Seq2Seq | ❌ requires pairs | Format 3 |
| 3b BERT retrieval | ✅ | Format 1 or 2 for pre-training |
| 4a LoRA fine-tuning | ❌ requires JSONL | Format 4 (Alpaca) |
| 4b DPO alignment | ❌ requires pairs | Format 4 (DPO preference pairs) |

**Conclusion:** Format 2 (D5 Q&A synthesis / Template generation) is a prerequisite for experiments 3a, 4a, and 4b — not an optional augmentation. It should be implemented before those experiments begin.

---

## Linguistic Granularity Note

The format choice also affects where on the granularity spectrum the tokenizer operates:

```
Document → Section → Paragraph → Sentence → Word → Subword → Character → Byte
                                                        ↑
                                              BPE/Unigram operate here
```

Today's plain text gives the tokenizer no signal above the word level. Format 2 (boundary tokens) explicitly marks paragraph, section, and document boundaries — extending the tokenizer's effective granularity upward without changing the subword algorithm.

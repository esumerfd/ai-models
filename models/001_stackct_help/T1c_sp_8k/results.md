# T1c_sp_8k Results

**Experiment:** SentencePiece Unigram tokenizer, 3K vocab (corpus-limited from target 8K)
**Date:** 2026-04-29
**Model:** GPT-style causal transformer, 6.3M params, trained 10,000 steps on STACK support corpus

---

## Tokenizer: Subword Splits on Domain Terms

Vocabulary size: 3,000 tokens

| Term | Pieces | Notes |
|---|---|---|
| `construction` | `▁construction` | Whole word — high-frequency domain term |
| `estimating` | `▁estimating` | Whole word |
| `subcontractor` | `▁subcontractor` | Whole word — good domain coverage |
| `takeoff` | `▁takeoff` | Whole word |
| `changeorder` | `▁change` `order` | Splits compound (no space in input) |
| `RFI` | `▁R` `F` `I` | Acronym split to characters — OOV-style |
| `submittal` | `▁submit` `t` `al` | Morpheme-ish but suffix fragmented |
| `STACK` | `▁STACK` | Whole word — brand name in vocabulary |
| `preconstruction` | `▁pre` `c` `onstruction` | Prefix correct; stem fragmented |
| `bid` | `▁bid` | Whole word |
| `material` | `▁material` | Whole word |
| `labor` | `▁labor` | Whole word |

**Observations:**
- High-frequency domain nouns (`construction`, `estimating`, `takeoff`, `subcontractor`, `STACK`) are preserved as single tokens — vocabulary has good domain coverage at 3K.
- Acronyms (`RFI`) reduce to character sequences — the Unigram model has no mechanism for treating them as atomic units without explicit user-defined symbols.
- Prefix morphology is inconsistent: `▁pre` is learned but the stem `c + onstruction` still fragments.
- Suffix pieces appear mid-word (`t`, `al`, `c`) — a known Unigram artifact where suffix-only pieces enter the vocabulary.

---

## Generation Samples

Standard test prompts against `gen/model_final.pth`:

---

**USER:** How do I create a takeoff?

**AI:** s the data is added to a new measurements/counts that quantities, or measurements can also be created as expected. Apply a measurement betweened inputs, an expanded height of the arc. Use Rectangle Equ Yd edges to its measurements and shapes or even when same measurement points incorrect te, a measurement points at a label. TIP The same measurement includes a pill relude Use Cut Line tool, an existing measurement. TIP For existing measurement on a measurement by drawing a plan sheet instead of the line item. Cut Rectangle C Removes an area measurement. Click the ⁇ Flip Horizontal to start between sheets between all groups, press Delete and ends...

---

**USER:** What is a change order?

**AI:** STACK estimate contains data ⁇ You will be an individual measurement. Use Filter or Create a new Estimate or Create a new Estimate or create an estimate. ⁇ like manually when you add an estimate to a takeoff. (Fride the desired changes made) can change or count or with quantity or with notes related to estimates or takeoffs. Rotates Multiple groupings can be created to estimates for selections...

---

**USER:** How do I add a subcontractor?

**AI:** than one plan, and/or all basical input value). This setup is the currently want to show more accurate Item Name. The updated name can be applied to the Customied to that type. Using regional and remov cost data is an Item to fonyze generated in both. Open the cell for a project in STACK Build & Operate...

---

**USER:** What is STACK?

**AI:** - Directly found in your function. X Samed company name ⁇ This usually workflows, while work for each job cost data for the feature used in your team can be used formula, and can be Construction, likely any other projects...

---

**USER:** How do I export a bid?

**AI:** = mything changes the box is automatically generated and cannot be edited. STACK Field App: If you take it has been added to a project while signs being. STACK Connection Sage 300 CRE Job Material & Labor Material &ioger Sage 300 Before exporting your approved including...

---

## Artifact Analysis

Two artifact classes observed (not present in BPE experiments):

**1. Suffix-fragment generation start**
The first generated token is sometimes a bare suffix piece (e.g., `s`, `t`, `al`) with no word-initial `▁` marker. This happens because the `<|ai|>` boundary doesn't guarantee the model lands at a word-initial position in its learned distribution.

**2. Boundary-less concatenation**
Adjacent mid-word tokens decode without spaces, producing spurious strings (`betweened`, `Customied`, `fonyze`). In SentencePiece, word-initial tokens carry `▁`; mid-word tokens don't. Two consecutive mid-word tokens concatenate directly.

The `⁇` artifact (replacement character) indicates the tokenizer encountered token IDs it couldn't decode cleanly — likely boundary tokens surfacing in unexpected positions.

---

## Comparison to BPE Baselines (T1a 2K, T1b 5K)

| Dimension | T1a BPE 2K | T1b BPE 5K | T1c SP 3K |
|---|---|---|---|
| Tokenizer | ByteLevel BPE | ByteLevel BPE | SentencePiece Unigram |
| Vocab size | 2,000 | 5,000 | 3,000 |
| Domain terms as whole tokens | Partial | Better | Good |
| Acronym handling | Character-merged via BPE | Character-merged | Character-split |
| Artifact type | `Ġ` prefix on every token | `Ġ` prefix on every token | Suffix fragments, `⁇` replacement chars |
| Generation coherence | Recites training fragments | Recites training fragments | Recites training fragments |
| Improvement over prior | — | Marginal | None |

---

## Conclusion

Tokenizer algorithm (BPE vs Unigram) does not affect generation coherence at this model scale. All three experiments produce the same failure mode: recitation of training document fragments with no generalization.

Unigram at 3K offers slightly better domain-term preservation than BPE at 2K, but introduces new artifact classes (suffix fragments, boundary-less concatenation) not seen in BPE. These artifacts are harder to post-process than BPE's `Ġ` prefix.

**The limiting factor is not tokenization strategy.** The root constraint is model capacity (~6M params) and training data format (support articles are not Q&A pairs). Tokenizer experiments are exhausted; next workstream should address data preparation or architecture.

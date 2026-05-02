# T1d_bbpe_10k Results

**Experiment:** Byte-level BPE tokenizer, 10K vocab
**Date:** 2026-04-30
**Model:** GPT-style causal transformer, 9.9M params, trained 10,000 steps on STACK support corpus

---

## UNK Token Elimination

Byte-level BPE operates on raw bytes, making UNK tokens structurally impossible — every byte sequence is representable. Confirmed: no `<unk>` token ID assigned in this vocabulary.

| Input | Tokens | UNK |
|---|---|---|
| `STACK v2.14.3` | `ĠSTACK`, `Ġv`, `2`, `.`, `14`, `.`, `3` | none |
| `https://support.stackct.com/hc/en-us` | `Ġhttps`, `://`, `sup`, `port`, `.`, `stackct`, `.`, `com`, `/`, `h`, `c`, `/`, `en`, `-`, `us` | none |
| `RFI#1042` | `ĠRFI`, `#`, `10`, `42` | none |
| `cost_code_001` | `Ġcost`, `_`, `c`, `ode`, `_`, `001` | none |
| `☑ takeoff` | `Ġâ`, `ĺ`, `Ġtakeoff` | none |

Unicode outside the training distribution (e.g., `☑`) degrades gracefully to byte fragments rather than producing UNK — the expected byte-level BPE behaviour.

---

## Domain Term Splits

Vocabulary size: 9,995 tokens (5 slots reserved for special tokens)

| Term | Pieces | vs T1b (5K) |
|---|---|---|
| `construction` | `Ġconstruction` | Same — whole word |
| `estimating` | `Ġestimating` | Same — whole word |
| `subcontractor` | `Ġsubcontractor` | Same — whole word |
| `takeoff` | `Ġtakeoff` | Same — whole word |
| `changeorder` | `Ġchange`, `order` | Same — compound splits |
| `RFI` | `ĠRFI` | **Improved** — T1b split to `Ġr`, `fi`; now a single token |
| `submittal` | `Ġsubmit`, `tal` | Same |
| `STACK` | `ĠSTACK` | Same — whole word |
| `preconstruction` | `Ġpre`, `construction` | **Improved** — T1b/T1c fragmented `construction`; now whole |
| `bid` | `Ġbid` | Same |
| `material` | `Ġmaterial` | Same |
| `labor` | `Ġlabor` | Same |

At 10K vocab, `RFI` is a single token and `preconstruction` splits cleanly at the morpheme boundary — improvements over the 5K baseline.

---

## Generation Samples

Standard test prompts against `gen/model_final.pth`. Note: raw output contains `Ġ` (U+0120, ByteLevel space prefix) — this is correct BPE encoding, not an error. The HuggingFace `tokenizer.decode()` converts it to normal text.

---

**USER:** How do I create a takeoff?

**AI (raw):** `Ġthe Ġsame Ġas Ġan Ġitem . ĠThe Ġitem Ġyou Ġwhen Ġyou Ġcreate Ġa Ġnew Ġtakeoff Ġor Ġassembly Ġitems Ġin Ġone , Ġin Ġone Ġwill Ġbe Ġone Ġof Ġthe Ġlisted Ġwhen Ġcreating Ġan Ġaccurate , Ġgo Ġto Ġthe Ġadded Ġan Ġitem ( s ) Ġyou Ġwant Ġto Ġthe Ġblank ĠItem ĠDetails...`

**AI (decoded):** the same as an item. The item you when you create a new takeoff or assembly items in one, in one will be one of the listed when creating an accurate, go to the added an item(s) you want to the blank Item Details...

---

**USER:** What is a change order?

**AI (decoded):** the same order to enter the relevant Date. Click the end of the X in the issue option. The search term appears when you are the issue. Unselect the issue is displayed in the issue Search field in the name of the category of the category, so you are well as any other category...

---

**USER:** How do I add a subcontractor?

**AI (decoded):** the bottom of the rebar labor sources. The number of the footing when rebar that was uploaded. In cases, the way you want to open them as well as the STACK Field App. Related Articles Upload a list STACK | Takeoff & Estimate in the Documents...

---

**USER:** What is STACK?

**AI (decoded):** projects and you will be able to access to your organization. By default, the function in STACK when you create a project is created, the same data is removed from the different types. When a project is not connected GMS Subsidiary...

---

**USER:** How do I export a bid?

**AI (decoded):** on Project For precise level, Project, Project Quotes, Project Creator: If applicable, enter the project information to bid invite, and see ing New Project: PROJECT NAME and items add the project...

---

## Comparison Across Tokenization Experiments

| Dimension | T1a BPE 2K | T1b BPE 5K | T1c SP 3K | T1d BPE 10K |
|---|---|---|---|---|
| Tokenizer | ByteLevel BPE | ByteLevel BPE | SentencePiece Unigram | ByteLevel BPE |
| Vocab size | 2,000 | 5,000 | 3,000 | 10,000 |
| Model params | ~5.8M | ~5.8M | ~6.3M | ~9.9M |
| UNK tokens | None (byte-level) | None (byte-level) | None (sentencepiece) | None (byte-level) |
| `RFI` handling | Character-split | Character-split | Character-split | **Single token** |
| `preconstruction` | Fragmented | Fragmented | `▁pre`+`c`+`onstruction` | `Ġpre`+`construction` |
| Output artifact | `Ġ` prefix (cosmetic) | `Ġ` prefix (cosmetic) | Suffix fragments, `⁇` | `Ġ` prefix (cosmetic) |
| Generation coherence | Fragment recitation | Fragment recitation | Fragment recitation | Fragment recitation |

---

## Conclusion

At 10K vocab, byte-level BPE achieves the best domain-term coverage of the four experiments: `RFI` is now a single token and morpheme boundaries are cleaner. The `Ġ` artifact is cosmetic and fully reversible via `tokenizer.decode()`, unlike T1c's suffix fragments.

Generation coherence does not improve. All four tokenizer variants produce the same failure mode: recitation of training document fragments. Larger vocabulary and larger embedding table (~9.9M vs ~5.8M params) provide no qualitative improvement.

**Tokenization experiments are now exhausted.** The limiting factor is not tokenizer algorithm or vocabulary size. Next phase should address training data structure (Q&A synthesis) or model architecture — not tokenization.

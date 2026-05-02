# T1e_morph Results

**Experiment:** Morfessor unsupervised morphological tokenizer, corpus-derived vocab
**Date:** 2026-05-02
**Model:** GPT-style causal transformer, 7.5M params, trained 10,000 steps on STACK support corpus

---

## Tokenizer: Subword Splits on Domain Terms

Vocabulary size: 5,348 morphemes (corpus-derived — not a fixed target)

| Term | Pieces | Notes |
|---|---|---|
| `construction` | `construction` | Whole word — high-frequency, unsplit |
| `estimating` | `estimating` | Whole word |
| `subcontractor` | `subcontractor` | Whole word — long compound preserved |
| `takeoff` | `takeoff` | Whole word |
| `changeorder` | `change`, `order` | Compound split at word boundary |
| `RFI` | `RFI` | Whole word — corpus frequency keeps it atomic |
| `submittal` | `submit`, `tal` | Morpheme split: root + suffix |
| `STACK` | `STACK` | Whole word |
| `preconstruction` | `pre`, `construction` | Clean prefix split — best result across all experiments |
| `bid` | `bid` | Whole word |
| `material` | `material` | Whole word |
| `labor` | `labor` | Whole word |

**Observations:**
- On the full corpus Morfessor learns a 5,348-token vocabulary — larger than expected, comparable to T1b BPE 5K.
- Domain nouns are overwhelmingly preserved as whole tokens. High corpus frequency drives MDL toward keeping them atomic.
- `preconstruction` → `pre` + `construction` is the cleanest morphological split across all five experiments.
- `RFI` is a single token (corpus frequency prevents character-level decomposition).
- No byte-level artifacts — morphemes are plain Unicode, no `Ġ` prefix.

---

## Generation Samples

Standard test prompts against `gen/model_final.pth`:

---

**USER:** How do I create a takeoff?

**AI:** to do this directly from the Estimate Worksheet . NOTE : You must set up the connection / integration in STACK that source must have a corresponding project with external source information . NOTE : You must have a Purchase > Acumatica in Acumatica order to do not have permission to delete once in order to Use Acumatica , understand data that require items and may be transferred / customized to the version of the More Information if needed for the Web Portal . Click the three dots and select Download Description . Before : For the permission to Display

---

**USER:** What is a change order?

**AI:** is important to give you a detail Issues . If needed - Open this document , make sure you need to save them as the email of the project . Tap OK . Click the Open icon to open the project . Scroll down to the Web Portal , or the project status in order to validate your document and your email . Manage Collaborator This changes will no longer appear in the left - side navigation menu . Related Articles Assign Users See how others use this feature in the STACK Community . If you need additional assistance ,

---

**USER:** How do I add a subcontractor?

**AI:** ( for Material and Labor or Unit Cost Estimates ) . Duplicate the project must be set in STACK available to use even to rearrange estimates per project . For each item to a Takeoff - External Library copy a workflow item Click the LIBRARIES tab , you can add custom items and assemblies to the same STACK . The Assembly in which can contain a material list of items to assemblies if needed . You can edit your Scope of Work and Terms and Conditions , then just NOT click what you see what data you need .

---

**USER:** What is STACK?

**AI:** to Know The first time you will automatically identifying the most current type of Items and Estimates in your estimate information to your accounting and number . TIP : You can add often as many items as you need . How to Create an Item from the Estimate Worksheet for more details and instructions . NOTE Since in the Unit Cost field are renow next to each Cost will appear ( s ) rounded to the Estimate Worksheet ( no cost data ) so you can see line that Group will be displayed on the Estimate Worksheets .

---

**USER:** How do I export a bid?

**AI:** ( Project See troubleshooting section from an Estimate Worksheet for details . ) To ensure data that uses each own ways to better content within a Custom Libraries , modified , and managed within the Plans tab . This includes controls across all projects based on an accurate plan . The one is the same as thping based on the measurement and number you selected . This can be used when you create a project , create the editing or the public it . Once you have created an Item ( more info on the STACK | Build &

---

## Artifact Analysis

Morfessor output is notably cleaner than prior experiments:

- **No `Ġ` prefix** — morphemes are plain Unicode; no ByteLevel encoding artifact.
- **No suffix fragments** — unlike T1c SentencePiece, Morfessor does not produce bare suffix pieces at generation start.
- **Word spacing is explicit** — the `▁` separator inserted between words during encode means spacing is deterministic on decode. Output reads as space-separated tokens rather than concatenated fragments.
- **Remaining artifacts** — occasional mid-word splits visible in output (`thping`, `renow`) suggest the model sometimes generates morpheme sequences that don't reconstruct to valid words. This is the morphological equivalent of BPE's subword concatenation problem.

---

## Comparison Across All Tokenization Experiments

| Dimension | T1a BPE 2K | T1b BPE 5K | T1c SP 3K | T1d BPE 10K | T1e Morph |
|---|---|---|---|---|---|
| Tokenizer | ByteLevel BPE | ByteLevel BPE | SP Unigram | ByteLevel BPE | Morfessor MDL |
| Vocab size | 2,000 | 5,000 | 3,000 | 10,000 | 5,348 |
| Model params | ~5.8M | ~5.8M | ~6.3M | ~9.9M | ~7.5M |
| `RFI` handling | char-split | char-split | char-split | single token | single token |
| `preconstruction` | fragmented | fragmented | `pre`+`c`+`onstruction` | `pre`+`construction` | `pre`+`construction` |
| Primary artifact | `Ġ` prefix | `Ġ` prefix | suffix fragments, `⁇` | `Ġ` prefix | mid-word splits |
| Output readability | poor | poor | poor | poor | **best** |
| Generation coherence | fragment recitation | fragment recitation | fragment recitation | fragment recitation | fragment recitation |

Output readability is meaningfully better with Morfessor — space-separated morpheme output is easier to read than BPE's run-together tokens or SP's suffix fragments — but coherence is unchanged.

---

## Conclusion

Morfessor produces the cleanest tokenization of all five experiments: no byte-level artifacts, clean morphological splits on domain terms, and deterministic word spacing. On this corpus it learns ~5,300 morphemes — comparable to the BPE 5K baseline in vocab size.

Generation coherence does not improve. All five tokenizer variants produce the same failure mode: recitation of training document fragments with no question-answering behaviour.

**The tokenization phase of experiments is complete.** Five algorithms tested (BPE 2K, BPE 5K, SentencePiece Unigram 3K, ByteLevel BPE 10K, Morfessor MDL). None affect coherence. The root constraint is training data format — support articles are not Q&A pairs. The next phase must address data preparation.

# T2c_eda_qa Results

**Experiment:** EDA augmentation on Q&A answer spans only
**Date:** 2026-05-05
**Model:** GPT-style causal transformer, 7.8M params, trained 10,000 steps on 675 augmented Q&A pairs
**Tokenizer:** Byte-level BPE 10K (T1d)
**Control:** T2a_qa_synth — identical setup, no augmentation (225 pairs)

---

## Training Corpus

| Metric | T2a (control) | T2c (this) |
|---|---|---|
| Source | 225 Q&A pairs | 225 original + 225×10% + 225×20% augmented |
| Total pairs | 225 | 675 (3.0×) |
| Augmentation target | — | Answer spans only (`<\|ai\|>...<\|end\|>`) |
| Training file | `qa_pairs.txt` | `qa_pairs_augmented.txt` |

---

## Generation Comparison

Decoded outputs (Ġ prefix stripped).

---

**USER: How do I create a takeoff?**

T2a: Key Information The Takeoff Toolbar provides access to the tools used to create, edit, and manage measurements on plan sheets.

T2c: Overview Plan icons appear next to the document name in the STACK Field App when in list view or in the upper right corner of the thumbnail in gallery view. A **papers** can display more than one icon. Icon **contrive** that appear in the STACK Field App **testament** also **presentation** the same info in the STATUS column...

---

**USER: What is a change order?**

T2a: This feature is temporarily unavailable. This feature when the report creator assigns a Reporter but does not assign a Reviewer to the report...

T2c: Below are the most commonly used items across industries in STACK. You will need to download a single view more detailed material and cost information on your Unit Cost Estimates and the capability to apply the users can allow you greater.

---

**USER: How do I add a subcontractor?**

T2a: The following table shows the access permissions for each Role: WORKFLOWS OWNER ADMINISTRATOR...

T2c: Overview STACK **Discover** Information The STACK Community is an on-line space where STACK users connect to ask questions, share ideas, and learn from real-world experiences. It works alongside the STACK **Assistance** Center. The Help Center provides **prescribed** guidance and step-by-step instructions...

---

**USER: What is STACK?**

T2a: Key Information The STACK Community is an online space where STACK users connect to ask questions, share ideas, and learn from real-world experiences...

T2c: Key Information The STACK encourages the creation of project templates to streamline the setup of new projects. Administrators can design multiple templates tailored to different project types. Using project templates reduces configuration time and ensures greater consistency across projects.

---

**USER: How do I export a bid?**

T2a: The Estimate Worksheet gives you the ability to calculate labor hours. To start, you must add at least one labor Item to the takeoff measurements in the project...

T2c: The Takeoff **Inclination**, placed on a project's Takeoffs tab, is where you will find **wholly** the **parody** for that project. It displays the name and total measurement/count for each takeoff. The Takeoff List is where you will **loose** a takeoff to cut the Takeoff Settings or add measurements/counts; create a copy of **amp** takeoff; delete a takeoff from the project if needed; or add a **burlesque** to the Takeoff Library...

---

## Analysis

**Synonym contamination reappears:** Despite restricting augmentation to answer spans, contaminated synonyms leak into generation output:

| Augmented synonym | Likely original | Context |
|---|---|---|
| `papers` | `document` | "A papers can display..." |
| `contrive` | `types` | "Icon contrive that appear..." |
| `testament` | `will` | "testament also presentation the same info" |
| `Assistance` | `Help` | "STACK Assistance Center" |
| `prescribed` | `official` | "prescribed guidance" |
| `Inclination` | `List` | "Takeoff Inclination" |
| `parody` | unknown | "find wholly the parody" |
| `burlesque` | unknown | "add a burlesque to the Takeoff Library" |
| `amp` | `a` | "a copy of amp takeoff" |

`burlesque` and `parody` as synonyms for domain terms suggest WordNet is pulling in figurative/theatrical senses of words that have no business in construction software documentation.

**On-topic rate dropped:** T2a returned 3/5 relevant responses; T2c returns 1/5 ("What is STACK?" is partially on-topic). The 3× corpus expansion did not improve retrieval — it diluted the signal with noisy variants.

---

## Comparison to T2a (Control)

| Dimension | T2a (225 plain pairs) | T2c (675 augmented pairs) |
|---|---|---|
| On-topic rate | 3 / 5 | 1 / 5 |
| Response shape | Answer-shaped, stops cleanly | Answer-shaped, stops cleanly |
| Synonym contamination | None | Present — answer span leaks into generation |
| Coherence | Best seen so far | Degraded by noise |
| Overall | **Better** | **Worse than T2a** |

---

## Conclusion

EDA augmentation is harmful regardless of where it is applied. Restricting augmentation to answer spans (T2c) versus raw articles (T2b) does not prevent synonym contamination from degrading output — the model memorises augmented answer variants and samples them at generation time.

WordNet-based synonym replacement is unsuitable for this domain. Construction software documentation contains many words with multiple senses (e.g., `list`, `assembly`, `takeoff`, `estimate`) and WordNet resolves them to figurative or uncommon senses (`Inclination`, `burlesque`, `parody`) that are semantically wrong in context.

**EDA is ruled out as a data augmentation strategy for this corpus.** T2a (225 plain Q&A pairs) remains the best result. The next data-prep experiment should increase pair volume through better synthesis coverage — more question types per article, or model-assisted generation — rather than lexical augmentation.

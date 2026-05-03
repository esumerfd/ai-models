# T2b_eda_augment Results

**Experiment:** EDA synonym augmentation on raw articles
**Date:** 2026-05-04
**Model:** GPT-style causal transformer, 9.9M params, trained 10,000 steps
**Tokenizer:** Byte-level BPE 10K (T1d)
**Control:** T1d_bbpe_10k — identical setup, no augmentation

---

## Training Corpus

| Metric | T1d (control) | T2b (this) |
|---|---|---|
| Source | Raw articles | Raw articles + EDA 10% + EDA 20% |
| Corpus size | 1.1M chars | 3.5M chars (3.0×) |
| Training file | `training.txt` | `training_augmented.txt` |

---

## Generation Comparison

Decoded outputs (Ġ prefix stripped for readability).

---

**USER: How do I create a takeoff?**

T1d: or Flooring in STACK for your estimate into how to use the same item. Example: If you are grouped, the total square footage, linear foot, etc. Purchase Unit variable is covered by default. 8 - Purchase Quantity report format cost type. Accounting Code The material quantity for each material list for each material...

T2b: from an new Takeoff List You leave automatically finding and will not be able to retrieve the project. * not requisite or Pitched Linear, Takeoff List Decimalboard / how the data is based on the Takeoff List / Protrude All of measurements/counts from/counts times when quantities. Takeoff created by mechanically exist measurements exist promised projects from a takeoff...

---

**USER: What is a change order?**

T1d: of the list in this column, including the exact on the entire plan and its real-world measurement values. Each custom cost is set can have more selected. Open the Estimate Worksheet and enter the Purchase Unit Cost Data...

T2b: from STACK Takeoff & Estimate directly to the most recent successful exports. Annotation Permissions: Cost Types Unit Cost Estimate Permissions Cost Estimate Reports Permissions Catch how others use this feature in the STACK Community. If you **pauperization** additional assistance, **delight** Link Us...

---

**USER: How do I add a subcontractor?**

T1d: , More Information, and customization if the labor cost of materials, 000s in a Material, Labor, and the library settings in your estimate. Changes made it appears automatically...

T2b: from the Select Takeoff from the dropdown. Click Settings on the left panel, then following cry Dust dropdown astatine the Appearance and click NOT prime the bottom of the chief item name and drop to the desired positions...

---

**USER: What is STACK?**

T1d: the Item Cost Report. Items & Services fields are optional. When you export your current item information so they need a takeoff, you will be an item...

T2b: from STACK | Takeoff & Estimate into STACK | Build & Operate, you'll see Acumatica, or new markups. Logging how others use this have in to create additional Takeoffs. This integration with STACK Community. If you have more about different: Acumatica version of formalizing the STACK Customization Package...

---

**USER: How do I export a bid?**

T1d: settings — ft, each, etc.) Submitting an item(s) EX: When the takeoff values are selected, a takeoff unit of measure is shown in its type. Measured Coverage Rate Measured Coverage Rate Measured Items are set will automatically added to the Estimate Worksheet...

T2b: from STACK | Takeoff & Estimate Use the list of users and complete by using new features. BulkION: Information STACK Assist]) = QBO Item (Categories + Extra Costs Accounting Code) STACK 88 = Token Description * proposals you External Type / Company De p a Name so ha les ce & errors) on the judge match the name of the report... **leaflet** for all the cost... **Exploiter** New

---

## Analysis

**Coherence:** No improvement over T1d. Both models produce article-fragment recitation filling the 200-token budget. The 3× corpus expansion did not teach the model to answer questions.

**Synonym contamination — new failure mode:** EDA-augmented synonyms leak into generated output, producing nonsensical substitutions:

| Original word | EDA synonym in output | Context |
|---|---|---|
| `need` | `pauperization` | "If you pauperization additional assistance" |
| `please` | `delight` | "delight Link Us" |
| `document` | `leaflet` | "a separate leaflet for all the cost" |
| `User` | `Exploiter` | "Create Exploiter New" |
| `automatically` | `mechanically` | "Takeoff created by mechanically exist" |
| `chief` | `chief` | (benign, but unusual) |

The model memorised augmented variants alongside originals. At generation time it sometimes samples the synonym variant, producing fluent-but-wrong output. This is strictly worse than the T1d control.

---

## Comparison to T1d (Control)

| Dimension | T1d (control) | T2b (EDA augmented) |
|---|---|---|
| Corpus size | 1.1M chars | 3.5M chars |
| Response shape | Article fragment recitation | Article fragment recitation |
| On-topic rate | 0 / 5 | 0 / 5 |
| Coherence | Poor | Poor |
| New failure mode | None | Synonym contamination in output |
| Overall | Baseline | **Worse** |

---

## Conclusion

EDA synonym augmentation on raw articles does not improve generation coherence and introduces a new failure mode: synonym contamination. The model treats augmented variants as valid alternatives and samples them at generation time, producing outputs like "pauperization additional assistance" and "Create Exploiter New."

Corpus expansion (3×) on raw article data is not sufficient to change generation behaviour — the format problem identified in T1a–T1e remains. The winning insight from T2a (Q&A format) still stands.

The next experiment (T2c) applies EDA augmentation only to the answer spans of synthesized Q&A pairs, where synonym contamination is less damaging because the answer text is already domain-grounded prose rather than UI copy.

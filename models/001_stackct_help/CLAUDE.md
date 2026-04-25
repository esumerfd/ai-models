# Project: First Small Language Model

## Goal

Build a small language model (SLM) from scratch using machine learning techniques.

The ultimate deployment target is the Raspberry Pi cluster located at:
`projects/ai/pi-cluster`

## Reference

We are following this guide:

**[Building a Small Language Model from Scratch](https://medium.com/@rajasami408/building-a-small-language-model-from-scratch-a-practical-guide-to-domain-specific-ai-59539131437f)**
A practical guide to domain-specific AI by Raja Sami.

## Key Constraints

- Model must be small enough to run on Raspberry Pi hardware
- Focus on domain-specific AI capabilities
- Build from scratch to understand the underlying mechanics

## Hardware

Deployment target is the **[Raspberry Pi AI HAT+ 2](https://www.raspberrypi.com/products/ai-hat-plus-2/)** which uses a Hailo-10H chip (10 TOPS).

**Idea:** Upgrade `NUM_HEADS` to 10 in `config.py` to align with the 10H in the Hailo-10H — the number of attention heads can reflect the parallelism available on the accelerator.

Note though — NUM_HEADS must divide evenly into EMBEDDING_DIM. With EMBEDDING_DIM=256, valid head counts are 1, 2, 4, 8, 16, 32... so 10 doesn't divide cleanly into 256. When
  we act on this idea we'd need to either bump EMBEDDING_DIM to 320 (32×10) or keep NUM_HEADS=8 and note the constraint.

## Training Data

* https://support.stackct.com/hc/en-us


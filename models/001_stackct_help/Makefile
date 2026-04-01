ROOT_DIR := $(CURDIR)
VENV := $(ROOT_DIR)/.venv
PYTHON := $(VENV)/bin/python

.PHONY: help venv retrieve train generate convert ollama-load ollama-run capabilities clean-gguf clean-all

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Pipeline:"
	@echo "  retrieve      Download training data"
	@echo "  train         Train the model (runs clean-all first)"
	@echo "  generate      Run text generation with the trained model"
	@echo "  convert       Convert model to GGUF format"
	@echo ""
	@echo "Ollama:"
	@echo "  ollama-load   Load GGUF model into Ollama"
	@echo "  ollama-run    Run the model via Ollama"
	@echo ""
	@echo "Tools:"
	@echo "  capabilities  Show model capabilities"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean-gguf    Remove GGUF artifacts and Ollama model"
	@echo "  clean-all     Remove all generated artifacts"

$(VENV)/bin/python:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q -r requirements.txt

venv: $(VENV)/bin/python

retrieve: venv
	$(MAKE) -C phase_1_training ROOT_DIR=$(ROOT_DIR) retrieve

train: clean-all venv
	$(MAKE) -C phase_1_training ROOT_DIR=$(ROOT_DIR) train

generate: venv
	$(MAKE) -C phase_2_generation ROOT_DIR=$(ROOT_DIR) generate

convert: venv
	$(MAKE) -C phase_3_conversion ROOT_DIR=$(ROOT_DIR) convert

ollama-load:
	$(MAKE) -C phase_3_conversion ROOT_DIR=$(ROOT_DIR) ollama-load

ollama-run:
	ollama run stackslm

capabilities:
	$(MAKE) -C phase_4_tools ROOT_DIR=$(ROOT_DIR) capabilities

clean-gguf:
	$(MAKE) -C phase_3_conversion ROOT_DIR=$(ROOT_DIR) clean-gguf

clean-all: clean-gguf
	rm -f gen/*.pth gen/tokenizer.json

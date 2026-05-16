ROOT_DIR := $(CURDIR)
VENV := $(ROOT_DIR)/.venv
PYTHON := $(VENV)/bin/python

.PHONY: help venv train index retrieve visualize clean-all

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Pipeline:"
	@echo "  train      MLM-pretrain the BERT encoder on article corpus"
	@echo "  index      Encode all articles and build retrieval index"
	@echo "  retrieve   Interactive retrieval REPL"
	@echo "  visualize  PCA scatter + cosine similarity histogram → gen/embedding_viz.png"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean-all  Remove all generated model artifacts"

$(VENV)/bin/python:
	python3.12 -m venv $(VENV)
	$(VENV)/bin/pip install -q -r requirements.txt

venv: $(VENV)/bin/python

train: clean-all venv
	$(MAKE) -C step_1_pretrain ROOT_DIR=$(ROOT_DIR) train

index: venv
	$(MAKE) -C step_2_index ROOT_DIR=$(ROOT_DIR) index

retrieve: venv
	$(MAKE) -C step_3_retrieve ROOT_DIR=$(ROOT_DIR) retrieve

visualize: venv
	cd $(ROOT_DIR) && $(PYTHON) visualize_embeddings.py

clean-all:
	rm -f gen/model_*.pth gen/article_index.npz gen/embedding_viz.png

ROOT_DIR := $(CURDIR)
VENV := $(ROOT_DIR)/.venv
PYTHON := $(VENV)/bin/python

.PHONY: help venv train generate visualize clean-all

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Pipeline:"
	@echo "  train         Train the seq2seq model"
	@echo "  generate      Run interactive inference"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean-all     Remove all generated model artifacts"

$(VENV)/bin/python:
	python3.12 -m venv $(VENV)
	$(VENV)/bin/pip install -q -r requirements.txt

venv: $(VENV)/bin/python

train: clean-all venv
	$(MAKE) -C step_1_training ROOT_DIR=$(ROOT_DIR) train

generate: venv
	$(MAKE) -C step_2_generation ROOT_DIR=$(ROOT_DIR) generate

visualize: venv
	$(MAKE) -C step_3_visualize ROOT_DIR=$(ROOT_DIR) visualize

clean-all:
	rm -f gen/model_*.pth

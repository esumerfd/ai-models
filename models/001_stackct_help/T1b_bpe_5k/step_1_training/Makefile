ROOT_DIR ?= ..

.PHONY: retrieve train

retrieve:
	cd $(ROOT_DIR) && .venv/bin/python step_1_training/retrieve_data.py

train:
	cd $(ROOT_DIR) && time .venv/bin/python step_1_training/train.py

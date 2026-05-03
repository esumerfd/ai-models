ROOT_DIR ?= ..

.PHONY: retrieve synthesize train

retrieve:
	cd $(ROOT_DIR) && .venv/bin/python phase_1_training/retrieve_data.py

synthesize:
	cd $(ROOT_DIR) && .venv/bin/python phase_1_training/synthesize_qa.py

train:
	cd $(ROOT_DIR) && time .venv/bin/python phase_1_training/train.py

ROOT_DIR ?= ..

.PHONY: retrieve augment train

retrieve:
	cd $(ROOT_DIR) && .venv/bin/python step_1_training/retrieve_data.py

augment:
	cd $(ROOT_DIR) && .venv/bin/python step_1_training/augment_eda.py

train:
	cd $(ROOT_DIR) && time .venv/bin/python step_1_training/train.py

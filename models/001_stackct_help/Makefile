ROOT_DIR := $(CURDIR)

.PHONY: train generate convert ollama-load ollama-run capabilities clean-gguf clean-all

train: clean-all
	$(MAKE) -C phase_1_training ROOT_DIR=$(ROOT_DIR) train

generate:
	$(MAKE) -C phase_2_generation ROOT_DIR=$(ROOT_DIR) generate

convert:
	$(MAKE) -C phase_3_conversion ROOT_DIR=$(ROOT_DIR) convert

ollama-load:
	$(MAKE) -C phase_3_conversion ROOT_DIR=$(ROOT_DIR) ollama-load

ollama-run:
	$(MAKE) -C phase_3_conversion ROOT_DIR=$(ROOT_DIR) ollama-run

capabilities:
	$(MAKE) -C phase_4_tools ROOT_DIR=$(ROOT_DIR) capabilities

clean-gguf:
	$(MAKE) -C phase_3_conversion ROOT_DIR=$(ROOT_DIR) clean-gguf

clean-all: clean-gguf
	rm -f gen/*.pth gen/tokenizer.json

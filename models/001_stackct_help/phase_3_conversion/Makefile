ROOT_DIR ?= ..

.PHONY: convert ollama-load ollama-run clean-gguf

convert:
	cd $(ROOT_DIR) && .venv/bin/pip install -q gguf && .venv/bin/python phase_3_conversion/convert_to_gguf.py

ollama-load:
	cd $(ROOT_DIR)/phase_3_conversion && ollama create stackslm -f modelfile-ollama

ollama-run:
	ollama run stackslm

clean-gguf:
	rm -f $(ROOT_DIR)/gen/*.gguf
	ollama rm stackslm 2>/dev/null || true

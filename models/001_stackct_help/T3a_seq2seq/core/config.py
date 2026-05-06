"""
Hyperparameters for the Seq2Seq (encoder-decoder) transformer.
Targeting Hailo-8 (10 TOPS) accelerator on Raspberry Pi cluster.
"""

# Tokenizer — shared with T2a (BPE 10K, best from tokenization phase)
VOCAB_SIZE = 10000

# Model architecture
EMBEDDING_DIM = 256
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
FFN_DIM = 512
DROPOUT = 0.1

# Sequence lengths
MAX_SRC_LEN = 64    # max question tokens
MAX_TGT_LEN = 128   # max answer tokens

# Training
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
STEPS = 10000
EVAL_INTERVAL = 250
SAVE_INTERVAL = 1000
TRAIN_SPLIT = 0.9

# Paths
DATA_FILE = "gen/qa_pairs.txt"
CHECKPOINT_DIR = "gen"
TOKENIZER_FILE = "gen/tokenizer.json"

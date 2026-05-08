VOCAB_SIZE = 10000
EMBEDDING_DIM = 256
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 6
FFN_DIM = 512
DROPOUT = 0.1
MAX_SEQ_LEN = 256

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
STEPS = 10000
EVAL_INTERVAL = 250
SAVE_INTERVAL = 1000
MLM_MASK_PROB = 0.15

# Reuse special token IDs from the shared BPE 10K tokenizer
CLS_ID = 0   # <|start|>
PAD_ID = 1   # <|end|>
MASK_ID = 2  # <|system|> repurposed as [MASK]

TOP_K = 3

DATA_FILE = "gen/training.txt"
CHECKPOINT_DIR = "gen"
TOKENIZER_FILE = "gen/tokenizer.json"
INDEX_FILE = "gen/article_index.npz"

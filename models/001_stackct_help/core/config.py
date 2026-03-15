"""
Hyperparameters for the Small Language Model.
Targeting Hailo-8 (10 TOPS) accelerator on Raspberry Pi cluster.
"""

# Tokenizer
VOCAB_SIZE = 2000         # Small vocabulary for domain-specific text

# Model architecture
EMBEDDING_DIM = 256       # Size of token/position embedding vectors
CONTEXT_LENGTH = 128      # Number of tokens the model can "see" at once
NUM_HEADS = 8             # Number of attention heads per block
NUM_LAYERS = 6            # Number of transformer blocks stacked
DROPOUT = 0.1             # Dropout rate for regularization

# Training
BATCH_SIZE = 16           # Samples per gradient update
LEARNING_RATE = 3e-4      # Adam optimizer learning rate (standard for transformers)
STEPS = 10000             # ~35 passes through the training data
EVAL_INTERVAL = 250       # Print loss every N steps
SAVE_INTERVAL = 1000      # Save checkpoint every N steps
TRAIN_SPLIT = 0.9         # Fraction of data used for training

# Paths
DATA_FILE = "gen/training.txt"
CHECKPOINT_DIR = "gen"
TOKENIZER_FILE = "gen/tokenizer.json"

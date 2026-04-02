
## Data Retrieval

- Zendesk Help Center API -> strip HTML -> `gen/training.txt` (~1.2 MB)
- Problem: stripping HTML destroys all structure (tables, lists, steps)
- Alternatives: Markdown, QA pairs, chunked sections, JSON Lines

## Training

- GPT-style causal transformer — predict the next token
- 10,000 steps, batch size 16, learning rate 3e-4
- 29 minutes on CPU (Apple Silicon, no GPU)

## The Model File

- Output: `gen/model_final.pth` (22 MB)
- `file model_final.pth` reveals it's a **zip archive**
- `unzip -l model_final.pth` shows **358 files**:
  - `data.pkl` — pickle mapping parameter names to numbered data files
  - `data/0` .. `data/351` — raw tensor weights as flat binary
- File sizes reveal the architecture: the pattern repeats 6 times (one per transformer block)

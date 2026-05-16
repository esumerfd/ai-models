# Phase 1 — Training

Trains a GPT-style causal transformer on STACK support site data.

## Training Configuration

All hyperparameters are defined in `core/config.py`.

### Learning Rate (`LEARNING_RATE = 3e-4`)

Controls how much the model weights adjust on each optimization step. The value
`3e-4` (0.0003) is a widely used default for the Adam optimizer with transformer
models — large enough to make steady progress, small enough to avoid divergence.

The effective learning rate during training is not constant. It is controlled by
the **OneCycleLR** scheduler (see below).

### Steps (`STEPS = 10000`)

The total number of gradient updates during training. Each step processes one
batch of `BATCH_SIZE` (16) samples. With the current dataset size this gives
roughly 35 passes (epochs) over the training data.

More steps allow the model to see the data more times, but too many risk
overfitting — the model memorizes training data rather than learning general
patterns.

### OneCycleLR Scheduler and `pct_start`

The training script uses PyTorch's `OneCycleLR` learning rate scheduler, which
varies the learning rate over the course of training in a single "super
convergence" cycle:

```
LR
 ^
 |     /\
 |    /  \
 |   /    \
 |  /      --------__
 | /                  \
 +--------------------------> step
   |    |                |
   0  pct_start       STEPS
       * STEPS
```

**`pct_start = 0.05`** means the first 5% of steps (500 out of 10,000) are the
warm-up phase where the learning rate ramps up from near zero to `LEARNING_RATE`.
The remaining 95% is the annealing phase where it gradually decays back down.

The warm-up prevents early instability — randomly initialized weights produce
large gradients, and a high learning rate at that point can push the model into
a bad region of the loss landscape. By starting low and ramping up, the model
finds a stable trajectory before training at full speed.

### Other Key Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `BATCH_SIZE` | 16 | Samples per gradient update. Larger batches give more stable gradients but use more memory. |
| `CONTEXT_LENGTH` | 128 | Number of tokens the model sees at once (its "window"). Limits how far back the model can look when predicting the next token. |
| `TRAIN_SPLIT` | 0.9 | 90% of data for training, 10% for validation. |
| `EVAL_INTERVAL` | 250 | Print loss every 250 steps to monitor progress. |
| `SAVE_INTERVAL` | 1000 | Save a checkpoint every 1,000 steps for recovery. |

### Gradient Clipping (`max_norm=1.0`)

Applied via `clip_grad_norm_` each step. Caps the total gradient magnitude at
1.0 to prevent exploding gradients — a common issue in deep transformer
networks where gradients can compound across layers.

### Optimizer — Adam

Adam combines momentum (smoothed gradient direction) with adaptive per-parameter
learning rates. It is the standard optimizer for transformer training because it
handles the varying gradient scales across embedding, attention, and feed-forward
layers without manual tuning.

# Classification Hyperparameter Tuning

This example demonstrates the `ClassifyTuner` API for automatic hyperparameter optimization of the shell safety classifier.

## Running the Example

```bash
# Quick demo with built-in 15-sample corpus
cargo run --example classify_tune_demo

# With a real corpus (exported from bashrs)
cargo run --example classify_tune_demo -- /tmp/ssc-corpus.jsonl
```

## Code

```rust
{{#include ../../../examples/classify_tune_demo.rs}}
```

## Overview

The example runs a **scout** tuning session: 3 trials, 1 epoch each, using TPE (Tree-structured Parzen Estimators) to explore the hyperparameter space. This is the fast first pass before a full tuning run.

### Pipeline

1. **Load corpus** - 15-sample built-in demo (or JSONL file from `bashrs corpus export-dataset`)
2. **Build ClassifyTuner** - Configures TPE search over a 9-parameter space
3. **Run trials** - Each trial: suggest params, build ClassifyPipeline, train 1 epoch, record result
4. **Display leaderboard** - Sorted by validation loss (best first)
5. **Export JSON** - Machine-readable `TuneResult` for downstream tooling

## Search Space

The default classification search space has 9 parameters:

| Parameter | Domain | Range | Description |
|-----------|--------|-------|-------------|
| `learning_rate` | Continuous (log) | 5e-6 .. 5e-4 | AdamW learning rate |
| `lora_rank` | Discrete | 4 .. 64 (step 4) | LoRA adapter rank |
| `lora_alpha_ratio` | Continuous | 0.5 .. 2.0 | Alpha = rank * ratio |
| `batch_size` | Categorical | {8, 16, 32, 64, 128} | Training batch size |
| `warmup_fraction` | Continuous | 0.01 .. 0.2 | LR warmup proportion |
| `gradient_clip_norm` | Continuous | 0.5 .. 5.0 | Max gradient norm |
| `class_weights` | Categorical | {uniform, inverse_freq, sqrt_inverse} | Loss weighting strategy |
| `target_modules` | Categorical | {qv, qkv, all_linear} | LoRA target projections |
| `lr_min_ratio` | Continuous (log) | 0.001 .. 0.1 | Cosine decay floor |

## Sample Output

```text
======================================================
  Classification HP Tuning Demo (SPEC-TUNE-2026-001)
  Powered by ClassifyTuner + TPE search
======================================================

Corpus: 15 samples
  [0] safe                 3 samples
  [1] needs-quoting        3 samples
  [2] non-deterministic    3 samples
  [3] non-idempotent       3 samples
  [4] unsafe               3 samples

--- Running Scout Trials ---

Trial 0: lr=1.58e-4, rank=32, alpha=48.0, weights=uniform, targets=qv
  -> loss=1.6094, accuracy=20.0%, time=12ms
Trial 1: lr=5.00e-5, rank=16, alpha=16.0, weights=inverse_freq, targets=qkv
  -> loss=1.5987, accuracy=26.7%, time=10ms
Trial 2: lr=2.50e-4, rank=8, alpha=12.0, weights=sqrt_inverse, targets=qv
  -> loss=1.5823, accuracy=33.3%, time=8ms

--- Leaderboard (sorted by val_loss) ---

  Trial  Val Loss   Accuracy   LR       Rank   Alpha    Time
  --------------------------------------------------------------
  2      1.5823     33.3%      2.50e-4  8      12.0     8ms
  1      1.5987     26.7%      5.00e-5  16     16.0     10ms
  0      1.6094     20.0%      1.58e-4  32     48.0     12ms

  Best trial: #2 (val_loss=1.5823)
```

## CLI Equivalent

The `apr` CLI wraps the same `ClassifyTuner` API:

```bash
# Scout: find good HP region (fast, 1 epoch per trial)
apr tune --task classify --budget 5 --scout --data corpus.jsonl --json

# Full: run with best strategy (multi-epoch, ASHA early stopping)
apr tune --task classify --budget 10 --data corpus.jsonl \
    --strategy tpe --scheduler asha
```

## See Also

- [Shell Safety Classification](./finetune-real.md) - Full fine-tuning example
- [Test Generation Fine-Tuning](./finetune-test-gen.md) - QLoRA fine-tuning for test generation
- [HPO Overview](../mlops/hpo.md) - Hyperparameter optimization concepts

# SPEC: Simplify Entrenar — Dirt Simple Fine-Tune to HuggingFace

**Status:** Draft
**Date:** 2026-02-17
**Author:** PAIML Engineering
**Goal:** Make it trivially easy to fine-tune a model and upload it to HuggingFace Hub

## 1. Vision

Three commands to fine-tune any model and ship it:

```bash
# 1. Initialize a project (generates config.yaml)
entrenar init --base Qwen/Qwen2.5-Coder-0.5B --method qlora --data ./my-data/

# 2. Train (auto-downloads model, trains, saves checkpoint)
entrenar train config.yaml

# 3. Publish (merges adapters, exports safetensors, uploads to HF)
entrenar publish --repo myuser/my-finetuned-model
```

Or even simpler with a YAML `publish:` section:

```bash
entrenar train config.yaml   # trains AND publishes
```

## 2. Architecture: Entrenar + Aprender Boundary

### 2.1 Design Principle

**Entrenar = Training UX & Orchestration.** Entrenar owns the training loop, fine-tuning
methods (LoRA/QLoRA), model merging, distillation, and the CLI. It is the user-facing tool.

**Aprender = ML Primitives & Format I/O.** Aprender owns loss functions, metrics, evaluation,
pruning algorithms, serialization formats (APR/SafeTensors/GGUF), HF Hub client, tokenization,
and general-purpose NN layers.

Entrenar imports from aprender. Aprender never imports from entrenar.

### 2.2 Boundary Table

| Capability | Owner | Notes |
|------------|-------|-------|
| **Training loop** | Entrenar | Epoch/step, callbacks, checkpointing |
| **Autograd (tape-based)** | Entrenar | Training-focused, CUDA-accelerated |
| **Optimizers (SGD/Adam/AdamW)** | Entrenar | Training-specific with SIMD, grad clip |
| **LR Schedulers** | Entrenar | Cosine, warmup, step decay |
| **LoRA / QLoRA** | Entrenar | Parameter-efficient fine-tuning |
| **Quantization (QAT/PTQ)** | Entrenar | Training-time quantization |
| **Model Merging (TIES/DARE/SLERP)** | Entrenar | Post-training model combination |
| **Knowledge Distillation** | Entrenar | Multi-teacher, progressive |
| **Transformer architecture** | Entrenar | LLM-specific blocks, CUDA blocks |
| **HF Pipeline (fetch/train/publish)** | Entrenar | End-to-end orchestration |
| **Data loading for training** | Entrenar | Batching, collation, tokenization |
| **TUI monitoring** | Entrenar | Real-time training dashboard |
| **CLI** | Entrenar | `entrenar train/publish/merge/quantize` |
| **YAML declarative config** | Entrenar | Ludwig-style training config |
| --- | --- | --- |
| **Loss functions** | **Aprender** | MSE, CE, Huber, contrastive, triplet |
| **Metrics (eval)** | **Aprender** | Accuracy, F1, precision, recall, ranking |
| **Pruning algorithms** | **Aprender** | Magnitude, lottery, SparseGPT, WANDA |
| **Serialization (APR/SafeTensors/GGUF)** | **Aprender** | Format read/write |
| **HF Hub client** | **Aprender** | HTTP upload/download, token resolution |
| **Tokenization (BPE, Llama)** | **Aprender** | Text processing |
| **NN layers (general)** | **Aprender** | Linear, Conv, LSTM, GRU, etc. |
| **Classical optimization** | **Aprender** | LBFGS, ADMM, CG, interior point |
| **Data primitives** | **Aprender** | Vector, Matrix, DataFrame |

### 2.3 What Entrenar Should Delegate to Aprender

These modules currently exist in both projects. Entrenar should migrate to using aprender's
implementations and delete its own copies over time.

#### Phase 1: Immediate Delegation (Low Risk)

| Module | Entrenar LOC | Aprender LOC | Migration |
|--------|-------------|-------------|-----------|
| **Loss functions** | 1,557 | 1,776 | Import `aprender::loss::*`; keep only `CausalLMLoss` in entrenar |
| **Eval metrics** | 723 | 3,334 | Import `aprender::metrics::*`; delete `src/train/metrics/` |
| **Tokenization** | (re-export) | (owned) | Already delegated via `pub use aprender::text::bpe` |
| **Format I/O** | (re-export) | 80,968 | Already delegated via `aprender::format::*` |

#### Phase 2: Gradual Delegation (Medium Risk)

| Module | Entrenar LOC | Aprender LOC | Migration |
|--------|-------------|-------------|-----------|
| **Pruning integration** | 8,148 | 11,093 | Keep entrenar's training-loop integration; use aprender's algorithms |
| **HF Hub client** | 12,789 | 1,510 | Entrenar keeps full ownership (`crate::hf_pipeline`) |

#### Phase 3: Keep in Entrenar (No Delegation)

These are training-specific and have no aprender equivalent:

| Module | Entrenar LOC | Reason |
|--------|-------------|--------|
| **Autograd (tape)** | 10,545 | Training-specific tape-based design; aprender's is inference-only |
| **Optimizers** | 6,203 | SIMD-accelerated, grad clip, training-loop integration |
| **LR Schedulers** | (in optim) | Training-specific scheduling |
| **LoRA / QLoRA** | 3,942 | Unique to entrenar |
| **Distillation** | 1,532 | Unique to entrenar |
| **Model Merging** | 3,566 | Unique to entrenar |
| **Transformer blocks** | 3,959 | LLM-specific architecture |
| **Training loop** | (various) | Core value-add |

### 2.4 Aprender Hook Points

Entrenar should use these aprender traits/types as extension points:

```rust
// Loss functions: implement aprender's Loss trait for training-specific losses
use aprender::loss::Loss;

impl Loss for CausalLMLoss {
    fn compute(&self, y_pred: &Vector<f32>, y_true: &Vector<f32>) -> f32 { ... }
    fn name(&self) -> &str { "causal_lm" }
}

// Metrics: use aprender's evaluation directly
use aprender::metrics::{accuracy, f1_score, precision, recall};

// Pruning: wrap aprender's algorithms in entrenar's training scheduler
use aprender::pruning::{MagnitudePruner, WandaPruner, Importance};

// HF Hub: entrenar owns pipeline orchestration via HfPublisher/HfModelFetcher
use crate::hf_pipeline::publish::publisher::HfPublisher;
use crate::hf_pipeline::HfModelFetcher;

// Serialization: use aprender's format writers
use aprender::format::gguf::{export_tensors_to_gguf, GgmlType};
use aprender::serialization::safetensors::SafeTensorsWriter;
```

## 3. Missing CLI Features

### 3.1 `entrenar publish` (Priority: HIGH)

The `merge_export_publish()` function exists in `src/lora/adapter/merge_pipeline.rs`
but has no CLI entry point.

**Add to `Command` enum:**

```rust
/// Publish a trained model to HuggingFace Hub
Publish(PublishArgs),
```

**`PublishArgs`:**

```rust
pub struct PublishArgs {
    /// HuggingFace repo ID (e.g., myuser/my-model)
    #[arg(long)]
    pub repo: String,

    /// Path to trained model output directory
    #[arg(default_value = "./output")]
    pub model_dir: PathBuf,

    /// Make the repository private
    #[arg(long)]
    pub private: bool,

    /// Generate and upload a model card
    #[arg(long, default_value_t = true)]
    pub model_card: bool,

    /// Merge LoRA adapters before publishing
    #[arg(long)]
    pub merge_adapters: bool,

    /// Base model for adapter merging (HF repo ID or local path)
    #[arg(long)]
    pub base_model: Option<String>,

    /// Export format (safetensors or gguf)
    #[arg(long, default_value = "safetensors")]
    pub format: String,
}
```

**Workflow:**

```
entrenar publish --repo myuser/my-model ./output/
  1. Detect output format (checkpoint, LoRA adapters, full weights)
  2. If LoRA adapters: merge with base model -> SafeTensors
  3. Generate model card from training metadata (final_model.json)
  4. Create HF repo (or reuse existing)
  5. Upload model.safetensors + README.md + config.json
  6. Print repo URL
```

### 3.2 HF Model ID Auto-Resolve in YAML (Priority: HIGH)

Currently `model.path` must be a local file. Enable HF repo IDs:

```yaml
model:
  path: Qwen/Qwen2.5-Coder-0.5B   # auto-detected as HF repo ID
  # OR
  path: ./local-model.safetensors   # local file (existing behavior)
```

**Detection logic:** If `model.path` contains `/` and no file extension, treat as HF repo ID.
Download to `~/.cache/huggingface/hub/models--{org}--{name}/` (standard HF cache).

### 3.3 YAML `publish:` Section (Priority: MEDIUM)

Add optional publish config to the training spec:

```yaml
model:
  path: Qwen/Qwen2.5-Coder-0.5B

data:
  train_path: ./data/train.jsonl
  batch_size: 8

optimizer:
  name: adamw
  lr: 2e-4

lora:
  rank: 64
  alpha: 16

training:
  epochs: 3
  output_dir: ./output

# NEW: auto-publish after training completes
publish:
  repo: myuser/qwen-finetuned
  private: false
  model_card: true
  merge_adapters: true
```

When `publish:` is present, `entrenar train config.yaml` automatically runs the publish
pipeline after training completes successfully.

### 3.4 Smart `entrenar init` (Priority: MEDIUM)

```bash
entrenar init --base Qwen/Qwen2.5-Coder-0.5B --method qlora --data ./my-data/
```

Should:
1. Detect model architecture from HF config.json (hidden_size, num_layers, vocab_size)
2. Suggest LoRA rank based on model size (e.g., rank 64 for 0.5B, 128 for 7B)
3. Auto-detect data format (JSONL, Parquet, plain text)
4. Generate config.yaml with sensible defaults
5. Include `publish:` section placeholder

### 3.5 `entrenar push` Alias (Priority: LOW)

Alias for `entrenar publish` to match the git mental model.

## 4. End-to-End Happy Path

### 4.1 The Three-Command Flow

```bash
# Step 1: User has training data in ./data/train.jsonl
$ entrenar init --base Qwen/Qwen2.5-Coder-0.5B --method qlora --data ./data/
  Created config.yaml with:
    Model: Qwen/Qwen2.5-Coder-0.5B (896 hidden, 24 layers)
    Method: QLoRA (rank=64, alpha=16, 4-bit base)
    Data: ./data/train.jsonl (1,234 samples detected)
    Optimizer: AdamW (lr=2e-4, warmup=100 steps)
    Output: ./output/

# Step 2: Train
$ entrenar train config.yaml
  Downloading Qwen/Qwen2.5-Coder-0.5B... [cached]
  Loading model weights (SafeTensors)...
  Applying QLoRA adapters (rank=64)...
  Training:
    Epoch 1/3: loss=2.341, lr=2.0e-4
    Epoch 2/3: loss=1.876, lr=1.5e-4
    Epoch 3/3: loss=1.543, lr=5.0e-5
  Saved: ./output/model.safetensors (1.2 GB)
  Saved: ./output/adapter_config.json
  Saved: ./output/final_model.json

# Step 3: Publish
$ entrenar publish --repo myuser/qwen-rust-coder ./output/
  Merging LoRA adapters with base model...
  Exporting as SafeTensors...
  Creating repo: myuser/qwen-rust-coder
  Uploading model.safetensors (1.2 GB)...
  Uploading README.md (model card)...
  Published: https://huggingface.co/myuser/qwen-rust-coder
```

### 4.2 The One-Command Flow (with `publish:` in YAML)

```bash
$ entrenar train config.yaml
  # ... training output ...
  Auto-publishing to myuser/qwen-rust-coder...
  Published: https://huggingface.co/myuser/qwen-rust-coder
```

## 5. Implementation Plan

### Phase 1: `entrenar publish` CLI (est. 8h)

| Task | File | Description |
|------|------|-------------|
| Add `PublishArgs` | `src/config/cli/core.rs` | New struct + `Command::Publish` variant |
| Add `publish.rs` | `src/cli/commands/publish.rs` | Wire `merge_export_publish()` to CLI |
| Register command | `src/cli/commands/mod.rs` | Add `publish` module + match arm |
| Tests | `src/cli/commands/tests/` | Dry-run publish, invalid args, etc. |

### Phase 2: HF Model ID Auto-Resolve (est. 12h)

| Task | File | Description |
|------|------|-------------|
| Detect HF IDs | `src/config/train/loader.rs` | Is path an HF repo ID or local file? |
| Auto-download | `src/config/train/loader.rs` | Use `HfModelFetcher` to download + cache |
| Config schema update | `src/config/schema.rs` | Document HF ID support in `ModelRef` |
| Tests | `src/config/tests/` | HF ID detection, cache behavior |

### Phase 3: YAML `publish:` Section (est. 8h)

| Task | File | Description |
|------|------|-------------|
| Schema | `src/config/schema.rs` | Add `PublishSpec` to `TrainSpec` |
| Validation | `src/config/validate/validator.rs` | Validate repo ID format |
| Integration | `src/config/train/loader.rs` | Call publish after training |
| Tests | `src/config/tests/` | YAML parsing, publish-after-train |

### Phase 4: Smart `entrenar init` (est. 12h)

| Task | File | Description |
|------|------|-------------|
| Model detection | `src/cli/commands/init.rs` | Fetch HF config.json, detect architecture |
| Data detection | `src/cli/commands/init.rs` | Detect JSONL/Parquet/text format |
| Template generation | `src/yaml_mode/templates.rs` | Model-aware config generation |
| Tests | various | Architecture detection, data format detection |

### Phase 5: Aprender Delegation (est. 40h)

| Task | Entrenar Files to Modify | Aprender Dependency |
|------|--------------------------|---------------------|
| Loss delegation | Delegate forward in `src/train/loss/mse/` | `aprender::loss::*` |
| Metrics delegation | Delete `src/train/metrics/` | `aprender::metrics::*` |
| Pruning delegation | Refactor `src/prune/` | `aprender::pruning::*` |
| HF client delegation | Refactor `src/hf_pipeline/publish/publisher.rs` | Keep in entrenar (pipeline orchestration) |

## 6. Aprender Changes Needed

For entrenar to delegate cleanly, aprender may need minor additions:

| Addition | Location | Reason |
|----------|----------|--------|
| `Loss` trait must be `Send + Sync` | `aprender::loss` | Entrenar uses multi-threaded training |
| `HfPublisher` must support large file uploads | `crate::hf_pipeline` | SafeTensors can be multi-GB |
| Export `Pruner` trait publicly | `aprender::pruning` | Entrenar needs to wrap pruners |
| Add `CausalLMLoss` to aprender (optional) | `aprender::loss` | Or keep in entrenar as extension |

## 7. Non-Goals

- **Multi-GPU distributed training** - Future work, not part of this simplification
- **GUI/web dashboard** - TUI is sufficient; WASM dashboard exists separately
- **Python interop** - Sovereign AI Stack is pure Rust
- **Replacing aprender's NN layers** - Entrenar's transformer blocks are LLM-specific; aprender keeps general-purpose layers
- **Merging the two crates** - They remain separate crates with clear boundaries

## 8. Success Criteria

1. A user with training data can go from zero to a published HF model in under 5 minutes
2. `entrenar publish` works as a single CLI command
3. YAML config accepts HF model IDs (no manual download)
4. `entrenar init` generates a working config for any supported base model
5. Loss functions, metrics, and pruning algorithms are imported from aprender (not duplicated)
6. Test coverage remains above 95%
7. All quality gates pass (A+ TDG, PMAT compliant)

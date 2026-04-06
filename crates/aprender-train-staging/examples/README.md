# Examples

Runnable Rust examples demonstrating entrenar features. All examples compile and run
with `cargo run --example <name>`. Some require feature flags as noted.

## GPU Sharing (GPU-SHARE Spec)

| Example | Description | Features |
|---------|-------------|----------|
| `gpu_ledger` | VRAM ledger with reservation, Display trait, reserve factor (§1.1) | — |
| `multi_adapter_training` | Multi-adapter pipeline: shared base model, round-robin scheduling, per-adapter checkpointing (§2) | — |
| `cluster_training` | Multi-node cluster: placement, SSH launch, checkpoint coordination, MPS validation, health check, adapters-config TOML (§3) | — |

```bash
cargo run --example gpu_ledger
cargo run --example multi_adapter_training
cargo run --example cluster_training
cargo run --example cluster_training -- --config cluster.yaml
```

## Fine-Tuning

| Example | Description | Features |
|---------|-------------|----------|
| `classify_tune_demo` | End-to-end classification pipeline with LoRA on a tiny model | — |
| `finetune_test_gen` | Fine-tune for test generation with specification scoring | — |
| `shell_safety_classify` | Shell command safety classifier with production config | — |
| `finetune_real` | Real HuggingFace model fine-tuning | `hub` |

## LLaMA2

| Example | Description | Features |
|---------|-------------|----------|
| `llama2-train` | LLaMA2 pre-training loop | — |
| `llama2-finetune-lora` | LoRA fine-tuning on LLaMA2 | — |
| `llama2-finetune-qlora` | QLoRA (4-bit) fine-tuning on LLaMA2 | — |
| `llama2-memory-benchmarks` | VRAM usage comparison: full vs LoRA vs QLoRA | — |

## Training Infrastructure

| Example | Description | Features |
|---------|-------------|----------|
| `training_loop` | Core training loop with autograd and optimizer | — |
| `train_from_yaml` | YAML-driven training config | — |
| `train_from_yaml_example` | Extended YAML training with checkpoint loading | — |
| `distillation` | Knowledge distillation (teacher → student) | — |
| `hf_distillation` | HuggingFace model distillation | `hub` |
| `mnist_train` | MNIST digit classifier (parquet data) | `parquet` |
| `mnist_train_gpu` | MNIST on GPU | `gpu`, `parquet` |

## Model Operations

| Example | Description | Features |
|---------|-------------|----------|
| `model_io` | Model save/load round-trip | — |
| `merge_models` | Multi-model merge (SLERP, TIES, DARE) | — |
| `pruning_pipeline` | Structured/unstructured pruning | — |

## Monitoring & Diagnostics

| Example | Description | Features |
|---------|-------------|----------|
| `monitoring` | Real-time training metrics and visualization | — |
| `inference_monitor` | Inference latency/throughput monitoring | — |
| `drift_simulation` | Data drift detection with Andon callbacks | — |
| `calibration_check` | Model calibration verification | — |
| `explainability` | Feature attribution and model explanations | — |

## CLI Tools

| Example | Description | Features |
|---------|-------------|----------|
| `cli_bench` | Throughput benchmarking | — |
| `cli_inspect` | Data file inspection | — |
| `cli_audit` | PII/secret scanning | — |
| `cli_monitor` | Threshold-based drift monitoring | — |

## GPU & CUDA

| Example | Description | Features |
|---------|-------------|----------|
| `cuda_backend` | CUDA backend initialization and capabilities | — |
| `cuda_training_benchmark` | GPU training micro-benchmarks | — |
| `profile_cuda_trainer` | CudaTrainer profiling | — |
| `nvml_test` | NVIDIA Management Library bindings | `nvml` |

## Other

| Example | Description | Features |
|---------|-------------|----------|
| `design_by_contract` | Provable contract pattern demo | — |
| `sovereign` | Sovereign model deployment pipeline | — |
| `research` | Research experiment framework | — |
| `citl` | Continuous-in-the-loop training | `citl` |
| `open_asr_leaderboard` | ASR leaderboard evaluation | — |

## Data Files

- `example-model.gguf` — Mock GGUF model for integration tests
- `train.parquet` — Sample training data for YAML config examples
- `tokenizer.json` — Tokenizer for examples that need token IDs
- `yaml/` — YAML training configs for Toyota Way QA scenarios
- `yaml_fixed/` — Fixed YAML configs matching binary TrainSpec schema

## Running All Examples

```bash
# No feature flags (most examples)
cargo run --example training_loop
cargo run --example cluster_training

# With feature flags
cargo run --example mnist_train --features parquet
cargo run --example hf_distillation --features hub
cargo run --example nvml_test --features nvml
```

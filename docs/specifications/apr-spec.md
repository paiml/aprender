# APR Specification — Single Source of Truth

**Version**: 3.0.0
**Status**: Active
**Created**: 2025-12-16
**Last Updated**: 2026-03-10

> This is the **mono spec** for the APR ecosystem. Each section summarizes the
> component and links to its detailed specification in `components/`. No other
> specification files outside this directory structure are authoritative.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [APR Binary Format](#2-apr-binary-format)
3. [CLI Architecture](#3-cli-architecture)
4. [Merge](#4-merge)
5. [Fine-Tune](#5-fine-tune)
6. [Distill](#6-distill)
7. [Prune](#7-prune)
8. [Quantize](#8-quantize)
9. [Train](#9-train)
10. [Tune (HPO)](#10-tune-hpo)
11. [Serve](#11-serve)
12. [Eval](#12-eval)
13. [Data](#13-data)
14. [Export / Import](#14-export--import)
15. [Inference](#15-inference)
16. [Inspect / Debug / Validate](#16-inspect--debug--validate)
17. [Compile](#17-compile)
18. [QA](#18-qa)
19. [Profile](#19-profile)
20. [Tokenize](#20-tokenize)

---

## 1. Design Principles

- **Row-Major Mandate (LAYOUT-002)**: APR is exclusively row-major. GGUF
  column-major data is transposed at import. Realizar kernels assume row-major.
- **Sovereign AI**: All operations run locally. No data leaves the machine.
  `--offline` flag blocks all network access.
- **Plan/Apply Pattern**: Destructive operations (train, merge, distill, prune)
  support `--plan` mode for dry-run estimation before committing GPU time.
- **Toyota Production System**: Jidoka (stop-on-error), Poka-Yoke (privacy
  tiers), Heijunka (load leveling), Kaizen (continuous optimization).
- **Format Agnostic Input**: Accept APR, GGUF, SafeTensors. Output APR by default.

---

## 2. APR Binary Format

**Detail**: [components/format.md](components/format.md)

APR v2 is a zero-copy binary model format with LZ4/ZSTD tensor compression.

| Section | Content |
|---------|---------|
| Header (32B) | Magic `APR\x02`, version, flags, metadata offset/size |
| Metadata | JSON key-value (arch, vocab_size, hidden_size, etc.) |
| Tensor Index | Binary: name_len(u16) + name + dtype(u8) + ndim(u8) + dims + offset + size |
| Tensor Data | 64-byte aligned, optionally compressed (LZ4/ZSTD) |
| Footer (16B) | Checksum, total size verification |

**Sharding**: Multi-file support for models >2GB. WASM-compatible streaming.

---

## 3. CLI Architecture

**Binary**: `apr` (crate: `apr-cli`)

Commands are organized into groups:
- **Core**: run, chat, serve, pull, list, rm
- **Model Ops**: finetune, prune, distill, merge, quantize
- **Analysis**: inspect, debug, validate, diff, tensors, trace, lint, explain
- **Training**: train (plan/apply/watch/sweep/submit), tune, monitor, runs
- **Evaluation**: eval, bench, profile, qa, parity, qualify
- **Tools**: export, import, convert, compile, hex, tree, flow, data, tokenize
- **Visualization**: tui, cbtop, ptx-map, ptx, probar

---

## 4. Merge

**Detail**: [components/merge.md](components/merge.md)

Combine multiple models into one using weight-space interpolation strategies.

### Implemented Strategies

| Strategy | Description | Models |
|----------|-------------|--------|
| `average` | Simple mean of weights | 2+ |
| `weighted` | User-specified per-model weights | 2+ |
| `slerp` | Spherical linear interpolation | 2 |
| `ties` | Trim, Elect Sign, Merge (task vectors from base) | 2+ |
| `dare` | Drop And REscale with random pruning | 2+ |

### Planned Strategies (Arcee MergeKit Parity)

| Strategy | Description | Ticket |
|----------|-------------|--------|
| `task-arithmetic` | Linear combination of task vectors | GH-442 |
| `nuslerp` | Enhanced SLERP with faster execution | GH-442 |
| `multi-slerp` | Barycentric SLERP for >2 models | GH-442 |
| `della` | Task arithmetic + adaptive magnitude pruning | GH-442 |
| `breadcrumbs` | Task arithmetic + outlier removal | GH-442 |
| `sce` | Adaptive matrix-level weighting (variance-based) | GH-442 |
| `passthrough` | Direct tensor copy for layer stacking / frankenmerge | GH-443 |
| `evolutionary` | CMA-ES optimization of merge configs vs benchmarks | GH-444 |
| `moe` | Construct MoE from dense models (Mixtral-style) | GH-445 |
| `dam` | Differentiable Adaptive Merging (trainable coefficients) | GH-446 |

### Planned Features

- **Tokenizer surgery** (GH-447): Transplant tokenizers between models for
  speculative decoding draft model vocabulary alignment.
- **Per-layer granularity** (GH-452): Specify different strategies or weights per layer.
- **Multi-GPU acceleration**: `--parallel` flag for near-linear speedup.
- **YAML config** (GH-452): Declarative merge specification (MergeKit-compatible).

---

## 5. Fine-Tune

**Detail**: [components/finetune.md](components/finetune.md)

Adapt pre-trained models to domain-specific tasks.

### Implemented Methods

| Method | Description | VRAM |
|--------|-------------|------|
| `auto` | Auto-select based on model size and VRAM | varies |
| `full` | Full parameter fine-tuning | high |
| `lora` | Low-Rank Adaptation (trainable rank-r matrices) | medium |
| `qlora` | QLoRA with NF4 frozen weights (~8x VRAM savings) | low |

### Implemented Features

- Multi-adapter concurrent training (GPU-SHARE Phase 2)
- CUDA MPS for GPU sharing (experimental)
- Multi-GPU data-parallel training
- Distributed training (coordinator/worker)
- Imbalanced dataset oversampling
- Checkpoint format selection (APR, SafeTensors, both)
- Classification task support with configurable class count
- Adapter merge into base model

### Planned Methods (Arcee Parity)

| Method | Description | Ticket |
|--------|-------------|--------|
| `cpt` | Continual Pre-Training on raw text corpora | GH-448 |
| `dpo` | Direct Preference Optimization (preference pairs) | GH-449 |
| `rlvr` | RL on Verifiable Rewards | GH-450 |

### Planned Features

- **Synthetic data generation** (GH-453): EvolKit-style instruction evolution
- **PII filtering** (GH-453): Automatic PII removal from training corpora
- **Domain-specific data filtering** (GH-453): Quality scoring and filtering
- **Checkpoint-to-deployment pipeline**: finetune → eval → serve

---

## 6. Distill

**Detail**: [components/distill.md](components/distill.md)

Transfer knowledge from a large teacher to a smaller student.

### Implemented Strategies

| Strategy | Description |
|----------|-------------|
| `standard` | KL divergence + task loss between teacher/student logits |
| `progressive` | Layer-by-layer progressive distillation with mapping |
| `ensemble` | Multiple teachers with weighted contribution |

### Implemented Features

- Two-stage pipeline: precompute teacher logits, then train student
- YAML config for complex distillation pipelines
- Configurable temperature and alpha weight
- LoRA on student model
- Attention transfer with layer mapping

### Planned Strategies (Arcee DistillKit Parity)

| Strategy | Description | Ticket |
|----------|-------------|--------|
| `hidden-state` | Match intermediate hidden states with linear projection | GH-451 |
| `quantization-aware` | Polynomial approximation + error-diffusion quantization | GH-451 |
| `online` | Concurrent teacher inference (no precompute step) | GH-451 |

---

## 7. Prune

**Detail**: [components/prune.md](components/prune.md)

Remove redundant weights to reduce model size and inference cost.

### Implemented Methods

| Method | Description |
|--------|-------------|
| `magnitude` | Remove smallest-magnitude weights |
| `structured` | Remove entire neurons/attention heads |
| `depth` | Remove entire layers (e.g., `--remove-layers 20-24`) |
| `width` | Reduce hidden dimensions |
| `wanda` | Weights And Activations (calibration-based) |
| `sparsegpt` | SparseGPT (second-order pruning) |

### Features

- Analyze mode (identify pruning opportunities without executing)
- Plan mode (estimate impact)
- Calibration data support
- Target ratio and sparsity level controls

---

## 8. Quantize

**Detail**: [components/quantize.md](components/quantize.md)

Reduce weight precision for smaller models and faster inference.

### Implemented Schemes

| Scheme | Bits | Description |
|--------|------|-------------|
| `fp16` | 16 | Half-precision floating point |
| `int8` | 8 | Symmetric 8-bit integer |
| `int4` | 4 | 4-bit integer |
| `q4k` | 4 | Q4_K super-block format (fused kernel compatible) |

### Features

- Output format selection (APR, GGUF, SafeTensors)
- Batch quantization (multiple schemes in one pass)
- Plan mode (estimate compression ratio)
- Streaming quantization for large models (GH-434: in progress)

---

## 9. Train

**Detail**: [components/train.md](components/train.md)

Full training pipeline with plan/apply pattern.

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `plan` | Generate training plan (validate data, estimate resources) |
| `apply` | Execute training plan (allocate GPU, run trials) |
| `watch` | Monitor with auto-restart on crash and hang detection |
| `sweep` | Generate hyperparameter sweep configs |
| `archive` | Package checkpoint into release bundle |
| `submit` | Submit multi-adapter jobs to cluster (GPU-SHARE Phase 3) |
| `cluster-status` | Show cluster nodes, GPUs, adapter capacity |

### Features

- HPO strategies: TPE, grid, random, manual
- Distributed data-parallel training
- Deterministic training mode (bitwise reproducibility)
- YAML config for pre-training and fine-tuning
- Scout mode (1 epoch per trial for fast exploration)

---

## 10. Tune (HPO)

**Detail**: [components/tune.md](components/tune.md)

Hyperparameter optimization with automatic search.

- Strategies: TPE, grid, random
- Schedulers: ASHA (early stopping), median, none
- Scout → full two-phase workflow
- Time limits and budget controls
- Warm-start from previous results

---

## 11. Serve

**Detail**: [components/serve.md](components/serve.md)

Inference server with OpenAI-compatible API.

- Plan/run subcommands
- Privacy tier enforcement (Sovereign/Private/Standard)
- Failover and circuit breakers
- Streaming responses

---

## 12. Eval

**Detail**: [components/eval.md](components/eval.md)

Model evaluation and benchmarking.

| Mode | Description |
|------|-------------|
| Perplexity | PPL on wikitext-2, lambada, custom text |
| Classification | Accuracy, F1, confusion matrix on JSONL data |
| Pass@k | Code generation evaluation with sampling |
| Throughput | tok/s benchmarking (≥10 tok/s spec) |

### Planned (Arcee Parity)

- **lm-eval harness integration** (GH-454): Standard benchmarks (HellaSwag, MMLU, TruthfulQA)
- **Model drift detection** (GH-454): Compare production outputs over time

---

## 13. Data

**Detail**: [components/data.md](components/data.md)

Data quality pipeline powered by alimentar.

- Subcommands: load, validate, transform, statistics
- JSONL format for training data
- Split, balance, audit operations
- Integration with tokenizer for vocab analysis

### Planned (Arcee EvolKit Parity)

- **Synthetic data generation**: Instruction complexity enhancement
- **PII detection and filtering**: Automatic PII removal from training corpora
- **Data mixing optimization**: Weighted sampling across sources

---

## 14. Export / Import

**Detail**: [components/format-conversion.md](components/format-conversion.md)

Convert between model formats.

### Export Targets

APR → SafeTensors, GGUF, MLX, ONNX, OpenVINO, CoreML

### Import Sources

HuggingFace (hf://), SafeTensors, GGUF (with Q4K preservation), URLs

### Features

- Architecture auto-detection (14+ architectures)
- Provenance chain enforcement (`--enforce-provenance`)
- Quantization during export/import
- Batch export to multiple formats

---

## 15. Inference

**Detail**: [components/inference.md](components/inference.md)

Direct model execution via `apr run` and `apr chat`.

- Streaming output
- Chat template support (ChatML)
- GPU/CPU selection
- Inference tracing (APR-TRACE-001)
- Roofline profiling
- Offline mode (sovereign compliance)

---

## 16. Inspect / Debug / Validate

**Detail**: [components/inspection.md](components/inspection.md)

Model introspection tools.

| Command | Purpose |
|---------|---------|
| `inspect` | Metadata, vocab, structure, weights |
| `debug` | Drama mode, hex dump, ASCII extraction |
| `validate` | Integrity check, 100-point quality score |
| `diff` | Two-model comparison (metadata, weights, values) |
| `tensors` | List tensor names, shapes, statistics |
| `trace` | Layer-by-layer analysis with reference comparison |
| `lint` | Best practices checking |
| `explain` | Error codes, tensors, kernel dispatch explanation |
| `hex` | Format-aware binary forensics |
| `tree` | Architecture tree visualization |

---

## 17. Compile

**Detail**: [components/compile.md](components/compile.md)

Compile model into standalone executable binary.

- Cross-compilation targets (x86_64, aarch64, wasm32)
- Quantization during compilation
- Release mode with LTO and stripping
- Self-contained binary with embedded weights

---

## 18. QA

**Detail**: [components/qa.md](components/qa.md)

Falsifiable quality assurance pipeline.

- 8+ gate checklist (golden output, throughput, parity, contracts)
- Regression detection against previous reports
- GPU/CPU parity validation
- Cross-format parity testing
- PTX parity validation
- CI integration with JSON output and exit codes

---

## 19. Profile

**Detail**: [components/profile.md](components/profile.md)

Performance analysis and optimization.

- Roofline analysis
- Flamegraph generation
- Naive implementation detection
- Energy measurement (RAPL)
- CI assertion mode (throughput/latency thresholds)
- Baseline comparison (Ollama, llama.cpp, vLLM, any OpenAI-compatible)

---

## 20. Tokenize

**Detail**: [components/tokenize.md](components/tokenize.md)

Tokenizer operations.

- BPE vocabulary training pipeline (plan/apply)
- Token breakdown with vocabulary lookup
- Integration with data pipeline for corpus analysis

---

## Appendix A: Component Spec Index

| Component | File | Status |
|-----------|------|--------|
| APR Binary Format | [components/format.md](components/format.md) | Active |
| Merge | [components/merge.md](components/merge.md) | Active |
| Fine-Tune | [components/finetune.md](components/finetune.md) | Active |
| Distill | [components/distill.md](components/distill.md) | Active |
| Prune | [components/prune.md](components/prune.md) | Active |
| Quantize | [components/quantize.md](components/quantize.md) | Active |
| Train | [components/train.md](components/train.md) | Active |
| Tune (HPO) | [components/tune.md](components/tune.md) | Active |
| Serve | [components/serve.md](components/serve.md) | Active |
| Eval | [components/eval.md](components/eval.md) | Active |
| Data | [components/data.md](components/data.md) | Active |
| Export / Import | [components/format-conversion.md](components/format-conversion.md) | Active |
| Inference | [components/inference.md](components/inference.md) | Active |
| Inspection | [components/inspection.md](components/inspection.md) | Active |
| Compile | [components/compile.md](components/compile.md) | Active |
| QA | [components/qa.md](components/qa.md) | Active |
| Profile | [components/profile.md](components/profile.md) | Active |
| Tokenize | [components/tokenize.md](components/tokenize.md) | Active |

## Appendix B: Archived Specifications

Legacy specs remain in `docs/specifications/` root for historical reference.
They are **not authoritative** — this mono spec and the component specs in
`components/` are the single source of truth.

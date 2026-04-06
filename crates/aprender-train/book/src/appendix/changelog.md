# Changelog

All notable changes to Entrenar will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.6] - 2026-01-22

### Changed

#### Code Organization (Technical Debt Reduction)
- **Exploded 16 large modules** (>1000 lines) into directory module structures
  - `storage/sqlite/backend.rs` → `backend/` (sqlite_backend, state, tests)
  - `config/train.rs` → `train/` (loader, batches, arrow, demo, tests)
  - `eval/evaluator.rs` → `evaluator/` (metric, config, result, leaderboard, kfold, model_evaluator, tests)
  - `hf_pipeline/fetcher.rs` → `fetcher/` (types, options, hf_fetcher, tests)
  - `efficiency/device.rs` → `device/` (simd, cpu, gpu, tpu, apple, compute, tests)
  - `monitor/llm.rs` → `llm/` (error, metrics, prompt, eval_result,
    traits, stats, memory_evaluator, heuristics, tests)
  - `monitor/inference/provenance.rs` → `provenance/`
    (node, edge, graph, attack, reconstructor, tests)
  - `citl/pattern_store.rs` → `pattern_store/` (chunk_id, fix_pattern, suggestion, config, store, data, tests)
  - `storage/registry.rs` → `registry/` (stage, version, comparison, transition, policy, error, traits, memory)
  - `citl/trainer.rs` → `trainer/` (span, trace, outcome, correlation, config, stats, citl)
  - `tokenizer/mod.rs` → separate files (error, config, traits, bpe, char, hf)
  - `monitor/wasm.rs` → `wasm/` (collector, options, dashboard, utils)
  - Plus 4 from previous release: preflight, cloud, code_gan, manifest

- **Extracted main.rs CLI logic** into `src/cli/` library module
  - `main.rs` reduced from 1656 lines to 41-line shim
  - Created `cli/commands/` with modular command handlers
  - Split `research.rs` (534 lines) into research/ subdirectory

### Quality
- **3727 tests passing** (100% success rate)
- **96.66% code coverage** (exceeds 95% target)
- **PMAT compliance: COMPLIANT**
- **File health**: No files >500 lines in src/ (100% compliance)
- All API compatibility maintained via re-exports

### Dependencies
- Updated PAIML stack dependencies

[0.5.6]: https://github.com/paiml/entrenar/releases/tag/v0.5.6

## [0.1.0] - 2025-11-21

### Added

#### Core Framework
- **Autograd Engine** - Tape-based automatic differentiation with backward propagation
  - Tensor abstraction with gradient tracking
  - BackwardOp trait for custom operations
  - Attention, matmul, softmax, layer norm operations
  - Property-based gradient checking (200K+ iterations)

#### Optimizers
- **SGD** with momentum support
- **Adam** optimizer with bias correction
- **AdamW** with decoupled weight decay
- **Gradient clipping** via L2 norm
- **Learning rate scheduling** (Cosine, Linear)
- **SIMD acceleration** for parameter updates via Trueno
- Convergence property tests for all optimizers

#### LoRA & QLoRA
- **LoRA layers** with configurable rank and alpha
- **QLoRA** with 4-bit quantized base weights
- **Adapter management** (save/load separately from base model)
- **Memory benchmarks** showing 4× reduction with QLoRA
- **Gradient flow tests** ensuring proper backpropagation

#### Quantization
- **QAT (Quantization-Aware Training)** with fake quantize
- **PTQ (Post-Training Quantization)** with calibration
- **4-bit and 8-bit** quantization support
- **Symmetric and asymmetric** quantization modes
- **Per-channel and per-tensor** quantization
- Compression ratio validation and accuracy degradation tests

#### Model Merging (Arcee Methods)
- **TIES** (Task Inference via Elimination and Sign voting)
- **DARE** (Drop And REscale with Bernoulli masking)
- **SLERP** (Spherical Linear intERPolation)
- Property tests for permutation invariance
- Multi-model ensemble support

#### Knowledge Distillation
- **Temperature-scaled KL divergence** loss
- **Multi-teacher ensemble** distillation
- **Progressive layer-wise** distillation
- **44 distillation tests** including 13 property tests
- Temperature smoothing validation

#### Declarative Configuration
- **YAML-based training** configuration (Ludwig-style)
- **Schema validation** with comprehensive error messages
- **Auto-inference** of feature types from data
- **Single-command training** via `train_from_yaml()`
- Builder pattern for optimizers and models from config

#### Training Loop
- **High-level Trainer** abstraction
- **Batch processing** with configurable batch size
- **Metrics tracking** (loss history, learning rates, steps)
- **Gradient clipping** integration
- **Learning rate scheduling** during training
- **train_step()** and **train_epoch()** methods

#### Model I/O
- **Save/load models** with multiple formats
  - **JSON** (pretty-printed or compact)
  - **YAML** for human-readable configs
  - Placeholder for **GGUF** (future Realizar integration)
- **ModelMetadata** with custom fields
- **Round-trip integrity** validation
- Automatic format detection from file extension

### Testing & Quality
- **258 tests** passing (100% success rate)
  - Unit tests for all modules
  - Integration tests for end-to-end workflows
  - Property-based tests (200K+ iterations)
  - Gradient correctness validation
  - Round-trip serialization tests
- **0 clippy warnings** (strict mode)
- **0 TODOs** remaining in codebase
- **55 Rust source files** with full documentation

### Examples
- **training_loop.rs** - Demonstrates Trainer API
- **model_io.rs** - Save/load workflow
- **train_from_yaml_example.rs** - Declarative training
- **distillation.rs** - Knowledge distillation
- **merge_models.rs** - Model merging methods
- **train_from_yaml.rs** - YAML configuration
- Plus LLAMA2 examples (train, finetune-lora, finetune-qlora, memory-benchmarks)

### Documentation
- Comprehensive API documentation for all public modules
- README with quick start guide
- Specification documents for all major components
- Example configurations (config.yaml)

### Dependencies
- **trueno 0.4.1** - SIMD-accelerated compute engine
- **ndarray 0.16** - N-dimensional arrays
- **serde 1.0** - Serialization framework
- **thiserror 2.0** - Error handling
- **proptest 1.4** - Property-based testing (dev)
- **tempfile 3.8** - Testing utilities (dev)

### Notes
- This is the initial release of Entrenar
- GGUF loading requires future Realizar integration
- Real data loading (Parquet/CSV) to be added
- Performance benchmarks to be published

[0.1.0]: https://github.com/paiml/entrenar/releases/tag/v0.1.0

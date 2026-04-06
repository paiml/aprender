# Changelog

All notable changes to Entrenar will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.4] - 2026-02-28

### Added

- `ClassifyTrainer::auto_balance_classes()` — auto-detects class imbalance (ratio >2:1) and applies `SqrtInverse` weights when no explicit weights configured

#### Classification Evaluation Harness
- **`ClassifyEvalReport`**: Comprehensive evaluation report with 13 metrics across 4 categories
  - Accuracy & Agreement: accuracy, top-2 accuracy, Cohen's kappa, MCC (all with bootstrap 95% CIs)
  - Per-class: precision, recall, F1, support for each class
  - Proper scoring rules: Brier score (multi-class MSE), log loss (negative log-likelihood)
  - Calibration: ECE, mean confidence, confidence gap, reliability diagram
  - Baselines: random (1/K), majority-class, lift over majority
  - Error analysis: top-N most confused class pairs from confusion matrix
- **`evaluate_checkpoint()`**: Standalone function to load a LoRA adapter checkpoint and evaluate against a JSONL test set
  - Resolves base model path from `adapter_config.json`
  - Builds `ClassifyPipeline::from_pretrained()` and runs `forward_only_with_probs()`
- **`to_report()`**: sklearn-style text classification report with per-class rows and macro/weighted averages
- **`to_json()`**: Machine-readable JSON with all metrics, confidence intervals, calibration data
- **`to_model_card()`**: Publication-quality HuggingFace README.md with YAML front matter, `model-index` metrics, confusion matrix (raw + normalized), calibration curve, error analysis, intended use, limitations (auto-detected weak classes), ethical considerations
- Bootstrap CIs use deterministic LCG PRNG (seed=42, 1,000 resamples) for reproducible results

### Changed
- `from_predictions_with_probs()` refactored into focused helper methods for complexity compliance
- `to_model_card()` decomposed from 217-line monolithic method into 14 section helpers
- Removed unused `from_predictions()` (no-probs fallback) — all evaluation paths now use probability-aware methods

## [0.7.3] - 2026-02-28

### Added

#### HuggingFace-Complete Checkpoint Output
- **config.json**: Auto-generated HF model architecture config (architectures, model_type, num_labels, hidden_size, etc.) from `TransformerConfig`
- **adapter_config.json**: PEFT-compatible LoRA adapter config (rank, alpha, target_modules, task_type=SEQ_CLS) via `PeftAdapterConfig`
- **tokenizer.json**: Copied from base model directory when loaded via `from_pretrained()`
- **model_dir accessor**: `ClassifyPipeline::model_dir()` returns the base model path for tokenizer copying

Checkpoints are now directly publishable to HuggingFace Hub:
```bash
apr finetune --task classify ... -o ./checkpoints/
apr publish ./checkpoints/best/ paiml/shell-safety-classifier
```

### Changed
- `ClassifyPipeline` now stores `model_dir: Option<PathBuf>` (set by `from_pretrained()`)
- `ClassifyTrainer::save_checkpoint()` now produces 6 files: model.safetensors, model.apr, metadata.json, config.json, adapter_config.json, tokenizer.json

## [0.7.2] - 2026-02-26

### Changed
- Bumped trueno dependency to 0.16.0
- Bumped trueno-viz dependency to 0.2.x
- Bumped aprender dependency to 0.27
- Bumped realizar dependency to 0.8
- Minor version bump for PAIML Sovereign AI Stack coordinated release

## [0.7.1] - 2026-02-26

### Fixed
- Fixed trueno-viz dependency to 0.2.x for WithDimensions trait compatibility

## [0.7.0] - 2026-02-26

### Changed
- Major dependency updates for stack release

## [0.5.3] - 2026-01-21

### Added
#### Model Evaluation Framework (APR-073)
- **Standardized Metrics** - Classification (Accuracy, F1, Precision, Recall, Confusion Matrix) with sklearn parity
  (1e-6 precision).
- **Drift Detection** - KS, Chi-Square, and PSI (Population Stability Index) statistical tests for data/concept drift
  detection.
- **ModelEvaluator** - Multi-model comparison and leaderboard generation.
- **Entrenar Integration** - `AutoRetrainer` with Andon loop; <10ms callback latency verified.
- **Renacer Tracing** - Feature-gated tracing for evaluation overhead monitoring.
- **WASM Compatibility** - Core evaluation logic compiles to `wasm32-unknown-unknown`.
- **Performance** - O(N) complexity for all metrics; verified zero-allocation hot loops in WASM.

### Changed
- Total test count increased to 3,000+ passing library tests.
- 17 new property tests with 100,000 iterations each for metric stability.

### Fixed
- WASM build for `hf-hub` and `memmap2` via platform-specific dependency gating.

## [0.2.3] - 2025-12-02

### Fixed

#### Dependency Resolution
- **Arrow conflict resolved** - Updated renacer and aprender to fix arrow version conflicts
- **Switched to crates.io** - All PAIML stack dependencies now use published versions (no path/git deps)

### Changed
- Updated `aprender` from 0.12.0 to 0.14.0
- Updated `renacer` from 0.6.x to 0.7.0
- Test count: 2155 passing

## [0.2.2] - 2025-11-29

### Added

#### SafeTensors Support (ENT-IO)
- **SafeTensors format** - Full save/load support with metadata preservation
- **HuggingFace compatibility** - Direct upload/download from HuggingFace Hub
- **Auto-detection** - Format automatically detected from file extension
- **Metadata serialization** - Model name, architecture, version in headers

#### HuggingFace Pipeline
- **HfModelFetcher** - Download models from HuggingFace Hub
- **SafeTensors parsing** - Load models in SafeTensors format
- **Projection layers** - Handle dimension-mismatched distillation

#### Sub-Crate Demos
- **entrenar-common** - CLI utilities, progress bars, structured output (37 tests)
- **entrenar-lora** - LoRA optimization and memory planning (53 tests)
- **entrenar-inspect** - Model inspection and validation (64 tests)
- **entrenar-shell** - Interactive REPL shell (55 tests)
- **entrenar-bench** - Distillation benchmarking (52 tests)
- **entrenar-distill** - Distillation pipeline demo

### Changed
- All clippy warnings resolved across workspace
- Test count increased to 1,900+
- Book documentation updated with SafeTensors chapter

### Fixed
- `&PathBuf` → `&Path` for proper slice semantics
- Type complexity warnings with allow attributes
- Manual clamp patterns replaced with `.clamp()`
- Display trait implemented for TestResult

## [0.2.0] - 2025-11-28

### Added

#### Training Loop (Phase 8: ENT-048 to ENT-052)
- **Trainer** - High-level training abstraction with epoch/step methods
- **CallbackManager** - Composable callback system for training events
- **EarlyStopping** - Patience-based early stopping with min_delta threshold
- **CheckpointCallback** - Periodic model saving with best model tracking
- **ProgressCallback** - Training progress logging with configurable frequency
- **MonitorCallback** - NaN/Inf detection and Andon alert integration
- **MetricsTracker** - Loss history, learning rate, and step tracking

#### Explainability Integration (Phase 9: ENT-053)
- **ExplainabilityCallback** - Feature attribution during training evaluation
  - Permutation Importance
  - Integrated Gradients
  - Saliency Maps
- Integration with aprender::interpret module

#### CITL Integration (entrenar → depyler)
- **TieredCurriculum** - Automatic tier advancement (60%→70%→80% thresholds)
- **AdaptiveCurriculum** - Error-based tier selection with Feldman weighting
- **efficiency_score()** - E(T) = Accuracy / log(CorpusSize) metric

### Changed
- Upgraded trueno dependency to 0.7.3
- Upgraded aprender dependency to 0.12.0
- Test count increased from 258 to 800+

### Quality
- 800+ tests passing
- >90% code coverage
- Property tests: 200K+ iterations
- Mutation testing: >80% kill rate

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

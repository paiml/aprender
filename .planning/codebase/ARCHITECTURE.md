# Architecture

**Analysis Date:** 2026-08-07

## Pattern Overview

**Overall:** Contract-first, layered Rust workspace with a thin public facade and CLI, domain-specific ML engines, a sovereign APR container leaf, and hardware-specific compute adapters.

**Key Characteristics:**
- Keep installation and public re-export concerns thin: `src/lib.rs` re-exports `aprender-core`, while `src/bin/apr.rs` delegates to `apr-cli`.
- Separate model-container mechanics from model semantics: `crates/apr-format/src/` owns byte-level APR v2 I/O, while `crates/aprender-core/src/format/` owns conversion, dequantization, model-family interpretation, and validation.
- Route user operations through centralized command validation and dispatch in `crates/apr-cli/src/dispatch_run.rs`, `crates/apr-cli/src/dispatch.rs`, and `crates/apr-cli/src/dispatch_analysis.rs`.
- Keep training, inference, data, registry, compute, GPU, and contracts in distinct crates under `crates/`; join them through explicit public APIs and workspace dependencies declared in `Cargo.toml`.
- Treat YAML contracts in `contracts/` and their Rust bindings in `crates/aprender-contracts/src/` as executable cross-layer specifications, not passive documentation.
- Gate optional capabilities at crate and command boundaries through Cargo features in `Cargo.toml` and `crates/apr-cli/Cargo.toml`.

## Layers

**Facade and Installation Layer:**
- Purpose: Present the `aprender` library crate and installable `apr` binary without duplicating implementation.
- Location: `src/lib.rs`, `src/bin/apr.rs`, `Cargo.toml`
- Contains: Workspace facade re-exports, binary startup, and top-level feature forwarding.
- Depends on: `crates/aprender-core/` and feature-gated `crates/apr-cli/`.
- Used by: Library consumers importing `aprender` and users installing or running `apr`.

**CLI Definition and Dispatch Layer:**
- Purpose: Parse commands, enforce pre-dispatch model contracts, map feature gates, build operation-specific configurations, and convert failures to stable process exit codes.
- Location: `crates/apr-cli/src/lib.rs`, `crates/apr-cli/src/commands_enum.rs`, `crates/apr-cli/src/dispatch_run.rs`, `crates/apr-cli/src/dispatch.rs`, `crates/apr-cli/src/dispatch_analysis.rs`, `crates/apr-cli/src/error.rs`
- Contains: Clap command trees, command fragments, two-stage dispatch, source/model resolution, output routing, and CLI error translation.
- Depends on: High-level APIs from `crates/aprender-core/`, `crates/aprender-train/`, `crates/aprender-serve/`, `crates/aprender-data/`, `crates/aprender-registry/`, `crates/aprender-orchestrate/`, and related presentation crates.
- Used by: `src/bin/apr.rs`, `crates/apr-cli/src/main.rs`, and CLI-focused integration tests under `crates/apr-cli/tests/`.

**Command Implementation Layer:**
- Purpose: Implement individual user-facing actions after argument parsing and shared validation.
- Location: `crates/apr-cli/src/commands/`, including `crates/apr-cli/src/commands/run.rs`, `crates/apr-cli/src/commands/inference_output.rs`, `crates/apr-cli/src/commands/serve/`, and `crates/apr-cli/src/commands/pipeline.rs`
- Contains: Model operations, run/serve adapters, inspection, conversion, training, data, registry, diagnostics, and external pipeline orchestration.
- Depends on: Command argument types in `crates/apr-cli/src/` and domain crates under `crates/`.
- Used by: Core dispatch in `crates/apr-cli/src/dispatch.rs` and extended dispatch in `crates/apr-cli/src/dispatch_analysis.rs`.

**Core Machine-Learning Layer:**
- Purpose: Provide classical ML algorithms, neural-network building blocks, preprocessing, model selection, serialization adapters, model-family logic, and the public estimator contracts.
- Location: `crates/aprender-core/src/`
- Contains: Algorithm families such as `crates/aprender-core/src/classification/`, `crates/aprender-core/src/linear_model/`, `crates/aprender-core/src/tree/`, `crates/aprender-core/src/nn/`, `crates/aprender-core/src/preprocessing/`, and `crates/aprender-core/src/model_selection/`.
- Depends on: Lower-level workspace crates including `crates/apr-format/`, `crates/aprender-compute/`, `crates/aprender-quant/`, and selected data/training support crates declared in `crates/aprender-core/Cargo.toml`.
- Used by: The facade in `src/lib.rs`, CLI commands in `crates/apr-cli/src/commands/`, training in `crates/aprender-train/`, and serving in `crates/aprender-serve/`.

**APR Container Layer:**
- Purpose: Read, validate, and write the dependency-light APR v2 byte container without importing high-level ML or inference policy.
- Location: `crates/apr-format/src/lib.rs`, `crates/apr-format/src/v2/`, `crates/apr-format/src/validate.rs`
- Contains: The 64-byte APR header, aligned tensor index/data regions, typed metadata, checksums, borrowed and owned readers, and buffered/streaming writers.
- Depends on: External serialization/checksum primitives declared in `crates/apr-format/Cargo.toml`; it has no internal workspace crate dependency.
- Used by: High-level format and serialization code in `crates/aprender-core/src/format/` and `crates/aprender-core/src/serialization/apr/`, plus APR consumers in training and serving.

**Model Format and Lifecycle Layer:**
- Purpose: Add model semantics to APR and bridge APR, GGUF, SafeTensors, ONNX, sharded artifacts, and model-family conventions.
- Location: `crates/aprender-core/src/format/`, `crates/aprender-core/src/loading/`, `crates/aprender-core/src/serialization/apr/`
- Contains: `RosettaStone`, converters, import/export paths, dequantization extensions, sharding, linting, validated tensors, model-family detection, and atomic convenience writers.
- Depends on: `crates/apr-format/src/v2/` for the container and model/quantization modules in `crates/aprender-core/src/` for semantic interpretation.
- Used by: Lifecycle commands in `crates/apr-cli/src/commands/`, training I/O in `crates/aprender-train/src/io/`, and inference loaders in `crates/aprender-serve/src/`.

**Training Layer:**
- Purpose: Own autograd, optimizers, trainer loops, LoRA/QLoRA, distillation, pruning, evaluation, checkpoint policy, and CPU/CUDA transformer training.
- Location: `crates/aprender-train/src/`
- Contains: Generic trainers in `crates/aprender-train/src/train/trainer/`, contract-driven pretraining in `crates/aprender-train/src/train/pretrain.rs`, concrete transformer trainers in `crates/aprender-train/src/train/transformer_trainer/`, and model I/O in `crates/aprender-train/src/io/`.
- Depends on: Core model types from `crates/aprender-core/`, data pipelines from `crates/aprender-data/`, and feature-gated device implementations from `crates/aprender-gpu/` and `crates/aprender-compute/`.
- Used by: Training commands dispatched from `crates/apr-cli/src/dispatch_analysis.rs` and specialized training-family crates such as `crates/aprender-train-lora/` and `crates/aprender-train-distill/`.

**Inference and Serving Layer:**
- Purpose: Detect model formats, prepare tokens, load CPU/GPU model representations, generate outputs, schedule requests, and expose HTTP-compatible serving APIs.
- Location: `crates/aprender-serve/src/`
- Contains: High-level inference in `crates/aprender-serve/src/infer/`, APR loading in `crates/aprender-serve/src/apr/`, GGUF/SafeTensors paths in `crates/aprender-serve/src/gguf/` and `crates/aprender-serve/src/safetensors/`, and HTTP routes in `crates/aprender-serve/src/api/`.
- Depends on: Model semantics from `crates/aprender-core/`, kernels and vector operations from `crates/aprender-compute/`, low-level device support from `crates/aprender-gpu/`, and quantization from `crates/aprender-quant/`.
- Used by: `apr run` through `crates/apr-cli/src/commands/inference_output.rs` and `apr serve` through `crates/apr-cli/src/commands/serve/`.

**Portable Compute Layer:**
- Purpose: Provide safe vectors, matrices, verified compute bricks, runtime CPU SIMD dispatch, and hardware-neutral inference operations.
- Location: `crates/aprender-compute/src/lib.rs`, `crates/aprender-compute/src/vector/`, `crates/aprender-compute/src/matrix/`, `crates/aprender-compute/src/brick/`, `crates/aprender-compute/src/backends/`
- Contains: Public `Vector`/`Matrix` abstractions, scalar/SSE2/AVX/AVX2/AVX512/NEON/WASM implementations, operation dispatch, execution graphs, profiling, and quantized compute paths.
- Depends on: Backend modules isolated below safe public types in `crates/aprender-compute/src/backends/`.
- Used by: Algorithms in `crates/aprender-core/`, training in `crates/aprender-train/`, and inference in `crates/aprender-serve/`.

**GPU and Device Layer:**
- Purpose: Isolate device discovery, CUDA driver FFI, RAII memory and stream ownership, PTX/kernel construction, compute graphs, GPU monitoring, and alternate backend interfaces.
- Location: `crates/aprender-gpu/src/lib.rs`, `crates/aprender-gpu/src/backend/`, `crates/aprender-gpu/src/driver/`, `crates/aprender-gpu/src/kernels/`, `crates/aprender-gpu/src/graph/`
- Contains: The `Backend` capability interface, CUDA/WGPU/Metal/Vulkan backend modules, safe wrappers around `crates/aprender-gpu/src/driver/sys/`, device buffers, streams, modules, cuBLAS adapters, and reusable graphs.
- Depends on: Feature-gated native device libraries declared in `crates/aprender-gpu/Cargo.toml`.
- Used by: Device selection in `crates/aprender-train/src/train/device.rs`, CUDA trainers in `crates/aprender-train/src/train/transformer_trainer/`, and GPU inference in `crates/aprender-serve/src/gpu/` and `crates/aprender-serve/src/cuda/`.

**Data and Artifact Layer:**
- Purpose: Represent datasets, batch loading, transforms, storage backends, model registry state, and artifact lookup independent of CLI rendering.
- Location: `crates/aprender-data/src/`, `crates/aprender-registry/src/`, `crates/aprender-db/src/`
- Contains: Dataset/dataloader/split APIs in `crates/aprender-data/src/dataset.rs`, `crates/aprender-data/src/dataloader.rs`, and `crates/aprender-data/src/split.rs`, plus registry and persistence services in their respective crates.
- Depends on: Storage and serialization dependencies declared in each crate manifest under `crates/`.
- Used by: Training loaders in `crates/aprender-train/src/`, model-management commands in `crates/apr-cli/src/commands/`, and serving registry state in `crates/aprender-serve/src/api/mod.rs`.

**Contract Layer:**
- Purpose: Parse, validate, generate, bind, audit, and compose executable contracts across models, tensors, kernels, training, CLI behavior, and lifecycle operations.
- Location: `contracts/`, `crates/aprender-contracts/src/`, `crates/aprender-contracts-macros/src/`, `crates/aprender-contracts-cli/src/`
- Contains: YAML specifications such as `contracts/apr-model-lifecycle-v1.yaml` and `contracts/tensor-layout-v1.yaml`, schema parsing in `crates/aprender-contracts/src/schema/`, function bindings in `crates/aprender-contracts/src/binding.rs`, and pipeline obligations in `crates/aprender-contracts/src/pipeline.rs`.
- Depends on: Contract schema/types in `crates/aprender-contracts/src/` and proc-macro expansion in `crates/aprender-contracts-macros/src/lib.rs`.
- Used by: Generated bindings such as `crates/aprender-core/src/generated_contracts.rs`, `crates/aprender-train/src/generated_contracts.rs`, `crates/aprender-serve/src/generated_contracts.rs`, and `crates/apr-cli/src/generated_contracts.rs`.

## Data Flow

**Classical ML Fit and Predict:**

1. Callers construct an estimator exposed through `crates/aprender-core/src/lib.rs` or the curated exports in `crates/aprender-core/src/prelude.rs`.
2. Preprocessing components implement the `Transformer` contract from `crates/aprender-core/src/traits.rs`; estimators implement `Estimator` or `UnsupervisedEstimator` from the same file.
3. Algorithms operate through numerical types and backends provided by `crates/aprender-compute/src/lib.rs` and return model-specific predictions or scores.
4. Persistence uses model-specific serialization or the APR adapters in `crates/aprender-core/src/serialization/apr/`.

**CLI Model Lifecycle:**

1. `src/bin/apr.rs` calls `apr_cli::cli_main()` in `crates/apr-cli/src/lib.rs`; the standalone crate binary in `crates/apr-cli/src/main.rs` reaches the same command execution path.
2. Clap parses command variants assembled from `crates/apr-cli/src/commands_enum.rs`, `crates/apr-cli/src/model_ops_commands.rs`, `crates/apr-cli/src/extended_commands.rs`, and feature-gated command fragments.
3. `execute_command` in `crates/apr-cli/src/dispatch_run.rs` extracts applicable model paths and invokes `validate_model_contract` from `crates/apr-cli/src/validate.rs` unless the command is exempt or contract checks are explicitly skipped.
4. Core actions route through `dispatch_core_command` in `crates/apr-cli/src/dispatch.rs`; remaining action families route through `dispatch_extended_command` in `crates/apr-cli/src/dispatch_analysis.rs`.
5. Command modules under `crates/apr-cli/src/commands/` invoke core format, registry, training, serving, or data APIs and return `CliError` for stable exit-code translation in `crates/apr-cli/src/error.rs`.

**Training to APR Checkpoint:**

1. Data is represented and batched through `crates/aprender-data/src/dataset.rs` and `crates/aprender-data/src/dataloader.rs`, then adapted to the generic or transformer training APIs in `crates/aprender-train/src/train/`.
2. `Trainer` in `crates/aprender-train/src/train/trainer/core.rs` owns parameters, optimizer, optional loss, metrics, callbacks, and configuration; its loop in `crates/aprender-train/src/train/trainer/train_loop/basic.rs` accumulates gradients, clips, steps, and evaluates stopping rules.
3. Pretraining workflows use `PretrainLoop` and the `StepFn`, `ValFn`, and `CheckpointFn` boundaries in `crates/aprender-train/src/train/pretrain.rs`; CPU and CUDA implementations enter through `crates/aprender-train/src/train/pretrain_real.rs` and `crates/aprender-train/src/train/pretrain_real_cuda.rs`.
4. Explicit device requests are resolved by `resolve_device` in `crates/aprender-train/src/train/device.rs`; explicit CUDA requests fail when unavailable instead of silently selecting CPU.
5. Model output passes through `save_model` in `crates/aprender-train/src/io/save.rs` or transformer-specific `save_apr` methods in `crates/aprender-train/src/train/transformer_trainer/`; APR output is written through `AprWriter` in `crates/aprender-core/src/serialization/apr/`.
6. The convenience APR writer uses a sibling temporary file, synchronizes it, and renames it atomically in `crates/aprender-core/src/serialization/apr/mod.rs`.

**APR Load and Inference:**

1. `apr run` resolves local or cached model sources in `crates/apr-cli/src/commands/run.rs` and delegates real inference from `crates/apr-cli/src/commands/inference_output.rs`.
2. `run_inference` in `crates/aprender-serve/src/infer/inference_result.rs` verifies the input, detects APR/GGUF/SafeTensors, and handles sharded SafeTensors as a distinct path.
3. `prepare_tokens` in `crates/aprender-serve/src/infer/mod.rs` loads tokenizer/chat-template state and produces a `PreparedTokens` request before format-specific execution.
4. APR execution memory-maps the container through the APR implementation assembled in `crates/aprender-serve/src/apr/mod.rs`; quantized tensors can remain in their encoded representation through `OwnedQuantizedModel::from_apr` in `crates/aprender-serve/src/gguf/loader_apr_quantized.rs`.
5. CPU generation uses portable operations from `crates/aprender-compute/`; CUDA generation enters feature-gated paths such as `crates/aprender-serve/src/infer/gguf_gpu_generate.rs` and device primitives from `crates/aprender-gpu/`.
6. CLI rendering and output policy return through `crates/apr-cli/src/commands/run_entry.rs`.

**HTTP Serving:**

1. Serve command arguments are dispatched by `crates/apr-cli/src/dispatch_run.rs` into server configuration implemented under `crates/apr-cli/src/commands/serve/`.
2. Model, tokenizer, cache, registry, metrics, audit, and optional device state are held in the shared `AppState` variants in `crates/aprender-serve/src/api/mod.rs`.
3. `crates/aprender-serve/src/api/router.rs` installs health/metrics routes, native inference routes, OpenAI-compatible `/v1` routes, APR-specific predict/explain/audit routes, optional GPU batch routes, and Ollama-compatible `/api` routes.
4. Requests flow through scheduling/cache/inference modules under `crates/aprender-serve/src/scheduler/`, `crates/aprender-serve/src/paged_kv/`, and `crates/aprender-serve/src/infer/`, then return API-specific response types.

**Format Conversion and Validation:**

1. CLI lifecycle operations enter command modules under `crates/apr-cli/src/commands/` after pre-dispatch metadata validation in `crates/apr-cli/src/validate.rs`.
2. `RosettaStone` and format adapters in `crates/aprender-core/src/format/rosetta/` detect source format, normalize metadata/tensor meaning, and enforce layout/model-family requirements.
3. Conversion and import logic in `crates/aprender-core/src/format/converter/` maps names, shapes, dtypes, and quantization policy into canonical APR tensors.
4. `AprV2Writer` or `AprV2StreamingWriter` from `crates/apr-format/src/v2/` emits a sorted, aligned, checksummed container.
5. The lifecycle phases and tensor layout obligations are specified in `contracts/apr-model-lifecycle-v1.yaml` and `contracts/tensor-layout-v1.yaml`.

**Contract Binding and Enforcement:**

1. Contract authors define versioned YAML under `contracts/`, with model architectures grouped under `contracts/model-families/`.
2. `parse_contract` and `validate_contract` in `crates/aprender-contracts/src/schema/` produce and validate typed contract representations.
3. The `#[contract(...)]` macro in `crates/aprender-contracts-macros/src/lib.rs` attaches equations, generated debug assertions, and registration data to Rust functions.
4. `BindingRegistry` in `crates/aprender-contracts/src/binding.rs` connects equations to concrete modules/functions, while `crates/aprender-contracts/src/pipeline.rs` composes stage obligations across system boundaries.
5. Generated Rust modules named `generated_contracts.rs` expose compile-time contract bindings within participating crates such as `crates/aprender-core/src/generated_contracts.rs`.

**State Management:**
- Keep training state explicit and owned by trainer structs in `crates/aprender-train/src/train/trainer/core.rs` and `crates/aprender-train/src/train/transformer_trainer/trainer.rs`; checkpoint persistent state through `crates/aprender-train/src/io/`.
- Keep inference model data immutable and borrow or memory-map it where possible through `crates/apr-format/src/v2/` and `crates/aprender-serve/src/apr/`.
- Share long-lived server state through the synchronized `AppState` fields in `crates/aprender-serve/src/api/mod.rs`, rather than process-global mutable model state.
- Represent CLI state as parsed command/configuration values in `crates/apr-cli/src/lib.rs` and operation-specific configuration types under `crates/apr-cli/src/commands/`.
- Persist lifecycle, registry, and dataset state through explicit storage/artifact crates such as `crates/aprender-registry/`, `crates/aprender-db/`, and `crates/aprender-data/`.

## Key Abstractions

**Estimator and Transformer Traits:**
- Purpose: Define sklearn-style fit/transform/predict/score contracts for high-level algorithms.
- Examples: `crates/aprender-core/src/traits.rs`, `crates/aprender-core/src/prelude.rs`
- Pattern: Implement public behavior through generic traits while model structs retain domain-specific configuration and learned state.

**APR v2 Reader/Writer Family:**
- Purpose: Provide canonical container access independently of high-level model logic.
- Examples: `crates/apr-format/src/v2/reader_impl.rs`, `crates/apr-format/src/v2/writer.rs`, `crates/aprender-core/src/serialization/apr/mod.rs`
- Pattern: Use the low-level `AprV2Reader`, borrowed `AprV2ReaderRef`, buffered/streaming writers, and high-level atomic `AprReader`/`AprWriter` adapters according to ownership and lifecycle needs.

**RosettaStone and Format Extensions:**
- Purpose: Normalize external model formats into common metadata, tensor names, layout, and model-family concepts.
- Examples: `crates/aprender-core/src/format/rosetta/`, `crates/aprender-core/src/format/dequant_ext.rs`, `crates/aprender-core/src/format/converter/`
- Pattern: Keep container parsing in `crates/apr-format/`; implement f32 semantics, conversion, dequantization, and external-format policy in `crates/aprender-core/`.

**Trainer and Pretrain Adapters:**
- Purpose: Separate reusable optimization-loop policy from model/device-specific forward and validation execution.
- Examples: `crates/aprender-train/src/train/trainer/core.rs`, `crates/aprender-train/src/train/pretrain.rs`, `crates/aprender-train/src/train/pretrain_real.rs`, `crates/aprender-train/src/train/pretrain_real_cuda.rs`
- Pattern: Put invariant loop sequencing in generic orchestration; inject `StepFn`, `ValFn`, checkpoint, model, optimizer, and device implementations through explicit traits and owned collaborators.

**InferenceConfig and PreparedTokens:**
- Purpose: Make inference request preparation a shared, validated boundary before format/device dispatch.
- Examples: `crates/aprender-serve/src/infer/mod.rs`, `crates/aprender-serve/src/infer/inference_result.rs`
- Pattern: Resolve tokenizer and chat-template behavior once, then pass typed prepared input to APR/GGUF/SafeTensors and CPU/GPU executors.

**Portable Backend and Device Backend:**
- Purpose: Distinguish runtime SIMD operation selection from low-level accelerator ownership and capabilities.
- Examples: `crates/aprender-compute/src/backends/mod.rs`, `crates/aprender-compute/src/vector/dispatch.rs`, `crates/aprender-gpu/src/backend/mod.rs`, `crates/aprender-gpu/src/driver/mod.rs`
- Pattern: Add safe numerical operations to `aprender-compute`; add accelerator FFI, buffers, streams, modules, and kernels to `aprender-gpu`; expose capability checks rather than leaking raw device handles upward.

**ComputeGraph and Compute Bricks:**
- Purpose: Represent reusable operation sequences and reduce repeated kernel/operation dispatch.
- Examples: `crates/aprender-gpu/src/graph/mod.rs`, `crates/aprender-compute/src/brick/`
- Pattern: Build typed nodes/operations at the compute boundary, then let serving or training compose them without embedding hardware-specific launch details.

**CLI Command and Error Boundary:**
- Purpose: Decouple user-facing syntax and stable exit behavior from domain implementation.
- Examples: `crates/apr-cli/src/commands_enum.rs`, `crates/apr-cli/src/dispatch.rs`, `crates/apr-cli/src/dispatch_analysis.rs`, `crates/apr-cli/src/error.rs`
- Pattern: Define arguments in command fragments, dispatch centrally, implement behavior in `crates/apr-cli/src/commands/`, and return `CliError` rather than terminating inside domain code.

**Executable Contracts:**
- Purpose: Bind mathematical and lifecycle obligations to Rust functions and multi-stage pipelines.
- Examples: `crates/aprender-contracts/src/binding.rs`, `crates/aprender-contracts/src/pipeline.rs`, `crates/aprender-contracts-macros/src/lib.rs`, `contracts/tensor-layout-v1.yaml`
- Pattern: Version contract files, bind equations to concrete functions, generate crate-local bindings, and validate boundary invariants before expensive execution.

**Compatibility Crate Names:**
- Purpose: Preserve concise domain imports while package directories use the `aprender-*` namespace.
- Examples: `crates/aprender-core/Cargo.toml` exposes `aprender`, `crates/aprender-compute/Cargo.toml` exposes `trueno`, `crates/aprender-serve/Cargo.toml` exposes `realizar`, `crates/aprender-train/Cargo.toml` exposes `entrenar`, `crates/aprender-registry/Cargo.toml` exposes `pacha`, and `crates/aprender-data/Cargo.toml` exposes `alimentar`.
- Pattern: Use the manifest-declared library name in Rust imports; use the package/directory name in Cargo workspace and file-path discussions.

## Entry Points

**Public Library Facade:**
- Location: `src/lib.rs`
- Triggers: A downstream crate imports the root `aprender` package.
- Responsibilities: Re-export the high-level API from the `aprender-core` package/library and keep the root crate free of duplicated ML implementation.

**Installed APR Binary:**
- Location: `src/bin/apr.rs`
- Triggers: A user invokes the `apr` binary installed from the root package.
- Responsibilities: Delegate process execution to `apr_cli::cli_main()` from `crates/apr-cli/src/lib.rs`.

**APR CLI Crate Binary:**
- Location: `crates/apr-cli/src/main.rs`
- Triggers: The `apr-cli` package binary is built or executed directly.
- Responsibilities: Perform process setup and enter the shared command execution path implemented by the `apr_cli` library.

**CLI Execution Gateway:**
- Location: `crates/apr-cli/src/dispatch_run.rs`
- Triggers: Parsed `Cli` input from `crates/apr-cli/src/lib.rs` or `crates/apr-cli/src/main.rs`.
- Responsibilities: Apply pre-dispatch model validation, select core versus extended dispatch, and return typed CLI failures.

**Core and Extended Dispatch:**
- Location: `crates/apr-cli/src/dispatch.rs`, `crates/apr-cli/src/dispatch_analysis.rs`
- Triggers: The execution gateway receives a parsed command.
- Responsibilities: Route runtime, inspection, diagnostics, format, lifecycle, training, data, orchestration, and analysis commands into implementations under `crates/apr-cli/src/commands/`.

**Inference Gateway:**
- Location: `crates/aprender-serve/src/infer/inference_result.rs`
- Triggers: CLI run logic or another Rust caller invokes `run_inference` with an `InferenceConfig`.
- Responsibilities: Validate the path, detect format, prepare input, and select the APR/GGUF/SafeTensors and CPU/GPU execution path.

**HTTP Router:**
- Location: `crates/aprender-serve/src/api/router.rs`
- Triggers: Server startup constructs an application from `AppState` in `crates/aprender-serve/src/api/mod.rs`.
- Responsibilities: Install health, metrics, native, OpenAI-compatible, APR-specific, GPU-batch, and Ollama-compatible endpoints.

**Generic Training Loop:**
- Location: `crates/aprender-train/src/train/trainer/train_loop/basic.rs`
- Triggers: A configured `Trainer` from `crates/aprender-train/src/train/trainer/core.rs` starts training.
- Responsibilities: Iterate batches/epochs, accumulate and clip gradients, step the optimizer, update metrics, invoke callbacks, and apply stopping policy.

**Contract-Driven Pretraining Loop:**
- Location: `crates/aprender-train/src/train/pretrain.rs`
- Triggers: A CPU or CUDA adapter supplies `StepFn`, `ValFn`, and checkpoint implementations.
- Responsibilities: Sequence training, finite-value gates, validation, divergence checks, checkpointing, metadata/artifact emission, and convergence policy.

**APR Container API:**
- Location: `crates/apr-format/src/v2/mod.rs`
- Triggers: Core conversion/serialization, training save/load, or serving APR load code requires container access.
- Responsibilities: Expose canonical APR header, tensor index, metadata, validation, reader, and writer types.

**External Pipeline Adapter:**
- Location: `crates/apr-cli/src/commands/pipeline.rs`
- Triggers: `apr pipeline plan`, `apply`, `status`, or `validate` is dispatched from `crates/apr-cli/src/dispatch_analysis.rs`.
- Responsibilities: Translate APR pipeline arguments and invoke the external `forjar` executable rather than embedding orchestration internals in the CLI crate.

## Error Handling

**Strategy:** Use typed `Result` boundaries inside crates, validate contracts and capabilities before execution, isolate unsafe/device failures in lower layers, and translate top-level errors into stable CLI exit codes.

**Patterns:**
- Return the core `AprenderError`/`Result` types for public high-level ML operations from `crates/aprender-core/src/error.rs`; do not terminate the process inside algorithms.
- Return container-specific parse, checksum, bounds, and validation errors from `crates/apr-format/src/error.rs` and `crates/apr-format/src/validate.rs`; add model semantics only after successful container validation in `crates/aprender-core/src/format/`.
- Run metadata-only model contract checks before CLI dispatch through `crates/apr-cli/src/validate.rs`, including the distinct sharded SafeTensors manifest path.
- Preserve stable CLI exit categories in `crates/apr-cli/src/error.rs`: invalid input/validation/load/I/O/inference/feature/network failures remain distinguishable to shell callers.
- Reject unavailable explicitly requested CUDA devices in `crates/aprender-train/src/train/device.rs`; only automatic device selection may choose an available fallback.
- Check non-finite loss/metrics and divergence before checkpoint progression in `crates/aprender-train/src/train/pretrain.rs`.
- Encapsulate raw CUDA calls beneath RAII wrappers in `crates/aprender-gpu/src/driver/`; keep `unsafe` device bindings in `crates/aprender-gpu/src/driver/sys/`.
- Use temporary-file synchronization and rename for APR persistence in `crates/aprender-core/src/serialization/apr/mod.rs` to avoid exposing partial model artifacts.

## Cross-Cutting Concerns

**Logging:** Use command-facing diagnostics at the CLI boundary in `crates/apr-cli/src/`, structured serving observability in `crates/aprender-serve/src/observability/`, API metrics in `crates/aprender-serve/src/api/mod.rs`, and device telemetry in `crates/aprender-gpu/src/monitor/`.

**Validation:** Enforce byte/container integrity in `crates/apr-format/src/validate.rs`, model/layout semantics in `crates/aprender-core/src/format/`, command preconditions in `crates/apr-cli/src/validate.rs`, and executable specifications through `crates/aprender-contracts/src/` and `contracts/`.

**Authentication:** Keep request authentication and access policy at the serving/API boundary under `crates/aprender-serve/src/api/` and its server configuration path under `crates/apr-cli/src/commands/serve/`; do not couple authentication to model, compute, or APR container crates.

**Feature Gating:** Add optional CPU/GPU/training/inference behavior behind features declared in root `Cargo.toml`, `crates/apr-cli/Cargo.toml`, and the owning crate manifest; mirror feature gates on command variants and dispatch arms in `crates/apr-cli/src/`.

**Tensor Layout and Quantization:** Preserve canonical row-major APR/SafeTensors semantics and explicit GGUF layout transforms as specified by `contracts/tensor-layout-v1.yaml`; keep container dtypes in `crates/apr-format/src/types.rs`, quantization implementations in `crates/aprender-quant/`, and runtime quantized execution in serving/compute loaders such as `crates/aprender-serve/src/gguf/loader_apr_quantized.rs`.

**Extensibility:** Add a new CLI feature by extending an argument fragment in `crates/apr-cli/src/`, adding a command implementation under `crates/apr-cli/src/commands/`, wiring exactly one central dispatch path in `crates/apr-cli/src/dispatch.rs` or `crates/apr-cli/src/dispatch_analysis.rs`, and adding its domain implementation to the owning crate under `crates/`.

**Crate Boundaries:** Keep new byte-level APR mechanics in `crates/apr-format/`, external-format/model semantics in `crates/aprender-core/src/format/`, training policy in `crates/aprender-train/`, request/runtime inference in `crates/aprender-serve/`, safe portable numerical operations in `crates/aprender-compute/`, and device/FFI/kernel details in `crates/aprender-gpu/`.

---

*Architecture analysis: 2026-08-07*

# Codebase Structure

**Analysis Date:** 2026-08-07

## Directory Layout

```text
aprender/
├── Cargo.toml                 # Workspace membership, shared dependencies, facade package, and feature forwarding
├── Cargo.lock                 # Locked Rust dependency graph
├── rust-toolchain.toml        # Pinned Rust toolchain configuration
├── rustfmt.toml               # Repository Rust formatting policy
├── .clippy.toml               # Repository Clippy configuration
├── src/                       # Thin root library facade and installable apr binary
├── crates/                    # Flat workspace of domain, runtime, tooling, and support crates
│   ├── apr-format/            # Sovereign dependency-light APR v2 container
│   ├── apr-cli/               # Clap definitions, validation, dispatch, and command adapters
│   ├── aprender-core/         # High-level ML algorithms, traits, formats, and serialization
│   ├── aprender-train/        # Autograd, optimization, training loops, evaluation, and checkpoints
│   ├── aprender-serve/        # Inference runtimes, scheduling, and HTTP serving
│   ├── aprender-compute/      # Safe vector/matrix APIs and portable CPU/SIMD compute
│   ├── aprender-gpu/          # Accelerator backends, CUDA driver wrappers, kernels, and graphs
│   ├── aprender-contracts/    # Contract parsing, binding, validation, and composition
│   ├── aprender-data/         # Dataset, dataloader, split, transform, and storage APIs
│   └── aprender-*/            # Additional focused ecosystem and tooling crates
├── contracts/                 # Versioned executable YAML specifications and model-family contracts
├── book/                      # mdBook source and configuration
├── docs/                      # Specifications, examples, QA material, and supporting documentation
├── scripts/                   # Repository automation and validation scripts
├── configs/                   # Checked-in tool and workflow configurations
├── datasets/                  # Repository dataset metadata/assets
├── tools/                     # Standalone development tools outside the main workspace
├── fuzz/                      # Cargo-fuzz targets excluded from the main workspace
├── evidence/                  # Generated or collected verification evidence
├── experiments/               # Experimental programs and research artifacts
├── playbooks/                 # Operational and development procedures
├── .github/workflows/         # CI and release workflow definitions
├── .claude/skills/            # Project-specific pre-release and APR dogfood workflow guidance
├── .planning/                 # GSD planning and codebase analysis artifacts
└── target/                    # Generated Cargo build output
```

## Directory Purposes

**Root Facade (`src/`):**
- Purpose: Keep the published root package thin while exposing the core library and installable CLI.
- Contains: A library re-export and a binary delegate.
- Key files: `src/lib.rs`, `src/bin/apr.rs`

**Workspace Crates (`crates/`):**
- Purpose: Hold independently versioned/package-scoped architectural units in a flat workspace declared by `Cargo.toml`.
- Contains: Core ML, format, CLI, compute, GPU, training, serving, contracts, data, registry, visualization, QA, testing, and tooling crates.
- Key files: `crates/aprender-core/Cargo.toml`, `crates/apr-format/Cargo.toml`, `crates/apr-cli/Cargo.toml`, `crates/aprender-train/Cargo.toml`, `crates/aprender-serve/Cargo.toml`

**Core ML (`crates/aprender-core/`):**
- Purpose: Own the high-level public ML surface and algorithms rather than command, server, or device concerns.
- Contains: Estimator traits, classical algorithms, neural networks, preprocessing, model selection, external format semantics, model loading, and serialization adapters under `crates/aprender-core/src/`.
- Key files: `crates/aprender-core/src/lib.rs`, `crates/aprender-core/src/traits.rs`, `crates/aprender-core/src/prelude.rs`, `crates/aprender-core/src/error.rs`

**APR Container (`crates/apr-format/`):**
- Purpose: Own the dependency-light APR container schema and byte-level mechanics as a workspace leaf.
- Contains: Headers, typed metadata, checksums, validation, readers, writers, and streaming I/O under `crates/apr-format/src/v2/`.
- Key files: `crates/apr-format/src/lib.rs`, `crates/apr-format/src/types.rs`, `crates/apr-format/src/validate.rs`, `crates/apr-format/src/v2/mod.rs`

**Core Format Semantics (`crates/aprender-core/src/format/`):**
- Purpose: Interpret model meaning above the APR container and bridge APR/GGUF/SafeTensors/ONNX.
- Contains: Conversion, dequantization, lint, sharding, model families, validated tensors, and Rosetta format adapters.
- Key files: `crates/aprender-core/src/format/mod.rs`, `crates/aprender-core/src/format/converter/mod.rs`, `crates/aprender-core/src/format/rosetta/`, `crates/aprender-core/src/format/dequant_ext.rs`

**APR Convenience Serialization (`crates/aprender-core/src/serialization/apr/`):**
- Purpose: Wrap canonical APR v2 I/O with model-oriented metadata, tensor filtering/dequantization, and atomic file replacement.
- Contains: `AprReader`, `AprWriter`, tensor descriptors, and training-state filtering.
- Key files: `crates/aprender-core/src/serialization/apr/mod.rs`

**CLI (`crates/apr-cli/`):**
- Purpose: Define user-facing syntax, enforce command-level contracts, dispatch actions, and adapt domain results to terminal output/exit codes.
- Contains: Command enum fragments in `crates/apr-cli/src/`, implementations in `crates/apr-cli/src/commands/`, and crate-level integration tests in `crates/apr-cli/tests/`.
- Key files: `crates/apr-cli/src/lib.rs`, `crates/apr-cli/src/dispatch_run.rs`, `crates/apr-cli/src/dispatch.rs`, `crates/apr-cli/src/dispatch_analysis.rs`, `crates/apr-cli/src/validate.rs`, `crates/apr-cli/src/error.rs`

**CLI Command Implementations (`crates/apr-cli/src/commands/`):**
- Purpose: Implement one user action or closely related command family per module after shared parsing/validation.
- Contains: Run, serve, lifecycle, inspection, conversion, training, data, registry, profiling, QA, and diagnostics handlers.
- Key files: `crates/apr-cli/src/commands/mod.rs`, `crates/apr-cli/src/commands/run.rs`, `crates/apr-cli/src/commands/inference_output.rs`, `crates/apr-cli/src/commands/serve/`, `crates/apr-cli/src/commands/pipeline.rs`

**Training (`crates/aprender-train/`):**
- Purpose: Own train-time state and policy independently of request-time inference.
- Contains: Autograd in `crates/aprender-train/src/autograd/`, optimizers in `crates/aprender-train/src/optim/`, loops in `crates/aprender-train/src/train/`, evaluation in `crates/aprender-train/src/eval/`, and I/O in `crates/aprender-train/src/io/`.
- Key files: `crates/aprender-train/src/lib.rs`, `crates/aprender-train/src/train/mod.rs`, `crates/aprender-train/src/train/pretrain.rs`, `crates/aprender-train/src/train/device.rs`

**Training Family Crates (`crates/aprender-train-*/`):**
- Purpose: Keep specialized training applications and frontends outside the main training engine.
- Contains: Common support, LoRA, distillation, inspection, shell, benchmark, and WASM packages.
- Key files: `crates/aprender-train-common/Cargo.toml`, `crates/aprender-train-lora/Cargo.toml`, `crates/aprender-train-distill/Cargo.toml`, `crates/aprender-train-wasm/Cargo.toml`

**Inference and Serving (`crates/aprender-serve/`):**
- Purpose: Own model-format runtime loading, token preparation, generation, scheduling, caching, device routing, and HTTP APIs.
- Contains: Shared inference in `crates/aprender-serve/src/infer/`, model-specific loaders under `crates/aprender-serve/src/apr/`, `crates/aprender-serve/src/gguf/`, and `crates/aprender-serve/src/safetensors/`, plus routes under `crates/aprender-serve/src/api/`.
- Key files: `crates/aprender-serve/src/lib.rs`, `crates/aprender-serve/src/infer/mod.rs`, `crates/aprender-serve/src/infer/inference_result.rs`, `crates/aprender-serve/src/api/mod.rs`, `crates/aprender-serve/src/api/router.rs`

**Portable Compute (`crates/aprender-compute/`):**
- Purpose: Provide safe hardware-neutral numerical APIs and runtime-selected CPU/SIMD implementations.
- Contains: Vectors in `crates/aprender-compute/src/vector/`, matrices in `crates/aprender-compute/src/matrix/`, compute bricks in `crates/aprender-compute/src/brick/`, and isolated backends in `crates/aprender-compute/src/backends/`.
- Key files: `crates/aprender-compute/src/lib.rs`, `crates/aprender-compute/src/backends/mod.rs`, `crates/aprender-compute/src/vector/dispatch.rs`

**GPU (`crates/aprender-gpu/`):**
- Purpose: Contain accelerator capabilities and unsafe/native boundaries below safe training and inference APIs.
- Contains: Backend discovery in `crates/aprender-gpu/src/backend/`, CUDA wrappers in `crates/aprender-gpu/src/driver/`, raw FFI in `crates/aprender-gpu/src/driver/sys/`, kernels in `crates/aprender-gpu/src/kernels/`, and graphs in `crates/aprender-gpu/src/graph/`.
- Key files: `crates/aprender-gpu/src/lib.rs`, `crates/aprender-gpu/src/backend/mod.rs`, `crates/aprender-gpu/src/driver/mod.rs`, `crates/aprender-gpu/src/graph/mod.rs`

**Quantization (`crates/aprender-quant/`):**
- Purpose: Centralize quantization formats and algorithms reusable by training, conversion, compute, and serving.
- Contains: Quantized data representations and kernels under `crates/aprender-quant/src/`.
- Key files: `crates/aprender-quant/Cargo.toml`, `crates/aprender-quant/src/lib.rs`

**Contracts (`contracts/`):**
- Purpose: Store checked-in, versioned executable specifications used across code generation, validation, and audit paths.
- Contains: Top-level lifecycle/tensor/kernel/CLI contracts and model architecture definitions under `contracts/model-families/`.
- Key files: `contracts/apr-model-lifecycle-v1.yaml`, `contracts/tensor-layout-v1.yaml`, `contracts/model-families/`

**Contract Runtime (`crates/aprender-contracts/`):**
- Purpose: Parse and validate contract documents, bind equations to code, and compose multi-stage obligations.
- Contains: Schema handling in `crates/aprender-contracts/src/schema/`, bindings in `crates/aprender-contracts/src/binding.rs`, and composition in `crates/aprender-contracts/src/pipeline.rs`.
- Key files: `crates/aprender-contracts/src/lib.rs`, `crates/aprender-contracts/src/binding.rs`, `crates/aprender-contracts/src/pipeline.rs`

**Contract Macros and CLI (`crates/aprender-contracts-macros/`, `crates/aprender-contracts-cli/`):**
- Purpose: Attach contracts to Rust code and expose contract tooling without coupling those concerns to core algorithms.
- Contains: Procedural macros and command-line tooling.
- Key files: `crates/aprender-contracts-macros/src/lib.rs`, `crates/aprender-contracts-cli/src/main.rs`

**Data and Registry (`crates/aprender-data/`, `crates/aprender-registry/`):**
- Purpose: Separate dataset iteration/transformation and artifact registry behavior from model algorithms and terminal commands.
- Contains: Datasets, dataloaders, splits, backends, transforms, registry records, and artifact lookup.
- Key files: `crates/aprender-data/src/dataset.rs`, `crates/aprender-data/src/dataloader.rs`, `crates/aprender-data/src/split.rs`, `crates/aprender-registry/src/lib.rs`

**Documentation (`book/`, `docs/`):**
- Purpose: Hold user/developer narrative documentation separately from executable contracts.
- Contains: mdBook content under `book/src/`, specifications under `docs/specifications/`, examples under `docs/examples/`, and QA/roadmap material under `docs/`.
- Key files: `book/book.toml`, `book/src/SUMMARY.md`, `docs/specifications/`

**Automation (`scripts/`, `.github/workflows/`, `Makefile`):**
- Purpose: Run local checks, generation, CI, release, and repository maintenance around the Cargo workspace.
- Contains: Shell/Python automation, GitHub Actions workflow definitions, and Make targets.
- Key files: `Makefile`, `scripts/`, `.github/workflows/`

**Standalone/Excluded Development Code (`tools/`, `fuzz/`):**
- Purpose: Keep special-purpose exporters and fuzz harnesses isolated from default workspace builds.
- Contains: Standalone Cargo packages such as `tools/ccpa-sft-export/` and cargo-fuzz targets in `fuzz/`.
- Key files: `tools/ccpa-sft-export/Cargo.toml`, `fuzz/Cargo.toml`, workspace `exclude` entries in `Cargo.toml`

## Key File Locations

**Entry Points:**
- `src/lib.rs`: Published root library facade over `aprender-core`.
- `src/bin/apr.rs`: Root-package `apr` binary delegate.
- `crates/apr-cli/src/lib.rs`: Shared Clap application and `cli_main` entry.
- `crates/apr-cli/src/main.rs`: Direct `apr-cli` package binary.
- `crates/aprender-serve/src/infer/inference_result.rs`: Programmatic inference gateway.
- `crates/aprender-serve/src/api/router.rs`: HTTP route composition.
- `crates/aprender-train/src/train/trainer/train_loop/basic.rs`: Generic training execution loop.
- `crates/aprender-train/src/train/pretrain.rs`: Contract-driven pretraining entry abstraction.

**Configuration:**
- `Cargo.toml`: Workspace membership, exclusions, package metadata, shared dependencies, facade features, and binary/library targets.
- `Cargo.lock`: Reproducible dependency resolution for the workspace.
- `rust-toolchain.toml`: Rust toolchain selection.
- `rustfmt.toml`: Rust formatter policy.
- `.clippy.toml`: Clippy policy shared by workspace crates.
- `crates/apr-cli/Cargo.toml`: CLI feature and dependency integration matrix.
- `crates/aprender-serve/Cargo.toml`: Serving/runtime features and dependencies.
- `crates/aprender-train/Cargo.toml`: Training features, library name, and dependencies.

**Core Logic:**
- `crates/aprender-core/src/traits.rs`: Public estimator and transformer contracts.
- `crates/aprender-core/src/lib.rs`: High-level ML module surface.
- `crates/apr-format/src/v2/mod.rs`: Canonical APR v2 container API.
- `crates/aprender-core/src/format/`: External-format/model semantic layer.
- `crates/aprender-core/src/serialization/apr/mod.rs`: Model-oriented APR convenience API.
- `crates/aprender-train/src/train/`: Generic and transformer training orchestration.
- `crates/aprender-train/src/io/`: Training model load/save boundary.
- `crates/aprender-serve/src/infer/`: Request-time inference flow.
- `crates/aprender-serve/src/api/`: Serving state and protocol routes.
- `crates/aprender-compute/src/backends/`: Portable compute implementations.
- `crates/aprender-gpu/src/driver/`: Safe device ownership over low-level CUDA FFI.
- `crates/aprender-contracts/src/`: Executable contract infrastructure.

**Testing:**
- `crates/<crate>/tests/`: Black-box or cross-module integration tests for an individual package.
- `crates/<crate>/src/**/*_tests.rs`: Source-adjacent unit-test shards used by large modules.
- `crates/apr-cli/tests/`: CLI integration and behavior tests.
- `crates/aprender-serve/src/testing/`: Serving test support and runtime test utilities.
- `crates/aprender-gpu/src/testing/`: GPU integration/stress support behind device-aware gates.
- `fuzz/`: Cargo-fuzz targets outside the default workspace.

**Contracts:**
- `contracts/`: Versioned YAML contract source of truth.
- `contracts/model-families/`: Model architecture contracts.
- `crates/*/src/generated_contracts.rs`: Generated crate-local contract bindings.
- `crates/aprender-contracts-macros/src/lib.rs`: `#[contract]` procedural macro.

## Naming Conventions

**Files:**
- Use Rust `snake_case.rs` for modules, as in `crates/aprender-train/src/train/pretrain_real_cuda.rs` and `crates/aprender-serve/src/infer/inference_result.rs`.
- Use `lib.rs` for library surfaces and `main.rs` or `src/bin/<name>.rs` for binaries, as in `crates/aprender-core/src/lib.rs`, `crates/apr-cli/src/main.rs`, and `src/bin/apr.rs`.
- Use `mod.rs` to assemble directory modules and re-export their public surface, as in `crates/aprender-core/src/format/mod.rs` and `crates/aprender-train/src/train/mod.rs`.
- Split very large implementation modules into focused suffix/numbered fragments when the owning module already follows that pattern, as in `crates/apr-cli/src/lib_07.rs` and included fragments under `crates/aprender-serve/src/apr/`.
- Name source-adjacent test shards `*_tests.rs` or place `tests` submodules beside the implementation, as in `crates/aprender-core/src/format/v2_dequant_tests/` and `crates/aprender-compute/src/vector/tests/`.
- Name generated contract modules `generated_contracts.rs`, as in `crates/aprender-core/src/generated_contracts.rs` and `crates/aprender-serve/src/generated_contracts.rs`.
- Name versioned contracts in kebab case with a version suffix, as in `contracts/apr-model-lifecycle-v1.yaml` and `contracts/tensor-layout-v1.yaml`.

**Directories:**
- Name Cargo package directories in kebab case and use the `aprender-*` namespace for ecosystem crates, as in `crates/aprender-core/`, `crates/aprender-train/`, and `crates/aprender-serve/`.
- Keep specialized families adjacent through a shared prefix, as in `crates/aprender-train-lora/`, `crates/aprender-train-distill/`, and `crates/aprender-train-wasm/`.
- Match a module directory to its `snake_case` Rust module name, as in `crates/aprender-serve/src/paged_kv/` and `crates/aprender-train/src/transformer/`.
- Use plural domain directories for collections of implementations, as in `crates/aprender-gpu/src/kernels/`, `crates/aprender-core/src/models/`, and `crates/aprender-data/src/datasets/`.
- Use the manifest library alias in Rust code even when it differs from the package directory: `aprender` for `crates/aprender-core/`, `trueno` for `crates/aprender-compute/`, `realizar` for `crates/aprender-serve/`, `entrenar` for `crates/aprender-train/`, `pacha` for `crates/aprender-registry/`, and `alimentar` for `crates/aprender-data/`.

## Where to Add New Code

**New Classical ML Feature:**
- Primary code: Add the algorithm to a matching domain module under `crates/aprender-core/src/`, export it from that module and `crates/aprender-core/src/lib.rs`, and use the traits in `crates/aprender-core/src/traits.rs` where applicable.
- Tests: Co-locate unit tests under the owning module in `crates/aprender-core/src/<domain>/` and add public-behavior integration tests under `crates/aprender-core/tests/` when the feature crosses module boundaries.

**New CLI Command:**
- Primary code: Define arguments in the appropriate command-enum fragment under `crates/apr-cli/src/`, implement the handler under `crates/apr-cli/src/commands/`, expose it from `crates/apr-cli/src/commands/mod.rs`, and wire it through `crates/apr-cli/src/dispatch.rs` or `crates/apr-cli/src/dispatch_analysis.rs`.
- Tests: Add parser/dispatch integration tests under `crates/apr-cli/tests/` and handler-focused tests beside the command module in `crates/apr-cli/src/commands/`.

**New APR Container Capability:**
- Primary code: Add byte-level, metadata-schema, checksum, alignment, or reader/writer mechanics under `crates/apr-format/src/`; expose them through `crates/apr-format/src/lib.rs` or `crates/apr-format/src/v2/mod.rs`.
- Tests: Add container round-trip and malformed-input tests under `crates/apr-format/tests/` or beside `crates/apr-format/src/v2/`.

**New Format or Model-Family Adapter:**
- Primary code: Add parsing/conversion/model semantics under `crates/aprender-core/src/format/`, place universal-format behavior under `crates/aprender-core/src/format/rosetta/`, and keep APR byte parsing delegated to `crates/apr-format/`.
- Tests: Add conversion fixtures/tests under the closest directory in `crates/aprender-core/src/format/tests/` or `crates/aprender-core/tests/`.

**New Training Strategy:**
- Primary code: Put reusable loop policy under `crates/aprender-train/src/train/`, optimizer logic under `crates/aprender-train/src/optim/`, evaluation under `crates/aprender-train/src/eval/`, and specialized standalone packaging in a focused `crates/aprender-train-*/` crate only when it has a distinct dependency/runtime boundary.
- Tests: Co-locate deterministic unit tests under the owning `crates/aprender-train/src/` module and add end-to-end training tests under `crates/aprender-train/tests/`.

**New Inference Model or Runtime Path:**
- Primary code: Add format/model loading under `crates/aprender-serve/src/<format-or-family>/`, shared request preparation under `crates/aprender-serve/src/infer/`, and protocol-neutral generation under the closest runtime module in `crates/aprender-serve/src/`.
- Tests: Add loader/generation tests beside the module and API-level cases under `crates/aprender-serve/src/api/tests/` or `crates/aprender-serve/tests/`.

**New HTTP Endpoint:**
- Primary code: Define request/response and handler logic under `crates/aprender-serve/src/api/`, add required shared fields to `crates/aprender-serve/src/api/mod.rs`, and register the route in `crates/aprender-serve/src/api/router.rs`.
- Tests: Add router/handler tests under `crates/aprender-serve/src/api/tests/`.

**New Portable Compute Operation:**
- Primary code: Add the safe operation surface under `crates/aprender-compute/src/vector/`, `crates/aprender-compute/src/matrix/`, or `crates/aprender-compute/src/brick/`; add scalar and architecture-specific implementations under `crates/aprender-compute/src/backends/`; update runtime routing in `crates/aprender-compute/src/vector/dispatch.rs` when needed.
- Tests: Add parity tests under the corresponding `crates/aprender-compute/src/**/tests/` module and compare optimized backends with the scalar reference.

**New GPU Kernel or Device Capability:**
- Primary code: Add backend capability reporting under `crates/aprender-gpu/src/backend/`, safe ownership/wrappers under `crates/aprender-gpu/src/driver/`, raw CUDA bindings only under `crates/aprender-gpu/src/driver/sys/`, and kernels under `crates/aprender-gpu/src/kernels/`.
- Tests: Add device-gated parity tests under `crates/aprender-gpu/src/testing/` or the owning kernel's `tests` module; retain a CPU/reference comparison when possible.

**New Dataset or Data Backend:**
- Primary code: Add dataset abstractions or implementations under `crates/aprender-data/src/datasets/`, storage integrations under `crates/aprender-data/src/backend/`, and transforms under the matching module in `crates/aprender-data/src/`.
- Tests: Add data-contract and iterator tests under `crates/aprender-data/tests/` or beside the owning source module.

**New Contract:**
- Primary code: Add a versioned `contracts/<scope>-v1.yaml`, add parser/runtime behavior to `crates/aprender-contracts/src/` only when the schema requires it, and use `crates/aprender-contracts-macros/src/lib.rs` for function bindings.
- Tests: Validate parsing/binding in `crates/aprender-contracts/tests/` and regenerate the participating `crates/*/src/generated_contracts.rs` files through repository automation in `scripts/`.

**Utilities:**
- Shared helpers: Put helpers in the lowest owning crate under `crates/<owner>/src/`; create a separate crate under `crates/` only for a stable cross-domain API with an independent dependency boundary. Avoid placing implementation in root `src/`, which remains a facade.

## Special Directories

**`contracts/`:**
- Purpose: Executable, versioned source specifications for architectural and behavioral invariants.
- Generated: No
- Committed: Yes

**`crates/*/src/generated_contracts.rs`:**
- Purpose: Generated Rust bindings from contract definitions for crate-local compilation and enforcement.
- Generated: Yes
- Committed: Yes

**`target/`:**
- Purpose: Cargo compilation output, incremental state, binaries, and build-script artifacts.
- Generated: Yes
- Committed: No

**`fuzz/`:**
- Purpose: Cargo-fuzz harnesses built independently from the default workspace.
- Generated: No
- Committed: Yes

**`tools/`:**
- Purpose: Standalone development/export tools that do not participate in normal workspace builds.
- Generated: No
- Committed: Yes

**`.planning/`:**
- Purpose: GSD project planning, phase state, and codebase maps such as `.planning/codebase/ARCHITECTURE.md` and `.planning/codebase/STRUCTURE.md`.
- Generated: Yes
- Committed: Yes

**`.claude/skills/`:**
- Purpose: Project-specific operating constraints for pre-release validation and APR ecosystem dogfooding.
- Generated: No
- Committed: Yes

**`evidence/`:**
- Purpose: Verification artifacts produced or collected by repository quality workflows.
- Generated: Yes
- Committed: Yes

---

*Structure analysis: 2026-08-07*

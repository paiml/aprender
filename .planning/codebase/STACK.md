# Technology Stack

**Analysis Date:** 2026-08-07

## Languages

**Primary:**
- Rust 2021 edition - All production libraries, CLI binaries, inference servers, data pipelines, GPU dispatch, orchestration, registry, and MCP tooling live under `crates/`; the workspace and shared package metadata are defined in `Cargo.toml`.
- Rust toolchain 1.93.0 - The repository pins the compiler, formatter, and Clippy components in `rust-toolchain.toml`; the flagship `aprender` package declares MSRV 1.91 in `Cargo.toml`.

**Secondary:**
- YAML - Provable contracts, QA specifications, workflow configuration, and model/data aliases are stored in `contracts/`, `.github/workflows/`, and `configs/aliases.yaml`.
- Shell - Release, QA, benchmark, and environment automation is implemented in `scripts/` and coordinated by `Makefile`.
- Python - Reference implementations, interoperability checks, and benchmark helpers are kept outside the production runtime, including scripts under `scripts/` and examples under `examples/`.
- WGSL and CUDA/PTX - Optional GPU kernels and shaders are embedded in compute crates such as `crates/aprender-wgpu/`, `crates/aprender-cuda/`, and related GPU modules under `crates/`.
- JavaScript/TypeScript and WebAssembly bindings - Browser-facing support and examples accompany Rust WASM modules such as `crates/aprender-wasm/`; Rust remains the implementation language.

## Runtime

**Environment:**
- Native Rust binaries and libraries are the primary runtime; `src/bin/apr.rs` and `crates/apr-cli/` provide the installed `apr` CLI from the root `aprender` package in `Cargo.toml`.
- Tokio 1.52.3 is the resolved asynchronous runtime for HTTP servers, remote clients, orchestration, and distributed execution; shared dependency policy is declared in `Cargo.toml`.
- Axum 0.8.9 and 0.7.9 are both resolved in `Cargo.lock`: the workspace baseline is Axum 0.8 in `Cargo.toml`, while serving crates including `crates/aprender-serve/Cargo.toml` and `crates/aprender-orchestrate/Cargo.toml` retain Axum 0.7 dependencies.
- WebAssembly is an optional runtime through `wasm-bindgen`, `web-sys`, and WASM-specific crates such as `crates/aprender-wasm/Cargo.toml`.
- GPU execution is optional and selected by feature: WGPU targets Vulkan, Metal, DirectX 12, and WebGPU, while CUDA targets NVIDIA devices through crates such as `crates/aprender-wgpu/Cargo.toml` and `crates/aprender-cuda/Cargo.toml`.

**Package Manager:**
- Cargo 1.93.0 - Workspace membership, shared dependency policy, features, release profiles, and lint baselines are centralized in `Cargo.toml`.
- Lockfile: present at `Cargo.lock`; use locked resolution for reproducible application builds and CI.
- Workspace version: 0.63.0 across the root package and in-tree crates in `Cargo.toml`.
- Crates intended for publication use both an explicit `version = "0.63.0"` and an in-workspace `path` for internal dependencies, as demonstrated throughout crate manifests such as `crates/apr-cli/Cargo.toml` and required by `.claude/skills/pre-release/SKILL.md`.

## Frameworks

**Core:**
- Aprender 0.63.0 - Root facade and feature coordinator in `Cargo.toml`; it exposes classical ML, neural networks, model serialization, inference, training, and the `apr` command surface.
- `aprender-core` - Core ML algorithms, serialization, model-hub support, compression, encryption/signing options, and parallel execution are configured in `crates/aprender-core/Cargo.toml`.
- `apr-format` - Leaf crate for the sovereign `.apr` model container format in `crates/apr-format/Cargo.toml`; preserve this boundary when adding format-level functionality.
- `apr-cli` - Clap-based command framework in `crates/apr-cli/Cargo.toml`; its default feature set includes Hugging Face access, inference, training, visualization, and zram support.
- `aprender-serve` - Pure-Rust GGUF, SafeTensors, and `.apr` inference with Axum HTTP APIs in `crates/aprender-serve/Cargo.toml`; it deliberately avoids Candle and llama.cpp bindings.
- `aprender-train` - Training and evaluation subsystem in `crates/aprender-train/Cargo.toml`, integrated into the CLI through `crates/apr-cli/Cargo.toml`.
- `aprender-data` - Arrow/Parquet data ingestion and local, memory-mapped, provenance-aware data backends in `crates/aprender-data/Cargo.toml`; HTTP, Hugging Face, S3, WASM, encryption, and signing are feature-gated.
- `aprender-orchestrate` - Agent, RAG, registry, MCP, serving, and remote-provider orchestration in `crates/aprender-orchestrate/Cargo.toml`.
- `aprender-mcp` - JSON-RPC 2.0 Model Context Protocol server over standard I/O in `crates/aprender-mcp/Cargo.toml` and `crates/aprender-mcp/src/server.rs`.

**Testing:**
- Cargo test - Baseline unit and integration runner; commands and repository-wide targets are defined in `Makefile`.
- cargo-nextest - Parallel test execution configured by `.config/nextest.toml` and invoked from `Makefile`.
- Proptest 1.6 and QuickCheck - Property-based testing dependencies and case counts are configured in `Cargo.toml`, `.proptest.toml`, and `Makefile`.
- Criterion 0.7 - Microbenchmark framework declared in `Cargo.toml` and used by crate benchmark targets under `crates/*/benches/`.
- cargo-llvm-cov and Codecov - Coverage generation and publication are configured in `Makefile`, `codecov.yml`, and `.github/workflows/coverage.yml`.
- cargo-fuzz - Nightly fuzzing is coordinated from `Makefile` and fuzz targets under `fuzz/`.
- Provable contracts - Contract-first acceptance gates use YAML specifications under `contracts/`, repository settings in `.pv.toml`, and the `pv` commands documented by `.claude/skills/pre-release/SKILL.md`.

**Build/Dev:**
- Cargo - Debug, release, cross-target, feature-gated, and package builds are coordinated through `Cargo.toml` and `Makefile`.
- rustfmt and Clippy - The pinned toolchain in `rust-toolchain.toml`, formatting policy in `rustfmt.toml`, and workspace lint policy in `Cargo.toml` define the required source checks.
- cargo-deny and cargo-audit - Dependency license, source, duplicate, vulnerability, and advisory policy is stored in `deny.toml` and `.cargo/audit.toml`.
- mdBook - Project documentation is built from `book.toml` and source content under `book/` through targets in `Makefile`.
- Docker - CPU and GPU serving images are defined in `crates/aprender-serve/Dockerfile` and `crates/aprender-serve/Dockerfile.gpu`; orchestration images are defined in `crates/aprender-orchestrate/Dockerfile`.
- GitHub Actions - Reusable sovereign CI, clean-room tests, coverage, CUDA, documentation, and binary releases are configured under `.github/workflows/`.

## Key Dependencies

**Critical:**
- Serde 1.0.228, serde_json 1.0.150, bincode 1.3, rmp-serde 1.3, and TOML 0.8 - Model, configuration, API, and registry serialization policy originates in `Cargo.toml` and resolves through `Cargo.lock`.
- Clap 4.6.1 - CLI parsing for `apr` and supporting binaries, declared centrally in `Cargo.toml` and consumed by `crates/apr-cli/Cargo.toml`.
- Tokio 1.52.3, Axum 0.7/0.8, Tower 0.4/0.5, and tower-http - Async runtime and HTTP middleware used by `crates/aprender-serve/`, `crates/aprender-orchestrate/`, `crates/aprender-db/`, and CLI serve commands.
- Reqwest 0.12 and Ureq 2.12 - Async and synchronous HTTP clients declared in `Cargo.toml`; older/newer transitive client versions also coexist in `Cargo.lock`.
- Arrow 57 and Parquet 57 - Columnar data interchange and storage used by `crates/aprender-data/Cargo.toml`, `crates/apr-cli/Cargo.toml`, and distributed/database crates; additional Arrow/Parquet major versions occur transitively in `Cargo.lock`.
- SafeTensors 0.4.5 and custom GGUF/APR loaders - Interoperable model ingestion is configured by `crates/aprender-core/Cargo.toml`, `crates/aprender-serve/Cargo.toml`, and `crates/apr-format/Cargo.toml`.
- Rayon - CPU data and inference parallelism, declared in `Cargo.toml` and used across compute, data, training, and serving crates.
- `thiserror` 2 and `anyhow` 1 - Typed library errors and application-level context are centralized in `Cargo.toml`.

**Infrastructure:**
- hf-hub 0.4.3 and hf-xet 1.5.2 - Optional Hugging Face download/upload and large-file Xet transport in `crates/aprender-core/Cargo.toml` and `crates/aprender-core/src/hf_hub/`.
- rusqlite 0.32.1 with bundled SQLite - Embedded registry and RAG persistence in `crates/aprender-registry/Cargo.toml` and `crates/aprender-rag/Cargo.toml`.
- WGPU 27 workspace baseline - Portable GPU compute in `Cargo.toml`; multiple older WGPU versions are still resolved for some workspace/transitive packages in `Cargo.lock`.
- AWS SDK for Rust - Optional S3-compatible storage backend in `crates/aprender-data/Cargo.toml` and `crates/aprender-data/src/backend/s3.rs`.
- OpenTelemetry 0.31 - Optional OTLP trace export in `crates/aprender-profile/Cargo.toml` and `crates/aprender-profile/src/otlp_exporter.rs`.
- PMCP 2.9 resolved - Optional MCP dispatcher and orchestration integration declared by `crates/aprender-mcp/Cargo.toml` and `crates/aprender-orchestrate/Cargo.toml`.
- Wasmtime 43 - Optional WebAssembly execution used by feature-gated runtime components and resolved in `Cargo.lock`.
- BLAKE3, SHA-2, LZ4, and Zstandard - Content addressing, integrity, and compression across registry, format, model, and data crates; dependency declarations are in the relevant `crates/*/Cargo.toml` files.

## Configuration

**Environment:**
- Feature flags in root `Cargo.toml` are the primary compile-time configuration. Use `cuda`, `cuda-batch`, `wgpu`, `inference`, `training`, `training-gpu`, `visualization`, `zram`, `xet`, `whisper`, or `full` only when their platform requirements are available.
- Runtime integrations use environment variables for credentials and endpoints; the exact supported names and wiring are catalogued in `.planning/codebase/INTEGRATIONS.md` and implemented in files such as `crates/aprender-core/src/hf_hub/client_impl.rs`, `crates/apr-cli/src/commands/serve/auth.rs`, and `crates/aprender-profile/src/otlp_exporter.rs`.
- No root `.env` file is present. Keep secrets out of repository configuration and provide them through the process environment or external secret management.
- Model/data aliases are configured in `configs/aliases.yaml`; orchestration accepts a config path through `APR_CONFIG` in `crates/aprender-orchestrate/`.

**Build:**
- `rust-toolchain.toml` pins Rust 1.93.0 and installs rustfmt and Clippy.
- `Cargo.toml` defines the 78-package workspace, shared dependencies, release profiles, features, lint levels, and package metadata.
- `Cargo.lock` is committed and is authoritative for application and CI resolution.
- `rustfmt.toml` sets Rust 2021 formatting, 100-column width, four-space indentation, and Unix newlines.
- `deny.toml`, `.cargo/audit.toml`, and `.clippy.toml` define dependency and static-analysis policy.
- `.config/nextest.toml`, `.proptest.toml`, `.cargo-mutants.toml`, `.pv.toml`, and `codecov.yml` configure test execution, property tests, mutation testing, contract validation, and coverage.
- Preserve crates.io publishability: internal dependencies in publishable crate manifests must retain both explicit versions and workspace paths, following `.claude/skills/pre-release/SKILL.md`.
- Use worktree-safe build metadata and keep acceptance checks contract-first, following `.claude/skills/pre-release/SKILL.md` and `.claude/skills/apr-dogfood/SKILL.md`.

## Platform Requirements

**Development:**
- Rust 1.93.0 with Cargo, rustfmt, and Clippy, as specified by `rust-toolchain.toml`.
- Standard native build tooling is required for the selected target; the default CPU path does not require Python or an external ML runtime.
- Optional NVIDIA paths require a compatible CUDA driver/toolkit and, for profiling, CUPTI; manifests and feature gates live in CUDA-related crates under `crates/`.
- Optional WGPU paths require a supported Vulkan, Metal, DirectX 12, or WebGPU backend, configured through WGPU crates under `crates/`.
- Optional audio and OCR features may require platform ALSA libraries or the Tesseract executable, as declared by feature-specific crates such as `crates/aprender-rag/Cargo.toml`.
- Full repository validation additionally uses cargo-nextest, cargo-deny, cargo-audit, cargo-llvm-cov, cargo-fuzz, `pv`, and mdBook through `Makefile`.

**Production:**
- Native CLI/library deployment is supported from the root `aprender` package in `Cargo.toml`; `cargo install aprender` installs the `apr` command.
- HTTP inference runs on bare metal or in containers using `crates/aprender-serve/`; Docker CPU and CUDA images are provided by `crates/aprender-serve/Dockerfile` and `crates/aprender-serve/Dockerfile.gpu`.
- Runtime target detection covers AWS Lambda, Docker, WASM edge, and bare metal in `crates/aprender-serve/src/target.rs`; availability of an enum or detector does not by itself provision the corresponding infrastructure.
- WASM/browser deployments use crates such as `crates/aprender-wasm/` and require a host that supplies the selected WebAssembly/WebGPU capabilities.
- Release automation builds Linux artifacts and GitHub release assets through `.github/workflows/binary-release.yml`; crates.io publication is coordinated through `Makefile`.

---

*Stack analysis: 2026-08-07*

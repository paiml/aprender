# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.5] - 2025-01-22

### Added
- Interactive TUI viewer with `alimentar view` command
- Crossterm-based terminal driver with raw mode support
- Keyboard navigation (arrows, vim keys, page up/down, home/end)
- Search functionality with `/` key and `--search` flag
- `DatasetAdapter` enum with InMemory and Streaming variants
- Automatic mode selection based on dataset size (100K row threshold)
- Unicode width calculation for proper column alignment
- `tui_viewer.rs` example demonstrating viewer usage
- CLI documentation for view command in book

### Changed
- TUI module refactored for WASM compatibility (55KB binary)
- Search uses case-insensitive linear scan

## [0.2.4] - 2025-01-21

### Changed
- Arrow/Parquet upgraded to v57 (from v54) - fixes comfy-table let_chains compatibility with Rust stable

## [0.2.2] - 2025-11-30

### Added
- 100 Executable Examples specification with Toyota Way QA methodology
- Book chapters for 100 examples (10 sections covering all functionality)
- Fixture generation script (`scripts/generate_fixtures.rs`)
- QA test harness script (`scripts/run_qa_spec.sh`)
- Epic roadmap for 100 examples implementation (`docs/roadmaps/epic-100-examples.yaml`)
- Integration tests for example scenarios (`tests/example_scenarios.rs`)
- 100-point QA checklist with Toyota Production System principles
- Stratified train/test split for MNIST dataset

### Fixed
- MNIST split now uses stratified sampling to ensure all digit classes in both sets
- WASM build test gracefully handles missing target
- Various clippy lints and code quality improvements

### Changed
- Arrow/Parquet upgraded to v54 (from v53)
- Test suite expanded to 1636+ tests
- Coverage maintained at 90.94%

## [0.2.1] - 2025-11-29

### Added
- `DatasetCardValidator` utility methods: `is_valid_task_category()`, `is_valid_license()`, `is_valid_size_category()`, `suggest_task_category()`
- Native HuggingFace Hub upload API with `HfHubUploader`
- Abstract quality profile system for configurable quality scoring
- 100-point weighted quality scoring system with grade calculation
- Function signature extraction in doctest parser
- REPL module with command parsing and tab completion
- Doctest extraction from Python source files

### Fixed
- Handle nullable columns in constant check quality validation
- Clippy lint warnings across codebase
- Duplicate `DatasetCardValidator` struct definitions

### Changed
- Test suite expanded from 571 to 1648 tests
- Test coverage improved to 92.10%

## [0.2.0] - 2025-11-27

### Added
- REPL command interface with session management
- Quality scoring profiles (ML training, doctest corpus)
- Drift detection CLI commands
- Federated learning split strategies

## [0.1.0] - 2025-11-26

### Added
- Core `ArrowDataset` for in-memory datasets backed by Arrow RecordBatches
- `StreamingDataset` for lazy loading large datasets
- `DataLoader` with batching, shuffling, and drop_last support
- Storage backends: Local, Memory, S3, HTTP
- HuggingFace Hub integration for importing datasets
- Dataset transforms: Select, Filter, Sort, Sample, Shuffle, Take, Skip, Rename, Drop, Cast, Normalize, FillNull, Unique, Map, Chain
- Local dataset registry with versioning and metadata
- CLI tool for convert, info, head, schema, import, and registry operations
- WASM support for browser environments
- Data quality checking with null/duplicate/outlier detection
- Drift detection with KS, Chi-square, and PSI tests
- Federated learning splits (local, proportional, stratified)
- Native .alimentar format with compression (zstd, lz4)
- Optional encryption and signing for datasets (feature-gated)
- Built-in datasets: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, Iris
- Arrow 53 and Parquet 53 support
- Zero-copy data access throughout
- Memory-mapped file support
- Feature flags for optional dependencies (s3, http, hf-hub, wasm)
- Comprehensive test suite with 571 tests

[Unreleased]: https://github.com/paiml/alimentar/compare/v0.2.4...HEAD
[0.2.4]: https://github.com/paiml/alimentar/compare/v0.2.2...v0.2.4
[0.2.2]: https://github.com/paiml/alimentar/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/paiml/alimentar/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/paiml/alimentar/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/paiml/alimentar/releases/tag/v0.1.0

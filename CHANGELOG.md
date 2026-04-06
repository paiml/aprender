# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2025-11-23 (v2.0: Data Integration - Phase 1, 2 & 3)

### Phase 3: Tensor Operations (NEW)
- **TensorExecutor API**:
  - High-level wrapper around trueno library for SIMD-accelerated tensor operations
  - Builder pattern for executor creation with backend selection (CPU/GPU)
  - Async methods for all operations to support future distributed execution
- **SIMD-Accelerated Operations**:
  - Element-wise operations: `add()`, `sub()`, `mul()`, `div()`
  - Dot product: `dot()` for vector inner products
  - Scalar operations: `scalar_mul()` for element-wise scalar multiplication
  - All operations leverage trueno's AVX2/AVX-512 SIMD optimizations
  - 2-8x performance improvement vs scalar operations on modern CPUs
- **Tensor Type Alias**:
  - `Tensor` = `trueno::Vector<f32>` for optimal SIMD performance
  - f32 (single precision) chosen for wider SIMD register utilization
  - Convenient API matching common ML frameworks
- **Error Handling**:
  - Added `RepartirError::Tensor` variant for tensor operation errors
  - Automatic conversion from `trueno::TruenoError` via `From` trait
  - Consistent error propagation through async API
- **Comprehensive Testing**:
  - 8 unit tests covering all tensor operations
  - 1 comprehensive example (`tensor_example.rs`)
  - Doc tests for API documentation
  - All tests validate SIMD-accelerated computation correctness

### Phase 2: Locality-Aware Scheduling
- **Affinity-Based Task Assignment**:
  - Scheduler automatically calculates worker affinity based on data dependencies
  - Affinity score: (data items present) / (total data items requested)
  - `submit_with_affinity()` API for explicit worker preferences
  - `calculate_affinity()` internal method for automatic scoring
- **Locality Metrics Tracking**:
  - New `LocalityMetrics` struct tracks scheduling efficiency
  - Metrics: total_tasks, tasks_with_dependencies, tasks_with_locality
  - `hit_rate()` method calculates locality hit rate (0.0 to 1.0)
  - Real-time tracking via `locality_metrics()` accessor
- **Enhanced Scheduler**:
  - Integrated DataLocationTracker for data-aware scheduling
  - `data_tracker()` accessor for direct tracker access
  - `clear()` method resets data locations and metrics
  - 6 new comprehensive tests for locality scheduling

### Phase 1: Foundational Data Tracking

### Added
- **Parquet Checkpoint Storage**: Checkpoints now use Apache Parquet format for efficient columnar storage
  - SNAPPY compression reduces checkpoint size by ~5-10x compared to JSON
  - Backward compatible: supports reading both Parquet (.parquet) and JSON (.json) formats
  - Schema: checkpoint_id, task_id, iteration, timestamp_micros, data (binary)
- **Apache Arrow Integration**: Added arrow (v53.0) and parquet (v53.0) dependencies for checkpoint storage
- **Enhanced Checkpoint Manager**:
  - `write_parquet()` method for efficient checkpoint serialization
  - `read_parquet()` method for checkpoint restoration
  - Automatic format detection in `restore()` and `list_checkpoints()`
  - Cleanup method handles both Parquet and JSON checkpoint files
- **Data-Locality Tracking** (Phase 1 foundation):
  - New `DataLocationTracker` component in scheduler module
  - Tracks which data items (by string key) are present on which workers
  - Methods: `track_data()`, `locate_data()`, `locate_data_batch()`, `remove_data()`, `remove_worker()`
  - Batch queries return worker affinity scores (count of matching data items)
  - Foundation for Phase 2 locality-aware scheduling
  - 9 comprehensive tests covering all tracker operations

### Changed
- Checkpoint feature now includes `arrow` and `parquet` dependencies
- CheckpointManager writes Parquet format by default when `checkpoint` feature is enabled
- Improved checkpoint storage efficiency with columnar format and compression
- Scheduler module now includes HashSet for data location tracking

### Technical Details
- Parquet schema uses timestamp microseconds for precise temporal ordering
- Single-row record batches per checkpoint for optimal small-file performance
- Backward compatibility maintained: existing JSON checkpoints continue to work
- DataLocationTracker uses Arc<RwLock<HashMap>> for thread-safe concurrent access
- Worker IDs use UUIDs (per Iron Lotus Framework) to prevent invalidation on disconnects

## [1.0.0] - 2025-11-22

### Added
- Comprehensive test suite with 105+ tests achieving 88.58% code coverage
  - 92 unit tests
  - 9 integration tests
  - 4 property-based tests
- pmat quality enforcement with pre-commit hooks (TDG score: 94.6/100, Grade A)
- Integration test suite covering end-to-end workflows
- Mock TCP server for remote executor testing
- TLS configuration validation and error handling tests
- CPU executor edge case tests (large output, empty output, signal termination)
- Coverage report files added to .gitignore
- Complete CHANGELOG following Keep a Changelog format

### Changed
- Improved remote executor test coverage from 28.92% to 80.60%
- Improved TLS executor test coverage from 40.46% to 70.70%
- Improved CPU executor test coverage from 93.14% to 94.73%
- Overall project coverage improved from 64.97% to 88.58%

### Fixed
- Pre-commit hook now correctly analyzes only library code (excludes src/bin/)
- SATD check updated to match current pmat output format

## [0.1.0] - 2024-01-01

### Added
- Initial release with v1.1 features complete
- CPU executor with async process execution
- GPU executor with wgpu compute support (skeleton)
- Remote executor with TCP-based task distribution
- TLS support for secure remote connections
- PubSub and PushPull messaging patterns
- Work-stealing task scheduler
- Comprehensive error handling with RepartirError
- Iron Lotus Framework principles (Jidoka, Kaizen, Genchi Genbutsu)
- Certeza testing methodology (Tier 1: sub-second)

### Dependencies
- tokio 1.35+ for async runtime
- wgpu 23.0+ for GPU compute
- rustls 0.23+ for TLS (with aws-lc-rs)
- serde 1.0+ for serialization
- uuid 1.6+ for stable task IDs

[Unreleased]: https://github.com/paiml/repartir/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/paiml/repartir/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/paiml/repartir/releases/tag/v0.1.0

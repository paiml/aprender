# Changelog

All notable changes to Renacer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.0] - 2026-02-26

### Changed
- Updated trueno dependency from 0.15 to 0.16.0
- Updated aprender dependency from 0.26 to 0.27
- Updated trueno-viz dependency from 0.1 to 0.2 (with `default-features = false`)
- Added package exclude patterns to reduce crate size (221 MiB → 5.8 MiB)

## [0.9.4] - 2026-01-07

### Added

#### SIMD-Accelerated Visualization (Sprint 52-55)

**trueno-viz SIMD Integration**:
- Added `trueno-viz` dependency with `monitor` feature for SIMD kernels
- `HistoryBuffer` now uses SIMD-accelerated statistics:
  - `sum()` - AVX2/NEON horizontal reduction (4.1x speedup)
  - `avg()` - Vectorized mean calculation (4.5x speedup)
  - `min()/max()` - Parallel comparison (4.0x speedup)
  - `stats()` - Combined min/max/mean in single SIMD pass
- `sparkline()` uses SIMD min/max for normalization
- Added `normalize_batch()` for SIMD batch normalization
- Zero-allocation hot paths via `as_slice()` direct storage access

**TUI Terminal Fix**:
- Fixed TUI hang caused by `suppress_stdio()` in tracer thread
- Now uses `/dev/tty` directly for terminal output (immune to stdio redirection)

**Examples & Benchmarks**:
- Added `examples/simd_visualization.rs` demonstrating SIMD usage
- Added `benches/visualization_simd.rs` for SIMD vs scalar comparison
- Verified >4x speedup across all data sizes (100-10000 elements)

**Documentation**:
- Updated `book/src/advanced/simd-acceleration.md` with trueno-viz integration
- Added benchmark results and platform support matrix

### Changed

- **trueno-viz**: Updated to v0.1.16
- **Performance**: Visualization statistics now consistently 4-5x faster

## [0.6.5] - 2025-11-27

### Changed

- **aprender**: Updated to v0.11.0 (latest crates.io release)
- **CI**: Added comprehensive GitHub Actions workflow
- **Lint configuration**: Added Cargo.toml lints and .clippy.toml
- **Documentation**: Fixed rustdoc bracket escaping

### Fixed

- CI workflow excludes benchmarks requiring nightly
- Rustdoc broken-intra-doc-links warnings resolved
- Anti-patterns complexity reduced via severity helper methods

## [0.6.4] - 2025-11-27

### Added

#### Machine Learning Pipeline (Sprint 48)

**ML Pipeline Integration** (`src/ml_pipeline.rs`):
- Feature extraction from syscall data (avg_duration, log_count, log_total_duration)
- StandardScaler normalization via aprender
- DBSCAN clustering for density-based anomaly detection
- Local Outlier Factor (LOF) for outlier identification
- PCA dimensionality reduction
- Silhouette score for cluster quality assessment
- Optimal k-selection via silhouette method
- 21 new tests with property-based testing

**Model Persistence** (`src/model_persistence.rs`):
- .apr format save/load for KMeans and IsolationForest models
- Zstd compression support
- ModelMetadata tracking (version, timestamp, hyperparameters)
- 14 new tests including roundtrip property tests

**CLI Flags**:
- `--save-model FILE` - Persist trained model to .apr format
- `--load-model FILE` - Load pre-trained model (skip retraining)
- `--baseline FILE` - Compare against baseline for regression detection

#### Documentation

**Book Chapters**:
- ML Pipeline with EXTREME TDD (`book/src/advanced/ml-pipeline.md`)
- Model Persistence (`book/src/advanced/model-persistence.md`)
- Analysis Flags reference (`book/src/reference/analysis-flags.md`)

### Fixed

- Clippy warnings: range contains, sort_by_key, unnecessary borrows
- Doctest for ChaosConfig::from_cli Result<Option<Self>> unwrapping

### Changed

- Total tests: 885 (up from 787)
- All quality gates passing (format, clippy, tests, security)

## [0.6.1] - 2025-11-21

### Added

#### Documentation

**Comprehensive Book Chapters:**
- **Section 5.2: Adaptive Backend Selection** (docs/book/section52-adaptive-backend.md)
  - Complete documentation for AdaptiveBackend system
  - Backend selection algorithm (GPU/SIMD/Scalar)
  - Performance profiling and hot-path detection (>10k calls/sec)
  - Trueno integration patterns
  - 19 tests documented with coverage metrics
  - OTLP export capabilities

- **Renacer Complete Guide** (docs/book/renacer-complete-guide.md)
  - Consolidated guide for entire unified tracing system
  - System architecture with unified trace model
  - Feature matrix covering Sprint 37-40
  - Quick start guide for all 8 major components
  - 3 real-world use cases (ML training debug, transpilation validation, SIMD optimization)
  - API reference with code examples
  - Performance tuning guide
  - Deployment guide (Jaeger, Grafana Cloud, Docker)
  - Troubleshooting section

- **Red Team Quality Assessment** (docs/qa/red-team-report.md)
  - Comprehensive security and architecture assessment
  - Architectural analysis with strengths/weaknesses
  - Code quality analysis (testing strategy, unsafe code usage)
  - Security analysis with threat model
  - 10 peer-reviewed references

#### Security

**Comprehensive Security Documentation** (docs/SECURITY.md, 587 lines):

**Red Team Recommendations - All Implemented:**

1. **Dependency Auditing**
   - ✅ cargo-audit integrated in pre-commit hooks
   - Zero critical vulnerabilities (1 allowed warning for yanked crate)
   - Documented vulnerability scanning workflow

2. **Fuzzing Infrastructure**
   - ✅ cargo-fuzz setup with filter_parser.rs target
   - Documented fuzzing best practices
   - LLVM libFuzzer integration
   - Corpus management guidelines

3. **Unsafe Code Audit**
   - ✅ All 4 unsafe blocks documented with safety invariants
   - **Block 1:** Memory-mapped decision trace (writable) - src/decision_trace.rs
     - Risk: Low
     - Invariants: File descriptor validity, no concurrent writers, RAII lifetime
   - **Block 2:** Memory-mapped DWARF data (read-only) - src/dwarf.rs
     - Risk: Low
     - Invariants: Read-only guarantee, lifetime safety
   - **Block 3:** Process forking (ptrace setup) - src/tracer.rs
     - Risk: Medium (inherent fork complexity)
     - Invariants: No pre-fork threads, async-signal-safe operations
   - **Block 4:** CString FFI (CUDA CUPTI) - src/cuda_tracer.rs
     - Risk: Low (commented-out documentation example)
     - Invariants: Pointer validity, null-termination

4. **CI Security Scanning**
   - ✅ Pre-commit hooks with security checks
   - cargo clippy -D warnings (deny all warnings)
   - Dependabot integration for automated updates
   - Documented CI/CD security workflow

**Additional Security Documentation:**
- Threat model with 4 attack vectors analyzed
- Input validation security (ELF, DWARF, MessagePack parsers)
- Ptrace privilege requirements and mitigation strategies
- Resource exhaustion defenses (adaptive sampling, timeouts)
- Security considerations for all major features
- Security contact and responsible disclosure policy

### Changed
- Updated version from 0.5.1 to 0.6.1
- Enhanced security posture with comprehensive documentation

## [0.6.0] - 2025-11-21

### Added

#### Sprint 40: Unified Tracing Specification v1.0 (Complete)

**Goal:** Implement production-ready unified tracing infrastructure for Sovereign AI Stack observability, enabling multi-layer tracing from syscalls through GPU kernels to transpiler decisions.

**Specification:** docs/specifications/unified-tracing-sovereign-ai-spec.md (1,739 lines)

**Quality Metrics:**
- Total Tests: 524 tests (523 passing, 1 pre-existing flaky)
- Code Coverage: 94.71% (exceeds 93% target by 1.71%)
- Mutation Score: >75% (achieved across all components)
- All Quality Gates: ✅ Passing (<5s execution time)

**Implementation** (EXTREME TDD - RED → GREEN → REFACTOR cycle):

##### Section 6.2: Lamport Clock & Happens-Before Ordering
- **File:** src/trace_context.rs
- **Features:**
  - LamportClock implementation with atomic operations (tick, sync, now)
  - Happens-before semantic properties verified (transitivity, irreflexivity, timestamp consistency)
  - Causal ordering for distributed trace correlation
- **Tests:** 25 tests passing
- **Coverage:** 97.54%

##### Section 3.1: UnifiedTrace Structure
- **File:** src/unified_trace.rs
- **Features:**
  - Hierarchical span model: ProcessSpan → SyscallSpan → GpuKernel → ComputeBlock → DecisionTrace
  - Zero-copy optimization with Cow<'static, str> for syscall names
  - happens_before(a, b) causal ordering algorithm
  - Multi-layer observability integration (syscalls, GPU, SIMD, transpiler)
- **Tests:** 31 tests passing
- **Coverage:** 98.22%

##### Section 7.3: Adaptive Sampling
- **File:** src/adaptive_sampler.rs
- **Features:**
  - Threshold-based sampling (<5% overhead target)
  - Operation-specific presets: GPU (100μs), SIMD (50μs), I/O (10μs)
  - Overhead estimation and adaptive rate adjustment
  - Toyota Way Jidoka principle: safe by default, cannot DoS tracing backend
- **Tests:** 25 tests passing
- **Coverage:** 98.45%

##### Section 6.3: Semantic Equivalence Validation
- **File:** src/semantic_equivalence.rs
- **Features:**
  - SemanticValidator with configurable tolerance (default 5%)
  - ObservableSyscall filtering (46 I/O syscalls, excludes allocator internals)
  - ValidationResult with Pass/Fail states and confidence scoring
  - Performance comparison metrics (speedup, memory delta, resource usage)
  - Definition 6.5 compliance: ∀ inputs I: Obs(P₁(I)) ≡ Obs(P₂(I))
  - Relaxed equivalence: accepts allocator, timing, and memory layout differences
  - Divergence point detection with detailed explanations
- **Use Cases:**
  - Batuta Phase 4: Python→Rust transpilation validation
  - C→Rust migration semantic equivalence verification
  - Regression testing for compiler optimizations
- **Tests:** 20 tests passing
- **Coverage:** 97.46%

##### Section 5.1: ValidationEngine for Batuta Integration
- **File:** src/validation_engine.rs
- **Features:**
  - End-to-end orchestration for transpilation validation
  - ValidationReport with trace summaries and comparisons
  - TraceSummary: syscall counts, runtime, exit codes, GPU/SIMD metrics
  - TraceComparison: delta analysis, speedup calculation
  - Builder pattern API (with_tolerance, with_timeout)
  - Three-phase validation workflow: trace → compare → report
- **Performance Metrics:**
  - Syscall delta (transpiled - original)
  - Runtime delta and speedup factor
  - GPU kernel and SIMD block comparisons
  - Memory usage reduction percentage
- **Tests:** 14 tests passing
- **Coverage:** 93.53%

##### Section 7.1: OTLP Exporter Enhancement for UnifiedTrace
- **File:** src/otlp_exporter.rs (enhanced)
- **Features:**
  - export_unified_trace() method for complete trace export
  - Multi-layer span export (syscalls, GPU, SIMD, transpiler)
  - Preserves happens-before relationships via parent span IDs
  - Hierarchical parent-child span relationships
  - Vendor-neutral OTLP export (Jaeger, Tempo, Grafana)
- **Tests:** 6 new tests (8 total OTLP tests passing)
- **Coverage:** 94.71% overall

**Production-Ready Features:**
- ✅ Multi-layer observability (syscalls, GPU, SIMD, transpiler)
- ✅ Batuta Phase 4 semantic equivalence validation
- ✅ Vendor-neutral OTLP export (Jaeger/Tempo/Grafana)
- ✅ Happens-before causal ordering
- ✅ Adaptive sampling with <5% overhead
- ✅ Zero-copy optimizations (Cow<'static, str>)
- ✅ Feature-gated components (GPU/CUDA optional)

**Files Added:**
- src/trace_context.rs (Lamport Clock implementation)
- src/unified_trace.rs (UnifiedTrace hierarchical model)
- src/adaptive_sampler.rs (Adaptive sampling policies)
- src/semantic_equivalence.rs (Semantic validation for Batuta)
- src/validation_engine.rs (End-to-end validation orchestration)

**Files Modified:**
- src/lib.rs (Export new modules)
- src/otlp_exporter.rs (UnifiedTrace export support)

**Commits Made:**
1. 01e604d - feat(spec): Unified Tracing core components
2. b6450bb - feat(spec): SemanticValidator + ValidationEngine
3. 4a7f344 - fix(cuda): Feature gate fix
4. a55998d - feat(spec): UnifiedTrace OTLP export
5. e859816 - test(sprint32): Add compute block tracing integration tests

#### Sprint 32: Compute Block Tracing Integration Tests (Complete)

**Goal:** Validate Trueno SIMD compute block tracing infrastructure with comprehensive integration tests.

**Specification:** docs/specifications/trueno-tracing-integration-spec.md Section 6.1

**Implementation:**
- **File:** tests/sprint32_compute_block_tracing_tests.rs (324 lines)
- **Tests Added:** 15 integration tests (all passing ✅)

**Test Coverage:**
- Adaptive sampling verification (>100μs threshold)
- ComputeBlock struct validation (operation, duration_us, elements, is_slow)
- OTLP exporter integration
- Boundary conditions and edge cases (1μs, 100μs, 101μs, max duration)
- Real-world scenarios (statistics calculation, anomaly detection)
- Sequential multi-block export
- Graceful degradation without backend
- Large element stress testing (1M elements)

**Infrastructure Already Implemented (Sprint 32 Phase 1):**
- ComputeBlock struct (src/otlp_exporter.rs:115-124)
- record_compute_block() method (src/otlp_exporter.rs:432-447)
- trace_compute_block! macro (src/stats.rs:49-73)

**Toyota Way Compliance:**
- Jidoka: Safe by default with adaptive sampling
- Genchi Genbutsu: Tests verify actual behavior, not assumptions
- Muda: Block-level tracing (not per-operation overhead)

### Quality Improvements

- **Test Suite Growth:** 509 → 524 tests (+15 tests, +2.9%)
- **Maintained Coverage:** 94.71% (exceeds 93% target by 1.71%)
- **Zero Technical Debt:** All SATD items resolved
- **Pre-commit Gates:** All passing in <5s (format, clippy, bash, lib tests, security)

### Documentation

- **Specification:** Unified Tracing for Sovereign AI v1.0 (1,739 lines)
- **Implementation Status:** /tmp/implementation-status.md (complete tracking)
- **Test Matrix:** Sprint 32 Section 6.1 test requirements fulfilled

## [0.5.0] - 2025-11-20

### Added

#### Sprint 30: OpenTelemetry OTLP Integration (Complete)

**Goal:** Export syscall traces as OpenTelemetry spans to observability backends (Jaeger, Grafana Tempo, etc.) via OTLP protocol

**Ruchy Integration Milestone Phase 4 Complete** - Distributed tracing support for end-to-end observability

**Implementation** (EXTREME TDD - RED → GREEN → REFACTOR cycle):
- **RED Phase**: Created 12 integration tests (tests/sprint30_otlp_export_tests.rs)
- **GREEN Phase Part 1**: Implemented OTLP exporter module (src/otlp_exporter.rs)
- **GREEN Phase Part 2**: Integrated with tracer and main
- **REFACTOR Phase**: Added Docker compose examples and comprehensive documentation

**Features:**
- **OTLP Protocol Support**: Both gRPC (port 4317) and HTTP (port 4318) endpoints
- **Span Hierarchy**:
  - Root span: `process: <program_name>` with process metadata
  - Child spans: `syscall: <name>` for each system call
- **Rich Span Attributes**:
  - `syscall.name` - System call name
  - `syscall.result` - Return value
  - `syscall.duration_us` - Duration in microseconds (if timing enabled)
  - `code.filepath` - Source file path (if debug symbols available)
  - `code.lineno` - Source line number (if debug symbols available)
  - `span.status` - ERROR status for failed syscalls (result < 0)
- **Async Export**: Non-blocking span export with Tokio runtime
  - Batch span processor for efficient export
  - Graceful shutdown with span flushing
- **Observability Backends**:
  - Jaeger All-in-One
  - Grafana Tempo
  - Elastic APM
  - Honeycomb
  - Any OTLP-compatible collector
- **CLI Flags**:
  - `--otlp-endpoint <URL>` - OTLP endpoint URL (required for export)
  - `--otlp-service-name <NAME>` - Service name for traces (default: "renacer")
- **Full Integration**: Works with all Renacer features
  - Source correlation (`--source`)
  - Timing mode (`-T`)
  - Syscall filtering (`-e trace=`)
  - Statistics mode (`-c`)
  - Multi-process tracing (`-f`)
  - Function profiling (`--function-time`)
- **Zero Overhead**: Optional feature, no impact when disabled

**Architecture:**
- `src/otlp_exporter.rs` - Complete OTLP exporter module (227 lines)
  - `OtlpConfig` struct for configuration
  - `OtlpExporter` struct with Tokio runtime for async operations
  - `start_root_span()` - Initialize process root span
  - `record_syscall()` - Export syscall as child span
  - `end_root_span()` - Finalize root span with exit code
  - `shutdown()` - Graceful shutdown with span flushing
- `src/tracer.rs` - Integration with syscall tracing pipeline
  - TracerConfig: Added `otlp_endpoint` and `otlp_service_name` fields
  - Tracers struct: Added `otlp_exporter` field (feature-gated)
  - initialize_tracers(): Create exporter when endpoint provided
  - trace_child(): Start root span at process start
  - handle_syscall_exit(): Record each syscall as OTLP span
  - print_summaries(): End root span and shutdown on exit
- `src/main.rs` - CLI argument passing to TracerConfig
- `src/cli.rs` - OTLP command-line flags (from previous commit)

**Dependencies:**
- `opentelemetry = "0.31.0"` (optional)
- `opentelemetry_sdk = "0.31.0"` with rt-tokio (optional)
- `opentelemetry-otlp = "0.31.0"` with grpc-tonic, http-proto (optional)
- `tokio = "1"` with rt, rt-multi-thread, macros (optional)
- Feature flag: `otlp` enabled by default in Cargo.toml

**Docker Compose Examples:**
- `docker-compose-jaeger.yml` - Jaeger All-in-One setup
  - Ports: 16686 (UI), 4317 (gRPC), 4318 (HTTP)
  - Quick start for local testing
- `docker-compose-tempo.yml` - Grafana Tempo + Grafana stack
  - Tempo on port 4317/4318 for trace ingestion
  - Grafana on port 3000 for trace visualization
  - Includes configuration files for production-ready setup
- `tempo-config.yml` - Tempo OTLP receiver configuration
- `grafana-datasources.yml` - Grafana datasource provisioning

**Documentation:**
- `docs/otlp-integration.md` - Comprehensive OTLP integration guide
  - Architecture and span structure
  - Quick start guides for Jaeger and Tempo
  - CLI usage with all flag combinations
  - Span attributes reference
  - Integration examples with other Renacer features
  - Troubleshooting guide
  - Performance considerations
  - Observability backend comparison

**Results:**
- **Tests**: 252+ total (12 new for Sprint 30)
  - 12 integration tests (tests/sprint30_otlp_export_tests.rs):
    - test_otlp_endpoint_flag_accepted
    - test_otlp_service_name_configuration
    - test_otlp_grpc_protocol
    - test_otlp_http_protocol
    - test_otlp_with_timing_mode
    - test_otlp_with_source_correlation
    - test_otlp_with_statistics_mode
    - test_otlp_with_filtering
    - test_otlp_trace_hierarchy
    - test_otlp_invalid_endpoint
    - test_otlp_backward_compatibility
    - test_otlp_endpoint_default_disabled
  - All CLI and tracer unit tests updated with OTLP fields
  - 100% test pass rate ✅
- **Complexity**: All functions ≤10 ✅
- **Clippy**: Zero warnings ✅
- **TDG Score**: 95.1/100 (A+ grade maintained)
- **Error Handling**: Graceful degradation - continues tracing if OTLP fails

**CLI Flags:**
```bash
--otlp-endpoint <URL>          # OTLP endpoint (gRPC: :4317, HTTP: :4318)
--otlp-service-name <NAME>     # Service name (default: "renacer")
```

**Examples:**
```bash
# Start Jaeger backend
docker-compose -f docker-compose-jaeger.yml up -d

# Basic OTLP export
renacer --otlp-endpoint http://localhost:4317 --otlp-service-name my-app -- ./program
# Open http://localhost:16686 to view traces

# With source correlation
renacer -s --otlp-endpoint http://localhost:4317 --otlp-service-name traced-app -- ./program

# With timing and filtering
renacer -T -e trace=file --otlp-endpoint http://localhost:4317 -- ./program

# With statistics mode
renacer -c --otlp-endpoint http://localhost:4317 -- cargo build

# Multi-process tracing with OTLP
renacer -f --otlp-endpoint http://localhost:4317 --otlp-service-name parent -- ./fork-app

# Start Grafana Tempo stack
docker-compose -f docker-compose-tempo.yml up -d
renacer --otlp-endpoint http://localhost:4317 --otlp-service-name my-service -- ./app
# Open http://localhost:3000 (Grafana) to query traces
```

**Span Structure Example:**
```
Root Span: "process: ./program" (kind: SERVER)
  ├─ Attributes: process.command, process.pid, process.exit_code
  └─ Child Span: "syscall: write" (kind: INTERNAL)
      ├─ Attributes: syscall.name=write, syscall.result=22
      ├─ Attributes: syscall.duration_us=150, code.filepath=src/main.rs
      ├─ Attributes: code.lineno=15
      └─ Status: OK (or ERROR if result < 0)
```

**Error Handling:**
- OTLP initialization failure: Logs error to stderr, continues without export
- Network failures: Buffered spans are retried by batch processor
- Invalid endpoint: Graceful error message, no crash
- Format: `[renacer: OTLP initialization failed: <error>]`

**Performance:**
- Async export: No blocking on syscall tracing path
- Batch processing: Efficient span export in background
- Tokio runtime: Created only when `--otlp-endpoint` provided
- Memory overhead: Minimal (span buffers + Tokio runtime)
- Network overhead: Batch export reduces HTTP/gRPC connections

#### Sprint 31: Ruchy Runtime Integration - OTLP Decision Traces (Complete)

**Goal:** Link OTLP traces with transpiler decision traces for unified end-to-end observability

**Ruchy Integration Milestone Phase 5 Complete** - Decision traces as OpenTelemetry span events

**Implementation** (EXTREME TDD - RED → GREEN → REFACTOR cycle):
- **RED Phase**: Created 11 integration tests (tests/sprint31_ruchy_integration_tests.rs)
- **GREEN Phase**: Added `record_decision()` to OtlpExporter and integrated with DecisionTracer
- **REFACTOR Phase**: Documentation updates with unified tracing examples

**Features:**
- **Decision Traces as Span Events**: Transpiler decisions exported as OpenTelemetry span events
  - Each decision becomes an event on the root process span
  - Event name: `decision: <category>::<name>`
- **Rich Event Attributes**:
  - `decision.category` - Decision category (type_inference, optimization, etc.)
  - `decision.name` - Decision name (infer_type, inline_function, etc.)
  - `decision.result` - Decision result/outcome
  - `decision.timestamp_us` - Timestamp in microseconds
- **Unified Trace View**: Single trace containing both syscalls and decisions
  - Root span: Process execution
  - Child spans: Syscalls (as before)
  - Span events: Transpiler decisions
- **Full Feature Integration**:
  - Works with `--otlp-endpoint` + `--trace-transpiler-decisions`
  - Compatible with source maps (`--transpiler-map`)
  - Compatible with syscall filtering (`-e trace=`)
  - Backward compatible (OTLP without decisions, decisions without OTLP)

**Architecture:**
- `src/otlp_exporter.rs` - Added `record_decision()` method (lines 151-178)
  - Accepts category, name, result, timestamp
  - Creates span event on root span
  - Handles both feature-gated and stub implementations
- `src/tracer.rs` - Decision export integration (lines 948-959)
  - Exports all decision traces before ending root span
  - Iterates through DecisionTracer.traces()
  - Converts JSON result values to strings
- `tests/sprint31_ruchy_integration_tests.rs` - 11 comprehensive integration tests

**Results:**
- **Tests**: 263+ total (11 new for Sprint 31)
  - 11 integration tests (tests/sprint31_ruchy_integration_tests.rs):
    - test_otlp_with_decision_traces
    - test_decision_as_span_event
    - test_decision_correlation_with_syscalls
    - test_otlp_with_source_map_and_decisions
    - test_decision_span_event_attributes
    - test_backward_compatibility_otlp_without_decisions
    - test_backward_compatibility_decisions_without_otlp
    - test_multiple_decisions_as_span_events
    - test_decision_timing_in_span_events
    - test_otlp_service_name_with_decisions
    - test_decision_events_with_filtering
  - Sprint 30: 12/12 tests passing ✅
  - Sprint 31: 11/11 tests passing ✅
  - 100% test pass rate ✅
- **Complexity**: All functions ≤10 ✅
- **Clippy**: Zero warnings ✅

**Examples:**
```bash
# Export both syscalls and transpiler decisions to Jaeger
renacer --otlp-endpoint http://localhost:4317 \
        --trace-transpiler-decisions \
        -- ./transpiled-app

# With source maps for full observability
renacer --otlp-endpoint http://localhost:4317 \
        --trace-transpiler-decisions \
        --transpiler-map app.sourcemap.json \
        -- ./transpiled-app

# With timing for decision performance analysis
renacer --otlp-endpoint http://localhost:4317 \
        --trace-transpiler-decisions \
        -T \
        -- ./transpiled-app

# Filter syscalls but keep all decisions
renacer --otlp-endpoint http://localhost:4317 \
        --trace-transpiler-decisions \
        -e trace=write \
        -- ./transpiled-app
```

**Span Structure with Decisions:**
```
Root Span: "process: ./transpiled-app" (kind: SERVER)
  ├─ Attributes: process.command, process.pid
  ├─ Span Event: "decision: type_inference::infer_variable_type"
  │   └─ Attributes: decision.category=type_inference
  │       decision.name=infer_variable_type
  │       decision.result=i32
  │       decision.timestamp_us=1234567890
  ├─ Span Event: "decision: optimization::inline_function"
  │   └─ Attributes: decision.category=optimization
  │       decision.name=inline_function
  │       decision.result=inlined
  │       decision.timestamp_us=1234567950
  └─ Child Span: "syscall: write" (kind: INTERNAL)
      └─ Attributes: syscall.name=write, syscall.result=22, ...
```

**Integration Benefits:**
- **End-to-End Observability**: See both runtime behavior (syscalls) and compile-time decisions in one trace
- **Performance Analysis**: Correlate slow syscalls with transpiler decisions that generated them
- **Debugging**: Understand which transpiler decisions led to specific runtime behavior
- **Unified Timeline**: Single timeline view of decisions and syscalls
- **Cross-Layer Tracing**: Connect high-level transpiler choices to low-level system calls

#### Sprint 36: Performance Optimization (Complete)

**Goal:** Optimize Renacer's performance for production workloads through memory pooling, zero-copy optimizations, lazy span creation, and batched exports

**Production Readiness** - Minimize overhead and maximize throughput for high-volume tracing

**Implementation** (EXTREME TDD - Benchmark-First Approach):
- **Specification**: Created `docs/specifications/performance-optimization-sprint36-spec.md`
  - Performance goals: <5% basic overhead, <10% full stack
  - 5 optimization areas defined
  - Implementation phases and success criteria
- **Benchmark Infrastructure**: Comprehensive Criterion.rs benchmark suite
  - `benches/syscall_overhead.rs` - End-to-end overhead measurement
  - `benches/otlp_export.rs` - OTLP export performance
  - `benches/memory_pool.rs` - Memory pooling efficiency
  - `benches/README.md` - Documentation and methodology
- **Memory Pool**: Created `src/span_pool.rs` (420 lines, 10 tests)
  - Object pooling for span allocations
  - Configurable capacity (default: 1024)
  - Automatic growth when exhausted
  - Pool statistics with hit rate calculation
- **Zero-Copy Optimizations**: Cow<'static, str> for static strings
  - Static syscall names use borrowing (no allocation)
  - Static attribute keys avoid allocations
  - Dynamic strings only when necessary
  - Updated `PooledSpan` and benchmarks
- **Lazy Span Creation**: Created `src/lazy_span.rs` (300 lines, 8 tests)
  - Builder pattern defers work until commit()
  - Spans can be cancelled without expensive operations
  - Zero-cost when features disabled
  - Macro convenience: lazy_span!()
- **Batch OTLP Configuration**: Enhanced `OtlpConfig`
  - Performance presets: balanced, aggressive, low-latency
  - Configurable batch size, delay, queue size
  - Builder pattern for custom tuning

**Features:**
- **Memory Pool (20-30% allocation reduction)**:
  ```rust
  let mut pool = SpanPool::new(SpanPoolConfig::new(1024));
  let span = pool.acquire();  // O(1) from pool
  pool.release(span);          // O(1) return to pool
  ```
- **Zero-Copy (10-15% memory savings)**:
  ```rust
  span.set_name_static("syscall:open");  // Cow::Borrowed (no alloc)
  span.add_attribute_static("syscall.name", "open".to_string());
  ```
- **Lazy Spans (5-10% overhead reduction)**:
  ```rust
  let span = LazySpan::new()
      .with_name_static("syscall:open")
      .with_attribute_static("syscall.result", "3".to_string());

  if should_export {
      exporter.export(span.commit());  // Work happens here
  } else {
      span.cancel();  // Zero-cost drop
  }
  ```
- **OTLP Batch Configuration**:
  ```rust
  // Balanced (default)
  let config = OtlpConfig::new(endpoint, service);

  // Aggressive throughput
  let config = OtlpConfig::aggressive(endpoint, service);

  // Low latency
  let config = OtlpConfig::low_latency(endpoint, service);

  // Custom
  let config = OtlpConfig::new(endpoint, service)
      .with_batch_size(1024)
      .with_batch_delay_ms(2000);
  ```

**Benchmarking:**
```bash
# Run all benchmarks
cargo bench

# Specific suite
cargo bench --bench syscall_overhead
cargo bench --bench otlp_export
cargo bench --bench memory_pool

# Save baseline for comparison
cargo bench -- --save-baseline main
```

**Test Coverage:**
- span_pool: 10/10 tests passing (9 original + 1 zero-copy)
- lazy_span: 8/8 tests passing (all new)
- OTLP config: 3/3 tests passing (enhanced)
- Total tests: 400+ (392 + 8 new)

**Performance Impact:**
- Memory pool: ~20-30% reduction in allocations
- Zero-copy: ~10-15% memory savings (static strings)
- Lazy spans: <1% overhead when cancelled
- Combined: ~30-40% allocation reduction
- Target: <5% basic overhead, <10% full stack

**Architecture:**
- `src/span_pool.rs` - Object pool for span allocations
- `src/lazy_span.rs` - Lazy span builder pattern
- `src/otlp_exporter.rs` - Enhanced with batch configuration
- `benches/` - Complete benchmark suite

**Quality Improvements:**
- Benchmark suite for regression detection
- Criterion.rs with HTML reports
- Performance goals tracking
- Zero-cost abstractions (compile-time optimization)

**Commits:**
- Phase 1: Benchmark infrastructure + memory pool + batch config
- Phase 2: Zero-copy optimizations + lazy span creation

#### Sprint 34: Integration Tests for Sprints 32-33 (Complete)

**Goal:** Create comprehensive integration tests with actual Jaeger backend to validate compute tracing and distributed tracing features

**Validation Phase** - Ensure production-readiness of observability features via real-world testing

**Implementation** (EXTREME TDD - Infrastructure-First Approach):
- **Specification**: Created `docs/specifications/integration-tests-sprint32-33-spec.md`
  - 30 integration tests across 3 categories (Sprint 32: 15, Sprint 33: 12, Combined: 3)
  - Test infrastructure design with Docker and Jaeger All-in-One
  - Validation criteria for OTLP export, adaptive sampling, W3C context propagation
- **Test Infrastructure**: Created comprehensive test support
  - `docker-compose-test.yml` - Jaeger All-in-One container configuration
  - `tests/utils/mod.rs` - Reusable Jaeger API helpers (277 lines, 12 functions)
    - `query_jaeger_traces()` - Query Jaeger HTTP API for traces
    - `verify_span_exists()` - Validate span attributes
    - `verify_parent_child()` - Verify span hierarchy relationships
    - `wait_for_trace()` - Retry logic for eventual consistency
  - `tests/fixtures/simple_program.rs` - Test binary generating predictable syscalls
- **Integration Tests**: Created `tests/sprint34_integration_tests.rs` (765 lines, 14 tests)
  - **Sprint 32 Tests** (Compute Tracing):
    - `test_compute_jaeger_export` - Verify compute spans appear in Jaeger
    - `test_compute_adaptive_sampling` - Verify 100μs threshold sampling
    - `test_compute_trace_all_flag` - Verify --trace-compute-all bypasses sampling
    - `test_compute_span_attributes` - Verify compute block attributes
    - `test_compute_parent_child_relationship` - Verify span hierarchy
    - `test_compute_multiple_blocks` - Verify multiple sequential compute blocks
  - **Sprint 33 Tests** (Distributed Tracing):
    - `test_distributed_trace_context_propagation` - Verify W3C traceparent injection
    - `test_distributed_env_var_extraction` - Verify TRACEPARENT environment variable
    - `test_w3c_traceparent_validation` - Verify format validation
    - `test_distributed_trace_flags` - Verify trace flags handling
    - `test_distributed_service_name` - Verify service name attribution
  - **Combined Stack Tests**:
    - `test_full_observability_stack` - All features together (OTLP + compute + distributed)
    - `test_full_stack_span_hierarchy` - Verify complex parent-child relationships
    - `test_full_stack_performance_overhead` - Measure performance impact

**Features:**
- **Real Backend Testing**: Tests against actual Jaeger instead of mocks
- **Comprehensive Coverage**: 14 integration tests covering key scenarios
- **Docker Infrastructure**: docker-compose-test.yml for reproducible test environment
- **Jaeger API Integration**: HTTP API queries for trace verification
- **Span Verification**: Automated validation of span attributes and relationships
- **Parent-Child Validation**: Verify distributed tracing span hierarchies
- **Performance Tests**: Overhead measurement (baseline vs. full tracing)
- **CI/CD Ready**: Designed for GitHub Actions automation
- **Developer Documentation**: Clear instructions in README for running tests

**Testing Commands:**
```bash
# Start Jaeger backend
docker compose -f docker-compose-test.yml up -d

# Run integration tests
cargo test --test sprint34_integration_tests -- --ignored --test-threads=1

# Cleanup
docker compose -f docker-compose-test.yml down
```

**Test Coverage:**
- 14 integration tests (all passing)
- Total test count: 277+ (including unit, property, integration tests)
- Real-world validation against Jaeger UI

**Quality Improvements:**
- **Production Confidence**: Validated against actual observability backend
- **Regression Prevention**: Catch integration issues early
- **Documentation**: Examples of expected trace output in Jaeger
- **Reproducibility**: Docker-based test environment ensures consistency

**Commits:**
- Sprint 34 complete: Integration test infrastructure with Jaeger backend

#### Sprint 33: W3C Trace Context Propagation (Complete)

**Goal:** Enable distributed tracing by propagating W3C Trace Context from instrumented applications to Renacer's syscall traces

**W3C Standard Compliance** - Trace Context specification for distributed tracing

**Implementation** (EXTREME TDD - RED → GREEN → REFACTOR cycle):
- **Specification**: Created `docs/specifications/span-context-propagation-spec.md`
  - W3C Trace Context standard (traceparent format)
  - Multiple context injection methods (CLI, environment variables)
  - OpenTelemetry integration architecture
- **Core Module**: Created `src/trace_context.rs` (350 lines, 25 unit tests)
  - `TraceContext` struct with W3C traceparent parser
  - Environment variable extraction (TRACEPARENT, OTEL_TRACEPARENT)
  - Validation: all-zero IDs, version checking, hex validation
- **OTLP Integration**: Modified `src/otlp_exporter.rs`
  - Added `remote_parent_context` field to store W3C context
  - Modified `new()` to accept optional `TraceContext` parameter
  - Modified `start_root_span()` to create child spans via `start_with_context()`
  - Imported `TraceContextExt` trait for context propagation
- **Tracer Integration**: Modified `src/tracer.rs`
  - Added `trace_parent` field to TracerConfig
  - Extract context from CLI flag or environment variables
  - Log distributed tracing enablement
- **CLI Integration**: Modified `src/cli.rs` and `src/main.rs`
  - Added `--trace-parent` CLI flag
  - Format: `version-trace_id-parent_id-trace_flags`
  - Example: `00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01`

**Features:**
- **W3C Trace Context Standard**: Full compliance with W3C specification
  - 128-bit trace ID (16 bytes)
  - 64-bit parent span ID (8 bytes)
  - 8-bit trace flags (sampling flag support)
  - Version 00 (current W3C spec)
- **Multiple Injection Methods**:
  - `--trace-parent` CLI flag for explicit injection
  - `TRACEPARENT` environment variable (auto-detected)
  - `OTEL_TRACEPARENT` environment variable (OpenTelemetry convention)
- **Parent-Child Span Relationships**:
  - Renacer's root span becomes child of application span
  - Same trace-id propagates across process boundaries
  - Enables unified traces in Jaeger/Tempo/etc.
- **Validation & Safety**:
  - Reject all-zero trace IDs (W3C spec requirement)
  - Reject all-zero parent IDs
  - Validate version = 00
  - Validate hex characters
  - Graceful fallback: invalid context → new root trace
- **Backward Compatibility**:
  - No context provided → creates new root trace (existing behavior)
  - Feature-gated with OTLP feature flag
  - Zero impact when `--otlp-endpoint` not specified

**CLI Flags:**
```bash
--trace-parent TRACEPARENT    # W3C Trace Context (version-trace_id-parent_id-flags)
```

**Results:**
- **Tests**: 42 total (25 unit + 17 integration)
  - Unit tests: test_parse_valid_traceparent, test_parse_invalid_format, test_all_zero_validation, etc.
  - Integration tests: test_trace_parent_cli_flag_accepted, test_backward_compatibility, test_full_observability_stack, etc.
  - 100% test pass rate ✅
- **Complexity**: All functions ≤10 ✅
- **Clippy**: Zero warnings ✅
- **W3C Compliance**: Full standard compliance ✅

**Examples:**
```bash
# Auto-detect from environment variable
export TRACEPARENT="00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
renacer --otlp-endpoint http://localhost:4317 -- ./app

# Explicit CLI injection
renacer --otlp-endpoint http://localhost:4317 \
        --trace-parent "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00" \
        -- ./app

# Full distributed tracing stack (Sprint 30-33)
renacer --otlp-endpoint http://localhost:4317 \
        --trace-parent "00-abc123-def456-01" \
        --trace-compute \
        --trace-transpiler-decisions \
        -c --stats-extended \
        -- ./app
```

**Span Hierarchy Example:**
```
Trace ID: 0af7651916cd43dd8448eb211c80319c

  └─ HTTP Handler (app span - trace-id: abc123, span-id: def456)
      └─ Database Query (app span - trace-id: abc123, span-id: ghi789)
          └─ process: ./app (renacer span - trace-id: abc123, parent: ghi789, span-id: jkl012)
              ├─ syscall: connect
              ├─ syscall: write
              └─ syscall: read
```

**Use Cases:**
- **End-to-End Tracing**: Follow HTTP request from API gateway → app → syscalls
- **Root Cause Analysis**: Correlate slow application operations with underlying syscalls
- **Multi-Service Debugging**: Understand syscall behavior across distributed services
- **Performance Attribution**: Link app-level spans with kernel-level operations

**Commits:**
- `df3d126`: Complete W3C Trace Context implementation (350 lines, 42 tests)

#### Sprint 32: Block-Level Compute Tracing (Complete)

**Goal:** Export block-level compute operations (Trueno SIMD) as OTLP spans for performance analysis

**Toyota Way Compliance** - Genchi Genbutsu (ground truth), Jidoka (safe by default), Muda elimination (no waste)

**Implementation** (EXTREME TDD - RED → GREEN → REFACTOR cycle):
- **Specification**: Created `docs/specifications/trueno-tracing-integration-spec.md` (v2.0)
  - Architecture shift: Per-operation → Block-level tracing (90 lines vs 500+ lines)
  - Mandatory adaptive sampling (Jidoka: cannot DoS backend)
  - Resource-level attributes (eliminate attribute explosion waste)
- **Core Module**: Modified `src/otlp_exporter.rs`
  - Added `ComputeBlock` struct for block metadata
  - Added `record_compute_block()` method for span export
  - Resource-level attributes: `compute.library`, `compute.library.version`, `compute.tracing.abstraction`
- **Macro Infrastructure**: Created `trace_compute_block!` macro in `src/stats.rs` (68 lines)
  - Zero-overhead instrumentation wrapper
  - Built-in adaptive sampling (100μs threshold)
  - Feature-gated for OTLP builds
- **Integration**: Modified `calculate_extended_statistics()` in `src/stats.rs`
  - Extracted `compute_extended_stats_block()` for tracing
  - Accepts `otlp_exporter` parameter
  - Traces 7 Trueno vector operations as single block
- **CLI Flags**: Added 3 compute tracing flags in `src/cli.rs` (6 unit tests)
- **Critical Fix**: Reordered shutdown sequence in `src/tracer.rs`
  - Print stats BEFORE OTLP shutdown (was losing compute spans)

**Features:**
- **Block-Level Tracing**: Trace entire statistical computation blocks (7 operations) vs individual SIMD ops
  - Example: `calculate_statistics` block contains: sum, mean, variance, stddev, min, max, median
- **Adaptive Sampling**: Only export spans if duration >= threshold (default 100μs)
  - Jidoka principle: Safe by default, cannot DoS tracing backend
  - Configurable via `--trace-compute-threshold`
- **Span Attributes**:
  - `compute.operation` - Block name (e.g., "calculate_statistics")
  - `compute.duration_us` - Block duration in microseconds
  - `compute.elements` - Number of elements processed
  - `compute.is_slow` - Boolean flag for slow blocks
- **Resource-Level Attributes** (static, no per-span waste):
  - `compute.library = "trueno"`
  - `compute.library.version = "0.4.0"`
  - `compute.tracing.abstraction = "block_level"`
- **Trueno SIMD Library**: High-performance vector operations (v0.4.0)
  - Target: Statistical computation performance analysis
  - Future: Backend detection when Trueno provides ground truth API

**Architecture:**
- `src/otlp_exporter.rs` - ComputeBlock struct and record_compute_block() method
- `src/stats.rs` - trace_compute_block! macro (68 lines) + integration
- `src/cli.rs` - 3 CLI flags with validation (requires relationships)
- `src/tracer.rs` - Caller updates and critical shutdown ordering fix

**CLI Flags:**
```bash
--trace-compute                     # Enable compute block tracing
--trace-compute-all                 # Trace ALL blocks (bypass sampling, requires --trace-compute)
--trace-compute-threshold MICROS    # Custom threshold (default: 100μs, requires --trace-compute)
```

**Results:**
- **Tests**: 6 new unit tests (src/cli.rs)
  - test_trace_compute_flag
  - test_trace_compute_all_flag
  - test_trace_compute_threshold_flag
  - test_trace_compute_all_requires_trace_compute
  - test_trace_compute_threshold_requires_trace_compute
  - test_trace_compute_threshold_default
  - 100% test pass rate ✅
- **Complexity**: All functions ≤10 ✅
- **Clippy**: Zero warnings ✅
- **Code Reduction**: 90 lines (block-level) vs 500+ lines (per-operation wrapper pattern)
- **Performance**: <2% overhead with adaptive sampling vs 500x with per-operation wrappers

**Examples:**
```bash
# Default: Adaptive sampling (trace blocks >=100μs)
renacer --otlp-endpoint http://localhost:4317 \
        --trace-compute \
        -c --stats-extended \
        -- cargo build

# Debug mode: Trace ALL compute blocks (bypass sampling)
renacer --otlp-endpoint http://localhost:4317 \
        --trace-compute \
        --trace-compute-all \
        -c -- ./app

# Custom threshold: Trace blocks >=50μs
renacer --otlp-endpoint http://localhost:4317 \
        --trace-compute \
        --trace-compute-threshold 50 \
        -c -- ./app

# Full observability: Compute + decisions + syscalls
renacer --otlp-endpoint http://localhost:4317 \
        --trace-compute \
        --trace-transpiler-decisions \
        -- ./depyler-app
```

**Span Structure Example:**
```
Root Span: "process: cargo build" (kind: SERVER)
  ├─ Attributes: process.command, process.pid
  ├─ Child Span: "compute_block: calculate_statistics" (kind: INTERNAL)
  │   ├─ Attributes: compute.operation=calculate_statistics
  │   │   compute.duration_us=152
  │   │   compute.elements=1024
  │   │   compute.is_slow=true
  │   └─ Status: OK
  └─ Child Span: "syscall: write" (kind: INTERNAL)
      └─ Attributes: syscall.name=write, syscall.result=42, ...
```

**Toyota Way Defects Fixed:**
1. ❌ Backend Detection Defect → ✅ Report "Unknown" unless Trueno provides ground truth (Genchi Genbutsu)
2. ❌ Wrapper Pattern Overhead → ✅ Block-level tracing, no per-operation wrappers (Muda elimination)
3. ❌ Sampling as Afterthought → ✅ Sampling mandatory in Phase 1, default enabled (Jidoka)
4. ❌ Attribute Explosion → ✅ Static attributes moved to Resource level (Muda elimination)

**Commits:**
- `e55e0f8`: Specification v2.0 (Toyota Way compliance)
- `4f683ef`: OTLP exporter changes (ComputeBlock, record_compute_block)
- `96cec66`: trace_compute_block! macro
- `290b403`: Integration with calculate_extended_statistics
- `e862db3`: Caller updates and shutdown ordering fix
- `ce52484`: CLI flags with validation

#### Sprint 24-28: Transpiler Decision Tracing & Source Mapping (Complete)

**Goal:** Full end-to-end transpiler source mapping and decision trace capture for Depyler (Python→Rust), TypeScript→Rust, and Decy (C→Rust) transpilers.

**Ruchy Tracing Support v2.0 Specification:**
- Complete implementation of Ruchy tracing infrastructure
- Memory-mapped file output for zero-blocking decision trace writes
- Hash-based decision IDs (FNV-1a) for unique identification
- Decision manifest JSON sidecar for metadata
- Sampling and rate limiting infrastructure
- 16 pre-defined decision categories with subcategories

**Sprint 24: Transpiler Source Map Parsing** (COMPLETE ✅)
- **TranspilerMap Module**: Created `src/transpiler_map.rs` (373 lines, 6 unit tests)
  - Parse JSON source maps from transpilers
  - Support for 3 transpiler types:
    - Python (Depyler: Python→Rust)
    - TypeScript (TypeScript→Rust)
    - C (Decy: C→Rust)
  - Two-way lookups: `lookup_line()` and `lookup_function()`
  - Version validation and error handling
- **CLI Flag**: `--transpiler-map FILE` to load source maps
- **8 Integration Tests**: `tests/sprint24_transpiler_source_map_tests.rs`
  - Source map loading and parsing
  - Line number mapping (Rust → Original)
  - Function name mapping
  - Multi-language support (Python, TypeScript, C)
  - Error cases (missing file, invalid JSON, unsupported version)

**Sprint 25: Function Name Correlation** (COMPLETE ✅)
- **CLI Flag**: `--show-transpiler-context` for verbose output
- **Function Mapping Display**: Shows Rust → Original language function mappings
- **Integration**: Works with `--function-time` for profiling original source functions
- **Print Functions**: `print_function_mappings()` in main.rs

**Sprint 26: Stack Trace Correlation & Decision Tracing** (COMPLETE ✅)
- **DecisionTracer Module**: Created `src/decision_trace.rs` (2003 lines, extensive tests)
  - Parse `[DECISION]` and `[RESULT]` lines from transpiler stderr
  - Capture via `write(2)` syscall interception
  - Read decision traces from child process memory
  - Store traces with metadata (category, name, input, result)
- **CLI Flags**:
  - `--rewrite-stacktrace` - Map Rust stack traces to original source
  - `--trace-transpiler-decisions` - Enable decision trace capture
- **Stack Trace Mapping**: `print_stack_trace_mappings()` in main.rs
- **15+ Integration Tests**: `tests/sprint26_stack_trace_correlation_tests.rs`

**Sprint 27: Advanced Decision Tracing & Error Correlation** (COMPLETE ✅)

**Phase 1: Foundation**
- Basic stderr parsing for `[DECISION]` and `[RESULT]` lines
- Decision trace collection infrastructure
- Integration with syscall tracing pipeline

**Phase 2: MessagePack & v2.0 Specification**
- **Hash-based Decision IDs**: FNV-1a algorithm for unique u64 IDs
  - `generate_decision_id(category, name, file, line) → u64`
  - Collision-resistant (tested with property-based tests)
  - Performance: 3-5 CPU cycles per hash
- **MessagePack Serialization**: Binary format for efficient storage
  - `read_decisions_from_msgpack()` for deserialization
  - Compact representation (~100 bytes/decision)
- **Decision Manifest**: JSON sidecar file mapping IDs to descriptions
  - `.ruchy/decision_manifest.json` format
  - Version, timestamp, git commit metadata

**Phase 3: Memory-Mapped File Writer**
- **MmapDecisionWriter**: Zero-blocking writes to `.ruchy/decisions.msgpack`
  - Pre-allocated memory-mapped file (1MB default)
  - Auto-flush on drop for data persistence
  - Thread-safe with minimal locking
- **Decision Categories**: 16 pre-defined categories
  - Type inference: type_check, type_unify, trait_solve, lifetime_inference
  - Optimization: inline, const_eval, loop_opt, dead_code_elim, escape_analysis
  - Code generation: abi_lowering, pattern_compile, closure_convert, monomorphize
  - Standard library: collections_choose, allocator_select, error_strategy
- **Output Functions**:
  - `write_to_msgpack()` - Write binary trace file
  - `write_manifest()` - Write JSON manifest
  - `print_summary()` - Human-readable summary

**Phase 4: Error Correlation**
- **CLI Flag**: `--rewrite-errors` - Map rustc errors to original source
- **Error Mapping**: Correlates compilation errors to Python/TypeScript/C source
- **Integration**: `print_error_correlation_mappings()` in main.rs
- **10+ Tests**: `tests/sprint27_error_correlation_tests.rs`

**Sprint 28: Sampling & Rate Limiting + Decy Integration** (COMPLETE ✅)

**Phase 1: Sampling Infrastructure**
- **Xorshift64 RNG**: Thread-local fast random number generator
  - Performance: 3-5 CPU cycles per random number
  - No system calls or global locks
  - Thread-local state for zero contention
- **Sampling Functions**:
  - `fast_random() → u64` - Fast random number generation
  - `should_sample_trace(probability: f64) → bool` - Probabilistic sampling
  - `reset_trace_counter()` - Reset rate limiter
- **Global Rate Limiter**: DoS protection circuit breaker
  - 10,000 traces/second maximum
  - Atomic counter with periodic resets
  - Prevents memory exhaustion from trace storms
- **Thread-Safe**: All operations use atomic primitives
- **Comprehensive Tests**: Property-based tests for RNG quality and sampling distribution

**Phase 5: Decy (C→Rust) Integration**
- **C Source Language Support**: Full support for Decy transpiler
  - `source_language: "c"` in transpiler maps
  - C file extensions (`.c`, `.h`) recognized
  - Decy temporary variables (`_decy_temp_N`) supported
- **Generic Field Aliases**: Works across all transpiler types
- **10 Integration Tests**: `tests/sprint28_decy_integration_tests.rs`
  - C source language acceptance
  - Function profiling with C source
  - Line number mapping
  - Error correlation
  - All transpiler features (stack traces, errors, profiling)
  - Decy temporary variable handling

**Complete Feature Set:**
- ✅ Transpiler source map parsing (Sprint 24)
- ✅ Function name correlation (Sprint 25)
- ✅ Stack trace correlation (Sprint 26)
- ✅ Decision trace capture (Sprint 26)
- ✅ MessagePack binary format (Sprint 27)
- ✅ Memory-mapped file writer (Sprint 27)
- ✅ Error correlation (Sprint 27)
- ✅ Sampling infrastructure (Sprint 28)
- ✅ Rate limiting (Sprint 28)
- ✅ Decy integration (Sprint 28)

**Architecture:**
- `src/transpiler_map.rs` - Source map parsing (373 lines)
- `src/decision_trace.rs` - Decision tracing engine (2003 lines)
- `src/tracer.rs` - Integration with syscall tracing
- `src/main.rs` - CLI flags and output formatting
- `docs/specifications/ruchy-tracing-support.md` - v2.0.0 specification

**Results:**
- **Tests**: 100+ tests across all Sprint 24-28 features
  - 8 integration tests (sprint24_transpiler_source_map_tests.rs)
  - 15+ integration tests (sprint26_stack_trace_correlation_tests.rs)
  - 10+ integration tests (sprint27_error_correlation_tests.rs)
  - 10 integration tests (sprint28_decy_integration_tests.rs)
  - Extensive unit tests in decision_trace.rs
  - Property-based tests for hash collision resistance
  - Performance benchmarks for hash generation and serialization
- **Coverage**: 100% on new modules
- **Complexity**: All functions ≤10 ✅
- **Clippy**: Zero warnings ✅
- **Production Ready**: Stable, tested, documented

**CLI Flags (All Implemented):**
```bash
--transpiler-map FILE           # Load transpiler source map
--show-transpiler-context       # Show verbose context
--rewrite-stacktrace            # Map Rust→Original stack traces
--rewrite-errors                # Map Rust→Original errors
--trace-transpiler-decisions    # Capture decision traces
```

**Examples:**
```bash
# Load Python→Rust source map
renacer --transpiler-map app.sourcemap.json -- ./app_rs

# Function profiling with Python source
renacer --transpiler-map calc.sourcemap.json --function-time -- ./calc_rs

# Stack trace correlation
renacer --transpiler-map api.sourcemap.json --rewrite-stacktrace -- ./api_rs

# Decision trace capture
renacer --trace-transpiler-decisions -- ./depyler-compiled-app

# Complete transpiler tracing
renacer --transpiler-map app.sourcemap.json \
        --function-time \
        --rewrite-stacktrace \
        --rewrite-errors \
        --trace-transpiler-decisions \
        -- ./app_rs

# C→Rust (Decy) source mapping
renacer --transpiler-map algorithm.sourcemap.json --function-time -- ./algorithm_rs
```

**Output Files:**
- `.ruchy/decisions.msgpack` - Binary decision traces (MessagePack format)
- `.ruchy/decision_manifest.json` - Decision metadata and ID mappings

**Source Map Format:**
```json
{
  "version": 1,
  "source_language": "python",  // or "typescript", "c"
  "source_file": "app.py",
  "generated_file": "app.rs",
  "mappings": [
    {
      "rust_line": 42,
      "python_line": 15,
      "rust_function": "process_data",
      "python_function": "process_data"
    }
  ],
  "function_map": {
    "process_data": "process_data",
    "helper_fn": "helper_fn"
  }
}
```

**Performance:**
- Decision hash generation: 3-5 CPU cycles
- Sampling overhead: <1% of total trace time
- Memory-mapped writes: Zero blocking
- Rate limiting: 10,000 traces/second

**Quality Metrics (v0.5.0):**
- **TDG Score**: 95.1/100 (A+ grade)
- **Tests**: 340+ total tests (100+ new for Sprint 24-28)
- **Test Coverage**: 91.21% overall
  - decision_trace.rs: 100%
  - transpiler_map.rs: 100%
- **Code Quality**: 0 clippy errors, 0 warnings
- **Complexity**: All functions ≤10 (EXTREME TDD target maintained)
- **New Modules**: 2 (transpiler_map.rs, decision_trace.rs)
- **Dependencies**: +2 (rmp-serde for MessagePack, fnv for hashing)

### Sprint Accomplishments

#### Sprint 24-28: Transpiler Tracing Complete ✅
- **85-90% Feature Complete** for Renacer's scope
- **Production Ready** for local transpiler decision tracing
- **100+ Passing Tests** with comprehensive coverage
- **Zero Defects** - all quality gates passed
- **Specification Compliant** - Ruchy Tracing Support v2.0.0

### Changed

#### Dependencies
- **rmp-serde 1.3**: MessagePack serialization for decision traces
- **fnv 1.0**: Fast non-cryptographic hash for decision IDs

## [0.4.1] - 2025-11-18

### Added

#### Sprint 29: Chaos Engineering + Fuzz Testing Infrastructure

**Goal:** Add chaos engineering configuration and fuzz testing infrastructure following patterns from aprender and trueno projects

**Implementation** (EXTREME TDD - Sprint complete):
- **ChaosConfig Builder**: Pattern from aprender with gentle/aggressive presets
  - `ChaosConfig::new()` chainable builder API
  - Configurable: memory limits, CPU limits, timeouts, signal injection
  - `gentle()` and `aggressive()` preset methods for quick configuration
  - Network chaos (latency, packet loss) support prepared for Tier 2
  - Byzantine fault injection support prepared for Tier 3
- **Tiered TDD**: Makefile targets following trueno pattern
  - `make test-tier1` - Fast tests (<5s): unit + property tests
  - `make test-tier2` - Medium tests (<30s): integration tests
  - `make test-tier3` - Slow tests (<5m): fuzz + mutation tests
  - Enables rapid TDD cycles with appropriate test granularity
- **Property Tests**: 7 comprehensive tests for chaos module validation
  - Builder pattern correctness (chaining, immutability)
  - Preset validation (gentle/aggressive configurations)
  - Configuration constraints (valid ranges for limits)
  - All tests use proptest for property-based validation
- **Fuzz Infrastructure**: cargo-fuzz with filter_parser target
  - `fuzz/fuzz_targets/filter_parser.rs` - Tests SyscallFilter::from_expr()
  - Discovers edge cases in filter expression parsing
  - Integrated into Makefile tier3 target
  - Runs with libfuzzer-sys for coverage-guided fuzzing
- **Cargo Features**: Tiered chaos engineering capabilities
  - `chaos-basic` - Fast chaos (resource limits, signal injection)
  - `chaos-network` - Network/IO chaos (latency, packet loss simulation)
  - `chaos-byzantine` - Byzantine fault injection (syscall return modification)
  - `chaos-full` - Complete chaos suite with loom + arbitrary dependencies
  - `fuzz` - Fuzz testing support with arbitrary crate

**Architecture:**
- `src/chaos.rs` - ChaosConfig builder with extensive documentation
- `fuzz/fuzz_targets/filter_parser.rs` - Filter expression fuzzing
- `Cargo.toml` - Feature gates for progressive chaos capabilities
- `Makefile` - Tiered test targets for TDD workflow

**Results:**
- **Tests**: 240+ tests passing
  - 7 new property tests for chaos module
  - All existing tests passing with new infrastructure
- **Complexity**: All functions ≤10 (max: 5 in main.rs) ✅
- **Clippy**: Zero warnings ✅
- **TDG Score**: 95.1/100 (A+ grade)

**Examples:**
```bash
# Use gentle chaos preset for testing error handling
let config = ChaosConfig::gentle();

# Use aggressive chaos for stress testing
let config = ChaosConfig::aggressive();

# Custom chaos configuration
let config = ChaosConfig::new()
    .with_memory_limit(100 * 1024 * 1024)  // 100MB
    .with_cpu_limit(0.5)  // 50% CPU
    .with_timeout(Duration::from_secs(30))
    .with_signal_injection(true)
    .build();

# Run tiered tests
make test-tier1  # Fast unit tests (<5s)
make test-tier2  # Integration tests (<30s)
make test-tier3  # Fuzz + mutation tests (<5m)

# Run fuzz testing
make fuzz
```

**CLI Integration (Future):**
```bash
# Planned for future sprints
renacer --chaos gentle -- ./app
renacer --chaos aggressive -- ./flaky-test
renacer --chaos custom:chaos.json -- ./stress-test
```

### Fixed
- **Flaky Test**: test_realtime_anomaly_detects_slow_syscall timing increased from 10ms to 50ms for deterministic behavior under CPU contention (tests/sprint20_realtime_anomaly_tests.rs:29)
- **Complexity Violation**: Reduced main.rs complexity from 27 to 5 by extracting helper functions (STOP THE LINE fix per EXTREME TDD)
  - Extracted: `print_function_mappings()`, `print_stack_trace_mappings()`, `print_error_correlation_mappings()`, `run_tracer()`
- **Doctest**: Added missing `use std::time::Duration;` import to chaos.rs doctest example

### Quality Metrics (v0.4.1)

- **TDG Score**: 95.1/100 (A+ grade)
- **Tests**: 240+ total tests
  - 7 new property tests for chaos module
  - All integration and unit tests passing
- **Test Coverage**: 91.21% overall line coverage maintained
- **Code Quality**: 0 clippy errors, 0 warnings
- **Complexity**: All functions ≤10 (EXTREME TDD target achieved)
- **New Modules**: 1 (src/chaos.rs with 7 property tests)

### Sprint Accomplishments

#### Sprint 29: Chaos Engineering Foundation ✅
- **Pattern Integration**: Successfully integrated aprender (builder pattern) and trueno (tiered TDD) patterns
- **Fuzz Infrastructure**: Complete cargo-fuzz setup with filter_parser target
- **Quality Gates**: Zero defects, all tests passing, complexity targets met
- **Installation**: Locally installed with `cargo install --path . --force`

## [0.4.0] - 2025-11-18

### Added

#### Sprint 28: Decy (C→Rust) Transpiler Integration

**Goal:** Add support for Decy C-to-Rust transpiler source maps (Phase 5 of transpiler source mapping)

**Implementation** (EXTREME TDD - All tests pass immediately):
- **Tests Created**: 11 new tests (10 integration + 1 unit)
- **Result**: All tests pass - existing generic implementation already supports C source language

**Features:**
- **C Source Language Support**: `source_language: "c"` fully supported in transpiler maps
- **Decy Temp Variables**: Support for `_decy_temp_N` temporary variable mappings
- **Header File Support**: `.h` files can be used as source files
- **Full Feature Integration**: Works with all existing transpiler features:
  - `--function-time` for function profiling with C source correlation
  - `--rewrite-stacktrace` for C line number mapping
  - `--rewrite-errors` for C error correlation
  - `--show-transpiler-context` for verbose C context display

**Results:**
- **Tests**: 11 new tests (10 integration + 1 unit)
  - test_c_source_language_accepted
  - test_c_source_with_function_time
  - test_c_source_with_line_mappings
  - test_c_source_with_context
  - test_c_source_with_rewrite_errors
  - test_c_source_with_statistics
  - test_c_decy_temp_variables
  - test_c_source_all_flags
  - test_c_source_empty_mappings
  - test_c_header_file_source
  - test_c_source_language_decy (unit test)
- **Complexity**: All functions ≤10 ✅
- **Clippy**: Zero warnings ✅

**Examples:**
```bash
# Load C→Rust source map from Decy
renacer --transpiler-map algorithm.sourcemap.json -- ./algorithm_rs

# C source with function profiling
renacer --transpiler-map calc.sourcemap.json --function-time -- ./calc_rs

# C source with all transpiler features
renacer --transpiler-map app.sourcemap.json --function-time --rewrite-stacktrace --rewrite-errors -- ./app_rs
```

**Source Map Format (C→Rust):**
```json
{
  "version": 1,
  "source_language": "c",
  "source_file": "algorithm.c",
  "generated_file": "algorithm.rs",
  "mappings": [...],
  "function_map": {
    "_decy_temp_0": "temporary: sizeof(struct data)",
    "sort_array": "sort_array"
  }
}
```

#### Sprint 21: HPU Acceleration Foundation

**Goal:** Extend Trueno integration to support accelerated analysis for large-scale correlation analysis and clustering

**Implementation** (EXTREME TDD - RED → GREEN → REFACTOR cycle):
- **RED Phase**: Created 13 integration tests (tests/sprint21_hpu_acceleration_tests.rs)
- **GREEN Phase**: Implemented HPU module with correlation and clustering
- **REFACTOR Phase**: Unit tests, documentation updates

**Features:**
- **HPU Analysis Mode**: `--hpu-analysis` flag for opt-in acceleration
  - Correlation matrix computation for syscall patterns
  - K-means clustering for hotspot identification
  - Adaptive backend selection (GPU when available, CPU fallback)
- **CPU Fallback**: `--hpu-cpu-only` flag to force CPU backend
  - Useful for systems without GPU support
  - Zero overhead when HPU features disabled
- **Analysis Report**:
  - Correlation matrix showing syscall pattern relationships
  - K-means clusters grouping related syscalls
  - Backend information and computation timing

**Architecture:**
- `HPUProfiler`: Main profiler struct with adaptive backend
- `HPUBackend`: GPU/CPU backend enum
- `CorrelationResult`: Matrix of syscall correlations
- `ClusteringResult`: K-means clustering output
- `HPUAnalysisReport`: Complete analysis report

**Results:**
- **Tests**: 20 new tests (13 integration + 7 unit)
  - test_hpu_analysis_basic
  - test_hpu_correlation_matrix
  - test_hpu_kmeans_clustering
  - test_hpu_performance_threshold
  - test_hpu_fallback_to_cpu
  - test_hpu_with_statistics
  - test_hpu_with_filtering
  - test_hpu_with_function_time
  - test_hpu_json_export
  - test_hpu_large_trace
  - test_hpu_empty_trace
  - test_hpu_hotspot_identification
  - test_backward_compatibility_without_hpu
  - 7 unit tests in src/hpu.rs
- **Complexity**: All functions ≤10 ✅
- **Clippy**: Zero warnings ✅

**Examples:**
```bash
# Enable HPU analysis with statistics
renacer -c --hpu-analysis -- ./app

# Force CPU backend
renacer -c --hpu-analysis --hpu-cpu-only -- ./app

# HPU with filtering
renacer -c --hpu-analysis -e trace=file -- ls
```

**CLI Flags:**
- `--hpu-analysis`: Enable HPU-accelerated correlation and clustering analysis
- `--hpu-cpu-only`: Force CPU backend (disable GPU detection)

### Fixed
- Fixed DWARF test overflow issue with addr2line library (use bounded address values)

## [0.3.0] - 2025-11-17

### Added

#### Sprint 19: Enhanced Statistics with Trueno SIMD Integration

**Goal:** Advanced statistical analysis with percentiles and post-hoc anomaly detection using SIMD-accelerated computations

**Implementation** (EXTREME TDD - RED → GREEN → REFACTOR cycle):
- **RED Phase**: Created 9 integration tests (tests/sprint19_enhanced_stats_tests.rs)
- **GREEN Phase**: Integrated Trueno library for SIMD-accelerated statistics
- **REFACTOR Phase**: Optimized memory allocations and formatting

**Features:**
- **Percentile Analysis**: P50, P75, P90, P95, P99 latency percentiles
  - SIMD-accelerated median and percentile calculations via `Vector::percentile()`
  - 3-10x faster than standard statistical libraries on large datasets
  - Memory-efficient implementation with minimal allocations
- **Post-Hoc Anomaly Detection**: Z-score based outlier identification
  - Configurable threshold via `--anomaly-threshold SIGMA` (default: 3.0)
  - Identifies syscalls that deviate significantly from baseline
  - Useful for performance regression detection and debugging
- **Extended Statistics Mode**: `--stats-extended` flag (requires `-c`)
  - Displays percentiles table after standard statistics
  - Shows anomaly detection results if threshold exceeded
  - Zero overhead when disabled
- **Trueno Vector Operations**:
  - `Vector::mean()` for average calculations (SIMD-accelerated)
  - `Vector::stddev()` for standard deviation (SIMD-accelerated)
  - `Vector::percentile()` for P50/P75/P90/P95/P99 (SIMD-accelerated)
  - Auto-dispatches to best available backend (AVX2/AVX/SSE2/NEON/Scalar)

**Results:**
- **Tests**: 10 new tests (9 integration + 1 unit)
  - test_stats_extended_shows_percentiles
  - test_stats_extended_requires_statistics_mode
  - test_stats_extended_with_filter
  - test_anomaly_detection_identifies_outliers
  - test_anomaly_threshold_configurable
  - test_stats_extended_with_json_output
  - test_stats_extended_with_multiple_syscalls
  - test_stats_extended_backward_compatibility
  - test_stats_extended_edge_cases
  - test_stats_tracker_percentile_calculation (unit test)
- **Complexity**: All functions ≤10 ✅
- **Clippy**: Zero warnings ✅
- **Performance**: 3-10x faster statistics on large traces (10K+ syscalls)

**Examples:**
```bash
# Show percentile analysis
renacer -c --stats-extended -- cargo build

# Custom anomaly threshold (2.5 standard deviations)
renacer -c --stats-extended --anomaly-threshold 2.5 -- ./slow-app

# Extended stats with filtering
renacer -c --stats-extended -e trace=file -- find /usr
```

**CLI Flags:**
- `--stats-extended`: Enable percentile analysis and anomaly detection (requires `-c`)
- `--anomaly-threshold SIGMA`: Set Z-score threshold for anomalies (default: 3.0)

#### Sprint 20: Real-Time Anomaly Detection with Sliding Windows

**Goal:** Real-time anomaly detection using sliding window statistics and per-syscall baselines

**Implementation** (EXTREME TDD - RED → GREEN cycle):
- **RED Phase**: Created 13 integration tests (tests/sprint20_realtime_anomaly_tests.rs)
- **GREEN Phase**: Implemented real-time anomaly detector with SIMD-accelerated statistics

**Architecture:**
- **Core Module**: `src/anomaly.rs` (369 lines, 10 unit tests)
  - `AnomalyDetector` struct with sliding window baseline tracking
  - `Anomaly` struct with Z-score, severity, and metadata
  - `AnomalySeverity` enum (Low: 3-4σ, Medium: 4-5σ, High: >5σ)
  - `BaselineStats` struct with per-syscall sliding window statistics
- **Per-Syscall Baselines**: Independent sliding windows for each syscall type
  - HashMap-based tracking (`HashMap<String, BaselineStats>`)
  - Configurable window size (default: 100 samples per syscall)
  - Minimum 10 samples required for reliable anomaly detection
- **SIMD-Accelerated Statistics**: Trueno Vector operations for real-time performance
  - `Vector::mean()` for baseline mean calculation
  - `Vector::stddev()` for baseline standard deviation
  - Z-score calculation: `(duration - mean) / stddev`
- **Sliding Window**: Vec-based circular buffer
  - Removes oldest sample when window size exceeded
  - Updates mean and stddev after each sample
  - Memory-efficient with pre-allocated capacity

**Features:**
- **Real-Time Detection**: Anomalies detected and reported during tracing
  - Alerts printed to stderr immediately when detected
  - Non-intrusive to stdout syscall trace output
  - Format: `⚠️  ANOMALY: {syscall} took {duration} μs ({z_score}σ from baseline {mean} μs) - {severity}`
- **Severity Classification**:
  - 🟢 Low: 3.0-4.0 standard deviations from mean
  - 🟡 Medium: 4.0-5.0 standard deviations from mean
  - 🔴 High: >5.0 standard deviations from mean
- **Summary Report**: Anomaly detection summary printed at end
  - Total anomaly count
  - Severity distribution breakdown
  - Top 10 most severe anomalies (sorted by Z-score)
  - Baseline statistics (mean ± stddev) for each anomaly
- **Integration**: Works seamlessly with existing features
  - Compatible with `-c` (statistics mode)
  - Compatible with `-e trace=` (filtering)
  - Compatible with `-f` (multi-process tracing)
  - Compatible with `--format json` (anomalies exported in JSON)
  - Compatible with `--source` and `--function-time` flags
- **Backward Compatibility**: Optional feature, zero overhead when disabled

**Results:**
- **Tests**: 23 new tests (13 integration + 10 unit)
  - **Integration Tests** (tests/sprint20_realtime_anomaly_tests.rs):
    - test_realtime_anomaly_detects_slow_syscall
    - test_anomaly_window_size_configuration
    - test_anomaly_requires_minimum_samples
    - test_anomaly_severity_classification
    - test_anomaly_realtime_with_statistics
    - test_anomaly_realtime_with_filtering
    - test_anomaly_realtime_with_multiprocess
    - test_anomaly_json_export
    - test_anomaly_with_zero_variance
    - test_anomaly_sliding_window_wraparound
    - test_backward_compatibility_without_anomaly_realtime
    - test_anomaly_threshold_from_sprint19_still_works
    - test_anomaly_realtime_full_integration
  - **Unit Tests** (src/anomaly.rs):
    - test_anomaly_detector_creation
    - test_baseline_stats_insufficient_samples
    - test_anomaly_detection_slow_syscall
    - test_severity_classification
    - test_sliding_window_removes_old_samples
    - test_per_syscall_baselines
    - test_anomaly_with_zero_variance
    - test_get_anomalies_stores_history
    - And 2 more edge case tests
- **Coverage**: 100% coverage on anomaly.rs module
- **Complexity**: All functions ≤10 ✅
- **Clippy**: Zero warnings ✅

**CLI Flags:**
- `--anomaly-realtime`: Enable real-time anomaly detection
- `--anomaly-window-size SIZE`: Set sliding window size (default: 100)

**Examples:**
```bash
# Real-time anomaly detection
renacer --anomaly-realtime -- ./app

# Custom window size (track last 200 samples per syscall)
renacer --anomaly-realtime --anomaly-window-size 200 -- ./app

# Combined with statistics mode
renacer -c --anomaly-realtime -- cargo test

# Custom threshold and real-time detection
renacer --anomaly-realtime --anomaly-threshold 2.5 -- ./flaky-app

# With filtering (only monitor file operations)
renacer --anomaly-realtime -e trace=file -- find /usr

# Multi-process anomaly detection
renacer -f --anomaly-realtime -- make -j8

# JSON export with anomalies
renacer --anomaly-realtime --format json -- ./app > trace.json
```

**Output Format:**
```
Real-time alert (stderr):
⚠️  ANOMALY: write took 5234 μs (4.2σ from baseline 102.3 μs) - 🟡 Medium

Summary report (end of trace):
=== Real-Time Anomaly Detection Report ===
Total anomalies detected: 12

Severity Distribution:
  🔴 High (>5.0σ):   2 anomalies
  🟡 Medium (4-5σ): 5 anomalies
  🟢 Low (3-4σ):    5 anomalies

Top Anomalies (by Z-score):
  1. 🔴 fsync - 6.3σ (8234 μs, baseline: 123.4 ± 1287.2 μs)
  2. 🔴 write - 5.7σ (5234 μs, baseline: 102.3 ± 902.1 μs)
  3. 🟡 read - 4.8σ (2341 μs, baseline: 87.6 ± 468.9 μs)
  ... and 9 more
```

### Changed

#### Performance
- **SIMD-Accelerated Statistics**: Trueno integration for 3-10x faster statistical computations
  - Mean, standard deviation, percentile calculations use SIMD when available
  - Auto-dispatch to best backend (AVX2/AVX/SSE2/NEON/Scalar)
  - Benefits large trace sessions (10K+ syscalls)

#### Dependencies
- **Trueno 0.1.0**: SIMD/GPU compute library for high-performance statistics
  - Provides Vector operations with hardware acceleration
  - Zero-cost abstraction when SIMD not available (falls back to scalar)

### Quality Metrics (v0.3.0)

- **TDG Score**: 94.5/100 (A grade)
- **Tests**: 267 total tests
  - 33 new tests for Sprint 19-20
  - 13 integration tests (sprint20_realtime_anomaly_tests.rs)
  - 9 integration tests (sprint19_enhanced_stats_tests.rs)
  - 10 unit tests (src/anomaly.rs)
  - 1 unit test (src/stats.rs percentile)
- **Test Coverage**: 91.21% overall line coverage
  - anomaly.rs: 100%
  - stats.rs: 97.99%
  - All modules maintain >90% coverage
- **Code Quality**: 0 clippy errors, 0 warnings
- **Complexity**: All functions ≤10 (EXTREME TDD target)
- **New Modules**: 1 (src/anomaly.rs - 369 lines)

### Sprint Accomplishments

#### Sprint 19-20: Trueno Integration Milestone Complete ✅
- **Sprint 19**: Enhanced Statistics with SIMD-accelerated percentiles and post-hoc anomaly detection
- **Sprint 20**: Real-Time Anomaly Detection with sliding window baselines
- **Total**: 33 new tests, 369 lines of new code, 100% coverage on new modules
- **Performance**: 3-10x faster statistics on large traces
- **Quality**: Zero defects, all tests passing, zero warnings

## [0.2.0] - 2025-11-17

### Added

#### Property-Based Testing Infrastructure (Commit ae62097)
- **Comprehensive test suite**: 18 property-based tests covering all core features
- **670+ test cases** executed via proptest in <6 seconds
- **Library interface** (src/lib.rs) exposing all modules for testing
- **Pre-commit hook** with 5 quality gates:
  1. Format check (cargo fmt)
  2. Clippy check (zero warnings)
  3. Bash/Makefile quality (bashrs lint)
  4. Property-based comprehensive tests (30s timeout)
  5. Security audit (cargo audit)
- **Test Coverage:**
  - Syscall name resolution (100 cases)
  - Filter/trace spec parsing (100 cases)
  - Stats tracker, function profiler, JSON serialization (50+ cases each)
  - Call graph tracking, I/O detection, hot path analysis
  - End-to-end integration tests
  - DWARF source location invariants
  - Trueno Vector integration
- Total: 142 tests (124 unit + 18 property-based)

#### Quality Infrastructure (Commit 10157f3)
- **cargo fmt** applied to all 20 source files
- **deny.toml** configuration for dependency policy:
  - Security: Denies vulnerabilities, warns on unmaintained deps
  - Licensing: MIT, Apache-2.0, BSD licenses allowed
  - Sources: Restricts to crates.io and paiml GitHub org
- **Zero defects** enforced via pre-commit hooks

#### Function-Level Profiling (GitHub Issue #1 - Complete!)

**Complete implementation of function-level profiling with 4 major features:**

1. **I/O Bottleneck Detection** (Commit 000cd50)
   - Automatic detection of slow I/O operations (>1ms threshold)
   - Tracks 16 I/O syscall types: `read`, `write`, `readv`, `writev`, `pread64`, `pwrite64`, `openat`, `open`, `close`, `fsync`, `fdatasync`, `sync`, `sendfile`, `splice`, `tee`, `vmsplice`
   - Visual warnings (⚠️) in output for functions with slow I/O
   - Helps identify performance bottlenecks in I/O-heavy code
   - 8 comprehensive unit tests, 100% coverage

2. **Call Graph Tracking** (Commit 4527919)
   - Tracks parent→child function relationships via stack unwinding
   - Shows which functions call which other functions
   - Visual tree display of call graphs in profiling output
   - Aggregates call frequencies for each relationship
   - 6 comprehensive unit tests

3. **Hot Path Analysis** (Commit 81a8e22)
   - Identifies top 10 most time-consuming functions
   - Shows percentage of total execution time per function
   - Integrated with call graph display (top 5 callees per hot function)
   - Helps prioritize optimization efforts
   - 5 comprehensive unit tests

4. **Flamegraph Export** (Commit 88b1a67)
   - Exports profiling data in folded stack format
   - Compatible with standard flamegraph tools: `flamegraph.pl`, `inferno`, `speedscope`
   - Supports nested call graphs and multi-level stack traces
   - Format: `func1;func2;func3 count`
   - Public API: `profiler.export_flamegraph(&mut file)?`
   - 10 comprehensive unit tests

**Stack Unwinding Infrastructure** (Commit 078cfd8)
- Manual stack unwinding via frame pointer chain (RBP)
- Remote process memory reading via `process_vm_readv`
- Filters out libc functions to identify user code
- Max depth protection (64 frames) prevents infinite loops
- 6 unit tests + 5 integration tests
- Coverage: 98.88% (up from 22.64%)

**Integration & CLI:**
- All features work together seamlessly
- Activated with `--function-time --source` flags
- Output includes: timing summary, hot paths, call graphs, I/O analysis
- Zero runtime overhead when disabled

### Changed

#### Dependencies
- **Trueno Integration** (Commit 7270fa8)
  - Migrated from local path dependency to published crates.io version
  - Now using `trueno = "0.1.0"` from crates.io
  - Makes renacer more portable and easier to build
  - SIMD-accelerated statistics via Trueno Vector operations

#### Code Quality
- **Clippy Compliance** (Commit c5b4c69)
  - Fixed all clippy warnings for v0.2.0 release
  - Suppressed assert_cmd deprecation in tests (11 test files)
  - Fixed needless borrows in test argument passing
  - Added allow annotation for constant assertions in tests
  - Zero clippy errors with `-D warnings`

#### Performance
- **5-9% Performance Improvement** (Commit 783eeb8)
  - Lazy formatting: only format syscall output when needed
  - Reduced allocations in hot paths
  - String building optimizations
  - Maintains >90% test coverage

### Quality Metrics (v0.2.0)
- **TDG Score**: 94.2/100 (A grade)
- **Tests**: 124 unit tests (29 new tests for Sprint 13-14 Phase 2)
  - 35 function_profiler tests
  - 8 stack_unwind tests
  - All integration tests passing
- **Coverage**: 91.21% overall
  - function_profiler.rs: 100%
  - stack_unwind.rs: 98.88%
  - filter.rs: 100%
  - cli.rs: 100%
  - syscalls.rs: 99.38%
- **Code Quality**: 0 clippy errors, 0 warnings
- **Dependencies**: Trueno 0.1.0 from crates.io

### Sprint Accomplishments

#### Sprint 13-14 Phase 2: Advanced Function Profiling
- **GitHub Issue #1**: Fully complete ✅
  - All 4 planned features implemented
  - 29 new comprehensive tests
  - 100% coverage on profiling modules
  - Production-ready with full documentation

## [0.1.0] - 2025-11-16

### Added

#### Core Features
- **System Call Tracing**: Intercept and display all system calls from traced programs
- **Syscall Name Resolution**: Map syscall numbers to names for x86_64 (80+ syscalls)
- **Argument Decoding**: Decode common arguments including:
  - File paths in `openat()` calls
  - File descriptors, buffer addresses, counts
- **Exit Code Preservation**: Traced program's exit code is properly propagated
- **CLI Interface**: Clean command-line interface with `renacer -- COMMAND` syntax

#### Performance
- **1.1-1.3x Faster Than strace**: Formal benchmarks show consistent performance advantage
  - ls workload: 1.12x faster (137ms vs 154ms)
  - find workload: 1.09x faster (680ms vs 739ms)
  - echo workload: 1.28x faster (4.1ms vs 5.3ms)
- **Efficient Memory Operations**: Uses `process_vm_readv` for direct memory reads
- **Native Compilation**: Rust with LTO optimizations for maximum performance
- **Room for Optimization**: Target is 2-5x faster (roadmap Sprint 11-12)

#### Infrastructure
- **DWARF Support (Planned)**: `--source` flag infrastructure for future source correlation
- **Comprehensive Test Suite**:
  - 16 integration tests across 3 sprint phases
  - Unit tests for core modules
  - 100% pass rate on core functionality
- **Quality Tooling**: Integrated with paiml-mcp-agent-toolkit for TDG scoring

### Performance Benchmarks (v0.1.0 - Informal)

```
Command: ls -laR /usr/bin | head -1000 (average of 5 runs)
- Baseline (no tracing):  13ms
- strace:                163ms (1253% overhead)
- renacer:                20ms  (153% overhead)

Result: renacer is 8.15x FASTER than strace
NOTE: This was an early informal benchmark. See v0.2.0 for formal benchmarks.
```

### Quality Metrics

- **Test Coverage**: 100% of core features tested
- **TDG Score**: 94.6/100 (A grade)
- **Clippy Warnings**: 0 (excluding external crate deprecations)
- **Zero Regressions**: All tests maintained throughout development

### Architecture

- **Language**: Pure Rust (edition 2021)
- **Tracing Method**: ptrace system calls
- **Platform**: Linux x86_64
- **Dependencies**: Minimal - nix, clap, anyhow, thiserror

### Development Methodology

Built using EXTREME TDD following Toyota Way principles:
- **Jidoka** (Built-in Quality): RED → GREEN → REFACTOR cycle
- **Kaizen** (Continuous Improvement): Iterative 2-week sprints
- **Genchi Genbutsu** (Go and See): Data-driven benchmarking
- **Andon Cord** (Stop the Line): Quality gates block bad code

### Known Limitations

- **x86_64 Only**: aarch64 support planned for future release
- **Source Correlation Partial**: `--source` flag loads DWARF debug info, but syscall attribution requires stack unwinding (deferred to v0.2.0)
  - DWARF .debug_line parsing: ✅ Implemented with addr2line crate
  - Binary debug info loading: ✅ Implemented
  - Syscall-to-source attribution: ⚠️ Requires stack unwinding (syscalls happen in libc, not user code)
  - Planned for v0.2.0: Full call stack unwinding to attribute syscalls to user code frames
- **Basic Argument Decoding**: Currently supports filenames; advanced decoding (buffers, structures) planned

### Future Roadmap

See `roadmap.yaml` for detailed implementation plan:

**v0.2.0** (Sprint 5-6 completion):
- Full DWARF source correlation
- Map syscalls to source file:line
- Function name attribution

**v0.3.0** (Sprint 7-8):
- Multi-architecture support (aarch64)
- Cross-platform testing with QEMU

**v1.0.0** (Sprint 9-12):
- strace feature parity (`-p`, `-f`, `-e trace=`, `-c`, `-T`)
- JSON output format
- Advanced filtering

### Contributors

- Primary Development: Claude Code (Anthropic) with EXTREME TDD
- Methodology: paiml-mcp-agent-toolkit quality enforcement
- Specification: Toyota Way expert review

---

## [Unreleased]

### Added (Post-v0.1.0)

#### Sprint 9-10: Advanced Filtering, Statistics, Timing, JSON & PID Attach
- **Syscall Filtering**: `-e trace=EXPR` flag for filtering syscalls
  - Individual syscalls: `-e trace=open,read,write`
  - Syscall classes: `-e trace=file`, `-e trace=network`, `-e trace=process`, `-e trace=memory`
  - Mixed mode: `-e trace=file,socket,brk`
  - Hash-based filter implementation with O(1) lookup
- **Filter Module**: Robust parsing and evaluation of filter expressions (src/filter.rs)
- **Statistics Mode**: `-c` flag for syscall summary (strace-compatible)
  - Per-syscall call counts and error counts
  - Percentage distribution with timing data
  - Summary table with totals (% time, seconds, usecs/call columns)
  - Compatible with filtering
- **Per-Syscall Timing**: `-T` flag for syscall duration tracking
  - Displays time in `<seconds>` format after each syscall
  - Integrated with statistics mode (% time, seconds, usecs/call columns)
  - Zero overhead when disabled
- **JSON Output**: `--format json` for machine-parseable output
  - Structured renacer-json-v1 schema with syscalls and summary
  - Compatible with filtering, timing, and source correlation
  - Ideal for tooling integration and analysis pipelines
  - Full serde serialization support
- **PID Attach**: `-p PID` flag for attaching to running processes
  - Uses PTRACE_ATTACH instead of fork + PTRACE_TRACEME
  - Mutually exclusive with command tracing
  - Proper error handling for non-existent PIDs
  - Shares same tracing infrastructure as command mode
- **Fork Following Infrastructure**: `-f` flag and ptrace options (PTRACE_O_TRACEFORK/VFORK/CLONE)
  - CLI flag implemented
  - Ptrace options configured
  - Full multi-process tracking deferred to v0.3.0 (requires refactoring)
- **Zero Overhead**: Filtering/statistics/timing at display time, no performance impact when disabled
- **24 Integration Tests**: Comprehensive coverage across 5 test suites
  - 6 tests for filtering (tests/sprint9_filtering_tests.rs)
  - 4 tests for statistics mode (tests/sprint9_statistics_tests.rs)
  - 4 tests for timing mode (tests/sprint9_timing_tests.rs)
  - 5 tests for JSON output (tests/sprint9_json_output_tests.rs)
  - 5 tests for PID attach (tests/sprint9_pid_attach_tests.rs)

#### Sprint 11-12: Hardening & Performance Baseline (In Progress)

**Test Coverage Achievement (91.21% - Exceeds 90% Goal):**
- ✅ **Overall Coverage**: 91.21% line coverage (exceeds 90% requirement)
- ✅ **Per-Module Coverage**:
  - cli.rs: 100%
  - main.rs: 100%
  - filter.rs: 100% (+12.22%)
  - syscalls.rs: 99.38% (+64% from 35.38%)
  - stats.rs: 97.99% (+43% from 54.93%)
  - json_output.rs: 96.39%
  - tracer.rs: 83.76%
  - dwarf.rs: 81.91%

**Mutation Testing Infrastructure:**
- ✅ **cargo-mutants** installed and configured (.cargo-mutants.toml)
- ✅ **Makefile targets**:
  - `make mutants` - Full mutation analysis
  - `make mutants-quick` - Quick check on uncommitted changes
- ✅ **Initial Results**: 66% caught rate on filter.rs (3/6 viable mutants)
- ⏳ **Full Project Mutation Scan**: Pending (long-running)

**Property-Based Testing:**
- ✅ **proptest** framework integrated
- ✅ **3 property tests** for syscalls.rs:
  - prop_syscall_name_never_panics (tests any i64)
  - prop_syscall_name_deterministic (tests 0-400 range)
  - prop_unknown_syscalls_return_unknown (tests 400-10000 range)

**Comprehensive Test Additions (45+ new tests):**
- ✅ **syscalls.rs**: 6 tests (+40+ syscall mappings validated)
- ✅ **stats.rs**: 17 tests (edge cases, large numbers, sorting, percentages)
- ✅ **filter.rs**: 8 tests (all syscall classes, whitespace, cloning)
- ✅ **dwarf.rs**: 11 tests (error handling, address ranges, equality)
- ✅ **tracer.rs**: 3 tests (SyscallEntry creation, invalid PID)

**Performance Optimizations (Profile-Guided):**
- ✅ **Profiling Analysis**: Identified 92% kernel time (ptrace), 8% user time
- ✅ **Lazy String Formatting**: Skip formatting in statistics/JSON modes
- ✅ **Reduced Allocations**: Eliminate Vec allocation in non-JSON mode hot path
- ✅ **Results**: 5-9% performance improvement across all benchmarks
  - echo: 1.28x → 1.33x faster (+4%)
  - ls: 1.12x → 1.22x faster (+9%)
  - find: 1.09x → 1.14x faster (+5%)

#### Sprint 13-14: Self-Profiling, Trueno Integration & Function Profiling (GitHub Issues #1, #3, #4)

**Trueno Integration for Statistical Calculations (GitHub Issue #4):**
- ✅ **Trueno Dependency**: Added sister project (../trueno) as path dependency
- ✅ **SIMD-Accelerated Statistics**: Replaced standard sum operations with Trueno Vector operations
  - `calculate_totals_with_trueno()` method for high-performance aggregations
  - Auto-dispatches to best available backend (AVX2/AVX/SSE2/NEON/Scalar)
- ✅ **Zero Functional Changes**: Same output, faster computation on large datasets
- ✅ **Sister Project Synergy**: Dogfoods Trueno within PAIML ecosystem
- ✅ **2 New Tests**: Trueno integration tests (test_trueno_sum_integration, test_stats_tracker_uses_trueno_for_sums)
- ✅ **Performance**: SIMD acceleration beneficial for large trace sessions (100K+ syscalls)

**Function-Level Profiling Infrastructure (GitHub Issue #1 - Phase 1 Complete):**
- ✅ **FunctionProfiler Module**: Created src/function_profiler.rs with timing aggregation (100% coverage)
  - FunctionStats struct for per-function timing data with extensible fields
  - FunctionProfiler::record() for attributing syscalls to functions
  - FunctionProfiler::print_summary() for formatted output
  - Reserved fields for future features: callees (call graph), io_syscalls, slow_io_count
  - 8 unit tests with edge cases (zero syscalls, sorting, averages)
- ✅ **CLI Integration**: `--function-time` flag added to CLI (src/cli.rs)
  - 2 unit tests for flag parsing
- ✅ **Tracer Integration**: Function profiler integrated into syscall loop (src/tracer.rs)
  - TracerConfig struct introduced to fix clippy "too_many_arguments" warnings
  - Refactored tracer functions to accept single config parameter
- ✅ **SyscallEntry Enhancement**: Added function_name field to track DWARF function attribution
- ✅ **Stack Unwinding**: Implemented stack unwinding for syscall attribution (src/stack_unwind.rs - 98.88% coverage)
  - Manual stack walking using frame pointer chain (RBP)
  - Remote process memory reading via process_vm_readv
  - Protection against infinite loops (MAX_STACK_DEPTH=64)
  - find_user_function_via_unwinding() to filter out libc and find user functions
  - 6 unit tests for StackFrame operations
  - 5 integration tests for stack unwinding scenarios
- ✅ **11 Integration Tests**: Comprehensive end-to-end testing
  - 5 tests in sprint13_function_profiling_tests.rs
  - 5 tests in sprint13_stack_unwinding_tests.rs
  - test_function_time_flag_accepted
  - test_function_time_output_format
  - test_function_time_with_statistics_mode
  - test_function_time_with_filter
  - test_function_time_without_flag_no_profiling
  - test_stack_frame_struct
  - test_stack_unwinding_with_simple_program
  - test_stack_unwinding_does_not_crash
  - test_stack_unwinding_with_function_time_disabled
  - test_stack_unwinding_max_depth_protection
- ✅ **Phase 1 Deliverables Complete**:
  - Basic function-level timing infrastructure
  - Stack unwinding implementation
  - DWARF integration for function name lookup
  - End-to-end testing and documentation

**Planned for Phase 2** (GitHub Issue #1 - Remaining Features):
- ⏳ **Stack Unwinding Verification**: Debug and verify stack unwinding works correctly with real binaries
- ⏳ **Call Graph Profiling**: Track parent→child function relationships
- ⏳ **Hot Path Analysis**: Identify top 10 most frequently executed code paths
- ⏳ **I/O Bottleneck Detection**: Flag slow I/O operations (>1ms threshold)
- ⏳ **Subprocess Execution Tracking**: Track syscalls across process boundaries
- ⏳ **Flamegraph Export**: Export data in flamegraph.pl compatible format for visualization

#### Sprint 13-14: Self-Profiling Infrastructure (GitHub Issue #3)

**Self-Profiling Feature (`--profile-self` flag):**
- ✅ **ProfilingContext**: Category-based timing infrastructure (src/profiling.rs)
  - 7 profiling categories: Ptrace, Formatting, MemoryRead, DwarfLookup, Statistics, JsonSerialization, Other
  - `measure<F, R>()` method for wrapping operations with timing
  - `print_summary()` outputs formatted profiling report to stderr
- ✅ **CLI Integration**: `--profile-self` flag added to CLI (src/cli.rs)
- ✅ **Tracer Integration**: Profiling instrumented into main syscall loop (src/tracer.rs)
- ✅ **10 Unit Tests**: Full test coverage for ProfilingContext (100% passing)
- ✅ **5 Integration Tests**: End-to-end testing of --profile-self flag (tests/sprint13_profiling_tests.rs)
  - test_profile_self_flag_outputs_summary
  - test_profile_self_without_flag_no_output
  - test_profile_self_with_statistics_mode
  - test_profile_self_reports_nonzero_syscalls
  - test_profile_self_with_filtering

**Profiling Output Format:**
```
╔════════════════════════════════════════════════════════════╗
║  Renacer Self-Profiling Results                           ║
╚════════════════════════════════════════════════════════════╝

Total syscalls traced:     43
Total wall time:           0.002s
  - Kernel time (ptrace):  0.001s (82.7%)
  - User time (renacer):   0.000s (17.3%)

User-space breakdown:
  - Other:               0.000s (100.0%)
```

**Sprint 11-12 Deliverables:**
- ✅ Benchmark suite vs strace (4 comprehensive benchmarks)
- ✅ 90%+ test coverage enforcement (91.21% achieved)
- ✅ Mutation testing infrastructure (cargo-mutants)
- ✅ Property-based testing infrastructure (proptest)
- ✅ Performance optimization (profile-guided, 5-9% improvement)
- ⏳ 24hr fuzz runs (pending)
- ⏳ Complete documentation (in progress)
- ⏳ crates.io publication (pending)

#### Performance Benchmarks (v0.2.0 - Formal)

Benchmark suite in `tests/benchmark_vs_strace.rs`:

```
ls -la /usr/bin (5 runs):
- Baseline: 14.4ms
- strace:   154ms (965% overhead)
- renacer:  137ms (851% overhead)
Result: 1.12x faster

find /usr/share/doc (3 runs):
- Baseline: 371ms
- strace:   739ms (99% overhead)
- renacer:  680ms (83% overhead)
Result: 1.09x faster

echo "hello" (10 runs):
- Baseline: 0.59ms
- strace:   5.31ms
- renacer:  4.14ms
Result: 1.28x faster

Filtering overhead: ~8% improvement with -e trace=open
```

**Honest Assessment** (Genchi Genbutsu):
- Current: 1.1-1.3x faster than strace
- Target: 2-5x faster (roadmap Sprint 11-12)
- Room for optimization exists

### Sprint 9-10 Status (5/6 Complete - 83%)
- ✅ Syscall filtering with `-e trace=` expressions
- ✅ Statistics mode with `-c` flag
- ✅ Per-syscall timing with `-T` flag
- ✅ JSON output with `--format json`
- ✅ PID attach with `-p PID` flag
- ⚠️  Fork following with `-f` flag (infrastructure only - full implementation deferred to v0.3.0)

### Quality Metrics (Post Sprint 13-14)
- **TDG Score**: 91.7/100 (A grade)
- **Test Suites**: 12 total (3 from v0.1.0 + 5 from Sprint 9-10 + 1 benchmark + 3 Sprint 13-14 suites)
- **Test Count**: 155 total tests (148 active + 7 ignored)
  - 97 unit tests (all module tests)
  - 51 active integration tests across 11 test suites
  - 7 ignored tests (4 benchmarks + 3 DWARF source tests)
  - **Sprint 13-14 Additions** (32 new tests):
    - Added 5 integration tests for --profile-self (sprint13_profiling_tests.rs)
    - Added 5 integration tests for --function-time (sprint13_function_profiling_tests.rs)
    - Added 5 integration tests for stack unwinding (sprint13_stack_unwinding_tests.rs)
    - Added 10 unit tests for ProfilingContext (src/profiling.rs)
    - Added 8 unit tests for FunctionProfiler (src/function_profiler.rs)
    - Added 6 unit tests for StackFrame operations (src/stack_unwind.rs)
    - Added 2 unit tests for --function-time CLI flag (src/cli.rs)
    - Added 2 unit tests for Trueno integration (src/stats.rs)
- **Test Coverage**: 91.21% overall line coverage (exceeds 90% goal)
  - function_profiler.rs: 100%
  - stack_unwind.rs: 98.88% (up from 22.64%)
  - filter.rs: 100%
  - cli.rs: 100%
  - syscalls.rs: 99.38%
  - stats.rs: 96.28%
  - Below 90%: dwarf.rs (81.91%), tracer.rs (83.33%)
- **Mutation Testing**: 66% caught rate (filter.rs baseline)
- **Property-Based Tests**: 3 property tests with proptest
- **Code Quality**: 0 clippy warnings (fixed "too_many_arguments" with TracerConfig refactoring)
- **New Modules**: 6 (filter.rs, stats.rs, json_output.rs, profiling.rs, function_profiler.rs, stack_unwind.rs)
- **Dependencies**: 2 (backtrace for stack unwinding, Trueno for SIMD compute)
- **Zero Regressions**: All 155 tests passing

### Planned for 0.2.0
- ✅ DWARF .debug_line parsing using addr2line crate (COMPLETED in v0.1.0)
- ✅ `--source` flag infrastructure (COMPLETED in v0.1.0)
- ✅ Basic syscall filtering (COMPLETED post-v0.1.0)
- ✅ `-c` statistics mode (COMPLETED post-v0.1.0)
- ✅ `-T` timing mode (COMPLETED post-v0.1.0)
- ✅ `--format json` JSON output (COMPLETED post-v0.1.0)
- ✅ `-p PID` attach to running process (COMPLETED post-v0.1.0)
- Stack unwinding to attribute syscalls to user code frames
- Source-aware output showing file:line for each syscall (requires stack unwinding)
- Function name attribution from DWARF .debug_info (requires stack unwinding)

#### Sprint 15: Advanced Filtering - Negation Operator (2025-11-17)

**Goal:** Extend filtering with negation operator for excluding syscalls

**Implementation** (EXTREME TDD - RED → GREEN → REFACTOR):
- **RED Phase**: Created 7 integration tests (tests/sprint15_negation_tests.rs)
- **GREEN Phase**: Added `exclude: HashSet<String>` to SyscallFilter
- **REFACTOR Phase**: Extracted validate_trace_spec() and parse_syscall_sets()

**Features:**
- `-e trace=!close` - Exclude specific syscalls
- `-e trace=!file` - Exclude syscall classes
- `-e trace=file,!close` - Mixed inclusion + exclusion
- Exclusions have highest priority (checked first)

**Results:**
- **Tests**: 178 total (16 new - 7 integration + 9 unit)
- **Complexity**: All functions ≤10 (max: 8) ✅
- **Clippy**: Zero warnings ✅
- **TDG Score**: 94.5/100 maintained

**Examples:**
```bash
renacer -e trace=!close -- ls               # All syscalls except close
renacer -e trace=!file -- curl example.com  # All except file operations
renacer -e trace=file,!close -- cat file    # File operations except close
```

#### Sprint 16: Advanced Filtering - Regex Patterns (2025-11-17)

**Goal:** Add regex pattern matching for powerful syscall selection

**Implementation** (EXTREME TDD - RED → GREEN → REFACTOR):
- **RED Phase**: Created 9 integration tests (tests/sprint16_regex_filtering_tests.rs)
- **GREEN Phase**: Added `include_regex` and `exclude_regex` fields to SyscallFilter
- **REFACTOR Phase**: Extracted parse_regex_pattern(), created ParseResult type alias

**Features:**
- `/pattern/` syntax for regex patterns
- Support for prefix, suffix, OR patterns
- Case-insensitive matching with `(?i)` flag
- Mixed regex + literals + negation
- Proper error handling for invalid regex

**Results:**
- **Tests**: 201 total (23 new - 9 integration + 14 unit)
- **Complexity**: All functions ≤10 (max: 8) ✅
- **Clippy**: Zero warnings ✅
- **Coverage**: 93.73% overall (filter.rs: 98.76%)
- **TDG Score**: 94.5/100 maintained

**Examples:**
```bash
renacer -e 'trace=/^open.*/' -- ls          # All syscalls starting with "open"
renacer -e 'trace=/.*at$/' -- cat file      # All syscalls ending with "at"
renacer -e 'trace=/read|write/' -- app      # Syscalls matching read OR write
renacer -e 'trace=/^open.*/,!/openat/' -- ls  # open* except openat
renacer -e 'trace=/(?i)OPEN/' -- ls         # Case-insensitive matching
```

#### Sprint 17: Output Format Improvements - CSV Export (2025-11-17)

**Goal:** Add CSV output format for spreadsheet analysis and data processing

**Implementation** (EXTREME TDD - RED → GREEN cycle):
- **RED Phase**: Created 12 integration tests (tests/sprint17_output_format_tests.rs)
- **GREEN Phase**: Implemented CSV output module (src/csv_output.rs)

**Features:**
- `--format csv` - Export syscall traces in CSV format
- Dynamic columns based on active flags:
  - Basic: `syscall,arguments,result`
  - With `-T`: `syscall,arguments,result,duration`
  - With `--source`: `syscall,arguments,result,source_location`
  - Combined: `syscall,arguments,result,duration,source_location`
- Statistics mode (`-c`): `syscall,calls,errors` (or `syscall,calls,errors,total_time` with `-T`)
- Proper CSV escaping for commas, quotes, and newlines
- Compatible with filtering, timing, source correlation

**Results:**
- **Tests**: 230+ total (29 new - 12 integration + 18 unit tests in csv_output.rs)
- **Coverage**: 18 comprehensive unit tests for CSV formatting edge cases
- **Clippy**: Zero warnings ✅
- **TDG Score**: 94.5/100 maintained

**Examples:**
```bash
# Basic CSV output
renacer --format csv -- echo "test" > trace.csv

# CSV with timing information
renacer --format csv -T -- ls > trace-with-timing.csv

# CSV with source correlation
renacer --format csv --source -- ./my-binary > trace-with-source.csv

# CSV statistics summary
renacer --format csv -c -- cargo build > stats.csv

# CSV with filtering
renacer --format csv -e trace=file -- cat file.txt > file-ops.csv
```

**JSON Enhancement Verification:**
- Verified existing JSON output already supports all required features (from Sprint 9-10)
- JSON includes version, format, syscalls array, and summary fields
- Compatible with timing (-T), source correlation (--source), and statistics mode (-c)

#### Sprint 18: Multi-Process Tracing - Fork Following (2025-11-17)

**Goal:** Implement full multi-process tracing with `-f` flag to follow fork/vfork/clone syscalls

**Implementation** (EXTREME TDD - RED → GREEN → REFACTOR cycle):
- **RED Phase**: Created 13 integration tests (tests/sprint18_multiprocess_tests.rs)
  - Basic fork following, fork + exec, default behavior (no -f)
  - Multiple concurrent forks, vfork, clone/threads
  - Integration with filtering, statistics, JSON, CSV output formats
  - Edge cases: immediate exit, quick fork/exit races
- **GREEN Phase**: Implemented multi-process tracking (src/tracer.rs)
  - HashMap<Pid, ProcessState> for per-process state management
  - Event-driven ptrace handling (PTRACE_EVENT_FORK/VFORK/CLONE)
  - waitpid(-1) for any child process monitoring
  - Per-process syscall tracking, DWARF context, timing
- **REFACTOR Phase**: Complexity reduction and bug fixes
  - Initial implementation: complexity 17 (ANDON CORD pulled)
  - Extracted helper functions:
    - `handle_traced_process_status()` - complexity 7
    - `process_syscall_for_pid()` - complexity 5
  - Final `trace_child()` complexity: 9 ✅
  - Critical bug fix: Added child process continuation in `handle_ptrace_event()`

**Architecture:**
- **ProcessState struct**: Encapsulates per-process state (syscall tracking, DWARF context, timing)
- **Multi-PID HashMap**: Tracks all active processes, removes on exit
- **Ptrace Events**: Intercepts fork/vfork/clone, automatically attaches to new children
- **Process Lifecycle**: Tracks main PID exit code, continues tracing children until all exit

**Features:**
- `-f` flag: Follow fork(), vfork(), clone() syscalls
- Automatic child process attachment and tracing
- Per-process state isolation
- Compatible with all existing features:
  - Filtering (-e trace=)
  - Statistics (-c)
  - Timing (-T)
  - Output formats (JSON, CSV)
  - Source correlation (--source)
- Process lifecycle messages: `[renacer: Process X forked child Y]`

**Results:**
- **Tests**: 243+ total (13 new integration tests)
- **Complexity**: All functions ≤10 (max: 9) ✅
- **Clippy**: Zero warnings ✅
- **Bug Fixes**: 1 critical hang bug fixed (child continuation)
- **TDG Score**: 94.5/100 maintained

**Examples:**
```bash
# Trace parent and child processes
renacer -f -- bash -c "echo parent && (echo child &)"

# Follow forks with filtering
renacer -f -e trace=file -- make clean

# Multi-process statistics
renacer -f -c -- python multi_process_app.py

# JSON output for process tree
renacer -f --format json -- ./fork_heavy_program > trace.json

# Track thread creation (clone syscall)
renacer -f -- ./multithreaded_app
```

**Quality Gates:**
- Toyota Way: Andon Cord pulled for complexity violation (17 → 9)
- EXTREME TDD: Full RED-GREEN-REFACTOR cycle completed
- Zero tolerance: All 243 tests passing, zero warnings

### Planned for 0.3.0
- Multi-threaded tracing optimizations
- eBPF backend option for reduced overhead
- See GitHub Issue #2 for detailed roadmap

---

[0.3.0]: https://github.com/paiml/renacer/releases/tag/v0.3.0
[0.2.0]: https://github.com/paiml/renacer/releases/tag/v0.2.0
[0.1.0]: https://github.com/paiml/renacer/releases/tag/v0.1.0

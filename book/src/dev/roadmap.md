# Roadmap

Trueno-DB follows a phased development approach, with each phase building on the previous foundation. Our development is guided by **EXTREME TDD** (Test-Driven Development) and **Toyota Way** principles to ensure built-in quality.

## Current Release: v0.2.1 (November 2025)

**Status**: Phase 1 (Core Engine) and Phase 2 (Multi-GPU) complete with production-grade quality (96.3/100 TDG score, A+ grade).

## Development Phases

### Phase 1: Core Engine ✅ **COMPLETE**

**Goal**: Establish foundational analytics engine with GPU/SIMD backends and proven performance benchmarks.

**Status**: Shipped in v0.1.0 and v0.2.0

#### Completed Features
- **Storage Engine**
  - Apache Arrow columnar format integration
  - Parquet file reader with schema validation
  - Morsel-driven iteration (128 MB chunks for out-of-core execution)
  - GPU transfer queue with bounded backpressure

- **Backend Dispatcher**
  - Cost-based GPU vs SIMD selection
  - Physics-based 5x rule (compute > 5x PCIe transfer)
  - Conservative performance estimates (32 GB/s PCIe, 100 GFLOP/s GPU)

- **GPU Kernels** (via wgpu)
  - MIN/MAX aggregations with parallel reduction
  - SUM aggregation with atomic operations
  - Kernel fusion via JIT WGSL compiler
  - Fused filter+sum eliminating intermediate buffers

- **SIMD Integration** (via trueno v0.6.0)
  - AVX-512/AVX2/SSE2 auto-detection
  - Graceful degradation to scalar fallback
  - 10-20x speedup vs baseline implementations

- **Top-K Selection**
  - O(N log K) heap-based algorithm
  - 28.75x speedup vs full sort (1M rows, K=10)
  - Support for Int32, Int64, Float32, Float64

- **SQL Query Parsing**
  - SELECT, WHERE, GROUP BY, ORDER BY, LIMIT
  - Integration with sqlparser crate

#### Quality Metrics (v0.2.1)
- **Tests**: 156/156 passing (100%)
- **Coverage**: 95.24% (exceeds 90% target)
- **TDG Score**: 96.3/100 (A+)
- **Critical Defects**: 0 (eliminated 25 unwraps, 100% production-safe)
- **Clippy**: 0 warnings in strict mode
- **Examples**: 5 comprehensive demos (gaming, finance, benchmarks)

#### Performance Validation
- **Competitive Benchmarks**: vs DuckDB, SQLite, Polars
- **PCIe Analysis**: Empirical 5x rule validation
- **GPU Syscall Tracing**: renacer verification of zero-copy operations
- **Property-Based Testing**: 11 tests, 1,100 scenarios

---

### Phase 2: Multi-GPU ✅ **COMPLETE**

**Goal**: Unlock multi-GPU parallelism with data partitioning and workload distribution across local GPUs.

**Status**: Shipped in v0.2.0

#### Completed Features
- **Multi-GPU Infrastructure**
  - Device enumeration and capability detection
  - Multi-device initialization with error handling
  - Device selection based on workload characteristics

- **Data Partitioning**
  - Range-based partitioning for sorted data
  - Hash-based partitioning for aggregations
  - Chunk-based partitioning for parallel scans

- **Query Execution**
  - Parallel query execution across multiple GPUs
  - Result aggregation with reduction operators
  - Load balancing based on device capabilities

- **Benchmarks**
  - 2 GPU vs 1 GPU scaling benchmarks
  - Multi-GPU aggregation performance validation
  - Near-linear scaling verification

#### Quality Gates Passed
- **Tests**: 156/156 passing
- **Backend Equivalence**: Multi-GPU == Single GPU == SIMD
- **Documentation**: Architecture diagrams, usage examples
- **Benchmarks**: Scaling validation across 1-4 GPUs

---

### Phase 3: Distribution 🔄 **NEXT UP**

**Goal**: Enable distributed query execution across networked GPU clusters with fault tolerance and horizontal scaling.

**Target**: v0.3.0 (Q1 2026)

#### Planned Features
- **gRPC Worker Protocol**
  - Worker discovery and heartbeat protocol
  - Query dispatch and result collection
  - Network topology-aware query planning

- **Distributed Query Execution**
  - Query fragmentation and distribution
  - Shuffle operations for distributed GROUP BY
  - Distributed JOIN algorithms (broadcast, shuffle)

- **Fault Tolerance**
  - Query retry logic with exponential backoff
  - Worker failure detection and failover
  - Checkpoint/restart for long-running queries

- **Resource Management**
  - Cluster-wide GPU memory tracking
  - Dynamic workload rebalancing
  - Priority-based query scheduling

#### Success Criteria
- 4+ node distributed query benchmarks
- Fault injection testing (worker failures, network partitions)
- Scalability tests up to 16 GPUs across 8 nodes
- Performance: 90% efficiency vs ideal linear scaling

---

### Phase 4: WASM 🔮 **FUTURE**

**Goal**: Deploy Trueno-DB analytics to browsers via WebAssembly and WebGPU for client-side analytics dashboards.

**Target**: v0.4.0 (Q2 2026)

#### Planned Features
- **WASM Build Target**
  - `wasm32-unknown-unknown` build configuration
  - wasm-bindgen integration for JavaScript interop
  - WASM-optimized binary size (<2 MB gzipped)

- **WebGPU Backend**
  - Browser GPU access via WebGPU API
  - Graceful fallback to SIMD128 (Wasm SIMD)
  - SharedArrayBuffer for zero-copy operations

- **Browser Integration**
  - JavaScript/TypeScript SDK
  - React/Vue component examples
  - Real-time dashboard demos

- **Client-Side Use Cases**
  - Interactive dashboards (no backend required)
  - Privacy-preserving analytics (data stays local)
  - Offline analytics applications

#### Success Criteria
- Browser compatibility: Chrome, Firefox, Safari, Edge
- Performance: <100ms query latency for 1M row dataset
- Bundle size: <2 MB total (WASM + JS glue)
- Example dashboards deployed to GitHub Pages

---

## Contributing to the Roadmap

We welcome community input on roadmap priorities! Here's how to get involved:

### Feature Requests
1. **Check existing issues** tagged `roadmap` or `enhancement`
2. **Open a discussion** describing your use case
3. **Propose implementation** with architecture sketch
4. **Align with quality standards** (EXTREME TDD, Toyota Way)

### Current Priorities
Based on user feedback and project goals, current priorities are:

1. **Phase 3 gRPC Protocol** (High Priority)
   - Foundation for distributed execution
   - Enables horizontal scaling

2. **Query Optimizer** (Medium Priority)
   - Cost-based plan selection
   - Predicate pushdown
   - Join reordering

3. **Window Functions** (Medium Priority)
   - ROW_NUMBER, RANK, LAG, LEAD
   - GPU-accelerated implementation

4. **Production Hardening** (Ongoing)
   - Additional error handling improvements
   - Performance profiling and optimization
   - Memory leak detection

### Quality Gates for All Phases
Every phase must pass these gates before release:

- ✅ **Tests**: 100% passing, >90% code coverage
- ✅ **TDG Score**: ≥85/100 (B+ minimum)
- ✅ **Benchmarks**: Performance claims validated
- ✅ **Documentation**: Complete API docs + examples
- ✅ **CI/CD**: All GitHub Actions workflows passing
- ✅ **Red Team Audit**: Adversarial verification complete

## Historical Milestones

| Date | Version | Milestone |
|------|---------|-----------|
| 2025-11-21 | v0.2.1 | Quality improvements (96.3/100 TDG, 0 critical defects) |
| 2025-11-21 | v0.2.0 | Phase 2 complete (Multi-GPU, JIT compiler, 95.24% coverage) |
| 2025-11-19 | v0.1.0 | Phase 1 MVP (Top-K, Storage, Backend dispatcher) |
| 2025-11-01 | - | Project inception |

## Toyota Way Applied

Our roadmap reflects Toyota Production System principles:

- **Jidoka** (Built-in Quality): EXTREME TDD at every phase
- **Kaizen** (Continuous Improvement): Incremental feature delivery
- **Genchi Genbutsu** (Go and See): Benchmarks validate all claims
- **Muda** (Waste Elimination): Feature-gating prevents bloat
- **Heijunka** (Level Loading): Balanced workload across phases

## Next Steps

Want to contribute to the roadmap? Start here:

1. **Review CLAUDE.md** - Understand project architecture
2. **Run quality gates** - `make quality-gate` to ensure environment setup
3. **Pick a Phase 3 task** - Check GitHub issues tagged `phase-3`
4. **Follow EXTREME TDD** - RED → GREEN → REFACTOR with mutation testing
5. **Submit PR** - With benchmarks, tests, and documentation

See [Contributing Guide](./contributing.md) for detailed guidelines.
